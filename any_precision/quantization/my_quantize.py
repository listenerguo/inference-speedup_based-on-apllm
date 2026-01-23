import os
import logging
from pickletools import uint8

import torch
import numpy as np
from tqdm import tqdm

import numba
from concurrent.futures import ThreadPoolExecutor
import flash1dkmeans
logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s] %(message)s', datefmt='%H:%M:%S')

# 改版新加
from torch import device
from .utils_gptq import move_to_device, get_device, nested_move_to_device
from .gptq import GPTQ
from .datautils import get_tokens
from .config import *
import transformers
import torch.nn as nn


# torch.backends.cuda.matmul.allow_tf32 = False
# torch.backends.cudnn.allow_tf32 = False


def root_and_downgrade(
        analyzer,
        output_folder,          # 输出保存的权重文件位置
        descendant_precision,   # 对应 seed_precision 精度 最低
        root_precision,         # 对应 parent_precision 精度 最高
        num_examples=DEFAULT_NUM_EXAMPLES,
        dataset=DEFAULT_DATASET,
        seq_len=DEFAULT_SEQ_LEN,
        cpu_count=None,
        random_state=None,
        group_size=-1,  # (128? 256?)
        gradients=None,
        # input_all_of_l0,
):
    """ 实现 由高比特逐级量化为低比特  -- based on GPTQ """
    assert descendant_precision <= root_precision, "descendant precision should be equal or lower than root precision"
    # 后代精度应等于或低于基准精度

    # # 执行并行流水线计算的线程参数的基本设置
    # if cpu_count is None:
    #     cpu_count = os.cpu_count()
    # # Determine IO and threading settings based on the number of cores
    # if cpu_count >= 8:
    #     pipelined_io = True     # 启用 “流水线式 IO”
    #     io_workers = 2 if cpu_count >= 64 else 1    # 设置专门的线程执行IO，确定IO的线程数worker
    #     numba.set_num_threads(cpu_count - io_workers)   # 计算任务的线程数 = 核心总数-IO-worker
    # else:
    #     pipelined_io = False    # 核心数小于8 ，不执行流水线，关闭并行
    #     io_workers = 0  # No separate IO workers needed for non-pipelined IO
    #     numba.set_num_threads(cpu_count)


    # ###########################  获取 layer[0] 输入 ，即 Decoder_0 中的 注意力机制的输入 #######################
    layer_inputs = []  # 各层的输入
    attention_masks = []  # 注意力掩码
    position_ids = []  # 位置 ID
    layer_input_kwargs = []  # 模型layer kwargs 输入

    model = analyzer.model  # 预训练模型
    # model = model.bfloat16()    # new add  -- save as fp16

    tokenizer = analyzer.tokenizer  # 通过 analyzer.utils.load_tokenizer
    examples = get_tokens(dataset, 'train', tokenizer, seq_len, num_examples, seed=random_state)    # 将数据集中的训练集作为校准数据，并转为token

    # forward_pass_use_cache = model.config.use_cache
    model.config.use_cache = False  # 临时禁用模型缓存  # 可节省内存

    layers = analyzer.get_layers()      # print('模型结构', model, "\nlayer结构", layers)

    cur_layer_device = get_device(layers[0])  # 模型通常位于GPU中
    cache_examples_on_gpu = True
    data_device = cur_layer_device if cache_examples_on_gpu else device("cpu")  # # data_device: 校准数据的缓存设备（GPU 缓存可加速后续计算，CPU 缓存节省 GPU 内存）

    force_layer_back_to_cpu = False  # 记录是否将layers[0]强制转移到GPU
    if get_device(layers[0]) == device("cpu"):
        layers[0] = layers[0].to(device("cuda:0"))
        force_layer_back_to_cpu = True

    # 单独对最初的输入 进行捕获，设计store_input_hook函数，通过register_forward_pre_hook执行
    def store_input_hook(_, args, kwargs):
        layer_input = []
        for inp in args:
            layer_input.append(move_to_device(inp, data_device))
        layer_inputs.append(layer_input)

        # Keyword arguments.
        if kwargs["attention_mask"] is not None:
            attention_masks.append(kwargs["attention_mask"].to(data_device))
        else:
            attention_masks.append(None)
        pos_ids = kwargs.get("position_ids", None)
        if pos_ids is not None:
            position_ids.append(move_to_device(pos_ids, data_device))

        one_kwargs = {}
        for (k, v,) in kwargs.items():  # make sure other arguments also be captured
            if k not in ["hidden_states", "attention_mask", "position_ids"]:
                one_kwargs[k] = nested_move_to_device(v, data_device)  # 递归将数据转移到data_device上
        layer_input_kwargs.append(one_kwargs)  # 保留的数据
        raise ValueError

    handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
    # # 将layers[0]的 args 与 kwargs 转移到data_device上， 并分别进行保存，存至四个数据中
    for example in examples:
        for k, v in example.items():  # example {input_ids:tensor([]) , attention_mask:tensor([])}
            # print(k,'\n v-shape', v.shape,'\n  len(v-shape)', len(v.shape) )
            if len(v.shape) == 1:  # v.shape -- batch*seqlen;
                v = v.unsqueeze(0)
            example[k] = move_to_device(v, cur_layer_device)
        try:
            model(**example)  # 将输入字典，按照关键字执行 函数，该执行过程会启动hook函数
        except ValueError:
            pass
    handle.remove()  # 移除hook函数

    # print('device to save：', CPU if force_layer_back_to_cpu else cur_layer_device)
    move_to_device(layers[0], device("cpu") if force_layer_back_to_cpu else cur_layer_device)
    torch.cuda.empty_cache()  # 清空 CUDA 缓存，释放临时占用的内存

    # print(layer_inputs)
    # print(attention_masks)
    # print(position_ids)
    # print(layer_input_kwargs)
    # return 0

    # logging.info(f"Using {cpu_count} cores for parallelization")
    logging.info(f"Root & downgrade from {root_precision}-bit to {descendant_precision}-bit")

    quantizers = {}     # 保存量化结果？
    logging.info(f"Quantizing Layer-by-layer")

    root_bit = root_precision
    des_bit = descendant_precision
    layers_to_process = list(range(analyzer.num_layers))    # 列表，元素为 索引
    # 不执行pipeline ，依次对LLM的每一层执行 descendant-bit to root-bit 的量化
    for l in tqdm(layers_to_process, desc="Quantizing layers..."):
        # 得到当前layer
        layer = layers[l]
        # print('test-3: ', analyzer.get_modules(layer)['fc2'].weight)    # 获取 模型 某一layer中具体module的权重
        # print('test-4: ', analyzer.get_layer_weights(l)['fc2'])         # 获取 模型 某一layer中具体module的权重
        # # 执行量化
        # quant_infos, layer_outputs, force_layer_back_to_cpu, cur_layer_device = _root_and_downgrade_layer(
        #     analyzer,
        #     layer,
        #     l,
        #     layer_inputs, attention_masks, position_ids, layer_input_kwargs,
        #     root_precision, descendant_precision,
        #     num_examples,group_size
        # )
        if get_device(layer) == device("cpu"):
            move_to_device(layer, device("cuda:0"))
        force_layer_back_to_cpu = True
        cache_examples_on_gpu = True
        cur_layer_device = get_device(layer)  # 记录layer当前位置

        # 数据预处理
        layer_outputs = []  # 各层的输出
        # quantizers = {}
        layer_grad = gradients[l]
        error1 = [0] * (root_bit - des_bit)

        # 得到当前layer下的所有模型模块 - modules为字典- k:模块名; v:模块
        modules = analyzer.get_modules(layer)
        inside_layer_modules = analyzer.module_insides  # [[q,k,v],[out],[fc1],[fc2]]
        # print(inside_layer_modules)
        model_name = analyzer.model_name

        for module_names in inside_layer_modules:  # 对当前layer中的所有模块分为多个子集进行量化补偿(true_sequential=True)
            module_subset = {n: modules[n] for n in module_names if n in modules}  # 得到模块子集- k:模块名; v:模块

            ###  --------- 1.执行 gptq 量化： fp -> des_bit ---------  ### root_bit

            # if layer_index == 0 and "fc1" in module_subset:
            #     o1w = module_subset["fc1"].weight.data.clone().cpu()
            #     print("原来权重：", o1w)

            gptq = {}
            # name:{'_self_attn.k_proj', '_self_attn.q_proj', '_self_attn.v_proj', '_self_attn.out_proj', 'fc1', 'fc2'}
            for name in module_subset:  # 遍历字典module_subset 的键
                gptq[name] = GPTQ(module_subset[name], f"{l}_{name}", f"{name}")
                gptq[name].quantizer.configure(
                    des_bit,  # 确定 量化 精度 root_bit
                    perchannel=True,  # 逐channel量化，对应权重矩阵的列//每列为一个量化单位
                    sym=False,  # 原来的默认参数：True  # 是否对称量化 // 默认为True 来自 auto_gptq/quantization/config.py
                    mse=False,  # False,
                )


            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)  # noqa: F821
                return tmp

            handles = []
            for name in module_subset:  # 遍历键// 模型名称
                handles.append(module_subset[name].register_forward_hook(add_batch(name)))

            for i in range(num_examples):  # 遍历每个样本
                layer_input = []
                for k, layer_inp in enumerate(layer_inputs[i]):  # layer_inputs 对应的输入
                    layer_input.append(move_to_device(layer_inp, cur_layer_device))

                layer_attention_mask = move_to_device(attention_masks[i], cur_layer_device)
                additional_layer_inputs = {"attention_mask": layer_attention_mask}
                layer_position_ids = (
                    None if not position_ids else move_to_device(position_ids[i], cur_layer_device)
                )
                if layer_position_ids is not None:
                    additional_layer_inputs["position_ids"] = layer_position_ids
                for k, v in layer_input_kwargs[i].items():
                    additional_layer_inputs[k] = nested_move_to_device(v, cur_layer_device)  # 将其他输入参数都保存在该字典中
                layer(*layer_input, **additional_layer_inputs)  # 执行对应的层， 同时执行hook函数 即add_batch，更新H
            for h in handles:
                h.remove()



            for name in module_subset:  # 遍历每个模块
                # gptq[name].plot_hessian()   #data=module_grad
                # print('!@#@!', type(gradients),type(layer_grad))
                module_grad = layer_grad[name]
                module_grad = move_to_device(module_grad, cur_layer_device)
                print(f'!!!!!!! 读取与计算的梯度：{module_grad.shape}')
                logging.info(f"Quantizing {name} in layer...")
                # # 绘制海森矩阵热力图 以及 激活值通道分布
                # 执行量化参数 s,c,g 计算
                scale, zero, g_idx, qw = gptq[name].fasterquant(percdamp=0.01,  # self.quantize_config.damp_percent,
                                                                hessian_f=module_grad,
                                                                group_size=group_size,  # -1,
                                                                actorder=False,  # 如果设置为 Ture 则考虑激活值影响？
                                                                static_groups=False,
                                                                root_bit=root_bit,
                                                                des_bit=des_bit,
                                                                )  # 执行量化操作
                del module_grad
                quantizers[f"{model_name}.{analyzer.layers_name}.{l}.{name}"] = (
                    # gptq[name].quantizer.to(device("cpu") if force_layer_back_to_cpu else cur_layer_device),    # 保存量化设置    # 可以不保存？
                    move_to_device(scale, device("cpu") if force_layer_back_to_cpu else cur_layer_device),
                    move_to_device(zero, device("cpu") if force_layer_back_to_cpu else cur_layer_device),
                    move_to_device(g_idx, device("cpu") if force_layer_back_to_cpu else cur_layer_device),
                    move_to_device(qw, device("cpu") if force_layer_back_to_cpu else cur_layer_device),
                )  # 整理量化参数
                gptq[name].free()
                

                # # # print('*/*/*/*/*/*/*',type(err))
                # for i in range(len(error1)):
                #     error1[i] += e_c[i]
                #     # error2[i] += e_noc[i]
        # return 0,0

        # 根据输入计算输出，然后将 输出 更新为下一个layer的 输入
        for j in range(num_examples):  # 遍历每个样本
            layer_input = []
            for k, layer_inp in enumerate(layer_inputs[j]):  # layer_inputs 模型的输入
                layer_input.append(move_to_device(layer_inp, cur_layer_device))

            layer_attention_mask = move_to_device(attention_masks[j], cur_layer_device)
            additional_layer_inputs = {"attention_mask": layer_attention_mask}
            layer_position_ids = None if not position_ids else move_to_device(position_ids[j], cur_layer_device)
            if layer_position_ids is not None:
                additional_layer_inputs["position_ids"] = layer_position_ids
            for k, v in layer_input_kwargs[j].items():
                additional_layer_inputs[k] = nested_move_to_device(v, cur_layer_device)
            layer_output = move_to_device(
                # 用Q更新W后的模型的layer，会根据当前输入，计算(更新后)的模型输出，因此，后续的模型量化校准结果会更加准确。
                layer(*layer_input, **additional_layer_inputs)[0],  # layer() 执行前向传播，计算layer的输出结果 [0]是为了只提取hidden states
                cur_layer_device if cache_examples_on_gpu else device("cpu"),
            )
            layer_outputs.append([layer_output])  # 保存当前layer最终 每个样本的输出，用于作为输入再次传入本函数，
        # return quantizers, layer_outputs, force_layer_back_to_cpu, cur_layer_device
        # quantizers = quant_infos

        layers[l] = move_to_device(layer, device("cpu") if force_layer_back_to_cpu else cur_layer_device)  # 适配可能的分层设备部署场景（如模型层分布在不同 GPU 上）。
        del layer
        del gptq
        del layer_inputs
        layer_inputs, layer_outputs = layer_outputs, []  # TODO: is it really OK to cache only the first positional argument?
        torch.cuda.empty_cache()


        # quantizers.update(quant_infos)

    # # 计算模型 量化后的整数权重
    # module_names = analyzer.module_names
    # layers = analyzer.get_layers()
    # # maxq = torch.tensor(2 ** root_precision - 1)
    # weight_int = {}
    # for l, layer in enumerate(layers):
    #     modules = analyzer.get_modules(layer)
    #     module_subset = {name: modules[name] for name in module_names if name in modules}
    #     for name, module in module_subset.items():  # self_attn.k_proj  对应层的模块
    #         full_name = f'model.decoder.layers.{l}.{name}'
    #         weights_fp = module.weight.data.clone()
    #         if isinstance(module, nn.Conv2d):
    #             weights_fp = weights_fp.flatten(1)
    #         if isinstance(module, transformers.Conv1D):
    #             weights_fp = weights_fp.t()
    #         weights_fp = weights_fp.float().cpu()
    #         s, z, g = quantizers[full_name][0], quantizers[full_name][1], quantizers[full_name][2]
    #         # print('current module:', full_name, '\t shape: ', weights_fp.shape, s.shape, z.shape)
    #
    #         out_ft, in_ft = weights_fp.shape
    #
    #         # # case 1
    #         s_z = z * s
    #         intweight = []
    #         for i in range(in_ft):
    #             g_index = g[i]
    #             sz_i = s_z[:, g_index]
    #             si = s[:, g_index]
    #             w_col = weights_fp[:, i]
    #             int_w = torch.round((w_col + sz_i) / si).to(torch.int)
    #             intweight.append(int_w[:, None])
    #
    #         intweight = torch.cat(intweight, dim=1).contiguous()
    #         weight_int[full_name] = intweight

            # # case 2
            # s = s.t().contiguous()
            # z = z.t().contiguous()
            # s_z = z * s
            # st = s.clone()
            # intweight = []
            # for i in range(in_ft):
            #     g_index = i // group_size
            #     sz_i = s_z[g_index]
            #     si = st[g_index]
            #     w_col = weights_fp[:, i]
            #     int_w = torch.round((w_col + sz_i) / si).to(torch.int)
            #     # int_w = torch.clamp(torch.round((w_col + sz_i) / si), 0, maxq).to(torch.uint8)
            #     intweight.append(int_w[:, None])

            # intweight = torch.cat(intweight, dim=1).contiguous()
            # weight_int[full_name] = intweight
    # print('O3W:' , weight_int["model.decoder.layers.0.fc1"], weight_int["model.decoder.layers.0.fc1"].shape)
    # print('优化',error_tongji)
    # for k, v in quantizers.items():
    #     print(f"{k}-scale :\n{v[1]}")
    #     print(f"{k}-zeros :\n{v[2]}")

    return quantizers   #, weight_int


# def _root_and_downgrade_layer(
#         analyzer,
#         layer,
#         layer_index,
#         layer_inputs,
#         attention_masks,
#         position_ids,
#         layer_input_kwargs,
#         root_bit, des_bit,
#         num_examples,group_size
# ):
#     """
#     """
#     # 将当前layer的位置转移到GPU
#     # force_layer_back_to_cpu = False
#     if get_device(layer) == device("cpu"):
#         move_to_device(layer, device("cuda:0"))
#     force_layer_back_to_cpu = True
#     # print('force_layer_back_to_cpu: ',force_layer_back_to_cpu)
#     cache_examples_on_gpu = True
#     cur_layer_device = get_device(layer)  # 记录layer当前位置
#
#     # 数据预处理
#     layer_outputs = []  # 各层的输出
#     quantizers = {}
#     error1 = [0] * (root_bit - des_bit)
#     error2 = [0] * (root_bit - des_bit)
#
#     # 得到当前layer下的所有模型模块 - modules为字典- k:模块名; v:模块
#     modules = analyzer.get_modules(layer)
#     # if not true_sequential:  # 如果true_sequential设置为False，则将所有组模块合并为一组
#     #     inside_layer_modules = [sum(analyzer.module_insides, [])]
#     inside_layer_modules =  analyzer.module_insides     # [[q,k,v],[out],[fc1],[fc2]]
#     # print(inside_layer_modules)
#     model_name = analyzer.model_name
#
#     for module_names in inside_layer_modules:   # 对当前layer中的所有模块分为多个子集进行量化补偿(true_sequential=True)
#         module_subset = {n:modules[n] for n in module_names if n in modules}    # 得到模块子集- k:模块名; v:模块
#         ###  --------- 1.执行 gptq 量化： fp -> root_bit ---------  ###
#
#         # if layer_index == 0 and "fc1" in module_subset:
#         #     o1w = module_subset["fc1"].weight.data.clone().cpu()
#         #     print("原来权重：", o1w)
#
#         gptq = {}
#         for name in module_subset:  # 遍历字典module_subset 的键
#             gptq[name] = GPTQ(module_subset[name], f"{layer_index}_{name}")
#             gptq[name].quantizer.configure(
#                 root_bit,           # 确定 量化 精度
#                 perchannel=True,    # 逐channel量化，对应权重矩阵的列//每列为一个量化单位
#                 sym=False,           #原来的默认参数：True  # 是否对称量化 // 默认为True 来自 auto_gptq/quantization/config.py
#                 mse=False,          # False,
#             )
#
#         def add_batch(name):
#             def tmp(_, inp, out):
#                 gptq[name].add_batch(inp[0].data, out.data)  # noqa: F821
#             return tmp
#
#         handles = []
#         for name in module_subset:  # 遍历键// 模型名称
#             handles.append(module_subset[name].register_forward_hook(add_batch(name)))
#
#         for i in range(num_examples):  # 遍历每个样本
#             layer_input = []
#             for k, layer_inp in enumerate(layer_inputs[i]):  # layer_inputs 对应的输入
#                 layer_input.append(move_to_device(layer_inp, cur_layer_device))
#
#             layer_attention_mask = move_to_device(attention_masks[i], cur_layer_device)
#             additional_layer_inputs = {"attention_mask": layer_attention_mask}
#             layer_position_ids = (
#                 None if not position_ids else move_to_device(position_ids[i], cur_layer_device)
#             )
#             if layer_position_ids is not None:
#                 additional_layer_inputs["position_ids"] = layer_position_ids
#             for k, v in layer_input_kwargs[i].items():
#                 additional_layer_inputs[k] = nested_move_to_device(v, cur_layer_device)  # 将其他输入参数都保存在该字典中
#             layer(*layer_input, **additional_layer_inputs)  # 执行对应的层， 同时执行hook函数 即add_batch，更新H
#         for h in handles:
#             h.remove()
#
#         # if 'fc2' in module_subset:
#         #     print(gptq["fc2"].act_channel.shape, ' -vs- ',gptq["fc2"].layer.weight.data.shape)
#
#         for name in module_subset:  # 遍历每个模块
#             logging.info(f"Quantizing {name} in layer...")
#             scale, zero, g_idx, hes= gptq[name].fasterquant(    # , err
#                 percdamp=0.01,  #self.quantize_config.damp_percent,
#                 group_size=group_size,#-1,
#                 actorder=False,     # 如果设置为 Ture 则考虑激活值影响？
#                 static_groups=False,
#                 root_bit=root_bit,
#                 des_bit=des_bit,
#             )  # 执行量化操作
#             # print(f"scale-{model_name}.{analyzer.layers_name}.{layer_index}.{name}: \n{scale}")
#             comp, e_c, e_noc = gptq[name].quanterrorcomp(hessian = hes,
#                                       group_size=group_size,
#                                       actorder=False,
#                                       static_groups=False,
#                                       root_bit=root_bit,
#                                       des_bit=des_bit,
#                                       )
#             # print(f"{scale.shape}--{q_int.shape}--{comp.shape}")  #--{comp[0].shape}--{comp[1].shape}
#             quantizers[f"{model_name}.{analyzer.layers_name}.{layer_index}.{name}"] = (
#                 # gptq[name].quantizer.to(device("cpu") if force_layer_back_to_cpu else cur_layer_device),    # 保存量化设置    # 可以不保存？
#                 move_to_device(scale, device("cpu") if force_layer_back_to_cpu else cur_layer_device),
#                 move_to_device(zero, device("cpu") if force_layer_back_to_cpu else cur_layer_device),
#                 move_to_device(g_idx, device("cpu") if force_layer_back_to_cpu else cur_layer_device),
#                 move_to_device(comp, device("cpu") if force_layer_back_to_cpu else cur_layer_device),
#                 # move_to_device(q_int, device("cpu") if force_layer_back_to_cpu else cur_layer_device),
#             )  # 整理量化参数
#             gptq[name].free()
#
#             # # print('*/*/*/*/*/*/*',type(err))
#             for i in range(len(error1)):
#                 error1[i] += e_c[i]
#                 error2[i] += e_noc[i]
#
#
#     # 根据输入计算输出，然后将 输出 更新为下一个layer的 输入
#     for j in range(num_examples):  # 遍历每个样本
#         layer_input = []
#         for k, layer_inp in enumerate(layer_inputs[j]):  # layer_inputs 模型的输入
#             layer_input.append(move_to_device(layer_inp, cur_layer_device))
#
#         layer_attention_mask = move_to_device(attention_masks[j], cur_layer_device)
#         additional_layer_inputs = {"attention_mask": layer_attention_mask}
#         layer_position_ids = None if not position_ids else move_to_device(position_ids[j], cur_layer_device)
#         if layer_position_ids is not None:
#             additional_layer_inputs["position_ids"] = layer_position_ids
#         for k, v in layer_input_kwargs[j].items():
#             additional_layer_inputs[k] = nested_move_to_device(v, cur_layer_device)
#         layer_output = move_to_device(
#             # 用Q更新W后的模型的layer，会根据当前输入，计算(更新后)的模型输出，因此，后续的模型量化校准结果会更加准确。
#             layer(*layer_input, **additional_layer_inputs)[0],  # layer() 执行前向传播，计算layer的输出结果 [0]是为了只提取hidden states
#             cur_layer_device if cache_examples_on_gpu else device("cpu"),
#         )
#         layer_outputs.append([layer_output])    # 保存当前layer最终 每个样本的输出，用于作为输入再次传入本函数，
#     # print('layer-output-',layer_outputs)
#
#     # 根据激活值输入状态 - 得到 激活量化因子； 并根据海森矩阵，进行排序得到，权重量化因子，
#     # 截取高比特量化结果作为当前量化后的权重 ，采用 MES (参考 Quantizer 中的 find_params) 计算最佳补偿参数，使得 加权后的 系统/权重均方误差 最小。
#     # 逐级计算每个子比特的 权重和 补偿参数
#     # print('ERROR-SUM:',error_tongji)
#
#     del gptq
#     del layer_inputs
#     # layer_inputs, layer_outputs = layer_outputs, []  # TODO: is it really OK to cache only the first positional argument?
#     torch.cuda.empty_cache()
#     # print('补偿后：',error1)
#     # print('补偿前：',error2)
#     return quantizers, layer_outputs, force_layer_back_to_cpu, cur_layer_device

# root_and_downgrade(root_bit, desc_bit)
# 主函数：量化范围[desc_bit(例如4bit), root_bit(例如8bit)],但量化顺序为 root_bit->desc_bit
# ├── 1. 参数与环境准备
# │   ├── 校验输入参数（root_bit >= desc_bit，避免逻辑错误）
# │   ├── 初始化缓存路径与并行执行器（复用原代码CPU多线程能力）
# │   └── 加载模型元信息（通过analyzer获取层结构、模块名 以及 梯度、海森矩阵等信息）
# │
# ├── 2. 核心依赖组件生成
# │   ├── _get_layer_loader()  # 工厂函数：生成层数据加载器
# │   │   └── 加载当前层浮点权重+梯度（用于GPTQ/AWQ量化时的感知优化）
# │   └── _get_saver()         # 工厂函数：生成结果保存器
# │       └── 定义按精度分目录的存储格式（兼容原cache结构，便于后续pack步骤）
# │
# ├── 3. 逐层量化处理（循环遍历模型所有层）
# │   └── _root_and_downgrade_layer()  # 单一层的量化主逻辑
# │       ├── 3.1 高精度根模型量化  # “fp -> root_bit”
# │       │   └── _root_quantization()  # 基于GPTQ/AWQ的初始量化
# │       │       ├── 选择量化方法（通过参数控制GPTQ/AWQ分支）
# │       │       ├── 执行量化（可以尝试复用group_count参数，与原代码对齐）
# │       │       └── 输出：root_bit精度的量化参数（qweight, scale_r, zero_point）
# │       │
# │       └── 3.2 从高精度降级至低精度 # “root_bit 逐级-> desc_bit”
# │           └── _downgrade_group()  # 可以尝试按组处理降级（与原group逻辑兼容）
# │               ├── 遍历 目标精度范围（从root_bit到最小目标精度）
# │               └── _decrement_group()  # 单次精度降级（如8bit→7bit）
# │                   ├── _truncate_high_bits()  # 截取高bit位作为低精度权重
# │                   │   └── 右移操作实现比特截断（如6bit→5bit：右移1位）
# │                   ├── _adjust_scale_zp()  # 重新计算低精度的scale和zero_point
# │                   │   └── scale = scale_r × 2^比特差（因量化范围缩小）
# │                   └── _compute_compensation()  # 计算补偿常数修正截断误差
# │                       ├── 方案1: 通过 对比 当前bit 与 root_bit的 量化误差，并根据激活值和权重灵敏度进行加权，计算结果最小值补偿参数
# │                       └── 方案2：通过 对比 当前bit 与 当前bit+1 的 量化误差，并根据激活值和权重灵敏度进行加权，计算结果最小值补偿参数
# │
# └── 4. 结果保存与清理
#     └── _save_results()  # 保存所有精度的量化参数
#         └──  存储内容：每个精度的qweight、scale、zero_point、compensation

# ，量化差为当前bit相比于root_bit