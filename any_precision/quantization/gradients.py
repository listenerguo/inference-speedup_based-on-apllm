import torch
from numba.core.types import float16
from tqdm import tqdm
import os
import logging
from .config import *
from any_precision.analyzer.analyzer import get_analyzer
from .datautils import get_tokens
from .utils_gptq import move_to_device, get_device, nested_move_to_device


def get_gradients(
        analyzer,
        dataset=DEFAULT_DATASET,
        seq_len=DEFAULT_SEQ_LEN,
        num_examples=DEFAULT_NUM_EXAMPLES,
        save_path=None,
        random_state=None,
):
    if save_path is not None and os.path.isfile(save_path):
        logging.info(f"Gradients already calculated and saved at {save_path}.")
        logging.info(f"Loading gradients...")
        logging.info(f"Layer[0] input is None! ")
        gradients = torch.load(save_path, weights_only=False)
        return gradients
    logging.info(f"Calculating gradients on dataset {dataset} with sequence length {seq_len} and "f"{num_examples} examples...")

    model = analyzer.model      # 预训练模型-such as： FP16
    tokenizer = analyzer.tokenizer  # 通过 analyzer.utils.load_tokenizer

    # 加载数据集  根据 dataset 获取数据的split-训练集， 并随即采样num_examples个，通过tokenizer转为token，并限制转换后的长度为seq_len
    input_tokens = get_tokens(dataset, 'train', tokenizer, seq_len, num_examples, seed=random_state)
    # print('数据集选择：',input_tokens)

    if analyzer is None:
        analyzer = get_analyzer(model)

    model = model.bfloat16()    # 精度 转化为bf16
    model.eval()    # 将模型设置为评估模式

    layers = analyzer.get_layers()  # 通过分析器获取模型的 layer层， 为decoder中的 一部分，非layer包括 embedding层等
    # print('模型结构', model, "\nlayer结构", layers)

    # 定义钩子函数：接收梯度，返回其平方 //这个函数将在反向传播时被PyTorch自动调用
    def square_grad_hook(grad):
        return grad.pow(2)
    hooks = []
    for layer in layers:    # 遍历模型的每一层
        for module in analyzer.get_modules(layer).values(): # 遍历 模型layer下的每个模块
            hooks.append(module.weight.register_hook(square_grad_hook))
            # register_hook 为 module.weight (为每个张量或 nn.Module), 注册 square_grad_hook (hook函数)
            # 每个钩子句柄对应一个模块的权重参数

    # Calculate gradients through loss.backward()   # change input_tokens
    for tokens in tqdm(input_tokens, desc="Calculating gradients"): # tqdm 进度条
        for k,v in tokens.items():
            if len(v.shape) == 1:
                v = v.unsqueeze(0)
            tokens[k] = v.to(model.device)    # 把token移到模型所在设备（如GPU）
        outputs = model(**tokens, labels=tokens['input_ids'])
        # outputs = model(input_ids=tokens, labels=tokens)    # 模型预测（输入=标签，类似“自监督学习”）
        loss = outputs.loss # 计算损失（预测结果与标签的差距）
        loss.backward()     # 反向传播：计算各参数的梯度（触发hook函数-square_grad_hook）
    for hook in hooks:
        hook.remove()
    model.cpu()

    # Harvest the gradients  收集各层的梯度
    gradients = []
    for layer in layers:
        gradients_per_layer = {}
        for module_name, module in analyzer.get_modules(layer).items():
            gradients_per_layer[module_name] = module.weight.grad.cpu().float()    # 由于hook函数 这里实际上为梯度的平方
            # print('梯度形状：',gradients_per_layer[module_name].shape) # [out_features, in_features]
        gradients.append(gradients_per_layer)

    # print("//* 梯度计算结果 *//")
    # for dict in gradients:
    #     for k, v in dict.items():
    #         print(k,'\t',v.shape)


    #     for module_name, module in analyzer.get_modules(analyzer.get_layers()[0]).items():
    #         print(module_name, module, type(module))
    #     # module_name   self_attn.q_proj
    #     # module        Linear(in_features=2048, out_features=2048, bias=True)
    #     # type(module)  <class 'torch.nn.modules.linear.Linear'>


    # # # TODO 计算模型初始输入，便于逐层进行海森矩阵计算
    # layer_inputs = []  # 各层的输入
    # attention_masks = []  # 注意力掩码
    # position_ids = []  # 位置 ID
    # layer_input_kwargs = []  # 模型layer kwargs 输入
    #
    #
    # # 获取第0层 的layer  # layers = analyzer.get_layers()
    # cur_layer_device = get_device(layers[0])  # 模型通常位于GPU中
    # def store_input_hook(_, args, kwargs):
    #     # Positional arguments.
    #     layer_input = []
    #     for inp in args:
    #         layer_input.append(move_to_device(inp, cur_layer_device))
    #     layer_inputs.append(layer_input)
    #
    #     # Keyword arguments.
    #     if kwargs["attention_mask"] is not None:
    #         attention_masks.append(kwargs["attention_mask"].to(cur_layer_device))
    #     else:
    #         attention_masks.append(None)
    #     pos_ids = kwargs.get("position_ids", None)
    #     if pos_ids is not None:
    #         position_ids.append(move_to_device(pos_ids, cur_layer_device))
    #
    #     one_kwargs = {}
    #     for (k, v,) in kwargs.items():  # make sure other arguments also be captured
    #         if k not in ["hidden_states", "attention_mask", "position_ids"]:
    #             one_kwargs[k] = nested_move_to_device(v, cur_layer_device)  # 递归将数据转移到data_device上
    #     layer_input_kwargs.append(one_kwargs)  # 保留的数据
    #     raise ValueError
    #
    # # 注册 hook 函数，用于保存第0层由原始数据输入，进入 注意力机制时的输入// 激活值
    # handle = layers[0].register_forward_pre_hook(store_input_hook, with_kwargs=True)
    # # # 将layers[0]的 args 与 kwargs 转移到data_device上， 并分别进行保存，存至四个数据中
    # for example in input_tokens:
    #     for k, v in example.items():  # example {input_ids:tensor([]) , attention_mask:tensor([])}
    #         # print(k, '\n v-shape', v.shape, '\n  len(v-shape)', len(v.shape))
    #         if len(v.shape) == 1:  # v.shape -- batch*seqlen;
    #             v = v.unsqueeze(0)
    #         example[k] = move_to_device(v, cur_layer_device)
    #     try:
    #         model(**example)  # 将输入字典，按照关键字执行 函数，该执行过程会启动hook函数
    #     except ValueError:
    #         pass
    # handle.remove()  # 移除hook函数


    # # # Move model back to cpu  模型移回CPU（释放GPU内存）
    # model.cpu()
    # torch.cuda.empty_cache()

    # Save the gradients to file    # 将 梯度结果 保存至chche文件下
    # Note that when saving, the gradients are stored as bf16,
    # but are converted to np.float32 before returning, for the next steps in the pipeline
    if save_path is not None:
        logging.info(f"Saving gradients to {save_path}...")
        # add file extension if not present
        if not save_path.endswith('.pt'):
            save_path = save_path + '.pt'
        # check if the file already exists
        if os.path.exists(save_path):
            input(f"[WARNING] File {save_path} already exists. Press enter to overwrite or Ctrl+C to cancel.")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(gradients, save_path)
    return gradients
