# 设计 自己的量化算法 主程序

# 计算梯度与异常值分布分析 -- 生成父模型 -- 由父模型生成子模型 - 增加补偿参数 -- 模型权重pack

import os
import os.path
import shutil
import logging

from .config import *
from ..analyzer import get_analyzer
# from .gradients import get_gradients
# from .quantize import seed_and_upscale
# from .my_quantize import root_and_downgrade
from .pack import pack
from .dense_and_sparse import remove_outliers
import torch

import time

# Disable parallelism in tokenizers to prevent warnings when forking in the seed generation step
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logging with time sans date, level name, and message
logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s] %(message)s', datefmt='%H:%M:%S')


def any_precision_quantize_my(
        model,  # 输入模型路径 / 或名称
        seed_precision=DEFAULT_SEED_PRECISION,  # 设置最小量化精度
        parent_precision=DEFAULT_PARENT_PRECISION,  # 设置最大量化精度
        mode='pack',    # ？ 将模型打包
        yaml_path=None, cache_dir=DEFAULT_CACHE_DIR,    # cache_dir : 模型打包保存位置
        dataset=DEFAULT_DATASET,        # dataset 数据集名称(实现数据集路径为输入)  # 函数 get_gradients -> get_tokens -> _get_dataset
        seq_len=DEFAULT_SEQ_LEN, num_examples=DEFAULT_NUM_EXAMPLES,   # 数据集单个样本数据长度seq_len // 样本数量num_examples
        cpu_count=os.cpu_count(),       # cpu数量？ 用途？
        overwrite_gradients=False,
        overwrite_quantize=False,
        overwrite_pack=False,
        random_state=None,
        group_size=-1,      # group 大小
        dns=False,          # Dense & Sparse : 找出异常权重，得到稀疏化权重并保存
        sensitivity_outlier_percent=0.05,   # squeezeLLM 敏感度
        threshold_outlier_percent=0.40,     # squeezeLLM 异常值
        cpu_only=False,
):
    print(" 执行 所设计的多精度量化算法 -- working.... ")
    assert mode in ['gradients', 'quantize', 'pack'], \
        "mode must be one of 'gradients', 'quantize', or 'pack'. Use 'pack' to run the entire pipeline."
    # 如果 mode 不在 列表 里，直接报错，提示后面的句子，并退出执行

    # 设置逻辑，如果 梯度，则一定量化，打包； 如果 量化，则一定打包； 这是一个链式反应，纠正参数输入
    if overwrite_gradients:
        if not overwrite_quantize:
            logging.warning("Parent model needs to be recalculated if gradients are recalculated. "
                            "Setting overwrite_quantize to True.")
            overwrite_quantize = True

    if overwrite_quantize:
        if not overwrite_pack:
            logging.warning("Packed model needs to be recalculated if parent model is recalculated. "
                            "Setting overwrite_pack to True.")
            overwrite_pack = True

    # 展示当前mode 对应的执行步骤
    if mode == 'gradients':
        logging.info("Running: [Gradients]")
    elif mode == 'quantize':
        logging.info("Running: [Gradients -> Quantize]")
    else:   # 如果mode 属于 pack，则执行 梯度，量化，打包的过程
        logging.info("Running: [Gradients -> Quantize -> Pack]")

    # 得到模型名称， 如果为字符串，则model_string = model； 如果为模型类，则model_string = model.name_or_path；并根据是否为路径，截取关键名称
    model_string = model if isinstance(model, str) else model.name_or_path
    model_name = model_string.split("/")[-1]
    dataset_name = dataset.split("/")[-1]

    logging.info(f"Running Any-Precision Quantization on {model_name} with seed precision {seed_precision} and "
                 f"parent precision {parent_precision} using {dataset_name} for gradient calculation")


    # ------------------- Load model -------------------
    # 通过analyzer 返回一个 ModelAnalyzer 对象，它能提取模型结构信息（层、权重、模块名），方便后续量化      # **具体组成-未分析**
    analyzer = get_analyzer(
        model,
        yaml_path=yaml_path,
        include_tokenizer=True,
        cpu_only=cpu_only
    )
    logging.info("model load and analysis success! ")
    # print(type(analyzer.tokenizer))

    print("model.device: ",analyzer.model.device)

    # ------------------- Set cache paths -------------------
    dataset_short_name = os.path.basename(dataset.rstrip('/'))  # 如果输入为路径，确保此处仅包括数据集名称

    gradients_cache_path = (f"{cache_dir}/gradients/"
                            f"{model_name}-{dataset_short_name}_s{num_examples}_blk{seq_len}.pt")

    quantized_cache_path = (f"{cache_dir}/quantized/"
                          f"{'dns-' if dns else ''}{model_name}-w{parent_precision}_orig{seed_precision}"
                          f"-gc{group_size}-{dataset}_s{num_examples}_blk{seq_len}")

    model_output_path = (f"{cache_dir}/packed/"
                         f"anyprec-{model_name}-MY-w{parent_precision}_orig{seed_precision}"
                         f"-gc{group_size}-{dataset_short_name}_s{num_examples}_blk{seq_len}-ablation_loss_cs")
    # l2h_v1.0 : low to high version 1.0
    # 版本说明： 将比特位确定方式 由 根据误差的正负，改为 根据 补偿0/1的误差相对幅度 进行确定
    # l2h_v1.1 : low to high version 1.1
    # 版本说明： 在 l2h_v1.0 基础上，增加了 交替搜索的迭代次数，从3次增加到10次

    # l2h_v1.2 : low to high version 1.2.2
    # 版本说明： 在 l2h_v1.1 基础上，增加 交替优化scale 和 比特位 ，计算在不同补偿下的量化误差，选择最小(误差平方)的作为最终结果，然后基于加权均方和 通过最小二乘法求解 scale
    # 增加了 失效的补救措施：当 交替优化后 误差反而变大时，直接选择 补0 的结果，避免过冲  
   
    # l2h_v1.3 : 基于low to high version 1.2
    # 版本说明： 在 l2h_v1.2.2 基础上， 将 判断比特位补偿0/1的依据 由 绝对误差值平方，改为 余弦相似性

    # # l2h_v1.3-1 : low to high version 1.3
    # 版本说明： 在 l2h_v1.3 基础上，  尝试增加交替优化的次数至 9 次 -- 三次就够了  # 交替优化次数的 消融

    # # l2h_v1.3-2 : low to high version 1.3
    # 版本说明： 在 l2h_v1.3 基础上，  尝试增加交替优化的次数 恢复为 3 次； 调整 降级 判断依据 为最早的量化误差

    # # l2h_v1.3-3 : low to high version 1.3
    # 版本说明： 在 l2h_v1.3-2 基础上，  尝试 关闭 失效恢复机制     # 有无误差补偿机制 的消融   效果更好？

    
    
    # fb-h    -  将 海森矩阵 与 重要性关系 改为 反比
    # c-range -  将参数comp的范围调整为 gap_c = q_root - q_desc
    print(f'梯度保存路径{gradients_cache_path}, 输出模型路径为{model_output_path}')
    # ------------------- Gradients -------------------
    logging.info("------------------- Gradients -------------------")
    logging.info("Beginning gradient calculation...")
    # 根据 overwrite_gradients 以及 gradients_cache_path下有无梯度计算文件 对原有梯度文件进行删除
    if overwrite_gradients and os.path.exists(gradients_cache_path):
        logging.info(f"Detected cached gradients at {gradients_cache_path}. Will delete and recalculate.")
        os.remove(gradients_cache_path)
    # this will load and return the gradients if they exist, or calculate them if they don't exist

    my_ap_start = time.time()

    # 根据 数据集 dataset 进行梯度计算or读取  //     # **具体组成-未分析**
    model_gradients = get_gradients(
        analyzer=analyzer,
        dataset=dataset,
        seq_len=seq_len,
        num_examples=num_examples,
        save_path=gradients_cache_path,
        random_state=random_state
    )
    logging.info("Gradient calculation complete.")



    # ------------------- Quantize: Seed + Upscale -------------------
    logging.info("------------------- Quantize: Root + downgrade -------------------")

    # Calculate or load parent
    logging.info(f"Beginning {seed_precision}~{parent_precision}-bit Any-Precision Quantization...")
    # Note that this saves the seed model to the cache path and must be loaded for the upscale step
    if overwrite_quantize and os.path.exists(quantized_cache_path):
        # if the user wants to recalculate the seed, delete the cached seed
        logging.info(f"Detected cached parent at {quantized_cache_path}. Will delete and recalculate.")
        shutil.rmtree(quantized_cache_path)     # 递归删除文件夹下的所有子文件夹和子文件(包括文件夹本身)

    # this skips over existing layers in the cache, and doesn't overwrite them
    # 量化

    # TODO 注释起点
    quant_res = root_and_downgrade( #, weights_int
        analyzer=analyzer,                      #
        output_folder=quantized_cache_path,     # 量化结果保存路径
        descendant_precision=seed_precision,    # 量化范围下限
        root_precision=parent_precision,        # 量化范围上限
        num_examples=num_examples,
        dataset=dataset,
        seq_len=seq_len,
        cpu_count=cpu_count,
        random_state=random_state,
        group_size=group_size,
        gradients = model_gradients #None,#
    )

    if mode == 'quantize':
        return

    # del model_gradients  # free up memory
    analyzer.drop_original_weights()  # drop the original weights to save memory
    torch.cuda.empty_cache()    # new add

    logging.info("Quantization(Root + Downgrade) complete.")
    # return 0
    # TODO 注释终点


    my_ap_end = time.time()
    # # 打包    把量化后的 LUT 和权重编码为最终可用的模型格式。
    pack(
        analyzer=analyzer,
        quant_res=quant_res,
        # weights_int = weights_int,
        output_model_path=model_output_path,
        des_precision=seed_precision,
        root_precision=parent_precision,
        cpu_count=cpu_count,
        group_size=group_size,
        dns=dns,
    )

    logging.info("Packing complete.")
    my_ap_end_pack = time.time()

    #   #  保存 量化参数以及有关结果
    # saved_quant = {}
    # for k, v in quant_res.items():
    #     print(k)
    #     saved_quant[k] = (v[0], v[1], v[2],v[3], weights_int[k])    # scale, zero, g_ids, comp, w_int   # K
    # torch.save(saved_quant, ".\\cache\\quantized\\q_my.pt")
    # # torch.save(quant_res, '.\\cache\\quantized\\q_my.pt')

    del quant_res

    print('MY Any-precision 量化所需时间(不含保存)为：', my_ap_end - my_ap_start, '    ,with pack:', my_ap_end_pack - my_ap_start)

    # layers = analyzer.get_layers()  # 通过分析器获取模型的所有层 得到模型 layer 结构 // 即transformer层
    # num=1
    # for layer in layers:
    #     modules_dict = analyzer.get_modules(layer)
    #     if num ==1:
    #         print(modules_dict)
    #         num+=1
    #     for module_name, module in modules_dict.items():
    #         print(f"Module: {module_name}, Type: {type(module).__name__}")
    # # 得到get_gradients()中， 关于 layers = analyzer.get_layers()  # 得到模型 layer 结构的相同结果
    # module = analyzer.model
    # print('a',analyzer.model_name)
    # for attrib_name in analyzer.model_name.split('.'):
    #     module = getattr(module, attrib_name)
    #     print('B', attrib_name,'b1\n',module)
    #
    # for attrib_name in analyzer.layers_name.split('.'):
    #     module = getattr(module, attrib_name)  # 返回对象module 的 attrib_name 属性
    #     print('C', attrib_name,'c1\n',module)