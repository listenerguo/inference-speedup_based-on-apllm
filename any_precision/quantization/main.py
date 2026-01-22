
import os
import os.path
import shutil
import logging

from .config import *
from ..analyzer import get_analyzer
from .gradients import get_gradients
# from .quantize import seed_and_upscale
from .pack import pack
from .dense_and_sparse import remove_outliers
import torch

# Disable parallelism in tokenizers to prevent warnings when forking in the seed generation step
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Logging with time sans date, level name, and message
logging.basicConfig(level=logging.INFO, format='[%(asctime)s | %(levelname)s] %(message)s', datefmt='%H:%M:%S')


def any_precision_quantize(
        model,  # 输入模型路径 / 或名称
        seed_precision=DEFAULT_SEED_PRECISION,  # 设置最小量化精度
        parent_precision=DEFAULT_PARENT_PRECISION,  # 设置最大量化精度
        mode='pack',    # ？ 将模型打包
        yaml_path=None, cache_dir=DEFAULT_CACHE_DIR,    # cache_dir : 模型打包保存位置
        dataset=DEFAULT_DATASET,        # dataset 数据集名称(实现数据集路径为输入)  # 函数 get_gradients -> get_tokens -> _get_dataset
        seq_len=DEFAULT_SEQ_LEN, num_examples=DEFAULT_NUM_EXAMPLES,    # ！！！不理解
        cpu_count=os.cpu_count(),       # cpu数量？ 用途？
        overwrite_gradients=False,
        overwrite_quantize=False,
        overwrite_pack=False,
        random_state=None,
        group_count=1,      # group 大小
        dns=False,          # Dense & Sparse : 找出异常权重，得到稀疏化权重并保存
        sensitivity_outlier_percent=0.05,   # squeezeLLM 敏感度
        threshold_outlier_percent=0.40,     # squeezeLLM 异常值
        cpu_only=False,
):
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

    logging.info(f"Running Any-Precision Quantization on {model_name} with seed precision {seed_precision} and "
                 f"parent precision {parent_precision} using {dataset} for gradient calculation")

    # ------------------- Load model -------------------
    # 通过analyzer 返回一个 ModelAnalyzer 对象，它能提取模型结构信息（层、权重、模块名），方便后续量化      # **具体组成-未分析**
    analyzer = get_analyzer(
        model,
        yaml_path=yaml_path,
        include_tokenizer=True,
        cpu_only=cpu_only
    )
    logging.info("model load success! ")

    # ------------------- Set cache paths -------------------
    gradients_cache_path = (f"{cache_dir}/gradients/"
                            f"({model_name})-{dataset}_s{num_examples}_blk{seq_len}.pt")

    quantized_cache_path = (f"{cache_dir}/quantized/"
                          f"{'dns-' if dns else ''}({model_name})-w{parent_precision}_orig{seed_precision}"
                          f"-gc{group_count}-{dataset}_s{num_examples}_blk{seq_len}")

    model_output_path = (f"{cache_dir}/packed/"
                         f"anyprec-({model_name})-w{parent_precision}_orig{seed_precision}"
                         f"-gc{group_count}-{dataset}_s{num_examples}_blk{seq_len}")


    # ------------------- Gradients -------------------
    logging.info("------------------- Gradients -------------------")
    logging.info("Beginning gradient calculation...")
    # Calculate or load gradients
    if overwrite_gradients and os.path.exists(gradients_cache_path):
        # if the user wants to recalculate the gradients, delete the cached gradients
        # 根据 overwrite_gradients 以及 gradients_cache_path下有无梯度计算文件 对原有梯度文件进行删除
        logging.info(f"Detected cached gradients at {gradients_cache_path}. Will delete and recalculate.")
        os.remove(gradients_cache_path)

    # this will load and return the gradients if they exist, or calculate them if they don't exist
    # 根据 数据集 dataset 进行梯度计算  //  首先会根据 gradients_cache_path 下有无数据文件，确定执行计算 还是直接读取   # **具体组成-未分析**
    model_gradients = get_gradients(
        analyzer=analyzer,
        dataset=dataset,
        seq_len=seq_len,
        num_examples=num_examples,
        save_path=gradients_cache_path,
        random_state=random_state,
    )
    logging.info("Gradient calculation complete.")

    if mode == 'gradients':
        return

    # ------------------- Dense & Sparse -------------------
    if dns:
        logging.info("------------------- Dense & Sparse -------------------")
        # 根据squeezeLLM 设计   # **具体组成-未分析**
        sparse_model_weights = remove_outliers(
            analyzer=analyzer,
            gradients=model_gradients,
            sensitivity_outlier_percent=sensitivity_outlier_percent,
            threshold_outlier_percent=threshold_outlier_percent,
        )
        sparse_path = f"{quantized_cache_path}/sparse"
        os.makedirs(sparse_path, exist_ok=True)
        for l in range(analyzer.num_layers):
            torch.save(sparse_model_weights[l], f"{sparse_path}/l{l}.pt")

        del sparse_model_weights

    # ------------------- Quantize: Seed + Upscale -------------------
    logging.info("------------------- Quantize: Seed + Upscale -------------------")

    # Calculate or load parent
    logging.info(f"Beginning {seed_precision}~{parent_precision}-bit Any-Precision Quantization...")
    # Note that this saves the seed model to the cache path and must be loaded for the upscale step
    if overwrite_quantize and os.path.exists(quantized_cache_path):
        # if the user wants to recalculate the seed, delete the cached seed
        logging.info(f"Detected cached parent at {quantized_cache_path}. Will delete and recalculate.")
        shutil.rmtree(quantized_cache_path)

    # this skips over existing layers in the cache, and doesn't overwrite them
    # 量化    # 得到 LUT（查找表）和量化权重 -- 非均匀量化
    seed_and_upscale(
        analyzer=analyzer,
        gradients=model_gradients,
        output_folder=quantized_cache_path,
        seed_precision=seed_precision,
        parent_precision=parent_precision,
        cpu_count=cpu_count,
        random_state=random_state,
        group_count=group_count,
    )

    if mode == 'quantize':
        return

    del model_gradients  # free up memory
    analyzer.drop_original_weights()  # drop the original weights to save memory

    logging.info("Quantization(Seed + Upscale) complete.")

    # ------------------- Pack -------------------
    logging.info("------------------- Pack -------------------")

    # check for non-empty directory
    if os.path.exists(model_output_path) and os.path.isdir(model_output_path) and os.listdir(model_output_path):
        if overwrite_pack:
            logging.info(f"Model output path {model_output_path} already exists and is not empty. Will delete and "
                         f"re-pack.")
            shutil.rmtree(model_output_path)
        else:
            # if the user doesn't want to overwrite the pack, but the directory is not empty, skip packing
            logging.info(f"Model output path {model_output_path} already exists and is not empty. Will skip packing.")
            return
    # 打包    把量化后的 LUT 和权重编码为最终可用的模型格式。
    pack(
        analyzer=analyzer,
        lut_path=quantized_cache_path,
        output_model_path=model_output_path,
        seed_precision=seed_precision,
        parent_precision=parent_precision,
        cpu_count=cpu_count,
        group_count=group_count,
        dns=dns,
    )

    logging.info("Packing complete.")
