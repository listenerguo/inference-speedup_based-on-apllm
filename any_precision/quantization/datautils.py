from datasets import load_dataset,load_from_disk
import random
import numpy as np
import logging
import os
import torch


def _get_wikitext2(split, path_sign=False, datapath=None):
    assert split in ['train', 'validation', 'test'], f"Unknown split {split} for wikitext2"
    if path_sign:
        # 手动下载数据
        # data = load_dataset('wikitext', 'wikitext-2-raw-v1', trust_remote_code=True)
        # data.save_to_disk("xxxx//wikitext2")
        data = load_from_disk(datapath)
        data = data[split]
    else:
        data = load_dataset('wikitext', 'wikitext-2-raw-v1', split=split, trust_remote_code=True)
    return data['text']


def _get_ptb(split, slice_unk=True, path_sign=False, datapath=None):
    assert split in ['train', 'validation', 'test'], f"Unknown split {split} for ptb"

    if path_sign:
        # 手动下载数据
        # data = load_dataset('ptb_text_only', 'penn_treebank', trust_remote_code=True)
        # data.save_to_disk("xxxx//ptb")
        data = load_from_disk(datapath)
        data = data[split]
    else:
        data = load_dataset('ptb_text_only', 'penn_treebank', split=split,
                            trust_remote_code=True)
    data_list = data['sentence']

    if slice_unk:
        data_list = [s.replace('<unk>', '< u n k >') for s in data_list]
    return data_list


def _get_c4(split, path_sign=False, datapath=None):
    assert split in ['train', 'validation'], f"Unknown split {split} for c4"
    if path_sign:
        # 手动下载数据
        # dataset = load_dataset('allenai/c4',
        #     data_files={'train': 'en/c4-train.00000-of-01024.json.gz', 'validation': 'en/c4-validation.00000-of-00008.json.gz'},
        #     trust_remote_code=True
        # )
        # dataset.save_to_disk("xxxx//c4")
        new_datapath = os.path.join(datapath, split)
        print(new_datapath)
        data = load_from_disk(new_datapath)
    else:
        if split == 'train':
            data = load_dataset(
                'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train',
                trust_remote_code=True
            )
        else:
            assert split == 'validation'
            data = load_dataset(
                'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'}, split='validation',
                trust_remote_code=True
            )

    return data['text']


def _get_pileval(split, path_sign=False, datapath=None):
    if split != 'validation':
        logging.warning(f"Pileval only has a validation split, but got split={split}. Using validation split.")
    # 手动下载数据
    # data = load_dataset("mit-han-lab/pile-val-backup", split="validation", trust_remote_code=True)
    # data.save_to_disk("xxxx//pileval")
    if path_sign:
        data = load_from_disk(datapath)
    else:
        data = load_dataset("mit-han-lab/pile-val-backup", split="validation", trust_remote_code=True)
    return data['text']


def _sample_and_tokenize(texts, tokenizer, seq_len, num_samples, seed=None):
    """
    function：
        从 数据文本 texts 中，随机抽取 数量为 num_samples 的样本，并将其转为模型可识别的 tokens, 同时确保每个 令牌序列长度不超过seq_len
    parameters：
        texts：          输入数据文本 例如：
        tokenizer：      分词器 - 用于将文本转为数字序列或向量化
        seq_len：        向量化后，单个样本的最大长度
        num_samples：    随机取样的样本数
        seed：           随机数种子， 便于复现
    """
    assert num_samples <= len(texts), \
        f"num_samples({num_samples}) should be less than or equal to the number of texts({len(texts)})"
    # 基于文档级别的独立采样 -- 与gptq库中的 run_Q-opt_model.py 中的 get_wikitext2 都是获取校准数据，但处理方式不同
    # this works for None too, effectively setting random seeds
    random.seed(seed)
    np.random.seed(seed)

    selected_indices = set()

    samples = []
    while len(samples) < num_samples:
        idx = random.randint(0, len(texts) - 1)
        if idx in selected_indices:  # we don't want to sample the same text twice
            continue    # 如果重复，跳过以下内容，重新随机选择数据
        text = texts[idx]

        tokens = tokenizer(text, return_tensors='pt') # ['input_ids'][0]
        # return_tensors='pt'- 指定返回的张量类型为PyTorch张量
        # input_ids - 是文本被转换为的token ID序列  分词器返回的是一个字典，其中包含多个键，
        # [0] - input_ids是tokenizer返回的字典中键为input_ids的一个二维张量 (形状为[batch, 序列长度]-- 即tensor[[102, 24, 111, 238]])
        # print(tokens['input_ids'])
        # print(tokens['input_ids'][0])
        if len(tokens['input_ids'][0]) < seq_len:  # if the text is too short, we skip it
            continue    # 如果不够长，则将其跳过，重新选择数据

        # tokens = tokens[:seq_len]   #匹配tokenizer()['input_ids'][0] #  对于超出长度的输出token，截取长度为seq_len
        tokens = tokens.input_ids[:,0:seq_len]  # 匹配tokenizer() #  对于超出长度的输出token，截取长度为seq_len
        attention_mask = torch.ones_like(tokens)

        # 不重复，足够长 --> 保存数据
        selected_indices.add(idx)
        # samples.append(tokens)    #匹配tokenizer()['input_ids'][0]
        samples.append({'input_ids': tokens, 'attention_mask': attention_mask}) # 匹配tokenizer()

    return samples


def _get_dataset(dataset_name, split):
    # 添加一个通过本地路径，加载数据集的方式
    """
    加载数据集
    - dataset_name 是已知名称 (wikitext2, ptb, c4, pileval)，则调用相应函数
                   是路径 则用 load_from_disk 加载
    - split 数据集划分 -- 指明为训练集/验证集
    """
    if os.path.exists(dataset_name):
        # 从路径中提取最后的目录名
        basename = os.path.basename(dataset_name.rstrip('/'))
        # print(dataset_name, '\t', basename)
        if basename == 'wikitext2':
            # print(f'执行读取本地{basename}的数据集中的{split}')
            return _get_wikitext2(split, path_sign=True, datapath=dataset_name)
        elif basename == 'ptb':
            return _get_ptb(split, path_sign=True, datapath=dataset_name)
        elif basename == 'c4':
            return _get_c4(split, path_sign=True, datapath=dataset_name)
        elif basename == 'pileval':
            return _get_pileval(split, path_sign=True, datapath=dataset_name)
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")
    else:
        print(F'Input {dataset_name} not a path, will download')
        # 原始代码--备份
        if dataset_name == 'wikitext2':
            return _get_wikitext2(split)
        elif dataset_name == 'ptb':
            return _get_ptb(split)
        elif dataset_name == 'c4':
            return _get_c4(split)
        elif dataset_name == 'pileval':
            return _get_pileval(split)
        else:
            raise ValueError(f"Unknown dataset {dataset_name}")


def get_tokens(dataset_name, split, tokenizer, seq_len, num_samples, seed=None):
    logging.info(f"Fetching dataset: {dataset_name}")
    texts = _get_dataset(dataset_name, split)   # 取指定数据集/数据集路径 下的训练集/验证集
    logging.info(f"Sampling {num_samples} samples of length {seq_len} from {dataset_name}...")
    return _sample_and_tokenize(texts, tokenizer, seq_len, num_samples, seed)   # 将数据集随机采样并转为token返回

def get_dataset_from_disk(dataset_name,split):
    # 对 wikitext2 / c4 / ptb 验证成功
    texts = _get_dataset(dataset_name, split)
    print(f" 加载成功 success-{len(texts)}")
    return texts