# Originally from https://github.com/IST-DASLab/gptq/blob/main/datautils.py
# Modified to:
# - Only return the test set
# - Skip the tokenization step (return the datasets as-is)

#  new add function --- by guo  2025.12.30
#  load dataset from disk

import numpy as np
import torch
from datasets import load_dataset, load_from_disk


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def get_wikitext2(path=None):
    if path is None:
        testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
    else:
        testdata = load_from_disk(path)
        testdata = testdata['test']
    return "\n\n".join(testdata['text'])


def get_ptb(path=None):
    if path is None:
        valdata = load_dataset('ptb_text_only', 'penn_treebank', split='validation')
    else:
        valdata = load_from_disk(path)
        valdata = valdata['validation']
    return "\n\n".join(valdata['sentence'])


def get_c4():
    raise NotImplementedError("Only C4-new has been refactored to use the new dataset API")


def get_ptb_new(path=None):
    if path is None:
        testdata = load_dataset('ptb_text_only', 'penn_treebank', split='test')
    else:
        testdata = load_from_disk(path)
        testdata = testdata['test']
    return " ".join(testdata['sentence'])


def get_ptb_new_sliced(path=None):
    raw_text = get_ptb_new(path)
    sliced = raw_text.replace('<unk>', '< u n k >')
    return sliced


def get_c4_new(path=None):
    if path is None:
        valdata = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00004-of-00008.json.gz'},
            split='validation'
        )
    else:
        valdata = load_from_disk(path)
        valdata = valdata['validation']

    # The original datautils from the GPTQ paper had two filters:
    #   1. get only the first 1100 examples
    #   2. tokenize, then get the first seqlen * 256 tokens, where seqlen defaulted to 2048
    # This resulted in 524288 tokens, which in turn decode back into 2088532 characters.

    # However in my version, I am only returning the text, and leaving the tokenization to the caller.
    # Therefore, I replace the second filter of tokens into an equivalent filter of characters.
    return " ".join(valdata[:1100]['text'])[:2088528]


def get_loaders(name):
    if 'wikitext2' in name:
        return get_wikitext2()
    if 'ptb' in name:
        if 'new' in name:
            if 'sliced' in name:
                return get_ptb_new_sliced()
            else:
                return get_ptb_new()
        return get_ptb()
    if 'c4' in name:
        if 'new' in name:
            return get_c4_new()
        return get_c4()

    raise ValueError(f"Unknown dataset {name}")


def get_loaders_with_path(name):
    if 'wikitext2' in name:
        return get_wikitext2(name)
    if 'ptb' in name:
        return get_ptb_new_sliced(name)
        # if 'new' in name:
        #     if 'sliced' in name:
        #         return get_ptb_new_sliced(name)
        #     else:
        #         return get_ptb_new(name)
        # return get_ptb(name)
    if 'c4' in name:
        return get_c4_new(name)
        # if 'new' in name:
        #     return get_c4_new(name)
        # return get_c4()

    raise ValueError(f"Unknown dataset {name}")
