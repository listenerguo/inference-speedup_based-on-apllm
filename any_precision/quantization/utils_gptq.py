import json
import logging
import os
from logging import getLogger
from typing import List, Optional, Union

import accelerate
import numpy as np
import torch
import torch.nn as nn
import transformers
import threadpoolctl as tctl
from tqdm import tqdm
from transformers import AutoConfig
from transformers.utils.hub import cached_file

# from ..utils.import_utils import dynamically_import_QuantLinear
# from ..utils.modeling_utils import recurse_setattr
# from ._const import CPU, CUDA_0, EXLLAMA_DEFAULT_MAX_INPUT_LENGTH, SUPPORTED_MODELS


def get_device(obj: Union[torch.Tensor, nn.Module]):
    if isinstance(obj, torch.Tensor):
        return obj.device
    return next(obj.parameters()).device


def move_to_device(obj: Optional[Union[torch.Tensor, nn.Module]], device: torch.device):
    if obj is None:
        return obj
    else:
        if get_device(obj) != device:
            obj = obj.to(device)
        return obj

def nested_move_to_device(v, device):
    if isinstance(v, torch.Tensor):
        return move_to_device(v, device)
    elif isinstance(v, (list, tuple)):
        return type(v)([nested_move_to_device(e, device) for e in v])
    else:
        return v


# __all__ = [
#     "get_device",
#     "move_to_device",
# ]