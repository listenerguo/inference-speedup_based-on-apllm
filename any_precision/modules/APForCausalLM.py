import gc
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    AutoConfig,
    AutoModelForCausalLM,
)
from accelerate.big_modeling import (
    init_empty_weights,
    load_checkpoint_and_dispatch,
)

from .APLinear import APLinear
from any_precision.analyzer.analyzer import get_analyzer


def replace_module_by_name(layer, module_name, new_module):
    levels = module_name.split('.')
    module = layer
    for level in levels[:-1]:
        module = getattr(module, level) if not level.isdigit() else module[int(level)]
    setattr(module, levels[-1], new_module)


class APForCausalLM(nn.Module):
    def __init__(
            self,
            model_path,
            config,
            precisions=None,
            torch_dtype=torch.float16,
            fuse_layers=False,
            trust_remote_code=True,
    ):
        super().__init__()

        self.config = config    # 保存传入的模型配置

        self.supported_bits = list(range(self.config.anyprec['des_precision'],
                                         self.config.anyprec['root_precision'] + 1))   # 读取模型的 精度范围
        if precisions is None:
            self.precisions = self.supported_bits   # 如果没有传入精度，则读取模型配置中的 所支持的精度范围
        else:
            assert len(precisions) == len(set(precisions)), "Precisions must be unique"
            assert all(bit in self.supported_bits for bit in precisions), \
                f"Supported bits {precisions} must be a subset of model supported bits {self.supported_bits}"
            self.precisions = precisions    # 当前模型的 精度 集合

        self.precision = max(self.precisions)   # 设置当前精度为支持的最大精度值（默认使用最高精度）

        with init_empty_weights():  # 使用HuggingFace的init_empty_weights上下文管理器，该管理器允许在不实际分配内存的情况下初始化模型权重
            self.model = AutoModelForCausalLM.from_config(
                config=config,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                # attn_implementation="flash_attention_2",
            )
        # print("初始化 apforcausallm")
        self.group_size = self.model.config.anyprec['group_count']

        # print(model_path)
        # print(self.model)
        self.analyzer = get_analyzer(self.model)    # 调用 analyzer 中的函数    #获取模型分析器，用于分析模型结构和提取特定层
        self.ap_linears = []

        # Replace to AnyPrecisionLinear layers
        self._load_quantized_modules()  # 修改1：将普通的线性层替换为任意精度线性层 -- aplinear

        print('绑定权重')
        self.tie_weights()  #  绑定权重

        device_map = {key: 'cpu' for key in self.model.state_dict().keys()}

        # loads the weights into modules and distributes
        # across available devices automatically
        # 将权重加载到模块中，并自动在可用设备间分配 // 可基于该部分代码,进一步拓展为多设备按比特并行
        load_checkpoint_and_dispatch(
            self.model,
            checkpoint=model_path,
            device_map=device_map,
            no_split_module_classes=[self.layer_type],
            dtype=torch_dtype,
        )

        # Dispath to devices
        if fuse_layers:
            self.fuse_layers()

        self.prune_precisions()

    def forward(self, *args, **kwargs):
        prev_precision = self.precision
        if 'precision' in kwargs:
            precision = kwargs.pop('precision')
            self.set_precision(precision)
        # print('精度：', prev_precision)
        results = self.model.forward(*args, **kwargs)

        self.set_precision(prev_precision)
        return results

    def generate(self, *args, **kwargs):
        if 'precision' in kwargs:
            # 获取传入generate函数的precision参数
            prev_precision = self.precision
            precision = kwargs.pop('precision')
            self.set_precision(precision)
        else:
            prev_precision = self.precision # 将默认参数作为当前推理参数

        with torch.inference_mode():
            results = self.model.generate(*args, **kwargs)  # 执行推理

        self.set_precision(prev_precision)
        return results

    @staticmethod
    def _load_config(
            model_path,
            trust_remote_code=True,
    ):
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        return config

    @classmethod
    def from_quantized(
            cls,
            quant_model_path,
            trust_remote_code=True,
            fuse_layers=False,
            precisions=None
    ):
        config = cls._load_config(quant_model_path, trust_remote_code)

        ap_model = cls(
            model_path=quant_model_path,
            precisions=precisions,
            config=config,
            fuse_layers=fuse_layers,
            trust_remote_code=trust_remote_code,
        )

        return ap_model

    def _load_quantized_modules(self):
        # Get blocks of model
        layers = self.analyzer.get_layers()
        layer_id = 0
        for layer in tqdm(layers, desc="Loading AP Layers"):
            # Get every linear layer in a block
            named_linears = self.analyzer.get_modules(layer)

            # Replace nn.Linear with APLinear
            for name, module in named_linears.items():
                name_temp =f"model.decoder.layers.{layer_id}.{name}"
                wqlinear = APLinear(
                    module.in_features, module.out_features,
                    self.supported_bits, self.group_size,
                    bias=module.bias is not None,
                    precisions=self.precisions,
                    device=module.weight.device,
                    layer_name = name_temp,
                )
                # print('name:',name, wqlinear.comp7.shape,'//',wqlinear.scale.shape, '--vs--', module.in_features, module.out_features)
                self.ap_linears.append(wqlinear)
                replace_module_by_name(layer, name, wqlinear)

            torch.cuda.empty_cache()
            gc.collect()
            layer_id+=1

    def prune_precisions(self):
        for ap_linear in self.ap_linears:
            ap_linear.prune_precisions()

        torch.cuda.empty_cache()
        gc.collect()

    def set_precision(self, precision):
        for ap_linear in self.ap_linears:
            ap_linear.set_precision(precision)
        self.precision = precision

    def tie_weights(self):
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()

    def get_model_layers(self):
        module = self.model
        for attrib_name in self.config.anyprec['arch_config']['model_name'].split('.'):
            module = getattr(module, attrib_name)
        return getattr(module, self.config.anyprec['arch_config']['layers_name'])

    def fuse_layers(self):
        if 'fuse_target_layers' not in self.model_config:
            raise NotImplementedError("This model does not support layer fusion")
        # TODO implement layer fusion
        pass

    @property
    def layer_type(self):
        for layer in self.get_model_layers():
            layer_class_name = layer.__class__.__name__
            if layer_class_name.endswith("DecoderLayer"):
                return layer_class_name
        return None

    @property
    def device(self):
        return self.model.device
