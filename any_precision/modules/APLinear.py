import torch
import torch.nn as nn

# try:
#     from any_precision_ext import matmul_kbit, dequant_kbit
# except:
#     matmul_kbit, dequant_kbit = None, None


class APLinear(nn.Module):
    def __init__(self, in_features, out_features, supported_bits, group_size, bias=True, precisions=None, device=None,layer_name=None,
                 dtype=None):
        super().__init__()
        # if dequant_kbit is None or matmul_kbit is None:
        #     # 检查是否已经安装了自定义的CUDA内核拓展 (dequant_kbit 和 matmul_kbit)，
        #     raise ModuleNotFoundError('Please install any precision CUDA kernel extension from modules/kernels.')
        if precisions is None:
            precisions = supported_bits
        if not isinstance(precisions, list):
            # 确保 precisions为列表
            raise RuntimeError('supported_bits must be a list of integers.')
        if dtype is not None and dtype != torch.float16:
            # 仅支持torch.float16类型
            raise RuntimeError('Only float16 is supported for now.')

        self.in_features = in_features
        self.out_features = out_features
        self.precisions = precisions
        self.precision = max(self.precisions)   # 当前精度  默认为当前支持精度中的最高精度; 会根据set_precision进行变化
        self.supported_bits = supported_bits

        self.group_size = group_size
        self.max_supported_bits = max(self.precisions)  # 当前支持精度中的最高精度

        self.register_buffer(
            'qweight',
            torch.empty((max(supported_bits), out_features, in_features // 32), dtype=torch.int32, device=device)
        )

        self.register_buffer('zero', torch.empty((out_features, in_features//self.group_size)))

        for bit in range(min(self.precisions), max(self.precisions)+1):
            self.register_buffer(f'scale{bit}', torch.empty((out_features, in_features // self.group_size)))
            # self.register_buffer(f'zero{bit}', torch.empty((out_features, in_features // self.group_size)))

        if bias:
            self.register_buffer(
                "bias",
                torch.empty((out_features,), dtype=dtype, device=device)
            )
        else:
            self.bias = None

        self.res_save = {}
        self.count = 0
        self.name = layer_name

    def prune_precisions(self):
        # 只保留当前精度模型？
        # print('X-OUT-TEST: ',max(self.precisions))
        self.qweight = self.qweight[:max(self.precisions)]
        for bit in self.supported_bits:
            if bit not in self.precisions:
                # delattr(self, f'lut{bit}')
                delattr(self, f'comp{bit}')

    def forward(self, x, **kwargs):
        if 'precision' in kwargs:
            w_bits = kwargs['precision']
        else:
            w_bits = self.precision

        # print(" MAT- CODE-PATH ")
        weight = self._dequant_temp(w_bits, self.group_size, self.qweight, self._buffers[f'scale{w_bits}'], self._buffers[f'zero'])

        out = torch.matmul(x, weight.T)
        out = out + self.bias if self.bias is not None else out

        return out


    def _dequant_temp(self, w_bits, group_size, qweight, scale, zero):
        """
        Dequantize weights stored in bitplanes.

        Args:
          w_bits 当前bit，group_size 分组大小； qweight 当前比特的权重， scale 缩放因子， zero 偏移值， comp补偿值(仅用于非最高比特)
          w_bits: precision to dequantize to (e.g. 4/5/6/7/8).
          group_size: size of channel group used for per-group scale/zero.
          qweight: Tensor[int32] shape [max_precision, out_features, in_chunks],
                   where in_chunks = in_features // 32
          scale: Tensor[float32] shape [out_features, in_features//group_size]
          zero: Tensor[float32] shape [out_features, in_features//group_size]
          comp: Tensor[float32] same shape as scale/zero or None.

        Returns:
          weight: Tensor[float32] shape [out_features, in_features]
        """
        # 1. 解析位平面存储的权重wbits；
        # 2. 根据 group_size 以及 是否 完成量化参数 和 权重的匹配
        # 3. 根据有无comp 实现具体的反量化； 如果comp 为 none 则默认当前wbits为最高精度

        # device = qweight.device
        # weight_dtype = qweight.dtype    # torch.int32
        _, out_features, in_chunks = qweight.shape
        in_features = in_chunks * 32
        qweight_sub = qweight[:w_bits]  # 只是用前bit为的位平面
        # max_bits = self.max_supported_bits

        # 将权重由位平面重新解析得到 weight
        weight = restore_uint8_from_weighttensor_torch(qweight_sub, w_bits)

        # --- 3) apply scale/zero/comp formula ---
        # Convert w_int to float for arithmetic
        weight_f = weight.to(torch.float16)  # torch.uint8

        
        bit_err = w_bits -  min(self.precisions)
        zero = zero * 2**bit_err

        # repeat_interleave (fast)
        scale_pc = scale.repeat_interleave(group_size, dim=1)
        zero_pc = zero.repeat_interleave(group_size, dim=1)

        if scale_pc.shape[1] > in_features:
            scale_pc = scale_pc[:, :in_features]
            zero_pc = zero_pc[:, :in_features]

        out = scale_pc * (weight_f - zero_pc)
        return out
    
    def _dequant_temp_for_noscale(self, w_bits, group_size, qweight, scale, zero):
        """
        Dequantize weights stored in bitplanes.
        Args:
          w_bits 当前bit，group_size 分组大小； qweight 当前比特的权重， scale 缩放因子， zero 偏移值， comp补偿值(仅用于非最高比特)
          w_bits: precision to dequantize to (e.g. 4/5/6/7/8).
          group_size: size of channel group used for per-group scale/zero.
          qweight: Tensor[int32] shape [max_precision, out_features, in_chunks],
                   where in_chunks = in_features // 32
          scale: Tensor[float32] shape [out_features, in_features//group_size]
          zero: Tensor[float32] shape [out_features, in_features//group_size]
          comp: Tensor[float32] same shape as scale/zero or None.

        Returns:
          weight: Tensor[float32] shape [out_features, in_features]
        """
        # 1. 解析位平面存储的权重wbits；
        # 2. 根据 group_size 以及 是否 完成量化参数 和 权重的匹配
        # 3. 根据有无comp 实现具体的反量化； 如果comp 为 none 则默认当前wbits为最高精度

        # device = qweight.device
        # weight_dtype = qweight.dtype    # torch.int32
        lowest_bit = 4

        _, out_features, in_chunks = qweight.shape
        in_features = in_chunks * 32
        qweight_sub = qweight[:lowest_bit]  # 只是用前bit为的位平面
        # max_bits = self.max_supported_bits

        # 将权重由位平面重新解析得到 weight
        weight = restore_uint8_from_weighttensor_torch(qweight_sub, lowest_bit)
        bit_err = w_bits -  min(self.precisions)
        # --- 3) apply scale/zero/comp formula ---
        # Convert w_int to float for arithmetic
        weight_f = weight.to(torch.float16) * 2**bit_err # torch.uint8
        zero = zero * 2**bit_err

        # repeat_interleave (fast)
        scale_pc = scale.repeat_interleave(group_size, dim=1)
        zero_pc = zero.repeat_interleave(group_size, dim=1)

        if scale_pc.shape[1] > in_features:
            scale_pc = scale_pc[:, :in_features]
            zero_pc = zero_pc[:, :in_features]

        out = scale_pc * (weight_f - zero_pc)
        return out

    def set_precision(self, precision):
        # 设置当前layer的精度
        # print('精度：', precision)
        if precision not in self.precisions:
            raise RuntimeError(f"{self.precisions}-bit precisions are supported but {precision}-bit was specified.")
        self.precision = precision

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'



def make_new_byte_indices_torch(total_bytes: int, device):
    bytes_per_thread = 4
    threads_per_warp = 32
    bytes_per_warp = threads_per_warp * bytes_per_thread

    full_warp_bytes = (total_bytes // bytes_per_warp) * bytes_per_warp
    remaining_bytes = total_bytes - full_warp_bytes

    # --- full warps ---
    if full_warp_bytes > 0:
        byte_indices = torch.arange(full_warp_bytes, device=device)
        warp_idx = byte_indices // bytes_per_warp
        byte_offsets_within_warp = byte_indices % bytes_per_warp
        warp_offsets = warp_idx * bytes_per_warp
        thread_indices = byte_indices % threads_per_warp
        byte_offsets_within_thread = byte_offsets_within_warp // threads_per_warp
        byte_offsets_within_thread ^= 3

        new_full = warp_offsets + thread_indices * bytes_per_thread + byte_offsets_within_thread
    else:
        new_full = torch.empty((0,), dtype=torch.long, device=device)

    # --- remaining bytes ---
    if remaining_bytes > 0:
        rem_indices = torch.arange(remaining_bytes, device=device)
        adj_threads = remaining_bytes // bytes_per_thread

        if adj_threads > 0:
            warp_idx = rem_indices // bytes_per_warp
            warp_offsets = warp_idx * bytes_per_warp
            thread_indices = rem_indices % adj_threads
            byte_offsets_within_thread = (rem_indices // adj_threads) ^ 3
            new_rem = warp_offsets + thread_indices * 4 + byte_offsets_within_thread + full_warp_bytes
        else:
            new_rem = torch.arange(full_warp_bytes,
                                   full_warp_bytes + remaining_bytes,
                                   device=device)
    else:
        new_rem = torch.empty((0,), dtype=torch.long, device=device)

    return torch.cat([new_full, new_rem], dim=0)

def unpermute_bytes_torch(permuted_bytes, new_byte_indices):
    # p = argsort(new_byte_indices)
    p = torch.argsort(new_byte_indices)

    # inv_p = argsort(p)
    inv_p = torch.argsort(p)

    # permuted_bytes shape: [w_bits, N, total_bytes]
    return permuted_bytes.index_select(dim=2, index=inv_p)

def unpackbits_torch(x):  # x: uint8, shape (..., B)
    # 输出 shape (..., B*8)
    bits = [(x >> i) & 1 for i in reversed(range(8))]
    return torch.stack(bits, dim=-1).reshape(*x.shape[:-1], -1)

def restore_uint8_from_weighttensor_torch(weighttensor, wbit=8):
    device = weighttensor.device

    w_bits, N, int32_per_row = weighttensor.shape
    total_bytes = int32_per_row * 4 # total_bytes 以8bit长度保存的字节总数 = 每个int32为保存的数据拆成4份

    # 1. int32 → uint8 view
    bytes_view = weighttensor.view(torch.uint8).reshape(w_bits, N, total_bytes)
    # 2. 构造 new_byte_indices
    new_idx = make_new_byte_indices_torch(total_bytes, device)
    # 3. 逆置换
    unpermuted = unpermute_bytes_torch(bytes_view, new_idx)
    # 4. unpack bits
    unpacked = unpackbits_torch(unpermuted)
    effective_bits = unpacked[:wbit]
    # K = unpacked.shape[-1]  # == total_bytes * 8
    # 5. 按 bitplane 组合成 uint8
    # weight = 1 << (root_precision - 1 - bit)
    weights = torch.tensor([1 << (wbit - 1 - bit) for bit in range(wbit)], dtype=torch.uint8, device=device)
    qweights = (effective_bits * weights[:, None, None]).sum(dim=0)
    return qweights