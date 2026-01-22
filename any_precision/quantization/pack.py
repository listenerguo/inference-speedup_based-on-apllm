import numpy as np
from tqdm import tqdm
import os
import torch
import logging
from multiprocessing import Pool
import numba


_bytes_per_thread = 4


@numba.njit(cache=True)
def _permute_bitmaps(bitmaps):
    _, _, total_bytes = bitmaps.shape
    assert total_bytes % 4 == 0, "Number of bytes must be a multiple of 4"

    threads_per_warp = 32
    bytes_per_warp = threads_per_warp * _bytes_per_thread

    # Calculate the number of full warps and the starting index of remaining bytes
    full_warps_bytes = (total_bytes // bytes_per_warp) * bytes_per_warp
    remaining_bytes_start_idx = full_warps_bytes

    # Create an array of byte indices for full warps
    full_warp_byte_indices = np.arange(full_warps_bytes)
    # Calculate new indices for full warp bytes
    new_full_warp_byte_indices = _calculate_new_indices(full_warp_byte_indices, threads_per_warp)

    remaining_bytes = total_bytes - full_warps_bytes
    # Handle remaining bytes
    if remaining_bytes:
        remaining_byte_indices = np.arange(remaining_bytes)
        # Adjust the calculation for remaining bytes, which might not fill a complete warp
        adjusted_threads_per_warp = remaining_byte_indices.size // _bytes_per_thread
        new_remaining_byte_indices = _calculate_new_indices(remaining_byte_indices,
                                                            adjusted_threads_per_warp,
                                                            offset=remaining_bytes_start_idx)

        # Combine indices - the choice to not use np.concatenate is for numba compatibility
        new_byte_indices = np.empty(total_bytes, dtype=np.int64)
        new_byte_indices[:full_warps_bytes] = new_full_warp_byte_indices
        new_byte_indices[full_warps_bytes:] = new_remaining_byte_indices
    else:
        new_byte_indices = new_full_warp_byte_indices

    permuted_bitmaps = bitmaps[:, :, np.argsort(new_byte_indices)]

    return permuted_bitmaps


@numba.njit(cache=True)
def _calculate_new_indices(byte_indices, threads_per_warp, offset=0):
    """
    Calculate new byte indices for a given array of byte indices.
    """
    bytes_per_warp = threads_per_warp * _bytes_per_thread

    warp_idx, byte_offsets_within_warp = np.divmod(byte_indices, bytes_per_warp)

    warp_offsets = warp_idx * bytes_per_warp
    thread_indices = byte_indices % threads_per_warp

    # Change endianness within each thread and calculate new byte positions
    byte_offsets_within_thread = byte_offsets_within_warp // threads_per_warp
    byte_offsets_within_thread ^= 3  # Change endianness
    new_byte_indices = warp_offsets + thread_indices * _bytes_per_thread + byte_offsets_within_thread + offset

    return new_byte_indices


@numba.njit(cache=True)
def _permute_bitmaps_int32(bitmaps):
    """Return a permuted version of the input bitmaps, reshaped to int32."""
    w_bits, N, total_bytes = bitmaps.shape
    bitmaps = _permute_bitmaps(bitmaps)
    return bitmaps.reshape(-1, 4).view(np.int32).reshape(w_bits, N, total_bytes // 4)


def _process_layer_data(args):
    # (0, {xxxx}, 'model.decoder', 'layers', ['self_attn.q_proj', 'k_proj', 'v_proj', 'out_proj', 'fc1', 'fc2'], 8, 4)
    layer_idx, quantizer, model_name, layers_name, module_names, root_precision, des_precision = args
    layer_data = {}

    for i, name in enumerate(module_names):
        param_name = f'{model_name}.{layers_name}.{layer_idx}.{name}'
        scale, zero, g_idx, weights_int = quantizer[param_name]

        layer_weights = weights_int.cpu().numpy()   # weights_int[param_name]
        # 计算量化后的 root-bit 权重
        N, K = layer_weights.shape
        # K = group_count * group_size

        # 将其转换为 二进制并通过位平面形式打包，最后转为int32 保存至 weighttensor layer_data[param_name + '.qweight']
        qweight_flattened = layer_weights.flatten()   # # 展平为一维数组 # len(qweight_flattened) = N * K
        bitarray = np.empty((root_precision, len(qweight_flattened) // 8), dtype=np.uint8)  # 形状为 root_precision，len(qweight_flattened) // 8
        mask = 1 << (root_precision - 1)  # MSB first
        for bit in range(root_precision):
            curbitpack = np.packbits((qweight_flattened & mask).astype(bool))   # 将8位二进制数组打包成字节流
            bitarray[bit] = curbitpack
            mask >>= 1

        bitarray = bitarray.reshape((root_precision, N, K // 8))    # 将形状由 [root_precision, (N*K)//8] 调整为 [root_precision, N, K//8]
        weighttensor = _permute_bitmaps_int32(bitarray)     # 转换为int32类型    # weighttensor [root_precision, N, K//8//4]
        layer_data[param_name + '.qweight'] = weighttensor  # 存储到layer_data，键格式：模型名.层容器.层索引.模块名.qweight

        # 保存 量化参数zero
        # curParam_s = np.empty((N, scale.shape[1]), dtype=np.float16)    # np.float32
        curParam_z = np.empty((N, zero.shape[1]), dtype=np.float16)     # np.float32
        for r_idx in range(N):
            # curParam_s[r_idx] = scale[r_idx].cpu().numpy()
            curParam_z[r_idx] = zero[r_idx].cpu().numpy()
        # layer_data[param_name + '.scale'] = curParam_s
        layer_data[param_name + '.zero'] = curParam_z

        # 保存 量化参数scale
        qbit_keys = list(range(des_precision, root_precision+1))
        for i, bit in enumerate(qbit_keys):
            scale_bit_param =  scale[i]
            # 将每个精度的补偿 comp 保存 至 layer_data[param_name + '.lut' + str(bit)]
            curParam = np.empty((N, scale_bit_param.shape[1]), dtype=np.float16)     # np.float32
            for r_idx in range(N):
                curParam[r_idx] = scale_bit_param[r_idx].cpu().numpy()
            layer_data[param_name + '.scale' + str(bit)] = curParam

        # # 保存 量化参数scale and zero for multiple bits
        # qbit_keys = list(range(des_precision, root_precision+1))
        # for i, bit in enumerate(qbit_keys):
        #     scale_bit_param =  scale[i]
        #     zero_bit_param =  zero[i]
        #     # 将每个精度的补偿 comp 保存 至 layer_data[param_name + '.lut' + str(bit)]
        #     curParam_s = np.empty((N, scale_bit_param.shape[1]), dtype=np.float16)     # np.float32
        #     curParam_z = np.empty((N, zero_bit_param.shape[1]), dtype=np.float16)     # np.float32
        #     for r_idx in range(N):
        #         curParam_s[r_idx] = scale_bit_param[r_idx].cpu().numpy()
        #         curParam_z[r_idx] = zero_bit_param[r_idx].cpu().numpy()
        #     layer_data[param_name + '.scale' + str(bit)] = curParam_s
        #     layer_data[param_name + '.zero' + str(bit)] = curParam_z
    return layer_idx, layer_data

def pack(
        analyzer,
        quant_res,   # 改为 quantizers
        # weights_int,
        output_model_path,
        des_precision,
        root_precision,
        group_size=-1,
        dns=False,
        cpu_count=None
):

    # if group_count != 1:
    #     raise NotImplementedError("Group counts other than 1 are not supported yet for packing")

    # if dns:
    #     raise NotImplementedError("D&S packing is not supported yet")

    if cpu_count is None:
        cpu_count = os.cpu_count()

    # Limit cpu_count to 8 as larger values use too much memory, without much speedup
    _max_cpu_count = 1
    if cpu_count > _max_cpu_count:
        logging.warning(f"cpu_count will be limited to 8 to avoid excessive memory usage. "
                        f"Original value: {cpu_count}")
        cpu_count = _max_cpu_count

    tokenizer = analyzer.tokenizer
    num_layers = analyzer.num_layers

    model_name = analyzer.model_name
    layers_name = analyzer.layers_name
    module_names = analyzer.module_names
    config = analyzer.config  # original model config   # 原始模型信息文件
    arch_config = analyzer.get_arch_config()    # 自定义的模型相关信息文件，以yaml格式存于architectures

    state_dict = analyzer.state_dict

    args_list = [(layer_idx, quant_res, model_name, layers_name, module_names, root_precision, des_precision) for
                 layer_idx in range(num_layers)]    # 每个decoder进行循环
    # args_list = [(layer_idx, weights_int, quant_res, model_name, layers_name, module_names, root_precision, des_precision) for
                 # layer_idx in range(num_layers)]    # 每个decoder进行循环

    # # return 0
    # with Pool(cpu_count) as pool:
    #     # 遍历每个layer
    #     # 分线程执行 _process_layer_data函数，输入为args_list；输出为 layer_idx， layer_data
    #     for layer_idx, layer_data in tqdm(pool.imap(_process_layer_data, args_list), total=num_layers, desc="Packing"):
    #         for key, value in layer_data.items():
    #             state_dict[key] = torch.from_numpy(value)  # 将数据转化，替换到模型state_dict中对应key的位置 # 即Update with modified weights

    for args in tqdm(args_list, total=num_layers, desc="Packing"):
        layer_idx, layer_data = _process_layer_data(args)
        for key, value in layer_data.items():
            state_dict[key] = torch.from_numpy(value)

    # add new config parameters
    anyprec_configs = {
        'des_precision': des_precision,
        'root_precision': root_precision,
        'group_count': group_size,
        'arch_config': arch_config
    }
    config.anyprec = anyprec_configs

    logging.info(f"Writing model to disk...")
    os.makedirs(output_model_path, exist_ok=True)
    torch.save(state_dict, os.path.join(output_model_path, 'pytorch_model.bin'))
    tokenizer.save_pretrained(output_model_path)
    config.save_pretrained(output_model_path)
    logging.info(f"Model saved to {output_model_path}")
