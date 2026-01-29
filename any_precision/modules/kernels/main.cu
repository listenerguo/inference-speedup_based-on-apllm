#include <assert.h>
#include <torch/extension.h>
#include "matmul.cuh"
#include "dequant_formula.cuh"

void cudaError(cudaError_t errCode, const char * filename, int linenum) {
    if(errCode != cudaSuccess) {
        printf("Error : %s (%s : %d)\n", cudaGetErrorString(errCode), filename, linenum);
        exit(EXIT_FAILURE);
    }
}

#define HANDLE_ERROR(err) (cudaError(err, __FILE__, __LINE__))

typedef void (* matmul_func) (
    const __half *, const uint32_t *,
    const uint32_t, const uint32_t, const uint32_t,
    const __half *, const __half *, __half *
);

template <int s, int e>
struct get_matmul_func {
    void operator()(matmul_func func[][9][2]) const {
        if constexpr (s <= e) {
            func[s][1][0] = matmul_kbit_32<1, s, false>;
            func[s][1][1] = matmul_kbit_32<1, s, true>;
            func[s][2][0] = matmul_kbit_32<2, s, false>;
            func[s][3][0] = matmul_kbit_32<3, s, false>;
            func[s][4][0] = matmul_kbit_32<4, s, false>;
            func[s][5][0] = matmul_kbit_32<5, s, false>;
            func[s][6][0] = matmul_kbit_32<6, s, false>;
            func[s][7][0] = matmul_kbit_32<7, s, false>;
            func[s][8][0] = matmul_kbit_32<8, s, false>;
            get_matmul_func<s+1, e>()(func);
        }
    }
};

typedef void (* dequant_func) (
    const uint32_t *,
    const uint32_t, const uint32_t,
    const __half *, __half *
);

template <int s, int e>
struct get_dequant_func {
    void operator()(dequant_func func[]) const {
        if constexpr (s <= e) {
            func[s] = dequant_kbit_store<s>;
            get_dequant_func<s+1, e>()(func);
        }
    }
};

// Formula-based dequantization function pointer type
typedef void (* dequant_formula_func) (
    const uint32_t *,
    const uint32_t, const uint32_t,
    const __half *, const __half *, __half *
);

// Template to generate formula dequant function pointers for different bit widths and group sizes
template <int bits, int group_size>
struct get_dequant_formula_func_gs {
    void operator()(dequant_formula_func func[][5]) const {
        // group_size index: 0=32, 1=64, 2=128, 3=256, 4=512
        constexpr int gs_idx = (group_size == 32) ? 0 :
                               (group_size == 64) ? 1 :
                               (group_size == 128) ? 2 :
                               (group_size == 256) ? 3 : 4;
        func[bits][gs_idx] = dequant_formula_kbit_store_optimized<bits, group_size>;
    }
};

template <int s, int e>
struct get_dequant_formula_func {
    void operator()(dequant_formula_func func[][5]) const {
        if constexpr (s <= e) {
            get_dequant_formula_func_gs<s, 32>()(func);
            get_dequant_formula_func_gs<s, 64>()(func);
            get_dequant_formula_func_gs<s, 128>()(func);
            get_dequant_formula_func_gs<s, 256>()(func);
            get_dequant_formula_func_gs<s, 512>()(func);
            get_dequant_formula_func<s+1, e>()(func);
        }
    }
};

// Per-group matmul function pointer type
typedef void (* matmul_pergroup_func) (
    const __half *, const uint32_t *,
    const uint32_t, const uint32_t, const uint32_t,
    const __half *, const __half *, __half *
);

// Template to generate per-group matmul function pointers
template <int bits, int group_size>
struct get_matmul_pergroup_func_gs {
    void operator()(matmul_pergroup_func func[][5][9]) const {
        constexpr int gs_idx = (group_size == 32) ? 0 :
                               (group_size == 64) ? 1 :
                               (group_size == 128) ? 2 :
                               (group_size == 256) ? 3 : 4;
        func[bits][gs_idx][1] = matmul_kbit_32_pergroup<1, bits, group_size, false>;
        func[bits][gs_idx][2] = matmul_kbit_32_pergroup<2, bits, group_size, false>;
        func[bits][gs_idx][3] = matmul_kbit_32_pergroup<3, bits, group_size, false>;
        func[bits][gs_idx][4] = matmul_kbit_32_pergroup<4, bits, group_size, false>;
        func[bits][gs_idx][5] = matmul_kbit_32_pergroup<5, bits, group_size, false>;
        func[bits][gs_idx][6] = matmul_kbit_32_pergroup<6, bits, group_size, false>;
        func[bits][gs_idx][7] = matmul_kbit_32_pergroup<7, bits, group_size, false>;
        func[bits][gs_idx][8] = matmul_kbit_32_pergroup<8, bits, group_size, false>;
    }
};

template <int s, int e>
struct get_matmul_pergroup_func {
    void operator()(matmul_pergroup_func func[][5][9]) const {
        if constexpr (s <= e) {
            get_matmul_pergroup_func_gs<s, 32>()(func);
            get_matmul_pergroup_func_gs<s, 64>()(func);
            get_matmul_pergroup_func_gs<s, 128>()(func);
            get_matmul_pergroup_func_gs<s, 256>()(func);
            get_matmul_pergroup_func_gs<s, 512>()(func);
            get_matmul_pergroup_func<s+1, e>()(func);
        }
    }
};

bool dequant_initalized = false;
bool dequant_formula_initialized = false;
bool matmul_initialized = false;
bool matmul_pergroup_initialized = false;
bool is_orin = false;
matmul_func matmul_functions[9][9][2] = {NULL, };
matmul_pergroup_func matmul_pergroup_functions[9][5][9] = {NULL, };
dequant_func dequant_functions[9] = {NULL, };
dequant_formula_func dequant_formula_functions[9][5] = {NULL, };

torch::Tensor dequant_kbit(
    torch::Tensor qweight,
    torch::Tensor lut,
    int w_bits
) {
    // Set correct device
    HANDLE_ERROR(cudaSetDevice(qweight.device().index()));

    assert(qweight.ndimension() == 3 && qweight.dtype() == torch::kInt && lut.dtype() == torch::kHalf);
    assert(qweight.device() == lut.device() && qweight.is_cuda());
    assert(w_bits >= 3 && w_bits <= 8);
    const int N = qweight.size(1);
    const int K = qweight.size(2) * 32;

    if (!dequant_initalized) {
        get_dequant_func<3, 8>()(dequant_functions);
        dequant_initalized = true;
    }

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(qweight.device());
    at::Tensor weight = torch::empty({N, K}, options);

    dim3 grid(N/num_rows), block(32, num_rows);
    dequant_functions[w_bits]<<<grid, block>>>(
        (uint32_t *)qweight.data_ptr<int>(),
        N, K,
        (__half *)lut.data_ptr<at::Half>(),
        (__half *)weight.data_ptr<at::Half>()
    );

    return weight;
}


/**
 * Formula-based dequantization: w = scale * (w' - zero)
 * Supports per-group scale/zero with various group sizes.
 *
 * @param qweight   Quantized weights [max_bits, N, K/32]
 * @param scales    Per-group scales [N, K/group_size]
 * @param zeros     Per-group zeros [N, K/group_size]
 * @param w_bits    Number of bits (3-8)
 * @param group_size Group size (32, 64, 128, 256, or 512)
 * @return Dequantized weights [N, K]
 */
torch::Tensor dequant_formula_kbit(
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int w_bits,
    int group_size
) {
    // Set correct device
    HANDLE_ERROR(cudaSetDevice(qweight.device().index()));

    assert(qweight.ndimension() == 3 && qweight.dtype() == torch::kInt);
    assert(scales.dtype() == torch::kHalf && zeros.dtype() == torch::kHalf);
    assert(qweight.device() == scales.device() && qweight.device() == zeros.device() && qweight.is_cuda());
    assert(w_bits >= 3 && w_bits <= 8);
    assert(group_size == 32 || group_size == 64 || group_size == 128 || group_size == 256 || group_size == 512);

    const int N = qweight.size(1);
    const int K = qweight.size(2) * 32;

    // Validate scales/zeros shape
    assert(scales.size(0) == N && scales.size(1) == K / group_size);
    assert(zeros.size(0) == N && zeros.size(1) == K / group_size);

    if (!dequant_formula_initialized) {
        get_dequant_formula_func<3, 8>()(dequant_formula_functions);
        dequant_formula_initialized = true;
    }

    // Map group_size to index
    int gs_idx = (group_size == 32) ? 0 :
                 (group_size == 64) ? 1 :
                 (group_size == 128) ? 2 :
                 (group_size == 256) ? 3 : 4;

    auto options = torch::TensorOptions().dtype(torch::kHalf).device(qweight.device());
    at::Tensor weight = torch::empty({N, K}, options);

    dim3 grid(N/num_rows), block(32, num_rows);
    dequant_formula_functions[w_bits][gs_idx]<<<grid, block>>>(
        (uint32_t *)qweight.data_ptr<int>(),
        N, K,
        (__half *)scales.data_ptr<at::Half>(),
        (__half *)zeros.data_ptr<at::Half>(),
        (__half *)weight.data_ptr<at::Half>()
    );

    return weight;
}


torch::Tensor matmul_kbit(
    torch::Tensor in,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int w_bits
) {
    // Set correct device
    HANDLE_ERROR(cudaSetDevice(qweight.device().index()));

    const int N = qweight.size(1);
    const int K = qweight.size(2) * 32;
    int64_t in_ndim = in.ndimension();
    const int M = in.numel() / K;

    // TODO assert with size or dtype
    assert(M >= 1 && M <= 8 && w_bits >= 3 && w_bits <= 8);
    assert(in.device() == qweight.device() && in.device() == scales.device() && in.device() == zeros.device() && in.is_cuda());
    assert(qweight.ndimension() == 3 && qweight.dtype() == torch::kInt);
    assert(scales.dtype() == torch::kHalf && zeros.dtype() == torch::kHalf);
    assert(in.dtype() == torch::kHalf);

    if (!matmul_initialized) {
        int device;
        HANDLE_ERROR(cudaGetDevice(&device));
        cudaDeviceProp prop;
        HANDLE_ERROR(cudaGetDeviceProperties(&prop, device));
        is_orin = strcmp(prop.name, "Orin") == 0;

        get_matmul_func<3, 8>()(matmul_functions);
        matmul_initialized = true;
    }

    auto sizes = in.sizes().vec();
    sizes.at(in_ndim - 1) = N;
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(in.device());
    at::Tensor out = torch::empty(sizes, options);

    const int multi_row = (M == 1 ? 1 : 4);
    const int use_ksplit = !is_orin && M == 1 && K > 4096 && w_bits >= 7;
    const int num_ksplit = (use_ksplit ? DIV_ROUND_UP(K, 4096) : 1);

    dim3 grid(N/(num_rows*multi_row)), block(32, num_rows, num_ksplit);
    matmul_functions[w_bits][M][use_ksplit]<<<grid, block>>>(
        (__half *)in.data_ptr<at::Half>(),
        (uint32_t *)qweight.data_ptr<int>(),
        M, N, K,
        (__half *)scales.data_ptr<at::Half>(),
        (__half *)zeros.data_ptr<at::Half>(),
        (__half *)out.data_ptr<at::Half>()
    );

    return out;
}


/**
 * Per-group formula-based quantized matmul: O = I @ W^T
 * Where W is dequantized as: w = scale * (w' - zero)
 *
 * @param in         Input tensor [M, K] or [batch, M, K]
 * @param qweight    Quantized weights [max_bits, N, K/32]
 * @param scales     Per-group scales [N, K/group_size]
 * @param zeros      Per-group zeros [N, K/group_size]
 * @param w_bits     Number of bits (3-8)
 * @param group_size Group size (32, 64, 128, 256, or 512)
 * @return Output tensor [M, N] or [batch, M, N]
 */
torch::Tensor matmul_kbit_pergroup(
    torch::Tensor in,
    torch::Tensor qweight,
    torch::Tensor scales,
    torch::Tensor zeros,
    int w_bits,
    int group_size
) {
    // Set correct device
    HANDLE_ERROR(cudaSetDevice(qweight.device().index()));

    const int N = qweight.size(1);
    const int K = qweight.size(2) * 32;
    int64_t in_ndim = in.ndimension();
    const int M = in.numel() / K;

    assert(M >= 1 && M <= 8 && w_bits >= 3 && w_bits <= 8);
    assert(in.device() == qweight.device() && in.device() == scales.device() && in.device() == zeros.device() && in.is_cuda());
    assert(qweight.ndimension() == 3 && qweight.dtype() == torch::kInt);
    assert(scales.dtype() == torch::kHalf && zeros.dtype() == torch::kHalf);
    assert(in.dtype() == torch::kHalf);
    assert(group_size == 32 || group_size == 64 || group_size == 128 || group_size == 256 || group_size == 512);

    // Validate scales/zeros shape
    assert(scales.size(0) == N && scales.size(1) == K / group_size);
    assert(zeros.size(0) == N && zeros.size(1) == K / group_size);

    if (!matmul_pergroup_initialized) {
        get_matmul_pergroup_func<3, 8>()(matmul_pergroup_functions);
        matmul_pergroup_initialized = true;
    }

    // Map group_size to index
    int gs_idx = (group_size == 32) ? 0 :
                 (group_size == 64) ? 1 :
                 (group_size == 128) ? 2 :
                 (group_size == 256) ? 3 : 4;

    auto sizes = in.sizes().vec();
    sizes.at(in_ndim - 1) = N;
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(in.device());
    at::Tensor out = torch::empty(sizes, options);

    const int multi_row = (M == 1 ? 1 : 4);

    dim3 grid(N/(num_rows*multi_row)), block(32, num_rows);
    matmul_pergroup_functions[w_bits][gs_idx][M]<<<grid, block>>>(
        (__half *)in.data_ptr<at::Half>(),
        (uint32_t *)qweight.data_ptr<int>(),
        M, N, K,
        (__half *)scales.data_ptr<at::Half>(),
        (__half *)zeros.data_ptr<at::Half>(),
        (__half *)out.data_ptr<at::Half>()
    );

    return out;
}

PYBIND11_MODULE(any_precision_ext, m) {
    m.def("matmul_kbit", &matmul_kbit, "kbit quantized matmul (legacy per-row scale/zero)");
    m.def("matmul_kbit_pergroup", &matmul_kbit_pergroup, "kbit quantized matmul with per-group scale/zero");
    m.def("dequant_kbit", &dequant_kbit, "kbit dequantize function (LUT-based)");
    m.def("dequant_formula_kbit", &dequant_formula_kbit, "kbit dequantize function (Formula-based: w = s*(w'-z))");
}
