#pragma once

#include <cuda_fp16.h>
#include "dequant.cuh"

/* warp-wide sum with tree-reduction */
__device__ __forceinline__ __half warp_reduce_sum(
    __half sum
) {
    #pragma unroll
    for (int i = 4; i >= 0; i--)
        sum = __hadd(sum, __shfl_down_sync(0xffffffff, sum, 1<<i));
    return sum;
}

/**
 * Formula-based quantized matrix multiplication with per-group scale/zero support.
 *
 * Implements: O = I @ W^T, where W is dequantized as w = scale * (w' - zero)
 *
 * This version supports per-group scale/zero when group_size >= 32.
 * Each thread processes 32 columns, and if group_size >= 32, all 32 columns
 * share the same scale/zero parameters.
 *
 * @tparam maxm      Maximum batch size (1-8)
 * @tparam bits      Number of quantization bits (3-8)
 * @tparam group_size Number of columns per group (must be >= 32)
 * @tparam use_ksplit Whether to use K-dimension splitting for large K
 */
template <int maxm, int bits, int group_size, bool use_ksplit>
__global__ void matmul_kbit_32_pergroup(
    const __half * I, const uint32_t * W,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const __half * scales, const __half * zeros, __half * O
) {
    static_assert(maxm >= 1 && bits >= 3 && bits <= 8);
    static_assert(group_size >= 32, "group_size must be >= 32 for this kernel");
    static_assert(!use_ksplit || maxm == 1);
    constexpr int multi_row = (maxm == 1 ? 1 : 4);

    constexpr int warp_size = 32;
    constexpr int q_w_siz = 8;

    const uint32_t row_idx_base = blockIdx.x * num_rows * multi_row + threadIdx.y;
    const uint32_t num_groups = K / group_size;

    int eff_warp_size = warp_size;
    __half partial_sum[maxm * multi_row] = {__float2half(0.0), };
    uint32_t q[bits], q_w[q_w_siz];
    __half2 dq_w[16];

    int mini = (use_ksplit ? threadIdx.z * 4 : 0);
    int maxi = DIV_ROUND_UP(K, 32 * warp_size);
    if (use_ksplit && maxi > mini + 4) maxi = mini + 4;
    for (int i = mini; i < maxi; i++) {
        if (i == K / (32 * warp_size)) {
            eff_warp_size = (K % (32 * warp_size)) / 32;
            if (threadIdx.x >= eff_warp_size) break;
        }

        // Calculate column base for this thread's 32 weights
        const int col_base = i * warp_size * 32 + threadIdx.x * 32;
        const int g_idx = col_base / group_size;

        #pragma unroll
        for (int h = 0; h < multi_row; h++) {
            const uint32_t row_idx = row_idx_base + h * num_rows;

            // Load per-group scale and zero point using __ldg for cache optimization
            const __half scale = __ldg(&scales[row_idx * num_groups + g_idx]);
            const __half zero = __ldg(&zeros[row_idx * num_groups + g_idx]);
            const __half2 scale2 = __half2half2(scale);
            const __half2 zero2 = __half2half2(zero);

            // load quantized weight from bit-planes
            #pragma unroll
            for (int j = 0; j < bits; j++) {
                const int k = (j * N + row_idx) * (K / 32) + i * 32 + threadIdx.x;
                q[j] = W[k];
            }

            // Unpack bit-planes to integer indices
            dequant<bits, false>(q, q_w);

            // Formula dequantization: w = scale * (w' - zero)
            #pragma unroll
            for (int j = 3; j >= 0; j--) {
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    const __half x = __int2half_rn(q_w[k*2+0] & 0xff);
                    const __half y = __int2half_rn(q_w[k*2+1] & 0xff);
                    __half2 indices = make_half2(x, y);
                    dq_w[j * 4 + k] = __hmul2(scale2, __hsub2(indices, zero2));
                }
                #pragma unroll
                for (int k = 0; k < q_w_siz; k++)
                    q_w[k] >>= 8;
            }

            // accumulate: partial_sum += dq_w * input
            #pragma unroll
            for (int l = 0; l < maxm; l++) {
                __half2 sum = make_half2(__float2half(0.0), __float2half(0.0));
                #pragma unroll
                for (int j = 3; j >= 0; j--) {
                    const int idx = (l*K/8 + eff_warp_size*j) + i*warp_size*4 + threadIdx.x;
                    float4 in_buf = ((float4 *)I)[idx];
                    __half2 * in_half = (__half2 *)&in_buf;
                    #pragma unroll
                    for (int k = 0; k < 4; k++)
                        sum = __hfma2(dq_w[j * 4 + k], in_half[k], sum);
                }
                partial_sum[l + h * maxm] = __hadd(partial_sum[l + h * maxm], __hadd(sum.x, sum.y));
            }
        }
    }

    // Warp-wide reduction
    #pragma unroll
    for (int i = 0; i < maxm * multi_row; i++)
        partial_sum[i] = warp_reduce_sum(partial_sum[i]);

    // K-split reduction (if enabled)
    if constexpr (use_ksplit) {
        __shared__ __half shO[maxm * multi_row * num_rows];
        if (threadIdx.x == 0 && threadIdx.z == 0)
            #pragma unroll
            for (int j = 0; j < multi_row; j++)
                shO[j + threadIdx.y * multi_row] = __float2half(0.0);
        __syncthreads();
        if (threadIdx.x == 0)
            #pragma unroll
            for (int j = 0; j < multi_row; j++)
                atomicAdd(shO + j + threadIdx.y * multi_row, partial_sum[j]);
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.z == 0)
            #pragma unroll
            for (int j = 0; j < multi_row; j++)
                partial_sum[j] = shO[j + threadIdx.y * multi_row];
    }

    // Write output
    if (threadIdx.x == 0 && (!use_ksplit || threadIdx.z == 0)) {
        #pragma unroll
        for (int i = 0; i < maxm; i++) {
            #pragma unroll
            for (int j = 0; j < multi_row; j++) {
                const uint32_t row_idx = row_idx_base + j * num_rows;
                O[i * N + row_idx] = partial_sum[i + j * maxm];
            }
        }
    }
}

/**
 * Legacy per-row scale/zero kernel (kept for backward compatibility).
 * Use matmul_kbit_32_pergroup for per-group support.
 */
template <int maxm, int bits, bool use_ksplit>
__global__ void matmul_kbit_32(
    const __half * I, const uint32_t * W,
    const uint32_t M, const uint32_t N, const uint32_t K,
    const __half * scales, const __half * zeros, __half * O
) {
    static_assert(maxm >= 1 && bits >= 3 && bits <= 8);
    static_assert(!use_ksplit || maxm == 1);
    constexpr int multi_row = (maxm == 1 ? 1 : 4);

    constexpr int warp_size = 32;
    constexpr int q_w_siz = 8;

    const uint32_t row_idx_base = blockIdx.x * num_rows * multi_row + threadIdx.y;

    int eff_warp_size = warp_size;
    __half partial_sum[maxm * multi_row] = {__float2half(0.0), };
    uint32_t q[bits], q_w[q_w_siz];
    __half2 dq_w[16];

    int mini = (use_ksplit ? threadIdx.z * 4 : 0);
    int maxi = DIV_ROUND_UP(K, 32 * warp_size);
    if (use_ksplit && maxi > mini + 4) maxi = mini + 4;
    for (int i = mini; i < maxi; i++) {
        if (i == K / (32 * warp_size)) {
            eff_warp_size = (K % (32 * warp_size)) / 32;
            if (threadIdx.x >= eff_warp_size) break;
        }

        #pragma unroll
        for (int h = 0; h < multi_row; h++) {
            const uint32_t row_idx = row_idx_base + h * num_rows;

            // Load per-row scale and zero point (legacy behavior)
            const __half scale = scales[row_idx];
            const __half zero = zeros[row_idx];
            const __half2 scale2 = __half2half2(scale);
            const __half2 zero2 = __half2half2(zero);

            // load quantized weight
            #pragma unroll
            for (int j = 0; j < bits; j++) {
                const int k = (j * N + row_idx) * (K / 32) + i * 32 + threadIdx.x;
                q[j] = W[k];
            }

            // dequantize
            dequant<bits, false>(q, q_w);

            // linear dequantization: w = s * (w' - z)
            #pragma unroll
            for (int j = 3; j >= 0; j--) {
                #pragma unroll
                for (int k = 0; k < 4; k++) {
                    const __half x = __int2half_rn(q_w[k*2+0] & 0xff);
                    const __half y = __int2half_rn(q_w[k*2+1] & 0xff);
                    __half2 indices = make_half2(x, y);
                    dq_w[j * 4 + k] = __hmul2(scale2, __hsub2(indices, zero2));
                }
                #pragma unroll
                for (int k = 0; k < q_w_siz; k++)
                    q_w[k] >>= 8;
            }

            // accumulate
            #pragma unroll
            for (int l = 0; l < maxm; l++) {
                __half2 sum = make_half2(__float2half(0.0), __float2half(0.0));
                #pragma unroll
                for (int j = 3; j >= 0; j--) {
                    const int idx = (l*K/8 + eff_warp_size*j) + i*warp_size*4 + threadIdx.x;
                    float4 in_buf = ((float4 *)I)[idx];
                    __half2 * in_half = (__half2 *)&in_buf;
                    #pragma unroll
                    for (int k = 0; k < 4; k++)
                        sum = __hfma2(dq_w[j * 4 + k], in_half[k], sum);
                }
                partial_sum[l + h * maxm] = __hadd(partial_sum[l + h * maxm], __hadd(sum.x, sum.y));
            }
        }
    }

    #pragma unroll
    for (int i = 0; i < maxm * multi_row; i++)
        partial_sum[i] = warp_reduce_sum(partial_sum[i]);

    if constexpr (use_ksplit) {
        __shared__ __half shO[maxm * multi_row * num_rows];
        if (threadIdx.x == 0 && threadIdx.z == 0)
            #pragma unroll
            for (int j = 0; j < multi_row; j++)
                shO[j + threadIdx.y * multi_row] = __float2half(0.0);
        __syncthreads();
        if (threadIdx.x == 0)
            #pragma unroll
            for (int j = 0; j < multi_row; j++)
                atomicAdd(shO + j + threadIdx.y * multi_row, partial_sum[j]);
        __syncthreads();
        if (threadIdx.x == 0 && threadIdx.z == 0)
            #pragma unroll
            for (int j = 0; j < multi_row; j++)
                partial_sum[j] = shO[j + threadIdx.y * multi_row];
    }

    if (threadIdx.x == 0 && (!use_ksplit || threadIdx.z == 0)) {
        #pragma unroll
        for (int i = 0; i < maxm; i++) {
            #pragma unroll
            for (int j = 0; j < multi_row; j++) {
                const uint32_t row_idx = row_idx_base + j * num_rows;
                O[i * N + row_idx] = partial_sum[i + j * maxm];
            }
        }
    }
}
