#pragma once

#include <cuda_fp16.h>
#include "dequant.cuh"

/**
 * dequant_formula.cuh - Formula-based Dequantization Kernel
 *
 * Implements: w = scale * (w' - zero)
 * where w' is the unpacked integer weight from bit-planes.
 *
 * This replaces the LUT-based dequantization with per-group scale/zero support.
 *
 * Author: 浮浮酱 (Nekomata Engineer)
 */

/**
 * Formula-based dequantization kernel that stores results to global memory.
 * Used for standalone verification against Python implementation.
 *
 * @tparam bits      Number of bits for quantization (3-8)
 * @tparam group_size Number of columns sharing the same scale/zero
 *
 * @param W          Quantized weights in bit-plane format [bits, N, K/32]
 * @param N          Number of output features (rows)
 * @param K          Number of input features (columns)
 * @param scales     Per-group scale factors [N, K/group_size]
 * @param zeros      Per-group zero points [N, K/group_size]
 * @param O          Output dequantized weights [N, K]
 */
template <int bits, int group_size>
__global__ void dequant_formula_kbit_store(
    const uint32_t * __restrict__ W,
    const uint32_t N, const uint32_t K,
    const __half * __restrict__ scales,
    const __half * __restrict__ zeros,
    __half * __restrict__ O
) {
    static_assert(bits >= 3 && bits <= 8, "bits must be between 3 and 8");
    constexpr int warp_size = 32;

    const uint32_t row_idx = blockIdx.x * num_rows + threadIdx.y;
    const uint32_t num_groups = K / group_size;

    // No shared memory needed for LUT anymore!

    int eff_warp_size = warp_size;
    uint32_t q[bits], q_w[8];
    half2 dq_w[16];

    const uint32_t maxi = DIV_ROUND_UP(K, 32 * warp_size);
    for (int i = 0; i < maxi; i++) {
        if (i == K / (32 * warp_size)) {
            eff_warp_size = (K % (32 * warp_size)) / 32;
            if (threadIdx.x >= eff_warp_size) break;
        }

        // Calculate base column index for this iteration
        // Each thread processes 32 weights, but they are interleaved across warps
        // Thread i handles columns: [i*8, i*8+8), [i*8+256, i*8+264), etc.
        const int col_base = i * warp_size * 32 + threadIdx.x * 8;

        // load quantized weight from bit-planes
        #pragma unroll
        for (int j = 0; j < bits; j++) {
            const int k = (j * N + row_idx) * (K / 32) + i * 32 + threadIdx.x;
            q[j] = W[k];
        }

        // Unpack bit-planes to integer indices
        dequant<bits, false>(q, q_w);

        // Formula dequantization: w = scale * (w' - zero)
        // Process 32 weights (4 bytes per q_w element, 8 elements, each byte = 1 weight)
        // The loop processes in reverse order for correct output ordering
        #pragma unroll
        for (int j = 3; j >= 0; j--) {
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                // Extract two 8-bit indices
                const uint8_t idx0 = q_w[k*2+0] & 0xff;
                const uint8_t idx1 = q_w[k*2+1] & 0xff;

                // Calculate column indices for these two weights
                // j corresponds to byte position in q_w (j=3 for highest byte, j=0 for lowest)
                // Due to interleaved storage: j=0 -> cols [0,8), j=1 -> cols [256,264), etc.
                const int col_offset = j * 8 * eff_warp_size + k * 2;
                const int col0 = col_base + col_offset;
                const int col1 = col_base + col_offset + 1;

                // Calculate group indices
                const int g_idx0 = col0 / group_size;
                const int g_idx1 = col1 / group_size;

                // Load scale and zero using __ldg for read-only cache optimization
                const __half s0 = __ldg(&scales[row_idx * num_groups + g_idx0]);
                const __half z0 = __ldg(&zeros[row_idx * num_groups + g_idx0]);
                const __half s1 = __ldg(&scales[row_idx * num_groups + g_idx1]);
                const __half z1 = __ldg(&zeros[row_idx * num_groups + g_idx1]);

                // Convert indices to half and apply formula
                const __half w0 = __hmul(s0, __hsub(__int2half_rn(idx0), z0));
                const __half w1 = __hmul(s1, __hsub(__int2half_rn(idx1), z1));

                dq_w[j * 4 + k] = make_half2(w0, w1);
            }
            // Shift to next byte in each q_w element
            #pragma unroll
            for (int k = 0; k < 8; k++)
                q_w[k] >>= 8;
        }

        // Write dequantized weights to global memory
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int out_idx = (row_idx * K + 8 * eff_warp_size * j + i * warp_size * 32 + 8 * threadIdx.x) / 8;
            ((float4 *)O)[out_idx] = ((float4 *)dq_w)[j];
        }
    }
}

/**
 * Optimized version for per-group dequantization.
 * Correctly handles interleaved storage pattern where weights from a single thread
 * may span multiple groups depending on group_size.
 */
template <int bits, int group_size>
__global__ void dequant_formula_kbit_store_optimized(
    const uint32_t * __restrict__ W,
    const uint32_t N, const uint32_t K,
    const __half * __restrict__ scales,
    const __half * __restrict__ zeros,
    __half * __restrict__ O
) {
    static_assert(bits >= 3 && bits <= 8, "bits must be between 3 and 8");
    static_assert(group_size >= 32, "This optimized kernel requires group_size >= 32");
    constexpr int warp_size = 32;

    const uint32_t row_idx = blockIdx.x * num_rows + threadIdx.y;
    const uint32_t num_groups = K / group_size;

    int eff_warp_size = warp_size;
    uint32_t q[bits], q_w[8];
    half2 dq_w[16];

    const uint32_t maxi = DIV_ROUND_UP(K, 32 * warp_size);
    for (int i = 0; i < maxi; i++) {
        if (i == K / (32 * warp_size)) {
            eff_warp_size = (K % (32 * warp_size)) / 32;
            if (threadIdx.x >= eff_warp_size) break;
        }

        // load quantized weight from bit-planes
        #pragma unroll
        for (int j = 0; j < bits; j++) {
            const int k = (j * N + row_idx) * (K / 32) + i * 32 + threadIdx.x;
            q[j] = W[k];
        }

        // Unpack bit-planes
        dequant<bits, false>(q, q_w);

        // Formula dequantization with correct column index calculation
        // Each thread processes 32 weights stored in interleaved pattern:
        // - q_w[0-1] bytes j=3: columns [col_base + 24*eff_warp_size, col_base + 24*eff_warp_size + 8)
        // - q_w[0-1] bytes j=2: columns [col_base + 16*eff_warp_size, col_base + 16*eff_warp_size + 8)
        // - q_w[0-1] bytes j=1: columns [col_base + 8*eff_warp_size, col_base + 8*eff_warp_size + 8)
        // - q_w[0-1] bytes j=0: columns [col_base, col_base + 8)
        const int col_base = i * warp_size * 32 + threadIdx.x * 8;

        #pragma unroll
        for (int j = 3; j >= 0; j--) {
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                // Extract two 8-bit indices
                const uint8_t idx0 = q_w[k*2+0] & 0xff;
                const uint8_t idx1 = q_w[k*2+1] & 0xff;

                // Calculate actual column indices (interleaved pattern)
                const int col_offset = j * 8 * eff_warp_size + k * 2;
                const int col0 = col_base + col_offset;
                const int col1 = col_base + col_offset + 1;

                // Calculate group indices for each weight
                const int g_idx0 = col0 / group_size;
                const int g_idx1 = col1 / group_size;

                // Load scale and zero for each weight separately
                const __half s0 = __ldg(&scales[row_idx * num_groups + g_idx0]);
                const __half z0 = __ldg(&zeros[row_idx * num_groups + g_idx0]);
                const __half s1 = __ldg(&scales[row_idx * num_groups + g_idx1]);
                const __half z1 = __ldg(&zeros[row_idx * num_groups + g_idx1]);

                // Apply formula: w = scale * (w' - zero)
                const __half w0 = __hmul(s0, __hsub(__int2half_rn(idx0), z0));
                const __half w1 = __hmul(s1, __hsub(__int2half_rn(idx1), z1));

                dq_w[j * 4 + k] = make_half2(w0, w1);
            }
            // Shift to next byte in each q_w element
            #pragma unroll
            for (int k = 0; k < 8; k++)
                q_w[k] >>= 8;
        }

        // Write to global memory
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const int out_idx = (row_idx * K + 8 * eff_warp_size * j + i * warp_size * 32 + 8 * threadIdx.x) / 8;
            ((float4 *)O)[out_idx] = ((float4 *)dq_w)[j];
        }
    }
}
