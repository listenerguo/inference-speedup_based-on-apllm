#!/usr/bin/env python3
"""
Test script to verify CUDA dequant_formula_kbit against Python _dequant_temp.

This script validates that the new Formula-based CUDA kernel produces
identical results to the Python implementation.

Author: 浮浮酱 (Nekomata Engineer)
"""

import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from any_precision.modules.APLinear import restore_uint8_from_weighttensor_torch


def python_dequant(qweight, scale, zero, w_bits, group_size, min_bits=3):
    """
    Python reference implementation of formula-based dequantization.
    Matches APLinear._dequant_temp logic.

    IMPORTANT: This function expects the RAW zero (not pre-adjusted).
    It will apply bit error correction internally to match APLinear behavior.
    """
    _, out_features, in_chunks = qweight.shape
    in_features = in_chunks * 32
    qweight_sub = qweight[:w_bits]

    # Restore integer weights from bit-planes
    weight = restore_uint8_from_weighttensor_torch(qweight_sub, w_bits)
    weight_f = weight.to(torch.float16)

    # Apply bit error correction for zero point (matches APLinear.forward)
    bit_err = w_bits - min_bits
    zero_adjusted = zero * (2 ** bit_err)

    # Apply scale and zero
    scale_pc = scale.repeat_interleave(group_size, dim=1)
    zero_pc = zero_adjusted.repeat_interleave(group_size, dim=1)

    if scale_pc.shape[1] > in_features:
        scale_pc = scale_pc[:, :in_features]
        zero_pc = zero_pc[:, :in_features]

    out = scale_pc * (weight_f - zero_pc)
    return out


def test_dequant_formula(w_bits=6, group_size=128, N=256, K=1024, device="cuda:0"):
    """
    Test CUDA dequant_formula_kbit against Python implementation.

    Args:
        w_bits: Number of quantization bits (3-8)
        group_size: Group size for per-group quantization
        N: Number of output features
        K: Number of input features
        device: CUDA device
    """
    print(f"\n{'='*60}")
    print(f"Testing: w_bits={w_bits}, group_size={group_size}, N={N}, K={K}")
    print(f"{'='*60}")

    # Import CUDA extension
    try:
        from any_precision_ext import dequant_formula_kbit
        print("✓ CUDA extension loaded successfully")
    except ImportError as e:
        print(f"✗ Failed to import CUDA extension: {e}")
        print("  Please rebuild with: cd any_precision/modules/kernels && python setup.py install")
        return False

    # Create test data
    max_bits = 8
    in_chunks = K // 32
    num_groups = K // group_size

    # Random quantized weights in bit-plane format
    qweight = torch.randint(0, 2**31, (max_bits, N, in_chunks), dtype=torch.int32, device=device)

    # Random scale and zero (per-group)
    scale = torch.randn(N, num_groups, dtype=torch.float16, device=device) * 0.1
    zero = torch.randn(N, num_groups, dtype=torch.float16, device=device) * 0.5

    print(f"  qweight shape: {qweight.shape}")
    print(f"  scale shape: {scale.shape}")
    print(f"  zero shape: {zero.shape}")

    # Run Python implementation
    print("\n[1] Running Python implementation...")
    torch.cuda.synchronize()

    # Debug: Check intermediate values
    _, out_features, in_chunks = qweight.shape
    in_features = in_chunks * 32
    qweight_sub = qweight[:w_bits]
    weight = restore_uint8_from_weighttensor_torch(qweight_sub, w_bits)
    print(f"  Python weight range: min={weight.min().item()}, max={weight.max().item()}")

    min_bits = 3
    bit_err = w_bits - min_bits
    print(f"  Original zero range: min={zero.min().item():.6f}, max={zero.max().item():.6f}")
    zero_adjusted_dbg = zero * (2 ** bit_err)
    print(f"  Adjusted zero range: min={zero_adjusted_dbg.min().item():.6f}, max={zero_adjusted_dbg.max().item():.6f}")

    python_result = python_dequant(qweight, scale, zero, w_bits, group_size)
    torch.cuda.synchronize()
    print(f"  Result shape: {python_result.shape}")
    print(f"  Python result range: min={python_result.min().item():.6f}, max={python_result.max().item():.6f}")

    # Run CUDA implementation
    # IMPORTANT: CUDA kernel expects pre-adjusted zero (matches APLinear.forward behavior)
    print("\n[2] Running CUDA implementation...")
    zero_adjusted = zero * (2 ** bit_err)

    torch.cuda.synchronize()
    cuda_result = dequant_formula_kbit(qweight, scale, zero_adjusted, w_bits, group_size)
    torch.cuda.synchronize()
    print(f"  Result shape: {cuda_result.shape}")
    print(f"  CUDA result range: min={cuda_result.min().item():.6f}, max={cuda_result.max().item():.6f}")

    # Compare results
    print("\n[3] Comparing results...")

    # Check shapes match
    if python_result.shape != cuda_result.shape:
        print(f"✗ Shape mismatch: Python {python_result.shape} vs CUDA {cuda_result.shape}")
        return False
    print(f"  ✓ Shapes match: {python_result.shape}")

    # Check values match (with tolerance for FP16)
    python_fp32 = python_result.float()
    cuda_fp32 = cuda_result.float()

    abs_diff = (python_fp32 - cuda_fp32).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    # Calculate percentiles for better understanding of error distribution
    diff_flat = abs_diff.flatten()
    percentiles = [50, 75, 90, 95, 99, 99.9]
    print(f"\n  📊 Error Distribution Analysis:")
    print(f"    Max absolute difference:  {max_diff:.6e}")
    print(f"    Mean absolute difference: {mean_diff:.6e}")
    print(f"    Median (50th percentile): {torch.quantile(diff_flat, 0.5).item():.6e}")
    for p in [75, 90, 95, 99, 99.9]:
        val = torch.quantile(diff_flat, p/100).item()
        print(f"    {p}th percentile:          {val:.6e}")

    # Count how many values exceed different thresholds
    thresholds = [0.001, 0.01, 0.1, 1.0]
    print(f"\n  📈 Values exceeding thresholds:")
    total = diff_flat.numel()
    for thresh in thresholds:
        count = (diff_flat > thresh).sum().item()
        percentage = 100.0 * count / total
        print(f"    > {thresh:6.3f}: {count:8d} / {total} ({percentage:5.2f}%)")

    # Relative tolerance for FP16
    rtol = 1e-3
    atol = 1e-3

    is_close = torch.allclose(python_fp32, cuda_fp32, rtol=rtol, atol=atol)

    print(f"\n  🎯 Tolerance Check (rtol={rtol}, atol={atol}):")
    if is_close:
        print(f"    ✓ Results match within tolerance")
    else:
        print(f"    ✗ Results do NOT match within tolerance")

    # Find and analyze worst cases (top 5)
    print(f"\n  🔍 Top 5 Worst Cases:")
    _, worst_indices = torch.topk(diff_flat, min(5, diff_flat.numel()))

    for rank, idx in enumerate(worst_indices, 1):
        idx_val = idx.item()
        worst_row = idx_val // K
        worst_col = idx_val % K

        py_val = python_result[worst_row, worst_col].item()
        cu_val = cuda_result[worst_row, worst_col].item()
        diff_val = abs_diff[worst_row, worst_col].item()

        # Calculate which group this column belongs to
        group_idx = worst_col // group_size
        col_in_group = worst_col % group_size

        print(f"    #{rank}: Position [{worst_row}, {worst_col}] (group {group_idx}, col {col_in_group} in group)")
        print(f"        Python: {py_val:12.6f}")
        print(f"        CUDA:   {cu_val:12.6f}")
        print(f"        Diff:   {diff_val:12.6e}")

        # Show the quantized weight, scale, and zero for this position
        qweight_sub = qweight[:w_bits]
        weight_restored = restore_uint8_from_weighttensor_torch(qweight_sub, w_bits)
        qw_val = weight_restored[worst_row, worst_col].item()
        scale_val = scale[worst_row, group_idx].item()
        zero_val = zero[worst_row, group_idx].item()
        zero_adj_val = zero_adjusted_dbg[worst_row, group_idx].item()

        print(f"        Quantized weight: {qw_val}")
        print(f"        Scale: {scale_val:.6f}, Zero (raw): {zero_val:.6f}, Zero (adjusted): {zero_adj_val:.6f}")
        print(f"        Expected: scale * (qw - zero_adj) = {scale_val:.6f} * ({qw_val} - {zero_adj_val:.6f}) = {scale_val * (qw_val - zero_adj_val):.6f}")

    if is_close:
        return True
    else:
        return False


def benchmark_dequant(w_bits=6, group_size=128, N=2048, K=2048, device="cuda:0", warmup=10, iterations=100):
    """
    Benchmark CUDA vs Python dequantization performance.
    """
    print(f"\n{'='*60}")
    print(f"Benchmark: w_bits={w_bits}, group_size={group_size}, N={N}, K={K}")
    print(f"{'='*60}")

    try:
        from any_precision_ext import dequant_formula_kbit
    except ImportError as e:
        print(f"✗ Failed to import CUDA extension: {e}")
        return

    # Create test data
    max_bits = 8
    in_chunks = K // 32
    num_groups = K // group_size

    qweight = torch.randint(0, 2**31, (max_bits, N, in_chunks), dtype=torch.int32, device=device)
    scale = torch.randn(N, num_groups, dtype=torch.float16, device=device) * 0.1
    zero = torch.randn(N, num_groups, dtype=torch.float16, device=device) * 0.5

    # Apply bit error correction for CUDA kernel (matches APLinear.forward)
    min_bits = 3
    bit_err = w_bits - min_bits
    zero_adjusted = zero * (2 ** bit_err)

    # Warmup
    print(f"\n[Warmup] {warmup} iterations...")
    for _ in range(warmup):
        _ = python_dequant(qweight, scale, zero, w_bits, group_size)
        _ = dequant_formula_kbit(qweight, scale, zero_adjusted, w_bits, group_size)
    torch.cuda.synchronize()

    # Benchmark Python
    print(f"\n[Python] {iterations} iterations...")
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        _ = python_dequant(qweight, scale, zero, w_bits, group_size)
    end.record()
    torch.cuda.synchronize()
    python_time = start.elapsed_time(end) / iterations
    print(f"  Average time: {python_time:.3f} ms")

    # Benchmark CUDA
    print(f"\n[CUDA] {iterations} iterations...")
    torch.cuda.synchronize()

    start.record()
    for _ in range(iterations):
        _ = dequant_formula_kbit(qweight, scale, zero_adjusted, w_bits, group_size)
    end.record()
    torch.cuda.synchronize()
    cuda_time = start.elapsed_time(end) / iterations
    print(f"  Average time: {cuda_time:.3f} ms")

    # Summary
    speedup = python_time / cuda_time
    print(f"\n[Summary]")
    print(f"  Python: {python_time:.3f} ms")
    print(f"  CUDA:   {cuda_time:.3f} ms")
    print(f"  Speedup: {speedup:.2f}x")


def main():
    print("=" * 60)
    print("  Formula Dequantization Verification Test")
    print("  w = scale * (w' - zero)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return 1

    device = "cuda:0"
    print(f"\nUsing device: {torch.cuda.get_device_name(device)}")

    # Test various configurations
    test_configs = [
        # (w_bits, group_size, N, K)
        (4, 128, 256, 1024),
        (6, 128, 256, 2048),
        (8, 128, 512, 2048),
        (6, 64, 256, 1024),
        (6, 256, 256, 2048),
    ]

    all_passed = True
    for w_bits, group_size, N, K in test_configs:
        passed = test_dequant_formula(w_bits, group_size, N, K, device)
        all_passed = all_passed and passed

    print("\n" + "=" * 60)
    if all_passed:
        print("  ALL TESTS PASSED! ✓")
    else:
        print("  SOME TESTS FAILED! ✗")
    print("=" * 60)

    # Run benchmark if all tests passed
    if all_passed:
        benchmark_dequant(w_bits=6, group_size=128, N=2048, K=2048, device=device)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
