#!/usr/bin/env python3
"""
End-to-end test for APLinear CUDA integration.

This script verifies that the CUDA-accelerated APLinear produces
identical results to the Python fallback implementation.

Author: 浮浮酱 (Nekomata Engineer)
"""

import torch
import torch.nn as nn
import sys
import os
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_mock_aplinear(in_features, out_features, group_size, precisions, device):
    """
    Create a mock APLinear module with random quantized weights.
    """
    from any_precision.modules.APLinear import APLinear

    layer = APLinear(
        in_features=in_features,
        out_features=out_features,
        supported_bits=precisions,
        group_size=group_size,
        bias=True,
        precisions=precisions,
        device=device,
        layer_name="test_layer"
    )

    # Initialize with random data
    max_bits = max(precisions)
    in_chunks = in_features // 32
    num_groups = in_features // group_size

    # Random quantized weights in bit-plane format
    layer.qweight = torch.randint(
        0, 2**31,
        (max_bits, out_features, in_chunks),
        dtype=torch.int32,
        device=device
    )

    # Random scale and zero (per-group)
    for bit in precisions:
        layer._buffers[f'scale{bit}'] = torch.randn(
            out_features, num_groups,
            dtype=torch.float16,
            device=device
        ) * 0.1

    layer._buffers['zero'] = torch.randn(
        out_features, num_groups,
        dtype=torch.float16,
        device=device
    ) * 0.5

    # Random bias
    layer.bias = torch.randn(out_features, dtype=torch.float16, device=device) * 0.01

    return layer


def test_aplinear_cuda_vs_python(
    in_features=1024,
    out_features=256,
    group_size=128,
    w_bits=6,
    batch_size=1,
    seq_len=32,
    device="cuda:0"
):
    """
    Compare CUDA and Python implementations of APLinear.
    """
    print(f"\n{'='*60}")
    print(f"Testing APLinear: in={in_features}, out={out_features}")
    print(f"  group_size={group_size}, w_bits={w_bits}")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"{'='*60}")

    # Check CUDA availability
    from any_precision.modules.APLinear import CUDA_AVAILABLE
    print(f"  CUDA Extension Available: {CUDA_AVAILABLE}")

    if not CUDA_AVAILABLE:
        print("  Skipping test - CUDA extension not installed")
        return True

    precisions = list(range(4, 9))  # [4, 5, 6, 7, 8]

    # Create mock layer
    layer = create_mock_aplinear(in_features, out_features, group_size, precisions, device)
    layer.set_precision(w_bits)
    layer.eval()

    # Create random input
    x = torch.randn(batch_size, seq_len, in_features, dtype=torch.float16, device=device)

    # ========== Test 1: Force Python path ==========
    print("\n[1] Running Python implementation...")

    # Temporarily disable CUDA
    import any_precision.modules.APLinear as aplinear_module
    original_cuda_available = aplinear_module.CUDA_AVAILABLE
    aplinear_module.CUDA_AVAILABLE = False

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out_python = layer(x)
    torch.cuda.synchronize()
    python_time = time.time() - t0
    print(f"  Output shape: {out_python.shape}")
    print(f"  Time: {python_time*1000:.3f} ms")

    # ========== Test 2: CUDA path ==========
    print("\n[2] Running CUDA implementation...")

    # Re-enable CUDA
    aplinear_module.CUDA_AVAILABLE = original_cuda_available

    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        out_cuda = layer(x)
    torch.cuda.synchronize()
    cuda_time = time.time() - t0
    print(f"  Output shape: {out_cuda.shape}")
    print(f"  Time: {cuda_time*1000:.3f} ms")

    # ========== Compare results ==========
    print("\n[3] Comparing results...")

    if out_python.shape != out_cuda.shape:
        print(f"  ✗ Shape mismatch: Python {out_python.shape} vs CUDA {out_cuda.shape}")
        return False
    print(f"  ✓ Shapes match: {out_python.shape}")

    # Compare values
    python_fp32 = out_python.float()
    cuda_fp32 = out_cuda.float()

    abs_diff = (python_fp32 - cuda_fp32).abs()
    max_diff = abs_diff.max().item()
    mean_diff = abs_diff.mean().item()

    rtol = 1e-2  # Relaxed tolerance for FP16 accumulation differences
    atol = 1e-2

    is_close = torch.allclose(python_fp32, cuda_fp32, rtol=rtol, atol=atol)

    print(f"  Max absolute difference: {max_diff:.6e}")
    print(f"  Mean absolute difference: {mean_diff:.6e}")

    if is_close:
        print(f"  ✓ Results match within tolerance (rtol={rtol}, atol={atol})")
    else:
        print(f"  ✗ Results do NOT match within tolerance")

        # Find worst case
        diff_flat = abs_diff.flatten()
        worst_idx = diff_flat.argmax().item()
        print(f"\n  Worst case at flat index {worst_idx}:")
        print(f"    Python: {out_python.flatten()[worst_idx].item():.6f}")
        print(f"    CUDA:   {out_cuda.flatten()[worst_idx].item():.6f}")

    # ========== Performance summary ==========
    print(f"\n[4] Performance Summary")
    print(f"  Python: {python_time*1000:.3f} ms")
    print(f"  CUDA:   {cuda_time*1000:.3f} ms")
    if cuda_time > 0:
        speedup = python_time / cuda_time
        print(f"  Speedup: {speedup:.2f}x")

    return is_close


def benchmark_aplinear(
    in_features=2048,
    out_features=2048,
    group_size=128,
    w_bits=6,
    batch_size=1,
    seq_len=1,
    device="cuda:0",
    warmup=10,
    iterations=100
):
    """
    Benchmark APLinear CUDA vs Python performance.
    """
    print(f"\n{'='*60}")
    print(f"Benchmark APLinear: in={in_features}, out={out_features}")
    print(f"  group_size={group_size}, w_bits={w_bits}")
    print(f"  batch_size={batch_size}, seq_len={seq_len}")
    print(f"  warmup={warmup}, iterations={iterations}")
    print(f"{'='*60}")

    from any_precision.modules.APLinear import CUDA_AVAILABLE
    if not CUDA_AVAILABLE:
        print("  Skipping benchmark - CUDA extension not installed")
        return

    import any_precision.modules.APLinear as aplinear_module

    precisions = list(range(4, 9))
    layer = create_mock_aplinear(in_features, out_features, group_size, precisions, device)
    layer.set_precision(w_bits)
    layer.eval()

    x = torch.randn(batch_size, seq_len, in_features, dtype=torch.float16, device=device)

    # Warmup
    print(f"\n[Warmup] {warmup} iterations...")
    for _ in range(warmup):
        with torch.no_grad():
            _ = layer(x)
    torch.cuda.synchronize()

    # Benchmark CUDA
    print(f"\n[CUDA] {iterations} iterations...")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iterations):
        with torch.no_grad():
            _ = layer(x)
    end.record()
    torch.cuda.synchronize()
    cuda_time = start.elapsed_time(end) / iterations

    # Benchmark Python
    print(f"\n[Python] {iterations} iterations...")
    aplinear_module.CUDA_AVAILABLE = False

    start.record()
    for _ in range(iterations):
        with torch.no_grad():
            _ = layer(x)
    end.record()
    torch.cuda.synchronize()
    python_time = start.elapsed_time(end) / iterations

    aplinear_module.CUDA_AVAILABLE = True

    # Summary
    print(f"\n[Summary]")
    print(f"  Python: {python_time:.3f} ms")
    print(f"  CUDA:   {cuda_time:.3f} ms")
    print(f"  Speedup: {python_time/cuda_time:.2f}x")


def main():
    print("=" * 60)
    print("  APLinear CUDA Integration Test")
    print("  Formula-based dequantization: w = s * (w' - z)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        return 1

    device = "cuda:0"
    print(f"\nUsing device: {torch.cuda.get_device_name(device)}")

    # Test configurations
    test_configs = [
        # (in_features, out_features, group_size, w_bits, batch_size, seq_len)
        (1024, 256, 128, 6, 1, 32),
        (2048, 512, 128, 4, 1, 16),
        (2048, 2048, 64, 8, 1, 1),
        (4096, 4096, 128, 6, 1, 1),
    ]

    all_passed = True
    for config in test_configs:
        in_f, out_f, gs, wb, bs, sl = config
        passed = test_aplinear_cuda_vs_python(in_f, out_f, gs, wb, bs, sl, device)
        all_passed = all_passed and passed

    print("\n" + "=" * 60)
    if all_passed:
        print("  ALL TESTS PASSED! ✓")
    else:
        print("  SOME TESTS FAILED! ✗")
    print("=" * 60)

    # Run benchmark
    if all_passed:
        benchmark_aplinear(
            in_features=4096,
            out_features=4096,
            group_size=128,
            w_bits=6,
            batch_size=1,
            seq_len=1,
            device=device
        )

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
