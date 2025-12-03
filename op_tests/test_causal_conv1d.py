#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import run_perftest, checkAllclose, benchmark
import pandas as pd
import argparse


def torch_causal_conv1d(x, weight, bias=None, use_silu=False):
    """
    PyTorch reference implementation of causal 1D convolution.

    Args:
        x: [batch, dim, seqlen]
        weight: [dim, width]
        bias: [dim] or None
        use_silu: bool

    Returns:
        out: [batch, dim, seqlen]
    """
    batch, dim, seqlen = x.shape
    width = weight.shape[1]

    # Pad the input on the left for causality
    # For width=4, we need 3 zeros on the left
    x_padded = F.pad(x, (width - 1, 0), mode='constant', value=0)

    # Perform 1D convolution
    # Reshape weight from [dim, width] to [dim, 1, width] for conv1d
    weight_reshaped = weight.unsqueeze(1)  # [dim, 1, width]

    # Apply grouped convolution (each channel has its own filter)
    out = F.conv1d(x_padded, weight_reshaped, bias=bias, groups=dim)

    # Apply SiLU if requested
    if use_silu:
        out = F.silu(out)

    return out


@benchmark()
def test_causal_conv1d_fwd(batch, dim, seqlen, width, dtype, use_silu=False):
    """
    Test causal conv1d forward pass.

    Args:
        batch: batch size
        dim: number of channels
        seqlen: sequence length
        width: kernel width (2, 3, or 4)
        dtype: data type (fp16, bf16, fp32)
        use_silu: whether to apply SiLU activation
    """
    ret = {}

    # Create input tensors
    x = torch.randn(batch, dim, seqlen, dtype=dtype, device="cuda")
    weight = torch.randn(dim, width, dtype=dtype, device="cuda")
    bias = torch.randn(dim, dtype=dtype, device="cuda")
    out = torch.empty(batch, dim, seqlen, dtype=dtype, device="cuda")

    # Compute reference using PyTorch
    ref = torch_causal_conv1d(x, weight, bias, use_silu)

    # Run AIter implementation
    _, us_aiter = run_perftest(
        aiter.causal_conv1d_fwd,
        out,
        x,
        weight,
        bias,
        use_silu,
    )

    # Check accuracy
    err = checkAllclose(ref, out, rtol=1e-3, atol=1e-3)

    # Calculate performance metrics
    # Data movement: read x, weight, bias; write out
    bytes_read = x.nbytes + weight.nbytes + bias.nbytes
    bytes_write = out.nbytes
    total_bytes = bytes_read + bytes_write

    ret["us"] = us_aiter
    ret["TB/s"] = total_bytes / us_aiter / 1e6
    ret["RD TB/s"] = bytes_read / us_aiter / 1e6
    ret["WR TB/s"] = bytes_write / us_aiter / 1e6
    ret["err"] = err
    ret["GFLOPS"] = (batch * dim * seqlen * width * 2) / (us_aiter * 1e3)  # 2 ops per MAC

    return ret


@benchmark()
def test_causal_conv1d_no_bias(batch, dim, seqlen, width, dtype, use_silu=False):
    """Test causal conv1d without bias."""
    ret = {}

    x = torch.randn(batch, dim, seqlen, dtype=dtype, device="cuda")
    weight = torch.randn(dim, width, dtype=dtype, device="cuda")
    bias = torch.empty(0, dtype=dtype, device="cuda")  # Empty tensor
    out = torch.empty(batch, dim, seqlen, dtype=dtype, device="cuda")

    ref = torch_causal_conv1d(x, weight, None, use_silu)

    _, us_aiter = run_perftest(
        aiter.causal_conv1d_fwd,
        out,
        x,
        weight,
        bias,
        use_silu,
    )

    err = checkAllclose(ref, out, rtol=1e-3, atol=1e-3)

    bytes_read = x.nbytes + weight.nbytes
    bytes_write = out.nbytes
    total_bytes = bytes_read + bytes_write

    ret["us"] = us_aiter
    ret["TB/s"] = total_bytes / us_aiter / 1e6
    ret["err"] = err
    ret["GFLOPS"] = (batch * dim * seqlen * width * 2) / (us_aiter * 1e3)

    return ret


# Test configurations (默认使用小规模配置快速验证)
l_dtype = ["float32"]
l_batch = [2, 4]
l_dim = [64, 256]
l_seqlen = [2048]
l_width = [4]
l_use_silu = [False, True]  # 测试两种场景: 无激活 和 SiLU激活

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="Causal Conv1D test configuration",
)
parser.add_argument("--batch", type=int, default=None, help="Batch size")
parser.add_argument("--dim", type=int, default=None, help="Number of channels")
parser.add_argument("--seqlen", type=int, default=None, help="Sequence length")
parser.add_argument("--width", type=int, default=None, help="Kernel width (2, 3, or 4)")
parser.add_argument("--dtype", type=str, default=None, help="Data type: float16, bfloat16, float32")
parser.add_argument("--use_silu", action="store_true", help="Use SiLU activation (if specified, only test with SiLU=True)")
parser.add_argument("--no_silu", action="store_true", help="Don't use SiLU activation (if specified, only test with SiLU=False)")
parser.add_argument("--no_bias", action="store_true", help="Test without bias")
parser.add_argument("--verbose", action="store_true", help="Verbose output")


def main():
    args = parser.parse_args()

    # Override default lists if specific values provided
    batches = [args.batch] if args.batch else l_batch
    dims = [args.dim] if args.dim else l_dim
    seqlens = [args.seqlen] if args.seqlen else l_seqlen
    widths = [args.width] if args.width else l_width
    dtypes = [args.dtype] if args.dtype else l_dtype

    # Handle use_silu: if --use_silu specified, only test True; if --no_silu, only test False; otherwise test both
    if args.use_silu:
        use_silus = [True]
    elif args.no_silu:
        use_silus = [False]
    else:
        use_silus = l_use_silu  # Test both by default

    results = []

    print("=" * 80)
    print("Testing AIter Causal Conv1D Implementation")
    print("=" * 80)
    print(f"SiLU scenarios to test: {use_silus}")
    print("=" * 80)

    for dtype_str in dtypes:
        dtype = getattr(torch, dtype_str)

        for batch in batches:
            for dim in dims:
                for seqlen in seqlens:
                    for width in widths:
                        for use_silu in use_silus:  # Add loop for use_silu
                            test_name = f"B{batch}_D{dim}_S{seqlen}_W{width}_{dtype_str}"
                            if use_silu:
                                test_name += "_silu"
                            if args.no_bias:
                                test_name += "_nobias"

                            print(f"\n{'='*60}")
                            print(f"Test: {test_name}")
                            print(f"{'='*60}")

                            try:
                                if args.no_bias:
                                    result = test_causal_conv1d_no_bias(
                                        batch, dim, seqlen, width, dtype, use_silu
                                    )
                                else:
                                    result = test_causal_conv1d_fwd(
                                        batch, dim, seqlen, width, dtype, use_silu
                                    )

                                result["test"] = test_name
                                result["batch"] = batch
                                result["dim"] = dim
                                result["seqlen"] = seqlen
                                result["width"] = width
                                result["dtype"] = dtype_str
                                result["use_silu"] = use_silu
                                result["no_bias"] = args.no_bias

                                results.append(result)

                                print(f"✅ Time: {result['us']:.2f} us")
                                print(f"✅ Bandwidth: {result['TB/s']:.2f} TB/s")
                                print(f"✅ GFLOPS: {result['GFLOPS']:.2f}")
                                print(f"✅ Error: {result['err']:.2e}")

                            except Exception as e:
                                print(f"❌ Failed: {e}")
                                if args.verbose:
                                    import traceback
                                    traceback.print_exc()

    # Create summary DataFrame
    if results:
        df = pd.DataFrame(results)
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(df.to_string(index=False))

        # Save to CSV
        output_file = "causal_conv1d_test_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")

        # Print statistics
        print("\n" + "=" * 80)
        print("STATISTICS")
        print("=" * 80)
        print(f"Total tests: {len(results)}")
        print(f"Average time: {df['us'].mean():.2f} us")
        print(f"Average bandwidth: {df['TB/s'].mean():.2f} TB/s")
        print(f"Average GFLOPS: {df['GFLOPS'].mean():.2f}")
        print(f"Max error: {df['err'].max():.2e}")
        print(f"Tests passed (err < 1e-2): {(df['err'] < 1e-2).sum()}/{len(results)}")


if __name__ == "__main__":
    main()

