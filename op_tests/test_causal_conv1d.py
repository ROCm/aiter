#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from aiter.test_common import run_perftest, checkAllclose, benchmark
import pandas as pd
import argparse


def torch_causal_conv1d(x, weight, bias=None, use_silu=False, initial_states=None, seq_idx=None):
    """
    PyTorch reference implementation of causal 1D convolution.

    Args:
        x: [batch, dim, seqlen]
        weight: [dim, width]
        bias: [dim] or None
        use_silu: bool
        initial_states: [batch, dim, width-1] or None
        seq_idx: [batch, seqlen] or None (for handling sub-sequences)

    Returns:
        out: [batch, dim, seqlen]
        final_states: [batch, dim, width-1] if initial_states provided, else None
    """
    # Save original dtype and convert x to weight's dtype (like original causal-conv1d)
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    if initial_states is not None:
        initial_states = initial_states.to(weight.dtype)
    
    batch, dim, seqlen = x.shape
    width = weight.shape[1]

    # Handle initial states
    if initial_states is not None:
        # Concatenate initial states with input
        x_with_states = torch.cat([initial_states, x], dim=-1)
        # Perform convolution without padding
        weight_reshaped = weight.unsqueeze(1)  # [dim, 1, width]
        out = F.conv1d(x_with_states, weight_reshaped, bias=bias, groups=dim)
        # Extract the output corresponding to input sequence
        out = out[..., :seqlen]
        # Extract final states (last width-1 elements from x_with_states)
        final_states = x_with_states[..., -(width-1):]
    else:
        # Pad the input on the left for causality
        x_padded = F.pad(x, (width - 1, 0), mode='constant', value=0)
        # Perform 1D convolution
        weight_reshaped = weight.unsqueeze(1)  # [dim, 1, width]
        out = F.conv1d(x_padded, weight_reshaped, bias=bias, groups=dim)
        final_states = None

    # Handle seq_idx (mask out padding tokens)
    if seq_idx is not None:
        # seq_idx < 0 indicates padding tokens
        mask = (seq_idx >= 0).unsqueeze(1).float()  # [batch, 1, seqlen]
        out = out * mask

    # Apply SiLU if requested
    if use_silu:
        out = F.silu(out)

    # Convert back to original dtype
    out = out.to(dtype_in)
    if final_states is not None:
        final_states = final_states.to(dtype_in)

    if initial_states is not None:
        return out, final_states
    else:
        return out


@benchmark()
def test_causal_conv1d_fn(batch, dim, seqlen, width, dtype, use_silu=False, is_channel_last=False, 
                           test_seq_idx=False, test_states=False):
    """
    Test causal conv1d forward pass.

    Args:
        batch: batch size
        dim: number of channels
        seqlen: sequence length
        width: kernel width (2, 3, or 4)
        dtype: data type (fp16, bf16, fp32)
        use_silu: whether to apply SiLU activation
        is_channel_last: whether to use channel-last layout
        test_seq_idx: whether to test seq_idx parameter (only for channel-last)
        test_states: whether to test initial_states and final_states (only for channel-last)
    """
    ret = {}

    # Create input tensors
    x = torch.randn(batch, dim, seqlen, dtype=dtype, device="cuda")
    # Use fp32 for weight and bias to match original causal-conv1d implementation
    # This will automatically upcast computation to fp32 for bf16/fp16 inputs
    weight = torch.randn(dim, width, dtype=torch.float32, device="cuda")
    bias = torch.randn(dim, dtype=torch.float32, device="cuda")
    
    # Optional parameters (only valid for channel-last)
    seq_idx = None
    initial_states = None
    final_states_out = None
    
    if is_channel_last:
        # Validate dim % 8 == 0 for channel-last
        if dim % 8 != 0:
            raise ValueError(f"Channel-last layout requires dim % 8 == 0, got dim={dim}")
        
        # Test seq_idx if requested
        if test_seq_idx:
            # Create seq_idx: 0 for valid tokens, -1 for padding
            # Let's make first half valid, second half padding for demo
            seq_idx = torch.zeros(batch, seqlen, dtype=torch.int32, device="cuda")
            # Optionally add some padding (set to -1)
            # seq_idx[:, seqlen//2:] = -1
        
        # Test states if requested
        if test_states:
            initial_states = torch.randn(batch, dim, width - 1, dtype=dtype, device="cuda")
            # Ensure initial_states is channel-last layout
            initial_states = initial_states.contiguous()
            # Create output buffer for final_states
            final_states_out = torch.empty(batch, dim, width - 1, dtype=dtype, device="cuda")
        
        # Convert x to channel-last layout
        x_cl = torch.empty(batch, seqlen, dim, dtype=dtype, device="cuda")
        x_cl.copy_(x.permute(0, 2, 1))
        x = x_cl.permute(0, 2, 1)  # Back to [batch, dim, seqlen] but with channel-last stride
        
        # Verify the stride pattern
        assert x.stride(1) == 1, f"Expected stride[1]=1 for channel-last, got {x.stride(1)}"
        assert x.stride(2) > 1, f"Expected stride[2]>1 for channel-last, got {x.stride(2)}"
    elif test_seq_idx or test_states:
        raise ValueError("seq_idx and states are only supported for channel-last layout")
    
    out = torch.empty_like(x)

    # Compute reference using PyTorch (always use contiguous layout for reference)
    x_ref = x.contiguous() if is_channel_last else x
    initial_states_ref = initial_states.contiguous() if initial_states is not None else None
    
    if test_states:
        ref, final_states_ref = torch_causal_conv1d(x_ref, weight, bias, use_silu, 
                                                     initial_states_ref, seq_idx)
    else:
        ref = torch_causal_conv1d(x_ref, weight, bias, use_silu, initial_states_ref, seq_idx)
        final_states_ref = None

    # Run AIter implementation
    # New signature: causal_conv1d_fn(x, weight, bias, seq_idx, initial_states, out, final_states_out, use_silu)
    _, us_aiter = run_perftest(
        aiter.causal_conv1d_fn,
        x,                    # input
        weight,               # weight
        bias,                 # bias
        seq_idx,              # seq_idx (None or tensor)
        initial_states,       # initial_states (None or tensor)
        out,                  # output tensor
        final_states_out,     # final_states_out (None or tensor)
        use_silu,             # activation flag
    )

    # Check accuracy
    out_contiguous = out.contiguous() if is_channel_last else out
    err = checkAllclose(ref, out_contiguous, rtol=1e-3, atol=1e-3)
    
    # Check final_states if tested
    if test_states and final_states_out is not None:
        final_states_contiguous = final_states_out.contiguous()
        err_states = checkAllclose(final_states_ref, final_states_contiguous, rtol=1e-3, atol=1e-3)
        ret["err_states"] = err_states

    # Calculate performance metrics
    bytes_read = x.nbytes + weight.nbytes + bias.nbytes
    if initial_states is not None:
        bytes_read += initial_states.nbytes
    bytes_write = out.nbytes
    if final_states_out is not None:
        bytes_write += final_states_out.nbytes
    total_bytes = bytes_read + bytes_write

    ret["us"] = us_aiter
    ret["TB/s"] = total_bytes / us_aiter / 1e6
    ret["RD TB/s"] = bytes_read / us_aiter / 1e6
    ret["WR TB/s"] = bytes_write / us_aiter / 1e6
    ret["err"] = err
    ret["GFLOPS"] = (batch * dim * seqlen * width * 2) / (us_aiter * 1e3)  # 2 ops per MAC
    ret["layout"] = "channel_last" if is_channel_last else "channel_first"
    ret["test_seq_idx"] = test_seq_idx
    ret["test_states"] = test_states

    return ret


@benchmark()
def test_causal_conv1d_no_bias(batch, dim, seqlen, width, dtype, use_silu=False, is_channel_last=False):
    """Test causal conv1d without bias."""
    ret = {}

    x = torch.randn(batch, dim, seqlen, dtype=dtype, device="cuda")
    # Use fp32 for weight to match original causal-conv1d implementation
    weight = torch.randn(dim, width, dtype=torch.float32, device="cuda")
    bias = torch.empty(0, dtype=torch.float32, device="cuda")  # Empty tensor
    
    # Convert to channel-last layout if needed
    if is_channel_last:
        if dim % 8 != 0:
            raise ValueError(f"Channel-last layout requires dim % 8 == 0, got dim={dim}")
        
        x_cl = torch.empty(batch, seqlen, dim, dtype=dtype, device="cuda")
        x_cl.copy_(x.permute(0, 2, 1))
        x = x_cl.permute(0, 2, 1)
        
        assert x.stride(1) == 1, f"Expected stride[1]=1 for channel-last"
        assert x.stride(2) > 1, f"Expected stride[2]>1 for channel-last"
    
    out = torch.empty_like(x)

    x_ref = x.contiguous() if is_channel_last else x
    ref = torch_causal_conv1d(x_ref, weight, None, use_silu)

    # New signature: causal_conv1d_fn(x, weight, bias, seq_idx, initial_states, out, final_states_out, use_silu)
    _, us_aiter = run_perftest(
        aiter.causal_conv1d_fn,
        x,           # input
        weight,      # weight
        bias,        # bias (empty tensor for no bias)
        None,        # seq_idx
        None,        # initial_states
        out,         # output tensor
        None,        # final_states_out
        use_silu,    # activation flag
    )

    out_contiguous = out.contiguous() if is_channel_last else out
    err = checkAllclose(ref, out_contiguous, rtol=1e-3, atol=1e-3)

    bytes_read = x.nbytes + weight.nbytes
    bytes_write = out.nbytes
    total_bytes = bytes_read + bytes_write

    ret["us"] = us_aiter
    ret["TB/s"] = total_bytes / us_aiter / 1e6
    ret["err"] = err
    ret["GFLOPS"] = (batch * dim * seqlen * width * 2) / (us_aiter * 1e3)
    ret["layout"] = "channel_last" if is_channel_last else "channel_first"

    return ret


# Test configurations (默认使用小规模配置快速验证)
l_dtype = ["float32", "float16", "bfloat16"]  # 测试所有支持的数据类型
l_batch = [2, 4]
l_dim = [64, 256]
l_seqlen = [2048]
l_width = [4]
l_use_silu = [False, True]  # 测试两种场景: 无激活 和 SiLU激活
l_layout = ["channel_first", "channel_last"]  # 测试两种布局

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
parser.add_argument("--layout", type=str, default=None, choices=["channel_first", "channel_last"], 
                    help="Memory layout: channel_first or channel_last (if not specified, test both)")
parser.add_argument("--test_seq_idx", action="store_true", help="Test seq_idx parameter (channel-last only)")
parser.add_argument("--test_states", action="store_true", help="Test initial_states and final_states (channel-last only)")
parser.add_argument("--test_all_features", action="store_true", help="Test all advanced features (seq_idx + states)")
parser.add_argument("--verbose", action="store_true", help="Verbose output")


def main():
    args = parser.parse_args()

    # Override default lists if specific values provided
    batches = [args.batch] if args.batch else l_batch
    dims = [args.dim] if args.dim else l_dim
    seqlens = [args.seqlen] if args.seqlen else l_seqlen
    widths = [args.width] if args.width else l_width
    dtypes = [args.dtype] if args.dtype else l_dtype
    layouts = [args.layout] if args.layout else l_layout

    # Handle use_silu
    if args.use_silu:
        use_silus = [True]
    elif args.no_silu:
        use_silus = [False]
    else:
        use_silus = l_use_silu

    # Determine which advanced features to test
    test_features = []
    if args.test_all_features:
        test_features = [(False, False), (True, False), (False, True), (True, True)]  # seq_idx, states
    elif args.test_seq_idx or args.test_states:
        test_features = [(args.test_seq_idx, args.test_states)]
    else:
        test_features = [(False, False)]  # Default: no advanced features

    results = []

    print("=" * 80)
    print("Testing AIter Causal Conv1D Implementation")
    print("=" * 80)
    print(f"Layouts to test: {layouts}")
    print(f"SiLU scenarios to test: {use_silus}")
    if args.test_seq_idx:
        print(f"Testing seq_idx: Yes")
    if args.test_states:
        print(f"Testing states: Yes")
    if args.test_all_features:
        print(f"Testing all feature combinations: Yes")
    print("=" * 80)

    for dtype_str in dtypes:
        dtype = getattr(torch, dtype_str)

        for batch in batches:
            for dim in dims:
                # Skip channel-last tests if dim is not divisible by 8
                valid_layouts = layouts.copy()
                if "channel_last" in valid_layouts and dim % 8 != 0:
                    print(f"\n⚠️  Skipping channel_last for dim={dim} (requires dim % 8 == 0)")
                    valid_layouts = [l for l in valid_layouts if l != "channel_last"]
                
                for seqlen in seqlens:
                    for width in widths:
                        for layout in valid_layouts:
                            is_channel_last = (layout == "channel_last")
                            
                            # Determine valid feature combinations for this layout
                            valid_features = test_features
                            if not is_channel_last:
                                # Advanced features only work with channel-last
                                valid_features = [(False, False)]
                            
                            for test_seq_idx, test_states in valid_features:
                                for use_silu in use_silus:
                                    test_name = f"B{batch}_D{dim}_S{seqlen}_W{width}_{dtype_str}_{layout}"
                                    if use_silu:
                                        test_name += "_silu"
                                    if args.no_bias:
                                        test_name += "_nobias"
                                    if test_seq_idx:
                                        test_name += "_seqidx"
                                    if test_states:
                                        test_name += "_states"

                                    print(f"\n{'='*60}")
                                    print(f"Test: {test_name}")
                                    print(f"{'='*60}")

                                    try:
                                        if args.no_bias:
                                            result = test_causal_conv1d_no_bias(
                                                batch, dim, seqlen, width, dtype, use_silu, is_channel_last
                                            )
                                        else:
                                            result = test_causal_conv1d_fn(
                                                batch, dim, seqlen, width, dtype, use_silu, is_channel_last,
                                                test_seq_idx, test_states
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
                                        if 'err_states' in result:
                                            print(f"✅ States Error: {result['err_states']:.2e}")
                                        print(f"✅ Layout: {result['layout']}")

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
        
        # Print statistics by layout
        if 'layout' in df.columns and len(df['layout'].unique()) > 1:
            print("\n" + "=" * 80)
            print("STATISTICS BY LAYOUT")
            print("=" * 80)
            for layout in df['layout'].unique():
                df_layout = df[df['layout'] == layout]
                print(f"\n{layout.upper()}:")
                print(f"  Average time: {df_layout['us'].mean():.2f} us")
                print(f"  Average bandwidth: {df_layout['TB/s'].mean():.2f} TB/s")
                print(f"  Average GFLOPS: {df_layout['GFLOPS'].mean():.2f}")
                print(f"  Max error: {df_layout['err'].max():.2e}")


if __name__ == "__main__":
    main()

