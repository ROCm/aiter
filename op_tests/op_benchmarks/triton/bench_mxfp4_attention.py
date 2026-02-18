"""
Example script demonstrating the use of MXFP4-enhanced unified attention.

This script shows how to:
1. Use different MXFP4 modes
2. Compare performance and accuracy
3. Handle fallback for incompatible head sizes
"""
import os
import torch
import time
from aiter.ops.triton.attention.unified_attention import unified_attention


def create_attention_inputs(
    batch_size=4,
    num_query_heads=32,
    num_kv_heads=8,
    head_size=128,
    seq_len=512,
    block_size=16,
    device='cuda',
    dtype=torch.float16,
):
    """Create sample inputs for attention"""
    # Calculate total tokens
    total_tokens = batch_size * seq_len
    
    # Create cumulative sequence lengths
    cu_seqlens_q = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * seq_len
    
    # Create input tensors
    q = torch.randn(total_tokens, num_query_heads, head_size, device=device, dtype=dtype)
    
    # Calculate number of blocks
    num_blocks_per_seq = (seq_len + block_size - 1) // block_size
    total_blocks = batch_size * num_blocks_per_seq
    
    k = torch.randn(total_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype)
    v = torch.randn(total_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype)
    
    # Create block table
    block_table = torch.arange(total_blocks, device=device, dtype=torch.int32)
    block_table = block_table.reshape(batch_size, num_blocks_per_seq)
    
    # Create sequence lengths
    seq_lens = torch.full((batch_size,), seq_len, device=device, dtype=torch.int32)
    
    # Output tensor
    out = torch.zeros_like(q)
    
    # Attention parameters
    softmax_scale = 1.0 / (head_size ** 0.5)
    window_size = torch.tensor([-1, -1], device=device)  # No sliding window
    
    return {
        'q': q,
        'k': k,
        'v': v,
        'out': out,
        'cu_seqlens_q': cu_seqlens_q,
        'max_seqlen_q': seq_len,
        'seqused_k': seq_lens,
        'max_seqlen_k': seq_len,
        'softmax_scale': softmax_scale,
        'causal': True,
        'window_size': window_size,
        'block_table': block_table,
        'softcap': 0.0,
        'q_descale': None,
        'k_descale': None,
        'v_descale': None,
    }


def run_attention_benchmark(inputs, use_native_fp4=0, num_iterations=10):
    """Benchmark attention with specified MXFP4 mode"""
    # Warmup
    for _ in range(3):
        inputs['out'].zero_()
        unified_attention(**inputs, use_native_fp4=use_native_fp4)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iterations):
        inputs['out'].zero_()
        unified_attention(**inputs, use_native_fp4=use_native_fp4)
    
    torch.cuda.synchronize()
    end = time.time()
    
    avg_time = (end - start) / num_iterations * 1000  # Convert to ms
    return avg_time


def main():
    print("=" * 80)
    print("MXFP4 Unified Attention Example")
    print("=" * 80)
    
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. This example requires a CUDA-capable GPU.")
        return
    
    # Configuration
    config = {
        'batch_size': 4,
        'num_query_heads': 32,
        'num_kv_heads': 8,
        'head_size': 128,
        'seq_len': 512,
        'block_size': 16,
    }
    
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Create inputs
    print("\nCreating input tensors...")
    inputs = create_attention_inputs(**config)
    
    # Test different MXFP4 modes
    modes = {
        0: "Original (No MXFP4)",
        1: "Native MXFP4 QK",
        2: "Smoothed MXFP4 QK",
        3: "Smoothed MXFP4 QK + PV",
    }
    
    print("\n" + "=" * 80)
    print("Running Attention with Different MXFP4 Modes")
    print("=" * 80)
    
    results = {}
    
    for mode, description in modes.items():
        print(f"\nMode {mode}: {description}")
        print("-" * 40)
        
        try:
            # Run attention
            inputs['out'].zero_()
            unified_attention(**inputs, use_native_fp4=mode)
            
            # Basic checks
            output_mean = inputs['out'].mean().item()
            output_std = inputs['out'].std().item()
            has_nan = torch.any(torch.isnan(inputs['out'])).item()
            has_inf = torch.any(torch.isinf(inputs['out'])).item()
            
            print(f"  Output mean: {output_mean:.6f}")
            print(f"  Output std:  {output_std:.6f}")
            print(f"  Has NaN:     {has_nan}")
            print(f"  Has Inf:     {has_inf}")
            
            if has_nan or has_inf:
                print("  ⚠️  Warning: Invalid values detected!")
                continue
            
            # Benchmark
            avg_time = run_attention_benchmark(inputs, use_native_fp4=mode, num_iterations=10)
            print(f"  Avg time:    {avg_time:.3f} ms")
            
            # Store results
            results[mode] = {
                'output': inputs['out'].clone(),
                'time': avg_time,
            }
            
            print("  ✓ Success")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    # Compare accuracy
    if 0 in results and len(results) > 1:
        print("\n" + "=" * 80)
        print("Accuracy Comparison (vs Original)")
        print("=" * 80)
        
        original_output = results[0]['output']
        
        for mode in [1, 2, 3]:
            if mode not in results:
                continue
            
            mxfp4_output = results[mode]['output']
            
            # Calculate errors
            abs_error = torch.abs(mxfp4_output - original_output)
            rel_error = abs_error / (torch.abs(original_output) + 1e-8)
            
            max_abs_error = abs_error.max().item()
            mean_abs_error = abs_error.mean().item()
            max_rel_error = rel_error.max().item()
            mean_rel_error = rel_error.mean().item()
            
            print(f"\nMode {mode}: {modes[mode]}")
            print(f"  Max absolute error:  {max_abs_error:.6f}")
            print(f"  Mean absolute error: {mean_abs_error:.6f}")
            print(f"  Max relative error:  {max_rel_error:.4%}")
            print(f"  Mean relative error: {mean_rel_error:.4%}")
            
            if mean_rel_error < 0.05:
                print("  ✓ Excellent accuracy (<5% mean error)")
            elif mean_rel_error < 0.10:
                print("  ✓ Good accuracy (<10% mean error)")
            elif mean_rel_error < 0.20:
                print("  ⚠️  Acceptable accuracy (<20% mean error)")
            else:
                print("  ⚠️  High error (>20% mean error)")
    
    # Performance summary
    if len(results) > 1:
        print("\n" + "=" * 80)
        print("Performance Summary")
        print("=" * 80)
        
        if 0 in results:
            baseline_time = results[0]['time']
            print(f"\nBaseline (Mode 0): {baseline_time:.3f} ms")
            
            for mode in [1, 2, 3]:
                if mode not in results:
                    continue
                
                mode_time = results[mode]['time']
                speedup = baseline_time / mode_time
                
                print(f"Mode {mode}: {mode_time:.3f} ms ({speedup:.2f}x speedup)")
    
    # Test fallback with incompatible head size
    print("\n" + "=" * 80)
    print("Testing Fallback for Incompatible Head Size")
    print("=" * 80)
    
    print("\nCreating inputs with head_size=16 (incompatible with MXFP4)...")
    config_small = config.copy()
    config_small['head_size'] = 16
    inputs_small = create_attention_inputs(**config_small)
    
    print("Attempting to use MXFP4 mode 1 (should fallback to mode 0)...")
    try:
        unified_attention(**inputs_small, use_native_fp4=1)
        print("✓ Fallback successful - computation completed without error")
    except Exception as e:
        print(f"✗ Fallback failed: {e}")
    
    print("\n" + "=" * 80)
    print("Example Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
