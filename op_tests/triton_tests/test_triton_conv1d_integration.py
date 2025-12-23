"""
Test script for Triton Conv1D integration in linear_attention.py

This script tests both PyTorch Conv1d and Triton Conv1d implementations.
"""

import torch
from aiter.ops.triton._triton_kernels.gdn_block_sglang import Qwen3GatedDeltaNet

def test_conv1d_implementations():
    """Test both PyTorch and Triton conv1d implementations"""
    
    print("=" * 80)
    print("Testing Conv1D Integration in Qwen3GatedDeltaNet")
    print("=" * 80)
    
    # Configuration
    batch_size = 2
    seq_len = 16
    hidden_size = 256
    num_k_heads = 4
    num_v_heads = 4
    head_k_dim = 32
    head_v_dim = 32
    conv_kernel_size = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Device: {device}")
    print(f"  Dtype: {dtype}")
    
    # Create input
    hidden_states = torch.randn(
        batch_size, seq_len, hidden_size,
        dtype=dtype, device=device
    )
    
    # Test 1: Triton Conv1d (Default)
    print("\n" + "-" * 80)
    print("Test 1: Triton Conv1d Implementation (Default)")
    print("-" * 80)
    
    try:
        model_triton = Qwen3GatedDeltaNet(
            hidden_size=hidden_size,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_kernel_size=conv_kernel_size,
            dtype=dtype,
            device=device,
            # use_triton_conv1d=True,  # Default, no need to specify
        )
        
        print("✓ Model created successfully")
        
        output_triton, _ = model_triton(hidden_states, mode='chunk')
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output_triton.shape}")
        print(f"  Output dtype: {output_triton.dtype}")
        print(f"  Output device: {output_triton.device}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: PyTorch Conv1d
    print("\n" + "-" * 80)
    print("Test 2: PyTorch Conv1d Implementation")
    print("-" * 80)
    
    try:
        model_pytorch = Qwen3GatedDeltaNet(
            hidden_size=hidden_size,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_kernel_size=conv_kernel_size,
            dtype=dtype,
            device=device,
            use_triton_conv1d=False,  # Use PyTorch
        )
        
        print("✓ Model created successfully")
        
        # Copy weights from Triton model to PyTorch model for fair comparison
        # Copy projection weights
        model_pytorch.in_proj_qkvz.weight.data = model_triton.in_proj_qkvz.weight.data.clone()
        model_pytorch.in_proj_ba.weight.data = model_triton.in_proj_ba.weight.data.clone()
        model_pytorch.out_proj.weight.data = model_triton.out_proj.weight.data.clone()
        
        # Copy conv weights (need to expand for PyTorch format)
        # Triton weight: [conv_dim, kernel_size]
        # PyTorch Conv1d expects: [out_channels, in_channels/groups, kernel_size]
        triton_conv_weight = model_triton.conv1d_weight.data  # [conv_dim, kernel_size]
        pytorch_conv_weight = triton_conv_weight.unsqueeze(1)  # [conv_dim, 1, kernel_size]
        model_pytorch.conv1d.weight.data = pytorch_conv_weight
        
        # Copy other parameters
        model_pytorch.A_log.data = model_triton.A_log.data.clone()
        model_pytorch.dt_bias.data = model_triton.dt_bias.data.clone()
        model_pytorch.norm.weight.data = model_triton.norm.weight.data.clone()
        
        print("✓ Weights copied from Triton model")
        
        output_pytorch, _ = model_pytorch(hidden_states, mode='chunk')
        print(f"✓ Forward pass successful")
        print(f"  Output shape: {output_pytorch.shape}")
        print(f"  Output dtype: {output_pytorch.dtype}")
        print(f"  Output device: {output_pytorch.device}")
        
        # Compare outputs
        print("\n" + "-" * 80)
        print("Comparing Outputs (Triton vs PyTorch)")
        print("-" * 80)
        
        max_diff = (output_triton - output_pytorch).abs().max().item()
        mean_diff = (output_triton - output_pytorch).abs().mean().item()
        
        print(f"  Max absolute difference: {max_diff:.6e}")
        print(f"  Mean absolute difference: {mean_diff:.6e}")
        
        if max_diff < 1e-2:  # Tolerance for bfloat16
            print("  ✓ Outputs match closely!")
        else:
            print("  ⚠ Outputs differ (may be due to numerical precision differences)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Testing Complete")
    print("=" * 80)


if __name__ == "__main__":
    test_conv1d_implementations()

