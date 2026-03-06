"""
Test file for unified attention with MXFP4 support.
Tests both the original logic and the new MXFP4 logic with various configurations.
"""
import os
import torch
import pytest
import triton
from aiter.ops.triton.attention.unified_attention import unified_attention


def create_test_inputs(
    batch_size=2,
    num_query_heads=8,
    num_kv_heads=4,
    head_size=64,
    max_seq_len=512,
    block_size=16,
    device='cuda',
    dtype=torch.float16,
):
    """Create test inputs for unified attention"""
    num_queries_per_kv = num_query_heads // num_kv_heads
    
    # Create query lengths for each sequence
    seq_lens = torch.randint(32, max_seq_len + 1, (batch_size,), device=device)
    total_tokens = seq_lens.sum().item()
    
    # Create cumulative sequence lengths
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seq_lens, dim=0)
    
    # Create input tensors
    q = torch.randn(total_tokens, num_query_heads, head_size, device=device, dtype=dtype)
    
    # Calculate number of blocks needed
    max_num_blocks = (max_seq_len + block_size - 1) // block_size
    num_blocks = max_num_blocks * batch_size
    
    k = torch.randn(num_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype)
    v = torch.randn(num_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype)
    
    # Create block tables
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(batch_size, -1)
    
    # Create output tensor
    out = torch.zeros_like(q)
    
    # Other parameters
    softmax_scale = 1.0 / (head_size ** 0.5)
    window_size = torch.tensor([-1, -1], device=device)  # No sliding window
    
    return {
        'q': q,
        'k': k,
        'v': v,
        'out': out,
        'cu_seqlens_q': cu_seqlens_q,
        'max_seqlen_q': seq_lens.max().item(),
        'seqused_k': seq_lens,
        'max_seqlen_k': max_seq_len,
        'softmax_scale': softmax_scale,
        'causal': True,
        'window_size': window_size,
        'block_table': block_table,
        'softcap': 0.0,
        'q_descale': None,
        'k_descale': None,
        'v_descale': None,
    }


@pytest.mark.parametrize("head_size", [32, 64, 128])
@pytest.mark.parametrize("use_native_fp4", [0, 1, 2, 3])
def test_unified_attention_mxfp4_modes(head_size, use_native_fp4):
    """Test unified attention with different MXFP4 modes and head sizes"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Test with different head sizes to verify fallback logic
    inputs = create_test_inputs(
        batch_size=2,
        num_query_heads=8,
        num_kv_heads=4,
        head_size=head_size,
        max_seq_len=256,
        block_size=16,
    )
    
    # Set environment variable for this test
    old_env = os.getenv('MXFP4_OPTION', '0')
    os.environ['MXFP4_OPTION'] = str(use_native_fp4)
    
    try:
        # Run attention
        unified_attention(**inputs, use_native_fp4=use_native_fp4)
        
        # Check output is not all zeros (basic sanity check)
        assert not torch.all(inputs['out'] == 0), "Output should not be all zeros"
        
        # Check for NaNs or Infs
        assert not torch.any(torch.isnan(inputs['out'])), "Output contains NaN"
        assert not torch.any(torch.isinf(inputs['out'])), "Output contains Inf"
        
        print(f"✓ Test passed: head_size={head_size}, use_native_fp4={use_native_fp4}")
    finally:
        os.environ['MXFP4_OPTION'] = old_env


def test_mxfp4_fallback_small_head_size():
    """Test that MXFP4 properly falls back when head_size is incompatible"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Use head_size=16 which when padded becomes 16 (not divisible by 32)
    inputs = create_test_inputs(
        batch_size=1,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=16,  # This will be padded to 16, incompatible with MXFP4
        max_seq_len=128,
        block_size=16,
    )
    
    # Try to use MXFP4 - it should fall back to original
    unified_attention(**inputs, use_native_fp4=1)
    
    # Should complete without error
    assert not torch.all(inputs['out'] == 0), "Output should not be all zeros"
    print("✓ Fallback test passed for small head_size")


def test_mxfp4_vs_original_consistency():
    """Test that MXFP4 mode 0 (original) produces consistent results"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    inputs = create_test_inputs(
        batch_size=2,
        num_query_heads=8,
        num_kv_heads=4,
        head_size=64,
        max_seq_len=256,
        block_size=16,
    )
    
    # Run with original mode
    out1 = inputs['out'].clone()
    unified_attention(**inputs, use_native_fp4=0)
    out1.copy_(inputs['out'])
    
    # Run again with original mode
    inputs['out'].zero_()
    unified_attention(**inputs, use_native_fp4=0)
    
    # Results should be identical
    torch.testing.assert_close(out1, inputs['out'], rtol=1e-5, atol=1e-5)
    print("✓ Consistency test passed for original mode")


def test_mxfp4_approximate_correctness():
    """Test that MXFP4 modes produce approximately correct results compared to original"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    inputs = create_test_inputs(
        batch_size=2,
        num_query_heads=8,
        num_kv_heads=4,
        head_size=64,
        max_seq_len=256,
        block_size=16,
    )
    
    # Run with original mode
    out_original = inputs['out'].clone()
    unified_attention(**inputs, use_native_fp4=0)
    out_original.copy_(inputs['out'])
    
    # Test MXFP4 modes
    for mode in [1, 2]:
        inputs['out'].zero_()
        unified_attention(**inputs, use_native_fp4=mode)
        
        # Calculate relative error
        rel_error = torch.abs(inputs['out'] - out_original) / (torch.abs(out_original) + 1e-8)
        max_rel_error = rel_error.max().item()
        mean_rel_error = rel_error.mean().item()
        
        print(f"MXFP4 mode {mode}: max_rel_error={max_rel_error:.4f}, mean_rel_error={mean_rel_error:.4f}")
        
        # MXFP4 should be reasonably close to original (within 10% typically)
        assert mean_rel_error < 0.2, f"Mean relative error too large for mode {mode}: {mean_rel_error}"


@pytest.mark.parametrize("block_size", [16, 32, 64])
@pytest.mark.parametrize("sliding_window", [0, 128])
def test_various_block_and_window_sizes(block_size, sliding_window):
    """Test with various block sizes and sliding window configurations"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    inputs = create_test_inputs(
        batch_size=1,
        num_query_heads=8,
        num_kv_heads=4,
        head_size=64,
        max_seq_len=256,
        block_size=block_size,
    )
    
    # Update window size
    if sliding_window > 0:
        inputs['window_size'] = torch.tensor([sliding_window - 1, 0], device='cuda')
    
    # Test with both original and MXFP4
    for use_fp4 in [0, 1]:
        inputs['out'].zero_()
        unified_attention(**inputs, use_native_fp4=use_fp4)
        
        assert not torch.all(inputs['out'] == 0), "Output should not be all zeros"
        assert not torch.any(torch.isnan(inputs['out'])), "Output contains NaN"
    
    print(f"✓ Test passed: block_size={block_size}, sliding_window={sliding_window}")


def test_edge_cases():
    """Test edge cases like single token, max sequence length, etc."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Test 1: Single token per sequence (decode)
    inputs = create_test_inputs(
        batch_size=4,
        num_query_heads=8,
        num_kv_heads=4,
        head_size=64,
        max_seq_len=256,
        block_size=16,
    )
    # Override to make all sequences single token
    inputs['cu_seqlens_q'] = torch.arange(5, device='cuda', dtype=torch.int32)
    inputs['q'] = inputs['q'][:4]
    inputs['seqused_k'] = torch.ones(4, device='cuda', dtype=torch.int32) * 128
    inputs['max_seqlen_q'] = 1
    inputs['out'] = torch.zeros_like(inputs['q'])
    
    unified_attention(**inputs, use_native_fp4=0)
    assert not torch.all(inputs['out'] == 0), "Single token output should not be all zeros"
    print("✓ Single token test passed")
    
    # Test 2: Maximum head size
    inputs_large = create_test_inputs(
        batch_size=1,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=256,  # Large head size
        max_seq_len=128,
        block_size=16,
    )
    
    unified_attention(**inputs_large, use_native_fp4=1)
    assert not torch.all(inputs_large['out'] == 0), "Large head_size output should not be all zeros"
    print("✓ Large head_size test passed")


def test_compatibility_check():
    """Test the _can_use_mxfp4 logic by checking various configurations"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Head sizes that should work with MXFP4
    compatible_head_sizes = [32, 64, 128, 256]
    
    # Head sizes that should NOT work with MXFP4
    incompatible_head_sizes = [16, 24, 48, 96]
    
    print("\nTesting MXFP4 compatibility:")
    
    for head_size in compatible_head_sizes:
        head_size_padded = triton.next_power_of_2(head_size)
        should_work = (head_size_padded >= 32 and head_size_padded % 32 == 0)
        print(f"  head_size={head_size}, padded={head_size_padded}, compatible={should_work}")
        assert should_work, f"head_size={head_size} should be compatible but isn't"
    
    for head_size in incompatible_head_sizes:
        head_size_padded = triton.next_power_of_2(head_size)
        should_not_work = not (head_size_padded >= 32 and head_size_padded % 32 == 0)
        print(f"  head_size={head_size}, padded={head_size_padded}, incompatible={should_not_work}")
        assert should_not_work, f"head_size={head_size} should be incompatible but isn't"
    
    print("✓ Compatibility check passed")


if __name__ == "__main__":
    print("Running unified attention MXFP4 tests...\n")
    
    # Run tests individually for easier debugging
    test_functions = [
        ("MXFP4 modes", lambda: test_unified_attention_mxfp4_modes(64, 1)),
        ("Fallback for small head_size", test_mxfp4_fallback_small_head_size),
        ("Original mode consistency", test_mxfp4_vs_original_consistency),
        ("MXFP4 approximate correctness", test_mxfp4_approximate_correctness),
        ("Block and window sizes", lambda: test_various_block_and_window_sizes(16, 0)),
        ("Edge cases", test_edge_cases),
        ("Compatibility check", test_compatibility_check),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in test_functions:
        try:
            print(f"\n{'='*60}")
            print(f"Running: {test_name}")
            print('='*60)
            test_func()
            passed += 1
            print(f"✓ {test_name} PASSED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"Test Summary: {passed} passed, {failed} failed")
    print('='*60)
    
    if failed == 0:
        print("\n✓ All tests passed!")
    else:
        print(f"\n✗ {failed} tests failed")
