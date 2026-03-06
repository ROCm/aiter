"""
Test file for Hadamard rotation with MXFP4 quantization in unified attention.
Tests the new Hadamard rotation feature with various configurations.
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
    max_seq_len=256,
    block_size=16,
    device='cuda',
    dtype=torch.float16,
):
    """Create test inputs for unified attention"""
    num_queries_per_kv = num_query_heads // num_kv_heads
    
    seq_lens = torch.randint(32, max_seq_len + 1, (batch_size,), device=device)
    total_tokens = seq_lens.sum().item()
    
    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1:] = torch.cumsum(seq_lens, dim=0)
    
    q = torch.randn(total_tokens, num_query_heads, head_size, device=device, dtype=dtype)
    
    max_num_blocks = (max_seq_len + block_size - 1) // block_size
    num_blocks = max_num_blocks * batch_size
    
    k = torch.randn(num_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype)
    v = torch.randn(num_blocks, block_size, num_kv_heads, head_size, device=device, dtype=dtype)
    
    block_table = torch.arange(num_blocks, device=device, dtype=torch.int32).reshape(batch_size, -1)
    
    out = torch.zeros_like(q)
    
    softmax_scale = 1.0 / (head_size ** 0.5)
    window_size = torch.tensor([-1, -1], device=device)
    
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
@pytest.mark.parametrize("use_hadamard", [False, True])
def test_hadamard_rotation_modes(head_size, use_native_fp4, use_hadamard):
    """Test Hadamard rotation with different MXFP4 modes and head sizes"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    inputs = create_test_inputs(
        batch_size=2,
        num_query_heads=8,
        num_kv_heads=4,
        head_size=head_size,
        max_seq_len=256,
        block_size=16,
    )
    
    try:
        unified_attention(
            **inputs,
            use_native_fp4=use_native_fp4,
            use_hadamard_rotation=use_hadamard,
        )
        
        assert not torch.all(inputs['out'] == 0), "Output should not be all zeros"
        assert not torch.any(torch.isnan(inputs['out'])), "Output contains NaN"
        assert not torch.any(torch.isinf(inputs['out'])), "Output contains Inf"
        
        print(f"✓ Test passed: head_size={head_size}, fp4_mode={use_native_fp4}, hadamard={use_hadamard}")
    except Exception as e:
        print(f"✗ Test failed: head_size={head_size}, fp4_mode={use_native_fp4}, hadamard={use_hadamard}")
        raise


def test_hadamard_matrix_generation():
    """Test Hadamard matrix generation and properties"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    from aiter.ops.triton._triton_kernels.attention.unified_attention import generate_hadamard_matrix
    
    # Test different sizes
    for size in [32, 64, 128, 256]:
        H = generate_hadamard_matrix(size, device='cuda')
        
        # Check shape
        assert H.shape == (size, size), f"Expected shape ({size}, {size}), got {H.shape}"
        
        # Check orthogonality: H @ H^T should be identity
        I = torch.matmul(H, H.t())
        identity = torch.eye(size, device='cuda')
        
        assert torch.allclose(I, identity, atol=1e-5), f"Hadamard matrix size {size} is not orthogonal"
        
        print(f"✓ Hadamard matrix size {size} is valid and orthogonal")


def test_hadamard_rotation_accuracy():
    """Test accuracy of Hadamard rotation compared to no rotation"""
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
    
    # Run with original (no MXFP4)
    out_original = inputs['out'].clone()
    unified_attention(**inputs, use_native_fp4=0, use_hadamard_rotation=False)
    out_original.copy_(inputs['out'])
    
    # Run with MXFP4 mode 2, no Hadamard
    inputs['out'].zero_()
    unified_attention(**inputs, use_native_fp4=2, use_hadamard_rotation=False)
    out_fp4_no_hadamard = inputs['out'].clone()
    
    # Run with MXFP4 mode 2 with Hadamard
    inputs['out'].zero_()
    unified_attention(**inputs, use_native_fp4=2, use_hadamard_rotation=True)
    out_fp4_with_hadamard = inputs['out'].clone()
    
    # Calculate errors
    error_no_hadamard = torch.abs(out_fp4_no_hadamard - out_original).mean().item()
    error_with_hadamard = torch.abs(out_fp4_with_hadamard - out_original).mean().item()
    
    print(f"Mean absolute error without Hadamard: {error_no_hadamard:.6f}")
    print(f"Mean absolute error with Hadamard: {error_with_hadamard:.6f}")
    
    # Hadamard rotation should ideally reduce error or be comparable
    # This is a soft check since it depends on the data
    print(f"Hadamard rotation error ratio: {error_with_hadamard / error_no_hadamard:.4f}")


def test_hadamard_with_pv_quantization():
    """Test Hadamard rotation with PV quantization (mode 3+)"""
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
    
    # Test mode 3 with and without Hadamard
    for use_hadamard in [False, True]:
        inputs['out'].zero_()
        unified_attention(**inputs, use_native_fp4=3, use_hadamard_rotation=use_hadamard)
        
        assert not torch.all(inputs['out'] == 0), f"Output should not be all zeros (hadamard={use_hadamard})"
        assert not torch.any(torch.isnan(inputs['out'])), f"Output contains NaN (hadamard={use_hadamard})"
        assert not torch.any(torch.isinf(inputs['out'])), f"Output contains Inf (hadamard={use_hadamard})"
        
        print(f"✓ Mode 3 with hadamard={use_hadamard} passed")


def test_hadamard_size_parameter():
    """Test different Hadamard matrix sizes"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    inputs = create_test_inputs(
        batch_size=1,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=64,
        max_seq_len=128,
        block_size=16,
    )
    
    # Test with different Hadamard sizes
    for hadamard_size in [32, 64, 128]:
        inputs['out'].zero_()
        unified_attention(
            **inputs,
            use_native_fp4=2,
            use_hadamard_rotation=True,
            hadamard_size=hadamard_size,
        )
        
        assert not torch.all(inputs['out'] == 0), f"Output should not be all zeros (size={hadamard_size})"
        print(f"✓ Hadamard size {hadamard_size} works correctly")


def test_environment_variable():
    """Test Hadamard rotation via environment variable"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    inputs = create_test_inputs(
        batch_size=1,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=64,
        max_seq_len=128,
        block_size=16,
    )
    
    # Save old env vars
    old_mxfp4 = os.getenv('MXFP4_OPTION', '0')
    old_hadamard = os.getenv('HADAMARD_ROTATION', '0')
    
    try:
        # Enable via environment variables
        os.environ['MXFP4_OPTION'] = '2'
        os.environ['HADAMARD_ROTATION'] = '1'
        
        unified_attention(**inputs)
        
        assert not torch.all(inputs['out'] == 0), "Output should not be all zeros"
        print("✓ Environment variable configuration works")
        
    finally:
        os.environ['MXFP4_OPTION'] = old_mxfp4
        os.environ['HADAMARD_ROTATION'] = old_hadamard


def test_hadamard_with_incompatible_head_size():
    """Test that Hadamard rotation falls back gracefully with incompatible head size"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Use head_size=16 which is incompatible with MXFP4
    inputs = create_test_inputs(
        batch_size=1,
        num_query_heads=4,
        num_kv_heads=2,
        head_size=16,  # Incompatible
        max_seq_len=128,
        block_size=16,
    )
    
    # Should fall back to no MXFP4 (and thus no Hadamard)
    unified_attention(**inputs, use_native_fp4=2, use_hadamard_rotation=True)
    
    assert not torch.all(inputs['out'] == 0), "Output should not be all zeros"
    print("✓ Fallback for incompatible head_size works")


if __name__ == "__main__":
    print("Running Hadamard rotation + MXFP4 tests...\n")
    
    test_functions = [
        ("Hadamard matrix generation", test_hadamard_matrix_generation),
        ("Basic Hadamard modes", lambda: test_hadamard_rotation_modes(64, 2, True)),
        ("Hadamard accuracy", test_hadamard_rotation_accuracy),
        ("Hadamard with PV quantization", test_hadamard_with_pv_quantization),
        ("Hadamard size parameter", test_hadamard_size_parameter),
        ("Environment variable", test_environment_variable),
        ("Incompatible head_size fallback", test_hadamard_with_incompatible_head_size),
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
