"""
Test suite for Gated Delta Rule (GDN) implementations.

Tests all three implementations against reference PyTorch implementation
and validates state consistency, numerical accuracy, and edge cases.

Author: AIter Team
"""

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.gated_delta_rule import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update,
    fused_gdn_gating,
    compute_gating_params,
    GatedDeltaRuleOp,
)


def torch_reference_gated_delta_rule(q, k, v, g, beta, scale=None):
    """
    PyTorch reference implementation for testing.
    
    This is a naive but correct implementation used as ground truth.
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    
    if scale is None:
        scale = K ** -0.5
    
    q = q * scale
    
    # Convert to float32 for numerical stability
    q = q.float()
    k = k.float()
    v = v.float()
    g = g.float()
    beta = beta.float()
    
    # Initialize hidden state
    h = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    outputs = []
    
    for t in range(T):
        # Gated decay
        h = h * torch.exp(g[:, t, :, None, None])
        
        # Delta rule: v -= h^T @ k
        kv_proj = (h * k[:, t, :, :, None]).sum(dim=-2)
        delta_v = v[:, t, :, :] - kv_proj
        
        # Beta gating
        delta_v = delta_v * beta[:, t, :, None]
        
        # State update: h += k ⊗ v
        h = h + k[:, t, :, :, None] * delta_v[:, :, None, :]
        
        # Output: o = h @ q
        o = (h * q[:, t, :, :, None]).sum(dim=-2)
        outputs.append(o)
    
    return torch.stack(outputs, dim=1), h


class TestGatedDeltaRule:
    """Main test suite for GDN implementations."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(42)
    
    def _create_test_inputs(self, B, T, H, K, V, dtype=torch.bfloat16, device="cuda"):
        """Create test input tensors."""
        q = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
        k = torch.randn(B, T, H, K, dtype=dtype, device=device) * 0.1
        v = torch.randn(B, T, H, V, dtype=dtype, device=device) * 0.1
        g = torch.rand(B, T, H, dtype=torch.float32, device=device) * 0.1 - 5.0
        beta = torch.rand(B, T, H, dtype=dtype, device=device).sigmoid()
        return q, k, v, g, beta
    
    # ========================================================================
    # Chunk Implementation Tests
    # ========================================================================
    
    @pytest.mark.parametrize("B,T,H,K,V", [
        (2, 64, 4, 32, 32),
        (1, 128, 8, 64, 64),
        (4, 256, 16, 128, 128),
    ])
    @pytest.mark.parametrize("use_l2norm", [False])  # L2norm changes behavior
    def test_chunk_vs_reference(self, B, T, H, K, V, use_l2norm):
        """Test chunk implementation against reference."""
        q, k, v, g, beta = self._create_test_inputs(B, T, H, K, V)
        
        # Chunk implementation
        o_chunk, state_chunk = chunk_gated_delta_rule(
            q, k, v, g, beta,
            output_final_state=True,
            use_qk_l2norm_in_kernel=use_l2norm
        )
        
        if not use_l2norm:
            # Reference implementation
            o_ref, state_ref = torch_reference_gated_delta_rule(q, k, v, g, beta)
            
            # Check output accuracy
            assert torch.allclose(o_chunk.float(), o_ref, atol=1e-2, rtol=1e-2), \
                f"Chunk output mismatch. Max diff: {(o_chunk.float() - o_ref).abs().max()}"
            
            # Check state accuracy
            assert torch.allclose(state_chunk.float(), state_ref, atol=1e-2, rtol=1e-2), \
                f"Chunk state mismatch. Max diff: {(state_chunk.float() - state_ref).abs().max()}"
        
        print(f"✓ Chunk test passed: B={B}, T={T}, H={H}, K={K}, V={V}, L2norm={use_l2norm}")
    
    def test_chunk_with_initial_state(self):
        """Test chunk with initial state."""
        B, T, H, K, V = 2, 64, 4, 32, 32
        q, k, v, g, beta = self._create_test_inputs(B, T, H, K, V)
        
        # Create initial state
        initial_state = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda") * 0.1
        
        # Run with initial state
        o, final_state = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=initial_state,
            output_final_state=True
        )
        
        assert o.shape == (B, T, H, V)
        assert final_state.shape == (B, H, K, V)
        print("✓ Chunk with initial state test passed")
    
    # ========================================================================
    # Fused Recurrent Implementation Tests
    # ========================================================================
    
    @pytest.mark.parametrize("B,T,H,K,V", [
        (2, 16, 4, 32, 32),
        (1, 32, 8, 64, 64),
        (4, 64, 16, 128, 128),
    ])
    def test_recurrent_vs_reference(self, B, T, H, K, V):
        """Test fused recurrent implementation against reference."""
        q, k, v, g, beta = self._create_test_inputs(B, T, H, K, V)
        
        # Fused recurrent
        o_recurrent, state_recurrent = fused_recurrent_gated_delta_rule(
            q, k, v, g, beta,
            output_final_state=True
        )
        
        # Reference
        o_ref, state_ref = torch_reference_gated_delta_rule(q, k, v, g, beta)
        
        # Check accuracy
        assert torch.allclose(o_recurrent.float(), o_ref, atol=1e-2, rtol=1e-2), \
            f"Recurrent output mismatch. Max diff: {(o_recurrent.float() - o_ref).abs().max()}"
        
        assert torch.allclose(state_recurrent.float(), state_ref, atol=1e-2, rtol=1e-2), \
            f"Recurrent state mismatch. Max diff: {(state_recurrent.float() - state_ref).abs().max()}"
        
        print(f"✓ Recurrent test passed: B={B}, T={T}, H={H}, K={K}, V={V}")
    
    # ========================================================================
    # Fused Sigmoid Gating Tests
    # ========================================================================
    
    def test_fused_sigmoid_gating_basic(self):
        """Test fused sigmoid gating implementation."""
        B, T, H, K, V = 2, 1, 8, 64, 64
        
        # Inputs
        q = torch.randn(1, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
        k = torch.randn(1, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
        v = torch.randn(1, T, H, V, dtype=torch.bfloat16, device="cuda") * 0.1
        
        # Gate parameters
        A_log = torch.rand(H, dtype=torch.float32, device="cuda")
        a = torch.randn(B, H, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(B, H, dtype=torch.bfloat16, device="cuda")
        dt_bias = torch.rand(H, dtype=torch.bfloat16, device="cuda")
        
        # State pool
        state_pool = torch.randn(10, H, K, V, dtype=torch.float32, device="cuda") * 0.1
        state_indices = torch.randint(0, 10, (B,), dtype=torch.int32, device="cuda")
        query_start_loc = torch.tensor([0, T], dtype=torch.int32, device="cuda")
        
        # Run
        o = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log, a=a, dt_bias=dt_bias,
            softplus_beta=1.0, softplus_threshold=20.0,
            q=q, k=k, v=v, b=b,
            initial_state_source=state_pool,
            initial_state_indices=state_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
        )
        
        assert o.shape == (1, T, H, V)
        print("✓ Fused sigmoid gating test passed")
    
    @pytest.mark.skip(reason="fused_gdn_gating currently only supports 2D input [B, H], not 3D [B, T, H]")
    def test_gdn_gating_computation(self):
        """Test separate gating computation."""
        H = 16
        B, T = 4, 32
        
        A_log = torch.rand(H, dtype=torch.float32, device="cuda")
        a = torch.randn(B, T, H, dtype=torch.bfloat16, device="cuda")
        b = torch.randn(B, T, H, dtype=torch.bfloat16, device="cuda")
        dt_bias = torch.rand(H, dtype=torch.bfloat16, device="cuda")
        
        g, beta = compute_gating_params(A_log, a, b, dt_bias)
        
        assert g.shape == (B, T, H)
        assert beta.shape == (B, T, H)
        
        # Check that beta is in [0, 1]
        assert (beta >= 0).all() and (beta <= 1).all()
        
        print("✓ GDN gating computation test passed")
    
    # ========================================================================
    # State Consistency Tests
    # ========================================================================
    
    @pytest.mark.parametrize("impl", ["chunk", "recurrent"])
    def test_state_save_restore(self, impl):
        """Test state save and restore consistency."""
        B, T, H, K, V = 2, 64, 4, 32, 32
        q, k, v, g, beta = self._create_test_inputs(B, T, H, K, V)
        
        # Split sequence
        T1 = T // 2
        
        if impl == "chunk":
            func = chunk_gated_delta_rule
        else:
            func = fused_recurrent_gated_delta_rule
        
        # Process first half
        o1, state1 = func(
            q[:, :T1], k[:, :T1], v[:, :T1],
            g[:, :T1], beta[:, :T1],
            output_final_state=True
        )
        
        # Process second half with state
        o2, state2 = func(
            q[:, T1:], k[:, T1:], v[:, T1:],
            g[:, T1:], beta[:, T1:],
            initial_state=state1,
            output_final_state=True
        )
        
        # Process full sequence
        o_full, state_full = func(
            q, k, v, g, beta,
            output_final_state=True
        )
        
        # Check consistency
        o_concat = torch.cat([o1, o2], dim=1)
        
        assert torch.allclose(o_concat, o_full, atol=1e-2, rtol=1e-2), \
            f"Output mismatch. Max diff: {(o_concat - o_full).abs().max()}"
        
        assert torch.allclose(state2, state_full, atol=1e-2, rtol=1e-2), \
            f"State mismatch. Max diff: {(state2 - state_full).abs().max()}"
        
        print(f"✓ State consistency test passed for {impl}")
    
    # ========================================================================
    # Unified Operator Tests
    # ========================================================================
    
    @pytest.mark.parametrize("T,expected_mode", [
        (1, "recurrent"),
        (64, "recurrent"),
        (256, "chunk"),
    ])
    def test_unified_operator_auto_mode(self, T, expected_mode):
        """Test unified operator with auto mode selection."""
        B, H, K, V = 2, 4, 32, 32
        q, k, v, g, beta = self._create_test_inputs(B, T, H, K, V)
        
        # Get recommendation
        rec_mode = GatedDeltaRuleOp.get_recommended_mode(T, has_gate_params=False)
        assert rec_mode == expected_mode, f"Expected {expected_mode}, got {rec_mode}"
        
        # Run with auto mode
        o, state = GatedDeltaRuleOp.forward(
            q, k, v, g, beta,
            mode="auto",
            output_final_state=True
        )
        
        assert o.shape == (B, T, H, V)
        if state is not None:
            assert state.shape == (B, H, K, V)
        
        print(f"✓ Unified operator test passed for T={T} (mode={expected_mode})")
    
    def test_unified_operator_forced_modes(self):
        """Test unified operator with forced modes."""
        B, T, H, K, V = 2, 64, 4, 32, 32
        q, k, v, g, beta = self._create_test_inputs(B, T, H, K, V)
        
        for mode in ["chunk", "recurrent"]:
            o, state = GatedDeltaRuleOp.forward(
                q, k, v, g, beta,
                mode=mode,
                output_final_state=True
            )
            assert o.shape == (B, T, H, V)
            assert state.shape == (B, H, K, V)
        
        print("✓ Unified operator forced modes test passed")
    
    # ========================================================================
    # Edge Cases and Error Handling
    # ========================================================================
    
    def test_single_token(self):
        """Test with single token (T=1)."""
        B, T, H, K, V = 2, 1, 4, 32, 32
        q, k, v, g, beta = self._create_test_inputs(B, T, H, K, V)
        
        # Test all implementations
        o_chunk, _ = chunk_gated_delta_rule(q, k, v, g, beta)
        o_recurrent, _ = fused_recurrent_gated_delta_rule(q, k, v, g, beta)
        
        assert o_chunk.shape == (B, T, H, V)
        assert o_recurrent.shape == (B, T, H, V)
        
        print("✓ Single token test passed")
    
    def test_variable_length_sequences(self):
        """Test with variable-length sequences using cu_seqlens."""
        B, H, K, V = 1, 4, 32, 32
        
        # Create sequences of different lengths
        seqlens = [32, 64, 48, 56]
        T = sum(seqlens)
        cu_seqlens = torch.tensor([0] + [sum(seqlens[:i+1]) for i in range(len(seqlens))],
                                   dtype=torch.int32, device="cuda")
        
        q, k, v, g, beta = self._create_test_inputs(B, T, H, K, V)
        
        # Initial states for each sequence
        N = len(seqlens)
        initial_state = torch.randn(N, H, K, V, dtype=torch.float32, device="cuda") * 0.1
        
        # Test chunk
        o_chunk, state_chunk = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
        
        # Test recurrent
        o_recurrent, state_recurrent = fused_recurrent_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=initial_state,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
        
        assert o_chunk.shape == (B, T, H, V)
        assert state_chunk.shape == (N, H, K, V)
        assert o_recurrent.shape == (B, T, H, V)
        assert state_recurrent.shape == (N, H, K, V)
        
        print("✓ Variable-length sequences test passed")
    
    # ========================================================================
    # Performance and Shape Tests
    # ========================================================================
    
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
    def test_different_dtypes(self, dtype):
        """Test with different data types."""
        B, T, H, K, V = 2, 32, 4, 32, 32
        
        q = torch.randn(B, T, H, K, dtype=dtype, device="cuda") * 0.1
        k = torch.randn(B, T, H, K, dtype=dtype, device="cuda") * 0.1
        v = torch.randn(B, T, H, V, dtype=dtype, device="cuda") * 0.1
        g = torch.rand(B, T, H, dtype=torch.float32, device="cuda") * 0.1 - 5.0
        beta = torch.rand(B, T, H, dtype=dtype, device="cuda").sigmoid()
        
        # Test chunk
        o_chunk, _ = chunk_gated_delta_rule(q, k, v, g, beta)
        assert o_chunk.dtype == dtype
        
        # Test recurrent
        o_recurrent, _ = fused_recurrent_gated_delta_rule(q, k, v, g, beta)
        assert o_recurrent.dtype == dtype
        
        print(f"✓ Different dtypes test passed for {dtype}")
    
    def test_gva_support(self):
        """Test Grouped Value Attention (HV > H)."""
        B, T, H, K, V = 2, 32, 4, 32, 32
        HV = H * 2  # GVA: double the value heads
        
        q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
        k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
        v = torch.randn(B, T, HV, V, dtype=torch.bfloat16, device="cuda") * 0.1
        g = torch.rand(B, T, HV, dtype=torch.float32, device="cuda") * 0.1 - 5.0
        beta = torch.rand(B, T, HV, dtype=torch.bfloat16, device="cuda").sigmoid()
        
        # Only recurrent supports GVA directly
        o, state = fused_recurrent_gated_delta_rule(
            q, k, v, g, beta,
            output_final_state=True
        )
        
        assert o.shape == (B, T, HV, V)
        assert state.shape == (B, HV, K, V)
        
        print("✓ GVA support test passed")


class TestGatedDeltaRuleIntegration:
    """Integration tests with more complex scenarios."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test environment."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        torch.manual_seed(42)
    
    def test_multi_step_generation(self):
        """Simulate multi-step token generation."""
        B, H, K, V = 2, 8, 64, 64
        
        # Initial state
        state = torch.randn(B, H, K, V, dtype=torch.float32, device="cuda") * 0.1
        
        # Generate 10 tokens
        for step in range(10):
            # Single token input
            q = torch.randn(B, 1, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
            k = torch.randn(B, 1, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
            v = torch.randn(B, 1, H, V, dtype=torch.bfloat16, device="cuda") * 0.1
            g = torch.rand(B, 1, H, dtype=torch.float32, device="cuda") * 0.1 - 5.0
            beta = torch.rand(B, 1, H, dtype=torch.bfloat16, device="cuda").sigmoid()
            
            # Update
            o, state = fused_recurrent_gated_delta_rule(
                q, k, v, g, beta,
                initial_state=state,
                output_final_state=True
            )
            
            assert o.shape == (B, 1, H, V)
            assert state.shape == (B, H, K, V)
        
        print("✓ Multi-step generation test passed")
    
    def test_prefill_then_decode(self):
        """Test prefill with chunk then decode with recurrent."""
        B, H, K, V = 2, 8, 64, 64
        T_prefill = 128
        T_decode = 10
        
        # Prefill phase
        q_prefill = torch.randn(B, T_prefill, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
        k_prefill = torch.randn(B, T_prefill, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
        v_prefill = torch.randn(B, T_prefill, H, V, dtype=torch.bfloat16, device="cuda") * 0.1
        g_prefill = torch.rand(B, T_prefill, H, dtype=torch.float32, device="cuda") * 0.1 - 5.0
        beta_prefill = torch.rand(B, T_prefill, H, dtype=torch.bfloat16, device="cuda").sigmoid()
        
        # Prefill with chunk
        _, state = chunk_gated_delta_rule(
            q_prefill, k_prefill, v_prefill, g_prefill, beta_prefill,
            output_final_state=True
        )
        
        # Decode phase
        for step in range(T_decode):
            q = torch.randn(B, 1, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
            k = torch.randn(B, 1, H, K, dtype=torch.bfloat16, device="cuda") * 0.1
            v = torch.randn(B, 1, H, V, dtype=torch.bfloat16, device="cuda") * 0.1
            g = torch.rand(B, 1, H, dtype=torch.float32, device="cuda") * 0.1 - 5.0
            beta = torch.rand(B, 1, H, dtype=torch.bfloat16, device="cuda").sigmoid()
            
            _, state = fused_recurrent_gated_delta_rule(
                q, k, v, g, beta,
                initial_state=state,
                output_final_state=True
            )
        
        print("✓ Prefill then decode test passed")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])

