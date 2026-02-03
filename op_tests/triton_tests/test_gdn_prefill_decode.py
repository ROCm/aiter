"""
Test SGLang-style implementation of Qwen3GatedDeltaNet

This test validates that the fixed GDNAttnBackend correctly handles forward_decode and forward_extend scenarios

Author: AIter Team
"""

import pytest
import torch

from aiter.ops.triton.gated_delta_net.gated_delta_net import Qwen3GatedDeltaNet
from aiter.ops.triton.gated_delta_net.gdn_attn_backend import GDNAttnBackend


# Define multiple test configurations
TEST_CONFIGS = [
    # Small config - fast testing
    {
        'name': 'small',
        'hidden_size': 512,
        'num_k_heads': 4,
        'num_v_heads': 8,
        'head_k_dim': 32,
        'head_v_dim': 32,
    },
    # Medium config
    {
        'name': 'medium',
        'hidden_size': 1024,
        'num_k_heads': 8,
        'num_v_heads': 16,
        'head_k_dim': 64,
        'head_v_dim': 64,
    },
    # Large config - similar to Qwen3-Next
    {
        'name': 'large',
        'hidden_size': 2048,
        'num_k_heads': 16,
        'num_v_heads': 32,
        'head_k_dim': 128,
        'head_v_dim': 128,
    },
]


class TestQwen3GDNSGLangStyle:
    """Test SGLang-style Qwen3GatedDeltaNet implementation"""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required")
        return torch.device('cuda')
    
    @pytest.fixture(params=TEST_CONFIGS, ids=[cfg['name'] for cfg in TEST_CONFIGS])
    def layer_config(self, request):
        """Parameterized layer configuration fixture"""
        config = request.param.copy()
        # Add common configuration
        config.update({
            'conv_kernel_size': 4,
            'rms_norm_eps': 1e-6,
            'dtype': torch.bfloat16,
        })
        return config
    
    def test_forward_decode_basic(self, device, layer_config):
        """Test basic decode mode (seq_len=1)"""
        config_name = layer_config.get('name', 'custom')
        print("\n" + "="*70)
        print(f"Test: forward_decode basic functionality [{config_name}]")
        print(f"Config: hidden={layer_config['hidden_size']}, heads={layer_config['num_k_heads']}, "
              f"head_dim={layer_config['head_k_dim']}")
        print("="*70)
        
        # Remove name field (used only for test identification)
        model_config = {k: v for k, v in layer_config.items() if k != 'name'}
        layer = Qwen3GatedDeltaNet(**model_config, device=device)
        attn_backend = GDNAttnBackend(device=device)
        
        batch_size = 4
        seq_len = 1
        hidden_size = layer_config['hidden_size']
        
        # Create SGLang-style input: [seq_len, hidden_size]
        hidden_states = torch.randn(
            seq_len, hidden_size,
            dtype=layer_config['dtype'],
            device=device
        )
        
        # Create caches
        conv_dim = layer.conv_dim
        conv_state = torch.zeros(
            batch_size, conv_dim, layer.conv_kernel_size - 1,
            dtype=layer_config['dtype'], device=device
        )
        ssm_state = torch.zeros(
            batch_size, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim,
            dtype=torch.float32, device=device
        )
        cache_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
        
        # Execute forward (will call forward_decode)
        output = layer(
            hidden_states,
            attn_backend=attn_backend,
            conv_state=conv_state,
            ssm_state=ssm_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            use_qk_l2norm=True,
        )
        
        # Validate output shape: input [seq_len, hidden] -> backend [1, seq_len, h, d] -> output [1, seq_len, hidden]
        print(f"Input shape: {hidden_states.shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == (1, seq_len, hidden_size), f"Expected {(1, seq_len, hidden_size)}, got {output.shape}"
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        print("✓ forward_decode test passed")
    
    def test_forward_extend_basic(self, device, layer_config):
        """Test basic extend mode (seq_len>1)"""
        config_name = layer_config.get('name', 'custom')
        print("\n" + "="*70)
        print(f"Test: forward_extend basic functionality [{config_name}]")
        print(f"Config: hidden={layer_config['hidden_size']}, heads={layer_config['num_k_heads']}, "
              f"head_dim={layer_config['head_k_dim']}")
        print("="*70)
        
        # Remove name field (used only for test identification)
        model_config = {k: v for k, v in layer_config.items() if k != 'name'}
        layer = Qwen3GatedDeltaNet(**model_config, device=device)
        attn_backend = GDNAttnBackend(device=device)
        
        batch_size = 2
        seq_len = 128
        hidden_size = layer_config['hidden_size']
        
        # Create SGLang-style input: [seq_len, hidden_size]
        hidden_states = torch.randn(
            seq_len, hidden_size,
            dtype=layer_config['dtype'],
            device=device
        )
        
        # Create caches
        conv_dim = layer.conv_dim
        conv_state = torch.zeros(
            batch_size, conv_dim, layer.conv_kernel_size - 1,
            dtype=layer_config['dtype'], device=device
        )
        ssm_state = torch.zeros(
            batch_size, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim,
            dtype=torch.float32, device=device
        )
        cache_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        query_start_loc = torch.tensor([0, seq_len//2, seq_len], dtype=torch.int32, device=device)
        seq_lens_cpu = [seq_len // 2, seq_len // 2]
        has_initial_state = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Execute forward (will call forward_extend)
        output = layer(
            hidden_states,
            attn_backend=attn_backend,
            conv_state=conv_state,
            ssm_state=ssm_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            has_initial_state=has_initial_state,
            seq_lens_cpu=seq_lens_cpu,
            use_qk_l2norm=True,
        )
        
        # Validate output shape: [1, seq_len, hidden_size]
        print(f"Input shape: {hidden_states.shape}")
        print(f"Output shape: {output.shape}")
        assert output.shape == (1, seq_len, hidden_size), f"Expected {(1, seq_len, hidden_size)}, got {output.shape}"
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        print("✓ forward_extend test passed")
    
    def test_decode_extend_integration(self, device, layer_config):
        """Test integration of decode and extend scenarios"""
        config_name = layer_config.get('name', 'custom')
        print("\n" + "="*70)
        print(f"Test: decode + extend integration [{config_name}]")
        print(f"Config: hidden={layer_config['hidden_size']}, heads={layer_config['num_k_heads']}, "
              f"head_dim={layer_config['head_k_dim']}")
        print("="*70)
        
        # Remove name field (used only for test identification)
        model_config = {k: v for k, v in layer_config.items() if k != 'name'}
        layer = Qwen3GatedDeltaNet(**model_config, device=device)
        attn_backend = GDNAttnBackend(device=device)
        
        batch_size = 2
        prefill_len = 256
        hidden_size = layer_config['hidden_size']
        
        # Create caches
        conv_dim = layer.conv_dim
        conv_state = torch.zeros(
            batch_size, conv_dim, layer.conv_kernel_size - 1,
            dtype=layer_config['dtype'], device=device
        )
        ssm_state = torch.zeros(
            batch_size, layer.num_v_heads, layer.head_k_dim, layer.head_v_dim,
            dtype=torch.float32, device=device
        )
        cache_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
        
        # Phase 1: Prefill (forward_extend)
        print("\nPhase 1: Prefill")
        prefill_states = torch.randn(
            prefill_len, hidden_size,
            dtype=layer_config['dtype'], device=device
        )
        query_start_loc = torch.tensor([0, prefill_len//2, prefill_len], dtype=torch.int32, device=device)
        seq_lens_cpu = [prefill_len // 2] * batch_size
        has_initial_state = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        prefill_output = layer(
            prefill_states,
            attn_backend=attn_backend,
            conv_state=conv_state,
            ssm_state=ssm_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            has_initial_state=has_initial_state,
            seq_lens_cpu=seq_lens_cpu,
            use_qk_l2norm=True,
        )
        
        print(f"Prefill output shape: {prefill_output.shape}")
        ssm_state_after_prefill = ssm_state.clone()
        
        # Phase 2: Decode (forward_decode)
        # In decode mode, seq_len must be 1 to trigger forward_decode
        # Process 1 token at a time, batch is managed through query_start_loc and cache_indices
        print("\nPhase 2: Decode")
        decode_steps = 5
        
        for step in range(decode_steps):
            # Input shape: [1, hidden_size] - single token
            decode_input = torch.randn(
                1, hidden_size,
                dtype=layer_config['dtype'], device=device
            )
            
            # Decode for all batch samples (sequentially or in parallel depending on implementation)
            # Simplified here to process the first sample
            query_start_loc_decode = torch.tensor([0, 1], dtype=torch.int32, device=device)
            cache_idx = torch.tensor([0], dtype=torch.int32, device=device)
            
            decode_output = layer(
                decode_input,
                attn_backend=attn_backend,
                conv_state=conv_state,
                ssm_state=ssm_state,
                cache_indices=cache_idx,
                query_start_loc=query_start_loc_decode,
                use_qk_l2norm=True,
            )
            
            if step == 0:
                print(f"Decode step {step} output shape: {decode_output.shape}")
        
        # Verify state has been updated
        state_diff = (ssm_state - ssm_state_after_prefill).norm().item()
        print(f"\nState change norm: {state_diff:.4f}")
        assert state_diff > 0, "State should be updated during decode process"
        print("✓ Integration test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])