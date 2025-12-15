"""
Test suite for Qwen3GatedDeltaNet layer in aiter.

This test demonstrates how to use the ported Qwen3GatedDeltaNet layer
for various scenarios: prefill, decode, and variable-length sequences.

Author: AIter Team
"""

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton._triton_kernels.gdn_block_sglang import Qwen3GatedDeltaNet


class TestQwen3GatedDeltaNet:
    """Test suite for Qwen3GatedDeltaNet layer."""
    
    @pytest.fixture
    def device(self):
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @pytest.fixture
    def layer_config(self):
        """Standard configuration for testing."""
        return {
            'hidden_size': 512,
            'num_k_heads': 8,
            'num_v_heads': 8,
            'head_k_dim': 64,
            'head_v_dim': 64,
            'conv_kernel_size': 4,
            'rms_norm_eps': 1e-6,
            'dtype': torch.bfloat16,
        }
    
    def test_layer_initialization(self, device, layer_config):
        """Test layer can be initialized correctly."""
        layer = Qwen3GatedDeltaNet(**layer_config, device=device)
        
        assert layer.hidden_size == 512
        assert layer.num_k_heads == 8
        assert layer.num_v_heads == 8
        assert layer.A_log.shape == (8,)
        assert layer.dt_bias.shape == (8,)
        print("✓ Layer initialization test passed")
    
    def test_prefill_chunk_mode(self, device, layer_config):
        """Test prefill with chunk-based computation."""
        layer = Qwen3GatedDeltaNet(**layer_config, device=device)
        
        batch_size = 2
        seq_len = 256  # Long sequence
        hidden_size = layer_config['hidden_size']
        
        # Create input
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size,
            dtype=layer_config['dtype'],
            device=device
        )
        
        # Forward pass with chunk mode
        output, final_state = layer(
            hidden_states,
            mode='chunk',
            output_final_state=True,
        )
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert final_state is not None
        assert final_state.shape == (
            batch_size, 
            layer.num_v_heads, 
            layer.head_k_dim, 
            layer.head_v_dim
        )
        
        print(f"✓ Prefill chunk mode test passed")
        print(f"  Input shape: {hidden_states.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  State shape: {final_state.shape}")
    
    def test_decode_mode(self, device, layer_config):
        """Test single-step decode."""
        layer = Qwen3GatedDeltaNet(**layer_config, device=device)
        
        batch_size = 4
        seq_len = 1  # Single token
        hidden_size = layer_config['hidden_size']
        
        # Create input
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size,
            dtype=layer_config['dtype'],
            device=device
        )
        
        # Create initial state
        initial_state = torch.randn(
            batch_size,
            layer.num_v_heads,
            layer.head_k_dim,
            layer.head_v_dim,
            dtype=torch.float32,
            device=device
        )
        
        # Forward pass with fused decode mode
        output, _ = layer(
            hidden_states,
            mode='fused_decode',
            initial_state=initial_state,
        )
        
        # Check output shape
        assert output.shape == (batch_size, seq_len, hidden_size)
        
        print(f"✓ Decode mode test passed")
        print(f"  Input shape: {hidden_states.shape}")
        print(f"  Output shape: {output.shape}")
    
    def test_auto_mode_selection(self, device, layer_config):
        """Test automatic mode selection based on sequence length."""
        layer = Qwen3GatedDeltaNet(**layer_config, device=device)
        
        hidden_size = layer_config['hidden_size']
        
        # Test short sequence (should use recurrent)
        hidden_states_short = torch.randn(
            2, 64, hidden_size,
            dtype=layer_config['dtype'],
            device=device
        )
        output_short, _ = layer(hidden_states_short, mode='auto')
        assert output_short.shape == hidden_states_short.shape
        
        # Test long sequence (should use chunk)
        hidden_states_long = torch.randn(
            2, 512, hidden_size,
            dtype=layer_config['dtype'],
            device=device
        )
        output_long, _ = layer(hidden_states_long, mode='auto')
        assert output_long.shape == hidden_states_long.shape
        
        # Test single token (should use fused_decode)
        hidden_states_single = torch.randn(
            2, 1, hidden_size,
            dtype=layer_config['dtype'],
            device=device
        )
        output_single, _ = layer(hidden_states_single, mode='auto')
        assert output_single.shape == hidden_states_single.shape
        
        print(f"✓ Auto mode selection test passed")
        print(f"  Short seq (64): {output_short.shape}")
        print(f"  Long seq (512): {output_long.shape}")
        print(f"  Single token (1): {output_single.shape}")
    
    def test_consistency_across_modes(self, device, layer_config):
        """Test that different modes produce similar results (approximately)."""
        layer = Qwen3GatedDeltaNet(**layer_config, device=device)
        
        # Use small sequence that all modes can handle
        batch_size = 1
        seq_len = 16
        hidden_size = layer_config['hidden_size']
        
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size,
            dtype=layer_config['dtype'],
            device=device
        )
        
        # Run with different modes
        with torch.no_grad():
            output_chunk, _ = layer(hidden_states, mode='chunk')
            output_recurrent, _ = layer(hidden_states, mode='recurrent')
        
        # Check they're approximately equal
        # (Note: Different kernels may have slight numerical differences)
        diff = (output_chunk - output_recurrent).abs().max()
        print(f"✓ Mode consistency test")
        print(f"  Max difference between chunk and recurrent: {diff.item():.6f}")
        
        # Loose tolerance due to different numerical implementations
        assert diff < 0.1, f"Outputs differ too much: {diff}"
    
    def test_state_persistence(self, device, layer_config):
        """Test that state can be passed between forward passes."""
        layer = Qwen3GatedDeltaNet(**layer_config, device=device)
        
        batch_size = 2
        hidden_size = layer_config['hidden_size']
        
        # First forward pass
        hidden_states_1 = torch.randn(
            batch_size, 32, hidden_size,
            dtype=layer_config['dtype'],
            device=device
        )
        output_1, state_1 = layer(
            hidden_states_1,
            mode='chunk',
            output_final_state=True,
        )
        
        # Second forward pass using state from first
        hidden_states_2 = torch.randn(
            batch_size, 32, hidden_size,
            dtype=layer_config['dtype'],
            device=device
        )
        output_2, state_2 = layer(
            hidden_states_2,
            mode='chunk',
            initial_state=state_1,
            output_final_state=True,
        )
        
        # Check shapes
        assert output_2.shape == (batch_size, 32, hidden_size)
        assert state_2.shape == state_1.shape
        
        # States should be different
        assert not torch.allclose(state_1, state_2, rtol=0.1)
        
        print(f"✓ State persistence test passed")
        print(f"  State 1 norm: {state_1.norm().item():.4f}")
        print(f"  State 2 norm: {state_2.norm().item():.4f}")


def test_integration_example():
    """
    Example of how to use Qwen3GatedDeltaNet in a model.
    """
    print("\n" + "="*60)
    print("Integration Example: Using Qwen3GatedDeltaNet in a model")
    print("="*60)
    
    # Configuration (similar to Qwen3-Next-80B)
    config = {
        'hidden_size': 2048,
        'num_k_heads': 32,
        'num_v_heads': 32,
        'head_k_dim': 64,
        'head_v_dim': 64,
        'conv_kernel_size': 4,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create layer (now from gdn_block_sglang)
    gdn_layer = Qwen3GatedDeltaNet(**config, device=device)
    
    print(f"\nLayer created with:")
    print(f"  Hidden size: {config['hidden_size']}")
    print(f"  Heads: {config['num_k_heads']}")
    print(f"  Head dim: {config['head_k_dim']}")
    
    # Simulate prefill phase
    print(f"\n1. Prefill Phase (long sequence)")
    batch_size = 4
    prefill_len = 1024
    hidden_states = torch.randn(
        batch_size, prefill_len, config['hidden_size'],
        dtype=torch.bfloat16, device=device
    )
    
    output, final_state = gdn_layer(
        hidden_states,
        mode='chunk',  # or 'auto'
        output_final_state=True,
    )
    print(f"  Input: {hidden_states.shape}")
    print(f"  Output: {output.shape}")
    print(f"  Final state: {final_state.shape}")
    
    # Simulate decode phase
    print(f"\n2. Decode Phase (single token generation)")
    for step in range(5):
        # Generate one token at a time
        new_token = torch.randn(
            batch_size, 1, config['hidden_size'],
            dtype=torch.bfloat16, device=device
        )
        
        output, final_state = gdn_layer(
            new_token,
            mode='auto',  # Will automatically use fused_decode
            initial_state=final_state,
            output_final_state=True,
        )
        
        if step == 0:
            print(f"  Step {step}: Input {new_token.shape} → Output {output.shape}")
    
    print(f"\n✓ Integration example completed successfully!")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "-s"])
    
    # Run integration example
    test_integration_example()

