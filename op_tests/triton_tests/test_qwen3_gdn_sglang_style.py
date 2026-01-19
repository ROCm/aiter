"""
测试 Qwen3GatedDeltaNet 的 SGLang 风格实现

该测试验证修复后的 GDNAttnBackend 能正确处理 forward_decode 和 forward_extend 场景

Author: AIter Team
"""

import pytest
import torch

from aiter.ops.triton._triton_kernels.gdn_block_sglang import Qwen3GatedDeltaNet
from aiter.ops.triton._triton_kernels.gdn_block_sglang.gdn_attn_backend import GDNAttnBackend


# 定义多组测试配置
TEST_CONFIGS = [
    # 小配置 - 快速测试
    {
        'name': 'small',
        'hidden_size': 512,
        'num_k_heads': 4,
        'num_v_heads': 8,
        'head_k_dim': 32,
        'head_v_dim': 32,
    },
    # 中等配置
    {
        'name': 'medium',
        'hidden_size': 1024,
        'num_k_heads': 8,
        'num_v_heads': 16,
        'head_k_dim': 64,
        'head_v_dim': 64,
    },
    # 大配置 - 类似 Qwen3-Next
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
    """测试 SGLang 风格的 Qwen3GatedDeltaNet 实现"""
    
    @pytest.fixture
    def device(self):
        if not torch.cuda.is_available():
            pytest.skip("需要 CUDA")
        return torch.device('cuda')
    
    @pytest.fixture(params=TEST_CONFIGS, ids=[cfg['name'] for cfg in TEST_CONFIGS])
    def layer_config(self, request):
        """参数化的层配置 fixture"""
        config = request.param.copy()
        # 添加通用配置
        config.update({
            'conv_kernel_size': 4,
            'rms_norm_eps': 1e-6,
            'dtype': torch.bfloat16,
        })
        return config
    
    def test_forward_decode_basic(self, device, layer_config):
        """测试基本的 decode 模式（seq_len=1）"""
        config_name = layer_config.get('name', 'custom')
        print("\n" + "="*70)
        print(f"测试: forward_decode 基本功能 [{config_name}]")
        print(f"配置: hidden={layer_config['hidden_size']}, heads={layer_config['num_k_heads']}, "
              f"head_dim={layer_config['head_k_dim']}")
        print("="*70)
        
        # 移除 name 字段（仅用于测试标识）
        model_config = {k: v for k, v in layer_config.items() if k != 'name'}
        layer = Qwen3GatedDeltaNet(**model_config, device=device)
        attn_backend = GDNAttnBackend(device=device)
        
        batch_size = 4
        seq_len = 1
        hidden_size = layer_config['hidden_size']
        
        # 创建 SGLang 风格的输入: [seq_len, hidden_size]
        hidden_states = torch.randn(
            seq_len, hidden_size,
            dtype=layer_config['dtype'],
            device=device
        )
        
        # 创建缓存
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
        
        # 执行 forward（会调用 forward_decode）
        output = layer(
            hidden_states,
            attn_backend=attn_backend,
            conv_state=conv_state,
            ssm_state=ssm_state,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            use_qk_l2norm=True,
        )
        
        # 验证输出形状: 输入 [seq_len, hidden] -> backend [1, seq_len, h, d] -> 输出 [1, seq_len, hidden]
        print(f"输入形状: {hidden_states.shape}")
        print(f"输出形状: {output.shape}")
        assert output.shape == (1, seq_len, hidden_size), f"期望 {(1, seq_len, hidden_size)}，得到 {output.shape}"
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        print("✓ forward_decode 测试通过")
    
    def test_forward_extend_basic(self, device, layer_config):
        """测试基本的 extend 模式（seq_len>1）"""
        config_name = layer_config.get('name', 'custom')
        print("\n" + "="*70)
        print(f"测试: forward_extend 基本功能 [{config_name}]")
        print(f"配置: hidden={layer_config['hidden_size']}, heads={layer_config['num_k_heads']}, "
              f"head_dim={layer_config['head_k_dim']}")
        print("="*70)
        
        # 移除 name 字段（仅用于测试标识）
        model_config = {k: v for k, v in layer_config.items() if k != 'name'}
        layer = Qwen3GatedDeltaNet(**model_config, device=device)
        attn_backend = GDNAttnBackend(device=device)
        
        batch_size = 2
        seq_len = 128
        hidden_size = layer_config['hidden_size']
        
        # 创建 SGLang 风格的输入: [seq_len, hidden_size]
        hidden_states = torch.randn(
            seq_len, hidden_size,
            dtype=layer_config['dtype'],
            device=device
        )
        
        # 创建缓存
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
        
        # 执行 forward（会调用 forward_extend）
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
        
        # 验证输出形状: [1, seq_len, hidden_size]
        print(f"输入形状: {hidden_states.shape}")
        print(f"输出形状: {output.shape}")
        assert output.shape == (1, seq_len, hidden_size), f"期望 {(1, seq_len, hidden_size)}，得到 {output.shape}"
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        print("✓ forward_extend 测试通过")
    
    def test_decode_extend_integration(self, device, layer_config):
        """测试 decode 和 extend 的集成场景"""
        config_name = layer_config.get('name', 'custom')
        print("\n" + "="*70)
        print(f"测试: decode + extend 集成 [{config_name}]")
        print(f"配置: hidden={layer_config['hidden_size']}, heads={layer_config['num_k_heads']}, "
              f"head_dim={layer_config['head_k_dim']}")
        print("="*70)
        
        # 移除 name 字段（仅用于测试标识）
        model_config = {k: v for k, v in layer_config.items() if k != 'name'}
        layer = Qwen3GatedDeltaNet(**model_config, device=device)
        attn_backend = GDNAttnBackend(device=device)
        
        batch_size = 2
        prefill_len = 256
        hidden_size = layer_config['hidden_size']
        
        # 创建缓存
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
        
        # 阶段 1: Prefill (forward_extend)
        print("\n阶段 1: Prefill")
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
        
        print(f"Prefill 输出形状: {prefill_output.shape}")
        ssm_state_after_prefill = ssm_state.clone()
        
        # 阶段 2: Decode (forward_decode)
        # 在 decode 模式下，seq_len 必须为 1 才会触发 forward_decode
        # 每次处理 1 个 token，batch 通过 query_start_loc 和 cache_indices 管理
        print("\n阶段 2: Decode")
        decode_steps = 5
        
        for step in range(decode_steps):
            # 输入形状: [1, hidden_size] - 单个 token
            decode_input = torch.randn(
                1, hidden_size,
                dtype=layer_config['dtype'], device=device
            )
            
            # 为所有batch样本解码（轮流或并行取决于实现）
            # 这里简化为处理第一个样本
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
                print(f"Decode 步骤 {step} 输出形状: {decode_output.shape}")
        
        # 验证状态已更新
        state_diff = (ssm_state - ssm_state_after_prefill).norm().item()
        print(f"\n状态变化范数: {state_diff:.4f}")
        assert state_diff > 0, "状态应该在 decode 过程中更新"
        print("✓ 集成测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

