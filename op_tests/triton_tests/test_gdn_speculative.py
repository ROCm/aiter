"""
Tests for GDN (Gated Delta Network) Speculative Decoding

参考sglang的eagle_worker_v2.py和aiter的test_eagle_lightweight.py，
为GDN创建speculative decoding测试。

GDN特点：
- 使用线性注意力机制（O(n)复杂度）
- 支持chunk模式（prefill）和recurrent模式（decode）
- 基于Gated Delta Rule进行状态更新

Author: AIter Team
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List

from aiter.ops.triton._triton_kernels.gdn_block_sglang import Qwen3GatedDeltaNet


class SimpleLMWithGDN(nn.Module):
    """
    简单的语言模型，使用GDN层作为注意力机制。
    
    用于测试GDN speculative decoding，不需要完整的transformer架构。
    """
    
    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 256,
        num_k_heads: int = 4,
        num_v_heads: int = 4,
        head_k_dim: int = 64,
        head_v_dim: int = 64,
        conv_kernel_size: int = 4,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.dtype = dtype
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        
        # Embedding - 使用与GDN相同的dtype
        self.embed = nn.Embedding(vocab_size, hidden_size, dtype=dtype, device=device)
        
        # GDN Layer
        self.gdn = Qwen3GatedDeltaNet(
            hidden_size=hidden_size,
            num_k_heads=num_k_heads,
            num_v_heads=num_v_heads,
            head_k_dim=head_k_dim,
            head_v_dim=head_v_dim,
            conv_kernel_size=conv_kernel_size,
            rms_norm_eps=1e-6,
            dtype=dtype,
            device=device,
            use_triton_conv1d=True,
        )
        
        # LM head - 使用与GDN相同的dtype
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=dtype, device=device)
        
        # State管理
        self.gdn_state = None
    
    def forward(
        self,
        input_ids: torch.Tensor,
        mode: str = "auto",
        past_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        return_hidden: bool = False,
    ):
        """
        前向传播。
        
        Args:
            input_ids: [batch_size, seq_len]
            mode: "chunk" for prefill, "recurrent" for decode, "auto" for auto-select
            past_state: 之前的GDN状态 [batch, num_v_heads, head_k_dim, head_v_dim]
            use_cache: 是否返回状态用于下一步
            return_hidden: 是否返回hidden states
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            past_state: 如果use_cache=True
            hidden_states: 如果return_hidden=True
        """
        # 1. Embedding
        hidden_states = self.embed(input_ids)  # [B, T, H]
        
        # 2. GDN Layer
        hidden_states, final_state = self.gdn(
            hidden_states=hidden_states,
            mode=mode,
            initial_state=past_state,
            output_final_state=use_cache,
        )
        
        # 3. LM head
        logits = self.lm_head(hidden_states)  # [B, T, V]
        
        # 构造返回值
        class Output:
            def __init__(self, logits, past_state=None, hidden_states=None):
                self.logits = logits
                self.past_key_values = past_state  # 兼容HF接口
                self.past_state = past_state
                self.hidden_states = hidden_states
        
        if return_hidden:
            return Output(logits, final_state, hidden_states)
        elif use_cache:
            return Output(logits, final_state)
        else:
            return Output(logits)


class GDNDraftWorker:
    """
    GDN Draft Worker - 负责生成draft tokens。
    
    参考sglang的eagle_worker_v2.py中的EagleDraftWorker实现。
    """
    
    def __init__(
        self,
        draft_model: SimpleLMWithGDN,
        topk: int = 4,
        num_steps: int = 2,
        device: str = "cuda",
    ):
        self.draft_model = draft_model
        self.topk = topk
        self.num_steps = num_steps
        self.device = device
        
        # 计算总的draft tokens数量
        # 简化版本：每步生成topk个tokens
        # 例如: topk=4, num_steps=2 -> 4 + 4 = 8 tokens
        # 完整版本应该是树形展开: 4 + 4*4 = 20 tokens
        # 这里使用简化版本便于测试
        self.num_draft_tokens = topk * num_steps
    
    def draft_step(
        self,
        input_ids: torch.Tensor,
        hidden_states: Optional[torch.Tensor] = None,
        past_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        执行一步draft生成。
        
        Args:
            input_ids: [batch_size, seq_len]
            hidden_states: Optional hidden states from previous step
            past_state: Optional GDN state
            
        Returns:
            scores: [batch_size, topk] - top-k token概率
            token_ids: [batch_size, topk] - top-k token IDs
            new_hidden_states: [batch_size, topk, hidden_size]
            past_state: Updated GDN state
        """
        with torch.no_grad():
            # Forward pass
            outputs = self.draft_model(
                input_ids=input_ids,
                mode="recurrent" if input_ids.size(1) == 1 else "auto",
                past_state=past_state,
                use_cache=True,
                return_hidden=True,
            )
            
            # 获取最后一个位置的logits
            logits = outputs.logits[:, -1, :]  # [B, V]
            
            # 计算概率并选择top-k
            probs = torch.softmax(logits, dim=-1)
            scores, token_ids = torch.topk(probs, k=self.topk, dim=-1)
            
            # 获取hidden states用于下一步
            hidden_states = outputs.hidden_states[:, -1:, :]  # [B, 1, H]
            
            # 扩展hidden_states到topk个候选
            hidden_states = hidden_states.expand(-1, self.topk, -1)  # [B, topk, H]
            
            return scores, token_ids, hidden_states, outputs.past_state
    
    def generate_draft_tree(
        self,
        input_ids: torch.Tensor,
        verified_id: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        生成draft token树。
        
        参考eagle_worker_v2.py的draft_forward方法。
        
        Args:
            input_ids: [batch_size, seq_len] - 输入序列
            verified_id: [batch_size] - 已验证的token ID
            
        Returns:
            draft_tokens: [batch_size, num_draft_tokens-1] - draft tokens（不包括root）
            parent_list: [batch_size, num_parents] - 父节点索引
            top_scores_index: [batch_size, num_draft_tokens-1] - 选中的token索引
        """
        batch_size = input_ids.size(0)
        
        # 存储每一步的结果
        score_list: List[torch.Tensor] = []
        token_list: List[torch.Tensor] = []
        parents_list: List[torch.Tensor] = []
        
        # 初始输入是verified token
        current_ids = verified_id.unsqueeze(1)  # [B, 1]
        past_state = None
        
        # 多步draft生成
        for step in range(self.num_steps):
            # Draft一步
            scores, token_ids, hidden_states, past_state = self.draft_step(
                input_ids=current_ids,
                past_state=past_state,
            )
            
            # 记录当前步的结果
            score_list.append(scores)  # [B, topk]
            token_list.append(token_ids)  # [B, topk]
            
            # 构造父节点索引
            if step == 0:
                # 第一步：所有token的父节点都是root（索引0）
                parents = torch.zeros(batch_size, 1, dtype=torch.long, device=self.device)
            else:
                # 后续步：每个token展开topk个子节点
                parents = torch.arange(
                    score_list[step-1].size(1), 
                    device=self.device
                ).unsqueeze(0).expand(batch_size, -1)
            
            parents_list.append(parents)
            
            # 准备下一步的输入
            if step < self.num_steps - 1:
                # 展开所有候选作为下一步的输入
                current_ids = token_ids.reshape(batch_size, -1)  # [B, topk]
        
        # 组织结果
        # 拼接所有scores和tokens
        all_scores = torch.cat([s.flatten(1) for s in score_list], dim=1)  # [B, total_tokens]
        all_tokens = torch.cat([t.flatten(1) for t in token_list], dim=1)  # [B, total_tokens]
        
        # 实际生成的tokens数量
        total_generated = all_scores.size(1)
        
        # 从所有draft tokens中选择top min(num_draft_tokens-1, total_generated)个
        num_to_select = min(self.num_draft_tokens - 1, total_generated)
        
        if num_to_select > 0:
            top_scores, top_scores_index = torch.topk(
                all_scores, 
                k=num_to_select,
                dim=-1
            )
            
            # 按索引排序（保持树结构的因果性）
            top_scores_index, sort_idx = torch.sort(top_scores_index, dim=-1)
            
            # 根据排序后的索引获取tokens
            draft_tokens = torch.gather(all_tokens, dim=1, index=top_scores_index)
        else:
            # 如果没有生成tokens，返回空tensor
            draft_tokens = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
            top_scores_index = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
        
        # 构造parent_list（排除最后一步，因为它们没有子节点）
        if len(parents_list) > 1:
            parent_list = torch.cat(parents_list[:-1], dim=1)
        else:
            parent_list = torch.empty(batch_size, 0, dtype=torch.long, device=self.device)
        
        return draft_tokens, parent_list, top_scores_index


class GDNVerifyWorker:
    """
    GDN Verify Worker - 负责验证draft tokens。
    
    参考sglang的eagle_worker_v2.py中的EAGLEWorkerV2.verify方法。
    """
    
    def __init__(
        self,
        target_model: SimpleLMWithGDN,
        topk: int = 4,
        num_steps: int = 2,
        num_draft_tokens: int = 20,
        device: str = "cuda",
    ):
        self.target_model = target_model
        self.topk = topk
        self.num_steps = num_steps
        self.num_draft_tokens = num_draft_tokens
        self.device = device
    
    def build_tree_attention_mask(
        self,
        batch_size: int,
        verified_id: torch.Tensor,
        draft_tokens: torch.Tensor,
        parent_list: torch.Tensor,
        seq_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建树形注意力mask。
        
        Args:
            batch_size: 批次大小
            verified_id: [batch_size] - 已验证的token
            draft_tokens: [batch_size, num_draft_tokens-1] - draft tokens
            parent_list: [batch_size, num_parents] - 父节点索引
            seq_lens: [batch_size] - 序列长度
            
        Returns:
            tree_mask: 树形注意力mask
            positions: 位置编码
        """
        # 简化实现：构造因果mask
        # 真实实现应该根据树结构构造更复杂的mask
        
        # 拼接verified token和draft tokens
        all_tokens = torch.cat([
            verified_id.unsqueeze(1),  # [B, 1]
            draft_tokens,  # [B, num_draft-1]
        ], dim=1)  # [B, num_draft]
        
        # 构造因果mask
        tree_mask = torch.tril(torch.ones(
            self.num_draft_tokens,
            self.num_draft_tokens,
            dtype=torch.bool,
            device=self.device
        ))
        
        # 扩展到batch维度
        tree_mask = tree_mask.unsqueeze(0).expand(batch_size, -1, -1)
        
        # 构造位置编码
        positions = torch.arange(
            self.num_draft_tokens,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0).expand(batch_size, -1)
        
        # 加上序列长度偏移
        positions = positions + seq_lens.unsqueeze(1)
        
        return tree_mask, positions
    
    def verify(
        self,
        verified_id: torch.Tensor,
        draft_tokens: torch.Tensor,
        parent_list: torch.Tensor,
        seq_lens: torch.Tensor,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        验证draft tokens。
        
        Args:
            verified_id: [batch_size] - 已验证的root token
            draft_tokens: [batch_size, num_draft_tokens-1] - draft tokens
            parent_list: [batch_size, num_parents] - 父节点索引
            seq_lens: [batch_size] - 当前序列长度
            temperature: 采样温度（0.0为greedy）
            
        Returns:
            accepted_tokens: [batch_size, num_accepted] - 接受的tokens
            accept_length: [batch_size] - 每个序列接受的token数量
        """
        batch_size = verified_id.size(0)
        
        # 构建完整的input（verified + draft）
        all_tokens = torch.cat([
            verified_id.unsqueeze(1),  # [B, 1]
            draft_tokens,  # [B, num_draft-1]
        ], dim=1)  # [B, num_draft]
        
        # 构建树形mask
        tree_mask, positions = self.build_tree_attention_mask(
            batch_size=batch_size,
            verified_id=verified_id,
            draft_tokens=draft_tokens,
            parent_list=parent_list,
            seq_lens=seq_lens,
        )
        
        # Target模型前向传播（使用chunk模式，因为是批量处理）
        with torch.no_grad():
            outputs = self.target_model(
                input_ids=all_tokens,
                mode="chunk",
                use_cache=False,
            )
            
            # 获取logits
            logits = outputs.logits  # [B, num_draft, V]
        
        # 验证每个draft token
        accept_length = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        accepted_tokens_list = []
        
        for b in range(batch_size):
            # 获取该序列的logits和tokens
            seq_logits = logits[b]  # [num_draft, V]
            seq_tokens = all_tokens[b]  # [num_draft]
            
            # 从第一个draft token开始验证
            accepted = [verified_id[b].item()]
            
            for i in range(1, self.num_draft_tokens):
                # 获取前一个位置的预测
                prev_logits = seq_logits[i-1]  # [V]
                
                # 采样或greedy选择
                if temperature == 0.0:
                    predicted_token = torch.argmax(prev_logits).item()
                else:
                    probs = torch.softmax(prev_logits / temperature, dim=-1)
                    predicted_token = torch.multinomial(probs, num_samples=1).item()
                
                # 检查是否匹配
                actual_token = seq_tokens[i].item()
                
                if predicted_token == actual_token:
                    # 接受该token
                    accepted.append(actual_token)
                else:
                    # 拒绝，使用target预测的token，并终止
                    accepted.append(predicted_token)
                    break
            
            accept_length[b] = len(accepted) - 1  # 不计算verified token
            accepted_tokens_list.append(torch.tensor(accepted, device=self.device))
        
        # 填充到相同长度
        max_len = max(len(tokens) for tokens in accepted_tokens_list)
        accepted_tokens = torch.zeros(
            batch_size, max_len,
            dtype=torch.long,
            device=self.device
        )
        
        for b, tokens in enumerate(accepted_tokens_list):
            accepted_tokens[b, :len(tokens)] = tokens
        
        return accepted_tokens, accept_length


class GDNSpeculativeWorker:
    """
    GDN Speculative Decoding Worker - 整合draft和verify。
    
    参考sglang的eagle_worker_v2.py中的EAGLEWorkerV2类。
    """
    
    def __init__(
        self,
        draft_model: SimpleLMWithGDN,
        target_model: SimpleLMWithGDN,
        topk: int = 4,
        num_steps: int = 2,
        device: str = "cuda",
    ):
        self.draft_worker = GDNDraftWorker(
            draft_model=draft_model,
            topk=topk,
            num_steps=num_steps,
            device=device,
        )
        
        self.verify_worker = GDNVerifyWorker(
            target_model=target_model,
            topk=topk,
            num_steps=num_steps,
            num_draft_tokens=self.draft_worker.num_draft_tokens,
            device=device,
        )
        
        self.device = device
        
        # 统计信息
        self.stats = {
            'total_steps': 0,
            'total_accepted_tokens': 0,
            'total_draft_tokens': 0,
            'acceptance_rates': [],
        }
    
    def reset_stats(self):
        """重置统计信息。"""
        self.stats = {
            'total_steps': 0,
            'total_accepted_tokens': 0,
            'total_draft_tokens': 0,
            'acceptance_rates': [],
        }
    
    def generate_step(
        self,
        input_ids: torch.Tensor,
        verified_id: torch.Tensor,
        seq_lens: torch.Tensor,
        temperature: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行一步speculative decoding。
        
        Args:
            input_ids: [batch_size, seq_len] - 当前输入序列
            verified_id: [batch_size] - 上一步验证的token
            seq_lens: [batch_size] - 序列长度
            temperature: 采样温度
            
        Returns:
            accepted_tokens: [batch_size, num_accepted] - 接受的tokens
            accept_length: [batch_size] - 接受的token数量
        """
        # 1. Draft阶段
        draft_tokens, parent_list, top_scores_index = self.draft_worker.generate_draft_tree(
            input_ids=input_ids,
            verified_id=verified_id,
        )
        
        # 2. Verify阶段
        accepted_tokens, accept_length = self.verify_worker.verify(
            verified_id=verified_id,
            draft_tokens=draft_tokens,
            parent_list=parent_list,
            seq_lens=seq_lens,
            temperature=temperature,
        )
        
        # 3. 更新统计
        self.stats['total_steps'] += 1
        self.stats['total_accepted_tokens'] += accept_length.sum().item()
        self.stats['total_draft_tokens'] += draft_tokens.numel()
        
        # 计算acceptance rate
        acceptance_rate = accept_length.float().mean().item() / self.draft_worker.num_draft_tokens
        self.stats['acceptance_rates'].append(acceptance_rate)
        
        return accepted_tokens, accept_length
    
    def get_statistics(self) -> Dict:
        """获取统计信息。"""
        if self.stats['total_draft_tokens'] == 0:
            return {
                'total_steps': 0,
                'mean_acceptance_rate': 0.0,
                'speedup_ratio': 1.0,
            }
        
        mean_acceptance_rate = (
            self.stats['total_accepted_tokens'] / self.stats['total_draft_tokens']
        )
        
        # Speedup ratio = accepted / (steps * draft_per_step)
        # 理论上每步可以接受多个tokens，speedup > 1
        speedup = mean_acceptance_rate * self.draft_worker.num_draft_tokens
        
        return {
            'total_steps': self.stats['total_steps'],
            'total_accepted_tokens': self.stats['total_accepted_tokens'],
            'total_draft_tokens': self.stats['total_draft_tokens'],
            'mean_acceptance_rate': mean_acceptance_rate,
            'speedup_ratio': max(speedup, 1.0),
        }


# ============================================================================
# Pytest测试用例
# ============================================================================

@pytest.fixture
def device():
    """获取CUDA设备（如果可用）。"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')


@pytest.fixture
def simple_gdn_models(device):
    """创建简单的测试模型。"""
    # Draft模型（较小）
    draft_model = SimpleLMWithGDN(
        vocab_size=1000,
        hidden_size=128,
        num_k_heads=2,
        num_v_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        dtype=torch.bfloat16,
        device=device,
    ).eval()
    
    # Target模型（可以相同或更大）
    target_model = SimpleLMWithGDN(
        vocab_size=1000,
        hidden_size=128,
        num_k_heads=2,
        num_v_heads=2,
        head_k_dim=32,
        head_v_dim=32,
        conv_kernel_size=4,
        dtype=torch.bfloat16,
        device=device,
    ).eval()
    
    return draft_model, target_model


class TestGDNLayer:
    """测试GDN层的基本功能。"""
    
    def test_gdn_layer_forward_chunk(self, device):
        """测试GDN层的chunk模式前向传播。"""
        batch_size = 2
        seq_len = 128
        hidden_size = 128
        
        layer = Qwen3GatedDeltaNet(
            hidden_size=hidden_size,
            num_k_heads=2,
            num_v_heads=2,
            head_k_dim=32,
            head_v_dim=32,
            conv_kernel_size=4,
        ).to(device)
        
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size,
            dtype=torch.bfloat16,
            device=device
        )
        
        # Chunk模式（prefill）
        output, final_state = layer(
            hidden_states=hidden_states,
            mode="chunk",
            output_final_state=True,
        )
        
        assert output.shape == (batch_size, seq_len, hidden_size)
        assert final_state.shape == (batch_size, 2, 32, 32)  # [B, num_v_heads, head_k_dim, head_v_dim]
        
        print(f"✓ GDN chunk模式测试通过: output shape={output.shape}, state shape={final_state.shape}")
    
    def test_gdn_layer_forward_recurrent(self, device):
        """测试GDN层的recurrent模式前向传播。"""
        batch_size = 2
        seq_len = 1  # Single token
        hidden_size = 128
        
        layer = Qwen3GatedDeltaNet(
            hidden_size=hidden_size,
            num_k_heads=2,
            num_v_heads=2,
            head_k_dim=32,
            head_v_dim=32,
            conv_kernel_size=4,
        ).to(device)
        
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size,
            dtype=torch.bfloat16,
            device=device
        )
        
        # 初始状态
        initial_state = torch.zeros(
            batch_size, 2, 32, 32,
            dtype=torch.float32,
            device=device
        )
        
        # Recurrent模式（decode）
        output, _ = layer(
            hidden_states=hidden_states,
            mode="recurrent",
            initial_state=initial_state,
            output_final_state=False,
        )
        
        assert output.shape == (batch_size, seq_len, hidden_size)
        
        print(f"✓ GDN recurrent模式测试通过: output shape={output.shape}")


class TestGDNDraftWorker:
    """测试GDN Draft Worker。"""
    
    def test_draft_step(self, simple_gdn_models, device):
        """测试单步draft生成。"""
        draft_model, _ = simple_gdn_models
        
        worker = GDNDraftWorker(
            draft_model=draft_model,
            topk=4,
            num_steps=2,
            device=device,
        )
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        
        scores, token_ids, hidden_states, past_state = worker.draft_step(
            input_ids=input_ids,
            past_state=None,
        )
        
        assert scores.shape == (batch_size, 4)
        assert token_ids.shape == (batch_size, 4)
        assert hidden_states.shape == (batch_size, 4, 128)
        
        print(f"✓ Draft step测试通过")
        print(f"  Scores shape: {scores.shape}")
        print(f"  Token IDs shape: {token_ids.shape}")
        print(f"  Hidden states shape: {hidden_states.shape}")
    
    def test_generate_draft_tree(self, simple_gdn_models, device):
        """测试draft树生成。"""
        draft_model, _ = simple_gdn_models
        
        worker = GDNDraftWorker(
            draft_model=draft_model,
            topk=4,
            num_steps=2,
            device=device,
        )
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        verified_id = input_ids[:, -1]  # 最后一个token作为verified
        
        draft_tokens, parent_list, top_scores_index = worker.generate_draft_tree(
            input_ids=input_ids,
            verified_id=verified_id,
        )
        
        # 检查形状
        assert draft_tokens.shape == (batch_size, worker.num_draft_tokens - 1)
        assert parent_list.shape[0] == batch_size
        assert top_scores_index.shape == (batch_size, worker.num_draft_tokens - 1)
        
        print(f"✓ Draft树生成测试通过")
        print(f"  Draft tokens shape: {draft_tokens.shape}")
        print(f"  Parent list shape: {parent_list.shape}")
        print(f"  Num draft tokens: {worker.num_draft_tokens}")


class TestGDNVerifyWorker:
    """测试GDN Verify Worker。"""
    
    def test_verify(self, simple_gdn_models, device):
        """测试draft tokens验证。"""
        _, target_model = simple_gdn_models
        
        topk = 4
        num_steps = 2
        num_draft_tokens = 4 + 16  # topk + topk^2
        
        worker = GDNVerifyWorker(
            target_model=target_model,
            topk=topk,
            num_steps=num_steps,
            num_draft_tokens=num_draft_tokens,
            device=device,
        )
        
        batch_size = 2
        verified_id = torch.randint(0, 1000, (batch_size,), device=device)
        draft_tokens = torch.randint(0, 1000, (batch_size, num_draft_tokens - 1), device=device)
        parent_list = torch.zeros(batch_size, topk, dtype=torch.long, device=device)
        seq_lens = torch.tensor([10, 15], dtype=torch.long, device=device)
        
        accepted_tokens, accept_length = worker.verify(
            verified_id=verified_id,
            draft_tokens=draft_tokens,
            parent_list=parent_list,
            seq_lens=seq_lens,
            temperature=0.0,
        )
        
        assert accepted_tokens.shape[0] == batch_size
        assert accept_length.shape == (batch_size,)
        assert (accept_length >= 0).all()
        assert (accept_length <= num_draft_tokens).all()
        
        print(f"✓ Verify测试通过")
        print(f"  Accepted tokens shape: {accepted_tokens.shape}")
        print(f"  Accept lengths: {accept_length}")


class TestGDNSpeculativeWorker:
    """测试完整的GDN Speculative Decoding。"""
    
    def test_generate_step(self, simple_gdn_models, device):
        """测试一步speculative decoding。"""
        draft_model, target_model = simple_gdn_models
        
        worker = GDNSpeculativeWorker(
            draft_model=draft_model,
            target_model=target_model,
            topk=4,
            num_steps=2,
            device=device,
        )
        
        batch_size = 2
        seq_len = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        verified_id = input_ids[:, -1]
        seq_lens = torch.tensor([seq_len, seq_len], dtype=torch.long, device=device)
        
        accepted_tokens, accept_length = worker.generate_step(
            input_ids=input_ids,
            verified_id=verified_id,
            seq_lens=seq_lens,
            temperature=0.0,
        )
        
        assert accepted_tokens.shape[0] == batch_size
        assert accept_length.shape == (batch_size,)
        
        # 获取统计
        stats = worker.get_statistics()
        
        print(f"✓ Speculative decoding步骤测试通过")
        print(f"  Accepted tokens shape: {accepted_tokens.shape}")
        print(f"  Accept lengths: {accept_length}")
        print(f"  Statistics: {stats}")
    
    def test_multi_step_generation(self, simple_gdn_models, device):
        """测试多步生成。"""
        draft_model, target_model = simple_gdn_models
        
        worker = GDNSpeculativeWorker(
            draft_model=draft_model,
            target_model=target_model,
            topk=4,
            num_steps=2,
            device=device,
        )
        
        worker.reset_stats()
        
        batch_size = 1
        max_new_tokens = 20
        input_ids = torch.randint(0, 1000, (batch_size, 10), device=device)
        
        generated_tokens = input_ids.clone()
        seq_lens = torch.tensor([input_ids.size(1)], dtype=torch.long, device=device)
        
        for step in range(max_new_tokens // 5):  # 每次可能生成多个tokens
            verified_id = generated_tokens[:, -1]
            
            accepted_tokens, accept_length = worker.generate_step(
                input_ids=generated_tokens,
                verified_id=verified_id,
                seq_lens=seq_lens,
                temperature=0.0,
            )
            
            # 添加新生成的tokens（跳过第一个verified token）
            new_tokens = accepted_tokens[:, 1:1+accept_length[0]]
            generated_tokens = torch.cat([generated_tokens, new_tokens], dim=1)
            seq_lens += accept_length
            
            if generated_tokens.size(1) >= input_ids.size(1) + max_new_tokens:
                break
        
        stats = worker.get_statistics()
        
        print(f"\n✓ 多步生成测试通过")
        print(f"  原始序列长度: {input_ids.size(1)}")
        print(f"  最终序列长度: {generated_tokens.size(1)}")
        print(f"  生成的新tokens: {generated_tokens.size(1) - input_ids.size(1)}")
        print(f"  总步数: {stats['total_steps']}")
        print(f"  平均acceptance rate: {stats['mean_acceptance_rate']:.2%}")
        print(f"  Speedup ratio: {stats['speedup_ratio']:.2f}x")


def test_import():
    """测试导入。"""
    from aiter.ops.triton._triton_kernels.gdn_block_sglang import Qwen3GatedDeltaNet
    from aiter.ops.triton._triton_kernels.gdr_sglang import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule_update,
    )
    print("✓ Import测试通过")


if __name__ == "__main__":
    # 运行测试
    print("=" * 60)
    print("GDN Speculative Decoding - Tests")
    print("=" * 60)
    
    # 检查设备
    if torch.cuda.is_available():
        if hasattr(torch.version, 'hip') and torch.version.hip:
            print(f"Device: AMD GPU (ROCm {torch.version.hip})")
        else:
            print("Device: NVIDIA GPU (CUDA)")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: No CUDA device available, tests will be skipped")
    
    print("=" * 60)
    
    # 运行pytest
    pytest.main([__file__, "-v", "-s"])

