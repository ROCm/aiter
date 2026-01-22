"""
GDN Attention Backend - Backend implementation for Gated Delta Network

This module provides the attention backend for GDN (Gated Delta Network) 
with support for decode and extend modes, compatible with SGLang's architecture.

Adapted from SGLang's GDNAttnBackend implementation for aiter.

Author: AIter Team
License: Apache 2.0
"""

from typing import Optional
import torch

from ..gdr_sglang import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update,
    fused_gdn_gating,
)
from .causal_conv1d_triton import causal_conv1d_fn, causal_conv1d_update


class GDNAttnBackend:
    """
    Attention backend for Gated Delta Network (GDN).
    
    This backend provides optimized forward_decode and forward_extend methods
    for efficient inference of GDN layers, following SGLang's design patterns.
    
    Reference: SGLang's GDNAttnBackend (hybrid_linear_attn_backend.py, lines 521-733)
    
    The backend handles:
    - Single-step decode (optimized for inference)
    - Multi-step extend/prefill (optimized for long sequences)
    - Speculative decoding with tree attention
    - Continuous batching with cache management
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
    
    def forward_decode(
        self,
        mixed_qkv: torch.Tensor,
        conv_weights: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation: Optional[str],
        key_dim: int,
        value_dim: int,
        attention_tp_size: int,
        head_k_dim: int,
        head_v_dim: int,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        use_qk_l2norm: bool = True,
    ) -> torch.Tensor:
        """
        Single-step decode forward pass.
        Reference: SGLang GDNAttnBackend.forward_decode (lines 524-596)
        
        Input: mixed_qkv shape is [l, h*d] where l is seq_len (batch_size in decode mode)
        Output: core_attn_out shape is [1, l, h, d]
        """
        # Conv1d update - SGLang line 555-562
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_state,
            conv_weights,
            bias,
            activation,
            conv_state_indices=cache_indices,
        )
        
        # Split Q, K, V - SGLang line 564-572
        query, key, value = torch.split(
            mixed_qkv,
            [
                key_dim // attention_tp_size,
                key_dim // attention_tp_size,
                value_dim // attention_tp_size,
            ],
            dim=-1,
        )
        
        # Reshape from [l, h*d] to [1, l, h, d] - SGLang line 573-578
        seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        query = query.view(1, seq_len, num_heads, head_k_dim)
        key = key.view(1, seq_len, num_heads, head_k_dim)
        value = value.view(1, seq_len, value.shape[1] // head_v_dim, head_v_dim)
        
        # Apply fused sigmoid gating delta rule - SGLang line 580-594
        core_attn_out = fused_sigmoid_gating_delta_rule_update(
            A_log=A_log,
            dt_bias=dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_state,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
        )
        
        return core_attn_out
    
    def forward_extend(
        self,
        mixed_qkv: torch.Tensor,
        conv_weights: torch.Tensor,
        bias: Optional[torch.Tensor],
        activation: Optional[str],
        key_dim: int,
        value_dim: int,
        attention_tp_size: int,
        head_k_dim: int,
        head_v_dim: int,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        seq_len: int,
        has_initial_state: Optional[torch.Tensor] = None,
        seq_lens_cpu: Optional[list] = None,
        use_qk_l2norm: bool = True,
        is_target_verify: bool = False,
        intermediate_state_cache: Optional[torch.Tensor] = None,
        intermediate_conv_window_cache: Optional[torch.Tensor] = None,
        retrieve_next_token: Optional[torch.Tensor] = None,
        retrieve_next_sibling: Optional[torch.Tensor] = None,
        retrieve_parent_token: Optional[torch.Tensor] = None,
        draft_token_num: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Multi-step extend/prefill forward pass.
        Reference: SGLang GDNAttnBackend.forward_extend (lines 598-733)
        """
        # Conv1d - SGLang line 649-679
        if is_target_verify:
            batch_size = seq_len // draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_state,
                conv_weights,
                bias,
                activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            if has_initial_state is None:
                has_initial_state = torch.zeros(len(cache_indices), dtype=torch.bool, device=mixed_qkv.device)
            
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv.transpose(0, 1),
                conv_weights,
                bias,
                activation=activation,
                conv_states=conv_state,
                has_initial_state=has_initial_state,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]
        
        # Split and reshape - SGLang line 681-696
        key_split_dim = key_dim // attention_tp_size
        value_split_dim = value_dim // attention_tp_size
        
        query, key, value = torch.split(
            mixed_qkv,
            [key_split_dim, key_split_dim, value_split_dim],
            dim=-1,
        )
        
        actual_seq_len = query.shape[0]
        num_heads = query.shape[1] // head_k_dim
        num_value_heads = value.shape[1] // head_v_dim
        
        query = query.view(1, actual_seq_len, num_heads, head_k_dim)
        key = key.view(1, actual_seq_len, num_heads, head_k_dim)
        value = value.view(1, actual_seq_len, num_value_heads, head_v_dim)
        
        # Gating - SGLang line 698
        g, beta = fused_gdn_gating(A_log, a, b, dt_bias)
        
        # GDN computation - SGLang line 700-733
        if is_target_verify:
            core_attn_out = fused_recurrent_gated_delta_rule_update(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state_source=ssm_state,
                initial_state_indices=cache_indices,
                cu_seqlens=query_start_loc,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                intermediate_states_buffer=intermediate_state_cache,
                cache_steps=draft_token_num,
                retrieve_parent_token=retrieve_parent_token,
            )
        else:
            recurrent_state = ssm_state[cache_indices]
            core_attn_out, last_recurrent_state = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=recurrent_state,
                output_final_state=True,
                cu_seqlens=query_start_loc,
                head_first=False,
                use_qk_l2norm_in_kernel=True,
            )
            last_recurrent_state = last_recurrent_state.to(ssm_state.dtype, copy=False)
            ssm_state[cache_indices] = last_recurrent_state
        
        return core_attn_out

