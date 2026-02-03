"""
Qwen3GatedDeltaNet - Linear Attention Layer Implementation

This module implements the Gated Delta Rule Network layer as used in Qwen3-Next models.
It provides O(n) linear complexity attention as an alternative to O(n²) self-attention.

Adapted from SGLang's implementation for aiter.

Author: AIter Team
License: Apache 2.0
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..gdr_sglang import (
    chunk_gated_delta_rule,
    fused_recurrent_gated_delta_rule_update,
    fused_sigmoid_gating_delta_rule_update,
    fused_gdn_gating,
)
from .causal_conv1d_triton import causal_conv1d_fn, causal_conv1d_update
from .gdn_attn_backend import GDNAttnBackend


class Qwen3GatedDeltaNet(nn.Module):
    """
    Gated Delta Network (GDN) layer for linear attention.
    
    This layer implements a linear attention mechanism using gated delta rule,
    achieving O(n) complexity instead of O(n²) for standard attention.
    
    Architecture:
        1. Input projection: hidden → (Q, K, V, Z, a, b)
        2. 1D Convolution: temporal sequence processing
        3. GDN computation: chunk-based (prefill) or recurrent (decode)
        4. Gated normalization: RMSNorm with Z gating
        5. Output projection: value → hidden
    
    Args:
        hidden_size (int): Hidden dimension size
        num_k_heads (int): Number of key/query heads
        num_v_heads (int): Number of value heads
        head_k_dim (int): Key/query head dimension
        head_v_dim (int): Value head dimension
        conv_kernel_size (int): 1D convolution kernel size
        rms_norm_eps (float): RMS normalization epsilon
        dtype (torch.dtype): Data type for computations
        device (torch.device): Device for computation
        use_triton_conv1d (bool): Use Triton implementation for conv1d (default: True)
            True: Use optimized Triton kernels (causal_conv1d_fn/causal_conv1d_update)
            False: Use standard PyTorch Conv1d
        
    Example:
        >>> import torch
        >>> from aiter.layers import Qwen3GatedDeltaNet
        >>> 
        >>> # Create layer
        >>> layer = Qwen3GatedDeltaNet(
        ...     hidden_size=2048,
        ...     num_k_heads=32,
        ...     num_v_heads=32,
        ...     head_k_dim=64,
        ...     head_v_dim=64,
        ...     conv_kernel_size=4,
        ... )
        >>> 
        >>> # Forward pass
        >>> hidden_states = torch.randn(8, 1024, 2048, device='cuda')
        >>> output = layer(hidden_states, mode='chunk')  # prefill
        >>> print(f"Output shape: {output.shape}")
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_k_heads: int,
        num_v_heads: int,
        head_k_dim: int,
        head_v_dim: int,
        conv_kernel_size: int = 4,
        rms_norm_eps: float = 1e-6,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
        use_triton_conv1d: bool = True,
    ):
        super().__init__()
        
        # Configuration
        self.hidden_size = hidden_size
        self.num_k_heads = num_k_heads
        self.num_v_heads = num_v_heads
        self.head_k_dim = head_k_dim
        self.head_v_dim = head_v_dim
        self.key_dim = head_k_dim * num_k_heads
        self.value_dim = head_v_dim * num_v_heads
        self.conv_kernel_size = conv_kernel_size
        self.rms_norm_eps = rms_norm_eps
        self.use_triton_conv1d = use_triton_conv1d
        
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 1. Input projection layers
        # Projects hidden state to Q, K, V, Z (gating), a (gate param), b (beta param)
        projection_size_qkvz = self.key_dim * 2 + self.value_dim * 2
        projection_size_ba = self.num_v_heads * 2
        
        self.in_proj_qkvz = nn.Linear(
            hidden_size, projection_size_qkvz, bias=False, dtype=dtype, device=device
        )
        self.in_proj_ba = nn.Linear(
            hidden_size, projection_size_ba, bias=False, dtype=dtype, device=device
        )
        
        # 2. 1D Convolution for temporal sequence processing
        conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_dim = conv_dim
        
        if use_triton_conv1d:
            # Use Triton implementation: only need weight and bias parameters
            self.conv1d_weight = nn.Parameter(
                torch.randn(conv_dim, conv_kernel_size, dtype=dtype, device=device)
            )
            self.conv1d_bias = nn.Parameter(
                torch.zeros(conv_dim, dtype=dtype, device=device)
            )
            self.conv1d = None  # No nn.Conv1d module
        else:
            # Use standard PyTorch Conv1d
            self.conv1d = nn.Conv1d(
                in_channels=conv_dim,
                out_channels=conv_dim,
                kernel_size=conv_kernel_size,
                groups=conv_dim,  # Depthwise convolution
                bias=False,
                padding=conv_kernel_size - 1,
                dtype=dtype,
                device=device,
            )
            self.conv1d_weight = None
            self.conv1d_bias = None
        
        # 3. Learnable gating parameters
        # A_log: log-space decay parameter (controls forgetting)
        # dt_bias: time step bias
        A_init = torch.empty(num_v_heads, dtype=torch.float32, device=device).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A_init))
        self.A_log._no_weight_decay = True  # Mark for optimizer
        
        self.dt_bias = nn.Parameter(torch.ones(num_v_heads, dtype=dtype, device=device))
        
        # 4. Gated RMS Normalization
        # norm_before_gate=True: norm(x) * sigmoid(z)
        self.norm = RMSNormGated(
            head_v_dim, 
            eps=rms_norm_eps, 
            norm_before_gate=True,
            dtype=dtype,
            device=device,
        )
        
        # 5. Output projection
        self.out_proj = nn.Linear(
            self.value_dim, hidden_size, bias=False, dtype=dtype, device=device
        )
        
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_backend: Optional['GDNAttnBackend'] = None,
        conv_state: Optional[torch.Tensor] = None,
        ssm_state: Optional[torch.Tensor] = None,
        cache_indices: Optional[torch.Tensor] = None,
        query_start_loc: Optional[torch.Tensor] = None,
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
        Forward pass of Qwen3GatedDeltaNet.
        
        Reference: SGLang Qwen3GatedDeltaNet._forward (qwen3_next.py, lines 391-473)
        
        Args:
            hidden_states: Input tensor [seq_len, hidden_size] or [batch, seq_len, hidden_size]
            attn_backend: GDN attention backend (required for inference with cache)
            conv_state: Convolution state cache [cache_size, conv_dim, kernel_size-1]
            ssm_state: SSM state cache [cache_size, num_v_heads, head_k_dim, head_v_dim]
            cache_indices: Cache indices [batch]
            query_start_loc: Cumulative sequence lengths [batch+1]
            has_initial_state: Flags for initial state [batch]
            seq_lens_cpu: Sequence lengths on CPU [batch]
            use_qk_l2norm: Apply L2 normalization to Q and K
            is_target_verify: Whether in target verification mode
            intermediate_state_cache: Intermediate SSM states
            intermediate_conv_window_cache: Intermediate conv windows
            retrieve_next_token: Token retrieval indices [batch, draft_token_num]
            retrieve_next_sibling: Sibling retrieval indices [batch, draft_token_num]
            retrieve_parent_token: Parent retrieval indices [batch, draft_token_num]
            draft_token_num: Number of draft tokens
            
        Returns:
            output: [seq_len, hidden_size] or [batch, seq_len, hidden_size]
        """
        seq_len = hidden_states.shape[0] if hidden_states.dim() == 2 else hidden_states.shape[1]
        
        # 1. Input projection
        projected_qkvz = self.in_proj_qkvz(hidden_states)
        projected_ba = self.in_proj_ba(hidden_states)
        
        # 2. Split projections
        key_split_dim = self.key_dim
        value_split_dim = self.value_dim
        
        # Extract mixed_qkv (Q, K, V concatenated)
        mixed_qkv = projected_qkvz[..., :key_split_dim * 2 + value_split_dim]
        # Extract z (gating signal)
        z = projected_qkvz[..., key_split_dim * 2 + value_split_dim:]
        
        # Extract a and b
        b, a = torch.split(projected_ba, [self.num_v_heads, self.num_v_heads], dim=-1)
        
        # 3. Call backend
        is_decode = seq_len == 1
        
        kwargs = {
            "mixed_qkv": mixed_qkv,
            "conv_weights": self.conv1d_weight,
            "bias": self.conv1d_bias,
            "activation": None,
            "key_dim": self.key_dim,
            "value_dim": self.value_dim,
            "attention_tp_size": 1,
            "head_k_dim": self.head_k_dim,
            "head_v_dim": self.head_v_dim,
            "a": a,
            "b": b,
            "A_log": self.A_log,
            "dt_bias": self.dt_bias,
            "conv_state": conv_state,
            "ssm_state": ssm_state,
            "cache_indices": cache_indices,
            "query_start_loc": query_start_loc,
            "use_qk_l2norm": use_qk_l2norm,
        }
        
        if is_decode:
            core_attn_out = attn_backend.forward_decode(**kwargs)
        else:
            kwargs.update({
                "seq_len": seq_len,
                "has_initial_state": has_initial_state,
                "seq_lens_cpu": seq_lens_cpu,
                "is_target_verify": is_target_verify,
                "intermediate_state_cache": intermediate_state_cache,
                "intermediate_conv_window_cache": intermediate_conv_window_cache,
                "retrieve_next_token": retrieve_next_token,
                "retrieve_next_sibling": retrieve_next_sibling,
                "retrieve_parent_token": retrieve_parent_token,
                "draft_token_num": draft_token_num,
            })
            core_attn_out = attn_backend.forward_extend(**kwargs)
        
        # 4. Gated normalization
        original_shape = core_attn_out.shape
        output_flat = core_attn_out.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        output_flat = self.norm(output_flat, z_flat)
        
        # Reshape back and flatten heads
        core_attn_out = output_flat.reshape(original_shape)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)
        
        # 5. Output projection
        output = self.out_proj(core_attn_out)
        
        return output


class RMSNormGated(nn.Module):
    """
    RMS Normalization with gating.
    
    Computes: norm(x) * sigmoid(gate)
    where norm(x) = x / sqrt(mean(x^2) + eps) * weight
    
    Args:
        normalized_shape: Size of the input dimension to normalize
        eps: Small value to avoid division by zero
        norm_before_gate: If True, apply norm before gate; else gate before norm
        dtype: Data type
        device: Device
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-6,
        norm_before_gate: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = None,
    ):
        super().__init__()
        self.eps = eps
        self.norm_before_gate = norm_before_gate
        self.weight = nn.Parameter(torch.ones(normalized_shape, dtype=dtype, device=device))
    
    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [..., normalized_shape]
            gate: Gate tensor [..., normalized_shape]
            
        Returns:
            Gated normalized tensor
        """
        # RMS Normalization
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + self.eps) * self.weight
        
        # Apply gating
        if self.norm_before_gate:
            # norm(x) * sigmoid(gate)
            return x_normed * torch.sigmoid(gate)
        else:
            # norm(x * sigmoid(gate))
            x_gated = x * torch.sigmoid(gate)
            variance = x_gated.pow(2).mean(-1, keepdim=True)
            return x_gated * torch.rsqrt(variance + self.eps) * self.weight

