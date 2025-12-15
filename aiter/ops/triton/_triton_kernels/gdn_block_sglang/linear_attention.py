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
        
    def _project_inputs(
        self, 
        hidden_states: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Project hidden states to Q, K, V, Z, a, b.
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            query: [batch, seq_len, num_k_heads, head_k_dim]
            key: [batch, seq_len, num_k_heads, head_k_dim]
            value: [batch, seq_len, num_v_heads, head_v_dim]
            z: [batch, seq_len, num_v_heads, head_v_dim] - gating signal
            a: [batch, seq_len, num_v_heads] - gate parameter
            b: [batch, seq_len, num_v_heads] - beta parameter
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to QKVZ
        projected_qkvz = self.in_proj_qkvz(hidden_states)  # [B, T, 2*key_dim + 2*value_dim]
        
        # Project to ba
        projected_ba = self.in_proj_ba(hidden_states)  # [B, T, 2*num_v_heads]
        
        # Split QKVZ
        # Reshape to [B, T, num_k_heads, dims_per_head]
        projected_qkvz = projected_qkvz.view(
            batch_size, seq_len, self.num_k_heads,
            self.head_k_dim * 2 + (self.head_v_dim * 2 * self.num_v_heads // self.num_k_heads)
        )
        
        # Split into Q, K, V, Z
        splits = [
            self.head_k_dim,  # Q
            self.head_k_dim,  # K
            self.head_v_dim * self.num_v_heads // self.num_k_heads,  # V
            self.head_v_dim * self.num_v_heads // self.num_k_heads,  # Z
        ]
        query, key, value, z = torch.split(projected_qkvz, splits, dim=-1)
        
        # Reshape V and Z to have separate head dimension
        value = value.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        z = z.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        
        # Split ba
        projected_ba = projected_ba.view(
            batch_size, seq_len, self.num_k_heads, 
            2 * self.num_v_heads // self.num_k_heads
        )
        b, a = torch.split(
            projected_ba, 
            [self.num_v_heads // self.num_k_heads] * 2, 
            dim=-1
        )
        
        # Reshape a and b
        b = b.reshape(batch_size, seq_len, self.num_v_heads)
        a = a.reshape(batch_size, seq_len, self.num_v_heads)
        
        return query, key, value, z, a, b
    
    def _apply_conv1d(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        conv_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply 1D convolution to Q, K, V.
        
        Args:
            query, key, value: [batch, seq_len, num_heads, head_dim]
            conv_state: Optional cached state for inference
            
        Returns:
            Convolved query, key, value and updated conv_state
        """
        batch_size, seq_len, num_heads, head_dim = query.shape
        
        # Flatten heads: [B, T, H, D] → [B, T, H*D]
        query_flat = query.reshape(batch_size, seq_len, -1)
        key_flat = key.reshape(batch_size, seq_len, -1)
        value_flat = value.reshape(batch_size, seq_len, -1)
        
        # Concatenate: [B, T, key_dim*2 + value_dim]
        mixed = torch.cat([query_flat, key_flat, value_flat], dim=-1)
        
        # Transpose for Conv1d: [B, C, T]
        mixed = mixed.transpose(1, 2)
        
        # Apply convolution
        mixed = self.conv1d(mixed)
        
        # Remove extra padding
        if self.conv_kernel_size > 1:
            mixed = mixed[:, :, :seq_len]
        
        # Transpose back: [B, T, C]
        mixed = mixed.transpose(1, 2)
        
        # Split back
        query_flat, key_flat, value_flat = torch.split(
            mixed, 
            [self.key_dim, self.key_dim, self.value_dim], 
            dim=-1
        )
        
        # Reshape back to multi-head format
        query = query_flat.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        key = key_flat.reshape(batch_size, seq_len, self.num_k_heads, self.head_k_dim)
        value = value_flat.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        
        return query, key, value, None  # No conv_state for now
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        mode: str = "auto",
        initial_state: Optional[torch.Tensor] = None,
        output_final_state: bool = False,
        cu_seqlens: Optional[torch.Tensor] = None,
        use_qk_l2norm: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of Qwen3GatedDeltaNet.
        
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size]
            mode: Computation mode
                - "auto": Automatically select based on seq_len
                - "chunk": Use chunk-based parallel computation (prefill)
                - "recurrent": Use recurrent computation (decode)
                - "fused_decode": Use fully fused decode kernel (single-step)
            initial_state: Initial hidden state [batch, num_v_heads, head_k_dim, head_v_dim]
            output_final_state: Whether to return final state
            cu_seqlens: Cumulative sequence lengths for variable-length sequences
            use_qk_l2norm: Apply L2 normalization to Q and K (Qwen3-Next default: True)
            
        Returns:
            output: [batch, seq_len, hidden_size]
            final_state: [batch, num_v_heads, head_k_dim, head_v_dim] if output_final_state=True
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # 1. Project inputs
        query, key, value, z, a, b = self._project_inputs(hidden_states)
        
        # 2. Apply convolution
        query, key, value, _ = self._apply_conv1d(query, key, value)
        
        # 3. Compute gating parameters
        # g: forget gate (in log space)
        # beta: input gate
        g, beta = fused_gdn_gating(
            A_log=self.A_log,
            a=a.reshape(-1, self.num_v_heads),  # Flatten batch and seq
            b=b.reshape(-1, self.num_v_heads),
            dt_bias=self.dt_bias,
        )
        # Reshape back: [1, B*T, H] → [B, T, H]
        g = g.squeeze(0).reshape(batch_size, seq_len, self.num_v_heads)
        beta = beta.squeeze(0).reshape(batch_size, seq_len, self.num_v_heads)
        
        # 4. Select computation mode
        if mode == "auto":
            # Auto-select based on sequence length
            if seq_len == 1:
                mode = "fused_decode"
            elif seq_len > 128:
                mode = "chunk"
            else:
                mode = "recurrent"
        
        # 5. GDN computation
        if mode == "chunk":
            # Chunk-based parallel computation (prefill)
            output, final_state = chunk_gated_delta_rule(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state=initial_state,
                output_final_state=output_final_state,
                cu_seqlens=cu_seqlens,
                head_first=False,
                use_qk_l2norm_in_kernel=use_qk_l2norm,
            )
        elif mode == "recurrent":
            # Recurrent computation (short sequences or decode)
            # Note: This mode requires initial_state and indices for state management
            if initial_state is None:
                # Create zero initial state
                initial_state = torch.zeros(
                    batch_size, self.num_v_heads, self.head_k_dim, self.head_v_dim,
                    dtype=torch.float32, device=hidden_states.device
                )
            
            # Always provide indices for state management
            state_indices = torch.arange(batch_size, dtype=torch.int32, device=hidden_states.device)
            
            output = fused_recurrent_gated_delta_rule_update(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                initial_state_source=initial_state,
                initial_state_indices=state_indices,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm_in_kernel=use_qk_l2norm,
            )
            final_state = None
        elif mode == "fused_decode":
            # Fully fused single-step decode
            # Reshape for decode kernel expectations
            output = fused_sigmoid_gating_delta_rule_update(
                A_log=self.A_log,
                a=a.reshape(-1, self.num_v_heads),
                dt_bias=self.dt_bias,
                softplus_beta=1.0,
                softplus_threshold=20.0,
                q=query.unsqueeze(0) if query.dim() == 3 else query,  # Ensure [1, T, H, K]
                k=key.unsqueeze(0) if key.dim() == 3 else key,
                v=value.unsqueeze(0) if value.dim() == 3 else value,
                b=b.reshape(-1, self.num_v_heads),
                initial_state_source=initial_state,
                initial_state_indices=torch.arange(batch_size, device=hidden_states.device) if initial_state is not None else None,
                use_qk_l2norm_in_kernel=use_qk_l2norm,
                cu_seqlens=cu_seqlens,
            )
            final_state = None
            if output.dim() == 4:
                output = output.squeeze(0)  # Remove batch dim if added
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # 6. Gated normalization
        # Flatten for normalization
        output_flat = output.reshape(-1, self.head_v_dim)
        z_flat = z.reshape(-1, self.head_v_dim)
        
        # Apply gated RMSNorm: norm(output) * sigmoid(z)
        output_flat = self.norm(output_flat, z_flat)
        
        # Reshape back
        output = output_flat.reshape(batch_size, seq_len, self.num_v_heads, self.head_v_dim)
        
        # Flatten heads for output projection
        output = output.reshape(batch_size, seq_len, self.value_dim)
        
        # 7. Output projection
        output = self.out_proj(output)
        
        if output_final_state:
            return output, final_state
        return output, None


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

