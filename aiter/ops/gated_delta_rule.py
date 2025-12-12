"""
Gated Delta Rule (GDN) operations for linear attention.

This module provides three implementations optimized for different scenarios:
1. chunk_gated_delta_rule: Parallel chunk-based for prefill (long sequences)
2. fused_recurrent_gated_delta_rule: Recurrent for decode with advanced features
3. fused_sigmoid_gating_delta_rule_update: Fully fused for single-step decode

Adapted from SGLang and Flash Linear Attention.

Author: AIter Team
License: Apache 2.0
"""

from typing import Optional, Tuple

import torch

from aiter.ops.triton._triton_kernels.gdr_sglang.chunk import (
    chunk_gated_delta_rule as _chunk_gated_delta_rule,
)
from aiter.ops.triton._triton_kernels.gdr_sglang.fused_recurrent import (
    fused_recurrent_gated_delta_rule as _fused_recurrent_gated_delta_rule,
    fused_recurrent_gated_delta_rule_update as _fused_recurrent_update,
)
from aiter.ops.triton._triton_kernels.gdr_sglang.fused_sigmoid_gating_recurrent import (
    fused_sigmoid_gating_delta_rule_update as _fused_sigmoid_gating,
)
from aiter.ops.triton._triton_kernels.gdr_sglang.fused_gdn_gating import (
    fused_gdn_gating,
)

__all__ = [
    "chunk_gated_delta_rule",
    "fused_recurrent_gated_delta_rule",
    "fused_recurrent_gated_delta_rule_update",
    "fused_sigmoid_gating_delta_rule_update",
    "fused_gdn_gating",
    "GatedDeltaRuleOp",
]


# ============================================================================
# 1. Chunk Implementation (for Prefill)
# ============================================================================

def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Chunk-based Gated Delta Rule for parallel processing of long sequences.
    
    Best for: Prefill phase with long sequences (T > 128).
    
    Algorithm:
        Uses WY representation to decompose the sequence into chunks and process
        them in parallel. Each chunk maintains its own state and the final states
        are propagated across chunks.
    
    Complexity:
        - Time: O(T) with large constants (~6-7 kernel launches)
        - Space: O(T * chunk_size^2) for intermediate matrices
        - Parallelism: High (chunk-level parallelism)
    
    Args:
        q (torch.Tensor): Query tensor of shape [B, T, H, K] or [B, H, T, K] if head_first
        k (torch.Tensor): Key tensor of shape [B, T, H, K]
        v (torch.Tensor): Value tensor of shape [B, T, H, V]
        g (torch.Tensor): Gate tensor (in log space!) of shape [B, T, H]
        beta (torch.Tensor): Beta gate tensor of shape [B, T, H]
        scale (Optional[float]): Attention scale factor. Default: 1/sqrt(K)
        initial_state (Optional[torch.Tensor]): Initial hidden state [N, H, K, V]
        output_final_state (bool): Whether to return final state
        cu_seqlens (Optional[torch.LongTensor]): Cumulative sequence lengths [N+1] for variable-length inputs
        head_first (bool): Whether input is in head-first format [B, H, T, ...]
        use_qk_l2norm_in_kernel (bool): Apply L2 normalization to Q/K (for Qwen3-Next)
        
    Returns:
        output (torch.Tensor): Attention output [B, T, H, V]
        final_state (Optional[torch.Tensor]): Final hidden state [N, H, K, V] if output_final_state=True
        
    Example:
        >>> B, T, H, K, V = 4, 2048, 32, 64, 64
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> g = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').log()
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> o, state = chunk_gated_delta_rule(q, k, v, g, beta, output_final_state=True)
        >>> print(f"Output shape: {o.shape}, State shape: {state.shape}")
    """
    return _chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


# ============================================================================
# 2. Fused Recurrent Implementation (for Decode with Features)
# ============================================================================

def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Fused recurrent Gated Delta Rule for sequential processing.
    
    Best for: Decode phase with moderate sequences or speculative decoding.
    
    Algorithm:
        Single Triton kernel that processes tokens sequentially, maintaining
        a compact hidden state. All operations (decay, delta rule, update)
        are fused into one kernel.
    
    Complexity:
        - Time: O(T) with small constants (1 kernel launch)
        - Space: O(K*V) for hidden state only
        - Parallelism: Medium (head-level parallelism)
    
    Args:
        q (torch.Tensor): Query tensor [B, T, H, K]
        k (torch.Tensor): Key tensor [B, T, H, K]
        v (torch.Tensor): Value tensor [B, T, HV, V] (supports GVA when HV > H)
        g (torch.Tensor): Gate tensor (decay) [B, T, HV]
        beta (torch.Tensor): Beta gate tensor [B, T, HV]
        scale (Optional[float]): Attention scale factor. Default: 1/sqrt(K)
        initial_state (Optional[torch.Tensor]): Initial hidden state [N, HV, K, V]
        output_final_state (bool): Whether to return final state
        cu_seqlens (Optional[torch.LongTensor]): Cumulative sequence lengths
        use_qk_l2norm_in_kernel (bool): Apply L2 normalization to Q/K
        
    Returns:
        output (torch.Tensor): Attention output [B, T, HV, V]
        final_state (Optional[torch.Tensor]): Final hidden state [N, HV, K, V]
        
    Example:
        >>> B, T, H, K, V = 2, 64, 16, 64, 64
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> g = torch.rand(B, T, H, dtype=torch.float32, device='cuda') * (-5.0)
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> o, state = fused_recurrent_gated_delta_rule(
        ...     q, k, v, g, beta, output_final_state=True
        ... )
    """
    return _fused_recurrent_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def fused_recurrent_gated_delta_rule_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    disable_state_update: bool = False,
    disable_output_calculation: bool = False,
    intermediate_states_buffer: Optional[torch.Tensor] = None,
    cache_steps: Optional[int] = None,
    retrieve_parent_token: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Advanced fused recurrent update with state caching and tree attention.
    
    Best for: Speculative decoding, EAGLE tree decoding.
    
    Features:
        - State caching: Cache intermediate states for multiple candidate paths
        - Tree attention: Support parent token retrieval for tree-based decoding
        - Conditional updates: Optionally disable state updates or output calculation
        
    This is the most feature-rich implementation, designed for advanced inference
    scenarios like speculative decoding where multiple candidate sequences need to
    be maintained simultaneously.
    
    Args:
        q, k, v, g, beta: Same as fused_recurrent_gated_delta_rule
        initial_state_source (torch.Tensor): State pool [num_states, HV, K, V]
        initial_state_indices (torch.Tensor): Indices into state pool [N]
        cu_seqlens (Optional[torch.Tensor]): Cumulative sequence lengths
        scale (Optional[float]): Attention scale factor
        use_qk_l2norm_in_kernel (bool): Apply L2 normalization
        disable_state_update (bool): Skip state updates (for verification phase)
        disable_output_calculation (bool): Skip output computation (state-only mode)
        intermediate_states_buffer (Optional[torch.Tensor]): Buffer for caching intermediate states
        cache_steps (Optional[int]): Number of steps to cache
        retrieve_parent_token (Optional[torch.Tensor]): Parent token indices for tree attention
        
    Returns:
        output (torch.Tensor): Attention output [B, T, HV, V]
        
    Note:
        The state is updated in-place in initial_state_source at the positions
        specified by initial_state_indices.
        
    Example (Speculative Decoding):
        >>> # Setup state pool for multiple candidates
        >>> num_candidates = 10
        >>> state_pool = torch.randn(num_candidates, H, K, V, dtype=torch.float32, device='cuda')
        >>> 
        >>> # Process draft tokens
        >>> batch_size = 2
        >>> draft_tokens = 4
        >>> state_indices = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
        >>> 
        >>> # ... prepare q, k, v, g, beta for draft tokens ...
        >>> 
        >>> o = fused_recurrent_gated_delta_rule_update(
        ...     q, k, v, g, beta,
        ...     initial_state_source=state_pool,
        ...     initial_state_indices=state_indices,
        ...     intermediate_states_buffer=cache_buffer,
        ...     cache_steps=draft_tokens,
        ... )
    """
    return _fused_recurrent_update(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        cu_seqlens=cu_seqlens,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        disable_state_update=disable_state_update,
        disable_output_calculation=disable_output_calculation,
        intermediate_states_buffer=intermediate_states_buffer,
        cache_steps=cache_steps,
        retrieve_parent_token=retrieve_parent_token,
    )


# ============================================================================
# 3. Fused Sigmoid Gating Implementation (for Single-Step Decode)
# ============================================================================

def fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    dt_bias: torch.Tensor,
    softplus_beta: float,
    softplus_threshold: float,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    initial_state_source: torch.Tensor,
    initial_state_indices: torch.Tensor,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fully fused sigmoid gating with delta rule for maximum efficiency.
    
    Best for: Single-step autoregressive decoding.
    
    This implementation achieves the lowest latency by fusing:
    1. Gate computation: g = -exp(A_log) * softplus(a + dt_bias)
    2. Beta computation: beta = sigmoid(b)
    3. Delta rule update: h *= exp(g); v -= h^T @ k; v *= beta; h += k ⊗ v
    4. Output computation: o = h @ q
    
    All in a single Triton kernel, minimizing memory traffic and kernel launch overhead.
    
    Complexity:
        - Time: O(1) for single token (1 kernel launch)
        - Space: O(K*V) for hidden state
        - Performance: Lowest latency among all implementations
    
    Args:
        A_log (torch.Tensor): Log gate parameter [HV]
        a (torch.Tensor): Gate input [B, T, HV]
        dt_bias (torch.Tensor): Time step bias [HV]
        softplus_beta (float): Softplus beta parameter (typically 1.0)
        softplus_threshold (float): Softplus threshold for numerical stability (typically 20.0)
        q (torch.Tensor): Query tensor [1, T, H, K] (typically T=1)
        k (torch.Tensor): Key tensor [1, T, H, K]
        v (torch.Tensor): Value tensor [1, T, HV, V]
        b (torch.Tensor): Beta input [B, T, HV]
        initial_state_source (torch.Tensor): State pool [num_states, HV, K, V]
        initial_state_indices (torch.Tensor): Indices into state pool [N]
        scale (Optional[float]): Attention scale factor
        use_qk_l2norm_in_kernel (bool): Apply L2 normalization
        cu_seqlens (Optional[torch.Tensor]): Cumulative sequence lengths
        
    Returns:
        output (torch.Tensor): Attention output [1, T, HV, V]
        
    Note:
        The state is updated in-place in initial_state_source.
        
    Example:
        >>> # Single-step decode for Qwen3-Next style models
        >>> H = 32
        >>> K, V = 64, 64
        >>> batch_size = 2
        >>> 
        >>> # Model parameters
        >>> A_log = torch.rand(H, dtype=torch.float32, device='cuda')
        >>> dt_bias = torch.rand(H, dtype=torch.bfloat16, device='cuda')
        >>> 
        >>> # Current token inputs
        >>> a = torch.randn(batch_size, H, dtype=torch.bfloat16, device='cuda')
        >>> b = torch.randn(batch_size, H, dtype=torch.bfloat16, device='cuda')
        >>> 
        >>> # Query, key, value for current token
        >>> q = torch.randn(1, 1, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = torch.randn(1, 1, H, K, dtype=torch.bfloat16, device='cuda')
        >>> v = torch.randn(1, 1, H, V, dtype=torch.bfloat16, device='cuda')
        >>> 
        >>> # State management
        >>> state_pool = torch.randn(100, H, K, V, dtype=torch.float32, device='cuda')
        >>> state_indices = torch.tensor([5, 10], dtype=torch.int32, device='cuda')
        >>> query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device='cuda')
        >>> 
        >>> # Single-step decode
        >>> output = fused_sigmoid_gating_delta_rule_update(
        ...     A_log=A_log, a=a, dt_bias=dt_bias,
        ...     softplus_beta=1.0, softplus_threshold=20.0,
        ...     q=q, k=k, v=v, b=b,
        ...     initial_state_source=state_pool,
        ...     initial_state_indices=state_indices,
        ...     cu_seqlens=query_start_loc,
        ...     use_qk_l2norm_in_kernel=True,
        ... )
    """
    return _fused_sigmoid_gating(
        A_log=A_log, a=a, dt_bias=dt_bias,
        softplus_beta=softplus_beta,
        softplus_threshold=softplus_threshold,
        q=q, k=k, v=v, b=b,
        initial_state_source=initial_state_source,
        initial_state_indices=initial_state_indices,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        cu_seqlens=cu_seqlens,
    )


# ============================================================================
# 4. Unified Operator Class
# ============================================================================

class GatedDeltaRuleOp:
    """
    Unified interface for Gated Delta Rule operations.
    
    Automatically selects the best implementation based on context and
    provides a consistent API for all use cases.
    
    Usage:
        >>> # Automatic mode selection
        >>> output, state = GatedDeltaRuleOp.forward(
        ...     q, k, v, g, beta, mode="auto"
        ... )
        
        >>> # Force specific implementation
        >>> output, _ = GatedDeltaRuleOp.forward(
        ...     q, k, v, g, beta, mode="chunk"
        ... )
    """
    
    @staticmethod
    def forward(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        mode: str = "auto",
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Unified forward pass with automatic implementation selection.
        
        Args:
            q, k, v, g, beta: Input tensors
            mode (str): Implementation mode
                - "auto": Automatically select based on sequence length and context
                - "chunk": Force chunk implementation (best for prefill)
                - "recurrent": Force recurrent implementation (best for moderate decode)
                - "sigmoid": Force sigmoid implementation (best for single-step)
            **kwargs: Additional arguments passed to the selected implementation
            
        Returns:
            output (torch.Tensor): Attention output
            final_state (Optional[torch.Tensor]): Final state if requested
            
        Example:
            >>> # Let the system choose the best implementation
            >>> o, state = GatedDeltaRuleOp.forward(
            ...     q, k, v, g, beta,
            ...     mode="auto",
            ...     output_final_state=True,
            ...     use_qk_l2norm_in_kernel=True
            ... )
        """
        seq_len = q.shape[1] if q.ndim >= 2 else 1
        
        # Automatic mode selection
        if mode == "auto":
            # Check for sigmoid mode indicators
            has_sigmoid_params = "A_log" in kwargs and "a" in kwargs and "b" in kwargs
            
            if seq_len > 128:
                # Long sequences: use chunk for parallel processing
                mode = "chunk"
            elif seq_len == 1 and has_sigmoid_params:
                # Single token with gate parameters: use fused sigmoid
                mode = "sigmoid"
            else:
                # Default to recurrent for flexibility
                mode = "recurrent"
        
        # Call the selected implementation
        if mode == "chunk":
            return chunk_gated_delta_rule(q, k, v, g, beta, **kwargs)
        
        elif mode == "recurrent":
            # Check if this is an update call (with state pool)
            if "initial_state_source" in kwargs and "initial_state_indices" in kwargs:
                output = fused_recurrent_gated_delta_rule_update(
                    q, k, v, g, beta, **kwargs
                )
                return output, None
            else:
                return fused_recurrent_gated_delta_rule(q, k, v, g, beta, **kwargs)
        
        elif mode == "sigmoid":
            if not all(k in kwargs for k in ["A_log", "a", "b", "dt_bias", 
                                              "initial_state_source", "initial_state_indices"]):
                raise ValueError(
                    "Sigmoid mode requires: A_log, a, b, dt_bias, "
                    "initial_state_source, initial_state_indices"
                )
            
            # Extract sigmoid-specific parameters
            sigmoid_kwargs = {
                "A_log": kwargs.pop("A_log"),
                "a": kwargs.pop("a"),
                "b": kwargs.pop("b"),
                "dt_bias": kwargs.pop("dt_bias"),
                "softplus_beta": kwargs.pop("softplus_beta", 1.0),
                "softplus_threshold": kwargs.pop("softplus_threshold", 20.0),
                "initial_state_source": kwargs.pop("initial_state_source"),
                "initial_state_indices": kwargs.pop("initial_state_indices"),
            }
            # Add remaining kwargs
            sigmoid_kwargs.update(kwargs)
            
            output = fused_sigmoid_gating_delta_rule_update(
                q=q, k=k, v=v, **sigmoid_kwargs
            )
            return output, None
        
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Choose from 'auto', 'chunk', 'recurrent', 'sigmoid'"
            )
    
    @staticmethod
    def get_recommended_mode(seq_len: int, has_gate_params: bool = False) -> str:
        """
        Get the recommended implementation mode based on context.
        
        Args:
            seq_len (int): Sequence length
            has_gate_params (bool): Whether gate parameters (A_log, a, b) are available
            
        Returns:
            str: Recommended mode ("chunk", "recurrent", or "sigmoid")
        """
        if seq_len > 128:
            return "chunk"
        elif seq_len == 1 and has_gate_params:
            return "sigmoid"
        else:
            return "recurrent"


# ============================================================================
# Convenience Functions
# ============================================================================

def compute_gating_params(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute g and beta from raw gate parameters.
    
    This is useful when you have the raw parameters but want to use
    the chunk or recurrent implementation that expects pre-computed g and beta.
    
    Args:
        A_log (torch.Tensor): Log gate parameter [HV]
        a (torch.Tensor): Gate input [B, T, HV]
        b (torch.Tensor): Beta input [B, T, HV]
        dt_bias (torch.Tensor): Time step bias [HV]
        
    Returns:
        g (torch.Tensor): Computed gate [B, T, HV]
        beta (torch.Tensor): Computed beta [B, T, HV]
        
    Example:
        >>> A_log = torch.rand(32, dtype=torch.float32, device='cuda')
        >>> a = torch.randn(4, 128, 32, dtype=torch.bfloat16, device='cuda')
        >>> b = torch.randn(4, 128, 32, dtype=torch.bfloat16, device='cuda')
        >>> dt_bias = torch.rand(32, dtype=torch.bfloat16, device='cuda')
        >>> 
        >>> g, beta = compute_gating_params(A_log, a, b, dt_bias)
        >>> 
        >>> # Now use with chunk or recurrent
        >>> output, _ = chunk_gated_delta_rule(q, k, v, g, beta)
    """
    return fused_gdn_gating(A_log, a, b, dt_bias)

