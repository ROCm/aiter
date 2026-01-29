# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Gated Delta Rule Operations (Forward Only).

This module provides high-level interfaces for gated delta rule computations,
including both fused recurrent and chunk-based implementations.

Important Note:
    Only forward pass is implemented in aiter. These functions do NOT support
    gradient computation or backward pass. For training with autograd, please
    use the flash-linear-attention library instead.

    These implementations are optimized for inference and forward-only operations.
"""

import torch
import triton

from aiter.ops.triton._triton_kernels.gated_delta_rule import (
    _fused_recurrent_gated_delta_rule_fwd_kernel,
    chunk_gated_delta_rule_fwd,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.utils import (
    l2norm_fwd,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fused_recurrent_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    gv: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Fused recurrent gated delta rule operation using Triton (Forward only).

    This function implements a recurrent gating mechanism with delta rule updates,
    optimized for GPU execution using Triton kernels. It supports variable-length
    sequences, initial/final states, and multiple gating options.

    Warning:
        This function only supports forward pass and does NOT compute gradients.
        Do not use this for training. For training, use flash-linear-attention library.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`.
            GVA is applied if `HV > H`.
        g (torch.Tensor, optional):
            g (decays) of shape `[B, T, HV]`. Default: `None`.
        gk (torch.Tensor, optional):
            gk (decays) of shape `[B, T, HV, K]`. Default: `None`.
        gv (torch.Tensor, optional):
            gv (decays) of shape `[B, T, HV, V]`. Default: `None`.
        beta (torch.Tensor, optional):
            betas of shape `[B, T, HV]` or `[B, T, HV, V]`.
            If None, defaults to ones. Default: `None`.
        scale (float, optional):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (torch.Tensor, optional):
            Initial state of shape `[N, HV, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (bool):
            Whether to output the final state of shape `[N, HV, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to use L2 normalization in the kernel. Default: `False`.
        cu_seqlens (torch.LongTensor, optional):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API. Default: `None`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - o (torch.Tensor): Outputs of shape `[B, T, HV, V]`.
            - final_state (torch.Tensor): Final state of shape `[N, HV, K, V]` if
              `output_final_state=True` else `None`.

    Examples:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from aiter.ops.triton.gated_delta_rule import fused_recurrent_gated_delta_rule
        >>> # inputs with equal lengths
        >>> B, T, H, HV, K, V = 4, 2048, 4, 8, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, HV, V, device='cuda')
        >>> g = F.logsigmoid(torch.rand(B, T, HV, device='cuda'))
        >>> beta = torch.rand(B, T, HV, device='cuda').sigmoid()
        >>> h0 = torch.randn(B, HV, K, V, device='cuda')
        >>> o, ht = fused_recurrent_gated_delta_rule(
        ...     q, k, v, g=g, beta=beta,
        ...     initial_state=h0,
        ...     output_final_state=True
        ... )
        >>> # for variable-length inputs, the batch size `B` is expected to be 1
        >>> # and `cu_seqlens` is required
        >>> from einops import rearrange
        >>> q, k, v, g, beta = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, g, beta))
        >>> # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = fused_recurrent_gated_delta_rule(
        ...     q, k, v, g=g, beta=beta,
        ...     initial_state=h0,
        ...     output_final_state=True,
        ...     cu_seqlens=cu_seqlens
        ... )
    """
    # Input validation
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )

    # Set default values
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if beta is None:
        beta = torch.ones_like(q[..., 0])

    # Extract dimensions
    B, T, H, K, V = *k.shape, v.shape[-1]
    HV = v.shape[2]
    N = B if cu_seqlens is None else len(cu_seqlens) - 1

    # Log operation
    _LOGGER.info(
        f"GATED_DELTA_RULE: q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}, "
        f"scale={scale}, use_g={g is not None}, use_gk={gk is not None}, use_gv={gv is not None}"
    )

    # Calculate block sizes
    BK = triton.next_power_of_2(K)
    BV = min(8, triton.next_power_of_2(V)) if gv is None else triton.next_power_of_2(V)
    NV = triton.cdiv(V, BV)

    # Prepare output tensors
    o = torch.empty_like(v)
    final_state = (
        q.new_empty(N, HV, K, V, dtype=torch.float32) if output_final_state else None
    )

    # Launch kernel
    grid = (NV, N * HV)
    _fused_recurrent_gated_delta_rule_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        g=g,
        gk=gk,
        gv=gv,
        beta=beta,
        o=o,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        scale=scale,
        T=T,
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        IS_BETA_HEADWISE=beta.ndim != v.ndim,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        num_warps=1,
        num_stages=3,
    )

    return o, final_state


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    Chunk-based gated delta rule operation using Triton (Forward only).

    This function implements chunk-based parallel computation for the gated delta rule,
    optimized for training and long sequences. It uses the native aiter implementation
    with Triton kernels.

    Warning:
        This function only supports forward pass and does NOT compute gradients.
        Do not use this for training. For training, use flash-linear-attention library.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            g (decays in log space) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (float, optional):
            Scale factor for the attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (torch.Tensor, optional):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (bool):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to use L2 normalization in the kernel. Default: `False`.
        cu_seqlens (torch.LongTensor, optional):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API. Default: `None`.

    Returns:
        tuple[torch.Tensor, torch.Tensor]:
            - o (torch.Tensor): Outputs of shape `[B, T, H, V]`.
            - final_state (torch.Tensor): Final state of shape `[N, H, K, V]` if
              `output_final_state=True` else `None`.

    Examples:
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from aiter.ops.triton.gated_delta_rule import chunk_gated_delta_rule
        >>> # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, device='cuda')
        >>> beta = torch.rand(B, T, H, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
        ...     q, k, v, g, beta,
        ...     initial_state=h0,
        ...     output_final_state=True
        ... )
        >>> # for variable-length inputs, the batch size `B` is expected to be 1
        >>> # and `cu_seqlens` is required
        >>> from einops import rearrange
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        >>> # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_rule(
        ...     q, k, v, g, beta,
        ...     initial_state=h0,
        ...     output_final_state=True,
        ...     cu_seqlens=cu_seqlens
        ... )

    Raises:
        ValueError: If input shapes are invalid when using cu_seqlens.
        NotImplementedError: If aiter implementation is incomplete.

    Note:
        The aiter chunk implementation is currently under development,
        and some auxiliary functions are not yet implemented.
    """
    # Input validation
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`. "
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )

    # Set default values
    if scale is None:
        scale = k.shape[-1] ** -0.5

    # Log operation
    _LOGGER.info(
        f"CHUNK_GATED_DELTA_RULE: q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}, "
        f"scale={scale}, use_qk_l2norm={use_qk_l2norm_in_kernel}"
    )

    # Apply L2 normalization if requested
    q_rstd, k_rstd = None, None
    if use_qk_l2norm_in_kernel:
        _LOGGER.info("Applying L2 normalization to q and k")
        q, q_rstd = l2norm_fwd(q)
        k, k_rstd = l2norm_fwd(k)

    # Call aiter's chunk forward pass
    g, o, A, final_state = chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )
    return o.to(q.dtype), final_state
