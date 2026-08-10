# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

"""
Kimi Delta Attention (KDA) chunked forward pass (Forward Only).

This is the public entry point for the ``chunk_delta_attn`` Triton kernels. The
signature mirrors ``fla.ops.kda.chunk_kda`` so serving stacks can swap the two
backends without reshaping tensors or re-deriving the gate.

Important Note:
    Only the forward pass is implemented. These functions do NOT support
    gradient computation. For training, use the flash-linear-attention library.

Notes on the fla-compatible surface:
    * ``state_v_first`` is spelled as fla spells it from 0.5.1 onward. fla's
      pre-0.5.1 name ``transpose_state_layout`` is deliberately not accepted, so
      a shared argument dict requires fla >= 0.5.1 on the other side.
    * ``disable_recompute`` is accepted for signature parity but has no effect.
      In both libraries it only decides whether the gated query ``qg`` is
      materialized for a backward pass to reuse -- neither forward output
      kernel reads it. Since this path is forward-only and returns no scratch,
      honouring ``True`` would allocate and write a ``T x HV x K`` tensor that
      nothing can observe, so the flag is pinned off internally.
    * ``allow_neg_eigval`` / ``return_intermediate_states`` / ``cp_context`` /
      ``cu_seqlens_cpu`` have no counterpart yet and are rejected rather than
      silently ignored.
    * Tensor arguments are made contiguous here, as fla's ``@input_guard``
      does, because some kernels below assume contiguous strides.
    * This path is forward-only, so the output carries no ``grad_fn`` even
      when the inputs require grad, whereas fla returns a differentiable
      tensor.
"""

import torch

from aiter.ops.triton._triton_kernels.chunk_delta_attn import chunk_delta_attn_fwd
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def chunk_kimi_delta_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    state_v_first: bool = False,
    disable_recompute: bool = False,
    chunk_size: int = 64,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Chunked Kimi Delta Attention forward pass using Triton (Forward only).

    Warning:
        Forward only; this function does NOT compute gradients.

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, HV, V]`. GVA is applied if `HV > H`, in which
            case `HV` must be divisible by `H`.
        g (torch.Tensor):
            (forget) gate of shape `[B, T, HV, K]`.
            When `use_gate_in_kernel=False` this is the pre-computed decay in log
            space; when `True` it is the raw pre-activation input and the kernel
            fuses the `A_log` / `dt_bias` activation plus the chunk cumsum.
        A_log (torch.Tensor, optional):
            per-head log-scale of shape `[HV]`. Required when
            `use_gate_in_kernel=True`.
        dt_bias (torch.Tensor, optional):
            per-K-channel bias of shape `[HV * K]`, added to `g` before the gate
            activation. Default: `None`.
        beta (torch.Tensor):
            betas of shape `[B, T, HV]`. Raw logits when
            `use_beta_sigmoid_in_kernel=True`, post-sigmoid otherwise. Pass this
            in fp32 to keep the delta-rule write strength from being rounded to
            the input dtype.
        scale (float, optional):
            Scale factor for the attention scores. Default: `1 / sqrt(K)`.
        initial_state (torch.Tensor, optional):
            Initial state of shape `[N, HV, K, V]` (`[N, HV, V, K]` when
            `state_v_first=True`) and dtype fp32, for `N` input sequences. For
            equal-length inputs `N` equals the batch size `B`. Default: `None`.
        output_final_state (bool):
            Whether to return the final state, same shape and dtype as
            `initial_state`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to L2-normalize `q` and `k` before the recurrence.
        use_gate_in_kernel (bool):
            Whether to compute the log-space decay from `g` internally.
        use_beta_sigmoid_in_kernel (bool):
            Whether to apply `sigmoid` to `beta` before the recurrence.
        safe_gate (bool):
            Whether to use the sub-chunk intra kernel, which keeps the gate
            bounded across sub-chunk boundaries. Requires `lower_bound`.
        lower_bound (float, optional):
            Lower bound of the forget gate in log space. When set, the gate
            activation becomes `lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`
            instead of `-exp(A_log) * softplus(g + dt_bias)`. Kimi uses `-5.0`.
        state_v_first (bool):
            Store the recurrent state V-first (`[V, K]`) instead of `[K, V]`.
        disable_recompute (bool):
            Ignored. In fla this keeps the gated query `qg` alive for the
            backward pass; this path is forward-only and returns no scratch,
            so the flag is accepted for signature parity only.
        chunk_size (int):
            Chunk size, either 32 or 64. Default: `64`.
        cu_seqlens (torch.LongTensor, optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length
            inputs, consistent with the FlashAttention API. Default: `None`.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]:
            - o (torch.Tensor): Outputs of shape `[B, T, HV, V]`.
            - final_state (torch.Tensor | None): Final state if
              `output_final_state=True` else `None`.

    Examples:
        >>> import torch
        >>> from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn
        >>> B, T, H, K, V = 1, 2048, 8, 128, 128
        >>> q = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
        >>> k = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
        >>> v = torch.randn(B, T, H, V, device='cuda', dtype=torch.bfloat16)
        >>> g = torch.randn(B, T, H, K, device='cuda', dtype=torch.bfloat16)
        >>> beta = torch.randn(B, T, H, device='cuda', dtype=torch.float32)
        >>> A_log = torch.randn(H, device='cuda', dtype=torch.float32)
        >>> dt_bias = torch.randn(H * K, device='cuda', dtype=torch.float32)
        >>> h0 = torch.zeros(B, H, K, V, device='cuda', dtype=torch.float32)
        >>> o, ht = chunk_kimi_delta_attn(
        ...     q, k, v, g, beta, A_log=A_log, dt_bias=dt_bias,
        ...     use_qk_l2norm_in_kernel=True,
        ...     use_gate_in_kernel=True,
        ...     use_beta_sigmoid_in_kernel=True,
        ...     safe_gate=True, lower_bound=-5.0,
        ...     initial_state=h0, output_final_state=True,
        ... )
    """
    B, T, H, K = q.shape
    HV = v.shape[2]

    if q.shape != k.shape:
        raise ValueError(
            f"q and k must have the same shape, got {q.shape} vs {k.shape}."
        )
    if K > 256:
        raise ValueError(f"Only key head dim <= 256 is supported, got {K}.")
    if HV % H != 0:
        raise ValueError(
            f"For GVA, num_v_heads (HV={HV}) must be divisible by num_qk_heads (H={H})."
        )
    if tuple(g.shape) != (B, T, HV, K):
        raise ValueError(f"g must have shape {(B, T, HV, K)}, got {tuple(g.shape)}.")
    if tuple(beta.shape) != (B, T, HV):
        raise ValueError(f"beta must have shape {(B, T, HV)}, got {tuple(beta.shape)}.")

    if cu_seqlens is not None:
        if B != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {B} when using "
                "`cu_seqlens`. Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                "The number of initial states is expected to be equal to the number "
                f"of input sequences, i.e., {len(cu_seqlens) - 1} rather than "
                f"{initial_state.shape[0]}."
            )
    if initial_state is not None and initial_state.dtype != torch.float32:
        raise ValueError(
            f"`initial_state` must be fp32, got {initial_state.dtype}. The recurrence "
            "accumulates in fp32 and the state is read back verbatim."
        )

    if use_gate_in_kernel and A_log is None:
        raise ValueError("`A_log` must be provided when `use_gate_in_kernel=True`.")
    if safe_gate and use_gate_in_kernel:
        if lower_bound is None:
            raise ValueError(
                "`lower_bound` must be specified when `safe_gate=True` and "
                "`use_gate_in_kernel=True`."
            )
        if not -5 <= lower_bound < 0:
            raise ValueError(
                f"`lower_bound` must be in the safe range [-5, 0), got {lower_bound}."
            )

    if scale is None:
        scale = K**-0.5
    elif scale <= 0:
        raise ValueError(f"`scale` must be positive, got {scale}.")

    _LOGGER.info(
        f"CHUNK_KIMI_DELTA_ATTN: q={tuple(q.shape)}, v={tuple(v.shape)}, "
        f"scale={scale}, chunk_size={chunk_size}, safe_gate={safe_gate}, "
        f"lower_bound={lower_bound}, state_v_first={state_v_first}, "
        f"varlen={cu_seqlens is not None}"
    )

    # Match fla, which puts `@input_guard` on its autograd Function. Several
    # kernels below index their operands with contiguous strides rather than
    # reading the tensor's own, so a view would be read as garbage instead of
    # raising: `q`/`k` reach the unguarded l2norm, `initial_state` the
    # unguarded state kernel. No-op for the contiguous inputs serving stacks
    # normally pass.
    q, k, v, g, beta = (t.contiguous() for t in (q, k, v, g, beta))
    if A_log is not None:
        A_log = A_log.contiguous()
    if dt_bias is not None:
        dt_bias = dt_bias.contiguous()
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    if cu_seqlens is not None:
        cu_seqlens = cu_seqlens.contiguous()

    o, final_state, *_ = chunk_delta_attn_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        use_gate_in_kernel=use_gate_in_kernel,
        A_log=A_log,
        dt_bias=dt_bias,
        # Pinned off: it would only materialize `qg` for a backward that does
        # not exist here, and the scratch is discarded below either way.
        disable_recompute=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        state_v_first=state_v_first,
    )
    return o.to(q.dtype), final_state
