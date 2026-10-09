# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

"""
Kimi Delta Attention (KDA) chunked forward and backward pass.

This is the public entry point for the ``chunk_delta_attn`` Triton kernels. The
signature mirrors ``fla.ops.kda.chunk_kda`` so serving stacks can swap the two
backends without reshaping tensors or re-deriving the gate.

When any tensor input requires grad, the call runs under an autograd function
whose backward is ``chunk_delta_attn_bwd``. By default it saves only the inputs
and the backward recomputes the default pipeline's intermediates; with
``disable_recompute=True`` the forward keeps them for the backward instead.

Notes on the fla-compatible surface:
    * ``state_v_first`` is spelled as fla spells it from 0.5.1 onward. fla's
      pre-0.5.1 name ``transpose_state_layout`` is deliberately not accepted, so
      a shared argument dict requires fla >= 0.5.1 on the other side.
    * ``disable_recompute=False`` recomputes more than fla's does: fla saves the
      activated q/k/beta, ``Aqk`` and ``Akk`` and recomputes the rest, while
      here only the inputs are saved, which also lets the forward take the
      FlashKDA path. ``True`` matches fla: every intermediate is saved, and the
      forward runs the default pipeline.
    * ``allow_neg_eigval`` / ``return_intermediate_states`` / ``cp_context`` /
      ``cu_seqlens_cpu`` have no counterpart yet and are rejected rather than
      silently ignored.
    * Tensor arguments are made contiguous here, as fla's ``@input_guard``
      does, because some kernels below assume contiguous strides.
    * ``out`` and the paged ``state_cache`` are inference-only and are rejected
      when an input requires grad.
"""

import torch

from aiter.ops.triton._triton_kernels.kimi_delta_attn import (
    chunk_delta_attn_bwd,
    chunk_delta_attn_fwd,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# What the backward recomputes with: the default pipeline's chunk size, which is
# what an unset `chunk_size` falls back to there.
_BWD_DEFAULT_CHUNK_SIZE = 64


class _ChunkKimiDeltaAttnFunction(torch.autograd.Function):
    """Runs ``chunk_delta_attn_fwd`` and ``chunk_delta_attn_bwd``.

    With ``disable_recompute`` the forward's intermediates are saved for the
    backward; otherwise only the inputs are, and the backward recomputes the rest.
    """

    @staticmethod
    def forward(
        ctx, q, k, v, g, beta, A_log, dt_bias, initial_state, disable_recompute, kwargs
    ):
        o, final_state, *saved = chunk_delta_attn_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            disable_recompute=disable_recompute,
            **kwargs,
        )
        if not disable_recompute:
            saved = []
        ctx.save_for_backward(
            q,
            k,
            v,
            g,
            beta,
            A_log,
            dt_bias,
            initial_state,
            kwargs["cu_seqlens"],
            *saved,
        )
        ctx.kwargs = {
            key: val
            for key, val in kwargs.items()
            if key not in ("cu_seqlens", "output_final_state")
        }
        return o, final_state

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, g, beta, A_log, dt_bias, initial_state, cu_seqlens, *saved = (
            ctx.saved_tensors
        )
        kwargs = dict(ctx.kwargs)
        if saved:
            # Aqk's last dim is the chunk size the forward resolved to.
            kwargs["chunk_size"] = saved[1].shape[-1]
        elif kwargs["chunk_size"] is None:
            kwargs["chunk_size"] = _BWD_DEFAULT_CHUNK_SIZE
        grads = chunk_delta_attn_bwd(
            saved=tuple(saved) if saved else None,
            do=do,
            dht=dht,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            **kwargs,
        )
        return (*grads, None, None)


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
    chunk_size: int | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    out: torch.Tensor | None = None,
    state_cache: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    r"""
    Chunked Kimi Delta Attention using Triton.

    Differentiable w.r.t. every float tensor input (`q`, `k`, `v`, `g`, `beta`,
    `A_log`, `dt_bias`, `initial_state`) when any of them requires grad. The
    backward recomputes the default pipeline's intermediates at chunk size 64
    unless `chunk_size` is given, whichever path ran the forward.

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
            `state_v_first=True`), for `N` input sequences. Any float dtype; the
            recurrence accumulates in fp32 whatever the state is stored as. For
            equal-length inputs `N` equals the batch size `B`. Default: `None`.
        output_final_state (bool):
            Whether to return the final state, same shape as `initial_state`.
            Its dtype is `initial_state`'s on the FlashKDA path and fp32 on the
            default pipeline. Default: `False`.
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
            Only matters when an input requires grad. `True` keeps the
            forward's intermediates (about `T * HV * (4K + 2V + 2BT)` elements
            plus `h`) for the backward instead of recomputing them; the forward
            then runs the default pipeline even where FlashKDA could serve it.
            Default: `False`.
        chunk_size (int, optional):
            Chunk size, either 32 or 64. Default: `None`, which lets the
            library choose. fla has no counterpart and pins 64 internally, so
            nothing on the shared surface sets this.

            The choice is really which kernel runs. 32 is the two-kernel
            FlashKDA path's entry ticket, and it is picked when the rest of the
            call also qualifies for it: the fused gate, in-kernel l2norm and
            beta sigmoid, `safe_gate`, and `K = V = 128` without GVA. Anything
            else gets 64, which is the faster of the two inside the default
            pipeline. Passing 32 or 64 explicitly is honoured as given.

            FlashKDA agrees with the default pipeline to bf16 rather than
            exactly, as does 32 against 64 within the default pipeline itself.
            Set `AITER_FDA_ENABLE=0` to pin the default pipeline.
        cu_seqlens (torch.LongTensor, optional):
            Cumulative sequence lengths of shape `[N+1]` for variable-length
            inputs, consistent with the FlashAttention API. Default: `None`.
        out (torch.Tensor, optional):
            Caller-owned output buffer, same shape as `v`, and contiguous.
            Written in place rather than allocated and copied. Required with
            `state_cache`. FlashKDA-only; the default pipeline rejects this
            argument rather than ignoring it, as does a call that needs grad.
        state_cache (torch.Tensor, optional):
            Paged fp32 V-first cache `[slots, H, V, K]`. Replaces
            `initial_state`: sequence `n` is row `state_indices[n]`. Each
            slot's `[H, V, K]` plane must be dense; `stride(0)` may be padded.
            FlashKDA-only; the default pipeline rejects this argument rather
            than ignoring it, as does a call that needs grad.
        state_indices (torch.Tensor, optional):
            Int32 `[N]` cache row per sequence, each in `[0, slots)`. Required
            with `state_cache`. Trusted as given, not range-checked.
        has_initial_state (torch.Tensor, optional):
            Bool `[N]`. False starts that sequence from zero. Required with
            `state_cache`.

    Returns:
        tuple[torch.Tensor, torch.Tensor | None]:
            - o (torch.Tensor): Outputs of shape `[B, T, HV, V]`.
            - final_state (torch.Tensor | None): Final state if
              `output_final_state=True` else `None`. `None` when
              `state_cache` is given, because the cache already holds it.

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
    if initial_state is not None and not initial_state.is_floating_point():
        raise ValueError(
            f"`initial_state` must be a float tensor, got {initial_state.dtype}."
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
        "CHUNK_KIMI_DELTA_ATTN: q=%s, v=%s, scale=%s, chunk_size=%s, safe_gate=%s, lower_bound=%s, state_v_first=%s, varlen=%s",
        tuple(q.shape),
        tuple(v.shape),
        scale,
        chunk_size or "auto",
        safe_gate,
        lower_bound,
        state_v_first,
        cu_seqlens is not None,
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
    if state_indices is not None:
        state_indices = state_indices.contiguous()
    if has_initial_state is not None:
        has_initial_state = has_initial_state.contiguous()
    # state_cache is not packed: stride(0) may be padded. Packing it would
    # copy the pool and write a clone.

    tensors = (q, k, v, g, beta, A_log, dt_bias, initial_state)
    if torch.is_grad_enabled() and any(
        t is not None and t.requires_grad for t in tensors
    ):
        if out is not None or state_cache is not None:
            raise ValueError(
                "`out` and `state_cache` are inference-only; they are not "
                "supported when an input requires grad."
            )
        o, final_state = _ChunkKimiDeltaAttnFunction.apply(
            *tensors,
            disable_recompute,
            {
                "scale": scale,
                "output_final_state": output_final_state,
                "cu_seqlens": cu_seqlens,
                "chunk_size": chunk_size,
                "safe_gate": safe_gate,
                "lower_bound": lower_bound,
                "use_gate_in_kernel": use_gate_in_kernel,
                "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
                "use_beta_sigmoid_in_kernel": use_beta_sigmoid_in_kernel,
                "state_v_first": state_v_first,
            },
        )
        return o.to(q.dtype), final_state

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
        # Pinned off: without grad nothing reads the intermediates it would keep.
        disable_recompute=False,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        state_v_first=state_v_first,
        out=out,
        state_cache=state_cache,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
    )
    return o.to(q.dtype), final_state
