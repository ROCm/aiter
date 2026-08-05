# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Attention-residual (residual candidate gating) forward.

For every token the op scores L residual candidates with an RMS-normalized dot
product against the query, turns the scores into a softmax gate over the
candidate axis, and mixes the raw residuals with that gate::

    rstd_l  = 1 / sqrt(mean_d v[l, n, d]^2 + eps)
    logit_l = rstd_l * sum_d v[l, n, d] * (q_d * w_d)
    o[n]    = onorm( sum_l softmax_l(scale * logit_l) * v[l, n] )

Two residual layouts are supported, each paired with the pass structure that
won for it in benchmarking:

* ``layout="discrete"``: L independent ``[.., D]`` tensors, served by a
  two-pass D-tiled kernel. Only per-source scalars stay resident (~100 VGPR),
  so occupancy is high and it saturates HBM at large N; the residual is read
  twice.
* ``layout="packed"``: one contiguous ``[.., L, D]`` tensor, served by a
  one-pass whole-row kernel that loads ``v[L, D]`` once into registers and
  reuses it for both the reduction and the output, reading the residual from
  HBM exactly once.

:func:`attn_res_gate` exposes the same math under the inference contract used
by serving stacks: the candidate set is a packed ``[.., B, D]`` block plus a
separate ``prefix`` row, the caller's ``prefix += hidden`` add can be folded
into the kernel, and the backward statistics are not produced.
"""

from collections.abc import Sequence

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.attn_res import (
    _attn_res_fwd_discrete_2pass_kernel,
    _attn_res_fwd_packed_1pass_kernel,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def _build_ptr_table(tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    # Pad the per-source tuple to a power-of-2 length so Triton compiles one
    # kernel per L2 bucket instead of one per L.
    L2 = max(1, triton.next_power_of_2(len(tensors)))
    assert 1 <= len(tensors) <= L2
    for t in tensors:
        assert (
            t.data_ptr() % 16 == 0
        ), "attn_res residual sources must be 16-byte aligned"
    return tuple(tensors) + (tensors[0],) * (L2 - len(tensors))


def _cache_modifiers(use_cache_modifier: bool) -> tuple[str, str]:
    # ".cg": bypass L1, stream the read-once residual through L2. ".cs":
    # evict-first store for the write-once output. Both off -> default caching.
    return (".cg", ".cs") if use_cache_modifier else ("", "")


def attn_res_fwd(
    query: torch.Tensor,
    residuals,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    checkpoint_level: int = 1,
    *,
    layout: str = "discrete",
    use_exp2: bool = True,
    use_cache_modifier: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention-residual forward.

    Key parameters:
    - query: ``[.., D]`` scoring query, flattened internally.
    - residuals: discrete layout -> Sequence of L tensors each ``[.., D]``;
      packed layout -> a single ``[.., L, D]`` tensor, or a Sequence that will
      be stacked into one.
    - rms_weight: ``[D]`` per-channel weight folded into the score.
    - output_rms_weight: optional ``[D]`` weight enabling the output RMSNorm.
    - rms_eps: epsilon of both the per-candidate and the output RMSNorm.
    - scale: multiplies the logits before the softmax.
    - checkpoint_level: 0 also returns the pre-norm mix ``o_pre`` for backward.
    - layout: "discrete" (two-pass) or "packed" (one-pass).
    - use_exp2: softmax via the hardware exp2 instead of exp.
    - use_cache_modifier: residual ``.cg`` load / output ``.cs`` store instead
      of default caching.

    Returns:
    - (o, o_pre, rstd, logit, lse); ``o_pre`` is None unless
      ``checkpoint_level == 0``.
    """
    if layout not in ("discrete", "packed"):
        raise ValueError(f"layout must be 'discrete' or 'packed', got {layout!r}")

    _LOGGER.info(
        f"ATTN_RES: query={tuple(query.shape)} rms_weight={tuple(rms_weight.shape)} "
        f"layout={layout}"
    )

    load_cache, store_cache = _cache_modifiers(use_cache_modifier)
    has_onorm = output_rms_weight is not None
    save_opre = checkpoint_level == 0
    q_flat = query.flatten().contiguous()
    w_flat = rms_weight.flatten().contiguous()
    ow_flat = output_rms_weight.flatten().contiguous() if has_onorm else None

    runner = _run_packed if layout == "packed" else _run_discrete
    return runner(
        q_flat,
        residuals,
        w_flat,
        ow_flat,
        rms_eps,
        scale,
        has_onorm,
        save_opre,
        load_cache,
        store_cache,
        use_exp2,
    )


def _run_discrete(
    q_flat,
    residuals,
    w_flat,
    ow_flat,
    rms_eps,
    scale,
    has_onorm,
    save_opre,
    load_cache,
    store_cache,
    use_exp2,
):
    if not residuals[0].is_cuda:
        raise ValueError("Triton attn_res requires CUDA/ROCm tensors")
    output_shape = residuals[0].shape
    D = output_shape[-1]
    # The slot-scan gather hints 16-element alignment (tl.multiple_of) on each
    # row base, which only holds when the row stride D is a multiple of 16.
    assert (
        D % 16 == 0
    ), f"attn_res discrete layout requires D to be a multiple of 16, got D={D}"
    flat_residuals = tuple(r.reshape(-1, D).contiguous() for r in residuals)
    res = _build_ptr_table(flat_residuals)
    L = len(flat_residuals)
    N = flat_residuals[0].numel() // D
    dtype = flat_residuals[0].dtype
    device = flat_residuals[0].device

    o = torch.empty((N, D), device=device, dtype=dtype)
    need_opre = save_opre or has_onorm
    opre_dtype = torch.float32 if has_onorm else dtype
    o_pre = torch.empty((N, D), device=device, dtype=opre_dtype) if need_opre else None
    lse = torch.empty((N,), device=device, dtype=torch.float32)
    rstd = torch.empty((L, N), device=device, dtype=torch.float32)
    logit = torch.empty_like(rstd)
    L2 = max(1, triton.next_power_of_2(L))

    _attn_res_fwd_discrete_2pass_kernel[(N,)](
        q=q_flat,
        res=res,
        w=w_flat,
        ow=ow_flat,
        o=o,
        o_pre=o_pre if o_pre is not None else o,
        rstd=rstd,
        logit=logit,
        lse=lse,
        N=N,
        L=L,
        L2=L2,
        D=D,
        eps=rms_eps,
        scale=scale,
        NS=1,
        HAS_ONORM=has_onorm,
        SAVE_OPRE=save_opre,
        LOAD_CACHE=load_cache,
        STORE_CACHE=store_cache,
        EXP2=use_exp2,
    )
    o = o.view(output_shape)
    o_pre_out = (
        o_pre.to(dtype).view(output_shape)
        if (save_opre and o_pre is not None)
        else None
    )
    rstd = rstd.view(L, *output_shape[:-1])
    logit = logit.view(L, *output_shape[:-1])
    lse = lse.view(output_shape[:-1])
    return o, o_pre_out, rstd, logit, lse


def _run_packed(
    q_flat,
    residuals,
    w_flat,
    ow_flat,
    rms_eps,
    scale,
    has_onorm,
    save_opre,
    load_cache,
    store_cache,
    use_exp2,
):
    if isinstance(residuals, (list, tuple)):
        L = len(residuals)
        output_shape = residuals[0].shape  # [.., D]
        packed = torch.stack([r.contiguous() for r in residuals], dim=-2)  # [.., L, D]
    else:
        packed = residuals
        L = packed.shape[-2]
        output_shape = packed.shape[:-2] + packed.shape[-1:]
    if not packed.is_cuda:
        raise ValueError("Triton attn_res requires CUDA/ROCm tensors")
    D = output_shape[-1]
    packed = packed.reshape(-1, L, D).contiguous()  # [N, L, D]
    N = packed.shape[0]
    dtype = packed.dtype
    device = packed.device

    o = torch.empty((N, D), device=device, dtype=dtype)
    need_opre = save_opre or has_onorm
    o_pre = torch.empty((N, D), device=device, dtype=dtype) if need_opre else None
    lse = torch.empty((N,), device=device, dtype=torch.float32)
    rstd = torch.empty((L, N), device=device, dtype=torch.float32)
    logit = torch.empty_like(rstd)
    L2 = max(1, triton.next_power_of_2(L))

    _attn_res_fwd_packed_1pass_kernel[(N,)](
        q=q_flat,
        res=packed,
        w=w_flat,
        ow=ow_flat,
        o=o,
        o_pre=o_pre if o_pre is not None else o,
        rstd=rstd,
        logit=logit,
        lse=lse,
        prefix=packed,
        add_hidden=packed,
        prefix_out=packed,
        N=N,
        L=L,
        stride_res_n=packed.stride(0),
        stride_res_l=packed.stride(1),
        L2=L2,
        D=D,
        eps=rms_eps,
        scale=scale,
        BD=triton.next_power_of_2(D),
        HAS_ONORM=has_onorm,
        SAVE_OPRE=save_opre,
        HAS_PREFIX=False,
        DO_ADD=False,
        WRITE_PREF=False,
        SAVE_STATS=True,
        HAS_W=True,
        LOAD_CACHE=load_cache,
        STORE_CACHE=store_cache,
        EXP2=use_exp2,
    )
    o = o.view(output_shape)
    o_pre_out = o_pre.view(output_shape) if (save_opre and o_pre is not None) else None
    rstd = rstd.view(L, *output_shape[:-1])
    logit = logit.view(L, *output_shape[:-1])
    lse = lse.view(output_shape[:-1])
    return o, o_pre_out, rstd, logit, lse


def attn_res(
    query: torch.Tensor,
    residuals,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    return_weights: bool = False,
    checkpoint_level: int = 1,
    *,
    layout: str = "discrete",
    use_exp2: bool = True,
    use_cache_modifier: bool = True,
):
    """Convenience forward wrapper around :func:`attn_res_fwd`.

    Returns the mixed residual ``o``, plus the depth-softmax probabilities
    ``p`` when ``return_weights`` is set.
    """
    if len(residuals) == 0:
        raise ValueError("residuals must contain at least one source")
    o, _o_pre, _rstd, logit, lse = attn_res_fwd(
        query,
        residuals,
        rms_weight,
        output_rms_weight,
        rms_eps,
        scale,
        checkpoint_level,
        layout=layout,
        use_exp2=use_exp2,
        use_cache_modifier=use_cache_modifier,
    )
    if return_weights:
        p = (logit * scale - lse.unsqueeze(0)).exp()
        return o, p
    return o


def attn_res_gate(
    prefix: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float = 1e-6,
    add_hidden: torch.Tensor | None = None,
    *,
    output_rms_weight: torch.Tensor | None = None,
    scale: float = 1.0,
    use_exp2: bool = True,
    use_cache_modifier: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference-shaped attention-residual gate over ``B + 1`` candidates.

    Same math as :func:`attn_res_fwd` on the packed layout, specialized for the
    decode/prefill contract: the candidate set is the ``B`` rows of
    ``block_residual`` plus ``prefix`` as the last candidate, and no backward
    statistics are produced.

    Key parameters:
    - prefix: ``[.., D]`` running residual, used as the last candidate.
    - block_residual: ``[.., B, D]`` packed candidate block.
    - score_weight: ``[D]`` pre-folded ``rms_weight * query`` scoring vector.
    - eps: RMSNorm epsilon (also used by the output RMSNorm when set).
    - add_hidden: optional ``[.., D]``; folds ``prefix = prefix + add_hidden``
      into the kernel, saving a launch and an HBM round trip.
    - output_rms_weight: optional ``[D]``; folds the prenorm that would
      otherwise follow this call into the kernel.
    - scale: multiplies the logits before the softmax.

    Returns:
    - (y, prefix_out); ``prefix_out`` is the summed prefix when ``add_hidden``
      is given, otherwise ``prefix`` unchanged.
    """
    if not prefix.is_cuda:
        raise ValueError("Triton attn_res requires CUDA/ROCm tensors")
    if block_residual.dtype != prefix.dtype:
        raise ValueError(
            f"prefix and block_residual must share a dtype, got {prefix.dtype} "
            f"and {block_residual.dtype}"
        )

    _LOGGER.info(
        f"ATTN_RES_GATE: prefix={tuple(prefix.shape)} "
        f"block_residual={tuple(block_residual.shape)}"
    )

    load_cache, store_cache = _cache_modifiers(use_cache_modifier)
    output_shape = prefix.shape  # [.., D]
    D = output_shape[-1]
    B = block_residual.shape[-2]
    L = B + 1  # candidates: the B packed rows plus the prefix

    br = block_residual.reshape(-1, B, D).contiguous()
    pf = prefix.reshape(-1, D).contiguous()
    sw = score_weight.flatten().contiguous()
    N = pf.shape[0]
    if br.shape[0] != N:
        raise ValueError(
            f"prefix has {N} rows but block_residual has {br.shape[0]}; the "
            "leading dimensions must match"
        )

    has_onorm = output_rms_weight is not None
    ow = output_rms_weight.flatten().contiguous() if has_onorm else sw

    y = torch.empty((N, D), device=pf.device, dtype=pf.dtype)
    do_add = add_hidden is not None
    if do_add:
        hs = add_hidden.reshape(-1, D).contiguous()
        prefix_out = torch.empty_like(pf)
    else:
        # Never dereferenced (DO_ADD / WRITE_PREF are off) but Triton still
        # needs a tensor for the argument.
        hs = pf
        prefix_out = pf

    _attn_res_fwd_packed_1pass_kernel[(N,)](
        q=sw,
        res=br,
        w=sw,
        ow=ow,
        o=y,
        o_pre=y,
        rstd=y,
        logit=y,
        lse=y,
        prefix=pf,
        add_hidden=hs,
        prefix_out=prefix_out,
        N=N,
        L=L,
        stride_res_n=br.stride(0),
        stride_res_l=br.stride(1),
        L2=max(1, triton.next_power_of_2(L)),
        D=D,
        eps=eps,
        scale=scale,
        BD=triton.next_power_of_2(D),
        HAS_ONORM=has_onorm,
        SAVE_OPRE=False,
        HAS_PREFIX=True,
        DO_ADD=do_add,
        WRITE_PREF=do_add,
        SAVE_STATS=False,
        HAS_W=False,
        LOAD_CACHE=load_cache,
        STORE_CACHE=store_cache,
        EXP2=use_exp2,
    )
    return y.view(output_shape), (prefix_out.view(output_shape) if do_add else prefix)
