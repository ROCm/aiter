# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Attention-residual (residual candidate gating) forward.

For every token the op scores L residual candidates with an RMS-normalized dot
product against the query, turns the scores into a softmax gate over the
candidate axis, and mixes the raw residuals with that gate::

    rstd_l  = 1 / sqrt(mean_d v[l, n, d]^2 + eps)
    logit_l = rstd_l * sum_d v[l, n, d] * (q_d * w_d)
    o[n]    = onorm( sum_l softmax_l(scale * logit_l) * v[l, n] )

This is a forward-only port of fla 0.5.2's ``fused_attnres`` (see
``attnres_fwd_kernel``): the kernel structure, math, and launch surface follow
fla, but the backward pass is not ported. The backward checkpoint tensors
(``o_pre`` and the per-candidate ``rstd`` / ``logit`` / softmax ``lse``) are kept
as kernel parameters so a backward kernel can be reintroduced later, but they are
passed as ``None`` with their save flags off here.

Both residual layouts are served by the single ``attnres_fwd_kernel`` via an
``IS_PACKED`` switch:

* ``layout="sequence"``: a ``Sequence`` of L independent ``[.., D]`` tensors, the
  native form of the fla ``fused_attnres`` API, gathered through a length-``L2``
  pointer table.
* ``layout="packed"``: one contiguous ``[.., L, D]`` tensor read with row strides.

:func:`attn_res_gate` exposes the same packed kernel under the inference contract
used by serving stacks (the candidate set is a packed ``[.., B, D]`` block plus a
separate ``prefix`` row, and the caller's ``prefix += hidden`` add can be folded
into the kernel) -- this mirrors ATOM's ``apply_attn_res``.
"""

from collections.abc import Sequence

import torch
import triton

from aiter.ops.triton._triton_kernels.fusions.attn_res import attnres_fwd_kernel
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


# Static per-token-count launch table (ATOM-style), replacing @triton.autotune.
# Each bucket maps an upper-bound token count N to (num_warps, num_stages); BL is
# set to L2 at launch so the whole candidate axis is a single tile. Dispatch rounds
# N UP to the smallest bucket >= N (ceil-to-bucket), so a handful of fixed sizes
# compile one config each -- bounded compile cost and CUDAGraph-capture safe (autotune
# would JIT many configs on a cold cache and invalidate the capture). N above the
# largest bucket falls through to the catch-all. The values are conservative defaults;
# retune them with op_benchmarks/triton/bench_attn_res.py in the perf PR.
_ATTN_RES_CONFIGS = (
    # (max_tokens, num_warps, num_stages)
    (16, 8, 1),
    (64, 8, 1),
    (256, 8, 1),
    (1024, 16, 1),
)
_ATTN_RES_CATCHALL = (16, 1)  # N > largest bucket


def _pick_attn_res_config(tokens: int) -> tuple[int, int]:
    for max_tokens, num_warps, num_stages in _ATTN_RES_CONFIGS:
        if tokens <= max_tokens:
            return num_warps, num_stages
    return _ATTN_RES_CATCHALL


def _build_ptr_table(tensors: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    # Pad the per-source tuple to a power-of-2 length so Triton compiles one
    # kernel per L2 bucket instead of one per L. Padded slots reuse tensors[0]
    # and are masked out in the kernel.
    L2 = max(1, triton.next_power_of_2(len(tensors)))
    assert 1 <= len(tensors) <= L2
    for t in tensors:
        assert (
            t.data_ptr() % 16 == 0
        ), "attn_res residual sources must be 16-byte aligned"
    return tuple(tensors) + (tensors[0],) * (L2 - len(tensors))


def attn_res_fwd(
    query: torch.Tensor,
    residuals,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor | None = None,
    rms_eps: float = 1e-6,
    scale: float = 1.0,
    *,
    layout: str = "sequence",
) -> torch.Tensor:
    """Attention-residual forward.

    Key parameters:
    - query: ``[.., D]`` scoring query, flattened internally.
    - residuals: sequence layout -> Sequence of L tensors each ``[.., D]``;
      packed layout -> a single ``[.., L, D]`` tensor, or a Sequence that will
      be stacked into one.
    - rms_weight: ``[D]`` per-channel weight folded into the score.
    - output_rms_weight: optional ``[D]`` weight enabling the output RMSNorm.
    - rms_eps: epsilon of both the per-candidate and the output RMSNorm.
    - scale: multiplies the logits before the softmax.
    - layout: "sequence" or "packed".

    Returns the mixed residual ``o`` of shape ``[.., D]``.
    """
    if layout not in ("sequence", "packed"):
        raise ValueError(f"layout must be 'sequence' or 'packed', got {layout!r}")

    _LOGGER.info(
        f"ATTN_RES: query={tuple(query.shape)} rms_weight={tuple(rms_weight.shape)} "
        f"layout={layout}"
    )

    has_onorm = output_rms_weight is not None
    q_flat = query.flatten().contiguous()
    w_flat = rms_weight.flatten().contiguous()
    ow_flat = output_rms_weight.flatten().contiguous() if has_onorm else None

    runner = _run_packed if layout == "packed" else _run_sequence
    return runner(q_flat, residuals, w_flat, ow_flat, rms_eps, scale, has_onorm)


def _run_sequence(q_flat, residuals, w_flat, ow_flat, rms_eps, scale, has_onorm):
    if not residuals[0].is_cuda:
        raise ValueError("Triton attn_res requires CUDA/ROCm tensors")
    output_shape = residuals[0].shape
    D = output_shape[-1]
    # The slot-scan gather hints 16-element alignment (tl.multiple_of) on each
    # row base, which only holds when the row stride D is a multiple of 16.
    assert (
        D % 16 == 0
    ), f"attn_res sequence layout requires D to be a multiple of 16, got D={D}"
    flat_residuals = tuple(r.reshape(-1, D).contiguous() for r in residuals)
    res = _build_ptr_table(flat_residuals)
    L = len(flat_residuals)
    N = flat_residuals[0].numel() // D
    dtype = flat_residuals[0].dtype
    device = flat_residuals[0].device

    o = torch.empty((N, D), device=device, dtype=dtype)
    L2 = max(1, triton.next_power_of_2(L))
    num_warps, num_stages = _pick_attn_res_config(N)

    attnres_fwd_kernel[(N,)](
        q=q_flat,
        res=res,
        w=w_flat,
        ow=ow_flat,
        o=o,
        o_pre=None,
        rstd=None,
        logit=None,
        lse=None,
        res_packed=None,
        prefix=None,
        add_hidden=None,
        prefix_out=None,
        N=N,
        L=L,
        stride_res_n=0,
        stride_res_l=0,
        L2=L2,
        D=D,
        eps=rms_eps,
        scale=scale,
        BL=L2,
        BD=triton.next_power_of_2(D),
        HAS_ONORM=has_onorm,
        SAVE_OPRE=False,
        SAVE_STATS=False,
        IS_PACKED=False,
        HAS_PREFIX=False,
        DO_ADD=False,
        WRITE_PREF=False,
        HAS_W=True,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o.view(output_shape)


def _run_packed(q_flat, residuals, w_flat, ow_flat, rms_eps, scale, has_onorm):
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
    L2 = max(1, triton.next_power_of_2(L))
    num_warps, num_stages = _pick_attn_res_config(N)

    attnres_fwd_kernel[(N,)](
        q=q_flat,
        res=None,  # unused when IS_PACKED (sequence branch is dead); None keeps
        # the L2 dead pointer slots out of the kernarg segment
        w=w_flat,
        ow=ow_flat,
        o=o,
        o_pre=None,
        rstd=None,
        logit=None,
        lse=None,
        res_packed=packed,
        prefix=None,
        add_hidden=None,
        prefix_out=None,
        N=N,
        L=L,
        stride_res_n=packed.stride(0),
        stride_res_l=packed.stride(1),
        L2=L2,
        D=D,
        eps=rms_eps,
        scale=scale,
        BL=L2,
        BD=triton.next_power_of_2(D),
        HAS_ONORM=has_onorm,
        SAVE_OPRE=False,
        SAVE_STATS=False,
        IS_PACKED=True,
        HAS_PREFIX=False,
        DO_ADD=False,
        WRITE_PREF=False,
        HAS_W=True,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o.view(output_shape)


def attn_res_gate(
    prefix: torch.Tensor,
    block_residual: torch.Tensor,
    score_weight: torch.Tensor,
    eps: float = 1e-6,
    add_hidden: torch.Tensor | None = None,
    *,
    output_rms_weight: torch.Tensor | None = None,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Inference-shaped attention-residual gate over ``B + 1`` candidates.

    Same math as :func:`attn_res_fwd` on the packed layout, specialized for the
    decode/prefill contract (mirrors ATOM's ``apply_attn_res``): the candidate set
    is the ``B`` rows of ``block_residual`` plus ``prefix`` as the last candidate.

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
        # add_hidden / prefix_out are unused (DO_ADD / WRITE_PREF are off) but
        # Triton still needs a tensor argument, so reuse the prefix.
        hs = pf
        prefix_out = pf

    L2 = max(1, triton.next_power_of_2(L))
    num_warps, num_stages = _pick_attn_res_config(N)

    attnres_fwd_kernel[(N,)](
        q=sw,
        res=None,  # unused when IS_PACKED (sequence branch is dead); None keeps
        # the L2 dead pointer slots out of the kernarg segment
        w=sw,
        ow=ow,
        o=y,
        o_pre=None,
        rstd=None,
        logit=None,
        lse=None,
        res_packed=br,
        prefix=pf,
        add_hidden=hs,
        prefix_out=prefix_out,
        N=N,
        L=L,
        stride_res_n=br.stride(0),
        stride_res_l=br.stride(1),
        L2=L2,
        D=D,
        eps=eps,
        scale=scale,
        BL=L2,
        BD=triton.next_power_of_2(D),
        HAS_ONORM=has_onorm,
        SAVE_OPRE=False,
        SAVE_STATS=False,
        IS_PACKED=True,
        HAS_PREFIX=True,
        DO_ADD=do_add,
        WRITE_PREF=do_add,
        HAS_W=False,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return y.view(output_shape), (prefix_out.view(output_shape) if do_add else prefix)
