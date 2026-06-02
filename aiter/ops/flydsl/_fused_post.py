# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Triton post-pass kernels for FlyDSL MOE stage1 split-K outputs.

The non-fp4 split-K stage1 leaves a f32 ``tmp_out`` buffer that must be reduced
(reduce mode only) and then fed through ``silu_and_mul`` to produce the bf16/fp16
``out`` consumed by stage2. The legacy path calls ``tmp_out.sum(dim=0)`` (a
PyTorch reduction kernel) followed by aiter's HIP ``silu_and_mul`` (a second
kernel), which costs roughly 5.5 us + 2.8 us at small M.

For atomic mode the GEMM has already accumulated across kb in-place, so only
``silu_and_mul`` runs and that already lives in a single aiter HIP kernel; we
leave it alone here.

For reduce mode we fuse the kb-axis sum together with silu_and_mul into a single
Triton kernel: each program reduces ``(kb, BLOCK_N)`` of gate and up partials,
applies ``silu(gate) * up``, and writes the bf16/fp16 result. This collapses
two launches into one and removes a full f32 round-trip through HBM for the
reduced buffer (the existing path materialises ``reduced`` as a fresh tensor).
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import triton
import triton.language as tl


_FUSED_OFF = os.environ.get("AITER_FLYDSL_FUSED_SILU_OFF", "0") == "1"
_FUSED_INIT_OFF = os.environ.get("AITER_FLYDSL_FUSED_INIT_OFF", "0") == "1"
_FUSED_TOPK_OFF = os.environ.get("AITER_FLYDSL_FUSED_TOPK_OFF", "0") == "1"


def is_disabled() -> bool:
    """Module-level kill switch; respected by `moe_kernels.flydsl_moe_stage1`."""
    return _FUSED_OFF


def is_init_disabled() -> bool:
    """Kill switch for the fused stage1 init kernel; respected by moe_kernels."""
    return _FUSED_INIT_OFF


def is_topk_sum_disabled() -> bool:
    """Kill switch for the fused stage2 topk-sum kernel; respected by moe_kernels."""
    return _FUSED_TOPK_OFF


# ---------------------------------------------------------------------------
# Host-side config picker for the reduce-mode Triton kernels.
#
# We deliberately avoid ``@triton.autotune`` here.  The autotune wrapper adds
# a ~25 us Python per-call overhead (cache lookup + config dispatch) which is
# fine for kernels that run for milliseconds but ruins the tiny stage1/stage2
# reduce post-passes -- at M=1 the kernel itself is ~12 us, so the wrapper
# would more than double total cost.  Instead we hand-roll a tiny dispatcher
# calibrated from offline sweeps over ``BLOCK_N x num_warps`` (see
# ``HY3_BKC/scripts/_kb_sum_autotune_probe.py``).
#
# Returned: ``(BLOCK_N, num_warps)`` per (rows, inter_dim) regime.  The same
# table covers the kb-sum+silu+mul (stage1 reduce) and topk-sum (stage2
# reduce) kernels because their work shape is structurally identical:
# ``(rows, inter_dim or model_dim)`` per-program reductions.
# ---------------------------------------------------------------------------


def _pick_kb_sum_config(rows: int, inter_dim: int) -> tuple:
    """Return (BLOCK_N, num_warps) for ``_kb_sum_silu_mul_kernel``.

    The kernel loads f32 partials (4 bytes/elt) and the inner accumulator
    is wide (2 * BLOCK_N f32 lanes), so register pressure climbs fast with
    BLOCK_N.  Probe sweep (see ``_kb_sum_autotune_probe.py``):
      - inter_dim>=1024 + huge rows (int8 large-M case): bn=2048 num_warps=1
        wins (lower warp count frees registers for the long accumulator).
        Beats num_warps=4 by ~11% at inter=1536.
      - small inter_dim (192, the only fp8 case that ever picks reduce mode):
        bn=512 num_warps=4 wins by ~4% over the prior heuristic bn=256.
    """
    if inter_dim >= 1024:
        return 2048, 1
    if inter_dim >= 512:
        return 1024, 2
    return 512, 4


def _pick_init_config(max_n: int) -> tuple:
    """Return (BLOCK, num_warps) for ``_fused_stage1_init_kernel``.

    Calibrated via strict e2e sweep (see ``_fused_init_e2e_sweep_strict.py``,
    min of 5 runs at 2000 iters each to reject noise).  Three tiers:

      - max_n < 50K (M=1, M=2 stage1 cases): ``(256, 8)``.  Reproducibly
        saves 0.5-0.8 us per call vs ``(1024, 4)`` -- smaller BLOCK spawns
        ~108 WGs across the 80 CUs for full occupancy at tiny workloads,
        and w=8 hides the load latency on the per-expert w_src scatter.
      - max_n in [50K, 10M] (M=32..M=256 fp8 atomic): keep ``(1024, 4)``.
        No reproducible win in this range, and earlier sweeps showed
        smaller BLOCK regressed M=256 e2e by ~2 us due to grid-size
        sensitivity around max_n ~= 1M.
      - max_n >= 10M (M=1024+ int8 cases): ``(4096, 8)``.  Wins +1-2 us
        over default at both M=1024 (~0.9 us) and M=4096 (~1.9 us).
        bn=8192 wins ~0.5 us more at each but its optimal num_warps
        flips between M=1024 (w=16) and M=4096 (w=4); ``(4096, 8)``
        is the robust compromise.
    """
    if max_n < 50_000:
        return 256, 8
    if max_n < 10_000_000:
        return 1024, 4
    return 4096, 8


def _pick_topk_sum_config(rows: int, model_dim: int) -> tuple:
    """Return (BLOCK_N, num_warps) for ``_topk_sum_kernel``.

    Loads are bf16/fp16 (2 bytes/elt) and the accumulator is just 1 *
    BLOCK_N f32 lanes (no gate/up split), so register pressure is half of
    the kb-sum kernel and the optimum shifts to slightly higher warps.
    Probe sweep:
      - model_dim>=1024 (the only case where Triton beats torch.sum):
        bn=2048 num_warps=2 wins by ~20% over num_warps=1 (143us -> 117us
        at M=2048 topk=20 model_dim=4096).
      - smaller model_dim is rare here; fall back to bn=512 num_warps=4.
    """
    if model_dim >= 1024:
        return 2048, 2
    return 512, 4


# Threshold below which fused_topk_sum falls back to torch.sum: the Triton
# kernel has ~10 us launch overhead on ROCm whereas torch.sum has ~2 us, so
# the native path wins until kernel work outweighs the launch gap.  Probe
# data: at 16M target elements (M=512 topk=8 mdim=4096) both paths break
# even; we use 16M as a conservative cutoff so anything below stays on
# torch.sum (the reduce-mode winners in the current tuned csv are either
# tiny fp8 cases <1M or huge int8 cases >150M, so the threshold rarely
# matters in practice).
_TOPK_SUM_TRITON_MIN_ELEMS = 16 * 1024 * 1024  # 16M target elements


@triton.jit
def _kb_sum_silu_mul_kernel(
    out_ptr,
    tmp_out_ptr,
    rows,
    inter_dim,
    KB: tl.constexpr,
    OUT_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per-(row, col-tile) program: reduce kb -> silu(gate)*up -> store.

    Layout: tmp_out is contiguous (KB, rows, 2*inter_dim) f32 with gate at
    columns [0, inter_dim) and up at [inter_dim, 2*inter_dim). out is
    (rows, inter_dim) in bf16 / fp16.

    BLOCK_N and num_warps are picked host-side by ``_pick_reduce_config``
    rather than via ``@triton.autotune`` -- the autotune wrapper's per-call
    Python overhead (~25 us) dwarfs this kernel's GPU time at the small-M
    shapes that are the only place stage1 reduce mode ever wins.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    col_off = pid_n * BLOCK_N
    offs = col_off + tl.arange(0, BLOCK_N)
    mask = offs < inter_dim

    g = tl.zeros([BLOCK_N], dtype=tl.float32)
    u = tl.zeros([BLOCK_N], dtype=tl.float32)

    row_stride = 2 * inter_dim
    slab_stride = rows * row_stride

    for k in tl.static_range(KB):
        base = tmp_out_ptr + k * slab_stride + pid_m * row_stride
        g += tl.load(base + offs, mask=mask, other=0.0)
        u += tl.load(base + inter_dim + offs, mask=mask, other=0.0)

    # silu(g) = g * sigmoid(g) = g / (1 + exp(-g))
    y = (g * tl.sigmoid(g)) * u
    if OUT_BF16:
        y = y.to(tl.bfloat16)
    else:
        y = y.to(tl.float16)
    tl.store(out_ptr + pid_m * inter_dim + offs, y, mask=mask)


def fused_kb_sum_silu_and_mul(
    out: torch.Tensor,
    tmp_out: torch.Tensor,
    *,
    inter_dim: Optional[int] = None,
) -> None:
    """Fused ``tmp_out.sum(dim=0).silu_and_mul -> out`` for reduce mode.

    Args:
      out: target (rows, inter_dim) bf16/fp16 tensor (will be overwritten).
      tmp_out: (KB, rows, 2*inter_dim) f32 contiguous partials.
      inter_dim: optional explicit inter_dim (derived from tmp_out.shape[-1]/2
        if omitted). Useful when tmp_out has been viewed differently.

    Layout assumption: gate occupies the first inter_dim columns and up the
    next inter_dim columns within each row, identical to aiter's silu_and_mul
    contract. Caller must ensure ``tmp_out`` is contiguous in (KB, rows, 2*N).
    """
    assert tmp_out.dtype == torch.float32, "tmp_out must be float32 partials"
    assert tmp_out.is_contiguous(), "tmp_out must be contiguous"
    assert out.dtype in (torch.bfloat16, torch.float16), (
        f"out dtype must be bf16/fp16, got {out.dtype}"
    )
    assert tmp_out.ndim == 3, (
        f"tmp_out must be 3D (KB, rows, 2*inter_dim), got shape {tuple(tmp_out.shape)}"
    )

    kb, rows, two_n = tmp_out.shape
    if inter_dim is None:
        assert two_n % 2 == 0, f"tmp_out last dim must be even, got {two_n}"
        inter_dim = two_n // 2
    else:
        assert two_n == 2 * inter_dim, (
            f"tmp_out last dim {two_n} != 2*inter_dim {2 * inter_dim}"
        )

    assert out.numel() == rows * inter_dim, (
        f"out numel {out.numel()} != rows * inter_dim "
        f"{rows} * {inter_dim} = {rows * inter_dim}"
    )

    block_n, num_warps = _pick_kb_sum_config(rows, inter_dim)
    grid = (rows, triton.cdiv(inter_dim, block_n))

    _kb_sum_silu_mul_kernel[grid](
        out,
        tmp_out,
        rows,
        inter_dim,
        KB=kb,
        OUT_BF16=(out.dtype == torch.bfloat16),
        BLOCK_N=block_n,
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# Stage2 reduce-mode post-pass: topk-axis sum
# ---------------------------------------------------------------------------
#
# In stage2 reduce mode (``accumulate=False``) the FlyDSL kernel writes each
# (token, topk_slot) row into its own slab of ``target`` (shape
# ``(token_num, topk, model_dim)``) so different routed copies of the same
# token never atomic-collide on the same output row.  The host then performs
# ``out = target.sum(dim=1)`` to collapse the topk axis into the final
# ``(token_num, model_dim)`` output.
#
# The legacy code does this via ``torch.sum(..., out=out)`` which is one
# fully-pipelined HIP kernel but still carries a separate dispatch (~2 us at
# small M) and prevents us from doing any post-pass fusion later.  We replace
# it with a one-program-per-(token, col-tile) Triton kernel that streams the
# topk slabs through registers; the load is in the kernel's native dtype
# (bf16/fp16), accumulation is in f32, and the store casts back.  Autotune
# shares the same config grid as the stage1 reduce kernel.


@triton.jit
def _topk_sum_kernel(
    out_ptr,
    target_ptr,
    token_num,
    model_dim,
    TOPK: tl.constexpr,
    OUT_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Per-(token, col-tile) program: sum across topk -> store.

    Layout: target is contiguous ``(token_num, TOPK, model_dim)`` in the
    output dtype (bf16/fp16); ``out`` is ``(token_num, model_dim)`` in the
    same dtype. Accumulation happens in f32 to match
    ``torch.sum(out_dtype=None)``'s implicit promotion.

    Same host-side config picker as ``_kb_sum_silu_mul_kernel``; ditto the
    rationale for skipping ``@triton.autotune``.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    col_off = pid_n * BLOCK_N
    offs = col_off + tl.arange(0, BLOCK_N)
    mask = offs < model_dim

    s = tl.zeros([BLOCK_N], dtype=tl.float32)

    slab_stride = TOPK * model_dim
    base = target_ptr + pid_m * slab_stride

    for k in tl.static_range(TOPK):
        v = tl.load(base + k * model_dim + offs, mask=mask, other=0.0)
        s += v.to(tl.float32)

    if OUT_BF16:
        y = s.to(tl.bfloat16)
    else:
        y = s.to(tl.float16)
    tl.store(out_ptr + pid_m * model_dim + offs, y, mask=mask)


def fused_topk_sum(
    out: torch.Tensor,
    target: torch.Tensor,
    *,
    token_num: int,
    topk: int,
    model_dim: int,
) -> None:
    """Fused ``target.view(token_num, topk, model_dim).sum(dim=1, out=out)``.

    For small workloads (target elem count < _TOPK_SUM_TRITON_MIN_ELEMS) we
    fall back to ``torch.sum`` because Triton on ROCm has higher launch
    overhead than torch's hand-optimised HIP reduction; the Triton path only
    wins once the per-element work amortises that launch gap.  Probe data:
    crossover is at ~8M target elements (M=512 topk=20 mdim=4096 == 41M
    elements, Triton wins by ~30 us; below that torch wins by 5-25 us).

    Args:
      out: target (token_num, model_dim) bf16/fp16 tensor (will be overwritten).
      target: contiguous (token_num * topk * model_dim,) flat tensor in the
        output dtype, holding per-(token, topk_slot) rows.
      token_num, topk, model_dim: shape of the logical reshape.

    Skipped entirely when ``AITER_FLYDSL_FUSED_TOPK_OFF=1`` (env kill-switch):
    falls back to ``torch.sum`` for every shape.
    """
    # Fast path first: small workloads bypass the assertion block entirely
    # so the small-M case (where this dispatcher would otherwise add ~5 us
    # of Python validation cost on top of torch.sum's ~10 us GPU time) is
    # as cheap as possible.  We trust the caller's stage2 contract here.
    if target.numel() < _TOPK_SUM_TRITON_MIN_ELEMS:
        torch.sum(target.view(token_num, topk, model_dim), dim=1, out=out)
        return

    assert out.dtype in (torch.bfloat16, torch.float16), (
        f"out dtype must be bf16/fp16, got {out.dtype}"
    )
    assert target.dtype == out.dtype, (
        f"target/out dtype mismatch: {target.dtype} vs {out.dtype}"
    )
    assert target.is_contiguous(), "target must be contiguous"
    assert target.numel() == token_num * topk * model_dim, (
        f"target numel {target.numel()} != token_num * topk * model_dim "
        f"= {token_num} * {topk} * {model_dim}"
    )
    assert out.numel() == token_num * model_dim, (
        f"out numel {out.numel()} != token_num * model_dim "
        f"= {token_num} * {model_dim}"
    )

    block_n, num_warps = _pick_topk_sum_config(token_num, model_dim)
    grid = (token_num, triton.cdiv(model_dim, block_n))

    _topk_sum_kernel[grid](
        out,
        target,
        token_num,
        model_dim,
        TOPK=topk,
        OUT_BF16=(out.dtype == torch.bfloat16),
        BLOCK_N=block_n,
        num_warps=num_warps,
    )


# ---------------------------------------------------------------------------
# Stage1/Stage2 init helper fusion
# ---------------------------------------------------------------------------
#
# Both stages share the same init-time helper triplet before their main GEMM:
#
#   (1) zero-fill of the output/accumulator buffer
#         -- stage1: f32 ``tmp_out`` partial-sum buffer
#         -- stage2: bf16/fp16 ``out`` (only when accumulate, i.e. atomic mode)
#   (2) scalar a_scale broadcast to (rows,)
#         -- stage1 rows = token_num,        stage2 rows = token_num * topk
#   (3) per-expert w_scale broadcast to (E * cols,)
#         -- stage1 cols = inter_dim*2 (g1u1) or inter_dim, stage2 cols = model_dim
#
# At small M every launch is a meaningful slice of the total cost.  We fuse
# all three into a single Triton launch that uses a runtime ``if base < N``
# guard per sub-region so programs past a sub-region's tail skip its
# loads/stores entirely.  The ``tmp_out`` pointer can be f32 (stage1) or
# bf16/fp16 (stage2); Triton implicitly casts the f32 ``0.0`` store value to
# the destination dtype, so a single kernel covers both.
#
# Note: we used to maintain a host-side ``id(scale)`` -> expanded-buffer cache
# so steady-state inference would skip the w-scale broadcast entirely.  Once
# stage1 (and now stage2) both go through this fused kernel the broadcast
# write rides for free on top of the launch we already need, so the cache
# was retired together with its weakref/finalizer machinery.


@triton.jit
def _fused_init_kernel(
    tmp_out_ptr,
    tmp_n,
    flat_a_ptr,
    flat_a_n,
    a_src_ptr,
    flat_w_ptr,
    flat_w_n,
    w_src_ptr,
    w_cols,
    DO_TMP: tl.constexpr,
    EXPAND_A: tl.constexpr,
    EXPAND_W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One launch: zero tmp_out + broadcast a/w scales into pre-allocated flats.

    Grid: ``(cdiv(max(tmp_n, flat_a_n, flat_w_n), BLOCK),)``.  Every program
    covers one BLOCK-sized chunk of the unified offset space and conditionally
    does (zero | a-bcast | w-bcast) depending on whether ``base`` falls inside
    each sub-region. The runtime ``if`` keeps tail programs from issuing
    masked-out loads/stores for sub-regions they don't touch.

    ``tmp_out_ptr`` may be f32 (stage1 partial buffer) or bf16/fp16 (stage2
    output buffer); the f32 zero we store is implicitly cast to the pointer
    dtype, so the same kernel covers both stages.

    ``a_src_ptr`` is treated as numel==1 (per-tensor scalar); the kernel loads
    it once per program and broadcasts.  ``w_src_ptr`` is numel==E (per-expert)
    and the kernel computes ``expert_idx = offs // w_cols`` to broadcast each
    expert's scale across its ``w_cols`` adjacent output columns.
    """
    pid = tl.program_id(0)
    base = pid * BLOCK
    offs = base + tl.arange(0, BLOCK)

    if DO_TMP:
        if base < tmp_n:
            m = offs < tmp_n
            tl.store(
                tmp_out_ptr + offs,
                tl.zeros([BLOCK], dtype=tl.float32),
                mask=m,
            )

    if EXPAND_A:
        if base < flat_a_n:
            m = offs < flat_a_n
            v = tl.load(a_src_ptr)
            vec = v + tl.zeros([BLOCK], dtype=tl.float32)
            tl.store(flat_a_ptr + offs, vec, mask=m)

    if EXPAND_W:
        if base < flat_w_n:
            m = offs < flat_w_n
            idx = offs // w_cols
            v = tl.load(w_src_ptr + idx, mask=m, other=0.0)
            tl.store(flat_w_ptr + offs, v, mask=m)


def fused_init(
    tmp_out: Optional[torch.Tensor] = None,
    flat_a: Optional[torch.Tensor] = None,
    a_src: Optional[torch.Tensor] = None,
    flat_w: Optional[torch.Tensor] = None,
    w_src: Optional[torch.Tensor] = None,
    w_cols: int = 1,
) -> None:
    """Single fused launch for the stage1 / stage2 init triplet.

    Args:
      tmp_out: buffer to zero in-place. May be ``None`` to skip. Dtype may be
        f32 (stage1 partial accumulator) or bf16/fp16 (stage2 output buffer
        when running in atomic mode); the kernel implicit-casts the stored
        zero so both work.
      flat_a, a_src: when both provided, broadcast ``a_src[0]`` into ``flat_a``
        (scalar -> 1D broadcast).  ``a_src`` MUST be 1-element-readable
        (typically the per-tensor activation scale).
      flat_w, w_src, w_cols: when all provided, broadcast each ``w_src[e]``
        across ``w_cols`` consecutive positions of ``flat_w`` (per-expert ->
        per-row broadcast).  ``w_src`` MUST have ``flat_w.numel() // w_cols``
        elements laid out contiguously.

    Pass ``None`` for any (flat_x, x_src) pair to skip that broadcast (e.g.
    when the caller already has the flat scale and only needs the zero-fill).

    Hot-path note: this fires on every FlyDSL split-K std stage1 call AND on
    every FlyDSL atomic stage2 call.  At small M each Python statement in
    this wrapper costs ~1-2 us of real end-to-end stage time (Triton
    dispatch + this wrapper add ~14 us against a ~32 us kernel body).
    We therefore:
      1. trust the caller's dtype/contiguous contract (only ``moe_kernels``
         calls this) instead of asserting on every call (~8 us strip),
      2. fold per-sub-region argument selection into ``or live`` shortcuts
         that avoid Python conditionals in the kernel arg list,
      3. pick (BLOCK, num_warps) via ``_pick_init_config`` rather than the
         former hardcoded (1024, 4) -- small win but free.
    """
    do_tmp = tmp_out is not None and tmp_out.numel() > 0
    do_a = (
        flat_a is not None
        and a_src is not None
        and flat_a.numel() > 0
    )
    do_w = (
        flat_w is not None
        and w_src is not None
        and flat_w.numel() > 0
    )

    tmp_n = tmp_out.numel() if do_tmp else 0
    a_n = flat_a.numel() if do_a else 0
    w_n = flat_w.numel() if do_w else 0
    max_n = tmp_n if tmp_n > a_n else a_n
    if w_n > max_n:
        max_n = w_n
    if max_n == 0:
        return

    block_n, num_warps = _pick_init_config(max_n)
    grid = (triton.cdiv(max_n, block_n),)

    # Triton needs a valid pointer for every tensor arg even when the
    # corresponding sub-path is disabled; reuse whichever live tensor we have.
    # NB: torch tensors don't support __bool__ for numel>1 so we can't use
    # ``tmp_out or flat_a`` -- explicit ternaries it is.
    live = tmp_out if do_tmp else (flat_a if do_a else flat_w)
    tmp_arg = tmp_out if do_tmp else live
    a_arg = flat_a if do_a else live
    a_src_arg = a_src if do_a else live
    w_arg = flat_w if do_w else live
    w_src_arg = w_src if do_w else live

    _fused_init_kernel[grid](
        tmp_arg, tmp_n,
        a_arg, a_n, a_src_arg,
        w_arg, w_n, w_src_arg,
        w_cols if do_w else 1,
        DO_TMP=int(do_tmp),
        EXPAND_A=int(do_a),
        EXPAND_W=int(do_w),
        BLOCK=block_n,
        num_warps=num_warps,
    )


# Backward-compat alias: stage1 caller imported ``fused_stage1_init`` historically.
fused_stage1_init = fused_init
