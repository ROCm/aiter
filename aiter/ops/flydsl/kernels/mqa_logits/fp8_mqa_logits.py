# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 MQA logits (DeepSeek lightning indexer) -- FlyDSL gfx942/gfx950 kernel builders.

Compute for each query row ``m`` and KV position ``n``
inside that row's window ``[cu_starts[m], cu_ends[m])``::

    logits[m, n] = sum_h ReLU(<Q[m, h, :], K[n, :]> * kv_scale[n]) * weights[m, h]

Low-level ``@flyc.kernel``/``@flyc.jit`` builders only. The public host-facing
API (``flydsl_fp8_mqa_logits``, variant selection/registry) lives in
``aiter.ops.flydsl.fp8_mqa_logits_kernels``.
"""

# No `from __future__ import annotations`: FlyDSL arg typing needs real
# annotation objects, not PEP 563 strings.

from collections.abc import Callable
from dataclasses import dataclass

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..kernels_common import (
    ceildiv,
    warp_reduce_scatter_strided,
    warp_reduce_strided,
)
from ..tensor_shim import GTensor

Vec = fx.Vector


def _make_out_row_t(logits, stride_i64, row_i32):
    """1-D output GTensor for one row, with the row's byte offset folded into
    the base pointer in i64.

    A 2-D (row, col) view computes ``row * stride + col`` in i32 and overflows
    past 2^31 (~46k-square dense outputs), silently mis-writing.
    """
    _byte = fx.Int64(fx.Uint32(row_i32)) * stride_i64 * fx.Int64(4)
    return GTensor(logits, dtype=T.f32, shape=(-1,), static_bytes_offset_i64=_byte)


def _load_pack_i32x8(i32_view, byte_off_i32):
    """32-byte fragment as ``vector<8xi32>`` (frag_bytes=32 atoms).

    buffer_load tops out at dwordx4 (16 bytes), so the fragment is two
    consecutive dwordx4 loads concatenated with vector.shuffle.
    ``byte_off_i32`` must already include this lane's fragment offset so the
    load hits the correct 32-byte chunk for its lane group.
    """
    dword_off = byte_off_i32 // fx.Int32(4)
    v4_lo = i32_view.vec_load((dword_off,), vec_size=4)
    v4_hi = i32_view.vec_load((dword_off + fx.Int32(4),), vec_size=4)
    return Vec(v4_lo).shuffle(v4_hi, list(range(8))).ir_value()


def _make_weight_copy(mma, lane):
    """C-side tiled copy used to distribute the per-head weights.

    The single-tile tiled MMA exists to derive the C-fragment partitioning.
    Sliced by ``lane``, not ``tid``: each wave owns disjoint column tiles.
    """
    tmma = fx.make_tiled_mma(mma, fx.make_layout((1, 1, 1), (0, 0, 0)))
    cp = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
    return cp, fx.make_tiled_copy_C(cp, tmma).get_slice(lane)


def _load_row_weights(weights, H, cp_atom, tc_c, m_tiles, mfma_n, row):
    """``weights[row, :]`` distributed to match this lane's accumulator elements.

    The source is a broadcast along N -- a weight depends on the head (the M
    mode) but not on the KV column -- hence the stride-0 N mode.
    ``partition_S`` handles the degenerate mode and derives the
    accumulator-element -> head mapping.
    """
    it = fx.add_offset(
        fx.get_iter(fx.rocdl.make_buffer_tensor(weights, max_size=True)),
        row * fx.Int32(H),
    )
    wv = fx.Tensor(fx.make_view(it, fx.make_layout((H, mfma_n), (1, 0))))
    pS = tc_c.partition_S(wv)  # ((1, V), M_TILES, 1)
    frag = fx.make_fragment_like(pS)
    fx.copy(cp_atom, pS, frag)
    # Vector(...) flattens the nested value mode into C-TV order, so the
    # epilogue can keep indexing w_row[mi][ii] flat.
    return [Vec(frag[None, mi, 0].load()) for mi in range_constexpr(m_tiles)]


def _emit_col_sum(
    mfma, mma, gemm_kw, a_row, b_pack, w_row, kv_scale, f32_0, reduce=True
):
    """One lane's logit contribution for a single (query row, n-tile) pair.

    ``a_row``/``w_row`` are that row's per-(mi, kk) A-fragments and per-(mi, ii)
    weights; ``b_pack`` is the n-tile's per-kk B-fragments.
    ``kv_scale`` (>=0) is hoisted out of the head sum: ReLU is
    positive-homogeneous, so ReLU(s*x) = s*ReLU(x) and the whole column sum is
    scaled once instead of every head term -- drops M_TILES*ACC_ELEMS muls to one.
    ``reduce=False`` skips the cross-lane head reduce and returns this lane's
    scaled partial, for ``warp_reduce_scatter_strided``. ``kv_scale=None``
    leaves the sum unscaled, for a caller that scales after the reduce.
    """
    col_sum = f32_0
    for mi in range_constexpr(len(a_row)):
        c_frag = fx.make_rmem_tensor(mfma.ACC_ELEMS, fx.Float32)
        c_frag.store(Vec.filled(mfma.ACC_ELEMS, 0.0, fx.Float32))
        for kk in range_constexpr(len(a_row[mi])):
            fx.gemm(mma, c_frag, a_row[mi][kk], b_pack[kk], c_frag, **gemm_kw)
        acc = c_frag.load()
        for ii in range_constexpr(mfma.ACC_ELEMS):
            col_sum = col_sum + Vec(acc)[ii].maximumf(f32_0) * w_row[mi][ii]
    if kv_scale is not None:
        col_sum = col_sum * kv_scale
    if not reduce:
        return col_sum

    # Head-reduce within the wave: the lanes holding distinct heads for a fixed
    # column are those differing in lane // MFMA_N.
    return warp_reduce_strided(col_sum, fx.ReductionOp.ADD, stride=mfma.MFMA_N)


def _emit_acc_issue(mfma, mma, gemm_kw, a_row, b_pack):
    """Phase 1 of a software-pipelined ``_emit_col_sum``: issue the MFMAs only."""
    c_frags = [None] * len(a_row)
    for mi in range_constexpr(len(a_row)):
        c_frag = fx.make_rmem_tensor(mfma.ACC_ELEMS, fx.Float32)
        c_frag.store(Vec.filled(mfma.ACC_ELEMS, 0.0, fx.Float32))
        for kk in range_constexpr(len(a_row[mi])):
            fx.gemm(mma, c_frag, a_row[mi][kk], b_pack[kk], c_frag, **gemm_kw)
        c_frags[mi] = c_frag
    return c_frags


def _emit_acc_reduce(mfma, c_frags, w_row, kv_scale, f32_0, reduce=True):
    """Phase 2 of a software-pipelined ``_emit_col_sum``: consume the accumulators."""
    col_sum = f32_0
    for mi in range_constexpr(len(c_frags)):
        acc = c_frags[mi].load()
        for ii in range_constexpr(mfma.ACC_ELEMS):
            col_sum = col_sum + Vec(acc)[ii].maximumf(f32_0) * w_row[mi][ii]
    if kv_scale is not None:
        col_sum = col_sum * kv_scale
    if not reduce:
        return col_sum
    return warp_reduce_strided(col_sum, fx.ReductionOp.ADD, stride=mfma.MFMA_N)


# ---- LDS-builder epilogues ----
# One function per epilogue shape of ``_build_kernel_mfma_lds_pipe``; the
# builder picks one at build time and calls it once per KV tile. They hold only
# trace-time Python control flow: the one runtime branch, the window-guarded
# store, is ``_EpilogueCtx.store_if``, which has to be defined lexically inside
# the ``@flyc.kernel`` body for the AST rewriter to turn it into an ``scf.if``.


@dataclass(frozen=True)
class _EpilogueCtx:
    """Per-kernel state shared by every tile's epilogue (one wave's view)."""

    mfma: "MfmaAtom"
    mma: object  # the MMA atom
    gemm_kw: dict  # fx.gemm atom state (identity scales)
    a_packs: list  # [row][mi][kk] A-fragments
    w_frag: list  # [row][mi] per-head weights in accumulator order
    starts: list  # [row] window start (fx.Int32)
    ends: list  # [row] window end (fx.Int32)
    lane: object  # fx.Int32 lane in the wave
    lane_div_N: object  # lane // MFMA_N
    f32_0: object
    store_if: Callable  # (pred, out_row_t, col, value) -> None
    iglp: bool = False  # rocdl.iglp_opt(0) ahead of the sw_pipe nest


@dataclass(frozen=True)
class _TileOperands:
    """One KV tile's operands, as read out of its LDS slot."""

    col0: object  # fx.Int32 first KV column of the tile
    cols: list  # [ni] this lane's column of n-tile ni
    b_packs: list  # [ni][kk] B-fragments
    kv_scales: list  # [ni] per-lane column scale, or None (scaled after rs)
    rs_scales: list  # [g] scale of the column a lane owns after the rs, or None
    out_row_ts: list  # [row] 1-D output views


def _store_tile_cols(ctx, j, out_row_t, col, col_sum):
    """Store one head-reduced n-tile: the MFMA_N lanes of group 0 write."""
    in_window = (col >= ctx.starts[j]) & (col < ctx.ends[j])
    is_writer = (ctx.lane_div_N == fx.Int32(0)) & in_window
    ctx.store_if(is_writer, out_row_t, col, col_sum)


def _store_rs_group(ctx, j, out_row_t, col0, g, parts, rs_scale):
    """Reduce-scatter one group of ``64 // MFMA_N`` n-tiles and store it.

    After the reduce-scatter lane ``l`` owns the group's column ``l``
    (tile-major), so all 64 lanes store one contiguous run.
    """
    col_sum = warp_reduce_scatter_strided(
        parts, fx.ReductionOp.ADD, stride=ctx.mfma.MFMA_N
    )
    if rs_scale is not None:
        col_sum = col_sum * rs_scale
    col = col0 + fx.Int32(g * 64) + ctx.lane
    in_window = (col >= ctx.starts[j]) & (col < ctx.ends[j])
    ctx.store_if(in_window, out_row_t, col, col_sum)


def _mfma_and_epilogue_per_tile(ctx, tile):
    """Per n-tile shuffle head-reduce; MFMA_N writer lanes per n-tile."""
    for j in range_constexpr(len(ctx.a_packs)):
        for ni in range_constexpr(len(tile.b_packs)):
            col_sum = _emit_col_sum(
                ctx.mfma,
                ctx.mma,
                ctx.gemm_kw,
                ctx.a_packs[j],
                tile.b_packs[ni],
                ctx.w_frag[j],
                tile.kv_scales[ni],
                ctx.f32_0,
            )
            _store_tile_cols(ctx, j, tile.out_row_ts[j], tile.cols[ni], col_sum)


def _mfma_and_reduce_scatter_epilogue(ctx, tile):
    """``rs_head``: the n-tiles of a group are head-reduced together by
    ``warp_reduce_scatter_strided`` and stored by all 64 lanes."""
    rs_group = 64 // ctx.mfma.MFMA_N
    for j in range_constexpr(len(ctx.a_packs)):
        for g in range_constexpr(len(tile.b_packs) // rs_group):
            parts = [
                _emit_col_sum(
                    ctx.mfma,
                    ctx.mma,
                    ctx.gemm_kw,
                    ctx.a_packs[j],
                    tile.b_packs[g * rs_group + q],
                    ctx.w_frag[j],
                    tile.kv_scales[g * rs_group + q],
                    ctx.f32_0,
                    reduce=False,
                )
                for q in range_constexpr(rs_group)
            ]
            _store_rs_group(
                ctx, j, tile.out_row_ts[j], tile.col0, g, parts, tile.rs_scales[g]
            )


def _mfma_and_epilogue_pipelined(ctx, tile, rs_head):
    """Depth-2 software pipeline over the flattened (row, n-tile) items.

    Issues item k+1's MFMAs before consuming item k's accumulators, so the
    next GEMMs overlap this item's exposed accumulator-read latency. With
    ``rs_head`` the consumed partials are reduce-scattered per group (items
    are row-major, so a group's n-tiles are consecutive items); otherwise each
    item is head-reduced and stored on its own.
    """
    mfma, mma, gemm_kw = ctx.mfma, ctx.mma, ctx.gemm_kw
    rs_group = 64 // mfma.MFMA_N
    items = [
        (j, ni)
        for j in range_constexpr(len(ctx.a_packs))
        for ni in range_constexpr(len(tile.b_packs))
    ]
    if ctx.iglp:
        rocdl.iglp_opt(0)
    cf = _emit_acc_issue(
        mfma, mma, gemm_kw, ctx.a_packs[items[0][0]], tile.b_packs[items[0][1]]
    )
    parts = []
    for k in range_constexpr(len(items)):
        j, ni = items[k]
        if k + 1 < len(items):
            jn, nin = items[k + 1]
            cf_next = _emit_acc_issue(
                mfma, mma, gemm_kw, ctx.a_packs[jn], tile.b_packs[nin]
            )
        col_sum = _emit_acc_reduce(
            mfma, cf, ctx.w_frag[j], tile.kv_scales[ni], ctx.f32_0, reduce=not rs_head
        )
        if not rs_head:
            _store_tile_cols(ctx, j, tile.out_row_ts[j], tile.cols[ni], col_sum)
        else:
            parts.append(col_sum)
            if len(parts) == rs_group:
                g = ni // rs_group
                _store_rs_group(
                    ctx, j, tile.out_row_ts[j], tile.col0, g, parts, tile.rs_scales[g]
                )
                parts = []
        if k + 1 < len(items):
            cf = cf_next


def _emit_row_neg_inf_fill(
    *,
    logits,  # the output kernel arg
    stride_i64,  # i64 row stride in elements (the builders' _stride_i64)
    rows,  # list[fx.Int32]: absolute query rows this thread group owns
    starts,  # list[fx.Int32]: max(cu_starts, 0),        parallel to rows
    ends,  # list[fx.Int32]: min(cu_ends, seq_len_kv), parallel to rows
    seq_len_kv,  # fx.Int32
    seq_len,  # fx.Int32
    by_i32,  # fx.Int32: block_idx.y
    num_splits,  # fx.Int32: grid.y (>= 1)
    fill_range,  # (out_row_t, lo, hi) -> None: thread-strided -inf fill
):
    """``clean_logits`` prefill, fused into the compute kernel.

    Only called when the ``clean_logits`` build flag is set -- it is a
    compile-time specialization, like ``convert_q_fn``/``convert_kv_fn``, so the
    ``clean_logits=False`` kernel contains none of this code at all.

    The epilogue writes column ``c`` of row ``rows[j]`` only for
    ``c in [starts[j], ends[j])``, so the complement inside ``[0, seq_len_kv)``
    is never written by anybody. This emits -inf over exactly that complement,
    which is the two contiguous ranges ``[0, s)`` and ``[e, seq_len_kv)`` with::

        s = min(starts[j], seq_len_kv)
        e = max(ends[j], s)

    ``e``'s max collapses an empty or inverted window (``cu_ends <= cu_starts``,
    or the negative ``cu_ends`` a causal mask yields when s_kv < s_q) to
    "fill the whole row", and keeps the two ranges from overlapping. ``s``'s min
    is load-bearing: ``starts`` is clamped only from below, and the per-row
    output descriptor is built with ``num_records`` = 4 GiB, so an unclamped
    ``cu_starts`` past the end would run off the row and corrupt the next ones
    with no hardware OOB net.

    Rows are already partitioned across grid.x (and across waves in the LDS
    builder), but ``num_splits`` blocks share every row. Since nobody computes
    the complement, it can be partitioned freely: block ``by`` takes its ``by``-th
    equal chunk of each range -- disjoint, gap-free, and balanced across grid.y.
    That is deliberately independent of the tile loop's
    ``tile_start``/``split_cols`` arithmetic and is emitted unconditionally, so a
    block whose ``by`` lands past the union window (zero tile iterations) still
    fills its share.

    MUST be emitted AFTER the tile loop. ``_build_kernel_mfma_lds_pipe`` waits on
    an exact ``s_waitcnt vmcnt(N)`` for its in-flight global->LDS DMAs, and on
    gfx9 vmcnt counts vector STORES too -- a fill store in flight inside the loop
    would inflate the count and let the kernel read a half-written LDS tile.
    Fill and compute addresses are disjoint, so no barrier or ordering is needed.

    ``fill_range`` is supplied by the caller rather than emitted here: it has to
    be defined lexically inside the ``@flyc.kernel`` body for the AST rewriter to
    turn its ``for``/``range`` into an ``scf.for``. It also carries the group
    identity (whole block vs. one wave), which differs between the builders.
    """
    slk = seq_len_kv

    def _split(lo, hi):
        """This block's chunk of ``[lo, hi)``: the ``by``-th of num_splits."""
        chunk = fx.Int32(ceildiv(fx.Uint32(hi - lo), fx.Uint32(num_splits)))
        b_lo = fx.min(lo + by_i32 * chunk, hi)
        return b_lo, fx.min(b_lo + chunk, hi)

    seq_len_m_1 = seq_len - fx.Int32(1)
    for j in range_constexpr(len(rows)):
        # An empty window fills the whole row, so a past-the-end slot must
        # collapse both ranges instead. The descriptor stays on a real row.
        in_rows = rows[j] < seq_len
        out_row_t = _make_out_row_t(logits, stride_i64, fx.min(rows[j], seq_len_m_1))
        s = fx.min(starts[j], slk)
        e = fx.max(ends[j], s)
        s = in_rows.select(s, fx.Int32(0))
        e = in_rows.select(e, slk)
        for lo_i32, hi_i32 in ((fx.Int32(0), s), (e, slk)):
            b_lo, b_hi = _split(lo_i32, hi_i32)
            fill_range(out_row_t, b_lo, b_hi)


# MfmaAtom bundles every MFMA-shape-derived constant plus the atom/fragment
# factories, so the kernel builders carry no hardcoded tile shape. Supporting a
# new MFMA instruction is a new MfmaAtom instance plus a _VARIANT_BUILDERS entry.


#: UE8M0 bias-127 in all four bytes -> multiplier 1.0.
_UE8M0_IDENTITY = 0x7F7F7F7F


def _no_gemm_kwargs():
    return {}


def _identity_scale_kwargs():
    """``fx.gemm`` state for the CDNA4 scaled atoms (K=128/64).

    These instructions always carry ``scale_a``/``scale_b`` UE8M0 operands as
    part of their encoding, and the atom defaults them to 0 -- which is not
    identity in UE8M0. A compile-time identity scale makes the hardware
    microscale a no-op; this kernel applies its own ``kv_scale`` in f32 after
    the MFMA (hoisted out of the ReLU), so no other scale is needed.
    """
    scale = fx.Int32(_UE8M0_IDENTITY)
    return {"scale_a": scale, "scale_b": scale}


def _frag_i64(raw):
    """One lane's i64 A/B fragment (dense CDNA3 atoms) as a register tensor."""
    frag = fx.make_rmem_tensor(1, fx.Int64)
    frag.store(Vec.from_elements([fx.Int64(raw)]))
    return frag


def _frag_i32x8(raw):
    """One lane's vector<8xi32> A/B fragment (CDNA4 scaled atoms)."""
    frag = fx.make_rmem_tensor(8, fx.Int32)
    frag.store(Vec(raw))
    return frag


@dataclass(frozen=True)
class MfmaAtom:
    """MFMA-shape descriptor for the fp8 MQA-logits kernel.

    Fields
    ------
    name : str
        Shape tag, e.g. ``"16x16x32"``.
    MFMA_M, MFMA_N, MFMA_K : int
        Output tile is MFMA_M x MFMA_N; MFMA_K fp8 elements reduced per step.
    make_atom : Callable
        ``() -> fx`` MMA atom. Must be called inside the ``@flyc.kernel`` body.
    make_frag : Callable
        Wraps one lane's raw A/B fragment value in the rank-1 register tensor
        ``fx.gemm`` takes. Rank-1 operands short-circuit to a single
        ``MmaAtomCall``, so the kernel's hand-computed fragment addressing is
        untouched and the atom bitcasts anything whose type does not match.
    frag_bytes : int
        A/B fragment bytes owned by one lane for one K-step. 8 for the dense
        atoms (one i64 load).
    gemm_kwargs : Callable
        ``() -> dict`` of atom state passed to ``fx.gemm`` (the identity
        scales for the scaled atoms; empty for the dense ones).
    kname_tag : str | None
        Shape tag used in the generated kernel symbol name. ``None`` means
        ``f"mfma{name}"``. ``_MFMA16`` pins the bare ``"mfma"`` it has always
        used so its generated symbols (and therefore its ISA) stay unchanged.
    """

    name: str
    MFMA_M: int
    MFMA_N: int
    MFMA_K: int
    make_atom: Callable
    make_frag: Callable
    frag_bytes: int = 8
    gemm_kwargs: Callable = _no_gemm_kwargs
    kname_tag: str | None = None

    @property
    def ACC_ELEMS(self) -> int:
        """f32 accumulator elements per lane (``vec<ACC_ELEMS x f32>``)."""
        return self.MFMA_M * self.MFMA_N // 64



#: gfx942/CDNA3 dense MFMA: 16x16 output tile, K=32 fp8 elements/step.
#: A-fragment layout: lane l -> A[row=l%16, k=(l//16)*8 + 0..7], col=l%16.
#: Writer lanes: l//16 == 0 (16 distinct output columns per tile).
_MFMA16 = MfmaAtom(
    name="16x16x32",
    MFMA_M=16,
    MFMA_N=16,
    MFMA_K=32,
    make_atom=lambda: fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ)),
    make_frag=_frag_i64,
    kname_tag="mfma",
)

#: gfx950/CDNA4 scaled MFMA: 16x16 output tile, K=128 fp8f6f4 elements/step.
#: Same accumulator layout as _MFMA16, because tv_layout_c depends only on
#: (M, N), not on the reduction depth. A-fragment: vector<8xi32> (32 bytes/lane),
#: 4x _MFMA16's, tracking the 4x K increase. Requires native FN operands (this
#: instruction rejects FNUZ) and, via the generic ``D % MFMA_K`` assert,
#: head_size % 128 == 0.
_MFMA16_K128 = MfmaAtom(
    name="16x16x128",
    MFMA_M=16,
    MFMA_N=16,
    MFMA_K=128,
    make_atom=lambda: fx.make_mma_atom(
        fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN)
    ),
    make_frag=_frag_i32x8,
    frag_bytes=32,
    gemm_kwargs=_identity_scale_kwargs,
)

#: gfx950/CDNA4 scaled MFMA: 32x32 output tile, K=64 fp8f6f4 elements/step.
#: Accumulator is vec<16 x f32> whose two lane groups interleave in blocks of 4
#: (heads 0..3, 8..11, 16..19, 24..27) -- non-contiguous, which is exactly why
#: the weight distribution is left to the layout algebra rather than typed out.
#: A-fragment: vector<8xi32> (32 bytes/lane). Requires native FN operands
#: (rejects FNUZ); serves head_size 64 and 128.
_MFMA32_K64 = MfmaAtom(
    name="32x32x64",
    MFMA_M=32,
    MFMA_N=32,
    MFMA_K=64,
    make_atom=lambda: fx.make_mma_atom(
        fx.rocdl.cdna4.MFMA_Scale(32, 32, 64, fx.Float8E4M3FN)
    ),
    make_frag=_frag_i32x8,
    frag_bytes=32,
    gemm_kwargs=_identity_scale_kwargs,
)


def _build_kernel_mfma_r_w(
    *,
    num_heads: int,
    head_size: int,
    block_kv: int,
    rows_per_block: int,
    waves_per_block: int,
    mfma: MfmaAtom = _MFMA16,
    convert_q_fn: bool = False,
    convert_kv_fn: bool = False,
    clean_logits: bool = True,
):
    """Multi-row, multi-wave MFMA kernel.

    ``rows_per_block`` query rows share one KV tile load (cuts KV traffic by RPB).
    ``waves_per_block`` waves execute per block; each wave owns a disjoint slice of
    the BKV column tiles (``N_TILES // WPB`` tiles per wave), so all WPB waves can
    execute in parallel with no cross-wave LDS or barrier.

    Thread decomposition:
      * ``tid = wave * 64 + lane``  (tid: 0..MR_BLOCK_THREADS-1)
      * Wave ``w`` owns n-tiles ``[w*N_TILES_PER_WAVE, (w+1)*N_TILES_PER_WAVE)``
        within each BKV tile.
      * A-operand (Q) layout and head-reduce are per-lane within the wave (width 64).

    Grid: ``(ceil(seq_len / RPB), num_splits, 1)``. The last block may be short.
    The host may split each row's KV window across ``grid.y``.
    """
    H = num_heads
    D = head_size
    BKV = block_kv
    RPB = rows_per_block
    WPB = waves_per_block
    MR_BLOCK_THREADS = 64 * WPB

    # MFMA tile dims come from the atom: MFMA_M x MFMA_N output tile, MFMA_K
    # fp8 elements reduced per MFMA step.
    MFMA_M = mfma.MFMA_M
    MFMA_N = mfma.MFMA_N
    MFMA_K = mfma.MFMA_K
    FRAG_BYTES = mfma.frag_bytes

    assert H % MFMA_M == 0, f"num_heads={H} must be a multiple of MFMA_M={MFMA_M}"
    assert BKV % MFMA_N == 0, f"block_kv={BKV} must be a multiple of MFMA_N={MFMA_N}"
    assert D % MFMA_K == 0, f"head_size={D} must be a multiple of MFMA_K={MFMA_K}"
    assert RPB >= 1, "rows_per_block must be >= 1"
    assert WPB >= 1, "waves_per_block must be >= 1"
    # The CDNA4 scaled atoms consume native FN operands and reject FNUZ, so the
    # in-kernel FN->FNUZ patch must never be combined with them. The host only
    # sets these flags on gfx942 (where only dense atoms are used), so this is a
    # guard against future mis-wiring rather than a reachable path.
    assert not (mfma.frag_bytes == 32 and (convert_q_fn or convert_kv_fn)), (
        f"atom {mfma.name} requires native FN operands; "
        "FN->FNUZ conversion is not supported for it"
    )
    N_TILES = BKV // MFMA_N  # total column-tiles per BKV block
    assert (
        N_TILES % WPB == 0
    ), f"BKV/MFMA_N={N_TILES} must be divisible by waves_per_block={WPB}"
    M_TILES = H // MFMA_M  # head row-tiles
    K_STEPS = D // MFMA_K  # MFMA K-steps over the head dim
    N_TILES_PER_WAVE = N_TILES // WPB  # column-tiles per wave

    _cvt_tag = ""
    if convert_q_fn:
        _cvt_tag += "_cq"
    if convert_kv_fn:
        _cvt_tag += "_ck"
    # Only the non-default is tagged, so the common clean_logits=True symbols
    # keep the names they have always had (same convention as _cvt_tag).
    _cl_tag = "" if clean_logits else "_nocl"
    _shape_tag = mfma.kname_tag or f"mfma{mfma.name}"
    _kname = (
        f"fp8_mqa_logits_H{H}_D{D}_bkv{BKV}_{_shape_tag}_r{RPB}_w{WPB}"
        f"{_cvt_tag}{_cl_tag}_flydsl"
    )

    @flyc.kernel(name=_kname, known_block_size=[MR_BLOCK_THREADS, 1, 1])
    def kernel(
        Q: fx.Tensor,  # [seq_len, H, D]       fp8 (bytes passed raw)
        KV: fx.Tensor,  # [seq_len_kv, D]       fp8 (bytes passed raw)
        kv_scales: fx.Tensor,  # [seq_len_kv]          f32
        weights: fx.Tensor,  # [seq_len, H]          f32
        cu_starts: fx.Tensor,  # [seq_len]             i32
        cu_ends: fx.Tensor,  # [seq_len]             i32
        logits: fx.Tensor,  # [seq_len, seq_len_kv] f32
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_logits_s: fx.Int32,
        num_splits: fx.Int32,  # grid.y KV-column splits (1 == no split)
    ):
        f32_0 = fx.Float32(0.0)
        mma = mfma.make_atom()
        gemm_kw = mfma.gemm_kwargs()

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        # Rows are handed out in reverse (bid=0 -> last rows) as a load-balancing
        # heuristic: KV windows tend to be longer for later query rows, so this
        # gets the heaviest work scheduled first instead of last.
        n_blocks = fx.Int32(ceildiv(fx.Uint32(seq_len), fx.Uint32(RPB)))
        r0 = (n_blocks - bid - fx.Int32(1)) * fx.Int32(RPB)

        # Decompose tid into wave index and in-wave lane.
        wave = tid // fx.Int32(64)
        lane = tid % fx.Int32(64)
        lane_div_N = lane // fx.Int32(MFMA_N)
        lane_mod_N = lane % fx.Int32(MFMA_N)
        # Byte offset of this lane's fragment within the K-step (FRAG_BYTES per
        # lane group). 8 for the dense atoms.
        lane_frag_off = lane_div_N * fx.Int32(FRAG_BYTES)
        cp_4xfp32, tc_c_w = _make_weight_copy(mma, lane)

        # fp8 operands are read 8 bytes at a time as 2 i32 dwords (v8i8
        # buffer_load fails to lower on gfx942), bitcast to i64 for the MFMA.
        q_i32 = GTensor(Q, dtype=T.i32, shape=(-1,))
        kv_i32 = GTensor(KV, dtype=T.i32, shape=(-1,))
        sc_t = GTensor(kv_scales, dtype=T.f32, shape=(-1,))
        cs_t = GTensor(cu_starts, dtype=T.i32, shape=(-1,))
        ce_t = GTensor(cu_ends, dtype=T.i32, shape=(-1,))
        # Per-row 1-D output view: the row's i64 byte offset goes into the base
        # pointer so the remaining column offset stays in i32. A 2-D (row, col)
        # view computes row * stride + col in i32 and overflows past 2^31
        # (~46k-square dense outputs), silently mis-writing.
        _stride_i64 = fx.Int64(fx.Uint32(stride_logits_s))

        def _load_pack_i64(i32_view, byte_off_i32):
            dword_off = byte_off_i32 // fx.Int32(4)
            v2 = i32_view.vec_load((dword_off,), vec_size=2)
            return Vec(v2).bitcast(fx.Int64)[0].ir_value()

        def _load_frag(i32_view, base_i32, k_byte_i32, convert_fn):
            """Load one lane's A/B fragment for one K-step.

            Dense atoms (frag_bytes=8) take a 64-bit load and may need the
            FN->FNUZ patch; the CDNA4 scaled atoms (frag_bytes=32) take the
            two-dwordx4 path and are always native FN.
            """
            off = base_i32 + k_byte_i32 + lane_frag_off
            if const_expr(FRAG_BYTES == 32):
                return mfma.make_frag(_load_pack_i32x8(i32_view, off))
            raw = _load_pack_i64(i32_view, off)
            return mfma.make_frag(_fn_to_fnuz_i64(raw) if convert_fn else raw)

        def _fn_to_fnuz_i64(raw_i64):
            """Map FN byte 0x80 (neg-zero) -> 0x00 in 8 packed fp8 bytes."""

            def _fix_i32(src):
                """Zero every 0x80 byte of one dword of 4 packed fp8 bytes.

                ``>>`` on a signed Int32 is arithmetic, but the ``& 0xFF``
                immediately after keeps only the byte being examined, so this
                matches the logical shift it replaced.
                """
                result = fx.Int32(0)
                for byte_idx in range_constexpr(4):
                    shift = fx.Int32(byte_idx * 8)
                    byte_val = (src >> shift) & fx.Int32(0xFF)
                    cleaned = (byte_val == fx.Int32(0x80)).select(fx.Int32(0), byte_val)
                    result = result | (cleaned << shift)
                return result

            # The widen/narrow steps stay unsigned: they move a bit pattern, so
            # a sign-extending ext would corrupt the high dword.
            raw = fx.Uint64(raw_i64)
            lo_fix = _fix_i32(fx.Int32(raw))
            hi_fix = _fix_i32(fx.Int32(raw >> 32))
            lo_64 = fx.Int64(fx.Uint32(lo_fix))
            hi_64 = fx.Int64(fx.Uint32(hi_fix)) << fx.Int64(32)
            return (lo_64 | hi_64).ir_value()

        # ---- Preload window bounds, Q frags, and weights for all RPB rows ----
        # A-operand layout is per in-wave lane, so `lane` (not `tid`) indexes Q.
        # Past-the-end slots reuse the last real row's addresses and take the
        # empty window [seq_len_kv, 0), so they are not stored. Starting it at
        # seq_len_kv, not 0, keeps the slot out of the block's union window.
        starts = [None] * RPB
        ends = [None] * RPB
        a_packs = [None] * RPB
        w_frag = [None] * RPB
        seq_len_m_1 = seq_len - fx.Int32(1)

        for j in range_constexpr(RPB):
            row = r0 + fx.Int32(j)
            row_ld = fx.min(row, seq_len_m_1)
            in_rows = row < seq_len
            ss = fx.max(fx.Int32(cs_t[row_ld]), fx.Int32(0))
            ee = fx.min(fx.Int32(ce_t[row_ld]), seq_len_kv)
            starts[j] = in_rows.select(ss, seq_len_kv)
            ends[j] = in_rows.select(ee, fx.Int32(0))

            # lane -> Q[row, h = mi*MFMA_M + lane%MFMA_N,
            #            d = kk*MFMA_K + (lane//MFMA_N)*8 + 0..7]
            row_a = [[None] * K_STEPS for _ in range_constexpr(M_TILES)]
            for mi in range_constexpr(M_TILES):
                h_a = fx.Int32(mi * MFMA_M) + lane_mod_N
                row_h = row_ld * fx.Int32(H) + h_a
                base_a = row_h * fx.Int32(D)
                for kk in range_constexpr(K_STEPS):
                    row_a[mi][kk] = _load_frag(
                        q_i32, base_a, fx.Int32(kk * MFMA_K), convert_q_fn
                    )
            a_packs[j] = row_a

            w_frag[j] = _load_row_weights(
                weights, H, cp_4xfp32, tc_c_w, M_TILES, MFMA_N, row_ld
            )

        # ---- Union window across all RPB rows ----
        tile_start = starts[0]
        tile_end = ends[0]
        for j in range_constexpr(1, RPB):
            tile_start = fx.min(tile_start, starts[j])
            tile_end = fx.max(tile_end, ends[j])
        # Align tile_start down to BKV boundary.
        tile_start = (tile_start // fx.Int32(BKV)) * fx.Int32(BKV)
        # Collapse an empty union window to a zero-width one at tile_start.
        # ``ends`` is clamped above by seq_len_kv but not below, so a row whose
        # cu_ends is negative or any row with cu_ends <= cu_starts
        # can leave tile_end < tile_start.
        tile_end = fx.max(tile_end, tile_start)

        # ---- KV-column split across grid.y. Block (.,by) takes a BKV-aligned
        # slice of the union window; logits[m,n] are independent across n, so
        # this is pure parallelism with no reduction. The slices tile [start,end)
        # exactly (disjoint, gap-free), so each column has one writer.
        # num_splits==1 collapses to the full window (by==0). ----
        by = fx.block_idx.y
        win_tiles = fx.Int32(ceildiv(fx.Uint32(tile_end - tile_start), fx.Uint32(BKV)))
        split_cols = fx.Int32(
            ceildiv(fx.Uint32(win_tiles), fx.Uint32(num_splits))
        ) * fx.Int32(BKV)
        tile_start = tile_start + by * split_cols
        tile_end = fx.min(tile_start + split_cols, tile_end)

        for col0_iv in range(tile_start, tile_end, fx.Int32(BKV)):
            col0 = fx.Int32(col0_iv)

            # ---- Load B-frags: wave w owns its own disjoint slice of n-tiles
            # [w*N_TILES_PER_WAVE, (w+1)*N_TILES_PER_WAVE) (no cross-wave sharing). ----
            wave_ni_base = wave * fx.Int32(N_TILES_PER_WAVE)
            b_packs = [[None] * K_STEPS for _ in range_constexpr(N_TILES_PER_WAVE)]
            kv_scales_tile = [None] * N_TILES_PER_WAVE
            cols = [None] * N_TILES_PER_WAVE
            for ni in range_constexpr(N_TILES_PER_WAVE):
                abs_ni = wave_ni_base + fx.Int32(ni)
                col = col0 + abs_ni * fx.Int32(MFMA_N) + lane_mod_N
                cols[ni] = col
                col_clamped = fx.min(col, seq_len_kv - fx.Int32(1))
                kv_scales_tile[ni] = fx.Float32(sc_t[col_clamped])
                base_b = col_clamped * fx.Int32(D)
                for kk in range_constexpr(K_STEPS):
                    b_packs[ni][kk] = _load_frag(
                        kv_i32, base_b, fx.Int32(kk * MFMA_K), convert_kv_fn
                    )

            # ---- Per-row MFMA + epilogue (inner loop over RPB rows) ----
            for j in range_constexpr(RPB):
                row = r0 + fx.Int32(j)
                out_row_t = _make_out_row_t(logits, _stride_i64, row)
                for ni in range_constexpr(N_TILES_PER_WAVE):
                    col = cols[ni]
                    col_sum = _emit_col_sum(
                        mfma,
                        mma,
                        gemm_kw,
                        a_packs[j],
                        b_packs[ni],
                        w_frag[j],
                        kv_scales_tile[ni],
                        f32_0,
                    )

                    # Only lane_div_N==0 lanes hold the MFMA_N distinct columns.
                    # `col >= start` is required: the tile loop is BKV-aligned
                    # below `start`, so it guards the -inf that the fused fill
                    # below writes into [aligned_start, start).
                    in_window = (col >= starts[j]) & (col < ends[j])
                    is_writer = (lane_div_N == fx.Int32(0)) & in_window

                    # Via a closure, not a bare `out_row_t[col] = ...` in the
                    # branch: the rewriter reads a subscript store as an
                    # assignment to `out_row_t` and tries to carry the
                    # TensorView out of the scf.if as a result.
                    def _store():
                        out_row_t[col] = col_sum  # noqa: B023

                    if is_writer:
                        _store()

        # ---- Fused clean_logits prefill (must come after the tile loop) ----
        if const_expr(clean_logits):
            neg_inf = fx.Float32(float("-inf"))

            def _store_neg_inf(t, c):
                t[c] = neg_inf

            def _fill_range(out_row_t, lo_i32, hi_i32):
                """Thread-strided ``out_row_t[c] = -inf`` over ``[lo, hi)``.

                All MR_BLOCK_THREADS threads of the block cooperate; thread ``t``
                writes ``lo+t, lo+t+nthreads, ...``, so consecutive lanes cover
                consecutive dwords and a wave iteration coalesces into one
                256-byte store. Zero-trip on an empty range (lb >= ub).

                Plain dwords on purpose: the fill is bandwidth-bound, not
                store-issue-bound, so dwordx4 buys nothing.
                """
                for c in range(lo_i32 + tid, hi_i32, fx.Int32(MR_BLOCK_THREADS)):
                    _store_neg_inf(out_row_t, fx.Int32(c))

            _emit_row_neg_inf_fill(
                logits=logits,
                stride_i64=_stride_i64,
                rows=[r0 + fx.Int32(j) for j in range_constexpr(RPB)],
                starts=starts,
                ends=ends,
                seq_len_kv=seq_len_kv,
                seq_len=seq_len,
                by_i32=by,
                num_splits=num_splits,
                fill_range=_fill_range,
            )

    @flyc.jit
    def launch_fp8_mqa_logits_mfma_r_w(
        Q: fx.Tensor,
        KV: fx.Tensor,
        kv_scales: fx.Tensor,
        weights: fx.Tensor,
        cu_starts: fx.Tensor,
        cu_ends: fx.Tensor,
        logits: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_logits_s: fx.Int32,
        num_splits: fx.Int32,
        stream: fx.Stream,
    ):
        gx = fx.Int64(fx.Int32(ceildiv(fx.Uint32(seq_len), fx.Uint32(RPB))))
        gy = fx.Int64(num_splits)
        kernel._func.__name__ = _kname
        kernel(
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            logits,
            seq_len,
            seq_len_kv,
            stride_logits_s,
            num_splits,
        ).launch(grid=(gx, gy, 1), block=(MR_BLOCK_THREADS, 1, 1), stream=stream)

    return launch_fp8_mqa_logits_mfma_r_w


def _build_kernel_mfma_lds_pipe(
    *,
    num_heads: int,
    head_size: int,
    block_kv: int,
    rows_per_block: int,
    waves_per_block: int,
    mfma: MfmaAtom,
    convert_q_fn: bool = False,
    convert_kv_fn: bool = False,
    clean_logits: bool = True,
    num_buffers: int = 2,
    prefetch_depth: int = 2,
    sw_pipe: bool = False,
    rs_head: bool = False,
    lds_scales: bool = False,
):
    """LDS multi-buffered variant for gfx950 MfmaAtoms (scaled CDNA4 atoms).

    Parallel to ``_build_kernel_mfma_r_w`` but stages KV through a multi-slot LDS
    buffer filled by async global->LDS DMA (``BufferCopyLDS`` atoms),
    with an explicit software pipeline (prefetch tile 0..PD-1, then per-tile
    ``s_waitcnt`` + prefetch(i+PD) + compute).

    Work partition:
      * All ``WPB`` waves cooperatively load one ``BKV``-wide K-tile into LDS and
        all read it. Each wave owns a disjoint group of ``RPW`` query rows and
        iterates over all ``N_TILES`` columns of the shared LDS tile.
      * A block owns ``ROWS_PER_BLOCK = RPW * WPB`` query rows (wave ``w`` owns
        rows ``[w*RPW, (w+1)*RPW)``). KV reuse factor becomes ``RPW * WPB``.

    A-frags are loaded from global memory to registers; B-frags are read from LDS.
    Both loads are tiled copies partitioned by the atom's tiled MMA.
    The epilogue is identical to the direct-load builder, unless ``rs_head``:
    then the ``64 // MFMA_N`` n-tiles of a group are head-reduced together by
    ``warp_reduce_scatter_strided`` and stored by all 64 lanes as one contiguous
    run, instead of one shuffle butterfly and one MFMA_N-lane store per n-tile.
    ``lds_scales`` stages the tile's kv_scales into the LDS slot with the same
    async DMA as the KV bytes, instead of a blocking per-n-tile global load
    that every wave repeats.
    """
    H = num_heads
    D = head_size
    BKV = block_kv
    RPW = rows_per_block  # rows per WAVE here (block owns RPW*WPB rows)
    WPB = waves_per_block
    MR_BLOCK_THREADS = 64 * WPB
    ROWS_PER_BLOCK = RPW * WPB

    assert mfma.frag_bytes == 32, (
        "_build_kernel_mfma_lds_pipe currently supports only the CDNA4 scaled "
        "atoms (frag_bytes=32)."
    )
    assert (
        H % mfma.MFMA_M == 0
    ), f"Number of heads must be a multiple of MFMA_M ({mfma.MFMA_M})"
    assert (
        BKV % mfma.MFMA_N == 0
    ), f"Block KV size must be a multiple of MFMA_N ({mfma.MFMA_N})"
    assert (
        D % mfma.MFMA_K == 0
    ), f"Head size must be a multiple of MFMA_K ({mfma.MFMA_K})"
    assert RPW >= 1 and WPB >= 1, "Rows per wave and waves per block must be >= 1"

    N_TILES = BKV // mfma.MFMA_N
    M_TILES = H // mfma.MFMA_M
    # Unroll the tile loop by NUM_BUFFERS so each slot's LDS offset is a
    # compile-time constant.
    UNROLL_KV_TILE_LOOP = rs_head and not sw_pipe and not lds_scales
    # Hand the sw_pipe MFMA/reduce interleave to the AMDGPU pipeline solver on
    # the low-occupancy routes.
    IGLP_SW_PIPE = sw_pipe and WPB <= 2
    K_STEPS = D // mfma.MFMA_K
    # n-tiles per reduce-scatter group: one per MFMA_N-lane group of the wave.
    RS_GROUP = 64 // mfma.MFMA_N
    assert not rs_head or N_TILES % RS_GROUP == 0, (
        f"rs_head needs N_TILES ({N_TILES}) to be a multiple of "
        f"64 // MFMA_N ({RS_GROUP})"
    )
    if sw_pipe:
        _mfma_and_epilogue = lambda ctx, tile: _mfma_and_epilogue_pipelined(ctx, tile, rs_head)  # noqa: E731
    elif rs_head:
        _mfma_and_epilogue = _mfma_and_reduce_scatter_epilogue
    else:
        _mfma_and_epilogue = _mfma_and_epilogue_per_tile

    # LDS multi-buffer: NUM_BUFFERS slots of [BKV, D] fp8 (row-major, row == KV
    # column index). Addressed as i32 dwords for the vector reads.
    #
    # Pipeline depth is set by PREFETCH_DEPTH (tiles kept in flight during the
    # per-tile compute).  The tile-i+PREFETCH_DEPTH prefetch targets slot
    # (i+PD)%NB; when NB > PD that slot differs from the one being read (slot
    # i%NB), so the reader-before-writer barrier ("barrier B") can be dropped.
    # NB == PD (e.g. the 2-buffer/depth-2 case) reuses the just-read slot
    # and still needs barrier B.
    NUM_BUFFERS = num_buffers
    PREFETCH_DEPTH = prefetch_depth
    assert (
        NUM_BUFFERS >= PREFETCH_DEPTH >= 1
    ), f"need num_buffers({NUM_BUFFERS}) >= prefetch_depth({PREFETCH_DEPTH}) >= 1"
    _need_barrier_b = NUM_BUFFERS <= PREFETCH_DEPTH
    KV_BYTES = BKV * D  # fp8, 1 byte/elem
    # lds_scales: the tile's BKV f32 kv_scales follow the KV bytes in the slot,
    # padded to whole 64-lane DMAs (one 4-byte element per lane); 256 B keeps
    # the slot a multiple of the DMA's 128-byte destination alignment.
    SCALE_DMAS = ceildiv(BKV, 64) if lds_scales else 0
    SCALE_BYTES = SCALE_DMAS * 64 * 4
    # Columns spanned by a tile's buffer views (the scale DMAs round up to 64).
    TILE_COLS = max(BKV, 64 * ceildiv(BKV, 64))
    SLOT_BYTES = KV_BYTES + SCALE_BYTES
    SLOT_I32 = SLOT_BYTES // 4
    SCALE_DW = KV_BYTES // 4  # slot-relative dword offset of the scales
    # gfx950 global->LDS DMA supports size=16 (dwordx4).
    DMA_BYTES = 16
    assert KV_BYTES % (MR_BLOCK_THREADS * DMA_BYTES) == 0, (
        f"KV_BYTES={KV_BYTES} must be divisible by "
        f"MR_BLOCK_THREADS*DMA_BYTES={MR_BLOCK_THREADS * DMA_BYTES}"
    )
    NUM_KV_DMAS = KV_BYTES // (MR_BLOCK_THREADS * DMA_BYTES)
    # Every wave issues every scale DMA (identical bytes, so the overlapping
    # writes are benign): that keeps the per-wave VMEM count uniform, which the
    # compile-time vmcnt below relies on.
    NUM_ASYNC_LOADS = NUM_KV_DMAS + SCALE_DMAS
    # vmcnt to leave outstanding at the top of each tile: the DMAs of the
    # PREFETCH_DEPTH-1 tiles queued behind the one about to be read.
    _WAIT_VMCNT = (PREFETCH_DEPTH - 1) * NUM_ASYNC_LOADS
    assert _WAIT_VMCNT <= 63, (
        f"prefetch_depth={PREFETCH_DEPTH} x {NUM_ASYNC_LOADS} DMAs/tile needs "
        f"vmcnt={_WAIT_VMCNT}, past the 63 the gfx9 encoding holds; "
        "lower prefetch_depth or block_kv, or raise waves_per_block"
    )

    # XOR swizzle (bank-conflict avoidance) of the [BKV, D] fp8 tile, as a
    # SwizzleType over the slot's byte (fp8 element) offsets. The DMA writes LDS
    # lane-linearly, so the swizzle is applied on the global source side 
    # and on the read view.
    #
    # gfx950 LDS is 64 banks x 4 B, and a ds_read_b128 serves 16 lanes per
    # pass, i.e. one 256-byte bank line. The 16 lanes of a B-fragment read take
    # 16 consecutive KV columns n at the same head-dim chunk, so with rows of
    # D bytes they cover only the 256/D distinct row positions within a bank
    # line. XOR-ing the 32-byte chunk index with the row bits above the bank
    # line (n >> log2(256/D)) spreads them over the NC chunk positions too.
    #
    # Chunks stay whole (base = log2 frag_bytes), so every 16-byte DMA write
    # and both ds_read_b128 of a fragment stay contiguous.
    NC = D // mfma.frag_bytes  # B-fragment chunks per column
    _BANK_LINE_BYTES = 256  # 64 banks x 4 B
    _FRAG_LOG2 = mfma.frag_bytes.bit_length() - 1
    LDS_SWZ = (
        NC.bit_length() - 1,
        _FRAG_LOG2,
        _BANK_LINE_BYTES.bit_length() - 1 - _FRAG_LOG2,
    )

    # The B-fragment reads view one n-tile (MFMA_N KV columns) at a time, with
    # the n-tile's offset applied to the base pointer rather than through the
    # swizzle. That is exact only if the offset is a multiple of the swizzle
    # period, and it lets every n-tile share the lane's read addresses
    # (immediate offsets).
    NTILE_BYTES = mfma.MFMA_N * D
    assert NTILE_BYTES % (1 << sum(LDS_SWZ)) == 0, (
        f"n-tile ({NTILE_BYTES} bytes) must be a multiple of the swizzle period"
    )

    # The global->LDS DMA requires its destination LDS address to be at
    # least 128-byte aligned
    @fx.struct
    class SharedStorage:
        slots: fx.Array[fx.Int32, NUM_BUFFERS * SLOT_I32, 128]

    _cl_tag = "" if clean_logits else "_nocl"
    _pd_tag = "" if PREFETCH_DEPTH == 2 else f"_pd{PREFETCH_DEPTH}"
    _kname = (
        f"fp8_mqa_logits_H{H}_D{D}_mfma{mfma.name}"
        f"_bkv{BKV}_r{RPW}_w{WPB}_lds{NUM_BUFFERS}{_pd_tag}"
        f"{'_swp' if sw_pipe else ''}"
        f"{'_rs' if rs_head else ''}{'_ls' if lds_scales else ''}{_cl_tag}_flydsl"
    )

    @flyc.kernel(name=_kname, known_block_size=[MR_BLOCK_THREADS, 1, 1])
    def kernel(
        Q: fx.Tensor,
        KV: fx.Tensor,
        kv_scales: fx.Tensor,
        weights: fx.Tensor,
        cu_starts: fx.Tensor,
        cu_ends: fx.Tensor,
        logits: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_logits_s: fx.Int32,
        num_splits: fx.Int32,
    ):
        f32_0 = fx.Float32(0.0)
        mma = mfma.make_atom()
        gemm_kw = mfma.gemm_kwargs()

        tid = fx.thread_idx.x
        bid = fx.block_idx.x

        # Reverse row order.
        n_blocks = fx.Int32(ceildiv(fx.Uint32(seq_len), fx.Uint32(ROWS_PER_BLOCK)))
        block_row0 = fx.Int32((n_blocks - bid - fx.Int32(1)) * fx.Int32(ROWS_PER_BLOCK))

        wave = tid // fx.Int32(64)
        lane = tid % fx.Int32(64)
        lane_div_N = lane // fx.Int32(mfma.MFMA_N)
        lane_mod_N = lane % fx.Int32(mfma.MFMA_N)
        cp_4xfp32, tc_c_w = _make_weight_copy(mma, lane)

        # One wave's tiled MMA: it owns the lane -> (row, K) mapping of the A
        # (Q heads x head dim) and B (KV columns x head dim) fragments, and the
        # tiled copies below derive every fragment load from it.
        tiled_mma = fx.make_tiled_mma(mma, fx.make_layout((1, 1, 1), (0, 0, 0)))
        thr_mma = tiled_mma.thr_slice(lane)
        # Q: one lane's 32-byte fragment is two 16-byte buffer loads.
        q_copy = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), 8)
        tc_q = fx.make_tiled_copy_A(q_copy, tiled_mma).get_slice(lane)
        # KV: one lane's 32-byte fragment is one LDS load (a ds_read_b128 pair).
        kv_copy = fx.make_copy_atom(fx.UniversalCopy(mfma.frag_bytes * 8), 8)
        tc_kv = fx.make_tiled_copy_B(kv_copy, tiled_mma).get_slice(lane)

        wave_s = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, wave.ir_value()))

        # First row owned by this wave.
        wave_row0 = block_row0 + wave * fx.Int32(RPW)

        # [row, head, head dim] fp8.
        q_buf = fx.rocdl.make_buffer_tensor(
            fx.Tensor(
                fx.make_view(
                    fx.get_iter(Q), fx.make_layout((seq_len, H, D), (H * D, D, 1))
                )
            )
        )
        cs_t = fx.rocdl.make_buffer_tensor(cu_starts)
        ce_t = fx.rocdl.make_buffer_tensor(cu_ends)
        _stride_i64 = fx.Int64(fx.Uint32(stride_logits_s))

        # ---- LDS region: one flat i32 array of NUM_BUFFERS slots ----
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().slots.ptr
        lds_f32 = fx.recast_iter(fx.Float32, lds_ptr)
        kv_elem = fx.get_iter(KV).element_type
        lds_kv = fx.recast_iter(kv_elem, lds_ptr)
        swz = fx.static(fx.SwizzleType.get(*LDS_SWZ))
        # Read view of one staged n-tile, [KV column, head dim]. The n-tile's
        # offset is applied to the base pointer, outside the swizzle (see
        # NTILE_BYTES), so every n-tile reuses the lane's read addresses.
        kv_read_layout = fx.make_composed_layout(
            swz, fx.make_layout((mfma.MFMA_N, D), (D, 1))
        )

        def _read_b_frags(slot_byte, ni):
            """All K-steps of n-tile ``ni``'s B-fragments out of a staged tile."""
            s_kv = fx.Tensor(
                fx.make_view(
                    fx.add_offset(lds_kv, slot_byte + fx.Int32(ni * NTILE_BYTES)),
                    kv_read_layout,
                )
            )
            frag = thr_mma.make_fragment_B(s_kv)  # (V, 1, K_STEPS)
            fx.copy(kv_copy, tc_kv.partition_S(s_kv), tc_kv.retile(frag))
            return [frag[None, 0, kk] for kk in range_constexpr(K_STEPS)]

        # Global->LDS DMA atoms: 16 bytes of KV / one f32 kv_scale per lane.
        kv_dma = fx.make_copy_atom(fx.rocdl.BufferCopyLDS128b(), 128)
        sc_dma = fx.make_copy_atom(fx.rocdl.BufferCopyLDS32b(), 32)

        def _tile_view(it, col0_i32, col_elems, elem_bytes, swizzle=None):
            """Flat buffer view of a per-KV-column array, starting at ``col0``.

            The column offset goes into the descriptor base (a scalar i64
            address add, not a per-lane voffset), so every lane's offset
            within the tile is loop-invariant. ``num_records`` ends at
            ``seq_len_kv``: columns past it read 0, and are masked out of the
            stored logits by the per-row window predicate.
            """
            rem_cols = fx.max(seq_len_kv - col0_i32, fx.Int32(0))
            it = fx.add_offset(it, fx.Int64(col0_i32) * fx.Int64(col_elems))
            layout = fx.make_layout(TILE_COLS * col_elems, 1)
            if const_expr(swizzle is not None):
                layout = fx.make_composed_layout(swizzle, layout)
            view = fx.make_view(it, layout)
            return fx.rocdl.make_buffer_tensor(
                fx.Tensor(view),
                num_records_bytes=fx.Int64(rem_cols * fx.Int32(col_elems * elem_bytes)),
            )

        def _kv_tile(col0_i32):
            """KV tile through the LDS swizzle: element ``p`` is the logical
            byte the swizzled slot stores at physical byte ``p``."""
            return _tile_view(fx.get_iter(KV), col0_i32, D, 1, swizzle=swz)

        def _scale_tile(col0_i32):
            return _tile_view(fx.get_iter(kv_scales), col0_i32, 1, 4)

        def _dma_kv_tile_to_lds(slot_dword, col0_i32):
            """Cooperatively async-copy KV[col0:col0+BKV, :] into LDS slot.

            All MR_BLOCK_THREADS threads participate; thread ``tid`` at load ``i``
            writes slot byte ``(i*MR_BLOCK_THREADS + tid)*DMA_BYTES``, reading
            the logical bytes the swizzle maps there.
            """
            lds_dst = fx.add_offset(
                lds_kv, slot_dword * fx.Int32(4) + wave_s * fx.Int32(64 * DMA_BYTES)
            )
            kv_src = fx.logical_divide(_kv_tile(col0_i32), fx.make_layout(1, 1))
            for i in range_constexpr(NUM_KV_DMAS):
                if const_expr(i > 0):
                    lds_dst = fx.add_offset(lds_dst, MR_BLOCK_THREADS * DMA_BYTES)
                phys_byte = (tid + fx.Int32(i * MR_BLOCK_THREADS)) * fx.Int32(DMA_BYTES)
                fx.copy(
                    kv_dma,
                    fx.slice(kv_src, (None, phys_byte)),
                    fx.make_view(lds_dst, fx.make_layout(1, 1)),
                )

            # kv_scales[col0 + s*64 + lane] -> slot scale dword s*64 + lane.
            # Same wave-uniform destination in every wave (see SCALE_DMAS).
            if const_expr(SCALE_DMAS > 0):
                sc_src = fx.logical_divide(_scale_tile(col0_i32), fx.make_layout(1, 1))
            for s in range_constexpr(SCALE_DMAS):
                sc_dst = fx.add_offset(
                    lds_f32, slot_dword + fx.Int32(SCALE_DW + s * 64)
                )
                fx.copy(
                    sc_dma,
                    fx.slice(sc_src, (None, fx.Int32(s * 64) + lane)),
                    fx.make_view(sc_dst, fx.make_layout(1, 1)),
                )

        def _lds_scales(slot_dword):
            """The staged tile's kv_scales, indexed by tile-relative column."""
            return fx.Tensor(
                fx.make_view(
                    fx.add_offset(lds_f32, slot_dword + fx.Int32(SCALE_DW)),
                    fx.make_layout(SCALE_DMAS * 64, 1),
                )
            )

        # ---- Preload this wave's RPW rows: window, Q A-frags, weights ----
        starts = [None] * RPW
        ends = [None] * RPW
        a_packs = [None] * RPW
        w_frag = [None] * RPW
        seq_len_m_1 = seq_len - fx.Int32(1)

        # Loop over the rows owned by this wave.
        for j in range_constexpr(RPW):
            row = wave_row0 + fx.Int32(j)
            row_ld = fx.min(row, seq_len_m_1)
            in_rows = row < seq_len
            ss = fx.max(fx.Int32(cs_t[row_ld]), fx.Int32(0))
            ee = fx.min(fx.Int32(ce_t[row_ld]), seq_len_kv)
            starts[j] = fx.Int32(
                rocdl.readfirstlane(
                    fx.Int32.ir_type, in_rows.select(ss, seq_len_kv).ir_value()
                )
            )
            ends[j] = fx.Int32(
                rocdl.readfirstlane(
                    fx.Int32.ir_type, in_rows.select(ee, fx.Int32(0)).ir_value()
                )
            )

            # A-fragments: the row's [H, D] Q tile, partitioned by the tiled MMA
            # into (V, M_TILES, K_STEPS).
            q_row = fx.slice(q_buf, (row_ld, None, None))
            frag_a = thr_mma.make_fragment_A(q_row)
            fx.copy(q_copy, tc_q.partition_S(q_row), tc_q.retile(frag_a))
            a_packs[j] = [
                [frag_a[None, mi, kk] for kk in range_constexpr(K_STEPS)]
                for mi in range_constexpr(M_TILES)
            ]

            w_frag[j] = _load_row_weights(
                weights, H, cp_4xfp32, tc_c_w, M_TILES, mfma.MFMA_N, row_ld
            )

        # ---- Union KV window across all block rows (all waves cooperate) ----
        u_start = None
        u_end = None
        # Compute the union KV window [u_start, u_end) across all rows in this block.
        # All ROWS_PER_BLOCK query rows share a single KV tile scan over this union
        # interval, so each KV tile is loaded once and reused by every row.
        #   u_start = min(cu_starts[rows])
        #   u_end = max(cu_ends[rows]).
        for jj in range_constexpr(ROWS_PER_BLOCK):
            rr = block_row0 + fx.Int32(jj)
            rr_ld = fx.min(rr, seq_len_m_1)
            in_rows = rr < seq_len
            ss = fx.max(fx.Int32(cs_t[rr_ld]), fx.Int32(0))
            ee = fx.min(fx.Int32(ce_t[rr_ld]), seq_len_kv)
            # Empty [seq_len_kv, 0) for a past-the-end slot leaves min/max
            # untouched.
            ss = in_rows.select(ss, seq_len_kv)
            ee = in_rows.select(ee, fx.Int32(0))
            if jj == 0:
                u_start = ss
                u_end = ee
            else:
                u_start = fx.min(u_start, ss)
                u_end = fx.max(u_end, ee)
        tile_start = (u_start // fx.Int32(BKV)) * fx.Int32(BKV)
        # Collapse an empty union window to zero width.
        tile_end = fx.max(u_end, tile_start)

        # KV-column split across grid.y
        # Each (grid.x, grid.y) block owns a disjoint vertical slice of the output logits [seq_len_q, seq_len_kv]:
        # grid.x cuts query rows (horizontal), grid.y cuts KV positions (vertical).
        # The relevant KV slice to be loaded is KV[tile_start:tile_end, :].
        block_y = fx.block_idx.y

        # How many tiles of BKV columns are in the union window.
        win_tiles = fx.Int32(ceildiv(fx.Uint32(tile_end - tile_start), fx.Uint32(BKV)))

        # How many KV columns (bytes/positions, rounded up to full tiles) each grid.y split owns.
        split_cols = fx.Int32(
            ceildiv(fx.Uint32(win_tiles), fx.Uint32(num_splits))
        ) * fx.Int32(BKV)

        # Each grid.y block (block_y) shifts its start forward by block_y * split_cols:
        tile_start = tile_start + block_y * split_cols
        tile_end = fx.min(tile_start + split_cols, tile_end)

        n_tiles = fx.Int32(
            ceildiv(
                fx.Uint32(fx.max(tile_end - tile_start, fx.Int32(0))), fx.Uint32(BKV)
            )
        )

        # ---- Prologue: prefetch tiles 0..PREFETCH_DEPTH-1 into buffers ----
        for _p in range_constexpr(PREFETCH_DEPTH):
            _dma_kv_tile_to_lds(
                fx.Int32((_p % NUM_BUFFERS) * SLOT_I32),
                tile_start + fx.Int32(_p * BKV),
            )

        def _store_if(pred, out_row_t, col, value):
            """``out_row_t[col] = value`` where ``pred`` holds.

            Via a closure, not a bare ``out_row_t[col] = ...`` in the branch:
            the rewriter reads a subscript store as an assignment to
            ``out_row_t`` and tries to carry the TensorView out of the scf.if.
            """

            def _store():
                out_row_t[col] = value

            if pred:
                _store()

        ep_ctx = _EpilogueCtx(
            mfma=mfma,
            mma=mma,
            gemm_kw=gemm_kw,
            a_packs=a_packs,
            w_frag=w_frag,
            starts=starts,
            ends=ends,
            lane=lane,
            lane_div_N=lane_div_N,
            f32_0=f32_0,
            store_if=_store_if,
            iglp=IGLP_SW_PIPE,
        )

        # ---- Steady-state software pipeline over BKV tiles ----
        # slot_static: compile-time slot (unroll_slots) or None (t % NB).
        def _tile(t, slot_static):
            col0 = tile_start + t * fx.Int32(BKV)
            if const_expr(slot_static is None):
                slot_idx = t % fx.Int32(NUM_BUFFERS)
                slot_dword = slot_idx * fx.Int32(SLOT_I32)
            else:
                slot_dword = fx.Int32(slot_static * SLOT_I32)

            # Wait until only the (PREFETCH_DEPTH-1) newer tiles remain in flight,
            # i.e. the current tile is complete; then sync so every wave sees the
            # full LDS tile.
            rocdl.s_waitcnt(vmcnt=_WAIT_VMCNT)
            gpu.barrier()

            # Read all B-frags for this tile from LDS into registers. Hoisting
            # every (ni,kk) read ahead of the compute nest lets the compiler
            # batch the LDS loads and hide their lgkmcnt latency behind the MFMA work.
            b_packs = [None] * N_TILES
            cols = [None] * N_TILES
            # kv_scales_tile[ni]: this lane's column scale, applied before the
            # head reduce. lds_scales + rs_head instead scales once after the
            # reduce-scatter.
            kv_scales_tile = [None] * N_TILES
            rs_scales = [None] * (N_TILES // RS_GROUP)
            if const_expr(not lds_scales):
                sc_tile = _scale_tile(col0)
            else:
                sc_lds = _lds_scales(slot_dword)
            for ni in range_constexpr(N_TILES):
                col = col0 + fx.Int32(ni * mfma.MFMA_N) + lane_mod_N
                cols[ni] = col
                if const_expr(not lds_scales):
                    # Column-relative offset into the tile's descriptor; past
                    # seq_len_kv the hardware returns 0 (masked column).
                    kv_scales_tile[ni] = fx.Float32(
                        sc_tile[fx.Int32(ni * mfma.MFMA_N) + lane_mod_N]
                    )
                col_local = fx.Int32(ni * mfma.MFMA_N) + lane_mod_N
                if const_expr(lds_scales and not rs_head):
                    kv_scales_tile[ni] = fx.Float32(sc_lds[col_local])
                b_packs[ni] = _read_b_frags(slot_dword * fx.Int32(4), ni)
            if const_expr(lds_scales and rs_head):
                for g in range_constexpr(N_TILES // RS_GROUP):
                    rs_scales[g] = fx.Float32(sc_lds[fx.Int32(g * 64) + lane])

            # Prefetch tile i+PREFETCH_DEPTH into slot (i+PD)%NB.  When NB>PD that
            # slot != the just-read slot, so no reader-before-writer barrier is
            # needed (the slot's last reader was iteration i-(NB-PD), already
            # past this iteration's barrier).  NB==PD reuses the read slot and
            # requires barrier B first.
            if const_expr(_need_barrier_b):
                gpu.barrier()
            t_next = t + fx.Int32(PREFETCH_DEPTH)
            if const_expr(slot_static is None):
                next_slot_dword = (t_next % fx.Int32(NUM_BUFFERS)) * fx.Int32(SLOT_I32)
            else:
                next_slot_dword = fx.Int32(
                    ((slot_static + PREFETCH_DEPTH) % NUM_BUFFERS) * SLOT_I32
                )
            col0_next = tile_start + t_next * fx.Int32(BKV)
            _dma_kv_tile_to_lds(next_slot_dword, col0_next)

            # ---- Per-row MFMA + epilogue (this wave's RPW rows, all columns) ----
            out_row_ts = [
                _make_out_row_t(logits, _stride_i64, wave_row0 + fx.Int32(j))
                for j in range_constexpr(RPW)
            ]

            _mfma_and_epilogue(
                ep_ctx,
                _TileOperands(
                    col0=col0,
                    cols=cols,
                    b_packs=b_packs,
                    kv_scales=kv_scales_tile,
                    rs_scales=rs_scales,
                    out_row_ts=out_row_ts,
                ),
            )

        # Run the loop over KV tiles either unrolled or rolled fashion.
        if const_expr(UNROLL_KV_TILE_LOOP):
            # Unrolled by NUM_BUFFERS so every slot offset is a constant (tile
            # t lives in slot t % NUM_BUFFERS).
            for t0_iv in range(fx.Int32(0), n_tiles, fx.Int32(NUM_BUFFERS)):
                for u in range_constexpr(NUM_BUFFERS):
                    t_u = fx.Int32(t0_iv) + fx.Int32(u)
                    if t_u < n_tiles:
                        _tile(t_u, u)
        else:
            for t_iv in range(fx.Int32(0), n_tiles, fx.Int32(1)):
                _tile(fx.Int32(t_iv), None)

        # ---- Fused clean_logits prefill: per-wave, over this wave's own rows.
        # A wave holds starts[]/ends[] only for its RPW rows; making all waves
        # cooperate would need extra cu_starts/cu_ends loads for no gain.
        # Emitting this after the tile loop is mandatory -- the loop's
        # s_waitcnt(vmcnt=_WAIT_VMCNT) counts vector stores too on gfx9, so a
        # fill store in flight inside it would let a half-written LDS tile
        # through. ----
        if const_expr(clean_logits):
            neg_inf = fx.Float32(float("-inf"))

            def _store_neg_inf(t, c):
                t[c] = neg_inf

            def _fill_range(out_row_t, lo_i32, hi_i32):
                """Thread-strided -inf fill over ``[lo, hi)``, one wave wide.

                Only the 64 lanes of THIS wave cooperate (the wave owns these
                rows), so the stride is 64 rather than the block width. See the
                direct-load builder for why plain dwords are used.
                """
                for c in range(lo_i32 + lane, hi_i32, fx.Int32(64)):
                    _store_neg_inf(out_row_t, fx.Int32(c))

            _emit_row_neg_inf_fill(
                logits=logits,
                stride_i64=_stride_i64,
                rows=[wave_row0 + fx.Int32(j) for j in range_constexpr(RPW)],
                starts=starts,
                ends=ends,
                seq_len_kv=seq_len_kv,
                seq_len=seq_len,
                by_i32=block_y,
                num_splits=num_splits,
                fill_range=_fill_range,
            )

    @flyc.jit
    def launch_fp8_mqa_logits_mfma_lds_pipe(
        Q: fx.Tensor,
        KV: fx.Tensor,
        kv_scales: fx.Tensor,
        weights: fx.Tensor,
        cu_starts: fx.Tensor,
        cu_ends: fx.Tensor,
        logits: fx.Tensor,
        seq_len: fx.Int32,
        seq_len_kv: fx.Int32,
        stride_logits_s: fx.Int32,
        num_splits: fx.Int32,
        stream: fx.Stream,
    ):
        gx = fx.Int64(fx.Int32(ceildiv(fx.Uint32(seq_len), fx.Uint32(ROWS_PER_BLOCK))))
        gy = fx.Int64(num_splits)
        kernel._func.__name__ = _kname
        kernel(
            Q,
            KV,
            kv_scales,
            weights,
            cu_starts,
            cu_ends,
            logits,
            seq_len,
            seq_len_kv,
            stride_logits_s,
            num_splits,
        ).launch(grid=(gx, gy, 1), block=(MR_BLOCK_THREADS, 1, 1), stream=stream)

    return launch_fp8_mqa_logits_mfma_lds_pipe
