# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Attention-residual (AttnRes) epilogue of the one-shot all-reduce.

Kimi-K3 replaces the plain residual add with an aggregation over a bank of
earlier block states. Per token row, after the all-reduce:

    prefix = bf16(prefix_in + bf16(sum_r input_r))     -> prefix_out
    score_j = dot(v_j, cw) / sqrt(mean(v_j^2) + score_eps)
              for v_j in bank[t, 0..nvb-1] and prefix
    w = softmax(score)                                  (global max first)
    mix = prefix * w_p + sum_j w_j * bank[t, j]
    out = bf16(mix * rsqrt(mean(mix^2) + out_eps) * ow)  (or bf16(mix))

and optionally ``bank[t, nvb] = prefix`` (the snapshot). This is SGLang's
``attn_res_hip._agg_kernel`` with the pending residual taken straight from the
all-reduce's registers, so the reduced row is never written to HBM and read
back by a second launch.

It reuses the RMSNorm epilogue's geometry: the one-shot's tile is pinned to one
token row, one workgroup per row (``split == 1``; a split build would have to
join ``2*(nvb+1)+1`` row sums across workgroups and is not built). ``nvb`` is a
build constant -- the bank rows are unrolled -- so each bank depth is its own
binary.

``cw`` is fp32 (SGLang precomputes ``score_norm.weight * score_proj.weight`` in
fp32); ``ow`` is the bf16 output-norm weight. The bank is ``(T, NB, hidden)``
bf16, rows ``NB`` apart; it is read and written outside the payload's buffer
window, through its own descriptor.

**SILOTIGER-1277 split (``split > 1``, acc_cache only).** The split build spreads
each token row over ``split`` workgroups that join the prefix dot (signed
exchange) + prefix sum of squares before softmax, and the out-norm mix sum of
squares after, through HBM exchange words -- the ~2x decode lever (see
reviews/SPLIT_ATTNRES_EPILOGUE_DESIGN.md). It is **72/72 at <=1 ulp** at TP8
(scratch/test_attnres_split3.py) and wins ~1.6-1.7x vs production (cdr_2stage +
agg) across decode: fused epilogue ~13us -> ~8.5us at T=1, and the fuse band now
extends past T16. Three subtleties the build had to get right (iter003):
  - ``row_stride_i32`` passed to the io tile addressing is the SLICE stride
    (``hidden//split//2``), NOT the full ``hidden//2`` -- a full value makes every
    workgroup but bid 0 address the wrong row. The bank rows are full ``hidden``
    wide, so they use ``bank_row_stride_i32`` instead.
  - the signed exchange recenters a small dot by a large bias; both the encode
    (``xchg_contribution_signed``) and decode (``xchg_total_signed``) must apply
    that bias in the INTEGER domain -- doing it in fp32 rounds the dot away.
  - the two join-point-1 words (dot, sq) are joined in ONE device function
    (``_xchg_join_dot_sq``): two back-to-back separate ``@flyc.jit`` spin-joins
    miscompile (the second reads all-arrivals-no-sum).
``split == 1`` compiles none of the exchange and is unchanged.
"""

import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from .one_shot_allreduce_rmsnorm import (
    _LdsWindow,
    _xchg_join,
    fused_block,
    fused_block_options,
    fused_split_options,
)
from .quick_allreduce_fusions import (
    ATOM_ELEMS,
    ATOM_I32,
    XCHG_LINE_BYTES,
    block_reduce_add,
    make_wave_partials,
    pack_bf16,
    residual_add,
    resolve_h_pad,
    scale_by_weight,
    wave_reduce_add,
    xchg_arrivals,
    xchg_clamp_units,
    xchg_contribution,
    xchg_contribution_signed,
    xchg_total,
    xchg_total_signed,
)
from .transport import (
    WAVE,
    _buffer_load,
    _buffer_ptr,
    _buffer_store,
    _global_ptr,
    _to_sgpr_i64,
    atom_bf16_to_f32,
)


@flyc.jit
def _xchg_join_dot_sq(dot_word, sq_word, dot_part, sq_part, dot_prev, sq_prev, n_writers, tid):
    """Join point 1 of the split AttnRes epilogue: the signed prefix dot and the
    non-negative prefix sum of squares, each through its own HBM exchange word,
    in ONE device function. Doing the two joins as two *separate* ``@flyc.jit``
    spin-join calls back to back miscompiles -- the second join reads its word as
    all-arrivals-no-sum (returns 0) -- so both atomics are issued, then both
    words are polled, within a single function body (SILOTIGER-1277 iter003)."""
    if (tid & fx.Int32(WAVE - 1)) == fx.Int32(0):
        fx.atomic_add(
            dot_word,
            xchg_contribution_signed(dot_part, n_writers),
            syncscope=rocdl.SyncScope.AgentOneAs,
        )
        fx.atomic_add(
            sq_word,
            xchg_contribution(sq_part, n_writers),
            syncscope=rocdl.SyncScope.AgentOneAs,
        )
    curd = fx.generic_load(
        dot_word, dtype=fx.Int64,
        memory_order=fx.AtomicOrdering.Monotonic, syncscope=rocdl.SyncScope.AgentOneAs,
    )
    while xchg_arrivals(curd - dot_prev) != fx.Int64(n_writers):
        curd = fx.generic_load(
            dot_word, dtype=fx.Int64,
            memory_order=fx.AtomicOrdering.Monotonic, syncscope=rocdl.SyncScope.AgentOneAs,
        )
    curs = fx.generic_load(
        sq_word, dtype=fx.Int64,
        memory_order=fx.AtomicOrdering.Monotonic, syncscope=rocdl.SyncScope.AgentOneAs,
    )
    while xchg_arrivals(curs - sq_prev) != fx.Int64(n_writers):
        curs = fx.generic_load(
            sq_word, dtype=fx.Int64,
            memory_order=fx.AtomicOrdering.Monotonic, syncscope=rocdl.SyncScope.AgentOneAs,
        )
    return (
        xchg_total_signed(curd - dot_prev, n_writers),
        xchg_total(curs - sq_prev),
        curd,
        curs,
    )


@flyc.jit
def _xchg_join_triple(
    dot_word, sq_word, cross_word,
    dot_part, sq_part, cross_part,
    dot_prev, sq_prev, cross_prev,
    n_writers, tid, poll_n=None,
):
    """Single-barrier join of the split AttnRes out-norm path (SILOTIGER-1277
    iter010): the prefix dot (signed), prefix sq, and cross=sum(pv*acc) (signed)
    -- the three joined words needed to form the output RMSNorm's sum of squares
    WITHOUT a second barrier.

    The acc_cache mix is ``mix = wp*pv + wb*acc`` with per-token scalars wp, wb,
    so ``sum(mix^2) = wp^2*sum(pv^2) + 2*wp*wb*sum(pv*acc) + wb^2*sum(acc^2)``.
    ``sum(acc^2)`` is a FULL-ROW scalar the producer precomputes (it holds acc
    and it is AR-independent), so only ``cross`` joins here on top of the prefix
    dot/sq -- the SAME three words the two-barrier path joined, but in ONE
    barrier (one fewer xGMI round-trip) with no extra atomic word, so it wins at
    every decode T. Like ``_xchg_join_dot_sq`` all atomics are issued first and
    all words polled after, in one function body, to dodge the back-to-back
    separate-join miscompile."""
    # SILOTIGER-1277 iter052 collapse-feasibility probe: ``poll_n`` is a
    # TRACE-TIME Python int (None -> n_writers = the shipped barrier). PERF ONLY
    # (the probe arms give the WRONG numerics -- never gated). Default unchanged.
    #   poll_n == 0  (env=1): keep the atomics, SKIP the cross-WG WAIT loops ->
    #     isolates the cross-WG barrier WAIT. fused(norm)-fused(poll0)==wait cost.
    #   poll_n == -1 (env=2): skip the atomics AND loads entirely, return each
    #     WG's own wave partial -> removes the whole HBM cross-WG join.
    #     fused(poll0)-fused(poll-1)==the HBM atomic-issue+readback cost.
    poll_n = n_writers if poll_n is None else poll_n
    if poll_n == -1:
        return (dot_part, sq_part, cross_part, dot_prev, sq_prev, cross_prev)
    if (tid & fx.Int32(WAVE - 1)) == fx.Int32(0):
        fx.atomic_add(
            dot_word,
            xchg_contribution_signed(dot_part, n_writers),
            syncscope=rocdl.SyncScope.AgentOneAs,
        )
        fx.atomic_add(
            sq_word,
            xchg_contribution(sq_part, n_writers),
            syncscope=rocdl.SyncScope.AgentOneAs,
        )
        fx.atomic_add(
            cross_word,
            xchg_contribution_signed(cross_part, n_writers),
            syncscope=rocdl.SyncScope.AgentOneAs,
        )
    curd = fx.generic_load(
        dot_word, dtype=fx.Int64,
        memory_order=fx.AtomicOrdering.Monotonic, syncscope=rocdl.SyncScope.AgentOneAs,
    )
    if poll_n != 0:
        while xchg_arrivals(curd - dot_prev) != fx.Int64(poll_n):
            curd = fx.generic_load(
                dot_word, dtype=fx.Int64,
                memory_order=fx.AtomicOrdering.Monotonic, syncscope=rocdl.SyncScope.AgentOneAs,
            )
    curs = fx.generic_load(
        sq_word, dtype=fx.Int64,
        memory_order=fx.AtomicOrdering.Monotonic, syncscope=rocdl.SyncScope.AgentOneAs,
    )
    if poll_n != 0:
        while xchg_arrivals(curs - sq_prev) != fx.Int64(poll_n):
            curs = fx.generic_load(
                sq_word, dtype=fx.Int64,
                memory_order=fx.AtomicOrdering.Monotonic, syncscope=rocdl.SyncScope.AgentOneAs,
            )
    curc = fx.generic_load(
        cross_word, dtype=fx.Int64,
        memory_order=fx.AtomicOrdering.Monotonic, syncscope=rocdl.SyncScope.AgentOneAs,
    )
    if poll_n != 0:
        while xchg_arrivals(curc - cross_prev) != fx.Int64(poll_n):
            curc = fx.generic_load(
                cross_word, dtype=fx.Int64,
                memory_order=fx.AtomicOrdering.Monotonic, syncscope=rocdl.SyncScope.AgentOneAs,
            )
    return (
        xchg_total_signed(curd - dot_prev, n_writers),
        xchg_total(curs - sq_prev),
        xchg_total_signed(curc - cross_prev, n_writers),
        curd, curs, curc,
    )


#: Deepest bank the epilogue builds: K3 keeps at most 8 earlier block states.
MAX_NVB = 8

#: Rows whose butterfly steps are emitted together by
#: :func:`_wave_reduce_add_xpose`. Tail at T=1, atoms=2, nvb=8, against the
#: one-row-at-a-time emission this replaces: that order 7.82 us, chunk 3 7.56,
#: chunk 18 7.51, chunk 9 7.28, **chunk 6 7.19**. Re-tuned later against the
#: finished kernel and 6 still won (3: 6.40, 4: 6.33, 6: 6.16, 9: 6.24,
#: 18: 6.17). Above 6 the extra live values start to spill -- chunk 18 spills
#: 8 VGPRs where chunk 6 spills none.
_XPOSE_CHUNK = 6

#: Widest lane group :func:`_block_reduce_many` will fold a row's per-wave
#: partials in, and so the most waves a block may have.
_SEG_MAX = 16


def _seg_width(n_waves: int) -> int:
    """Lanes per group in :func:`_block_reduce_many`'s transposed fold.

    Only ``n_waves`` partials exist per row, so a group wider than that folds
    zeros. The narrowest power of two that still holds them all is therefore
    both correct and cheapest: a wave covers ``WAVE // seg`` rows per pass, so
    halving the group halves the number of passes *and* drops a butterfly
    step. At atoms=2 (7 waves) this is 8 rather than 16, which takes the fold
    for ``2*(nvb+1) = 18`` rows from 5 passes x 4 steps to 3 x 3.

    The bits do not move. A narrower group drops exactly the slots that were
    adding ``0.0``, and the remaining tree has the same shape: with the steps
    taken in descending distance, the group of 16 computes
    ``((p0+p4)+(p2+p6)) + ((p1+p5)+(p3+p7))`` once ``p8..p15`` are zero, which
    is what the group of 8 computes outright.
    """
    if n_waves > _SEG_MAX:
        raise ValueError(
            f"_block_reduce_many takes at most {_SEG_MAX} waves, got {n_waves}"
        )
    seg = 1
    while seg < n_waves:
        seg *= 2
    return seg


def _wave_reduce_add_xpose(values, chunk):
    """A wave-wide sum per row, emitted distance-major in *chunk* rows.

    Bit for bit the same tree as ``quick_allreduce_fusions.wave_reduce_add``,
    which this replaces for the epilogue's many rows: each row still folds
    XOR 32, 16, 8, 4, 2, 1 in that order, and only the *emission* order
    changes. Per row, ``flydsl`` lowers the four short distances to
    ``ds_swizzle_b32``, and because the dependent add follows immediately the
    compiler puts a full ``s_waitcnt lgkmcnt(0)`` after every one of them.
    Taking one distance across *chunk* rows at a time puts that many swizzles
    in flight under a single wait. *chunk* bounds the extra live values.
    """
    out = []
    for base in range_constexpr(0, len(values), chunk):
        accs = list(values[base : base + chunk])
        for sh in range_constexpr(int(math.log2(WAVE))):
            d = WAVE // (2 << sh)
            moved = [a.shuffle_xor(d, WAVE) for a in accs]
            accs = [a + m for a, m in zip(accs, moved)]
        out.extend(accs)
    return out


def _block_reduce_many(values, *, tid, block: int, lds):
    """Block-wide sums of *values*, one per row, broadcast to every thread.

    ``quick_allreduce_fusions.block_reduce_add`` has every thread read back all
    ``rows * n_waves`` partials and add them itself, which is the cheapest
    choice for one or two rows. With the AttnRes epilogue's ``2*(nvb+1)`` rows
    that is 126 dependent LDS reads per thread at nvb=8, 7 waves (252 at 14),
    and it measured ~3 us (~5.7 us at 14 waves) at one token. Here the read is
    transposed: after the same partial stores and barrier, lane
    ``seg_width*k + w`` of every wave reads wave ``w``'s partial of one row
    (``WAVE // seg_width`` rows per pass), the groups fold them with a
    ``log2(seg_width)``-step butterfly, and ``readlane`` hands each row's
    total to the whole wave as a uniform value. Every wave computes the same
    totals in the same order, so all threads -- and all ranks -- get identical
    bits. See :func:`_seg_width` for why the group is only as wide as the
    wave count.

    *lds* holds ``rows * n_waves`` floats, laid out ``[row][wave]`` exactly as
    ``block_reduce_add`` writes them, so the store side is unchanged.
    """
    n_rows = len(values)
    n_waves = block // WAVE
    seg_width = _seg_width(n_waves)
    locals_ = _wave_reduce_add_xpose(values, _XPOSE_CHUNK)
    if const_expr(n_waves == 1):
        return locals_
    wid = tid // fx.Int32(WAVE)
    for row in range_constexpr(n_rows):
        lds[fx.Int32(row * n_waves) + wid] = locals_[row]
    gpu.barrier()

    lane = tid % fx.Int32(WAVE)
    seg = lane // fx.Int32(seg_width)
    w = lane % fx.Int32(seg_width)
    rows_per_pass = WAVE // seg_width
    n_steps = int(math.log2(seg_width))
    n_pass = (n_rows + rows_per_pass - 1) // rows_per_pass

    def _pass_value(p):
        row = fx.Int32(p * rows_per_pass) + seg
        live = (w < fx.Int32(n_waves)).select(row < fx.Int32(n_rows), w < fx.Int32(0))
        idx = live.select(row * fx.Int32(n_waves) + w, fx.Int32(0))
        return live.select(fx.Float32(lds[idx]), fx.Float32(0.0))

    def _pick(vals):
        totals = []
        for p in range_constexpr(n_pass):
            for k in range_constexpr(rows_per_pass):
                if const_expr(p * rows_per_pass + k < n_rows):
                    totals.append(
                        fx.Float32(rocdl.readlane(T.f32, vals[p], k * seg_width))
                    )
        return totals

    # The same butterfly per pass, emitted distance-major across the passes,
    # for the reason `_wave_reduce_add_xpose` gives. The passes are
    # independent and each keeps its own step order, so the bits are those of
    # a pass-at-a-time loop.
    accs = [_pass_value(p) for p in range_constexpr(n_pass)]
    for sh in range_constexpr(n_steps):
        d = seg_width // (2 << sh)
        moved = [a.shuffle_xor(d, WAVE) for a in accs]
        accs = [a + m for a, m in zip(accs, moved)]
    return _pick(accs)


class OneShotAttnResEpilogue:
    """AttnRes aggregation on the one-shot's reduced tile (see the module doc).

    Built by the kernel factory; ``bind`` is called once inside the kernel body
    and returns the object the schedule drives (hooks as in the plain epilogue
    of ``one_shot_allreduce``).
    """

    fusion = "attn_res"
    padded = False

    def __init__(
        self,
        *,
        hidden: int,
        h_pad: int | None,
        split: int,
        atoms: int,
        block: int | None,
        grid: int,
        n_colors: int,
        nvb: int,
        write_bank: bool,
        out_norm: bool,
        acc_cache: bool = False,
        outnorm_merge: bool = False,
    ):
        hidden = int(hidden)
        if resolve_h_pad(hidden, h_pad) != hidden:
            raise ValueError(
                f"the AttnRes epilogue needs a native geometry; hidden={hidden} "
                f"would run padded to {h_pad}"
            )
        split = int(split)
        if split != 1 and not acc_cache:
            # The raw/scores builds would have to join 2*(nvb+1) bank row sums
            # across workgroups; only the acc-cache build's in-kernel reduction
            # is small enough (prefix dot + prefix sq + out-norm sq) to split.
            raise ValueError(
                f"the AttnRes epilogue splits only in the acc_cache build, "
                f"got split={split}"
            )
        if split > 1:
            if hidden % split:
                raise ValueError(f"split={split} does not divide hidden={hidden}")
            if grid % split:
                raise ValueError(f"grid={grid} is not a multiple of split={split}")
        nvb = int(nvb)
        if not 1 <= nvb <= MAX_NVB:
            raise ValueError(f"nvb must be in [1, {MAX_NVB}], got {nvb}")
        row_width = fused_block(hidden // split, atoms)
        if block is not None and int(block) != row_width:
            legal = (
                fused_block_options(hidden)
                if split == 1
                else [(b, a) for k, b, a in fused_split_options(hidden) if k == split]
            )
            raise ValueError(
                f"AttnRes hidden={hidden} (split={split}) at atoms={atoms} needs "
                f"block={row_width}, got block={block}; legal (block, atoms): {legal}"
            )
        self.hidden = hidden
        self.h_pad = hidden
        self.split = split
        self.atoms = int(atoms)
        self.block = row_width
        # Slice width (elements) each workgroup owns: the whole row at split=1.
        self.slice_width = hidden // split
        # i32 between one tile row and the next for the io tile addressing
        # (input/prefix/output). A split build views (M, hidden) as
        # (M*split, hidden/split), so this is the SLICE stride -- the factory
        # passes it to make_row_io; a full-width value makes every workgroup but
        # bid 0 address the wrong row (SILOTIGER-1277 iter003 bug). split=1 keeps
        # the full width.
        self.row_stride_i32 = hidden // split // 2
        # The bank is (T, NB, hidden): its rows are always FULL hidden wide,
        # regardless of split, so bank addressing uses its own stride.
        self.bank_row_stride_i32 = hidden // 2
        self.n_colors = n_colors
        self.n_waves = self.block // WAVE
        self.nvb = nvb
        self.write_bank = bool(write_bank)
        self.out_norm = bool(out_norm)
        self.acc_cache = bool(acc_cache)
        # SILOTIGER-1277 iter010: single-barrier out-norm. When set (and the
        # build is split + out_norm), the output RMSNorm's sum of squares is
        # formed from three extra row sums joined in the SAME barrier as the
        # prefix dot/sq (``sum(mix^2) = wp^2*sq + 2*wp*wb*cross + wb^2*accsq``,
        # mix = wp*pv + wb*acc), removing join point 2. Measured TP8/gfx950 H7168
        # nvb8 split=14: wins T<=4 (-0.4us/-4.7% @T1) but the two extra exchange
        # words add atomic contention at T>=8 (+1.7us @T16), so the ladder only
        # enables it at low T. An env override forces it on for gating. Off by
        # default -> every other build is byte-identical to the two-barrier path.
        env_merge = os.environ.get("AITER_FLY_AR_OUTNORM_MERGE") == "1"
        self.outnorm_merge = bool(self.out_norm) and split > 1 and (
            bool(outnorm_merge) or env_merge
        )
        # Row sums of the first reduction: dot(row, cw) and sum(row^2) for the
        # prefix row only. The nvb bank rows' scores are precomputed in the
        # caller (SILOTIGER-1277 bank-norm cache) and no longer block-reduced,
        # so only 2 words are joined across waves here (was 2*(nvb+1)). The
        # output norm's sum of squares gets its own slots after them, so its
        # writes never race a slow wave still reading the first reduction's
        # partials (there is no barrier between the two reads and the second
        # writes).
        self.n_score_sums = 2
        self.slots_per_color = self.n_waves * (self.n_score_sums + 1)
        if split > 1:
            # A split build joins each row sum across workgroups through HBM
            # exchange words (one per (stream, colour, group)) instead of LDS,
            # exactly like the RMSNorm split -- so there is no LDS at all.
            # Streams (3 when out_norm, else 2). Both paths join 3 words -- dot,
            # sq, and a 3rd -- the merge just saves a round-trip, no extra word:
            #   two-barrier (outnorm_merge off): 3rd = out-norm mix sq, joined in
            #     a SECOND barrier after the mix is formed.
            #   single-barrier (outnorm_merge on, iter010): 3rd = cross=sum(pv*acc),
            #     joined in the SAME barrier as dot/sq; the out-norm's sum(acc^2)
            #     term is the producer's full-row accsq scalar (no join).
            self.n_streams = 3 if self.out_norm else 2
            self.n_writers = split * self.n_waves
            # iter052 collapse-feasibility probe (PERF ONLY, default off): when
            # set, the cross-WG join wait loops are skipped at trace time so the
            # fused latency excludes the join barrier (see ``_xchg_join_triple``).
            _probe = os.environ.get("AITER_FLY_AR_COLLAPSE_PROBE")
            self.join_poll_n = (
                0 if _probe == "1" else -1 if _probe == "2" else self.n_writers
            )
            xchg_clamp_units(self.n_writers)
            self.n_groups = grid // split
            self.xchg_prev_off = (
                self.n_streams * n_colors * self.n_groups * XCHG_LINE_BYTES
            )
            self.xchg_bytes = (
                self.xchg_prev_off + grid * n_colors * self.n_streams * 8
            )
            self.lds_bytes = 0
            self._shared = None
        else:
            self.n_streams = 0
            self.n_writers = 0
            self.n_groups = grid
            self.xchg_prev_off = 0
            self.xchg_bytes = 0
            # One set per colour, for the same reason as the RMSNorm epilogue:
            # nothing but the colour rotation separates one tile's reads of the
            # partials from the next tile's writes.
            multi_wave = self.n_waves > 1
            self.lds_bytes = n_colors * self.slots_per_color * 4 if multi_wave else 0
            self._shared = (
                make_wave_partials(n_colors * self.slots_per_color)
                if multi_wave
                else None
            )

    @property
    def tag(self) -> str:
        tag = f"_attnres_h{self.hidden}_v{self.nvb}"
        if self.split > 1:
            tag += f"_k{self.split}"
        if self.write_bank:
            tag += "_wb"
        if not self.out_norm:
            tag += "_nonorm"
        if self.acc_cache:
            tag += "_acc"
        return tag

    def spec(self) -> dict:
        return {
            "fusion": self.fusion,
            "hidden": self.hidden,
            "h_pad": self.h_pad,
            "padded": False,
            "split": self.split,
            "xchg_bytes": self.xchg_bytes,
            "nvb": self.nvb,
            "write_bank": self.write_bank,
            "out_norm": self.out_norm,
            "acc_cache": self.acc_cache,
        }

    def bind(
        self,
        *,
        tid,
        bid,
        io,
        out_buf,
        res_in_ptr,
        res_out_ptr,
        w_ptr,
        eps,
        bank_ptr,
        cw_ptr,
        bank_nb,
        score_eps,
        score_ptr=None,
        ms_ptr=None,
        xchg_ptr=None,
        wrap=None,
        **_unused,
    ):
        """``res_in``/``res_out`` carry the prefix in and out, ``w``/``eps``
        the output norm; ``bank_nb`` is the bank's row count per token.
        ``score_ptr`` is the caller's precomputed (T, nvb) fp32 bank scores, or
        -- in the acc_cache build -- the (T, hidden) fp32 weighted bank sum;
        ``ms_ptr`` is then the (T, 3) fp32 (m, sum, accsq -- iter010).
        ``xchg_ptr``/``wrap`` carry
        the split build's HBM exchange state and colour rotation."""
        return _BoundOneShotAttnRes(
            self,
            tid=tid,
            bid=bid,
            io=io,
            out_buf=out_buf,
            res_in_ptr=res_in_ptr,
            res_out_ptr=res_out_ptr,
            w_ptr=w_ptr,
            eps=eps,
            bank_ptr=bank_ptr,
            cw_ptr=cw_ptr,
            bank_nb=bank_nb,
            score_eps=score_eps,
            score_ptr=score_ptr,
            ms_ptr=ms_ptr,
            xchg_ptr=xchg_ptr,
            wrap=wrap,
        )


class _BoundOneShotAttnRes:
    def __init__(
        self,
        epi,
        *,
        tid,
        bid,
        io,
        out_buf,
        res_in_ptr,
        res_out_ptr,
        w_ptr,
        eps,
        bank_ptr,
        cw_ptr,
        bank_nb,
        score_eps,
        score_ptr=None,
        ms_ptr=None,
        xchg_ptr=None,
        wrap=None,
    ):
        self.epi = epi
        self.tid = tid
        self.bid = bid
        self.wrap = wrap
        self.xchg_ptr = xchg_ptr
        self.io = io
        self.out_buf = out_buf
        self.eps = eps
        self.score_eps = score_eps
        self.bank_nb = bank_nb
        self.prefix_in_buf = io.operand(res_in_ptr)
        self.prefix_out_buf = io.operand(res_out_ptr)
        self.bank_buf = _buffer_ptr(_to_sgpr_i64(bank_ptr), T.i32, 16)
        # Split build: ``split`` consecutive workgroups share one token row;
        # ``group`` names the row group, ``part`` this workgroup's slice. The
        # schedule's tile index is then the sliced-row index, so the token is
        # ``tile // split`` and this slice covers columns [part*slice_width, ..).
        if const_expr(epi.split > 1):
            self.group = bid // fx.Int32(epi.split)
            self.part = bid % fx.Int32(epi.split)
            self.slice_col_f32 = self.part * fx.Int32(epi.slice_width)
            self.slice_col_i32 = self.part * fx.Int32(epi.slice_width // 2)
        else:
            self.group = bid
            self.part = fx.Int32(0)
            self.slice_col_f32 = fx.Int32(0)
            self.slice_col_i32 = fx.Int32(0)
        # score_ptr: scores-only build -> (T, nvb) fp32 scores at tile*nvb;
        # acc_cache build -> (T, hidden) fp32 weighted bank sum (read by column
        # like a bank row). ms_ptr (acc_cache only): (T, 3) fp32 (m, sum, accsq).
        self.score_buf = _buffer_ptr(_to_sgpr_i64(score_ptr), T.f32, 4)
        self.ms_buf = _buffer_ptr(_to_sgpr_i64(ms_ptr), T.f32, 4)
        # Per-thread fp32 column offsets into an (T, hidden) fp32 row (acc).
        self.acc_col_f32 = [
            (fx.Int32(a * epi.block) + tid) * fx.Int32(ATOM_ELEMS)
            for a in range_constexpr(epi.atoms)
        ]

        # This thread's atom columns, as i32 offsets within one bf16 row.
        self.col_i32 = [
            (fx.Int32(a * epi.block) + tid) * fx.Int32(ATOM_I32)
            for a in range_constexpr(epi.atoms)
        ]

        # cw is one fp32 (hidden,) row shared by every token: read once. An
        # atom's 8 values are 32 B of fp32, two 16 B loads.
        cw_buf = _buffer_ptr(
            _to_sgpr_i64(cw_ptr), T.f32, 16, num_records_bytes=fx.Int64(epi.hidden * 4)
        )
        self.cw = []
        for a in range_constexpr(epi.atoms):
            base = (fx.Int32(a * epi.block) + tid) * fx.Int32(ATOM_ELEMS)
            if const_expr(epi.split > 1):
                # cw is a full (hidden,) row shared by every token; this slice
                # reads its own columns.
                base = base + self.slice_col_f32
            lo = _buffer_load(cw_buf, base, 4, fx.Float32)
            hi = _buffer_load(cw_buf, base + fx.Int32(4), 4, fx.Float32)
            self.cw.append(
                fx.Vector.from_elements(
                    [lo[e] for e in range_constexpr(4)]
                    + [hi[e] for e in range_constexpr(4)],
                    fx.Float32,
                )
            )

        self.ow_atoms = None
        if const_expr(epi.out_norm):
            # The gain is one (hidden,) row shared by every token; a split build
            # reads its own slice (row ``part`` of the (split, hidden/split)
            # view), the whole row otherwise.
            w_buf = io.operand(w_ptr, records=fx.Int64(epi.hidden * 2))
            self.ow_atoms = io.load_tile(w_buf, self.part)

        self.lds = None
        if const_expr(epi.split == 1 and epi.n_waves > 1):
            # Allocated once, at kernel scope (SharedAllocator is static). A
            # split build joins through HBM and has no LDS.
            self.lds = fx.SharedAllocator().allocate(epi._shared).peek().wave.ptr

    def _bank_row_i32(self, tile, j):
        """i32 offset of bank row ``(tile, j)``: rows ``bank_nb`` apart per token.
        Bank rows are full ``hidden`` wide (not sliced), so this uses the bank's
        own full-width stride, not the sliced io tile stride."""
        row = tile * self.bank_nb + fx.Int32(j)
        return row * fx.Int32(self.epi.bank_row_stride_i32)

    def _bank_rows(self, tile):
        """This thread's atoms of bank rows ``0..nvb-1`` of *tile*."""
        return [
            [
                _buffer_load(
                    self.bank_buf,
                    self._bank_row_i32(tile, j) + self.col_i32[a],
                    ATOM_I32,
                    fx.Int32,
                )
                for a in range_constexpr(self.epi.atoms)
            ]
            for j in range_constexpr(self.epi.nvb)
        ]

    def prefetch(self, tile):
        """The prefix row of *tile*; it does not depend on any peer, so it is
        issued before the push and retires during the poll.

        The bank rows are *not* prefetched here. Measured TP2 at T=1, atoms=2,
        nvb=8: issuing them before the push instead of in :meth:`finish` is
        worth 0.11 us of 12.3, because the poll's flag probe drains the whole
        VM counter (``s_waitcnt vmcnt(0)``) and so cannot retire until the bank
        loads have landed anyway. Loading them in ``finish`` instead keeps them
        live over a much shorter range, which is what lets the mix reuse them.

        iter052: tried prefetching the acc_cache weighted-bank-sum (the epilogue's
        one big AR-independent HBM read) here to hide it in the poll -- NO win
        (~noise +-1%, runs/iter052_acc_prefetch_tp8.json): the poll's vmcnt(0)
        drain already lands it, so it is not exposed. Same conclusion as the nvb
        bank rows above. Not prefetched.
        """
        return self.io.load_tile(self.prefix_in_buf, tile)

    def _xchg_word(self, stream, color):
        """The active exchange word for row group ``self.group`` of this colour
        and *stream* (0 dot, 1 prefix sq, and -- iter010 out-norm single-barrier
        path -- 2 cross=sum(pv*acc), 3 accsq=sum(acc^2))."""
        lane = (
            fx.Int32(stream * self.epi.n_colors) + color
        ) * fx.Int32(self.epi.n_groups) + self.group
        return _global_ptr(
            self.xchg_ptr + fx.Int64(lane) * fx.Int64(XCHG_LINE_BYTES), T.i64, 8
        )

    def _prev_word(self, slot, stream):
        """This workgroup's persisted word for colour *slot* and *stream*."""
        addr = self.prev_base + fx.Int64(
            slot * fx.Int32(self.epi.n_streams) + fx.Int32(stream)
        ) * fx.Int64(8)
        return _global_ptr(addr, T.i64, 8)

    def init_state(self, color):
        """Split build: the exchange word for each stream as each of the next
        three colours last completed it (one per stream x colour), persisted per
        workgroup across launches -- the two-join analogue of the RMSNorm
        epilogue's single-stream triple. Empty otherwise."""
        epi = self.epi
        if const_expr(epi.split == 1):
            return ()
        self.prev_base = (
            self.xchg_ptr
            + fx.Int64(epi.xchg_prev_off)
            + fx.Int64(self.bid) * fx.Int64(epi.n_colors * epi.n_streams * 8)
        )
        slots = [
            color,
            self.wrap(color + fx.Int32(1)),
            self.wrap(color + fx.Int32(2)),
        ]
        state = []
        for stream in range_constexpr(epi.n_streams):
            for s in range_constexpr(3):
                state.append(
                    fx.generic_load(self._prev_word(slots[s], stream), dtype=fx.Int64)
                )
        return tuple(state)

    def _row_dot_sq(self, row):
        """This thread's ``dot(row, cw)`` and ``sum(row^2)`` over its atoms.

        Both sums are reassociated. Without ``fastmath`` ``vector.reduction``
        is an *ordered* fp32 reduction and the ISA shows what that costs: four
        ``v_pk_mul_f32`` for the eight products and then eight **scalar**
        ``v_add_f32``. Reassociated, LLVM folds them as a tree of
        ``v_pk_add_f32`` -- over the whole kernel at atoms=2, nvb=8,
        ``v_add_f32`` drops 700 -> 268 while ``v_pk_add_f32`` rises 288 ->
        504, i.e. 216 fewer VALU instructions, worth 0.31 us of the tail at
        T=1. It changes which bits come out (a tree, not a chain), so it is
        gated: all three ladders stay at 72/72 and 1.0 ulp vs Triton.

        **Contracting this tree into a packed-FMA chain is deliberately not
        done**, although it is faster. Writing the sum as ``a * b + acc`` over
        the four packed pairs gives 1 ``v_pk_mul_f32`` + 3 ``v_pk_fma_f32``
        instead of 7 instructions, and measured a further 0.23 us. But it is
        also the one change tried tonight that makes this kernel *less*
        accurate than the Triton kernel it replaces: on the op_test's 60 cases
        at this geometry it takes the worst distance from the fp32 reference
        from 1.099 to 1.465 ulp, where Triton itself stays within 1.0. 2.4% of
        the kernel does not buy that, so the tree stays.
        """
        dot = None
        sq = None
        for a in range_constexpr(len(row)):
            d = fx.Float32(
                (row[a] * self.cw[a]).reduce(ReductionOp.ADD, fastmath="reassoc")
            )
            s = fx.Float32(
                (row[a] * row[a]).reduce(ReductionOp.ADD, fastmath="reassoc")
            )
            dot = d if dot is None else dot + d
            sq = s if sq is None else sq + s
        return dot, sq

    def _row_cross(self, pv, acc):
        """This thread's ``sum(pv*acc)`` (signed) over its atoms, reassociated as
        the dot/sq tree is (SILOTIGER-1277 iter010). It joins in the SAME barrier
        as dot/sq so the split out-norm needs no second barrier:
        ``sum(mix^2) = wp^2*sq + 2*wp*wb*cross + wb^2*accsq`` where accsq is the
        producer-supplied full-row ``sum(acc^2)`` scalar (no join)."""
        cross = None
        for a in range_constexpr(len(pv)):
            c = fx.Float32(
                (pv[a] * acc[a]).reduce(ReductionOp.ADD, fastmath="reassoc")
            )
            cross = c if cross is None else cross + c
        return cross

    def _acc_row(self, token):
        """This thread's atoms of the cached (T, hidden) fp32 weighted bank sum
        (acc), 8 fp32 per atom in two 16 B loads (score_buf aliases acc). A
        split build reads this token's slice at ``part*slice_width``."""
        out = []
        for a in range_constexpr(self.epi.atoms):
            base = token * fx.Int32(self.epi.hidden) + self.acc_col_f32[a]
            if const_expr(self.epi.split > 1):
                base = base + self.slice_col_f32
            lo = _buffer_load(self.score_buf, base, 4, fx.Float32)
            hi = _buffer_load(self.score_buf, base + fx.Int32(4), 4, fx.Float32)
            out.append(
                fx.Vector.from_elements(
                    [lo[e] for e in range_constexpr(4)]
                    + [hi[e] for e in range_constexpr(4)],
                    fx.Float32,
                )
            )
        return out

    def finish(self, tile, color, accs, prefetched, state):
        epi = self.epi
        prefix_atoms = prefetched
        inv_h = 1.0 / epi.hidden

        # A split build's tile index is the sliced-row index; the token (for
        # per-token data) is ``tile // split``, the slice is columns
        # [part*slice_width, ..). split=1 reads the whole row at ``tile``.
        if const_expr(epi.split > 1):
            token = tile // fx.Int32(epi.split)
        else:
            token = tile

        # prefix = bf16(prefix_in + bf16(all-reduce)), rounded before scoring:
        # downstream readers see these bits, so the score has to as well.
        packed = pack_bf16(residual_add(accs, prefix_atoms))
        self.io.store_tile(self.prefix_out_buf, tile, packed)
        if const_expr(epi.write_bank):
            snap = self._bank_row_i32(token, epi.nvb)
            for a in range_constexpr(epi.atoms):
                col = self.col_i32[a]
                if const_expr(epi.split > 1):
                    col = col + self.slice_col_i32
                _buffer_store(self.bank_buf, snap + col, packed[a])
        pv = [atom_bf16_to_f32(p) for p in packed]

        # The new prefix row's dot and sum of squares. split=1 folds them over
        # the row's waves through LDS; a split build reduces each slice's waves
        # and joins the row across workgroups through two HBM exchange words --
        # the dot is signed (see ``_xchg_join_dot_sq``), the sq is not.
        d_pv, s_pv = self._row_dot_sq(pv)
        done_d = done_s = done_c = None
        # iter010: when the split build also does the output RMSNorm, join the
        # out-norm's cross term (cross=sum(pv*acc)) in the SAME barrier as the
        # prefix dot/sq. The split build is acc_cache-only, so mix = wp*pv + wb*acc
        # with per-token scalars and
        #   sum(mix^2) = wp^2*sq + 2*wp*wb*cross + wb^2*accsq
        # where accsq=sum(acc^2) is the producer's full-row scalar (ms[.,2]); the
        # out-norm scale becomes pure scalar arithmetic -- join point 2 (a whole
        # extra xGMI round-trip of the ~3.5us epilogue) is gone, with no extra
        # joined word vs the two-barrier path (so it wins at every decode T).
        merge = const_expr(epi.split > 1 and epi.out_norm and epi.outnorm_merge)
        acc_atoms = None
        if merge:
            acc_atoms = self._acc_row(token)  # fp32 (T, hidden) weighted sum
            cross_pv = self._row_cross(pv, acc_atoms)
            d_wave = wave_reduce_add([d_pv])[0]
            s_wave = wave_reduce_add([s_pv])[0]
            c_wave = wave_reduce_add([cross_pv])[0]
            (
                total_dot, total_sq, total_cross, done_d, done_s, done_c,
            ) = _xchg_join_triple(
                self._xchg_word(0, color),
                self._xchg_word(1, color),
                self._xchg_word(2, color),
                d_wave, s_wave, c_wave,
                state[0], state[3], state[6],
                epi.n_writers,
                self.tid,
                epi.join_poll_n,
            )
            with fx.fastmath("afn"):
                prefix_score = total_dot * fmath.rsqrt(
                    total_sq * inv_h + self.score_eps
                )
        elif const_expr(epi.split > 1):
            d_wave = wave_reduce_add([d_pv])[0]
            s_wave = wave_reduce_add([s_pv])[0]
            total_dot, total_sq, done_d, done_s = _xchg_join_dot_sq(
                self._xchg_word(0, color),
                self._xchg_word(1, color),
                d_wave,
                s_wave,
                state[0],
                state[3],
                epi.n_writers,
                self.tid,
            )
            with fx.fastmath("afn"):
                prefix_score = total_dot * fmath.rsqrt(
                    total_sq * inv_h + self.score_eps
                )
        else:
            lds = None
            if const_expr(epi.n_waves > 1):
                lds = _LdsWindow(self.lds, color * fx.Int32(epi.slots_per_color))
            totals = _block_reduce_many(
                [d_pv, s_pv], tid=self.tid, block=epi.block, lds=lds
            )
            with fx.fastmath("afn"):
                prefix_score = totals[0] * fmath.rsqrt(
                    totals[1] * inv_h + self.score_eps
                )

        if const_expr(epi.acc_cache):
            # acc-cache (Kevin Mi, sglang#44988 bank_partial_hip.cuh ported to
            # flydsl): the fused read NEVER touches bank VALUES. The caller
            # cached, normalised to the bank's own max m,
            #     ms = (m, sum=sum_r exp(s_r - m), accsq=sum_h acc^2),
            #     acc = sum_r exp(s_r - m)*bank[t,r]
            # and the read folds in only the new row, renormalising by
            # exp(m - m') when the stream raises the max -- Kevin's exact order.
            # ms is (T, 3) fp32 (iter010 added accsq for the single-barrier
            # out-norm; m/sum are unchanged at columns 0/1).
            if not merge:
                acc_atoms = self._acc_row(token)  # fp32 (T, hidden) weighted sum
            bm = _buffer_load(self.ms_buf, token * fx.Int32(3), 1, fx.Float32)[0]
            bsum = _buffer_load(
                self.ms_buf, token * fx.Int32(3) + fx.Int32(1), 1, fx.Float32
            )[0]
            if merge:
                # Producer's full-row sum(acc^2) -- the out-norm's wb^2 term, so it
                # never has to be joined across workgroups.
                bank_accsq = _buffer_load(
                    self.ms_buf, token * fx.Int32(3) + fx.Int32(2), 1, fx.Float32
                )[0]
            m2 = (prefix_score > bm).select(prefix_score, bm)
            bank_scale = fmath.exp(bm - m2)
            stream = fmath.exp(prefix_score - m2)
            inv = fx.Float32(1.0) / (bsum * bank_scale + stream)
            wb = bank_scale * inv
            wp = stream * inv
            mix = []
            for a in range_constexpr(epi.atoms):
                with fx.fastmath("contract"):
                    mix.append(pv[a] * wp + acc_atoms[a] * wb)
        else:
            # Scores-only cache: read nvb bank VALUES + bank scores, do the full
            # weighted mix in-kernel. Global max first (bit-comparable to Triton).
            bank_atoms = self._bank_rows(tile)
            rows = [[atom_bf16_to_f32(x) for x in row] for row in bank_atoms]
            n_rows = epi.nvb + 1
            bank_scores = [
                _buffer_load(
                    self.score_buf,
                    tile * fx.Int32(epi.nvb) + fx.Int32(j),
                    1,
                    fx.Float32,
                )[0]
                for j in range_constexpr(epi.nvb)
            ]
            scores = bank_scores + [prefix_score]
            m = scores[0]
            for r in range_constexpr(1, n_rows):
                m = (scores[r] > m).select(scores[r], m)
            weights = [fmath.exp(s - m) for s in scores]
            denom = None
            for r in range_constexpr(epi.nvb):
                denom = weights[r] if denom is None else denom + weights[r]
            inv = fx.Float32(1.0) / (denom + weights[epi.nvb])
            mix = []
            for a in range_constexpr(epi.atoms):
                with fx.fastmath("contract"):
                    acc = None
                    for r in range_constexpr(epi.nvb):
                        term = rows[r][a] * (weights[r] * inv)
                        acc = term if acc is None else acc + term
                    pterm = pv[a] * (weights[epi.nvb] * inv)
                    mix.append(pterm if acc is None else pterm + acc)

        done_m = None
        if merge:
            # iter010: sum(mix^2) from the single-barrier joined sums + the
            # producer's full-row accsq, no 2nd join. mix = wp*pv + wb*acc ->
            # wp^2*sq + 2*wp*wb*cross + wb^2*accsq.
            with fx.fastmath("afn"):
                total = (
                    wp * wp * total_sq
                    + (fx.Float32(2.0) * wp * wb) * total_cross
                    + wb * wb * bank_accsq
                )
            scale = fx.Float32(1.0) / fmath.sqrt(total * inv_h + self.eps)
            out = scale_by_weight(mix, scale, self.ow_atoms)
        elif const_expr(epi.out_norm):
            local = None
            for a in range_constexpr(epi.atoms):
                s = fx.Float32(
                    (mix[a] * mix[a]).reduce(ReductionOp.ADD, fastmath="reassoc")
                )
                local = s if local is None else local + s
            if const_expr(epi.split > 1):
                # Second join of the tile: the out-norm mix sum of squares, on
                # its own stream (word 2). The colour rotation separates this
                # stream's words exactly as it does the first join's; the two
                # joins are sequential per tile (mix needs the prefix score).
                m_wave = wave_reduce_add([local])[0]
                total, done_m = _xchg_join(
                    self._xchg_word(2, color), m_wave, state[6], epi.n_writers, self.tid
                )
            else:
                lds2 = None
                if const_expr(epi.n_waves > 1):
                    lds2 = _LdsWindow(
                        self.lds,
                        color * fx.Int32(epi.slots_per_color)
                        + fx.Int32(epi.n_score_sums * epi.n_waves),
                    )
                total = block_reduce_add(
                    [local], tid=self.tid, block=epi.block, lds=lds2
                )[0]
            scale = fx.Float32(1.0) / fmath.sqrt(total * inv_h + self.eps)
            out = scale_by_weight(mix, scale, self.ow_atoms)
        else:
            out = pack_bf16(mix)
        self.io.store_tile(self.out_buf, tile, out)

        if const_expr(epi.split > 1):
            # Advance each stream's colour triple: the word just completed for
            # this colour becomes its "last complete" value, read again when the
            # colour returns three tiles on.
            new = [state[1], state[2], done_d, state[4], state[5], done_s]
            if merge:
                # iter010: stream 2 is cross=sum(pv*acc), joined in the same
                # barrier as dot/sq, advanced like the dot/sq triples.
                new += [state[7], state[8], done_c]
            elif const_expr(epi.out_norm):
                # Two-barrier path: stream 2 is the out-norm mix sq (done_m).
                new += [state[7], state[8], done_m]
            return tuple(new)
        return state

    def persist(self, color, state):
        epi = self.epi
        if const_expr(epi.split == 1):
            return
        slots = [
            color,
            self.wrap(color + fx.Int32(1)),
            self.wrap(color + fx.Int32(2)),
        ]
        for stream in range_constexpr(epi.n_streams):
            for s in range_constexpr(3):
                fx.generic_store(
                    self._prev_word(slots[s], stream), state[stream * 3 + s]
                )


__all__ = ["MAX_NVB", "OneShotAttnResEpilogue"]
