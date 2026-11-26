# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""o_proj + TP all-reduce + residual + Gemma RMSNorm in one persistent kernel
(gfx942/gfx950): the back half of mega paged attention, fed by the unfused
pa_decode (bf16, ``quantize``) or by it and the per-token FP8 quant.

One launch per TP rank and layer, for decode batches of at most ``max_rows``
rows. ``grid`` persistent CTAs, all co-resident, walk the phases in program
order:

0. x into ``xs``: 1 KB chunks of 16 rows x 64 K, which a wave's MFMA lanes
   load with one 1 KB load, in mega paged attention's x layout
   [K % 64 / 16][row % 16][16 K]. Either x is the quantizer's row-major FP8
   [rows, K] and a wave copies a chunk (relayout), or x is bf16 [rows, K] and
   a wave quantizes a row with one scale into ``x_scale``, as
   ``dynamic_per_token_scaled_quant`` (and mega paged attention) would
   (``quantize``; on gfx942 a chunk is [row // 4][K % 64 / 16][row % 4][16 K]
   and a CTA stages its 4 rows in LDS, then stores each chunk's 256 B of them
   as whole lines). Each CTA counts its chunks or rows once their stores have
   landed;
1. o_proj: ``block_n``-column tiles of ``x @ W^T``, claimed by the CTAs as
   they finish phase 0 (quantizing on gfx942 up to EARLY_CLAIM_ROWS rows, at
   launch by CTAs with no phase-0 rows when each CTA claims once). A claimed
   tile's W slice is loaded into VGPRs before the wait for every chunk; its
   bf16 partial tile goes to the rank owning its columns (two-shot) or to
   every rank (one-shot);
2. all-reduce + residual + Gemma RMSNorm, a row per CTA.

Phases 1 and 2 are mega paged attention's phases 3 and 4, unchanged: the same
tiles, reductions, inbox layout and rank-order bf16 rounding. On gfx942 o_proj
runs in 32-row passes (16-row passes up to PASS16_ROWS rows when quantizing
two-shot) and each wave stages its fp32 partial in its own LDS region; the
push adds them in the same order (the same sums). A pass's first K steps of x
go out before the previous pass's wave reduce and push. With 4+ passes, the
CTAs left without a tile reduce-scatter the rows of all but the last two
passes while those compute.

``xs`` (and ``x_scale`` when quantizing) must be uncached memory
(``IpcHeap`` uncached), like mega paged attention's hand-off buffers; rows of
the last 16 past the batch keep whatever they hold, so ``xs`` starts zeroed.
``sync`` (int32, zeroed once), a counter a 128 B line: x chunks or rows done
(2 parities), o_proj tiles claimed (2 parities), then
a word a CTA (epoch: launches each CTA has run, so every CTA agrees on the
parity under graph replay). Quantizing on gfx942, x is counted a 16-row
group (2 parities of ``max_rows / 16`` lines), and launch-time claims spread
over CLAIM_LINES counters that never reset; up to LAUNCH_ORDER_ROWS rows the
x loads and those claims go out before the epoch is read.

``trace`` (diagnostics): each CTA writes TRACE_STAMPS int64 100 MHz stamps
into ``trace``: start, phase 0 done, x ready (0 if it ran no o_proj tile),
phase 1 done, phase 2 done.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T
from flydsl.runtime.device import get_rocm_arch

from .comm.one_shot_allreduce import _canonicalize, _has_sentinel
from .comm.quick_allreduce_fusions import (
    residual_add,
    rms_local_sumsq,
    rstd_from_total,
    scale_by_weight,
)
from .comm.transport import (
    _ST_NT,
    _poll_v4i32,
    _store_v4i32_peer,
    atom_bf16_to_f32,
    atom_f32_to_bf16,
)
from .communication_ops_utils import memrealtime
from .mega_paged_attention import (
    _BYPASS,
    AMAX_FLOOR,
    ATOM_COLS,
    INBOX_BUFFERS,
    MAX_ROWS,
    PASS_ROWS,
    SENTINEL_FILL_BYTE,
    SYNC_LINE,
    X_CHUNK,
    _as_f32,
    _bload,
    _i32_ptr,
    _load16,
    _load_agent,
    _rsrc,
    _store16,
    _uniform,
    inbox_bytes,
)
from .pa_decode.traits import WAVE
from .tensor_shim import buf_base_i64
from .utils import rcp_f32

__all__ = [
    "SENTINEL_FILL_BYTE",
    "TRACE_STAMPS",
    "compile_oproj_allreduce_rmsnorm",
    "inbox_bytes",
]

THREADS = 4 * WAVE
TRACE_STAMPS = 8
# gfx942 one-claim builds: tile claim counters (CTA bid claims on bid % 8).
CLAIM_LINES = 8
# ... and the largest max_rows that claims at launch: at 96 and 128 rows the
# launch-time claims cost more than they save.
EARLY_CLAIM_ROWS = 80
# ... and the largest max_rows that issues phase 0's x loads and the
# launch-time claim before the epoch lands: from 48 rows on, the earlier W
# loads cost phase 0 more than they save.
LAUNCH_ORDER_ROWS = 32
# gfx942, quantizing, two-shot: the largest max_rows that runs o_proj in
# 16-row passes (32-row passes above): the first rows' push overlaps the next
# pass's gemm.
PASS16_ROWS = 48


def compile_oproj_allreduce_rmsnorm(
    *,
    row_k: int,
    max_rows: int,
    num_compute_units: int,
    hidden: int,
    world_size: int = 1,
    rank: int = 0,
    block_n: int = 48,
    prefetch: int | None = None,
    occupancy: int = 2,
    two_shot: bool = True,
    grid: int | None = None,
    quantize: bool = False,
    fp8_max: float | None = None,
    trace: bool = False,
):
    """Build the launcher for one rank of ``world_size``.

    ``row_k`` is o_proj's K (x's row), ``hidden`` its N, ``block_n`` its tile
    width and ``prefetch`` how many K steps of x a wave keeps in flight;
    ``None`` takes every step when the registers allow and four otherwise.
    ``grid`` CTAs (default ``occupancy`` a CU) must be resident together.
    ``quantize`` takes bf16 x and quantizes it to the FP8 whose largest
    value is ``fp8_max``. ``trace`` stamps each CTA's phases.
    """
    if max_rows > MAX_ROWS or max_rows % 16:
        raise ValueError(
            f"max_rows must be a multiple of 16 up to {MAX_ROWS}, got {max_rows}"
        )
    if not 0 <= rank < world_size:
        raise ValueError(f"rank {rank} is outside world_size {world_size}")
    two_shot = two_shot and world_size > 1
    grid = occupancy * num_compute_units if grid is None else int(grid)
    if grid > occupancy * num_compute_units:
        raise ValueError(
            f"grid {grid} exceeds the {occupancy * num_compute_units} CTAs that are "
            "resident at once; the persistent schedule would deadlock"
        )
    threads = THREADS
    waves = threads // WAVE
    # The relayout chunks of K: 64 bytes (FP8) of 16 rows each.
    k_chunks = row_k // 64
    if k_chunks * 64 != row_k:
        raise NotImplementedError(f"o_proj K {row_k} must be a multiple of 64")
    # Quantizing, a lane takes q_groups 16-K groups of its wave's row.
    q_groups = row_k // (16 * WAVE)
    if quantize:
        if fp8_max is None:
            raise ValueError("quantize needs fp8_max")
        if q_groups * 16 * WAVE != row_k:
            raise NotImplementedError(
                f"quantizing, o_proj K {row_k} must be a multiple of {16 * WAVE}"
            )
    m_sub = max_rows // 16
    # gfx942: 32-row o_proj passes, so a pass's partial tile fits an LDS
    # region a wave (``wave_regions``) and every K step of x stays in flight;
    # 16-row passes up to PASS16_ROWS rows when quantizing two-shot.
    gfx942 = get_rocm_arch().split(":")[0] == "gfx942"
    if gfx942:
        pass16 = quantize and two_shot and max_rows <= PASS16_ROWS
        pass_subs = min(m_sub, 1 if pass16 else 2)
    else:
        pass_subs = min(m_sub, PASS_ROWS // 16)
    passes = [(m0, min(pass_subs, m_sub - m0)) for m0 in range(0, m_sub, pass_subs)]
    if hidden % block_n or block_n % 16:
        raise ValueError(
            f"block_n {block_n} must be a multiple of 16 dividing {hidden}"
        )
    n_sub = block_n // 16
    n_tiles = hidden // block_n
    if row_k % (waves * 64):
        raise NotImplementedError(
            f"o_proj K {row_k} must split into 64-wide steps a wave"
        )
    k_steps = row_k // (waves * 64)
    if prefetch is None:
        all_x = 4 * (pass_subs * k_steps + n_sub * k_steps + pass_subs * n_sub)
        prefetch = k_steps if all_x <= 240 else min(4, k_steps)
        if len(passes) > 1 and (all_x > 240 or not gfx942):
            prefetch = min(2, k_steps)
    push_iters = -(-(16 * pass_subs * block_n // ATOM_COLS) // threads)
    if hidden % (ATOM_COLS * threads):
        raise NotImplementedError(
            f"hidden {hidden} must split into {ATOM_COLS}-column atoms over {threads} threads"
        )
    row_iters = hidden // (ATOM_COLS * threads)
    slice_cols = hidden // world_size
    if two_shot and (n_tiles % world_size or slice_cols % ATOM_COLS):
        raise ValueError(
            f"two-shot needs {n_tiles} o_proj tiles to split evenly over {world_size} ranks"
        )
    tiles_per_rank = n_tiles // world_size
    slice_atoms = slice_cols // ATOM_COLS
    slice_iters = -(-slice_atoms // threads)
    rs_bytes = INBOX_BUFFERS * world_size * max_rows * slice_cols * 2
    rs_in_tile = two_shot and max_rows <= 16
    gemm_stride = block_n + (4 - block_n) % 8
    gemm_region = 16 * pass_subs * gemm_stride * 4
    # A wave's partial tile in its own region (gfx942: 4 x 6.5 KB at 32 rows),
    # or waves 1 and 3 stage theirs for waves 0 and 2 to add to.
    wave_regions = gfx942
    gemm_regions = waves if wave_regions else 2
    # gfx942, quantizing: phase 0 hands x over in whole lines. A chunk is
    # [row // 4][K % 64 / 16][row % 4][16 K] (``_x_lane``), so a CTA's
    # ``waves`` rows own 256 B of each chunk. They are staged in LDS (in
    # ``buf``, before phase 1 needs it), a row every ``stage_stride`` bytes,
    # and each chunk's 256 B is stored as one run. The 32 B pad puts the 4
    # rows 8 banks apart, so any 8 consecutive storing lanes (4 rows x 2 K
    # groups) read 32 distinct banks. The FP8 x (relayout) keeps the original
    # layout and hand-off.
    staged_x = quantize and gfx942
    stage_stride = row_k + 32
    stage_bytes = waves * stage_stride if staged_x else 0
    stage_stores = (k_chunks * waves * 64) // (threads * 16)
    # Row groups (``waves`` rows each) a quantizing CTA takes.
    quant_iters = -(-max_rows // (grid * waves))
    # One x_done counter a 16-row group, polled by a lane each, when a CTA's
    # phase-0 rows (``waves`` from ``waves * bid``, once) fall in one group.
    grouped_x = staged_x and 16 % waves == 0 and grid * waves >= max_rows
    x_groups = m_sub if grouped_x else 1
    # One claim a CTA: CTA bid claims on counter bid % CLAIM_LINES, which
    # hands out tiles bid % CLAIM_LINES + CLAIM_LINES k. The counters never
    # reset (k is the count less the epoch's), so a CTA with no phase-0 rows
    # claims at launch instead of queueing on one line behind the rest. Only
    # with the group counters (alone it loses at 128 rows), up to
    # EARLY_CLAIM_ROWS.
    early_claim = (
        grouped_x
        and max_rows <= EARLY_CLAIM_ROWS
        and grid >= n_tiles
        and grid % CLAIM_LINES == 0
        and n_tiles % CLAIM_LINES == 0
    )
    # The epoch is an uncached load. Up to LAUNCH_ORDER_ROWS rows nothing waits
    # for it before phase 0's x loads and a launch-time claim go out: only the
    # counts, the claim's k and the parity resets use it.
    launch_order = staged_x and max_rows <= LAUNCH_ORDER_ROWS
    # gfx942, several passes: a wave issues pass p + 1's first x_ahead K steps
    # of x right after pass p's gemm, so they are not queued behind the peer
    # stores of its push (vector memory returns in issue order, stores
    # included). All of them with 4+ ranks; 2 with fewer remote stores or
    # when every tile starts at launch (early_claim), where all of them lose.
    cross_x = gfx942 and len(passes) > 1
    x_ahead = prefetch if world_size >= 4 and not early_claim else min(2, prefetch)
    # gfx942 two-shot, 4+ passes of 32 rows (112 and 128 rows): the rows of
    # the passes before the last two (``early_rows``) reach the reduce-scatter
    # inboxes two passes before the tile CTAs reach phase 2. The
    # ``free_ctas`` CTAs left without a tile (one claim each) reduce-scatter
    # them and push their all-gather meanwhile, the j-th to claim past the
    # tiles rows j + free_ctas e in pass order; CTA ``row`` still gathers,
    # adds the residual and normalizes. (Rows of the pass before the last
    # cost more than they save: their all-gather meets the last pass's push.)
    free_ctas = grid - n_tiles
    early_rows = 0
    if (
        gfx942
        and two_shot
        and pass_subs == 2
        and len(passes) >= 4
        and free_ctas > 0
        and not early_claim
    ):
        early_rows = 16 * passes[-2][0]
    early_per = -(-early_rows // max(free_ctas, 1))
    # ``sync`` words.
    x_done_at = 0
    tile_claim_at = x_done_at + 2 * x_groups * SYNC_LINE
    epoch_at = tile_claim_at + (CLAIM_LINES if early_claim else 2) * SYNC_LINE

    @fx.struct
    class OarStorage:
        # The o_proj partial tiles (a wave each, or waves 0+1, 2+3).
        buf: fx.Array[fx.Int32, max(gemm_regions * gemm_region, stage_bytes) // 4, 16]
        w_scale: fx.Array[fx.Float32, block_n, 16]
        red: fx.Array[fx.Float32, 16, 16]
        word: fx.Array[fx.Int32, 4, 16]

    def _lds(base, index, elem=fx.Int32):
        ptr_ty = fx.PointerType.get(elem.ir_type, fx.AddressSpace.Shared, 4)
        return fx.recast_iter(ptr_ty, fx.add_offset(base, fx.make_int_tuple(index * 4)))

    def _lds_shift(base, nbytes):
        return fx.add_offset(base, fx.make_int_tuple(nbytes))

    def _lds_i32(base, index):
        return fx.Int32(fx.ptr_load(_lds(base, index)))

    @flyc.jit
    def _stamp(trace_ptr, bid, point, tid):
        if const_expr(trace):  # noqa: SIM102
            if tid == 0:
                fx.ptr_store(
                    memrealtime(),
                    fx.add_offset(
                        fx.recast_iter(fx.Int64, trace_ptr), bid * TRACE_STAMPS + point
                    ),
                )

    # ---------------------------------------------------------------- phase 0
    def _x_lane(lane):
        """Byte offset in an x chunk of lane (n1, g)'s 16 B, row n1 = lane % 16
        and K [16 g, 16 g + 16) with g = lane // 16: ``staged_x``'s layout (a
        lane quad loads 64 B), or mega paged attention's ``_x_offset``
        [K % 64 / 16][row % 16][16 K]."""
        if staged_x:
            return (lane % 16 // 4) * 256 + (lane // 16) * 64 + (lane % 4) * 16
        return lane * 16

    @flyc.jit
    def _relayout(
        x_rsrc,
        xs_ptr,
        num_rows,
        bid,
        nblk,
        warp,
        lane,
        sync_ptr,
        x_done,
        word_base,
        tid,
    ):
        """Copy the live rows' chunks of row-major x into ``xs`` (MFMA order), a
        wave a chunk, and count this CTA's chunks once their stores landed.

        Lane (n1, g) of a wave reads row n1, K [16 g, 16 g + 16) of the chunk and
        stores it at ``lane * 16``: chunk bytes [K % 64 / 16][row % 16][16 K], the
        order ``_gemm_tile``'s lanes load (mega paged attention's ``_x_offset``)."""
        live_chunks = ((num_rows + 15) // 16) * k_chunks
        n1 = lane % 16
        g = lane // 16
        mine = fx.Int32(0)
        chunk = bid * waves + warp
        while chunk < live_chunks:
            m = chunk // k_chunks
            kc = chunk - m * k_chunks
            row = 16 * m + n1
            # Rows past the batch (graph padding of the last 16) read the last
            # row: their o_proj results are never pushed.
            row = (row < num_rows).select(row, num_rows - 1)
            atom = _bload(x_rsrc, row * row_k + kc * 64 + g * 16, 4)
            fx.ptr_store(atom, _i32_ptr(xs_ptr, (chunk * X_CHUNK + lane * 16) // 4))
            mine = mine + 1
            chunk = chunk + nblk * waves
        # xs is uncached: a retired store is visible device-wide.
        rocdl.s_waitcnt(vmcnt=0)
        fx.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Release)
        if lane == 0:
            fx.ptr_store(fx.Int32(mine), _lds(word_base, warp))
        gpu.barrier()
        if tid == 0:
            total = _lds_i32(word_base, 0)
            for w in range_constexpr(1, waves):
                total = total + _lds_i32(word_base, w)
            if total > 0:
                fx.atomic_add(_i32_ptr(sync_ptr, x_done), total, syncscope="agent")
        # The claim reuses the broadcast words.
        gpu.barrier()

    @flyc.jit
    def _quantize(
        x_rsrc,
        xs_ptr,
        x_scale_ptr,
        num_rows,
        bid,
        nblk,
        warp,
        lane,
        sync_ptr,
        x_done,
        word_base,
        tid,
    ):
        """Quantize the live rows of bf16 x into ``xs`` (MFMA order) and
        ``x_scale``, a wave a row, and count this CTA's rows once their stores
        landed.

        Lane l takes K [16 q_groups l, 16 q_groups (l + 1)): each 16-K group
        is 16 B of FP8 at its place in its chunk (``mega``'s ``_x_offset``)."""
        mine = fx.Int32(0)
        row = bid * waves + warp
        while row < num_rows:
            src = row * (row_k * 2) + lane * (q_groups * 32)
            atoms = [
                atom_bf16_to_f32(_bload(x_rsrc, src + 16 * i, 4))
                for i in range_constexpr(2 * q_groups)
            ]
            amax = fmath.absf(atoms[0]).reduce(ReductionOp.MAX)
            for atom in atoms[1:]:
                amax = fx.maxnumf(amax, fmath.absf(atom).reduce(ReductionOp.MAX))
            for off in range_constexpr(5, -1, -1):
                amax = fx.maxnumf(amax, amax.shuffle_xor(fx.Int32(1 << off), WAVE))
            scale = fx.maxnumf(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(1.0 / fp8_max)
            inv_scale = fx.full(ATOM_COLS, fx.Float32(rcp_f32(scale)), fx.Float32)
            chunk0 = (row // 16) * k_chunks
            for q in range_constexpr(q_groups):
                words = []
                for h in range_constexpr(2):
                    scaled = atoms[2 * q + h] * inv_scale
                    for i in range_constexpr(2):
                        lo = rocdl.cvt_pk_fp8_f32(
                            T.i32, scaled[4 * i], scaled[4 * i + 1], 0, False
                        )
                        words.append(
                            rocdl.cvt_pk_fp8_f32(
                                T.i32, scaled[4 * i + 2], scaled[4 * i + 3], lo, True
                            )
                        )
                kg = lane * q_groups + q
                dst = (chunk0 + kg // 4) * X_CHUNK + (kg % 4) * 256 + (row % 16) * 16
                fx.ptr_store(
                    fx.Vector.from_elements(words, dtype=fx.Int32),
                    _i32_ptr(xs_ptr, dst // 4),
                )
            if lane == 0:
                fx.ptr_store(
                    scale, fx.add_offset(fx.recast_iter(fx.Float32, x_scale_ptr), row)
                )
            mine = mine + 1
            row = row + nblk * waves
        rocdl.s_waitcnt(vmcnt=0)
        fx.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Release)
        if lane == 0:
            fx.ptr_store(fx.Int32(mine), _lds(word_base, warp))
        gpu.barrier()
        if tid == 0:
            total = _lds_i32(word_base, 0)
            for w in range_constexpr(1, waves):
                total = total + _lds_i32(word_base, w)
            if total > 0:
                fx.atomic_add(_i32_ptr(sync_ptr, x_done), total, syncscope="agent")
        gpu.barrier()

    @flyc.jit
    def _quantize_lines(
        x_rsrc,
        xs_ptr,
        x_scale_ptr,
        num_rows,
        bid,
        warp,
        lane,
        sync_ptr,
        x_done,
        word_base,
        stage_base,
        tid,
    ):
        """``_quantize`` into ``staged_x`` chunks (gfx942): a wave quantizes a
        row of its CTA's row group (``waves`` rows from ``waves * bid``, then a
        grid further) into the LDS stage, and the CTA stores each chunk's
        ``waves`` x 64 B as one contiguous run, whole lines no other CTA writes.
        With ``launch_order``, ``x_done`` is the CTA's epoch, which only the
        count reads (``_x_line``).

        Lane l takes the 16-K groups q WAVE + l of its wave's row, so a wave
        stages 1 KB a group without bank conflicts."""
        mine = fx.Int32(0)
        for it in range_constexpr(quant_iters):
            if const_expr(it > 0):
                # The last group's stores have read the stage.
                gpu.barrier()
            row0 = (bid + it * grid) * waves
            row = row0 + warp
            live = row < num_rows
            mine = mine + live.select(fx.Int32(1), fx.Int32(0))
            if row0 < num_rows:
                # A dead wave of a live group quantizes the last row again;
                # its stage row is never stored.
                src = live.select(row, num_rows - 1) * (row_k * 2) + lane * 32
                atoms = [
                    atom_bf16_to_f32(
                        _bload(x_rsrc, src + (i // 2) * (WAVE * 32) + 16 * (i % 2), 4)
                    )
                    for i in range_constexpr(2 * q_groups)
                ]
                amax = fmath.absf(atoms[0]).reduce(ReductionOp.MAX)
                for atom in atoms[1:]:
                    amax = fx.maxnumf(amax, fmath.absf(atom).reduce(ReductionOp.MAX))
                for off in range_constexpr(5, -1, -1):
                    amax = fx.maxnumf(amax, amax.shuffle_xor(fx.Int32(1 << off), WAVE))
                scale = fx.maxnumf(amax, fx.Float32(AMAX_FLOOR)) * fx.Float32(
                    1.0 / fp8_max
                )
                inv_scale = fx.full(ATOM_COLS, fx.Float32(rcp_f32(scale)), fx.Float32)
                for q in range_constexpr(q_groups):
                    words = []
                    for h in range_constexpr(2):
                        scaled = atoms[2 * q + h] * inv_scale
                        for i in range_constexpr(2):
                            lo = rocdl.cvt_pk_fp8_f32(
                                T.i32, scaled[4 * i], scaled[4 * i + 1], 0, False
                            )
                            words.append(
                                rocdl.cvt_pk_fp8_f32(
                                    T.i32,
                                    scaled[4 * i + 2],
                                    scaled[4 * i + 3],
                                    lo,
                                    True,
                                )
                            )
                    kg = q * WAVE + lane
                    fx.ptr_store(
                        fx.Vector.from_elements(words, dtype=fx.Int32),
                        _lds(stage_base, (warp * stage_stride + kg * 16) // 4),
                    )
                if live & (lane == 0):
                    fx.ptr_store(
                        scale,
                        fx.add_offset(fx.recast_iter(fx.Float32, x_scale_ptr), row),
                    )
                gpu.barrier()
                # Thread i stores 16 B of chunk c = i // 16: the run's 16 B
                # i % 16 is row r = i % 4 of the group, K group 4 c + i % 16 // 4
                # (``_x_lane``); 16 threads write 256 B.
                chunk0 = (row0 // 16) * k_chunks
                for st in range_constexpr(stage_stores):
                    i = tid + st * threads
                    c = i // 16
                    seg = i % 16
                    r = seg % 4
                    kg = c * 4 + seg // 4
                    if row0 + r < num_rows:
                        atom = fx.Vector(
                            fx.ptr_load(
                                _lds(stage_base, (r * stage_stride + kg * 16) // 4),
                                result_type=fx.Vector.make_type(4, fx.Int32),
                            )
                        )
                        dst = (chunk0 + c) * X_CHUNK + (row0 % 16) * 64 + seg * 16
                        fx.ptr_store(atom, _i32_ptr(xs_ptr, dst // 4))
        rocdl.s_waitcnt(vmcnt=0)
        fx.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Release)
        if lane == 0:
            fx.ptr_store(fx.Int32(mine), _lds(word_base, warp))
        gpu.barrier()
        if tid == 0:
            total = _lds_i32(word_base, 0)
            for w in range_constexpr(1, waves):
                total = total + _lds_i32(word_base, w)
            if total > 0:
                line = x_done
                if const_expr(launch_order):
                    line = _x_line(x_done, bid)
                fx.atomic_add(_i32_ptr(sync_ptr, line), total, syncscope="agent")
        gpu.barrier()

    def _x_line(epoch, bid):
        """This launch's x_done line for CTA bid's phase-0 rows."""
        line = x_done_at + (epoch & 1) * (x_groups * SYNC_LINE)
        if grouped_x:
            line = line + (bid * waves // 16) * SYNC_LINE
        return line

    def _phase0(
        x_rsrc,
        xs_ptr,
        x_scale_ptr,
        num_rows,
        bid,
        nblk,
        warp,
        lane,
        sync_ptr,
        x_done,
        word_base,
        stage_base,
        tid,
    ):
        """This build's phase 0, counting on ``x_done``."""
        if staged_x:
            _quantize_lines(
                x_rsrc,
                xs_ptr,
                x_scale_ptr,
                num_rows,
                bid,
                warp,
                lane,
                sync_ptr,
                x_done,
                word_base,
                stage_base,
                tid,
            )
        elif quantize:
            _quantize(
                x_rsrc,
                xs_ptr,
                x_scale_ptr,
                num_rows,
                bid,
                nblk,
                warp,
                lane,
                sync_ptr,
                x_done,
                word_base,
                tid,
            )
        else:
            _relayout(
                x_rsrc,
                xs_ptr,
                num_rows,
                bid,
                nblk,
                warp,
                lane,
                sync_ptr,
                x_done,
                word_base,
                tid,
            )

    @flyc.jit
    def _wait_x(sync_ptr, x_done, num_rows, tid):
        """Phase 1 entry: every live chunk (or row) of xs has landed."""
        if const_expr(grouped_x):
            # Lane g polls group g's counter for its live rows.
            if tid < x_groups:
                live = num_rows - tid * 16
                live = (live < 16).select(live, fx.Int32(16))
                live = (live > 0).select(live, fx.Int32(0))
                line = x_done + tid * SYNC_LINE
                done = _load_agent(sync_ptr, line)
                while done < live:
                    rocdl.s_sleep(1)
                    done = _load_agent(sync_ptr, line)
        else:
            if const_expr(quantize):
                live = num_rows
            else:
                live = ((num_rows + 15) // 16) * k_chunks
            if tid == 0:
                done = _load_agent(sync_ptr, x_done)
                while done < live:
                    rocdl.s_sleep(1)
                    done = _load_agent(sync_ptr, x_done)
        gpu.barrier()
        fx.memory_fence(syncscope="workgroup", ordering=fx.AtomicOrdering.Acquire)

    # ------------------------------------------------- phases 1-2 (mega 3-4)
    def _k_step(tile, s):
        return _uniform((tile + s) % k_steps)

    def _load_w_slice(tile, w_rsrc, lane, warp):
        n1 = lane % 16
        g = lane // 16
        lane_off = (g // 2) * 512 + (g % 2) * 256 + n1 * 16
        base = _uniform(tile * (n_sub * (row_k // 32) * 512) + warp * (k_steps * 1024))
        ws = []
        for s in range_constexpr(k_steps):
            step = _k_step(tile, s)
            ws.append(
                [
                    _bload(
                        w_rsrc,
                        lane_off,
                        4,
                        soffset=base + (j * (row_k // 32) * 512 + step * 1024),
                    )
                    for j in range_constexpr(n_sub)
                ]
            )
        rocdl.sched_barrier(0)
        return ws

    def _x_step(x_rsrc, lane_off, base, tile, m0, m_count, s):
        """A wave's x loads of K step s for rows [16 m0, 16 (m0 + m_count))."""
        return [
            _bload(
                x_rsrc,
                lane_off,
                4,
                _BYPASS,
                soffset=base + ((m0 + m) * (row_k // 64) + _k_step(tile, s)) * X_CHUNK,
            )
            for m in range_constexpr(m_count)
        ]

    def _issue_x(x_rsrc, lane, warp, tile, m0, m_count, steps):
        """The first ``steps`` K steps of a pass's x, issued ahead of its gemm."""
        lane_off = _x_lane(lane)
        base = _uniform(warp * (k_steps * X_CHUNK))
        return [
            _x_step(x_rsrc, lane_off, base, tile, m0, m_count, s) for s in range(steps)
        ]

    def _gemm_tile(ws, x_rsrc, lane, warp, tile, m0, m_count, pre=()):
        """``pre``: the first K steps' x, already issued (``_issue_x``)."""
        lane_off = _x_lane(lane)
        base = _uniform(warp * (k_steps * X_CHUNK))

        def issue_x(s):
            return _x_step(x_rsrc, lane_off, base, tile, m0, m_count, s)

        acc = [
            [fx.full(4, 0.0, fx.Float32) for _j in range(n_sub)]
            for _m in range(m_count)
        ]
        pending = list(pre) + [
            issue_x(s) for s in range(len(pre), min(prefetch, k_steps))
        ]
        for s in range_constexpr(k_steps):
            if s + prefetch < k_steps:
                pending.append(issue_x(s + prefetch))
                rocdl.sched_barrier(0)
            xs = pending[s]
            for m in range_constexpr(m_count):
                a = xs[m].bitcast(fx.Int64)
                for j in range_constexpr(n_sub):
                    b = ws[s][j].bitcast(fx.Int64)
                    v = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4, [a[0], b[0], acc[m][j], 0, 0, 0]
                    )
                    v = rocdl.mfma_f32_16x16x32_fp8_fp8(
                        T.f32x4, [a[1], b[1], v, 0, 0, 0]
                    )
                    acc[m][j] = fx.Vector(v)
        return [v for row in acc for v in row]

    @flyc.jit
    def _claim_tile(sync_ptr, claim, word_base, tid):
        gpu.barrier()
        if tid == 0:
            tile = fx.atomic_add(
                _i32_ptr(sync_ptr, claim), fx.Int32(1), syncscope="agent"
            )
            fx.ptr_store(fx.Int32(tile), _lds(word_base, 1))
        gpu.barrier()
        return _lds_i32(word_base, 1)

    @flyc.jit
    def _claim_once(sync_ptr, claim, epoch, bid, word_base, tid):
        """``early_claim``: broadcast this launch's tile of CTA bid, or
        ``n_tiles``. Each launch adds grid / CLAIM_LINES to every counter."""
        if tid == 0:
            old = fx.atomic_add(
                _i32_ptr(sync_ptr, claim), fx.Int32(1), syncscope="agent"
            )
            k = fx.Int32(old) - epoch * (grid // CLAIM_LINES)
            tile = (k < n_tiles // CLAIM_LINES).select(
                bid % CLAIM_LINES + k * CLAIM_LINES, fx.Int32(n_tiles)
            )
            fx.ptr_store(tile, _lds(word_base, 1))
        gpu.barrier()

    def _first_tile(sync_ptr, claim, word_base, tid):
        """The CTA's first tile: ``_claim_once``'s, or claimed now."""
        if early_claim:
            return _lds_i32(word_base, 1)
        return _claim_tile(sync_ptr, claim, word_base, tid)

    def _counts(epoch, bid):
        """This launch's parity, x_done line (group 0) and claim line."""
        parity = epoch & 1
        x_done = x_done_at + parity * (x_groups * SYNC_LINE)
        if early_claim:
            claim = tile_claim_at + (bid % CLAIM_LINES) * SYNC_LINE
        else:
            claim = tile_claim_at + parity * SYNC_LINE
        return parity, x_done, claim

    @flyc.jit
    def _reset_counts(sync_ptr, parity, bid, tid):
        """CTA 0 zeroes the other parity's counters for the next launch."""
        if (bid == 0) & (tid == 0):
            other = (parity ^ 1) * SYNC_LINE
            if const_expr(grouped_x):
                for g in range_constexpr(x_groups):
                    fx.ptr_store(
                        fx.Int32(0),
                        _i32_ptr(
                            sync_ptr, x_done_at + other * x_groups + g * SYNC_LINE
                        ),
                    )
            else:
                fx.ptr_store(fx.Int32(0), _i32_ptr(sync_ptr, x_done_at + other))
            if const_expr(not early_claim):
                fx.ptr_store(fx.Int32(0), _i32_ptr(sync_ptr, tile_claim_at + other))

    def _stage_partial(acc, lane, region, accumulate):
        for m in range_constexpr(len(acc) // n_sub):
            for j in range_constexpr(n_sub):
                for i in range_constexpr(4):
                    row = 16 * m + 4 * (lane // 16) + i
                    col = 16 * j + lane % 16
                    ptr = _lds(region, row * gemm_stride + col, fx.Float32)
                    v = acc[m * n_sub + j][i]
                    if const_expr(accumulate):
                        v = fx.Float32(fx.ptr_load(ptr)) + v
                    fx.ptr_store(v, ptr)

    @flyc.jit
    def _reduce_waves(acc, warp, lane, gemm_base):
        if const_expr(wave_regions):
            # One write a wave, one barrier: _push_tile adds the 4 partials.
            region = _lds_shift(gemm_base, warp * gemm_region)
            _stage_partial(acc, lane, region, False)
            gpu.barrier()
        else:
            region = gemm_base if warp < 2 else _lds_shift(gemm_base, gemm_region)
            if (warp == 1) | (warp == 3):
                _stage_partial(acc, lane, region, False)
            gpu.barrier()
            if (warp == 0) | (warp == 2):
                _stage_partial(acc, lane, region, True)
            gpu.barrier()

    def _inbox_off(buf, src, row, col):
        return fx.Int64(
            (((buf * world_size + src) * max_rows + row) * hidden + col) * 2
        )

    def _rs_off(buf, src, row, col_local):
        return fx.Int64(
            (((buf * world_size + src) * max_rows + row) * slice_cols + col_local) * 2
        )

    def _ag_off(buf, row, col):
        return fx.Int64(rs_bytes) + fx.Int64(
            ((buf * max_rows + row) * hidden + col) * 2
        )

    def _peer(peers, rank_id):
        addr = peers[0]
        for d in range_constexpr(1, world_size):
            addr = (rank_id == d).select(peers[d], addr)
        return addr

    def _pack(vecs):
        return fx.Vector.from_elements(
            [v[e] for v in vecs for e in range_constexpr(4)], fx.Int32
        )

    def _unpack(got, i):
        return fx.Vector.from_elements([got[4 * i + e] for e in range(4)], fx.Int32)

    def _push_atom(c, tid):
        atom = tid + c * threads
        row = atom // (block_n // ATOM_COLS)
        return atom, row, atom - row * (block_n // ATOM_COLS)

    def _load_col_scales(tile, tid, w_scale_ptr):
        quad = tid % (block_n // 4)
        return fx.Vector(
            fx.ptr_load(
                fx.add_offset(
                    fx.recast_iter(fx.Float32, w_scale_ptr), tile * block_n + 4 * quad
                ),
                result_type=fx.Vector.make_type(4, fx.Float32),
            )
        )

    @flyc.jit
    def _stage_col_scales(scales, tid, w_scale_base):
        if tid < block_n // 4:
            fx.ptr_store(scales, _lds(w_scale_base, 4 * tid, fx.Float32))

    def _col_scales(col, w_scale_base):
        return fx.Vector(
            fx.ptr_load(
                _lds(w_scale_base, col * ATOM_COLS, fx.Float32),
                result_type=fx.Vector.make_type(ATOM_COLS, fx.Float32),
            )
        )

    def _row_scales(tid, x_scale_ptr, row0, num_rows):
        """x's per-row scales of this thread's atoms of the pass at ``row0``,
        [rows] (padding rows read the last live one): the quantizer's, or
        phase 0's past the caches."""
        x_scale_rsrc = _rsrc(x_scale_ptr)
        scales = []
        for c in range_constexpr(push_iters):
            _atom, row, _col = _push_atom(c, tid)
            row = row + row0
            row = (row < num_rows).select(row, num_rows - 1)
            scales.append(
                _as_f32(_bload(x_scale_rsrc, row * 4, 1, _BYPASS if quantize else 0))
            )
        return scales

    @flyc.jit
    def _push_tile(
        tile,
        num_rows,
        tid,
        gemm_base,
        x_scales,
        w_scale_base,
        peers,
        buf,
        row0,
        m_count,
    ):
        tile_atoms = 16 * m_count * block_n // ATOM_COLS
        for c in range_constexpr(-(-tile_atoms // threads)):
            atom, local, col = _push_atom(c, tid)
            row = local + row0
            if (atom < tile_atoms) & (row < num_rows):
                pair = [
                    fx.Vector(
                        fx.ptr_load(
                            _lds(
                                gemm_base,
                                r * (gemm_region // 4)
                                + local * gemm_stride
                                + col * ATOM_COLS,
                                fx.Float32,
                            ),
                            result_type=fx.Vector.make_type(ATOM_COLS, fx.Float32),
                        )
                    )
                    for r in range_constexpr(gemm_regions)
                ]
                if const_expr(wave_regions):
                    # The pair regions' fp32 sums, in their order.
                    acc = (pair[1] + pair[0]) + (pair[3] + pair[2])
                else:
                    acc = pair[0] + pair[1]
                n = tile * block_n + col * ATOM_COLS
                y = (
                    acc
                    * fx.full(ATOM_COLS, x_scales[c], fx.Float32)
                    * _col_scales(col, w_scale_base)
                )
                payload = _canonicalize(atom_f32_to_bf16(y))
                if const_expr(two_shot):
                    owner = tile // tiles_per_rank
                    col_local = n - owner * slice_cols
                    if const_expr(rs_in_tile):
                        if owner == rank:
                            _reduce_atom(payload, peers, buf, row, n, col_local)
                        else:
                            _store_v4i32_peer(
                                _peer(peers, owner)
                                + _rs_off(buf, rank, row, col_local),
                                payload,
                                _ST_NT,
                            )
                    else:
                        _store_v4i32_peer(
                            _peer(peers, owner) + _rs_off(buf, rank, row, col_local),
                            payload,
                            _ST_NT,
                        )
                else:
                    for dst in range_constexpr(world_size):
                        _store_v4i32_peer(
                            peers[dst] + _inbox_off(buf, rank, row, n), payload, _ST_NT
                        )
        gpu.barrier()

    @flyc.jit
    def _block_sum(value, red_base, tid):
        for off in range_constexpr(5, -1, -1):
            value = value + value.shuffle_xor(fx.Int32(1 << off), WAVE)
        if tid % WAVE == 0:
            fx.ptr_store(value, _lds(red_base, tid // WAVE, fx.Float32))
        gpu.barrier()
        result = fx.Float32(fx.ptr_load(_lds(red_base, 0, fx.Float32)))
        for w in range_constexpr(1, waves):
            result = result + fx.Float32(fx.ptr_load(_lds(red_base, w, fx.Float32)))
        return result

    @flyc.jit
    def _reduce_row(
        row,
        buf,
        inbox,
        residual_ptr,
        residual_out_ptr,
        out_ptr,
        norm_w_ptr,
        eps,
        red_base,
        tid,
    ):
        cols = [(tid + c * threads) * ATOM_COLS for c in range_constexpr(row_iters)]
        residual = buf_base_i64(residual_ptr) + fx.Int64(row * hidden * 2)
        residual_out = buf_base_i64(residual_out_ptr) + fx.Int64(row * hidden * 2)
        out = buf_base_i64(out_ptr) + fx.Int64(row * hidden * 2)
        norm_w = buf_base_i64(norm_w_ptr)
        res_atoms = [_load16(residual + fx.Int64(col * 2)) for col in cols]
        w_atoms = [_load16(norm_w + fx.Int64(col * 2)) for col in cols]
        if const_expr(two_shot):
            addrs = [
                inbox + _ag_off(buf, row, cols[c]) for c in range_constexpr(row_iters)
            ]
        else:
            addrs = [
                inbox + _inbox_off(buf, src, row, cols[c])
                for src in range_constexpr(world_size)
                for c in range_constexpr(row_iters)
            ]
        got = _pack([_poll_v4i32(addr) for addr in addrs])
        while _has_sentinel(got):
            got = _pack([_poll_v4i32(addr) for addr in addrs])
        ar = []
        for c in range_constexpr(row_iters):
            if const_expr(two_shot):
                ar.append(atom_bf16_to_f32(_unpack(got, c)))
            else:
                acc = None
                for src in range_constexpr(world_size):
                    v = atom_bf16_to_f32(_unpack(got, src * row_iters + c))
                    acc = v if acc is None else acc + v
                ar.append(acc)
        vals = residual_add(ar, res_atoms)
        for c in range_constexpr(row_iters):
            _store16(residual_out + fx.Int64(cols[c] * 2), atom_f32_to_bf16(vals[c]))
        total = _block_sum(rms_local_sumsq([vals])[0], red_base, tid)
        rstd = rstd_from_total(total, eps, hidden)
        for c, atom in enumerate(scale_by_weight(vals, rstd, w_atoms, w_bias=1.0)):
            _store16(out + fx.Int64(cols[c] * 2), atom)
        sentinel = fx.Vector.filled((4,), -1, fx.Int32)
        for addr in addrs:
            _store_v4i32_peer(addr, sentinel, _ST_NT)

    @flyc.jit
    def _reduce_atom(payload, peers, buf, row, col, col_local):
        inbox = peers[rank]
        addrs = [
            inbox + _rs_off(buf, src, row, col_local)
            for src in range_constexpr(world_size)
            if src != rank
        ]
        got = _pack([_poll_v4i32(addr) for addr in addrs])
        while _has_sentinel(got):
            got = _pack([_poll_v4i32(addr) for addr in addrs])
        acc = None
        peer = 0
        for src in range_constexpr(world_size):
            if const_expr(src == rank):
                v = atom_bf16_to_f32(payload)
            else:
                v = atom_bf16_to_f32(_unpack(got, peer))
                peer += 1
            acc = v if acc is None else acc + v
        summed = _canonicalize(atom_f32_to_bf16(acc))
        for dst in range_constexpr(world_size):
            _store_v4i32_peer(peers[dst] + _ag_off(buf, row, col), summed, _ST_NT)
        sentinel = fx.Vector.filled((4,), -1, fx.Int32)
        for addr in addrs:
            _store_v4i32_peer(addr, sentinel, _ST_NT)

    @flyc.jit
    def _reduce_slice(row, buf, peers, tid):
        inbox = peers[rank]
        sentinel = fx.Vector.filled((4,), -1, fx.Int32)
        for c in range_constexpr(slice_iters):
            atom = tid + c * threads
            if atom < slice_atoms:
                col = atom * ATOM_COLS
                addrs = [
                    inbox + _rs_off(buf, src, row, col)
                    for src in range_constexpr(world_size)
                ]
                got = _pack([_poll_v4i32(addr) for addr in addrs])
                while _has_sentinel(got):
                    got = _pack([_poll_v4i32(addr) for addr in addrs])
                acc = atom_bf16_to_f32(_unpack(got, 0))
                for src in range_constexpr(1, world_size):
                    acc = acc + atom_bf16_to_f32(_unpack(got, src))
                payload = _canonicalize(atom_f32_to_bf16(acc))
                for dst in range_constexpr(world_size):
                    _store_v4i32_peer(
                        peers[dst] + _ag_off(buf, row, rank * slice_cols + col),
                        payload,
                        _ST_NT,
                    )
                for addr in addrs:
                    _store_v4i32_peer(addr, sentinel, _ST_NT)

    @flyc.jit
    def _tile_pass(
        tile,
        m0,
        m_count,
        ws,
        col_scales,
        w_scale_base,
        x_rsrc,
        x_scale_ptr,
        num_rows,
        gemm_base,
        peers,
        buf,
        lane,
        warp,
        tid,
    ):
        row0 = 16 * m0
        x_scales = _row_scales(tid, x_scale_ptr, row0, num_rows)
        acc = _gemm_tile(ws, x_rsrc, lane, warp, tile, m0, m_count)
        _stage_col_scales(col_scales, tid, w_scale_base)
        _reduce_waves(acc, warp, lane, gemm_base)
        _push_tile(
            tile,
            num_rows,
            tid,
            gemm_base,
            x_scales,
            w_scale_base,
            peers,
            buf,
            row0,
            m_count,
        )

    @flyc.jit
    def _pass_chain(
        p,
        pre,
        tile,
        ws,
        col_scales,
        w_scale_base,
        x_rsrc,
        x_scale_ptr,
        num_rows,
        gemm_base,
        peers,
        buf,
        lane,
        warp,
        tid,
    ):
        """``cross_x``: ``_tile_pass`` of pass p, which issues the next pass's
        first ``x_ahead`` K steps of x after its gemm, then the next pass if it
        has live rows (nested in this one, so those loads reach it)."""
        m0, m_count = passes[p]
        row0 = 16 * m0
        x_scales = _row_scales(tid, x_scale_ptr, row0, num_rows)
        acc = _gemm_tile(ws, x_rsrc, lane, warp, tile, m0, m_count, pre)
        nxt = ()
        if const_expr(p + 1 < len(passes)):
            # Pinned: in flight during the wave reduce and the push.
            rocdl.sched_barrier(0)
            nxt = _issue_x(x_rsrc, lane, warp, tile, *passes[p + 1], x_ahead)
            rocdl.sched_barrier(0)
        _stage_col_scales(col_scales, tid, w_scale_base)
        _reduce_waves(acc, warp, lane, gemm_base)
        _push_tile(
            tile,
            num_rows,
            tid,
            gemm_base,
            x_scales,
            w_scale_base,
            peers,
            buf,
            row0,
            m_count,
        )
        if const_expr(p + 1 < len(passes)):  # noqa: SIM102
            if 16 * passes[p + 1][0] < num_rows:
                _pass_chain(
                    p + 1,
                    nxt,
                    tile,
                    ws,
                    col_scales,
                    w_scale_base,
                    x_rsrc,
                    x_scale_ptr,
                    num_rows,
                    gemm_base,
                    peers,
                    buf,
                    lane,
                    warp,
                    tid,
                )

    kernel_name = (
        f"oproj_allreduce_rmsnorm_k{row_k}_r{max_rows}_n{hidden}_bn{block_n}"
        f"_w{world_size}r{rank}_pf{prefetch}_o{occupancy}"
        + ("_2s" if two_shot else "")
        + (f"_q{fp8_max:g}" if quantize else "")
        + ("_trace" if trace else "")
    )

    @flyc.kernel(name=kernel_name, known_block_size=(threads, 1, 1))
    def oproj_allreduce_rmsnorm_kernel(
        x_ptr: fx.Pointer,
        x_scale_ptr: fx.Pointer,
        xs_ptr: fx.Pointer,
        sync_ptr: fx.Pointer,
        w_ptr: fx.Pointer,
        w_scale_ptr: fx.Pointer,
        residual_ptr: fx.Pointer,
        residual_out_ptr: fx.Pointer,
        out_ptr: fx.Pointer,
        norm_w_ptr: fx.Pointer,
        peers_ptr: fx.Pointer,
        trace_ptr: fx.Pointer,
        num_rows: fx.Int32,
        eps: fx.Float32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        nblk = fx.Int32(gpu.grid_dim.x)
        lds = fx.SharedAllocator().allocate(OarStorage).peek()
        gemm_base = fx.recast_iter(fx.Uint8, lds.buf.ptr)
        red_base = fx.recast_iter(fx.Uint8, lds.red.ptr)
        w_scale_base = fx.recast_iter(fx.Uint8, lds.w_scale.ptr)
        word_base = fx.recast_iter(fx.Uint8, lds.word.ptr)
        lane = tid % WAVE
        warp = tid // WAVE
        _stamp(trace_ptr, bid, 0, tid)

        epoch = fx.Int32(fx.ptr_load(_i32_ptr(sync_ptr, epoch_at + bid)))
        if const_expr(not launch_order):
            parity, x_done, claim = _counts(epoch, bid)
            _reset_counts(sync_ptr, parity, bid, tid)

        # Phase 0: x into MFMA-order chunks.
        x_rsrc = _rsrc(x_ptr)
        xs_rsrc = _rsrc(xs_ptr)
        if const_expr(launch_order):
            # Its line is picked from the epoch after the x loads.
            x_mine = epoch
        else:
            if const_expr(grouped_x):
                # This CTA's rows count on their group's line.
                x_mine = x_done + (bid * waves // 16) * SYNC_LINE
            else:
                x_mine = x_done
        if const_expr(early_claim):
            # A CTA without phase-0 rows claims at launch.
            if bid * waves < num_rows:
                _phase0(
                    x_rsrc,
                    xs_ptr,
                    x_scale_ptr,
                    num_rows,
                    bid,
                    nblk,
                    warp,
                    lane,
                    sync_ptr,
                    x_mine,
                    word_base,
                    gemm_base,
                    tid,
                )
            if const_expr(launch_order):
                claim = tile_claim_at + (bid % CLAIM_LINES) * SYNC_LINE
            _claim_once(sync_ptr, claim, epoch, bid, word_base, tid)
        else:
            _phase0(
                x_rsrc,
                xs_ptr,
                x_scale_ptr,
                num_rows,
                bid,
                nblk,
                warp,
                lane,
                sync_ptr,
                x_mine,
                word_base,
                gemm_base,
                tid,
            )
        if const_expr(launch_order):
            parity, x_done, claim = _counts(epoch, bid)
            _reset_counts(sync_ptr, parity, bid, tid)
        _stamp(trace_ptr, bid, 1, tid)

        # Phase 1: o_proj tiles once every chunk has landed.
        peers = [
            fx.Int64(fx.ptr_load(fx.add_offset(fx.recast_iter(fx.Int64, peers_ptr), d)))
            for d in range_constexpr(world_size)
        ]
        w_rsrc = _rsrc(w_ptr)
        tile = _first_tile(sync_ptr, claim, word_base, tid)
        if const_expr(early_rows):
            # j for the j-th CTA to claim past the tiles; negative with a tile.
            free = tile - n_tiles
        while tile < n_tiles:
            col_scales = _load_col_scales(tile, tid, w_scale_ptr)
            ws = _load_w_slice(tile, w_rsrc, lane, warp)
            _wait_x(sync_ptr, x_done, num_rows, tid)
            _stamp(trace_ptr, bid, 2, tid)
            if const_expr(cross_x):
                _pass_chain(
                    0,
                    (),
                    tile,
                    ws,
                    col_scales,
                    w_scale_base,
                    xs_rsrc,
                    x_scale_ptr,
                    num_rows,
                    gemm_base,
                    peers,
                    parity,
                    lane,
                    warp,
                    tid,
                )
            else:
                for p in range_constexpr(len(passes)):
                    m0, m_count = passes[p]
                    if const_expr(p == 0):
                        _tile_pass(
                            tile,
                            m0,
                            m_count,
                            ws,
                            col_scales,
                            w_scale_base,
                            xs_rsrc,
                            x_scale_ptr,
                            num_rows,
                            gemm_base,
                            peers,
                            parity,
                            lane,
                            warp,
                            tid,
                        )
                    else:
                        if 16 * m0 < num_rows:
                            _tile_pass(
                                tile,
                                m0,
                                m_count,
                                ws,
                                col_scales,
                                w_scale_base,
                                xs_rsrc,
                                x_scale_ptr,
                                num_rows,
                                gemm_base,
                                peers,
                                parity,
                                lane,
                                warp,
                                tid,
                            )
            if const_expr(grid < n_tiles):
                tile = _claim_tile(sync_ptr, claim, word_base, tid)
            else:
                tile = fx.Int32(n_tiles)

        _stamp(trace_ptr, bid, 3, tid)

        # Phase 2: a row's all-reduce, residual add and RMSNorm.
        if const_expr(early_rows):
            early_live = (num_rows < early_rows).select(num_rows, fx.Int32(early_rows))
            if free >= 0:
                for e in range_constexpr(early_per):
                    early = free + e * free_ctas
                    if early < early_live:
                        _reduce_slice(early, parity, peers, tid)
        for row in range(bid, num_rows, nblk):
            if const_expr(early_rows):
                if row >= early_live:
                    _reduce_slice(row, parity, peers, tid)
            elif const_expr(two_shot and not rs_in_tile):
                _reduce_slice(row, parity, peers, tid)
            _reduce_row(
                row,
                parity,
                peers[rank],
                residual_ptr,
                residual_out_ptr,
                out_ptr,
                norm_w_ptr,
                eps,
                red_base,
                tid,
            )
            gpu.barrier()

        _stamp(trace_ptr, bid, 4, tid)
        if tid == 0:
            fx.ptr_store(epoch + 1, _i32_ptr(sync_ptr, epoch_at + bid))

    @flyc.jit
    def launch(
        x: fx.Pointer,
        x_scale: fx.Pointer,
        xs: fx.Pointer,
        sync: fx.Pointer,
        w: fx.Pointer,
        w_scale: fx.Pointer,
        residual: fx.Pointer,
        residual_out: fx.Pointer,
        out: fx.Pointer,
        norm_w: fx.Pointer,
        peers: fx.Pointer,
        trace: fx.Pointer,
        num_rows: fx.Int32,
        eps: fx.Float32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        oproj_allreduce_rmsnorm_kernel(
            x,
            x_scale,
            xs,
            sync,
            w,
            w_scale,
            residual,
            residual_out,
            out,
            norm_w,
            peers,
            trace,
            num_rows,
            eps,
        ).launch(grid=(grid, 1, 1), block=(threads, 1, 1), stream=stream)

    waves_per_eu = max(occupancy * threads // (4 * WAVE), 1)
    if waves_per_eu > 1:
        launch.compile_hints["waves_per_eu"] = waves_per_eu
    return {
        "launch": launch,
        "grid": grid,
        "threads": threads,
        "sync_words": epoch_at + grid,
        "inbox_bytes": inbox_bytes(
            world_size=world_size, max_rows=max_rows, hidden=hidden, two_shot=two_shot
        ),
    }
