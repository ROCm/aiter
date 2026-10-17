# Copyright © Advanced Micro Devices, Inc. All rights reserved.
#
# MIT License
"""Send-side compact route plan for gfx1250 TDM dispatch.

A low-LDS persistent-enough grid counts every local route into
``(dest_rank, local_expert)`` buckets, allgathers the histogram, then writes
the destination compact row into ``tok_map``. Dispatch copies payload straight
onto that row, so the receiver never runs ``moe_route_g2l_lds``.
"""

from __future__ import annotations

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl.expr import arith, const_expr, range_constexpr, rocdl
from flydsl.expr.rocdl import readfirstlane, readlane, s_nop
from flydsl.expr.typing import Int32, Int64, T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)

from . import tdm_prims as TDM
from .config import _LANE_MASK as LANE_MASK
from .config import _LOG2_WAVE_SIZE as LOG2_WAVE
from .config import _WAVE_SIZE as WAVE

PLAN_BLOCKS = 1
PLAN_MAX_BLOCKS = 16
PLAN_WAVES = 32
PLAN_THREADS = PLAN_WAVES * WAVE
_LDS_ROUTE_CAP = 8192


def compact_plan_blocks(routes: int) -> int:
    """Workgroups for ``tdm_compact_plan``: one per ``_LDS_ROUTE_CAP`` routes.

    A single workgroup keeps decode on the one-CU path; past the LDS route cap
    the tally and the tok_map rewrite are bound by one CU's memory throughput,
    so prefill splits the routes across blocks.
    """
    env = os.environ.get("AITER_TDM_COMPACT_PLAN_BLOCKS")
    if env:
        return max(1, min(PLAN_MAX_BLOCKS, int(env)))
    routes = max(1, int(routes))
    return max(1, min(PLAN_MAX_BLOCKS, -(-routes // _LDS_ROUTE_CAP)))


def compact_plan_waves(*, npes: int, max_routes: int) -> int:
    """Waves for ``tdm_compact_plan``. Must be >= ``npes`` (one warp TDM-stores each peer).

    The kernel is allgather-bound; 32 waves only help the route tally past ~1K
    routes. Decode (tens of routes) pays the extra waves as launch occupancy.
    """
    env = os.environ.get("AITER_TDM_COMPACT_PLAN_WAVES")
    if env:
        return max(int(npes), int(env))
    routes = max(1, int(max_routes))
    if routes <= 96:
        want = 4
    elif routes <= 384:
        want = 8
    else:
        want = 32
    return max(int(npes), want)


def _align32(n: int) -> int:
    return (int(n) + 31) // 32 * 32


def compact_hist_layout(*, npes: int, experts_per_rank: int, max_routes: int):
    """Per-parity histogram row layout for the symmetric arena.

    Returns ``(row_dwords, sparse_cap, use_sparse)``. Dense rows are ``npes*epr``
    dwords. Sparse rows pack ``nnz`` plus ``(seg, cnt)`` pairs when local routes
    cannot fill the dense table; they are only the fallback for a row TDM cannot
    ship whole, since a dense row carries its own arrival tags and even at one
    token that beats the sparse row's separate arrival counter.
    """
    segs = int(npes) * int(experts_per_rank)
    max_routes = max(1, int(max_routes))
    sparse_cap = min(segs, _align32(max_routes))
    # A padded dense row ships whole by TDM and carries its own arrival tags
    # (no release fence, done atomics or acquire), which beats the sparse
    # row's arrival counter at any route count: the sparse row is opt-in only
    # (AITER_TDM_COMPACT_PLAN_SPARSE=1, the pre-tag layout for EP x E % 32).
    use_sparse = (
        os.environ.get("AITER_TDM_COMPACT_PLAN_SPARSE", "0") == "1"
        and sparse_cap < segs
        and segs % 32 != 0
    )
    # Dense rows are padded to 32 dwords as well: the publish and the matrix
    # read both have a TDM bulk path gated on a 32-aligned row, and the
    # scalar fallback costs ~8us per layer at EP4 x 65 experts (segs=260).
    row_dwords = _align32(1 + sparse_cap) if use_sparse else _align32(segs)
    return row_dwords, sparse_cap, use_sparse


def compact_hist_stride(*, npes: int, experts_per_rank: int, max_routes: int) -> int:
    row_dwords, _, _ = compact_hist_layout(
        npes=npes, experts_per_rank=experts_per_rank, max_routes=max_routes
    )
    return int(npes) * int(row_dwords)


def compact_done_nbytes() -> int:
    """Two parity arrival counters (one i32 each)."""
    return 8


def _tag_stale(v, tag):
    return ((v & fx.Int32(-65536)) - tag) < fx.Int32(0)


def _load_v4_volatile(addr_i64):
    """IR-level ``load volatile <4 x i32>`` (16 B aligned): never hoisted out of
    a spin, and SCOPE_SYS on gfx12. Not a raw buffer load: that intrinsic is
    argmem-readonly in IR, so LLVM hoists it out of a loop with no other
    memory effect and the spin becomes a branch to itself (its aux volatile
    bit only reaches the machine operand, too late)."""
    ptr_type = fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16)
    return fx.Vector(
        fx.generic_load(
            fx.inttoptr(ptr_type, fx.Int64(addr_i64)), count=4, volatile=True
        )
    )


@flyc.jit
def _spin_until_tag4(addr_i64, tag):
    """Spin until the upper halves of the 4 i32 slots at ``addr_i64`` are all
    ``tag`` or newer (one 16 B poll; each dword is tagged on its own, so a mix
    of generations just polls again); return them.

    Newer, not just equal: with no global sync between two plans on one slot
    (back-to-back plans, no dispatch between) a fast peer can already have
    written the next generation, and waiting for an exact match would hang.
    The pipeline always dispatches in between, so there it is always equal.
    """
    v = _load_v4_volatile(addr_i64)
    a0 = fx.Int32(v[0])
    a1 = fx.Int32(v[1])
    a2 = fx.Int32(v[2])
    a3 = fx.Int32(v[3])
    while (
        _tag_stale(a0, tag)
        | _tag_stale(a1, tag)
        | _tag_stale(a2, tag)
        | _tag_stale(a3, tag)
    ):
        v = _load_v4_volatile(addr_i64)
        a0 = fx.Int32(v[0])
        a1 = fx.Int32(v[1])
        a2 = fx.Int32(v[2])
        a3 = fx.Int32(v[3])
    return a0, a1, a2, a3


@flyc.jit
def _wave32_inclusive_scan_i32(value, lane):
    """Inclusive sum within one gfx1250 wave32."""
    value_raw = value.ir_value()
    zero_raw = fx.Int32(0).ir_value()
    for shift, dpp in ((1, 0x111), (2, 0x112), (4, 0x114), (8, 0x118)):
        remote = fx.rocdl.update_dpp(T.i32, zero_raw, value_raw, dpp, 0xF, 0xF, True)
        value = (lane >= fx.Int32(shift)).select(value + fx.Int32(remote), value)
        value_raw = value.ir_value()
    source16 = (lane & fx.Int32(0x10)) - fx.Int32(1)
    remote16 = fx.rocdl.ds_bpermute(T.i32, source16 * fx.Int32(4), value)
    return (lane >= fx.Int32(16)).select(value + fx.Int32(remote16), value)


def compact_row_capacity(
    *,
    max_recv: int,
    topk: int,
    experts_per_rank: int,
    tile_m: int,
) -> int:
    """Static CUDAGraph-safe bound matching grouped_moe contiguous_m."""
    tile_m = int(tile_m)
    ub = int(max_recv) * int(topk) + int(experts_per_rank) * tile_m - int(topk)
    aligned = ((ub + tile_m - 1) // tile_m) * tile_m
    return max(tile_m, aligned)


#: Rows per dispatch-v2 run: one 2-D TDM store of up to this many consecutive
#: destination rows of one (dest, local expert, source) segment. A run never
#: crosses a destination tile (``tile_m`` rows), so it feeds exactly one GEMM
#: m-tile. The run table packs the row count in 4 bits.
PLAN_RUN_ROWS = 8

#: Run-buffer header (i32): [0] runs, [1] rows this rank sends (cnt sum),
#: [2..3] reserved, then one send-list offset per segment (segment order).
RUN_HDR_RUNS = 0
RUN_HDR_ROWS = 1
RUN_HDR_SEND_OFF = 4


def compact_run_header_dwords(*, npes: int, experts_per_rank: int) -> int:
    """Dwords in front of the run table: counts plus per-segment send offsets.

    Padded to 32 dwords so the table (8 B entries) starts on a 128 B line.
    """
    return _align32(RUN_HDR_SEND_OFF + int(npes) * int(experts_per_rank))


def compact_run_capacity(*, max_routes: int) -> int:
    """Static run-table bound. Every run carries at least one route, so a step
    can never have more runs than routes, whatever its tile or routing."""
    return max(1, int(max_routes))


def compact_run_buffer_dwords(
    *, npes: int, experts_per_rank: int, max_routes: int, s1f: bool = False
) -> int:
    """i32 words of one plan slot's run buffer (header + 2 words per run).

    ``s1f`` adds one word per run after the entries: the rows of the run's
    piece, i.e. of this rank's segment inside the run's destination tile.
    """
    words = 3 if s1f else 2
    return compact_run_header_dwords(
        npes=npes, experts_per_rank=experts_per_rank
    ) + words * compact_run_capacity(max_routes=max_routes)


def compact_tile_capacity(*, compact_cap: int, tile_m: int) -> int:
    """Destination m-tiles a compact layout of ``compact_cap`` rows can hold."""
    return (int(compact_cap) + int(tile_m) - 1) // int(tile_m)


@functools.cache
def compile_tdm_compact_plan(
    *,
    rank: int,
    npes: int,
    experts_per_rank: int,
    topk: int,
    tile_m: int,
    compact_cap: int,
    off_hist: int,
    off_done: int,
    hist_stride: int,
    max_routes: int,
    hist_pingpong: bool = True,
    plan_waves: int | None = None,
    plan_blocks: int = PLAN_BLOCKS,
    emit_runs: bool = False,
    run_rows: int = PLAN_RUN_ROWS,
    s1f: bool = False,
    write_tok_map: bool = True,
    align_m: int | None = None,
):
    """Compile the compact-plan kernel. ``hist_stride`` is ``npes * row_dwords``.

    ``hist_pingpong``: the single-buffer protocol indexes the 2-deep hist/done
    arena by ``gen & 1``. Double-buffered callers pass a slot-specific
    ``off_hist`` / ``off_done`` and set this False so two in-flight plans do
    not share a done counter.

    ``max_routes`` sizes the histogram layout and the LDS route table, which
    every rank of a step must share. ``plan_waves`` is only the launch width:
    a decode step passes the wave count for its own route count so the plan
    stays one small block beside the expert GEMM. Omit it to use the width
    ``max_routes`` itself selects.

    ``plan_blocks`` > 1 gives each block a contiguous chunk of routes and needs
    ``block_hist`` to hold ``plan_blocks * segs`` dwords. Block 0 alone runs the
    peer exchange and the scan; the others wait on its release in ``barrier``.

    ``emit_runs`` (dispatch v2) also writes this rank's send order, and the
    launcher takes two more pointers, ``addr_send_list`` and ``addr_runs``:

    * ``runs`` (``compact_run_buffer_dwords`` i32): a header (run count, rows
      sent, per-segment send-list offsets) and then one 2-word entry per run,
      ``((row0 << peer_bits) | dest, (send_off << 4) | nrows)``. A run is up to
      ``run_rows`` consecutive destination rows of one (dest, local expert,
      this rank) segment that never crosses a ``tile_m`` destination tile;
      ``row0 = send_base[seg] + intra``. Runs are ordered local expert outer,
      destination rotated by rank inside (``(rank + j) % npes``, local rows
      first), tiles ascending -- the order dispatch v2 walks them in.
    * ``send_list`` (``max_routes`` i32): the route id (``tok * topk + k``) of
      every row this rank sends, in run order, so run ``(off, n)`` ships routes
      ``send_list[off:off + n]`` to rows ``row0 .. row0 + n - 1``.

    Rows past ``compact_cap`` are dropped from both, exactly as from tok_map,
    which is still written: the default dispatch keeps reading it.

    ``s1f`` (the fused dispatch + GEMM1 stage-1 kernel) implies ``emit_runs``
    and adds, for the producers' per-(dest, tile) piece accounting, one word
    per run after the entries: ``runs[hdr + 2 * run_cap + run]`` = the rows of
    the run's piece (this rank's segment clipped to the run's destination
    tile). The launcher takes one more pointer, ``addr_tile_src``
    (``compact_tile_capacity`` i32): for each of THIS rank's m-tiles,
    ``tile_info = srcmask | expert << 8 | rows << 16``. ``srcmask`` is the
    bitmask of the sources with at least one row in it, which is the set of
    per-(tile, source) flags a GEMM1 item waits for; ``expert`` the local
    expert owning the tile and ``rows`` its valid rows (``min(tile_m, end -
    tile start)``, clipped to ``compact_cap``), which the GEMM1 item uses
    instead of bisecting psum.

    ``write_tok_map=False`` (S1F only) skips the tok_map rewrite: the stage-1
    kernel reads the send list and the run table, tok_map only feeds the
    default dispatch (and the S1F check mode, which keeps it).

    ``align_m`` (a multiple of ``tile_m``; default ``tile_m``) pads every
    expert to that many rows instead of ``tile_m``. ``tile_m`` then only cuts
    the runs and the tile_info m-tiles, so a stage-1 kernel whose GEMM1 tile is
    ``align_m`` keeps its producers' ``tile_m``-row pieces and reads ``align_m
    // tile_m`` tile_info words (and flags) per GEMM1 m-tile.

    """
    if WAVE != 32:
        raise ValueError("compact plan requires gfx1250 wave32")
    epr = int(experts_per_rank)
    segs = int(npes) * epr
    if segs > 1024:
        raise ValueError(
            f"compact plan LDS hist supports at most 1024 segments, got {segs}"
        )
    tile_m = int(tile_m)
    align_m = tile_m if align_m is None else int(align_m)
    if align_m < tile_m or align_m % tile_m:
        raise ValueError(f"align_m={align_m} must be a multiple of tile_m={tile_m}")
    compact_cap = int(compact_cap)
    max_routes = max(1, int(max_routes))
    hist_pingpong = bool(hist_pingpong)
    plan_blocks = int(plan_blocks)
    if not 1 <= plan_blocks <= PLAN_MAX_BLOCKS:
        raise ValueError(f"plan_blocks={plan_blocks} not in [1, {PLAN_MAX_BLOCKS}]")
    default_waves = compact_plan_waves(npes=npes, max_routes=max_routes)
    if plan_waves is None:
        plan_waves = default_waves
    else:
        plan_waves = max(int(npes), int(plan_waves))
    plan_threads = plan_waves * WAVE
    kernel_name = "tdm_compact_plan"
    if plan_blocks != 1:
        kernel_name += f"_b{plan_blocks}"
    if plan_waves != default_waves:
        kernel_name += f"_w{plan_waves}"
    if align_m != tile_m:
        kernel_name += f"_a{align_m}"
    s1f = bool(s1f)
    emit_runs = bool(emit_runs) or s1f
    if s1f and (epr > 255 or tile_m > 0xFFFF):
        raise ValueError(
            "tile_info packs the expert in 8 bits and the tile's rows in 16"
        )
    run_rows = int(run_rows)
    if emit_runs:
        kernel_name += "_s1f" if s1f else "_runs"
        if not 1 <= run_rows <= 15:
            raise ValueError(f"run_rows={run_rows} must fit the 4-bit run field")
        if max_routes >= (1 << 27):
            raise ValueError(f"max_routes={max_routes} overflows the run send offset")
    run_hdr = compact_run_header_dwords(npes=npes, experts_per_rank=epr)
    run_cap = compact_run_capacity(max_routes=max_routes)
    run_buf_bytes = 4 * (run_hdr + (3 if s1f else 2) * run_cap)
    tile_cap = compact_tile_capacity(compact_cap=compact_cap, tile_m=tile_m)
    peer_bits = max(1, (int(npes) - 1).bit_length())
    peer_mask = (1 << peer_bits) - 1
    if compact_cap >= (1 << (31 - peer_bits)):
        raise ValueError(
            "compact row encoding exceeds positive i32: "
            f"cap={compact_cap} peers={npes}"
        )
    hist_stride = int(hist_stride)
    row_dwords, _sparse_cap, use_sparse = compact_hist_layout(
        npes=npes, experts_per_rank=epr, max_routes=max_routes
    )
    if hist_stride != npes * row_dwords:
        raise ValueError(
            f"hist_stride={hist_stride} != npes*row_dwords={npes * row_dwords}"
        )
    # LDS hist/matrix are indexed by segment on every path, so they keep the
    # dense pitch; row_dwords is only the window row, which on the sparse path
    # is the shorter packed (nnz, seg|cnt) row.
    hist_dwords = _align32(segs)
    # Dense rows ship (gen16 << 16) | count, so a count must fit 16 bits; one
    # source's count toward an expert is at most its token count. A padded
    # dense row (segs % 32 != 0, e.g. EP4 x 65 experts = 260) is tagged too:
    # its pad dwords carry the tag like the counts, so the reader's per-dword
    # wait covers the whole TDM row and the release fence, the done-counter
    # atomics and the acquire are not needed (quickwins-02).
    use_tag = (
        not use_sparse
        and row_dwords % 32 == 0
        and max_routes // max(1, int(topk)) < (1 << 16)
    )
    if max_routes % 4 == 0:
        route_vec = 4
    elif max_routes % 2 == 0:
        route_vec = 2
    else:
        route_vec = 1
    # Upper bound on one block's chunk; chunks stay route_vec aligned so the
    # vector loads of every block start on an aligned route.
    per_block = -(-max_routes // plan_blocks)
    chunk_cap = -(-per_block // route_vec) * route_vec
    merge_routes = chunk_cap <= _LDS_ROUTE_CAP
    dropped = -1
    write_tok_map = bool(write_tok_map)
    if not write_tok_map and not emit_runs:
        raise ValueError(
            "write_tok_map=False needs emit_runs (S1F): tok_map is the default "
            "dispatch's only input"
        )
    if not write_tok_map:
        kernel_name += "_nm"
    # The send list is staged in LDS (one slot per route of this block) and
    # copied out coalesced: a per-route scattered store costs a cache line per
    # lane (~2.4 us at 7K routes on one CU). It needs only this rank's own
    # counts, so it is built while the X1 rows are in flight. Without the
    # LDS route table (chunk_cap > _LDS_ROUTE_CAP) it is scattered as before.
    stage_send = emit_runs and merge_routes
    # S1F without tok_map: the other blocks need only the send offsets, which
    # block 0 has before X1, so they are released then (barrier[3]).
    early_release = emit_runs and stage_send and not write_tok_map and plan_blocks > 1
    # Per-wave ranges of run-order positions (le outer, destination rotated
    # inner) or segments for the send offsets / copies: each wave owns
    # pos_per_wave consecutive ones and sums the prefix before them itself
    # (independent LDS loads + one reduction), so no block-wide scan.
    n_chunks = (segs + WAVE - 1) // WAVE
    pos_per_wave = -(-segs // plan_waves)
    tm_pow2 = tile_m & (tile_m - 1) == 0
    rr_pow2 = run_rows & (run_rows - 1) == 0
    tm_shift = tile_m.bit_length() - 1
    am_pow2 = align_m & (align_m - 1) == 0
    am_shift = align_m.bit_length() - 1
    rr_shift = run_rows.bit_length() - 1
    per_tile = (tile_m + run_rows - 1) // run_rows
    bnd_n = npes + 1

    def _lds_barrier():
        """Workgroup barrier ordering LDS only (and TDM data already waited on).

        ``fx.barrier()`` lowers to s_wait_loadcnt 0 + s_wait_storecnt 0 +
        s_barrier: it exposes every global load still in flight (the gen and
        window loads issued at entry) and every store's acknowledgement (psum,
        tables) at each phase boundary. The singlethread fences only keep the
        compiler from moving memory accesses across; they emit nothing."""
        fx.memory_fence(ordering=fx.AtomicOrdering.SeqCst, syncscope="singlethread")
        fx.rocdl.s_wait_dscnt(0)
        rocdl.s_barrier_signal(-1)
        rocdl.s_barrier_wait(-1)
        fx.memory_fence(ordering=fx.AtomicOrdering.SeqCst, syncscope="singlethread")

    def _tdiv(x):
        return x >> fx.Int32(tm_shift) if tm_pow2 else x // fx.Int32(tile_m)

    def _tmod(x):
        return x & fx.Int32(tile_m - 1) if tm_pow2 else x % fx.Int32(tile_m)

    def _aceil(x):
        """``x`` rounded up to the expert alignment ``align_m``."""
        x = x + fx.Int32(align_m - 1)
        q = x >> fx.Int32(am_shift) if am_pow2 else x // fx.Int32(align_m)
        return q * fx.Int32(align_m)

    def _rceil(x):
        x = x + fx.Int32(run_rows - 1)
        return x >> fx.Int32(rr_shift) if rr_pow2 else x // fx.Int32(run_rows)

    def _imin(a, b):
        return fx.Int32((a < b).select(a, b))

    def _imax(a, b):
        return fx.Int32((a > b).select(a, b))

    def _first_part(b, c):
        """Rows of [b, b+c) inside b's destination tile."""
        return _imin(c, fx.Int32(tile_m) - _tmod(b))

    def _run_count(b, c):
        """Greedy runs of <= run_rows rows over [b, b+c), split at tiles."""
        first = _first_part(b, c)
        rest = c - first
        full = _tdiv(rest)
        last = rest - full * fx.Int32(tile_m)
        return _rceil(first) + full * fx.Int32(per_tile) + _rceil(last)

    # One body, three kernel signatures: the default plan keeps its exact
    # argument list, dispatch v2 and S1F append their outputs.
    @flyc.jit
    def plan_body(
        arena,
        addr_inp_idx,
        addr_tok_map,
        addr_block_hist,
        addr_send_base,
        addr_masked_m,
        addr_psum,
        addr_barrier,
        my_lsa_rank,
        inp_cur_tok,
        addr_send_list,
        addr_runs,
        addr_tile_src,
    ):
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid & LANE_MASK
        warp = tid >> LOG2_WAVE
        warp_u = fx.Int32(readfirstlane(T.i32, arith.unwrap(fx.Int32(warp))))
        lane_i = fx.Int32(lane)
        # Window geometry, read once at entry: cco.Window.lsa_ptr reloads the
        # descriptor (a dependent flat_load round trip) at every address.
        win = cco.CachedWindow(arena)
        win_base = win.base
        win_stride = win.stride

        def lsa(peer, off):
            return win_base + fx.Int64(peer) * win_stride + fx.Int64(off)

        rsrc_idx = create_buffer_resource_from_addr(addr_inp_idx)
        rsrc_map = create_buffer_resource_from_addr(addr_tok_map)
        rsrc_bhist = create_buffer_resource_from_addr(addr_block_hist)
        rsrc_base = create_buffer_resource_from_addr(addr_send_base)
        rsrc_mm = create_buffer_resource_from_addr(addr_masked_m)
        rsrc_psum = create_buffer_resource_from_addr(addr_psum)
        rsrc_bar = create_buffer_resource_from_addr(addr_barrier)

        smem = fx.SharedAllocator(static=False)
        hist_ptr = smem.allocate(hist_dwords * 4, 128)._ptr
        total_ptr = smem.allocate(segs * 4, 16)._ptr
        pref_ptr = smem.allocate(segs * 4, 16)._ptr
        matrix_ptr = smem.allocate(npes * hist_dwords * 4, 128)._ptr
        lds_hist = fx.Int64(fx.ptrtoint(hist_ptr))
        lds_total = fx.Int64(fx.ptrtoint(total_ptr))
        lds_pref = fx.Int64(fx.ptrtoint(pref_ptr))
        lds_matrix = fx.Int64(fx.ptrtoint(matrix_ptr))
        if const_expr(merge_routes):
            routes_ptr = smem.allocate(chunk_cap * 4, 16)._ptr
            lds_routes = fx.Int64(fx.ptrtoint(routes_ptr))
        if const_expr(use_sparse):
            pack_ptr = smem.allocate(row_dwords * 4, 128)._ptr
            recv_ptr = smem.allocate(npes * row_dwords * 4, 128)._ptr
            lds_pack = fx.Int64(fx.ptrtoint(pack_ptr))
            lds_recv = fx.Int64(fx.ptrtoint(recv_ptr))

        def _lds_at(base, idx):
            return base + fx.Int64(idx) * fx.Int64(4)

        def _ld(base, idx):
            return fx.Int32(comm_ops.load_i32_lds(_lds_at(base, idx)))

        def _st(base, idx, v):
            comm_ops.store_i32_lds(_lds_at(base, idx), v)

        def _bcast(v, i):
            """Lane i's value (i wave-uniform), as a uniform value."""
            return fx.Int32(
                readlane(T.i32, arith.unwrap(fx.Int32(v)), arith.unwrap(fx.Int32(i)))
            )

        def _wsum(v):
            """Sum over the wave, uniform."""
            inc = _wave32_inclusive_scan_i32(fx.Int32(v), lane_i)
            return fx.Int32(readlane(T.i32, arith.unwrap(inc), WAVE - 1))

        if const_expr(emit_runs):
            # Sized: a stray index is dropped by the hardware, not written.
            rsrc_send = create_buffer_resource_from_addr(
                addr_send_list, num_records_bytes=4 * max_routes
            )
            rsrc_runs = create_buffer_resource_from_addr(
                addr_runs, num_records_bytes=run_buf_bytes
            )
            if const_expr(s1f):
                rsrc_tsrc = create_buffer_resource_from_addr(
                    addr_tile_src, num_records_bytes=4 * tile_cap
                )
            # sendpos[seg]: send-list slot of this block's first route of seg.
            sendpos_ptr = smem.allocate(segs * 4, 16)._ptr
            # Per run-order position (my rows of its segment): rows inside
            # compact_cap, and their runs.
            ccl_ptr = smem.allocate(segs * 4, 16)._ptr
            rcnt_ptr = smem.allocate(segs * 4, 16)._ptr
            # Per position: its first run within its 32-position chunk; per
            # chunk: its runs.
            rexcl_ptr = smem.allocate(segs * 4, 16)._ptr
            ctot_ptr = smem.allocate(n_chunks * 4, 16)._ptr
            # [0] rows this rank sends, [1] m-tiles of this rank.
            misc_ptr = smem.allocate(4 * 4, 16)._ptr
            lds_sendpos = fx.Int64(fx.ptrtoint(sendpos_ptr))
            lds_ccl = fx.Int64(fx.ptrtoint(ccl_ptr))
            lds_rcnt = fx.Int64(fx.ptrtoint(rcnt_ptr))
            lds_rexcl = fx.Int64(fx.ptrtoint(rexcl_ptr))
            lds_ctot = fx.Int64(fx.ptrtoint(ctot_ptr))
            lds_misc = fx.Int64(fx.ptrtoint(misc_ptr))
            if const_expr(s1f):
                # m-tile -> local expert, and each local expert's source
                # boundaries (bnd_n per expert; the last is its end row).
                t2e_ptr = smem.allocate(tile_cap * 4, 16)._ptr
                bnd_ptr = smem.allocate(epr * bnd_n * 4, 16)._ptr
                lds_t2e = fx.Int64(fx.ptrtoint(t2e_ptr))
                lds_bnd = fx.Int64(fx.ptrtoint(bnd_ptr))
            if const_expr(stage_send):
                # This block's send-list entries: one block -> at their final
                # slot; several -> segment-major at lpos[seg] + intra.
                send_ptr = smem.allocate(chunk_cap * 4, 16)._ptr
                lds_send = fx.Int64(fx.ptrtoint(send_ptr))
                if const_expr(plan_blocks > 1):
                    lpos_ptr = smem.allocate(segs * 4, 16)._ptr
                    bcnt_ptr = smem.allocate(segs * 4, 16)._ptr
                    lds_lpos = fx.Int64(fx.ptrtoint(lpos_ptr))
                    lds_bcnt = fx.Int64(fx.ptrtoint(bcnt_ptr))

            def _run_seg(p):
                """Run-order position -> (segment, dest): le outer, j inner."""
                le = p // fx.Int32(npes)
                d = p - le * fx.Int32(npes) + fx.Int32(rank)
                d = fx.Int32((d >= fx.Int32(npes)).select(d - fx.Int32(npes), d))
                return d * fx.Int32(epr) + le, d

            def _wave_positions(order, value_fn, emit_fn, total_fn=None):
                """Exclusive prefix of value_fn(i) over i in [0, segs), for this
                wave's i in [w * G, w * G + G) (G = pos_per_wave): emit_fn(i,
                live, excl, n) per batch of n <= 32 (one i per lane). The prefix
                before the wave's range is summed by the wave itself, so the
                waves need no barrier between them. total_fn(total): the wave
                holding the last i, with the sum over all i. ``order(i)`` maps
                i to the LDS index value_fn reads (segment or position)."""
                i_lo = warp_u * fx.Int32(pos_per_wave)
                i_hi = _imin(i_lo + fx.Int32(pos_per_wave), fx.Int32(segs))
                acc = fx.Int32(0)
                for k in range_constexpr(n_chunks):
                    i = fx.Int32(k * WAVE) + lane_i
                    i_s = fx.Int32((i < fx.Int32(segs)).select(i, fx.Int32(0)))
                    v = value_fn(order(i_s))
                    acc = acc + fx.Int32((i < i_lo).select(v, fx.Int32(0)))
                carry = _wsum(acc)
                for ib in range(i_lo, i_hi, fx.Int32(WAVE)):
                    il = ib + lane_i
                    live = il < i_hi
                    il_s = fx.Int32(live.select(il, fx.Int32(0)))
                    v = fx.Int32(live.select(value_fn(order(il_s)), fx.Int32(0)))
                    inc = _wave32_inclusive_scan_i32(v, lane_i)
                    emit_fn(
                        il_s, live, carry + inc - v, _imin(i_hi - ib, fx.Int32(WAVE))
                    )
                    carry = carry + fx.Int32(
                        readlane(T.i32, arith.unwrap(inc), WAVE - 1)
                    )
                if const_expr(total_fn is not None):  # noqa: SIM102
                    if (i_hi == fx.Int32(segs)) & (i_lo < fx.Int32(segs)):
                        total_fn(carry)

            def _seg_of_pos(p):
                return _run_seg(p)[0]

            def _ident(i):
                return i

            cmask = 0xFFFF if use_tag else -1

            def _lpos_emit(s_, live, excl, nb):
                if live:
                    _st(lds_lpos, s_, excl)

            def _stage_segments():
                """Several blocks: this block's segment-major staging offsets
                (from its own counts, saved in lds_bcnt) and the staging."""
                _wave_positions(_ident, lambda s_: _ld(lds_bcnt, s_), _lpos_emit)
                _lds_barrier()
                _stage_routes(lds_lpos)

            def _soff_emit(p, live, excl, nb):
                if live:
                    seg_p = _seg_of_pos(p)
                    _st(lds_sendpos, seg_p, excl)
                    if const_expr(plan_blocks > 1):
                        buffer_store(excl, rsrc_runs, RUN_HDR_SEND_OFF + seg_p)

            def _soff_total(total):
                if lane_i == fx.Int32(0):
                    buffer_store(total, rsrc_runs, RUN_HDR_ROWS)
                    _st(lds_misc, 0, total)

        # Issued now so the round trip hides behind the tally; the previous
        # plan on this slot finished before this launch, so it is current.
        # Every block reads it before it arrives below, and block 0 advances it
        # only once all have arrived.
        gen = buffer_load(rsrc_bar, 2, vec_width=1, dtype=T.i32) + arith.constant(1)
        # hist_dwords, not segs: the pad tail ships with the dense row.
        for s in range(tid, hist_dwords, plan_threads):
            comm_ops.store_i32_lds(
                lds_hist + fx.Int64(s) * fx.Int64(4), arith.constant(0)
            )
        _lds_barrier()

        n_routes = inp_cur_tok * fx.Int32(topk)
        if const_expr(plan_blocks == 1):
            r0 = fx.Int32(0)
            r_n = n_routes
        else:
            per_blk = (n_routes + fx.Int32(plan_blocks - 1)) // fx.Int32(plan_blocks)
            blk_routes = (
                (per_blk + fx.Int32(route_vec - 1))
                // fx.Int32(route_vec)
                * fx.Int32(route_vec)
            )
            r0 = fx.Int32(bid) * blk_routes
            r_end = (r0 + blk_routes < n_routes).select(r0 + blk_routes, n_routes)
            r_n = (r_end > r0).select(r_end - r0, fx.Int32(0))

        def _tally_one(local, expert):
            route = r0 + local
            dest_pe = expert // epr
            valid = (expert >= 0) & (dest_pe >= 0) & (dest_pe < npes)
            local_e = expert - dest_pe * fx.Int32(epr)
            segment = dest_pe * fx.Int32(epr) + local_e
            intra = arith.constant(0)
            if valid:
                intra = comm_ops.atomic_add_lds(
                    lds_hist + fx.Int64(segment) * fx.Int64(4), arith.constant(1)
                )
            packed = arith.select(
                valid,
                segment | (intra << arith.constant(16)),
                arith.constant(dropped),
            )
            if const_expr(merge_routes):
                comm_ops.store_i32_lds(
                    lds_routes + fx.Int64(local) * fx.Int64(4), packed
                )
            else:
                buffer_store(packed, rsrc_map, route)

        vec_n = r_n - (r_n & fx.Int32(route_vec - 1))
        if const_expr(route_vec == 1):
            for local in range(tid, vec_n, plan_threads):
                expert = buffer_load(rsrc_idx, r0 + local, vec_width=1, dtype=T.i32)
                _tally_one(local, expert)
        else:
            # Two vectors per trip, both loads issued before either is used:
            # one exposed load round trip for up to 2 * plan_threads vectors
            # (a whole block's chunk at the 8K-route cap), not one per vector.
            step = plan_threads * route_vec
            for local in range(tid * route_vec, vec_n, 2 * step):
                raw0 = fx.Vector(
                    buffer_load(rsrc_idx, r0 + local, vec_width=route_vec, dtype=T.i32)
                )
                has1 = local + step < vec_n
                # Past the end: reload the first vector (never out of bounds).
                local1 = fx.Int32(has1.select(local + step, local))
                raw1 = fx.Vector(
                    buffer_load(rsrc_idx, r0 + local1, vec_width=route_vec, dtype=T.i32)
                )
                for k in range_constexpr(route_vec):
                    _tally_one(local + k, raw0[k])
                if has1:
                    for k in range_constexpr(route_vec):
                        _tally_one(local + step + k, raw1[k])
        for local in range(vec_n + tid, r_n, plan_threads):
            expert = buffer_load(rsrc_idx, r0 + local, vec_width=1, dtype=T.i32)
            _tally_one(local, expert)

        if const_expr(merge_routes):
            _lds_barrier()
        else:
            # The rewrite reads the packed routes back from tok_map in
            # another thread order: the stores must land first.
            fx.barrier()

        def _stage_routes(slot_base):
            """This block's valid routes into lds_send[slot_base[seg] + intra]."""
            for local in range(tid, r_n, plan_threads):
                packed = _ld(lds_routes, local)
                if packed >= fx.Int32(0):
                    seg_r = packed & fx.Int32(0xFFFF)
                    _st(
                        lds_send,
                        _ld(slot_base, seg_r) + (packed >> fx.Int32(16)),
                        r0 + local,
                    )

        def _copy_segments():
            """Several blocks: this block's entries of each segment are
            consecutive at lpos[seg] here and at sendpos[seg] in the send
            list; waves over segments, lanes over a segment's entries."""
            s_lo = warp_u * fx.Int32(pos_per_wave)
            s_hi = _imin(s_lo + fx.Int32(pos_per_wave), fx.Int32(segs))
            for sb in range(s_lo, s_hi, fx.Int32(WAVE)):
                sl = sb + lane_i
                sl_s = fx.Int32((sl < s_hi).select(sl, fx.Int32(0)))
                n_l = _ld(lds_bcnt, sl_s)
                g_l = _ld(lds_sendpos, sl_s)
                l_l = _ld(lds_lpos, sl_s)
                nb = _imin(s_hi - sb, fx.Int32(WAVE))
                for i in range(fx.Int32(0), nb, fx.Int32(1)):
                    n_s = _bcast(n_l, i)
                    g_s = _bcast(g_l, i)
                    l_s = _bcast(l_l, i)
                    for j0 in range(fx.Int32(0), n_s, fx.Int32(WAVE)):
                        j = j0 + lane_i
                        if j < n_s:
                            buffer_store(_ld(lds_send, l_s + j), rsrc_send, g_s + j)

        if const_expr(plan_blocks > 1):
            # barrier[0] counts arrivals and is reset by block 0 once all are in:
            # plans on one slot share a stream, so the next launch starts after.
            for s in range(tid, segs, plan_threads):
                cnt = comm_ops.load_i32_lds(lds_hist + fx.Int64(s) * fx.Int64(4))
                buffer_store(cnt, rsrc_bhist, fx.Int32(bid) * segs + s)
                if const_expr(stage_send):
                    # This block's own counts, for its send-list staging
                    # (lds_hist is overwritten by the merge / the bases).
                    _st(lds_bcnt, s, cnt)
            comm_ops.waitcnt_stores()
            fx.barrier()
            if tid == 0:
                comm_ops.fence_agent_release()
                comm_ops.atomic_add_system(addr_barrier, arith.constant(1))
                if bid == 0:
                    comm_ops.spin_until_eq_i32(
                        addr_barrier, arith.constant(plan_blocks)
                    )
                    comm_ops.fence_agent_acquire()
                    buffer_store(arith.constant(0), rsrc_bar, 0)
            fx.barrier()
            if const_expr(stage_send):  # noqa: SIM102
                # Not on block 0's path to X1: the other blocks stage their
                # send-list entries while block 0 merges and exchanges.
                if bid != 0:
                    _stage_segments()

        if bid == 0:
            if const_expr(plan_blocks > 1):
                # Turn each block's count into its exclusive offset within the
                # segment; block 0's is always 0, so its row is left as is.
                for s in range(tid, segs, plan_threads):
                    total = arith.constant(0)
                    for blk in range_constexpr(plan_blocks):
                        cnt = buffer_load(
                            rsrc_bhist, blk * segs + s, vec_width=1, dtype=T.i32
                        )
                        if const_expr(blk > 0):
                            buffer_store(total, rsrc_bhist, blk * segs + s)
                        total = total + cnt
                    comm_ops.store_i32_lds(lds_hist + fx.Int64(s) * fx.Int64(4), total)
                fx.barrier()

            if tid == 0:
                buffer_store(gen, rsrc_bar, 2)
            if const_expr(hist_pingpong):
                parity = gen & arith.constant(1)
                hist_off = off_hist + parity * hist_stride * 4
            else:
                hist_off = off_hist
            done_off = off_done
            if const_expr(use_tag):
                tag = (gen & arith.constant(0xFFFF)) << arith.constant(16)

            if const_expr(use_sparse):
                for s in range(tid, row_dwords, plan_threads):
                    comm_ops.store_i32_lds(
                        lds_pack + fx.Int64(s) * fx.Int64(4), arith.constant(0)
                    )
                _lds_barrier()
                for s in range(tid, segs, plan_threads):
                    cnt = comm_ops.load_i32_lds(lds_hist + fx.Int64(s) * fx.Int64(4))
                    if cnt != 0:
                        slot = comm_ops.atomic_add_lds(lds_pack, arith.constant(1))
                        packed = fx.Int32(s) | (cnt << arith.constant(16))
                        comm_ops.store_i32_lds(
                            lds_pack + fx.Int64(slot + 1) * fx.Int64(4), packed
                        )
                _lds_barrier()
                tdm_rows = row_dwords // 32
                if warp < npes:
                    peer_hist = lsa(warp, hist_off) + fx.Int64(rank * row_dwords * 4)
                    TDM.tdm_store(
                        TDM.tdm_group0(
                            arith.trunci(T.i32, arith.unwrap(lds_pack)), peer_hist
                        ),
                        TDM.tdm_group1(32, tdm_rows, 4),
                    )
                    s_nop(0)
                fx.barrier()
                TDM.tdm_wait(0)
            elif const_expr(row_dwords % 32 == 0):
                if const_expr(use_tag):
                    # Pad slots carry the tag too, or the reader spins forever.
                    for s in range(tid, row_dwords, plan_threads):
                        slot_addr = lds_hist + fx.Int64(s) * fx.Int64(4)
                        comm_ops.store_i32_lds(
                            slot_addr, comm_ops.load_i32_lds(slot_addr) | tag
                        )
                    _lds_barrier()
                if warp < npes:
                    peer_hist = lsa(warp, hist_off) + fx.Int64(rank * row_dwords * 4)
                    TDM.tdm_store(
                        TDM.tdm_group0(
                            arith.trunci(T.i32, arith.unwrap(lds_hist)), peer_hist
                        ),
                        TDM.tdm_group1(32, row_dwords // 32, 4),
                    )
                    s_nop(0)
                if const_expr(not use_tag):
                    fx.barrier()
                    TDM.tdm_wait(0)
            else:
                hist_vec = 2 if segs % 2 == 0 else 1
                for peer in range_constexpr(npes):
                    peer_hist = lsa(peer, hist_off) + fx.Int64(rank * segs * 4)
                    peer_hist_rsrc = create_buffer_resource_from_addr(peer_hist)
                    for s in range(
                        tid * hist_vec,
                        segs,
                        plan_threads * hist_vec,
                    ):
                        vals = [
                            comm_ops.load_i32_lds(
                                lds_hist + fx.Int64(s + i) * fx.Int64(4)
                            )
                            for i in range_constexpr(hist_vec)
                        ]
                        buffer_store(
                            fx.Vector.from_elements(vals, dtype=fx.Int32),
                            peer_hist_rsrc,
                            s,
                        )
                comm_ops.waitcnt_stores()

            if const_expr(emit_runs):
                # While the rows are in flight: this rank's send offsets (run
                # order, its own counts only) and, staged in LDS, the send
                # list (every valid route at sendpos[seg] + intra; slots of
                # rows past compact_cap carry routes no run references).
                # lds_hist is read only (the TDM store may still read it).
                _wave_positions(
                    _seg_of_pos,
                    lambda seg_q: _ld(lds_hist, seg_q) & fx.Int32(cmask),
                    _soff_emit,
                    _soff_total,
                )
                if const_expr(early_release):
                    # The other blocks need only these offsets (and their own
                    # block offsets, written by the merge above).
                    comm_ops.waitcnt_stores()
                    fx.barrier()
                    if tid == 0:
                        # A release store, not fence + plain store: a plain
                        # store stays in this XCC's L2 until something writes
                        # it back (measured: the other blocks saw it only at
                        # this block's exit, ~7 us late).
                        comm_ops.store_i32_global_agent_release(
                            addr_barrier + fx.Int64(12), gen
                        )
                else:
                    _lds_barrier()
                if const_expr(stage_send):
                    if const_expr(plan_blocks == 1):
                        _stage_routes(lds_sendpos)
                        _lds_barrier()
                        n_send = fx.Int32(
                            readfirstlane(T.i32, arith.unwrap(_ld(lds_misc, 0)))
                        )
                        for q in range(tid, n_send, plan_threads):
                            buffer_store(_ld(lds_send, q), rsrc_send, q)
                    else:
                        _stage_segments()
                        _lds_barrier()
                        _copy_segments()

            if const_expr(not use_tag):
                fx.barrier()
                if tid == 0:
                    comm_ops.fence_system_release()
                fx.barrier()
                if tid < npes:
                    comm_ops.atomic_add_system(
                        lsa(tid, done_off),
                        arith.constant(1),
                    )
                comm_ops.waitcnt_stores()
                fx.barrier()
                if tid == 0:
                    comm_ops.wait_i32_until_equals(
                        lsa(rank, done_off),
                        gen * fx.Int32(npes),
                    )
                    comm_ops.fence_system_acquire()
                fx.barrier()

            if const_expr(use_sparse):
                tdm_rows = (npes * row_dwords) // 32
                if warp == 0:
                    TDM.tdm_load(
                        TDM.tdm_group0(
                            arith.trunci(T.i32, arith.unwrap(lds_recv)),
                            lsa(my_lsa_rank, hist_off),
                        ),
                        TDM.tdm_group1(32, tdm_rows, 4),
                    )
                    s_nop(0)
                    TDM.tdm_wait(0)
                for s in range(tid, npes * hist_dwords, plan_threads):
                    comm_ops.store_i32_lds(
                        lds_matrix + fx.Int64(s) * fx.Int64(4), arith.constant(0)
                    )
                _lds_barrier()
                for src in range_constexpr(npes):
                    src_base = fx.Int32(src * row_dwords)
                    nnz = comm_ops.load_i32_lds(
                        lds_recv + fx.Int64(src * row_dwords) * fx.Int64(4)
                    )
                    for i in range(tid, nnz, plan_threads):
                        packed = comm_ops.load_i32_lds(
                            lds_recv + fx.Int64(src_base + i + 1) * fx.Int64(4)
                        )
                        seg = packed & arith.constant(0xFFFF)
                        cnt = packed >> arith.constant(16)
                        comm_ops.store_i32_lds(
                            lds_matrix
                            + fx.Int64(src * hist_dwords) * fx.Int64(4)
                            + fx.Int64(seg) * fx.Int64(4),
                            cnt,
                        )
                _lds_barrier()
            else:
                matrix_n = npes * row_dwords
                if const_expr(use_tag):
                    # Every dword carries its writer's generation, so a slot is
                    # final once its tag matches: no separate arrival counter.
                    # 16 B per poll (row_dwords % 32 == 0): one round trip per
                    # thread for every row at EP <= 4 x 1K threads.
                    local_hist = lsa(my_lsa_rank, hist_off)
                    for q in range(tid, matrix_n // 4, plan_threads):
                        w4 = _spin_until_tag4(
                            local_hist + fx.Int64(q) * fx.Int64(16), tag
                        )
                        for k in range_constexpr(4):
                            comm_ops.store_i32_lds(
                                lds_matrix + fx.Int64(q * 4 + k) * fx.Int64(4),
                                w4[k] & arith.constant(0xFFFF),
                            )
                    # The scan below overwrites lds_hist, which the stores read.
                    if warp < npes:
                        TDM.tdm_wait(0)
                elif const_expr(matrix_n % 32 == 0):
                    if warp == 0:
                        TDM.tdm_load(
                            TDM.tdm_group0(
                                arith.trunci(T.i32, arith.unwrap(lds_matrix)),
                                lsa(my_lsa_rank, hist_off),
                            ),
                            TDM.tdm_group1(32, matrix_n // 32, 4),
                        )
                        s_nop(0)
                        TDM.tdm_wait(0)
                else:
                    local_hist_rsrc = create_buffer_resource_from_addr(
                        lsa(my_lsa_rank, hist_off)
                    )
                    for s in range(tid, matrix_n, plan_threads):
                        comm_ops.store_i32_lds(
                            lds_matrix + fx.Int64(s) * fx.Int64(4),
                            buffer_load(local_hist_rsrc, s, vec_width=1, dtype=T.i32),
                        )
                _lds_barrier()
            for idx in range(tid, segs, plan_threads):
                dest = idx // fx.Int32(epr)
                e = idx - dest * fx.Int32(epr)
                total = arith.constant(0)
                my_prefix = arith.constant(0)
                for src in range_constexpr(npes):
                    cnt = comm_ops.load_i32_lds(
                        lds_matrix
                        + fx.Int64(src * hist_dwords + dest * fx.Int32(epr) + e)
                        * fx.Int64(4)
                    )
                    if src == rank:
                        my_prefix = total
                    total = total + cnt
                comm_ops.store_i32_lds(lds_total + fx.Int64(idx) * fx.Int64(4), total)
                comm_ops.store_i32_lds(
                    lds_pref + fx.Int64(idx) * fx.Int64(4), my_prefix
                )
                if dest == rank:
                    buffer_store(total, rsrc_mm, e)
            _lds_barrier()

            def _scan_chunk(dest, chunk, carry):
                """Experts [chunk, chunk + 32) of dest: tile-aligned expert
                starts (carry = dest's aligned rows before chunk), my send base
                per segment, psum (dest == rank), and the S1F tile_info inputs.
                Returns the carry past the chunk."""
                e = fx.Int32(chunk) + lane_i
                in_expert = e < fx.Int32(epr)
                safe_e = fx.Int32(in_expert.select(e, fx.Int32(0)))
                idx = dest * fx.Int32(epr) + safe_e
                total = fx.Int32(0)
                my_prefix = fx.Int32(0)
                if in_expert:
                    total = _ld(lds_total, idx)
                    my_prefix = _ld(lds_pref, idx)
                aligned = _aceil(total)
                inclusive = _wave32_inclusive_scan_i32(aligned, lane_i)
                expert_start = carry + inclusive - aligned
                if in_expert:
                    send = expert_start + my_prefix
                    if dest == fx.Int32(rank):
                        buffer_store(expert_start + total, rsrc_psum, e)
                    if const_expr(plan_blocks > 1):
                        # Only the other blocks read it (after the release).
                        buffer_store(send, rsrc_base, idx)
                    _st(lds_hist, idx, send)
                    if const_expr(emit_runs):
                        # My rows of this segment inside compact_cap and their
                        # runs, stored by run-order position (le outer, dest
                        # rotated by rank inner) for the run emission.
                        c_me = _imin(
                            _ld(lds_matrix, fx.Int32(rank * hist_dwords) + idx),
                            _imax(fx.Int32(compact_cap) - send, fx.Int32(0)),
                        )
                        j_d = dest - fx.Int32(rank)
                        j_d = fx.Int32(
                            (j_d < fx.Int32(0)).select(j_d + fx.Int32(npes), j_d)
                        )
                        p_me = safe_e * fx.Int32(npes) + j_d
                        _st(lds_ccl, p_me, c_me)
                        _st(lds_rcnt, p_me, _run_count(send, c_me))
                    if const_expr(s1f):  # noqa: SIM102
                        if dest == fx.Int32(rank):
                            # tile_info inputs: the m-tile -> expert map
                            # (starts are tile-aligned) and the expert's
                            # source boundaries (rows are source-major).
                            t0 = _tdiv(expert_start)
                            t1 = _imin(t0 + _tdiv(aligned), fx.Int32(tile_cap))
                            for t in range(t0, t1):
                                _st(lds_t2e, t, safe_e)
                            bacc = expert_start
                            for src in range_constexpr(npes):
                                _st(
                                    lds_bnd,
                                    safe_e * fx.Int32(bnd_n) + fx.Int32(src),
                                    bacc,
                                )
                                bacc = bacc + _ld(
                                    lds_matrix, fx.Int32(src * hist_dwords) + idx
                                )
                            _st(
                                lds_bnd, safe_e * fx.Int32(bnd_n) + fx.Int32(npes), bacc
                            )
                carry = carry + fx.Int32(
                    readlane(T.i32, arith.unwrap(inclusive), WAVE - 1)
                )
                if const_expr(s1f):
                    last = fx.Int32(chunk) + fx.Int32(WAVE) >= fx.Int32(epr)
                    if (dest == fx.Int32(rank)) & last & (lane_i == fx.Int32(0)):
                        # m-tiles of this rank (all its rows, aligned).
                        _st(lds_misc, 1, _imin(_tdiv(carry), fx.Int32(tile_cap)))
                return carry

            n_ech = -(-epr // WAVE)
            if const_expr(npes * n_ech <= plan_waves):
                # One wave per (dest, 32-expert chunk): the aligned rows of
                # the dest's earlier chunks are summed by the wave itself, so
                # the scan is one chunk deep instead of n_ech in series.
                if warp_u < fx.Int32(npes * n_ech):
                    dest_w = warp_u // fx.Int32(n_ech)
                    ch_w = warp_u - dest_w * fx.Int32(n_ech)
                    pre = fx.Int32(0)
                    for k in range_constexpr(n_ech - 1):
                        e_k = fx.Int32(k * WAVE) + lane_i
                        live_k = (fx.Int32(k) < ch_w) & (e_k < fx.Int32(epr))
                        e_ks = fx.Int32((e_k < fx.Int32(epr)).select(e_k, fx.Int32(0)))
                        t_k = _ld(lds_total, dest_w * fx.Int32(epr) + e_ks)
                        a_k = _aceil(t_k)
                        pre = pre + fx.Int32(live_k.select(a_k, fx.Int32(0)))
                    _scan_chunk(dest_w, ch_w * fx.Int32(WAVE), _wsum(pre))
            else:
                if warp < npes:
                    dest = fx.Int32(warp_u)
                    carry = fx.Int32(0)
                    for chunk in range(0, epr, WAVE):
                        carry = _scan_chunk(dest, chunk, carry)
            _lds_barrier()

        if const_expr(plan_blocks > 1):
            if const_expr(early_release):
                # S1F without tok_map: released before X1 (barrier[3]).
                if bid != 0:
                    if tid == 0:
                        comm_ops.spin_until_eq_i32(addr_barrier + fx.Int64(12), gen)
                        comm_ops.fence_agent_acquire()
                    fx.barrier()
            else:
                # barrier[1] releases this generation's send_base and offsets
                # (emit_runs: and the send offsets).
                comm_ops.waitcnt_stores()
                fx.barrier()
                if tid == 0:
                    if bid == 0:
                        # Release store (see barrier[3] above): with fence +
                        # plain store the other blocks saw it ~2.5 us late.
                        comm_ops.store_i32_global_agent_release(
                            addr_barrier + fx.Int64(4), gen
                        )
                    else:
                        comm_ops.spin_until_eq_i32(addr_barrier + fx.Int64(4), gen)
                        comm_ops.fence_agent_acquire()
                fx.barrier()
            if bid != 0:
                for s in range(tid, segs, plan_threads):
                    off = buffer_load(
                        rsrc_bhist,
                        fx.Int32(bid) * segs + s,
                        vec_width=1,
                        dtype=T.i32,
                    )
                    if const_expr(write_tok_map):
                        base = buffer_load(rsrc_base, s, vec_width=1, dtype=T.i32)
                        comm_ops.store_i32_lds(
                            lds_hist + fx.Int64(s) * fx.Int64(4), base + off
                        )
                    if const_expr(emit_runs):
                        comm_ops.store_i32_lds(
                            _lds_at(lds_sendpos, s),
                            buffer_load(
                                rsrc_runs,
                                RUN_HDR_SEND_OFF + s,
                                vec_width=1,
                                dtype=T.i32,
                            )
                            + off,
                        )
                _lds_barrier()
                if const_expr(stage_send):
                    _copy_segments()

        if const_expr(emit_runs):  # noqa: SIM102
            if bid == 0:
                # (a) per 32-position chunk (run order): each position's first
                #     run within the chunk and the chunk's runs; the other
                #     waves (S1F) build tile_info meanwhile.
                # (b) one LDS barrier, then one run per thread: its chunk from
                #     the chunk totals, its position by a 5-step binary search
                #     in the chunk; balanced whatever the routing skew (one
                #     local shared-expert segment can hold 500+ runs).
                for ch_a in range(warp_u, fx.Int32(n_chunks), fx.Int32(plan_waves)):
                    p_a = ch_a * fx.Int32(WAVE) + lane_i
                    live_a = p_a < fx.Int32(segs)
                    nr_a = fx.Int32(
                        live_a.select(
                            _ld(lds_rcnt, _imin(p_a, fx.Int32(segs - 1))), fx.Int32(0)
                        )
                    )
                    inc_a = _wave32_inclusive_scan_i32(nr_a, lane_i)
                    if live_a:
                        _st(lds_rexcl, p_a, inc_a - nr_a)
                    tot_a = fx.Int32(readlane(T.i32, arith.unwrap(inc_a), WAVE - 1))
                    if lane_i == fx.Int32(0):
                        _st(lds_ctot, ch_a, tot_a)
                if const_expr(s1f):
                    # tile_info[t] = srcmask | expert << 8 | rows << 16, one
                    # m-tile per thread (coalesced): srcmask = the sources with
                    # rows in it, rows its valid rows (clipped to compact_cap).
                    t_w0 = n_chunks if plan_waves > n_chunks else 0
                    if warp_u >= fx.Int32(t_w0):
                        m_tiles = fx.Int32(
                            readfirstlane(T.i32, arith.unwrap(_ld(lds_misc, 1)))
                        )
                        t_thr = (plan_waves - t_w0) * WAVE
                        for t in range(
                            fx.Int32(tid) - fx.Int32(t_w0 * WAVE),
                            m_tiles,
                            fx.Int32(t_thr),
                        ):
                            e_t = _ld(lds_t2e, t)
                            bnd = [
                                _ld(lds_bnd, e_t * fx.Int32(bnd_n) + fx.Int32(s_))
                                for s_ in range_constexpr(bnd_n)
                            ]
                            lo = fx.Int32(t) * fx.Int32(tile_m)
                            hi = _imin(
                                lo + fx.Int32(tile_m),
                                _imin(bnd[npes], fx.Int32(compact_cap)),
                            )
                            mask = fx.Int32(0)
                            for src in range_constexpr(npes):
                                mask = mask | fx.Int32(
                                    (
                                        _imax(bnd[src], lo) < _imin(bnd[src + 1], hi)
                                    ).select(fx.Int32(1 << src), fx.Int32(0))
                                )
                            buffer_store(
                                mask
                                | (e_t << fx.Int32(8))
                                | (_imax(hi - lo, fx.Int32(0)) << fx.Int32(16)),
                                rsrc_tsrc,
                                t,
                            )
                _lds_barrier()
                # Chunk first runs from uniform LDS loads, not a wave scan +
                # readlane: the run loop below runs under a partial exec, and
                # the compiler may sink a lane-wise value into it, leaving the
                # lanes a readlane reads stale (measured: garbage runs).
                c_first = [fx.Int32(0)]
                for c_ in range_constexpr(n_chunks):
                    c_first.append(c_first[-1] + _ld(lds_ctot, fx.Int32(c_)))
                n_runs = fx.Int32(readfirstlane(T.i32, arith.unwrap(c_first[n_chunks])))
                if (warp_u == fx.Int32(0)) & (lane_i == fx.Int32(0)):
                    buffer_store(
                        _imin(n_runs, fx.Int32(run_cap)), rsrc_runs, RUN_HDR_RUNS
                    )
                for r in range(
                    fx.Int32(tid),
                    _imin(n_runs, fx.Int32(run_cap)),
                    fx.Int32(plan_threads),
                ):
                    # Chunk: the last one whose first run is <= r.
                    c_r = fx.Int32(0)
                    c_base = fx.Int32(0)
                    for c_ in range_constexpr(1, n_chunks):
                        take = c_first[c_] <= r
                        c_r = fx.Int32(take.select(fx.Int32(c_), c_r))
                        c_base = fx.Int32(take.select(c_first[c_], c_base))
                    rl = r - c_base
                    # Position in the chunk: the last whose first run is <= rl
                    # (a position without runs never is: the next one has the
                    # same first run).
                    p0 = c_r * fx.Int32(WAVE)
                    lo_p = fx.Int32(0)
                    lo_v = fx.Int32(0)
                    for step in (16, 8, 4, 2, 1):
                        cand = lo_p + fx.Int32(step)
                        pc = p0 + cand
                        v_c = _ld(lds_rexcl, _imin(pc, fx.Int32(segs - 1)))
                        ok = (pc < fx.Int32(segs)) & (v_c <= rl)
                        lo_p = fx.Int32(ok.select(cand, lo_p))
                        lo_v = fx.Int32(ok.select(v_c, lo_v))
                    p = p0 + lo_p
                    k = rl - lo_v
                    seg, d = _run_seg(p)
                    b = _ld(lds_hist, seg)
                    c = _ld(lds_ccl, p)
                    soff = _ld(lds_sendpos, seg)
                    f = _first_part(b, c)
                    nf = _rceil(f)
                    row_end = b + c
                    k2 = k - nf
                    if const_expr(per_tile & (per_tile - 1) == 0):
                        q = k2 >> fx.Int32(per_tile.bit_length() - 1)
                    else:
                        q = k2 // fx.Int32(per_tile)
                    kk = k2 - q * fx.Int32(per_tile)
                    row = fx.Int32(
                        (k < nf).select(
                            b + k * fx.Int32(run_rows),
                            b + f + q * fx.Int32(tile_m) + kk * fx.Int32(run_rows),
                        )
                    )
                    tile_end = row - _tmod(row) + fx.Int32(tile_m)
                    n = _imin(_imin(fx.Int32(run_rows), tile_end - row), row_end - row)
                    buffer_store(
                        fx.Vector.from_elements(
                            [
                                (row << fx.Int32(peer_bits)) | d,
                                ((soff + row - b) << fx.Int32(4)) | n,
                            ],
                            dtype=fx.Int32,
                        ),
                        rsrc_runs,
                        fx.Int32(run_hdr) + r * fx.Int32(2),
                    )
                    if const_expr(s1f):
                        # Rows of this segment inside the run's tile.
                        buffer_store(
                            _imin(row_end, tile_end)
                            - _imax(b, tile_end - fx.Int32(tile_m)),
                            rsrc_runs,
                            fx.Int32(run_hdr + 2 * run_cap) + r,
                        )

        if const_expr(write_tok_map or not stage_send):
            for local in range(tid, r_n, plan_threads):
                route = r0 + local
                if const_expr(merge_routes):
                    packed = comm_ops.load_i32_lds(
                        lds_routes + fx.Int64(local) * fx.Int64(4)
                    )
                else:
                    packed = buffer_load(rsrc_map, route, vec_width=1, dtype=T.i32)
                valid = packed >= 0
                segment = packed & arith.constant(0xFFFF)
                intra = packed >> arith.constant(16)
                dest_pe = segment // fx.Int32(epr)
                send_base = comm_ops.load_i32_lds(
                    lds_hist + fx.Int64(segment) * fx.Int64(4)
                )
                dest_row = send_base + intra
                in_cap = dest_row < compact_cap
                if const_expr(write_tok_map):
                    flat = (dest_row << fx.Int32(peer_bits)) | (
                        dest_pe & fx.Int32(peer_mask)
                    )
                    buffer_store(
                        arith.select(valid & in_cap, flat, arith.constant(dropped)),
                        rsrc_map,
                        route,
                    )
                if const_expr(emit_runs and not stage_send):  # noqa: SIM102
                    # Slot = this block's first slot of the segment + intra.
                    if valid & in_cap:
                        buffer_store(
                            route,
                            rsrc_send,
                            comm_ops.load_i32_lds(_lds_at(lds_sendpos, segment))
                            + intra,
                        )

    if s1f:

        @flyc.kernel(name=kernel_name, known_block_size=[plan_threads, 1, 1])
        def kernel(
            arena: Int64,
            addr_inp_idx: Int64,
            addr_tok_map: Int64,
            addr_block_hist: Int64,
            addr_send_base: Int64,
            addr_masked_m: Int64,
            addr_psum: Int64,
            addr_barrier: Int64,
            my_lsa_rank: Int32,
            inp_cur_tok: Int32,
            addr_send_list: Int64,
            addr_runs: Int64,
            addr_tile_src: Int64,
        ):
            plan_body(
                arena,
                addr_inp_idx,
                addr_tok_map,
                addr_block_hist,
                addr_send_base,
                addr_masked_m,
                addr_psum,
                addr_barrier,
                my_lsa_rank,
                inp_cur_tok,
                addr_send_list,
                addr_runs,
                addr_tile_src,
            )

        @flyc.jit
        def launch(
            arena: Int64,
            addr_inp_idx: Int64,
            addr_tok_map: Int64,
            addr_block_hist: Int64,
            addr_send_base: Int64,
            addr_masked_m: Int64,
            addr_psum: Int64,
            addr_barrier: Int64,
            my_lsa_rank: Int32,
            inp_cur_tok: Int32,
            addr_send_list: Int64,
            addr_runs: Int64,
            addr_tile_src: Int64,
            stream=fx.Stream(None),  # noqa: B008
        ):
            kernel(
                arena,
                addr_inp_idx,
                addr_tok_map,
                addr_block_hist,
                addr_send_base,
                addr_masked_m,
                addr_psum,
                addr_barrier,
                my_lsa_rank,
                inp_cur_tok,
                addr_send_list,
                addr_runs,
                addr_tile_src,
            ).launch(
                grid=(plan_blocks, 1, 1),
                block=[plan_threads, 1, 1],
                stream=stream,
            )

        return launch

    if emit_runs:

        @flyc.kernel(name=kernel_name, known_block_size=[plan_threads, 1, 1])
        def kernel(
            arena: Int64,
            addr_inp_idx: Int64,
            addr_tok_map: Int64,
            addr_block_hist: Int64,
            addr_send_base: Int64,
            addr_masked_m: Int64,
            addr_psum: Int64,
            addr_barrier: Int64,
            my_lsa_rank: Int32,
            inp_cur_tok: Int32,
            addr_send_list: Int64,
            addr_runs: Int64,
        ):
            plan_body(
                arena,
                addr_inp_idx,
                addr_tok_map,
                addr_block_hist,
                addr_send_base,
                addr_masked_m,
                addr_psum,
                addr_barrier,
                my_lsa_rank,
                inp_cur_tok,
                addr_send_list,
                addr_runs,
                None,
            )

        @flyc.jit
        def launch(
            arena: Int64,
            addr_inp_idx: Int64,
            addr_tok_map: Int64,
            addr_block_hist: Int64,
            addr_send_base: Int64,
            addr_masked_m: Int64,
            addr_psum: Int64,
            addr_barrier: Int64,
            my_lsa_rank: Int32,
            inp_cur_tok: Int32,
            addr_send_list: Int64,
            addr_runs: Int64,
            stream=fx.Stream(None),  # noqa: B008
        ):
            kernel(
                arena,
                addr_inp_idx,
                addr_tok_map,
                addr_block_hist,
                addr_send_base,
                addr_masked_m,
                addr_psum,
                addr_barrier,
                my_lsa_rank,
                inp_cur_tok,
                addr_send_list,
                addr_runs,
            ).launch(
                grid=(plan_blocks, 1, 1),
                block=[plan_threads, 1, 1],
                stream=stream,
            )

        return launch

    @flyc.kernel(name=kernel_name, known_block_size=[plan_threads, 1, 1])
    def kernel(
        arena: Int64,
        addr_inp_idx: Int64,
        addr_tok_map: Int64,
        addr_block_hist: Int64,
        addr_send_base: Int64,
        addr_masked_m: Int64,
        addr_psum: Int64,
        addr_barrier: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
    ):
        plan_body(
            arena,
            addr_inp_idx,
            addr_tok_map,
            addr_block_hist,
            addr_send_base,
            addr_masked_m,
            addr_psum,
            addr_barrier,
            my_lsa_rank,
            inp_cur_tok,
            None,
            None,
            None,
        )

    @flyc.jit
    def launch(
        arena: Int64,
        addr_inp_idx: Int64,
        addr_tok_map: Int64,
        addr_block_hist: Int64,
        addr_send_base: Int64,
        addr_masked_m: Int64,
        addr_psum: Int64,
        addr_barrier: Int64,
        my_lsa_rank: Int32,
        inp_cur_tok: Int32,
        stream=fx.Stream(None),  # noqa: B008
    ):
        kernel(
            arena,
            addr_inp_idx,
            addr_tok_map,
            addr_block_hist,
            addr_send_base,
            addr_masked_m,
            addr_psum,
            addr_barrier,
            my_lsa_rank,
            inp_cur_tok,
        ).launch(
            grid=(plan_blocks, 1, 1),
            block=[plan_threads, 1, 1],
            stream=stream,
        )

    return launch
