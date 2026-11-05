# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""S2F: gemm2 (LL128 gather scatter) and the combine reduce as ONE persistent
kernel (gfx1250).

Replaces gemm2 + ``ep_combine_fused_sync`` + ``ep_combine_fused`` of the
compact bf16 combine. Grid: ``workers`` CTAs of gemm2's 4 waves.

* GEMM2. Every CTA claims (m-tile, n-tile) items off one agent atomic head and
  runs the gemm2 tile body on each; the epilogue scatters the rows to their
  source ranks in the LL128 layout, every 128 B line tagged with the plan's
  generation (``psum[n_experts]``).
* REDUCE. Once the gemm2 head is past the last item, the CTA claims reduce
  items (``tokens`` tokens x one 256-column segment) off a second head: it
  loads the tile's ``tokens * topk`` slot segments SCOPE_SYS, waits on a stale
  line and reloads while any line of a live route still carries an older tag,
  sums the topk rows and stores the bf16 result.

Forward progress: a CTA only waits once every gemm2 item of its rank has been
claimed by a running CTA, and gemm2 items wait on nothing, so the grid need not
be resident at once; a peer's reduce waits only on this rank's gemm2. The next
dispatch cannot overwrite what this kernel still reads: its compact plan
exchanges histograms with every rank, and a rank only writes its row once its
own stream has run this kernel to the end.

The last CTA to finish (``ctl[CTL_DONE]``) zeroes the heads for the next
launch.

Nothing after the gemm2 phase branches on the lane: one lane's atomics address
out of bounds for the others, per-lane results meet in LDS and the waits run on
a whole wave. A lane-0 ``if`` ahead of the reduce left the rest of the CTA on
lane 0 alone in an earlier build.
"""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, range_constexpr, rocdl
from flydsl.expr.rocdl import ballot, readfirstlane, readlane
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels import communication_ops_utils as comm_ops
from aiter.ops.flydsl.kernels.buffer_ops import (
    buffer_load,
    buffer_store,
    create_buffer_resource_from_addr,
)
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.mxfp4_preshuffle_gfx1250_tdm import (
    build_gemm_a8w4_tdm_task,
    launch_gemm_a8w4_tdm,
)

from . import ll128 as LL128
from . import tdm_prims as TDM
from .combine import (
    LL_RELOAD_CAP,
    LL_SCOPE_SYS,
    LL_TILE_RETRIES,
    _trap,
    ll_poll_sleep,
)
from .config import _WAVE_SIZE as WAVE
from .s1f_stage1 import _buffer_atomic_add_ret, _ctz

#: Tokens of one reduce item: with 16 columns a lane, 8 tokens of a 256-column
#: segment are exactly one 4-wave CTA.
S2F_TOKENS = 8
#: Control block (i32 words): the gemm2 head, the reduce head and the done
#: counter, each on its own 128 B line.
CTL_G2 = 0
CTL_RED = 32
CTL_DONE = 64
CTL_WORDS = 96
#: gfx12 CPol SCOPE_DEV for the head atomics.
_SCOPE_DEV = 2 << 3
#: Byte offset past any resource: the lane issues no memory access.
_OOB = 0x7FFFFFF0
#: A gfx1250 CU's LDS, and the granule a workgroup's allocation rounds up to.
_LDS_PER_CU = 320 * 1024
_LDS_GRANULE = 2048


def _uniform(v):
    return fx.Int32(readfirstlane(T.i32, arith.unwrap(fx.Int32(v))))


def _rawbar():
    """Workgroup barrier ordering LDS only (no loadcnt/storecnt drain)."""
    fx.rocdl.s_wait_dscnt(0)
    rocdl.s_barrier_signal(-1)
    rocdl.s_barrier_wait(-1)


def make_s2f_stage2(
    *,
    gemm,
    topk,
    hidden_dim,
    slot_stride,
    n_experts_total,
    workers=None,
    tokens=S2F_TOKENS,
):
    """Build the S2F kernel; returns a ``@flyc.jit`` launcher.

    ``gemm``: ``build_gemm_a8w4_tdm_task`` keywords of gemm2 with its bf16
    gather scatter (``enable_ep_scatter=1`` and the arena geometry); the
    persistent, explicit-tile and LL128 knobs are set here.

    Launcher arguments (positional): ``arg_c`` (tensor), ``arg_a``, ``arg_b``
    (pointers), ``arg_scale_a``, ``arg_scale_b`` (tensors), ``arg_psum``,
    ``arg_bias`` (pointers), ``arg_rowmap`` (tensor), then i64 addresses
    ``comb_inp`` (this rank's), ``out``, ``topk_ids``, ``psum``, ``ctl``, then
    ``i32_m`` (gemm2 rows capacity), ``cur_tok``, ``stream``.

    ``workers`` defaults to the CTAs every CU holds at once.
    """
    TT = int(tokens)
    BLOCK = 4 * WAVE
    SEGS = hidden_dim // LL128.SEG_COLS
    ROWS = TT * topk
    SEG_Q = LL128.SEG_B // 16
    TAGS = ROWS * LL128.SEG_LINES
    LANES_PER_TOK = LL128.SEG_COLS // 16
    IN_B = ROWS * LL128.SEG_B
    OUT_ROW_B = LL128.SEG_COLS * 2
    OUT_OFF = IN_B
    VROW_OFF = OUT_OFF + TT * OUT_ROW_B
    STALES_OFF = VROW_OFF + BLOCK * 4
    ZERO_OFF = STALES_OFF + BLOCK * 4
    STALE_OFF = ZERO_OFF + 32
    RED_LDS = STALE_OFF + 16
    if hidden_dim % LL128.SEG_COLS:
        raise ValueError(f"hidden_dim {hidden_dim} is not whole LL128 segments")
    if TT * LANES_PER_TOK != BLOCK or ROWS > BLOCK:
        raise ValueError(
            f"a reduce item of {TT} tokens x topk {topk} does not fit a CTA"
        )
    if (ROWS * SEG_Q) % BLOCK:
        raise ValueError(f"{ROWS} rows x {SEG_Q} granules do not split over {BLOCK}")
    if slot_stride % 16 or ROWS * slot_stride >= _OOB:
        raise ValueError(f"slot stride {slot_stride} B")
    tile_m = int(gemm["tile_m"])
    tile_n = int(gemm["tile_n"])
    n_experts = int(gemm["n_experts"])
    if hidden_dim % tile_n:
        raise ValueError(f"tile_n {tile_n} does not divide hidden_dim {hidden_dim}")
    NT = hidden_dim // tile_n
    if workers is None:
        # As many CTAs as the device holds at once: LDS-bound, so it follows
        # gemm2's tile and buffer count (and the reduce's tile, if larger).
        from aiter.jit.utils.chip_info import get_cu_num

        _, _, _, _, lay0 = build_gemm_a8w4_tdm_task(
            **gemm, persistent_workers=1, explicit_tile=1, ep_ll128=1
        )
        lds = max(int(lay0["arena"]), RED_LDS) + 64
        lds = -(-lds // _LDS_GRANULE) * _LDS_GRANULE
        workers = max(1, _LDS_PER_CU // lds) * int(get_cu_num())
    G = int(workers)

    @flyc.jit
    def launch(
        arg_c: fx.Tensor,
        arg_a: fx.Pointer,
        arg_b: fx.Pointer,
        arg_scale_a: fx.Tensor,
        arg_scale_b: fx.Tensor,
        arg_psum: fx.Pointer,
        arg_bias: fx.Pointer,
        arg_rowmap: fx.Tensor,
        addr_comb: fx.Int64,
        addr_out: fx.Int64,
        addr_ids: fx.Int64,
        addr_psum: fx.Int64,
        addr_ctl: fx.Int64,
        i32_m: fx.Int32,
        cur_tok: fx.Int32,
        stream: fx.Stream,
    ):
        lds_holder = [None]
        g2_task, block, g2_name, _, lay = build_gemm_a8w4_tdm_task(
            **gemm,
            persistent_workers=G,
            explicit_tile=1,
            lds_base=lds_holder,
            ep_ll128=1,
        )
        if block != BLOCK:
            raise ValueError(f"S2F expects gemm2's 4-wave block, got {block}")
        LDS = max(int(lay["arena"]), RED_LDS)
        kname = f"s2f_stage2_w{G}_t{TT}_{g2_name}"
        # Keep the separate-kernel launcher in this cache key's dependency set.
        _ = launch_gemm_a8w4_tdm

        @flyc.kernel(name=kname, known_block_size=[BLOCK, 1, 1])
        def kernel(
            arg_c: fx.Pointer,
            arg_a: fx.Pointer,
            arg_b: fx.Pointer,
            arg_scale_a: fx.Pointer,
            arg_scale_b: fx.Pointer,
            arg_psum: fx.Pointer,
            arg_bias: fx.Pointer,
            arg_rowmap: fx.Pointer,
            addr_comb: fx.Int64,
            addr_out: fx.Int64,
            addr_ids: fx.Int64,
            addr_psum: fx.Int64,
            addr_ctl: fx.Int64,
            i32_m: fx.Int32,
            cur_tok: fx.Int32,
        ):
            tid = fx.Int32(fx.thread_idx.x)
            lane = tid & fx.Int32(WAVE - 1)
            wave = _uniform(tid >> fx.Int32(5))
            lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)

            smem = fx.SharedAllocator(static=False)
            arena_ptr = smem.allocate(LDS, 128)._ptr
            lds_holder[0] = arena_ptr
            bc_ptr = smem.allocate(64, 16)._ptr
            base = fx.Int32(
                arith.index_cast(T.i32, fx.index_cast(T.index, fx.ptrtoint(arena_ptr)))
            )
            bc = fx.Int32(
                arith.index_cast(T.i32, fx.index_cast(T.index, fx.ptrtoint(bc_ptr)))
            )

            def st(addr_i32, v):
                comm_ops.store_i32_lds(fx.Int64(addr_i32), fx.Int32(v))

            def ld(addr_i32):
                return fx.Int32(comm_ops.load_i32_lds(fx.Int64(addr_i32)))

            rsrc_psum = create_buffer_resource_from_addr(addr_psum)
            rsrc_ctl = create_buffer_resource_from_addr(
                addr_ctl, num_records_bytes=CTL_WORDS * 4
            )
            rsrc_ids = create_buffer_resource_from_addr(addr_ids)
            # Wave 0 lane 0 does the CTA's atomics; every other lane addresses
            # out of bounds, so the atomic needs no lane branch.
            first = (wave == fx.Int32(0)) & (lane == fx.Int32(0))

            def head_add(word):
                """Old value of ``ctl[word]`` += 1, for the whole CTA."""
                c = _buffer_atomic_add_ret(
                    rsrc_ctl,
                    fx.Int32(first.select(fx.Int32(word * 4), fx.Int32(_OOB))),
                    1,
                    _SCOPE_DEV,
                )
                if wave == fx.Int32(0):
                    st(bc, readlane(T.i32, arith.unwrap(c), 0))
                _rawbar()
                return _uniform(ld(bc))

            gen = _uniform(
                buffer_load(rsrc_psum, fx.Int32(n_experts), vec_width=1, dtype=T.i32)
            )
            rows = _uniform(
                buffer_load(
                    rsrc_psum, fx.Int32(n_experts - 1), vec_width=1, dtype=T.i32
                )
            )

            # ================= gemm2 =================
            items = (rows + fx.Int32(tile_m - 1)) // fx.Int32(tile_m) * fx.Int32(NT)
            code = head_add(CTL_G2)
            while code < items:
                g2_task(
                    arg_c,
                    arg_a,
                    arg_b,
                    arg_scale_a,
                    arg_scale_b,
                    arg_psum,
                    arg_bias,
                    arg_c,
                    arg_rowmap,
                    i32_m,
                    fx.Int32(hidden_dim),
                    fx.Float32(7.0),
                    fx.Float32(1.0),
                    fx.Float32(1.0),
                    code // fx.Int32(NT),
                    code % fx.Int32(NT),
                )
                code = head_add(CTL_G2)

            # ================= reduce =================
            def load_tile(g_in, nrows):
                """The tile's slot segments into LDS, 16 B a lane per load,
                SCOPE_SYS so a reload sees what a peer has written since. Rows
                past ``nrows`` load out of bounds, i.e. zeros."""
                rsrc_in = create_buffer_resource_from_addr(
                    g_in, num_records_bytes=ROWS * slot_stride
                )
                for k in range_constexpr(ROWS * SEG_Q // BLOCK):
                    q = fx.Int32(k * BLOCK) + tid
                    row = q // fx.Int32(SEG_Q)
                    col = q - row * fx.Int32(SEG_Q)
                    v = buffer_load(
                        rsrc_in,
                        fx.Int32(
                            (row < nrows).select(
                                row * fx.Int32(slot_stride // 4) + col * fx.Int32(4),
                                fx.Int32(_OOB // 4),
                            )
                        ),
                        vec_width=4,
                        dtype=T.i32,
                        cache_modifier=LL_SCOPE_SYS,
                    )
                    lds_store_b128(
                        base + row * fx.Int32(LL128.SEG_B) + col * fx.Int32(16), 0, v
                    )
                _rawbar()

            def tile_stale():
                """0 if every line of a live route carries ``gen``, else one
                stale tag's index + 1 (uniform)."""
                worst = fx.Int32(0)
                for k in range_constexpr((TAGS + BLOCK - 1) // BLOCK):
                    t = fx.Int32(k * BLOCK) + tid
                    in_t = t < fx.Int32(TAGS)
                    ts = fx.Int32(in_t.select(t, fx.Int32(0)))
                    row = ts // fx.Int32(LL128.SEG_LINES)
                    line = ts - row * fx.Int32(LL128.SEG_LINES)
                    live = in_t & (
                        ld(base + fx.Int32(VROW_OFF) + row * fx.Int32(4)) != fx.Int32(0)
                    )
                    tag = ld(
                        base + row * fx.Int32(LL128.SEG_B) + LL128.tag_offset(line)
                    )
                    bad = live & (tag != gen)
                    worst = fx.Int32(
                        (bad & (t + fx.Int32(1) > worst)).select(t + fx.Int32(1), worst)
                    )
                st(base + fx.Int32(STALES_OFF) + tid * fx.Int32(4), worst)
                _rawbar()
                if wave == fx.Int32(0):
                    m = fx.Int32(0)
                    for w in range_constexpr(BLOCK // WAVE):
                        v = ld(
                            base
                            + fx.Int32(STALES_OFF + w * WAVE * 4)
                            + lane * fx.Int32(4)
                        )
                        m = fx.Int32((v > m).select(v, m))
                    hit = fx.Int32(ballot(T.i32, arith.unwrap(m != fx.Int32(0))))
                    src = fx.Int32((hit != fx.Int32(0)).select(_ctz(hit), fx.Int32(0)))
                    got = fx.Int32(readlane(T.i32, arith.unwrap(m), arith.unwrap(src)))
                    st(
                        base + fx.Int32(STALE_OFF),
                        fx.Int32((hit != fx.Int32(0)).select(got, fx.Int32(0))),
                    )
                _rawbar()
                return _uniform(ld(base + fx.Int32(STALE_OFF)))

            def wait_line(g_in, stale):
                """Wave 0 polls the one stale line's tag in memory (128 B a
                poll, not the tile) until it carries ``gen``."""
                if wave == fx.Int32(0):
                    t = stale - fx.Int32(1)
                    row = t // fx.Int32(LL128.SEG_LINES)
                    line = t - row * fx.Int32(LL128.SEG_LINES)
                    addr = (
                        g_in
                        + fx.Int64(row) * fx.Int64(slot_stride)
                        + fx.Int64(LL128.tag_offset(line))
                    )
                    v = _uniform(comm_ops.load_i32_global_system(addr))
                    n = fx.Int32(0)
                    while (v != gen) & (n < fx.Int32(LL_RELOAD_CAP)):
                        ll_poll_sleep(n)
                        v = _uniform(comm_ops.load_i32_global_system(addr))
                        n = n + fx.Int32(1)
                    if v != gen:
                        _trap()
                _rawbar()

            if wave == fx.Int32(0):
                st(
                    base + fx.Int32(ZERO_OFF) + (lane & fx.Int32(7)) * fx.Int32(4),
                    fx.Int32(0),
                )
            tok_l = tid // fx.Int32(LANES_PER_TOK)
            g0 = (tid - tok_l * fx.Int32(LANES_PER_TOK)) * fx.Int32(2)
            n_red = (cur_tok + fx.Int32(TT - 1)) // fx.Int32(TT) * fx.Int32(SEGS)
            r = head_add(CTL_RED)
            while r < n_red:
                grp = r // fx.Int32(SEGS)
                seg = r - grp * fx.Int32(SEGS)
                tok0 = grp * fx.Int32(TT)
                left = cur_tok - tok0
                ntok = fx.Int32((left < fx.Int32(TT)).select(left, fx.Int32(TT)))
                nrows = ntok * fx.Int32(topk)
                g_in = (
                    addr_comb
                    + fx.Int64(tok0 * fx.Int32(topk)) * fx.Int64(slot_stride)
                    + fx.Int64(seg * fx.Int32(LL128.SEG_B))
                )
                load_tile(g_in, nrows)
                # Row ``tid`` of the tile is a live route (every lane writes).
                live_row = tid < nrows
                e = fx.Int32(
                    buffer_load(
                        rsrc_ids,
                        fx.Int32(
                            live_row.select(tok0 * fx.Int32(topk) + tid, fx.Int32(0))
                        ),
                        vec_width=1,
                        dtype=T.i32,
                    )
                )
                ok = live_row & (e >= fx.Int32(0)) & (e < fx.Int32(n_experts_total))
                st(
                    base + fx.Int32(VROW_OFF) + tid * fx.Int32(4),
                    fx.Int32(ok.select(fx.Int32(1), fx.Int32(0))),
                )
                _rawbar()
                tries = fx.Int32(0)
                stale = tile_stale()
                while (stale != fx.Int32(0)) & (tries < fx.Int32(LL_TILE_RETRIES)):
                    wait_line(g_in, stale)
                    load_tile(g_in, nrows)
                    stale = tile_stale()
                    tries = tries + fx.Int32(1)
                if stale != fx.Int32(0):
                    _trap()
                accs = [fx.full(8, 0.0, fx.Float32) for _ in range_constexpr(2)]
                for k in range_constexpr(topk):
                    row = tok_l * fx.Int32(topk) + fx.Int32(k)
                    live = ld(
                        base + fx.Int32(VROW_OFF) + row * fx.Int32(4)
                    ) != fx.Int32(0)
                    for j in range_constexpr(2):
                        addr = fx.Int32(
                            live.select(
                                base
                                + row * fx.Int32(LL128.SEG_B)
                                + LL128.granule_offset(g0 + fx.Int32(j)),
                                base + fx.Int32(ZERO_OFF + j * 16),
                            )
                        )
                        raw = lds_load_b128(addr, 0)
                        accs[j] = accs[j] + raw.bitcast(fx.BFloat16).to(fx.Float32)
                for j in range_constexpr(2):
                    lds_store_b128(
                        base
                        + fx.Int32(OUT_OFF)
                        + tok_l * fx.Int32(OUT_ROW_B)
                        + g0 * fx.Int32(16)
                        + fx.Int32(j * 16),
                        0,
                        accs[j].to(fx.BFloat16).bitcast(fx.Int32).ir_value(),
                    )
                _rawbar()
                if wave == fx.Int32(0):
                    TDM.tdm_store(
                        TDM.tdm_group0(
                            base + fx.Int32(OUT_OFF),
                            addr_out
                            + fx.Int64(
                                tok0 * fx.Int32(hidden_dim)
                                + seg * fx.Int32(LL128.SEG_COLS)
                            )
                            * fx.Int64(2),
                        ),
                        TDM.tdm_group1_rows(
                            ntok, OUT_ROW_B // 4, OUT_ROW_B // 4, 4, hidden_dim * 2 // 4
                        ),
                    )
                    rocdl.s_nop(0)
                    TDM.tdm_wait(0)
                _rawbar()
                r = head_add(CTL_RED)
            TDM.tdm_wait(0)
            comm_ops.waitcnt_stores()
            # The last CTA out zeroes the heads (and the counter) for the next
            # launch; every other CTA has claimed its last item by then.
            done = head_add(CTL_DONE)
            if done == fx.Int32(G - 1):
                buffer_store(
                    fx.Int32(0),
                    rsrc_ctl,
                    fx.Int32(
                        (wave == fx.Int32(0)).select(
                            (lane == fx.Int32(0)).select(
                                fx.Int32(CTL_G2),
                                (lane == fx.Int32(1)).select(
                                    fx.Int32(CTL_RED),
                                    (lane == fx.Int32(2)).select(
                                        fx.Int32(CTL_DONE), fx.Int32(_OOB // 4)
                                    ),
                                ),
                            ),
                            fx.Int32(_OOB // 4),
                        )
                    ),
                )

        kernel(
            fx.get_iter(arg_c),
            arg_a,
            arg_b,
            fx.get_iter(arg_scale_a),
            fx.get_iter(arg_scale_b),
            arg_psum,
            arg_bias,
            fx.get_iter(arg_rowmap),
            addr_comb,
            addr_out,
            addr_ids,
            addr_psum,
            addr_ctl,
            i32_m,
            cur_tok,
        ).launch(grid=(G, 1, 1), block=(BLOCK, 1, 1), stream=stream)

    launch.compile_hints = dict(launch_gemm_a8w4_tdm.compile_hints)
    return launch
