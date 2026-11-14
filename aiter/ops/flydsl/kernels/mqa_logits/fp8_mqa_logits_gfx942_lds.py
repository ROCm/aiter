# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP8 MQA logits -- gfx942 LDS-staged FlyDSL kernel builder.

Shares the atoms, fragment loaders and -inf fill with ``fp8_mqa_logits``; the
host-side registry and launcher live in
``aiter.ops.flydsl.fp8_mqa_logits_kernels``.
"""

# No `from __future__ import annotations`: FlyDSL arg typing needs real
# annotation objects, not PEP 563 strings.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .. import buffer_ops
from ..kernels_common import ceildiv, create_llvm_ptr
from ..tensor_shim import GTensor, buf_base_i64
from .fp8_mqa_logits import (
    _MFMA16,
    MfmaAtom,
    _emit_row_neg_inf_fill,
    _load_row_weights,
    _make_weight_copy,
)

Vec = fx.Vector


# Buffer cache policy for the paired RPW=2 logits stores, which write full
# 128-byte row segments: non-temporal.
_CPOL_NT = 2
# sched_group_barrier interleave: VALU_PER_MFMA VALU ops per MFMA.
_SCHED_VALU = 0x2
VALU_PER_MFMA = 2
# The next tile's DMA: DMA_FIRST ops go with the last B read, then DMA_PER_UNIT
# ops after each following MFMA chain.
DMA_FIRST = 3
DMA_PER_UNIT = 1
# MFMA chains between the stages of an in-tile head reduce.
STAGE_GAP = 3


def _fn_to_fnuz_i64(raw_i64):
    """Map FN byte 0x80 (-0) to 0x00 in 8 packed fp8 bytes."""

    def _fix_i32(src):
        # Per byte, bit 7 of z is set iff the low 7 bits are nonzero (0x7F + 0x7F
        # cannot carry out of the byte), so a byte equal to 0x80 is the only one
        # with bit 7 set in src and clear in z.
        z = (src & fx.Int32(0x7F7F7F7F)) + fx.Int32(0x7F7F7F7F)
        is80 = src & (z ^ fx.Int32(-1)) & fx.Int32(-0x7F7F7F80)  # 0x80808080
        return src ^ is80

    raw = fx.Uint64(raw_i64)
    lo_64 = fx.Int64(fx.Uint32(_fix_i32(fx.Int32(raw))))
    hi_64 = fx.Int64(fx.Uint32(_fix_i32(fx.Int32(raw >> 32)))) << fx.Int64(32)
    return (lo_64 | hi_64).ir_value()


def _opaque(v):
    """Identity LLVM cannot fold.

    Without it fast-math reassociates the row partials through the
    reduce-scatter, which costs extra VALU ops and VGPR spills.
    """
    r = llvm.inline_asm(
        fx.Float32.ir_type,
        [fx.Float32(v).ir_value()],
        "",
        "=v,0",
        has_side_effects=False,
    )
    return fx.Float32(r)


def _reduce_scatter_rows(vals, g_bit1, g_bit0, pair=False):
    """Head-reduce row partials across the 4 lane groups (reduce-scatter).

    4 values: slot j must hold row ``j ^ g``; lane group g ends with row g's sum.
    ``pair``: the 4 values are 2 rows x 2 n-tiles (slot = 2 * row + n-tile, rows
    permuted by ``g >> 1``); group g ends with row ``g >> 1``, n-tile ``g & 1``.
    2 values: group g ends with row ``g >> 1``.
    """
    if len(vals) == 4:
        v0, v1, v2, v3 = [_opaque(v) for v in vals]
        u0 = v0 + v2.shuffle_xor(32, 64)
        u1 = v1 + v3.shuffle_xor(32, 64)
        if not pair:
            return u0 + u1.shuffle_xor(16, 64)
        keep = g_bit0.select(u1, u0)
        send = g_bit0.select(u0, u1)
        return keep + send.shuffle_xor(16, 64)
    if len(vals) == 2:
        v0, v1 = vals
        keep = g_bit1.select(v1, v0)
        send = g_bit1.select(v0, v1)
        u = keep + send.shuffle_xor(32, 64)
        return u + u.shuffle_xor(16, 64)
    raise ValueError(f"reduce-scatter supports 2 or 4 rows, got {len(vals)}")


def _build_kernel_mfma_lds_gfx942(
    *,
    num_heads: int,
    head_size: int,
    block_kv: int,
    rows_per_wave: int,
    waves_per_block: int,
    mfma: MfmaAtom = _MFMA16,
    convert_q_fn: bool = False,
    convert_kv_fn: bool = False,
    clean_logits: bool = True,
):
    """gfx942 LDS-staged kernel: the block's waves share one KV tile in LDS.

    Each of the ``waves_per_block`` waves owns ``rows_per_wave`` query rows, so
    it keeps only those rows' Q and weights in registers (3 waves/SIMD). All
    waves DMA one ``block_kv``-column tile into LDS (gfx942 only has dword LDS
    DMA) and read the whole tile back.

    Two LDS slots, one barrier per tile: wait for tile t's DMA, barrier, read
    tile t and compute it. Tile t+1's DMA into the other slot goes out after
    tile t's last LDS read (an LDS read after it would make the compiler wait
    for it), and tile t's last head reduce + store is carried past the next
    barrier, so the vmcnt at the loop top is exact. A wave that skips a tile
    none of its rows can see drains with vmcnt(0).

    ``seq_len`` is not padded. Rows past it, and rows with an empty window, get
    an empty window and do not widen the block's KV range. grid.y splits that
    range into ``num_splits`` slices of decreasing size.
    """
    H = num_heads
    D = head_size
    BKV = block_kv
    RPW = rows_per_wave
    WPB = waves_per_block
    NT = 64 * WPB
    ROWS_PER_BLOCK = RPW * WPB
    MFMA_M, MFMA_N, MFMA_K = mfma.MFMA_M, mfma.MFMA_N, mfma.MFMA_K
    assert (
        mfma.frag_bytes == 8 and MFMA_N == 16
    ), "gfx942 LDS builder needs the dense 16x16x32 atom"
    assert H % MFMA_M == 0 and D % MFMA_K == 0 and BKV % MFMA_N == 0
    assert RPW in (1, 2, 4), "reduce-scatter supports RPW in {1, 2, 4}"
    N_TILES = BKV // MFMA_N
    M_TILES = H // MFMA_M
    K_STEPS = D // MFMA_K
    NUM_BUFFERS = 2

    # Swizzle: 8-byte chunks per column, XOR mask over the low chunk bits.
    SW = min(D // 8, 16)
    KV_BYTES = BKV * D
    # The scale region is always 64 dwords because the scale DMA writes one
    # dword per lane of a full wave; lanes >= BKV land in the unread tail.
    SLOT_BYTES = KV_BYTES + 64 * 4
    assert KV_BYTES % (NT * 4) == 0, "KV tile must split evenly into 4-byte DMA lanes"
    KV_DMA = KV_BYTES // (NT * 4)  # dword DMA ops per thread per tile
    # DMA steps per descriptor (the instruction offset must stay below 4096).
    DMA_GS = max(1, 4096 // (NT * 4))
    assert BKV <= 64, "scale DMA covers one dword per lane of one wave"
    # PAIR: RPW=2 reduces two n-tiles together and writes them with one store.
    PAIR = RPW == 2 and N_TILES % 2 == 0
    STORES = N_TILES // 2 if PAIR else N_TILES
    # Stores issued after the next tile's DMA, which the loop-top vmcnt must allow
    # for (masked lanes go to an out-of-range offset, so the count is fixed). The
    # DMA leaves after the last B read, so n-tiles 1..N_TILES-1 follow it, or all
    # pair stores; the tile's last store is carried past the next barrier.
    WAIT_TOP = max((STORES if PAIR else N_TILES - 1) - 1, 0)
    N_CARRY = 4 if PAIR else RPW
    # Staged finish: each head reduce-scatter is spread over 3 MFMA chains.
    STAGED = PAIR or RPW == 4
    # PAIR spreads the next tile's DMA over the MFMA chains of the last two n-tiles
    # when there are enough chains to carry every op.
    N_DMA = KV_DMA + 1
    SPLIT = (
        PAIR and N_TILES >= 3 and DMA_FIRST + 2 * RPW * M_TILES * DMA_PER_UNIT >= N_DMA
    )
    # H128 spills with the early n-tile 0 B read and with the scheduling hints.
    LOW_VGPR = H <= 64

    def dma_plan(k):
        """DMA ops issued at split step k (0: with the last B read)."""
        if k == 0:
            return list(range(DMA_FIRST))
        return list(
            range(
                DMA_FIRST + (k - 1) * DMA_PER_UNIT,
                min(DMA_FIRST + k * DMA_PER_UNIT, N_DMA),
            )
        )

    def dma_left(k):
        return DMA_FIRST + k * DMA_PER_UNIT < N_DMA

    assert N_DMA + STORES <= 63, "vmcnt encoding limit"

    _cvt_tag = ("_cq" if convert_q_fn else "") + ("_ck" if convert_kv_fn else "")
    _cl_tag = "" if clean_logits else "_nocl"
    _kname = (
        f"fp8_mqa_logits_H{H}_D{D}_bkv{BKV}_mfma_r{RPW}_w{WPB}_lds2"
        f"{_cvt_tag}{_cl_tag}_flydsl"
    )

    @fx.struct
    class SharedStorage:
        slots: fx.Array[fx.Int32, NUM_BUFFERS * SLOT_BYTES // 4, 128]

    @flyc.kernel(name=_kname, known_block_size=[NT, 1, 1])
    def kernel(
        Q: fx.Tensor,
        KV: fx.Tensor,
        kv_scales: fx.Tensor,
        weights: fx.Tensor,
        cu_starts: fx.Tensor,
        cu_ends: fx.Tensor,
        logits: fx.Tensor,
        seq_len: fx.Int32,  # not padded; the last block may be short
        seq_len_kv: fx.Int32,
        stride_logits_s: fx.Int32,
        num_splits: fx.Int32,
    ):
        f32_0 = fx.Float32(0.0)
        mma = mfma.make_atom()
        gemm_kw = mfma.gemm_kwargs()

        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        n_blocks = fx.Int32(ceildiv(fx.Uint32(seq_len), fx.Uint32(ROWS_PER_BLOCK)))
        cs_t = GTensor(cu_starts, dtype=T.i32, shape=(-1,))
        ce_t = GTensor(cu_ends, dtype=T.i32, shape=(-1,))
        # Equal causal segments (detected from the last row's start): interleave
        # them so every segment's longest blocks dispatch first. Any other layout
        # keeps the plain reversed order. Always a permutation.
        seg_l = seq_len_kv - fx.Int32(cs_t[seq_len - fx.Int32(1)])
        seg_ok = (seg_l > fx.Int32(0)) & (seg_l < seq_len)
        seg_ls = seg_ok.select(seg_l, seq_len)
        seg_ok = (
            seg_ok
            & ((seg_ls % fx.Int32(ROWS_PER_BLOCK)) == fx.Int32(0))
            & ((seq_len % seg_ls) == fx.Int32(0))
        )
        nseg = seg_ok.select(seq_len // seg_ls, fx.Int32(1))
        bps = seg_ok.select(seg_ls // fx.Int32(ROWS_PER_BLOCK), n_blocks)
        seg_i = nseg - fx.Int32(1) - bid % nseg
        rank = bid // nseg
        block_row0 = (seg_i * bps + bps - fx.Int32(1) - rank) * fx.Int32(ROWS_PER_BLOCK)

        # readfirstlane makes the wave index provably uniform, so the per-wave
        # rows, output descriptor and DMA bases all stay in SGPRs (a divergent
        # descriptor would make LLVM wrap every buffer op in a waterfall loop).
        wave = fx.Int32(
            rocdl.readfirstlane(fx.Int32.ir_type, (tid // fx.Int32(64)).ir_value())
        )
        lane = tid % fx.Int32(64)
        g = lane // fx.Int32(MFMA_N)  # lane group 0..3
        lane_mod_N = lane % fx.Int32(MFMA_N)
        lane_frag_off = g * fx.Int32(8)
        cp_4xfp32, tc_c_w = _make_weight_copy(mma, lane)
        wave_row0 = block_row0 + wave * fx.Int32(RPW)

        q_i32 = GTensor(Q, dtype=T.i32, shape=(-1,))
        _stride_i64 = fx.Int64(fx.Uint32(stride_logits_s))

        def _load_q_frag(off):
            v2 = q_i32.vec_load((off // fx.Int32(4),), vec_size=2)
            raw = Vec(v2).bitcast(fx.Int64)[0].ir_value()
            return mfma.make_frag(_fn_to_fnuz_i64(raw) if convert_q_fn else raw)

        # ---- LDS ----
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().slots.ptr
        lds_base_i64 = fx.Int64(fx.Uint32(fx.ptrtoint(lds_ptr)))
        lds_ptr0 = create_llvm_ptr(lds_base_i64, address_space=3)
        _i32x2 = Vec.make_type(2, fx.Int32)

        # DMA step i: thread tid writes LDS dword (i*NT + tid), which is column
        # c = i*R + c0 of the tile, and fetches the swizzled chunk of
        # KV[col0 + c]. Bounds come from the tile's descriptor (ending at
        # seq_len_kv), not a per-lane clamp, so the per-lane offset is
        # loop-invariant and does not spill.
        R = NT * 4 // D
        assert (NT * 4) % D == 0, "a DMA step must cover whole columns"
        p0 = tid * fx.Int32(4)
        c0 = p0 // fx.Int32(D)
        within = p0 - c0 * fx.Int32(D)
        pc = within // fx.Int32(8)
        h4 = within - pc * fx.Int32(8)
        lane_voff = {}
        for i in range_constexpr(KV_DMA):
            key = (i * R) % SW
            if key not in lane_voff:
                lc = pc ^ ((c0 + fx.Int32(key)) & fx.Int32(SW - 1))
                lane_voff[key] = c0 * fx.Int32(D) + lc * fx.Int32(8) + h4
        for i in range_constexpr(KV_DMA):
            kk = ((i * R) % SW, i // DMA_GS)
            if kk not in lane_voff:
                lane_voff[kk] = lane_voff[kk[0]] + fx.Int32(kk[1] * DMA_GS * NT * 4)
        kv_base = buf_base_i64(KV)
        sc_base = buf_base_i64(kv_scales)

        def _dma_tile(slot_byte, col0, ops=None, st=None):
            """DMA this wave's share of the KV tile at col0, plus all BKV scales.

            ``ops`` picks a subset of the op indices (KV_DMA is the scale op); ``st``
            carries the descriptor and LDS base between partial calls.
            """
            if ops is None:
                ops = list(range(KV_DMA + 1))
            if st is None:
                st = {}
            # One descriptor and M0 per tile: a step advances the global source and
            # the LDS destination by the same NT*4 bytes, so the instruction offset
            # (applied to both, and bounds-checked) carries the step inside a group
            # of DMA_GS steps. Later groups add gi*4096 to voffset and M0.
            rsrc = st.get("rsrc")
            dst = st.get("dst")
            for i in range_constexpr(KV_DMA):
                if const_expr(i not in ops):
                    continue
                if const_expr(rsrc is None):
                    rsrc = buffer_ops.create_buffer_resource_from_addr(
                        (kv_base + fx.Int64(fx.Uint32(col0)) * fx.Int64(D)).ir_value(),
                        num_records_bytes=fx.max(seq_len_kv - col0, fx.Int32(0))
                        * fx.Int32(D),
                    )
                    wbase = slot_byte + wave * fx.Int32(256)
                    wbase_s = rocdl.readfirstlane(
                        fx.Int64.ir_type, fx.Int64(fx.Uint32(wbase)).ir_value()
                    )
                    dst = buffer_ops.get_element_ptr(lds_ptr0, wbase_s)
                    st["rsrc"] = rsrc
                    st["dst"] = dst
                gi = i // DMA_GS
                if const_expr(gi > 0):
                    dst_i = buffer_ops.get_element_ptr(
                        dst, static_byte_offset=gi * DMA_GS * NT * 4
                    )
                else:
                    dst_i = dst
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc,
                    dst_i,
                    fx.Int32(4),
                    fx.Int32(lane_voff[((i * R) % SW, gi)]),
                    fx.Int32(0),
                    fx.Int32((i % DMA_GS) * NT * 4),
                    fx.Int32(0),
                )
            if const_expr(KV_DMA not in ops):
                return st
            # Scales: lane l writes the scale of column col0+l into scale[l];
            # columns past seq_len_kv read as 0. Every wave writes identical
            # bytes, which keeps every wave's VMEM op count equal for the exact
            # vmcnt wait.
            srsrc = buffer_ops.create_buffer_resource_from_addr(
                (sc_base + fx.Int64(fx.Uint32(col0)) * fx.Int64(4)).ir_value(),
                num_records_bytes=fx.max(seq_len_kv - col0, fx.Int32(0)) * fx.Int32(4),
            )
            sbase = slot_byte + fx.Int32(KV_BYTES)
            sbase_s = rocdl.readfirstlane(
                fx.Int64.ir_type, fx.Int64(fx.Uint32(sbase)).ir_value()
            )
            sdst = buffer_ops.get_element_ptr(lds_ptr0, sbase_s)
            rocdl.raw_ptr_buffer_load_lds(
                srsrc,
                sdst,
                fx.Int32(4),
                lane * fx.Int32(4),
                fx.Int32(0),
                fx.Int32(0),
                fx.Int32(0),
            )
            return st

        # ---- Per-wave rows: windows, Q frags, weights ----
        starts = [None] * RPW
        ends = [None] * RPW
        fill_starts = [None] * RPW
        fill_ends = [None] * RPW
        a_packs = [None] * RPW
        w_frag = [None] * RPW
        seq_len_m1 = seq_len - fx.Int32(1)
        # Permuted rows: accumulator lane group g (heads 4g..4g+3 of an m-tile)
        # keeps row j ^ g (RPW=4) or j ^ (g >> 1) (PAIR) in slot j, so the
        # reduce-scatter needs no row selects. A-lane l feeds head l % 16.
        if const_expr(RPW == 4):
            hg_q = lane_mod_N // fx.Int32(4)
            hg_w = g
        elif const_expr(PAIR):
            hg_q = lane_mod_N // fx.Int32(8)
            hg_w = g // fx.Int32(2)
        else:
            hg_q = None
            hg_w = None
        w_all = [None] * RPW
        # Issue every window load (this wave's rows and the block's union rows)
        # before any use: one memory round trip.
        _own_cs, _own_ce = [], []
        for j in range_constexpr(RPW):
            _r = fx.min(wave_row0 + fx.Int32(j), seq_len_m1)
            _own_cs.append(fx.Int32(cs_t[_r]))
            _own_ce.append(fx.Int32(ce_t[_r]))
        _blk_cs, _blk_ce = [], []
        for jj in range_constexpr(ROWS_PER_BLOCK):
            _r = fx.min(block_row0 + fx.Int32(jj), seq_len_m1)
            _blk_cs.append(fx.Int32(cs_t[_r]))
            _blk_ce.append(fx.Int32(ce_t[_r]))
        for j in range_constexpr(RPW):
            # A row past seq_len (short last block) reads row seq_len - 1 so
            # its loads stay in bounds, then gets an empty compute window
            # [seq_len_kv, 0) and an empty -inf fill window [0, seq_len_kv).
            row_v = wave_row0 + fx.Int32(j)
            row = fx.min(row_v, seq_len_m1)
            valid = row_v < seq_len
            s_j = fx.max(_own_cs[j], fx.Int32(0))
            e_j = fx.min(_own_ce[j], seq_len_kv)
            starts[j] = valid.select(s_j, seq_len_kv)
            ends[j] = valid.select(e_j, fx.Int32(0))
            fill_starts[j] = valid.select(s_j, fx.Int32(0))
            fill_ends[j] = valid.select(e_j, seq_len_kv)
            row_a = [[None] * K_STEPS for _ in range_constexpr(M_TILES)]
            row_q = row
            if const_expr(hg_q is not None):
                row_q = fx.min(wave_row0 + (fx.Int32(j) ^ hg_q), seq_len_m1)
            for mi in range_constexpr(M_TILES):
                base_a = (
                    row_q * fx.Int32(H) + fx.Int32(mi * MFMA_M) + lane_mod_N
                ) * fx.Int32(D)
                for kk in range_constexpr(K_STEPS):
                    row_a[mi][kk] = _load_q_frag(
                        base_a + fx.Int32(kk * MFMA_K) + lane_frag_off
                    )
            a_packs[j] = row_a
            w_all[j] = _load_row_weights(
                weights, H, cp_4xfp32, tc_c_w, M_TILES, MFMA_N, row
            )
        if const_expr(hg_w is None):
            w_frag = w_all
        else:
            for j in range_constexpr(RPW):
                rj = fx.Int32(j) ^ hg_w
                w_frag[j] = []
                for mi in range_constexpr(M_TILES):
                    els = []
                    for ii in range_constexpr(mfma.ACC_ELEMS):
                        v = w_all[0][mi][ii]
                        for r in range_constexpr(1, RPW):
                            v = (rj == fx.Int32(r)).select(w_all[r][mi][ii], v)
                        els.append(v)
                    w_frag[j].append(els)

        # ---- Union window over ALL block rows (uniform trip count -> safe barriers)
        u_start = None
        u_end = None
        for jj in range_constexpr(ROWS_PER_BLOCK):
            rr_v = block_row0 + fx.Int32(jj)
            ss = fx.max(_blk_cs[jj], fx.Int32(0))
            ee = fx.min(_blk_ce[jj], seq_len_kv)
            # Rows that write no logits (past seq_len, or an empty window) must
            # not widen the union.
            live = (ee > ss) & (rr_v < seq_len)
            ss = live.select(ss, seq_len_kv)
            ee = live.select(ee, fx.Int32(0))
            if jj == 0:
                u_start, u_end = ss, ee
            else:
                u_start = fx.min(u_start, ss)
                u_end = fx.max(u_end, ee)
        tile_start = (u_start // fx.Int32(BKV)) * fx.Int32(BKV)
        tile_end = fx.max(u_end, tile_start)
        by = fx.block_idx.y
        win_tiles = fx.Int32(ceildiv(fx.Uint32(tile_end - tile_start), fx.Uint32(BKV)))
        # Descending slices: slice y holds a share proportional to num_splits - y,
        # so the grid.y slices dispatched last are the smallest. Slice y starts at
        # ceil(win_tiles * y * (2 * ns - y + 1) / (ns * (ns + 1))); the launcher
        # caps num_splits so that this product fits in i32.
        ns = num_splits
        one = fx.Int32(1)
        den = ns * (ns + one)
        by1 = by + one
        lo_t = (win_tiles * by * (fx.Int32(2) * ns - by + one) + den - one) // den
        hi_t = (win_tiles * by1 * (fx.Int32(2) * ns - by) + den - one) // den
        win_base = tile_start
        tile_start = win_base + lo_t * fx.Int32(BKV)
        tile_end = fx.min(win_base + hi_t * fx.Int32(BKV), tile_end)
        n_tiles = fx.Int32(
            ceildiv(
                fx.Uint32(fx.max(tile_end - tile_start, fx.Int32(0))), fx.Uint32(BKV)
            )
        )

        # ---- Output ownership after the reduce-scatter ----
        g_bit1 = (g & fx.Int32(2)) != fx.Int32(0)
        g_bit0 = (g & fx.Int32(1)) != fx.Int32(0)
        if const_expr(RPW == 4):
            my_j = g
            grp_writer = g < fx.Int32(4)  # every lane group writes
        elif const_expr(RPW == 2):
            my_j = g // fx.Int32(2)
            grp_writer = (
                (g < fx.Int32(4)) if PAIR else ((g & fx.Int32(1)) == fx.Int32(0))
            )
        else:
            my_j = fx.Int32(0)
            grp_writer = g == fx.Int32(0)
        my_start = starts[0]
        my_end = ends[0]
        for j in range_constexpr(1, RPW):
            sel = my_j == fx.Int32(j)
            my_start = sel.select(starts[j], my_start)
            my_end = sel.select(ends[j], my_end)
        # This wave's window: the union of its live rows, used to skip tiles
        # none of them can see. Wave-uniform; readfirstlane keeps it in SGPRs.
        w_start = seq_len_kv
        w_end = fx.Int32(0)
        for j in range_constexpr(RPW):
            live_j = ends[j] > starts[j]
            w_start = fx.min(w_start, live_j.select(starts[j], seq_len_kv))
            w_end = fx.max(w_end, live_j.select(ends[j], fx.Int32(0)))
        w_start = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, w_start.ir_value()))
        w_end = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, w_end.ir_value()))
        # Output descriptor: base = this wave's first row, bounded to its RPW
        # rows, so masked lanes can store to an out-of-range offset and be
        # dropped by hardware (fixed store count per tile -> exact vmcnt).
        out_base = buf_base_i64(logits) + fx.Int64(
            fx.Uint32(wave_row0)
        ) * _stride_i64 * fx.Int64(4)
        out_rsrc = buffer_ops.create_buffer_resource_from_addr(
            out_base.ir_value(),
            num_records_bytes=(fx.Int32(RPW) * stride_logits_s * fx.Int32(4)),
        )
        my_row_off = my_j * stride_logits_s
        my_width = grp_writer.select(
            fx.max(my_end - my_start, fx.Int32(0)), fx.Int32(0)
        )
        OOB = fx.Int32(0x7FFFFFF0)

        b_inv_dw = []
        for kk in range_constexpr(K_STEPS):
            phys = (fx.Int32(kk * 4) + g) ^ (lane_mod_N & fx.Int32(SW - 1))
            b_inv_dw.append(
                (lane_mod_N * fx.Int32(D) + phys * fx.Int32(8)) // fx.Int32(4)
            )
        sc_inv_dw = fx.Int32(KV_BYTES // 4) + lane_mod_N
        store_cpol = _CPOL_NT if PAIR else 0

        def _emit_store(parts, sc, byte_off):
            if const_expr(PAIR):
                red = _reduce_scatter_rows(parts, g_bit1, g_bit0, pair=True)
            elif const_expr(RPW == 1):
                red = parts[0]
                for sh in mfma.shuffle_offsets:
                    red = red + red.shuffle_xor(sh, 64)
            else:
                red = _reduce_scatter_rows(parts, g_bit1, g_bit0)
            buffer_ops.buffer_store(
                red * sc,
                out_rsrc,
                byte_off,
                offset_is_bytes=True,
                cache_modifier=store_cpol,
            )

        # ---- Prologue: DMA tile 0; drain everything (Q, weights, windows, tile 0) ----
        _dma_tile(fx.Int32(0), tile_start)
        rocdl.s_waitcnt(vmcnt=0)

        _c_init = [f32_0] * N_CARRY + [f32_0, OOB]
        for t_iv, cstate in range(fx.Int32(0), n_tiles, fx.Int32(1), init=_c_init):
            t = fx.Int32(t_iv)
            c_parts = [fx.Float32(cstate[i]) for i in range_constexpr(N_CARRY)]
            c_sc = fx.Float32(cstate[N_CARRY])
            c_off = fx.Int32(cstate[N_CARRY + 1])
            col0 = tile_start + t * fx.Int32(BKV)
            slot = (t & fx.Int32(1)) * fx.Int32(SLOT_BYTES)
            # N-tile ni's column is ni*16 + lane_mod_N, so its swizzle is lane_mod_N
            # for every ni: one dword address per k step (and one for the scales),
            # with the n-tile as a constant ds_read offset.
            slot_dw = slot // fx.Int32(4)
            kk_dw = [slot_dw + b_inv_dw[kk] for kk in range_constexpr(K_STEPS)]
            sc_dw = slot_dw + sc_inv_dw
            nslot = fx.Int32(SLOT_BYTES) - slot
            rocdl.s_waitcnt(vmcnt=WAIT_TOP)
            gpu.barrier()

            def _read_b(ni, sc_dw=sc_dw, kk_dw=kk_dw):
                sc = fx.Float32(
                    fx.ptr_load(
                        lds_ptr + (sc_dw + fx.Int32(ni * MFMA_N)),
                        result_type=fx.Float32.ir_type,
                    )
                )
                bs = []
                for kk in range_constexpr(K_STEPS):
                    v = fx.ptr_load(
                        lds_ptr + (kk_dw[kk] + fx.Int32(ni * MFMA_N * D // 4)),
                        result_type=_i32x2,
                    )
                    raw = Vec(v).bitcast(fx.Int64)[0].ir_value()
                    bs.append(
                        mfma.make_frag(_fn_to_fnuz_i64(raw) if convert_kv_fn else raw)
                    )
                return bs, sc

            b0 = _read_b(0) if const_expr(LOW_VGPR) else None
            # The previous tile's carried head reduce + store, under the first B reads.
            _emit_store(c_parts, c_sc, c_off)

            # Prefetch tile t+1 into the other slot after this tile's last LDS read: a
            # ds_read after an LDS DMA in program order makes the compiler wait for
            # it. On the last iteration the descriptor bounds keep it in range.
            def _prefetch(ops=None, st=None, nslot=nslot, col0=col0):
                return _dma_tile(nslot, col0 + fx.Int32(BKV), ops, st)

            def _tile_compute(col0, b0):
                # Software pipeline: B fragments are read one n-tile ahead, chain
                # u+1's MFMAs run under chain u's ReLU x weight epilogue, and each
                # n-tile's head reduce + store runs under the next n-tile's MFMAs.
                units = [
                    (ni, j, mi)
                    for ni in range(N_TILES)
                    for j in range(RPW)
                    for mi in range(M_TILES)
                ]
                col_l = col0 + lane_mod_N
                if const_expr(PAIR):
                    col_l = col_l + (g & fx.Int32(1)) * fx.Int32(MFMA_N)
                d_col = col_l - my_start
                byte_base = (my_row_off + col_l) * fx.Int32(4)
                bufs = {0: b0 if b0 is not None else _read_b(0)}

                def _chain(u):
                    ni, j, mi = units[u]
                    c_frag = fx.make_rmem_tensor(mfma.ACC_ELEMS, fx.Float32)
                    c_frag.store(Vec.filled(mfma.ACC_ELEMS, 0.0, fx.Float32))
                    for kk in range_constexpr(K_STEPS):
                        fx.gemm(
                            mma,
                            c_frag,
                            a_packs[j][mi][kk],
                            bufs[ni][0][kk],
                            c_frag,
                            **gemm_kw,
                        )
                    return c_frag.load()

                def _finish_prep(ni, parts):
                    if const_expr(PAIR):
                        sc = g_bit0.select(bufs[ni + 1][1], bufs[ni][1])
                    else:
                        sc = bufs[ni][1]
                    # col - my_start < my_width as one unsigned compare.
                    ok = fx.Uint32(d_col + fx.Int32(ni * MFMA_N)) < fx.Uint32(my_width)
                    byte_off = ok.select(byte_base + fx.Int32(ni * MFMA_N * 4), OOB)
                    return list(parts), sc, byte_off

                def _finish(ni, parts):
                    _emit_store(*_finish_prep(ni, parts))

                def _finish_staged(ni, parts):
                    """The head reduce-scatter as 3 stages, so each shuffle's wait
                    lands STAGE_GAP MFMA chains after its issue."""
                    st = {}

                    def _s0():
                        vals, sc, byte_off = _finish_prep(ni, parts)
                        v = [_opaque(x) for x in vals]
                        st["v"] = v
                        st["sc"] = sc
                        st["off"] = byte_off
                        st["a2"] = v[2].shuffle_xor(32, 64)
                        st["a3"] = v[3].shuffle_xor(32, 64)

                    def _s1():
                        u0 = st["v"][0] + st["a2"]
                        u1 = st["v"][1] + st["a3"]
                        if const_expr(PAIR):
                            st["keep"] = g_bit0.select(u1, u0)
                            send = g_bit0.select(u0, u1)
                        else:
                            st["keep"] = u0
                            send = u1
                        st["b"] = send.shuffle_xor(16, 64)

                    def _s2():
                        stores_after_dma[0] += int(dma_issued[0])
                        buffer_ops.buffer_store(
                            (st["keep"] + st["b"]) * st["sc"],
                            out_rsrc,
                            st["off"],
                            offset_is_bytes=True,
                            cache_modifier=store_cpol,
                        )

                    return [_s0, _s1, _s2]

                if const_expr(N_TILES > 1):
                    bufs[1] = _read_b(1)
                if const_expr(N_TILES <= 2):
                    _prefetch()
                parts = {}
                pending = None
                dma_u0 = None
                dma_st = None
                # Staged finishes in flight: [stages, next stage, chains since last].
                inflight = []
                # The loop-top vmcnt(WAIT_TOP) is exact only if at least WAIT_TOP
                # stores follow the last DMA op.
                stores_after_dma = [0]
                dma_issued = [False]
                nxt = _chain(0)
                for u in range_constexpr(len(units)):
                    ni, j, mi = units[u]
                    cur = nxt
                    if const_expr(u + 1 < len(units)):
                        n_ni, n_j, n_mi = units[u + 1]
                        if const_expr(n_j == 0 and n_mi == 0 and n_ni + 1 < N_TILES):
                            bufs[n_ni + 1] = _read_b(n_ni + 1)
                            if const_expr(n_ni + 2 == N_TILES):
                                if const_expr(SPLIT):
                                    dma_u0 = u
                                    dma_st = _prefetch(dma_plan(0), None)
                                else:
                                    _prefetch()
                                dma_issued[0] = not SPLIT
                        nxt = _chain(u + 1)
                    if const_expr(SPLIT and dma_u0 is not None and u > dma_u0):
                        _ops = dma_plan(u - dma_u0)
                        if const_expr(len(_ops) > 0):
                            _prefetch(_ops, dma_st)
                            if const_expr(not dma_left(u - dma_u0)):
                                dma_issued[0] = True
                    dma_pending = SPLIT and (dma_u0 is None or dma_left(u - dma_u0))
                    if const_expr(STAGED):
                        _keep = []
                        for _f in inflight:
                            _f[2] += 1
                            if const_expr(
                                (_f[1] == 2 and dma_pending) or _f[2] < STAGE_GAP
                            ):
                                _keep.append(_f)
                                continue
                            _f[0][_f[1]]()
                            _f[1] += 1
                            _f[2] = 0
                            if const_expr(_f[1] < 3):
                                _keep.append(_f)
                        inflight = _keep
                        if const_expr(pending is not None):
                            _g = _finish_staged(*pending)
                            _g[0]()
                            inflight.append([_g, 1, 0])
                            pending = None
                    elif const_expr(pending is not None):
                        _finish(*pending)
                        pending = None
                    s = parts.get((ni, j), f32_0)
                    for ii in range_constexpr(mfma.ACC_ELEMS):
                        s = s + Vec(cur)[ii].maximumf(f32_0) * w_frag[j][mi][ii]
                    parts[(ni, j)] = s
                    if const_expr(PAIR):
                        if const_expr(
                            j == RPW - 1 and mi == M_TILES - 1 and ni % 2 == 1
                        ):
                            assert (
                                pending is None
                            ), "DMA split delayed a finish past the next one"
                            pending = (
                                ni - 1,
                                [
                                    parts[(ni - 1, 0)],
                                    parts[(ni, 0)],
                                    parts[(ni - 1, 1)],
                                    parts[(ni, 1)],
                                ],
                            )
                    elif const_expr(j == RPW - 1 and mi == M_TILES - 1):
                        pending = (ni, [parts[(ni, jj)] for jj in range_constexpr(RPW)])
                    if const_expr(LOW_VGPR):
                        for _k in range_constexpr(K_STEPS):
                            rocdl.sched_group_barrier(rocdl.mask_mfma, 1, 0)
                            rocdl.sched_group_barrier(_SCHED_VALU, VALU_PER_MFMA, 0)
                        rocdl.sched_barrier(0)
                for _f in inflight:
                    while const_expr(_f[1] < 3):
                        _f[0][_f[1]]()
                        _f[1] += 1
                if const_expr(STAGED):
                    assert stores_after_dma[0] >= WAIT_TOP, (
                        stores_after_dma[0],
                        WAIT_TOP,
                    )
                return _finish_prep(*pending)

            def _tile_skip():
                # A skipped tile issues no stores, so the next vmcnt(WAIT_TOP) would
                # retire before the prefetch lands. Drain instead.
                _prefetch()
                rocdl.s_waitcnt(vmcnt=0)

            # w_start, w_end and col0 are wave-uniform, so the branch is too.
            wave_live = (col0 < w_end) & ((col0 + fx.Int32(BKV)) > w_start)
            y0, y1, y2, y3, y4, y5 = f32_0, f32_0, f32_0, f32_0, f32_0, OOB
            if wave_live:
                n_parts, n_sc, n_off = _tile_compute(col0, b0)
                n_parts = list(n_parts) + [f32_0] * (4 - N_CARRY)
                y0 = fx.Float32(n_parts[0])
                y1 = fx.Float32(n_parts[1])
                y2 = fx.Float32(n_parts[2])
                y3 = fx.Float32(n_parts[3])
                y4 = fx.Float32(n_sc)
                y5 = fx.Int32(n_off)
            else:
                _tile_skip()
            cres = yield [y0, y1, y2, y3][:N_CARRY] + [y4, y5]

        rocdl.s_waitcnt(vmcnt=0)
        _emit_store(
            [fx.Float32(cres[i]) for i in range_constexpr(N_CARRY)],
            fx.Float32(cres[N_CARRY]),
            fx.Int32(cres[N_CARRY + 1]),
        )

        if const_expr(clean_logits):
            neg_inf = fx.Float32(float("-inf"))

            def _store_neg_inf(t, c):
                t[c] = neg_inf

            neg_inf4 = Vec.filled(4, float("-inf"), fx.Float32)

            def _fill_range(out_row_t, lo_i32, hi_i32):
                # Scalar head up to a multiple of 4 columns, 16-byte stores over the
                # body, scalar tail.
                a = fx.min((lo_i32 + fx.Int32(3)) // fx.Int32(4) * fx.Int32(4), hi_i32)
                b = fx.max((hi_i32 // fx.Int32(4)) * fx.Int32(4), a)
                ch = lo_i32 + lane
                if ch < a:
                    _store_neg_inf(out_row_t, ch)
                ct = b + lane
                if ct < hi_i32:
                    _store_neg_inf(out_row_t, ct)
                rsrc = buffer_ops.create_buffer_resource_from_addr(
                    out_row_t.base_i64.ir_value(),
                    num_records_bytes=(hi_i32 * fx.Int32(4)),
                )
                for c in range(a + lane * fx.Int32(4), b, fx.Int32(256)):
                    buffer_ops.buffer_store(
                        neg_inf4,
                        rsrc,
                        fx.Int32(c) * fx.Int32(4),
                        offset_is_bytes=True,
                    )

            _emit_row_neg_inf_fill(
                logits=logits,
                stride_i64=_stride_i64,
                rows=[wave_row0 + fx.Int32(j) for j in range_constexpr(RPW)],
                starts=fill_starts,
                ends=fill_ends,
                seq_len_kv=seq_len_kv,
                seq_len=seq_len,
                by_i32=by,
                num_splits=num_splits,
                fill_range=_fill_range,
            )

    @flyc.jit
    def launch_fp8_mqa_logits_mfma_lds_gfx942(
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
        ).launch(grid=(gx, gy, 1), block=(NT, 1, 1), stream=stream)

    return launch_fp8_mqa_logits_mfma_lds_gfx942
