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
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .. import buffer_ops
from ..kernels_common import ceildiv, create_llvm_ptr
from ..tensor_shim import GTensor, buf_base_i64
from .fp8_mqa_logits import (
    _MFMA16,
    MfmaAtom,
    _emit_row_neg_inf_fill,
    _fn_to_fnuz_i64,
    _load_row_weights,
    _make_weight_copy,
)

Vec = fx.Vector


def _emit_col_partial(mfma, mma, gemm_kw, a_row, b_pack, w_row, f32_0):
    """``_emit_col_sum`` without the head reduction and ``kv_scale`` multiply,
    which the caller applies once across several rows."""
    col_sum = f32_0
    for mi in range_constexpr(len(a_row)):
        c_frag = fx.make_rmem_tensor(mfma.ACC_ELEMS, fx.Float32)
        c_frag.store(Vec.filled(mfma.ACC_ELEMS, 0.0, fx.Float32))
        for kk in range_constexpr(len(a_row[mi])):
            fx.gemm(mma, c_frag, a_row[mi][kk], b_pack[kk], c_frag, **gemm_kw)
        acc = c_frag.load()
        for ii in range_constexpr(mfma.ACC_ELEMS):
            col_sum = col_sum + Vec(acc)[ii].maximumf(f32_0) * w_row[mi][ii]
    return col_sum


def _reduce_scatter_rows(vals, g_bit1, g_bit0):
    """Head-reduce 2 or 4 rows' partials across the 4 MFMA lane groups.

    A reduce-scatter: each lane group ends up with a different row's full sum
    (row g for 4 rows; row g // 2 for 2 rows), using 3 and 2 shuffles instead
    of 8 and 4. ``g_bit1``/``g_bit0`` are the lane-group bits.
    """
    if len(vals) == 4:
        v0, v1, v2, v3 = vals
        # Stage A (xor 32): keep the row pair selected by bit 1, send the other.
        keep0 = g_bit1.select(v2, v0)
        keep1 = g_bit1.select(v3, v1)
        send0 = g_bit1.select(v0, v2)
        send1 = g_bit1.select(v1, v3)
        u0 = keep0 + send0.shuffle_xor(32, 64)
        u1 = keep1 + send1.shuffle_xor(32, 64)
        # Stage B (xor 16): keep the row selected by bit 0.
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
    min_split_tiles: int,
):
    """gfx942 LDS-staged kernel: the block's waves share one KV tile in LDS.

    Each of the ``waves_per_block`` waves owns ``rows_per_wave`` query rows, so
    it keeps only those rows' Q and weights in registers (3 waves/SIMD). All
    waves DMA one ``block_kv``-column tile into LDS (gfx942 only has dword LDS
    DMA) and read the whole tile back.

    Two LDS slots, one barrier per tile: wait for tile t's DMA, barrier, read
    tile t, start the DMA of tile t+1 into the other slot, then compute. Every
    wave issues a fixed number of DMAs and stores per tile, so the wait at the
    top of the loop is an exact vmcnt; a wave that skips a tile none of its
    rows can see drains with vmcnt(0) instead.

    ``seq_len`` is not padded. Rows past it, and rows with an empty window, get
    an empty window and do not widen the block's KV range. ``min_split_tiles``
    floors each grid.y slice of that range.
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
    MIN_SPLIT_TILES = int(min_split_tiles)
    assert MIN_SPLIT_TILES >= 1, "min_split_tiles must be >= 1"

    # Swizzle: 8-byte chunks per column, XOR mask over the low chunk bits.
    NCH = D // 8
    SW = min(NCH, 16)
    KV_BYTES = BKV * D
    # The scale region is always 64 dwords because the scale DMA writes one
    # dword per lane of a full wave; lanes >= BKV land in the unread tail.
    SLOT_BYTES = KV_BYTES + 64 * 4
    assert KV_BYTES % (NT * 4) == 0, "KV tile must split evenly into 4-byte DMA lanes"
    KV_DMA = KV_BYTES // (NT * 4)  # dword DMA ops per thread per tile
    assert BKV <= 64, "scale DMA covers one dword per lane of one wave"
    # Stores per tile per wave: one buffer_store per n-tile (masked lanes are
    # dropped by the descriptor bound, so the count is fixed).
    STORES = N_TILES
    # At the top of iteration t the only VMEM ops issued after tile t's DMA are
    # iteration t-1's STORES stores, so vmcnt(STORES) retires the DMA exactly.
    WAIT_TOP = STORES
    assert KV_DMA + 1 + STORES <= 63, "vmcnt encoding limit"

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
        block_row0 = (n_blocks - bid - fx.Int32(1)) * fx.Int32(ROWS_PER_BLOCK)

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
        cs_t = GTensor(cu_starts, dtype=T.i32, shape=(-1,))
        ce_t = GTensor(cu_ends, dtype=T.i32, shape=(-1,))
        _stride_i64 = fx.Int64(fx.Uint32(stride_logits_s))

        def _load_q_frag(off):
            v2 = q_i32.vec_load((off // fx.Int32(4),), vec_size=2)
            raw = Vec(v2).bitcast(fx.Int64)[0].ir_value()
            return mfma.make_frag(_fn_to_fnuz_i64(raw) if convert_q_fn else raw)

        # ---- LDS ----
        lds_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().slots.ptr
        lds_base_i64 = fx.Int64(fx.Uint32(fx.ptrtoint(lds_ptr)))
        lds_ptr0 = create_llvm_ptr(lds_base_i64, address_space=3)
        _i64x1 = Vec.make_type(2, fx.Int32)

        def _lds_read_b(slot_byte, col_local, kchunk):
            """8-byte B fragment: column col_local, 8-byte chunk kchunk."""
            phys = kchunk ^ (col_local & fx.Int32(SW - 1))
            byte = slot_byte + col_local * fx.Int32(D) + phys * fx.Int32(8)
            v = fx.ptr_load(lds_ptr + byte // fx.Int32(4), result_type=_i64x1)
            raw = Vec(v).bitcast(fx.Int64)[0].ir_value()
            return _fn_to_fnuz_i64(raw) if convert_kv_fn else raw

        def _lds_read_scale(slot_byte, col_local):
            byte = slot_byte + fx.Int32(KV_BYTES) + col_local * fx.Int32(4)
            return fx.Float32(
                fx.ptr_load(
                    lds_ptr + byte // fx.Int32(4), result_type=fx.Float32.ir_type
                )
            )

        # DMA step i: thread tid writes LDS dword (i*NT + tid), which is column
        # c = i*R + c0 of the tile, and fetches the swizzled chunk of
        # KV[col0 + c]. Bounds come from each step's descriptor (based at
        # KV[col0 + i*R], ending at seq_len_kv), not a per-lane clamp, so the
        # per-lane offset is loop-invariant and does not spill.
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
        kv_base = buf_base_i64(KV)
        sc_base = buf_base_i64(kv_scales)

        def _dma_tile(slot_byte, col0):
            """This wave's share of the KV tile at col0 + all BKV scales."""
            for i in range_constexpr(KV_DMA):
                cstep = col0 + fx.Int32(i * R)
                rsrc = buffer_ops.create_buffer_resource_from_addr(
                    (kv_base + fx.Int64(fx.Uint32(cstep)) * fx.Int64(D)).ir_value(),
                    num_records_bytes=fx.max(seq_len_kv - cstep, fx.Int32(0))
                    * fx.Int32(D),
                )
                wbase = slot_byte + fx.Int32(i * NT * 4) + wave * fx.Int32(256)
                wbase_s = rocdl.readfirstlane(
                    fx.Int64.ir_type, fx.Int64(fx.Uint32(wbase)).ir_value()
                )
                dst = buffer_ops.get_element_ptr(lds_ptr0, wbase_s)
                rocdl.raw_ptr_buffer_load_lds(
                    rsrc,
                    dst,
                    fx.Int32(4),
                    fx.Int32(lane_voff[(i * R) % SW]),
                    fx.Int32(0),
                    fx.Int32(0),
                    fx.Int32(0),
                )
            # Scales: lane l writes the scale of column col0+l into scale[l];
            # columns past seq_len_kv read as 0. Every wave writes identical
            # bytes, so the overlap is benign, and it keeps every wave's VMEM
            # op count equal, which the exact vmcnt wait relies on.
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

        # ---- Per-wave rows: windows, Q frags, weights ----
        starts = [None] * RPW
        ends = [None] * RPW
        fill_starts = [None] * RPW
        fill_ends = [None] * RPW
        a_packs = [None] * RPW
        w_frag = [None] * RPW
        seq_len_m1 = seq_len - fx.Int32(1)
        for j in range_constexpr(RPW):
            # A row past seq_len (short last block) reads row seq_len - 1 so
            # its loads stay in bounds, then gets an empty compute window
            # [seq_len_kv, 0) and an empty -inf fill window [0, seq_len_kv).
            row_v = wave_row0 + fx.Int32(j)
            row = fx.min(row_v, seq_len_m1)
            valid = row_v < seq_len
            s_j = fx.max(fx.Int32(cs_t[row]), fx.Int32(0))
            e_j = fx.min(fx.Int32(ce_t[row]), seq_len_kv)
            starts[j] = valid.select(s_j, seq_len_kv)
            ends[j] = valid.select(e_j, fx.Int32(0))
            fill_starts[j] = valid.select(s_j, fx.Int32(0))
            fill_ends[j] = valid.select(e_j, seq_len_kv)
            row_a = [[None] * K_STEPS for _ in range_constexpr(M_TILES)]
            for mi in range_constexpr(M_TILES):
                base_a = (
                    row * fx.Int32(H) + fx.Int32(mi * MFMA_M) + lane_mod_N
                ) * fx.Int32(D)
                for kk in range_constexpr(K_STEPS):
                    row_a[mi][kk] = _load_q_frag(
                        base_a + fx.Int32(kk * MFMA_K) + lane_frag_off
                    )
            a_packs[j] = row_a
            w_frag[j] = _load_row_weights(
                weights, H, cp_4xfp32, tc_c_w, M_TILES, MFMA_N, row
            )

        # ---- Union window over ALL block rows (uniform trip count -> safe barriers) ----
        u_start = None
        u_end = None
        for jj in range_constexpr(ROWS_PER_BLOCK):
            rr_v = block_row0 + fx.Int32(jj)
            rr = fx.min(rr_v, seq_len_m1)
            ss = fx.max(fx.Int32(cs_t[rr]), fx.Int32(0))
            ee = fx.min(fx.Int32(ce_t[rr]), seq_len_kv)
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
        # Floor each slice at MIN_SPLIT_TILES (see the docstring).
        split_cols = fx.max(
            fx.Int32(ceildiv(fx.Uint32(win_tiles), fx.Uint32(num_splits))),
            fx.Int32(MIN_SPLIT_TILES),
        ) * fx.Int32(BKV)
        tile_start = tile_start + by * split_cols
        tile_end = fx.min(tile_start + split_cols, tile_end)
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
            grp_writer = (g & fx.Int32(1)) == fx.Int32(0)
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
        OOB = fx.Int32(0x7FFFFFF0)

        # ---- Prologue: DMA tile 0; drain everything (Q, weights, windows, tile 0) ----
        _dma_tile(fx.Int32(0), tile_start)
        rocdl.s_waitcnt(vmcnt=0)

        for t_iv in range(fx.Int32(0), n_tiles, fx.Int32(1)):
            t = fx.Int32(t_iv)
            col0 = tile_start + t * fx.Int32(BKV)
            slot = (t & fx.Int32(1)) * fx.Int32(SLOT_BYTES)
            nslot = fx.Int32(SLOT_BYTES) - slot
            # The only VMEM ops issued after tile t's DMA are the previous
            # iteration's STORES stores, so vmcnt(STORES) retires exactly the DMA.
            rocdl.s_waitcnt(vmcnt=WAIT_TOP)
            gpu.barrier()

            b_all = []
            sc_all = []
            for ni in range_constexpr(N_TILES):
                cl = fx.Int32(ni * MFMA_N) + lane_mod_N
                sc_all.append(_lds_read_scale(slot, cl))
                b_all.append(
                    [
                        _lds_read_b(slot, cl, fx.Int32(kk * 4) + g)
                        for kk in range_constexpr(K_STEPS)
                    ]
                )

            # Prefetch tile t+1 into the other slot. On the last iteration this
            # reads past the window; the descriptor bounds keep it in range and
            # the data is never consumed.
            _dma_tile(nslot, col0 + fx.Int32(BKV))

            def _tile_compute(b_all, sc_all, col0):
                for ni in range_constexpr(N_TILES):
                    b_pack = [
                        mfma.make_frag(b_all[ni][kk]) for kk in range_constexpr(K_STEPS)
                    ]
                    parts = [
                        _emit_col_partial(
                            mfma, mma, gemm_kw, a_packs[j], b_pack, w_frag[j], f32_0
                        )
                        for j in range_constexpr(RPW)
                    ]
                    if const_expr(RPW == 1):
                        red = parts[0]
                        for sh in mfma.shuffle_offsets:
                            red = red + red.shuffle_xor(sh, 64)
                    else:
                        red = _reduce_scatter_rows(parts, g_bit1, g_bit0)
                    val = red * sc_all[ni]
                    col = col0 + fx.Int32(ni * MFMA_N) + lane_mod_N
                    ok = grp_writer & (col >= my_start) & (col < my_end)
                    byte_off = ok.select((my_row_off + col) * fx.Int32(4), OOB)
                    buffer_ops.buffer_store(
                        val, out_rsrc, byte_off, offset_is_bytes=True
                    )

            def _tile_skip():
                # A skipped tile issues no stores, so the next iteration's
                # vmcnt(STORES) would retire before the prefetch DMA lands.
                # Drain instead; the wave has no work on this tile anyway.
                rocdl.s_waitcnt(vmcnt=0)

            # w_start, w_end and col0 are wave-uniform, so the branch is too.
            wave_live = (col0 < w_end) & ((col0 + fx.Int32(BKV)) > w_start)
            if wave_live:
                _tile_compute(b_all, sc_all, col0)
            else:
                _tile_skip()

        rocdl.s_waitcnt(vmcnt=0)

        if const_expr(clean_logits):
            neg_inf = fx.Float32(float("-inf"))

            def _store_neg_inf(t, c):
                t[c] = neg_inf

            def _fill_range(out_row_t, lo_i32, hi_i32):
                for c in range(lo_i32 + lane, hi_i32, fx.Int32(64)):
                    _store_neg_inf(out_row_t, fx.Int32(c))

            _emit_row_neg_inf_fill(
                logits=logits,
                stride_i64=_stride_i64,
                rows=[wave_row0 + fx.Int32(j) for j in range_constexpr(RPW)],
                starts=fill_starts,
                ends=fill_ends,
                seq_len_kv=seq_len_kv,
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
