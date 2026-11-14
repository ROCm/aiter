# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Row-packed gfx942 FP4 prefill index scorer emitting 64-token page maxima.

For single-head indexers (MiniMax M3). The single-row prefill scorer fills one
of the MFMA's 16 M slots, so each e2m1 decode and scale multiply (VALU work on
gfx942) serves one query. Here the 16 slots hold 16 query rows of the same
sequence, so each decoded K tile serves 16 queries, and the epilogue scales,
masks each row to its causal window and reduces to per-page maxima in-kernel.

Scores are signed and scaled by ``weight_scale`` only. A page maximum is
bit-identical to ``amax`` over the single-row kernel's ``relu=False`` logits.

Layouts (G groups of 16 rows, S = 16 G slots):
  q_fp4:     [G, 16, D/2] uint8   — slot s of group g is packed row 16 g + s.
  q_scale:   [G, K_TILES, 4, 16, 4] uint8 — byte [g][kt][kk][s][0] is the e8m0
             of slot s at 32-chunk kk; bytes [..][1:] are padding.
  slot_meta: [3, S] int32 — output row (-1: empty slot), causal start, causal
             end of each slot.
  out:       [rows, >= ceil(max_seq_len / 64)] fp32, caller-filled with -inf;
             pages no CTA reaches keep that fill.

With ``q_fp8=True`` the A operand is FP8 instead of MXFP4 (gfx950's indexer
keeps Q in e4m3 and only K in fp4; on gfx942 K is decoded to e4m3fnuz in
registers anyway, so an e4m3fnuz Q costs the same MFMA, skips the Q decode and
removes the dominant quantization error):
  q_fp4:     [G, 16, D] e4m3fnuz bytes — slot s of group g is row 16 g + s.
  q_scale:   [G, 16] fp32 — per-slot dequantization scale.
"""

from __future__ import annotations

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, rocdl
from flydsl.expr.primitive import const_expr, range_constexpr
from flydsl.expr.typing import Int32, T

from .pa_mqa_logits_fp4_common import _i32_buffer, _load_vec4_i32
from .pa_mqa_logits_fp4_prefill_gfx942 import (
    CTA_INFO_WIDTH,
    DEFAULT_NUM_WARPS,
    KK_PER_KTILE,
    MFMA_N,
    WARP_SIZE,
    _decode_fp4_dword,
    _frag_i64_from_pp,
    compute_prefill_schedule,
)

ROWS_PER_GROUP = 16
PAGE_TOKENS = 64
DEFAULT_HEAD_DIM = 128
# fp32 -inf as a signed i32, for the bit-level mask select.
_NEG_INF_BITS = -8388608
# Fused FP8-Q decode: max|q| of a row maps to this e4m3fnuz value (max 240),
# as vLLM's pack_decode_rows_fp8_gfx942 does.
FP8Q_ROW_MAX = 224.0


def build_pa_mqa_page_max_fp4_rowpack_prefill_gfx942_module(
    block_k=256,
    kv_block_size=64,
    max_blocks_per_seq=256,
    num_warps=DEFAULT_NUM_WARPS,
    head_dim=DEFAULT_HEAD_DIM,
    kv_page_stride=None,
    kv_scale_page_stride=None,
    kv_scale_offset=0,
    groups_per_cta=1,
    q_fp8=False,
    block_out=False,
    init_blocks=0,
    local_blocks=0,
    fused_q_heads=0,
    fused_q_len=1,
    sentinel_init=1e30,
    sentinel_local=1e29,
):
    # block_out: the epilogue maxes each 128-token block's two page maxima
    # across the wave pair that owns it (through LDS), applies the init /
    # local pinning, and writes pinned block scores into ``out`` viewed as
    # [rows, width] (``out_pages`` = width) instead of page maxima.
    #
    # fused_q_heads > 0 (decode, needs q_fp8 and groups_per_cta == 1): ``q_ptr``
    # is the bf16 [rows, H, D] index query (strides q_stride_row/head in
    # elements), group g is request g, and slot h * fused_q_len + r is head h
    # of its row r. The prologue computes each slot's max|q| -> 224 e4m3fnuz
    # bytes and fp32 scale in registers, and the slot metadata from
    # ``seq_lens_ptr``; ``q_scale_ptr`` and ``slot_meta_ptr`` are unused.
    FUSED_Q = fused_q_heads > 0
    if FUSED_Q:
        assert q_fp8 and groups_per_cta == 1
        assert fused_q_heads * fused_q_len <= ROWS_PER_GROUP
    if block_out:
        assert num_warps % 2 == 0 and block_k % 128 == 0
    # groups_per_cta > 1: cta_info's group id names a run of groups_per_cta
    # consecutive 16-row groups (same sequence); each decoded K tile then
    # feeds groups_per_cta MFMAs instead of one, amortizing the e2m1 decode
    # (the VALU bottleneck on gfx942) over 16 * groups_per_cta query rows.
    GPC = groups_per_cta
    heads = ROWS_PER_GROUP
    block_threads_k = num_warps * WARP_SIZE
    k_tiles = head_dim // 128
    assert head_dim % 128 == 0, f"head_dim must be a multiple of 128, got {head_dim}"
    N_TILES = block_k // MFMA_N
    assert N_TILES % num_warps == 0
    N_TILES_PER_WARP = N_TILES // num_warps
    assert kv_block_size == 64, "gfx942 packed KV loads require kv_block_size=64"
    assert (
        N_TILES_PER_WARP == 4
    ), "each wave covers one 64-token page per chunk: block_k == 64 * num_warps"
    pages_per_chunk = block_k // PAGE_TOKENS

    _stride_bt = max_blocks_per_seq
    _kv_chunk_bytes = 16
    _stride_kv_ktile = 4 * kv_block_size * _kv_chunk_bytes
    _stride_kv_block = k_tiles * _stride_kv_ktile
    _stride_kvs_ktile = 4 * kv_block_size
    _stride_kvs_block = k_tiles * _stride_kvs_ktile
    if kv_page_stride is not None:
        _stride_kv_block = kv_page_stride
    if kv_scale_page_stride is not None:
        _stride_kvs_block = kv_scale_page_stride
    qs_pad = 4
    N_KV = k_tiles * N_TILES_PER_WARP * KK_PER_KTILE
    N_KVS = k_tiles * KK_PER_KTILE
    # LDS page-max exchange: [2 (chunk parity), GPC, num_warps, 16 slots].
    XCHG_WORDS = 2 * GPC * num_warps * ROWS_PER_GROUP

    # Built with explicit annotations: this module defers annotations.
    BlockOutShared = fx.struct(
        type(
            "RowpackBlockOutShared",
            (),
            {"__annotations__": {"xchg": fx.Array[fx.Int32, XCHG_WORDS, 16]}},
        )
    )

    @flyc.kernel
    def pa_mqa_page_max_fp4_rowpack_prefill_gfx942_kernel(
        out_ptr: fx.Tensor,
        q_ptr: fx.Tensor,
        q_scale_ptr: fx.Tensor,
        kv_cache_ptr: fx.Tensor,
        kv_scale_ptr: fx.Tensor,
        kv_indices_ptr: fx.Tensor,
        slot_meta_ptr: fx.Tensor,
        cta_info_ptr: fx.Tensor,
        stride_out_row: Int32,
        out_pages: Int32,
        slot_plane: Int32,
        out_elems: Int32,
        weight_scale: fx.Float32,
        seq_lens_ptr: fx.Tensor,
        q_stride_row: Int32,
        q_stride_head: Int32,
        q_rows: Int32,
    ):
        tid = gpu.thread_idx.x
        pid = gpu.block_idx.x
        warp_id = tid >> 6
        lane_id = tid % WARP_SIZE
        lane_mod_16 = lane_id & 15
        lane_div_16 = (lane_id >> 4) & 3

        cta_src = fx.get_iter(cta_info_ptr)
        cta_it = fx.add_offset(
            fx.recast_iter(fx.PointerType.get(T.i32, cta_src.memspace, 4), cta_src),
            fx.Int64(pid) * fx.Int64(CTA_INFO_WIDTH),
        )
        cta_info_bt = fx.rocdl.make_buffer_tensor(
            fx.make_view(cta_it, fx.make_layout((1 << 28, 4), (4, 1)))
        )
        cta_info_vec = fx.Vector(_load_vec4_i32(cta_info_bt, fx.Int32(0)))
        super_id = cta_info_vec[0]
        batch_id = cta_info_vec[1]
        chunk_start = cta_info_vec[2]
        chunk_count = cta_info_vec[3]
        # chunk_count >= 1 always (unused slots get 1), so the page-max
        # variant always runs.
        cta_live = chunk_count
        if const_expr(block_out):
            cta_live = fx.Int32(
                _i32_buffer(cta_info_ptr, width=1)[
                    pid * fx.Int32(CTA_INFO_WIDTH) + fx.Int32(5)
                ]
            )

        kv_bt = _i32_buffer(kv_cache_ptr, width=1)
        kvs_bt = _i32_buffer(kv_scale_ptr, width=1)
        bt_bt = _i32_buffer(kv_indices_ptr, width=1)
        q_bt = _i32_buffer(q_ptr, width=1)
        qs_bt = _i32_buffer(q_scale_ptr, width=1)
        qs4_bt = _i32_buffer(q_scale_ptr, width=4) if q_fp8 else None

        # This lane's four MFMA output rows are slots 4 * (lane / 16) + t of
        # each of the CTA's GPC groups.
        meta_bt = _i32_buffer(slot_meta_ptr, width=4)
        group_ids = [super_id * fx.Int32(GPC) + fx.Int32(gi) for gi in range(GPC)]
        rows_vs, starts_vs, ends_vs = [], [], []
        for gi in range_constexpr(0 if FUSED_Q else GPC):
            slot0 = group_ids[gi] * fx.Int32(ROWS_PER_GROUP) + lane_div_16 * fx.Int32(4)
            rows_vs.append(fx.Vector(_load_vec4_i32(meta_bt, slot0)))
            starts_vs.append(fx.Vector(_load_vec4_i32(meta_bt, slot_plane + slot0)))
            ends_vs.append(
                fx.Vector(_load_vec4_i32(meta_bt, slot_plane * fx.Int32(2) + slot0))
            )

        # Dropped stores use offset out_elems, discarded by the range check.
        out_buf = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.recast_iter(
                    fx.PointerType.get(T.f32, out_ptr.memspace, 4),
                    fx.get_iter(out_ptr),
                ),
                fx.make_layout((out_elems, 1), (1, 1)),
            ),
            max_size=False,
            num_records_bytes=out_elems * fx.Int32(4),
        )
        out_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 1)
        out_reg_ty = fx.MemRefType.get(
            T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
        )
        out_reg_lay = fx.make_layout(1, 1)

        ZERO_F = fx.Float32(0.0)
        c0_i32 = fx.Int32(0)
        NEG_INF = fx.Int32(_NEG_INF_BITS)
        mfma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))

        def _e8m0_to_f32(b):
            v = fx.Vector.from_elements(
                [(b & fx.Int32(0xFF)) << fx.Int32(23)], dtype=fx.Int32
            ).bitcast(fx.Float32)
            return v[0]

        # ---- A-fragments: slot lane_mod_16 of each group, 4 K-steps per k_tile.
        q_a_ops_g = []
        sa_f32_tab_g = []
        # q_fp8: lane l's A-fragment of K-step kk is bytes kk*32 + 8*(l/16) ..
        # +8 of slot l%16, loaded as-is; the per-slot scale of this lane's four
        # output rows (times weight_scale) is applied once in the epilogue.
        row_scale_g = []
        lane_i32 = fx.Int32(lane_id)

        def _bperm_f32(src_lane, v):
            return fx.Int32(
                rocdl.ds_bpermute(T.i32, src_lane * fx.Int32(4), v.bitcast(fx.Int32))
            ).bitcast(fx.Float32)

        if const_expr(FUSED_Q):
            # Slot s = lane % 16 of request b = group id: head s // QL, row
            # b * QL + s % QL. Lane l holds bytes kk*32 + 8*(l/16) .. +8 of the
            # slot's row, so the 4 lanes l/16 = 0..3 of a slot cover all of it.
            QL = fused_q_len
            HQ = fused_q_heads * QL
            b = group_ids[0]
            s = lane_mod_16
            s_live = s < fx.Int32(HQ)
            q_head = s // fx.Int32(QL)
            q_r = s % fx.Int32(QL)
            q_row = b * fx.Int32(QL) + q_r
            # Dead slots read row 0 / head 0 and are zeroed below.
            elem0 = s_live.select(
                q_row * q_stride_row + q_head * q_stride_head, fx.Int32(0)
            ) + lane_div_16 * fx.Int32(8)
            xs = []  # [kk][8] fp32
            amax = ZERO_F
            for kk in range_constexpr(KK_PER_KTILE * k_tiles):
                xk = []
                for j in range_constexpr(4):
                    dw = fx.Int32(
                        q_bt[(elem0 + fx.Int32(kk * 32 + 2 * j)) // fx.Int32(2)]
                    )
                    dw = s_live.select(dw, fx.Int32(0))
                    lo = dw << fx.Int32(16)
                    hi = dw & fx.Int32(-65536)
                    xk += [lo.bitcast(fx.Float32), hi.bitcast(fx.Float32)]
                    for v in (lo, hi):
                        amax = amax.maximumf(
                            (v & fx.Int32(0x7FFFFFFF)).bitcast(fx.Float32)
                        )
                xs.append(xk)
            for sh in [16, 32]:
                amax = amax.maximumf(_bperm_f32(lane_i32 ^ fx.Int32(sh), amax))
            # Correctly rounded divides, as the Triton pack kernel (div_rn).
            pos = amax > ZERO_F
            safe = pos.select(amax, fx.Float32(1.0))
            inv = pos.select(fx.Float32(FP8Q_ROW_MAX) / safe, ZERO_F)
            slot_scale = amax / fx.Float32(FP8Q_ROW_MAX)

            def _q8_in(v):
                # e4m3fnuz has no -0 (0x80 is NaN): anything rounding to zero,
                # |v| <= 2^-11 (half the smallest subnormal, ties to even), is +0.
                sv = v * inv
                tiny = (sv.bitcast(fx.Int32) & fx.Int32(0x7FFFFFFF)) <= fx.Int32(
                    0x3A000000
                )
                return tiny.select(ZERO_F, sv).ir_value()

            frags = []
            for kk in range_constexpr(KK_PER_KTILE * k_tiles):
                words = []
                for half in range_constexpr(2):
                    pk = fx.Int32(0).ir_value()
                    for sel in range_constexpr(2):
                        e = half * 4 + sel * 2
                        pk = rocdl.cvt_pk_fp8_f32(
                            T.i32,
                            _q8_in(xs[kk][e]),
                            _q8_in(xs[kk][e + 1]),
                            pk,
                            sel,
                        )
                    words.append(fx.Int32(pk))
                frags.append(_frag_i64_from_pp(words[0], words[1]))
            q_a_ops_g.append(
                [
                    frags[kt * KK_PER_KTILE : (kt + 1) * KK_PER_KTILE]
                    for kt in range(k_tiles)
                ]
            )
            # This lane's four output rows are slots 4 * (l / 16) + t, whose
            # scale lives on lane 4 * (l / 16) + t.
            row_scale_g.append(
                [
                    _bperm_f32(lane_div_16 * fx.Int32(4) + fx.Int32(t), slot_scale)
                    * weight_scale
                    for t in range(4)
                ]
            )
            # Slot metadata, as pack_decode_rows_fp8_gfx942 writes it.
            seq_len = fx.Int32(_i32_buffer(seq_lens_ptr, width=1)[b])
            rows_l, starts_l, ends_l = [], [], []
            for t in range_constexpr(4):
                st = lane_div_16 * fx.Int32(4) + fx.Int32(t)
                live_t = st < fx.Int32(HQ)
                r_t = st % fx.Int32(QL)
                row_t = (st // fx.Int32(QL)) * q_rows + b * fx.Int32(QL) + r_t
                end_t = fx.max(seq_len - fx.Int32(QL) + r_t + fx.Int32(1), fx.Int32(0))
                rows_l.append(live_t.select(row_t, fx.Int32(-1)))
                starts_l.append(fx.Int32(0))
                ends_l.append(live_t.select(end_t, fx.Int32(0)))
            rows_vs.append(rows_l)
            starts_vs.append(starts_l)
            ends_vs.append(ends_l)
        for gi in range_constexpr(GPC if (q_fp8 and not FUSED_Q) else 0):
            q_row_base = (
                group_ids[gi] * fx.Int32(heads * head_dim)
                + lane_mod_16 * fx.Int32(head_dim)
                + lane_div_16 * fx.Int32(8)
            )
            q_a_ops = []
            for k_tile in range_constexpr(k_tiles):
                frags = []
                for kk in range_constexpr(KK_PER_KTILE):
                    dw = (q_row_base + fx.Int32(k_tile * 128 + kk * 32)) // fx.Int32(4)
                    lo = fx.Int32(q_bt[dw])
                    hi = fx.Int32(q_bt[dw + fx.Int32(1)])
                    frags.append(_frag_i64_from_pp(lo, hi))
                q_a_ops.append(frags)
            q_a_ops_g.append(q_a_ops)
            qs_v = fx.Vector(
                _load_vec4_i32(
                    qs4_bt,
                    group_ids[gi] * fx.Int32(ROWS_PER_GROUP)
                    + lane_div_16 * fx.Int32(4),
                )
            ).bitcast(fx.Float32)
            row_scale_g.append([qs_v[t] * weight_scale for t in range(4)])
        for gi in range_constexpr(0 if q_fp8 else GPC):
            q_base = group_ids[gi] * fx.Int32(heads * (head_dim // 2))
            q_row_base = q_base + lane_mod_16 * fx.Int32(head_dim // 2)
            q_a_ops = []
            for k_tile in range_constexpr(k_tiles):
                frags = []
                for kk in range_constexpr(KK_PER_KTILE):
                    off = (
                        q_row_base
                        + fx.Int32(k_tile * 64 + kk * 16)
                        + lane_div_16 * fx.Int32(4)
                    )
                    d = fx.Int32(q_bt[off // fx.Int32(4)])
                    p01, p23 = _decode_fp4_dword(d)
                    frags.append(_frag_i64_from_pp(p01, p23))
                q_a_ops.append(frags)
            q_a_ops_g.append(q_a_ops)

            # A-scales of this lane's four output rows (slots 4*(l/16)+t).
            qs_base = group_ids[gi] * fx.Int32(k_tiles * 64 * qs_pad)
            sa_f32_tab = []
            for k_tile in range_constexpr(k_tiles):
                sa_kt = []
                for kk in range_constexpr(KK_PER_KTILE):
                    sa_kk = []
                    for t in range_constexpr(4):
                        row = lane_div_16 * fx.Int32(4) + fx.Int32(t)
                        off = (
                            qs_base
                            + fx.Int32(k_tile * 64 * qs_pad)
                            + fx.Int32(kk * 16 * qs_pad)
                            + row * fx.Int32(qs_pad)
                        )
                        dw = fx.Int32(qs_bt[off // fx.Int32(4)])
                        sa_kk.append(_e8m0_to_f32(dw & fx.Int32(0xFF)))
                    sa_kt.append(sa_kk)
                sa_f32_tab.append(sa_kt)
            sa_f32_tab_g.append(sa_f32_tab)

        def _load_phys(c_i32_arg):
            token_local_base = (
                (chunk_start + c_i32_arg) * fx.Int32(block_k)
                + warp_id * fx.Int32(N_TILES_PER_WARP * MFMA_N)
                + lane_mod_16
            )
            # A chunk's tail may run past the row's last mapped page; masked
            # rows drop those tokens, but the page id must stay in the table.
            bi = fx.min(token_local_base >> fx.Int32(6), fx.Int32(_stride_bt - 1))
            phys_vec = bt_bt[batch_id * _stride_bt + bi]
            return [phys_vec] * N_TILES_PER_WARP

        def _prefetch_chunk(c_i32_arg, phys_list):
            kv_list = []
            kvs_list = []
            phys_shared = phys_list[0]
            for k_tile in range_constexpr(k_tiles):
                for kk in range_constexpr(KK_PER_KTILE):
                    kvs_off = (
                        fx.Int64(phys_shared) * fx.Int64(_stride_kvs_block)
                        + fx.Int64(kv_scale_offset)
                        + fx.Int64(k_tile * _stride_kvs_ktile)
                        + fx.Int64(kk * kv_block_size)
                        + fx.Int64(lane_mod_16) * fx.Int64(N_TILES_PER_WARP)
                    )
                    kvs_list.append(kvs_bt[kvs_off // fx.Int64(4)])
            token_in_block0 = (
                (chunk_start + c_i32_arg) * fx.Int32(block_k)
                + warp_id * fx.Int32(N_TILES_PER_WARP * MFMA_N)
            ) & fx.Int32(63)
            perm_src_byte = (lane_mod_16 * fx.Int32(4) + lane_div_16) * fx.Int32(4)
            for nt in range_constexpr(N_TILES_PER_WARP):
                for k_tile in range_constexpr(k_tiles):
                    for kk in range_constexpr(KK_PER_KTILE):
                        token_in_block = (
                            token_in_block0
                            + fx.Int32(nt * MFMA_N)
                            + (lane_id // fx.Int32(4))
                        ) & fx.Int32(63)
                        kv_off = (
                            fx.Int64(phys_shared) * fx.Int64(_stride_kv_block)
                            + fx.Int64(k_tile * _stride_kv_ktile)
                            + fx.Int64(kk * kv_block_size * _kv_chunk_bytes)
                            + fx.Int64(token_in_block) * fx.Int64(_kv_chunk_bytes)
                            + fx.Int64(lane_id % fx.Int32(4)) * fx.Int64(4)
                        )
                        raw_c = kv_bt[kv_off // fx.Int64(4)]
                        kv_list.append(
                            fx.Int32(rocdl.ds_bpermute(T.i32, perm_src_byte, raw_c))
                        )
            return kv_list, kvs_list

        def _frag_from_i64(v):
            frag = fx.make_rmem_tensor(1, fx.Int64)
            frag.store(fx.Vector.from_elements([v], dtype=fx.Int64))
            return frag

        def _decode_frags(kv_list_in):
            frags = []
            for i in range_constexpr(N_KV):
                p01, p23 = _decode_fp4_dword(fx.Int32(kv_list_in[i]))
                frags.append(
                    fx.Int64(fx.Uint32(p01))
                    | (fx.Int64(fx.Uint32(p23)) << fx.Int32(32))
                )
            return frags

        def _issue_nt_mfmas_frags(frags_in, kvs_list_in, nt):
            """One decoded K tile -> GPC groups' accumulators."""
            accs = [[ZERO_F for _ in range(4)] for _ in range(GPC)]
            for k_tile in range_constexpr(k_tiles):
                for kk in range_constexpr(KK_PER_KTILE):
                    b_frag = _frag_from_i64(
                        frags_in[((nt * k_tiles) + k_tile) * KK_PER_KTILE + kk]
                    )
                    sb_packed = fx.Int32(kvs_list_in[k_tile * KK_PER_KTILE + kk])
                    sb_f32 = _e8m0_to_f32(
                        (sb_packed >> fx.Int32(8 * nt)) & fx.Int32(0xFF)
                    )
                    for gi in range_constexpr(GPC):
                        c_frag = fx.make_rmem_tensor(4, fx.Float32)
                        c_frag.store(fx.Vector.filled(4, 0.0, fx.Float32))
                        fx.gemm(
                            mfma_atom, c_frag, q_a_ops_g[gi][k_tile][kk], b_frag, c_frag
                        )
                        partial = c_frag.load()
                        for t in range_constexpr(4):
                            if q_fp8:
                                accs[gi][t] = fx.fma(partial[t], sb_f32, accs[gi][t])
                            else:
                                accs[gi][t] = fx.fma(
                                    partial[t],
                                    sa_f32_tab_g[gi][k_tile][kk][t] * sb_f32,
                                    accs[gi][t],
                                )
            return accs

        def _masked_scaled(acc, nt, c_i32_arg, gi):
            starts_v = starts_vs[gi]
            ends_v = ends_vs[gi]
            token = (
                (chunk_start + c_i32_arg) * fx.Int32(block_k)
                + (warp_id * fx.Int32(N_TILES_PER_WARP) + fx.Int32(nt))
                * fx.Int32(MFMA_N)
                + lane_mod_16
            )
            vals = []
            for t in range_constexpr(4):
                scale_t = row_scale_g[gi][t] if q_fp8 else weight_scale
                bits = (acc[t] * scale_t).bitcast(fx.Int32)
                bits = (token < ends_v[t]).select(bits, NEG_INF)
                bits = (token < starts_v[t]).select(NEG_INF, bits)
                vals.append(bits.bitcast(fx.Float32))
            return fx.Vector.from_elements(vals, dtype=fx.Float32)

        lane_i32 = fx.Int32(lane_id)

        def _xor_max(vec, sh):
            peer_byte = (lane_i32 ^ fx.Int32(sh)) * fx.Int32(4)
            peers = []
            for t in range_constexpr(4):
                peer = fx.Int32(
                    rocdl.ds_bpermute(T.i32, peer_byte, vec[t].bitcast(fx.Int32))
                )
                peers.append(peer.bitcast(fx.Float32))
            return vec.maximumf(fx.Vector.from_elements(peers, dtype=fx.Float32))

        def _store_page_max(page_max, c_i32_arg, gi):
            rows_v = rows_vs[gi]
            page = (chunk_start + c_i32_arg) * fx.Int32(pages_per_chunk) + warp_id
            # Pages past the last column would land in the next row.
            writer = (lane_mod_16 < fx.Int32(1)) & (page < out_pages)
            for t in range_constexpr(4):
                row = rows_v[t]
                off = row * stride_out_row + page
                off = (row < fx.Int32(0)).select(out_elems, off)
                off = writer.select(off, out_elems)
                r_out = fx.memref_alloca(out_reg_ty, out_reg_lay)
                fx.memref_store_vec(
                    fx.Vector.from_elements([page_max[t]], dtype=fx.Float32), r_out
                )
                fx.copy(out_atom, r_out, fx.slice(out_buf, (off, None)))

        if const_expr(block_out):
            xchg_ptr = fx.SharedAllocator().allocate(BlockOutShared).peek().xchg.ptr
            blocks_per_chunk = block_k // 128
            # valid blocks of this lane's four output rows, per group
            valid_g = [
                [(ends_vs[gi][t] + fx.Int32(127)) // fx.Int32(128) for t in range(4)]
                for gi in range(GPC)
            ]

        def _store_block_scores(page_maxes, c_i32_arg):
            """Pair each even wave's page maxima with the next wave's (the
            other half of its 128-token block), pin and store block scores."""
            parity = c_i32_arg & fx.Int32(1)
            for gi in range_constexpr(GPC):
                base = (
                    (parity * fx.Int32(GPC) + fx.Int32(gi)) * fx.Int32(num_warps)
                    + warp_id
                ) * fx.Int32(ROWS_PER_GROUP) + lane_div_16 * fx.Int32(4)
                for t in range_constexpr(4):
                    fx.ptr_store(
                        page_maxes[gi][t].bitcast(fx.Int32),
                        xchg_ptr + fx.Int64(base + fx.Int32(t)),
                    )
            gpu.barrier()
            wid = fx.Int32(warp_id)
            is_even = (wid & fx.Int32(1)) == fx.Int32(0)
            writer = (lane_mod_16 < fx.Int32(1)) & is_even
            blk = (chunk_start + c_i32_arg) * fx.Int32(blocks_per_chunk) + (
                warp_id >> fx.Int32(1)
            )
            for gi in range_constexpr(GPC):
                # Partner wave warp_id ^ 1 (odd readers fetch their own pair
                # but never store).
                pbase = (
                    (parity * fx.Int32(GPC) + fx.Int32(gi)) * fx.Int32(num_warps)
                    + (wid ^ fx.Int32(1))
                ) * fx.Int32(ROWS_PER_GROUP) + lane_div_16 * fx.Int32(4)
                for t in range_constexpr(4):
                    other = fx.Int32(
                        fx.ptr_load(xchg_ptr + fx.Int64(pbase + fx.Int32(t)))
                    ).bitcast(fx.Float32)
                    score = page_maxes[gi][t].maximumf(other)
                    valid = valid_g[gi][t]
                    if const_expr(init_blocks > 0):
                        score = (blk < fx.Int32(init_blocks)).select(
                            fx.Float32(sentinel_init), score
                        )
                    if const_expr(local_blocks > 0):
                        score = (
                            blk >= fx.max(valid - fx.Int32(local_blocks), fx.Int32(0))
                        ).select(fx.Float32(sentinel_local), score)
                    row = rows_vs[gi][t]
                    off = row * stride_out_row + blk
                    ok = (
                        writer
                        & (blk < valid)
                        & (blk < out_pages)
                        & (row >= fx.Int32(0))
                    )
                    off = ok.select(off, out_elems)
                    r_out = fx.memref_alloca(out_reg_ty, out_reg_lay)
                    fx.memref_store_vec(
                        fx.Vector.from_elements([score], dtype=fx.Float32), r_out
                    )
                    fx.copy(out_atom, r_out, fx.slice(out_buf, (off, None)))

        def _compute_chunk_frags(frags_in, kvs_list_in, c_i32_arg):
            # Issue each tile's MFMAs before the previous tile's epilogue.
            acc0 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 0)
            acc1 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 1)
            pm = [_masked_scaled(acc0[gi], 0, c_i32_arg, gi) for gi in range(GPC)]
            acc2 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 2)
            pm = [
                pm[gi].maximumf(_masked_scaled(acc1[gi], 1, c_i32_arg, gi))
                for gi in range(GPC)
            ]
            acc3 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 3)
            page_maxes = []
            for gi in range_constexpr(GPC):
                page_max = pm[gi].maximumf(_masked_scaled(acc2[gi], 2, c_i32_arg, gi))
                page_max = page_max.maximumf(_masked_scaled(acc3[gi], 3, c_i32_arg, gi))
                # Reduce over the 16 lanes holding one row's tokens of a tile.
                for sh in [1, 2, 4, 8]:
                    page_max = _xor_max(page_max, sh)
                if const_expr(block_out):
                    page_maxes.append([page_max[t] for t in range(4)])
                else:
                    _store_page_max(page_max, c_i32_arg, gi)
            if const_expr(block_out):
                _store_block_scores(page_maxes, c_i32_arg)

        # Unused schedule slots (cta_info end == 0) would recompute chunk 0 of
        # group 0 and rewrite identical values; the block-score decode variant
        # skips them, so a grid sized for the largest batch stays cheap at short
        # context. (The page-max variant keeps its unconditional loop.)
        if cta_live > fx.Int32(0):
            last_c_i32 = chunk_count - fx.Int32(1)
            phys_pre = _load_phys(c0_i32)
            kv_pre, kvs_pre = _prefetch_chunk(c0_i32, phys_pre)
            frags_pre = _decode_frags(kv_pre)
            phys_next_pre = _load_phys(fx.min(fx.Int32(1), last_c_i32))

            chunk_count_minus_1_idx = fx.Int64(chunk_count - fx.Int32(1))
            init_args = list(frags_pre) + list(kvs_pre) + list(phys_next_pre)
            for c_idx, state in range(0, chunk_count_minus_1_idx, 1, init=init_args):
                frags_cur = [state[i] for i in range(N_KV)]
                kvs_cur_list = [state[N_KV + i] for i in range(N_KVS)]
                phys_next_list = [
                    state[N_KV + N_KVS + i] for i in range(N_TILES_PER_WARP)
                ]
                c_idx_i32 = fx.Int32(c_idx)
                c_next_i32 = c_idx_i32 + 1
                c_next_next_i32 = fx.min(c_next_i32 + 1, last_c_i32)
                kv_next, kvs_next = _prefetch_chunk(c_next_i32, phys_next_list)
                phys_next_next_list = _load_phys(c_next_next_i32)
                _compute_chunk_frags(frags_cur, kvs_cur_list, c_idx_i32)
                frags_next = _decode_frags(kv_next)
                results = yield (
                    list(frags_next) + list(kvs_next) + list(phys_next_next_list)
                )

            frags_last = [results[i] for i in range(N_KV)]
            kvs_last_list = [results[N_KV + i] for i in range(N_KVS)]
            _compute_chunk_frags(frags_last, kvs_last_list, last_c_i32)

    return pa_mqa_page_max_fp4_rowpack_prefill_gfx942_kernel, block_threads_k


@lru_cache(maxsize=32)
def compile_pa_mqa_page_max_fp4_rowpack_prefill_gfx942(
    *,
    block_k: int = 256,
    kv_block_size: int = 64,
    max_blocks_per_seq: int = 256,
    num_warps: int = DEFAULT_NUM_WARPS,
    head_dim: int = DEFAULT_HEAD_DIM,
    kv_page_stride: int | None = None,
    kv_scale_page_stride: int | None = None,
    kv_scale_offset: int = 0,
    groups_per_cta: int = 1,
    q_fp8: bool = False,
):
    kfn, block_threads = build_pa_mqa_page_max_fp4_rowpack_prefill_gfx942_module(
        block_k=block_k,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        num_warps=num_warps,
        head_dim=head_dim,
        kv_page_stride=kv_page_stride,
        kv_scale_page_stride=kv_scale_page_stride,
        kv_scale_offset=kv_scale_offset,
        groups_per_cta=groups_per_cta,
        q_fp8=q_fp8,
    )

    @flyc.jit
    def launch_pa_mqa_page_max_fp4_rowpack_prefill_gfx942(
        out,
        q,
        qs,
        kv,
        kvs,
        bt,
        meta,
        cta_info_,
        stride_out: fx.Int32,
        out_pages: fx.Int32,
        slot_plane: fx.Int32,
        out_elems: fx.Int32,
        weight_scale: fx.Float32,
        gx: fx.Int32,
        stream: fx.Stream,
    ):
        kfn(
            out,
            q,
            qs,
            kv,
            kvs,
            bt,
            meta,
            cta_info_,
            stride_out,
            out_pages,
            slot_plane,
            out_elems,
            weight_scale,
            meta,
            fx.Int32(0),
            fx.Int32(0),
            fx.Int32(0),
        ).launch(grid=(fx.Int64(gx),), block=(block_threads, 1, 1), stream=stream)

    return launch_pa_mqa_page_max_fp4_rowpack_prefill_gfx942, block_threads


@lru_cache(maxsize=32)
def compile_pa_mqa_block_score_fp4_rowpack_decode_gfx942(
    *,
    block_k: int = 256,
    max_blocks_per_seq: int = 256,
    head_dim: int = DEFAULT_HEAD_DIM,
    kv_page_stride: int | None = None,
    kv_scale_page_stride: int | None = None,
    kv_scale_offset: int = 0,
    init_blocks: int = 0,
    local_blocks: int = 0,
    sentinel_init: float = 1e30,
    sentinel_local: float = 1e29,
    fused_q_heads: int = 0,
    fused_q_len: int = 1,
):
    """Decode variant of the FP8-Q row-packed scorer that writes pinned
    128-token block scores (``block_out``) and, with ``fused_q_heads``, packs
    the bf16 index Q and slot metadata in its prologue.

    Launcher arguments: ``(out [rows_total, width] fp32 view, q, qs, kv, kvs,
    page64_table, slot_meta, cta_info, stride_out_row, width, slot_plane,
    out_elems, weight_scale, seq_lens, q_stride_row, q_stride_head, q_rows,
    num_ctas, stream)``. With ``fused_q_heads`` ``q`` is the bf16 ``[rows, H,
    D]`` query and ``qs`` / ``slot_meta`` are ignored (pass any buffers)."""
    kfn, block_threads = build_pa_mqa_page_max_fp4_rowpack_prefill_gfx942_module(
        block_k=block_k,
        kv_block_size=64,
        max_blocks_per_seq=max_blocks_per_seq,
        num_warps=block_k // PAGE_TOKENS,
        head_dim=head_dim,
        kv_page_stride=kv_page_stride,
        kv_scale_page_stride=kv_scale_page_stride,
        kv_scale_offset=kv_scale_offset,
        q_fp8=True,
        block_out=True,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
        sentinel_init=sentinel_init,
        sentinel_local=sentinel_local,
        fused_q_heads=fused_q_heads,
        fused_q_len=fused_q_len,
    )

    @flyc.jit
    def launch_pa_mqa_block_score_fp4_rowpack_decode_gfx942(
        out,
        q,
        qs,
        kv,
        kvs,
        bt,
        meta,
        cta_info_,
        stride_out: fx.Int32,
        out_width: fx.Int32,
        slot_plane: fx.Int32,
        out_elems: fx.Int32,
        weight_scale: fx.Float32,
        seq_lens,
        q_stride_row: fx.Int32,
        q_stride_head: fx.Int32,
        q_rows: fx.Int32,
        gx: fx.Int32,
        stream: fx.Stream,
    ):
        kfn(
            out,
            q,
            qs,
            kv,
            kvs,
            bt,
            meta,
            cta_info_,
            stride_out,
            out_width,
            slot_plane,
            out_elems,
            weight_scale,
            seq_lens,
            q_stride_row,
            q_stride_head,
            q_rows,
        ).launch(grid=(fx.Int64(gx),), block=(block_threads, 1, 1), stream=stream)

    return launch_pa_mqa_block_score_fp4_rowpack_decode_gfx942, block_threads


def rowpack_prefill_inputs_gfx942(
    q_rows_fp4: torch.Tensor,
    q_rows_scale: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    row_to_batch: torch.Tensor,
    local_starts: torch.Tensor,
    local_ends: torch.Tensor,
):
    """Pack ragged prefill rows into 16-row groups that never cross a sequence.

    Device-only, no host sync: there are ``rows // 16 + batches`` groups and
    the unused ones are empty. ``q_rows_scale`` is the ``[rows, D/32]`` e8m0 of
    ``dynamic_mxfp4_quant``.

    Returns:
        ``(q_fp4, q_scale, slot_meta, group_to_batch, group_starts,
        group_ends)`` in the kernel's layouts.
    """
    rows, packed_dim = q_rows_fp4.shape
    chunks = q_rows_scale.shape[1]
    batches = cu_seqlens_q.shape[0] - 1
    dev = q_rows_fp4.device
    groups = rows // ROWS_PER_GROUP + batches
    slots = groups * ROWS_PER_GROUP

    cu = cu_seqlens_q.to(torch.int64)
    per_batch = (cu[1:] - cu[:-1] + ROWS_PER_GROUP - 1) // ROWS_PER_GROUP
    group_base = torch.cumsum(per_batch, 0) - per_batch
    r = torch.arange(rows, device=dev)
    b = row_to_batch.to(torch.int64)
    n = r - cu[b]
    g = group_base[b] + n // ROWS_PER_GROUP
    slot = g * ROWS_PER_GROUP + n % ROWS_PER_GROUP

    slot_meta = torch.zeros((3, slots), dtype=torch.int32, device=dev)
    slot_meta[0].fill_(-1)
    slot_meta[0, slot] = r.to(torch.int32)
    slot_meta[1, slot] = local_starts.to(torch.int32)
    slot_meta[2, slot] = local_ends.to(torch.int32)

    q_fp4 = torch.zeros((slots, packed_dim), dtype=torch.uint8, device=dev)
    q_fp4[slot] = q_rows_fp4
    natural = torch.zeros((slots, chunks), dtype=torch.uint8, device=dev)
    natural[slot] = q_rows_scale
    # [g, slot, chunk] -> [g, k_tile, chunk-in-tile, slot, pad=4], byte 0 live.
    k_tiles = chunks // KK_PER_KTILE
    q_scale = torch.zeros(
        (groups, k_tiles, KK_PER_KTILE, ROWS_PER_GROUP, 4),
        dtype=torch.uint8,
        device=dev,
    )
    q_scale[..., 0] = natural.view(
        groups, ROWS_PER_GROUP, k_tiles, KK_PER_KTILE
    ).permute(0, 2, 3, 1)

    ends = local_ends.to(torch.int32)
    group_to_batch = torch.zeros(groups, dtype=torch.int32, device=dev)
    group_to_batch.scatter_(0, g, row_to_batch.to(torch.int32))
    group_ends = torch.zeros(groups, dtype=torch.int32, device=dev)
    group_ends.scatter_reduce_(0, g, ends, "amax")
    group_starts = torch.full((groups,), 2**31 - 1, dtype=torch.int32, device=dev)
    group_starts.scatter_reduce_(0, g, local_starts.to(torch.int32), "amin")
    group_starts = torch.where(group_ends > 0, group_starts, 0)
    return (
        q_fp4.view(groups, ROWS_PER_GROUP, packed_dim),
        q_scale,
        slot_meta,
        group_to_batch,
        group_starts,
        group_ends,
    )


def flydsl_pa_mqa_page_max_fp4_rowpack_prefill_gfx942(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    slot_meta: torch.Tensor,
    group_to_batch: torch.Tensor,
    group_starts: torch.Tensor,
    group_ends: torch.Tensor,
    max_seq_len: int,
    out: torch.Tensor,
    *,
    weight_scale: float = 1.0,
    kv_page_stride: int | None = None,
    kv_scale_page_stride: int | None = None,
    kv_scale_offset: int = 0,
    block_k: int = 256,
    kv_block_size: int = 64,
    num_warps: int = DEFAULT_NUM_WARPS,
    parallel_unit_num: int | None = None,
    cta_info: torch.Tensor | None = None,
    total_ctas: int | None = None,
    stream: torch.cuda.Stream | None = None,
    q_fp8: bool = False,
) -> torch.Tensor:
    """Signed per-64-token-page maxima of row-packed prefill index scores.

    ``out`` must be ``-inf``-filled by the caller. ``kv_page_stride``,
    ``kv_scale_page_stride`` and ``kv_scale_offset`` (bytes) read a paged cache
    whose data and scales share one allocation. ``parallel_unit_num`` bounds
    the persistent grid (default four CTAs per CU, at least one per group).

    ``cta_info``/``total_ctas`` pass a schedule built by
    :func:`compute_prefill_schedule` (with ``block_k``) instead of building
    one here: a caller scoring many layers over the same groups builds it once,
    and a CUDA-graph caller keeps it at a stable address. Every page of every
    group's ``[start, end)`` window is then written, so a caller that reads only
    those pages may skip the ``-inf`` fill.

    ``q_fp8=True``: ``q_fp4`` holds ``[groups, 16, D]`` e4m3fnuz bytes (uint8
    or float8_e4m3fnuz) and ``q_scale`` ``[groups, 16]`` fp32 per-slot scales.
    """
    groups = q_fp4.shape[0]
    if tuple(q_fp4.shape[1:]) != (ROWS_PER_GROUP, q_fp4.shape[2]):
        raise ValueError(f"q_fp4 must be [groups, 16, D/2], got {tuple(q_fp4.shape)}")
    head_dim = q_fp4.shape[2] * (1 if q_fp8 else 2)
    if q_fp8 and (q_fp4.element_size() != 1 or not q_fp4.is_contiguous()):
        raise ValueError("fp8 q must be a contiguous 1-byte [groups, 16, D] tensor")
    if q_fp8 and (
        q_scale.dtype != torch.float32 or q_scale.numel() < groups * ROWS_PER_GROUP
    ):
        raise ValueError(f"fp8 q_scale must be fp32 [{groups}, 16]")
    slots = groups * ROWS_PER_GROUP
    if slot_meta.dtype != torch.int32 or tuple(slot_meta.shape) != (3, slots):
        raise ValueError(f"slot_meta must be int32 [3, {slots}]")
    if not slot_meta.is_contiguous():
        raise ValueError("slot_meta must be contiguous")
    pages = (max_seq_len + PAGE_TOKENS - 1) // PAGE_TOKENS
    if out.dtype != torch.float32 or out.dim() != 2 or out.stride(1) != 1:
        raise ValueError("out must be a row-major fp32 [rows, pages] tensor")
    if out.shape[1] < pages:
        raise ValueError(f"out needs >= {pages} page columns, got {out.shape[1]}")
    if out.numel() >= 1 << 29:
        raise ValueError("out must stay below 2 GiB (one buffer descriptor)")
    if groups == 0:
        return out

    if (cta_info is None) != (total_ctas is None):
        raise ValueError("Pass both cta_info and total_ctas, or neither.")
    if cta_info is None:
        if parallel_unit_num is None:
            props = torch.cuda.get_device_properties(q_fp4.device)
            parallel_unit_num = 4 * props.multi_processor_count
        parallel_unit_num = max(parallel_unit_num, groups)
        _, cta_info, n_ctas = compute_prefill_schedule(
            group_to_batch,
            group_starts,
            group_ends,
            block_k,
            parallel_unit_num,
            max_seq_len,
        )
    else:
        if total_ctas < groups or cta_info.shape[0] < total_ctas:
            raise ValueError(
                f"cta_info schedules {total_ctas} CTAs over {cta_info.shape[0]} "
                f"rows for {groups} groups"
            )
        n_ctas = total_ctas
    launcher, _ = compile_pa_mqa_page_max_fp4_rowpack_prefill_gfx942(
        block_k=block_k,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=block_tables.shape[1],
        num_warps=num_warps,
        head_dim=head_dim,
        kv_page_stride=kv_page_stride,
        kv_scale_page_stride=kv_scale_page_stride,
        kv_scale_offset=kv_scale_offset,
        q_fp8=q_fp8,
    )
    if stream is None:
        stream = torch.cuda.current_stream()
    launcher(
        out,
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        slot_meta,
        cta_info,
        out.stride(0),
        out.shape[1],
        slots,
        out.numel(),
        float(weight_scale),
        n_ctas,
        stream,
    )
    return out
