# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in gfx942 FP4 MQA decode specialization for very long contexts.

CDNA3 has no FP4 MFMA, so four dense FP8 ``MFMA(16,16,32)`` operations replace
each gfx950 scaled-FP4 step.  This specialization folds the Q E8M0 scale into
the decoded E4M3-FNUZ operand, decodes the CTA-invariant Q fragments once, and
shares them across its four waves through 8 KiB of LDS.  K scaling remains an
exact power-of-two F32 FMA after each MFMA.

Eligibility is explicit: every Q E8M0 byte must be in [125, 127], corresponding
to scales 2^-2 through 2^0.  In this interval every nonzero E2M1 magnitude has
enough E4M3 exponent headroom, so subtracting ``(127-scale) << 3`` from each
nonzero decoded byte is exact and cannot borrow between packed bytes.  Callers
with arbitrary E8M0 Q scales must use ``pa_mqa_logits_fp4_gfx942`` instead.

This is intentionally a separate API.  The general gfx942 kernel and explicit
caller-provided schedules remain unchanged; systems may opt into this path for
the TP2 / million-token regime after validating their Q-scale distribution.
"""

from __future__ import annotations

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import gpu, rocdl
from flydsl.expr.primitive import range_constexpr
from flydsl.expr.typing import Int32, T

from aiter.ops.triton.utils.device_info import get_num_sms

from .pa_mqa_logits_fp4_common import (
    _NON_WRITER_LANE_OFF,
    _i32_buffer,
    _load_vec4_i32,
    compute_varctx_schedule,
)

DEFAULT_HEADS = 64
DEFAULT_HEAD_DIM = 128
DEFAULT_NUM_WARPS = 4
MFMA_M = 16
MFMA_N = 16
WARP_SIZE = 64
DEFAULT_BLOCK_THREADS = DEFAULT_NUM_WARPS * WARP_SIZE  # 256

# One dense fp8 MFMA covers 32 fp4 elements (one 32-chunk) per K-step; a
# gfx950 fp4 k_tile (128 fp4) is 4 K-steps.
KK_PER_KTILE = 4


# ---------------------------------------------------------------------------
# fp4 e2m1 (packed) -> e4m3 FNUZ (byte-per-code) decode via v_perm_b32.
# ---------------------------------------------------------------------------


def _decode_fp4_dword(d):
    """4 packed e2m1 bytes -> (p01, p23): two dwords of e4m3 FNUZ codes, K ascending.

    v_perm_b32-based: 6 perms + 11 integer ops per dword (the ALU-only version
    was ~77 ops). Window order: perm_b32(src_hi, src_lo, sel) -- the first
    operand is the MSB half of the 8-byte permute window (CDNA3 BYTE_PERMUTE
    spec). The four LUT perms use constant windows indexed by nibble & 7 (the
    sign bit is masked off first: selector bit3 set would trigger sign-extend
    mode). The two interleave perms carry (dec_e, dec_o) as the window with
    constant selectors, replacing the 20-op shift/or pack.

    FNUZ truth: x in 0..7 -> 0x00,0x38,0x40,0x44,0x48,0x4C,0x50,0x54, then
    |0x80 for signed codes with x != 0. Negative zero (nibble 8) stays 0x00:
    FNUZ 0x80 is a NaN.
    """
    even = d & fx.Int32(0x07070707)
    odd = d.shrui(fx.Int32(4)) & fx.Int32(0x07070707)
    mag_e = fx.Int32(rocdl.perm_b32(fx.Int32(0x54504C48), fx.Int32(0x44403800), even))
    mag_o = fx.Int32(rocdl.perm_b32(fx.Int32(0x54504C48), fx.Int32(0x44403800), odd))
    nz_e = fx.Int32(rocdl.perm_b32(fx.Int32(0xFFFFFFFF), fx.Int32(0xFFFFFF00), even))
    nz_o = fx.Int32(rocdl.perm_b32(fx.Int32(0xFFFFFFFF), fx.Int32(0xFFFFFF00), odd))
    dec_e = mag_e | (((d & fx.Int32(0x08080808)) << fx.Int32(4)) & nz_e)
    dec_o = mag_o | ((d & fx.Int32(0x80808080)) & nz_o)
    p01 = fx.Int32(rocdl.perm_b32(dec_o, dec_e, fx.Int32(0x05010400)))
    p23 = fx.Int32(rocdl.perm_b32(dec_o, dec_e, fx.Int32(0x07030602)))
    return p01, p23


def _decode_fp4_dword_scaled(d, scale_byte):
    """Decode FP4 and fold an eligible Q E8M0 byte into FP8 exponents.

    This helper requires the public long-context API contract ``125 <= scale <=
    127``.  The bounds mean an exponent decrement of at most two, which keeps
    every nonzero E2M1 value normal in E4M3-FNUZ and prevents packed-byte borrow.
    """
    even = d & fx.Int32(0x07070707)
    odd = d.shrui(fx.Int32(4)) & fx.Int32(0x07070707)
    mag_e = fx.Int32(rocdl.perm_b32(fx.Int32(0x54504C48), fx.Int32(0x44403800), even))
    mag_o = fx.Int32(rocdl.perm_b32(fx.Int32(0x54504C48), fx.Int32(0x44403800), odd))
    nz_e = fx.Int32(rocdl.perm_b32(fx.Int32(0xFFFFFFFF), fx.Int32(0xFFFFFF00), even))
    nz_o = fx.Int32(rocdl.perm_b32(fx.Int32(0xFFFFFFFF), fx.Int32(0xFFFFFF00), odd))
    dec_e = mag_e | (((d & fx.Int32(0x08080808)) << fx.Int32(4)) & nz_e)
    dec_o = mag_o | ((d & fx.Int32(0x80808080)) & nz_o)
    exp_sub = (fx.Int32(127) - scale_byte) << fx.Int32(3)
    exp_sub4 = exp_sub * fx.Int32(0x01010101)
    dec_e = dec_e - (exp_sub4 & nz_e)
    dec_o = dec_o - (exp_sub4 & nz_o)
    p01 = fx.Int32(rocdl.perm_b32(dec_o, dec_e, fx.Int32(0x05010400)))
    p23 = fx.Int32(rocdl.perm_b32(dec_o, dec_e, fx.Int32(0x07030602)))
    return p01, p23


def _frag_i64_from_pp(p01, p23):
    """One lane's 8-byte fp8 fragment (K ascending) as a rank-1 register tensor."""
    raw = fx.Int64(fx.Uint32(p01)) | (fx.Int64(fx.Uint32(p23)) << fx.Int32(32))
    frag = fx.make_rmem_tensor(1, fx.Int64)
    frag.store(fx.Vector.from_elements([fx.Int64(raw)]))
    return frag


def build_pa_mqa_logits_fp4_gfx942_longctx_module(
    block_k=128,
    kv_block_size=16,
    max_blocks_per_seq=256,
    num_warps=DEFAULT_NUM_WARPS,
    next_n=1,
    heads=DEFAULT_HEADS,
    head_dim=DEFAULT_HEAD_DIM,
    relu=True,
    kv_page_stride=None,
    kv_scale_page_stride=None,
    kv_scale_offset=0,
):
    block_threads_k = num_warps * WARP_SIZE
    head_dim_packed = head_dim // 2
    m_tiles = heads // MFMA_M
    k_tiles = head_dim // 128  # outer K-loop iters (128 fp4 each = 4 K-steps)
    assert (
        head_dim % 128 == 0
    ), f"head_dim must be a multiple of 128 (fp4 k_tile), got {head_dim}"
    assert heads % MFMA_M == 0, f"heads must be a multiple of {MFMA_M}, got {heads}"
    assert m_tiles <= 8, f"m_tiles={m_tiles} > 8 not supported. Use heads <= 128."

    N_TILES = block_k // MFMA_N
    assert (
        N_TILES % num_warps == 0
    ), f"block_k={block_k} → N_TILES={N_TILES} must be multiple of num_warps={num_warps}"
    N_TILES_PER_WARP = N_TILES // num_warps

    assert kv_block_size % MFMA_N == 0, (
        f"kv_block_size={kv_block_size} must be a multiple of MFMA_N={MFMA_N}; "
        f"sub-tile pages would require splitting one MFMA over multiple page lookups"
    )
    assert (
        block_k % kv_block_size == 0
    ), f"block_k={block_k} must be a multiple of kv_block_size={kv_block_size}"
    assert (
        kv_block_size == 64
    ), "gfx942 packed KV loads currently require kv_block_size=64"
    TILES_PER_BLOCK = kv_block_size // MFMA_N
    N_PHYS = (N_TILES_PER_WARP + TILES_PER_BLOCK - 1) // TILES_PER_BLOCK
    assert N_TILES_PER_WARP == 4 and N_PHYS == 1, (
        "gfx942 packed KV loads require four 16-token tiles per wave within "
        "one physical page; expected block_k == 64 * num_warps and "
        "kv_block_size >= 64"
    )

    _stride_q_next_n = heads * head_dim_packed  # bytes per next_n slice
    _stride_bt = max_blocks_per_seq

    _kv_chunk_bytes = 16
    _stride_kv_ktile = 4 * kv_block_size * _kv_chunk_bytes  # bytes per K_TILE block
    _stride_kv_block = k_tiles * _stride_kv_ktile  # bytes per phys block
    # KV_scale: [block_id, K_TILES, K_chunks=4, block_size]
    _stride_kvs_ktile = 4 * kv_block_size  # bytes per K_TILE block
    _stride_kvs_block = k_tiles * _stride_kvs_ktile
    if kv_page_stride is not None:
        _stride_kv_block = kv_page_stride
    if kv_scale_page_stride is not None:
        _stride_kvs_block = kv_scale_page_stride

    # Q_scale: [B, NN, K_TILES, K_chunks=4, 16, qs_pad];
    # byte [..][c][j][mi] = e8m0 of head (mi*MFMA_M + j) at 32-chunk c.
    QS_DW = (m_tiles + 3) // 4
    qs_pad = QS_DW * 4
    shared_bytes = k_tiles * m_tiles * KK_PER_KTILE * WARP_SIZE * 8 + heads * 2
    assert shared_bytes <= 65536, (
        f"long-context shared-memory requirement {shared_bytes} bytes exceeds "
        "the 64 KiB workgroup limit; reduce heads or head_dim"
    )

    if N_PHYS == 1:

        def _phys_to_list(phys_v):
            return [phys_v] * N_TILES_PER_WARP

    else:

        def _phys_to_list(phys_v):
            return [
                fx.Vector(phys_v)[nt // TILES_PER_BLOCK]
                for nt in range(N_TILES_PER_WARP)
            ]

    @flyc.kernel
    def pa_mqa_logits_fp4_gfx942_longctx_kernel(
        out_logits_ptr: fx.Tensor,
        q_ptr: fx.Tensor,
        q_scale_ptr: fx.Tensor,
        kv_cache_ptr: fx.Tensor,
        kv_scale_ptr: fx.Tensor,
        kv_indices_ptr: fx.Tensor,
        weights_ptr: fx.Tensor,
        cta_info_ptr: fx.Tensor,  # [total_ctas, 4] i32: [batch_packed, chunk_start, chunk_count, ctx_len]
        stride_out_batch: Int32,
        weight_scale: fx.Float32,
    ):
        tid = gpu.thread_idx.x
        pid = gpu.block_idx.x
        # One scaled-Q fragment set is identical for every warp in the CTA.
        # Decode it once with warp 0 and publish packed i64 fragments in LDS.
        q_bytes = k_tiles * m_tiles * KK_PER_KTILE * WARP_SIZE * 8
        shared = fx.SharedAllocator().allocate(shared_bytes, 16)
        q_lds_ptr = fx.recast_iter(fx.Int64, shared.peek().ptr)
        w_lds_ptr = fx.recast_iter(
            fx.BFloat16, fx.add_offset(shared.peek().ptr, q_bytes)
        )
        w_lds = fx.make_view(w_lds_ptr, fx.make_layout(heads, 1))
        q_lds = fx.make_view(
            q_lds_ptr,
            fx.make_layout(k_tiles * m_tiles * KK_PER_KTILE * WARP_SIZE, 1),
        )

        warp_id = tid >> 6
        lane_id = tid % WARP_SIZE
        lane_mod_16 = lane_id & 15
        lane_div_16 = (lane_id >> 4) & 3

        cta_info_bt = fx.rocdl.make_buffer_tensor(cta_info_ptr)
        cta_info_vec = fx.Vector(_load_vec4_i32(cta_info_bt, pid * fx.Int32(4)))

        kv_bt = _i32_buffer(kv_cache_ptr, width=1)
        kvs_bt = _i32_buffer(kv_scale_ptr, width=1)
        bt_bt = _i32_buffer(kv_indices_ptr, width=1)
        q_bt = _i32_buffer(q_ptr, width=1)
        qs_bt = _i32_buffer(q_scale_ptr, width=1)

        ZERO_F = fx.Float32(0.0)
        c0_i32 = fx.Int32(0)

        # Wave-uniform by construction, but they arrive in VGPRs via the buffer
        # load; the V# below needs num_records in an SGPR or the store is wrapped
        # in a waterfall loop.
        def _uniform(v):
            return fx.Int32(fx.rocdl.readfirstlane(T.i32, v.ir_value()))

        batch_packed = _uniform(cta_info_vec[0])
        chunk_start = cta_info_vec[1]
        chunk_count = cta_info_vec[2]
        context_len = _uniform(cta_info_vec[3])

        # out row base folded into an f32 global pointer: sizeof(f32) = 4, so the
        # per-token store offset below stays small (no i32 overflow for large
        # stride_out_batch * batch_packed).
        _row_elems = fx.Int64(batch_packed) * fx.Int64(stride_out_batch)
        out_base = fx.add_offset(fx.get_iter(out_logits_ptr), _row_elems)

        pid_b = batch_packed // fx.Int32(next_n)
        pid_next_n = batch_packed % fx.Int32(next_n)

        # A V# spanning exactly this row's visible range: num_records then IS
        # the `out_token + mask_off < context_len` test, in hardware. Lanes
        # 16..63 hold redundant copies of the butterfly result and must not
        # write; `lane_div_16 * win_len` puts them past num_records. The clamp
        # matters: a negative length wraps to a huge unsigned bound.
        _mask_off = fx.Int32(next_n - 1) - pid_next_n
        _win_raw = context_len - _mask_off
        win_len = (_win_raw < fx.Int32(0)).select(fx.Int32(0), _win_raw)
        out_win = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.recast_iter(
                    fx.PointerType.get(T.f32, out_base.memspace, 4), out_base
                ),
                fx.make_layout((win_len, 1), (1, 1)),
            ),
            max_size=False,
            num_records_bytes=win_len * fx.Int32(4),
        )
        out_lane_off = lane_mod_16 + (lane_div_16 > fx.Int32(0)).select(
            fx.Int32(_NON_WRITER_LANE_OFF), fx.Int32(0)
        )
        out_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 1)
        out_reg_ty = fx.MemRefType.get(
            T.f32, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
        )
        out_reg_lay = fx.make_layout(1, 1)

        # Dense fp8 16x16x32 MMA (FNUZ); 4 K-steps replace one gfx950 fp4 step.
        mfma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))

        def _e8m0_byte_to_f32(b):
            # Callers have already isolated one byte; avoid a duplicate mask in
            # the inner MFMA scale path.
            v = fx.Vector.from_elements([b << fx.Int32(23)], dtype=fx.Int32).bitcast(
                fx.Float32
            )
            return v[0]

        # ---- A-fragments: per (k_tile, mi, kk), one packed dword -> i64 frag ----
        # q byte offset = row*D/2 + k_tile*64 + kk*16 + (l//16)*4 (fp4 pair bytes)
        q_base = (pid_b * fx.Int32(next_n) + pid_next_n) * fx.Int32(_stride_q_next_n)
        qs_base = (pid_b * fx.Int32(next_n) + pid_next_n) * fx.Int32(
            k_tiles * 64 * qs_pad
        )
        W_buf = fx.rocdl.make_buffer_tensor(weights_ptr)
        w_row = fx.slice(W_buf, (batch_packed, None))
        if warp_id == fx.Int32(0):
            for h_base in range_constexpr(0, heads, WARP_SIZE):
                h = fx.Int32(h_base) + lane_id
                if h < fx.Int32(heads):
                    w_lds[h] = w_row[h]

        # Cooperatively decode CTA-invariant Q: wave w owns head tiles
        # w, w+num_warps, ... and publishes disjoint LDS slots.
        for mi_base in range_constexpr(0, m_tiles, num_warps):
            mi_idx = fx.Int32(mi_base) + warp_id
            if mi_idx < fx.Int32(m_tiles):
                mi_group = mi_idx // fx.Int32(4)
                mi_local = mi_idx & fx.Int32(3)
                q_row = mi_idx * fx.Int32(MFMA_M) + lane_mod_16
                q_row_base = q_base + q_row * fx.Int32(head_dim_packed)
                for k_tile in range_constexpr(k_tiles):
                    for kk in range_constexpr(KK_PER_KTILE):
                        qs_off = (
                            qs_base
                            + fx.Int32(k_tile * 64 * qs_pad)
                            + fx.Int32(kk * 16 * qs_pad)
                            + lane_mod_16 * fx.Int32(qs_pad)
                        )
                        qs_dw = fx.Int32(qs_bt[(qs_off // fx.Int32(4)) + mi_group])
                        qs_byte = (qs_dw >> (mi_local * fx.Int32(8))) & fx.Int32(0xFF)
                        off = (
                            q_row_base
                            + fx.Int32(k_tile * 64 + kk * 16)
                            + lane_div_16 * fx.Int32(4)
                        )
                        d = fx.Int32(q_bt[off // fx.Int32(4)])
                        p01, p23 = _decode_fp4_dword_scaled(d, qs_byte)
                        raw = fx.Int64(fx.Uint32(p01)) | (
                            fx.Int64(fx.Uint32(p23)) << fx.Int32(32)
                        )
                        slot = (
                            (
                                (fx.Int32(k_tile * m_tiles) + mi_idx)
                                * fx.Int32(KK_PER_KTILE)
                                + fx.Int32(kk)
                            )
                            * fx.Int32(WARP_SIZE)
                        ) + lane_id
                        q_lds[slot] = raw
        gpu.barrier()

        q_a_ops = []
        for k_tile in range_constexpr(k_tiles):
            q_a_ops_kt = []
            for mi_idx in range_constexpr(m_tiles):
                frags = []
                for kk in range_constexpr(KK_PER_KTILE):
                    slot = (
                        ((k_tile * m_tiles + mi_idx) * KK_PER_KTILE + kk) * WARP_SIZE
                    ) + lane_id
                    frag = fx.make_rmem_tensor(1, fx.Int64)
                    frag.store(fx.Vector.from_elements([q_lds[slot]], dtype=fx.Int64))
                    frags.append(frag)
                q_a_ops_kt.append(frags)
            q_a_ops.append(q_a_ops_kt)

        # The producer wave published the CTA-invariant BF16 weight row in LDS.
        # Every lane consumes the same four-head group as the original copy layout.
        w_per_lane = []
        for mi_idx in range_constexpr(m_tiles):
            first_h = fx.Int32(mi_idx * MFMA_M) + lane_div_16 * fx.Int32(4)
            w_per_lane.append(
                fx.Vector.from_elements(
                    [w_lds[first_h + fx.Int32(i)] for i in range_constexpr(4)],
                    dtype=fx.BFloat16,
                ).to(fx.Float32)
            )

        # ── Step 3: prologue + N-1 prefetch loop + epilogue ──
        # NOTE: flydsl's AST rewriter lowers in-body `if` statements to runtime
        # scf.if dispatches (they are NOT trace-time python conditionals), so
        # the load helpers stay monomorphic; the general address math below is
        # used for all configs.

        aligned_wave_page = (
            kv_block_size == 64
            and N_TILES_PER_WARP == 4
            and block_k % kv_block_size == 0
        )

        def _load_phys(c_i32_arg):
            if fx.const_expr(aligned_wave_page):
                chunk_page = (chunk_start + c_i32_arg) * fx.Int32(
                    block_k // kv_block_size
                )
                bi_base = chunk_page + warp_id
            else:
                ni_base = warp_id * fx.Int32(N_TILES_PER_WARP)
                token_global_base = (
                    (chunk_start + c_i32_arg) * fx.Int32(block_k)
                    + ni_base * fx.Int32(MFMA_N)
                    + lane_mod_16
                )
                bi_base = token_global_base // kv_block_size
            # A chunk's tail can run past the row's last mapped page; the
            # windowed store drops those tokens, but the page id itself must
            # stay inside the table or the KV load faults.
            bi_base = fx.min(bi_base, fx.Int32(_stride_bt - 1))
            phys_vec = bt_bt[pid_b * _stride_bt + bi_base]
            return _phys_to_list(phys_vec)

        def _prefetch_chunk(c_i32_arg, phys_list):
            assert N_TILES_PER_WARP == 4, "packed kvs assumes NTPW=4"
            assert N_PHYS == 1, "packed kvs assumes N_PHYS=1 (NTPW nts share one phys)"

            kv_list = []  # per (nt, k_tile, kk): packed-fp4 dword for this lane
            kvs_list = []  # per (k_tile, kk): packed sb dword (4 nt bytes)

            # ---- KV loads: contiguous + cross-lane permute. Lane l loads
            # flat dword (token-in-tile = l//4, group = l%4) of MFMA tile
            # nt's 16-token span, so the 64 lanes of one load instruction
            # cover 256B densely (the old mapping strided 16 dwords at 16B
            # token stride per lane-group = 4x sector over-request). The
            # B-fragment owner (token l%16, group l//16) then pulls its
            # dword from lane (l%16)*4 + (l//16) via ds_bpermute — same
            # dword values in the same order as the old mapping.
            # No address clamping — OOB tokens read garbage that is later
            # overwritten by NEG_INF via in_bounds.select on the store path.
            perm_src_byte = (lane_mod_16 * fx.Int32(4) + lane_div_16) * fx.Int32(4)
            for nt in range_constexpr(N_TILES_PER_WARP):
                ni_c = warp_id * fx.Int32(N_TILES_PER_WARP) + fx.Int32(nt)
                token_base_c = (chunk_start + c_i32_arg) * fx.Int32(
                    block_k
                ) + ni_c * fx.Int32(MFMA_N)
                phys_block_c = phys_list[nt]
                for k_tile in range_constexpr(k_tiles):
                    for kk in range_constexpr(KK_PER_KTILE):
                        if fx.const_expr(aligned_wave_page):
                            tok_c = fx.Int32(nt * MFMA_N) + lane_id // fx.Int32(4)
                        else:
                            tok_c = (
                                token_base_c + (lane_id // fx.Int32(4))
                            ) % fx.Int32(kv_block_size)
                        kv_off_bytes_c = (
                            fx.Int64(phys_block_c) * fx.Int64(_stride_kv_block)
                            + fx.Int64(k_tile * _stride_kv_ktile)
                            + fx.Int64(kk * kv_block_size * _kv_chunk_bytes)
                            + fx.Int64(tok_c) * fx.Int64(_kv_chunk_bytes)
                            + fx.Int64(lane_id % fx.Int32(4)) * fx.Int64(4)
                        )
                        raw_c = kv_bt[kv_off_bytes_c // fx.Int64(4)]
                        kv_list.append(
                            fx.Int32(rocdl.ds_bpermute(T.i32, perm_src_byte, raw_c))
                        )

            # ---- KVS packed load: 1 dword per (k_tile, kk) covering 4 nts ----
            # Byte layout: [phys][k_tile][kk][n*4 + nt]; 4 consecutive nt bytes.
            phys_shared = phys_list[0]
            for k_tile in range_constexpr(k_tiles):
                for kk in range_constexpr(KK_PER_KTILE):
                    kvs_packed_off_bytes = (
                        fx.Int64(phys_shared) * fx.Int64(_stride_kvs_block)
                        + fx.Int64(kv_scale_offset)
                        + fx.Int64(k_tile * _stride_kvs_ktile)
                        + fx.Int64(kk * kv_block_size)
                        + fx.Int64(lane_mod_16) * fx.Int64(N_TILES_PER_WARP)
                    )
                    kvs_list.append(kvs_bt[kvs_packed_off_bytes // fx.Int64(4)])

            return kv_list, kvs_list

        def _frag_from_i64(v):
            """Rebuild the 1x i64 rmem MFMA fragment from a carried i64 value."""
            frag = fx.make_rmem_tensor(1, fx.Int64)
            frag.store(fx.Vector.from_elements([v], dtype=fx.Int64))
            return frag

        def _decode_frags(kv_list_in, kvs_list_in):
            """Decode KV normally; only Q scales are folded into operands."""
            frags = []
            for i in range_constexpr(N_KV):
                p01, p23 = _decode_fp4_dword(fx.Int32(kv_list_in[i]))
                frags.append(
                    fx.Int64(fx.Uint32(p01))
                    | (fx.Int64(fx.Uint32(p23)) << fx.Int32(32))
                )
            return frags

        def _issue_nt_mfmas_frags(frags_in, kvs_list_in, nt):
            acc_s = [[ZERO_F for _ in range(4)] for _ in range(m_tiles)]
            for k_tile in range_constexpr(k_tiles):
                for kk in range_constexpr(KK_PER_KTILE):
                    b_frag = _frag_from_i64(
                        frags_in[((nt * k_tiles) + k_tile) * KK_PER_KTILE + kk]
                    )
                    sb_packed = fx.Int32(kvs_list_in[k_tile * KK_PER_KTILE + kk])
                    if fx.const_expr(nt == 0):
                        sb_byte = sb_packed & fx.Int32(0xFF)
                    elif fx.const_expr(nt == 3):
                        sb_byte = sb_packed.shrui(fx.Int32(24))
                    else:
                        sb_byte = sb_packed.shrui(fx.Int32(8 * nt)) & fx.Int32(0xFF)
                    sb_f32 = _e8m0_byte_to_f32(sb_byte)
                    for mi_idx in range_constexpr(m_tiles):
                        c_frag = fx.make_rmem_tensor(4, fx.Float32)
                        c_frag.store(fx.Vector.filled(4, 0.0, fx.Float32))
                        fx.gemm(
                            mfma_atom,
                            c_frag,
                            q_a_ops[k_tile][mi_idx][kk],
                            b_frag,
                            c_frag,
                        )
                        partial = c_frag.load()
                        for t in range_constexpr(4):
                            acc_s[mi_idx][t] = fx.fma(
                                partial[t], sb_f32, acc_s[mi_idx][t]
                            )
            return [
                fx.Vector.from_elements(acc_s[mi], dtype=fx.Float32)
                for mi in range(m_tiles)
            ]

        def _post_process_nt(accs, nt, c_i32_arg):
            zero = fx.full(4, 0.0, fx.Float32)
            ni_warp = warp_id * fx.Int32(N_TILES_PER_WARP) + fx.Int32(nt)
            token_base = (chunk_start + c_i32_arg) * fx.Int32(
                block_k
            ) + ni_warp * fx.Int32(MFMA_N)

            thread_sum = ZERO_F
            for mi_idx in range_constexpr(m_tiles):
                relu_v = (
                    fx.Vector(accs[mi_idx]).maximumf(zero)
                    if relu
                    else fx.Vector(accs[mi_idx])
                )
                prod_v = relu_v * fx.Vector(w_per_lane[mi_idx])
                for elem in [0, 1, 2, 3]:
                    thread_sum = thread_sum + prod_v[elem]

            lane_i32 = fx.Int32(lane_id)

            def _bperm_xor_add(val, sh):
                peer_byte = (lane_i32 ^ fx.Int32(sh)) * 4
                peer_i32 = fx.Int32(
                    rocdl.ds_bpermute(T.i32, peer_byte, val.bitcast(fx.Int32))
                )
                return val + peer_i32.bitcast(fx.Float32)

            thread_sum = _bperm_xor_add(thread_sum, 16)
            thread_sum = _bperm_xor_add(thread_sum, 32)
            thread_sum = thread_sum * weight_scale

            # Context bound and writer-lane guard both live in the V# built
            # above; nothing is tested here.
            off = token_base + out_lane_off
            r_out = fx.memref_alloca(out_reg_ty, out_reg_lay)
            fx.memref_store_vec(
                fx.Vector.from_elements([thread_sum], dtype=fx.Float32), r_out
            )
            fx.copy(out_atom, r_out, fx.slice(out_win, (off, None)))

        def _compute_chunk_frags(frags_in, kvs_list_in, c_i32_arg):
            """Process chunk c from pre-decoded B-fragments."""
            assert (
                N_TILES_PER_WARP == 4
            ), "pipelined-nt structure currently hardcoded for NTPW=4"

            # nt=0 MFMA first — its latency overlaps with the nt=1 issue.
            accs_nt0 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 0)

            # nt=1 MFMA early → its 16-cycle latency overlaps with nt=0 post-process
            accs_nt1 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 1)
            _post_process_nt(accs_nt0, 0, c_i32_arg)

            # nt=2 MFMA early → overlaps with nt=1 post-process
            accs_nt2 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 2)
            _post_process_nt(accs_nt1, 1, c_i32_arg)

            # nt=3 MFMA early → overlaps with nt=2 post-process
            accs_nt3 = _issue_nt_mfmas_frags(frags_in, kvs_list_in, 3)
            _post_process_nt(accs_nt2, 2, c_i32_arg)

            # Last nt: no further MFMA to overlap with, just post-process.
            _post_process_nt(accs_nt3, 3, c_i32_arg)

        # === Prologue ===
        N_KV = k_tiles * N_TILES_PER_WARP * KK_PER_KTILE
        N_KVS = k_tiles * KK_PER_KTILE
        last_c_i32 = chunk_count - fx.Int32(1)

        phys_pre = _load_phys(c0_i32)
        kv_pre, kvs_pre = _prefetch_chunk(c0_i32, phys_pre)
        frags_pre = _decode_frags(kv_pre, kvs_pre)
        phys_next_pre = _load_phys(fx.min(fx.Int32(1), last_c_i32))

        # === Main loop: chunk_count - 1 iterations ===
        # Loop state: pre-decoded B-fragments + packed scales for the CURRENT
        # chunk (the e2m1 decode ran in the previous iteration, overlapped with
        # that chunk's MFMA chain), plus the phys row for chunk c+1. This drops
        # the e2m1 decode from the per-chunk MFMA+scale critical path.
        chunk_count_minus_1_i32 = chunk_count - fx.Int32(1)
        chunk_count_minus_1_idx = fx.Int64(chunk_count_minus_1_i32)
        init_args = list(frags_pre) + list(kvs_pre) + list(phys_next_pre)
        for c_idx, state in range(0, chunk_count_minus_1_idx, 1, init=init_args):
            frags_cur = [state[i] for i in range(N_KV)]
            kvs_cur_list = [state[N_KV + i] for i in range(N_KVS)]
            phys_next_list = [state[N_KV + N_KVS + i] for i in range(N_TILES_PER_WARP)]
            c_idx_i32 = fx.Int32(c_idx)
            c_next_i32 = c_idx_i32 + 1
            c_next_next_i32 = fx.min(c_next_i32 + 1, last_c_i32)

            # Issue KV loads for chunk c+1 first so their latency overlaps the
            # current chunk's MFMA chain.
            kv_next, kvs_next = _prefetch_chunk(c_next_i32, phys_next_list)

            # Issue phys load for chunk c+2.
            phys_next_next_list = _load_phys(c_next_next_i32)

            # Compute current chunk from pre-decoded fragments.
            _compute_chunk_frags(frags_cur, kvs_cur_list, c_idx_i32)

            # Decode-in-prefetch: decode chunk c+1's B-fragments now — while
            # this chunk's MFMAs retire — instead of inside chunk c+1's compute.
            frags_next = _decode_frags(kv_next, kvs_next)

            results = yield (
                list(frags_next) + list(kvs_next) + list(phys_next_next_list)
            )

        # === Epilogue: process last chunk (chunk_count - 1) ===
        frags_last = [results[i] for i in range(N_KV)]
        kvs_last_list = [results[N_KV + i] for i in range(N_KVS)]
        _compute_chunk_frags(frags_last, kvs_last_list, last_c_i32)

    # Return the actual block threads count for the launcher (so the test can
    # use the right block dim when num_warps != module-level default).
    return pa_mqa_logits_fp4_gfx942_longctx_kernel, block_threads_k


# ============================================================================
# Cached compile + public host API
# ============================================================================


@lru_cache(maxsize=32)
def compile_pa_mqa_logits_fp4_gfx942_longctx(
    *,
    block_k: int = 256,
    kv_block_size: int = 64,
    max_blocks_per_seq: int = 256,
    num_warps: int = DEFAULT_NUM_WARPS,
    next_n: int = 1,
    heads: int = DEFAULT_HEADS,
    head_dim: int = DEFAULT_HEAD_DIM,
    relu: bool = True,
    kv_page_stride: int | None = None,
    kv_scale_page_stride: int | None = None,
    kv_scale_offset: int = 0,
):
    kfn, block_threads = build_pa_mqa_logits_fp4_gfx942_longctx_module(
        block_k=block_k,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        num_warps=num_warps,
        next_n=next_n,
        heads=heads,
        head_dim=head_dim,
        relu=relu,
        kv_page_stride=kv_page_stride,
        kv_scale_page_stride=kv_scale_page_stride,
        kv_scale_offset=kv_scale_offset,
    )

    @flyc.jit
    def launch_pa_mqa_logits_fp4_gfx942_longctx(
        out,
        q,
        qs,
        kv,
        kvs,
        bt,
        w,
        cta_info_,
        stride_out: fx.Int32,
        weight_scale: fx.Float32,
        gx: fx.Int32,
        stream: fx.Stream,
    ):
        gxi = fx.Int64(gx)
        kfn(out, q, qs, kv, kvs, bt, w, cta_info_, stride_out, weight_scale).launch(
            grid=(gxi,), block=(block_threads, 1, 1), stream=stream
        )

    return launch_pa_mqa_logits_fp4_gfx942_longctx, block_threads


def recommended_gfx942_fp4_longctx_config(
    max_seq_len: int,
    *,
    batch_size: int = 1,
    next_n: int = 1,
    num_sms: int | None = None,
) -> dict[str, int]:
    """Return the validated gfx942 geometry and persistent-grid size.

    This helper is side-effect free: serving integrations can build and cache
    ``cta_info`` outside their timed decode loop. Explicit kernel arguments and
    caller-provided schedules remain authoritative.
    """
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if next_n <= 0:
        raise ValueError(f"next_n must be positive, got {next_n}")
    if num_sms is not None and num_sms <= 0:
        raise ValueError(f"num_sms must be positive, got {num_sms}")

    block_k = 448 if 65536 < max_seq_len <= 262144 else 256
    num_warps = 7 if block_k == 448 else DEFAULT_NUM_WARPS
    total_chunks = max(1, (max_seq_len + block_k - 1) // block_k)
    total_work = total_chunks * batch_size * next_n
    if num_sms is None:
        num_sms = get_num_sms()
    if max_seq_len <= 65536:
        parallel_unit_num = total_work
    elif block_k == 448:
        chunks_per_cta = 1 if max_seq_len <= 131072 else 2
        target_ctas = max(1, (total_work + chunks_per_cta - 1) // chunks_per_cta)
        parallel_unit_num = min(
            total_work,
            ((target_ctas + num_sms - 1) // num_sms) * num_sms,
        )
    else:
        parallel_unit_num = min(total_work, 2 * num_sms)
    return {
        "block_k": block_k,
        "num_warps": num_warps,
        "parallel_unit_num": parallel_unit_num,
    }


def flydsl_pa_mqa_logits_fp4_gfx942_longctx(
    q_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_scale: torch.Tensor,
    block_tables: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor | None,
    max_seq_len: int,
    *,
    weight_scale: float = 1.0,
    kv_page_stride: int | None = None,
    kv_scale_page_stride: int | None = None,
    kv_scale_offset: int = 0,
    relu: bool = True,
    next_n: int = 1,
    block_k: int = 256,
    kv_block_size: int = 64,
    num_warps: int = DEFAULT_NUM_WARPS,
    parallel_unit_num: int | None = None,
    out: torch.Tensor | None = None,
    cta_info: torch.Tensor | None = None,
    total_ctas: int | None = None,
    row_ends: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int | None = None,
    pages_per_block: int = 1,
    plan: object | None = None,
    out_offsets: torch.Tensor | None = None,
    stream: torch.cuda.Stream | None = None,
    validate_q_scale: bool = True,
) -> torch.Tensor:
    """Run the opt-in long-context specialization.

    Used ``q_scale`` entries must contain E8M0 bytes in the inclusive range
    [125, 127]. Validation is safe by default and synchronizes with the device.
    A serving integration may set ``validate_q_scale=False`` only after its
    producer has been qualified. Arbitrary E8M0 scales are supported by
    :func:`flydsl_pa_mqa_logits_fp4_gfx942` in the general module. When an
    external schedule and ``out`` are supplied, the caller must initialize
    positions outside every scheduled context to ``-inf`` before launch.
    ``relu`` and the ``kv_*page_stride`` / ``kv_scale_offset`` layout arguments
    are as in :func:`flydsl_pa_mqa_logits_fp4_gfx942`.
    """
    if query_start_loc is not None:
        raise RuntimeError(
            "pa_mqa_logits_fp4_gfx942: the ragged-rows (rowgroup) path is not "
            "ported to gfx942"
        )
    if (
        row_ends is not None
        or plan is not None
        or pages_per_block != 1
        or out_offsets is not None
    ):
        raise RuntimeError(
            "pa_mqa_logits_fp4_gfx942: row_ends / plan / pages_per_block / "
            "out_offsets (ragged rows) are not ported to gfx942"
        )
    batch_size, q_next_n, heads, head_dim_packed = q_fp4.shape
    head_dim = head_dim_packed * 2
    m_tiles = heads // MFMA_M
    k_tiles = head_dim // 128
    qs_pad = ((m_tiles + 3) // 4) * 4
    expected_q_scale_shape = (batch_size, q_next_n, k_tiles, 4, 16, qs_pad)
    if q_scale.dtype != torch.uint8 or tuple(q_scale.shape) != expected_q_scale_shape:
        raise ValueError(
            "q_scale must be contiguous uint8 with shape "
            f"{expected_q_scale_shape}, got dtype={q_scale.dtype}, "
            f"shape={tuple(q_scale.shape)}"
        )
    if not q_scale.is_contiguous():
        raise ValueError("q_scale must be contiguous")
    if validate_q_scale:
        used_scale_bytes = q_scale[..., :m_tiles]
        scale_min = int(used_scale_bytes.min().item())
        scale_max = int(used_scale_bytes.max().item())
        if scale_min < 125 or scale_max > 127:
            raise ValueError(
                "gfx942 long-context Q-scale folding requires E8M0 bytes in "
                f"[125, 127], observed [{scale_min}, {scale_max}]. Use the "
                "general gfx942 FP4 MQA kernel for arbitrary E8M0 scales."
            )
    max_blocks_per_seq = block_tables.shape[1]
    if q_next_n != next_n:
        raise ValueError(f"q_fp4 next_n dim ({q_next_n}) != next_n arg ({next_n}).")

    if (cta_info is None) != (total_ctas is None):
        raise ValueError("Pass both cta_info and total_ctas, or neither.")
    schedule_internal = cta_info is None
    if schedule_internal:
        if parallel_unit_num is None:
            total_chunks = max(1, (max_seq_len + block_k - 1) // block_k)
            num_sms = get_num_sms()
            total_work = total_chunks * batch_size * next_n
            target_ctas = max(1, (total_work + 6) // 7)
            parallel_unit_num = min(
                total_work,
                ((target_ctas + num_sms - 1) // num_sms) * num_sms,
            )
        _, cta_info, total_ctas = compute_varctx_schedule(
            context_lens, block_k, parallel_unit_num, max_seq_len, next_n=next_n
        )

    if out is None:
        out = torch.full(
            (batch_size * next_n, max_seq_len),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp4.device,
        )
    elif schedule_internal:
        out.fill_(float("-inf"))

    launcher, _ = compile_pa_mqa_logits_fp4_gfx942_longctx(
        block_k=block_k,
        kv_block_size=kv_block_size,
        max_blocks_per_seq=max_blocks_per_seq,
        num_warps=num_warps,
        next_n=next_n,
        heads=heads,
        head_dim=head_dim,
        relu=relu,
        kv_page_stride=kv_page_stride,
        kv_scale_page_stride=kv_scale_page_stride,
        kv_scale_offset=kv_scale_offset,
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
        weights,
        cta_info,
        out.stride(0),
        float(weight_scale),
        total_ctas,
        stream,
    )
    return out
