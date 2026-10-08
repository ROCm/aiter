# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""gfx942 a8w4 stage1: MXFP8 A x MXFP4 W -> fp8 MFMA, MX scales applied after the MFMA.

gfx942 has no FP4 MFMA, so the a16w4 port decodes every weight to bf16 (FP4 -> f32 ->
x scale -> bf16) and runs bf16 MFMA 16x16x16. This kernel instead decodes FP4 straight
to an E4M3FNUZ byte (one v_perm per 4 weights, no scale) and runs the fp8 MFMA
16x16x32; the per-1x32 E8M0 scales of A and W are applied once per MFMA result:

    acc[m, n] += s_x[m, blk] * s_w[n, blk] * mfma(x_fp8[m, blk], w_fp8[blk, n])

That is exact only if one MFMA covers exactly one 32-wide MX block. In the a16w4 W
preshuffle a lane's dwordx4 holds one whole MX block, so a K32 MFMA built from it would
mix four blocks (one per 16-lane group). ``shuffle_weight_a8w4_gfx942`` transposes each
1 KB chunk so the same dwordx4 address instead returns one 8-weight part of each of the
four blocks of a 128-wide K group; for block ``j`` the four lane groups then hold its 32
consecutive K. ``shuffle_scale_a8w4_gfx942`` likewise packs a column's four block scales
into one dword. Both are one-time host relayouts of the a16w4 preshuffle.

A is MXFP8 (E4M3FNUZ ``[n_tokens, K]`` in ``A8W4_K8_ORDER`` + E8M0 ``[n_tokens, K/32]``),
quantised before the kernel (``mxfp8_quant_a8w4_gfx942``). Same sorting / cumsum / m_indices contract and the same bf16 intermediate
``[sorted_size, inter_dim]`` (by sorted position) as the a16w4 port, so stage2 is unchanged.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.act import gate_up_act, situ_params
from aiter.ops.flydsl.kernels.mxfp4_gemm_common import (
    _global_i32_buffer_tiles,
    lds_typed_ptr,
    lds_vec_load,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.flydsl.kernels.tensor_shim import _to_raw as _raw

from .utils import (
    _FP4_E2M1_FP8_LUT_HI,
    _FP4_E2M1_FP8_LUT_LO,
    _buffer_i32_scalar_read,
    _e8m0_byte_to_f32,
    _global_i32_at,
    _udiv,
    _umod,
    make_a_loader,
    make_b_loader,
)

MX_BLOCK = 32  # E8M0 scale group == one fp8 MFMA K step


# Byte order of one lane's 8 K values in both MFMA operands: the FP4 decode yields the
# even elements then the odd ones, so A is stored in the same order (see
# ``mxfp8_quant_a8w4_gfx942``) instead of interleaving W back with two more v_perm.
A8W4_K8_ORDER = (0, 2, 4, 6, 1, 3, 5, 7)


def _fp4x8_to_fp8x8(raw_i32):
    """8 E2M1 nibbles (bits[4e+3:4e] = element e) -> 8 E4M3FNUZ bytes as one i64.

    Byte p of the result is element ``A8W4_K8_ORDER[p]``. The magnitude comes from one
    v_perm per 4 nibbles (the #6144 table); the sign is kept only for non-zero
    magnitudes, since 0x80 is NaN in FNUZ (E2M1 -0 must become +0).
    """
    raw = fx.Int32(raw_i32)
    mag_e = raw & fx.Int32(0x07070707)  # elements 0,2,4,6 (low nibbles)
    mag_o = raw.shrui(fx.Int32(4)) & fx.Int32(0x07070707)  # elements 1,3,5,7

    def _lut(mag):
        return fx.Int32(
            rocdl.perm_b32(
                _raw(fx.Int32(_FP4_E2M1_FP8_LUT_HI)),
                _raw(fx.Int32(_FP4_E2M1_FP8_LUT_LO)),
                _raw(mag),
            )
        )

    def _nz(mag):  # 0x80 in every byte whose magnitude is non-zero
        return (mag + fx.Int32(0x7F7F7F7F)) & fx.Int32(0x80808080)

    sign_e = (raw << fx.Int32(4)) & fx.Int32(0x80808080)
    sign_o = raw & fx.Int32(0x80808080)
    ev = _lut(mag_e) | (sign_e & _nz(mag_e))
    od = _lut(mag_o) | (sign_o & _nz(mag_o))
    return fx.Vector.from_elements([_raw(ev), _raw(od)], fx.Int32).bitcast(fx.Int64)[0]


def _gemm1_body_a8w4(
    lds_raw_ptr,
    arg_x,
    arg_xscale,
    arg_bq,
    arg_bscale,
    arg_eids,
    arg_mind,
    arg_cumsum,
    arg_out,
    bx_i32,
    lane,
    wave,
    i32_ntok,
    situ,
    *,
    BM,
    TILE_N,
    TILE_K,
    K,
    INTER,
    NE,
    act,
    b_cache_mod,
    rocm_arch,
    x_scale="mx",
    k_wave=1,
):
    _row_xs = x_scale == "row"
    N_OUT = 2 * INTER
    m_repeat = BM // 16
    n_k0 = TILE_K // 128  # 128-wide K groups per tile (4 MX blocks each)
    n_blk = TILE_K // MX_BLOCK  # MX blocks (= fp8 MFMA K steps) per tile
    # Wave partition (as gemm1): (4/k_wave) N-waves x k_wave K-waves. Each K-wave runs
    # K/k_wave with its own A-LDS region; partials are LDS-reduced before the epilogue.
    num_n_waves = 4 // k_wave
    if const_expr(k_wave > 1):
        wave_n_id = wave % fx.Int32(num_n_waves)
        wave_k_id = rocdl.readfirstlane(T.i32, wave // fx.Int32(num_n_waves))
    else:
        wave_n_id = wave
        wave_k_id = fx.Int32(0)
    _n_per_wave = TILE_N // num_n_waves
    num_acc_n = _n_per_wave // 16
    klen = K // k_wave
    K_TILES_TOTAL = klen // TILE_K
    _PIPE = K_TILES_TOTAL > 1
    A_LDS_STAGES = 2 if _PIPE else 1
    A_SLOT_BYTES = BM * TILE_K  # fp8: one byte per element
    _A_GRP_BYTES = A_LDS_STAGES * A_SLOT_BYTES
    NUM_N_BLOCKS = INTER // TILE_N
    # 16 B per loading thread within a k-group; small tiles use fewer threads (the rest
    # repeat the load).
    a_load_threads = min(num_n_waves * 64, (BM * TILE_K) // 16)

    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    n_block_idx = bx_i32 % fx.Int32(NUM_N_BLOCKS)
    m_block_idx = bx_i32 // fx.Int32(NUM_N_BLOCKS)
    e = rocdl.readfirstlane(T.i32, _raw(_global_i32_at(arg_eids, m_block_idx)))
    bx_m = m_block_idx * fx.Int32(BM)
    by_n = n_block_idx * fx.Int32(TILE_N)
    inter_i32 = fx.Int32(INTER)
    # bf16 intermediate [sorted_size, inter]; padding-row stores are masked (see gemm1).
    _cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
    out_rsrc = buffer_ops.create_buffer_resource_from_addr(
        _raw(fx.Int64(arg_out)),
        num_records_bytes=_raw(fx.Int64(_cumsum0) * fx.Int64(INTER * 2)),
    )

    # ---- W addressing: reuse the a16w4 column descriptors (same preshuffle) ------
    b_loader = make_b_loader(
        arg_bq,
        arg_bscale,
        N_OUT=N_OUT,
        K=K,
        NE=NE,
        e=e,
        lane_div_16=lane_div_16,
        lane_mod_16=lane_mod_16,
        TILE_K=TILE_K,
        w_dtype="fp4",
        b_cache_mod=b_cache_mod,
        rocm_arch=rocm_arch,
    )
    # W: the a8w4 relayout (shuffle_weight_a8w4_gfx942) keeps the a16w4 dwordx4 address,
    # so b_loader.load_raw already returns raw[k0][j] = 8 weights of MX block j.
    # Scale: a8w4 relayout (shuffle_scale_a8w4_gfx942), dword (mni, s_ku, h, npack, nlane)
    # holds the 4 e8m0 bytes of the 128-wide K group h of that 256-wide s_ku.
    scale_k_padded = ((K + 255) // 256) * 256
    sc_stride_n0 = (((scale_k_padded // 32) // 4) // 2) * 64
    sw_tiles1 = _global_i32_buffer_tiles(
        arg_bscale, min(NE * N_OUT * (scale_k_padded // 32), 0xFFFFFFFF), 1
    )
    sw_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), fx.Int32)

    # ---- A: MXFP8 rows staged to LDS by the shared loader -------------------------
    # An fp8 row of TILE_K bytes has the byte layout of a bf16 row of TILE_K/2, so the
    # a16w4 loader is reused with half the K extent.
    c_k_div4 = K // 4  # dwords per fp8 row

    def _a_row_base_dwords(row_local):
        fused = fx.Int32(_global_i32_at(arg_mind, bx_m + row_local))
        return (fused & fx.Int32(0x00FFFFFF)) * fx.Int32(c_k_div4)

    if const_expr(k_wave > 1):
        k_grp_base_bytes = wave_k_id * fx.Int32(_A_GRP_BYTES)
    else:
        k_grp_base_bytes = fx.Int32(0)

    a_loader = make_a_loader(
        lds_raw_ptr,
        num_i32=k_wave * _A_GRP_BYTES // 4,
        BM=BM,
        TILE_K=TILE_K // 2,
        KH_TILE_BYTES=TILE_K,
        k_blocks16=TILE_K // 16,
        lane_div_16=lane_div_16,
        lane_mod_16=lane_mod_16,
        swizzle=False,
        a_ptr=arg_x,
        a_num_bytes=fx.Int64(i32_ntok) * fx.Int64(K),
        a_load_threads=a_load_threads,
        row_base_dwords=_a_row_base_dwords,
        dma_cache_mod=2,
        dma_via_vgpr=True,
        k_grp_base_bytes=k_grp_base_bytes,
        A_SLOT_BYTES=A_SLOT_BYTES,
    )
    lds_base_i32 = fx.Int32(fx.ptrtoint(lds_raw_ptr))

    # A scales; this lane needs the rows its accumulator holds. Padding rows (token >=
    # n_tokens) read past the resource -> 0 -> no contribution.
    #   "mx":  e8m0 [n_tokens, K/32], applied per MFMA together with the W scale.
    #   "row": f32 [n_tokens] (one scale per token), applied once in the epilogue.
    xs_row_dwords = K // 128
    xs_tiles1 = _global_i32_buffer_tiles(
        arg_xscale,
        fx.Int64(i32_ntok) * fx.Int64(4 if _row_xs else K // MX_BLOCK),
        1,
    )
    xs_row_base = []  # [mi][i]: "mx" dword index of the row's first scale; "row" f32
    for mi in range_constexpr(m_repeat):
        bases = []
        for ii in range_constexpr(4):
            row = bx_m + fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4) + fx.Int32(ii)
            tok = fx.Int32(_global_i32_at(arg_mind, row)) & fx.Int32(0x00FFFFFF)
            if const_expr(_row_xs):
                bases.append(
                    _buffer_i32_scalar_read(xs_tiles1, tok, sw_atom).bitcast(fx.Float32)
                )
            else:
                bases.append(tok * fx.Int32(xs_row_dwords))
        xs_row_base.append(bases)

    # ---- gate/up column descriptors (standard GGUU layout) ------------------------
    n_tile_base = wave_n_id * fx.Int32(_n_per_wave)
    col_g_list, cols_gate, cols_up = [], [], []
    for ni in range_constexpr(num_acc_n):
        _ni16 = fx.Int32(ni * 16)
        col_g_list.append(by_n + n_tile_base + _ni16 + lane_mod_16)
        _cg, _cu = b_loader.col_pair(by_n, n_tile_base, _ni16, shift=inter_i32)
        cols_gate.append(_cg)
        cols_up.append(_cu)

    def load_w_tile(base_k, col):
        """raw[b] = i32 with K [blk_b*32 + lane_grp*8, +8) of MX block b of the tile."""
        return [r for grp in b_loader.load_raw(base_k, col) for r in grp]

    def load_w_scale(base_k, col):
        """sc[b] = f32 e8m0 scale of this lane's column for MX block b of the tile."""
        scales = []
        for k0i in range_constexpr(n_k0):
            k0 = (base_k + fx.Int32(k0i * 128)) // fx.Int32(128)
            idx = (
                col.sc_blk * fx.Int32(sc_stride_n0)
                + (k0 // fx.Int32(2)) * fx.Int32(64)
                + (k0 % fx.Int32(2)) * fx.Int32(32)
                + col.sc_pack * fx.Int32(16)
                + lane_mod_16
            )
            dw = _buffer_i32_scalar_read(sw_tiles1, idx, sw_atom)
            for j in range_constexpr(4):
                scales.append(_e8m0_byte_to_f32(dw, fx.Int32(j)))
        return scales

    def load_x_scale(base_k):
        """xs[mi][i] = list of TILE_K/128 dwords (4 e8m0 bytes each) for that row."""
        if const_expr(_row_xs):
            return None
        out = []
        k_dw = base_k // fx.Int32(128)
        for mi in range_constexpr(m_repeat):
            rows = []
            for ii in range_constexpr(4):
                rows.append(
                    [
                        _buffer_i32_scalar_read(
                            xs_tiles1,
                            xs_row_base[mi][ii] + k_dw + fx.Int32(d),
                            sw_atom,
                        )
                        for d in range_constexpr(n_k0)
                    ]
                )
            out.append(rows)
        return out

    def load_tile(base_k):
        g_raw = [load_w_tile(base_k, c) for c in cols_gate]
        u_raw = [load_w_tile(base_k, c) for c in cols_up]
        g_sc = [load_w_scale(base_k, c) for c in cols_gate]
        u_sc = [load_w_scale(base_k, c) for c in cols_up]
        x_sc = load_x_scale(base_k)
        return g_raw, u_raw, g_sc, u_sc, x_sc

    def load_a_frags(slot):
        # a[mi][b]: 8 fp8 of row (mi*16 + lane%16), K [b*32 + lane_grp*8, +8).
        frags = []
        for mi in range_constexpr(m_repeat):
            row = fx.Int32(mi * 16) + lane_mod_16
            row_byte = (
                k_grp_base_bytes
                + fx.Int32(slot * A_SLOT_BYTES)
                + row * fx.Int32(TILE_K)
            )
            frags.append(
                [
                    lds_vec_load(
                        lds_base_i32,
                        row_byte + fx.Int32(b * MX_BLOCK) + lane_div_16 * fx.Int32(8),
                        T.i64,
                        T.i64,
                        align=8,
                    )
                    for b in range_constexpr(n_blk)
                ]
            )
        return frags

    zero4 = _raw(fx.Vector.filled(4, 0.0, fx.Float32))
    acc_gate = [[zero4 for _ in range(num_acc_n)] for _ in range(m_repeat)]
    acc_up = [[zero4 for _ in range(num_acc_n)] for _ in range(m_repeat)]

    def _mfma(a_i64, b_i64):
        return rocdl.mfma_f32_16x16x32_fp8_fp8(
            T.f32x4, [_raw(a_i64), _raw(b_i64), zero4, 0, 0, 0]
        )

    def _splat4(v):
        return Vec.from_elements([_raw(v)] * 4, fx.Float32)

    def compute_tile(tile, a_frags):
        g_raw, u_raw, g_sc, u_sc, x_sc = tile
        # One 128-wide K group (4 MX blocks) at a time: issue all its MFMAs first, then
        # scale-accumulate, so no MFMA result is consumed right after it is produced.
        for k0i in range_constexpr(n_k0):
            blks = [k0i * 4 + j for j in range_constexpr(4)]
            t_g, t_u = {}, {}
            for b in blks:
                for ni in range_constexpr(num_acc_n):
                    bg = _fp4x8_to_fp8x8(g_raw[ni][b])
                    bu = _fp4x8_to_fp8x8(u_raw[ni][b])
                    for mi in range_constexpr(m_repeat):
                        t_g[(b, ni, mi)] = _mfma(a_frags[mi][b], bg)
                        t_u[(b, ni, mi)] = _mfma(a_frags[mi][b], bu)
            for b in blks:
                if const_expr(_row_xs):
                    for ni in range_constexpr(num_acc_n):
                        sg4 = _splat4(g_sc[ni][b])
                        su4 = _splat4(u_sc[ni][b])
                        for mi in range_constexpr(m_repeat):
                            acc_gate[mi][ni] = _raw(
                                fx.math.fma(
                                    Vec(t_g[(b, ni, mi)]), sg4, Vec(acc_gate[mi][ni])
                                )
                            )
                            acc_up[mi][ni] = _raw(
                                fx.math.fma(
                                    Vec(t_u[(b, ni, mi)]), su4, Vec(acc_up[mi][ni])
                                )
                            )
                    continue
                # x scales of this lane's 4 accumulator rows for block b, as one vector
                xs4 = [
                    Vec.from_elements(
                        [
                            _raw(
                                _e8m0_byte_to_f32(x_sc[mi][ii][b // 4], fx.Int32(b % 4))
                            )
                            for ii in range_constexpr(4)
                        ],
                        fx.Float32,
                    )
                    for mi in range_constexpr(m_repeat)
                ]
                for ni in range_constexpr(num_acc_n):
                    sg4 = _splat4(g_sc[ni][b])
                    su4 = _splat4(u_sc[ni][b])
                    for mi in range_constexpr(m_repeat):
                        acc_gate[mi][ni] = _raw(
                            fx.math.fma(
                                Vec(t_g[(b, ni, mi)]),
                                xs4[mi] * sg4,
                                Vec(acc_gate[mi][ni]),
                            )
                        )
                        acc_up[mi][ni] = _raw(
                            fx.math.fma(
                                Vec(t_u[(b, ni, mi)]),
                                xs4[mi] * su4,
                                Vec(acc_up[mi][ni]),
                            )
                        )

    # ---- main K loop: A-LDS double buffer, W + scales for kt+1 in flight ---------
    def _a_store(base_k, slot):
        # loader works in "bf16" units of 2 bytes
        a_loader.store_tile(base_k // fx.Int32(2), slot=slot)

    k_base = wave_k_id * fx.Int32(klen) if const_expr(k_wave > 1) else fx.Int32(0)
    if const_expr(not _PIPE):
        _a_store(k_base, 0)
        t0 = load_tile(k_base)
        rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()
        compute_tile(t0, load_a_frags(0))
    else:
        _a_store(k_base, 0)
        t_cur = load_tile(k_base)
        for kt in range_constexpr(K_TILES_TOTAL):
            cur_slot = kt % A_LDS_STAGES
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()
            a_frags = load_a_frags(cur_slot)
            if const_expr(kt + 1 < K_TILES_TOTAL):
                nk = k_base + fx.Int32((kt + 1) * TILE_K)
                _a_store(nk, (kt + 1) % A_LDS_STAGES)
                t_nxt = load_tile(nk)
            compute_tile(t_cur, a_frags)
            if const_expr(kt + 1 < K_TILES_TOTAL):
                t_cur = t_nxt

    # ---- k_wave slice-K reduce: each wave parks its accumulators in LDS (the A region
    # is free now), then sums its K-peers' (peer = g*num_n_waves + wave_n_id) partials.
    # Gate and up in separate rounds to halve the scratch, as gemm1.
    if const_expr(k_wave > 1):
        nm = num_acc_n * m_repeat
        grp_stride = 64 * nm * 4  # f32 per wave
        lds_scr = lds_typed_ptr(lds_base_i32, T.f32)

        def _reduce_round(accs):
            gpu.barrier()
            my_base = wave * fx.Int32(grp_stride) + lane * fx.Int32(4)
            for ai in range_constexpr(nm):
                v = Vec(accs[ai // num_acc_n][ai % num_acc_n])
                sidx = my_base + fx.Int32(ai * 64 * 4)
                for vv in range_constexpr(4):
                    lds_scr[sidx + fx.Int32(vv)] = fx.Float32(v[vv])
            gpu.barrier()
            for ai in range_constexpr(nm):
                mi_, ni_ = ai // num_acc_n, ai % num_acc_n
                ai_off = fx.Int32(ai * 64 * 4) + lane * fx.Int32(4)
                sv = Vec(accs[mi_][ni_])
                for g in range_constexpr(1, k_wave):
                    peer = fx.Int32(g * num_n_waves) + wave_n_id
                    pv = Vec(
                        lds_vec_load(
                            lds_base_i32,
                            (peer * fx.Int32(grp_stride) + ai_off) * fx.Int32(4),
                            Vec.make_type(4, fx.Float32),
                            fx.Float32,
                            align=8,
                        )
                    )
                    sv = sv + pv
                accs[mi_][ni_] = _raw(sv)

        _reduce_round(acc_gate)
        _reduce_round(acc_up)
        _is_primary = wave_k_id == fx.Int32(0)

    # ---- epilogue: act(gate)*up -> bf16 intermediate by sorted position ---------
    for mi in range_constexpr(m_repeat):
        for ii in range_constexpr(4):
            row_in_tile = fx.Int32(mi * 16) + lane_div_16 * fx.Int32(4) + fx.Int32(ii)
            sorted_row = bx_m + row_in_tile
            fused = fx.Int32(_global_i32_at(arg_mind, sorted_row))
            token = fused & fx.Int32(0x00FFFFFF)
            valid = token < i32_ntok
            if const_expr(k_wave > 1):
                valid = valid & _is_primary
            for ni in range_constexpr(num_acc_n):
                g = fx.Float32(Vec(acc_gate[mi][ni])[ii])
                u = fx.Float32(Vec(acc_up[mi][ni])[ii])
                if const_expr(_row_xs):
                    g = g * xs_row_base[mi][ii]
                    u = u * xs_row_base[mi][ii]
                y = gate_up_act(act, [g], [u], situ)[0]
                out_idx = sorted_row * inter_i32 + col_g_list[ni]
                buffer_ops.buffer_store(
                    y.to(fx.BFloat16), _raw(out_rsrc), _raw(out_idx), mask=valid
                )


def mxfp8_quant_a8w4_gfx942(x):
    """bf16 ``[M, K]`` -> (E4M3FNUZ ``[M, K]`` in ``A8W4_K8_ORDER``, E8M0 ``[M, K/32]``).

    Per-1x32 power-of-two scale with ``amax / scale <= 240`` (FNUZ max). Reference
    implementation in torch; the order permutation is free inside a fused quant kernel.
    """
    M, K = x.shape
    xb = x.float().view(M, K // 32, 32)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    ex = torch.ceil(torch.log2(amax / 240.0)).clamp(-127, 127)
    q = (xb / torch.exp2(ex)).to(torch.float8_e4m3fnuz).view(M, K // 8, 8)
    q = q[..., list(A8W4_K8_ORDER)].reshape(M, K).contiguous()
    return q, (ex + 127).to(torch.uint8).view(M, K // 32).contiguous()


def shuffle_weight_a8w4_gfx942(w_a16w4_u8):
    """Relayout a16w4-preshuffled MXFP4 W (``shuffle_weight_a16w4(w, 16, False)``).

    In each 1 KB chunk (16 N rows x 128 K) the a16w4 order is [klane j][nlane][dword g]
    (klane j = MX block j, dword g = its K [g*8, +8)). This writes [g][nlane][j], so a
    lane of group g loads part g of all four blocks with the same dwordx4 address.
    """
    w = w_a16w4_u8.view(torch.uint8)
    assert w.numel() % 1024 == 0, "W bytes must be a multiple of 1 KB"
    out = w.reshape(-1, 4, 16, 4, 4).permute(0, 3, 2, 1, 4).contiguous()
    return out.view(w_a16w4_u8.shape)


def shuffle_scale_a8w4_gfx942(s_a16w4_u8):
    """Relayout a16w4-preshuffled E8M0 W scales (``shuffle_scale_a16w4(s, E, False)``).

    Per 256 B chunk (32 N x 256 K) a16w4 holds dwords [klane 4][nlane 16] with bytes
    [kpack_sub 2][npack 2]; this writes dwords [kpack_sub][npack][nlane] with bytes
    [klane], i.e. one dword = the 4 block scales of one column and one 128-wide K group.
    """
    s = s_a16w4_u8.view(torch.uint8)
    assert s.numel() % 256 == 0, "scale bytes must be a multiple of 256"
    out = s.reshape(-1, 4, 16, 2, 2).permute(0, 3, 4, 2, 1).contiguous()
    return out.view(s_a16w4_u8.shape)


def gemm1_a8w4_grid(BM, *, INTER, TILE_N, max_m_blocks):
    return int(max_m_blocks) * (INTER // TILE_N)


@functools.cache
def compile_gemm1_a8w4_gfx942(
    BM=16,
    *,
    D_HIDDEN,
    D_INTER,
    NE,
    TILE_N=64,
    TILE_K=128,
    act="silu",
    b_cache_mod=2,
    xcd_swizzle=0,
    waves_per_eu=None,
    x_scale="mx",
    k_wave=1,
    rev="r6",  # bump on kernel changes: the FlyDSL cache key hashes this factory only
    rocm_arch,
):
    """Build the gfx942 a8w4 (MXFP8 A x MXFP4 W) fused stage1 (gate+up + act)."""
    assert str(rocm_arch).startswith("gfx942"), f"gfx942 only, got {rocm_arch}"
    _K, _INTER = D_HIDDEN, D_INTER
    assert TILE_K % 128 == 0, f"TILE_K must be a multiple of 128, got {TILE_K}"
    assert _K % TILE_K == 0, f"D_HIDDEN must be a multiple of TILE_K, got {_K}"
    assert (2 * _INTER) % 256 == 0, f"2*D_INTER must be a multiple of 256, got {_INTER}"
    assert _INTER % TILE_N == 0, f"D_INTER must be a multiple of TILE_N={TILE_N}"
    assert k_wave in (1, 2, 4), f"k_wave must be 1, 2 or 4, got {k_wave}"
    assert (
        _K % (k_wave * TILE_K) == 0
    ), f"D_HIDDEN={_K} must be a multiple of k_wave*TILE_K={k_wave * TILE_K}"
    assert (
        TILE_N % (16 * (4 // k_wave)) == 0
    ), f"TILE_N must be a multiple of {16 * (4 // k_wave)} at k_wave={k_wave}, got {TILE_N}"
    assert BM % 16 == 0, f"BM must be a multiple of 16, got {BM}"
    assert (BM * TILE_K) % 1024 == 0, "BM*TILE_K must be a multiple of 1024"
    assert act in ("silu", "swiglu", "situv2"), f"bad act {act!r}"
    assert x_scale in ("mx", "row"), f"x_scale must be 'mx' or 'row', got {x_scale!r}"
    NUM_N_BLOCKS = _INTER // TILE_N
    _a_lds_stages = 2 if (_K // k_wave // TILE_K) > 1 else 1
    lds_bytes = k_wave * _a_lds_stages * BM * TILE_K
    if (
        k_wave > 1
    ):  # reduce scratch reuses the A region: 4 waves x nm vec4-f32 x 64 lanes
        _nm = (TILE_N // (4 // k_wave) // 16) * (BM // 16)
        lds_bytes = max(lds_bytes, 4 * _nm * 64 * 4 * 4)

    _act_tag = "" if act == "silu" else f"_{act}"
    _bcm_tag = "" if b_cache_mod == 2 else f"_bcm{b_cache_mod}"
    _xcd_tag = f"_xcd{xcd_swizzle}" if xcd_swizzle > 0 else ""
    _wpe_tag = f"_w{waves_per_eu}" if waves_per_eu else ""
    name_suffix = (
        f"a8w4_h{_K}_i{_INTER}_ne{NE}_bm{BM}_tn{TILE_N}_tk{TILE_K}"
        f"{_act_tag}{_bcm_tag}{_xcd_tag}{_wpe_tag}"
        f"{'' if x_scale == 'mx' else '_xrow'}{f'_kw{k_wave}' if k_wave > 1 else ''}_{rev}"
    )

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    @flyc.kernel(name=f"gemm1_a8w4_gfx942_{name_suffix}", known_block_size=[256, 1, 1])
    def gemm1_kernel(
        arg_x: fx.Int64,
        arg_xscale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        f32_situ_beta: fx.Float32,
        f32_situ_beta_rcp: fx.Float32,
        f32_situ_linbeta: fx.Float32,
        f32_situ_linbeta_rcp: fx.Float32,
        f32_swiglu_limit: fx.Float32,
        arg_out: fx.Int64,
    ):
        lds_raw_ptr = fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr
        tx_i32 = fx.Int32(gpu.thread_id("x"))
        bx_i32 = fx.Int32(gpu.block_id("x"))
        lane = tx_i32 % fx.Int32(64)
        wave = rocdl.readfirstlane(T.i32, tx_i32 // fx.Int32(64))
        cumsum0 = _global_i32_at(arg_cumsum, fx.Int32(0))
        total_m_blocks = cumsum0 // fx.Int32(BM)
        bound = total_m_blocks * fx.Int32(NUM_N_BLOCKS)

        _NXCD = 8
        _xq = _udiv(bound, _NXCD)
        _xr = _umod(bound, _NXCD)
        _SW = xcd_swizzle

        def _xcd(pid):
            xc = _umod(pid, _NXCD)
            wgid = xc * _xq + fx.min(xc, _xr) + _udiv(pid, _NXCD)
            _ng = fx.Int32(_SW * NUM_N_BLOCKS)
            group_id = wgid // _ng
            first_pid_m = group_id * fx.Int32(_SW)
            remaining_m = total_m_blocks - first_pid_m
            group_size_m = fx.min(remaining_m, fx.Int32(_SW))
            wig = wgid % _ng
            m_block = first_pid_m + (wig % group_size_m)
            n_block = wig // group_size_m
            return m_block * fx.Int32(NUM_N_BLOCKS) + n_block

        if bx_i32 < bound:
            _tile = _xcd(bx_i32) if const_expr(_SW > 0) else bx_i32
            if const_expr(act in ("swiglu", "situv2")):
                _situ = situ_params(
                    fx.Float32(f32_situ_beta),
                    fx.Float32(f32_situ_beta_rcp),
                    fx.Float32(f32_situ_linbeta),
                    fx.Float32(f32_situ_linbeta_rcp),
                    fx.Float32(f32_swiglu_limit),
                )
            else:
                _situ = None
            _gemm1_body_a8w4(
                lds_raw_ptr,
                arg_x,
                arg_xscale,
                arg_bq,
                arg_bscale,
                arg_eids,
                arg_mind,
                arg_cumsum,
                arg_out,
                _tile,
                lane,
                wave,
                i32_ntok,
                _situ,
                BM=BM,
                TILE_N=TILE_N,
                TILE_K=TILE_K,
                K=_K,
                INTER=_INTER,
                NE=NE,
                act=act,
                b_cache_mod=b_cache_mod,
                rocm_arch=rocm_arch,
                x_scale=x_scale,
                k_wave=k_wave,
            )

    @flyc.jit
    def launch_gemm1(
        arg_x: fx.Int64,
        arg_xscale: fx.Int64,
        arg_bq: fx.Int64,
        arg_bscale: fx.Int64,
        arg_eids: fx.Int64,
        arg_cumsum: fx.Int64,
        arg_mind: fx.Int64,
        i32_ntok: fx.Int32,
        i32_grid: fx.Int32,
        f32_situ_beta: fx.Float32,
        f32_situ_beta_rcp: fx.Float32,
        f32_situ_linbeta: fx.Float32,
        f32_situ_linbeta_rcp: fx.Float32,
        f32_swiglu_limit: fx.Float32,
        arg_out: fx.Int64,
        stream: fx.Stream,
    ):
        gemm1_kernel(
            arg_x,
            arg_xscale,
            arg_bq,
            arg_bscale,
            arg_eids,
            arg_cumsum,
            arg_mind,
            i32_ntok,
            f32_situ_beta,
            f32_situ_beta_rcp,
            f32_situ_linbeta,
            f32_situ_linbeta_rcp,
            f32_swiglu_limit,
            arg_out,
            value_attrs={"rocdl.waves_per_eu": waves_per_eu} if waves_per_eu else None,
        ).launch(grid=(fx.Int64(i32_grid), 1, 1), block=(256, 1, 1), stream=stream)

    return launch_gemm1


def flydsl_a8w4_gemm1_gfx942(
    *,
    a_fp8,
    a_scale_u8,
    w1_u8,
    w1_scale_u8,
    sorted_expert_ids,
    cumsum_tensor,
    m_indices,
    inter_sorted_bf16,
    n_tokens,
    NE,
    D_HIDDEN,
    D_INTER,
    tile_m=16,
    tile_n=64,
    tile_k=128,
    b_nt=None,
    xcd_swizzle=0,
    waves_per_eu=None,
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
    swiglu_limit=float("inf"),
    x_scale="mx",
    k_wave=1,
    stream=None,
):
    """gfx942 a8w4 fused stage1. ``a_fp8``: E4M3FNUZ ``[n_tokens, D_HIDDEN]``;
    ``a_scale_u8``: E8M0 ``[n_tokens, D_HIDDEN/32]`` row-major. W1/scale: the a16w4
    preshuffle relaid by ``shuffle_weight_a8w4_gfx942`` / ``shuffle_scale_a8w4_gfx942``.
    """
    launch = compile_gemm1_a8w4_gfx942(
        BM=tile_m,
        D_HIDDEN=D_HIDDEN,
        D_INTER=D_INTER,
        NE=NE,
        TILE_N=tile_n,
        TILE_K=tile_k,
        act=act,
        b_cache_mod=2 if b_nt is None else b_nt,
        xcd_swizzle=xcd_swizzle,
        waves_per_eu=waves_per_eu,
        x_scale=x_scale,
        k_wave=k_wave,
        rocm_arch=str(get_rocm_arch()),
    )
    grid = gemm1_a8w4_grid(
        tile_m, INTER=D_INTER, TILE_N=tile_n, max_m_blocks=sorted_expert_ids.numel()
    )
    _beta, _lbeta = float(situ_beta), float(situ_linear_beta)
    _run_compiled(
        launch,
        a_fp8.data_ptr(),
        a_scale_u8.data_ptr(),
        w1_u8.data_ptr(),
        w1_scale_u8.data_ptr(),
        sorted_expert_ids.data_ptr(),
        cumsum_tensor.data_ptr(),
        m_indices.data_ptr(),
        int(n_tokens),
        int(grid),
        _beta,
        1.0 / _beta,
        _lbeta,
        1.0 / _lbeta,
        float(swiglu_limit),
        inter_sorted_bf16.data_ptr(),
        torch.cuda.current_stream() if stream is None else stream,
    )
    return inter_sorted_bf16
