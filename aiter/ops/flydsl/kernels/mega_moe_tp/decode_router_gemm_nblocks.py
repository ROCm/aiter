# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Router GEMM split-K partials for 33-128 rows, optionally with MXFP4 + MXFP8 input-quant
workgroups appended to / prepended to the same launch.

Ported from amd-kernels' ``moe/mxfp4_h6144_i768_e128_top4/router_gemm_nblocks.py`` (Apache-2.0, "Copyright contributors to the
amd-kernels project"); the kernel bodies are unchanged, only imports, the warm-up ``@launcher`` wrapper and the
compile-cache helper were replaced by aiter's own (``aiter.ops.flydsl.kernels.buffer_ops`` / ``tensor_shim``).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

from .decode_router_common import (
    PRELOAD_HINTS,
    _as_f32,
    _cvt_pk_fp4,
    _e8m0_even_e2m1,
    _fmax3_abs,
    _split3,
)
from .decode_tile_primitives import _e8m0_fp8, _fp8

H, E = 6144, 128
BLOCKS = H // 32  # 32-value groups per row


def _dual_quant_group(i32_m, i32_sx, arg_x, arg_q, arg_s, arg_q8, arg_s8, blk):
    """One lane: 32-value group ``blk`` (row ``blk // BLOCKS``, group ``blk % BLOCKS``) of x (row stride
    ``i32_sx`` elements) -> MXFP4 (even e8m0) and MXFP8 codes and scales, ``router_topk_mxfp4_mxfp8_quant.py``'s
    arithmetic op for op. The outputs are packed [m, ...]."""
    xr = buffer_ops.create_buffer_resource_from_addr(
        arg_x, num_records_bytes=fx.Int64(i32_m) * fx.Int64(i32_sx * 2)
    )
    row = blk // BLOCKS
    group = blk - row * BLOCKS
    x_dw = ((row * i32_sx) >> 1) + group * 16  # dword offset of the group
    xv = [
        Vec(buffer_ops.buffer_load(xr, x_dw + j * 4, vec_width=4, dtype=fx.Int32))
        for j in range_constexpr(4)
    ]
    qr = buffer_ops.create_buffer_resource_from_addr(
        arg_q, num_records_bytes=fx.Int64(i32_m) * (H // 2)
    )
    sr = buffer_ops.create_buffer_resource_from_addr(
        arg_s, num_records_bytes=fx.Int64(i32_m) * (H // 32)
    )
    q8r = buffer_ops.create_buffer_resource_from_addr(
        arg_q8, num_records_bytes=fx.Int64(i32_m) * H
    )
    s8r = buffer_ops.create_buffer_resource_from_addr(
        arg_s8, num_records_bytes=fx.Int64(i32_m) * (H // 32)
    )
    vals = []
    for j in range_constexpr(4):
        for kk in range_constexpr(4):
            w = fx.Int32(xv[j][kk])
            vals.append(_as_f32(w << 16))
            vals.append(_as_f32(w & fx.Int32(-65536)))
    amax = _fmax3_abs(vals[0], vals[1], vals[2])
    for i in range_constexpr(3, 31, 2):
        amax = _fmax3_abs(vals[i], vals[i + 1], amax)
    amax = _fmax3_abs(vals[31], vals[31], amax)
    e8 = _e8m0_fp8(amax)
    sf8 = (e8 << 23).bitcast(fx.Float32)
    for half in range_constexpr(2):
        words = []
        for dw in range_constexpr(4):
            j = half * 16 + dw * 4
            pk = _fp8(fx.Int32(0), vals[j], vals[j + 1], sf8, False)
            pk = _fp8(pk, vals[j + 2], vals[j + 3], sf8, True)
            words.append(pk)
        buffer_ops.buffer_store(
            Vec.from_elements(words, fx.Int32), q8r, blk * 8 + half * 4
        )
    buffer_ops.buffer_store(e8.to(fx.Int8), s8r, blk)
    e8m0 = _e8m0_even_e2m1(amax)
    scale_f = _as_f32(e8m0 << 23)
    qo = []
    for d in range_constexpr(4):
        p = fx.Int32(0)
        for kk in range_constexpr(4):
            p = _cvt_pk_fp4(
                p, vals[8 * d + 2 * kk], vals[8 * d + 2 * kk + 1], scale_f, kk
            )
        qo.append(p)
    buffer_ops.buffer_store(Vec.from_elements(qo, fx.Int32), qr, blk * 4)
    buffer_ops.buffer_store(fx.Int32(e8m0).to(fx.Int8), sr, blk)


def compile_router_gemm_nblocks(
    split_k: int,
    block_k: int,
    waves: int,
    nbw: int,
    quant: int = 0,
    x_direct: bool = False,
    token_major: bool = False,
):
    """``router_gemm.compile_router_gemm(16, ...)``'s chains: ``waves`` m-tile waves, ``nbw`` of the 8 n-blocks per
    workgroup, ``quant`` 0 none, 1 appended, 2 prepended."""
    assert split_k % 8 == 0 and nbw in (1, 2, 4, 8) and quant in (0, 1, 2)
    NW = NT = waves
    NB = MT = 16
    SUBK = 32
    KC = H // split_k
    SPB = block_k // SUBK
    NSTEP = KC // block_k
    assert KC % block_k == 0 and block_k % SUBK == 0
    NACC = MT * NB // 64
    NBLK = E // NB
    NBG = NBLK // nbw  # n-block groups
    W_PIECES = nbw * NB * block_k // 4 // (64 * NT)
    X_PIECES = MT * block_k // 8 // 64
    assert nbw * NB * block_k // 4 == W_PIECES * 64 * NT and X_PIECES >= 1
    W_SLOT = 3 * SPB * 64
    X_SLOT = SPB * 64
    PPR = block_k // 8
    assert PPR <= 16
    Q8 = SUBK // 8

    def swz(chunk, kp):
        return chunk ^ (kp * (16 // PPR))

    name = (
        f"router_gemm_bf16x3_h6144_e128_mfma16_s{split_k}_bk{block_k}_w{NW}_nb{nbw}"
        + (f"_dual_quant{quant}" if quant else "")
        + ("_xd" if x_direct else "")
        + ("_token_partials" if token_major else "")
    )

    @fx.struct
    class Shared:
        w: fx.Array[fx.Int32, nbw * NSTEP * W_SLOT * 4, 16]
        x: fx.Array[fx.Int32, 2 * NW * X_SLOT * 4, 16]

    def mfma_op(a, b, acc):
        av = Vec(a).bitcast(fx.BFloat16)
        bv = Vec(b).bitcast(fx.BFloat16)
        return Vec(
            fx.rocdl.mfma_f32_16x16x32_bf16(
                Vec.make_type(NACC, fx.Float32), [av, bv, acc, 0, 0, 0]
            )
        )

    @flyc.kernel(name=name, known_block_size=[64 * NT, 1, 1])
    def kernel(
        arg_x: fx.Int64,  # [m, H] bf16, row stride i32_sx
        arg_w: fx.Int64,  # [E, H] fp32
        arg_p: fx.Int64,  # partials [split_k, m, E] fp32
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        arg_q: fx.Int64,  # quant: e2m1 codes [M, H / 2], e8m0 [M, H / 32], e4m3 [M, H], e8m0 [M, H / 32]
        arg_s: fx.Int64,
        arg_q8: fx.Int64,
        arg_s8: fx.Int64,
    ):
        tx, bid0 = fx.thread_idx.x, fx.block_idx.x
        n_router = (i32_m + (MT * NW - 1)) // (MT * NW) * (NBG * split_k)
        n_q = (i32_m * BLOCKS + (64 * NT * 8 - 1)) // (64 * NT * 8) * 8
        if const_expr(quant == 1) and bid0 >= n_router:
            blk = (bid0 - n_router) * (64 * NT) + tx
            if blk < i32_m * BLOCKS:
                _dual_quant_group(
                    i32_m, i32_sx, arg_x, arg_q, arg_s, arg_q8, arg_s8, blk
                )
        if const_expr(quant == 2) and bid0 < n_q:
            blk = bid0 * (64 * NT) + tx
            if blk < i32_m * BLOCKS:
                _dual_quant_group(
                    i32_m, i32_sx, arg_x, arg_q, arg_s, arg_q8, arg_s8, blk
                )
        bid = bid0 - n_q if const_expr(quant == 2) else bid0
        if (bid >= 0) & (bid < n_router):
            lane = tx % 64
            wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
            # bid = (g * NBG + nbg) * split_k + s: a split's workgroups share an XCD
            s = bid % split_k
            nbg = (bid // split_k) % NBG
            g = bid // (split_k * NBG)
            t0 = (g * NW + wave) * MT
            smem = fx.SharedAllocator().allocate(Shared).peek()

            def view16(arr, n):
                return fx.logical_divide(
                    fx.make_view(
                        fx.recast_iter(fx.Int32, arr.ptr), fx.make_layout(n, 1)
                    ),
                    fx.make_layout(4, 1),
                )

            w16 = view16(smem.w, nbw * NSTEP * W_SLOT * 4)
            w8 = fx.logical_divide(
                fx.make_view(
                    fx.recast_iter(fx.Int32, smem.w.ptr),
                    fx.make_layout(nbw * NSTEP * W_SLOT * 4, 1),
                ),
                fx.make_layout(2, 1),
            )
            x16 = view16(smem.x, 2 * NW * X_SLOT * 4)
            atom16 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
            atom8 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.Int32)
            wr = buffer_ops.create_buffer_resource_from_addr(
                arg_w, num_records_bytes=E * H * 4
            )
            xr = buffer_ops.create_buffer_resource_from_addr(
                arg_x, num_records_bytes=fx.Int64(i32_m) * fx.Int64(i32_sx * 2)
            )
            pr = buffer_ops.create_buffer_resource_from_addr(
                arg_p, num_records_bytes=fx.Int64(i32_m) * (split_k * E * 4)
            )
            k_base = s * KC

            def x_piece(it):
                u = it * 64 + lane
                return u // (block_k // 8), u % (block_k // 8)

            xl = []
            for st in range_constexpr(NSTEP if x_direct else 0):
                # x_direct: each wave's own m-tile in the MFMA lane order (row lane % 16, k piece lane // 16)
                for jj in range_constexpr(SPB):
                    row = t0 + lane % MT
                    xl.append(
                        Vec(
                            buffer_ops.buffer_load(
                                xr,
                                (
                                    row * i32_sx
                                    + k_base
                                    + st * block_k
                                    + jj * SUBK
                                    + (lane // MT) * 8
                                )
                                >> 1,
                                vec_width=4,
                                dtype=fx.Int32,
                                mask=row < i32_m,
                            )
                        )
                    )
            for st in range_constexpr(0 if x_direct else NSTEP):
                for it in range_constexpr(X_PIECES):
                    r, kp = x_piece(it)
                    row = t0 + r
                    xl.append(
                        Vec(
                            buffer_ops.buffer_load(
                                xr,
                                (row * i32_sx + k_base + st * block_k + kp * 8) >> 1,
                                vec_width=4,
                                dtype=fx.Int32,
                                mask=row < i32_m,
                            )
                        )
                    )

            def w_piece(it):
                """(n-block j of the group, expert of the n-block, 4-k piece)."""
                u = it * 64 * NT + tx
                n, kp = u // (block_k // 4), u % (block_k // 4)
                return n // NB, n % NB, kp

            wl = []
            for st in range_constexpr(NSTEP):
                for it in range_constexpr(W_PIECES):
                    j, n, kp = w_piece(it)
                    wl.append(
                        Vec(
                            buffer_ops.buffer_load(
                                wr,
                                ((nbg * nbw + j) * NB + n) * H
                                + k_base
                                + st * block_k
                                + kp * 4,
                                vec_width=4,
                                dtype=fx.Int32,
                            )
                        )
                    )

            def stage_w(st):
                for it in range_constexpr(W_PIECES):
                    j, n, kp = w_piece(it)
                    slot = (j * NSTEP + st) * W_SLOT
                    kq = kp // 2
                    chunk = slot + swz((kq // Q8) * 64 + n + NB * (kq % Q8), kq)
                    parts = _split3(
                        [
                            fx.Int32(wl[st * W_PIECES + it][e])
                            for e in range_constexpr(4)
                        ]
                    )
                    for p in range_constexpr(3):
                        r8 = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
                        r8.store(Vec.from_elements(parts[p], fx.Int32))
                        dst = fx.slice(w8, (None, (chunk + p * SPB * 64) * 2 + kp % 2))
                        fx.copy(atom8, r8, dst)

            def stage_x(st):
                if const_expr(x_direct):
                    return
                xslot = ((st % 2) * NW + wave) * X_SLOT
                for it in range_constexpr(X_PIECES):
                    r, kp = x_piece(it)
                    r16 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    r16.store(xl[st * X_PIECES + it])
                    c = swz((kp // Q8) * 64 + r + MT * (kp % Q8), kp)
                    fx.copy(atom16, r16, fx.slice(x16, (None, xslot + c)))

            def operands(st):
                """x of step ``st`` and each n-block's weight terms, in the MFMA lane order."""
                xslot = ((st % 2) * NW + wave) * X_SLOT
                xs, ws = [], []
                for jj in range_constexpr(SPB if not x_direct else 0):
                    c = swz(jj * 64 + lane, jj * Q8 + lane // MT)
                    r16 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    fx.copy(atom16, fx.slice(x16, (None, xslot + c)), r16)
                    xs.append(Vec(r16.load()))
                if const_expr(x_direct):
                    xs = xl[st * SPB : (st + 1) * SPB]
                for j in range_constexpr(nbw):
                    slot = (j * NSTEP + st) * W_SLOT
                    for p in range_constexpr(3):
                        for jj in range_constexpr(SPB):
                            c = swz(jj * 64 + lane, jj * Q8 + lane // MT)
                            r16 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                            fx.copy(
                                atom16,
                                fx.slice(w16, (None, slot + p * SPB * 64 + c)),
                                r16,
                            )
                            ws.append(Vec(r16.load()))
                return xs, ws

            for st in range_constexpr(NSTEP):
                stage_w(st)
            fx.gpu.barrier()
            stage_x(0)
            acc_hi = [Vec.filled(NACC, 0.0, fx.Float32) for _ in range_constexpr(nbw)]
            acc_lo = [Vec.filled(NACC, 0.0, fx.Float32) for _ in range_constexpr(nbw)]
            nxt = operands(0)
            for st in range_constexpr(NSTEP):
                xs, ws = nxt
                if const_expr(st + 1 < NSTEP):
                    stage_x(st + 1)
                    nxt = operands(st + 1)
                fx.rocdl.sched_barrier(0)
                for j in range_constexpr(nbw):
                    wj = ws[j * 3 * SPB : (j + 1) * 3 * SPB]
                    for jj in range_constexpr(SPB):
                        acc_hi[j] = mfma_op(wj[jj], xs[jj], acc_hi[j])
                    for p in range_constexpr(1, 3):
                        for jj in range_constexpr(SPB):
                            acc_lo[j] = mfma_op(wj[p * SPB + jj], xs[jj], acc_lo[j])
            m = t0 + lane % MT
            q = lane // MT
            for j in range_constexpr(nbw):
                out = [
                    fx.Float32(acc_hi[j][i]) + fx.Float32(acc_lo[j][i])
                    for i in range_constexpr(NACC)
                ]
                n = (nbg * nbw + j) * NB + q * 4
                buffer_ops.buffer_store(
                    Vec.from_elements(out, fx.Float32),
                    pr,
                    ((m * split_k + s) if const_expr(token_major) else (s * i32_m + m))
                    * E
                    + n,
                    mask=m < i32_m,
                )

    @flyc.jit
    def launch_router_gemm_nblocks_mxfp4_moe(
        arg_x: fx.Int64,
        arg_w: fx.Int64,
        arg_p: fx.Int64,
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        arg_q: fx.Int64,
        arg_s: fx.Int64,
        arg_q8: fx.Int64,
        arg_s8: fx.Int64,
        stream: fx.Stream,
    ):
        groups = (i32_m + (MT * NW - 1)) // (MT * NW)
        nq = 0
        if const_expr(quant == 1):
            nq = (i32_m * BLOCKS + (64 * NT - 1)) // (64 * NT)
        if const_expr(quant == 2):
            nq = (i32_m * BLOCKS + (64 * NT * 8 - 1)) // (64 * NT * 8) * 8
        kernel(arg_x, arg_w, arg_p, i32_m, i32_sx, arg_q, arg_s, arg_q8, arg_s8).launch(
            grid=(fx.Int64(groups * (NBG * split_k) + nq), 1, 1),
            block=(64 * NT, 1, 1),
            stream=stream,
        )

    launch_router_gemm_nblocks_mxfp4_moe.kernel_name = name
    launch_router_gemm_nblocks_mxfp4_moe.compile_hints = {
        **PRELOAD_HINTS,
        "waves_per_eu": 2,
    }
    return launch_router_gemm_nblocks_mxfp4_moe


@functools.cache
def _get_kernel(
    split_k: int,
    block_k: int,
    waves: int,
    nbw: int,
    quant: int = 0,
    x_direct: bool = False,
    token_major: bool = False,
):
    return compile_router_gemm_nblocks(
        split_k, block_k, waves, nbw, quant, x_direct, token_major
    )
