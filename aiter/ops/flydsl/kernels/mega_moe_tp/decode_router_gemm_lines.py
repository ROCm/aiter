# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Router GEMM split-K partials for 17-32 rows from the line-layout gate copy.

Ported from amd-kernels' ``moe/mxfp4_h6144_i768_e128_top4/router_gemm_nblocks_lines.py`` (Apache-2.0, "Copyright contributors to the
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
    _split3,
)

H, E = 6144, 128


def compile_router_gemm_nblocks_lines(token_major: bool = False):
    """The split-48, BLOCK_K-64 router with two 16-row m-tile waves."""
    split_k, block_k, waves, nbw = 48, 64, 2, 1
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
        f"router_gemm_nblocks_lines_bf16x3_h6144_e128_s{split_k}_bk{block_k}_w{NW}"
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
        arg_w: fx.Int64,  # layout.router_line_layout gate, fp32
        arg_p: fx.Int64,  # partials [split_k, m, E] fp32
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
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
                fx.make_view(fx.recast_iter(fx.Int32, arr.ptr), fx.make_layout(n, 1)),
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
        for st in range_constexpr(NSTEP):
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
            """(n-block j of the group, expert of the n-block, 4-k piece) of line-layout piece ``it``."""
            u = it * 64 * NT + tx
            j = u // (NB * block_k // 4)
            piece = u % (NB * block_k // 4)
            step = piece // 128
            h = (piece // 64) % 2
            lane_w = piece % 64
            return j, lane_w % 16, step * 8 + (lane_w // 16) * 2 + h

        wl = []
        for st in range_constexpr(NSTEP):
            for it in range_constexpr(W_PIECES):
                j, n, kp = w_piece(it)
                wl.append(
                    Vec(
                        buffer_ops.buffer_load(
                            wr,
                            ((k_base + st * block_k) // 32 + kp // 8) * (E * 32)
                            + (nbg * nbw + j) * (NB * 32)
                            + (kp % 2) * 256
                            + (kp % 8 // 2 * 16 + n) * 4,
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
                    [fx.Int32(wl[st * W_PIECES + it][e]) for e in range_constexpr(4)]
                )
                for p in range_constexpr(3):
                    r8 = fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32)
                    r8.store(Vec.from_elements(parts[p], fx.Int32))
                    dst = fx.slice(w8, (None, (chunk + p * SPB * 64) * 2 + kp % 2))
                    fx.copy(atom8, r8, dst)

        def stage_x(st):
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
            for jj in range_constexpr(SPB):
                c = swz(jj * 64 + lane, jj * Q8 + lane // MT)
                r16 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                fx.copy(atom16, fx.slice(x16, (None, xslot + c)), r16)
                xs.append(Vec(r16.load()))
            for j in range_constexpr(nbw):
                slot = (j * NSTEP + st) * W_SLOT
                for p in range_constexpr(3):
                    for jj in range_constexpr(SPB):
                        c = swz(jj * 64 + lane, jj * Q8 + lane // MT)
                        r16 = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                        fx.copy(
                            atom16, fx.slice(w16, (None, slot + p * SPB * 64 + c)), r16
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
                ((m * split_k + s) if const_expr(token_major) else (s * i32_m + m)) * E
                + n,
                mask=m < i32_m,
            )

    @flyc.jit
    def launch_router_gemm_nblocks_lines_mxfp4_moe(
        arg_x: fx.Int64,
        arg_w: fx.Int64,
        arg_p: fx.Int64,
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        stream: fx.Stream,
    ):
        groups = (i32_m + (MT * NW - 1)) // (MT * NW)
        kernel(arg_x, arg_w, arg_p, i32_m, i32_sx).launch(
            grid=(fx.Int64(groups * (NBG * split_k)), 1, 1),
            block=(64 * NT, 1, 1),
            stream=stream,
        )

    launch_router_gemm_nblocks_lines_mxfp4_moe.kernel_name = name
    launch_router_gemm_nblocks_lines_mxfp4_moe.compile_hints = {
        **PRELOAD_HINTS,
        "waves_per_eu": 2,
    }
    return launch_router_gemm_nblocks_lines_mxfp4_moe


@functools.cache
def _get_kernel(token_major: bool = False):
    return compile_router_gemm_nblocks_lines(token_major)
