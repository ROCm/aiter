# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Router GEMM split-K partials for 1-16 rows without LDS (bf16x3, MFMA 16x16x32).

Ported from amd-kernels' ``moe/mxfp4_h6144_i768_e128_top4/router_gemm_direct.py`` (Apache-2.0, "Copyright contributors to the
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


def compile_router_gemm_direct(
    split_k: int,
    block_k: int,
    tiles: int,
    mode: int = 0,
    lines: bool = False,
    token_major: bool = False,
):
    """``tiles`` (<= 4) 16-row m-tiles of one (n-block, split).

    Args:
        mode: 0 one wave takes the ``tiles`` m-tiles, 1 one wave per m-tile in a workgroup, 2 one workgroup per m-tile.
        lines: the weight is the ``layout.router_line_layout`` copy [192 k-steps, 8 n-blocks, 2, 4, 16, 4].
    """
    # split 48 only: S16 / S24 give wrong partials (logits off by 0.7-0.95 at 8 rows)
    assert split_k == 48 and 1 <= tiles <= 4 and mode in (0, 1, 2)
    NWV = tiles if mode == 1 else 1
    TPW = tiles if mode == 0 else 1  # m-tiles per wave
    NB = MT = 16
    SUBK = 32
    KC = H // split_k
    NK = KC // SUBK  # 32-k MFMA steps of the split
    SPB = block_k // SUBK  # MFMA steps per BLOCK_K
    assert KC % block_k == 0 and block_k % SUBK == 0
    NBLK = E // NB
    name = (
        f"router_gemm_bf16x3_direct_h6144_e128_mfma16_s{split_k}_bk{block_k}_t{tiles}_m{mode}"
        + ("_lines" if lines else "")
        + ("_token_partials" if token_major else "")
    )

    def mfma_op(a, b, acc):
        av = Vec(a).bitcast(fx.BFloat16)
        bv = Vec(b).bitcast(fx.BFloat16)
        return Vec(
            fx.rocdl.mfma_f32_16x16x32_bf16(
                Vec.make_type(4, fx.Float32), [av, bv, acc, 0, 0, 0]
            )
        )

    @flyc.kernel(name=name, known_block_size=[64 * NWV, 1, 1])
    def kernel(
        arg_x: fx.Int64,  # [m, H] bf16, row stride i32_sx
        arg_w: fx.Int64,  # [E, H] fp32, or its line layout (``lines``)
        arg_p: fx.Int64,  # partials [split_k, m, E] fp32
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
    ):
        tx, bid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        s = bid % split_k
        nb = (bid // split_k) % NBLK
        if const_expr(mode == 1):
            t_first = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64)) * MT
        elif const_expr(mode == 2):
            t_first = bid // (split_k * NBLK) * MT
        else:
            t_first = fx.Int32(0)
        i = lane % 16
        q = lane // 16
        wr = buffer_ops.create_buffer_resource_from_addr(
            arg_w, num_records_bytes=E * H * 4
        )
        xr = buffer_ops.create_buffer_resource_from_addr(
            arg_x, num_records_bytes=fx.Int64(i32_m) * fx.Int64(i32_sx * 2)
        )
        pr = buffer_ops.create_buffer_resource_from_addr(
            arg_p, num_records_bytes=fx.Int64(i32_m) * (split_k * E * 4)
        )
        k0 = s * KC + q * 8
        # weight: expert nb * 16 + i, 8 fp32 per 32-k step (two dwordx4)
        wl = []
        for kk in range_constexpr(NK):
            for h in range_constexpr(2):
                if const_expr(lines):
                    # [k-step, n-block, h, q, i, 4]: k-step s * NK + kk, lane = q * 16 + i
                    off = (
                        (s * NK + kk) * (E * SUBK)
                        + nb * (NB * SUBK)
                        + h * 256
                        + lane * 4
                    )
                else:
                    off = (nb * NB + i) * H + k0 + kk * SUBK + h * 4
                wl.append(
                    Vec(buffer_ops.buffer_load(wr, off, vec_width=4, dtype=fx.Int32))
                )
        xl = []
        for tl in range_constexpr(TPW):
            row = t_first + tl * MT + i
            for kk in range_constexpr(NK):
                xl.append(
                    Vec(
                        buffer_ops.buffer_load(
                            xr,
                            (row * i32_sx + k0 + kk * SUBK) >> 1,
                            vec_width=4,
                            dtype=fx.Int32,
                            mask=row < i32_m,
                        )
                    )
                )
        # three bf16 terms of each step's 8 weights, 4 dwords each in k order
        terms = []
        for kk in range_constexpr(NK):
            a = _split3([fx.Int32(wl[2 * kk][e]) for e in range_constexpr(4)])
            b = _split3([fx.Int32(wl[2 * kk + 1][e]) for e in range_constexpr(4)])
            terms.append(
                [Vec.from_elements(a[p] + b[p], fx.Int32) for p in range_constexpr(3)]
            )
        for tl in range_constexpr(TPW):
            acc_hi = Vec.filled(4, 0.0, fx.Float32)
            acc_lo = Vec.filled(4, 0.0, fx.Float32)
            for st in range_constexpr(NK // SPB):
                for j in range_constexpr(SPB):
                    kk = st * SPB + j
                    acc_hi = mfma_op(terms[kk][0], xl[tl * NK + kk], acc_hi)
                for p in range_constexpr(1, 3):
                    for j in range_constexpr(SPB):
                        kk = st * SPB + j
                        acc_lo = mfma_op(terms[kk][p], xl[tl * NK + kk], acc_lo)
            out = [
                fx.Float32(acc_hi[e]) + fx.Float32(acc_lo[e])
                for e in range_constexpr(4)
            ]
            m = t_first + tl * MT + i
            buffer_ops.buffer_store(
                Vec.from_elements(out, fx.Float32),
                pr,
                ((m * split_k + s) if const_expr(token_major) else (s * i32_m + m)) * E
                + nb * NB
                + q * 4,
                mask=m < i32_m,
            )

    @flyc.jit
    def launch_router_gemm_direct_mxfp4_moe(
        arg_x: fx.Int64,
        arg_w: fx.Int64,
        arg_p: fx.Int64,
        i32_m: fx.Int32,
        i32_sx: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(arg_x, arg_w, arg_p, i32_m, i32_sx).launch(
            grid=(fx.Int64(NBLK * split_k * (tiles if mode == 2 else 1)), 1, 1),
            block=(64 * NWV, 1, 1),
            stream=stream,
        )

    launch_router_gemm_direct_mxfp4_moe.kernel_name = name
    launch_router_gemm_direct_mxfp4_moe.compile_hints = PRELOAD_HINTS
    return launch_router_gemm_direct_mxfp4_moe


@functools.cache
def _get_kernel(
    split_k: int,
    block_k: int,
    tiles: int,
    mode: int = 0,
    lines: bool = False,
    token_major: bool = False,
):
    return compile_router_gemm_direct(split_k, block_k, tiles, mode, lines, token_major)
