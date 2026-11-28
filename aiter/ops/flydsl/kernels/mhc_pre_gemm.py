# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""fn-resident mHC pre GEMM, one 1024-wide K slice per split (FlyDSL, gfx950).

``out[s, row, n]`` and ``sqrsum[s, row]`` are the partial GEMM and the sum of squares of x
over split s, as ``mhc_pre_gemm_sqrsum`` returns them. The fn slice sits in LDS as bf16
hi/lo MFMA A fragments. Waves stream 16-row x tiles with no barrier after the fill.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops as bops
from aiter.ops.flydsl.kernels.tensor_shim import (
    AITER_FLYDSL_KERNARG_PRELOAD,
    AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
    ptr_rsrc,
)

KS = 1024
NCH = KS // 32
NFRAG = 3
# Words per chunk in LDS: NFRAG fragments of 64 lanes x 16 B, plus a 4-slot pad that spreads
# the fill's scattered writes over all LDS banks.
CW = (NFRAG * 64 + 4) * 4
XT = 8
WAVE = 64
N_FN = 24
# Byte voffset past any buffer range: a raw buffer load there reads 0, a store is dropped.
OOB = 0x7FFFFFFF


def opaque(v):
    # Empty asm that passes v through. LLVM cannot hoist or sink code across it: on the
    # lane it keeps the fragment reads inside the tile loop, where they would spill.
    raw = v.ir_value()
    out = _llvm.inline_asm(raw.type, [raw], "", "=v,0", has_side_effects=True)
    return type(v)(out)


def read_frags(frag, lane_v, c):
    fbase = c * CW + lane_v * 4
    return [
        Vec(
            fx.ptr_load(fx.add_offset(frag.ptr, fbase + q * WAVE * 4), T.vec(4, T.i32))
        ).bitcast(fx.BFloat16)
        for q in range(NFRAG)
    ]


def permlane32_sum(v):
    # v plus the same v from lane l ^ 32 (v_permlane32_swap), for every lane.
    raw = v.bitcast(fx.Int32).ir_value()
    pair_ty = ir.Type.parse("!llvm.struct<(i32, i32)>")
    swapped = rocdl.permlane32_swap(pair_ty, raw, raw, False, True)
    lhs = fx.Int32(_llvm.extractvalue(T.i32, swapped, [0])).bitcast(fx.Float32)
    rhs = fx.Int32(_llvm.extractvalue(T.i32, swapped, [1])).bitcast(fx.Float32)
    return lhs + rhs


def mfma(a_bf16, b_bf16, acc):
    atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.BFloat16))
    fa = fx.make_rmem_tensor(8, fx.BFloat16)
    fb = fx.make_rmem_tensor(8, fx.BFloat16)
    fc = fx.make_rmem_tensor(4, fx.Float32)
    fx.memref_store_vec(a_bf16, fa)
    fx.memref_store_vec(b_bf16, fb)
    fx.memref_store_vec(acc, fc)
    fx.gemm(atom, fc, fa, fb, fc)
    return fx.memref_load_vec(fc)


@functools.cache
def build_mhc_pre_gemm(K: int, num_waves: int, fn_packed: bool):
    """JIT launcher for hidden size ``K``, ``num_waves`` waves, and one fn layout."""
    if K % KS:
        raise ValueError(f"K {K} must be a multiple of {KS}")
    if num_waves & (num_waves - 1) or num_waves < 1:
        raise ValueError(f"num_waves must be a power of two, got {num_waves}")
    split_k = K // KS
    threads = num_waves * WAVE
    wave_shift = num_waves.bit_length() - 1
    n_seg = KS // 32
    if (3 * n_seg) % num_waves:
        raise ValueError(
            f"fill segments {3 * n_seg} do not split over {num_waves} waves"
        )
    per_wave = 3 * n_seg // num_waves
    name = (
        f"mhc_pre_gemm_sqrsum_fnres_flydsl_k{K}_w{num_waves}"
        f"_{'packed' if fn_packed else 'f32'}"
    )

    @fx.struct
    class Smem:
        frag: fx.Array[fx.Int32, NCH * CW, 16]

    @flyc.kernel(name=name, known_block_size=[threads, 1, 1])
    def mhc_pre_gemm_kernel(
        fn: fx.Pointer,
        x: fx.Pointer,
        out: fx.Pointer,
        sqrsum: fx.Pointer,
        m: fx.Int32,
        rows_per_group: fx.Int32,
        out_stride: fx.Int32,
    ):
        F32, BF16, I32 = fx.Float32, fx.BFloat16, fx.Int32
        tid = I32(fx.thread_idx.x)
        lane = tid % WAVE
        warp = tid // WAVE
        row_l = lane % 16
        grp = lane // 16
        row0 = I32(fx.block_idx.x) * rows_per_group
        k_split = I32(fx.block_idx.y)
        frag = fx.SharedAllocator().allocate(Smem).peek().frag

        fn_rsrc = ptr_rsrc(fn, num_records_bytes=N_FN * K * (4))
        x_rsrc = ptr_rsrc(x, num_records_bytes=fx.Int64(m) * (K * 2))
        out_rsrc = ptr_rsrc(
            out, num_records_bytes=fx.Int64(m) * fx.Int64(out_stride) * (split_k * 4)
        )
        sq_rsrc = ptr_rsrc(sqrsum, num_records_bytes=fx.Int64(m) * (split_k * 4))

        # Lane (rho, kp) loads 16 B of fn row 8 * rg + rho, so 8 lanes read one 128 B line.
        rho = lane // 8
        kp = lane % 8
        for i in range_constexpr(per_wave):
            q = warp * per_wave + i
            rg = q // n_seg
            seg = q % n_seg
            row = rg * 8 + rho
            k = seg * 32 + kp * 4
            low_rg = rg < 2
            f_hi = low_rg.select(I32(0), I32(2))
            f_lo = low_rg.select(I32(1), I32(2))
            r_hi = low_rg.select(row, rho)
            r_lo = low_rg.select(row, rho + 8)
            if fx.const_expr(fn_packed):
                # The 16 B is one half (hi or lo) of an 8-K group; kp selects the half.
                voff = (row * K + k_split * KS + k) * 4
                raw = Vec(
                    rocdl.raw_ptr_buffer_load(
                        T.vec(4, T.i32), fn_rsrc, voff, I32(0), aux=0
                    )
                )
                kg = (seg * 4 + kp // 2) * 8
                blk, rem = kg // (32 * XT), kg % (32 * XT)
                g, c = rem // (8 * XT), blk * XT + (rem % (8 * XT)) // 8
                odd = kp % 2 == 1
                f = odd.select(f_lo, f_hi)
                r = odd.select(r_lo, r_hi)
                word = c * CW + (f * WAVE + g * 16 + r) * 4
                fx.ptr_store(raw.ir_value(), fx.add_offset(frag.ptr, word))
            else:
                voff = (row * K + k_split * KS + k) * 4
                f4 = Vec(
                    rocdl.raw_ptr_buffer_load(
                        T.vec(4, T.f32), fn_rsrc, voff, I32(0), aux=0
                    )
                )
                hi4 = f4.to(BF16)
                lo4 = (f4 - hi4.to(F32)).to(BF16)
                blk, rem = k // (32 * XT), k % (32 * XT)
                g, c = rem // (8 * XT), blk * XT + (rem % (8 * XT)) // 8
                e2 = (rem % 8) // 2
                word_hi = c * CW + (f_hi * WAVE + g * 16 + r_hi) * 4 + e2
                word_lo = c * CW + (f_lo * WAVE + g * 16 + r_lo) * 4 + e2
                fx.ptr_store(
                    Vec(hi4).bitcast(I32).ir_value(), fx.add_offset(frag.ptr, word_hi)
                )
                fx.ptr_store(
                    Vec(lo4).bitcast(I32).ir_value(), fx.add_offset(frag.ptr, word_lo)
                )
        gpu.barrier()

        tiles = rows_per_group >> 4
        n_iter = (tiles + (num_waves - 1 - warp)) >> wave_shift

        def x_tile_off(j):
            x_row = row0 + (warp + j * num_waves) * 16 + row_l
            return x_row * (K * 2) + (k_split * KS + grp * (8 * XT)) * 2

        def load_block(off, b):
            xs = []
            for t in range_constexpr(XT):
                raw = rocdl.raw_ptr_buffer_load(
                    T.vec(4, T.i32),
                    x_rsrc,
                    off + (b * 32 * XT + t * 8) * 2,
                    I32(0),
                    aux=0,
                )
                xs.append(Vec(raw).bitcast(BF16))
            return xs

        loop_init = load_block(x_tile_off(I32(0)), 0)
        for j, state in range(fx.Int32(0), n_iter, fx.Int32(1), init=loop_init):
            j = I32(j)
            xcur = list(state)
            lane_t = opaque(lane)
            tile_idx = warp + j * num_waves
            x_row = row0 + tile_idx * 16 + row_l
            row_ok = x_row < m
            acc_a = Vec.filled(4, 0.0, F32)
            acc_2 = Vec.filled(4, 0.0, F32)
            sq = F32(0.0)
            ring = [read_frags(frag, lane_t, 0), read_frags(frag, lane_t, 1)]
            for b in range_constexpr(NCH // XT):
                for t in range_constexpr(XT):
                    c = b * XT + t
                    xv = xcur[t]
                    frag_hi, frag_lo, frag_rows16 = ring[c % 2]
                    acc_a = mfma(frag_hi, xv, acc_a)
                    acc_a = mfma(frag_lo, xv, acc_a)
                    acc_2 = mfma(frag_rows16, xv, acc_2)
                    for e in range_constexpr(4):
                        pair = Vec.from_elements(
                            [xv[2 * e], xv[2 * e + 1]], BF16
                        ).ir_value()
                        sq = F32(
                            rocdl.fdot2_f32_bf16_(T.f32, pair, pair, sq.ir_value())
                        )
                    # Keeps the sum per chunk. Sunk to the epilogue, it keeps every x register live.
                    sq = opaque(sq)
                    if c + 2 < NCH:
                        ring[c % 2] = read_frags(frag, lane_t, c + 2)
                    # Pins the ring. Without it the scheduler sinks each fragment read next to its MFMA.
                    rocdl.sched_barrier(0)
                if b + 1 < NCH // XT:
                    xcur = load_block(x_tile_off(j), b + 1)
                else:
                    xcur = load_block(x_tile_off(j + fx.Int32(1)), 0)

            # xor 16 and 32 sum the 4 K groups of a row.
            sq = sq + sq.shuffle_xor(16, WAVE)
            sq = sq + sq.shuffle_xor(32, WAVE)
            acc_b = Vec.from_elements(
                [permlane32_sum(acc_2[i]) for i in range_constexpr(4)], F32
            )

            # Lane grp holds columns 4 * grp to 4 * grp + 3. Rows 16-23 land at column 16 + 4 * grp.
            o_row = (k_split * m + x_row) * out_stride * 4 + grp * 16
            bops.buffer_store(
                acc_a,
                out_rsrc,
                I32(row_ok.select(o_row, OOB)),
                cache_modifier=0,
                soffset_bytes=I32(0),
                offset_is_bytes=True,
            )
            # Lanes of groups 2 and 3 zero-fill columns 24 to 31, so the row's 128 B line is
            # written whole. out_stride below 32 leaves no padding to fill.
            live = grp < 2
            acc_pad = Vec.from_elements(
                [live.select(acc_b[i], F32(0.0)) for i in range_constexpr(4)], F32
            )
            wr_b = live | (out_stride >= 32)
            voff_b = wr_b.select(row_ok.select(o_row + 64, OOB), OOB)
            bops.buffer_store(
                acc_pad,
                out_rsrc,
                I32(voff_b),
                cache_modifier=0,
                soffset_bytes=I32(0),
                offset_is_bytes=True,
            )
            voff_s = row_ok.select((k_split * m + x_row) * 4, OOB)
            voff_s = (grp == 0).select(voff_s, OOB)
            bops.buffer_store(
                sq,
                sq_rsrc,
                I32(voff_s),
                cache_modifier=0,
                soffset_bytes=I32(0),
                offset_is_bytes=True,
            )
            yield xcur

    @flyc.jit
    def launch_mhc_pre_gemm(
        fn: fx.Pointer,
        x: fx.Pointer,
        out: fx.Pointer,
        sqrsum: fx.Pointer,
        m: fx.Int32,
        rows_per_group: fx.Int32,
        out_stride: fx.Int32,
        groups: fx.Int32,
        stream: fx.Stream,
    ):
        mhc_pre_gemm_kernel(fn, x, out, sqrsum, m, rows_per_group, out_stride).launch(
            grid=(fx.Int64(groups), split_k, 1),
            block=(threads, 1, 1),
            stream=stream,
        )

    launch_mhc_pre_gemm.compile_hints = {
        "llvm_options": {
            "amdgpu-kernarg-preload": AITER_FLYDSL_KERNARG_PRELOAD,
            "amdgpu-kernarg-preload-count": AITER_FLYDSL_KERNARG_PRELOAD_COUNT,
        },
    }
    launch_mhc_pre_gemm.kernel_name = name
    return launch_mhc_pre_gemm
