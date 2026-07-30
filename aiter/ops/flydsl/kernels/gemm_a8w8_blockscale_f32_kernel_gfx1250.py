# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import math as math_dialect
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import (
    arith,
    const_expr,
    gpu,
    idx2crd,
    range_constexpr,
    rocdl,
    tdm_ops,
)
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.utils.smem_allocator import check_smem_capacity

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import store_acc_vec8_to_buffer

WMMA_M, WMMA_N, WMMA_K = 16, 16, 128
WAVE_SIZE = 32
LDS_PAD_A_BYTES = 16
B_CYCLE_BYTES = 16 * 16
B_KSTEP_BYTES = WMMA_K * 16


def _align_up(n, a):
    return ((n + a - 1) // a) * a


def lds_load_b128(lds_i8_ptr, byte_off):
    return arith.unwrap(
        fx.ptr_load(fx.add_offset(lds_i8_ptr, byte_off), result_type=T.vec(4, T.i32))
    )


def compile_gemm_a8w8_blockscale(
    *,
    K: int,
    tile_m: int = 128,
    tile_n: int = 128,
    tile_k: int = 128,
    m_warp: int = 2,
    n_warp: int = 4,
    scale_block_k: int = 128,
    scale_block_n: int = 128,
    num_buffers: int = 2,
    waves_per_eu: int | None = None,
    out_dtype: str = "bf16",
    variant: str = "compute_bound",
    kernarg_preload: bool = False,
    split_k: int = 1,
):
    num_warps, block_threads = m_warp * n_warp, m_warp * n_warp * WAVE_SIZE
    warp_tile_m, warp_tile_n = tile_m // m_warp, tile_n // n_warp

    def _req(cond, msg):
        if not cond:
            raise ValueError(msg)

    def _req_multiples(pairs):
        for name, val, div in pairs:
            _req(val % div == 0, f"{name} ({val}) must be a multiple of {div}")

    _req(variant in ("compute_bound", "memory_bound"), f"bad variant {variant!r}")
    _req(out_dtype in ("bf16", "fp16", "f32"), f"bad out_dtype {out_dtype!r}")
    _req(2 <= num_buffers <= 8, f"num_buffers must be in [2, 8], got {num_buffers}")
    _req(split_k >= 1, f"split_k must be >= 1, got {split_k}")
    # One readlane feeds the whole wave, so the W-scale must be wave-uniform.
    _req(warp_tile_n <= scale_block_n, f"warp_tile_n {warp_tile_n} > {scale_block_n}")
    _req_multiples(
        (
            ("tile_m", tile_m, WMMA_M),
            ("tile_n", tile_n, WMMA_N),
            ("tile_k", tile_k, WMMA_K),
            ("warp_tile_m", warp_tile_m, WMMA_M),
            ("warp_tile_n", warp_tile_n, WMMA_N),
            ("scale_block_k", scale_block_k, WMMA_K),
            ("tile_k", tile_k, scale_block_k),
            ("K", K, tile_k),
            ("K", K, scale_block_k),
        )
    )
    if split_k > 1:
        _req(
            variant == "memory_bound", f"split_k>1 needs memory_bound, got {variant!r}"
        )
        _req(out_dtype == "f32", "split_k>1 requires out_dtype='f32' (f32 atomic-fadd)")
        _req(K % split_k == 0, f"K ({K}) must be divisible by split_k ({split_k})")
        _req_multiples(
            (
                ("K/split_k", K // split_k, tile_k),
                ("K/split_k", K // split_k, scale_block_k),
            )
        )

    wmma_m_rep, wmma_n_rep = warp_tile_m // WMMA_M, warp_tile_n // WMMA_N
    n_accs = wmma_m_rep * wmma_n_rep
    k_wmma_steps, scales_per_tile = tile_k // WMMA_K, tile_k // scale_block_k
    wmma_steps_per_scale = scale_block_k // WMMA_K
    USE_KSTEP = k_wmma_steps > 1
    acc_coords = [
        (wm, wn, wm * wmma_n_rep + wn)
        for wm in range(wmma_m_rep)
        for wn in range(wmma_n_rep)
    ]

    split_k_chunk = K // split_k
    num_k_tiles, scale_k = split_k_chunk // tile_k, K // scale_block_k
    scale_k_per_split = split_k_chunk // scale_block_k

    # W-scale
    NUM_W_CHUNKS = (scale_k_per_split + 31) // 32
    USES_W_CHUNK_PREFETCH = NUM_W_CHUNKS > 1

    _req(
        num_k_tiles >= num_buffers - 1, f"num_k_tiles {num_k_tiles} < {num_buffers - 1}"
    )

    gpu_arch = str(get_hip_arch())
    assert gpu_arch.startswith("gfx1250"), f"Expected gfx1250, got {gpu_arch}"

    elem_bytes_d = 2 if out_dtype in ("bf16", "fp16") else 4

    lds_a_stride_bytes, lds_b_stride_bytes = tile_k + LDS_PAD_A_BYTES, tile_k * 16
    lds_a_data_bytes = tile_m * lds_a_stride_bytes
    lds_b_data_bytes = (tile_n // 16) * lds_b_stride_bytes

    xs_row_bytes = scales_per_tile * 4
    lds_x_scale_data_bytes = tile_m * xs_row_bytes

    unified_a_off = 0
    unified_b_off = _align_up(unified_a_off + num_buffers * lds_a_data_bytes, 16)
    unified_x_scale_off = _align_up(unified_b_off + num_buffers * lds_b_data_bytes, 16)
    lds_arena_bytes = _align_up(
        unified_x_scale_off + num_buffers * lds_x_scale_data_bytes, 16
    )
    check_smem_capacity(lds_arena_bytes, gpu_arch)

    _TDMS_PER_TILE = 3  # X + W + X-scale
    PROLOGUE_FILL = num_buffers if variant == "compute_bound" else num_buffers - 1
    MAIN_LOOP_ITERS = num_k_tiles - PROLOGUE_FILL
    DRAIN_COUNT = (
        PROLOGUE_FILL if MAIN_LOOP_ITERS > 0 else min(PROLOGUE_FILL, num_k_tiles)
    )
    REG_PROLOGUE_WAIT = (PROLOGUE_FILL - 1) * _TDMS_PER_TILE
    MAIN_TDM_OUTSTANDING = _TDMS_PER_TILE * (
        PROLOGUE_FILL - 1 if variant == "compute_bound" else max(0, PROLOGUE_FILL - 2)
    )

    READ_CUR = PROLOGUE_FILL < 2
    FRAGS_IN_COMPUTE = USE_KSTEP or READ_CUR

    _w_window_span = max(3, PROLOGUE_FILL) * scales_per_tile
    _req(
        not USES_W_CHUNK_PREFETCH or _w_window_span - 1 <= 32,
        f"2-chunk W-scale window too narrow: max(3, PROLOGUE_FILL)*scales_per_tile="
        f"{_w_window_span} exceeds 33. Reduce tile_k, raise scale_block_k, or lower "
        f"num_buffers.",
    )

    @flyc.kernel
    def kernel_gemm_a8w8_blockscale(
        arg_y: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_x_scale: fx.Tensor,
        arg_w_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
    ):
        tx, bx, by = gpu.thread_id("x"), gpu.block_id("x"), gpu.block_id("y")
        blk_m, blk_n = bx * arith.index(tile_m), by * arith.index(tile_n)
        if const_expr(split_k > 1):
            bz = gpu.block_id("z")
            split_k_base = bz * arith.index(split_k_chunk)
            split_kb_base = bz * arith.index(scale_k_per_split)
        else:
            split_k_base, split_kb_base = arith.index(0), arith.index(0)

        layout_thr = fx.make_layout(
            (m_warp, n_warp, 2, 16), (n_warp * WAVE_SIZE, WAVE_SIZE, 16, 1)
        )
        thr_coord = idx2crd(tx, layout_thr)
        wave_m_idx, wave_n_idx = fx.get(thr_coord, 0), fx.get(thr_coord, 1)
        lane_kgrp, lane16 = fx.get(thr_coord, 2), fx.get(thr_coord, 3)
        warp_m_base = wave_m_idx * arith.index(warp_tile_m)
        warp_n_base = wave_n_idx * arith.index(warp_tile_n)
        m_idx = arith.index_cast(T.index, i32_m.ir_value())
        n_idx = arith.index_cast(T.index, i32_n.ir_value())

        y_total_bytes = m_idx * n_idx * arith.index(elem_bytes_d)
        y_buf = fx.rocdl.get_buffer_rsrc(
            fx.rocdl.make_buffer_ptr(fx.get_iter(arg_y), y_total_bytes)
        )

        num_n_scale_blocks = (n_idx + arith.index(scale_block_n - 1)) / arith.index(
            scale_block_n
        )
        w_scale_total_bytes = num_n_scale_blocks * arith.index(scale_k) * arith.index(4)
        w_scale_buf = fx.rocdl.get_buffer_rsrc(
            fx.rocdl.make_buffer_ptr(fx.get_iter(arg_w_scale), w_scale_total_bytes)
        )

        scale_zero = arith.constant(0.0, type=T.f32)

        lds_base_ptr = fx.SharedAllocator(static=True).allocate(lds_arena_bytes)._ptr

        def _lds_region_ptr(byte_off):
            return fx.add_offset(lds_base_ptr, byte_off)

        def _slot_byte_off(buf_idx, region_off, slot_bytes):
            if const_expr(isinstance(buf_idx, int)):
                return region_off + buf_idx * slot_bytes
            return arith.index_cast(T.index, buf_idx) * arith.index(
                slot_bytes
            ) + arith.index(region_off)

        def _imm64(k_tile, mul):
            if const_expr(isinstance(k_tile, int)):
                return fx.Int64(k_tile * mul)
            as_index = arith.index_cast(
                T.index, arith.muli(k_tile, arith.constant(mul, type=T.i32))
            )
            return fx.Int64(arith.index_cast(T.i64, as_index))

        def _byte_view(tensor, byte_off, shape, stride):
            return fx.Tensor(
                fx.make_view(
                    fx.add_offset(
                        fx.recast_iter(fx.Int8, fx.get_iter(tensor)), byte_off
                    ),
                    fx.make_layout(shape, stride),
                )
            )

        def _issue_tile(buf_idx, k_tile):
            for atom, gt, roff, shape, row in _TDM_SPECS:
                fx.copy(
                    atom,
                    gt,
                    fx.Tensor(
                        fx.make_view(
                            _lds_region_ptr(
                                _slot_byte_off(buf_idx, roff, shape[0] * row)
                            ),
                            fx.make_layout(shape, (row, 1)),
                        )
                    ),
                    imm_offset=_imm64(k_tile, shape[1]),
                )

        def ds_read_x_scales(buf_idx):
            slot_elems = lds_x_scale_data_bytes // 4
            if const_expr(isinstance(buf_idx, int)):
                slot_off = arith.index(buf_idx * slot_elems)
            else:
                slot_off = arith.index_cast(
                    T.index,
                    arith.muli(buf_idx, arith.constant(slot_elems, type=T.i32)),
                )
            out = []
            for sc in range_constexpr(scales_per_tile):
                for wm in range_constexpr(wmma_m_rep):
                    row = warp_m_base + arith.index(wm * WMMA_M) + lane16
                    off = (
                        slot_off + row * arith.index(scales_per_tile) + arith.index(sc)
                    )
                    out.append(
                        arith.unwrap(fx.ptr_load(fx.add_offset(big_x_scale_mem, off)))
                    )
            return out

        def _issue_w_chunk(chunk):
            if const_expr(isinstance(chunk, int)):
                offset = arith.index(chunk * 32)
            else:
                clamped_i32 = arith.minui(
                    chunk, arith.constant(NUM_W_CHUNKS - 1, type=T.i32)
                )
                offset = arith.index_cast(
                    T.index, arith.muli(clamped_i32, arith.constant(32, type=T.i32))
                )
            idx = wave_n_block * arith.index(scale_k) + lane_id_full + offset
            if const_expr(split_k > 1):
                idx = idx + split_kb_base
            return buffer_ops.buffer_load(w_scale_buf, idx, vec_width=1, dtype=T.f32)

        def _w_readlane(kb_i32):
            if const_expr(NUM_W_CHUNKS == 1):
                return rocdl.readlane(T.f32, bulk_w_cur, kb_i32)
            kb_chunk_i32 = arith.shrui(kb_i32, arith.constant(5, type=T.i32))
            lane_in_chunk_i32 = arith.andi(kb_i32, arith.constant(31, type=T.i32))
            is_cur = arith.cmpi(arith.CmpIPredicate.eq, kb_chunk_i32, cur_chunk_idx_i32)
            chosen = arith.select(is_cur, bulk_w_cur, bulk_w_prefetch)
            return rocdl.readlane(T.f32, chosen, lane_in_chunk_i32)

        def _advance_w_chunks(next_compute_idx, one_i32):
            next_kb_i32 = arith.muli(
                next_compute_idx, arith.constant(scales_per_tile, type=T.i32)
            )
            next_chunk_i32 = arith.shrui(next_kb_i32, arith.constant(5, type=T.i32))
            need_advance = arith.cmpi(
                arith.CmpIPredicate.ne, next_chunk_i32, cur_chunk_idx_i32
            )
            new_cur = arith.select(need_advance, bulk_w_prefetch, bulk_w_cur)
            new_prefetch = _issue_w_chunk(arith.addi(next_chunk_i32, one_i32))
            return new_cur, new_prefetch, next_chunk_i32

        def _a_lane_base(wm):
            return (
                (warp_m_base + lane16) * arith.index(lds_a_stride_bytes)
                + arith.index(wm * WMMA_M * lds_a_stride_bytes)
                + k_half_byte_offset
            )

        def _b_lane_base(wn):
            stripe_offset = (
                (warp_n_base + arith.index(wn * WMMA_N))
                / arith.index(WMMA_N)
                * arith.index(lds_b_stride_bytes)
            )
            return stripe_offset + b_k_half_byte_offset + lane16 * arith.index(16)

        def _load_frag(
            lds_memref, lane_base, ks, cycle_stride_bytes=16, ks_stride_bytes=WMMA_K
        ):
            off = lane_base + arith.index(ks * ks_stride_bytes)
            v = [
                lds_load_b128(lds_memref, off + arith.index(cycle_stride_bytes * j))
                for j in range_constexpr(4)
            ]
            v01 = fx.Vector(v[0]).shuffle(fx.Vector(v[1]), list(range(8)))
            v23 = fx.Vector(v[2]).shuffle(fx.Vector(v[3]), list(range(8)))
            return v01.shuffle(v23, list(range(16))).ir_value()

        def _rmem_vec(v, n, dtype):
            t = fx.make_rmem_tensor(n, dtype)
            t.store(fx.Vector(arith.unwrap(v)))
            return t

        def _rmem_frag_lists(a_frags, b_frags):
            return (
                [_rmem_vec(f, 16, fx.Int32) for f in a_frags],
                [_rmem_vec(f, 16, fx.Int32) for f in b_frags],
            )

        def issue_wmma_step(sc_base, wm, wn, a_frags, b_frags):
            temp_t = _rmem_vec(acc_zero, 8, fx.Float32)
            for ks_inner in range_constexpr(wmma_steps_per_scale):
                ks = sc_base * wmma_steps_per_scale + ks_inner
                a_t = a_frags[ks * wmma_m_rep + wm]
                b_t = b_frags[ks * wmma_n_rep + wn]
                fx.gemm(wmma_atom, temp_t, b_t, a_t, temp_t)
            return arith.unwrap(temp_t.load())

        def apply_scale(temp, scale, acc):
            scale_vec = fx.Vector.filled(8, fx.Float32(scale), fx.Float32).ir_value()
            return math_dialect.fma(temp, scale_vec, acc)

        def _pack_state(accs_, a_, b_, x_, w_, pw_, load_idx, compute_idx):
            state = list(accs_) + list(a_) + list(b_) + list(x_) + list(w_) + list(pw_)
            if const_expr(USES_W_CHUNK_PREFETCH):
                state = state + [bulk_w_cur, bulk_w_prefetch, cur_chunk_idx_i32]
            return state + [load_idx, compute_idx]

        def _unpack_state(state):
            regs, i = [], 0
            for n in REG_STATE_SIZES:
                regs.append(list(state[i : i + n]))
                i += n
            return regs, list(state[i : i + N_W_TAIL]), state[-2], state[-1]

        def issue_w_raw_scales(k_base):
            kb_base = k_base / arith.index(scale_block_k)
            w_raw = []
            for sc in range_constexpr(scales_per_tile):
                kb_i32 = arith.index_cast(T.i32, kb_base + arith.index(sc))
                w_val = _w_readlane(kb_i32)
                for wn in range_constexpr(wmma_n_rep):
                    w_raw.append(w_val)
            return w_raw

        def issue_w_raw_scales_masked(future_tile_rt):
            future_tile_i32 = arith.index_cast(T.i32, future_tile_rt)
            valid_future = arith.cmpi(
                arith.CmpIPredicate.ult,
                future_tile_i32,
                arith.constant(num_k_tiles, type=T.i32),
            )
            safe_tile_i32 = arith.select(
                valid_future, future_tile_i32, arith.constant(0, type=T.i32)
            )
            safe_tile_idx = arith.index_cast(T.index, safe_tile_i32)
            safe_k_base = safe_tile_idx * arith.index(tile_k)
            raw_w = issue_w_raw_scales(safe_k_base)
            masked_w = [arith.select(valid_future, v, scale_zero) for v in raw_w]
            return masked_w

        def _slot_off(buffer_idx, data_bytes, stride_i32):
            if const_expr(isinstance(buffer_idx, int)):
                return arith.index(buffer_idx * data_bytes)
            return arith.index_cast(T.index, arith.muli(buffer_idx, stride_i32))

        def _a_step_frags(slot_off_a, ks):
            return [
                _load_frag(big_a_mem, _a_lane_base(wm) + slot_off_a, ks)
                for wm in range_constexpr(wmma_m_rep)
            ]

        def _b_step_frags(slot_off_b, ks):
            return [
                _load_frag(
                    big_b_mem,
                    _b_lane_base(wn) + slot_off_b,
                    ks,
                    cycle_stride_bytes=B_CYCLE_BYTES,
                    ks_stride_bytes=B_KSTEP_BYTES,
                )
                for wn in range_constexpr(wmma_n_rep)
            ]

        def load_a_frags(buffer_idx):
            slot_off_a = _slot_off(buffer_idx, lds_a_data_bytes, slot_stride_a_i32)
            return [
                f
                for ks in range_constexpr(k_wmma_steps)
                for f in _a_step_frags(slot_off_a, ks)
            ]

        def load_b_frags(buffer_idx):
            slot_off_b = _slot_off(buffer_idx, lds_b_data_bytes, slot_stride_b_i32)
            return [
                f
                for ks in range_constexpr(k_wmma_steps)
                for f in _b_step_frags(slot_off_b, ks)
            ]

        def precompute_scales(x_raw, w_raw):
            scales = []
            for sc in range_constexpr(scales_per_tile):
                row = [None] * n_accs
                for wm, wn, idx in acc_coords:
                    row[idx] = arith.mulf(
                        x_raw[sc * wmma_m_rep + wm], w_raw[sc * wmma_n_rep + wn]
                    )
                scales.append(row)
            return scales

        def compute_wmma(
            global_accs, a_frags, b_frags, x_raw, w_raw, scales=None, buf=None
        ):
            if const_expr(buf is None):
                a_ts, b_ts = _rmem_frag_lists(a_frags, b_frags)
            else:
                slot_a = _slot_off(buf, lds_a_data_bytes, slot_stride_a_i32)
                slot_b = _slot_off(buf, lds_b_data_bytes, slot_stride_b_i32)
            for sc in range_constexpr(scales_per_tile):
                sc_base = sc
                if const_expr(buf is not None):
                    a_step, b_step = [], []
                    for ki in range_constexpr(wmma_steps_per_scale):
                        ks = sc * wmma_steps_per_scale + ki
                        b_step += _b_step_frags(slot_b, ks)
                        a_step += _a_step_frags(slot_a, ks)
                    a_ts, b_ts = _rmem_frag_lists(a_step, b_step)
                    sc_base = 0
                for wm, wn, idx in acc_coords:
                    temp = issue_wmma_step(sc_base, wm, wn, a_ts, b_ts)
                    if const_expr(scales is None):
                        scale = arith.mulf(
                            x_raw[sc * wmma_m_rep + wm], w_raw[sc * wmma_n_rep + wn]
                        )
                    else:
                        scale = scales[sc][idx]
                    global_accs[idx] = apply_scale(temp, scale, global_accs[idx])
            return global_accs

        big_a_mem, big_b_mem = (
            _lds_region_ptr(unified_a_off),
            _lds_region_ptr(unified_b_off),
        )
        slot_stride_a_i32 = arith.constant(lds_a_data_bytes, type=T.i32)
        slot_stride_b_i32 = arith.constant(lds_b_data_bytes, type=T.i32)
        blk_m64, blk_n64 = fx.Int64(blk_m), fx.Int64(blk_n)
        split_k_base64 = fx.Int64(split_k_base)
        x_base_bytes = blk_m64 * fx.Int64(K) + split_k_base64
        w_base_bytes = (blk_n64 // fx.Int64(16)) * fx.Int64(
            K * 16
        ) + split_k_base64 * fx.Int64(16)

        gX = _byte_view(arg_x, x_base_bytes, (tile_m, tile_k), (tile_k, 1))
        gW = _byte_view(
            arg_w, w_base_bytes, (tile_n // 16, tile_k * 16), (tile_k * 16, 1)
        )
        atom_x = fx.rocdl.make_tdm_atom(
            gX,
            [None, None],
            strides=[fx.Int64(K), None],
            num_warps=num_warps,
            pad_interval=tile_k,
            pad_amount=LDS_PAD_A_BYTES,
        )
        atom_w = fx.rocdl.make_tdm_atom(
            gW,
            [None, None],
            strides=[fx.Int64(K * 16), None],
            num_warps=num_warps,
        )

        # ── X-scale TDM descriptor + LDS staging (hoisted) ──────────────────
        big_x_scale_mem = fx.recast_iter(
            fx.Float32, _lds_region_ptr(unified_x_scale_off)
        )

        xs_base_bytes = blk_m64 * fx.Int64(scale_k * 4) + fx.Int64(
            split_kb_base
        ) * fx.Int64(4)
        gXS = _byte_view(
            arg_x_scale, xs_base_bytes, (tile_m, xs_row_bytes), (xs_row_bytes, 1)
        )
        atom_xs = fx.rocdl.make_tdm_atom(
            gXS,
            [None, None],
            strides=[fx.Int64(scale_k * 4), None],
            num_warps=num_warps,
        )

        _TDM_SPECS = (
            (atom_x, gX, unified_a_off, (tile_m, tile_k), lds_a_stride_bytes),
            (
                atom_w,
                gW,
                unified_b_off,
                (tile_n // 16, tile_k * 16),
                lds_b_stride_bytes,
            ),
            (atom_xs, gXS, unified_x_scale_off, (tile_m, xs_row_bytes), xs_row_bytes),
        )

        wave_n_block = (blk_n + warp_n_base) / arith.index(scale_block_n)
        lane_id_full = lane_kgrp * arith.index(16) + lane16
        bulk_w_cur = bulk_w_prefetch = None
        cur_chunk_idx_i32 = arith.constant(0, type=T.i32)
        k_half_byte_offset = lane_kgrp * arith.index(64)
        b_k_half_byte_offset = lane_kgrp * arith.index(1024)
        acc_zero = arith.constant_vector(0.0, T.vec(8, T.f32))

        wmma_atom = fx.make_mma_atom(
            fx.rocdl.WMMA(WMMA_M, WMMA_N, WMMA_K, fx.Float8E4M3FN, fx.Float32)
        )

        N_ACCS = n_accs
        N_A_FRAGS = 0 if FRAGS_IN_COMPUTE else wmma_m_rep * k_wmma_steps
        N_B_FRAGS = 0 if FRAGS_IN_COMPUTE else wmma_n_rep * k_wmma_steps
        N_CUR_X_RAW = 0 if READ_CUR else scales_per_tile * wmma_m_rep
        N_CUR_W_RAW = scales_per_tile * wmma_n_rep
        N_PREFETCH_W = N_CUR_W_RAW
        REG_STATE_SIZES = (
            N_ACCS,
            N_A_FRAGS,
            N_B_FRAGS,
            N_CUR_X_RAW,
            N_CUR_W_RAW,
            N_PREFETCH_W,
        )
        N_W_TAIL = 3 if USES_W_CHUNK_PREFETCH else 0
        zero_w_raw = [scale_zero] * N_CUR_W_RAW

        # PROLOGUE — pre-fill state for main-loop iter 0.
        for i in range_constexpr(PROLOGUE_FILL):
            _issue_tile(i, i)

        bulk_w_cur = _issue_w_chunk(0)
        if const_expr(USES_W_CHUNK_PREFETCH):
            bulk_w_prefetch = _issue_w_chunk(1)
        else:
            bulk_w_prefetch = bulk_w_cur

        tdm_ops.tensor_wait(REG_PROLOGUE_WAIT)
        gpu.barrier()

        cur_b = [] if FRAGS_IN_COMPUTE else load_b_frags(0)
        cur_x_raw = [] if READ_CUR else ds_read_x_scales(0)
        cur_a = [] if FRAGS_IN_COMPUTE else load_a_frags(0)

        cur_w_raw = issue_w_raw_scales(arith.index(0))
        if const_expr(num_k_tiles > 1):
            prefetch_w_raw = issue_w_raw_scales(arith.index(tile_k))
        else:
            prefetch_w_raw = zero_w_raw

        # Accumulator init
        accs = [acc_zero] * n_accs
        load_idx_init = arith.constant(PROLOGUE_FILL, type=T.i32)
        compute_idx_init = arith.constant(0, type=T.i32)

        if const_expr(MAIN_LOOP_ITERS > 0):
            init_state = _pack_state(
                accs,
                cur_a,
                cur_b,
                cur_x_raw,
                cur_w_raw,
                prefetch_w_raw,
                load_idx_init,
                compute_idx_init,
            )

            nb_const_i32 = arith.constant(num_buffers, type=T.i32)
            one_i32, two_i32 = (
                arith.constant(1, type=T.i32),
                arith.constant(2, type=T.i32),
            )

            for tile_step, state in range(0, MAIN_LOOP_ITERS, 1, init=init_state):
                (
                    (cur_accs, cur_a, cur_b, cur_x_raw, cur_w_raw, prefetch_w_raw),
                    _w_tail,
                    cur_load_idx,
                    cur_compute_idx,
                ) = _unpack_state(state)
                if const_expr(USES_W_CHUNK_PREFETCH):
                    bulk_w_cur, bulk_w_prefetch, cur_chunk_idx_i32 = _w_tail

                # SSA buf indices for this iteration.
                load_buf_i32 = arith.remui(cur_load_idx, nb_const_i32)
                next_compute_idx = arith.addi(cur_compute_idx, one_i32)
                next_buf_i32 = arith.remui(next_compute_idx, nb_const_i32)
                cur_buf_i32 = (
                    arith.remui(cur_compute_idx, nb_const_i32)
                    if FRAGS_IN_COMPUTE
                    else None
                )

                # Main loop
                if const_expr(variant == "compute_bound"):
                    cur_accs = compute_wmma(
                        cur_accs, cur_a, cur_b, cur_x_raw, cur_w_raw, buf=cur_buf_i32
                    )
                    gpu.barrier()
                    _issue_tile(load_buf_i32, cur_load_idx)
                    tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING)
                    gpu.barrier()
                    next_b = [] if FRAGS_IN_COMPUTE else load_b_frags(next_buf_i32)
                    next_x_raw = ds_read_x_scales(next_buf_i32)
                    next_a = [] if FRAGS_IN_COMPUTE else load_a_frags(next_buf_i32)
                else:
                    tdm_ops.tensor_wait(MAIN_TDM_OUTSTANDING)
                    if const_expr(FRAGS_IN_COMPUTE):
                        gpu.barrier()
                    _issue_tile(load_buf_i32, cur_load_idx)
                    gpu.barrier()
                    if const_expr(READ_CUR):
                        cur_x_raw = ds_read_x_scales(cur_buf_i32)
                        next_x_raw = []
                    else:
                        next_x_raw = ds_read_x_scales(next_buf_i32)
                    scales = precompute_scales(cur_x_raw, cur_w_raw)
                    cur_accs = compute_wmma(
                        cur_accs, cur_a, cur_b, None, None, scales, buf=cur_buf_i32
                    )
                    next_b = [] if FRAGS_IN_COMPUTE else load_b_frags(next_buf_i32)
                    next_a = [] if FRAGS_IN_COMPUTE else load_a_frags(next_buf_i32)

                cur_a = next_a
                cur_b = next_b
                cur_x_raw = next_x_raw
                cur_w_raw = prefetch_w_raw
                future_tile_i32 = arith.addi(cur_compute_idx, two_i32)
                future_tile_idx = arith.index_cast(T.index, future_tile_i32)
                prefetch_w_raw = issue_w_raw_scales_masked(future_tile_idx)

                if const_expr(USES_W_CHUNK_PREFETCH):
                    (
                        bulk_w_cur,
                        bulk_w_prefetch,
                        cur_chunk_idx_i32,
                    ) = _advance_w_chunks(next_compute_idx, one_i32)

                results = yield _pack_state(
                    cur_accs,
                    cur_a,
                    cur_b,
                    cur_x_raw,
                    cur_w_raw,
                    prefetch_w_raw,
                    arith.addi(cur_load_idx, one_i32),
                    next_compute_idx,
                )

            final_compute_idx = results[-1]
            if const_expr(USES_W_CHUNK_PREFETCH):
                bulk_w_cur, bulk_w_prefetch, cur_chunk_idx_i32 = results[-5:-2]
            accs = list(results[:N_ACCS])
        else:
            accs = list(accs)
            final_compute_idx = arith.constant(0, type=T.i32)

        # EPILOGUE - scf.for drain, small carry (accs + tile idx, frags reloaded fresh); descending tensorcnt via range_constexpr if-ladder; DRAIN_COUNT tiles, no new TDMs.
        nb_const_i32_d = arith.constant(num_buffers, type=T.i32)
        one_i32_d = arith.constant(1, type=T.i32)
        drain_init = list(accs) + [final_compute_idx]
        for _drain_i, dstate in range(0, DRAIN_COUNT, 1, init=drain_init):
            accs_d = list(dstate[:N_ACCS])
            cur_dci_d = dstate[-1]
            drain_buf_i32 = arith.remui(cur_dci_d, nb_const_i32_d)

            # cur tile == num_k_tiles-1-_k  ->  _k later tiles still in flight.
            for _k in range_constexpr(DRAIN_COUNT):
                _is_k = arith.cmpi(
                    arith.CmpIPredicate.eq,
                    cur_dci_d,
                    arith.constant(num_k_tiles - 1 - _k, type=T.i32),
                )
                if _is_k:
                    tdm_ops.tensor_wait(_k * _TDMS_PER_TILE)
            gpu.barrier()

            # Reload this tile's operands FRESH (not carried across the loop).
            cur_b_d = [] if FRAGS_IN_COMPUTE else load_b_frags(drain_buf_i32)
            cur_x_d = ds_read_x_scales(drain_buf_i32)
            cur_a_d = [] if FRAGS_IN_COMPUTE else load_a_frags(drain_buf_i32)
            cur_w_d = issue_w_raw_scales_masked(arith.index_cast(T.index, cur_dci_d))
            accs_d = compute_wmma(
                accs_d,
                cur_a_d,
                cur_b_d,
                cur_x_d,
                cur_w_d,
                buf=drain_buf_i32 if FRAGS_IN_COMPUTE else None,
            )
            dresults = yield list(accs_d) + [arith.addi(cur_dci_d, one_i32_d)]
        accs = list(dresults[:N_ACCS])

        # Step 4: convert f32 accs to out_dtype, buffer_store to Y.
        if const_expr(num_buffers > 2):
            rocdl.sched_barrier(0)

        out_elem = (
            T.bf16 if out_dtype == "bf16" else T.f16 if out_dtype == "fp16" else None
        )
        is_half_out = out_dtype in ("bf16", "fp16")

        if const_expr(split_k > 1):
            zero_i32_s = arith.constant(0, type=T.i32)
        for wm, wn, idx in acc_coords:
            row = blk_m + warp_m_base + arith.index(wm * WMMA_M) + lane16
            col_base = (
                blk_n
                + warp_n_base
                + arith.index(wn * WMMA_N)
                + lane_kgrp * arith.index(8)
            )

            if const_expr(split_k > 1):
                for e in range_constexpr(8):
                    byte_off = arith.index_cast(
                        T.i32,
                        (row * n_idx + col_base + arith.index(e)) * arith.index(4),
                    )
                    rocdl.raw_ptr_buffer_atomic_fadd(
                        fx.Vector(accs[idx])[e].ir_value(),
                        y_buf,
                        byte_off,
                        zero_i32_s,
                        zero_i32_s,
                    )
            elif is_half_out:
                c_off_bytes = (row * n_idx + col_base) * arith.index(elem_bytes_d)
                store_acc_vec8_to_buffer(
                    accs[idx],
                    y_buf,
                    c_off_bytes,
                    out_elem=out_elem,
                    offset_is_bytes=True,
                )
            else:
                offsets = []
                for half in range_constexpr(2):
                    col = col_base + arith.index(half * 4)
                    offsets.append(row * n_idx + col)
                store_acc_vec8_to_buffer(accs[idx], y_buf, offsets)

    @flyc.jit
    def launch_gemm_a8w8_blockscale(
        arg_y: fx.Tensor,
        arg_x: fx.Tensor,
        arg_w: fx.Tensor,
        arg_x_scale: fx.Tensor,
        arg_w_scale: fx.Tensor,
        i32_m: fx.Int32,
        i32_n: fx.Int32,
        stream: fx.Stream,
    ):
        idx_m = arith.index_cast(T.index, i32_m.ir_value())
        idx_n = arith.index_cast(T.index, i32_n.ir_value())
        gx = _raw((idx_m + arith.index(tile_m - 1)) / arith.index(tile_m))
        gy = _raw((idx_n + arith.index(tile_n - 1)) / arith.index(tile_n))

        wpe = int(waves_per_eu) if waves_per_eu is not None else 0
        launcher = kernel_gemm_a8w8_blockscale(
            arg_y,
            arg_x,
            arg_w,
            arg_x_scale,
            arg_w_scale,
            i32_m,
            i32_n,
            value_attrs={
                "rocdl.flat_work_group_size": f"{block_threads},{block_threads}",
                "rocdl.waves_per_eu": wpe if wpe >= 1 else None,
            },
        )

        # Mark kernel args as inreg so AMDGPU preloads them into user SGPRs at dispatch.
        if kernarg_preload:
            ctx = CompilationContext.get_current()
            inreg_attr = ir.UnitAttr.get()
            for op in ctx.gpu_module_body.operations:
                if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
                    num_args = len(op.regions[0].blocks[0].arguments)
                    per_arg = [
                        ir.DictAttr.get({"llvm.inreg": inreg_attr})
                        for _ in range(num_args)
                    ]
                    op.attributes["arg_attrs"] = ir.ArrayAttr.get(per_arg)

        launcher.launch(
            grid=(gx, gy, split_k), block=(block_threads, 1, 1), stream=stream
        )

    # Commented out until coexec branch is merged.
    launch_gemm_a8w8_blockscale.compile_hints["llvm_options"] = {
        "amdgpu-expert-scheduling-mode": True,
        # "amdgpu-anti-hints-for-va-vdst": True,
        # "amdgpu-enable-static-simulator": True,
        # "amdgpu-static-sim-inline": True,
        # "amdgpu-sched-strategy": "coexec",
        "unroll-threshold": 0,
        # "amdgpu-block-carried-latency": EnumOpt("all"),  # enable per the note above
    }

    return launch_gemm_a8w8_blockscale


__all__ = ["compile_gemm_a8w8_blockscale"]
