# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""RMSNorm kernel builder using the @flyc.kernel API.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

Optimised paths:
  - Tile-fast path for N % (BLOCK_THREADS * VEC_WIDTH) == 0 using buffer_load/store.
  - Vector-generic path for arbitrary N using vectorised buffer_load/store for the bulk
    and a scalar tail only for the final leftover elements.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext

from flydsl.expr import arith, vector, gpu, range_constexpr
from flydsl.expr.typing import T, Int32
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch

from flydsl._mlir import ir
from flydsl.expr import buffer_ops


KERNEL_NAME = "rmsnorm"

EPS = 1e-5

import math
from kernels.kernels_common import dtype_to_elem_type, get_warp_size

WARP_SIZE = get_warp_size
VEC_WIDTH = 8


def _select_block_threads(N: int, dtype_str: str) -> int:
    # Prefer 256 threads for faster block reduction; medium-width bf16 paths use
    # bounded vector caching instead of smaller blocks.
    return 256


def build_rmsnorm_module(M: int, N: int, dtype_str: str):
    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    BLOCK_THREADS = _select_block_threads(N, dtype_str)
    UNROLL = 2 if (dtype_str != "f32" and N <= 4096) else 1
    USE_GENERIC_X_CACHE = (dtype_str != "f32" and N <= 4096)

    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16

    allocator = SmemAllocator(None, arch=arch)
    f32_bytes = 4
    red_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = red_offset + RED_SLOTS * f32_bytes

    @flyc.kernel
    def rmsnorm_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        _Unused: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_type = dtype_to_elem_type(dtype_str)
        compute_type = T.f32

        fm_fast = arith.FastMathFlags.fast
        eps_c = arith.constant(EPS, type=compute_type)
        n_float = arith.constant(float(N), type=compute_type)

        base_ptr = allocator.get_base()
        s_red = SmemPtr(base_ptr, red_offset, T.f32, shape=(RED_SLOTS,))
        s_red.get()

        def wave_reduce_add(x):
            width_i32 = fx.Int32(WARP_SIZE)
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = fx.Int32(WARP_SIZE // (2 << _sh_exp))
                peer = w.shuffle_xor(off, width_i32)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            if RED_SLOTS == 1:
                return wave_reduce_add(val)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val)

            if lane == fx.Int32(0):
                wave_idx = arith.index_cast(T.index, wave)
                s_red.store(w0, [wave_idx])
            gpu.barrier()

            if wave == fx.Int32(0):
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, fx.Int32(0))
                lane_safe_idx = arith.index_cast(T.index, lane_safe)
                v0 = s_red.load([lane_safe_idx])
                z = fx.Float32(0.0)
                ww0 = in_range.select(v0, z)
                ww0 = wave_reduce_add(ww0)

                if lane == fx.Int32(0):
                    c0_idx = fx.Index(0)
                    s_red.store(ww0, [c0_idx])
            gpu.barrier()

            c0_idx = fx.Index(0)
            return s_red.load([c0_idx])

        from flydsl.expr.arith import ArithValue

        elem_bytes = 4 if dtype_str == "f32" else 2
        vec_dwords = (VEC_WIDTH * elem_bytes) // 4

        vec_type_c = T.vec(VEC_WIDTH, compute_type)
        vec_type_e = T.vec(VEC_WIDTH, elem_type)

        in_rsrc = buffer_ops.create_buffer_resource(Input, max_size=True)
        out_rsrc = buffer_ops.create_buffer_resource(Output, max_size=True)
        gamma_rsrc = buffer_ops.create_buffer_resource(Gamma, max_size=True)

        row_soffset = ArithValue(bid) * (N * elem_bytes)

        def _load_vec(rsrc, col_byte_off, soff=None):
            dw = col_byte_off >> fx.Int32(2)
            raw = buffer_ops.buffer_load(
                rsrc, dw, vec_width=vec_dwords, dtype=T.i32, soffset_bytes=soff
            )
            if vec_dwords == VEC_WIDTH:
                return raw.bitcast(vec_type_e)
            return vector.bitcast(vec_type_e, raw)

        def _store_vec(data, rsrc, col_byte_off, soff=None):
            dw = col_byte_off >> fx.Int32(2)
            buffer_ops.buffer_store(data, rsrc, dw, soffset_bytes=soff)

        def _pack_output_vec(y_val):
            if dtype_str == "bf16":
                if USE_HW_CVT_PK_BF16_F32:
                    out_e = y_val.truncf(vec_type_e)
                else:
                    vec_i32_ty = T.vec(VEC_WIDTH, T.i32)
                    vec4_i32_ty = T.vec(VEC_WIDTH // 2, T.i32)
                    vec_bf16_ty = T.vec(VEC_WIDTH, elem_type)
                    c16_i32 = arith.constant(16, type=T.i32)
                    c16_v = vector.broadcast(vec_i32_ty, c16_i32)
                    u = y_val.bitcast(vec_i32_ty)
                    upper = u.shrui(c16_v)
                    c1_v = vector.broadcast(vec_i32_ty, arith.constant(1, type=T.i32))
                    lsb = upper & c1_v
                    c7fff_v = vector.broadcast(vec_i32_ty, arith.constant(0x7FFF, type=T.i32))
                    bias = ArithValue(c7fff_v) + ArithValue(lsb)
                    u_round = ArithValue(u) + bias
                    bf16_bits = u_round.shrui(c16_v)
                    even = vector.shuffle(bf16_bits, bf16_bits, [0, 2, 4, 6])
                    odd = vector.shuffle(bf16_bits, bf16_bits, [1, 3, 5, 7])
                    odd_sh = odd << vector.broadcast(vec4_i32_ty, c16_i32)
                    out_e = vector.bitcast(vec_bf16_ty, even | odd_sh)
            elif dtype_str == "f32":
                out_e = y_val
            else:
                out_e = y_val.truncf(vec_type_e)

            i32_vec_ty = T.vec(vec_dwords, T.i32)
            if vec_dwords != VEC_WIDTH:
                return vector.bitcast(i32_vec_ty, out_e)
            return out_e.bitcast(i32_vec_ty)

        # ==================================================================
        # Tile-fast path: full tiles only. No x caching to reduce register use.
        # ==================================================================
        if N >= tile_cols and N % tile_cols == 0:
            num_tiles = N // tile_cols
            thr_col_bytes = ArithValue(tid) * (VEC_WIDTH * elem_bytes)

            c_zero_f = arith.constant(0.0, type=compute_type)
            thread_sumsq = c_zero_f

            # Pass 1: sumsq only
            for tile_i in range_constexpr(num_tiles):
                col_bytes = ArithValue(thr_col_bytes) + (tile_i * tile_cols * elem_bytes)
                vec_e = _load_vec(in_rsrc, col_bytes, soff=row_soffset)
                x = vec_e if dtype_str == "f32" else vec_e.extf(vec_type_c)
                x_av = ArithValue(x)
                x2 = x_av * x_av
                red2 = vector.reduction(compute_type, vector.CombiningKind.ADD, x2, fastmath=fm_fast)
                thread_sumsq = ArithValue(thread_sumsq) + red2

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = ArithValue(sum_sq) / n_float
            ms_eps = ArithValue(mean_sq) + eps_c
            rrms = ms_eps.rsqrt(fastmath=fm_fast)
            rrms_splat = vector.broadcast(vec_type_c, rrms)
            rrms_splat_av = ArithValue(rrms_splat)

            # Pass 2: reload x + gamma + store
            for tile_i in range_constexpr(num_tiles):
                col_bytes = ArithValue(thr_col_bytes) + (tile_i * tile_cols * elem_bytes)

                x_e = _load_vec(in_rsrc, col_bytes, soff=row_soffset)
                g_e = _load_vec(gamma_rsrc, col_bytes)

                x = x_e if dtype_str == "f32" else x_e.extf(vec_type_c)
                g = g_e if dtype_str == "f32" else g_e.extf(vec_type_c)

                y_val = (ArithValue(x) * rrms_splat_av) * ArithValue(g)
                out_vec = _pack_output_vec(y_val)
                _store_vec(out_vec, out_rsrc, col_bytes, soff=row_soffset)

        else:
            # ==============================================================
            # Vector-generic path: vectorised bulk + scalar tail only.
            # ==============================================================
            row_in = fx.slice(Input, (bid, None))
            row_out = fx.slice(Output, (bid, None))

            copy_atom_s = fx.make_copy_atom(fx.UniversalCopy(elem_bits), elem_bits)
            scalar_reg_ty = fx.MemRefType.get(
                elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register
            )
            scalar_reg_lay = fx.make_layout(1, 1)

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            def _load_scalar(divided_tensor, index):
                view = fx.slice(divided_tensor, (None, index))
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                fx.copy_atom_call(copy_atom_s, view, r)
                v = fx.memref_load_vec(r)
                return vector.extract(v, static_position=[0])

            def _store_scalar(divided_tensor, index, val):
                r = fx.memref_alloca(scalar_reg_ty, scalar_reg_lay)
                vec_ty = T.vec(1, elem_type)
                v = vector.from_elements(vec_ty, [val])
                fx.memref_store_vec(v, r)
                view = fx.slice(divided_tensor, (None, index))
                fx.copy_atom_call(copy_atom_s, r, view)

            c_zero_f = arith.constant(0.0, type=compute_type)
            thread_sumsq = c_zero_f

            total_vecs = N // VEC_WIDTH
            tail_start = total_vecs * VEC_WIDTH

            x_cache = []

            # Pass 1a: vectorised bulk
            for vec_base_int in range_constexpr(0, total_vecs, BLOCK_THREADS * UNROLL):
                c_total_vecs_i32 = Int32(total_vecs)
                for u in range_constexpr(UNROLL):
                    vec_idx = tid + vec_base_int + (u * BLOCK_THREADS)
                    in_range = arith.cmpi(arith.CmpIPredicate.ult, vec_idx, c_total_vecs_i32)
                    safe_vec_idx = in_range.select(vec_idx, Int32(0))
                    col_bytes = ArithValue(safe_vec_idx) * (VEC_WIDTH * elem_bytes)
                    x_e = _load_vec(in_rsrc, col_bytes, soff=row_soffset)
                    x = x_e if dtype_str == "f32" else x_e.extf(vec_type_c)
                    if USE_GENERIC_X_CACHE:
                        x_cache.append((in_range, col_bytes, x))
                    x_av = ArithValue(x)
                    x2 = x_av * x_av
                    red2 = vector.reduction(
                        compute_type, vector.CombiningKind.ADD, x2, fastmath=fm_fast
                    )
                    red2_safe = in_range.select(red2, c_zero_f)
                    thread_sumsq = ArithValue(thread_sumsq) + red2_safe

            # Pass 1b: scalar tail
            for idx_int in range_constexpr(tail_start, N):
                idx = Int32(idx_int)
                if arith.cmpi(arith.CmpIPredicate.eq, tid, fx.Int32(0)):
                    x_e = _load_scalar(row_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.extf(compute_type)
                    x2 = ArithValue(x) * ArithValue(x)
                    thread_sumsq = ArithValue(thread_sumsq) + x2

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = ArithValue(sum_sq) / n_float
            ms_eps = ArithValue(mean_sq) + eps_c
            rrms = ms_eps.rsqrt(fastmath=fm_fast)
            rrms_splat = vector.broadcast(vec_type_c, rrms)
            rrms_splat_av = ArithValue(rrms_splat)

            # Pass 2a: vectorised bulk
            if USE_GENERIC_X_CACHE:
                for in_range, col_bytes, x in x_cache:
                    g_e = _load_vec(gamma_rsrc, col_bytes)
                    g = g_e if dtype_str == "f32" else g_e.extf(vec_type_c)
                    y_val = (ArithValue(x) * rrms_splat_av) * ArithValue(g)
                    out_vec = _pack_output_vec(y_val)
                    if in_range:
                        _store_vec(out_vec, out_rsrc, col_bytes, soff=row_soffset)
            else:
                for vec_base_int in range_constexpr(0, total_vecs, BLOCK_THREADS * UNROLL):
                    c_total_vecs_i32 = Int32(total_vecs)
                    for u in range_constexpr(UNROLL):
                        vec_idx = tid + vec_base_int + (u * BLOCK_THREADS)
                        in_range = arith.cmpi(arith.CmpIPredicate.ult, vec_idx, c_total_vecs_i32)
                        safe_vec_idx = in_range.select(vec_idx, Int32(0))
                        col_bytes = ArithValue(safe_vec_idx) * (VEC_WIDTH * elem_bytes)
                        x_e = _load_vec(in_rsrc, col_bytes, soff=row_soffset)
                        g_e = _load_vec(gamma_rsrc, col_bytes)

                        x = x_e if dtype_str == "f32" else x_e.extf(vec_type_c)
                        g = g_e if dtype_str == "f32" else g_e.extf(vec_type_c)

                        y_val = (ArithValue(x) * rrms_splat_av) * ArithValue(g)
                        out_vec = _pack_output_vec(y_val)
                        if in_range:
                            _store_vec(out_vec, out_rsrc, col_bytes, soff=row_soffset)

            # Pass 2b: scalar tail
            for idx_int in range_constexpr(tail_start, N):
                idx = Int32(idx_int)
                if arith.cmpi(arith.CmpIPredicate.eq, tid, fx.Int32(0)):
                    x_e = _load_scalar(row_div, idx)
                    g_e = _load_scalar(gamma_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.extf(compute_type)
                    g = g_e if dtype_str == "f32" else g_e.extf(compute_type)
                    y = (ArithValue(x) * ArithValue(rrms)) * ArithValue(g)
                    if dtype_str == "f32":
                        y_e = y
                    else:
                        y_e = y.truncf(elem_type)
                    _store_scalar(out_div, idx, y_e)

    @flyc.jit
    def launch_rmsnorm(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()

        idx_m = arith.index_cast(T.index, m_in)
        launcher = rmsnorm_kernel(Input, Gamma, Gamma, Output)
        launcher.launch(
            grid=(idx_m, 1, 1), 
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_rmsnorm

