# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""RMSNorm kernel builder using the @flyc.kernel API.

RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma

Two paths:
  - Fast path (N % tile_cols == 0): buffer_load/store vectorised access.
  - Generic path (arbitrary N): scalar copy_atom_call.
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

BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()
VEC_WIDTH = 8


def build_rmsnorm_module(M: int, N: int, dtype_str: str):
    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    tile_cols = BLOCK_THREADS * VEC_WIDTH
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16

    allocator = SmemAllocator(None, arch=arch)
    f32_bytes = 4
    red_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = red_offset + RED_SLOTS * f32_bytes
    red2_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = red2_offset + RED_SLOTS * f32_bytes

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
        s_red2 = SmemPtr(base_ptr, red2_offset, T.f32, shape=(RED_SLOTS,))
        s_red.get()
        s_red2.get()

        def wave_reduce_add(x):
            width_i32 = fx.Int32(WARP_SIZE)
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = fx.Int32(WARP_SIZE // (2 << _sh_exp))
                peer = w.shuffle_xor(off, width_i32)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add(val):
            dummy = fx.Float32(0.0)
            r0, _ = block_reduce_add2(val, dummy)
            return r0

        def block_reduce_add2(val0, val1):
            if RED_SLOTS == 1:
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == fx.Int32(0):
                wave_idx = arith.index_cast(T.index, wave)
                s_red.store(w0, [wave_idx])
                s_red2.store(w1, [wave_idx])
            gpu.barrier()

            if wave == fx.Int32(0):
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, fx.Int32(0))
                lane_safe_idx = arith.index_cast(T.index, lane_safe)
                v0 = s_red.load([lane_safe_idx])
                v1 = s_red2.load([lane_safe_idx])
                z = fx.Float32(0.0)
                ww0 = in_range.select(v0, z)
                ww1 = in_range.select(v1, z)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == fx.Int32(0):
                    c0_idx = fx.Index(0)
                    s_red.store(ww0, [c0_idx])
                    s_red2.store(ww1, [c0_idx])
            gpu.barrier()

            c0_idx = fx.Index(0)
            return s_red.load([c0_idx]), s_red2.load([c0_idx])

        # ==================================================================
        # Fast path: N is a multiple of tile_cols
        # ==================================================================
        if N >= tile_cols and N % tile_cols == 0 and elem_bits <= 16:
            from flydsl.expr.arith import ArithValue

            num_tiles = N // tile_cols
            elem_bytes = 4 if dtype_str == "f32" else 2
            vec_dwords = (VEC_WIDTH * elem_bytes) // 4

            vec_type_c = T.vec(VEC_WIDTH, compute_type)
            vec_type_e = T.vec(VEC_WIDTH, elem_type)

            in_rsrc = buffer_ops.create_buffer_resource(Input, max_size=True)
            out_rsrc = buffer_ops.create_buffer_resource(Output, max_size=True)
            gamma_rsrc = buffer_ops.create_buffer_resource(Gamma, max_size=True)

            row_soffset = ArithValue(bid) * (N * elem_bytes)
            thr_col_bytes = ArithValue(tid) * (VEC_WIDTH * elem_bytes)

            def _load_vec(rsrc, col_byte_off, soff=None):
                dw = col_byte_off >> fx.Int32(2)
                raw = buffer_ops.buffer_load(rsrc, dw, vec_width=vec_dwords, dtype=T.i32, soffset_bytes=soff)
                if vec_dwords == VEC_WIDTH:
                    return raw.bitcast(vec_type_e)
                return vector.bitcast(vec_type_e, raw)

            def _store_vec(data, rsrc, col_byte_off, soff=None):
                dw = col_byte_off >> fx.Int32(2)
                buffer_ops.buffer_store(data, rsrc, dw, soffset_bytes=soff)

            c_zero_f = arith.constant(0.0, type=compute_type)
            thread_sumsq = c_zero_f
            thread_dummy = c_zero_f
            cache_as_elem = (dtype_str != "f32")
            in_local = []

            # Pass 1: load + cache + sumsq
            for tile_i in range_constexpr(num_tiles):
                col_bytes = ArithValue(thr_col_bytes) + (tile_i * tile_cols * elem_bytes)
                vec_e = _load_vec(in_rsrc, col_bytes, soff=row_soffset)

                if cache_as_elem:
                    in_local.append(vec_e)
                    x = vec_e.extf(vec_type_c)
                else:
                    x = vec_e
                    in_local.append(x)

                x_av = ArithValue(x)
                x2 = x_av * x_av
                red2 = vector.reduction(compute_type, vector.CombiningKind.ADD, x2, fastmath=fm_fast)
                thread_sumsq = ArithValue(thread_sumsq) + red2

            _, sum_sq = block_reduce_add2(thread_dummy, thread_sumsq)
            mean_sq = ArithValue(sum_sq) / n_float
            ms_eps = ArithValue(mean_sq) + eps_c
            rrms = ms_eps.rsqrt(fastmath=fm_fast)
            rrms_splat = vector.broadcast(vec_type_c, rrms)
            rrms_splat_av = ArithValue(rrms_splat)

            # Pass 2: normalize + gamma + store (reuse cached input)
            for tile_i in range_constexpr(num_tiles):
                col_bytes = ArithValue(thr_col_bytes) + (tile_i * tile_cols * elem_bytes)

                g_e = _load_vec(gamma_rsrc, col_bytes)
                g = g_e if dtype_str == "f32" else g_e.extf(vec_type_c)

                x = in_local[tile_i]
                if cache_as_elem:
                    x = x.extf(vec_type_c)

                x_av = ArithValue(x)
                g_av = ArithValue(g)
                y = (x_av * rrms_splat_av) * g_av
                y_val = y

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
                        packed = even | odd_sh
                        out_e = vector.bitcast(vec_bf16_ty, packed)
                elif dtype_str == "f32":
                    out_e = y_val
                else:
                    out_e = y_val.truncf(vec_type_e)

                out_col = ArithValue(thr_col_bytes) + (tile_i * tile_cols * elem_bytes)
                i32_vec_ty = T.vec(vec_dwords, T.i32)
                out_vec = vector.bitcast(i32_vec_ty, out_e) if vec_dwords != VEC_WIDTH else out_e.bitcast(i32_vec_ty)
                _store_vec(out_vec, out_rsrc, out_col, soff=row_soffset)

        else:
            # ==============================================================
            # Generic path: scalar 2-pass for arbitrary N
            # ==============================================================
            from flydsl.expr.arith import ArithValue

            row_in = fx.slice(Input, (bid, None))
            row_out = fx.slice(Output, (bid, None))

            copy_atom_s = fx.make_copy_atom(fx.UniversalCopy(elem_bits), elem_bits)
            scalar_reg_ty = fx.MemRefType.get(elem_type, fx.LayoutType.get(1, 1), fx.AddressSpace.Register)
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

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                c_N_i32 = Int32(N)
                is_valid = idx < c_N_i32
                c0_i = Int32(0)
                idx_safe = is_valid.select(idx, c0_i)
                x_e = _load_scalar(row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.extf(compute_type)
                x_av = ArithValue(x)
                x2 = x_av * x_av
                x2_safe = is_valid.select(x2, c_zero_f)
                thread_sumsq = ArithValue(thread_sumsq) + x2_safe

            sum_sq = block_reduce_add(thread_sumsq)
            mean_sq = ArithValue(sum_sq) / n_float
            ms_eps = ArithValue(mean_sq) + eps_c
            rrms = ms_eps.rsqrt(fastmath=fm_fast)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                c_N_i32 = Int32(N)
                if arith.cmpi(arith.CmpIPredicate.ult, idx, c_N_i32):
                    x_e = _load_scalar(row_div, idx)
                    g_e = _load_scalar(gamma_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.extf(compute_type)
                    g = g_e if dtype_str == "f32" else g_e.extf(compute_type)
                    norm = ArithValue(x) * ArithValue(rrms)
                    y = norm * ArithValue(g)
                    if dtype_str == "f32":
                        y_e = y
                    elif dtype_str == "bf16":
                        y_e = y.truncf(elem_type)
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