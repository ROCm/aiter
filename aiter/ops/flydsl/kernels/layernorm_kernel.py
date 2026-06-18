# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""LayerNorm kernel builder using the @flyc.kernel API.

LayerNorm(x) = (x - mean) / sqrt(var + eps) * gamma + beta

Two paths:
  - Fast path (N == BLOCK_THREADS * VEC_WIDTH * 4): vectorised tiled copy,
    register caching, pipelined gamma/beta loads.
  - Generic path (arbitrary N): scalar 2-pass implementation.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.vector import ReductionOp, full
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from .kernels_common import dtype_to_elem_type, get_warp_size

KERNEL_NAME = "layernorm"

DEFAULT_EPS = 1e-5

BLOCK_THREADS = 256
WARP_SIZE = get_warp_size()
VEC_WIDTH = 8


# ── Shared-memory allocation for block reductions ─────────────────────
def _make_reduction_storage(red_slots: int):
    @fx.struct
    class SharedStorage:
        s_sum: fx.Array[fx.Float32, red_slots, 16]
        s_sumsq: fx.Array[fx.Float32, red_slots, 16]

    return SharedStorage


def _load_scalar(copy_atom, elem_dtype, divided_tensor, index):
    view = fx.slice(divided_tensor, (None, index))
    r = fx.make_rmem_tensor(1, elem_dtype)
    fx.copy_atom_call(copy_atom, view, r)
    return fx.memref_load_vec(r)[0]


def _store_scalar(copy_atom, elem_dtype, store_dtype, divided_tensor, index, val):
    r = fx.make_rmem_tensor(1, elem_dtype)
    ts = full(1, store_dtype(val), store_dtype)
    fx.memref_store_vec(ts, r)
    view = fx.slice(divided_tensor, (None, index))
    fx.copy_atom_call(copy_atom, r, view)


def _load_vec(copy_atom, vec_width, elem_dtype, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.copy_atom_call(copy_atom, fx.slice(div_tensor, (None, idx)), r)
    return fx.memref_load_vec(r)


def _store_vec(copy_atom, vec_width, elem_dtype, val, div_tensor, idx):
    r = fx.make_rmem_tensor(vec_width, elem_dtype)
    fx.memref_store_vec(val, r)
    fx.copy_atom_call(copy_atom, r, fx.slice(div_tensor, (None, idx)))


def _to_elem_scalar(dtype_str: str, elem_dtype, y):
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


def _to_elem_vec(dtype_str: str, elem_dtype, use_hw_cvt_bf16: bool, y):
    if const_expr(dtype_str == "bf16"):
        if const_expr(use_hw_cvt_bf16):
            return y.to(elem_dtype)
        u = y.bitcast(fx.Uint32)
        upper = u >> 16
        lsb = upper & 1
        bias = lsb + 0x7FFF
        u_round = y.bitcast(fx.Uint32) + bias
        bf16_bits = u_round >> 16
        even = bf16_bits.shuffle(bf16_bits, [0, 2, 4, 6])
        odd = bf16_bits.shuffle(bf16_bits, [1, 3, 5, 7])
        odd_sh = odd << 16
        packed = even | odd_sh
        return packed.bitcast(elem_dtype)
    if const_expr(dtype_str == "f32"):
        return y
    return y.to(elem_dtype)


def _store_yscale(scale_copy_atom, yscale_div, index, val):
    r = fx.make_rmem_tensor(1, fx.Float32)
    ts = full(1, fx.Float32(val), fx.Float32)
    fx.memref_store_vec(ts, r)
    fx.copy_atom_call(scale_copy_atom, r, fx.slice(yscale_div, (None, index)))


def _quant_dtype_to_elem_type(dtype_str: str):
    if dtype_str in ("i8", "int8"):
        return fx.Int8
    raise ValueError(f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')")


def _quant_dtype_max(dtype_str: str) -> float:
    if dtype_str in ("i8", "int8"):
        return 127.0
    raise ValueError(f"unsupported quant dtype: {dtype_str!r} (expected 'i8' or 'int8')")


def build_layernorm_module(M: int, N: int, dtype_str: str, eps: float = DEFAULT_EPS):
    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    # ── GPU kernel ────────────────────────────────────────────────────────
    @flyc.kernel
    def layernorm_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_sum = lds.s_sum.view(fx.make_layout(RED_SLOTS, 1))
        s_sumsq = lds.s_sumsq.view(fx.make_layout(RED_SLOTS, 1))

        # ── helpers: wave / block reduction ───────────────────────────────
        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE

            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_sum, wave)
                fx.memref_store(w1, s_sumsq, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_sum, lane_safe)
                v1 = fx.memref_load(s_sumsq, lane_safe)
                ww0 = in_range.select(v0, 0.0)
                ww1 = in_range.select(v1, 0.0)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_sum, 0)
                    fx.memref_store(ww1, s_sumsq, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0), fx.memref_load(s_sumsq, 0)

        def compute_mean_rstd(sum_val, sumsq_val):
            inv_n = 1.0 / float(N)
            mean = sum_val * inv_n
            mean_sq = sumsq_val * inv_n
            mean2 = mean * mean
            var = mean_sq - mean2
            is_neg = var < 0.0
            var = is_neg.select(0.0, var)
            var_eps = var + eps_c
            rstd = fmath.rsqrt(var_eps, fastmath=fm_fast)
            return mean, rstd

        # ==================================================================
        # Fast path: N == BLOCK_THREADS * VEC_WIDTH * 4
        # Uses buffer_load / buffer_store for high-bandwidth vectorised
        # memory access (same approach as preshuffle_gemm).
        # ==================================================================
        if const_expr(N == (BLOCK_THREADS * VEC_WIDTH * 4) and elem_bits <= 16):
            num_tiles_py = 4
            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            in_local = []

            # ── Layout API: buffer-backed tensors + tiled access ─────
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            # ── Pass 1: load input, accumulate sum / sumsq ───────────────
            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                vec = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx)
                in_local.append(vec)
                x = vec.to(fx.Float32)

                x2 = x * x
                red = x.reduce(ReductionOp.ADD, fastmath=fm_fast)
                red2 = x2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sum = thread_sum + red
                thread_sumsq = thread_sumsq + red2

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            g_cur = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, tid).to(fx.Float32)
            b_cur = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, tid).to(fx.Float32)

            # ── Pass 2: normalize + affine + store ───────────────────────
            for tile_i in range_constexpr(num_tiles_py):
                g_next = g_cur
                b_next = b_cur
                if const_expr(tile_i + 1 < num_tiles_py):
                    next_idx = tid + (tile_i + 1) * BLOCK_THREADS
                    g_next = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, next_idx).to(fx.Float32)
                    b_next = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, next_idx).to(fx.Float32)
                else:
                    g_next = g_cur
                    b_next = b_cur

                x = in_local[tile_i].to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g_cur + b_cur

                out_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                out_idx = tid + tile_i * BLOCK_THREADS
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, out_e, out_div, out_idx)

                g_cur = g_next
                b_cur = b_next

        else:
            # ==============================================================
            # Generic path: 2-pass scalar implementation for arbitrary N
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )

            row_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))

            # ── Pass 1: sum + sumsq ──────────────────────────────────────
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                x_safe = is_valid.select(x, c_zero_f)
                x2_safe = is_valid.select(x2, c_zero_f)
                thread_sum = thread_sum + x_safe
                thread_sumsq = thread_sumsq + x2_safe

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            # ── Pass 2: normalize + affine + store ───────────────────────
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, row_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    diff = x - mean
                    norm = diff * rstd
                    scaled = norm * g
                    y = scaled + b
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, out_div, idx, y_e)

    # ── JIT host launcher ─────────────────────────────────────────────────
    @flyc.jit
    def launch_layernorm(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = layernorm_kernel(Input, Gamma, Beta, Output)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_layernorm


def build_fused_add_layernorm_module(M: int, N: int, dtype_str: str, eps: float = DEFAULT_EPS):
    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def fused_add_layernorm_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        fm_fast = arith.FastMathFlags.fast
        eps_c = eps

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_sum = lds.s_sum.view(fx.make_layout(RED_SLOTS, 1))
        s_sumsq = lds.s_sumsq.view(fx.make_layout(RED_SLOTS, 1))

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_sum, wave)
                fx.memref_store(w1, s_sumsq, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_sum, lane_safe)
                v1 = fx.memref_load(s_sumsq, lane_safe)
                ww0 = in_range.select(v0, 0.0)
                ww1 = in_range.select(v1, 0.0)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)

                if lane == 0:
                    fx.memref_store(ww0, s_sum, 0)
                    fx.memref_store(ww1, s_sumsq, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0), fx.memref_load(s_sumsq, 0)

        def compute_mean_rstd(sum_val, sumsq_val):
            inv_n = 1.0 / float(N)
            mean = sum_val * inv_n
            mean_sq = sumsq_val * inv_n
            var = mean_sq - mean * mean
            var = (var < 0.0).select(0.0, var)
            return mean, fmath.rsqrt(var + eps_c, fastmath=fm_fast)

        # ==================================================================
        # Fast path: N == BLOCK_THREADS * VEC_WIDTH * 4
        # ==================================================================
        if const_expr(N == (BLOCK_THREADS * VEC_WIDTH * 4) and elem_bits <= 16):
            num_tiles_py = 4
            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            added_local = []

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(VEC_WIDTH, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(VEC_WIDTH, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            # Pass 1: add residual, cache/store it, and accumulate sum/sumsq.
            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx).to(fx.Float32)
                residual = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                added_local.append(added_e)
                added = added_e.to(fx.Float32)
                added2 = added * added
                thread_sum = thread_sum + added.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + added2.reduce(ReductionOp.ADD, fastmath=fm_fast)
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            # Pass 2: normalize + affine + store, reusing cached added values.
            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                added = added_local[tile_i].to(fx.Float32)
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx).to(fx.Float32)
                y = (added - mean) * rstd
                y = y * g + b
                y_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, y)
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, y_e, out_div, idx)

        else:
            # ==============================================================
            # Generic path: scalar 2-pass implementation for arbitrary N
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )

            in_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(1, 1))

            c_zero_f = fx.Float32(0.0)
            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, in_div, idx_safe)
                r_e = _load_scalar(copy_atom_s, elem_dtype, residual_in_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = r_e if dtype_str == "f32" else r_e.to(fx.Float32)
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                added_safe = is_valid.select(added, c_zero_f)
                thread_sum = thread_sum + added_safe
                thread_sumsq = thread_sumsq + is_valid.select(added * added, c_zero_f)
                if idx < N:
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, residual_out_div, idx, added_e)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean, rstd = compute_mean_rstd(sum_val, sumsq_val)

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    added_e = _load_scalar(copy_atom_s, elem_dtype, residual_out_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx)
                    added = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = (added - mean) * rstd
                    y = y * g + b
                    y_e = _to_elem_scalar(dtype_str, elem_dtype, y)
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, out_div, idx, y_e)

    @flyc.jit
    def launch_fused_add_layernorm(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
        m_in: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        launcher = fused_add_layernorm_kernel(Input, ResidualIn, Gamma, Beta, Output, ResidualOut)
        launcher.launch(
            grid=(m_in, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_fused_add_layernorm


def _build_layernorm_quant_module(
    M: int,
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
    eps: float = DEFAULT_EPS,
):
    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def layernorm_quant_kernel(
        Input: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        XScale: fx.Tensor,
        YScale: fx.Tensor,
        Output: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        quant_dtype = _quant_dtype_to_elem_type(quant_dtype_str)

        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_dtype_max = fx.Float32(quant_dtype_max)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_sum = lds.s_sum.view(fx.make_layout(RED_SLOTS, 1))
        s_sumsq = lds.s_sumsq.view(fx.make_layout(RED_SLOTS, 1))

        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)
        yscale_div = fx.logical_divide(YScale_buf, fx.make_layout(1, 1))
        scale_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def wave_reduce_max(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.maximumf(peer)
            return w

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_sum, wave)
                fx.memref_store(w1, s_sumsq, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_sum, lane_safe)
                v1 = fx.memref_load(s_sumsq, lane_safe)
                ww0 = in_range.select(v0, c_zero_f)
                ww1 = in_range.select(v1, c_zero_f)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)
                if lane == 0:
                    fx.memref_store(ww0, s_sum, 0)
                    fx.memref_store(ww1, s_sumsq, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0), fx.memref_load(s_sumsq, 0)

        def block_reduce_max(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_max(val)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w = wave_reduce_max(val)
            if lane == 0:
                fx.memref_store(w, s_sum, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_sum, lane_safe)
                ww = in_range.select(v, c_neg_inf)
                ww = wave_reduce_max(ww)
                if lane == 0:
                    fx.memref_store(ww, s_sum, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0)

        # ==================================================================
        # Fast path: N == BLOCK_THREADS * VEC_WIDTH * 4
        # ==================================================================
        if const_expr(N == (BLOCK_THREADS * VEC_WIDTH * 4) and elem_bits <= 16):
            num_tiles_py = 4
            quant_half_width = VEC_WIDTH // 2
            abs_mask = full(VEC_WIDTH, fx.Uint32(0x7FFFFFFF), fx.Uint32)

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(VEC_WIDTH, 1))
            out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_half_width, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 8)
            if const_expr(is_smooth):
                copy_atom_xs = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            norm_input_local = []

            # Pass 1: prepare normalization input and accumulate sum/sumsq.
            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x_e = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx)
                norm_input_local.append(x_e)
                x_norm = x_e.to(fx.Float32)
                x2 = x_norm * x_norm
                thread_sum = thread_sum + x_norm.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + x2.reduce(ReductionOp.ADD, fastmath=fm_fast)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = (var + eps_c).rsqrt(fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            # Pass 2: affine (+ optional smooth scale), cache y, accumulate row max.
            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x = norm_input_local[tile_i].to(fx.Float32)
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx).to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
                if const_expr(is_smooth):
                    s = _load_vec(copy_atom_xs, VEC_WIDTH, elem_dtype, xscale_div, idx).to(fx.Float32)
                    y = y * s
                y_local.append(y)
                y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
                tile_max = y_abs.reduce(ReductionOp.MAX)
                thread_row_max = thread_row_max.maximumf(tile_max)

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_yscale(scale_copy_atom, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            # Pass 3: quantize + store using per-row scale.
            for tile_i in range_constexpr(num_tiles_py):
                q = y_local[tile_i] * inv_scale
                q_i8 = q.to(quant_dtype)
                q_lo = q_i8.shuffle(q_i8, [0, 1, 2, 3])
                q_hi = q_i8.shuffle(q_i8, [4, 5, 6, 7])
                out_idx = tid * 2 + tile_i * BLOCK_THREADS * 2
                _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_lo, out_div_q, out_idx)
                _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_hi, out_div_q, out_idx + 1)

        else:
            # ==============================================================
            # Generic path: scalar 3-pass implementation for arbitrary N
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )
            copy_atom_qs = fx.make_copy_atom(fx.rocdl.BufferCopy(8), 8)

            in_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(1, 1))

            def _abs_scalar(val):
                is_neg = val < c_zero_f
                neg_val = c_zero_f - val
                return is_neg.select(neg_val, val)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, in_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                x2 = x * x
                thread_sum = thread_sum + is_valid.select(x, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = (var + eps_c).rsqrt(fastmath=fm_fast)

            thread_row_max = c_zero_f
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, in_div, idx_safe)
                g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx_safe)
                b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
                if const_expr(is_smooth):
                    s_e = _load_scalar(copy_atom_s, elem_dtype, xscale_div, idx_safe)
                    s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                    y = y * s
                y_abs = _abs_scalar(y)
                thread_row_max = thread_row_max.maximumf(is_valid.select(y_abs, c_zero_f))

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_yscale(scale_copy_atom, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, in_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = (x - mean) * rstd
                    y = y * g + b
                    if const_expr(is_smooth):
                        s_e = _load_scalar(copy_atom_s, elem_dtype, xscale_div, idx)
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q = y * inv_scale
                    q_i8 = q.to(quant_dtype)
                    _store_scalar(copy_atom_qs, quant_dtype, quant_dtype, out_div, idx, q_i8)

    if is_smooth:

        @flyc.jit
        def launch_layernorm_smoothquant(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = layernorm_quant_kernel(Input, Gamma, Beta, XScale, YScale, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_layernorm_smoothquant

    else:

        @flyc.jit
        def launch_layernorm_dynamicquant(
            Input: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            Output: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = layernorm_quant_kernel(Input, Gamma, Beta, Gamma, YScale, Output)
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    return launch_layernorm_dynamicquant


def _build_fused_add_layernorm_quant_module(
    M: int,
    N: int,
    dtype_str: str,
    *,
    is_smooth: bool,
    quant_dtype_str: str = "i8",
    eps: float = DEFAULT_EPS,
):
    arch = get_hip_arch()
    USE_HW_CVT_PK_BF16_F32 = (arch == "gfx950") or str(arch).startswith("gfx95")

    RED_SLOTS = max(1, (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE)
    elem_bits = 32 if dtype_str == "f32" else 16
    quant_dtype_max = _quant_dtype_max(quant_dtype_str)

    SharedStorage = _make_reduction_storage(RED_SLOTS)

    @flyc.kernel
    def fused_add_layernorm_quant_kernel(
        Input: fx.Tensor,
        ResidualIn: fx.Tensor,
        Gamma: fx.Tensor,
        Beta: fx.Tensor,
        XScale: fx.Tensor,
        YScale: fx.Tensor,
        Output: fx.Tensor,
        ResidualOut: fx.Tensor,
    ):
        bid = fx.block_idx.x
        tid = fx.thread_idx.x

        elem_dtype = dtype_to_elem_type(dtype_str)
        quant_dtype = _quant_dtype_to_elem_type(quant_dtype_str)

        fm_fast = arith.FastMathFlags.fast
        eps_c = eps
        n_float = float(N)
        c_zero_f = fx.Float32(0.0)
        c_one_f = fx.Float32(1.0)
        c_neg_inf = fx.Float32(float("-inf"))
        c_dtype_max = fx.Float32(quant_dtype_max)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_sum = lds.s_sum.view(fx.make_layout(RED_SLOTS, 1))
        s_sumsq = lds.s_sumsq.view(fx.make_layout(RED_SLOTS, 1))

        YScale_buf = fx.rocdl.make_buffer_tensor(YScale)
        yscale_div = fx.logical_divide(YScale_buf, fx.make_layout(1, 1))
        scale_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 32)

        def wave_reduce_add(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.addf(peer, fastmath=fm_fast)
            return w

        def wave_reduce_max(x):
            w = x
            for _sh_exp in range_constexpr(int(math.log2(WARP_SIZE))):
                off = WARP_SIZE // (2 << _sh_exp)
                peer = w.shuffle_xor(off, WARP_SIZE)
                w = w.maximumf(peer)
            return w

        def block_reduce_add2(val0, val1):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_add(val0), wave_reduce_add(val1)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w0 = wave_reduce_add(val0)
            w1 = wave_reduce_add(val1)

            if lane == 0:
                fx.memref_store(w0, s_sum, wave)
                fx.memref_store(w1, s_sumsq, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v0 = fx.memref_load(s_sum, lane_safe)
                v1 = fx.memref_load(s_sumsq, lane_safe)
                ww0 = in_range.select(v0, c_zero_f)
                ww1 = in_range.select(v1, c_zero_f)
                ww0 = wave_reduce_add(ww0)
                ww1 = wave_reduce_add(ww1)
                if lane == 0:
                    fx.memref_store(ww0, s_sum, 0)
                    fx.memref_store(ww1, s_sumsq, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0), fx.memref_load(s_sumsq, 0)

        def block_reduce_max(val):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_max(val)

            lane = tid % WARP_SIZE
            wave = tid // WARP_SIZE
            w = wave_reduce_max(val)
            if lane == 0:
                fx.memref_store(w, s_sum, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                v = fx.memref_load(s_sum, lane_safe)
                ww = in_range.select(v, c_neg_inf)
                ww = wave_reduce_max(ww)
                if lane == 0:
                    fx.memref_store(ww, s_sum, 0)
            gpu.barrier()

            return fx.memref_load(s_sum, 0)

        # ==================================================================
        # Fast path: N == BLOCK_THREADS * VEC_WIDTH * 4
        # ==================================================================
        if const_expr(N == (BLOCK_THREADS * VEC_WIDTH * 4) and elem_bits <= 16):
            num_tiles_py = 4
            quant_half_width = VEC_WIDTH // 2
            abs_mask = full(VEC_WIDTH, fx.Uint32(0x7FFFFFFF), fx.Uint32)

            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(VEC_WIDTH, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(VEC_WIDTH, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(VEC_WIDTH, 1))
            out_div_q = fx.logical_divide(row_out, fx.make_layout(quant_half_width, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(VEC_WIDTH, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(VEC_WIDTH, 1))

            copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)
            copy_atom_q = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), 8)
            if const_expr(is_smooth):
                copy_atom_xs = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), elem_bits)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f
            norm_input_local = []

            # Pass 1: add residual, store residual_out, and accumulate sum/sumsq.
            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, in_div, idx).to(fx.Float32)
                residual = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, residual_in_div, idx).to(fx.Float32)
                added_e = _to_elem_vec(dtype_str, elem_dtype, USE_HW_CVT_PK_BF16_F32, x + residual)
                norm_input_local.append(added_e)
                x_norm = added_e.to(fx.Float32)
                _store_vec(copy_atom, VEC_WIDTH, elem_dtype, added_e, residual_out_div, idx)
                x2 = x_norm * x_norm
                thread_sum = thread_sum + x_norm.reduce(ReductionOp.ADD, fastmath=fm_fast)
                thread_sumsq = thread_sumsq + x2.reduce(ReductionOp.ADD, fastmath=fm_fast)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = (var + eps_c).rsqrt(fastmath=fm_fast)

            thread_row_max = c_zero_f
            y_local = []

            # Pass 2: affine (+ optional smooth scale), cache y, accumulate row max.
            for tile_i in range_constexpr(num_tiles_py):
                idx = tid + tile_i * BLOCK_THREADS
                x = norm_input_local[tile_i].to(fx.Float32)
                g = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, gamma_div, idx).to(fx.Float32)
                b = _load_vec(copy_atom, VEC_WIDTH, elem_dtype, beta_div, idx).to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
                if const_expr(is_smooth):
                    s = _load_vec(copy_atom_xs, VEC_WIDTH, elem_dtype, xscale_div, idx).to(fx.Float32)
                    y = y * s
                y_local.append(y)
                y_abs = (y.bitcast(fx.Uint32) & abs_mask).bitcast(fx.Float32)
                tile_max = y_abs.reduce(ReductionOp.MAX)
                thread_row_max = thread_row_max.maximumf(tile_max)

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_yscale(scale_copy_atom, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            # Pass 3: quantize + store using per-row scale.
            for tile_i in range_constexpr(num_tiles_py):
                q = y_local[tile_i] * inv_scale
                q_i8 = q.to(quant_dtype)
                q_lo = q_i8.shuffle(q_i8, [0, 1, 2, 3])
                q_hi = q_i8.shuffle(q_i8, [4, 5, 6, 7])
                out_idx = tid * 2 + tile_i * BLOCK_THREADS * 2
                _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_lo, out_div_q, out_idx)
                _store_vec(copy_atom_q, quant_half_width, quant_dtype, q_hi, out_div_q, out_idx + 1)

        else:
            # ==============================================================
            # Generic path: scalar 3-pass implementation for arbitrary N
            # ==============================================================
            Input_buf = fx.rocdl.make_buffer_tensor(Input)
            ResidualIn_buf = fx.rocdl.make_buffer_tensor(ResidualIn)
            Gamma_buf = fx.rocdl.make_buffer_tensor(Gamma)
            Beta_buf = fx.rocdl.make_buffer_tensor(Beta)
            Output_buf = fx.rocdl.make_buffer_tensor(Output)
            ResidualOut_buf = fx.rocdl.make_buffer_tensor(ResidualOut)
            if const_expr(is_smooth):
                XScale_buf = fx.rocdl.make_buffer_tensor(XScale)

            row_in = fx.slice(Input_buf, (bid, None))
            row_residual_in = fx.slice(ResidualIn_buf, (bid, None))
            row_out = fx.slice(Output_buf, (bid, None))
            row_residual_out = fx.slice(ResidualOut_buf, (bid, None))

            copy_atom_s = fx.make_copy_atom(
                fx.rocdl.BufferCopy16b() if elem_bits <= 16 else fx.rocdl.BufferCopy32b(),
                elem_bits,
            )
            copy_atom_qs = fx.make_copy_atom(fx.rocdl.BufferCopy(8), 8)

            in_div = fx.logical_divide(row_in, fx.make_layout(1, 1))
            residual_in_div = fx.logical_divide(row_residual_in, fx.make_layout(1, 1))
            gamma_div = fx.logical_divide(Gamma_buf, fx.make_layout(1, 1))
            beta_div = fx.logical_divide(Beta_buf, fx.make_layout(1, 1))
            out_div = fx.logical_divide(row_out, fx.make_layout(1, 1))
            residual_out_div = fx.logical_divide(row_residual_out, fx.make_layout(1, 1))
            if const_expr(is_smooth):
                xscale_div = fx.logical_divide(XScale_buf, fx.make_layout(1, 1))

            def _abs_scalar(val):
                is_neg = val < c_zero_f
                neg_val = c_zero_f - val
                return is_neg.select(neg_val, val)

            thread_sum = c_zero_f
            thread_sumsq = c_zero_f

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, in_div, idx_safe)
                r_e = _load_scalar(copy_atom_s, elem_dtype, residual_in_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                residual = r_e if dtype_str == "f32" else r_e.to(fx.Float32)
                added_e = _to_elem_scalar(dtype_str, elem_dtype, x + residual)
                if idx < N:
                    _store_scalar(copy_atom_s, elem_dtype, elem_dtype, residual_out_div, idx, added_e)
                x = added_e if dtype_str == "f32" else added_e.to(fx.Float32)
                x2 = x * x
                thread_sum = thread_sum + is_valid.select(x, c_zero_f)
                thread_sumsq = thread_sumsq + is_valid.select(x2, c_zero_f)

            sum_val, sumsq_val = block_reduce_add2(thread_sum, thread_sumsq)
            mean = sum_val / n_float
            var = sumsq_val / n_float - mean * mean
            var = (var < c_zero_f).select(c_zero_f, var)
            rstd = (var + eps_c).rsqrt(fastmath=fm_fast)

            thread_row_max = c_zero_f
            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                is_valid = idx < N
                idx_safe = is_valid.select(idx, 0)
                x_e = _load_scalar(copy_atom_s, elem_dtype, residual_out_div, idx_safe)
                g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx_safe)
                b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx_safe)
                x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                y = (x - mean) * rstd
                y = y * g + b
                if const_expr(is_smooth):
                    s_e = _load_scalar(copy_atom_s, elem_dtype, xscale_div, idx_safe)
                    s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                    y = y * s
                y_abs = _abs_scalar(y)
                thread_row_max = thread_row_max.maximumf(is_valid.select(y_abs, c_zero_f))

            row_max = block_reduce_max(thread_row_max)
            scale = row_max / c_dtype_max
            final_scale = (scale == c_zero_f).select(c_one_f, scale)

            if tid == 0:
                _store_yscale(scale_copy_atom, yscale_div, bid, final_scale)

            inv_scale = c_one_f / final_scale

            for base_idx_int in range_constexpr(0, N, BLOCK_THREADS):
                idx = tid + base_idx_int
                if idx < N:
                    x_e = _load_scalar(copy_atom_s, elem_dtype, residual_out_div, idx)
                    g_e = _load_scalar(copy_atom_s, elem_dtype, gamma_div, idx)
                    b_e = _load_scalar(copy_atom_s, elem_dtype, beta_div, idx)
                    x = x_e if dtype_str == "f32" else x_e.to(fx.Float32)
                    g = g_e if dtype_str == "f32" else g_e.to(fx.Float32)
                    b = b_e if dtype_str == "f32" else b_e.to(fx.Float32)
                    y = (x - mean) * rstd
                    y = y * g + b
                    if const_expr(is_smooth):
                        s_e = _load_scalar(copy_atom_s, elem_dtype, xscale_div, idx)
                        s = s_e if dtype_str == "f32" else s_e.to(fx.Float32)
                        y = y * s
                    q = y * inv_scale
                    q_i8 = q.to(quant_dtype)
                    _store_scalar(copy_atom_qs, quant_dtype, quant_dtype, out_div, idx, q_i8)

    if is_smooth:

        @flyc.jit
        def launch_fused_add_layernorm_smoothquant(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            XScale: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = fused_add_layernorm_quant_kernel(
                Input, ResidualIn, Gamma, Beta, XScale, YScale, Output, ResidualOut
            )
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

        return launch_fused_add_layernorm_smoothquant

    else:

        @flyc.jit
        def launch_fused_add_layernorm_dynamicquant(
            Input: fx.Tensor,
            ResidualIn: fx.Tensor,
            Gamma: fx.Tensor,
            Beta: fx.Tensor,
            Output: fx.Tensor,
            ResidualOut: fx.Tensor,
            YScale: fx.Tensor,
            m_in: fx.Int32,
            stream: fx.Stream = fx.Stream(None),
        ):
            launcher = fused_add_layernorm_quant_kernel(
                Input, ResidualIn, Gamma, Beta, Gamma, YScale, Output, ResidualOut
            )
            launcher.launch(
                grid=(m_in, 1, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    return launch_fused_add_layernorm_dynamicquant


def build_layernorm_dynamicquant_module(
    M: int,
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = DEFAULT_EPS,
):
    return _build_layernorm_quant_module(
        M,
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


def build_layernorm_smoothquant_module(
    M: int,
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = DEFAULT_EPS,
):
    return _build_layernorm_quant_module(
        M,
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


def build_fused_add_layernorm_dynamicquant_module(
    M: int,
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = DEFAULT_EPS,
):
    return _build_fused_add_layernorm_quant_module(
        M,
        N,
        dtype_str,
        is_smooth=False,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )


def build_fused_add_layernorm_smoothquant_module(
    M: int,
    N: int,
    dtype_str: str,
    quant_dtype_str: str = "i8",
    eps: float = DEFAULT_EPS,
):
    return _build_fused_add_layernorm_quant_module(
        M,
        N,
        dtype_str,
        is_smooth=True,
        quant_dtype_str=quant_dtype_str,
        eps=eps,
    )
