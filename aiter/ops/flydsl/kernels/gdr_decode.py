# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
import numpy as np
from typing import Optional, Any, Callable, Dict, Literal, Optional, Tuple

import flydsl
import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl.expr.typing import T
from flydsl._mlir.dialects import gpu as mlir_gpu, vector as mlir_vector, math as mlir_math
from flydsl.expr import range_constexpr, arith, vector, gpu, rocdl
from flydsl._mlir import ir
from flydsl.runtime.device import get_rocm_arch
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.compiler.kernel_function import CompilationContext
from flydsl._mlir.dialects import llvm, fly, memref, scf
from flydsl.compiler.protocol import fly_values

from .tensor_shim import get_dtype_in_kernel, get_dtype_vec_size, get_dtype_str, GTensor, STensor, _to_raw

fm_fast = arith.FastMathFlags.fast


@functools.lru_cache(maxsize=1024)
def create_shuffle_gdr_decode_kernel(
    dtype: str,
    seq_length: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    use_qk_l2norm: bool,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    NUM_BLOCKS_PER_V_DIM: int = 1,
    NUM_WARPS: int = 4,
    WARP_THREADS_K: int = 8,
):
    SCALE_VALUE = float(1.0 / (float(head_k_dim) ** 0.5))
    WARP_THREADS_V = 64 // WARP_THREADS_K
    VEC_SIZE = get_dtype_vec_size(dtype)
    DTYPE_BYTES = 16 // VEC_SIZE
    VALUES_PER_THREAD_K = 4 # 16B

    WARP_SIZE = WARP_THREADS_V * WARP_THREADS_K
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE
    assert WARP_SIZE == 64

    WARP_TILE_K = WARP_THREADS_K * VALUES_PER_THREAD_K
    WARP_TILE_K_ITERS = head_k_dim // WARP_TILE_K
    assert WARP_TILE_K_ITERS >= 1
    assert head_k_dim % WARP_TILE_K == 0
    TILE_K = head_k_dim

    WARP_TILE_V = WARP_THREADS_V
    WARP_GROUP_TILE_V = NUM_WARPS * WARP_TILE_V
    TILE_V = head_v_dim // NUM_BLOCKS_PER_V_DIM
    WARP_TILE_V_ITERS = TILE_V // WARP_GROUP_TILE_V
    assert TILE_V >= 1 and head_v_dim % NUM_BLOCKS_PER_V_DIM == 0
    assert WARP_TILE_V_ITERS >= 1 and TILE_V % WARP_GROUP_TILE_V == 0

    WARP_THREADS_K_SHFL_OFFSETS = []
    offsets_ = WARP_THREADS_K // 2
    while offsets_ >= 1:
        WARP_THREADS_K_SHFL_OFFSETS.append(int(offsets_))
        offsets_ /= 2
    WARP_THREADS_K_SHFL_OFFSETS = WARP_THREADS_K_SHFL_OFFSETS[::-1]
    
    WARP_SIZE_SHFL_OFFSETS = []
    offsets_ = WARP_SIZE // 2
    while offsets_ >= 1:
        WARP_SIZE_SHFL_OFFSETS.append(int(offsets_))
        offsets_ /= 2
    
    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(None, arch=GPU_ARCH, global_sym_name="smem")
    smem_sr_offset = allocator._align(allocator.ptr, 16)
    allocator.ptr = smem_sr_offset + 2 * NUM_WARPS * 4

    KERNEL_NAME = f"gdr_decode_{dtype}_kh{num_k_heads}x{head_k_dim}_vh{num_v_heads}x{head_v_dim}_q{seq_length}"
    KERNEL_NAME += f"_{NUM_WARPS}w{WARP_THREADS_V}x{WARP_THREADS_K}"
    KERNEL_NAME += f"_vs{NUM_BLOCKS_PER_V_DIM}"

    @flyc.kernel
    def gdr_decode_kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        dt_bias: fx.Tensor,
        A_log: fx.Tensor,
        indices: fx.Tensor,
        state: fx.Tensor,
        out: fx.Tensor,
        batch_size: fx.Int32,
    ):
        scale = arith.constant(SCALE_VALUE, type=T.f32)
        softplus_beta_ = arith.constant(softplus_beta, type=T.f32)
        softplus_threshold_ = arith.constant(softplus_threshold, type=T.f32)

        dtype_ = get_dtype_in_kernel(dtype)
        i32_0 = arith.constant(0, type=T.i32)
        f32_0 = arith.constant(0.0, type=T.f32)
        f32_1 = arith.constant(1.0, type=T.f32)
        width_i32 = arith.constant(WARP_SIZE, type=T.i32)
        acc_vec_t = T.vec(VALUES_PER_THREAD_K, T.f32)

        tidx = fx.thread_idx.x
        bidx = fx.block_idx.x
        w_tid = tidx % WARP_SIZE
        wid = tidx // WARP_SIZE

        b_hv_i = bidx // NUM_BLOCKS_PER_V_DIM
        tile_v_start = bidx % NUM_BLOCKS_PER_V_DIM * TILE_V

        b_i = b_hv_i // num_v_heads
        hv_i = b_hv_i % num_v_heads
        hk_i = hv_i // (num_v_heads // num_k_heads)

        warp_k_vec_start = w_tid % WARP_THREADS_K * VALUES_PER_THREAD_K
        global_v_start = tile_v_start + wid * WARP_TILE_V + w_tid // WARP_THREADS_K

        indices_tensor = GTensor(indices, dtype=T.i32, shape=(-1,))
        pool_idx = fx.Int32(indices_tensor[b_i])

        q_tensor = GTensor(query, dtype=dtype_, shape=(-1, seq_length, num_k_heads, head_k_dim))
        k_tensor = GTensor(key, dtype=dtype_, shape=(-1, seq_length, num_k_heads, head_k_dim))
        v_tensor = GTensor(value, dtype=dtype_, shape=(-1, seq_length, num_v_heads, head_v_dim))
        a_tensor = GTensor(a, dtype=dtype_, shape=(-1, seq_length, num_v_heads))
        b_tensor = GTensor(b, dtype=dtype_, shape=(-1, seq_length, num_v_heads))
        dt_bias_tensor = GTensor(dt_bias, dtype=dtype_, shape=(num_v_heads,))
        A_log_tensor = GTensor(A_log, dtype=T.f32, shape=(num_v_heads,))
        state_tensor = GTensor(state, dtype=T.f32, shape=(-1, num_v_heads, head_v_dim, head_k_dim))
        out_tensor = GTensor(out, dtype=dtype_, shape=(-1, seq_length, num_v_heads, head_v_dim))

        base_ptr = allocator.get_base()
        smem_sr_ptr = SmemPtr(base_ptr, smem_sr_offset, T.f32, shape=(2 * NUM_WARPS,))
        sr_tensor = STensor(smem_sr_ptr, dtype=T.f32, shape=(-1,))

        def fast_exp(x, use_exp2=True):
            if use_exp2:
                log2e = 1.4426950408889634
                out = rocdl.exp2(T.f32, x * log2e)
                return out
            return mlir_math.exp(x, fastmath=fm_fast)
        
        def fast_log1p(x):
            return mlir_math.log1p(x, fastmath=fm_fast)

        cond_valid = arith.cmpi(arith.CmpIPredicate.sge, pool_idx, fx.Int32(0))
        cond_valid_if = scf.IfOp(cond_valid, results_=[], has_else=False)
        with ir.InsertionPoint(cond_valid_if.then_block):

            r_A_log = A_log_tensor[hv_i]
            r_dt_bias = dt_bias_tensor[hv_i].extf(T.f32)

            state_vecs = [0] * (WARP_TILE_V_ITERS * WARP_TILE_K_ITERS)
            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    state_vecs[vi * WARP_TILE_K_ITERS + ki] = state_tensor.vec_load((pool_idx, hv_i, global_v_i, warp_k_vec_i), VALUES_PER_THREAD_K)
          
            for sq_i in range_constexpr(seq_length):

                r_a = a_tensor[b_i, sq_i, hv_i].extf(T.f32)
                r_b = b_tensor[b_i, sq_i, hv_i].extf(T.f32)
                x = r_a + r_dt_bias
                beta_x = softplus_beta_ * x
                
                cond_sp = arith.cmpf(arith.CmpFPredicate.OLE, beta_x, fx.Float32(softplus_threshold_))
                cond_sp_if = scf.IfOp(cond_sp, results_=[T.f32], has_else=True)
                with ir.InsertionPoint(cond_sp_if.then_block):
                    softplus_x_ = (f32_1 / softplus_beta_) * fast_log1p(fast_exp(beta_x))
                    scf.YieldOp([softplus_x_])
                with ir.InsertionPoint(cond_sp_if.else_block):
                    softplus_x_ = x
                    scf.YieldOp([softplus_x_])
                softplus_x = cond_sp_if.results[0]

                r_g_value = - fast_exp(r_A_log) * softplus_x
                r_beta = f32_1 / (f32_1 + fast_exp(-r_b))
                r_g = fast_exp(r_g_value)

                r_g_vec = vector.BroadcastOp(acc_vec_t, r_g).vector

                sq_vecs = [0] * WARP_TILE_K_ITERS
                sk_vecs = [0] * WARP_TILE_K_ITERS
                
                scale_vec = vector.BroadcastOp(acc_vec_t, scale).vector

                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    q_vec = q_tensor.vec_load((b_i, sq_i, hk_i, warp_k_vec_i), VALUES_PER_THREAD_K)
                    k_vec = k_tensor.vec_load((b_i, sq_i, hk_i, warp_k_vec_i), VALUES_PER_THREAD_K)
                    sq_vecs[ki] = q_vec.extf(acc_vec_t)
                    sk_vecs[ki] = k_vec.extf(acc_vec_t)

                if use_qk_l2norm:
                    sum_q_partial_vec = vector.from_elements(acc_vec_t, [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)])
                    sum_k_partial_vec = vector.from_elements(acc_vec_t, [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)])
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sum_q_partial_vec = sum_q_partial_vec + sq_vecs[ki] * sq_vecs[ki]
                        sum_k_partial_vec = sum_k_partial_vec + sk_vecs[ki] * sk_vecs[ki]
                        sum_q_partial = vector.ReductionOp(T.f32, vector.CombiningKind.ADD, sum_q_partial_vec).dest
                        sum_k_partial = vector.ReductionOp(T.f32, vector.CombiningKind.ADD, sum_k_partial_vec).dest
                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        sum_q_partial = sum_q_partial + mlir_gpu.ShuffleOp(sum_q_partial, _to_raw(arith.constant(offset, type=T.i32)), width_i32, mode="xor").shuffleResult
                        sum_k_partial = sum_k_partial + mlir_gpu.ShuffleOp(sum_k_partial, _to_raw(arith.constant(offset, type=T.i32)), width_i32, mode="xor").shuffleResult
                    local_sum_q = mlir_gpu.ShuffleOp(sum_q_partial, _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)), width_i32, mode="idx").shuffleResult
                    local_sum_k = mlir_gpu.ShuffleOp(sum_k_partial, _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)), width_i32, mode="idx").shuffleResult
                    inv_norm_q = mlir_math.rsqrt(local_sum_q + 1e-6)
                    inv_norm_k = mlir_math.rsqrt(local_sum_k + 1e-6)
                    inv_norm_q_vec = vector.BroadcastOp(acc_vec_t, inv_norm_q).vector
                    inv_norm_k_vec = vector.BroadcastOp(acc_vec_t, inv_norm_k).vector
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sq_vecs[ki] = sq_vecs[ki] * scale_vec * inv_norm_q_vec
                        sk_vecs[ki] = sk_vecs[ki] * inv_norm_k_vec
                else:
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sq_vecs[ki] = sq_vecs[ki] * scale_vec

                for vi in range_constexpr(WARP_TILE_V_ITERS):

                    global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                    r_v = v_tensor[b_i, sq_i, hv_i, global_v_i].extf(T.f32)

                    sum_hk = vector.from_elements(acc_vec_t, [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)])

                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] *= r_g_vec
                        sum_hk = vector.FMAOp(state_vecs[vi * WARP_TILE_K_ITERS + ki], sk_vecs[ki], sum_hk).result
                    
                    sum_hk = vector.ReductionOp(T.f32, vector.CombiningKind.ADD, sum_hk).dest
                    
                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        sum_hk = sum_hk + mlir_gpu.ShuffleOp(sum_hk, _to_raw(fx.Int32(offset)), width_i32, mode="xor").shuffleResult
                        
                    v_new = (r_v - sum_hk) * r_beta
                    v_new = mlir_gpu.ShuffleOp(v_new, _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)), width_i32, mode="idx").shuffleResult
                    v_new = vector.BroadcastOp(acc_vec_t, v_new)

                    sum_hq = vector.from_elements(acc_vec_t, [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)])

                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        h_old = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                        r_q_val = sq_vecs[ki]
                        r_k_val = sk_vecs[ki]
                        h_new = vector.FMAOp(r_k_val, v_new, h_old).result
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] = h_new
                        sum_hq = vector.FMAOp(h_new, r_q_val, sum_hq).result
                    
                    sum_hq = vector.ReductionOp(T.f32, vector.CombiningKind.ADD, sum_hq).dest

                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        sum_hq = sum_hq + mlir_gpu.ShuffleOp(sum_hq, _to_raw(arith.constant(offset, type=T.i32)), width_i32, mode="xor").shuffleResult
                    
                    sum_hq = sum_hq.truncf(dtype_)
                    write_cond = arith.cmpi(arith.CmpIPredicate.eq, fx.Index(warp_k_vec_start), fx.Index(0))
                    write_cond_if = scf.IfOp(write_cond, results_=[], has_else=False)
                    with ir.InsertionPoint(write_cond_if.then_block):
                        out_tensor[b_i, sq_i, hv_i, global_v_i] = sum_hq
                        scf.YieldOp([])

            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    state_tensor.vec_store((pool_idx, hv_i, global_v_i, warp_k_vec_i), state_vecs[vi * WARP_TILE_K_ITERS + ki], VALUES_PER_THREAD_K)
            scf.YieldOp([])
        return

    @flyc.jit
    def launch_gdr_decode_kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        dt_bias: fx.Tensor,
        A_log: fx.Tensor,
        indices: fx.Tensor,
        state: fx.Tensor,
        out: fx.Tensor,
        batch_size: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        
        gx = batch_size * num_v_heads * NUM_BLOCKS_PER_V_DIM
        gdr_decode_kernel._func.__name__ = KERNEL_NAME
        gdr_decode_kernel(
            query, key, value, a, b, dt_bias, A_log, indices, state, out, batch_size,
        ).launch(grid=(gx, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    _compile_hints = {}

    def _launch(*args, **kwargs):
        with CompilationContext.compile_hints(_compile_hints):
            return launch_gdr_decode_kernel(*args, **kwargs)

    _compile_cache = {}
    def _compile(query, key, value, a, b, dt_bias, A_log, indices, state, out, batch_size, stream):
        with CompilationContext.compile_hints(_compile_hints):
            lookup_key = (query.dtype, batch_size)
            if _compile_cache.get(lookup_key, None) is None:
                _compile_cache[lookup_key] = flyc.compile(launch_gdr_decode_kernel, 
                    query, key, value, a, b, dt_bias, A_log, indices, state.clone(), out, batch_size, stream)
            return _compile_cache[lookup_key]

    _launch.compile = _compile
    
    return _launch
