import torch
from dataclasses import dataclass
from _mlir import ir
from _mlir.ir import F16Type, BF16Type, F32Type, VectorType
from .ftensor import GTensor, STensor
import _mlir.extras.types as T
import functools

import flydsl
from flydsl.dialects.ext import flir, gpu, arith, vector
from flydsl.runtime.device import get_rocm_arch
from flydsl.compiler.pipeline import Pipeline, run_pipeline
from flydsl.dialects.ext.python_control_flow import (
    range_constexpr,
)
from flydsl.utils import SmemAllocator
from flydsl.kernels.kernels_common import stream_ptr_to_async_token

fm_fast = flir.arith.FastMathFlags.fast


@dataclass
class Args:
    dtype: torch.dtype
    b: int
    sq: int
    num_k_heads: int
    num_v_heads: int
    head_k_dim: int
    head_v_dim: int
    use_qk_l2norm: bool = True

    def __hash__(self):
        return hash(
            (
                self.dtype,
                self.b,
                self.sq,
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                self.use_qk_l2norm,
            )
        )


def create_fused_preshuffle_gdr_decode_kernel(
    dtype,
    VEC_SIZE: int,
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
    USE_PREFETCH: bool = False,
):
    _asv = arith.as_value
    _asid = flir.const_index
    _extf = flir.arith.extf
    fm_fast = flir.arith.FastMathFlags.fast

    def _extf32(value):
        return _extf(T.f32(), value)

    def _create_f32(value):
        return _extf32(_asv(float(value)))

    DYN = ir.ShapedType.get_dynamic_size()
    ARCH = get_rocm_arch()
    allocator = SmemAllocator(None, arch=ARCH)

    WARP_THREADS_V = 64 // WARP_THREADS_K
    VALUES_PER_THREAD_K = 4  # 16B

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
    assert TILE_V >= 1
    assert WARP_TILE_V_ITERS >= 1
    assert head_v_dim % NUM_BLOCKS_PER_V_DIM == 0
    assert TILE_V % WARP_GROUP_TILE_V == 0

    PREFETCH_VEC_SIZE = VEC_SIZE
    assert TILE_K <= (BLOCK_THREADS * PREFETCH_VEC_SIZE)

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

    class FusedGDRDecode(flir.MlirModule):
        GPU_MODULE_NAME = "linear_attention_kernels"
        GPU_MODULE_TARGETS = [f'#rocdl.target<chip = "{ARCH}">']

        def init_gpu_module(self):
            self.dtype = dtype.get()
            self.acc_type = T.f32()
            self.sq = allocator.allocate_array(T.f32(), seq_length * TILE_K)
            self.sk = allocator.allocate_array(T.f32(), seq_length * TILE_K)
            self.sr = allocator.allocate_array(T.f32(), 2 * NUM_WARPS)
            allocator.finalize()

        @flir.kernel
        def fused_gdr_decode_kernel(
            self: flir.T.i64,
            query: lambda: T.memref(DYN, dtype.get()),
            key: lambda: T.memref(DYN, dtype.get()),
            value: lambda: T.memref(DYN, dtype.get()),
            a: lambda: T.memref(DYN, dtype.get()),
            b: lambda: T.memref(DYN, dtype.get()),
            dt_bias: lambda: T.memref(DYN, dtype.get()),
            A_log: lambda: T.memref(DYN, F32Type.get()),
            indices: lambda: T.memref(DYN, T.i32()),
            state: lambda: T.memref(DYN, F32Type.get()),
            out: lambda: T.memref(DYN, dtype.get()),
            batch_size: lambda: T.index(),
            scale: lambda: T.f32(),
        ):
            width_i32 = arith.as_value(arith.constant(WARP_SIZE, type=T.i32()))
            acc_vec_t = VectorType.get([VALUES_PER_THREAD_K], self.acc_type)
            prefetch_acc_vec_t = VectorType.get([PREFETCH_VEC_SIZE], self.acc_type)

            tidx = flir.thread_idx("x")
            bidx = flir.block_idx("x")
            w_tid = tidx % WARP_SIZE
            wid = tidx // WARP_SIZE

            b_hv_i = bidx // NUM_BLOCKS_PER_V_DIM
            tile_v_start = bidx % NUM_BLOCKS_PER_V_DIM * TILE_V

            b_i = b_hv_i // num_v_heads
            hv_i = b_hv_i % num_v_heads
            hk_i = hv_i // (num_v_heads // num_k_heads)

            warp_k_vec_start = w_tid % WARP_THREADS_K * VALUES_PER_THREAD_K
            global_v_start = tile_v_start + wid * WARP_TILE_V + w_tid // WARP_THREADS_K
            k_prefetch_vec_i = tidx * PREFETCH_VEC_SIZE

            indices_tensor = GTensor(indices, T.i32(), (-1,))
            pool_idx = arith.index_cast(T.index(), indices_tensor[b_i])

            q_tensor = GTensor(
                query, dtype.get(), shape=(-1, seq_length, num_k_heads, head_k_dim)
            )
            k_tensor = GTensor(
                key, dtype.get(), shape=(-1, seq_length, num_k_heads, head_k_dim)
            )
            v_tensor = GTensor(
                value, dtype.get(), shape=(-1, seq_length, num_v_heads, head_v_dim)
            )
            a_tensor = GTensor(a, dtype.get(), shape=(-1, seq_length, num_v_heads))
            b_tensor = GTensor(b, dtype.get(), shape=(-1, seq_length, num_v_heads))
            dt_bias_tensor = GTensor(dt_bias, dtype.get(), shape=(num_v_heads,))
            A_log_tensor = GTensor(A_log, F32Type.get(), shape=(num_v_heads,))
            state_tensor = GTensor(
                state, F32Type.get(), shape=(-1, num_v_heads, head_v_dim, head_k_dim)
            )
            out_tensor = GTensor(
                out, dtype.get(), shape=(-1, seq_length, num_v_heads, head_v_dim)
            )

            sbase = allocator.get_base()
            sq_tensor = STensor(
                self.sq(sbase),
                T.f32(),
                shape=(
                    seq_length,
                    TILE_K,
                ),
            )
            sk_tensor = STensor(
                self.sk(sbase),
                T.f32(),
                shape=(
                    seq_length,
                    TILE_K,
                ),
            )
            sr_tensor = STensor(self.sr(sbase), T.f32(), shape=(-1,))

            if pool_idx >= 0:

                r_A_log = A_log_tensor[hv_i]
                r_dt_bias = _extf32(dt_bias_tensor[hv_i])

                if USE_PREFETCH:
                    scale_vec = vector.BroadcastOp(prefetch_acc_vec_t, _asv(scale))
                    for sq_i in range_constexpr(seq_length):
                        if use_qk_l2norm:
                            q_val_vec = vector.from_elements(
                                prefetch_acc_vec_t,
                                [
                                    _create_f32(0)
                                    for i in range_constexpr(PREFETCH_VEC_SIZE)
                                ],
                            )
                            k_val_vec = vector.from_elements(
                                prefetch_acc_vec_t,
                                [
                                    _create_f32(0)
                                    for i in range_constexpr(PREFETCH_VEC_SIZE)
                                ],
                            )
                            sum_q_partial_vec = vector.from_elements(
                                prefetch_acc_vec_t,
                                [
                                    _create_f32(0)
                                    for i in range_constexpr(PREFETCH_VEC_SIZE)
                                ],
                            )
                            sum_k_partial_vec = vector.from_elements(
                                prefetch_acc_vec_t,
                                [
                                    _create_f32(0)
                                    for i in range_constexpr(PREFETCH_VEC_SIZE)
                                ],
                            )
                            if k_prefetch_vec_i < TILE_K:
                                q_val_vec = q_tensor.vec_load(
                                    (b_i, sq_i, hk_i, k_prefetch_vec_i),
                                    PREFETCH_VEC_SIZE,
                                )
                                k_val_vec = k_tensor.vec_load(
                                    (b_i, sq_i, hk_i, k_prefetch_vec_i),
                                    PREFETCH_VEC_SIZE,
                                )
                                q_val_vec = flir.arith.extf(
                                    prefetch_acc_vec_t, _asv(q_val_vec)
                                )
                                k_val_vec = flir.arith.extf(
                                    prefetch_acc_vec_t, _asv(k_val_vec)
                                )
                                sum_q_partial_vec = q_val_vec * q_val_vec
                                sum_k_partial_vec = k_val_vec * k_val_vec
                            sum_q_partial = vector.ReductionOp(
                                T.f32(),
                                vector.CombiningKind.ADD,
                                _asv(sum_q_partial_vec),
                            ).dest
                            sum_k_partial = vector.ReductionOp(
                                T.f32(),
                                vector.CombiningKind.ADD,
                                _asv(sum_k_partial_vec),
                            ).dest
                            for offset in WARP_SIZE_SHFL_OFFSETS:
                                sum_q_partial = (
                                    sum_q_partial
                                    + gpu.ShuffleOp(
                                        _asv(sum_q_partial),
                                        _asv(arith.constant(offset, type=T.i32())),
                                        width_i32,
                                        mode="xor",
                                    ).shuffleResult
                                )
                                sum_k_partial = (
                                    sum_k_partial
                                    + gpu.ShuffleOp(
                                        _asv(sum_k_partial),
                                        _asv(arith.constant(offset, type=T.i32())),
                                        width_i32,
                                        mode="xor",
                                    ).shuffleResult
                                )
                            if w_tid == 0:
                                sr_tensor[wid] = sum_q_partial
                                sr_tensor[NUM_WARPS + wid] = sum_k_partial
                            gpu.barrier()
                            inv_norm_q = _create_f32(0)
                            inv_norm_k = _create_f32(0)
                            if wid == 0:
                                local_sum_q = _create_f32(0)
                                local_sum_k = _create_f32(0)
                                if w_tid < NUM_WARPS:
                                    local_sum_q = sr_tensor[w_tid]
                                    local_sum_k = sr_tensor[NUM_WARPS + w_tid]
                                for offset in WARP_SIZE_SHFL_OFFSETS:
                                    local_sum_q = (
                                        local_sum_q
                                        + gpu.ShuffleOp(
                                            _asv(local_sum_q),
                                            _asv(arith.constant(offset, type=T.i32())),
                                            width_i32,
                                            mode="xor",
                                        ).shuffleResult
                                    )
                                    local_sum_k = (
                                        local_sum_k
                                        + gpu.ShuffleOp(
                                            _asv(local_sum_k),
                                            _asv(arith.constant(offset, type=T.i32())),
                                            width_i32,
                                            mode="xor",
                                        ).shuffleResult
                                    )
                                if w_tid == 0:
                                    sr_tensor[0] = _extf32(
                                        _asv(
                                            flir.math.rsqrt(
                                                _extf32(_asv(local_sum_q + 1e-6)).value
                                            )
                                        )
                                    )
                                    sr_tensor[1] = _extf32(
                                        _asv(
                                            flir.math.rsqrt(
                                                _extf32(_asv(local_sum_k + 1e-6)).value
                                            )
                                        )
                                    )
                            gpu.barrier()
                            inv_norm_q = sr_tensor[0]
                            inv_norm_k = sr_tensor[1]
                            inv_norm_q_vec = vector.BroadcastOp(
                                prefetch_acc_vec_t, _asv(inv_norm_q)
                            )
                            inv_norm_k_vec = vector.BroadcastOp(
                                prefetch_acc_vec_t, _asv(inv_norm_k)
                            )
                            if k_prefetch_vec_i < TILE_K:
                                q_new_vec = q_val_vec * scale_vec * inv_norm_q_vec
                                k_new_vec = k_val_vec * inv_norm_k_vec
                                sq_tensor.vec_store(
                                    (sq_i, k_prefetch_vec_i),
                                    q_new_vec,
                                    PREFETCH_VEC_SIZE,
                                )
                                sk_tensor.vec_store(
                                    (sq_i, k_prefetch_vec_i),
                                    k_new_vec,
                                    PREFETCH_VEC_SIZE,
                                )
                        else:
                            if k_prefetch_vec_i < TILE_K:
                                q_vec = q_tensor.vec_load(
                                    (b_i, sq_i, hk_i, k_prefetch_vec_i),
                                    PREFETCH_VEC_SIZE,
                                )
                                k_vec = k_tensor.vec_load(
                                    (b_i, sq_i, hk_i, k_prefetch_vec_i),
                                    PREFETCH_VEC_SIZE,
                                )
                                q_vec = flir.arith.extf(prefetch_acc_vec_t, _asv(q_vec))
                                k_vec = flir.arith.extf(prefetch_acc_vec_t, _asv(k_vec))
                                q_vec = q_vec * scale_vec
                                sq_tensor.vec_store(
                                    (sq_i, k_prefetch_vec_i), q_vec, PREFETCH_VEC_SIZE
                                )
                                sk_tensor.vec_store(
                                    (sq_i, k_prefetch_vec_i), k_vec, PREFETCH_VEC_SIZE
                                )
                        gpu.barrier()

                state_vecs = [0] * (WARP_TILE_V_ITERS * WARP_TILE_K_ITERS)
                for vi in range_constexpr(WARP_TILE_V_ITERS):
                    global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] = state_tensor.vec_load(
                            (pool_idx, hv_i, global_v_i, warp_k_vec_i),
                            VALUES_PER_THREAD_K,
                        )

                for sq_i in range_constexpr(seq_length):

                    r_g = _create_f32(0)
                    r_beta = _create_f32(0)
                    if True:
                        r_a = _extf32(a_tensor[b_i, sq_i, hv_i])
                        r_b = _extf32(b_tensor[b_i, sq_i, hv_i])
                        x = r_a + r_dt_bias
                        beta_x = _create_f32(softplus_beta) * x
                        softplus_x = _create_f32(0)
                        if beta_x <= softplus_threshold:
                            softplus_x = _create_f32(
                                1.0 / softplus_beta
                            ) * flir.math.log1p(
                                _asv(flir.math.exp(_asv(beta_x), fastmath=fm_fast)),
                                fastmath=fm_fast,
                            )
                        else:
                            softplus_x = x
                        r_g_value = (
                            _create_f32(0)
                            - flir.math.exp(_asv(r_A_log), fastmath=fm_fast)
                            * softplus_x
                        )
                        r_beta = _create_f32(1) / (
                            _create_f32(1)
                            + flir.math.exp(
                                _asv(_create_f32(0) - r_b), fastmath=fm_fast
                            )
                        )
                        r_g = flir.math.exp(_asv(r_g_value), fastmath=fm_fast)
                    r_g_vec = vector.BroadcastOp(acc_vec_t, _asv(r_g))

                    sq_vecs = [0] * WARP_TILE_K_ITERS
                    sk_vecs = [0] * WARP_TILE_K_ITERS
                    if USE_PREFETCH:
                        for ki in range_constexpr(WARP_TILE_K_ITERS):
                            warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                            sq_vecs[ki] = sq_tensor.vec_load(
                                (sq_i, warp_k_vec_i), VALUES_PER_THREAD_K
                            )
                            sk_vecs[ki] = sk_tensor.vec_load(
                                (sq_i, warp_k_vec_i), VALUES_PER_THREAD_K
                            )
                    else:
                        scale_vec = vector.BroadcastOp(acc_vec_t, _asv(scale))
                        for ki in range_constexpr(WARP_TILE_K_ITERS):
                            warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                            q_vec = q_tensor.vec_load(
                                (b_i, sq_i, hk_i, warp_k_vec_i), VALUES_PER_THREAD_K
                            )
                            k_vec = k_tensor.vec_load(
                                (b_i, sq_i, hk_i, warp_k_vec_i), VALUES_PER_THREAD_K
                            )
                            sq_vecs[ki] = flir.arith.extf(acc_vec_t, _asv(q_vec))
                            sk_vecs[ki] = flir.arith.extf(acc_vec_t, _asv(k_vec))
                        if use_qk_l2norm:
                            sum_q_partial_vec = vector.from_elements(
                                acc_vec_t,
                                [
                                    _create_f32(0)
                                    for i in range_constexpr(VALUES_PER_THREAD_K)
                                ],
                            )
                            sum_k_partial_vec = vector.from_elements(
                                acc_vec_t,
                                [
                                    _create_f32(0)
                                    for i in range_constexpr(VALUES_PER_THREAD_K)
                                ],
                            )
                            for ki in range_constexpr(WARP_TILE_K_ITERS):
                                sum_q_partial_vec = (
                                    sum_q_partial_vec + sq_vecs[ki] * sq_vecs[ki]
                                )
                                sum_k_partial_vec = (
                                    sum_k_partial_vec + sk_vecs[ki] * sk_vecs[ki]
                                )
                            sum_q_partial = vector.ReductionOp(
                                T.f32(),
                                vector.CombiningKind.ADD,
                                _asv(sum_q_partial_vec),
                            ).dest
                            sum_k_partial = vector.ReductionOp(
                                T.f32(),
                                vector.CombiningKind.ADD,
                                _asv(sum_k_partial_vec),
                            ).dest
                            for offset in WARP_THREADS_K_SHFL_OFFSETS:
                                sum_q_partial = (
                                    sum_q_partial
                                    + gpu.ShuffleOp(
                                        _asv(sum_q_partial),
                                        _asv(arith.constant(offset, type=T.i32())),
                                        width_i32,
                                        mode="xor",
                                    ).shuffleResult
                                )
                                sum_k_partial = (
                                    sum_k_partial
                                    + gpu.ShuffleOp(
                                        _asv(sum_k_partial),
                                        _asv(arith.constant(offset, type=T.i32())),
                                        width_i32,
                                        mode="xor",
                                    ).shuffleResult
                                )
                            local_sum_q = gpu.ShuffleOp(
                                _asv(sum_q_partial),
                                _asv(
                                    arith.index_cast(
                                        T.i32(),
                                        w_tid // WARP_THREADS_K * WARP_THREADS_K,
                                    )
                                ),
                                width_i32,
                                mode="idx",
                            ).shuffleResult
                            local_sum_k = gpu.ShuffleOp(
                                _asv(sum_k_partial),
                                _asv(
                                    arith.index_cast(
                                        T.i32(),
                                        w_tid // WARP_THREADS_K * WARP_THREADS_K,
                                    )
                                ),
                                width_i32,
                                mode="idx",
                            ).shuffleResult
                            inv_norm_q = _extf32(
                                _asv(
                                    flir.math.rsqrt(
                                        _extf32(_asv(local_sum_q + 1e-6)).value
                                    )
                                )
                            )
                            inv_norm_k = _extf32(
                                _asv(
                                    flir.math.rsqrt(
                                        _extf32(_asv(local_sum_k + 1e-6)).value
                                    )
                                )
                            )
                            inv_norm_q_vec = vector.BroadcastOp(
                                acc_vec_t, _asv(inv_norm_q)
                            )
                            inv_norm_k_vec = vector.BroadcastOp(
                                acc_vec_t, _asv(inv_norm_k)
                            )
                            for ki in range_constexpr(WARP_TILE_K_ITERS):
                                sq_vecs[ki] = sq_vecs[ki] * scale_vec * inv_norm_q_vec
                                sk_vecs[ki] = sk_vecs[ki] * inv_norm_k_vec
                        else:
                            for ki in range_constexpr(WARP_TILE_K_ITERS):
                                sq_vecs[ki] = sq_vecs[ki] * scale_vec

                    dot_kq_vec = vector.from_elements(
                        acc_vec_t,
                        [_create_f32(0) for i in range_constexpr(VALUES_PER_THREAD_K)],
                    )
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        dot_kq_vec = vector.FMAOp(
                            _asv(sk_vecs[ki]), _asv(sq_vecs[ki]), _asv(dot_kq_vec)
                        ).result
                    dot_kq = vector.ReductionOp(
                        T.f32(), vector.CombiningKind.ADD, _asv(dot_kq_vec)
                    ).dest
                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        dot_kq = (
                            dot_kq
                            + gpu.ShuffleOp(
                                _asv(dot_kq),
                                _asv(arith.constant(offset, type=T.i32())),
                                width_i32,
                                mode="xor",
                            ).shuffleResult
                        )

                    for vi in range_constexpr(WARP_TILE_V_ITERS):

                        global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                        r_v = _extf32(v_tensor[b_i, sq_i, hv_i, global_v_i])

                        sum_hk = vector.from_elements(
                            acc_vec_t,
                            [
                                _create_f32(0)
                                for i in range_constexpr(VALUES_PER_THREAD_K)
                            ],
                        )
                        sum_hq_old = vector.from_elements(
                            acc_vec_t,
                            [
                                _create_f32(0)
                                for i in range_constexpr(VALUES_PER_THREAD_K)
                            ],
                        )

                        for ki in range_constexpr(WARP_TILE_K_ITERS):
                            state_vecs[vi * WARP_TILE_K_ITERS + ki] *= r_g_vec
                            h_cur = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                            sum_hk = vector.FMAOp(
                                _asv(h_cur), _asv(sk_vecs[ki]), _asv(sum_hk)
                            ).result
                            sum_hq_old = vector.FMAOp(
                                _asv(h_cur), _asv(sq_vecs[ki]), _asv(sum_hq_old)
                            ).result

                        sum_hk = vector.ReductionOp(
                            T.f32(), vector.CombiningKind.ADD, _asv(sum_hk)
                        ).dest
                        sum_hq_old = vector.ReductionOp(
                            T.f32(), vector.CombiningKind.ADD, _asv(sum_hq_old)
                        ).dest

                        for offset in WARP_THREADS_K_SHFL_OFFSETS:
                            sum_hk = (
                                sum_hk
                                + gpu.ShuffleOp(
                                    _asv(sum_hk),
                                    _asv(arith.constant(offset, type=T.i32())),
                                    width_i32,
                                    mode="xor",
                                ).shuffleResult
                            )
                            sum_hq_old = (
                                sum_hq_old
                                + gpu.ShuffleOp(
                                    _asv(sum_hq_old),
                                    _asv(arith.constant(offset, type=T.i32())),
                                    width_i32,
                                    mode="xor",
                                ).shuffleResult
                            )

                        v_new = (r_v - sum_hk) * r_beta
                        v_new = gpu.ShuffleOp(
                            _asv(v_new),
                            _asv(
                                arith.index_cast(
                                    T.i32(), w_tid // WARP_THREADS_K * WARP_THREADS_K
                                )
                            ),
                            width_i32,
                            mode="idx",
                        ).shuffleResult
                        sum_hq = sum_hq_old + v_new * dot_kq
                        v_new_bcast = vector.BroadcastOp(acc_vec_t, _asv(v_new))

                        for ki in range_constexpr(WARP_TILE_K_ITERS):
                            h_new = vector.FMAOp(
                                _asv(sk_vecs[ki]),
                                _asv(v_new_bcast),
                                _asv(state_vecs[vi * WARP_TILE_K_ITERS + ki]),
                            ).result
                            state_vecs[vi * WARP_TILE_K_ITERS + ki] = h_new

                        sum_hq = flir.arith.truncf(self.dtype, _asv(sum_hq))
                        if warp_k_vec_start == 0:
                            # sum_hq = flir.arith.truncf(self.dtype, _asv(sum_hq))
                            out_tensor[b_i, sq_i, hv_i, global_v_i] = sum_hq

                for vi in range_constexpr(WARP_TILE_V_ITERS):
                    global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                        state_tensor.vec_store(
                            (pool_idx, hv_i, global_v_i, warp_k_vec_i),
                            state_vecs[vi * WARP_TILE_K_ITERS + ki],
                            VALUES_PER_THREAD_K,
                        )
            return

        @flir.jit
        def __call__(
            self: flir.T.i64,
            query: lambda: T.memref(DYN, dtype.get()),
            key: lambda: T.memref(DYN, dtype.get()),
            value: lambda: T.memref(DYN, dtype.get()),
            a: lambda: T.memref(DYN, dtype.get()),
            b: lambda: T.memref(DYN, dtype.get()),
            dt_bias: lambda: T.memref(DYN, dtype.get()),
            A_log: lambda: T.memref(DYN, F32Type.get()),
            indices: lambda: T.memref(DYN, T.i32()),
            state: lambda: T.memref(DYN, F32Type.get()),
            out: lambda: T.memref(DYN, dtype.get()),
            batch_size: lambda: T.index(),
            scale: lambda: T.f32(),
            stream_ptr: lambda: T.i64(),
        ):
            c1 = arith.index(1)
            bx = arith.index(BLOCK_THREADS)
            gx = batch_size * num_v_heads * arith.index(NUM_BLOCKS_PER_V_DIM)
            stream_token = stream_ptr_to_async_token(stream_ptr)
            flir.gpu_ext.LaunchFuncOp(
                [self.GPU_MODULE_NAME, "fused_gdr_decode_kernel"],
                grid_size=(gx, c1, c1),
                block_size=(bx, c1, c1),
                kernel_operands=[
                    query,
                    key,
                    value,
                    a,
                    b,
                    dt_bias,
                    A_log,
                    indices,
                    state,
                    out,
                    batch_size,
                    scale,
                ],
                async_dependencies=[stream_token],
            )

    return FusedGDRDecode().module


def choose_kwargs(args):
    d = {}
    # if args.b == 1 or (args.b >= 16 and args.b <= 64):
    #     d['NUM_BLOCKS_PER_V_DIM'] = 2
    # else:
    #     d['NUM_BLOCKS_PER_V_DIM'] = 1
    return d


@functools.lru_cache(maxsize=1024)
def get_func(args):
    kwargs = choose_kwargs(args)
    func = create_fused_preshuffle_gdr_decode_kernel
    if args.dtype == torch.float:
        module = func(
            F32Type,
            4,
            args.sq,
            args.num_k_heads,
            args.num_v_heads,
            args.head_k_dim,
            args.head_v_dim,
            args.use_qk_l2norm,
            **kwargs,
        )
    elif args.dtype == torch.half:
        module = func(
            F16Type,
            8,
            args.sq,
            args.num_k_heads,
            args.num_v_heads,
            args.head_k_dim,
            args.head_v_dim,
            args.use_qk_l2norm,
            **kwargs,
        )
    elif args.dtype == torch.bfloat16:
        module = func(
            BF16Type,
            8,
            args.sq,
            args.num_k_heads,
            args.num_v_heads,
            args.head_k_dim,
            args.head_v_dim,
            args.use_qk_l2norm,
            **kwargs,
        )
    optimized = run_pipeline(module, Pipeline().canonicalize().cse())
    EXE = flydsl.compile(optimized)
    return EXE
