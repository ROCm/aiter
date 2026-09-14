# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import (
    gpu as mlir_gpu,
)
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from .tensor_shim import (
    _to_raw,
    get_dtype_bytes,
    get_dtype_in_kernel,
)

_LOG2E = 1.4426950408889634


def _gview(tensor, base, shape, stride):
    it = fx.get_iter(fx.rocdl.make_buffer_tensor(tensor, max_size=True))
    if base is not None:
        it = fx.add_offset(it, base)
    return fx.Tensor(fx.make_view(it, fx.make_layout(shape, stride)))


def _gview64(tensor, base, shape, stride):
    """Like :func:`_gview`, but ``base`` holds past a 32-bit offset.

    Shifting the descriptor pointer instead can land the term in the
    descriptor's own 32-bit offset and wrap silently; building the descriptor
    off an already-shifted global pointer keeps it 64-bit.
    """
    it = fx.add_offset(fx.get_iter(tensor), base)
    return fx.Tensor(
        fx.make_view(fx.rocdl.make_buffer_ptr(it), fx.make_layout(shape, stride))
    )


def _load_vec(atom, tile, width, numeric):
    frag = fx.make_rmem_tensor(width, numeric)
    fx.copy(atom, tile, frag)
    vec = frag.load()
    return vec[0] if width == 1 else vec


def _store_vec(atom, tile, value, width, numeric):
    frag = fx.make_rmem_tensor(width, numeric)
    frag.store(fx.Vector.from_elements([value], dtype=numeric) if width == 1 else value)
    fx.copy(atom, frag, tile)


def _fast_exp(x):
    return rocdl.exp2(T.f32, _to_raw(fx.Float32(x) * _LOG2E))


@functools.lru_cache(maxsize=1024)
def create_vk_gdr_decode_kernel(
    dtype: str,
    A_log_dtype: str,
    state_dtype: str,
    seq_length: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    q_strides: tuple,
    k_strides: tuple,
    v_strides: tuple,
    state_strides: tuple,
    a_strides: tuple,
    b_strides: tuple,
    use_qk_l2norm: bool,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    NUM_BLOCKS_PER_V_DIM: int = 1,
    NUM_WARPS: int = 4,
    WARP_THREADS_K: int = 8,
):
    SCALE_VALUE = float(1.0 / (float(head_k_dim) ** 0.5))
    WARP_THREADS_V = 64 // WARP_THREADS_K

    if "f32" in state_dtype:
        VALUES_PER_THREAD_K = 4  # 16B
    else:
        VALUES_PER_THREAD_K = 8  # 16B
    data_num = fx.BFloat16 if dtype == "bf16" else fx.Float16
    A_log_num = {
        "f32": fx.Float32,
        "f16": fx.Float16,
        "bf16": fx.BFloat16,
    }[A_log_dtype]
    state_num = {
        "f32": fx.Float32,
        "f16": fx.Float16,
        "bf16": fx.BFloat16,
    }[state_dtype]

    WARP_SIZE = WARP_THREADS_V * WARP_THREADS_K
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE
    assert WARP_SIZE == 64

    WARP_TILE_K = WARP_THREADS_K * VALUES_PER_THREAD_K
    WARP_TILE_K_ITERS = head_k_dim // WARP_TILE_K
    assert WARP_TILE_K_ITERS >= 1
    assert head_k_dim % WARP_TILE_K == 0

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
        read_indices: fx.Tensor,
        write_indices: fx.Tensor,
        state: fx.Tensor,
        out: fx.Tensor,
        batch_size: fx.Int32,
    ):
        scale = fx.Float32(SCALE_VALUE)
        softplus_beta_ = fx.Float32(softplus_beta)
        softplus_threshold_ = fx.Float32(softplus_threshold)

        f32_0 = fx.Float32(0.0)
        f32_1 = fx.Float32(1.0)
        width_i32 = _to_raw(fx.Int32(WARP_SIZE))

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

        cp_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        cp_data = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), data_num)
        cp_data_vec = fx.make_copy_atom(
            fx.rocdl.BufferCopy(data_num.width * VALUES_PER_THREAD_K), data_num
        )
        cp_A_log = fx.make_copy_atom(fx.rocdl.BufferCopy(A_log_num.width), A_log_num)
        cp_state_vec = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), state_num)

        read_indices_view = _gview(read_indices, None, (batch_size, 1), (1, 1))
        write_indices_view = _gview(write_indices, None, (batch_size, 1), (1, 1))
        read_pool_idx = _load_vec(
            cp_i32, fx.slice(read_indices_view, (b_i, None)), 1, fx.Int32
        )
        write_pool_idx = _load_vec(
            cp_i32, fx.slice(write_indices_view, (b_i, None)), 1, fx.Int32
        )

        q_view = _gview(
            query,
            None,
            (
                batch_size,
                seq_length,
                num_k_heads,
                head_k_dim // VALUES_PER_THREAD_K,
                VALUES_PER_THREAD_K,
            ),
            (*q_strides[:-1], VALUES_PER_THREAD_K, 1),
        )
        k_view = _gview(
            key,
            None,
            (
                batch_size,
                seq_length,
                num_k_heads,
                head_k_dim // VALUES_PER_THREAD_K,
                VALUES_PER_THREAD_K,
            ),
            (*k_strides[:-1], VALUES_PER_THREAD_K, 1),
        )
        v_view = _gview(
            value,
            None,
            (batch_size, seq_length, num_v_heads, head_v_dim, 1),
            (*v_strides, 1),
        )
        a_view = _gview(
            a,
            None,
            (batch_size, seq_length, num_v_heads, 1),
            (*a_strides, 1),
        )
        b_view = _gview(
            b,
            None,
            (batch_size, seq_length, num_v_heads, 1),
            (*b_strides, 1),
        )
        dt_bias_view = _gview(dt_bias, None, (num_v_heads, 1), (1, 1))
        A_log_view = _gview(A_log, None, (num_v_heads, 1), (1, 1))
        out_view = _gview(
            out,
            None,
            (batch_size, seq_length, num_v_heads, head_v_dim, 1),
            (
                seq_length * num_v_heads * head_v_dim,
                num_v_heads * head_v_dim,
                head_v_dim,
                1,
                1,
            ),
        )

        state_shape = (
            num_v_heads,
            head_v_dim,
            head_k_dim // VALUES_PER_THREAD_K,
            VALUES_PER_THREAD_K,
        )
        state_stride = (
            state_strides[1],
            state_strides[2],
            VALUES_PER_THREAD_K,
            1,
        )
        read_state_view = _gview(
            state,
            fx.Int64(read_pool_idx) * fx.Int64(state_strides[0]),
            state_shape,
            state_stride,
        )
        write_state_view = _gview(
            state,
            fx.Int64(write_pool_idx) * fx.Int64(state_strides[0]),
            state_shape,
            state_stride,
        )

        # Skip CG-pad slots (indices sentinel < 0). The guarded body is a
        # closure so the runtime `if` sees an opaque call and lowers to scf.if.
        def _do_decode():
            r_A_log = _load_vec(
                cp_A_log, fx.slice(A_log_view, (hv_i, None)), 1, A_log_num
            )
            if const_expr("f32" not in A_log_dtype):
                r_A_log = r_A_log.to(fx.Float32)
            r_dt_bias = _load_vec(
                cp_data, fx.slice(dt_bias_view, (hv_i, None)), 1, data_num
            ).to(fx.Float32)

            state_vecs = [0] * (WARP_TILE_V_ITERS * WARP_TILE_K_ITERS)
            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    state_vecs[vi * WARP_TILE_K_ITERS + ki] = _load_vec(
                        cp_state_vec,
                        fx.slice(
                            read_state_view,
                            (
                                hv_i,
                                global_v_i,
                                warp_k_vec_i // VALUES_PER_THREAD_K,
                                None,
                            ),
                        ),
                        VALUES_PER_THREAD_K,
                        state_num,
                    )
                    if const_expr("f32" not in state_dtype):
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] = state_vecs[
                            vi * WARP_TILE_K_ITERS + ki
                        ].to(fx.Float32)

            for sq_i in range_constexpr(seq_length):
                r_a = _load_vec(
                    cp_data,
                    fx.slice(a_view, (b_i, sq_i, hv_i, None)),
                    1,
                    data_num,
                ).to(fx.Float32)
                r_b = _load_vec(
                    cp_data,
                    fx.slice(b_view, (b_i, sq_i, hv_i, None)),
                    1,
                    data_num,
                ).to(fx.Float32)
                x = r_a + r_dt_bias
                beta_x = softplus_beta_ * x

                # For beta_x > threshold, softplus(x) == x; both arms run and
                # the overflowing one is dropped.
                softplus_big = (f32_1 / softplus_beta_) * fx.math.log1p(
                    _fast_exp(beta_x)
                )
                softplus_x = (
                    fx.Float32(beta_x) <= fx.Float32(softplus_threshold_)
                ).select(softplus_big, x)

                r_g_value = -_fast_exp(r_A_log) * softplus_x
                r_beta = f32_1 / (f32_1 + _fast_exp(-r_b))
                r_g = _fast_exp(r_g_value)

                r_g_vec = fx.Vector.filled(
                    VALUES_PER_THREAD_K, fx.Float32(r_g), fx.Float32
                )

                sq_vecs = [0] * WARP_TILE_K_ITERS
                sk_vecs = [0] * WARP_TILE_K_ITERS

                scale_vec = fx.Vector.filled(
                    VALUES_PER_THREAD_K, fx.Float32(scale), fx.Float32
                )

                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    q_vec = _load_vec(
                        cp_data_vec,
                        fx.slice(
                            q_view,
                            (
                                b_i,
                                sq_i,
                                hk_i,
                                warp_k_vec_i // VALUES_PER_THREAD_K,
                                None,
                            ),
                        ),
                        VALUES_PER_THREAD_K,
                        data_num,
                    )
                    k_vec = _load_vec(
                        cp_data_vec,
                        fx.slice(
                            k_view,
                            (
                                b_i,
                                sq_i,
                                hk_i,
                                warp_k_vec_i // VALUES_PER_THREAD_K,
                                None,
                            ),
                        ),
                        VALUES_PER_THREAD_K,
                        data_num,
                    )
                    sq_vecs[ki] = q_vec.to(fx.Float32)
                    sk_vecs[ki] = k_vec.to(fx.Float32)

                if const_expr(use_qk_l2norm):
                    sum_q_partial_vec = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    sum_k_partial_vec = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sum_q_partial_vec = (
                            sum_q_partial_vec + sq_vecs[ki] * sq_vecs[ki]
                        )
                        sum_k_partial_vec = (
                            sum_k_partial_vec + sk_vecs[ki] * sk_vecs[ki]
                        )
                    sum_q_partial = fx.Vector(sum_q_partial_vec).reduce(
                        fx.ReductionOp.ADD
                    )
                    sum_k_partial = fx.Vector(sum_k_partial_vec).reduce(
                        fx.ReductionOp.ADD
                    )
                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        sum_q_partial = sum_q_partial + sum_q_partial.shuffle_xor(
                            offset, WARP_SIZE
                        )
                        sum_k_partial = sum_k_partial + sum_k_partial.shuffle_xor(
                            offset, WARP_SIZE
                        )
                    local_sum_q = mlir_gpu.ShuffleOp(
                        _to_raw(sum_q_partial),
                        _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)),
                        width_i32,
                        mode="idx",
                    ).shuffleResult
                    local_sum_k = mlir_gpu.ShuffleOp(
                        _to_raw(sum_k_partial),
                        _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)),
                        width_i32,
                        mode="idx",
                    ).shuffleResult
                    inv_norm_q = fx.math.rsqrt(local_sum_q + 1e-6)
                    inv_norm_k = fx.math.rsqrt(local_sum_k + 1e-6)
                    inv_norm_q_vec = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(inv_norm_q), fx.Float32
                    )
                    inv_norm_k_vec = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(inv_norm_k), fx.Float32
                    )
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sq_vecs[ki] = sq_vecs[ki] * inv_norm_q_vec * scale_vec
                        sk_vecs[ki] = sk_vecs[ki] * inv_norm_k_vec
                else:
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sq_vecs[ki] = sq_vecs[ki] * scale_vec

                dot_kq_vec = fx.Vector.from_elements(
                    [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)], fx.Float32
                )
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    dot_kq_vec = fx.math.fma(sk_vecs[ki], sq_vecs[ki], dot_kq_vec)
                dot_kq = dot_kq_vec.reduce(fx.ReductionOp.ADD)
                for offset in WARP_THREADS_K_SHFL_OFFSETS:
                    dot_kq = dot_kq + dot_kq.shuffle_xor(offset, WARP_SIZE)

                for vi in range_constexpr(WARP_TILE_V_ITERS):
                    global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                    r_v = _load_vec(
                        cp_data,
                        fx.slice(v_view, (b_i, sq_i, hv_i, global_v_i, None)),
                        1,
                        data_num,
                    ).to(fx.Float32)

                    sum_hk = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    sum_hq_old = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )

                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] *= r_g_vec
                        h_cur = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                        sum_hk = fx.math.fma(h_cur, sk_vecs[ki], sum_hk)
                        sum_hq_old = fx.math.fma(h_cur, sq_vecs[ki], sum_hq_old)

                    sum_hk = sum_hk.reduce(fx.ReductionOp.ADD)
                    sum_hq_old = sum_hq_old.reduce(fx.ReductionOp.ADD)

                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        sum_hk = sum_hk + sum_hk.shuffle_xor(offset, WARP_SIZE)
                        sum_hq_old = sum_hq_old + sum_hq_old.shuffle_xor(
                            offset, WARP_SIZE
                        )

                    v_new = (r_v - sum_hk) * r_beta
                    v_new = mlir_gpu.ShuffleOp(
                        _to_raw(v_new),
                        _to_raw(fx.Int32(w_tid // WARP_THREADS_K * WARP_THREADS_K)),
                        width_i32,
                        mode="idx",
                    ).shuffleResult
                    sum_hq = sum_hq_old + v_new * dot_kq
                    v_new_bcast = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(v_new), fx.Float32
                    )

                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        h_new = fx.math.fma(
                            sk_vecs[ki],
                            v_new_bcast,
                            state_vecs[vi * WARP_TILE_K_ITERS + ki],
                        )
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] = h_new

                    sum_hq = sum_hq.to(data_num)

                    # Only k-vec lane 0 writes the q output.
                    def _write_q(_sum_hq=sum_hq, _gv=global_v_i, _sq=sq_i):
                        _store_vec(
                            cp_data,
                            fx.slice(out_view, (b_i, _sq, hv_i, _gv, None)),
                            _sum_hq,
                            1,
                            data_num,
                        )

                    if warp_k_vec_start == 0:
                        _write_q()

            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    if const_expr("f32" in state_dtype):
                        out_vec = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                    else:
                        out_vec = state_vecs[vi * WARP_TILE_K_ITERS + ki].to(state_num)
                    _store_vec(
                        cp_state_vec,
                        fx.slice(
                            write_state_view,
                            (
                                hv_i,
                                global_v_i,
                                warp_k_vec_i // VALUES_PER_THREAD_K,
                                None,
                            ),
                        ),
                        out_vec,
                        VALUES_PER_THREAD_K,
                        state_num,
                    )

        def _zero_padding_output():
            zero = fx.Float32(0.0).to(data_num)
            for sq_i in range_constexpr(seq_length):
                for vi in range_constexpr(WARP_TILE_V_ITERS):
                    global_v_i = global_v_start + vi * WARP_GROUP_TILE_V

                    def _write_zero(_sq=sq_i, _gv=global_v_i):
                        _store_vec(
                            cp_data,
                            fx.slice(out_view, (b_i, _sq, hv_i, _gv, None)),
                            zero,
                            1,
                            data_num,
                        )

                    if warp_k_vec_start == 0:
                        _write_zero()

        if (read_pool_idx >= 0) & (write_pool_idx >= 0):
            _do_decode()
        else:
            _zero_padding_output()

    @flyc.jit
    def launch_gdr_decode_kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        dt_bias: fx.Tensor,
        A_log: fx.Tensor,
        read_indices: fx.Tensor,
        write_indices: fx.Tensor,
        state: fx.Tensor,
        out: fx.Tensor,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        gx = batch_size * num_v_heads * NUM_BLOCKS_PER_V_DIM
        gdr_decode_kernel._func.__name__ = KERNEL_NAME
        gdr_decode_kernel(
            query,
            key,
            value,
            a,
            b,
            dt_bias,
            A_log,
            read_indices,
            write_indices,
            state,
            out,
            batch_size,
        ).launch(grid=(gx, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_gdr_decode_kernel


# Non-temporal bit a copy atom's cache modifier takes; only stores honour it.
NT_STORE = 2

MTP_MODE_CHAIN = "chain"
MTP_MODE_SNAPSHOT = "snapshot"


@functools.lru_cache(maxsize=1024)
def create_vk_gdr_mtp_kernel(
    dtype: str,
    A_log_dtype: str,
    state_dtype: str,
    inter_dtype: str,
    seq_length: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    q_strides: tuple,
    k_strides: tuple,
    v_strides: tuple,
    state_strides: tuple,
    a_strides: tuple,
    b_strides: tuple,
    si_strides: tuple,
    inter_strides: tuple,
    parent_strides: tuple,
    use_qk_l2norm: bool,
    mode: str,
    has_tree: bool = False,
    disable_state_update: bool = False,
    softplus_beta: float = 1.0,
    softplus_threshold: float = 20.0,
    NUM_BLOCKS_PER_V_DIM: int = 1,
    NUM_WARPS: int = 4,
    WARP_THREADS_K: int = 8,
    # Unused here; a parameter so it joins the cache key.
    WAVES_PER_EU: int = 0,
    min_live_slot: int = 0,
):
    """Gated delta rule over a speculative draft window.

    Undoing a rejected token needs a rollback point and a per-token record;
    ``mode`` picks whose contract to follow.

    ``MTP_MODE_CHAIN`` is vLLM's: rollback at
    ``state_indices[n, num_accepted - 1]``, and every token checkpoints into
    ``state_indices[n, t]``, the last of which is the final store.

    ``MTP_MODE_SNAPSHOT`` is SGLang's: rollback at ``state_indices[n]``, record
    in ``intermediate_states_buffer``. ``has_tree`` makes the draft an EAGLE
    tree, each token restarting from its parent's snapshot;
    ``disable_state_update`` suppresses the write-back.
    """
    assert mode in (MTP_MODE_CHAIN, MTP_MODE_SNAPSHOT), f"unknown MTP mode {mode!r}"
    assert min_live_slot in (0, 1)
    CHAIN = mode == MTP_MODE_CHAIN
    SNAPSHOT = not CHAIN
    TREE = bool(has_tree)
    NO_STATE_WRITE = bool(disable_state_update)
    assert not TREE or SNAPSHOT, "the EAGLE tree is the snapshot mode's"
    assert not TREE or len(inter_strides) == 5, "tree needs a snapshot buffer"
    SAVE_INTER = SNAPSHOT and len(inter_strides) == 5

    # Only an f32 snapshot reloads exactly; a narrower one rounds.
    LOSSLESS_SNAPSHOT = TREE and "f32" in inter_dtype

    SCALE_VALUE = float(1.0 / (float(head_k_dim) ** 0.5))
    WARP_THREADS_V = 64 // WARP_THREADS_K

    if "f32" in state_dtype:
        VALUES_PER_THREAD_K = 4  # 16B
    else:
        VALUES_PER_THREAD_K = 8  # 16B

    _NUM_BY_DTYPE = {"f32": fx.Float32, "f16": fx.Float16, "bf16": fx.BFloat16}
    data_num = _NUM_BY_DTYPE[dtype]
    A_log_num = _NUM_BY_DTYPE[A_log_dtype]
    state_num = _NUM_BY_DTYPE[state_dtype]
    inter_num = _NUM_BY_DTYPE[inter_dtype]

    WARP_SIZE = WARP_THREADS_V * WARP_THREADS_K
    BLOCK_THREADS = NUM_WARPS * WARP_SIZE
    assert WARP_SIZE == 64

    WARP_TILE_K = WARP_THREADS_K * VALUES_PER_THREAD_K
    WARP_TILE_K_ITERS = head_k_dim // WARP_TILE_K
    assert WARP_TILE_K_ITERS >= 1
    assert head_k_dim % WARP_TILE_K == 0

    WARP_TILE_V = WARP_THREADS_V
    WARP_GROUP_TILE_V = NUM_WARPS * WARP_TILE_V
    TILE_V = head_v_dim // NUM_BLOCKS_PER_V_DIM
    WARP_TILE_V_ITERS = TILE_V // WARP_GROUP_TILE_V
    assert TILE_V >= 1 and head_v_dim % NUM_BLOCKS_PER_V_DIM == 0
    assert WARP_TILE_V_ITERS >= 1 and TILE_V % WARP_GROUP_TILE_V == 0

    STATE_REGS = WARP_TILE_V_ITERS * WARP_TILE_K_ITERS * VALUES_PER_THREAD_K
    VGPR_PER_WAVE_AT_4 = 512 // 4

    WARP_THREADS_K_SHFL_OFFSETS = []
    offsets_ = WARP_THREADS_K // 2
    while offsets_ >= 1:
        WARP_THREADS_K_SHFL_OFFSETS.append(int(offsets_))
        offsets_ /= 2
    WARP_THREADS_K_SHFL_OFFSETS = WARP_THREADS_K_SHFL_OFFSETS[::-1]

    INTER_BYTES = get_dtype_bytes(inter_dtype) if SAVE_INTER else 0

    assert not SAVE_INTER or VALUES_PER_THREAD_K * INTER_BYTES <= 16, (
        f"a {state_dtype} state splits K {VALUES_PER_THREAD_K} ways, so a "
        f"{inter_dtype} snapshot needs a "
        f"{VALUES_PER_THREAD_K * INTER_BYTES}-byte store; the snapshot dtype "
        f"cannot be wider than the state's"
    )

    KERNEL_NAME = f"gdr_mtp_{mode}_{dtype}_kh{num_k_heads}x{head_k_dim}_vh{num_v_heads}x{head_v_dim}_q{seq_length}"
    if TREE:
        KERNEL_NAME += "_tree"
    if SAVE_INTER:
        KERNEL_NAME += "_snap"
    if NO_STATE_WRITE:
        KERNEL_NAME += "_nowrite"
    KERNEL_NAME += f"_{NUM_WARPS}w{WARP_THREADS_V}x{WARP_THREADS_K}"
    KERNEL_NAME += f"_vs{NUM_BLOCKS_PER_V_DIM}"

    @flyc.kernel
    def gdr_mtp_kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        dt_bias: fx.Tensor,
        A_log: fx.Tensor,
        state_indices: fx.Tensor,
        num_accepted: fx.Tensor,
        inter_indices: fx.Tensor,
        parent_tokens: fx.Tensor,
        state: fx.Tensor,
        inter_buffer: fx.Tensor,
        out: fx.Tensor,
        batch_size: fx.Int32,
    ):
        scale = fx.Float32(SCALE_VALUE)
        softplus_beta_ = fx.Float32(softplus_beta)
        inv_softplus_beta_ = fx.Float32(1.0 / softplus_beta)
        softplus_threshold_ = fx.Float32(softplus_threshold)

        f32_0 = fx.Float32(0.0)
        f32_1 = fx.Float32(1.0)
        state_vec_t = T.vec(VALUES_PER_THREAD_K, get_dtype_in_kernel(state_dtype))
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

        cp_i32 = fx.make_copy_atom(fx.rocdl.BufferCopy32b(), fx.Int32)
        cp_data = fx.make_copy_atom(fx.rocdl.BufferCopy16b(), data_num)
        cp_data_vec = fx.make_copy_atom(
            fx.rocdl.BufferCopy(data_num.width * VALUES_PER_THREAD_K), data_num
        )
        cp_A_log = fx.make_copy_atom(fx.rocdl.BufferCopy(A_log_num.width), A_log_num)
        cp_state_vec = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), state_num)
        cp_inter_vec = fx.make_copy_atom(
            fx.rocdl.BufferCopy(inter_num.width * VALUES_PER_THREAD_K), inter_num
        )
        # A snapshot slot nothing descends from is written past the cache.
        cp_inter_vec_nt = fx.make_copy_atom(
            fx.rocdl.BufferCopy(
                inter_num.width * VALUES_PER_THREAD_K, cache_modifier=NT_STORE
            ),
            inter_num,
        )

        # Addressed with the caller's strides, so 1-D [B] and 2-D [B, T] both map.
        si_view = _gview(
            state_indices, None, (batch_size, seq_length, 1), (*si_strides, 1)
        )

        def _slot_at(token):
            return fx.Int32(
                _load_vec(cp_i32, fx.slice(si_view, (b_i, token, None)), 1, fx.Int32)
            )

        if const_expr(CHAIN):
            token_slots = [_slot_at(t) for t in range_constexpr(seq_length)]

        if const_expr(CHAIN):
            nacc_view = _gview(num_accepted, None, (batch_size, 1), (1, 1))
            read_token = fx.Int32(
                _load_vec(cp_i32, fx.slice(nacc_view, (b_i, None)), 1, fx.Int32)
            ) - fx.Int32(1)
            read_slot = _slot_at(read_token)
        else:
            read_slot = _slot_at(0)

        if const_expr(SAVE_INTER):
            isi_view = _gview(inter_indices, None, (batch_size, 1), (1, 1))
            cache_idx = fx.Int32(
                _load_vec(cp_i32, fx.slice(isi_view, (b_i, None)), 1, fx.Int32)
            )
        if const_expr(TREE):
            parent_idx_view = _gview(
                parent_tokens, None, (batch_size, seq_length, 1), (*parent_strides, 1)
            )

        q_view = _gview(
            query,
            None,
            (
                batch_size,
                seq_length,
                num_k_heads,
                head_k_dim // VALUES_PER_THREAD_K,
                VALUES_PER_THREAD_K,
            ),
            (*q_strides[:-1], VALUES_PER_THREAD_K, 1),
        )
        k_view = _gview(
            key,
            None,
            (
                batch_size,
                seq_length,
                num_k_heads,
                head_k_dim // VALUES_PER_THREAD_K,
                VALUES_PER_THREAD_K,
            ),
            (*k_strides[:-1], VALUES_PER_THREAD_K, 1),
        )
        v_view = _gview(
            value,
            None,
            (batch_size, seq_length, num_v_heads, head_v_dim, 1),
            (*v_strides, 1),
        )
        a_view = _gview(
            a, None, (batch_size, seq_length, num_v_heads, 1), (*a_strides, 1)
        )
        b_view = _gview(
            b, None, (batch_size, seq_length, num_v_heads, 1), (*b_strides, 1)
        )
        dt_bias_view = _gview(dt_bias, None, (num_v_heads, 1), (1, 1))
        A_log_view = _gview(A_log, None, (num_v_heads, 1), (1, 1))
        out_view = _gview(
            out,
            None,
            (batch_size, seq_length, num_v_heads, head_v_dim, 1),
            (
                seq_length * num_v_heads * head_v_dim,
                num_v_heads * head_v_dim,
                head_v_dim,
                1,
                1,
            ),
        )

        # Only the term scaling with the pool or the snapshot buffer needs the
        # 64-bit base; the buffer offset is 32 bits.
        vec_shape = (
            num_v_heads,
            head_v_dim,
            head_k_dim // VALUES_PER_THREAD_K,
            VALUES_PER_THREAD_K,
        )

        def _state_at(slot):
            return _gview64(
                state,
                fx.Int64(slot) * fx.Int64(state_strides[0]),
                vec_shape,
                (state_strides[1], state_strides[2], VALUES_PER_THREAD_K, 1),
            )

        # One descriptor for the sequence's whole record: a step is a layout
        # dimension, since the window's own span stays well inside 32 bits.
        if const_expr(SAVE_INTER):
            inter_view = _gview64(
                inter_buffer,
                fx.Int64(cache_idx) * fx.Int64(inter_strides[0]),
                (seq_length, *vec_shape),
                (
                    inter_strides[1],
                    inter_strides[2],
                    inter_strides[3],
                    VALUES_PER_THREAD_K,
                    1,
                ),
            )

        # The tree emits the body once per arm, so a hoisted value is live across both.
        HOIST_ENTRY = not TREE

        def _taps(sq):
            """The gate and value scalars one token reads."""
            return (
                _load_vec(
                    cp_data, fx.slice(a_view, (b_i, sq, hv_i, None)), 1, data_num
                ).to(fx.Float32),
                _load_vec(
                    cp_data, fx.slice(b_view, (b_i, sq, hv_i, None)), 1, data_num
                ).to(fx.Float32),
                [
                    _load_vec(
                        cp_data,
                        fx.slice(
                            v_view,
                            (
                                b_i,
                                sq,
                                hv_i,
                                global_v_start + vi * WARP_GROUP_TILE_V,
                                None,
                            ),
                        ),
                        1,
                        data_num,
                    ).to(fx.Float32)
                    for vi in range_constexpr(WARP_TILE_V_ITERS)
                ],
            )

        def _A_log_tap():
            r = _load_vec(cp_A_log, fx.slice(A_log_view, (hv_i, None)), 1, A_log_num)
            if const_expr("f32" not in A_log_dtype):
                r = r.to(fx.Float32)
            return r

        def _dt_bias_tap():
            return _load_vec(
                cp_data, fx.slice(dt_bias_view, (hv_i, None)), 1, data_num
            ).to(fx.Float32)

        if const_expr(HOIST_ENTRY):
            entry_A_log = _A_log_tap()
            entry_dt_bias = _dt_bias_tap()
            entry_taps = _taps(0)

        # reload_parents and snapshot are traced flags, not runtime ones: a value
        # defined inside an scf.if does not dominate its use after it.
        def _do_mtp(reload_parents=False, snapshot="no"):
            if const_expr(not HOIST_ENTRY):
                r_A_log = _A_log_tap()
                r_dt_bias = _dt_bias_tap()
            else:
                r_A_log = entry_A_log
                r_dt_bias = entry_dt_bias

            read_state_view = _state_at(read_slot)
            state_vecs = [0] * (WARP_TILE_V_ITERS * WARP_TILE_K_ITERS)
            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    state_vecs[vi * WARP_TILE_K_ITERS + ki] = _load_vec(
                        cp_state_vec,
                        fx.slice(
                            read_state_view,
                            (
                                hv_i,
                                global_v_i,
                                warp_k_vec_i // VALUES_PER_THREAD_K,
                                None,
                            ),
                        ),
                        VALUES_PER_THREAD_K,
                        state_num,
                    )
                    if const_expr("f32" in state_dtype):
                        pass
                    else:
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] = state_vecs[
                            vi * WARP_TILE_K_ITERS + ki
                        ].extf(acc_vec_t)

            taps = entry_taps if const_expr(HOIST_ENTRY) else _taps(0)

            for sq_i in range_constexpr(seq_length):
                # Token 1's parent is token 0, still in registers; any later
                # token's parent is only known at runtime.
                held = reload_parents and sq_i == 1
                if const_expr(held and not LOSSLESS_SNAPSHOT):
                    snap_vec_t = T.vec(
                        VALUES_PER_THREAD_K, get_dtype_in_kernel(inter_dtype)
                    )
                    for si in range_constexpr(WARP_TILE_V_ITERS * WARP_TILE_K_ITERS):
                        state_vecs[si] = (
                            state_vecs[si].truncf(snap_vec_t).extf(acc_vec_t)
                        )
                if const_expr(reload_parents and sq_i != 0 and not held):
                    parent_step = fx.Int32(
                        _load_vec(
                            cp_i32,
                            fx.slice(parent_idx_view, (b_i, sq_i, None)),
                            1,
                            fx.Int32,
                        )
                    )
                    for vi in range_constexpr(WARP_TILE_V_ITERS):
                        gv = global_v_start + vi * WARP_GROUP_TILE_V
                        for ki in range_constexpr(WARP_TILE_K_ITERS):
                            kv = warp_k_vec_start + ki * WARP_TILE_K
                            loaded = _load_vec(
                                cp_inter_vec,
                                fx.slice(
                                    inter_view,
                                    (
                                        parent_step,
                                        hv_i,
                                        gv,
                                        kv // VALUES_PER_THREAD_K,
                                        None,
                                    ),
                                ),
                                VALUES_PER_THREAD_K,
                                inter_num,
                            )
                            if const_expr("f32" in inter_dtype):
                                state_vecs[vi * WARP_TILE_K_ITERS + ki] = loaded
                            else:
                                state_vecs[vi * WARP_TILE_K_ITERS + ki] = loaded.extf(
                                    acc_vec_t
                                )

                r_a, r_b, r_v_taps = taps
                if const_expr(sq_i + 1 < seq_length):
                    taps = _taps(sq_i + 1)

                x = r_a + r_dt_bias
                beta_x = softplus_beta_ * x

                # For beta_x > threshold, softplus(x) == x; both arms run and one is dropped.
                softplus_big = inv_softplus_beta_ * fx.math.log1p(_fast_exp(beta_x))
                softplus_x = (beta_x <= softplus_threshold_).select(softplus_big, x)

                r_g_value = -_fast_exp(r_A_log) * softplus_x
                r_beta = f32_1 / (f32_1 + _fast_exp(-r_b))
                r_g = _fast_exp(r_g_value)

                r_g_vec = fx.Vector.filled(
                    VALUES_PER_THREAD_K, fx.Float32(r_g), fx.Float32
                )

                sq_vecs = [0] * WARP_TILE_K_ITERS
                sk_vecs = [0] * WARP_TILE_K_ITERS

                scale_vec = fx.Vector.filled(VALUES_PER_THREAD_K, scale, fx.Float32)

                if const_expr(STATE_REGS >= VGPR_PER_WAVE_AT_4):
                    # Only the arithmetic is held; the mask exempts every
                    # memory class.
                    rocdl.sched_barrier(
                        "all_vmem|vmem_read|vmem_write|all_ds|ds_read|ds_write"
                    )

                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    q_vec = _load_vec(
                        cp_data_vec,
                        fx.slice(
                            q_view,
                            (
                                b_i,
                                sq_i,
                                hk_i,
                                warp_k_vec_i // VALUES_PER_THREAD_K,
                                None,
                            ),
                        ),
                        VALUES_PER_THREAD_K,
                        data_num,
                    )
                    k_vec = _load_vec(
                        cp_data_vec,
                        fx.slice(
                            k_view,
                            (
                                b_i,
                                sq_i,
                                hk_i,
                                warp_k_vec_i // VALUES_PER_THREAD_K,
                                None,
                            ),
                        ),
                        VALUES_PER_THREAD_K,
                        data_num,
                    )
                    sq_vecs[ki] = q_vec.extf(acc_vec_t)
                    sk_vecs[ki] = k_vec.extf(acc_vec_t)

                if const_expr(use_qk_l2norm):
                    sum_q_partial_vec = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    sum_k_partial_vec = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sum_q_partial_vec = (
                            sum_q_partial_vec + sq_vecs[ki] * sq_vecs[ki]
                        )
                        sum_k_partial_vec = (
                            sum_k_partial_vec + sk_vecs[ki] * sk_vecs[ki]
                        )
                    sum_q_partial = fx.Vector(sum_q_partial_vec).reduce(
                        fx.ReductionOp.ADD
                    )
                    sum_k_partial = fx.Vector(sum_k_partial_vec).reduce(
                        fx.ReductionOp.ADD
                    )
                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        sum_q_partial = sum_q_partial + sum_q_partial.shuffle_xor(
                            offset, WARP_SIZE
                        )
                        sum_k_partial = sum_k_partial + sum_k_partial.shuffle_xor(
                            offset, WARP_SIZE
                        )
                    lane0 = w_tid // WARP_THREADS_K * WARP_THREADS_K
                    local_sum_q = fx.shuffle_idx(sum_q_partial, lane0, WARP_SIZE)
                    local_sum_k = fx.shuffle_idx(sum_k_partial, lane0, WARP_SIZE)
                    inv_norm_q = fx.math.rsqrt(local_sum_q + 1e-6)
                    inv_norm_k = fx.math.rsqrt(local_sum_k + 1e-6)
                    inv_norm_q_vec = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(inv_norm_q), fx.Float32
                    )
                    inv_norm_k_vec = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(inv_norm_k), fx.Float32
                    )
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sq_vecs[ki] = sq_vecs[ki] * inv_norm_q_vec * scale_vec
                        sk_vecs[ki] = sk_vecs[ki] * inv_norm_k_vec
                else:
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        sq_vecs[ki] = sq_vecs[ki] * scale_vec

                dot_kq_vec = fx.Vector.from_elements(
                    [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)], fx.Float32
                )
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    dot_kq_vec = fx.math.fma(sk_vecs[ki], sq_vecs[ki], dot_kq_vec)
                dot_kq = dot_kq_vec.reduce(fx.ReductionOp.ADD)
                for offset in WARP_THREADS_K_SHFL_OFFSETS:
                    dot_kq = dot_kq + dot_kq.shuffle_xor(offset, WARP_SIZE)

                for vi in range_constexpr(WARP_TILE_V_ITERS):
                    global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                    r_v = r_v_taps[vi]

                    sum_hk = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )
                    sum_hq_old = fx.Vector.from_elements(
                        [f32_0 for i in range_constexpr(VALUES_PER_THREAD_K)],
                        fx.Float32,
                    )

                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] *= r_g_vec
                        h_cur = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                        sum_hk = fx.math.fma(h_cur, sk_vecs[ki], sum_hk)
                        sum_hq_old = fx.math.fma(h_cur, sq_vecs[ki], sum_hq_old)

                    sum_hk = sum_hk.reduce(fx.ReductionOp.ADD)
                    sum_hq_old = sum_hq_old.reduce(fx.ReductionOp.ADD)

                    for offset in WARP_THREADS_K_SHFL_OFFSETS:
                        sum_hk = sum_hk + sum_hk.shuffle_xor(offset, WARP_SIZE)
                        sum_hq_old = sum_hq_old + sum_hq_old.shuffle_xor(
                            offset, WARP_SIZE
                        )

                    v_new = (r_v - sum_hk) * r_beta
                    v_new = fx.shuffle_idx(
                        v_new, w_tid // WARP_THREADS_K * WARP_THREADS_K, WARP_SIZE
                    )
                    sum_hq = sum_hq_old + v_new * dot_kq
                    v_new_bcast = fx.Vector.filled(
                        VALUES_PER_THREAD_K, fx.Float32(v_new), fx.Float32
                    )

                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        h_new = fx.math.fma(
                            sk_vecs[ki],
                            v_new_bcast,
                            state_vecs[vi * WARP_TILE_K_ITERS + ki],
                        )
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] = h_new

                    sum_hq = sum_hq.to(data_num)

                    # Closure keeps the store opaque to the runtime-if state analysis.
                    def _write_q(_sum_hq=sum_hq, _gv=global_v_i, _sq=sq_i):
                        _store_vec(
                            cp_data,
                            fx.slice(out_view, (b_i, _sq, hv_i, _gv, None)),
                            _sum_hq,
                            1,
                            data_num,
                        )

                    if warp_k_vec_start == 0:
                        _write_q()

                if const_expr(CHAIN):
                    write_slot = token_slots[sq_i]
                    write_view = _state_at(write_slot)

                    def _checkpoint(_view=write_view):
                        for vi in range_constexpr(WARP_TILE_V_ITERS):
                            gv = global_v_start + vi * WARP_GROUP_TILE_V
                            for ki in range_constexpr(WARP_TILE_K_ITERS):
                                kv = warp_k_vec_start + ki * WARP_TILE_K
                                acc = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                                if const_expr("f32" in state_dtype):
                                    out_vec = acc
                                else:
                                    out_vec = acc.truncf(state_vec_t)
                                _store_vec(
                                    cp_state_vec,
                                    fx.slice(
                                        _view,
                                        (hv_i, gv, kv // VALUES_PER_THREAD_K, None),
                                    ),
                                    out_vec,
                                    VALUES_PER_THREAD_K,
                                    state_num,
                                )

                    if write_slot >= min_live_slot:
                        _checkpoint()

                if const_expr(snapshot != "no"):
                    # The tree rereads a step when it walks to a child; nothing
                    # descends from the last.
                    keeps_reading = reload_parents and sq_i != seq_length - 1
                    snap_atom = cp_inter_vec if keeps_reading else cp_inter_vec_nt
                    inter_vec_t = T.vec(
                        VALUES_PER_THREAD_K, get_dtype_in_kernel(inter_dtype)
                    )

                    def _snapshot(_step=sq_i, _vec_t=inter_vec_t, _atom=snap_atom):
                        for vi in range_constexpr(WARP_TILE_V_ITERS):
                            gv = global_v_start + vi * WARP_GROUP_TILE_V
                            for ki in range_constexpr(WARP_TILE_K_ITERS):
                                kv = warp_k_vec_start + ki * WARP_TILE_K
                                acc = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                                if const_expr("f32" in inter_dtype):
                                    out_vec = acc
                                else:
                                    out_vec = acc.truncf(_vec_t)
                                _store_vec(
                                    _atom,
                                    fx.slice(
                                        inter_view,
                                        (
                                            _step,
                                            hv_i,
                                            gv,
                                            kv // VALUES_PER_THREAD_K,
                                            None,
                                        ),
                                    ),
                                    out_vec,
                                    VALUES_PER_THREAD_K,
                                    inter_num,
                                )

                    if const_expr(snapshot == "always"):
                        _snapshot()
                    else:
                        if cache_idx >= 0:
                            _snapshot()

            # The chain's last token already checkpointed into its own slot.
            # Snapshot mode writes back into the slot it read, so the read's
            # descriptor already addresses it.
            if const_expr(SNAPSHOT and not NO_STATE_WRITE):
                write_view = read_state_view
                for vi in range_constexpr(WARP_TILE_V_ITERS):
                    global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                        acc = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                        if const_expr("f32" in state_dtype):
                            out_vec = acc
                        else:
                            out_vec = acc.truncf(state_vec_t)
                        _store_vec(
                            cp_state_vec,
                            fx.slice(
                                write_view,
                                (
                                    hv_i,
                                    global_v_i,
                                    warp_k_vec_i // VALUES_PER_THREAD_K,
                                    None,
                                ),
                            ),
                            out_vec,
                            VALUES_PER_THREAD_K,
                            state_num,
                        )

        # Flat rather than nested, so no scf.if carries a value out of itself.
        if const_expr(TREE):
            if (read_slot >= min_live_slot) & (cache_idx >= 0):
                _do_mtp(reload_parents=True, snapshot="always")
            if (read_slot >= min_live_slot) & (cache_idx < 0):
                _do_mtp(reload_parents=False, snapshot="no")
        else:
            if read_slot >= min_live_slot:
                _do_mtp(
                    reload_parents=False,
                    snapshot="guarded" if SAVE_INTER else "no",
                )

    @flyc.jit
    def launch_gdr_mtp_kernel(
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        a: fx.Tensor,
        b: fx.Tensor,
        dt_bias: fx.Tensor,
        A_log: fx.Tensor,
        state_indices: fx.Tensor,
        num_accepted: fx.Tensor,
        inter_indices: fx.Tensor,
        parent_tokens: fx.Tensor,
        state: fx.Tensor,
        inter_buffer: fx.Tensor,
        out: fx.Tensor,
        batch_size: fx.Int32,
        stream: fx.Stream,
    ):
        gx = batch_size * num_v_heads * NUM_BLOCKS_PER_V_DIM
        gdr_mtp_kernel._func.__name__ = KERNEL_NAME
        gdr_mtp_kernel(
            query,
            key,
            value,
            a,
            b,
            dt_bias,
            A_log,
            state_indices,
            num_accepted,
            inter_indices,
            parent_tokens,
            state,
            inter_buffer,
            out,
            batch_size,
        ).launch(grid=(gx, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    return launch_gdr_mtp_kernel
