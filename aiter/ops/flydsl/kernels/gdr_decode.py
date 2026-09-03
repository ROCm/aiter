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
    GTensor,
    _to_raw,
    get_dtype_bytes,
    get_dtype_in_kernel,
)


def _gview(tensor, base, shape, stride):
    it = fx.get_iter(fx.rocdl.make_buffer_tensor(tensor, max_size=True))
    if base is not None:
        it = fx.add_offset(it, base)
    return fx.Tensor(fx.make_view(it, fx.make_layout(shape, stride)))


def _load_vec(atom, tile, width, numeric):
    frag = fx.make_rmem_tensor(width, numeric)
    fx.copy(atom, tile, frag)
    vec = frag.load()
    return vec[0] if width == 1 else vec


def _store_vec(atom, tile, value, width, numeric):
    frag = fx.make_rmem_tensor(width, numeric)
    frag.store(fx.Vector.from_elements([value], dtype=numeric) if width == 1 else value)
    fx.copy(atom, frag, tile)


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

        def fast_exp(x, use_exp2=True):
            if const_expr(use_exp2):
                log2e = 1.4426950408889634
                return rocdl.exp2(T.f32, _to_raw(fx.Float32(x) * log2e))
            return fx.math.exp(x, fastmath=fx.FastMathFlags.fast)

        def fast_log1p(x):
            return fx.math.log1p(x, fastmath=fx.FastMathFlags.fast)

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

                # softplus with the large-x identity: for beta_x > threshold,
                # softplus(x) == x. select computes both arms (the overflow arm
                # is discarded) -> bit-identical to the old branch.
                softplus_big = (f32_1 / softplus_beta_) * fast_log1p(fast_exp(beta_x))
                softplus_x = (
                    fx.Float32(beta_x) <= fx.Float32(softplus_threshold_)
                ).select(softplus_big, x)

                r_g_value = -fast_exp(r_A_log) * softplus_x
                r_beta = f32_1 / (f32_1 + fast_exp(-r_b))
                r_g = fast_exp(r_g_value)

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

        if (read_pool_idx >= 0) & (write_pool_idx >= 0):
            _do_decode()

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
):
    """Gated delta rule over a speculative draft window.

    Verify has to be able to undo the tokens the target model rejects, so it
    needs both a rollback point and a per-token record. The two upstreams keep
    that record in different places, and ``mode`` picks between them.

    ``MTP_MODE_CHAIN`` is vLLM's. The draft is a linear chain, the rollback
    point is ``state_indices[n, num_accepted - 1]``, and the record is the state
    pool itself: every token checkpoints into ``state_indices[n, t]``. The last
    token's checkpoint is the final store, so there is no separate one.

    ``MTP_MODE_SNAPSHOT`` is SGLang's. The rollback point is the sequence's one
    slot ``state_indices[n]`` and the record lives in
    ``intermediate_states_buffer``. With ``has_tree`` the draft is an EAGLE tree,
    so each token restarts from its parent's snapshot rather than from its
    predecessor; ``disable_state_update`` suppresses the write-back, which is
    how a verify pass leaves the committed state alone.

    Buffer offsets are 32 bits, so any term scaling with the state pool or the
    snapshot buffer belongs in the 64-bit descriptor base, with both factors
    widened before the multiply.
    """
    assert mode in (MTP_MODE_CHAIN, MTP_MODE_SNAPSHOT), f"unknown MTP mode {mode!r}"
    CHAIN = mode == MTP_MODE_CHAIN
    SNAPSHOT = not CHAIN
    TREE = bool(has_tree)
    NO_STATE_WRITE = bool(disable_state_update)
    assert not TREE or SNAPSHOT, "the EAGLE tree is the snapshot mode's"
    assert not TREE or len(inter_strides) == 5, "tree needs a snapshot buffer"
    SAVE_INTER = SNAPSHOT and len(inter_strides) == 5

    # A snapshot read back returns the accumulator that was written only at f32.
    # At a narrower dtype the reload is a rounding, so the tree cannot skip it.
    LOSSLESS_SNAPSHOT = TREE and "f32" in inter_dtype

    SCALE_VALUE = float(1.0 / (float(head_k_dim) ** 0.5))
    WARP_THREADS_V = 64 // WARP_THREADS_K

    if "f32" in state_dtype:
        VALUES_PER_THREAD_K = 4  # 16B
    else:
        VALUES_PER_THREAD_K = 8  # 16B

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

    # Registers the resident state occupies. Every other live value in the token
    # body is a scalar or a single K vector.
    STATE_REGS = WARP_TILE_V_ITERS * WARP_TILE_K_ITERS * VALUES_PER_THREAD_K

    # A wave64 SIMD has a 512-register vector file, so four resident waves is
    # 128 each. A tiling whose state alone reaches that leaves nothing for
    # scheduling freedom.
    VGPR_PER_WAVE_AT_4 = 512 // 4

    WARP_THREADS_K_SHFL_OFFSETS = []
    offsets_ = WARP_THREADS_K // 2
    while offsets_ >= 1:
        WARP_THREADS_K_SHFL_OFFSETS.append(int(offsets_))
        offsets_ /= 2
    WARP_THREADS_K_SHFL_OFFSETS = WARP_THREADS_K_SHFL_OFFSETS[::-1]

    STATE_BYTES = get_dtype_bytes(state_dtype)
    INTER_BYTES = get_dtype_bytes(inter_dtype) if SAVE_INTER else 0

    # The snapshot reuses the state's lane count, which is picked so the state
    # vector is 16 bytes. A wider snapshot element overruns what one buffer op
    # carries, and this is the layer that cannot express the store.
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
        # Reciprocal of a builder constant, so the token body holds a multiply
        # and not a divide. At the default beta of 1.0 it folds away entirely.
        inv_softplus_beta_ = fx.Float32(1.0 / softplus_beta)
        softplus_threshold_ = fx.Float32(softplus_threshold)

        dtype_ = get_dtype_in_kernel(dtype)
        fx_dtype_ = fx.BFloat16 if dtype == "bf16" else fx.Float16
        A_log_dtype_ = get_dtype_in_kernel(A_log_dtype)
        state_dtype_ = get_dtype_in_kernel(state_dtype)
        f32_0 = fx.Float32(0.0)
        f32_1 = fx.Float32(1.0)
        state_vec_t = T.vec(VALUES_PER_THREAD_K, state_dtype_)
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

        # Flat views: the index tensors are addressed with the caller's strides
        # rather than a shape, so a 1-D [B] and a 2-D [B, T] map the same way.
        si_tensor = GTensor(state_indices, dtype=T.i32, shape=(-1,))

        # Every token's checkpoint slot, read together in the prologue that is
        # already waiting on the rollback lookup, so one wait covers all of them
        # at a cost of `seq_length` registers. The snapshot contract has one
        # slot per sequence, so there is nothing to spread.
        if const_expr(CHAIN):
            token_slots = [
                fx.Int32(si_tensor[b_i * si_strides[0] + t * si_strides[1]])
                for t in range_constexpr(seq_length)
            ]

        # The two contracts spell a dead slot differently: SGLang pads with a
        # negative sentinel and slot 0 is ordinary, while vLLM reserves slot 0
        # as its null block and aiter's Triton passes a negative sentinel
        # through the same entry point. So the chain rejects both and the
        # snapshot mode must not reject slot 0.
        MIN_LIVE_SLOT = 1 if CHAIN else 0

        # Rollback point. The chain rolls back to the last accepted token's
        # checkpoint; the snapshot mode's sequence has a single slot and rolls
        # back through the snapshot buffer instead.
        if const_expr(CHAIN):
            nacc_tensor = GTensor(num_accepted, dtype=T.i32, shape=(-1,))
            read_token = fx.Int32(nacc_tensor[b_i]) - fx.Int32(1)
            read_slot = fx.Int32(
                si_tensor[b_i * si_strides[0] + read_token * si_strides[1]]
            )
        else:
            read_slot = fx.Int32(si_tensor[b_i * si_strides[0]])

        if const_expr(SAVE_INTER):
            isi_tensor = GTensor(inter_indices, dtype=T.i32, shape=(-1,))
            cache_idx = fx.Int32(isi_tensor[b_i])
        if const_expr(TREE):
            parent_tensor = GTensor(parent_tokens, dtype=T.i32, shape=(-1,))

        q_tensor = GTensor(
            query,
            dtype=dtype_,
            shape=(-1, seq_length, num_k_heads, head_k_dim),
            stride=q_strides,
        )
        k_tensor = GTensor(
            key,
            dtype=dtype_,
            shape=(-1, seq_length, num_k_heads, head_k_dim),
            stride=k_strides,
        )
        v_tensor = GTensor(
            value,
            dtype=dtype_,
            shape=(-1, seq_length, num_v_heads, head_v_dim),
            stride=v_strides,
        )
        a_tensor = GTensor(
            a,
            dtype=dtype_,
            stride=(a_strides[0], a_strides[1], a_strides[2]),
            shape=(-1, seq_length, num_v_heads),
        )
        b_tensor = GTensor(
            b,
            dtype=dtype_,
            stride=(b_strides[0], b_strides[1], b_strides[2]),
            shape=(-1, seq_length, num_v_heads),
        )
        dt_bias_tensor = GTensor(dt_bias, dtype=dtype_, shape=(num_v_heads,))
        A_log_tensor = GTensor(A_log, dtype=A_log_dtype_, shape=(num_v_heads,))
        out_tensor = GTensor(
            out, dtype=dtype_, shape=(-1, seq_length, num_v_heads, head_v_dim)
        )

        def _state_at(slot):
            """State-pool view whose descriptor base already carries the slot.

            ``slot * state_strides[0]`` scales with the pool, so it cannot sit
            in the 32-bit buffer offset. Both factors are widened first.
            """
            return GTensor(
                state,
                dtype=state_dtype_,
                shape=(num_v_heads, head_v_dim, head_k_dim),
                stride=(state_strides[1], state_strides[2], state_strides[3]),
                static_bytes_offset_i64=fx.Int64(slot)
                * fx.Int64(state_strides[0])
                * STATE_BYTES,
            )

        def _inter_at(slot, step):
            """Snapshot view for one (sequence slot, draft step).

            A slot here spans ``cache_steps`` states where a pool slot spans
            one, so it crosses 2^31 elements that much sooner. Slot and step go
            into the 64-bit base for the same reason the pool's does.
            """
            inter_dtype_ = get_dtype_in_kernel(inter_dtype)
            return GTensor(
                inter_buffer,
                dtype=inter_dtype_,
                shape=(num_v_heads, head_v_dim, head_k_dim),
                stride=(inter_strides[2], inter_strides[3], inter_strides[4]),
                static_bytes_offset_i64=(
                    fx.Int64(slot) * fx.Int64(inter_strides[0])
                    + fx.Int64(step) * fx.Int64(inter_strides[1])
                )
                * INTER_BYTES,
            )

        def fast_exp(x, use_exp2=True):
            if const_expr(use_exp2):
                log2e = 1.4426950408889634
                return rocdl.exp2(T.f32, _to_raw(fx.Float32(x) * log2e))
            return fx.math.exp(x, fastmath=fx.FastMathFlags.fast)

        def fast_log1p(x):
            return fx.math.log1p(x, fastmath=fx.FastMathFlags.fast)

        def fast_rsqrt(x):
            """Hardware reciprocal square root, good to about one ULP.

            That is well inside a body which already takes exp as exp2 and
            log1p under fast math, and far inside a bf16 output.
            """
            return rocdl.rsq(T.f32, _to_raw(fx.Float32(x)))

        def fast_rcp(x):
            """Hardware reciprocal, at the accuracy of `fast_rsqrt`.

            The sigmoid gate's denominator is itself an exp2, so a correctly
            rounded divide on top of it is finer than the input supports.
            """
            return rocdl.rcp(T.f32, _to_raw(fx.Float32(x)))

        # The inputs that do not depend on the rollback slot are issued ahead of
        # the dead-slot test, so their latency runs under the lookup's:
        # `read_slot` is two dependent round trips deep -- `num_accepted[b_i]`
        # names the token, that token's entry names the slot -- while the decay
        # scalars and token 0's taps are addressed by `b_i` and `hv_i` alone.
        #
        # A dead sequence issues these reads and discards them. They are in
        # bounds either way, which is why the slot loads above are unguarded.
        #
        # Chain contracts only: the tree emits the body once per snapshot arm,
        # so a hoisted value is live across both copies.
        HOIST_ENTRY = not TREE

        def _taps(sq):
            """The gate and value scalars one token reads, issued together."""
            return (
                a_tensor[b_i, sq, hv_i],
                b_tensor[b_i, sq, hv_i],
                [
                    v_tensor[b_i, sq, hv_i, global_v_start + vi * WARP_GROUP_TILE_V]
                    for vi in range_constexpr(WARP_TILE_V_ITERS)
                ],
            )

        if const_expr(HOIST_ENTRY):
            if const_expr("f32" in A_log_dtype):
                entry_A_log = A_log_tensor[hv_i]
            else:
                entry_A_log = A_log_tensor[hv_i].extf(T.f32)
            entry_dt_bias = dt_bias_tensor[hv_i].extf(T.f32)
            entry_taps = _taps(0)

        # Skip pad slots (a negative sentinel). The guarded body is a closure so
        # the runtime `if` sees an opaque call rather than a GTensor to thread
        # through an scf.if yield.
        #
        # ``reload_parents`` and ``snapshot`` are traced flags, not runtime
        # ones: the parent reload produces the running state, and a value
        # defined inside an scf.if does not dominate its use after it.
        # ``cache_idx`` is fixed for the whole kernel, so the test sits at the
        # entry and neither traced body carries a per-token snapshot guard.
        def _do_mtp(reload_parents=False, snapshot="no"):
            if const_expr(not HOIST_ENTRY):
                if const_expr("f32" in A_log_dtype):
                    r_A_log = A_log_tensor[hv_i]
                else:
                    r_A_log = A_log_tensor[hv_i].extf(T.f32)
                r_dt_bias = dt_bias_tensor[hv_i].extf(T.f32)
            else:
                r_A_log = entry_A_log
                r_dt_bias = entry_dt_bias

            read_state_tensor = _state_at(read_slot)
            state_vecs = [0] * (WARP_TILE_V_ITERS * WARP_TILE_K_ITERS)
            for vi in range_constexpr(WARP_TILE_V_ITERS):
                global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    state_vecs[vi * WARP_TILE_K_ITERS + ki] = (
                        read_state_tensor.vec_load(
                            (hv_i, global_v_i, warp_k_vec_i), VALUES_PER_THREAD_K
                        )
                    )
                    if const_expr("f32" in state_dtype):
                        pass
                    else:
                        state_vecs[vi * WARP_TILE_K_ITERS + ki] = state_vecs[
                            vi * WARP_TILE_K_ITERS + ki
                        ].extf(acc_vec_t)

            # Each token issues the next token's reads before its own work. A
            # token ends by storing its state, and nothing tells the compiler
            # that store cannot land on the gate or value inputs, so a read
            # placed where it is used cannot be hoisted over it. Taking the
            # value reads as a group lets one wait cover the whole v loop.
            #
            # Depth stays at one: depth d costs
            # `d * (2 + WARP_TILE_V_ITERS)` registers, and one token of work
            # already covers the round trip.
            taps = entry_taps if const_expr(HOIST_ENTRY) else _taps(0)

            for sq_i in range_constexpr(seq_length):
                # EAGLE tree: restart from the parent token's snapshot. Token 0
                # has no parent and keeps the rollback state loaded above.
                #
                # Token 1's parent needs no read: token 0 is the only token
                # before it and that state is still in registers, so the
                # snapshot round trip is skipped where it returns the
                # accumulator exactly (LOSSLESS_SNAPSHOT). Any later token can
                # have any earlier parent, which would take a runtime test
                # carrying the whole state out through an scf.if yield.
                held = LOSSLESS_SNAPSHOT and sq_i == 1
                if const_expr(reload_parents and sq_i != 0 and not held):
                    parent_step = fx.Int32(
                        parent_tensor[
                            b_i * parent_strides[0] + sq_i * parent_strides[1]
                        ]
                    )
                    parent_view = _inter_at(cache_idx, parent_step)
                    for vi in range_constexpr(WARP_TILE_V_ITERS):
                        gv = global_v_start + vi * WARP_GROUP_TILE_V
                        for ki in range_constexpr(WARP_TILE_K_ITERS):
                            kv = warp_k_vec_start + ki * WARP_TILE_K
                            loaded = parent_view.vec_load(
                                (hv_i, gv, kv), VALUES_PER_THREAD_K
                            )
                            if const_expr("f32" in inter_dtype):
                                state_vecs[vi * WARP_TILE_K_ITERS + ki] = loaded
                            else:
                                state_vecs[vi * WARP_TILE_K_ITERS + ki] = loaded.extf(
                                    acc_vec_t
                                )

                tap_a, tap_b, r_v_raw = taps
                if const_expr(sq_i + 1 < seq_length):
                    taps = _taps(sq_i + 1)

                r_a = tap_a.extf(T.f32)
                r_b = tap_b.extf(T.f32)
                x = r_a + r_dt_bias
                beta_x = softplus_beta_ * x

                # softplus with the large-x identity: for beta_x > threshold,
                # softplus(x) == x. Both arms are computed and the overflowing
                # one discarded.
                softplus_big = inv_softplus_beta_ * fast_log1p(fast_exp(beta_x))
                softplus_x = (beta_x <= softplus_threshold_).select(softplus_big, x)

                r_g_value = -fast_exp(r_A_log) * softplus_x
                r_beta = fast_rcp(f32_1 + fast_exp(-r_b))
                r_g = fast_exp(r_g_value)

                r_g_vec = fx.Vector.filled(
                    VALUES_PER_THREAD_K, fx.Float32(r_g), fx.Float32
                )

                sq_vecs = [0] * WARP_TILE_K_ITERS
                sk_vecs = [0] * WARP_TILE_K_ITERS

                scale_vec = fx.Vector.filled(VALUES_PER_THREAD_K, scale, fx.Float32)

                if const_expr(STATE_REGS >= VGPR_PER_WAVE_AT_4):
                    # Only where the state already fills the register file: the
                    # gate arithmetic and the L2-norm reduction are kept from
                    # interleaving, whose register cost decides whether a
                    # fourth wave stays resident. The mask exempts every memory
                    # class, so the load pipeline that limits this kernel keeps
                    # its freedom and only the arithmetic is held.
                    rocdl.sched_barrier(
                        "all_vmem|vmem_read|vmem_write|all_ds|ds_read|ds_write"
                    )

                for ki in range_constexpr(WARP_TILE_K_ITERS):
                    warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                    q_vec = q_tensor.vec_load(
                        (b_i, sq_i, hk_i, warp_k_vec_i), VALUES_PER_THREAD_K
                    )
                    k_vec = k_tensor.vec_load(
                        (b_i, sq_i, hk_i, warp_k_vec_i), VALUES_PER_THREAD_K
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
                    inv_norm_q = fast_rsqrt(local_sum_q + 1e-6)
                    inv_norm_k = fast_rsqrt(local_sum_k + 1e-6)
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
                    r_v = r_v_raw[vi].extf(T.f32)

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

                    sum_hq = sum_hq.to(fx_dtype_)

                    # Only k-vec lane 0 writes the q output; closure keeps the
                    # GTensor store opaque to the runtime-if state analysis.
                    def _write_q(_sum_hq=sum_hq, _gv=global_v_i, _sq=sq_i):
                        out_tensor[b_i, _sq, hv_i, _gv] = _sum_hq

                    if warp_k_vec_start == 0:
                        _write_q()

                # Per-token record: without it a later rejection has nothing to
                # roll back to.
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
                                _view.vec_store(
                                    (hv_i, gv, kv), out_vec, VALUES_PER_THREAD_K
                                )

                    if write_slot >= MIN_LIVE_SLOT:
                        _checkpoint()

                if const_expr(snapshot != "no"):
                    snap_view = _inter_at(cache_idx, sq_i)
                    inter_vec_t = T.vec(
                        VALUES_PER_THREAD_K, get_dtype_in_kernel(inter_dtype)
                    )

                    def _snapshot(_view=snap_view, _vec_t=inter_vec_t):
                        for vi in range_constexpr(WARP_TILE_V_ITERS):
                            gv = global_v_start + vi * WARP_GROUP_TILE_V
                            for ki in range_constexpr(WARP_TILE_K_ITERS):
                                kv = warp_k_vec_start + ki * WARP_TILE_K
                                acc = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                                if const_expr("f32" in inter_dtype):
                                    out_vec = acc
                                else:
                                    out_vec = acc.truncf(_vec_t)
                                _view.vec_store(
                                    (hv_i, gv, kv), out_vec, VALUES_PER_THREAD_K
                                )

                    if const_expr(snapshot == "always"):
                        _snapshot()
                    else:
                        if cache_idx >= 0:
                            _snapshot()

            # The chain's last token already checkpointed into its own slot, so
            # only the snapshot mode has a final store left, and a verify pass
            # asks for it to be suppressed.
            if const_expr(SNAPSHOT and not NO_STATE_WRITE):
                write_view = _state_at(read_slot)
                for vi in range_constexpr(WARP_TILE_V_ITERS):
                    global_v_i = global_v_start + vi * WARP_GROUP_TILE_V
                    for ki in range_constexpr(WARP_TILE_K_ITERS):
                        warp_k_vec_i = warp_k_vec_start + ki * WARP_TILE_K
                        acc = state_vecs[vi * WARP_TILE_K_ITERS + ki]
                        if const_expr("f32" in state_dtype):
                            out_vec = acc
                        else:
                            out_vec = acc.truncf(state_vec_t)
                        write_view.vec_store(
                            (hv_i, global_v_i, warp_k_vec_i),
                            out_vec,
                            VALUES_PER_THREAD_K,
                        )

        # One entry test per traced body, flat rather than nested, so no scf.if
        # has to carry a value out of itself. The tree's two arms are
        # complementary: with a snapshot slot a sequence reloads its parents,
        # without one it has nothing to reload and nothing to record.
        if const_expr(TREE):
            if (read_slot >= MIN_LIVE_SLOT) & (cache_idx >= 0):
                _do_mtp(reload_parents=True, snapshot="always")
            if (read_slot >= MIN_LIVE_SLOT) & (cache_idx < 0):
                _do_mtp(reload_parents=False, snapshot="no")
        else:
            if read_slot >= MIN_LIVE_SLOT:
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
