# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL fused KDA decode kernels on gfx950 with 12 heads and dim 128."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T

from .kernels_common import LOG2E
from .tensor_shim import GTensor

_HEADS = 12
_DIM = 128
_SCALE = _DIM**-0.5
_BLOCK_THREADS = 256
_NUM_WARPS = 4
_WARP_SIZE = 64
_WARP_THREADS_K = 8
_VALUES_PER_THREAD_K = 4
_WARP_TILE_K = _WARP_THREADS_K * _VALUES_PER_THREAD_K
_K_ITERS = _DIM // _WARP_TILE_K
_WARP_THREADS_V = _WARP_SIZE // _WARP_THREADS_K
_V_GROUP_TILE = _NUM_WARPS * _WARP_THREADS_V
_V_ITERS = _DIM // _V_GROUP_TILE
_K_SHUFFLE_OFFSETS = (1, 2, 4)
_PROJECTION_VECTOR = 4
_PROJECTION_ITERS = _DIM // _PROJECTION_VECTOR
_SPECIALIZATION_CACHE_SIZE = 8


def _one_plus_exp_neg(x):
    return fx.Float32(1.0) + fx.math.exp2(-x * fx.Float32(LOG2E))


def _sigmoid(x, scale=1.0):
    return fx.Float32(scale) / _one_plus_exp_neg(x)


def _k_group_sum(value):
    """Sum a value across the eight lanes that split one K row."""
    for offset in _K_SHUFFLE_OFFSETS:
        value = value + value.shuffle_xor(offset, _WARP_SIZE)
    return value


def _vector_sum(vec):
    return fx.Vector(vec).reduce(fx.ReductionOp.ADD)


# Each entry holds a JIT launcher and compiled module; keep only a few
# (norm_eps, lower_bound, fuse_f_b) specializations alive.
@functools.lru_cache(maxsize=_SPECIALIZATION_CACHE_SIZE)
def create_kda_decode_kernel(norm_eps: float, lower_bound: float, fuse_f_b: bool):
    """Build the fixed gfx950 BF16 KDA decode specialization.

    With ``fuse_f_b``, the kernel computes the head-local ``f_b`` projection
    instead of reading a precomputed raw gate from global memory.
    """

    @fx.struct
    class SharedStorage:
        q: fx.Array[fx.BFloat16, _DIM, 16]
        k: fx.Array[fx.BFloat16, _DIM, 16]
        v: fx.Array[fx.BFloat16, _DIM, 16]
        if fuse_f_b:
            gate: fx.Array[fx.BFloat16, _DIM, 16]
        recurrent_out: fx.Array[fx.BFloat16, _DIM, 16]
        norm_partial: fx.Array[fx.Float32, 2, 16]

    @flyc.jit
    def decode_body(
        gate_inputs,
        x_mem,
        weight_mem,
        conv_state_mem,
        raw_beta_mem,
        A_log_mem,
        dt_bias_mem,
        state_mem,
        state_indices_mem,
        output_gate_mem,
        norm_weight_mem,
        out_mem,
        num_cache_slots,
        stride_x_token,
        stride_weight_channel,
        stride_weight_width,
        stride_conv_slot,
        stride_conv_channel,
        stride_conv_width,
        stride_beta_token,
        stride_state_slot,
        stride_gate_token,
        stride_gate_head,
        stride_out_token,
        stride_out_head,
    ):
        x = GTensor(x_mem, dtype=T.bf16, shape=(-1,))
        weight = GTensor(weight_mem, dtype=T.f32, shape=(-1,))
        raw_beta = GTensor(raw_beta_mem, dtype=T.bf16, shape=(-1,))
        A_log = GTensor(A_log_mem, dtype=T.f32, shape=(-1,))
        dt_bias = GTensor(dt_bias_mem, dtype=T.f32, shape=(-1,))
        state_indices = GTensor(state_indices_mem, dtype=T.i32, shape=(-1,))
        output_gate = GTensor(output_gate_mem, dtype=T.bf16, shape=(-1,))
        norm_weight = GTensor(norm_weight_mem, dtype=T.bf16, shape=(-1,))
        out = GTensor(out_mem, dtype=T.bf16, shape=(-1,))

        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        q_lds = shared.q.ptr
        k_lds = shared.k.ptr
        v_lds = shared.v.ptr
        gate_lds = shared.gate.ptr if fuse_f_b else None
        out_lds = shared.recurrent_out.ptr
        norm_lds = shared.norm_partial.ptr

        tid = fx.thread_idx.x
        block = fx.block_idx.x
        batch = block // _HEADS
        head = block % _HEADS
        lane = tid % _WARP_SIZE
        warp = tid // _WARP_SIZE
        lane_k = lane % _WARP_THREADS_K
        subgroup_leader = lane // _WARP_THREADS_K * _WARP_THREADS_K
        out_base = batch * stride_out_token + head * stride_out_head

        state_idx = fx.Int32(state_indices[batch])
        # Slot zero and negative entries are padding. Slots past the shared
        # capacity of both caches are treated as padding instead of being
        # dereferenced out of bounds.
        valid = (state_idx > 0) & (state_idx < num_cache_slots)
        # Rebase both caches on the selected slot in 64-bit. Buffer accesses
        # take a 32-bit byte offset, so slot * stride must not reach them.
        state_slot = state_idx.to(fx.Int64)
        conv_state = GTensor(
            conv_state_mem,
            dtype=T.bf16,
            shape=(-1,),
            static_bytes_offset_i64=state_slot * stride_conv_slot * fx.Int64(2),
        )
        state = GTensor(
            state_mem,
            dtype=T.f32,
            shape=(-1,),
            static_bytes_offset_i64=state_slot * stride_state_slot * fx.Int64(4),
        )

        if const_expr(fuse_f_b):
            (
                f_a_mem,
                f_b_weight_mem,
                stride_f_a_token,
                stride_f_b_head,
                stride_f_b_output,
            ) = gate_inputs
            f_a = GTensor(f_a_mem, dtype=T.bf16, shape=(-1,))
            f_b_weight = GTensor(f_b_weight_mem, dtype=T.bf16, shape=(-1,))
        else:
            raw_g_mem, stride_g_token = gate_inputs
            raw_g = GTensor(raw_g_mem, dtype=T.bf16, shape=(-1,))

        # The runtime-if rewriter carries every object whose method is called
        # in a branch, which tensor views cannot be. Each phase is therefore a
        # closure; the dispatch below calls them under the runtime guards.
        def project_gate():
            # Threads 0..127 own one output each. Accumulation is FP32 and the
            # single BF16 store is the same numerical boundary as F.linear.
            vec_f32_projection = T.vec(_PROJECTION_VECTOR, T.f32)
            accum = fx.Vector.filled(_PROJECTION_VECTOR, 0.0, fx.Float32)
            f_a_base = batch * stride_f_a_token
            f_b_base = head * stride_f_b_head + tid * stride_f_b_output
            for projection_iter in range_constexpr(_PROJECTION_ITERS):
                projection_offset = projection_iter * _PROJECTION_VECTOR
                f_a_values = f_a.vec_load(
                    (f_a_base + projection_offset,),
                    _PROJECTION_VECTOR,
                ).extf(vec_f32_projection)
                weight_values = f_b_weight.vec_load(
                    (f_b_base + projection_offset,),
                    _PROJECTION_VECTOR,
                ).extf(vec_f32_projection)
                accum = fx.math.fma(f_a_values, weight_values, accum)
            fx.ptr_store(fx.BFloat16(_vector_sum(accum)), gate_lds + tid)

        def convolve_channel(channel):
            cs_base = channel * stride_conv_channel
            c0 = fx.Float32(conv_state[cs_base])
            c1 = fx.Float32(conv_state[cs_base + stride_conv_width])
            c2 = fx.Float32(conv_state[cs_base + 2 * stride_conv_width])
            current = fx.BFloat16(x[batch * stride_x_token + channel])
            w_base = channel * stride_weight_channel
            acc = c0 * fx.Float32(weight[w_base])
            acc = acc + c1 * fx.Float32(weight[w_base + stride_weight_width])
            acc = acc + c2 * fx.Float32(weight[w_base + 2 * stride_weight_width])
            acc = acc + fx.Float32(current) * fx.Float32(
                weight[w_base + 3 * stride_weight_width]
            )
            conv_state.store(cs_base, fx.BFloat16(c1))
            conv_state.store(cs_base + stride_conv_width, fx.BFloat16(c2))
            conv_state.store(cs_base + 2 * stride_conv_width, current)
            # SiLU as a single division matches the reference rounding.
            return (acc / _one_plus_exp_neg(acc)).to(fx.BFloat16)

        def convolve_qkv():
            # A workgroup exclusively owns all three convolution channels for
            # its (batch, head), so every cache entry is shifted exactly once.
            head_channel = head * _DIM + tid
            fx.ptr_store(convolve_channel(head_channel), q_lds + tid)
            fx.ptr_store(convolve_channel(_HEADS * _DIM + head_channel), k_lds + tid)
            fx.ptr_store(
                convolve_channel(2 * _HEADS * _DIM + head_channel), v_lds + tid
            )

        def recurrent_update():
            # Four waves split V into 32-row groups. Eight-lane subgroups
            # reduce K; each lane issues one aligned f32x4 state transaction.
            k_vec_start = lane_k * _VALUES_PER_THREAD_K
            global_v_start = warp * _WARP_THREADS_V + lane // _WARP_THREADS_K
            vec_f32 = T.vec(_VALUES_PER_THREAD_K, T.f32)
            vec_bf16 = T.vec(_VALUES_PER_THREAD_K, T.bf16)
            zero_vec = fx.Vector.filled(_VALUES_PER_THREAD_K, 0.0, fx.Float32)

            q_vecs = []
            k_vecs = []
            decay_vecs = []
            sum_q_partial = fx.Float32(0.0)
            sum_k_partial = fx.Float32(0.0)
            a = fx.math.exp2(fx.Float32(A_log[head]) * fx.Float32(LOG2E))

            for ki in range_constexpr(_K_ITERS):
                k_base = k_vec_start + ki * _WARP_TILE_K
                q_f32 = fx.ptr_load(q_lds + k_base, result_type=vec_bf16).extf(vec_f32)
                k_f32 = fx.ptr_load(k_lds + k_base, result_type=vec_bf16).extf(vec_f32)
                q_vecs.append(q_f32)
                k_vecs.append(k_f32)
                sum_q_partial = sum_q_partial + _vector_sum(q_f32 * q_f32)
                sum_k_partial = sum_k_partial + _vector_sum(k_f32 * k_f32)

                if const_expr(fuse_f_b):
                    # The projection is rounded in LDS before the lower-bound gate.
                    gate_bf16 = fx.ptr_load(gate_lds + k_base, result_type=vec_bf16)
                else:
                    gate_bf16 = raw_g.vec_load(
                        (batch * stride_g_token + head * _DIM + k_base,),
                        _VALUES_PER_THREAD_K,
                    )
                dt = dt_bias.vec_load((head * _DIM + k_base,), _VALUES_PER_THREAD_K)
                gate = _sigmoid((gate_bf16.extf(vec_f32) + dt) * a, lower_bound)
                decay_vecs.append(fx.math.exp2(gate * fx.Float32(LOG2E)))

            sum_q_partial = _k_group_sum(sum_q_partial)
            sum_k_partial = _k_group_sum(sum_k_partial)
            norm_q = fx.shuffle_idx(sum_q_partial, subgroup_leader, _WARP_SIZE)
            norm_k = fx.shuffle_idx(sum_k_partial, subgroup_leader, _WARP_SIZE)
            inv_q = fx.math.rsqrt(norm_q + fx.Float32(1e-6))
            inv_k = fx.math.rsqrt(norm_k + fx.Float32(1e-6))

            for ki in range_constexpr(_K_ITERS):
                q_vecs[ki] = q_vecs[ki] * inv_q * fx.Float32(_SCALE)
                k_vecs[ki] = k_vecs[ki] * inv_k

            dot_kq_vec = zero_vec
            for ki in range_constexpr(_K_ITERS):
                dot_kq_vec = fx.math.fma(k_vecs[ki], q_vecs[ki], dot_kq_vec)
            dot_kq = _k_group_sum(_vector_sum(dot_kq_vec))

            beta = _sigmoid(fx.Float32(raw_beta[batch * stride_beta_token + head]))
            state_head_base = head * (_DIM * _DIM)

            state_vecs = []
            for vi in range_constexpr(_V_ITERS):
                global_v = global_v_start + vi * _V_GROUP_TILE
                for ki in range_constexpr(_K_ITERS):
                    k_base = k_vec_start + ki * _WARP_TILE_K
                    state_off = state_head_base + global_v * _DIM + k_base
                    state_vecs.append(
                        state.vec_load((state_off,), _VALUES_PER_THREAD_K)
                    )

            for vi in range_constexpr(_V_ITERS):
                global_v = global_v_start + vi * _V_GROUP_TILE
                sum_hk_vec = zero_vec
                sum_hq_vec = zero_vec
                for ki in range_constexpr(_K_ITERS):
                    state_pos = vi * _K_ITERS + ki
                    decayed = state_vecs[state_pos] * decay_vecs[ki]
                    state_vecs[state_pos] = decayed
                    sum_hk_vec = fx.math.fma(decayed, k_vecs[ki], sum_hk_vec)
                    sum_hq_vec = fx.math.fma(decayed, q_vecs[ki], sum_hq_vec)

                sum_hk = _k_group_sum(_vector_sum(sum_hk_vec))
                sum_hq = _k_group_sum(_vector_sum(sum_hq_vec))

                conv_v = fx.Float32(fx.ptr_load(v_lds + global_v))
                v_new = (conv_v - sum_hk) * beta
                v_new = fx.shuffle_idx(v_new, subgroup_leader, _WARP_SIZE)
                recurrent_value = sum_hq + v_new * dot_kq
                v_new_vec = fx.Vector.filled(_VALUES_PER_THREAD_K, v_new, fx.Float32)

                for ki in range_constexpr(_K_ITERS):
                    state_pos = vi * _K_ITERS + ki
                    updated = fx.math.fma(k_vecs[ki], v_new_vec, state_vecs[state_pos])
                    k_base = k_vec_start + ki * _WARP_TILE_K
                    state_off = state_head_base + global_v * _DIM + k_base
                    state.vec_store((state_off,), updated, _VALUES_PER_THREAD_K)

                if lane_k == 0:
                    fx.ptr_store(fx.BFloat16(recurrent_value), out_lds + global_v)

        def reduce_output_square():
            # Preserve the model's BF16 boundary before RMSNorm and gating.
            recurrent = fx.Float32(fx.ptr_load(out_lds + tid))
            square = recurrent * recurrent
            for offset in (32, 16, 8, 4, 2, 1):
                square = square + square.shuffle_xor(offset, _WARP_SIZE)
            if lane == 0:
                fx.ptr_store(square, norm_lds + warp)

        def store_gated_output():
            norm_sum = fx.Float32(fx.ptr_load(norm_lds))
            norm_sum = norm_sum + fx.Float32(fx.ptr_load(norm_lds + 1))
            inv_rms = fx.math.rsqrt(
                norm_sum * fx.Float32(1.0 / _DIM) + fx.Float32(norm_eps)
            )
            recurrent = fx.Float32(fx.ptr_load(out_lds + tid))
            norm_w = fx.Float32(norm_weight[tid])
            gate_value = fx.Float32(
                output_gate[batch * stride_gate_token + head * stride_gate_head + tid]
            )
            result = recurrent * inv_rms * norm_w * _sigmoid(gate_value)
            out.store(out_base + tid, result.to(fx.BFloat16))

        def store_zero_output():
            out.store(out_base + tid, fx.BFloat16(0.0))

        if valid:
            if tid < _DIM:
                if const_expr(fuse_f_b):
                    project_gate()
                convolve_qkv()
            # Projection and convolution LDS values must be visible before the
            # recurrent core begins.
            fx.gpu.barrier()
            recurrent_update()
            fx.gpu.barrier()
            if tid < _DIM:
                reduce_output_square()
            fx.gpu.barrier()
            if tid < _DIM:
                store_gated_output()
        else:
            if tid < _DIM:
                store_zero_output()

    if fuse_f_b:

        @flyc.kernel(
            name="kda_decode_fused_projection_bf16_gfx950",
            known_block_size=[_BLOCK_THREADS, 1, 1],
        )
        def kernel(
            f_a_mem: fx.Tensor,
            f_b_weight_mem: fx.Tensor,
            x_mem: fx.Tensor,
            weight_mem: fx.Tensor,
            conv_state_mem: fx.Tensor,
            raw_beta_mem: fx.Tensor,
            A_log_mem: fx.Tensor,
            dt_bias_mem: fx.Tensor,
            state_mem: fx.Tensor,
            state_indices_mem: fx.Tensor,
            output_gate_mem: fx.Tensor,
            norm_weight_mem: fx.Tensor,
            out_mem: fx.Tensor,
            num_cache_slots: fx.Int32,
            stride_f_a_token: fx.Int32,
            stride_f_b_head: fx.Int32,
            stride_f_b_output: fx.Int32,
            stride_x_token: fx.Int32,
            stride_weight_channel: fx.Int32,
            stride_weight_width: fx.Int32,
            stride_conv_slot: fx.Int64,
            stride_conv_channel: fx.Int32,
            stride_conv_width: fx.Int32,
            stride_beta_token: fx.Int32,
            stride_state_slot: fx.Int64,
            stride_gate_token: fx.Int32,
            stride_gate_head: fx.Int32,
            stride_out_token: fx.Int32,
            stride_out_head: fx.Int32,
        ):
            decode_body(
                (
                    f_a_mem,
                    f_b_weight_mem,
                    stride_f_a_token,
                    stride_f_b_head,
                    stride_f_b_output,
                ),
                x_mem,
                weight_mem,
                conv_state_mem,
                raw_beta_mem,
                A_log_mem,
                dt_bias_mem,
                state_mem,
                state_indices_mem,
                output_gate_mem,
                norm_weight_mem,
                out_mem,
                num_cache_slots,
                stride_x_token,
                stride_weight_channel,
                stride_weight_width,
                stride_conv_slot,
                stride_conv_channel,
                stride_conv_width,
                stride_beta_token,
                stride_state_slot,
                stride_gate_token,
                stride_gate_head,
                stride_out_token,
                stride_out_head,
            )

        @flyc.jit
        def launch(
            f_a_mem: fx.Tensor,
            f_b_weight_mem: fx.Tensor,
            x_mem: fx.Tensor,
            weight_mem: fx.Tensor,
            conv_state_mem: fx.Tensor,
            raw_beta_mem: fx.Tensor,
            A_log_mem: fx.Tensor,
            dt_bias_mem: fx.Tensor,
            state_mem: fx.Tensor,
            state_indices_mem: fx.Tensor,
            output_gate_mem: fx.Tensor,
            norm_weight_mem: fx.Tensor,
            out_mem: fx.Tensor,
            batch_size: fx.Int32,
            num_cache_slots: fx.Int32,
            stride_f_a_token: fx.Int32,
            stride_f_b_head: fx.Int32,
            stride_f_b_output: fx.Int32,
            stride_x_token: fx.Int32,
            stride_weight_channel: fx.Int32,
            stride_weight_width: fx.Int32,
            stride_conv_slot: fx.Int64,
            stride_conv_channel: fx.Int32,
            stride_conv_width: fx.Int32,
            stride_beta_token: fx.Int32,
            stride_state_slot: fx.Int64,
            stride_gate_token: fx.Int32,
            stride_gate_head: fx.Int32,
            stride_out_token: fx.Int32,
            stride_out_head: fx.Int32,
            stream: fx.Stream = fx.Stream(None),  # noqa: B008
        ):
            kernel(
                f_a_mem,
                f_b_weight_mem,
                x_mem,
                weight_mem,
                conv_state_mem,
                raw_beta_mem,
                A_log_mem,
                dt_bias_mem,
                state_mem,
                state_indices_mem,
                output_gate_mem,
                norm_weight_mem,
                out_mem,
                num_cache_slots,
                stride_f_a_token,
                stride_f_b_head,
                stride_f_b_output,
                stride_x_token,
                stride_weight_channel,
                stride_weight_width,
                stride_conv_slot,
                stride_conv_channel,
                stride_conv_width,
                stride_beta_token,
                stride_state_slot,
                stride_gate_token,
                stride_gate_head,
                stride_out_token,
                stride_out_head,
            ).launch(
                grid=(batch_size * _HEADS, 1, 1),
                block=(_BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    else:

        @flyc.kernel(
            name="kda_decode_bf16_gfx950",
            known_block_size=[_BLOCK_THREADS, 1, 1],
        )
        def kernel(
            x_mem: fx.Tensor,
            weight_mem: fx.Tensor,
            conv_state_mem: fx.Tensor,
            raw_g_mem: fx.Tensor,
            raw_beta_mem: fx.Tensor,
            A_log_mem: fx.Tensor,
            dt_bias_mem: fx.Tensor,
            state_mem: fx.Tensor,
            state_indices_mem: fx.Tensor,
            output_gate_mem: fx.Tensor,
            norm_weight_mem: fx.Tensor,
            out_mem: fx.Tensor,
            num_cache_slots: fx.Int32,
            stride_x_token: fx.Int32,
            stride_weight_channel: fx.Int32,
            stride_weight_width: fx.Int32,
            stride_conv_slot: fx.Int64,
            stride_conv_channel: fx.Int32,
            stride_conv_width: fx.Int32,
            stride_g_token: fx.Int32,
            stride_beta_token: fx.Int32,
            stride_state_slot: fx.Int64,
            stride_gate_token: fx.Int32,
            stride_gate_head: fx.Int32,
            stride_out_token: fx.Int32,
            stride_out_head: fx.Int32,
        ):
            decode_body(
                (raw_g_mem, stride_g_token),
                x_mem,
                weight_mem,
                conv_state_mem,
                raw_beta_mem,
                A_log_mem,
                dt_bias_mem,
                state_mem,
                state_indices_mem,
                output_gate_mem,
                norm_weight_mem,
                out_mem,
                num_cache_slots,
                stride_x_token,
                stride_weight_channel,
                stride_weight_width,
                stride_conv_slot,
                stride_conv_channel,
                stride_conv_width,
                stride_beta_token,
                stride_state_slot,
                stride_gate_token,
                stride_gate_head,
                stride_out_token,
                stride_out_head,
            )

        @flyc.jit
        def launch(
            x_mem: fx.Tensor,
            weight_mem: fx.Tensor,
            conv_state_mem: fx.Tensor,
            raw_g_mem: fx.Tensor,
            raw_beta_mem: fx.Tensor,
            A_log_mem: fx.Tensor,
            dt_bias_mem: fx.Tensor,
            state_mem: fx.Tensor,
            state_indices_mem: fx.Tensor,
            output_gate_mem: fx.Tensor,
            norm_weight_mem: fx.Tensor,
            out_mem: fx.Tensor,
            batch_size: fx.Int32,
            num_cache_slots: fx.Int32,
            stride_x_token: fx.Int32,
            stride_weight_channel: fx.Int32,
            stride_weight_width: fx.Int32,
            stride_conv_slot: fx.Int64,
            stride_conv_channel: fx.Int32,
            stride_conv_width: fx.Int32,
            stride_g_token: fx.Int32,
            stride_beta_token: fx.Int32,
            stride_state_slot: fx.Int64,
            stride_gate_token: fx.Int32,
            stride_gate_head: fx.Int32,
            stride_out_token: fx.Int32,
            stride_out_head: fx.Int32,
            stream: fx.Stream = fx.Stream(None),  # noqa: B008
        ):
            kernel(
                x_mem,
                weight_mem,
                conv_state_mem,
                raw_g_mem,
                raw_beta_mem,
                A_log_mem,
                dt_bias_mem,
                state_mem,
                state_indices_mem,
                output_gate_mem,
                norm_weight_mem,
                out_mem,
                num_cache_slots,
                stride_x_token,
                stride_weight_channel,
                stride_weight_width,
                stride_conv_slot,
                stride_conv_channel,
                stride_conv_width,
                stride_g_token,
                stride_beta_token,
                stride_state_slot,
                stride_gate_token,
                stride_gate_head,
                stride_out_token,
                stride_out_head,
            ).launch(
                grid=(batch_size * _HEADS, 1, 1),
                block=(_BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    # The projection raises register pressure, so it targets fewer waves.
    launch.compile_hints = {
        "waves_per_eu": 2 if fuse_f_b else 3,
        "llvm_options": {
            "amdgpu-expert-scheduling-mode": True,
        },
    }
    return launch


__all__ = ["create_kda_decode_kernel"]
