# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL candidate selection and min-unique kernels for HERD routing."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import Float32, Int32, Int64, const_expr, gpu, range_constexpr
from flydsl.expr import rocdl as fly_rocdl
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.act import sigmoid_batch, sigmoid_f32
from aiter.ops.flydsl.kernels.kernels_common import (
    F32_INF_BITS,
    atomic_add_i32,
    kernel_signature,
    ord_signed_f32,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_copy_atom, ptr_buf_tensor

_FINALIZE_THREADS = 256
_K3_EXPERTS = 896
_K3_KP1 = 17
_K3_BLOCK_THREADS = 256
_K3_WAVE_SIZE = 64
_K3_VEC = 4
_K3_EXPERT_BITS = 10
_K3_ELEMS_PER_THREAD = _K3_VEC
_K3_WAVES_PER_BLOCK = _K3_BLOCK_THREADS // _K3_WAVE_SIZE
_K3_SURVIVORS = _K3_KP1 * _K3_WAVES_PER_BLOCK * _K3_VEC
_INT32_MIN = -2147483648


def _isnan_f32(value):
    bits = value.bitcast(Int32) & Int32(0x7FFFFFFF)
    return bits > Int32(F32_INF_BITS)


@cache
def build_kimi_k3_herd_candidates_module(bias_bf16: bool):
    """Build Kimi-K3's NaN-safe, Triton-ordered Top-17 selector."""

    @fx.struct
    class SharedStorage:
        chunk_max: fx.Array[Int64, _K3_BLOCK_THREADS, 16]
        survivor_key: fx.Array[Int64, _K3_SURVIVORS, 16]
        survivor_col: fx.Array[Int32, _K3_SURVIVORS, 16]
        survivor_count: fx.Array[Int32, 1, 4]
        cut: fx.Array[Int64, 1, 8]
        chosen_mask: fx.Array[Int64, 1, 8]

    @flyc.kernel(
        name="kimi_k3_herd_candidates_"
        + kernel_signature(bb=bias_bf16, blk=_K3_BLOCK_THREADS),
        known_block_size=[_K3_BLOCK_THREADS, 1, 1],
    )
    def candidate_kernel(
        scores: fx.Tensor,
        candidate_ids: fx.Tensor,
        candidate_values: fx.Tensor,
        bias: fx.Tensor,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid % Int32(_K3_WAVE_SIZE)
        wave = tid // Int32(_K3_WAVE_SIZE)
        zero = Int32(0)
        one = Int32(1)

        storage = fx.SharedAllocator().allocate(SharedStorage)
        chunk_max = storage.chunk_max.peek().view(
            fx.make_layout(_K3_BLOCK_THREADS, 1)
        )
        survivor_key = storage.survivor_key.peek().view(
            fx.make_layout(_K3_SURVIVORS, 1)
        )
        survivor_col = storage.survivor_col.peek().view(
            fx.make_layout(_K3_SURVIVORS, 1)
        )
        survivor_count = storage.survivor_count.peek().view(fx.make_layout(1, 1))
        cut_shared = storage.cut.peek().view(fx.make_layout(1, 1))
        chosen_mask_shared = storage.chosen_mask.peek().view(fx.make_layout(1, 1))

        score_row = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(fx.slice(scores, (row, None)), max_size=False),
            fx.make_layout(_K3_VEC, 1),
        )
        score_scalar_row = fx.slice(scores, (row, None))
        row_ids = fx.slice(candidate_ids, (row, None))
        row_values = fx.slice(candidate_values, (row, None))

        src = fx.slice(score_row, (None, tid))
        fragment = fx.make_fragment_like(src)
        fx.copy(buf_copy_atom(16, Float32), src, fragment)
        loaded = fx.Vector(fx.memref_load_vec(fragment))
        unbiased = sigmoid_batch(
            [Float32(loaded[j]) for j in range_constexpr(_K3_VEC)]
        )

        keys = fx.make_rmem_tensor(_K3_ELEMS_PER_THREAD, Int64)
        lane_max = Int64(-1)
        col_base = tid * Int32(_K3_VEC)
        for j in range_constexpr(_K3_VEC):
            col = col_base + Int32(j)
            live = col < Int32(_K3_EXPERTS)
            selection = Float32(float("-inf"))
            if live:
                selection = unbiased[j] + Float32(bias[col])
            # Match topk_gating's public NaN contract: an invalid score ranks
            # alongside -Inf, below every finite score, and carries a zero
            # routing weight if the row is too starved to avoid selecting it.
            selection = _isnan_f32(selection).select(
                Float32(float("-inf")),
                selection,
            )
            score_order = ord_signed_f32(selection)
            unsigned_order = Int64(score_order ^ Int32(_INT32_MIN)) & Int64(
                0xFFFFFFFF
            )
            key = (unsigned_order << Int64(_K3_EXPERT_BITS)) | Int64(col)
            key = live.select(key, Int64(-1))
            keys[j] = key
            lane_max = fx.max(lane_max, key)

        chunk_max[tid] = lane_max
        if tid == zero:
            survivor_count[0] = zero
        gpu.barrier()

        if wave == zero:
            chunk_key = Int64(-1)
            for s in range_constexpr(_K3_WAVES_PER_BLOCK):
                chunk_key = fx.max(
                    chunk_key,
                    chunk_max[Int32(s * _K3_WAVE_SIZE) + lane],
                )

            # Composite keys are non-negative and unique (score bits, expert id),
            # so a signed 42-bit binary search finds an exact Top-17 threshold.
            cut = Int64(0)
            for bit in range_constexpr(32 + _K3_EXPERT_BITS):
                probe = cut | (Int64(1) << Int64(31 + _K3_EXPERT_BITS - bit))
                hits = fly_rocdl.ballot(T.i64, chunk_key >= probe)
                cut = (Int32(fx.math.ctpop(hits)) >= Int32(_K3_KP1)).select(
                    probe, cut
                )
            chosen = fly_rocdl.ballot(T.i64, chunk_key >= cut)
            if lane == zero:
                cut_shared[0] = cut
                chosen_mask_shared[0] = chosen
        gpu.barrier()

        cut = cut_shared[0]
        chosen_mask = chosen_mask_shared[0]
        in_chosen = ((chosen_mask >> Int64(lane)) & Int64(1)) == Int64(1)
        for j in range_constexpr(_K3_ELEMS_PER_THREAD):
            col = col_base + Int32(j)
            if in_chosen & (keys[j] >= cut) & (col < Int32(_K3_EXPERTS)):
                at = atomic_add_i32(survivor_count, one, 0, "workgroup")
                survivor_key[at] = keys[j]
                survivor_col[at] = col
        gpu.barrier()

        found = survivor_count[0]
        for mine in range(tid, found, Int32(_K3_BLOCK_THREADS)):
            my_key = survivor_key[mine]
            my_col = survivor_col[mine]
            place = zero
            for other in range(zero, found, one):
                place = place + (survivor_key[other] > my_key).select(one, zero)
            if place < Int32(_K3_KP1):
                raw = sigmoid_f32(Float32(score_scalar_row[my_col]))
                invalid = _isnan_f32(raw) | _isnan_f32(Float32(bias[my_col]))
                raw = invalid.select(Float32(0.0), raw)
                row_ids[place] = my_col
                row_values[place] = raw

    @flyc.jit
    def launch_candidates(
        scores: fx.Tensor,
        candidate_ids: fx.Tensor,
        candidate_values: fx.Tensor,
        bias: fx.Tensor,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        candidate_kernel(scores, candidate_ids, candidate_values, bias).launch(
            grid=(rows, 1, 1),
            block=(_K3_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_candidates


@cache
def build_herd_finalize_module(
    topk: int,
    num_experts: int,
    renormalize: bool,
    round_to_bf16: bool,
    block_threads: int = _FINALIZE_THREADS,
):
    """Build the fused popularity, min-unique, and Top-K output launcher."""
    if topk < 1 or topk + 1 > num_experts:
        raise ValueError(
            f"invalid HERD geometry: topk={topk}, num_experts={num_experts}"
        )
    if block_threads not in (64, 128, 256, 512, 1024):
        raise ValueError(f"unsupported HERD block size {block_threads}")
    kp1 = topk + 1

    @fx.struct
    class SharedStorage:
        popularity: fx.Array[Int32, num_experts, 16]

    @flyc.kernel(
        name="herd_finalize_"
        + kernel_signature(
            k=topk,
            e=num_experts,
            rn=renormalize,
            rb=round_to_bf16,
            blk=block_threads,
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def finalize_kernel(
        candidate_ids: fx.Tensor,
        candidate_values: fx.Tensor,
        topk_weights: fx.Tensor,
        topk_ids: fx.Tensor,
        topk_weights_stride: Int32,
        topk_ids_stride: Int32,
        rows: Int32,
        routed_scaling_factor: Float32,
    ):
        tid = fx.thread_idx.x
        storage = fx.SharedAllocator().allocate(SharedStorage)
        popularity = storage.popularity.peek().view(fx.make_layout(num_experts, 1))
        topk_weights_flat = ptr_buf_tensor(topk_weights, elem=Float32)
        topk_ids_flat = ptr_buf_tensor(topk_ids, elem=Int32)

        for expert in range(tid, Int32(num_experts), Int32(block_threads)):
            popularity[expert] = Int32(0)
        gpu.barrier()

        total = rows * Int32(kp1)
        for idx in range(tid, total, Int32(block_threads)):
            candidate_row = idx // Int32(kp1)
            candidate_col = idx - candidate_row * Int32(kp1)
            expert = candidate_ids[candidate_row, candidate_col]
            atomic_add_i32(popularity, 1, expert, "workgroup")
        gpu.barrier()

        row = tid
        if row < rows:
            ids = [candidate_ids[row, j] for j in range_constexpr(kp1)]
            values = [candidate_values[row, j] for j in range_constexpr(kp1)]
            pops = [popularity[ids[j]] for j in range_constexpr(kp1)]

            drop = Int32(0)
            drop_id = ids[0]
            drop_value = values[0]
            drop_pop = pops[0]
            for j in range_constexpr(1, kp1):
                take = (pops[j] < drop_pop) | (
                    (pops[j] == drop_pop)
                    & (
                        (values[j] < drop_value)
                        | ((values[j] == drop_value) & (ids[j] < drop_id))
                    )
                )
                drop = take.select(Int32(j), drop)
                drop_id = take.select(ids[j], drop_id)
                drop_value = take.select(values[j], drop_value)
                drop_pop = take.select(pops[j], drop_pop)

            kept_sum = Float32(0.0)
            for j in range_constexpr(kp1):
                kept_sum = kept_sum + (drop != Int32(j)).select(values[j], Float32(0.0))
            if const_expr(renormalize):
                scale = routed_scaling_factor / fx.max(kept_sum, Float32(1e-20))
            else:
                scale = routed_scaling_factor

            for j in range_constexpr(kp1):
                keep = drop != Int32(j)
                rank = Int32(0)
                for other in range_constexpr(kp1):
                    before = (drop != Int32(other)) & (ids[other] < ids[j])
                    rank = rank + before.select(Int32(1), Int32(0))
                if keep:
                    weight = values[j] * scale
                    if const_expr(round_to_bf16):
                        weight = weight.to(fx.BFloat16).to(Float32)
                    topk_ids_flat[row * topk_ids_stride + rank] = ids[j]
                    topk_weights_flat[row * topk_weights_stride + rank] = weight

    @flyc.jit
    def launch_finalize(
        candidate_ids: fx.Tensor,
        candidate_values: fx.Tensor,
        topk_weights: fx.Tensor,
        topk_ids: fx.Tensor,
        topk_weights_stride: fx.Int32,
        topk_ids_stride: fx.Int32,
        rows: Int32,
        routed_scaling_factor: Float32,
        stream: fx.Stream,
    ):
        finalize_kernel(
            candidate_ids,
            candidate_values,
            topk_weights,
            topk_ids,
            topk_weights_stride,
            topk_ids_stride,
            rows,
            routed_scaling_factor,
        ).launch(
            grid=(1, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_finalize
