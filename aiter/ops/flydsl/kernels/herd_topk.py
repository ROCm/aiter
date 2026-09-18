# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL popularity and min-unique finalize kernel for HERD routing."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import Float32, Int32, const_expr, gpu, range_constexpr

from aiter.ops.flydsl.kernels.kernels_common import atomic_add_i32, kernel_signature
from aiter.ops.flydsl.kernels.tensor_shim import ptr_buf_tensor

_FINALIZE_THREADS = 256


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
