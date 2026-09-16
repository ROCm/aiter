# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Partitioned-softmax reduction kernel for FlyDSL paged attention."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T

MAX_CONTEXT_PARTITIONS = 256
_DTYPE_MAP = {
    "f32": fx.Float32,
    "f16": fx.Float16,
    "bf16": fx.BFloat16,
}


def _validate_pa_decode_ps_reduce_config(
    *,
    max_context_partition_num: int,
    head_size: int,
    output_dtype_str: str,
    logits_dtype_str: str,
    sink_dtype_str: str,
) -> None:
    if not 1 <= max_context_partition_num <= MAX_CONTEXT_PARTITIONS:
        raise ValueError(
            f"max_context_partition_num must be in [1, {MAX_CONTEXT_PARTITIONS}], "
            f"got {max_context_partition_num}"
        )
    if head_size <= 0 or head_size > 1024 or head_size % 64:
        raise ValueError(
            f"head_size must be a multiple of 64 in [64, 1024], got {head_size}"
        )
    for dtype_str in (output_dtype_str, logits_dtype_str, sink_dtype_str):
        if dtype_str not in _DTYPE_MAP:
            raise ValueError(f"Unsupported FlyDSL dtype: {dtype_str!r}")


def is_pa_decode_ps_reduce_supported(
    *,
    max_context_partition_num: int,
    head_size: int,
    output_dtype_str: str,
    logits_dtype_str: str,
    sink_dtype_str: str,
) -> bool:
    """Return whether the FlyDSL reducer supports a dispatch configuration."""
    try:
        _validate_pa_decode_ps_reduce_config(
            max_context_partition_num=max_context_partition_num,
            head_size=head_size,
            output_dtype_str=output_dtype_str,
            logits_dtype_str=logits_dtype_str,
            sink_dtype_str=sink_dtype_str,
        )
    except ValueError:
        return False
    return True


@lru_cache(maxsize=256)
def compile_pa_decode_ps_reduce(
    *,
    max_context_partition_num: int,
    head_size: int,
    output_dtype_str: str,
    logits_dtype_str: str,
    sink_dtype_str: str,
    use_sinks: bool,
    use_work_plan: bool = False,
):
    """Build the partitioned-softmax reduction used by ``pa_decode``.

    Counts up to one wave use one partition per lane and one output element per
    thread.  For D=128 and larger counts, a 2-D workgroup materializes weights
    once in LDS and splits each output element's partition chain over several
    waves.  Other head sizes retain the register-only lane-striped fallback.
    """
    _validate_pa_decode_ps_reduce_config(
        max_context_partition_num=max_context_partition_num,
        head_size=head_size,
        output_dtype_str=output_dtype_str,
        logits_dtype_str=logits_dtype_str,
        sink_dtype_str=sink_dtype_str,
    )

    output_dtype = _DTYPE_MAP[output_dtype_str]
    logits_dtype = _DTYPE_MAP[logits_dtype_str]
    sink_dtype = _DTYPE_MAP[sink_dtype_str]

    warp_size = 64
    log2e = 1.4426950408889634
    reduce_width = (
        1
        if max_context_partition_num == 1
        else 1 << ((max_context_partition_num - 1).bit_length())
    )
    reduce_shuffle_offsets = [
        offset for offset in (32, 16, 8, 4, 2, 1) if offset < reduce_width
    ]

    # The original mapping gives each output element to one thread, so every
    # thread walks every partition.  That is a good fit for <=1 wave of
    # partitions, but it leaves NP=160..256 as a long dependent load/FMA
    # chain.  For the decode shape used by PA (D=128), split that chain over
    # two or eight independent wave pairs.  A pair covers the two 64-element
    # halves of the output vector, while its y-coordinate selects a disjoint
    # contiguous range of partitions.
    use_parallel_lds = head_size == 128 and max_context_partition_num > warp_size
    parallel_groups = 1
    if use_parallel_lds:
        # Eight groups win from NP=128 onward on gfx950; two avoid excessive
        # synchronization/thread overhead for the small >64 tail.
        parallel_groups = 2 if max_context_partition_num <= 96 else 8
    head_waves = head_size // warp_size
    worker_waves = head_waves * parallel_groups
    block_shape = (
        [warp_size, worker_waves, 1] if use_parallel_lds else [head_size, 1, 1]
    )
    parts_per_group = (
        max_context_partition_num + parallel_groups - 1
    ) // parallel_groups

    # Keep the legacy specializations effectively LDS-free.  The fields are
    # only allocated from the compile-time parallel branch below.
    shared_weight_elems = max_context_partition_num if use_parallel_lds else 1
    shared_partial_elems = (parallel_groups - 1) * head_size if use_parallel_lds else 1

    @fx.struct
    class SharedStorage:
        weights: fx.Array[fx.Float32, shared_weight_elems, 16]
        partials: fx.Array[fx.Float32, shared_partial_elems, 16]

    @flyc.kernel(known_block_size=block_shape)
    def pa_decode_ps_reduce_kernel(
        output_ptr: fx.Pointer,
        exp_sums_ptr: fx.Pointer,
        max_logits_ptr: fx.Pointer,
        logits_ptr: fx.Pointer,
        sink_token_ptr: fx.Pointer,
        stride_output_bs: fx.Int32,
        stride_output_len: fx.Int32,
        stride_output_kv_head: fx.Int32,
        stride_output_group_size: fx.Int32,
        stride_exp_sums_seq: fx.Int32,
        stride_exp_sums_head: fx.Int32,
        stride_exp_sums_part: fx.Int32,
        stride_logits_seq: fx.Int32,
        stride_logits_head: fx.Int32,
        stride_logits_part: fx.Int32,
        stride_logits_group: fx.Int32,
        query_group_size: fx.Int32,
        reduce_info_ptr: fx.Pointer,
    ):
        tid = fx.thread_idx.x
        worker = fx.thread_idx.y
        batch_idx = fx.block_idx.x
        kv_head_idx = fx.block_idx.y
        eqgs_idx = fx.block_idx.z

        output = fx.recast_iter(output_dtype, output_ptr)
        exp_sums = fx.recast_iter(fx.Float32, exp_sums_ptr)
        max_logits = fx.recast_iter(fx.Float32, max_logits_ptr)
        logits = fx.recast_iter(logits_dtype, logits_ptr)
        if fx.const_expr(use_sinks):
            sink_token = fx.recast_iter(sink_dtype, sink_token_ptr)

        zero_f = fx.Float32(0.0)
        one_f = fx.Float32(1.0)
        neg_inf = fx.Float32(float("-inf"))
        c_log2e = fx.Float32(log2e)
        zero_i = fx.Int32(0)
        c_warp_size = fx.Int32(warp_size)
        c_wave_mask = fx.Int32(warp_size - 1)
        c_part_num = fx.Int32(max_context_partition_num)
        if fx.const_expr(use_work_plan):
            reduce_info = fx.recast_iter(fx.Int32, reduce_info_ptr)
            first_part = fx.Int32(reduce_info[batch_idx * 2])
            c_part_num = fx.Int32(reduce_info[batch_idx * 2 + 1])
            stats_seq_offset = first_part * stride_exp_sums_part
            logits_seq_offset = first_part * stride_logits_part
        else:
            stats_seq_offset = batch_idx * stride_exp_sums_seq
            logits_seq_offset = batch_idx * stride_logits_seq
        c_reduce_width = fx.Int32(reduce_width)
        c_four = fx.Int32(4)
        c_qgs = query_group_size
        lane = tid & c_wave_mask
        group_idx = eqgs_idx % c_qgs

        def _wave_reduce_max(value):
            reduced = value
            for offset in reduce_shuffle_offsets:
                reduced = reduced.maximumf(
                    reduced.shuffle_xor(fx.Int32(offset), c_warp_size)
                )
            return reduced

        def _wave_reduce_sum(value):
            reduced = value
            for offset in reduce_shuffle_offsets:
                reduced = fx.Float32(
                    reduced.addf(
                        reduced.shuffle_xor(fx.Int32(offset), c_warp_size),
                        fastmath="fast",
                    )
                )
            return reduced

        if fx.const_expr(use_parallel_lds):
            # One wave materializes the normalized partition weights once.
            # All output waves then reuse those weights from LDS and split the
            # long partition loop.  This avoids both duplicated exp2 work and
            # a ds_bpermute for every output FMA.
            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            lds_weights = lds.weights
            lds_partials = lds.partials

            if worker == zero_i:
                partitions_per_lane = (
                    max_context_partition_num + warp_size - 1
                ) // warp_size
                part_sums = []
                part_maxes = []
                lane_max = neg_inf
                for chunk_idx in fx.range_constexpr(partitions_per_lane):
                    chunk_base = chunk_idx * warp_size
                    chunk_size = min(warp_size, max_context_partition_num - chunk_base)
                    part_idx = lane + fx.Int32(chunk_base)
                    if fx.const_expr(chunk_size == warp_size and not use_work_plan):
                        stats_offset = (
                            stats_seq_offset
                            + kv_head_idx * stride_exp_sums_head
                            + part_idx * stride_exp_sums_part
                            + eqgs_idx
                        )
                        part_sum = fx.Float32(exp_sums[stats_offset])
                        part_max = fx.Float32(max_logits[stats_offset])
                    else:
                        lane_in_range = (lane < fx.Int32(chunk_size)) & (
                            part_idx < c_part_num
                        )
                        stats_offset = (
                            stats_seq_offset
                            + kv_head_idx * stride_exp_sums_head
                            + part_idx * stride_exp_sums_part
                            + eqgs_idx
                        )
                        part_sum = zero_f
                        part_max = neg_inf
                        if lane_in_range:
                            part_sum = fx.Float32(exp_sums[stats_offset])
                            part_max = fx.Float32(max_logits[stats_offset])
                    part_sums.append(part_sum)
                    part_maxes.append(part_max)
                    lane_max = lane_max.maximumf(part_max)

                global_max = _wave_reduce_max(lane_max)
                safe_global_max = (global_max > neg_inf).select(global_max, zero_f)
                scaled_sums = []
                lane_exp_sum = zero_f
                for chunk_idx in fx.range_constexpr(partitions_per_lane):
                    part_max = part_maxes[chunk_idx]
                    if fx.const_expr(use_work_plan):
                        # ``select`` evaluates its exp2 operand even for an
                        # inactive planned partition. Keep the static path
                        # branch-free, but predicate that expensive operation
                        # when the per-request count is dynamic.
                        part_scale = zero_f
                        if part_max > neg_inf:
                            part_scale = fx.exp2(
                                (part_max - safe_global_max) * c_log2e,
                                fastmath="fast",
                            )
                    else:
                        part_scale = (part_max > neg_inf).select(
                            fx.exp2(
                                (part_max - safe_global_max) * c_log2e,
                                fastmath="fast",
                            ),
                            zero_f,
                        )
                    scaled_sum = part_sums[chunk_idx] * part_scale
                    scaled_sums.append(scaled_sum)
                    lane_exp_sum = lane_exp_sum + scaled_sum

                global_exp_sum = _wave_reduce_sum(lane_exp_sum)
                if fx.const_expr(use_sinks):
                    sink_value = fx.Float32(sink_token[kv_head_idx * c_qgs + group_idx])
                    sink_scale = (global_max > neg_inf).select(
                        fx.exp2(
                            (sink_value - safe_global_max) * c_log2e,
                            fastmath="fast",
                        ),
                        zero_f,
                    )
                    global_exp_sum = global_exp_sum + sink_scale
                safe_global_exp_sum = (global_exp_sum > zero_f).select(
                    global_exp_sum, one_f
                )

                for chunk_idx in fx.range_constexpr(partitions_per_lane):
                    chunk_base = chunk_idx * warp_size
                    chunk_size = min(warp_size, max_context_partition_num - chunk_base)
                    part_idx = lane + fx.Int32(chunk_base)
                    if fx.const_expr(use_work_plan):
                        lane_in_range = (lane < fx.Int32(chunk_size)) & (
                            part_idx < c_part_num
                        )
                        if lane_in_range:
                            weight = scaled_sums[chunk_idx] / safe_global_exp_sum
                            lds_weights[part_idx] = weight
                    else:
                        weight = scaled_sums[chunk_idx] / safe_global_exp_sum
                        if fx.const_expr(chunk_size == warp_size):
                            lds_weights[part_idx] = weight
                        else:
                            lane_in_range = lane < fx.Int32(chunk_size)
                            if lane_in_range:
                                lds_weights[part_idx] = weight

            fx.gpu.barrier()

            head_wave = worker % fx.Int32(head_waves)
            partition_group = worker // fx.Int32(head_waves)
            output_element = head_wave * c_warp_size + lane
            group_part_begin = partition_group * fx.Int32(parts_per_group)
            acc = zero_f
            if fx.const_expr(use_work_plan):
                group_in_range = group_part_begin < c_part_num
                if group_in_range:
                    for local_part in fx.range_constexpr(parts_per_group):
                        part_idx = group_part_begin + fx.Int32(local_part)
                        if part_idx < c_part_num:
                            weight = fx.Float32(lds_weights[part_idx])
                            logits_offset = (
                                logits_seq_offset
                                + kv_head_idx * stride_logits_head
                                + part_idx * stride_logits_part
                                + eqgs_idx * stride_logits_group
                                + output_element
                            )
                            part_logits = fx.Float32(logits[logits_offset])
                            acc = acc + part_logits * weight
            else:
                for local_part in fx.range_constexpr(parts_per_group):
                    part_idx = group_part_begin + fx.Int32(local_part)
                    part_in_range = part_idx < c_part_num
                    if part_in_range:
                        weight = fx.Float32(lds_weights[part_idx])
                        logits_offset = (
                            logits_seq_offset
                            + kv_head_idx * stride_logits_head
                            + part_idx * stride_logits_part
                            + eqgs_idx * stride_logits_group
                            + output_element
                        )
                        part_logits = fx.Float32(logits[logits_offset])
                        acc = acc + part_logits * weight

            if partition_group > zero_i:
                if fx.const_expr(use_work_plan):
                    if group_in_range:
                        partial_offset = (partition_group - fx.Int32(1)) * fx.Int32(
                            head_size
                        ) + output_element
                        lds_partials[partial_offset] = acc
                else:
                    partial_offset = (partition_group - fx.Int32(1)) * fx.Int32(
                        head_size
                    ) + output_element
                    lds_partials[partial_offset] = acc

            fx.gpu.barrier()

            if partition_group == zero_i:
                for other_group in fx.range_constexpr(1, parallel_groups):
                    if fx.const_expr(use_work_plan):
                        if fx.Int32(other_group * parts_per_group) < c_part_num:
                            partial_offset = (
                                fx.Int32((other_group - 1) * head_size) + output_element
                            )
                            acc = acc + fx.Float32(lds_partials[partial_offset])
                    else:
                        partial_offset = (
                            fx.Int32((other_group - 1) * head_size) + output_element
                        )
                        acc = acc + fx.Float32(lds_partials[partial_offset])

        elif fx.const_expr(max_context_partition_num <= warp_size):
            # Exact powers of two have no inactive lanes inside their reduction
            # subgroup. Keep that original path unchanged; only partial
            # subgroups need an EXEC-masked load to avoid carrying a predicate
            # across the shuffle sequence.
            if fx.const_expr(
                max_context_partition_num == reduce_width and not use_work_plan
            ):
                lane_in_range = lane < c_part_num
                lane_in_reduce = lane < c_reduce_width
                part_sum = zero_f
                part_max = neg_inf
                if lane_in_reduce:
                    part_idx = lane_in_range.select(lane, zero_i)
                    stats_offset = (
                        stats_seq_offset
                        + kv_head_idx * stride_exp_sums_head
                        + part_idx * stride_exp_sums_part
                        + eqgs_idx
                    )
                    loaded_sum = fx.Float32(exp_sums[stats_offset])
                    loaded_max = fx.Float32(max_logits[stats_offset])
                    part_sum = lane_in_range.select(loaded_sum, zero_f)
                    part_max = lane_in_range.select(loaded_max, neg_inf)
            else:
                lane_in_range = lane < c_part_num
                stats_offset = (
                    stats_seq_offset
                    + kv_head_idx * stride_exp_sums_head
                    + lane * stride_exp_sums_part
                    + eqgs_idx
                )
                part_sum = zero_f
                part_max = neg_inf
                if lane_in_range:
                    part_sum = fx.Float32(exp_sums[stats_offset])
                    part_max = fx.Float32(max_logits[stats_offset])

            global_max = _wave_reduce_max(part_max)
            safe_global_max = (global_max > neg_inf).select(global_max, zero_f)
            if fx.const_expr(use_work_plan):
                part_scale = zero_f
                if part_max > neg_inf:
                    part_scale = fx.exp2(
                        (part_max - safe_global_max) * c_log2e,
                        fastmath="fast",
                    )
            else:
                part_scale = (part_max > neg_inf).select(
                    fx.exp2((part_max - safe_global_max) * c_log2e, fastmath="fast"),
                    zero_f,
                )
            scaled_sum = part_sum * part_scale
            global_exp_sum = _wave_reduce_sum(scaled_sum)
            if fx.const_expr(use_sinks):
                sink_value = fx.Float32(sink_token[kv_head_idx * c_qgs + group_idx])
                sink_scale = (global_max > neg_inf).select(
                    fx.exp2(
                        (sink_value - safe_global_max) * c_log2e,
                        fastmath="fast",
                    ),
                    zero_f,
                )
                global_exp_sum = global_exp_sum + sink_scale
            safe_global_exp_sum = (global_exp_sum > zero_f).select(
                global_exp_sum, one_f
            )
            if fx.const_expr(use_work_plan):
                weight_local = zero_f
                if lane_in_range:
                    weight_local = scaled_sum / safe_global_exp_sum
            else:
                weight_local = scaled_sum / safe_global_exp_sum
            weight_local_i32 = weight_local.bitcast(fx.Int32)

            acc = zero_f
            if fx.const_expr(use_work_plan):
                for part_idx in fx.range_constexpr(max_context_partition_num):
                    c_part_idx = fx.Int32(part_idx)
                    if c_part_idx < c_part_num:
                        weight_i32 = fx.Int32(
                            fx.rocdl.ds_bpermute(
                                T.i32,
                                c_part_idx * c_four,
                                weight_local_i32,
                            )
                        )
                        weight = weight_i32.bitcast(fx.Float32)
                        logits_offset = (
                            logits_seq_offset
                            + kv_head_idx * stride_logits_head
                            + c_part_idx * stride_logits_part
                            + eqgs_idx * stride_logits_group
                            + tid
                        )
                        part_logits = fx.Float32(logits[logits_offset])
                        acc = acc + part_logits * weight
            else:
                for part_idx in fx.range_constexpr(max_context_partition_num):
                    c_part_idx = fx.Int32(part_idx)
                    weight_i32 = fx.Int32(
                        fx.rocdl.ds_bpermute(
                            T.i32,
                            c_part_idx * c_four,
                            weight_local_i32,
                        )
                    )
                    weight = weight_i32.bitcast(fx.Float32)
                    logits_offset = (
                        logits_seq_offset
                        + kv_head_idx * stride_logits_head
                        + c_part_idx * stride_logits_part
                        + eqgs_idx * stride_logits_group
                        + tid
                    )
                    part_logits = fx.Float32(logits[logits_offset])
                    acc = acc + part_logits * weight
        else:
            # A wave covers several 64-partition chunks. Lane ``l`` owns
            # partitions l, l+64, l+128, and l+192 (as present). Reduce the
            # lane-local maxima/sums before the usual wave reduction; later,
            # select the corresponding local weight and broadcast from
            # ``part_idx % 64``. This stays register-only through NP=256.
            partitions_per_lane = (
                max_context_partition_num + warp_size - 1
            ) // warp_size
            part_sums = []
            part_maxes = []
            lane_max = neg_inf
            for chunk_idx in fx.range_constexpr(partitions_per_lane):
                chunk_base = chunk_idx * warp_size
                chunk_size = min(warp_size, max_context_partition_num - chunk_base)
                part_idx = lane + fx.Int32(chunk_base)
                if fx.const_expr(chunk_size == warp_size and not use_work_plan):
                    stats_offset = (
                        stats_seq_offset
                        + kv_head_idx * stride_exp_sums_head
                        + part_idx * stride_exp_sums_part
                        + eqgs_idx
                    )
                    part_sum = fx.Float32(exp_sums[stats_offset])
                    part_max = fx.Float32(max_logits[stats_offset])
                else:
                    lane_in_range = (lane < fx.Int32(chunk_size)) & (
                        part_idx < c_part_num
                    )
                    stats_offset = (
                        stats_seq_offset
                        + kv_head_idx * stride_exp_sums_head
                        + part_idx * stride_exp_sums_part
                        + eqgs_idx
                    )
                    part_sum = zero_f
                    part_max = neg_inf
                    if lane_in_range:
                        part_sum = fx.Float32(exp_sums[stats_offset])
                        part_max = fx.Float32(max_logits[stats_offset])
                part_sums.append(part_sum)
                part_maxes.append(part_max)
                lane_max = lane_max.maximumf(part_max)

            global_max = _wave_reduce_max(lane_max)
            safe_global_max = (global_max > neg_inf).select(global_max, zero_f)
            scaled_sums = []
            lane_exp_sum = zero_f
            for chunk_idx in fx.range_constexpr(partitions_per_lane):
                part_max = part_maxes[chunk_idx]
                if fx.const_expr(use_work_plan):
                    part_scale = zero_f
                    if part_max > neg_inf:
                        part_scale = fx.exp2(
                            (part_max - safe_global_max) * c_log2e,
                            fastmath="fast",
                        )
                else:
                    part_scale = (part_max > neg_inf).select(
                        fx.exp2(
                            (part_max - safe_global_max) * c_log2e,
                            fastmath="fast",
                        ),
                        zero_f,
                    )
                scaled_sum = part_sums[chunk_idx] * part_scale
                scaled_sums.append(scaled_sum)
                lane_exp_sum = lane_exp_sum + scaled_sum

            global_exp_sum = _wave_reduce_sum(lane_exp_sum)
            if fx.const_expr(use_sinks):
                sink_value = fx.Float32(sink_token[kv_head_idx * c_qgs + group_idx])
                sink_scale = (global_max > neg_inf).select(
                    fx.exp2(
                        (sink_value - safe_global_max) * c_log2e,
                        fastmath="fast",
                    ),
                    zero_f,
                )
                global_exp_sum = global_exp_sum + sink_scale
            safe_global_exp_sum = (global_exp_sum > zero_f).select(
                global_exp_sum, one_f
            )

            acc = zero_f
            for chunk_idx in fx.range_constexpr(partitions_per_lane):
                chunk_base = chunk_idx * warp_size
                chunk_size = min(warp_size, max_context_partition_num - chunk_base)
                if fx.const_expr(use_work_plan):
                    # Initialize in the enclosing constexpr-loop scope so the
                    # FlyDSL dynamic-if rewriter never observes a stale value
                    # from a previous unrolled chunk.
                    weight_local_i32 = zero_f.bitcast(fx.Int32)
                    if fx.Int32(chunk_base) < c_part_num:
                        weight_local_i32 = (
                            scaled_sums[chunk_idx] / safe_global_exp_sum
                        ).bitcast(fx.Int32)
                        for part_lane in fx.range_constexpr(chunk_size):
                            part_idx = chunk_base + part_lane
                            c_part_idx = fx.Int32(part_idx)
                            if c_part_idx < c_part_num:
                                weight_i32 = fx.Int32(
                                    fx.rocdl.ds_bpermute(
                                        T.i32,
                                        fx.Int32(part_lane) * c_four,
                                        weight_local_i32,
                                    )
                                )
                                weight = weight_i32.bitcast(fx.Float32)
                                logits_offset = (
                                    logits_seq_offset
                                    + kv_head_idx * stride_logits_head
                                    + c_part_idx * stride_logits_part
                                    + eqgs_idx * stride_logits_group
                                    + tid
                                )
                                part_logits = fx.Float32(logits[logits_offset])
                                acc = acc + part_logits * weight
                else:
                    weight_local_i32 = (
                        scaled_sums[chunk_idx] / safe_global_exp_sum
                    ).bitcast(fx.Int32)
                    for part_lane in fx.range_constexpr(chunk_size):
                        part_idx = chunk_base + part_lane
                        c_part_idx = fx.Int32(part_idx)
                        weight_i32 = fx.Int32(
                            fx.rocdl.ds_bpermute(
                                T.i32,
                                fx.Int32(part_lane) * c_four,
                                weight_local_i32,
                            )
                        )
                        weight = weight_i32.bitcast(fx.Float32)
                        logits_offset = (
                            logits_seq_offset
                            + kv_head_idx * stride_logits_head
                            + c_part_idx * stride_logits_part
                            + eqgs_idx * stride_logits_group
                            + tid
                        )
                        part_logits = fx.Float32(logits[logits_offset])
                        acc = acc + part_logits * weight

        query_idx = eqgs_idx // c_qgs
        if fx.const_expr(use_parallel_lds):
            if partition_group == zero_i:
                output_offset = (
                    batch_idx * stride_output_bs
                    + query_idx * stride_output_len
                    + kv_head_idx * stride_output_kv_head
                    + group_idx * stride_output_group_size
                    + output_element
                )
                output[output_offset] = acc.to(output_dtype)
        else:
            output_offset = (
                batch_idx * stride_output_bs
                + query_idx * stride_output_len
                + kv_head_idx * stride_output_kv_head
                + group_idx * stride_output_group_size
                + tid
            )
            output[output_offset] = acc.to(output_dtype)

    @flyc.jit
    def launch_pa_decode_ps_reduce_kernel(
        output: fx.Pointer,
        exp_sums: fx.Pointer,
        max_logits: fx.Pointer,
        logits: fx.Pointer,
        sink_token: fx.Pointer,
        stride_output_bs: fx.Int32,
        stride_output_len: fx.Int32,
        stride_output_kv_head: fx.Int32,
        stride_output_group_size: fx.Int32,
        stride_exp_sums_seq: fx.Int32,
        stride_exp_sums_head: fx.Int32,
        stride_exp_sums_part: fx.Int32,
        stride_logits_seq: fx.Int32,
        stride_logits_head: fx.Int32,
        stride_logits_part: fx.Int32,
        stride_logits_group: fx.Int32,
        query_seq_len: fx.Int32,
        query_group_size: fx.Int32,
        batch_size: fx.Int32,
        num_kv_heads: fx.Int32,
        reduce_info: fx.Pointer,
        stream: fx.Stream,
    ):
        pa_decode_ps_reduce_kernel(
            output,
            exp_sums,
            max_logits,
            logits,
            sink_token,
            stride_output_bs,
            stride_output_len,
            stride_output_kv_head,
            stride_output_group_size,
            stride_exp_sums_seq,
            stride_exp_sums_head,
            stride_exp_sums_part,
            stride_logits_seq,
            stride_logits_head,
            stride_logits_part,
            stride_logits_group,
            query_group_size,
            reduce_info,
        ).launch(
            grid=(batch_size, num_kv_heads, query_seq_len * query_group_size),
            block=tuple(block_shape),
            stream=stream,
        )

    return {
        "launch": launch_pa_decode_ps_reduce_kernel,
        "kernel": pa_decode_ps_reduce_kernel,
    }
