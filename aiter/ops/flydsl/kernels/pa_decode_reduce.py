# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Partitioned-softmax reduction kernel for FlyDSL paged attention."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T

from .tensor_shim import buf_base_i64, buf_copy_load, ptr_buf_tensor
from .utils import rcp_f32, udiv_const, urem_const

# Static scheduling limit; work plans use their device's CU count instead.
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
    use_work_plan: bool = False,
) -> None:
    # The plan and launch wrappers validate the device-specific CU bound.
    if max_context_partition_num < 1 or (
        not use_work_plan and max_context_partition_num > MAX_CONTEXT_PARTITIONS
    ):
        bound = "positive" if use_work_plan else f"in [1, {MAX_CONTEXT_PARTITIONS}]"
        raise ValueError(
            f"max_context_partition_num must be {bound}, "
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
    use_work_plan: bool = False,
) -> bool:
    """Return whether the FlyDSL reducer supports a dispatch configuration."""
    try:
        _validate_pa_decode_ps_reduce_config(
            max_context_partition_num=max_context_partition_num,
            head_size=head_size,
            output_dtype_str=output_dtype_str,
            logits_dtype_str=logits_dtype_str,
            sink_dtype_str=sink_dtype_str,
            use_work_plan=use_work_plan,
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
    query_group_size: int | None = None,
    bounded_plan_logits: bool = False,
    vectorize_plan_logits: bool = False,
):
    """Build the partitioned-softmax reduction used by ``pa_decode``.

    Counts up to one wave use one partition per lane and one output element per
    thread.  For D=128 and larger counts, a 2-D workgroup materializes weights
    once in LDS and splits each output element's partition chain over several
    waves.  Other head sizes retain the register-only lane-striped fallback.
    A sink is a per-query-head zero-value logit: include it in the shared max
    for stability and add its mass to the denominator once after summing KV.
    ``query_group_size=None`` retains runtime GQA addressing; a positive
    specialization constant must match the GQA passed to the launch wrapper.
    ``bounded_plan_logits`` requires packed, contiguous logits rows and a
    positive NP-part byte span no larger than INT32_MAX. A per-sequence buffer
    bound then zero-fills inactive parts without reading adjacent sequences.
    ``vectorize_plan_logits`` additionally requires D=128, bf16/f16 logits,
    a four-byte-aligned logits base and even head/part/group strides. One wave
    then assigns two consecutive output elements to each lane, sharing the
    unchanged partition weights. Output stores retain scalar alignment.
    """
    _validate_pa_decode_ps_reduce_config(
        max_context_partition_num=max_context_partition_num,
        head_size=head_size,
        output_dtype_str=output_dtype_str,
        logits_dtype_str=logits_dtype_str,
        sink_dtype_str=sink_dtype_str,
        use_work_plan=use_work_plan,
    )
    if query_group_size is not None and query_group_size <= 0:
        raise ValueError("query_group_size must be positive when specified")
    if bounded_plan_logits and (not use_work_plan or max_context_partition_num > 64):
        raise ValueError("bounded_plan_logits requires a planned NP<=64 reducer")
    if vectorize_plan_logits and (
        not bounded_plan_logits
        or head_size != 128
        or logits_dtype_str not in ("bf16", "f16")
    ):
        raise ValueError(
            "vectorize_plan_logits requires bounded planned D=128 bf16/f16 logits"
        )
    static_query_group_size = query_group_size
    planned_elements_per_thread = 2 if vectorize_plan_logits else 1

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
        [warp_size, worker_waves, 1]
        if use_parallel_lds
        else [head_size // planned_elements_per_thread, 1, 1]
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
            # Empty sequences may point one past the packed allocation. The
            # <=64 path uses clamped loads, so give them a valid base slot;
            # every loaded value is masked before it contributes to a result.
            first_part = (c_part_num > zero_i).select(first_part, zero_i)
            stats_seq_offset = first_part * stride_exp_sums_part
            logits_seq_offset = first_part * stride_logits_part
        else:
            stats_seq_offset = batch_idx * stride_exp_sums_seq
            logits_seq_offset = batch_idx * stride_logits_seq
        if fx.const_expr(bounded_plan_logits):
            # Rebase before narrowing offsets: the whole packed workspace may
            # exceed 2 GiB even when this sequence's NP-part span is small.
            logits_item_bytes = logits_dtype.width // 8
            planned_logits_base = (
                buf_base_i64(logits_ptr)
                + (
                    fx.Int64(first_part) * fx.Int64(stride_logits_part)
                    + fx.Int64(kv_head_idx) * fx.Int64(stride_logits_head)
                )
                * logits_item_bytes
            )
            planned_logits_bytes = (
                fx.Int64(c_part_num) * fx.Int64(stride_logits_part) * logits_item_bytes
            )
            planned_logits_buffer = ptr_buf_tensor(
                planned_logits_base,
                logits_dtype,
                unit_elems=planned_elements_per_thread,
                unit_stride=1,
                num_records_bytes=planned_logits_bytes,
            )
        c_reduce_width = fx.Int32(reduce_width)
        c_four = fx.Int32(4)
        lane = tid & c_wave_mask
        if fx.const_expr(static_query_group_size is not None):
            c_qgs = fx.Int32(static_query_group_size)
            group_idx = urem_const(eqgs_idx, static_query_group_size)
        else:
            c_qgs = query_group_size
            group_idx = eqgs_idx % c_qgs

        def _wave_reduce_max(value, offsets=reduce_shuffle_offsets):
            reduced = value
            for offset in offsets:
                reduced = reduced.maximumf(
                    reduced.shuffle_xor(fx.Int32(offset), c_warp_size)
                )
            return reduced

        def _wave_reduce_sum(value, offsets=reduce_shuffle_offsets):
            reduced = value
            for offset in offsets:
                reduced = fx.Float32(
                    reduced.addf(
                        reduced.shuffle_xor(fx.Int32(offset), c_warp_size),
                        fastmath="fast",
                    )
                )
            return reduced

        def _sink_exp(sink_value, safe_max):
            # +inf sinks have unit mass at the shared max and suppress KV;
            # -inf contributes zero, including for an entirely empty request.
            shift = (sink_value == safe_max).select(zero_f, sink_value - safe_max)
            return fx.exp2(shift * c_log2e, fastmath="fast")

        def _reduce_planned_wave(part_count, shuffle_offsets):
            # Both widths use the same packed-scratch and softmax rules.
            # Invalid lanes read this sequence's first partition; count=0
            # has a safe global slot 0 base. Select out NaNs before math.
            planned_lane_in_range = lane < c_part_num
            planned_stats_part_idx = planned_lane_in_range.select(lane, zero_i)
            planned_stats_offset = (
                stats_seq_offset
                + kv_head_idx * stride_exp_sums_head
                + planned_stats_part_idx * stride_exp_sums_part
                + eqgs_idx
            )
            planned_loaded_sum = fx.Float32(exp_sums[planned_stats_offset])
            planned_loaded_max = fx.Float32(max_logits[planned_stats_offset])
            planned_part_sum = planned_lane_in_range.select(planned_loaded_sum, zero_f)
            planned_part_max = planned_lane_in_range.select(planned_loaded_max, neg_inf)

            # Preload before softmax so independent loads can overlap. The
            # caller's uniform branch encloses these loads, not just the FMAs.
            planned_logits = []
            for planned_part in fx.range_constexpr(part_count):
                planned_part_idx = fx.Int32(planned_part)
                if fx.const_expr(bounded_plan_logits):
                    # The descriptor ends at count*part_stride, before the
                    # next sequence or NaN padding. Even count=0 is a valid
                    # zero-sized resource; raw buffer loads return zero.
                    planned_logits_offset = (
                        planned_part_idx * stride_logits_part
                        + eqgs_idx * stride_logits_group
                        + tid * fx.Int32(planned_elements_per_thread)
                    )
                    # With unit_stride=1 this remains an element offset, not
                    # a vector-unit index. A D=128 pair ends within its row;
                    # even count=0 suppresses the complete 32-bit buffer load.
                    planned_loaded_logits = buf_copy_load(
                        planned_logits_buffer,
                        planned_logits_offset,
                        logits_dtype,
                        unit_elems=planned_elements_per_thread,
                    )
                    if fx.const_expr(vectorize_plan_logits):
                        # Keep the prefetched pair packed in 32 bits across
                        # softmax; widen only the part being accumulated below.
                        planned_logits.append(planned_loaded_logits)
                    else:
                        planned_logits.append(fx.Float32(planned_loaded_logits))
                else:
                    planned_part_in_range = planned_part_idx < c_part_num
                    planned_safe_part_idx = planned_part_in_range.select(
                        planned_part_idx, zero_i
                    )
                    planned_logits_offset = (
                        logits_seq_offset
                        + kv_head_idx * stride_logits_head
                        + planned_safe_part_idx * stride_logits_part
                        + eqgs_idx * stride_logits_group
                        + tid
                    )
                    planned_loaded_logits = fx.Float32(logits[planned_logits_offset])
                    planned_logits.append(
                        planned_part_in_range.select(planned_loaded_logits, zero_f)
                    )

            planned_global_max = _wave_reduce_max(planned_part_max, shuffle_offsets)
            if fx.const_expr(use_sinks):
                planned_sink_value = fx.Float32(
                    sink_token[kv_head_idx * c_qgs + group_idx]
                )
                planned_global_max = planned_global_max.maximumf(planned_sink_value)
            planned_safe_max = (planned_global_max > neg_inf).select(
                planned_global_max, zero_f
            )
            planned_part_scale = (planned_part_max > neg_inf).select(
                fx.exp2(
                    (planned_part_max - planned_safe_max) * c_log2e,
                    fastmath="fast",
                ),
                zero_f,
            )
            planned_scaled_sum = planned_part_sum * planned_part_scale
            planned_exp_sum = _wave_reduce_sum(planned_scaled_sum, shuffle_offsets)
            if fx.const_expr(use_sinks):
                planned_exp_sum = planned_exp_sum + _sink_exp(
                    planned_sink_value, planned_safe_max
                )
            planned_safe_exp_sum = (planned_exp_sum > zero_f).select(
                planned_exp_sum, one_f
            )
            planned_inv_exp_sum = fx.Float32(rcp_f32(planned_safe_exp_sum))
            planned_weight_local_i32 = (
                planned_scaled_sum * planned_inv_exp_sum
            ).bitcast(fx.Int32)

            if fx.const_expr(vectorize_plan_logits):
                planned_acc = fx.Vector.filled(2, 0.0, fx.Float32)
            else:
                planned_acc = zero_f
            for planned_acc_part in fx.range_constexpr(part_count):
                planned_acc_part_idx = fx.Int32(planned_acc_part)
                planned_weight_i32 = fx.Int32(
                    fx.rocdl.ds_bpermute(
                        T.i32,
                        planned_acc_part_idx * c_four,
                        planned_weight_local_i32,
                    )
                )
                planned_weight = planned_weight_i32.bitcast(fx.Float32)
                if fx.const_expr(vectorize_plan_logits):
                    planned_part_logits = planned_logits[planned_acc_part].to(
                        fx.Float32
                    )
                else:
                    planned_part_logits = planned_logits[planned_acc_part]
                # Elementwise mul then add, with the same increasing-part
                # order as the scalar path: no FMA or reassociation hint.
                planned_acc = planned_acc + planned_part_logits * planned_weight
            return planned_acc

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
                if fx.const_expr(use_sinks):
                    sink_value = fx.Float32(sink_token[kv_head_idx * c_qgs + group_idx])
                    global_max = global_max.maximumf(sink_value)
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
                    sink_scale = _sink_exp(sink_value, safe_global_max)
                    global_exp_sum = global_exp_sum + sink_scale
                safe_global_exp_sum = (global_exp_sum > zero_f).select(
                    global_exp_sum, one_f
                )
                parallel_inv_exp_sum = fx.Float32(rcp_f32(safe_global_exp_sum))

                for chunk_idx in fx.range_constexpr(partitions_per_lane):
                    chunk_base = chunk_idx * warp_size
                    chunk_size = min(warp_size, max_context_partition_num - chunk_base)
                    part_idx = lane + fx.Int32(chunk_base)
                    if fx.const_expr(use_work_plan):
                        lane_in_range = (lane < fx.Int32(chunk_size)) & (
                            part_idx < c_part_num
                        )
                        if lane_in_range:
                            weight = scaled_sums[chunk_idx] * parallel_inv_exp_sum
                            lds_weights[part_idx] = weight
                    else:
                        weight = scaled_sums[chunk_idx] * parallel_inv_exp_sum
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
            if fx.const_expr(use_work_plan):
                # Both arms of the dynamic short-count branch, and its
                # initial carried value, must have the same scalar/vector type.
                if fx.const_expr(vectorize_plan_logits):
                    acc = fx.Vector.filled(2, 0.0, fx.Float32)
                else:
                    acc = zero_f
                if fx.const_expr(max_context_partition_num > 8):
                    # The count is uniform across this CTA. Keep the complete
                    # short reduction under the branch so large NP does not
                    # force short requests to preload and accumulate padding.
                    # Small NP stays straight-line: the branch costs more than
                    # the few clamped loads it removes on those specializations.
                    if c_part_num <= c_four:
                        acc = _reduce_planned_wave(4, (2, 1))
                    elif c_part_num <= fx.Int32(8):
                        # Medium-count requests need only the first eight
                        # lanes/partials, even when NP reserves up to 64.
                        # Keep the preload, softmax and accumulation inside
                        # this uniform branch, including empty-row handling.
                        acc = _reduce_planned_wave(8, (4, 2, 1))
                    else:
                        acc = _reduce_planned_wave(
                            max_context_partition_num, reduce_shuffle_offsets
                        )
                else:
                    acc = _reduce_planned_wave(
                        max_context_partition_num, reduce_shuffle_offsets
                    )
            else:
                if fx.const_expr(max_context_partition_num == reduce_width):
                    # Preserve the static mapping: exact powers of two have no
                    # inactive lanes inside their reduction subgroup, while
                    # partial subgroups use EXEC-masked loads.
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
                if fx.const_expr(use_sinks):
                    sink_value = fx.Float32(sink_token[kv_head_idx * c_qgs + group_idx])
                    global_max = global_max.maximumf(sink_value)
                safe_global_max = (global_max > neg_inf).select(global_max, zero_f)
                part_scale = (part_max > neg_inf).select(
                    fx.exp2((part_max - safe_global_max) * c_log2e, fastmath="fast"),
                    zero_f,
                )
                scaled_sum = part_sum * part_scale
                global_exp_sum = _wave_reduce_sum(scaled_sum)
                if fx.const_expr(use_sinks):
                    sink_scale = _sink_exp(sink_value, safe_global_max)
                    global_exp_sum = global_exp_sum + sink_scale
                safe_global_exp_sum = (global_exp_sum > zero_f).select(
                    global_exp_sum, one_f
                )
                wave_inv_exp_sum = fx.Float32(rcp_f32(safe_global_exp_sum))
                weight_local = scaled_sum * wave_inv_exp_sum
                weight_local_i32 = weight_local.bitcast(fx.Int32)

                acc = zero_f
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
            if fx.const_expr(use_sinks):
                sink_value = fx.Float32(sink_token[kv_head_idx * c_qgs + group_idx])
                global_max = global_max.maximumf(sink_value)
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
                sink_scale = _sink_exp(sink_value, safe_global_max)
                global_exp_sum = global_exp_sum + sink_scale
            safe_global_exp_sum = (global_exp_sum > zero_f).select(
                global_exp_sum, one_f
            )
            striped_inv_exp_sum = fx.Float32(rcp_f32(safe_global_exp_sum))

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
                            scaled_sums[chunk_idx] * striped_inv_exp_sum
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
                        scaled_sums[chunk_idx] * striped_inv_exp_sum
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

        if fx.const_expr(static_query_group_size is not None):
            query_idx = udiv_const(eqgs_idx, static_query_group_size)
        else:
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
                + tid * fx.Int32(planned_elements_per_thread)
            )
            if fx.const_expr(vectorize_plan_logits):
                # Do not require a four-byte-aligned output pointer/row base.
                output[output_offset] = acc[0].to(output_dtype)
                output[output_offset + fx.Int32(1)] = acc[1].to(output_dtype)
            else:
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
