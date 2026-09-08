# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Partitioned-softmax reduction kernel for FlyDSL paged attention."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr.typing import T


@lru_cache(maxsize=256)
def compile_pa_decode_ps_reduce(
    *,
    max_context_partition_num: int,
    head_size: int,
    output_dtype_str: str,
    logits_dtype_str: str,
    sink_dtype_str: str,
    use_sinks: bool,
):
    """Build the partitioned-softmax reduction used by ``pa_decode``.

    Each wave reduces the partition statistics independently, then reuses the
    per-partition weights to combine one output element per thread.  The host
    API limits the partition count to one wave, so the LDS path from the more
    general Gluon implementation is unnecessary here.
    """
    if not 1 <= max_context_partition_num <= 64:
        raise ValueError(
            "max_context_partition_num must be in [1, 64], "
            f"got {max_context_partition_num}"
        )
    if head_size <= 0 or head_size > 1024 or head_size % 64:
        raise ValueError(
            f"head_size must be a multiple of 64 in [64, 1024], got {head_size}"
        )

    dtype_map = {
        "f32": fx.Float32,
        "f16": fx.Float16,
        "bf16": fx.BFloat16,
    }
    try:
        output_dtype = dtype_map[output_dtype_str]
        logits_dtype = dtype_map[logits_dtype_str]
        sink_dtype = dtype_map[sink_dtype_str]
    except KeyError as exc:
        raise ValueError(f"Unsupported FlyDSL dtype: {exc.args[0]!r}") from exc

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

    @flyc.kernel(known_block_size=[head_size, 1, 1])
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
    ):
        tid = fx.thread_idx.x
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

        lane_in_range = lane < c_part_num
        lane_in_reduce = lane < c_reduce_width
        part_sum = zero_f
        part_max = neg_inf
        if lane_in_reduce:
            part_idx = lane_in_range.select(lane, zero_i)
            stats_offset = (
                batch_idx * stride_exp_sums_seq
                + kv_head_idx * stride_exp_sums_head
                + part_idx * stride_exp_sums_part
                + eqgs_idx
            )
            loaded_sum = fx.Float32(exp_sums[stats_offset])
            loaded_max = fx.Float32(max_logits[stats_offset])
            part_sum = lane_in_range.select(loaded_sum, zero_f)
            part_max = lane_in_range.select(loaded_max, neg_inf)

        global_max = _wave_reduce_max(part_max)
        safe_global_max = (global_max > neg_inf).select(global_max, zero_f)
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
        safe_global_exp_sum = (global_exp_sum > zero_f).select(global_exp_sum, one_f)
        weight_local = scaled_sum / safe_global_exp_sum
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
                batch_idx * stride_logits_seq
                + kv_head_idx * stride_logits_head
                + c_part_idx * stride_logits_part
                + eqgs_idx * stride_logits_group
                + tid
            )
            part_logits = fx.Float32(logits[logits_offset])
            acc = acc + part_logits * weight

        query_idx = eqgs_idx // c_qgs
        group_idx = eqgs_idx % c_qgs
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
        ).launch(
            grid=(batch_size, num_kv_heads, query_seq_len * query_group_size),
            block=(head_size, 1, 1),
            stream=stream,
        )

    return {
        "launch": launch_pa_decode_ps_reduce_kernel,
        "kernel": pa_decode_ps_reduce_kernel,
    }
