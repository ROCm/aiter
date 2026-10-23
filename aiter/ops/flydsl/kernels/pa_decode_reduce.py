# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Partitioned-softmax reduction kernel for FlyDSL paged attention."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx

from aiter.jit.utils.chip_info import get_gfx_runtime

from .tensor_shim import buf_base_i64, buf_copy_load, ptr_buf_tensor
from .utils import rcp_f32, udiv_const, urem_const

_DTYPE_MAP = {
    "f32": fx.Float32,
    "f16": fx.Float16,
    "bf16": fx.BFloat16,
}

WAVE = 64
# Above this partition bound the CTA reduces in LDS; at most DUAL_REGISTER
# live partitions still take its register branch.
REGISTER_PARTITIONS = 64
DUAL_REGISTER_PARTITIONS = 32
MAX_CTA_WAVES = 16


def _shuffle_offsets(parts):
    width = 1 if parts == 1 else 1 << (parts - 1).bit_length()
    return tuple(o for o in (32, 16, 8, 4, 2, 1) if o < width)


def _fast_div(value, magic):
    """floor(value / d) for magic = ceil(2**30 / d), exact while value * d <= 2**30."""
    return fx.Int32((fx.Int64(value) * fx.Int64(magic)) >> 30)


@lru_cache(maxsize=256)
def compile_pa_decode_ps_reduce(
    *,
    architecture: str | None = None,
    max_context_partition_num: int,
    head_size: int,
    output_dtype_str: str,
    logits_dtype_str: str,
    sink_dtype_str: str,
    use_sinks: bool,
    query_group_size: int,
    pairs: bool = False,
    parallel_groups: int = 1,
    num_xcds: int = 0,
):
    """Build PA decode's partitioned-softmax reducer.

    One CTA reduces one (sequence, KV head, query row). Logits are packed
    [KV heads, capacity, rows, D] rows whose span of NP partitions per sequence
    is at most INT32_MAX bytes: a per-sequence buffer bound zero-fills
    inactive partitions without reading adjacent sequences or NaN padding.

    NP up to the wave size builds a register-only CTA: lane p owns partition
    p's statistics and a CTA-uniform count branch selects one unrolled rung.
    Larger bounds build a dual-mode CTA of ``parallel_groups`` partition groups.
    Up to 32 live partitions, group 0 runs the register ladder; above, wave 0
    publishes the softmax weights to LDS, each group sums a contiguous chunk
    and group 0 adds the partials.

    ``pairs`` gives each lane two adjacent bf16/f16 elements (D % 128 == 0, a
    four-byte aligned base and even strides); output stores stay scalar.
    ``num_xcds`` > 0 launches a 1-D grid that gives each XCD a contiguous run
    of CTAs; the launch then needs ``rows_magic`` and ``seq_magic`` =
    ceil(2**30 / d) for d = rows and rows * KV heads, valid while
    batch * (rows * KV heads)**2 <= 2**30.
    Sinks are per-head zero-value logits: include them in the shared maximum
    and add their denominator mass once.
    """
    architecture = (architecture or get_gfx_runtime()).split(":", 1)[0]
    if architecture not in ("gfx942", "gfx950", "gfx1250"):
        raise NotImplementedError(f"PA decode reducer does not support {architecture}")
    wave = 32 if architecture == "gfx1250" else WAVE
    register_partitions = min(REGISTER_PARTITIONS, wave)
    max_cta_waves = MAX_CTA_WAVES * WAVE // wave
    partitions = max_context_partition_num
    if partitions < 1:
        raise ValueError(
            f"max_context_partition_num must be positive, got {partitions}"
        )
    if head_size <= 0 or head_size > 1024 or head_size % 64:
        raise ValueError(
            f"head_size must be a multiple of 64 in [64, 1024], got {head_size}"
        )
    for dtype_str in (output_dtype_str, logits_dtype_str, sink_dtype_str):
        if dtype_str not in _DTYPE_MAP:
            raise ValueError(f"Unsupported FlyDSL dtype: {dtype_str!r}")
    if query_group_size <= 0:
        raise ValueError(f"query_group_size must be positive, got {query_group_size}")
    if pairs and (head_size % 128 or logits_dtype_str not in ("bf16", "f16")):
        raise ValueError("pairs requires D % 128 == 0 and bf16/f16 logits")
    elems = 2 if pairs else 1
    group_waves = head_size // elems // wave
    dual = partitions > register_partitions
    groups = parallel_groups
    if groups < 1 or groups & (groups - 1):
        raise ValueError(f"parallel_groups must be a power of two, got {groups}")
    if groups > 1 and not dual:
        raise ValueError(f"parallel_groups > 1 requires NP > {register_partitions}")
    if groups * group_waves > max_cta_waves:
        raise ValueError(f"{groups} groups of {group_waves} waves exceed the CTA")
    if num_xcds < 0:
        raise ValueError(f"num_xcds must be nonnegative, got {num_xcds}")

    output_dtype = _DTYPE_MAP[output_dtype_str]
    logits_dtype = _DTYPE_MAP[logits_dtype_str]
    sink_dtype = _DTYPE_MAP[sink_dtype_str]
    log2e = 1.4426950408889634

    low_rungs = high_rungs = dual_rungs = lds_rungs = ()
    if not dual:
        block = [head_size // elems, 1, 1]
        # NP <= 8 is one rung; above, test c > 8 first, then the largest rung
        # in each half.
        low_rungs = (1, 2, 4, 8)
        high_rungs = tuple(r for r in (16, 32) if r < partitions) + (partitions,)
    else:
        block = [wave, group_waves * groups, 1]
        dual_rungs = (1, 2, 4, 8, 16, DUAL_REGISTER_PARTITIONS)
        # Group g owns partitions [g * R, (g + 1) * R) for the selected rung R.
        # Above 32 live partitions the first rung covers up to 64.
        per_group = -(-partitions // groups)
        rung = max(1, 2 * DUAL_REGISTER_PARTITIONS // groups)
        while rung < per_group:
            lds_rungs += (rung,)
            rung *= 2
        lds_rungs += (per_group,)
    # Bounds above 32 leave 1-2 live partitions in rungs whose first load LLVM
    # would otherwise hoist above the count branch; issuing rung 2's loads in
    # reverse keeps them inside it.
    reverse_pair = not dual and partitions > 32
    # Groups tile whole wave-sized weight chunks, so their slots stay in range.
    weight_chunks = -(-partitions // wave)

    @fx.struct
    class SharedStorage:
        weights: fx.Array[fx.Float32, weight_chunks * wave, 16]
        partials: fx.Array[fx.Float32, max(1, (groups - 1) * head_size), 16]

    @flyc.kernel(known_block_size=block)
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
        stride_exp_sums_head: fx.Int32,
        stride_exp_sums_part: fx.Int32,
        stride_logits_head: fx.Int32,
        stride_logits_part: fx.Int32,
        stride_logits_group: fx.Int32,
        reduce_info_ptr: fx.Pointer,
        query_rows: fx.Int32,
        num_kv_heads: fx.Int32,
        num_ctas: fx.Int32,
        rows_magic: fx.Int32,
        seq_magic: fx.Int32,
    ):
        if fx.const_expr(num_xcds > 0):
            # Hardware deals linear workgroup ids round-robin over XCDs. Give
            # each XCD a contiguous run of (sequence, KV head, row) so rows
            # whose statistics share cache lines share an L2.
            linear = fx.block_idx.x
            xcd = urem_const(linear, num_xcds)
            per_xcd = udiv_const(num_ctas, num_xcds)
            extra = urem_const(num_ctas, num_xcds)
            logical = (
                xcd * per_xcd
                + (xcd < extra).select(xcd, extra)
                + udiv_const(linear, num_xcds)
            )
            batch_idx = _fast_div(logical, seq_magic)
            inner = logical - batch_idx * (query_rows * num_kv_heads)
            kv_head_idx = _fast_div(inner, rows_magic)
            row_idx = inner - kv_head_idx * query_rows
        else:
            batch_idx = fx.block_idx.x
            kv_head_idx = fx.block_idx.y
            row_idx = fx.block_idx.z

        output = fx.recast_iter(output_dtype, output_ptr)
        exp_sums = fx.recast_iter(fx.Float32, exp_sums_ptr)
        max_logits = fx.recast_iter(fx.Float32, max_logits_ptr)
        reduce_info = fx.recast_iter(fx.Int32, reduce_info_ptr)

        zero_f = fx.Float32(0.0)
        one_f = fx.Float32(1.0)
        neg_inf = fx.Float32(float("-inf"))
        c_log2e = fx.Float32(log2e)
        zero_i = fx.Int32(0)
        c_wave = fx.Int32(wave)
        lane = fx.thread_idx.x & fx.Int32(wave - 1)
        group_idx = urem_const(row_idx, query_group_size)
        query_idx = udiv_const(row_idx, query_group_size)
        output_base = (
            batch_idx * stride_output_bs
            + query_idx * stride_output_len
            + kv_head_idx * stride_output_kv_head
            + group_idx * stride_output_group_size
        )
        if fx.const_expr(use_sinks):
            sink_token = fx.recast_iter(sink_dtype, sink_token_ptr)
            sink_value = fx.Float32(
                sink_token[kv_head_idx * fx.Int32(query_group_size) + group_idx]
            )

        first_part = fx.Int32(reduce_info[batch_idx * 2])
        part_count = fx.Int32(reduce_info[batch_idx * 2 + 1])
        # Empty sequences may point past packed storage; clamped loads need
        # slot 0. Every inactive value is masked before arithmetic.
        first_part = (part_count > zero_i).select(first_part, zero_i)
        stats_seq_offset = first_part * stride_exp_sums_part
        # Rebase in i64 before narrowing: packed storage can exceed 2 GiB. A
        # count of 0 gives a zero-sized descriptor that returns zeros.
        logits_item_bytes = logits_dtype.width // 8
        logits_buffer = ptr_buf_tensor(
            buf_base_i64(logits_ptr)
            + (
                fx.Int64(first_part) * fx.Int64(stride_logits_part)
                + fx.Int64(kv_head_idx) * fx.Int64(stride_logits_head)
            )
            * logits_item_bytes,
            logits_dtype,
            unit_elems=elems,
            unit_stride=1,
            num_records_bytes=fx.Int64(part_count)
            * fx.Int64(stride_logits_part)
            * logits_item_bytes,
        )

        def _zero():
            if fx.const_expr(pairs):
                return fx.full(2, 0.0, fx.Float32)
            return zero_f

        def _load_logits(part, unit):
            return buf_copy_load(
                logits_buffer,
                part * stride_logits_part
                + row_idx * stride_logits_group
                + unit * fx.Int32(elems),
                logits_dtype,
                unit_elems=elems,
            )

        def _to_f32(value):
            if fx.const_expr(pairs):
                return value.to(fx.Float32)
            return fx.Float32(value)

        def _store(acc, unit):
            # Output rows need only element alignment, so pairs store scalars.
            offset = output_base + unit * fx.Int32(elems)
            if fx.const_expr(pairs):
                output[offset] = acc[0].to(output_dtype)
                output[offset + fx.Int32(1)] = acc[1].to(output_dtype)
            else:
                output[offset] = acc.to(output_dtype)

        def _lane_stats(chunk):
            # Lane l owns partition l + wave * chunk; part_count <= NP.
            part = lane + fx.Int32(chunk * wave)
            live = part < part_count
            offset = (
                stats_seq_offset
                + kv_head_idx * stride_exp_sums_head
                + live.select(part, zero_i) * stride_exp_sums_part
                + row_idx
            )
            part_sum = live.select(fx.Float32(exp_sums[offset]), zero_f)
            part_max = live.select(fx.Float32(max_logits[offset]), neg_inf)
            return part_sum, part_max

        def _softmax_weights(sums, maxes, offsets):
            lane_max = maxes[0]
            for part_max in maxes[1:]:
                lane_max = lane_max.maximumf(part_max)
            for offset in offsets:
                lane_max = lane_max.maximumf(
                    lane_max.shuffle_xor(fx.Int32(offset), c_wave)
                )
            if fx.const_expr(use_sinks):
                lane_max = lane_max.maximumf(sink_value)
            safe_max = (lane_max > neg_inf).select(lane_max, zero_f)
            scaled = []
            for part_sum, part_max in zip(sums, maxes):
                scale = (part_max > neg_inf).select(
                    fx.exp2((part_max - safe_max) * c_log2e, fastmath="fast"), zero_f
                )
                scaled.append(part_sum * scale)
            exp_sum = scaled[0]
            for value in scaled[1:]:
                exp_sum = exp_sum + value
            for offset in offsets:
                exp_sum = fx.Float32(
                    exp_sum.addf(
                        exp_sum.shuffle_xor(fx.Int32(offset), c_wave),
                        fastmath="fast",
                    )
                )
            if fx.const_expr(use_sinks):
                # +inf sinks have unit mass at the shared max and suppress KV;
                # -inf contributes zero, including for an empty request.
                shift = (sink_value == safe_max).select(zero_f, sink_value - safe_max)
                exp_sum = exp_sum + fx.exp2(shift * c_log2e, fastmath="fast")
            inv_exp_sum = fx.Float32(rcp_f32((exp_sum > zero_f).select(exp_sum, one_f)))
            return [value * inv_exp_sum for value in scaled]

        def _reduce_register(rung, unit):
            part_sum, part_max = _lane_stats(0)
            order = range(rung)
            if fx.const_expr(reverse_pair and rung == 2):
                order = (1, 0)
            # Loads stay in the logits dtype until accumulation, so a rung
            # holds half the registers of its FP32 values.
            logits = [None] * rung
            for part in order:
                logits[part] = _load_logits(fx.Int32(part), unit)
            # Wave32 reductions broadcast statistics to every output lane,
            # including lanes without a live partition.
            offsets = _shuffle_offsets(wave if architecture == "gfx1250" else rung)
            (weight,) = _softmax_weights([part_sum], [part_max], offsets)
            acc = _zero()
            for part in fx.range_constexpr(rung):
                part_weight = fx.gpu.shuffle_idx(weight, part, wave)
                # Preserve increasing-part order and separate multiply/add rounding.
                acc = acc + _to_f32(logits[part]) * part_weight
            return acc

        def _top_first(rungs, reduce, unit, scale=1):
            # CTA-uniform count: the largest rung is the first test and the
            # smallest the fall-through. Branches carry the same acc type.
            acc = _zero()
            if fx.const_expr(len(rungs) == 1):
                acc = reduce(rungs[0], unit)
            else:
                if part_count > fx.Int32(rungs[-2] * scale):
                    acc = reduce(rungs[-1], unit)
                else:
                    acc = _top_first(rungs[:-1], reduce, unit, scale)
            return acc

        def _bottom_first(rungs, reduce, unit):
            acc = _zero()
            if fx.const_expr(len(rungs) == 1):
                acc = reduce(rungs[0], unit)
            else:
                if part_count <= fx.Int32(rungs[0]):
                    acc = reduce(rungs[0], unit)
                else:
                    acc = _bottom_first(rungs[1:], reduce, unit)
            return acc

        if fx.const_expr(not dual):
            # Select one unrolled rung, so loads past it are skipped rather
            # than masked.
            unit = fx.thread_idx.x
            acc = _zero()
            if fx.const_expr(partitions <= 8):
                acc = _reduce_register(partitions, unit)
            else:
                if part_count > 8:
                    acc = _top_first(high_rungs, _reduce_register, unit)
                else:
                    acc = _top_first(low_rungs, _reduce_register, unit)
            _store(acc, unit)
        else:
            worker = fx.thread_idx.y
            partition_group = udiv_const(worker, group_waves)
            unit = urem_const(worker, group_waves) * c_wave + lane

            def _reduce_lds():
                # Every thread reaches each barrier; the count is CTA-uniform.
                storage = fx.SharedAllocator().allocate(SharedStorage).peek()
                weights = storage.weights
                partials = storage.partials.view(
                    fx.make_layout((max(1, groups - 1), head_size), (head_size, 1))
                )
                if worker == zero_i:
                    sums, maxes = [], []
                    for chunk in fx.range_constexpr(weight_chunks):
                        part_sum, part_max = _lane_stats(chunk)
                        sums.append(part_sum)
                        maxes.append(part_max)
                    chunk_weights = _softmax_weights(
                        sums, maxes, _shuffle_offsets(wave)
                    )
                    for chunk in fx.range_constexpr(weight_chunks):
                        weights[lane + fx.Int32(chunk * wave)] = chunk_weights[chunk]
                fx.gpu.barrier()

                def _group_sum(rung, unit):
                    # Inactive slots have zero weight and load zero past the
                    # sequence bound, so they add exactly +0.
                    group_acc = _zero()
                    begin = partition_group * fx.Int32(rung)
                    if begin < part_count:
                        for local in fx.range_constexpr(rung):
                            part = begin + fx.Int32(local)
                            weight = fx.Float32(weights[part])
                            group_acc = (
                                group_acc + _to_f32(_load_logits(part, unit)) * weight
                            )
                    return group_acc

                acc = _top_first(lds_rungs, _group_sum, unit, groups)
                if fx.const_expr(groups > 1):
                    # Idle groups publish +0, so the combine needs no range checks.
                    if partition_group > zero_i:
                        if fx.const_expr(pairs):
                            partials[partition_group - 1, unit * 2] = acc[0]
                            partials[partition_group - 1, unit * 2 + 1] = acc[1]
                        else:
                            partials[partition_group - 1, unit] = acc
                    fx.gpu.barrier()
                    if partition_group == zero_i:
                        for other in fx.range_constexpr(1, groups):
                            if fx.const_expr(pairs):
                                acc = acc + fx.Vector.from_elements(
                                    [
                                        partials[other - 1, unit * 2],
                                        partials[other - 1, unit * 2 + 1],
                                    ],
                                    dtype=fx.Float32,
                                )
                            else:
                                acc = acc + fx.Float32(partials[other - 1, unit])
                return acc

            if part_count <= fx.Int32(DUAL_REGISTER_PARTITIONS):
                # Few live partitions skip LDS and both barriers.
                if partition_group == zero_i:
                    register_acc = _bottom_first(dual_rungs, _reduce_register, unit)
                    _store(register_acc, unit)
            else:
                lds_acc = _reduce_lds()
                if partition_group == zero_i:
                    _store(lds_acc, unit)

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
        stride_exp_sums_head: fx.Int32,
        stride_exp_sums_part: fx.Int32,
        stride_logits_head: fx.Int32,
        stride_logits_part: fx.Int32,
        stride_logits_group: fx.Int32,
        query_seq_len: fx.Int32,
        batch_size: fx.Int32,
        num_kv_heads: fx.Int32,
        reduce_info: fx.Pointer,
        rows_magic: fx.Int32,
        seq_magic: fx.Int32,
        stream: fx.Stream,
    ):
        query_rows = query_seq_len * fx.Int32(query_group_size)
        num_ctas = batch_size * num_kv_heads * query_rows
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
            stride_exp_sums_head,
            stride_exp_sums_part,
            stride_logits_head,
            stride_logits_part,
            stride_logits_group,
            reduce_info,
            query_rows,
            num_kv_heads,
            num_ctas,
            rows_magic,
            seq_magic,
        ).launch(
            grid=(
                (num_ctas, 1, 1)
                if num_xcds > 0
                else (batch_size, num_kv_heads, query_rows)
            ),
            block=tuple(block),
            stream=stream,
        )

    return {
        "launch": launch_pa_decode_ps_reduce_kernel,
        "kernel": pa_decode_ps_reduce_kernel,
    }
