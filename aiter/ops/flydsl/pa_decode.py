# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Readable tile-programming reference for paged-attention fp8 decode.

K/V are fp8 e4m3 (FNUZ on gfx942, OCP on gfx950) fed straight into
``mfma_f32_16x16x32_fp8_fp8``; Q (bf16/f16) and the softmax probabilities P are
quantized to fp8 too. Scales fold out of the matmuls (q/key scale into the QK
score, value scale + 1/FP8_MAX into the epilogue); softmax max/sum stay f32.
``key_scale``/``value_scale`` are either a ``[1]`` per-tensor scalar or a
``[num_blocks, num_kv_heads, block_size, 1]`` per-token tensor.

``block_size`` (16/64) and ``head_dim`` (multiple of 64) are compile-time
constants. Layouts are logical, not production's preshuffle.

* ``query``        [num_seqs, num_q_heads, head_dim]  f16/bf16 (head_dim contiguous)
* ``key_cache``    [num_blocks, num_kv_heads, head_dim//16, block_size, 16]  fp8
* ``value_cache``  [num_blocks, num_kv_heads, block_size//16, head_dim, 16] (trans_v)
                   or [num_blocks, num_kv_heads, head_dim, block_size] (plain), by rank
* ``block_tables`` [num_seqs, max_blocks_per_seq]  int32
* ``context_lengths`` [num_seqs]  int32
* ``output``       [num_seqs, num_q_heads, head_dim]  same dtype as query

One CTA (4 waves) per (seq, kv_head) runs a flash-style online softmax over
256-token blocks; the 4 waves split tokens for Q.KT and head-dim for P.V, with
an LDS round-trip on P transposing ownership between the two MMAs.
"""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.pa_decode_tile import KV_COMPUTE_BLOCK, compile_pa_decode_tile
from .kernels.tensor_shim import _run_compiled, ptr_arg
from .kernels.utils import cdiv


def get_recommended_splits(
    num_sequences: int,
    num_kv_heads: int,
    split_kv_blocks: int = 1,
) -> int:
    props = torch.cuda.get_device_properties(torch.device("cuda"))
    num_sm = props.multi_processor_count * 2
    denom = max(1, num_sequences * num_kv_heads * split_kv_blocks)
    n = cdiv(num_sm, denom) * split_kv_blocks
    return max(4, min(n, 8))


def _flydsl_dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "f32"
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise ValueError(f"Unsupported FlyDSL dtype: {dtype!r}")


def _flydsl_pointer_dtype(dtype: torch.dtype):
    return {
        torch.float32: fx.Float32,
        torch.float16: fx.Float16,
        torch.bfloat16: fx.BFloat16,
    }[dtype]


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


def launch_pa_decode_ps_reduce(
    output: torch.Tensor,
    exp_sums: torch.Tensor,
    max_logits: torch.Tensor,
    logits: torch.Tensor,
    sink_token: torch.Tensor | None,
    stride_output_bs: int,
    stride_output_len: int,
    stride_output_kv_head: int,
    stride_output_group_size: int,
    stride_exp_sums_seq: int,
    stride_exp_sums_head: int,
    stride_exp_sums_part: int,
    stride_logits_seq: int,
    stride_logits_head: int,
    stride_logits_part: int,
    stride_logits_group: int,
    *,
    query_seq_len: int,
    query_group_size: int,
    head_size: int,
    context_partition_num: int,
    stream: torch.cuda.Stream,
) -> None:
    if context_partition_num > 64:
        raise ImportError("FlyDSL pa_decode reduce supports at most 64 partitions")
    use_sinks = sink_token is not None
    compiled = compile_pa_decode_ps_reduce(
        max_context_partition_num=context_partition_num,
        head_size=head_size,
        output_dtype_str=_flydsl_dtype_str(output.dtype),
        logits_dtype_str=_flydsl_dtype_str(logits.dtype),
        sink_dtype_str=_flydsl_dtype_str(
            output.dtype if sink_token is None else sink_token.dtype
        ),
        use_sinks=use_sinks,
    )
    sink_ptr = (
        ptr_arg(sink_token, _flydsl_pointer_dtype(sink_token.dtype))
        if use_sinks
        else flyc.from_c_void_p(_flydsl_pointer_dtype(output.dtype), 0)
    )
    _run_compiled(
        compiled["launch"],
        ptr_arg(output, _flydsl_pointer_dtype(output.dtype)),
        ptr_arg(exp_sums, fx.Float32),
        ptr_arg(max_logits, fx.Float32),
        ptr_arg(logits, _flydsl_pointer_dtype(logits.dtype)),
        sink_ptr,
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
        query_seq_len,
        query_group_size,
        output.shape[0],
        output.shape[2],
        stream,
    )


def pa_decode(
    output: torch.Tensor,  # [num_seqs * query_length, num_query_heads, head_size]
    query: torch.Tensor,  # [num_seqs * query_length, num_query_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, kv_block_size] or [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    context_lengths: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    softmax_scale: float,
    query_length: int,
    max_context_partition_num: int,
    context_partition_size: int = 256,
    compute_type: torch.dtype = torch.bfloat16,
    query_scale: torch.Tensor = None,  # [num_seqs * query_length, num_query_heads, 1] or [1]
    key_scale: torch.Tensor = None,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale: torch.Tensor = None,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    exp_sums: torch.Tensor = None,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
    max_logits: torch.Tensor = None,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
    temporary_output: torch.Tensor = None,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size, head_size]
    alibi_slopes: torch.Tensor = None,
    sinks: torch.Tensor = None,
    sliding_window: int = 0,
    ps: bool = True,
) -> None:
    """FlyDSL paged-attention fp8 decode.

    The call signature and intermediate-buffer layouts follow the shared
    aiter paged-attention decode API. This kernel currently supports FP8 K/V caches,
    BF16/FP16 queries, a 256-token context partition, and block sizes 16/64.
    ALiBi, attention sinks, sliding-window attention, and externally quantized
    FP8 queries are not supported.
    """
    if context_partition_size != KV_COMPUTE_BLOCK:
        raise NotImplementedError(
            "pa_decode only supports context_partition_size=256, "
            f"got {context_partition_size}"
        )
    if query_scale is not None:
        raise NotImplementedError(
            "pa_decode does not support externally quantized FP8 queries"
        )
    if alibi_slopes is not None:
        raise NotImplementedError("pa_decode does not support ALiBi")
    if sinks is not None:
        raise NotImplementedError("pa_decode does not support attention sinks")
    if sliding_window != 0:
        raise NotImplementedError("pa_decode does not support sliding-window attention")
    if query_length < 1:
        raise ValueError(f"query_length must be positive, got {query_length}")
    if not 1 <= max_context_partition_num <= 64:
        raise ValueError(
            "max_context_partition_num must be in [1, 64], "
            f"got {max_context_partition_num}"
        )

    arch = get_gfx_runtime()
    expected_fp8_dtype = {
        "gfx942": torch.float8_e4m3fnuz,
        "gfx950": torch.float8_e4m3fn,
    }.get(arch)
    if expected_fp8_dtype is None:
        raise NotImplementedError(
            f"pa_decode only supports gfx942 and gfx950, got {arch}"
        )
    if compute_type != expected_fp8_dtype:
        raise NotImplementedError(
            f"pa_decode only supports FP8 compute ({expected_fp8_dtype}) on {arch}, "
            f"got {compute_type}"
        )

    # ``ps`` is retained for drop-in API compatibility. Both partitioning
    # policies are represented by the caller-provided partition count.
    del ps

    num_seqs = context_lengths.shape[0]
    total_q_rows, num_q_heads, head_dim = query.shape
    assert total_q_rows == num_seqs * query_length, (
        f"query.shape[0] ({total_q_rows}) must equal "
        f"context_lengths.shape[0] * query_length ({num_seqs} * {query_length})"
    )
    assert output.shape == query.shape, (
        f"output shape {tuple(output.shape)} must match "
        f"query shape {tuple(query.shape)}"
    )
    _, num_kv_heads, num_hgroups, block_size, hgroup_width = key_cache.shape

    assert num_hgroups == head_dim // 16 and hgroup_width == 16
    # Q staging loads each lane's head_dim//16 elements in <=8-wide pieces, so a
    # chunk wider than 8 that isn't a multiple of 8 would silently drop its tail
    # (head_dim 192/320/448/...). Reject those rather than return wrong results.
    q_chunk = head_dim // 16
    if q_chunk > 8 and q_chunk % 8 != 0:
        raise NotImplementedError(
            f"pa_decode does not support head_dim={head_dim}: head_dim//16 "
            f"({q_chunk}) must be <=8 or a multiple of 8"
        )
    assert block_size in (
        16,
        64,
    ), f"pa_decode only supports block_size in (16, 64), got {block_size}"

    trans_v = value_cache.dim() == 5
    if trans_v:
        _, v_num_kv_heads, v_subblocks, v_head_dim, v_width = value_cache.shape
        assert (
            v_head_dim == head_dim and v_width == 16 and v_subblocks == block_size // 16
        ), f"value_cache shape {tuple(value_cache.shape)} doesn't match block_size={block_size}, head_dim={head_dim}"
    else:
        _, v_num_kv_heads, v_head_dim, v_block_size = value_cache.shape
        assert v_head_dim == head_dim and v_block_size == block_size, (
            f"value_cache shape {tuple(value_cache.shape)} doesn't match "
            f"block_size={block_size}, head_dim={head_dim}"
        )
    assert v_num_kv_heads == num_kv_heads
    assert (
        num_q_heads % num_kv_heads == 0
    ), f"num_q_heads ({num_q_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
    assert (
        block_tables.dtype == torch.int32
    ), f"block_tables must be int32, got {block_tables.dtype}"
    assert (
        context_lengths.dtype == torch.int32
    ), f"context_lengths must be int32, got {context_lengths.dtype}"
    query_group_size = num_q_heads // num_kv_heads
    max_blocks_per_seq = block_tables.shape[1]
    if query.dtype == torch.bfloat16:
        query_dtype = "bf16"
    elif query.dtype == torch.float16:
        query_dtype = "f16"
    else:
        raise ValueError(f"pa_decode only supports f16/bf16 query, got {query.dtype}")
    assert (
        output.dtype == query.dtype
    ), f"pa_decode requires output.dtype == query.dtype, got {output.dtype} vs {query.dtype}"

    assert (
        key_cache.dtype == expected_fp8_dtype
    ), f"pa_decode requires {expected_fp8_dtype} key cache on {arch}, got {key_cache.dtype}"
    assert (
        value_cache.dtype == expected_fp8_dtype
    ), f"pa_decode requires {expected_fp8_dtype} value cache on {arch}, got {value_cache.dtype}"

    assert (
        query.stride(2) == 1
    ), f"pa_decode requires a contiguous head_dim axis, got strides {query.stride()}"

    dev = query.device
    for name, tensor in (
        ("output", output),
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("block_tables", block_tables),
        ("context_lengths", context_lengths),
    ):
        assert tensor.device == dev, (
            f"{name} must be on the same device as query ({dev}), "
            f"got {tensor.device}"
        )
    for name, tensor in (
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("block_tables", block_tables),
        ("context_lengths", context_lengths),
    ):
        assert tensor.is_contiguous(), f"{name} must be contiguous"

    def normalize_scale(scale, name):
        if scale is None:
            return torch.ones(1, dtype=torch.float32, device=dev)
        if not isinstance(scale, torch.Tensor):
            return torch.tensor([float(scale)], dtype=torch.float32, device=dev)
        if scale.numel() == 1:
            return scale.reshape(1)
        if scale.dim() == 4:
            assert scale.shape[-1] == 1, (
                f"{name} must have a trailing singleton dimension, "
                f"got shape {tuple(scale.shape)}"
            )
            return scale.squeeze(-1)
        assert scale.dim() == 3, (
            f"{name} must be scalar or have shape "
            "[num_blocks, num_kv_heads, block_size, 1]"
        )
        return scale

    if (key_scale is None) != (value_scale is None):
        raise ValueError(
            "key_scale and value_scale must either both be provided or both be None"
        )
    key_scale_t = normalize_scale(key_scale, "key_scale")
    value_scale_t = normalize_scale(value_scale, "value_scale")
    per_token_kv = key_scale_t.numel() > 1
    assert per_token_kv == (
        value_scale_t.numel() > 1
    ), "key_scale and value_scale must both be per-tensor or both be per-token"
    if per_token_kv:
        assert (
            key_scale_t.shape == value_scale_t.shape
        ), f"key_scale/value_scale shape mismatch: {tuple(key_scale_t.shape)} vs {tuple(value_scale_t.shape)}"
        assert key_scale_t.shape == (key_cache.shape[0], num_kv_heads, block_size), (
            "per-token key_scale/value_scale must be [num_blocks, num_kv_heads, block_size] "
            f"matching the KV cache, got {tuple(key_scale_t.shape)}"
        )
        stride_ks_block = int(key_scale_t.stride(0))
        stride_ks_head = int(key_scale_t.stride(1))
        assert key_scale_t.stride(2) == 1, (
            f"per-token key_scale token dimension must be contiguous, "
            f"got strides {key_scale_t.stride()}"
        )
        assert value_scale_t.stride() == key_scale_t.stride(), (
            "per-token key_scale and value_scale must have matching strides, "
            f"got {key_scale_t.stride()} vs {value_scale_t.stride()}"
        )
    else:
        stride_ks_block = 0
        stride_ks_head = 0
    assert (
        key_scale_t.dtype == torch.float32 and key_scale_t.device == dev
    ), f"key_scale tensor must be float32 on {dev}, got {key_scale_t.dtype} on {key_scale_t.device}"
    assert (
        value_scale_t.dtype == torch.float32 and value_scale_t.device == dev
    ), f"value_scale tensor must be float32 on {dev}, got {value_scale_t.dtype} on {value_scale_t.device}"

    num_partitions = max_context_partition_num
    pmax = max_logits
    psum = exp_sums
    pout = temporary_output

    # An i32 element offset wraps once a single cache tensor passes 2^31
    # elements (2 GiB at fp8). The wider math costs ~18% at block_size=64, so
    # pay it only where it is needed. Both caches share the code path, so widen
    # if either does.
    wide_kv_addressing = max(key_cache.numel(), value_cache.numel()) >= 2**31

    with torch.cuda.device(dev):
        compiled = compile_pa_decode_tile(
            head_dim=head_dim,
            query_group_size=query_group_size,
            block_size=int(block_size),
            num_partitions=num_partitions,
            softmax_scale=softmax_scale,
            query_dtype=query_dtype,
            per_token_kv=per_token_kv,
            query_length=query_length,
            trans_v=trans_v,
            wide_kv_addressing=wide_kv_addressing,
        )

    if num_partitions == 1:
        # NP==1 writes output directly; partials unused (caller buffers ignored).
        dummy = torch.empty(1, dtype=torch.float32, device=dev)
        pmax = psum = pout = dummy
    else:
        total_rows = query_length * query_group_size
        expected_scalar_shape = (num_seqs, num_kv_heads, num_partitions, total_rows)
        if pmax is None or psum is None or pout is None:
            if pmax is None:
                pmax = torch.empty(
                    *expected_scalar_shape, dtype=torch.float32, device=dev
                )
            if psum is None:
                psum = torch.empty(
                    *expected_scalar_shape, dtype=torch.float32, device=dev
                )
            if pout is None:
                pout = torch.empty(
                    *expected_scalar_shape, head_dim, dtype=output.dtype, device=dev
                )
        assert (
            pmax.shape == expected_scalar_shape
        ), f"max_logits shape {tuple(pmax.shape)} != {expected_scalar_shape}"
        assert (
            psum.shape == expected_scalar_shape
        ), f"exp_sums shape {tuple(psum.shape)} != {expected_scalar_shape}"
        assert pout.shape == (
            *expected_scalar_shape,
            head_dim,
        ), (
            f"temporary_output shape {tuple(pout.shape)} != "
            f"{(*expected_scalar_shape, head_dim)}"
        )
        assert pmax.dtype == torch.float32 and pmax.device == dev
        assert psum.dtype == torch.float32 and psum.device == dev
        assert pout.dtype == output.dtype and pout.device == dev
        assert pmax.is_contiguous()
        assert psum.is_contiguous()
        assert pout.is_contiguous()
    with torch.cuda.device(dev):
        s = torch.cuda.current_stream(dev)
        _run_compiled(
            compiled["launch"],
            output,
            pmax.view(-1),
            psum.view(-1),
            pout.view(-1),
            query,
            key_cache,
            value_cache,
            block_tables,
            context_lengths,
            key_scale_t,
            value_scale_t,
            int(max_blocks_per_seq),
            int(num_seqs),
            int(num_kv_heads),
            stride_ks_block,
            stride_ks_head,
            int(query.stride(0)),
            int(query.stride(1)),
            s,
        )
        if num_partitions > 1:
            output_5d = output.reshape(
                num_seqs, query_length, num_kv_heads, query_group_size, head_dim
            )
            launch_pa_decode_ps_reduce(
                output_5d,
                psum,
                pmax,
                pout,
                None,
                output_5d.stride(0),
                output_5d.stride(1),
                output_5d.stride(2),
                output_5d.stride(3),
                pmax.stride(0),
                pmax.stride(1),
                pmax.stride(2),
                pout.stride(0),
                pout.stride(1),
                pout.stride(2),
                pout.stride(3),
                query_seq_len=query_length,
                query_group_size=query_group_size,
                head_size=head_dim,
                context_partition_num=num_partitions,
                stream=s,
            )
