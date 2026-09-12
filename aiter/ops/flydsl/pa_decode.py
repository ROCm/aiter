# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Readable tile-programming reference for paged-attention fp8 decode.

K/V are fp8 e4m3 (FNUZ on gfx942, OCP on gfx950) fed straight into
``mfma_f32_16x16x32_fp8_fp8``; Q (bf16/f16) and the softmax probabilities P are
quantized to fp8 too. Scales fold out of the matmuls (q/key scale into the QK
score, value scale + 1/FP8_MAX into the epilogue); softmax max/sum stay f32.
``key_scale``/``value_scale`` are either a ``[1]`` per-tensor scalar or a
``[num_blocks, num_kv_heads, block_size, 1]`` per-token tensor.

``block_size`` (16/64/128) and ``head_dim`` (multiple of 64) are compile-time
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

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.pa_decode_reduce import compile_pa_decode_ps_reduce
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
        torch.float8_e4m3fn: fx.Float8E4M3FN,
        torch.float8_e4m3fnuz: fx.Float8E4M3FNUZ,
        torch.int32: fx.Int32,
    }[dtype]


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
    BF16/FP16 queries, a 256-token compute tile, and block sizes 16/64/128.
    Sparse attention uses caller-prepared block tables and selected context
    lengths. Each independently selected MTP query must have its own table row
    and use query_length=1; query_length>1 applies dense causal masking.
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
    if sliding_window not in (0, -1):
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
        128,
    ), f"pa_decode only supports block_size in (16, 64, 128), got {block_size}"

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
    assert (
        output.stride(2) == 1
    ), f"pa_decode requires a contiguous output head_dim axis, got strides {output.stride()}"

    dev = query.device
    for name, tensor in (
        ("output", output),
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("block_tables", block_tables),
        ("context_lengths", context_lengths),
    ):
        assert (
            tensor.device == dev
        ), f"{name} must be on the same device as query ({dev}), got {tensor.device}"
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

    # Early V loads help the measured one-to-two-workgroup-per-CU regime.
    # Keep other grids on the existing schedule: early loads regress
    # short-context decode at larger grid sizes.
    prefetch_v = False
    if (
        arch == "gfx950"
        and head_dim == 128
        and block_size == 128
        and trans_v
        and not per_token_kv
        and query_length * query_group_size <= 16
    ):
        num_cus = torch.cuda.get_device_properties(dev).multi_processor_count
        workgroups = num_seqs * num_kv_heads * num_partitions
        prefetch_v = num_cus < workgroups <= 2 * num_cus

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
            prefetch_v=prefetch_v,
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
            ptr_arg(output, _flydsl_pointer_dtype(output.dtype)),
            ptr_arg(pmax, fx.Float32),
            ptr_arg(psum, fx.Float32),
            ptr_arg(pout, _flydsl_pointer_dtype(pout.dtype)),
            ptr_arg(query, _flydsl_pointer_dtype(query.dtype)),
            ptr_arg(key_cache, _flydsl_pointer_dtype(key_cache.dtype)),
            ptr_arg(value_cache, _flydsl_pointer_dtype(value_cache.dtype)),
            ptr_arg(block_tables, fx.Int32),
            ptr_arg(context_lengths, fx.Int32),
            ptr_arg(key_scale_t, fx.Float32),
            ptr_arg(value_scale_t, fx.Float32),
            int(max_blocks_per_seq),
            int(num_seqs),
            int(num_kv_heads),
            stride_ks_block,
            stride_ks_head,
            int(output.stride(0)),
            int(output.stride(1)),
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
