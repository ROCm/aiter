# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""Readable tile-programming reference for paged-attention fp8 decode.

K/V are fp8 e4m3 (FNUZ on gfx942, OCP on gfx950) fed straight into
``mfma_f32_16x16x32_fp8_fp8``; Q (bf16/f16) and the softmax probabilities P are
quantized to fp8 too. Scales fold out of the matmuls (q/key scale into the QK
score, value scale + 1/FP8_MAX into the epilogue); softmax max/sum stay f32.
``key_scale``/``value_scale`` are either a ``[1]`` per-tensor scalar or a
``[num_blocks, num_kv_heads, block_size]`` per-token tensor (chosen by rank).

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

import torch

from .kernels.pa_decode_tile import KV_COMPUTE_BLOCK, compile_pa_decode_tile
from .kernels.tensor_shim import _run_compiled
from .kernels.utils import cdiv


def _is_current_stream_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_current_stream_capturing()
    except RuntimeError:
        return False


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


def pa_decode_tile(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    context_lengths: torch.Tensor,
    key_scale: float | torch.Tensor,
    value_scale: float | torch.Tensor,
    softmax_scale: float | None = None,
    stream=None,
    *,
    num_partitions: int | None = None,
    pmax: torch.Tensor | None = None,
    psum: torch.Tensor | None = None,
    pout: torch.Tensor | None = None,
) -> None:
    """Host entry point. See module docstring for tensor layouts.

    ``num_partitions``/``pmax``/``psum``/``pout`` are optional caller overrides
    (e.g. for CUDA-graph capture, where nothing may be allocated on-the-fly);
    when omitted they are picked/allocated here.
    """
    num_seqs = context_lengths.shape[0]
    total_q_rows, num_q_heads, head_dim = query.shape
    assert (
        total_q_rows % num_seqs == 0
    ), f"query.shape[0] ({total_q_rows}) must be a multiple of context_lengths.shape[0] ({num_seqs})"
    query_length = total_q_rows // num_seqs
    _, num_kv_heads, num_hgroups, block_size, hgroup_width = key_cache.shape

    assert num_hgroups == head_dim // 16 and hgroup_width == 16
    assert block_size in (
        16,
        64,
    ), f"pa_decode_tile only supports block_size in (16, 64), got {block_size}"

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
        raise ValueError(
            f"pa_decode_tile only supports f16/bf16 query, got {query.dtype}"
        )
    assert (
        output.dtype == query.dtype
    ), f"pa_decode_tile requires output.dtype == query.dtype, got {output.dtype} vs {query.dtype}"

    assert (
        query.stride(2) == 1
    ), f"pa_decode_tile requires a contiguous head_dim axis, got strides {query.stride()}"

    dev = query.device
    key_scale_t = (
        key_scale
        if isinstance(key_scale, torch.Tensor)
        else torch.tensor([float(key_scale)], device=dev)
    )
    value_scale_t = (
        value_scale
        if isinstance(value_scale, torch.Tensor)
        else torch.tensor([float(value_scale)], device=dev)
    )
    per_token_kv = key_scale_t.dim() > 1
    if per_token_kv:
        assert (
            value_scale_t.dim() > 1
        ), "value_scale must also be per-token (dim>1) when key_scale is per-token"
        assert (
            key_scale_t.shape == value_scale_t.shape
        ), f"key_scale/value_scale shape mismatch: {tuple(key_scale_t.shape)} vs {tuple(value_scale_t.shape)}"
        assert key_scale_t.shape == (key_cache.shape[0], num_kv_heads, block_size), (
            "per-token key_scale/value_scale must be [num_blocks, num_kv_heads, block_size] "
            f"matching the KV cache, got {tuple(key_scale_t.shape)}"
        )
        stride_ks_block = int(key_scale_t.stride(0))
        stride_ks_head = int(key_scale_t.stride(1))
    else:
        stride_ks_block = 0
        stride_ks_head = 0
    assert (
        key_scale_t.dtype == torch.float32 and key_scale_t.device == dev
    ), f"key_scale tensor must be float32 on {dev}, got {key_scale_t.dtype} on {key_scale_t.device}"
    assert (
        value_scale_t.dtype == torch.float32 and value_scale_t.device == dev
    ), f"value_scale tensor must be float32 on {dev}, got {value_scale_t.dtype} on {value_scale_t.device}"

    if not num_partitions:
        blocks_per_partition = KV_COMPUTE_BLOCK // block_size
        num_partitions = get_recommended_splits(
            num_seqs,
            num_kv_heads,
            split_kv_blocks=blocks_per_partition,
        )

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
    )

    is_graph_capturing = _is_current_stream_capturing()
    if num_partitions == 1:
        # NP==1 writes output directly; partials unused (caller buffers ignored).
        if pmax is None:
            if is_graph_capturing:
                raise ValueError(
                    "CUDA graph capture requires preallocated `pmax`/`psum`/`pout` "
                    "even when num_partitions==1 (nothing may be allocated mid-capture)."
                )
            pmax = psum = pout = torch.empty(1, dtype=torch.float32, device=dev)
    else:
        total_rows = query_length * query_group_size
        expected_scalar_shape = (num_seqs, num_kv_heads, num_partitions, total_rows)
        if pmax is None or psum is None or pout is None:
            if is_graph_capturing:
                raise ValueError(
                    "CUDA graph capture requires preallocated `pmax`/`psum`/`pout` "
                    "for num_partitions>1 (nothing may be allocated mid-capture)."
                )
            pmax = torch.empty(*expected_scalar_shape, dtype=torch.float32, device=dev)
            psum = torch.empty(*expected_scalar_shape, dtype=torch.float32, device=dev)
            pout = torch.empty(
                *expected_scalar_shape, head_dim, dtype=output.dtype, device=dev
            )
        else:
            assert (
                pmax.shape == expected_scalar_shape
            ), f"pmax shape {tuple(pmax.shape)} != {expected_scalar_shape}"
            assert (
                psum.shape == expected_scalar_shape
            ), f"psum shape {tuple(psum.shape)} != {expected_scalar_shape}"
            assert pout.shape == (
                *expected_scalar_shape,
                head_dim,
            ), f"pout shape {tuple(pout.shape)} != {(*expected_scalar_shape, head_dim)}"
    s = stream or torch.cuda.current_stream()

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
        from csrc.cpp_itfs.pa.pa_ps import launch_pa_decode_ps_reduce

        output_5d = output.reshape(
            num_seqs, query_length, num_kv_heads, query_group_size, head_dim
        )
        with torch.cuda.stream(s):
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
            )
