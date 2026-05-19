# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paged-attention decode wrapper for FlyDSL on gfx1250.

Computes: out = Softmax(Q @ K^T * scale) @ V
with Q, K, V stored in a paged KV-cache and bf16/f16 precision.

Usage:
    from aiter.ops.flydsl.pa_decode import flydsl_paged_attention_decode

    flydsl_paged_attention_decode(
        output,       # [num_seqs, num_q_heads, head_size]
        query,        # [num_seqs, num_q_heads, head_size]
        key_cache,    # [num_blocks, num_kv_heads, kv_block_size, head_size]
        value_cache,  # [num_blocks, num_kv_heads, kv_block_size, head_size]
        block_tables, # [num_seqs, max_num_blocks_per_seq]
        seq_lens,     # [num_seqs]
        attn_scale,   # float
    )
"""

from __future__ import annotations

import struct

import torch

from .kernels.pa_decode_gfx1250 import (
    compile_pa_decode_main,
    compile_pa_decode_reduce,
)

_DEFAULT_PARTITION_SIZE = 256
_DEFAULT_KV_COMPUTE_BLOCK_SIZE = 64


def _dtype_to_str(dt: torch.dtype) -> str:
    if dt == torch.bfloat16:
        return "bf16"
    if dt == torch.float16:
        return "f16"
    raise ValueError(f"Unsupported dtype for pa_decode: {dt}")


def flydsl_paged_attention_decode(
    output: torch.Tensor,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    attn_scale: float,
    *,
    partition_size: int = _DEFAULT_PARTITION_SIZE,
    kv_compute_block_size: int = _DEFAULT_KV_COMPUTE_BLOCK_SIZE,
    stream: torch.cuda.Stream | None = None,
) -> torch.Tensor:
    """Paged-attention decode for gfx1250.

    Args:
        output: Output tensor [num_seqs, num_q_heads, head_size], bf16/f16.
        query:  Query tensor [num_seqs, num_q_heads, head_size], bf16/f16.
        key_cache: Flat KV cache [num_blocks, num_kv_heads, kv_block_size, head_size].
        value_cache: Same shape as key_cache.
        block_tables: [num_seqs, max_num_blocks_per_seq], int32.
        seq_lens: [num_seqs], int32.
        attn_scale: Softmax scale (usually 1/sqrt(head_size)).
        partition_size: Number of KV tokens per partition. Must be a multiple of
            ``kv_compute_block_size``.
        kv_compute_block_size: Number of KV tokens handled per loop iteration. Must
            be a multiple of page table's block size.
        stream: Optional CUDA stream to launch on. If ``None``, uses
            ``torch.cuda.current_stream(device=query.device)``. If provided,
            it must be on the same device as ``query``.

    Returns:
        The ``output`` tensor.
    """
    # --- Rank checks ---
    if output.dim() != 3:
        raise ValueError(f"output must be 3D, got {output.shape}")
    if query.dim() != 3:
        raise ValueError(f"query must be 3D, got {query.shape}")
    if key_cache.dim() != 4:
        raise ValueError(f"key_cache must be 4D, got {key_cache.shape}")
    if value_cache.dim() != 4:
        raise ValueError(f"value_cache must be 4D, got {value_cache.shape}")
    if block_tables.dim() != 2:
        raise ValueError(f"block_tables must be 2D, got {block_tables.shape}")
    if seq_lens.dim() != 1:
        raise ValueError(f"seq_lens must be 1D, got {seq_lens.shape}")

    # --- Dtype / shape parity checks ---
    if not (output.dtype == query.dtype == key_cache.dtype == value_cache.dtype):
        raise ValueError(
            f"Q / KV / output dtypes must match, got "
            f"output={output.dtype}, query={query.dtype}, "
            f"key_cache={key_cache.dtype}, value_cache={value_cache.dtype}"
        )
    if output.shape != query.shape:
        raise ValueError(
            f"output shape must match query shape, got "
            f"output={output.shape}, query={query.shape}"
        )
    if value_cache.shape != key_cache.shape:
        raise ValueError(
            f"value_cache shape must match key_cache shape, got "
            f"value_cache={value_cache.shape}, key_cache={key_cache.shape}"
        )

    # --- Device checks: all tensors on the same CUDA/HIP device as query ---
    named_tensors = {
        "output": output,
        "query": query,
        "key_cache": key_cache,
        "value_cache": value_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
    }
    for name, t in named_tensors.items():
        if not t.is_cuda:
            raise ValueError(
                f"{name} must be on a CUDA/HIP device, got device={t.device}"
            )
        if t.device != query.device:
            raise ValueError(
                f"all tensors must be on the same device as query "
                f"({query.device}), but {name} is on {t.device}"
            )

    # --- Architecture check: kernel is gfx1250-only. ---
    try:
        arch = torch.cuda.get_device_properties(query.device.index).gcnArchName
    except Exception:
        arch = ""
    arch_base = arch.lower().split(":")[0] if arch else ""
    if not arch_base.startswith("gfx1250"):
        raise ValueError(
            f"flydsl_paged_attention_decode requires gfx1250, got {arch!r}"
        )

    # --- Shape derivations + GQA / head-size checks ---
    num_seqs, num_q_heads, head_size = query.shape
    num_blocks, num_kv_heads, kv_block_size, head_size_kv = key_cache.shape
    if head_size != head_size_kv:
        raise ValueError(f"Q head_size {head_size} != KV head_size {head_size_kv}")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_q_heads {num_q_heads} not divisible by num_kv_heads {num_kv_heads}"
        )
    query_group_size = num_q_heads // num_kv_heads

    if block_tables.shape[0] != num_seqs:
        raise ValueError(
            f"block_tables.shape[0] ({block_tables.shape[0]}) must equal "
            f"num_seqs ({num_seqs})"
        )
    if seq_lens.shape[0] != num_seqs:
        raise ValueError(
            f"seq_lens.shape[0] ({seq_lens.shape[0]}) must equal num_seqs ({num_seqs})"
        )
    if block_tables.dtype != torch.int32:
        raise ValueError(f"block_tables must be int32, got {block_tables.dtype}")
    if seq_lens.dtype != torch.int32:
        raise ValueError(f"seq_lens must be int32, got {seq_lens.dtype}")

    # --- Kernel tile constraints (mirrored in compile_pa_decode_main). ---
    if head_size % 32 != 0:
        raise ValueError(f"head_size must be multiple of 32 (WMMA_K), got {head_size}")
    if kv_block_size % 16 != 0:
        raise ValueError(f"kv_block_size must be multiple of 16, got {kv_block_size}")
    if kv_compute_block_size % 32 != 0:
        raise ValueError(
            f"kv_compute_block_size must be multiple of 32 (WMMA_K), "
            f"got {kv_compute_block_size}"
        )
    if kv_compute_block_size % kv_block_size != 0:
        raise ValueError(
            f"kv_compute_block_size {kv_compute_block_size} must be multiple of "
            f"kv_block_size {kv_block_size}"
        )
    if partition_size % kv_compute_block_size != 0:
        raise ValueError(
            f"partition_size {partition_size} must be multiple of "
            f"kv_compute_block_size {kv_compute_block_size}"
        )
    if not (1 <= query_group_size <= 16):
        raise ValueError(
            f"query_group_size must be in [1, 16] (WMMA_M), got {query_group_size}"
        )

    # --- Contiguity: we assume canonical strides. ---
    if not query.is_contiguous():
        raise ValueError("query must be contiguous")
    if not key_cache.is_contiguous():
        raise ValueError("key_cache must be contiguous")
    if not value_cache.is_contiguous():
        raise ValueError("value_cache must be contiguous")
    if not output.is_contiguous():
        raise ValueError("output must be contiguous")
    if not block_tables.is_contiguous():
        raise ValueError("block_tables must be contiguous")
    if not seq_lens.is_contiguous():
        raise ValueError("seq_lens must be contiguous")

    max_seq_len = int(seq_lens.max().item()) if num_seqs > 0 else 0
    if max_seq_len == 0:
        output.zero_()
        return output

    num_partitions = max(1, (max_seq_len + partition_size - 1) // partition_size)
    max_blocks_per_seq = block_tables.shape[1]

    device = query.device
    dtype_str = _dtype_to_str(query.dtype)

    # Allocate tmp buffers for split-KV outputs.
    tmp_out = torch.empty(
        (num_seqs, num_kv_heads, num_partitions, query_group_size, head_size),
        dtype=torch.float32,
        device=device,
    )
    max_logits = torch.full(
        (num_seqs, num_kv_heads, num_partitions, query_group_size),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    exp_sums = torch.zeros(
        (num_seqs, num_kv_heads, num_partitions, query_group_size),
        dtype=torch.float32,
        device=device,
    )

    # Resolve / validate the launch stream for the input's device.
    if stream is None:
        stream = torch.cuda.current_stream(device=device)
    elif stream.device != device:
        raise ValueError(f"`stream` must be on {device}, got {stream.device}")

    # Pack float32 attn_scale as i32 for kernel arg (kernel bitcasts back).
    scale_i32 = struct.unpack("<i", struct.pack("<f", float(attn_scale)))[0]

    # Pin the current device so JIT compile + kernel launches go to query.device,
    # regardless of the caller's current CUDA context.
    with torch.cuda.device(device):
        # Compile kernels (cached)
        main_launch = compile_pa_decode_main(
            HEAD_SIZE=head_size,
            KV_BLOCK_SIZE=kv_block_size,
            QUERY_GROUP_SIZE=query_group_size,
            PARTITION_SIZE=partition_size,
            KV_COMPUTE_BLOCK_SIZE=kv_compute_block_size,
            dtype=dtype_str,
        )
        reduce_launch = compile_pa_decode_reduce(
            HEAD_SIZE=head_size,
            QUERY_GROUP_SIZE=query_group_size,
            PARTITION_SIZE=partition_size,
            NUM_PARTITIONS=num_partitions,
            dtype=dtype_str,
        )

        # Launch main kernel. Flatten tensors to 1D views the kernels can index.
        main_launch(
            tmp_out.view(-1),
            max_logits.view(-1),
            exp_sums.view(-1),
            query.view(-1),
            key_cache.view(-1),
            value_cache.view(-1),
            block_tables.view(-1),
            seq_lens.view(-1),
            scale_i32,
            num_seqs,
            num_kv_heads,
            num_partitions,
            max_blocks_per_seq,
            stream,
        )

        reduce_launch(
            output.view(-1),
            tmp_out.view(-1),
            max_logits.view(-1),
            exp_sums.view(-1),
            seq_lens.view(-1),
            num_seqs,
            num_kv_heads,
            num_partitions,
            stream,
        )
    return output


__all__ = ["flydsl_paged_attention_decode"]
