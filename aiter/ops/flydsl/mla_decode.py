# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import struct

import torch

from .kernels.mla_decode_shuffled_gfx1250 import (
    compile_mla_decode_main,
    compile_mla_decode_reduce,
)
from .kernels.tensor_shim import _run_compiled


# returns (num_segs, num_warps, kv_compute_block_size, warp_token_split, warp_head_split)
def _resolve_mla_config(num_q_heads, num_seqs, max_seqlen):

    warp_token_split = num_q_heads <= 16
    warp_head_split = not warp_token_split

    kvc = 32 if num_q_heads >= 128 else 64

    if warp_token_split:
        num_warps = min(
            2, max(1, kvc // 32)
        )  # WMMA_K=32 so can't have num_warps > kvc//32
    else:
        # num_warps should be power of two and divide 16 head tile count
        # n & -n gives the largest power of 2 dividing n
        head_tiles = (num_q_heads + 16 - 1) // 16
        num_warps = min(head_tiles & -head_tiles, 8)

    wgs_to_fill = 1024 // min(num_warps, 4)
    num_segs = (wgs_to_fill + num_seqs - 1) // num_seqs
    num_segs = min(num_segs, (max_seqlen + kvc - 1) // kvc, 64)
    return num_segs, num_warps, kvc, warp_token_split, warp_head_split


def _dtype_to_str(dt: torch.dtype) -> str:
    if dt == torch.bfloat16:
        return "bf16"
    if dt == torch.float16:
        return "f16"
    raise ValueError(f"Unsupported dtype for mla_decode: {dt}")


def _mla_num_partitions(num_segs: int, num_warps: int, warp_token_split: bool) -> int:
    return num_segs * (num_warps if warp_token_split else 1)


def _alloc_partials(num_seqs, num_partitions, num_q_heads, kv_lora_rank, device):
    tmp_out = torch.empty(
        (num_seqs, num_partitions, num_q_heads, kv_lora_rank),
        dtype=torch.float32,
        device=device,
    )
    max_logits = torch.full(
        (num_seqs, num_partitions, num_q_heads),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    exp_sums = torch.zeros(
        (num_seqs, num_partitions, num_q_heads),
        dtype=torch.float32,
        device=device,
    )
    return tmp_out, max_logits, exp_sums


def flydsl_mla_decode(
    output: torch.Tensor,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    attn_scale: float,
    *,
    max_seqlen: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    # Pre-allocated (tmp_out, max_logits, exp_sums) to reuse across calls when capturing a cudagraph
    partials: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    stream: torch.cuda.Stream | None = None,
):
    # --- Rank / dtype checks ---
    if output.dim() != 3:
        raise ValueError(f"output must be 3D, got {output.shape}")
    if query.dim() != 3:
        raise ValueError(f"query must be 3D, got {query.shape}")
    if kv_cache.dim() != 4:
        raise ValueError(f"kv_cache must be 4D, got {kv_cache.shape}")
    if block_tables.dim() != 2:
        raise ValueError(f"block_tables must be 2D, got {block_tables.shape}")
    if seq_lens.dim() != 1:
        raise ValueError(f"seq_lens must be 1D, got {seq_lens.shape}")
    if not (output.dtype == query.dtype == kv_cache.dtype):
        raise ValueError(
            f"Q / KV / output dtypes must match, got output={output.dtype}, "
            f"query={query.dtype}, kv_cache={kv_cache.dtype}"
        )

    # --- Device checks ---
    for name, t in {
        "output": output,
        "query": query,
        "kv_cache": kv_cache,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
    }.items():
        if not t.is_cuda:
            raise ValueError(f"{name} must be on a CUDA/HIP device, got {t.device}")
        if t.device != query.device:
            raise ValueError(
                f"all tensors must be on query's device ({query.device}), "
                f"but {name} is on {t.device}"
            )

    # --- Architecture check ---
    try:
        arch = torch.cuda.get_device_properties(query.device.index).gcnArchName
    except Exception:  # noqa: BLE001 - any failure to read the arch means "not gfx1250"
        arch = ""
    if not (arch.lower().split(":")[0] if arch else "").startswith("gfx1250"):
        raise ValueError(f"flydsl_mla_decode requires gfx1250, got {arch!r}")

    # --- Shape derivations ---
    num_seqs, num_q_heads, qk_head_dim = query.shape
    # Pre-shuffled cache axis order: [nb, kv_heads, block, head].
    num_blocks, num_kv_heads, block_size, qk_head_dim_kv = kv_cache.shape
    if qk_head_dim != qk_head_dim_kv:
        raise ValueError(
            f"Q qk_head_dim {qk_head_dim} != KV qk_head_dim {qk_head_dim_kv}"
        )
    if kv_lora_rank + qk_rope_head_dim != qk_head_dim:
        raise ValueError(
            f"kv_lora_rank ({kv_lora_rank}) + qk_rope_head_dim ({qk_rope_head_dim}) "
            f"must equal qk_head_dim ({qk_head_dim})"
        )
    if num_kv_heads != 1:
        raise ValueError(f"MLA expects num_kv_heads == 1, got {num_kv_heads}")
    if output.shape != (num_seqs, num_q_heads, kv_lora_rank):
        raise ValueError(
            f"output must be [num_seqs, num_q_heads, kv_lora_rank] = "
            f"{(num_seqs, num_q_heads, kv_lora_rank)}, got {output.shape}"
        )
    if block_tables.shape[0] != num_seqs:
        raise ValueError(
            f"block_tables.shape[0] ({block_tables.shape[0]}) must equal num_seqs ({num_seqs})"
        )
    if seq_lens.shape[0] != num_seqs:
        raise ValueError(
            f"seq_lens.shape[0] ({seq_lens.shape[0]}) must equal num_seqs ({num_seqs})"
        )
    if block_tables.dtype != torch.int32:
        raise ValueError(f"block_tables must be int32, got {block_tables.dtype}")
    if seq_lens.dtype != torch.int32:
        raise ValueError(f"seq_lens must be int32, got {seq_lens.dtype}")

    # --- Contiguity ---
    for name, t in {
        "query": query,
        "kv_cache": kv_cache,
        "output": output,
        "block_tables": block_tables,
        "seq_lens": seq_lens,
    }.items():
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if num_seqs == 0 or max_seqlen == 0:
        output.zero_()
        return output

    max_blocks_per_seq = block_tables.shape[1]
    device = query.device
    dtype_str = _dtype_to_str(query.dtype)

    num_segs, num_warps, kv_compute_block_size, warp_token_split, warp_head_split = (
        _resolve_mla_config(num_q_heads, num_seqs, max_seqlen)
    )
    num_partitions = _mla_num_partitions(num_segs, num_warps, warp_token_split)

    # A single partition has nothing to merge, so the main kernel normalizes in-register and
    # writes `output` directly and no need for reduce kernel
    write_final_output = num_partitions == 1

    if write_final_output:
        tmp_out = max_logits = exp_sums = output
    elif partials is None:
        tmp_out, max_logits, exp_sums = _alloc_partials(
            num_seqs, num_partitions, num_q_heads, kv_lora_rank, device
        )
    else:
        tmp_out, max_logits, exp_sums = partials
        tmp_shape = (num_seqs, num_partitions, num_q_heads, kv_lora_rank)
        lse_shape = (num_seqs, num_partitions, num_q_heads)
        if tuple(tmp_out.shape) != tmp_shape:
            raise ValueError(
                f"partials[0] (tmp_out) must be {tmp_shape}, got {tuple(tmp_out.shape)}"
            )
        if tuple(max_logits.shape) != lse_shape or tuple(exp_sums.shape) != lse_shape:
            raise ValueError(
                f"partials[1:] (max_logits, exp_sums) must be {lse_shape}, got "
                f"{tuple(max_logits.shape)} and {tuple(exp_sums.shape)}"
            )
        for nm, t in (
            ("tmp_out", tmp_out),
            ("max_logits", max_logits),
            ("exp_sums", exp_sums),
        ):
            if t.dtype != torch.float32:
                raise ValueError(f"partials {nm} must be float32, got {t.dtype}")
            if t.device != device:
                raise ValueError(f"partials {nm} must be on {device}, got {t.device}")

    if stream is None:
        stream = torch.cuda.current_stream(device=device)
    elif stream.device != device:
        raise ValueError(f"`stream` must be on {device}, got {stream.device}")

    scale_i32 = struct.unpack("<i", struct.pack("<f", float(attn_scale)))[0]
    with torch.cuda.device(device):
        main_launch = compile_mla_decode_main(
            KV_LORA_RANK=kv_lora_rank,
            QK_ROPE_HEAD_DIM=qk_rope_head_dim,
            KV_BLOCK_SIZE=block_size,
            NUM_Q_HEADS=num_q_heads,
            NUM_SEGS=num_segs,
            KV_COMPUTE_BLOCK_SIZE=kv_compute_block_size,
            NUM_WARPS=num_warps,
            dtype=dtype_str,
            WARP_TOKEN_SPLIT=warp_token_split,
            WARP_HEAD_SPLIT=warp_head_split,
            WRITE_FINAL_OUTPUT=write_final_output,
        )
        if not write_final_output:
            reduce_launch = compile_mla_decode_reduce(
                KV_LORA_RANK=kv_lora_rank,
                NUM_Q_HEADS=num_q_heads,
                NUM_SEGS=num_segs,
                KV_COMPUTE_BLOCK_SIZE=kv_compute_block_size,
                dtype=dtype_str,
                ATTN_NUM_WARPS=num_warps,
                WARP_TOKEN_SPLIT=warp_token_split,
            )

        _run_compiled(
            main_launch,
            tmp_out.view(-1),
            max_logits.view(-1),
            exp_sums.view(-1),
            query.view(-1),
            kv_cache.view(num_blocks, -1),
            block_tables.view(-1),
            seq_lens.view(-1),
            scale_i32,
            num_seqs,
            max_blocks_per_seq,
            stream,
        )
        if write_final_output:
            return output
        _run_compiled(
            reduce_launch,
            output.view(-1),
            tmp_out.view(-1),
            max_logits.view(-1),
            exp_sums.view(-1),
            seq_lens.view(-1),
            num_seqs,
            stream,
        )
    return output


__all__ = ["flydsl_mla_decode"]
