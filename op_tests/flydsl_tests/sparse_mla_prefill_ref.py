# SPDX-License-Identifier: MIT
"""Shared helpers for sparse MLA prefill tests and benchmarks (no vLLM dep)."""

from __future__ import annotations

import math

import torch

NOPE_HEAD_DIM = 448
ROPE_HEAD_DIM = 64
PACKED_HEAD_DIM = NOPE_HEAD_DIM + ROPE_HEAD_DIM
H_PROD = 128
_FNUZ = torch.float8_e4m3fnuz
_OCP = torch.float8_e4m3fn


def pack_fp8_ds_mla_cache(
    kv: torch.Tensor,
    block_size: int,
    *,
    is_extra: bool = False,
    scale_byte: int | list[int] = 127,
) -> torch.Tensor:
    assert kv.shape[-1] == PACKED_HEAD_DIM
    num_tokens = kv.shape[0]
    num_blocks = (num_tokens + block_size - 1) // block_size
    cache = torch.zeros((num_blocks, block_size, 584), dtype=torch.uint8, device=kv.device)
    cache_flat = cache.view(torch.uint8).flatten()
    nope_dtype = _OCP if is_extra else _FNUZ
    kv_nope_fp8 = kv[:, :NOPE_HEAD_DIM].to(nope_dtype).view(torch.uint8)
    kv_rope_u8 = kv[:, NOPE_HEAD_DIM:].contiguous().view(torch.uint8)
    for slot in range(num_tokens):
        block_idx = slot // block_size
        pos = slot % block_size
        block_base = block_idx * cache.stride(0)
        token_base = block_base + pos * 576
        scale_base = block_base + block_size * 576 + pos * 8
        cache_flat[token_base : token_base + NOPE_HEAD_DIM].copy_(kv_nope_fp8[slot])
        cache_flat[
            token_base + NOPE_HEAD_DIM : token_base + NOPE_HEAD_DIM + ROPE_HEAD_DIM * 2
        ].copy_(kv_rope_u8[slot])
        if isinstance(scale_byte, int):
            cache_flat[scale_base : scale_base + 7].fill_(scale_byte)
        else:
            sb = torch.tensor(list(scale_byte), dtype=torch.uint8, device=kv.device)
            cache_flat[scale_base : scale_base + 7].copy_(sb)
    return cache


def ragged_from_rows(rows: list[list[int]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    flat = [slot for row in rows for slot in row]
    indptr = [0]
    for row in rows:
        indptr.append(indptr[-1] + len(row))
    return (
        torch.tensor(flat, dtype=torch.int32, device=device),
        torch.tensor(indptr, dtype=torch.int32, device=device),
    )


def identity_block_table(num_slots: int, block_size: int, device: torch.device) -> torch.Tensor:
    num_blocks = (num_slots + block_size - 1) // block_size
    return torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(1, num_blocks)


def gen_kv(num_tokens: int, seed: int, device: str = "cuda", scale: float = 0.125) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return (
        torch.randn(num_tokens, PACKED_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device)
        * scale
    )


def gen_q(sq: int, h: int, seed: int, device: str = "cuda", scale: float = 0.125) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(sq, h, PACKED_HEAD_DIM, generator=g, dtype=torch.bfloat16, device=device) * scale


def gen_ragged_rows(
    num_queries: int,
    topk: int,
    num_slots: int,
    seed: int,
    device: str = "cuda",
) -> list[list[int]]:
    g = torch.Generator(device=device).manual_seed(seed)
    rows: list[list[int]] = []
    for _ in range(num_queries):
        slots = torch.randint(0, num_slots, (topk,), generator=g, device=device)
        rows.append(slots.tolist())
    return rows


def merge_two_region_csrs(
    main_rows: list[list[int]],
    extra_rows: list[list[int]],
    extra_offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge per-query main+extra slot lists into one flat CSR (main then extra)."""
    device = torch.device("cuda")
    merged: list[list[int]] = []
    for m_row, e_row in zip(main_rows, extra_rows):
        merged.append(list(m_row) + [int(s) + extra_offset for s in e_row])
    return ragged_from_rows(merged, device)


def default_scale() -> float:
    return PACKED_HEAD_DIM ** -0.5
