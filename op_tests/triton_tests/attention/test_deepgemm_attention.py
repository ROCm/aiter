# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# tests are adapted from https://github.com/deepseek-ai/DeepGEMM/blob/main/tests/test_attention.py

import random

import pytest
import torch

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.attention.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits,
    deepgemm_fp8_paged_mqa_logits_ragged_k,
    deepgemm_fp8_paged_mqa_logits_schedule,
)


def cdiv(x: int, y: int) -> int:
    return (x + y - 1) // y


def calc_diff(x: torch.Tensor, y: torch.Tensor):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def kv_cache_cast_to_fp8(x: torch.Tensor, padding: bool = False) -> torch.Tensor:
    num_blocks, block_size, num_heads, head_dim = x.shape
    assert num_heads == 1
    x_amax = x.abs().float().amax(dim=3, keepdim=True).clamp(1e-4)
    sf = x_amax / 240.0
    x_scaled = (x * (1.0 / sf)).to(torch.float8_e4m3fnuz)

    padding_size = 0 if not padding else (16 - (block_size * 4) % 16) % 16
    x_fp8 = torch.empty(
        (num_blocks, block_size * (head_dim + 4 + padding_size)),
        device=x.device,
        dtype=torch.uint8,
    )
    x_fp8[:, : block_size * head_dim] = x_scaled.view(
        num_blocks, block_size * head_dim
    ).view(dtype=torch.uint8)
    x_fp8[:, block_size * head_dim : block_size * head_dim + 4 * block_size] = sf.view(
        num_blocks, block_size
    ).view(dtype=torch.uint8)
    return x_fp8.view(num_blocks, block_size, num_heads, head_dim + 4 + padding_size)


def ref_fp8_paged_mqa_logits(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    max_model_len: int,
):
    """Mirrors DeepGEMM's ref_paged_mqa_logits with use_2d_context_lens=False."""
    batch_size, next_n, num_heads, dim = q.shape
    num_blocks, block_size, _, _ = kv_cache.shape
    logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    context_lens_list = context_lens.tolist()
    for i in range(batch_size):
        context_len = context_lens_list[i]
        q_offsets = torch.arange(context_len - next_n, context_len, device=q.device)
        weight_slice = (
            weights[i * next_n : (i + 1) * next_n, :].transpose(0, 1).contiguous()
        )

        num_blks = (context_len + block_size - 1) // block_size
        block_idxs = block_tables[i][:num_blks].long()
        kv_slice = kv_cache[block_idxs]  # [num_blks, block_size, 1, dim]
        kx = kv_slice.permute(2, 3, 0, 1).reshape(1, dim, -1).float()
        qx = q[i].transpose(0, 1).float()  # [num_heads, next_n, dim]
        s = torch.matmul(qx, kx)  # [num_heads, next_n, total_tokens]

        total_len = num_blks * block_size
        k_offsets = torch.arange(0, total_len, device=q.device)
        mask = (k_offsets[None, :] < context_len) & (
            k_offsets[None, :] <= q_offsets[:, None]
        )
        s = torch.where(mask[None, :, :], s, float("-inf"))
        s = torch.relu(s) * weight_slice[..., None].float()
        s = s.sum(dim=0)
        logits[i * next_n : (i + 1) * next_n, :total_len] = torch.where(
            k_offsets[None, :] <= q_offsets[:, None], s, float("-inf")
        )
    return logits


def make_paged_inputs(
    batch_size: int,
    next_n: int,
    heads: int,
    index_dim: int,
    avg_kv_length: int,
    blocksize: int = 1,
    padding: bool = False,
    var_ratio: float = 0.5,
    qk_datatype: torch.dtype = torch.float8_e4m3fnuz,
):
    """Build inputs for paged MQA logits kernels.

    Returns a dict with the bf16 reference tensors (``q``, ``kv_cache``),
    the FP8 packed tensors (``q_fp8``, ``kv_cache_fp8``), ``weights``, the
    per-batch ``context_lens`` and ``block_tables`` (used by the non-ragged
    kernel), and the ragged-form ``prefix_sum_context_lens`` / ``kv_indices``
    (used by the ragged kernel).
    """
    max_model_len = 2 * avg_kv_length
    num_blocks = (max_model_len + blocksize - 1) // blocksize

    context_lens = (
        torch.randint(
            int((1 - var_ratio) * avg_kv_length),
            int((1 + var_ratio) * avg_kv_length) + 1,
            (batch_size,),
        )
        .cuda()
        .to(torch.int32)
    )
    prefix_sum_context_lens = torch.zeros(
        (batch_size + 1,), device="cuda", dtype=torch.int32
    )
    prefix_sum_context_lens[1:] = torch.cumsum(context_lens, dim=0)

    q = torch.randn(
        (batch_size, next_n, heads, index_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    kv_cache = torch.randn(
        (num_blocks, blocksize, 1, index_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        (batch_size * next_n, heads),
        device="cuda",
        dtype=torch.float32,
    )

    max_block_len = (context_lens.max().item() + blocksize - 1) // blocksize * blocksize
    block_tables = torch.zeros(
        (batch_size, max_block_len), device="cuda", dtype=torch.int32
    )
    counter = 0
    block_idx_pool = list(range(num_blocks))
    random.shuffle(block_idx_pool)
    for i in range(batch_size):
        ctx_len = context_lens[i].item()
        for j in range(cdiv(ctx_len, blocksize)):
            block_tables[i][j] = block_idx_pool[counter % num_blocks]
            counter += 1

    q_fp8 = q.to(qk_datatype)
    kv_cache_fp8 = kv_cache_cast_to_fp8(kv_cache, padding=padding)

    # kv_indices is the ragged-form (flat per-token) view of block_tables.
    # Coherence with block_tables means the ragged kernel and the non-ragged
    # ref read the same physical KV rows, so a single ref works for both.
    # Only well-defined for blocksize=1 (the ragged kernel's effective layout).
    kv_indices = torch.zeros(
        prefix_sum_context_lens[-1], device="cuda", dtype=torch.int32
    )
    for i in range(batch_size):
        ctx_len = int(context_lens[i].item())
        kv_indices[prefix_sum_context_lens[i] : prefix_sum_context_lens[i + 1]] = (
            block_tables[i, :ctx_len]
        )

    return dict(
        q=q,
        q_fp8=q_fp8,
        kv_cache=kv_cache,
        kv_cache_fp8=kv_cache_fp8,
        weights=weights,
        context_lens=context_lens,
        prefix_sum_context_lens=prefix_sum_context_lens,
        block_tables=block_tables,
        kv_indices=kv_indices,
        max_model_len=max_model_len,
        num_blocks=num_blocks,
        blocksize=blocksize,
    )


def causal_logits_mask(
    batch_size: int,
    next_n: int,
    max_model_len: int,
    context_lens: torch.Tensor,
) -> torch.Tensor:
    """Mask of valid output positions (per-row causal up to context_len)."""
    positions = (
        torch.arange(max_model_len, device="cuda")
        .unsqueeze(0)
        .expand(batch_size * next_n, -1)
    )
    row_indices = torch.arange(batch_size * next_n, device="cuda") // next_n
    next_n_offset = torch.arange(batch_size * next_n, device="cuda") % next_n
    return positions <= (context_lens[row_indices] - next_n + next_n_offset).unsqueeze(
        1
    )


def apply_preshuffle(kv_cache_fp8: torch.Tensor, blocksize: int, index_dim: int):
    """In-place preshuffle of the FP8 KV cache used by the preshuffle kernel."""
    kv_num_block, kv_block_size, _, kv_index_dim = kv_cache_fp8.size()
    split_kv_cache = kv_cache_fp8.view(-1, blocksize * kv_index_dim)
    split_kv_cache_data = shuffle_weight(
        split_kv_cache[..., : kv_block_size * index_dim]
        .contiguous()
        .view([kv_num_block, kv_block_size, index_dim])
    )
    split_kv_cache[..., : kv_block_size * index_dim] = split_kv_cache_data.view(
        kv_num_block, kv_block_size * index_dim
    )


# Representative shape set: covers single-batch decode, MTP, and multi-batch decode.
_PAGED_SHAPES = [
    (1, 1, 64, 128, 1024),
    (1, 2, 64, 128, 1024),
    (2, 1, 64, 128, 2048),
    (2, 2, 64, 128, 2048),
    (4, 2, 64, 128, 4096),
]


@pytest.mark.parametrize(
    "batch_size, next_n, heads, index_dim, avg_kv_length",
    _PAGED_SHAPES,
)
@pytest.mark.parametrize(
    "preshuffle, blocksize",
    [(False, 1), (True, 16)],
)
@pytest.mark.parametrize("var_ctx_opt", [False, True])
@pytest.mark.parametrize("padding", [False, True])
@torch.inference_mode()
def test_deepgemm_fp8_paged_mqa_logits(
    batch_size: int,
    next_n: int,
    heads: int,
    index_dim: int,
    avg_kv_length: int,
    preshuffle: bool,
    blocksize: int,
    var_ctx_opt: bool,
    padding: bool,
) -> None:
    torch.manual_seed(0)
    random.seed(0)

    ChunkK = 128
    WavePerEU = 5

    inputs = make_paged_inputs(
        batch_size,
        next_n,
        heads,
        index_dim,
        avg_kv_length,
        blocksize=blocksize,
        padding=padding,
    )

    q = inputs["q"]
    q_fp8 = inputs["q_fp8"]
    kv_cache = inputs["kv_cache"]
    kv_cache_fp8 = inputs["kv_cache_fp8"]
    weights = inputs["weights"]
    context_lens = inputs["context_lens"]
    block_tables = inputs["block_tables"]
    max_model_len = inputs["max_model_len"]

    ref_logits = ref_fp8_paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )

    if preshuffle:
        apply_preshuffle(kv_cache_fp8, blocksize, index_dim)

    safe_chunks_per_cta = None
    if var_ctx_opt:
        safe_chunks_per_cta = deepgemm_fp8_paged_mqa_logits_schedule(
            batch_size,
            next_n,
            context_lens,
            max_model_len,
            ChunkK=ChunkK,
            WavePerEU=WavePerEU,
        )

    out_logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )
    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        kv_cache_fp8,
        weights,
        out_logits,
        context_lens,
        block_tables,
        max_model_len,
        ChunkK=ChunkK,
        Preshuffle=preshuffle,
        KVBlockSize=blocksize,
        WavePerEU=WavePerEU,
        VarCtxSchedule=safe_chunks_per_cta,
    )

    mask = causal_logits_mask(batch_size, next_n, max_model_len, context_lens)
    out_logits = out_logits.masked_fill(~mask, 0)
    ref_logits = ref_logits.masked_fill(~mask, 0)
    diff = calc_diff(out_logits, ref_logits)
    assert diff < 1e-3, f"{diff=}"


_RAGGED_SHAPES = [
    (1, 1, 64, 128, 1024),
    (1, 2, 64, 128, 2048),
    (2, 2, 64, 128, 4096),
]


@pytest.mark.parametrize(
    "batch_size, next_n, heads, index_dim, avg_kv_length",
    _RAGGED_SHAPES,
)
@torch.inference_mode()
def test_deepgemm_fp8_paged_mqa_logits_ragged(
    batch_size: int,
    next_n: int,
    heads: int,
    index_dim: int,
    avg_kv_length: int,
) -> None:
    torch.manual_seed(0)
    random.seed(0)

    inputs = make_paged_inputs(
        batch_size,
        next_n,
        heads,
        index_dim,
        avg_kv_length,
        blocksize=1,
        padding=False,
    )

    q = inputs["q"]
    q_fp8 = inputs["q_fp8"]
    kv_cache = inputs["kv_cache"]
    kv_cache_fp8 = inputs["kv_cache_fp8"]
    weights = inputs["weights"]
    context_lens = inputs["context_lens"]
    block_tables = inputs["block_tables"]
    prefix_sum_context_lens = inputs["prefix_sum_context_lens"]
    kv_indices = inputs["kv_indices"]
    max_model_len = inputs["max_model_len"]

    # The ragged kernel uses kv_indices + prefix_sum to address KV rows; the
    # non-ragged ref uses block_tables + context_lens. make_paged_inputs builds
    # them coherently (kv_indices is the flat view of block_tables for
    # blocksize=1), so both schemes resolve to the same KV rows and the same
    # ref serves both kernels.
    ref_logits = ref_fp8_paged_mqa_logits(
        q, kv_cache, weights, context_lens, block_tables, max_model_len
    )

    out_logits = torch.full(
        (batch_size * next_n, max_model_len),
        float("-inf"),
        device="cuda",
        dtype=torch.float32,
    )
    deepgemm_fp8_paged_mqa_logits_ragged_k(
        q_fp8,
        kv_cache_fp8,
        weights,
        out_logits,
        prefix_sum_context_lens,
        kv_indices,
        max_model_len,
    )

    mask = causal_logits_mask(batch_size, next_n, max_model_len, context_lens)
    out_logits = out_logits.masked_fill(~mask, 0)
    ref_logits = ref_logits.masked_fill(~mask, 0)
    diff = calc_diff(out_logits, ref_logits)
    assert diff < 1e-3, f"{diff=}"
