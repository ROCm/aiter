# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility.fp4_utils import f32_to_mxfp4

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
HEAD_DIM = 128
ROPE_DIM = 64
BLOCK_SIZE = 16
MAX_POSITION = 128
EPSILON = 1e-6
WEIGHTS_SCALE = HEAD_DIM**-0.5 * 32**-0.5

# FP4 output mode (pa_mqa_logits_fp4 ABI).
FP4_GROUP_SIZE = 32
FP4_MAX = 6.0
FP4_KV_BLOCK_SIZE = 64
MFMA_M = 16


def _apply_rope(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    is_neox: bool,
) -> torch.Tensor:
    rope = x[..., :ROPE_DIM].float()
    tail = x[..., ROPE_DIM:]
    cos = cos_cache[positions].float()
    sin = sin_cache[positions].float()
    while cos.ndim < rope.ndim:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)

    if is_neox:
        x1, x2 = rope.chunk(2, dim=-1)
    else:
        x1, x2 = rope[..., ::2], rope[..., 1::2]
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    rotated = (
        torch.cat((y1, y2), dim=-1)
        if is_neox
        else torch.stack((y1, y2), dim=-1).flatten(-2)
    )
    return torch.cat((rotated.to(x.dtype), tail), dim=-1)


def _quantize_ue8m0(
    x: torch.Tensor, min_amax: float
) -> tuple[torch.Tensor, torch.Tensor]:
    fp8_max = torch.finfo(dtypes.fp8).max
    scale = x.float().abs().amax(dim=-1).clamp(min=min_amax) / fp8_max
    scale = torch.pow(2.0, torch.ceil(torch.log2(scale)))
    return (x.float() / scale.unsqueeze(-1)).to(dtypes.fp8), scale


def _layernorm(k: torch.Tensor, norm_weight, norm_bias) -> torch.Tensor:
    k_f32 = k.float()
    mean = k_f32.mean(dim=-1, keepdim=True)
    centered = k_f32 - mean
    inv_std = torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + EPSILON)
    return (centered * inv_std * norm_weight.float() + norm_bias.float()).to(k.dtype)


def _make_inputs(num_tokens, num_heads, dtype, valid_fraction):
    torch.manual_seed(1)
    q = torch.randn(num_tokens, num_heads, HEAD_DIM, dtype=dtype)
    weights = torch.randn(num_tokens, num_heads, dtype=dtype)
    k = torch.randn(num_tokens, HEAD_DIM, dtype=dtype)
    norm_weight = torch.randn(HEAD_DIM, dtype=torch.float32)
    norm_bias = torch.randn(HEAD_DIM, dtype=torch.float32)
    angles = torch.randn(MAX_POSITION, ROPE_DIM // 2, dtype=torch.float32)
    cos_cache = angles.cos().to(dtype)
    sin_cache = angles.sin().to(dtype)

    num_valid = max(1, int(num_tokens * valid_fraction))
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    slot_mapping[num_valid:] = -1
    positions = torch.arange(num_tokens, dtype=torch.int64) % MAX_POSITION
    # DCP non-owner rows may carry stale positions: the compute-all path must
    # clamp them before RoPE, the default path must skip them.
    if num_valid < num_tokens:
        stale = torch.tensor([-7, MAX_POSITION, MAX_POSITION + 99], dtype=torch.int64)
        positions[num_valid:] = stale[
            torch.arange(num_tokens - num_valid) % stale.numel()
        ]
    return (
        q,
        weights,
        k,
        slot_mapping,
        norm_weight,
        norm_bias,
        positions,
        cos_cache,
        sin_cache,
        num_valid,
    )


def _rope_preamble(
    q: torch.Tensor,
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    compute_all_q_rope: bool,
    is_neox: bool,
):
    # Shared by both oracles so the FP8 and FP4 references cannot drift apart in
    # the pre-quantization math. k_rope is zero on rows without a cache slot.
    valid = slot_mapping >= 0
    active_q = torch.ones_like(valid) if compute_all_q_rope else valid
    safe_positions = positions.clamp(0, cos_cache.shape[0] - 1)
    q_rope = _apply_rope(q, safe_positions, cos_cache, sin_cache, is_neox)
    k_rope = torch.zeros_like(k)
    if valid.any():
        k_rope[valid] = _apply_rope(
            _layernorm(k[valid], norm_weight, norm_bias),
            safe_positions[valid],
            cos_cache,
            sin_cache,
            is_neox,
        )
    return valid, active_q, q_rope, k_rope


def _flops(active_q: int, num_valid: int, num_heads: int) -> int:
    q_ops = active_q * num_heads * (ROPE_DIM * 3 + HEAD_DIM * 2 + 2)
    k_ops = num_valid * (HEAD_DIM * 6 + ROPE_DIM * 3 + HEAD_DIM)
    return q_ops + k_ops


def run_torch(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    compute_all_q_rope: bool,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = q.shape[0]
    num_blocks = max(1, (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE)
    q_out = torch.zeros_like(q, dtype=dtypes.fp8)
    weights_out = torch.zeros_like(weights, dtype=torch.float32)
    kv_cache = torch.zeros((num_blocks, BLOCK_SIZE, HEAD_DIM + 4), dtype=dtypes.fp8)

    valid, active_q, q_rope, k_rope = _rope_preamble(
        q,
        k,
        slot_mapping,
        norm_weight,
        norm_bias,
        positions,
        cos_cache,
        sin_cache,
        compute_all_q_rope,
        is_neox,
    )
    q_quant, q_scale = _quantize_ue8m0(q_rope, 1e-10)
    q_out[active_q] = q_quant[active_q]
    weights_out[active_q] = (
        weights[active_q].float() * q_scale[active_q] * WEIGHTS_SCALE
    )

    if valid.any():
        k_quant, k_scale = _quantize_ue8m0(k_rope[valid], 1e-4)
        cache_flat = kv_cache.view(num_blocks, -1)
        for row, slot in enumerate(slot_mapping[valid].tolist()):
            block, offset = divmod(slot, BLOCK_SIZE)
            data_start = offset * HEAD_DIM
            cache_flat[block, data_start : data_start + HEAD_DIM] = k_quant[row]
            scale_start = BLOCK_SIZE * HEAD_DIM + offset * 4
            cache_flat[block, scale_start : scale_start + 4].view(torch.float32)[0] = (
                k_scale[row]
            )

    return q_out, weights_out, kv_cache


def _alloc_fp8_outputs(num_tokens: int, num_heads: int, num_blocks: int):
    return (
        torch.zeros(num_tokens, num_heads, HEAD_DIM, dtype=dtypes.fp8),
        torch.zeros(num_tokens, num_heads, dtype=torch.float32),
        torch.zeros((num_blocks, BLOCK_SIZE, HEAD_DIM + 4), dtype=dtypes.fp8),
    )


def _make_aiter_candidate(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    compute_all_q_rope: bool | None,
    is_neox: bool,
    outs: tuple[torch.Tensor, ...],
):
    # Handing over the 5-tuple from _alloc_fp4_outputs picks FP4 mode the same way
    # a real caller does. compute_all_q_rope=None omits the argument entirely.
    fp4 = len(outs) == 5
    if fp4:
        q_out, q_scale_out, weights_out, kv_cache, kv_cache_scale = outs
    else:
        q_out, weights_out, kv_cache = outs

    def run():
        extra = (
            {}
            if compute_all_q_rope is None
            else {"compute_all_q_rope": compute_all_q_rope}
        )
        if fp4:
            extra["q_scale_out"] = q_scale_out
            extra["kv_cache_scale"] = kv_cache_scale
        aiter.indexer_qk_rope_quant_and_cache(
            q,
            q_out,
            weights,
            weights_out,
            k,
            kv_cache,
            slot_mapping,
            norm_weight,
            norm_bias,
            positions,
            cos_cache,
            sin_cache,
            EPSILON,
            FP4_GROUP_SIZE if fp4 else HEAD_DIM,
            "ue8m0",
            WEIGHTS_SCALE,
            preshuffle=False,
            is_neox=is_neox,
            **extra,
        )
        return outs

    return run


def _fp4_quant(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-32-group e2m1 + e8m0, matching the HIP `v_cvt_scalef32_pk_fp4_f32`
    rounding (round-half-even) that every AITER FP4 writer uses."""
    *prefix, d = x.shape
    blocks = x.reshape(*prefix, d // FP4_GROUP_SIZE, FP4_GROUP_SIZE).float()
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=FP4_MAX * 2.0**-126)
    scale = torch.pow(2.0, torch.ceil(torch.log2(amax * (1.0 / FP4_MAX))))
    normalized = (blocks / scale).clamp(min=-FP4_MAX, max=FP4_MAX)
    packed = f32_to_mxfp4(normalized.reshape(*prefix, d)).view(torch.uint8)
    e8m0 = (torch.log2(scale).squeeze(-1) + 127.0).to(torch.uint8)
    return packed.contiguous(), e8m0.contiguous()


def _qs_pad(num_heads: int) -> int:
    return ((num_heads // MFMA_M) + 3) & ~3


def _shuffle_q_scale(e8m0: torch.Tensor, num_heads: int) -> torch.Tensor:
    """[T, H, D/32] -> [T, K_TILES, 4, 16, QS_PAD] (dsv4 preshuffled scale)."""
    num_tokens = e8m0.shape[0]
    m_tiles = num_heads // MFMA_M
    k_tiles = HEAD_DIM // 128
    shuffled = (
        e8m0.reshape(num_tokens, m_tiles, MFMA_M, k_tiles, 4)
        .permute(0, 3, 4, 2, 1)
        .contiguous()
    )
    return torch.nn.functional.pad(
        shuffled, (0, _qs_pad(num_heads) - m_tiles)
    ).contiguous()


def _paged_k_fp4(
    packed: torch.Tensor,
    e8m0: torch.Tensor,
    slot_mapping: torch.Tensor,
    num_blocks: int,
    kv_block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scatter dense per-token FP4 K rows into the paged preshuffle layout."""
    k_tiles = HEAD_DIM // 128
    kv_cache = torch.zeros(num_blocks, k_tiles, 4, kv_block_size, 16, dtype=torch.uint8)
    kv_scale = torch.zeros(num_blocks, k_tiles, 4, kv_block_size, dtype=torch.uint8)
    valid = slot_mapping >= 0
    slots = slot_mapping[valid].long()
    if slots.numel() == 0:
        return kv_cache, kv_scale
    block = slots // kv_block_size
    pos = slots % kv_block_size
    kv_cache[block, :, :, pos, :] = packed[valid].view(-1, k_tiles, 4, 16)
    sflat = (pos % 16) * (kv_block_size // 16) + (pos // 16)
    kv_scale[block, :, :, sflat] = e8m0[valid].view(-1, k_tiles, 4)
    return kv_cache, kv_scale


def run_torch_fp4(
    q: torch.Tensor,
    weights: torch.Tensor,
    k: torch.Tensor,
    slot_mapping: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_bias: torch.Tensor,
    positions: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    num_blocks: int,
    compute_all_q_rope: bool,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens, num_heads, _ = q.shape
    _, active_q, q_rope, k_rope = _rope_preamble(
        q,
        k,
        slot_mapping,
        norm_weight,
        norm_bias,
        positions,
        cos_cache,
        sin_cache,
        compute_all_q_rope,
        is_neox,
    )

    q_out = torch.zeros(num_tokens, num_heads, HEAD_DIM // 2, dtype=torch.uint8)
    q_scale_out = torch.zeros(
        num_tokens, HEAD_DIM // 128, 4, MFMA_M, _qs_pad(num_heads), dtype=torch.uint8
    )
    weights_out = torch.zeros_like(weights)

    q_packed, q_e8m0 = _fp4_quant(q_rope.reshape(num_tokens * num_heads, HEAD_DIM))
    q_out[active_q] = q_packed.reshape(num_tokens, num_heads, HEAD_DIM // 2)[active_q]
    q_scale_out[active_q] = _shuffle_q_scale(
        q_e8m0.reshape(num_tokens, num_heads, HEAD_DIM // FP4_GROUP_SIZE), num_heads
    )[active_q]
    weights_out[active_q] = weights[active_q]

    k_packed, k_e8m0 = _fp4_quant(k_rope)
    kv_cache, kv_cache_scale = _paged_k_fp4(
        k_packed, k_e8m0, slot_mapping, num_blocks, FP4_KV_BLOCK_SIZE
    )
    return q_out, q_scale_out, weights_out, kv_cache, kv_cache_scale


def _alloc_fp4_outputs(num_tokens: int, num_heads: int, num_blocks: int, dtype):
    k_tiles = HEAD_DIM // 128
    return (
        torch.zeros(num_tokens, num_heads, HEAD_DIM // 2, dtype=torch.uint8),
        torch.zeros(
            num_tokens, k_tiles, 4, MFMA_M, _qs_pad(num_heads), dtype=torch.uint8
        ),
        torch.zeros(num_tokens, num_heads, dtype=dtype),
        torch.zeros(num_blocks, k_tiles, 4, FP4_KV_BLOCK_SIZE, 16, dtype=torch.uint8),
        torch.zeros(num_blocks, k_tiles, 4, FP4_KV_BLOCK_SIZE, dtype=torch.uint8),
    )


def _split_cache(
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_flat = kv_cache.view(kv_cache.shape[0], -1)
    data = cache_flat[:, : BLOCK_SIZE * HEAD_DIM].view(-1, BLOCK_SIZE, HEAD_DIM)
    scales = (
        cache_flat[:, BLOCK_SIZE * HEAD_DIM :]
        .view(torch.float32)
        .view(kv_cache.shape[0], BLOCK_SIZE)
    )
    return data, scales


@benchmark()
def test_indexer_qk_rope_quant_and_cache(
    num_tokens: int,
    num_heads: int,
    dtype: torch.dtype,
    valid_fraction: float,
    compute_all_q_rope: bool,
    is_neox: bool,
):
    *inputs, num_valid = _make_inputs(num_tokens, num_heads, dtype, valid_fraction)
    q, weights, k = inputs[:3]
    num_blocks = max(1, (num_tokens + BLOCK_SIZE - 1) // BLOCK_SIZE)

    ref = run_torch(*inputs, compute_all_q_rope, is_neox)
    candidates = {
        "hip": _make_aiter_candidate(
            *inputs,
            compute_all_q_rope,
            is_neox,
            _alloc_fp8_outputs(num_tokens, num_heads, num_blocks),
        )
    }
    if not compute_all_q_rope:
        # Omitting the argument must behave exactly like passing False.
        candidates["hip_default"] = _make_aiter_candidate(
            *inputs,
            None,
            is_neox,
            _alloc_fp8_outputs(num_tokens, num_heads, num_blocks),
        )

    active_q = num_tokens if compute_all_q_rope else num_valid
    flops = _flops(active_q, num_valid, num_heads)
    nbytes = sum(t.numel() * t.element_size() for t in (q, weights, k, *ref))

    ref_cache_data, ref_cache_scales = _split_cache(ref[2])
    ret = {"gfx": get_gfx()}
    for name, candidate in candidates.items():
        (q_out, weights_out, kv_cache), us = run_perftest(candidate)
        cache_data, cache_scales = _split_cache(kv_cache)
        errors = [
            checkAllclose(
                ref[0].float(),
                q_out.float(),
                rtol=0,
                atol=0,
                msg=f"{name}: q_out",
            ),
            checkAllclose(
                ref[1],
                weights_out,
                rtol=1e-5,
                atol=1e-7,
                msg=f"{name}: weights_out",
            ),
            checkAllclose(
                ref_cache_data.float(),
                cache_data.float(),
                rtol=0,
                atol=0,
                msg=f"{name}: kv_cache data",
            ),
            checkAllclose(
                ref_cache_scales,
                cache_scales,
                rtol=1e-6,
                atol=0,
                msg=f"{name}: kv_cache scale",
            ),
        ]
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = max(errors)
    return ret


@benchmark()
def test_indexer_qk_rope_quant_and_cache_fp4(
    num_tokens: int,
    num_heads: int,
    dtype: torch.dtype,
    valid_fraction: float,
    compute_all_q_rope: bool,
    is_neox: bool,
):
    *inputs, num_valid = _make_inputs(num_tokens, num_heads, dtype, valid_fraction)
    q, weights, k = inputs[:3]
    num_blocks = max(1, (num_tokens + FP4_KV_BLOCK_SIZE - 1) // FP4_KV_BLOCK_SIZE)

    ref = run_torch_fp4(*inputs, num_blocks, compute_all_q_rope, is_neox)
    candidate = _make_aiter_candidate(
        *inputs,
        compute_all_q_rope,
        is_neox,
        _alloc_fp4_outputs(num_tokens, num_heads, num_blocks, dtype),
    )
    got, us = run_perftest(candidate)

    active_q = num_tokens if compute_all_q_rope else num_valid
    flops = _flops(active_q, num_valid, num_heads)
    nbytes = sum(t.numel() * t.element_size() for t in (q, weights, k, *ref))

    # The FP4 ABI is a byte contract with the pa_mqa_logits_fp4 reader, so every
    # comparison here is exact rather than approximate.
    names = ["q_out", "q_scale_out", "weights_out", "kv_cache", "kv_cache_scale"]
    mismatch = {}
    for name, want, have in zip(names, ref, got):
        mismatch[name] = int((want != have).sum().item())

    ret = {"gfx": get_gfx(), "hip us": us}
    ret["hip TFLOPS"] = flops / us / 1e6
    ret["hip TB/s"] = nbytes / us / 1e6
    for name in names:
        ret[f"{name} mismatch"] = mismatch[name]
    assert all(v == 0 for v in mismatch.values()), f"fp4 byte mismatch: {mismatch}"
    return ret


def test_indexer_fp4_e2e_pa_mqa_logits(
    batch: int,
    next_n: int,
    ctx_len: int,
    num_heads: int = 32,
    kv_block_size: int = FP4_KV_BLOCK_SIZE,
    block_k: int = 256,
    dtype: torch.dtype = dtypes.bf16,
):
    """Feed the kernel's own FP4 Q / K cache straight into the FlyDSL FP4
    paged-MQA-logits kernel and score it against the PyTorch reference."""
    from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4
    from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
        compute_varctx_schedule,
    )
    from op_tests.test_flydsl_pa_mqa_logits_fp4 import ref_mqa_logits_mixed

    torch.manual_seed(3)
    max_blocks_per_seq = max(
        (ctx_len + block_k - 1) // block_k * (block_k // kv_block_size),
        block_k // kv_block_size,
    )
    num_blocks = max_blocks_per_seq * batch
    t_max = max_blocks_per_seq * kv_block_size
    block_tables = torch.arange(num_blocks, dtype=torch.int32).reshape(
        batch, max_blocks_per_seq
    )
    context_lens = torch.full((batch,), ctx_len, dtype=torch.int32)

    norm_weight = torch.randn(HEAD_DIM, dtype=torch.float32)
    norm_bias = torch.randn(HEAD_DIM, dtype=torch.float32)
    angles = torch.randn(MAX_POSITION, ROPE_DIM // 2, dtype=torch.float32)
    cos_cache, sin_cache = angles.cos().to(dtype), angles.sin().to(dtype)

    kv_tokens = batch * t_max
    k_all = torch.randn(kv_tokens, HEAD_DIM, dtype=dtype)
    tok = torch.arange(t_max).repeat(batch)
    bat = torch.arange(batch).repeat_interleave(t_max)
    kv_slots = (
        block_tables[bat, tok // kv_block_size].long() * kv_block_size
        + tok % kv_block_size
    )
    kv_slots = torch.where(
        tok < context_lens[bat].long(), kv_slots, torch.full_like(kv_slots, -1)
    )
    kv_positions = tok.long() % MAX_POSITION
    kv_outs = _alloc_fp4_outputs(kv_tokens, num_heads, num_blocks, dtype)
    aiter.indexer_qk_rope_quant_and_cache(
        torch.randn(kv_tokens, num_heads, HEAD_DIM, dtype=dtype),
        kv_outs[0],
        torch.randn(kv_tokens, num_heads, dtype=dtype),
        kv_outs[2],
        k_all,
        kv_outs[3],
        kv_slots,
        norm_weight,
        norm_bias,
        kv_positions,
        cos_cache,
        sin_cache,
        EPSILON,
        FP4_GROUP_SIZE,
        "ue8m0",
        WEIGHTS_SCALE,
        is_neox=True,
        q_scale_out=kv_outs[1],
        kv_cache_scale=kv_outs[4],
    )
    kv_cache, kv_cache_scale = kv_outs[3], kv_outs[4]

    # slot < 0 keeps the query rows out of the K cache.
    num_q = batch * next_n
    q_bf16 = torch.randn(num_q, num_heads, HEAD_DIM, dtype=dtype)
    weights = (torch.randn(num_q, num_heads, dtype=torch.float32) * 0.1).to(dtype)
    q_positions = torch.arange(num_q, dtype=torch.int64) % MAX_POSITION
    q_outs = _alloc_fp4_outputs(num_q, num_heads, num_blocks, dtype)
    aiter.indexer_qk_rope_quant_and_cache(
        q_bf16,
        q_outs[0],
        weights,
        q_outs[2],
        torch.zeros(num_q, HEAD_DIM, dtype=dtype),
        q_outs[3],
        torch.full((num_q,), -1, dtype=torch.int64),
        norm_weight,
        norm_bias,
        q_positions,
        cos_cache,
        sin_cache,
        EPSILON,
        FP4_GROUP_SIZE,
        "ue8m0",
        WEIGHTS_SCALE,
        is_neox=True,
        compute_all_q_rope=True,
        q_scale_out=q_outs[1],
        kv_cache_scale=q_outs[4],
    )
    q_fp4, q_scale, weights_out = q_outs[0], q_outs[1], q_outs[2]

    # The reference dequantizes the oracle's own dense FP4 values instead of
    # reading the kernel's paged bytes back through an inverse transform, so a
    # wrong scatter offset shows up here rather than cancelling out.
    kv_rope = _apply_rope(
        _layernorm(k_all, norm_weight, norm_bias),
        kv_positions,
        cos_cache,
        sin_cache,
        True,
    )
    kv_fp4_dense, kv_e8m0_dense = _fp4_quant(kv_rope)
    q_rope = _apply_rope(q_bf16, q_positions, cos_cache, sin_cache, True)
    q_fp4_dense, q_e8m0_dense = _fp4_quant(q_rope.reshape(num_q * num_heads, HEAD_DIM))
    ref_logits = ref_mqa_logits_mixed(
        q_fp4_dense.reshape(batch, next_n, num_heads, HEAD_DIM // 2),
        q_e8m0_dense.reshape(batch, next_n, num_heads, HEAD_DIM // FP4_GROUP_SIZE),
        kv_fp4_dense.reshape(batch, t_max, HEAD_DIM // 2),
        kv_e8m0_dense.reshape(batch, t_max, HEAD_DIM // FP4_GROUP_SIZE),
        weights,
        context_lens,
        next_n=next_n,
        weight_scale=WEIGHTS_SCALE,
    )

    _safe, cta_info, total_ctas = compute_varctx_schedule(
        context_lens, block_k, None, t_max, next_n=next_n
    )
    out_logits = torch.full((num_q, t_max), float("-inf"), dtype=torch.float32)
    flydsl_pa_mqa_logits_fp4(
        q_fp4.reshape(batch, next_n, num_heads, HEAD_DIM // 2),
        q_scale.reshape(batch, next_n, HEAD_DIM // 128, 4, MFMA_M, _qs_pad(num_heads)),
        kv_cache,
        kv_cache_scale,
        block_tables,
        weights_out,
        context_lens,
        t_max,
        weight_scale=WEIGHTS_SCALE,
        next_n=next_n,
        block_k=block_k,
        kv_block_size=kv_block_size,
        parallel_unit_num=total_ctas,
        out=out_logits,
        cta_info=cta_info,
        total_ctas=total_ctas,
    )
    torch.cuda.synchronize()

    mask = ~torch.isneginf(ref_logits)
    got, want = out_logits[mask].double(), ref_logits[mask].double()
    cos = (got * want).sum() / (got.norm() * want.norm() + 1e-12)

    topk = min(64, ctx_len)
    overlap = []
    for row in range(num_q):
        row_mask = mask[row]
        if int(row_mask.sum()) < topk:
            continue
        idx_ref = (
            ref_logits[row].masked_fill(~row_mask, float("-inf")).topk(topk).indices
        )
        idx_got = (
            out_logits[row].masked_fill(~row_mask, float("-inf")).topk(topk).indices
        )
        overlap.append(len(set(idx_ref.tolist()) & set(idx_got.tolist())) / topk)
    topk_overlap = sum(overlap) / max(1, len(overlap))

    ret = {
        "gfx": get_gfx(),
        "batch": batch,
        "next_n": next_n,
        "ctx_len": ctx_len,
        "num_heads": num_heads,
        "cosine": cos.item(),
        f"top{topk}_overlap": topk_overlap,
        "max_abs_err": (got - want).abs().max().item(),
    }
    assert cos.item() > 0.99, f"e2e cosine {cos.item():.6f} < 0.99"
    assert topk_overlap > 0.95, f"e2e top-{topk} overlap {topk_overlap:.4f} < 0.95"
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "indexer_qk_rope_quant_and_cache unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Test fused indexer Q/K RoPE, quantization, and cache writes",
    )
    parser.add_argument(
        "-n",
        "--num_tokens",
        type=int,
        nargs="*",
        default=[8, 32],
        help="Number of tokens. e.g.: -n 8 32",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        nargs="*",
        default=[32, 64],
        help="Number of indexer heads. e.g.: --num_heads 32 64",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.bf16],
        nargs="*",
        default="bf16,",
        help="Input dtype. e.g.: -d bf16",
    )
    parser.add_argument(
        "--valid_fraction",
        type=float,
        nargs="*",
        default=[0.5],
        help="Fraction of rows with valid cache slots. e.g.: --valid_fraction 0.5",
    )
    parser.add_argument(
        "--compute_all_q_rope",
        type=dtypes.str2bool,
        nargs="*",
        default=[False, True],
        help="Compute Q/weights for slot=-1 rows. e.g.: --compute_all_q_rope 0 1",
    )
    parser.add_argument(
        "--is_neox",
        type=dtypes.str2bool,
        nargs="*",
        default=[True, False],
        help="RoPE layout. e.g.: --is_neox 1 0",
    )
    parser.add_argument(
        "--mode",
        type=str,
        nargs="*",
        choices=["fp8", "fp4", "e2e"],
        default=["fp8", "fp4", "e2e"],
        help="Output modes to exercise. e.g.: --mode fp8 fp4",
    )
    args = parser.parse_args()

    sweep = list(
        itertools.product(
            args.num_tokens,
            args.num_heads,
            args.dtype,
            args.valid_fraction,
            args.compute_all_q_rope,
            args.is_neox,
        )
    )

    if "fp8" in args.mode:
        rows = [test_indexer_qk_rope_quant_and_cache(*cfg) for cfg in sweep]
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "indexer_qk_rope_quant_and_cache summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

    if "fp4" in args.mode:
        rows = [test_indexer_qk_rope_quant_and_cache_fp4(*cfg) for cfg in sweep]
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "indexer_qk_rope_quant_and_cache fp4 summary (markdown):\n%s",
            df.to_markdown(index=False),
        )

    # The FP4 MQA-logits kernel is gfx950-only, so the end-to-end leg is skipped
    # elsewhere even though the fused op itself runs on gfx942.
    if "e2e" in args.mode and get_gfx() == "gfx950":
        rows = [
            test_indexer_fp4_e2e_pa_mqa_logits(batch, next_n, ctx_len)
            for batch, next_n, ctx_len in [(2, 1, 512), (2, 4, 512), (3, 1, 1024)]
        ]
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "indexer fp4 -> flydsl_pa_mqa_logits_fp4 summary (markdown):\n%s",
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
