# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pure-PyTorch reference for ``flydsl_fused_compress_attn`` /
``flydsl_hca_compress_attn``.

Mirrors the kernel's plan-driven per-boundary online-softmax pool → RMSNorm
→ GPT-J RoPE → paged cache scatter (BF16 or per-row FP8 with optional ue8m0
scale + MFMA 16x16 preshuffle).

Used by ``op_tests/test_flydsl_compress_attn.py`` to gate numerical
correctness against the flydsl kernels. Not on the inference hot path —
the Python-side per-boundary loop makes it ~100x slower than the kernel.
"""

import math
from functools import lru_cache
from typing import Optional

import torch

_PRESHUFFLE_TILE = 16


@lru_cache(maxsize=1)
def _fp8_dtype():
    from aiter.utility import dtypes as aiter_dtypes

    return aiter_dtypes.fp8


def _ue8m0_ceil_pow2(scale_f32: torch.Tensor) -> torch.Tensor:
    """Round positive scalar ``scale_f32`` up to next power of 2.

    Equivalent to the kernel's bit-trick (mantissa increment + mask): for
    exact powers of 2 the value is unchanged, otherwise the exponent is
    bumped by 1 and the mantissa cleared.
    """
    s = float(scale_f32.detach().item())
    val = math.ldexp(1.0, math.ceil(math.log2(s)))
    return torch.tensor(val, dtype=torch.float32, device=scale_f32.device)


def _preshuffled_offsets(slot_in_block: int, head_dim: int, device) -> torch.Tensor:
    """Return [head_dim] int64 offsets into the per-block region."""
    tile = _PRESHUFFLE_TILE
    token_tile_id = slot_in_block // tile
    token_in_tile = slot_in_block % tile
    d_arr = torch.arange(head_dim, device=device, dtype=torch.int64)
    col_tile_id = d_arr // tile
    col_in_tile = d_arr % tile
    return (
        token_tile_id * (tile * head_dim)
        + col_tile_id * (tile * tile)
        + token_in_tile * tile
        + col_in_tile
    )


def fused_compress_attn(
    *,
    kv_in: torch.Tensor,
    score_in: torch.Tensor,
    kv_state: torch.Tensor,
    score_state: torch.Tensor,
    plan_gpu: torch.Tensor,
    state_slot_mapping: torch.Tensor,
    ape: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    k_per_block: int,
    overlap: bool,
    ratio: int,
    head_dim: int,
    rope_head_dim: int,
    quant: bool = False,
    cache_scale: Optional[torch.Tensor] = None,
    use_ue8m0: bool = True,
    preshuffle: bool = True,
) -> None:
    """Side-effecting reference: writes into ``kv_cache`` (+ ``cache_scale``
    when ``quant=True``). Mirrors flydsl kernel's plan-driven dispatch.

    Sentinel plan rows (``position == -1``) are skipped, matching the kernel.
    """
    plan_capacity = plan_gpu.shape[0]
    if plan_capacity == 0:
        return

    K = (2 if overlap else 1) * ratio
    state_size = kv_state.shape[1]
    device = kv_in.device
    plan_cpu = plan_gpu.detach().cpu()
    slot_map_cpu = state_slot_mapping.detach().cpu()
    bt_cpu = block_tables.detach().cpu()
    rms_w_f32 = rms_weight.float()

    if quant:
        fp8_dtype = _fp8_dtype()
        fp8_max = float(torch.finfo(fp8_dtype).max)
        if kv_cache.dtype != fp8_dtype:
            raise TypeError(
                f"quant=True expects kv_cache dtype {fp8_dtype}, got {kv_cache.dtype}"
            )
        kv_cache_flat = kv_cache.view(-1)
        kv_block_stride = kv_cache.stride(0)

    for pid in range(plan_capacity):
        ragged_id, batch_id, position, window_len = plan_cpu[pid].tolist()
        if position < 0:
            continue
        slot = int(slot_map_cpu[batch_id].item())

        kv_rows = []
        score_rows = []
        for k in range(K):
            s = position - K + 1 + k
            col_off = head_dim if (overlap and k >= ratio) else 0
            d_slice = slice(col_off, col_off + head_dim)
            ape_row = k % ratio
            is_padding = s < 0
            is_input = k >= window_len

            if is_padding:
                kv_rows.append(
                    torch.zeros(head_dim, dtype=torch.float32, device=device)
                )
                score_rows.append(
                    torch.full(
                        (head_dim,), float("-inf"), dtype=torch.float32, device=device
                    )
                )
            elif is_input:
                in_row = ragged_id - (K - 1 - k)
                kv_rows.append(kv_in[in_row, d_slice].float())
                score_rows.append(
                    score_in[in_row, d_slice].float() + ape[ape_row, d_slice].float()
                )
            else:
                ring = s % state_size
                kv_rows.append(kv_state[slot, ring, d_slice].float())
                score_rows.append(score_state[slot, ring, d_slice].float())

        kv_stack = torch.stack(kv_rows, dim=0)
        sc_stack = torch.stack(score_rows, dim=0)
        weights = torch.softmax(sc_stack, dim=0)
        compressed = (weights * kv_stack).sum(dim=0)

        var = (compressed * compressed).mean()
        normed = compressed * torch.rsqrt(var + rms_eps) * rms_w_f32

        comp_pos = (position // ratio) * ratio
        rope_seg = normed[-rope_head_dim:].clone()
        cos_v = cos_cache[comp_pos].reshape(-1).float()
        sin_v = sin_cache[comp_pos].reshape(-1).float()
        even = rope_seg[0::2]
        odd = rope_seg[1::2]
        new_even = even * cos_v - odd * sin_v
        new_odd = odd * cos_v + even * sin_v
        rotated_seg = torch.stack([new_even, new_odd], dim=-1).flatten()
        normed = normed.clone()
        normed[-rope_head_dim:] = rotated_seg

        ci = position // ratio
        block_in_seq = ci // k_per_block
        slot_in_block = ci % k_per_block
        physical = int(bt_cpu[batch_id, block_in_seq].item())

        if quant:
            # Mirror kernel exactly: ``am_safe * (1.0/fp8_max)`` as a fp32 mul
            # with a pre-folded fp32 reciprocal constant, NOT ``amax/fp8_max``
            # (fp32 div differs in the last bit and would shift ue8m0 boundary
            # rounding, breaking bit-exact cache_scale comparison).
            amax = normed.abs().max()
            am_safe = torch.clamp(amax, min=1e-4)
            inv_fp8_max = torch.tensor(
                1.0 / fp8_max, dtype=torch.float32, device=device
            )
            scale_raw = am_safe * inv_fp8_max
            scale = _ue8m0_ceil_pow2(scale_raw) if use_ue8m0 else scale_raw
            inv_scale = 1.0 / scale
            scaled = torch.clamp(normed * inv_scale, -fp8_max, fp8_max)
            fp8_val = scaled.to(fp8_dtype)
            if preshuffle:
                offs = _preshuffled_offsets(slot_in_block, head_dim, device)
                base = physical * kv_block_stride
                kv_cache_flat[base + offs] = fp8_val
            else:
                kv_cache[physical, slot_in_block] = fp8_val
            cache_scale[physical, slot_in_block] = scale.float()
        else:
            kv_cache[physical, slot_in_block] = normed.to(kv_cache.dtype)
