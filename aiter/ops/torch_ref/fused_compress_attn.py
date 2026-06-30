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

# FP4 (E2M1) MX block-scale group size — must match the kernel's _FP4_GROUP_SIZE.
_FP4_GROUP_SIZE = 32
_FP4_K_TILE = 128
# fp32 bit pattern of 1.0 / 6.0 (max_normal of FP4 E2M1), matching the FlyDSL
# emit_mx_e8m0_scale RoundUp reciprocal — used for bit-exact e8m0 scales.
_FP4_INV_MAX_POS_BITS = 0x3E2AAAAB


@lru_cache(maxsize=1)
def _fp8_dtype():
    from aiter.utility import dtypes as aiter_dtypes

    return aiter_dtypes.fp8


def _fp4_e8m0_roundup(amax: torch.Tensor) -> torch.Tensor:
    """Bit-exact mirror of ``quant_utils.emit_mx_e8m0_scale(RoundUp, FP4_E2M1)``.

    Returns the biased e8m0 exponent as uint8 (range [0, 0xFF]). The kernel
    multiplies by the fp32 reciprocal bit pattern of 1/6 (NOT a float divide),
    then ceil-bumps the exponent when any mantissa bit is set.
    """
    inv = torch.tensor(
        _FP4_INV_MAX_POS_BITS, dtype=torch.int32, device=amax.device
    ).view(torch.float32)
    working = (amax.float() * inv).view(torch.int32)
    mantissa = working & 0x7FFFFF
    biased_exp = (working >> 23) & 0xFF
    exp_field = biased_exp + (mantissa != 0).to(torch.int32)
    return exp_field.clamp_(0, 0xFF).to(torch.uint8)


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
    quant_mode: Optional[str] = None,
) -> None:
    """Side-effecting reference: writes into ``kv_cache`` (+ ``cache_scale``
    when quantizing). Mirrors flydsl kernel's plan-driven dispatch.

    ``quant_mode`` ∈ {"none","fp8","fp4"} (default derived from ``quant``).
    FP4 uses per-group(32) e8m0 scale + FP4 KV preshuffle layout.

    Sentinel plan rows (``position == -1``) are skipped, matching the kernel.
    """
    plan_capacity = plan_gpu.shape[0]
    if plan_capacity == 0:
        return

    mode = quant_mode if quant_mode is not None else ("fp8" if quant else "none")
    _fp4 = mode == "fp4"
    _fp8 = mode == "fp8"

    K = (2 if overlap else 1) * ratio
    state_size = kv_state.shape[1]
    device = kv_in.device
    plan_cpu = plan_gpu.detach().cpu()
    slot_map_cpu = state_slot_mapping.detach().cpu()
    bt_cpu = block_tables.detach().cpu()
    rms_w_f32 = rms_weight.float()

    if _fp8:
        fp8_dtype = _fp8_dtype()
        fp8_max = float(torch.finfo(fp8_dtype).max)
        if kv_cache.dtype != fp8_dtype:
            raise TypeError(
                f"fp8 quant expects kv_cache dtype {fp8_dtype}, got {kv_cache.dtype}"
            )
        kv_cache_flat = kv_cache.view(-1)
        kv_block_stride = kv_cache.stride(0)
    elif _fp4:
        from aiter.utility.fp4_utils import f32_to_mxfp4

        k_tiles = head_dim // _FP4_K_TILE
        kvbs = k_per_block
        if kv_cache.dtype != torch.uint8:
            raise TypeError(f"fp4 quant expects uint8 kv_cache, got {kv_cache.dtype}")
        kv_cache_flat = kv_cache.reshape(-1)  # [NB*k_tiles*4*kvbs*16] u8
        scale_flat = cache_scale.reshape(-1)  # [NB*k_tiles*4*kvbs] u8

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

        if _fp8:
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
        elif _fp4:
            # Per-group(32) e8m0 RoundUp scale (bit-exact w/ kernel), then
            # E2M1 pack via the shared CPU ref. quant_scale = (254-e8m0)<<23.
            ng = head_dim // _FP4_GROUP_SIZE
            groups = normed.view(ng, _FP4_GROUP_SIZE)
            amax_g = torch.clamp(
                groups.abs().amax(dim=1), min=6.0 * float.fromhex("0x1p-126")
            )
            e8m0 = _fp4_e8m0_roundup(amax_g)  # [ng] uint8
            quant_exp = (254 - e8m0.to(torch.int32)) << 23
            qscale = quant_exp.view(torch.float32)  # [ng]
            scaled = (groups * qscale[:, None]).reshape(head_dim)
            # f32_to_mxfp4 packs 2 nibbles/byte → [head_dim//2] uint8.
            packed = f32_to_mxfp4(scaled).view(torch.uint8).reshape(-1)
            for b in range(head_dim // 2):
                if preshuffle:
                    k_tile = b // 64
                    rem = b % 64
                    group4 = rem // 16
                    sub16 = rem % 16
                    off = (
                        physical * (k_tiles * 4 * kvbs * 16)
                        + k_tile * (4 * kvbs * 16)
                        + group4 * (kvbs * 16)
                        + slot_in_block * 16
                        + sub16
                    )
                else:
                    flat_slot = physical * k_per_block + slot_in_block
                    off = flat_slot * (head_dim // 2) + b
                kv_cache_flat[off] = packed[b]
            # Scale slot axis is INTERLEAVED (sflat = (slot%16)*4 + slot//16)
            # so the mqa-logits reader's packed-dword load is contiguous —
            # matches the kernel writer + op-test reference + packed readers.
            sflat = (slot_in_block % 16) * 4 + (slot_in_block // 16)
            for g in range(ng):
                if preshuffle:
                    k_tile_s = g // 4
                    group4_s = g % 4
                    s_off = (
                        physical * (k_tiles * 4 * kvbs)
                        + k_tile_s * (4 * kvbs)
                        + group4_s * kvbs
                        + sflat
                    )
                else:
                    flat_slot = physical * k_per_block + slot_in_block
                    s_off = flat_slot * ng + g
                scale_flat[s_off] = e8m0[g]
        else:
            kv_cache[physical, slot_in_block] = normed.to(kv_cache.dtype)
