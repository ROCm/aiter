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

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
HEAD_DIM = 128
ROPE_DIM = 64
BLOCK_SIZE = 16
MAX_POSITION = 128
EPSILON = 1e-6
WEIGHTS_SCALE = HEAD_DIM**-0.5 * 32**-0.5


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

    valid = slot_mapping >= 0
    active_q = torch.ones_like(valid) if compute_all_q_rope else valid
    safe_positions = positions.clamp(0, cos_cache.shape[0] - 1)

    q_rope = _apply_rope(q, safe_positions, cos_cache, sin_cache, is_neox)
    q_quant, q_scale = _quantize_ue8m0(q_rope, 1e-10)
    q_out[active_q] = q_quant[active_q]
    weights_out[active_q] = (
        weights[active_q].float() * q_scale[active_q] * WEIGHTS_SCALE
    )

    if valid.any():
        k_valid = k[valid].float()
        mean = k_valid.mean(dim=-1, keepdim=True)
        centered = k_valid - mean
        inv_std = torch.rsqrt(centered.square().mean(dim=-1, keepdim=True) + EPSILON)
        k_norm = (centered * inv_std * norm_weight.float() + norm_bias.float()).to(
            k.dtype
        )
        k_rope = _apply_rope(
            k_norm,
            safe_positions[valid],
            cos_cache,
            sin_cache,
            is_neox,
        )
        k_quant, k_scale = _quantize_ue8m0(k_rope, 1e-4)
        valid_slots = slot_mapping[valid]
        cache_flat = kv_cache.view(num_blocks, -1)
        for row, slot in enumerate(valid_slots.tolist()):
            block, offset = divmod(slot, BLOCK_SIZE)
            data_start = offset * HEAD_DIM
            cache_flat[block, data_start : data_start + HEAD_DIM] = k_quant[row]
            scale_start = BLOCK_SIZE * HEAD_DIM + offset * 4
            cache_flat[block, scale_start : scale_start + 4].view(torch.float32)[0] = (
                k_scale[row]
            )

    return q_out, weights_out, kv_cache


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
):
    num_blocks = max(1, (q.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE)
    q_out = torch.zeros_like(q, dtype=dtypes.fp8)
    weights_out = torch.zeros_like(weights, dtype=torch.float32)
    kv_cache = torch.zeros((num_blocks, BLOCK_SIZE, HEAD_DIM + 4), dtype=dtypes.fp8)

    def run():
        extra = (
            {}
            if compute_all_q_rope is None
            else {"compute_all_q_rope": compute_all_q_rope}
        )
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
            HEAD_DIM,
            "ue8m0",
            WEIGHTS_SCALE,
            preshuffle=False,
            is_neox=is_neox,
            **extra,
        )
        return q_out, weights_out, kv_cache

    return run


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
    # DCP non-owner rows may carry stale positions. The compute-all path must
    # clamp these rows before applying RoPE; the default path must skip them.
    if num_valid < num_tokens:
        stale = torch.tensor([-7, MAX_POSITION, MAX_POSITION + 99], dtype=torch.int64)
        positions[num_valid:] = stale[
            torch.arange(num_tokens - num_valid) % stale.numel()
        ]

    ref = run_torch(
        q,
        weights,
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
    candidates = {
        "hip": _make_aiter_candidate(
            q,
            weights,
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
    }
    if not compute_all_q_rope:
        # The omitted argument must remain identical to explicit False for
        # backward compatibility with existing callers.
        candidates["hip_default"] = _make_aiter_candidate(
            q,
            weights,
            k,
            slot_mapping,
            norm_weight,
            norm_bias,
            positions,
            cos_cache,
            sin_cache,
            None,
            is_neox,
        )

    active_q = num_tokens if compute_all_q_rope else num_valid
    q_ops = active_q * num_heads * (ROPE_DIM * 3 + HEAD_DIM * 2 + 2)
    k_ops = num_valid * (HEAD_DIM * 6 + ROPE_DIM * 3 + HEAD_DIM)
    flops = q_ops + k_ops
    nbytes = (
        q.numel() * q.element_size()
        + weights.numel() * weights.element_size()
        + k.numel() * k.element_size()
        + ref[0].numel() * ref[0].element_size()
        + ref[1].numel() * ref[1].element_size()
        + ref[2].numel() * ref[2].element_size()
    )

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
    args = parser.parse_args()

    rows = []
    for (
        num_tokens,
        num_heads,
        dtype,
        valid_fraction,
        compute_all_q_rope,
        is_neox,
    ) in itertools.product(
        args.num_tokens,
        args.num_heads,
        args.dtype,
        args.valid_fraction,
        args.compute_all_q_rope,
        args.is_neox,
    ):
        rows.append(
            test_indexer_qk_rope_quant_and_cache(
                num_tokens,
                num_heads,
                dtype,
                valid_fraction,
                compute_all_q_rope,
                is_neox,
            )
        )

    df = pd.DataFrame(rows)
    aiter.logger.info(
        "indexer_qk_rope_quant_and_cache summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
