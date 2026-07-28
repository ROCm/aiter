# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and latency sweep for the FlyDSL fused QK norm/MRoPE/cache op."""

import argparse
import itertools
import sys
from pathlib import Path

import torch

import aiter
from aiter.ops.flydsl import (
    flydsl_fused_qk_norm_mrope_3d_cache_pts_quant_shuffle,
)
from aiter.test_common import checkAllclose, run_perftest
from aiter.utility import dtypes

sys.path.insert(0, str(Path(__file__).resolve().parent))

EPS = 1e-6
MAX_POSITIONS = 4096
MROPE_SECTIONS = {
    64: [12, 10, 10],
    128: [24, 20, 20],
    256: [48, 40, 40],
}


def run_case(
    *,
    num_tokens: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype: torch.dtype,
    page_size: int,
    interleaved: bool,
    slot_pattern: str,
    strided_positions: bool,
    gemma_norm: bool,
    return_kv: bool,
    seed: int,
    warmup: int,
    iters: int,
) -> None:
    torch.manual_seed(seed)
    device = "cuda"
    num_blocks = max(4, (num_tokens + page_size - 1) // page_size)
    x = 16 // torch.empty((), dtype=cache_dtype).element_size()
    sections = MROPE_SECTIONS[head_size]

    qkv = torch.randn(
        num_tokens,
        num_q_heads + 2 * num_kv_heads,
        head_size,
        dtype=torch.bfloat16,
        device=device,
    )
    qw = torch.randn(head_size, dtype=torch.bfloat16, device=device)
    kw = torch.randn(head_size, dtype=torch.bfloat16, device=device)
    cos_sin = (
        torch.randn(MAX_POSITIONS, head_size, dtype=torch.bfloat16, device=device)
        * 0.25
    )
    positions_storage = torch.randint(
        0,
        MAX_POSITIONS,
        (3, num_tokens * (2 if strided_positions else 1)),
        dtype=torch.int64,
        device=device,
    )
    positions = positions_storage[:, ::2] if strided_positions else positions_storage

    if slot_pattern == "aligned":
        slots = torch.arange(num_tokens, dtype=torch.int64, device=device)
    else:
        slots = torch.randperm(
            num_blocks * page_size, dtype=torch.int64, device=device
        )[:num_tokens]
        if slot_pattern == "negative":
            slots[0] = -1

    cache_shape = (num_blocks, page_size, num_kv_heads, head_size)
    initial_k_cache = torch.randn(cache_shape, dtype=torch.bfloat16, device=device).to(
        cache_dtype
    )
    initial_v_cache = torch.randn(cache_shape, dtype=torch.bfloat16, device=device).to(
        cache_dtype
    )
    initial_k_out = (
        torch.randn(
            num_tokens,
            num_kv_heads,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        ).to(cache_dtype)
        if return_kv
        else None
    )
    initial_v_out = (
        torch.randn(
            num_tokens,
            num_kv_heads,
            head_size,
            dtype=torch.bfloat16,
            device=device,
        ).to(cache_dtype)
        if return_kv
        else None
    )
    k_scale = torch.tensor(1.5, dtype=torch.float32, device=device)
    v_scale = torch.tensor(2.0, dtype=torch.float32, device=device)

    q_fly = torch.empty(
        num_tokens, num_q_heads, head_size, dtype=torch.bfloat16, device=device
    )
    k_fly = initial_k_cache.clone()
    v_fly = initial_v_cache.clone()
    k_out_fly = initial_k_out.clone() if return_kv else None
    v_out_fly = initial_v_out.clone() if return_kv else None

    q_hip = torch.empty_like(q_fly)
    k_hip = initial_k_cache.clone()
    v_hip = initial_v_cache.clone()
    k_out_hip = initial_k_out.clone() if return_kv else None
    v_out_hip = initial_v_out.clone() if return_kv else None

    common = (
        qw,
        kw,
        cos_sin,
        positions,
        num_tokens,
        num_q_heads,
        num_kv_heads,
        num_kv_heads,
        head_size,
        True,
        sections,
        interleaved,
        EPS,
    )
    suffix = (
        return_kv,
        True,
        page_size,
        x,
        head_size,
        gemma_norm,
    )

    _, flydsl_us = run_perftest(
        flydsl_fused_qk_norm_mrope_3d_cache_pts_quant_shuffle,
        qkv,
        *common,
        q_fly,
        k_fly,
        v_fly,
        slots,
        k_scale,
        v_scale,
        k_out_fly,
        v_out_fly,
        *suffix,
        num_iters=iters,
        num_warmup=warmup,
        use_cuda_event=True,
    )
    _, hip_us = run_perftest(
        aiter.fused_qk_norm_mrope_3d_cache_pts_quant_shuffle,
        qkv.view(num_tokens, -1),
        *common,
        q_hip.view(num_tokens, -1),
        k_hip,
        v_hip,
        slots,
        k_scale,
        v_scale,
        k_out_hip,
        v_out_hip,
        *suffix,
        num_iters=iters,
        num_warmup=warmup,
        use_cuda_event=True,
    )

    label = (
        f"T={num_tokens} Hq={num_q_heads} Hkv={num_kv_heads} D={head_size} "
        f"cache={cache_dtype} page={page_size} interleaved={interleaved} "
        f"slots={slot_pattern} strided_pos={strided_positions} "
        f"gemma={gemma_norm} return_kv={return_kv}"
    )
    print(f"[case] {label}")

    outputs = [
        ("q_out", q_fly, q_hip),
        ("k_cache", k_fly.view(-1), k_hip.view(-1)),
        ("v_cache", v_fly.view(-1), v_hip.view(-1)),
    ]
    if return_kv:
        outputs.extend(
            [
                ("k_out", k_out_fly, k_out_hip),
                ("v_out", v_out_fly, v_out_hip),
            ]
        )

    for name, fly_output, hip_output in outputs:
        checkAllclose(
            fly_output.float(),
            hip_output.float(),
            rtol=0.0,
            atol=0.0,
            msg=f"{name} vs reference",
        )

    print(
        f"  latency: flydsl={flydsl_us:.2f} us, "
        f"production={hip_us:.2f} us, "
        f"speedup={hip_us / flydsl_us:.2f}x"
    )


def _str_to_bool(value: str) -> bool:
    value = value.lower()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean: {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[63, 64, 256, 4096, 10885, 20317, 29136, 30584, 32768],
    )
    parser.add_argument("--num-q-heads", type=int, nargs="+", default=[64])
    parser.add_argument("--num-kv-heads", type=int, nargs="+", default=[4])
    parser.add_argument(
        "--head-sizes", type=int, nargs="+", choices=MROPE_SECTIONS, default=[128]
    )
    parser.add_argument(
        "--cache-dtypes",
        nargs="+",
        choices=["fp8", "bf16"],
        default=["fp8"],
    )
    parser.add_argument(
        "--page-sizes", type=int, nargs="+", choices=[16, 64], default=[64]
    )
    parser.add_argument("--interleaved", type=_str_to_bool, nargs="+", default=[True])
    parser.add_argument(
        "--slot-patterns",
        nargs="+",
        choices=["aligned", "random", "negative"],
        default=["aligned"],
    )
    parser.add_argument(
        "--strided-positions", type=_str_to_bool, nargs="+", default=[False]
    )
    parser.add_argument("--gemma-norm", type=_str_to_bool, nargs="+", default=[False])
    parser.add_argument("--return-kv", type=_str_to_bool, nargs="+", default=[False])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--iters", type=int, default=1)
    args = parser.parse_args()

    dtype_by_name = {"fp8": dtypes.fp8, "bf16": torch.bfloat16}
    sweep = itertools.product(
        args.tokens,
        args.num_q_heads,
        args.num_kv_heads,
        args.head_sizes,
        args.cache_dtypes,
        args.page_sizes,
        args.interleaved,
        args.slot_patterns,
        args.strided_positions,
        args.gemma_norm,
        args.return_kv,
    )
    case_count = 0
    for case_count, case in enumerate(sweep, start=1):
        (
            num_tokens,
            num_q_heads,
            num_kv_heads,
            head_size,
            cache_dtype_name,
            page_size,
            interleaved,
            slot_pattern,
            strided_positions,
            gemma_norm,
            return_kv,
        ) = case
        run_case(
            num_tokens=num_tokens,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            cache_dtype=dtype_by_name[cache_dtype_name],
            page_size=page_size,
            interleaved=interleaved,
            slot_pattern=slot_pattern,
            strided_positions=strided_positions,
            gemma_norm=gemma_norm,
            return_kv=return_kv,
            seed=args.seed,
            warmup=args.warmup,
            iters=args.iters,
        )


if __name__ == "__main__":
    main()
