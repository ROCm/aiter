#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and latency sweep for the FlyDSL fused QK norm/MRoPE/cache op."""

import argparse
import itertools
import sys
from pathlib import Path

import torch

import aiter
from aiter import per_tensor_quant
from aiter.ops.flydsl import (
    flydsl_fused_qk_norm_mrope_3d_cache_pts_quant_shuffle,
)
from aiter.test_common import checkAllclose
from aiter.utility import dtypes

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_fused_qk_norm_mrope_cache_quant import (  # noqa: E402
    apply_interleaved_rope,
    apply_rotary_emb_torch,
    gemma_rms_norm_forward,
    rms_norm_forward,
)


EPS = 1e-6
MAX_POSITIONS = 4096
MROPE_SECTIONS = {
    64: [12, 10, 10],
    128: [24, 20, 20],
    256: [48, 40, 40],
}


def _k_head_stride(head_size: int, page_size: int) -> int:
    return head_size * page_size


def _k_per_block(head_size: int, page_size: int, num_kv_heads: int) -> int:
    return num_kv_heads * _k_head_stride(head_size, page_size)


def _v_head_stride(head_size: int, page_size: int, x: int) -> int:
    return (page_size // x) * head_size * x


def _v_per_block(
    head_size: int, page_size: int, x: int, num_kv_heads: int
) -> int:
    return num_kv_heads * _v_head_stride(head_size, page_size, x)


def _scatter_shuffle_cache(
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slots: torch.Tensor,
    page_size: int,
    x: int,
) -> None:
    valid = slots >= 0
    if not valid.any():
        return

    slots = slots[valid].to(torch.int64)
    k = k[valid]
    v = v[valid]
    num_kv_heads = k.shape[1]
    head_size = k.shape[2]
    block_id = slots // page_size
    block_offset = slots % page_size
    d = torch.arange(head_size, device=slots.device)

    k_flat = k_cache.view(-1)
    k_chunk = d // x
    k_in_x = d % x
    for head in range(num_kv_heads):
        base = (
            block_id * _k_per_block(head_size, page_size, num_kv_heads)
            + head * _k_head_stride(head_size, page_size)
            + block_offset * x
        )
        dst = (
            base[:, None]
            + k_chunk[None, :] * (page_size * x)
            + k_in_x[None, :]
        )
        k_flat[dst.reshape(-1)] = k[:, head, :].reshape(-1)

    v_flat = v_cache.view(-1)
    v_chunk = block_offset // x
    v_in_x = block_offset % x
    for head in range(num_kv_heads):
        base = (
            block_id * _v_per_block(head_size, page_size, x, num_kv_heads)
            + head * _v_head_stride(head_size, page_size, x)
            + v_chunk * (head_size * x)
            + v_in_x
        )
        dst = base[:, None] + d[None, :] * x
        v_flat[dst.reshape(-1)] = v[:, head, :].reshape(-1)


def torch_ref(
    qkv: torch.Tensor,
    qw: torch.Tensor,
    kw: torch.Tensor,
    cos_sin: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    initial_k_cache: torch.Tensor,
    initial_v_cache: torch.Tensor,
    initial_k_out: torch.Tensor | None,
    initial_v_out: torch.Tensor | None,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    page_size: int,
    x: int,
    interleaved: bool,
    gemma_norm: bool,
    return_kv: bool,
):
    num_tokens = qkv.shape[0]
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    q, k, v = qkv.view(num_tokens, -1).split(
        [q_size, kv_size, kv_size], dim=-1
    )

    norm = gemma_rms_norm_forward if gemma_norm else rms_norm_forward
    q = norm(q.view(num_tokens, num_q_heads, head_size), qw, EPS)
    k = norm(k.view(num_tokens, num_kv_heads, head_size), kw, EPS)
    v = v.view(num_tokens, num_kv_heads, head_size)

    cos_sin = cos_sin.view(-1, head_size)
    indexed = cos_sin[positions.view(3, num_tokens)]
    cos, sin = indexed.chunk(2, dim=-1)
    sections = MROPE_SECTIONS[head_size]
    if interleaved:
        cos = apply_interleaved_rope(cos, sections)
        sin = apply_interleaved_rope(sin, sections)
    else:
        cos = torch.cat(
            [part[axis] for axis, part in enumerate(cos.split(sections, dim=-1))],
            dim=-1,
        )
        sin = torch.cat(
            [part[axis] for axis, part in enumerate(sin.split(sections, dim=-1))],
            dim=-1,
        )

    q = apply_rotary_emb_torch(q, cos, sin, is_neox_style=True)
    k = apply_rotary_emb_torch(k, cos, sin, is_neox_style=True)
    cache_dtype = initial_k_cache.dtype
    if cache_dtype == qkv.dtype:
        k_quant = k.to(cache_dtype)
        v_quant = v.to(cache_dtype)
    else:
        k_quant, _ = per_tensor_quant(k, scale=k_scale, quant_dtype=cache_dtype)
        v_quant, _ = per_tensor_quant(v, scale=v_scale, quant_dtype=cache_dtype)

    k_cache = initial_k_cache.clone()
    v_cache = initial_v_cache.clone()
    _scatter_shuffle_cache(
        k_quant, v_quant, k_cache, v_cache, slots, page_size, x
    )

    k_out = initial_k_out.clone() if initial_k_out is not None else None
    v_out = initial_v_out.clone() if initial_v_out is not None else None
    if return_kv:
        valid = slots >= 0
        k_out[valid] = k_quant[valid]
        v_out[valid] = v_quant[valid]

    return q, k_cache, v_cache, k_out, v_out


def _time(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


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
        torch.randn(
            MAX_POSITIONS, head_size, dtype=torch.bfloat16, device=device
        )
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
    initial_k_cache = torch.randn(
        cache_shape, dtype=torch.bfloat16, device=device
    ).to(cache_dtype)
    initial_v_cache = torch.randn(
        cache_shape, dtype=torch.bfloat16, device=device
    ).to(cache_dtype)
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

    q_prod = torch.empty_like(q_fly)
    k_prod = initial_k_cache.clone()
    v_prod = initial_v_cache.clone()
    k_out_prod = initial_k_out.clone() if return_kv else None
    v_out_prod = initial_v_out.clone() if return_kv else None

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

    def run_flydsl():
        flydsl_fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
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
        )

    def run_production():
        aiter.fused_qk_norm_mrope_3d_cache_pts_quant_shuffle(
            qkv.view(num_tokens, -1),
            *common,
            q_prod.view(num_tokens, -1),
            k_prod,
            v_prod,
            slots,
            k_scale,
            v_scale,
            k_out_prod,
            v_out_prod,
            *suffix,
        )

    ref_args = (
        qkv,
        qw,
        kw,
        cos_sin,
        positions,
        slots,
        k_scale,
        v_scale,
        initial_k_cache,
        initial_v_cache,
        initial_k_out,
        initial_v_out,
    )
    ref_kwargs = {
        "num_q_heads": num_q_heads,
        "num_kv_heads": num_kv_heads,
        "head_size": head_size,
        "page_size": page_size,
        "x": x,
        "interleaved": interleaved,
        "gemma_norm": gemma_norm,
        "return_kv": return_kv,
    }

    run_flydsl()
    run_production()
    reference = torch_ref(*ref_args, **ref_kwargs)
    torch.cuda.synchronize()

    label = (
        f"T={num_tokens} Hq={num_q_heads} Hkv={num_kv_heads} D={head_size} "
        f"cache={cache_dtype} page={page_size} interleaved={interleaved} "
        f"slots={slot_pattern} strided_pos={strided_positions} "
        f"gemma={gemma_norm} return_kv={return_kv}"
    )
    print(f"[case] {label}")

    outputs = [
        ("q_out", q_fly, q_prod, reference[0]),
        ("k_cache", k_fly.view(-1), k_prod.view(-1), reference[1].view(-1)),
        ("v_cache", v_fly.view(-1), v_prod.view(-1), reference[2].view(-1)),
    ]
    if return_kv:
        outputs.extend(
            [
                ("k_out", k_out_fly, k_out_prod, reference[3]),
                ("v_out", v_out_fly, v_out_prod, reference[4]),
            ]
        )

    for name, fly_output, prod_output, torch_output in outputs:
        checkAllclose(
            fly_output.float(),
            torch_output.float(),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} vs torch_ref",
        )
        checkAllclose(
            fly_output.float(),
            prod_output.float(),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name} vs production",
        )

    flydsl_us = _time(run_flydsl, iters, warmup)
    production_us = _time(run_production, iters, warmup)
    torch_ref_us = _time(
        lambda: torch_ref(*ref_args, **ref_kwargs), 1, 0
    )
    print(
        f"  latency: flydsl={flydsl_us:.2f} us, "
        f"production={production_us:.2f} us, "
        f"torch_ref={torch_ref_us:.2f} us, "
        f"speedup={production_us / flydsl_us:.2f}x"
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
        default=[256, 4096, 32768, 30584, 29136, 20317, 10885, 64, 63],
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
    parser.add_argument(
        "--interleaved", type=_str_to_bool, nargs="+", default=[True]
    )
    parser.add_argument(
        "--slot-patterns",
        nargs="+",
        choices=["aligned", "random", "negative"],
        default=["aligned"],
    )
    parser.add_argument(
        "--strided-positions", type=_str_to_bool, nargs="+", default=[False]
    )
    parser.add_argument(
        "--gemma-norm", type=_str_to_bool, nargs="+", default=[False]
    )
    parser.add_argument(
        "--return-kv", type=_str_to_bool, nargs="+", default=[False]
    )
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
