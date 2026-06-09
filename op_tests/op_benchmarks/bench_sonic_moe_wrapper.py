# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiter.sonic_moe import ActivationType, KernelBackendMoE, MoE


def _parse_shape(value: str) -> tuple[int, int, int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("expected T,H,I,E,K")
    return parts


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _activation(name: str) -> ActivationType:
    return ActivationType(name.lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the SonicMoE-compatible AITER wrapper."
    )
    parser.add_argument("--shape", type=_parse_shape, default=(512, 128, 64, 8, 2))
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument(
        "--activation",
        type=_activation,
        choices=list(ActivationType),
        default=ActivationType.SWIGLU,
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-torch", action="store_true")
    return parser.parse_args()


def bench(fn, warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    with torch.no_grad():
        for _ in range(warmup):
            y = fn()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            y = fn()
        end.record()
        torch.cuda.synchronize()

    return start.elapsed_time(end) / iters, y


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device required")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    token, hidden, intermediate, experts, topk = args.shape
    dtype = _dtype(args.dtype)

    print(f"torch={torch.__version__}")
    print(f"torch.version.hip={getattr(torch.version, 'hip', None)}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(
        f"shape=T{token},H{hidden},I{intermediate},E{experts},K{topk}, "
        f"dtype={dtype}, activation={args.activation.value}"
    )

    moe = MoE(
        num_experts=experts,
        num_experts_per_tok=topk,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation_function=args.activation,
        add_bias=False,
        std=0.02,
    ).to(device="cuda", dtype=dtype)
    moe.eval()
    x = 0.02 * torch.randn(token, hidden, device="cuda", dtype=dtype)

    def run_aiter() -> torch.Tensor:
        return moe(
            x,
            kernel_backend_moe=KernelBackendMoE.aiter,
            is_inference_mode=True,
        )[0]

    aiter_ms, aiter_out = bench(run_aiter, args.warmup, args.iters)
    print(f"aiter_ms={aiter_ms:.4f}")

    if args.skip_torch:
        return

    def run_torch() -> torch.Tensor:
        return moe(
            x,
            kernel_backend_moe=KernelBackendMoE.torch,
            is_inference_mode=True,
        )[0]

    torch_ms, torch_out = bench(run_torch, args.warmup, args.iters)
    diff = (aiter_out.float() - torch_out.float()).abs()
    print(f"torch_ms={torch_ms:.4f}")
    print(f"speedup={torch_ms / aiter_ms:.2f}x")
    print(f"max_abs={diff.max().item():.6e}")
    print(f"mean_abs={diff.mean().item():.6e}")


if __name__ == "__main__":
    main()
