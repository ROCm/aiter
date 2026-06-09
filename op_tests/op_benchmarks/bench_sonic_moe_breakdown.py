# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aiter.fused_moe import fused_moe
from aiter.ops.enum import QuantType
from aiter.sonic_moe import ActivationType, KernelBackendMoE, MoE, _to_aiter_activation


@dataclass
class Route:
    logits: torch.Tensor | None
    weights: torch.Tensor
    ids: torch.Tensor
    prepare_ms: float
    moved: int = 0


def _parse_shape(value: str) -> tuple[int, int, int, int, int]:
    parts = tuple(int(part.strip()) for part in value.split(","))
    if len(parts) != 5:
        raise argparse.ArgumentTypeError("expected T,H,I,E,K")
    return parts


def _dtype(name: str) -> torch.dtype:
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[name]


def _activation(name: str) -> ActivationType:
    return ActivationType(name.lower())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Break down SonicMoE wrapper overhead and direct AITER fused_moe time. "
            "The rounded and balanced routing modes are benchmark prototypes."
        )
    )
    parser.add_argument("--shape", type=_parse_shape, default=(512, 128, 64, 8, 2))
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--activation",
        type=_activation,
        choices=list(ActivationType),
        default=ActivationType.SWIGLU,
    )
    parser.add_argument(
        "--routing",
        choices=["topk", "rounded", "balanced"],
        default="topk",
        help=(
            "topk uses normal SonicMoE routing; rounded heuristically moves low-weight "
            "assignments to tile-rounded expert counts; balanced is a synthetic kernel "
            "upper-bound route with uniform weights."
        ),
    )
    parser.add_argument("--rounding-tile", type=int, default=128)
    parser.add_argument(
        "--block-size-m",
        type=str,
        default="auto",
        help="AITER fused_moe block_size_M override, or auto.",
    )
    parser.add_argument("--dispatch-policy", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--skip-wrapper",
        action="store_true",
        help="Skip wrapper end-to-end timing; useful for sweep runs.",
    )
    return parser.parse_args()


def bench_cuda(fn, warmup: int, iters: int) -> tuple[float, torch.Tensor]:
    with torch.no_grad():
        y = None
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

    assert y is not None
    return start.elapsed_time(end) / iters, y


def wall_ms(fn) -> tuple[float, object]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0, out


def compute_topk_route(moe: MoE, x: torch.Tensor) -> Route:
    logits = moe.router(x)
    topk_logits, ids = logits.topk(moe.top_k, dim=-1)
    weights = F.softmax(topk_logits.float(), dim=-1)
    return Route(logits=logits, weights=weights, ids=ids, prepare_ms=0.0)


def _rounded_targets(counts: torch.Tensor, tile: int) -> torch.Tensor:
    if tile <= 1:
        return counts.clone()

    counts_cpu = counts.to(device="cpu", dtype=torch.long)
    total = int(counts_cpu.sum().item())
    floors = (counts_cpu // tile) * tile
    targets = floors.clone()
    remaining = total - int(targets.sum().item())
    remainders = counts_cpu - floors

    order = sorted(
        range(counts_cpu.numel()),
        key=lambda idx: (int(remainders[idx]), int(counts_cpu[idx])),
        reverse=True,
    )
    pos = 0
    while remaining >= tile and order:
        targets[order[pos % len(order)]] += tile
        remaining -= tile
        pos += 1

    # Most benchmark shapes have total % tile == 0. Keep a conservative fallback
    # for small synthetic shapes so the assignment count remains exactly stable.
    pos = 0
    while remaining > 0 and order:
        targets[order[pos % len(order)]] += 1
        remaining -= 1
        pos += 1

    return targets


def _make_balanced_route(
    token: int,
    experts: int,
    topk: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    # For the SonicMoE paper shapes T*K is divisible by E, so this produces exact
    # per-expert tile multiples and isolates the fused_moe kernel from router skew.
    token_idx = torch.arange(token, device=device).view(-1, 1)
    slot_idx = torch.arange(topk, device=device).view(1, -1)
    ids = (token_idx * topk + slot_idx) % experts
    weights = torch.full((token, topk), 1.0 / topk, device=device, dtype=torch.float32)
    return weights, ids.to(torch.int64)


def _make_rounded_route(moe: MoE, x: torch.Tensor, tile: int) -> Route:
    route = compute_topk_route(moe, x)
    assert route.logits is not None

    ids_cpu = route.ids.detach().to("cpu", dtype=torch.long).contiguous()
    weights_cpu = route.weights.detach().to("cpu", dtype=torch.float32).contiguous()
    flat_ids = ids_cpu.view(-1)
    counts = torch.bincount(flat_ids, minlength=moe.num_experts)
    targets = _rounded_targets(counts, tile)
    surplus = (counts - targets).tolist()
    deficit = (targets - counts).tolist()

    candidates: list[tuple[float, int, int]] = []
    flat_weights = weights_cpu.view(-1)
    for flat_idx, expert in enumerate(flat_ids.tolist()):
        if surplus[expert] > 0:
            candidates.append((float(flat_weights[flat_idx]), flat_idx, expert))
    candidates.sort(key=lambda item: item[0])

    deficits: list[int] = []
    for expert, need in enumerate(deficit):
        if need > 0:
            deficits.extend([expert] * int(need))

    moved = 0
    cursor = 0
    topk = moe.top_k
    for dst_expert in deficits:
        while cursor < len(candidates):
            _, flat_idx, src_expert = candidates[cursor]
            cursor += 1
            if surplus[src_expert] <= 0:
                continue
            token_idx = flat_idx // topk
            if dst_expert in ids_cpu[token_idx].tolist():
                continue
            ids_cpu.view(-1)[flat_idx] = dst_expert
            surplus[src_expert] -= 1
            moved += 1
            break

    rounded_ids = ids_cpu.to(device=x.device, dtype=torch.long)
    rounded_logits = route.logits.gather(1, rounded_ids)
    rounded_weights = F.softmax(rounded_logits.float(), dim=-1)
    return Route(
        logits=route.logits,
        weights=rounded_weights,
        ids=rounded_ids,
        prepare_ms=0.0,
        moved=moved,
    )


def make_route(moe: MoE, x: torch.Tensor, routing: str, tile: int) -> Route:
    if routing == "topk":
        prepare_ms, route = wall_ms(lambda: compute_topk_route(moe, x))
        route.prepare_ms = prepare_ms
        return route
    if routing == "balanced":
        prepare_ms, data = wall_ms(
            lambda: _make_balanced_route(x.shape[0], moe.num_experts, moe.top_k, x.device)
        )
        weights, ids = data
        return Route(logits=None, weights=weights, ids=ids, prepare_ms=prepare_ms)
    if routing == "rounded":
        prepare_ms, route = wall_ms(lambda: _make_rounded_route(moe, x, tile))
        route.prepare_ms = prepare_ms
        return route
    raise ValueError(f"unknown routing mode {routing}")


def route_stats(ids: torch.Tensor, experts: int, tile: int) -> dict[str, float | int]:
    flat = ids.reshape(-1).to(torch.long)
    counts = torch.bincount(flat, minlength=experts).to(torch.float32)
    total = int(flat.numel())
    if tile > 0:
        padded = torch.ceil(counts / tile) * tile
        padded_total = int(padded.sum().item())
    else:
        padded_total = total
    return {
        "assignments": total,
        "count_min": int(counts.min().item()),
        "count_max": int(counts.max().item()),
        "count_mean": float(counts.mean().item()),
        "count_std": float(counts.std(unbiased=False).item()),
        "padded_assignments": padded_total,
        "padding_overhead": padded_total - total,
        "tile_efficiency": float(total / padded_total) if padded_total else 0.0,
    }


def expert_tflops(token: int, hidden: int, intermediate: int, topk: int, ms: float) -> float:
    if ms <= 0:
        return float("nan")
    flops = 6 * token * topk * hidden * intermediate
    return flops / (ms * 1.0e-3) / 1.0e12


def maybe_int(value: str) -> int | None:
    if value == "auto":
        return None
    return int(value)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    token, hidden, intermediate, experts, topk = args.shape
    dtype = _dtype(args.dtype)
    block_size_m = maybe_int(args.block_size_m)

    print(f"torch={torch.__version__}")
    print(f"torch.version.hip={getattr(torch.version, 'hip', None)}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(
        f"shape=T{token},H{hidden},I{intermediate},E{experts},K{topk}, "
        f"dtype={dtype}, activation={args.activation.value}, routing={args.routing}, "
        f"rounding_tile={args.rounding_tile}, block_size_m={args.block_size_m}, "
        f"dispatch_policy={args.dispatch_policy}"
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

    router_ms, _ = bench_cuda(lambda: moe.router(x), args.warmup, args.iters)
    logits = moe.router(x)
    topk_softmax_ms, _ = bench_cuda(
        lambda: F.softmax(logits.topk(topk, dim=-1)[0].float(), dim=-1),
        args.warmup,
        args.iters,
    )

    route = make_route(moe, x, args.routing, args.rounding_tile)
    stats = route_stats(route.ids, experts, args.rounding_tile)
    print(f"router_linear_ms={router_ms:.4f}")
    print(f"topk_softmax_ms={topk_softmax_ms:.4f}")
    print(f"route_prepare_wall_ms={route.prepare_ms:.4f}")
    print(f"route_moved={route.moved}")
    print(
        "route_stats="
        + ",".join(
            f"{key}:{value:.6g}" if isinstance(value, float) else f"{key}:{value}"
            for key, value in stats.items()
        )
    )

    w1 = moe._aiter_w1()
    w2 = moe.c_proj.weight.contiguous()
    aiter_activation = _to_aiter_activation(args.activation)

    def run_direct_fused_moe() -> torch.Tensor:
        return fused_moe(
            x,
            w1,
            w2,
            route.weights.contiguous(),
            route.ids.to(torch.int32).contiguous(),
            activation=aiter_activation,
            quant_type=QuantType.No,
            block_size_M=block_size_m,
            moe_sorting_dispatch_policy=args.dispatch_policy,
        )

    direct_ms, direct_out = bench_cuda(run_direct_fused_moe, args.warmup, args.iters)
    direct_tflops = expert_tflops(token, hidden, intermediate, topk, direct_ms)
    print(f"direct_fused_moe_ms={direct_ms:.4f}")
    print(f"direct_fused_moe_expert_tflops={direct_tflops:.2f}")

    wrapper_ms = None
    wrapper_tflops = None
    if not args.skip_wrapper and args.routing == "topk" and block_size_m is None:
        wrapper_ms, wrapper_out = bench_cuda(
            lambda: moe(
                x,
                kernel_backend_moe=KernelBackendMoE.aiter,
                is_inference_mode=True,
            )[0],
            args.warmup,
            args.iters,
        )
        wrapper_tflops = expert_tflops(token, hidden, intermediate, topk, wrapper_ms)
        diff = (direct_out.float() - wrapper_out.float()).abs()
        print(f"wrapper_e2e_ms={wrapper_ms:.4f}")
        print(f"wrapper_e2e_expert_tflops={wrapper_tflops:.2f}")
        print(f"wrapper_direct_max_abs={diff.max().item():.6e}")
        print(f"wrapper_direct_mean_abs={diff.mean().item():.6e}")

    result = {
        "shape": [token, hidden, intermediate, experts, topk],
        "dtype": args.dtype,
        "activation": args.activation.value,
        "routing": args.routing,
        "rounding_tile": args.rounding_tile,
        "block_size_m": args.block_size_m,
        "dispatch_policy": args.dispatch_policy,
        "router_linear_ms": router_ms,
        "topk_softmax_ms": topk_softmax_ms,
        "route_prepare_wall_ms": route.prepare_ms,
        "route_moved": route.moved,
        "direct_fused_moe_ms": direct_ms,
        "direct_fused_moe_expert_tflops": direct_tflops,
        "wrapper_e2e_ms": wrapper_ms,
        "wrapper_e2e_expert_tflops": wrapper_tflops,
        **stats,
    }
    print("result_json=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
