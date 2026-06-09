# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from bench_sonic_moe_breakdown import (  # noqa: E402
    Route,
    _activation,
    _dtype,
    _parse_shape,
    bench_cuda,
    expert_tflops,
    make_route,
    maybe_int,
    route_stats,
)

import aiter  # noqa: E402
from aiter.fused_moe import (  # noqa: E402
    fused_moe,
    get_2stage_cfgs,
    get_inter_dim,
    get_padded_M,
    is_flydsl_stage2_reduce,
    moe_sorting,
)
from aiter.jit.utils.chip_info import get_cu_num, get_gfx  # noqa: E402
from aiter.ops.enum import QuantType  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
from aiter.sonic_moe import (  # noqa: E402
    ActivationType,
    MoE,
    _to_aiter_activation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SonicMoE-on-AITER and split fused_moe into routing, sorting, "
            "stage1, stage2, and direct fused_moe timings. This script intentionally "
            "uses the same metadata selection path as aiter.fused_moe."
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
        "--skip-direct",
        action="store_true",
        help="Skip direct fused_moe timing; useful while debugging stage kernels.",
    )
    parser.add_argument(
        "--skip-correctness",
        action="store_true",
        help="Skip one-shot direct-vs-stage correctness comparison.",
    )
    return parser.parse_args()


def _callable_name(fn: Callable[..., Any] | None) -> str:
    if fn is None:
        return ""
    func = getattr(fn, "func", fn)
    return getattr(func, "__name__", str(func))


def _callable_keyword(fn: Callable[..., Any] | None, key: str, default: Any = "") -> Any:
    if fn is None:
        return default
    return getattr(fn, "keywords", {}).get(key, default)


def _metadata_dict(metadata: Any, block_m: int) -> dict[str, Any]:
    return {
        "stage1_backend": _callable_name(metadata.stage1),
        "stage2_backend": _callable_name(metadata.stage2),
        "kernelName1": _callable_keyword(metadata.stage1, "kernelName", ""),
        "kernelName2": _callable_keyword(metadata.stage2, "kernelName", ""),
        "block_m": int(block_m),
        "metadata_block_m": int(metadata.block_m),
        "ksplit": int(metadata.ksplit),
        "run_1stage": bool(metadata.run_1stage),
        "has_bias": bool(metadata.has_bias),
        "stage2_has_bias": bool(metadata.stage2_has_bias),
        "flat": bool(metadata.flat),
        "fuse_quant": str(metadata.fuse_quant),
        "use_non_temporal_load": bool(metadata.use_non_temporal_load),
    }


def _bottleneck(parts: dict[str, float]) -> tuple[str, float]:
    active = {key: value for key, value in parts.items() if value is not None}
    total = sum(active.values())
    if total <= 0 or not active:
        return "", 0.0
    name, value = max(active.items(), key=lambda item: item[1])
    return name, value / total


def _make_moe(
    token: int,
    hidden: int,
    intermediate: int,
    experts: int,
    topk: int,
    dtype: torch.dtype,
    activation: ActivationType,
) -> tuple[MoE, torch.Tensor]:
    moe = MoE(
        num_experts=experts,
        num_experts_per_tok=topk,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation_function=activation,
        add_bias=False,
        std=0.02,
    ).to(device="cuda", dtype=dtype)
    moe.eval()
    x = 0.02 * torch.randn(token, hidden, device="cuda", dtype=dtype)
    # Materialize/caches the Sonic interleaved -> AITER gate/up layout once.
    with torch.no_grad():
        moe._aiter_w1()
    return moe, x


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    token, hidden, intermediate, experts, topk = args.shape
    dtype = _dtype(args.dtype)
    block_size_m_override = maybe_int(args.block_size_m)

    print(f"torch={torch.__version__}")
    print(f"torch.version.hip={getattr(torch.version, 'hip', None)}")
    print(f"device={torch.cuda.get_device_name(0)}")
    print(f"gfx={get_gfx()} cu_num={get_cu_num()}")
    print(
        f"shape=T{token},H{hidden},I{intermediate},E{experts},K{topk}, "
        f"dtype={dtype}, activation={args.activation.value}, routing={args.routing}, "
        f"rounding_tile={args.rounding_tile}, block_size_m={args.block_size_m}, "
        f"dispatch_policy={args.dispatch_policy}"
    )

    moe, x = _make_moe(
        token, hidden, intermediate, experts, topk, dtype, args.activation
    )
    route: Route = make_route(moe, x, args.routing, args.rounding_tile)
    stats = route_stats(route.ids, experts, args.rounding_tile)

    router_ms, _ = bench_cuda(lambda: moe.router(x), args.warmup, args.iters)
    logits = moe.router(x)
    topk_softmax_ms, _ = bench_cuda(
        lambda: F.softmax(logits.topk(topk, dim=-1)[0].float(), dim=-1),
        args.warmup,
        args.iters,
    )

    w1 = moe._aiter_w1()
    w2 = moe.c_proj.weight.contiguous()
    aiter_activation = _to_aiter_activation(args.activation)
    E, model_dim, inter_dim = get_inter_dim(tuple(w1.shape), tuple(w2.shape))
    is_g1u1 = inter_dim != w1.shape[1]
    q_type = QuantType.No
    q_dtype_a = w1.dtype
    q_dtype_w = w1.dtype if w1.dtype != torch.uint32 else aiter.dtypes.fp8
    dtype_out = dtype

    metadata = get_2stage_cfgs(
        get_padded_M(token),
        model_dim,
        inter_dim,
        E,
        topk,
        dtype_out,
        q_dtype_a,
        q_dtype_w,
        q_type,
        is_g1u1,
        aiter_activation,
        False,
        0,
        0,
        getattr(w1, "is_shuffled", False) or getattr(w2, "is_shuffled", False),
        GateMode.SEPARATED.value,
        is_ep=False,
    )
    block_m = int(metadata.block_m if block_size_m_override is None else block_size_m_override)
    meta = _metadata_dict(metadata, block_m)

    topk_ids_i32 = route.ids.to(torch.int32).contiguous()
    topk_weights = route.weights.contiguous()
    sorting_accumulate = not is_flydsl_stage2_reduce(metadata.stage2)

    def sort_once():
        return moe_sorting(
            topk_ids_i32,
            topk_weights,
            experts,
            model_dim,
            dtype_out,
            block_m,
            expert_mask=None,
            num_local_tokens=None,
            dispatch_policy=args.dispatch_policy,
            accumulate=sorting_accumulate,
            flat=metadata.flat,
        )

    sorting_ms, sorting_ret = bench_cuda(sort_once, args.warmup, args.iters)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = sorting_ret

    def run_direct_fused_moe() -> torch.Tensor:
        return fused_moe(
            x,
            w1,
            w2,
            topk_weights,
            topk_ids_i32,
            activation=aiter_activation,
            quant_type=q_type,
            block_size_M=block_size_m_override,
            moe_sorting_dispatch_policy=args.dispatch_policy,
        )

    direct_ms = None
    direct_out = None
    if not args.skip_direct:
        direct_ms, direct_out = bench_cuda(run_direct_fused_moe, args.warmup, args.iters)

    stage1_ms = None
    stage2_ms = None
    one_stage_ms = None

    def run_one_stage() -> torch.Tensor:
        s_ids, s_weights, s_expert_ids, n_valid_ids, out_buf = sort_once()
        return metadata.stage1(
            x,
            w1,
            w2,
            topk,
            s_ids,
            s_weights,
            s_expert_ids,
            n_valid_ids,
            out_buf,
            is_g1u1,
            block_m,
            q_dtype_a=q_dtype_a,
            q_dtype_w=q_dtype_w,
            w1_scale=None,
            w2_scale=None,
            a1_scale=None,
            a2_scale=None,
            num_local_tokens=None,
            M=token,
            device=x.device,
            doweight_stage1=False,
        )

    if metadata.run_1stage:
        one_stage_ms, _ = bench_cuda(run_one_stage, args.warmup, args.iters)
    else:

        def run_stage1() -> torch.Tensor:
            a2 = torch.empty((token, topk, inter_dim), dtype=dtype_out, device=x.device)
            return metadata.stage1(
                x,
                w1,
                w2,
                sorted_ids,
                sorted_expert_ids,
                num_valid_ids,
                a2,
                topk,
                block_m=block_m,
                a1_scale=None,
                w1_scale=None,
                sorted_weights=None,
            )

        stage1_ms, a2_for_stage2 = bench_cuda(run_stage1, args.warmup, args.iters)

        # Pure stage2 kernel timing. Output is intentionally reused; the final
        # correctness path below uses a fresh sorting output.
        stage2_out = (
            moe_buf
            if moe_buf.numel() != 0
            else torch.empty((token, model_dim), dtype=dtype_out, device=x.device)
        )

        def run_stage2() -> torch.Tensor:
            metadata.stage2(
                a2_for_stage2,
                w1,
                w2,
                sorted_ids,
                sorted_expert_ids,
                num_valid_ids,
                stage2_out,
                topk,
                w2_scale=None,
                a2_scale=None,
                block_m=block_m,
                sorted_weights=sorted_weights,
            )
            return stage2_out

        stage2_ms, _ = bench_cuda(run_stage2, args.warmup, args.iters)

    stage_out = None
    max_abs = None
    mean_abs = None
    if not args.skip_correctness and direct_out is not None:
        if metadata.run_1stage:
            stage_out = run_one_stage()
        else:
            s_ids, s_weights, s_expert_ids, n_valid_ids, out_buf = sort_once()
            a2 = torch.empty((token, topk, inter_dim), dtype=dtype_out, device=x.device)
            a2 = metadata.stage1(
                x,
                w1,
                w2,
                s_ids,
                s_expert_ids,
                n_valid_ids,
                a2,
                topk,
                block_m=block_m,
                a1_scale=None,
                w1_scale=None,
                sorted_weights=None,
            )
            if out_buf.numel() == 0:
                out_buf = torch.empty((token, model_dim), dtype=dtype_out, device=x.device)
            metadata.stage2(
                a2,
                w1,
                w2,
                s_ids,
                s_expert_ids,
                n_valid_ids,
                out_buf,
                topk,
                w2_scale=None,
                a2_scale=None,
                block_m=block_m,
                sorted_weights=s_weights,
            )
            stage_out = out_buf
        torch.cuda.synchronize()
        diff = (direct_out.float() - stage_out.float()).abs()
        max_abs = float(diff.max().item())
        mean_abs = float(diff.mean().item())

    parts = (
        {"moe_sorting_ms": sorting_ms, "one_stage_ms": one_stage_ms}
        if metadata.run_1stage
        else {
            "moe_sorting_ms": sorting_ms,
            "stage1_ms": stage1_ms,
            "stage2_ms": stage2_ms,
        }
    )
    bottleneck_name, bottleneck_share = _bottleneck(
        {key: value for key, value in parts.items() if value is not None}
    )
    stage_sum_ms = sum(value for value in parts.values() if value is not None)
    direct_tflops = (
        None
        if direct_ms is None
        else expert_tflops(token, hidden, intermediate, topk, direct_ms)
    )
    stage_sum_tflops = expert_tflops(token, hidden, intermediate, topk, stage_sum_ms)
    a2_bytes = token * topk * inter_dim * torch.empty((), dtype=dtype_out).element_size()
    a2_read_write_mib = 2.0 * a2_bytes / (1024.0 * 1024.0)
    stage_over_direct = None if direct_ms is None else stage_sum_ms / direct_ms
    sorting_plus_stage_gap_ms = (
        None if direct_ms is None else stage_sum_ms - direct_ms
    )

    print(f"router_linear_ms={router_ms:.4f}")
    print(f"topk_softmax_ms={topk_softmax_ms:.4f}")
    print(f"route_prepare_wall_ms={route.prepare_ms:.4f}")
    print(f"route_moved={route.moved}")
    print(f"moe_sorting_ms={sorting_ms:.4f}")
    if one_stage_ms is not None:
        print(f"one_stage_ms={one_stage_ms:.4f}")
    if stage1_ms is not None:
        print(f"stage1_ms={stage1_ms:.4f}")
    if stage2_ms is not None:
        print(f"stage2_ms={stage2_ms:.4f}")
    if direct_ms is not None:
        print(f"direct_fused_moe_ms={direct_ms:.4f}")
        print(f"direct_fused_moe_expert_tflops={direct_tflops:.2f}")
    print(f"stage_sum_ms={stage_sum_ms:.4f}")
    print(f"stage_sum_expert_tflops={stage_sum_tflops:.2f}")
    if stage_over_direct is not None:
        print(f"stage_over_direct={stage_over_direct:.4f}")
        print(f"sorting_plus_stage_gap_ms={sorting_plus_stage_gap_ms:.4f}")
    print(f"bottleneck={bottleneck_name}")
    print(f"bottleneck_share={bottleneck_share:.4f}")
    print(
        "route_stats="
        + ",".join(
            f"{key}:{value:.6g}" if isinstance(value, float) else f"{key}:{value}"
            for key, value in stats.items()
        )
    )
    print(
        "metadata="
        + ",".join(f"{key}:{value}" for key, value in sorted(meta.items()))
    )
    if max_abs is not None:
        print(f"direct_stage_max_abs={max_abs:.6e}")
        print(f"direct_stage_mean_abs={mean_abs:.6e}")

    result = {
        "shape": [token, hidden, intermediate, experts, topk],
        "shape_str": f"{token},{hidden},{intermediate},{experts},{topk}",
        "padded_token": get_padded_M(token),
        "dtype": args.dtype,
        "torch_dtype": str(dtype_out),
        "activation": args.activation.value,
        "aiter_activation": str(aiter_activation),
        "routing": args.routing,
        "rounding_tile": args.rounding_tile,
        "dispatch_policy": args.dispatch_policy,
        "cu_num": get_cu_num(),
        "gfx": get_gfx(),
        "q_dtype_a": str(q_dtype_a),
        "q_dtype_w": str(q_dtype_w),
        "q_type": str(q_type),
        "use_g1u1": int(is_g1u1),
        "doweight_stage1": 0,
        "router_linear_ms": router_ms,
        "topk_softmax_ms": topk_softmax_ms,
        "route_prepare_wall_ms": route.prepare_ms,
        "route_moved": route.moved,
        "moe_sorting_ms": sorting_ms,
        "stage1_ms": stage1_ms,
        "stage2_ms": stage2_ms,
        "one_stage_ms": one_stage_ms,
        "stage_sum_ms": stage_sum_ms,
        "stage_sum_expert_tflops": stage_sum_tflops,
        "direct_fused_moe_ms": direct_ms,
        "direct_fused_moe_expert_tflops": direct_tflops,
        "stage_over_direct": stage_over_direct,
        "sorting_plus_stage_gap_ms": sorting_plus_stage_gap_ms,
        "bottleneck": bottleneck_name,
        "bottleneck_share": bottleneck_share,
        "a2_bytes": a2_bytes,
        "a2_mib": a2_bytes / (1024.0 * 1024.0),
        "a2_read_write_mib": a2_read_write_mib,
        "direct_stage_max_abs": max_abs,
        "direct_stage_mean_abs": mean_abs,
        **stats,
        **meta,
    }
    print("result_json=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
