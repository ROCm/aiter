# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

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
    make_route,
    maybe_int,
    route_stats,
)
from bench_sonic_moe_stage_breakdown import _make_moe, _metadata_dict  # noqa: E402

import aiter  # noqa: E402
from aiter.fused_moe import (  # noqa: E402
    get_2stage_cfgs,
    get_inter_dim,
    get_padded_M,
    is_flydsl_stage2_reduce,
    moe_sorting,
)
from aiter.jit.utils.chip_info import get_cu_num, get_gfx  # noqa: E402
from aiter.ops.enum import QuantType  # noqa: E402
from aiter.ops.flydsl.moe_common import GateMode  # noqa: E402
from aiter.sonic_moe import ActivationType, _to_aiter_activation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only one SonicMoE/AITER MoE stage kernel in a tight loop for "
            "rocprof/rocprofv3 counter collection."
        )
    )
    parser.add_argument("--shape", type=_parse_shape, required=True, help="T,H,I,E,K")
    parser.add_argument("--stage", choices=["stage1", "stage2"], required=True)
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
        default="balanced",
    )
    parser.add_argument("--rounding-tile", type=int, default=128)
    parser.add_argument("--block-size-m", type=str, default="128")
    parser.add_argument("--dispatch-policy", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device required")

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    token, hidden, intermediate, experts, topk = args.shape
    dtype = _dtype(args.dtype)
    block_size_m_override = maybe_int(args.block_size_m)
    if block_size_m_override is None:
        raise ValueError("profile driver expects an explicit --block-size-m")

    moe, x = _make_moe(
        token, hidden, intermediate, experts, topk, dtype, args.activation
    )
    route: Route = make_route(moe, x, args.routing, args.rounding_tile)
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
    if metadata.run_1stage:
        raise RuntimeError("selected metadata is a 1-stage kernel, not stage1/stage2")

    block_m = int(block_size_m_override)
    sorting_accumulate = not is_flydsl_stage2_reduce(metadata.stage2)
    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, moe_buf = moe_sorting(
        route.ids.to(torch.int32).contiguous(),
        route.weights.contiguous(),
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

    a2 = torch.empty((token, topk, inter_dim), dtype=dtype_out, device=x.device)
    out = (
        moe_buf
        if moe_buf.numel() != 0
        else torch.empty((token, model_dim), dtype=dtype_out, device=x.device)
    )

    def run_stage1() -> torch.Tensor:
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

    # Stage2 needs a valid A2 input. Compute it once before profiling stage2.
    if args.stage == "stage2":
        run_stage1()
        torch.cuda.synchronize()

    def run_stage2() -> torch.Tensor:
        metadata.stage2(
            a2,
            w1,
            w2,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            out,
            topk,
            w2_scale=None,
            a2_scale=None,
            block_m=block_m,
            sorted_weights=sorted_weights,
        )
        return out

    fn = run_stage1 if args.stage == "stage1" else run_stage2
    ms, y = bench_cuda(fn, args.warmup, args.iters)
    torch.cuda.synchronize()

    result = {
        "shape": [token, hidden, intermediate, experts, topk],
        "shape_str": f"{token},{hidden},{intermediate},{experts},{topk}",
        "stage": args.stage,
        "dtype": args.dtype,
        "activation": args.activation.value,
        "routing": args.routing,
        "block_m": block_m,
        "dispatch_policy": args.dispatch_policy,
        "warmup": args.warmup,
        "iters": args.iters,
        "stage_ms": ms,
        "output_checksum": float(y.float().sum().item()),
        "a2_mib": token
        * topk
        * inter_dim
        * torch.empty((), dtype=dtype_out).element_size()
        / (1024.0 * 1024.0),
        "gfx": get_gfx(),
        "cu_num": get_cu_num(),
        **route_stats(route.ids, experts, args.rounding_tile),
        **_metadata_dict(metadata, block_m),
    }
    print("profile_stage_json=" + json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
