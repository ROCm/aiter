# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark DSV4.1 Flash EP4 A8W4 FMoE with production-shaped routing."""

import argparse
import importlib
import json
import os
import statistics
from functools import partial
from pathlib import Path

import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.jit.core import AITER_CONFIGS, AITER_ROOT_DIR
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

GLOBAL_EXPERTS = 384
LOCAL_EXPERTS = 96
HIDDEN = 5120
INTERMEDIATE = 2304
TOPK = 6
ROUTED_SCALE = 1.5
SWIGLU_LIMIT = 10.0


def _config_files(candidate: Path | None) -> list[Path]:
    config_dir = Path(AITER_ROOT_DIR) / "aiter/configs"
    excluded = candidate.resolve() if candidate is not None else None
    files = [config_dir / "tuned_fmoe.csv"]
    files.extend(
        path
        for path in sorted((config_dir / "model_configs").glob("*tuned_fmoe*.csv"))
        if "untuned" not in path.name
        and (excluded is None or path.resolve() != excluded)
    )
    if candidate is not None:
        files.append(candidate.resolve())
    return files


def _install_config(fused_moe, files: list[Path]) -> None:
    os.environ["AITER_CONFIG_FMOE"] = os.pathsep.join(map(str, files))
    AITER_CONFIGS.get_config_file.cache_clear()
    fused_moe.cfg_2stages = None
    fused_moe.cfg_2stages_by_file.clear()
    fused_moe.get_2stage_cfgs.cache_clear()


def _routing(tokens: int, mode: str) -> tuple[torch.Tensor, torch.Tensor]:
    rows = torch.arange(tokens, device="cuda", dtype=torch.int64)[:, None]
    slots = torch.arange(TOPK, device="cuda", dtype=torch.int64)[None, :]
    if mode == "balanced":
        expert_ids = (rows * TOPK + slots) % GLOBAL_EXPERTS
    elif mode == "skewed":
        expert_ids = (rows + slots) % (LOCAL_EXPERTS // 4)
    else:
        expert_ids = (LOCAL_EXPERTS + rows * TOPK + slots) % GLOBAL_EXPERTS
    raw = (slots + 1).to(torch.float32).expand(tokens, -1)
    weights = raw / raw.sum(dim=1, keepdim=True) * ROUTED_SCALE
    return expert_ids.to(torch.int32), weights.contiguous()


def _expert_mask(rank: int) -> torch.Tensor:
    mask = torch.zeros(GLOBAL_EXPERTS, device="cuda", dtype=torch.int32)
    start = rank * LOCAL_EXPERTS
    mask[start : start + LOCAL_EXPERTS] = 1
    return mask


def _make_weights():
    def make(rows: int, cols: int, gate_up: bool):
        packed = torch.randint(
            0,
            256,
            (LOCAL_EXPERTS, rows, cols // 2),
            device="cuda",
            dtype=torch.uint8,
        ).view(dtypes.fp4x2)
        scales = torch.full(
            (LOCAL_EXPERTS * rows, cols // 32),
            121,
            device="cuda",
            dtype=torch.uint8,
        )
        return (
            shuffle_weight_a16w4(packed, 16, gate_up),
            shuffle_scale_a16w4(scales, LOCAL_EXPERTS, gate_up),
        )

    weight1, scale1 = make(2 * INTERMEDIATE, HIDDEN, True)
    weight2, scale2 = make(HIDDEN, INTERMEDIATE, False)
    return weight1, scale1, weight2, scale2


def _kernel_name(stage) -> str:
    keywords = getattr(stage, "keywords", {})
    return str(keywords.get("kernelName") or keywords.get("kernelName1") or "")


def _metadata(fused_moe, tokens: int):
    metadata = fused_moe.get_2stage_cfgs(
        fused_moe.get_padded_M(tokens),
        HIDDEN,
        INTERMEDIATE,
        LOCAL_EXPERTS,
        TOPK,
        torch.bfloat16,
        dtypes.fp8,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        "interleave",
        is_ep=True,
        ep_has_fake_expert=False,
    )
    return {
        "stage1": _kernel_name(metadata.stage1),
        "stage2": _kernel_name(metadata.stage2),
        "block_m": int(metadata.block_m),
        "fuse_quant": metadata.fuse_quant,
    }


def _event_latency_us(call, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        call()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / iterations


def _graph_call(call):
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph.replay, output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--baseline-csv", type=Path)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096],
    )
    parser.add_argument("--ep-ranks", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument(
        "--routing", choices=("balanced", "skewed", "remote"), default="balanced"
    )
    parser.add_argument("--execution", choices=("graph", "eager"), default="graph")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    if get_gfx() != "gfx950":
        parser.error("this benchmark requires gfx950")
    if os.getenv("AITER_BF16_FP8_MOE_BOUND") != "0":
        parser.error("set AITER_BF16_FP8_MOE_BOUND=0")
    if any(rank not in range(4) for rank in args.ep_ranks):
        parser.error("--ep-ranks must be in [0, 3]")

    fused_moe = importlib.import_module("aiter.fused_moe")
    torch.manual_seed(42)
    torch.cuda.set_stream(torch.cuda.Stream())
    weight1, scale1, weight2, scale2 = _make_weights()
    arms = {
        "baseline": _config_files(args.baseline_csv),
        "candidate": _config_files(args.csv),
    }

    for tokens in args.tokens:
        hidden = torch.randn(tokens, HIDDEN, device="cuda", dtype=torch.bfloat16) / 4
        expert_ids, routing_weights = _routing(tokens, args.routing)
        for rank in args.ep_ranks:
            mask = _expert_mask(rank)
            launchers = {}
            outputs = {}
            metadata = {}
            for arm, files in arms.items():
                _install_config(fused_moe, files)
                output = torch.empty_like(hidden)
                launch = partial(
                    fused_moe.fused_moe,
                    hidden,
                    weight1,
                    weight2,
                    routing_weights,
                    expert_ids,
                    expert_mask=mask,
                    activation=ActivationType.Silu,
                    quant_type=QuantType.per_1x32,
                    w1_scale=scale1,
                    w2_scale=scale2,
                    gate_mode="interleave",
                    swiglu_limit=SWIGLU_LIMIT,
                    output=output,
                    ep_has_fake_expert=False,
                )
                metadata[arm] = _metadata(fused_moe, tokens)
                if args.execution == "graph":
                    launchers[arm], output = _graph_call(launch)
                else:
                    for _ in range(3):
                        launch()
                    launchers[arm] = launch
                outputs[arm] = output.clone()

            baseline_f64 = outputs["baseline"].double()
            candidate_f64 = outputs["candidate"].double()
            delta = candidate_f64 - baseline_f64
            denominator = baseline_f64.norm().clamp_min(1e-12)
            samples = {arm: [] for arm in arms}
            for round_index in range(args.rounds):
                order = ("baseline", "candidate")
                if round_index % 2:
                    order = tuple(reversed(order))
                for arm in order:
                    _install_config(fused_moe, arms[arm])
                    samples[arm].append(
                        _event_latency_us(launchers[arm], args.iterations)
                    )
            medians = {
                arm: statistics.median(values) for arm, values in samples.items()
            }
            print(
                json.dumps(
                    {
                        "tokens": tokens,
                        "ep_rank": rank,
                        "routing": args.routing,
                        "execution": args.execution,
                        "metadata": metadata,
                        "latency_us": medians,
                        "speedup": medians["baseline"] / medians["candidate"],
                        "max_abs_delta": float(delta.abs().max()),
                        "relative_l2": float(delta.norm() / denominator),
                        "finite": {
                            "baseline": bool(torch.isfinite(baseline_f64).all()),
                            "candidate": bool(torch.isfinite(candidate_f64).all()),
                        },
                        "samples_us": samples,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
