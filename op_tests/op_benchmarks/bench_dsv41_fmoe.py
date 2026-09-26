# SPDX-License-Identifier: MIT
"""Compare DSV4.1 Flash EP4 FMoE tuning against the shipped fallback on gfx950."""

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


def graph_latency_us(launch):
    for _ in range(3):
        launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    samples = []
    for _ in range(5):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(100):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 10)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument(
        "--tokens", type=int, nargs="+", default=[1, 32, 128, 512, 4096]
    )
    args = parser.parse_args()
    if get_gfx() != "gfx950":
        parser.error("this configuration targets gfx950")
    if os.getenv("AITER_BF16_FP8_MOE_BOUND") != "0":
        parser.error("set AITER_BF16_FP8_MOE_BOUND=0 to select a8w4 at small batches")
    fused_moe = importlib.import_module("aiter.fused_moe")
    torch.cuda.set_stream(torch.cuda.Stream())
    torch.manual_seed(42)
    experts, hidden, intermediate, topk = 96, 5120, 2304, 5

    def make_weight(rows, cols, gate_up):
        # Legal FP4 codes avoid allocating full-size floating-point expert weights.
        packed = torch.randint(
            0, 256, (experts, rows, cols // 2), device="cuda", dtype=torch.uint8
        ).view(dtypes.fp4x2)
        scales = torch.full(
            (experts * rows, cols // 32), 121, device="cuda", dtype=torch.uint8
        )
        return (
            shuffle_weight_a16w4(packed, 16, gate_up),
            shuffle_scale_a16w4(scales, experts, gate_up),
        )

    weight1, scale1 = make_weight(2 * intermediate, hidden, True)
    weight2, scale2 = make_weight(hidden, intermediate, False)
    config_dir = Path(AITER_ROOT_DIR) / "aiter/configs"
    baseline_files = [config_dir / "tuned_fmoe.csv"] + sorted(
        path
        for path in (config_dir / "model_configs").glob("*tuned_fmoe*.csv")
        if "untuned" not in path.name and path.resolve() != args.csv.resolve()
    )
    for tokens in args.tokens:
        activation = (
            torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) / 4
        )
        expert_ids = torch.stack(
            [torch.randperm(experts, device="cuda")[:topk] for _ in range(tokens)]
        ).to(torch.int32)
        routing_weights = torch.full((tokens, topk), 1 / topk, device="cuda")

        launch = partial(
            fused_moe.fused_moe,
            activation,
            weight1,
            weight2,
            routing_weights,
            expert_ids,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=scale1,
            w2_scale=scale2,
            gate_mode="interleave",
        )

        outputs, timings = {}, {}
        for mode, files in (
            ("default", baseline_files),
            ("tuned", [*baseline_files, args.csv.resolve()]),
        ):
            os.environ["AITER_CONFIG_FMOE"] = ":".join(map(str, files))
            AITER_CONFIGS.get_config_file.cache_clear()
            fused_moe.cfg_2stages = None
            fused_moe.get_2stage_cfgs.cache_clear()
            outputs[mode] = launch().clone()
            assert torch.isfinite(outputs[mode]).all()
            timings[mode] = graph_latency_us(launch)
        # This is a timing comparison, not an independent numerical oracle.
        max_delta = (outputs["tuned"].float() - outputs["default"].float()).abs().max()
        print(
            json.dumps(
                {
                    "tokens": tokens,
                    "default_us": timings["default"],
                    "tuned_us": timings["tuned"],
                    "speedup": timings["default"] / timings["tuned"],
                    "max_abs_delta": float(max_delta),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
