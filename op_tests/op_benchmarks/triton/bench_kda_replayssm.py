# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Measure the full ReplaySSM entry with standard and captured timers."""

import argparse
import statistics

import torch
from triton.testing import do_bench

from aiter import logger
from aiter.ops.triton.attention.kda_replayssm import replayssm_sigmoid_gating_delta_rule
from op_tests.triton_tests.attention.test_kda_replayssm import make_inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=128)
    parser.add_argument("--live", type=int, default=None)
    parser.add_argument("--pool", type=int, nargs="+", default=[1, 16])
    parser.add_argument(
        "--timer", choices=["both", "do_bench", "graph_event"], default="both"
    )
    args = parser.parse_args()
    live = args.requests if args.live is None else args.live
    if not 0 < live <= args.requests or any(n < 1 for n in args.pool):
        parser.error("live must be within requests and pools must be positive")
    inputs, _ = make_inputs(args.requests, "spec_verify", pad_rows=0)
    for key in ("q", "k", "v", "a", "b"):
        inputs[key] = inputs[key][:, : live * 5]
    inputs["slot_idx"][live:] = -1
    inputs["cu_seqlens"][live:] = live * 5
    for size in args.pool:
        states = [
            {
                **inputs,
                **{k: inputs[k].clone() for k in ("ckpt", "buf_k", "buf_u", "buf_g")},
            }
            for _ in range(size)
        ]
        outputs = [
            torch.zeros((live * 5, 12, 128), device="cuda", dtype=torch.bfloat16)
            for _ in states
        ]
        cursor = 0

        def run(states=states, outputs=outputs, size=size):
            nonlocal cursor
            i = cursor % size
            cursor += 1
            return replayssm_sigmoid_gating_delta_rule(
                **states[i],
                o=outputs[i],
                use_qk_l2norm_in_kernel=True,
                lower_bound=-5.0
            )

        for _ in range(16):
            run()
        torch.cuda.synchronize()
        if args.timer in ("both", "do_bench"):
            us = do_bench(run, warmup=25, rep=100, return_mode="median") * 1000
            logger.warning(
                "requests=%s live=%s pool=%s timer=do_bench latency_us=%.3f",
                args.requests,
                live,
                size,
                us,
            )
        if args.timer in ("both", "graph_event"):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(64):
                    run()
            for _ in range(3):
                graph.replay()
            samples = []
            for _ in range(9):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                    enable_timing=True
                )
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) * 1000 / 64)
            logger.warning(
                "requests=%s live=%s pool=%s timer=graph_event latency_us=%.3f",
                args.requests,
                live,
                size,
                statistics.median(samples),
            )


if __name__ == "__main__":
    main()
