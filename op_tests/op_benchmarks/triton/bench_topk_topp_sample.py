# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark: topk_topp_sample against the eager sampler it replaces
(softmax(logits / t) in fp32, top_k_top_p_sampling_from_probs, the argmax of
greedy rows). Times are CUDA-graph replays."""

import argparse

import torch
import triton

from aiter import logger
from aiter.ops import sampling
from aiter.ops.triton.fusions.topk_topp_sample import topk_topp_sample

VOCAB = 248320


def _eager(logits, t, k, p):
    probs = (logits / t.unsqueeze(-1)).softmax(dim=-1, dtype=torch.float32)
    tokens = sampling.top_k_top_p_sampling_from_probs(
        probs, None, None, k, None, p, deterministic=True
    )
    greedy = torch.argmax(probs, dim=-1).to(torch.int32)
    return torch.where(t <= 1e-10, greedy, tokens.view(-1))


def _graph_us(fn):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return triton.testing.do_bench(graph.replay) * 1000


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[3, 47, 160, 512])
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()
    sampling._SMALL_K_TOPP = True
    for rows in args.rows:
        logits = torch.randn((rows, VOCAB), dtype=torch.float32, device="cuda")
        logits = (logits * 2.5).to(torch.bfloat16)
        t = torch.ones((rows,), dtype=torch.float32, device="cuda")
        call = (logits, t, args.k, args.top_p)
        fused = _graph_us(lambda c=call: topk_topp_sample(*c))
        eager = _graph_us(lambda c=call: _eager(*c))
        logger.warning("rows=%d fused=%.1f us eager=%.1f us", rows, fused, eager)


if __name__ == "__main__":
    main()
