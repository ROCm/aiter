# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Shared helpers for the gfx1250 gluon GEMM-A16W16 DSR1 QKV_A (N=2112, K=7168) experiments.

"old" is the config file before this tuning (configs/ here), "new" the file installed in the
checkout under AITER_ROOT. make_cold_copies and time_with_cuda_graph repeat the timing of the
gluon tuning harness (CUDA-graph replay over cold input copies, median of 25 replays).
"""

import copy
import gc
import json
import os

import torch

from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.utils.gemm_config_utils import STANDARD_M_BOUNDS

HERE = os.path.dirname(os.path.abspath(__file__))
AITER_ROOT = os.environ.get(
    "AITER_ROOT", os.path.abspath(os.path.join(HERE, "..", "..", ".."))
)
N, K = 2112, 7168
OLD = os.path.join(HERE, "configs", "GEMM-A16W16-N=2112-K=7168.before.json")
NEW = os.path.join(
    AITER_ROOT,
    "aiter/ops/triton/configs/gfx1250/gluon/gemm/gemm_a16w16",
    "GEMM-A16W16-N=2112-K=7168.json",
)


def load_tables():
    tables = {}
    for label, path in (("old", OLD), ("new", NEW)):
        with open(path) as f:
            tables[label] = json.load(f)
    return tables


def resolve(table, M):
    """The bucket get_gemm_config() walks to in a file without M_BOUNDS."""
    for bound in STANDARD_M_BOUNDS:
        key = f"M_LEQ_{bound}"
        if M <= bound and key in table:
            return key, dict(table[key])
    return "any", dict(table["any"])


def make_inputs(M):
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda")
    y = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    return x, w, y


def call(config, x, w, y):
    """DSR1's call on gfx1250 (aiter.tuned_gemm -> triton): default (gluon) backend, no bias."""
    return gemm_a16w16(x, w, bias=None, dtype=torch.bfloat16, y=y, config=config)


def make_cold_copies(inputs, cold_mb=1024):
    """copies[0] is the original input tuple; the rest are clones, within the byte budget."""
    per_copy = sum(a.numel() * a.element_size() for a in inputs)
    budget = min(cold_mb << 20, torch.cuda.mem_get_info()[0] // 4)
    n = max(1, min(64, budget // max(per_copy, 1)))
    return [tuple(inputs)] + [tuple(a.clone() for a in inputs) for _ in range(n - 1)]


def time_with_cuda_graph(config, copies, calls=24, replays=25):
    """Median/min/max microseconds per launch: one eager warm-up, then graph replay."""
    stream = torch.cuda.Stream()
    n_calls = max(calls, len(copies))
    with torch.cuda.stream(stream):
        call(copy.deepcopy(config), *copies[0])
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        gc.disable()
        try:
            with torch.cuda.graph(graph, stream=stream):
                for i in range(n_calls):
                    call(copy.deepcopy(config), *copies[i % len(copies)])
        finally:
            gc.enable()
    torch.cuda.synchronize()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(replays):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1e3 / n_calls)
    times.sort()
    return times[len(times) // 2], times[0], times[-1]
