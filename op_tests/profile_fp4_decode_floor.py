# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Measure graph-replayed empty dispatches, separately from scorer results."""

import statistics

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import rocdl
import torch

from op_tests.bench_fp4_decode_pipeline import capture, measure


def empty_launch(blocks, waves):
    @flyc.kernel
    def empty_kernel():
        rocdl.sched_barrier(0)

    @flyc.jit
    def launch(stream: fx.Stream):
        empty_kernel().launch(grid=(blocks,), block=(waves * 64,), stream=stream)

    return lambda: launch(torch.cuda.current_stream())


for blocks, waves in [(1, 1), (4, 4), (16, 2), (256, 4), (1024, 2), (2048, 2)]:
    function = empty_launch(blocks, waves)
    function()
    graph = capture(function, 128)
    samples = [measure(graph, 128) for _ in range(11)]
    print(f"empty blocks={blocks} waves={waves} median_us={statistics.median(samples):.6f} samples={samples}", flush=True)
