# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression test for the ASM split-K semaphore under graph replay."""

import os
import subprocess
import sys

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

_CHILD_ENV = "AITER_ASM_SPLITK_GRAPH_CHILD"
_KERNEL = "_ZN5aiter39bf16gemm_fp32bf16_tn_64x64_splitk_cleanE"


def _run_child() -> None:
    from aiter.ops.gemm_op_a16w16 import (
        gemm_a16w16_asm,
        get_semaphore_workspace,
    )

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    x = torch.randn(64, 5120, dtype=torch.bfloat16, device=device)
    weight = torch.randn(256, 5120, dtype=torch.bfloat16, device=device)

    eager = torch.empty(64, 256, dtype=torch.float32, device=device)
    gemm_a16w16_asm(x, weight, eager, splitK=13, kernelName=_KERNEL)
    torch.cuda.synchronize()

    capture_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(capture_stream):
        stale = get_semaphore_workspace(device)
        stale.fill_(2)
    capture_stream.synchronize()

    first = torch.empty_like(eager)
    second = torch.empty_like(eager)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        gemm_a16w16_asm(x, weight, first, splitK=13, kernelName=_KERNEL)
        gemm_a16w16_asm(x, weight, second, splitK=13, kernelName=_KERNEL)

    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(first, eager, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(second, eager, rtol=1e-2, atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a ROCm GPU")
def test_asm_splitk_graph_replay_uses_fresh_semaphores() -> None:
    if get_gfx_runtime() != "gfx950":
        pytest.skip("the production regression was observed on gfx950")

    env = os.environ.copy()
    env[_CHILD_ENV] = "1"
    proc = subprocess.run(
        [sys.executable, __file__],
        capture_output=True,
        env=env,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, (
        "captured ASM split-K GEMM hung or returned an incorrect result "
        f"(exit {proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )


if __name__ == "__main__" and os.environ.get(_CHILD_ENV) == "1":
    _run_child()
