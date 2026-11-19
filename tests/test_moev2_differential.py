# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 against the legacy MoE code.

The CPU test compares dispatch plans and log lines on every snapshot key a v2 path
claims. The GPU test runs tests/moe_dispatch/gpu_differential.py for each registered
path: same kernels on the same inputs, same outputs and log lines as legacy.
"""

import pathlib
import subprocess
import sys
from functools import partial

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def test_whole_graph_launches_are_compared_without_a_stage1_record():
    import torch
    from moe_dispatch.differential import compare_launches

    from aiter.fused_moe_registry import FusedMoeRequest

    def whole_graph(request):
        return request.hidden_states

    def request():
        return FusedMoeRequest(
            hidden_states=torch.ones(1, 2),
            w1=torch.ones(1, 2, 2),
            w2=torch.ones(1, 2, 2),
            topk_weight=torch.ones(1, 1),
            topk_ids=torch.zeros(1, 1, dtype=torch.int32),
        )

    legacy_request, v2_request = request(), request()
    call = {"topk_ids": legacy_request.topk_ids, "expert_mask": None}
    problems = compare_launches(
        [("whole_graph", partial(whole_graph, legacy_request))],
        [("whole_graph", partial(whole_graph, v2_request))],
        call,
        torch.empty(1, 2),
        torch.empty(1, 2),
    )
    assert not problems


def test_dispatch_plans_match_legacy():
    proc = subprocess.run(
        [sys.executable, str(_HERE / "moe_dispatch" / "differential.py"), "--check"],
        cwd=_HERE.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, (
        proc.stdout[-6000:]
        + proc.stderr[-3000:]
        + "\nv2 must make the same choice as legacy for every key it claims."
    )


def _registered():
    from aiter.moev2 import paths

    return sorted(paths.registered())


@pytest.mark.parametrize("path_name", _registered())
def test_outputs_match_legacy(path_name):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    proc = subprocess.run(
        [
            sys.executable,
            str(_HERE / "moe_dispatch" / "gpu_differential.py"),
            "--path",
            path_name,
        ],
        cwd=_HERE.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-3000:]
