# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 against the legacy MoE code (steps doc 1.3).

The CPU test compares dispatch plans on every snapshot key a v2 path claims. The GPU test
compares outputs on each registered path's examples. Both are no-ops until the first
path registers.
"""

import pathlib
import subprocess
import sys

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


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


def _examples():
    from aiter.moev2 import paths

    for name, path in sorted(paths.registered().items()):
        for i, call in enumerate(path.examples() if path.examples else ()):
            yield pytest.param(name, call, id=f"{name}-{i}")


@pytest.mark.parametrize("path_name,call", list(_examples()))
def test_outputs_match_legacy(path_name, call):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    from moe_dispatch.differential import run_differential

    run_differential(path_name, call)
