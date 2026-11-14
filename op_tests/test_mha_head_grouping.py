# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CK-tile FA LLC head grouping: grouped vs DISABLE=1 is bitwise equal.

Grouping is forced, not observed. CK_TILE_FMHA_LLC_CACHE_MB (declared in
example/ck_tile/01_fmha/fmha_fwd_head_grouping.hpp) overrides the probed LLC
size, so pinning it to 1 MB makes get_head_group_size trip on any RDNA part
regardless of the real cache. Both subprocesses get the same override; the
only difference is CK_TILE_FMHA_DISABLE_HEAD_GROUPING. Without this the
comparison would be ungrouped-vs-ungrouped on machines with a large LLC,
i.e. silently vacuous.

Subprocesses are required because ck_tile caches CK_TILE_FMHA_* in
function-local statics and resolves the LLC size once into a static.
"""

import os
import subprocess
import sys

import pytest

from aiter.jit.utils.chip_info import get_gfx

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROBE = os.path.join(REPO_ROOT, "op_tests", "head_grouping_probe.py")

SUPPORTED_GFX11 = ["gfx1100", "gfx1101", "gfx1102", "gfx1103"]
SUPPORTED_GFX12 = ["gfx1200", "gfx1201"]
SUPPORTED_GFX = SUPPORTED_GFX11 + SUPPORTED_GFX12

# With LLC pinned to 1 MB the trip point is 1.5 MB of total KV and the group
# size is 1 MB / kv_bytes_per_head. Every shape below engages; see the plan
# for the arithmetic.
BASE = [
    "--batch",
    "1",
    "--nheads",
    "8",
    "--nheads-k",
    "8",
    "--seqlen",
    "2048",
    "--hdim",
    "128",
]
FP8 = ["--dtype", "fp8bf16"] + BASE
FP8_GQA = [
    "--dtype",
    "fp8bf16",
    "--batch",
    "1",
    "--nheads",
    "8",
    "--nheads-k",
    "2",
    "--seqlen",
    "2048",
    "--hdim",
    "128",
]
FP8_BATCH2 = [
    "--dtype",
    "fp8bf16",
    "--batch",
    "2",
    "--nheads",
    "8",
    "--nheads-k",
    "8",
    "--seqlen",
    "2048",
    "--hdim",
    "128",
]

# Pin the LLC in both runs so grouping is deterministic, not machine-dependent.
FORCE_LLC = {"CK_TILE_FMHA_LLC_CACHE_MB": "1"}

pytestmark = [
    pytest.mark.skipif(
        int(os.environ.get("ENABLE_CK", "1")) == 0, reason="ENABLE_CK=0"
    ),
]


def requires_gfx(*archs: str):
    return pytest.mark.skipif(
        get_gfx() not in archs,
        reason=f"CK fmha head grouping is {' / '.join(archs)} only",
    )


def run_probe(args, env=None, timeout=300):
    child = dict(os.environ)
    child["PYTHONPATH"] = REPO_ROOT + os.pathsep + child.get("PYTHONPATH", "")
    child.update(FORCE_LLC)
    if env:
        child.update(env)
    proc = subprocess.run(
        [sys.executable, PROBE] + list(args),
        env=child,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=REPO_ROOT,
        check=False,
    )
    assert proc.returncode == 0, (
        f"probe failed rc={proc.returncode}\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    return proc


@pytest.mark.parametrize(
    "name,args",
    [
        pytest.param("fp8bf16", FP8, marks=requires_gfx(*SUPPORTED_GFX12)),
        pytest.param(
            "fp8bf16_causal",
            FP8 + ["--causal"],
            marks=requires_gfx(*SUPPORTED_GFX12),
        ),
        pytest.param("fp8bf16_gqa", FP8_GQA, marks=requires_gfx(*SUPPORTED_GFX12)),
        pytest.param(
            "fp8bf16_batch2", FP8_BATCH2, marks=requires_gfx(*SUPPORTED_GFX12)
        ),
        pytest.param(
            "bf16", ["--dtype", "bf16"] + BASE, marks=requires_gfx(*SUPPORTED_GFX)
        ),
        pytest.param(
            "fp16", ["--dtype", "fp16"] + BASE, marks=requires_gfx(*SUPPORTED_GFX)
        ),
        pytest.param(
            "bf16_lse",
            ["--dtype", "bf16"] + BASE + ["--lse"],
            marks=requires_gfx(*SUPPORTED_GFX),
        ),
    ],
)
def test_head_grouping_is_bitwise_identical(tmp_path, name, args):
    on_pt = str(tmp_path / f"{name}_on.pt")
    off_pt = str(tmp_path / f"{name}_off.pt")
    run_probe(args + ["--save", on_pt])
    run_probe(
        args + ["--save", off_pt],
        env={"CK_TILE_FMHA_DISABLE_HEAD_GROUPING": "1"},
    )
    import torch

    loaded_on = torch.load(on_pt)
    loaded_off = torch.load(off_pt)
    pairs = (
        list(zip(loaded_on, loaded_off))
        if "--lse" in args
        else [(loaded_on, loaded_off)]
    )
    for a, b in pairs:
        assert a.shape == b.shape and a.dtype == b.dtype
        assert torch.equal(a, b), (
            f"{name}: not bitwise identical, "
            f"max_abs_diff={(a.float() - b.float()).abs().max().item()}"
        )
