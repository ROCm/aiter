# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Triton 384-expert sqrtsoftplus top-k against the HIP topk_gating kernel."""

import os
import subprocess
import sys
import textwrap

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import topk as topk_ops
from aiter.ops.triton.moe.moe_routing.topk_gating import (
    SQRTSOFTPLUS_384_ARCHS,
    SQRTSOFTPLUS_384_EXPERTS,
)


def _arch() -> str | None:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return None
    return get_gfx_runtime()


pytestmark = pytest.mark.skipif(
    _arch() not in SQRTSOFTPLUS_384_ARCHS,
    reason="requires an arch in SQRTSOFTPLUS_384_ARCHS",
)

EXPERTS = SQRTSOFTPLUS_384_EXPERTS
SCALE = 1.5


def _max_rows() -> int:
    return SQRTSOFTPLUS_384_ARCHS[_arch()].max_rows


def _hip(logits, bias, topk, num_valid_rows):
    rows = logits.shape[0]
    weights = torch.empty((rows, topk), dtype=torch.float32, device="cuda")
    ids = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    if bias is None:
        bias = torch.empty(0, dtype=torch.float32, device="cuda")
    topk_ops.topk_gating_fwd(
        weights, ids, logits, bias, True, SCALE, "sqrtsoftplus", 0, num_valid_rows
    )
    return weights, ids


def _triton(logits, bias, topk, num_valid_rows):
    """topk_gating, which must take the Triton kernel; fails if it runs HIP."""
    rows = logits.shape[0]
    weights = torch.empty((rows, topk), dtype=torch.float32, device="cuda")
    ids = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(topk_ops, "topk_gating_fwd", _no_hip)
        topk_ops.topk_gating(
            weights, ids, logits, bias, True, SCALE, num_valid_rows=num_valid_rows
        )
    return weights, ids


def _logits(rows, seed, case):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = (
        torch.randn((rows, EXPERTS), generator=gen, dtype=torch.float32, device="cuda")
        * 4
    )
    if case == "ties":
        # Few distinct values: many exact ties in a lane and across lanes.
        logits = (logits * 2).round() / 2
        logits[0, ::7] = float("nan")
    elif case == "nan":
        logits[:, ::3] = float("nan")
        logits[0, 5] = float("inf")
        logits[0, 6] = float("-inf")
        logits[-1] = float("nan")
    elif case == "constant":
        logits.fill_(0.25)
    return logits.to(torch.bfloat16)


def _bias(kind):
    if kind == "none":
        return None
    if kind == "linear":
        return torch.linspace(-1, 1, EXPERTS, dtype=torch.float32, device="cuda")
    return torch.randn(EXPERTS, dtype=torch.float32, device="cuda") * 0.1


def _assert_bitwise(got, ref, valid):
    weights, ids = got
    ref_weights, ref_ids = ref
    assert torch.equal(ids, ref_ids)
    # Compare the weights as bit patterns: renorm sum order must match too.
    assert torch.equal(weights.view(torch.int32), ref_weights.view(torch.int32))
    assert (ids[valid:] == -1).all()
    assert (weights[valid:] == 0).all()


@pytest.mark.parametrize("rows", [1, 7, 96, 576, -1])
@pytest.mark.parametrize("topk", [1, 6, 8])
@pytest.mark.parametrize("case", ["random", "ties", "nan", "constant"])
@pytest.mark.parametrize("bias_kind", ["none", "linear", "random"])
def test_sqrtsoftplus_384_topk_matches_hip(rows, topk, case, bias_kind):
    rows = _max_rows() if rows == -1 else rows
    torch.manual_seed(rows + topk)
    logits = _logits(rows, 17 * rows + topk, case)
    bias = _bias(bias_kind)
    valid = max(1, rows - 3)
    num_valid_rows = torch.tensor([valid], dtype=torch.int32, device="cuda")
    _assert_bitwise(
        _triton(logits, bias, topk, num_valid_rows),
        _hip(logits, bias, topk, num_valid_rows),
        valid,
    )


def test_sqrtsoftplus_384_topk_strided_logits():
    rows, topk = 64, 6
    backing = torch.randn((rows, EXPERTS + 64), dtype=torch.bfloat16, device="cuda")
    logits = backing[:, 32 : 32 + EXPERTS]
    bias = _bias("random")
    _assert_bitwise(
        _triton(logits, bias, topk, None), _hip(logits, bias, topk, None), rows
    )


def test_sqrtsoftplus_384_topk_strided_outputs():
    """Outputs are column slices of wider buffers, with a different row stride
    for weights and ids (as for the routed columns of a fused-shared buffer)."""
    rows, topk = 64, 6
    logits = _logits(rows, 11, "random")
    bias = _bias("random")
    weight_buf = torch.full((rows, topk + 2), -7.0, dtype=torch.float32, device="cuda")
    id_buf = torch.full((rows, topk + 4), -7, dtype=torch.int32, device="cuda")
    weights, ids = weight_buf[:, :topk], id_buf[:, :topk]
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(topk_ops, "topk_gating_fwd", _no_hip)
        topk_ops.topk_gating(weights, ids, logits, bias, True, SCALE)
    _assert_bitwise((weights, ids), _hip(logits, bias, topk, None), rows)
    assert (weight_buf[:, topk:] == -7.0).all()
    assert (id_buf[:, topk:] == -7).all()


@pytest.mark.parametrize(
    "numel, dtype", [(0, torch.int32), (1, torch.int64)], ids=["empty", "int64"]
)
def test_sqrtsoftplus_384_topk_rejects_a_bad_row_count(numel, dtype):
    logits = torch.randn((8, EXPERTS), dtype=torch.bfloat16, device="cuda")
    count = torch.full((numel,), 4, dtype=dtype, device="cuda")
    with pytest.raises(ValueError, match="num_valid_rows"):
        _triton(logits, None, 6, count)


def test_topk_gating_dispatch():
    arch = SQRTSOFTPLUS_384_ARCHS[_arch()]
    max_rows = arch.max_rows
    bias = torch.zeros(EXPERTS, dtype=torch.float32, device="cuda")
    for rows, expected in ((1, True), (max_rows, True), (max_rows + 1, False)):
        logits = torch.empty((rows, EXPERTS), dtype=torch.bfloat16, device="cuda")
        weights = torch.empty((rows, 6), dtype=torch.float32, device="cuda")
        ids = torch.empty((rows, 6), dtype=torch.int32, device="cuda")
        can_use = topk_ops._can_use_sqrtsoftplus_384_topk
        assert can_use(arch, weights, ids, logits, bias) is expected
        assert not can_use(arch, weights, ids, logits, bias.bfloat16())


@pytest.mark.parametrize(
    "need_renorm, score_func", [(False, "sqrtsoftplus"), (True, "sigmoid")]
)
def test_topk_gating_keeps_hip_for_another_call(monkeypatch, need_renorm, score_func):
    """The 384-expert checks in topk_gating send these calls to HIP."""
    # topk_gating binds the launcher at its first call; replace the bound one.
    topk_ops._resolve_sqrtsoftplus_384()
    monkeypatch.setattr(topk_ops, "_sqrtsoftplus_384_launch", _no_triton)
    rows, topk = 8, 6
    logits = _logits(rows, 5, "random")
    bias = _bias("linear")
    weights = torch.empty((rows, topk), dtype=torch.float32, device="cuda")
    ids = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    topk_ops.topk_gating(
        weights, ids, logits, bias, need_renorm, SCALE, score_func=score_func
    )


def _no_triton(*args, **kwargs):
    pytest.fail("topk_gating ran the Triton kernel")


def _no_hip(*args, **kwargs):
    pytest.fail("topk_gating ran the HIP kernel")


def test_topk_gating_graph_replay_with_a_changing_row_count(monkeypatch):
    """Captured topk_gating takes the Triton kernel and stays bit-identical to
    HIP when each replay has another num_valid_rows."""
    rows, topk = 96, 6
    logits = _logits(rows, 3, "random")
    bias = _bias("linear")
    num_valid_rows = torch.tensor([rows], dtype=torch.int32, device="cuda")
    weights = torch.empty((rows, topk), dtype=torch.float32, device="cuda")
    ids = torch.empty((rows, topk), dtype=torch.int32, device="cuda")

    def route():
        topk_ops.topk_gating(
            weights, ids, logits, bias, True, SCALE, num_valid_rows=num_valid_rows
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        # Compile the kernel before capture.
        route()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with monkeypatch.context() as patch:
        patch.setattr(topk_ops, "topk_gating_fwd", _no_hip)
        with torch.cuda.graph(graph):
            route()
    for count in (rows, 40, 1, 0):
        num_valid_rows.fill_(count)
        weights.fill_(-7.0)
        graph.replay()
        _assert_bitwise((weights, ids), _hip(logits, bias, topk, num_valid_rows), count)


_FIRST_LAUNCH_UNDER_CAPTURE = textwrap.dedent("""
    import os

    import torch

    from aiter.ops import topk as topk_ops

    def compiled():
        return any(
            name.startswith("_sqrtsoftplus_384_topk")
            for _, _, names in os.walk(os.environ["TRITON_CACHE_DIR"])
            for name in names
        )

    rows, topk, scale = 96, 6, 1.5
    gen = torch.Generator(device="cuda").manual_seed(5)
    logits = (
        torch.randn((rows, 384), generator=gen, dtype=torch.float32, device="cuda")
        * 4
    ).to(torch.bfloat16)
    bias = torch.linspace(-1, 1, 384, dtype=torch.float32, device="cuda")
    num_valid_rows = torch.tensor([rows], dtype=torch.int32, device="cuda")
    weights = torch.empty((rows, topk), dtype=torch.float32, device="cuda")
    ids = torch.empty((rows, topk), dtype=torch.int32, device="cuda")

    def route():
        topk_ops.topk_gating(
            weights,
            ids,
            logits,
            bias,
            True,
            scale,
            "sqrtsoftplus",
            num_valid_rows=num_valid_rows,
        )

    # The HIP binding gives the reference for each row count.
    counts = (rows, 40, 0)
    refs = []
    for count in counts:
        num_valid_rows.fill_(count)
        topk_ops.topk_gating_fwd(
            weights, ids, logits, bias, True, scale, "sqrtsoftplus", 0, num_valid_rows
        )
        refs.append((weights.clone(), ids.clone()))
    torch.cuda.synchronize()
    assert not compiled(), "the HIP calls compiled the Triton kernel"

    def no_hip(*args, **kwargs):
        raise AssertionError("topk_gating ran the HIP kernel under capture")

    topk_ops.topk_gating_fwd = no_hip
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        route()
    assert compiled(), "the capture did not compile the Triton kernel"

    for count, (ref_weights, ref_ids) in zip(counts, refs):
        num_valid_rows.fill_(count)
        weights.fill_(-7.0)
        ids.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(ids, ref_ids), count
        assert torch.equal(weights.view(torch.int32), ref_weights.view(torch.int32))
    """)


def test_topk_gating_first_triton_launch_under_capture(tmp_path):
    """In a server the first call that takes the Triton kernel can be inside a
    graph capture, so its compile and module load happen there. Run that in a
    new process with an empty Triton cache and check the replays against HIP."""
    env = dict(os.environ, TRITON_CACHE_DIR=str(tmp_path))
    result = subprocess.run(
        [sys.executable, "-c", _FIRST_LAUNCH_UNDER_CAPTURE],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
