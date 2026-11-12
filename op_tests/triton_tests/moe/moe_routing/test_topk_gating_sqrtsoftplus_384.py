# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The Triton 384-expert sqrtsoftplus top-k against the HIP topk_gating kernel."""

import pytest
import torch

from aiter.ops import topk as topk_ops
from aiter.ops.triton.moe.moe_routing.topk_gating import (
    SQRTSOFTPLUS_384_EXPERTS,
    SQRTSOFTPLUS_384_LAYOUTS,
    sqrtsoftplus_384_topk,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch


def _arch() -> str | None:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return None
    return get_arch()


pytestmark = pytest.mark.skipif(
    _arch() not in SQRTSOFTPLUS_384_LAYOUTS,
    reason="requires an arch in SQRTSOFTPLUS_384_LAYOUTS",
)

EXPERTS = SQRTSOFTPLUS_384_EXPERTS
SCALE = 1.5


def _max_rows() -> int:
    return topk_ops._SQRTSOFTPLUS_384_ROWS[_arch()][0]


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
    rows = logits.shape[0]
    weights = torch.empty((rows, topk), dtype=torch.float32, device="cuda")
    ids = torch.empty((rows, topk), dtype=torch.int32, device="cuda")
    sqrtsoftplus_384_topk(logits, bias, weights, ids, SCALE, num_valid_rows)
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


@pytest.mark.parametrize(
    "count",
    [
        torch.empty(0, dtype=torch.int32),
        torch.tensor([4], dtype=torch.int64),
    ],
    ids=["empty", "int64"],
)
def test_sqrtsoftplus_384_topk_rejects_a_bad_row_count(count):
    logits = torch.randn((8, EXPERTS), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="num_valid_rows"):
        _triton(logits, None, 6, count.cuda())


@pytest.mark.parametrize("capturing", [False, True])
def test_topk_gating_dispatch(monkeypatch, capturing):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    max_rows, graph_only = topk_ops._SQRTSOFTPLUS_384_ROWS[_arch()]
    taken = capturing or not graph_only
    bias = torch.zeros(EXPERTS, dtype=torch.float32, device="cuda")
    for rows, expected in ((1, taken), (max_rows, taken), (max_rows + 1, False)):
        logits = torch.empty((rows, EXPERTS), dtype=torch.bfloat16, device="cuda")
        weights = torch.empty((rows, 6), dtype=torch.float32, device="cuda")
        ids = torch.empty((rows, 6), dtype=torch.int32, device="cuda")
        can_use = topk_ops._can_use_sqrtsoftplus_384_topk
        assert can_use(weights, ids, logits, bias, True, "sqrtsoftplus") is expected
        assert not can_use(weights, ids, logits, bias, False, "sqrtsoftplus")
        assert not can_use(weights, ids, logits, bias, True, "sigmoid")
        assert not can_use(weights, ids, logits, bias.bfloat16(), True, "sqrtsoftplus")


def _no_hip(*args, **kwargs):
    pytest.fail("topk_gating ran the HIP kernel under capture")


def test_topk_gating_graph_replay_with_a_changing_row_count(monkeypatch):
    """topk_gating takes the Triton kernel under capture and stays bit-identical
    to HIP when each replay has another num_valid_rows."""
    rows, topk = 96, 6
    logits = _logits(rows, 3, "random")
    bias = _bias("linear")
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
            SCALE,
            "sqrtsoftplus",
            num_valid_rows=num_valid_rows,
        )

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        # Compile the kernel before capture; eager topk_gating may run HIP.
        _triton(logits, bias, topk, num_valid_rows)
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
