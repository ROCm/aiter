# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops import topk as topk_ops


def _is_gfx1250() -> bool:
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    return arch.split(":", 1)[0] == "gfx1250"


pytestmark = pytest.mark.skipif(not _is_gfx1250(), reason="requires gfx1250")

EXPERTS = 896
TOPK = 16


def _outputs(
    rows: int, *, stride: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if stride is None:
        return (
            torch.empty((rows, TOPK), dtype=torch.float32, device="cuda"),
            torch.empty((rows, TOPK), dtype=torch.int32, device="cuda"),
        )
    return (
        torch.empty_strided(
            (rows, TOPK), (stride, 1), dtype=torch.float32, device="cuda"
        ),
        torch.empty_strided(
            (rows, TOPK), (stride, 1), dtype=torch.int32, device="cuda"
        ),
    )


def _run_legacy(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    groups: int = 1,
    topk_groups: int = 1,
    renorm: bool = True,
    scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights, ids = _outputs(logits.shape[0])
    topk_ops.biased_grouped_topk_hip(
        logits,
        bias,
        weights,
        ids,
        groups,
        topk_groups,
        renorm,
        scale,
    )
    return weights, ids


def _run_public(
    api: str,
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    renorm: bool = True,
    scale: float = 1.0,
    output_stride: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights, ids = _outputs(logits.shape[0], stride=output_stride)
    if api == "biased_grouped_topk":
        topk_ops.biased_grouped_topk(logits, bias, weights, ids, 1, 1, renorm, scale)
    else:
        topk_ops.topk_gating(
            weights,
            ids,
            logits,
            bias,
            need_renorm=renorm,
            routed_scaling_factor=scale,
            score_func="sigmoid",
        )
    return weights, ids


def _assert_matches_legacy(
    expected: tuple[torch.Tensor, torch.Tensor],
    actual: tuple[torch.Tensor, torch.Tensor],
) -> None:
    expected_weights, expected_ids = expected
    actual_weights, actual_ids = actual
    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(
        actual_weights,
        expected_weights,
        atol=2e-5,
        rtol=2e-5,
        equal_nan=True,
    )


@pytest.mark.parametrize("api", ["biased_grouped_topk", "topk_gating"])
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 32, 128])
@pytest.mark.parametrize("renorm,scale", [(True, 1.0), (False, 2.5)])
def test_exact_sigmoid_topk_matches_legacy(api, rows, renorm, scale):
    torch.manual_seed(17 + rows)
    logits = torch.randn((rows, EXPERTS), dtype=torch.bfloat16, device="cuda")
    bias = (torch.randn(EXPERTS, device="cuda") * 0.1).to(torch.bfloat16)
    expected = _run_legacy(logits, bias, renorm=renorm, scale=scale)
    actual = _run_public(api, logits, bias, renorm=renorm, scale=scale)
    _assert_matches_legacy(expected, actual)


@pytest.mark.parametrize("api", ["biased_grouped_topk", "topk_gating"])
@pytest.mark.parametrize("case", ["all_tie", "plateau", "extreme", "nan"])
def test_exact_sigmoid_topk_preserves_legacy_edge_semantics(api, case):
    torch.manual_seed(2026)
    bias = torch.zeros(EXPERTS, dtype=torch.bfloat16, device="cuda")
    if case == "all_tie":
        logits = torch.zeros((8, EXPERTS), dtype=torch.bfloat16, device="cuda")
    elif case == "plateau":
        logits = (
            (torch.arange(EXPERTS, device="cuda") % 7)
            .sub_(3)
            .repeat(8, 1)
            .to(torch.bfloat16)
        )
    elif case == "extreme":
        logits = torch.empty((8, EXPERTS), dtype=torch.bfloat16, device="cuda")
        logits[:, 0::4] = 100
        logits[:, 1::4] = -100
        logits[:, 2::4] = 0
        logits[:, 3::4] = torch.linspace(-20, 20, EXPERTS // 4, device="cuda")
        bias = torch.linspace(-1, 1, EXPERTS, device="cuda").to(torch.bfloat16)
    else:
        logits = torch.randn((8, EXPERTS), dtype=torch.bfloat16, device="cuda")
        logits[:, ::113] = float("nan")
        bias[::127] = float("nan")

    _assert_matches_legacy(
        _run_legacy(logits, bias),
        _run_public(api, logits, bias),
    )


@pytest.mark.parametrize("api", ["biased_grouped_topk", "topk_gating"])
def test_exact_sigmoid_topk_supports_row_strides_and_cudagraph(api):
    torch.manual_seed(7)
    backing = torch.randn((8, 1536), dtype=torch.bfloat16, device="cuda")
    logits = backing[:, 113 : 113 + EXPERTS]
    bias = (torch.randn(EXPERTS, device="cuda") * 0.1).to(torch.bfloat16)
    expected = _run_legacy(logits, bias)
    actual = _run_public(api, logits, bias, output_stride=TOPK + 7)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        if api == "biased_grouped_topk":
            topk_ops.biased_grouped_topk(
                logits, bias, actual[0], actual[1], 1, 1, True, 1.0
            )
        else:
            topk_ops.topk_gating(
                actual[0],
                actual[1],
                logits,
                bias,
                need_renorm=True,
                score_func="sigmoid",
            )
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_legacy(expected, actual)


def test_biased_grouped_topk_noncontract_groups_keep_legacy_path():
    torch.manual_seed(11)
    logits = torch.randn((32, EXPERTS), dtype=torch.bfloat16, device="cuda")
    bias = (torch.randn(EXPERTS, device="cuda") * 0.1).to(torch.bfloat16)
    expected = _run_legacy(logits, bias, groups=8, topk_groups=4)
    weights, ids = _outputs(logits.shape[0])
    topk_ops.biased_grouped_topk(logits, bias, weights, ids, 8, 4, True, 1.0)
    _assert_matches_legacy(expected, (weights, ids))


def test_topk_gating_noncontract_bias_keeps_hip_path():
    torch.manual_seed(13)
    logits = torch.randn((32, EXPERTS), dtype=torch.bfloat16, device="cuda")
    bias = torch.randn(EXPERTS, dtype=torch.float32, device="cuda") * 0.1
    expected = _outputs(logits.shape[0])
    actual = _outputs(logits.shape[0])
    topk_ops.topk_gating_fwd(
        expected[0], expected[1], logits, bias, True, 1.0, "sigmoid"
    )
    topk_ops.topk_gating(
        actual[0],
        actual[1],
        logits,
        bias,
        need_renorm=True,
        score_func="sigmoid",
    )
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])


def test_exact_sigmoid_topk_capability_guard():
    logits = torch.empty((8, EXPERTS), dtype=torch.bfloat16, device="cuda")
    bias = torch.empty(EXPERTS, dtype=torch.bfloat16, device="cuda")
    weights, ids = _outputs(8)
    assert topk_ops._can_use_gfx1250_exact_sigmoid_topk(logits, bias, weights, ids)
    assert not topk_ops._can_use_gfx1250_exact_sigmoid_topk(
        logits, bias.float(), weights, ids
    )
    assert not topk_ops._can_use_gfx1250_exact_sigmoid_topk(
        logits[:, ::2], bias, weights, ids
    )
    assert not topk_ops._can_use_gfx1250_exact_sigmoid_topk(
        logits, bias, weights.to(torch.bfloat16), ids
    )
