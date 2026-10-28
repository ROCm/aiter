# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Opus reverse-map and zero-weight padding regressions on gfx942."""

from contextlib import contextmanager

import pytest
import torch

from aiter import ActivationType, QuantType
from aiter.fused_moe import AUX_SORT_OPUS, moe_sorting
from aiter.ops.flydsl.fused_moe_gfx942 import run_flydsl_moe_gfx942
from aiter.ops.flydsl.kernels.moe_gemm_2stage_gfx942 import sorted_sum
from aiter.ops.shuffle import shuffle_weight


@pytest.fixture(autouse=True)
def gfx942_device():
    if not torch.cuda.is_available():
        pytest.skip("requires gfx942")
    arch = torch.cuda.get_device_properties(0).gcnArchName
    if not arch.startswith("gfx942"):
        pytest.skip("requires gfx942")


@contextmanager
def poison_uninitialized_memory():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = True
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = fill


@pytest.mark.parametrize("policy", (0, 1, 2))
@pytest.mark.parametrize("batch,experts", ((65, 16), (2048, 513), (16346, 513)))
def test_reverse_routes(batch, experts, policy):
    topk = 11
    ids = torch.arange(topk, dtype=torch.int32, device="cuda").expand(batch, -1).clone()
    ids[-1, :-1] = 0
    weights = torch.ones((batch, topk), dtype=torch.float32, device="cuda")
    weights[-1] = 0
    with poison_uninitialized_memory():
        sorted_ids, _, _, valid_rows, _, m_indices, reverse = moe_sorting(
            ids,
            weights,
            experts,
            256,
            torch.bfloat16,
            64,
            dispatch_policy=policy,
            output_aux=AUX_SORT_OPUS,
        )
    reverse = reverse.view(batch, topk)
    valid = reverse >= 0
    assert int((~valid).sum()) == 9
    assert bool((reverse[~valid] == -1).all())
    assert bool((reverse[valid] < valid_rows[0]).all())
    # Check every emitted inverse against the actual sorted token/topk pair.
    packed = sorted_ids[reverse[valid].long()]
    slots = torch.arange(batch * topk, device="cuda").view(batch, topk)[valid]
    torch.testing.assert_close(packed & 0xFFFFFF, (slots // topk).int())
    torch.testing.assert_close(packed >> 24, (slots % topk).int())
    torch.testing.assert_close(m_indices[reverse[valid].long()], (slots // topk).int())


@pytest.mark.parametrize("padding", (0, 32, 64, 128))
def test_all_routes_skipped(padding):
    batch, topk, dim = 65, 11, 2560
    locs = torch.full((batch, topk), -1, dtype=torch.int32, device="cuda")
    # The dummy address is in bounds but contains NaN. Select must yield zero.
    rows = torch.full(
        (1, dim + padding // 2), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    guard = 64
    storage = torch.full(
        (batch * dim + 2 * guard,), -123, dtype=torch.bfloat16, device="cuda"
    )
    output = storage[guard:-guard].view(batch, dim)
    sorted_sum(topk, dim, padding)(locs, rows, output, batch)
    assert bool((output == 0).all())
    assert bool((storage[:guard] == -123).all())
    assert bool((storage[-guard:] == -123).all())


def make_padding_case(batch, masked, quant, routing):
    experts, topk, dim, inter = 513, 11, 2560, 320
    fp8 = quant == "fp8"
    dtype = torch.float8_e4m3fnuz if fp8 else torch.bfloat16
    w1 = shuffle_weight(
        torch.full(
            (experts, 2 * inter, dim),
            1 if fp8 else 1 / dim,
            dtype=torch.bfloat16,
            device="cuda",
        ).to(dtype)
    )
    w2 = shuffle_weight(
        torch.full(
            (experts, dim, inter),
            1 if fp8 else 1 / inter,
            dtype=torch.bfloat16,
            device="cuda",
        ).to(dtype)
    )
    hidden = torch.ones((batch, dim), dtype=torch.bfloat16, device="cuda")
    ids = torch.arange(topk, dtype=torch.int32, device="cuda").expand(batch, -1).clone()
    ids[:, -1] = 512
    weights = torch.full((batch, topk), 1 / topk, device="cuda")
    weights[-masked:] = 0
    if routing == "mixed":
        ids[-masked:, :-1] = 0
    elif routing == "allzero":
        ids[-masked:] = 0
    scale1 = torch.full((experts, 2 * inter), 1 / dim, device="cuda") if fp8 else None
    scale2 = torch.full((experts, dim), 1 / inter, device="cuda") if fp8 else None
    config = "64_128_256_True:1x4_64x256:64:128" if fp8 else "64_128_128_True"

    def run():
        return run_flydsl_moe_gfx942(
            hidden,
            w1,
            w2,
            weights,
            ids,
            ActivationType.Silu,
            QuantType.per_Token if fp8 else QuantType.No,
            scale1,
            scale2,
            None,
            None,
            0,
            config,
        )

    return run, weights, ids


def check_padding_output(output, masked):
    assert bool(torch.isfinite(output).all())
    if output.shape[0] > masked:
        assert float((output[:-masked].float() - 0.7310585786).abs().max()) < 0.005
    assert bool((output[-masked:] == 0).all())


@pytest.mark.parametrize("quant", ("bf16", "fp8"))
@pytest.mark.parametrize(
    "batch,masked,routing",
    (
        (2048, 1, "mixed"),
        (16346, 65, "mixed"),
        (2048, 2048, "allzero"),
        (2048, 1, "distinct"),
    ),
)
def test_zero_weight_padding(batch, masked, quant, routing):
    with poison_uninitialized_memory():
        run, _, _ = make_padding_case(batch, masked, quant, routing)
        check_padding_output(run(), masked)


@pytest.mark.parametrize("quant", ("bf16", "fp8"))
def test_padding_graph_replay(quant):
    run, weights, ids = make_padding_case(2048, 65, quant, "mixed")
    run()  # Compile before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    for masked in (65, 2048, 1):
        ids[:, :-1] = torch.arange(10, dtype=torch.int32, device="cuda")
        ids[-masked:, :-1] = 0
        weights.fill_(1 / 11)
        weights[-masked:] = 0
        graph.replay()
        check_padding_output(output, masked)
