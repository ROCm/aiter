# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""fused_moe(herd_topk) on gfx1250: Top-(K+1) candidates finalized to K routes."""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.herd_topk import herd_finalize_candidates

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx_runtime() != "gfx1250",
    reason="K+1 HERD candidates reach fused_moe only on gfx1250",
)


def _reference(ids, values, experts, scale):
    popularity = torch.bincount(ids.flatten().long(), minlength=experts)
    order = ids.argsort(1)
    ids, values = ids.gather(1, order), values.gather(1, order)
    pops = popularity[ids.long()]
    least = pops == pops.min(1, keepdim=True).values
    drop = torch.where(least, values, torch.inf).argmin(1)
    keep = torch.arange(ids.shape[1], device=ids.device)[None, :] != drop[:, None]
    kept = ids[keep].reshape(ids.shape[0], -1)
    weights = values[keep].reshape_as(kept)
    weights = weights / weights.sum(1, keepdim=True).clamp_min(1e-20)
    return kept, weights * scale


def _candidates(tokens, experts, topk):
    generator = torch.Generator(device="cuda").manual_seed(1250 + tokens)
    ids = (
        torch.rand(tokens, experts, device="cuda", generator=generator)
        .topk(topk + 1, dim=1)
        .indices.int()
    )
    # Tied weights exercise the expert-ID tiebreak.
    values = torch.randint(1, 5, ids.shape, device="cuda", generator=generator).float()
    return ids, values


def _moe_weights(experts, dim, generator):
    """MXFP4 GUGU weights and the fused_moe arguments the HERD tests share."""
    from aiter import ActivationType, QuantType, benchmark_data_init
    from aiter.ops.flydsl.moe_common import GateMode
    from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight
    from aiter.utility import dtypes

    def weight(rows, gate_up):
        plain = (
            benchmark_data_init.fill_fp4((experts * rows, dim), "uniform", generator)
            .view(experts, rows, dim // 2)
            .view(dtypes.fp4x2)
        )
        return moe_shuffle_weight(
            plain, experts_cnt=experts, is_guinterleave=True, gate_up=gate_up
        )

    def scale(rows, gate_up):
        plain = torch.full(
            (experts, rows, dim // 32), 127, device="cuda", dtype=torch.uint8
        ).view(dtypes.fp8_e8m0)
        return moe_shuffle_scale(
            plain, experts_cnt=experts, is_guinterleave=True, gate_up=gate_up
        )

    w1, w2 = weight(2 * dim, True), weight(dim, False)
    kwargs = {
        "w1_scale": scale(2 * dim, True),
        "w2_scale": scale(dim, False),
        "activation": ActivationType.Silu,
        "quant_type": QuantType.per_1x32,
        "quant_dtype_a": dtypes.fp8,
        "gate_mode": GateMode.INTERLEAVE.value,
    }
    return w1, w2, kwargs


def _logits_diff(actual, expected):
    x, y = actual.float(), expected.float()
    return float((x - y).square().sum() / (x.square().sum() + y.square().sum()))


@pytest.mark.parametrize("tokens", [16, 64, 128])
@pytest.mark.parametrize("experts,topk", [(128, 4), (896, 16)])
def test_finalize_matches_reference(tokens, experts, topk):
    ids, values = _candidates(tokens, experts, topk)
    expected_ids, expected_weights = _reference(ids, values, experts, 2.5)
    weights, kept = herd_finalize_candidates(values, ids, experts, 2.5)
    torch.testing.assert_close(kept, expected_ids, rtol=0, atol=0)
    torch.testing.assert_close(weights, expected_weights, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("tokens", [16, 128])
@pytest.mark.parametrize("experts,topk", [(128, 4), (896, 16)])
def test_fused_moe_outputs_match_router_drop(tokens, experts, topk):
    from aiter.fused_moe import fused_moe

    dim = 512
    generator = torch.Generator(device="cuda").manual_seed(2713)
    w1, w2, kwargs = _moe_weights(experts, dim, generator)
    hidden = (
        torch.randn(
            tokens, dim, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.01
    )
    ids, values = _candidates(tokens, experts, topk)
    kept_ids, weights = _reference(ids, values, experts, 2.5)
    expected = fused_moe(hidden, w1, w2, weights, kept_ids, **kwargs)

    def run():
        return fused_moe(
            hidden,
            w1,
            w2,
            values,
            ids,
            **kwargs,
            herd_topk=topk,
            herd_routed_scaling_factor=2.5,
        )

    def check_output(actual):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype
        assert torch.isfinite(actual).all()
        # GEMM2 accumulates with bf16 atomics; compare by logits_diff rather
        # than elementwise, which magnifies rounding near zero.
        error = _logits_diff(actual, expected)
        assert error < 1e-4, f"HERD MoE logits_diff={error}"

    check_output(run())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    graph.replay()
    check_output(result)


def test_herd_topk_rejects_k_routes():
    from aiter.fused_moe import fused_moe

    generator = torch.Generator(device="cuda").manual_seed(7)
    w1, w2, kwargs = _moe_weights(128, 512, generator)
    hidden = torch.zeros(16, 512, device="cuda", dtype=torch.bfloat16)
    ids, values = _candidates(16, 128, 4)
    with pytest.raises(ValueError, match="K\\+1 candidate"):
        fused_moe(hidden, w1, w2, values, ids, **kwargs, herd_topk=5)
