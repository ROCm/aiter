# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""HERD routing, sorting, dispatch and graph replay regression tests."""

import importlib

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.grouped_moe_gfx1250 import (
    _get_compiled_route_psum_herd,
    herd_route_psum_remap,
)
from aiter.ops.flydsl.herd_topk import herd_finalize_candidates, herd_topk_candidates
from aiter.ops.flydsl.kernels.herd_topk import build_herd_finalize_module

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires an AMD GPU"
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


def _candidates(tokens, experts, topk, softmax=False):
    generator = torch.Generator(device="cuda").manual_seed(1250 + tokens)
    ids = (
        torch.rand(tokens, experts, device="cuda", generator=generator)
        .topk(topk + 1, dim=1)
        .indices.int()
    )
    # Tied weights exercise the expert-ID tiebreak.
    values = torch.randint(1, 5, ids.shape, device="cuda", generator=generator).float()
    return ids, values.softmax(1) if softmax else values


@pytest.mark.parametrize("gfx,wave", [("gfx950", 64), ("gfx1250", 32)])
def test_legacy_topk_and_atom_candidates(monkeypatch, gfx, wave):
    topk = importlib.import_module("aiter.ops.topk")
    herd = importlib.import_module("aiter.ops.flydsl.herd_topk")
    monkeypatch.setattr(topk, "get_gfx", lambda: gfx)
    monkeypatch.setattr(topk, "_FLYDSL_USE_HERD", True)
    monkeypatch.setattr(topk, "_FLYDSL_HERD_MIN_M", 16)
    monkeypatch.setattr(topk, "_FLYDSL_HERD_MAX_M", 128)
    monkeypatch.setattr(herd, "wave_size_of", lambda _: wave)
    logits = torch.zeros(16, 256, device="cuda", dtype=torch.bfloat16)
    bias = torch.zeros(256, device="cuda", dtype=torch.float32)
    weights = torch.empty(16, 6, device="cuda", dtype=torch.float32)
    ids = torch.empty_like(weights, dtype=torch.int32)
    calls = []
    support_calls = []
    original_supported = herd.herd_topk_gating_supported

    def supported(*args):
        support_calls.append(args)
        return original_supported(*args)

    monkeypatch.setattr(herd, "herd_topk_gating_supported", supported)
    monkeypatch.setattr(
        herd, "herd_topk_gating", lambda *args: calls.append(("herd", args))
    )
    monkeypatch.setattr(
        topk, "topk_gating_fwd", lambda *args: calls.append(("standard", args))
    )
    valid = torch.tensor([13], device="cuda", dtype=torch.int32)
    topk.topk_gating(weights, ids, logits, bias, num_valid_rows=valid)
    # Automatic router drop is only for gfx950. gfx1250 bypasses even the
    # HERD profile check and enters the ordinary candidate-selection binding.
    assert len(calls) == 1
    assert calls[0][0] == ("herd" if gfx == "gfx950" else "standard")
    assert calls[0][1][-1] is valid
    assert len(support_calls) == (1 if gfx == "gfx950" else 0)
    # ATOM #2713 selects K+1 without a drop, renormalization or routed scale.
    weights = torch.empty(16, 7, device="cuda", dtype=torch.float32)
    ids = torch.empty_like(weights, dtype=torch.int32)
    topk.topk_gating(weights, ids, logits, bias, False, 1.0)
    assert len(calls) == 2 and calls[-1][0] == "standard"
    assert calls[-1][1][4:6] == (False, 1.0)
    assert len(support_calls) == (2 if gfx == "gfx950" else 0)


@pytest.mark.parametrize("valid", [64, 41, 0])
def test_route_skips_unrouted_pad_rows(valid):
    """The router routes pad rows to expert -1, and a router-dropped HERD step
    reaches the grouped MoE the same way: those routes take no grouped row."""
    if get_gfx_runtime() not in ("gfx950", "gfx1250"):
        pytest.skip("requires gfx950 or gfx1250")
    from aiter.ops.flydsl.kernels.moe_route_maps import DROPPED_ROUTE_ROW
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_topids_to_rows

    experts, topk, tokens = 256, 6, 64
    max_m = tokens * topk
    ids, _ = _candidates(tokens, experts, topk - 1)
    ids[valid:] = -1
    # The counter starts one element into its buffer, so a -1 route reaching
    # the atomic lands in guard[0] instead of in someone else's memory.
    guard = torch.zeros(experts + 1, device="cuda", dtype=torch.int32)
    counts, rows = flydsl_moe_topids_to_rows(ids, experts, max_m, counter=guard[1:])
    assert int(guard[0]) == 0, "a -1 route reached the per-expert counter"
    hist = torch.bincount(ids[:valid].flatten().long(), minlength=experts)
    torch.testing.assert_close(counts.long(), hist, rtol=0, atol=0)
    assert (rows[valid:] == DROPPED_ROUTE_ROW).all()
    expert = ids[:valid].long()
    slot = rows[:valid].long() - expert * max_m
    assert ((slot >= 0) & (slot < hist[expert])).all()
    assert rows[:valid].unique().numel() == valid * topk


@pytest.mark.parametrize("experts,topk", [(128, 4), (256, 6), (512, 16)])
def test_grouped_route_psum_reuses_kernel_across_batches(experts, topk):
    if get_gfx_runtime() not in ("gfx950", "gfx1250"):
        pytest.skip("requires gfx950 or gfx1250")
    launch = _get_compiled_route_psum_herd(topk, False)
    cache_keys = None
    for tokens in (1, 16, 32, 64, 128):
        ids, values = _candidates(tokens, experts, topk)
        kept, weights = _reference(ids, values, experts, 2.5)
        counts, rows, psum, gather_weights = herd_route_psum_remap(
            ids, values, experts, tokens, 32, 2.5
        )
        hist = torch.bincount(kept.flatten().long(), minlength=experts)
        padded = (hist + 31) // 32 * 32
        starts = padded.cumsum(0) - padded
        torch.testing.assert_close(counts.long(), hist, rtol=0, atol=0)
        torch.testing.assert_close(psum.long(), starts + hist, rtol=0, atol=0)
        torch.testing.assert_close(gather_weights, weights, rtol=2e-5, atol=1e-6)
        lower = starts[kept.long()]
        assert ((rows >= lower) & (rows < lower + hist[kept.long()])).all()
        assert rows.unique().numel() == tokens * topk
        assert _get_compiled_route_psum_herd(topk, False) is launch
        if cache_keys is None:
            cache_keys = tuple(launch._mem_cache)
            assert len(cache_keys) == 1
        assert tuple(launch._mem_cache) == cache_keys


def _pad_candidates(ids, values, count, unrouted):
    """The candidates as a router under num_valid_rows leaves them: rows past
    `count` routed to -1 with weight 0, or left as they are."""
    ids, values = ids.clone(), values.clone()
    if unrouted:
        ids[count:] = -1
        values[count:] = 0
    return ids, values


@pytest.mark.parametrize("experts,topk", [(128, 4), (256, 6)])
@pytest.mark.parametrize("unrouted", [True, False])
def test_grouped_route_psum_skips_pad_rows(experts, topk, unrouted):
    """Rows past num_valid_rows stay out of the popularity and take no grouped
    row whatever their candidates hold; the count is read on device."""
    if get_gfx_runtime() not in ("gfx950", "gfx1250"):
        pytest.skip("requires gfx950 or gfx1250")
    from aiter.ops.flydsl.kernels.moe_route_maps import DROPPED_ROUTE_ROW

    tokens = 64
    ids, values = _candidates(tokens, experts, topk)
    cand_ids, cand_values = ids.clone(), values.clone()
    valid = torch.tensor([tokens], device="cuda", dtype=torch.int32)

    def run():
        return herd_route_psum_remap(
            cand_ids, cand_values, experts, tokens, 32, 2.5, num_valid_rows=valid
        )

    run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        counts, rows, psum, gather_weights = run()
    for count in (tokens, 41, 0):
        padded_ids, padded_values = _pad_candidates(ids, values, count, unrouted)
        cand_ids.copy_(padded_ids)
        cand_values.copy_(padded_values)
        valid.fill_(count)
        graph.replay()
        if count:
            kept, weights = _reference(ids[:count], values[:count], experts, 2.5)
        else:
            kept = ids.new_empty(0, topk)
            weights = values.new_empty(0, topk)
        hist = torch.bincount(kept.flatten().long(), minlength=experts)
        padded = (hist + 31) // 32 * 32
        starts = padded.cumsum(0) - padded
        torch.testing.assert_close(counts.long(), hist, rtol=0, atol=0)
        torch.testing.assert_close(psum.long(), starts + hist, rtol=0, atol=0)
        torch.testing.assert_close(
            gather_weights[:count], weights, rtol=2e-5, atol=1e-6
        )
        lower = starts[kept.long()]
        real = rows[:count]
        assert ((real >= lower) & (real < lower + hist[kept.long()])).all()
        assert real.unique().numel() == count * topk
        assert (rows[count:] == DROPPED_ROUTE_ROW).all()
        assert (gather_weights[count:] == 0).all()


@pytest.mark.parametrize("experts,topk", [(128, 4), (256, 6), (896, 16)])
def test_router_finalize_reuses_kernel_across_batches(experts, topk):
    threads = 1024 if experts > 512 else 256
    launch = build_herd_finalize_module(
        topk, experts, True, False, block_threads=threads, has_num_valid_rows=False
    )
    compiled = None
    for tokens in (16, 32, 64, 128):
        ids, values = _candidates(tokens, experts, topk)
        expected_ids, expected_weights = _reference(ids, values, experts, 2.5)
        weights, kept = herd_finalize_candidates(values, ids, experts, 2.5)
        torch.testing.assert_close(kept, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(weights, expected_weights, rtol=2e-5, atol=1e-6)
        if compiled is None:
            compiled = launch._cf
        assert launch._cf is compiled


@pytest.mark.parametrize("score_func", ["sigmoid", "softmax", "sqrtsoftplus"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_candidate_scoring(score_func, dtype):
    tokens, experts, topk = 16, 256, 6
    generator = torch.Generator(device="cuda").manual_seed(6527)
    logits = torch.randn(
        tokens, experts, device="cuda", dtype=dtype, generator=generator
    )
    bias = torch.randn(experts, device="cuda", generator=generator) * 0.025
    if score_func == "softmax":
        scores, bias = logits.float(), None
    elif score_func == "sigmoid":
        scores = logits.float().sigmoid()
    else:
        scores = torch.nn.functional.softplus(logits.float()).sqrt()
    choice = scores if bias is None else scores + bias
    ids = torch.empty(tokens, topk + 1, device="cuda", dtype=torch.int32)
    values = torch.empty_like(ids, dtype=torch.float32)
    herd_topk_candidates(values, ids, logits, bias, score_func)
    torch.testing.assert_close(
        choice.gather(1, ids.long()).sort(1).values,
        choice.topk(topk + 1, dim=1).values.sort(1).values,
        rtol=2e-5,
        atol=1e-6,
    )
    expected = scores.gather(1, ids.long())
    if score_func == "softmax":
        expected = expected.softmax(1)
    torch.testing.assert_close(values, expected, rtol=2e-5, atol=1e-6)


def test_legacy_herd_valid_rows_follow_graph_replay(monkeypatch):
    if get_gfx_runtime() != "gfx950":
        pytest.skip("automatic HERD router drop requires gfx950")
    topk_module = importlib.import_module("aiter.ops.topk")
    monkeypatch.setattr(topk_module, "_FLYDSL_USE_HERD", True)
    monkeypatch.setattr(topk_module, "_FLYDSL_HERD_MIN_M", 16)
    monkeypatch.setattr(topk_module, "_FLYDSL_HERD_MAX_M", 128)
    tokens, experts, topk = 16, 256, 6
    generator = torch.Generator(device="cuda").manual_seed(6938)
    logits = torch.randn(
        tokens, experts, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    bias = torch.randn(experts, device="cuda", generator=generator) * 0.025
    candidate_ids = torch.empty(tokens, topk + 1, device="cuda", dtype=torch.int32)
    candidate_values = torch.empty_like(candidate_ids, dtype=torch.float32)
    herd_topk_candidates(candidate_values, candidate_ids, logits, bias, "sqrtsoftplus")
    ids = torch.empty(tokens, topk, device="cuda", dtype=torch.int32)
    weights = torch.empty_like(ids, dtype=torch.float32)
    valid = torch.tensor([tokens], device="cuda", dtype=torch.int32)

    def run():
        topk_module.topk_gating(
            weights, ids, logits, bias, True, 2.5, num_valid_rows=valid
        )

    run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for count in (tokens, 7, 0):
        valid.fill_(count)
        graph.replay()
        if count:
            expected_ids, expected_weights = _reference(
                candidate_ids[:count], candidate_values[:count], experts, 2.5
            )
            torch.testing.assert_close(ids[:count], expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(
                weights[:count], expected_weights, rtol=2e-5, atol=1e-6
            )
        assert (ids[count:] == -1).all()
        assert (weights[count:] == 0).all()


@pytest.mark.parametrize("gfx", ["gfx950", "gfx1250"])
@pytest.mark.parametrize("tokens", [128, 129])
@pytest.mark.parametrize("drop", ["0", "1"])
def test_herd_topk_is_only_for_gfx1250(monkeypatch, gfx, tokens, drop):
    from aiter import ActivationType, QuantType
    from aiter.ops.flydsl.moe_common import GateMode
    from aiter.utility import dtypes

    moe = importlib.import_module("aiter.fused_moe")
    grouped = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
    monkeypatch.setattr(moe, "get_gfx", lambda: gfx)
    monkeypatch.setenv("AITER_HERD_FUSED_DROP", drop)
    experts, topk, dim = 128, 4, 256
    ids, values = _candidates(tokens, experts, topk)
    hidden = torch.zeros(tokens, dim, device="cuda", dtype=torch.bfloat16)
    w1 = torch.empty(experts, 2 * dim, dim // 2, device="cuda", dtype=dtypes.fp4x2)
    w2 = torch.empty(experts, dim, dim // 2, device="cuda", dtype=dtypes.fp4x2)
    calls = []

    def mock_grouped(hidden, w1, w2, weights, route_ids, **kwargs):
        calls.append((route_ids, weights, kwargs["herd_topk"]))
        return torch.zeros_like(hidden)

    monkeypatch.setattr(grouped, "grouped_gemm_gfx1250_a8w4", mock_grouped)
    if gfx != "gfx1250":
        # Reject before candidate finalize or grouped MoE, regardless of drop.
        herd = importlib.import_module("aiter.ops.flydsl.herd_topk")
        finalize_calls = []
        monkeypatch.setattr(
            herd, "herd_finalize_candidates", lambda *args: finalize_calls.append(args)
        )
        with pytest.raises(ValueError, match="herd_topk is supported only on gfx1250"):
            moe.fused_moe(
                hidden,
                w1,
                w2,
                values,
                ids,
                activation=ActivationType.Silu,
                quant_type=QuantType.per_1x32,
                quant_dtype_a=dtypes.fp8,
                gate_mode=GateMode.INTERLEAVE.value,
                herd_topk=topk,
                herd_routed_scaling_factor=2.5,
            )
        assert not calls and not finalize_calls
        return
    moe.fused_moe(
        hidden,
        w1,
        w2,
        values,
        ids,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        quant_dtype_a=dtypes.fp8,
        gate_mode=GateMode.INTERLEAVE.value,
        herd_topk=topk,
        herd_routed_scaling_factor=2.5,
    )
    selected, weights, selected_topk = calls[-1]
    if drop == "0":
        expected_ids, expected_weights = _reference(ids, values, experts, 2.5)
        assert selected_topk == 0
        torch.testing.assert_close(selected, expected_ids, rtol=0, atol=0)
        torch.testing.assert_close(weights, expected_weights, rtol=2e-5, atol=1e-6)
    else:
        assert selected_topk == topk
        torch.testing.assert_close(selected, ids, rtol=0, atol=0)
        torch.testing.assert_close(weights, values, rtol=0, atol=0)


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
        if get_gfx_runtime() != "gfx1250":
            plain = plain.flatten(0, 1)
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


@pytest.mark.parametrize("tokens", [16, 128])
def test_fused_moe_outputs_match_router_drop(monkeypatch, tokens):
    if get_gfx_runtime() not in ("gfx950", "gfx1250"):
        pytest.skip("HERD MoE supports gfx950 and gfx1250")
    from aiter.fused_moe import fused_moe

    monkeypatch.setenv("AITER_HERD_FUSED_DROP", "1")
    experts, topk, dim = 128, 4, 512
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
        if get_gfx_runtime() == "gfx950":
            # gfx950 finishes HERD in the router and passes ordinary K routes.
            weights, kept = herd_finalize_candidates(values, ids, experts, 2.5)
            return fused_moe(hidden, w1, w2, weights, kept, **kwargs)
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
        # GEMM2 uses bf16 atomic accumulation. Compare the repository's
        # logits_diff metric to avoid magnifying rounding near zero.
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
    monkeypatch.setenv("AITER_HERD_FUSED_DROP", "0")
    check_output(run())


@pytest.mark.parametrize("drop", ["0", "1"])
@pytest.mark.parametrize("unrouted", [True, False])
def test_fused_moe_herd_skips_pad_rows(monkeypatch, drop, unrouted):
    """fused_moe(herd_topk) under num_local_tokens: the real rows match a MoE
    over them alone, the pad rows come out zero, and a replayed CUDA graph
    follows each step's count."""
    if get_gfx_runtime() != "gfx1250":
        pytest.skip("K+1 HERD candidates reach fused_moe only on gfx1250")
    from aiter.fused_moe import fused_moe

    monkeypatch.setenv("AITER_HERD_FUSED_DROP", drop)
    experts, topk, dim, tokens = 128, 4, 512, 64
    generator = torch.Generator(device="cuda").manual_seed(6938)
    w1, w2, kwargs = _moe_weights(experts, dim, generator)
    hidden = (
        torch.randn(
            tokens, dim, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        * 0.01
    )
    ids, values = _candidates(tokens, experts, topk)
    cand_ids, cand_values = ids.clone(), values.clone()
    valid = torch.tensor([tokens], device="cuda", dtype=torch.int32)

    def run():
        return fused_moe(
            hidden,
            w1,
            w2,
            cand_values,
            cand_ids,
            **kwargs,
            herd_topk=topk,
            herd_routed_scaling_factor=2.5,
            num_local_tokens=valid,
        )

    run()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    for count in (tokens, 41, 0):
        padded_ids, padded_values = _pad_candidates(ids, values, count, unrouted)
        cand_ids.copy_(padded_ids)
        cand_values.copy_(padded_values)
        valid.fill_(count)
        graph.replay()
        assert torch.isfinite(result).all()
        assert (result[count:] == 0).all(), f"count={count}: pad rows not zero"
        if count:
            kept, weights = _reference(ids[:count], values[:count], experts, 2.5)
            expected = fused_moe(hidden[:count], w1, w2, weights, kept, **kwargs)
            error = _logits_diff(result[:count], expected)
            assert error < 1e-4, f"count={count}: HERD MoE logits_diff={error}"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32, torch.float16])
@pytest.mark.parametrize("requant", ["0", "1"])
@pytest.mark.parametrize("activation", ["silu", "swiglu", "situv2"])
@pytest.mark.parametrize("stage", [1, 2])
def test_gfx1250_bias_with_and_without_requant(
    monkeypatch, dtype, requant, activation, stage
):
    if get_gfx_runtime() != "gfx1250":
        pytest.skip("requires gfx1250 grouped GEMM hardware")
    from aiter import ActivationType
    from op_tests.flydsl_tests import test_flydsl_grouped_gemm as grouped_test

    monkeypatch.setenv("AITER_FLYDSL_DISABLE_GEMM1_REQUANT", requant)
    original = grouped_test._pattern_packed

    def bias_only_weights(experts, rows, k_pack, **kwargs):
        # GEMM1 has 2*inter rows; GEMM2 has model_dim rows. Zero one
        # stage's weights so its result must come from the bias alone.
        if rows == (1024 if stage == 1 else 512):
            return torch.zeros((experts, rows, k_pack), dtype=torch.uint8)
        return original(experts, rows, k_pack, **kwargs)

    monkeypatch.setattr(grouped_test, "_pattern_packed", bias_only_weights)
    grouped_test.run_moe(
        "a8w4",
        model_dim=512,
        inter_dim=512,
        data_init="constant",
        bias_dtype=dtype,
        bias_column_pattern=True,
        check_aot_cache=False,
        activation={
            "silu": ActivationType.Silu,
            "swiglu": ActivationType.Swiglu,
            "situv2": ActivationType.Situv2,
        }[activation],
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
