# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Exact packed-route, padding, auxiliary-index and M3 dispatch contracts."""

import importlib

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes, get_hip_quant
from aiter.jit.utils import chip_info
from aiter.ops.triton.moe import moe_sorting_tiled as tiled

fm = importlib.import_module("aiter.fused_moe")
EXPERTS = 129
MODEL_DIM = 6144


@pytest.fixture(autouse=True)
def _require_gfx950():
    if not torch.cuda.is_available() or torch.version.hip is None:
        pytest.skip("requires a ROCm device")
    if chip_info.get_gfx_runtime() != "gfx950":
        pytest.skip("M3 tiled sorting is enabled on gfx950")


def _routes(tokens, seed=0, skew=False):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if skew:
        starts = torch.randint(4, (tokens, 1), generator=generator, device="cpu")
        routed = (starts + torch.arange(4, device="cpu")) % 4
    else:
        starts = torch.randint(128, (tokens, 1), generator=generator, device="cpu")
        routed = (starts + torch.tensor([0, 13, 37, 73], device="cpu")) % 128
    ids = torch.cat((routed, torch.full((tokens, 1), 128, device="cpu")), dim=1)
    weights = torch.rand(
        (tokens, 5), generator=generator, device="cpu", dtype=torch.float32
    )
    weights[:, -1] = 1
    return ids.to(device="cuda", dtype=torch.int32), weights.to("cuda")


def _reference(ids, weights, block_size):
    """Stable CPU reference, including auxiliary mappings and padded rows."""
    ids, weights = ids.cpu(), weights.cpu()
    tokens, topk = ids.shape
    flat_ids = ids.flatten().long()
    order = torch.argsort(flat_ids, stable=True)
    counts = torch.bincount(flat_ids, minlength=EXPERTS)
    padded = (counts + block_size - 1) // block_size * block_size
    total = int(padded.sum())
    packed = torch.full(
        (total,), tokens | (topk << 24), dtype=torch.int32, device="cpu"
    )
    sorted_weights = torch.zeros(total, dtype=torch.float32, device="cpu")
    sorted_experts = torch.repeat_interleave(
        torch.arange(EXPERTS, dtype=torch.int32, device="cpu"), padded // block_size
    )
    m_indices = torch.full((total,), tokens, dtype=torch.int32, device="cpu")
    reverse = torch.empty(tokens * topk, dtype=torch.int32, device="cpu")
    source, destination = 0, 0
    for count, padded_count in zip(counts.tolist(), padded.tolist()):
        selected = order[source : source + count]
        token, slot = selected // topk, selected % topk
        span = slice(destination, destination + count)
        packed[span] = (token | (slot << 24)).int()
        sorted_weights[span] = weights.flatten()[selected]
        m_indices[span] = token.int()
        reverse[selected] = torch.arange(
            destination, destination + count, device="cpu"
        ).int()
        source += count
        destination += padded_count
    return packed, sorted_weights, sorted_experts, m_indices, reverse


def _assert_result(ids, weights, result, block_size, aux):
    expected = _reference(ids, weights, block_size)
    total = expected[0].numel()
    tokens, topk = ids.shape
    capacity = (
        (tokens * topk + EXPERTS * block_size - topk + block_size - 1)
        // block_size
        * block_size
    )
    assert len(result) == (7 if aux else 5)
    assert result[0].shape == result[1].shape == (capacity,)
    assert result[2].shape == (capacity // block_size,)
    assert result[3].shape == (2,)
    assert result[4].numel() == 0 and result[4].dtype == torch.bfloat16
    assert all(value.device == ids.device for value in result)
    torch.testing.assert_close(
        result[3].cpu(),
        torch.tensor([total, tokens], dtype=torch.int32, device="cpu"),
        rtol=0,
        atol=0,
    )
    for actual, wanted in zip(
        (result[0][:total], result[1][:total], result[2][: total // block_size]),
        expected[:3],
    ):
        torch.testing.assert_close(actual.cpu(), wanted, rtol=0, atol=0)
    if aux:
        assert result[5].shape == (capacity,)
        assert result[6].shape == (tokens * topk,)
        torch.testing.assert_close(result[5][:total].cpu(), expected[3], rtol=0, atol=0)
        torch.testing.assert_close(result[6].cpu(), expected[4], rtol=0, atol=0)


@pytest.mark.parametrize("aux", [False, True])
@pytest.mark.parametrize(
    "tokens,block_size,skew",
    [(16, 64, True), (2039, 32, False), (2039, 128, True), (32768, 64, False)],
)
def test_tiled_sort_stable_packed_routes(tokens, block_size, skew, aux):
    ids, weights = _routes(tokens, skew=skew)
    result = tiled.tiled_sort(
        ids, weights, EXPERTS, MODEL_DIM, torch.bfloat16, block_size, output_aux=aux
    )
    _assert_result(ids, weights, result, block_size, aux)


@pytest.fixture
def m3_routes():
    return _routes(32768)


@pytest.fixture
def opus_dispatch(monkeypatch):
    monkeypatch.setattr(fm, "_USE_CK_MOE_SORTING", False)
    monkeypatch.setattr(fm, "_USE_FLYDSL_MOE_SORTING", False)


@pytest.mark.parametrize("aux", [False, "opus"])
def test_m3_opt_in_changed_input_graph(m3_routes, opus_dispatch, monkeypatch, aux):
    ids, weights = m3_routes
    original = tiled.tiled_sort
    calls = []

    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(tiled, "tiled_sort", tracked)

    def run(enabled):
        return fm.moe_sorting(
            ids,
            weights,
            EXPERTS,
            MODEL_DIM,
            torch.bfloat16,
            64,
            accumulate=False,
            output_aux=aux,
            use_tiled_sort=enabled,
        )

    _assert_result(ids, weights, run(False), 64, bool(aux))
    _assert_result(ids, weights, run(True), 64, bool(aux))
    assert calls, "M3 opt-in silently fell back instead of launching tiled sorting"
    warmup = torch.cuda.Stream()
    warmup.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup):
        for _ in range(3):
            run(True)
    torch.cuda.current_stream().wait_stream(warmup)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(True)
    for seed, skew in [(7, True), (11, False)]:
        new_ids, new_weights = _routes(32768, seed, skew)
        assert not torch.equal(ids, new_ids) and not torch.equal(weights, new_weights)
        ids.copy_(new_ids)
        weights.copy_(new_weights)
        graph.replay()
        _assert_result(ids, weights, captured, 64, bool(aux))
        native = run(False)
        total = int(captured[3][0])
        for index, length in [(0, total), (1, total), (2, total // 64), (3, 2)]:
            torch.testing.assert_close(
                captured[index][:length], native[index][:length], rtol=0, atol=0
            )
        if aux:
            torch.testing.assert_close(
                captured[5][:total], native[5][:total], rtol=0, atol=0
            )
            torch.testing.assert_close(captured[6], native[6], rtol=0, atol=0)


@pytest.mark.parametrize(
    "case",
    [
        "tokens",
        "topk",
        "ids_dtype",
        "weights_dtype",
        "ids_stride",
        "weights_stride",
        "experts",
        "model_dim",
        "buffer_dtype",
        "block_size",
        "expert_mask",
        "num_local_tokens",
        "dispatch_policy",
        "local_topk",
        "accumulate",
        "flat",
        "aux",
    ],
)
def test_m3_gate_falls_back(m3_routes, monkeypatch, case):
    ids, weights = m3_routes
    options = {
        "num_experts": EXPERTS,
        "model_dim": MODEL_DIM,
        "moebuf_dtype": torch.bfloat16,
        "block_size": 64,
        "accumulate": False,
    }
    if case == "tokens":
        ids, weights = ids[:16], weights[:16]
    elif case == "topk":
        ids, weights = ids[:, :4].contiguous(), weights[:, :4].contiguous()
    elif case == "ids_dtype":
        ids = ids.long()
    elif case == "weights_dtype":
        weights = weights.bfloat16()
    elif case == "ids_stride":
        ids = ids.T.contiguous().T
    elif case == "weights_stride":
        weights = weights.T.contiguous().T
    else:
        key, value = {
            "experts": ("num_experts", 128),
            "model_dim": ("model_dim", 4096),
            "buffer_dtype": ("moebuf_dtype", torch.float32),
            "block_size": ("block_size", 32),
            "expert_mask": ("expert_mask", torch.ones(EXPERTS, device=ids.device)),
            "num_local_tokens": (
                "num_local_tokens",
                torch.tensor([32768], device=ids.device),
            ),
            "dispatch_policy": ("dispatch_policy", 1),
            "local_topk": ("return_local_topk_ids", True),
            "accumulate": ("accumulate", True),
            "flat": ("flat", True),
            "aux": ("output_aux", True),
        }[case]
        options[key] = value

    def unexpected(*args, **kwargs):
        pytest.fail("unsupported contract launched tiled sorting")

    monkeypatch.setattr(tiled, "tiled_sort", unexpected)
    assert tiled.try_m3_tiled_sort(ids, weights, **options) is None


def test_m3_gate_rejects_other_arch(m3_routes, monkeypatch):
    ids, weights = m3_routes
    monkeypatch.setattr(chip_info, "get_gfx_runtime", lambda: "gfx942")
    assert (
        tiled.try_m3_tiled_sort(
            ids, weights, EXPERTS, MODEL_DIM, torch.bfloat16, 64, accumulate=False
        )
        is None
    )


def test_non_m3_opt_in_retains_native_sort(opus_dispatch, monkeypatch):
    ids, weights = _routes(2039, seed=19)

    def unexpected(*args, **kwargs):
        pytest.fail("decode/other prefill shape must retain native sorting")

    monkeypatch.setattr(tiled, "tiled_sort", unexpected)
    result = fm.moe_sorting(
        ids,
        weights,
        EXPERTS,
        MODEL_DIM,
        torch.bfloat16,
        64,
        accumulate=False,
        output_aux="opus",
        use_tiled_sort=True,
    )
    _assert_result(ids, weights, result, 64, True)


@pytest.mark.parametrize("supplied_output", [False, True])
def test_default_and_supplied_output_keep_legacy_dispatch(
    m3_routes, opus_dispatch, monkeypatch, supplied_output
):
    ids, weights = m3_routes
    sentinel = object()
    output = object() if supplied_output else None

    def legacy(*args, **kwargs):
        assert kwargs["output"] is output
        return sentinel

    def unexpected(*args, **kwargs):
        pytest.fail("default/off or caller-owned output must use legacy dispatch")

    monkeypatch.setattr(fm, "_moe_sorting_impl", legacy)
    monkeypatch.setattr(tiled, "tiled_sort", unexpected)
    options = {"use_tiled_sort": True} if supplied_output else {}
    assert (
        fm.moe_sorting(
            ids,
            weights,
            EXPERTS,
            MODEL_DIM,
            torch.bfloat16,
            64,
            accumulate=False,
            output=output,
            **options,
        )
        is sentinel
    )


def test_fused_moe_m3_tiled_sort_is_exact(m3_routes, opus_dispatch, monkeypatch):
    """The route-reduce consumer must receive identical sorted data and scales."""
    torch.manual_seed(921)
    tokens, hidden, intermediate = 32768, MODEL_DIM, 768
    x = torch.randn((tokens, hidden), dtype=torch.bfloat16, device="cuda")
    quant = get_hip_quant(QuantType.per_1x32)
    activation, activation_scale = quant(x, quant_dtype=dtypes.fp4x2)
    del x
    w1 = torch.randint(
        256, (EXPERTS, 2 * intermediate, hidden // 2), dtype=torch.uint8, device="cuda"
    ).view(dtypes.fp4x2)
    w2 = torch.randint(
        256, (EXPERTS, hidden, intermediate // 2), dtype=torch.uint8, device="cuda"
    ).view(dtypes.fp4x2)
    w1.is_shuffled = w2.is_shuffled = True
    s1 = torch.full(
        (EXPERTS, 2 * intermediate, hidden // 32), 120, dtype=torch.uint8, device="cuda"
    ).view(dtypes.fp8_e8m0)
    s2 = torch.full(
        (EXPERTS, hidden, intermediate // 32), 120, dtype=torch.uint8, device="cuda"
    ).view(dtypes.fp8_e8m0)
    ids, weights = m3_routes
    original = tiled.tiled_sort
    calls = []

    def tracked(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(tiled, "tiled_sort", tracked)

    def run(enabled):
        return fm.fused_moe(
            activation,
            w1,
            w2,
            weights,
            ids,
            activation=ActivationType.Swiglu,
            quant_type=QuantType.per_1x32,
            w1_scale=s1,
            w2_scale=s2,
            a1_scale=activation_scale,
            dtype=torch.bfloat16,
            swiglu_limit=10.0,
            use_tiled_sort=enabled,
        )

    native = run(False)
    candidate = run(True)
    assert calls, "fused_moe did not propagate its M3 sorting opt-in"
    assert torch.isfinite(native).all() and torch.isfinite(candidate).all()
    torch.testing.assert_close(
        candidate.view(torch.uint8), native.view(torch.uint8), rtol=0, atol=0
    )
