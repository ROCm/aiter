# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Numerical test for the MXFP8 activation passthrough in fused_moe.

The passthrough lets a caller hand over activations that are already fp8 with
group-32 e8m0 microscales, so fused_moe sorts the scale and skips
requantization. Its entire claim is that taking the shortcut changes nothing,
and that is testable exactly: pre-quantize with the same HIP MX quantizer the
internal path uses, and both sides feed identical fp8 bytes and identical scale
values to the same GEMM. The only thing left varying is which code sorts the
scale, so the two outputs must be *bit-identical* -- not merely close.

Exactness matters here because the failure mode is silent. The scale is consumed
bytewise, so a mis-sorted or mis-strided scale returns plausible-looking wrong
numbers rather than raising, and any tolerance wide enough to absorb quantizer
noise would also absorb a real defect. There is no quantizer noise to absorb.

Scope, stated so it is not overclaimed: this pins the passthrough against the
path it replaces. It is not an absolute-correctness test for the a8w4 MoE, which
op_tests/test_moe_2stage.py and op_tests/flydsl_tests/test_flydsl_moe.py
already cover against a torch reference.

The harness mirrors how SGLang's MoRI dispatch calls this: per_1x32 MXFP4 expert
weights in the shuffled a8w4 layout and GateMode.INTERLEAVE. Swiglu tests force
the decode-sized activation policy to FP8. Plain-SiLU EP selects FP8 directly
on gfx950, so its regression runs with the default threshold.
"""

import importlib
import os

import pytest
import torch

from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk, get_2stage_cfgs
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.quant import (
    _mxfp8_ep_quant,
    _mxfp8_ep_quant_hip,
    dynamic_per_group_scaled_quant,
    per_1x32_f4_quant,
    per_1x32_mx_quant_hip,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

_SKIP_GFX950_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() not in ("gfx950",),
    reason="requires a gfx950 ROCm device for A8W4 MoE",
)

# Decode-range token counts, including the single-token EP rank case.
TOKENS = [1, 8, 112]
MODEL_DIM = 2048
INTER_DIM = 512
EXPERTS = 8
TOPK = 2


@pytest.fixture(autouse=True)
def _force_fp8_activation_path():
    """Match serving: keep q_dtype_a at fp8 even for decode-sized batches."""
    old = os.environ.get("AITER_BF16_FP8_MOE_BOUND")
    os.environ["AITER_BF16_FP8_MOE_BOUND"] = "0"
    yield
    if old is None:
        os.environ.pop("AITER_BF16_FP8_MOE_BOUND", None)
    else:
        os.environ["AITER_BF16_FP8_MOE_BOUND"] = old


def _build(tokens, device="cuda", dtype=dtypes.bf16):
    torch.manual_seed(0)
    x = torch.randn(tokens, MODEL_DIM, dtype=dtype, device=device) / 10
    w1 = torch.randn(EXPERTS, INTER_DIM * 2, MODEL_DIM, dtype=dtype, device=device) / 10
    w2 = torch.randn(EXPERTS, MODEL_DIM, INTER_DIM, dtype=dtype, device=device) / 10
    score = torch.randn(tokens, EXPERTS, dtype=dtype, device=device)
    topk_weight, topk_ids = fused_topk(x, score, TOPK, True)

    w1_q, w1_scale = per_1x32_f4_quant(w1, quant_dtype=dtypes.fp4x2)
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(EXPERTS, INTER_DIM * 2, MODEL_DIM // 2)
    w2_q = w2_q.view(EXPERTS, MODEL_DIM, INTER_DIM // 2)

    w1_shuffled = shuffle_weight_a16w4(w1_q, 16, True)
    w2_shuffled = shuffle_weight_a16w4(w2_q, 16, False)
    w1_shuffled.is_shuffled = True
    w2_shuffled.is_shuffled = True

    return (
        x,
        w1_shuffled,
        w2_shuffled,
        {
            "topk_weight": topk_weight,
            "topk_ids": topk_ids,
            "quant_type": QuantType.per_1x32,
            "activation": ActivationType.Swiglu,
            "gate_mode": GateMode.INTERLEAVE.value,
            "w1_scale": shuffle_scale_a16w4(w1_scale, EXPERTS, True),
            "w2_scale": shuffle_scale_a16w4(w2_scale, EXPERTS, False),
            "dtype": dtypes.bf16,
        },
    )


@_SKIP_GFX950_GPU
@pytest.mark.parametrize("tokens", TOKENS)
def test_passthrough_is_bit_identical_to_internal_quantization(tokens):
    """Taking the shortcut must change nothing at all."""
    x, w1, w2, kwargs = _build(tokens)

    internal = fused_moe(x, w1, w2, **kwargs)

    # Same quantizer the internal path uses, so the GEMM sees identical bytes.
    a1, a1_scale = per_1x32_mx_quant_hip(
        x,
        scale=None,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    assert a1.dtype == dtypes.fp8 and a1_scale.dtype == dtypes.fp8_e8m0
    passthrough = fused_moe(a1, w1, w2, a1_scale=a1_scale, **kwargs)

    assert torch.isfinite(passthrough).all(), "passthrough produced non-finite output"
    assert torch.equal(passthrough, internal), (
        "passthrough is not bit-identical to internal requantization; the sorted "
        "activation scale is not the scale that belongs to these tokens "
        f"(max |delta| = {(passthrough.float() - internal.float()).abs().max().item():.3e})"
    )


@pytest.mark.parametrize(
    "build_gfx,runtime_gfx,is_ep,force_a8w4,expected_dtype",
    [
        ("gfx942", "gfx950", True, "0", dtypes.fp8),
        ("gfx942", "gfx950", False, "0", dtypes.bf16),
        ("gfx950", "gfx942", True, "0", dtypes.bf16),
        ("gfx950", "gfx1250", True, "0", dtypes.fp4x2),
        ("gfx950", "gfx1250", True, "1", dtypes.fp8),
    ],
)
def test_ep_silu_activation_dispatch_uses_runtime_gfx(
    monkeypatch, build_gfx, runtime_gfx, is_ep, force_a8w4, expected_dtype
):
    """Host dispatch: only live gfx950 EP overrides the default activation dtype."""
    fused_moe_module = importlib.import_module("aiter.fused_moe")
    grouped_moe_module = importlib.import_module("aiter.ops.flydsl.grouped_moe_gfx1250")
    monkeypatch.setattr(fused_moe_module, "get_gfx", lambda: build_gfx)
    monkeypatch.setattr(fused_moe_module, "get_gfx_runtime", lambda: runtime_gfx)
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "256")
    monkeypatch.setenv("AITER_FORCE_A8W4", force_a8w4)

    # Exercise the actual fused_moe dispatch, stopping before any GPU launch.
    tokens, model_dim, inter_dim, experts, topk = 8, 512, 256, 8, 2
    x = torch.empty((tokens, model_dim), dtype=dtypes.bf16, device="cpu")
    w1 = torch.empty(
        (experts, 2 * inter_dim, model_dim // 2), dtype=dtypes.fp4x2, device="cpu"
    )
    w2 = torch.empty(
        (experts, model_dim, inter_dim // 2), dtype=dtypes.fp4x2, device="cpu"
    )
    w1.is_shuffled = w2.is_shuffled = True
    topk_ids = torch.zeros((tokens, topk), dtype=torch.int32, device="cpu")
    topk_weights = torch.ones((tokens, topk), dtype=torch.float32, device="cpu")
    expert_mask = (
        torch.ones(experts, dtype=torch.int32, device="cpu") if is_ep else None
    )
    live_rows = torch.tensor([3], dtype=torch.int32, device="cpu")
    result = torch.empty_like(x)

    def capture_dispatch(*args, **kwargs):
        assert kwargs["q_dtype_a"] == expected_dtype
        assert kwargs["expert_mask"] is expert_mask
        assert kwargs["num_local_tokens"] is live_rows
        return result

    monkeypatch.setattr(
        grouped_moe_module, "grouped_gemm_gfx1250_a8w4", capture_dispatch
    )
    out = fused_moe_module._fused_moe_impl(
        x,
        w1,
        w2,
        topk_weights,
        topk_ids,
        expert_mask=expert_mask,
        activation=ActivationType.Silu.value,
        quant_type=QuantType.per_1x32.value,
        gate_mode=GateMode.INTERLEAVE.value,
        num_local_tokens=live_rows,
    )
    assert out is result


@pytest.mark.parametrize(
    "build_gfx,runtime_gfx",
    [("gfx942", "gfx950"), ("gfx950", "gfx942"), ("gfx950", "gfx1250")],
)
def test_ep_silu_a8w4_fallback_uses_runtime_gfx(monkeypatch, build_gfx, runtime_gfx):
    """Host config lookup: the unforced SiLU A8W4 FlyDSL fallback is gfx950-only."""
    fused_moe_module = importlib.import_module("aiter.fused_moe")
    monkeypatch.setattr(fused_moe_module, "get_gfx", lambda: build_gfx)
    monkeypatch.setattr(fused_moe_module, "get_gfx_runtime", lambda: runtime_gfx)
    monkeypatch.setattr(fused_moe_module, "get_cu_num", lambda: 256)
    monkeypatch.setattr(fused_moe_module, "cfg_2stages", ({}, {}))
    monkeypatch.setenv("AITER_FLYDSL_FORCE", "0")
    monkeypatch.setenv("AITER_ONLINE_TUNE", "0")
    get_2stage_cfgs.cache_clear()
    try:
        meta = get_2stage_cfgs(
            8,
            512,
            256,
            8,
            2,
            dtypes.bf16,
            dtypes.fp8,
            dtypes.fp4x2,
            QuantType.per_1x32,
            True,
            ActivationType.Silu,
            False,
            0,
            0,
            is_shuffled=True,
            gate_mode=GateMode.INTERLEAVE.value,
            is_ep=True,
        )
    finally:
        get_2stage_cfgs.cache_clear()

    if runtime_gfx == "gfx950":
        assert meta.stage1.func is fused_moe_module._flydsl_stage1_wrapper
        assert meta.stage2.func is fused_moe_module._flydsl_stage2_wrapper
    else:
        assert meta.stage1.func is fused_moe_module.ck_moe_stage1
        assert meta.stage2.func is fused_moe_module.aiter.ck_moe_stage2_fwd


@_SKIP_GFX950_GPU
def test_ep_silu_interleave_uses_a8w4_below_default_threshold(monkeypatch):
    """On gfx950, small plain-SiLU EP batches preserve exact FP8 passthrough."""
    get_2stage_cfgs.cache_clear()
    monkeypatch.setenv("AITER_BF16_FP8_MOE_BOUND", "256")
    monkeypatch.setenv("AITER_FLYDSL_FORCE", "0")
    x, w1, w2, kwargs = _build(8)
    kwargs["activation"] = ActivationType.Silu
    kwargs["expert_mask"] = torch.ones(EXPERTS, dtype=torch.int32, device=x.device)
    kwargs["ep_has_fake_route"] = False

    internal = fused_moe(x, w1, w2, **kwargs)
    a1, a1_scale = per_1x32_mx_quant_hip(
        x,
        scale=None,
        quant_dtype=dtypes.fp8,
        scale_type=dtypes.fp8_e8m0,
        shuffle=False,
    )
    passthrough = fused_moe(a1, w1, w2, a1_scale=a1_scale, **kwargs)

    assert torch.isfinite(internal).all()
    assert torch.equal(passthrough, internal)


@_SKIP_GFX950_GPU
def test_ep_config_key_supports_legacy_and_routed_only_topk():
    """Both EP route conventions must select the routed-top-k=8 MiMo row."""
    common = (8192, 6144, 2048, 24)
    kwargs = {
        "dtype": torch.bfloat16,
        "q_dtype_a": dtypes.fp8,
        "q_dtype_w": dtypes.fp4x2,
        "q_type": QuantType.per_1x32,
        "use_g1u1": True,
        "activation": ActivationType.Silu,
        "doweight_stage1": False,
        "hidden_pad": 0,
        "intermediate_pad": 0,
        "is_shuffled": True,
        "gate_mode": GateMode.INTERLEAVE.value,
        "is_ep": True,
    }

    get_2stage_cfgs.cache_clear()
    legacy = get_2stage_cfgs(*common, 9, ep_has_fake_route=True, **kwargs)
    routed_only = get_2stage_cfgs(*common, 8, ep_has_fake_route=False, **kwargs)

    legacy_names = (
        legacy.stage1.keywords["kernelName"],
        legacy.stage2.keywords["kernelName"],
    )
    routed_only_names = (
        routed_only.stage1.keywords["kernelName"],
        routed_only.stage2.keywords["kernelName"],
    )
    assert legacy_names == routed_only_names
    assert legacy_names == (
        "flydsl_moe1_afp8_wfp4_bf16_t64x128x256_w3_gui_persist_fp8",
        "flydsl_moe2_afp8_wfp4_bf16_t64x128x256_atomic_persist",
    )


@_SKIP_GFX950_GPU
def test_routed_only_ep_topk_reaches_fused_moe():
    """The public API must accept an EP mask without requiring a fake route."""
    x, w1, w2, kwargs = _build(8)
    kwargs["activation"] = ActivationType.Silu
    expert_mask = torch.ones(EXPERTS, dtype=torch.int32, device=x.device)

    reference = fused_moe(x, w1, w2, **kwargs)
    routed_only = fused_moe(
        x,
        w1,
        w2,
        expert_mask=expert_mask,
        ep_has_fake_route=False,
        **kwargs,
    )

    torch.testing.assert_close(routed_only, reference, atol=1.0, rtol=0.05)


@_SKIP_GFX950_GPU
@pytest.mark.parametrize(
    "capacity,dtype",
    [(512, dtypes.bf16), (8192, dtypes.bf16), (8192, dtypes.fp16)],
)
def test_device_row_limited_mxfp8_quant_graph_replay(capacity, dtype):
    """The dedicated EP op and its small-buffer fallback read replay-time rows."""
    torch.manual_seed(13)
    cols = 512
    x = torch.randn((capacity, cols), dtype=dtype, device="cuda") / 8
    full_out = torch.empty((capacity, cols), dtype=dtypes.fp8, device="cuda")
    full_scale = torch.empty(
        (capacity, cols // 32), dtype=dtypes.fp8_e8m0, device="cuda"
    )
    dynamic_per_group_scaled_quant(full_out, x, full_scale, 32, False)

    num_rows = torch.tensor([257], dtype=torch.int32, device="cuda")
    _mxfp8_ep_quant(x, num_rows)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, scale = _mxfp8_ep_quant(x, num_rows)

    for live_rows in (257, min(1025, capacity), 0):
        num_rows.fill_(live_rows)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out[:live_rows], full_out[:live_rows])
        assert torch.equal(scale[:live_rows], full_scale[:live_rows])


@_SKIP_GFX950_GPU
def test_ep_quant_clamps_device_row_count():
    """Malformed replay counts must not write past the logical tensor capacity."""
    capacity, cols, padding = 8192, 512, 64
    sentinel = 0xA5
    # Extra backing storage keeps the test safe if the logical bound regresses.
    storage = torch.ones(capacity + padding, cols, dtype=dtypes.bf16, device="cuda")
    output = torch.empty_like(storage, dtype=dtypes.fp8)
    scale_bytes = torch.empty(
        capacity + padding, cols // 32, dtype=torch.uint8, device="cuda"
    )
    x, out = storage[:capacity], output[:capacity]
    scales = scale_bytes[:capacity].view(dtypes.fp8_e8m0)
    expected = torch.empty_like(out)
    expected_scales = torch.empty_like(scales)
    dynamic_per_group_scaled_quant(expected, x, expected_scales, 32, False)
    count = torch.tensor([capacity], dtype=torch.int32, device="cuda")
    _mxfp8_ep_quant_hip(out, x, scales, count)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _mxfp8_ep_quant_hip(out, x, scales, count)

    for reported in (capacity, capacity + 1, -1, 0):
        output.view(torch.uint8).fill_(sentinel)
        scale_bytes.fill_(sentinel)
        count.fill_(reported)
        graph.replay()
        torch.cuda.synchronize()
        assert bool((output[capacity:].view(torch.uint8) == sentinel).all())
        assert bool((scale_bytes[capacity:] == sentinel).all())
        if reported > 0:
            assert torch.equal(out, expected)
            assert torch.equal(scales, expected_scales)
        else:
            assert bool((out.view(torch.uint8) == sentinel).all())
            assert bool((scale_bytes[:capacity] == sentinel).all())


@_SKIP_GFX950_GPU
def test_fused_moe_selects_mimo_ep_quant(monkeypatch):
    """The normal public call selects the internal quantizer at MiMo EP16 geometry."""
    module = importlib.import_module("aiter.fused_moe")
    for name, value in {
        "MODEL_DIM": 6144,
        "INTER_DIM": 2048,
        "EXPERTS": 24,
        "TOPK": 8,
    }.items():
        monkeypatch.setitem(_build.__globals__, name, value)
    x, w1, w2, kwargs = _build(8192)
    kwargs["activation"] = ActivationType.Silu
    mask = torch.zeros(384, dtype=torch.int32, device="cuda")
    mask[-24:] = 1
    kwargs["expert_mask"] = mask
    kwargs["topk_ids"].add_(360)
    live_rows = torch.tensor([257], dtype=torch.int32, device="cuda")
    kwargs["num_local_tokens"] = live_rows
    original = module._mxfp8_ep_quant
    calls = []

    def capture(hidden, count):
        quantized, scales = original(hidden, count)
        reference, reference_scales = per_1x32_mx_quant_hip(
            hidden,
            quant_dtype=dtypes.fp8,
            scale_type=dtypes.fp8_e8m0,
            shuffle=False,
            num_rows=count,
        )
        assert torch.equal(quantized[:257], reference[:257])
        assert torch.equal(scales[:257], reference_scales[:257])
        calls.append(count)
        return quantized, scales

    monkeypatch.setattr(module, "_mxfp8_ep_quant", capture)
    actual = fused_moe(x, w1, w2, **kwargs)
    assert len(calls) == 1 and calls[0] is live_rows
    assert torch.isfinite(actual[:257]).all()


@pytest.mark.parametrize(
    "case,expected",
    [
        ({}, "ep"),
        ({"experts": 48, "topk": 9, "fake": True}, "ep"),
        ({"arch": "gfx942"}, "generic"),
        ({"arch": "gfx1250"}, "generic"),
        ({"model_dim": 2048, "inter_dim": 512, "experts": 8}, "generic"),
        ({"ep": False}, "generic"),
        ({"live": False}, "generic"),
        ({"capacity": 4096}, "generic"),
        ({"flydsl": False}, "generic"),
    ],
    ids=[
        "ep16",
        "ep8-legacy",
        "gfx942",
        "gfx1250",
        "other-geometry",
        "dense",
        "no-live-count",
        "small",
        "other-backend",
    ],
)
def test_fused_moe_ep_quant_dispatch_scope(monkeypatch, case, expected):
    """Exercise real two-stage selection with metadata-only tensors, stopping at quantization."""
    module = importlib.import_module("aiter.fused_moe")
    model_dim, inter_dim = case.get("model_dim", 6144), case.get("inter_dim", 2048)
    experts, topk = case.get("experts", 24), case.get("topk", 8)
    capacity = case.get("capacity", 8192)
    stage1 = (
        module._flydsl_stage1_wrapper
        if case.get("flydsl", True)
        else module.ck_moe_stage1
    )
    metadata = module.MOEMetadata(stage1, None, 64, 0, fuse_quant="fp8")
    monkeypatch.setattr(module, "get_2stage_cfgs", lambda *a, **kw: metadata)
    monkeypatch.setattr(module, "get_gfx_runtime", lambda: case.get("arch", "gfx950"))
    calls = []

    class QuantSelected(Exception):
        pass

    def capture(name):
        def call(*args, **kwargs):
            calls.append(name)
            raise QuantSelected

        return call

    monkeypatch.setattr(module, "_mxfp8_ep_quant", capture("ep"))
    monkeypatch.setattr(
        module, "fused_dynamic_mxfp8_quant_moe_sort", capture("generic")
    )
    x = torch.empty(capacity, model_dim, dtype=dtypes.bf16, device="meta")
    ids = torch.empty(1, dtype=torch.int32, device="meta")
    with pytest.raises(QuantSelected):
        module.fused_moe_2stages(
            x,
            torch.empty(
                experts,
                inter_dim * 2,
                model_dim // 2,
                dtype=dtypes.fp4x2,
                device="meta",
            ),
            torch.empty(
                experts, model_dim, inter_dim // 2, dtype=dtypes.fp4x2, device="meta"
            ),
            topk,
            ids,
            torch.empty(1, device="meta"),
            ids,
            ids,
            torch.empty_like(x),
            True,
            64,
            activation=ActivationType.Silu,
            quant_type=QuantType.per_1x32,
            q_dtype_a=dtypes.fp8,
            q_dtype_w=dtypes.fp4x2,
            gate_mode=GateMode.INTERLEAVE,
            expert_mask=ids if case.get("ep", True) else None,
            num_local_tokens=ids if case.get("live", True) else None,
            ep_has_fake_route=case.get("fake", False),
        )
    assert calls == [expected]
