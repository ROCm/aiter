# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``fused_moe(..., no_combine=True)``.

``no_combine=True`` returns raw per-route outputs with shape
``(token_num, topk, model_dim)``. The caller is responsible for applying
``topk_weight`` and reducing the topk axis.
"""

import functools
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import aiter
import aiter.fused_moe as _fmoe
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import (
    _flydsl_stage2_wrapper,
    _validate_no_combine_route,
    fused_moe,
    fused_moe_fake,
    torch_moe_stage1,
    torch_moe_stage2_no_combine,
)


HAS_GPU = torch.cuda.is_available()
try:
    from aiter.ops.flydsl.utils import is_flydsl_available

    HAS_FLYDSL = bool(is_flydsl_available())
except Exception:
    HAS_FLYDSL = False

requires_gpu = pytest.mark.skipif(not HAS_GPU, reason="needs CUDA/ROCm GPU")
requires_flydsl = pytest.mark.skipif(
    not (HAS_GPU and HAS_FLYDSL), reason="needs FlyDSL + GPU"
)


def _make_topk_routing(token_num, topk, experts, device, *, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    topk_ids = torch.stack(
        [torch.randperm(experts, generator=gen)[:topk] for _ in range(token_num)]
    ).to(device=device, dtype=dtypes.i32)
    topk_weight = torch.softmax(
        torch.randn(token_num, topk, generator=gen), dim=-1
    ).to(device=device, dtype=dtypes.fp32)
    return topk_weight, topk_ids


def _make_bf16_inputs(token_num=4, topk=2, experts=8, model_dim=64, inter_dim=128):
    hidden_states = torch.empty((token_num, model_dim), dtype=dtypes.bf16)
    w1 = torch.empty((experts, inter_dim * 2, model_dim), dtype=dtypes.bf16)
    w2 = torch.empty((experts, model_dim, inter_dim), dtype=dtypes.bf16)
    topk_weight, topk_ids = _make_topk_routing(
        token_num, topk, experts, device="cpu"
    )
    return hidden_states, w1, w2, topk_weight, topk_ids


def _make_fp4_inputs(token_num, topk, experts, model_dim, inter_dim, device, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    hidden_states = torch.randn(
        token_num, model_dim, generator=gen, dtype=dtypes.bf16
    ).to(device)
    w1 = torch.randint(
        0,
        256,
        (experts, inter_dim * 2, model_dim // 2),
        generator=gen,
        dtype=torch.uint8,
    ).to(device).view(dtypes.fp4x2)
    w2 = torch.randint(
        0,
        256,
        (experts, model_dim, inter_dim // 2),
        generator=gen,
        dtype=torch.uint8,
    ).to(device).view(dtypes.fp4x2)
    w1_scale = torch.randint(
        120,
        130,
        (experts, inter_dim * 2, model_dim // 32),
        generator=gen,
        dtype=torch.uint8,
    ).to(device).view(dtypes.fp8_e8m0)
    w2_scale = torch.randint(
        120,
        130,
        (experts, model_dim, inter_dim // 32),
        generator=gen,
        dtype=torch.uint8,
    ).to(device).view(dtypes.fp8_e8m0)
    topk_weight, topk_ids = _make_topk_routing(
        token_num, topk, experts, device, seed=seed + 1
    )
    return hidden_states, w1, w2, w1_scale, w2_scale, topk_weight, topk_ids


def _shuffle_a16w4(w1, w2, w1_scale, w2_scale, experts):
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    w1 = shuffle_weight_a16w4(w1, 16, True)
    w2 = shuffle_weight_a16w4(w2, 16, False)
    w1_scale = shuffle_scale_a16w4(
        w1_scale.view(-1, w1_scale.shape[-1]), experts, True
    ).view(dtypes.fp8_e8m0)
    w2_scale = shuffle_scale_a16w4(
        w2_scale.view(-1, w2_scale.shape[-1]), experts, False
    ).view(dtypes.fp8_e8m0)
    return w1, w2, w1_scale, w2_scale


class _FakeMetadata:
    def __init__(self, *, stage2=None, run_1stage=False):
        self.stage1 = None
        self.stage2 = stage2 or _flydsl_stage2("flydsl_moe2_test")
        self.block_m = 32
        self.ksplit = 1
        self.run_1stage = run_1stage
        self.has_bias = False


def _flydsl_stage2(kernel_name):
    return functools.partial(_flydsl_stage2_wrapper, kernelName=kernel_name)


def _patch_kernel_params(*, a_dtype="fp4", b_dtype="fp4"):
    import aiter.ops.flydsl.moe_kernels as mk

    params = {
        "a_dtype": a_dtype,
        "b_dtype": b_dtype,
        "out_dtype": "bf16",
        "tile_m": 32,
        "tile_n": 128,
        "tile_k": 256,
        "mode": "atomic",
    }
    return patch.object(mk, "get_flydsl_kernel_params", return_value=params)


def _get_metadata(
    *,
    token=32,
    model_dim=4096,
    inter_dim=4096,
    experts=32,
    topk=8,
    q_dtype_a=dtypes.fp4x2,
    q_dtype_w=dtypes.fp4x2,
    activation=ActivationType.Swiglu,
    use_g1u1=True,
):
    return _fmoe.get_2stage_cfgs(
        token,
        model_dim,
        inter_dim,
        experts,
        topk,
        dtypes.bf16,
        q_dtype_a,
        q_dtype_w,
        QuantType.per_1x32,
        use_g1u1,
        activation,
        False,
        0,
        0,
        True,
        _fmoe.GateMode.SEPARATED.value,
    )


def _stage_func(stage):
    return getattr(stage, "func", stage)


def _flydsl_selected(
    token_num,
    topk,
    experts,
    model_dim,
    inter_dim,
    *,
    q_dtype_a=dtypes.fp4x2,
    activation=ActivationType.Swiglu,
):
    try:
        metadata = _get_metadata(
            token=_fmoe.get_padded_M(token_num),
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            q_dtype_a=q_dtype_a,
            activation=activation,
        )
    except Exception:
        return False
    return _stage_func(metadata.stage2) is _flydsl_stage2_wrapper


@contextmanager
def _zero_combined_moe_buffer(shape, dtype):
    real_empty = torch.empty

    def patched_empty(*args, **kwargs):
        out = real_empty(*args, **kwargs)
        if out.dim() == 2 and tuple(out.shape) == shape and out.dtype == dtype:
            out.zero_()
        return out

    with patch.object(torch, "empty", side_effect=patched_empty):
        yield


def _assert_reconstructs_combined(common, topk_weight, *, atol_scale=4):
    token_num, model_dim = common["hidden_states"].shape
    with _zero_combined_moe_buffer(
        (token_num, model_dim), common["hidden_states"].dtype
    ):
        combined = fused_moe(**common, no_combine=False)
        per_slot = fused_moe(**common, no_combine=True)

    assert per_slot.shape == (token_num, topk_weight.shape[1], model_dim)
    assert torch.isfinite(combined).all()
    assert torch.isfinite(per_slot).all()

    reconstructed = (
        per_slot.float() * topk_weight.unsqueeze(-1).float()
    ).sum(dim=1).to(combined.dtype)
    diff = (reconstructed - combined).abs()
    max_abs = float(combined.abs().max().item())
    atol = max(max_abs / 128 * atol_scale, 1.0)
    assert float(diff.max().item()) <= atol
    rel_l2 = float(
        (
            diff.float().pow(2).sum().sqrt()
            / (combined.float().pow(2).sum().sqrt() + 1e-6)
        ).item()
    )
    assert rel_l2 < 0.02


# ---------------------------------------------------------------------------
# Route gating and metadata selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "a_dtype,b_dtype",
    [("fp4", "fp4"), ("fp8", "fp4"), ("bf16", "fp4")],
)
def test_no_combine_route_accepts_supported_flydsl_dtype_pairs(a_dtype, b_dtype):
    with _patch_kernel_params(a_dtype=a_dtype, b_dtype=b_dtype):
        _validate_no_combine_route(_FakeMetadata(), doweight_stage1=False)


@pytest.mark.parametrize(
    "metadata,doweight_stage1,patch_params,pattern",
    [
        (_FakeMetadata(run_1stage=True), False, None, "1-stage"),
        (_FakeMetadata(stage2=lambda *args, **kwargs: None), False, None, "stage2"),
        (_FakeMetadata(), True, ("fp4", "fp4"), "doweight_stage1"),
        (_FakeMetadata(), False, ("fp4", "int4"), "a_dtype, b_dtype"),
        (_FakeMetadata(), False, ("fp16", "fp4"), "a_dtype, b_dtype"),
    ],
)
def test_no_combine_route_rejects_unsupported_dispatches(
    metadata, doweight_stage1, patch_params, pattern
):
    ctx = (
        _patch_kernel_params(a_dtype=patch_params[0], b_dtype=patch_params[1])
        if patch_params
        else nullcontext()
    )
    with ctx, pytest.raises(NotImplementedError, match=pattern):
        _validate_no_combine_route(metadata, doweight_stage1=doweight_stage1)


def test_no_combine_route_rejects_unparseable_flydsl_kernel():
    import aiter.ops.flydsl.moe_kernels as mk

    with patch.object(mk, "get_flydsl_kernel_params", return_value=None):
        with pytest.raises(NotImplementedError, match="cannot parse"):
            _validate_no_combine_route(
                _FakeMetadata(stage2=_flydsl_stage2("bogus")),
                doweight_stage1=False,
            )


def test_bf16_metadata_uses_ck_wrapper_for_no_combine(monkeypatch):
    monkeypatch.setattr(_fmoe, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(_fmoe, "get_cu_num", lambda: 256)
    _fmoe.get_2stage_cfgs.cache_clear()

    metadata = _fmoe.get_2stage_cfgs(
        32,
        4096,
        4096,
        32,
        8,
        dtypes.bf16,
        dtypes.bf16,
        dtypes.bf16,
        QuantType.No,
        True,
        ActivationType.Swiglu,
        False,
        0,
        0,
        False,
        _fmoe.GateMode.SEPARATED.value,
    )

    assert _stage_func(metadata.stage2) is _fmoe.ck_moe_stage2
    _validate_no_combine_route(metadata, doweight_stage1=False)


def test_w4a16_metadata_uses_cktile_by_default(monkeypatch):
    monkeypatch.setattr(_fmoe, "_ENABLE_FLYDSL_W4A16", False)
    monkeypatch.setattr(_fmoe, "is_flydsl_available", lambda: True)
    monkeypatch.setattr(_fmoe, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(_fmoe, "get_cu_num", lambda: 256)
    _fmoe.get_2stage_cfgs.cache_clear()

    metadata = _get_metadata(q_dtype_a=dtypes.bf16)

    assert _stage_func(metadata.stage1) is _fmoe.cktile_moe_stage1
    assert _stage_func(metadata.stage2) is _fmoe.cktile_moe_stage2
    _validate_no_combine_route(metadata, doweight_stage1=False)


@pytest.mark.parametrize("activation", [ActivationType.Swiglu, ActivationType.Silu])
def test_w4a16_metadata_can_opt_into_flydsl(monkeypatch, activation):
    import aiter.ops.flydsl.moe_kernels as mk

    monkeypatch.setattr(_fmoe, "_ENABLE_FLYDSL_W4A16", True)
    monkeypatch.setattr(_fmoe, "is_flydsl_available", lambda: True)
    monkeypatch.setattr(_fmoe, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(_fmoe, "get_cu_num", lambda: 256)
    _fmoe.get_2stage_cfgs.cache_clear()

    metadata = _get_metadata(q_dtype_a=dtypes.bf16, activation=activation)

    assert _stage_func(metadata.stage2) is _flydsl_stage2_wrapper
    parsed = mk.get_flydsl_kernel_params(metadata.stage2.keywords["kernelName"])
    assert parsed is not None
    assert (parsed["a_dtype"], parsed["b_dtype"]) == ("bf16", "fp4")


@pytest.mark.parametrize("activation", [ActivationType.Swiglu, ActivationType.Silu])
def test_w4a8_metadata_uses_flydsl(monkeypatch, activation):
    import aiter.ops.flydsl.moe_kernels as mk

    monkeypatch.setattr(_fmoe, "cfg_2stages", ({}, {}))
    monkeypatch.setattr(_fmoe, "is_flydsl_available", lambda: True)
    monkeypatch.setattr(_fmoe, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(_fmoe, "get_cu_num", lambda: 256)
    _fmoe.get_2stage_cfgs.cache_clear()

    metadata = _get_metadata(q_dtype_a=dtypes.fp8, activation=activation)

    assert _stage_func(metadata.stage1) is _fmoe._flydsl_stage1_wrapper
    assert _stage_func(metadata.stage2) is _flydsl_stage2_wrapper
    assert metadata.fuse_quant == ""
    parsed_stage1 = mk.get_flydsl_kernel_params(metadata.stage1.keywords["kernelName"])
    parsed_stage2 = mk.get_flydsl_kernel_params(metadata.stage2.keywords["kernelName"])
    assert (parsed_stage1["a_dtype"], parsed_stage1["b_dtype"]) == ("fp8", "fp4")
    assert (parsed_stage2["a_dtype"], parsed_stage2["b_dtype"]) == ("fp8", "fp4")
    assert not parsed_stage1.get("a_scale_one", False)


def test_public_api_forwards_mxfp4_activation_dtype(monkeypatch):
    captured = {}

    def fake_fused_moe_(**kwargs):
        captured.update(kwargs)
        return torch.empty((4, 64), dtype=dtypes.bf16)

    monkeypatch.setattr(_fmoe, "fused_moe_", fake_fused_moe_)
    hidden_states, _, _, topk_weight, topk_ids = _make_bf16_inputs()
    w1 = torch.empty((8, 128, 32), dtype=dtypes.fp4x2)
    w2 = torch.empty((8, 64, 32), dtype=dtypes.fp4x2)

    fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        quant_type=QuantType.per_1x32,
        mxfp4_activation_dtype="fp8",
    )

    assert captured["mxfp4_activation_dtype"] == "fp8"


@pytest.mark.parametrize(
    "override,expected",
    [("bf16", dtypes.bf16), ("fp8", dtypes.fp8), ("fp4", dtypes.fp4x2)],
)
def test_mxfp4_activation_dtype_override_selects_metadata_dtype(
    monkeypatch, override, expected
):
    class StopAfterMetadata(Exception):
        pass

    captured = {}

    def fake_get_2stage_cfgs(*args):
        captured["q_dtype_a"] = args[6]
        raise StopAfterMetadata

    monkeypatch.setattr(_fmoe, "get_2stage_cfgs", fake_get_2stage_cfgs)
    hidden_states, _, _, topk_weight, topk_ids = _make_bf16_inputs()
    w1 = torch.empty((8, 128, 32), dtype=dtypes.fp4x2)
    w2 = torch.empty((8, 64, 32), dtype=dtypes.fp4x2)

    with pytest.raises(StopAfterMetadata):
        _fmoe.fused_moe_(
            hidden_states,
            w1,
            w2,
            topk_weight,
            topk_ids,
            quant_type=QuantType.per_1x32.value,
            mxfp4_activation_dtype=override,
        )

    assert captured["q_dtype_a"] == expected


# ---------------------------------------------------------------------------
# Backend wrappers
# ---------------------------------------------------------------------------


def _slot_sort_inputs(token_num=3, topk=2, inter_dim=4, model_dim=5):
    inter_states = torch.arange(
        token_num * topk * inter_dim, dtype=dtypes.bf16
    ).view(token_num, topk, inter_dim)
    w1 = torch.empty((1, inter_dim * 2, model_dim), dtype=dtypes.bf16)
    w2 = torch.empty((1, model_dim, inter_dim), dtype=dtypes.bf16)
    sorted_token_ids = torch.tensor(
        [0 | (0 << 24), 0 | (1 << 24), 2 | (1 << 24), 1 | (0 << 24)],
        dtype=dtypes.i32,
    )
    sorted_expert_ids = torch.zeros((1,), dtype=dtypes.i32)
    num_valid_ids = torch.tensor([4], dtype=dtypes.i32)
    return inter_states, w1, w2, sorted_token_ids, sorted_expert_ids, num_valid_ids


def test_ck_stage2_no_combine_flattens_topk_slots(monkeypatch):
    captured = {}

    def fake_ck_stage2(*args, **kwargs):
        captured["args"] = args
        args[6].fill_(5)
        return args[6]

    monkeypatch.setattr(aiter, "ck_moe_stage2_fwd", fake_ck_stage2)
    inter_states, w1, w2, sorted_ids, sorted_experts, num_valid = _slot_sort_inputs()

    out = _fmoe.ck_moe_stage2(
        inter_states,
        w1,
        w2,
        sorted_ids,
        sorted_experts,
        num_valid,
        out=None,
        topk=2,
        kernelName="",
        w2_scale=None,
        a2_scale=None,
        block_m=32,
        no_combine=True,
    )

    assert tuple(out.shape) == (3, 2, 5)
    assert out.flatten().tolist() == [5] * out.numel()
    assert tuple(captured["args"][0].shape) == (6, 4)
    assert tuple(captured["args"][6].shape) == (6, 5)
    assert captured["args"][7] == 1
    assert captured["args"][3].tolist() == [0, 1, 5, 2]


def test_cktile_stage2_no_combine_flattens_topk_slots(monkeypatch):
    captured = {}

    def fake_cktile_stage2(*args, **kwargs):
        captured["args"] = args
        args[2].fill_(3)
        return args[2]

    monkeypatch.setattr(aiter, "moe_cktile2stages_gemm2", fake_cktile_stage2)
    inter_states, w1, w2, sorted_ids, sorted_experts, num_valid = _slot_sort_inputs()

    out = _fmoe.cktile_moe_stage2(
        inter_states,
        w1,
        w2,
        sorted_ids,
        sorted_experts,
        num_valid,
        out=None,
        topk=2,
        w2_scale=None,
        a2_scale=None,
        block_m=32,
        no_combine=True,
    )

    assert tuple(out.shape) == (3, 2, 5)
    assert out.flatten().tolist() == [3] * out.numel()
    assert tuple(captured["args"][0].shape) == (6, 4)
    assert tuple(captured["args"][2].shape) == (6, 5)
    assert captured["args"][6] == 1
    assert captured["args"][3].tolist() == [0, 1, 5, 2]


# ---------------------------------------------------------------------------
# Fake tensor and codegen invariants
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "no_combine,expected",
    [(False, (4, 64)), (True, (4, 2, 64))],
)
def test_fused_moe_fake_shape_tracks_no_combine(no_combine, expected):
    hidden_states, w1, w2, topk_weight, topk_ids = _make_bf16_inputs()

    out = fused_moe_fake(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        no_combine=no_combine,
    )

    assert out.shape == expected


def test_default_and_explicit_false_fake_shapes_match():
    hidden_states, w1, w2, topk_weight, topk_ids = _make_bf16_inputs()

    implicit = fused_moe_fake(hidden_states, w1, w2, topk_weight, topk_ids)
    explicit = fused_moe_fake(
        hidden_states, w1, w2, topk_weight, topk_ids, no_combine=False
    )

    assert implicit.shape == explicit.shape == (4, 64)
    assert implicit.dtype == explicit.dtype


def test_fake_tensor_mode_returns_per_slot_shape():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        hidden_states = torch.empty((4, 64), dtype=dtypes.bf16, device="cuda")
        w1 = torch.empty((8, 128, 64), dtype=dtypes.bf16, device="cuda")
        w2 = torch.empty((8, 64, 128), dtype=dtypes.bf16, device="cuda")
        topk_weight = torch.empty((4, 2), dtype=dtypes.fp32, device="cuda")
        topk_ids = torch.empty((4, 2), dtype=dtypes.i32, device="cuda")

        out = fused_moe_fake(
            hidden_states, w1, w2, topk_weight, topk_ids, no_combine=True
        )

    assert out.shape == (4, 2, 64)


def test_stage2_codegen_cache_key_distinguishes_accumulate():
    src_path = (
        Path(aiter.__file__).parent
        / "ops"
        / "flydsl"
        / "kernels"
        / "mixed_moe_gemm_2stage.py"
    )
    src = src_path.read_text()

    assert '_acc_tag = "" if accumulate else "_acc0"' in src
    assert "{_acc_tag}" in src


def test_stage2_codegen_module_names_differ_for_accumulate(monkeypatch):
    import flydsl.compiler as flyc
    from aiter.ops.flydsl.kernels import mixed_moe_gemm_2stage as mmg2

    captured = []
    orig_kernel = flyc.kernel

    def spy_kernel(*args, **kwargs):
        captured.append(kwargs.get("name", args[0] if args else None))
        return orig_kernel(*args, **kwargs)

    common = dict(
        model_dim=128,
        inter_dim=128,
        experts=4,
        topk=2,
        tile_m=32,
        tile_n=128,
        tile_k=256,
        doweight_stage2=False,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype="bf16",
        persist_m=1,
        sort_block_m=0,
        b_nt=0,
        model_dim_pad=0,
        inter_dim_pad=0,
        xcd_swizzle=0,
        enable_bias=False,
    )

    mmg2.compile_mixed_moe_gemm2.cache_clear()
    with patch.object(flyc, "kernel", side_effect=spy_kernel):
        mmg2.compile_mixed_moe_gemm2(**common, accumulate=True)
        names_true = [name for name in captured if isinstance(name, str)]
        captured.clear()
        mmg2.compile_mixed_moe_gemm2.cache_clear()
        mmg2.compile_mixed_moe_gemm2(**common, accumulate=False)
        names_false = [name for name in captured if isinstance(name, str)]

    name_true = next(name for name in names_true if name.startswith("mfma_moe2_"))
    name_false = next(name for name in names_false if name.startswith("mfma_moe2_"))
    assert "_acc0" not in name_true
    assert "_acc0" in name_false
    assert name_true != name_false


@pytest.mark.parametrize(
    "case,expected",
    [
        ("w4a16", ("mfma_moe1_swiglu_mul_abf16_wfp4", "mfma_moe2_abf16_wfp4")),
        ("w4a8", ("mfma_moe1_silu_mul_afp8_wfp4", None)),
    ],
)
def test_mixed_precision_flydsl_builders_use_expected_dtype_names(case, expected):
    import flydsl.compiler as flyc
    from aiter.ops.flydsl.kernels import mixed_moe_gemm_2stage as mmg2

    captured = []
    orig_kernel = flyc.kernel

    def spy_kernel(*args, **kwargs):
        captured.append(kwargs.get("name", args[0] if args else None))
        return orig_kernel(*args, **kwargs)

    common = dict(
        model_dim=256,
        inter_dim=128,
        experts=4,
        topk=2,
        tile_m=32,
        tile_n=128,
        tile_k=256,
        b_dtype="fp4",
        out_dtype="bf16",
        persist_m=1,
        model_dim_pad=0,
        inter_dim_pad=0,
    )

    mmg2.compile_mixed_moe_gemm1.cache_clear()
    mmg2.compile_mixed_moe_gemm2.cache_clear()
    with patch.object(flyc, "kernel", side_effect=spy_kernel):
        if case == "w4a16":
            mmg2.compile_mixed_moe_gemm1(
                **common,
                doweight_stage1=False,
                a_dtype="bf16",
                act="swiglu",
                use_async_copy=False,
                k_batch=1,
                gate_mode=_fmoe.GateMode.SEPARATED,
            )
            mmg2.compile_mixed_moe_gemm2(
                **common,
                doweight_stage2=False,
                a_dtype="bf16",
                accumulate=True,
                sort_block_m=0,
            )
        else:
            mmg2.compile_mixed_moe_gemm1(
                **common,
                doweight_stage1=False,
                a_dtype="fp8",
                act="silu",
                use_async_copy=False,
                k_batch=1,
                gate_mode=_fmoe.GateMode.INTERLEAVE,
            )

    assert any(name and expected[0] in name for name in captured)
    if expected[1] is not None:
        assert any(name and expected[1] in name for name in captured)


# ---------------------------------------------------------------------------
# FlyDSL stage2 validation and end-to-end checks
# ---------------------------------------------------------------------------


def _minimal_stage2_inputs(token_num, topk, model_dim, inter_dim, experts, device):
    inter_states = torch.zeros(
        (token_num, topk, inter_dim), dtype=dtypes.bf16, device=device
    )
    w2 = torch.zeros((experts, model_dim, inter_dim), dtype=dtypes.bf16, device=device)
    sorted_ids = torch.zeros(
        (token_num * topk + experts * 32,), dtype=dtypes.i32, device=device
    )
    sorted_expert_ids = torch.zeros((1024,), dtype=dtypes.i32, device=device)
    num_valid_ids = torch.zeros((2,), dtype=dtypes.i32, device=device)
    return inter_states, w2, sorted_ids, sorted_expert_ids, num_valid_ids


@requires_gpu
@pytest.mark.parametrize("bad_case,pattern", [
    ("shape", "out.shape"),
    ("dtype", "out.dtype"),
    ("device", "out.device"),
    ("strides", "contiguous"),
])
def test_flydsl_stage2_rejects_invalid_user_out_buffer(bad_case, pattern):
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage2

    device = "cuda"
    token_num, topk, model_dim, inter_dim, experts = 4, 2, 64, 128, 8
    inter_states, w2, sorted_ids, sorted_experts, num_valid = _minimal_stage2_inputs(
        token_num, topk, model_dim, inter_dim, experts, device
    )
    if bad_case == "shape":
        out = torch.empty((token_num, model_dim), dtype=dtypes.bf16, device=device)
    elif bad_case == "dtype":
        out = torch.empty(
            (token_num, topk, model_dim), dtype=dtypes.fp16, device=device
        )
    elif bad_case == "device":
        out = torch.empty((token_num, topk, model_dim), dtype=dtypes.bf16)
    else:
        out = torch.empty(
            (topk, token_num, model_dim), dtype=dtypes.bf16, device=device
        ).transpose(0, 1)
        assert not out.is_contiguous()

    with pytest.raises(ValueError, match=pattern):
        flydsl_moe_stage2(
            inter_states=inter_states,
            w2=w2,
            sorted_token_ids=sorted_ids,
            sorted_expert_ids=sorted_experts,
            num_valid_ids=num_valid,
            out=out,
            topk=topk,
            return_per_slot=True,
        )


@requires_flydsl
@pytest.mark.parametrize("activation", [ActivationType.Swiglu, ActivationType.Silu])
def test_w4a16_flydsl_no_combine_matches_torch_reference(monkeypatch, activation):
    monkeypatch.setattr(_fmoe, "_ENABLE_FLYDSL_W4A16", True)
    _fmoe.get_2stage_cfgs.cache_clear()

    device = "cuda"
    token_num, topk, experts, model_dim, inter_dim = 16, 2, 4, 512, 512
    if not _flydsl_selected(
        token_num,
        topk,
        experts,
        model_dim,
        inter_dim,
        q_dtype_a=dtypes.bf16,
        activation=activation,
    ):
        pytest.skip("W4A16 FlyDSL route is not selected for this shape")

    (
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weight,
        topk_ids,
    ) = _make_fp4_inputs(token_num, topk, experts, model_dim, inter_dim, device, seed=17)
    w1_shuffled, w2_shuffled, w1_scale_shuffled, w2_scale_shuffled = _shuffle_a16w4(
        w1, w2, w1_scale, w2_scale, experts
    )

    got = fused_moe(
        hidden_states,
        w1_shuffled,
        w2_shuffled,
        topk_weight,
        topk_ids,
        activation=activation,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale_shuffled,
        w2_scale=w2_scale_shuffled,
        mxfp4_activation_dtype="bf16",
        no_combine=True,
    ).float()
    stage1_ref = torch_moe_stage1(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        dtype=dtypes.bf16,
        activation=activation,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
    )
    ref = torch_moe_stage2_no_combine(
        stage1_ref,
        w1,
        w2,
        topk_ids,
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=w2_scale,
    ).float()

    rmse = (got - ref).pow(2).mean().sqrt()
    rel = rmse / (ref.pow(2).mean().sqrt() + 1e-6)
    assert float(rel) < 0.08


@requires_flydsl
def test_flydsl_no_combine_reconstructs_combined_output():
    device = "cuda"
    token_num, topk, experts, model_dim, inter_dim = 256, 4, 128, 3072, 3072
    activation = ActivationType.Swiglu
    if not _flydsl_selected(
        token_num,
        topk,
        experts,
        model_dim,
        inter_dim,
        q_dtype_a=dtypes.fp4x2,
        activation=activation,
    ):
        pytest.skip("W4A4 FlyDSL route is not selected for this shape")

    (
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weight,
        topk_ids,
    ) = _make_fp4_inputs(token_num, topk, experts, model_dim, inter_dim, device)
    common = dict(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weight=topk_weight,
        topk_ids=topk_ids,
        activation=activation,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )
    _assert_reconstructs_combined(common, topk_weight)


@requires_flydsl
def test_w4a8_flydsl_no_combine_returns_finite_per_slot_output(monkeypatch):
    monkeypatch.setattr(_fmoe, "cfg_2stages", ({}, {}))
    _fmoe.get_2stage_cfgs.cache_clear()

    device = "cuda"
    token_num, topk, experts, model_dim, inter_dim = 512, 4, 128, 3072, 3072
    activation = ActivationType.Silu
    if not _flydsl_selected(
        token_num,
        topk,
        experts,
        model_dim,
        inter_dim,
        q_dtype_a=dtypes.fp8,
        activation=activation,
    ):
        pytest.skip("W4A8 FlyDSL route is not selected for this shape")

    (
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weight,
        topk_ids,
    ) = _make_fp4_inputs(token_num, topk, experts, model_dim, inter_dim, device)
    w1, w2, w1_scale, w2_scale = _shuffle_a16w4(
        w1, w2, w1_scale, w2_scale, experts
    )

    out = fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        activation=activation,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        mxfp4_activation_dtype="fp8",
        no_combine=True,
    )

    assert out.shape == (token_num, topk, model_dim)
    assert out.dtype == dtypes.bf16
    assert torch.isfinite(out).all()
    assert out.abs().max() > 0


@requires_flydsl
def test_flydsl_no_combine_zero_fills_unrouted_slots():
    device = "cuda"
    token_num, topk, experts, model_dim, inter_dim = 256, 4, 128, 3072, 3072
    if not _flydsl_selected(
        token_num, topk, experts, model_dim, inter_dim, activation=ActivationType.Swiglu
    ):
        pytest.skip("FlyDSL route is not selected for this shape")

    topk_ids = torch.zeros((token_num, topk), dtype=dtypes.i32, device=device)
    topk_weight = torch.softmax(torch.randn(token_num, topk), dim=-1).to(
        device=device, dtype=dtypes.fp32
    )
    hidden_states, w1, w2, w1_scale, w2_scale, _, _ = _make_fp4_inputs(
        token_num, topk, experts, model_dim, inter_dim, device
    )
    expert_mask = torch.ones(experts, dtype=dtypes.i32, device=device)

    out = fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        expert_mask=expert_mask,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        no_combine=True,
    )

    assert out.shape == (token_num, topk, model_dim)
    assert torch.isfinite(out).all()
    assert (out[:, 0, :] != 0).any()
    assert (out[:, 1:, :] == 0).all()


@requires_flydsl
def test_default_and_explicit_false_preserve_combined_path_shape():
    device = "cuda"
    token_num, topk, experts, model_dim, inter_dim = 256, 4, 128, 3072, 3072
    if not _flydsl_selected(
        token_num, topk, experts, model_dim, inter_dim, activation=ActivationType.Swiglu
    ):
        pytest.skip("FlyDSL route is not selected for this shape")

    (
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weight,
        topk_ids,
    ) = _make_fp4_inputs(token_num, topk, experts, model_dim, inter_dim, device)
    common = dict(
        hidden_states=hidden_states,
        w1=w1,
        w2=w2,
        topk_weight=topk_weight,
        topk_ids=topk_ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )

    out_default = fused_moe(**common)
    out_explicit = fused_moe(**common, no_combine=False)

    assert out_default.shape == out_explicit.shape == (token_num, model_dim)
    assert out_default.dtype == out_explicit.dtype == hidden_states.dtype
    assert torch.isfinite(out_default).all()
    assert torch.isfinite(out_explicit).all()


@requires_flydsl
def test_torch_compile_accepts_no_combine_shape():
    if not hasattr(torch, "compile"):
        pytest.skip("torch.compile is not available")

    device = "cuda"
    token_num, topk, experts, model_dim, inter_dim = 256, 4, 128, 3072, 3072
    if not _flydsl_selected(
        token_num, topk, experts, model_dim, inter_dim, activation=ActivationType.Swiglu
    ):
        pytest.skip("FlyDSL route is not selected for this shape")

    (
        hidden_states,
        w1,
        w2,
        w1_scale,
        w2_scale,
        topk_weight,
        topk_ids,
    ) = _make_fp4_inputs(token_num, topk, experts, model_dim, inter_dim, device)

    def run(hs, w1_, w2_, tw, ti, w1s, w2s):
        return fused_moe(
            hs,
            w1_,
            w2_,
            tw,
            ti,
            activation=ActivationType.Swiglu,
            quant_type=QuantType.per_1x32,
            w1_scale=w1s,
            w2_scale=w2s,
            no_combine=True,
        )

    compiled = torch.compile(run, fullgraph=False, dynamic=False)
    try:
        out = compiled(hidden_states, w1, w2, topk_weight, topk_ids, w1_scale, w2_scale)
    except Exception as exc:
        msg = str(exc).lower()
        if any(token in msg for token in ("schema", "infer_schema", "polymorphic", "rank")):
            pytest.fail(f"torch.compile rejected no_combine=True schema: {exc}")
        pytest.skip(f"runtime failure outside schema coverage: {exc}")

    assert out.shape == (token_num, topk, model_dim)
    assert out.dtype == dtypes.bf16


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
