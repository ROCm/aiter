# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for FlashKDA's versioned raw-pointer Python adapter."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

_FLASH_KDA = importlib.import_module("aiter.ops.flash_kda")


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> None:
        self.calls.append(args)


class _IntSubclass(int):
    pass


def _cpu_native_inputs() -> dict[str, object]:
    shape = (1, 10, 1, 128)
    return {
        "q": torch.zeros(shape, dtype=torch.bfloat16),
        "k": torch.zeros(shape, dtype=torch.bfloat16),
        "v": torch.zeros(shape, dtype=torch.bfloat16),
        "g": torch.zeros(shape, dtype=torch.bfloat16),
        "beta": torch.zeros((1, 10, 1), dtype=torch.float32),
        "cu_seqlens": torch.tensor([0, 4, 10], dtype=torch.int32),
    }


def _cpu_native_support_inputs() -> dict[str, object]:
    inputs = _cpu_native_inputs()
    return {
        **inputs,
        "A_log": torch.zeros(1, dtype=torch.float32),
        "dt_bias": torch.zeros(128, dtype=torch.float32),
        "lower_bound": -5.0,
    }


def _cpu_gva_native_inputs() -> dict[str, object]:
    inputs = _cpu_native_support_inputs()
    q = torch.zeros((1, 10, 2, 128), dtype=torch.bfloat16)
    return {
        **inputs,
        "q": q,
        "k": q.clone(),
        "v": torch.zeros((1, 10, 4, 128), dtype=torch.bfloat16),
        "g": torch.zeros((1, 10, 4, 128), dtype=torch.bfloat16),
        "beta": torch.zeros((1, 10, 4), dtype=torch.float32),
        "A_log": torch.zeros(4, dtype=torch.float32),
        "dt_bias": torch.zeros(4 * 128, dtype=torch.float32),
    }


@pytest.fixture(autouse=True)
def _reset_raw_binding_cache(monkeypatch):
    monkeypatch.setattr(_FLASH_KDA, "_RAW_POINTER_BINDING", None)
    monkeypatch.setattr(_FLASH_KDA._jit_core, "AITER_REBUILD", False)


@pytest.mark.parametrize("value", [True, False, 1.0, "4", object(), _IntSubclass(4)])
def test_max_seqlen_upper_bound_rejects_non_int_and_bool(value):
    with pytest.raises(TypeError, match="must be a Python int or None"):
        _FLASH_KDA._normalize_max_seqlen_upper_bound(
            value,
            total_tokens=10,
            num_seqs=3,
            dense_seqlen=10,
            is_varlen=True,
        )


@pytest.mark.parametrize("value", [None, 4, 7, 10])
def test_packed_max_seqlen_upper_bound_accepts_closed_range(value):
    assert (
        _FLASH_KDA._normalize_max_seqlen_upper_bound(
            value,
            total_tokens=10,
            num_seqs=3,
            dense_seqlen=10,
            is_varlen=True,
        )
        == value
    )


@pytest.mark.parametrize("value", [-1, 0, 3, 11])
def test_packed_max_seqlen_upper_bound_rejects_out_of_range(value):
    with pytest.raises(ValueError, match=r"ceil\(total_tokens / num_seqs\)"):
        _FLASH_KDA._normalize_max_seqlen_upper_bound(
            value,
            total_tokens=10,
            num_seqs=3,
            dense_seqlen=10,
            is_varlen=True,
        )


@pytest.mark.parametrize("value", [None, -1, 0, 1, 999])
def test_dense_max_seqlen_upper_bound_is_normalized_to_exact_length(value):
    assert (
        _FLASH_KDA._normalize_max_seqlen_upper_bound(
            value,
            total_tokens=20,
            num_seqs=2,
            dense_seqlen=10,
            is_varlen=False,
        )
        == 10
    )


def test_raw_resolver_prefers_callable_v2_and_caches_it(monkeypatch):
    v1 = _Recorder()
    v2 = _Recorder()
    module = SimpleNamespace(
        flash_kda_fwd_hip_raw=v1,
        flash_kda_fwd_hip_raw_v2=v2,
    )
    lookups: list[str] = []

    def fake_get_module(name: str):
        lookups.append(name)
        return module

    monkeypatch.setattr(_FLASH_KDA, "get_module", fake_get_module)

    assert _FLASH_KDA._get_raw_pointer_binding() == (v2, 2)
    assert _FLASH_KDA._get_raw_pointer_op() is v2
    assert lookups == [_FLASH_KDA.MD_NAME]


def test_raw_resolver_prefers_callable_v3(monkeypatch):
    v1 = _Recorder()
    v2 = _Recorder()
    v3 = _Recorder()
    module = SimpleNamespace(
        flash_kda_fwd_hip_raw=v1,
        flash_kda_fwd_hip_raw_v2=v2,
        flash_kda_fwd_hip_raw_v3=v3,
    )
    monkeypatch.setattr(_FLASH_KDA, "get_module", lambda _name: module)

    assert _FLASH_KDA._get_raw_pointer_binding() == (v3, 3)
    assert _FLASH_KDA._get_raw_pointer_op() is v3


@pytest.mark.parametrize("v2_candidate", [None, object()])
def test_raw_resolver_falls_back_to_callable_v1(monkeypatch, v2_candidate):
    v1 = _Recorder()
    module = SimpleNamespace(
        flash_kda_fwd_hip_raw=v1,
        flash_kda_fwd_hip_raw_v2=v2_candidate,
    )
    monkeypatch.setattr(_FLASH_KDA, "get_module", lambda _name: module)

    assert _FLASH_KDA._get_raw_pointer_binding() == (v1, 1)
    assert _FLASH_KDA._get_raw_pointer_op() is v1


def test_raw_resolver_falls_back_when_v2_symbol_is_absent(monkeypatch):
    v1 = _Recorder()
    module = SimpleNamespace(flash_kda_fwd_hip_raw=v1)
    monkeypatch.setattr(_FLASH_KDA, "get_module", lambda _name: module)

    assert _FLASH_KDA._get_raw_pointer_binding() == (v1, 1)


def test_raw_resolver_returns_none_for_descriptor_fallback(monkeypatch):
    monkeypatch.setattr(
        _FLASH_KDA,
        "get_module",
        lambda _name: SimpleNamespace(),
    )
    assert _FLASH_KDA._get_raw_pointer_binding() is None
    assert _FLASH_KDA._get_raw_pointer_op() is None


def test_raw_resolver_retries_after_missing_jit_module(monkeypatch):
    v1 = _Recorder()
    attempts = iter((None, SimpleNamespace(flash_kda_fwd_hip_raw=v1)))

    def fake_get_module(_name: str):
        module = next(attempts)
        if module is None:
            raise ModuleNotFoundError
        return module

    monkeypatch.setattr(_FLASH_KDA, "get_module", fake_get_module)

    assert _FLASH_KDA._get_raw_pointer_binding() is None
    assert _FLASH_KDA._get_raw_pointer_binding() == (v1, 1)


def test_pending_rebuild_discards_stale_raw_binding(monkeypatch):
    stale_v1 = _Recorder()
    fresh_v2 = _Recorder()
    monkeypatch.setattr(_FLASH_KDA, "_RAW_POINTER_BINDING", (stale_v1, 1))
    monkeypatch.setattr(_FLASH_KDA._jit_core, "AITER_REBUILD", True)
    monkeypatch.setattr(_FLASH_KDA._jit_core, "rebuilded_list", [])
    monkeypatch.setattr(
        _FLASH_KDA,
        "get_module",
        lambda _name: SimpleNamespace(flash_kda_fwd_hip_raw_v2=fresh_v2),
    )

    assert _FLASH_KDA._get_raw_pointer_binding() is None
    assert _FLASH_KDA._RAW_POINTER_BINDING is None

    _FLASH_KDA._jit_core.rebuilded_list.append(_FLASH_KDA.MD_NAME)
    assert _FLASH_KDA._get_raw_pointer_binding() == (fresh_v2, 2)


def test_raw_v1_receives_exactly_25_arguments():
    recorder = _Recorder()
    raw_v1_args = tuple(range(25))

    _FLASH_KDA._call_raw_pointer_binding((recorder, 1), raw_v1_args, 9, 7)

    assert recorder.calls == [raw_v1_args]
    assert len(recorder.calls[0]) == 25


@pytest.mark.parametrize("bound", [None, 9])
def test_raw_v2_receives_25_v1_arguments_plus_bound(bound):
    recorder = _Recorder()
    raw_v1_args = tuple(range(25))

    _FLASH_KDA._call_raw_pointer_binding((recorder, 2), raw_v1_args, bound, 7)

    expected_bound = 0 if bound is None else bound
    assert recorder.calls == [(*raw_v1_args, expected_bound)]
    assert len(recorder.calls[0]) == 26


@pytest.mark.parametrize("bound", [None, 9])
def test_raw_v3_receives_bound_and_qk_head_count(bound):
    recorder = _Recorder()
    raw_v1_args = tuple(range(25))

    _FLASH_KDA._call_raw_pointer_binding((recorder, 3), raw_v1_args, bound, 7)

    expected_bound = 0 if bound is None else bound
    assert recorder.calls == [(*raw_v1_args, expected_bound, 7)]
    assert len(recorder.calls[0]) == 27


def test_raw_dispatch_rejects_internal_arity_or_version_drift():
    recorder = _Recorder()
    with pytest.raises(RuntimeError, match="must contain 25 values"):
        _FLASH_KDA._call_raw_pointer_binding((recorder, 1), tuple(range(24)), None, 7)
    with pytest.raises(RuntimeError, match="unsupported FlashKDA raw ABI version"):
        _FLASH_KDA._call_raw_pointer_binding((recorder, 4), tuple(range(25)), None, 7)


@pytest.mark.parametrize(
    ("packed", "bound", "expected_bound"),
    [(True, 6, 6), (True, None, 0), (False, None, 10)],
)
def test_descriptor_receives_bound(monkeypatch, packed, bound, expected_bound):
    recorder = _Recorder()
    inputs = _cpu_gva_native_inputs()
    if not packed:
        inputs["cu_seqlens"] = None

    monkeypatch.setattr(_FLASH_KDA, "_get_raw_pointer_binding", lambda: None)
    monkeypatch.setattr(_FLASH_KDA, "flash_kda_fwd_hip", recorder)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: None)

    output, final_state = _FLASH_KDA._flash_kda_fwd_prevalidated(
        **inputs,
        max_seqlen_upper_bound=bound,
    )

    assert output.shape == inputs["v"].shape
    assert final_state is None
    assert len(recorder.calls) == 1
    assert len(recorder.calls[0]) == 18
    assert recorder.calls[0][-1] == expected_bound


def test_direct_native_api_rejects_int_subclass_before_arch_admission():
    inputs = _cpu_native_support_inputs()
    direct_keys = ("q", "k", "v", "g", "beta", "A_log", "dt_bias", "cu_seqlens")

    with pytest.raises(TypeError, match="must be a Python int or None"):
        _FLASH_KDA.flash_kda_fwd(
            **{key: inputs[key] for key in direct_keys},
            max_seqlen_upper_bound=_IntSubclass(6),
        )


def test_native_supported_rejects_int_subclass(monkeypatch):
    monkeypatch.setattr(_FLASH_KDA, "_device_arch", lambda _device: "gfx950")
    inputs = _cpu_native_support_inputs()

    assert _FLASH_KDA.flash_kda_native_supported(
        **inputs,
        max_seqlen_upper_bound=6,
    )
    assert not _FLASH_KDA.flash_kda_native_supported(
        **inputs,
        max_seqlen_upper_bound=_IntSubclass(6),
    )


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan"), 1e300])
def test_native_supported_rejects_unrepresentable_scale(monkeypatch, scale):
    monkeypatch.setattr(_FLASH_KDA, "_device_arch", lambda _device: "gfx950")
    inputs = _cpu_native_support_inputs()

    assert not _FLASH_KDA.flash_kda_native_supported(**inputs, scale=scale)


@pytest.mark.parametrize(
    "scale",
    [1e300, pytest.param(10**10000, id="integer-overflow")],
)
def test_direct_native_api_rejects_unrepresentable_scale(monkeypatch, scale):
    monkeypatch.setattr(_FLASH_KDA, "_device_arch", lambda _device: "gfx950")
    inputs = _cpu_native_support_inputs()

    with pytest.raises(ValueError, match="representable as float32"):
        _FLASH_KDA.flash_kda_fwd(**inputs, scale=scale)


def test_native_supported_returns_false_for_scale_conversion_overflow(monkeypatch):
    monkeypatch.setattr(_FLASH_KDA, "_device_arch", lambda _device: "gfx950")
    inputs = _cpu_native_support_inputs()

    assert not _FLASH_KDA.flash_kda_native_supported(**inputs, scale=10**10000)


def test_native_supported_rejects_dense_grid_y_overflow(monkeypatch):
    monkeypatch.setattr(_FLASH_KDA, "_device_arch", lambda _device: "gfx950")
    shape = (_FLASH_KDA._HIP_GRID_Y_MAX + 1, 1, 1, 128)
    q = torch.empty(shape, device="meta", dtype=torch.bfloat16)
    inputs = {
        "q": q,
        "k": torch.empty_like(q),
        "v": torch.empty_like(q),
        "g": torch.empty_like(q),
        "beta": torch.empty(shape[:-1], device="meta", dtype=torch.float32),
        "A_log": torch.empty(1, device="meta", dtype=torch.float32),
        "dt_bias": torch.empty(128, device="meta", dtype=torch.float32),
    }

    assert not _FLASH_KDA.flash_kda_native_supported(**inputs)
    assert "grid.y" in _FLASH_KDA._native_rejection_reason(**inputs)


def test_native_supported_accepts_gva_and_rejects_invalid_head_ratio(monkeypatch):
    monkeypatch.setattr(_FLASH_KDA, "_device_arch", lambda _device: "gfx950")
    monkeypatch.setattr(
        _FLASH_KDA,
        "_get_raw_pointer_binding",
        lambda: (_Recorder(), 3),
    )
    gva = _cpu_gva_native_inputs()

    assert _FLASH_KDA.flash_kda_native_supported(**gva)
    ratio4 = {
        **gva,
        "v": torch.zeros((1, 10, 8, 128), dtype=torch.bfloat16),
        "g": torch.zeros((1, 10, 8, 128), dtype=torch.bfloat16),
        "beta": torch.zeros((1, 10, 8), dtype=torch.float32),
        "A_log": torch.zeros(8, dtype=torch.float32),
        "dt_bias": torch.zeros(8 * 128, dtype=torch.float32),
    }
    assert _FLASH_KDA.flash_kda_native_supported(**ratio4)
    invalid = {
        **gva,
        "v": torch.zeros((1, 10, 3, 128), dtype=torch.bfloat16),
        "g": torch.zeros((1, 10, 3, 128), dtype=torch.bfloat16),
        "beta": torch.zeros((1, 10, 3), dtype=torch.float32),
        "A_log": torch.zeros(3, dtype=torch.float32),
        "dt_bias": torch.zeros(3 * 128, dtype=torch.float32),
    }
    assert not _FLASH_KDA.flash_kda_native_supported(**invalid)


def test_native_supported_rejects_gva_with_stale_raw_abi(monkeypatch):
    monkeypatch.setattr(_FLASH_KDA, "_device_arch", lambda _device: "gfx950")
    monkeypatch.setattr(
        _FLASH_KDA,
        "_get_raw_pointer_binding",
        lambda: (_Recorder(), 2),
    )
    inputs = _cpu_gva_native_inputs()

    assert not _FLASH_KDA.flash_kda_native_supported(**inputs)
    assert "predates the raw-v3 GVA ABI" in _FLASH_KDA._native_rejection_reason(
        **inputs
    )
