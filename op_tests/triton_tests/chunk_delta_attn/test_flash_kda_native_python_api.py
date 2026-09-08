# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU-only tests for FlashKDA's versioned raw-pointer Python adapter."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

_FLASH_KDA = importlib.import_module("aiter.ops.flash_kda")
_PUBLIC_KDA = importlib.import_module(
    "aiter.ops.triton.kimi_delta_attn.chunk_delta_attn"
)


class _Recorder:
    def __init__(self) -> None:
        self.calls: list[tuple[object, ...]] = []

    def __call__(self, *args: object) -> None:
        self.calls.append(args)


class _CapturingLogger:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def get_logger(self):
        return self

    def isEnabledFor(self, _level: int) -> bool:  # noqa: N802
        return True

    def info(self, message: str) -> None:
        self.messages.append(message)


class _IntSubclass(int):
    pass


def _cpu_public_inputs() -> dict[str, object]:
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
    inputs = _cpu_public_inputs()
    return {
        **inputs,
        "A_log": torch.zeros(1, dtype=torch.float32),
        "dt_bias": torch.zeros(128, dtype=torch.float32),
        "use_qk_l2norm_in_kernel": True,
        "use_gate_in_kernel": True,
        "use_beta_sigmoid_in_kernel": True,
        "safe_gate": True,
        "lower_bound": -5.0,
        "state_v_first": True,
    }


def _cpu_gva_public_inputs() -> dict[str, object]:
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


@pytest.mark.parametrize(
    "value", [True, False, 1.0, "4", object(), _IntSubclass(4)]
)
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
        _FLASH_KDA._call_raw_pointer_binding(
            (recorder, 1), tuple(range(24)), None, 7
        )
    with pytest.raises(RuntimeError, match="unsupported FlashKDA raw ABI version"):
        _FLASH_KDA._call_raw_pointer_binding(
            (recorder, 4), tuple(range(25)), None, 7
        )


def test_public_wrapper_passes_bound_to_native_route(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_native(**kwargs):
        calls.append(kwargs)
        return torch.empty_like(kwargs["v"]), None

    monkeypatch.setattr(
        _PUBLIC_KDA, "flash_kda_native_supported", lambda **_kwargs: True
    )
    monkeypatch.setattr(_PUBLIC_KDA, "flash_kda_native_fwd", fake_native)

    _PUBLIC_KDA.chunk_kimi_delta_attn(
        **_cpu_public_inputs(),
        backend="native",
        max_seqlen_upper_bound=6,
    )

    assert len(calls) == 1
    assert calls[0]["max_seqlen_upper_bound"] == 6


def test_public_wrapper_logs_bound_but_does_not_pass_it_to_triton(monkeypatch):
    calls: list[dict[str, object]] = []
    logger = _CapturingLogger()

    def fake_triton(**kwargs):
        calls.append(kwargs)
        return torch.empty_like(kwargs["v"]), None

    monkeypatch.setattr(_PUBLIC_KDA, "_LOGGER", logger)
    monkeypatch.setattr(_PUBLIC_KDA, "chunk_delta_attn_fwd", fake_triton)

    _PUBLIC_KDA.chunk_kimi_delta_attn(
        **_cpu_public_inputs(),
        backend="triton",
        max_seqlen_upper_bound=6,
    )

    assert len(calls) == 1
    assert "max_seqlen_upper_bound" not in calls[0]
    assert any("max_seqlen_upper_bound=6" in message for message in logger.messages)


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


def test_native_supported_accepts_gva_and_rejects_invalid_head_ratio(monkeypatch):
    monkeypatch.setattr(_FLASH_KDA, "_device_arch", lambda _device: "gfx950")
    monkeypatch.setattr(
        _FLASH_KDA,
        "_get_raw_pointer_binding",
        lambda: (_Recorder(), 3),
    )
    gva = _cpu_gva_public_inputs()

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
    inputs = _cpu_gva_public_inputs()

    assert not _FLASH_KDA.flash_kda_native_supported(**inputs)
    assert "predates the raw-v3 GVA ABI" in _FLASH_KDA._native_rejection_reason(
        **inputs
    )


def test_public_auto_routes_supported_gva_to_native(monkeypatch):
    native_calls: list[dict[str, object]] = []

    def fake_native(**kwargs):
        native_calls.append(kwargs)
        return torch.full_like(kwargs["v"], 1), None

    monkeypatch.delenv("AITER_TRITON_ONLY", raising=False)
    monkeypatch.setattr(_FLASH_KDA, "_device_arch", lambda _device: "gfx950")
    monkeypatch.setattr(
        _FLASH_KDA,
        "_get_raw_pointer_binding",
        lambda: (_Recorder(), 3),
    )
    monkeypatch.setattr(_PUBLIC_KDA, "flash_kda_native_fwd", fake_native)
    monkeypatch.setattr(
        _PUBLIC_KDA,
        "chunk_delta_attn_fwd",
        lambda **_kwargs: pytest.fail("supported GVA unexpectedly reached Triton"),
    )

    inputs = _cpu_gva_public_inputs()
    output, final_state = _PUBLIC_KDA.chunk_kimi_delta_attn(
        **inputs,
        backend="auto",
    )

    assert len(native_calls) == 1
    assert native_calls[0]["q"].shape[2] == 2
    assert native_calls[0]["v"].shape[2] == 4
    assert torch.equal(output, torch.full_like(inputs["v"], 1))
    assert final_state is None


def test_public_auto_falls_back_for_native_ineligible_gva(monkeypatch):
    triton_calls: list[dict[str, object]] = []

    def fake_triton(**kwargs):
        triton_calls.append(kwargs)
        return torch.full_like(kwargs["v"], 2), None

    monkeypatch.delenv("AITER_TRITON_ONLY", raising=False)
    monkeypatch.setattr(_FLASH_KDA, "_device_arch", lambda _device: "gfx950")
    monkeypatch.setattr(
        _FLASH_KDA,
        "_get_raw_pointer_binding",
        lambda: (_Recorder(), 3),
    )
    monkeypatch.setattr(
        _PUBLIC_KDA,
        "flash_kda_native_fwd",
        lambda **_kwargs: pytest.fail("native-ineligible GVA reached native"),
    )
    monkeypatch.setattr(_PUBLIC_KDA, "chunk_delta_attn_fwd", fake_triton)

    inputs = _cpu_gva_public_inputs()
    output, final_state = _PUBLIC_KDA.chunk_kimi_delta_attn(
        **inputs,
        backend="auto",
        chunk_size=64,
    )

    assert len(triton_calls) == 1
    assert triton_calls[0]["q"].shape[2] == 2
    assert triton_calls[0]["v"].shape[2] == 4
    assert triton_calls[0]["allow_flash_kda"] is True
    assert torch.equal(output, torch.full_like(inputs["v"], 2))
    assert final_state is None


def test_public_native_fails_loud_for_native_ineligible_gva(monkeypatch):
    monkeypatch.delenv("AITER_TRITON_ONLY", raising=False)
    monkeypatch.setattr(_FLASH_KDA, "_device_arch", lambda _device: "gfx950")
    monkeypatch.setattr(
        _FLASH_KDA,
        "_get_raw_pointer_binding",
        lambda: (_Recorder(), 3),
    )
    monkeypatch.setattr(
        _PUBLIC_KDA,
        "flash_kda_native_fwd",
        lambda **_kwargs: pytest.fail("native-ineligible GVA reached native"),
    )
    monkeypatch.setattr(
        _PUBLIC_KDA,
        "chunk_delta_attn_fwd",
        lambda **_kwargs: pytest.fail("explicit native unexpectedly fell back"),
    )

    with pytest.raises(
        ValueError,
        match="an explicit chunk_size belongs to the Triton implementation",
    ):
        _PUBLIC_KDA.chunk_kimi_delta_attn(
            **_cpu_gva_public_inputs(),
            backend="native",
            chunk_size=64,
        )


def test_auto_fallback_validates_hint_and_keeps_it_out_of_triton(monkeypatch):
    support_calls: list[dict[str, object]] = []
    triton_calls: list[dict[str, object]] = []

    def fake_supported(**kwargs):
        support_calls.append(kwargs)
        return False

    def fake_triton(**kwargs):
        triton_calls.append(kwargs)
        return torch.empty_like(kwargs["v"]), None

    monkeypatch.setattr(_PUBLIC_KDA, "flash_kda_native_supported", fake_supported)
    monkeypatch.setattr(_PUBLIC_KDA, "chunk_delta_attn_fwd", fake_triton)

    _PUBLIC_KDA.chunk_kimi_delta_attn(
        **_cpu_public_inputs(),
        backend="auto",
        max_seqlen_upper_bound=6,
    )

    assert len(support_calls) == 1
    assert support_calls[0]["max_seqlen_upper_bound"] == 6
    assert len(triton_calls) == 1
    assert "max_seqlen_upper_bound" not in triton_calls[0]


def test_triton_fallback_does_not_hide_invalid_hint(monkeypatch):
    monkeypatch.setattr(
        _PUBLIC_KDA,
        "chunk_delta_attn_fwd",
        lambda **_kwargs: pytest.fail("invalid hint reached Triton"),
    )

    with pytest.raises(TypeError, match="must be a Python int or None"):
        _PUBLIC_KDA.chunk_kimi_delta_attn(
            **_cpu_public_inputs(),
            backend="triton",
            max_seqlen_upper_bound=_IntSubclass(6),
        )
