# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Routing, fallback, and small integration tests for ``gdn_prefill``."""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn.functional as F

import aiter.ops.gdn_prefill as adapter
from aiter.ops.triton._triton_kernels.gated_delta_rule.gated_delta_rule_utils import (
    assert_close,
)

_D = 128


def _shape_only_tensor(b: int, t: int, h: int) -> torch.Tensor:
    """Represent a route shape without allocating its feature payload."""
    return torch.empty((b, t, h, 0))


def _patch_eligible_gfx942(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass tensor guards only in tests that exercise pure route policy."""
    monkeypatch.setattr(adapter, "_opus_input_error", lambda *args, **kwargs: None)
    monkeypatch.setattr(adapter, "_runtime_target", lambda q: ("gfx942", 80))


@pytest.mark.parametrize(
    ("b", "t", "h", "state_on", "expected"),
    (
        pytest.param(1, 64, 10, False, "cf", id="exact-cf"),
        pytest.param(1, 64, 1, False, "cs", id="exact-cs"),
        pytest.param(1, 192, 128, False, "wf", id="exact-wf"),
        pytest.param(1, 128, 11, False, "ws", id="exact-ws"),
    ),
)
def test_auto_uses_representative_exact_closeout_routes(
    monkeypatch: pytest.MonkeyPatch,
    b: int,
    t: int,
    h: int,
    state_on: bool,
    expected: str,
) -> None:
    _patch_eligible_gfx942(monkeypatch)
    q = _shape_only_tensor(b, t, h)

    selected = adapter.select_gdn_prefill_path(
        q,
        q,
        q,
        initial_state=object() if state_on else None,
        output_final_state=state_on,
    )

    assert selected == expected


@pytest.mark.parametrize(
    ("t", "batch_heads", "with_state_io", "expected"),
    (
        pytest.param(64, 1, False, "wf", id="t64-always-wf"),
        pytest.param(128, 20, False, "ws", id="t128-off-below-21"),
        pytest.param(128, 21, False, "wf", id="t128-off-at-21"),
        pytest.param(256, 23, True, "ws", id="t256-state-below-24"),
        pytest.param(256, 24, True, "wf", id="t256-state-at-24"),
        pytest.param(512, 39, False, "ws", id="t512-below-40"),
        pytest.param(512, 40, False, "wf", id="t512-at-40"),
        pytest.param(8192, 47, False, "ws", id="t8192-below-48"),
        pytest.param(8192, 48, False, "wf", id="t8192-at-48"),
        pytest.param(8256, 63, False, "ws", id="long-below-64"),
        pytest.param(8256, 64, False, "wf", id="long-at-64"),
    ),
)
def test_auto_table_miss_uses_fixed_wu_thresholds(
    monkeypatch: pytest.MonkeyPatch,
    t: int,
    batch_heads: int,
    with_state_io: bool,
    expected: str,
) -> None:
    _patch_eligible_gfx942(monkeypatch)
    monkeypatch.setattr(adapter, "lookup_dense_gfx942_path", lambda *args: None)
    q = _shape_only_tensor(1, t, batch_heads)

    selected = adapter.select_gdn_prefill_path(
        q,
        q,
        q,
        initial_state=object() if with_state_io else None,
        output_final_state=with_state_io,
    )

    assert selected == expected


@pytest.mark.parametrize(
    ("initial_state", "output_final_state"),
    (
        pytest.param(object(), False, id="initial-only"),
        pytest.param(None, True, id="final-only"),
    ),
)
def test_auto_single_sided_state_io_skips_closeout_table(
    monkeypatch: pytest.MonkeyPatch,
    initial_state: object | None,
    output_final_state: bool,
) -> None:
    _patch_eligible_gfx942(monkeypatch)

    def unexpected_lookup(*args: Any) -> str:
        raise AssertionError("single-sided state I/O has no measured table key")

    monkeypatch.setattr(adapter, "lookup_dense_gfx942_path", unexpected_lookup)
    q = _shape_only_tensor(1, 128, 21)

    selected = adapter.select_gdn_prefill_path(
        q,
        q,
        q,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )

    # State-I/O policy uses threshold 24, whereas the no-state threshold is 21.
    assert selected == "ws"


def test_path_triton_forwards_every_argument_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = {
        name: object()
        for name in ("q", "k", "v", "o", "g", "beta", "initial_state", "cu_seqlens")
    }
    expected_scalars = {
        "scale": 0.375,
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "use_chunk_hip": True,
        "use_chunk_flydsl": True,
        "state_dtype": torch.bfloat16,
        "use_exp2": False,
        "num_decodes": 7,
        "num_decode_tokens": 11,
    }
    fallback_result = (object(), object())
    seen: dict[str, Any] = {}

    def fallback(**kwargs: Any) -> tuple[object, object]:
        seen.update(kwargs)
        return fallback_result

    monkeypatch.setattr(adapter, "chunk_gated_delta_rule_opt_vk", fallback)

    result = adapter.gdn_prefill(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        o=inputs["o"],
        g=inputs["g"],
        beta=inputs["beta"],
        scale=expected_scalars["scale"],
        initial_state=inputs["initial_state"],
        output_final_state=expected_scalars["output_final_state"],
        use_qk_l2norm_in_kernel=expected_scalars["use_qk_l2norm_in_kernel"],
        cu_seqlens=inputs["cu_seqlens"],
        use_chunk_hip=expected_scalars["use_chunk_hip"],
        use_chunk_flydsl=expected_scalars["use_chunk_flydsl"],
        state_dtype=expected_scalars["state_dtype"],
        use_exp2=expected_scalars["use_exp2"],
        num_decodes=expected_scalars["num_decodes"],
        num_decode_tokens=expected_scalars["num_decode_tokens"],
        path="triton",
    )

    assert result is fallback_result
    assert set(seen) == set(inputs) | set(expected_scalars)
    for name, value in inputs.items():
        assert seen[name] is value
    for name, value in expected_scalars.items():
        assert seen[name] == value


def _valid_cpu_inputs() -> tuple[torch.Tensor, ...]:
    q = torch.empty((1, 64, 1, _D), dtype=torch.bfloat16)
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    g = torch.empty((1, 64, 1), dtype=torch.float32)
    beta = torch.empty_like(g)
    return q, k, v, g, beta


@pytest.mark.parametrize("path", ("cf", "cs", "wf", "ws"))
def test_explicit_opus_path_raises_on_failed_guard(
    monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    q, k, v, g, beta = _valid_cpu_inputs()
    monkeypatch.setattr(
        adapter,
        "chunk_gated_delta_rule_opt_vk",
        lambda **kwargs: pytest.fail("explicit Opus must not fall back"),
    )

    with pytest.raises(
        ValueError,
        match=rf"path='{path}' is unavailable: q must be a HIP tensor",
    ):
        adapter.gdn_prefill(q, k, v, g=g, beta=beta, path=path)


def test_auto_failed_guard_falls_back_without_mutating_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, k, v, g, beta = _valid_cpu_inputs()
    out = torch.empty_like(v)
    fallback_result = (out, None)
    seen: dict[str, Any] = {}

    def fallback(**kwargs: Any) -> tuple[torch.Tensor, None]:
        seen.update(kwargs)
        return fallback_result

    monkeypatch.setattr(adapter, "chunk_gated_delta_rule_opt_vk", fallback)
    monkeypatch.setattr(
        adapter,
        "opus_gdn_c_prefill_fwd",
        lambda *args, **kwargs: pytest.fail("invalid auto input reached C"),
    )
    monkeypatch.setattr(
        adapter,
        "opus_gdn_wu_prefill_fwd",
        lambda *args, **kwargs: pytest.fail("invalid auto input reached W/U"),
    )

    result = adapter.gdn_prefill(q, k, v, o=out, g=g, beta=beta)

    assert result is fallback_result
    for name, value in {
        "q": q,
        "k": k,
        "v": v,
        "o": out,
        "g": g,
        "beta": beta,
    }.items():
        assert seen[name] is value


def test_unknown_path_is_rejected_before_any_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    q, k, v, g, beta = _valid_cpu_inputs()
    monkeypatch.setattr(
        adapter,
        "chunk_gated_delta_rule_opt_vk",
        lambda **kwargs: pytest.fail("unknown path reached fallback"),
    )

    with pytest.raises(ValueError, match="unsupported path='unknown'"):
        adapter.gdn_prefill(q, k, v, g=g, beta=beta, path="unknown")


@pytest.mark.parametrize(
    ("shape", "backend", "expected_mode", "expected_warps"),
    (
        pytest.param(
            (1, 64, 10),
            "c",
            adapter.OPUS_GDN_C_FUSED,
            None,
            id="auto-cf-mode",
        ),
        pytest.param(
            (1, 64, 1),
            "c",
            adapter.OPUS_GDN_C_SPLIT,
            None,
            id="auto-cs-mode",
        ),
        pytest.param(
            (1, 192, 128),
            "wu",
            adapter.OPUS_GDN_K2_WU_FUSED,
            8,
            id="auto-wf-nw8",
        ),
        pytest.param(
            (1, 128, 11),
            "wu",
            adapter.OPUS_GDN_K2_SPLIT,
            4,
            id="auto-ws-nw4",
        ),
    ),
)
def test_dispatch_forces_backend_family_configuration(
    monkeypatch: pytest.MonkeyPatch,
    shape: tuple[int, int, int],
    backend: str,
    expected_mode: int,
    expected_warps: int | None,
) -> None:
    _patch_eligible_gfx942(monkeypatch)
    # This value would force a different family if the adapter delegated to the
    # W/U backend's environment-controlled auto mode.
    monkeypatch.setenv("OPUS_GDN_SPLIT_THRESHOLD", "-999999")
    q = torch.empty((*shape, 1))
    k = torch.empty_like(q)
    v = torch.empty_like(q)
    out = torch.empty_like(q)
    g, beta = object(), object()
    backend_result = (object(), object())
    seen: dict[str, Any] = {}

    def c_backend(*args: Any, **kwargs: Any) -> tuple[object, object]:
        if backend != "c":
            pytest.fail("W/U route reached the C backend")
        seen["args"] = args
        seen["kwargs"] = kwargs
        return backend_result

    def wu_backend(*args: Any, **kwargs: Any) -> tuple[object, object]:
        if backend != "wu":
            pytest.fail("C route reached the W/U backend")
        seen["args"] = args
        seen["kwargs"] = kwargs
        return backend_result

    monkeypatch.setattr(adapter, "opus_gdn_c_prefill_fwd", c_backend)
    monkeypatch.setattr(adapter, "opus_gdn_wu_prefill_fwd", wu_backend)

    result = adapter.gdn_prefill(
        q,
        k,
        v,
        o=out,
        g=g,
        beta=beta,
        scale=0.25,
        path="auto",
    )

    assert result is backend_result
    assert all(
        actual is expected
        for actual, expected in zip(seen["args"], (q, k, v, g, beta), strict=True)
    )
    kwargs = seen["kwargs"]
    assert kwargs["out"] is out
    assert kwargs["scale"] == 0.25
    if backend == "c":
        assert kwargs["c_mode"] == expected_mode
    else:
        assert kwargs["k2_mode"] == expected_mode
        assert kwargs["BT"] == 64
        assert kwargs["BV"] == 64
        assert kwargs["num_warps"] == expected_warps
        assert kwargs["k1_algo"] == 1
    assert kwargs["use_env_overrides"] is False


def _require_path_runtime(path: str) -> None:
    if not torch.cuda.is_available():
        pytest.skip("GDN prefill integration requires a ROCm GPU")
    try:
        gfx = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).gcnArchName.split(":", 1)[0]
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - broken runtime only
        pytest.skip(f"unable to query ROCm GPU architecture: {exc}")
    supported = ("gfx942",) if path in ("cf", "cs") else ("gfx942", "gfx950")
    if gfx not in supported:
        pytest.skip(f"path={path} requires {'/'.join(supported)}, got {gfx}")


def _make_integration_inputs(seed: int) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    b, t, h = 1, 64, 1
    q = F.normalize(
        torch.randn(b, t, h, _D, dtype=torch.bfloat16, device="cuda"),
        p=2,
        dim=-1,
    )
    k = F.normalize(
        torch.randn(b, t, h, _D, dtype=torch.bfloat16, device="cuda"),
        p=2,
        dim=-1,
    )
    v = torch.randn(b, t, h, _D, dtype=torch.bfloat16, device="cuda") * 0.5
    g = F.logsigmoid(torch.randn(b, t, h, dtype=torch.float32, device="cuda"))
    beta = torch.randn(b, t, h, dtype=torch.float32, device="cuda").sigmoid()
    return q, k, v, g, beta


def _recurrent_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
) -> torch.Tensor:
    q, k, v, g, beta = (
        tensor.transpose(1, 2).contiguous().float() for tensor in (q, k, v, g, beta)
    )
    b, h, t, d = q.shape
    state = torch.zeros((b, h, d, d), dtype=torch.float32, device=q.device)
    output = torch.empty((b, h, t, d), dtype=torch.float32, device=q.device)
    q = q * (d**-0.5)

    for token in range(t):
        k_t = k[:, :, token]
        state = state * g[:, :, token].exp()[..., None, None]
        update = v[:, :, token] - (state * k_t[..., None]).sum(dim=-2)
        update = update * beta[:, :, token, None]
        state = state + k_t.unsqueeze(-1) * update.unsqueeze(-2)
        output[:, :, token] = torch.einsum("bhk,bhkv->bhv", q[:, :, token], state)

    return output.transpose(1, 2).contiguous()


@pytest.mark.parametrize("path", ("cf", "cs", "wf", "ws"))
def test_forced_path_small_shape_matches_reference_and_reuses_output(
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_path_runtime(path)
    # These values are deliberately invalid. The production adapter must not
    # inherit benchmark/debug overrides from the surrounding model process.
    for name in (
        "OPUS_GDN_OUT_VARIANT",
        "OPUS_GDN_SPLIT_NW",
        "OPUS_GDN_HIP_SCAN",
        "OPUS_GDN_SCAN32",
        "OPUS_GDN_REF",
        "OPUS_GDN_OUT_BV",
        "OPUS_GDN_OUT_NW",
        "OPUS_GDN_WF_VARIANT",
        "OPUS_GDN_K2C_SCAN_BV",
        "OPUS_GDN_K2C_OUT_BV",
        "OPUS_GDN_K2C_VARIANT",
    ):
        monkeypatch.setenv(name, "999")
    q, k, v, g, beta = _make_integration_inputs(
        20260803 + {"cf": 1, "cs": 2, "wf": 3, "ws": 4}[path]
    )
    out = torch.empty_like(v)

    with torch.inference_mode():
        reference = _recurrent_reference(q, k, v, g, beta)
        output, final_state = adapter.gdn_prefill(
            q,
            k,
            v,
            o=out,
            g=g,
            beta=beta,
            path=path,
        )

    assert output is out
    assert final_state is None
    assert_close(f"adapter_{path}", reference, output, 0.01)
