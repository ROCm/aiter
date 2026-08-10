# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Correctness coverage for the standalone dense C-input GDN backend."""

from __future__ import annotations

from functools import cache

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.opus_gdn_c_prefill import (
    OPUS_GDN_C_AUTO,
    OPUS_GDN_C_FUSED,
    OPUS_GDN_C_SPLIT,
    opus_gdn_c_prefill_fwd,
)
from aiter.ops.triton._triton_kernels.gated_delta_rule.gated_delta_rule_utils import (
    assert_close,
)

D = 128
C_MODES = {
    OPUS_GDN_C_AUTO: "auto",
    OPUS_GDN_C_FUSED: "cf",
    OPUS_GDN_C_SPLIT: "cs",
}
DENSE_CASES = (
    (1, 64, 4),
    (1, 96, 8),
    (2, 128, 4),
)


def require_c_runtime() -> None:
    if not torch.cuda.is_available():
        pytest.skip("Opus GDN C-prefill tests require a ROCm GPU")
    try:
        gfx = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).gcnArchName.split(":", 1)[0]
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - broken runtime only
        pytest.skip(f"unable to query ROCm GPU architecture: {exc}")
    if gfx != "gfx942":
        pytest.skip(f"Opus GDN C-prefill requires gfx942, got {gfx}")


def recurrent_gdn_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    *,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sequential fp32 reference; state layout is [B, H, K, V]."""
    q, k, v, beta, g = (
        tensor.transpose(1, 2).contiguous().float() for tensor in (q, k, v, beta, g)
    )
    B, H, T, K = k.shape
    V = v.shape[-1]
    output = torch.zeros(B, H, T, V, dtype=torch.float32, device=q.device)
    state = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    if initial_state is not None:
        state = initial_state.float()
    q = q * (K**-0.5)

    for token in range(T):
        q_t = q[:, :, token]
        k_t = k[:, :, token]
        v_t = v[:, :, token]
        state = state * g[:, :, token].exp()[..., None, None]
        correction = (state * k_t[..., None]).sum(dim=-2)
        v_new = (v_t - correction) * beta[:, :, token, None]
        state = state + k_t.unsqueeze(-1) * v_new.unsqueeze(-2)
        output[:, :, token] = torch.einsum("bhk,bhkv->bhv", q_t, state)

    final = state if output_final_state else None
    return output.transpose(1, 2).contiguous(), final


def make_dense_inputs(
    B: int,
    T: int,
    H: int,
    *,
    with_initial_state: bool,
    seed: int,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(seed)
    q = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda"),
        p=2,
        dim=-1,
    )
    k = F.normalize(
        torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda"),
        p=2,
        dim=-1,
    )
    v = torch.randn(B, T, H, D, dtype=torch.bfloat16, device="cuda") * 0.5
    g = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device="cuda"))
    beta = torch.randn(B, T, H, dtype=torch.float32, device="cuda").sigmoid()

    state_kv = None
    state_vk = None
    if with_initial_state:
        state_kv = torch.randn(B, H, D, D, dtype=torch.float32, device="cuda") * 0.1
        state_vk = state_kv.transpose(-1, -2).contiguous()
    return q, k, v, g, beta, state_kv, state_vk


@cache
def case_with_reference(
    B: int,
    T: int,
    H: int,
    with_initial_state: bool,
    output_final_state: bool,
):
    inputs = make_dense_inputs(
        B,
        T,
        H,
        with_initial_state=with_initial_state,
        seed=20260803 + T * 17 + H * 101 + int(with_initial_state),
    )
    q, k, v, g, beta, state_kv, state_vk = inputs
    with torch.inference_mode():
        ref_output, ref_final_kv = recurrent_gdn_reference(
            q,
            k,
            v,
            beta,
            g,
            initial_state=state_kv,
            output_final_state=output_final_state,
        )
    return q, k, v, g, beta, state_vk, ref_output, ref_final_kv


@pytest.mark.parametrize(
    "c_mode",
    [pytest.param(mode, id=name) for mode, name in C_MODES.items()],
)
@pytest.mark.parametrize(
    ("B", "T", "H"),
    [
        pytest.param(*case, id=f"B{case[0]}-T{case[1]}-H{case[2]}")
        for case in DENSE_CASES
    ],
)
@pytest.mark.parametrize(
    ("with_initial_state", "output_final_state"),
    [
        pytest.param(False, False, id="no-state"),
        pytest.param(True, True, id="with-state"),
    ],
)
def test_opus_gdn_c_prefill_vs_reference(
    B: int,
    T: int,
    H: int,
    with_initial_state: bool,
    output_final_state: bool,
    c_mode: int,
):
    require_c_runtime()
    q, k, v, g, beta, state_vk, ref_output, ref_final_kv = case_with_reference(
        B, T, H, with_initial_state, output_final_state
    )

    with torch.inference_mode():
        output, final_vk = opus_gdn_c_prefill_fwd(
            q,
            k,
            v,
            g,
            beta,
            initial_state=state_vk,
            output_final_state=output_final_state,
            c_mode=c_mode,
        )

    assert output.shape == (B, T, H, D)
    assert output.dtype == torch.bfloat16
    assert_close(f"{C_MODES[c_mode]}_output", ref_output, output, 0.01)
    if output_final_state:
        assert final_vk is not None
        assert ref_final_kv is not None
        assert_close(
            f"{C_MODES[c_mode]}_final_state",
            ref_final_kv,
            final_vk.transpose(-1, -2).contiguous(),
            0.01,
        )
    else:
        assert final_vk is None


@pytest.mark.parametrize(
    "c_mode",
    [
        pytest.param(OPUS_GDN_C_FUSED, id="cf"),
        pytest.param(OPUS_GDN_C_SPLIT, id="cs"),
    ],
)
def test_opus_gdn_c_prefill_ignores_invalid_env_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
    c_mode: int,
):
    require_c_runtime()
    for name in (
        "OPUS_GDN_K2C_SCAN_BV",
        "OPUS_GDN_K2C_OUT_BV",
        "OPUS_GDN_OUT_VARIANT",
        "OPUS_GDN_K2C_VARIANT",
    ):
        monkeypatch.setenv(name, "999")

    q, k, v, g, beta, _state_vk, ref_output, _ = case_with_reference(
        1, 64, 4, False, False
    )
    with torch.inference_mode():
        output, final = opus_gdn_c_prefill_fwd(
            q,
            k,
            v,
            g,
            beta,
            c_mode=c_mode,
            use_env_overrides=False,
        )

    assert_close(f"{C_MODES[c_mode]}_env_disabled_output", ref_output, output, 0.01)
    assert final is None


def test_opus_gdn_c_prefill_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported c_mode=3"):
        opus_gdn_c_prefill_fwd(
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            torch.empty(0),
            c_mode=3,
        )


@pytest.mark.parametrize(
    "c_mode",
    [
        pytest.param(OPUS_GDN_C_AUTO, id="auto"),
        pytest.param(OPUS_GDN_C_FUSED, id="cf"),
        pytest.param(OPUS_GDN_C_SPLIT, id="cs"),
    ],
)
def test_opus_gdn_c_prefill_reuses_preallocated_output(c_mode: int):
    require_c_runtime()
    q, k, v, g, beta, _state_kv, _state_vk = make_dense_inputs(
        1, 64, 4, with_initial_state=False, seed=20260804 + c_mode
    )
    out = torch.empty_like(v)

    output, final = opus_gdn_c_prefill_fwd(q, k, v, g, beta, c_mode=c_mode, out=out)

    assert output.data_ptr() == out.data_ptr()
    assert final is None


def test_opus_gdn_c_prefill_rejects_unaligned_preallocated_output():
    require_c_runtime()
    q, k, v, g, beta, _state_kv, _state_vk = make_dense_inputs(
        1, 96, 4, with_initial_state=False, seed=20260808
    )
    with pytest.raises(ValueError, match="requires T to be divisible by 64"):
        opus_gdn_c_prefill_fwd(q, k, v, g, beta, out=torch.empty_like(v))


def test_opus_gdn_c_prefill_rejects_output_aliasing_v():
    require_c_runtime()
    q, k, v, g, beta, _state_kv, _state_vk = make_dense_inputs(
        1, 64, 4, with_initial_state=False, seed=20260809
    )
    with pytest.raises(ValueError, match="must not alias v storage"):
        opus_gdn_c_prefill_fwd(q, k, v, g, beta, out=v)


@pytest.mark.parametrize("invalid_kind", ("dtype", "shape", "noncontiguous"))
def test_opus_gdn_c_prefill_validates_preallocated_output(invalid_kind: str):
    require_c_runtime()
    q, k, v, g, beta, _state_kv, _state_vk = make_dense_inputs(
        1, 64, 4, with_initial_state=False, seed=20260810
    )
    if invalid_kind == "dtype":
        out = torch.empty_like(v, dtype=torch.float32)
    elif invalid_kind == "shape":
        out = torch.empty((1, 64, 4, 64), dtype=torch.bfloat16, device="cuda")
    else:
        out = torch.empty(
            (1, 64, 128, 4), dtype=torch.bfloat16, device="cuda"
        ).transpose(-1, -2)

    with pytest.raises(
        ValueError, match="contiguous bf16 tensor matching v shape/device"
    ):
        opus_gdn_c_prefill_fwd(q, k, v, g, beta, out=out)
