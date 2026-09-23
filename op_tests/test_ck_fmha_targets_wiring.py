# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Behavior tests for CK fmha --targets wiring and RDNA gates."""

from unittest.mock import patch

import pytest
import torch

from aiter import dtypes


def _dummy_qkv(dtype):
    q = torch.empty(1, 8, 2, 64, dtype=dtype)
    k = torch.empty(1, 8, 2, 64, dtype=dtype)
    v = torch.empty(1, 8, 2, 64, dtype=dtype)
    return q, k, v


def test_core_exports_helpers_for_json_eval():
    from aiter.jit import core as jit_core

    assert callable(jit_core.ck_fmha_targets)
    assert callable(jit_core.ck_fmha_batch_prefill_gen_targets)


@pytest.mark.parametrize(
    "arch, should_raise",
    [
        ("gfx1100", True),
        ("gfx1151", True),
        ("gfx1201", False),
        ("gfx942", False),
    ],
)
def test_fp8_cmdGenFunc_gate(arch, should_raise):
    from aiter.ops.mha import cmdGenFunc_mha_fwd

    q, k, v = _dummy_qkv(dtypes.fp8)
    gfx_patch = patch("aiter.ops.mha.get_gfx", return_value=arch)
    if should_raise:
        with gfx_patch, pytest.raises(NotImplementedError, match=arch) as exc:
            cmdGenFunc_mha_fwd(q, k, v, 0.0, 0.1, False, -1, -1, 0, False, False)
        assert "fp8" in str(exc.value).lower()
        return
    key = {"gfx1201": "gfx12", "gfx942": "gfx9"}[arch]
    with gfx_patch, patch("aiter.ops.mha.ck_fmha_targets", lambda: key):
        out = cmdGenFunc_mha_fwd(q, k, v, 0.0, 0.1, False, -1, -1, 0, False, False)
        assert "blob_gen_cmd" in out


def test_fp8_out_cmdGenFunc_not_implemented():
    from aiter.ops.mha import cmdGenFunc_mha_fwd

    q, k, v = _dummy_qkv(dtypes.fp8)
    out = torch.empty_like(q)
    with (
        patch("aiter.ops.mha.get_gfx", return_value="gfx1201"),
        pytest.raises(NotImplementedError, match="Unsupported output dtype"),
    ):
        cmdGenFunc_mha_fwd(q, k, v, 0.0, 0.1, False, -1, -1, 0, False, False, out=out)


def test_fp8_api_gate_uses_runtime_gfx():
    from aiter.ops.mha import flash_attn_fp8_pertensor_func

    q, k, v = _dummy_qkv(dtypes.fp8)
    scale = torch.ones(1)
    with (
        patch("aiter.ops.mha.ENABLE_CK", True),
        patch("aiter.ops.mha.get_gfx_runtime", return_value="gfx1100"),
        pytest.raises(NotImplementedError, match="no fp8 factory"),
    ):
        flash_attn_fp8_pertensor_func(q, k, v, scale, scale, scale)


def test_fp8_api_uses_runtime_gfx():
    """GPU_ARCHS last-token must not reject live gfx1201 fp8."""
    from aiter.ops.mha import flash_attn_fp8_pertensor_func

    q, k, v = _dummy_qkv(dtypes.fp8)
    scale = torch.ones(1)
    with (
        patch("aiter.ops.mha.ENABLE_CK", True),
        patch("aiter.ops.mha.get_gfx", return_value="gfx1100"),
        patch("aiter.ops.mha.get_gfx_runtime", return_value="gfx1201"),
    ):
        # Must not raise the CK fp8 factory gate. Downstream CK/JIT may
        # still fail in this mocked environment; only that gate is forbidden.
        try:
            flash_attn_fp8_pertensor_func(q, k, v, scale, scale, scale)
        except NotImplementedError as e:
            msg = str(e)
            if "no fp8 factory" in msg.lower() or "gfx1100" in msg:
                pytest.fail(f"runtime gfx1201 fp8 was rejected: {e}")
        except Exception:  # noqa: S110, BLE001
            pass


def test_fp8_api_gate_does_not_preempt_triton():
    """ENABLE_CK=0 on gfx11 must still reach Triton, not hit the CK gate (I3).

    The gate sits *after* the Triton branch, so patching ENABLE_CK=0 must
    return the Triton result. A bare try/except here would pass vacuously on
    any unrelated exception, so assert the Triton entry point was actually
    called. flash_attn_func is imported inside the function body, so patch the
    source module attribute, not aiter.ops.mha.
    """
    from aiter.ops.mha import flash_attn_fp8_pertensor_func

    q, k, v = _dummy_qkv(dtypes.fp8)
    scale = torch.ones(1)
    sentinel = object()
    with (
        patch("aiter.ops.mha.ENABLE_CK", False),
        patch("aiter.ops.mha.get_gfx_runtime", return_value="gfx1100"),
        patch(
            "aiter.ops.triton.attention.mha_v3.flash_attn_func",
            return_value=sentinel,
        ) as tri,
    ):
        assert flash_attn_fp8_pertensor_func(q, k, v, scale, scale, scale) is sentinel
        assert tri.called


def test_batch_prefill_cmdGenFunc_uses_gen_targets():
    """batch_prefill must wire ck_fmha_batch_prefill_gen_targets, not the
    generic helper. If it used ck_fmha_targets(), a GPU_ARCHS=gfx942;gfx1100
    build emits --targets gfx9,gfx11 and CK's has_non_gfx9 drops *every*
    batch_prefill kernel — a silent link/runtime failure.
    """
    from aiter.ops.mha import cmdGenFunc_mha_batch_prefill

    q = torch.empty(8, 2, 64, dtype=torch.float16)
    k = torch.empty(4, 16, 2, 64, dtype=torch.float16)
    v = torch.empty(4, 16, 2, 64, dtype=torch.float16)
    cu = torch.zeros(2, dtype=torch.int32)
    with (
        patch("aiter.ops.mha.ck_fmha_batch_prefill_gen_targets", lambda: "gfx9"),
        patch("aiter.ops.mha.ck_fmha_targets", lambda: "SHOULD-NOT-BE-USED"),
        patch("aiter.ops.mha.get_gfx", lambda: "gfx942"),
    ):
        cmd = cmdGenFunc_mha_batch_prefill(
            q,
            k,
            v,
            cu,
            cu,
            cu,
            8,
            8,
            0.0,
            0.1,
            0.0,
            False,
            False,
            -1,
            -1,
            0,
            False,
            False,
        )["blob_gen_cmd"][0]
        assert "--targets gfx9 " in cmd
        assert "SHOULD-NOT-BE-USED" not in cmd


def test_batch_prefill_cmdGenFunc_uses_build_gfx():
    from aiter.ops.mha import cmdGenFunc_mha_batch_prefill

    q = torch.empty(8, 2, 64, dtype=torch.float16)
    k = torch.empty(4, 16, 2, 64, dtype=torch.float16)
    v = torch.empty(4, 16, 2, 64, dtype=torch.float16)
    cu = torch.zeros(2, dtype=torch.int32)
    with (
        patch("aiter.ops.mha.get_gfx", return_value="gfx1100"),
        pytest.raises(NotImplementedError, match="gfx9-only"),
    ):
        cmdGenFunc_mha_batch_prefill(
            q,
            k,
            v,
            cu,
            cu,
            cu,
            8,
            8,
            0.0,
            0.1,
            0.0,
            False,
            False,
            -1,
            -1,
            0,
            False,
            False,
        )


def test_batch_prefill_api_uses_runtime_gfx():
    """GPU_ARCHS last-token must not reject a live gfx942 (B2)."""
    from aiter.ops.mha import mha_batch_prefill_func

    with (
        patch("aiter.ops.mha.get_gfx", return_value="gfx1100"),
        patch("aiter.ops.mha.get_gfx_runtime", return_value="gfx942"),
        patch("aiter.ops.mha.ck_fmha_targets", lambda: "gfx9"),
    ):
        # Must not raise NotImplementedError from the RDNA gate. Downstream
        # CK/JIT may still fail in this mocked environment; only the gate
        # message is forbidden.
        try:
            mha_batch_prefill_func(
                torch.empty(8, 2, 64),
                torch.empty(4, 16, 2, 64),
                torch.empty(4, 16, 2, 64),
                torch.zeros(2, dtype=torch.int32),
                torch.zeros(2, dtype=torch.int32),
                torch.zeros(4, dtype=torch.int32),
                8,
                8,
            )
        except NotImplementedError as e:
            pytest.fail(f"runtime gfx942 was rejected: {e}")
        except Exception:  # noqa: S110, BLE001
            pass


def test_batch_prefill_api_rejects_live_rdna():
    from aiter.ops.mha import mha_batch_prefill_func

    with (
        patch("aiter.ops.mha.get_gfx_runtime", return_value="gfx1100"),
        pytest.raises(NotImplementedError, match="gfx9-only"),
    ):
        mha_batch_prefill_func(
            torch.empty(8, 2, 64),
            torch.empty(4, 16, 2, 64),
            torch.empty(4, 16, 2, 64),
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int32),
            torch.zeros(4, dtype=torch.int32),
            8,
            8,
        )


def test_get_args_of_build_all_keeps_fmha_when_unmapped():
    """B3-A: cpu still emits FMHA modules (CK default gfx9,gfx950)."""
    from aiter.jit import core as jit_core

    with patch("chip_info.get_gfx_list", return_value=["cpu"]):
        all_ops, _ = jit_core.get_args_of_build("all")
        names = {op["md_name"] for op in all_ops}
        assert "module_mha_fwd" in names
        assert "module_activation" in names
