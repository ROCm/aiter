# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only policy tests for the CK/OPUS BF16 BMM caller."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import torch

from aiter.ops import batched_gemm_op_bf16 as op
from aiter.ops.opus import _arch as opus_arch


_ROOT = Path(__file__).resolve().parents[1]


def _inputs():
    return (
        torch.empty((2, 3, 8), dtype=torch.bfloat16),
        torch.empty((2, 5, 8), dtype=torch.bfloat16),
    )


def _load_ck_bf16_bmm_codegen():
    """Import the script-style CK generator with its sibling on sys.path."""
    codegen_dir = _ROOT / "csrc" / "ck_batched_gemm_bf16"
    sys.path.insert(0, str(codegen_dir))
    try:
        return importlib.import_module(
            "csrc.ck_batched_gemm_bf16.gen_instances"
        )
    finally:
        sys.path.remove(str(codegen_dir))


def test_legacy_bf16_bmm_config_defaults_to_ck():
    assert op._canonical_batched_gemm_bf16_libtype(None) == "ck"
    assert op._canonical_batched_gemm_bf16_libtype(float("nan")) == "ck"
    assert op._canonical_batched_gemm_bf16_libtype(0) == "ck"
    assert op._canonical_batched_gemm_bf16_libtype("opus") == "opus"


def test_mixed_config_loader_keeps_backend_and_global_opus_kid():
    rows = pd.DataFrame(
        [
            {
                "gfx": "gfx950",
                "cu_num": 256,
                "B": 2,
                "M": 3,
                "N": 5,
                "K": 8,
                "libtype": "opus",
                "kernelId": 208,
                "splitK": 0,
                "kernelName": "opus-kernel",
            },
            {
                "gfx": "gfx950",
                "cu_num": 256,
                "B": 4,
                "M": 3,
                "N": 5,
                "K": 8,
                "kernelId": 7,
                "splitK": 0,
                "kernelName": "bf16_batched_256x64x64x32",
            },
        ]
    )
    op._clear_batched_gemm_bf16_config_caches()
    try:
        with patch.object(op.pd, "read_csv", return_value=rows):
            opus = op._get_batched_gemm_bf16_config_for_device(
                "gfx950", 256, 2, 3, 5, 8
            )
            legacy = op._get_batched_gemm_bf16_config_for_device(
                "gfx950", 256, 4, 3, 5, 8
            )
        assert opus is not None
        assert opus["libtype"] == "opus"
        assert opus["kernelId"] == 208
        assert legacy is not None
        assert legacy["libtype"] == "ck"
    finally:
        op._clear_batched_gemm_bf16_config_caches()


def test_ck_codegen_consumes_only_ck_rows_from_mixed_csv(tmp_path, monkeypatch):
    codegen = _load_ck_bf16_bmm_codegen()
    rows = pd.DataFrame(
        [
            {
                "gfx": "gfx950",
                "cu_num": 256,
                "B": 2,
                "M": 3,
                "N": 5,
                "K": 8,
                "libtype": " CK ",
                "kernelId": 7,
            },
            {
                "gfx": "gfx950",
                "cu_num": 256,
                "B": 4,
                "M": 3,
                "N": 5,
                "K": 8,
                "libtype": "opus",
                # A canonical OPUS global kid, intentionally outside the CK
                # registry.  It must be filtered before CK kernel lookup.
                "kernelId": 1206,
            },
        ]
    )
    config = tmp_path / "mixed_bf16_bmm.csv"
    rows.to_csv(config, index=False)
    monkeypatch.setenv("GPU_ARCHS", "gfx950")
    monkeypatch.setenv("CU_NUM", "256")

    tune_dict = codegen.get_tune_dict(str(config))

    ck_key = ("gfx950", 256, 2, 3, 5, 8)
    opus_key = ("gfx950", 256, 4, 3, 5, 8)
    assert tune_dict[ck_key] is codegen.kernels_list[7]
    assert opus_key not in tune_dict


def test_ck_codegen_treats_legacy_csv_without_libtype_as_ck(
    tmp_path, monkeypatch
):
    codegen = _load_ck_bf16_bmm_codegen()
    rows = pd.DataFrame(
        [
            {
                "gfx": "gfx950",
                "cu_num": 256,
                "B": 3,
                "M": 4,
                "N": 6,
                "K": 8,
                "kernelId": 5,
            }
        ]
    )
    config = tmp_path / "legacy_bf16_bmm.csv"
    rows.to_csv(config, index=False)
    monkeypatch.setenv("GPU_ARCHS", "gfx950")
    monkeypatch.setenv("CU_NUM", "256")

    tune_dict = codegen.get_tune_dict(str(config))

    key = ("gfx950", 256, 3, 4, 6, 8)
    assert tune_dict[key] is codegen.kernels_list[5]


def test_explicit_opus_bmm_uses_resolved_exact_kid():
    x, w = _inputs()
    output = object()
    plan = SimpleNamespace(resolved_kid=1200)
    with (
        patch.object(opus_arch, "_device_arch_and_cu", return_value=("gfx950", 256)),
        patch.object(op, "_resolve_opus_bf16_bmm_candidate", return_value=plan) as resolve,
        patch.object(op, "_launch_batched_gemm_bf16_opus", return_value=output) as launch,
    ):
        actual = op.batched_gemm_bf16_OPUS(x, w, kid=1206, splitK=3)

    assert actual is output
    assert resolve.call_args.kwargs["B"] == 2
    assert resolve.call_args.kwargs["kid"] == 1206
    assert resolve.call_args.kwargs["split_k"] == 3
    assert launch.call_args.kwargs == {"kid": 1200, "split_k": 3}


def test_tuned_dispatch_selects_opus_row():
    x, w = _inputs()
    output = object()
    config = {"libtype": "opus", "kernelId": 206, "splitK": 2}
    plan = SimpleNamespace(resolved_kid=206)
    with (
        patch.object(opus_arch, "_device_arch_and_cu", return_value=("gfx950", 256)),
        patch.object(
            op,
            "_get_batched_gemm_bf16_config_for_device",
            return_value=config,
        ),
        patch.object(op, "_resolve_opus_bf16_bmm_candidate", return_value=plan),
        patch.object(op, "_launch_batched_gemm_bf16_opus", return_value=output) as launch,
        patch.object(op, "batched_gemm_bf16_CK") as ck,
    ):
        actual = op.batched_gemm_bf16_tuned(x, w)

    assert actual is output
    launch.assert_called_once()
    ck.assert_not_called()


def test_tuned_dispatch_falls_back_to_ck_for_invalid_opus_row():
    x, w = _inputs()
    output = object()
    config = {"libtype": "opus", "kernelId": 999999, "splitK": 0}
    with (
        patch.object(opus_arch, "_device_arch_and_cu", return_value=("gfx950", 256)),
        patch.object(
            op,
            "_get_batched_gemm_bf16_config_for_device",
            return_value=config,
        ),
        patch.object(op, "_resolve_opus_bf16_bmm_candidate", return_value=None),
        patch.object(op, "batched_gemm_bf16_CK", return_value=output) as ck,
    ):
        actual = op.batched_gemm_bf16_tuned(x, w)

    assert actual is output
    ck.assert_called_once()
