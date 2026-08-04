# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

import aiter
from aiter.fused_moe import _maybe_force_atomic_flydsl_stage2

REDUCE_KERNEL = "flydsl_moe2_afp4_wfp4_bf16_t128x256x128_reduce"
ATOMIC_KERNEL = "flydsl_moe2_afp4_wfp4_bf16_t128x256x128_atomic"


def test_force_atomic_stage2_uses_available_variant(monkeypatch):
    monkeypatch.setenv("AITER_FLYDSL_FORCE_ATOMIC", "1")
    monkeypatch.delenv("AITER_FLYDSL_FORCE_REDUCE", raising=False)
    monkeypatch.setattr(
        aiter.ops.flydsl.moe_kernels,
        "get_flydsl_kernel_params",
        lambda name: {"mode": "atomic"} if name == ATOMIC_KERNEL else None,
    )

    assert _maybe_force_atomic_flydsl_stage2(REDUCE_KERNEL) == ATOMIC_KERNEL


def test_force_atomic_stage2_keeps_reduce_when_variant_is_unavailable(monkeypatch):
    monkeypatch.setenv("AITER_FLYDSL_FORCE_ATOMIC", "1")
    monkeypatch.delenv("AITER_FLYDSL_FORCE_REDUCE", raising=False)
    monkeypatch.setattr(
        aiter.ops.flydsl.moe_kernels,
        "get_flydsl_kernel_params",
        lambda _name: None,
    )

    assert _maybe_force_atomic_flydsl_stage2(REDUCE_KERNEL) == REDUCE_KERNEL


@pytest.mark.parametrize(
    "kernel_name",
    [
        "",
        "flydsl_moe2_afp4_wfp4_bf16_t128x256x128_atomic",
        "cktile_moe2_afp4_wfp4_bf16_reduce",
    ],
)
def test_force_atomic_stage2_ignores_ineligible_kernels(monkeypatch, kernel_name):
    monkeypatch.setenv("AITER_FLYDSL_FORCE_ATOMIC", "1")
    monkeypatch.delenv("AITER_FLYDSL_FORCE_REDUCE", raising=False)
    monkeypatch.setattr(
        aiter.ops.flydsl.moe_kernels,
        "get_flydsl_kernel_params",
        lambda _name: pytest.fail("ineligible kernels must not be looked up"),
    )

    assert _maybe_force_atomic_flydsl_stage2(kernel_name) == kernel_name


def test_force_reduce_takes_precedence(monkeypatch):
    monkeypatch.setenv("AITER_FLYDSL_FORCE_ATOMIC", "1")
    monkeypatch.setenv("AITER_FLYDSL_FORCE_REDUCE", "1")
    monkeypatch.setattr(
        aiter.ops.flydsl.moe_kernels,
        "get_flydsl_kernel_params",
        lambda _name: pytest.fail("force-reduce must prevent atomic lookup"),
    )

    assert _maybe_force_atomic_flydsl_stage2(REDUCE_KERNEL) == REDUCE_KERNEL
