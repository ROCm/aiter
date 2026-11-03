# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only tests for the #7150 deprecation aliases (batch 1: MoE)."""

from __future__ import annotations

import warnings

import pytest

import aiter
from aiter import fused_moe_bf16_asm
from aiter._deprecation import deprecated_alias


def test_deprecated_alias_warns_once_per_call_site_and_delegates():
    calls = []

    def target(x):
        calls.append(x)
        return x + 1

    alias = deprecated_alias(target, "foo_fwd", "aiter.foo")

    with pytest.warns(DeprecationWarning, match="foo_fwd"):
        assert alias(1) == 2
    # The default filter action reports each call site once: two calls from
    # the same line produce one record, and delegation is unchanged.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        for _ in range(2):
            assert alias(2) == 3
    assert len(caught) == 1
    assert calls == [1, 2, 2]
    assert alias.__wrapped__ is target


def test_fused_moe_bf16_asm_alias_module_delegates_and_warns_once():
    # The alias warns once per process on first attribute access; reset the
    # flag in case an earlier test in this process already touched the alias.
    fused_moe_bf16_asm._warned = False

    with pytest.warns(DeprecationWarning, match="fused_moe_bf16_asm"):
        asm_moe = fused_moe_bf16_asm.asm_moe
    assert asm_moe is aiter.fused_moe_bf16.asm_moe
    # Second access from the same process is silent.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert fused_moe_bf16_asm.asm_moe is asm_moe


def test_moe_sorting_opus_public_names_are_deprecated_aliases():
    # The public backend-suffixed names delegate to the backend module and
    # carry the deprecation warning; the neutral dispatcher is unchanged.
    backend = aiter.ops.moe_sorting_opus
    assert aiter.moe_sorting_opus_fwd.__wrapped__ is backend.moe_sorting_opus_fwd
    assert (
        aiter.moe_sorting_opus_get_workspace_size.__wrapped__
        is backend.moe_sorting_opus_get_workspace_size
    )
    # mxfp4_moe_sort_quant_fwd is a neutral name and stays a direct binding.
    assert aiter.mxfp4_moe_sort_quant_fwd is backend.mxfp4_moe_sort_quant_fwd
