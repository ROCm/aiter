# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest

# `op_tests.test_moe_small_m_direct` imports `aiter.fused_moe`, which imports
# FlyDSL unconditionally -- 39 names across three `aiter.ops.flydsl` modules, so
# the dependency is structural rather than a corner of the file. FlyDSL ships
# Linux-only wheels and `requirements.txt` says so with
# `flydsl==0.4.2; sys_platform != "win32"`, so on Windows the import raises
# `ModuleNotFoundError: No module named 'flydsl'` while pytest is COLLECTING
# this module. A collection error is not a skip: it fails the whole smoke job,
# and takes the other Windows tests in the run down with it.
#
# Skipping here keeps that failure mode out of a platform where the dependency
# is deliberately absent. It costs no coverage that Windows had: this module is
# a re-export shim, and `aiter.fused_moe` has never been importable there.
pytest.importorskip(
    "flydsl", reason="FlyDSL ships Linux-only wheels; see requirements.txt"
)

from op_tests.test_moe_small_m_direct import (  # noqa: F401
    pytestmark,
    test_bf16_g1u1_small_m_direct,
    test_bf16_g1u1_small_m_direct_gate_rejects_unsupported_integration,
    test_bf16_g1u1_small_m_direct_gate_shape_contract,
    test_bf16_g1u1_small_m_direct_output_buffer,
    test_bf16_g1u1_small_m_direct_plain_shuffle_metadata_contract,
    test_bf16_g1u1_small_m_raw_entrypoint_rejects_strided_tensors,
)
