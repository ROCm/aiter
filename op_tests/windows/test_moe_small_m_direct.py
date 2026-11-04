# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from op_tests.test_moe_small_m_direct import (  # noqa: F401
    pytestmark,
    test_bf16_g1u1_small_m_direct,
    test_bf16_g1u1_small_m_direct_gate_rejects_unsupported_integration,
    test_bf16_g1u1_small_m_direct_gate_shape_contract,
    test_bf16_g1u1_small_m_direct_output_buffer,
    test_bf16_g1u1_small_m_direct_plain_shuffle_metadata_contract,
)
