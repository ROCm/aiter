# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import os

import pytest
import torch

from op_tests.triton_tests.utils import minimax_m3 as m3

pytestmark = pytest.mark.skipif(
    os.getenv("AITER_TEST_MINIMAX_M3_ATOM") != "1",
    reason="Install ATOM and set AITER_TEST_MINIMAX_M3_ATOM=1",
)


@pytest.mark.parametrize("tokens", [16, 128, 2048, 3072])
def test_swiglu_oai(tokens):
    data = m3.swiglu_inputs(tokens)
    gate, up = data["gate_up"].float().chunk(2, dim=-1)
    gate = gate.clamp(max=7.0)
    expected = gate * torch.sigmoid(1.702 * gate) * (up.clamp(-7, 7) + 1)
    torch.testing.assert_close(
        m3.swiglu(**data), expected.bfloat16(), atol=0.02, rtol=0.02
    )
