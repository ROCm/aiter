# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.utils._triton.arch_info import get_arch
from op_tests.triton_tests.utils.minimax_m3 import router_inputs


@pytest.mark.parametrize("tokens", [16, 128, 2048, 3072])
@pytest.mark.skipif(
    get_arch() != "gfx1250",
    reason="Requires published gfx1250 Gluon configs; strict has a separate config tree",
)
def test_router(tokens):
    data = router_inputs(tokens)
    actual = gemm_a16w16(**data, backend="gluon")
    expected = F.linear(data["x"].float(), data["w"].float()).bfloat16()
    torch.testing.assert_close(actual, expected, atol=0.125, rtol=0.02)
