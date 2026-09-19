# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter.ops.triton._triton_kernels.gated_delta_rule.utils.l2norm import (
    l2norm_fwd,
)


@pytest.mark.parametrize("tokens", [17, 256])
def test_l2norm_accepts_strided_leading_dimensions(tokens: int):
    torch.manual_seed(tokens)
    head_dim = 128
    x = torch.randn(
        tokens,
        16,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
    ).transpose(0, 1)
    assert x.stride(-1) == 1
    assert not x.is_contiguous()

    actual, rstd = l2norm_fwd(x)
    expected = x.float() * torch.rsqrt(
        torch.sum(x.float().square(), dim=-1, keepdim=True) + 1e-6
    )

    assert rstd is None
    assert actual.shape == x.shape
    torch.testing.assert_close(
        actual.float(),
        expected,
        rtol=2e-2,
        atol=2e-3,
    )
