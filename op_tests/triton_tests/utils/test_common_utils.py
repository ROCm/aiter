# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton.utils.common_utils import max_addressable_bytes


def test_max_addressable_bytes():
    def meta(*shape):
        return torch.empty(*shape, dtype=torch.uint8, device="meta")

    c = meta(47233, 64, 584)
    assert max_addressable_bytes(c) == c.nelement() < 2**31 - 1

    v = meta(1435968 * 47232 + 64 * 584).as_strided((47233, 64, 584), (1435968, 584, 1))
    assert v.nelement() == c.nelement()
    assert max_addressable_bytes(v) == 67823677952 >= 2**31 - 1
