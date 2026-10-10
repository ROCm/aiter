# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from unittest.mock import Mock

import pytest
import torch

from aiter.ops import gemm_op_a8w8 as ops
from aiter.ops.triton.gemm.basic import gemm_a8w8 as triton_ops
from aiter.utility import dtypes


@pytest.mark.parametrize("arch", ["gfx1201", "gfx1200", "gfx942", "gfx950"])
def test_bpreshuffle_dispatch_is_gfx1201_only(monkeypatch, arch):
    # CPU tensors: the test must select a backend without launching any kernel.
    x = torch.empty((4, 96), dtype=dtypes.fp8)
    w = torch.empty((64, 128), dtype=dtypes.fp8)
    sx, sw = torch.ones(4, 1), torch.ones(64, 1)
    expected = torch.empty((4, 64), dtype=torch.bfloat16)
    triton = Mock(return_value=expected)
    ck = Mock(return_value=expected)
    config = Mock(return_value={"libtype": "ck", "splitK": 0})
    monkeypatch.setattr(ops, "get_gfx", lambda: arch)
    monkeypatch.setattr(triton_ops, "gemm_a8w8", triton)
    monkeypatch.setattr(ops, "get_GEMM_config_with_quant_type", config)
    monkeypatch.setattr(ops, "gemm_a8w8_bpreshuffle_ck", ck)

    assert ops.gemm_a8w8_bpreshuffle(x, w, sx, sw) is expected
    if arch == "gfx1201":
        triton.assert_called_once_with(
            x, w, sx, sw, dtype=torch.bfloat16, b_preshuffled=True
        )
        ck.assert_not_called()
        config.assert_not_called()
    else:
        triton.assert_not_called()
        ck.assert_called_once()
        config.assert_called_once()


def test_bpreshuffle_keeps_existing_bias_contract(monkeypatch):
    monkeypatch.setattr(ops, "get_gfx", lambda: "gfx1201")
    x = torch.empty((1, 32), dtype=dtypes.fp8)
    w = torch.empty((16, 32), dtype=dtypes.fp8)
    with pytest.raises(AssertionError, match="does not support bias"):
        ops.gemm_a8w8_bpreshuffle(x, w, torch.ones(1), torch.ones(16), torch.ones(16))


def test_shuffle_address_mapping():
    # Exact bytes, independent of GPU arithmetic and FP8 tolerances. Include
    # multiple N tiles and K tiles so a wrong permutation cannot pass by chance.
    from aiter.ops.shuffle import shuffle_weight

    n, k = 48, 160
    torch.manual_seed(17)
    original = torch.randint(0, 256, (n, k), dtype=torch.uint8)
    shuffled = shuffle_weight(original).flatten()
    ns, ks = torch.arange(n)[:, None], torch.arange(k)[None, :]
    offsets = (ns // 16) * k * 16 + (ks // 16) * 256 + (ns % 16) * 16 + ks % 16
    torch.testing.assert_close(shuffled[offsets], original)
