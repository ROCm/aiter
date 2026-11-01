# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha import fmha_fwd_bf16_opus_fwd, mha_fwd_native_splitkv

# The old -1e30 maximum seed clips this finite BF16 dot product.
_TARGET_LOGIT = -(2.0**101)


def _extreme_inputs(head_dim_qk, head_dim_v):
    seqlen = 64
    shape_qk = (1, seqlen, 1, head_dim_qk)
    q = torch.ones(shape_qk, dtype=dtypes.bf16, device="cuda")
    k = torch.full(
        shape_qk,
        _TARGET_LOGIT / head_dim_qk,
        dtype=dtypes.bf16,
        device="cuda",
    )
    v = torch.ones((1, seqlen, 1, head_dim_v), dtype=dtypes.bf16, device="cuda")
    return q, k, v


def _assert_unit_output(out):
    # Softmax weights sum to one, so constant V must pass through exactly.
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)


def test_mha_native_extreme_negative_logits():
    if get_gfx() != "gfx942":
        pytest.skip("native D64 regression runs on gfx942")

    q, k, v = _extreme_inputs(64, 64)
    out, lse = mha_fwd_native_splitkv(
        q,
        k,
        v,
        out=None,
        softmax_scale=64**-0.5,
        causal=False,
        return_lse=True,
        num_splits=1,
    )

    _assert_unit_output(out)
    assert torch.isfinite(lse).all()


@pytest.mark.parametrize("head_dim_qk", [128, 192])
def test_fmha_opus_extreme_negative_logits(head_dim_qk):
    if get_gfx() != "gfx950":
        pytest.skip("OPUS regression requires gfx950")

    q, k, v = _extreme_inputs(head_dim_qk, 128)
    out = fmha_fwd_bf16_opus_fwd(
        q,
        k,
        v,
        softmax_scale=head_dim_qk**-0.5,
        causal=False,
    )

    _assert_unit_output(out)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
