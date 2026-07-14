# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit test for the fused RMSNorm + per_1x128 FP8 quant path that emits an
e8m0 (1-byte) block scale, added in PR #4231.

The e8m0 output path is selected purely by the scale tensor's element_size()==1
(host side), with group_size=128 using the plain col-major transpose layout
(shuffle) / row-major layout (non-shuffle) to feed the gfx1250 a8w8 blockscale
bpreshuffle GEMM. This mirrors ATOM's q_norm -> wq_b / indexer.wq_b usage.

Run:
    pytest op_tests/test_rmsnorm_blockscale_e8m0.py -v
or standalone:
    python op_tests/test_rmsnorm_blockscale_e8m0.py
"""

import pytest
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose

torch.set_default_device("cuda")

_GROUP = 128


def _ref_rmsnorm(x, weight, eps, residual=None):
    if residual is not None:
        residual_out = x + residual
    else:
        residual_out = None
    normed = F.rms_norm(
        input=residual_out if residual is not None else x,
        normalized_shape=(x.shape[-1],),
        weight=weight,
        eps=eps,
    )
    return normed, residual_out


def _alloc_e8m0_scale(M, num_groups, shuffle):
    # Match ATOM's layernorm allocation: preshuffle (shuffle) GEMM consumes a
    # col-major (num_groups, M) scale viewed as (M, num_groups); non-shuffle is
    # plain row-major (M, num_groups). e8m0 = 1 byte/element -> opts into the
    # kernel's e8m0 path.
    if shuffle:
        storage = torch.empty(
            (num_groups, M), dtype=torch.float8_e8m0fnu, device="cuda"
        )
        return storage.view(M, num_groups), storage
    storage = torch.empty((M, num_groups), dtype=torch.float8_e8m0fnu, device="cuda")
    return storage, storage


def _dequant(out_fp8, storage, num_groups, shuffle):
    of = out_fp8.to(torch.float32)
    sc = storage.to(torch.float32)  # power-of-2 values
    if shuffle:
        # storage is (num_groups, M) -> (M, num_groups)
        per_group = sc.t()
    else:
        per_group = sc  # (M, num_groups)
    per_elem = per_group.repeat_interleave(_GROUP, dim=1)  # (M, N)
    return of * per_elem


@pytest.mark.parametrize("m", [1, 7, 128, 4096])
@pytest.mark.parametrize("n", [1024, 4096])  # 1024 = V4-Flash q_lora_rank
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("add_residual", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_rmsnorm_blockscale_e8m0(m, n, shuffle, add_residual, dtype):
    assert n % _GROUP == 0
    num_groups = n // _GROUP
    eps = 1e-5

    x = torch.randn((m, n), dtype=dtype, device="cuda")
    weight = torch.randn((n,), dtype=dtype, device="cuda")
    residual = torch.randn((m, n), dtype=dtype, device="cuda") if add_residual else None

    out = torch.empty((m, n), dtype=dtypes.fp8, device="cuda")
    scale, storage = _alloc_e8m0_scale(m, num_groups, shuffle)

    # This call coredumped before the PR #4231 kernel fix.
    if add_residual:
        residual_out = torch.empty_like(x)
        aiter.add_rmsnorm_quant(
            out, x, residual, residual_out, scale, weight, eps, _GROUP, shuffle
        )
    else:
        residual_out = None
        aiter.rmsnorm_quant(out, x, scale, weight, eps, _GROUP, shuffle)

    torch.cuda.synchronize()  # surface any async device fault here

    ref_norm, ref_res = _ref_rmsnorm(x, weight, eps, residual)
    deq = _dequant(out, storage, num_groups, shuffle)

    # e8m0 block scale (power-of-2) + fp8 e4m3 (3-bit mantissa) -> coarse; use a
    # loose relative tolerance, just guard against garbage/NaN.
    assert torch.isfinite(deq).all(), "dequantized output has non-finite values"
    checkAllclose(
        ref_norm.to(torch.float32),
        deq,
        rtol=0.15,
        atol=0.15,
        msg=f"e8m0 blockscale rmsnorm out (m={m},n={n},shuffle={shuffle},add={add_residual})",
    )
    if add_residual:
        checkAllclose(
            ref_res.to(torch.float32),
            residual_out.to(torch.float32),
            rtol=1e-2,
            atol=1e-2,
            msg="residual_out",
        )


if __name__ == "__main__":
    for add_residual in (False, True):
        for shuffle in (True, False):
            for n in (1024, 4096):
                for m in (1, 7, 128, 4096):
                    print(
                        f"--- m={m} n={n} shuffle={shuffle} add_residual={add_residual} ---"
                    )
                    test_rmsnorm_blockscale_e8m0(
                        m, n, shuffle, add_residual, torch.bfloat16
                    )
    print("all e8m0 blockscale rmsnorm cases passed")
