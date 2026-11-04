# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch

from aiter.ops.triton.fusions.fused_rmsnorm_add import fused_rmsnorm_add
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.gemm.fused.fused_rmsnorm_gemm_a16w16 import (
    _get_config,
    fused_rmsnorm_gemm_a16w16,
    fused_rmsnorm_gemm_a16w16_supported,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(get_arch() != "gfx1250", reason="gfx1250 only")

H = 2048
EPS = 1e-6
V = 4096  # rows of the gathered table


def _aiter_norm(x, w, row_index=None):
    """aiter's Gemma norm (fused_qk_rmsnorm_group_quant, no quant), the
    reference of norm_mode="aiter"."""
    from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant

    rows = x.shape[0] if row_index is None else row_index.shape[0]
    out = torch.empty((rows, H), dtype=x.dtype, device=x.device)
    fused_qk_rmsnorm_group_quant(
        q=x,
        q_weight=w,
        q_epsilon=EPS,
        q_out_unquantized=out,
        gemma_norm=True,
        q_row_index=row_index,
    )
    return out


def _check(y, y_ref, bitwise):
    if bitwise:
        differ = (y != y_ref).sum().item()
        assert differ == 0, f"not bitwise: {differ} of {y.numel()} differ"
    else:
        # Only the fp32 summation order differs: at most one bf16 ulp.
        torch.testing.assert_close(y, y_ref, atol=2e-2, rtol=8e-3)


def _bitwise(M, N, K):
    config, _ = _get_config(M, N, K)
    return not config.get("K_SPLIT", True)


@pytest.mark.parametrize("M", [1, 13, 16, 32, 80])
@pytest.mark.parametrize("normed_table", [False, True])
def test_two_segments_gather(M, normed_table):
    """The drafter's fc: cat(norm_e(embed[ids]), norm_h(hidden)) @ W^T."""
    torch.manual_seed(0)
    N = 2048
    table = torch.randn((V, H), dtype=torch.bfloat16, device="cuda")
    hidden = torch.randn((M, H), dtype=torch.bfloat16, device="cuda") * 3
    w = torch.randn((N, 2 * H), dtype=torch.bfloat16, device="cuda") * 0.02
    we = torch.randn((H,), dtype=torch.bfloat16, device="cuda") * 0.1
    wh = torch.randn((H,), dtype=torch.bfloat16, device="cuda") * 0.1
    ids = torch.randint(0, V, (M,), dtype=torch.int32, device="cuda")
    if M > 1:
        ids[0] = V + 5  # out of range: a zero row
    assert fused_rmsnorm_gemm_a16w16_supported(M, N, H, 2, torch.bfloat16)

    a_ref = torch.cat([_aiter_norm(table, we, ids), _aiter_norm(hidden, wh)], dim=1)
    y_ref = gemm_a16w16(a_ref, w)
    if normed_table:
        y = fused_rmsnorm_gemm_a16w16(
            _aiter_norm(table, we), w, None, None, hidden, wh, EPS, row_index=ids
        )
    else:
        y = fused_rmsnorm_gemm_a16w16(table, w, we, EPS, hidden, wh, EPS, row_index=ids)
    _check(y, y_ref, _bitwise(M, N, 2 * H))


@pytest.mark.parametrize("M", [1, 13, 16, 32, 80])
def test_one_segment_gluon_norm(M):
    """The drafter's qkv_proj: input_layernorm (fused_rmsnorm_add) @ W^T."""
    torch.manual_seed(0)
    N = 2560
    x = torch.randn((M, H), dtype=torch.bfloat16, device="cuda") * 2
    w = torch.randn((N, H), dtype=torch.bfloat16, device="cuda") * 0.02
    nw = torch.randn((H,), dtype=torch.bfloat16, device="cuda") * 0.1
    y_ref = gemm_a16w16(fused_rmsnorm_add(x, nw, EPS, gemma_norm=True), w)
    y = fused_rmsnorm_gemm_a16w16(x, w, nw, EPS, norm_mode="gluon")
    _check(y, y_ref, _bitwise(M, N, H))
