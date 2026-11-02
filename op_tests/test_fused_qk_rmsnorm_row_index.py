# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_qk_rmsnorm_group_quant's q_row_index: q's rows gathered by an index
(an embedding lookup fused in front of the norm) give bit for bit what the
gather followed by the norm gives; an index outside q reads as zeros."""

import pytest
import torch

from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant


def _norm(q, w, out, **kw):
    fused_qk_rmsnorm_group_quant(q=q, q_weight=w, q_epsilon=1e-6, q_out_unquantized=out, **kw)


@pytest.mark.parametrize("rows", [1, 33, 1024, 1025, 4099])
@pytest.mark.parametrize("n", [512, 2048, 4096])
@pytest.mark.parametrize("gemma", [False, True])
@pytest.mark.parametrize("dual", [False, True])
def test_row_index_is_the_gather_then_the_norm(rows, n, gemma, dual):
    g = torch.Generator(device="cuda").manual_seed(rows * n)
    table = (torch.randn(3000, n, device="cuda", generator=g) * 2).to(torch.bfloat16)
    w = torch.randn(n, device="cuda", generator=g).to(torch.bfloat16)
    ids = torch.randint(0, 3000, (rows,), device="cuda", generator=g, dtype=torch.int32)
    k = torch.randn(rows, n, device="cuda", generator=g).to(torch.bfloat16)
    wk = torch.randn(n, device="cuda", generator=g).to(torch.bfloat16)
    extra = {"gemma_norm": gemma}
    ref, got = torch.empty(rows, n, dtype=table.dtype, device="cuda"), torch.empty_like(k)
    ref_k, got_k = torch.empty_like(k), torch.empty_like(k)
    if dual:
        _norm(table[ids.long()], w, ref, k=k, k_weight=wk, k_out=ref_k, **extra)
        _norm(table, w, got, k=k, k_weight=wk, k_out=got_k, q_row_index=ids, **extra)
        assert torch.equal(ref_k.view(torch.int16), got_k.view(torch.int16))
    else:
        _norm(table[ids.long()], w, ref, **extra)
        _norm(table, w, got, q_row_index=ids, **extra)
    assert torch.equal(ref.view(torch.int16), got.view(torch.int16))


def test_an_index_outside_the_table_reads_zeros():
    table = torch.randn(16, 2048, device="cuda").to(torch.bfloat16)
    w = torch.randn(2048, device="cuda").to(torch.bfloat16)
    ids = torch.tensor([-1, 5, 16], device="cuda", dtype=torch.int32)
    out = torch.empty(3, 2048, dtype=table.dtype, device="cuda")
    _norm(table, w, out, q_row_index=ids, gemma_norm=True)
    ref = torch.empty(1, 2048, dtype=table.dtype, device="cuda")
    _norm(table[5:6], w, ref, gemma_norm=True)
    assert out[0].abs().max().item() == 0 and out[2].abs().max().item() == 0
    assert torch.equal(out[1:2].view(torch.int16), ref.view(torch.int16))
