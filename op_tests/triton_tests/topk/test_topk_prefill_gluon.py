# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import pytest
import torch

from aiter.ops.triton.topk_prefill import top_k_per_row_prefill
from aiter.ops.triton.utils._triton.arch_info import get_arch

K = 2048
pytestmark = pytest.mark.skipif(get_arch() != "gfx1250", reason="gfx1250 only")


def _logits(dist, rows, width, g):
    if dist == "random":
        return torch.randn((rows, width), device="cuda", generator=g)
    if dist == "relu_sparse":
        return torch.relu(torch.randn((rows, width), device="cuda", generator=g) - 2.0)
    if dist == "few_values":
        return torch.randint(0, 16, (rows, width), device="cuda", generator=g).float()
    if dist == "all_equal":
        return torch.zeros((rows, width), device="cuda")
    x = torch.randn((rows, width), device="cuda", generator=g)
    x[::3] = float("nan")
    x[1::3, ::7] = float("nan")
    return x


def _ranges(layout, rows, ctx):
    if layout == "causal":
        ks = torch.zeros(rows, dtype=torch.int32)
        ke = torch.arange(ctx - rows + 1, ctx + 1, dtype=torch.int32)
        return ctx, ks.cuda(), ke.cuda()
    j, i = torch.arange(rows) // ctx, torch.arange(rows) % ctx
    return rows, (j * ctx).int().cuda(), (j * ctx + i + 1).int().cuda()


def generate_topk_prefill_inputs(dist, layout, rows, ctx, k=K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    width, ks, ke = _ranges(layout, rows, ctx)
    x = _logits(dist, rows, width, g)
    out = torch.full((rows, k), -7, dtype=torch.int32, device="cuda")
    return x, ks, ke, out


def _check_row(x, s, e, o):
    L = max(e, s) - s
    if L <= K:
        exp = torch.cat([torch.arange(L), torch.full((K - L,), -1)]).int().cuda()
        return torch.equal(torch.sort(o)[0], torch.sort(exp)[0])
    if (o < 0).any() or (o >= L).any() or o.unique().numel() != K:
        return False
    row = x[s:e]
    nan = torch.isnan(row)
    sel = torch.zeros(L, dtype=torch.bool, device="cuda")
    sel[o.long()] = True
    if nan[sel].any():
        return bool(nan[~sel].all())
    return bool(((row[~sel] <= row[sel].min()) | nan[~sel]).all())


@pytest.mark.parametrize(
    "dist", ["random", "relu_sparse", "few_values", "all_equal", "nan_rows"]
)
@pytest.mark.parametrize(
    "layout,rows,ctx",
    [
        ("causal", 64, 4096),
        ("causal", 16, 24576),
        ("causal", 64, 65536),
        ("causal", 8, 131072),
        ("causal", 300, 2300),
        ("packed", 2048, 1024),
        ("packed", 4096, 2048),
    ],
)
def test_topk_prefill(dist, layout, rows, ctx):
    x, ks, ke, out = generate_topk_prefill_inputs(dist, layout, rows, ctx)
    top_k_per_row_prefill(x, ks, ke, out, K)
    again = torch.full_like(out, -7)
    top_k_per_row_prefill(x, ks, ke, again, K)
    assert torch.equal(out, again), "not deterministic"
    for r in torch.linspace(0, rows - 1, min(rows, 32)).long().tolist():
        assert _check_row(x[r], int(ks[r]), int(ke[r]), out[r]), f"row {r}"
