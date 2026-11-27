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
    if dist == "zeros50":  # half exact zeros (ReLU-weighted indexer logits)
        return torch.relu(torch.randn((rows, width), device="cuda", generator=g))
    if dist == "bf16":  # bf16-rounded logits: many ties around the k-th value
        return torch.randn((rows, width), device="cuda", generator=g).bfloat16().float()
    # +0 / -0 / a few small integers, k-th value tied at zero
    if dist == "signed_zeros":
        x = torch.randint(-1, 2, (rows, width), device="cuda", generator=g).float()
        x[x == 1] = torch.randint(
            1, 4, (int((x == 1).sum()),), device="cuda", generator=g
        ).float()
        return torch.where(
            torch.rand((rows, width), device="cuda", generator=g) < 0.5, x, -x * 0.0
        )
    if dist == "inf":  # +-inf and NaN mixed into random logits
        x = torch.randn((rows, width), device="cuda", generator=g)
        r = torch.rand((rows, width), device="cuda", generator=g)
        x[r < 0.01] = float("inf")
        x[(r >= 0.01) & (r < 0.02)] = float("-inf")
        x[(r >= 0.02) & (r < 0.03)] = float("nan")
        return x
    # sorted rows: every value is a new maximum (list overflow, prune and redo)
    if dist == "ascending":
        return torch.sort(
            torch.randn((rows, width), device="cuda", generator=g), dim=1
        )[0]
    # logits trending up with position plus noise (causal attention-like)
    if dist == "recency":
        pos = torch.arange(width, device="cuda", dtype=torch.float32) / max(width, 1)
        return 4.0 * pos + torch.randn((rows, width), device="cuda", generator=g)
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


def generate_topk_prefill_inputs(dist, layout, rows, ctx, k=K):
    g = torch.Generator(device="cuda").manual_seed(0)
    width, ks, ke = _ranges(layout, rows, ctx)
    x = _logits(dist, rows, width, g)
    out = torch.full((rows, k), -7, dtype=torch.int32, device="cuda")
    return x, ks, ke, out


def _check_row(x, s, e, o, k=K):
    L = max(e, s) - s
    if L <= k:
        exp = torch.cat([torch.arange(L), torch.full((k - L,), -1)]).int().cuda()
        return torch.equal(torch.sort(o)[0], torch.sort(exp)[0])
    if (o < 0).any() or (o >= L).any() or o.unique().numel() != k:
        return False
    row = x[s:e]
    nan = torch.isnan(row)
    sel = torch.zeros(L, dtype=torch.bool, device="cuda")
    sel[o.long()] = True
    if nan[sel].any():
        return bool(nan[~sel].all())
    return bool(((row[~sel] <= row[sel].min()) | nan[~sel]).all())


# each value class once; shapes: classify kernel (rows <= k), ring fallback, 2-level sample, max width
@pytest.mark.parametrize(
    "dist,rows,ctx",
    [
        ("random", 300, 2300),
        ("random", 64, 6144),
        ("random", 16, 24576),
        ("random", 4, 262143),
        ("few_values", 64, 6144),
        ("all_equal", 300, 2300),
        ("nan_rows", 16, 24576),
        ("bf16", 64, 6144),
        ("signed_zeros", 300, 2300),
        ("inf", 16, 24576),
        ("ascending", 64, 6144),
        ("relu_sparse", 4, 262143),
    ],
)
def test_topk_prefill(dist, rows, ctx):
    x, ks, ke, out = generate_topk_prefill_inputs(dist, "causal", rows, ctx)
    top_k_per_row_prefill(x, ks, ke, out, K)
    again = torch.full_like(out, -7)
    top_k_per_row_prefill(x, ks, ke, again, K)
    assert torch.equal(out, again), "not deterministic"
    for r in torch.linspace(0, rows - 1, min(rows, 32)).long().tolist():
        assert _check_row(x[r], int(ks[r]), int(ke[r]), out[r]), f"row {r}"


# ragged rows at unaligned offsets, lengths 1 to past k, in the classify (4096) and prefill (40000) kernels
@pytest.mark.parametrize("width", [4096, 40000])
def test_topk_prefill_row_offsets(width):
    g = torch.Generator(device="cuda").manual_seed(1)
    rows = 56
    x = torch.randn((rows, width), device="cuda", generator=g)
    ks = torch.randint(0, width // 8, (rows,), device="cuda", generator=g).int()
    lens = torch.tensor(
        [1, K - 1, K, K + 1, K + 3, 3 * K + 5, 30001] * 8, device="cuda"
    ).int()
    ke = (ks + lens).clamp(max=width).int()
    out = torch.full((rows, K), -7, dtype=torch.int32, device="cuda")
    top_k_per_row_prefill(x, ks, ke, out, K)
    for r in range(rows):
        assert _check_row(x[r], int(ks[r]), int(ke[r]), out[r]), f"row {r}"


# short_rows_classify=False: also the prefill kernel's SPARSE compaction (rows > CUs) and its ring fallback
@pytest.mark.parametrize(
    "opts,dist,rows,ctx",
    [
        ({"zero_agg": True}, "relu_sparse", 32, 2300),
        ({"short_rows_classify": False}, "nan_rows", 300, 2300),
        ({"route_few": False}, "few_values", 32, 12288),
    ],
)
def test_topk_prefill_options(opts, dist, rows, ctx):
    x, ks, ke, out = generate_topk_prefill_inputs(dist, "causal", rows, ctx)
    top_k_per_row_prefill(x, ks, ke, out, K, **opts)
    for r in range(rows):
        assert _check_row(x[r], int(ks[r]), int(ke[r]), out[r]), f"row {r}"


@pytest.mark.parametrize(
    "k,dist,ctx",
    [(256, "nan_rows", 12288), (512, "few_values", 12288), (1024, "random", 2300)],
)
def test_topk_prefill_other_k(k, dist, ctx):
    rows = 32
    x, ks, ke, out = generate_topk_prefill_inputs(dist, "causal", rows, ctx, k)
    top_k_per_row_prefill(x, ks, ke, out, k)
    for r in range(rows):
        assert _check_row(x[r], int(ks[r]), int(ke[r]), out[r], k), f"row {r}"
