# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 prefill index block scores over the NVFP4 index cache (gfx1250).

``aiter.ops.flydsl.index_score_prefill`` against an FP32 torch reference:
score[h, r, blk] = max over the block's keys at or before r's position of
e4m3(clamp(q)) . dequant(K) times sm_scale * log2(e). The kernel's dots are the
hardware's 128-deep scaled WMMA, the reference's an FP32 matmul: they agree to
~2e-3 relative (MI455X: the WMMA's internal sum is not FP32-exact). (Bitwise equality with ATOM's Triton scorer, which issues
the same WMMA, is checked on the ATOM side: ``index_check.py``.)

The fused scorer + top-16 (``index_topk_prefill_nvfp4``) is held to the scorer
itself: the 16 largest of the keys it returns are, index for index and in order,
ATOM's Triton selection (``_pack_score_key`` over the forced blocks) of the
scores ``index_score_prefill_nvfp4`` writes for the same inputs. Its batches
repeat pages (blocks that score the same: the tie rule) and hold zero query
rows (+-0 scores).

Run with: python -m pytest -q op_tests/test_flydsl_index_score_prefill_nvfp4.py
"""

import importlib.util

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("ROCm is not available")
    if get_gfx_runtime() != "gfx1250":
        pytest.skip("the NVFP4 index scorer runs on gfx1250")
    if importlib.util.find_spec("flydsl") is None:
        pytest.skip("FlyDSL is not installed")
    return torch.device("cuda")


def dequant(payload, scale):
    table = torch.tensor(E2M1, device=payload.device)
    codes = torch.stack([payload & 15, payload >> 4], -1).flatten(-2).long()
    values = table[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    values = values.reshape(*values.shape[:-1], -1, 16)
    return (values * scale.view(torch.float8_e4m3fn).float()[..., None]).flatten(-2)


def make(reqs, device, seed=0, heads=1, ties=False):
    g = torch.Generator(device=device).manual_seed(seed)
    nbs = [(c + n + 127) // 128 for n, c in reqs]
    pages = sum(nbs) + 2
    payload = torch.randint(0, 256, (pages, 128, 64), dtype=torch.uint8, device=device, generator=g)
    scale = (torch.rand(pages, 128, 8, device=device, generator=g) * 0.5 + 0.01).to(
        torch.float8_e4m3fn
    ).view(torch.uint8)
    total = sum(n for n, _ in reqs)
    q = (torch.randn(total, heads, 128, device=device, generator=g) * 3).to(
        torch.bfloat16
    )
    perm = torch.randperm(pages, device=device).int()
    bt = torch.zeros(len(reqs), max(nbs), dtype=torch.int32, device=device)
    s = 0
    for i, nb in enumerate(nbs):
        bt[i, :nb] = perm[s : s + nb]
        if ties:
            # Every third block is its neighbour's page again.
            bt[i, 1:nb:3] = bt[i, 0 : nb - 1 : 3]
        s += nb
    if ties:
        q[::7] = 0  # and every seventh query row zero
    lens = torch.tensor([n for n, _ in reqs])
    cu = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]).int().to(device)
    prefix = torch.tensor([c for _, c in reqs], dtype=torch.int32, device=device)
    return q, payload, scale, bt, cu, prefix, prefix + lens.int().to(device)


@pytest.mark.parametrize(
    "reqs", [[(64, 4096)], [(300, 0)], [(77, 1000), (259, 5000)], [(1024, 30000)]]
)
# block_q:waves:ring[:fold] (fold: fold_max)
@pytest.mark.parametrize("variant", ["256:8:4", "256:4:8", "128:2:2:fold", "512:8:4:fold", "256:4:4:fold:lean"])
def test_index_score_prefill_nvfp4(reqs, variant, device):
    from aiter.ops.flydsl.index_score_prefill import index_score_prefill_nvfp4
    from aiter.ops.flydsl.kernels.index_score_prefill_nvfp4_gfx1250 import sm_scale_log2e

    bq, waves, ring = (int(x) for x in variant.split(":")[:3])
    fold_max = ":fold" in variant
    lean = variant.endswith(":lean")
    q, payload, scale, bt, cu, prefix, seq = make(reqs, device)
    max_block = (int(seq.max()) + 127) // 128
    score = torch.full((1, q.shape[0], max_block), float("nan"), device=device)
    index_score_prefill_nvfp4(
        q, payload, scale, score, bt, cu, seq, prefix, 128**-0.5,
        max(n for n, _ in reqs), 37, block_q=bq, waves=waves, ring=ring,
        fold_max=fold_max, lean=lean,
    )
    torch.cuda.synchronize()
    qe = q[:, 0].float().clamp(-448, 448).to(torch.float8_e4m3fn).float()
    c = sm_scale_log2e(128**-0.5)
    for b, (n, cached) in enumerate(reqs):
        nb = (cached + n + 127) // 128
        k = dequant(payload[bt[b, :nb].long()], scale[bt[b, :nb].long()]).reshape(-1, 128)
        rows = qe[cu[b] : cu[b + 1]]
        dots = rows @ k.T  # [n, nb * 128]
        pos = torch.arange(cached, cached + n, device=device)
        visible = torch.arange(nb * 128, device=device)[None] <= pos[:, None]
        dots = dots.masked_fill(~visible, float("-inf"))
        ref = dots.view(n, nb, 128).amax(-1) * c
        valid = torch.arange(nb, device=device)[None] < ((pos + 128) // 128)[:, None]
        got = score[0, cu[b] : cu[b + 1], :nb]
        err = ((got - ref).abs() / ref.abs().clamp(min=1e-3))[valid]
        assert torch.isfinite(got[valid]).all()
        assert err.max().item() < 5e-3, err.max().item()


def triton_selection(score, valid_blocks, init_blocks, local_blocks, k=16):
    """ATOM's Triton top-k of block scores ``[rows, blocks]``: by score, then the
    larger block; -0.0 as 0.0, a NaN under -inf; the first ``init_blocks`` and
    a row's last ``local_blocks`` blocks pinned over every score; -1 pads."""
    col = torch.arange(score.shape[1], device=score.device)[None]
    vb = valid_blocks[:, None]
    valid = col < vb
    s = torch.where(valid & (col < init_blocks), 1e30, score)
    s = torch.where(valid & (col >= vb - local_blocks), 1e29, s)
    bits = s.view(torch.int32).long() & 0xFFFFFFFF
    bits = torch.where(bits == 0x80000000, 0, bits)
    ordered = bits ^ torch.where(bits >> 31 != 0, 0xFFFFFFFF, 0x80000000)
    ordered = torch.where((bits & 0x7FFFFFFF) > 0x7F800000, 0, ordered)
    key = torch.where(valid, (1 << 48) | (ordered << 16) | (col + 1), 0)
    key = torch.nn.functional.pad(key, (0, max(0, k - key.shape[1])))
    top = key.sort(dim=1, descending=True)[0][:, :k]
    idx = (top & 0xFFFF) - 1
    return idx.masked_fill(torch.arange(k, device=score.device)[None] >= vb, -1)


@pytest.mark.parametrize(
    "reqs",
    [
        [(300, 0)],
        [(77, 1000), (259, 5000), (0, 9), (40, 100)],
        [(2048, 40000)],
        # MiniMax-M3.1 P: a ragged batch, contexts 50K-300K
        [(1100, 48900), (900, 149100), (1024, 299000)],
    ],
)
@pytest.mark.parametrize("pins", [(0, 1), (2, 3), (0, 0)])
@pytest.mark.parametrize(
    "variant",
    # chunk_blocks:block_q:waves:ring[:fold] (fold: fold_max, the same keys)
    [
        "256:256:4:4",
        "64:128:2:2",
        "1000:256:8:4",
        "512:256:4:8",
        "256:256:4:4:fold",
        "1024:128:2:2:fold",
        "256:512:8:2:fold",
        "128:512:8:2:fold:lean",
    ],
)
def test_index_topk_prefill_nvfp4(reqs, pins, variant, device):
    from aiter.ops.flydsl.index_score_prefill import (
        index_score_prefill_nvfp4,
        index_topk_prefill_nvfp4,
        index_topk_prefill_nvfp4_supported,
    )

    chunk, bq, waves, ring = (int(x) for x in variant.split(":")[:4])
    fold_max = ":fold" in variant
    lean = variant.endswith(":lean")
    heads = 4
    q, payload, scale, bt, cu, prefix, seq = make(reqs, device, heads=heads, ties=True)
    max_q = max(n for n, _ in reqs)
    max_block = (int(seq.max()) + 127) // 128
    assert index_topk_prefill_nvfp4_supported(q, payload, scale, 16, max_block)
    assert not index_topk_prefill_nvfp4_supported(q, payload, scale, 8, max_block)
    score = torch.full((heads, q.shape[0], max_block), float("nan"), device=device)
    index_score_prefill_nvfp4(
        q, payload, scale, score, bt, cu, seq, prefix, 128**-0.5, max_q, 37
    )
    keys = index_topk_prefill_nvfp4(
        q, payload, scale, bt, cu, seq, prefix, 128**-0.5, max_q, max_block, *pins,
        chunk_blocks=chunk, block_q=bq, waves=waves, ring=ring, fold_max=fold_max, lean=lean,
    )  # fmt: skip
    torch.cuda.synchronize()
    assert keys.shape == (heads, q.shape[0], -(-max_block // chunk) * 16)
    for b, (n, cached) in enumerate(reqs):
        rows = slice(int(cu[b]), int(cu[b + 1]))
        vb = (cached + torch.arange(n, device=device) + 128) // 128
        # A row's lists: of the chunks it reaches (the others are not written).
        reached = torch.arange(keys.shape[2], device=device)[None] < (
            (vb + chunk - 1) // chunk * 16
        )[:, None]
        for h in range(heads):
            want = triton_selection(score[h, rows], vb, *pins)
            top = keys[h, rows].masked_fill(~reached, 0).sort(1, True)[0][:, :16]
            got = ((top & 0xFFFF) - 1).masked_fill(
                torch.arange(16, device=device)[None] >= vb[:, None], -1
            )
            assert torch.equal(got, want), (b, h)
