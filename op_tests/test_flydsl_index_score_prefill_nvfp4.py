# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 prefill index block scores over the NVFP4 index cache (gfx1250).

``aiter.ops.flydsl.index_score_prefill`` against an FP32 torch reference of the
whole score tensor: score[h, r, blk] = max over the block's keys at or before
r's position of e4m3(clamp(q)) . dequant(K) times sm_scale * log2(e), -inf for
a block past r's own, and untouched (the NaN the buffer is filled with) past the
last block of r's ``block_q``-row tile.

Two kinds of Q. One-hot rows make every dot a single product, which the scaled
WMMA returns exactly: those scores must equal the reference bit for bit, and
they carry the clamp (+-1000, inf), the E4M3 rounding of Q, the dequant, the
causal mask and the scale. Dense rows sum 128 products in the hardware's
128-deep WMMA, whose internal sum is not FP32's: on MI455X it is within 6e-4
absolute of the FP32 matmul for scores up to ~30. Bitwise equality with ATOM's
Triton scorer, which issues the same WMMA, is checked on the ATOM side.

Run with: python3 op_tests/test_flydsl_index_score_prefill_nvfp4.py
"""

import importlib.util
import sys

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
PAGE_BYTES = 128 * 64 + 128 * 8
# One-hot Q values: past the E4M3 range on both sides, and not E4M3 (3.3).
ONEHOT = (1.0, -2.0, 0.375, 448.0, -1000.0, 3.3, float("inf"), float("-inf"))


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


def make(reqs, device, seed=0, heads=1, onehot=False, paged=False):
    """``reqs`` = [(new rows, cached prefix)]. ``paged``: payload and scales are
    views of one [pages, 9216] cache (one page stride), else two tensors."""
    g = torch.Generator(device=device).manual_seed(seed)
    nbs = [(c + n + 127) // 128 for n, c in reqs]
    pages = sum(nbs) + 2
    payload = torch.randint(
        0, 256, (pages, 128, 64), dtype=torch.uint8, device=device, generator=g
    )
    scale = (
        (torch.rand(pages, 128, 8, device=device, generator=g) * 0.5 + 0.01)
        .to(torch.float8_e4m3fn)
        .view(torch.uint8)
    )
    if paged:
        cache = torch.cat([payload.flatten(1), scale.flatten(1)], 1)
        payload = cache[:, : 128 * 64].unflatten(1, (128, 64))
        scale = cache[:, 128 * 64 :].unflatten(1, (128, 8))
    total = sum(n for n, _ in reqs)
    q = torch.randn(total, heads, 128, device=device, generator=g) * 3
    if onehot:
        rows = torch.arange(total, device=device)[:, None]
        head = torch.arange(heads, device=device)
        dim = (rows * 7 + head * 13) % 128
        value = torch.tensor(ONEHOT, device=device)[(rows + head) % len(ONEHOT)]
        q = torch.zeros_like(q).scatter_(2, dim[..., None], value[..., None])
    q = q.to(torch.bfloat16)
    perm = torch.randperm(pages, device=device, generator=g).int()
    bt = torch.zeros(len(reqs), max(nbs), dtype=torch.int32, device=device)
    s = 0
    for i, nb in enumerate(nbs):
        bt[i, :nb] = perm[s : s + nb]
        s += nb
    lens = torch.tensor([n for n, _ in reqs])
    cu = torch.cat([torch.zeros(1, dtype=torch.long), lens.cumsum(0)]).int().to(device)
    prefix = torch.tensor([c for _, c in reqs], dtype=torch.int32, device=device)
    return q, payload, scale, bt, cu, prefix, prefix + lens.int().to(device)


def reference(reqs, q, payload, scale, bt, cu, c, max_block, block_q):
    """The whole score tensor; NaN where the scorer must not write."""
    heads, device = q.shape[1], q.device
    ref = torch.full((heads, q.shape[0], max_block), float("nan"), device=device)
    qe = q.float().clamp(-448, 448).to(torch.float8_e4m3fn).float()
    blocks = torch.arange(max_block, device=device)
    for b, (n, cached) in enumerate(reqs):
        if n == 0:
            continue
        nb = (cached + n + 127) // 128
        pages = bt[b, :nb].long()
        k = dequant(payload[pages], scale[pages]).reshape(-1, 128)
        local = torch.arange(n, device=device)
        pos = cached + local
        visible = torch.arange(nb * 128, device=device)[None] <= pos[:, None]
        # Blocks up to the one of the last row of the row's block_q tile.
        last = torch.clamp((local // block_q + 1) * block_q, max=n) + cached
        written = blocks[None] < ((last + 127) // 128)[:, None]
        for h in range(heads):
            dots = qe[cu[b] : cu[b + 1], h] @ k.T  # [n, nb * 128]
            dots = dots.masked_fill(~visible, float("-inf"))
            full = torch.full((n, max_block), float("nan"), device=device)
            full[:, :nb] = dots.view(n, nb, 128).amax(-1) * c
            ref[h, cu[b] : cu[b + 1]] = full.masked_fill(~written, float("nan"))
    return ref


@pytest.mark.parametrize(
    "reqs,heads,paged",
    [
        ([(64, 4096)], 1, False),
        ([(300, 0)], 1, True),
        # Ends on an empty request: its CTAs must not read idx_q, which ends
        # where that request starts.
        ([(77, 1000), (259, 5000), (0, 700)], 1, False),
        # Starts on one; a tile boundary inside a page; a one-row request.
        ([(0, 500), (513, 127), (1, 0), (300, 128)], 3, True),
        ([(1024, 30000)], 2, False),
    ],
)
# block_q:waves:ring; the first is the wrapper's default, the last the smallest.
@pytest.mark.parametrize("variant", ["256:4:4", "256:8:4", "256:4:8", "128:2:2"])
@pytest.mark.parametrize("onehot", [False, True])
def test_index_score_prefill_nvfp4(reqs, heads, paged, variant, onehot, device):
    from aiter.ops.flydsl.index_score_prefill import index_score_prefill_nvfp4
    from aiter.ops.flydsl.kernels.index_score_prefill_nvfp4_gfx1250 import (
        sm_scale_log2e,
    )

    bq, waves, ring = (int(x) for x in variant.split(":"))
    q, payload, scale, bt, cu, prefix, seq = make(
        reqs, device, heads=heads, onehot=onehot, paged=paged
    )
    max_block = (int(seq.max()) + 127) // 128
    score = torch.full((heads, q.shape[0], max_block), float("nan"), device=device)
    index_score_prefill_nvfp4(
        q,
        payload,
        scale,
        score,
        bt,
        cu,
        seq,
        prefix,
        128**-0.5,
        max(n for n, _ in reqs),
        5 if onehot else 37,  # 37 is capped to CHUNK_MAX
        block_q=bq,
        waves=waves,
        ring=ring,
    )
    torch.cuda.synchronize()
    ref = reference(
        reqs, q, payload, scale, bt, cu, sm_scale_log2e(128**-0.5), max_block, bq
    )
    # Written exactly where Triton writes, masked exactly where it masks.
    assert torch.equal(torch.isnan(score), torch.isnan(ref))
    assert torch.equal(score == float("-inf"), ref == float("-inf"))
    finite = torch.isfinite(ref)
    assert finite.any()
    err = (score - ref)[finite].abs()
    if onehot:
        assert err.max().item() == 0.0, err.max().item()
    else:
        tol = 2e-3 + 1e-4 * ref[finite].abs()
        assert (err <= tol).all(), (err / tol).max().item()


def test_index_score_prefill_nvfp4_rejects(device):
    from aiter.ops.flydsl.index_score_prefill import index_score_prefill_nvfp4

    q, payload, scale, bt, cu, prefix, seq = make([(64, 0)], device)
    score = torch.empty(1, 64, 1, device=device)
    args = (q, payload, scale, score, bt, cu, seq, prefix, 0.1, 64, 32)
    index_score_prefill_nvfp4(*args)
    for i, bad in (
        (6, seq.long()),  # an int64 seq_lens would be read as int32 pairs
        (2, scale.float()),  # FP32 scales would be read as E4M3 bytes
        (7, prefix[:0]),  # shorter than the batch
    ):
        with pytest.raises(ValueError, match="unsupported"):
            index_score_prefill_nvfp4(*args[:i], bad, *args[i + 1 :])
    with pytest.raises(ValueError, match="waves"):  # wave 1 loads the scales
        index_score_prefill_nvfp4(*args, block_q=16, waves=1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
