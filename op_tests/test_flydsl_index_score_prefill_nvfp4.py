# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# One-hot Q: every dot is one product, exact in the WMMA; covers clamp and E4M3.
ONEHOT = (1.0, -2.0, 0.375, 448.0, -1000.0, 3.3, float("inf"), float("-inf"))
# (new rows, cached prefix) per request, heads, payload/scale in one paged cache
CASES = {
    "1req": ([(64, 4096)], 1, False),
    "ends_empty": ([(77, 1000), (259, 5000), (0, 700)], 1, False),
    "paged_3h": ([(0, 500), (513, 127), (1, 0), (300, 128)], 3, True),
}


# Fused scorer + top-16: batches, (init, local) pins, launch options (the same keys)
TOPK_CASES = {
    "1req": [(300, 0)],
    "ends_empty": [(77, 1000), (259, 5000), (0, 9), (40, 100)],
    "2k_40k": [(2048, 40000)],
    # MiniMax-M3.1 prefill: a ragged batch, contexts 50K-300K
    "m31_ragged": [(1100, 48900), (900, 149100), (1024, 299000)],
}
TOPK_PINS = [(0, 1), (2, 3), (0, 0)]
TOPK_OPTS = {
    "default": {},
    "c64_128x2": {"chunk_blocks": 64, "block_q": 128, "waves": 2, "ring": 2},
    "c1000_512x8": {"chunk_blocks": 1000, "block_q": 512, "waves": 8, "ring": 2},
    "c256_plain": {
        "chunk_blocks": 256,
        "ring": 4,
        "fold_max": 0,
        "pad_lds": 0,
        "sel32": 0,
    },
    "c1024_pad64": {"chunk_blocks": 1024, "ring": 4, "fold_max": 0, "sel32": 0},
}


def make(reqs, heads, paged, onehot, dev="cuda", ties=False):
    g = torch.Generator().manual_seed(0)
    nbs = [(c + n + 127) // 128 for n, c in reqs]
    pages = sum(nbs) + 2
    payload = torch.randint(0, 256, (pages, 128, 64), dtype=torch.uint8, generator=g)
    scale = torch.rand(pages, 128, 8, generator=g) * 0.5 + 0.01
    scale = scale.to(torch.float8_e4m3fn).view(torch.uint8)
    if paged:
        cache = torch.cat([payload.flatten(1), scale.flatten(1)], 1).to(dev)
        payload = cache[:, : 128 * 64].unflatten(1, (128, 64))
        scale = cache[:, 128 * 64 :].unflatten(1, (128, 8))
    total = sum(n for n, _ in reqs)
    q = torch.randn(total, heads, 128, generator=g) * 3
    if onehot:
        rows, head = torch.arange(total)[:, None], torch.arange(heads)
        value = torch.tensor(ONEHOT)[(rows + head) % len(ONEHOT)]
        q = torch.zeros_like(q).scatter_(
            2, ((rows * 7 + head * 13) % 128)[..., None], value[..., None]
        )
    perm = torch.randperm(pages, generator=g).int()
    bt = torch.zeros(len(reqs), max(nbs), dtype=torch.int32)
    for i, nb in enumerate(nbs):
        bt[i, :nb] = perm[sum(nbs[:i]) : sum(nbs[: i + 1])]
        if ties:  # every third block is its neighbour's page again
            bt[i, 1:nb:3] = bt[i, 0 : nb - 1 : 3]
    if ties:
        q[::7] = 0  # and every seventh query row zero (+-0 scores)
    lens = torch.tensor([n for n, _ in reqs], dtype=torch.int32)
    cu = torch.cat([lens.new_zeros(1), lens.cumsum(0)]).int()
    prefix = torch.tensor([c for _, c in reqs], dtype=torch.int32)
    meta = (bt, cu, prefix, prefix + lens)
    return (
        q.bfloat16().to(dev),
        payload.to(dev),
        scale.to(dev),
        *(t.to(dev) for t in meta),
    )


def run_torch(reqs, q, payload, scale, bt, cu, c, max_block, block_q=256):
    """FP32 scores on the CPU; NaN where the scorer must not write."""
    codes = torch.stack([payload & 15, payload >> 4], -1).flatten(-2).long()
    k = torch.tensor(E2M1)[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    k = k.unflatten(-1, (-1, 16)) * scale.view(torch.float8_e4m3fn).float()[..., None]
    k = k.flatten(-2)
    qe = q.float().clamp(-448, 448).to(torch.float8_e4m3fn).float()
    ref = torch.full((q.shape[1], q.shape[0], max_block), float("nan"))
    for b, (n, cached) in enumerate(reqs):
        nb = (cached + n + 127) // 128
        kb = k[bt[b, :nb].long()].reshape(-1, 128)
        local = torch.arange(n)
        visible = torch.arange(nb * 128)[None] <= (cached + local)[:, None]
        last = torch.clamp((local // block_q + 1) * block_q, max=n) + cached
        written = torch.arange(nb)[None] < ((last + 127) // 128)[:, None]
        dots = torch.einsum("nhd,kd->hnk", qe[cu[b] : cu[b + 1]], kb)
        blk = dots.masked_fill(~visible, float("-inf")).unflatten(-1, (nb, 128))
        blk = (blk.amax(-1) * c).masked_fill(~written, float("nan"))
        ref[:, cu[b] : cu[b + 1], :nb] = blk
    return ref


@benchmark()
def test_index_score_prefill_nvfp4(case, onehot):
    from aiter.ops.flydsl.index_score_prefill import index_score_prefill_nvfp4
    from aiter.ops.flydsl.kernels.index_score_prefill_nvfp4_gfx1250 import (
        sm_scale_log2e,
    )

    reqs, heads, paged = CASES[case]
    q, payload, scale, bt, cu, prefix, seq = make(reqs, heads, paged, onehot)
    max_block = (int(seq.max()) + 127) // 128
    score = torch.full((heads, q.shape[0], max_block), float("nan"), device="cuda")
    args = (q, payload, scale, score, bt, cu, seq, prefix, 128**-0.5)
    max_q, chunk = max(n for n, _ in reqs), 5 if onehot else 37  # 37 -> CHUNK_MAX
    _, us = run_perftest(index_score_prefill_nvfp4, *args, max_q, chunk)
    score = score.cpu()
    host = (t.cpu() for t in (q, payload, scale, bt, cu))
    ref = run_torch(reqs, *host, sm_scale_log2e(128**-0.5), max_block)
    assert torch.equal(score.isnan(), ref.isnan()), "written blocks differ"
    assert torch.equal(score == -torch.inf, ref == -torch.inf), "masked blocks differ"
    atol, rtol = (0, 0) if onehot else (2e-3, 1e-4)  # dense: 128-deep WMMA sum
    err = checkAllclose(ref, score, rtol=rtol, atol=atol, mask=ref.isfinite())
    assert err == 0, f"{case} onehot={onehot}: {err:.2%} mismatched"
    return {"us": us, "err": err}


@benchmark()
def test_index_score_prefill_nvfp4_opts(case):
    """fold_max / pad_lds / the tile change the schedule, not a score."""
    from aiter.ops.flydsl.index_score_prefill import index_score_prefill_nvfp4

    reqs, heads, paged = CASES[case]
    q, payload, scale, bt, cu, prefix, seq = make(reqs, heads, paged, False)
    max_block = (int(seq.max()) + 127) // 128
    want = torch.full((heads, q.shape[0], max_block), float("nan"), device="cuda")
    args = (bt, cu, seq, prefix, 128**-0.5, max(n for n, _ in reqs), 32)
    index_score_prefill_nvfp4(q, payload, scale, want, *args)
    for opts in (
        {"fold_max": True, "pad_lds": True},
        {"block_q": 128, "waves": 2, "ring": 2, "fold_max": True, "pad_lds": True},
    ):
        got = torch.full_like(want, float("nan"))
        index_score_prefill_nvfp4(q, payload, scale, got, *args, **opts)
        assert torch.equal(got.isnan(), want.isnan()), opts
        assert torch.equal(got.nan_to_num(), want.nan_to_num()), opts
    return {"ok": 1}


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


@benchmark()
def test_index_topk_prefill_nvfp4(case, pins, opts):
    """The 16 largest keys of a row are, index for index and in order, ATOM's
    Triton selection of the scores ``index_score_prefill_nvfp4`` writes."""
    from aiter.ops.flydsl.index_score_prefill import (
        index_score_prefill_nvfp4,
        index_topk_prefill_chunk_blocks,
        index_topk_prefill_nvfp4,
    )

    reqs, heads = TOPK_CASES[case], 4
    q, payload, scale, bt, cu, prefix, seq = make(reqs, heads, True, False, ties=True)
    max_q = max(n for n, _ in reqs)
    max_block = (int(seq.max()) + 127) // 128
    score = torch.full((heads, q.shape[0], max_block), float("nan"), device="cuda")
    meta = (bt, cu, seq, prefix, 128**-0.5, max_q)
    index_score_prefill_nvfp4(q, payload, scale, score, *meta, 32)
    kw = TOPK_OPTS[opts]
    chunk = kw.get("chunk_blocks") or index_topk_prefill_chunk_blocks(
        q, max_q, len(reqs), max_block
    )
    keys, us = run_perftest(
        index_topk_prefill_nvfp4, q, payload, scale, *meta, max_block, *pins, **kw
    )
    assert keys.shape == (heads, q.shape[0], -(-max_block // chunk) * 16)
    for b, (n, cached) in enumerate(reqs):
        rows = slice(int(cu[b]), int(cu[b + 1]))
        vb = (cached + torch.arange(n, device="cuda") + 128) // 128
        # A row's lists: of the chunks it reaches (the others are not written).
        reached = (
            torch.arange(keys.shape[2], device="cuda")[None]
            < ((vb + chunk - 1) // chunk * 16)[:, None]
        )
        for h in range(heads):
            want = triton_selection(score[h, rows], vb, *pins)
            top = keys[h, rows].masked_fill(~reached, 0).sort(1, True)[0][:, :16]
            got = ((top & 0xFFFF) - 1).masked_fill(
                torch.arange(16, device="cuda")[None] >= vb[:, None], -1
            )
            assert torch.equal(got, want), (case, pins, opts, b, h)
    return {"us": us}


def main():
    parser = argparse.ArgumentParser(description="NVFP4 prefill index scores")
    parser.add_argument(
        "-c", "--case", nargs="*", default=list(CASES), choices=list(CASES)
    )
    args = parser.parse_args()
    if get_gfx_runtime() != "gfx1250":
        aiter.logger.warning(
            "NVFP4 prefill index scorer runs on gfx1250 only; skipping"
        )
        return
    df = [test_index_score_prefill_nvfp4(c, o) for c in args.case for o in (1, 0)]
    aiter.logger.info(
        "index_score_prefill_nvfp4:\n%s", pd.DataFrame(df).to_markdown(index=False)
    )
    df = [test_index_score_prefill_nvfp4_opts(c) for c in args.case]
    aiter.logger.info(
        "index_score_prefill_nvfp4 options (same bits):\n%s",
        pd.DataFrame(df).to_markdown(index=False),
    )
    # Every pin set at the defaults, every launch option at the first pins.
    runs = [(p, "default") for p in TOPK_PINS]
    runs += [(TOPK_PINS[0], o) for o in TOPK_OPTS if o != "default"]
    df = [test_index_topk_prefill_nvfp4(c, p, o) for c in TOPK_CASES for p, o in runs]
    aiter.logger.info(
        "index_topk_prefill_nvfp4:\n%s", pd.DataFrame(df).to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
