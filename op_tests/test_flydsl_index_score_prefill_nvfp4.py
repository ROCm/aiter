# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
from unittest import mock

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


def make(reqs, heads, paged, onehot, dev="cuda"):
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


def test_supported_gate():
    from aiter.ops.flydsl import index_score_prefill as m

    q = torch.zeros(4, 1, 128, dtype=torch.bfloat16)
    payload, scale = torch.zeros(2, 128, 64, dtype=torch.uint8), torch.zeros(2, 128, 8)
    e4m3 = scale.to(torch.float8_e4m3fn)
    for gfx, s, ok in [
        ("gfx1250", e4m3, 1),
        ("gfx950", e4m3, 0),
        ("gfx1250", scale, 0),
    ]:
        with mock.patch.object(m, "get_gfx_runtime", return_value=gfx):
            assert m.index_score_prefill_nvfp4_supported(q, payload, s) == ok, gfx


def main():
    parser = argparse.ArgumentParser(description="NVFP4 prefill index scores")
    parser.add_argument(
        "-c", "--case", nargs="*", default=list(CASES), choices=list(CASES)
    )
    args = parser.parse_args()
    test_supported_gate()
    if get_gfx_runtime() != "gfx1250":
        aiter.logger.warning(
            "NVFP4 prefill index scorer runs on gfx1250 only; skipping"
        )
        return
    df = [test_index_score_prefill_nvfp4(c, o) for c in args.case for o in (1, 0)]
    aiter.logger.info(
        "index_score_prefill_nvfp4:\n%s", pd.DataFrame(df).to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
