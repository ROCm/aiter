# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
from unittest import mock

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

PAGE, D, GROUP, TOPK = 128, 128, 16, 16
SCALE = D**-0.5
E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
# [(cached, new)] tokens a request; each token picks its own block + 15 earlier
CASES = {
    "2req": [(1000, 77), (5000, 259)],
    "no_prefix": [(0, 300)],
    "long": [(20000, 512)],
}


def make_case(requests):
    g = torch.Generator().manual_seed(0)
    nbs = [(c + n + PAGE - 1) // PAGE for c, n in requests]
    pages = sum(nbs) + 1
    perm = torch.randperm(pages, generator=g, device="cpu")
    start, table, ctx, tails = 0, [], [], []
    for (cached, new), nb in zip(requests, nbs):
        bt = perm[start : start + nb]
        for pos in range(cached, cached + new):
            keys = torch.rand(pos // PAGE, generator=g, device="cpu")
            full = keys.argsort()[: TOPK - 1]  # 15 random earlier blocks
            row = torch.cat([bt[full], bt[pos // PAGE :][:1]])  # tail page last
            table.append(F.pad(row, (0, TOPK - row.numel())))
            ctx.append(full.numel() * PAGE + pos % PAGE + 1)
        tails.append((int(bt[-1]), (cached + new) % PAGE or PAGE))
        start += nb
    ctx[-3:] = [0, 0, 0]  # a padded bucket's rows store zeros
    torch.manual_seed(0)
    kv = torch.randint(0, 256, (2, pages, 1, PAGE, D // 2), dtype=torch.uint8)
    scale = torch.rand(2, pages, 1, PAGE, D // 16) * 0.5 + 0.01
    scale = scale.to(torch.float8_e4m3fn)
    for page, tail in tails:
        scale[:, page, :, tail:] = float("nan")  # past the end: never visible
    q = (torch.randn(len(ctx), GROUP, D) * 2).bfloat16()
    table, ctx = torch.stack(table).int(), torch.tensor(ctx).int()
    return q, kv[0], kv[1], scale[0], scale[1], table, ctx


def dequant(payload, scale):
    codes = torch.stack([payload & 15, payload >> 4], -1).flatten(-2).long()
    v = torch.tensor(E2M1)[codes & 7] * torch.where(codes >= 8, -1.0, 1.0)
    return (v.unflatten(-1, (-1, 16)) * scale.float()[..., None]).flatten(-2)


def run_torch(q, kp, vp, ks, vs, table, ctx):
    k = dequant(kp[:, 0], ks[:, 0]).nan_to_num()[table.long()].flatten(1, 2)
    v = dequant(vp[:, 0], vs[:, 0]).nan_to_num()[table.long()].flatten(1, 2)
    s = torch.einsum("rhd,rtd->rht", q.float(), k) * SCALE
    visible = torch.arange(TOPK * PAGE)[None] < ctx[:, None]
    s = s.masked_fill(~visible[:, None], float("-inf"))
    return torch.einsum("rht,rtd->rhd", s.softmax(-1).nan_to_num(), v)


def mxfp8_ref(x):
    g = x.float().reshape(x.shape[0], -1, 32)
    m, e = torch.frexp(g.abs().amax(-1).clamp(min=1e-10) / 448.0)
    code = (torch.where(m == 0.5, e - 1, e) + 127).clamp(0, 254)
    q = (g * torch.exp2(127.0 - code)[..., None]).clamp(-448, 448)
    return q.to(torch.float8_e4m3fn).view(torch.uint8).flatten(1), code.byte()


@benchmark()
def test_pa_sparse_prefill_nvfp4(case):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode
    from aiter.ops.flydsl.pa_sparse_prefill import pa_sparse_prefill_nvfp4 as reuse

    host = make_case(CASES[case])  # the FP32 reference runs on the CPU
    q, kp, vp, ks, vs, table, ctx = (t.cuda() for t in host)
    plan = plan_pa_decode(ctx, 1, max_partitions=1)
    args = (q, kp, vp, ks, vs, table, ctx, SCALE)
    pa_args = (q, kp, vp, ctx, table, SCALE, 1)
    pa_kw = {"key_scale": ks, "value_scale": vs, "work_plan": plan}
    candidates = {  # pa_decode: the per-token path the reuse kernel replaces
        "pa_decode": lambda o: pa_decode(o, *pa_args, **pa_kw),
        "reuse_v1": lambda o: reuse(o, *args),
        "reuse_v2": lambda o: reuse(o, *args, tile_rows=2, impl="v2"),  # ATOM's
    }
    ref, ret = run_torch(*host), {}
    for name, fn in candidates.items():
        out = torch.full_like(q, float("nan"))
        _, us = run_perftest(fn, out)
        err = checkAllclose(ref, out.cpu().float(), atol=3e-2, rtol=3e-2, msg=name)
        assert err == 0, f"{case} {name}: {err:.2%} mismatched"
        ret[f"{name} us"], ret[f"{name} err"] = us, err
    rows, pad = q.shape[0], 5  # MXFP8 output: o_proj's group-32 quant, bitwise
    u8 = {"dtype": torch.uint8, "device": "cuda"}
    codes = torch.full((rows + pad, GROUP, D), 0x55, **u8)
    scales = torch.full((rows + pad, GROUP * D // 32), 0x33, **u8)
    out = torch.empty_like(q)
    reuse(out, *args, tile_rows=2, impl="v2")
    reuse(
        torch.empty_like(q),
        *args,
        tile_rows=2,
        impl="v2",
        quantized_output=codes,
        output_scale=scales,
        quantized_pad_rows=pad,
    )
    ref_codes, ref_scales = mxfp8_ref(out.cpu())
    codes, scales = codes.cpu(), scales.cpu()
    assert torch.equal(codes[:rows].flatten(1), ref_codes)
    assert torch.equal(scales[:rows], ref_scales)
    assert (codes[rows:] == 0).all() and (scales[rows:] == 127).all()
    return ret


def test_supported_gate():
    from aiter.ops.flydsl import pa_sparse_prefill as m

    cache, table = torch.zeros(4, 1, PAGE, D // 2), torch.zeros(8, TOPK)
    for gfx, heads, ok in [("gfx1250", 16, 1), ("gfx950", 16, 0), ("gfx1250", 8, 0)]:
        q = torch.zeros(8, heads, D, dtype=torch.bfloat16)
        with mock.patch.object(m, "get_gfx_runtime", return_value=gfx):
            assert m.pa_sparse_prefill_nvfp4_supported(q, cache, table) == ok, gfx


def main():
    parser = argparse.ArgumentParser(description="NVFP4 sparse prefill, K/V reuse")
    parser.add_argument("-c", "--case", nargs="*", default=list(CASES), choices=CASES)
    args = parser.parse_args()
    test_supported_gate()
    if get_gfx_runtime() != "gfx1250":
        aiter.logger.warning("NVFP4 sparse prefill runs on gfx1250 only; skipping")
        return
    df = pd.DataFrame([test_pa_sparse_prefill_nvfp4(c) for c in args.case])
    aiter.logger.info("pa_sparse_prefill_nvfp4:\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
