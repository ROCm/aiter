# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
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
# MiniMax-M3.1 P = PP4 x TP1 (64 q heads, 4 KV heads on the rank): 32K-token
# prefill chunks, [(cached, new)] a request, and the P trace's median us of the
# kernel per layer (AgentX C192, MI455X; None: not traced).
M31_KV_HEADS = 4
M31_CASES = {
    "1req-32k-on-82k": ([(81920, 32768)], 2653.8),
    "2req-on-53k": ([(52736, 21720), (128, 11008)], 2527.8),
    "2req-short": ([(11136, 11456), (128, 21248)], 2469.3),
    "5req-on-21k-206k": (
        [(21376, 1793), (165760, 24165), (66560, 1592), (23808, 3073), (205568, 2048)],
        2620.9,
    ),
    # Higher prefix hit (what P sees once the hit rate rises): many requests per 32K chunk, each with a few
    # thousand new tokens on a long cached prefix. No trace time yet (None).
    "hi95-5req-6.5k-on-125k": ([(124928, 6528)] * 5, None),
    "hi98-12req-2.7k-on-128k": ([(128000, 2688)] * 12, None),
    "hi95-4req-8k-on-350k": ([(349952, 8192)] * 4, None),
    "hi99-24req-1.3k-on-130k": ([(129920, 1344)] * 24, None),
}
PAGE_BYTES = 2 * PAGE * (D // 2 + D // 16)  # K + V codes and scales a page


def picks(cached, new, g, swap):
    """[new, 15] earlier logical blocks of each token (-1: none). ``swap`` None:
    15 random ones; else the indexer-like pattern: a request-wide base set,
    each slot swapped for a random earlier block with probability ``swap``."""
    cur = torch.arange(cached, cached + new) // PAGE
    if swap is None:
        keys = torch.rand(new, max(int(cur.max()) + 1, TOPK), generator=g)
        keys = keys.masked_fill(torch.arange(keys.shape[1]) >= cur[:, None], 2)
        cand = keys.argsort(-1)[:, : TOPK - 1]
    else:
        base = torch.randperm(max(cached // PAGE, TOPK), generator=g)[: TOPK - 1]
        alt = (
            torch.rand(new, TOPK - 1, generator=g) * cur.clamp(min=1)[:, None]
        ).long()
        cand = torch.where(torch.rand(new, TOPK - 1, generator=g) < swap, alt, base)
        cand = torch.where(cand < cur[:, None], cand, -1)
        dup = (cand[:, :, None] == cand[:, None, :]).tril(-1).any(-1)
        cand = cand.masked_fill(dup, -1)
    return torch.where(cand < cur[:, None], cand, -1)


def make_case(requests, kv_heads=1, swap=None, pad=3):
    """Rows ``token * kv_heads + h`` (ATOM's [tokens * Hkv, 16, 128] view of
    q [tokens, Hkv * 16, 128]); page ``p * kv_heads + h`` is page p of KV head h;
    each KV head picks its own blocks. The last ``pad`` rows are a padded bucket."""
    g, H = torch.Generator().manual_seed(0), kv_heads
    nbs = [(c + n + PAGE - 1) // PAGE for c, n in requests]
    pages = sum(nbs) + 1
    perm = torch.randperm(pages, generator=g, device="cpu")
    start, table, ctx, tails = 0, [], [], []
    for (cached, new), nb in zip(requests, nbs):
        bt = perm[start : start + nb]
        pos = torch.arange(cached, cached + new)
        t, c = [], []
        for h in range(H):
            full = picks(cached, new, g, swap)
            full = full.gather(1, (full < 0).int().sort(dim=1, stable=True).indices)
            n = (full >= 0).sum(1)
            row = torch.where(full >= 0, bt[full.clamp(min=0)] * H + h, 0)
            row = F.pad(row, (0, 1)).scatter(
                1, n[:, None], bt[pos // PAGE, None] * H + h
            )
            t.append(row)  # full pages in pick order, the tail page last
            c.append(n * PAGE + pos % PAGE + 1)
        table.append(torch.stack(t, 1).flatten(0, 1))
        ctx.append(torch.stack(c, 1).flatten())
        tails += [
            (int(bt[-1]) * H + h, (cached + new) % PAGE or PAGE) for h in range(H)
        ]
        start += nb
    table, ctx = torch.cat(table).int(), torch.cat(ctx).int()
    if pad:
        ctx[-pad:] = 0  # a padded bucket's rows store zeros
    torch.manual_seed(0)
    kv = torch.randint(0, 256, (2, pages * H, 1, PAGE, D // 2), dtype=torch.uint8)
    scale = torch.rand(2, pages * H, 1, PAGE, D // 16) * 0.5 + 0.01
    scale = scale.to(torch.float8_e4m3fn)
    for page, tail in tails:
        scale[:, page, :, tail:] = float("nan")  # past the end: never visible
    q = (torch.randn(ctx.numel() // H, H * GROUP, D) * 2).bfloat16().view(-1, GROUP, D)
    return q, kv[0], kv[1], scale[0], scale[1], table, ctx


def dequant(payload, scale):
    codes = torch.stack([payload & 15, payload >> 4], -1).flatten(-2).long()
    v = torch.tensor(E2M1, device=codes.device)[codes & 7] * torch.where(
        codes >= 8, -1.0, 1.0
    )
    return (v.unflatten(-1, (-1, 16)) * scale.float()[..., None]).flatten(-2)


def run_torch(q, kp, vp, ks, vs, table, ctx, chunk=1024):
    kd, vd = (dequant(p[:, 0], s[:, 0]).nan_to_num() for p, s in ((kp, ks), (vp, vs)))
    visible = torch.arange(TOPK * PAGE, device=q.device)[None] < ctx[:, None]
    out = torch.empty(q.shape, device=q.device)
    for r in range(0, q.shape[0], chunk):  # bounded FP32 scratch for 32K-token chunks
        rows = slice(r, r + chunk)
        k, v = (x[table[rows].long()].flatten(1, 2) for x in (kd, vd))
        s = torch.einsum("rhd,rtd->rht", q[rows].float(), k) * SCALE
        s = s.masked_fill(~visible[rows, None], float("-inf"))
        out[rows] = torch.einsum("rht,rtd->rhd", s.softmax(-1).nan_to_num(), v)
    return out


def tile_pages(table, ctx, kv_heads, tq):
    """Distinct pages the kernel walks: a tile is ``tq`` consecutive tokens of one KV head."""
    n = (ctx.long() + PAGE - 1) // PAGE
    t = torch.where(
        torch.arange(TOPK, device=table.device) < n[:, None], table.long(), -1
    )
    t = t.view(-1, kv_heads, TOPK).transpose(0, 1)
    t = (
        F.pad(t, (0, 0, 0, -t.shape[1] % tq), value=-1)
        .reshape(-1, tq * TOPK)
        .sort(-1)
        .values
    )
    return int(((t[:, 1:] != t[:, :-1]) & (t[:, 1:] >= 0)).sum() + (t[:, 0] >= 0).sum())


def mxfp8_ref(x):
    g = x.float().reshape(x.shape[0], -1, 32)
    m, e = torch.frexp(g.abs().amax(-1).clamp(min=1e-10) / 448.0)
    code = (torch.where(m == 0.5, e - 1, e) + 127).clamp(0, 254)
    q = (g * torch.exp2(127.0 - code)[..., None]).clamp(-448, 448)
    return q.to(torch.float8_e4m3fn).view(torch.uint8).flatten(1), code.byte()


@benchmark()
def test_pa_sparse_prefill_nvfp4(case, kv_heads):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode
    from aiter.ops.flydsl.pa_sparse_prefill import M31_PREFILL_CALL
    from aiter.ops.flydsl.pa_sparse_prefill import pa_sparse_prefill_nvfp4 as reuse

    host = make_case(CASES[case], kv_heads)  # the FP32 reference runs on the CPU
    q, kp, vp, ks, vs, table, ctx = (t.cuda() for t in host)
    plan = plan_pa_decode(ctx, 1, max_partitions=1)
    args = (q, kp, vp, ks, vs, table, ctx, SCALE)
    kv = {"kv_heads": kv_heads}
    # the same K/V planes in the other address order (v2 and v3 fold a K/V pair
    # of planes into one 2-D tensor DMA, the lower-addressed plane leading)
    vk, sk = torch.stack([vp, kp]), torch.stack([vs, ks])
    args_rev = (q, vk[1], vk[0], sk[1], sk[0], table, ctx, SCALE)
    pa_args = (q, kp, vp, ctx, table, SCALE, 1)
    pa_kw = {"key_scale": ks, "value_scale": vs, "work_plan": plan}
    candidates = {  # pa_decode: the per-token path the reuse kernel replaces
        "pa_decode": lambda o: pa_decode(o, *pa_args, **pa_kw),
        "reuse_v1": lambda o: reuse(o, *args, **kv),
        "reuse_v2": lambda o: reuse(o, *args, tile_rows=2, impl="v2", **kv),
        # ATOM's call: v3, here on v2's two-plane DMAs (grids under min_ctas)
        "reuse_m31": lambda o: reuse(o, *args, **kv, **M31_PREFILL_CALL),
        "reuse_m31_kvrev": lambda o: reuse(o, *args_rev, **kv, **M31_PREFILL_CALL),
        # v3 itself (these grids are under its v2 fallback threshold)
        "reuse_v3": lambda o: reuse(
            o, *args, tile_rows=8, impl="v3", v3={"min_ctas": 0}, **kv
        ),
        "reuse_v3_tq4": lambda o: reuse(
            o, *args, tile_rows=4, impl="v3", v3={"min_ctas": 0}, **kv
        ),
        # a CTA ring of 2: the B rings get their own LDS frames
        "reuse_v3_nb2": lambda o: reuse(
            o, *args, tile_rows=8, impl="v3", nvfp4_buffers=2, v3={"min_ctas": 0}, **kv
        ),
        # a 3-slot B ring (each wave stages two quarters ahead)
        "reuse_v3_bb3": lambda o: reuse(
            o, *args, tile_rows=8, impl="v3", v3={"min_ctas": 0, "nb_b": 3}, **kv
        ),
        "reuse_v3_kvrev": lambda o: reuse(
            o, *args_rev, tile_rows=8, impl="v3", v3={"min_ctas": 0}, **kv
        ),
        "reuse_v3_nofold": lambda o: reuse(
            o,
            *args,
            tile_rows=8,
            impl="v3",
            v3={"min_ctas": 0, "tdm_fold": False},
            **kv,
        ),
    }
    ref, ret = run_torch(*host), {}
    for name, fn in candidates.items():
        out = torch.full_like(q, float("nan"))
        _, us = run_perftest(fn, out)
        err = checkAllclose(ref, out.cpu().float(), atol=3e-2, rtol=3e-2, msg=name)
        assert err == 0, f"{case} {name}: {err:.2%} mismatched"
        ret[f"{name} us"], ret[f"{name} err"] = us, err
    if kv_heads > 1:  # the MXFP8 output serves one KV head
        return ret
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


def test_v3_spread_scores(case="long", kv_heads=M31_KV_HEADS):
    """v3 (lazy softmax reference, FP16 P) on rows whose scores span many
    octaves: per-row query gains 2**-3 .. 2**6, from near-uniform to one-token
    softmax, so references jump far and P runs over FP16's whole range."""
    from aiter.ops.flydsl.pa_sparse_prefill import pa_sparse_prefill_nvfp4 as reuse

    host = list(make_case(CASES[case], kv_heads))
    g = torch.Generator().manual_seed(1)
    gain = torch.exp2(torch.rand(host[0].shape[0], 1, 1, generator=g) * 9 - 3)
    host[0] = (host[0].float() * gain).bfloat16()
    q, kp, vp, ks, vs, table, ctx = (t.cuda() for t in host)
    out = torch.full_like(q, float("nan"))
    v3 = {"min_ctas": 0}
    args = (q, kp, vp, ks, vs, table, ctx, SCALE)
    reuse(out, *args, tile_rows=8, impl="v3", v3=v3, kv_heads=kv_heads)
    err = checkAllclose(
        run_torch(*host), out.cpu().float(), atol=3e-2, rtol=3e-2, msg="v3 spread"
    )
    assert err == 0, f"{case} v3 spread scores: {err:.2%} mismatched"


@benchmark()
def bench_m31_prefill(case):
    """One M3.1 chunk as ATOM's P calls the kernel (``_flydsl_attention``,
    ``M31_PREFILL_CALL``, BF16 output) over 4 interleaved KV heads, every row
    against FP32."""
    from aiter.ops.flydsl.pa_sparse_prefill import M31_PREFILL_CALL
    from aiter.ops.flydsl.pa_sparse_prefill import pa_sparse_prefill_nvfp4 as reuse

    requests, trace_us = M31_CASES[case]
    host = make_case(requests, M31_KV_HEADS, swap=0.1, pad=0)
    q, kp, vp, ks, vs, table, ctx = (t.cuda() for t in host)
    del host
    out = torch.full_like(q, float("nan"))  # ATOM's output, a view like q
    args = (q, kp, vp, ks, vs, table, ctx, SCALE)
    _, us = run_perftest(
        reuse, out, *args, kv_heads=M31_KV_HEADS, **M31_PREFILL_CALL,
        num_rotate_args=1,
    )  # fmt: skip
    err = checkAllclose(
        run_torch(*args[:-1]), out.float(), atol=3e-2, rtol=3e-2, msg=case
    )
    assert err == 0, f"{case}: {err:.2%} mismatched"
    nbytes = 2 * q.nbytes + tile_pages(table, ctx, M31_KV_HEADS, 2) * PAGE_BYTES
    return {
        "reqs": len(requests),
        "new": sum(n for _, n in requests),
        "trace us": trace_us,
        "us": us,
        "TFLOPS": 4 * GROUP * D * ctx.sum().item() / us / 1e6,  # QK + PV
        "TB/s": nbytes / us / 1e6,
        "err": err,
    }


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
    parser.add_argument(
        "-k", "--kv_heads", type=int, nargs="*", default=[1, M31_KV_HEADS]
    )
    parser.add_argument(
        "-m", "--m31", nargs="*", default=list(M31_CASES), choices=M31_CASES
    )
    args = parser.parse_args()
    test_supported_gate()
    if get_gfx_runtime() != "gfx1250":
        aiter.logger.warning("NVFP4 sparse prefill runs on gfx1250 only; skipping")
        return
    rows = itertools.product(args.case, args.kv_heads)
    df = pd.DataFrame([test_pa_sparse_prefill_nvfp4(*r) for r in rows])
    aiter.logger.info("pa_sparse_prefill_nvfp4:\n%s", df.to_markdown(index=False))
    test_v3_spread_scores()
    if args.m31:
        df = pd.DataFrame([bench_m31_prefill(c) for c in args.m31])
        aiter.logger.info(
            "M3.1 PP4 x TP1 (4 KV heads):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
