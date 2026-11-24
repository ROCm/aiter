# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.m3_tp1_qknorm import qknorm_v3
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
HQ, HKV, HIQ, D, ROT, PAGE = 64, 4, 4, 128, 64, 16
ROWS = HQ + 2 * HKV + HIQ + 1  # q | k | v | index q | index k
EPS = 1e-6
FP8_MAX = 448.0


def make_layout(layout, T, gen):
    """Positions, KV slots and index slots [T] for the layout, and the 128-token
    blocks they span (8 SHUFFLE pages each)."""
    if layout == "ragged":  # 4 prompts behind cached prefixes; most start mid-page
        cuts = sorted({0, T} | {min(T, c) for c in (T // 8 + 3, T // 3, T // 2 + 7)})
        lens = [b - a for a, b in itertools.pairwise(cuts)]
        prefixes = [0, 50, 384, 8195][: len(lens)]
    else:  # one prompt; shift8 starts it 8 tokens into a page (no whole-page group)
        lens, prefixes = [T], [8 if layout == "shift8" else 0]
    blocks = [(p + n + 127) // 128 for p, n in zip(prefixes, lens)]
    tables = torch.randperm(sum(blocks), generator=gen).split(blocks)
    pos = [p + torch.arange(n) for p, n in zip(prefixes, lens)]
    slots = torch.cat([bt[q // 128] * 128 + q % 128 for bt, q in zip(tables, pos)])
    index_slots = slots.clone()
    if layout == "ragged":  # an index slot mapping of its own, padded tail
        index_slots = torch.randperm(sum(blocks) * 128, generator=gen)[:T]
        slots[T - min(T, 3) :] = -1
        index_slots[T - min(T, 3) :] = -1
    elif layout == "holes":  # a -1 slot inside every third page; index holes apart
        slots[7::48] = -1
        index_slots[19::80] = -1
    return torch.cat(pos), slots, index_slots, sum(blocks)


def run_torch(qkv, w, cos_sin, positions):
    """fp32 Gemma RMSNorm + partial NeoX RoPE of the q / k / index rows; raw v."""
    x = qkv.view(-1, ROWS, D).float()
    cs = cos_sin[positions].float()[:, None]
    cos, sin = cs[..., : ROT // 2], cs[..., ROT // 2 :]

    def norm_rope(h, weight):
        h = (
            h
            * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + EPS)
            * (1 + weight.float())
        )
        h1, h2 = h[..., : ROT // 2], h[..., ROT // 2 : ROT]
        return torch.cat([h1 * cos - h2 * sin, h2 * cos + h1 * sin, h[..., ROT:]], -1)

    return {
        "q": norm_rope(x[:, :HQ], w["q"]),
        "k": norm_rope(x[:, HQ : HQ + HKV], w["k"]),
        "v": x[:, HQ + HKV : HQ + 2 * HKV],
        "iq": norm_rope(x[:, HQ + 2 * HKV : ROWS - 1], w["iq"]),
        "ik": norm_rope(x[:, ROWS - 1 :], w["ik"])[:, 0],
    }


def check(name, out, ref, slots, index_slots):
    """Grade one candidate's outputs against the torch reference."""
    _, q_out, iq_out, kc, vc, ic, ks, vs = out
    err = checkAllclose(q_out.view(-1, HQ, D).float(), ref["q"], msg=f"{name} q: ")
    err += checkAllclose(iq_out.view(-1, HIQ, D).float(), ref["iq"], msg=f"{name} iq: ")
    tok = (slots >= 0).nonzero()[:, 0]
    page, t = slots[tok] // PAGE, slots[tok] % PAGE
    n = torch.arange(len(tok))
    k8 = kc[page].float()[n, :, :, t, :].reshape(-1, HKV, D)  # [d / 16][t][16] pages
    v8 = vc[page].float()[n, :, 0, :, t]  # [d][t] pages
    for x8, scale, r in ((k8, ks, ref["k"]), (v8, vs, ref["v"])):
        r, scale = r[tok], scale[page][n, :, t]
        amax = r.abs().amax(-1)  # of the fp32 values; K is rounded to bf16 after
        ref_scale = torch.where(amax > 0, amax / FP8_MAX, 1.0)
        err += checkAllclose(scale, ref_scale, rtol=1e-3, atol=0, msg=f"{name} scale: ")
        r = r.to(dtypes.bf16).float()
        err += checkAllclose(
            x8, r / scale[..., None], rtol=0.07, atol=2**-9, msg=f"{name} kv: "
        )
    itok = (index_slots >= 0).nonzero()[:, 0]
    ik = ref["ik"][itok].to(dtypes.bf16).float()
    ik8 = ic.view(-1, D)[index_slots[itok]].float()
    err += checkAllclose(ik8, ik, rtol=0.07, atol=2**-9, msg=f"{name} index k: ")
    assert err == 0, f"{name}: outputs differ from the torch reference"
    return err


@benchmark()
def test_qknorm_v3(num_tokens, layout, seed):
    T = num_tokens
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, ROWS, D, generator=gen)
    x = x * torch.exp(torch.randn(T, ROWS, 1, generator=gen))  # per-row magnitudes
    x[:, :, 7] *= 30  # an outlier channel
    qkv = x.reshape(T, ROWS * D).to(dtypes.bf16)
    w = {
        h: (0.1 * torch.randn(D, generator=gen)).to(dtypes.bf16)
        for h in ["q", "k", "iq", "ik"]
    }
    positions, slots, index_slots, blocks = make_layout(layout, T, gen)
    inv_freq = 1e7 ** (-torch.arange(0, ROT, 2).float() / ROT)
    angles = torch.arange(int(positions.max()) + 1).float()[:, None] * inv_freq
    cos_sin = torch.cat([angles.cos(), angles.sin()], -1).to(dtypes.bf16)

    def random_bytes(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, generator=gen)

    pages = blocks * (128 // PAGE)
    buffers = (  # outputs start as NaN, caches as random bytes, so a missed write shows
        qkv,
        torch.full((T, HQ * D), float("nan"), dtype=dtypes.bf16),
        torch.full((T, HIQ * D), float("nan"), dtype=dtypes.bf16),
        random_bytes(pages, HKV, D // PAGE, PAGE, PAGE).view(dtypes.fp8),
        random_bytes(pages, HKV, 1, D, PAGE).view(dtypes.fp8),
        random_bytes(blocks, 128, D).view(dtypes.fp8),
        torch.rand(pages, HKV, PAGE, generator=gen),
        torch.rand(pages, HKV, PAGE, generator=gen),
    )

    def ref_kernel(qkv, q_out, iq_out, kc, vc, ic, ks, vs):
        aiter.fused_qknorm_idxrqknorm(
            qkv, w["q"], w["k"], cos_sin, positions, HQ, HKV, ROT, EPS, w["iq"],
            w["ik"], HIQ, slots, kc, vc, ic, PAGE, q_out, iq_out, index_slots,
            kv_cache_dtype="fp8", index_cache_dtype="fp8", k_scale=ks, v_scale=vs,
            asm_layout=True,
        )  # fmt: skip

    def v3(qkv, q_out, iq_out, kc, vc, ic, ks, vs):
        qknorm_v3(
            qkv, q_out, iq_out, w["q"], w["k"], w["iq"], w["ik"], cos_sin, positions,
            slots, index_slots, kc, vc, ic, ks, vs, EPS, PAGE,
        )  # fmt: skip

    candidates = {"fused_qknorm_idxrqknorm": ref_kernel, "qknorm_v3": v3}
    ref = run_torch(qkv, w, cos_sin, positions)
    outs = {}
    for name, fn in candidates.items():
        outs[name] = tuple(t.clone() for t in buffers)
        fn(*outs[name])
    # qknorm_v3's contract: fused_qknorm_idxrqknorm's bytes, qkv left as it was
    for a, b in zip(*outs.values()):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), "bytes differ"
    assert torch.equal(outs["qknorm_v3"][0], qkv), "qkv was modified"

    # bytes per call: qkv in; q, index q, K / V / index K, K / V scales out
    nbytes = T * (ROWS * D * 2 + (HQ + HIQ) * D * 2 + (2 * HKV + 1) * D + 2 * HKV * 4)
    flops = T * (ROWS - HKV) * D * 6  # RMSNorm ~4, RoPE ~1.5, quant ~0.5 per element
    rotate = min(64, max(2, math.ceil(2**30 / sum(t.nbytes for t in buffers))))
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        err = check(name, outs[name], ref, slots, index_slots)
        _, us = run_perftest(fn, *outs[name], num_rotate_args=rotate)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("qknorm_v3 unsupported on %s; skipping", get_gfx_runtime())
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="qknorm_v3 vs fused_qknorm_idxrqknorm (MiniMax-M3.1 TP1 layout)",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[1, 77, 512, 1000, 3001, 8192, 16384],
        help="tokens per call, e.g. -t 8192",
    )
    parser.add_argument(
        "-l",
        "--layout",
        nargs="*",
        default=["aligned", "ragged", "shift8", "holes"],
        choices=["aligned", "ragged", "shift8", "holes"],
        help="aligned: one prompt from position 0; ragged: 4 prompts behind cached\n"
        "prefixes, own index slots, padded tail; shift8: one prompt 8 tokens into a page;\n"
        "holes: aligned with a -1 KV slot in every third page and other index-slot holes",
    )
    parser.add_argument("--seed", type=int, nargs="*", default=[0, 1, 2])
    args = parser.parse_args()
    df = [
        test_qknorm_v3(t, layout, seed)
        for t, layout, seed in itertools.product(args.tokens, args.layout, args.seed)
    ]
    df = pd.DataFrame(df)
    aiter.logger.info("qknorm_v3 summary (markdown):\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
