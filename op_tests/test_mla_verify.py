# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA target verify (QLEN 8, 12 heads, FP8 KV): HK (direct and through
mla_decode_fwd) vs Gluon vs torch, and CUDA-graph replay of HK with changing
lengths."""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.mla import mla_decode_fwd
from aiter.mla_verify import mla_verify_hk
from aiter.ops.triton.gluon.mla_gluon import mla_gluon
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
H, QLEN, D, DV = 12, 8, 576, 512
SCALE = D**-0.5
# SGLang passes 1.0 for Kimi-K3; 0.75 checks that the scale is applied.
KV_SCALES = (0.75, 1.0)
REF_RTOL, REF_ATOL = 2e-2, 3e-2
# The fp8 Q/P rounding moves a few elements of very short requests (KV length 9-24,
# kv_scale 1.0) past REF_ATOL, up to ~0.045 (asm: 0.031 on the same kind of batch);
# a wrong kernel moves far more.
REF_MAX_BAD_RATIO = 1e-4
REF_MAX_DELTA = 6e-2


def assert_close_to_ref(ref, out, what):
    delta = (ref.float() - out.float()).abs()
    bad = (delta > REF_ATOL + REF_RTOL * ref.float().abs()).float().mean().item()
    max_delta = delta.max().item()
    assert (
        bad <= REF_MAX_BAD_RATIO and max_delta <= REF_MAX_DELTA
    ), f"{what}: {bad:.2e} of elements outside tolerance, max delta {max_delta:.4f}"


def run_torch(q, kv, lens, slots, kv_scale):
    # Reference only: fp32 math with the causal tail mask of target verify.
    kv = kv.view(-1, D).float() * kv_scale
    out, start = [], 0
    for b, kv_len in enumerate(lens):
        k = kv[slots[start : start + kv_len]]
        start += kv_len
        qb = q[b * QLEN : (b + 1) * QLEN].float()
        s = torch.einsum("qhd,kd->hqk", qb, k) * SCALE
        visible = torch.arange(kv_len) <= (kv_len - QLEN + torch.arange(QLEN)[:, None])
        p = s.masked_fill(~visible, float("-inf")).softmax(-1)
        out.append(torch.einsum("hqk,kd->qhd", p, k[:, :DV]))
    return torch.cat(out).to(q.dtype)


def make_inputs(lens, seed=0):
    torch.manual_seed(seed)
    total = sum(lens)
    # Paged KV pool with page size 1, as the model's MLA KV buffer [N, 1, 576].
    kv = (torch.randn(total + 128, 1, D) * 0.5).to(torch.float8_e4m3fn)
    slots = torch.randperm(total + 128)[:total].to(torch.int32)
    indptr = torch.tensor([0] + list(lens)).cumsum(0).to(torch.int32)
    q = torch.randn(len(lens) * QLEN, H, D, dtype=torch.bfloat16)
    return q, kv, slots, indptr


def run_gluon(q, kv, slots, indptr, out, kv_scale):
    bs = q.shape[0] // QLEN
    q4 = q.view(bs, QLEN, H, D)
    mla_gluon(
        q4[..., :DV],
        q4[..., DV:],
        kv.view(-1, D),
        out.view(bs, QLEN, H, DV),
        slots,
        indptr,
        SCALE,
        kv_pe_offset=DV,
        use_2d_view=False,
        kv_scale=kv_scale,
    )
    return out


def run_decode_fwd(q, kv, out, qo_indptr, indptr, slots, last_page_lens, kv_scale):
    mla_decode_fwd(
        q,
        kv,
        out,
        qo_indptr,
        indptr,
        slots,
        last_page_lens,
        QLEN,
        sm_scale=SCALE,
        kv_scale=kv_scale,
    )
    return out


@benchmark()
def test_mla_verify(bs, ctx, kv_scale_val):
    # Uneven lengths: request b has ctx * (b + 1) / bs tokens incl. the 8 drafts.
    lens = [max(QLEN, ctx * (b + 1) // bs) for b in range(bs)]
    q, kv, slots, indptr = make_inputs(lens, seed=bs * ctx)
    kv_scale = torch.tensor([kv_scale_val], dtype=torch.float32)
    ref = run_torch(q, kv, lens, slots.long(), kv_scale_val)
    qo_indptr = torch.arange(bs + 1, dtype=torch.int32) * QLEN
    last_page_lens = torch.ones(bs, dtype=torch.int32)
    candidates = {
        "hk": lambda out: mla_verify_hk(q, kv, indptr, slots, SCALE, kv_scale, out=out),
        "mla_decode_fwd": lambda out: run_decode_fwd(
            q, kv, out, qo_indptr, indptr, slots, last_page_lens, kv_scale
        ),
        "gluon": lambda out: run_gluon(q, kv, slots, indptr, out, kv_scale_val),
    }
    flops = 2 * QLEN * H * (D + DV) * sum(lens)
    nbytes = sum(lens) * D + q.numel() * 2 * (1 + DV / D)
    ret = {
        "gfx": get_gfx(),
        "max_kv_len": max(lens),
    }
    outs = {}
    for name, fn in candidates.items():
        out_buf = torch.empty(q.shape[0], H, DV, dtype=torch.bfloat16)
        out, us = run_perftest(fn, out_buf)
        err = checkAllclose(
            ref.float(),
            out.float(),
            rtol=REF_RTOL,
            atol=REF_ATOL,
            msg=f"{name}: mla_verify kv_scale={kv_scale_val}",
        )
        assert_close_to_ref(
            ref, out, f"{name} bs={bs} ctx={ctx} kv_scale={kv_scale_val}"
        )
        outs[name] = out
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    # mla_decode_fwd routes this shape to the HK kernel
    assert torch.equal(outs["mla_decode_fwd"], outs["hk"]), f"bs={bs} ctx={ctx}"
    return ret


@benchmark()
def test_mla_verify_hk_graph(bs, ctx, kv_scale_val):
    # Capture once at the largest lengths, then replay with shorter and uneven
    # lengths written into the same index buffers, as a decode CUDA graph does.
    # Replay must match an eager call on the same lengths bit for bit.
    max_lens = [ctx] * bs
    q, kv, slots, indptr = make_inputs(max_lens, seed=bs)
    kv_scale = torch.tensor([kv_scale_val], dtype=torch.float32)
    out = torch.empty(q.shape[0], H, DV, dtype=torch.bfloat16)
    mla_verify_hk(q, kv, indptr, slots, SCALE, kv_scale, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mla_verify_hk(q, kv, indptr, slots, SCALE, kv_scale, out=out)
    ret = {"gfx": get_gfx()}
    replays = {
        "full": max_lens,
        "short": [QLEN + 1 + b for b in range(bs)],
        "uneven": [max(QLEN, (ctx * (b + 1)) // (2 * bs)) for b in range(bs)],
    }
    for name, lens in replays.items():
        new_indptr = torch.tensor([0] + lens).cumsum(0).to(torch.int32)
        indptr.copy_(new_indptr)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        eager = mla_verify_hk(q, kv, indptr, slots, SCALE, kv_scale)
        assert torch.equal(eager, out), f"graph {name} bs={bs} ctx={ctx} != eager"
        ref = run_torch(q, kv, lens, slots.long(), kv_scale_val)
        ret[f"{name} ref err"] = checkAllclose(
            ref.float(), out.float(), rtol=REF_RTOL, atol=REF_ATOL, msg=f"graph {name}"
        )
        assert_close_to_ref(
            ref, out, f"graph {name} bs={bs} ctx={ctx} kv_scale={kv_scale_val}"
        )
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("mla_verify_hk unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MLA target verify (QLEN 8, 12 heads, FP8 KV)",
    )
    parser.add_argument(
        "-b", "--batch", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument(
        "-c",
        "--ctx",
        type=int,
        nargs="*",
        default=[512, 2048, 8192, 16384, 32768, 131072],
        help="longest request KV length incl. the 8 draft tokens",
    )
    args = parser.parse_args()

    df = pd.DataFrame(
        [
            test_mla_verify(b, c, s)
            for b, c, s in itertools.product(args.batch, args.ctx, KV_SCALES)
        ]
    )
    aiter.logger.info("mla_verify summary (markdown):\n%s", df.to_markdown(index=False))
    df = pd.DataFrame(
        [
            test_mla_verify_hk_graph(b, c, s)
            for b, c, s in itertools.product(
                args.batch, [min(args.ctx), max(args.ctx)], KV_SCALES
            )
        ]
    )
    aiter.logger.info(
        "mla_verify_hk graph replay summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
