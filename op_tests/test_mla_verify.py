# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA target verify (QLEN 8, 12 heads, FP8 KV): HK vs Gluon vs torch, host
kernel selection, and CUDA-graph replay of HK with changing lengths."""

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter.jit.utils.chip_info import get_gfx
from aiter.mla_verify import (
    MLA_VERIFY_GLUON,
    MLA_VERIFY_HK,
    mla_verify_hk,
    mla_verify_hk_threshold,
    select_mla_verify_kernel,
)
from aiter.ops.triton.gluon.mla_gluon import mla_gluon
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
H, QLEN, D, DV = 12, 8, 576, 512
SCALE = D**-0.5
KV_SCALE = 0.75


def run_torch(q, kv, lens, slots):
    # Reference only: fp32 math with the causal tail mask of target verify.
    kv = kv.view(-1, D).float() * KV_SCALE
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


def run_gluon(q, kv, slots, indptr, out):
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
        kv_scale=KV_SCALE,
    )
    return out


@benchmark()
def test_mla_verify(bs, ctx):
    # Uneven lengths: request b has ctx * (b + 1) / bs tokens incl. the 8 drafts.
    lens = [max(QLEN, ctx * (b + 1) // bs) for b in range(bs)]
    q, kv, slots, indptr = make_inputs(lens, seed=bs * ctx)
    kv_scale = torch.tensor([KV_SCALE], dtype=torch.float32)
    ref = run_torch(q, kv, lens, slots.long())
    candidates = {
        MLA_VERIFY_HK: lambda out: mla_verify_hk(
            q, kv, indptr, slots, SCALE, kv_scale, out=out
        ),
        MLA_VERIFY_GLUON: lambda out: run_gluon(q, kv, slots, indptr, out),
    }
    flops = 2 * QLEN * H * (D + DV) * sum(lens)
    nbytes = sum(lens) * D + q.numel() * 2 * (1 + DV / D)
    ret = {
        "gfx": get_gfx(),
        "max_kv_len": max(lens),
        "selected": select_mla_verify_kernel(bs, max(lens)),
        "threshold": mla_verify_hk_threshold(bs),
    }
    for name, fn in candidates.items():
        out_buf = torch.empty(q.shape[0], H, DV, dtype=torch.bfloat16)
        out, us = run_perftest(fn, out_buf)
        err = checkAllclose(
            ref.float(), out.float(), rtol=2e-2, atol=2e-2, msg=f"{name}: mla_verify"
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_mla_verify_hk_graph(bs, ctx):
    # Capture once at the largest lengths, then replay with shorter and uneven
    # lengths written into the same index buffers, as a decode CUDA graph does.
    # Replay must match an eager call on the same lengths bit for bit.
    max_lens = [ctx] * bs
    q, kv, slots, indptr = make_inputs(max_lens, seed=bs)
    kv_scale = torch.tensor([KV_SCALE], dtype=torch.float32)
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
        ret[f"{name} err"] = checkAllclose(
            eager.float(), out.float(), rtol=0, atol=0, msg=f"graph {name} vs eager"
        )
        ref = run_torch(q, kv, lens, slots.long())
        ret[f"{name} ref err"] = checkAllclose(
            ref.float(), out.float(), rtol=2e-2, atol=3e-2, msg=f"graph {name}"
        )
    return ret


def check_selection():
    for bs in (1, 2, 4, 8, 16, 32):
        thresh = mla_verify_hk_threshold(bs)
        assert thresh is not None and thresh > 0
        below = -(-thresh // bs) - 1
        assert select_mla_verify_kernel(bs, below) == MLA_VERIFY_GLUON, bs
        assert select_mla_verify_kernel(bs, -(-thresh // bs)) == MLA_VERIFY_HK, bs
    big = 1 << 20
    assert select_mla_verify_kernel(1, big, num_heads=16) == MLA_VERIFY_GLUON
    assert select_mla_verify_kernel(1, big, qlen=4) == MLA_VERIFY_GLUON
    assert select_mla_verify_kernel(1, big, kv_dtype=torch.bfloat16) == MLA_VERIFY_GLUON
    assert select_mla_verify_kernel(1, big, page_size=64) == MLA_VERIFY_GLUON
    assert select_mla_verify_kernel(0, big) == MLA_VERIFY_GLUON


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

    check_selection()
    aiter.logger.info("select_mla_verify_kernel: ok")

    df = pd.DataFrame(
        [test_mla_verify(b, c) for b, c in itertools.product(args.batch, args.ctx)]
    )
    aiter.logger.info("mla_verify summary (markdown):\n%s", df.to_markdown(index=False))
    df = pd.DataFrame(
        [
            test_mla_verify_hk_graph(b, c)
            for b, c in itertools.product(args.batch, [min(args.ctx), max(args.ctx)])
        ]
    )
    aiter.logger.info(
        "mla_verify_hk graph replay summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
