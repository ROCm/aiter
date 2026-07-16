# SPDX-License-Identifier: MIT
# Scratch: opus fp8 blockscale (a8w8_scale + uniform_scale) vs CK fp8 blockscale
# for the DSV4 wo_a batched shape [G,M,K]x[G,N,K]. hipBLASLt has no fp8
# blockscale, so CK (gemm_a8w8_blockscale) is the library reference. CK is 2D,
# so it is looped over the G batches (one launch each); opus is one batched
# launch (grid.z=batch). Reference = dequant fp32 einsum.

import argparse

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale
from aiter.test_common import checkAllclose, run_perftest
from aiter.ops.opus.gemm_op_a16w16 import (
    _opus_gemm_a8w8_scale_mmajor_raw,
    _opus_gemm_uniform_scale_mmajor_raw,
)

torch.set_default_device("cuda")
GROUP = 128


def _q_tok(x):  # [G,M,K] -> fp8, [G,M,K/128]
    G, M, K = x.shape
    xb = x.to(dtypes.fp32).view(G, M, K // GROUP, GROUP)
    s = xb.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 448.0
    return (xb / s).clamp(-448, 448).to(dtypes.fp8).view(G, M, K), s.squeeze(-1).to(
        dtypes.fp32
    )


def _q_blk(w):  # [G,N,K] -> fp8, [G,N/128,K/128]
    G, N, K = w.shape
    wb = w.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
    s = wb.abs().amax((2, 4), keepdim=True).clamp(min=1e-8) / 448.0
    return (wb / s).clamp(-448, 448).to(dtypes.fp8).view(G, N, K), s.view(
        G, N // GROUP, K // GROUP
    ).to(dtypes.fp32)


def ref(act, W, xs, ws):  # [G,M,N] fp32
    G, M, K = act.shape
    N = W.shape[1]
    Od = (act.to(dtypes.fp32).view(G, M, K // GROUP, GROUP) * xs.unsqueeze(-1)).view(
        G, M, K
    )
    Wd = (
        W.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
        * ws.view(G, N // GROUP, 1, K // GROUP, 1)
    ).view(G, N, K)
    return torch.einsum("gmk,gnk->gmn", Od, Wd)


def run(m, n, k, g):
    act = (torch.rand((g, m, k), dtype=dtypes.fp32) / 10).to(dtypes.bf16)
    W = (torch.rand((g, n, k), dtype=dtypes.fp32) / 10).to(dtypes.bf16)
    Ofp8, xs = _q_tok(act)
    Wfp8, ws = _q_blk(W)
    R = ref(Ofp8, Wfp8, xs, ws)  # [g,m,n]

    Omm = Ofp8.transpose(0, 1)  # [m,g,k] view (mmajor)
    xsmm = xs.transpose(0, 1)  # [m,g,k/128] view

    def opus_a8w8():
        Y = torch.empty((m, g, n), dtype=dtypes.fp32)
        _opus_gemm_a8w8_scale_mmajor_raw(Omm, Wfp8, Y, xsmm, ws)
        return Y

    def opus_uniform():
        Y = torch.empty((m, g, n), dtype=dtypes.fp32)
        _opus_gemm_uniform_scale_mmajor_raw(Omm, Wfp8, Y, xsmm, ws, 700)
        return Y

    # CK: 2D per-batch, loop over G (one launch each), stack.
    Ofp8_c = Ofp8.contiguous()

    def ck():
        outs = [
            gemm_a8w8_blockscale(Ofp8_c[i], Wfp8[i], xs[i], ws[i], dtypes.bf16)
            for i in range(g)
        ]
        return torch.stack(outs, 0)  # [g,m,n]

    flops = 2 * m * g * n * k
    row = {"m": m, "n": n, "k": k, "g": g}
    for name, fn, rf in (
        ("opus_a8w8", opus_a8w8, R.transpose(0, 1)),
        ("opus_uniform700", opus_uniform, R.transpose(0, 1)),
        ("ck_loop", ck, R),
    ):
        out, us = run_perftest(fn)
        err = checkAllclose(
            rf.to(dtypes.fp32), out.to(dtypes.fp32), rtol=5e-2, atol=5e-2, msg=name
        )
        row[f"{name} us"] = round(us, 2)
        row[f"{name} TFLOPS"] = round(flops / us / 1e6, 1)
        row[f"{name} err"] = err
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-b", type=int, default=8)
    p.add_argument(
        "-s",
        nargs="*",
        default=[
            "256,1024,4096",
            "512,1024,4096",
            "1024,1024,4096",
            "2048,1024,4096",
            "4096,1024,4096",
        ],
    )
    a = p.parse_args()
    df = [run(*(int(x) for x in s.split(",")), a.b) for s in a.s]
    df = pd.DataFrame(df)
    ck = df["ck_loop us"]
    df["a8w8/ck"] = (df["opus_a8w8 us"] / ck).round(2)
    aiter.logger.info(
        "opus fp8 vs CK fp8 blockscale (DSV4 wo_a batched):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
