# SPDX-License-Identifier: MIT
# Scratch: opus a8w8_scale (128x128 BLOCKSCALE) vs CK batched_gemm_a8w8
# (ROWWISE: x_scale[b,m,1] * w_scale[b,1,n]) at the DSV4 wo_a batched shape.
# NOTE different quant schemes -> timing-only comparison (each vs its own ref).
# Isolates the "is opus's win just batching" question: batched_gemm_a8w8 is a
# single batched launch (no G-loop).

import argparse
import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.test_common import run_perftest
from aiter.ops.batched_gemm_op_a8w8 import batched_gemm_a8w8
from aiter.ops.opus.bmm_op import _opus_bmm_a8w8_scale_mmajor_raw

torch.set_default_device("cuda")
GROUP = 128


def _q_tok(x):  # [G,M,K]->fp8,[G,M,K/128] blockscale
    G, M, K = x.shape
    xb = x.to(dtypes.fp32).view(G, M, K // GROUP, GROUP)
    s = xb.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 448.0
    return (xb / s).clamp(-448, 448).to(dtypes.fp8).view(G, M, K), s.squeeze(-1).to(
        dtypes.fp32
    )


def _q_blk(w):  # [G,N,K]->fp8,[G,N/128,K/128]
    G, N, K = w.shape
    wb = w.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
    s = wb.abs().amax((2, 4), keepdim=True).clamp(min=1e-8) / 448.0
    return (wb / s).clamp(-448, 448).to(dtypes.fp8).view(G, N, K), s.view(
        G, N // GROUP, K // GROUP
    ).to(dtypes.fp32)


def run(m, n, k, g):
    act = (torch.rand((g, m, k), dtype=dtypes.fp32) / 10).to(dtypes.bf16)
    W = (torch.rand((g, n, k), dtype=dtypes.fp32) / 10).to(dtypes.bf16)

    # opus: blockscale (mmajor)
    Ofp8, xs = _q_tok(act)
    Wfp8, ws = _q_blk(W)
    Omm, xsmm = Ofp8.transpose(0, 1), xs.transpose(0, 1)

    def opus_a8w8_blockscale():
        Y = torch.empty((m, g, n), dtype=dtypes.fp32)
        _opus_bmm_a8w8_scale_mmajor_raw(Omm, Wfp8, Y, xsmm, ws)
        return Y

    # CK batched: rowwise (per-token x_scale[b,m,1], per-channel w_scale[b,1,n])
    Ork = (
        (
            act.to(dtypes.fp32)
            / act.to(dtypes.fp32).abs().amax(-1, keepdim=True).clamp(min=1e-8)
            * 448
        )
        .clamp(-448, 448)
        .to(dtypes.fp8)
    )
    Wrk = (
        (
            W.to(dtypes.fp32)
            / W.to(dtypes.fp32).abs().amax(-1, keepdim=True).clamp(min=1e-8)
            * 448
        )
        .clamp(-448, 448)
        .to(dtypes.fp8)
    )
    xs_rw = torch.rand((g, m, 1), dtype=dtypes.bf16)
    ws_rw = torch.rand((g, 1, n), dtype=dtypes.bf16)
    out = torch.empty((g, m, n), dtype=dtypes.bf16)

    def ck_batched_rowwise():
        batched_gemm_a8w8(Ork, Wrk, xs_rw, ws_rw, out)
        return out

    flops = 2 * m * g * n * k
    row = {"m": m, "n": n, "k": k, "g": g}
    for name, fn in (
        ("opus_a8w8_blockscale", opus_a8w8_blockscale),
        ("ck_batched_rowwise", ck_batched_rowwise),
    ):
        try:
            _, us = run_perftest(fn)
            row[f"{name} us"] = round(us, 2)
            row[f"{name} TFLOPS"] = round(flops / us / 1e6, 1)
        except Exception as e:
            row[f"{name} us"] = float("nan")
            row[f"{name} err"] = str(e)[:40]
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
    df = pd.DataFrame([run(*(int(x) for x in s.split(",")), a.b) for s in a.s])
    if "ck_batched_rowwise us" in df and "opus_a8w8_blockscale us" in df:
        df["opus/ck"] = (
            df["opus_a8w8_blockscale us"] / df["ck_batched_rowwise us"]
        ).round(2)
    aiter.logger.info(
        "opus blockscale vs CK batched rowwise (DSV4 wo_a):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
