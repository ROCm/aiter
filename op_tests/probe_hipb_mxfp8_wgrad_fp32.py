# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Probe: can hipBLASLt run the KDA MXFP8 wgrad natively with an fp32 output?

dW = dY^T X with dY [M, N] and X [M, K] bf16, both quantized along the tokens
(TE MXFP8 columnwise, ``scale_mode="te"``), as the qkv-sized wgrad of the
Kimi-K3 KDA layer: [N x M] x [M x K] = [12288 x 8192] x [8192 x 7168] by default.

Compares, on the same quantized operands:
  * ``hipblaslt``: ``hipb_mm`` with VEC32_UE8M0 scales and ``out_dtype=fp32``;
  * ``qdq``: the current path, a bf16 GEMM with fp32 output on the dequantized
    operands (exact in bf16);
  * ``emulate``: dequantized fp32 matmul.
Prints the fraction of bit-identical outputs, max |diff| and rel-L2 between
them, and the time of each path (operand quantization included).

    python op_tests/probe_hipb_mxfp8_wgrad_fp32.py [--n 12288 --m 8192 --k 7168]
"""

import argparse

import torch

from aiter.ops import mxfp8_linear as mx


def rel_l2(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def time_ms(fn, iters=10):
    fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--n", type=int, default=12288, help="output rows (dY columns)")
    p.add_argument("--m", type=int, default=8192, help="tokens (reduction)")
    p.add_argument("--k", type=int, default=7168, help="output columns (X columns)")
    args = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda"
    # per-channel magnitudes so the column blocks get different scales
    dy = (
        torch.randn(args.m, args.n, device=dev)
        * torch.logspace(-2, 2, args.n, device=dev)
    ).bfloat16()
    x = (
        torch.randn(args.m, args.k, device=dev)
        * torch.logspace(-1, 1, args.k, device=dev)
    ).bfloat16()
    lhs, rhs = dy.t(), x  # [N, M] @ [M, K]

    outs, times = {}, {}
    for name, fn in (
        ("hipblaslt", lambda: mx._hipb_mm_fp32(lhs, rhs)),
        ("qdq", lambda: mx.mxfp8_mm(lhs, rhs, impl="qdq")),
        ("emulate", lambda: mx.mxfp8_mm(lhs, rhs, impl="emulate")),
    ):
        try:
            outs[name] = fn()
            times[name] = time_ms(fn, iters=3 if name == "emulate" else 10)
        except Exception as err:  # noqa: BLE001 - a probe reports every failure
            print(f"{name}: FAILED {type(err).__name__}: {err}")
    for name, t in times.items():
        o = outs[name]
        print(f"{name:10s} dtype={o.dtype} shape={tuple(o.shape)} {t:8.3f} ms")
    if "hipblaslt" in outs and "qdq" in outs:
        h, q = outs["hipblaslt"], outs["qdq"]
        same = (h.view(torch.int32) == q.view(torch.int32)).float().mean().item()
        print(
            f"hipblaslt vs qdq: bit-identical {same:.4%}, max |diff| "
            f"{(h - q).abs().max().item():.3e}, rel-L2 {rel_l2(h, q):.3e}"
        )
    for name in ("hipblaslt", "qdq"):
        if name in outs and "emulate" in outs:
            print(
                f"{name} vs emulate: rel-L2 {rel_l2(outs[name], outs['emulate']):.3e}"
            )
    ok = "hipblaslt" in outs and outs["hipblaslt"].dtype == torch.float32
    print("VERDICT:", "fp32-out hipBLASLt MXFP8 GEMM works" if ok else "not available")


if __name__ == "__main__":
    main()
