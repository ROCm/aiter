# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for the HipKittens sparse MLA (DSA) forward against an fp32 torch reference.

Covers split-K decode (small T) and prefill, kpool-tail index padding, masked keys (-1 and slots past the pool),
dense 512-byte rows and a KV pool past 4 GiB (64-bit gather addresses).
"""

import argparse
import sys

import torch

import aiter
from aiter.ops.hk_sparse_mla import DIM, HEADS, hk_sparse_mla_fwd
from aiter.test_common import perftest

FP8 = torch.float8_e4m3fn
# fp8 q / kv rounding is shared with the reference; the rest is fp8 P (x448) and the bf16 output.
TOL = 0.04


def make_inputs(T, num_slots, topk_real, topk, row_bytes):
    """Random fp8 q / kv; topk_real random indices, then 3 live tail slots and -1 padding up to topk."""
    q = (torch.randn(T, HEADS, DIM, device="cuda") * 0.4).to(FP8)
    kv = (torch.randn(num_slots, 1, row_bytes, device="cuda") * 0.4).to(FP8)
    idx = torch.randint(0, num_slots, (T, topk), device="cuda", dtype=torch.int32)
    idx[:, topk_real:] = -1
    if topk > topk_real:
        idx[:, topk_real : topk_real + 3] = torch.randint(
            0, num_slots, (T, 3), device="cuda", dtype=torch.int32
        )
    idx[:, ::7] = -1  # scattered masked keys
    return q, kv, idx


def reference(q, kv, idx, sm_scale, rows):
    out = []
    for t in rows:
        valid = (idx[t] >= 0) & (idx[t] < kv.shape[0])
        k = kv[idx[t].clamp(0, kv.shape[0] - 1).long(), 0, :DIM].float()
        s = (q[t].float() @ k.T) * sm_scale
        s[:, ~valid] = -float("inf")
        out.append(torch.softmax(s, -1) @ k)
    return torch.stack(out)


def rel_err(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


@perftest()
def run(q, kv, idx, sm_scale):
    return hk_sparse_mla_fwd(q, kv, idx, sm_scale)


def test_case(T, topk_real, topk, row_bytes=576, num_slots=16384):
    torch.manual_seed(T + topk)
    q, kv, idx = make_inputs(T, num_slots, topk_real, topk, row_bytes)
    sm_scale = DIM**-0.5
    out, us = run(q, kv, idx, sm_scale)
    rows = list(range(0, T, max(1, T // 16)))[:16]
    err = rel_err(out[rows], reference(q, kv, idx, sm_scale, rows))
    ok = err < TOL and not torch.isnan(out).any()
    aiter.logger.info(
        f"hk_sparse_mla T={T:6d} topk={topk:4d} row={row_bytes}B: rel err {err:.4f} {us:8.1f} us  "
        f"{'ok' if ok else 'FAIL'}"
    )
    return ok


def test_out_of_pool_slots_masked():
    """Slots past the pool are masked like -1 (no out-of-bounds reads)."""
    q, kv, idx = make_inputs(16, 1024, 2048, 2048, 576)
    far, masked = idx.clone(), idx.clone()
    far[:, 1::5] = torch.where(idx[:, 1::5] >= 0, 1024 + idx[:, 1::5], idx[:, 1::5])
    masked[:, 1::5] = -1
    ok = torch.equal(
        hk_sparse_mla_fwd(q, kv, far, DIM**-0.5),
        hk_sparse_mla_fwd(q, kv, masked, DIM**-0.5),
    )
    aiter.logger.info(
        f"hk_sparse_mla out-of-pool slots masked: {'ok' if ok else 'FAIL'}"
    )
    return ok


def test_large_pool():
    """A pool past 4 GiB: slots on both sides of the 4 GiB mark."""
    num_slots = (5 << 30) // 576
    torch.manual_seed(5)
    q = (torch.randn(64, HEADS, DIM, device="cuda") * 0.4).to(FP8)
    kv = torch.empty(num_slots, 1, 576, dtype=FP8, device="cuda")
    kv[:8192] = (torch.randn(8192, 1, 576, device="cuda") * 0.4).to(FP8)
    kv[-8192:] = (torch.randn(8192, 1, 576, device="cuda") * 0.4).to(FP8)
    idx = torch.randint(
        num_slots - 8192, num_slots, (64, 2048), device="cuda", dtype=torch.int32
    )
    idx[:, ::3] = torch.randint(0, 8192, (64, 683), device="cuda", dtype=torch.int32)
    out = hk_sparse_mla_fwd(q, kv, idx, DIM**-0.5)
    rows = list(range(0, 64, 4))
    err = rel_err(out[rows], reference(q, kv, idx, DIM**-0.5, rows))
    del kv
    aiter.logger.info(
        f"hk_sparse_mla 5 GiB pool: rel err {err:.4f} {'ok' if err < TOL else 'FAIL'}"
    )
    return err < TOL


def test_rejects():
    q, kv, idx = make_inputs(4, 1024, 2048, 2048, 576)
    ok = True
    for bad in (
        lambda: hk_sparse_mla_fwd(
            q, kv, torch.zeros(4, 2432, dtype=torch.int32, device="cuda"), 1.0
        ),
        lambda: hk_sparse_mla_fwd(q[:, :16], kv, idx, 1.0),
    ):
        try:
            bad()
            ok = False
        except ValueError:
            pass
    aiter.logger.info(
        f"hk_sparse_mla rejects unsupported shapes: {'ok' if ok else 'FAIL'}"
    )
    return ok


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="HipKittens sparse MLA (DSA) forward: correctness vs fp32 and timing",
)
parser.add_argument(
    "-t", "--tokens", type=int, nargs="*", default=None, help="Query tokens"
)
parser.add_argument(
    "--no-large-pool", action="store_true", help="skip the 5 GiB pool case"
)
args = parser.parse_args()

if aiter.get_gfx() != "gfx950":
    print(f"Skipping test_hk_sparse_mla.py: requires gfx950, got {aiter.get_gfx()}")
    sys.exit(0)

if args.tokens:
    results = [test_case(T, 2048, 2048) for T in args.tokens]
else:
    results = [
        test_case(1, 2048, 2048),  # decode: split-K + combine
        test_case(8, 2048, 2048),
        test_case(32, 2048, 2051),  # kpool tail, padded to a 128 multiple by the op
        test_case(128, 2048, 2048),
        test_case(512, 2048, 2048),  # no split
        test_case(4096, 2048, 2176),
        test_case(16384, 2048, 2048, row_bytes=512),  # prefill, dense rows
        test_case(64, 1024, 1024),  # short context
        test_out_of_pool_slots_masked(),
        test_rejects(),
    ]
    if not args.no_large_pool:
        results.append(test_large_pool())
sys.exit(0 if all(results) else 1)
