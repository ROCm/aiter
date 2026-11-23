#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Check the ballot top-k against aiter's stable decode top-k, and time both.

Bit-exact: the same 512 indices in the same (ascending) order, and the same
values when values are requested, unsplit and split over 2, 4 and 8
workgroups. Times are CUDA-graph replays, as the server runs decode.

    python op_tests/op_benchmarks/hip/bench_topk_ballot.py

The kernel is csrc/kernels/topk_ballot_kernels.cu, which no aiter module
builds; build_so() compiles it with hipcc, once per source version.
"""

import argparse
import ctypes
import hashlib
import os
import subprocess
import tempfile

import torch

from aiter.ops.topk import top_k_per_row_decode

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
SOURCE = os.path.join(REPO, "csrc", "kernels", "topk_ballot_kernels.cu")
K = 512


def build_so(source=SOURCE):
    """Path to topk_ballot_kernels.cu built for this GPU, compiling it if the
    source or the GPU differ from the last build."""
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    with open(source, "rb") as f:
        digest = hashlib.sha1(f.read() + arch.encode()).hexdigest()[:12]
    out = os.path.join(tempfile.gettempdir(), f"topk_ballot_{arch}_{digest}.so")
    if not os.path.exists(out):
        tmp = f"{out}.{os.getpid()}"
        subprocess.run(
            ["hipcc", "-O3", f"--offload-arch={arch}", "-std=c++17", "-shared", "-fPIC",
             "-Wno-hip-deprecated-builtins", "-o", tmp, source],
            check=True,
        )  # fmt: skip
        os.replace(tmp, out)
    return out


def load(path):
    lib = ctypes.CDLL(path)
    fn = lib.topk_ballot_launch
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
    return fn


class Split:
    """topk_ballot_split_launch with a cached, zeroed workspace per shape."""

    def __init__(self, path):
        lib = ctypes.CDLL(path)
        self.launch = lib.topk_ballot_split_launch
        self.launch.restype = ctypes.c_int
        self.launch.argtypes = (
            [ctypes.c_void_p] * 4 + [ctypes.c_int] * 5 + [ctypes.c_void_p] * 2
        )
        self.parts_for = lib.topk_ballot_parts
        self.parts_for.restype = ctypes.c_int
        self.parts_for.argtypes = [ctypes.c_int, ctypes.c_int]
        self.bytes_for = lib.topk_ballot_workspace_bytes
        self.bytes_for.restype = ctypes.c_longlong
        self.bytes_for.argtypes = [ctypes.c_int, ctypes.c_int]
        self.ws = {}

    def __call__(self, logits, lens, idx, vals, next_n=1, parts=0):
        rows, stride = logits.shape[0], logits.stride(0)
        if parts <= 0:
            parts = self.parts_for(rows, stride)
        key = (rows, parts)
        if key not in self.ws:
            self.ws[key] = torch.zeros(
                max(self.bytes_for(rows, parts), 1), dtype=torch.uint8, device="cuda"
            )
        rc = self.launch(
            logits.data_ptr(),
            lens.data_ptr(),
            idx.data_ptr(),
            0 if vals is None else vals.data_ptr(),
            rows,
            stride,
            next_n,
            idx.shape[1],
            parts,
            self.ws[key].data_ptr(),
            torch.cuda.current_stream().cuda_stream,
        )
        if rc:
            raise RuntimeError(f"topk_ballot_split_launch returned {rc}")
        return parts


def ballot_call(fn, logits, lens, idx, vals, next_n=1):
    rc = fn(
        logits.data_ptr(),
        lens.data_ptr(),
        idx.data_ptr(),
        0 if vals is None else vals.data_ptr(),
        logits.shape[0],
        logits.stride(0),
        next_n,
        idx.shape[1],
        torch.cuda.current_stream().cuda_stream,
    )
    if rc:
        raise RuntimeError(f"topk_ballot_launch returned {rc}")


def aiter_call(logits, lens, idx, vals, next_n=1):
    top_k_per_row_decode(
        logits,
        next_n,
        lens,
        idx,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        k=idx.shape[1],
        stable=True,
        values=vals,
    )


def indexer_logits(rows, n, gen):
    # sum_h w_h * relu(q_h . k): the shape of the DeepSeek indexer score.
    heads, dim = 32, 128
    out = torch.empty(rows, n, device="cuda")
    keys = torch.randn(n, dim, device="cuda", generator=gen).to(torch.bfloat16)
    for r in range(rows):
        q = torch.randn(heads, dim, device="cuda", generator=gen).to(torch.bfloat16)
        w = torch.rand(heads, device="cuda", generator=gen)
        out[r] = torch.relu((keys @ q.T).float()) @ w
    return out


def make(dist, rows, n, gen):
    shape = (rows, n)
    if dist == "zeros":  # what the microbench feeds: its fp8 cache is torch.empty
        return torch.zeros(shape, device="cuda")
    if dist == "randn":
        return torch.randn(shape, device="cuda", generator=gen)
    if dist == "one_exp":
        return 1 + torch.rand(shape, device="cuda", generator=gen)
    if dist == "indexer":
        return indexer_logits(rows, n, gen)
    if dist == "ties8":
        return torch.randint(0, 8, shape, device="cuda", generator=gen).float()
    if dist == "ties_top":  # 1000 copies of the max, the rest below
        x = torch.rand(shape, device="cuda", generator=gen)
        pos = torch.randperm(n, device="cuda", generator=gen)[:1000]
        x[:, pos] = 2.0
        return x
    if dist == "half_zero":
        x = torch.randn(shape, device="cuda", generator=gen).abs()
        x[:, ::2] = 0
        return x
    if dist == "ascending":
        return torch.linspace(-5, 5, n, device="cuda").expand(rows, n).contiguous()
    if dist == "descending":
        return torch.linspace(5, -5, n, device="cuda").expand(rows, n).contiguous()
    if dist == "recent_hot":  # higher scores at the end of the row, like recency
        x = torch.randn(shape, device="cuda", generator=gen)
        return x + torch.linspace(0, 6, n, device="cuda")
    if dist == "specials":
        x = torch.randn(shape, device="cuda", generator=gen)
        m = torch.rand(shape, device="cuda", generator=gen)
        x[m < 0.01] = float("inf")
        x[(m >= 0.01) & (m < 0.02)] = float("-inf")
        x[(m >= 0.02) & (m < 0.03)] = 0.0
        x[(m >= 0.03) & (m < 0.04)] = -0.0
        x[(m >= 0.04) & (m < 0.045)] = float("nan")
        return x
    raise ValueError(dist)


def check(fn, split, dist, rows, n, lens_kind, next_n, with_values, gen):
    logits = make(dist, rows, n, gen)
    seqs = rows // next_n
    if lens_kind == "full":
        lens = torch.full((seqs,), n, device="cuda", dtype=torch.int32)
    else:  # ragged, including rows at or below k
        lo = max(next_n, K // 2)
        lens = torch.randint(
            lo, n + 1, (seqs,), device="cuda", generator=gen, dtype=torch.int32
        )
        lens[0] = K
        if seqs > 1:
            lens[1] = K + 1
    ref_i = torch.full((rows, K), -7, device="cuda", dtype=torch.int32)
    got_i = torch.full((rows, K), -7, device="cuda", dtype=torch.int32)
    ref_v = torch.zeros(rows, K, device="cuda") if with_values else None
    got_v = torch.zeros(rows, K, device="cuda") if with_values else None
    aiter_call(logits, lens, ref_i, ref_v, next_n)
    ok = True
    for parts in (1, 2, 4, 8):
        got_i.fill_(-7)
        if got_v is not None:
            got_v.zero_()
        if parts == 1:
            ballot_call(fn, logits, lens, got_i, got_v, next_n)
        else:
            split(logits, lens, got_i, got_v, next_n, parts)
        torch.cuda.synchronize()
        same = torch.equal(ref_i, got_i)
        if with_values:
            same = same and torch.equal(
                ref_v.view(torch.int32), got_v.view(torch.int32)
            )
        if not same:
            bad = (ref_i != got_i).any(1).nonzero().flatten().tolist()
            r = bad[0] if bad else 0
            print(
                f"  MISMATCH parts={parts} {dist} rows={rows} n={n} {lens_kind} next_n={next_n} values={with_values} rows {bad[:8]}"
            )
            print("   ref", ref_i[r, :12].tolist(), "...", ref_i[r, -4:].tolist())
            print("   got", got_i[r, :12].tolist(), "...", got_i[r, -4:].tolist())
        ok = ok and same
    return ok


def graph_us(fn, iters=200):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    g.replay()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters * 1000.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--so", default=None, help="a prebuilt topk_ballot.so; built if absent"
    )
    ap.add_argument("--skip-check", action="store_true")
    ap.add_argument("--widths", default="16384,50000,100000")
    ap.add_argument("--dists", default="zeros,indexer,randn,one_exp,ties8,recent_hot")
    args = ap.parse_args()
    so = args.so or build_so()
    fn = load(so)
    split = Split(so)
    gen = torch.Generator(device="cuda")
    gen.manual_seed(0)
    widths = [int(w) for w in args.widths.split(",")]

    if not args.skip_check:
        dists = [
            "zeros",
            "randn",
            "one_exp",
            "indexer",
            "ties8",
            "ties_top",
            "half_zero",
            "ascending",
            "descending",
            "recent_hot",
            "specials",
        ]
        n_ok = n_all = 0
        for n in widths + [600, 4100]:
            for dist in dists:
                for rows, lens_kind, next_n, vals in (
                    (1, "full", 1, False),
                    (8, "full", 1, True),
                    (8, "ragged", 1, False),
                    (8, "ragged", 2, True),
                ):
                    n_all += 1
                    n_ok += check(
                        fn, split, dist, rows, n, lens_kind, next_n, vals, gen
                    )
        print(
            f"bit-exact against aiter stable top-k, unsplit and 2/4/8 parts: {n_ok}/{n_all}"
        )

    print(
        f"{'dist':<11} {'width':>7} {'rows':>4} {'aiter us':>9} {'ballot us':>9} {'split us':>9} {'parts':>5} {'speedup':>7}"
    )
    for dist in args.dists.split(","):
        for n in widths:
            for rows in (1, 4, 8):
                logits = make(dist, rows, n, gen)
                lens = torch.full((rows,), n, device="cuda", dtype=torch.int32)
                i0 = torch.empty(rows, K, device="cuda", dtype=torch.int32)
                i1 = torch.empty_like(i0)
                i2 = torch.empty_like(i0)
                a = graph_us(lambda x=logits, n=lens, i=i0: aiter_call(x, n, i, None))
                b = graph_us(
                    lambda x=logits, n=lens, i=i1: ballot_call(fn, x, n, i, None)
                )
                parts = split(logits, lens, i2, None)
                c = graph_us(lambda x=logits, n=lens, i=i2: split(x, n, i, None))
                same = (
                    "" if torch.equal(i0, i1) and torch.equal(i0, i2) else "  MISMATCH"
                )
                print(
                    f"{dist:<11} {n:>7} {rows:>4} {a:>9.1f} {b:>9.1f} {c:>9.1f} {parts:>5} {a / min(b, c):>6.1f}x{same}"
                )


if __name__ == "__main__":
    main()
