# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic ("stable") top_k_per_row correctness.

The stable contract for DSA + tensor parallel: identical input -> identical,
ascending-index-ordered, smallest-index tie-broken output, run after run and
byte-identical across ranks.

    python op_tests/test_topk_per_row_stable.py
"""

import argparse

import torch

from aiter.ops.topk import top_k_per_row_decode, top_k_per_row_prefill


def _twiddle_float(bits: int) -> int:
    """Mirror the kernel's twiddle_in for float: order-preserving unsigned key
    where ascending key == descending value, distinguishing -0.0 < +0.0."""
    bits &= 0xFFFFFFFF
    mask = 0 if (bits >> 31) else 0x7FFFFFFF
    return bits ^ mask


def ref_stable_topk(row_vals: torch.Tensor, k: int) -> list:
    """Bit-faithful reference: rank by the same twiddle_in key (ties -> smaller
    index), then output ascending by index."""
    n = row_vals.numel()
    kk = min(k, n)
    raw = row_vals.detach().cpu().contiguous().view(torch.int32).numpy()
    keys = [(_twiddle_float(int(raw[i])), i) for i in range(n)]
    order = sorted(range(n), key=lambda i: keys[i])
    return sorted(order[:kk])


def make_logits(num_rows, seq, tie_level, seed):
    torch.manual_seed(seed)
    x = torch.randn(num_rows, seq, dtype=torch.float32, device="cuda")
    if tie_level == "none":
        return x
    levels = 8 if tie_level == "heavy" else 64
    return (x * levels).round() / levels


def check_prefill(num_rows, seq, k, tie_level):
    logits = make_logits(num_rows, seq, tie_level, seed=123)
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_ends = torch.full((num_rows,), seq, dtype=torch.int32, device="cuda")

    def run():
        idx = torch.empty((num_rows, k), dtype=torch.int32, device="cuda")
        top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            idx,
            None,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            k=k,
            stable=True,
        )
        torch.cuda.synchronize()
        return idx

    a, b = run(), run()
    det = torch.equal(a, b)
    kk = min(k, seq)
    ok_order = all(
        a[r][:kk].cpu().tolist() == sorted(a[r][:kk].cpu().tolist())
        for r in range(num_rows)
    )
    ok_ref = all(
        a[r][:kk].cpu().tolist() == ref_stable_topk(logits[r], k)
        for r in range(num_rows)
    )
    tag = f"prefill nr={num_rows} seq={seq} k={k} tie={tie_level}"
    print(f"[{tag}] deterministic={det} ascending={ok_order} matches_ref={ok_ref}")
    return det and ok_order and ok_ref


def check_prefill_many_tied_rows():
    """Cover the cross-wave tile-base handoff in the stable emitters.

    A 1024-thread block spans multiple wavefronts.  Before the tile-base
    barrier was added, thread 0 could advance the shared base while a later
    wave was still reading it, leaving output slots unwritten.  Repeating a
    high-concurrency tied shape exercises different wave scheduling; the
    negative sentinel makes any exact-cardinality failure immediately visible.
    """
    num_rows, seq, k = 512, 40000, 2048
    logits = make_logits(num_rows, seq, "mild", seed=777)
    row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
    row_ends = torch.full((num_rows,), seq, dtype=torch.int32, device="cuda")

    def run():
        idx = torch.full(
            (num_rows, k), -123456789, dtype=torch.int32, device="cuda"
        )
        top_k_per_row_prefill(
            logits,
            row_starts,
            row_ends,
            idx,
            None,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            k=k,
            stable=True,
        )
        torch.cuda.synchronize()
        return idx

    a = run()
    valid = bool(torch.all((a >= 0) & (a < seq)))
    det = True
    ascending = bool(torch.all(a[:, 1:] >= a[:, :-1]))
    for _ in range(15):
        b = run()
        valid &= bool(torch.all((b >= 0) & (b < seq)))
        det &= torch.equal(a, b)
        ascending &= bool(torch.all(b[:, 1:] >= b[:, :-1]))
    # Full CPU sorting of 512 x 40K is unnecessary; sample rows on both sides
    # of the old concurrency boundary for bit-faithful reference coverage.
    sample_rows = (0, 255, 256, num_rows - 1)
    matches_ref = all(
        a[r].cpu().tolist() == ref_stable_topk(logits[r], k)
        for r in sample_rows
    )
    print(
        "[prefill many tied rows] "
        f"valid={valid} deterministic={det} ascending={ascending} "
        f"matches_ref={matches_ref}"
    )
    return valid and det and ascending and matches_ref


def check_decode(batch, ctx, k, next_n, tie_level):
    num_rows = batch * next_n
    seq_lens = torch.full((batch,), ctx, dtype=torch.int32, device="cuda")
    row_idx = torch.arange(num_rows, device="cuda") // next_n
    off = torch.arange(num_rows, device="cuda") % next_n
    row_ends = seq_lens[row_idx] - next_n + off + 1
    logits = make_logits(num_rows, ctx, tie_level, seed=321)
    for i in range(num_rows):
        logits[i, row_ends[i] :] = float("-inf")

    def run():
        idx = torch.empty((num_rows, k), dtype=torch.int32, device="cuda")
        top_k_per_row_decode(
            logits,
            next_n,
            seq_lens,
            idx,
            num_rows,
            logits.stride(0),
            logits.stride(1),
            k=k,
            stable=True,
        )
        torch.cuda.synchronize()
        return idx

    a, b = run(), run()
    det = torch.equal(a, b)
    ok_order = True
    ok_ref = True
    for r in range(num_rows):
        rlen = int(row_ends[r].item())
        kk = min(k, rlen)
        row = a[r][:kk].cpu().tolist()
        if row != sorted(row):
            ok_order = False
        if row != ref_stable_topk(logits[r][:rlen], k):
            ok_ref = False
    tag = f"decode b={batch} ctx={ctx} k={k} n={next_n} tie={tie_level}"
    print(f"[{tag}] deterministic={det} ascending={ok_order} matches_ref={ok_ref}")
    return det and ok_order and ok_ref


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=None, help="top-k (default: sweep)")
    args = parser.parse_args()

    ks = [args.k] if args.k else [512, 1024, 2048]
    all_ok = True
    for tie in ("none", "mild", "heavy"):
        for k in ks:
            # short row -> scan path; long row -> fast (BlockRadixSort) path
            all_ok &= check_prefill(4, 4096, k, tie)
            all_ok &= check_prefill(2, 61440, k, tie)
            all_ok &= check_prefill(8, 1000, k, tie)  # row_len < k edge
        for k in ks:
            all_ok &= check_decode(4, 4096, k, 1, tie)
            all_ok &= check_decode(4, 61440, k, 1, tie)
    # k > 2048 -> scan fallback
    for tie in ("none", "heavy"):
        all_ok &= check_prefill(2, 16384, 4096, tie)
        all_ok &= check_decode(2, 16384, 4096, 1, tie)
    all_ok &= check_prefill_many_tied_rows()
    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    assert all_ok, "stable top_k_per_row correctness failed"


if __name__ == "__main__":
    main()
