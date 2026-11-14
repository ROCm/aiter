# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic ("stable") top_k_per_row correctness.

The stable contract for DSA + tensor parallel: identical input -> identical,
ascending-index-ordered, smallest-index tie-broken output, run after run and
byte-identical across ranks.

    python op_tests/test_topk_per_row_stable.py
"""

import argparse

import pandas as pd
import torch

from aiter.ops.opus.pa_mqa_logits_mxfp4 import mqa_logits_mxfp4_lane_max_elems
from aiter.ops.topk import top_k_per_row_decode, top_k_per_row_prefill
from aiter.test_common import checkAllclose

# Lane maxima as mqa_logits_mxfp4(lane_max=...) lays them out
# (pa_mqa_logits_mxfp4.h): row r's slots start at (row start / 16) + 64 r and
# slot 64 g + j holds the max of keys (16 g + i) * 64 + j, i in [0, 16).
LANE_MAX_TILES, LANE_MAX_TILE_KEYS, LANE_MAX_ROW_PAD = 16, 64, 64
# What a call's lane_max buffer may hold: the scorer's summary of these
# logits, a stale one (another step's logits), or junk. The top-k must select
# the same whatever it holds.
LANE_MAX_FILLS = ("scorer", "stale", "nan", "+inf", "-inf")
# The prefilter's shortest row (radix_topk_one_block._PREFILTER_MIN_ROW_LEN):
# reported, so a table shows how many rows took it on gfx1250.
PREFILTER_MIN_ROW_LEN = 28 * 1024


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
        idx = torch.full((num_rows, k), -123456789, dtype=torch.int32, device="cuda")
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
        a[r].cpu().tolist() == ref_stable_topk(logits[r], k) for r in sample_rows
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


def scorer_lane_max(flat, starts, lens, num_rows):
    """The lane maxima mqa_logits_mxfp4 would write beside packed logits
    `flat`: every row's slots, -inf elsewhere."""
    out = torch.full(
        (mqa_logits_mxfp4_lane_max_elems(flat.numel(), num_rows),),
        float("-inf"),
        device="cuda",
    )
    group = LANE_MAX_TILES * LANE_MAX_TILE_KEYS
    for r, (s, n) in enumerate(zip(starts.tolist(), lens.tolist())):
        groups = -(-n // group)
        v = torch.full((groups * group,), float("-inf"), device="cuda")
        v[:n] = flat[s : s + n]
        slots = v.view(groups, LANE_MAX_TILES, LANE_MAX_TILE_KEYS).amax(1).flatten()
        base = s // LANE_MAX_TILES + r * LANE_MAX_ROW_PAD
        out[base : base + slots.numel()] = slots
    return out


def lane_max_fill(fill, flat, starts, lens, num_rows):
    """A lane_max buffer for packed logits `flat` holding `fill`."""
    if fill == "scorer":
        return scorer_lane_max(flat, starts, lens, num_rows)
    if fill == "stale":
        g = torch.Generator(device="cuda").manual_seed(99)
        other = torch.randn(flat.shape, generator=g, device="cuda") * 4
        return scorer_lane_max(other, starts, lens, num_rows)
    n = mqa_logits_mxfp4_lane_max_elems(flat.numel(), num_rows)
    value = {"nan": float("nan"), "+inf": float("inf"), "-inf": float("-inf")}
    return torch.full((n,), value[fill], device="cuda")


def check_decode_packed(
    num_rows, width, k, tie_level, lane_max_results=None, overflow_row=False
):
    """Packed rows (row_starts, width) against the same rows in a width-wide
    plane: the same kernel, bit-identical indices and values. Lengths are
    ragged (some under k, most not a multiple of 4), starts 64-element
    aligned, NaN between rows so a read past a row's end shows.

    With `lane_max_results` (a list), the packed call is also made with each
    of LANE_MAX_FILLS as its lane_max and must equal the call without it bit
    for bit; a row each goes into the list. `overflow_row`: row 1 is 40000
    equal values, more than the prefilter's candidate buffer holds."""
    torch.manual_seed(7)
    top = min(40000, width) + 1
    lens = torch.randint(1, top, (num_rows,), dtype=torch.int32, device="cuda")
    lens[0] = min(k // 2, width)
    if overflow_row:
        lens[1] = min(40000, width)
    spans = (lens + 63) // 64 * 64
    starts = (torch.cumsum(spans, 0) - spans).to(torch.int32)
    values = make_logits(1, int(spans.sum()), tie_level, seed=11)[0]
    if overflow_row:
        s1 = int(starts[1])
        values[s1 : s1 + int(lens[1])] = 1.0
    plane = torch.full((num_rows, width), float("-inf"), device="cuda")
    flat = torch.full((int(spans.sum()) + k,), float("nan"), device="cuda")
    for r, (s, n) in enumerate(zip(starts.tolist(), lens.tolist())):
        plane[r, :n] = values[s : s + n]
        flat[s : s + n] = values[s : s + n]

    def run(packed, lane_max=None):
        idx = torch.empty((num_rows, k), dtype=torch.int32, device="cuda")
        val = torch.empty((num_rows, k), dtype=torch.float32, device="cuda")
        if packed:
            top_k_per_row_decode(
                flat, 1, lens, idx, num_rows, 0, 1, k=k, stable=True, values=val,
                row_starts=starts, plane_width=width, lane_max=lane_max,
            )  # fmt: skip
        else:
            top_k_per_row_decode(
                plane, 1, lens, idx, num_rows, width, 1, k=k, stable=True,
                values=val,
            )  # fmt: skip
        torch.cuda.synchronize()
        return idx, val

    (want, want_val), (got, got_val) = run(False), run(True)
    ok = torch.equal(want, got) and torch.equal(want_val, got_val)
    tag = f"decode packed rows={num_rows} width={width} k={k} tie={tie_level}"
    print(f"[{tag}] matches_plane={ok}")
    if lane_max_results is None:
        return ok
    for fill in LANE_MAX_FILLS:
        lm_idx, lm_val = run(True, lane_max_fill(fill, flat, starts, lens, num_rows))
        err = checkAllclose(
            got_val,
            lm_val,
            rtol=0,
            atol=0,
            msg=f"{tag} lane_max={fill}: ",
            printLog=False,
        )
        same = torch.equal(got, lm_idx) and torch.equal(got_val, lm_val) and err == 0
        lane_max_results.append(
            {
                "rows": num_rows,
                "width": width,
                "k": k,
                "ties": tie_level,
                "overflow row": overflow_row,
                "lane_max": fill,
                "rows past prefilter min": int((lens >= PREFILTER_MIN_ROW_LEN).sum()),
                "equal without lane_max": same,
            }
        )
        ok &= same
    return ok


def check_lane_max_errors():
    """What top_k_per_row_decode must refuse rather than ignore or read past:
    lane maxima too small, not fp32, not flat, strided, or for plane rows."""
    num_rows, width, k = 4, 65536, 512
    lens = torch.full((num_rows,), 40000, dtype=torch.int32, device="cuda")
    spans = (lens + 63) // 64 * 64
    starts = (torch.cumsum(spans, 0) - spans).to(torch.int32)
    flat = torch.randn(int(spans.sum()), device="cuda")
    plane = torch.randn(num_rows, width, device="cuda")
    need = mqa_logits_mxfp4_lane_max_elems(flat.numel(), num_rows)
    idx = torch.empty((num_rows, k), dtype=torch.int32, device="cuda")

    def packed(lane_max):
        return lambda: top_k_per_row_decode(
            flat, 1, lens, idx, num_rows, 0, 1, k=k, stable=True,
            row_starts=starts, plane_width=width, lane_max=lane_max,
        )  # fmt: skip

    def plane_rows():
        top_k_per_row_decode(
            plane, 1, lens, idx, num_rows, width, 1, k=k, stable=True,
            lane_max=torch.empty(need, device="cuda"),
        )  # fmt: skip

    tries = {
        "one element short": packed(torch.empty(need - 1, device="cuda")),
        "fp16": packed(torch.empty(need, dtype=torch.float16, device="cuda")),
        "2-D": packed(torch.empty(1, need, device="cuda")),
        "strided": packed(torch.empty(2 * need, device="cuda")[::2]),
        "plane rows (no row_starts)": plane_rows,
    }
    rows = []
    for what, call in tries.items():
        try:
            call()
            torch.cuda.synchronize()
            refused = False
        except ValueError:
            refused = True
        rows.append({"lane_max": what, "refused": refused})
    print(pd.DataFrame(rows).to_markdown(index=False))
    return all(r["refused"] for r in rows)


def check_decode_cross_part_tie():
    """Stable merge must keep the smallest global indices across equal parts."""
    rows, width, k = 1, 131072, 512
    logits = torch.zeros((rows, width), dtype=torch.float32, device="cuda")
    part_width = width // 2
    part0_ties = k // 4
    logits[0, part_width - part0_ties : part_width + k] = 1
    seq_lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    indices = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    top_k_per_row_decode(
        logits,
        1,
        seq_lens,
        indices,
        rows,
        width,
        1,
        k=k,
        stable=True,
    )
    torch.cuda.synchronize()
    expected = torch.arange(
        part_width - part0_ties,
        part_width + k - part0_ties,
        dtype=torch.int32,
        device="cuda",
    )
    ok = torch.equal(indices[0], expected)
    print(f"[decode cross-part tie] smallest_indices={ok}")
    return ok


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
    # packed rows: few rows at a long width take the chunked kernel, many
    # rows the one-block kernel, a short width the one-workgroup kernel, as
    # their plane would
    # Each also with lane maxima (scorer-written, stale, NaN, +-inf), which must
    # not change a bit. On gfx1250 the 96-row calls take the prefilter for
    # their long rows; one has a row of ties too long for its candidate buffer.
    lane_max_rows = []
    for tie in ("none", "heavy"):
        for k in (512, 2048):
            for num_rows, width, overflow in (
                (6, 262144, False),
                (96, 262144, False),
                (96, 262144, True),
                (6, 16384, False),
            ):
                all_ok &= check_decode_packed(
                    num_rows, width, k, tie, lane_max_rows, overflow_row=overflow
                )
    print(pd.DataFrame(lane_max_rows).to_markdown(index=False))
    all_ok &= check_lane_max_errors()
    all_ok &= check_decode_cross_part_tie()
    print("\nRESULT:", "ALL PASS" if all_ok else "FAILURES PRESENT")
    assert all_ok, "stable top_k_per_row correctness failed"


if __name__ == "__main__":
    main()
