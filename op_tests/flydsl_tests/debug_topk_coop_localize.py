#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Localize cooperative FlyDSL TopK K=512 correctness failures."""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from op_tests.benchmark_topk_per_row_decode import make_logits, make_row_ends  # noqa: E402


K = 512
RADIX_BINS = 2048
COOP_META_SLOTS = 16
META_NAMES = (
    "arrival_count",
    "phase_done",
    "out_front",
    "out_back",
    "init_epoch",
    "first_above",
    "first_threshold",
    "second_above",
    "second_threshold",
    "third_above",
    "third_threshold",
)


@dataclass(frozen=True)
class PassMeta:
    threshold: int
    above: int


@dataclass(frozen=True)
class RadixReference:
    first: PassMeta
    second: PassMeta
    third: PassMeta
    need_after_first: int
    need_after_second: int
    num_needed: int
    strictly_above_count: int
    boundary_count: int
    selected_set: set[int]
    pass1_hist: torch.Tensor


@dataclass(frozen=True)
class PublishedCounts:
    pass1_above: int
    pass2_above: int
    pass3_above: int
    need_after_first: int
    need_after_second: int
    num_needed: int
    strictly_above: int
    boundary: int
    selected_set: set[int]


def ordered_keys(values: torch.Tensor) -> torch.Tensor:
    vals = values.detach().cpu().contiguous().to(torch.float32)
    vals = torch.where(vals == 0, torch.zeros_like(vals), vals)
    bits = vals.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    sign = bits >> 31
    neg_key = (~bits) & 0xFFFFFFFF
    pos_key = bits ^ 0x80000000
    return torch.where(sign != 0, neg_key, pos_key)


def choose_threshold(hist: torch.Tensor, target_k: int) -> PassMeta:
    total = int(hist.sum().item())
    kprime = total - int(target_k)
    prefix = torch.cumsum(hist.to(torch.int64), dim=0)
    excl = prefix - hist
    crosses = (excl <= kprime) & (prefix > kprime)
    if not bool(crosses.any().item()):
        raise RuntimeError(f"no threshold found: total={total} target_k={target_k}")
    threshold = int(torch.nonzero(crosses, as_tuple=False)[0].item())
    above = total - int(prefix[threshold].item())
    return PassMeta(threshold=threshold, above=above)


def radix_reference(logits_row: torch.Tensor, row_len: int) -> RadixReference:
    keys = ordered_keys(logits_row[:row_len])
    high = keys >> 21
    mid = (keys >> 10) & 0x7FF
    low = keys & 0x3FF

    hist1 = torch.bincount(high, minlength=RADIX_BINS)
    first = choose_threshold(hist1, K)
    need_after_first = K - first.above

    high_match = high == first.threshold
    hist2 = torch.bincount(mid[high_match], minlength=RADIX_BINS)
    second = choose_threshold(hist2, need_after_first)
    need_after_second = need_after_first - second.above

    high_mid_match = high_match & (mid == second.threshold)
    hist3 = torch.bincount(low[high_mid_match], minlength=RADIX_BINS)
    third = choose_threshold(hist3, need_after_second)
    num_needed = need_after_second - third.above

    strictly_above = (high > first.threshold) | (
        (high == first.threshold)
        & ((mid > second.threshold) | ((mid == second.threshold) & (low > third.threshold)))
    )
    boundary = high_mid_match & (low == third.threshold)
    strict_idx = torch.nonzero(strictly_above, as_tuple=False).flatten().tolist()
    boundary_idx = torch.nonzero(boundary, as_tuple=False).flatten().tolist()
    selected = set(int(i) for i in strict_idx)
    selected.update(int(i) for i in boundary_idx[:num_needed])

    return RadixReference(
        first=first,
        second=second,
        third=third,
        need_after_first=need_after_first,
        need_after_second=need_after_second,
        num_needed=num_needed,
        strictly_above_count=len(strict_idx),
        boundary_count=len(boundary_idx),
        selected_set=selected,
        pass1_hist=hist1.to(torch.int32),
    )


def counts_for_thresholds(
    logits_row: torch.Tensor,
    row_len: int,
    first_threshold: int,
    second_threshold: int,
    third_threshold: int,
    first_above_meta: int,
    second_above_meta: int,
    third_above_meta: int,
) -> PublishedCounts:
    keys = ordered_keys(logits_row[:row_len])
    high = keys >> 21
    mid = (keys >> 10) & 0x7FF
    low = keys & 0x3FF

    pass1_above_mask = high > first_threshold
    pass2_domain = high == first_threshold
    pass2_above_mask = pass2_domain & (mid > second_threshold)
    pass3_domain = pass2_domain & (mid == second_threshold)
    pass3_above_mask = pass3_domain & (low > third_threshold)
    boundary = pass3_domain & (low == third_threshold)
    strictly_above = pass1_above_mask | pass2_above_mask | pass3_above_mask

    need_after_first = K - first_above_meta
    need_after_second = need_after_first - second_above_meta
    num_needed = need_after_second - third_above_meta
    strict_idx = torch.nonzero(strictly_above, as_tuple=False).flatten().tolist()
    boundary_idx = torch.nonzero(boundary, as_tuple=False).flatten().tolist()
    selected = set(int(i) for i in strict_idx)
    selected.update(int(i) for i in boundary_idx[: max(0, num_needed)])

    return PublishedCounts(
        pass1_above=int(pass1_above_mask.sum().item()),
        pass2_above=int(pass2_above_mask.sum().item()),
        pass3_above=int(pass3_above_mask.sum().item()),
        need_after_first=need_after_first,
        need_after_second=need_after_second,
        num_needed=num_needed,
        strictly_above=len(strict_idx),
        boundary=int(boundary.sum().item()),
        selected_set=selected,
    )


def set_summary(actual: torch.Tensor, expected: set[int], row_len: int) -> tuple[bool, str]:
    actual_list = [int(x) for x in actual.detach().cpu().tolist()]
    invalid = [x for x in actual_list if x < 0 or x >= row_len]
    duplicates = len(actual_list) - len(set(actual_list))
    actual_set = set(actual_list)
    missing = sorted(expected - actual_set)
    extra = sorted(actual_set - expected)
    ok = not invalid and duplicates == 0 and not missing and not extra
    detail = (
        f"invalid={invalid[:8]} dup_count={duplicates} "
        f"missing_count={len(missing)} extra_count={len(extra)} "
        f"missing_sample={missing[:8]} extra_sample={extra[:8]}"
    )
    return ok, detail


def run_one_case(args: argparse.Namespace, seed: int) -> bool:
    device = torch.device("cuda")
    batch_size = args.num_rows // args.next_n
    seq_lens, row_ends = make_row_ends(
        batch_size, args.num_rows, args.length, args.next_n, device
    )
    logits = make_logits(
        args.num_rows,
        args.length,
        row_ends,
        torch.float32,
        args.distribution,
        "unit",
        seed,
        device,
    )

    topk_mod = importlib.import_module("aiter.ops.flydsl.topk_per_row_decode")
    kernel_mod = importlib.import_module(
        "aiter.ops.flydsl.kernels.topk_per_row_decode_coop"
    )
    flydsl_topk = topk_mod.flydsl_top_k_per_row_decode

    one_cta = torch.empty((args.num_rows, K), dtype=torch.int32, device=device)
    os.environ["FLYDSL_TOPK_COOP"] = "0"
    flydsl_topk(
        logits,
        args.next_n,
        seq_lens,
        one_cta,
        args.num_rows,
        logits.stride(0),
        logits.stride(1),
        k=K,
        ordered=False,
    )
    torch.cuda.synchronize()

    topk_mod._COOP_WORKSPACES.clear()
    coop = torch.empty((args.num_rows, K), dtype=torch.int32, device=device)
    os.environ["FLYDSL_TOPK_COOP"] = "1"
    os.environ["FLYDSL_TOPK_COOP_PARTITIONS"] = str(args.partitions)
    os.environ["FLYDSL_TOPK_COOP_ATOMIC_HIST"] = "1" if args.atomic_histogram else "0"
    os.environ["FLYDSL_TOPK_COOP_MIN_ROW_LEN"] = "0"
    flydsl_topk(
        logits,
        args.next_n,
        seq_lens,
        coop,
        args.num_rows,
        logits.stride(0),
        logits.stride(1),
        k=K,
        ordered=False,
    )
    torch.cuda.synchronize()

    if len(topk_mod._COOP_WORKSPACES) != 1:
        raise RuntimeError(f"expected one coop workspace, got {len(topk_mod._COOP_WORKSPACES)}")
    workspace = next(iter(topk_mod._COOP_WORKSPACES.values())).detach().cpu()
    row_slots = kernel_mod.cooperative_workspace_slots(
        args.num_rows,
        args.partitions,
        atomic_histogram=args.atomic_histogram,
    ) // args.num_rows
    workspace_rows = workspace.view(args.num_rows, row_slots)

    all_ok = True
    print(
        f"CASE seed={seed} K={K} L={args.length} rows={args.num_rows} "
        f"P={args.partitions} atomic_hist={int(args.atomic_histogram)}"
    )
    for row in range(args.num_rows):
        row_len = int(row_ends[row].item())
        ref = radix_reference(logits[row].detach().cpu(), row_len)
        meta_raw = [int(x) for x in workspace_rows[row, :COOP_META_SLOTS].tolist()]
        meta = {name: meta_raw[i] for i, name in enumerate(META_NAMES)}
        published = counts_for_thresholds(
            logits[row].detach().cpu(),
            row_len,
            meta["first_threshold"],
            meta["second_threshold"],
            meta["third_threshold"],
            meta["first_above"],
            meta["second_above"],
            meta["third_above"],
        )
        coop_set = set(int(x) for x in coop[row].detach().cpu().tolist())
        one_cta_set = set(int(x) for x in one_cta[row].detach().cpu().tolist())
        torch_set = set(
            int(x)
            for x in torch.topk(logits[row, :row_len], K).indices.detach().cpu().tolist()
        )

        expected_by_stage = {
            "pass1": (meta["first_threshold"], meta["first_above"], ref.first),
            "pass2": (meta["second_threshold"], meta["second_above"], ref.second),
            "pass3": (meta["third_threshold"], meta["third_above"], ref.third),
        }
        divergence = "none"
        for stage, (actual_threshold, actual_above, expected) in expected_by_stage.items():
            if actual_threshold != expected.threshold or actual_above != expected.above:
                divergence = f"{stage} threshold/above"
                break
        if divergence == "none":
            if meta["out_front"] != ref.strictly_above_count:
                divergence = "final out_front counter"
            elif meta["out_back"] != ref.boundary_count:
                divergence = "final out_back counter"
            elif coop_set != ref.selected_set:
                divergence = "final output append/boundary"

        selected_ok, selected_detail = set_summary(coop[row], ref.selected_set, row_len)
        published_ok, published_detail = set_summary(
            coop[row], published.selected_set, row_len
        )
        one_cta_ok = one_cta_set == torch_set
        coop_vs_one = coop_set == one_cta_set
        coop_vs_torch = coop_set == torch_set
        all_ok = all_ok and selected_ok and one_cta_ok and coop_vs_one

        print(f"ROW {row} row_len={row_len} first_divergence={divergence}")
        print(
            "  expected pass meta: "
            f"p1(thr={ref.first.threshold},above={ref.first.above}) "
            f"p2(thr={ref.second.threshold},above={ref.second.above}) "
            f"p3(thr={ref.third.threshold},above={ref.third.above})"
        )
        print(
            "  cooperative meta:   "
            f"p1(thr={meta['first_threshold']},above={meta['first_above']}) "
            f"p2(thr={meta['second_threshold']},above={meta['second_above']}) "
            f"p3(thr={meta['third_threshold']},above={meta['third_above']})"
        )
        print(
            "  expected needs/counts: "
            f"need1={ref.need_after_first} need2={ref.need_after_second} "
            f"num_needed={ref.num_needed} strict={ref.strictly_above_count} "
            f"boundary={ref.boundary_count}"
        )
        print(
            "  cooperative counters: "
            f"out_front={meta['out_front']} out_back={meta['out_back']} "
            f"arrival_count={meta['arrival_count']} phase_done={meta['phase_done']}"
        )
        print(
            "  counts under cooperative thresholds: "
            f"p1_above={published.pass1_above} p2_above={published.pass2_above} "
            f"p3_above={published.pass3_above} need1={published.need_after_first} "
            f"need2={published.need_after_second} num_needed={published.num_needed} "
            f"strict={published.strictly_above} boundary={published.boundary}"
        )
        print(
            f"  sets: one_cta_vs_torch={one_cta_ok} coop_vs_one={coop_vs_one} "
            f"coop_vs_torch={coop_vs_torch} expected_threshold_set={selected_ok} "
            f"coop_threshold_set={published_ok}"
        )
        print(f"  output detail: {selected_detail}")
        print(f"  coop-threshold detail: {published_detail}")

        if args.atomic_histogram:
            hist_base = COOP_META_SLOTS
            pass1_hist = workspace_rows[row, hist_base : hist_base + RADIX_BINS]
            hist_delta = pass1_hist.to(torch.int32) - ref.pass1_hist
            bad = torch.nonzero(hist_delta != 0, as_tuple=False).flatten()
            if bad.numel() == 0:
                print("  pass1 global histogram: matches expected")
            else:
                samples = [
                    (
                        int(i),
                        int(pass1_hist[i].item()),
                        int(ref.pass1_hist[i].item()),
                    )
                    for i in bad[:8].tolist()
                ]
                print(
                    "  pass1 global histogram: "
                    f"bad_bins={int(bad.numel())} samples(actual,expected)={samples}"
                )

    return all_ok


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length", "-L", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scan-seeds", type=int, default=0)
    parser.add_argument("--num-rows", type=int, default=1)
    parser.add_argument("--next-n", type=int, default=1)
    parser.add_argument("--partitions", type=int, default=8, choices=(4, 8, 16))
    parser.add_argument("--atomic-histogram", action="store_true")
    parser.add_argument(
        "--distribution",
        choices=("random", "10LSBits", "mixed", "ties"),
        default="random",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("SKIP: CUDA/ROCm is not available")
        return 0

    seeds = [args.seed]
    if args.scan_seeds > 0:
        seeds = list(range(args.seed, args.seed + args.scan_seeds))

    for seed in seeds:
        ok = run_one_case(args, seed)
        if not ok or args.scan_seeds == 0:
            return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
