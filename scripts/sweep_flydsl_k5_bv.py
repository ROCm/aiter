"""Sweep BV ? {16, 32, 64} for every PREFILL_PARAMS shape, find the per-shape
optimum under the *current* kernel code (e.g. OPT-VC rev17), and print a
final summary table.

Usage:
    HIP_VISIBLE_DEVICES=4 FLYDSL_RUNTIME_ENABLE_CACHE=0 \
        python3 scripts/sweep_flydsl_k5_bv.py [--include-trace] [--out-csv PATH]

The script monkey-patches ``_lookup_tuned_bv`` so each shape is forced to use a
specific BV. Each (shape, BV) combo is timed via the same ``_bench_fn``
(NUM_WARMUP=5 + NUM_ITERS=50, torch.profiler) used by the existing
TestPerformance harness, so numbers are directly comparable.

Flags:
    --include-trace     : also sweep the 396 ``slow``-marked trace shapes,
                          giving a 427-shape coverage matching the perf
                          test full sweep (default off, only PREFILL_PARAMS).
    --out-csv PATH      : append/overwrite CSV rows for ``chunk_gdn_h_tuned.csv``
                          format using the sweep's best BV per shape. Use this
                          to seed a tuned table; merge into the canonical
                          ``aiter/ops/flydsl/chunk_gdn_h_tuned.csv`` manually.
                          When omitted, no CSV is written.
"""

from __future__ import annotations

import argparse
import csv as _csv
import os
import sys

# Make sure the repo's aiter package is importable.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import torch  # noqa: E402

from aiter.ops.flydsl import (  # noqa: E402
    linear_attention_prefill_kernels as _lap,
)
from aiter.ops.flydsl.linear_attention_prefill_kernels import (  # noqa: E402
    chunk_gated_delta_rule_fwd_h_flydsl,
)
from aiter.ops.flydsl.test_flydsl_linear_attention_prefill import (  # noqa: E402
    NUM_ITERS,
    NUM_WARMUP,
    PERF_PARAMS,
    PREFILL_PARAMS,
    _bench_fn,
    _make_inputs,
)


def _unwrap(p):
    """Return the underlying ``PrefillArgs`` regardless of whether
    ``p`` is one directly or a ``pytest.param(args, marks=...)`` wrapper."""
    return p.values[0] if hasattr(p, "values") else p


def _force_bv(bv_value: int):
    """Return a monkey-patched ``_lookup_tuned_bv`` that always returns
    ``bv_value`` for one of the swept BV candidates.
    """
    assert bv_value in (16, 32, 64), f"unexpected BV={bv_value}"

    def _fn(*args, **kwargs):
        return bv_value

    return _fn


def _runtime_T_flat_N(args) -> tuple[int, int]:
    """Return (T_flat, N) the way the runtime kernel call sees them.

    For trace shapes (which set ``context_lens``), T_flat is the sum of
    every segment and N equals the number of segments.

    For hand-written shapes T_flat == max_num_batched_tokens (the bench
    harness's ``_build_context_lens`` slices it into N equal segments
    when is_varlen).
    """
    if args.context_lens is not None:
        return sum(args.context_lens), len(args.context_lens)
    if args.is_varlen:
        N_val = max(1, args.max_num_batched_tokens // args.full_prompt_len)
    else:
        N_val = 1
    return args.max_num_batched_tokens, N_val


def _runtime_head_mid_tail(args) -> tuple[int, int, int]:
    """Return (head, mid, tail) the way the kernel sees them.

    Mirrors the same head/mid/tail derivation the host wrapper applies
    on ``cu_seqlens`` so the sweep's lookup key matches the runtime
    caller's.
    """
    if args.context_lens is not None:
        lens = list(args.context_lens)
    elif args.is_varlen:
        # _build_context_lens(full_prompt_len, max_num_batched_tokens):
        # ceil(mnbt/full_len) equal segments, last one possibly short.
        full_len = args.full_prompt_len
        rem = args.max_num_batched_tokens
        lens = []
        while rem > 0:
            cur = min(full_len, rem)
            lens.append(cur)
            rem -= cur
    else:
        return 0, 0, 0
    n = len(lens)
    if n == 0:
        return 0, 0, 0
    if n == 1:
        return int(lens[0]), 0, 0
    if n == 2:
        return int(lens[0]), 0, int(lens[1])
    mids = lens[1:-1]
    mid = int(mids[0]) if all(m == mids[0] for m in mids) else 0
    return int(lens[0]), mid, int(lens[-1])


def _orig_tuned_bv(args) -> int:
    """Look up the existing tuned BV for this shape (no patch). Used
    purely for the "csv BV" column in the output table.
    """
    H = args.Hv // args.tp
    Hg = args.Hk // args.tp
    T_flat, N_val = _runtime_T_flat_N(args)
    head, mid, tail = _runtime_head_mid_tail(args)
    return _lap._lookup_tuned_bv(
        dtype_str=str(args.dtype),
        K=args.K,
        V=args.V,
        BT=args.BT,
        H=H,
        Hg=Hg,
        T_flat=T_flat,
        N=N_val,
        use_g=True,
        use_gk=False,
        use_h0=True,
        store_fs=bool(args.output_final_state),
        save_vn=True,
        is_varlen=args.is_varlen,
        wu_contig=True,
        head_seqlen=head,
        mid_seqlen=mid,
        tail_seqlen=tail,
    )


def _time_shape_with_bv(args, bv: int) -> float:
    """Time one (shape, BV) combo. Returns per-iter FlyDSL kernel time in us."""
    context_lens = args.resolve_context_lens()
    k, _w_orig, _u_orig, w_c, u_c, g, h0, cu, _ = _make_inputs(context_lens, args=args)

    # Monkey-patch the BV lookup just before the timed window.
    saved = _lap._lookup_tuned_bv
    _lap._lookup_tuned_bv = _force_bv(bv)
    try:

        def launch():
            chunk_gated_delta_rule_fwd_h_flydsl(
                k=k,
                w=w_c,
                u=u_c,
                g=g,
                initial_state=h0,
                output_final_state=args.output_final_state,
                cu_seqlens=cu,
            )

        # Warmup once outside the timed window so JIT compile + autotune
        # do not leak into _bench_fn's NUM_WARMUP=5.
        launch()
        torch.cuda.synchronize()
        return _bench_fn(launch)
    finally:
        _lap._lookup_tuned_bv = saved


def _csv_row_for(args, best_bv: int, best_us: float) -> dict:
    """Render one sweep result as a dict that matches the schema of
    ``aiter/ops/flydsl/chunk_gdn_h_tuned.csv``.
    """
    H = args.Hv // args.tp
    Hg = args.Hk // args.tp
    T_flat, N_val = _runtime_T_flat_N(args)
    head, mid, tail = _runtime_head_mid_tail(args)
    return {
        "arch": "gfx950",
        "dtype": str(args.dtype),
        "K": args.K,
        "V": args.V,
        "BT": args.BT,
        "H": H,
        "Hg": Hg,
        "T_flat": T_flat,
        "N": N_val,
        "head_seqlen": head,
        "mid_seqlen": mid,
        "tail_seqlen": tail,
        "use_g": "True",
        "use_gk": "False",
        "use_h0": "True",
        "store_fs": "True" if args.output_final_state else "False",
        "save_vn": "True",
        "is_varlen": "True" if args.is_varlen else "False",
        "wu_contig": "True",
        "BV": best_bv,
        "duration": f"{best_us:.4f}",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sweep BV for FlyDSL chunk-gated-delta-h K5 kernel"
    )
    parser.add_argument(
        "--include-trace",
        action="store_true",
        help="also sweep the slow-marked trace shapes if any (full PERF_PARAMS coverage)",
    )
    parser.add_argument(
        "--only-bench407",
        action="store_true",
        help="only sweep the 333 bench407 shapes (model_name=Qwen3.5-prefill-bench407); "
        "implies --include-trace is unused. Pairs naturally with "
        "--out-csv chunk_gdn_h_bench407_tuned.csv.",
    )
    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="write tuned-BV CSV rows to this path (overwrites). Drop into "
        "aiter/ops/flydsl/ as either chunk_gdn_h_tuned.csv or "
        "chunk_gdn_h_bench407_tuned.csv to deploy.",
    )
    cli_args = parser.parse_args()

    if cli_args.only_bench407:
        params = [
            _unwrap(p)
            for p in PERF_PARAMS
            if _unwrap(p).model_name == "prefill-bench333"
        ]
        print(f"Coverage: bench407 only ({len(params)} shapes)")
    elif cli_args.include_trace:
        params = [_unwrap(p) for p in PERF_PARAMS]
        print(f"Coverage: full PERF_PARAMS ({len(params)} shapes)")
    else:
        params = [
            p for p in PREFILL_PARAMS if p.model_name != "Qwen3.5-prefill-bench407"
        ]
        print(
            f"Coverage: PREFILL_PARAMS without bench407 ({len(params)} shapes; "
            f"pass --only-bench407 to sweep just bench407, or "
            f"--include-trace for full PERF_PARAMS coverage)"
        )

    candidate_bvs = (16, 32, 64)
    print(f"NUM_WARMUP={NUM_WARMUP} NUM_ITERS={NUM_ITERS}")
    print(f"Sweeping BV ? {candidate_bvs}")
    print()

    bv_col_headers = "  ".join(f"{'BV=' + str(bv):>10}" for bv in candidate_bvs)
    header = (
        f"{'#':>4}  {'shape':<60}  {'csv BV':>6}  "
        f"{bv_col_headers}  {'best BV':>7}  {'best us':>9}"
    )
    print(header)
    print("-" * len(header))

    results = []
    for idx, args in enumerate(params):
        tag = repr(args)[:60]
        csv_bv = _orig_tuned_bv(args)
        per_bv = {}
        for bv in candidate_bvs:
            try:
                t = _time_shape_with_bv(args, bv)
            except Exception as e:
                print(f"[warn] shape={tag} BV={bv} failed: {e!r}")
                t = float("inf")
            per_bv[bv] = t
        best_bv = min(per_bv, key=lambda b: per_bv[b])
        best_us = per_bv[best_bv]
        results.append((args, tag, csv_bv, per_bv, best_bv, best_us))
        row = (
            f"{idx + 1:>4}  {tag:<60}  {csv_bv:>6}  "
            + "  ".join(f"{per_bv[bv]:>10.1f}" for bv in candidate_bvs)
            + f"  {best_bv:>7}  {best_us:>9.1f}"
        )
        print(row, flush=True)

    # Final summary.
    print()
    print("Summary:")
    csv_optimal = sum(1 for _, _, csv_bv, _, best_bv, _ in results if best_bv == csv_bv)
    print(f"  shapes where csv BV == new best BV: {csv_optimal}/{len(results)}")
    differ = [
        (tag, csv_bv, best_bv, per_bv[csv_bv], best_us)
        for (_, tag, csv_bv, per_bv, best_bv, best_us) in results
        if best_bv != csv_bv
    ]
    if differ:
        print(f"  shapes where the new best BV differs from csv: {len(differ)}")
        # Cap output to the top-30 worst gaps (csv_us - best_us) so the
        # full 427-shape sweep stays readable; keep all in the CSV.
        differ_sorted = sorted(
            differ,
            key=lambda x: -(x[3] - x[4]),  # csv_us - best_us, biggest first
        )
        for tag, csv_bv, best_bv, csv_us, best_us in differ_sorted[:30]:
            delta = (best_us - csv_us) / csv_us * 100.0
            print(
                f"    {tag}: csv BV={csv_bv} ({csv_us:.1f} us) "
                f"-> best BV={best_bv} ({best_us:.1f} us, {delta:+.1f}%)"
            )

    # Optional: write CSV rows in the chunk_gdn_h_tuned.csv schema.
    if cli_args.out_csv:
        out_rows = [
            _csv_row_for(a, best_bv, best_us)
            for (a, _, _, _, best_bv, best_us) in results
        ]
        # Dedupe by the lookup key (drops ``duration``/``BV``). When
        # two PrefillArgs map to the same key, keep the row with the
        # lowest duration (tiebreak: first occurrence).
        seen = {}
        for r in out_rows:
            key = (
                r["arch"],
                r["dtype"],
                r["K"],
                r["V"],
                r["BT"],
                r["H"],
                r["Hg"],
                r["T_flat"],
                r["N"],
                r["head_seqlen"],
                r["mid_seqlen"],
                r["tail_seqlen"],
                r["use_g"],
                r["use_gk"],
                r["use_h0"],
                r["store_fs"],
                r["save_vn"],
                r["is_varlen"],
                r["wu_contig"],
            )
            existing = seen.get(key)
            if existing is None or float(r["duration"]) < float(existing["duration"]):
                seen[key] = r
        deduped = list(seen.values())
        print(
            f"\n  After dedupe by lookup key: {len(deduped)} unique rows "
            f"(from {len(out_rows)} sweeps)."
        )
        fieldnames = list(deduped[0].keys()) if deduped else []
        with open(cli_args.out_csv, "w", encoding="utf-8", newline="") as f:
            w = _csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in deduped:
                w.writerow(r)
        print(f"  Wrote: {cli_args.out_csv}")


if __name__ == "__main__":
    main()
