#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Merge clk_trace.py sysfs samples with the F8GEMM UT's soak throughput into one
DPM table: sustained GB/s against the sclk/fclk/mclk it actually ran at, plus
GB/s-per-GHz and GB/s-per-Watt efficiency.

Why median (not mean) for the clock -- and why this script prints BOTH:
  DPM clocks are DISCRETE levels; a run mostly sits at ONE level with brief
  excursions (a ramp tail past --settle, a boost blip, a throttle dip). The MEDIAN
  lands on the level the chip actually held (and, like clk_trace.py, ignores the
  -1 failed reads), while the MEAN smears across transients to a value no DPM level
  ever is. So the median is the right "what clock did it run at" estimator. But if
  the run was bimodal (throttle), median hides that -- so this script also prints
  the mean, the p5/p95 spread, and a per-level histogram to expose it.

Usage:
  # 1) sample clocks while the UT soaks (clk_mxfp8fp4gemm.sh does both):
  test_mxfp8fp4gemm.py --mode clk --intype a8w8 -s 2,1048576,16384 \
      --metrics-json m.json &
  clk_trace.py --wait-pid $! --period 1 --settle 0.5 --csv trace.csv
  # 2) merge:
  clk_merge.py --clk-csv trace.csv --metrics-json m.json --settle 0.5

--metrics-json is optional: without it, only the clock stats + histogram print.
One clk trace corresponds to one UT run, so every metrics record is joined against
that single trace's clock stats.
"""
import argparse
import csv
import json
import statistics as st
from collections import Counter

FIELDS = ("sclk_mhz", "mclk_mhz", "fclk_mhz", "power_w")


def load_clk(path, settle):
    """Return {field: [values]} for t_s >= settle, dropping -1 failed reads."""
    cols = {k: [] for k in FIELDS}
    with open(path) as f:
        for row in csv.DictReader(f):
            if float(row["t_s"]) < settle:
                continue
            for k in FIELDS:
                v = float(row[k])
                if v >= 0:  # clk_trace.py writes -1 for a read that failed this tick
                    cols[k].append(v)
    return cols


def stats(xs):
    if not xs:
        return dict(n=0, median=float("nan"), mean=float("nan"),
                    std=float("nan"), p5=float("nan"), p95=float("nan"))
    s = sorted(xs)

    def pct(p):
        return s[min(len(s) - 1, int(p * len(s)))]

    return dict(
        n=len(xs),
        median=st.median(xs),
        mean=st.fmean(xs),
        std=st.pstdev(xs) if len(xs) > 1 else 0.0,
        p5=pct(0.05),
        p95=pct(0.95),
    )


def histogram(xs):
    """DPM-level occupancy: (level_mhz, pct_of_samples) sorted by dominance."""
    c = Counter(round(x) for x in xs)
    tot = sum(c.values()) or 1
    return sorted(((lvl, 100.0 * n / tot) for lvl, n in c.items()), key=lambda t: -t[1])


def _get(rec, key):
    try:
        return float(rec[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clk-csv", required=True, help="clk_trace.py --csv output")
    ap.add_argument("--metrics-json", help="test_mxfp8fp4gemm.py --metrics-json output")
    ap.add_argument("--settle", type=float, default=0.5,
                    help="drop CSV samples with t_s < this (align with the soak "
                    "window / clk_trace --settle; default 0.5)")
    args = ap.parse_args()

    cols = load_clk(args.clk_csv, args.settle)
    stat = {k: stats(cols[k]) for k in FIELDS}

    print(f"\n# clock samples (t_s >= {args.settle}s, n={stat['sclk_mhz']['n']})\n")
    print("| domain | median | mean | std | p5 | p95 |")
    print("|---|---|---|---|---|---|")
    for k in FIELDS:
        s = stat[k]
        unit = "W" if k == "power_w" else "MHz"
        print(f"| {k.replace('_mhz','').replace('_w','')} "
              f"| {s['median']:.0f}{unit} | {s['mean']:.0f} | {s['std']:.1f} "
              f"| {s['p5']:.0f} | {s['p95']:.0f} |")

    # Per-level occupancy -- a single dominant row = clean steady clock; two big
    # rows = bimodal (throttle), and the median above is only the dominant one.
    for k in ("sclk_mhz", "fclk_mhz", "mclk_mhz"):
        h = histogram(cols[k])
        top = "  ".join(f"{lvl}MHz:{pct:.0f}%" for lvl, pct in h[:4])
        flag = "  <-- BIMODAL (throttle?)" if len(h) > 1 and h[1][1] >= 20 else ""
        print(f"\n{k.replace('_mhz','')} DPM levels: {top}{flag}")

    if not args.metrics_json:
        return

    with open(args.metrics_json) as f:
        recs = json.load(f)

    sclk = stat["sclk_mhz"]["median"] / 1000.0  # GHz
    fclk = stat["fclk_mhz"]["median"] / 1000.0
    mclk = stat["mclk_mhz"]["median"] / 1000.0
    power = stat["power_w"]["median"]

    print("\n# sustained throughput vs measured clock "
          "(soak = wall-clock, same thermal window as the clock samples)\n")
    print("| intype | M | N | K | soak TB/s | device TB/s | GB/s·GHz⁻¹(mclk) "
          "| GB/s·GHz⁻¹(fclk) | GB/s·GHz⁻¹(sclk) | GB/s·W⁻¹ |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in recs:
        soak_tbs = _get(r, "asm soak TB/s")
        dev_tbs = _get(r, "asm TB/s")
        gbs = soak_tbs * 1000.0  # TB/s -> GB/s
        per = lambda ghz: (gbs / ghz if ghz and ghz == ghz and ghz > 0 else float("nan"))
        print(
            f"| {r.get('intype','')} | {r.get('M','')} | {r.get('N','')} "
            f"| {r.get('K','')} | {soak_tbs:.2f} | {dev_tbs:.2f} "
            f"| {per(mclk):.1f} | {per(fclk):.1f} | {per(sclk):.1f} "
            f"| {gbs/power if power and power==power and power>0 else float('nan'):.1f} |"
        )


if __name__ == "__main__":
    main()
