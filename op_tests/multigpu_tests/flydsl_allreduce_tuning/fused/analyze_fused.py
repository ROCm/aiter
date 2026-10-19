"""Summarize the fused AR+RMSNorm one-shot sweep (<sweep_dir>/tp{tp}_w{h}.csv).

Per (tp, width, M): incumbent cdr (min of fused 1stage/2stage), cdr with a
separate norm, what the shipped fused dispatcher runs in exact mode, the
shipped fused one-shot ladder, best unsplit / best split pinned row, overall
winner and its resolved knobs (atoms, grid cap, split) parsed from the variant.
"""

import argparse
import glob
import math
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_io

SQNR_FLOOR = 40.0
_VAR = re.compile(
    r"_a(?P<a>\d+)_g(?P<g>\d+)_\w*?_b(?P<b>\d+)(?P<peer>_peer)?_rms_h\d+(?:p(?P<p>\d+))?"
    r"(?:_k(?P<k>\d+))?/"
)


def knobs(variant: str) -> str:
    m = _VAR.search(str(variant))
    if not m:
        return "?"
    s = f"a{m['a']} g{m['g']} b{m['b']}"
    if m["k"]:
        s += f" k{m['k']}"
    if m["p"]:
        s += f" pad{m['p']}"
    return s


def _time(r, key) -> float:
    """*key*'s time in row *r*, or NaN when it did not run or missed the floor."""
    v = r.get(f"{key} us")
    q = r.get(f"{key} SQNR dB")
    if v is None or pd.isna(v) or (pd.notna(q) and q < SQNR_FLOOR):
        return math.nan
    return float(v)


def load(sweep_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(sweep_dir, "tp*_w*.csv"))):
        d = bench_io.read_bench_csv(path)
        fly = sorted(
            c[: -len(" variant")]
            for c in d.columns
            if c.startswith("fused_fly_1stage_") and c.endswith(" variant")
        )
        for _, r in d.iterrows():
            times = {k: _time(r, k) for k in fly}
            times = {k: v for k, v in times.items() if not math.isnan(v)}
            unsplit = {
                k: v
                for k, v in times.items()
                if "_k" not in k[len("fused_fly_1stage") :]
            }
            split = {k: v for k, v in times.items() if k not in unsplit}
            best = min(times, key=times.get) if times else None
            bu = min(unsplit, key=unsplit.get) if unsplit else None
            bs = min(split, key=split.get) if split else None
            cdr = min(_time(r, "fused_cdr_1stage"), _time(r, "fused_cdr_2stage"))
            ladder = _time(r, "fused_fly_1stage")
            rows.append(
                {
                    "tp": int(r["TP"]),
                    "h": int(r["K"]),
                    "M": int(r["M"]),
                    "kib": int(r["_nbytes"]) // 1024,
                    "cdr": cdr,
                    "sep_cdr": _time(r, "separate_cdr"),
                    # The shipped dispatcher in exact mode (the default): the
                    # one-shot inside its window, the caller's path outside.
                    "auto": ladder if r.get("route exact") == "oneshot" else math.nan,
                    "ladder": ladder,
                    "best_unsplit": unsplit.get(bu, math.nan),
                    "best_split": split.get(bs, math.nan),
                    "best": times.get(best, math.nan),
                    "best_key": best,
                    "best_knobs": knobs(r.get(f"{best} variant")) if best else "",
                    "bu_knobs": knobs(r.get(f"{bu} variant")) if bu else "",
                    "bs_knobs": knobs(r.get(f"{bs} variant")) if bs else "",
                }
            )
    return pd.DataFrame(rows).sort_values(["tp", "h", "M"]) if rows else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "sweep_dir", help="directory holding tp{tp}_w{h}.csv from the fused sweep"
    )
    ap.add_argument(
        "--summary-csv", default=None, help="also write the per-shape summary here"
    )
    args = ap.parse_args()
    df = load(args.sweep_dir)
    if df.empty:
        sys.exit(f"no tp*_w*.csv in {args.sweep_dir}")
    if args.summary_csv:
        df.to_csv(args.summary_csv, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    for tp, g in df.groupby("tp"):
        print(f"\n## TP{tp}")
        out = g[
            [
                "h",
                "M",
                "kib",
                "cdr",
                "sep_cdr",
                "auto",
                "ladder",
                "best_unsplit",
                "best_split",
                "bu_knobs",
                "bs_knobs",
            ]
        ].copy()
        out["split/unsplit"] = (out["best_unsplit"] / out["best_split"]).round(3)
        out["best/cdr"] = (
            out["cdr"] / out[["best_unsplit", "best_split"]].min(axis=1)
        ).round(3)
        out["fusion gain (cdr)"] = (out["sep_cdr"] / out["cdr"]).round(3)
        print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
