"""Summarize the fused AR+RMSNorm one-shot sweep (<sweep_dir>/tp{tp}_w{h}.csv).

Per (tp, width, M): incumbent cdr (min of fused 1stage/2stage), the shipped
fused one-shot ladder, best unsplit / best split pinned row, overall winner and
its resolved knobs (atoms, grid cap, split) parsed from the variant.
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

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


def load(sweep_dir):
    rows = []
    for path in sorted(glob.glob(os.path.join(sweep_dir, "tp*_w*.csv"))):
        d = pd.read_csv(path)
        fly = sorted(
            c[: -len(" variant")]
            for c in d.columns
            if c.startswith("fused_fly_1stage") and c.endswith(" variant")
        )
        for _, r in d.iterrows():

            def t(key):
                v = r.get(f"{key} us")
                q = r.get(f"{key} SQNR dB")
                if pd.isna(v) or (pd.notna(q) and q < SQNR_FLOOR):
                    return float("nan")
                return float(v)

            times = {k: t(k) for k in fly}
            times = {k: v for k, v in times.items() if v == v}
            unsplit = {k: v for k, v in times.items() if "_k" not in k[len("fused_fly_1stage") :]}
            split = {k: v for k, v in times.items() if k not in unsplit}
            best = min(times, key=times.get) if times else None
            bu = min(unsplit, key=unsplit.get) if unsplit else None
            bs = min(split, key=split.get) if split else None
            cdr = min(t("fused_cdr_1stage"), t("fused_cdr_2stage"))
            rows.append(
                dict(
                    tp=int(r["TP"]),
                    h=int(r["K"]),
                    M=int(r["M"]),
                    kib=int(r["_nbytes"]) // 1024,
                    cdr=cdr,
                    sep_cdr=t("separate_cdr"),
                    auto=t("fused_fly_auto"),
                    mesh=t("fused_fly_mesh"),
                    ladder=t("fused_fly_1stage"),
                    best_unsplit=unsplit.get(bu, float("nan")),
                    best_split=split.get(bs, float("nan")),
                    best=times.get(best, float("nan")),
                    best_key=best,
                    best_knobs=knobs(r.get(f"{best} variant")) if best else "",
                    bu_knobs=knobs(r.get(f"{bu} variant")) if bu else "",
                    bs_knobs=knobs(r.get(f"{bs} variant")) if bs else "",
                )
            )
    return pd.DataFrame(rows).sort_values(["tp", "h", "M"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_dir", help="directory holding tp{tp}_w{h}.csv from the fused sweep")
    ap.add_argument("--summary-csv", default=None, help="also write the per-shape summary here")
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
            ["h", "M", "kib", "cdr", "auto", "ladder", "best_unsplit", "best_split",
             "bu_knobs", "bs_knobs"]
        ].copy()
        out["split/unsplit"] = (out["best_unsplit"] / out["best_split"]).round(3)
        out["best/cdr"] = (out["cdr"] / out[["best_unsplit", "best_split"]].min(axis=1)).round(3)
        print(out.round(2).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
