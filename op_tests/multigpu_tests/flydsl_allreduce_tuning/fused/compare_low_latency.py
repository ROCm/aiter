"""Is the low-latency mesh build worth a ladder rung? Compare it with the
shipping fused mesh in the ``tune.sh low-latency`` sweep.

Reads <sweep_dir>/ll_tp{tp}_w{h}.csv: the ``ar_rmsnorm`` bench of
``test_flydsl_quick_allreduce.py -a mesh`` with ``--candidate-set fused_ll``.
Per shape:

* ``ll`` -- the best ``fused_fly_mesh_ll_st1_g<cap>`` row, and its cap;
* ``mesh`` -- the shipping fused mesh (the policy's ladder at this width), and
  the kernel it ran; ``mesh/ll`` is what a low-latency rung would gain;
* ``oneshot``, ``hip_fused`` -- the shipping fused one-shot and the HIP quick
  all-reduce's fused kernel, for context: where the one-shot is faster, the
  mesh does not dispatch.

Ratios above 1 mean the low-latency build is faster. Rows below the fused fit's
mesh SQNR floor are dropped, and a low-latency column whose variant is not a
low-latency kernel is reported and ignored.

usage: compare_low_latency.py SWEEP_DIR [--link L] [--min-gain G] [--csv FILE]
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

# The mesh rows' floor, as in fit_fused.py.
MESH_SQNR_FLOOR = 15.0
_LL = re.compile(r"^fused_fly_mesh_ll_st1_g(?P<g>\d+)$")
# ``<jit symbol>/g<grid>/x<grid_x>``: the super-tile and grid the mesh ran.
_MESH_VAR = re.compile(r"_st(?P<st>\d+)_.*/g(?P<g>\d+)/")


def _time(r, key, floor) -> float:
    """*key*'s time in row *r*, or NaN when it did not run or missed *floor*."""
    v, q = r.get(f"{key} us"), r.get(f"{key} SQNR dB")
    if v is None or pd.isna(v) or (floor is not None and pd.notna(q) and q < floor):
        return math.nan
    return float(v)


def _mesh_knobs(variant) -> str:
    m = _MESH_VAR.search(str(variant))
    return f"st{m['st']} g{m['g']}" if m else "?"


def load(sweep_dir):
    """``(per-shape DataFrame, arch, [skipped low-latency cells])``."""
    rows, archs, skipped = [], set(), []
    for path in sorted(glob.glob(os.path.join(sweep_dir, "ll_tp*_w*.csv"))):
        d = bench_io.read_bench_csv(path)
        archs |= set(d["gfx"].dropna())
        keys = [c[: -len(" variant")] for c in d.columns if c.endswith(" variant")]
        ll_keys = [k for k in keys if _LL.match(k)]
        for _, r in d.iterrows():
            tp, h, m = int(r["TP"]), int(r["K"]), int(r["M"])
            ll = {}
            for k in ll_keys:
                v = _time(r, k, MESH_SQNR_FLOOR)
                if math.isnan(v):
                    continue
                variant = str(r.get(f"{k} variant"))
                if "_low_lat" not in variant:
                    skipped.append(f"tp{tp} h{h} M{m} {k}: ran {variant}")
                    continue
                ll[k] = v
            best = min(ll, key=ll.get) if ll else None
            rows.append(
                {
                    "tp": tp,
                    "h": h,
                    "M": m,
                    "kib": int(r["_nbytes"]) // 1024,
                    "ll": ll.get(best, math.nan),
                    "ll_cap": int(_LL.match(best)["g"]) if best else None,
                    "ll_spread": r.get(f"{best} spread us") if best else math.nan,
                    "mesh": _time(r, "fused_fly_mesh", MESH_SQNR_FLOOR),
                    "mesh_knobs": _mesh_knobs(r.get("fused_fly_mesh variant")),
                    "mesh_spread": r.get("fused_fly_mesh spread us"),
                    "oneshot": _time(r, "oneshot", None),
                    "hip_fused": _time(r, "fused_qr_int4", None),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["tp", "h", "M"]).reset_index(drop=True)
        df["mesh/ll"] = df["mesh"] / df["ll"]
    if len(archs) > 1:
        raise SystemExit(f"{sweep_dir} mixes archs {sorted(archs)}")
    return df, (archs.pop() if archs else None), skipped


def _geomean(s: pd.Series) -> float:
    s = s.dropna()
    return math.exp(s.map(math.log).mean()) if len(s) else math.nan


def summarize(df: pd.DataFrame, min_gain: float) -> pd.DataFrame:
    out = []
    for (tp, h), g in df.groupby(["tp", "h"]):
        ok = g.dropna(subset=["mesh/ll"])
        wins = ok[ok["mesh/ll"] > 1 + min_gain]
        losses = ok[ok["mesh/ll"] < 1 / (1 + min_gain)]
        top = ok.loc[ok["mesh/ll"].idxmax()] if len(ok) else None
        out.append(
            {
                "tp": tp,
                "h": h,
                "shapes": len(g),
                "compared": len(ok),
                "geo mesh/ll": _geomean(ok["mesh/ll"]),
                "ll wins": len(wins),
                "ll loses": len(losses),
                "win KiB": (
                    f"{wins['kib'].min()}..{wins['kib'].max()}" if len(wins) else "-"
                ),
                "max mesh/ll": top["mesh/ll"] if top is not None else math.nan,
                "at KiB": top["kib"] if top is not None else None,
            }
        )
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__
    )
    ap.add_argument("sweep_dir", help="directory holding ll_tp{tp}_w{h}.csv")
    ap.add_argument("--link", default="?", help="fabric label for the report")
    ap.add_argument(
        "--min-gain",
        type=float,
        default=0.03,
        help="speedup over the shipping mesh, minus 1, that counts as a win (and "
        "its inverse as a loss); default 0.03, below that is timing noise",
    )
    ap.add_argument("--csv", default=None, help="also write the per-shape table here")
    args = ap.parse_args()
    df, arch, skipped = load(args.sweep_dir)
    if df.empty:
        sys.exit(f"no ll_tp*_w*.csv in {args.sweep_dir} (run: tune.sh low-latency)")
    if args.csv:
        df.assign(arch=arch, link=args.link).to_csv(args.csv, index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(f"# low-latency mesh vs the shipping fused mesh ({arch}, {args.link})")
    print("mesh/ll > 1: the low-latency build is faster; times in us")
    if skipped:
        print(f"\nignored {len(skipped)} low-latency cell(s) that ran another build:")
        for s in skipped:
            print(f"  {s}")
    cols = [
        "M",
        "kib",
        "ll",
        "ll_cap",
        "ll_spread",
        "mesh",
        "mesh_knobs",
        "mesh_spread",
        "mesh/ll",
        "oneshot",
        "hip_fused",
    ]
    for (tp, h), g in df.groupby(["tp", "h"]):
        print(f"\n## TP{tp} hidden={h}")
        print(g[cols].round(3).to_string(index=False))
    summary = summarize(df, args.min_gain)
    print(f"\n## summary (win: mesh/ll > {1 + args.min_gain:g})")
    print(summary.round(3).to_string(index=False))
    n_win, n = int(summary["ll wins"].sum()), int(summary["compared"].sum())
    print(
        f"\nverdict ({arch}, {args.link}): the low-latency build beats the shipping "
        f"mesh by > {args.min_gain:.0%} at {n_win} of {n} shape(s); "
        f"geomean mesh/ll {_geomean(df['mesh/ll']):.3f}"
    )


if __name__ == "__main__":
    sys.exit(main())
