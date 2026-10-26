"""Is the low-latency mesh build worth a ladder rung? Compare it with what the
fused policy dispatches today, and with the default build at its geometry.

Reads <sweep_dir>/ll_tp{tp}_w{h}.csv: the ``ar_rmsnorm`` bench of
``test_flydsl_quick_allreduce.py`` with ``--candidate-set fused_ll``. Per shape:

* ``ll`` -- the best ``fused_fly_mesh_ll_st1_g<cap>`` row, and its cap;
* ``auto`` -- the shipping engine of the family the fused policy routes the
  payload to (``route``: oneshot, mesh or ring; ``fallback`` reads the best of
  ``CustomAllreduce``'s fused kernels, what the caller runs instead).
  ``auto/ll`` is what a low-latency rung would gain in production;
* ``twin`` -- the best ``fused_fly_mesh_b256_st1_g<cap>`` row: the default
  build at the low-latency build's block and super-tile, so ``twin/ll``
  separates what the low-latency build changes from what its block or cap does;
* ``mesh`` -- the shipping fused mesh, also where the policy does not route to
  it (``mesh/ll``); its ladder is only fitted where the mesh dispatches.

Ratios above 1 mean the low-latency build is faster. Rows below the fused fit's
SQNR floors are dropped, and a low-latency column whose variant is not a
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

# The quantized rows' floors, as in fit_fused.py.
QR_SQNR_FLOOR = {"mesh": 15.0, "ring": 10.0}
_LL = re.compile(r"^fused_fly_mesh_ll_st1_g(?P<g>\d+)$")
_TWIN = re.compile(r"^fused_fly_mesh_b256_st1_g(?P<g>\d+)$")
# ``<jit symbol>/g<grid>/x<grid_x>``: the super-tile and grid a mesh/ring ran.
_VAR = re.compile(r"_st(?P<st>\d+)_.*/g(?P<g>\d+)/")
# The quick test's routed shipping engine is its ``flydsl`` column, which
# bench_io reads as ``fused_fly_1stage``.
_ROUTED = bench_io.SHIPPING[("ar_rmsnorm", "flydsl")]


def _time(r, key, floor) -> float:
    """*key*'s time in row *r*, or NaN when it did not run or missed *floor*."""
    v, q = r.get(f"{key} us"), r.get(f"{key} SQNR dB")
    if v is None or pd.isna(v) or (floor is not None and pd.notna(q) and q < floor):
        return math.nan
    return float(v)


def _knobs(variant) -> str:
    m = _VAR.search(str(variant))
    return f"st{m['st']} g{m['g']}" if m else ""


def _best(r, keys, floor, skipped=None, must=None):
    """``(us, cap)`` of the fastest of *keys*; a cell whose variant lacks
    *must* goes to *skipped* instead."""
    best, cap = math.nan, None
    for k, pat in keys:
        v = _time(r, k, floor)
        if math.isnan(v):
            continue
        if must is not None and must not in str(r.get(f"{k} variant")):
            skipped.append(f"{k}: ran {r.get(f'{k} variant')}")
            continue
        if math.isnan(best) or v < best:
            best, cap = v, int(pat.match(k)["g"])
    return best, cap


def load(sweep_dir):
    """``(per-shape DataFrame, arch, [skipped low-latency cells])``."""
    rows, archs, skipped = [], set(), []
    for path in sorted(glob.glob(os.path.join(sweep_dir, "ll_tp*_w*.csv"))):
        d = bench_io.read_bench_csv(path)
        archs |= set(d["gfx"].dropna())
        keys = [c[: -len(" variant")] for c in d.columns if c.endswith(" variant")]
        ll_keys = [(k, _LL) for k in keys if _LL.match(k)]
        twin_keys = [(k, _TWIN) for k in keys if _TWIN.match(k)]
        for _, r in d.iterrows():
            tp, h, m = int(r["TP"]), int(r["K"]), int(r["M"])
            bad = []
            ll, ll_cap = _best(r, ll_keys, QR_SQNR_FLOOR["mesh"], bad, "_low_lat")
            skipped += [f"tp{tp} h{h} M{m} {s}" for s in bad]
            twin, twin_cap = _best(r, twin_keys, QR_SQNR_FLOOR["mesh"])
            route = str(r.get("route"))
            if route == "fallback":
                auto = min(
                    _time(r, "fused_cdr_1stage", None),
                    _time(r, "fused_cdr_2stage", None),
                )
            else:
                auto = _time(r, _ROUTED, QR_SQNR_FLOOR.get(route))
            rows.append(
                {
                    "tp": tp,
                    "h": h,
                    "M": m,
                    "kib": int(r["_nbytes"]) // 1024,
                    "ll": ll,
                    "ll_cap": ll_cap,
                    "route": route,
                    "auto": auto,
                    "twin": twin,
                    "twin_cap": twin_cap,
                    "mesh": _time(r, "fused_fly_mesh", QR_SQNR_FLOOR["mesh"]),
                    "mesh_knobs": _knobs(r.get("fused_fly_mesh variant")),
                    "ring": _time(r, "fused_fly_ring", QR_SQNR_FLOOR["ring"]),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["tp", "h", "M"]).reset_index(drop=True)
        for col in ("auto", "twin", "mesh"):
            df[f"{col}/ll"] = df[col] / df["ll"]
    if len(archs) > 1:
        raise SystemExit(f"{sweep_dir} mixes archs {sorted(archs)}")
    return df, (archs.pop() if archs else None), skipped


def _geomean(s: pd.Series) -> float:
    s = s.dropna()
    return math.exp(s.map(math.log).mean()) if len(s) else math.nan


def _span(g: pd.DataFrame) -> str:
    return f"{g['kib'].min()}..{g['kib'].max()}" if len(g) else "-"


def summarize(df: pd.DataFrame, min_gain: float) -> pd.DataFrame:
    out = []
    for (tp, h), g in df.groupby(["tp", "h"]):
        ok = g.dropna(subset=["ll"])
        wins = ok[ok["auto/ll"] > 1 + min_gain]
        twin_wins = ok[ok["twin/ll"] > 1 + min_gain]
        top = ok.loc[ok["auto/ll"].idxmax()] if ok["auto/ll"].notna().any() else None
        out.append(
            {
                "tp": tp,
                "h": h,
                "shapes": len(g),
                "ll ran": len(ok),
                "geo auto/ll": _geomean(ok["auto/ll"]),
                "wins vs auto": len(wins),
                "at KiB": _span(wins),
                "routes there": ",".join(sorted(set(wins["route"]))) or "-",
                "geo twin/ll": _geomean(ok["twin/ll"]),
                "wins vs twin": len(twin_wins),
                "max auto/ll": top["auto/ll"] if top is not None else math.nan,
                "max at KiB": top["kib"] if top is not None else None,
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
        help="speedup, minus 1, that counts as a low-latency win (default 0.03: "
        "below that is timing noise)",
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
    print(f"# low-latency mesh vs what the fused policy runs ({arch}, {args.link})")
    print(
        "x/ll > 1: the low-latency build is faster than x; times in us. auto: the "
        "routed shipping engine; twin: the default build at block 256, ST=1"
    )
    if skipped:
        print(f"\nignored {len(skipped)} low-latency cell(s) that ran another build:")
        for s in skipped:
            print(f"  {s}")
    cols = [
        "M",
        "kib",
        "ll",
        "ll_cap",
        "route",
        "auto",
        "auto/ll",
        "twin",
        "twin_cap",
        "twin/ll",
        "mesh",
        "mesh_knobs",
        "mesh/ll",
        "ring",
    ]
    for (tp, h), g in df.groupby(["tp", "h"]):
        print(f"\n## TP{tp} hidden={h}")
        print(g[cols].round(3).to_string(index=False))
    summary = summarize(df, args.min_gain)
    print(f"\n## summary (win: x/ll > {1 + args.min_gain:g})")
    print(summary.round(3).to_string(index=False))
    n_win = int(summary["wins vs auto"].sum())
    n_twin = int(summary["wins vs twin"].sum())
    n = int(summary["ll ran"].sum())
    print(
        f"\nverdict ({arch}, {args.link}): of {n} shape(s) the low-latency build ran, "
        f"it beats what the policy dispatches by > {args.min_gain:.0%} at {n_win} "
        f"(geomean auto/ll {_geomean(df['auto/ll']):.3f}) and its default-build "
        f"twin at {n_twin} (geomean twin/ll {_geomean(df['twin/ll']):.3f})"
    )


if __name__ == "__main__":
    sys.exit(main())
