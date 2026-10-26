"""Is the low-latency mesh build worth a ladder rung? Compare it with every other
quantized (mesh/ring) instance of the ``tune.sh low-latency`` sweep.

Reads <sweep_dir>/ll_tp{tp}_w{h}.csv: the ``ar_rmsnorm`` bench of
``test_flydsl_quick_allreduce.py`` with ``--candidate-set fused_ll``. Per shape:

* ``ll`` -- the best ``fused_fly_mesh_ll_st1_g<cap>`` row, and its cap;
* ``twin`` -- the best ``fused_fly_mesh_a<apr>_st1_g<cap>`` row at the same
  geometry (block 256 and super-tile 1, hence ``apr = hidden / 2048``): the same
  kernel without the residual prefetch and the second workgroup per CU, so
  ``twin/ll`` isolates what the low-latency build changes;
* ``best_qr`` -- the best of every other quantized row (each pinned mesh and
  ring row and the shipping mesh/ring engines), and which; ``best_qr/ll`` is
  what a low-latency rung would gain over the best ladder the fit can build;
* ``mesh``, ``oneshot``, ``hip_fused`` -- the shipping fused mesh, the shipping
  fused one-shot and the HIP quick all-reduce's fused kernel, for context: at
  sizes where the one-shot is faster, the quantized schedules do not dispatch.

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
# The atoms of one 256-thread row: the low-latency build's only block.
LL_ROW = 2048
_LL = re.compile(r"^fused_fly_mesh_ll_st1_g(?P<g>\d+)$")
_QR = re.compile(
    r"^fused_fly_(?P<alg>mesh|ring)_a(?P<a>\d+)_st(?P<st>\d+)_g(?P<g>\d+)$"
)
SHIPPED_QR = {"fused_fly_mesh": "mesh", "fused_fly_ring": "ring"}


def _time(r, key, floor) -> float:
    """*key*'s time in row *r*, or NaN when it did not run or missed *floor*."""
    v, q = r.get(f"{key} us"), r.get(f"{key} SQNR dB")
    if v is None or pd.isna(v) or (floor is not None and pd.notna(q) and q < floor):
        return math.nan
    return float(v)


def _best(times: dict) -> tuple:
    if not times:
        return None, math.nan
    k = min(times, key=times.get)
    return k, times[k]


def _short(key):
    return None if key is None else key.removeprefix("fused_fly_")


def load(sweep_dir):
    """``(per-shape DataFrame, arch, [skipped low-latency cells])``."""
    rows, archs, skipped = [], set(), []
    for path in sorted(glob.glob(os.path.join(sweep_dir, "ll_tp*_w*.csv"))):
        d = bench_io.read_bench_csv(path)
        archs |= set(d["gfx"].dropna())
        keys = [c[: -len(" variant")] for c in d.columns if c.endswith(" variant")]
        for _, r in d.iterrows():
            tp, h, m = int(r["TP"]), int(r["K"]), int(r["M"])
            ll, twin, qr = {}, {}, {}
            for k in keys:
                if _LL.match(k):
                    v = _time(r, k, QR_SQNR_FLOOR["mesh"])
                    variant = str(r.get(f"{k} variant"))
                    if not math.isnan(v) and "_low_lat" not in variant:
                        skipped.append(f"tp{tp} h{h} M{m} {k}: ran {variant}")
                        continue
                    if not math.isnan(v):
                        ll[k] = v
                elif mt := _QR.match(k):
                    v = _time(r, k, QR_SQNR_FLOOR[mt["alg"]])
                    if math.isnan(v):
                        continue
                    qr[k] = v
                    if (
                        mt["alg"] == "mesh"
                        and mt["st"] == "1"
                        and int(mt["a"]) * LL_ROW == h
                    ):
                        twin[k] = v
            for k, alg in SHIPPED_QR.items():
                v = _time(r, k, QR_SQNR_FLOOR[alg])
                if not math.isnan(v):
                    qr[k] = v
            ll_key, ll_us = _best(ll)
            twin_key, twin_us = _best(twin)
            qr_key, qr_us = _best(qr)
            rows.append(
                {
                    "tp": tp,
                    "h": h,
                    "M": m,
                    "kib": int(r["_nbytes"]) // 1024,
                    "ll": ll_us,
                    "ll_cap": int(_LL.match(ll_key)["g"]) if ll_key else None,
                    "ll_spread": (
                        r.get(f"{ll_key} spread us") if ll_key else math.nan
                    ),
                    "twin": twin_us,
                    "twin_key": _short(twin_key),
                    "best_qr": qr_us,
                    "best_qr_key": _short(qr_key),
                    "mesh": qr.get("fused_fly_mesh", math.nan),
                    "oneshot": _time(r, "oneshot", None),
                    "hip_fused": _time(r, "fused_qr_int4", None),
                }
            )
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["tp", "h", "M"]).reset_index(drop=True)
        df["twin/ll"] = df["twin"] / df["ll"]
        df["best_qr/ll"] = df["best_qr"] / df["ll"]
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
        ok = g.dropna(subset=["ll", "best_qr"])
        wins = ok[ok["best_qr/ll"] > 1 + min_gain]
        top = ok.loc[ok["best_qr/ll"].idxmax()] if len(ok) else None
        out.append(
            {
                "tp": tp,
                "h": h,
                "shapes": len(g),
                "ll ran": int(g["ll"].notna().sum()),
                "geo twin/ll": _geomean(g["twin/ll"]),
                "geo best_qr/ll": _geomean(g["best_qr/ll"]),
                "geo mesh/ll": _geomean(g["mesh/ll"]),
                "ll wins": len(wins),
                "win KiB": (
                    f"{wins['kib'].min()}..{wins['kib'].max()}" if len(wins) else "-"
                ),
                "max best_qr/ll": top["best_qr/ll"] if top is not None else math.nan,
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
        help="speedup over the best other quantized row, minus 1, that counts as "
        "a low-latency win (default 0.03: below that is timing noise)",
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
    print(f"# low-latency mesh vs every other quantized instance ({arch}, {args.link})")
    print("ratios > 1: the low-latency build is faster; times in us")
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
        "twin",
        "twin_key",
        "twin/ll",
        "best_qr",
        "best_qr_key",
        "best_qr/ll",
        "mesh/ll",
        "oneshot",
        "hip_fused",
    ]
    for (tp, h), g in df.groupby(["tp", "h"]):
        print(f"\n## TP{tp} hidden={h}")
        print(g[cols].round(3).to_string(index=False))
    summary = summarize(df, args.min_gain)
    print(f"\n## summary (win: best_qr/ll > {1 + args.min_gain:g})")
    print(summary.round(3).to_string(index=False))
    n_win, n = int(summary["ll wins"].sum()), int(summary["ll ran"].sum())
    print(
        f"\nverdict ({arch}, {args.link}): the low-latency build beats every other "
        f"quantized instance by > {args.min_gain:.0%} at {n_win} of {n} shape(s) "
        f"it ran; geomean best_qr/ll {_geomean(df['best_qr/ll']):.3f}, "
        f"twin/ll {_geomean(df['twin/ll']):.3f}"
    )


if __name__ == "__main__":
    sys.exit(main())
