"""Fit xGMI FUSED_ONESHOT_LADDER rungs and FUSED_FAMILY_POLICY bounds from
the fused sweep (<sweep_dir>/tp{tp}_w{h}.csv).

A rung is ``(min_bytes, atoms, grid_cap, fanout, split)`` and is
shared by every width, so a rung's time at (width, M) is the measured row whose
geometry the engine would resolve it to (``OneShotAllReduceRMSNorm._geom_for``
on a bare, unpinned, pad-on instance) -- looked up by the knobs parsed from each
row's variant. Rungs are chosen by DP over payload breakpoints minimizing the
summed log-regret against the per-shape best measured one-shot row.

usage: fit_fused.py SWEEP_DIR [--link xgmi] [--rungs N] [--metric "median us"|us] [--rung-penalty X]
"""

import argparse
import glob
import math
import os
import re
from collections import defaultdict

import pandas as pd

from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduceRMSNorm

SQNR_FLOOR = 40.0
_VAR = re.compile(
    r"_a(?P<a>\d+)_g(?P<g>\d+)_\w*?_b(?P<b>\d+)(?P<peer>_peer)?_rms_h\d+(?:p(?P<p>\d+))?"
    r"(?:_k(?P<k>\d+))?/"
)
ATOMS = (1, 2, 4)
CAPS = (8, 32, 64, 128, 256)
SPLITS = (1, 2, 4, 8, 16)


def parse(variant):
    m = _VAR.search(str(variant))
    if not m:
        return None
    return (
        int(m["a"]),
        int(m["g"]),
        int(m["b"]),
        int(m["k"] or 1),
        int(m["p"] or 0),
    )


def _bare():
    eng = OneShotAllReduceRMSNorm.__new__(OneShotAllReduceRMSNorm)
    eng.block = None
    eng.pad = True
    return eng


_ENG = _bare()


def resolve(h, atoms, cap, split):
    """Knob tuple the engine builds for this rung at width *h*."""
    a, h_pad, k = _ENG._geom_for(h, atoms, split)
    block = h_pad // (8 * a * k)
    cap_eff = max(k, cap // k * k)
    return (a, cap_eff, block, k, h_pad if h_pad != h else 0)


def load(sweep_dir, metric):
    """{tp: [shape dict]} with per-shape measured {knobs: us}, cdr and mesh."""
    data = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(sweep_dir, "tp*_w*.csv"))):
        d = pd.read_csv(path)
        keys = [c[: -len(" variant")] for c in d.columns if c.endswith(" variant")]
        fly = [k for k in keys if k.startswith("fused_fly_1stage")]
        for _, r in d.iterrows():

            def t(key):
                v, q = r.get(f"{key} {metric}"), r.get(f"{key} SQNR dB")
                if pd.isna(v) or (pd.notna(q) and q < SQNR_FLOOR and key.startswith("fused_fly_1s")):
                    return math.inf
                return float(v)

            meas = {}
            for k in fly:
                kn = parse(r.get(f"{k} variant"))
                v = t(k)
                if kn is None or v == math.inf:
                    continue
                meas[kn] = min(v, meas.get(kn, math.inf))
            data[int(r["TP"])].append(
                dict(
                    h=int(r["K"]),
                    M=int(r["M"]),
                    nbytes=int(r["_nbytes"]),
                    meas=meas,
                    best=min(meas.values()) if meas else math.inf,
                    cdr=min(t("fused_cdr_1stage"), t("fused_cdr_2stage")),
                    mesh=t("fused_fly_mesh"),
                    shipped=t("fused_fly_1stage"),
                )
            )
    return data


def configs():
    for a in ATOMS:
        for cap in CAPS:
            for k in SPLITS:
                yield (a, cap, k)


def cfg_time(shape, cfg):
    a, cap, k = cfg
    return shape["meas"].get(resolve(shape["h"], a, cap, k), math.inf)


def fit_ladder(shapes, max_rungs, penalty):
    """DP: split the payload-sorted shapes into <= max_rungs segments, each
    served by one config. Returns [(min_bytes, cfg, cost)]."""
    sizes = sorted({s["nbytes"] for s in shapes})
    by_size = defaultdict(list)
    for s in shapes:
        by_size[s["nbytes"]].append(s)
    cfgs = list(configs())
    # cost[c][i] = summed log-regret of cfg c over size bucket i
    cost = [[0.0] * len(sizes) for _ in cfgs]
    for ci, c in enumerate(cfgs):
        for i, n in enumerate(sizes):
            acc = 0.0
            for s in by_size[n]:
                v = cfg_time(s, c)
                acc += math.inf if v == math.inf else math.log(v / s["best"])
            cost[ci][i] = acc
    prefix = [[0.0] * (len(sizes) + 1) for _ in cfgs]
    for ci in range(len(cfgs)):
        for i in range(len(sizes)):
            prefix[ci][i + 1] = prefix[ci][i] + cost[ci][i]

    def seg(i, j):  # best cfg for buckets [i, j)
        best = (math.inf, None)
        for ci in range(len(cfgs)):
            v = prefix[ci][j] - prefix[ci][i]
            if v < best[0]:
                best = (v, ci)
        return best

    n = len(sizes)
    segc = {}
    for i in range(n):
        for j in range(i + 1, n + 1):
            segc[(i, j)] = seg(i, j)
    INF = math.inf
    dp = [[INF] * (n + 1) for _ in range(max_rungs + 1)]
    back = [[None] * (n + 1) for _ in range(max_rungs + 1)]
    dp[0][0] = 0.0
    for r in range(1, max_rungs + 1):
        for j in range(1, n + 1):
            for i in range(j):
                if dp[r - 1][i] == INF:
                    continue
                v = dp[r - 1][i] + segc[(i, j)][0] + penalty
                if v < dp[r][j]:
                    dp[r][j], back[r][j] = v, i
    r = min(range(1, max_rungs + 1), key=lambda r: dp[r][n])
    rungs, j = [], n
    while r > 0:
        i = back[r][j]
        c, ci = segc[(i, j)]
        rungs.append((0 if i == 0 else sizes[i], cfgs[ci], c))
        j, r = i, r - 1
    return rungs[::-1]


def ladder_time(shape, rungs):
    cfg = [c for mb, c, _ in rungs if shape["nbytes"] >= mb][-1]
    return cfg_time(shape, cfg)


def best_threshold(shapes, below, above):
    """Byte threshold T minimizing sum log(time), below(s) if nbytes <= T else above(s)."""
    sizes = sorted({s["nbytes"] for s in shapes})
    best = (math.inf, 0)
    for T in [0] + sizes:
        tot = 0.0
        for s in shapes:
            v = below(s) if s["nbytes"] <= T else above(s)
            tot += math.log(v) if v < math.inf else 50.0
        if tot < best[0] - 1e-9:
            best = (tot, T)
    return best[1]


def fmt_rung(mb, cfg):
    a, cap, k = cfg
    kb = f"{mb >> 10} << 10" if mb else "0"
    return f'({kb}, {a}, {cap}, "peer", {k})'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("sweep_dir", help="directory holding tp{tp}_w{h}.csv from the fused sweep")
    ap.add_argument("--link", default="xgmi", help="fabric the sweep ran on (labels the output)")
    ap.add_argument("--rungs", type=int, default=3)
    ap.add_argument("--metric", default="median us")
    ap.add_argument("--rung-penalty", type=float, default=0.25)
    args = ap.parse_args()
    data = load(args.sweep_dir, args.metric)
    if not data:
        raise SystemExit(f"no tp*_w*.csv in {args.sweep_dir}")
    for tp in sorted(data):
        shapes = data[tp]
        # Window: fit on everything first, then refit inside the exact window.
        rungs = fit_ladder(shapes, args.rungs, args.rung_penalty)
        exact_max = best_threshold(shapes, lambda s: ladder_time(s, rungs), lambda s: s["cdr"])
        win = [s for s in shapes if s["nbytes"] <= exact_max]
        if win:
            rungs = fit_ladder(win, args.rungs, args.rung_penalty)
            exact_max = best_threshold(shapes, lambda s: ladder_time(s, rungs), lambda s: s["cdr"])
        fast_max = best_threshold(shapes, lambda s: ladder_time(s, rungs), lambda s: s["mesh"])
        print(f"\n## TP{tp}  ({args.link}, metric={args.metric!r})")
        print("FUSED_ONESHOT_LADDER rungs:")
        for mb, c, cost in rungs:
            print(f"    {fmt_rung(mb, c)},   # seg log-regret {cost:.3f}")
        print(f"oneshot_max_exact (vs cdr)  = {exact_max} B ({exact_max / 1024:.0f} KiB)")
        print(f"oneshot_max       (vs mesh) = {fast_max} B ({fast_max / 1024:.0f} KiB)")
        print(f"{'h':>5} {'M':>5} {'KiB':>6} {'best':>7} {'ladder':>7} {'regret':>6} "
              f"{'shipped':>8} {'cdr':>7} {'mesh':>7} {'lad/cdr':>7}  win")
        for s in sorted(shapes, key=lambda s: (s["h"], s["M"])):
            lt = ladder_time(s, rungs)
            mark = "*" if s["nbytes"] <= exact_max else ""
            print(
                f"{s['h']:>5} {s['M']:>5} {s['nbytes'] >> 10:>6} {s['best']:>7.2f} {lt:>7.2f} "
                f"{lt / s['best']:>6.3f} {s['shipped']:>8.2f} {s['cdr']:>7.2f} {s['mesh']:>7.2f} "
                f"{s['cdr'] / lt:>7.3f}  {mark}"
            )
        inw = [s for s in shapes if s["nbytes"] <= exact_max]
        if inw:
            g = lambda f: math.exp(sum(math.log(f(s)) for s in inw) / len(inw))
            print(
                f"in-window geomean: ladder/best={g(lambda s: ladder_time(s, rungs) / s['best']):.3f}  "
                f"cdr/ladder={g(lambda s: s['cdr'] / ladder_time(s, rungs)):.3f}  "
                f"shipped/ladder={g(lambda s: s['shipped'] / ladder_time(s, rungs)):.3f}"
            )


if __name__ == "__main__":
    main()
