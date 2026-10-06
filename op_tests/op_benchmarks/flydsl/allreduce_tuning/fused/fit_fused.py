"""Fit the fused all-reduce + RMSNorm dispatch tables from the fused sweep.

Reads <sweep_dir>/tp{tp}_w{h}.csv (one-shot rows and baselines) and, when
present, <sweep_dir>/qr_tp{tp}_w{h}.csv (the quantized mesh/ring grid), and fits:

* ``FUSED_ONESHOT_LADDER`` -- a rung is ``(min_bytes, atoms, grid_cap, fanout,
  split)`` and is shared by every width, so a rung's time at (width, M) is the
  measured row whose geometry the engine would resolve it to
  (``OneShotAllReduceRMSNorm._geom_for`` on a bare, unpinned instance)
  -- looked up by the knobs parsed from each row's variant.
* ``FUSED_MESH_ST_LADDER`` / ``FUSED_RING_ST_LADDER`` and ``FUSED_QR_ROW_ATOMS``
  -- per schedule, one ``atoms_per_row`` for every width (a width that lacks it
  runs the nearest one it has, as the engine does) and a ``(min_bytes,
  super_tile, grid_cap)`` ladder, from the ``fused_fly_<alg>_a<apr>_st<st>_g<cap>``
  rows. Super-tile and grid cap are chosen together, per rung.
* ``FUSED_FAMILY_POLICY`` -- ``oneshot_max_exact`` against cdr, ``oneshot_max``
  against the fitted quantized schedules, and ``mesh_max`` between the two.

Ladders are chosen by DP over payload breakpoints minimizing the summed
log-regret against the per-shape best measured row of that family.

usage: fit_fused.py SWEEP_DIR [--link xgmi] [--rungs N] [--metric "median us"|us] [--rung-penalty X]
"""

import argparse
import functools
import glob
import math
import os
import re
from collections import defaultdict

import pandas as pd

from aiter.ops.flydsl.kernels.quick_allreduce_fusions import fused_qr_row_atoms
from aiter.ops.flydsl.kernels.quick_allreduce_mesh import fused_mesh_st_ladder
from aiter.ops.flydsl.kernels.quick_allreduce_ring import fused_ring_st_ladder
from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduceRMSNorm
from aiter.ops.flydsl.quick_allreduce import (
    ALGORITHMS,
    FlyQuickAllReduceRMSNorm,
    _resolve_codecs,
)

SQNR_FLOOR = 40.0
# The quantized rows' floors, as in bench_comm_allreduce.py.
QR_SQNR_FLOOR = {"mesh": 15.0, "ring": 10.0}
QR_ALGOS = ("mesh", "ring")
SHIPPED_QR_LADDER = {"mesh": fused_mesh_st_ladder, "ring": fused_ring_st_ladder}
_VAR = re.compile(
    r"_a(?P<a>\d+)_g(?P<g>\d+)_\w*?_b(?P<b>\d+)(?P<peer>_peer)?_rms_h\d+(?:p(?P<p>\d+))?"
    r"(?:_k(?P<k>\d+))?/"
)
_QR_KEY = re.compile(
    r"^fused_fly_(?P<alg>mesh|ring)_a(?P<a>\d+)_st(?P<st>\d+)_g(?P<g>\d+)$"
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
    return eng


_ENG = _bare()


def resolve(h, atoms, cap, split):
    """Knob tuple the engine builds for this rung at width *h*."""
    a, h_pad, k = _ENG._geom_for(h, atoms, split)
    block = h_pad // (8 * a * k)
    cap_eff = max(k, cap // k * k)
    return (a, cap_eff, block, k, h_pad if h_pad != h else 0)


@functools.cache
def qr_resolve(h, tp, alg, apr):
    """``atoms_per_row`` a fused *alg* engine pinned to *apr* (None: unpinned)
    actually builds at width *h* -- the nearest the width has -- or None when
    the width does not fuse at all. ``FlyQuickAllReduceRMSNorm._geom_for`` on a
    bare instance."""
    algo = ALGORITHMS[alg]
    eng = FlyQuickAllReduceRMSNorm.__new__(FlyQuickAllReduceRMSNorm)
    eng.world_size = int(tp)
    eng._algo = algo
    eng.rs_codec, _ag = _resolve_codecs(algo, int(tp), None, None)
    eng.block = None
    eng.atoms_per_row = None if apr is None else int(apr)
    if not eng.supports_hidden(int(h)):
        return None
    return eng._geom_for(int(h))[1]


def load(sweep_dir, metric):
    """{tp: [shape dict]} with per-shape measured {knobs: us}, cdr and mesh.

    ``qr`` holds the quantized grid, ``{alg: {(atoms_per_row, st): us}}``, joined
    on (TP, K, M) from the ``qr_`` CSVs; empty for a sweep that has none.
    """
    qr = defaultdict(lambda: defaultdict(dict))
    for path in sorted(glob.glob(os.path.join(sweep_dir, "qr_tp*_w*.csv"))):
        d = pd.read_csv(path)
        keys = [c[: -len(" variant")] for c in d.columns if c.endswith(" variant")]
        for _, r in d.iterrows():
            for k in keys:
                m = _QR_KEY.match(k)
                if not m:
                    continue
                v, q = r.get(f"{k} {metric}"), r.get(f"{k} SQNR dB")
                if pd.isna(v) or (pd.notna(q) and q < QR_SQNR_FLOOR[m["alg"]]):
                    continue
                shape = (int(r["TP"]), int(r["K"]), int(r["M"]))
                knobs = (int(m["a"]), int(m["st"]), int(m["g"]))
                qr[shape][m["alg"]][knobs] = float(v)

    data = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(sweep_dir, "tp*_w*.csv"))):
        d = pd.read_csv(path)
        keys = [c[: -len(" variant")] for c in d.columns if c.endswith(" variant")]
        fly = [k for k in keys if k.startswith("fused_fly_1stage")]
        for _, r in d.iterrows():

            def t(key):
                v, q = r.get(f"{key} {metric}"), r.get(f"{key} SQNR dB")
                if pd.isna(v) or (
                    pd.notna(q) and q < SQNR_FLOOR and key.startswith("fused_fly_1s")
                ):
                    return math.inf
                return float(v)

            meas = {}
            for k in fly:
                kn = parse(r.get(f"{k} variant"))
                v = t(k)
                if kn is None or v == math.inf:
                    continue
                meas[kn] = min(v, meas.get(kn, math.inf))
            shape = (int(r["TP"]), int(r["K"]), int(r["M"]))
            data[int(r["TP"])].append(
                dict(
                    tp=int(r["TP"]),
                    h=int(r["K"]),
                    M=int(r["M"]),
                    nbytes=int(r["_nbytes"]),
                    meas=meas,
                    best=min(meas.values()) if meas else math.inf,
                    cdr=min(t("fused_cdr_1stage"), t("fused_cdr_2stage")),
                    mesh=t("fused_fly_mesh"),
                    shipped=t("fused_fly_1stage"),
                    qr={alg: dict(qr[shape].get(alg, {})) for alg in QR_ALGOS},
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


def qr_time(shape, alg, apr, cfg):
    """Measured time of a fused *alg* engine pinned to *apr* at *cfg* =
    ``(super_tile, grid_cap)``."""
    st, cap = cfg
    got = qr_resolve(shape["h"], shape["tp"], alg, apr)
    meas = shape["qr"][alg]
    caps = [g for a, t, g in meas if (a, t) == (got, st) and g <= cap]
    return meas[(got, st, max(caps))] if caps else math.inf


def fit_ladder(shapes, max_rungs, penalty, cfgs, time, best):
    """DP: split the payload-sorted shapes into <= max_rungs segments, each
    served by one of *cfgs*. ``time(shape, cfg)`` is a config's measured time,
    ``best(shape)`` the regret reference. Returns [(min_bytes, cfg, cost)]."""
    sizes = sorted({s["nbytes"] for s in shapes})
    by_size = defaultdict(list)
    for s in shapes:
        by_size[s["nbytes"]].append(s)
    cfgs = list(cfgs)
    # cost[c][i] = summed log-regret of cfg c over size bucket i
    cost = [[0.0] * len(sizes) for _ in cfgs]
    for ci, c in enumerate(cfgs):
        for i, n in enumerate(sizes):
            acc = 0.0
            for s in by_size[n]:
                v = time(s, c)
                acc += math.inf if v == math.inf else math.log(v / best(s))
            cost[ci][i] = acc
    prefix = [[0.0] * (len(sizes) + 1) for _ in cfgs]
    for ci in range(len(cfgs)):
        for i in range(len(sizes)):
            prefix[ci][i + 1] = prefix[ci][i] + cost[ci][i]

    def seg(i, j):  # best cfg for buckets [i, j)
        best_seg = (math.inf, None)
        for ci in range(len(cfgs)):
            v = prefix[ci][j] - prefix[ci][i]
            if v < best_seg[0]:
                best_seg = (v, ci)
        return best_seg

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
    if dp[r][n] == INF:
        return None
    rungs, j = [], n
    while r > 0:
        i = back[r][j]
        c, ci = segc[(i, j)]
        rungs.append((0 if i == 0 else sizes[i], cfgs[ci], c))
        j, r = i, r - 1
    return rungs[::-1]


def rung_cfg(shape, rungs):
    return [c for mb, c, *_ in rungs if shape["nbytes"] >= mb][-1]


def ladder_time(shape, rungs):
    return cfg_time(shape, rung_cfg(shape, rungs))


def qr_best(shape, alg):
    meas = shape["qr"][alg]
    return min(meas.values()) if meas else math.inf


def fit_qr(shapes, alg, max_rungs, penalty):
    """``(atoms_per_row, rungs, cost by atoms_per_row)`` for one schedule, or
    None when the sweep has no rows for it.

    One ``atoms_per_row`` serves every width, so it is the outer choice: each
    candidate gets its own ``(super_tile, grid_cap)`` ladder, and the one with
    the least total log-regret wins (ties to the narrower row, i.e. the wider
    block).
    """
    shapes = [s for s in shapes if s["qr"][alg]]
    if not shapes:
        return None
    tp = shapes[0]["tp"]
    cfgs = sorted({(st, g) for s in shapes for _a, st, g in s["qr"][alg]})
    fits = {}
    for apr in ATOMS:
        if (8 // tp) % apr:
            continue
        rungs = fit_ladder(
            shapes,
            max_rungs,
            penalty,
            cfgs,
            lambda s, cfg, apr=apr: qr_time(s, alg, apr, cfg),
            lambda s: qr_best(s, alg),
        )
        if rungs is not None:
            fits[apr] = (sum(c for *_, c in rungs), rungs)
    if not fits:
        return None
    apr = min(fits, key=lambda a: (fits[a][0], a))
    return apr, fits[apr][1], {a: c for a, (c, _r) in fits.items()}


def qr_ladder_time(shape, alg, apr, rungs):
    return qr_time(shape, alg, apr, rung_cfg(shape, rungs))


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


def _kb(mb):
    if not mb:
        return "0"
    if mb % (1 << 20) == 0:
        return f"{mb >> 20} << 20"
    return f"{mb >> 10} << 10"


def fmt_rung(mb, cfg):
    a, cap, k = cfg
    return f'({_kb(mb)}, {a}, {cap}, "peer", {k})'


def geomean(vals):
    vals = [v for v in vals if 0 < v < math.inf]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else math.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "sweep_dir", help="directory holding tp{tp}_w{h}.csv from the fused sweep"
    )
    ap.add_argument(
        "--link", default="xgmi", help="fabric the sweep ran on (labels the output)"
    )
    ap.add_argument("--rungs", type=int, default=3)
    ap.add_argument("--metric", default="median us")
    ap.add_argument("--rung-penalty", type=float, default=0.25)
    args = ap.parse_args()
    data = load(args.sweep_dir, args.metric)
    if not data:
        raise SystemExit(f"no tp*_w*.csv in {args.sweep_dir}")
    for tp in sorted(data):
        fit_tp(tp, data[tp], args)


def fit_tp(tp, shapes, args):
    """Fit and print every fused table for one world size."""
    print(f"\n## TP{tp}  ({args.link}, metric={args.metric!r})")

    # Fused one-shot: fit on everything first, then refit inside the exact
    # window.
    rungs = fit_ladder(
        shapes, args.rungs, args.rung_penalty, configs(), cfg_time, lambda s: s["best"]
    )
    exact_max = best_threshold(
        shapes, lambda s: ladder_time(s, rungs), lambda s: s["cdr"]
    )
    win = [s for s in shapes if s["nbytes"] <= exact_max]
    if win:
        rungs = fit_ladder(
            win, args.rungs, args.rung_penalty, configs(), cfg_time, lambda s: s["best"]
        )
        exact_max = best_threshold(
            shapes, lambda s: ladder_time(s, rungs), lambda s: s["cdr"]
        )

    # Quantized schedules: fit each on every shape, take the one-shot
    # boundary against the faster of the two, then refit each above it --
    # a rung below the boundary is an engine nothing dispatches to. Once the
    # boundaries are set, each is refit once more on its own window below.
    qfit = {alg: fit_qr(shapes, alg, args.rungs, args.rung_penalty) for alg in QR_ALGOS}
    have = [alg for alg in QR_ALGOS if qfit[alg] is not None]

    def qtime(s, alg):
        apr, qrungs, _ = qfit[alg]
        return qr_ladder_time(s, alg, apr, qrungs)

    if have:
        fast_max = best_threshold(
            shapes,
            lambda s: ladder_time(s, rungs),
            lambda s: min(qtime(s, alg) for alg in have),
        )
        above = [s for s in shapes if s["nbytes"] > fast_max]
        for alg in have:
            refit = fit_qr(above, alg, args.rungs, args.rung_penalty)
            if refit is not None:
                qfit[alg] = refit
        top = max(s["nbytes"] for s in shapes)
        if have == ["mesh"]:
            mesh_max = None
        elif have == ["ring"]:
            mesh_max = fast_max
        else:
            mesh_max = best_threshold(
                above or shapes, lambda s: qtime(s, "mesh"), lambda s: qtime(s, "ring")
            )
            if mesh_max >= top:
                mesh_max = None

        def quant(s):
            if mesh_max is None or s["nbytes"] <= mesh_max:
                return qtime(s, "mesh")
            return qtime(s, "ring")

        fast_max = best_threshold(shapes, lambda s: ladder_time(s, rungs), quant)
        if mesh_max is not None and mesh_max < fast_max:
            mesh_max = fast_max  # the mesh window is empty
    windows = {
        "mesh": [
            s
            for s in shapes
            if s["nbytes"] > fast_max and (mesh_max is None or s["nbytes"] <= mesh_max)
        ],
        "ring": [s for s in shapes if mesh_max is not None and s["nbytes"] > mesh_max],
    }
    if have:
        # A family's ladder only ever serves its own window, so fit it there:
        # the boundaries above were taken with ladders fitted across both
        # windows, and the shapes the other family serves must not pick this
        # one's rungs. A family whose window holds no swept shape keeps no fit.
        for alg in have:
            refit = (
                fit_qr(windows[alg], alg, args.rungs, args.rung_penalty)
                if windows[alg]
                else None
            )
            if refit is not None:
                qfit[alg] = refit
    else:
        print(
            "(no qr_tp*_w*.csv rows: mesh/ring not fitted; oneshot_max is against fused_fly_mesh)"
        )
        mesh_max = None
        fast_max = best_threshold(
            shapes, lambda s: ladder_time(s, rungs), lambda s: s["mesh"]
        )

    print("FUSED_ONESHOT_LADDER rungs:")
    for mb, c, cost in rungs:
        print(f"    {fmt_rung(mb, c)},   # seg log-regret {cost:.3f}")
    for alg in have:
        apr, qrungs, by_apr = qfit[alg]
        table = f"FUSED_{alg.upper()}_ST_LADDER"
        if not windows[alg]:
            print(
                f"{table}: not dispatched -- no swept shape in its window; keep the "
                "shipped rungs and FUSED_QR_ROW_ATOMS entry"
            )
            continue
        print(
            f"{table} rungs (atoms_per_row={apr}, fitted on the "
            f"{len(windows[alg])} shapes in its window):"
        )
        for mb, (st, cap), cost in qrungs:
            print(f"    ({_kb(mb)}, {st}, {cap}),   # seg log-regret {cost:.3f}")
        print(f'FUSED_QR_ROW_ATOMS[("{args.link}", {tp}, "{alg}")] = {apr}')
        print(
            "    log-regret by atoms_per_row: "
            + ", ".join(f"{a}: {c:.3f}" for a, c in sorted(by_apr.items()))
        )
        # What ships today on this fabric, timed off the same pinned rows.
        ship_apr = fused_qr_row_atoms(tp, alg, args.link)
        ship = tuple(
            (mb, (st, cap), 0.0)
            for mb, st, cap in SHIPPED_QR_LADDER[alg](tp, args.link)
        )
        ratio = geomean(
            qr_ladder_time(s, alg, ship_apr, ship) / qr_ladder_time(s, alg, apr, qrungs)
            for s in windows[alg]
        )
        print(
            f"    shipped/fitted in its window: {ratio:.3f}  (shipped atoms_per_row={ship_apr})"
        )
    print(f"oneshot_max_exact (vs cdr)  = {exact_max} B ({exact_max / 1024:.0f} KiB)")
    print(f"oneshot_max       (vs quant) = {fast_max} B ({fast_max / 1024:.0f} KiB)")
    mesh_txt = "None" if mesh_max is None else _kb(mesh_max)
    if have:
        print(f"mesh_max          (vs ring)  = {mesh_txt}")
        print(
            f"FusedPolicy(oneshot_max={_kb(fast_max)}, oneshot_max_exact={_kb(exact_max)}, "
            f"mesh_max={mesh_txt}, ring_max=None)"
        )
    print(
        f"{'h':>5} {'M':>5} {'KiB':>6} {'best':>7} {'ladder':>7} {'regret':>6} "
        f"{'shipped':>8} {'cdr':>7} {'mesh':>7} {'ring':>7} {'lad/cdr':>7}  win"
    )
    for s in sorted(shapes, key=lambda s: (s["h"], s["M"])):
        lt = ladder_time(s, rungs)
        mt = qtime(s, "mesh") if "mesh" in have else s["mesh"]
        rt = qtime(s, "ring") if "ring" in have else math.nan
        mark = "*" if s["nbytes"] <= exact_max else ""
        print(
            f"{s['h']:>5} {s['M']:>5} {s['nbytes'] >> 10:>6} {s['best']:>7.2f} {lt:>7.2f} "
            f"{lt / s['best']:>6.3f} {s['shipped']:>8.2f} {s['cdr']:>7.2f} {mt:>7.2f} {rt:>7.2f} "
            f"{s['cdr'] / lt:>7.3f}  {mark}"
        )
    inw = [s for s in shapes if s["nbytes"] <= exact_max]
    if inw:
        print(
            f"in-window geomean: ladder/best={geomean(ladder_time(s, rungs) / s['best'] for s in inw):.3f}  "
            f"cdr/ladder={geomean(s['cdr'] / ladder_time(s, rungs) for s in inw):.3f}  "
            f"shipped/ladder={geomean(s['shipped'] / ladder_time(s, rungs) for s in inw):.3f}"
        )


if __name__ == "__main__":
    main()
