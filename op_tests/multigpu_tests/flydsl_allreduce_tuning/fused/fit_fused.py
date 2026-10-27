"""Fit the fused all-reduce + RMSNorm dispatch tables from the fused sweep.

Reads <sweep_dir>/tp{tp}_w{h}.csv (the ``ar_rmsnorm`` bench of
``test_flydsl_one_shot_allreduce.py``: one-shot rows and baselines) and, when
present, <sweep_dir>/qr_tp{tp}_w{h}.csv (that of ``test_flydsl_quick_allreduce.py``:
the quantized mesh/ring grid), and fits the ``allreduce_policy`` tables:

* the fused one-shot ladders (``FUSED_ONESHOT_LADDER``) -- a default ladder,
  shared by every width and never split, and a ladder per swept width. A
  rung's time at (width, M) is the measured row whose geometry the engine would
  resolve it to (``OneShotAllReduceRMSNorm._geom_for`` on a bare, unpinned
  instance) -- looked up by the knobs parsed from each row's variant.
* the fused mesh and ring ladders (``FUSED_MESH_ST_LADDER`` /
  ``FUSED_RING_ST_LADDER``, with ``FUSED_QR_ROW_ATOMS``) -- per schedule, a
  default ``atoms_per_row`` and ``(min_bytes, super_tile, grid_cap)`` ladder
  for every width (a width that lacks that ``atoms_per_row`` runs the nearest
  one it has, as the engine does), and one per swept width, from the
  ``fused_fly_<alg>_a<apr>_st<st>_g<cap>`` rows. Super-tile and grid cap are
  chosen together, per rung. On xGMI a per-width mesh rung may also be the
  low-latency build (``fused_fly_mesh_ll_a<apr>_st1_g<cap>``), which exists
  only where the row's block is 256 threads -- one width per
  ``atoms_per_row`` -- so the default ladder, which runs at every width,
  never takes it (as the default one-shot ladder never splits).
* ``FUSED_FAMILY_POLICIES`` -- the exact one-shot window against cdr, the
  start of the quantized windows against what runs below them with
  quantization on (the one-shot inside its window, production's fallback
  above it), the mesh/ring boundary between the two, and the top of the
  quantized windows against production's fallback. That fallback is the HIP
  quick all-reduce's fused kernel at INT4 (``hip_fused`` of the ``qr_`` CSVs)
  where ``CudaCommunicator`` dispatches it, and ``CustomAllreduce``'s fused
  kernel elsewhere. Windows are half-open, as in the policy.

Ladders are chosen by DP over payload breakpoints minimizing the summed
log-regret against the per-shape best measured row of that family. A width
keeps a ladder of its own only when it beats the default ladder at that width
by ``--min-width-gain`` (geomean); elsewhere the width runs the default.

The ladders are written as a ladder file (``--out-json``): the base file
(``--base``, by default the one the policy reads) with the fitted families of
the swept platforms replaced. Try it with ``AITER_FLY_AR_FUSED_LADDERS=<file>``,
then copy it over ``aiter/ops/flydsl/configs/comm/fused_allreduce_ladders.json``.

usage: fit_fused.py SWEEP_DIR [--link xgmi] [--rungs N] [--rung-penalty X]
                    [--min-width-gain G] [--base FILE] [--out-json FILE]
"""

import argparse
import functools
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd

from aiter.ops.flydsl.allreduce_policy import (
    fused_ladders_path,
    fused_mesh_st_ladder,
    fused_qr_row_atoms,
    fused_ring_st_ladder,
    load_fused_ladders,
)
from aiter.ops.flydsl.one_shot_allreduce import OneShotAllReduceRMSNorm
from aiter.ops.flydsl.quick_allreduce import (
    ALGORITHMS,
    FlyQuickAllReduceRMSNorm,
    _resolve_codecs,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bench_io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ladders_json

SQNR_FLOOR = 40.0
# The HIP quick all-reduce's fused kernel at INT4, as the quick test grades it.
HIP_SQNR_FLOOR = 14.0
# The quantized rows' floors.
QR_SQNR_FLOOR = {"mesh": 15.0, "ring": 10.0}
QR_ALGOS = ("mesh", "ring")
SHIPPED_QR_LADDER = {"mesh": fused_mesh_st_ladder, "ring": fused_ring_st_ladder}
_VAR = re.compile(
    r"_a(?P<a>\d+)_g(?P<g>\d+)_\w*?_b(?P<b>\d+)(?P<peer>_peer)?_rms_h\d+(?:p(?P<p>\d+))?"
    r"(?:_k(?P<k>\d+))?/"
)
_QR_KEY = re.compile(
    r"^fused_fly_(?P<alg>mesh|ring)(?P<ll>_ll)?_a(?P<a>\d+)_st(?P<st>\d+)_g(?P<g>\d+)$"
)
ATOMS = (1, 2, 4)
# ``atoms_per_row`` a quantized schedule's row may take, and what it must divide:
# a ring row fits a rank's reduce-scatter chunk (8 // tp atoms), a mesh row
# may span the gathered tile's 8.
QR_ROW_ATOMS = {"mesh": (1, 2, 4, 8), "ring": (1, 2, 4)}
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
    """``atoms_per_row`` a fused *alg* engine that wants *apr* (None: the
    widest block) actually builds at width *h* -- the nearest the width has --
    or None when the width does not fuse at all.
    ``FlyQuickAllReduceRMSNorm._geom_for`` on a bare instance."""
    algo = ALGORITHMS[alg]
    eng = FlyQuickAllReduceRMSNorm.__new__(FlyQuickAllReduceRMSNorm)
    eng.world_size = int(tp)
    eng._algo = algo
    eng.rs_codec, _ag = _resolve_codecs(algo, int(tp), None, None)
    eng.block = None
    eng.atoms_per_row = None if apr is None else int(apr)
    eng._apr_substituted = set()
    # The fit decides the row geometry, not the policy file.
    eng._row_atoms = lambda _h: eng.atoms_per_row
    if not eng.supports_hidden(int(h)):
        return None
    return eng._geom_for(int(h))[1]


def _time(r, key, metric, floor):
    """*key*'s time in row *r*, or inf when it did not run or missed *floor*."""
    v, q = r.get(f"{key} {metric}"), r.get(f"{key} SQNR dB")
    if v is None or pd.isna(v) or (floor is not None and pd.notna(q) and q < floor):
        return math.inf
    return float(v)


def load(sweep_dir, metric):
    """``({tp: [shape dict]}, arch)`` with per-shape measured {knobs: us}, cdr
    and the shipping mesh/ring.

    ``qr`` holds the quantized grid, ``{alg: {(atoms_per_row, st, cap,
    low_latency): us}}``, joined on (TP, K, M) from the ``qr_`` CSVs, as are
    the shipping mesh/ring times; empty for a sweep that has none.
    """
    qr = defaultdict(lambda: defaultdict(dict))
    ship = defaultdict(dict)
    archs = set()
    for path in sorted(glob.glob(os.path.join(sweep_dir, "qr_tp*_w*.csv"))):
        d = bench_io.read_bench_csv(path)
        archs |= set(d["gfx"].dropna())
        keys = [c[: -len(" variant")] for c in d.columns if c.endswith(" variant")]
        for _, r in d.iterrows():
            shape = (int(r["TP"]), int(r["K"]), int(r["M"]))
            for k in ("fused_fly_mesh", "fused_fly_ring"):
                ship[shape][k] = _time(r, k, metric, None)
            # The HIP quick all-reduce's fused kernel, where production
            # would run it on the payload (its size gate; the row gate is
            # the cell being empty).
            hip = _time(r, "fused_qr_int4", metric, HIP_SQNR_FLOOR)
            hip_prod = str(r.get("hip_fused in prod")) == "True"
            ship[shape]["hip"] = hip if hip_prod else math.inf
            for k in keys:
                m = _QR_KEY.match(k)
                if not m:
                    continue
                v = _time(r, k, metric, QR_SQNR_FLOOR[m["alg"]])
                if v == math.inf:
                    continue
                knobs = (int(m["a"]), int(m["st"]), int(m["g"]), bool(m["ll"]))
                qr[shape][m["alg"]][knobs] = v

    data = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(sweep_dir, "tp*_w*.csv"))):
        d = bench_io.read_bench_csv(path)
        archs |= set(d["gfx"].dropna())
        keys = [c[: -len(" variant")] for c in d.columns if c.endswith(" variant")]
        fly = [k for k in keys if k.startswith("fused_fly_1stage")]
        for _, r in d.iterrows():
            meas = {}
            for k in fly:
                kn = parse(r.get(f"{k} variant"))
                v = _time(r, k, metric, SQNR_FLOOR)
                if kn is None or v == math.inf:
                    continue
                meas[kn] = min(v, meas.get(kn, math.inf))
            shape = (int(r["TP"]), int(r["K"]), int(r["M"]))
            data[int(r["TP"])].append(
                {
                    "tp": int(r["TP"]),
                    "h": int(r["K"]),
                    "M": int(r["M"]),
                    "nbytes": int(r["_nbytes"]),
                    "meas": meas,
                    "best": min(meas.values()) if meas else math.inf,
                    "cdr": min(
                        _time(r, "fused_cdr_1stage", metric, None),
                        _time(r, "fused_cdr_2stage", metric, None),
                    ),
                    "hip": ship[shape].get("hip", math.inf),
                    "mesh": ship[shape].get("fused_fly_mesh", math.inf),
                    "shipped": _time(r, "fused_fly_1stage", metric, None),
                    "qr": {alg: dict(qr[shape].get(alg, {})) for alg in QR_ALGOS},
                }
            )
    if len(archs) > 1:
        raise SystemExit(f"{sweep_dir} mixes archs {sorted(archs)}")
    return data, (archs.pop() if archs else None)


def configs(splits=SPLITS):
    for a in ATOMS:
        for cap in CAPS:
            for k in splits:
                yield (a, cap, k)


def cfg_time(shape, cfg):
    a, cap, k = cfg
    return shape["meas"].get(resolve(shape["h"], a, cap, k), math.inf)


def qr_time(shape, alg, apr, cfg):
    """Measured time of a fused *alg* engine pinned to *apr* at *cfg* =
    ``(super_tile, grid_cap, low_latency)``."""
    st, cap, ll = cfg
    got = qr_resolve(shape["h"], shape["tp"], alg, apr)
    meas = shape["qr"][alg]
    caps = [g for a, t, g, l in meas if (a, t, l) == (got, st, ll) and g <= cap]
    return meas[(got, st, max(caps), ll)] if caps else math.inf


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


def fit_qr(shapes, alg, max_rungs, penalty, low_latency=False):
    """``(atoms_per_row, rungs, cost by atoms_per_row)`` for one schedule, or
    None when the sweep has no rows for it.

    One ``atoms_per_row`` serves every width of *shapes*, so it is the outer
    choice: each candidate gets its own ``(super_tile, grid_cap,
    low_latency)`` ladder, and the one with the least total log-regret wins
    (ties to the narrower row, i.e. the wider block). A rung may be the
    low-latency build only with *low_latency*.
    """
    shapes = [s for s in shapes if s["qr"][alg]]
    if not shapes:
        return None
    tp = shapes[0]["tp"]
    cfgs = sorted(
        {
            (st, g, ll)
            for s in shapes
            for _a, st, g, ll in s["qr"][alg]
            if low_latency or not ll
        }
    )
    fits = {}
    span = 8 if alg == "mesh" else 8 // tp
    for apr in QR_ROW_ATOMS[alg]:
        if span % apr:
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
    """Byte threshold T minimizing sum log(time): below(s) if nbytes < T, else
    above(s). T is a swept size, or None (infinity) for "below everywhere"."""
    sizes = sorted({s["nbytes"] for s in shapes})
    best = (math.inf, 0)
    for T in [*sizes, None]:
        tot = 0.0
        for s in shapes:
            v = below(s) if T is None or s["nbytes"] < T else above(s)
            tot += math.log(v) if v < math.inf else 50.0
        if tot < best[0] - 1e-9:
            best = (tot, T)
    return best[1]


def _below(nbytes, T):
    return T is None or nbytes < T


def _lower(a, b):
    """The lower of two thresholds, None being no limit."""
    return b if a is None else a if b is None else min(a, b)


def use_1stage(tp, nbytes):
    """``use_1stage`` of ``CudaCommunicator.fused_allreduce_rmsnorm`` at its
    defaults: where it is set, production skips the HIP fused kernel."""
    return nbytes <= 128 * 7168 * 2 // tp


def fallback(s):
    """What production runs where FlyDSL declines a payload with quantization
    on: the HIP quick all-reduce's fused kernel at INT4 where
    ``CudaCommunicator`` dispatches it, else ``CustomAllreduce``'s fused
    kernel."""
    if not use_1stage(s["tp"], s["nbytes"]) and s["hip"] < math.inf:
        return s["hip"]
    return s["cdr"]


def _kb(mb):
    """*mb* bytes as the policy tables write them: ``384 * KiB``, ``4 * MiB``."""
    if mb is None:
        return "None"
    if not mb:
        return "0"
    for unit, shift in (("MiB", 20), ("KiB", 10)):
        if mb % (1 << shift) == 0:
            return f"{mb >> shift} * {unit}"
    return str(mb)


def policy_literal(
    arch, link, tp, fast_max, exact_max, mesh_max, have, quant_max=None
) -> str:
    """The ``FUSED_FAMILY_POLICIES`` row, checked by building it.

    The one-shot window is the exact ceiling *exact_max*; the quantized ceiling
    *fast_max* only places the start of the mesh or the ring, which take their
    payloads first in fast mode. *quant_max* ends the quantized windows: above
    it production's fallback (``fallback``) is faster.
    """
    from aiter.ops.flydsl.allreduce_policy import FamilyPolicy, Window

    one = Window(0, exact_max)
    mesh = ring = None
    if fast_max is not None and not (quant_max is not None and quant_max <= fast_max):
        if have == ["ring"]:
            ring = Window(fast_max, quant_max)
        else:
            mesh_end = _lower(mesh_max if "ring" in have else None, quant_max)
            if mesh_end is None or mesh_end > fast_max:
                mesh = Window(fast_max, mesh_end)
            if (
                "ring" in have
                and mesh_max is not None
                and (quant_max is None or quant_max > mesh_max)
            ):
                ring = Window(mesh_max, quant_max)
    try:
        FamilyPolicy(
            arch=arch,
            link=link,
            world_size=tp,
            oneshot=one,
            mesh=mesh,
            ring=ring,
        )
    except ValueError as exc:
        return f"# the fitted windows do not form a policy: {exc}"

    def w(win):
        return "None" if win is None else f"Window({_kb(win.min)}, {_kb(win.max)})"

    return (
        f"FUSED_FAMILY_POLICIES row:\n"
        f"    FamilyPolicy(\n"
        f"        arch={arch!r},\n"
        f"        link={link!r},\n"
        f"        world_size={tp},\n"
        f"        oneshot={w(one)},\n"
        f"        mesh={w(mesh)},\n"
        f"        ring={w(ring)},\n"
        f"    ),"
    )


def geomean(vals):
    vals = [v for v in vals if 0 < v < math.inf]
    return math.exp(sum(math.log(v) for v in vals) / len(vals)) if vals else math.nan


@dataclass
class WidthFits:
    """One family's fits: ``default``, for every width, and ``widths``, the
    widths whose own fit beats the default there by the margin. ``gains`` is
    the default's over the width's own fit, geomean, per swept width;
    ``time_of(shape, fit)`` a fit's time at a shape."""

    default: object
    widths: dict
    gains: dict
    time_of: object

    def time(self, shape):
        """The time at *shape* of the fit its width runs."""
        return self.time_of(shape, self.widths.get(shape["h"], self.default))


def fit_by_width(shapes, fit_default, fit_width, time_of, min_gain):
    """Fit *shapes* with one default ladder (``fit_default``) and each width
    with one of its own (``fit_width``). A width keeps its own only when its
    geomean speedup over the default at that width is at least ``1 +
    min_gain``, or when the default cannot run there. None when the default
    cannot be fitted."""
    default = fit_default(shapes)
    if default is None:
        return None
    widths, gains = {}, {}
    for h in sorted({s["h"] for s in shapes}):
        mine = [s for s in shapes if s["h"] == h]
        own = fit_width(mine)
        if own is None:
            continue
        pairs = [(time_of(s, default), time_of(s, own)) for s in mine]
        if any(d == math.inf and o < math.inf for d, o in pairs):
            gains[h] = math.inf
        else:
            gains[h] = geomean(d / o for d, o in pairs)
        if gains[h] >= 1 + min_gain:
            widths[h] = own
    return WidthFits(default, widths, gains, time_of)


def fit_oneshot(shapes, args):
    """The one-shot fits: a default ladder that never splits, and per-width
    ladders that may."""

    def fit(cfgs):
        return lambda ss: fit_ladder(
            ss, args.rungs, args.rung_penalty, cfgs, cfg_time, lambda s: s["best"]
        )

    return fit_by_width(
        shapes,
        fit(list(configs(splits=(1,)))),
        fit(list(configs())),
        ladder_time,
        args.min_width_gain,
    )


def fit_qr_widths(shapes, alg, args):
    """The *alg* fits, ``atoms_per_row`` and ladder, default and per width.
    Only a width's own ladder may take the low-latency build, and only on
    xGMI, the one link the ladder file accepts it on."""

    def fit(low_latency):
        return lambda ss: fit_qr(
            ss, alg, args.rungs, args.rung_penalty, low_latency=low_latency
        )

    return fit_by_width(
        shapes,
        fit(False),
        fit(args.link == "xgmi"),
        lambda s, f: qr_ladder_time(s, alg, f[0], f[1]),
        args.min_width_gain,
    )


def oneshot_entry(rungs) -> dict:
    """A fitted one-shot ladder as a ladder-file entry."""
    return ladders_json.entry(
        ladders_json.oneshot_rung(mb, a, cap, k) for mb, (a, cap, k), _c in rungs
    )


def qr_entry(fit) -> dict:
    """A fitted mesh or ring ``(atoms_per_row, rungs, _)`` as a ladder-file
    entry."""
    apr, rungs, _by_apr = fit
    return ladders_json.entry(
        (
            ladders_json.two_shot_rung(mb, st, cap, low_latency=ll)
            for mb, (st, cap, ll), _c in rungs
        ),
        atoms_per_row=apr,
    )


def print_fits(name, fits, entry_of, rungs_of, args):
    """Print *fits* as ladder-file entries, one rung per line."""

    def show(label, fit):
        entry = entry_of(fit)
        print(f"{name} {label}:")
        if "atoms_per_row" in entry:
            by_apr = ", ".join(f"{a}: {c:.3f}" for a, c in sorted(fit[2].items()))
            print(
                f"    atoms_per_row={entry['atoms_per_row']}  "
                f"(log-regret by atoms_per_row: {by_apr})"
            )
        for rung, (*_r, cost) in zip(entry["rungs"], rungs_of(fit)):
            print(f"    {json.dumps(rung)},  # seg log-regret {cost:.3f}")

    show("default -- every width", fits.default)
    for h, gain in sorted(fits.gains.items()):
        if h in fits.widths:
            show(f"hidden={h} -- default/own {gain:.3f} at this width", fits.widths[h])
        else:
            print(
                f"{name} hidden={h} -- runs the default: default/own {gain:.3f} "
                f"< {1 + args.min_width_gain:.3f}"
            )


def shipped_qr_time(shape, alg, args):
    """What the shipped *alg* ladder and ``atoms_per_row`` at this width take,
    timed off the same pinned rows."""
    kw = {
        "world_size": shape["tp"],
        "link": args.link,
        "arch": args.arch,
        "hidden": shape["h"],
    }
    apr = fused_qr_row_atoms(algorithm=alg, **kw)
    ship = tuple(
        (r.min_bytes, (r.super_tile, r.grid_cap, r.low_latency), 0.0)
        for r in SHIPPED_QR_LADDER[alg](**kw)
    )
    return qr_ladder_time(shape, alg, apr, ship)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "sweep_dir", help="directory holding tp{tp}_w{h}.csv from the fused sweep"
    )
    ap.add_argument(
        "--link", default="xgmi", help="fabric the sweep ran on (labels the output)"
    )
    ap.add_argument("--rungs", type=int, default=3)
    ap.add_argument("--metric", default="us", choices=["us"])
    ap.add_argument("--rung-penalty", type=float, default=0.25)
    ap.add_argument(
        "--min-width-gain",
        type=float,
        default=0.02,
        help="geomean speedup over the default ladder, minus 1, that a width "
        "needs to keep a ladder of its own (default 0.02)",
    )
    ap.add_argument(
        "--base",
        default=None,
        help="ladder file the fit is merged into (default: the one the policy "
        "reads, AITER_FLY_AR_FUSED_LADDERS or the shipped file)",
    )
    ap.add_argument(
        "--out-json", default=None, help="write the merged ladder file here"
    )
    args = ap.parse_args()
    data, arch = load(args.sweep_dir, args.metric)
    if not data:
        raise SystemExit(f"no tp*_w*.csv in {args.sweep_dir}")
    if arch is None and args.out_json:
        raise SystemExit(f"{args.sweep_dir} names no arch (gfx column)")
    args.arch = arch
    base = args.base or fused_ladders_path()
    doc = ladders_json.read(base)
    for tp in sorted(data):
        swept = sorted({s["h"] for s in data[tp]})
        for family, (default, widths) in fit_tp(tp, data[tp], args).items():
            kept = ladders_json.set_family(
                doc, arch, args.link, tp, family, default, widths, swept
            )
            if kept:
                print(
                    f"{family}: TP{tp} keeps the base file's ladders for the "
                    f"widths it did not sweep: {list(kept)}"
                )
    if args.out_json:
        ladders_json.write(doc, args.out_json)
        load_fused_ladders(args.out_json)  # the checks the policy makes
        print(
            f"\nladder file -> {args.out_json} (merged into {base}). Try it with "
            f"AITER_FLY_AR_FUSED_LADDERS={args.out_json}, then copy it over "
            "aiter/ops/flydsl/configs/comm/fused_allreduce_ladders.json."
        )


def fit_tp(tp, shapes, args):
    """Fit and print every fused table for one world size. Returns the fitted
    ladder-file entries, ``{family: (default entry, {width: entry})}``; a
    family that is not fitted, or that no swept shape dispatches to, is left
    out."""
    print(f"\n## TP{tp}  ({args.arch}/{args.link}, metric={args.metric!r})")

    # Fused one-shot: fit on everything first, then refit inside the exact
    # window.
    one = fit_oneshot(shapes, args)
    exact_max = best_threshold(shapes, one.time, lambda s: s["cdr"])
    win = [s for s in shapes if _below(s["nbytes"], exact_max)]
    if win:
        one = fit_oneshot(win, args)
        exact_max = best_threshold(shapes, one.time, lambda s: s["cdr"])

    def below_quant(s):
        # With quantization on, what runs below the quantized windows: the
        # one-shot inside its (exact) window, production's fallback above it.
        return one.time(s) if _below(s["nbytes"], exact_max) else fallback(s)

    # Quantized schedules: fit each on every shape, take the one-shot
    # boundary against the faster of the two, then refit each above it --
    # a rung below the boundary is an engine nothing dispatches to. Once the
    # boundaries are set, each is refit once more on its own window below.
    qfit = {alg: fit_qr_widths(shapes, alg, args) for alg in QR_ALGOS}
    have = [alg for alg in QR_ALGOS if qfit[alg] is not None]

    def qtime(s, alg):
        return qfit[alg].time(s)

    if have:
        fast_max = best_threshold(
            shapes, below_quant, lambda s: min(qtime(s, alg) for alg in have)
        )
        above = [s for s in shapes if not _below(s["nbytes"], fast_max)]
        for alg in have:
            refit = fit_qr_widths(above, alg, args)
            if refit is not None:
                qfit[alg] = refit
        if have == ["mesh"]:
            mesh_max = None
        elif have == ["ring"]:
            mesh_max = fast_max
        else:
            mesh_max = best_threshold(
                above or shapes, lambda s: qtime(s, "mesh"), lambda s: qtime(s, "ring")
            )

        def quant(s):
            if _below(s["nbytes"], mesh_max):
                return qtime(s, "mesh")
            return qtime(s, "ring")

        fast_max = best_threshold(shapes, below_quant, quant)
        if fast_max is None:
            mesh_max = None  # the one-shot wins everywhere: no quantized window
        elif mesh_max is not None and mesh_max < fast_max:
            mesh_max = fast_max  # the mesh window is empty
        # The top of the quantized windows: above it production's fallback,
        # the HIP fused kernel where it dispatches it, is faster.
        top = [s for s in shapes if not _below(s["nbytes"], fast_max)]
        quant_max = best_threshold(top, quant, fallback) if top else None
    else:
        print(
            "(no qr_tp*_w*.csv rows: mesh/ring not fitted; oneshot_max is against fused_fly_mesh)"
        )
        mesh_max = quant_max = None
        fast_max = best_threshold(shapes, one.time, lambda s: s["mesh"])
    windows = {
        "mesh": [
            s
            for s in shapes
            if not _below(s["nbytes"], fast_max)
            and _below(s["nbytes"], _lower(mesh_max, quant_max))
        ],
        "ring": [
            s
            for s in shapes
            if not _below(s["nbytes"], mesh_max) and _below(s["nbytes"], quant_max)
        ],
    }
    if have:
        # A family's ladder only ever serves its own window, so fit it there:
        # the boundaries above were taken with ladders fitted across both
        # windows, and the shapes the other family serves must not pick this
        # one's rungs. A family whose window holds no swept shape keeps no fit.
        for alg in have:
            refit = fit_qr_widths(windows[alg], alg, args) if windows[alg] else None
            if refit is not None:
                qfit[alg] = refit

    fitted = {
        "oneshot": (
            oneshot_entry(one.default),
            {h: oneshot_entry(r) for h, r in one.widths.items()},
        )
    }
    print_fits("oneshot", one, oneshot_entry, lambda r: r, args)
    for alg in have:
        if not windows[alg]:
            print(
                f"{alg}: not dispatched -- no swept shape in its window; the "
                "base file's ladders stay"
            )
            continue
        print(f"# {alg}: fitted on the {len(windows[alg])} shapes in its window")
        print_fits(alg, qfit[alg], qr_entry, lambda f: f[1], args)
        fitted[alg] = (
            qr_entry(qfit[alg].default),
            {h: qr_entry(f) for h, f in qfit[alg].widths.items()},
        )
        ratio = geomean(
            shipped_qr_time(s, alg, args) / qtime(s, alg) for s in windows[alg]
        )
        print(f"    shipped/fitted in its window: {ratio:.3f}")
    print(f"exact one-shot  < {_kb(exact_max)}  (vs cdr)")
    print(f"quant one-shot  < {_kb(fast_max)}  (vs the quantized schedules)")
    if have:
        print(f"mesh            < {_kb(mesh_max)}  (vs ring)")
        print(
            f"quantized       < {_kb(quant_max)}  (vs the fallback: the HIP fused "
            "kernel where production dispatches it, else cdr)"
        )
    print(
        policy_literal(
            args.arch, args.link, tp, fast_max, exact_max, mesh_max, have, quant_max
        )
    )
    print(
        f"{'h':>5} {'M':>5} {'KiB':>6} {'best':>7} {'ladder':>7} {'regret':>6} "
        f"{'default':>7} {'shipped':>8} {'cdr':>7} {'hip':>7} {'mesh':>7} {'ring':>7} "
        f"{'lad/cdr':>7}  win"
    )
    for s in sorted(shapes, key=lambda s: (s["h"], s["M"])):
        lt = one.time(s)
        dt = ladder_time(s, one.default)
        mt = qtime(s, "mesh") if "mesh" in have else s["mesh"]
        rt = qtime(s, "ring") if "ring" in have else math.nan
        mark = "*" if _below(s["nbytes"], exact_max) else ""
        print(
            f"{s['h']:>5} {s['M']:>5} {s['nbytes'] >> 10:>6} {s['best']:>7.2f} {lt:>7.2f} "
            f"{lt / s['best']:>6.3f} {dt:>7.2f} {s['shipped']:>8.2f} {s['cdr']:>7.2f} "
            f"{s['hip']:>7.2f} {mt:>7.2f} {rt:>7.2f} {s['cdr'] / lt:>7.3f}  {mark}"
        )
    inw = [s for s in shapes if _below(s["nbytes"], exact_max)]
    if inw:
        print(
            f"in-window geomean: ladder/best={geomean(one.time(s) / s['best'] for s in inw):.3f}  "
            f"default/ladder={geomean(ladder_time(s, one.default) / one.time(s) for s in inw):.3f}  "
            f"cdr/ladder={geomean(s['cdr'] / one.time(s) for s in inw):.3f}  "
            f"shipped/ladder={geomean(s['shipped'] / one.time(s) for s in inw):.3f}"
        )
    return fitted


if __name__ == "__main__":
    main()
