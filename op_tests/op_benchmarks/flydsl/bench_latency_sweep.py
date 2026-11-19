# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Time per call vs message size: FlyDSL quick all-reduce and attention A2A, vs RCCL.

    python3 op_tests/op_benchmarks/flydsl/bench_latency_sweep.py
    python3 op_tests/op_benchmarks/flydsl/bench_latency_sweep.py \\
        --collectives allreduce --tp 2 4 --tokens 1 16 256 4096
    python3 op_tests/op_benchmarks/flydsl/bench_latency_sweep.py --plot-only

For each TP size, one spawn runs the ``_worker`` of the matching bandwidth
script, so the setup, the timing method and the RCCL baseline are exactly
those of the bandwidth reports:

* all-reduce (``bench_bw_quick_allreduce``): a CUDA graph of back-to-back
  calls, rotated rounds, median, slowest rank. The message is one rank's
  ``(tokens, hidden)`` bf16 input; ``--tokens`` sweeps it.
* all-to-all (``bench_bw_attention_a2a``): the eager median of the fused
  Q/K/V trio, rotated input sets, slowest rank. The message is one rank's
  send payload of the trio (three ``(B, S/TP, H, D)`` bf16 tensors);
  ``--seq-lens`` sweeps the global sequence length S. The RCCL line is the
  median over recipes of the same (three ``all_to_all_single``) baseline.

Each collective writes ``<out-dir>/<collective>_sweep[_<tag>].csv`` after
every TP size (rows of the TP sizes not re-run are kept, so TP sizes can be
run separately), then one figure per metric, ``<collective>_sweep[_<tag>]_<metric>.png``,
with one subplot per TP size against the per-rank message size: ``time`` per
call (log-log), ``bw`` (algorithmic bandwidth) and ``speedup`` over mesh
(all-reduce) or RCCL (all-to-all). ``--plot-only`` redraws the figures from the
CSVs without touching the GPUs.

The TP4 all-to-all always runs with NCCL_P2P_READ_ENABLE=1 (``TP_ENV``).
TP=N uses the first N visible GPUs; set ``HIP_VISIBLE_DEVICES`` to pick them.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[2]
_COLLECTIVE_BW = _HERE.parent / "collective_bw"
for _path in (
    str(_HERE),
    str(_COLLECTIVE_BW),
    str(_REPO),
    str(_REPO / "op_tests" / "multigpu_tests"),
):
    if _path not in sys.path:
        sys.path.insert(0, _path)

COLLECTIVES = ("allreduce", "a2a")
# Environment forced on the spawn of a (collective, TP size). At TP4, RCCL's
# all-to-all reaches only ~5 GB/s per rank by default (0.09 of the fabric peak,
# and flat in size) and ~35 GB/s with P2P reads, so the default would make
# every comparison with RCCL meaningless. The all-reduce is not affected and
# runs as is. The fused ops do not read the variable.
TP_ENV = {("a2a", 4): {"NCCL_P2P_READ_ENABLE": "1"}}
TP_SIZES = (2, 4, 8)
# 14 KiB .. 224 MiB per rank at hidden 7168.
DEFAULT_TOKENS = tuple(2**i for i in range(15))
DEFAULT_HIDDEN = 7168
# Wan2.2's heads and head dim; S is the global sequence length.
DEFAULT_SEQ_LENS = tuple(2**i for i in range(8, 16))
DEFAULT_BATCH = 1
DEFAULT_HEADS = 40
DEFAULT_HEAD_DIM = 128
DEFAULT_OUT_DIR = _REPO / "op_tests" / "dump_data" / "bench" / "latency"

CSV_COLUMNS = (
    "collective",
    "tp",
    "impl",
    "recipe",
    "size",
    "bytes",
    "us",
    "spread_us",
    "timing",
    "variant",
    "case",
)

RCCL = "rccl"
# Plot order and colour slot of every known implementation, one group per
# collective: a figure's colours start at slot 1. Colour follows the
# implementation, so a run without one of them does not repaint the rest.
# Unknown names take the slots after every group's, in sorted order.
_SLOT_GROUPS = (
    ("fly_mesh", "fly_ring", "fly_hier"),
    (
        "fused bf16",
        "fused mxfp4_qk_mxfp4_v",
        "fused mxfp4_qk_mxfp6p_v",
        "fused e4m3_qkv",
        "fused int8_qk_e4m3_v",
    ),
)
_KNOWN_IMPLS = tuple(i for group in _SLOT_GROUPS for i in group)
_LABELS = {
    "fly_mesh": "FlyDSL mesh",
    "fly_ring": "FlyDSL ring",
    "fly_hier": "FlyDSL hier",
    RCCL: "RCCL",
}
# The dataviz reference categorical palette (light), in its fixed order.
_PALETTE = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
_MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
_RCCL_COLOR = "#52514e"
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK_2 = "#52514e"
_GRID = "#e4e3df"


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def _gpu_count() -> int:
    import flydsl_allreduce_bench as fab

    return fab.gpu_count()


def _init_method() -> str:
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    return get_distributed_init_method(get_ip(), get_open_port())


def _run_allreduce(tp: int, args) -> list[dict]:
    import bench_bw_quick_allreduce as bwq
    import flydsl_allreduce_bench as fab
    from measure import spawn

    cases = [(m, args.hidden) for m in args.tokens]
    ranks = spawn(
        bwq._worker,
        tp,
        timeout=args.timeout,
        init_method=_init_method(),
        algorithms=tuple(args.algorithms),
        cases=cases,
        reps=args.reps or fab.DEFAULT_REPS,
        rounds=args.rounds or fab.DEFAULT_ROUNDS,
        rccl_backend=args.rccl_backend,
        hier_handshake=args.hier_handshake,
    )
    tokens_of = {f"{m}x{h}": m for m, h in cases}
    return [
        {
            "collective": "allreduce",
            "tp": tp,
            "impl": r.impl,
            "recipe": "",
            "size": tokens_of[r.case],
            "bytes": r.traffic.logical_bytes,
            "us": r.us,
            "spread_us": r.spread_us,
            "timing": r.timing,
            "variant": r.variant or "",
            "case": r.case,
        }
        for r in ranks[0]
    ]


def _run_a2a(tp: int, args) -> list[dict]:
    import bench_attention_a2a as a2a
    import bench_bw_attention_a2a as bwa
    from measure import spawn

    if args.heads % tp:
        raise ValueError(f"heads={args.heads} must be divisible by TP={tp}")
    names = {r.name for r in a2a._RECIPES}
    selected = names if "all" in args.recipes else set(args.recipes)
    if selected - names:
        raise ValueError(
            f"unknown recipe(s) {sorted(selected - names)}; one of {sorted(names)}"
        )
    recipes = [r for r in a2a._RECIPES if r.name in selected]
    shapes = {
        f"S{s}": (args.batch, s, args.heads, args.head_dim) for s in args.seq_lens
    }
    link, inbox = a2a._resolve_inbox(args.inbox_memory, tp)
    print(f"[a2a TP{tp}] link {link}, inbox {inbox}", flush=True)
    ranks = spawn(
        bwa._worker,
        tp,
        timeout=args.timeout,
        init_method=_init_method(),
        shapes=shapes,
        recipes=recipes,
        warmup=args.warmup,
        iterations=args.iterations,
        inbox=inbox,
    )
    rows = []
    for r in ranks[0]:
        shape_name, recipe = r.case.split(" ", 1)
        rows.append(
            {
                "collective": "a2a",
                "tp": tp,
                "impl": RCCL if r.impl == RCCL else f"{r.impl} {recipe}",
                "recipe": recipe,
                "size": int(shape_name[1:]),
                "bytes": r.traffic.logical_bytes,
                "us": r.us,
                "spread_us": r.spread_us,
                "timing": r.timing,
                "variant": r.variant or "",
                "case": r.case,
            }
        )
    return rows


@contextlib.contextmanager
def _tp_env(collective: str, tp: int):
    """Set ``TP_ENV[collective, tp]`` for the spawn: the workers inherit the parent's
    environment. The previous values are restored afterwards."""
    env = TP_ENV.get((collective, tp), {})
    old = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if old[k] not in (None, v):
            print(f"overriding {k}={old[k]} with {v} for TP{tp}", flush=True)
        os.environ[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


_RUNNERS = {"allreduce": _run_allreduce, "a2a": _run_a2a}


def _can_run(collective: str, tp: int, args) -> str | None:
    """Why *collective* cannot run at *tp*, or None."""
    n_gpu = _gpu_count()
    if n_gpu < tp:
        return f"TP{tp} needs {tp} GPUs, {n_gpu} visible"
    if collective == "a2a":
        from aiter.jit.utils.chip_info import get_gfx_runtime

        gfx = get_gfx_runtime()
        if gfx != "gfx950":
            return f"attention A2A needs gfx950, found {gfx}"
        if args.heads % tp:
            return f"heads={args.heads} is not divisible by TP={tp}"
    return None


# --------------------------------------------------------------------------- #
# CSV
# --------------------------------------------------------------------------- #


def _paths(
    out_dir: Path, collective: str, tag: str | None
) -> tuple[Path, dict[str, Path]]:
    """The CSV, and one PNG per metric of ``METRICS``."""
    stem = f"{collective}_sweep" + (f"_{tag}" if tag else "")
    return out_dir / f"{stem}.csv", {m: out_dir / f"{stem}_{m}.png" for m in METRICS}


def _save(csv_path: Path, tp: int, rows: list[dict]) -> None:
    """Replace *tp*'s rows of *csv_path* with *rows*, keeping the other TPs."""
    import pandas as pd

    new = pd.DataFrame(rows, columns=list(CSV_COLUMNS))
    if csv_path.exists():
        old = pd.read_csv(csv_path, keep_default_na=False)
        new = pd.concat([old[old["tp"] != tp], new], ignore_index=True)
    new = new.sort_values(["tp", "impl", "bytes"], kind="stable")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new.to_csv(csv_path, index=False)
    print(f"wrote {csv_path}", flush=True)


def _load(csv_path: Path):
    import pandas as pd

    df = pd.read_csv(csv_path, keep_default_na=False)
    missing = set(CSV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing column(s) {sorted(missing)}")
    return df


def _series(df):
    """``{impl: DataFrame(size, bytes, us)}``, sorted by bytes. RCCL is timed
    once per a2a recipe on the same bf16 payload, so its points are the
    median over recipes."""
    out = {}
    for impl, g in df.groupby("impl", sort=False):
        if impl == RCCL:
            g = g.groupby(["size", "bytes"], as_index=False)["us"].median()
        out[impl] = g[["size", "bytes", "us"]].sort_values("bytes")
    return out


def _table(df, collective: str) -> str:
    """Time per call (us) per size and implementation, for every TP."""
    lines = []
    size_name = "tokens" if collective == "allreduce" else "S"
    for tp, g in df.groupby("tp", sort=True):
        series = _series(g)
        wide = None
        for impl in _ordered(series):
            s = series[impl].rename(columns={"us": _label(impl)})
            wide = (
                s if wide is None else wide.merge(s, on=["size", "bytes"], how="outer")
            )
        wide = wide.sort_values("bytes").rename(columns={"size": size_name})
        wide.insert(1, "KiB", wide.pop("bytes") / 1024)
        # Sizes in full; ".4g" would print 16384 as 1.638e+04.
        fmt = (".10g", ".10g") + (".4g",) * (wide.shape[1] - 2)
        lines += [
            f"### {collective} TP{tp}: us per call",
            "",
            wide.to_markdown(index=False, floatfmt=fmt),
            "",
        ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Plot
# --------------------------------------------------------------------------- #


def _ordered(impls) -> list[str]:
    known = [i for i in _KNOWN_IMPLS if i in impls]
    rest = sorted(i for i in impls if i not in _KNOWN_IMPLS and i != RCCL)
    return known + rest + ([RCCL] if RCCL in impls else [])


def _styles(impls) -> dict[str, dict]:
    """A fixed colour and marker per implementation, RCCL a dashed grey baseline."""
    unknown = sorted(i for i in impls if i not in _KNOWN_IMPLS and i != RCCL)
    first_unknown = max(len(group) for group in _SLOT_GROUPS)

    def slot(impl):
        for group in _SLOT_GROUPS:
            if impl in group:
                return group.index(impl)
        return first_unknown + unknown.index(impl)

    styles = {}
    for impl in impls:
        if impl == RCCL:
            styles[impl] = {
                "color": _RCCL_COLOR,
                "marker": "x",
                "linestyle": "--",
                "markersize": 8,
            }
            continue
        k = slot(impl)
        styles[impl] = {
            "color": _PALETTE[k % len(_PALETTE)],
            "marker": _MARKERS[k % len(_MARKERS)],
            "linestyle": "-",
            "markersize": 6,
        }
    return styles


def _label(impl: str) -> str:
    return _LABELS.get(impl, impl)


def _fmt_bytes(v: float, _pos=None) -> str:
    for unit, scale in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if v >= scale:
            return f"{v / scale:g} {unit}"
    return f"{v:g} B"


def _fmt_us(v: float, _pos=None) -> str:
    if v >= 1000:
        return f"{v / 1000:g} ms"
    return f"{v:g} µs"


_TITLES = {
    "allreduce": "FlyDSL quick all-reduce vs RCCL",
    "a2a": "Fused attention all-to-all (Q/K/V trio) vs RCCL",
}
_XLABELS = {
    "allreduce": "per-rank message (bf16 input)",
    "a2a": "per-rank message (Q+K+V send payload, bf16)",
}


# Implementations drawn and tabulated, per collective; the CSV keeps all of them.
# A collective not listed here shows everything.
_SHOWN_IMPLS = {"a2a": ("fused bf16", "fused e4m3_qkv", RCCL)}
# The implementation the speedup row is relative to, per collective.
_SPEEDUP_BASE = {"allreduce": "fly_mesh", "a2a": RCCL}


def _style_axis(ax, *, log_y: bool, y_formatter=None) -> None:
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

    ax.set_facecolor(_SURFACE)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(LogLocator(base=4))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt_bytes))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if log_y:
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_minor_formatter(NullFormatter())
    if y_formatter is not None:
        ax.yaxis.set_major_formatter(FuncFormatter(y_formatter))
    ax.grid(True, which="major", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_INK_2)
    ax.tick_params(colors=_INK_2, labelsize=9)


def _speedups(series: dict, base: str):
    """``{impl: DataFrame(bytes, speedup)}``: ``base`` us over the impl's us,
    where both were timed. Above 1.0 is faster than ``base``."""
    if base not in series:
        return {}
    ref = series[base].set_index("bytes")["us"]
    out = {}
    for impl, s in series.items():
        if impl == base:
            continue
        both = (
            s.set_index("bytes")["us"]
            .to_frame("us")
            .join(ref.rename("ref"), how="inner")
        )
        out[impl] = both.assign(speedup=both["ref"] / both["us"]).reset_index()
    return out


METRICS = ("time", "bw", "speedup")
_METRIC_TITLES = {
    "time": "time per call",
    "bw": "algorithmic bandwidth",
    "speedup": "speedup",
}


def _ylabel(metric: str, base: str) -> str:
    return {
        "time": "time per call (slowest rank)",
        "bw": "algorithmic bandwidth, GB/s (payload / time)",
        "speedup": f"speedup over {_label(base)}",
    }[metric]


def plot(
    df, collective: str, metric: str, png: Path, tps=TP_SIZES, subtitle: str = ""
) -> None:
    """One figure of *metric*, one panel per TP.

    ``time`` is time per call on log-log axes. ``bw`` is the algorithmic
    bandwidth (payload / time) and ``speedup`` the time of the collective's
    reference over each implementation's, both linear. Time on log-log axes
    turns a constant factor between two bandwidth-bound curves into a thin
    gap, which the other two show directly.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    impls = _ordered(set(df["impl"]))
    styles = _styles(impls)
    base = _SPEEDUP_BASE[collective]
    fig, axes = plt.subplots(
        1,
        len(tps),
        figsize=(5.2 * len(tps), 4.6),
        sharey=True,
        squeeze=False,
        facecolor=_SURFACE,
    )
    axes = axes[0]
    handles = {}
    for ax, tp in zip(axes, tps):
        g = df[df["tp"] == tp]
        series = _series(g)
        ax.set_title(f"TP{tp}", color=_INK, fontsize=12, loc="left")
        if metric == "time":
            curves = {i: (s["bytes"], s["us"]) for i, s in series.items()}
        elif metric == "bw":
            # GB/s = bytes / us / 1e3.
            curves = {
                i: (s["bytes"], s["bytes"] / s["us"] / 1e3) for i, s in series.items()
            }
        else:
            curves = {
                i: (s["bytes"], s["speedup"])
                for i, s in _speedups(series, base).items()
            }
        for impl in _ordered(curves):
            x, y = curves[impl]
            line_kw = {
                "linewidth": 2,
                "markeredgecolor": _SURFACE,
                "markeredgewidth": 1,
            }
            line_kw.update(styles[impl])
            (line,) = ax.plot(x, y, label=_label(impl), **line_kw)
            handles.setdefault(impl, line)
        if metric == "speedup":
            ax.axhline(1.0, color=_INK_2, linewidth=1.2, linestyle=":")
        if g.empty:
            ax.text(
                0.5,
                0.5,
                "no data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=_INK_2,
            )
        _style_axis(
            ax,
            log_y=metric == "time",
            y_formatter=_fmt_us if metric == "time" else None,
        )
        if metric == "bw":
            ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", labelrotation=30)
        ax.set_xlabel(_XLABELS[collective], color=_INK_2)
    axes[0].set_ylabel(_ylabel(metric, base), color=_INK_2)
    order = _ordered(handles)
    fig.legend(
        [handles[i] for i in order],
        [_label(i) for i in order],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=min(len(order), 6),
        frameon=False,
        labelcolor=_INK,
    )
    title = f"{_TITLES[collective]}: {_METRIC_TITLES[metric]}"
    fig.suptitle(title, x=0.01, y=0.99, ha="left", color=_INK, fontsize=14)
    if subtitle:
        fig.text(0.01, 0.925, subtitle, ha="left", va="top", color=_INK_2, fontsize=9)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.76, bottom=0.2, wspace=0.08)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=150, facecolor=_SURFACE)
    plt.close(fig)
    print(f"wrote {png}", flush=True)


def _subtitle(df, collective: str) -> str:
    timing = ", ".join(sorted(set(df["timing"]))) or "?"
    if collective == "allreduce":
        return f"timing: {timing} (CUDA graph replay, median of rotated rounds)"
    return f"timing: {timing} (median of iterations, rotated input sets)"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--collectives", nargs="+", choices=COLLECTIVES, default=list(COLLECTIVES)
    )
    p.add_argument(
        "--tp", type=int, nargs="+", choices=TP_SIZES, default=list(TP_SIZES)
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--tag", default=None, help="suffix of the output file names, e.g. p2p_read"
    )
    p.add_argument(
        "--plot-only",
        action="store_true",
        help="redraw the figures from the CSVs; no GPU work",
    )
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds per TP spawn")

    ar = p.add_argument_group("all-reduce")
    ar.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOKENS),
        help="token counts; each case is (tokens, --hidden) bf16",
    )
    ar.add_argument("--hidden", type=int, default=DEFAULT_HIDDEN)
    ar.add_argument(
        "--algorithms",
        nargs="+",
        choices=("mesh", "ring", "hier"),
        default=["mesh", "ring"],
        help="hier needs GPUs that split into two socket groups (TP8)",
    )
    ar.add_argument(
        "--hier-handshake",
        choices=("flag", "lamport"),
        default=None,
        help="handshake of the hier algorithm (default: the engine's, flag)",
    )
    ar.add_argument(
        "--reps", type=int, default=None, help="default: flydsl_allreduce_bench's"
    )
    ar.add_argument(
        "--rounds", type=int, default=None, help="default: flydsl_allreduce_bench's"
    )
    ar.add_argument("--rccl-backend", choices=("torch", "pynccl"), default="torch")

    a2a = p.add_argument_group("all-to-all")
    a2a.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEQ_LENS),
        help="global sequence lengths S of (B, S, H, D)",
    )
    a2a.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    a2a.add_argument("--heads", type=int, default=DEFAULT_HEADS)
    a2a.add_argument("--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    a2a.add_argument(
        "--recipes",
        nargs="+",
        default=["all"],
        help="'all' or names from bench_attention_a2a._RECIPES",
    )
    a2a.add_argument("--warmup", type=int, default=3)
    a2a.add_argument("--iterations", type=int, default=15)
    a2a.add_argument(
        "--inbox-memory",
        default="auto",
        help="memory of the peer-written buffers (auto: op's policy)",
    )
    return p


def _check(p, args) -> None:
    if any(m < 1 for m in args.tokens) or any(s < 1 for s in args.seq_lens):
        p.error("--tokens and --seq-lens must be positive")
    if (args.hidden * 2) % 16:
        p.error("--hidden must make a row a multiple of 16 B")
    if args.warmup < 0 or args.iterations < 1:
        p.error("--warmup must be nonnegative and --iterations positive")
    args.tokens = sorted(set(args.tokens))
    args.seq_lens = sorted(set(args.seq_lens))


def main(argv=None) -> int:
    p = _parser()
    args = p.parse_args(argv)
    _check(p, args)
    failed = []
    for collective in args.collectives:
        csv_path, pngs = _paths(args.out_dir, collective, args.tag)
        if not args.plot_only:
            for tp in args.tp:
                why = _can_run(collective, tp, args)
                if why:
                    print(f"SKIP {collective} TP{tp}: {why}", flush=True)
                    continue
                env = " ".join(
                    f"{k}={v}" for k, v in TP_ENV.get((collective, tp), {}).items()
                )
                print(f"=== {collective} TP{tp} {env} ===", flush=True)
                try:
                    with _tp_env(collective, tp):
                        rows = _RUNNERS[collective](tp, args)
                except Exception:  # noqa: BLE001 - keep the other TP sizes
                    traceback.print_exc()
                    failed.append(f"{collective} TP{tp}")
                    continue
                _save(csv_path, tp, rows)
        if not csv_path.exists():
            print(f"no data for {collective}: {csv_path} does not exist", flush=True)
            continue
        df = _load(csv_path)
        if collective in _SHOWN_IMPLS:
            df = df[df["impl"].isin(_SHOWN_IMPLS[collective])]
        df = df[df["us"].map(lambda v: math.isfinite(float(v)) and float(v) > 0)]
        print(_table(df, collective), flush=True)
        for metric, png in pngs.items():
            plot(df, collective, metric, png, subtitle=_subtitle(df, collective))
    if failed:
        print(f"FAILED: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
