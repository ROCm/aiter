"""Render the two-warpgroup overlap timeline (headless PNG).

For a chosen SIMD we take the two co-resident waves (slot0 / slot1 -- one per
warp group on the FA4 pipeline) and draw, on a shared absolute-cycle axis, two
rows per wave:

    [phase] colored by MATRIX / SOFTMAX / barrier / memwait   (what work it is)
    [state] colored by exec / wait / stall / idle             (what the HW did)

Stacking the two warp groups makes pipeline overlap and imbalance obvious: when
one group is in MATRIX the co-resident group should be in SOFTMAX, and large
stall/wait bands flag bubbles.
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .model import Trace, Wave, find_ui_dir, STATE_NAME
from .phases import PhaseTagger, PHASE_COLORS, PHASE_ORDER, OTHER
from .window import pick_window

STATE_COLORS = {"idle": "#ffffff", "exec": "#2ca02c", "wait": "#7f7f7f", "stall": "#ff7f0e"}
STATE_EDGE = {"idle": "#cccccc", "exec": "#2ca02c", "wait": "#7f7f7f", "stall": "#ff7f0e"}


def _phase_bars(trace, tagger, wave: Wave, lo: int, hi: int):
    """(xranges_by_phase) tiling [lo,hi] contiguously.

    Each instruction owns cycles from its issue until the next instruction's
    issue, so the phase row fully tiles the window (no misleading white gaps --
    those would otherwise be exec-state cycles merely uncovered by token.total).
    """
    insts = wave.insts_in_window(lo, hi)
    by_phase: dict[str, list[tuple[int, int]]] = {}
    for k, i in enumerate(insts):
        ph = tagger.phase_of_lineno(i.lineno)
        nxt = insts[k + 1].cycle if k + 1 < len(insts) else min(i.cycle + max(i.total, 1), hi)
        width = max(nxt - i.cycle, 1)
        by_phase.setdefault(ph, []).append((i.cycle, width))
    return by_phase


def _state_bars(wave: Wave, lo: int, hi: int):
    by_state: dict[str, list[tuple[int, int]]] = {}
    for s0, s1, st in wave.state_segments():
        if s1 < lo or s0 > hi:
            continue
        a, b = max(s0, lo), min(s1, hi)
        if b <= a:
            continue
        by_state.setdefault(STATE_NAME.get(st, "idle"), []).append((a, b - a))
    return by_state


def render(trace: Trace, simd: int = 0, se: int = 0, n_iters: int = 3,
           out_path: str = "overlap.png", title: str | None = None):
    w0, w1 = trace.coresident_pair(se=se, simd=simd)
    tagger = PhaseTagger(trace)

    # Window from wave0's barriers; both waves share the clock domain.
    lo, hi, bounds = pick_window(trace, w0, n_iters=n_iters)

    fig, ax = plt.subplots(figsize=(15, 4.2))

    # Row layout, read top-to-bottom: WG0 (phase, state), gap, WG1 (phase, state).
    row_h = 0.8
    rows = [
        (5.2, "phase", w0, f"WG0 sl0 wv{w0.wid}  phase"),
        (4.3, "state", w0, "state"),
        (2.6, "phase", w1, f"WG1 sl1 wv{w1.wid}  phase"),
        (1.7, "state", w1, "state"),
    ]
    yticks, ylabels = [], []
    for y, kind, wv, label in rows:
        if kind == "phase":
            bars = _phase_bars(trace, tagger, wv, lo, hi)
            for ph, xr in bars.items():
                ax.broken_barh(xr, (y, row_h), facecolors=PHASE_COLORS.get(ph, OTHER),
                               edgecolor="none")
        else:
            bars = _state_bars(wv, lo, hi)
            for st, xr in bars.items():
                ax.broken_barh(xr, (y, row_h), facecolors=STATE_COLORS.get(st, "#fff"),
                               edgecolor=STATE_EDGE.get(st, "#ccc"), linewidth=0.1)
        yticks.append(y + row_h / 2); ylabels.append(label)

    # Iteration boundaries.
    for b in bounds:
        if lo <= b <= hi:
            ax.axvline(b, color="#d62728", lw=0.8, ls="--", alpha=0.6)

    ax.set_xlim(lo, hi)
    ax.set_ylim(1.2, 6.3)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_xlabel(f"cycle (absolute; window = {hi - lo} cyc, {n_iters} iters)")

    phase_legend = [Patch(facecolor=PHASE_COLORS[p], label=p) for p in PHASE_ORDER]
    state_legend = [Patch(facecolor=STATE_COLORS[s], edgecolor=STATE_EDGE[s], label=f"state:{s}")
                    for s in ("exec", "stall", "wait", "idle")]
    ax.legend(handles=phase_legend + state_legend, ncol=6, fontsize=7,
              loc="upper center", bbox_to_anchor=(0.5, -0.18))

    if title is None:
        title = (f"FA4 warp-group overlap  |  SE{se} SIMD{simd} CU{w0.cu}  |  "
                 f"phase mode: {tagger.mode}")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_path, (lo, hi, tagger.mode)


def main():
    ap = argparse.ArgumentParser(description="FA4 two-warpgroup ATT overlap timeline")
    ap.add_argument("run_dir", help="run dir, att/ dir, or ui_output_* dir")
    ap.add_argument("--simd", type=int, default=0)
    ap.add_argument("--se", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    ui = find_ui_dir(args.run_dir)
    trace = Trace(ui)
    out = args.out or os.path.join(os.path.dirname(ui.rstrip("/")), f"overlap_simd{args.simd}.png")
    path, (lo, hi, mode) = render(trace, simd=args.simd, se=args.se,
                                  n_iters=args.iters, out_path=out)
    print(f"[ok] {path}")
    print(f"     window cycles [{lo}, {hi}]  phase-mode={mode}  has_source={trace.has_source}")


if __name__ == "__main__":
    main()
