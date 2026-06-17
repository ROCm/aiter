"""End-to-end: load an ATT run, render the overlap timeline, emit a report.

    python -m att_analysis.report <run_dir> [--simd 0] [--iters 3]

Writes into ``<run_dir>/att_analysis/``:
    overlap_simd{N}.png   the two-warpgroup timeline
    report.md             window, phase mode, per-wave/per-phase cycle table,
                          overlap & imbalance metrics
"""
from __future__ import annotations

import argparse
import os

from .model import Trace, find_ui_dir
from .window import pick_window
from .timeline import render
from .aggregate import summarize, WaveStats


def _phase_table(s: WaveStats) -> str:
    rows = ["| phase | busy cyc | stall cyc | stall% |", "|---|---:|---:|---:|"]
    for ph in sorted(s.phase_busy, key=lambda p: -s.phase_busy[p]):
        busy = s.phase_busy[ph]
        stall = s.phase_stall.get(ph, 0)
        rows.append(f"| {ph} | {busy} | {stall} | {100*stall/max(busy,1):.0f}% |")
    return "\n".join(rows)


def _state_line(s: WaveStats) -> str:
    parts = [f"{st}={s.state_cycles.get(st,0)} ({100*s.state_frac(st):.0f}%)"
             for st in ("exec", "stall", "wait", "idle")]
    return "  ".join(parts)


def build_report(run_dir: str, simd: int = 0, se: int = 0, n_iters: int = 3) -> str:
    ui = find_ui_dir(run_dir)
    trace = Trace(ui)
    out_dir = os.path.join(run_dir, "att_analysis")
    os.makedirs(out_dir, exist_ok=True)

    w0, _ = trace.coresident_pair(se=se, simd=simd)
    lo, hi, _bounds = pick_window(trace, w0, n_iters=n_iters)

    png = os.path.join(out_dir, f"overlap_simd{simd}.png")
    render(trace, simd=simd, se=se, n_iters=n_iters, out_path=png)

    summ = summarize(trace, lo, hi, se=se, simd=simd)
    s0, s1 = summ["wg0"], summ["wg1"]
    ov = summ["overlap"]

    md = [
        f"# FA4 ATT overlap report",
        f"- run: `{os.path.basename(run_dir.rstrip('/'))}`",
        f"- ui dir: `{os.path.relpath(ui, run_dir)}`",
        f"- SE {se}  SIMD {simd}  CU {w0.cu}",
        f"- window: cycles [{lo}, {hi}]  ({hi-lo} cyc, {n_iters} iters)",
        f"- phase mode: **{summ['phase_mode']}**  (source mapping "
        f"{'present' if trace.has_source else 'ABSENT — rebuild with -gline-tables-only'})",
        "",
        f"![overlap](overlap_simd{simd}.png)",
        "",
        "## Warp-group overlap",
        f"- MATRIX//SOFTMAX overlap (good): **{100*ov['matrix_softmax_overlap']:.0f}%** of window",
        f"- same-phase collision (bad): **{100*ov['same_phase_collision']:.0f}%** of window",
        "",
        f"## WG0 (slot0 wv{s0.wid}) — states",
        f"`{_state_line(s0)}`",
        "",
        _phase_table(s0),
        "",
        f"## WG1 (slot1 wv{s1.wid}) — states",
        f"`{_state_line(s1)}`",
        "",
        _phase_table(s1),
        "",
        "## Imbalance",
        f"- exec%: WG0 {100*s0.state_frac('exec'):.0f}%  vs  WG1 {100*s1.state_frac('exec'):.0f}%",
        f"- stall%: WG0 {100*s0.state_frac('stall'):.0f}%  vs  WG1 {100*s1.state_frac('stall'):.0f}%",
        f"- wait%: WG0 {100*s0.state_frac('wait'):.0f}%  vs  WG1 {100*s1.state_frac('wait'):.0f}%",
    ]
    report_md = os.path.join(out_dir, "report.md")
    with open(report_md, "w") as f:
        f.write("\n".join(md) + "\n")
    return report_md


def main():
    ap = argparse.ArgumentParser(description="FA4 ATT overlap report")
    ap.add_argument("run_dir")
    ap.add_argument("--simd", type=int, default=0)
    ap.add_argument("--se", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()
    path = build_report(args.run_dir, simd=args.simd, se=args.se, n_iters=args.iters)
    print(f"[ok] report -> {path}")
    with open(path) as f:
        print(f.read())


if __name__ == "__main__":
    main()
