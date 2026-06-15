#!/usr/bin/env python3
"""Generate a self-contained, shareable roofline / MFU / MBU report from a tuned
MoE CSV (e.g. tuned_hybrid_fmoe.csv).

The output is a single .html file with everything inlined (CSS + SVG, no network,
no JS libraries), so it can be emailed or sent to anyone and opened in a browser.

Hardware peaks are a pluggable interface: pass a small JSON describing the HBM
bandwidth and the per-precision matrix-compute peaks. A sample is built in and
can be dumped with --emit-sample-peaks.

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
  # 1) dump a sample peaks file you can edit
  python3 moe_roofline_report.py --emit-sample-peaks peaks.sample.json

  # 2) generate a report from a tuned csv + peaks
  python3 moe_roofline_report.py \
      -i tuned_hybrid_fmoe.csv \
      -o moe_report.html \
      --peaks peaks.sample.json \
      --title "DSv4 hybrid fMoE — gfx950"

  # peaks can also be given inline (override the json / defaults)
  python3 moe_roofline_report.py -i tuned_hybrid_fmoe.csv -o moe_report.html \
      --hbm-gbs 8000 --tf int8=3567 --tf fp8=3567 --tf fp4=5663

The CSV is expected to have at least these columns (extra columns are ignored):
  token, model_dim, inter_dim, expert, topk, use_g1u1,
  q_dtype_a, q_dtype_w, dtype, us, [active_expert]
`active_expert` is optional; if absent, total `expert` is used (over-estimates
weight traffic / MBU at small token counts).
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
from dataclasses import dataclass, field

# ── Peak hardware interface ────────────────────────────────────────────────
# Built-in sample: AMD gfx950 (CDNA4, 256 CU). Edit freely or pass --peaks.
SAMPLE_PEAKS = {
    "hbm_gbs": 8000.0,  # HBM peak bandwidth, GB/s (8 TB/s)
    "compute_tflops": {  # matrix-engine peak TFLOPS by precision category
        "fp4": 5663.0,
        "fp8": 3567.0,
        "int8": 3567.0,
        "fp16": 1783.0,
        "bf16": 1783.0,
    },
}

# bytes-per-element for traffic accounting (packed dtypes are < 1).
BPE = {
    "fp4": 0.5,   # 2 values per byte (fp4x2)
    "int4": 0.5,  # 2 values per byte (i4x2)
    "fp8": 1.0,
    "int8": 1.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp32": 4.0,
    "int32": 4.0,
}

# colors for up to a handful of precision groups (kept flat / minimal).
GROUP_COLORS = ["#2E79B5", "#F0A040", "#1F8A65", "#7B64B8", "#C85898", "#C06028"]


def classify_dtype(s: str) -> str:
    """Map a torch dtype string to a precision category key."""
    t = (s or "").lower()
    if "float4" in t or "fp4" in t or "e2m1" in t:
        return "fp4"
    if "int4" in t or "i4" in t:
        return "int4"
    if "float8" in t or "fp8" in t or "e4m3" in t or "e5m2" in t or "e8m0" in t:
        return "fp8"
    if "int8" in t or "i8" in t:
        return "int8"
    if "bfloat16" in t or "bf16" in t:
        return "bf16"
    if "float16" in t or "fp16" in t or "half" in t:
        return "fp16"
    if "float32" in t or "fp32" in t:
        return "fp32"
    if "int32" in t or "i32" in t:
        return "int32"
    return t or "unknown"


def bpe_of(cat: str) -> float:
    return BPE.get(cat, 1.0)


@dataclass
class Peaks:
    hbm_gbs: float
    compute_tflops: dict

    def peak_tf(self, cat: str):
        return self.compute_tflops.get(cat)


@dataclass
class Row:
    token: int
    model_dim: int
    inter_dim: int
    expert: int
    topk: int
    use_g1u1: int
    nact: int
    cat_a: str       # activation precision (drives compute peak)
    cat_w: str       # weight precision
    cat_out: str     # output precision (bf16 typically)
    us: float
    flop: float = 0.0
    bytes: float = 0.0
    tflops: float = 0.0
    bw: float = 0.0
    ai: float = 0.0
    mfu: float = field(default=float("nan"))
    mbu: float = field(default=float("nan"))


# ── CSV parsing + metric computation ───────────────────────────────────────
def _to_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _to_float(v, default=float("nan")):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def load_rows(csv_path: str, peaks: Peaks):
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            if not raw:
                continue
            token = _to_int(raw.get("token"))
            us = _to_float(raw.get("us"))
            if token <= 0 or not (us == us) or us <= 0:
                continue  # skip blanks / failed rows
            expert = _to_int(raw.get("expert"))
            topk = _to_int(raw.get("topk"))
            model_dim = _to_int(raw.get("model_dim"))
            inter_dim = _to_int(raw.get("inter_dim"))
            use_g1u1 = _to_int(raw.get("use_g1u1"))
            nact = _to_int(raw.get("active_expert"), default=0) or expert
            nact = max(1, min(expert or nact, nact))
            cat_a = classify_dtype(raw.get("q_dtype_a", ""))
            cat_w = classify_dtype(raw.get("q_dtype_w", ""))
            cat_out = classify_dtype(raw.get("dtype", "")) or "bf16"

            n = inter_dim * 2 if use_g1u1 else inter_dim
            flop = (
                token * n * model_dim * topk * 2
                + topk * token * model_dim * inter_dim * 2
            )
            data_bytes = (
                token * model_dim * bpe_of(cat_a)
                + n * model_dim * bpe_of(cat_w) * nact
                + inter_dim * model_dim * bpe_of(cat_w) * nact
                + token * model_dim * bpe_of(cat_out)
            )
            r = Row(
                token=token, model_dim=model_dim, inter_dim=inter_dim,
                expert=expert, topk=topk, use_g1u1=use_g1u1, nact=nact,
                cat_a=cat_a, cat_w=cat_w, cat_out=cat_out, us=us,
                flop=flop, bytes=data_bytes,
            )
            r.tflops = flop / us / 1e6
            r.bw = data_bytes / us / 1e3
            r.ai = flop / data_bytes if data_bytes else 0.0
            ptf = peaks.peak_tf(cat_a)
            r.mfu = (r.tflops / ptf * 100.0) if ptf else float("nan")
            r.mbu = (r.bw / peaks.hbm_gbs * 100.0) if peaks.hbm_gbs else float("nan")
            rows.append(r)
    rows.sort(key=lambda x: (x.cat_a, x.token))
    return rows


def group_by_precision(rows):
    groups = {}
    for r in rows:
        groups.setdefault(r.cat_a, []).append(r)
    for g in groups.values():
        g.sort(key=lambda x: x.token)
    return groups


# ── tiny SVG helpers (log-log roofline + line charts) ──────────────────────
def _fmt(x, d=1):
    if x != x:
        return "—"
    return f"{x:,.{d}f}"


def _decade_label(d):
    return f"{d/1000:g}k" if d >= 1000 else f"{d:g}"


def roofline_svg(groups, peaks, colors):
    W, H = 880, 500
    ml, mr, mt, mb = 70, 130, 24, 56
    pw, ph = W - ml - mr, H - mt - mb

    all_ai = [r.ai for g in groups.values() for r in g if r.ai > 0]
    all_tf = [r.tflops for g in groups.values() for r in g]
    ridges = {c: (peaks.peak_tf(c) / (peaks.hbm_gbs / 1000.0))
              for c in groups if peaks.peak_tf(c)}
    max_ai = max(all_ai + list(ridges.values()) + [10])
    max_tf = max(all_tf + [peaks.peak_tf(c) or 0 for c in groups] + [10])

    xhi = max(1, math.ceil(math.log10(max_ai * 1.6)))
    yhi = max(1, math.ceil(math.log10(max_tf * 1.3)))
    xlo, ylo = 0, 0

    def sx(ai):
        ai = max(ai, 10 ** xlo)
        return ml + (math.log10(ai) - xlo) / (xhi - xlo) * pw

    def sy(p):
        p = max(p, 10 ** ylo)
        return mt + ph - (math.log10(p) - ylo) / (yhi - ylo) * ph

    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
         f'font-family="ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,sans-serif">']
    # grid
    for k in range(xlo, xhi + 1):
        d = 10 ** k
        x = sx(d)
        s.append(f'<line x1="{x:.1f}" y1="{mt}" x2="{x:.1f}" y2="{mt+ph}" stroke="#e5e7eb"/>')
        s.append(f'<text x="{x:.1f}" y="{mt+ph+18}" font-size="11" fill="#6b7280" '
                 f'text-anchor="middle">{_decade_label(d)}</text>')
    for k in range(ylo, yhi + 1):
        d = 10 ** k
        y = sy(d)
        s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        s.append(f'<text x="{ml-8}" y="{y+4:.1f}" font-size="11" fill="#6b7280" '
                 f'text-anchor="end">{_decade_label(d)}</text>')

    # shade the compute-bound region (right of the lowest precision ridge)
    if ridges:
        rx = sx(min(ridges.values()))
        s.append(f'<rect x="{rx:.1f}" y="{mt}" width="{ml+pw-rx:.1f}" height="{ph}" '
                 f'fill="#2E79B5" opacity="0.04"/>')
        s.append(f'<text x="{(ml+rx)/2:.0f}" y="{mt+ph-10}" font-size="10.5" '
                 f'fill="#9ca3af" text-anchor="middle" font-style="italic">memory-bound</text>')
        s.append(f'<text x="{(rx+ml+pw)/2:.0f}" y="{mt+ph-10}" font-size="10.5" '
                 f'fill="#9ca3af" text-anchor="middle" font-style="italic">compute-bound</text>')

    # HBM memory diagonal: perf = (hbm/1000) * AI, up to the largest ridge
    slope = peaks.hbm_gbs / 1000.0
    rmax = max(ridges.values()) if ridges else max_ai
    s.append(f'<line x1="{sx(10**xlo):.1f}" y1="{sy(slope*(10**xlo)):.1f}" '
             f'x2="{sx(rmax):.1f}" y2="{sy(slope*rmax):.1f}" stroke="#374151" '
             f'stroke-width="2"/>')
    midx = 10 ** ((xlo + min(xhi, math.log10(max(rmax, 10))) ) / 2)
    s.append(f'<text x="{sx(midx):.1f}" y="{sy(slope*midx)-7:.1f}" font-size="11" '
             f'fill="#374151" transform="rotate(30 {sx(midx):.1f} {sy(slope*midx)-7:.1f})">'
             f'HBM {peaks.hbm_gbs/1000:g} TB/s</text>')

    # per-precision compute ceilings + points
    for i, (cat, g) in enumerate(groups.items()):
        col = colors[i % len(colors)]
        ptf = peaks.peak_tf(cat)
        if ptf:
            ridge = ridges[cat]
            s.append(f'<line x1="{sx(ridge):.1f}" y1="{sy(ptf):.1f}" '
                     f'x2="{sx(10**xhi):.1f}" y2="{sy(ptf):.1f}" stroke="{col}" '
                     f'stroke-width="2"/>')
            s.append(f'<text x="{ml+pw+6}" y="{sy(ptf)+4:.1f}" font-size="11" '
                     f'fill="{col}">{cat.upper()} {ptf:g}</text>')
        for r in g:
            if r.ai > 0:
                s.append(f'<circle cx="{sx(r.ai):.1f}" cy="{sy(r.tflops):.1f}" r="4" '
                         f'fill="{col}"><title>{cat} token={r.token}: {_fmt(r.tflops,0)} TFLOPS, '
                         f'AI={_fmt(r.ai,1)}, MFU={_fmt(r.mfu)}%</title></circle>')

    # legend
    lx, ly = ml + 12, mt + 14
    for i, cat in enumerate(groups):
        col = colors[i % len(colors)]
        s.append(f'<circle cx="{lx}" cy="{ly+i*18}" r="4" fill="{col}"/>')
        s.append(f'<text x="{lx+10}" y="{ly+i*18+4}" font-size="11" fill="#374151">'
                 f'{cat.upper()}</text>')

    # axis titles
    s.append(f'<text x="{ml+pw/2:.0f}" y="{H-8}" font-size="12" fill="#374151" '
             f'text-anchor="middle">Arithmetic Intensity (FLOP / byte, log)</text>')
    s.append(f'<text x="16" y="{mt+ph/2:.0f}" font-size="12" fill="#374151" '
             f'text-anchor="middle" transform="rotate(-90 16 {mt+ph/2:.0f})">'
             f'Performance (TFLOPS, log)</text>')
    s.append("</svg>")
    return "".join(s)


def line_chart_svg(groups, colors, metric, title, ymax=None):
    """metric: 'mbu' or 'mfu'. x = token (even spacing), y = percent."""
    W, H = 440, 280
    ml, mr, mt, mb = 48, 16, 34, 46
    pw, ph = W - ml - mr, H - mt - mb
    tokens = sorted({r.token for g in groups.values() for r in g})
    if not tokens:
        return ""
    vals = [getattr(r, metric) for g in groups.values() for r in g
            if getattr(r, metric) == getattr(r, metric)]
    ytop = ymax if ymax else max(10, math.ceil((max(vals) if vals else 10) / 10) * 10)
    n = len(tokens)
    idx = {t: i for i, t in enumerate(tokens)}

    def sx(i):
        return ml + (i / max(1, n - 1)) * pw

    def sy(v):
        return mt + ph - (v / ytop) * ph

    s = [f'<svg viewBox="0 0 {W} {H}" width="100%" role="img" '
         f'font-family="ui-sans-serif,system-ui,sans-serif">']
    s.append(f'<text x="{ml}" y="14" font-size="13" fill="#111827" '
             f'font-weight="600">{html.escape(title)}</text>')
    # legend (top-right)
    lx = ml + pw
    for gi, cat in enumerate(reversed(list(groups))):
        col = colors[(len(groups) - 1 - gi) % len(colors)]
        s.append(f'<line x1="{lx-46}" y1="10" x2="{lx-32}" y2="10" stroke="{col}" stroke-width="3"/>')
        s.append(f'<text x="{lx-28}" y="14" font-size="11" fill="#374151">{cat.upper()}</text>')
        lx -= 70
    # y gridlines + labels
    for k in range(0, 5):
        v = ytop * k / 4
        y = sy(v)
        s.append(f'<line x1="{ml}" y1="{y:.1f}" x2="{ml+pw}" y2="{y:.1f}" stroke="#eef0f2"/>')
        s.append(f'<text x="{ml-6}" y="{y+4:.1f}" font-size="10" fill="#9ca3af" '
                 f'text-anchor="end">{v:g}</text>')
    # x labels (subset to avoid clutter)
    step = max(1, n // 7)
    for i, t in enumerate(tokens):
        if i % step == 0 or i == n - 1:
            s.append(f'<text x="{sx(i):.1f}" y="{mt+ph+16}" font-size="9" fill="#9ca3af" '
                     f'text-anchor="middle">{_decade_label(t)}</text>')
    # series: area fill + line + points
    base = sy(0)
    for gi, (cat, g) in enumerate(groups.items()):
        col = colors[gi % len(colors)]
        pts = [(sx(idx[r.token]), sy(getattr(r, metric)))
               for r in g if getattr(r, metric) == getattr(r, metric)]
        if not pts:
            continue
        area = (f'{pts[0][0]:.1f},{base:.1f} '
                + " ".join(f'{x:.1f},{y:.1f}' for x, y in pts)
                + f' {pts[-1][0]:.1f},{base:.1f}')
        s.append(f'<polygon points="{area}" fill="{col}" opacity="0.08"/>')
        s.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" '
                 f'fill="none" stroke="{col}" stroke-width="2"/>')
        for r in g:
            v = getattr(r, metric)
            if v == v:
                s.append(f'<circle cx="{sx(idx[r.token]):.1f}" cy="{sy(v):.1f}" r="2.6" '
                         f'fill="{col}"><title>{cat} t={r.token}: {_fmt(v)}%</title></circle>')
    # axis titles
    s.append(f'<text x="{ml+pw/2:.0f}" y="{H-6}" font-size="10.5" fill="#6b7280" '
             f'text-anchor="middle">tokens (log spacing)</text>')
    s.append(f'<text x="13" y="{mt+ph/2:.0f}" font-size="10.5" fill="#6b7280" '
             f'text-anchor="middle" transform="rotate(-90 13 {mt+ph/2:.0f})">percent (%)</text>')
    s.append("</svg>")
    return "".join(s)


# ── HTML assembly ──────────────────────────────────────────────────────────
def stat_card(value, label, accent="#111827"):
    return (f'<div class="card"><div class="stat" style="color:{accent}">{value}</div>'
            f'<div class="lbl">{html.escape(label)}</div></div>')


def metrics_table(cat, rows):
    head = ("<tr><th>token</th><th>n_act/E</th><th>us</th><th>AI</th>"
            "<th>TFLOPS</th><th>MFU %</th><th>BW GB/s</th><th>MBU %</th></tr>")
    body = []
    for r in rows:
        tone = ("good" if r.mbu >= 60 else "mid" if r.mbu >= 35 else "low")
        body.append(
            f'<tr class="{tone}"><td>{r.token:,}</td>'
            f'<td>{r.nact}/{r.expert}</td><td>{_fmt(r.us)}</td>'
            f'<td>{_fmt(r.ai)}</td><td>{_fmt(r.tflops,0)}</td>'
            f'<td>{_fmt(r.mfu)}</td><td>{_fmt(r.bw,0)}</td>'
            f'<td>{_fmt(r.mbu)}</td></tr>')
    return (f'<h3>{cat.upper()} <span class="muted">({len(rows)} shapes)</span></h3>'
            f'<table>{head}{"".join(body)}</table>')


def callout(tone, title, body):
    tones = {"info": "#2E79B5", "good": "#1F8A65", "warn": "#B45309"}
    bgs = {"info": "#f0f6fc", "good": "#f0faf5", "warn": "#fdf6ec"}
    c = tones.get(tone, "#374151")
    bg = bgs.get(tone, "#f7f7f8")
    return (f'<div class="callout" style="border-left-color:{c};background:{bg}">'
            f'<div class="ct" style="color:{c}">{html.escape(title)}</div>'
            f'<div class="cb">{body}</div></div>')


def build_callouts(groups, peaks):
    outs = []

    def peak_of(g, metric):
        cand = [r for r in g if getattr(r, metric) == getattr(r, metric)]
        return max(cand, key=lambda r: getattr(r, metric)) if cand else None

    # regime callout
    bits = []
    for cat, g in groups.items():
        pm = peak_of(g, "mbu")
        pf = peak_of(g, "mfu")
        if pm and pf:
            bits.append(f"<b>{cat.upper()}</b>: peak MBU {_fmt(pm.mbu)}% (t={pm.token:,}), "
                        f"plateau MFU {_fmt(pf.mfu)}% (t={pf.token:,})")
    if bits:
        outs.append(callout(
            "info", "Two regimes: memory-bound → compute-bound",
            "Small batches sit left of the ridge (HBM-bound: high MBU, tiny MFU because "
            "expert-weight loads dominate). As tokens grow, arithmetic intensity passes the "
            "ridge and the kernels turn compute-bound — MFU climbs to a plateau while MBU "
            "collapses. " + "; ".join(bits) + "."))

    # fp4 vs int8 comparison at the largest shared token
    if "fp4" in groups and "int8" in groups:
        f_by = {r.token: r for r in groups["fp4"]}
        i_by = {r.token: r for r in groups["int8"]}
        shared = sorted(set(f_by) & set(i_by))
        if shared:
            t = shared[-1]
            fr, ir = f_by[t], i_by[t]
            spd = fr.tflops / ir.tflops if ir.tflops else float("nan")
            outs.append(callout(
                "good", "FP4 vs INT8",
                f"At the top end (t={t:,}) FP4 delivers <b>{_fmt(fr.tflops,0)} TFLOPS</b> vs "
                f"INT8 <b>{_fmt(ir.tflops,0)} TFLOPS</b> — about <b>{_fmt(spd)}&times;</b> faster, "
                f"and moves half the weight bytes (0.5 B/elem vs 1 B/elem), so it stays "
                f"bandwidth-efficient longer."))
    return "".join(outs)


def build_html(rows, peaks, title, src_name):
    groups = group_by_precision(rows)
    colors = GROUP_COLORS

    def best(metric):
        cand = [r for r in rows if getattr(r, metric) == getattr(r, metric)]
        return max(cand, key=lambda r: getattr(r, metric)) if cand else None
    bmbu, bmfu = best("mbu"), best("mfu")

    cards = []
    if bmbu:
        cards.append(stat_card(f"{_fmt(bmbu.mbu)}%",
                     f"peak MBU · {bmbu.cat_a} t={bmbu.token:,}", "#1F8A65"))
    if bmfu:
        cards.append(stat_card(f"{_fmt(bmfu.mfu)}%",
                     f"peak MFU · {bmfu.cat_a} t={bmfu.token:,}", "#2E79B5"))
    cards.append(stat_card(f"{peaks.hbm_gbs/1000:g} TB/s", "HBM peak", "#374151"))
    peak_str = ", ".join(f"{c}={peaks.peak_tf(c):g}" for c in groups if peaks.peak_tf(c))
    cards.append(stat_card(f"{len(rows)}", "tuned shapes", "#374151"))

    roof = roofline_svg(groups, peaks, colors)
    mbu_chart = line_chart_svg(groups, colors, "mbu", "MBU vs token", ymax=100)
    mfu_chart = line_chart_svg(groups, colors, "mfu", "MFU vs token")
    tables = "".join(metrics_table(c, g) for c, g in groups.items())
    callouts = build_callouts(groups, peaks)

    legend = " &nbsp; ".join(
        f'<span class="dot" style="background:{colors[i%len(colors)]}"></span>{c.upper()}'
        for i, c in enumerate(groups))

    style = """
    :root{color-scheme:light}
    *{box-sizing:border-box}
    body{margin:0;background:#f6f7f9;color:#111827;
         font-family:ui-sans-serif,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
         line-height:1.55;-webkit-font-smoothing:antialiased}
    .wrap{max-width:1060px;margin:0 auto;padding:32px 24px 72px}
    .hd{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap}
    h1{font-size:23px;margin:0;letter-spacing:-.3px}
    .pill{font-size:11.5px;color:#2E79B5;background:#eaf2fb;border:1px solid #d6e6f7;
          border-radius:999px;padding:2px 10px;font-weight:600}
    h2{font-size:15px;margin:32px 0 12px;color:#111827;text-transform:uppercase;
       letter-spacing:.04em;font-weight:700}
    h3{font-size:13.5px;margin:18px 0 8px}
    .sub{color:#6b7280;font-size:13px;margin:6px 0 4px}
    .muted{color:#9ca3af;font-weight:400}
    code{background:#eceef1;padding:1px 5px;border-radius:4px;font-size:12px}
    .dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px;
         vertical-align:middle}
    .cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0 6px}
    .card{background:#fff;border:1px solid #e7e9ec;border-radius:12px;padding:15px 17px}
    .stat{font-size:25px;font-weight:750;letter-spacing:-.6px}
    .lbl{font-size:12px;color:#6b7280;margin-top:3px}
    .panel{background:#fff;border:1px solid #e7e9ec;border-radius:12px;padding:18px}
    .two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .callout{border:1px solid #e7e9ec;border-left-width:4px;border-radius:8px;
             padding:11px 15px;margin:10px 0}
    .callout .ct{font-weight:700;font-size:13px;margin-bottom:2px}
    .callout .cb{font-size:12.5px;color:#374151}
    table{width:100%;border-collapse:collapse;font-size:12.5px;margin-bottom:10px}
    th,td{text-align:right;padding:6px 9px;border-bottom:1px solid #f0f1f3}
    th{color:#6b7280;font-weight:600;border-bottom:1px solid #e5e7eb}
    td:first-child,th:first-child{text-align:left}
    tbody tr:hover{background:#fafbfc}
    tr.good td:last-child{color:#1F8A65;font-weight:700}
    tr.low td:last-child{color:#C0392B;font-weight:700}
    .note{font-size:12px;color:#6b7280;margin-top:8px}
    footer{margin-top:44px;color:#9ca3af;font-size:12px;border-top:1px solid #e7e9ec;padding-top:14px}
    @media(max-width:760px){.cards{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}}
    """

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title><style>{style}</style></head>
<body><div class="wrap">
<div class="hd"><h1>{html.escape(title)}</h1><span class="pill">{legend}</span></div>
<p class="sub">2-stage fused MoE roofline · source <code>{html.escape(src_name)}</code> ·
Peaks: HBM <code>{peaks.hbm_gbs:g} GB/s</code>, compute <code>{html.escape(peak_str)}</code> TFLOPS.
Bytes use packed-aware element sizes &times; <code>active_expert</code> (distinct experts hit by routing).</p>
<div class="cards">{''.join(cards)}</div>
{callouts}
<h2>Roofline</h2>
<div class="panel">{roof}</div>
<p class="note">Diagonal = HBM bandwidth ceiling; horizontals = per-precision compute peaks.
Each dot is one token shape (hover for details). Shaded band = compute-bound region (right of the lowest ridge).</p>
<h2>Utilization vs batch size</h2>
<div class="two"><div class="panel">{mbu_chart}</div><div class="panel">{mfu_chart}</div></div>
<h2>Per-shape metrics</h2>
{tables}
<p class="note">AI = FLOP / bytes. n_act = distinct experts hit by routing (column
<code>active_expert</code>; falls back to total E if absent). MFU = TFLOPS / precision peak;
MBU = BW / HBM peak. MBU row tint: green &ge;60%, red &lt;35%.</p>
<footer>Generated by <code>moe_roofline_report.py</code> — self-contained single file, no external assets. Share freely.</footer>
</div></body></html>"""


# ── peaks loading / CLI ────────────────────────────────────────────────────
def load_peaks(args) -> Peaks:
    data = json.loads(json.dumps(SAMPLE_PEAKS))  # deep copy of defaults
    if args.peaks:
        with open(args.peaks) as f:
            user = json.load(f)
        if "hbm_gbs" in user:
            data["hbm_gbs"] = float(user["hbm_gbs"])
        data["compute_tflops"].update(
            {k: float(v) for k, v in user.get("compute_tflops", {}).items()})
    if args.hbm_gbs is not None:
        data["hbm_gbs"] = float(args.hbm_gbs)
    for item in args.tf or []:
        if "=" not in item:
            sys.exit(f"--tf expects cat=value, got: {item}")
        k, v = item.split("=", 1)
        data["compute_tflops"][classify_dtype(k) if classify_dtype(k) != "unknown" else k] = float(v)
    return Peaks(hbm_gbs=data["hbm_gbs"], compute_tflops=data["compute_tflops"])


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Generate a shareable HTML roofline/MFU/MBU report from a tuned MoE CSV.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument("-i", "--input", help="tuned CSV (e.g. tuned_hybrid_fmoe.csv)")
    p.add_argument("-o", "--output", default="moe_report.html", help="output .html path")
    p.add_argument("--peaks", help="JSON file with hbm_gbs + compute_tflops{cat:TF}")
    p.add_argument("--hbm-gbs", type=float, help="override HBM bandwidth (GB/s)")
    p.add_argument("--tf", action="append", metavar="cat=TF",
                   help="override a precision peak, e.g. --tf fp4=5663 (repeatable)")
    p.add_argument("--title", default="MoE Roofline Report", help="report title")
    p.add_argument("--emit-sample-peaks", metavar="PATH",
                   help="write a sample peaks JSON to PATH and exit")
    args = p.parse_args(argv)

    if args.emit_sample_peaks:
        with open(args.emit_sample_peaks, "w") as f:
            json.dump(SAMPLE_PEAKS, f, indent=2)
        print(f"wrote sample peaks -> {args.emit_sample_peaks}")
        return 0

    if not args.input:
        p.error("the following argument is required: -i/--input (or use --emit-sample-peaks)")
    if not os.path.exists(args.input):
        sys.exit(f"input not found: {args.input}")

    peaks = load_peaks(args)
    rows = load_rows(args.input, peaks)
    if not rows:
        sys.exit("no valid rows found in CSV (need token/us columns with data)")
    html_doc = build_html(rows, peaks, args.title, os.path.basename(args.input))
    with open(args.output, "w") as f:
        f.write(html_doc)
    groups = group_by_precision(rows)
    print(f"report written -> {args.output}")
    print(f"  rows: {len(rows)} | precision groups: "
          + ", ".join(f"{c}({len(g)})" for c, g in groups.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
