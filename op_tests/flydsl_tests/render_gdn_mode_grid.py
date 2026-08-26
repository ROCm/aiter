# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Render the grid from sweep_gdn_mode_grid.py as a single self-contained HTML
page: one table per GQA ratio, rows being (H, B) pairs ordered by B*H."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

DEFAULT_JSON = Path(__file__).with_name("gdn_prefill_mode_grid.json")
# The page is a deliverable next to the document that embeds its screenshot, so
# a plain re-run refreshes the checked-in copy instead of leaving a stray one.
DEFAULT_HTML = Path(__file__).resolve().parents[2] / "docs" / "gdn-mode-by-seqlen.html"

# Mode colours.  CF blue / CS orange / WS green / WF pink follow the existing
# GDN mode charts; prep (the FlyDSL path) is new, so it gets violet.
PALETTE = {
    "ws": ("#16a34a", "#22c55e", "#dcfce7"),
    "wf": ("#db2777", "#ec4899", "#fce7f3"),
    "cf": ("#2563eb", "#3b82f6", "#dbeafe"),
    "cs": ("#ea580c", "#f97316", "#ffedd5"),
    "prepare": ("#7c3aed", "#8b5cf6", "#ede9fe"),
}
# Win margin over the runner-up -> opacity band, so a deeper cell is a safer call.
BANDS = ((0.20, 1.00), (0.10, 0.82), (0.05, 0.64), (0.02, 0.46), (0.0, 0.30))


def fmt_len(n: int) -> str:
    if n >= 1024 and n % 1024 == 0:
        return f"{n // 1024}K"
    return str(n)


def fmt_us(us: float) -> str:
    return f"{us / 1000:.3f} ms" if us >= 1000 else f"{us:.1f} us"


def fmt_us_tight(us: float) -> str:
    """Same number for the in-cell overlay, where the column has to stay narrow."""
    return f"{us / 1000:.2f}ms" if us >= 1000 else f"{us:.0f}us"


def band(margin: float) -> float:
    for lo, alpha in BANDS:
        if margin >= lo:
            return alpha
    return 0.30


def mix(a: str, b: str, t: float) -> str:
    """``t`` of colour ``a`` over ``b``, resolved here rather than left to the
    browser's ``color-mix()``, which only lands in Chrome 111 / Safari 16.2."""
    ca = tuple(int(a[i : i + 2], 16) for i in (1, 3, 5))
    cb = tuple(int(b[i : i + 2], 16) for i in (1, 3, 5))
    return "#" + "".join(f"{round(x * t + y * (1 - t)):02x}" for x, y in zip(ca, cb))


def cell_html(cell: dict, backends: list[str], short: dict) -> str:
    walls = cell.get("walls", {})
    if not walls:
        why = cell.get("skipped") or "; ".join(
            f"{short.get(b, b)}: {m}" for b, m in cell.get("errors", {}).items()
        )
        return f'<td class="empty" data-tip="{html.escape(why or "no data")}">—</td>'

    order = sorted(walls, key=walls.get)
    best, best_us = order[0], walls[order[0]]
    margin = (walls[order[1]] / best_us - 1) if len(order) > 1 else 1.0
    deep, mid, _pale = PALETTE[best]

    n_seqs = cell["n_seqs"]
    lines = [
        (
            f"TP={cell.get('tp', '?')} · H={cell['H']} · Hg={cell['Hg']}"
            f" · B={n_seqs} · B·H={n_seqs * cell['H']}"
        ),
        f"seqlen={cell['seqlen']:,} × B={n_seqs}  ⇒  T={cell['total_tokens']:,}",
        f"最快 {short[best]} {fmt_us(best_us)}"
        + (f" · 领先第二名 {margin * 100:.1f}%" if len(order) > 1 else ""),
        "",
    ]
    for b in order:
        rel = walls[b] / best_us
        lines.append(
            f"{short[b]:>4}  {fmt_us(walls[b]):>10}"
            + ("  —" if b == best else f"  +{(rel - 1) * 100:.1f}%")
        )
    for b, msg in cell.get("errors", {}).items():
        lines.append(f"{short.get(b, b):>4}  {msg}")

    tip = html.escape("\n".join(lines))
    a = band(margin)
    style = (
        f"--deep:{deep};"
        f"--bg:{mix(mid, '#ffffff', a * 0.46)};"
        f"--bd:{mix(mid, '#e5e7eb', a * 0.62)};"
        f"--fg:{mix(deep, '#000000', 0.88)}"
    )
    return (
        f'<td class="cell" data-tip="{tip}" data-mode="{best}" '
        f'data-us="{fmt_us_tight(best_us)}" data-margin="{margin * 100:.1f}" '
        f'style="{style}"><span class="pill">{short[best]}</span></td>'
    )


def table_html(tdata: dict, meta: dict, highlight: tuple[int, int] | None) -> str:
    backends, short = meta["backends"], meta["short"]
    seqlens, ratio = meta["seqlens"], tdata["gqa_ratio"]
    by_key = {(c.get("H"), c.get("n_seqs"), c.get("seqlen")): c for c in tdata["cells"]}

    head = "".join(f"<th>{fmt_len(s)}</th>" for s in seqlens)
    rows, prev_bh, prev_split = [], None, None
    for r in sorted(meta["rows"], key=lambda r: (r["bh"], r["tp"])):
        h, n, tp, bh = r["H"], r["n_seqs"], r["tp"], r["bh"]
        cells = "".join(
            cell_html(by_key.get((h, n, s), {}), backends, short) for s in seqlens
        )
        winners = {
            min(c["walls"], key=c["walls"].get)
            for s in seqlens
            if (c := by_key.get((h, n, s), {})).get("walls")
        }
        split = next(iter(winners)) if len(winners) == 1 else None
        cls = []
        # Sorted by B*H, the winner flips exactly once, so one rule marks it;
        # the lighter rules just group the rows sharing a B*H.
        if prev_split is not None and split != prev_split:
            cls.append("brk")
        elif prev_bh is not None and bh != prev_bh:
            cls.append("grp")
        prev_bh, prev_split = bh, split
        is_hl = highlight == (h, n)
        if is_hl:
            cls.append("hl")
        tag = '<span class="tag">生产</span>' if is_hl else ""
        rows.append(
            f'<tr class="{" ".join(cls)}">'
            f'<th class="rh">{tag}{tp}</th>'
            f'<th class="rh sub">{h}</th>'
            f'<th class="rh sub">{h // ratio}</th>'
            f'<th class="rh">{n}</th>'
            f'<th class="rh bh">{bh}</th>{cells}</tr>'
        )

    wins: dict[str, int] = {}
    for c in tdata["cells"]:
        w = c.get("walls")
        if w:
            wins[min(w, key=w.get)] = wins.get(min(w, key=w.get), 0) + 1
    total = sum(wins.values())
    tally = " · ".join(
        f'<span style="color:{PALETTE[b][0]}">{short[b]} {n} 格</span>'
        for b, n in sorted(wins.items(), key=lambda kv: -kv[1])
    )

    return f"""
    <section>
      <h2>GQA ratio {ratio}（Hg = H/{ratio}）<span class="tally">{tally} · 共 {total} 格</span></h2>
      <table>
        <thead>
          <tr class="grouprow">
            <th colspan="5" class="grouphead left">形状（按链条数 B·H 递增）</th>
            <th colspan="{len(seqlens)}" class="grouphead">
              每条序列多长（总 token 数 T = B × seqlen）
            </th>
          </tr>
          <tr>
            <th class="rh">TP</th><th class="rh sub">H</th><th class="rh sub">Hg</th>
            <th class="rh">B</th><th class="rh bh">B·H</th>{head}
          </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </section>"""


def legend_html(meta: dict, data: dict) -> str:
    wins: dict[str, int] = {}
    for t in data["tables"]:
        for c in t["cells"]:
            w = c.get("walls")
            if w:
                wins[min(w, key=w.get)] = wins.get(min(w, key=w.get), 0) + 1
    total = sum(wins.values())
    items = []
    for b in sorted(meta["backends"], key=lambda x: -wins.get(x, 0)):
        deep, mid, pale = PALETTE[b]
        n = wins.get(b, 0)
        # A mode that never wins still belongs in the legend -- "never fastest
        # anywhere on this grid" is a result -- but it should not read as live.
        cls = "lg" if n else "lg zero"
        score = f"{n} 格 · {n / total * 100:.0f}%" if n else "从未最快"
        items.append(
            f'<span class="{cls}" style="--deep:{deep};--mid:{mid};--pale:{pale}">'
            f"<i></i>{html.escape(meta['short'][b])}"
            f"<em>{html.escape(meta['labels'][b])}</em>"
            f"<b>{score}</b></span>"
        )
    return "".join(items)


def render(data: dict) -> str:
    meta = data
    smin, smax = min(data["seqlens"]), max(data["seqlens"])
    bhs = [r["bh"] for r in data["rows"]]
    tps = sorted({r["tp"] for r in data["rows"]})
    sub = (
        f"{data['device']} · {data['gfx']} · {data['cus']} CU · torch {data['torch']} · "
        f"wall = {data['num_iters']} 次迭代中位数 · packed varlen, state I/O on · "
        f"K=V=128 bf16 · Hv={data['hv_model']} · TP {'/'.join(map(str, tps))} · "
        f"seqlen {fmt_len(smin)}~{fmt_len(smax)} · B·H {min(bhs)}~{max(bhs)}"
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GDN Prefill 最快 mode · varlen</title>
<style>
:root {{
  --bg:#f6f7f9; --card:#fff; --ink:#111827; --dim:#6b7280; --line:#e5e7eb;
}}
* {{ box-sizing:border-box }}
body {{
  margin:0; padding:28px; background:var(--bg); color:var(--ink);
  font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI","Microsoft YaHei",
       "PingFang SC",Roboto,Helvetica,Arial,sans-serif;
}}
.wrap {{ max-width:1400px; margin:0 auto }}
header {{
  background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:18px 20px; margin-bottom:20px;
}}
h1 {{ margin:0 0 4px; font-size:18px; font-weight:650; letter-spacing:.2px }}
.sub {{ color:var(--dim); font-size:12px; margin-bottom:14px }}
.legend {{ display:flex; flex-wrap:wrap; gap:10px }}
.lg {{
  display:flex; align-items:center; gap:7px; padding:6px 11px 6px 8px;
  border:1px solid var(--line); border-radius:9px; background:var(--pale);
  font-weight:640; font-size:13px;
}}
.lg i {{ width:11px; height:11px; border-radius:3px; background:var(--mid); flex:none }}
.lg em {{ font-style:normal; color:var(--dim); font-weight:450; font-size:12px }}
.lg b {{ color:var(--deep); font-weight:600; font-size:12px; margin-left:2px }}
.lg.zero {{ background:#fafafa; opacity:.55 }}
.lg.zero b {{ color:var(--dim); font-weight:450 }}
.controls {{ display:flex; gap:8px; margin:14px 0 0; align-items:center }}
button {{
  font:inherit; font-size:12px; padding:5px 12px; border-radius:7px; cursor:pointer;
  border:1px solid var(--line); background:#fff; color:var(--dim);
}}
button.on {{ background:var(--ink); color:#fff; border-color:var(--ink) }}
section {{
  background:var(--card); border:1px solid var(--line); border-radius:12px;
  padding:16px 20px 20px; margin-bottom:20px; overflow-x:auto;
}}
h2 {{
  margin:0 0 12px; font-size:14px; font-weight:640;
  display:flex; align-items:baseline; gap:12px; flex-wrap:wrap;
}}
.tally {{ font-size:12px; font-weight:500; color:var(--dim) }}
th.bh {{ color:var(--ink); font-weight:650; padding-right:12px }}
tr.grp td, tr.grp th {{ border-top:1px solid var(--line); padding-top:7px }}
tr.brk td, tr.brk th {{ border-top:2px solid #111827; padding-top:5px }}
tr.hl th.rh {{ color:var(--ink); font-weight:700 }}
tr.hl td.cell {{ box-shadow:0 0 0 1px rgba(17,24,39,.34) }}
.tag {{
  font-size:9px; font-weight:600; color:#fff; background:#111827; letter-spacing:.3px;
  border-radius:4px; padding:1.5px 4px; margin-right:7px; vertical-align:1.5px;
}}
body.show-us .cell {{ min-width:84px }}
body.show-us .cell .pill {{ font-size:11px }}
table {{ border-collapse:separate; border-spacing:4px; width:100% }}
th {{ font-weight:600; font-size:12px; color:var(--dim); padding:2px 6px; white-space:nowrap }}
thead th {{ text-align:center; padding-bottom:6px }}
.grouphead {{
  font-weight:500; font-size:11px; color:#9ca3af; padding-bottom:2px;
  border-bottom:1px solid var(--line);
}}
.grouphead.left {{ text-align:right; padding-right:10px }}
th.rh {{ text-align:right; min-width:34px }}
th.rh.sub {{ color:#9ca3af; font-weight:450 }}
td {{ text-align:center; padding:0 }}
.cell {{
  border-radius:8px; cursor:default; min-width:62px; height:30px;
  background:var(--bg); border:1px solid var(--bd);
}}
.cell .pill {{
  font-size:12px; font-weight:660; letter-spacing:.2px; color:var(--fg);
}}
.cell:hover {{ outline:2px solid var(--deep); outline-offset:1px }}
.empty {{ color:#d1d5db; font-size:12px }}
body.show-us .cell .pill::after {{ content:" " attr(data-us) }}
#tip {{
  position:fixed; z-index:50; pointer-events:none; opacity:0; transition:opacity .09s;
  background:#111827; color:#f9fafb; padding:9px 11px; border-radius:8px;
  font:12px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
  white-space:pre; box-shadow:0 8px 22px rgba(0,0,0,.22); max-width:340px;
}}
footer {{ color:var(--dim); font-size:12px; line-height:1.8; padding:0 4px 10px }}
footer p {{ margin:0 0 9px; max-width:1000px }}
footer code {{ background:#eef0f3; padding:1px 5px; border-radius:4px; font-size:11px }}
</style></head>
<body>
<div class="wrap">
<header>
  <h1>GDN Prefill 各参数下最快的 mode</h1>
  <div class="sub">{html.escape(sub)}</div>
  <div class="legend">{legend_html(meta, data)}</div>
  <div class="controls">
    <button id="b-mode" class="on">显示 mode</button>
    <button id="b-us">叠加耗时</button>
    <span style="color:var(--dim);font-size:12px">
      色深 = 领先第二名的幅度（越深越稳），悬停看该格全部 mode 的耗时
    </span>
  </div>
</header>
{"".join(table_html(t, meta, (data["hv_model"] // 8, 1) if t["gqa_ratio"] == 4 else None) for t in data["tables"])}
<footer>
  <p>这是 <b>packed varlen</b> 的排布：一次调用里打包 <b>B</b> 条序列，每条 seqlen 个
  token，总 token 数 T = B × seqlen。每卡 value head 数 <b>H</b> 不是自由参数，它由
  TP 决定（Hv={data["hv_model"]}，所以 TP {"/".join(map(str, tps))} 对应
  H {"/".join(str(data["hv_model"] // t) for t in tps)}），<b>Hg</b> 是相应的 key head 数。<br>
  行按 <b>B·H</b> 递增排 —— 这是 state scan 要跑的 (序列, head) 链条数，也就是它的
  并行度，和本机 {data["cus"]} 个 CU 一比就知道有没有喂饱。胜负只跟着 B·H 走，与 H 和 B
  各自是多少无关：B·H 相同的几行（比如 TP1/B=1 与 TP8/B=8 都是 64，浅线圈在一起）
  结论完全一致，所以<b>整张表只有一条粗线</b>，横穿全部 seqlen 列。
  标了「生产」的一行是 Qwen3.5 TP8 单条请求（Hk=16 / Hv=64 切到每卡 Hg=2 / H=8，B=1）。</p>
  <p>每格取该形状下 wall 最短的那条路径，wall 是 {data["num_iters"]} 次迭代的中位数。
  色深表示领先第二名的幅度：越深越是稳赢，最浅的一档意味着前两名相差不到 2%，
  换一次测量就可能易主。<code>—</code> 表示该形状下没有任何路径跑出结果。</p>
  <p>数据 <code>mode_grid.json</code> 由 <code>sweep_gdn_mode_grid.py</code> 采集，
  本页由 <code>render_gdn_mode_grid.py</code> 渲染，aiter 来自
  <code>{html.escape(data["aiter_path"])}</code>。</p>
</footer>
</div>
<div id="tip"></div>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el => {{
  el.addEventListener('mouseenter', e => {{
    tip.textContent = el.dataset.tip; tip.style.opacity = 1;
  }});
  el.addEventListener('mousemove', e => {{
    const r = tip.getBoundingClientRect();
    let x = e.clientX + 14, y = e.clientY + 14;
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 14;
    tip.style.left = x + 'px'; tip.style.top = y + 'px';
  }});
  el.addEventListener('mouseleave', () => tip.style.opacity = 0);
}});
document.querySelectorAll('.cell').forEach(td => {{
  td.querySelector('.pill').dataset.us = td.dataset.us;
}});
const bm = document.getElementById('b-mode'), bu = document.getElementById('b-us');
bm.onclick = () => {{ document.body.classList.remove('show-us'); bm.classList.add('on'); bu.classList.remove('on'); }};
bu.onclick = () => {{ document.body.classList.add('show-us'); bu.classList.add('on'); bm.classList.remove('on'); }};
</script>
</body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(DEFAULT_JSON))
    ap.add_argument("--out", default=str(DEFAULT_HTML))
    args = ap.parse_args()
    with open(args.json) as fh:
        data = json.load(fh)
    with open(args.out, "w") as fh:
        fh.write(render(data))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
