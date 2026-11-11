#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Turn one DSR1 combo run directory into a static report.html.

Usage: python3 report.py <run_dir>

<run_dir> holds combo.log (bench_gfx1250_combo.py --dsr1 --json), plus the
provenance.txt and status.txt that run.sh writes. Standard library only.
"""

import html
import json
import math
import sys
from pathlib import Path

OPS = [
    {
        "key": "flydsl_grouped_gemm",
        "name": "MoE FC1 + FC2",
        "op": "moe",
        "shape": (
            "Per GPU: 65 experts (256 routed / EP4 = 64, plus the fused shared "
            "expert), top-9 (8 routed + the shared expert on every token). "
            "FC1: N=4096 (gate/up interleaved), K=7168, SiLU and MXFP4 requant "
            "in the epilogue; FC2: N=7168, K=2048. 9 x bs routed rows."
        ),
        "dtype": "a4w4 MXFP4 (fp4 A and W, e8m0 scale per 32), bf16 out",
        "ut": (
            "op_tests/flydsl_tests/test_flydsl_grouped_gemm.py: run_moe('a4w4', "
            "experts=65, topk=9, model_dim=7168, inter_dim=2048, "
            "activation=Silu, kernel_bench=True)"
        ),
        "how": (
            "gemm1 and gemm2 are timed separately; the activation quant kernel "
            "before gemm1 is not included. Plain grouped GEMM: the S1F "
            "(dispatch + GEMM1) fusion exists only inside MegaMoE."
        ),
    },
    {
        "key": "mla decode",
        "name": "MLA decode (absorbed)",
        "op": "mla_decode",
        "shape": (
            "128 heads, one query per sequence. Q [bs,128,576] (512 latent + 64 "
            "RoPE); KV 576 B per token shared by all heads; out [bs,128,512]. "
            "Cases with batch x ctx above 32M KV tokens are skipped."
        ),
        "dtype": "Q fp8, KV fp8 (page 64, kv-shuffled), bf16 out",
        "ut": (
            "op_tests/test_mla_decode_pagesize64.py: test_mla(bs, ctx, 128, 1, 1, "
            "1, fp8, fp8, 'randn', 'kv_shuffled') -> aiter.mla.mla_decode_fwd"
        ),
        "how": (
            "The kernel ATOM runs with ATOM_MLA_KV_SHUFFLED=1, checked against "
            "the UT's fp32 reference (err column)."
        ),
    },
    {
        "key": "mla prefill",
        "name": "MLA prefill (MHA)",
        "op": "mla_prefill",
        "shape": (
            "Causal varlen FMHA, 128 heads, Q/K [T,128,192] (128 nope + 64 "
            "RoPE), V [T,128,128]; T tokens packed from equal-length sequences."
        ),
        "dtype": "bf16",
        "ut": (
            "op_tests/test_fmha_fwd_with_sink_varlen_asm.py: "
            "test_fmha_fwd_with_sink_varlen_asm_perf(192, 128, 128, 128, "
            "seqlens, True, 'randn')"
        ),
        "how": (
            "asm is fmha_fwd_with_sink_varlen_asm; public is flash_attn_varlen_func, "
            "the call ATOM makes (the asm kernel plus one torch copy). DSR1 prefill "
            "never takes FlyDSL, so the UT's FlyDSL candidate is switched off."
        ),
    },
]

NOTES = [
    (
        "All numbers are single-GPU kernel times. MoE GB/s is the UT's own "
        "effective-bandwidth model, reported as is."
    ),
    (
        "Shapes match deepseek-ai/DeepSeek-R1 config.json. Differences: the UTs "
        "skip the YaRN mscale^2 in the softmax scale (no effect on time), and "
        "MoE uses the MLPerf Quark MXFP4 weights instead of FP8 128x128 blocks."
    ),
    "Model level: 58 MoE layers (first 3 of 61 are dense) and 61 MLA layers.",
]

CSS = """
body{font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;margin:0;
background:#f9f9f7;color:#0b0b0b}
.wrap{max-width:1180px;margin:0 auto;padding:24px 16px 48px}
h1{font-size:22px;margin:0 0 6px}h2{font-size:17px;margin:28px 0 8px}
h3{font-size:14px;margin:18px 0 6px}.muted{color:#6b6a66;font-size:13px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:4px 0 8px}
th,td{text-align:left;padding:4px 8px;border-bottom:1px solid #e1e0d9;
vertical-align:top}th{color:#52514e;white-space:nowrap}
.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.m{font-family:ui-monospace,Consolas,monospace;font-size:12px;
word-break:break-all}.c{font-family:ui-monospace,Consolas,monospace;
font-size:12px;white-space:nowrap}.scroll{overflow-x:auto}
code{font-family:ui-monospace,Consolas,monospace;font-size:12.5px;
background:#f0efec;padding:2px 4px;border-radius:4px}
summary{cursor:pointer;color:#52514e}
"""


def esc(v):
    return html.escape("" if v is None else str(v))


def num(v, d=1):
    if not isinstance(v, (int, float)) or math.isnan(v):
        return "-"
    return f"{v:,.{d}f}"


def table(cols, rows):
    """cols: [(header, getter, kind)]; kind is a CSS class: n, m, c or ""."""
    head = "".join(f'<th class="{k}">{esc(h)}</th>' for h, _, k in cols)
    body = "".join(
        "<tr>"
        + "".join(f'<td class="{k}">{esc(g(r))}</td>' for _, g, k in cols)
        + "</tr>"
        for r in rows
    )
    return f'<div class="scroll"><table><tr>{head}</tr>{body}</table></div>'


def read_kv(path):
    out = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def read_tables(path):
    tables = {}
    for line in path.read_text(errors="replace").splitlines():
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(t, dict) and "name" in t and "rows" in t:
            tables[t["name"]] = t["rows"]
    return tables


def find(tables, key):
    return next((rows for name, rows in tables.items() if name.startswith(key)), [])


def kernels(rows, key, label):
    seen = {}
    for r in rows:
        for k in str(r.get(key) or "").split(" + "):
            if k:
                seen.setdefault(k, []).append(label(r))
    return [(k, v if len(v) < len(rows) else ["all"]) for k, v in seen.items()]


def render(run_dir):
    prov = read_kv(run_dir / "provenance.txt")
    status = read_kv(run_dir / "status.txt")
    tables = read_tables(run_dir / "combo.log")
    moe = sorted(find(tables, OPS[0]["key"]), key=lambda r: r["token"])
    dec = sorted(find(tables, OPS[1]["key"]), key=lambda r: (r["batch"], r["ctx_len"]))
    pre = find(tables, OPS[2]["key"])
    errors = sum(1 for rows in tables.values() for r in rows if r.get("err_msg"))
    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>DSR1 MLPerf microbench</title><style>{CSS}</style></head><body>",
        "<div class='wrap'><h1>DSR1 MLPerf microbench</h1>",
        (
            "<div class='muted'>MoE FC1/FC2 and MLA attention at DeepSeek-R1 "
            "MLPerf shapes, one GPU of TP4 + DP attention + EP4. "
            f"Run {esc(run_dir.name)}, {esc(prov.get('date_utc'))}, exit code "
            f"{esc(status.get('rc'))}, {errors} failed case(s).</div>"
        ),
    ]

    # 1. Operators
    ranges = {
        "moe": f"bs {moe[0]['token']}..{moe[-1]['token']}" if moe else "-",
        "mla_decode": (
            f"bs {dec[0]['batch']}..{dec[-1]['batch']} x ctx "
            + ", ".join(str(c) for c in sorted({r["ctx_len"] for r in dec}))
            if dec
            else "-"
        ),
        "mla_prefill": (
            "T " + ", ".join(str(r["tokens"]) for r in pre) if pre else "-"
        ),
    }
    counts = {"moe": len(moe), "mla_decode": len(dec), "mla_prefill": len(pre)}
    out.append("<h2>1. Operators</h2>")
    out.append(
        table(
            [
                ("Operator", lambda o: o["name"], ""),
                ("--ops", lambda o: o["op"], "c"),
                ("Shape (per GPU)", lambda o: o["shape"], ""),
                ("Precision", lambda o: o["dtype"], ""),
                ("Range", lambda o: ranges[o["op"]], ""),
                ("Cases", lambda o: counts[o["op"]], "n"),
            ],
            OPS,
        )
    )
    krows = (
        [("MoE", *k) for k in kernels(moe, "kernel", lambda r: f"bs {r['token']}")]
        + [("MLA decode", *k) for k in kernels(dec, "kernel", lambda r: "")]
        + [
            (f"MLA prefill {b}", *k)
            for b in ("asm", "public", "flydsl")
            for k in kernels(pre, f"{b} kernel", lambda r: "")
        ]
    )
    out.append("<h3>Kernels</h3>")
    out.append(
        table(
            [
                ("Operator", lambda r: r[0], ""),
                ("Kernel", lambda r: r[1], "m"),
                ("Used by", lambda r: ", ".join(r[2]), ""),
            ],
            krows,
        )
    )

    # 2. How to run
    out.append("<h2>2. How to run</h2>")
    out.append(
        "<p>From the aiter root: <code>bash .claude/skills/dsr1-microbench/run.sh "
        "&lt;out_dir&gt; [gpu]</code>, which runs the command below and writes "
        "this report. <code>--ops</code> picks operators and "
        "<code>AITER_BENCH_TOKENS</code> overrides the bs / T sweep.</p>"
    )
    out.append(f"<p><code>{esc(prov.get('cmd', '-'))}</code></p>")
    out.append(
        table(
            [("Item", lambda kv: kv[0], ""), ("Value", lambda kv: kv[1], "m")],
            [(k, v) for k, v in prov.items() if k != "cmd"],
        )
    )
    out.append(
        table(
            [
                ("Operator", lambda o: o["name"], ""),
                ("UT call", lambda o: o["ut"], "m"),
                ("Notes", lambda o: o["how"], ""),
            ],
            OPS,
        )
    )

    # 3. Performance
    out.append("<h2>3. Performance per case</h2>")
    out.append(f"<h3>MoE FC1 + FC2 ({len(moe)} cases)</h3>")
    out.append(
        table(
            [
                ("bs", lambda r: f"{r['token']:,}", "n"),
                ("FC1 us", lambda r: num(r.get("gemm1_us")), "n"),
                ("FC1 TFLOPS", lambda r: num(r.get("gemm1 TFLOPS")), "n"),
                ("FC1 GB/s", lambda r: num(r.get("gemm1 GB/s"), 0), "n"),
                ("FC2 us", lambda r: num(r.get("gemm2_us")), "n"),
                ("FC2 TFLOPS", lambda r: num(r.get("gemm2 TFLOPS")), "n"),
                ("FC2 GB/s", lambda r: num(r.get("gemm2 GB/s"), 0), "n"),
                ("Check", lambda r: "pass" if r.get("pass") else "FAIL", ""),
                ("Error", lambda r: r.get("err_msg", ""), ""),
            ],
            moe,
        )
    )

    out.append(f"<h3>MLA decode ({len(dec)} cases): time in us, rows = batch, ")
    out.append("columns = context length; '-' = not run</h3>")
    ctxs = sorted({r["ctx_len"] for r in dec})
    by = {(r["batch"], r["ctx_len"]): r for r in dec}
    grid = [
        [b] + [by.get((b, c)) for c in ctxs] for b in sorted({r["batch"] for r in dec})
    ]
    out.append(
        table(
            [("batch \\ ctx", lambda g: f"{g[0]:,}", "n")]
            + [
                (f"{c:,}", lambda g, i=i: num((g[i + 1] or {}).get("asm us")), "n")
                for i, c in enumerate(ctxs)
            ],
            grid,
        )
    )
    out.append("<details><summary>All cases with TFLOPS, TB/s, err</summary>")
    out.append(
        table(
            [
                ("batch", lambda r: f"{r['batch']:,}", "n"),
                ("ctx", lambda r: f"{r['ctx_len']:,}", "n"),
                ("us", lambda r: num(r.get("asm us")), "n"),
                ("TFLOPS", lambda r: num(r.get("asm TFLOPS")), "n"),
                ("TB/s", lambda r: num(r.get("asm TB/s"), 2), "n"),
                ("err", lambda r: r.get("asm err"), "n"),
                ("Error", lambda r: r.get("err_msg", ""), ""),
            ],
            dec,
        )
    )
    out.append("</details>")

    out.append(f"<h3>MLA prefill ({len(pre)} cases)</h3>")
    cols = [
        ("T", lambda r: f"{r['tokens']:,}", "n"),
        ("seqs", lambda r: r.get("num_seqs"), "n"),
    ]
    if any("seq_len" in r for r in pre):
        cols = [
            ("sweep", lambda r: r.get("scenario"), ""),
            ("seq len", lambda r: f"{r['seq_len']:,}", "n"),
        ] + cols
    backends = [("asm", "asm"), ("public", "public"), ("flydsl", "FlyDSL")]
    for b, label in [x for x in backends if any(f"{x[0]} us" in r for r in pre)]:
        cols += [
            (f"{label} us", lambda r, b=b: num(r.get(f"{b} us")), "n"),
            (f"{label} TFLOPS", lambda r, b=b: num(r.get(f"{b} TFLOPS")), "n"),
        ]
    cols.append(("asm TB/s", lambda r: num(r.get("asm TB/s"), 2), "n"))
    cols.append(("Error", lambda r: r.get("err_msg", ""), ""))
    out.append(table(cols, pre))

    out.append("<h2>Notes</h2><ul>")
    out.extend(f"<li>{esc(n)}</li>" for n in NOTES)
    out.append("</ul></div></body></html>")
    return "\n".join(out)


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    run_dir = Path(sys.argv[1])
    report = run_dir / "report.html"
    report.write_text(render(run_dir), encoding="utf-8")
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
