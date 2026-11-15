#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Turn one combo run directory into a static report.html.

Usage: python3 report.py <run_dir>

<run_dir> holds combo.log (bench_gfx1250_combo.py --<suite> --json), plus the
provenance.txt and status.txt that run.sh writes. DSR1 gets its own layout;
any other suite lists every table combo.log holds. Standard library only.
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
        "heads": "-",
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
        "heads": "128 query heads, 1 KV head (the 576-wide latent all heads share)",
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
            "Causal (is_causal=True) varlen FMHA, Q/K [T,128,192] (128 nope + 64 "
            "RoPE), V [T,128,128]; T tokens packed from equal-length sequences."
        ),
        "dtype": "bf16",
        "heads": "128 query heads, 128 KV heads (MHA after the kv_b_proj expand)",
        "ut": (
            "op_tests/test_fmha_fwd_with_sink_varlen_asm.py: "
            "test_fmha_fwd_with_sink_varlen_asm_perf(192, 128, 128, 128, "
            "seqlens, True, 'randn')"
        ),
        "how": (
            "Times are the asm kernel ATOM reaches through flash_attn_varlen_func. "
            "DSR1 prefill never takes FlyDSL, so the UT's FlyDSL candidate is "
            "switched off. "
            "ATOM calls it with causal=True; causal=False is only on the "
            "chunked-context paths, which the recipe (prefix caching and chunked "
            "prefill off) never reaches."
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
    (
        "Heads are per GPU: the recipe runs attention data-parallel (TP4 + DP "
        "attention), so every GPU computes all 128 heads for its own requests."
    ),
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


def suite_of(prov):
    if prov.get("suite"):
        return prov["suite"]
    cmd = prov.get("cmd", "")
    return next(
        (w[2:] for w in cmd.split() if w.startswith("--") and w != "--json"), ""
    )


def table_list(path):
    """Every JSON table in combo.log, in order, duplicates kept."""
    out = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            t = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(t, dict) and "name" in t and isinstance(t.get("rows"), list):
            out.append((t["name"], t["rows"]))
    return out


def cell(v):
    if isinstance(v, bool) or v is None:
        return "-" if v is None else str(v)
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return num(v, 2 if abs(v) < 100 else 1)
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x) for x in v)
    return v


# Operator notes for suites other than dsr1; "key" matches table-name prefixes.
SUITE_OPS = {
    "dsv41": [
        {
            "key": "flydsl_grouped_gemm",
            "name": "Grouped MoE FC1 + FC2",
            "op": "moe",
            "shape": (
                "96 experts a rank (384 routed / EP4; the shared expert is not "
                "fused), top-6. FC1 N=4608 (gate/up), K=5120; FC2 N=5120, "
                "K=2304. Rows = tokens x 6."
            ),
            "dtype": "a4w4 MXFP4 (fp4 A and W, e8m0 scale per 32), bf16 out",
            "ut": (
                "op_tests/flydsl_tests/test_flydsl_grouped_gemm.py: run_moe('a4w4', "
                "experts=96, topk=6, model_dim=5120, inter_dim=2304, "
                "kernel_bench=True)"
            ),
        },
        {
            "key": "mega_moe",
            "name": "Mega MoE (dispatch + MoE + combine)",
            "op": "mega_moe",
            "shape": (
                "EP4 over 4 GPUs: 384 experts, top-6, hidden 5120, inter 2304, "
                "40 layers. base+Mega and the recipe's S1F+Mega on an fp4 wire."
            ),
            "dtype": "a4w4_mxfp4",
            "ut": "op_tests/multigpu_tests/bench_mega_moe.py under torchrun x4",
        },
        {
            "key": "mla_v4 decode",
            "name": "Sparse MLA decode",
            "op": "mla_v4_decode",
            "shape": (
                "64 query heads (DP attention), head dim 512, one KV head. "
                "KV pool = compressed history (1K..64K tokens at ratio 2 or 1, "
                "512..65536 rows); each token reads its top-512 plus a 128-token "
                "window."
            ),
            "dtype": "bf16 (opus, triton); v41_fp4 (asm)",
            "ut": "op_tests/test_mla_sparse_prefill.py",
        },
        {
            "key": "mla_v4 prefill",
            "name": "Sparse MLA prefill",
            "op": "mla_v4_prefill",
            "shape": (
                "64 query heads, head dim 512. Same KV pools as decode (512..65536 "
                "rows); each token reads its top-512 plus a 128-token window."
            ),
            "dtype": "bf16 (opus, triton); v41_fp4 (asm)",
            "ut": "op_tests/test_mla_sparse_prefill.py",
        },
        {
            "key": "score_qk_fp4",
            "name": "Indexer score (paged MQA logits)",
            "op": "score_qk_fp4",
            "shape": (
                "32 index heads, dim 128. Full layers (ratio 2) and candidate "
                "layers; prefill 16384 rows, decode batch 1024 at MTP 1 and 6."
            ),
            "dtype": "MXFP4 Q and K (OPUS), fp32 logits",
            "ut": "op_tests/test_pa_mqa_logits_mxfp4_opus.py (with ROCm/aiter#7925)",
        },
        {
            "key": "topk",
            "name": "Indexer top-k",
            "op": "topk",
            "shape": (
                "k=512, stable, indices only. Prefill rows of causal length; "
                "decode batch x 6 rows over 16384 (candidate) and 20480 (full "
                "layer) columns. Rows are 2-D, not packed as ATOM passes them."
            ),
            "dtype": "fp32 scores",
            "ut": "op_tests/test_topk_per_row.py: test_top_k_per_row_prefill / _decode",
        },
    ],
}


def size_range(rows):
    """'<key> lo..hi' for the first size column the rows carry."""
    for key in ("token", "batch", "n", "rows", "tokens"):
        vals = sorted({r[key] for r in rows if isinstance(r.get(key), int)})
        if len(vals) > 1:
            return f"{key} {vals[0]:,}..{vals[-1]:,}"
        if vals:
            return f"{key} {vals[0]:,}"
    return "-"


def op_range(op_tables):
    """Each matching table's size range, duplicates dropped."""
    seen = []
    for _, rows in op_tables:
        r = size_range(rows)
        if r not in seen:
            seen.append(r)
    return "; ".join(seen) if seen else "not in this run"


def kind(key, rows):
    """CSS class for a generic column: monospace text or right-aligned numbers."""
    if "kernel" in key or key in ("err_msg", "output_tail"):
        return "m"
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals)
    return "n" if vals and numeric else ""


def render_generic(run_dir, suite):
    prov = read_kv(run_dir / "provenance.txt")
    status = read_kv(run_dir / "status.txt")
    tables = table_list(run_dir / "combo.log")
    failed = [(n, r) for n, rows in tables for r in rows if r.get("err_msg")]
    title = f"{suite.upper() or 'combo'} microbench"
    out = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        f"<title>{esc(title)}</title><style>{CSS}</style></head><body>",
        f"<div class='wrap'><h1>{esc(title)}</h1>",
        (
            f"<div class='muted'>Run {esc(run_dir.name)}, "
            f"{esc(prov.get('date_utc'))}, exit code {esc(status.get('rc'))}, "
            f"{len(tables)} table(s), {len(failed)} failed case(s).</div>"
        ),
    ]
    meta = SUITE_OPS.get(suite)
    if meta:
        ops = []
        for o in meta:
            mine = [(n, rs) for n, rs in tables if n.startswith(o["key"])]
            rows = [r for _, rs in mine for r in rs]
            ops.append({**o, "rows": rows, "range": op_range(mine)})
        out += [
            "<h2>1. Operators</h2>",
            table(
                [
                    ("Operator", lambda o: o["name"], ""),
                    ("--ops", lambda o: o["op"], "c"),
                    ("Shape (per GPU)", lambda o: o["shape"], ""),
                    ("Precision", lambda o: o["dtype"], ""),
                    ("Range", lambda o: o["range"], ""),
                    ("Cases", lambda o: len(o["rows"]), "n"),
                    (
                        "Failed",
                        lambda o: sum(1 for r in o["rows"] if r.get("err_msg")),
                        "n",
                    ),
                    ("UT", lambda o: o["ut"], "m"),
                ],
                ops,
            ),
        ]
    else:
        out += [
            "<h2>1. Tables</h2>",
            table(
                [
                    ("#", lambda t: t[0], "n"),
                    ("Table", lambda t: t[1], ""),
                    ("Cases", lambda t: len(t[2]), "n"),
                    ("Failed", lambda t: sum(1 for r in t[2] if r.get("err_msg")), "n"),
                ],
                [(i + 1, n, rows) for i, (n, rows) in enumerate(tables)],
            ),
        ]
    out += [
        "<h2>2. How to run</h2>",
        (
            "<p>From the aiter root: <code>bash .claude/skills/run-bench/run.sh "
            f"{esc(suite)} &lt;out_dir&gt; [gpus]</code>, which runs the command "
            "below and writes this report.</p>"
        ),
        f"<p><code>{esc(prov.get('cmd', '-'))}</code></p>",
        table(
            [("Item", lambda kv: kv[0], ""), ("Value", lambda kv: kv[1], "m")],
            [(k, v) for k, v in prov.items() if k != "cmd"],
        ),
        "<h2>3. Performance per case</h2>",
    ]
    for i, (name, rows) in enumerate(tables):
        cols = []
        for r in rows:
            cols += [k for k in r if k not in cols]
        kcols = [k for k in cols if "kernel" in k]
        fixed = [
            k
            for k in cols
            if k not in kcols
            and len(rows) > 1
            and len({json.dumps(r.get(k), sort_keys=True) for r in rows}) == 1
        ]
        shown = [k for k in cols if k not in kcols and k not in fixed]
        out.append(f"<h3>{i + 1}. {esc(name)} ({len(rows)} cases)</h3>")
        if fixed:
            same = ", ".join(f"{k}={cell(rows[0].get(k))}" for k in fixed)
            out.append(f"<div class='muted'>Same for every case: {esc(same)}</div>")
        out.append(
            table(
                [(k, lambda r, k=k: cell(r.get(k)), kind(k, rows)) for k in shown],
                rows,
            )
        )
        size = next(
            (k for k in ("token", "batch", "n", "rows", "tokens") if k in cols), None
        )
        krows = [
            (k, kern, used)
            for k in kcols
            for kern, used in kernels(
                rows, k, lambda r, size=size: r.get(size, "") if size else ""
            )
        ]
        if krows:
            kcol = [("Column", lambda r: r[0], "c")] if len(kcols) > 1 else []
            out.append(
                table(
                    [("Kernel", lambda r: r[1], "m")]
                    + kcol
                    + [
                        (
                            f"Used at {size or 'case'}",
                            lambda r: ", ".join(str(u) for u in dict.fromkeys(r[2])),
                            "",
                        ),
                    ],
                    krows,
                )
            )
    out.append("</div></body></html>")
    return "\n".join(out)


def render(run_dir):
    suite = suite_of(read_kv(run_dir / "provenance.txt"))
    if suite != "dsr1":
        return render_generic(run_dir, suite)
    prov = read_kv(run_dir / "provenance.txt")
    status = read_kv(run_dir / "status.txt")
    tables = read_tables(run_dir / "combo.log")
    moe = sorted(find(tables, OPS[0]["key"]), key=lambda r: r["token"])
    dec = sorted(find(tables, OPS[1]["key"]), key=lambda r: (r["batch"], r["ctx_len"]))
    pre = sorted(
        find(tables, OPS[2]["key"]),
        key=lambda r: (r.get("seq_len", 0), r["tokens"]),
    )
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
            "seq len "
            + ", ".join(str(v) for v in sorted({r.get("seq_len", 1024) for r in pre}))
            + "; T "
            + ", ".join(str(v) for v in sorted({r["tokens"] for r in pre}))
            if pre
            else "-"
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
                ("Heads (per GPU)", lambda o: o["heads"], ""),
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
        + [("MLA prefill", *k) for k in kernels(pre, "asm kernel", lambda r: "")]
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
        "<p>From the aiter root: <code>bash .claude/skills/run-bench/run.sh dsr1 "
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

    q_heads = ", ".join(str(h) for h in sorted({r.get("nhead") for r in dec}))
    out.append(f"<h3>MLA decode ({len(dec)} cases, {q_heads} query heads): ")
    out.append("time in us, rows = batch, columns = context length; '-' = not run</h3>")
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
                ("q heads", lambda r: r.get("nhead"), "n"),
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
        ("q / kv heads", lambda r: f"{r.get('hq')} / {r.get('hk')}", "n"),
        ("causal", lambda r: r.get("is_causal"), ""),
    ]
    if any("seq_len" in r for r in pre):
        cols = [("seq len", lambda r: f"{r['seq_len']:,}", "n")] + cols
    cols += [
        ("us", lambda r: num(r.get("asm us")), "n"),
        ("TFLOPS", lambda r: num(r.get("asm TFLOPS")), "n"),
        ("TB/s", lambda r: num(r.get("asm TB/s"), 2), "n"),
        ("Error", lambda r: r.get("err_msg", ""), ""),
    ]
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
