#!/usr/bin/env python3
"""Build Confluence markdown from results CSVs (tables + graph paste placeholders)."""

from __future__ import annotations

import csv
from pathlib import Path

BENCH = Path(__file__).resolve().parent
RESULTS = BENCH / "results"
ASSETS = BENCH / "confluence_assets"


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ratio(ck: float, jax: float) -> str:
    return f"{jax / ck:.2f}" if ck > 0 else ""


def md_table(rows: list[dict[str, str]]) -> str:
    has_jax = rows and "jax_unfused(ms)" in rows[0] and rows[0].get("jax_unfused(ms)", "").strip()
    header = ["batch", "s_q", "s_kv", "ck_pr_6764 (ms)"]
    if has_jax:
        header.extend(["jax_unfused (ms)", "jax/ck"])
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * len(header)) + " |",
    ]
    for r in rows:
        ck = float(r["ck_pr_6764(ms)"])
        cells = [r["batch"], r["s_q"], r["s_kv"], f"{ck:.3f}"]
        if has_jax:
            jax = float(r["jax_unfused(ms)"])
            cells.extend([f"{jax:.3f}", _ratio(ck, jax)])
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def graph_block(asset_name: str, caption: str) -> str:
    """Heading for screenshot paste (user attaches PNG under this heading in Confluence)."""
    return f"""#### Graph — {caption}

*(Screenshot: paste below. Optional reference PNG: `{ASSETS.name}/{asset_name}` from `python3 gen_confluence_assets.py`.)*

---

"""


def direction_block(
    scen: str,
    kind: str,
    *,
    x_label: str,
    chart_caption: str,
) -> str:
    rows = load(RESULTS / scen / f"{kind}.csv")
    asset = f"{scen}_{kind}.png"
    title = "Forward" if kind == "fwd" else "Backward"
    return f"""### {title}

#### Table — {title.lower()} ({x_label} sweep)

{md_table(rows)}

{graph_block(asset, chart_caption)}
"""


def _find_row(rows: list[dict[str, str]], batch: str, sq: str, sk: str) -> dict[str, str] | None:
    for r in rows:
        if r["batch"] == batch and r["s_q"] == sq and r["s_kv"] == sk:
            return r
    return None


def summary_table() -> str:
    """Auto-filled jax/ck highlights from current CSVs."""
    specs = [
        ("scenario1", "fwd", "2048", "16", "16", "P=16"),
        ("scenario1", "bwd", "2048", "16", "16", "P=16"),
        ("scenario2", "fwd", "2048", "1", "16", "P=16 (sq=1)"),
        ("scenario2", "bwd", "2048", "1", "16", "P=16"),
        ("scenario3_4", "fwd", "2048", "16", "16", "s=16"),
        ("scenario3_4", "bwd", "2048", "16", "16", "s=16"),
    ]
    lines = [
        "| Scenario | Direction | Point | jax/ck (B=2048) | Notes |",
        "|----------|-----------|-------|-----------------|-------|",
    ]
    for scen, kind, batch, sq, sk, point in specs:
        path = RESULTS / scen / f"{kind}.csv"
        if not path.is_file():
            continue
        r = _find_row(load(path), batch, sq, sk)
        if not r or not r.get("jax_unfused(ms)", "").strip():
            continue
        ck, jax = float(r["ck_pr_6764(ms)"]), float(r["jax_unfused(ms)"])
        note = "CK faster" if jax > ck else "JAX faster"
        scen_label = scen.replace("scenario", "").replace("_", "+")
        lines.append(
            f"| {scen_label} | {kind} | {point} | {_ratio(ck, jax)} | {note} |"
        )
    return "\n".join(lines)


def bwd_outlier_table() -> str:
    rows = load(RESULTS / "scenario3_4" / "bwd.csv")
    lines = [
        "| batch | s_q | s_kv | ck_pr_6764 (ms) | jax_unfused (ms) | jax/ck |",
        "|-------|-----|------|-----------------|------------------|--------|",
    ]
    for batch in ("2048", "4096"):
        r = _find_row(rows, batch, "17", "17")
        if not r:
            continue
        ck, jax = float(r["ck_pr_6764(ms)"]), float(r["jax_unfused(ms)"])
        lines.append(
            f"| {batch} | 17 | 17 | **{ck:.3f}** | {jax:.3f} | {_ratio(ck, jax)} |"
        )
    return "\n".join(lines)


def section(
    scen: str,
    title: str,
    script_note: str,
    *,
    x_label: str,
    fwd_chart: str,
    bwd_chart: str,
) -> str:
    return f"""## {title}

{script_note}

{direction_block(scen, "fwd", x_label=x_label, chart_caption=fwd_chart)}

{direction_block(scen, "bwd", x_label=x_label, chart_caption=bwd_chart)}

"""


def main() -> None:
    import sys

    publish = "--publish" in sys.argv
    if publish:
        sys.argv.remove("--publish")
    body = f"""# Benchmark CK_PR_6764

Small-sequence MHA forward/backward: **ck_pr_6764** (CK tile, `fwd_v3=0`, `bwd_v3=0`) vs **jax_unfused** (reference einsum; scenarios 1–2 fwd use THD + `cu_seqlens`, same packing as CK group mode). Times are mean step latency in **ms** (warmup 5, repeat 25).

---

## Test environment

| Item | Value |
|------|--------|
| **Host** | `smci355-ccs-aus-m07-05` (MI355 class system) |
| **Benchmark harness** | [aiter `op_tests/cpp/mha/small_attn_benchmark`](https://github.com/ROCm/aiter/tree/veergopu/check_ck_small_seq/op_tests/cpp/mha/small_attn_benchmark) |
| **CK run** | `cd op_tests/cpp/mha && bash build_mha.sh` then `cd small_attn_benchmark && ./run_all.sh` |
| **JAX run** | `python3 run_jax_benchmark.py all` (after CK CSVs exist; adds `jax_unfused(ms)` column) |
| **Graphs** | Paste screenshots under each **Graph** heading; optional PNGs from `python3 gen_confluence_assets.py` |
| **Results** | `results/scenario*/{{fwd,bwd}}.csv` |

### Composable Kernel (CK)

| Item | Value |
|------|--------|
| **Repository** | [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) |
| **Branch** | [veergopu/add_6764](https://github.com/ROCm/composable_kernel/tree/veergopu/add_6764) |
| **Commit** | [1eafdc8](https://github.com/ROCm/composable_kernel/commit/1eafdc8bd705bdc201563905d505331e39eb17cb) |

### aiter

| Item | Value |
|------|--------|
| **Repository** | [ROCm/aiter](https://github.com/ROCm/aiter) |
| **Branch** | [veergopu/check_ck_small_seq](https://github.com/ROCm/aiter/tree/veergopu/check_ck_small_seq) |
| **Commit** | [88e129697a6d](https://github.com/ROCm/aiter/commit/88e129697a6db9763da5ff5cb60d242c70dfa86e) |

---

## Common configuration

| Parameter | CK | JAX unfused |
|-----------|-----|-------------|
| Precision | bf16 | bf16 |
| Layout | BHSD (`iperm=1`, `operm=1`) | BSHD (einsum); varlen fwd via THD + `cu_seqlens` |
| Heads / dim | h=32, d=128 | same |
| Mask / bias / dropout | non-causal, no bias, p_drop=0 | non-causal |
| Scale | `scale_s=0` → 1/√128 | 1/√128 |
| Batch sizes | 2048, 4096 | same (from CSV rows) |
| Warmup / repeat | 5 / 25 | 5 / 25 |
| Length RNG (scen 1–2 fwd) | seed 6764, i.i.d. uniform {{2,…,P}} per row | same lists |

**Table columns:** `jax/ck` = JAX time ÷ CK time (**>1** means CK is faster).

---

## Customer requests vs what we ran

| # | Customer request | Included? | How we ran it | Gap |
|---|------------------|-----------|---------------|-----|
| **1** | sq, skv ≤ 16, padding + varlen | Partial | Fwd: CK group packed; JAX THD+cu_seqlens. Bwd: uniform batch P. | No physical padding; bwd not per-row varlen. |
| **2** | sq=1, skv ≤ 16 varlen | Partial | Fwd: CK group; JAX cross + THD KV. Bwd: batch s_q=1, s_kv=P. | Same padding/bwd limits. |
| **3** | sq=skv=16 fixed | Yes | Row in scenario_3_4 sweep. | — |
| **4** | sq=skv=17 fixed (priority) | Yes | Row in scenario_3_4 sweep. | CK bwd outlier at s=17 (see below). |

---

"""
    body += section(
        "scenario1",
        "Scenario 1 — sq, skv ≤ 16 (packed varlen Q and KV)",
        "**Script:** `scenario_1.sh` · **CK fwd:** `-mode=1` · **CK bwd:** `-mode=0`, uniform `s_q=s_kv=P` · **JAX fwd:** THD + `cu_seqlens` · **JAX bwd:** dense `(B,P,…)`.",
        x_label="P (max logical length)",
        fwd_chart="Scenario 1 forward — CK vs JAX vs P (batch 2048 & 4096)",
        bwd_chart="Scenario 1 backward — uniform P",
    )
    body += section(
        "scenario2",
        "Scenario 2 — sq = 1, skv ≤ 16 (cross-attn / packed KV)",
        "**Script:** `scenario_2.sh` · **CK fwd:** group, sq=1 · **JAX fwd:** Q `(B,1,H,D)`, KV THD packed.",
        x_label="P (max KV length)",
        fwd_chart="Scenario 2 forward — sq=1, CK vs JAX vs P",
        bwd_chart="Scenario 2 backward — s_q=1, s_kv=P",
    )
    body += section(
        "scenario3_4",
        "Scenarios 3 & 4 — fixed self-attention (sq = skv, 2…17)",
        "**Script:** `scenario_3_4.sh` · Customer **#3** = s=16, **#4** = s=17 rows in tables below.",
        x_label="seq len (sq = skv)",
        fwd_chart="Scenario 3+4 forward — fixed self-attn vs seq len",
        bwd_chart="Scenario 3+4 backward — fixed self-attn vs seq len",
    )
    body += f"""---

## Summary (CK vs JAX)

{summary_table()}

### CK bwd outlier (scenario 3+4, seq = 17)

{bwd_outlier_table()}

At seq=17 backward, JAX is faster than CK — opposite of other rows. Worth CK/asm follow-up (4×4 vs 16×16 tile paths).

---

## Reproduce

```bash
cd op_tests/cpp/mha && bash build_mha.sh
cd small_attn_benchmark
./run_all.sh
python3 run_jax_benchmark.py all
python3 gen_confluence_assets.py   # PNGs for Confluence paste
python3 build_confluence_md.py     # refresh this page
```
"""
    out = BENCH / "confluence_page.md"
    out.write_text(body, encoding="utf-8")
    print(f"wrote {out} ({len(body)} chars)")
    if publish:
        print("To update Confluence: use MCP confluence_update_page on page 1690335375")
        print("  or paste confluence_page.md into https://amd.atlassian.net/wiki/spaces/~veergopu/pages/1690335375")


if __name__ == "__main__":
    main()
