"""Render matrix_scan/perf_matrix.md from the matrix_* and splits_* JSONs."""
import json, os, glob, math

BASE = os.environ.get("SCAN_DIR", "/app/aiter/bench_gluon_ua/matrix_scan")
VERS = ["3.6.0", "3.7.0", "3.8.0"]
HSS = [256, 128, 64]
HEADCFG = [(8, 1, "MQA 8/1"), (64, 8, "GQA-8 64/8"), (8, 8, "MHA 8/8"), (64, 64, "MHA 64/64")]

M = {}
for hs in HSS:
    for v in VERS:
        p = f"{BASE}/matrix_hs{hs}_{v}.json"
        if os.path.exists(p):
            M[(hs, v)] = json.load(open(p))
HAVE = sorted(M.keys(), key=lambda x: (-x[0], x[1]))
VERS_OK = [v for v in VERS if any(k[1] == v for k in M)]


def key(r):
    return (r["phase"], r["N"], r["L"], r["Hq"], r["Hkv"], r["dtype"])


def idx(hs, v):
    return {key(r): r for r in M[(hs, v)] if "skipped" not in r}


def elem(r):
    return 1 if r["dtype"] == "fp8" else 2


def bytes_of(r):
    """KV read + Q read + O write. Output is bf16 regardless of the input dtype."""
    hs, e = r["head_size"], elem(r)
    tokens = r["N"] * (1 if r["phase"] == "decode" else r["L"])
    kv = 2 * r["N"] * r["L"] * r["Hkv"] * hs * e
    return kv + tokens * r["Hq"] * hs * e + tokens * r["Hq"] * hs * 2


def flops_of(r):
    """Causal attention: QK + PV = 4 * Hq * HEAD_SIZE MACs per (query, key) pair."""
    hs = r["head_size"]
    if r["phase"] == "decode":
        return 4 * r["N"] * r["Hq"] * hs * r["L"]
    pairs = r["L"] * (r["L"] + 1) / 2
    return r["N"] * 4 * r["Hq"] * hs * pairs


def tput(r, impl):
    """Decode -> TB/s (memory bound). Prefill -> TFLOP/s (compute bound)."""
    x = r.get(impl, {})
    if "us" not in x:
        return None
    sec = x["us"] / 1e6
    if r["phase"] == "decode":
        return bytes_of(r) / sec / 1e9        # GB/s
    return flops_of(r) / sec / 1e12           # TFLOP/s


def sp(r):
    t, g = r.get("triton", {}), r.get("gluon", {})
    if "error" in t or "error" in g or not t or not g:
        return None
    return t["us"] / g["us"]


L = [
    "# gfx950 unified attention — Triton vs Gluon across Triton 3.6 / 3.7 / 3.8", "",
    "Full matrix scan: both implementations driven through their **public wrappers**, so "
    "each picks its own configuration exactly as production would "
    "(`aiter.ops.triton.attention.unified_attention` vs the gfx950 gluon "
    "`unified_attention`).", "",
    "| axis | values |", "|---|:--|",
    "| Triton | " + ", ".join(VERS_OK) + " |",
    "| head config (Hq/Hkv) | 8/1 (MQA), 64/8 (GQA-8), 8/8 (MHA), 64/64 (MHA wide) |",
    "| HEAD_SIZE | " + ", ".join(str(h) for h in sorted({k[0] for k in HAVE}, reverse=True)) + " |",
    "| dtype | bf16, fp8 (q/k/v all e4m3, output bf16) |",
    "| decode | C ∈ {1, 8, 32, 64, 128} × ctx ∈ {1024, 8192} |",
    "| prefill | (B,N) ∈ {(1,1024), (4,1024), (8,1024), (1,8192)} |",
    "", "**Method.** Time is attention **+ reduce**, taken from `torch.profiler` and "
    "filtered by kernel name. That filtering matters: the harness flushes 512 MB of L2 "
    "between iterations and that memset is ~93 µs, several times larger than the smaller "
    "decode cells — timing it by wall clock flattens every ratio toward 1.0. 8 warmup + 30 "
    "timed iterations, gfx950 / 256 CU, page size = TILE_SIZE = 64. Every cell is checked "
    "against an fp32 torch reference (one sequence, the causal-hardest row); a cell that "
    "fails is reported rather than silently timed.", "",
    "`speedup` is **triton / gluon**, so >1 means gluon is faster.", "",
]

# ---------------------------------------------------------------- headline
L += ["## 1. Headline", ""]
rows = []
for (hs, v) in HAVE:
    for phase in ("decode", "prefill"):
        vals = [sp(r) for r in M[(hs, v)]
                if "skipped" not in r and r["phase"] == phase and sp(r)]
        if not vals:
            continue
        vals.sort()
        geo = math.exp(sum(math.log(x) for x in vals) / len(vals))
        rows.append((hs, v, phase, len(vals), geo, vals[0], vals[-1],
                     sum(1 for x in vals if x > 1.0)))
L += ["| HEAD_SIZE | triton | phase | cells | geomean speedup | min | max | gluon wins |",
      "|--:|:--|:--|--:|--:|--:|--:|--:|"]
for hs, v, phase, n, geo, lo, hi, w in rows:
    L.append(f"| {hs} | {v} | {phase} | {n} | **{geo:.2f}×** | {lo:.2f}× | {hi:.2f}× | "
             f"{w}/{n} |")
L.append("")

# ---------------------------------------------------------------- per head config
L += ["## 2. By head configuration", "",
      "Geomean speedup (triton/gluon), decode and prefill separately.", ""]
for hs in sorted({k[0] for k in HAVE}, reverse=True):
    vs = [v for v in VERS_OK if (hs, v) in M]
    if not vs:
        continue
    L += [f"### HEAD_SIZE = {hs}", "",
          "| heads | dtype | phase | " + " | ".join(vs) + " |",
          "|---|:--|:--|" + "--:|" * len(vs)]
    for (Hq, Hkv, hlabel) in HEADCFG:
        for dt in ("bf16", "fp8"):
            for phase in ("decode", "prefill"):
                cells = []
                for v in vs:
                    vals = [sp(r) for r in M[(hs, v)]
                            if "skipped" not in r and r["phase"] == phase
                            and r["Hq"] == Hq and r["Hkv"] == Hkv and r["dtype"] == dt
                            and sp(r)]
                    if not vals:
                        cells.append("–")
                    else:
                        geo = math.exp(sum(math.log(x) for x in vals) / len(vals))
                        cells.append(f"{'**' if geo > 1 else ''}{geo:.2f}×"
                                     f"{'**' if geo > 1 else ''}")
                L.append(f"| {hlabel} | {dt} | {phase} | " + " | ".join(cells) + " |")
    L.append("")

# ---------------------------------------------------------------- throughput
L += ["## 2b. Achieved throughput", "",
      "Decode is memory bound, so the metric is **GB/s** against a measured streaming-read "
      "ceiling of ~6500 GB/s on this card (a plain read kernel; the 8 TB/s HBM3E spec is "
      "not reachable in practice). Prefill is compute bound, so the metric is **TFLOP/s**, "
      "counting causal QK+PV as `4 * Hq * HEAD_SIZE` MACs per (query, key) pair. Bytes "
      "count the KV read plus the Q read and O write.", "",
      "Peak and median over each group — `triton / gluon`.", ""]
for hs in sorted({k[0] for k in HAVE}, reverse=True):
    vs = [v for v in VERS_OK if (hs, v) in M]
    if not vs:
        continue
    L += [f"### HEAD_SIZE = {hs}", "",
          "| phase | dtype | metric | " + " | ".join(f"{v} peak" for v in vs) + " | "
          + " | ".join(f"{v} median" for v in vs) + " |",
          "|---|:--|:--|" + "--:|" * (2 * len(vs))]
    for phase in ("decode", "prefill"):
        unit = "GB/s" if phase == "decode" else "TFLOP/s"
        for dt in ("bf16", "fp8"):
            peaks, meds = [], []
            for v in vs:
                pairs = [(tput(r, "triton"), tput(r, "gluon")) for r in M[(hs, v)]
                         if "skipped" not in r and r["phase"] == phase
                         and r["dtype"] == dt and tput(r, "gluon")]
                if not pairs:
                    peaks.append("–"); meds.append("–"); continue
                tt = sorted(x[0] for x in pairs if x[0])
                gg = sorted(x[1] for x in pairs if x[1])
                f = "{:.0f}"
                peaks.append(f.format(tt[-1]) + " / **" + f.format(gg[-1]) + "**")
                meds.append(f.format(tt[len(tt)//2]) + " / " + f.format(gg[len(gg)//2]))
            L.append(f"| {phase} | {dt} | {unit} | " + " | ".join(peaks) + " | "
                     + " | ".join(meds) + " |")
    L.append("")

# ---------------------------------------------------------------- version effect
L += ["## 3. Triton version effect", "",
      "Same shape, same implementation, different compiler — time relative to 3.6.0 "
      "(>1 = slower than 3.6.0).", ""]
for hs in sorted({k[0] for k in HAVE}, reverse=True):
    vs = [v for v in VERS_OK if (hs, v) in M and v != "3.6.0"]
    if "3.6.0" not in [v for v in VERS_OK if (hs, v) in M] or not vs:
        continue
    base = idx(hs, "3.6.0")
    L += [f"### HEAD_SIZE = {hs}", "",
          "| impl | phase | " + " | ".join(f"{v} vs 3.6.0" for v in vs) + " |",
          "|---|:--|" + "--:|" * len(vs)]
    for impl in ("triton", "gluon"):
        for phase in ("decode", "prefill"):
            cells = []
            for v in vs:
                cur = idx(hs, v)
                rel = []
                for k_, r in cur.items():
                    if k_[0] != phase or k_ not in base:
                        continue
                    a, b = base[k_].get(impl, {}), r.get(impl, {})
                    if "us" in a and "us" in b:
                        rel.append(b["us"] / a["us"])
                if rel:
                    geo = math.exp(sum(math.log(x) for x in rel) / len(rel))
                    cells.append(f"{geo:.3f}×")
                else:
                    cells.append("–")
            L.append(f"| {impl} | {phase} | " + " | ".join(cells) + " |")
    L.append("")

# ---------------------------------------------------------------- split-KV
SP = {}
for hs in HSS:
    p = f"{BASE}/splits_full_hs{hs}.json"
    if os.path.exists(p):
        SP[hs] = json.load(open(p))
if SP:
    L += ["## 4. Decode split-KV — the heuristic added for this scan", "",
          "The gluon wrapper had **no split-KV**: it launched a 2-D grid of "
          "`num_seqs × num_kv_heads` workgroups and never used the kernel's `NUM_SPLITS` "
          "path, so at low batch most of the GPU sat idle (8 seqs × 8 kv heads = 64 "
          "workgroups on 256 CUs). Triton's public API has always split, via its 3d kernel "
          "+ `reduce_segments`, so the comparison was not like-for-like.", "",
          "Splitting is now chosen by `_select_num_splits`:", "",
          "```python",
          "if num_tiles <= 4:  return 1              # reduce costs more than the split wins",
          "target_wgs = num_cus * 4 // num_warps     # ~1 workgroup per SIMD",
          "splits     = round(target_wgs / (num_seqs * num_kv_heads))",
          "if splits < max(3, 64 // num_tiles):  return 1   # bias to the no-reduce path",
          "return max(1, min(num_tiles, splits))",
          "```", "",
          "Three measured facts shaped it:", "",
          "- **Target one workgroup per SIMD, not full `waves_per_eu` residency.** Decode is "
          "bandwidth-bound, so past ~one wave per SIMD extra parallelism buys nothing while "
          "each split adds a partial to write, reload and fold. Targeting 2 waves/SIMD (so "
          "2× the splits) cost up to 49% on shapes that already fill the machine — "
          "`waves_per_eu` is deliberately absent from the formula.",
          "- **Bias toward not splitting, like Triton's `use_2d_kernel`.** But Triton's exact "
          "condition does not transfer: it stays un-split when `max_seqlen_k <= 512`, which "
          "for gluon would cost up to **+83%** — Triton's 2d path packs "
          "`total_q_blocks × num_kv_heads` programs, whereas gluon's `ALL_DECODE` launches "
          "exactly `num_seqs × num_kv_heads`, which at C1 is *one* workgroup. The right "
          "form is a work-scaled one: require `splits × num_tiles >= 64`.",
          "- **Never split below 5 tiles.** At 4 tiles the attention kernel is shorter than "
          "the reduce launch, and splitting loses on every shape measured — even at one "
          "workgroup total.", ""]
    L += ["| HEAD_SIZE | cells | mean loss vs per-shape optimum | p90 | worst | never-split baseline |",
          "|--:|--:|--:|--:|--:|--:|"]
    for hs, recs in sorted(SP.items(), reverse=True):
        losses, base_losses = [], []
        for r in recs:
            t = {int(k_): v for k_, v in r["times"].items()}
            best = min(t.values())
            s = r["heur_S"]
            while s not in t and s > 1:
                s //= 2
            losses.append((t[s] / best - 1) * 100)
            base_losses.append((t[1] / best - 1) * 100)
        losses.sort()
        L.append(f"| {hs} | {len(losses)} | **+{sum(losses)/len(losses):.1f}%** | "
                 f"+{losses[int(0.9*len(losses))]:.1f}% | +{losses[-1]:.1f}% | "
                 f"+{sum(base_losses)/len(base_losses):.0f}% mean |")
    L += ["", "The last column is what the wrapper did before this change — never "
          "splitting — scored against the same per-shape optima.", ""]

# ---------------------------------------------------------------- raw
# ---------------------------------------------------------------- per-shape tables
L += ["## 5. Per-shape results", "",
      "Same layout as the earlier scans in `perf_scan_hs64_triton_36_37_38_7_28/`: one "
      "table per metric, columns `<version> tri | <version> glu`. Decode's headline metric "
      "is **GB/s** (memory bound), prefill's is **TFLOP/s** (compute bound); both are shown "
      "for each phase. Bytes = KV read + Q read + O write, at the real element size (1 B "
      "for fp8), so GB/s counts actual traffic. FLOPs count causal QK+PV as "
      "`4 * Hq * HEAD_SIZE` per (query, key) pair. `⚠` marks a cell that failed the "
      "correctness check.", ""]

def shape_label(r):
    if r["phase"] == "decode":
        return f"C{r['N']} ctx{r['L']} {r['Hq']}/{r['Hkv']}"
    return f"b{r['N']} N{r['L']} {r['Hq']}/{r['Hkv']}"


def metric_table(hs, vs, phase, dt, fn, fmt):
    """One table: rows = shapes, columns = <ver> tri / <ver> glu."""
    rows = ["| shape | " + " | ".join(f"{v} {i}" for v in vs for i in ("tri", "glu")) + " |",
            "|---|" + "--:|" * (2 * len(vs))]
    order = [r for r in M[(hs, vs[0])]
             if r["phase"] == phase and r["dtype"] == dt and "skipped" not in r]
    order.sort(key=lambda r: (r["Hkv"] != 1, r["Hq"], r["Hkv"], r["L"], r["N"]))
    for r0 in order:
        cells = []
        for v in vs:
            r = idx(hs, v).get(key(r0))
            for impl in ("triton", "gluon"):
                if r is None or impl not in r or "error" in r[impl]:
                    cells.append("–")
                    continue
                val = fn(r, impl)
                flag = "" if r[impl].get("ok", True) else " ⚠"
                cells.append((fmt.format(val) if val is not None else "–") + flag)
        rows.append(f"| {shape_label(r0)} | " + " | ".join(cells) + " |")
    return rows


for hs in sorted({k[0] for k in HAVE}, reverse=True):
    vs = [v for v in VERS_OK if (hs, v) in M]
    if not vs:
        continue
    L += [f"### HEAD_SIZE = {hs}", ""]
    for phase in ("decode", "prefill"):
        for dt in ("bf16", "fp8"):
            head = f"{phase.capitalize()} {dt} (HS{hs})"
            L += [f"#### {head} — time (µs / iter, lower is better)", ""]
            L += metric_table(hs, vs, phase, dt, lambda r, i: r[i]["us"], "{:.1f}")
            L += ["", f"#### {head} — bandwidth (GB/s, higher is better)", ""]
            L += metric_table(hs, vs, phase, dt,
                              lambda r, i: bytes_of(r) / (r[i]["us"] / 1e6) / 1e9, "{:.0f}")
            L += ["", f"#### {head} — compute (TFLOP/s, higher is better)", ""]
            L += metric_table(hs, vs, phase, dt,
                              lambda r, i: flops_of(r) / (r[i]["us"] / 1e6) / 1e12, "{:.0f}")
            L += ["", f"#### {head} — gluon speedup (time_triton / time_gluon)", ""]
            srows = ["| shape | " + " | ".join(vs) + " |", "|---|" + "--:|" * len(vs)]
            order = [r for r in M[(hs, vs[0])]
                     if r["phase"] == phase and r["dtype"] == dt and "skipped" not in r]
            order.sort(key=lambda r: (r["Hkv"] != 1, r["Hq"], r["Hkv"], r["L"], r["N"]))
            for r0 in order:
                cs = []
                for v in vs:
                    r = idx(hs, v).get(key(r0))
                    x = sp(r) if r else None
                    cs.append("–" if x is None else
                              (f"**{x:.2f}×**" if x > 1 else f"{x:.2f}×"))
                srows.append(f"| {shape_label(r0)} | " + " | ".join(cs) + " |")
            L += srows + [""]

open(f"{BASE}/perf_matrix.md", "w").write("\n".join(L) + "\n")
print(f"wrote {BASE}/perf_matrix.md")
