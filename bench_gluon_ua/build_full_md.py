"""Aggregate scanfull_<ver>.json (3.6.0 / 3.7.0 / 3.8.0) into perf_scan.md.

Sections: Decode (bf16, fp8) and Prefill (bf16, fp8) — each with per-metric tables and a
gluon-speedup table — then bf16-vs-fp8 per impl, takeaways and a raw appendix.
Reads/writes $SCAN_DIR (default: this folder)."""
import json, os

BASE = os.environ.get("SCAN_DIR", "/app/aiter/bench_gluon_ua")
ALT_SUFFIX = os.environ.get("ALT_SUFFIX", "_fp8wpe3")   # sensitivity pass to compare against
VERS = ["3.6.0", "3.7.0", "3.8.0"]
VLABEL = {"3.6.0": "3.6.0", "3.7.0": "3.7.0", "3.8.0": "3.8.0 (ToT)"}
IMPLS = ["triton", "gluon"]
DTYPES = ["bf16", "fp8"]

# Primary data: scanfull_<ver>.json (gluon at waves_per_eu=2 for both dtypes - the value the
# gfx950 wrapper already picks for bf16). scanfull_fp8wpe3_<ver>.json is the sensitivity pass
# at the wrapper's shipped fp8 waves_per_eu=3, which collapses on 3.8.0.
data, full, errors, labels = {}, {}, [], {}
alt = {}
for v in VERS:
    for r in json.load(open(f"{BASE}/scanfull_{v}.json")):
        full[v] = r["ver_full"]
        if "error" in r:
            errors.append(r)
            continue
        data[(v, r["phase"], r["dtype"], r["label"], r["impl"])] = r
        labels.setdefault(r["phase"], [])
        if r["label"] not in labels[r["phase"]]:
            labels[r["phase"]].append(r["label"])
    try:
        for r in json.load(open(f"{BASE}/scanfull{ALT_SUFFIX}_{v}.json")):
            if "error" not in r:
                alt[(v, r["phase"], r["dtype"], r["label"], r["impl"])] = r
    except FileNotFoundError:
        pass


HEADSZ = next((r.get("head_size", 128) for r in data.values()), 128)
ALT_DTYPES = [d for d in DTYPES if any(k[2] == d for k in alt)]


def cfg_wpe(r):
    """pull the wpe token out of a cfg string like 'BM16 nw1 mfma16 nb2 wpe2 TILE64'."""
    for tok in r["cfg"].split():
        if tok.startswith("wpe"):
            return tok[3:]
    return "?"


def get(v, phase, dt, lab, impl):
    return data.get((v, phase, dt, lab, impl))


def col_headers():
    return " | ".join(f"{VLABEL[v]} {'tri' if im == 'triton' else 'glu'}"
                      for v in VERS for im in IMPLS)


def metric_table(phase, dt, key, fmt):
    L = [f"| shape | {col_headers()} |", "|---|" + "--:|" * (len(VERS) * len(IMPLS))]
    for lab in labels[phase]:
        cells = []
        for v in VERS:
            for im in IMPLS:
                r = get(v, phase, dt, lab, im)
                cells.append(fmt(r[key]) if r else "–")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    return L + [""]


def speedup_table(phase, dt):
    L = ["| shape | " + " | ".join(VLABEL[v] for v in VERS) + " |",
         "|---|" + "--:|" * len(VERS)]
    for lab in labels[phase]:
        cells = []
        for v in VERS:
            t, g = get(v, phase, dt, lab, "triton"), get(v, phase, dt, lab, "gluon")
            cells.append(f"{t['time_us'] / g['time_us']:.2f}×" if t and g else "–")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    return L + [""]


def cfg_table(phase, dt):
    L = ["| shape | triton cfg | split S | gluon cfg | split Sg |", "|---|:--|--:|:--|--:|"]
    for lab in labels[phase]:
        t, g = get(VERS[-1], phase, dt, lab, "triton"), get(VERS[-1], phase, dt, lab, "gluon")
        if not (t and g):
            continue
        note = " (no split, no reduce)" if phase == "decode" and g["split"] == 1 else ""
        L.append(f"| {lab} | `{t['cfg']}` | {t['split']} | `{g['cfg']}` | {g['split']}{note} |")
    return L + [""]


def section(phase, dt):
    head = "Decode" if phase == "decode" else "Prefill"
    prim = ("GB/s (memory-bound)" if phase == "decode" else "TFLOP/s (compute-bound)")
    L = [f"## {head} — {dt}", "",
         f"Headline metric: **{prim}**. Time is total kernel time per iter"
         + (" (attention + reduce; gluon Sg=1 is attention only)." if phase == "decode" else "."),
         ""]
    if not any(get(v, phase, dt, lab, im) for v in VERS for lab in labels[phase] for im in IMPLS):
        return L + [f"_All {head.lower()} {dt} cells failed to compile — see "
                    f"[Failed cells](#failed-cells)._", ""]
    L += [f"### {head} {dt} — time (µs / iter, lower is better)", ""]
    L += metric_table(phase, dt, "time_us", lambda x: f"{x:.1f}")
    if phase == "decode":
        L += [f"### {head} {dt} — bandwidth (GB/s, higher is better)", ""]
        L += metric_table(phase, dt, "gbps", lambda x: f"{x:.0f}")
    L += [f"### {head} {dt} — compute (TFLOP/s, higher is better)", ""]
    L += metric_table(phase, dt, "tflops", lambda x: f"{x:.0f}")
    if phase == "prefill":
        L += [f"### {head} {dt} — bandwidth (GB/s)", ""]
        L += metric_table(phase, dt, "gbps", lambda x: f"{x:.0f}")
    L += [f"### {head} {dt} — gluon speedup (time_triton / time_gluon)", ""]
    L += speedup_table(phase, dt)
    L += [f"### {head} {dt} — configuration used", ""]
    L += cfg_table(phase, dt)
    return L


def dtype_ratio_table(phase):
    """time_bf16 / time_fp8 per impl on the ToT version — >1 means fp8 is faster."""
    L = ["| shape | " + " | ".join(f"{VLABEL[v]} {'tri' if im == 'triton' else 'glu'}"
                                   for v in VERS for im in IMPLS) + " |",
         "|---|" + "--:|" * (len(VERS) * len(IMPLS))]
    for lab in labels[phase]:
        cells = []
        for v in VERS:
            for im in IMPLS:
                a, b = get(v, phase, "bf16", lab, im), get(v, phase, "fp8", lab, im)
                cells.append(f"{a['time_us'] / b['time_us']:.2f}×" if a and b else "–")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    return L + [""]


def wpe_table(phase, dt):
    """gluon time at the reported wpe vs the alternative, per version."""
    a0 = next((get(v, phase, dt, lab, "gluon") for v in VERS for lab in labels[phase]
               if get(v, phase, dt, lab, "gluon")), None)
    b0 = next((alt[(v, phase, dt, lab, "gluon")] for v in VERS for lab in labels[phase]
               if alt.get((v, phase, dt, lab, "gluon"))), None)
    wa, wb = (cfg_wpe(a0) if a0 else "?"), (cfg_wpe(b0) if b0 else "?")
    L = ["| shape | " + " | ".join(f"{VLABEL[v]} wpe{wa} | {VLABEL[v]} wpe{wb} | ×"
                                   for v in VERS) + " |",
         "|---|" + "--:|" * (3 * len(VERS))]
    for lab in labels[phase]:
        cells = []
        for v in VERS:
            a, b = get(v, phase, dt, lab, "gluon"), alt.get((v, phase, dt, lab, "gluon"))
            if a and b:
                cells += [f"{a['time_us']:.1f}", f"{b['time_us']:.1f}",
                          f"**{b['time_us'] / a['time_us']:.2f}×**"]
            else:
                cells += ["–", "–", "–"]
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    return L + [""]


def wpe_speedup_table(phase, dt):
    """gluon-vs-triton speedup at each waves_per_eu setting."""
    a0 = next((get(v, phase, dt, lab, "gluon") for v in VERS for lab in labels[phase]
               if get(v, phase, dt, lab, "gluon")), None)
    b0 = next((alt[(v, phase, dt, lab, "gluon")] for v in VERS for lab in labels[phase]
               if alt.get((v, phase, dt, lab, "gluon"))), None)
    wa, wb = (cfg_wpe(a0) if a0 else "?"), (cfg_wpe(b0) if b0 else "?")
    L = ["| shape | " + " | ".join(f"{VLABEL[v]} wpe{wa} | {VLABEL[v]} wpe{wb}"
                                   for v in VERS) + " |",
         "|---|" + "--:|" * (2 * len(VERS))]
    for lab in labels[phase]:
        cells = []
        for v in VERS:
            t = get(v, phase, dt, lab, "triton")
            a, b = get(v, phase, dt, lab, "gluon"), alt.get((v, phase, dt, lab, "gluon"))
            cells.append(f"{t['time_us'] / a['time_us']:.2f}×" if (t and a) else "–")
            cells.append(f"{t['time_us'] / b['time_us']:.2f}×" if (t and b) else "–")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    return L + [""]


def raw_table():
    L = ["| ver | phase | dtype | shape | impl | split | time µs | GB/s | TFLOP/s | xcheck | cfg |",
         "|---|:--|:--|:--|:--:|--:|--:|--:|--:|--:|:--|"]
    for phase in ("decode", "prefill"):
        for dt in DTYPES:
            for lab in labels[phase]:
                for v in VERS:
                    for im in IMPLS:
                        r = get(v, phase, dt, lab, im)
                        if not r:
                            continue
                        L.append(f"| {VLABEL[v]} | {phase} | {dt} | {lab} | {im} | {r['split']} | "
                                 f"{r['time_us']:.1f} | {r['gbps']:.0f} | {r['tflops']:.1f} | "
                                 f"{r['xc']:.0e} | `{r['cfg']}` |")
    return L + [""]


def xc_max(dt):
    xs = [r["xc"] for k, r in data.items() if k[2] == dt]
    return max(xs) if xs else float("nan")


def span(phase, dt, v, pred):
    sp = [get(v, phase, dt, lab, "triton")["time_us"] / get(v, phase, dt, lab, "gluon")["time_us"]
          for lab in labels[phase] if pred(lab) and get(v, phase, dt, lab, "triton")
          and get(v, phase, dt, lab, "gluon")]
    return (min(sp), max(sp)) if sp else (float("nan"), float("nan"))


gqa = lambda lab: lab.endswith("64/8")
mqa = lambda lab: lab.endswith("8/1")
V = VERS[-1]
ncells = len(data) // 2

hdr = [
    f"# gfx950 unified attention (HEAD_SIZE={HEADSZ}) — gluon vs Triton, decode + prefill, "
    "bf16 + FP8",
    "",
    f"Across **Triton 3.6.0 / 3.7.0 / 3.8.0**. HEAD_SIZE={HEADSZ}, causal, paged KV with "
    "block_size = TILE_SIZE = 64, output always bf16.",
    "",
    "| phase | shapes | heads | dtypes |",
    "|---|---|---|---|",
    "| decode | C ∈ {16,32,64,128} × ctx ∈ {1024,8192} | 64/8 GQA, 8/1 MQA | bf16, fp8 |",
    "| prefill | (B,N) ∈ {(1,1024),(4,1024),(8,1024),(1,8192)}, query_len = kv_len = N "
    "| 64/8 GQA, 8/1 MQA | bf16, fp8 |",
    "",
    "- **triton** = `kernel_unified_attention_3d` + `reduce_segments` for decode "
    "(split `S` from `select_3d_config`), `kernel_unified_attention_2d` for prefill "
    "(`select_2d_config`).",
    "- **gluon** = the gfx950 `kernel_unified_attention_2d`; decode right-sizes the split to "
    "~CU·4 workgroups (`select_gluon_num_splits`, **Sg=1 ⇒ non-split path, no reduce kernel**), "
    "prefill runs the wrapper's BLOCK_M=128 / num_warps=4 / 32×32 MFMA config.",
    "- **FP8** = q, k and v all `torch.float8_e4m3fn`, per-tensor scalar descales fixed at 1.0 "
    "(identical on both sides, so the cross-check stays meaningful and the magnitude cannot "
    "affect kernel time). GB/s is computed at 1 byte/element, so it counts real traffic.",
    "- **gluon `waves_per_eu` = 2 for both dtypes.** That is the gfx950 wrapper's own bf16 "
    "value; its fp8 branch picks 3, which collapses on 3.8.0 (see the sensitivity section — "
    "both fp8 runs are reported). NUM_BUFFERS=2 everywhere (the wrapper prefers 1 for decode; "
    "2 keeps all three Triton versions and both dtypes apples-to-apples).",
    "- Method: 512 MB L2 flush every iter, torch.profiler per-kernel self-time, 8 warmup + "
    "30 iters, one Triton version at a time on GPU 0.",
    "- triton per column: " + ", ".join(f"`{full[v]}`" for v in VERS) + ".",
    f"- Cross-check (gluon vs Triton output, max abs diff): ≤ **{xc_max('bf16'):.0e}** bf16, "
    f"≤ **{xc_max('fp8'):.0e}** fp8.",
    "",
]

out = list(hdr)
for phase in ("decode", "prefill"):
    for dt in DTYPES:
        out += section(phase, dt)

out += ["## bf16 vs FP8 (same impl, `time_bf16 / time_fp8` — >1× means fp8 is faster)", ""]
for phase in ("decode", "prefill"):
    out += [f"### {phase}", ""] + dtype_ratio_table(phase)

if alt:
    dts = ", ".join(ALT_DTYPES)
    if HEADSZ >= 128:
        why = ("The gfx950 gluon wrapper used to pick `waves_per_eu=3` for fp8 at "
               "HEAD_SIZE=128 (2 for bf16). That is roughly neutral on 3.6.0 and 3.7.0 but "
               "falls off a cliff on 3.8.0, so **every fp8 number above uses wpe=2** — the "
               "same value the wrapper picks for bf16. A probe over wpe ∈ {1,2,3,4} backs "
               "this up: bf16 decode is flat across 1–3 (±3%) and fp8 decode matches wpe=1 "
               "within 3% at wpe=2, while wpe=4 is a large loss for both.")
    else:
        why = (f"The gfx950 gluon wrapper picks `waves_per_eu=4` for every dtype below "
               f"HEAD_SIZE=128. At HEAD_SIZE={HEADSZ} that is the *worst* of "
               "{1,2,3,4} on 3.7.0/3.8.0, so the tables above use the tuned values instead "
               "(**decode wpe=1, prefill wpe=3**; see `tune_hs64.py`). Decode is flat over "
               "wpe 1–3 for bf16 and clearly prefers 1 for fp8; prefill prefers 3, with 4 "
               "winning only on 3.6.0 and losing badly on the newer two.")
    out += [
        f"## `waves_per_eu` sensitivity — tuned vs the wrapper's shipped value", "",
        why + f" Both full runs ({dts}) are kept below.", "",
        "### gluon time — reported wpe vs the shipped one (µs/iter; × = shipped / reported, "
        ">1 means the shipped setting is slower)", "",
    ]
    for phase in ("decode", "prefill"):
        for dt in ALT_DTYPES:
            out += [f"**{phase} {dt}**", ""] + wpe_table(phase, dt)
    out += ["### gluon-vs-Triton speedup at each setting", ""]
    for phase in ("decode", "prefill"):
        for dt in ALT_DTYPES:
            out += [f"**{phase} {dt}**", ""] + wpe_speedup_table(phase, dt)

# ---- computed takeaways ----
d_gqa, d_mqa = span("decode", "bf16", V, gqa), span("decode", "bf16", V, mqa)
d8_gqa, d8_mqa = span("decode", "fp8", V, gqa), span("decode", "fp8", V, mqa)
p_all, p8_all = span("prefill", "bf16", V, lambda l: True), span("prefill", "fp8", V, lambda l: True)


def dt_gain(phase, im):
    g = [get(V, phase, "bf16", lab, im)["time_us"] / get(V, phase, "fp8", lab, im)["time_us"]
         for lab in labels[phase]
         if get(V, phase, "bf16", lab, im) and get(V, phase, "fp8", lab, im)]
    return (min(g), max(g)) if g else (float("nan"), float("nan"))


def wpe_ratios(phase, v, dt="fp8"):
    """alternative-wpe time / reported-wpe time; >1 means the alternative is slower."""
    return [alt[(v, phase, dt, lab, "gluon")]["time_us"]
            / get(v, phase, dt, lab, "gluon")["time_us"]
            for lab in labels[phase] if alt.get((v, phase, dt, lab, "gluon"))
            and get(v, phase, dt, lab, "gluon")]


def wpe_note(phase):
    """e.g. 'a mean shipped/reported time ratio of 1.00 / 1.05 / 1.52 on 3.6/3.7/3.8 ...'."""
    if not alt:
        return "unmeasured"
    rr = {v: [x for d in ALT_DTYPES for x in wpe_ratios(phase, v, d)] for v in VERS}
    rr = {v: x for v, x in rr.items() if x}
    if not rr:
        return "unmeasured"
    means = [sum(x) / len(x) for x in rr.values()]
    worst = max(max(x) for x in rr.values())
    return ("a mean shipped/reported time ratio of " + " / ".join(f"{m:.2f}" for m in means)
            + f" on 3.6/3.7/3.8 (worst cell {worst:.2f}×)")


def alt_span(phase, v, shipped):
    """gluon-vs-triton fp8 speedup span; shipped=True uses the wpe=3 gluon run."""
    src = alt if shipped else data
    sp = [get(v, phase, "fp8", lab, "triton")["time_us"] / src[(v, phase, "fp8", lab, "gluon")]["time_us"]
          for lab in labels[phase] if src.get((v, phase, "fp8", lab, "gluon"))
          and get(v, phase, "fp8", lab, "triton")]
    return (min(sp), max(sp)) if sp else (float("nan"), float("nan"))


span38_wpe2 = alt_span("prefill", "3.8.0", False)
span38_wpe3 = alt_span("prefill", "3.8.0", True) if alt else (float("nan"), float("nan"))

def wpe_takeaway():
    """The waves_per_eu bullet, worded for whichever head size this report covers."""
    if not alt:
        return "- `waves_per_eu` sensitivity not measured for this head size."
    if HEADSZ >= 128:
        return ("- **The wrapper's old fp8 `waves_per_eu=3` is mistuned on Triton 3.8.0** — "
                "hence wpe=2 everywhere here. Against the same kernel at wpe=2, the shipped "
                "setting shows " + wpe_note("decode") + " on decode and " + wpe_note("prefill")
                + " on prefill. On 3.8.0 that is the difference between losing to Triton "
                f"(fp8 prefill {span38_wpe3[0]:.2f}–{span38_wpe3[1]:.2f}×) and beating it "
                f"({span38_wpe2[0]:.2f}–{span38_wpe2[1]:.2f}×). bf16 is unaffected — it "
                "already runs wpe=2 — so it is the fp8 branch of the heuristic that needed "
                "revisiting.")
    return (f"- **The wrapper's `waves_per_eu=4` for HEAD_SIZE<128 is mistuned** — hence the "
            f"tuned decode wpe=1 / prefill wpe=3 used here. Against those, wpe=4 shows "
            + wpe_note("decode") + " on decode and " + wpe_note("prefill") + " on prefill. "
            "On 3.8.0 it is the difference between losing to Triton (fp8 prefill "
            f"{span38_wpe3[0]:.2f}–{span38_wpe3[1]:.2f}×) and beating it "
            f"({span38_wpe2[0]:.2f}–{span38_wpe2[1]:.2f}×). Unlike the HEAD_SIZE=128 case "
            "this hits **both** dtypes, since the <128 branch never distinguished them.")


tk = ["## Takeaways", "",
      f"- **Decode bf16** on {VLABEL[V]}: gluon {d_gqa[0]:.2f}–{d_gqa[1]:.2f}× on GQA 64/8, "
      f"{d_mqa[0]:.2f}–{d_mqa[1]:.2f}× on MQA 8/1.",
      f"- **Decode fp8** on {VLABEL[V]}: gluon {d8_gqa[0]:.2f}–{d8_gqa[1]:.2f}× on GQA 64/8, "
      f"{d8_mqa[0]:.2f}–{d8_mqa[1]:.2f}× on MQA 8/1.",
      f"- **Prefill bf16** on {VLABEL[V]}: gluon {p_all[0]:.2f}–{p_all[1]:.2f}×; "
      f"**prefill fp8**: {p8_all[0]:.2f}–{p8_all[1]:.2f}×.",
      f"- **fp8 vs bf16 speedup** on {VLABEL[V]}: decode {dt_gain('decode', 'gluon')[0]:.2f}–"
      f"{dt_gain('decode', 'gluon')[1]:.2f}× (gluon) / {dt_gain('decode', 'triton')[0]:.2f}–"
      f"{dt_gain('decode', 'triton')[1]:.2f}× (triton); prefill "
      f"{dt_gain('prefill', 'gluon')[0]:.2f}–{dt_gain('prefill', 'gluon')[1]:.2f}× (gluon) / "
      f"{dt_gain('prefill', 'triton')[0]:.2f}–{dt_gain('prefill', 'triton')[1]:.2f}× (triton).",
      wpe_takeaway(),
      "- **fp8 decode runs the 32×32 tile, not the 16×16 one bf16 uses.** Emitted instruction "
      "is `v_mfma_scale_f32_32x32x64_f8f6f4`; its 16×16×128 sibling does not lower at the "
      "kernel's `K_WIDTH=16` (`ConvertTritonAMDGPUToLLVM` → `PassManager::run failed`, on all "
      "three versions, NUM_BUFFERS 1 or 2, BLOCK_M 16 or 32). `K_WIDTH=32` compiles but is "
      "numerically wrong (91% relative-mean error vs a bf16 reference where 32×32 gives 2.6% "
      "= pure quantization), so 16×16 fp8 needs a dot-operand layout fix rather than a "
      "K_WIDTH bump — and the upside is small: measured 16×16 is only 1.00–1.04× faster than "
      "32×32, decode being memory-bound. So the fp8 decode rows carry 32×32's wasted M rows "
      "(24 of 32 at nqpk=8) and that is the correct trade today.",
      f"- Correctness: gluon-vs-Triton max abs diff ≤ {xc_max('bf16'):.0e} (bf16) and "
      f"≤ {xc_max('fp8'):.0e} (fp8) over all {ncells} measured cells.", ""]
out += tk

if errors:
    out += ["## Failed cells", "",
            "Cells that did not compile; they are blank in the tables above.", "",
            "| ver | phase | dtype | shape | error |", "|---|:--|:--|:--|:--|"]
    for r in errors:
        out.append(f"| {VLABEL.get(r['ver'], r['ver'])} | {r['phase']} | {r['dtype']} | "
                   f"{r['label']} | `{r['error']}` |")
    out.append("")
else:
    out += ["## Failed cells", "", "None — every cell compiled and ran on all three versions.", ""]

out += ["## Appendix — raw per-run data (every measurement, all metrics per row)", "",
        "One row per profiled run. `time µs` is total kernel time/iter (decode: attention + "
        "reduce, gluon Sg=1 is attention only), `GB/s` and `TFLOP/s` are derived from it, "
        "`split` is the segment/split count.", ""] + raw_table()

open(f"{BASE}/perf_scan.md", "w").write("\n".join(out) + "\n")
print(f"wrote {BASE}/perf_scan.md  ({len(data)} records, {len(errors)} failed cells)")
