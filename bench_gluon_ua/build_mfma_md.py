"""Render mfma_shape_study_7_28/mfma_shapes.md from the mfma_study_*.json sweeps."""
import json, os

BASE = os.environ.get("MFMA_DIR", "/app/aiter/bench_gluon_ua/mfma_shape_study_7_28")
DS = [("hs128_3.6.0", "HS128 · 3.6.0", 128), ("hs128_3.8.0", "HS128 · 3.8.0", 128),
      ("hs64_3.6.0", "HS64 · 3.6.0", 64)]
R = {n: json.load(open(f"{BASE}/mfma_study_{n}.json")) for n, _, _ in DS}
DT_DS = [("hs128_3.6.0", "HS128 · 3.6.0"), ("hs128_3.8.0", "HS128 · 3.8.0"),
         ("hs64_3.6.0", "HS64 · 3.6.0"), ("hs64_3.8.0", "HS64 · 3.8.0")]
DT = {}
for n, _ in DT_DS:
    try:
        DT[n] = json.load(open(f"{BASE}/dectile_{n}.json"))
    except FileNotFoundError:
        pass
INSTR = {("bf16", 32): "32x32x16", ("bf16", 16): "16x16x32",
         ("fp8", 32): "32x32x64", ("fp8", 16): "16x16x128"}

# nqpk (GQA-ratio) scan, incl. MHA -- ../mfma_nqpk_scan.py
NQ_DS = [("hs128_3.6.0", "HS128 · 3.6.0"), ("hs128_3.7.0", "HS128 · 3.7.0"),
         ("hs64_3.6.0", "HS64 · 3.6.0"), ("hs64_3.7.0", "HS64 · 3.7.0")]
NQ = {}
for n, _ in NQ_DS:
    try:
        NQ[n] = json.load(open(f"{BASE}/nqpk_{n}.json"))
    except FileNotFoundError:
        pass
# supporting probes (register pressure / buffer-op path / fp8 TILE) -- ../mfma_tile_probes.py
PR = {}
for hs in (128, 64):
    try:
        PR[hs] = json.load(open(f"{BASE}/probes_hs{hs}_3.6.0.json"))
    except FileNotFoundError:
        pass
MAX_INT32 = 2**31 - 1
NOISE = 0.03          # cross-card control: run-to-run spread on this bench (see intro)
# Headline numbers for section 5, filled in by the sections that measure them so the
# bottom line can never drift from the tables above it.
HEAD = {"off_path": "–", "spill_pen": "–", "fp8_gain": "–", "prefill": "–"}


def kv_bytes(r):
    """Bytes in ONE of the two kv tensors -- what the wrapper's buffer-op guard tests."""
    return r["C"] * r["ctx"] * r["Hkv"] * r["head_size"] * 2


def buf_on(r):
    return kv_bytes(r) <= MAX_INT32


def sel(n, phase, dt, mfma):
    return [r for r in R[n] if r["phase"] == phase and r["dtype"] == dt and r["mfma"] == mfma]


def correct(r):
    return "error" not in r and r["rel"] < 0.05


def best(n, phase, dt, mfma):
    g = [r for r in sel(n, phase, dt, mfma) if correct(r)]
    return min(g, key=lambda r: r["time_us"]) if g else None


L = [
    "# gfx950 MFMA instruction-shape study — gluon unified attention",
    "",
    "gfx950 gives this kernel **two MFMA shapes per dtype**, selected by the kernel's "
    "`MFMA_DIM` knob:",
    "",
    "| dtype | `MFMA_DIM=32` | `MFMA_DIM=16` | MACs per instruction |",
    "|---|:--|:--|:--|",
    "| bf16 | `v_mfma_f32_32x32x16_bf16` | `v_mfma_f32_16x16x32_bf16` | 16384 vs 8192 (**2×**) |",
    "| fp8 | `v_mfma_scale_f32_32x32x64_f8f6f4` | `v_mfma_scale_f32_16x16x128_f8f6f4` | 65536 vs 32768 (**2×**) |",
    "",
    "Within a pair both shapes consume the same K per unit of work, so the 32×32 form does "
    "twice the MACs per issue; what differs is the **M granularity**. `BLOCK_M` must be a "
    "multiple of `MFMA_DIM × num_warps`, and in decode only `NUM_QUERIES_PER_KV` of the M rows "
    "hold real queries — so the wide tile wastes lanes exactly where the narrow one does not.",
    "",
    "Sweep: `MFMA_DIM ∈ {16,32}` × `K_WIDTH_QK` × `K_WIDTH_PV` (the kernel hard-codes both per "
    "shape), for decode and prefill, bf16 and fp8, at HEAD_SIZE 128 and 64. Every variant is "
    "correctness-checked against the Triton kernel on the same inputs. Method as in the perf "
    "scans: 512 MB L2 flush per iter, torch.profiler, 8 warmup + 30 iters, gfx950 / 256 CU. "
    "Driver: `../mfma_shape_study.py`, which loads *patched copies* of the kernel module so the "
    "checked-in kernel is never modified.",
    "",
    "Shapes: decode `C128 ctx8192 64/8` (BLOCK_M = MFMA_DIM, num_warps=1), prefill "
    "`b8 1024 64/8` (BLOCK_M=128, num_warps = 128/MFMA_DIM).",
    "",
    "### Method notes for this revision",
    "",
    "- **Harness fix — buffer ops.** `bench_ua.launch_glu_2d` used to pass "
    "`USE_LOAD_BUFFER_OP=True` unconditionally. Buffer ops address through a 32-bit "
    "descriptor offset, so above ~2 GB per tensor they read out of range: a 17 GB MHA cache "
    "measured **103% relative error at an impossible 12.3 TB/s** (vs 0.40% at 6.6 TB/s with "
    "them off). The harness now derives the flags exactly as the production wrapper does "
    "(`buffer_op_flags()`), and all six call sites in `bench_gluon_ua/` route through it. "
    "Shipped code was never affected — the wrapper always had the guard. GQA-only shapes stay "
    "under the limit, which is why earlier revisions never hit it; **MHA reaches it 8× "
    "sooner**. The affected first-pass runs are archived under `bufop_bug_runs/`.",
    "- **Noise floor ~3%.** Cards 0–3 were busy with an unrelated job, so this revision's runs "
    "are on card 4 while the older JSONs were taken on card 0. Re-running the HS128 `dectile` "
    "points on card 4 (`xcheck_card4_dectile_hs128_3.6.0.json`) reproduces the 16×16/32×32 "
    "**ratios** within 0.03 while absolute times move −2.3%…+3.7%. Effects below ~3% are "
    "reported as ties.",
    "- **Triton 3.8.0 is unavailable** in the current environment: the only `libtriton.so` in "
    "`/home/mekaymak/triton` is a Python 3.10 build and the interpreter is 3.12, so `import "
    "triton` fails. The 3.8.0 columns below predate that; **3.7.0 stands in** as the second "
    "compiler version for the new scans, and `mfma_study_hs64_3.8.0` is still missing.",
    "",
    "## 1. Lane utilisation — what the shapes *should* do",
    "",
    "M rows carrying real work, for these shapes (`NUM_QUERIES_PER_KV = 8`):",
    "",
    "| phase | BLOCK_M | valid M rows | `MFMA_DIM=16` | `MFMA_DIM=32` |",
    "|---|--:|--:|:--|:--|",
    "| decode | = MFMA_DIM | 8 (one token × nqpk) | 8/16 = **50%** | 8/32 = **25%** |",
    "| prefill | 128 | 128 (16 tokens × nqpk) | 128/128 = **100%** | 128/128 = **100%** |",
    "",
    "So the prediction is: **decode favours 16×16** (half the wasted lanes), **prefill is a wash "
    "on waste and should favour 32×32** on raw instruction efficiency (2× MACs/issue, and "
    "num_warps 4 instead of 8 for the same BLOCK_M). The measurements below say the second half "
    "of that is right and the first half is real but almost free.",
    "",
    "## 2. Best correct configuration per shape",
    "",
    "Time µs/iter, best over the K_WIDTH grid; `×tri` = Triton kernel / gluon on the same input.",
    "",
]

hdr = "| dataset | phase | dtype | " + " | ".join(
    f"{INSTR[(d, m)]}" for d in ("bf16",) for m in (32, 16)) + " | winner |"
L += ["| dataset | phase | dtype | 32×32 | 16×16 | winner |", "|---|:--|:--|--:|--:|:--|"]
for n, lab, hs in DS:
    for phase in ("decode", "prefill"):
        for dt in ("bf16", "fp8"):
            b32, b16 = best(n, phase, dt, 32), best(n, phase, dt, 16)
            c32 = f"{b32['time_us']:.1f} ({b32['triton_us'] / b32['time_us']:.2f}×tri)" if b32 else "–"
            c16 = f"{b16['time_us']:.1f} ({b16['triton_us'] / b16['time_us']:.2f}×tri)" if b16 else "**no correct config**"
            if b32 and b16:
                hi, lo = max(b32["time_us"], b16["time_us"]), min(b32["time_us"], b16["time_us"])
                if hi / lo < 1.02:
                    w = f"tie ({hi / lo:.2f}×)"
                elif b32["time_us"] < b16["time_us"]:
                    w = f"**32×32** by {b16['time_us'] / b32['time_us']:.2f}×"
                else:
                    w = f"**16×16** by {b32['time_us'] / b16['time_us']:.2f}×"
            else:
                w = "32×32 (only option)"
            L.append(f"| {lab} | {phase} | {dt} | {c32} | {c16} | {w} |")
L.append("")

L += [
    "### What that says",
    "",
    "- **Prefill: 32×32 wins decisively — 1.32–1.79× (bf16).** Both shapes use 100% of the M "
    "lanes here, so this is pure instruction efficiency: half the issues for the same work, and "
    "`num_warps=4` instead of 8 for BLOCK_M=128 (fewer cross-warp reductions and less LDS "
    "pressure). This is the largest single effect in the study.",
    "- **Decode: the two shapes tie *on this shape*** — `C128 ctx8192 64/8` is fully "
    "bandwidth-saturated (gluon runs it at ~6.5–6.9 TB/s ≈ HBM peak), so the matrix units idle "
    "waiting for KV either way and the 50%-vs-25% lane difference is invisible. That is not the "
    "whole decode story — see section 2b, where the rest of the shape range does favour 16×16.",
    "- **fp8 has no working 16×16 at all** (section 4), so fp8 decode is stuck at 25% M "
    "utilisation. Timing the (numerically broken) 16×16 builds puts the ceiling on fixing it at "
    "**~1–5%** — 315.5 vs 314.8 µs at HS128/3.6, 315.2 vs 323.0 at HS128/3.8, 189.3 vs 198.8 at "
    "HS64. Worth doing, not urgent.",
    "",
]

if DT:
    L += ["## 2b. Decode across the shape range — where the lane waste does show", "",
          "The shape used above is bandwidth-saturated, which hides the M-utilisation "
          "difference. Walking the decode shape range instead (bf16, `BLOCK_M = MFMA_DIM`, "
          "num_warps=1, tuned waves_per_eu per head size; `../mfma_decode_tile_scan.py`):", "",
          "| shape | " + " | ".join(lab for n, lab in DT_DS if n in DT) + " |",
          "|---|" + ":--:|" * len([1 for n, _ in DT_DS if n in DT])]
    present = [n for n, _ in DT_DS if n in DT]
    for i, lab in enumerate(r["label"] for r in DT[present[0]]):
        cells = []
        for n in present:
            r = DT[n][i]
            ratio = r["t_16x16"] / r["t_32x32"]
            tag = ("**16×16 +%.0f%%**" % ((1 / ratio - 1) * 100) if ratio < 0.98
                   else ("32×32 +%.0f%%" % ((ratio - 1) * 100) if ratio > 1.02 else "tie"))
            cells.append(f"{r['t_16x16']:.1f} / {r['t_32x32']:.1f} — {tag}")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    rr = [(n, r, r["t_16x16"] / r["t_32x32"]) for n in present for r in DT[n]]
    wins = [x for x in rr if x[2] < 0.98]
    ties = [x for x in rr if 0.98 <= x[2] <= 1.02]
    loss = [x for x in rr if x[2] > 1.02]
    margins = sorted((1 / x[2] - 1) * 100 for x in wins)
    lose_txt = (", ".join(f"`{r['label']}` @ HS{r['head_size']}/{r['ver']} "
                          f"({(g - 1) * 100:.0f}%)" for _, r, g in loss)
                if loss else "none")
    L += ["", "_Cells are `16×16 µs / 32×32 µs — winner`._", "",
          f"**16×16 is the better decode default**: it wins {len(wins)} of the {len(rr)} cells "
          f"(by {margins[0]:.0f}–{margins[-1]:.0f}%), ties on {len(ties)}, loses on "
          f"{len(loss)} — {lose_txt}. That matches the lane-utilisation prediction: the gap "
          "shows wherever the kernel is not yet HBM-limited, so the wasted M lanes cost real "
          "issue slots, and narrows to a tie on the most bandwidth-saturated cells.", "",
          "Note these use the kernel's shipped K_WIDTHs, whereas section 2 compares each tile at "
          "its *best* K_WIDTH — which is why the large shape reads as a tie there and a 16×16 "
          "win here. Either way 16×16 is never materially behind.", ""]

if NQ:
    present = [n for n, _ in NQ_DS if n in NQ]
    dec0 = [r for r in NQ[present[0]] if r["phase"] == "decode"]
    L += [
        "## 2c. The GQA ratio — from MHA (nqpk=1) to nqpk=32", "",
        "Everything above measures a single point on the axis that actually drives the M-tile "
        "choice: both `64/8` and `8/1` are **nqpk=8**. Decode packs `nqpk` valid rows into "
        "`BLOCK_M`, and the wrapper sizes the tile as `block_m = max(MFMA_DIM, "
        "next_pow2(nqpk))`, `num_warps = block_m / MFMA_DIM`, so the two shapes trade off "
        "differently at each end:", "",
        "| nqpk | 16×16 tile | 32×32 tile | valid M rows |",
        "|--:|:--|:--|:--|",
        "| 1 (**MHA**) | BM16 / nw1 | BM32 / nw1 | 1/16 = 6% vs 1/32 = 3% |",
        "| 8 | BM16 / nw1 | BM32 / nw1 | 50% vs 25% |",
        "| 16 | BM16 / nw1 | BM32 / nw1 | 100% vs 50% |",
        "| 32 | BM32 / **nw2** | BM32 / nw1 | 100% vs 100% — 16×16 needs 2 warps |", "",
        "So the lane-utilisation argument predicts 16×16 wins biggest at MHA and *loses* at "
        "nqpk=32. Scan: `NUM_QUERY_HEADS=64` fixed, `Hkv = 64/nqpk` (so KV traffic scales "
        "with the ratio, as it does in a real model), bf16, `../mfma_nqpk_scan.py`.", "",
        "Ratio = `t(16×16) / t(32×32)`; **<1 means 16×16 faster**. `*` marks cells whose KV "
        "cache exceeds the buffer-op limit, so the wrapper runs them on the non-buffer path "
        "(see 2d — that is where the big numbers come from).", "",
        "| shape | nqpk | " + " | ".join(lab for n, lab in NQ_DS if n in NQ) + " |",
        "|---|--:|" + "--:|" * len(present),
    ]
    for i, r0 in enumerate(dec0):
        cells = []
        for n in present:
            r = NQ[n][i]
            g = r["r16"]["time_us"] / r["r32"]["time_us"]
            cells.append(f"{g:.3f}{'*' if not buf_on(r) else ''}")
        L.append(f"| {r0['label']} | {r0['nqpk']} | " + " | ".join(cells) + " |")

    allc = [(n, r) for n in present for r in NQ[n] if r["phase"] == "decode"]
    on = [(n, r) for n, r in allc if buf_on(r)]
    off = [(n, r) for n, r in allc if not buf_on(r)]
    g = lambda r: r["r16"]["time_us"] / r["r32"]["time_us"]
    on_win = [x for x in on if g(x[1]) < 1 - NOISE]
    on_tie = [x for x in on if 1 - NOISE <= g(x[1]) <= 1 + NOISE]
    on_los = [x for x in on if g(x[1]) > 1 + NOISE]
    off128 = [x for x in off if x[1]["head_size"] == 128]
    off64 = [x for x in off if x[1]["head_size"] == 64]
    mk = lambda xs: (f"{min(g(r) for _, r in xs):.2f}–{max(g(r) for _, r in xs):.2f}"
                     if xs else "–")
    if off128:
        HEAD["off_path"] = (f"{1/max(g(r) for _, r in off128):.1f}–"
                            f"{1/min(g(r) for _, r in off128):.1f}×")
    pre_all = [r for n in present for r in NQ[n] if r["phase"] == "prefill"]
    if pre_all:
        pr = [r["r16"]["time_us"] / r["r32"]["time_us"] for r in pre_all]
        HEAD["prefill"] = f"{min(pr):.2f}–{max(pr):.2f}×"
    L += [
        "", f"_Noise floor is ~{NOISE:.0%} (see intro), so |effect| < {NOISE:.0%} is a tie._",
        "",
        f"**On the buffer-op path ({len(on)} cells): 16×16 wins {len(on_win)}, ties "
        f"{len(on_tie)}, loses {len(on_los)}** — ratios {mk(on)}. The margin is small and, "
        "importantly, **flat in nqpk**: MHA is not the 16×16 stronghold the lane arithmetic "
        "predicts, and nqpk=32 is not the 32×32 win it predicts. Two effects cancel the "
        "lane math:",
        "",
        "- **MHA saturates HBM at a far smaller batch.** KV traffic scales with "
        "`num_kv_heads`, so at nqpk=1 the kernel reads 8× more cache per token than at "
        "nqpk=8 and hits the bandwidth roof early — at `C16 ctx1024` MHA already runs at "
        "~5.9 TB/s against a ~6.4–6.7 TB/s ceiling, where the matrix units idle either way. "
        "The 16×16 edge only survives at `C1`, which is too small to saturate.",
        "- **At nqpk=32 the predicted 32×32 win does not appear.** Both tiles are 100% "
        "M-packed and 16×16 pays `num_warps=2`, yet it still ties or wins. Whatever the "
        "second warp costs is below the noise floor on a memory-bound kernel.",
        "",
        f"**Off the buffer-op path, HEAD_SIZE=128 ({len(off128)} cells): 16×16 wins by "
        f"{HEAD['off_path']}** (ratios {mk(off128)}). At HEAD_SIZE=64 the same cells are a tie "
        f"(ratios {mk(off64)}). That asymmetry is a register-spill cliff, not a tile "
        "property — section 2d.",
        "",
    ]

    pre0 = [r for r in NQ[present[0]] if r["phase"] == "prefill"]
    if pre0:
        L += ["**Prefill is nqpk-independent, as predicted.** `BLOCK_Q = BLOCK_M/nqpk` "
              "tokens keep the M tile 100% full at any ratio, so 32×32 wins everywhere by "
              "its raw instruction advantage:", "",
              "| shape | nqpk | " + " | ".join(lab for n, lab in NQ_DS if n in NQ) + " |",
              "|---|--:|" + "--:|" * len(present)]
        for i, r0 in enumerate(pre0):
            j = len(dec0) + i
            cells = [f"**{NQ[n][j]['r32']['time_us'] and NQ[n][j]['r16']['time_us'] / NQ[n][j]['r32']['time_us']:.2f}×**"
                     for n in present]
            L.append(f"| {r0['label']} | {r0['nqpk']} | " + " | ".join(cells) + " |")
        L += ["", "_Values are `t(16×16)/t(32×32)`: 32×32 is that many times faster._", ""]

if PR:
    L += [
        "## 2d. Why the decode margin explodes off the buffer-op path", "",
        "Buffer ops address through a **32-bit descriptor offset**, so the wrapper disables "
        "them once the KV cache passes `MAX_INT32`:", "",
        "```python",
        "kv_size = k.nelement() * k.element_size()",
        "USE_LOAD_BUFFER_OP = kv_size <= MAX_INT32        # 2 GB",
        "```", "",
        "The non-buffer path widens the block index to 64-bit (`blk.to(gl.int64)`), which "
        "costs registers. **MHA reaches that threshold 8× sooner than GQA-8**, since the "
        "cache scales with `num_kv_heads` — which is why this study surfaced it and the "
        "earlier GQA-only ones did not.", "",
        "Compile-only VGPR/spill audit (`../mfma_tile_probes.py`), bf16, TILE=64:", "",
        "| BLOCK_M | MFMA | nw | buffer ops on | buffer ops off |",
        "|--:|--:|--:|:--|:--|",
    ]
    for hs in (128, 64):
        if hs not in PR:
            continue
        rows = [r for r in PR[hs]["reg"] if r["dtype"] == "bf16" and r["tile"] == 64]
        if hs == 64:
            L.append(f"| _HEAD_SIZE=64_ | | | | |")
        for (BM, mfma, nw) in [(16, 16, 1), (32, 32, 1), (32, 16, 2), (64, 32, 2), (64, 16, 4)]:
            g2 = {r["buffer_ops"]: r for r in rows
                  if (r["block_m"], r["mfma"], r["num_warps"]) == (BM, mfma, nw)}
            if len(g2) != 2:
                continue
            def cell(r):
                if not r.get("ok"):
                    return f"`{r['reason']}`"
                s = f"{r['vgpr']} vgpr"
                return s + (f", **{r['spill']} spills**, {r['scratch']} B scratch"
                            if r["spill"] > 0 else ", 0 spills")
            pre = "" if hs == 128 else "_(HS64)_ "
            L.append(f"| {pre}{BM} | {mfma} | {nw} | {cell(g2[True])} | {cell(g2[False])} |")
    L += [
        "",
        "**One configuration spills: `BLOCK_M=32 / MFMA_DIM=32 / num_warps=1` at "
        "HEAD_SIZE=128** — and it is the only one that collapses in the timings below. The "
        "mechanism: a 32×128 fp32 accumulator held by a *single* warp is 64 VGPRs of "
        "accumulator alone, which already pins that config at the 256-VGPR ceiling with "
        "buffer ops on. Dropping them adds 64-bit addressing (worth ~59 VGPRs elsewhere — "
        "visible in the 16/16/1 row going 189→248) with no headroom left, so the allocator "
        "spills into the inner KV loop.", "",
        "It is specific to that combination, not to 32×32 or to BLOCK_M=32: spreading the "
        "same tile over two warps (`64/32/2`) avoids it, and at HEAD_SIZE=64 the accumulator "
        "halves and nothing exceeds ~170 VGPRs — which is exactly why the HS64 cells in 2c "
        "stay tied.", "",
    ]
    if PR.get(128, {}).get("bufop"):
        b = PR[128]["bufop"]
        L += ["Timed on a cache small enough that **both** settings are legal, so this "
              "isolates the code path from the cache size (bf16, HS128, "
              f"C{b[0]['C']} ctx{b[0]['ctx']} 64/{b[0]['Hkv']} — "
              f"{b[0]['bytes']/2e9:.2f} GB per k/v tensor, which is what the guard tests):",
              "",
              "| BLOCK_M | MFMA | nw | buffer ops on | buffer ops off | penalty |",
              "|--:|--:|--:|--:|--:|--:|"]
        for r in b:
            pen = r["us_nobufop"] / r["us_bufop"]
            mark = " **← spills**" if pen > 1.5 else ""
            L.append(f"| {r['block_m']} | {r['mfma']} | {r['num_warps']} | "
                     f"{r['us_bufop']:.1f} µs | {r['us_nobufop']:.1f} µs | "
                     f"**{pen:.2f}×**{mark} |")
        worst = max(r["us_nobufop"] / r["us_bufop"] for r in b)
        HEAD["spill_pen"] = f"{worst:.1f}×"
        L += ["",
              "So the practical rule is not \"16×16 is 3% better\". It is: **once the KV "
              "cache passes 2 GB at HEAD_SIZE=128, the 32×32 decode tile falls off a "
              f"register cliff and 16×16 is {worst:.1f}× faster.** Nothing in the shipped wrapper "
              "selects the spilling config today (bf16 decode uses MFMA_DIM=16; fp8 has "
              "lower register pressure — see 4), but collapsing decode onto a single 32×32 "
              "path would walk straight into it.", ""]

L += [
    "## 3. K_WIDTH sensitivity",
    "",
    "`K_WIDTH` is the dot-operand width (elements per lane per fetch) for the QK and PV dots. "
    "It is nearly free to get wrong in decode and expensive to get wrong in prefill:",
    "",
]

L += ["| dataset | dtype | K_WIDTH_QK → | " + " | ".join(str(k) for k in (4, 8, 16, 32)) + " |",
      "|---|:--|:--|--:|--:|--:|--:|"]
for n, lab, hs in DS:
    for dt in ("bf16", "fp8"):
        for phase in ("prefill", "decode"):
            row = []
            for kq in (4, 8, 16, 32):
                c = [r for r in sel(n, phase, dt, 32) if r["kq"] == kq and correct(r)]
                row.append(f"{min(r['time_us'] for r in c):.1f}" if c else "–")
            L.append(f"| {lab} | {dt} | {phase} (32×32, best kp) | " + " | ".join(row) + " |")
L.append("")

L += [
    "- **Prefill bf16 needs `K_WIDTH_QK ≥ 8`**: dropping to 4 costs 14–20% "
    "(291 vs 242 µs at HS128/3.6). 8 and 16 are within noise of each other — the kernel's 8 is "
    "fine.",
    "- **Prefill fp8 wants `K_WIDTH_QK = 16`, not 8**: 170 vs 194 µs at HS128/3.6 (**14%**), "
    "168 vs 186 on 3.8, 108 vs 113 at HS64. The kernel already uses 16. ",
    "- **`K_WIDTH_PV` prefers 4 for bf16 prefill** (the kernel's value); 16 costs 2–6%.",
    "- **Decode is flat in both K_WIDTHs** (spread ≤ 2%, no consistent winner across versions) — "
    "again the bandwidth bound.",
    "",
    "So the checked-in K_WIDTHs are at or within noise of optimal everywhere they are reachable. "
    "No change recommended.",
    "",
    "## 4. fp8 16×16×128 — a TILE_SIZE constraint, not a layout bug",
    "",
    "**This section corrects the previous revision of this study**, which concluded that fp8 "
    "16×16 \"is numerically wrong in every configuration that lowers\" and needed a "
    "dot-operand layout fix. The real constraint is the **K dimension of the two dots**, and "
    "with it satisfied the instruction lowers *and* is numerically correct.",
    "",
    "The two dots reduce over different axes:",
    "",
    "| dot | shapes | K dimension |",
    "|---|:--|:--|",
    "| QK | `[BLOCK_M, HEAD_SIZE] × [HEAD_SIZE, TILE_SIZE]` | **HEAD_SIZE** |",
    "| PV | `[BLOCK_M, TILE_SIZE] × [TILE_SIZE, HEAD_SIZE]` | **TILE_SIZE** |",
    "",
    "**Each dot's K must be a multiple of the MFMA instruction's K.** Both dots in the "
    "QK→PV chain share one `#mma` layout, so one instruction shape has to satisfy both:",
    "",
    "| instruction | instr K | requires |",
    "|---|--:|:--|",
    "| `v_mfma_scale_f32_16x16x128_f8f6f4` | 128 | `HEAD_SIZE % 128 == 0` **and** `TILE_SIZE % 128 == 0` |",
    "| `v_mfma_scale_f32_32x32x64_f8f6f4` | 64 | multiples of 64 — satisfied at HS 64/128, TILE 64 |",
    "| `v_mfma_f32_16x16x32_bf16` | 32 | satisfied by every tile this kernel uses |",
    "",
    "That is why this is an **fp8-only** problem: the fp8 matrix instructions have 4× the K "
    "depth of their bf16 counterparts (128 vs 32, 64 vs 16), so they outrun the kernel's tile "
    "dimensions. `MFMA_DIM=16` + e4m3 resolves to the only 16×16 fp8 scaled MFMA there is, "
    "`16x16x128`, and nothing checks that both dots have enough K.",
    "",
    "The failure surfaces in the **scale operands**, which is why it aborts rather than "
    "erroring cleanly. Scaled MFMA carries one scale byte per 32 elements of K, so a K=128 "
    "instruction consumes 4 scale groups per row. Captured from the failing compiles:",
    "",
    "| config | QK dot | PV dot | short one |",
    "|---|:--|:--|:--|",
    "| HS128, TILE=64 | `16x128 × 128x64`, scale `16x4` ✓ | `16x64 × 64x128`, scale `16x2` ✗ | "
    "**PV** (K = TILE_SIZE) |",
    "| HS64, TILE=128 | `16x64 × 64x128`, scale `16x2` ✗ | `16x128 × 128x64`, scale `16x4` ✓ | "
    "**QK** (K = HEAD_SIZE) |",
    "| HS128, TILE=128 | 4 groups ✓ | 4 groups ✓ | neither — compiles |",
    "",
    "The lowering builds a `SmallVector` of per-group scale values and indexes it by the "
    "instruction's group count, reading element [2] of a 2-element vector: "
    "`SmallVector.h:293: Assertion 'idx < size()' failed` inside "
    "`ConvertTritonAMDGPUToLLVM`. An unchecked out-of-bounds index, not a shape validator — "
    "hence a hard process abort that a `try/except` cannot catch.",
    "",
    "For contrast, the working 32×32 path picks `instrShape = [32, 32, 64]` and its PV dot is "
    "`32x64 × 64x128` with scale `32x2`: 2 groups needed, 2 supplied.",
    "",
    "This also re-reads the old finding that \"the gate is the PV dot, not the QK dot\": the "
    "right observation, attributed to the wrong knob (`K_WIDTH` instead of `TILE_SIZE`) — and "
    "it is only the PV dot at HEAD_SIZE=128; at HEAD_SIZE=64 it is the QK dot.",
    "",
]

if PR:
    L += ["Compile matrix for fp8 `MFMA_DIM=16` (`../mfma_tile_probes.py`):", "",
          "| | TILE=64 | TILE=128 |", "|---|:--|:--|"]
    for hs in (128, 64):
        if hs not in PR:
            continue
        cells = []
        for tile in (64, 128):
            r = next((x for x in PR[hs]["reg"] if x["dtype"] == "fp8" and x["tile"] == tile
                      and x["mfma"] == 16 and x["block_m"] == 16), None)
            if r is None:
                cells.append("–")
            elif r.get("ok"):
                cells.append(f"✓ compiles, {r['vgpr']} vgpr")
            else:
                cells.append(f"✗ `{r['reason']}`")
        L.append(f"| **HEAD_SIZE={hs}** | {cells[0]} | {cells[1]} |")
    L += ["", "At HEAD_SIZE=64 it can never lower — the QK dot's K *is* the head size, so no "
          "`TILE_SIZE` rescues it. HS64 is therefore locked to `32x32x64`, which is the "
          "correct fit there anyway (K=64 matches both dots exactly).", ""]

if PR.get(128, {}).get("fp8_tile"):
    L += ["Correctness and speed once it does lower (fp8 decode, HEAD_SIZE=128, "
          "`C64 ctx8192 64/8`, `rel` vs an fp32 torch reference):", "",
          "| TILE | MFMA | BLOCK_M | time | TB/s | rel err | M-utilisation |",
          "|--:|:--|--:|--:|--:|--:|--:|"]
    for r in PR[128]["fp8_tile"]:
        if r.get("ok"):
            L.append(f"| {r['tile']} | {INSTR[('fp8', r['mfma'])]} | {r['block_m']} | "
                     f"{r['time_us']:.1f} µs | {r['tbs']:.2f} | {r['rel']:.2%} | "
                     f"{r['m_util']:.0%} |")
        else:
            L.append(f"| {r['tile']} | {INSTR[('fp8', r['mfma'])]} | {r['block_m']} | "
                     f"**does not lower** | – | – | {r['m_util']:.0%} |")
    ok16 = next((r for r in PR[128]["fp8_tile"]
                 if r["tile"] == 128 and r["mfma"] == 16 and r.get("ok")), None)
    ok32 = next((r for r in PR[128]["fp8_tile"]
                 if r["tile"] == 128 and r["mfma"] == 32 and r.get("ok")), None)
    if ok16 and ok32:
        HEAD["fp8_gain"] = f"{(ok32['time_us']/ok16['time_us'] - 1)*100:.0f}%"
        L += ["",
              f"At TILE=128 the 16×16 tile is **correct** — {ok16['rel']:.2%} relative error, "
              f"identical to 32×32's {ok32['rel']:.2%}, i.e. pure fp8 quantisation and not a "
              "layout error. That is the real result here: the instruction is usable, and the "
              "previous revision's \"needs a dot-operand layout fix\" is withdrawn.", "",
              f"The **speed** gain is not established. This run shows "
              f"{ok32['time_us']/ok16['time_us']:.2f}×; three repeats of this cell spanned "
              f"2–6%, i.e. at or below the ~{NOISE:.0%} noise floor. Doubling M-utilisation "
              "(50% vs 25%) simply does not buy much on a kernel that is bandwidth-bound — "
              "both tiles sit near the ~6.4 TB/s roof.", "",
              "**Practical read:** the branch `fp8 and HEAD_SIZE ≥ 128 and TILE_SIZE ≥ 128 → "
              "MFMA_DIM=16` is safe and costs nothing where 128-token pages are already in "
              "use, but it should be justified as *unlocking the tile*, not as a speedup. "
              "Do not change page size to get it. If a compute-bound fp8 use of this tile "
              "appears (larger BLOCK_M, or prefill), it is worth re-measuring there.", ""]

L += [
    "### The old K_WIDTH matrix (for the record)",
    "",
    "Compile / correctness over the `K_WIDTH` grid at `TILE_SIZE=64`, where the instruction "
    "cannot work for the reason above (`•` = compiles and is correct, `✗` = fails to lower, "
    "`!` = compiles but numerically wrong):",
    "",
]

L += ["| dataset | phase | " + " | ".join(f"kq{kq}/kp{kp}" for kq in (8, 16, 32)
                                          for kp in (8, 16, 32)) + " |",
      "|---|:--|" + ":--:|" * 9]
for n, lab, hs in DS:
    for phase in ("decode", "prefill"):
        cells = []
        for kq in (8, 16, 32):
            for kp in (8, 16, 32):
                r = next((x for x in sel(n, phase, "fp8", 16)
                          if x["kq"] == kq and x["kp"] == kp), None)
                cells.append("✗" if (r is None or "error" in r) else ("•" if correct(r) else "!"))
        L.append(f"| {lab} | {phase} | " + " | ".join(cells) + " |")
L.append("")

relbad = sorted({round(r["rel"] * 100, 1) for n, _, _ in DS for ph in ("decode", "prefill")
                 for r in sel(n, ph, "fp8", 16) if "error" not in r and r["rel"] >= 0.05})
L += [
    f"Every cell that lowers here is numerically wrong (relative-mean error "
    f"{'%, '.join(str(x) for x in relbad)}%), and the cells that lower at all are the ones "
    "where `K_WIDTH_PV = 32` happens to let the compiler emit *something* despite the K "
    "mismatch. None of this is a `K_WIDTH` problem — raise `TILE_SIZE` and the whole grid "
    "becomes moot.",
    "",
    "## 5. Bottom line — what the heuristic should do",
    "",
    "| case | choose | why | margin |",
    "|---|:--|:--|:--|",
    "| **decode, bf16** | **16×16**, every nqpk incl. MHA | No nqpk branch needed: the ratio "
    "barely moves the answer (2c). The real reason is 2d — the 32×32 alternative falls off a "
    "register-spill cliff once the cache passes 2 GB at HS128. | 0–5% on the buffer-op path; "
    f"**{HEAD['off_path']}** off it |",
    "| **decode, fp8** | **32×32** at TILE=64; 16×16 only if `HEAD_SIZE ≥ 128 and TILE_SIZE ≥ "
    "128` | 16×16×128 cannot lower at TILE=64 — the PV dot only supplies K=64 (4). Where it "
    f"does lower it is correct; the speed gain ({HEAD['fp8_gain']}) is within noise. | "
    "correctness, not perf |",
    "| **prefill, both** | **32×32** | Full M utilisation at BLOCK_M=128 either way, so the 2× "
    f"MACs/issue and halved warp count both land. nqpk-independent. | **{HEAD['prefill']}** |",
    "| **K_WIDTH** | leave as shipped | QK≥8 (bf16) / QK=16 (fp8) matter in prefill; the rest "
    "is noise. | — |",
    "",
    "The shipped heuristic in "
    "`aiter/ops/triton/_gluon_kernels/gfx950/attention/unified_attention.py` — decode 16×16 "
    "(fp8 forced to 32×32), prefill 32×32 — is **correct as written**, including for MHA. "
    "What this revision adds is the reason it is correct, and two things to not break:",
    "",
    "1. **Do not collapse decode onto a single 32×32 path.** `MFMA_DIM=32 + num_warps=1` at "
    f"HEAD_SIZE=128 spills 30 VGPRs whenever buffer ops are off (KV > 2 GB), costing {HEAD['spill_pen']}. "
    "Nothing selects it today; a \"simplification\" would.",
    "2. **`BLOCK_M` does not cost workgroups in decode.** The `ALL_DECODE` grid is "
    "`(NUM_SEQS, NUM_KV_HEADS, NUM_SPLITS)` — `total_query_blocks = NUM_SEQS` regardless of "
    "`BLOCK_M`. Raising it wastes M lanes and accumulator registers but does *not* reduce "
    "occupancy or force more splits. (In prefill it does: "
    "`total_query_blocks = q.shape[0] // BLOCK_Q + NUM_SEQS`.)",
    "",
    "Open, not measured here: whether `TILE_SIZE=128` shifts the **bf16** tile conclusion — "
    "every bf16 number in this study is at `TILE_SIZE=64`, and the register audit in 2d shows "
    "bf16 already spilling at TILE=128 for several configs.",
    "",
    "## Appendix — every measurement",
    "",
    "| dataset | phase | dtype | MFMA | BLOCK_M | nw | kq | kp | time µs | ×tri | rel err | status |",
    "|---|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|:--|",
]
for n, lab, hs in DS:
    for r in R[n]:
        if "error" in r:
            L.append(f"| {lab} | {r['phase']} | {r['dtype']} | {INSTR[(r['dtype'], r['mfma'])]} | "
                     f"{r['block_m']} | {r['num_warps']} | {r['kq']} | {r['kp']} | – | – | – | "
                     f"`{r['error']}` |")
        else:
            st = "ok" if correct(r) else "**WRONG**"
            L.append(f"| {lab} | {r['phase']} | {r['dtype']} | {INSTR[(r['dtype'], r['mfma'])]} | "
                     f"{r['block_m']} | {r['num_warps']} | {r['kq']} | {r['kp']} | "
                     f"{r['time_us']:.1f} | {r['triton_us'] / r['time_us']:.2f}× | "
                     f"{r['rel']:.1%} | {st} |")
L.append("")

open(f"{BASE}/mfma_shapes.md", "w").write("\n".join(L) + "\n")
print(f"wrote {BASE}/mfma_shapes.md")
