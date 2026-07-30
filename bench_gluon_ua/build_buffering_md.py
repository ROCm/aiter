"""Aggregate buffering_<ver>.json (3.6.0 / 3.8.0) into decode_buffering.md:
gluon decode KV staging as direct-to-registers vs single-LDS vs double-LDS buffer,
with the triton baseline, on triton 3.6.0 and 3.8.0 (ToT)."""
import json

BASE = "/app/aiter/bench_gluon_ua"
VERS = ["3.6.0", "3.8.0"]
VLABEL = {"3.6.0": "3.6.0", "3.8.0": "3.8.0 (ToT)"}
MODES = ["triton", "registers", "single_lds", "double_lds"]
MLABEL = {"triton": "triton", "registers": "registers (nb0)",
          "single_lds": "single-LDS (nb1)", "double_lds": "double-LDS (nb2)"}

data, full = {}, {}
for v in VERS:
    try:
        recs = json.load(open(f"{BASE}/buffering_{v}.json"))
    except FileNotFoundError:
        continue
    for r in recs:
        data[(v, r["C"], r["ctx"], r["Hq"], r["Hkv"], r["mode"])] = r
        full[v] = r.get("ver", v)
HAVE = [v for v in VERS if any(k[0] == v for k in data)]

SHAPES = [(C, ctx, Hq, Hkv) for ctx in (1024, 8192) for (Hq, Hkv) in ((64, 8), (8, 1))
          for C in (16, 32, 64, 128)]


def sl(C, ctx, Hq, Hkv):
    return f"C{C} ctx{ctx} {Hq}/{Hkv}"


def g(v, sh, mode):
    return data.get((v, *sh, mode))


def gbps_table(v):
    L = [f"### {VLABEL[v]} — decode bandwidth (GB/s); parenthetical = speedup vs triton", "",
         "| shape | S | triton | registers | single-LDS | double-LDS |",
         "|---|--:|--:|--:|--:|--:|"]
    for sh in SHAPES:
        tri = g(v, sh, "triton")
        cells = []
        for mode in MODES[1:]:
            r = g(v, sh, mode)
            if not r or r.get("gbps") is None:
                cells.append(r.get("error", "–") if r else "–"); continue
            cells.append(f"{r['gbps']:.0f} ({tri['time_us'] / r['time_us']:.2f}×)")
        S = g(v, sh, "registers")["split"] if g(v, sh, "registers") else "?"
        L.append(f"| {sl(*sh)} | {S} | {tri['gbps']:.0f} | " + " | ".join(cells) + " |")
    L.append("")
    return L


def resource_table():
    # resources depend on (mode, heads, split) but not on C/ctx for decode; sample
    # one GQA (C128 ctx8192 64/8) and one MQA (C32 ctx8192 8/1).
    L = ["### Kernel resources (compiled): LDS / VGPRs / spills", "",
         "| sample shape | mode | LDS (KB) | VGPRs | spills |", "|---|---|--:|--:|--:|"]
    v = HAVE[-1]
    for sh in [(128, 8192, 64, 8), (32, 8192, 8, 1)]:
        for mode in MODES[1:]:
            r = g(v, sh, mode)
            if not r or r.get("lds") is None:
                continue
            L.append(f"| {sl(*sh)} | {MLABEL[mode]} | {r['lds'] / 1024:.1f} | {r['regs']} | {r['spills']} |")
    L.append(f"\n_(resources from triton {VLABEL[v]}; the same three binaries are used on all versions.)_\n")
    return L


hdr = [
    "# gfx950 gluon decode — KV buffering: registers vs single-LDS vs double-LDS",
    "",
    "> **Result up front:** the direct-to-registers prototype is **spill-bound and ~5–8× slower**, so",
    "> it was **removed** from the kernel. Single-LDS (32 KB, occ 2) is the shipping decode staging;",
    "> the register numbers below are the (now-deleted) prototype, kept as the justification.",
    "",
    "Three ways the gluon decode kernel stages each K/V tile before the MFMA dot:",
    "",
    "- **registers (nb=0)** — `gl.load` each tile from global **straight into the MFMA dot-operand",
    "  layout**; no LDS staging, no async pipeline. *(Prototype, removed — see result above.)*",
    "- **single-LDS (nb=1)** — one K + one V tile in a 32 KB LDS buffer, `thread_barrier` between tiles",
    "  (`attention_loop_single_buffer`). Occupancy 2 wg/CU.",
    "- **double-LDS (nb=2)** — K + V double-buffered in 64 KB LDS, async prefetch pipelined",
    "  (`attention_loop_standard`). Occupancy 1 wg/CU.",
    "",
    "Same right-sized split count on all three (S=1 ⇒ non-split, no reduce). bf16, HEAD_SIZE=128,",
    "TILE_SIZE=64, causal, BLOCK_M=16 / 16×16 MFMA / num_warps=1. 512 MB L2 flush per iter,",
    "torch.profiler, per-kernel filter. triton = same-version reference (its own heuristic split).",
    f"Correctness (gluon vs triton) ≤ ~5e-4 on all cells. Versions present: {', '.join(VLABEL[v] for v in HAVE)}.",
    "",
]
out = hdr + resource_table()
for v in HAVE:
    out += gbps_table(v)

# takeaways
best_counts = {m: 0 for m in MODES[1:]}
for v in HAVE:
    for sh in SHAPES:
        cand = [(m, g(v, sh, m)) for m in MODES[1:] if g(v, sh, m) and g(v, sh, m).get("gbps")]
        if cand:
            best_counts[max(cand, key=lambda x: x[1]["gbps"])[0]] += 1
out += [
    "## Takeaways", "",
    "- **direct-to-registers frees all LDS (0 KB) but spills catastrophically** — loading whole K/V",
    "  tiles into VGPRs blows the register file (≈480 spills vs 0 for the LDS paths), so the freed",
    "  occupancy is more than eaten by scratch traffic. It is the **slowest** mode on essentially every",
    "  shape.",
    "- **single-LDS (32 KB, occ 2) is the best or tied-best** decode staging: half the LDS of the",
    "  double buffer (2× occupancy) with no spills, and the cross-WG overlap replaces the lost",
    "  in-WG async pipeline.",
    "- **double-LDS (64 KB, occ 1)** is competitive only where there are already enough workgroups to",
    "  hide latency at occupancy 1 (large GQA); elsewhere the halved occupancy costs it.",
    f"- Best-mode tally across measured cells: " +
    ", ".join(f"{MLABEL[m]} {best_counts[m]}" for m in MODES[1:]) + ".",
    "- Note: `gl.thread_barrier` (needed by single-LDS) is absent on triton 3.7/3.8, so the shipping",
    "  decode default is double-LDS there; single-LDS numbers here on 3.8 use the ToT build that has it.",
    "",
]
open(f"{BASE}/decode_buffering.md", "w").write("\n".join(out) + "\n")
print("wrote decode_buffering.md")
