"""Build decode_stride_dyn_vs_constexpr.md: older dynamic-stride kernel (HEAD,
loader-based runtime strides) vs the new constexpr config-routed kernel, on
triton 3.6.0 and 3.8.0, decode nb=2 (the config the dynamic version can compile
in). The dynamic-stride variant of the *new* structure does not compile on either
version (see note), so HEAD is the compilable dynamic baseline."""
import json

BASE = "/app/aiter/bench_gluon_ua"


def load(f):
    try:
        return {(r["C"], r["ctx"], r["Hq"], r["Hkv"]): r for r in json.load(open(f"{BASE}/{f}"))}
    except FileNotFoundError:
        return {}


FILES = {
    ("3.8.0", "dyn"): "ab_cmp_3.8.0_dynamicHEAD_nb2.json",
    ("3.8.0", "cxpr"): "ab_cmp_3.8.0_constexpr.json",
    ("3.6.0", "dyn"): "ab_cmp_3.6.0_dynamicHEAD_nb2.json",
    ("3.6.0", "cxpr"): "ab_cmp_3.6.0_constexpr_nb2.json",
}
data = {k: load(v) for k, v in FILES.items()}
# constexpr nb=1 (shipping decode default) for reference
ref_nb1 = {"3.8.0": load("ab_B_configrouted.json"), "3.6.0": load("ab_cmp_3.6.0_constexpr_nb1.json")}
SHAPES = [(C, ctx, Hq, Hkv) for ctx in (1024, 8192) for (Hq, Hkv) in ((64, 8), (8, 1))
          for C in (16, 32, 64, 128)]


def sl(s):
    return f"C{s[0]} ctx{s[1]} {s[2]}/{s[3]}"


def reg(r):
    return str(r["regs"]) if r and r.get("regs") is not None else "–"


def table(ver):
    dyn = data[(ver, "dyn")]; cx = data[(ver, "cxpr")]
    L = ["| shape | S | dyn GB/s | cxpr GB/s | Δ% | dyn VGPR | cxpr VGPR |",
         "|---|--:|--:|--:|--:|--:|--:|"]
    ds, dr = [], []
    for s in SHAPES:
        d = dyn.get(s); c = cx.get(s)
        if not d or not c:
            continue
        dd = 100 * (c["gbps"] - d["gbps"]) / d["gbps"]; ds.append(dd)
        if d.get("regs") and c.get("regs"):
            dr.append(d["regs"] - c["regs"])
        L.append(f"| {sl(s)} | {c['split']} | {d['gbps']:.0f} | {c['gbps']:.0f} | {dd:+.1f} | {reg(d)} | {reg(c)} |")
    L.append("")
    return L, (sum(ds) / len(ds) if ds else 0), (min(dr) if dr else 0), (max(dr) if dr else 0)


out = [
    "# gfx950 gluon decode — older dynamic-stride vs new constexpr-stride kernel (triton 3.6.0 & 3.8.0)",
    "",
    "**old** = pre-refactor kernel (HEAD `033550e66`): KV/block-table strides are runtime `gl.int32`",
    "args threaded through the loader. **new** = current kernel: strides are `gl.constexpr` on",
    "`AttentionConfig` (gfx1250-style, baked in). Decode grid, right-sized split, **nb=2** (double-buffer;",
    "the config the dynamic kernel can compile in), 512 MB L2 flush, torch.profiler. VGPRs from the",
    "compiled kernel (0 spills). Δ% = constexpr vs dynamic bandwidth.",
    "",
    "> **Why nb=2 / HEAD:** the dynamic-stride variant of the *new* structure (constexpr routing +",
    "> the single-buffer/ALL_DECODE decode-path changes) **fails to compile on both 3.6.0 and 3.8.0**",
    "> — `builtin.unrealized_conversion_cast` fails to lower. Constexpr strides are what let the current",
    "> kernel compile. HEAD is the last dynamic-stride version that compiles, so it's the baseline here.",
    "",
]
for ver in ("3.8.0", "3.6.0"):
    tbl, meanΔ, rlo, rhi = table(ver)
    label = "3.8.0 (ToT)" if ver == "3.8.0" else "3.6.0"
    out += [f"## triton {label} — dynamic (old) vs constexpr (new), nb=2",
            f"_bandwidth mean Δ {meanΔ:+.1f}%; constexpr uses ~{rlo}–{rhi} fewer VGPRs_", ""]
    out += tbl

out += ["## Reference: new constexpr kernel at the shipping decode default (nb=1, single-buffer)",
        "", "| shape | S | 3.6.0 GB/s / VGPR | 3.8.0 GB/s / VGPR |", "|---|--:|--:|--:|"]
for s in SHAPES:
    a = ref_nb1["3.6.0"].get(s); b = ref_nb1["3.8.0"].get(s)
    if not a or not b:
        continue
    out.append(f"| {sl(s)} | {a['split']} | {a['gbps']:.0f} / {reg(a)} | {b['gbps']:.0f} / {reg(b)} |")
out.append("")

out += [
    "## Takeaways", "",
    "- **Bandwidth is unchanged** (mean Δ within noise on both versions) — decode is bandwidth-bound",
    "  with LDS-limited occupancy, so baking the strides in doesn't move throughput.",
    "- **Constexpr uses fewer VGPRs**: ~11 fewer on 3.8.0 (≈175→164) and ~30 fewer on 3.6.0",
    "  (≈222→190). The 3.6 saving is larger because its (older) compiler spilled the runtime strides",
    "  into more registers. (This delta is the *net* old→new: constexpr strides + the decode-path",
    "  simplifications; the isolated pure-stride effect measured earlier was ~neutral / −8 VGPR.)",
    "- **Constexpr is required by the current kernel**, not just an optimization: the dynamic-stride",
    "  variant of the new structure won't compile on 3.6 or 3.8 (`unrealized_conversion_cast`). Keeping",
    "  strides constexpr (gfx1250-style) is the working, and correct-on-large-KV, choice.",
    "",
]
open(f"{BASE}/decode_stride_dyn_vs_constexpr.md", "w").write("\n".join(out) + "\n")
print("wrote decode_stride_dyn_vs_constexpr.md")
