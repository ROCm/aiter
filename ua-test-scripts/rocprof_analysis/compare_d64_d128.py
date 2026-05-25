#!/usr/bin/env python3
"""
Aggregate rocprofv3 PMC CSVs from rocprof_d64_vs_d128.sh into a side-by-side
table that highlights the d=128 hot spots (LDS bank conflicts, extra LDS
instructions, etc.).

CSV is in tidy format: one row per (kernel_launch, counter). We collapse on
kernel name == anything containing `UnifiedAttentionKernel`, then median
across launches per counter.

Usage:
    python3 compare_d64_d128.py <root_dir>
"""
import csv
import sys
from pathlib import Path
from collections import defaultdict


def is_ua_kernel(name: str) -> bool:
    return "UnifiedAttention" in name


def load_pmc(path: Path) -> dict[str, list[float]]:
    """{counter_name -> [per-launch values]} for CK UA kernel rows."""
    out = defaultdict(list)
    if not path.exists():
        print(f"WARNING: {path} missing", file=sys.stderr)
        return out
    with open(path) as fh:
        rdr = csv.DictReader(fh)
        # Group by (Dispatch_Id, Counter_Name) to be safe — but in practice
        # tidy rows are one (kernel, counter) per row already.
        for row in rdr:
            name = row.get("Kernel_Name", "")
            if not is_ua_kernel(name):
                continue
            cname = row.get("Counter_Name")
            cval = row.get("Counter_Value")
            if cname is None or cval is None:
                continue
            try:
                out[cname].append(float(cval))
            except (TypeError, ValueError):
                pass
    return out


def median(xs):
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def warmup_drop(xs, n=3):
    return xs[n:] if len(xs) > n else xs


def aggregate(root: Path, leaf: str) -> dict[str, float]:
    out = {}
    for phase in ("p1_compute", "p2_lds", "p3_cache"):
        csv_path = root / leaf / phase / "pmc_counter_collection.csv"
        data = load_pmc(csv_path)
        for k, vs in data.items():
            vs = warmup_drop(vs)
            out[k] = median(vs)
    return out


def fmt(v):
    if v != v:
        return "n/a"
    av = abs(v)
    if av >= 1e9:
        return f"{v/1e9:.2f}G"
    if av >= 1e6:
        return f"{v/1e6:.2f}M"
    if av >= 1e3:
        return f"{v/1e3:.2f}K"
    if av >= 1:
        return f"{v:.2f}"
    return f"{v:.3g}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = Path(sys.argv[1])
    leaf_a = sys.argv[2] if len(sys.argv) > 2 else "d64_b128"
    leaf_b = sys.argv[3] if len(sys.argv) > 3 else "d128_b128"
    print(f"\n# {leaf_a} vs {leaf_b}\n")
    a = aggregate(root, leaf_a)
    b = aggregate(root, leaf_b)

    # Derived metrics (per launch).
    derived = {}
    label_a = leaf_a.split("_")[0]
    label_b = leaf_b.split("_")[0]
    for label, vals in ((label_a, a), (label_b, b)):
        active = vals.get("GRBM_GUI_ACTIVE", float("nan"))
        bank_cycles = vals.get("SQ_LDS_BANK_CONFLICT", float("nan"))
        addr_cycles = vals.get("SQ_LDS_ADDR_CONFLICT", float("nan"))
        lds_idx_active = vals.get("SQ_LDS_IDX_ACTIVE", float("nan"))
        lds_insts = vals.get("SQ_INSTS_LDS", float("nan"))
        valu = vals.get("SQ_INSTS_VALU", float("nan"))
        mfma = vals.get("SQ_INSTS_MFMA", float("nan"))
        salu = vals.get("SQ_INSTS_SALU", float("nan"))
        vmem = vals.get("SQ_INSTS_VMEM", float("nan"))
        tcc_hit = vals.get("TCC_HIT_sum", float("nan"))
        tcc_miss = vals.get("TCC_MISS_sum", float("nan"))
        # The active cycles counter sums across all CUs on gfx950 (256 CUs)
        # for derived %s.
        CU_NUM = 256.0
        d = {}
        d["bank_conflict_%_lds_active"] = (
            100.0 * bank_cycles / lds_idx_active if lds_idx_active else float("nan")
        )
        d["bank_conflict_%_gpu_active"] = (
            100.0 * bank_cycles / (active * CU_NUM) if active else float("nan")
        )
        d["addr_conflict_%_lds_active"] = (
            100.0 * addr_cycles / lds_idx_active if lds_idx_active else float("nan")
        )
        d["lds_util_%"] = (
            100.0 * lds_idx_active / (active * CU_NUM) if active else float("nan")
        )
        d["valu/mfma"] = valu / mfma if mfma else float("nan")
        d["lds/mfma"] = lds_insts / mfma if mfma else float("nan")
        d["salu/mfma"] = salu / mfma if mfma else float("nan")
        d["vmem/mfma"] = vmem / mfma if mfma else float("nan")
        d["tcc_hit_rate_%"] = (
            100.0 * tcc_hit / (tcc_hit + tcc_miss)
            if (tcc_hit + tcc_miss) > 0
            else float("nan")
        )
        derived[label] = d

    print("\n## Raw PMC counters (median across post-warmup launches)\n")
    print(f"| counter | {label_a} | {label_b} | {label_b}/{label_a} |")
    print("|---|---:|---:|---:|")
    for k in sorted(set(a) | set(b)):
        va = a.get(k, float("nan"))
        vb = b.get(k, float("nan"))
        try:
            ratio = vb / va if va else float("nan")
            ratio_s = f"{ratio:.2f}x"
        except Exception:
            ratio_s = "n/a"
        print(f"| {k} | {fmt(va)} | {fmt(vb)} | {ratio_s} |")

    print("\n## Derived metrics\n")
    print(f"| metric | {label_a} | {label_b} | delta |")
    print("|---|---:|---:|---:|")
    for k in derived[label_a]:
        va = derived[label_a][k]
        vb = derived[label_b][k]
        try:
            delta_s = f"{vb - va:+.2f}"
        except Exception:
            delta_s = "n/a"
        print(f"| {k} | {va:.2f} | {vb:.2f} | {delta_s} |")


if __name__ == "__main__":
    main()
