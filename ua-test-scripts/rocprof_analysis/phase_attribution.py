#!/usr/bin/env python3
"""
Bucket PC samples into the unified-attention pipeline phases.

The kernel's steady-state main loop is structured as

    [phase0: COMPUTE / LOAD]
    s_barrier
    [phase1: LOAD / COMPUTE]
    s_barrier
    [phase2: COMPUTE / LOAD]
    s_barrier
    [phase3: LOAD / COMPUTE]
    s_barrier               -- (wraps around to next iteration)

(`cl_p=0` warps do the 4 alternations in COMPUTE / LOAD order; `cl_p=1`
warps do the opposite ordering.  Both groups hit the SAME s_barrier
instructions because barriers are block-wide.)

We don't have ASM_MARKER labels in the binary (they're asm comments), so
this script:

  1. Reads every distinct (code_object_offset, instruction) pair recorded
     in the rocprofv3 host-trap dump.
  2. Sorts those PCs and walks them; every time we hit an `s_barrier`
     instruction we close the current window and open a new one.
  3. Classifies each window by its dominant work:
        COMPUTE_QK  = MFMA-heavy, first half of an iteration (after we
                      see an alu1/softmax-style ds_write/ds_read mix)
        COMPUTE_PV  = MFMA-heavy, second half (after fmha_alu_D_upd
                      pk_mul pattern)
        LOAD_K      = buffer_load_*_lds heavy + the K_dram_window advance
                      (recognised by the page-table v_mad_i64_i32 chain)
        LOAD_V      = buffer_load_*_lds heavy + similar V pattern
        OTHER       = pre/post stage, fmha_post_process, epilog
  4. Sums the PC sample hits per window and per category, prints the
     answer ranked by hit count.

Usage:
  python3 phase_attribution.py <pcsamp_results.json> [--kernel-pat PAT]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict


def is_barrier(op: str) -> bool:
    return op.startswith("s_barrier")


def is_mfma(op: str) -> bool:
    return "mfma" in op


def is_vmem_load(op: str) -> bool:
    return (op.startswith("buffer_load") or op.startswith("global_load")
            or op.startswith("flat_load"))


def is_lds_write(op: str) -> bool:
    return op.startswith("ds_write")


def is_lds_read(op: str) -> bool:
    return op.startswith("ds_read")


def is_exp(op: str) -> bool:
    return op.startswith("v_exp_") or op.startswith("v_log_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_file")
    ap.add_argument("--kernel-pat", default="UnifiedAttentionKernel")
    ap.add_argument("--top-windows", type=int, default=12)
    args = ap.parse_args()

    with open(args.json_file) as f:
        data = json.load(f)
    node = data["rocprofiler-sdk-tool"][0]
    samples = node["buffer_records"]["pc_sample_host_trap"]
    insts = node["strings"]["pc_sample_instructions"]
    ksyms = node["kernel_symbols"]
    disps = node["buffer_records"]["kernel_dispatch"]
    disp_to_kid = {
        d["dispatch_info"]["dispatch_id"]: d["dispatch_info"]["kernel_id"]
        for d in disps
    }
    kid_name = {k["kernel_id"]: k["kernel_name"] for k in ksyms}

    pc_hits: Counter = Counter()
    pc_inst: dict = {}
    for s in samples:
        rec = s["record"]
        idx = s["inst_index"]
        if idx < 0:
            continue
        did = rec["dispatch_id"]
        kid = disp_to_kid.get(did, 0)
        if args.kernel_pat not in kid_name.get(kid, ""):
            continue
        pc = rec["pc"]
        key = (pc["code_object_id"], pc["code_object_offset"])
        pc_hits[key] += 1
        pc_inst[key] = insts[idx]

    if not pc_hits:
        print("No samples in target kernel.")
        return

    # Pick the code object with the most samples (=our kernel).
    per_cobj: dict[int, list[int]] = defaultdict(list)
    for (coid, off) in pc_hits:
        per_cobj[coid].append(off)
    main_coid = max(per_cobj, key=lambda c: sum(pc_hits[(c, o)] for o in per_cobj[c]))
    offs_sorted = sorted(per_cobj[main_coid])

    # Walk and split into windows at every s_barrier.
    windows: list[dict] = []  # each: {start, end, ops: Counter, hits, insts: list}
    cur = {
        "start": offs_sorted[0],
        "end": offs_sorted[0],
        "ops": Counter(),
        "hits": 0,
        "barrier_offs": [],
        "insts": [],
    }
    for off in offs_sorted:
        ins = pc_inst[(main_coid, off)]
        op = ins.split()[0]
        n = pc_hits[(main_coid, off)]
        cur["end"] = off
        cur["ops"][op] += n
        cur["hits"] += n
        cur["insts"].append((off, ins, n))
        if is_barrier(op):
            cur["barrier_offs"].append(off)
            windows.append(cur)
            cur = {
                "start": off + 4,
                "end": off + 4,
                "ops": Counter(),
                "hits": 0,
                "barrier_offs": [],
                "insts": [],
            }
    if cur["hits"]:
        windows.append(cur)

    # Classify each window using *non-barrier* samples only, so the
    # waiting-at-barrier tail doesn't drown out the real work signal.
    def classify(w):
        ops = w["ops"]
        mfma = sum(n for k, n in ops.items() if is_mfma(k))
        vmem = sum(n for k, n in ops.items() if is_vmem_load(k))
        lds_w = sum(n for k, n in ops.items() if is_lds_write(k))
        lds_r = sum(n for k, n in ops.items() if is_lds_read(k))
        exp_ = sum(n for k, n in ops.items() if is_exp(k))
        barr = sum(n for k, n in ops.items() if is_barrier(k))
        wait = sum(n for k, n in ops.items() if k.startswith("s_wait"))
        work = w["hits"] - barr - wait  # samples on real instructions
        if work <= 0:
            return "EMPTY"
        # Compute phase signatures, judged on real-work samples
        if mfma > 0.05 * work:
            if exp_ > 0.05 * work:
                return "COMPUTE_QK"   # softmax (v_exp) + QK MFMAs
            return "COMPUTE_PV"        # MFMAs without v_exp = PV
        if vmem > 0.02 * work or lds_w > 0.02 * work:
            return "LOAD_KV"           # async loads + LDS write/read prep
        return "OTHER"                  # pre/post stage etc.

    for w in windows:
        w["class"] = classify(w)
        ops = w["ops"]
        w["barrier_hits"] = sum(n for k, n in ops.items() if is_barrier(k))
        w["wait_hits"] = sum(n for k, n in ops.items() if k.startswith("s_wait"))
        w["work_hits"] = w["hits"] - w["barrier_hits"] - w["wait_hits"]

    # Aggregate per class
    cls_hits: Counter = Counter()
    cls_window_count: Counter = Counter()
    for w in windows:
        cls_hits[w["class"]] += w["hits"]
        cls_window_count[w["class"]] += 1
    total = sum(cls_hits.values())

    print(f"\nTotal recognised samples in main kernel (cobj {main_coid}): {total}")
    print(f"Total distinct PCs: {len(offs_sorted)}")
    print(f"Total windows (between s_barriers): {len(windows)}\n")

    # Aggregate work / barrier / wait separately per class
    cls_work: Counter = Counter()
    cls_barrier: Counter = Counter()
    cls_wait: Counter = Counter()
    for w in windows:
        cls_work[w["class"]] += w["work_hits"]
        cls_barrier[w["class"]] += w["barrier_hits"]
        cls_wait[w["class"]] += w["wait_hits"]

    total_work = sum(cls_work.values())
    total_bar = sum(cls_barrier.values())
    total_wait = sum(cls_wait.values())

    print("Per-class breakdown (samples; %% = of total)")
    print(f"  {'class':<12} {'#win':>5} {'work':>9} {'barrier':>9} {'wait':>9}"
          f" {'%work':>7} {'%bar':>7} {'%wait':>7} {'tot':>9} {'%tot':>7}")
    for cls, _ in sorted(cls_hits.items(), key=lambda kv: -kv[1]):
        nw = cls_window_count[cls]
        w_ = cls_work[cls]
        b_ = cls_barrier[cls]
        wt_ = cls_wait[cls]
        tot = w_ + b_ + wt_
        print(f"  {cls:<12} {nw:>5} {w_:>9} {b_:>9} {wt_:>9} "
              f"{w_/total*100:>6.2f}% {b_/total*100:>6.2f}% {wt_/total*100:>6.2f}% "
              f"{tot:>9} {tot/total*100:>6.2f}%")
    print(f"\n  TOTAL          work={total_work} ({total_work/total*100:.2f}%)"
          f"  barrier={total_bar} ({total_bar/total*100:.2f}%)"
          f"  wait={total_wait} ({total_wait/total*100:.2f}%)")
    print(f"  → fraction of time NOT doing useful work: "
          f"{(total_bar + total_wait)/total*100:.2f}%\n")

    # Top windows by hit count
    print(f"Top {args.top_windows} windows by sample count:")
    ranked = sorted(windows, key=lambda w: -w["hits"])
    for i, w in enumerate(ranked[: args.top_windows]):
        ops = w["ops"]
        mfma = sum(n for k, n in ops.items() if is_mfma(k))
        vmem = sum(n for k, n in ops.items() if is_vmem_load(k))
        lds_w = sum(n for k, n in ops.items() if is_lds_write(k))
        lds_r = sum(n for k, n in ops.items() if is_lds_read(k))
        exp_ = sum(n for k, n in ops.items() if is_exp(k))
        barr = sum(n for k, n in ops.items() if is_barrier(k))
        wait = sum(n for k, n in ops.items() if k.startswith("s_wait"))
        salu = sum(n for k, n in ops.items() if k.startswith("s_") and not is_barrier(k) and not k.startswith("s_wait"))
        valu = sum(n for k, n in ops.items()
                   if k.startswith("v_") and not is_mfma(k))
        print(f"\n  Window #{i+1}  class={w['class']:<11} hits={w['hits']:>6} ({w['hits']/total*100:.2f}%)")
        print(f"    offsets 0x{w['start']:08x} .. 0x{w['end']:08x}  size={w['end']-w['start']:>5}B  insts={len(w['insts'])}")
        cat_str = []
        for label, val in [("MFMA", mfma), ("VMEM", vmem), ("LDS_w", lds_w),
                           ("LDS_r", lds_r), ("VEXP", exp_), ("VALU", valu),
                           ("WAIT", wait), ("SALU", salu), ("BARRIER", barr)]:
            if val:
                cat_str.append(f"{label}:{val}")
        print(f"    inst mix:  " + "  ".join(cat_str))
        # Show top 3 hot instructions
        ranked_pc = sorted(w["insts"], key=lambda t: -t[2])[:4]
        for off, ins, n in ranked_pc:
            print(f"      0x{off:08x}  hits={n:>4}  {ins[:80]}")


if __name__ == "__main__":
    main()
