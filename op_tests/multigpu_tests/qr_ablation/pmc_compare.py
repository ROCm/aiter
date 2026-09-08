#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""EA write-request counters for qr_int4's fanout against the FlyDSL primitive.

WHY

Section 13 of op_tests/dump_data/docs/mi350p_peer_write_primitive_2026-09-07.md
ends on an unexplained factor of two. On one MI350P, in one session, at matched
publish cadence, block count, allocation, instruction and byte count:

    qr_int4 `nowait` (fanout + publish only)      13.6 GB/s
    FlyDSL peer-write primitive                   28-33 GB/s

Thirteen structural hypotheses were measured and each came back at 0-9%. ATT
says where the waves *wait* but cannot say what a stall was made of, and no
counter pass has ever been run on either kernel. This is that pass.

THE QUESTION IT ANSWERS

Both kernels issue the same instruction -- `global_store_dwordx4 ... nt`, 16 B
per lane, four lanes per 64 B sector. If the memory system coalesces each quad
into one 64 B EA write request, both should show

    payload bytes / TCC_EA0_WRREQ = 64

If qr_int4 instead shows ~32, its sectors are being split into 32 B
transactions: the same bytes at twice the request count, which at a fixed
request rate is exactly the observed 2x. That is a different failure from
"the same requests, served slower", and nothing measured so far can tell them
apart. `TCC_EA0_WRREQ_WRITE_IO_32B` separates the remote (PCIe) traffic from the
local self-write, and `LEVEL/WRREQ` gives the average cycles in flight per
request, so a latency explanation is visible too.

COUNTER RELIABILITY

Section 9 of the earlier report found `TCC_EA0_WRREQ_WRITE_IO_32B_sum` returning
different values for the same kernel and shape depending on which other counters
shared the pass -- once reading zero for a kernel that demonstrably wrote to
peers. So this does not trust a single pass: `TCC_EA0_WRREQ_sum` appears in
*every* group and the groups are cross-checked against each other. A spread
above a few percent on that anchor invalidates the rest of the run, and is
reported rather than averaged away.

WHAT IS NORMALISED AGAINST WHAT

The byte counts are not estimates. For qr_int4 they were derived from the ATT
trace's own hit counts (section 12.3) and agree exactly with the source:
147,456 B of fanout per workgroup, 224 workgroups. The primitive's is its
`--total`, all of it remote. Both land near 33 MB per dispatch, so the two are
comparable almost without normalisation.

USAGE

    python3 pmc_compare.py                 # both workloads, all groups
    python3 pmc_compare.py --only qr       # or: prim
    python3 pmc_compare.py --dry-run       # print the commands, run nothing

Takes ~10 min: counter collection serialises dispatches, and there are three
groups per workload.
"""

import argparse
import csv
import glob
import json
import os
import shlex
import subprocess
import sys
import time
from collections import defaultdict


def say(msg):
    """Progress line, flushed immediately so `tail -f` on the log is live.

    Counter collection serialises dispatches and a group takes minutes, so a
    buffered run looks indistinguishable from a hung one -- which is how the
    first attempt at this was misdiagnosed.
    """
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

REPO = "/home/vpietila/git/aiter"
SP = "/opt/venv/lib/python3.14/site-packages"
ROCM_CORE = f"{SP}/_rocm_sdk_core"

# `TCC_EA0_WRREQ_sum` is in every group on purpose: it is the cross-check that
# says whether the groups are measuring the same execution at all.
GROUPS = [
    # the coalescing question: how many requests, and how many were full 64 B
    ["TCC_EA0_WRREQ_sum", "TCC_EA0_WRREQ_64B_sum", "TCC_EA0_WRREQ_WRITE_IO_32B_sum"],
    # the latency question: cycles in flight per request, and credit exhaustion
    ["TCC_EA0_WRREQ_sum", "TCC_EA0_WRREQ_LEVEL_sum", "TCC_EA0_WRREQ_IO_CREDIT_STALL_sum"],
    # context for normalising the two above
    ["TCC_EA0_WRREQ_sum", "GRBM_GUI_ACTIVE", "GRBM_COUNT"],
    # machine-wide concurrency. Added after the 2026-09-07 MI350X/MI350P trace
    # comparison, which showed ATT cannot answer this question: normalised per
    # byte on MI350P, qr_int4 `nowait` stalls 3.11 cy/B against the primitive's
    # 5.31, and 0.52 against 1.63 on the stores alone. The kernel that is 2x
    # slower in wall clock has *fewer* stall cycles per byte on the traced CU.
    # A per-CU instrument cannot see that, so the limiter has to be in how much
    # of the machine is making progress at once. SQ_LEVEL_WAVES/SQ_WAVES is the
    # average resident wave count; SQ_BUSY_CU_CYCLES vs GRBM_GUI_ACTIVE is how
    # much of the GPU was occupied at all.
    ["SQ_WAVES", "SQ_LEVEL_WAVES", "SQ_BUSY_CU_CYCLES"],
]

# argv, not a shell string. `rocprofv3 --pmc -- bash -lc '<cmd>'` hangs on this
# stack: rocprofv3 intercepts the login shell's own children (run-parts,
# locale-check), logs those, and the real workload never starts. `--att` and
# `--kernel-trace` are fine through `bash -lc`; only counter collection is not.
# Environment goes through `env=` rather than an `env ...` prefix for the same
# reason -- no shell in the chain at all.
QR_ARGV = [
    "python3", f"{REPO}/op_tests/multigpu_tests/qr_ablation/run_ablation.py",
    "--inbox", "finegrained", "--ablate", "nowait", "--shapes", "4096",
    "--warmup", "2", "--iters", "3",
]
QR_ENV = {"HIP_VISIBLE_DEVICES": "0,1,2,3", "AITER_QRINT4_ABLATE": "nowait"}
PRIM_ARGV = [
    "python3", f"{REPO}/op_tests/flydsl_tests/test_flydsl_peer_write_bw.py",
    "--one", "--blocks", "224", "--chunk", "512", "--publish", "55296",
    "--warmup", "2", "--iters", "3",
]
# NOT "0,1,2,3". The single-writer primitive launches only on HIP device 0, and
# under that ordering device 0 is physical GPU 0 -- the agent whose TCC EA and
# GRBM counters section 14.2 found to under-report. Section 14.2 prescribes
# "1,2,3,0" for exactly this reason; landing it here makes the tool match.
#
# It does not change the answer: re-measured on 2026-09-08 under this ordering
# and filtered to Agent 3, `prim` reads 251 cy in flight and 0 credit stalls
# against the 242/0 section 14.3 recorded. Section 14.3's row is confirmed, not
# corrected -- GPU 1 reports the same thing GPU 0 did for this kernel.
PRIM_ENV = {"HIP_VISIBLE_DEVICES": "1,2,3,0"}


def _ranks_argv(writers):
    """The multi-rank primitive, section 14.4's control.

    `run_peer_write_ranks.py` drives the same kernel from `multiprocessing`
    with the spawn start method, which rocprofv3 follows -- unlike the `fork()`
    in `peer_write_bw --ipc`, which produced no counter rows at all.
    """
    return [
        "python3", f"{REPO}/op_tests/multigpu_tests/qr_ablation/run_peer_write_ranks.py",
        "--writers", str(writers), "--blocks", "224", "--chunk", "512",
        "--publish", "55296", "--warmup", "2", "--iters", "3",
    ]


# Physical GPU 1 (Agent 3) is the writer of interest in every prim* workload,
# because GPU 0's counters are broken (section 14.2). HIP device 0 under this
# ordering is physical GPU 1, so rank 0 always lands there.
RANKS_ENV = {"HIP_VISIBLE_DEVICES": "1,2,3,0"}

WORKLOADS = {
    # name: (kernel regex, launch, expected workgroups, fanout bytes per
    #        workgroup, remote fraction, measured wall clock us)
    "qr": {
        "regex": "^qr_int4_0$",
        "argv": QR_ARGV, "env": QR_ENV,
        # qr_int4 runs on four agents at once, one per rank. Without this filter
        # `per_dispatch` averages the valid agents together with GPU 0's broken
        # ones. Under "0,1,2,3", Agent 3 is physical GPU 1 -- the same physical
        # device every `prim*` workload is filtered to.
        "agent": "Agent 3",
        "wgs": 224,
        # From the ATT hit counts, section 12.3: 8 rows x 16 hits x 64 lanes x
        # 16 B + 4 rows x 8 hits x 32 lanes x 16 B. Confirmed against the source.
        "bytes_per_wg": 147456,
        # peer == rank is a local write; the other three go over PCIe.
        "remote_frac": 0.75,
        "wall_us": 1822.8,
        "label": "qr_int4 nowait, ST=8, finegrained",
    },
    "prim": {
        "regex": "^peer_write_0$",
        "argv": PRIM_ARGV, "env": PRIM_ENV,
        "agent": "Agent 3",
        "wgs": 224,
        "bytes_per_wg": (32 << 20) // 224,
        "remote_frac": 1.0,      # npeers=3, no self-write
        "wall_us": 1100.0,       # 32 MiB at the measured ~30.5 GB/s
        "label": "peer_write primitive, 224 blk, 55296 B publish",
    },
    # Section 14.4's control pair. Same kernel, same allocation, same geometry,
    # same process layout -- the writer count is the only difference, so any
    # move between these two rows is incast and nothing else.
    "prim1": {
        "regex": "^peer_write_0$",
        "argv": _ranks_argv(1), "env": RANKS_ENV,
        "agent": "Agent 3",
        "wgs": 224,
        "bytes_per_wg": (32 << 20) // 224,
        "remote_frac": 1.0,
        "wall_us": 1009.0,       # measured 2026-09-08: 33.26 GB/s
        "label": "multi-rank primitive, 1 writer -> 3 peers",
    },
    "prim4": {
        "regex": "^peer_write_0$",
        "argv": _ranks_argv(4), "env": RANKS_ENV,
        "agent": "Agent 3",
        "wgs": 224,
        "bytes_per_wg": (32 << 20) // 224,
        "remote_frac": 1.0,
        "wall_us": 1006.0,       # measured 2026-09-08: 33.34 GB/s mean
        "label": "multi-rank primitive, 4 writers, all-to-all incast",
    },
}


def rocprof_cmd(outdir, group, regex, argv):
    return [
        "rocprofv3", "--pmc", *group,
        "--output-format", "csv",
        "--output-directory", outdir,
        "--rocm-root", ROCM_CORE, "--sdk-soversion", "1",
        "--kernel-include-regex", regex,
        "--", *argv,
    ]


def read_counters(outdir, kernel, min_grid_wgs, agent=None):
    """{counter: [per-dispatch values]} for dispatches of the right geometry.

    Filtered on workgroup count because a run also launches `compile()`'s other
    super-tile engine, which writes a different number of bytes and would
    otherwise be averaged in.

    Filtered on *agent* because section 14.2 found GPU 0's TCC EA and GRBM
    counters under-report on this host, and because a multi-writer run has one
    writing process per GPU -- without this, `per_dispatch` would average a
    valid agent together with the broken one and with three other GPUs' traffic.
    `Agent_Id` is a string ("Agent 3") and indexes the *physical* device: HIP
    device 0 under `HIP_VISIBLE_DEVICES=1,2,3,0` is physical GPU 1 is Agent 3.
    """
    out = defaultdict(list)
    seen_grids = set()
    for path in glob.glob(os.path.join(outdir, "**", "*counter_collection.csv"),
                          recursive=True):
        with open(path) as fh:
            for row in csv.DictReader(fh):
                if row.get("Kernel_Name") != kernel:
                    continue
                if agent is not None and row.get("Agent_Id") != agent:
                    continue
                try:
                    wgs = int(row["Grid_Size"]) // int(row["Workgroup_Size"])
                except (KeyError, ValueError, ZeroDivisionError):
                    wgs = None
                if wgs is not None:
                    seen_grids.add(wgs)
                    if wgs != min_grid_wgs:
                        continue
                key = (row.get("Dispatch_Id"), path)
                out[row["Counter_Name"]].append((key, float(row["Counter_Value"])))
    return out, seen_grids


def per_dispatch(values):
    """Mean per dispatch, summing the per-agent rows that share a dispatch."""
    by_dispatch = defaultdict(float)
    for key, v in values:
        by_dispatch[key] += v
    return (sum(by_dispatch.values()) / len(by_dispatch)) if by_dispatch else 0.0


def run(name, spec, outroot, dry):
    say(f"=== {name}: {spec['label']} ===")
    results = {}
    anchors = []
    for i, group in enumerate(GROUPS):
        outdir = os.path.join(outroot, f"{name}_g{i}")
        cmd = rocprof_cmd(outdir, group, spec["regex"], spec["argv"])
        env = dict(os.environ, **spec["env"])
        if dry:
            print("  " + " ".join(shlex.quote(c) for c in cmd))
            continue
        os.makedirs(outdir, exist_ok=True)
        say(f"  group {i + 1}/{len(GROUPS)}: {' '.join(group)}")
        log = os.path.join(outdir, "rocprofv3.log")
        t0 = time.time()
        try:
            with open(log, "w") as fh:
                p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                   text=True, timeout=1800, env=env)
            rc = p.returncode
        except subprocess.TimeoutExpired:
            say(f"    TIMED OUT after 1800 s; log at {log}")
            continue
        dt = time.time() - t0
        if rc != 0:
            tail = ""
            try:
                tail = open(log).read()[-300:].strip().replace("\n", " | ")
            except OSError:
                pass
            say(f"    FAILED rc={rc} after {dt:.0f} s: {tail}")
            continue
        got, grids = read_counters(outdir, spec["regex"].strip("^$"), spec["wgs"],
                                   spec.get("agent"))
        if not got:
            say(f"    {dt:.0f} s, but no rows at {spec['wgs']} wg; "
                f"geometries seen: {sorted(grids)}")
            continue
        for counter, vals in got.items():
            v = per_dispatch(vals)
            results[counter] = v
            if counter == "TCC_EA0_WRREQ_sum":
                anchors.append(v)
        n = len({k for k, _ in next(iter(got.values()))})
        say(f"    {dt:.0f} s, {len(got)} counter(s) over {n} dispatch(es): "
            + ", ".join(f"{c.replace('TCC_EA0_', '')}={results[c]:,.0f}"
                        for c in group if c in results))
        # Partial results after every group, so an interrupted run is not lost
        # and progress is inspectable from outside the process.
        with open(os.path.join(outroot, f"{name}.partial.json"), "w") as fh:
            json.dump(results, fh, indent=2)
    if anchors:
        lo, hi = min(anchors), max(anchors)
        spread = (hi - lo) / hi if hi else 0
        flag = "ok" if spread < 0.05 else "SUSPECT - groups disagree"
        say(f"  cross-check TCC_EA0_WRREQ_sum across groups: "
            f"{lo:,.0f}..{hi:,.0f}  ({spread * 100:.1f}% spread, {flag})")
        results["_anchor_spread"] = spread
    return results


def report(all_results):
    print("\n" + "=" * 78)
    print("EA write requests per dispatch, and bytes per request")
    print("=" * 78)
    hdr = f"{'metric':<38}" + "".join(f"{n:>19}" for n in all_results)
    print(hdr)

    def line(label, fn, fmt="{:>19,.0f}"):
        row = f"{label:<38}"
        for name in all_results:
            r = all_results[name]
            spec = WORKLOADS[name]
            try:
                v = fn(r, spec)
                row += fmt.format(v) if v is not None else f"{'-':>19}"
            except (KeyError, ZeroDivisionError, TypeError):
                row += f"{'-':>19}"
        print(row)

    fanout = lambda s: s["wgs"] * s["bytes_per_wg"]          # noqa: E731
    remote = lambda s: fanout(s) * s["remote_frac"]           # noqa: E731

    line("fanout bytes / dispatch", lambda r, s: fanout(s))
    line("  of which remote (PCIe)", lambda r, s: remote(s))
    line("wall clock us / dispatch", lambda r, s: s["wall_us"], "{:>19,.1f}")
    line("remote GB/s (wall clock)",
         lambda r, s: remote(s) / (s["wall_us"] * 1e3), "{:>19,.2f}")
    print("-" * 78)
    line("TCC_EA0_WRREQ", lambda r, s: r["TCC_EA0_WRREQ_sum"])
    line("TCC_EA0_WRREQ_64B", lambda r, s: r["TCC_EA0_WRREQ_64B_sum"])
    line("  64B share of requests",
         lambda r, s: 100 * r["TCC_EA0_WRREQ_64B_sum"] / r["TCC_EA0_WRREQ_sum"],
         "{:>18,.1f}%")
    line("TCC_EA0_WRREQ_WRITE_IO_32B",
         lambda r, s: r["TCC_EA0_WRREQ_WRITE_IO_32B_sum"])
    print("-" * 78)
    print("  the question this run exists to answer:")
    line("BYTES PER EA WRITE REQUEST",
         lambda r, s: fanout(s) / r["TCC_EA0_WRREQ_sum"], "{:>19,.1f}")
    print("     64 = each 64 B sector became one request (what both should show)")
    print("     32 = sectors are being split, so twice the requests per byte")
    print("-" * 78)
    line("avg cycles in flight per req",
         lambda r, s: r["TCC_EA0_WRREQ_LEVEL_sum"] / r["TCC_EA0_WRREQ_sum"],
         "{:>19,.0f}")
    line("IO credit stall cycles", lambda r, s: r["TCC_EA0_WRREQ_IO_CREDIT_STALL_sum"])
    line("  per GPU active cycle",
         lambda r, s: r["TCC_EA0_WRREQ_IO_CREDIT_STALL_sum"] / r["GRBM_GUI_ACTIVE"],
         "{:>19,.2f}")
    print("-" * 78)
    line("SQ_WAVES (waves launched)", lambda r, s: r["SQ_WAVES"])
    line("avg resident waves",
         lambda r, s: r["SQ_LEVEL_WAVES"] / r["SQ_WAVES"], "{:>19,.1f}")
    line("SQ busy CU cycles", lambda r, s: r["SQ_BUSY_CU_CYCLES"])
    line("  vs GRBM_GUI_ACTIVE",
         lambda r, s: r["SQ_BUSY_CU_CYCLES"] / r["GRBM_GUI_ACTIVE"], "{:>19,.2f}")
    print("-" * 78)
    line("counter cross-check spread",
         lambda r, s: 100 * r["_anchor_spread"], "{:>18,.1f}%")
    print()
    print("A bytes-per-request near 64 on both means the same requests are being")
    print("served at different rates -- look at cycles in flight and credit stalls.")
    print("A 2x split between them means the same bytes are being issued as twice")
    print("the requests, and that alone accounts for the factor of two.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=sorted(WORKLOADS), action="append", default=[])
    ap.add_argument("--out", default="/tmp/pmc_compare")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    names = args.only or list(WORKLOADS)
    all_results = {}
    for name in names:
        r = run(name, WORKLOADS[name], args.out, args.dry_run)
        if r:
            all_results[name] = r
    if all_results and not args.dry_run:
        report(all_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
