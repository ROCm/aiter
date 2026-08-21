# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Read back REAL kernel timing from a rocprofv3 --att output directory and derive
# the average GPU (gfx) clock of the ATT-traced dispatch.
#
# Why this exists: the UT's own summary table cannot report real time when run
# under rocprofv3 (its internal torch.profiler collides with rocprof's
# roctracer/rocprofiler layer -> garbage us). Instead we run the UT with
# --under-rocprof (bare, no inner profiler) and read the truth from rocprof:
#
#   time   : dispatch wall time (end - start), from the KERNEL_DISPATCH domain in
#            att_results.db (rocpd). REQUIRES the rocprofv3 command to include
#            --kernel-trace; with --att alone that table is empty.
#   cycles : span of the gfx_clock over the trace, from ui_output/realtime.json
#            (metadata.descriptor == '[gfx_clock, realtime_clock]', so column 0 is
#            the shader clock in cycles). This is the ATT trace ("ttrace").
#   freq   : cycles / time  (cycles per ns == GHz).
#
# time and cycles are read for the SAME dispatch (the one ATT traced, e.g.
# --kernel-iteration-range [50]), keyed by the dispatch id encoded in the
# ui_output_*_dispatch_<ID> directory name, so freq = cycles/time is
# self-consistent even though ATT slightly perturbs that dispatch.

import argparse
import glob
import json
import os
import re
import sqlite3


def _find_table(con, prefix):
    """rocpd table names carry a per-run UUID suffix; resolve by prefix."""
    row = con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE ?",
        (prefix + "%",),
    ).fetchone()
    return row[0] if row else None


def _gfx_cycle_span(realtime_path):
    """Span (max-min) of the gfx_clock column across all shader-engine series."""
    rt = json.load(open(realtime_path))
    meta = rt.get("metadata", {})
    desc = meta.get("descriptor", "")
    if "gfx_clock" not in desc:
        # column 0 is expected to be gfx_clock; warn but proceed on that assumption
        print(f"[att_freq] WARN {realtime_path}: unexpected descriptor {desc!r}")
    lo = hi = None
    for key, series in rt.items():
        if not isinstance(series, list):
            continue  # skip 'metadata'
        for row in series:
            if not (isinstance(row, list) and len(row) >= 1):
                continue
            c = row[0]
            lo = c if lo is None else min(lo, c)
            hi = c if hi is None else max(hi, c)
    if lo is None:
        return None
    return hi - lo


def _dispatch_time_ns(con, kd_tab, ks_tab, dispatch_id, kernel_substr):
    """(end-start) ns for the traced dispatch. Prefer an exact dispatch_id match;
    else fall back to the single kernel-name match. Returns (ns, id, name)."""
    rows = con.execute(
        f"""SELECT d.dispatch_id, d.start, d.end, s.kernel_name
              FROM {kd_tab} d JOIN {ks_tab} s ON d.kernel_id = s.id
             WHERE s.kernel_name LIKE ?""",
        ("%" + kernel_substr + "%",),
    ).fetchall()
    if not rows:
        return None
    chosen = None
    if dispatch_id is not None:
        chosen = next((r for r in rows if r[0] == dispatch_id), None)
    if chosen is None:
        # no id match: unambiguous only when a single dispatch was captured
        chosen = rows[0] if len(rows) == 1 else rows[-1]
    did, start, end, name = chosen
    return (end - start, did, name)


def report(prof_dir, kernel_substr="f4gemm", mnk=(16384, 16384, 16384)):
    dbs = glob.glob(os.path.join(prof_dir, "*.db"))
    if not dbs:
        print(f"[att_freq] no *.db in {prof_dir}; nothing to read")
        return
    con = sqlite3.connect(dbs[0])
    kd_tab = _find_table(con, "rocpd_kernel_dispatch")
    ks_tab = _find_table(con, "rocpd_info_kernel_symbol")
    if not kd_tab or not ks_tab:
        print(f"[att_freq] {dbs[0]}: kernel dispatch/symbol tables missing")
        return

    realtimes = sorted(glob.glob(os.path.join(prof_dir, "**", "realtime.json"), recursive=True))
    if not realtimes:
        print(f"[att_freq] no realtime.json under {prof_dir} (was --att enabled?)")
        return

    for rt_path in realtimes:
        parent = os.path.basename(os.path.dirname(rt_path))
        m = re.search(r"dispatch_(\d+)", parent)
        disp_id = int(m.group(1)) if m else None
        cycles = _gfx_cycle_span(rt_path)
        t = _dispatch_time_ns(con, kd_tab, ks_tab, disp_id, kernel_substr)

        print(f"[att_freq] {prof_dir}  (dispatch {disp_id})")
        # Print each quantity that is available; freq needs both.
        if t is not None:
            time_ns, did, name = t
            us = time_ns / 1e3
            # Same formula as the UT: flops / us / 1e6, flops = 2*M*N*K.
            flops = 2 * mnk[0] * mnk[1] * mnk[2]
            tflops = round(flops / us / 1e6, 1)
            print(f"  kernel        : {name}")
            print(f"  dispatch time : {us:.3f} us      (rocpd end-start, dispatch_id={did})")
            print(f"  TFLOPS        : {tflops}      (2*M*N*K/time, M,N,K={mnk[0]},{mnk[1]},{mnk[2]})")
        else:
            print(
                "  dispatch time : N/A  -- rocpd_kernel_dispatch has no matching "
                "row. Add --kernel-trace to the rocprofv3 command."
            )
        if cycles is not None:
            print(f"  gfx cycles    : {cycles}      (ATT gfx_clock span)")
        else:
            print("  gfx cycles    : N/A  -- could not read gfx_clock from realtime.json")
        if t is not None and cycles is not None:
            freq_ghz = cycles / t[0]  # cycles / ns == GHz
            print(f"  avg gfx clock : {freq_ghz:.4f} GHz    (cycles / time)")


def main():
    ap = argparse.ArgumentParser(
        description="Derive avg gfx clock from a rocprofv3 --att output dir "
        "(real dispatch time from --kernel-trace / gfx_clock span from ATT)."
    )
    ap.add_argument("prof_dir", help="rocprofv3 -d output directory")
    ap.add_argument(
        "--kernel-substr",
        default="f4gemm",
        help="substring to match the traced kernel name (default: f4gemm)",
    )
    ap.add_argument(
        "--mnk",
        default="16384,16384,16384",
        help="problem size M,N,K for the TFLOPS calc (default: 16384,16384,16384)",
    )
    args = ap.parse_args()
    mnk = tuple(int(x) for x in args.mnk.split(","))
    if len(mnk) != 3:
        ap.error("--mnk must be three comma-separated ints, e.g. 16384,16384,16384")
    report(args.prof_dir, args.kernel_substr, mnk)


if __name__ == "__main__":
    main()
