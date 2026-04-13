#!/usr/bin/env python3
"""Analyze AITER kernel ISA and rocprofv3 profiling results.

Two modes:
  1. ISA analysis — instruction mix breakdown from a .co file
  2. Profile analysis — parse rocprofv3 --kernel-trace SQLite output

Usage:
  # Analyze a .co file directly
  python3 analyze_kernel.py isa kernel.co --mcpu gfx942

  # Analyze rocprofv3 results
  rocprofv3 --kernel-trace -d ./profile_out -- python bench.py
  python3 analyze_kernel.py profile ./profile_out

  # Filter profile results by kernel name pattern
  python3 analyze_kernel.py profile ./profile_out --filter "pa_"
"""
import argparse
import glob
import os
import re
import sqlite3
import subprocess
import sys


def analyze_isa(co_path: str, mcpu: str):
    """Disassemble a .co and print instruction mix analysis."""
    objdump = "/opt/rocm/llvm/bin/llvm-objdump"
    if not os.path.exists(objdump):
        print(f"Error: {objdump} not found. Is ROCm installed?", file=sys.stderr)
        sys.exit(1)

    result = subprocess.run(
        [objdump, "-d", f"--mcpu={mcpu}", co_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Error: llvm-objdump failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    lines = result.stdout.splitlines()

    # Find kernel symbols
    symbols = []
    for line in lines:
        m = re.match(r"[0-9a-fA-F]+ <(.+)>:", line)
        if m:
            symbols.append(m.group(1))

    # Count instructions by category
    instructions = [l for l in lines if re.match(r"\s+[0-9a-f]+:", l)]
    categories = {
        "MFMA (matrix)":   [l for l in instructions if "v_mfma_" in l],
        "Buffer load":     [l for l in instructions if "buffer_load" in l],
        "Buffer store":    [l for l in instructions if "buffer_store" in l],
        "Global load":     [l for l in instructions if "global_load" in l],
        "Global store":    [l for l in instructions if "global_store" in l],
        "LDS (ds_*)":      [l for l in instructions if re.search(r"\bds_", l)],
        "DPP (*_dpp)":     [l for l in instructions if "_dpp" in l],
        "Scalar (s_*)":    [l for l in instructions if re.search(r"\bs_\w+", l)],
        "Vector ALU":      [l for l in instructions if re.search(r"\bv_(?!mfma_)", l)],
        "Wait states":     [l for l in instructions if "s_waitcnt" in l or "s_nop" in l],
    }

    print(f"File: {co_path}")
    print(f"Architecture: {mcpu}")
    print(f"Kernel symbols: {len(symbols)}")
    for s in symbols:
        print(f"  {s}")
    print(f"\nTotal instructions: {len(instructions)}")
    print(f"\n{'Category':<25s} {'Count':>8s} {'Pct':>8s}")
    print("-" * 43)
    for cat, matches in categories.items():
        pct = len(matches) / len(instructions) * 100 if instructions else 0
        print(f"  {cat:<23s} {len(matches):>8d} {pct:>7.1f}%")

    # Compute-to-memory ratio
    n_compute = len(categories["MFMA (matrix)"]) + len(categories["Vector ALU"])
    n_memory = (len(categories["Buffer load"]) + len(categories["Buffer store"]) +
                len(categories["Global load"]) + len(categories["Global store"]))
    if n_memory > 0:
        print(f"\nCompute/Memory ratio: {n_compute / n_memory:.2f}")
    print()


def analyze_profile(profile_dir: str, name_filter: str | None = None):
    """Parse rocprofv3 --kernel-trace SQLite output."""
    db_files = glob.glob(os.path.join(profile_dir, "**/*results.db"), recursive=True)
    if not db_files:
        # Try flat directory
        db_files = glob.glob(os.path.join(profile_dir, "*.db"))
    if not db_files:
        print(f"Error: no .db file found in {profile_dir}", file=sys.stderr)
        sys.exit(1)

    db_path = db_files[0]
    print(f"Database: {db_path}\n")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Find table names (they have UUID suffixes)
    c.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in c.fetchall()]

    dispatch_table = next((t for t in tables if "kernel_dispatch" in t.lower()), None)
    symbol_table = next((t for t in tables if "kernel_symbol" in t.lower()), None)

    if not dispatch_table or not symbol_table:
        print(f"Error: expected kernel_dispatch and kernel_symbol tables", file=sys.stderr)
        print(f"Available tables: {tables}", file=sys.stderr)
        sys.exit(1)

    # Build query
    where_clause = ""
    if name_filter:
        where_clause = f"WHERE ks.kernel_name LIKE '%{name_filter}%'"

    c.execute(f"""
        SELECT ks.kernel_name, COUNT(*) as cnt,
               AVG(d.end - d.start) as avg_ns,
               MIN(d.end - d.start) as min_ns,
               MAX(d.end - d.start) as max_ns
        FROM {dispatch_table} d
        JOIN {symbol_table} ks ON d.kernel_id = ks.id
        {where_clause}
        GROUP BY ks.kernel_name
        ORDER BY avg_ns DESC
    """)
    rows = c.fetchall()

    if not rows:
        print("No kernel dispatches found.")
        conn.close()
        return

    print(f"{'Kernel':<70s} {'Count':>6s} {'Avg(us)':>10s} {'Min(us)':>10s} {'Max(us)':>10s}")
    print("-" * 100)
    total_ns = 0
    total_dispatches = 0
    for name, cnt, avg, mn, mx in rows:
        display = name[:68]
        print(f"  {display:<70s} {cnt:>6d} {avg/1000:>10.1f} {mn/1000:>10.1f} {mx/1000:>10.1f}")
        total_ns += avg * cnt
        total_dispatches += cnt

    print(f"\nTotal dispatches: {total_dispatches}")
    print(f"Total GPU time: {total_ns/1e6:.2f} ms")

    # Register usage (if available in symbol table)
    try:
        c.execute(f"PRAGMA table_info({symbol_table})")
        columns = [row[1] for row in c.fetchall()]
        if "arch_vgpr_count" in columns:
            print(f"\n{'Kernel':<50s} {'VGPR':>6s} {'AGPR':>6s} {'SGPR':>6s} {'LDS':>8s}")
            print("-" * 78)
            c.execute(f"""
                SELECT DISTINCT kernel_name, arch_vgpr_count, accum_vgpr_count,
                       sgpr_count, group_segment_size
                FROM {symbol_table}
                {where_clause.replace('ks.', '')}
                ORDER BY kernel_name
            """)
            for name, vgpr, agpr, sgpr, lds in c.fetchall():
                display = name[:48]
                print(f"  {display:<50s} {vgpr:>6d} {agpr:>6d} {sgpr:>6d} {lds:>8d}")
    except Exception:
        pass

    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze AITER kernel ISA or rocprofv3 profile results")
    sub = parser.add_subparsers(dest="mode", required=True)

    # ISA mode
    p_isa = sub.add_parser("isa", help="Analyze .co file instruction mix")
    p_isa.add_argument("co_file", help="Path to .co kernel object")
    p_isa.add_argument("--mcpu", default="gfx942", help="GPU architecture (default: gfx942)")

    # Profile mode
    p_prof = sub.add_parser("profile", help="Analyze rocprofv3 --kernel-trace output")
    p_prof.add_argument("profile_dir", help="rocprofv3 output directory")
    p_prof.add_argument("--filter", help="Filter kernels by name substring")

    args = parser.parse_args()

    if args.mode == "isa":
        analyze_isa(args.co_file, args.mcpu)
    elif args.mode == "profile":
        analyze_profile(args.profile_dir, args.filter)


if __name__ == "__main__":
    main()
