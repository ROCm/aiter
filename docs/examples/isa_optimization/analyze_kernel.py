"""Analyze AITER kernel ISA and rocprofv3 profiling results.

Two modes:
  1. ISA analysis - instruction mix and resource usage of a .co file
  2. Profile analysis - per-kernel timing from a rocprofv3 --kernel-trace database

Usage:
  # Analyze a .co file (architecture is read from the ELF header)
  python3 analyze_kernel.py isa kernel.co

  # Analyze rocprofv3 results (default output format is a rocpd SQLite .db)
  rocprofv3 --kernel-trace -d ./profile_out -- python3 bench.py
  python3 analyze_kernel.py profile ./profile_out

  # Filter profile results by kernel name substring
  python3 analyze_kernel.py profile ./profile_out --filter pa_
"""

import argparse
import glob
import os
import re
import sqlite3
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_asm import ADDR_COMMENT, Tools, default_llvm_bin, detect_mcpu

# Ordered (first match wins), keyed on the instruction mnemonic.
CATEGORIES = [
    ("MFMA / matrix", r"v_(mfma|smfma|wmma)"),
    ("VALU transcendental", r"v_(rcp|rsq|sqrt|log|exp|sin|cos)_"),
    ("VALU convert / pack", r"v_(cvt|pk|pack|perm|alignb|bfe|bfi|lshl_or|and_or|or3)"),
    ("VALU other", r"v_"),
    ("VMEM buffer load", r"buffer_load"),
    ("VMEM buffer store/atomic", r"buffer_"),
    ("VMEM global/flat/scratch", r"(global|flat|scratch)_"),
    ("LDS", r"ds_"),
    ("SMEM (s_load / s_buffer_load)", r"s_(buffer_)?(load|store)"),
    ("s_waitcnt", r"s_waitcnt"),
    ("s_nop", r"s_nop"),
    ("s_barrier / s_sleep / s_setprio", r"s_(barrier|sleep|setprio|sethalt)"),
    ("branch", r"s_(cbranch|branch|setpc|swappc|getpc|call)"),
    ("s_endpgm", r"s_endpgm"),
    ("SALU other", r"s_"),
]


def analyze_isa(co_path: str, mcpu: str | None, llvm_bin: str):
    """Disassemble a .co and print instruction mix and resource usage."""
    tools = Tools(llvm_bin)
    mcpu = mcpu or detect_mcpu(tools, co_path)
    dis = tools.run("llvm-objdump", "-d", f"--mcpu={mcpu}", co_path)

    # AMDGPU llvm-objdump prints "<TAB>mnemonic operands ... // ADDR: ENCODING";
    # symbol headers are "ADDR <name>:".
    kernels, instrs = [], []
    for line in dis.splitlines():
        m = re.match(r"^[0-9a-fA-F]+ <(.+)>:$", line)
        if m and not m.group(1).startswith("label_"):
            kernels.append(m.group(1))
        elif line.startswith("\t") and "//" in line:
            text = ADDR_COMMENT.sub("", line).strip()
            if text and not text.startswith("."):
                instrs.append(text)

    mix = Counter()
    modifiers = Counter()
    for text in instrs:
        mnemonic = text.split()[0]
        for name, pattern in CATEGORIES:
            if re.match(pattern, mnemonic):
                mix[name] += 1
                break
        else:
            mix["other"] += 1
        if "_dpp" in mnemonic or " row_" in text or " quad_perm" in text:
            modifiers["DPP (cross-lane)"] += 1
        if "_sdwa" in mnemonic:
            modifiers["SDWA"] += 1
        if re.search(r"\ba\[?\d", text):
            modifiers["uses AGPRs"] += 1

    print(f"File:         {co_path}")
    print(f"Architecture: {mcpu}")
    print(f"Kernels:      {', '.join(kernels) or '(none)'}")
    print(f"Instructions: {len(instrs)}")
    print()
    print(f"  {'Category':<32s} {'Count':>7s} {'Pct':>7s}")
    print("  " + "-" * 48)
    for name, _ in CATEGORIES + [("other", None)]:
        n = mix.get(name, 0)
        if n:
            print(f"  {name:<32s} {n:>7d} {100.0 * n / len(instrs):>6.1f}%")
    if modifiers:
        print()
        for name, n in modifiers.items():
            print(f"  {name:<32s} {n:>7d}")

    # Resource usage from the kernel descriptor and the metadata note
    kd = tools.run("llvm-objdump", "-D", "-j", ".rodata", f"--mcpu={mcpu}", co_path)
    notes = tools.run("llvm-readelf", "--notes", co_path)
    want_kd = (
        "group_segment_fixed_size",
        "private_segment_fixed_size",
        "next_free_vgpr",
        "next_free_sgpr",
        "accum_offset",
        "ieee_mode",
        "dx10_clamp",
        "float_denorm_mode_32",
        "float_denorm_mode_16_64",
    )
    want_md = (
        "vgpr_count",
        "agpr_count",
        "sgpr_count",
        "kernarg_segment_size",
        "wavefront_size",
        "max_flat_workgroup_size",
    )
    print()
    print("  Kernel descriptor (.rodata):")
    for line in kd.splitlines():
        d = line.split()
        if d and d[0].startswith(".amdhsa_") and d[0][len(".amdhsa_") :] in want_kd:
            print(f"    {d[0]:<40s} {d[1]}")
    print("  Metadata (.note):")
    for line in notes.splitlines():
        m = re.match(r"^\s+\.(\w+):\s+(\S+)$", line)
        if m and m.group(1) in want_md:
            print(f"    .{m.group(1):<39s} {m.group(2)}")
    print()


def analyze_profile(profile_dir: str, name_filter: str | None = None):
    """Parse rocprofv3 --kernel-trace SQLite (rocpd) output."""
    db_files = glob.glob(os.path.join(profile_dir, "**/*.db"), recursive=True)
    if not db_files:
        sys.exit(
            f"Error: no .db file found under {profile_dir} "
            "(was rocprofv3 run with the default rocpd output format?)"
        )
    db_path = min(db_files)
    print(f"Database: {db_path}\n")

    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # rocpd tables carry a UUID suffix (rocpd_kernel_dispatch<UUID>); recent
    # rocprofiler-sdk versions also create suffix-free views over them.
    c.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
    names = [row[0] for row in c.fetchall()]

    def pick(view, needle):
        if view in names:
            return view
        return next((t for t in names if needle in t.lower()), None)

    dispatch_table = pick("rocpd_kernel_dispatch", "kernel_dispatch")
    symbol_table = pick("rocpd_info_kernel_symbol", "kernel_symbol")
    if not dispatch_table or not symbol_table:
        sys.exit(
            "Error: expected kernel_dispatch and kernel_symbol tables; "
            f"available: {names}"
        )

    where, params = "", ()
    if name_filter:
        where, params = "WHERE ks.kernel_name LIKE ?", (f"%{name_filter}%",)

    c.execute(
        f"""
        SELECT ks.kernel_name, COUNT(*) AS cnt,
               AVG(d."end" - d.start) AS avg_ns,
               MIN(d."end" - d.start) AS min_ns,
               MAX(d."end" - d.start) AS max_ns
        FROM "{dispatch_table}" d
        JOIN "{symbol_table}" ks ON d.kernel_id = ks.id
        {where}
        GROUP BY ks.kernel_name
        ORDER BY avg_ns DESC
        """,
        params,
    )
    rows = c.fetchall()
    if not rows:
        print("No kernel dispatches found.")
        conn.close()
        return

    print(
        f"{'Kernel':<70s} {'Count':>6s} {'Avg(us)':>10s} {'Min(us)':>10s} {'Max(us)':>10s}"
    )
    print("-" * 110)
    total_ns = 0
    total_dispatches = 0
    for name, cnt, avg, mn, mx in rows:
        print(
            f"  {name[:68]:<70s} {cnt:>6d} {avg / 1000:>10.1f} "
            f"{mn / 1000:>10.1f} {mx / 1000:>10.1f}"
        )
        total_ns += avg * cnt
        total_dispatches += cnt
    print(f"\nTotal dispatches: {total_dispatches}")
    print(f"Total GPU time: {total_ns / 1e6:.2f} ms")

    # Register / LDS usage as recorded by the runtime (if the columns exist)
    c.execute(f'PRAGMA table_info("{symbol_table}")')
    columns = {row[1] for row in c.fetchall()}
    wanted = ["arch_vgpr_count", "accum_vgpr_count", "sgpr_count", "group_segment_size"]
    if all(col in columns for col in wanted):
        print(f"\n{'Kernel':<50s} {'VGPR':>6s} {'AGPR':>6s} {'SGPR':>6s} {'LDS':>8s}")
        print("-" * 80)
        c.execute(
            f"""
            SELECT DISTINCT kernel_name, {", ".join(wanted)}
            FROM "{symbol_table}" ks
            {where}
            ORDER BY kernel_name
            """,
            params,
        )
        for name, vgpr, agpr, sgpr, lds in c.fetchall():
            print(f"  {name[:48]:<50s} {vgpr:>6d} {agpr:>6d} {sgpr:>6d} {lds:>8d}")
    conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze AITER kernel ISA or rocprofv3 profile results"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_isa = sub.add_parser("isa", help="instruction mix / resource usage of a .co file")
    p_isa.add_argument("co_file", help="path to .co kernel object")
    p_isa.add_argument("--mcpu", help="GPU architecture (default: read from the ELF)")
    p_isa.add_argument(
        "--llvm-bin",
        default=default_llvm_bin(),
        help=f"directory with llvm-objdump etc. (default: {default_llvm_bin()})",
    )

    p_prof = sub.add_parser("profile", help="analyze rocprofv3 --kernel-trace output")
    p_prof.add_argument("profile_dir", help="rocprofv3 output directory")
    p_prof.add_argument("--filter", help="filter kernels by name substring")

    args = parser.parse_args()
    if args.mode == "isa":
        analyze_isa(args.co_file, args.mcpu, args.llvm_bin)
    else:
        analyze_profile(args.profile_dir, args.filter)


if __name__ == "__main__":
    main()
