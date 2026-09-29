#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regenerate hsa/gfx950/mla_v4/mla_a8w8_qh64_qseqlen1_gqaratio64_nm_ps.co from sources (python3 stdlib only).

  python3 gen.py --out <path.co> [--check <expected.co>] [--keep <dir>] [--verify-rt <nm.co>]

Steps: asm/rt.s --(asm/patch.py: persistent walk + splice/pre.hip planner + splice/fin.hip fused combine, both
compiled with hipcc -S)--> patched .s --(kernel symbol renamed to the _ps name)--> clang -x assembler (gfx950)
--> ld.lld -shared --> .co. Tools come from $ROCM_PATH (default /opt/rocm)."""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "asm"))
import patch as PATCH  # noqa: E402

RT_SHA256 = "7fc7516ef916ba83770744ed5d55faafb57684908fd31b76727adbc6ae9bd20b"
OLD_NAME = "mla_a8w8_qh64_qseqlen1_gqaratio64_nm"
NEW_NAME = "mla_a8w8_qh64_qseqlen1_gqaratio64_nm_ps"


def mangle(name):
    """aiter::<name> (Itanium mangling of a namespace-scope name, as used by the AITER ASM kernels)."""
    return f"_ZN5aiter{len(name)}{name}E"


def tools():
    rocm = os.environ.get("ROCM_PATH") or "/opt/rocm"
    t = dict(
        clang=f"{rocm}/llvm/bin/clang",
        lld=f"{rocm}/llvm/bin/ld.lld",
        readelf=f"{rocm}/llvm/bin/llvm-readelf",
        objcopy=f"{rocm}/llvm/bin/llvm-objcopy",
        hipcc=f"{rocm}/bin/hipcc",
    )
    for k, p in t.items():
        if not os.access(p, os.X_OK):
            sys.exit(f"gen.py: {k} not found at {p} (set ROCM_PATH)")
    return t


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def rename(text, old, new):
    """Rename the kernel symbol everywhere (symbol, .kd, metadata .name / .symbol); returns (text, occurrences)."""
    mo, mn = mangle(old), mangle(new)
    assert mn not in text, mn
    pat = re.compile(
        re.escape(mo) + r"(?![A-Za-z0-9_])"
    )  # also matches the "<sym>.kd" occurrences
    text, n = pat.subn(mn, text)
    assert n > 0 and mo not in text, (mo, n)
    return text, n


def assemble(t, s_path, co_path):
    o_path = os.path.splitext(co_path)[0] + ".o"
    subprocess.run(
        [
            t["clang"],
            "-x",
            "assembler",
            "-target",
            "amdgcn-amd-amdhsa",
            "-mcpu=gfx950",
            "-c",
            s_path,
            "-o",
            o_path,
        ],
        check=True,
    )
    subprocess.run([t["lld"], "-shared", o_path, "-o", co_path], check=True)


def section(t, co, name, d):
    out = os.path.join(d, os.path.basename(co) + name)
    subprocess.run(
        [t["objcopy"], "-O", "binary", f"--only-section={name}", co, out], check=True
    )
    return open(out, "rb").read()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True, help="output code object (.co)")
    ap.add_argument(
        "--check",
        help="expected .co: fail unless the output is byte-identical (sha256)",
    )
    ap.add_argument(
        "--keep",
        help="directory to keep the intermediates in (hipcc .s, patched .s, .o)",
    )
    ap.add_argument(
        "--verify-rt",
        help="the #5195 .co (mla_a8w8_qh64_qseqlen1_gqaratio64_nm.co): check that asm/rt.s "
        "reassembles to its .text and kernel descriptor",
    )
    a = ap.parse_args()
    t = tools()
    rt = os.path.join(HERE, "asm", "rt.s")
    got = sha256(rt)
    if got != RT_SHA256:
        sys.exit(f"gen.py: asm/rt.s sha256 {got} != {RT_SHA256}")
    tmp = tempfile.TemporaryDirectory()
    work = a.keep or tmp.name
    os.makedirs(work, exist_ok=True)
    if a.verify_rt:
        rco = os.path.join(work, "rt.co")
        assemble(t, rt, rco)
        for sec in (".text", ".rodata"):
            same = section(t, rco, sec, work) == section(t, a.verify_rt, sec, work)
            print(f"verify-rt {sec}: {'identical' if same else 'DIFFERENT'}")
            if not same:
                sys.exit(1)
    text, marks, pads = PATCH.patch(
        open(rt).read(), t["clang"], t["readelf"], t["hipcc"], work
    )
    print("patched", marks, "pads", pads)
    text, n = rename(text, OLD_NAME, NEW_NAME)
    print(f"renamed {mangle(OLD_NAME)} -> {mangle(NEW_NAME)} ({n} occurrences)")
    s_path = os.path.join(work, NEW_NAME + ".s")
    open(s_path, "w").write(text)
    co_tmp = os.path.join(work, NEW_NAME + ".co")
    assemble(t, s_path, co_tmp)
    out = os.path.abspath(a.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copyfile(co_tmp, out)
    print(f"wrote {out}  sha256 {sha256(out)}")
    if a.check:
        exp = sha256(a.check)
        ok = exp == sha256(out)
        print(f"check {a.check}  sha256 {exp}: {'IDENTICAL' if ok else 'MISMATCH'}")
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
