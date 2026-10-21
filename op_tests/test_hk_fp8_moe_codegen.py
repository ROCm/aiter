# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Audit FP8 MoE assembly using the compile command from an existing AITER JIT build.

Compiler instructions must avoid v128-v255 and all AGPRs: these registers hold
HipKittens ART tiles. Inline assembly implementing those tiles is exempt. This
check supplements runtime correctness tests; zero spills alone do not prove that
the compiler avoids pinned registers. No PyTorch import or GPU is required.
"""

import argparse
import json
import re
import shlex
import subprocess
import tempfile
from pathlib import Path


def emit_assembly(build_dir: Path, output: Path) -> None:
    database = json.loads(
        subprocess.check_output(["ninja", "-t", "compdb"], cwd=build_dir, text=True)
    )
    entries = [e for e in database if Path(e["file"]).name == "hk_moe.cu"]
    if len(entries) != 1:
        raise ValueError(
            f"expected one hk_moe.cu compile command, found {len(entries)}"
        )
    entry = entries[0]
    command = shlex.split(entry["command"])
    if command.count("-c") != 1 or command.count("-o") != 1:
        raise ValueError("expected a compile command with one -c and one -o")
    command.remove("-c")
    command[command.index("-o") + 1] = str(output.resolve())
    command += ["--cuda-device-only", "-S"]
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=entry["directory"], check=True)


def check_assembly(path: Path) -> None:
    source = path.read_text()
    kernels = list(re.finditer(r"^(_Z\w*moe_gemm_kernel\w*):", source, re.MULTILINE))
    # Gate/up and down GEMMs, each with and without the small-batch specialization.
    if len(kernels) != 4:
        raise ValueError(f"expected four GEMM variants, found {len(kernels)}")
    bad_total = 0
    for match in kernels:
        name = match.group(1)
        end = source.find(".Lfunc_end", match.end())
        if end == -1:
            raise ValueError(f"missing function end for {name}")
        in_asm, bad = False, []
        for line in source[match.end() : end].splitlines():
            if ";;#ASMSTART" in line:
                if in_asm:
                    raise ValueError(f"nested inline assembly in {name}")
                in_asm = True
                continue
            if ";;#ASMEND" in line:
                if not in_asm:
                    raise ValueError(f"unmatched inline assembly end in {name}")
                in_asm = False
                continue
            instruction = line.split(";")[0].strip()
            if (
                in_asm
                or not instruction
                or instruction.startswith(".")
                or instruction.endswith(":")
            ):
                continue
            vgprs = re.findall(r"\bv(?:\d+\b|\[\d+(?::\d+)?\])", instruction)
            reserved_vgpr = any(
                int(reg) >= 128
                for operand in vgprs
                for reg in re.findall(r"\d+", operand)
            )
            agpr = re.search(
                r"\ba(?:\d+\b|\[\d+(?::\d+)?\])|\b\w*accvgpr\w*\b", instruction
            )
            if reserved_vgpr or agpr:
                bad.append(instruction)
        if in_asm:
            raise ValueError(f"unterminated inline assembly in {name}")
        bad_total += len(bad)
        print(
            f"{'BAD' if bad else 'OK'} {name}: "
            f"{len(bad)} compiler instructions touch pinned registers"
            + (f"; examples: {bad[:3]}" if bad else "")
        )
    spills = [int(n) for n in re.findall(r"\.vgpr_spill_count:\s*(\d+)", source)]
    if len(spills) < len(kernels):
        raise ValueError("missing VGPR spill metadata")
    if any(spills):
        raise ValueError(f"nonzero VGPR spill counts: {spills}")
    if bad_total:
        raise ValueError(f"{bad_total} compiler instructions touch pinned registers")
    print(
        "PASS: all four GEMM variants avoid pinned registers; all VGPR spill counts are zero"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--build-dir", type=Path, help="JIT directory containing build.ninja"
    )
    mode.add_argument("--assembly", type=Path, help="check an existing assembly file")
    parser.add_argument(
        "--output", type=Path, help="retain emitted assembly (--build-dir only)"
    )
    args = parser.parse_args()
    if args.output and not args.build_dir:
        parser.error("--output requires --build-dir")
    try:
        if args.assembly:
            check_assembly(args.assembly)
        else:
            with tempfile.TemporaryDirectory(prefix="hk-moe-codegen-") as temp:
                output = args.output or Path(temp) / "hk_moe.s"
                emit_assembly(args.build_dir.resolve(), output)
                check_assembly(output)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
