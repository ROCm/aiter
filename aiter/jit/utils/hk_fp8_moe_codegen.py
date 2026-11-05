# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fail closed on unsafe register allocation in the gfx950 HipKittens MoE."""

import json
import re
import shlex
import subprocess
from pathlib import Path

_NUMBER = r"(?:0x[0-9a-fA-F]+|[0-9]+)"
_REGISTER = re.compile(rf"\b([va])(?:({_NUMBER})\b|\[({_NUMBER})(?::({_NUMBER}))?\])")


def _registers(instruction):
    def number(s):
        return int(s, 16 if s.startswith("0x") else 10)

    return [
        (kind, number(single or lo), number(single or hi or lo))
        for kind, single, lo, hi in _REGISTER.findall(instruction)
    ]


def _tile_instruction(instruction):
    """Only the four explicit pinned-register macro forms used by this kernel.

    Compiler-chosen address, scale, and copy operands must still be below v128.
    These exemptions apply only inside an inline-assembly block.
    """
    v = rf"v(?:{_NUMBER}|\[{_NUMBER}\])"
    a = rf"a(?:{_NUMBER}|\[{_NUMBER}\])"
    vr = rf"v\[{_NUMBER}:{_NUMBER}\]"
    ar = rf"a\[{_NUMBER}:{_NUMBER}\]"
    regs = _registers(instruction)

    def temporary(reg):
        kind, lo, hi = reg
        return kind == "v" and 0 <= lo == hi < 128

    def accumulator(reg, width=1):
        kind, lo, hi = reg
        return (
            kind == "a"
            and 0 <= lo <= hi < 256
            and hi - lo + 1 == width
            and lo % width == 0
        )

    if re.fullmatch(
        rf"ds_read_b128\s+{vr},\s*{v}(?:\s+offset:{_NUMBER})?", instruction
    ):
        kind, lo, hi = regs[0]
        return (
            kind == "v"
            and 128 <= lo <= hi < 256
            and hi - lo == 3
            and lo % 4 == 0
            and temporary(regs[1])
        )
    if re.fullmatch(rf"v_accvgpr_read_b32\s+{v},\s*{a}", instruction):
        return temporary(regs[0]) and accumulator(regs[1])
    if re.fullmatch(rf"v_accvgpr_write_b32\s+{a},\s*(?:0|{v})", instruction):
        return accumulator(regs[0]) and (len(regs) == 1 or temporary(regs[1]))
    if re.fullmatch(
        rf"v_mfma_scale_f32_16x16x128_f8f6f4\s+{ar},\s*{vr},\s*{vr},\s*{ar},\s*{v},\s*{v}"
        r"(?:\s+op_sel:\[[0-3],[0-3],[0-3]\])?(?:\s+op_sel_hi:\[[01],[01],[01]\])?",
        instruction,
    ):
        acc, weight, act, source, scale_w, scale_a = regs
        return (
            accumulator(acc, 4)
            and acc == source
            and weight[0] == act[0] == "v"
            and 192 <= weight[1] <= weight[2] < 256
            and 128 <= act[1] <= act[2] < 192
            and weight[2] - weight[1] == act[2] - act[1] == 7
            and weight[1] % 8 == act[1] % 8 == 0
            and temporary(scale_w)
            and temporary(scale_a)
        )
    return False


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
    # hipcc cannot emit a single -S output for multiple targets. Preserve the
    # gfx950 compilation flags while dropping the other fat-binary targets.
    targets = [flag for flag in command if flag.startswith("--offload-arch=")]
    if not any(
        flag in targets for flag in ("--offload-arch=gfx950", "--offload-arch=native")
    ):
        raise ValueError("expected a gfx950 or native compile target")
    command = [flag for flag in command if flag not in targets]
    command.append("--offload-arch=gfx950")
    command.remove("-c")
    command[command.index("-o") + 1] = str(output.resolve())
    command += ["--cuda-device-only", "-S"]
    print(shlex.join(command), flush=True)
    subprocess.run(command, cwd=entry["directory"], check=True)


def check_assembly(path: Path) -> None:
    source = path.read_text()
    kernels = list(re.finditer(r"^(_Z\w*moe_gemm_kernel\w*):", source, re.MULTILINE))
    if len(kernels) != 4:
        raise ValueError(f"expected four GEMM variants, found {len(kernels)}")
    metadata = {}
    for block in re.split(r"^  - ", source, flags=re.MULTILINE)[1:]:
        name = re.search(r"^\s+\.name:\s*(\S+)", block, re.MULTILINE)
        if name:
            metadata[name.group(1)] = block
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
                not instruction
                or instruction.startswith(".")
                or instruction.endswith(":")
            ):
                continue
            scratch = re.search(r"\b(?:scratch_\w*|flat_scratch\w*)\b", instruction)
            pinned = any(
                kind == "a" or hi >= 128 for kind, lo, hi in _registers(instruction)
            )
            if scratch or (pinned and not (in_asm and _tile_instruction(instruction))):
                bad.append(instruction)
        if in_asm:
            raise ValueError(f"unterminated inline assembly in {name}")
        if bad:
            raise ValueError(
                f"{name}: {len(bad)} unsafe register/scratch instructions; examples: {bad[:3]}"
            )
        block = metadata.get(name, "")
        # SGPR-to-VGPR lane spills are allowed only within the checked temporary
        # registers. Memory-backed spills and dynamic stacks are never allowed.
        for key, expected in (
            ("vgpr_spill_count", "0"),
            ("private_segment_fixed_size", "0"),
            ("uses_dynamic_stack", "false"),
        ):
            value = re.search(rf"^\s+\.{key}:\s*(\S+)", block, re.MULTILINE)
            if value is None or value.group(1) != expected:
                raise ValueError(f"{name}: missing or unsafe {key} metadata")
        print(f"OK {name}: pinned registers and zero scratch verified")
    print(
        "PASS: all four GEMM variants have safe pinned-register allocation and no scratch"
    )


def audit_build(build_dir: Path) -> None:
    output = build_dir / "hk_moe.audit.s"
    emit_assembly(build_dir, output)
    check_assembly(output)
