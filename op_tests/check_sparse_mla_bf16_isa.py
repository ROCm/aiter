# SPDX-License-Identifier: MIT
"""Gate retained gfx950 compiler assembly before running pinned-VGPR kernels.

Compile sparse_mla_bf16.cu with --save-temps, then pass its device .s here.
Checks compiler-managed VGPR exclusion, FP32 denormals, spills and private memory.
This is a conservative build check, not a proof of synchronization or liveness.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path


def audit(path):
    text = path.read_text()
    metadata = {}
    for block in re.split(r"(?m)^  - (?=\.)", text.split("\t.amdgpu_metadata", 1)[-1])[
        1:
    ]:
        fields = dict(re.findall(r"(?m)^\s*\.(\w+):\s*([^\n]+)", block))
        if "name" in fields:
            metadata[fields["name"].strip()] = fields
    errors, kernels = [], {}
    current, inline = None, False
    for number, line in enumerate(text.splitlines(), 1):
        function = re.match(r"\s*\.type\s+(\S+),@function", line)
        if function:
            if inline:
                errors.append(f"unclosed inline asm before {number}")
            current, inline = function[1], False
            if "sparse_mla_bf16" in current and current in metadata:
                kernels[current] = {"compiler_vgpr_max": -1, "denorm32": None}
        if re.search(r"#(?:ASMSTART|APP)\b", line):
            inline = True
            continue
        if re.search(r"#(?:ASMEND|NO_APP)\b", line):
            inline = False
            continue
        if current not in kernels:
            continue
        info = kernels[current]
        mode = re.search(r"\.amdhsa_float_denorm_mode_32\s+(\d+)", line)
        if mode:
            info["denorm32"] = int(mode[1])
        instruction = re.match(r"^\s+([a-z][a-z0-9_]*)\s*(.*)$", line)
        if not instruction:
            continue
        if instruction[1].startswith("scratch_"):
            errors.append(f"{current}:{number}: scratch instruction")
        if not inline:
            registers = [
                int(b or a or c, 0)
                for a, b, c in re.findall(
                    r"\bv(?:\[(0x[\da-fA-F]+|\d+)(?::(0x[\da-fA-F]+|\d+))?\]|(\d+))",
                    line,
                )
            ]
            info["compiler_vgpr_max"] = max(info["compiler_vgpr_max"], *registers, -1)
    main = [name for name in kernels if "sparse_mla_v" in name]
    tiles = [name for name in kernels if "sparse_mla_headtiles" in name]
    if len(main) != 8 or len(tiles) != 2:
        errors.append(
            f"expected 8 v1/v2 and 2 head-tiled mains, got {len(main)} and {len(tiles)}"
        )
    if len(kernels) != 21:
        errors.append(f"expected 21 kernels, got {len(kernels)}")
    for name, info in kernels.items():
        if name in main + tiles and info["compiler_vgpr_max"] >= 40:
            errors.append(f"{name}: compiler crosses pinned v40 boundary")
        if info["denorm32"] != 3:
            errors.append(f"{name}: FP32 denormals are not preserved")
        for field in (
            "vgpr_count",
            "agpr_count",
            "sgpr_count",
            "vgpr_spill_count",
            "sgpr_spill_count",
            "private_segment_fixed_size",
            "group_segment_fixed_size",
        ):
            value = metadata[name].get(field, "")
            info[field] = int(value) if value.isdecimal() else None
            if info[field] is None:
                errors.append(f"{name}: missing {field}")
        for field in (
            "vgpr_spill_count",
            "sgpr_spill_count",
            "private_segment_fixed_size",
        ):
            if info[field] != 0:
                errors.append(f"{name}: {field} must be zero")
    return {
        "passed": not errors,
        "errors": errors,
        "kernels": kernels,
        "assembly_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.assembly)
    report = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report)
    raise SystemExit(not result["passed"])
