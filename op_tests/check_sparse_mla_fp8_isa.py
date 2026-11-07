# SPDX-License-Identifier: MIT
"""Audit retained gfx950 device assembly for the FP8 sparse MLA operator.

Pass the device .s files from an AITER build with --save-temps. This checks
resource limits and arithmetic instructions; runtime correctness/replay tests
are still required. It does not prove all assembly dependency hazards.
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
    starts = list(re.finditer(r"(?m)^\s*\.type\s+(\S+),@function", text))
    for i, match in enumerate(starts):
        name = match[1]
        if name not in metadata or "sparse_mla_fp8" not in name:
            continue
        body = text[
            match.start() : starts[i + 1].start() if i + 1 < len(starts) else len(text)
        ]
        fields = metadata[name]
        info = {}
        for field in [
            "vgpr_count",
            "agpr_count",
            "sgpr_count",
            "vgpr_spill_count",
            "sgpr_spill_count",
            "private_segment_fixed_size",
            "group_segment_fixed_size",
        ]:
            value = fields.get(field, "")
            info[field] = int(value) if value.isdecimal() else None
            if info[field] is None:
                errors.append(f"{name}: missing {field}")
        for field in [
            "vgpr_spill_count",
            "sgpr_spill_count",
            "private_segment_fixed_size",
        ]:
            if info[field] != 0:
                errors.append(f"{name}: {field} must be zero")
        if (
            info["group_segment_fixed_size"] is not None
            and info["group_segment_fixed_size"] > 160 * 1024
        ):
            errors.append(f"{name}: exceeds gfx950 LDS capacity")
        modes = re.findall(r"\.amdhsa_float_denorm_mode_32\s+(\d+)", body)
        # Function descriptors can be emitted separately from the function body;
        # every descriptor in the input must preserve FP32 denormals.
        info["mfma"] = sorted(set(re.findall(r"\bv_mfma\w+", body)))
        if any(op != "v_mfma_scale_f32_16x16x128_f8f6f4" for op in info["mfma"]):
            errors.append(f"{name}: unexpected matrix instruction")
        if "attention" in name and not info["mfma"]:
            errors.append(f"{name}: missing FP8 matrix instructions")
        if re.search(r"(?m)^\s+scratch_", body):
            errors.append(f"{name}: scratch instructions")
        kernels[name] = info
    if not kernels:
        errors.append("no sparse MLA kernels found")
    modes = re.findall(r"\.amdhsa_float_denorm_mode_32\s+(\d+)", text)
    if not modes or any(mode != "3" for mode in modes):
        errors.append("FP32 denormals must be preserved")
    return {
        "passed": not errors,
        "errors": errors,
        "kernels": kernels,
        "assembly_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("assembly", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = {str(p): audit(p) for p in args.assembly}
    report = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(report)
    print(report)
    raise SystemExit(not all(v["passed"] for v in result.values()))
