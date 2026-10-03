#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Generate the per-family Opus MoE backward dispatch manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from opus_moe_backward_common import (  # noqa: E402
    OPUS_MOE_BACKWARD_INSTANCES,
    OpusMoeBackwardFamily,
    OpusMoeBackwardInstance,
    validate_instances,
)


_HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit.
// See csrc/opus_moe/moe_backward/gen_instances.py.
//
// Each entry expands to {kid, name, launcher}.  Empty tables are deliberate:
// they make an unimplemented family fail in host dispatch rather than silently
// selecting a placeholder kernel.

"""


def _cpp_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def emit_manifest(
    instances: tuple[OpusMoeBackwardInstance, ...] = OPUS_MOE_BACKWARD_INSTANCES,
    *,
    arch: str = "gfx950",
) -> str:
    validated = validate_instances(instances)
    if arch != "gfx950":
        raise ValueError(f"Opus MoE backward only supports gfx950, got {arch!r}")
    selected = tuple(inst for inst in validated if inst.arch == arch)

    lines = [_HEADER]
    lines.append(f"#define OPUS_MOE_BACKWARD_MANIFEST_INSTANCE_COUNT {len(selected)}\n\n")
    for family in OpusMoeBackwardFamily:
        family_instances = tuple(
            inst for inst in selected if inst.family is family
        )
        prefix = f"OPUS_MOE_BACKWARD_{family.macro_name}_MANIFEST"
        lines.append(f"#define {prefix}_SIZE {len(family_instances)}\n")
        if not family_instances:
            lines.append(f"#define {prefix}_ENTRIES\n\n")
            continue

        lines.append(f"#define {prefix}_ENTRIES \\\n")
        for index, inst in enumerate(family_instances):
            suffix = ", \\\n" if index + 1 != len(family_instances) else "\n"
            lines.append(
                "    {"
                f"{inst.kid}, \"{_cpp_string(inst.name)}\", "
                f"&{inst.launcher}<{inst.trait}>"
                "}" + suffix
            )
        lines.append("\n")
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Opus MoE backward dispatch headers"
    )
    parser.add_argument("--working_path", required=True)
    parser.add_argument(
        "--arch", default="gfx950", help="Target architecture (only gfx950 today)"
    )
    parser.add_argument(
        "--tune_files", default="", help="Reserved for JIT generator compatibility"
    )
    parser.add_argument(
        "--cu-num", type=int, default=None, help="Reserved for JIT compatibility"
    )
    args = parser.parse_args()

    out_dir = Path(args.working_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "opus_moe_backward_manifest.h"
    out_path.write_text(emit_manifest(arch=args.arch), encoding="utf-8")
    print(
        f"[opus_moe_backward gen_instances] wrote {out_path} with "
        f"{len(OPUS_MOE_BACKWARD_INSTANCES)} kernel instance(s)"
    )


if __name__ == "__main__":
    main()
