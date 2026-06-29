# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Generate Opus MoE stage2 dispatch headers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from opus_moe_common import STAGE2_BF16_KERNELS  # noqa: E402

MANIFEST_HEADER = """#pragma once
// SPDX-License-Identifier: MIT
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
//
// Auto-generated. Do not edit. See csrc/opus_moe/gen_instances.py.
//
// BF16 stage2 kid -> launcher manifest. This is deliberately generated from
// opus_moe_common.py so Python tuner metadata and C++ dispatch tables do not
// drift as more stage2 kids land.

"""


def _emit_bf16_manifest_header() -> str:
    lines = [MANIFEST_HEADER]
    bf16_kernels = [STAGE2_BF16_KERNELS[kid] for kid in sorted(STAGE2_BF16_KERNELS)]

    lines.append(f"#define OPUS_MOE_STAGE2_BF16_TUNE_LOOKUP_SIZE {len(bf16_kernels)}\n")
    if not bf16_kernels:
        lines.append("#define GENERATE_OPUS_MOE_STAGE2_BF16_TUNE_LOOKUP\n\n")
    else:
        lines.append("#define GENERATE_OPUS_MOE_STAGE2_BF16_TUNE_LOOKUP \\\n")
        for idx, inst in enumerate(bf16_kernels):
            suffix = " \\\n" if idx != len(bf16_kernels) - 1 else "\n"
            lines.append(
                "    {"
                f"{inst.kid}, "
                f"&{inst.launcher}<"
                f"{inst.trait}>"
                "},"
                + suffix
            )
    lines.append("\n")

    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Opus MoE stage2 dispatch headers"
    )
    parser.add_argument("--working_path", required=True)
    parser.add_argument(
        "--tune_files", default="", help="Accepted for JIT compatibility."
    )
    parser.add_argument(
        "--tune_file", default=None, help="Deprecated alias for --tune_files."
    )
    parser.add_argument("--arch", default=None, help="Optional arch filter")
    parser.add_argument(
        "--cu-num", type=int, default=None, help="Optional CU-count filter"
    )
    args = parser.parse_args()

    out_dir = Path(args.working_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    bf16_manifest_path = out_dir / "opus_moe_stage2_manifest.h"
    bf16_manifest_path.write_text(_emit_bf16_manifest_header(), encoding="utf-8")

    print(
        f"[opus_moe gen_instances] wrote {bf16_manifest_path} with "
        f"{len(STAGE2_BF16_KERNELS)} BF16 stage2 kid(s)"
    )


if __name__ == "__main__":
    main()
