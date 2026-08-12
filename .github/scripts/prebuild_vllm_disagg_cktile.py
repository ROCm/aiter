#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prebuild the CKTile module required by the vLLM disaggregated smoke test."""

import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
# Match setup.py's build-time import path.  Importing ``aiter.jit`` would run
# aiter/__init__.py and probe the live GPU while initializing runtime dtypes;
# build-only CI runners intentionally have no GPU.
sys.path.insert(0, str(REPO_ROOT / "aiter"))

core = importlib.import_module("jit.core")

MODULE = "module_gemm_a8w8_blockscale_cktile"
EXPECTED_ARCH = "gfx950"


def main() -> None:
    args = core.get_args_of_build(MODULE)
    core.build_module(
        md_name=MODULE,
        srcs=args["srcs"],
        flags_extra_cc=args["flags_extra_cc"],
        flags_extra_hip=args["flags_extra_hip"],
        blob_gen_cmd=args["blob_gen_cmd"],
        extra_include=args["extra_include"],
        extra_ldflags=args["extra_ldflags"],
        verbose=args["verbose"],
        is_python_module=args["is_python_module"],
        is_standalone=args["is_standalone"],
        torch_exclude=args["torch_exclude"],
        third_party=args.get("third_party", []),
        hipify=args.get("hipify", False),
        flags_extra_hip_per_source=args.get("flags_extra_hip_per_source", {}),
    )

    target = Path(core.get_user_jit_dir()) / f"{MODULE}.so"
    expected = REPO_ROOT / "aiter" / "jit" / target.name
    if target.resolve() != expected.resolve() or not target.is_file():
        raise SystemExit(f"targeted prebuild did not create {expected}: got {target}")

    arches = core._so_offload_archs(target)
    if arches != {EXPECTED_ARCH}:
        raise SystemExit(
            f"targeted prebuild has unexpected GPU arches: {sorted(arches)}"
        )
    print(
        f"Targeted prebuild ready: {target} "
        f"({target.stat().st_size} bytes, arches={sorted(arches)})"
    )


if __name__ == "__main__":
    main()
