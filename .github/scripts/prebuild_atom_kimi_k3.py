#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Build Kimi-K3's slow GEMM dependency before the ATOM startup deadline."""

import importlib
import sys
from pathlib import Path

MODULE = "module_gemm_a8w8_bpreshuffle_cktile"
EXPECTED_ARCH = "gfx950"


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    # Use the build-time import path without initializing AITER's runtime.
    sys.path.insert(0, str(root / "aiter"))
    core = importlib.import_module("jit.core")

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
    expected = root / "aiter" / "jit" / target.name
    if target.resolve() != expected.resolve() or not target.is_file():
        raise RuntimeError(f"Prebuild did not create {expected}: got {target}")

    arches = core._so_offload_archs(target)
    if arches != {EXPECTED_ARCH}:
        raise RuntimeError(f"Unexpected GPU architectures: {sorted(arches)}")
    print(
        f"Prebuilt {MODULE}: {target.stat().st_size} bytes, {EXPECTED_ARCH}", flush=True
    )


if __name__ == "__main__":
    main()
