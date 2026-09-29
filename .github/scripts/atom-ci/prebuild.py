#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Prebuild a known slow ATOM dependency outside the inference watchdog."""

import importlib
import os
import sys
from pathlib import Path


def build(core, root, module):
    args = core.get_args_of_build(module)
    core.build_module(
        md_name=module,
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
    target = Path(core.get_user_jit_dir()) / f"{module}.so"
    expected = root / "aiter" / "jit" / target.name
    if target.resolve() != expected.resolve() or not target.is_file():
        raise RuntimeError(f"Prebuild did not create {expected}: got {target}")
    arches = core._so_offload_archs(target)
    if arches != {"gfx950"}:
        raise RuntimeError(f"Unexpected prebuild architectures: {sorted(arches)}")
    print(f"Prebuilt {module}: {target.stat().st_size} bytes, gfx950", flush=True)


def main():
    root = Path(sys.argv[1]).resolve()
    module = sys.argv[2]
    if not (root / "aiter" / "jit" / "core.py").is_file():
        raise SystemExit(f"Not an AITER source tree: {root}")
    # Match the existing targeted vLLM prebuild's build-time import path.
    sys.path.insert(0, str(root / "aiter"))
    workers = len(os.sched_getaffinity(0))
    os.environ["MAX_JOBS"] = str(workers)
    print(f"Prebuild workers: {workers} (allocated CPU affinity)", flush=True)
    core = importlib.import_module("jit.core")
    build(core, root, module)


if __name__ == "__main__":
    main()
