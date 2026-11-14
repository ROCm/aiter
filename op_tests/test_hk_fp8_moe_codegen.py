# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Audit FP8 MoE assembly using the AITER JIT compile command.

Compiler instructions must avoid v128-v255 and all AGPRs: these registers hold
HipKittens ART tiles. Only explicit tile instruction forms are exempt; their
compiler-selected operands must still avoid pinned registers. Scratch is forbidden. This
check supplements runtime correctness tests; zero spills alone do not prove that
the compiler avoids pinned registers. Every GEMM descriptor must retain 256 AGPRs
and the full 512-register vector allocation. With no arguments, build and audit on
gfx950; --build-dir and --assembly need neither a PyTorch import nor a GPU.
"""

import argparse
import importlib.util
import inspect
import subprocess
import tempfile
from pathlib import Path

# Load without importing aiter/torch so saved-assembly audits also work on CPUs.
_PATH = Path(__file__).resolve().parents[1] / "aiter/jit/utils/hk_fp8_moe_codegen.py"
_SPEC = importlib.util.spec_from_file_location("hk_fp8_moe_codegen", _PATH)
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)
check_assembly, emit_assembly = _AUDIT.check_assembly, _AUDIT.emit_assembly


def build_for_audit():
    import torch

    if (
        not torch.version.hip
        or not torch.cuda.is_available()
        or torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).gcnArchName.split(":")[0]
        != "gfx950"
    ):
        print("Skipping test_hk_fp8_moe_codegen.py: requires gfx950")
        return None
    from aiter.jit.core import bd_dir, build_module, get_args_of_build

    module = "module_hk_fp8_moe"
    args = get_args_of_build(module)
    args["md_name"] = module
    build_module(
        **{
            k: v
            for k, v in args.items()
            if k in inspect.signature(build_module).parameters
        }
    )
    return Path(bd_dir) / module / "build"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--build-dir", type=Path, help="JIT directory containing build.ninja"
    )
    mode.add_argument("--assembly", type=Path, help="check an existing assembly file")
    parser.add_argument("--output", type=Path, help="retain emitted assembly")
    args = parser.parse_args()
    if args.output and args.assembly:
        parser.error("--output cannot be used with --assembly")
    try:
        if args.assembly:
            check_assembly(args.assembly)
        else:
            build_dir = args.build_dir or build_for_audit()
            if build_dir is None:
                return
            with tempfile.TemporaryDirectory(prefix="hk-moe-codegen-") as temp:
                output = args.output or Path(temp) / "hk_moe.s"
                emit_assembly(build_dir.resolve(), output)
                check_assembly(output)
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"FAIL: {error}\n")


if __name__ == "__main__":
    main()
