# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU test suites per MoE migration path.

A path is the unit that moves from the legacy code to aiter.moev2.
Before a path graduates, its suites must pass with `AITER_MOEV2=<path>`; `SHARED`
suites run for every path.

    python tests/moe_dispatch/gpu_suites.py                 # list paths and commands
    python tests/moe_dispatch/gpu_suites.py --path a8w8_pt  # commands for one path
    python tests/moe_dispatch/gpu_suites.py --path a8w8_pt --run

Commands match how .github/scripts/aiter_test.sh runs each file. `--run` skips, and lists,
the suites whose archs or GPU count the machine does not have.
"""

import argparse
import os
import pathlib
import shlex
import subprocess
import sys
from dataclasses import dataclass

_REPO = pathlib.Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Suite:
    cmd: tuple
    archs: tuple = ("gfx942", "gfx950")
    gpus: int = 1
    note: str = ""


def _py(file, *args, **kw):
    return Suite(("python3", file, *args), **kw)


def _pytest(file, **kw):
    return Suite(("python3", "-m", "pytest", "-q", file), **kw)


def _torchrun(file, *args, **kw):
    return Suite(
        ("torchrun", "--standalone", "--nproc_per_node=8", file, *args), gpus=8, **kw
    )


_MOE_2STAGE = "op_tests/test_moe_2stage.py"
_LEGACY_SWEEP = ("--no-flydsl-csv",)

SHARED = (
    _py("op_tests/test_moe_sorting.py"),
    _py("op_tests/test_moe_ep.py"),
    _pytest("op_tests/test_moe_local_expert_ids.py"),
    _py("op_tests/test_moe_dp_share_expert.py"),
)

PATHS = {
    "a16w16": (
        _py(_MOE_2STAGE, "-q", "0", *_LEGACY_SWEEP),
        _pytest("op_tests/test_moe_a16w16_default_dispatch.py", archs=("gfx942",)),
        _py("op_tests/test_flydsl_moe_gfx942.py", archs=("gfx942",)),
        _py("op_tests/test_moe.py"),
    ),
    "a8w8_pt": (
        _py(_MOE_2STAGE, "-q", "1", *_LEGACY_SWEEP),
        _pytest("op_tests/test_moe_per_tensor_a2_init.py"),
    ),
    "a8w8_ptok": (
        _py(_MOE_2STAGE, "-q", "2", *_LEGACY_SWEEP),
        _py(_MOE_2STAGE, "-q", "3", *_LEGACY_SWEEP),
        _py("op_tests/test_moe_tkw1.py"),
    ),
    "a8w8_blockscale": (
        _py(_MOE_2STAGE, "-q", "5", *_LEGACY_SWEEP),
        _py("op_tests/test_moe_blockscale.py"),
    ),
    "a4w4": (
        _py(_MOE_2STAGE, "-q", "4", *_LEGACY_SWEEP, archs=("gfx950",)),
        _py(
            _MOE_2STAGE,
            "--no-legacy",
            archs=("gfx950",),
            note="tuned FlyDSL/Opus CSV rows",
        ),
        _py("op_tests/test_moe_sorting_mxfp4.py", archs=("gfx950",)),
        _py("op_tests/test_flydsl_moe_aux.py", archs=("gfx950",)),
        _py("op_tests/test_moe_mxfp4_inter_dim_dispatch.py", archs=("gfx950",)),
        _py("op_tests/test_mxfp4_moe_aux_codegen.py", archs=("gfx950",)),
        _py("op_tests/test_moe_glm5_fp4_ep_dispatch.py", archs=("gfx950",)),
        _pytest("op_tests/flydsl_tests/test_flydsl_moe.py", archs=("gfx950",)),
    ),
    "a8w4": (_py(_MOE_2STAGE, "-q", "7", *_LEGACY_SWEEP, archs=("gfx950",)),),
    "a16w4": (
        _py(_MOE_2STAGE, "-q", "6", *_LEGACY_SWEEP, archs=("gfx950",)),
        _pytest("op_tests/test_flydsl_moe_a16wfp4.py", archs=("gfx950",)),
        _pytest(
            "op_tests/flydsl_tests/test_moe_mxfp4_inter_dim_align.py", archs=("gfx950",)
        ),
    ),
    "a16wi4": (_py(_MOE_2STAGE, "-q", "8", *_LEGACY_SWEEP, archs=("gfx950",)),),
    "mxfp8": (
        _py(_MOE_2STAGE, "-q", "9", *_LEGACY_SWEEP, archs=("gfx950",)),
        _pytest("op_tests/test_moe_mxfp8_passthrough.py", archs=("gfx950",)),
    ),
    "fhmoe": (_pytest("op_tests/test_fhmoe.py", archs=("gfx950",)),),
    "comm_fused": (_torchrun("op_tests/multigpu_tests/test_comm_fused_moe.py"),),
    "mega_moe_v2": (
        _torchrun(
            "op_tests/multigpu_tests/test_mega_moe_v2.py",
            archs=("gfx950",),
            note="skipped by aiter_test.sh today; run by hand",
        ),
    ),
    "mega_moe_tp": (_torchrun("op_tests/multigpu_tests/test_mega_moe_tp.py"),),
    "mega_moe_gfx1250": (
        _torchrun(
            "op_tests/multigpu_tests/bench_mega_moe.py",
            "--combine",
            "fused",
            "--dispatch_backend",
            "flydsl",
            "--stage1_fused",
            "1",
            "--layers",
            "2",
            "--acc_verify",
            "1",
            archs=("gfx1250",),
            note="needs MORI_SHMEM_HEAP_SIZE=40G; the only gfx1250 accuracy check today",
        ),
    ),
}


def suites_for(path):
    return PATHS[path] + SHARED


def _device():
    """The running GPU's arch and the number of visible GPUs."""
    # The repository root so `aiter` is this checkout rather than an installed copy,
    # even one earlier on the path.
    sys.path[:] = [str(_REPO)] + [p for p in sys.path if p != str(_REPO)]
    import torch

    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime(), torch.cuda.device_count()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", choices=sorted(PATHS))
    parser.add_argument("--run", action="store_true", help="run the commands")
    args = parser.parse_args(argv)
    paths = [args.path] if args.path else list(PATHS)
    if args.run:
        gfx, gpus = _device()
    ran, failed, skipped = 0, [], []
    # The full listing prints SHARED once at the end; a run needs it for every path.
    full = args.path or args.run
    for path in paths:
        print(f"[{path}]")
        for suite in suites_for(path) if full else PATHS[path]:
            line = shlex.join(suite.cmd)
            extra = f"  # {', '.join(suite.archs)}" + (
                f"; {suite.note}" if suite.note else ""
            )
            print(f"  {line}{extra}")
            if not args.run:
                continue
            if gfx not in suite.archs or gpus < suite.gpus:
                skipped.append(
                    f"[{path}] {line}  # needs {', '.join(suite.archs)} and"
                    f" {suite.gpus} GPU(s)"
                )
                continue
            env = os.environ.copy()
            env["AITER_MOEV2"] = path
            # A script's own directory heads its sys.path, so without this `aiter` would be
            # whatever is installed.
            env["PYTHONPATH"] = os.pathsep.join(
                [str(_REPO)] + [p for p in [env.get("PYTHONPATH")] if p]
            )
            ran += 1
            proc = subprocess.run(suite.cmd, check=False, env=env, cwd=_REPO)
            if proc.returncode != 0:
                failed.append(f"[{path}] {line}")
    if not full:
        print("[shared, every path]")
        for suite in SHARED:
            print(f"  {shlex.join(suite.cmd)}")
    if args.run:
        for line in skipped:
            print(f"SKIPPED on {gfx} with {gpus} GPU(s): {line}")
        if not ran:
            print(f"no suite runs on {gfx} with {gpus} GPU(s)")
            return 1
    for line in failed:
        print(f"FAILED: {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
