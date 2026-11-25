# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""GPU test suites per MoE migration path.

A path is the unit that moves from the legacy code to aiter.moev2.
Before a path graduates, its suites must pass with `AITER_MOEV2=<path>`; `SHARED`
suites run for every path.

    python tests/moe_dispatch/gpu_suites.py                 # list paths and commands
    python tests/moe_dispatch/gpu_suites.py --path a8w8_pt  # commands for one path
    python tests/moe_dispatch/gpu_suites.py --path a8w8_pt --run      # with the path
    python tests/moe_dispatch/gpu_suites.py --path a8w8_pt --compare  # off, then the path

`--run` and `--compare` set AITER_MOEV2 themselves; a value in the caller's environment
is ignored. `--compare` runs each suite with `AITER_MOEV2=off` and with the path, writes
the output to `--log-dir` (default: a new temporary directory), and fails where v2 fails
and legacy passes. It also prints the checkAllclose outcomes of both runs and the checks
whose outcome differs; these do not set the exit status (a failed check need not fail
its script, and some checks are noisy run to run), so read them before graduating.
Commands match how .github/scripts/aiter_test.sh runs each file. `--run` and `--compare`
skip, and list, the suites whose archs or GPU count the machine does not have.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import argparse
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
from collections import Counter
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


def _passes(suite, mode, log=None):
    """Whether `suite` exits 0 with AITER_MOEV2=`mode`; its output goes to `log` if given."""
    env = os.environ.copy()
    env["AITER_MOEV2"] = mode
    # A script's own directory heads its sys.path, so without this `aiter` would be
    # whatever is installed.
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_REPO)] + [p for p in [env.get("PYTHONPATH")] if p]
    )
    if log is None:
        proc = subprocess.run(suite.cmd, check=False, env=env, cwd=_REPO)
    else:
        with open(log, "w") as out:
            proc = subprocess.run(
                suite.cmd,
                check=False,
                env=env,
                cwd=_REPO,
                stdout=out,
                stderr=subprocess.STDOUT,
            )
    return proc.returncode == 0


_CHECK = re.compile(
    r"(?P<msg>.*)\[checkAllclose (?P<tol>[^\x1b\]]*?) ?\x1b\[\d+m"
    r"(?P<outcome>passed|failed|warning|catastrophic)"
)
# Timings in a check's message change from run to run.
_TIMING = re.compile(r"\d+(?:\.\d+)?\s*(?:us|ms|tflops|TB/s|GB/s)\b,?")


def checks(log):
    """(check, outcome) for every checkAllclose line in `log`, timings left out."""
    found = []
    for line in pathlib.Path(log).read_text(errors="replace").splitlines():
        match = _CHECK.search(line)
        if match:
            msg = _TIMING.sub("", match["msg"]).removeprefix("[aiter] ")
            msg = " ".join(msg.split())
            found.append((f"{msg} [{match['tol']}]", match["outcome"]))
    return found


def _check_report(off_log, v2_log, limit=20):
    """Outcome counts of both runs, then the checks whose outcome differs.

    Checks pair up by position: both runs make the same checks in the same order, and
    a message can hold values that change between runs (perf checks print timings).
    """
    off, v2 = checks(off_log), checks(v2_log)

    def counts(found):
        tally = Counter(outcome for _, outcome in found)
        return ", ".join(f"{n} {o}" for o, n in sorted(tally.items())) or "none"

    lines = [f"    checkAllclose: off {counts(off)}; v2 {counts(v2)}"]
    if len(off) != len(v2):
        lines.append(
            f"      the runs made {len(off)} and {len(v2)} checks; compare the logs"
        )
        return lines
    differ = [(check, a, b) for (check, a), (_, b) in zip(off, v2) if a != b]
    for check, a, b in differ[:limit]:
        lines.append(f"      off {a}, v2 {b}: {check}")
    if len(differ) > limit:
        lines.append(f"      ... {len(differ) - limit} more; see the logs")
    return lines


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--path", choices=sorted(PATHS))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--run", action="store_true", help="run with the path")
    mode.add_argument(
        "--compare", action="store_true", help="run with off, then with the path"
    )
    parser.add_argument("--log-dir", help="--compare output (default: a temp dir)")
    args = parser.parse_args(argv)
    if args.compare and not args.path:
        parser.error("--compare needs --path")
    running = args.run or args.compare
    if running:
        gfx, gpus = _device()
    if args.compare:
        from aiter.moev2 import paths as v2_paths

        if args.path not in v2_paths.registered():
            # AITER_MOEV2=<path> would run legacy in both modes.
            print(f"{args.path} is not implemented: nothing to compare")
            return 1
        log_dir = pathlib.Path(
            args.log_dir or tempfile.mkdtemp(prefix="moe_suite_logs_")
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"logs: {log_dir}")
    paths = [args.path] if args.path else list(PATHS)
    ran, failed, skipped = 0, [], []
    # The full listing prints SHARED once at the end; a run needs it for every path.
    full = args.path or running
    for path in paths:
        print(f"[{path}]")
        for i, suite in enumerate(suites_for(path) if full else PATHS[path]):
            line = shlex.join(suite.cmd)
            extra = f"  # {', '.join(suite.archs)}" + (
                f"; {suite.note}" if suite.note else ""
            )
            # The suites write to the same stdout; flush so their output follows this line.
            print(f"  {line}{extra}", flush=True)
            if not running:
                continue
            if gfx not in suite.archs or gpus < suite.gpus:
                skipped.append(
                    f"[{path}] {line}  # needs {', '.join(suite.archs)} and"
                    f" {suite.gpus} GPU(s)"
                )
                continue
            ran += 1
            if args.run:
                if not _passes(suite, path):
                    failed.append(f"[{path}] {line}")
                continue
            logs = {m: log_dir / f"{path}.{i}.{m}.log" for m in ("off", path)}
            ok = {m: _passes(suite, m, log) for m, log in logs.items()}
            word = {True: "pass", False: "FAIL"}
            print(f"    off {word[ok['off']]}, v2 {word[ok[path]]}")
            print("\n".join(_check_report(logs["off"], logs[path])), flush=True)
            if ok["off"] and not ok[path]:
                failed.append(f"[{path}] {line}  # passes with off")
    if not full:
        print("[shared, every path]")
        for suite in SHARED:
            print(f"  {shlex.join(suite.cmd)}")
    if running:
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
