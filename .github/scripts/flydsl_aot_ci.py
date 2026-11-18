#!/usr/bin/env python3
"""FlyDSL AOT CI for one GPU arch (gfx1250 unless ``--arch`` says otherwise).

``build`` runs every FlyDSL AOT module (``python -m aiter.aot.flydsl.<module>``,
the entry points ``setup.py``'s ``run_aot`` drives) with ``GPU_ARCHS`` naming
the arch, into a fresh FlyDSL cache, then runs each module again with
``--check``, which compiles nothing and fails on any kernel the runtime would
look up but the cache lacks. No GPU is needed: the replays run aiter's host
code on CPU (``aiter/aot/flydsl/dry_run.py``). The contract:

- every module exits 0, or every failure it reports is a known failure;
- each known failure (``--known-failures``: ``<module> <count> <regex>``
  lines) matches exactly ``<count>`` of the module's ``[FAIL]`` lines in
  every pass, so a fixed or a new failure turns the run red for review;
- the cache holds at least one artifact when any module had jobs, and every
  artifact was compiled for the arch.

``load`` needs a GPU of the arch: it loads every code object in the cache
with ``hipModuleLoadData`` and looks up every kernel it launches.

Both write ``summary.json``, ``summary.md`` and per-module logs under
``--out-dir``, and append ``summary.md`` to ``$GITHUB_STEP_SUMMARY`` when set.
Exit status: 0 when the contract holds, 1 when it does not, 2 for bad usage.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
AOT_DIR = REPO_ROOT / "aiter" / "aot" / "flydsl"
JIT_DIR = REPO_ROOT / "aiter" / "jit"
# The C++ modules whose host code the replays call. Built for ``--host-arch``
# before the replays: not every toolchain builds them for every arch (ROCm
# 7.2's HIP lacks the cluster launch API a gfx1250 build turns on).
HOST_CTYPES_MODULES = ("module_gemm_common",)

_JOBS = re.compile(
    r"^\[aiter\] FlyDSL (?P<label>.+?) replay: (?:\d+ rows, )?"
    r"(?P<jobs>\d+) jobs, archs (?P<archs>.+)$"
)
_SUMMARY = re.compile(
    r"^\[aiter\] FlyDSL (?P<label>.+?) (?:replay|cache check)(?: (?P<arch>\S+))?: "
    r"(?P<ok>\d+) ok, (?P<failed>\d+) failed"
)
_PROBES = re.compile(
    r"^\s+\[(?P<tag>BISECT|SCAN)\] (?P<n>\d+) .*, (?P<failed>\d+) failed$"
)
_FAIL = re.compile(r"\s*\[FAIL\] ")
_CHIP = re.compile(r'#rocdl\.target<[^>]*chip\s*=\s*"([^"]+)"')
_ECHO = re.compile(
    r"^\s*\[(FAIL|SKIP|REFUSED|UNREACHED|BISECT|SCAN)\]|^\[aiter\] FlyDSL"
)


def default_modules() -> list[str]:
    """The modules ``run_aot`` replays, in its order: MoE, MegaMoE, then
    ``AOT_MODULES``, read from ``common.py`` without importing aiter."""
    tree = ast.parse((AOT_DIR / "common.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "AOT_MODULES"
            for target in node.targets
        ):
            names = ast.literal_eval(node.value)
            return ["moe", "mega_moe", *(name.rsplit(".", 1)[-1] for name in names)]
    raise SystemExit(f"AOT_MODULES not found in {AOT_DIR / 'common.py'}")


# ---------------------------------------------------------------------------
# Known failures
# ---------------------------------------------------------------------------


@dataclass
class KnownFailure:
    module: str
    count: int
    pattern: re.Pattern
    line: int

    def __str__(self) -> str:
        return f"{self.module} {self.count} {self.pattern.pattern}"


def load_known_failures(path: Path | None) -> list[KnownFailure]:
    if path is None:
        return []
    entries = []
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 2)
        if len(parts) != 3 or not parts[1].isdigit() or int(parts[1]) < 1:
            raise ValueError(f"{path}:{number}: expected '<module> <count> <regex>'")
        try:
            pattern = re.compile(parts[2])
        except re.error as error:
            raise ValueError(f"{path}:{number}: bad regex: {error}") from None
        entries.append(KnownFailure(parts[0], int(parts[1]), pattern, number))
    return entries


# ---------------------------------------------------------------------------
# One module pass
# ---------------------------------------------------------------------------


@dataclass
class PassResult:
    module: str
    phase: str  # "compile" or "check"
    returncode: int
    seconds: float
    log: str
    jobs: int = 0
    ok: int = 0
    failed: int = 0
    fails: list[str] = field(default_factory=list)
    status: str = ""
    problems: list[str] = field(default_factory=list)


def parse_log(result: PassResult, text: str) -> None:
    for line in text.splitlines():
        if match := _JOBS.match(line):
            result.jobs += int(match["jobs"])
        elif match := _SUMMARY.match(line):
            result.ok += int(match["ok"])
            result.failed += int(match["failed"])
        elif match := _PROBES.match(line):
            result.failed += int(match["failed"])
        # Every marker of the line: lines of parallel workers can share one.
        result.fails += [text.rstrip() for text in _FAIL.split(line)[1:]]


def judge(result: PassResult, known: list[KnownFailure]) -> None:
    """Set ``result.status``: ``ok``, ``known`` (only known failures) or
    ``failed``, with the reasons in ``result.problems``."""
    entries = [entry for entry in known if entry.module == result.module]
    unknown = [
        text
        for text in result.fails
        if not any(e.pattern.search(text) for e in entries)
    ]
    for text in unknown:
        result.problems.append(f"new failure: {text}")
    for entry in entries:
        hits = sum(1 for text in result.fails if entry.pattern.search(text))
        if hits != entry.count:
            result.problems.append(
                f"known failure (line {entry.line}) matched {hits} [FAIL] lines, "
                f"expected {entry.count}: {entry.pattern.pattern}"
            )
    if result.failed != len(result.fails):
        # A worker that died or timed out takes its jobs with it silently.
        result.problems.append(
            f"{result.failed} replays failed but {len(result.fails)} [FAIL] lines "
            "were printed (a worker died or was killed?)"
        )
    if result.returncode != 0 and not result.fails:
        result.problems.append(f"exited {result.returncode} without a [FAIL] line")
    if result.returncode == 0 and result.fails:
        result.problems.append("printed [FAIL] lines but exited 0")
    if result.problems:
        result.status = "failed"
    elif result.fails:
        result.status = "known"
    else:
        result.status = "ok"


def run_pass(module: str, phase: str, env: dict[str, str], log_dir: Path) -> PassResult:
    log_path = log_dir / f"{module}.{phase}.log"
    command = [sys.executable, "-m", f"aiter.aot.flydsl.{module}"]
    if phase == "check":
        command.append("--check")
    print(f"::group::{module} ({phase})", flush=True)
    print("$ " + " ".join(command), flush=True)
    started = time.monotonic()
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            if _ECHO.search(line):
                print(line, end="", flush=True)
        returncode = process.wait()
    seconds = time.monotonic() - started
    print(f"exit {returncode} after {seconds:.0f}s; full log: {log_path}", flush=True)
    print("::endgroup::", flush=True)
    result = PassResult(module, phase, returncode, seconds, str(log_path))
    parse_log(result, log_path.read_text(encoding="utf-8", errors="replace"))
    return result


# ---------------------------------------------------------------------------
# Cache audit
# ---------------------------------------------------------------------------


def artifacts(cache_dir: Path) -> list[Path]:
    return sorted(cache_dir.glob("*/*.pkl"))


def read_artifact(path: Path):
    import pickle

    with open(path, "rb") as stream:
        return pickle.load(stream)


def audit_cache(cache_dir: Path, arch: str) -> tuple[dict[str, Any], list[str]]:
    """Artifacts per launcher and per compile target; problems when the cache
    has none or holds an artifact for another arch."""
    paths = artifacts(cache_dir)
    chips: Counter = Counter()
    launchers: Counter = Counter()
    unreadable = []
    for path in paths:
        launchers[path.parent.name] += 1
        try:
            ir_text = getattr(read_artifact(path), "_ir_text", "") or ""
        except Exception as error:  # noqa: BLE001  report, do not stop the audit
            unreadable.append(f"{path.relative_to(cache_dir)}: {error}")
            continue
        for chip in set(_CHIP.findall(ir_text)) or {"<none>"}:
            chips[chip] += 1
    problems = [f"unreadable artifact {item}" for item in unreadable[:10]]
    others = {chip: n for chip, n in chips.items() if chip != arch}
    if others:
        problems.append(f"artifacts compiled for other targets than {arch}: {others}")
    audit = {
        "artifacts": len(paths),
        "unreadable": len(unreadable),
        "chips": dict(chips),
        "launchers": dict(launchers.most_common()),
    }
    return audit, problems


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def base_env(arch: str, cache_dir: Path) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("AITER_AOT_IMPORT", None)
    pythonpath = [str(REPO_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env.update(
        # Not PYTHONUNBUFFERED: a buffered, flushed print() is one write, so
        # the lines of parallel workers sharing the pipe stay whole.
        PYTHONPATH=os.pathsep.join(pythonpath),
        GPU_ARCHS=arch,
        FLYDSL_RUNTIME_CACHE_DIR=str(cache_dir),
        AITER_FLYDSL_AOT_TRACEBACK="1",
        # Compile only: no process of the build may reach a GPU, even on a
        # runner that has one. An empty list hides every GPU.
        HIP_VISIBLE_DEVICES="",
        ROCR_VISIBLE_DEVICES="",
        CUDA_VISIBLE_DEVICES="",
    )
    return env


def jit_modules() -> dict[str, float]:
    return {path.name: path.stat().st_mtime for path in JIT_DIR.glob("module_*.so")}


def prebuild_host_modules(host_arch: str, env: dict[str, str], log_dir: Path) -> int:
    """Build aiter's C++ core and ``HOST_CTYPES_MODULES`` for ``host_arch``;
    the replays call only their host code, which any arch's build has."""
    code = (
        "import aiter\n"
        "from aiter.jit.core import build_ctypes_module\n"
        f"for name in {HOST_CTYPES_MODULES!r}:\n"
        "    build_ctypes_module(name)\n"
    )
    log_path = log_dir / "host_modules.log"
    print(f"::group::host C++ modules for {host_arch}", flush=True)
    with open(log_path, "w", encoding="utf-8") as log:
        returncode = subprocess.run(
            [sys.executable, "-c", code],
            cwd=REPO_ROOT,
            env=dict(env, GPU_ARCHS=host_arch),
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    print(log_path.read_text(errors="replace")[-4000:], flush=True)
    print("::endgroup::", flush=True)
    return returncode


def build(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    log_dir = out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir.exists() and any(cache_dir.iterdir()) and not args.reuse_cache:
        # A cache hit recompiles nothing, so a stale artifact would pass
        # unseen; the build starts from an empty cache unless told otherwise.
        print(f"error: {cache_dir} is not empty (pass --reuse-cache to keep it)")
        return 2
    cache_dir.mkdir(parents=True, exist_ok=True)
    try:
        known = load_known_failures(
            Path(args.known_failures) if args.known_failures else None
        )
    except ValueError as error:
        print(f"error: {error}")
        return 2
    modules = args.modules or default_modules()
    unknown_modules = sorted({e.module for e in known} - set(default_modules()))
    if unknown_modules:
        print(f"error: known failures name unknown modules {unknown_modules}")
        return 2

    env = base_env(args.arch, cache_dir)
    problems: list[str] = []
    started = time.monotonic()
    if args.host_arch and prebuild_host_modules(args.host_arch, env, log_dir) != 0:
        problems.append(f"building the host C++ modules for {args.host_arch} failed")
    before = jit_modules()

    results: list[PassResult] = []
    phases = ["compile"] + ([] if args.no_check else ["check"])
    if not problems:
        for phase in phases:
            for module in modules:
                result = run_pass(module, phase, env, log_dir)
                judge(result, known)
                results.append(result)

    built = sorted(
        name for name, mtime in jit_modules().items() if before.get(name) != mtime
    )
    if built and args.host_arch:
        print(
            f"::warning::the replays built C++ module(s) {built} for {args.arch}; "
            f"add them to HOST_CTYPES_MODULES so they build for {args.host_arch}"
        )
    audit, audit_problems = audit_cache(cache_dir, args.arch)
    problems += audit_problems
    if audit["artifacts"] == 0 and any(r.jobs for r in results):
        problems.append("the modules had jobs but the cache holds no artifact")
    problems += [
        f"{r.module} ({r.phase}): {problem}" for r in results for problem in r.problems
    ]
    passed = not problems
    summary = {
        "arch": args.arch,
        "passed": passed,
        "seconds": round(time.monotonic() - started, 1),
        "cache_dir": str(cache_dir),
        "host_arch": args.host_arch,
        "jit_modules_built_by_replays": built,
        "cache": audit,
        "passes": [
            {
                k: getattr(r, k)
                for k in (
                    "module",
                    "phase",
                    "status",
                    "returncode",
                    "jobs",
                    "ok",
                    "failed",
                    "seconds",
                    "fails",
                    "problems",
                    "log",
                )
            }
            for r in results
        ],
        "problems": problems,
    }
    write_summary(out_dir, summary, build_markdown(summary))
    for problem in problems:
        print(f"::error::{problem}")
    print("FlyDSL AOT build " + ("passed" if passed else "FAILED"), flush=True)
    return 0 if passed else 1


def build_markdown(summary: dict[str, Any]) -> str:
    icon = {"ok": "✅", "known": "⚠️", "failed": "❌"}
    lines = [
        f"## FlyDSL AOT {summary['arch']}: "
        + ("passed" if summary["passed"] else "FAILED"),
        "",
        "| Module | Pass | Jobs | OK | Failed | Status | Time |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: |",
    ]
    for p in summary["passes"]:
        lines.append(
            f"| {p['module']} | {p['phase']} | {p['jobs']} | {p['ok']} | {p['failed']} "
            f"| {icon.get(p['status'], '')} {p['status']} | {p['seconds']:.0f}s |"
        )
    cache = summary["cache"]
    lines += [
        "",
        (
            f"Cache: {cache['artifacts']} artifacts, targets {cache['chips']}, "
            f"{len(cache['launchers'])} launchers. Total {summary['seconds']:.0f}s."
        ),
    ]
    known = [
        (p, text)
        for p in summary["passes"]
        if p["status"] == "known"
        for text in p["fails"]
    ]
    if known:
        lines += ["", f"<details><summary>{len(known)} known failures</summary>", ""]
        lines += [
            f"- `{p['module']}` ({p['phase']}): `{text}`" for p, text in known[:200]
        ]
        lines += ["", "</details>"]
    if summary["problems"]:
        lines += ["", "### Problems", ""]
        lines += [f"- {problem}" for problem in summary["problems"][:200]]
    return "\n".join(lines) + "\n"


def write_summary(out_dir: Path, summary: dict[str, Any], markdown: str) -> None:
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "summary.md").write_text(markdown)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        with open(step_summary, "a", encoding="utf-8") as stream:
            stream.write(markdown)


# ---------------------------------------------------------------------------
# load (needs a GPU of the arch)
# ---------------------------------------------------------------------------


def code_objects(ir_text: str) -> list[tuple[bytes, list[str]]]:
    """The code object the JIT loads of each ``gpu.binary`` (object 0) and
    the kernels launched from it."""
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import gpu
    from flydsl.compiler.jit_function import _create_mlir_context

    binaries: dict[str, bytes] = {}
    kernels: dict[str, set[str]] = {}

    def visit(op):
        if op.name == "gpu.binary":
            objects = ir.ArrayAttr(op.attributes["objects"])
            name = ir.StringAttr(op.attributes["sym_name"]).value
            binaries[name] = bytes(gpu.ObjectAttr(objects[0]).object)
        elif op.name == "gpu.launch_func":
            module, kernel = ir.SymbolRefAttr(op.opview.kernel).value
            kernels.setdefault(module, set()).add(kernel)
        return ir.WalkResult.ADVANCE

    with _create_mlir_context():
        ir.Module.parse(ir_text).operation.walk(visit)
    return [(blob, sorted(kernels.get(name, ()))) for name, blob in binaries.items()]


def hip_library():
    """The HIP runtime torch loaded, so both share one runtime."""
    import ctypes

    import torch

    torch.cuda.init()
    with open("/proc/self/maps") as maps:
        paths = {line.split()[-1] for line in maps if "libamdhip64" in line}
    if not paths:
        raise RuntimeError("torch did not load libamdhip64")
    hip = ctypes.CDLL(min(paths))
    hip.hipGetErrorName.restype = ctypes.c_char_p
    return hip, torch.cuda.get_device_properties(0).gcnArchName


def load(args: argparse.Namespace) -> int:
    import ctypes

    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    hip, device_arch = hip_library()
    print(f"device 0: {device_arch}; cache: {cache_dir}", flush=True)

    def check(code: int, what: str) -> str | None:
        return None if code == 0 else f"{what}: {hip.hipGetErrorName(code).decode()}"

    paths = artifacts(cache_dir)
    problems: list[str] = []
    loaded = kernels = 0
    started = time.monotonic()
    for path in paths:
        label = str(path.relative_to(cache_dir))
        try:
            objects = code_objects(read_artifact(path)._ir_text)
        except Exception as error:  # noqa: BLE001  report, keep loading the rest
            problems.append(f"{label}: unreadable: {type(error).__name__}: {error}")
            continue
        if not objects:
            problems.append(f"{label}: no gpu.binary")
        for blob, names in objects:
            module = ctypes.c_void_p()
            image = ctypes.create_string_buffer(blob, len(blob))
            error = check(
                hip.hipModuleLoadData(ctypes.byref(module), image), "hipModuleLoadData"
            )
            if error:
                problems.append(f"{label}: {error}")
                continue
            loaded += 1
            for name in names:
                function = ctypes.c_void_p()
                error = check(
                    hip.hipModuleGetFunction(
                        ctypes.byref(function), module, name.encode()
                    ),
                    f"hipModuleGetFunction({name})",
                )
                if error:
                    problems.append(f"{label}: {error}")
                else:
                    kernels += 1
            hip.hipModuleUnload(module)
    if not paths:
        problems.append(f"no artifact in {cache_dir}")
    if device_arch.split(":")[0] != args.arch.split("-")[0]:
        print(f"::warning::device 0 is {device_arch}, the cache targets {args.arch}")
    passed = not problems
    summary = {
        "arch": args.arch,
        "device_arch": device_arch,
        "passed": passed,
        "artifacts": len(paths),
        "code_objects_loaded": loaded,
        "kernels_found": kernels,
        "seconds": round(time.monotonic() - started, 1),
        "problems": problems,
    }
    markdown = (
        f"## FlyDSL AOT {args.arch} load on {device_arch}: "
        + ("passed" if passed else "FAILED")
        + f"\n\n{len(paths)} artifacts, {loaded} code objects loaded, "
        f"{kernels} kernels found.\n"
        + "".join(f"\n- {problem}" for problem in problems[:200])
        + "\n"
    )
    write_summary(out_dir, summary, markdown)
    for problem in problems[:200]:
        print(f"::error::{problem}")
    print(markdown, flush=True)
    return 0 if passed else 1


# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="compile and --check every AOT module")
    p_build.add_argument("--arch", default="gfx1250", help="GPU_ARCHS of the build")
    p_build.add_argument("--cache-dir", required=True, help="FlyDSL cache to build")
    p_build.add_argument("--out-dir", required=True, help="logs and summary")
    p_build.add_argument("--known-failures", help="'<module> <count> <regex>' lines")
    p_build.add_argument(
        "--modules", nargs="+", help="module names (default: those run_aot replays)"
    )
    p_build.add_argument(
        "--host-arch",
        default="gfx950",
        help="arch to build the host C++ modules for first ('' builds them "
        "lazily, for --arch)",
    )
    p_build.add_argument("--no-check", action="store_true", help="skip --check")
    p_build.add_argument(
        "--reuse-cache", action="store_true", help="allow a non-empty --cache-dir"
    )
    p_build.set_defaults(func=build)

    p_load = sub.add_parser("load", help="load every cached code object on GPU 0")
    p_load.add_argument("--arch", default="gfx1250", help="arch the cache targets")
    p_load.add_argument("--cache-dir", required=True)
    p_load.add_argument("--out-dir", required=True)
    p_load.set_defaults(func=load)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
