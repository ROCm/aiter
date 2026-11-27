#!/usr/bin/env python3
"""Run a pytest selector manifest under a strict, fail-closed CI contract.

Architecture-neutral, so any curated AITER suite can reuse it. Each manifest
line is a pytest selector (a test file, ``file::test``, ``file::Class::test``,
or one exact parametrized node ID), optionally followed by the exact number of
nodes it must collect::

    op_tests/test_moe_sorting.py::test_moe_sorting_invalid_topk_ids 1
    op_tests/triton_tests/moe/moe_routing/test_moe_routing.py::test_topk_in_range 12

The contract:

- every selected file exists and every selector collects its declared count
  (or at least one node when no count is given); no node is collected twice,
  outside the manifest, deselected, or lost to a collection error;
- every collected node runs exactly once and passes setup, call, and teardown,
  with no skip, xfail, or xpass;
- with ``--expect-failure`` (a manifest of known upstream bugs), every node
  instead fails in setup or call, so a node that starts to pass, skips, or
  crashes turns the run red for review.

Each test file runs in its own pytest process, so a crash, hang, or poisoned
GPU context in one file cannot hide the results of the others, and the runner
process itself never initializes the GPU. Per-file evidence (manifest,
command, log, JUnit, phase events) and ``contract-summary.json`` are written
under ``--artifact-dir``.

Exit status: 0 when the contract holds, 1 when it does not, 2 for an invalid
manifest or a failed ``--require-gfx`` check.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

EVENTS_ENV = "PYTEST_MANIFEST_EVENTS"
PLUGIN = Path(__file__).stem
PHASES = ("setup", "call", "teardown")


# pytest plugin hooks: active only in the pytest processes this runner starts.
# They record every collection and phase result to an append-only JSON-lines
# file, flushed per line, so the evidence survives a crash mid-run.


def _emit(event: str, **fields: Any) -> None:
    path = os.environ.get(EVENTS_ENV)
    if not path:
        return
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def pytest_collection_finish(session: Any) -> None:
    _emit("collection", nodes=[item.nodeid for item in session.items])


def pytest_collectreport(report: Any) -> None:
    if not report.passed:
        _emit("collect_error", node=report.nodeid, outcome=report.outcome)


def pytest_deselected(items: Any) -> None:
    _emit("deselected", nodes=[item.nodeid for item in items])


def pytest_runtest_logreport(report: Any) -> None:
    _emit(
        "report",
        node=report.nodeid,
        phase=report.when,
        outcome=report.outcome,
        wasxfail=hasattr(report, "wasxfail"),
    )


def pytest_sessionfinish(session: Any, exitstatus: int) -> None:
    _emit("finish", exitstatus=int(exitstatus))


# Manifest handling.


class ManifestError(ValueError):
    pass


def read_manifest(path: Path) -> dict[str, int | None]:
    """Map each selector to its declared node count (None: at least one)."""
    if not path.is_file():
        raise ManifestError(f"test manifest not found: {path}")
    expectations: dict[str, int | None] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) > 2 or (len(fields) == 2 and not fields[1].isdigit()):
            raise ManifestError(f"invalid manifest line: {line!r}")
        selector = fields[0]
        count = int(fields[1]) if len(fields) == 2 else None
        name, separator, test = selector.partition("::")
        file = PurePosixPath(name)
        if (
            selector in expectations
            or count == 0
            or file.is_absolute()
            or ".." in file.parts
            or file.suffix != ".py"
            or (separator and not test)
        ):
            raise ManifestError(f"invalid or duplicate selector: {line!r}")
        if "[" in selector:
            if not separator or not selector.endswith("]") or count not in (None, 1):
                raise ManifestError(
                    f"a parametrized node ID must select one node: {line!r}"
                )
            count = 1
        expectations[selector] = count
    if not expectations:
        raise ManifestError(f"no test selectors found in {path}")
    files = {s.partition("::")[0] for s in expectations}
    missing = sorted(f for f in files if not Path(f).is_file())
    if missing:
        raise ManifestError(f"selected test files do not exist: {', '.join(missing)}")
    return expectations


def group_by_file(expectations: dict[str, int | None]) -> list[dict[str, int | None]]:
    """Split a manifest into per-file groups, preserving manifest order."""
    groups: dict[str, dict[str, int | None]] = {}
    for selector, count in expectations.items():
        groups.setdefault(selector.partition("::")[0], {})[selector] = count
    return list(groups.values())


def matching_selectors(node: str, selectors: list[str]) -> list[str]:
    return [s for s in selectors if node == s or node.startswith((s + "[", s + "::"))]


# Auditing one pytest process.


def load_events(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            break  # A torn final line from a killed process.
    return events


def failed_as_filed(phases: list[dict[str, Any]]) -> bool:
    """A known failure: setup or call failed, nothing skipped or xfailed."""
    outcomes = {(p["phase"], p["outcome"]) for p in phases}
    return (
        len(phases) == len({p["phase"] for p in phases})
        and not any(p["wasxfail"] or p["outcome"] == "skipped" for p in phases)
        and bool(outcomes & {("setup", "failed"), ("call", "failed")})
    )


def audit_group(
    expectations: dict[str, int | None],
    events: list[dict[str, Any]],
    returncode: int,
    expect_failure: bool,
    collect_only: bool,
) -> dict[str, Any]:
    problems: list[str] = []
    by_kind: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        by_kind.setdefault(event["event"], []).append(event)
    expected_rc = 1 if expect_failure and not collect_only else 0
    if returncode != expected_rc:
        problems.append(f"pytest exited {returncode}, expected {expected_rc}")
    for kind in ("collection", "finish"):
        if len(by_kind.get(kind, [])) != 1:
            problems.append(
                f"expected one {kind} event, found {len(by_kind.get(kind, []))}"
            )
    if by_kind.get("collect_error"):
        problems.append(
            "collection errors: "
            + ", ".join(e["node"] or "<session>" for e in by_kind["collect_error"])
        )
    if by_kind.get("deselected"):
        problems.append("tests were deselected")

    nodes = by_kind["collection"][0]["nodes"] if by_kind.get("collection") else []
    if len(nodes) != len(set(nodes)):
        problems.append("duplicate collected node IDs")
    selectors = list(expectations)
    counts = {s: 0 for s in selectors}
    for node in nodes:
        matches = matching_selectors(node, selectors)
        if len(matches) != 1:
            reason = (
                "outside the manifest" if not matches else "matches several selectors"
            )
            problems.append(f"collected node {reason}: {node}")
            continue
        counts[matches[0]] += 1
    for selector, expected in expectations.items():
        if counts[selector] == 0:
            problems.append(f"collected no nodes: {selector}")
        elif expected is not None and counts[selector] != expected:
            problems.append(
                f"expected {expected} nodes, collected {counts[selector]}: {selector}"
            )

    outcomes: dict[str, str] = {}
    if not collect_only:
        reports: dict[str, list[dict[str, Any]]] = {}
        for event in by_kind.get("report", []):
            reports.setdefault(event["node"], []).append(event)
        for node in sorted(set(reports) - set(nodes)):
            problems.append(f"reported but not collected: {node}")
        for node in nodes:
            phases = reports.get(node, [])
            passed = [(p["phase"], p["outcome"], p["wasxfail"]) for p in phases] == [
                (phase, "passed", False) for phase in PHASES
            ]
            if expect_failure:
                if failed_as_filed(phases):
                    outcomes[node] = "failed as filed"
                elif passed:
                    outcomes[node] = "passed"
                    problems.append(f"known failure now passes; review the fix: {node}")
                else:
                    outcomes[node] = "changed"
                    problems.append(
                        f"known failure changed (skip, xfail, or incomplete): {node}"
                    )
            elif passed:
                outcomes[node] = "passed"
            else:
                outcomes[node] = (
                    "/".join(
                        f"{p['phase']}:{'xfail' if p['wasxfail'] else p['outcome']}"
                        for p in phases
                    )
                    or "not run"
                )
                problems.append(f"did not pass cleanly ({outcomes[node]}): {node}")
    return {
        "collected": len(nodes),
        "nodes": nodes,
        "counts": counts,
        "outcomes": outcomes,
        "problems": problems,
    }


# Running.


def child_environment(events: Path | None = None) -> dict[str, str]:
    environment = dict(os.environ)
    script_dir = str(Path(__file__).resolve().parent)
    environment["PYTHONPATH"] = os.pathsep.join(
        p for p in (script_dir, environment.get("PYTHONPATH")) if p
    )
    environment.pop(EVENTS_ENV, None)
    if events is not None:
        environment[EVENTS_ENV] = str(events)
    return environment


def run_process(
    command: list[str], log: Path, environment: dict[str, str], timeout: float
) -> int:
    """Run in a new session; on timeout, kill the whole process group (rc 124)."""
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            command,
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=max(1.0, timeout))
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            stream.write(
                f"\n[run_pytest_manifest] killed after {timeout:.0f} s budget\n"
            )
            return 124


ENVIRONMENT_PROBE = r"""
import importlib.metadata, json, os, platform, subprocess
record = {"python": platform.python_version(), "platform": platform.platform()}
def run(*args):
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"
record["git_head"] = run("git", "rev-parse", "HEAD")
record["git_submodules"] = run("git", "submodule", "status", "--recursive")
for key in ("CONTAINER_IMAGE", "ROCM_PATH", "GPU_ARCHS"):
    record[key.lower()] = os.environ.get(key)
packages = {}
for name in ("torch", "triton", "triton-kernels", "flydsl", "amd-aiter", "pytest"):
    try:
        packages[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        packages[name] = None
record["packages"] = packages
try:
    import torch
    record.update(torch=torch.__version__, torch_hip=torch.version.hip,
                  device_count=torch.cuda.device_count())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        record.update(device_name=props.name, gcn_arch=props.gcnArchName)
except Exception as exc:
    record["torch_error"] = repr(exc)
try:
    import triton
    record.update(triton=triton.__version__, triton_module=triton.__file__)
except Exception as exc:
    record["triton_error"] = repr(exc)
try:
    import aiter
    from aiter.jit.utils.chip_info import get_gfx
    record.update(aiter_module=aiter.__file__, gfx=get_gfx())
except Exception as exc:
    record["gfx_error"] = repr(exc)
print(json.dumps(record, sort_keys=True))
"""


def environment_record(artifact_dir: Path) -> dict[str, Any]:
    """Probe the environment in a child process, keeping this one GPU-free.

    The first ``import aiter`` in a fresh checkout JIT-builds its core module,
    so the probe gets a generous budget of its own.
    """
    log = artifact_dir / "environment-probe.log"
    rc = run_process(
        [sys.executable, "-c", ENVIRONMENT_PROBE], log, child_environment(), 1200
    )
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    try:
        record = json.loads(lines[-1])
    except (IndexError, json.JSONDecodeError):
        record = {"probe_error": f"exit {rc}; see environment-probe.log"}
    (artifact_dir / "environment.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def run_group(
    directory: Path,
    group: dict[str, int | None],
    args: argparse.Namespace,
    deadline: float,
) -> tuple[dict[str, Any], int, float]:
    directory.mkdir()
    (directory / "manifest.txt").write_text(
        "".join(f"{s}\n" if c is None else f"{s} {c}\n" for s, c in group.items()),
        encoding="utf-8",
    )
    events = directory / "events.jsonl"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-p",
        PLUGIN,
        "-p",
        "no:cacheprovider",
        "-o",
        "addopts=",
        "-o",
        "xfail_strict=true",
        "-v",
        "-ra",
        f"--junitxml={directory / 'junit.xml'}",
    ]
    if args.collect_only:
        command.append("--collect-only")
    command.extend(group)
    (directory / "command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    started = time.monotonic()
    rc = run_process(
        command,
        directory / "pytest.log",
        child_environment(events),
        deadline - time.monotonic(),
    )
    seconds = round(time.monotonic() - started, 1)
    result = audit_group(
        group, load_events(events), rc, args.expect_failure, args.collect_only
    )
    (directory / "result.json").write_text(
        json.dumps({"returncode": rc, "seconds": seconds, **result}, indent=2) + "\n",
        encoding="utf-8",
    )
    return result, rc, seconds


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--expected-count",
        type=int,
        help="total node count; must equal the sum of the manifest's counts",
    )
    parser.add_argument(
        "--require-gfx", help="fail unless aiter reports this gfx target"
    )
    parser.add_argument(
        "--expect-failure",
        action="store_true",
        help="known-failure manifest: every node must fail in setup or call",
    )
    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="check collection counts without running tests",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800,
        help="seconds for all pytest processes together (default 1800)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        expectations = read_manifest(args.manifest)
    except ManifestError as error:
        print(error, file=sys.stderr)
        return 2
    declared = list(expectations.values())
    if args.expected_count is not None and (
        None in declared or sum(declared) != args.expected_count
    ):
        print(
            f"--expected-count {args.expected_count} does not match the manifest's "
            "per-selector counts (every selector needs a count)",
            file=sys.stderr,
        )
        return 2
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    groups_dir = args.artifact_dir / "groups"
    if groups_dir.exists():
        print(f"artifact directory already holds a run: {groups_dir}", file=sys.stderr)
        return 2

    record = environment_record(args.artifact_dir)
    if args.require_gfx and record.get("gfx") != args.require_gfx:
        print(
            f"required architecture {args.require_gfx}, detected "
            f"{record.get('gfx', 'unknown')}: {record.get('gfx_error', '')}",
            file=sys.stderr,
        )
        return 2

    groups_dir.mkdir()
    deadline = time.monotonic() + args.timeout
    summary_groups, problems, nodes = [], [], []
    counts: dict[str, int] = {}
    outcomes: dict[str, str] = {}
    for index, group in enumerate(group_by_file(expectations)):
        file = next(iter(group)).partition("::")[0]
        name = f"{index:02d}-{Path(file).stem}"
        print(f"[run_pytest_manifest] {name}: {len(group)} selectors", flush=True)
        result, rc, seconds = run_group(groups_dir / name, group, args, deadline)
        nodes.extend(result["nodes"])
        counts.update(result["counts"])
        outcomes.update(result["outcomes"])
        problems.extend(f"{name}: {p}" for p in result["problems"])
        summary_groups.append(
            {
                "group": name,
                "returncode": rc,
                "seconds": seconds,
                "collected": result["collected"],
                "problems": len(result["problems"]),
            }
        )
        print(
            f"[run_pytest_manifest] {name}: rc={rc} collected={result['collected']} "
            f"problems={len(result['problems'])} {seconds}s",
            flush=True,
        )

    (args.artifact_dir / "collected-nodes.txt").write_text(
        "".join(f"{node}\n" for node in nodes), encoding="utf-8"
    )
    expected_total = None if None in declared else sum(declared)
    if expected_total is not None and len(nodes) != expected_total:
        problems.append(
            f"expected {expected_total} nodes in total, collected {len(nodes)}"
        )
    summary = {
        "manifest": str(args.manifest),
        "mode": (
            "collect-only"
            if args.collect_only
            else ("expect-failure" if args.expect_failure else "pass")
        ),
        "expected": expected_total,
        "collected": len(nodes),
        "outcomes": dict(sorted(Counter(outcomes.values()).items())),
        "per_selector": counts,
        "groups": summary_groups,
        "problems": problems,
        "strict_contract_passed": not problems,
    }
    (args.artifact_dir / "contract-summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    for problem in problems:
        print(f"[run_pytest_manifest] FAIL {problem}", flush=True)
    verdict = "holds" if not problems else "FAILED"
    print(
        f"[run_pytest_manifest] contract {verdict}: {len(nodes)} collected, "
        f"{summary['outcomes']}",
        flush=True,
    )
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
