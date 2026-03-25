#!/usr/bin/env python3
"""Auto-update FILE_TIMES in split_tests.sh from recent workflow artifacts."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple


ALLOWED_CONCLUSIONS = {"success", "failure"}
LOG_TIME_RE = re.compile(r"Time elapsed of\s+([^:]+):\s*([0-9]+(?:\.[0-9]+)?)")
ASSIGN_RE = re.compile(r"^\s*FILE_TIMES\[(?P<key>[^\]]+)\]=(?P<value>\d+)\s*$")
DEFAULT_TIME = 15


@dataclass(frozen=True)
class UpdateStats:
    added: int
    updated: int
    unchanged: int
    defaulted: int
    removed: int
    discovered: int
    sampled: int


class ComputeResult(NamedTuple):
    merged: dict[str, int]
    stats: UpdateStats
    defaulted_files: list[str]


def run_cmd(cmd: list[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout


def gh_json(cmd: list[str], cwd: Path | None = None) -> object:
    output = run_cmd(cmd, cwd=cwd)
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON from command: {' '.join(cmd)}") from exc


def fetch_recent_run_ids(repo: str, workflow_file: str, runs_count: int) -> list[int]:
    runs: list[int] = []
    page = 1
    per_page = 100
    while len(runs) < runs_count:
        endpoint = (
            f"/repos/{repo}/actions/workflows/{workflow_file}/runs"
            f"?branch=main&status=completed&per_page={per_page}&page={page}"
        )
        payload = gh_json(["gh", "api", endpoint])
        workflow_runs = payload.get("workflow_runs", []) if isinstance(payload, dict) else []
        if not workflow_runs:
            break
        for run in workflow_runs:
            conclusion = run.get("conclusion")
            run_id = run.get("id")
            if conclusion in ALLOWED_CONCLUSIONS and isinstance(run_id, int):
                runs.append(run_id)
                if len(runs) >= runs_count:
                    break
        page += 1
    return runs


def list_artifacts(repo: str, run_id: int) -> list[dict]:
    endpoint = f"/repos/{repo}/actions/runs/{run_id}/artifacts?per_page=100"
    payload = gh_json(["gh", "api", endpoint])
    if not isinstance(payload, dict):
        return []
    artifacts = payload.get("artifacts", [])
    if not isinstance(artifacts, list):
        return []
    return [a for a in artifacts if isinstance(a, dict)]


def download_artifact(repo: str, run_id: int, artifact_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_cmd(
        [
            "gh",
            "run",
            "download",
            str(run_id),
            "--repo",
            repo,
            "--name",
            artifact_name,
            "--dir",
            str(out_dir),
        ]
    )


def parse_aiter_log(path: Path) -> dict[str, int]:
    file_time: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = LOG_TIME_RE.search(line)
        if not match:
            continue
        file_path = match.group(1).strip()
        seconds = int(round(float(match.group(2))))
        file_time[file_path] = max(1, seconds)
    return file_time


def list_test_files(repo_root: Path, test_type: str) -> list[str]:
    if test_type == "aiter":
        glob_pattern = "op_tests/test_*.py"
    else:
        glob_pattern = "op_tests/triton_tests/**/test_*.py"
    files = sorted(str(p.relative_to(repo_root)) for p in repo_root.glob(glob_pattern) if p.is_file())
    return files


def guess_test_file_from_testcase(
    testcase: ET.Element,
    known_files: set[str],
    known_by_stem: dict[str, str],
) -> str | None:
    file_attr = testcase.attrib.get("file", "").strip()
    if file_attr:
        normalized = file_attr.replace("\\", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if normalized in known_files:
            return normalized
        idx = normalized.find("op_tests/")
        if idx >= 0:
            candidate = normalized[idx:]
            if candidate in known_files:
                return candidate

    classname = testcase.attrib.get("classname", "").strip()
    if classname:
        candidate = f"{classname.replace('.', '/')}.py"
        if candidate in known_files:
            return candidate
        if "/" in candidate:
            stem = Path(candidate).stem
            if stem in known_by_stem:
                return known_by_stem[stem]

    name = testcase.attrib.get("name", "").strip()
    stem_match = re.search(r"(test_[A-Za-z0-9_]+)", name)
    if stem_match:
        stem = stem_match.group(1)
        if stem in known_by_stem:
            return known_by_stem[stem]
    return None


def parse_triton_junit(path: Path, known_files: set[str]) -> dict[str, int]:
    known_by_stem = {Path(item).stem: item for item in known_files}
    tree = ET.parse(path)
    root = tree.getroot()
    totals: dict[str, float] = defaultdict(float)
    for testcase in root.iter("testcase"):
        test_file = guess_test_file_from_testcase(testcase, known_files, known_by_stem)
        if not test_file:
            continue
        try:
            test_time = float(testcase.attrib.get("time", "0"))
        except ValueError:
            test_time = 0.0
        totals[test_file] += max(0.0, test_time)
    return {k: max(1, int(round(v))) for k, v in totals.items()}


def aggregate_values(values: list[int], mode: str) -> int:
    if not values:
        raise ValueError("Cannot aggregate empty values")
    if mode == "mean":
        return max(1, int(round(statistics.fmean(values))))
    return max(1, int(round(statistics.median(values))))


def parse_existing_file_times(script_path: Path, test_type: str) -> dict[str, int]:
    content = script_path.read_text(encoding="utf-8")
    section = extract_file_times_section(content)
    lines = section.splitlines()
    state = "search"
    collected: dict[str, int] = {}
    marker = 'if [[ "$TEST_TYPE" == "aiter" ]]; then' if test_type == "aiter" else 'elif [[ "$TEST_TYPE" == "triton" ]]; then'
    end_marker = 'elif [[ "$TEST_TYPE" == "triton" ]]; then' if test_type == "aiter" else "fi"
    for line in lines:
        if state == "search":
            if line.strip() == marker:
                state = "collect"
            continue
        if state == "collect":
            if line.strip() == end_marker:
                break
            match = ASSIGN_RE.match(line)
            if match:
                collected[match.group("key")] = int(match.group("value"))
    return collected


def compute_new_times(
    discovered_files: list[str],
    samples: dict[str, list[int]],
    existing_times: dict[str, int],
    default_time: int,
    aggregate_mode: str,
) -> ComputeResult:
    merged: dict[str, int] = {}
    defaulted_files: list[str] = []
    added = 0
    updated = 0
    unchanged = 0
    defaulted = 0

    for file_path in discovered_files:
        values = samples.get(file_path, [])
        if values:
            new_value = aggregate_values(values, aggregate_mode)
        else:
            new_value = default_time
            defaulted += 1
            defaulted_files.append(file_path)
        merged[file_path] = new_value
        old_value = existing_times.get(file_path)
        if old_value is None:
            added += 1
        elif old_value == new_value:
            unchanged += 1
        else:
            updated += 1

    removed = len([k for k in existing_times if k not in merged])
    sampled = len([k for k, v in samples.items() if v])
    stats = UpdateStats(
        added=added,
        updated=updated,
        unchanged=unchanged,
        defaulted=defaulted,
        removed=removed,
        discovered=len(discovered_files),
        sampled=sampled,
    )
    return ComputeResult(merged=merged, stats=stats, defaulted_files=defaulted_files)


def build_file_times_block(aiter_times: dict[str, int], triton_times: dict[str, int]) -> str:
    def sorted_items(d: dict[str, int]) -> list[tuple[str, int]]:
        return sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))

    lines: list[str] = []
    lines.append('if [[ "$TEST_TYPE" == "aiter" ]]; then')
    lines.append('    echo "Aiter test files:"')
    for path, sec in sorted_items(aiter_times):
        lines.append(f"    FILE_TIMES[{path}]={sec}")
    lines.append('elif [[ "$TEST_TYPE" == "triton" ]]; then')
    lines.append('    echo "Triton test files:"')
    for path, sec in sorted_items(triton_times):
        lines.append(f"    FILE_TIMES[{path}]={sec}")
    lines.append("fi")
    return "\n".join(lines)


def rewrite_split_tests(script_path: Path, new_block: str) -> bool:
    content = script_path.read_text(encoding="utf-8")
    start_marker = "declare -A FILE_TIMES"
    end_marker = "get_time() {"
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
        raise RuntimeError("Failed to locate FILE_TIMES section boundaries")
    section_start = content.find("\n", start_idx)
    if section_start < 0:
        raise RuntimeError("Malformed split_tests.sh around FILE_TIMES declaration")
    section_start += 1
    replacement = new_block.rstrip() + "\n\n"
    updated = content[:section_start] + replacement + content[end_idx:]
    changed = updated != content
    if changed:
        script_path.write_text(updated, encoding="utf-8")
    return changed


def extract_file_times_section(content: str) -> str:
    start_marker = "declare -A FILE_TIMES"
    end_marker = "get_time() {"
    start_idx = content.find(start_marker)
    end_idx = content.find(end_marker)
    if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
        raise RuntimeError("Failed to locate FILE_TIMES section boundaries")
    section_start = content.find("\n", start_idx)
    if section_start < 0:
        raise RuntimeError("Malformed split_tests.sh around FILE_TIMES declaration")
    section_start += 1
    return content[section_start:end_idx]


def format_summary(
    repo: str,
    runs_count: int,
    aggregate_mode: str,
    default_time: int,
    aiter_stats: UpdateStats,
    triton_stats: UpdateStats,
    aiter_runs: int,
    triton_runs: int,
    changed: bool,
    aiter_defaulted_files: list[str],
    triton_defaulted_files: list[str],
) -> str:
    status = "yes" if changed else "no"
    return (
        "## Split Tests FILE_TIMES Update\n"
        f"- repo: `{repo}`\n"
        f"- runs_count target: `{runs_count}`\n"
        f"- aggregate mode: `{aggregate_mode}`\n"
        f"- default time: `{default_time}s`\n"
        f"- file changed: `{status}`\n"
        "\n"
        "### Aiter\n"
        f"- runs used: `{aiter_runs}`\n"
        f"- discovered files: `{aiter_stats.discovered}`\n"
        f"- with samples: `{aiter_stats.sampled}`\n"
        f"- added: `{aiter_stats.added}`\n"
        f"- updated: `{aiter_stats.updated}`\n"
        f"- unchanged: `{aiter_stats.unchanged}`\n"
        f"- defaulted (no history): `{aiter_stats.defaulted}`\n"
        f"- removed stale entries: `{aiter_stats.removed}`\n"
        f"- defaulted files list: `{', '.join(aiter_defaulted_files) if aiter_defaulted_files else 'none'}`\n"
        "\n"
        "### Triton\n"
        f"- runs used: `{triton_runs}`\n"
        f"- discovered files: `{triton_stats.discovered}`\n"
        f"- with samples: `{triton_stats.sampled}`\n"
        f"- added: `{triton_stats.added}`\n"
        f"- updated: `{triton_stats.updated}`\n"
        f"- unchanged: `{triton_stats.unchanged}`\n"
        f"- defaulted (no history): `{triton_stats.defaulted}`\n"
        f"- removed stale entries: `{triton_stats.removed}`\n"
        f"- defaulted files list: `{', '.join(triton_defaulted_files) if triton_defaulted_files else 'none'}`\n"
    )


def collect_aiter_samples(repo: str, run_ids: list[int], scratch: Path) -> dict[str, list[int]]:
    samples: dict[str, list[int]] = defaultdict(list)
    found_artifact = False
    parsed_entries = 0
    for run_id in run_ids:
        artifacts = list_artifacts(repo, run_id)
        names = [a.get("name", "") for a in artifacts if isinstance(a.get("name"), str)]
        target_names = [n for n in names if n.startswith("standard-test-log-") and "-shard-" in n]
        if target_names:
            found_artifact = True
        for name in target_names:
            out_dir = scratch / "aiter" / str(run_id) / name
            try:
                download_artifact(repo, run_id, name, out_dir)
            except RuntimeError:
                continue
            for log in out_dir.rglob("latest_test.log"):
                parsed = parse_aiter_log(log)
                for file_path, sec in parsed.items():
                    samples[file_path].append(sec)
                    parsed_entries += 1
    if not found_artifact:
        raise RuntimeError("Missing aiter standard-test-log artifacts in sampled runs")
    if parsed_entries == 0:
        raise RuntimeError("No parseable timing entries found in aiter artifacts")
    return samples


def collect_triton_samples(
    repo: str,
    run_ids: list[int],
    scratch: Path,
    known_triton_files: set[str],
) -> dict[str, list[int]]:
    samples: dict[str, list[int]] = defaultdict(list)
    found_artifact = False
    parsed_entries = 0
    for run_id in run_ids:
        artifacts = list_artifacts(repo, run_id)
        names = [a.get("name", "") for a in artifacts if isinstance(a.get("name"), str)]
        target_names = [n for n in names if n.startswith("triton-test-shard-")]
        if target_names:
            found_artifact = True
        run_totals: dict[str, int] = defaultdict(int)
        for name in target_names:
            out_dir = scratch / "triton" / str(run_id) / name
            try:
                download_artifact(repo, run_id, name, out_dir)
            except RuntimeError:
                continue
            for xml_file in out_dir.rglob("*.xml"):
                parsed = parse_triton_junit(xml_file, known_triton_files)
                for file_path, sec in parsed.items():
                    run_totals[file_path] += sec
                    parsed_entries += 1
        for file_path, sec in run_totals.items():
            samples[file_path].append(sec)
    if not found_artifact:
        raise RuntimeError("Missing triton-test-shard artifacts in sampled runs")
    if parsed_entries == 0:
        raise RuntimeError("No parseable timing entries found in triton artifacts")
    return samples


def write_output_changed(changed: bool) -> None:
    output_file = os.getenv("GITHUB_OUTPUT")
    if not output_file:
        return
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"changed={'true' if changed else 'false'}\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Update split_tests.sh FILE_TIMES using recent CI runs")
    parser.add_argument("--repo", default=os.getenv("GITHUB_REPOSITORY", "ROCm/aiter"))
    parser.add_argument("--runs-count", type=int, default=10)
    parser.add_argument("--aggregate", choices=["median", "mean"], default="median")
    parser.add_argument("--default-time", type=int, default=DEFAULT_TIME)
    parser.add_argument("--script-path", default=".github/scripts/split_tests.sh")
    parser.add_argument("--summary-path", default="split-tests-update-summary.md")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.runs_count <= 0:
        raise ValueError("--runs-count must be positive")
    if args.default_time <= 0:
        raise ValueError("--default-time must be positive")

    repo_root = Path.cwd()
    script_path = repo_root / args.script_path
    if not script_path.exists():
        raise FileNotFoundError(f"split_tests.sh not found at: {script_path}")

    aiter_files = list_test_files(repo_root, "aiter")
    triton_files = list_test_files(repo_root, "triton")
    aiter_existing = parse_existing_file_times(script_path, "aiter")
    triton_existing = parse_existing_file_times(script_path, "triton")

    aiter_run_ids = fetch_recent_run_ids(args.repo, "aiter-test.yaml", args.runs_count)
    triton_run_ids = fetch_recent_run_ids(args.repo, "triton-test.yaml", args.runs_count)
    if not aiter_run_ids:
        raise RuntimeError("No eligible completed (success/failure) runs found for aiter-test.yaml")
    if not triton_run_ids:
        raise RuntimeError("No eligible completed (success/failure) runs found for triton-test.yaml")

    with tempfile.TemporaryDirectory(prefix="split-test-times-") as temp_dir:
        scratch = Path(temp_dir)
        aiter_samples = collect_aiter_samples(args.repo, aiter_run_ids, scratch)
        triton_samples = collect_triton_samples(args.repo, triton_run_ids, scratch, set(triton_files))

    aiter_result = compute_new_times(
        discovered_files=aiter_files,
        samples=aiter_samples,
        existing_times=aiter_existing,
        default_time=args.default_time,
        aggregate_mode=args.aggregate,
    )
    triton_result = compute_new_times(
        discovered_files=triton_files,
        samples=triton_samples,
        existing_times=triton_existing,
        default_time=args.default_time,
        aggregate_mode=args.aggregate,
    )

    changed = False
    if not args.dry_run:
        new_block = build_file_times_block(aiter_result.merged, triton_result.merged)
        changed = rewrite_split_tests(script_path, new_block)

    summary = format_summary(
        repo=args.repo,
        runs_count=args.runs_count,
        aggregate_mode=args.aggregate,
        default_time=args.default_time,
        aiter_stats=aiter_result.stats,
        triton_stats=triton_result.stats,
        aiter_runs=len(aiter_run_ids),
        triton_runs=len(triton_run_ids),
        changed=changed,
        aiter_defaulted_files=aiter_result.defaulted_files,
        triton_defaulted_files=triton_result.defaulted_files,
    )
    summary_path = repo_root / args.summary_path
    summary_path.write_text(summary, encoding="utf-8")
    print(summary)
    write_output_changed(changed)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # pragma: no cover
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
