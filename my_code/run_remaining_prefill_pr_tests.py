#!/usr/bin/env python3
"""Run the remaining PR comparisons from the host with user-supplied availability.

Only this entry point needs to be started. Compute workers run serially in the
existing containers and use immutable complete-source manifests. This driver
never changes a source tree, a branch, a published revision, or a PR.
"""

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shlex
import socket
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / ".codex_tmp/gfx1250_prefill_pr_20261104"
LOG_ROOT = ROOT / "my_code/remaining_prefill_pr_test_logs"
PUBLISHED = "bdb170a199a231b3df6519c94e31cbab74988c3b"
MAIN = "d0fd39399b46a60d8b497ec622d531e743bd27ea"
REFERENCE = "ad9882df24b734748974afa161eb66ed76108fcf"
ARCHIVE = TASK / "extended_archive_e530_runtime_20261104_r1/result.json"
ARCHIVE_SHA = "d3b1bdb3efd27be5da8fcb6935f45f9c25cdb5c3ac1ed29c144384baaaedbc1b"
PREVIOUS_PREFILL = (
    TASK
    / "published_bdb170a1_release_acceptance_10057a3f_20261106_r1_prefill/result.json"
)
PREVIOUS_PREFILL_SHA = (
    "998113f6662bd90309f622d04a14d4e644d99e0c16f422175a4f35cdc70c8d4f"
)
PREVIOUS_LEGACY = TASK / "manual_previous_regression_prefix_20261106_r1.json"
PREVIOUS_CPU_ABI = (
    TASK / "published_bdb170a1_cpu_abi_no_gpu_10057a3f_20261106_r1/result.json"
)
SMALL_TOKENS = (256, 512, 1024, 2048, 3072)
LARGE_TOKENS = (4096, 5120, 6144, 16384)
FIELDS = (
    "gemm1_us",
    "gemm2_us",
    "gemm1_tflops",
    "gemm2_tflops",
    "gemm1_bandwidth_gbs",
    "gemm2_bandwidth_gbs",
)
LEGACY_CASES = {
    "e64_t1536": (64, 1536, 8, 7168, 2048, False),
    "e96_t16384": (96, 16384, 6, 7168, 3072, True),
    "e1_t16384": (1, 16384, 1, 16384, 8192, True),
    "e96_t512": (96, 512, 6, 7168, 3072, True),
}
SOURCE_PATHS = {
    "main": "manual_pr_base_main_d0fd3939_hyg_20261106_r78/revision.json",
    "published": "published_bdb170a1_release_sources_20261105_r1/prefill/after_revision.json",
    "reference": "published_bdb170a1_release_sources_20261105_r1/legacy/extended_revision.json",
    "short_rows": "ordinary_short_row_wide_outputs_20261106_r1_complete_manifest/revision.json",
    "paired_scales": "native_paired_scale_accounting_20261106_r1_complete_manifest/revision.json",
    "combined": "short_rows_paired_scale_schedule_20261106_r1_complete_manifest/revision.json",
}
EXPECTED_COMMITS = {
    "main": MAIN,
    "published": PUBLISHED,
    "reference": REFERENCE,
    "short_rows": "355ae1028bd7f05b71fb35bb2a1307ad30900c8c",
    "paired_scales": "b64abcf53823e0363c11cb41432f3696a2b12c98",
    "combined": "be4241bf82ddd14260053e87642b8e78a5f42312",
}
NON_BALANCED_CASES = ("e96_t16384", "e1_t16384", "e96_t512")
NON_BALANCED_PUBLISHED = "9bebc86b4e8ac4c3c6b1db8021318b45b580fc6f"
NON_BALANCED_SOURCE_PATHS = {
    "main": SOURCE_PATHS["main"],
    "reference": SOURCE_PATHS["reference"],
    "published": "published_9bebc86b_complete_20261107_r94_manifest/revision.json",
}
SOURCE_PATHS["non_balanced_published"] = NON_BALANCED_SOURCE_PATHS["published"]
EXPECTED_COMMITS["non_balanced_published"] = NON_BALANCED_PUBLISHED
SUPPORT_FILES = (
    "run_complete_commit_compute.py",
    "run_revision_compute.py",
    "config_lookup_compat.py",
    "run_full_source_artifact_capture.py",
)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(
        (
            json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        ).encode()
    )
    temporary.replace(path)


def boot_id():
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def source_root(revision):
    path = (ROOT / revision["source_root"]).resolve()
    path.relative_to(TASK.resolve())
    return path


def shape(name, kind="regression"):
    values = (
        (96, int(name[1:]), 6, 5120, 2304, False)
        if kind == "prefill"
        else LEGACY_CASES[name]
    )
    keys = ("experts", "tokens", "topk", "model_dim", "inter_dim", "balanced")
    return dict(zip(keys, values), name=name, kind=kind)


def configuration(label, revision, pin, apre, container="hyg_fyd1"):
    return {
        "label": label,
        "revision": revision,
        "pin": pin,
        "apre": apre,
        "container": container,
    }


def prefill_configurations(tokens, after):
    large = tokens in LARGE_TOKENS
    return [
        configuration("before", "main", 0, int(large)),
        configuration("after", after, int(large), int(large)),
    ]


def best_pin_configurations(after):
    return [
        configuration("main_pin0_hyg", "main", 0, 1),
        configuration("after_pin0_hyg", after, 0, 1),
        configuration("after_pin1_hyg", after, 1, 1),
    ]


def non_balanced_cases():
    return [dict(shape(name), balanced=False) for name in NON_BALANCED_CASES]


def non_balanced_configurations(after="published"):
    return [
        configuration("main_pin0_hyg", "main", 0, 1),
        configuration("reference_pin0_hyg", "reference", 0, 1),
        configuration("after_pin0_hyg", after, 0, 1),
        configuration("reference_pin1_hyg", "reference", 1, 1),
        configuration("after_pin1_hyg", after, 1, 1),
    ]


def is_non_balanced_run(state):
    return state.get("plan", {}).get("scope") == "non_balanced_only"


def non_balanced_phase(after):
    return {
        "name": "non_balanced",
        "cases": non_balanced_cases(),
        "configurations": non_balanced_configurations(after),
        "samples_per_variant": 3,
        "verification_workers": 15,
        "performance_workers": 45,
        "compatibility_gate": "same-run ad9882df, same pin, +0.5%",
        "best_pin_gate": "after best pin versus same-run main pin0, +0.5%",
    }


def cli_shape(case, verify):
    arguments = [
        "--scenario",
        "verify" if verify else "kernel",
        "--data-format",
        "a4w4",
        "--act",
        "silu",
        "--no-bias",
        "--no-check-aot-cache",
        "--warmup",
        "5",
        "--iters",
        "101",
        "--seed",
        "0",
        "--data-init",
        "uniform",
        "--scale-init",
        "auto",
    ]
    for key in ("experts", "tokens", "topk", "model_dim", "inter_dim"):
        arguments.extend(["--" + key.replace("_", "-"), str(case[key])])
    return arguments


def metrics_median(records):
    # Never choose a subset by its measured speed.
    import statistics

    if len(records) != 3 or not all(record.get("valid") for record in records):
        raise RuntimeError("三次完整有效样本不足；保留日志，不重试挑选更快结果。")
    return {
        field: statistics.median(record["metrics"][field] for record in records)
        for field in FIELDS
    }


def comparison(before, after, limit=0.5):
    change = (after / before - 1) * 100
    return {
        "before_us": before,
        "after_us": after,
        "change_percent": change,
        "limit_percent": limit,
        "passed": change <= limit,
    }


def pr_comparison_rows(matrix):
    medians = matrix.get("medians", {})
    if matrix.get("name") == "non_balanced":
        medians = {
            prefix + name: medians[label + "_" + name]
            for name in NON_BALANCED_CASES
            for prefix, label in (
                ("before_", "main_pin0_hyg"),
                ("after_", "after_pin0_hyg"),
            )
            if label + "_" + name in medians
        }
    rows = []
    for key, before in medians.items():
        if not key.startswith("before_"):
            continue
        workload = key.removeprefix("before_")
        # Historical logs retain main pin1 samples, but the current PR
        # comparison contract uses only main pin0 as its before baseline.
        if workload.endswith("_p1"):
            continue
        after = medians.get("after_" + workload)
        if after is None:
            continue
        rows.append(
            {
                "workload": workload,
                "before": before,
                "after": after,
                "change_percent": {
                    metric: (after[metric] / before[metric] - 1) * 100
                    for metric in ("gemm1_us", "gemm2_us")
                },
            }
        )
    return rows


def same_kernel_artifact(before, after):
    return (
        before["elf_sha256"] == after["elf_sha256"]
        and before["launch"] == after["launch"]
    )


def reusable_prefill_history(history, after, current_boot):
    if (
        after != "published"
        or history.get("boot_id") != current_boot
        or history.get("status") != "complete"
    ):
        return False
    expected = {"before": MAIN, "after": PUBLISHED}
    records = [row for row in history.get("records", []) if row.get("valid")]
    return (
        bool(records)
        and {row.get("variant") for row in records} == set(expected)
        and all(
            row.get("result", {}).get("commit") == expected.get(row.get("variant"))
            for row in records
        )
    )


def elf_text_digest(path):
    # Read ELF64 section headers directly; no GPU or disassembler is needed.
    import struct

    data = path.read_bytes()
    if data[:6] != b"\x7fELF\x02\x01":
        raise RuntimeError("目标 code object 不是 little-endian ELF64。")
    offset = struct.unpack_from("<Q", data, 40)[0]
    entry_size, count, names_index = struct.unpack_from("<HHH", data, 58)
    headers = [
        struct.unpack_from("<IIQQQQIIQQ", data, offset + i * entry_size)
        for i in range(count)
    ]
    names_header = headers[names_index]
    names = data[names_header[4] : names_header[4] + names_header[5]]
    for header in headers:
        end = names.index(b"\0", header[0])
        if names[header[0] : end] == b".text":
            payload = data[header[4] : header[4] + header[5]]
            return {
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
    raise RuntimeError("目标 code object 缺少 .text。")


def make_plan(non_balanced_only=False):
    plan = {
        "entry_point": str(Path(__file__).resolve()),
        "execution": "host starts serial docker exec workers; user supplies performance availability",
        "production_sources_modified": False,
        "timer": "unchanged original torch.profiler",
        "warmup": 5,
        "iters": 101,
        "seed": 0,
        "log_more": 0,
        "per_worker_timeout_seconds": 115,
        "kill_after_seconds": 2,
        "outer_timeout_seconds": 119,
        "candidate_selection": "focused screening 3, crossover 6 (3 each direction), archive 3; improvement >=0.8%",
        "prefill_workload": {
            "global_experts": 384,
            "ep_ranks": 4,
            "local_experts": 96,
            "topk": 6,
            "model_dim": 5120,
            "inter_dim": 2304,
            "balanced": False,
            "measurement": "single GPU per-rank compute equivalent; not a four-GPU EP/P2P runtime test",
        },
        "pr_before_commit": MAIN,
        "pr_before_pin_policy": "main pin0/hyg only; no main pin1 measurement",
        "phases": [
            {
                "name": "candidate_short_rows",
                "tokens": [256],
                "target_stage": 2,
                "pin": 0,
                "apre": 0,
            },
            {
                "name": "candidate_paired_scales",
                "tokens": list(LARGE_TOKENS),
                "screen_and_control_tokens": [4096],
                "target_stage": 1,
                "pin": 1,
                "apre": 1,
            },
            {"name": "final_cpu_launch_abi", "expected_tests": 43, "cpu_only": True},
            {
                "name": "final_prefill",
                "tokens": list(SMALL_TOKENS + LARGE_TOKENS),
                "samples_per_variant": 3,
                "configurations_per_case": 2,
                "comparison": "complete main versus selected after; report both GEMMs and percentage changes for all nine T",
                "small_gate": "main same-run median +0.5%",
                "large_gate": "unchanged fixed extended archive +0.5%",
            },
            {
                "name": "final_legacy",
                "cases": list(LEGACY_CASES),
                "pins": [0, 1],
                "samples_per_variant": 3,
                "configurations_per_case": {"pin0": 3, "pin1": 2},
                "configurations": {
                    "pin0": ["main pin0/hyg", "ad9882df pin0/hyg", "after pin0/hyg"],
                    "pin1": ["ad9882df pin1/hyg", "after pin1/hyg"],
                },
                "gate": "ad9882df adjacent median +0.5%",
            },
            {
                "name": "best_pin",
                "cases": list(LEGACY_CASES),
                "samples_per_variant": 3,
                "configurations_per_case": 3,
                "configurations": [
                    "main pin0/hyg",
                    "after pin0/hyg",
                    "after pin1/hyg",
                ],
                "selection": "before main pin0 only; choose lower after median separately for each stage",
                "gate": "after best pin versus main pin0 +0.5%",
            },
        ],
        "reuse": "same boot, unchanged complete source and accepted whole groups only; preserve first failed release cohort",
        "source_manifests": {
            key: str(TASK / path) for key, path in SOURCE_PATHS.items()
        },
        "evidence_scope": "single-GPU per-rank compute; EP/P2P runtime is not validated",
    }
    if non_balanced_only:
        plan.pop("candidate_selection")
        plan.pop("prefill_workload")
        plan.update(
            scope="non_balanced_only",
            pr_after_commit=NON_BALANCED_PUBLISHED,
            phases=[non_balanced_phase("published")],
            expected_checks=18,
            reuse="fresh non-balanced cohort only; no balanced history or fixed prefill archive",
            source_manifests={
                key: str(TASK / path) for key, path in NON_BALANCED_SOURCE_PATHS.items()
            },
        )
    else:
        plan["phases"].insert(0, non_balanced_phase("non_balanced_published"))
        plan.update(
            expected_checks=60,
            non_balanced_after_commit=NON_BALANCED_PUBLISHED,
        )
    return plan


class Runner:
    def __init__(self, directory, resume=False, non_balanced_only=False):
        self.directory = directory.resolve()
        self.output = self.directory / "result.json"
        self.non_balanced_only = non_balanced_only
        self.source_paths = (
            NON_BALANCED_SOURCE_PATHS if non_balanced_only else SOURCE_PATHS
        )
        self.expected_commits = dict(EXPECTED_COMMITS)
        if non_balanced_only:
            self.expected_commits["published"] = NON_BALANCED_PUBLISHED
        self.revisions = {
            name: read(TASK / relative) for name, relative in self.source_paths.items()
        }
        self.fingerprints = {
            str(TASK / path): digest(TASK / path) for path in self.source_paths.values()
        }
        self.fingerprints.update(
            {str(TASK / path): digest(TASK / path) for path in SUPPORT_FILES}
        )
        self.fingerprints[str(Path(__file__).resolve())] = digest(Path(__file__))
        self.fingerprints.update(
            {
                str(path): digest(path)
                for path in (
                    ARCHIVE,
                    PREVIOUS_PREFILL,
                    PREVIOUS_LEGACY,
                    PREVIOUS_CPU_ABI,
                )
                if not non_balanced_only and path.exists()
            }
        )
        self.archive = {}
        self.previous = {}
        if not non_balanced_only:
            self.archive = read(ARCHIVE)
            if digest(ARCHIVE) != ARCHIVE_SHA:
                raise RuntimeError("固定通用扩展归档的 SHA256 不匹配。")
            self.previous = read(PREVIOUS_PREFILL)
            if digest(PREVIOUS_PREFILL) != PREVIOUS_PREFILL_SHA:
                raise RuntimeError("已完成的首次发布版九 T 日志发生了变化。")
        if resume:
            self.state = read(self.output)
            if is_non_balanced_run(self.state) != non_balanced_only:
                raise RuntimeError("旧 run 的测试范围不同，不能混用。")
            if (
                self.state["boot_id"] != boot_id()
                or self.state["fingerprints"] != self.fingerprints
            ):
                raise RuntimeError("boot 或冻结源码/脚本已变，不能混用旧 run。")
            if any(record["status"] != "complete" for record in self.state["records"]):
                raise RuntimeError(
                    "旧 run 含失败或中断 worker；先分析日志，不自动重试。"
                )
        else:
            self.directory.mkdir(parents=True, exist_ok=False)
            for name in ("logs", "results", "artifacts", "inputs"):
                (self.directory / name).mkdir(mode=0o777)
            self.state = {
                "status": "prepared",
                "boot_id": boot_id(),
                "host": socket.gethostname(),
                "start_utc": datetime.now(timezone.utc).isoformat(),
                "fingerprints": self.fingerprints,
                "plan": make_plan(non_balanced_only),
                "records": [],
                "matrices": {},
                "candidates": {},
                "groups": {},
                "sources": {},
                "performance_validity_basis": "user_guarantee; no driver availability audit",
                "production_modified": False,
                "release_acceptance_complete": False,
                "ep4_runtime_validated": False,
            }
            for name, source in (
                ("fixed_extended_archive", ARCHIVE),
                ("first_published_prefill", PREVIOUS_PREFILL),
                ("previous_legacy_prefix", PREVIOUS_LEGACY),
            ):
                if not non_balanced_only and source.exists():
                    (self.directory / "inputs" / (name + ".json")).write_bytes(
                        source.read_bytes()
                    )
            for name, relative in self.source_paths.items():
                (self.directory / "inputs" / (name + "_revision.json")).write_bytes(
                    (TASK / relative).read_bytes()
                )
            self.save()
        write_json(
            LOG_ROOT / "latest_run.json",
            {"directory": str(self.directory), "result": str(self.output)},
        )

    def save(self):
        write_json(self.output, self.state)

    def announce(self, message):
        line = f"[{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        with (self.directory / "controller.log").open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(line + "\n")

    def check_context(self, revisions=()):
        if boot_id() != self.state["boot_id"]:
            raise RuntimeError("机器已重启；停止当前 run，不跨 boot 汇总。")
        for path, expected in self.fingerprints.items():
            if digest(Path(path)) != expected:
                raise RuntimeError(f"冻结 manifest 或 worker 已改变：{path}")
        for name in set(revisions):
            for entry in self.revisions[name]["files"].values():
                if digest(ROOT / entry["path"]) != entry["sha256"]:
                    raise RuntimeError(f"冻结源码已改变：{entry['path']}")

    def preflight(self):
        self.announce("核验完整源码、container 和依赖版本；不启动 GPU 测试。")
        self.check_context(self.revisions)
        for name, revision in self.revisions.items():
            expected = self.expected_commits[name]
            if revision["commit"] != expected:
                raise RuntimeError(f"commit 不匹配：{name}")
            if not revision.get("complete_commit_source") or revision["snapshot"]:
                raise RuntimeError(f"必须使用完整 commit 源码：{name}")
            root = source_root(revision)
            actual = (
                subprocess.check_output(
                    ["git", "-C", str(root), "rev-parse", "HEAD"], cwd=ROOT
                )
                .decode()
                .strip()
            )
            if actual != revision["commit"]:
                raise RuntimeError(f"隔离 worktree 的 HEAD 不匹配：{name}")
            self.state["sources"][name] = {
                "commit": actual,
                "source_root": str(root),
                "manifest_sha256": digest(TASK / self.source_paths[name]),
                "verified_files": len(revision["files"]),
            }
        timer = self.revisions["main"]["files"]["aiter/test_common.py"]["sha256"]
        if any(
            revision["files"]["aiter/test_common.py"]["sha256"] != timer
            for revision in self.revisions.values()
        ):
            raise RuntimeError("aiter/test_common.py 与完整 main 不同；停止测试。")
        environments = {}
        for container in ("hyg_fyd1",):
            inspection = subprocess.check_output(
                [
                    "docker",
                    "inspect",
                    "--format",
                    '{"id":{{json .Id}},"image":{{json .Config.Image}},"running":{{json .State.Running}}}',
                    container,
                ],
                timeout=10,
                cwd=ROOT,
            ).decode()
            info = json.loads(inspection)
            if not info["running"]:
                raise RuntimeError(f"container 未运行：{container}")
            command = [
                "docker",
                "exec",
                container,
                "python3",
                "-c",
                "import importlib.metadata as m,json; print(json.dumps({k:m.version(k) for k in ('flydsl','torch')}))",
            ]
            info["packages"] = json.loads(
                subprocess.check_output(command, timeout=10, cwd=ROOT)
            )
            environments[container] = info
        if "containers" in self.state and self.state["containers"] != environments:
            raise RuntimeError("container 或工具链版本已改变，不能继续旧 run。")
        self.state["containers"] = environments
        self.save()

    def command(self, identifier, case, config, verify=False, artifact=False):
        revision = self.revisions[config["revision"]]
        result = self.directory / "results" / (identifier + ".json")
        env = {
            "PYTHONPATH": str(source_root(revision)),
            "AITER_USE_GROUPED_GEMM": "1",
            "AITER_GROUPED_DEBUG": "0",
            "ENABLE_CK": "0",
            "FLYDSL_DUMP_IR": "0",
            "AITER_LOG_MORE": "0",
            "AITER_FORCE_GFX1250": "1",
            "GPU_ARCHS": "gfx1250",
            "FLYDSL_GPU_ARCH": "gfx1250",
            "CU_NUM": "256",
            "AITER_FORCE_A8W4": "0",
            "AITER_MOE_EXPERT_BALANCE": str(case["balanced"]).lower(),
            "AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE": "1",
            "AITER_FLYDSL_EXPLICIT_VGPR_PARTITION": str(config["pin"]),
            "AITER_GROUPED_A4W4_A_PRESHUFFLE": str(config["apre"]),
            "AITER_FLYDSL_DISABLE_GEMM1_ACT": "0",
            "AITER_FLYDSL_DISABLE_GEMM1_REQUANT": "0",
        }
        worker = TASK / (
            "run_full_source_artifact_capture.py"
            if artifact
            else "run_complete_commit_compute.py"
        )
        arguments = [
            "python3",
            "-u",
            str(worker),
            "--revision",
            str(TASK / self.source_paths[config["revision"]]),
            "--kind",
            case["kind"],
            "--output",
            str(result),
        ]
        if case["kind"] == "prefill":
            arguments += [
                "--effective-tokens",
                str(case["tokens"]),
                "--a-preshuffle",
                str(config["apre"]),
            ]
        else:
            arguments += cli_shape(case, verify)
        if verify:
            arguments.append("--verify-only")
        if artifact:
            env["FLYDSL_RUNTIME_ENABLE_CACHE"] = "0"
            arguments += [
                "--artifact-directory",
                str(self.directory / "artifacts" / identifier),
            ]
        return self.wrap(config, identifier, env, arguments), result

    def wrap(self, config, identifier, env, arguments):
        token = self.directory.name + ":" + identifier
        pid_file = self.directory / "results" / (identifier + ".pid")
        # The marker allows cleanup of this exact timeout process only.
        command = [
            "docker",
            "exec",
            "--env",
            "CODEX_MANUAL_RUN_TOKEN=" + token,
            "--workdir",
            str(source_root(self.revisions[config["revision"]])),
            config["container"],
            "bash",
            "-c",
            'umask 000; printf "%s\\n" "$$" > "$1"; shift; exec "$@"',
            "manual-worker",
            str(pid_file),
            "timeout",
            "--signal=TERM",
            "--kill-after=2s",
            "115s",
            "env",
        ]
        return command + [key + "=" + value for key, value in env.items()] + arguments

    def stop_own_worker(self, config, identifier):
        code = (
            "import os,sys,signal; from pathlib import Path; "
            "p=Path(sys.argv[1]); pid=int(p.read_text()) if p.exists() else -1; "
            "e=Path('/proc')/str(pid)/'environ'; "
            "token=('CODEX_MANUAL_RUN_TOKEN='+sys.argv[2]).encode(); "
            "owned=e.exists() and token in e.read_bytes().split(b'\\0'); "
            "os.kill(pid,signal.SIGTERM) if owned else None; "
            "print('owned_timeout_term_sent',owned)"
        )
        command = [
            "docker",
            "exec",
            config["container"],
            "python3",
            "-c",
            code,
            str(self.directory / "results" / (identifier + ".pid")),
            self.directory.name + ":" + identifier,
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=8,
                cwd=ROOT,
                check=False,
            )
            return {
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        except (OSError, subprocess.SubprocessError) as error:
            return {"error": repr(error)}

    def execute(self, identifier, command, case, config, mode):
        old = next(
            (row for row in self.state["records"] if row["id"] == identifier), None
        )
        if old:
            if old["status"] != "complete" or old["command"] != command:
                raise RuntimeError(f"旧 worker 不完整或命令改变：{identifier}")
            return old
        self.check_context([config["revision"]])
        record = {
            "id": identifier,
            "mode": mode,
            "case": case,
            "configuration": config,
            "command": command,
            "command_shell": shlex.join(command),
            "status": "running",
            "start": time.time(),
            "raw_log": str(self.directory / "logs" / (identifier + ".log")),
        }
        self.state["records"].append(record)
        self.save()
        with (self.directory / "commands.sh").open(
            "a", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(f"# {identifier}\n{record['command_shell']}\n")
        self.announce(f"执行 {identifier}；原始日志：logs/{identifier}.log")
        with Path(record["raw_log"]).open("w", encoding="utf-8", newline="\n") as log:
            child = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            record["host_docker_pid"] = child.pid
            self.save()
            try:
                record["returncode"] = child.wait(timeout=119)
            except BaseException as error:
                record["cleanup"] = self.stop_own_worker(config, identifier)
                child.kill()
                child.wait(timeout=5)
                record.update(
                    status="failed",
                    error=repr(error),
                    returncode="interrupted_or_outer_timeout",
                )
                raise
            finally:
                record["end"] = time.time()
                record["elapsed_seconds"] = record["end"] - record["start"]
                record["log_sha256"] = digest(Path(record["raw_log"]))
                self.save()
        if record["returncode"] != 0:
            record["status"] = "failed"
            self.save()
            raise RuntimeError(
                f"worker 失败或超时：{identifier}；保留日志，停止后续测试。"
            )
        record["status"] = "complete"
        self.check_context([config["revision"]])
        self.save()
        return record

    def compute(self, identifier, case, config, verify=False, artifact=False):
        command, output = self.command(identifier, case, config, verify, artifact)
        mode = "artifact" if artifact else "correctness" if verify else "performance"
        record = self.execute(identifier, command, case, config, mode)
        report = read(output)
        revision = self.revisions[config["revision"]]
        if report["commit"] != revision["commit"]:
            raise RuntimeError("实际运行的 commit 与 manifest 不匹配。")
        if report["pin_requested"] != str(config["pin"]) or report["log_more"] != "0":
            raise RuntimeError("实际 pin 或 AITER_LOG_MORE 不匹配。")
        if report["warmup"] != 5 or report["iters"] != 101:
            raise RuntimeError("实际 warmup/iters 不匹配。")
        if (
            self.non_balanced_only or identifier.startswith("non_balanced_")
        ) and report.get("expert_balance") != "false":
            raise RuntimeError("Non-balanced 测试实际未设置 expert_balance=false。")
        if (
            bool(report["correctness_only"]) != verify
            or not report["metrics"]["passed"]
        ):
            raise RuntimeError("原始 numerical reference 未通过。")
        if report["routes"]["valid_routes"] != case["tokens"] * case["topk"]:
            raise RuntimeError("有效 route 数量发生了变化。")
        if report["ep4_runtime_validated"]:
            raise RuntimeError("当前 worker 只能作为 per-rank compute 证据。")
        if not verify:
            for field in FIELDS:
                value = report["metrics"][field]
                if (
                    not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value <= 0
                ):
                    raise RuntimeError(f"性能指标无效：{identifier}/{field}")
        record.update(
            result=str(output),
            result_sha256=digest(output),
            metrics=report["metrics"],
            routes=report["routes"],
            launches=report["launches"],
            valid=True,
            performance_valid=not verify,
            performance_validity_basis=(
                "user_guarantee" if not verify else "untimed correctness only"
            ),
        )
        if artifact:
            path = self.directory / "artifacts" / identifier / "result.json"
            captured = read(path)
            if captured["status"] != "complete" or captured["timer_called"]:
                raise RuntimeError("实际 ELF 导出失败，或误调用了 timer。")
            record["artifacts"] = []
            for row in captured["records"]:
                binary = Path(row["binaries"][0]["path"])
                isa_identifier = identifier + "_isa_g" + str(row["stage"])
                isa_command = self.wrap(
                    config,
                    isa_identifier,
                    {
                        "LD_LIBRARY_PATH": (
                            "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/rocm_sysdeps/lib:"
                            "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/rocm_sysdeps/lib"
                        ),
                    },
                    [
                        "/data/yanguahe/code/wk_sp1/llvm-project/mlir_install/bin/llvm-objdump",
                        "--disassemble",
                        "--mcpu=gfx1250",
                        str(binary),
                    ],
                )
                isa = self.execute(
                    isa_identifier, isa_command, case, config, "cpu_only"
                )
                record["artifacts"].append(
                    {
                        "stage": row["stage"],
                        "path": str(binary),
                        "elf_sha256": digest(binary),
                        "text": elf_text_digest(binary),
                        "isa_log": isa["raw_log"],
                        "isa_sha256": isa["log_sha256"],
                    }
                )
        self.save()
        return record

    def group(self, name, case, configs, sample):
        identifier = f"{name}_{case['name']}_s{sample}"
        if identifier in self.state["groups"]:
            return self.state["groups"][identifier]["records"]
        order = configs if sample % 2 == 0 else list(reversed(configs))
        rows = [
            self.compute(identifier + "_" + item["label"], case, item) for item in order
        ]
        routes = {
            (row["routes"]["ids_sha256"], row["routes"]["weights_sha256"])
            for row in rows
        }
        gaps = [later["start"] - earlier["end"] for earlier, later in pairwise(rows)]
        if len(routes) != 1 or any(gap < 0 or gap > 30 for gap in gaps):
            raise RuntimeError(
                f"配对 routing 或启动间隔不符合要求：{identifier}；不自动重测。"
            )
        self.state["groups"][identifier] = {
            "records": rows,
            "accepted": True,
            "sample": sample,
            "direction": sample % 2,
            "launch_gaps_seconds": gaps,
        }
        self.save()
        return rows

    def matrix(self, name, cases, configurations, count=3, prefix=None):
        if name in self.state["matrices"]:
            return self.state["matrices"][name]
        records = []
        for case in cases:
            configs = configurations(case)
            for sample in range(count):
                key = (case["name"], sample)
                rows = prefix.get(key) if prefix else None
                records.extend(
                    rows if rows else self.group(name, case, configs, sample)
                )
        medians = {}
        for case in cases:
            for config in configurations(case):
                rows = [
                    row
                    for row in records
                    if row["case"]["name"] == case["name"]
                    and row["configuration"]["label"] == config["label"]
                ]
                if count == 3:
                    medians[config["label"] + "_" + case["name"]] = metrics_median(rows)
                else:
                    for direction in (0, 1):
                        part = [
                            row
                            for row in rows
                            if int(row["id"].rsplit("_s", 1)[1].split("_", 1)[0]) % 2
                            == direction
                        ]
                        medians[f"{config['label']}_{case['name']}_d{direction}"] = (
                            metrics_median(part)
                        )
        result = {
            "name": name,
            "count": count,
            "status": "complete",
            "performance_valid": True,
            "cases": cases,
            "medians": medians,
            "records": records,
            "performance_validity_basis": "user_guarantee; imported accepted history identified per record",
        }
        self.state["matrices"][name] = result
        self.save()
        return result

    def candidate(self, name, stage, screen_token, tokens, pin, apre):
        if name in self.state["candidates"]:
            return self.state["candidates"][name]["passed"]

        def configs(_case):
            return [
                configuration("before", "published", pin, apre),
                configuration("after", name, pin, apre),
            ]

        def main_configs(_case):
            return [
                configuration("before", "main", pin, apre),
                configuration("after", name, pin, apre),
            ]

        case = shape("t" + str(screen_token), "prefill")
        target_artifacts = []
        for config in configs(case):
            row = self.compute(
                name + "_artifact_" + config["label"],
                case,
                config,
                verify=True,
                artifact=True,
            )
            target_artifacts.append(
                dict(
                    next(item for item in row["artifacts"] if item["stage"] == stage),
                    launch=next(
                        item for item in row["launches"] if item["stage"] == stage
                    ),
                )
            )
        outcome = {"passed": False, "artifacts": target_artifacts, "assessments": []}
        if same_kernel_artifact(*target_artifacts):
            outcome["status"] = "no_target_elf_or_launch_change"
        else:
            for phase, count in (("screen", 3), ("control", 6), ("archive", 3)):
                phase_tokens = tokens if phase == "archive" else (screen_token,)
                # Archival small-token validation uses its actual main denominator.
                selected = configs
                if phase == "archive" and apre == 0:
                    selected = main_configs
                result = self.matrix(
                    name + "_" + phase,
                    [shape("t" + str(token), "prefill") for token in phase_tokens],
                    selected,
                    count=count,
                )
                checks = []
                for item in result["cases"]:
                    for direction in ((0, 1) if phase == "control" else (None,)):
                        suffix = "_d" + str(direction) if direction is not None else ""
                        before = result["medians"]["before_" + item["name"] + suffix]
                        after = result["medians"]["after_" + item["name"] + suffix]
                        for metric in ("gemm1_us", "gemm2_us"):
                            limit = (
                                -0.8
                                if metric == f"gemm{stage}_us"
                                and not (phase == "archive" and apre == 0)
                                else 0.5
                            )
                            check = comparison(before[metric], after[metric], limit)
                            check.update(
                                case=item["name"], metric=metric, direction=direction
                            )
                            checks.append(check)
                            if phase == "archive" and apre == 1:
                                fixed = self.archive["medians"][
                                    "extended_" + item["name"]
                                ][metric]
                                check = comparison(fixed, after[metric])
                                check.update(
                                    case=item["name"],
                                    metric=metric,
                                    denominator="fixed_extended_archive",
                                )
                                checks.append(check)
                passed = all(check["passed"] for check in checks)
                outcome["assessments"].append(
                    {"phase": phase, "passed": passed, "checks": checks}
                )
                outcome["status"] = phase + ("_passed" if passed else "_failed")
                if not passed:
                    break
            outcome["passed"] = outcome["status"] == "archive_passed"
        self.state["candidates"][name] = outcome
        self.save()
        self.announce(f"候选 {name}：{outcome['status']}；继续收集剩余验收数据。")
        return outcome["passed"]

    def cpu_abi(self, after):
        if after == "published" and PREVIOUS_CPU_ABI.exists():
            previous = read(PREVIOUS_CPU_ABI)
            if (
                previous["boot_id"] == self.state["boot_id"]
                and previous["status"] == "complete"
                and previous["published_head"] == PUBLISHED
                and previous["cpu_abi_counts"]
                == {"tests": 43, "failures": 0, "errors": 0, "skipped": 0}
            ):
                self.state["cpu_abi"] = {
                    "counts": previous["cpu_abi_counts"],
                    "passed": True,
                    "cpu_only": True,
                    "reused": True,
                    "source": str(PREVIOUS_CPU_ABI),
                    "sha256": digest(PREVIOUS_CPU_ABI),
                }
                self.save()
                return
        config = configuration("after", after, 1, 1)
        source = source_root(self.revisions[after])
        identifier = "final_cpu_launch_abi"
        xml = self.directory / "results" / (identifier + ".xml")
        env = {
            "PYTHONPATH": str(source),
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "ENABLE_CK": "0",
            "GPU_ARCHS": "gfx1250",
            "FLYDSL_GPU_ARCH": "gfx1250",
            "AITER_FORCE_GFX1250": "1",
            "CU_NUM": "256",
            "AITER_LOG_MORE": "0",
            "ROCR_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "CUDA_VISIBLE_DEVICES": "",
        }
        args = [
            "python3",
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "--junitxml=" + str(xml),
            "op_tests/tuning_tests/test_grouped_gemm_launch_abi.py",
        ]
        self.execute(
            identifier,
            self.wrap(config, identifier, env, args),
            {"name": "cpu_launch_abi"},
            config,
            "cpu_only",
        )
        suites = list(ET.parse(xml).getroot().iter("testsuite"))
        counts = {
            key: sum(int(suite.attrib.get(key, 0)) for suite in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        if counts != {"tests": 43, "failures": 0, "errors": 0, "skipped": 0}:
            raise RuntimeError(f"CPU launch ABI 验收未通过：{counts}")
        self.state["cpu_abi"] = {
            "counts": counts,
            "passed": True,
            "xml": str(xml),
            "xml_sha256": digest(xml),
            "cpu_only": True,
        }
        self.save()

    def legacy_prefix(self, after):
        if after != "published" or not PREVIOUS_LEGACY.exists():
            return {}
        previous = read(PREVIOUS_LEGACY)
        if previous["boot_id"] != self.state["boot_id"]:
            return {}
        prefix = {}
        for pair in previous["pairs"]:
            if not pair["accepted"]:
                continue
            rows = [
                row
                for row in previous["records"]
                if row.get("valid")
                and row["workload"] == pair["workload"]
                and row["sample"] == pair["sample"]
            ]
            if len(rows) != 3:
                continue
            expected = {"before": MAIN, "extended": REFERENCE, "after": PUBLISHED}
            if any(
                row.get("result", {}).get("commit") != expected.get(row.get("variant"))
                for row in rows
            ):
                continue
            case_name, pin = pair["workload"].rsplit("_p", 1)
            normalized = []
            for row in rows:
                if pin == "1" and row["variant"] == "before":
                    continue
                role = {
                    "before": "main",
                    "extended": "reference",
                    "after": "published",
                }[row["variant"]]
                if row["result"]["commit"] != self.revisions[role]["commit"]:
                    raise RuntimeError(
                        "旧 accepted 配对的源码与当前完整 manifest 不同。"
                    )
                label = "reference" if row["variant"] == "extended" else row["variant"]
                normalized.append(
                    {
                        "id": f"historical_{pair['workload']}_s{pair['sample']}_{label}",
                        "case": dict(shape(case_name), name=pair["workload"]),
                        "configuration": configuration(label, role, int(pin), 1),
                        "metrics": {key: row[key] for key in FIELDS},
                        "valid": True,
                        "mode": "performance",
                        "performance_valid": True,
                        "routes": row["result"]["routes"],
                        "launches": row["result"]["launches"],
                        "raw_log": row["raw_log"],
                        "performance_validity_basis": "previous accepted whole pair; original audits retained",
                        "historical_result": str(PREVIOUS_LEGACY),
                        "historical_result_sha256": digest(PREVIOUS_LEGACY),
                    }
                )
            prefix[(pair["workload"], pair["sample"])] = normalized
        self.announce(f"复用同 boot、同源码的旧表 accepted 配对：{len(prefix)} 组。")
        return prefix

    def final_tests(self, after):
        self.state["selected_after"] = after
        self.state["selected_after_commit"] = self.revisions[after]["commit"]
        self.save()
        self.cpu_abi(after)
        same_published_cohort = reusable_prefill_history(
            self.previous, after, self.state["boot_id"]
        )
        if same_published_cohort:
            self.state["matrices"]["final_prefill"] = {
                "name": "final_prefill",
                "reused": True,
                "status": "complete",
                "source": str(PREVIOUS_PREFILL),
                "source_sha256": PREVIOUS_PREFILL_SHA,
                "medians": self.previous["medians"],
                "records": [],
                "cases": [
                    shape("t" + str(token), "prefill")
                    for token in SMALL_TOKENS + LARGE_TOKENS
                ],
                "reason": "Preserve first complete failed release cohort; do not remeasure unchanged code to select a passing cohort",
            }
            self.save()
        else:
            cases = [
                shape("t" + str(token), "prefill")
                for token in SMALL_TOKENS + LARGE_TOKENS
            ]
            # Complete source changes require complete numerical and timing coverage.
            for case in cases:
                for config in prefill_configurations(case["tokens"], after):
                    self.compute(
                        f"final_reference_{case['name']}_{config['label']}",
                        case,
                        config,
                        verify=True,
                    )
            self.matrix(
                "final_prefill",
                cases,
                lambda case: prefill_configurations(case["tokens"], after),
            )
        cases = [
            dict(shape(name), name=f"{name}_p{pin}", pin=pin)
            for name in LEGACY_CASES
            for pin in (0, 1)
        ]

        def legacy_configs(case):
            before = [configuration("before", "main", 0, 1)] if case["pin"] == 0 else []
            return before + [
                configuration("reference", "reference", case["pin"], 1),
                configuration("after", after, case["pin"], 1),
            ]

        self.matrix(
            "final_legacy", cases, legacy_configs, prefix=self.legacy_prefix(after)
        )
        self.finish_best_pin_tests(after)

    def finish_best_pin_tests(self, after):
        self.matrix(
            "best_pin",
            [shape(name) for name in LEGACY_CASES],
            lambda _case: best_pin_configurations(after),
        )
        assess_results(self.state, self.archive)
        self.state["status"] = (
            "complete"
            if self.state["acceptance"]["passed"]
            else "complete_with_regressions"
        )
        self.save()
        export_results(self.directory, self.state)

    def finish_previous_best_pin(self, directory):
        """Retain completed evidence and replace only the withdrawn main pin1 test."""
        self.preflight()
        directory = directory.resolve()
        directory.relative_to(LOG_ROOT.resolve())
        previous_path = directory / "result.json"
        previous = read(previous_path)
        after = previous["selected_after"]
        if (
            previous["boot_id"] != self.state["boot_id"]
            or previous["selected_after_commit"] != self.revisions[after]["commit"]
            or previous["containers"]["hyg_fyd1"]
            != self.state["containers"]["hyg_fyd1"]
        ):
            raise RuntimeError("boot、完整 after 或 hyg_fyd1 已改变，不能复用旧测试。")
        for path, expected in self.fingerprints.items():
            if path == str(Path(__file__).resolve()):
                continue
            if previous["fingerprints"].get(path) != expected:
                raise RuntimeError(f"旧测试的冻结输入不匹配：{path}")
        failed = [row for row in previous["records"] if row["status"] != "complete"]
        if not failed or any(
            row["configuration"]["label"] != "main_pin1_dsr"
            or row["configuration"]["container"] != "dsr1_ep4"
            or row.get("returncode") != 1
            for row in failed
        ):
            raise RuntimeError("仅支持撤销 main pin1 后补跑；其他失败不能自动重试。")
        if "best_pin" in previous["matrices"]:
            raise RuntimeError("最佳 pin 已完成，不能重测相同源码挑选结果。")
        for name in ("final_prefill", "final_legacy"):
            matrix = previous["matrices"].get(name, {})
            if matrix.get("status") != "complete":
                raise RuntimeError(f"旧测试未完成 {name}，不能仅补最佳 pin。")
            inherited = copy.deepcopy(matrix)
            inherited.update(
                reused=True,
                source=str(previous_path),
                source_sha256=digest(previous_path),
                reason="Preserve completed user cohort under the revised main pin0-only policy",
            )
            self.state["matrices"][name] = inherited
        if not previous.get("cpu_abi", {}).get("passed"):
            raise RuntimeError("旧测试的 CPU launch ABI 未通过。")
        self.state.update(
            selected_after=after,
            selected_after_commit=self.revisions[after]["commit"],
            candidates=copy.deepcopy(previous["candidates"]),
            cpu_abi=copy.deepcopy(previous["cpu_abi"]),
            inherited_run={
                "path": str(previous_path),
                "sha256": digest(previous_path),
                "original_status": previous["status"],
                "withdrawn_test_ids": [row["id"] for row in failed],
                "policy": "PR before main pin0/hyg only; after pin0/pin1/hyg",
            },
        )
        self.save()
        self.announce(
            "保留旧九 T、ad9882df 验收和失败候选；仅补 main pin0 对 after pin0/pin1。"
        )
        self.finish_best_pin_tests(after)

    def run(self):
        self.preflight()
        self.announce("默认先测试新增三个 Non-balanced workload，再继续原有完整测试。")
        self.non_balanced_tests("non_balanced_published")
        short_rows = self.candidate("short_rows", 2, 256, (256,), 0, 0)
        paired_scales = self.candidate("paired_scales", 1, 4096, LARGE_TOKENS, 1, 1)
        after = (
            "combined"
            if short_rows and paired_scales
            else (
                "short_rows"
                if short_rows
                else "paired_scales" if paired_scales else "published"
            )
        )
        self.announce(
            f"完整验收使用 {after} / {self.revisions[after]['commit']}；正式源码保持不变。"
        )
        self.final_tests(after)
        self.announce(
            f"测试脚本已结束：{self.state['status']}。汇总：{self.directory / 'summary.md'}"
        )

    def non_balanced_tests(self, after="published"):
        cases = non_balanced_cases()
        configs = non_balanced_configurations(after)
        self.state["non_balanced_after_commit"] = self.revisions[after]["commit"]
        self.save()
        for case in cases:
            for config in configs:
                self.compute(
                    f"non_balanced_reference_{case['name']}_{config['label']}",
                    case,
                    config,
                    verify=True,
                )
        self.matrix("non_balanced", cases, lambda _case: configs)

    def run_non_balanced(self):
        self.preflight()
        self.state.update(
            selected_after="published",
            selected_after_commit=self.revisions["published"]["commit"],
        )
        self.announce("单独测试三个 Non-balanced workload；每配置三次，同期正反交替。")
        self.non_balanced_tests()
        assess_results(self.state, self.archive)
        self.state["status"] = (
            "complete"
            if self.state["acceptance"]["passed"]
            else "complete_with_regressions"
        )
        self.save()
        self.announce(
            f"测试脚本已结束：{self.state['status']}。汇总：{self.directory / 'summary.md'}"
        )


def assess_results(state, archive):
    matrices = state.get("matrices", {})
    non_balanced_only = is_non_balanced_run(state)
    include_non_balanced = non_balanced_only or any(
        phase["name"] == "non_balanced"
        for phase in state.get("plan", {}).get("phases", [])
    )
    state["pr_before_after_comparison"] = {
        name: pr_comparison_rows(matrices[name])
        for name in ("final_prefill", "final_legacy", "non_balanced")
        if name in matrices
    }
    checks = []
    pending = []

    def assess_pair(scope, workload, metric, denominator, before, after):
        details = {
            "scope": scope,
            "workload": workload,
            "metric": metric,
            "denominator": denominator,
        }
        if before is None or after is None:
            pending.append(dict(details, reason="缺少完整三次中值"))
            return
        checks.append(dict(comparison(before, after), **details))

    prefill = matrices.get("final_prefill", {}).get("medians", {})
    for tokens in (() if non_balanced_only else SMALL_TOKENS + LARGE_TOKENS):
        for metric in ("gemm1_us", "gemm2_us"):
            denominator = (
                prefill.get(f"before_t{tokens}", {}).get(metric)
                if tokens in SMALL_TOKENS
                else archive.get("medians", {})
                .get(f"extended_t{tokens}", {})
                .get(metric)
            )
            assess_pair(
                "prefill",
                f"t{tokens}",
                metric,
                "same_run_main" if tokens in SMALL_TOKENS else "fixed_extended_archive",
                denominator,
                prefill.get(f"after_t{tokens}", {}).get(metric),
            )
    legacy = matrices.get("final_legacy", {}).get("medians", {})
    for name in (() if non_balanced_only else LEGACY_CASES):
        for pin in (0, 1):
            key = f"{name}_p{pin}"
            for metric in ("gemm1_us", "gemm2_us"):
                assess_pair(
                    "legacy",
                    key,
                    metric,
                    "same_run_ad9882df",
                    legacy.get("reference_" + key, {}).get(metric),
                    legacy.get("after_" + key, {}).get(metric),
                )
    non_balanced = matrices.get("non_balanced", {}).get("medians", {})
    if include_non_balanced:
        for name in NON_BALANCED_CASES:
            for pin in (0, 1):
                for metric in ("gemm1_us", "gemm2_us"):
                    assess_pair(
                        "non_balanced",
                        f"{name}_p{pin}",
                        metric,
                        "same_run_ad9882df",
                        non_balanced.get(f"reference_pin{pin}_hyg_{name}", {}).get(
                            metric
                        ),
                        non_balanced.get(f"after_pin{pin}_hyg_{name}", {}).get(metric),
                    )

    def best_pin_rows(best, names, scope):
        selections = []
        for name in names:
            for metric in ("gemm1_us", "gemm2_us"):
                before = "main_pin0_hyg"
                after_configs = ("after_pin0_hyg", "after_pin1_hyg")
                missing = [
                    item
                    for item in (before, *after_configs)
                    if best.get(item + "_" + name, {}).get(metric) is None
                ]
                # Selecting the minimum of an incomplete set hides an untested pin.
                if missing:
                    pending.append(
                        {
                            "scope": scope,
                            "workload": name,
                            "metric": metric,
                            "denominator": "main_pin0_hyg",
                            "reason": "缺少完整三次中值",
                            "missing_configurations": missing,
                        }
                    )
                    continue
                after = min(
                    after_configs,
                    key=lambda item: best[item + "_" + name][metric],
                )
                row = comparison(
                    best[before + "_" + name][metric],
                    best[after + "_" + name][metric],
                )
                row.update(
                    workload=name,
                    metric=metric,
                    before_configuration=before,
                    after_configuration=after,
                )
                selections.append(row)
                checks.append(dict(row, scope=scope, denominator="main_pin0_hyg"))
        return selections

    state["best_pin_comparison"] = best_pin_rows(
        matrices.get("best_pin", {}).get("medians", {}),
        () if non_balanced_only else LEGACY_CASES,
        "legacy_best_pin",
    )
    state["non_balanced_best_pin_comparison"] = best_pin_rows(
        non_balanced,
        NON_BALANCED_CASES if include_non_balanced else (),
        "non_balanced_best_pin",
    )
    cpu_abi = state.get("cpu_abi", {})
    cpu_abi_required = not non_balanced_only
    complete = not pending and (not cpu_abi_required or "passed" in cpu_abi)
    state["acceptance"] = {
        "checks": checks,
        "failed": [row for row in checks if not row["passed"]],
        "pending": pending,
        "expected_checks": len(checks) + len(pending),
        "completed_checks": len(checks),
        "passed_checks": sum(row["passed"] for row in checks),
        "failed_checks": sum(not row["passed"] for row in checks),
        "pending_checks": len(pending),
        "cpu_abi_passed": cpu_abi.get("passed"),
        "cpu_abi_required": cpu_abi_required,
        "pr_comparison_pin_policy": "PR before main pin0/hyg only; after best of pin0/pin1/hyg",
        "complete": complete,
        "passed": complete
        and all(row["passed"] for row in checks)
        and (not cpu_abi_required or cpu_abi.get("passed")),
    }
    # Publishing and validating the eventual live PR HEAD remain separate steps.
    state["release_acceptance_complete"] = False


def export_results(directory, state):
    before_commit = (
        state.get("sources", {})
        .get("main", {})
        .get("commit", state.get("plan", {}).get("pr_before_commit", MAIN))
    )
    lines = [
        "# 剩余测试汇总",
        "",
        f"状态：`{state['status']}`。",
        f"PR 前完整 main 源码：`{before_commit}`。",
        f"选定的验收源码：`{state.get('selected_after_commit', '尚未选定')}`。",
        "",
        "性能有效性由用户保证；本脚本未进行 GPU 空闲或所有权检查。",
        "计时使用原 torch.profiler，warmup=5、iters=101、seed=0、AITER_LOG_MORE=0。",
        "证据范围为单 GPU per-rank compute；真实四卡 EP/P2P runtime 尚未验收。",
        "",
    ]
    if is_non_balanced_run(state):
        lines += [
            "本次仅运行新增的三个 Non-balanced workload；不包含原九 T、balanced 旧表或 CPU ABI 全表。",
            "",
        ]
    if "non_balanced_after_commit" in state:
        lines += [
            f"新增 Non-balanced 测试的发布版源码：`{state['non_balanced_after_commit']}`。",
            "所有新增配置均设置 AITER_MOE_EXPERT_BALANCE=false、APRE=1，并在 hyg_fyd1 执行。",
            "",
        ]
    if state.get("error"):
        lines += [f"停止原因：`{state['error']}`。", ""]
    for name, matrix in state["matrices"].items():
        lines += [
            f"## {name}",
            "",
            "| 配置 / workload | GEMM1 μs | GEMM1 TFLOPS | GEMM1 BW GB/s | GEMM2 μs | GEMM2 TFLOPS | GEMM2 BW GB/s |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for key, values in matrix["medians"].items():
            numbers = [
                values[field]
                for field in (
                    "gemm1_us",
                    "gemm1_tflops",
                    "gemm1_bandwidth_gbs",
                    "gemm2_us",
                    "gemm2_tflops",
                    "gemm2_bandwidth_gbs",
                )
            ]
            lines.append(
                "| "
                + key
                + " | "
                + " | ".join(f"{number:.6f}" for number in numbers)
                + " |"
            )
        if matrix.get("reused"):
            lines += [
                "",
                f"复用已完成的完整验收：`{matrix['source']}`；保留其原始失败结论。",
            ]
        lines.append("")
    pr_comparisons = {
        name: pr_comparison_rows(matrix)
        for name, matrix in state["matrices"].items()
        if name in ("final_prefill", "final_legacy", "non_balanced")
    }
    for name, title in (
        ("final_prefill", "九个 T 的 PR 修改前后性能对比"),
        ("final_legacy", "旧表四 workload 的 PR 修改前后性能对比（固定 pin）"),
        (
            "non_balanced",
            "新增三个 Non-balanced workload 的 PR 修改前后性能对比（pin0）",
        ),
    ):
        if name not in pr_comparisons:
            continue
        lines += [
            f"## {title}",
            "",
            "本表以完整 main 的三次中值为分母；正值表示耗时增加，负值表示耗时减少。",
            "TFLOPS、BW 和实际 APRE/pin 配置见上方中值表及原始记录；规定门槛另列。",
            "",
            "| workload | PR 前 GEMM1 μs | PR 后 GEMM1 μs | GEMM1 耗时变化 | PR 前 GEMM2 μs | PR 后 GEMM2 μs | GEMM2 耗时变化 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for row in pr_comparisons[name]:
            before, after = row["before"], row["after"]
            changes = row["change_percent"]
            lines.append(
                f"| {row['workload']} | {before['gemm1_us']:.6f} | {after['gemm1_us']:.6f} | {changes['gemm1_us']:+.6f}% | {before['gemm2_us']:.6f} | {after['gemm2_us']:.6f} | {changes['gemm2_us']:+.6f}% |"
            )
        lines.append("")
    write_json(
        directory / "pr_before_after_comparison.json",
        {
            "run_status": state["status"],
            "before_commit": before_commit,
            "after_commit": state.get("selected_after_commit"),
            "non_balanced_after_commit": state.get("non_balanced_after_commit"),
            "matrices": pr_comparisons,
        },
    )
    if "acceptance" in state:
        acceptance = state["acceptance"]
        write_json(directory / "assessment.json", acceptance)
        lines += [
            "## 0.5% 门槛",
            "",
            (
                f"已完成 {acceptance['completed_checks']}/{acceptance['expected_checks']} 项："
                f"通过 {acceptance['passed_checks']} 项、未通过 {acceptance['failed_checks']} 项、"
                f"未完成 {acceptance['pending_checks']} 项。CPU ABI："
                + (
                    str(acceptance["cpu_abi_passed"])
                    if acceptance.get("cpu_abi_required", True)
                    else "本模式不执行"
                )
                + "。"
            ),
            "未完成项目不计为通过；全部门槛通过：" + str(acceptance["passed"]) + "。",
            "",
            "| workload | kernel | 分母 | 分母 μs | after μs | 耗时变化 | 通过 |",
            "|---|---|---|---:|---:|---:|---|",
        ]
        for row in state["acceptance"]["checks"]:
            lines.append(
                f"| {row['workload']} | {row['metric']} | {row['denominator']} | {row['before_us']:.6f} | {row['after_us']:.6f} | {row['change_percent']:+.6f}% | {row['passed']} |"
            )
        if acceptance["pending"]:
            lines += [
                "",
                "### 未完成项目",
                "",
                "| 范围 | workload | kernel | 缺失原因 |",
                "|---|---|---|---|",
            ]
            for row in acceptance["pending"]:
                missing = ", ".join(row.get("missing_configurations", []))
                lines.append(
                    f"| {row['scope']} | {row['workload']} | {row['metric']} | {row['reason']} {missing} |"
                )
        lines += [
            "",
            "## PR 前 pin0 与 PR 后各阶段最佳 pin 对比",
            "",
            "PR 前仅使用 main 的 pin0/hyg_fyd1；PR 后在 hyg_fyd1 的 pin0/pin1 中逐 GEMM 选择三次中值更快者。",
            "",
            "| workload | kernel | before 配置 | after 配置 | before μs | after μs | 耗时变化 |",
            "|---|---|---|---|---:|---:|---:|",
        ]
        for row in state["best_pin_comparison"]:
            lines.append(
                f"| {row['workload']} | {row['metric']} | {row['before_configuration']} | {row['after_configuration']} | {row['before_us']:.6f} | {row['after_us']:.6f} | {row['change_percent']:+.6f}% |"
            )
        if state.get("non_balanced_best_pin_comparison"):
            lines += [
                "",
                "## 新增三个 Non-balanced workload：main pin0 与发布版最佳 pin",
                "",
                "两种发布版 pin 均完成三次采样后，逐 GEMM 选择中值更快者；所有配置均为 Non-balanced。",
                "",
                "| workload | kernel | before 配置 | after 配置 | before μs | after μs | 耗时变化 |",
                "|---|---|---|---|---:|---:|---:|",
            ]
            for row in state["non_balanced_best_pin_comparison"]:
                lines.append(
                    f"| {row['workload']} | {row['metric']} | {row['before_configuration']} | {row['after_configuration']} | {row['before_us']:.6f} | {row['after_us']:.6f} | {row['change_percent']:+.6f}% |"
                )
        lines += [
            "",
            "正式源码、branch、commit、GitHub PR 均未由本脚本修改；发布与实际 PR HEAD 复核仍待后续执行。",
            "",
        ]
    (directory / "summary.md").write_bytes(("\n".join(lines) + "\n").encode())
    with (directory / "samples.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(
            [
                "phase",
                "id",
                "workload",
                "balanced",
                "revision",
                "container",
                "pin",
                "apre",
                *FIELDS,
                "raw_log",
                "validity_basis",
            ]
        )
        seen = set()
        records = list(state["records"])
        for matrix in state["matrices"].values():
            records.extend(matrix["records"])
        for row in records:
            if (
                row["id"] in seen
                or row.get("mode") != "performance"
                or not row.get("valid")
            ):
                continue
            seen.add(row["id"])
            config = row["configuration"]
            writer.writerow(
                [
                    row["id"].rsplit("_s", 1)[0],
                    row["id"],
                    row["case"]["name"],
                    str(row["case"]["balanced"]).lower(),
                    config["revision"],
                    config["container"],
                    config["pin"],
                    config["apre"],
                    *[row["metrics"][field] for field in FIELDS],
                    row["raw_log"],
                    row["performance_validity_basis"],
                ]
            )


def main():
    parser = argparse.ArgumentParser(
        description="从 host 串行运行剩余测试，默认先测三个新增 Non-balanced 规模，再完成原测试；GPU 可用性由用户保证。"
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--plan-only",
        action="store_true",
        help="仅显示测试计划，不运行 docker 或 GPU 程序",
    )
    modes.add_argument(
        "--resume",
        type=Path,
        help="继续同 boot、同源码的中断 run；不重跑已有失败 worker",
    )
    modes.add_argument(
        "--summarize", type=Path, help="仅从已有日志生成汇总，不运行测试"
    )
    modes.add_argument(
        "--finish-best-pin",
        type=Path,
        help="保留同 boot/源码的已完成九 T 和旧表，只补撤销 main pin1 后的最后对比",
    )
    parser.add_argument(
        "--non-balanced-only",
        action="store_true",
        help="单独测试 E96/T16384、E1/T16384、E96/T512，全部为 Non-balanced",
    )
    args = parser.parse_args()
    if args.non_balanced_only and args.finish_best_pin:
        parser.error("--non-balanced-only 不能与 --finish-best-pin 一起使用。")
    if args.plan_only:
        print(
            json.dumps(make_plan(args.non_balanced_only), indent=2, ensure_ascii=False)
        )
        return 0
    if args.summarize:
        directory = args.summarize.resolve()
        state = read(directory / "result.json")
        if args.non_balanced_only and not is_non_balanced_run(state):
            parser.error("指定的日志不是 Non-balanced 单独运行，不能改变其测试范围。")
        archive = {}
        if not is_non_balanced_run(state):
            archive_path = directory / "inputs/fixed_extended_archive.json"
            if digest(archive_path) != ARCHIVE_SHA:
                raise RuntimeError("日志中的固定通用扩展归档 SHA256 不匹配。")
            archive = read(archive_path)
        assess_results(state, archive)
        # Derived reports are separate; never rewrite the original run state.
        export_results(directory, state)
        print(f"已生成汇总：{directory / 'summary.md'}")
        return 0
    if (
        os.name != "posix"
        or Path("/.dockerenv").exists()
        or "a07-3" not in socket.gethostname()
    ):
        parser.error(
            "请在 a07-3 的 host 上运行；本地仅使用 --plan-only 或 --summarize。"
        )
    non_balanced_only = args.non_balanced_only
    if args.resume:
        recorded_mode = is_non_balanced_run(read(args.resume / "result.json"))
        if args.non_balanced_only and not recorded_mode:
            parser.error("指定的旧 run 不是 Non-balanced 单独运行，不能混用。")
        non_balanced_only = recorded_mode
    directory = args.resume or LOG_ROOT / (
        datetime.now(timezone.utc).astimezone().strftime("%Y%m%d_%H%M%S")
        + ("_non_balanced_" if non_balanced_only else "_")
        + uuid.uuid4().hex[:8]
    )
    runner = Runner(
        directory,
        resume=bool(args.resume),
        non_balanced_only=non_balanced_only,
    )
    try:
        if args.finish_best_pin:
            runner.finish_previous_best_pin(args.finish_best_pin)
        elif non_balanced_only:
            runner.run_non_balanced()
        else:
            runner.run()
        return 0 if runner.state["acceptance"]["passed"] else 2
    # Persist diagnostics even for an unexpected failure at the CLI boundary.
    except BaseException as error:  # noqa: BLE001
        runner.state.update(status="failed", error=repr(error))
        assess_results(runner.state, runner.archive)
        runner.save()
        export_results(runner.directory, runner.state)
        runner.announce(f"测试停止，保留全部日志：{error}")
        return 130 if isinstance(error, KeyboardInterrupt) else 1


if __name__ == "__main__":
    sys.exit(main())
