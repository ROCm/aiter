#!/usr/bin/env python3
"""Control SGLang downstream test selection, patching, and model resolution."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

TESTS = [
    {
        "runner": "linux-aiter-mi35x-8",
        "label": "MI35X",
        "model": "DeepSeek-R1-MXFP4",
        "model_id": "amd/DeepSeek-R1-MXFP4-Preview",
        "model_path_env": "DEEPSEEK_R1_MXFP4_MODEL_PATH",
        "test_type": "Accuracy",
        "timeout_minutes": 130,
        "extra_exec_args": "",
        "test_command": "python3 run_suite.py --hw amd --suite nightly-amd-8-gpu-mi35x-deepseek-r1-mxfp4 --nightly --timeout-per-file 7200",
        "run_on_pr": True,
        "run_on_schedule": True,
    },
    {
        "runner": "linux-aiter-mi35x-8",
        "label": "MI35X",
        "model": "DeepSeek-R1-MXFP4",
        "model_id": "amd/DeepSeek-R1-MXFP4-Preview",
        "model_path_env": "DEEPSEEK_R1_MXFP4_MODEL_PATH",
        "test_type": "Performance",
        "timeout_minutes": 180,
        "extra_exec_args": "",
        "test_command": "python3 registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py",
        "run_on_pr": False,
        "run_on_schedule": True,
    },
    {
        "runner": "linux-aiter-mi35x-8",
        "label": "MI35X",
        "model": "Qwen3-235B-MXFP4",
        "model_id": "amd/Qwen3-235B-A22B-Instruct-2507-mxfp4",
        "model_path_env": "QWEN3_MODEL_PATH",
        "test_type": "Accuracy + Performance",
        "timeout_minutes": 70,
        "extra_exec_args": "",
        "test_command": "python3 run_suite.py --hw amd --suite nightly-8-gpu-mi35x-qwen3-235b-mxfp4 --nightly --timeout-per-file 3600",
        "run_on_pr": True,
        "run_on_schedule": True,
    },
    {
        "runner": "linux-aiter-mi35x-8",
        "label": "MI35X",
        "model": "Qwen 3.5",
        "model_id": "Qwen/Qwen3.5-397B-A17B",
        "model_path_env": "QWEN35_MODEL_PATH",
        "test_type": "Accuracy",
        "timeout_minutes": 70,
        "extra_exec_args": "",
        "test_command": "python3 run_suite.py --hw amd --suite nightly-amd-accuracy-8-gpu-mi35x-qwen35 --nightly --timeout-per-file 3600",
        "run_on_pr": True,
        "run_on_schedule": True,
    },
    {
        "runner": "linux-aiter-mi35x-8",
        "label": "MI35X",
        "model": "Qwen 3.5 FP8",
        "model_id": "Qwen/Qwen3.5-397B-A17B-FP8",
        "model_path_env": "QWEN35_FP8_MODEL_PATH",
        "test_type": "Performance",
        "timeout_minutes": 100,
        "extra_exec_args": "-e SGLANG_USE_AITER=1",
        "test_command": "python3 run_suite.py --hw amd --suite nightly-perf-8-gpu-mi35x-qwen35-fp8 --nightly --timeout-per-file 5400",
        "run_on_pr": False,
        "run_on_schedule": True,
    },
    {
        "runner": "linux-aiter-mi35x-8",
        "label": "MI35X",
        "model": "DeepSeek-V3.2",
        "model_id": "deepseek-ai/DeepSeek-V3.2",
        "model_path_env": "DEEPSEEK_V32_MODEL_PATH",
        "test_type": "Accuracy",
        "timeout_minutes": 70,
        "extra_exec_args": "",
        "test_command": "python3 run_suite.py --hw amd --suite nightly-amd-8-gpu-mi35x-deepseek-v32 --nightly --timeout-per-file 3600",
        "run_on_pr": True,
        "run_on_schedule": True,
    },
    {
        "runner": "linux-aiter-mi35x-8",
        "label": "MI35X",
        "model": "DeepSeek-V3.2 Basic",
        "model_id": "deepseek-ai/DeepSeek-V3.2",
        "model_path_env": "DEEPSEEK_V32_MODEL_PATH",
        "test_type": "Performance",
        "timeout_minutes": 100,
        "extra_exec_args": "",
        "test_command": "python3 run_suite.py --hw amd --suite nightly-perf-8-gpu-mi35x-deepseek-v32-basic --nightly --timeout-per-file 5400",
        "run_on_pr": False,
        "run_on_schedule": True,
    },
]


SGLANG_CI_PATCHES = [
    {
        "path": "scripts/ci/amd/amd_ci_start_container.sh",
        "old": "HOSTNAME_VALUE=$(hostname)",
        "new": 'HOSTNAME_VALUE="${SGLANG_CI_HOSTNAME_OVERRIDE:-$(hostname)}"',
    },
    {
        "path": "scripts/ci/amd/amd_ci_install_dependency.sh",
        "old": "HOSTNAME_VALUE=$(hostname)",
        "new": 'HOSTNAME_VALUE="${SGLANG_CI_HOSTNAME_OVERRIDE:-$(hostname)}"',
    },
    {
        "path": "scripts/ci/amd/amd_ci_exec.sh",
        "old": "HOSTNAME_VALUE=$(hostname)",
        "new": 'HOSTNAME_VALUE="${SGLANG_CI_HOSTNAME_OVERRIDE:-$(hostname)}"',
    },
    {
        "path": "scripts/ci/amd/amd_ci_install_dependency.sh",
        "old": "install_with_retry docker exec -w /human-eval ci_sglang pip install --cache-dir=/sgl-data/pip-cache -e .",
        "new": "install_with_retry docker exec -w /human-eval ci_sglang pip install --cache-dir=/sgl-data/pip-cache --no-build-isolation -e .",
    },
    {
        "path": "scripts/ci/amd/amd_ci_start_container.sh",
        "old": "$CACHE_VOLUME \\",
        "new": "$CACHE_VOLUME \\\n  -v /models:/models \\",
    },
    {
        "path": "test/registered/amd/test_qwen3_instruct_mxfp4.py",
        "old": 'QWEN3_MODEL_PATH = "amd/Qwen3-235B-A22B-Instruct-2507-mxfp4"',
        "new": 'QWEN3_MODEL_PATH = os.environ.get("QWEN3_MODEL_PATH", "amd/Qwen3-235B-A22B-Instruct-2507-mxfp4")',
    },
    {
        "path": "test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py",
        "old": 'QWEN35_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B"',
        "new": 'QWEN35_MODEL_PATH = os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-397B-A17B")',
    },
    {
        "path": "test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py",
        "old": 'model_path="deepseek-ai/DeepSeek-V3.2",',
        "new": 'model_path=os.environ.get("DEEPSEEK_V32_MODEL_PATH", "deepseek-ai/DeepSeek-V3.2"),',
    },
]


def write_output(name: str, value: str) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"{name}={value}\n")


def write_summary(
    selected: list[dict], skipped: list[dict], disabled: list[dict], event_name: str
) -> None:
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return

    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write("## SGLang Downstream Test Selection\n\n")
        summary.write(f"- Event: `{event_name}`\n")
        summary.write(f"- Selected tests: `{len(selected)}`\n")
        summary.write(f"- Disabled tests: `{len(disabled)}`\n")
        summary.write(f"- Event-skipped tests: `{len(skipped)}`\n\n")
        summary.write("| Model | Test | Run on PR | Run on schedule |\n")
        summary.write("| --- | --- | --- | --- |\n")
        for test in TESTS:
            summary.write(
                f"| {test['model']} | {test['test_type']} | "
                f"{'yes' if test.get('run_on_pr', False) else 'no'} | "
                f"{'yes' if test.get('run_on_schedule', False) else 'no'} |\n"
            )


def select_tests() -> None:
    event_name = os.environ.get("EVENT_NAME") or os.environ.get("GITHUB_EVENT_NAME", "")
    run_key = "run_on_pr" if event_name == "pull_request" else "run_on_schedule"
    disabled = [
        test
        for test in TESTS
        if not test.get("run_on_pr", False) and not test.get("run_on_schedule", False)
    ]
    runnable = [test for test in TESTS if test not in disabled]
    selected = [test for test in runnable if test.get(run_key, False)]
    skipped = [test for test in runnable if not test.get(run_key, False)]

    write_output("matrix", json.dumps({"include": selected}, separators=(",", ":")))
    write_output("has_tests", "true" if selected else "false")
    write_summary(selected, skipped, disabled, event_name or "unknown")


def replace_once(root: Path, patch: dict[str, str]) -> None:
    path = root / patch["path"]
    text = path.read_text()
    if patch["old"] not in text:
        raise SystemExit(f"Expected snippet not found in {path}: {patch['old']!r}")
    path.write_text(text.replace(patch["old"], patch["new"], 1))


def patch_sglang_checkout() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(f"Usage: {sys.argv[0]} patch-sglang SGLANG_WORKSPACE")

    root = Path(sys.argv[2])
    for patch in SGLANG_CI_PATCHES:
        replace_once(root, patch)


def model_env_args() -> None:
    test = json.loads(os.environ["TEST_SPEC"])
    env_name = test.get("model_path_env")
    model_id = test.get("model_id")
    if not env_name or not model_id:
        return

    model_dir = f"/models/{model_id}"
    result = subprocess.run(
        ["docker", "exec", "ci_sglang", "test", "-r", f"{model_dir}/config.json"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode == 0:
        print(f"Using local model path: {model_dir}", file=sys.stderr)
        print("-e")
        print(f"{env_name}={model_dir}")
    else:
        print(
            f"Local model path not readable, using default: {model_id}", file=sys.stderr
        )


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1] == "select-tests":
        select_tests()
    elif sys.argv[1] == "patch-sglang":
        patch_sglang_checkout()
    elif sys.argv[1] == "model-env-args":
        model_env_args()
    else:
        raise SystemExit(f"Unknown command: {sys.argv[1]}")


if __name__ == "__main__":
    main()
