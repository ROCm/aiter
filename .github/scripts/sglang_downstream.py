#!/usr/bin/env python3
"""Control SGLang downstream test selection, patching, and model setup."""

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
        "enabled": True,
        "run_on_pr": True,
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
        "enabled": True,
        "run_on_pr": False,
        "pr_disabled_reason": "Standalone performance job runs outside PR test.",
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
        "enabled": False,
        "disabled_reason": "Disabled due to https://github.com/ROCm/aiter/issues/2857",
        "run_on_pr": True,
    },
    {
        "runner": "linux-aiter-mi35x-8",
        "label": "MI35X",
        "model": "Qwen 3.5 Series",
        "model_id": "Qwen/Qwen3.5-397B-A17B",
        "model_path_env": "QWEN35_MODEL_PATH",
        "extra_model_id": "Qwen/Qwen3.5-397B-A17B-FP8",
        "extra_model_path_env": "QWEN35_FP8_MODEL_PATH",
        "test_type": "Accuracy + FP8 Performance",
        "timeout_minutes": 180,
        "extra_exec_args": "",
        "test_command": "python3 run_suite.py --hw amd --suite nightly-amd-accuracy-8-gpu-mi35x-qwen35 --nightly --timeout-per-file 3600",
        "second_test_type": "FP8 Performance",
        "second_extra_exec_args": "-e SGLANG_USE_AITER=1",
        "second_test_command": "python3 run_suite.py --hw amd --suite nightly-perf-8-gpu-mi35x-qwen35-fp8 --nightly --timeout-per-file 5400",
        "enabled": True,
        "run_on_pr": True,
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
        "enabled": False,
        "disabled_reason": "Disabled due to https://github.com/ROCm/aiter/issues/2792",
        "run_on_pr": True,
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
        "enabled": False,
        "disabled_reason": "Disabled due to https://github.com/ROCm/aiter/issues/2792",
        "run_on_pr": True,
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
        "path": "scripts/ci/amd/amd_ci_start_container.sh",
        "old": "-e HF_HOME=/sgl-data/hf-cache",
        "new": "-e HF_HOME=/models/huggingface",
    },
    {
        "path": "test/registered/amd/accuracy/mi35x/test_qwen35_eval_mi35x.py",
        "old": 'QWEN35_MODEL_PATH = "Qwen/Qwen3.5-397B-A17B"',
        "new": 'QWEN35_MODEL_PATH = os.environ.get("QWEN35_MODEL_PATH", "Qwen/Qwen3.5-397B-A17B")',
    },
    {
        "path": "test/registered/amd/test_qwen3_instruct_mxfp4.py",
        "old": 'QWEN3_MODEL_PATH = "amd/Qwen3-235B-A22B-Instruct-2507-mxfp4"',
        "new": 'QWEN3_MODEL_PATH = os.environ.get("QWEN3_MODEL_PATH", "amd/Qwen3-235B-A22B-Instruct-2507-mxfp4")',
    },
    {
        "path": "test/registered/amd/accuracy/mi35x/test_deepseek_v32_eval_mi35x.py",
        "old": 'model_path="deepseek-ai/DeepSeek-V3.2",',
        "new": 'model_path=os.environ.get("DEEPSEEK_V32_MODEL_PATH", "deepseek-ai/DeepSeek-V3.2"),',
    },
]


PREPARE_MODEL_SCRIPT = r"""
set -ex
if [ -f "${MODEL_DIR}/.complete" ] || [ -f "${MODEL_DIR}/config.json" ]; then
  echo "Model already cached at ${MODEL_DIR}"
else
  echo "Model not found at ${MODEL_DIR}, downloading..."
  mkdir -p "$(dirname "${MODEL_DIR}")" /models/.tmp
  TMP_MODEL_DIR="/models/.tmp/${MODEL_ID//\//__}-$(date +%s%N)-${RANDOM}-$$"
  rm -rf "${TMP_MODEL_DIR}"
  huggingface-cli download "${MODEL_ID}" --local-dir "${TMP_MODEL_DIR}"
  test -f "${TMP_MODEL_DIR}/config.json"
  echo "ok" > "${TMP_MODEL_DIR}/.complete"

  if [ -f "${MODEL_DIR}/.complete" ] || [ -f "${MODEL_DIR}/config.json" ]; then
    echo "Model was cached by another runner at ${MODEL_DIR}"
    rm -rf "${TMP_MODEL_DIR}"
  elif mv -T "${TMP_MODEL_DIR}" "${MODEL_DIR}"; then
    echo "Published model at ${MODEL_DIR}"
  else
    echo "Failed to publish ${MODEL_DIR}; checking if another runner published it"
    if rmdir "${MODEL_DIR}" 2>/dev/null && mv -T "${TMP_MODEL_DIR}" "${MODEL_DIR}"; then
      echo "Published model at ${MODEL_DIR} after removing empty target directory"
    else
      rm -rf "${TMP_MODEL_DIR}"
      test -f "${MODEL_DIR}/.complete" -o -f "${MODEL_DIR}/config.json"
    fi
  fi
fi
echo "Model ready at ${MODEL_DIR}"
ls -la "${MODEL_DIR}"
"""


def write_output(name: str, value: str) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"{name}={value}\n")


def write_summary(
    selected: list[dict], pr_skipped: list[dict], disabled: list[dict]
) -> None:
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as summary:
        summary.write("## SGLang Downstream Test Selection\n\n")
        summary.write(f"- Selected tests: `{len(selected)}`\n")
        summary.write(f"- Globally disabled tests: `{len(disabled)}`\n")
        summary.write(f"- PR-skipped enabled tests: `{len(pr_skipped)}`\n\n")
        summary.write("| Model | Test | Enabled | PR enabled |\n")
        summary.write("| --- | --- | --- | --- |\n")
        for test in TESTS:
            summary.write(
                f"| {test['model']} | {test['test_type']} | {'yes' if test['enabled'] else 'no'} | {'yes' if test['run_on_pr'] else 'no'} |\n"
            )

        if disabled:
            summary.write("\n### Globally Disabled\n\n")
            for test in disabled:
                summary.write(
                    f"- {test['model']} / {test['test_type']}: {test['disabled_reason']}\n"
                )

        if pr_skipped:
            summary.write("\n### Skipped In PR\n\n")
            for test in pr_skipped:
                summary.write(
                    f"- {test['model']} / {test['test_type']}: {test.get('pr_disabled_reason', 'PR disabled')}\n"
                )


def select_tests() -> None:
    is_pr = os.environ.get("EVENT_NAME") == "pull_request"
    enabled = [test for test in TESTS if test["enabled"]]
    disabled = [test for test in TESTS if not test["enabled"]]
    selected = [test for test in enabled if not is_pr or test["run_on_pr"]]
    pr_skipped = [test for test in enabled if is_pr and not test["run_on_pr"]]

    write_output("matrix", json.dumps({"include": selected}, separators=(",", ":")))
    write_output("has_tests", "true" if selected else "false")
    write_summary(selected, pr_skipped, disabled)


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


def prepare_model() -> None:
    if len(sys.argv) < 3:
        raise SystemExit(f"Usage: {sys.argv[0]} prepare-model MODEL_ID [MODEL_ID ...]")

    hf_token = os.environ.get("HF_TOKEN", "")
    for model_id in sys.argv[2:]:
        subprocess.run(
            [
                "docker",
                "exec",
                "-e",
                f"HF_TOKEN={hf_token}",
                "-e",
                f"MODEL_ID={model_id}",
                "-e",
                f"MODEL_DIR=/models/{model_id}",
                "ci_sglang",
                "bash",
                "-lc",
                PREPARE_MODEL_SCRIPT,
            ],
            check=True,
        )


def main() -> None:
    if len(sys.argv) == 1 or sys.argv[1] == "select-tests":
        select_tests()
    elif sys.argv[1] == "patch-sglang":
        patch_sglang_checkout()
    elif sys.argv[1] == "prepare-model":
        prepare_model()
    else:
        raise SystemExit(f"Unknown command: {sys.argv[1]}")


if __name__ == "__main__":
    main()
