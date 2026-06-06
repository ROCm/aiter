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
        "runner": "linux-aiter-do-mi350x-8",
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
        "runner": "linux-aiter-do-mi350x-8",
        "label": "MI35X",
        "model": "DeepSeek-R1-MXFP4",
        "model_id": "amd/DeepSeek-R1-MXFP4-Preview",
        "model_path_env": "DEEPSEEK_R1_MXFP4_MODEL_PATH",
        "test_type": "Performance",
        "timeout_minutes": 180,
        "extra_exec_args": "",
        "test_command": "python3 registered/amd/perf/mi35x/test_deepseek_r1_mxfp4_perf_mi35x.py",
        "run_on_pr": False,
        "comment": "Standalone performance job is too long for PR validation.",
        "run_on_schedule": True,
    },
    {
        "runner": "linux-aiter-do-mi350x-8",
        "label": "MI35X",
        "model": "Qwen3-235B-MXFP4",
        "model_id": "amd/Qwen3-235B-A22B-Instruct-2507-mxfp4",
        "model_path_env": "QWEN3_MODEL_PATH",
        "test_type": "Accuracy + Performance",
        "timeout_minutes": 70,
        "extra_exec_args": "",
        "test_command": "python3 run_suite.py --hw amd --suite nightly-8-gpu-mi35x-qwen3-235b-mxfp4 --nightly --timeout-per-file 3600",
        "run_on_pr": False,
        "run_on_schedule": False,
        "comment": "issue https://github.com/ROCm/aiter/issues/2857 not resolved yet",
    },
    {
        "runner": "linux-aiter-do-mi350x-8",
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
        "runner": "linux-aiter-do-mi350x-8",
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
        "runner": "linux-aiter-do-mi350x-8",
        "label": "MI35X",
        "model": "DeepSeek-V3.2",
        "model_id": "deepseek-ai/DeepSeek-V3.2",
        "model_path_env": "DEEPSEEK_V32_MODEL_PATH",
        "test_type": "Accuracy",
        "timeout_minutes": 70,
        "extra_exec_args": "",
        "test_command": "python3 run_suite.py --hw amd --suite nightly-amd-8-gpu-mi35x-deepseek-v32 --nightly --timeout-per-file 3600",
        # Temporarily disabled: the DSv3.2 indexer eval hangs and hits the 3600s
        # timeout (HIP backtrace) on current AITER main; verified the #3451 fix
        # (cache_kernels.cu) cherry-picked does NOT resolve it yet. Re-enable
        # (run_on_pr/run_on_schedule -> True) once the DSv3.2 indexer kernel fix
        # lands. Tracked in #3451 / dsv32-indexer-fused-kernel-fixes.
        "run_on_pr": False,
        "run_on_schedule": False,
    },
    {
        "runner": "linux-aiter-do-mi350x-8",
        "label": "MI35X",
        "model": "DeepSeek-V3.2 Basic",
        "model_id": "deepseek-ai/DeepSeek-V3.2",
        "model_path_env": "DEEPSEEK_V32_MODEL_PATH",
        "test_type": "Performance",
        "timeout_minutes": 100,
        "extra_exec_args": "",
        "test_command": "python3 run_suite.py --hw amd --suite nightly-perf-8-gpu-mi35x-deepseek-v32-basic --nightly --timeout-per-file 5400",
        "run_on_pr": False,
        "comment": "Standalone performance job is too long for PR validation.",
        "run_on_schedule": True,
    },
    {
        # MoRI expert-parallel accuracy gate. Guards the AITER moe_kernels
        # stage-2 reduce path on the MoRI EP backend, where a small per-rank
        # dispatch buffer silently corrupted decode output (gsm8k -> 0 while
        # output stayed fluent and even ~1.5% faster). Reported by SemiAnalysis
        # / InferenceX, root-caused to AITER. See sgl-project/sglang#27194.
        # Runs SGLang's stage-c MoRI EP suite (test_moriep_small.py): DeepSeek-V3
        # with TP8/EP8/DP8 + dp-attention + --moe-a2a-backend mori +
        # --attention-backend aiter, SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK
        # =128 (the small-buffer regime that exposed the bug), gated on
        # gsm8k >= 0.90. A silent corruption now fails CI loudly.
        #
        # DISABLED pending ROCm/aiter#3579. The mori EP dispatch kernel fails
        # CUDA graph capture with "HIP error 401: hipModuleLaunchKernel" on BOTH
        # dispatch paths — normal/intra-node (EpDispatchIntraNodeKernel_bf16) and
        # low_latency (mori dispatch_combine launch_multi). Weights load fine; the
        # server reaches capture and the mori JIT kernel launch then fails. Only
        # one run ever passed (gsm8k 0.945, TestLowLatency @ 832966dd); later runs
        # fail at capture. Likely a mori build skew in the harness (image's mori
        # vs the AITER under test) or a real capture issue — needs the mori owner.
        # CI plumbing (predownload to /models, env resolution, class scoping) is
        # in place and verified; flip run_on_* back on once #3579 is resolved.
        # test_command kept on TestLowLatency (the closest-to-green class).
        "runner": "linux-aiter-do-mi350x-8",
        "label": "MI35X",
        "model": "DeepSeek-V3 MoRI-EP",
        "model_id": "deepseek-ai/DeepSeek-V3-0324",
        "model_path_env": "DEEPEP_MODEL_PATH",
        "draft_model_id": "lmsys/DeepSeek-V3-NextN",
        "draft_model_path_env": "DEEPEP_NEXTN_MODEL_PATH",
        "test_type": "Accuracy (MoRI EP low-latency)",
        "timeout_minutes": 90,
        "extra_exec_args": "",
        "test_command": "python3 registered/amd/test_moriep_small.py TestLowLatency",
        "run_on_pr": False,
        "run_on_schedule": False,
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
    # Make the shared DeepEP test models (used by the MoRI EP suite) resolve from
    # the /models cache when present, so the predownload step's local copy is
    # used instead of an inline HF download inside the test step. test_utils.py
    # already imports os; defaults are unchanged when the env vars are unset.
    {
        "path": "python/sglang/test/test_utils.py",
        "old": 'DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST = "deepseek-ai/DeepSeek-V3-0324"',
        "new": 'DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST = os.environ.get("DEEPEP_MODEL_PATH", "deepseek-ai/DeepSeek-V3-0324")',
    },
    {
        "path": "python/sglang/test/test_utils.py",
        "old": 'DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST_NEXTN = "lmsys/DeepSeek-V3-NextN"',
        "new": 'DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST_NEXTN = os.environ.get("DEEPEP_NEXTN_MODEL_PATH", "lmsys/DeepSeek-V3-NextN")',
    },
]


def write_output(name: str, value: str) -> None:
    with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as output:
        output.write(f"{name}={value}\n")


def run_cell(test: dict, key: str) -> str:
    if test.get(key, False):
        return "yes"

    comment = test.get("comment")
    if comment:
        return f"no ({comment})"
    return "no"


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
                f"{run_cell(test, 'run_on_pr')} | "
                f"{run_cell(test, 'run_on_schedule')} |\n"
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
    pairs = [
        (test.get("model_path_env"), test.get("model_id")),
        (test.get("draft_model_path_env"), test.get("draft_model_id")),
    ]
    for env_name, model_id in pairs:
        if not env_name or not model_id:
            continue
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
                f"Local model path not readable, using default: {model_id}",
                file=sys.stderr,
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
