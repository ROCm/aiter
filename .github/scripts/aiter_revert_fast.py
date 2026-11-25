#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
CONTAINER = f"aiter_revert_fast_{os.environ.get('GITHUB_RUN_ID', 'local')}"


def run(
    args: list[str], *, check: bool = True, timeout: int | None = None
) -> subprocess.CompletedProcess[str]:
    print("+ " + " ".join(shlex.quote(arg) for arg in args), flush=True)
    proc = subprocess.run(args, cwd=ROOT, text=True, timeout=timeout, check=False)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def docker_exec(
    command: str, *, extra_env: dict[str, str] | None = None, timeout: int | None = None
) -> None:
    args = ["docker", "exec", "-w", "/workspace"]
    for key, value in (extra_env or {}).items():
        args.extend(["-e", f"{key}={value}"])
    args.extend([CONTAINER, "bash", "-lc", command])
    run(args, timeout=timeout)


def split_test_files(raw: str) -> list[str]:
    return [token.strip() for token in re.split(r"[\s,]+", raw) if token.strip()]


def group_id(name: str) -> str | None:
    proc = subprocess.run(
        ["getent", "group", name],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    return proc.stdout.split(":", 3)[2]


def render_device_args() -> list[str]:
    podinfo = Path("/etc/podinfo/gha-render-devices")
    if podinfo.exists():
        return shlex.split(podinfo.read_text(encoding="utf-8").strip())
    return ["--device", "/dev/dri"]


def write_summary(lines: list[str]) -> None:
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary:
        return
    with open(summary, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")


def ensure_log_header(test_files: list[str], mori_smoke: bool) -> None:
    with open(ROOT / "latest_test.log", "a", encoding="utf-8") as log:
        log.write("Aiter revert-fast focused validation\n")
        log.write(f"ref={os.environ.get('GITHUB_SHA', '')}\n")
        log.write(f"failed_run_id={os.environ.get('FAILED_RUN_ID', '')}\n")
        log.write(f"mori_smoke={mori_smoke}\n")
        log.write(f"test_files={' '.join(test_files)}\n")
        log.write("\n")


def main() -> int:
    docker_image = os.environ.get("DOCKER_IMAGE", "").strip()
    if not docker_image:
        raise SystemExit("DOCKER_IMAGE is not set")

    mori_smoke = os.environ.get("MORI_SMOKE", "").lower() == "true"
    test_files = split_test_files(os.environ.get("TEST_FILES", ""))
    if not mori_smoke and not test_files:
        raise SystemExit("Either TEST_FILES must be set or MORI_SMOKE must be true")

    ensure_log_header(test_files, mori_smoke)

    if run(["docker", "image", "inspect", docker_image], check=False).returncode != 0:
        run(
            ["bash", ".github/scripts/docker_pull_with_retry.sh", docker_image],
            timeout=1800,
        )

    run(["docker", "rm", "-f", CONTAINER], check=False)
    docker_run = [
        "docker",
        "run",
        "-dt",
        "--device=/dev/kfd",
        *render_device_args(),
        "--shm-size=16G",
        "--network=host",
        "-e",
        "TRITON_WHEEL_DIR=/workspace/triton_wheels",
        "-v",
        f"{ROOT}:/workspace",
        "-w",
        "/workspace",
        "--name",
        CONTAINER,
    ]
    for group in ("render", "video"):
        gid = group_id(group)
        if gid:
            docker_run.extend(["--group-add", gid])
    docker_run.append(docker_image)
    run(docker_run)

    docker_exec(
        """
        set -euo pipefail
        pip uninstall -y amd-aiter aiter || true
        pip install -r requirements.txt
        pip install --upgrade pandas pyzmq einops numpy==1.26.2
        pip install --upgrade 'pybind11>=3.0.1'
        pip install --upgrade 'ninja>=1.11.1'
        pip install tabulate
        PREBUILD_KERNELS=0 pip install -e . --no-build-isolation
        pip show amd-aiter || pip show aiter
        ./.github/scripts/install_triton.sh
        pip show triton || true
        """,
        timeout=3600,
    )

    if mori_smoke:
        docker_exec(
            "pip install 'amd-mori==1.2.5' && "
            "set -o pipefail; timeout 25m python3 op_tests/multigpu_tests/test_mori_smoke.py 2>&1 | tee -a latest_test.log",
            timeout=2100,
        )
    if test_files:
        docker_exec(
            "set -o pipefail; "
            "./.github/scripts/aiter_test.sh 2>&1 | tee -a latest_test.log",
            extra_env={"AITER_TEST": " ".join(test_files)},
            timeout=7200,
        )

    write_summary(
        [
            "## Aiter Revert-Fast Validation",
            "",
            "- Result: passed",
            f"- Failed run id: {os.environ.get('FAILED_RUN_ID') or '-'}",
            f"- Mori smoke: {str(mori_smoke).lower()}",
            f"- Test files: {' '.join(test_files) if test_files else '-'}",
            "",
            "This is focused validation and is not a full Aiter Test replacement.",
        ]
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.TimeoutExpired as exc:
        print(f"Timed out: {exc}", file=sys.stderr)
        raise SystemExit(124)
