#!/usr/bin/env python3
"""Run FAV3 flash-attention benchmarks for CI.

This script moves the shell loop logic from workflow YAML into Python so it is
easier to maintain and closer to the benchmark script style used in
`op_tests/op_benchmarks/triton`.
"""

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


FA_CONFIGS = [("1", "bf16"), ("0", "bf16"), ("1", "fp16"), ("0", "fp16")]


def log(msg: str) -> None:
    print(msg, flush=True)


def run_and_tee(cmd: List[str], cwd: Path, log_file: Path) -> int:
    """Run command, stream output to stdout and a log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log("Command: {}".format(" ".join(shlex.quote(x) for x in cmd)))

    with log_file.open("w", encoding="utf-8") as fp:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            fp.write(line)
        proc.wait()
        return proc.returncode


def run_fa_benchmarks(
    bench_dir: Path,
    workspace: Path,
    suffix: str,
    python_bin: str,
) -> int:
    fa_script = bench_dir / "fa" / "flash-attention.py"
    if not fa_script.is_file():
        log("WARNING: fa/flash-attention.py not found, skipping FAV3 benchmarks")
        return 0

    for causal, dtype in FA_CONFIGS:
        log_file = workspace / "fa_{}_causal{}_{}.log".format(dtype, causal, suffix)
        cmd = [
            python_bin,
            str(fa_script),
            "-causal={}".format(causal),
            "-dtype={}".format(dtype),
            "-branch={}".format(suffix),
        ]
        ret = run_and_tee(cmd, bench_dir, log_file)
        if ret != 0:
            return ret
    return 0


def clear_git_url_rewrite() -> None:
    """Remove github url rewrite entries to avoid embedded token detection."""
    list_cmd = [
        "git",
        "config",
        "--global",
        "--get-regexp",
        r"url\..*github\.com.*\.insteadOf",
    ]
    listed = subprocess.run(
        list_cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        universal_newlines=True,
    )

    for line in listed.stdout.splitlines():
        key = line.split(maxsplit=1)[0].strip()
        if not key:
            continue
        normalized = key[:-10] + ".insteadOf" if key.endswith(".insteadof") else key
        subprocess.run(
            ["git", "config", "--global", "--unset", normalized],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )


def run_iglp10_setup(script_dir: Path, python_bin: str) -> int:
    """Install iglp_10 optimized triton path used by FAV3."""
    installer = script_dir / "fav3_tmp_test.py"
    if not installer.is_file():
        log("WARNING: fav3_tmp_test.py not found, skipping iglp_10 benchmarks")
        return 0

    log("=== Installing iglp_10 optimized triton ===")
    subprocess.run(
        [python_bin, "-m", "pip", "uninstall", "-y", "triton"],
        check=False,
    )

    log("=== Clearing git URL rewrite config for LLVM build ===")
    clear_git_url_rewrite()

    return subprocess.call([python_bin, str(installer)])


def merge_fa_results(
    script_dir: Path,
    workspace: Path,
    bench_dir: Path,
    python_bin: str,
) -> int:
    merge_script = script_dir / "merge_fa_results.py"
    if merge_script.is_file():
        cmd = [
            python_bin,
            str(merge_script),
            "--workspace",
            str(workspace),
            "-o",
            str(bench_dir),
        ]
        log("Command: {}".format(" ".join(shlex.quote(x) for x in cmd)))
        ret = subprocess.call(cmd, cwd=str(bench_dir))
        if ret != 0:
            return ret

    copied = 0
    for csv_file in bench_dir.glob("bench_fused-attention-*.csv"):
        dst = workspace / csv_file.name
        shutil.copy2(csv_file, dst)
        copied += 1
    if copied == 0:
        log("No FA CSV files")
    return 0


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="bench_fav3.py",
        description="Run FAV3 flash-attention benchmark matrix",
    )
    parser.add_argument(
        "--workspace",
        default=".",
        help="Workspace path for benchmark logs/artifacts",
    )
    parser.add_argument(
        "--bench-dir",
        default=str(Path(__file__).resolve().parent),
        help="Benchmark directory containing fa/flash-attention.py (default: bench_fav3.py directory)",
    )
    parser.add_argument(
        "--script-dir",
        default=".",
        help="Directory containing helper scripts (fav3_tmp_test.py, merge_fa_results.py)",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable to run child scripts",
    )
    parser.add_argument(
        "--run-iglp10",
        action="store_true",
        help="Also run iglp_10 branch benchmark comparison",
    )
    parser.add_argument(
        "--suffix",
        choices=["upstream", "iglp_10"],
        default="upstream",
        help="Run FA benchmark matrix for one suffix only (skip setup and merge)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    workspace = Path(args.workspace).resolve()
    bench_dir = Path(args.bench_dir).resolve()
    script_dir = Path(args.script_dir).resolve()
    python_bin = args.python_bin

    if args.suffix is not None:
        return run_fa_benchmarks(
            bench_dir=bench_dir,
            workspace=workspace,
            suffix=args.suffix,
            python_bin=python_bin,
        )

    log("=== Running Flash Attention with upstream triton ===")
    ret = run_fa_benchmarks(
        bench_dir=bench_dir,
        workspace=workspace,
        suffix="upstream",
        python_bin=python_bin,
    )
    if ret != 0:
        return ret

    if args.run_iglp10:
        ret = run_iglp10_setup(script_dir=script_dir, python_bin=python_bin)
        if ret != 0:
            return ret
        log("=== Running Flash Attention with iglp_10 triton ===")
        ret = run_fa_benchmarks(
            bench_dir=bench_dir,
            workspace=workspace,
            suffix="iglp_10",
            python_bin=python_bin,
        )
        if ret != 0:
            return ret
    else:
        log("=== Skipping iglp_10 benchmark (temporarily disabled) ===")

    log("=== Preparing benchmark results ===")
    return merge_fa_results(
        script_dir=script_dir,
        workspace=workspace,
        bench_dir=bench_dir,
        python_bin=python_bin,
    )


if __name__ == "__main__":
    sys.exit(main())

