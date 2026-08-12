#!/usr/bin/env python3

# SPDX-License-Identifier: MIT

"""Fresh-process production dispatch replay for installed decode GEMM rows."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

import torch

import aiter
import aiter.tuned_gemm as tuned_gemm
from aiter.aot.flydsl.gemm import parse_csv
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.kernels import gemm_decode as gemm_decode_module
from aiter.ops.flydsl.kernels.gemm_decode_block_mfma import (
    compile_gemm_decode_block_mfma_bf16,
)
from aiter.ops.flydsl.kernels.gemm_decode_wave import (
    compile_gemm_decode_wave_bf16,
)


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(self.format(record))


def _inside(path: str | Path, root: Path) -> bool:
    return Path(path).resolve().is_relative_to(root.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkout", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("compile-aot", "replay"),
        default="replay",
    )
    args = parser.parse_args()

    checkout = args.checkout.resolve()
    config_path = args.config.resolve()
    cache_dir = args.cache_dir.resolve()
    expected_config = (checkout / "aiter/configs/bf16_tuned_gemm.csv").resolve()
    assert config_path == expected_config
    assert Path(os.environ["AITER_CONFIG_GEMM_BF16"]).resolve() == expected_config
    assert Path(AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE).resolve() == expected_config
    assert Path(tuned_gemm.tune_path).resolve() == expected_config
    assert Path(os.environ["FLYDSL_RUNTIME_CACHE_DIR"]).resolve() == cache_dir

    imported_paths = {
        "aiter": str(Path(aiter.__file__).resolve()),
        "tuned_gemm": str(Path(tuned_gemm.__file__).resolve()),
        "gemm_decode": str(Path(gemm_decode_module.__file__).resolve()),
    }
    assert all(_inside(path, checkout) for path in imported_paths.values())

    with config_path.open(newline="") as stream:
        csv_rows = [
            row
            for row in csv.DictReader(stream)
            if row.get("libtype") == "flydsl_decode"
        ]
    jobs = [job for job in parse_csv(str(config_path)) if job["kind"] == "decode"]
    assert len(jobs) == len(csv_rows)
    assert {job["kernel_name"] for job in jobs} == {
        row["kernelName"] for row in csv_rows
    }

    if args.mode == "compile-aot":
        from aiter.aot.flydsl import common
        from aiter.aot.flydsl.common import OpKind

        assert os.environ.get("FLYDSL_RUNTIME_ENABLE_CACHE") == "1"
        assert os.environ.get("FLYDSL_RUNTIME_RUN_ONLY") != "1"
        package_gemm_jobs = common.collect_aot_jobs_for(OpKind.GEMM)
        package_decode_jobs = [
            job for job in package_gemm_jobs if job.get("kind") == "decode"
        ]
        assert {job["kernel_name"] for job in package_decode_jobs} == {
            job["kernel_name"] for job in jobs
        }
        common.run_aot(str(cache_dir), kinds=(OpKind.GEMM,))
        args.evidence.write_text(
            json.dumps(
                {
                    "checkout": str(checkout),
                    "config_source": str(config_path),
                    "cache_dir": str(cache_dir),
                    "imported_paths": imported_paths,
                    "driver": "aiter.aot.flydsl.common.run_aot",
                    "package_scope": "GEMM",
                    "package_gemm_jobs": len(package_gemm_jobs),
                    "package_decode_jobs": len(package_decode_jobs),
                    "replay_decode_jobs": len(jobs),
                },
                indent=2,
            )
            + "\n"
        )
        print(
            "PRODUCTION_DECODE_AOT_OK "
            f"package_gemm_jobs={len(package_gemm_jobs)} "
            f"package_decode_jobs={len(package_decode_jobs)} "
            f"replay_decode_jobs={len(jobs)}"
        )
        return

    assert os.environ.get("FLYDSL_RUNTIME_RUN_ONLY") == "1"
    assert os.environ.get("FLYDSL_RUNTIME_ENABLE_CACHE") == "0"
    assert os.environ.get("COMPILE_ONLY", "0") != "1"
    assert os.environ.get("FLYDSL_DUMP_IR", "0") != "1"

    runtime_arch = get_gfx_runtime()
    runtime_cu = get_cu_num()
    assert all(job["gfx"] == runtime_arch for job in jobs)
    assert all(job["cu_num"] == runtime_cu for job in jobs)
    assert all(not job["has_bias"] for job in jobs)

    # These are the process-owned caches on the production selection/launcher
    # path. The replay process itself is fresh; clearing them makes that
    # precondition explicit without deleting or mutating package artifacts.
    tuned_gemm.get_GEMM_A16W16_config_.cache_clear()
    tuned_gemm.get_GEMM_A16W16_config.cache_clear()
    compile_gemm_decode_wave_bf16.cache_clear()
    compile_gemm_decode_block_mfma_bf16.cache_clear()

    fallback_calls: list[str] = []
    original_decode_dispatch = tuned_gemm.solMap["flydsl_decode"]

    def reject_fallback(kind):
        def fail(*_args, **_kwargs):
            fallback_calls.append(kind)
            raise AssertionError(f"unexpected GEMM fallback: {kind}")

        return fail

    for kind in tuple(tuned_gemm.solMap):
        if kind != "flydsl_decode":
            tuned_gemm.solMap[kind] = reject_fallback(kind)

    launched_names: list[str] = []
    launchers = []
    expected_name: str | None = None
    original_launch_name = gemm_decode_module.launch_gemm_decode_kernel_name
    original_compile = gemm_decode_module.compile_gemm_decode_bf16

    def observe_launch(A, B, C, kernel_name, *launch_args, **launch_kwargs):
        assert kernel_name == expected_name
        launched_names.append(kernel_name)
        return original_launch_name(
            A,
            B,
            C,
            kernel_name,
            *launch_args,
            **launch_kwargs,
        )

    def observe_compile(*compile_args, **compile_kwargs):
        launcher = original_compile(*compile_args, **compile_kwargs)
        launchers.append(launcher)
        return launcher

    gemm_decode_module.launch_gemm_decode_kernel_name = observe_launch
    gemm_decode_module.compile_gemm_decode_bf16 = observe_compile

    log_capture = _LogCapture()
    logging.getLogger("aiter").addHandler(log_capture)
    evidence_rows = []
    torch.manual_seed(29)
    for job in jobs:
        m, n, k = job["m"], job["n"], job["k"]
        expected_name = job["kernel_name"]
        selected = tuned_gemm.get_GEMM_A16W16_config(
            m,
            n,
            k,
            False,
            "torch.bfloat16",
            "torch.bfloat16",
        )
        assert selected["libtype"] == "flydsl_decode"
        assert selected["kernelName"] == expected_name

        before_launches = len(launched_names)
        before_launchers = len(launchers)
        a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        b = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
        output = tuned_gemm.gemm_a16w16(a, b)
        torch.cuda.synchronize()
        assert len(launched_names) == before_launches + 1
        assert len(launchers) == before_launchers + 1
        assert launched_names[-1] == expected_name

        assert torch.isfinite(output).all()
        reference = (a.float() @ b.float().T).bfloat16()
        torch.testing.assert_close(output, reference, atol=0.125, rtol=0.01)
        evidence_rows.append(
            {
                "M": m,
                "N": n,
                "K": k,
                "bias": False,
                "arch": runtime_arch,
                "cu_num": runtime_cu,
                "selected_libtype": selected["libtype"],
                "selected_kernel": selected["kernelName"],
                "run_only_artifact_loaded": True,
                "finite": True,
                "correct": True,
            }
        )

    fallback_logs = [
        message
        for message in log_capture.messages
        if "fallback" in message.lower()
        or "falling back" in message.lower()
        or "not found" in message.lower()
        or "missing" in message.lower()
    ]
    assert not fallback_calls
    assert not fallback_logs
    assert launched_names == [job["kernel_name"] for job in jobs]

    args.evidence.write_text(
        json.dumps(
            {
                "checkout": str(checkout),
                "config_source": str(config_path),
                "cache_dir": str(cache_dir),
                "imported_paths": imported_paths,
                "runtime_arch": runtime_arch,
                "runtime_cu": runtime_cu,
                "run_only": True,
                "jit_disabled": True,
                "discovered_decode_rows": len(jobs),
                "selected_decode_rows": len(evidence_rows),
                "fallback_calls": fallback_calls,
                "fallback_logs": fallback_logs,
                "rows": evidence_rows,
            },
            indent=2,
        )
        + "\n"
    )
    print(
        f"PRODUCTION_DECODE_REPLAY_OK rows={len(evidence_rows)} "
        f"config={config_path}"
    )


if __name__ == "__main__":
    main()
