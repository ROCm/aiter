# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The process tune_gemm.py runs under rocprofv3. Do not run it by hand.

It runs one GEMM at one shape: first with the config the library resolves,
then with each candidate config in the spec. After every config it launches a
`split_dummy` kernel, so the kernel trace splits into one segment per config:
segment 0 is the current config, then one per candidate. Each config also gets
a line in the status file: `ok`, `skipped`, or `error: <what went wrong>`.
tune_gemm.py matches the two up.
"""

import json
import math
import os
import sys

import torch
import triton
import triton.language as tl
from gemm_cases import CASES, backend_kwarg
from triton.testing import runtime

from aiter.ops.triton.utils import gemm_config_utils

# Runs per config; the median kernel time over them is what counts.
N_RUNS = 250
# Largest relative difference allowed between a candidate's output and the
# current config's. Tile sizes and split-K change the summation order, not the
# result, so a correct config stays far below this.
MAX_OUTPUT_ERROR = 0.05
GPU_FAULTS = ("illegal memory access", "HIP error", "hipError", "device-side assert")


@triton.jit
def split_dummy(d_ptr):
    pid = tl.program_id(axis=0)
    x = tl.load(d_ptr + pid)
    tl.store(d_ptr + pid, x + 1)


@triton.jit
def run_dummy(d_ptr):
    x = tl.load(d_ptr)
    tl.store(d_ptr, x + 1)


class LookupHook:
    """Stands in for gemm_config_utils._get_gemm_config_cached, which every
    get_gemm_config() call goes through. Records what the op asks for and the
    config it gets (to a file, so it survives a crash) and, while `override`
    is set, hands that config back instead of reading JSON."""

    def __init__(self, record_path):
        self.real = gemm_config_utils._get_gemm_config_cached
        self.record_path = record_path
        self.override = None
        self.lookups = []
        gemm_config_utils._get_gemm_config_cached = self

    def __call__(
        self,
        config_name,
        M,
        N=None,
        K=None,
        bounds=None,
        specialized_filename=None,
        backend="triton",
        B=None,
    ):
        lookup = {
            "config_name": config_name,
            "M": M,
            "N": N,
            "K": K,
            "B": B,
            "bounds": bounds,
            "specialized_filename": specialized_filename,
            "backend": backend,
        }
        if self.override is not None and self.lookups and lookup not in self.lookups:
            raise ValueError("candidate changed the GEMM config lookup")
        if lookup not in self.lookups:
            self.lookups.append(lookup)
        if self.override is not None:
            return dict(self.override), True
        self.record(None)  # the lookup is known even if resolving it fails
        config, is_tuned = self.real(
            config_name, M, N, K, bounds, specialized_filename, backend, B
        )
        self.record(config)
        return config, is_tuned

    def record(self, config):
        with open(self.record_path, "w") as f:
            json.dump({"lookups": self.lookups, "config": config}, f)


def profile(run):
    """Run the op N_RUNS times with a cold L2 before each run."""
    device = runtime.driver.active.get_device_interface()
    cache = runtime.driver.active.get_empty_cache_for_benchmark()
    marker = torch.zeros(1, dtype=torch.float32, device="cuda")
    run_dummy[(1,)](marker)
    for _ in range(N_RUNS):
        cache.zero_()
        device.synchronize()
        run()
        device.synchronize()
        run_dummy[(1,)](marker)


def mark():
    """Launch split_dummy, which ends the current segment of the kernel trace."""
    torch.cuda.synchronize()
    split_dummy[(128,)](torch.empty(128, dtype=torch.float32, device="cuda"))
    torch.cuda.synchronize()


def output_of(result):
    """Snapshot every returned tensor, including reused float32 output buffers."""
    if isinstance(result, torch.Tensor):
        return [result.detach().clone()]
    if isinstance(result, (tuple, list)):
        return [out for item in result for out in output_of(item)]
    if isinstance(result, dict):
        return [out for item in result.values() for out in output_of(item)]
    return []


def check_output(out, ref):
    """Check all outputs against a successful, finite baseline snapshot."""
    if not ref:
        raise ValueError("no tensor output from a successful current config to check")
    if len(out) != len(ref):
        raise ValueError(f"output tensor count {len(out)} != {len(ref)}")
    for index, (actual, expected) in enumerate(zip(out, ref)):
        if actual.dtype != expected.dtype:
            raise ValueError(f"output {index} dtype {actual.dtype} != {expected.dtype}")
        if actual.shape != expected.shape:
            raise ValueError(
                f"output {index} shape {tuple(actual.shape)} != {tuple(expected.shape)}"
            )
        if not actual.is_floating_point():
            if not torch.equal(actual, expected):
                raise ValueError(
                    f"output {index} integer values differ from the current config's"
                )
            continue
        # isfinite is not implemented for every FP8 dtype. Convert only after
        # checking the original dtype and handling integer scale bytes.
        actual, expected = actual.float(), expected.float()
        if not torch.isfinite(expected).all():
            raise ValueError(f"reference output {index} has NaN or Inf")
        if not torch.isfinite(actual).all():
            raise ValueError(f"output {index} has NaN or Inf")
        error = ((actual - expected).norm() / expected.norm().clamp_min(1e-12)).item()
        if not math.isfinite(error) or error > MAX_OUTPUT_ERROR:
            raise ValueError(
                f"output {index} differs from the current config's by {error:.1%}"
            )


def one_line(error):
    lines = str(error).strip().splitlines() or [""]
    text = lines[0] if len(lines) == 1 else f"{lines[0]} ... {lines[-1]}"
    return f"{type(error).__name__}: {text}"


def main(spec_path):
    with open(spec_path) as f:
        spec = json.load(f)

    def note(line):
        with open(spec["status"], "a") as f:
            f.write(line + "\n")

    case = CASES[spec["op"]]
    run = case(**spec["dims"], **backend_kwarg(spec["backend"]))
    hook = LookupHook(spec["record"])
    note("ready")

    # The current config. Its output is what the candidates are checked against.
    ref = None
    if spec["run_current"]:
        try:
            current_output = output_of(run())
            torch.cuda.synchronize()
            if spec["check"]:
                check_output(current_output, current_output)
            profile(run)
            ref = current_output
            note("ok")
        except Exception as error:  # noqa: BLE001 -- report failure to the parent
            text = one_line(error)
            note(f"error: {text}")
            if any(s in text for s in GPU_FAULTS):
                os._exit(3)
    else:
        note("skipped")
    mark()

    for config in spec["candidates"]:
        hook.override = config
        try:
            out = output_of(run())
            torch.cuda.synchronize()
            if spec["check"]:
                check_output(out, ref)
            profile(run)
            note("ok")
        except Exception as error:  # noqa: BLE001 -- a bad config fails only itself
            text = one_line(error)
            note(f"error: {text}")
            if any(s in text for s in GPU_FAULTS):
                # The GPU context is gone. Leave at once; tune_gemm.py starts
                # a new process for the remaining configs.
                os._exit(3)
        hook.override = None
        mark()


if __name__ == "__main__":
    main(sys.argv[1])
