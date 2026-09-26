"""The process tune_gemm.py runs under rocprofv3. Do not run it by hand.

It runs one GEMM at one shape: first with the config the library resolves,
then with each candidate config in the spec. After every config it launches a
`split_dummy` kernel, so the kernel trace splits into one segment per config:
segment 0 is the current config, then one per candidate. Each config also gets
a line in the status file: `ok`, `skipped`, or `error: <what went wrong>`.
tune_gemm.py matches the two up.
"""

import json
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
RESOURCE_ERRORS = ("OutOfResources", "exceeds triton maximum tensor numel")
GPU_FAULTS = ("illegal memory access", "HIP error", "hipError", "device-side assert")


@triton.jit
def split_dummy(d_ptr):
    pid = tl.program_id(axis=0)
    x = tl.load(d_ptr + pid)
    tl.store(d_ptr + pid, x + 1)


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
            "N": N,
            "K": K,
            "B": B,
            "bounds": bounds,
            "specialized_filename": specialized_filename,
            "backend": backend,
        }
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
    for _ in range(N_RUNS):
        cache.zero_()
        device.synchronize()
        run()
        device.synchronize()


def mark():
    """Launch split_dummy, which ends the current segment of the kernel trace."""
    torch.cuda.synchronize()
    split_dummy[(128,)](torch.empty(128, dtype=torch.float32, device="cuda"))
    torch.cuda.synchronize()


def output_of(result):
    """The first tensor an op returned, as float32, or None."""
    if isinstance(result, torch.Tensor):
        return result.float()
    if isinstance(result, (tuple, list)):
        for item in result:
            out = output_of(item)
            if out is not None:
                return out
    return None


def check_output(out, ref):
    """Raises when `out` is not close to the current config's output `ref`."""
    if out is None or ref is None:
        return
    if out.shape != ref.shape:
        raise ValueError(f"output shape {tuple(out.shape)} != {tuple(ref.shape)}")
    if not torch.isfinite(out).all():
        raise ValueError("output has NaN or Inf")
    error = ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    if error > MAX_OUTPUT_ERROR:
        raise ValueError(f"output differs from the current config's by {error:.1%}")


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
            ref = output_of(run())
            torch.cuda.synchronize()
            profile(run)
            note("ok")
        except Exception as error:  # noqa: BLE001 -- a candidate may still work
            note(f"error: {one_line(error)}")
    else:
        note("skipped")
    mark()

    bad_tiles = set()
    for config in spec["candidates"]:
        tiles = tuple(config.get(key) for key in spec["tile_keys"])
        if tiles in bad_tiles:
            note("skipped: these tile sizes ran out of resources")
            mark()
            continue
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
            if any(s in text for s in RESOURCE_ERRORS):
                bad_tiles.add(tiles)
            if any(s in text for s in GPU_FAULTS):
                # The GPU context is gone. Leave at once; tune_gemm.py starts
                # a new process for the remaining configs.
                os._exit(3)
        hook.override = None
        mark()


if __name__ == "__main__":
    main(sys.argv[1])
