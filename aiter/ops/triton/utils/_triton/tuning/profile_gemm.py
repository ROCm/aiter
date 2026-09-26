# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run one config under rocprofv3; tune_gemm.py owns timeouts and selection."""

import copy
import inspect
import json
import math
import sys
from pathlib import Path

import torch
import triton
import triton.language as tl
from gemm_cases import CASES, backend_kwarg
from triton.testing import runtime

from aiter.ops.triton.utils import gemm_config_utils
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir


@triton.jit
def tuning_run_marker(pointer):
    value = tl.load(pointer)
    tl.store(pointer, value + 1)


def family_record(lookup):
    """Capture the config files and bucket the parent will use without a GPU."""
    name = lookup["config_name"]
    directory = Path(resolve_config_dir("gemm", name, backend=lookup["backend"]))
    default = directory / "DEFAULT.json"
    defaults = load_config_json(str(default))
    suffix = lookup["specialized_filename"]
    if suffix is not None:
        paths = [directory / f"{name}-{suffix}.json"]
    else:
        N, K, B = lookup["N"], lookup["K"], lookup["B"]
        if N is None or K is None:
            raise ValueError("The lookup needs N/K or a specialized filename")
        paths = [directory / f"{name}-N={N}-K={K}.json"]
        if B is not None:
            paths.insert(0, directory / f"{name}-B={B}-N={N}-K={K}.json")
    source = next((path for path in paths if path.exists()), default)
    table = load_config_json(str(source))
    bounds = lookup["bounds"] or table.get(
        "M_BOUNDS", gemm_config_utils.STANDARD_M_BOUNDS
    )
    bucket = next(
        (f"M_LEQ_{bound}" for bound in bounds if bound >= lookup["M"]),
        f"M_GEQ_{bounds[-1]}",
    )
    seen = {}
    for path in sorted(directory.glob("*.json")):
        for entry in load_config_json(str(path)).values():
            if isinstance(entry, dict):
                for key, value in entry.items():
                    if value not in seen.setdefault(key, []):
                        seen[key].append(value)
    return {
        "lookup": lookup,
        "config": None,
        "is_tuned": False,
        "arch": arch_info.get_arch(),
        "defaults": defaults,
        "seen": seen,
        "source": str(source),
        "target": str(paths[0]),
        "table": table,
        "bucket": bucket,
    }


class ConfigLookup:
    def __init__(self, original, spec):
        self.original = original
        self.signature = inspect.signature(original)
        self.lookup = spec.get("lookup")
        self.candidate = spec["config"]
        self.record_path = Path(spec["record"])
        self.record = None
        self.called = False

    def save_record(self):
        temporary = self.record_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.record))
        temporary.replace(self.record_path)

    def __call__(self, *args, **kwargs):
        call = self.signature.bind(*args, **kwargs)
        call.apply_defaults()
        lookup = dict(call.arguments)
        if lookup["bounds"] is not None:
            lookup["bounds"] = list(lookup["bounds"])
        if self.lookup is None:
            self.lookup = lookup
        elif self.lookup != lookup:
            raise ValueError(
                "This case reads multiple configs; tune its GEMMs separately"
            )
        self.called = True
        if self.candidate is not None:
            return copy.deepcopy(self.candidate), True
        if self.record is None:
            self.record = family_record(lookup)
            # Keep discovery even when the current config cannot resolve or launch.
            self.save_record()
            config, is_tuned = self.original(*args, **kwargs)
            self.record.update(config=config, is_tuned=is_tuned)
            self.save_record()
        return copy.deepcopy(self.record["config"]), self.record["is_tuned"]


def snapshot(result):
    outputs = result if isinstance(result, (tuple, list)) else (result,)
    return tuple(output.detach().clone() for output in outputs)


def check_outputs(outputs, reference):
    if not outputs:
        raise ValueError("The case returned no output tensors")
    if reference is not None and len(outputs) != len(reference):
        raise ValueError("Output tensor count changed")
    for index, output in enumerate(outputs):
        if not torch.isfinite(output.float()).all():
            raise ValueError(f"Output {index} contains NaN or Inf")
        if reference is None:
            continue
        expected = reference[index].to(output.device)
        if output.shape != expected.shape or output.dtype != expected.dtype:
            raise ValueError(f"Output {index} shape or dtype changed")
        if output.is_floating_point():
            output, expected = output.float(), expected.float()
            error = (
                (output - expected).norm() / expected.norm().clamp_min(1e-12)
            ).item()
            if not math.isfinite(error) or error > 0.05:
                raise ValueError(f"Output {index} differs from the current config")
        elif not torch.equal(output, expected):
            raise ValueError(f"Output {index} integer values changed")


def profile(run, runs):
    """Mark each invocation so the parent can sum only its selected GPU kernels."""
    marker = torch.zeros(1, dtype=torch.float32, device="cuda")
    cache = runtime.driver.active.get_empty_cache_for_benchmark()
    torch.cuda.synchronize()
    tuning_run_marker[(1,)](marker)
    torch.cuda.synchronize()
    for _ in range(runs):
        cache.zero_()
        torch.cuda.synchronize()
        run()
        torch.cuda.synchronize()
        tuning_run_marker[(1,)](marker)
        torch.cuda.synchronize()


def main(spec_path):
    spec = json.loads(Path(spec_path).read_text())
    runs = spec.get("runs", 250)
    if runs <= 0:
        raise ValueError("runs must be positive")
    torch.manual_seed(0)
    run = CASES[spec["op"]](**spec["dims"], **backend_kwarg(spec["backend"]))
    original = gemm_config_utils._get_gemm_config_cached
    hook = ConfigLookup(original, spec)
    gemm_config_utils._get_gemm_config_cached = hook
    try:
        outputs = snapshot(run())  # Compile and warm up before the first marker.
        torch.cuda.synchronize()
        if not hook.called:
            raise ValueError("The case never called get_gemm_config()")
        reference_path = Path(spec["reference"])
        reference = None
        if spec["config"] is not None and reference_path.exists():
            reference = torch.load(
                reference_path, map_location="cpu", weights_only=True
            )
        check_outputs(outputs, reference)
        if spec["config"] is None:
            temporary = reference_path.with_suffix(".tmp")
            torch.save(tuple(output.cpu() for output in outputs), temporary)
            temporary.replace(reference_path)
        profile(run, runs)
    finally:
        gemm_config_utils._get_gemm_config_cached = original


if __name__ == "__main__":
    main(sys.argv[1])
