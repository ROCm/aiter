# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tune a GEMM, log failed configs, and save the fastest working config.

python tune_gemm.py gemm_a8w8 M=16 N=1024 K=1024
python tune_gemm.py --list
"""

import argparse
import copy
import inspect
import itertools
import json
import math
import traceback
from pathlib import Path

from gemm_cases import CASES, backend_kwarg


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("op", nargs="?", choices=CASES)
    parser.add_argument("dims", nargs="*", metavar="DIM=INT")
    parser.add_argument("--list", action="store_true", help="list GEMMs without a GPU")
    parser.add_argument("--backend", choices=("triton", "gluon"))
    parser.add_argument("--space", nargs="+", default=[], metavar="KEY=V1,V2")
    args = parser.parse_args()
    if not args.list and not args.op:
        parser.error("choose a GEMM, or use --list")
    return args


def parse_dims(tokens):
    dims = {}
    for token in tokens:
        name, _, value = token.partition("=")
        if not name or not value.isdigit() or int(value) <= 0 or name in dims:
            raise ValueError(f"Expected a positive DIM=INT, got {token!r}")
        dims[name] = int(value)
    return dims


def parse_space(tokens):
    space = {}
    for token in tokens:
        key, separator, values = token.partition("=")
        if not key or not separator or not values:
            raise ValueError(f"Expected KEY=V1,V2, got {token!r}")
        space[key] = []
        for value in values.split(","):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                pass  # Strings such as .cg do not need quotes.
            space[key].append(value)
    return space


class ConfigLookup:
    """Record the wrapper's lookup, then supply each candidate at that lookup."""

    def __init__(self, original):
        self.original = original
        self.signature = inspect.signature(original)
        self.lookup = None
        self.current = None
        self.is_tuned = False
        self.candidate = None

    def __call__(self, *args, **kwargs):
        call = self.signature.bind(*args, **kwargs)
        call.apply_defaults()
        if self.lookup is None:
            self.lookup = dict(call.arguments)
        elif self.lookup != call.arguments:
            raise ValueError(
                "This case reads multiple configs; tune its GEMMs separately"
            )
        if self.candidate is not None:
            return copy.deepcopy(self.candidate), True
        if self.current is None:
            self.current, self.is_tuned = self.original(*args, **kwargs)
        return copy.deepcopy(self.current), self.is_tuned


def find_family(lookup):
    """Use the same default, batch and custom filenames as get_gemm_config."""
    from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

    if lookup is None:
        raise ValueError("The case never called get_gemm_config()")
    name = lookup["config_name"]
    directory = Path(resolve_config_dir("gemm", name, backend=lookup["backend"]))
    default = directory / "DEFAULT.json"
    default_table = load_config_json(str(default))
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
    return directory, default_table, source, paths[0]


def candidate_configs(directory, default_table, lookup, overrides):
    """Try the family's config keys, using shared ranges and published values."""
    from aiter.ops.triton.utils._triton.arch_info import get_arch
    from aiter.ops.triton.utils.config_utils import load_config_json

    keys = dict.fromkeys(
        key
        for bucket in default_table.values()
        if isinstance(bucket, dict)
        for key in bucket
    )
    seen = {}
    for path in sorted(directory.glob("*.json")):
        for bucket in load_config_json(str(path)).values():
            if isinstance(bucket, dict):
                for key, value in bucket.items():
                    if value not in seen.setdefault(key, []):
                        seen[key].append(value)
    unknown = overrides.keys() - keys.keys()
    if unknown:
        print(
            f"Skipping search keys absent from this backend's DEFAULT.json: {sorted(unknown)}"
        )
    space = {}
    for key in keys:
        # The config rules retire kpack when retuning architectures other than gfx942.
        if key == "kpack" and get_arch() != "gfx942":
            continue
        space[key] = overrides.get(
            key, candidate_values(key, lookup["M"], lookup["N"], lookup["K"], seen)
        )
    print("Search space:", json.dumps(space))
    for values in itertools.product(*space.values()):
        yield dict(zip(space, values))


def candidate_values(key, M, N, K, seen):
    """Values to try for one config key."""
    fixed = {
        "GROUP_SIZE_M": [1, 4, 8],
        "num_warps": [1, 2, 4, 8],
        "num_stages": [1, 2],
        "waves_per_eu": [1, 2, 4, 6, 8],
        "matrix_instr_nonkdim": [16, 32],
        "cache_modifier": [".cg", None],
        "kpack": [1, 2],
    }
    if key in ("BLOCK_SIZE_M", "BLOCK_M"):
        values = [4, 8, 16] + [v for v in (32, 64, 128, 256, 512) if v <= M]
    elif key in ("BLOCK_SIZE_N", "BLOCK_N"):
        values = [16] + [v for v in (32, 64, 128, 256) if N is None or v <= N]
    elif key in ("BLOCK_SIZE_K", "BLOCK_K"):
        values = [16, 32, 64, 128] + [
            v for v in (256, 512, 1024) if K is None or v <= K
        ]
    elif key == "NUM_KSPLIT":
        values = [1] + [v for v in (2, 3, 4, 7, 8, 14, 16, 28) if K is None or v <= K]
    else:
        values = list(fixed.get(key, []))
    # Keep published values even when a generic heuristic would omit them.
    # Unknown keys (buffers, variants, ...) need no change to this script:
    # the author's JSON values or --space define their search.
    for value in seen.get(key, []):
        if value not in values:
            values.append(value)
    return values


def snapshot(result):
    """Copy outputs because the next call may reuse the same buffers."""
    outputs = result if isinstance(result, (tuple, list)) else (result,)
    return tuple(output.detach().clone() for output in outputs)


def check_outputs(outputs, reference):
    import torch

    if not outputs:
        raise ValueError("The case returned no output tensors")
    if reference is not None and len(outputs) != len(reference):
        raise ValueError("Output tensor count changed")
    for index, output in enumerate(outputs):
        if not torch.isfinite(output.float()).all():
            raise ValueError(f"Output {index} contains NaN or Inf")
        if reference is not None:
            expected = reference[index]
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


def measure(run, hook, errors, reference=None):
    """Check and time one config. Log failures and return infinity so they cannot win."""
    from triton.testing import do_bench

    try:
        outputs = snapshot(run())
        check_outputs(outputs, reference)
        elapsed = do_bench(run, return_mode="median")
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"Invalid benchmark time: {elapsed}")
        return elapsed, outputs
    except Exception as error:
        config = hook.candidate if hook.candidate is not None else hook.current
        errors.write(json.dumps(config) + "\n" + traceback.format_exc() + "\n")
        errors.flush()
        print(f"Failed: {type(error).__name__}: {error}")
        # These errors poison the GPU context; continuing would fail every config.
        if any(
            message in str(error).lower()
            for message in (
                "illegal memory access",
                "device-side assert",
                "memory access fault",
            )
        ):
            raise
        return float("inf"), None


def write_config(source, target, lookup, config):
    """Replace only this M bucket, retaining the rest of the source table."""
    from aiter.ops.triton.utils.config_utils import load_config_json
    from aiter.ops.triton.utils.gemm_config_utils import STANDARD_M_BOUNDS

    table = dict(load_config_json(str(source)))
    bounds = lookup["bounds"] or table.get("M_BOUNDS", STANDARD_M_BOUNDS)
    bucket = next(
        (f"M_LEQ_{b}" for b in bounds if b >= lookup["M"]), f"M_GEQ_{bounds[-1]}"
    )
    table[bucket] = config
    target.write_text(json.dumps(table, indent=4) + "\n")
    load_config_json.cache_clear()
    print(f"Saved {bucket} in {target}")


def tune(run, hook, overrides, errors, backend=None):
    # 1. Measure the config the wrapper currently uses.
    current_ms, reference = measure(run, hook, errors)
    directory, defaults, source, target = find_family(hook.lookup)
    if backend is not None and hook.lookup["backend"] != backend:
        raise ValueError(
            f"Requested {backend}, but the wrapper selected {hook.lookup['backend']}"
        )
    print(f"Current: {current_ms:.4f} ms ({source})")
    if reference is None:
        print(
            "Current config failed; candidates will only be checked for finite outputs."
        )
    best_config = hook.current if math.isfinite(current_ms) else None
    best_ms = current_ms

    # 2. Try each candidate. Failed configs are logged by measure() and skipped.
    for config in candidate_configs(directory, defaults, hook.lookup, overrides):
        hook.candidate = config
        elapsed, _ = measure(run, hook, errors, reference)
        if elapsed < best_ms:
            best_config, best_ms = copy.deepcopy(config), elapsed
            print(f"Best: {best_ms:.4f} ms {json.dumps(best_config)}")

    # 3. Add a missing tuned config, or replace an existing one only if faster.
    if best_config is None:
        raise RuntimeError("No config worked; see the error log")
    if not hook.is_tuned or best_ms < current_ms:
        write_config(source, target, hook.lookup, best_config)
    else:
        print("No faster config found; kept the existing config.")


def main():
    args = parse_args()
    if args.list:
        for name, case in CASES.items():
            print(f"{name}{inspect.signature(case)}")
        return
    case = CASES[args.op]
    dims = parse_dims(args.dims)
    kwargs = dict(dims, **backend_kwarg(args.backend))
    # Report missing or unknown dimensions before building GPU inputs.
    inspect.signature(case).bind(**kwargs)
    overrides = {**case.space, **parse_space(args.space)}

    from aiter.ops.triton.utils import gemm_config_utils

    run = case(**kwargs)
    original = gemm_config_utils._get_gemm_config_cached
    hook = ConfigLookup(original)
    shape = "-".join(f"{key}={value}" for key, value in dims.items())
    error_path = Path(f"errors-{args.op}-{shape}.txt")
    print(f"Errors will be written to {error_path}")
    gemm_config_utils._get_gemm_config_cached = hook
    try:
        with error_path.open("w") as errors:
            tune(run, hook, overrides, errors, args.backend)
    finally:
        gemm_config_utils._get_gemm_config_cached = original
        original.cache_clear()


if __name__ == "__main__":
    main()
