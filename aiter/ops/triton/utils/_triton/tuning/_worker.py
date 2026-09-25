"""The process the tuning scripts run under ``rocprofv3``.

``sweep_configs.py`` and ``verify_configs.py`` start ``python3 _worker.py
<spec.json>``. The worker builds one GEMM case's inputs, then

- with no candidates: runs the op with whatever config the library resolves
  and records that lookup (config family, backend, arch, the file it read);
- with candidates: runs the op once per candidate config, answering the op's
  own ``get_gemm_config()`` lookup with the candidate instead of the JSON.

Every profiled block ends with a ``split_dummy`` launch, so the kernel trace
splits into one segment per block: segment 0 is setup, then one segment per
attempted candidate (or one for the resolved config).
"""

import glob
import json
import os
import sys

import torch
import triton
import triton.language as tl
from triton.testing import runtime

# Largest relative L2 error a candidate's output may have against the output of
# the config the library resolves. Tilings and split-K change the accumulation
# order, not the math, so a correct config lands far below this.
REL_TOL = 0.05

_RESOURCE_ERRORS = ("OutOfResources", "exceeds triton maximum tensor numel")
_FATAL_GPU_ERRORS = (
    "illegal memory access",
    "HIP error",
    "hipError",
    "CUDA error",
    "device-side assert",
)


@triton.jit
def split_dummy(d_ptr):
    pid = tl.program_id(axis=0)
    x = tl.load(d_ptr + pid)
    x = x + 1
    tl.store(d_ptr + pid, x)


def _device():
    return runtime.driver.active.get_device_interface()


def profile(fn, n_run: int):
    """Run ``fn`` ``n_run`` times with a cold L2 before each run."""
    di = _device()
    cache = runtime.driver.active.get_empty_cache_for_benchmark()
    for _ in range(n_run):
        cache.zero_()
        di.synchronize()
        fn()
        di.synchronize()


def mark_segment():
    """Launch ``split_dummy``, which ends the current trace segment."""
    di = _device()
    d = torch.empty(128, dtype=torch.float32, device="cuda")
    di.synchronize()
    split_dummy[(128,)](d)
    di.synchronize()


class ConfigLookup:
    """Replaces ``gemm_config_utils._get_gemm_config_cached``, which every
    ``get_gemm_config()`` call goes through whatever module the caller imported
    it into. Records each lookup and, while ``override`` is set, returns that
    config as if a tuned file for the shape existed."""

    def __init__(self):
        from aiter.ops.triton.utils import gemm_config_utils

        self._module = gemm_config_utils
        try:
            self._resolve = gemm_config_utils._get_gemm_config_cached
        except AttributeError as error:
            raise RuntimeError(
                "gemm_config_utils._get_gemm_config_cached is gone; update "
                "ConfigLookup in tuning/_worker.py to the new lookup entry point"
            ) from error
        self.override = None
        self.calls = 0
        self.lookups = []
        self.resolved = None

    def install(self):
        self._module._get_gemm_config_cached = self

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
        self.calls += 1
        lookup = {
            "config_name": config_name,
            "backend": backend,
            "N": N,
            "K": K,
            "B": B,
            "bounds": list(bounds) if bounds is not None else None,
            "specialized_filename": specialized_filename,
        }
        if lookup not in self.lookups:
            self.lookups.append(lookup)
        if self.override is not None:
            return dict(self.override), True
        self.resolved = self._resolve(
            config_name, M, N, K, bounds, specialized_filename, backend, B
        )
        return self.resolved


def _specialized_files(config_dir, lookup):
    """Specialized files ``get_gemm_config()`` tries for this lookup, in order."""
    name, N, K, B = (lookup[k] for k in ("config_name", "N", "K", "B"))
    if lookup["specialized_filename"] is not None:
        suffixes = [lookup["specialized_filename"]]
    elif N is not None and K is not None:
        suffixes = ([f"B={B}-N={N}-K={K}"] if B is not None else []) + [f"N={N}-K={K}"]
    else:
        suffixes = []
    return [os.path.join(config_dir, f"{name}-{suffix}.json") for suffix in suffixes]


def describe_family(lookup):
    """Where the looked-up family lives and what its configs contain: the keys of
    this arch/backend's DEFAULT.json (the knobs the tuner sweeps), the values
    each key takes across the family's files for this backend on any arch, and
    the M bounds the lookup searches."""
    from aiter.ops.triton.utils import config_utils, gemm_config_utils
    from aiter.ops.triton.utils._triton import arch_info

    backend = lookup["backend"]
    config_dir = config_utils.resolve_config_dir(
        "gemm", lookup["config_name"], backend=backend
    )
    default = config_utils.load_config_json(
        os.path.join(config_dir, "DEFAULT.json"), required=False
    )
    keys = []
    for params in (default or {}).values():
        if isinstance(params, dict):
            keys += [key for key in params if key not in keys]

    observed = {}
    family_files = os.path.join(
        config_utils.AITER_TRITON_CONFIGS_PATH,
        "*",
        backend,
        "gemm",
        os.path.basename(config_dir),
        "*.json",
    )
    for path in sorted(glob.glob(family_files)):
        for params in config_utils.load_config_json(path).values():
            if isinstance(params, dict):
                for key, value in params.items():
                    values = observed.setdefault(key, [])
                    # Compare types too: True == 1, but they are different knobs.
                    if not any(type(v) is type(value) and v == value for v in values):
                        values.append(value)

    tuned_file = next(
        (p for p in _specialized_files(config_dir, lookup) if os.path.isfile(p)),
        None,
    )
    bounds = (
        lookup["bounds"]
        or (default or {}).get("M_BOUNDS")
        or list(gemm_config_utils.STANDARD_M_BOUNDS)
    )
    return {
        "arch": arch_info.get_arch(),
        "configs_root": config_utils.AITER_TRITON_CONFIGS_PATH,
        "config_dir": config_dir,
        "default_exists": default is not None,
        "keys": keys,
        "observed": observed,
        "bounds": list(bounds),
        "tuned_file": tuned_file,
    }


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _mismatch(out, ref):
    """Why ``out`` does not match the reference output, or None if it does."""
    out = _first_tensor(out)
    if out is None:
        return None
    if out.shape != ref.shape:
        return f"output shape {tuple(out.shape)} != {tuple(ref.shape)}"
    out = out.float()
    if not torch.isfinite(out).all():
        return "output has NaN/Inf"
    error = ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    if error > REL_TOL:
        return f"output differs from the resolved config's by {error:.3g} (relative L2)"
    return None


def _short(error):
    lines = str(error).strip().splitlines()
    return f"{type(error).__name__}: {lines[0][:300] if lines else ''}"


def run(spec):
    from gemm_cases import CASES

    case = CASES[spec["op"]]
    kwargs = dict(spec["dims"])
    if spec.get("backend"):
        kwargs["backend"] = spec["backend"]

    lookup = ConfigLookup()
    lookup.install()
    fn = case.make_fn(spec["M"], **kwargs)

    # Run once with the config the library resolves: records the lookup and
    # gives the reference output that candidates are checked against.
    record = {"error": None}
    ref = None
    try:
        ref = _first_tensor(fn())
        _device().synchronize()
        if lookup.calls == 0:
            raise RuntimeError(
                f"{spec['op']} never called get_gemm_config(), so it cannot be tuned"
            )
        if ref is not None:
            ref = ref.float().clone()
            if not torch.isfinite(ref).all():
                ref = None
    except Exception as error:  # noqa: BLE001 -- reported to the driver
        record["error"] = _short(error)
    record["lookups"] = lookup.lookups
    if lookup.lookups:
        record["family"] = describe_family(lookup.lookups[0])
    if lookup.resolved is not None:
        record["config"], record["is_tuned"] = lookup.resolved
    with open(spec["record"], "w") as f:
        json.dump(record, f)
    mark_segment()

    candidates = spec.get("candidates")
    if candidates is None:
        if record["error"] is not None:
            return 1
        profile(fn, spec["n_run"])
        mark_segment()
        return 0

    block_keys = spec["block_keys"]
    if not spec["check"]:
        ref = None
    failed_blocks = set()
    with open(spec["status"], "a") as status:
        for index, config in enumerate(candidates):
            block = tuple(config.get(key) for key in block_keys)
            if block in failed_blocks:
                _emit(
                    status, index, "skipped", "these block sizes ran out of resources"
                )
                continue
            error = _try_candidate(fn, lookup, config, ref, spec["n_run"])
            if error is None:
                _emit(status, index, "ok")
            else:
                text = f"{type(error).__name__}: {error}"
                resource = any(e in text for e in _RESOURCE_ERRORS)
                if resource:
                    failed_blocks.add(block)
                _emit(status, index, "error", _short(error), resource)
                if any(e in text for e in _FATAL_GPU_ERRORS):
                    # The GPU context is gone; the driver reruns the rest.
                    return 3
            mark_segment()
    return 0


def _try_candidate(fn, lookup, config, ref, n_run):
    """Profile ``fn`` with ``config`` answering its config lookup. Returns the
    exception that stopped it, or None."""
    lookup.override = config
    calls = lookup.calls
    try:
        out = fn()
        _device().synchronize()
        if lookup.calls == calls:
            raise RuntimeError("the op did not look up its config")
        if ref is not None:
            reason = _mismatch(out, ref)
            if reason is not None:
                raise ValueError(reason)
        profile(fn, n_run)
    except Exception as error:  # noqa: BLE001 -- disqualifies this config only
        return error
    finally:
        lookup.override = None
    return None


def _emit(status, index, state, error=None, resource=False):
    entry = {"index": index, "status": state, "error": error, "resource": resource}
    status.write(json.dumps(entry) + "\n")
    status.flush()


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        sys.exit(run(json.load(f)))
