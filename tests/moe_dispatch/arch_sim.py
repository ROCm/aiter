# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run MoE dispatch as if on a given GPU arch, in a process that has no GPU.

`prepare_process()` must run before torch or aiter is imported: it hides the GPUs and
removes environment variables that change dispatch, so results depend only on the code
and the shipped tuned tables.

`simulated_arch(name)` then patches, for the duration of the block:
- the chip-info functions (`get_gfx`, `get_gfx_runtime`, `get_cu_num`) in every loaded
  module that bound them by name;
- `aiter.dtypes.fp8` and the import-time copies of it in `fused_moe_1stage_dict`;
- hardware probes that ask the compiled kernels about the real GPU (`PROBE_STUBS`).

Probe stubs are a documented approximation: they answer from the shape alone, so a
snapshot records dispatch logic, not the contents of each arch's kernel tables.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import contextlib
import os
import sys

ARCHS = {
    # name: (gfx, cu_num)
    "gfx942": ("gfx942", 304),
    "gfx942_cu80": ("gfx942", 80),
    "gfx950": ("gfx950", 256),
    "gfx1201": ("gfx1201", 32),
}

_KEEP_ENV = {
    "AITER_ROOT_DIR",
    "AITER_JIT_DIR",
    "AITER_ROCM_ARCH",
    "AITER_REBUILD",
}
_EXTRA_SCRUB_ENV = {"GPU_ARCHS", "GPTOSS_SWIGLU_MXFP4_BF16_BOUND"}

_CHIP_FUNCS = ("get_gfx", "get_gfx_runtime", "get_cu_num")


def prepare_process(tuned_tables=None):
    """Hide GPUs and drop dispatch-affecting env vars. Call before importing torch.

    `tuned_tables` maps `AITER_CONFIG_*` names to CSV paths to use instead of the
    shipped tables (the snapshot freezes its inputs this way).
    """
    if "torch" in sys.modules:
        raise RuntimeError("prepare_process() must run before torch is imported")
    for name in list(os.environ):
        if (name.startswith("AITER_") and name not in _KEEP_ENV) or (
            name in _EXTRA_SCRUB_ENV
        ):
            del os.environ[name]
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["ROCR_VISIBLE_DEVICES"] = "-1"
    os.environ["AITER_LOG_LEVEL"] = "ERROR"
    for name, path in (tuned_tables or {}).items():
        os.environ[name] = str(path)


def _probe_fmoe_fp8_blockscale_g1u1_has_kernel(
    activation, xquant, block_size_M, inter_dim, model_dim, *args, **kwargs
):
    # The real probe also needs an ASM kernel for the running arch that fits the tile.
    return int(inter_dim % 128 == 0)


PROBE_STUBS = {
    "fmoe_fp8_blockscale_g1u1_has_kernel": _probe_fmoe_fp8_blockscale_g1u1_has_kernel,
}


def _fp8_for(gfx):
    from aiter.utility import dtypes

    return dtypes.defaultDtypes[gfx]["fp8"]


def clear_dispatch_caches():
    """Clear every dispatch cache, legacy and v2 (their keys do not include the arch)."""
    import aiter.fhmoe
    import aiter.fused_moe
    from aiter.moev2.dispatch import tuned

    for module in (aiter.fused_moe, aiter.fhmoe):
        for value in vars(module).values():
            clear = getattr(value, "cache_clear", None)
            if callable(clear):
                clear()
    tuned.clear_cache()


def _bound(module, func):
    # Lazy modules may raise anything from __getattr__; they are not ours.
    try:
        return getattr(module, func, None)
    except Exception:  # noqa: BLE001
        return None


def _unbind_fakes(fakes, originals):
    """Point modules still bound to `fakes` back at `originals`."""
    for module in list(sys.modules.values()):
        for func, fake in fakes.items():
            if _bound(module, func) is fake:
                setattr(module, func, originals[func])


@contextlib.contextmanager
def _patched(obj, name, value):
    old = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, old)


@contextlib.contextmanager
def simulated_arch(name):
    import aiter
    import aiter.fhmoe as _fhmoe  # noqa: F401 - preload before scanning bindings
    import aiter.fused_moe as fm
    from aiter.jit.utils import chip_info
    from aiter.moev2 import paths
    from aiter.utility import dtypes

    # Built-in paths bind chip-info helpers at import time. Load them before the module
    # scan so every alias is patched for this context and restored when it exits.
    paths.registered()
    gfx, cu_num = ARCHS[name]
    fakes = {
        "get_gfx": lambda: gfx,
        "get_gfx_runtime": lambda: gfx,
        "get_cu_num": lambda: cu_num,
    }
    originals = {f: getattr(chip_info, f) for f in _CHIP_FUNCS}
    arch_fp8 = _fp8_for(gfx)
    import_fp8 = dtypes.fp8
    one_stage = {
        arch: {
            tuple(arch_fp8 if el is import_fp8 else el for el in key): api
            for key, api in table.items()
        }
        for arch, table in fm.fused_moe_1stage_dict.items()
    }

    with contextlib.ExitStack() as stack:
        for module in list(sys.modules.values()):
            for func, orig in originals.items():
                if _bound(module, func) is orig:
                    stack.enter_context(_patched(module, func, fakes[func]))
        stack.enter_context(_patched(dtypes, "fp8", arch_fp8))
        stack.enter_context(_patched(fm, "fused_moe_1stage_dict", one_stage))
        for probe, stub in PROBE_STUBS.items():
            stack.enter_context(_patched(aiter, probe, stub))
        clear_dispatch_caches()
        try:
            yield gfx, cu_num
        finally:
            clear_dispatch_caches()
            # Modules first imported inside the block (kernel modules load at launch) bound
            # these fakes; the scan above never saw them, and later contexts would skip them.
            _unbind_fakes(fakes, originals)
