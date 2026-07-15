# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Generic, kernel-agnostic helpers to load and launch a HIP ``.co`` (code
object) directly from Python via ctypes — no C++ host bridge required.

Nothing here is MLA- or op-specific; it is the shared "launch a
prebuilt asm kernel" layer meant to be reused when porting other aiter asm
kernels off their C++ dispatchers:

  * device / arch info            -> get_warp_size
  * HIP ``.co`` launch via ctypes -> load_hip / hip_check / get_function /
                                     launch_co
  * asm kernel registry helpers   -> dtype_str / strip_csv_comments /
                                     load_asm_cfg_csv

The HIP handle is deliberately the SAME ``libamdhip64`` that torch already
mapped (found by scanning ``/proc/self/maps``), so module-load / launch share
torch's driver context and stream state and we avoid ROCR symbol-version
clashes with a second copy under /opt/rocm.

Only depends on PyTorch (for device streams) + ctypes.
"""

import csv
import ctypes
import functools
import glob
import os

import torch


# ---------------------------------------------------------------------------
# Device / arch info
# ---------------------------------------------------------------------------
def get_warp_size() -> int:
    """Hardware wave size of device 0 (32 on RDNA-family gfx1250, 64 on CDNA)."""
    try:
        return int(torch.cuda.get_device_properties(0).warp_size)
    except AttributeError:
        return 32  # gfx1250 (RDNA-family) is wave32


# ---------------------------------------------------------------------------
# HIP runtime binding (ctypes). torch owns device memory; we only module-load
# and launch. Bind the SAME libamdhip64 torch already mapped, to avoid ROCR
# symbol-version clashes with /opt/rocm and keep stream/context state consistent.
# ---------------------------------------------------------------------------
# HIP magic launch-param constants (hip_runtime_api.h).
HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(0x01)
HIP_LAUNCH_PARAM_BUFFER_SIZE = ctypes.c_void_p(0x02)
HIP_LAUNCH_PARAM_END = ctypes.c_void_p(0x03)


def load_hip():
    """Return a ctypes handle to the libamdhip64 torch has already loaded.

    Preference order: the .so mapped into this process (per /proc/self/maps) ->
    the one shipped inside the torch wheel -> the system SONAME. Sharing torch's
    handle keeps the HIP context / stream state consistent with torch.
    """
    candidates = []
    try:
        with open("/proc/self/maps") as f:
            for line in f:
                path = line.rstrip().split(" ")[-1]
                if "libamdhip64.so" in path and os.path.exists(path):
                    candidates.append(path)
    except OSError:
        pass
    candidates += glob.glob(
        os.path.join(os.path.dirname(torch.__file__), "lib", "libamdhip64.so*")
    )
    candidates.append("libamdhip64.so")
    last = None
    for cand in candidates:
        try:
            return ctypes.CDLL(cand)
        except OSError as exc:
            last = exc
    raise RuntimeError(f"could not load libamdhip64: {last}")


_hip = None


def _get_hip():
    """Lazily bind libamdhip64 on first launch (NOT at import time).

    Deferring the bind keeps ``import aiter.mla`` side-effect-free on hosts
    where this launcher is never exercised (e.g. non-gfx1250 arches that still
    route through the C++ dispatcher), and avoids a hard import failure if HIP
    is momentarily unavailable.
    """
    global _hip
    if _hip is not None:
        return _hip
    hip = load_hip()
    hip.hipGetErrorString.restype = ctypes.c_char_p
    hip.hipGetErrorString.argtypes = [ctypes.c_int]
    hip.hipModuleLoad.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
    hip.hipModuleGetFunction.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_char_p,
    ]
    hip.hipModuleLaunchKernel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _hip = hip
    return _hip


def hip_check(err, what):
    if err != 0:
        msg = _get_hip().hipGetErrorString(err).decode()
        raise RuntimeError(f"HIP error in {what}: ({err}) {msg}")


_module_cache = {}
_func_cache = {}


def get_function(co_path, symbol):
    """Load ``co_path`` once (process-cached) and resolve ``symbol`` to a
    function handle (also cached). Mirrors the ``AiterAsmKernel`` /
    ``SynchronizedCache`` behaviour of the C++ dispatcher: a given .co is mapped
    exactly once and reused across launches.
    """
    hip = _get_hip()
    module = _module_cache.get(co_path)
    if module is None:
        module = ctypes.c_void_p()
        hip_check(
            hip.hipModuleLoad(ctypes.byref(module), co_path.encode()),
            "hipModuleLoad",
        )
        _module_cache[co_path] = module
    key = (co_path, symbol)
    func = _func_cache.get(key)
    if func is None:
        func = ctypes.c_void_p()
        hip_check(
            hip.hipModuleGetFunction(ctypes.byref(func), module, symbol.encode()),
            "hipModuleGetFunction",
        )
        _func_cache[key] = func
    return func


def launch_co(func, grid, block, kernarg, stream=None, shared_mem=0):
    """Launch ``func`` with a single packed ctypes.Structure kernarg via the HIP
    BUFFER_POINTER/SIZE extra-config protocol (== ``AiterAsmKernel::launch``).

    ``grid`` / ``block`` are (x, y, z) tuples. ``stream`` defaults to the current
    torch stream so the launch is stream- and CUDA-graph-correct. ``kernarg`` is
    kept alive by the caller for the duration of this call (ctypes holds a
    borrowed pointer into it).
    """
    arg_size = ctypes.c_size_t(ctypes.sizeof(kernarg))
    extra = (ctypes.c_void_p * 5)(
        HIP_LAUNCH_PARAM_BUFFER_POINTER,
        ctypes.cast(ctypes.byref(kernarg), ctypes.c_void_p),
        HIP_LAUNCH_PARAM_BUFFER_SIZE,
        ctypes.cast(ctypes.byref(arg_size), ctypes.c_void_p),
        HIP_LAUNCH_PARAM_END,
    )
    if stream is None:
        stream = torch.cuda.current_stream().cuda_stream
    gx, gy, gz = grid
    bx, by, bz = block
    hip_check(
        _get_hip().hipModuleLaunchKernel(
            func,
            gx,
            gy,
            gz,
            bx,
            by,
            bz,
            shared_mem,
            ctypes.c_void_p(stream),
            None,
            extra,
        ),
        "hipModuleLaunchKernel",
    )


# ---------------------------------------------------------------------------
# asm kernel registry (.csv) helpers. The aiter `hsa/<arch>/<op>/*.csv` files
# map a shape/dtype tuple to a kernel symbol + .co name; the C++ dispatchers
# consume them via hsa/codegen.py. These helpers let a Python launcher read the
# SAME csv directly (no codegen), so the kernel registry stays single-sourced.
# Nothing here is op-specific — pass the csv path (and which columns are
# strings) at the call site.
# ---------------------------------------------------------------------------
# torch dtype -> short kernel-table string. Kept generic (the asm csv `qType` /
# `kvType` columns use these). Extend as new kernel dtypes ship.
_DTYPE_TO_STR = {
    torch.float8_e4m3fn: "fp8",
    torch.float8_e4m3fnuz: "fp8",
    torch.bfloat16: "bf16",
}


def dtype_str(t: torch.Tensor) -> str:
    """torch tensor/dtype -> short kernel-table string ('fp8' / 'bf16')."""
    dt = t.dtype if isinstance(t, torch.Tensor) else t
    s = _DTYPE_TO_STR.get(dt)
    if s is None:
        raise RuntimeError(f"unsupported dtype {dt} (no kernel-table string)")
    return s


def strip_csv_comments(lines):
    """Yield only the header + data lines of an asm-registry csv, dropping blank
    lines and ``//`` / ``#`` / ``;`` line comments and ``/* ... */`` blocks.

    Mirrors the comment stripping hsa/codegen.py applies, so a Python reader
    sees exactly the rows codegen would. Generic — works on any such csv."""
    in_block = False
    for line in lines:
        s = line.strip()
        if in_block:
            if "*/" in s:
                in_block = False
            continue
        if s.startswith("/*"):
            if "*/" not in s:
                in_block = True
            continue
        if not s or s.startswith(("//", "#", ";")):
            continue
        yield line


# Columns that stay as strings; every other column is coerced to int. Matches
# the asm-csv convention (all shape/flag columns are integers; only the dtype
# and name columns are text). Callers may override via `str_cols`.
DEFAULT_ASM_CSV_STR_COLS = frozenset({"qType", "kvType", "knl_name", "co_name"})


@functools.lru_cache(maxsize=None)
def load_asm_cfg_csv(csv_path, str_cols=DEFAULT_ASM_CSV_STR_COLS):
    """Parse an asm-registry csv into a list of dict rows (process-cached per
    path). Integer columns are coerced to int; `str_cols` stay as text. Rows
    whose first column is empty (defensive) are skipped.

    Reading the shipped csv at runtime keeps the kernel registry single-sourced
    with the C++ codegen path instead of duplicating it in Python."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"asm registry csv not found: {csv_path}")
    cfgs = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(strip_csv_comments(f))
        first_field = reader.fieldnames[0] if reader.fieldnames else None
        for row in reader:
            if first_field is not None and not (row.get(first_field) or "").strip():
                continue
            parsed = {}
            for k, v in row.items():
                if k is None or v is None:
                    continue
                v = v.strip()
                parsed[k] = v if k in str_cols else int(v)
            cfgs.append(parsed)
    if not cfgs:
        raise RuntimeError(f"no kernel rows parsed from {csv_path}")
    return cfgs
