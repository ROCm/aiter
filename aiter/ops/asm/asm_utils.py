# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Generic, kernel-agnostic helpers to load and launch a HIP ``.co`` (code
object) directly from Python via ctypes — no C++ host bridge required.

Nothing here is MLA- or op-specific; it is the shared "launch a
prebuilt asm kernel" layer meant to be reused when porting other aiter asm
kernels off their C++ dispatchers:

  * device / arch info            -> get_warp_size
  * HIP ``.co`` launch via ctypes -> load_hip / hip_check / get_function /
                                     launch_co / launch_co_cluster
  * asm kernel registry helpers   -> dtype_str / strip_csv_comments /
                                     load_asm_cfg_csv
  * torch.compile interop          -> register_asm_custom_op

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
from typing import ClassVar

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
# hipLaunchAttributeID::hipLaunchAttributeClusterDimension.
HIP_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4


class _HIPClusterDim(ctypes.Structure):
    _fields_: ClassVar = [
        ("x", ctypes.c_uint),
        ("y", ctypes.c_uint),
        ("z", ctypes.c_uint),
    ]


class _HIPLaunchAttrVal(ctypes.Union):
    # hipLaunchAttributeValue is a 64-byte union.
    _fields_: ClassVar = [
        ("clusterDim", _HIPClusterDim),
        ("_pad", ctypes.c_ubyte * 64),
    ]


class _HIPLaunchAttr(ctypes.Structure):
    _fields_: ClassVar = [
        ("id", ctypes.c_int),
        ("_align", ctypes.c_ubyte * 4),
        ("val", _HIPLaunchAttrVal),
    ]


class _HIPLaunchConfig(ctypes.Structure):
    # HIP_LAUNCH_CONFIG: 7 x uint32, stream, attrs pointer, numAttrs.
    _fields_: ClassVar = [
        ("gridDimX", ctypes.c_uint),
        ("gridDimY", ctypes.c_uint),
        ("gridDimZ", ctypes.c_uint),
        ("blockDimX", ctypes.c_uint),
        ("blockDimY", ctypes.c_uint),
        ("blockDimZ", ctypes.c_uint),
        ("sharedMemBytes", ctypes.c_uint),
        ("hStream", ctypes.c_void_p),
        ("attrs", ctypes.POINTER(_HIPLaunchAttr)),
        ("numAttrs", ctypes.c_uint),
    ]


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
_hip_launch_kernel_ex = None


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


def _get_hip_launch_kernel_ex():
    """Bind the optional cluster-launch entry point only when it is needed."""
    global _hip_launch_kernel_ex
    if _hip_launch_kernel_ex is not None:
        return _hip_launch_kernel_ex
    hip = _get_hip()
    try:
        launch_kernel_ex = hip.hipDrvLaunchKernelEx
    except AttributeError as exc:
        raise RuntimeError(
            "launch_co_cluster requires hipDrvLaunchKernelEx, but the loaded "
            "HIP runtime does not export it"
        ) from exc
    launch_kernel_ex.argtypes = [
        ctypes.POINTER(_HIPLaunchConfig),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    _hip_launch_kernel_ex = launch_kernel_ex
    return _hip_launch_kernel_ex


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


def launch_co_cluster(
    func, grid, block, kernarg, cluster_dim, stream=None, shared_mem=0
):
    """Like :func:`launch_co`, but launches through ``hipDrvLaunchKernelEx``
    with ``hipLaunchAttributeClusterDimension`` set to ``cluster_dim``
    (``(cx, cy, cz)``). Each grid dim must be a multiple of the matching
    cluster dim, and the ``.co`` must be built for that cluster shape. A
    ``(1, 1, 1)`` cluster falls back to the plain launch.
    """
    if cluster_dim is None or tuple(cluster_dim) == (1, 1, 1):
        return launch_co(
            func, grid, block, kernarg, stream=stream, shared_mem=shared_mem
        )
    gx, gy, gz = grid
    bx, by, bz = block
    cx, cy, cz = cluster_dim
    if gx % cx or gy % cy or gz % cz:
        raise ValueError(
            f"launch_co_cluster: grid {tuple(grid)} is not a multiple of "
            f"cluster {tuple(cluster_dim)}"
        )
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
    attr = _HIPLaunchAttr()
    attr.id = HIP_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    attr.val.clusterDim.x = cx
    attr.val.clusterDim.y = cy
    attr.val.clusterDim.z = cz
    cfg = _HIPLaunchConfig()
    cfg.gridDimX = gx
    cfg.gridDimY = gy
    cfg.gridDimZ = gz
    cfg.blockDimX = bx
    cfg.blockDimY = by
    cfg.blockDimZ = bz
    cfg.sharedMemBytes = shared_mem
    cfg.hStream = ctypes.c_void_p(stream)
    cfg.attrs = ctypes.pointer(attr)
    cfg.numAttrs = 1
    hip_check(
        _get_hip_launch_kernel_ex()(ctypes.byref(cfg), func, None, extra),
        "hipDrvLaunchKernelEx",
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


@functools.cache
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


# ---------------------------------------------------------------------------
# torch.compile interop
#
# A pure-Python ctypes ``.co`` launcher (see launch_co above) is opaque to
# TorchDynamo: it can neither be traced into (untraceable C builtins + the
# ctypes FFI) nor fake-run (it calls ``data_ptr()`` on FakeTensors). So any
# function that reaches such a launch graph-breaks, and `torch.compile(
# fullgraph=True)` over it raises.
#
# Registering the launch as an aiter custom op (schema inferred from the op
# function's type hints, plus a fake/meta impl) makes Dynamo treat it as ONE
# opaque graph node with known metadata: no graph break, and the real ctypes
# launch still runs at execution time. This helper is the generic, op-agnostic
# glue; each asm op only has to supply a schema-clean, type-annotated launch
# function and the list of buffers it mutates.
# ---------------------------------------------------------------------------
def register_asm_custom_op(
    op_name,
    op_func,
    mutates_args,
    fake_impl=None,
    dispatch_key="CUDA",
):
    """Register a pure-Python asm ``.co`` launcher as an aiter custom op so it is
    ``torch.compile(fullgraph=True)``-safe.

    Args:
        op_name: op name under the ``aiter`` torch library (callable as
            ``torch.ops.aiter.<op_name>``). Must be unique process-wide.
        op_func: the launch function. It MUST be fully type-annotated (Tensor /
            Optional[Tensor] / int / float / bool ...) because the op schema is
            inferred from those hints via ``torch.library.infer_schema``; it may
            NOT take a ``torch.cuda.Stream`` or other non-schema types (wrap the
            real launcher in a thin, schema-clean adapter if needed).
        mutates_args: names of the tensor args the kernel writes in place (the
            aliasing the schema must record). For a pure in-place launch, list
            every output/scratch buffer here.
        fake_impl: meta/fake implementation. Omit it for a pure in-place op that
            returns nothing (a no-op fake is registered); supply one that returns
            correctly-shaped ``torch.empty_like``/``new_empty`` tensors if the op
            has real return values.
        dispatch_key: backend dispatch key (default "CUDA").

    Returns:
        The registered op callable (``torch.ops.aiter.<op_name>``).
    """
    # Lazy import: keeps this module import-light and dependency-scoped to torch
    # + ctypes for callers that never register a compile-safe op.
    from csrc.cpp_itfs.torch_utils import direct_register_custom_op

    if fake_impl is None:

        def fake_impl(*args, **kwargs):
            # Pure in-place op: mutated buffers are declared via `mutates_args`,
            # so the fake produces no new tensors.
            return None

    direct_register_custom_op(
        op_name,
        op_func,
        mutates_args=list(mutates_args),
        fake_impl=fake_impl,
        dispatch_key=dispatch_key,
    )
    return getattr(torch.ops.aiter, op_name)
