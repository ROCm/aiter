# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Persistent, trace-free FlyDSL kernel cache built on ``export_to_c``.

FlyDSL 0.5.0 keys its artifacts by the traced IR, so every first call of a
specialization traces the Python kernel again in each new process, even when
its disk cache hits. This module skips that trace. A prebuild (a launch under
``COMPILE_ONLY``, as the replays in ``aiter/aot/flydsl`` make) compiles the
specialization once with CPU placeholders and exports it with
``CompiledFunction.export_to_c``. A process that launches kernels loads the
object with ``flyc.load_module``, so a hit never traces. It never builds one:
a launch with no entry on disk logs an error and traces on the JIT path, as
it did before this cache. Only a prebuild, a process that imports its sources
and builds within seconds, writes the cache, so the code a long-running
process imported earlier cannot end up in an entry.

See ``docs/flydsl_aot_cache.md`` for the design and how cache entries are
invalidated.

Usage::

    _CACHE = FlyDSLAotCache("preshuffle_gemm_a8")
    _CACHE.launch(key, lambda: compile_fn(**kw), *launcher_args)

``launch`` replaces ``_run_compiled(compile_fn(**kw), *launcher_args)`` and
covers every mode, including ``0``. Within one cache ``name``, ``key`` must
determine the specialization: the launcher and every constexpr it is built
with. It is made of ints, floats, bools, strings, ``None``, torch dtypes and
devices, and tuples of those (hashable: it also keys the kernels in memory). The disk key also hashes the
``cache_signature`` of each runtime argument (dtype, rank, unit-stride axis),
so another process cannot load an object compiled for other argument types.
In memory, entries are looked up by ``key`` alone, so ``key`` must also hold
every tensor dtype and layout that can vary, as for ``_run_compiled``'s
``specialization_key``.

If an entry is missing, or cannot be exported or loaded (an argument type
``export_to_c`` does not support), ``launch`` logs an error once and runs
that key through ``_run_compiled``. A FlyDSL without ``flyc.load_module`` runs every key that
way.

Invalidation has two levels. The directory is named by a fingerprint of the
toolchain (FlyDSL, HIP, Python), the target (arch, CU count) and the
variables that change what the toolchain makes of any kernel (FlyDSL's own
list). Each cache also declares the source files its launchers are built
from and the environment variables those files read, ``AITER_FLYDSL_*``
included: setting one makes that cache miss and no other (the file creating
the cache, ``_COMMON_SOURCES`` and ``_COMMON_ENV`` are always among them)::

    FlyDSLAotCache("preshuffle_gemm_a8",
                   sources=["kernels/preshuffle_gemm.py", ...],
                   env=["MY_TUNING_SWITCH"])

Their contents and values are hashed into every entry's name, so an edit to a
declared file, or another value of a declared variable, makes new entries for
that cache only, and going back finds the earlier ones. Nothing is detected:
a file or a variable that is not declared does not invalidate anything.
``tests/test_flydsl_aot_launch.py`` checks that the sources of every cache
include the files under ``aiter/ops/flydsl`` those import.

``AITER_FLYDSL_AOT_CACHE`` selects the mode:

* ``1`` (default): on.
* ``0``: off. ``launch`` uses ``_run_compiled``.
* ``verify``: like ``1``, but each disk hit is re-traced, and the traced IR
  must be what the entry was built from (FlyDSL's key of it; the exported
  objects are compared byte for byte when an entry has none). A failure to
  load an entry, or to export one in a prebuild, raises instead of falling
  back. A launch with no entry still takes the JIT path.

Every launch a bound launcher sends to the JIT path, for whatever reason, is
logged once per key with ``logger.error`` (not under ``0``); nothing raises.

``AITER_FLYDSL_AOT_CACHE_DIR`` sets the cache root. The default is
``flydsl_aot`` in aiter's JIT directory (``get_user_jit_dir``), so entries a
prebuild writes ship with the package, like its JIT modules.
"""

from __future__ import annotations

import ast
import contextlib
import fcntl
import functools
import hashlib
import inspect
import json
import os
import platform
import re
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path

import flydsl
import flydsl.compiler as flyc
import torch
from flydsl.compiler import JitArgumentRegistry
from flydsl.compiler.jit_argument import PointerJitArg, is_type_param_annotation
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import cache_signature
from flydsl.expr.typing import Constexpr
from flydsl.utils import env as fly_env

from aiter import logger

__all__ = ["FlyDSLAotCache", "aot_cache_mode", "cache_root", "fingerprint"]

_SCHEMA = 7
_MODE = os.environ.get("AITER_FLYDSL_AOT_CACHE", "1").strip().lower()
if _MODE not in ("0", "1", "verify"):
    raise ValueError(f"AITER_FLYDSL_AOT_CACHE must be 0, 1 or verify, got {_MODE!r}")


def _flydsl_cache_env() -> tuple | None:
    """The variables FlyDSL keys its own artifact cache on, for every arch;
    None if this FlyDSL does not list them."""
    try:
        from flydsl.compiler import jit_function as jf

        names = set(jf._CACHE_INVALIDATING_ENV_VARS)
        for arch_names in jf._TARGET_CACHE_INVALIDATING_ENV_VARS.values():
            names.update(arch_names)
    except (ImportError, AttributeError):  # FlyDSL internals moved
        return None
    return tuple(sorted(names))


# The fingerprint holds the variables that change what the toolchain makes of
# any kernel: FlyDSL's own list, and these. A variable that kernel code reads
# is not here: the cache of that kernel declares it (``env=``), so setting it
# makes that cache miss and no other.
_FLYDSL_ENV = _flydsl_cache_env()
_ENV_FINGERPRINT = frozenset(
    {
        "HSA_OVERRIDE_GFX_VERSION",
        "COMPILE_OPT_LEVEL",
        # Changes gfx1250 codegen; FlyDSL keys its own cache on it for that arch.
        "LLVM_GFX1250_ANY_XNACK_CODEGEN",
        *(_FLYDSL_ENV or ()),
    }
)
# With a FlyDSL that does not list its variables, every FLYDSL_ one but these.
_ENV_IGNORED = frozenset(
    {
        "FLYDSL_DUMP_IR",
        "FLYDSL_DUMP_DIR",
        "FLYDSL_RUNTIME_CACHE_DIR",
        "FLYDSL_RUNTIME_ENABLE_CACHE",
        "FLYDSL_RUNTIME_RUN_ONLY",
        "FLYDSL_GPU_ARCH",
    }
)

# Variables aiter's FlyDSL code reads that cannot change generated code, or
# that reach an entry's identity another way. The check of a cache's sources
# (``_undeclared_env``) accepts them without a declaration.
_ENV_NOT_CODE = frozenset(
    {
        # Whether to launch; the cache's own switches.
        "COMPILE_ONLY",
        "AITER_FLYDSL_AOT_CACHE",
        "AITER_FLYDSL_AOT_CACHE_DIR",
        "AITER_FLYDSL_AOT_MAX_RETRIES",
        "AITER_FLYDSL_AOT_TIMEOUT",
        "AITER_FLYDSL_AOT_TRACEBACK",
        # The target: hashed as the arch and CU count they select.
        "FLYDSL_GPU_ARCH",
        "GPU_ARCHS",
        "AITER_GPU_ARCHS",
        "CU_NUM",
        "AITER_AOT_REPLAY",
        # FlyDSL's own cache.
        "FLYDSL_RUNTIME_CACHE_DIR",
        "FLYDSL_RUNTIME_ENABLE_CACHE",
        "FLYDSL_RUNTIME_RUN_ONLY",
    }
)

# Variables the common sources read (``_COMMON_SOURCES``): inputs of every
# cache, like those files.
_COMMON_ENV = (
    "AITER_FLYDSL_KERNARG_PRELOAD",
    "AITER_FLYDSL_KERNARG_PRELOAD_COUNT",
    "AITER_FLYDSL_MOE_EXPERT_SCHEDULING_MODE",
)


def aot_cache_mode() -> str:
    return _MODE


@functools.cache
def _default_root() -> Path:
    from aiter.jit.core import get_user_jit_dir

    return Path(get_user_jit_dir()) / "flydsl_aot"


def cache_root() -> Path:
    root = os.environ.get("AITER_FLYDSL_AOT_CACHE_DIR")
    return Path(root) if root else _default_root()


def _hash_tree(h, root: Path, patterns, *, contents: bool) -> None:
    files = sorted({p for pat in patterns for p in root.rglob(pat) if p.is_file()})
    for p in files:
        h.update(str(p.relative_to(root)).encode())
        if contents:
            h.update(p.read_bytes())
        else:
            st = p.stat()
            h.update(f"{st.st_size}:{st.st_mtime_ns}".encode())


def _target_arch() -> str:
    # Overrides (FLYDSL_GPU_ARCH, HSA_OVERRIDE_GFX_VERSION) are also hashed with
    # the environment. FLYDSL_GPU_ARCH names the target FlyDSL compiles for, so it
    # wins; get_rocm_arch() lower-cases it as FlyDSL's compiler does.
    if os.environ.get("FLYDSL_GPU_ARCH"):
        from flydsl.runtime.device import get_rocm_arch

        return get_rocm_arch()
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        return props.gcnArchName.split(":")[0]
    # No device visible to torch: resolve the target as FlyDSL's compiler does
    # (HSA_OVERRIDE_GFX_VERSION, then rocm_agent_enumerator / rocminfo). FlyDSL
    # >= 0.5.1 raises when none of them names a target; 0.5.0 guessed gfx942
    # instead, which would file entries under the wrong arch, so refuse that.
    from flydsl.runtime.device import get_rocm_arch
    from packaging.version import Version

    fly_version = Version(Version(flydsl.__version__).base_version)
    if os.environ.get("HSA_OVERRIDE_GFX_VERSION") or fly_version >= Version("0.5.1"):
        return get_rocm_arch()
    raise RuntimeError(
        "FlyDSL AOT cache: no GPU is visible and neither FLYDSL_GPU_ARCH nor "
        "HSA_OVERRIDE_GFX_VERSION is set; set FLYDSL_GPU_ARCH to the target gfx arch"
    )


def _wheel_record(fly_root: Path) -> bytes | None:
    """The file hashes of the FlyDSL wheel installed at ``fly_root``, if any.

    Only the package's own entries: an installer adds its own files and
    ``.pyc`` entries, which differ between installs of the same wheel.
    """
    from importlib.metadata import distributions

    for dist in distributions(path=[str(fly_root.parent)]):
        # An in-place upgrade can leave a dist-info with no METADATA (only
        # REQUESTED); reading its name fails or warns on some Pythons.
        if dist.read_text("METADATA") is None or dist.metadata["Name"] != "flydsl":
            continue
        record = dist.read_text("RECORD")
        if record is None:
            return None
        lines = sorted(
            line
            for line in record.splitlines()
            if line.startswith("flydsl/") and "__pycache__" not in line
        )
        return "\n".join(lines).encode()
    return None


def _cu_num() -> int:
    """CU count of the target, as the kernels' host code sees it.

    ``get_cu_num`` honours ``CU_NUM``, which the AOT dry run sets per tuned row.
    """
    from aiter.jit.utils.chip_info import get_cu_num

    try:
        return get_cu_num()
    except Exception:  # noqa: BLE001  no GPU and no CU_NUM: an unknown target
        return 0


def _env_in_fingerprint(name: str) -> bool:
    """Whether ``fingerprint`` hashes the variable ``name``."""
    if name in _ENV_FINGERPRINT:
        return True
    return (
        _FLYDSL_ENV is None and name.startswith("FLYDSL_") and name not in _ENV_IGNORED
    )


@functools.cache
def _toolchain_digest() -> str:
    """Hash of the FlyDSL install: its Python sources and native libraries."""
    h = hashlib.sha256()
    fly_root = Path(flydsl.__file__).parent
    _hash_tree(h, fly_root, ("*.py",), contents=True)
    record = _wheel_record(fly_root)
    if record is not None:
        # Content hashes, so a reinstall of the same wheel (e.g. in another
        # Docker stage) keeps the fingerprint; file mtimes would not.
        h.update(record)
    else:
        # A source build has no wheel RECORD.
        _hash_tree(h, fly_root, ("*.so", "*.so.*", "*.a"), contents=False)
    return h.hexdigest()


def _device_bitcode_digest() -> str:
    """Hash of the AMDGCN device libraries (ocml, ockl, ...) FlyDSL links into
    a kernel that calls them. Bundled with FlyDSL they are in its wheel, but
    without in-process LLD they come from ``ROCM_PATH`` or ``/opt/rocm``.
    FlyDSL caches the toolkit it resolves and the digest of each."""
    try:
        from flydsl.compiler.backends import rocm

        return rocm._device_bitcode_fingerprint(rocm.rocm_toolkit_path())
    except Exception as e:  # noqa: BLE001  FlyDSL internals moved
        return f"unknown:{type(e).__name__}"


def fingerprint() -> str:
    """Hash of the toolchain and environment every entry depends on.

    It covers FlyDSL (version, Python sources, native libraries), the ROCm
    device bitcode, torch HIP, Python, the target arch and CU count, and the
    ``AITER_FLYDSL_*`` / ``FLYDSL_*`` environment variables. Changing any of them selects a new
    cache directory. Kernel sources and the other variables they read are
    declared per cache instead (``FlyDSLAotCache``). Only the FlyDSL hash is
    cached: an AOT build switches arch and environment within one process.
    """
    env_items = sorted((k, v) for k, v in os.environ.items() if _env_in_fingerprint(k))
    parts = [
        _SCHEMA,
        sys.version_info[:2],
        platform.machine(),
        flydsl.__version__,
        torch.version.hip,
        _target_arch(),
        _cu_num(),
        _toolchain_digest(),
        _device_bitcode_digest(),
        env_items,
    ]
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


@functools.cache
def _tracked_root() -> str:
    import aiter

    return str(Path(aiter.__file__).parent.resolve()) + os.sep


def _portable(path: str) -> str:
    """``path`` as an entry names it: relative to the aiter package if inside
    it, else its file name, so entries built in a source tree still match once
    the package is installed elsewhere (a prebuilt wheel)."""
    root = _tracked_root()
    return path.removeprefix(root) if path.startswith(root) else os.path.basename(path)


@functools.cache
def _file_digest(path: str) -> str | None:
    """Content hash of a source file, once per process; None if it is gone."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _get_gfx():
    from aiter.jit.utils.chip_info import get_gfx

    try:
        return get_gfx()
    except Exception:  # noqa: BLE001  no GPU and no GPU_ARCHS
        return None


# Inputs a cache may declare besides environment variables. get_gfx() honours
# GPU_ARCHS (its last arch when it names several), while the fingerprint's arch
# is the device's or FlyDSL's target: an image setting GPU_ARCHS="gfx942;gfx950"
# answers gfx950 on a gfx942. A kernel that picks code by get_gfx() declares it.
_DERIVED = {"get_gfx()": _get_gfx}


def _input(name: str):
    """The current value of a declared input: a variable or a ``_DERIVED`` one."""
    return _DERIVED[name]() if name in _DERIVED else os.environ.get(name)


# Sources of every cache, so that no cache has to list them: the helpers
# nearly every kernel is built with, and the per-arch hardware constants
# kernels size themselves by (LDS capacity, CUs per XCD), which are outside
# ``aiter/ops/flydsl/kernels``, the only place the import check looks. A
# cache declares only the files specific to it.
_COMMON_SOURCES = (
    "kernels/kernels_common.py",
    "kernels/tensor_shim.py",
    "kernels/buffer_ops.py",
    "../../jit/utils/chip_info.py",
    "../../jit/utils/lds_capacity.py",
)


def _source_files(sources) -> list:
    """The files a cache declares: each path as given if absolute, else under
    ``aiter/ops/flydsl``; a directory stands for every ``.py`` below it."""
    base = os.path.dirname(os.path.realpath(__file__))
    files = set()
    for src in sources:
        path = os.path.realpath(os.path.join(base, os.fspath(src)))
        if os.path.isdir(path):
            found = [str(f) for f in Path(path).rglob("*.py")]
            if not found:
                raise _SourceListError(f"no .py file under declared source {src!r}")
            files.update(os.path.realpath(f) for f in found)
        elif os.path.isfile(path):
            files.add(path)
        else:
            raise _SourceListError(f"declared source {src!r} ({path}) does not exist")
    return sorted(files)


# The cache's own modules: they generate no kernel code.
_NOT_KERNEL_CODE = ("__init__.py", "aot_cache.py", "aot_launch.py")


@functools.cache
def _kernel_imports(path: str) -> frozenset:
    """The files under ``aiter/ops/flydsl/kernels`` that ``path`` imports,
    anywhere in it (read from its text; nothing is imported)."""
    root = Path(_tracked_root())
    kernels = Path(__file__).resolve().parent / "kernels"
    module = ".".join(("aiter", *Path(path).relative_to(root).with_suffix("").parts))
    found = set()
    for node in ast.walk(ast.parse(Path(path).read_bytes())):
        names = ()
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = module.rsplit(".", node.level)[0] if node.level else ""
            target = ".".join(part for part in (base, node.module) if part)
            names = [target, *(f"{target}.{alias.name}" for alias in node.names)]
        for name in names:
            if not name.startswith("aiter."):
                continue
            file = root.joinpath(*name.split(".")[1:]).with_suffix(".py")
            if (
                kernels in file.parents
                and file.name not in _NOT_KERNEL_CODE
                and file.is_file()
            ):
                found.add(str(file))
    return frozenset(found)


def _undeclared_imports(files) -> list:
    """Kernel files that the declared kernel files import and that are not
    declared themselves: sources someone forgot."""
    kernels = str(Path(__file__).resolve().parent / "kernels") + os.sep
    declared = set(files)
    missing = set()
    for f in files:
        if f.startswith(kernels) and f.endswith(".py"):
            missing |= _kernel_imports(f) - declared
    return sorted(missing)


@functools.cache
def _env_reads(path: str) -> frozenset:
    """The environment variables ``path`` reads by a literal name, anywhere
    in it (from its text). A name that is computed is not seen."""

    def is_environ(node) -> bool:
        return ast.unparse(node).endswith("environ")

    names = set()
    for node in ast.walk(ast.parse(Path(path).read_bytes())):
        arg = None
        if isinstance(node, ast.Call) and node.args:
            func = ast.unparse(node.func)
            if func.endswith(("environ.get", "environ.setdefault", "getenv")):
                arg = node.args[0]
        elif isinstance(node, ast.Subscript) and is_environ(node.value):
            arg = node.slice
        elif isinstance(node, ast.Compare) and any(map(is_environ, node.comparators)):
            arg = node.left
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            names.add(arg.value)
    return frozenset(names)


def _undeclared_env(files, declared) -> list:
    """Variables the declared sources read that are neither declared nor in
    the fingerprint nor known not to change code: inputs someone forgot."""
    missing = set()
    for f in files:
        if f.endswith(".py"):
            missing |= _env_reads(f)
    return sorted(
        name
        for name in missing - set(declared) - _ENV_NOT_CODE
        if not _env_in_fingerprint(name)
    )


def _launcher_signature(exe):
    if isinstance(exe, functools.partial):
        # A @flyc.jit method reached through its instance: the JitFunction is
        # the bound one, and its signature leaves ``self`` out, as ``args`` do.
        exe = getattr(exe.func, "__self__", None)
        if getattr(exe, "_sig", None) is None:
            raise RuntimeError("cannot read the signature of a bound launcher")
    sig = getattr(exe, "_sig", None)  # resolved by the compile that just ran
    if sig is None:
        try:
            sig = inspect.signature(getattr(exe, "func", exe), eval_str=True)
        except Exception as e:  # annotations that do not evaluate
            raise RuntimeError(f"cannot read the launcher's signature: {e}") from e
    return sig


def _constexpr_params(exe) -> list:
    """``(position, parameter)`` of the launcher's ``Constexpr[T]`` and
    ``Type[T]`` parameters: what a compile bakes into the object."""
    out = []
    for index, param in enumerate(_launcher_signature(exe).parameters.values()):
        ann = param.annotation
        if ann is inspect.Parameter.empty:
            continue
        if Constexpr.is_constexpr_annotation(ann) or is_type_param_annotation(ann):
            out.append((index, param))
    return out


def _constexpr_count(exe) -> int:
    return len(_constexpr_params(exe))


def _baked_args(exe, args) -> dict:
    """The launch arguments a build compiles into the object: ``Constexpr[T]``
    and ``Type[T]`` parameters of the launcher, by position, each as the
    ``repr`` of its normalized value.

    A loaded object ignores these arguments, so an entry serves only calls
    that pass the values it was built with (``_entry_current``).
    """
    baked = {}
    for index, param in _constexpr_params(exe):
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise RuntimeError(f"baked launch argument *{param.name} has no position")
        if param.kind is param.KEYWORD_ONLY or index >= len(args):
            # Left to its default, which the launcher's source (in the digest) fixes.
            continue
        try:
            baked[str(index)] = repr(_normalize_key(args[index]))
        except _UsageError as e:
            # A lambda or a type: nothing a later process can compare.
            raise RuntimeError(f"baked launch argument {param.name!r}: {e}") from None
    for index, value in enumerate(args):
        if _const_only(value) and str(index) not in baked:
            raise RuntimeError(
                f"launch argument {index} = {value!r} is not a Constexpr parameter"
            )
    return baked


def _baked_current(baked: dict, args) -> bool:
    for index, value in baked.items():
        i = int(index)
        if i >= len(args):
            return False
        try:
            if repr(_normalize_key(args[i])) != value:
                return False
        except _UsageError:
            return False
    return True


@contextlib.contextmanager
def _file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _normalize_key(value):
    """A process-independent form of ``key`` for the disk digest.

    The ``repr`` of anything else may hold a memory address and would make the
    digest differ in every process.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        # Entries are device independent: load_module loads every device.
        return ("device", value.type)
    if isinstance(value, (tuple, list)):
        return (type(value).__name__, *map(_normalize_key, value))
    if isinstance(value, dict):
        items = ((_normalize_key(k), _normalize_key(v)) for k, v in value.items())
        return ("dict", *sorted(items, key=repr))
    raise _UsageError(
        f"[FlyDSL AOT] unsupported key element {value!r} ({type(value).__name__})"
    )


def _arg_signature(args) -> tuple:
    """Specialization-relevant identity of the runtime arguments, without ``exe``."""
    sig = []
    for v in args:
        if isinstance(v, torch.Tensor):
            ctor, _ = JitArgumentRegistry.get(type(v))
            sig.append(repr(cache_signature(ctor(v))))
        elif hasattr(v, "__cache_signature__"):
            sig.append(repr(cache_signature(v)))
        elif isinstance(v, (bool, int, float)):
            sig.append(type(v).__name__)
        elif _const_only(v):
            # No runtime type: only a Constexpr parameter takes it, and the
            # build checks that it is one (_baked_args).
            sig.append("const")
        else:
            raise TypeError(f"[FlyDSL AOT] unsupported launch argument {v!r}")
    return tuple(sig)


def _const_only(v) -> bool:
    """Whether ``v`` can only be a constexpr launch argument: a string, None,
    or a tuple (of values the key can hold)."""
    return v is None or isinstance(v, (str, tuple))


def _placeholder(v):
    """CPU stand-in for a launch argument: same dtype, shape and strides.

    ``flyc.compile`` compiles CPU tensors without launching. CPU allocations
    are lazy, so untouched placeholder pages cost no memory.
    """
    if isinstance(v, torch.Tensor):
        from torch._subclasses.fake_tensor import unset_fake_temporarily

        with unset_fake_temporarily():
            return torch.empty_strided(
                tuple(v.shape), tuple(v.stride()), dtype=v.dtype, device="cpu"
            )
    if isinstance(v, PointerJitArg):
        # A non-null pointer counts as a live device argument and would launch.
        return PointerJitArg(v.element_type, None, v.address_space, v.alignment)
    if hasattr(v, "_is_stream_param"):
        return type(v)(None)
    return v


def _compile(exe, args):
    """Trace and compile ``exe`` for ``args`` without launching."""
    placeholders = [_placeholder(v) for v in args]
    cf = flyc.compile(exe, *placeholders, compile_only=True)
    if cf is None:
        # A hook stopped the compile, such as an AOT replay that only
        # identifies kernels.
        raise _NoExport("flyc.compile returned no function to export")
    return cf


def _ir_key(exe, cf) -> str | None:
    """FlyDSL's key of the compile that made ``cf``: a hash of the traced IR
    and of what lowers it (target, pipeline, options). Two compiles with one
    key had the same input, whatever bytes the code generator made of it.

    Private to FlyDSL (``JitFunction._last_compiled``); None if it is not
    there, and ``verify`` then compares objects.
    """
    jit = (
        getattr(exe.func, "__self__", None)
        if isinstance(exe, functools.partial)
        else exe
    )
    last = getattr(jit, "_last_compiled", None)
    try:
        key, artifact = last
    except (TypeError, ValueError):
        return None
    if isinstance(key, str) and artifact is getattr(cf, "_keepalive", None):
        return key
    return None


def _export(
    exe, args, out_dir: Path, sym: str, link_libs: set | None = None, info=None
) -> Path:
    """Compile ``exe`` for ``args`` without launching and export ``<sym>.o``.

    ``link_libs`` collects the bitcode files the kernel links (``link_extern``)
    and ``info`` (a dict) the compile's ``"ir"`` key.
    """
    cf = _compile(exe, args)
    if info is not None:
        info["ir"] = _ir_key(exe, cf)
    if link_libs is not None:
        # Private to FlyDSL (the CompiledArtifact a CompiledFunction keeps):
        # without it a linked library could not be checked against the
        # declared sources, so the entry is not written.
        try:
            libs = cf._keepalive._link_libs
        except AttributeError as e:
            raise RuntimeError(f"cannot read the kernel's linked bitcode: {e}") from e
        link_libs.update(os.path.realpath(lib) for lib in libs)
    cf.export_to_c(str(out_dir), sym)
    return out_dir / f"{sym}.o"


class _JitKernel:
    """The in-process FlyDSL path, for mode ``0`` and launches with no entry."""

    def __init__(self, exe, key):
        from .kernels.tensor_shim import _run_compiled  # imports this module

        self._exe = exe
        self._key = key
        self._run = _run_compiled

    def __call__(self, *args) -> None:
        self._run(self._exe, *args, specialization_key=self._key)


class _BakedKernel:
    """A loaded launcher that has constexpr arguments. The object ignores
    them, so a call passing other values than it was built with would run the
    wrong kernel; the JIT path's cached function has the same limit, silently.
    """

    __slots__ = ("_baked", "_kernel", "_min_args", "_nargs", "_where")

    def __init__(self, kernel, name, key, args, indexes):
        self._kernel = kernel
        self._where = f"{name}: {key!r}"
        # A call with more arguments may pass a constexpr this one left to
        # its default; one with fewer may leave a baked one out. Trailing
        # runtime arguments (a stream) may be left to their defaults.
        self._nargs = len(args)
        self._min_args = max(indexes, default=-1) + 1
        self._baked = tuple(
            (i, args[i], repr(_normalize_key(args[i]))) for i in indexes
        )

    def __call__(self, *args) -> None:
        if not self._min_args <= len(args) <= self._nargs:
            raise _BakedArgumentError(
                f"[FlyDSL AOT] {self._where} was compiled for {self._nargs} launch "
                f"arguments and is launched with {len(args)}; a launcher with "
                f"constexpr launch arguments takes the same ones at every launch "
                f"of one key"
            )
        for i, value, normalized in self._baked:
            arg = args[i]
            # The usual call passes the same object or an equal int, bool or
            # string; anything else (a tuple may hold True for 1, and -0.0
            # equals 0.0) is compared as the disk entry is.
            if arg is value or (
                type(arg) is type(value)
                and type(arg) in (int, bool, str)
                and arg == value
            ):
                continue
            try:
                same = repr(_normalize_key(arg)) == normalized
            except _UsageError:
                same = False
            if not same:
                raise _BakedArgumentError(
                    f"[FlyDSL AOT] {self._where} was compiled with constexpr "
                    f"launch argument {i} = {value!r} and is launched with "
                    f"{arg!r}; pass every constexpr launch argument in "
                    f"_run_compiled's specialization_key"
                )
        self._kernel(*args)


# The loading counterpart of export_to_c; FlyDSL releases without it run every
# key through the JIT path.
_load_module = getattr(flyc, "load_module", None)

_STALE_TMP_SECONDS = 3600


def _remove_stale_tmp(out_dir: Path) -> None:
    """Delete build directories left by processes that died mid-build.

    A build takes seconds; an hour-old one belongs to no live process.
    """
    for tmp in out_dir.glob(".tmp-*"):
        try:
            if time.time() - tmp.stat().st_mtime > _STALE_TMP_SECONDS:
                shutil.rmtree(tmp, ignore_errors=True)
        except OSError:
            pass


class _UsageError(TypeError):
    """A caller bug, such as an unsupported ``key``; never falls back."""


class _BakedArgumentError(RuntimeError):
    """One key launched with two values of a constexpr launch argument."""


class _UndeclaredSourceError(RuntimeError):
    """A cache's kernels are built from a file it does not declare; never
    falls back."""


# Set (under tensor_shim's preload lock) while a preload has COMPILE_ONLY on
# for the whole process to warm one launcher up: that is not a prebuild, and
# a launch of another thread must not build an entry in a long-lived process.
_preloading = False


def _prebuilding() -> bool:
    """Whether this launch is a prebuild's, which alone builds entries."""
    return fly_env.compile.compile_only and not _preloading


def _is_object(path: Path) -> bool:
    """Whether ``path`` starts like an ELF object; cheap, for a prebuild."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


class _SourceListError(RuntimeError):
    """A cache's ``sources=`` names a file that is not there. A launch takes
    the JIT path; a prebuild fails."""


class _Miss(RuntimeError):
    """No entry on disk for a launch; only a prebuild builds one."""


class _NoExport(RuntimeError):
    """A hook stopped the compile, so there is nothing to export: not a failure."""


class _StaleEntryError(RuntimeError):
    """``verify`` found a cached object that no longer matches its sources."""


# Cache name -> the file that created its cache, and the caches by name. Two
# caches of one name would share a directory and load each other's entries.
_NAMES = {}
_CACHES = {}


def _creation_file() -> str:
    """The file creating a cache, outside this module and ``aot_launch``: the
    same file again (a module reloaded) is the same cache."""
    here = (__file__, os.path.join(os.path.dirname(__file__), "aot_launch.py"))
    frame = sys._getframe(1)
    while frame is not None and frame.f_code.co_filename in here:
        frame = frame.f_back
    return frame.f_code.co_filename if frame else "?"


class FlyDSLAotCache:
    """Persistent ``export_to_c`` cache for one launcher family."""

    def __init__(self, name: str, sources=(), env=()):
        """``sources``: the files the launchers of this cache are built from,
        besides the one creating it and ``_COMMON_SOURCES``: each relative to ``aiter/ops/flydsl`` or
        absolute; a directory stands for every ``.py`` below it. ``env``: the environment variables those
        files read, besides ``_COMMON_ENV`` and what the fingerprint holds,
        and ``"get_gfx()"`` if their code picks by ``chip_info.get_gfx()``. An edit
        to a source, or another value of an input, makes new entries."""
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError(f"cache name must be a C identifier fragment: {name!r}")
        if isinstance(sources, (str, os.PathLike)):
            sources = (sources,)
        site = _creation_file()
        first = _NAMES.setdefault(name, site)
        if first != site:
            raise ValueError(
                f"FlyDSL AOT cache name {name!r} is already used in {first}; "
                f"a cache name must be unique"
            )
        _CACHES[name] = self
        self.name = name
        if os.path.isfile(site):  # not an interactive session
            sources = (site, *sources)
        # Resolved on first use (_files), not here: a module that creates a
        # cache must import whatever its sources list says, in every mode.
        self._declared = (*_COMMON_SOURCES, *sources)
        self._resolved = None
        env = {env} if isinstance(env, str) else set(env)
        self._env = tuple(sorted(env | set(_COMMON_ENV)))
        # [(source, sha256)] and its hash, once the files were read (_sources),
        # and the same hashes by path (_check_unchanged).
        self._hashes = None
        self._read = {}
        self._digest = None
        self._imports_checked = False
        self._mem = {}
        # Compile-only launchers, by key.
        self._jit = {}
        # Entries this process built; _prebuild skips the trace a build ran.
        self._builds = 0
        # Set by invalidate(): this process then runs every key through the JIT.
        self._jit_only = False
        self._jit_logged = False
        # Bumped by invalidate(), for those who keep a kernel from _mem.
        self.epoch = 0
        # Reentrant: a factory may reset itself (invalidate) while it is built.
        self._lock = threading.RLock()

    def directory(self) -> Path:
        return cache_root() / fingerprint() / self.name

    @property
    def _files(self) -> list:
        """The declared source files, as absolute paths."""
        if self._resolved is None:
            self._resolved = _source_files(self._declared)
        return self._resolved

    def _sources(self) -> str:
        """Hash of the declared sources, read once per process. An entry is
        looked up by it, and built under it only by a prebuild (``_resolve``),
        which checks before and after the build that the files still have
        these contents (``_check_unchanged``)."""
        if self._digest is None:
            self._read = {f: _file_digest(f) for f in self._files}
            gone = [f for f, d in self._read.items() if d is None]
            if gone:
                raise _SourceListError(f"declared source {gone[0]} cannot be read")
            self._hashes = sorted((_portable(f), d) for f, d in self._read.items())
            blob = json.dumps(self._hashes, sort_keys=True).encode()
            self._digest = hashlib.sha256(blob).hexdigest()
        return self._digest

    def _check_unchanged(self) -> None:
        """Raise if a declared source no longer has the contents the digest was
        made from. The digest is read once per process, but a kernel module
        may be imported later than that (a factory importing it on its first
        miss, a helper importing another while it traces): a build would then
        store the edited code under the digest of the earlier contents, for
        every later process to load."""
        self._sources()
        for f, read in self._read.items():
            try:
                now = hashlib.sha256(Path(f).read_bytes()).hexdigest()
            except OSError:
                now = None
            if now != read:
                raise RuntimeError(
                    f"{f} changed after this process hashed it; not caching "
                    f"a build that may hold either version of it"
                )

    def _inputs(self) -> dict:
        return {name: _input(name) for name in self._env}

    def _disk_key(self, key, args) -> str:
        ident = (
            self.name,
            _normalize_key(key),
            _arg_signature(args),
            self._sources(),
            tuple(self._inputs().items()),
        )
        # FlyDSL merges the caller's CompilationContext.compile_hints(...)
        # into the compile; the launcher's own hints come from its sources.
        hints = CompilationContext.get_compile_hints()
        if hints:
            try:
                ident += (_normalize_key(hints),)
            except _UsageError as e:
                raise TypeError(f"compile hints the cache cannot key: {e}") from None
        return hashlib.sha256(repr(ident).encode()).hexdigest()[:32]

    def invalidate(self) -> None:
        """Forget every launcher this process loaded, and run the cache's keys
        through the JIT path from now on.

        For a caller that resets its launcher factory (``cache_clear``) to
        retrace after changing state the cache does not track, such as a
        patched module global: an entry on disk would still be current, and a
        build would store the patched code under the unpatched sources.
        """
        with self._lock:
            self._jit_only = True
            self.epoch += 1
            self._mem.clear()
            self._jit.clear()

    def launch(self, key, make_exe, *args) -> None:
        """Launch the specialization ``key`` (built by ``make_exe()``) on ``args``.

        Under ``COMPILE_ONLY`` the entry is built but nothing is launched,
        even for a key this process already launched. Otherwise an entry is
        loaded if there is one and never built.
        """
        kernel = self._mem.get(key)
        # Reading the FlyDSL option costs about 0.8 us; the membership test 0.1 us.
        if kernel is None or "COMPILE_ONLY" in os.environ:
            if fly_env.compile.compile_only:
                self._prebuild(key, make_exe, args)
                return
            if kernel is None:
                with self._lock:
                    kernel = self._mem.get(key)
                    if kernel is None:
                        kernel = self._kernel(key, make_exe, args)
                        if kernel is None:
                            # COMPILE_ONLY was set meanwhile (another thread
                            # preloading): the entry is built, nothing launches.
                            return
                        self._mem[key] = kernel
        kernel(*args)

    def _prebuild(self, key, make_exe, args) -> None:
        """Build the entry for ``key`` without launching, e.g. in an AOT replay.

        Unless the export just traced the kernel, FlyDSL's own compile-only
        launch runs too, as on the JIT path: an AOT replay records it as the
        kernel the call requests, which an entry already on disk would
        otherwise hide, and it fills FlyDSL's cache for mode ``0`` and the
        fallback. The export's compile does both for a new entry.
        """
        with self._lock:
            # The export first: its compile is also the compile-only launch.
            if _MODE != "0" and _load_module is not None and not self._jit_only:
                builds = self._builds
                self._kernel(key, make_exe, args)
                if self._builds != builds:
                    return
            jit = self._jit.get(key)
            if jit is None:
                jit = self._jit[key] = _JitKernel(make_exe(), key)
        jit(*args)

    def _kernel(self, key, make_exe, args):
        if _MODE == "0":
            return _JitKernel(make_exe(), key)
        if _load_module is None or self._jit_only:
            if not self._jit_logged:
                self._jit_logged = True
                why = (
                    "this FlyDSL has no flyc.load_module"
                    if _load_module is None
                    else "its factory was reset (cache_clear)"
                )
                logger.error(
                    f"[FlyDSL AOT] {self.name}: {why}; launching through the "
                    f"JIT path in this process"
                )
            return _JitKernel(make_exe(), key)
        try:
            return self._resolve(key, make_exe, args)
        except _NoExport:
            # e.g. an AOT replay that only identifies the kernels a call
            # requests: the JIT path's compile-only launch is what it watches.
            return _JitKernel(make_exe(), key)
        except _Miss as e:
            # Not a failure, in any mode: the prebuild did not cover this key
            # (another shape, CU count or environment).
            logger.error(
                f"[FlyDSL AOT] {self.name}: {e} for {key!r}; tracing it on "
                f"the JIT path. Run the AOT prebuild to add it."
            )
            return _JitKernel(make_exe(), key)
        except (TypeError, ValueError, NotImplementedError, RuntimeError, OSError) as e:
            if (
                _MODE == "verify"
                or isinstance(
                    e, (_StaleEntryError, _UsageError, _UndeclaredSourceError)
                )
                or (isinstance(e, _SourceListError) and _prebuilding())
            ):
                raise
            logger.error(
                f"[FlyDSL AOT] {self.name}: no exported entry for {key!r} ({e}); "
                f"using the JIT path for it"
            )
            return _JitKernel(make_exe(), key)

    def _resolve(self, key, make_exe, args):
        """Load the entry for ``key`` from disk.

        A hit reads only the object, its sidecar and the declared sources; it
        never calls ``make_exe``, whose factory runs FlyDSL's AST rewrite.
        An entry is built only under ``COMPILE_ONLY``, i.e. by a prebuild: a
        process that launches kernels never writes the cache, so what it
        imported long ago cannot end up in an entry (``_Miss``).
        """
        digest = self._disk_key(key, args)
        sym = f"fly_{self.name}_{digest[:16]}"
        out_dir = self.directory()
        prebuild = _prebuilding()
        found = self._find(out_dir, digest, args)
        if found is not None and prebuild and not _is_object(found.with_suffix(".o")):
            found = None  # truncated or foreign: built again below
        if found is None:
            if not prebuild:
                raise _Miss(f"no prebuilt entry in {out_dir}")
            with _file_lock(out_dir / f"{digest}.lock"):
                found = self._find(out_dir, digest, args)
                if found is None or not _is_object(found.with_suffix(".o")):
                    found = self._build(make_exe, args, out_dir, sym, digest)
        elif _MODE == "verify":
            self._verify(make_exe(), args, sym, found)
        if fly_env.compile.compile_only:
            return None
        obj = found.with_suffix(".o")
        try:
            kernel = _load_module(obj)
            meta = json.loads(found.with_suffix(".json").read_text())
            baked = meta["baked"]
        except Exception as e:  # whatever a bad object makes the loader raise
            # Not removed here: the error may not be the file's (no file
            # handles left, a missing runtime library), and a launch never
            # writes the cache. A prebuild replaces an object that is not one.
            raise RuntimeError(f"loading {obj} failed ({type(e).__name__}: {e})") from e
        # Also for a launcher whose constexprs were all left to their defaults:
        # a later launch that passes one must not run this object.
        if baked or meta.get("constexpr"):
            return _BakedKernel(kernel, self.name, key, args, sorted(map(int, baked)))
        return kernel

    @staticmethod
    def _find(out_dir: Path, digest: str, args):
        """The stem of the entry of ``digest`` that bakes the constexpr values
        ``args`` pass.

        The digest holds the declared sources and inputs, so switching a
        kernel file or a variable back and forth (e.g. between branches)
        finds the earlier build again instead of overwriting it.
        """
        for sidecar in sorted(out_dir.glob(f"{digest}-*.json")):
            stem = sidecar.with_suffix("")
            try:
                meta = json.loads(sidecar.read_text())
                current = _baked_current(meta["baked"], args)
            except (OSError, ValueError, KeyError, AttributeError, TypeError):
                continue  # unreadable, or not a sidecar this version wrote
            if current and stem.with_suffix(".o").exists():
                return stem
        return None

    def _build(self, make_exe, args, out_dir, sym, digest):
        out_dir.mkdir(parents=True, exist_ok=True)
        _remove_stale_tmp(out_dir)
        tmp = Path(tempfile.mkdtemp(dir=out_dir, prefix=".tmp-"))
        try:
            self._check_declared()
            # Before the trace, which a changed source makes pointless, and
            # after it: the trace is what imports the kernel's modules.
            self._check_unchanged()
            libs = set()
            info = {}
            exe = make_exe()
            tmp_obj = _export(exe, args, tmp, sym, libs, info)
            self._check_declared(libs)
            self._check_unchanged()
            meta = {
                # In the digest already; kept here to say what an entry is.
                "sources": dict(self._hashes),
                "env": self._inputs(),
                # What the code generator was given; verify compares it.
                "ir": info.get("ir"),
                "baked": _baked_args(exe, args),
                "constexpr": _constexpr_count(exe),
            }
            variant = hashlib.sha256(json.dumps(meta).encode()).hexdigest()[:12]
            stem = out_dir / f"{digest}-{variant}"
            sidecar = tmp / "meta.json"
            sidecar.write_text(json.dumps(meta, indent=1))
            # The object is published last; _find needs both files.
            os.replace(sidecar, stem.with_suffix(".json"))
            os.replace(tmp_obj, stem.with_suffix(".o"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        self._builds += 1
        logger.info(f"[FlyDSL AOT] built {stem}.o")
        return stem

    def _check_declared(self, libs=()) -> None:
        """Raise if this cache is built from something it does not declare: a
        kernel file its declared kernel files import, or bitcode its kernel
        links. Run when an entry is built, so a hit pays nothing; the AOT
        replays build every cache, and a unit test runs the import check
        over every cache without building."""
        if not self._imports_checked:
            missing = _undeclared_imports(self._files)
            if missing:
                raise _UndeclaredSourceError(
                    f"[FlyDSL AOT] cache {self.name!r} does not declare "
                    f"{[_portable(f) for f in missing]}, which its sources import: "
                    f"add them to its sources="
                )
            unread = _undeclared_env(self._files, self._env)
            if unread:
                raise _UndeclaredSourceError(
                    f"[FlyDSL AOT] cache {self.name!r} does not declare the "
                    f"variables {unread}, which its sources read: add them to "
                    f"its env= (or, if one cannot change generated code, to "
                    f"aot_cache._ENV_NOT_CODE)"
                )
            self._imports_checked = True
        missing = sorted(set(libs) - set(self._files))
        if missing:
            raise _UndeclaredSourceError(
                f"[FlyDSL AOT] cache {self.name!r} links {missing}, which it "
                f"does not declare: add the bitcode to its sources="
            )

    def _verify(self, exe, args, sym, stem) -> None:
        """Raise if the entry at ``stem`` is not what ``exe`` compiles to now.

        Compared by FlyDSL's key of the traced IR when the entry has one: the
        code generator can make different bytes of one input (seen: one
        gfx950 GEMM in about seven compiles), so comparing objects would call
        a current entry stale. Without it, the objects are compared.
        """
        obj = stem.with_suffix(".o")
        built = json.loads(stem.with_suffix(".json").read_text()).get("ir")
        if built is not None:
            now = _ir_key(exe, _compile(exe, args))
            if now is not None:
                if now == built and obj.exists():
                    return
                raise _StaleEntryError(
                    f"[FlyDSL AOT] stale cache entry {obj}: the kernel traces "
                    f"to other IR now ({now[:16]}, built from {built[:16]}). "
                    f"Something the key does not cover changed the kernel; "
                    f"delete {obj.parent} or add the input to the cache key."
                )
        with tempfile.TemporaryDirectory() as tmp:
            fresh = _export(exe, args, Path(tmp), sym).read_bytes()
        if not obj.exists() or obj.read_bytes() != fresh:
            raise _StaleEntryError(
                f"[FlyDSL AOT] stale cache entry {obj}: re-exported object differs. "
                f"Something the fingerprint does not cover changed the kernel; "
                f"delete {obj.parent} or add the input to the cache key."
            )
