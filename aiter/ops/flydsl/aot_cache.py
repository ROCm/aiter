# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Persistent, trace-free FlyDSL kernel cache built on ``export_to_c``.

FlyDSL 0.5.0 keys its artifacts by the traced IR, so every first call of a
specialization traces the Python kernel again in each new process, even when
its disk cache hits. This module skips that trace on later runs. On a miss it
compiles the specialization once with CPU placeholders and exports it with
``CompiledFunction.export_to_c``. Every process, including the one that built
it, loads the object with ``flyc.load_module``, so a hit never traces.

See ``docs/flydsl_aot_cache.md`` for the design and how cache entries are
invalidated.

Usage::

    _CACHE = FlyDSLAotCache("preshuffle_gemm_a8")
    _CACHE.launch(key, lambda: compile_fn(**kw), *launcher_args)

``launch`` replaces ``_run_compiled(compile_fn(**kw), *launcher_args)`` and
covers every mode, including ``0``. Within one cache ``name``, ``key`` must
determine the specialization: the launcher and every constexpr it is built
with. It is made of ints, floats, bools, strings, ``None``, torch dtypes and
devices, and tuples, lists and dicts of those. The disk key also hashes the
``cache_signature`` of each runtime argument (dtype, rank, unit-stride axis),
so another process cannot load an object compiled for other argument types.
In memory, entries are looked up by ``key`` alone, so ``key`` must also hold
every tensor dtype and layout that can vary, as for ``_run_compiled``'s
``specialization_key``.

If an entry cannot be exported or loaded (an argument type ``export_to_c``
does not support), ``launch`` logs a warning once and runs that key through
``_run_compiled``. A FlyDSL without ``flyc.load_module`` runs every key that
way.

Invalidation has two levels. The directory is named by a fingerprint of the
toolchain (FlyDSL, HIP, Python), the target (arch, CU count) and the
``AITER_FLYDSL_*`` / ``FLYDSL_*`` environment. Each entry also records the
Python source files that ran while it was built (the factory and the trace)
and the aiter modules that code imports values from, with their hashes; the
environment variables the build-feeding code of those files names or that
aiter code read during the build; and ``get_gfx()`` when the build used it.
A hit whose sources or inputs changed is rebuilt. Editing one kernel, or setting a variable one
kernel reads, therefore invalidates only the entries that depend on it.

``AITER_FLYDSL_AOT_CACHE`` selects the mode:

* ``1`` (default): on.
* ``0``: off. ``launch`` uses ``_run_compiled``.
* ``verify``: like ``1``, but each disk hit is re-traced and re-exported, and
  the new object must match the cached ``.o`` byte for byte. Export and load
  failures raise instead of falling back.

``AITER_FLYDSL_AOT_CACHE_DIR`` sets the cache root. The default is
``flydsl_aot`` in aiter's JIT directory (``get_user_jit_dir``), so entries a
prebuild writes ship with the package, like its JIT modules.
"""

from __future__ import annotations

import _collections_abc
import ast
import contextlib
import dis
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
from flydsl.compiler.jit_argument import PointerJitArg
from flydsl.compiler.protocol import cache_signature
from flydsl.utils import env as fly_env

from aiter import logger

__all__ = ["FlyDSLAotCache", "aot_cache_mode", "cache_root", "fingerprint"]

_SCHEMA = 5
_MODE = os.environ.get("AITER_FLYDSL_AOT_CACHE", "1").strip().lower()
if _MODE not in ("0", "1", "verify"):
    raise ValueError(f"AITER_FLYDSL_AOT_CACHE must be 0, 1 or verify, got {_MODE!r}")

# Environment variables that do not change generated code. Every other
# AITER_FLYDSL_* / FLYDSL_* variable goes into the fingerprint, because kernels
# and FlyDSL read them while tracing.
_ENV_IGNORED = {
    "AITER_FLYDSL_AOT_CACHE",
    "AITER_FLYDSL_AOT_CACHE_DIR",
    "FLYDSL_DUMP_IR",
    "FLYDSL_DUMP_DIR",
    "FLYDSL_RUNTIME_CACHE_DIR",
    "FLYDSL_RUNTIME_ENABLE_CACHE",
    # The replay's --check sets it; it must look up the entries a build made.
    "FLYDSL_RUNTIME_RUN_ONLY",
    "AITER_FLYDSL_AOT_MAX_RETRIES",
    "AITER_FLYDSL_AOT_TIMEOUT",
    "AITER_FLYDSL_AOT_TRACEBACK",
    # Hashed as the arch it selects (_target_arch), so that an AOT build, which
    # sets it, and the runtime, which reads the device, share a directory.
    "FLYDSL_GPU_ARCH",
}
_ENV_EXTRA = ("HSA_OVERRIDE_GFX_VERSION", "COMPILE_OPT_LEVEL")
# Variables that describe the build rather than the code it makes. The AOT dry
# run sets them (CU_NUM and GPU_ARCHS to the target, hashed in the fingerprint
# as _cu_num and _target_arch); recording them per entry would keep the
# runtime, which leaves them unset, from hitting what the dry run built.
# AITER_AOT_* are set by setup.py and the replay; AITER_CONFIG_* point the
# replay at the CSV it replays, and the config it selects is in the key.
_ENV_SETUP = {"AITER_GPU_ARCHS", "COMPILE_ONLY", "CU_NUM", "GPU_ARCHS"}
_ENV_SETUP_PREFIXES = ("AITER_AOT_", "AITER_CONFIG_")
# A string literal read as a variable name: os.environ.get("X"), os.getenv("X"),
# os.environ["X"], "X" in os.environ.
_ENV_NAME_RE = re.compile(
    r"""(?:environ\.get|getenv|environ\.setdefault)\(\s*["']([A-Za-z_]\w*)["']"""
    r"""|environ\[\s*["']([A-Za-z_]\w*)["']\s*\]"""
    r"""|["']([A-Za-z_]\w*)["']\s+(?:not\s+)?in\s+os\.environ"""
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
    """Whether ``fingerprint`` owns ``name``: hashes it or, if in ``_ENV_IGNORED``,
    knows it changes no code."""
    return name.startswith(("AITER_FLYDSL_", "FLYDSL_")) or name in _ENV_EXTRA


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


def fingerprint() -> str:
    """Hash of the toolchain and environment every entry depends on.

    It covers FlyDSL (version, Python sources, native libraries), torch HIP,
    Python, the target arch and CU count, and the ``AITER_FLYDSL_*`` /
    ``FLYDSL_*`` environment variables. Changing any of them selects a new
    cache directory. Kernel sources and the other variables they read are
    tracked per entry instead (``_record_sources``). Only the FlyDSL hash is
    cached: an AOT build switches arch and environment within one process.
    """
    env_items = sorted(
        (k, v)
        for k, v in os.environ.items()
        if _env_in_fingerprint(k) and k not in _ENV_IGNORED
    )
    parts = [
        _SCHEMA,
        sys.version_info[:2],
        platform.machine(),
        flydsl.__version__,
        torch.version.hip,
        _target_arch(),
        _cu_num(),
        _toolchain_digest(),
        env_items,
    ]
    return hashlib.sha256(repr(parts).encode()).hexdigest()[:16]


@functools.cache
def _untracked_roots() -> tuple:
    """Source trees whose files are not recorded as entry dependencies.

    FlyDSL is covered by the fingerprint; torch, the standard library and
    third-party packages change only with an environment change.
    """
    import sysconfig

    roots = [Path(flydsl.__file__).parent, Path(torch.__file__).parent]
    roots += [Path(sysconfig.get_paths()[k]) for k in ("stdlib", "platstdlib")]
    return tuple(str(r.resolve()) + os.sep for r in roots)


@functools.cache
def _tracked_root() -> str:
    import aiter

    return str(Path(aiter.__file__).parent.resolve()) + os.sep


def _portable(path: str) -> str:
    """``path`` as a sidecar stores it: relative to the aiter package if inside
    it, so entries built in a source tree still match once the package is
    installed elsewhere (a prebuilt wheel); absolute otherwise."""
    return path.removeprefix(_tracked_root())


def _is_tracked(path: str) -> bool:
    if not path.endswith(".py") or path == os.path.realpath(__file__):
        return False
    if path.startswith(_tracked_root()):
        # aiter/aot holds build tooling; the replay's hooks run during the trace.
        return not path.startswith(_tracked_root() + "aot" + os.sep)
    if path.startswith(_untracked_roots()):
        return False
    return "site-packages" not in path and "dist-packages" not in path


@functools.cache
def _process_start_ns() -> int:
    """When this process started, to a clock tick: ``/proc/self/stat`` gives it
    in ticks since boot."""
    with open("/proc/self/stat") as f:
        ticks = int(f.read().rpartition(")")[2].split()[19])
    since_boot = ticks * 10**9 // os.sysconf("SC_CLK_TCK")
    return time.time_ns() - time.clock_gettime_ns(time.CLOCK_BOOTTIME) + since_boot


def _changed_since_start(files) -> list:
    """Sources modified after this process started. The process may run the code
    it imported before the change, so an entry built now could hold the old code
    under the new file's hash."""
    start = _process_start_ns()
    changed = []
    for f in files:
        with contextlib.suppress(OSError):
            if os.stat(f).st_mtime_ns > start:
                changed.append(f)
    return changed


@functools.cache
def _file_digest(path: str) -> str | None:
    """Content hash of a source file, once per process; None if it is gone."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def _first_line(node) -> int:
    """The line a function's code object starts at: its first decorator."""
    return min([node.lineno] + [d.lineno for d in node.decorator_list])


def _cached(node) -> bool:
    """Whether a function is memoized, so a later call need not enter it."""
    for d in node.decorator_list:
        d = d.func if isinstance(d, ast.Call) else d
        name = d.attr if isinstance(d, ast.Attribute) else getattr(d, "id", "")
        if name in ("cache", "lru_cache"):
            return True
    return False


def _build_spans(body, called, at_import=True) -> list:
    """Line spans of ``body`` that can feed a build: statements run at import,
    functions called in it, and memoized functions, which may have run before."""
    spans = []
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _first_line(node) in called or _cached(node):
                spans.append((_first_line(node), node.end_lineno))
            else:
                spans += _build_spans(node.body, called, False)
        elif isinstance(node, ast.ClassDef):
            spans += _build_spans(node.body, called, at_import)
        elif at_import:
            spans.append((node.lineno, node.end_lineno))
    return spans


@functools.cache
def _parse(path: str):
    """``(lines, tree)`` of a source file, once per build; None if unreadable."""
    try:
        text = Path(path).read_text(errors="replace")
        return text.splitlines(), ast.parse(text)
    except (OSError, SyntaxError, ValueError):
        return None


def _build_text(path: str, called: frozenset) -> str:
    """The code of ``path`` that can feed a build (``_build_spans``)."""
    parsed = _parse(path)
    if parsed is None:
        return ""
    lines, tree = parsed
    return "\n".join(
        "\n".join(lines[a - 1 : b]) for a, b in _build_spans(tree.body, called)
    )


def _env_names(path: str, called: frozenset) -> frozenset:
    """Environment variables ``path`` reads by a literal name in the code that
    can feed a build. Host-only functions the build never called, such as an
    allocator or IPC setup, are skipped."""
    return frozenset(
        n
        for m in _ENV_NAME_RE.finditer(_build_text(path, called))
        for n in m.groups()
        if n
    )


def _module_file(name: str) -> str | None:
    """The source of aiter module ``name``, e.g. ``aiter.ops.flydsl.x``."""
    parts = name.split(".")
    if parts[0] != "aiter" or not all(parts):
        return None
    base = os.path.join(_tracked_root(), *parts[1:])
    for f in (base + ".py", os.path.join(base, "__init__.py")):
        if os.path.isfile(f):
            return f
    return None


def _import_base(path: str, node: ast.ImportFrom) -> str:
    """The absolute module ``node`` imports from, in file ``path``."""
    if node.level == 0:
        return node.module or ""
    rel = _portable(path)
    if os.path.isabs(rel):
        return ""
    package = ["aiter"] + rel.split(os.sep)[:-1]
    package = package[: len(package) - node.level + 1]
    return ".".join(package + ([node.module] if node.module else []))


def _bound_names(node) -> set:
    """The module-level names statement ``node`` binds or mutates
    (``X[k] = ...``, ``X.a = ...``), not counting the statements nested in it."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {(a.asname or a.name).split(".")[0] for a in node.names}
    names = set()
    for c in _header(node):
        for n in ast.walk(c):
            if not isinstance(getattr(n, "ctx", None), ast.Store):
                continue
            while isinstance(n, (ast.Subscript, ast.Attribute)):
                n = n.value
            if isinstance(n, ast.Name):
                names.add(n.id)
    return names


def _header(node) -> list:
    """The expressions of statement ``node``, without its nested statements:
    the condition of an ``if``, the iterable of a ``for``, an assignment."""
    return [
        c
        for c in ast.iter_child_nodes(node)
        if not isinstance(c, (ast.stmt, ast.ExceptHandler))
    ]


def _module_statements(body, outer=()):
    """Statements run at import, inside ``if``/``try``/``with`` blocks too,
    each with the compound statements around it."""
    for node in body:
        yield node, outer
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        for c in ast.iter_child_nodes(node):
            if isinstance(c, ast.ExceptHandler):
                yield from _module_statements(c.body, outer + (node,))
            elif isinstance(c, ast.stmt):
                yield from _module_statements([c], outer + (node,))


@functools.cache
def _bindings(path: str) -> dict:
    """Module-level name -> ``(statement, enclosing statements)`` of ``path``
    that bind it. ``"*"`` holds the star imports."""
    parsed = _parse(path)
    out = {}
    if parsed is not None:
        for node, outer in _module_statements(parsed[1].body):
            for name in _bound_names(node):
                out.setdefault(name, []).append((node, outer))
    return out


def _star_files(path: str) -> list:
    """The aiter modules ``path`` star-imports."""
    return [
        f
        for node, _ in _bindings(path).get("*", ())
        if (f := _module_file(_import_base(path, node))) is not None
    ]


def _defines(path: str, name: str, seen: frozenset = frozenset()) -> bool:
    """Whether ``path`` binds ``name``, itself or through its star imports."""
    if path in seen:
        return False
    if name in _bindings(path):
        return True
    return any(_defines(f, name, seen | {path}) for f in _star_files(path))


def _binds_logger(node) -> bool:
    """Whether ``node`` is ``x = logging.getLogger(...)``: a logger does not
    change a kernel, and ``from aiter import logger`` would otherwise tie
    every entry to ``aiter/__init__.py``."""
    v = getattr(node, "value", None)
    f = v.func if isinstance(v, ast.Call) else None
    return (
        isinstance(f, (ast.Attribute, ast.Name))
        and (getattr(f, "attr", None) or getattr(f, "id", None)) == "getLogger"
    )


def _value_sources(path: str, name: str, seen: set, run: bool = False) -> set:
    """The files module-level ``name`` of ``path`` takes its value from.

    A function the build calls is recorded by the profile hook, so it adds
    nothing unless memoized (a cached call does not enter it) or ``run``: it
    was called at import to compute a value. Any other value adds ``path`` and,
    through the names its statement and the conditions around it read, what
    it was computed from.
    """
    if (path, name, run) in seen:
        return set()
    seen.add((path, name, run))
    nodes = _bindings(path).get(name)
    if not nodes:
        # A star re-export, or bound in a way this does not follow.
        out = {path}
        for f in _star_files(path):
            if _defines(f, name):
                out |= _value_sources(f, name, seen, run)
        return out
    out = set()
    for node, outer in nodes:
        if isinstance(node, ast.ImportFrom):
            base = _import_base(path, node)
            for a in node.names:
                if (a.asname or a.name) == name:
                    out |= _imported(base, a.name, seen, run)
            continue
        if isinstance(node, ast.Import):
            for a in node.names:
                if (a.asname or a.name).split(".")[0] == name:
                    out.add(_module_file(a.name))
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not (run or _cached(node)):
                continue
            reads = [node]
        elif isinstance(node, ast.ClassDef):
            reads = [node]
        elif _binds_logger(node):
            continue
        else:
            reads = _header(node)
        out.add(path)
        bound = _bindings(path)
        for expr in reads + [h for o in outer for h in _header(o)]:
            for n in ast.walk(expr):
                if isinstance(n, ast.Name) and n.id in bound and n.id != name:
                    out |= _value_sources(path, n.id, seen, True)
    out.discard(None)
    return out


def _imported(module: str, name: str, seen: set, run: bool = False) -> set:
    """The files ``from module import name`` takes a value from."""
    sub = _module_file(f"{module}.{name}")
    if sub is not None:
        return {sub}
    f = _module_file(module)
    if f is None:
        return set()
    if name == "*":
        return {f}
    return _value_sources(f, name, seen, run)


def _import_sources(files, called: dict) -> set:
    """Aiter modules whose module-level values the code of ``files`` imports.

    Reading an imported constant calls nothing, so the profile hook does not
    see the module that defines it. Static, and on the generous side: an extra
    file costs a rebuild when it is edited, a missing one a stale kernel.
    """
    out = set()
    seen = set()
    for path in files:
        parsed = _parse(path)
        if parsed is None:
            continue
        spans = _build_spans(parsed[1].body, frozenset(called.get(path, ())))
        for node in ast.walk(parsed[1]):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if not any(a <= node.lineno <= b for a, b in spans):
                continue
            if isinstance(node, ast.Import):
                out.update(_module_file(a.name) for a in node.names)
                continue
            base = _import_base(path, node)
            for a in node.names:
                out |= _imported(base, a.name, seen)
    out.discard(None)
    out = {os.path.realpath(f) for f in out}
    return {f for f in out if _is_tracked(f)} - set(files)


def _clear_get_gfx():
    from aiter.jit.utils.chip_info import get_gfx

    get_gfx.cache_clear()


def _get_gfx():
    from aiter.jit.utils.chip_info import get_gfx

    try:
        return get_gfx()
    except Exception:  # noqa: BLE001  no GPU and no GPU_ARCHS
        return None


# Values aiter derives from the environment, kept with the variables an entry
# depends on. get_gfx() honours GPU_ARCHS (its last arch when it names
# several), while the fingerprint's arch is the device's or FlyDSL's target:
# an image setting GPU_ARCHS="gfx942;gfx950" answers gfx950 on a gfx942.
_DERIVED = {"get_gfx()": _get_gfx}


def _calls_get_gfx(path: str) -> bool:
    """Whether code of ``path`` that may have run before a build calls
    get_gfx(): module-level statements and memoized functions."""
    parsed = _parse(path)
    if parsed is None:
        return False
    spans = _build_spans(parsed[1].body, frozenset())
    for node in ast.walk(parsed[1]):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
            if name == "get_gfx" and any(a <= node.lineno <= b for a, b in spans):
                return True
    return False


def _uses_get_gfx(files, calls) -> bool:
    """Whether the build called get_gfx(), directly or through a value
    computed before it. Its memo is cleared before each build, so a call
    during the build enters it and the profile hook sees it."""
    from aiter.jit.utils import chip_info

    code = inspect.unwrap(chip_info.get_gfx).__code__
    if (code.co_filename, code.co_firstlineno) in calls:
        return True
    return any(_calls_get_gfx(f) for f in files)


def _input(name: str):
    """The current value of an entry's input: a variable or a ``_DERIVED`` one."""
    return _DERIVED[name]() if name in _DERIVED else os.environ.get(name)


# os.getenv, os.environ.get and `in` reach _Environ.__getitem__ through these.
# Taken from the code objects: on Python 3.11+ both modules are frozen, and
# their co_filename is "<frozen os>" rather than os.__file__.
_ENV_PLUMBING = frozenset(
    {os.getenv.__code__.co_filename, _collections_abc.Mapping.get.__code__.co_filename}
)
# The plumbing that reads one variable by name. copy(), items(), dict(os.environ)
# and the like read every variable; they are not a dependency on each, and
# recording them would write the whole environment, secrets included, into
# the entry.
_ENV_BY_NAME = frozenset({"get", "getenv", "__contains__", "setdefault"})


def _subscript(frame) -> bool:
    """Whether ``frame`` is at ``os.environ[...]``, not a C-level bulk read."""
    for ins in dis.get_instructions(frame.f_code):
        if ins.offset == frame.f_lasti:
            return ins.opname == "BINARY_SUBSCR" or (
                ins.opname == "BINARY_OP" and ins.argrepr == "[]"
            )
    return False


@contextlib.contextmanager
def _record_sources(out: set, env: set, calls: set):
    """Collect the files of every Python function called in this block, the
    ``(file, first line)`` of each, and the environment variables aiter code
    reads in it.

    Run around the factory and the trace, it names the sources the generated
    code can depend on: the kernel, its helpers and any aiter code they call.
    A variable read under a computed name is caught here; one read at import or
    inside a cached function is caught only if kernel code names it
    (``_entry_env``). Reads of the whole environment are not recorded.
    """
    seen = set()
    reads = set()
    getitem = os._Environ.__getitem__.__code__

    def prof(frame, event, _arg):
        if event != "call":
            return
        seen.add(frame.f_code.co_filename)
        calls.add((frame.f_code.co_filename, frame.f_code.co_firstlineno))
        if frame.f_code is getitem:
            caller = frame.f_back
            by_name = True
            while caller is not None and caller.f_code.co_filename in _ENV_PLUMBING:
                by_name &= caller.f_code.co_name in _ENV_BY_NAME
                caller = caller.f_back
            if caller is frame.f_back:
                by_name = caller is not None and _subscript(caller)
            if caller is not None and by_name:
                reads.add((caller.f_code.co_filename, frame.f_locals.get("key")))

    old = sys.getprofile()
    sys.setprofile(prof)
    try:
        yield
    finally:
        sys.setprofile(old)
        out.update(os.path.realpath(f) for f in seen if _is_tracked(f))
        env.update(
            k
            for f, k in reads
            if isinstance(k, str) and _is_tracked(os.path.realpath(f))
        )


def _entry_env(files, read, calls, called) -> dict:
    """The variables an entry built from ``files`` depends on, with their values.

    Names are taken from the text of kernel code only (``aiter/ops/flydsl``),
    and there only from code that can feed the build (``_env_names``).
    Shared modules such as ``aiter/jit/core.py`` name many variables that do
    not affect a kernel, so for them only the reads seen during the build count.
    ``get_gfx()`` is added as an input when the build used it (``_uses_get_gfx``).
    """
    kernel_root = _tracked_root() + "ops" + os.sep + "flydsl" + os.sep
    names = set(read).union(
        *(
            _env_names(f, frozenset(called.get(f, ())))
            for f in files
            if f.startswith(kernel_root)
        )
    )
    env = {
        k: os.environ.get(k)
        for k in sorted(names)
        if not _env_in_fingerprint(k)
        and k not in _ENV_SETUP
        and not k.startswith(_ENV_SETUP_PREFIXES)
    }
    if _uses_get_gfx(files, calls):
        env["get_gfx()"] = _get_gfx()
    return env


def _called_lines(calls) -> dict:
    """Real path -> first lines of the functions ``calls`` holds in it."""
    by_file = {}
    for f, line in calls:
        by_file.setdefault(f, set()).add(line)
    called = {}
    for f, lines in by_file.items():
        called.setdefault(os.path.realpath(f), set()).update(lines)
    return called


def _entry_current(meta: dict) -> bool:
    root = _tracked_root()
    sources = meta["sources"].items()
    # os.path.join keeps an absolute path as it is.
    return all(_file_digest(os.path.join(root, p)) == d for p, d in sources) and all(
        _input(k) == v for k, v in meta["env"].items()
    )


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
        else:
            raise TypeError(f"[FlyDSL AOT] unsupported launch argument {v!r}")
    return tuple(sig)


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


def _export(exe, args, out_dir: Path, sym: str) -> Path:
    """Compile ``exe`` for ``args`` without launching and export ``<sym>.o``."""
    placeholders = [_placeholder(v) for v in args]
    cf = flyc.compile(exe, *placeholders, compile_only=True)
    if cf is None:
        # A hook stopped the compile, such as an AOT replay that only
        # identifies kernels.
        raise RuntimeError("flyc.compile returned no function to export")
    cf.export_to_c(str(out_dir), sym)
    return out_dir / f"{sym}.o"


class _JitKernel:
    """The in-process FlyDSL path, for mode ``0`` and entries that cannot export."""

    def __init__(self, exe, key):
        self._exe = exe
        self._key = key

    def __call__(self, *args) -> None:
        from .kernels.tensor_shim import _run_compiled

        _run_compiled(self._exe, *args, specialization_key=self._key)


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


class _StaleEntryError(RuntimeError):
    """``verify`` found a cached object that no longer matches its sources."""


class FlyDSLAotCache:
    """Persistent ``export_to_c`` cache for one launcher family."""

    def __init__(self, name: str, extra_sources=()):
        if not re.fullmatch(r"[A-Za-z0-9_]+", name):
            raise ValueError(f"cache name must be a C identifier fragment: {name!r}")
        self.name = name
        # Files a kernel depends on without calling into them while tracing,
        # e.g. a module whose constants it imports. Hashed into every entry.
        self._extra = tuple(os.path.realpath(p) for p in extra_sources)
        self._mem = {}
        # Compile-only launchers, by key.
        self._jit = {}
        # Entries this process built; _prebuild skips the trace a build ran.
        self._builds = 0
        self._lock = threading.Lock()

    def directory(self) -> Path:
        return cache_root() / fingerprint() / self.name

    def _disk_key(self, key, args) -> str:
        ident = repr((self.name, _normalize_key(key), _arg_signature(args)))
        return hashlib.sha256(ident.encode()).hexdigest()[:32]

    def launch(self, key, make_exe, *args) -> None:
        """Launch the specialization ``key`` (built by ``make_exe()``) on ``args``.

        Under ``COMPILE_ONLY`` the entry is built but nothing is launched,
        even for a key this process already launched.
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
                        kernel = self._mem[key] = self._kernel(key, make_exe, args)
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
            # The export first: a trace FlyDSL already cached in this process
            # would hide the sources the entry records.
            if _MODE != "0" and _load_module is not None:
                builds = self._builds
                self._kernel(key, make_exe, args)
                if self._builds != builds:
                    return
            jit = self._jit.get(key)
            if jit is None:
                jit = self._jit[key] = _JitKernel(make_exe(), key)
        jit(*args)

    def _kernel(self, key, make_exe, args):
        if _MODE == "0" or _load_module is None:
            return _JitKernel(make_exe(), key)
        try:
            return self._resolve(key, make_exe, args)
        except (TypeError, ValueError, NotImplementedError, RuntimeError, OSError) as e:
            if _MODE == "verify" or isinstance(e, (_StaleEntryError, _UsageError)):
                raise
            logger.warning(
                f"[FlyDSL AOT] {self.name}: no exported entry for {key!r} ({e}); "
                f"using the JIT path for it"
            )
            return _JitKernel(make_exe(), key)

    def _resolve(self, key, make_exe, args):
        """Load the entry for ``key`` from disk, building it first on a miss.

        A hit reads only the object, its sidecar and the recorded sources; it
        never calls ``make_exe``, whose factory runs FlyDSL's AST rewrite.
        """
        digest = self._disk_key(key, args)
        sym = f"fly_{self.name}_{digest[:16]}"
        out_dir = self.directory()
        found = self._find(out_dir, digest)
        if found is None:
            with _file_lock(out_dir / f"{digest}.lock"):
                found = self._find(out_dir, digest)
                if found is None:
                    found = self._build(make_exe, args, out_dir, sym, digest)
        elif _MODE == "verify":
            self._verify(make_exe(), args, sym, found.with_suffix(".o"))
        if fly_env.compile.compile_only:
            return None
        try:
            return _load_module(found.with_suffix(".o"))
        except (OSError, ValueError) as e:
            # A truncated or foreign object: rebuild once.
            logger.warning(f"[FlyDSL AOT] loading {found}.o failed ({e}); rebuilding")
            with _file_lock(out_dir / f"{digest}.lock"):
                found.with_suffix(".o").unlink(missing_ok=True)
                found = self._build(make_exe, args, out_dir, sym, digest)
            return _load_module(found.with_suffix(".o"))

    @staticmethod
    def _find(out_dir: Path, digest: str):
        """The stem of the variant of ``digest`` built from the current sources
        and environment.

        An entry is stored per version of both, ``<digest>-<variant hash>``, so
        switching a kernel file or a variable back and forth (e.g. between
        branches) reuses the earlier build instead of overwriting it.
        """
        for sidecar in sorted(out_dir.glob(f"{digest}-*.json")):
            stem = sidecar.with_suffix("")
            try:
                meta = json.loads(sidecar.read_text())
            except (OSError, ValueError):
                continue
            if stem.with_suffix(".o").exists() and _entry_current(meta):
                return stem
        return None

    def _build(self, make_exe, args, out_dir, sym, digest):
        out_dir.mkdir(parents=True, exist_ok=True)
        _remove_stale_tmp(out_dir)
        tmp = Path(tempfile.mkdtemp(dir=out_dir, prefix=".tmp-"))
        try:
            files = set(self._extra)
            read = set()
            calls = set()
            _clear_get_gfx()
            with _record_sources(files, read, calls):
                exe = make_exe()
                tmp_obj = _export(exe, args, tmp, sym)
            # Parsed now, not from the per-process cache: the build read the
            # files as they are on disk at this point.
            _parse.cache_clear()
            _bindings.cache_clear()
            called = _called_lines(calls)
            files |= _import_sources(files, called)
            changed = _changed_since_start(files)
            if changed:
                raise RuntimeError(
                    f"{changed[0]} changed after this process started; not caching "
                    f"a build that may hold its earlier code"
                )
            # Hashed now, not from the per-process cache: the build read the
            # files as they are on disk at this point.
            _file_digest.cache_clear()
            sources = {_portable(f): _file_digest(f) for f in sorted(files)}
            meta = {"sources": sources, "env": _entry_env(files, read, calls, called)}
            variant = hashlib.sha256(json.dumps(meta).encode()).hexdigest()[:12]
            stem = out_dir / f"{digest}-{variant}"
            sidecar = tmp / "meta.json"
            sidecar.write_text(json.dumps(meta, indent=1))
            # The object is published last; _find needs both files.
            os.replace(sidecar, stem.with_suffix(".json"))
            os.replace(tmp_obj, stem.with_suffix(".o"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
            _parse.cache_clear()
            _bindings.cache_clear()
        self._builds += 1
        logger.info(f"[FlyDSL AOT] built {stem}.o")
        return stem

    def _verify(self, exe, args, sym, obj) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fresh = _export(exe, args, Path(tmp), sym).read_bytes()
        if not obj.exists() or obj.read_bytes() != fresh:
            raise _StaleEntryError(
                f"[FlyDSL AOT] stale cache entry {obj}: re-exported object differs. "
                f"Something the fingerprint does not cover changed the kernel; "
                f"delete {obj.parent} or add the input to the cache key."
            )
