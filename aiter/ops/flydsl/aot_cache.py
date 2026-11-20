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
toolchain (FlyDSL, HIP, Python, arch) and the code-affecting environment.
Each entry also records the Python source files that ran while it was built
(the factory and the trace) with their hashes; a hit whose sources changed is
rebuilt. Editing one kernel therefore invalidates only the entries that ran it.

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

import contextlib
import fcntl
import functools
import hashlib
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

_SCHEMA = 3
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
    # Hashed as the arch it selects (_target_arch), so that an AOT build, which
    # sets it, and the runtime, which reads the device, share a directory.
    "FLYDSL_GPU_ARCH",
}
_ENV_EXTRA = ("HSA_OVERRIDE_GFX_VERSION", "COMPILE_OPT_LEVEL")


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
    # the environment. Without a device (AOT builds) ask FlyDSL, as its compiler does.
    if os.environ.get("FLYDSL_GPU_ARCH") or not torch.cuda.is_available():
        from flydsl.runtime.device import get_rocm_arch

        return get_rocm_arch()
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return props.gcnArchName.split(":")[0]


def _wheel_record(fly_root: Path) -> bytes | None:
    """The file hashes of the FlyDSL wheel installed at ``fly_root``, if any.

    Only the package's own entries: an installer adds its own files and
    ``.pyc`` entries, which differ between installs of the same wheel.
    """
    from importlib.metadata import distributions

    for dist in distributions(path=[str(fly_root.parent)]):
        if dist.metadata["Name"] != "flydsl":
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
    Python, the target arch and the code-affecting environment variables.
    Changing any of them selects a new cache directory. Kernel sources are
    tracked per entry instead (``_record_sources``). Only the FlyDSL hash is
    cached: an AOT build switches arch and environment within one process.
    """
    env_items = sorted(
        (k, v)
        for k, v in os.environ.items()
        if (k.startswith(("AITER_FLYDSL_", "FLYDSL_")) or k in _ENV_EXTRA)
        and k not in _ENV_IGNORED
    )
    parts = [
        _SCHEMA,
        sys.version_info[:2],
        platform.machine(),
        flydsl.__version__,
        torch.version.hip,
        _target_arch(),
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
def _file_digest(path: str) -> str | None:
    """Content hash of a source file, once per process; None if it is gone."""
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


@contextlib.contextmanager
def _record_sources(out: set):
    """Collect the files of every Python function called in this block.

    Run around the factory and the trace, it names the sources the generated
    code can depend on: the kernel, its helpers and any aiter code they call.
    """
    seen = set()

    def prof(frame, event, _arg):
        if event == "call":
            seen.add(frame.f_code.co_filename)

    old = sys.getprofile()
    sys.setprofile(prof)
    try:
        yield
    finally:
        sys.setprofile(old)
        out.update(os.path.realpath(f) for f in seen if _is_tracked(f))


def _sources_current(sources: dict) -> bool:
    return all(_file_digest(p) == d for p, d in sources.items())


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
        """The stem of the variant of ``digest`` built from the current sources.

        An entry is stored per source version, ``<digest>-<sources hash>``, so
        switching a kernel file back and forth (e.g. between branches) reuses
        the earlier build instead of overwriting it.
        """
        for sidecar in sorted(out_dir.glob(f"{digest}-*.json")):
            stem = sidecar.with_suffix("")
            try:
                meta = json.loads(sidecar.read_text())
            except (OSError, ValueError):
                continue
            if stem.with_suffix(".o").exists() and _sources_current(meta["sources"]):
                return stem
        return None

    def _build(self, make_exe, args, out_dir, sym, digest):
        out_dir.mkdir(parents=True, exist_ok=True)
        _remove_stale_tmp(out_dir)
        tmp = Path(tempfile.mkdtemp(dir=out_dir, prefix=".tmp-"))
        try:
            files = set(self._extra)
            with _record_sources(files):
                exe = make_exe()
                tmp_obj = _export(exe, args, tmp, sym)
            # Hashed now, not from the per-process cache: the build read the
            # files as they are on disk at this point.
            _file_digest.cache_clear()
            sources = {f: _file_digest(f) for f in sorted(files)}
            meta = {"sources": sources}
            src_hash = hashlib.sha256(json.dumps(sources).encode()).hexdigest()[:12]
            stem = out_dir / f"{digest}-{src_hash}"
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

    def _verify(self, exe, args, sym, obj) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fresh = _export(exe, args, Path(tmp), sym).read_bytes()
        if not obj.exists() or obj.read_bytes() != fresh:
            raise _StaleEntryError(
                f"[FlyDSL AOT] stale cache entry {obj}: re-exported object differs. "
                f"Something the fingerprint does not cover changed the kernel; "
                f"delete {obj.parent} or add the input to the cache key."
            )
