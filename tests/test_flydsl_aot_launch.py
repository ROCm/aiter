# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``_run_compiled`` launches routed through ``FlyDSLAotCache`` by
``aiter/ops/flydsl/aot_launch.py``, on the gfx942 whole-MoE backend.

Its call sites still call ``_run_compiled`` with a torch stream. Each mode runs
in a fresh process on the prefill rows (decode accumulates with atomics, so its
output is not bitwise reproducible):

* ``jit``: ``AITER_FLYDSL_AOT_CACHE=0``. Nothing is bound or written.
* ``aot miss``: ``=1`` on an empty directory. Every launch traces, as ``jit``
  does, and writes nothing; a ``COMPILE_ONLY`` run then prebuilds the entries.
* ``aot hit``: ``=1`` again. Every launch loads its object, nothing builds.

All modes must produce bitwise identical outputs. The GDN hidden-state kernel
and the paged-attention decode planner are checked the same way.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import flydsl.compiler as flyc
import pytest

from aiter.ops.flydsl.aot_launch import _ATTR, bind_aot

ROOT = Path(__file__).resolve().parents[1]
QUANTS = ("no", "per_tensor")
CACHES = (
    "aiter.ops.flydsl.fused_moe_gfx942:_get_compiled_kernel.aot_cache",
    "aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.moe_reduce:_SORTED_SUM_AOT_CACHE",
    "aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.quant:_ABSMAX_AOT_CACHE",
    "aiter.ops.flydsl.kernels.moe_gemm_2stage.gfx9_fp8.quant:_QUANT_AOT_CACHE",
)


class _Launcher:
    pass


class _Cache:
    name = "fake"
    epoch = 0


@pytest.mark.skipif(
    os.environ.get("AITER_FLYDSL_AOT_CACHE", "1").strip() == "0",
    reason="mode 0 binds nothing",
)
def test_one_launcher_under_two_keys_runs_the_jit():
    exe, cache = _Launcher(), _Cache()
    bind_aot(exe, cache, ("a",))
    bind_aot(exe, cache, ("a",))
    assert getattr(exe, _ATTR).key == ("a",)
    bind_aot(exe, cache, ("b",))
    assert getattr(exe, _ATTR) is False
    bind_aot(exe, cache, ("a",))
    assert getattr(exe, _ATTR) is False


@pytest.mark.skipif(
    os.environ.get("AITER_FLYDSL_AOT_CACHE", "1").strip() == "0",
    reason="mode 0 binds nothing",
)
def test_argument_the_cache_cannot_key_runs_the_jit():
    import functools

    import numpy as np

    from aiter.ops.flydsl.aot_launch import _LazyLauncher, aot_factory

    @aot_factory("test_unkeyable")
    @functools.cache
    def factory(limit):
        return _Launcher()

    # A numpy scalar hashes like a float but has no disk key: no binding, so
    # _run_compiled takes the JIT path instead of raising on every launch.
    exe = factory(np.float32(7.0))
    assert getattr(exe, _ATTR) is False
    handle = factory.lazy(np.float32(7.0))
    assert not isinstance(handle, _LazyLauncher)
    assert getattr(handle, _ATTR) is False
    assert isinstance(factory.lazy(8.0), _LazyLauncher)


@pytest.mark.skipif(
    os.environ.get("AITER_FLYDSL_AOT_CACHE", "1").strip() == "0",
    reason="mode 0 binds nothing",
)
def test_specialization_key_the_cache_cannot_key_runs_the_jit(monkeypatch):
    import enum

    from aiter.ops.flydsl import aot_cache

    class Mode(enum.Enum):
        A = 1

    # The real launch, as far as the disk key: nothing is built or loaded.
    resolved = []

    def resolve(self, key, make_exe, args):
        self._disk_key(key, args)
        resolved.append(key)
        return lambda *a: None

    monkeypatch.setattr(aot_cache.FlyDSLAotCache, "_resolve", resolve)
    monkeypatch.setattr(aot_cache, "_load_module", object())
    exe, cache = _Launcher(), aot_cache.FlyDSLAotCache("test_unkeyable_spec")
    bind_aot(exe, cache, ("a",))
    binding = getattr(exe, _ATTR)
    assert binding((), None) is True
    assert binding((), ("ok", 1)) is True
    # Not raised on every launch: the JIT path takes this key, and the cache
    # is not asked again.
    assert binding((), Mode.A) is False
    assert binding((), Mode.A) is False
    assert resolved == [("a",), (("a",), ("ok", 1))]
    assert binding.unkeyable == {Mode.A}
    assert binding((), ("ok", 1)) is True


@pytest.mark.skipif(
    os.environ.get("AITER_FLYDSL_AOT_CACHE", "1").strip() == "0",
    reason="mode 0 binds nothing",
)
def test_binding_keeps_the_loaded_kernel_until_the_cache_is_invalidated(monkeypatch):
    from aiter.ops.flydsl import aot_cache

    ran, resolved = [], []

    def resolve(self, key, make_exe, args):
        resolved.append(key)
        return lambda *a: ran.append((key, a))

    monkeypatch.setattr(aot_cache.FlyDSLAotCache, "_resolve", resolve)
    monkeypatch.setattr(aot_cache, "_load_module", object())
    monkeypatch.delenv("COMPILE_ONLY", raising=False)
    exe, cache = _Launcher(), aot_cache.FlyDSLAotCache("test_binding_kernels")
    bind_aot(exe, cache, ("a",))
    binding = getattr(exe, _ATTR)
    for spec in (None, 7, None, 7):
        assert binding((1,), spec) is True
    assert resolved == [("a",), (("a",), 7)]
    assert [key for key, _ in ran] == [("a",), (("a",), 7), ("a",), (("a",), 7)]
    # Later launches go straight to the loaded kernel, without the cache.
    assert set(binding.kernels) == {None, 7}
    launches = []
    monkeypatch.setattr(cache, "launch", lambda *a: launches.append(a))
    assert binding((1,), 7) is True and not launches and len(ran) == 5
    monkeypatch.undo()
    monkeypatch.setattr(aot_cache.FlyDSLAotCache, "_resolve", resolve)
    monkeypatch.setattr(aot_cache, "_load_module", object())

    # A cache invalidated while a launch is under way (by another thread):
    # the kernel that launch read is not kept.
    def stale(*a):
        ran.append(("stale", a))

    launch = cache.launch

    def racing_launch(key, make_exe, *args):
        cache._mem[key] = stale
        cache.epoch += 1

    monkeypatch.setattr(cache, "launch", racing_launch)
    assert binding((1,), 8) is True
    assert 8 not in binding.kernels
    monkeypatch.setattr(cache, "launch", launch)

    # A cache that was reset does not launch what it loaded before, and the
    # binding lets go of it.
    cache.invalidate()
    made = []
    binding.make_exe = lambda: made.append(1) or _Launcher()
    monkeypatch.setattr(aot_cache._JitKernel, "__call__", lambda self, *a: None)
    n = len(ran)
    assert binding((1,), None) is True
    assert made == [1] and len(ran) == n
    # What it keeps now is the JIT kernel of that key.
    assert isinstance(binding.kernels[None], aot_cache._JitKernel)


@pytest.mark.skipif(
    os.environ.get("AITER_FLYDSL_AOT_CACHE", "1").strip() == "0",
    reason="mode 0 returns the factory itself",
)
def test_cache_clear_retraces_through_the_jit():
    import functools

    from aiter.ops.flydsl.aot_cache import _JitKernel
    from aiter.ops.flydsl.aot_launch import aot_factory

    @aot_factory("test_cache_clear")
    @functools.cache
    def factory(n):
        return _Launcher()

    cache = factory.aot_cache
    handle = factory.lazy(1)
    cache._mem[getattr(handle, _ATTR).key] = "loaded"
    factory.cache_clear()
    # A launcher loaded before the reset is not launched again, and no entry
    # on disk is read or written for it: the factory's next launcher is traced.
    assert not cache._mem
    assert factory.lazy(1) is not handle
    made = []
    kernel = cache._kernel(("k",), lambda: made.append(1) or _Launcher(), ())
    assert isinstance(kernel, _JitKernel) and made == [1]


def test_constexpr_launch_arguments_select_the_entry():
    import inspect

    import flydsl.expr as fx

    from aiter.ops.flydsl.aot_cache import (
        _arg_signature,
        _baked_args,
        _baked_current,
        _BakedArgumentError,
        _BakedKernel,
    )

    def launch(
        out: fx.Int64,
        n: fx.Int32,
        tile: fx.Constexpr[int],
        mode: fx.Constexpr[str] = "a",
        dims: fx.Constexpr[tuple] = (1, 2),
        stream: fx.Stream = None,
    ):
        pass

    exe = _Launcher()
    exe._sig = inspect.signature(launch)
    args = (0, 4, 64, "b", (3, 4))
    # A string or a tuple has no runtime type: it is keyed as a constexpr, and
    # its value is recorded with the entry.
    assert _arg_signature(args) == ("int", "int", "int", "const", "const")
    baked = _baked_args(exe, args)
    assert baked == {"2": "64", "3": "'b'", "4": "('tuple', 3, 4)"}

    # An entry serves only the constexpr values it was built with; a runtime
    # argument may differ.
    meta = {"libs": {}, "baked": baked}
    assert _baked_current(meta["baked"], (1, 8, 64, "b", (3, 4)))
    assert not _baked_current(meta["baked"], (0, 4, 32, "b", (3, 4)))
    assert not _baked_current(meta["baked"], (0, 4, 64, "c", (3, 4)))
    assert not _baked_current(meta["baked"], (0, 4, 64, "b", (3, 5)))
    assert not _baked_current(meta["baked"], (0, 4, 64))
    # A constexpr left to its default is fixed by the launcher's source.
    assert _baked_args(exe, (0, 4, 64)) == {"2": "64"}
    # A value no later process can compare is not exported.
    with pytest.raises(RuntimeError):
        _baked_args(exe, (0, 4, lambda: 1))
    # Nor is a string where the launcher takes a runtime argument.
    with pytest.raises(RuntimeError):
        _baked_args(exe, (0, "x", 64))

    # Within a process, one key launched with another value raises instead of
    # running the kernel built for the first.
    ran = []
    kernel = _BakedKernel(lambda *a: ran.append(a), "t", ("k",), args, [2, 3, 4])
    kernel(1, 8, 64, "b", (3, 4))
    for other in ((1, 8, 32, "b", (3, 4)), (1, 8, 64, "c", (3, 4)), (1, 8, 64, "b")):
        with pytest.raises(_BakedArgumentError):
            kernel(*other)
    # A constexpr the first call left to its default cannot be passed later;
    # a trailing runtime argument (a stream) may be left to its default.
    short = _BakedKernel(lambda *a: ran.append(a), "t", ("k",), (0, 4, 64, 7), [2])
    with pytest.raises(_BakedArgumentError):
        short(0, 4, 64, 7, "b")
    short(0, 4, 64)
    with pytest.raises(_BakedArgumentError):
        short(0, 4)
    # -0.0 equals 0.0 and is another constexpr.
    with pytest.raises(_BakedArgumentError):
        _BakedKernel(lambda *a: None, "t", ("k",), (0.0,), [0])(-0.0)
    # Compared as the disk entry is: True is not 1 (in a tuple too), and NaN
    # is itself.
    with pytest.raises(_BakedArgumentError):
        _BakedKernel(lambda *a: None, "t", ("k",), ((1, 2),), [0])((True, 2))
    with pytest.raises(_BakedArgumentError):
        _BakedKernel(lambda *a: None, "t", ("k",), (True,), [0])(1)
    _BakedKernel(lambda *a: ran.append(a), "t", ("k",), (float("nan"),), [0])(
        float("nan")
    )
    assert len(ran) == 3


def test_constexpr_of_a_bound_launcher():
    import functools
    import inspect

    import flydsl.expr as fx

    from aiter.ops.flydsl.aot_cache import _baked_args

    def launch(x: fx.Int32, tile: fx.Constexpr[int]):
        pass

    class _Jit:
        def __call__(self, *args):
            pass

    # A @flyc.jit method is launched as partial(JitFunction.__call__, obj).
    jit = _Jit()
    jit._sig = inspect.signature(launch)
    assert _baked_args(functools.partial(jit.__call__), (1, 64)) == {"1": "64"}
    # Its constexprs cannot be missed silently when the signature is not there.
    with pytest.raises(RuntimeError):
        _baked_args(functools.partial(_Jit().__call__), (1, 64))


def test_compile_hints_of_the_caller_select_the_entry():
    from flydsl.compiler.kernel_function import CompilationContext

    from aiter.ops.flydsl.aot_cache import FlyDSLAotCache

    cache = FlyDSLAotCache("test_compile_hints")
    plain = cache._disk_key(("k",), (1,))
    with CompilationContext.compile_hints({"fastmath": "contract"}):
        hinted = cache._disk_key(("k",), (1,))
        assert hinted == cache._disk_key(("k",), (1,))
    with CompilationContext.compile_hints({"fastmath": "fast"}):
        assert cache._disk_key(("k",), (1,)) not in (plain, hinted)
    assert hinted != plain == cache._disk_key(("k",), (1,))


def test_linked_bitcode_must_be_declared(monkeypatch, tmp_path):
    from aiter.ops.flydsl import aot_cache

    # Written after this process started, as a library extracted on first use.
    lib = tmp_path / "device.bc"
    lib.write_bytes(b"BC one")

    class Artifact:
        _link_libs = (str(lib),)

    class Compiled:
        _keepalive = Artifact()

        def export_to_c(self, out_dir, sym):
            Path(out_dir, f"{sym}.o").write_bytes(b"\x7fELF object")

    monkeypatch.setattr(aot_cache.flyc, "compile", lambda *a, **k: Compiled())

    def make_exe():
        return lambda: None

    out_dir = tmp_path / "cache"
    cache = aot_cache.FlyDSLAotCache("test_link_libs_undeclared")
    with pytest.raises(aot_cache._UndeclaredSourceError, match=r"device\.bc"):
        cache._build(make_exe, (), out_dir, "sym", "digest")
    assert not list(out_dir.glob("*.o")), "an entry was written"
    # A prebuild never falls back silently to the JIT path, in any mode.
    monkeypatch.setattr(aot_cache, "_MODE", "1")
    monkeypatch.setattr(aot_cache, "_load_module", lambda path: None)
    monkeypatch.setenv("AITER_FLYDSL_AOT_CACHE_DIR", str(tmp_path / "root"))
    monkeypatch.setenv("COMPILE_ONLY", "1")
    with pytest.raises(aot_cache._UndeclaredSourceError):
        cache._kernel(("k",), make_exe, ())
    monkeypatch.delenv("COMPILE_ONLY")

    def disk_key():
        aot_cache._file_digest.cache_clear()
        cache = aot_cache.FlyDSLAotCache("test_link_libs", sources=[lib])
        return cache, cache._disk_key(("k",), ())

    cache, first = disk_key()
    stem = cache._build(make_exe, (), out_dir, "sym", first)
    assert cache._find(out_dir, first, ()) == stem
    # A rebuilt library (e.g. another MORI) is another entry.
    lib.write_bytes(b"BC two")
    assert disk_key()[1] != first


def test_compile_hook_that_only_identifies_kernels_is_not_a_verify_failure(
    monkeypatch,
):
    from aiter.ops.flydsl import aot_cache

    monkeypatch.setattr(aot_cache, "_MODE", "verify")
    monkeypatch.setattr(aot_cache, "_load_module", object())
    cache = aot_cache.FlyDSLAotCache("test_no_export")
    made = []

    def make_exe():
        made.append(1)
        return _Launcher()

    def no_export(self, key, make_exe, args):
        raise aot_cache._NoExport("flyc.compile returned no function to export")

    # An AOT replay that only identifies kernels stops flyc.compile: nothing
    # to export, and the JIT path's compile-only launch is what it watches.
    monkeypatch.setattr(aot_cache.FlyDSLAotCache, "_resolve", no_export)
    assert isinstance(cache._kernel(("k",), make_exe, ()), aot_cache._JitKernel)
    assert made == [1]

    def export_failed(self, key, make_exe, args):
        raise RuntimeError("export failed")

    # A real export failure still fails under verify.
    monkeypatch.setattr(aot_cache.FlyDSLAotCache, "_resolve", export_failed)
    with pytest.raises(RuntimeError, match="export failed"):
        cache._kernel(("k",), make_exe, ())


class _FakeCompiled:
    class _keepalive:
        _link_libs = ()

    def export_to_c(self, out_dir, sym):
        Path(out_dir, f"{sym}.o").write_bytes(b"\x7fELF object")


def _old_file(path, text):
    """A source file with an old mtime, as an installed one has."""
    path.write_text(text)
    old = time.time() - 3600
    os.utime(path, (old, old))
    return path


def test_declared_sources_and_variables_select_the_entry(monkeypatch, tmp_path):
    from aiter.ops.flydsl import aot_cache

    kernel = _old_file(tmp_path / "kernel.py", "TILE = 4\n")
    other = _old_file(tmp_path / "other.py", "X = 1\n")
    monkeypatch.delenv("AOT_TEST_SWITCH", raising=False)

    def disk_key():
        # A new process: the files are read again.
        aot_cache._file_digest.cache_clear()
        cache = aot_cache.FlyDSLAotCache(
            "test_declared", sources=[kernel], env=["AOT_TEST_SWITCH"]
        )
        assert str(kernel) in cache._files and __file__ in cache._files
        return cache._disk_key(("k",), ())

    first = disk_key()
    assert disk_key() == first
    # Not declared: nothing is detected.
    _old_file(other, "X = 2\n")
    monkeypatch.setenv("AOT_TEST_UNDECLARED", "1")
    assert disk_key() == first
    _old_file(kernel, "TILE = 8\n")
    edited = disk_key()
    assert edited != first
    monkeypatch.setenv("AOT_TEST_SWITCH", "1")
    assert disk_key() not in (first, edited)
    # Going back finds the earlier entries.
    monkeypatch.delenv("AOT_TEST_SWITCH")
    _old_file(kernel, "TILE = 4\n")
    assert disk_key() == first


def test_directory_source_is_every_file_below_it(tmp_path):
    from aiter.ops.flydsl import aot_cache

    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    a = _old_file(tmp_path / "pkg" / "a.py", "")
    b = _old_file(tmp_path / "pkg" / "sub" / "b.py", "")
    cache = aot_cache.FlyDSLAotCache("test_dir_source", sources=[tmp_path / "pkg"])
    assert {str(a), str(b)} <= set(cache._files)
    # Relative paths are under aiter/ops/flydsl.
    cache = aot_cache.FlyDSLAotCache("test_rel_source", sources="kernels/act.py")
    assert any(f.endswith("aiter/ops/flydsl/kernels/act.py") for f in cache._files)
    # A list naming a missing file does not break the import of the module
    # that creates the cache; its launches take the JIT path.
    cache = aot_cache.FlyDSLAotCache("test_missing_source", sources=["kernels/nope.py"])
    with pytest.raises(RuntimeError, match="does not exist"):
        cache._disk_key(("k",), ())


def test_launch_without_an_entry_traces_and_writes_nothing(monkeypatch, tmp_path):
    from aiter.ops.flydsl import aot_cache

    monkeypatch.setattr(aot_cache, "_MODE", "1")
    monkeypatch.setattr(aot_cache, "_load_module", lambda path: None)
    monkeypatch.setattr(aot_cache.flyc, "compile", lambda *a, **k: _FakeCompiled())
    monkeypatch.setattr(aot_cache.FlyDSLAotCache, "directory", lambda self: tmp_path)
    monkeypatch.delenv("COMPILE_ONLY", raising=False)
    cache = aot_cache.FlyDSLAotCache("test_runtime_miss")

    def make_exe():
        return lambda: None

    # Only a prebuild builds: this process may have imported its sources long
    # before they were hashed, so it must not store what it would trace.
    with pytest.raises(aot_cache._Miss):
        cache._resolve(("k",), make_exe, ())
    for mode in ("1", "verify"):  # a miss is not a failure, under verify either
        monkeypatch.setattr(aot_cache, "_MODE", mode)
        kernel = cache._kernel(("k",), make_exe, ())
        assert isinstance(kernel, aot_cache._JitKernel)
    assert not list(tmp_path.iterdir()), "a launch wrote the cache"
    # The same call under COMPILE_ONLY is the prebuild.
    monkeypatch.setenv("COMPILE_ONLY", "1")
    assert cache._kernel(("k",), make_exe, ()) is None
    assert len(list(tmp_path.glob("*.o"))) == 1


def _errors(monkeypatch, *modules):
    """The messages ``modules`` log with ``logger.error`` from here on."""
    seen = []
    for module in modules:
        monkeypatch.setattr(module.logger, "error", seen.append)
    return seen


def test_every_launch_sent_to_the_jit_path_logs_one_error(monkeypatch, tmp_path):
    from aiter.ops.flydsl import aot_cache, aot_launch
    from aiter.ops.flydsl.kernels import tensor_shim

    monkeypatch.setattr(aot_cache, "_MODE", "1")
    monkeypatch.setattr(aot_cache, "_load_module", lambda path: None)
    monkeypatch.setattr(aot_cache.FlyDSLAotCache, "directory", lambda self: tmp_path)
    monkeypatch.delenv("COMPILE_ONLY", raising=False)
    ran = []
    monkeypatch.setattr(
        tensor_shim,
        "_run_compiled",
        lambda exe, *args, specialization_key=None: ran.append(args),
    )
    errors = _errors(monkeypatch, aot_cache)
    exe, cache = _Launcher(), aot_cache.FlyDSLAotCache("test_jit_error")
    bind_aot(exe, cache, ("k",))
    binding = getattr(exe, _ATTR)
    for n in (1, 2, 3):
        assert binding((n,), None)
    # Launched each time, logged once, and nothing was written.
    assert ran == [(1,), (2,), (3,)]
    assert len(errors) == 1 and "no prebuilt entry" in errors[0], errors
    assert not list(tmp_path.iterdir())
    # The key's later launches skip the cache lookup.
    assert isinstance(binding.kernels[None], aot_cache._JitKernel)
    # Another key of the launcher is another error.
    assert binding((4,), "spec")
    assert len(errors) == 2
    # A key the cache cannot store.
    errors2 = _errors(monkeypatch, aot_launch)
    bind_aot(_Launcher(), cache, (object(),))
    assert len(errors2) == 1 and "JIT path" in errors2[0]


def test_entry_that_does_not_load_is_kept_and_the_launch_traces(monkeypatch, tmp_path):
    from aiter.ops.flydsl import aot_cache

    monkeypatch.setattr(aot_cache.flyc, "compile", lambda *a, **k: _FakeCompiled())
    monkeypatch.setattr(aot_cache.FlyDSLAotCache, "directory", lambda self: tmp_path)
    monkeypatch.setattr(aot_cache, "_MODE", "1")
    cache = aot_cache.FlyDSLAotCache("test_load_failure")

    def make_exe():
        return lambda: None

    monkeypatch.setenv("COMPILE_ONLY", "1")
    monkeypatch.setattr(aot_cache, "_load_module", lambda path: None)
    assert cache._kernel(("k",), make_exe, ()) is None
    monkeypatch.delenv("COMPILE_ONLY")
    (obj,) = tmp_path.glob("*.o")

    def no_handles(path):
        raise OSError(24, "Too many open files")

    # Not the file's fault: a launch must not delete a good entry.
    monkeypatch.setattr(aot_cache, "_load_module", no_handles)
    errors = _errors(monkeypatch, aot_cache)
    assert isinstance(cache._kernel(("k",), make_exe, ()), aot_cache._JitKernel)
    assert obj.exists() and len(errors) == 1 and "failed" in errors[0]
    # Whatever a bad object makes the loader raise.
    monkeypatch.setattr(aot_cache, "_load_module", lambda path: {}["abi"])
    assert isinstance(cache._kernel(("k",), make_exe, ()), aot_cache._JitKernel)
    assert obj.exists() and len(errors) == 2
    monkeypatch.setattr(aot_cache, "_MODE", "verify")
    with pytest.raises(RuntimeError, match="loading"):
        cache._kernel(("k",), make_exe, ())
    assert obj.exists()
    # A prebuild replaces an object that is not one.
    monkeypatch.setattr(aot_cache, "_MODE", "1")
    obj.write_bytes(b"")
    monkeypatch.setenv("COMPILE_ONLY", "1")
    builds = cache._builds
    assert cache._kernel(("k",), make_exe, ()) is None
    assert cache._builds == builds + 1 and obj.read_bytes().startswith(b"\x7fELF")


def test_verify_compares_the_traced_ir_not_the_object_bytes(monkeypatch, tmp_path):
    from aiter.ops.flydsl import aot_cache

    monkeypatch.setattr(aot_cache.FlyDSLAotCache, "directory", lambda self: tmp_path)
    monkeypatch.setattr(aot_cache, "_MODE", "1")
    monkeypatch.setattr(aot_cache, "_load_module", lambda path: None)
    state = {"ir": "ir-one", "bytes": b"\x7fELF as the code generator made it"}

    def export(exe, args, out_dir, sym, libs=None, info=None):
        if info is not None:
            info["ir"] = state["ir"]
        obj = out_dir / f"{sym}.o"
        obj.write_bytes(state["bytes"])
        return obj

    monkeypatch.setattr(aot_cache, "_export", export)
    monkeypatch.setattr(aot_cache, "_compile", lambda exe, args: None)
    monkeypatch.setattr(aot_cache, "_ir_key", lambda exe, cf: state["ir"])
    cache = aot_cache.FlyDSLAotCache("test_verify_ir")

    def make_exe():
        return lambda: None

    monkeypatch.setenv("COMPILE_ONLY", "1")
    assert cache._kernel(("k",), make_exe, ()) is None
    monkeypatch.delenv("COMPILE_ONLY")
    monkeypatch.setattr(aot_cache, "_MODE", "verify")
    # The code generator may make other bytes of the same IR: not stale.
    state["bytes"] = b"\x7fELF with two instructions swapped"
    cache._kernel(("k",), make_exe, ())
    # Other IR under the same key is.
    state["ir"] = "ir-two"
    with pytest.raises(aot_cache._StaleEntryError, match="other IR"):
        cache._kernel(("k",), make_exe, ())
    # An entry with no IR key is compared by its bytes.
    state["ir"] = None
    (sidecar,) = tmp_path.glob("*.json")
    meta = json.loads(sidecar.read_text())
    meta["ir"] = None
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(aot_cache._StaleEntryError, match="object differs"):
        cache._kernel(("k",), make_exe, ())
    state["bytes"] = b"\x7fELF as the code generator made it"
    cache._kernel(("k",), make_exe, ())


def test_preload_and_bad_source_list_under_compile_only(monkeypatch, tmp_path):
    from aiter.ops.flydsl import aot_cache

    monkeypatch.setattr(aot_cache.flyc, "compile", lambda *a, **k: _FakeCompiled())
    monkeypatch.setattr(aot_cache.FlyDSLAotCache, "directory", lambda self: tmp_path)
    monkeypatch.setattr(aot_cache, "_MODE", "1")
    monkeypatch.setattr(aot_cache, "_load_module", lambda path: None)
    monkeypatch.setenv("COMPILE_ONLY", "1")

    def make_exe():
        return lambda: None

    # A preload turns COMPILE_ONLY on to warm one launcher up in a process
    # that serves: launches of other threads meanwhile build nothing.
    cache = aot_cache.FlyDSLAotCache("test_preload")
    monkeypatch.setattr(aot_cache, "_preloading", True)
    assert isinstance(cache._kernel(("k",), make_exe, ()), aot_cache._JitKernel)
    assert not list(tmp_path.glob("*.o"))
    monkeypatch.setattr(aot_cache, "_preloading", False)
    assert cache._kernel(("k",), make_exe, ()) is None
    assert len(list(tmp_path.glob("*.o"))) == 1

    # The preload itself: the flag is on for as long as COMPILE_ONLY is, also
    # when preloads nest and when the compile raises.
    from aiter.ops.flydsl.kernels import tensor_shim

    monkeypatch.delenv("COMPILE_ONLY")
    seen = []

    def compile(exe, *args):
        seen.append((os.environ.get("COMPILE_ONLY"), aot_cache._prebuilding()))
        if exe == "outer":
            tensor_shim._preload_compiled("inner")
            seen.append((os.environ.get("COMPILE_ONLY"), aot_cache._prebuilding()))
        if exe == "raises":
            raise ValueError("no")

    monkeypatch.setattr(tensor_shim.flyc, "compile", compile)
    tensor_shim._preload_compiled("outer")
    with pytest.raises(ValueError):
        tensor_shim._preload_compiled("raises")
    assert seen == [("1", False)] * 4
    assert "COMPILE_ONLY" not in os.environ and aot_cache._preloading is False
    monkeypatch.setenv("COMPILE_ONLY", "1")

    # A sources= list naming a missing file fails a prebuild; a launch only
    # logs it and traces.
    bad = aot_cache.FlyDSLAotCache("test_bad_sources", sources=["kernels/nope.py"])
    with pytest.raises(aot_cache._SourceListError, match="does not exist"):
        bad._kernel(("k",), make_exe, ())
    monkeypatch.delenv("COMPILE_ONLY")
    assert isinstance(bad._kernel(("k",), make_exe, ()), aot_cache._JitKernel)


def _bound_preload_setup(monkeypatch, tmp_path, name):
    """A cached factory bound to a cache in ``tmp_path``, a FlyDSL whose
    compile and load record what they do, and the error log."""
    import functools

    from aiter.ops.flydsl import aot_cache
    from aiter.ops.flydsl.aot_launch import aot_factory

    monkeypatch.setattr(aot_cache, "_MODE", "1")
    monkeypatch.setattr(aot_cache.FlyDSLAotCache, "directory", lambda self: tmp_path)
    monkeypatch.delenv("COMPILE_ONLY", raising=False)
    events = []

    def launch(a, b):
        pass

    def compile(exe, *args, compile_only=False):
        assert exe is launch, exe  # the launcher, never a lazy handle
        if compile_only:  # the export's compile
            events.append("export")
            return _FakeCompiled()
        if "COMPILE_ONLY" in os.environ:
            events.append("jit compile-only")
            return None
        events.append("jit launch")
        return lambda *a: events.append("jit function")

    def load_module(path):
        events.append("load")
        return lambda *a: events.append(("loaded function", a))

    monkeypatch.setattr(aot_cache.flyc, "compile", compile)
    monkeypatch.setattr(aot_cache, "_load_module", load_module)

    @aot_factory(name)
    @functools.cache
    def factory(n):
        return launch

    return factory, events, _errors(monkeypatch, aot_cache)


def test_a_prebuild_does_not_launch_an_entry_already_built(monkeypatch, tmp_path):
    from aiter.ops.flydsl import aot_cache
    from aiter.ops.flydsl.kernels import tensor_shim

    factory, events, errors = _bound_preload_setup(
        monkeypatch, tmp_path, "test_prebuild_built"
    )
    in_flydsl_cache, requested = [], []
    monkeypatch.setattr(aot_cache, "_ir_key", lambda exe, cf: "kernel key")
    monkeypatch.setattr(aot_cache, "_jit_artifact_built", in_flydsl_cache.__contains__)
    monkeypatch.setenv("COMPILE_ONLY", "1")

    def prebuild():
        events.clear()
        with aot_cache.entry_requests(requested.append):
            tensor_shim._run_compiled(factory.lazy(4), 1, 2)
        return list(events)

    # The export's compile is the launch of a new entry.
    assert prebuild() == ["export"] and not requested
    # Built, but FlyDSL's own cache lacks its kernel: the launch fills it.
    assert prebuild() == ["jit compile-only"] and not requested
    # Both there: nothing to trace, and the entry's key stands for the launch.
    in_flydsl_cache.append("kernel key")
    assert prebuild() == [] and requested == ["kernel key"]
    assert prebuild() == [] and requested == ["kernel key"] * 2
    # Another process of the same prebuild finds the entry as well.
    factory, events, errors = _bound_preload_setup(
        monkeypatch, tmp_path, "test_prebuild_built"
    )
    monkeypatch.setenv("COMPILE_ONLY", "1")
    assert prebuild() == [] and len(requested) == 3
    # The COMPILE_ONLY of a preload is no prebuild: it makes the launch.
    monkeypatch.setattr(aot_cache, "_preloading", True)
    assert prebuild() == ["jit compile-only"] and len(requested) == 3
    monkeypatch.setattr(aot_cache, "_preloading", False)
    # So does a launch inside another launcher's trace, which FlyDSL inlines.
    monkeypatch.setattr(aot_cache, "_tracing", lambda: True)
    assert prebuild() == ["jit compile-only"] and len(requested) == 3
    monkeypatch.setattr(aot_cache, "_tracing", lambda: False)
    assert prebuild() == [] and len(requested) == 4
    # Every collection under way is told, and none once it has ended.
    outer = []
    with aot_cache.entry_requests(outer.append):
        assert prebuild() == [] and outer == ["kernel key"] and len(requested) == 5
    assert prebuild() == [] and outer == ["kernel key"] and len(requested) == 6
    assert not aot_cache._request_listeners
    # Mode verify makes the launch, to compare its key with the entry's.
    monkeypatch.setattr(aot_cache, "_MODE", "verify")
    monkeypatch.setattr(aot_cache.FlyDSLAotCache, "_verify", lambda *a: None)
    assert prebuild() == ["jit compile-only"] and len(requested) == 6
    monkeypatch.setattr(aot_cache, "_MODE", "1")
    # An entry that records no key (FlyDSL kept none to read) is launched.
    for sidecar in tmp_path.glob("*.json"):
        meta = json.loads(sidecar.read_text())
        sidecar.write_text(json.dumps({**meta, "ir": None}))
    assert prebuild() == ["jit compile-only"] and len(requested) == 6
    assert len(list(tmp_path.glob("*.o"))) == 1 and not errors


def test_a_launch_is_compared_with_the_key_its_entry_recorded(monkeypatch, tmp_path):
    from aiter.ops.flydsl import aot_cache

    monkeypatch.setattr(aot_cache, "_MODE", "1")
    cache = aot_cache.FlyDSLAotCache("test_check_launch")
    errors = _errors(monkeypatch, aot_cache)
    entry = tmp_path / "digest-variant"
    first, again, other = ("built", object()), ("built", object()), ("other", object())

    def check(before, after):
        cache._check_launch(("k",), entry, "built", before, after)

    # The launch derived the recorded key, or none (an identifying replay
    # stops it before; a FlyDSL that keeps no last compile).
    check(None, first)
    check(first, again)
    check(first, first)
    check(first, None)
    assert not errors
    # Another key: the entry is not the kernel this request traces to.
    check(first, other)
    assert len(errors) == 1 and "built from other IR" in errors[0]
    # Said once per entry, however often a prebuild requests it.
    check(first, other)
    assert len(errors) == 1
    monkeypatch.setattr(aot_cache, "_MODE", "verify")
    with pytest.raises(aot_cache._StaleEntryError, match="built from other IR"):
        check(None, other)


def test_the_kernel_of_an_entry_is_looked_up_in_flydsl_s_cache(monkeypatch, tmp_path):
    from flydsl.compiler import jit_function

    from aiter.ops.flydsl import aot_cache

    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", str(tmp_path))
    assert not aot_cache._jit_artifact_built("kernel key")
    launcher_dir = tmp_path / "launch_some_kernel"
    launcher_dir.mkdir()
    (launcher_dir / f"{jit_function._disk_cache_key('kernel key')}.pkl").touch()
    assert aot_cache._jit_artifact_built("kernel key")
    assert not aot_cache._jit_artifact_built("another key")
    # A FlyDSL without the key of its cache files: the launch is made, and
    # the prebuild is told once why it is slow.
    warnings = []
    monkeypatch.setattr(aot_cache.logger, "warning", warnings.append)
    monkeypatch.setattr(aot_cache, "_jit_layout_warned", False)
    monkeypatch.delattr(jit_function, "_disk_cache_key")
    assert not aot_cache._jit_artifact_built("kernel key")
    assert not aot_cache._jit_artifact_built("kernel key")
    assert len(warnings) == 1 and "traces every request" in warnings[0]
    monkeypatch.undo()
    monkeypatch.setenv("FLYDSL_RUNTIME_CACHE_DIR", "")
    assert not aot_cache._jit_artifact_built("kernel key")


def test_preload_of_a_bound_launcher(monkeypatch, tmp_path):
    from aiter.ops.flydsl.kernels import tensor_shim

    factory, events, errors = _bound_preload_setup(
        monkeypatch, tmp_path, "test_preload_bound"
    )
    # A prebuild builds the entry, as a launch does, and dispatches nothing.
    monkeypatch.setenv("COMPILE_ONLY", "1")
    assert tensor_shim._preload_compiled(factory.lazy(4), 1, 2) is None
    assert events == ["export"] and len(list(tmp_path.glob("*.o"))) == 1
    # Its entry on disk: FlyDSL's own compile-only launch, which an AOT replay
    # watches, and no second build.
    events.clear()
    assert tensor_shim._preload_compiled(factory.lazy(4), 1, 2) is None
    assert events == ["jit compile-only"] and len(list(tmp_path.glob("*.o"))) == 1

    # A process that serves (another cache state: the factory of a new
    # process) loads the entry and launches nothing; the function it returns
    # takes the launcher's arguments, and so do later launches of the key.
    monkeypatch.delenv("COMPILE_ONLY")
    factory, events, errors = _bound_preload_setup(
        monkeypatch, tmp_path, "test_preload_bound"
    )
    loaded = tensor_shim._preload_compiled(factory.lazy(4), 1, 2)
    assert events == ["load"]
    loaded(3, 4)
    tensor_shim._run_compiled(factory.lazy(4), 5, 6)
    assert events == ["load", ("loaded function", (3, 4)), ("loaded function", (5, 6))]
    # A caller that keeps the function gets the loaded one, launched.
    events.clear()
    kept = tensor_shim._compile_and_keep(factory.lazy(4), 7, 8)
    assert events == [("loaded function", (7, 8))] and kept is not None
    assert not errors


def test_preload_of_a_bound_launcher_without_an_entry(monkeypatch, tmp_path):
    from aiter.ops.flydsl.kernels import tensor_shim

    factory, events, errors = _bound_preload_setup(
        monkeypatch, tmp_path, "test_preload_bound_miss"
    )
    # No entry: one error, and the launcher (not the handle) is preloaded on
    # the JIT path; nothing is written.
    assert tensor_shim._preload_compiled(factory.lazy(5), 1, 2) is None
    assert events == ["jit compile-only"]
    assert len(errors) == 1 and "no prebuilt entry" in errors[0], errors
    assert not list(tmp_path.iterdir())
    # The key's launches take the JIT path, without another error.
    tensor_shim._run_compiled(factory.lazy(5), 3, 4)
    assert events == ["jit compile-only", "jit launch"] and len(errors) == 1
    # A caller that keeps the function gets the JIT one, launched.
    events.clear()
    kept = tensor_shim._compile_and_keep(factory.lazy(6), 7, 8)
    assert events == ["jit launch"] and len(errors) == 2
    kept()
    assert events == ["jit launch", "jit function"]
    assert not list(tmp_path.iterdir())


def test_deleted_source_is_not_hashed(tmp_path):
    from aiter.ops.flydsl import aot_cache

    src = _old_file(tmp_path / "gone.py", "X = 1\n")
    cache = aot_cache.FlyDSLAotCache("test_deleted_source", sources=[src])
    assert str(src) in cache._files
    src.unlink()
    aot_cache._file_digest.cache_clear()
    with pytest.raises(RuntimeError, match="cannot be read"):
        cache._disk_key(("k",), ())


def test_source_edited_after_its_digest_was_read_is_not_stored(tmp_path):
    from aiter.ops.flydsl import aot_cache

    src = _old_file(tmp_path / "late_import.py", "X = 1\n")
    cache = aot_cache.FlyDSLAotCache("test_edited_late", sources=[src])
    # The digest is read once, here as for a key that hit the disk: a hit
    # imports nothing, so a factory may import its kernel module after this.
    cache._disk_key(("k",), ())
    src.write_text("X = 2\n")
    made = []
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="changed after"):
        cache._build(lambda: made.append(1), (), out_dir, "sym", "digest")
    # Nothing was traced and nothing stored under the earlier digest.
    assert not made
    assert not list(out_dir.glob("*.o")) and not list(out_dir.glob("*.json"))


def test_source_edited_while_its_entry_is_built_is_not_stored(monkeypatch, tmp_path):
    from aiter.ops.flydsl import aot_cache

    src = _old_file(tmp_path / "imported_by_the_trace.py", "X = 1\n")
    cache = aot_cache.FlyDSLAotCache("test_edited_in_build", sources=[src])
    cache._disk_key(("k",), ())

    def export(exe, args, out_dir, sym, libs=None, info=None):
        # The trace imports the kernel's modules; one was edited before that.
        src.write_text("X = 2\n")
        obj = out_dir / f"{sym}.o"
        obj.write_bytes(b"object of the edited code")
        return obj

    monkeypatch.setattr(aot_cache, "_export", export)
    out_dir = tmp_path / "out"
    with pytest.raises(RuntimeError, match="changed after"):
        cache._build(_Launcher, (), out_dir, "sym", "digest")
    assert not list(out_dir.glob("*.o")) and not list(out_dir.glob("*.json"))


def test_cache_names_are_unique_per_file(monkeypatch, tmp_path):
    from aiter.ops.flydsl import aot_cache
    from aiter.ops.flydsl.aot_cache import FlyDSLAotCache

    FlyDSLAotCache("test_unique_name")
    FlyDSLAotCache("test_unique_name")  # this file again, as a reloaded module
    other = _old_file(tmp_path / "other_module.py", "")
    monkeypatch.setattr(aot_cache, "_creation_file", lambda: str(other))
    with pytest.raises(ValueError, match="already used"):
        FlyDSLAotCache("test_unique_name")


@pytest.mark.skipif(
    os.environ.get("AITER_FLYDSL_AOT_CACHE", "1").strip() == "0",
    reason="mode 0 binds nothing",
)
def test_every_cache_declares_the_kernel_files_its_sources_import(monkeypatch):
    """Sources are declared by hand. A forgotten one is found here, and by the
    cache when it builds an entry: a declared file under ``kernels/`` imports
    a file under ``kernels/`` that is not declared."""
    import importlib
    import importlib.util
    import re
    import types

    from aiter.ops.flydsl import aot_cache

    if "mori" not in sys.modules and importlib.util.find_spec("mori") is None:
        # The MegaMoE modules import mori at module level, and only use it
        # at run time: a stand-in, as the MegaMoE AOT replay has.
        package, shmem = types.ModuleType("mori"), types.ModuleType("mori.shmem")
        package.shmem = shmem
        shmem.mori_shmem_create_tensor = None
        monkeypatch.setitem(sys.modules, "mori", package)
        monkeypatch.setitem(sys.modules, "mori.shmem", shmem)

    base = Path(aot_cache.__file__).resolve().parent
    creates = re.compile(r"^(@aot_factory\(|.*\bFlyDSLAotCache\()", re.MULTILINE)
    for file in sorted(base.rglob("*.py")):
        if file.name in aot_cache._NOT_KERNEL_CODE:
            continue
        if not creates.search(file.read_text()):
            continue
        name = ".".join(("aiter", "ops", "flydsl", *file.relative_to(base).parts))
        importlib.import_module(name.removesuffix(".py"))
    real = {
        name: cache
        for name, cache in aot_cache._CACHES.items()
        if not name.startswith("test_")
    }
    assert len(real) >= 20, sorted(real)
    # What each one is built from is declared in one table, and nowhere else.
    from aiter.aot.flydsl.sources import AOT_CACHES

    assert set(real) == set(AOT_CACHES), sorted(set(real) ^ set(AOT_CACHES))
    with pytest.raises(ValueError, match="declared in"):
        aot_cache.FlyDSLAotCache(next(iter(real)), sources=["kernels/act.py"])
    # A cache of aiter's own that is not in the table (or whose name is
    # misspelled where it is created) would depend on its own file only.
    in_tree = str(base / "kernels" / "act.py")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(aot_cache, "_creation_file", lambda: in_tree)
        with pytest.raises(ValueError, match="not declared in"):
            aot_cache.FlyDSLAotCache("mixed_moe_gemm")
    missing = {
        name: [os.path.relpath(f, base) for f in undeclared]
        for name, cache in real.items()
        if (undeclared := aot_cache._undeclared_imports(cache._files))
    }
    assert not missing, "\n".join(
        f'cache {name!r} must also list {files} under its "sources" in '
        f"aiter/aot/flydsl/sources.py"
        for name, files in sorted(missing.items())
    )
    # And the variables its sources read.
    unread = {
        name: names
        for name, cache in real.items()
        if (names := aot_cache._undeclared_env(cache._files, cache._env))
    }
    assert not unread, "\n".join(
        f'cache {name!r} must also list {names} under its "env" in '
        f"aiter/aot/flydsl/sources.py"
        for name, names in sorted(unread.items())
    )


def test_cache_that_forgets_a_kernel_file_does_not_build(monkeypatch, tmp_path):
    from aiter.ops.flydsl import aot_cache

    monkeypatch.setattr(aot_cache.flyc, "compile", lambda *a, **k: _FakeCompiled())
    # preshuffle_gemm.py imports mfma_preshuffle_pipeline.py, among others.
    cache = aot_cache.FlyDSLAotCache(
        "test_forgotten_source", sources=["kernels/preshuffle_gemm.py"]
    )
    with pytest.raises(
        aot_cache._UndeclaredSourceError, match="mfma_preshuffle_pipeline"
    ):
        cache._build(lambda: (lambda: None), (), tmp_path / "c", "s", "d")
    assert not list((tmp_path / "c").glob("*.o")), "an entry was written"


def test_launcher_with_a_defaulted_constexpr_is_guarded(monkeypatch, tmp_path):
    import inspect

    import flydsl.expr as fx

    from aiter.ops.flydsl import aot_cache

    def launch(n: fx.Int32, flag: fx.Constexpr[bool] = False):
        pass

    exe = _Launcher()
    exe._sig = inspect.signature(launch)
    monkeypatch.setattr(aot_cache.flyc, "compile", lambda *a, **k: _FakeCompiled())
    ran = []
    monkeypatch.setattr(aot_cache, "_load_module", lambda path: ran.append)
    monkeypatch.setattr(aot_cache.FlyDSLAotCache, "directory", lambda self: tmp_path)
    cache = aot_cache.FlyDSLAotCache("test_default_constexpr")
    # The first launch leaves the constexpr to its default: nothing is baked.
    monkeypatch.setenv("COMPILE_ONLY", "1")  # the prebuild
    assert cache._resolve(("k",), lambda: exe, (4,)) is None
    monkeypatch.delenv("COMPILE_ONLY")
    kernel = cache._resolve(("k",), lambda: exe, (4,))
    kernel(4)
    kernel(8)
    # A later launch of the key that passes it must not run that object.
    with pytest.raises(aot_cache._BakedArgumentError):
        kernel(4, True)
    assert ran == [4, 8]


def test_compile_only_set_during_a_first_launch_launches_nothing(monkeypatch):
    from aiter.ops.flydsl import aot_cache

    monkeypatch.delenv("COMPILE_ONLY", raising=False)
    cache = aot_cache.FlyDSLAotCache("test_compile_only_race")
    # _resolve returns None when COMPILE_ONLY appeared after launch() looked.
    monkeypatch.setattr(aot_cache.FlyDSLAotCache, "_kernel", lambda *a: None)
    cache.launch(("k",), lambda: _Launcher())
    assert ("k",) not in cache._mem


def test_sidecar_of_another_shape_is_skipped(tmp_path):
    from aiter.ops.flydsl.aot_cache import FlyDSLAotCache

    for i, text in enumerate(('{"sources": {}}', "[]", "{}", '{"baked": 3}')):
        (tmp_path / f"d-{i}.json").write_text(text)
        (tmp_path / f"d-{i}.o").write_bytes(b"\x7fELF object")
    assert FlyDSLAotCache._find(tmp_path, "d", ()) is None


def test_fingerprint_covers_what_flydsl_keys_its_cache_on():
    from flydsl.compiler import jit_function

    from aiter.ops.flydsl import aot_cache

    # FlyDSL's own cache is keyed on these; ours must be too.
    names = set(jit_function._CACHE_INVALIDATING_ENV_VARS)
    for arch_names in jit_function._TARGET_CACHE_INVALIDATING_ENV_VARS.values():
        names.update(arch_names)
    assert "LLVM_GFX1250_ANY_XNACK_CODEGEN" in names
    for name in names:
        assert aot_cache._env_in_fingerprint(name), name


def test_variable_is_an_input_of_the_caches_that_declare_it(monkeypatch, tmp_path):
    from aiter.ops.flydsl import aot_cache

    monkeypatch.setattr(aot_cache.flyc, "compile", lambda *a, **k: _FakeCompiled())
    for name in ("AITER_FLYDSL_TEST_SWITCH", "AITER_FLYDSL_KERNARG_PRELOAD"):
        monkeypatch.delenv(name, raising=False)
    reads = _old_file(
        tmp_path / "reads.py",
        'import os\nX = os.environ.get("AITER_FLYDSL_TEST_SWITCH")\n',
    )
    declares = aot_cache.FlyDSLAotCache(
        "test_env_declared", sources=[reads], env=["AITER_FLYDSL_TEST_SWITCH"]
    )
    other = aot_cache.FlyDSLAotCache("test_env_other")
    before = (
        aot_cache.fingerprint(),
        declares._disk_key(("k",), ()),
        other._disk_key(("k",), ()),
    )
    # An AITER_FLYDSL_ variable of one kernel is not in the fingerprint: the
    # cache that declares it misses, no other does.
    monkeypatch.setenv("AITER_FLYDSL_TEST_SWITCH", "1")
    assert aot_cache.fingerprint() == before[0]
    assert declares._disk_key(("k",), ()) != before[1]
    assert other._disk_key(("k",), ()) == before[2]
    # A variable the common sources read is an input of every cache.
    monkeypatch.setenv("AITER_FLYDSL_KERNARG_PRELOAD", "0")
    assert other._disk_key(("k",), ()) != before[2]
    # A prebuild of a cache whose sources read a variable it does not declare
    # fails, and names it.
    forgets = aot_cache.FlyDSLAotCache("test_env_forgotten", sources=[reads])
    with pytest.raises(
        aot_cache._UndeclaredSourceError, match="AITER_FLYDSL_TEST_SWITCH"
    ):
        forgets._build(lambda: (lambda: None), (), tmp_path / "c", "s", "d")
    assert not list((tmp_path / "c").glob("*.o"))


def test_common_sources_are_sources_of_every_cache(monkeypatch, tmp_path):
    from aiter.ops.flydsl import aot_cache

    # The helpers nearly every kernel is built with, and the per-arch hardware
    # constants in aiter/jit/utils: no cache lists them.
    base = Path(aot_cache.__file__).resolve().parent
    common = {str((base / src).resolve()) for src in aot_cache._COMMON_SOURCES}
    assert {Path(f).name for f in common} >= {
        "kernels_common.py",
        "tensor_shim.py",
        "buffer_ops.py",
        "chip_info.py",
        "lds_capacity.py",
    }
    assert common <= set(aot_cache.FlyDSLAotCache("test_common_sources")._files)

    # An edit to one selects other entries in a cache that declares nothing.
    digests = []
    for value in (1, 2):
        (tmp_path / str(value)).mkdir()
        table = _old_file(tmp_path / str(value) / "lds_capacity.py", f"X = {value}\n")
        monkeypatch.setattr(aot_cache, "_COMMON_SOURCES", (str(table),))
        cache = aot_cache.FlyDSLAotCache(f"test_common_table_{value}")
        digests.append(cache._sources())
    assert digests[0] != digests[1]


@pytest.mark.skipif(
    os.environ.get("AITER_FLYDSL_AOT_CACHE", "1").strip() == "0",
    reason="mode 0 returns the factory itself",
)
def test_factory_of_several_launchers_runs_the_jit(monkeypatch):
    import functools

    from aiter.ops.flydsl import aot_launch

    binds = []
    bind = aot_launch.bind_aot
    monkeypatch.setattr(
        aot_launch, "bind_aot", lambda *a, **k: binds.append(a) or bind(*a, **k)
    )

    @aot_launch.aot_factory("test_tuple_factory")
    @functools.cache
    def factory(n):
        return (_Launcher(), _Launcher())

    # A tuple cannot be bound: its launchers take the JIT path, and the
    # factory is not asked to bind it again on every launch.
    pair = factory(1)
    assert isinstance(pair, tuple) and factory(1) is pair and factory(1) is pair
    assert len(binds) == 1
    assert not hasattr(pair[0], _ATTR)


def child():
    import importlib

    import torch

    sys.path.insert(0, str(ROOT / "op_tests"))
    import test_flydsl_moe_gfx942 as t

    from aiter.ops.flydsl.aot_cache import _JitKernel
    from aiter.ops.flydsl.fused_moe_gfx942 import Config
    from aiter.ops.flydsl.kernels.moe_gemm_2stage import gfx9_fp8

    # The GEMM launchers launch lazily: a hit never builds one.
    factories = 0
    compile_gemm = gfx9_fp8.compile_gemm

    def counting_compile_gemm(**kwargs):
        nonlocal factories
        factories += 1
        return compile_gemm(**kwargs)

    gfx9_fp8.compile_gemm = counting_compile_gemm

    picked = {}
    for row in t.shipped_rows():
        if row[5] in QUANTS and Config.from_string(row[-1]).use_prefill:
            picked.setdefault(row[5], row)
    prebuild = "COMPILE_ONLY" in os.environ
    for token, md, inter, e, topk, quant, act, config in picked.values():
        try:
            impl, request, _, _ = t.check_moe(
                token, md, inter, e, topk, quant, act, "random", config
            )
        except AssertionError:
            if not prebuild:
                raise
            continue  # nothing was launched, so there is no output to check
        out = impl(request)
        torch.cuda.synchronize()
        digest = hashlib.sha256(out.float().cpu().numpy().tobytes()).hexdigest()
        print("RESULT " + json.dumps([f"{quant}/{token}/{config}", digest]))
    state = {}
    for spec in CACHES:
        module, attrs = spec.split(":")
        cache = importlib.import_module(module)
        for attr in attrs.split("."):
            cache = getattr(cache, attr)
        kernels = list(cache._mem.values())
        state[cache.name] = {
            "entries": len(kernels),
            "jit": sum(isinstance(k, _JitKernel) for k in kernels),
            "builds": cache._builds,
        }
    state["gemm factories"] = factories
    print("STATE " + json.dumps(state), flush=True)


def child_gdn_h_and_pa_plan():
    """Two configs each of the GDN hidden-state kernel and the paged-attention
    decode planner, through ``_run_compiled`` as their public ops launch them."""
    import torch

    from aiter.ops.flydsl import linear_attention_prefill_kernels as gdn
    from aiter.ops.flydsl.aot_cache import _JitKernel
    from aiter.ops.flydsl.kernels import pa_decode_plan as planner

    def digest(*tensors):
        h = hashlib.sha256()
        for t in tensors:
            if t is not None:
                h.update(t.contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return h.hexdigest()

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(0)
    H, Hg, K = 4, 2, 128
    for name, seqs in (("dense", None), ("varlen", (100, 157))):
        T = 256 if seqs is None else sum(seqs)
        k = torch.randn(1, T, Hg, K, generator=gen, device=device) * 0.1
        w = torch.randn(1, H, T, K, generator=gen, device=device) * 0.1
        u = torch.randn(1, H, T, K, generator=gen, device=device) * 0.1
        g = -torch.rand(1, T, H, generator=gen, device=device).cumsum(1) * 0.05
        kwargs = {"g": g, "use_exp2": False}
        if seqs is not None:
            n = len(seqs)
            offsets = [0, *torch.tensor(seqs).cumsum(0).tolist()]
            kwargs.update(
                cu_seqlens=torch.tensor(offsets, dtype=torch.int32, device=device),
                seq_lens_cpu=list(seqs),
                initial_state=torch.randn(n, H, K, K, generator=gen, device=device),
                output_final_state=True,
            )
        h, v_new, final = gdn.chunk_gated_delta_rule_fwd_h_flydsl_opt(
            k.bfloat16(), w.bfloat16(), u.bfloat16(), **kwargs
        )
        torch.cuda.synchronize()
        print("RESULT " + json.dumps([f"gdn_h/{name}", digest(h, v_new, final)]))
    for batch, window in ((7, 0), (300, 512)):
        lengths = torch.randint(
            -5, 70000, (batch,), generator=gen, device=device, dtype=torch.int32
        )
        plan = planner.plan_pa_decode(lengths, 2, sliding_window=window)
        torch.cuda.synchronize()
        print(
            "RESULT "
            + json.dumps(
                [f"pa_plan/{batch}/{window}", digest(plan.work_info, plan.reduce_info)]
            )
        )
    state = {}
    for cache in (
        gdn._get_or_compile_opt.aot_cache,
        planner.compile_pa_decode_plan.aot_cache,
    ):
        kernels = list(cache._mem.values())
        state[cache.name] = {
            "entries": len(kernels),
            "jit": sum(isinstance(k, _JitKernel) for k in kernels),
            "builds": cache._builds,
        }
    print("STATE " + json.dumps(state), flush=True)


def run_mode(mode, cache_dir, extra_env=None, child="--child"):
    env = {
        **os.environ,
        "AITER_FLYDSL_AOT_CACHE": mode,
        "AITER_FLYDSL_AOT_CACHE_DIR": cache_dir,
        **(extra_env or {}),
    }
    proc = subprocess.run(
        [sys.executable, __file__, child],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"mode {mode} failed:\n{proc.stdout}\n{proc.stderr}")
    outs, state = {}, None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            name, digest = json.loads(line[len("RESULT ") :])
            outs[name] = digest
        elif line.startswith("STATE "):
            state = json.loads(line[len("STATE ") :])
    assert outs or "COMPILE_ONLY" in env, f"mode {mode} ran no case"
    return outs, state


def test_moe_gfx942_through_aot_cache():
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() != "gfx942":
        pytest.skip(f"gfx942 FlyDSL whole-MoE does not run on {get_gfx_runtime()}")
    if not hasattr(flyc, "load_module"):
        pytest.skip("FlyDSL has no flyc.load_module, so every mode would run the JIT")

    with tempfile.TemporaryDirectory() as cache_dir:
        jit, state = run_mode("0", cache_dir)
        state.pop("gemm factories")
        assert all(s["entries"] == 0 for s in state.values()), state
        assert not any(Path(cache_dir).rglob("*.o")), "mode 0 wrote the AOT cache"

        # A launch never builds: with no entry it traces, as mode 0 does.
        miss, state = run_mode("1", cache_dir)
        state.pop("gemm factories")
        entries = {name: s["entries"] for name, s in state.items()}
        for name, s in state.items():
            assert s["entries"] > 0, f"{name} launched nothing through the cache"
            assert s["jit"] == s["entries"], f"{name} did not trace a miss: {s}"
            assert s["builds"] == 0, f"{name} built at a launch: {s}"
        assert not any(Path(cache_dir).rglob("*.o")), "a launch wrote the AOT cache"

        # The prebuild: the same calls under COMPILE_ONLY export every entry.
        _, state = run_mode("1", cache_dir, {"COMPILE_ONLY": "1"})
        state.pop("gemm factories")
        for name, s in state.items():
            assert s["builds"] == entries[name], f"{name} prebuild: {s}"
        built = {p: p.stat().st_mtime_ns for p in Path(cache_dir).rglob("*.o")}
        assert len(built) == sum(entries.values())

        hit, state = run_mode("1", cache_dir)
        assert state.pop("gemm factories") == 0, "a hit built a GEMM launcher"
        for name, s in state.items():
            # Every launcher still goes through the cache, and loads.
            assert s["entries"] == entries[name], f"{name} skipped the cache: {s}"
            assert s["jit"] == 0 and s["builds"] == 0, f"{name} traced: {s}"
        after = {p: p.stat().st_mtime_ns for p in Path(cache_dir).rglob("*.o")}
        assert after == built, "aot hit rewrote the AOT cache"

    assert jit == miss == hit, f"outputs differ:\n{jit}\n{miss}\n{hit}"


def test_gdn_h_and_pa_plan_through_aot_cache():
    from aiter.jit.utils.chip_info import get_gfx_runtime

    if get_gfx_runtime() not in ("gfx942", "gfx950"):
        pytest.skip(f"the GDN hidden-state kernel is not run on {get_gfx_runtime()}")
    if not hasattr(flyc, "load_module"):
        pytest.skip("FlyDSL has no flyc.load_module, so every mode would run the JIT")

    child = "--child-gdn-h-pa-plan"
    with tempfile.TemporaryDirectory() as cache_dir:
        jit, state = run_mode("0", cache_dir, child=child)
        assert all(s["entries"] == 0 for s in state.values()), state
        assert not any(Path(cache_dir).rglob("*.o")), "mode 0 wrote the AOT cache"

        # A launch never builds: with no entry it traces, as mode 0 does.
        miss, state = run_mode("1", cache_dir, child=child)
        for name, s in state.items():
            assert s["entries"] == s["jit"] == 2, f"{name}: two misses: {s}"
            assert s["builds"] == 0, f"{name} built at a launch: {s}"
        assert not any(Path(cache_dir).rglob("*.o")), "a launch wrote the AOT cache"

        # The prebuild: the same calls under COMPILE_ONLY export every entry.
        _, state = run_mode("1", cache_dir, {"COMPILE_ONLY": "1"}, child=child)
        for name, s in state.items():
            assert s["builds"] == 2, f"{name} prebuild: {s}"
        built = {p: p.stat().st_mtime_ns for p in Path(cache_dir).rglob("*.o")}

        hit, state = run_mode("1", cache_dir, child=child)
        for name, s in state.items():
            assert s["entries"] == 2, f"{name} skipped the cache: {s}"
            assert s["jit"] == 0 and s["builds"] == 0, f"{name} traced: {s}"
        after = {p: p.stat().st_mtime_ns for p in Path(cache_dir).rglob("*.o")}
        assert after == built, "aot hit rewrote the AOT cache"

    assert jit == miss == hit, f"outputs differ:\n{jit}\n{miss}\n{hit}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--child", action="store_true")
    group.add_argument("--child-gdn-h-pa-plan", action="store_true")
    if parser.parse_args().child:
        child()
    else:
        child_gdn_h_and_pa_plan()
