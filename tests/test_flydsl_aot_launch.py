# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""``_run_compiled`` launches routed through ``FlyDSLAotCache`` by
``aiter/ops/flydsl/aot_launch.py``, on the gfx942 whole-MoE backend.

Its call sites still call ``_run_compiled`` with a torch stream. Each mode runs
in a fresh process on the prefill rows (decode accumulates with atomics, so its
output is not bitwise reproducible):

* ``jit``: ``AITER_FLYDSL_AOT_CACHE=0``. Nothing is bound or written.
* ``aot miss``: ``=1`` on an empty directory. Every launch exports.
* ``aot hit``: ``=1`` again. Every launch loads its object, nothing builds.

All modes must produce bitwise identical outputs.
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
    assert made == [1] and len(ran) == n and not binding.kernels


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
            Path(out_dir, f"{sym}.o").write_bytes(b"object")

    monkeypatch.setattr(aot_cache.flyc, "compile", lambda *a, **k: Compiled())

    def make_exe():
        return lambda: None

    out_dir = tmp_path / "cache"
    cache = aot_cache.FlyDSLAotCache("test_link_libs_undeclared")
    with pytest.raises(aot_cache._UndeclaredSourceError, match=r"device\.bc"):
        cache._build(make_exe, (), out_dir, "sym", "digest")
    assert not list(out_dir.glob("*.o")), "an entry was written"
    # Never a silent fall back to the JIT path, in any mode.
    monkeypatch.setattr(aot_cache, "_MODE", "1")
    monkeypatch.setattr(aot_cache, "_load_module", lambda path: None)
    monkeypatch.setenv("AITER_FLYDSL_AOT_CACHE_DIR", str(tmp_path / "root"))
    with pytest.raises(aot_cache._UndeclaredSourceError):
        cache._kernel(("k",), make_exe, ())

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
        Path(out_dir, f"{sym}.o").write_bytes(b"object")


def _old_file(path, text):
    """A source written before this process started, as an installed file."""
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
    with pytest.raises(aot_cache._UsageError, match="does not exist"):
        aot_cache.FlyDSLAotCache("test_missing_source", sources=["kernels/nope.py"])


def test_source_edited_after_this_process_imported_it_is_not_cached(
    monkeypatch, tmp_path
):
    from aiter.ops.flydsl import aot_cache

    monkeypatch.setattr(aot_cache.flyc, "compile", lambda *a, **k: _FakeCompiled())
    # Written now: after this process imported aiter's FlyDSL code, so what it
    # would trace is not what the file says.
    src = tmp_path / "edited.py"
    src.write_text("X = 2\n")
    cache = aot_cache.FlyDSLAotCache("test_edited", sources=[src])
    with pytest.raises(RuntimeError, match="changed after"):
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

    def export(exe, args, out_dir, sym, libs=None):
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
def test_every_cache_declares_the_kernel_files_its_sources_import():
    """Sources are declared by hand. A forgotten one is found here, and by the
    cache when it builds an entry: a declared file under ``kernels/`` imports
    a file under ``kernels/`` that is not declared."""
    import importlib
    import re

    from aiter.ops.flydsl import aot_cache

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
    missing = {
        name: [os.path.relpath(f, base) for f in undeclared]
        for name, cache in real.items()
        if (undeclared := aot_cache._undeclared_imports(cache._files))
    }
    assert not missing, "\n".join(
        f"cache {name!r} must also declare sources={files}"
        for name, files in sorted(missing.items())
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
        (tmp_path / f"d-{i}.o").write_bytes(b"object")
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
    for token, md, inter, e, topk, quant, act, config in picked.values():
        impl, request, _, _ = t.check_moe(
            token, md, inter, e, topk, quant, act, "random", config
        )
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


def run_mode(mode, cache_dir):
    env = {
        **os.environ,
        "AITER_FLYDSL_AOT_CACHE": mode,
        "AITER_FLYDSL_AOT_CACHE_DIR": cache_dir,
    }
    proc = subprocess.run(
        [sys.executable, __file__, "--child"],
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
    assert outs, f"mode {mode} ran no prefill row"
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

        miss, state = run_mode("1", cache_dir)
        state.pop("gemm factories")
        entries = {name: s["entries"] for name, s in state.items()}
        for name, s in state.items():
            assert s["entries"] > 0, f"{name} launched nothing through the cache"
            assert s["jit"] == 0, f"{name} fell back to the JIT: {s}"
            assert s["builds"] == s["entries"], f"{name} did not export: {s}"
        built = {p: p.stat().st_mtime_ns for p in Path(cache_dir).rglob("*.o")}

        hit, state = run_mode("1", cache_dir)
        assert state.pop("gemm factories") == 0, "a hit built a GEMM launcher"
        for name, s in state.items():
            # Every launcher still goes through the cache, and loads.
            assert s["entries"] == entries[name], f"{name} skipped the cache: {s}"
            assert s["jit"] == 0 and s["builds"] == 0, f"{name} rebuilt: {s}"
        after = {p: p.stat().st_mtime_ns for p in Path(cache_dir).rglob("*.o")}
        assert after == built, "aot hit rewrote the AOT cache"

    assert jit == miss == hit, f"outputs differ:\n{jit}\n{miss}\n{hit}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", required=True)
    parser.parse_args()
    child()
