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
        _BakedArgumentError,
        _BakedKernel,
        _entry_current,
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
    meta = {"sources": {}, "env": {}, "baked": baked}
    assert _entry_current(meta, (1, 8, 64, "b", (3, 4)))
    assert not _entry_current(meta, (0, 4, 32, "b", (3, 4)))
    assert not _entry_current(meta, (0, 4, 64, "c", (3, 4)))
    assert not _entry_current(meta, (0, 4, 64, "b", (3, 5)))
    assert not _entry_current(meta, (0, 4, 64))
    # An entry written before values were recorded is not one to trust.
    with pytest.raises(KeyError):
        _entry_current({"sources": {}, "env": {}}, args)

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


def test_linked_bitcode_is_an_input_of_the_entry(monkeypatch, tmp_path):
    from aiter.ops.flydsl import aot_cache

    # Written after this process started, as a library extracted on first use.
    lib = tmp_path / "device.bc"
    lib.write_bytes(b"BC one")

    class Artifact:
        _link_libs = [str(lib)]

    class Compiled:
        _keepalive = Artifact()

        def export_to_c(self, out_dir, sym):
            Path(out_dir, f"{sym}.o").write_bytes(b"object")

    monkeypatch.setattr(aot_cache.flyc, "compile", lambda *a, **k: Compiled())
    cache = aot_cache.FlyDSLAotCache("test_link_libs")

    def make_exe():
        return lambda: None

    out_dir = tmp_path / "cache"
    stem = cache._build(make_exe, (), out_dir, "sym", "digest")
    meta = json.loads(stem.with_suffix(".json").read_text())
    assert str(lib) in meta["sources"]
    assert cache._find(out_dir, "digest", ()) == stem

    # A rebuilt library (e.g. another MORI) no longer matches the entry.
    lib.write_bytes(b"BC two")
    aot_cache._file_digest.cache_clear()
    assert cache._find(out_dir, "digest", ()) is None
    rebuilt = cache._build(make_exe, (), out_dir, "sym", "digest")
    assert rebuilt != stem and cache._find(out_dir, "digest", ()) == rebuilt


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
