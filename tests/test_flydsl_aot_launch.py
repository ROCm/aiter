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
