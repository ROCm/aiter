# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL AOT cache (``aiter/ops/flydsl/aot_cache.py``) on the preshuffle GEMM.

Each mode runs in a fresh process, because trace cost only shows there:

* ``jit``: ``AITER_FLYDSL_AOT_CACHE=0``. FlyDSL traces every config, even
  when its own disk cache is warm.
* ``aot miss``: ``=1`` with an empty cache directory. Traces, as ``jit``
  does, and writes nothing. A ``COMPILE_ONLY`` run then prebuilds the entries.
* ``aot hit``: ``=1`` again on the same directory. Loads the ``.o`` with
  ``flyc.load_module``, no trace.
* ``aot verify``: ``=verify`` on the same directory. Re-traces each hit and
  requires the re-exported object to match the cached one byte for byte.

Each mode records the first-call time and the error against torch. All modes must also produce bitwise identical outputs, and each
must have taken the path it names: the prebuild exports an ``.o``/``.json``
pair per entry, a miss and a hit write nothing, and a hit does not trace. The
test skips with a FlyDSL that has no ``flyc.load_module``, where every mode
would run the JIT.

It also checks that another value of a declared variable selects other
entries.
"""

import argparse
import functools
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import flydsl.compiler as flyc
import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import checkAllclose

# (tile_m, tile_n, tile_k, split_k) per (m, n, k)
CASES = {
    (200, 2048, 1024): (32, 64, 128, 1),
    (64, 1024, 2048): (16, 64, 128, 2),
}
DTYPES = {"fp8": dtypes.fp8}
# mode name -> {((m, n, k), qdtype): record}, filled by check_paths()
RESULTS = {}
# Read while building, as a kernel reads a tuning switch.
TEST_ENV = "AITER_TEST_AOT_CACHE_VARIANT"


def make_inputs(m, n, k, qdtype):
    torch.manual_seed(m * 7 + n + k)
    x = torch.randint(-8, 8, (m, k), device="cuda").float()
    w = torch.randint(-8, 8, (n, k), device="cuda").float()
    x, w = x / 4, w / 4
    xs = torch.rand(m, 1, device="cuda") + 0.5
    ws = torch.rand(1, n, device="cuda") + 0.5
    return x.to(qdtype), w.to(qdtype), xs, ws


def run_torch(x, w, xs, ws):
    return (x.float() @ w.float().t()) * xs * ws


def run_flydsl(x, w_shuffled, xs, ws, out, tm, tn, tk, sk):
    aiter.ops.flydsl.flydsl_preshuffle_gemm_a8(
        x, w_shuffled, xs, ws, out, tm, tn, tk, split_k=sk
    )
    return out


def child(cases):
    """Run every case once from a cold process; print one JSON line per case,
    then one with the path the cache took."""
    from aiter.ops.flydsl import (
        flydsl_preshuffle_gemm_a8,  # noqa: F401
        gemm_kernels,
    )
    from aiter.ops.flydsl.aot_cache import _JitKernel

    # Every trace, exported or JIT, goes through flyc.compile: count those.
    traces = 0
    compile = flyc.compile

    @functools.wraps(compile)
    def counting_compile(*args, **kwargs):
        nonlocal traces
        traces += 1
        return compile(*args, **kwargs)

    flyc.compile = counting_compile

    # A hit never calls the launcher factory: count those calls. The factory
    # also reads a variable, as a kernel reads a tuning switch, which the
    # cache declares.
    cache = gemm_kernels._preshuffle_launcher.aot_cache
    cache._env = (*cache._env, TEST_ENV)
    factories = 0
    compile_fn = gemm_kernels._get_compile_fn

    @functools.wraps(compile_fn)
    def reading_compile_fn():
        nonlocal factories
        factories += 1
        os.environ.get(TEST_ENV)
        return compile_fn()

    gemm_kernels._get_compile_fn = reading_compile_fn

    for (m, n, k), qname in cases:
        tm, tn, tk, sk = CASES[(m, n, k)]
        x, w, xs, ws = make_inputs(m, n, k, DTYPES[qname])
        w_shuffled = shuffle_weight(w, layout=(16, 16))
        out = torch.empty(m, n, dtype=dtypes.bf16, device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        run_flydsl(x, w_shuffled, xs, ws, out, tm, tn, tk, sk)
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - t0) * 1e3
        err = checkAllclose(
            run_torch(x, w, xs, ws),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"[{m}x{n}x{k} {qname}]",
            printLog=False,
        )
        digest = hashlib.sha256(out.view(torch.int16).cpu().numpy().tobytes())
        rec = {"first_ms": first_ms, "err": err, "out": digest.hexdigest()}
        print("RESULT " + json.dumps([[m, n, k], qname, rec]), flush=True)
    cache = gemm_kernels._preshuffle_launcher.aot_cache
    kernels = list(cache._mem.values())
    state = {
        "entries": len(kernels),
        "jit": sum(isinstance(k, _JitKernel) for k in kernels),
        "builds": cache._builds,
        "traces": traces,
        "factories": factories,
    }
    print("STATE " + json.dumps(state), flush=True)


def run_mode(mode, cache_dir, cases, extra_env=None):
    env = {
        **os.environ,
        "AITER_FLYDSL_AOT_CACHE": mode,
        "AITER_FLYDSL_AOT_CACHE_DIR": cache_dir,
        **(extra_env or {}),
    }
    arg = json.dumps([[list(s), q] for s, q in cases])
    proc = subprocess.run(
        [sys.executable, __file__, "--child", arg],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"mode {mode} failed:\n{proc.stdout}\n{proc.stderr}")
    res, state = {}, None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            shape, qname, rec = json.loads(line[len("RESULT ") :])
            res[(tuple(shape), qname)] = rec
        elif line.startswith("STATE "):
            state = json.loads(line[len("STATE ") :])
    return res, state


def cache_files(cache_dir):
    """Exported entries under ``cache_dir``: path -> mtime, ``.o`` and ``.json``."""
    return {
        str(p): p.stat().st_mtime_ns
        for p in Path(cache_dir).rglob("*")
        if p.suffix in (".o", ".json")
    }


def check_paths(cache_dir, cases):
    """Run every mode on ``cache_dir`` and assert the path each one took."""
    n = len(cases)

    RESULTS["jit"], state = run_mode("0", cache_dir, cases)
    assert state["entries"] == 0, f"mode 0 launched through the cache: {state}"
    assert state["traces"] == n, f"mode 0 did not trace every case: {state}"
    assert not cache_files(cache_dir), "mode 0 wrote the AOT cache"

    # A process that launches never builds: a miss traces, as mode 0 does.
    RESULTS["aot miss"], state = run_mode("1", cache_dir, cases)
    assert state["jit"] == state["entries"] > 0, f"aot miss: {state}"
    assert state["builds"] == 0, f"aot miss built an entry: {state}"
    assert not cache_files(cache_dir), "a launch wrote the AOT cache"

    # The prebuild: the same calls under COMPILE_ONLY export every entry and
    # launch nothing.
    _, state = run_mode("1", cache_dir, cases, {"COMPILE_ONLY": "1"})
    entries = state["builds"]
    assert 0 < entries <= n, f"prebuild: {state}"
    assert state["entries"] == 0, f"prebuild launched: {state}"
    built = cache_files(cache_dir)
    objects = {p[: -len(".o")] for p in built if p.endswith(".o")}
    sidecars = {p[: -len(".json")] for p in built if p.endswith(".json")}
    assert len(objects) == entries, f"{entries} entries, objects: {sorted(objects)}"
    assert objects == sidecars, f"unpaired entries: {sorted(objects ^ sidecars)}"
    for p in sidecars:
        env = json.loads(Path(p + ".json").read_text())["env"]
        assert TEST_ENV in env and env[TEST_ENV] is None, f"{p}.json: {env}"

    RESULTS["aot hit"], state = run_mode("1", cache_dir, cases)
    expected = {"entries": entries, "jit": 0, "builds": 0, "traces": 0, "factories": 0}
    assert state == expected, f"aot hit traced or built: {state}"
    assert cache_files(cache_dir) == built, "aot hit rewrote the AOT cache"

    RESULTS["aot verify"], state = run_mode("verify", cache_dir, cases)
    assert state["jit"] == 0 and state["builds"] == 0, f"aot verify: {state}"
    assert cache_files(cache_dir) == built, "aot verify rewrote the AOT cache"

    # Another value of a declared variable selects other entries: none here.
    _, state = run_mode("1", cache_dir, cases, {TEST_ENV: "1"})
    assert state["jit"] == state["entries"] > 0, f"a declared variable hit: {state}"
    assert cache_files(cache_dir) == built, "a miss rewrote the AOT cache"


def test_aot_cache():
    # The preshuffle GEMM this test drives targets gfx942 and gfx950.
    if get_gfx_runtime() not in ("gfx942", "gfx950"):
        pytest.skip(f"FlyDSL preshuffle GEMM does not target {get_gfx_runtime()}")
    if not hasattr(flyc, "load_module"):
        pytest.skip(
            f"FlyDSL {getattr(sys.modules['flydsl'], '__version__', '?')} has no "
            "flyc.load_module, so every mode would run the JIT"
        )

    cases = list(itertools.product(CASES, DTYPES))
    with tempfile.TemporaryDirectory() as cache_dir:
        check_paths(cache_dir, cases)
    rows = []
    for (m, n, k), qname in cases:
        row = {"shape": f"{m}x{n}x{k}", "qdtype": qname}
        outs = set()
        for name, res in RESULTS.items():
            rec = res[((m, n, k), qname)]
            row[f"{name} first ms"] = rec["first_ms"]
            row[f"{name} err"] = rec["err"]
            outs.add(rec["out"])
        assert len(outs) == 1, f"{m}x{n}x{k} {qname}: AOT and JIT outputs differ"
        rows.append(row)
    aiter.logger.info(
        "FlyDSL AOT cache summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    # run_mode() starts this file as a cold child process.
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", required=True)
    child([(tuple(s), q) for s, q in json.loads(parser.parse_args().child)])
