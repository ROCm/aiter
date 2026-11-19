# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL AOT cache (``aiter/ops/flydsl/aot_cache.py``) on the preshuffle GEMM.

Each mode runs in a fresh process, because trace cost only shows there:

* ``jit``: ``AITER_FLYDSL_AOT_CACHE=0``. FlyDSL traces every config, even
  when its own disk cache is warm.
* ``aot miss``: ``=1`` with an empty cache directory. Traces and exports.
* ``aot hit``: ``=1`` again on the same directory. Loads the ``.o`` with
  ``flyc.load_module``, no trace.
* ``aot verify``: ``=verify`` on the same directory. Re-traces each hit and
  requires the re-exported object to match the cached one byte for byte.

Each mode records the first-call time, the steady-state time and the error
against torch. All modes must also produce bitwise identical outputs.
"""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
import time

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

# (tile_m, tile_n, tile_k, split_k) per (m, n, k)
CASES = {
    (1, 1024, 512): (16, 64, 128, 1),
    (200, 2048, 1024): (32, 64, 128, 1),
    (64, 1024, 2048): (16, 64, 128, 2),
    (128, 4096, 1024): (64, 128, 128, 1),
}
DTYPES = {"fp8": dtypes.fp8, "int8": dtypes.i8}
# mode name -> {((m, n, k), qdtype): record}, filled by main()
RESULTS = {}


def make_inputs(m, n, k, qdtype):
    torch.manual_seed(m * 7 + n + k)
    x = torch.randint(-8, 8, (m, k), device="cuda").float()
    w = torch.randint(-8, 8, (n, k), device="cuda").float()
    if qdtype != dtypes.i8:
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
    """Run every case once from a cold process; print one JSON line per case."""
    from aiter.ops.flydsl import flydsl_preshuffle_gemm_a8  # noqa: F401

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
        _, us = run_perftest(run_flydsl, x, w_shuffled, xs, ws, out, tm, tn, tk, sk)
        err = checkAllclose(
            run_torch(x, w, xs, ws),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"[{m}x{n}x{k} {qname}]",
            printLog=False,
        )
        digest = hashlib.sha256(out.view(torch.int16).cpu().numpy().tobytes())
        rec = {"first_ms": first_ms, "us": us, "err": err, "out": digest.hexdigest()}
        print("RESULT " + json.dumps([[m, n, k], qname, rec]), flush=True)


def run_mode(mode, cache_dir, cases):
    env = {
        **os.environ,
        "AITER_FLYDSL_AOT_CACHE": mode,
        "AITER_FLYDSL_AOT_CACHE_DIR": cache_dir,
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
    res = {}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            shape, qname, rec = json.loads(line[len("RESULT ") :])
            res[(tuple(shape), qname)] = rec
    return res


@benchmark()
def test_aot_cache(m, n, k, qdtype):
    ret = {}
    outs = set()
    for name, res in RESULTS.items():
        rec = res[((m, n, k), qdtype)]
        ret[f"{name} first ms"] = rec["first_ms"]
        ret[f"{name} us"] = rec["us"]
        ret[f"{name} err"] = rec["err"]
        outs.add(rec["out"])
    ret["bitwise equal"] = len(outs) == 1
    assert ret["bitwise equal"], f"{m}x{n}x{k} {qdtype}: AOT and JIT outputs differ"
    return ret


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument(
        "-q", "--qdtype", nargs="*", default=list(DTYPES), choices=list(DTYPES)
    )
    args = parser.parse_args()
    if args.child:
        child([(tuple(s), q) for s, q in json.loads(args.child)])
        return

    # The preshuffle GEMM this test drives targets gfx942 and gfx950.
    if get_gfx_runtime() not in ("gfx942", "gfx950"):
        aiter.logger.info("FlyDSL AOT cache test skipped on %s", get_gfx_runtime())
        return

    cases = list(itertools.product(CASES, args.qdtype))
    with tempfile.TemporaryDirectory() as cache_dir:
        RESULTS["jit"] = run_mode("0", cache_dir, cases)
        RESULTS["aot miss"] = run_mode("1", cache_dir, cases)
        RESULTS["aot hit"] = run_mode("1", cache_dir, cases)
        RESULTS["aot verify"] = run_mode("verify", cache_dir, cases)
    df = pd.DataFrame([test_aot_cache(m, n, k, q) for (m, n, k), q in cases])
    aiter.logger.info(
        "FlyDSL AOT cache summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
