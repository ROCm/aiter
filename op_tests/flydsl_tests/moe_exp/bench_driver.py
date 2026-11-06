# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kernel timing of the FlyDSL grouped MoE kernels, driven through test_flydsl_grouped_gemm.py.

The test builds the inputs, runs the MoE once (the output its accuracy check uses), then calls
aiter.test_common.run_perftest once per kernel. For each selected kernel this driver replaces that call with:

  1. one launch on each rotated weight copy, then a synchronize (warm-up);
  2. ``MOE_EXP_IDLE_S`` seconds of idle GPU, so every process starts its rounds from the same state;
  3. ``MOE_EXP_ROUNDS`` rounds. A round is ``MOE_EXP_DISPATCHES`` back-to-back launches between two CUDA events,
     then a synchronize. ``[moe-exp] <kernel>: event_us = ...`` is the median over the rounds of the event time
     per dispatch.

Launches rotate over ``MOE_EXP_ROTATE`` copies of the weights (W and W scales; ``MOE_EXP_ROTATE_A=1`` also copies A
and its scales), so every launch reads them from HBM. The driver starts no profiler: run it under
``rocprofv3 --kernel-trace`` (bench.sh's default) for the kernel time of every dispatch. The host timestamps of every
round (time.monotonic_ns, the clock of rocprofv3's timestamps) go to ``<MOE_EXP_SAVE_OUT>.rounds.json``, which
summarize.py uses to split the kernel trace into rounds.

env:
  MOE_EXP_ROUNDS      rounds (default 50)
  MOE_EXP_DISPATCHES  dispatches per round (default 128; a multiple of MOE_EXP_ROTATE)
  MOE_EXP_KERNELS     timed kernels: comma list of quant_a1, gemm1, gemm2, or all (default gemm1)
  MOE_EXP_ROTATE      weight copies (default 4); MOE_EXP_ROTATE_A=1 also rotates A and its scales
  MOE_EXP_IDLE_S      idle seconds before the rounds (default 1)
  MOE_EXP_SAVE_OUT    <file.pt>: save the MoE output (for a bit-exact comparison) and <file.pt>.rounds.json

``CLOCK_ANCHOR <time.time_ns> <time.monotonic_ns>`` lines tie the rounds to the wall clock of an external log, such as
bench.sh's optional power-management log.

usage (from the repository root):
    python op_tests/flydsl_tests/moe_exp/bench_driver.py <test_flydsl_grouped_gemm.py args>
"""

import functools
import gc
import importlib.util
import json
import os
import statistics
import sys
import time

import torch

from aiter import test_common

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = os.path.join(os.path.dirname(HERE), "test_flydsl_grouped_gemm.py")
ALL_KERNELS = ("quant_a1", "gemm1", "gemm2")
_k = os.environ.get("MOE_EXP_KERNELS", "gemm1").strip()
KERNELS = ALL_KERNELS if _k in ("", "all") else tuple(_k.split(","))
ROTATE = int(os.environ.get("MOE_EXP_ROTATE", "4"))
ROTATE_A = os.environ.get("MOE_EXP_ROTATE_A", "0") == "1"
ROUNDS = int(os.environ.get("MOE_EXP_ROUNDS", "50"))
DISPATCHES = int(os.environ.get("MOE_EXP_DISPATCHES", "128"))
IDLE_S = float(os.environ.get("MOE_EXP_IDLE_S", "1"))
SAVE_OUT = os.environ.get("MOE_EXP_SAVE_OUT")
for _name in KERNELS:
    if _name not in ALL_KERNELS:
        raise SystemExit(f"MOE_EXP_KERNELS: unknown kernel {_name!r} (use {ALL_KERNELS} or all)")
if ROTATE < 1 or ROUNDS < 1 or DISPATCHES < 1:
    raise SystemExit("MOE_EXP_ROTATE, MOE_EXP_ROUNDS and MOE_EXP_DISPATCHES must be >= 1")
if DISPATCHES % ROTATE:
    raise SystemExit(f"MOE_EXP_DISPATCHES={DISPATCHES} must be a multiple of MOE_EXP_ROTATE={ROTATE}")

# Positional arguments of flydsl_grouped_gemm_a8w4_masked(out, a, w, a_scales, w_scales, m_tile_map, ...)
_A, _W, _AS, _WS = 1, 2, 3, 4
_rounds = {}  # kernel name -> {"dispatches": D, "rounds": [{"t": [...], "event_us": ...}]}


def kernel_name(func):
    """Which stage a kernel-bench launch callable (a functools.partial) runs."""
    fn = getattr(func, "func", None)
    kw = getattr(func, "keywords", None) or {}
    name = getattr(fn, "__name__", "")
    if name == "flydsl_grouped_gemm_a8w4_masked":
        return "gemm1" if (kw.get("stage1_act") or kw.get("gemm_stage") == 1) else "gemm2"
    if name == "flydsl_moe_fused_quant_preshuffle":
        return "quant_a1" if kw.get("topids_to_rows") is not None else "quant_a2"
    return None


def storage_copy(t):
    """A tensor like ``t`` (dtype, shape, strides, storage offset) over a new copy of its whole storage."""
    src = t.untyped_storage()
    dst = torch.empty(src.nbytes(), dtype=torch.uint8, device=t.device).untyped_storage()
    dst.copy_(src)
    out = torch.empty(0, dtype=t.dtype, device=t.device)
    out.set_(dst, t.storage_offset(), t.size(), t.stride())
    return out


def rotated_calls(name, func):
    """The launch callables to cycle through, and the bytes each extra copy holds."""
    if ROTATE == 1 or not name.startswith("gemm"):
        return [func], 0
    idx = [_W, _WS] + ([_A, _AS] if ROTATE_A else [])
    calls = [func]
    for _ in range(ROTATE - 1):
        args = list(func.args)
        for i in idx:
            args[i] = storage_copy(args[i])
        calls.append(functools.partial(func.func, *args, **func.keywords))
    torch.cuda.synchronize()
    return calls, sum(func.args[i].untyped_storage().nbytes() for i in idx)


def clock_anchor():
    print(f"CLOCK_ANCHOR {time.time_ns()} {time.monotonic_ns()}", flush=True)


def time_rounds(calls):
    """Warm-up, idle, then ROUNDS rounds of DISPATCHES launches; event time per dispatch and host stamps per round."""
    for f in calls:
        f()
    torch.cuda.synchronize()
    time.sleep(IDLE_S)
    clock_anchor()
    rounds = []
    n = len(calls)
    gc.collect()
    gc.disable()
    for _ in range(ROUNDS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        t1 = time.monotonic_ns()
        start.record()
        t2 = time.monotonic_ns()
        for i in range(DISPATCHES):
            calls[i % n]()
        t3 = time.monotonic_ns()
        end.record()
        t4 = time.monotonic_ns()
        end.synchronize()
        t5 = time.monotonic_ns()
        rounds.append({"t": [t1, t2, t3, t4, t5], "event_us": start.elapsed_time(end) * 1000.0 / DISPATCHES})
    gc.enable()
    clock_anchor()
    return rounds


_orig_run_perftest = test_common.run_perftest


def run_perftest(func, *args, **kwargs):
    # The test's kernel-bench loop passes a bare launch callable with testGraph=False; anything else is not ours.
    if args or kwargs.get("testGraph", False):
        return _orig_run_perftest(func, *args, **kwargs)
    name = kernel_name(func) or f"callable{len(_rounds)}"
    if name not in KERNELS:
        print(f"[moe-exp] {name}: not timed (MOE_EXP_KERNELS={','.join(KERNELS)})", flush=True)
        return None, float("nan")
    calls, nbytes = rotated_calls(name, func)
    if nbytes:
        what = "W, W scales" + (", A, A scales" if ROTATE_A else "")
        print(f"[moe-exp] {name}: rotating {ROTATE} copies of {what} ({nbytes / 1e6:.1f} MB per copy)", flush=True)
    rounds = time_rounds(calls)
    _rounds[name] = {"dispatches": DISPATCHES, "rounds": rounds}
    us = [r["event_us"] for r in rounds]
    med = statistics.median(us)
    print(
        f"[moe-exp] {name}: event_us = {med:.2f} (median of {ROUNDS} rounds x {DISPATCHES} dispatches, "
        f"range {min(us):.2f}-{max(us):.2f}, after {IDLE_S:g} s idle)",
        flush=True,
    )
    return None, med


test_common.run_perftest = run_perftest

sys.path.insert(0, os.path.dirname(TEST))
spec = importlib.util.spec_from_file_location("test_flydsl_grouped_gemm", TEST)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
_orig_run = mod._run_grouped_via_fused_moe


def _run_grouped_via_fused_moe(*args, **kwargs):
    out, ref, us, kernel_us = _orig_run(*args, **kwargs)
    if SAVE_OUT:
        with open(SAVE_OUT + ".rounds.json", "w") as f:
            json.dump({"pid": os.getpid(), "kernels": _rounds}, f)
        torch.save(out.detach().cpu(), SAVE_OUT)
        print(f"[moe-exp] saved output {SAVE_OUT} {tuple(out.shape)}", flush=True)
    return out, ref, us, kernel_us


mod._run_grouped_via_fused_moe = _run_grouped_via_fused_moe
print(
    f"[moe-exp] pid {os.getpid()}; kernels {','.join(KERNELS)}; {ROUNDS} rounds x {DISPATCHES} dispatches; "
    f"rotate {ROTATE}{' (W and A)' if ROTATE_A else ' (W only)'}",
    flush=True,
)
sys.argv = [TEST] + sys.argv[1:]
mod.main()
