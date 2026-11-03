# SPDX-License-Identifier: MIT
"""ROUND-LAUNCH: host-side per-call cost of the QSA wrapper, no device.

The real ``qsa_paged_mqa_logits_page4`` runs on CPU tensors with the gate's
decode/draft fixture shapes; ``_run`` (the FlyDSL launcher), the builders and
``torch.cuda.current_stream`` are stubbed.  Each wrapper phase is then timed in
isolation (same calls, same arguments) and, when the real ``flydsl`` package is
importable (the image), the FlyDSL ``CallState`` dispatch closure is rebuilt from
the real ``TorchTensorJitArg``/``Stream`` C-ABI fill specs with a no-op ctypes
``func_exe`` so the argument-marshalling cost is measured exactly.

Usage: python3 launch_overhead_cpu.py [--json out.json] [--pages N]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import sys
import time
from unittest.mock import patch

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from _load import load_kernel_module  # noqa: E402

K, HOW = load_kernel_module(allow_stub=True)


def timeit(fn, n=3000, rounds=5):
    for _ in range(min(n, 300)):
        fn()
    per = []
    for _ in range(rounds):
        t0 = time.perf_counter_ns()
        for _ in range(n):
            fn()
        per.append((time.perf_counter_ns() - t0) / n / 1e3)
    return statistics.median(per)


class _Launch:
    cols_per_block = 32
    dtype_tag = "stub"


def fixture(rows, requests, fmt, pages, width=16384, cols=65536):
    q = torch.randn(rows, K.NUM_HEADS, K.HEAD_DIM).to(torch.bfloat16)
    if fmt == "bf16":
        cache = torch.empty((pages, 4, 1, 128), dtype=torch.bfloat16)
        scale = None
    else:
        cache = torch.empty((pages, 4, 1, 128), dtype=torch.float8_e4m3fn)
        scale = torch.ones((pages, 4, 1), dtype=torch.float32)
    pt = torch.zeros((requests, width), dtype=torch.int32)
    t2r = torch.arange(requests, dtype=torch.int32).repeat_interleave(rows // requests)
    qp = torch.arange(rows, dtype=torch.int64)  # golden fixtures: int64 positions
    cl = torch.full((requests,), 4 * width, dtype=torch.int32)
    return (q, cache, pt, t2r, qp, cl), dict(compress_ratio=4, max_columns=cols,
                                           compressed_k_scale=scale)


def flydsl_dispatch_cost(args17):
    """Rebuild the real CallState for the 17-arg bf16 launch with a no-op func_exe."""
    try:
        from flydsl.compiler.jit_argument import TorchTensorJitArg
        from flydsl.compiler.jit_executor import CallState
        from flydsl.expr.typing import Stream
    except Exception as exc:  # noqa: BLE001 - Windows box has no flydsl
        return {"available": False, "reason": repr(exc)}
    specs = []
    for i, a in enumerate(args17):
        if isinstance(a, torch.Tensor):
            for ct, fill in TorchTensorJitArg(a).__c_abi_spec__():
                specs.append((i, ct, fill))
        elif isinstance(a, int):
            def fill(v, s):
                s.value = v
            specs.append((i, ctypes.c_int32, fill))
        else:  # stream
            for ct, fill in Stream(None).__c_abi_spec__():
                specs.append((i, ct, fill))
    libc = ctypes.CDLL(None)
    noop = libc.labs
    noop.argtypes = [ctypes.c_void_p]
    noop.restype = ctypes.c_long
    state = CallState(specs, noop)
    res = {"available": True, "slots": len(specs)}
    res["callstate_dispatch_us"] = timeit(lambda: state(args17))
    res["ctypes_noop_call_us"] = timeit(lambda: noop(None))
    raw = list(args17)
    raw[-1] = 0x1234  # int raw stream pointer (fast fill branch)
    raw = tuple(raw)
    res["callstate_dispatch_int_stream_us"] = timeit(lambda: state(raw))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    ap.add_argument("--pages", type=int, default=738964)
    a = ap.parse_args()
    out = {"how": HOW, "torch": torch.__version__, "python": sys.version.split()[0]}
    captured = []

    def run(launch, *args):
        captured.append(args)

    stubs = [patch.object(K, "_run", run),
             patch.object(K, "_build_wmma_kernel", lambda *a, **k: _Launch()),
             patch.object(K, "_build_wmma_fp8_kernel", lambda *a, **k: _Launch()),
             patch.object(torch.cuda, "current_stream", lambda dev=None: None)]
    for s in stubs:
        s.start()
    try:
        for name, rows, req, fmt, cols in (("decode_bf16_c65536", 64, 16, "bf16", 65536),
                                          ("decode_bf16_c32768", 64, 16, "bf16", 32768),
                                          ("draft_bf16_c65536", 16, 16, "bf16", 65536),
                                          ("decode_fp8_c32768", 64, 16, "fp8", 32768)):
            args, kw = fixture(rows, req, fmt, a.pages, cols=cols)
            kw["max_columns"] = cols
            rec = {}
            rec["wrapper_total_stubbed_launch_us"] = timeit(
                lambda: K.qsa_paged_mqa_logits_page4(*args, **kw), n=500)
            argsi32 = (args[0], args[1], args[2], args[3], args[4].to(torch.int32), args[5])
            rec["wrapper_total_i32_meta_us"] = timeit(
                lambda: K.qsa_paged_mqa_logits_page4(*argsi32, **kw), n=500)
            pre = dict(kw, logits=torch.empty((rows, cols), dtype=torch.float32),
                       visible_groups=torch.empty(rows, dtype=torch.int32),
                       row_starts=torch.empty(rows, dtype=torch.int32))
            rec["wrapper_total_prealloc_outputs_us"] = timeit(
                lambda: K.qsa_paged_mqa_logits_page4(*args, **pre), n=500)
            rec["wrapper_total_prealloc_outputs_i32_meta_us"] = timeit(
                lambda: K.qsa_paged_mqa_logits_page4(*argsi32, **pre), n=500)
            q, cache, pt, t2r, qp, cl = args
            dev = q.device
            # phases, same calls as the wrapper
            rec["phase_outputs_alloc_3x_empty_us"] = timeit(lambda: (
                torch.empty((rows, cols), dtype=torch.float32, device=dev),
                torch.empty(rows, dtype=torch.int32, device=dev),
                torch.empty(rows, dtype=torch.int32, device=dev)), n=200)
            rec["phase_check_plane_10x_us"] = timeit(lambda: [K._check_plane("x", 4096) for _ in range(10)])
            lg = torch.empty((rows, cols), dtype=torch.float32)
            rec["phase_check_align_9x_us"] = timeit(lambda: (
                K._check_align(q, "q", 16), K._check_align(cache, "k", 16), K._check_align(pt, "p", 4),
                K._check_align(lg, "l", 4), K._check_align(t2r, "v", 4), K._check_align(t2r, "r", 4),
                K._check_align(t2r, "t", 4), K._check_align(qp, "qp", 4), K._check_align(cl, "c", 4)))
            rec["phase_int64_clamp_qp_us(cpu op; cuda adds launch)"] = timeit(
                lambda: K._to_i32_saturated(qp, K._MAX_I32 - 1))
            q32 = K._to_i32_saturated(qp, K._MAX_I32 - 1)
            rec["phase_contig_view_3x_us"] = timeit(lambda: (t2r.contiguous().view(-1),
                                                           q32.contiguous().view(-1),
                                                           cl.contiguous().view(-1)))
            rec["phase_q_kv_u8_views_us"] = timeit(lambda: (cache.view(torch.uint8).view(cache.shape[0], -1),
                                                          q.view(torch.uint8).view(rows, -1)))
            vg = torch.empty(rows, dtype=torch.int32)
            rec["phase_out_view_3x_us"] = timeit(lambda: (vg.view(-1), vg.view(-1), vg.view(-1)))
            rec["phase_getenv_pad_bucket_us"] = timeit(lambda: os.environ.get("QSA_PAD_BUCKET", "0").strip() == "1")
            rec["phase_routing_us"] = timeit(lambda: (K.wmma_bf16_rows_per_block(rows, req),
                                                    K._resolve_bf16_k_load(16),
                                                    K.wmma_grid(rows, cols, 16, 32),
                                                    cache.data_ptr() % 128))
            rec["phase_validation_shape_dtype_us"] = timeit(lambda: (
                q.dtype != torch.bfloat16, q.ndim, tuple(q.shape[1:]), q.is_contiguous(),
                int(q.shape[0]), cache.ndim, tuple(cache.shape[1:]), cache.is_contiguous(),
                int(cache.shape[0]), cache.dtype == torch.float8_e4m3fn, pt.ndim, pt.shape[0],
                pt.dtype, pt.is_contiguous(), int(pt.shape[0]), int(pt.shape[1]),
                t2r.ndim, int(t2r.shape[0]), t2r.dtype, qp.ndim, int(qp.shape[0]), qp.dtype,
                cl.ndim, int(cl.shape[0]), cl.dtype, lg.device != dev, vg.device != dev, vg.device != dev))
            if captured:
                rec["flydsl"] = flydsl_dispatch_cost(captured[-1])
            out[name] = rec
            print(name, json.dumps(rec, indent=1), flush=True)
            captured.clear()
    finally:
        for s in stubs:
            s.stop()
    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
