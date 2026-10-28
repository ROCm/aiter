#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""C++ Gemma RMSNorm shape sweep: BF16, N=6144/128, no quantization.
Run each git revision in a separate process and AITER_JIT_DIR on an idle GPU.
HIP Graph captures 128 calls; 9 samples >=5 ms each; outputs are preallocated.
This is a warm-cache microbenchmark, not end-to-end model performance.
"""
import os

os.environ.setdefault("AITER_AOT_IMPORT", "1")
os.environ.setdefault("GPU_ARCHS", "gfx1250")
import argparse
import datetime
import gc
import hashlib
import json
import math
import random
import statistics
import subprocess
from pathlib import Path
import torch
import aiter
from aiter.utility import dtypes

aiter.dtypes = dtypes
from aiter.ops.rmsnorm import rmsnorm, add_rmsnorm
from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant

M6144 = [
    1,
    2,
    3,
    4,
    7,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
]
M128 = [
    1,
    2,
    3,
    4,
    7,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    112,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
    6144,
    8192,
    12288,
    16384,
    24576,
    32768,
    49152,
    65536,
    98304,
    131072,
    196608,
    262144,
    393216,
    524288,
]
EPS = 1e-6
GRAPH_CALLS = 128


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False, default=str) + "\n")
    tmp.replace(path)


def prepare(m, n, add):
    x = torch.randn((m, n), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((n,), dtype=torch.bfloat16, device="cuda")
    residual = torch.randn_like(x) if add else None
    out = torch.empty_like(x)
    res_out = torch.empty_like(x) if add else None
    return x, w, residual, out, res_out


def operation(path, tensors):
    x, w, r, y, ro = tensors
    if path == "atom":

        def fn():
            fused_qk_rmsnorm_group_quant(
                q=x,
                q_weight=w,
                q_epsilon=EPS,
                q_out_scale=None,
                q_out_unquantized=y,
                q_res_out=ro,
                q_residual=r,
                gemma_norm=True,
            )

    elif r is None:

        def fn():
            rmsnorm(y, x, w, EPS, True)

    else:

        def fn():
            add_rmsnorm(y, x, r, ro, w, EPS, True)

    return fn


def verify(tensors):
    x, w, r, y, ro = tensors
    xf = x.float()
    if r is not None:
        xf = xf + r.float()  # NOT rounded residual for normalization
    ref = (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + EPS)) * (1 + w.float())
    yf = y.float()
    torch.testing.assert_close(yf, ref, rtol=0.01, atol=0.01)
    residual_exact = None
    if r is not None:
        torch.testing.assert_close(ro, xf.to(torch.bfloat16), rtol=0, atol=0)
        residual_exact = True
    error = yf - ref
    abs_max = float(error.abs().max().item())
    rel_rms = float(
        (error.square().mean().sqrt() / (ref.square().mean().sqrt() + 1e-30)).item()
    )
    assert math.isfinite(abs_max) and math.isfinite(rel_rms)
    return {
        "max_abs_error": abs_max,
        "relative_rms_error": rel_rms,
        "residual_exact": residual_exact,
    }


def measure(fn, samples):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(GRAPH_CALLS):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(4):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    def elapsed(replays):
        start.record()
        for _ in range(replays):
            graph.replay()
        stop.record()
        stop.synchronize()
        ms = float(start.elapsed_time(stop))
        if not math.isfinite(ms) or ms <= 0:
            raise RuntimeError("Invalid GPU timer")
        return ms

    calibration = statistics.median(elapsed(1) for _ in range(3))
    replays = max(1, math.ceil(7.0 / calibration))
    raw = []
    for _ in range(samples):
        duration = elapsed(replays)
        while duration < 5.0:
            replays *= 2
            duration = elapsed(replays)
        raw.append(
            {
                "elapsed_ms": duration,
                "graph_replays": replays,
                "kernel_calls": replays * GRAPH_CALLS,
                "us": duration * 1000 / (replays * GRAPH_CALLS),
            }
        )
    return raw


def metadata(args):
    prop = torch.cuda.get_device_properties(0)
    source = Path(aiter.__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"], capture_output=True, text=True
    )
    from aiter.jit import core

    modules = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in Path(core.get_user_jit_dir()).glob("*.so")
    }
    return {
        "module_sha256": modules,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "started_utc": stamp(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": str(prop),
        "gcnArchName": prop.gcnArchName,
        "CU": prop.multi_processor_count,
        "aiter_commit": revision.stdout.strip(),
        "aiter_file": aiter.__file__,
        "seed": args.seed,
        "samples_per_case": args.samples,
        "graph_calls": GRAPH_CALLS,
        "minimum_sample_ms": 5,
        "dtype": "bfloat16",
        "gemma_norm": True,
        "epsilon": EPS,
        "warm_cache": True,
        "HIP_VISIBLE_DEVICES": os.getenv("HIP_VISIBLE_DEVICES"),
        "AITER_JIT_DIR": os.getenv("AITER_JIT_DIR"),
    }


def profile(out):
    workloads = []
    for n, m, add in [
        (6144, 128, False),
        (6144, 128, True),
        (6144, 256, False),
        (6144, 256, True),
        (128, 128, False),
    ]:
        t = prepare(m, n, add)
        for path in ("atom", "generic"):
            fn = operation(path, t)
            for _ in range(3):
                fn()
            workloads.append((f"{path}_{m}_{n}_{add}", fn))
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        for label, fn in workloads:
            with torch.profiler.record_function(label):
                for _ in range(3):
                    fn()
        torch.cuda.synchronize()
    prof.export_chrome_trace(str(out))
    obj = json.loads(Path(out).read_text())
    events = [e for e in obj.get("traceEvents", []) if e.get("cat") == "kernel"]
    names = sorted({e.get("name", "") for e in events})
    if not any("fused_qk_rmsnorm_group_quant_kernel" in n for n in names):
        raise RuntimeError("ATOM C++ kernel missing from profiler")
    if not any("add_rmsnorm_quant_kernel" in n for n in names):
        raise RuntimeError("generic C++ kernel missing from profiler")
    if any("triton" in n.lower() for n in names):
        raise RuntimeError("Unexpected Triton kernel in operator profile")
    dump(
        Path(out).with_suffix(".kernels.json"),
        {
            "names": names,
            "launches": [
                {"name": e["name"], "args": e.get("args"), "duration_us": e.get("dur")}
                for e in events
            ],
        },
    )
    print("PROFILE_OK", len(events), "kernel launches", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results")
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--profile-only", action="store_true")
    args = p.parse_args()
    torch.manual_seed(args.seed)
    if args.profile_only:
        profile(Path(args.out + ".json"))
        return
    tasks = []
    for n, ms, adds in [(6144, M6144, (False, True)), (128, M128, (False,))]:
        for m in ms:
            for add in adds:
                for path in ("atom", "generic"):
                    tasks.append((m, n, add, path))
    assert len(tasks) == 198
    if args.smoke:
        tasks = [
            (m, n, add, path)
            for m, n in ((1, 6144), (128, 6144), (32768, 6144), (1, 128), (524288, 128))
            for add in ((False, True) if n == 6144 else (False,))
            for path in ("atom", "generic")
        ]
    random.Random(args.seed).shuffle(tasks)
    meta = metadata(args)
    dump(args.out + "_meta.json", meta)
    results = []
    try:
        for i, (m, n, add, path) in enumerate(tasks):
            t = prepare(m, n, add)
            fn = operation(path, t)
            fn()
            torch.cuda.synchronize()
            checks = verify(t)
            raw = measure(fn, args.samples)
            verify(t)
            logical_bytes = (8 if add else 4) * m * n + 2 * n
            us = statistics.median(r["us"] for r in raw)
            row = {
                "M": m,
                "N": n,
                "path": path,
                "mode": "add_norm" if add else "norm",
                "median_us": us,
                "samples": raw,
                **checks,
                "logical_bytes": logical_bytes,
                "logical_TBps": logical_bytes / (us * 1e6),
            }
            results.append(row)
            dump(args.out + ".json", results)
            print(
                f"{i+1}/{len(tasks)} {path} {row['mode']} [{m},{n}] {us:.4f} us err={checks['max_abs_error']:.6g}",
                flush=True,
            )
            del fn, t
            gc.collect()
        meta.update(
            finished_utc=stamp(),
            completed=len(results),
            success=True,
            module_sha256=metadata(args)["module_sha256"],
        )
        dump(args.out + "_meta.json", meta)
    except Exception as e:
        meta.update(
            finished_utc=stamp(), completed=len(results), success=False, error=repr(e)
        )
        dump(args.out + "_meta.json", meta)
        raise


if __name__ == "__main__":
    main()
