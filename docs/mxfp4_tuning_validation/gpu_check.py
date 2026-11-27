# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
import argparse
import hashlib
import json
import os
import statistics
import sys
from functools import partial
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("mode")
p.add_argument("--baseline")
args = p.parse_args()
mode = args.mode
if mode == "old" and not args.baseline:
    p.error("old mode requires --baseline")
root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(root))
if mode == "old":
    os.environ["AITER_CONFIG_GEMM_A4W4"] = str(Path(args.baseline).resolve())
elif mode == "stock":
    os.environ["AITER_CONFIG_GEMM_A4W4"] = str(
        root / "aiter/configs/a4w4_blockscale_tuned_gemm.csv"
    )
else:
    os.environ.pop("AITER_CONFIG_GEMM_A4W4", None)
import torch
import triton

import aiter
from aiter.ops.gemm_op_a4w4 import gemm_a4w4, get_GEMM_config
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import assertAllclose
from aiter.utility.mxfp4_tuning import configure_mxfp4_tuning
from op_tests.test_gemm_a4w4 import run_torch

if mode not in ("old", "stock", "default"):
    configure_mxfp4_tuning(profile=mode)
print(
    json.dumps(
        {
            "mode": mode,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "triton": triton.__version__,
            "aiter": aiter.__file__,
            "device": str(torch.cuda.get_device_properties(0)),
        }
    ),
    flush=True,
)
quant = aiter.get_triton_quant(aiter.QuantType.per_1x32)
torch.manual_seed(7538)
for m, n, k in [
    (1, 512, 4096),
    (128, 7168, 4608),
    (1024, 1024, 8192),
    (1023, 1024, 8192),
    (8192, 4096, 1024),
    (8191, 4096, 1024),
]:
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda")
    xq, xs = quant(x, shuffle=False)
    wq, ws = quant(w, shuffle=False)
    _, xss = quant(x, shuffle=True)
    _, wss = quant(w, shuffle=True)
    wsh = shuffle_weight(wq, layout=(16, 16))
    ref = run_torch(xq, wq, xs.view(torch.uint8), ws.view(torch.uint8), torch.bfloat16)
    fn = partial(gemm_a4w4, xq, wsh, xss, wss, dtype=torch.bfloat16)
    result = fn()
    err = assertAllclose(ref, result, msg=str((mode, m, n, k)), catastrophic_check=True)
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(50):
        a = torch.cuda.Event(enable_timing=True)
        b = torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):
            fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1000 / 5)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    kernels = sorted(
        {
            e.name
            for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
        }
    )
    cfg = get_GEMM_config(m, n, k)
    digest = hashlib.sha256(
        result.cpu().contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    print(
        json.dumps(
            {
                "mode": mode,
                "shape": [m, n, k],
                "config": (
                    None
                    if cfg is None
                    else {"kernelName": cfg["kernelName"], "splitK": int(cfg["splitK"])}
                ),
                "median_us": statistics.median(times),
                "reference_error": err,
                "gpu_kernels": kernels,
                "output_sha256": digest,
            }
        ),
        flush=True,
    )
