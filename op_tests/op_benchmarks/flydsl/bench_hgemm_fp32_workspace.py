# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paired full-path HGEMM timing. Run correctness op_test separately.

python op_tests/op_benchmarks/flydsl/bench_hgemm_fp32_workspace.py \
    --hip-solutions 440342 440676 440108 --json workspace.json

Solution IDs are hipBLASLt-version-specific and optional; no HIP IDs are
hardcoded into dispatch. Hot and rotating weights are different workloads.
GPU events time ALL kernels (including BF16 conversion), not Python overhead.
"""

import argparse
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl import flydsl_hgemm, flydsl_hgemm_fp32_workspace
from aiter.test_common import checkAllclose


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hip-solutions", type=int, nargs="*", default=[])
    p.add_argument("--rotate", type=int, default=16)
    p.add_argument("--nodes", type=int, default=128)
    p.add_argument("--rounds", type=int, default=16)
    p.add_argument("--seed", type=int, default=7201)
    p.add_argument("--json", type=Path, default=Path("workspace_results.json"))
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()
    if (
        args.rotate < 1
        or args.nodes < args.rotate
        or args.nodes % args.rotate
        or args.rounds < 2
    ):
        p.error("nodes must be a positive multiple of rotate; rounds >= 2")
    if get_gfx_runtime() != "gfx950" or get_cu_num() != 256:
        aiter.logger.warning("This benchmark requires gfx950 / 256 CU")
        return

    m, n, k = 128, 7168, 12288
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    need = args.rotate * 2 * (m * k + n * k + m * n) + 4 * (m * k + n * k + m * n)
    if torch.cuda.mem_get_info()[0] < need * 1.2:
        raise RuntimeError("Insufficient free VRAM; do not silently reduce rotation")
    buffers = [
        (
            torch.randn(m, k, device="cuda", dtype=torch.bfloat16),
            torch.randn(n, k, device="cuda", dtype=torch.bfloat16),
            torch.empty(m, n, device="cuda", dtype=torch.bfloat16),
        )
        for _ in range(args.rotate)
    ]
    workspace = torch.empty(m, n, device=buffers[0][0].device, dtype=torch.float32)
    funcs = {
        "split1": lambda a, w, o: flydsl_hgemm(
            a,
            w,
            out=o,
            block_m=64,
            block_n=64,
            block_k=64,
            stages=6,
            split_k=1,
            m_waves=2,
            n_waves=2,
        ),
        "workspace": lambda a, w, o: flydsl_hgemm_fp32_workspace(a, w, o, workspace),
    }
    funcs["workspace_AA"] = funcs["workspace"]
    if args.hip_solutions:
        aiter.hipb_create_extension()
        for sid in args.hip_solutions:
            funcs[f"hip{sid}"] = lambda a, w, o, sid=sid: aiter.hipb_mm(
                a, w.T, sid, out_dtype=torch.bfloat16, out=o
            )

    graphs = {}
    for name, fn in funcs.items():
        for tensors in buffers:
            fn(*tensors)
        torch.cuda.synchronize()
        graphs[name] = {}
        for mode in ("hot", "rotation"):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(args.nodes):
                    fn(*buffers[i % args.rotate if mode == "rotation" else 0])
            graphs[name][mode] = g

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "shape": [m, n, k],
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "aiter": aiter.__file__,
        "device": str(torch.cuda.get_device_properties(0)),
        "args": {
            key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()
        },
        "contract": "all GPU work of nodes full calls; 16 (default) A/W/C sets; "
        "one sequentially reused FP32 scratch. Not E2E or zero cache hit.",
        "checks": [],
        "paired": {},
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.json.write_text(json.dumps(result, indent=2))

    # Check every slot after modifying inputs, not only the last kernel output.
    for update in range(2):
        refs = []
        for a, w, _ in buffers:
            a.normal_()
            w.normal_()
            refs.append((a.float() @ w.float().T).bfloat16())
        for name, modes in graphs.items():
            for mode, g in modes.items():
                ids = range(args.rotate) if mode == "rotation" else [0]
                for i in ids:
                    buffers[i][2].fill_(float("nan"))
                g.replay()
                torch.cuda.synchronize()
                for i in ids:
                    err = checkAllclose(
                        refs[i].float(),
                        buffers[i][2].float(),
                        rtol=0.01,
                        atol=0.01,
                        tol_err_ratio=0,
                        printLog=False,
                    )
                    assert torch.isfinite(buffers[i][2]).all() and err == 0, (
                        name,
                        mode,
                        i,
                        err,
                    )
                result["checks"].append(
                    {"name": name, "mode": mode, "update": update, "strict01": True}
                )
    save()

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )

    def measure(g):
        for _ in range(2):
            g.replay()
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / args.nodes

    pairs = [("workspace", "workspace_AA"), ("split1", "workspace")]
    for sid in args.hip_solutions:
        pairs.extend([(f"hip{sid}", f"hip{sid}"), (f"hip{sid}", "workspace")])
    random.Random(args.seed).shuffle(pairs)
    rows = []
    for left, right in pairs:
        label = f"{left}__{right}"
        result["paired"][label] = {}
        for mode in ("rotation", "hot"):
            blocks = []
            for rep in range(args.rounds):
                times = [[], []]
                for side in ([0, 1, 1, 0] if rep % 2 == 0 else [1, 0, 0, 1]):
                    times[side].append(
                        measure(graphs[left if side == 0 else right][mode])
                    )
                a, b = map(statistics.mean, times)
                blocks.append(
                    {
                        "left": a,
                        "right": b,
                        "ratio": b / a,
                        "raw": times,
                        "spread": max(
                            abs(times[0][0] - times[0][1]) / a,
                            abs(times[1][0] - times[1][1]) / b,
                        ),
                    }
                )
            a = statistics.median(x["left"] for x in blocks)
            b = statistics.median(x["right"] for x in blocks)
            result["paired"][label][mode] = {
                "left_us": a,
                "right_us": b,
                "blocks": blocks,
                "paired_reduction_pct": 100
                * (1 - statistics.median(x["ratio"] for x in blocks)),
                "noisy_blocks": sum(x["spread"] > 0.05 for x in blocks),
            }
            for name, u in ((left, a), (right, b)):
                rows.append(
                    {
                        "pair": label,
                        "mode": mode,
                        "path": name,
                        "us": u,
                        "TFLOPS": 2 * m * n * k / u / 1e6,
                        "logical_TBs": 2 * (m * k + n * k + m * n) / u / 1e6,
                        "err": 0,
                    }
                )
            save()
    aiter.logger.info(
        "Full chain (logical TB/s, not measured HBM):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    if args.trace:
        # Separate eager diagnostic: not the graph from which timings were taken.
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(8):
                funcs["workspace"](*buffers[0])
                torch.cuda.synchronize()
        path = args.json.with_suffix(".trace.json")
        prof.export_chrome_trace(str(path))
        kernels = [
            e
            for e in json.loads(path.read_text())["traceEvents"]
            if e.get("cat") == "kernel"
        ]
        assert kernels, "GPU trace missing"
        result["eager_trace"] = {
            "file": str(path),
            "count": len(kernels),
            "names": sorted({e["name"] for e in kernels}),
        }
    result["complete"] = True
    save()
    print("SAVED", args.json, flush=True)


if __name__ == "__main__":
    main()
