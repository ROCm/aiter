# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Same-config HGEMM K-stagger A/B, without profiler instrumentation.

By default each graph walks at least 512 MiB of distinct weights. The reported
time is per GEMM, with no allocation/copy in the graph. This is a streaming
proxy, not AKLR's per-call 1-GiB cache-flush benchmark. --rotate-mib 0 measures
hot replay instead. Compilation and correctness checks are outside timing.

    python op_tests/op_benchmarks/flydsl/bench_hgemm_k_stagger.py --json result.json
    python op_tests/op_benchmarks/flydsl/bench_hgemm_k_stagger.py --rotate-mib 0
    python op_tests/op_benchmarks/flydsl/bench_hgemm_k_stagger.py --aklr-root /path/to/AKLR

--aklr-root optionally benchmarks the embedded AKLR kernel too, not its
dispatch fallback. The source path and commit are recorded in the results.
"""

import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import tempfile

import torch
from triton.testing import do_bench_cudagraph

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_hgemm

# M, N, K, BM, BN, BK, stages, WM, WN. First two are existing AITER tuned tiles.
CASES = [
    (16, 18432, 1536, 16, 96, 64, 6, 1, 2),
    (32, 18432, 1536, 32, 96, 64, 4, 2, 2),
    (32, 20480, 7168, 32, 64, 128, 5, 2, 2),
    (112, 20480, 7168, 128, 128, 64, 5, 4, 2),
    (16, 6144, 16384, 16, 64, 128, 5, 1, 2),
    (32, 6144, 16384, 32, 64, 128, 5, 2, 2),
    (112, 6144, 16384, 64, 64, 128, 5, 2, 2),
    (16, 4096, 4096, 16, 64, 128, 5, 1, 2),
    (16, 18432, 1536, 16, 96, 128, 5, 1, 2),
    (32, 18432, 1536, 32, 96, 128, 5, 2, 2),
    (64, 18432, 1536, 64, 96, 128, 3, 2, 2),
    (112, 18432, 1536, 128, 96, 128, 2, 2, 2),
    (16, 20480, 7168, 16, 96, 128, 5, 1, 2),
    (32, 20480, 7168, 32, 96, 128, 5, 2, 2),
    (64, 20480, 7168, 64, 96, 128, 3, 2, 2),
    (112, 20480, 7168, 128, 96, 128, 2, 2, 2),
]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def aklr_kernel(root, m, n, k, directory):
    """Extract only an explicitly supplied repository's matching kernel."""
    for path in sorted(
        (root / "aklr/ops/gemm/impls/flydsl" / f"gemm_n{n}_k{k}").glob("*.json")
    ):
        spec = json.loads(path.read_text())
        if not any(w["axes"].get("m") == m for w in spec["workloads"]):
            continue
        sources = spec["kernel"]["sources"]
        if len(sources) != 1 or sources[0]["path"] != "kernel.py":
            raise ValueError(f"Expected a single kernel.py in {path}")
        file = Path(directory) / f"aklr_{n}_{k}.py"
        file.write_text(sources[0]["content"])
        module = load_module(f"aklr_stagger_bench_{n}_{k}", file)
        return module.run, str(path.relative_to(root))
    return None, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rotate-mib", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=9)
    parser.add_argument("--rep-ms", type=float, default=60)
    parser.add_argument("--b-non-temporal", action="store_true")
    parser.add_argument(
        "--tuned", action="store_true", help="also measure current AITER dispatch"
    )
    parser.add_argument("--stagger", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--case", type=int, nargs="+", default=list(range(len(CASES))))
    parser.add_argument("--aklr-root", type=Path)
    parser.add_argument(
        "--baseline-kernel", type=Path, help="optional unmodified gfx950 kernel source"
    )
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if get_gfx() != "gfx950":
        parser.error("requires gfx950")
    if args.rotate_mib < 0 or args.repeat < 1 or args.rep_ms <= 0:
        parser.error("invalid timing arguments")
    if any(s <= 0 for s in args.stagger):
        parser.error("--stagger values must be positive")
    baseline = None
    if args.baseline_kernel:
        baseline = load_module(
            "aiter.ops.flydsl.kernels.stagger_bench_upstream", args.baseline_kernel
        ).gemm_a16w16
    import importlib.metadata

    report = {
        "torch": torch.__version__,
        "flydsl": importlib.metadata.version("flydsl"),
        "gpu": str(torch.cuda.get_device_properties(0)),
        "mode": "rotating weights" if args.rotate_mib else "hot replay",
        "rotate_mib": args.rotate_mib,
        "repeat": args.repeat,
        "rep_ms": args.rep_ms,
        "rows": [],
    }
    if args.aklr_root:
        report["aklr_commit"] = subprocess.check_output(
            ["git", "-C", str(args.aklr_root), "rev-parse", "HEAD"], text=True
        ).strip()
    with tempfile.TemporaryDirectory(prefix="hgemm-stagger-") as directory:
        for index in args.case:
            m, n, k, bm, bn, bk, stages, wm, wn = CASES[index]
            config = dict(
                block_m=bm,
                block_n=bn,
                block_k=bk,
                stages=stages,
                m_waves=wm,
                n_waves=wn,
                k_waves=1,
                split_k=1,
                group_m=0,
                policy="ft",
            )
            torch.manual_seed(17)
            count = max(1, math.ceil(args.rotate_mib * 1024**2 / (n * k * 2)))
            a = torch.empty(m, k, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
            weights = [
                torch.empty(n, k, device="cuda", dtype=a.dtype).uniform_(-1, 1)
                for _ in range(count)
            ]
            labels = [0] + args.stagger
            if args.b_non_temporal:
                labels += [f"bnt{s}" for s in [0] + args.stagger]
            if args.tuned:
                labels += ["aiter_tuned"]
            if baseline:
                labels += ["upstream"]
            aklr, source = (
                aklr_kernel(args.aklr_root, m, n, k, directory)
                if args.aklr_root
                else (None, None)
            )
            if aklr:
                labels += ["aklr"]
            outputs = {
                label: torch.empty(m, n, device="cuda", dtype=a.dtype)
                for label in labels
            }

            def one(label, b):
                if label == "aiter_tuned":
                    from aiter.tuned_gemm import gemm_a16w16

                    return gemm_a16w16(a, b, otype=a.dtype)
                if label == "aklr":
                    aklr(
                        a=a,
                        b=b,
                        trans_b=True,
                        bias=None,
                        out_dtype=a.dtype,
                        out=outputs[label],
                    )
                elif label == "upstream":
                    cfg = {
                        key: value for key, value in config.items() if key != "policy"
                    }
                    baseline(
                        a,
                        b.T,
                        out=outputs[label],
                        user_kwargs=dict(cfg, use_half_tile_interleaved=False),
                    )
                else:
                    flydsl_hgemm(
                        a,
                        b,
                        out=outputs[label],
                        k_stagger=int(label[3:]) if isinstance(label, str) else label,
                        b_non_temporal=isinstance(label, str),
                        **config,
                    )
                return outputs[label]

            # Check every weight in the rotation, with the same strict tolerance.
            for b in weights:
                ref = (a.float() @ b.float().T).to(a.dtype)
                for label in labels:
                    torch.testing.assert_close(one(label, b), ref, rtol=1e-2, atol=1e-2)

            def ring(label):
                for b in weights:
                    one(label, b)

            times = {label: [] for label in labels}
            for repeat in range(args.repeat):
                order = list(labels)
                random.Random(repeat).shuffle(order)
                for label in order:
                    us = (
                        do_bench_cudagraph(lambda: ring(label), rep=args.rep_ms)
                        * 1000
                        / count
                    )
                    times[label].append(us)
            medians = {
                str(label): statistics.median(values) for label, values in times.items()
            }
            row = dict(
                shape=[m, n, k],
                config=config,
                weight_buffers=count,
                aklr_source=source,
                us=medians,
                samples_us=times,
            )
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
            if args.json:
                args.json.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
