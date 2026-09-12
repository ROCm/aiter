# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compare split-K epilogues with fixed workspaces and long CUDA-graph runs.

Example:
    ENABLE_CK=0 python3 op_tests/bench_gemm_a8w8_splitk_gfx1250.py \
        --kernel flydsl_mxfp8_128_bpreshuffle_compute_wmma_t256x256x128_mw2_nw2_nb2_sk4_cm2_cn4 \
        --include-eightwave --include-cluster --graph-iters 1000 \
        --replays 10 --repeats 10

The benchmark fails before importing the GPU runtime if a driver query is stuck
or another KFD process exists. It never clears processes or changes GPU clocks.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import threading
import time


def _gpu_pids():
    return {int(path.name) for path in Path("/sys/class/kfd/kfd/proc").iterdir()}


def _check_idle():
    blocked = []
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            status = (path / "status").read_text()
            if "State:\tD" not in status:
                continue
            wait = (path / "wchan").read_text().strip()
            command = (path / "cmdline").read_bytes().replace(b"\0", b" ")
            if "amdgpu" in wait or any(
                name in command for name in (b"rocminfo", b"amd-smi", b"rocm-smi")
            ):
                blocked.append((path.name, wait))
        except (FileNotFoundError, ProcessLookupError):
            continue
    if blocked:
        raise RuntimeError(f"Driver recovery required; blocked GPU tasks: {blocked}")
    for _ in range(2):
        pids = _gpu_pids()
        if pids:
            raise RuntimeError(f"GPU is occupied by KFD PIDs: {sorted(pids)}")
        time.sleep(0.1)


class _GpuMonitor:
    """Detect another tenant throughout compilation, capture, and timing."""

    def __init__(self):
        self.owned = _gpu_pids()
        if len(self.owned) != 1:
            raise RuntimeError(f"Expected one benchmark process, got {self.owned}")
        self.changed = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._poll, daemon=True)
        self.thread.start()

    def _poll(self):
        while not self.stop.wait(0.05):
            try:
                current = _gpu_pids()
            except OSError as error:
                self.changed = str(error)
                return
            if current != self.owned:
                self.changed = sorted(current)
                return

    def check(self):
        current = _gpu_pids()
        if self.changed is not None or current != self.owned:
            raise RuntimeError(
                f"GPU occupancy changed; discard this run: {self.changed or current}"
            )

    def close(self):
        self.stop.set()
        self.thread.join()


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("-m", type=_positive_int, default=512)
    parser.add_argument("-n", type=_positive_int, default=7168)
    parser.add_argument("-k", type=_positive_int, default=16384)
    parser.add_argument("--graph-iters", type=_positive_int, default=1000)
    parser.add_argument("--replays", type=_positive_int, default=10)
    parser.add_argument("--repeats", type=_positive_int, default=10)
    parser.add_argument("--warmup-replays", type=_positive_int, default=10)
    parser.add_argument("--include-eightwave", action="store_true")
    parser.add_argument("--include-cluster", action="store_true")
    parser.add_argument("--apre", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()
    _check_idle()
    os.environ.setdefault("ENABLE_CK", "0")

    import flydsl.compiler as flyc
    import flydsl.expr as fx
    import torch

    from aiter import dtypes
    from aiter.ops.flydsl import mxfp8_128_bpreshuffle_gemm_gfx1250 as backend
    from aiter.ops.flydsl.kernels.gemm_a8w8_256x256_gfx1250 import (
        launch_gemm_a8w8_256x256,
    )
    from aiter.ops.flydsl.kernels.gemm_a8w8_splitk_reduce_gfx1250 import (
        compile_gemm_a8w8_splitk_reduce,
    )
    from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
    from aiter.ops.shuffle import shuffle_mxfp8fp4_a, shuffle_weight

    torch.cuda.init()
    if not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1250"):
        raise RuntimeError("This benchmark requires gfx1250")
    registration = torch.empty(1, device="cuda")
    monitor = _GpuMonitor()
    try:
        cfg = backend.parse_wmma_kernel_name(args.kernel)
        if cfg is None or not backend.is_compute_wmma_kernel_name(args.kernel):
            raise ValueError("--kernel must name a compute WMMA profile")
        if cfg["split_k"] == 1 or cfg["persistent_n_tiles"] != 1:
            raise ValueError("The comparison requires split-K > 1 and one N tile")
        m, n, k = args.m, args.n, args.k
        apre = args.apre or cfg["a_preshuffle"]
        cfg["a_preshuffle"] = apre
        tm, tn, sk = cfg["tile_m"], cfg["tile_n"], cfg["split_k"]
        if n % (tn * cfg["cluster_n"]) or not backend.cluster_m_grid_ok(
            m, tm, cfg["cluster_m"]
        ):
            raise ValueError("Shape must fill the M/N cluster without padding")
        if k % (
            sk * cfg["tile_k"] * backend.compute_kernel_k_pair(cfg["num_buffers"], tn)
        ):
            raise ValueError("K must contain whole pipeline stages in each split")
        if k // sk < 512 or (m % 2 and apre):
            raise ValueError("Each split needs K >= 512; A-preshuffle needs even M")
        tile_count = ((m + tm - 1) // tm) * (n // tn)
        if tile_count * 32 > backend.SPLIT_K_FLAG_MAX_LEN:
            raise ValueError("Output tiles exceed the last-arrival counter capacity")

        cases = [("separate", dict(cfg)), ("last_arrival", dict(cfg))]
        if args.include_eightwave:
            if (tm, tn, cfg["num_buffers"]) not in ((256, 256, 2), (256, 256, 4)):
                raise ValueError("Eight-wave candidates require 256x256, nb2 or nb4")
            cases.append(("eightwave", dict(cfg, m_warp=4, n_warp=2)))
        if args.include_cluster:
            cm = cfg["cluster_m"]
            cn = min(cfg["cluster_n"], 16 // (cm * sk))
            while cn > 0 and n % (tn * cn):
                cn -= 1
            if cn < 1 or cm * cn < 2:
                raise ValueError("K splits cannot fit in a supported 16-block cluster")
            cases.append(("cluster", dict(cfg, cluster_n=cn)))

        free_bytes, _ = torch.cuda.mem_get_info()
        scratch_bytes = sk * tile_count * tm * tn * 2
        estimate = 6 * (m * k + n * k) + len(cases) * (scratch_bytes + m * n * 2)
        if estimate > free_bytes * 0.75:
            raise RuntimeError(f"Insufficient free VRAM: need about {estimate} bytes")
        torch.manual_seed(42)
        a = (torch.randn((m, k), device="cuda") * 0.1).to(dtypes.fp8)
        b = (torch.randn((n, k), device="cuda") * 0.1).to(dtypes.fp8)
        b = shuffle_weight(b, layout=(16, 16))
        if apre:
            a = shuffle_mxfp8fp4_a(a)
        sa = torch.randint(124, 128, (k // 128, m), device="cuda", dtype=torch.uint8)
        sa = sa.view(dtypes.fp8_e8m0).view(m, k // 128)
        sb = torch.randint(
            124, 128, (n // 128, k // 128), device="cuda", dtype=torch.uint8
        )
        sb = sb.view(dtypes.fp8_e8m0)
        dtype = torch.float16 if args.fp16 else torch.bfloat16
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        outputs, flags, graphs, keepalive = {}, {}, {}, [registration, a, b, sa, sb]
        with torch.cuda.stream(stream):
            for mode, case in cases:
                monitor.check()
                fused = mode != "separate"
                partial_shape = (tile_count, sk, tm, tn) if fused else (sk, m, n)
                partial = torch.empty(partial_shape, dtype=dtype, device="cuda")
                out = torch.empty((m, n), dtype=dtype, device="cuda")
                flag = torch.zeros(
                    backend.SPLIT_K_FLAG_MAX_LEN, dtype=torch.int32, device="cuda"
                )
                keepalive.extend((partial, out, flag))
                launch_args = (
                    ptr_arg(partial),
                    ptr_arg(a),
                    ptr_arg(b),
                    ptr_arg(sa),
                    ptr_arg(sb),
                    m,
                    fx.Stream(stream),
                    n,
                    k,
                    m,
                    k,
                    n,
                    ptr_arg(flag),
                    ptr_arg(out),
                    tm,
                    tn,
                    case["tile_k"],
                    case["m_warp"],
                    case["n_warp"],
                    int(args.fp16),
                    case["num_buffers"],
                    case["cluster_m"],
                    case["cluster_n"],
                    True,
                    128,
                    sk,
                    apre,
                    1,
                    fused,
                    bool(m % tm),
                    mode == "cluster",
                )
                gemm = flyc.compile(launch_gemm_a8w8_256x256, *launch_args)
                reduce_args, reduce_fn = (), None
                if not fused:
                    reduce_args = (
                        ptr_arg(partial),
                        ptr_arg(out),
                        m * n,
                        1,
                        n,
                        m * n * 2,
                        fx.Stream(stream),
                    )
                    reduce_fn = flyc.compile(
                        compile_gemm_a8w8_splitk_reduce(
                            split_k=sk, out_dtype_str="f16" if args.fp16 else "bf16"
                        ),
                        *reduce_args,
                    )

                def run():
                    gemm(*launch_args)
                    if reduce_fn is not None:
                        reduce_fn(*reduce_args)

                for _ in range(3):
                    run()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                # Workspaces stay alive and are reused by every node and replay.
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(args.graph_iters):
                        run()
                keepalive.extend((gemm, reduce_fn))
                graphs[mode], outputs[mode], flags[mode] = graph, out, flag
            for mode, graph in graphs.items():
                for _ in range(args.warmup_replays):
                    monitor.check()
                    graph.replay()
                stream.synchronize()
                torch.testing.assert_close(
                    outputs[mode], outputs["separate"], rtol=0, atol=0
                )
            samples = {mode: [] for mode in graphs}
            modes = list(graphs)
            for repeat in range(args.repeats):
                # Alternating order reduces bias from temperature and DPM drift.
                for mode in modes if repeat % 2 == 0 else reversed(modes):
                    monitor.check()
                    start, end = (
                        torch.cuda.Event(enable_timing=True) for _ in range(2)
                    )
                    start.record()
                    for _ in range(args.replays):
                        monitor.check()
                        graphs[mode].replay()
                    end.record()
                    end.synchronize()
                    monitor.check()
                    samples[mode].append(
                        start.elapsed_time(end)
                        * 1000
                        / (args.graph_iters * args.replays)
                    )
            a.view(torch.uint8).bitwise_xor_(0x80)
            for graph in graphs.values():
                graph.replay()
            stream.synchronize()
            for mode in modes:
                torch.testing.assert_close(
                    outputs[mode], outputs["separate"], rtol=0, atol=0
                )
                assert flags[mode].count_nonzero().item() == 0
        monitor.check()
        print(
            json.dumps(
                {
                    "shape": [m, n, k],
                    "kernel": args.kernel,
                    "apre": apre,
                    "fp16": args.fp16,
                    "graph_iters": args.graph_iters,
                    "replays": args.replays,
                    "repeats": args.repeats,
                    "calls_per_mode": args.graph_iters * args.replays * args.repeats,
                    "cases": dict(cases),
                    "median_us": {
                        mode: statistics.median(v) for mode, v in samples.items()
                    },
                    "samples_us": samples,
                    "correctness": "bitwise, including changed-input replay",
                },
                indent=2,
            )
        )
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
