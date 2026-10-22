# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paired gfx950 signed-INT8 GEMM benchmark for the FlyDSL 4-wave kernel."""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
from pathlib import Path

import torch

from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import (
    get_cdna4_num_xcds,
    get_cu_num,
    get_gfx_runtime,
)
from aiter.ops.flydsl.gemm_a8w8_bpreshuffle_4wave import (
    make_4wave_kernel_name,
    run_gemm_a8w8_bpreshuffle_4wave,
)
from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8
from aiter.ops.gemm_op_a8w8 import (
    gemm_a8w8_asm,
    gemm_a8w8_ck,
    get_GEMM_config_with_quant_type,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_a8w8 import (
    gemm_a8w8 as gluon_gemm_a8w8,
)
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8_preshuffle

DEFAULT_SHAPES = (
    (9450, 5120, 5120),
    (9450, 13824, 5120),
    (9450, 5120, 13824),
)
REPRESENTATIVE_M_VALUES = (512, 9450, 9472, 18900, 37800, 75600)
DEFAULT_BACKENDS = ("4wave", "4wave_nonpersistent", "gluon")
ASM_KERNEL_TILES = (16, 32, 48, 64, 80, 96, 128, 160)
ASM_KERNEL_TEMPLATE = "_ZN5aiter{length}I8gemm_bf16_perTokenI8_BpreShuffle_{tile}x128E"


def _shape(value: str) -> tuple[int, int, int]:
    parts = tuple(int(part) for part in value.lower().replace("x", ",").split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be M,N,K")
    return parts


def _event_time_us(launch, iterations: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        launch()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end) * 1000.0 / iterations)


def _make_inputs(shape: tuple[int, int, int], seed: int):
    m, n, k = shape
    generator = torch.Generator(device="cuda").manual_seed(seed)
    # This bounded range keeps the integer dot product exactly representable in
    # FP32 even at K=13824, so backend output comparisons can be bit-exact.
    x = torch.randint(
        -31, 32, (m, k), dtype=torch.int8, device="cuda", generator=generator
    )
    weight = torch.randint(
        -31, 32, (n, k), dtype=torch.int8, device="cuda", generator=generator
    )
    x_scale = (
        torch.rand((m, 1), dtype=torch.float32, device="cuda", generator=generator)
        * 1e-3
        + 1e-4
    )
    w_scale = (
        torch.rand((1, n), dtype=torch.float32, device="cuda", generator=generator)
        * 1e-3
        + 1e-4
    )
    weight_shuffled_native = shuffle_weight(weight, layout=(16, 16)).contiguous()
    weight_shuffled_gluon = weight_shuffled_native.reshape(n // 16, k * 16)
    return (
        x,
        weight,
        weight_shuffled_native,
        weight_shuffled_gluon,
        x_scale,
        w_scale,
    )


@torch.no_grad()
def _benchmark_shape(
    shape: tuple[int, int, int],
    *,
    backends: tuple[str, ...],
    warmup: int,
    iterations: int,
    repeats: int,
    seed: int,
    use_cuda_graph: bool,
) -> list[dict[str, object]]:
    m, n, k = shape
    (
        x,
        weight,
        weight_shuffled_native,
        weight_shuffled_gluon,
        x_scale,
        w_scale,
    ) = _make_inputs(shape, seed)
    cu_num = get_cu_num()
    num_xcds = get_cdna4_num_xcds(cu_num)
    grid = (m + 255) // 256 * ((n + 255) // 256)
    persistent_cu = cu_num if k == 5120 and grid > cu_num else 0
    if "asm_best" in backends and "4wave" not in backends:
        raise ValueError("asm_best selection requires the 4wave correctness reference")

    outputs = {
        name: torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        for name in backends
    }
    launches = {}
    kernel_names: dict[str, str] = {}
    asm_kernel = {"name": "", "weight": None}
    asm_bias = torch.zeros((1, n), dtype=torch.float32, device="cuda")

    def add_4wave(name: str, num_cu: int) -> None:
        kernel_name = make_4wave_kernel_name(
            K=k,
            xcd_swizzle=8,
            num_cu=num_cu,
            num_xcds=num_xcds,
        )
        kernel_names[name] = kernel_name
        launches[name] = lambda: run_gemm_a8w8_bpreshuffle_4wave(
            x,
            weight_shuffled_native,
            x_scale,
            w_scale,
            outputs[name],
            kernel_name,
        )

    if "4wave" in backends:
        add_4wave("4wave", persistent_cu)
    if "4wave_nonpersistent" in backends:
        add_4wave("4wave_nonpersistent", 0)
    if "4wave_persistent" in backends:
        add_4wave("4wave_persistent", cu_num)

    if "gluon" in backends:
        launches["gluon"] = lambda: gluon_gemm_a8w8(
            x,
            weight,
            x_scale,
            w_scale,
            None,
            torch.bfloat16,
            outputs["gluon"],
            backend="gluon",
        )
    if "gluon_preshuffle" in backends:
        launches["gluon_preshuffle"] = lambda: gemm_a8w8_preshuffle(
            x,
            weight_shuffled_gluon,
            x_scale,
            w_scale,
            None,
            torch.bfloat16,
            outputs["gluon_preshuffle"],
        )
    if "ck" in backends:
        config = get_GEMM_config_with_quant_type(
            m,
            n,
            k,
            torch.int8,
            AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE,
        )
        split_k = 0 if config is None else int(config["splitK"])
        launches["ck"] = lambda: gemm_a8w8_ck(
            x,
            weight,
            x_scale,
            w_scale,
            outputs["ck"],
            None,
            split_k,
        )
    if "flydsl_generic" in backends:
        launches["flydsl_generic"] = lambda: flydsl_preshuffle_gemm_a8(
            x,
            weight_shuffled_native,
            x_scale,
            w_scale,
            outputs["flydsl_generic"],
            tile_m=128,
            tile_n=256,
            tile_k=256,
            use_async_copy=True,
            waves_per_eu=1,
            xcd_swizzle=8,
            lds_stage=2,
            enable_scheduler=True,
            split_k=1,
        )
    if "asm_best" in backends:
        launches["4wave"]()
        torch.cuda.synchronize()

        def run_asm():
            return gemm_a8w8_asm(
                x,
                asm_kernel["weight"],
                x_scale,
                w_scale,
                outputs["asm_best"],
                asm_kernel["name"],
                asm_bias,
                True,
                1,
            )

        best = None
        for layout in ((32, 16), (16, 16)):
            asm_kernel["weight"] = shuffle_weight(weight, layout=layout)
            for tile in ASM_KERNEL_TILES:
                name = ASM_KERNEL_TEMPLATE.format(
                    length=len(f"I8gemm_bf16_perTokenI8_BpreShuffle_{tile}x128"),
                    tile=tile,
                )
                asm_kernel["name"] = name
                try:
                    run_asm()
                    torch.cuda.synchronize()
                    torch.testing.assert_close(
                        outputs["asm_best"],
                        outputs["4wave"],
                        atol=0.02,
                        rtol=1e-2,
                    )
                    us = _event_time_us(run_asm, 20)
                except Exception:  # noqa: BLE001, S112 - unsupported candidate
                    continue
                if best is None or us < best[0]:
                    best = (us, name, layout)
            if best is not None:
                break
        if best is None:
            raise RuntimeError(f"no correct ASM INT8 kernel found for {shape}")
        asm_kernel["name"] = best[1]
        asm_kernel["weight"] = shuffle_weight(weight, layout=best[2])
        kernel_names["asm_best"] = best[1]
        launches["asm_best"] = run_asm

    missing = set(backends) - set(launches)
    if missing:
        raise ValueError(f"unknown or unavailable backends: {sorted(missing)}")

    for launch in launches.values():
        launch()
    torch.cuda.synchronize()
    reference_name = "4wave" if "4wave" in outputs else next(iter(outputs))
    reference = outputs[reference_name]
    for name, output in outputs.items():
        if name in {"ck", "asm_best"}:
            torch.testing.assert_close(reference, output, atol=0.02, rtol=1e-2)
        elif not torch.equal(reference, output):
            mismatch = int(torch.count_nonzero(reference != output).item())
            raise RuntimeError(
                f"{shape}: {name} differs from {reference_name}: "
                f"{mismatch} BF16 elements"
            )

    for _ in range(warmup):
        for launch in launches.values():
            launch()
    torch.cuda.synchronize()

    if use_cuda_graph:
        replay = {}
        for name, launch in launches.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                launch()
            replay[name] = graph.replay
        launches = replay
        torch.cuda.synchronize()

    samples = {name: [] for name in backends}
    names = list(backends)
    for repeat in range(repeats):
        offset = repeat % len(names)
        order = names[offset:] + names[:offset]
        if repeat % 2:
            order.reverse()
        for name in order:
            samples[name].append(_event_time_us(launches[name], iterations))

    flops = 2.0 * m * n * k
    rows = []
    for name in names:
        values = samples[name]
        median_us = statistics.median(values)
        rows.append(
            {
                "M": m,
                "N": n,
                "K": k,
                "backend": name,
                "cu_num": cu_num,
                "num_xcds": num_xcds,
                "kernel_name": kernel_names.get(name, ""),
                "launch_mode": "cuda_graph" if use_cuda_graph else "eager",
                "median_us": median_us,
                "min_us": min(values),
                "max_us": max(values),
                "tflops": flops / median_us / 1e6,
                "samples_us": ";".join(f"{value:.6f}" for value in values),
                "correct": True,
            }
        )

    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=_shape, action="append")
    parser.add_argument(
        "--shape-matrix",
        action="store_true",
        help="benchmark N=K=5120 at representative M values",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=list(DEFAULT_BACKENDS),
        choices=(
            "4wave",
            "4wave_nonpersistent",
            "4wave_persistent",
            "gluon",
            "gluon_preshuffle",
            "ck",
            "asm_best",
            "flydsl_generic",
        ),
    )
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--expected-cu", type=int, required=True)
    parser.add_argument("--confirmed-idle", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not args.confirmed_idle:
        raise RuntimeError("verify the selected GPU is idle before benchmarking")
    if get_gfx_runtime() != "gfx950":
        raise RuntimeError(f"expected gfx950, got {get_gfx_runtime()}")
    if get_cu_num() != args.expected_cu:
        raise RuntimeError(
            f"expected {args.expected_cu} CUs, active device has {get_cu_num()}"
        )

    shapes = list(args.shape or DEFAULT_SHAPES)
    if args.shape_matrix:
        shapes.extend((m, 5120, 5120) for m in REPRESENTATIVE_M_VALUES)
    shapes = list(dict.fromkeys(shapes))

    rows = []
    for index, shape in enumerate(shapes):
        result = _benchmark_shape(
            shape,
            backends=tuple(args.backends),
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
            seed=args.seed + index,
            use_cuda_graph=not args.no_cuda_graph,
        )
        rows.extend(result)
        best = min(result, key=lambda row: row["median_us"])
        print(
            f"{shape}: {best['backend']} {best['median_us']:.3f} us "
            f"{best['tflops']:.2f} TFLOP/s",
            flush=True,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
