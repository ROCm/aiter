# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paired true-INT8 GEMM baselines for gfx950 CK and Gluon backends."""

import argparse
import csv
import gc
import statistics
from pathlib import Path

import torch

from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.gemm_kernels import flydsl_preshuffle_gemm_a8
from aiter.ops.gemm_op_a8w8 import (
    gemm_a8w8_ck,
    get_GEMM_config_with_quant_type,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_a8w8 import (
    gemm_a8w8 as gluon_gemm_a8w8,
)
from aiter.ops.triton.gemm.basic.gemm_a8w8 import gemm_a8w8_preshuffle

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SHAPES = ROOT / "aiter" / "configs" / "a8w8_bpreshuffle_untuned_gemm.csv"
WAN_SHAPES = (
    (9450, 5120, 5120),
    (9450, 13824, 5120),
    (9450, 5120, 13824),
)


def parse_shape(value):
    parts = tuple(int(part) for part in value.split(","))
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("shape must be M,N,K")
    return parts


def load_shapes(path):
    with path.open(newline="") as stream:
        return [
            (int(row["M"]), int(row["N"]), int(row["K"]))
            for row in csv.DictReader(stream)
            if row["q_dtype_w"].strip() == "torch.int8"
        ]


def event_time_us(launch, iterations):
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        launch()
    end.record()
    end.synchronize()
    return float(begin.elapsed_time(end) * 1000.0 / iterations)


def make_inputs(shape, seed, with_bias):
    m, n, k = shape
    torch.manual_seed(seed)
    x_float = torch.randn((m, k), dtype=torch.float32, device="cuda")
    w_float = torch.randn((n, k), dtype=torch.float32, device="cuda")
    x_scale = x_float.abs().amax(dim=1, keepdim=True).clamp_min_(1e-12) / 127.0
    w_scale = (
        w_float.abs().amax(dim=1, keepdim=True).T.contiguous().clamp_min_(1e-12) / 127.0
    )
    x = (x_float / x_scale).to(torch.int8)
    weight = (w_float / w_scale.T).to(torch.int8)
    weight_shuffled_native = shuffle_weight(weight, layout=(16, 16))
    weight_shuffled = weight_shuffled_native.reshape(n // 16, k * 16)
    bias = (
        torch.randn((1, n), dtype=torch.float32, device="cuda") if with_bias else None
    )
    return (
        x,
        weight,
        weight_shuffled,
        weight_shuffled_native,
        x_scale,
        w_scale,
        bias,
    )


@torch.no_grad()
def benchmark_shape(
    shape,
    seed,
    warmup,
    iterations,
    repeats,
    with_bias,
    check_correctness,
    include_flydsl,
    flydsl_config,
):
    m, n, k = shape
    (
        x,
        weight,
        weight_shuffled,
        weight_shuffled_native,
        x_scale,
        w_scale,
        bias,
    ) = make_inputs(shape, seed, with_bias)
    backend_names = ["ck", "gluon", "gluon_preshuffle"]
    if include_flydsl:
        if with_bias:
            raise RuntimeError("FlyDSL go/no-go candidate does not include bias")
        backend_names.append("flydsl_primary")
    outputs = {
        name: torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
        for name in backend_names
    }

    ck_config = get_GEMM_config_with_quant_type(
        m,
        n,
        k,
        torch.int8,
        AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_FILE,
    )
    ck_split_k = 0 if ck_config is None else int(ck_config["splitK"])

    def run_ck():
        return gemm_a8w8_ck(
            x,
            weight,
            x_scale,
            w_scale,
            outputs["ck"],
            bias,
            ck_split_k,
        )

    def run_gluon():
        return gluon_gemm_a8w8(
            x,
            weight,
            x_scale,
            w_scale,
            bias,
            torch.bfloat16,
            outputs["gluon"],
            backend="gluon",
        )

    def run_gluon_preshuffle():
        return gemm_a8w8_preshuffle(
            x,
            weight_shuffled,
            x_scale,
            w_scale,
            bias,
            torch.bfloat16,
            outputs["gluon_preshuffle"],
        )

    def run_flydsl_primary():
        return flydsl_preshuffle_gemm_a8(
            x,
            weight_shuffled_native,
            x_scale,
            w_scale,
            outputs["flydsl_primary"],
            tile_m=flydsl_config["tile_m"],
            tile_n=flydsl_config["tile_n"],
            tile_k=flydsl_config["tile_k"],
            use_async_copy=flydsl_config["async_copy"],
            waves_per_eu=flydsl_config["waves_per_eu"],
            xcd_swizzle=flydsl_config["xcd_swizzle"],
            lds_stage=flydsl_config["lds_stage"],
            enable_scheduler=True,
            split_k=1,
        )

    launches = {
        "ck": run_ck,
        "gluon": run_gluon,
        "gluon_preshuffle": run_gluon_preshuffle,
    }
    if include_flydsl:
        launches["flydsl_primary"] = run_flydsl_primary
    for launch in launches.values():
        launch()
    torch.cuda.synchronize()
    if check_correctness:
        torch.testing.assert_close(
            outputs["ck"], outputs["gluon"], atol=0.02, rtol=1e-2
        )
        torch.testing.assert_close(
            outputs["ck"], outputs["gluon_preshuffle"], atol=0.02, rtol=1e-2
        )
        if include_flydsl:
            torch.testing.assert_close(
                outputs["ck"], outputs["flydsl_primary"], atol=0.02, rtol=1e-2
            )

    for _ in range(warmup):
        for launch in launches.values():
            launch()
    torch.cuda.synchronize()

    samples = {name: [] for name in launches}
    names = tuple(launches)
    for repeat in range(repeats):
        offset = repeat % len(names)
        order = names[offset:] + names[:offset]
        if repeat % 2:
            order = tuple(reversed(order))
        for name in order:
            samples[name].append(event_time_us(launches[name], iterations))

    rows = []
    for name in names:
        values = samples[name]
        rows.append(
            {
                "M": m,
                "N": n,
                "K": k,
                "backend": name,
                "median_us": statistics.median(values),
                "min_us": min(values),
                "max_us": max(values),
                "samples_us": ";".join(f"{value:.6f}" for value in values),
                "ck_split_k": ck_split_k if name == "ck" else "",
                "bias": with_bias,
            }
        )
    print(
        f"{shape}: "
        + " ".join(f"{row['backend']}={row['median_us']:.3f}us" for row in rows),
        flush=True,
    )
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", type=parse_shape, nargs="*")
    parser.add_argument("--shapes-file", type=Path, default=DEFAULT_SHAPES)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument(
        "--no-wan",
        action="store_true",
        help="Exclude the three Wan projection shapes from the default matrix.",
    )
    parser.add_argument("--include-flydsl", action="store_true")
    parser.add_argument("--flydsl-tile", type=parse_shape, default=(128, 128, 128))
    parser.add_argument("--flydsl-async-copy", type=int, choices=(0, 1), default=1)
    parser.add_argument("--flydsl-waves-per-eu", type=int, default=0)
    parser.add_argument("--flydsl-xcd-swizzle", type=int, default=8)
    parser.add_argument("--flydsl-lds-stage", type=int, default=2)
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--confirmed-idle",
        action="store_true",
        help="Confirm two consecutive GPU checks showed the selected device idle.",
    )
    args = parser.parse_args()
    if not args.confirmed_idle:
        parser.error("refusing to benchmark without --confirmed-idle")
    if get_gfx_runtime() != "gfx950":
        parser.error(f"gfx950 required, got {get_gfx_runtime()}")
    if get_cu_num() != 256:
        parser.error(f"initial baseline requires 256 CUs, got {get_cu_num()}")

    if args.shape:
        shapes = args.shape
    else:
        shapes = load_shapes(args.shapes_file)
        if not args.no_wan:
            shapes.extend(shape for shape in WAN_SHAPES if shape not in shapes)
    flydsl_config = {
        "tile_m": args.flydsl_tile[0],
        "tile_n": args.flydsl_tile[1],
        "tile_k": args.flydsl_tile[2],
        "async_copy": args.flydsl_async_copy,
        "waves_per_eu": args.flydsl_waves_per_eu,
        "xcd_swizzle": args.flydsl_xcd_swizzle,
        "lds_stage": args.flydsl_lds_stage,
    }
    rows = []
    for index, shape in enumerate(shapes):
        rows.extend(
            benchmark_shape(
                shape,
                args.seed + index,
                args.warmup,
                args.iterations,
                args.repeats,
                args.bias,
                not args.skip_correctness,
                args.include_flydsl,
                flydsl_config,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
