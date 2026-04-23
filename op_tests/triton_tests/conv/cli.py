# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
from __future__ import annotations
import argparse
import torch
from .suite import TestSuite
from ._registry import ALL_METHODS
from .test_edge import test_edge_cases, test_activations
from .test_fuzz import test_random_fuzzing
from .test_models import test_models


def pick_torch_dtype(d: str):
    if d == "fp16":
        return torch.float16
    if d == "bf16":
        return torch.bfloat16
    raise ValueError(d)


def main():
    p = argparse.ArgumentParser(
        description="Conv2D Triton correctness tests & benchmarks"
    )
    p.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--test-mode",
        type=str,
        default="edge",
        choices=["edge", "random", "stability", "activations", "models", "all"],
    )
    p.add_argument(
        "--mode",
        type=str,
        default=None,
        dest="mode_alias",
        choices=["edge", "models"],
        help="alias for --test-mode (backward compat)",
    )
    p.add_argument("--num-random", type=int, default=200)
    p.add_argument("--model-name", type=str, default=None, dest="model_name")
    p.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="max conv layers per model (default: 5 for tests, 53 for bench)",
    )
    p.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="path to local model directory (required for sd_unet/sd35_vae/flux2_vae)",
    )
    p.add_argument(
        "--no-print-shapes", action="store_true", help="disable per-case shape prints"
    )
    p.add_argument(
        "--layout", type=str, default="nchw", choices=["nchw", "nhwc", "both"]
    )
    p.add_argument(
        "--method",
        choices=ALL_METHODS,
        default="default",
        help="kernel method to test/benchmark",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="batch size N for model benchmarks (default: 1 for all models)",
    )
    p.add_argument(
        "--height",
        type=int,
        default=None,
        help="override input spatial height H for all layers (default: real model shapes)",
    )
    p.add_argument(
        "--width",
        type=int,
        default=None,
        help="override input spatial width W for all layers (default: real model shapes)",
    )
    p.add_argument("--benchmark", action="store_true", help="enable benchmarking")
    p.add_argument(
        "--pretrained",
        action="store_true",
        help="use real pretrained weights for all models (default: random init). "
        "TFLOPS doesn't depend on weight values; correctness compares Triton "
        "vs PyTorch with the same tensors regardless.",
    )
    args = p.parse_args()

    test_mode = args.mode_alias if args.mode_alias else args.test_mode

    if args.model_path and not args.model_name:
        p.error("--model-path requires --model-name to specify which model to load")

    num_layers = args.num_layers
    if num_layers is None:
        num_layers = 53 if args.benchmark else 5

    torch.set_grad_enabled(False)
    device = args.device
    dtype = pick_torch_dtype(args.dtype)
    method = args.method
    backend = "CUDA" if torch.version.cuda is not None else "HIP"
    print(f"Backend: {backend} | device: {device} | dtype: {dtype} | method: {method}")

    suite = TestSuite(
        device=device,
        dtype=dtype,
        bench_enabled=args.benchmark,
        print_shapes=not args.no_print_shapes,
        layout_mode=args.layout,
    )

    if test_mode in ("edge", "all"):
        test_edge_cases(suite, method=method)
    if test_mode in ("random", "all"):
        test_random_fuzzing(suite, num_tests=args.num_random, method=method)
    if test_mode in ("stability", "all"):
        test_random_fuzzing(suite, num_tests=20, method=method)
    if test_mode in ("activations", "all"):
        for act in ["none", "relu", "relu6", "gelu"]:
            test_activations(suite, method=method, activation=act)
    if test_mode in ("models", "all"):
        test_models(
            suite,
            models=args.model_name,
            num_layers=num_layers,
            method=method,
            model_path=args.model_path,
            batch_size=args.batch_size,
            spatial_h=args.height,
            spatial_w=args.width,
            pretrained=args.pretrained,
        )

    ok = suite.summary()
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
