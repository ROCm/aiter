#!/usr/bin/env python3
"""Standalone single-kernel runner for kernel-only rocprofv3 tracing.

Runs exactly ONE top_k_per_row_decode implementation for ONE shape so that a
wrapping ``rocprofv3 --kernel-trace`` invocation captures only that kernel's
dispatches. Host timing is intentionally NOT performed here -- timing comes
from the rocprof kernel trace (Start/End timestamps).

Kernels:
  * flydsl                -> FlyDSL standalone unordered radix-select path (K=2048)
  * flydsl_tiered         -> K=512 persistent path with in-kernel tier dispatch
  * aiter_hip             -> aiter.top_k_per_row_decode (HIP radix one/multi-block)
  * vllm                  -> torch.ops._C.top_k_per_row_decode (vLLM sampler.cu kernel)

The vLLM op is registered by loading the prebuilt stable-libtorch .so directly,
which avoids importing the full vllm package (and its heavy python deps).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def add_path_if_present(path: str | Path | None) -> None:
    if not path:
        return
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        return
    path_str = str(resolved)
    if path_str in sys.path:
        sys.path.remove(path_str)
    sys.path.insert(0, path_str)


def add_flydsl_package_path() -> None:
    configured = os.environ.get("FLYDSL_PATH")
    # Prefer the packaged runtime when present; the source tree alone does not
    # include the generated MLIR extension modules used by local benchmark runs.
    add_path_if_present(REPO_ROOT.parent / "FlyDSL" / "python")
    if configured:
        add_path_if_present(Path(configured) / "python")
        add_path_if_present(configured)
    add_path_if_present(REPO_ROOT.parent / ".r1_flydsl_pkgs")

# Reuse the harness data generators for parity with the benchmark.
from op_tests.benchmark_topk_per_row_decode import make_logits, make_row_ends  # noqa: E402

VLLM_SO = "/home/AMD/samremes/dev/vllm/vllm/_C_stable_libtorch.abi3.so"


def load_vllm():
    torch.ops.load_library(VLLM_SO)
    return torch.ops._C.top_k_per_row_decode


def load_flydsl():
    add_flydsl_package_path()
    from aiter.ops.flydsl.topk_per_row_decode import flydsl_top_k_per_row_decode

    return flydsl_top_k_per_row_decode


def load_aiter():
    import aiter

    return aiter.top_k_per_row_decode


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--kernel",
        required=True,
        choices=[
            "flydsl",
            "flydsl_tiered",
            "aiter_hip",
            "vllm",
        ],
    )
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument(
        "--max-width",
        type=int,
        default=None,
        help="Physical logits width; defaults to L. seq_lens still uses L.",
    )
    ap.add_argument("--num-rows", type=int, default=1)
    ap.add_argument("--next-n", type=int, default=1)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--distribution", default="random")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = torch.device("cuda")
    torch.set_default_device("cuda:0")
    max_width = args.L if args.max_width is None else args.max_width
    if max_width < args.L:
        raise ValueError(f"--max-width={max_width} must be >= --L={args.L}")
    if args.kernel == "flydsl" and args.k == 512:
        raise ValueError("Use --kernel flydsl_tiered for K=512; standalone K=512 was pruned")
    if args.kernel == "flydsl_tiered" and args.k != 512:
        raise ValueError("flydsl_tiered is the K=512 tiered path")

    batch_size = args.num_rows // args.next_n
    seq_lens, row_ends = make_row_ends(
        batch_size, args.num_rows, args.L, args.next_n, device
    )
    logits = make_logits(
        args.num_rows,
        max_width,
        row_ends,
        torch.float32,
        args.distribution,
        "unit",  # stride1 == 1
        args.seed,
        device,
    )
    indices = torch.empty((args.num_rows, args.k), dtype=torch.int32, device=device)
    stride0 = logits.stride(0)
    stride1 = logits.stride(1)

    if args.kernel == "vllm":
        op = load_vllm()

        def run():
            op(logits, args.next_n, seq_lens, indices, args.num_rows, stride0, stride1, args.k)

    elif args.kernel == "aiter_hip":
        op = load_aiter()

        def run():
            op(logits, args.next_n, seq_lens, indices, args.num_rows, stride0, stride1, k=args.k)

    else:  # FlyDSL variants
        op = load_flydsl()

        def run():
            op(
                logits,
                args.next_n,
                seq_lens,
                indices,
                args.num_rows,
                stride0,
                stride1,
                k=args.k,
                ordered=False,
            )

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    for _ in range(args.iters):
        run()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
