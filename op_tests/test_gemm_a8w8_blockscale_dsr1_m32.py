# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and microbenchmarks for DeepSeek-R1 M=32 FP8 GEMMs.

This compares the frozen pre-change CK winners with the exact-shape gfx950
dispatcher.  Run on an MI35X GPU with::

    python op_tests/test_gemm_a8w8_blockscale_dsr1_m32.py
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_ck
from aiter.ops.triton.gluon.gemm_a8w8_blockscale_dsr1_m32 import (
    DSR1_M32_CK_FALLBACK_CONFIGS,
    DSR1_M32_KERNEL_SHAPES,
    try_gemm_a8w8_blockscale_dsr1_m32,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = frozenset({"gfx950"})
BLOCK_N = 128
BLOCK_K = 128
DEFAULT_NUM_ITERS = 100
DEFAULT_NUM_WARMUP = 10


def _make_inputs(
    m: int, n: int, k: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create deterministic signed FP8 inputs and positive FP32 scales."""

    generator = torch.Generator(device="cuda")
    generator.manual_seed(seed)
    x = (
        (
            torch.rand((m, k), device="cuda", dtype=torch.float32, generator=generator)
            - 0.5
        )
        / 4
    ).to(torch.float8_e4m3fn)
    weight = (
        (
            torch.rand((n, k), device="cuda", dtype=torch.float32, generator=generator)
            - 0.5
        )
        / 4
    ).to(torch.float8_e4m3fn)
    x_scale = torch.rand(
        (m, (k + BLOCK_K - 1) // BLOCK_K),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    weight_scale = torch.rand(
        ((n + BLOCK_N - 1) // BLOCK_N, (k + BLOCK_K - 1) // BLOCK_K),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    return x, weight, x_scale, weight_scale


def _torch_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> torch.Tensor:
    """Independent FP32 reference for 128x128 block scaling."""

    m, k = x.shape
    n = weight.shape[0]
    x_scale_full = x_scale.repeat_interleave(BLOCK_K, dim=1)[:m, :k]
    weight_scale_full = weight_scale.repeat_interleave(BLOCK_N, dim=0)
    weight_scale_full = weight_scale_full.repeat_interleave(BLOCK_K, dim=1)[:n, :k]
    x_fp32 = x.float() * x_scale_full
    weight_fp32 = weight.float() * weight_scale_full
    return F.linear(x_fp32, weight_fp32)


def _effective_nbytes(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
) -> int:
    """Logical input reads plus output writes; excludes internal workspaces."""

    return sum(
        tensor.numel() * tensor.element_size()
        for tensor in (x, weight, x_scale, weight_scale, out)
    )


def _ck_baseline_config(selector: str) -> tuple[int, str]:
    try:
        return DSR1_M32_CK_FALLBACK_CONFIGS[selector]
    except KeyError as error:
        raise RuntimeError(f"No frozen CK baseline for {selector}") from error


def _record_candidate(
    ret: dict[str, object],
    name: str,
    output: torch.Tensor,
    reference: torch.Tensor,
    elapsed_us: float,
    flops: int,
    effective_nbytes: int,
) -> None:
    output_fp32 = output.float()
    ret[f"{name} us"] = elapsed_us
    ret[f"{name} TFLOPS"] = flops / elapsed_us / 1e6
    ret[f"{name} TB/s"] = effective_nbytes / elapsed_us / 1e6
    ret[f"{name} err ratio"] = checkAllclose(
        reference,
        output_fp32,
        rtol=1e-2,
        atol=1e-2,
        msg=f"{name}: ",
        catastrophic_check=True,
    )
    ret[f"{name} max abs"] = (reference - output_fp32).abs().max().item()


@benchmark()
def test_gemm_a8w8_blockscale_dsr1_m32(
    selector: str,
    m: int,
    n: int,
    k: int,
    seed: int,
    num_iters: int,
    num_warmup: int,
) -> dict[str, object]:
    """Compare the tuned CK baseline with the exact dispatch entry point."""

    x, weight, x_scale, weight_scale = _make_inputs(m, n, k, seed)
    reference = _torch_reference(x, weight, x_scale, weight_scale)
    flops = 2 * m * n * k
    split_k, kernel_name = _ck_baseline_config(selector)
    aiter.logger.info(
        "CK baseline M=%d N=%d K=%d splitK=%d kernelName=%s",
        m,
        n,
        k,
        split_k,
        kernel_name,
    )

    ck_out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    exact_out = torch.empty_like(ck_out)

    def run_ck() -> torch.Tensor:
        return gemm_a8w8_blockscale_ck(
            x,
            weight,
            x_scale,
            weight_scale,
            ck_out,
            splitK=split_k,
            kernelName=kernel_name,
        )

    def run_exact() -> torch.Tensor:
        result = try_gemm_a8w8_blockscale_dsr1_m32(
            x,
            weight,
            x_scale,
            weight_scale,
            exact_out,
            kernel_name=selector,
            gfx=get_gfx(),
        )
        if result is None:
            raise RuntimeError(
                f"Exact dispatcher rejected its registered contract for {selector}"
            )
        return result

    ret: dict[str, object] = {"gfx": get_gfx(), "ck splitK": split_k}
    ck_result, ck_us = run_perftest(run_ck, num_iters=num_iters, num_warmup=num_warmup)
    exact_result, exact_us = run_perftest(
        run_exact, num_iters=num_iters, num_warmup=num_warmup
    )

    effective_nbytes = _effective_nbytes(x, weight, x_scale, weight_scale, ck_out)
    _record_candidate(
        ret,
        "ck",
        ck_result,
        reference,
        ck_us,
        flops,
        effective_nbytes,
    )
    _record_candidate(
        ret,
        "exact",
        exact_result,
        reference,
        exact_us,
        flops,
        effective_nbytes,
    )
    ret["speedup vs ck"] = ck_us / exact_us
    return ret


def main() -> None:
    gfx = get_gfx()
    if gfx not in SUPPORTED_GFX:
        aiter.logger.warning(
            "DeepSeek-R1 M=32 blockscale GEMMs support %s only; got %s, skipping",
            sorted(SUPPORTED_GFX),
            gfx,
        )
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-iters", type=int, default=DEFAULT_NUM_ITERS)
    parser.add_argument("--num-warmup", type=int, default=DEFAULT_NUM_WARMUP)
    args = parser.parse_args()
    if args.num_iters < 1 or args.num_warmup < 0:
        parser.error("--num-iters must be positive and --num-warmup non-negative")

    rows = []
    for selector, (m, n, k) in DSR1_M32_KERNEL_SHAPES.items():
        torch.cuda.empty_cache()
        rows.append(
            test_gemm_a8w8_blockscale_dsr1_m32(
                selector,
                m,
                n,
                k,
                args.seed,
                args.num_iters,
                args.num_warmup,
            )
        )

    frame = pd.DataFrame(rows)
    aiter.logger.info(
        "DeepSeek-R1 M=32 blockscale GEMM summary (markdown):\n%s",
        frame.to_markdown(index=False, floatfmt=".4f"),
    )
    error_columns = ["ck err ratio", "exact err ratio"]
    if frame[error_columns].gt(0).any().any():
        raise AssertionError("At least one candidate failed the correctness tolerance")


if __name__ == "__main__":
    main()
