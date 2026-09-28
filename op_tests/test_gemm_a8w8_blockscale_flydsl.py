# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness/performance of both public blockscale APIs with FlyDSL configs."""

import argparse
import itertools
from importlib import import_module
from unittest.mock import patch

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops import gemm_op_a8w8 as gemm_ops
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx950"]


def run_torch(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    M, K = x.shape
    N = weight.shape[0]
    KB = K // 128
    a = (x.float().reshape(M, KB, 128) * x_scale[..., None]).reshape(M, K)
    b = (
        weight.float().reshape(N, KB, 128)
        * w_scale.repeat_interleave(128, dim=0)[:N, :, None]
    ).reshape(N, K)
    return (a @ b.T).to(dtype)


@benchmark()
def test_gemm(m, n, k, dtype, layout, scale_layout, data_init):
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_a8w8_blockscale_common import (
        kernel_fits_shape,
        kernels_list,
    )

    preshuffle_b = layout == "preshuffle"
    gfx = get_gfx()
    torch.manual_seed(0)
    if data_init == "signed":
        x = torch.randn((m, k), device="cuda").to(torch.float8_e4m3fn)
        weight = torch.randn((n, k), device="cuda").to(torch.float8_e4m3fn)
    else:
        x = (torch.rand((m, k), device="cuda") / 10).to(torch.float8_e4m3fn)
        weight = (torch.rand((n, k), device="cuda") / 10).to(torch.float8_e4m3fn)
    # Vary scales by both row and K block; all-ones scales miss layout errors.
    x_scale = torch.rand((m, k // 128), device="cuda") + 0.1
    w_scale = torch.rand(((n + 127) // 128, k // 128), device="cuda") + 0.1
    ref = run_torch(x, weight, x_scale, w_scale, dtype)
    gemm_weight = shuffle_weight(weight, layout=(16, 16)) if preshuffle_b else weight
    if preshuffle_b:
        transposed = x_scale.T.contiguous()
        gemm_scale = (
            transposed.T if scale_layout == "strided" else transposed.view_as(x_scale)
        )
    else:
        gemm_scale = x_scale

    candidates = {
        ki.name: ki
        for ki in kernels_list.values()
        if ki.preshuffle_b == preshuffle_b and kernel_fits_shape(ki, m, n, k, gfx)
    }
    if not candidates:
        raise ValueError(f"No FlyDSL blockscale candidate supports {(m, n, k, layout)}")
    flops = 2 * m * n * k
    nbytes = x.nbytes + weight.nbytes + x_scale.nbytes + w_scale.nbytes + m * n * 2
    ret = {"gfx": gfx}
    # Keep guard storage adjacent to Out to catch over-stores on M/N tails.
    guarded_out = torch.full((m * n + 256,), 42, dtype=dtype, device="cuda")
    out = guarded_out[128:-128].view(m, n)
    for name in candidates:
        out.fill_(float("nan"))
        config = {"libtype": "flydsl", "kernelName": name, "splitK": 0}
        with patch.object(gemm_ops, "get_CKGEMM_config", return_value=config):
            if preshuffle_b:
                fn = lambda: aiter.gemm_a8w8_blockscale_bpreshuffle(
                    x, gemm_weight, gemm_scale, w_scale, dtype=dtype, out=out
                )
            else:
                fn = lambda: aiter.gemm_a8w8_blockscale(
                    x, gemm_weight, gemm_scale, w_scale, dtype=dtype
                )
            result, us = run_perftest(fn, num_iters=21)
        if preshuffle_b:
            assert result.data_ptr() == out.data_ptr(), "out= must be honored"
        assert bool(torch.isfinite(result).all()), name
        assert bool((guarded_out[:128] == 42).all())
        assert bool((guarded_out[-128:] == 42).all())
        err = checkAllclose(
            ref.float(),
            result.float(),
            rtol=1e-2,
            atol=1e-2,
            tol_err_ratio=0,
            msg=name,
            catastrophic_check=True,
        )
        assert err == 0, f"{name}: error ratio {err}"
        tag = "flydsl"
        ret[f"{tag} us"] = us
        ret[f"{tag} TFLOPS"] = flops / us / 1e6
        ret[f"{tag} TB/s"] = nbytes / us / 1e6
        ret[f"{tag} err"] = err
    return ret


def main():
    if not torch.cuda.is_available() or get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("FlyDSL blockscale requires gfx950; skipping")
        return
    try:
        import_module("aiter.ops.flydsl")
    except ImportError as exc:
        aiter.logger.warning("FlyDSL blockscale unavailable; skipping: %s", exc)
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="+",
        choices=[dtypes.bf16],
        default=[dtypes.bf16],
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="+",
        default=[
            (1, 16, 256),
            # The same N/K with changing M exercises the cached launch ABI.
            (1, 384, 512),
            (33, 384, 512),
            (257, 384, 512),
            (257, 272, 768),
            (1024, 1024, 2048),
            (4096, 4096, 16384),
        ],
    )
    parser.add_argument(
        "--layout",
        nargs="+",
        choices=["plain", "preshuffle"],
        default=["plain", "preshuffle"],
    )
    parser.add_argument(
        "--scale-layout",
        nargs="+",
        choices=["packed", "strided"],
        default=["packed", "strided"],
    )
    parser.add_argument(
        "--data-init",
        nargs="+",
        choices=["uniform", "signed"],
        default=["uniform", "signed"],
    )
    args = parser.parse_args()
    rows = []
    for dtype, (m, n, k), layout, scale_layout, data_init in itertools.product(
        args.dtype, args.mnk, args.layout, args.scale_layout, args.data_init
    ):
        if layout == "plain" and scale_layout != "packed":
            continue
        rows.append(test_gemm(m, n, k, dtype, layout, scale_layout, data_init))
    aiter.logger.info(
        "FlyDSL blockscale summary:\n%s", pd.DataFrame(rows).to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
