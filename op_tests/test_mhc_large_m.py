# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx950 large-M (M > 1024) mhc_post_pre kernel test.

Exercises the gfx950 large-M ``mhc_fused_post_pre_large_m`` path
without modifying PR #3623 kernels used at M <= 1024.
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.mhc import mhc_fused_post_pre_large_m
from aiter.test_common import benchmark, checkAllclose, run_perftest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_mhc import mhc_post_pre_ref, mhc_post_pre_unfused_hip  # noqa: E402

LARGE_M_MIN = 1025
torch.set_default_device("cuda")


@benchmark()
def test_mhc_large_m_post_pre(
    m: int, hidden_size: int, hc_mult: int = 4, fuse_rmsnorm: bool = False
):
    if m <= 1024:
        aiter.logger.info("skip large_m: m=%s <= 1024", m)
        return {"skipped": True}
    if hidden_size < 512:
        aiter.logger.info("skip large_m: hidden_size=%s < 512", hidden_size)
        return {"skipped": True}
    if get_gfx_runtime() != "gfx950":
        aiter.logger.info("skip large_m: gfx=%s (gfx950 only)", get_gfx_runtime())
        return {"skipped": True}
    if not hasattr(aiter, "mhc_fused_post_pre_large_m"):
        aiter.logger.info("skip large_m: mhc_fused_post_pre_large_m unavailable")
        return {"skipped": True}

    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    layer_input = torch.randn(m, hidden_size, dtype=dtypes.bf16)
    residual_in = torch.randn(m, hc_mult, hidden_size, dtype=dtypes.bf16)
    post_layer_mix = torch.randn(m, hc_mult, 1, dtype=dtypes.fp32)
    comb_res_mix = torch.randn(m, hc_mult, hc_mult, dtype=dtypes.fp32)
    fn = torch.randn(hc_mult3, hc_mult * hidden_size, dtype=dtypes.fp32)
    hc_scale = torch.randn((3,), dtype=dtypes.fp32) * 0.1
    hc_base = torch.randn((hc_mult3,), dtype=dtypes.fp32) * 0.1
    norm_weight = torch.randn(hidden_size, dtype=dtypes.bf16) if fuse_rmsnorm else None

    extra_args = {
        "rms_eps": 1e-6,
        "hc_pre_eps": 1e-6,
        "hc_sinkhorn_eps": 1e-6,
        "hc_post_mult_value": 2.0,
        "sinkhorn_repeat": 20,
    }
    if fuse_rmsnorm:
        extra_args["norm_eps"] = 1e-6

    post_mix_ref, comb_mix_ref, layer_input_ref, next_residual_ref = mhc_post_pre_ref(
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        norm_weight=norm_weight,
        **extra_args,
    )

    hip_kwargs = {**extra_args}
    if fuse_rmsnorm:
        hip_kwargs["norm_weight"] = norm_weight

    (_, _, _, _), unfused_us = run_perftest(
        mhc_post_pre_unfused_hip,
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        **hip_kwargs,
    )

    (_, _, layer_input_large_m, _), large_m_us = run_perftest(
        mhc_fused_post_pre_large_m,
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        **hip_kwargs,
    )

    (_, _, layer_input_dispatch, _), dispatch_us = run_perftest(
        aiter.mhc_fused_post_pre,
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        force_fused=True,
        **hip_kwargs,
    )

    hip_large_m_err = checkAllclose(
        layer_input_ref, layer_input_large_m, msg="large_m/layer_input"
    )
    hip_dispatch_err = checkAllclose(
        layer_input_ref, layer_input_dispatch, msg="dispatch/layer_input"
    )

    return {
        "m": m,
        "hidden_size": hidden_size,
        "hc_mult": hc_mult,
        "fuse_rmsnorm": fuse_rmsnorm,
        "unfused_us": unfused_us,
        "large_m_us": large_m_us,
        "dispatch_us": dispatch_us,
        "hip_large_m_err": hip_large_m_err,
        "hip_dispatch_err": hip_dispatch_err,
    }


parser = argparse.ArgumentParser(description="gfx950 large-M mhc_post_pre benchmark")
parser.add_argument(
    "-m",
    type=int,
    nargs="*",
    default=[2048, 8192, 65536],
    help="M values (must be > 1024 for large-M kernel)",
)
parser.add_argument(
    "-n",
    "--hidden_size",
    type=int,
    nargs="*",
    default=[4096, 7168],
    help="hidden_size",
)
parser.add_argument(
    "--fuse_rmsnorm",
    action="store_true",
    help="Fuse RMSNorm in large-M path",
)

args = parser.parse_args()
rows = []
for m in args.m:
    for hidden_size in args.hidden_size:
        torch.cuda.empty_cache()
        gc.collect()
        try:
            ret = test_mhc_large_m_post_pre(
                m=m,
                hidden_size=hidden_size,
                fuse_rmsnorm=args.fuse_rmsnorm,
            )
        except torch.OutOfMemoryError as e:
            aiter.logger.warning("OOM m=%s hidden_size=%s: %s", m, hidden_size, e)
            continue
        if ret.get("skipped"):
            continue
        rows.append(ret)
        torch.cuda.empty_cache()
        gc.collect()

if rows:
    df = pd.DataFrame(rows)
    aiter.logger.info("mhc_large_m summary:\n%s", df.to_markdown(index=False))
else:
    aiter.logger.info("mhc_large_m: all cases skipped")
