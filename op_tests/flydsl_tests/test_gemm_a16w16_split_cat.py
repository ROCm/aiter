# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import argparse

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_a16w16_gfx1250 import gemm_a16w16
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")
BF16 = torch.bfloat16
HEADS, K_DIM, K_NOPE, V_DIM, ROPE_DIM = 128, 512, 128, 128, 64
CONFIG = {
    "tile_m": 128,
    "tile_n": 256,
    "tile_k": 64,
    "m_warp": 2,
    "n_warp": 4,
    "num_buffers": 2,
}


def run_split_cat(x, w, rope):  # the path it replaces: GEMM, then split + cat
    kv = gemm_a16w16(x, w, dtype=BF16, **CONFIG)
    kv = kv.view(x.shape[0], HEADS, K_NOPE + V_DIM)
    k = torch.cat((kv[..., :K_NOPE], rope.expand(-1, HEADS, -1)), dim=-1)
    return k, kv[..., K_NOPE:]


def pitched(m, width, pad):  # head and row padding, as a KV cache view
    hs = width + pad
    return torch.empty_strided(
        (m, HEADS, width), (HEADS * hs + 2 * pad, hs, 1), dtype=BF16
    )


@benchmark()
def test_gemm_a16w16_split_cat(m, pad):
    x, rope = torch.randn(m, K_DIM, dtype=BF16), torch.randn(m, 1, ROPE_DIM, dtype=BF16)
    w = torch.randn(HEADS * (K_NOPE + V_DIM), K_DIM, dtype=BF16)
    k_out, v_out = pitched(m, K_NOPE + ROPE_DIM, pad), pitched(m, V_DIM, pad)
    (ref_k, ref_v), ref_us = run_perftest(run_split_cat, x, w, rope)
    out = {
        "k_out": k_out,
        "v_out": v_out,
        "k_rope": rope,
        "split_sizes": (K_NOPE, V_DIM),
    }
    _, us = run_perftest(gemm_a16w16, x, w, dtype=BF16, **out, **CONFIG)
    err = checkAllclose(ref_k, k_out, 0, 0, msg="k") + checkAllclose(
        ref_v, v_out, 0, 0, msg="v"
    )
    assert err == 0, f"split-cat epilogue is not bitwise equal at M={m}"
    return {"gemm+split/cat us": ref_us, "fused us": us, "err": err}


def main():
    if get_gfx() != "gfx1250":
        aiter.logger.warning("gemm_a16w16 split-cat is gfx1250-only; skipping")
        return
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", type=int, nargs="*", default=[1, 31, 1024, 9424])
    ap.add_argument("--pad", type=int, nargs="*", default=[0, 8])
    args = ap.parse_args()
    df = [test_gemm_a16w16_split_cat(m, p) for m in args.m for p in args.pad]
    aiter.logger.info("split-cat:\n%s", pd.DataFrame(df).to_markdown(index=False))


if __name__ == "__main__":
    main()
