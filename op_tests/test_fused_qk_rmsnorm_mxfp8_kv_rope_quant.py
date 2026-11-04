#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""HIP fused_qk_rmsnorm_mxfp8_kv_rope_quant: the MXFP8 dual RMSNorm whose bf16 kv row
runs straight into the KV half of DeepSeek V4.1's rope_quant_window.

The KV sinks (qat, values / scales, window ring) must equal, byte for byte, the torch
rope_quant_window reference run on the op's own normed kv row. q and that row follow the
triton fused_dual_rmsnorm_mxfp8_quant.
"""

import argparse
import random

import torch

import aiter
from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_mxfp8_kv_rope_quant
from aiter.ops.triton.quant.fused_mxfp8_quant import fused_dual_rmsnorm_mxfp8_quant
from aiter.test_common import checkAllclose, perftest
from op_tests.rope_quant_window_ref import (
    COS,
    RD,
    SIN,
    D,
    Step,
    byte_diff,
    dev,
    rope_quant_window_ref,
    sink_kwargs,
    sinks,
    with_zeros,
)

MODES = ["rows", "qat+vs", "ringb", "ringbu", "ringh", "qat+vs+ringb"]


@perftest()
def run_hip(q, gq, kv, gk, positions, kv_out, kw):
    return fused_qk_rmsnorm_mxfp8_kv_rope_quant(
        q,
        gq,
        1e-6,
        kv,
        gk,
        1e-6,
        COS,
        SIN,
        positions,
        rope_dim=RD,
        amax_floor=1e-10,
        kv_out=kv_out,
        **kw,
    )


def test_fused_qk_rmsnorm_mxfp8_kv_rope_quant(mode, tokens, idx64, n1=1536):
    step = Step(tokens, idx64)
    # q and kv are adjacent slices of one projection, as in the model.
    qkv = with_zeros((torch.randn(tokens, n1 + D, device=dev) * 3).to(torch.bfloat16))
    q, kv = torch.split(qkv, [n1, D], dim=1)
    gq = (torch.rand(n1, device=dev) * 2).to(torch.bfloat16)
    gk = ((torch.rand(D, device=dev) - 0.2) * 2).to(torch.bfloat16)
    got, ref = sinks(mode, tokens, 0), sinks(mode, tokens, 0)
    kv_out = torch.empty_like(kv)
    (yq, sq), us = run_hip(
        q, gq, kv, gk, step.positions, kv_out, sink_kwargs(got, step)
    )
    empty = torch.empty(tokens, 0, D, device=dev, dtype=torch.bfloat16)
    rope_quant_window_ref(
        empty, kv_out, step.positions, rope_dim=RD, **sink_kwargs(ref, step)
    )
    yq_t, sq_t, yk_t = fused_dual_rmsnorm_mxfp8_quant(
        q, kv, gq, gk, 1e-6, eps_k=1e-6, amax_floor=1e-10
    )
    sq_t = sq_t.view(torch.uint8)
    torch.cuda.synchronize()

    tag = f"{mode} T={tokens} idx64={idx64}"
    checkAllclose(yk_t, kv_out, rtol=0.02, atol=0.02, msg=f"{tag} kv_out vs triton")
    # An amax on a rounding boundary may move its e8m0 code by one; the fp8 values are
    # compared where the codes agree, within one e4m3 step.
    checkAllclose(
        sq_t.float(), sq.float(), rtol=0, atol=1, msg=f"{tag} q scale vs triton"
    )
    checkAllclose(
        yq_t.float(),
        yq.float(),
        rtol=0.13,
        atol=2**-9,
        mask=(sq_t == sq).repeat_interleave(32, dim=1),
        msg=f"{tag} q vs triton",
    )
    bad = byte_diff(ref, got)
    result = f"\033[31mbytes differ {bad}\033[0m" if bad else "byte exact"
    aiter.logger.info(
        "[result] %s | %.2f us | kv sinks vs reference: %s", tag, us, result
    )
    return not bad


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-t", "--token", type=int, nargs="*", default=[1, 5, 13, 128, 1000, 1025, 9000]
    )
    parser.add_argument("--mode", type=str, nargs="*", default=MODES, choices=MODES)
    args = parser.parse_args()
    random.seed(1)
    torch.manual_seed(1)

    failed = [
        f"{mode} T={tokens} idx64={idx64}"
        for mode in args.mode
        for tokens in args.token
        for idx64 in (False, True)
        if not test_fused_qk_rmsnorm_mxfp8_kv_rope_quant(mode, tokens, idx64)
    ]
    if failed:
        raise SystemExit(f"{len(failed)} cases differ: {failed}")
    aiter.logger.info("fused_qk_rmsnorm_mxfp8_kv_rope_quant: every case byte exact")
