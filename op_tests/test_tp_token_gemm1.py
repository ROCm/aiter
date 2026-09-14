# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Step 1–2: MegaMoE GEMM1 over dense TP A gathered by sorted token ids.

Reference is the original contiguous ``ATileLoader`` on host-packed rows.
``wait`` is the same token-id kernel with host-prefilled ``payload_ready``.
No communication. Run with::

    python op_tests/test_tp_token_gemm1.py
"""

from __future__ import annotations

import argparse
import itertools

import flydsl.expr as fx
import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.mega_moe.gemm1 import gemm1_kernel
from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
from aiter.ops.flydsl.kernels.mega_moe.tp_incremental_schedule import (
    expected_token_counts,
    make_tile_row_base,
    pack_rows_by_sorted_ids,
)
from aiter.ops.flydsl.kernels.mega_moe.tp_incremental_stage1 import (
    gemm1_incremental_kernel,
)
from aiter.ops.flydsl.kernels.mega_moe.tp_token_gemm1 import gemm1_token_kernel
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]
SORT_BLOCK_M = 32
TILE_N = 256
TILE_K = 256


def _gathered_inputs(tp, m_local, model_dim, experts, topk, seed, device):
    xs, weights, ids = [], [], []
    for rank in range(tp):
        g = torch.Generator(device=device).manual_seed(seed + rank)
        xs.append(
            torch.randn(
                (m_local, model_dim),
                dtype=torch.bfloat16,
                device=device,
                generator=g,
            )
        )
        scores = torch.randn(
            (m_local, experts), dtype=torch.float32, device=device, generator=g
        )
        values, expert_ids = torch.topk(scores, topk, dim=-1)
        weights.append(values.softmax(dim=-1))
        ids.append(expert_ids.to(torch.int32))
    return torch.cat(xs, dim=0), torch.cat(weights, dim=0), torch.cat(ids, dim=0)


def _make_w1(experts, model_dim, inter_dim, device):
    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1 = torch.randn(
        (experts, 2 * inter_dim, model_dim), dtype=torch.bfloat16, device=device
    )
    w1.mul_(model_dim**-0.25)
    w1_q, w1_scale = quantize(w1, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(experts, 2 * inter_dim, model_dim // 2)
    w1_kernel = shuffle_weight_a16w4(w1_q, 16, True).contiguous()
    w1_scale_kernel = shuffle_scale_a16w4(w1_scale, experts, True).contiguous()
    return w1_kernel, w1_scale_kernel


def _alloc_out(num_valid, inter_dim, device):
    out = torch.zeros((num_valid, inter_dim), dtype=torch.float8_e4m3fn, device=device)
    scale_cols = (inter_dim // 32 + 7) // 8 * 8
    prows = ((num_valid + 255) // 256) * 256
    out_scale = torch.zeros(
        prows * scale_cols + inter_dim, dtype=torch.uint8, device=device
    )
    return out, out_scale


def _pad_dense(payload, scale):
    pad_row = torch.zeros(
        (1, payload.shape[1]), dtype=payload.dtype, device=payload.device
    )
    pad_scale = torch.zeros((1, scale.shape[1]), dtype=scale.dtype, device=scale.device)
    return torch.cat([payload, pad_row], dim=0), torch.cat([scale, pad_scale], dim=0)


def _fx_stream():
    return fx.Stream(torch.cuda.current_stream().cuda_stream)


def _gemm_kwargs(model_dim, inter_dim):
    return {
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "expert_offset": 0,
        "sort_block_m": SORT_BLOCK_M,
        "tile_n": TILE_N,
        "tile_k": TILE_K,
        "num_cu": torch.cuda.get_device_properties(0).multi_processor_count,
        "swiglu_limit": 0.0,
    }


@benchmark()
def test_tp_token_gemm1(tp, m_local, model_dim, inter_dim, experts, topk):
    device = torch.device("cuda")
    tokens = tp * m_local
    x_bf16, topk_weights, topk_ids = _gathered_inputs(
        tp, m_local, model_dim, experts, topk, seed=0, device=device
    )
    w1, w1_scale = _make_w1(experts, model_dim, inter_dim, device)
    x_fp8, x_scale = per_1x32_mx_quant(x_bf16, quant_mode="fp8")
    x_dense, scale_dense = _pad_dense(x_fp8, x_scale)

    sorted_ids, _, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, experts, model_dim, dtypes.bf16, SORT_BLOCK_M
    )
    num_valid = int(num_valid_ids[0].item())
    n_m = num_valid // SORT_BLOCK_M
    tile_row_base = make_tile_row_base(num_valid, SORT_BLOCK_M, device=device)
    expert_ids = sorted_expert_ids[:n_m].contiguous()
    x_packed = pack_rows_by_sorted_ids(x_dense, sorted_ids, num_valid, tokens)
    scale_packed = pack_rows_by_sorted_ids(scale_dense, sorted_ids, num_valid, tokens)

    kw = _gemm_kwargs(model_dim, inter_dim)
    stream = _fx_stream()
    out_ref, os_ref = _alloc_out(num_valid, inter_dim, device)
    gemm1_kernel(
        out_ref,
        x_packed,
        w1,
        scale_packed,
        w1_scale,
        tile_row_base,
        expert_ids,
        os_ref,
        num_valid,
        stream,
        **kw,
    )

    out_packed, os_packed = _alloc_out(num_valid, inter_dim, device)
    out_token, os_token = _alloc_out(num_valid, inter_dim, device)

    def run_packed():
        return gemm1_kernel(
            out_packed,
            x_packed,
            w1,
            scale_packed,
            w1_scale,
            tile_row_base,
            expert_ids,
            os_packed,
            num_valid,
            stream,
            **kw,
        )[0]

    def run_token():
        return gemm1_token_kernel(
            out_token,
            x_dense,
            w1,
            scale_dense,
            w1_scale,
            tile_row_base,
            expert_ids,
            sorted_ids,
            os_token,
            num_valid,
            tokens,
            stream,
            **kw,
        )[0]

    expected = expected_token_counts(topk_ids, experts).to(
        dtype=torch.int32, device=device
    )
    payload_ready = expected.clone()
    out_wait, os_wait = _alloc_out(num_valid, inter_dim, device)

    def run_wait():
        return gemm1_incremental_kernel(
            out_wait,
            x_dense,
            w1,
            scale_dense,
            w1_scale,
            tile_row_base,
            expert_ids,
            sorted_ids,
            os_wait,
            payload_ready,
            expected,
            num_valid,
            tokens,
            stream,
            **kw,
        )[0]

    candidates = {"packed": run_packed, "token": run_token, "wait": run_wait}
    flops = 2 * num_valid * model_dim * (2 * inter_dim)
    nbytes = (
        num_valid * model_dim
        + experts * 2 * inter_dim * (model_dim // 2)
        + num_valid * inter_dim
    )
    ret = {"gfx": get_gfx(), "num_valid": num_valid, "tokens": tokens}
    ref_fp32 = out_ref.float()
    for name, fn in candidates.items():
        out, us = run_perftest(fn, num_iters=21, num_warmup=2)
        err = checkAllclose(
            ref_fp32,
            out.float(),
            rtol=0,
            atol=0,
            msg=f"{name}: gemm1 out vs packed ATileLoader",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    os_err = checkAllclose(
        os_ref.float(),
        os_token.float(),
        rtol=0,
        atol=0,
        msg="token: gemm1 out_scale vs packed ATileLoader",
    )
    ret["token scale err"] = os_err
    wait_os_err = checkAllclose(
        os_ref.float(),
        os_wait.float(),
        rtol=0,
        atol=0,
        msg="wait: gemm1 out_scale vs packed ATileLoader",
    )
    ret["wait scale err"] = wait_os_err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("tp token gemm1 unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="activation dtype before MXFP8 quant (bf16 only)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[16, 32],
        help="m_local (tokens per fake TP rank)",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(256, 128)],
        help="model_dim,inter_dim. e.g.: -s 256,128",
    )
    parser.add_argument("--tp", type=int, nargs="*", default=[2], help="fake TP degree")
    parser.add_argument("--experts", type=int, nargs="*", default=[8])
    parser.add_argument("--topk", type=int, nargs="*", default=[2])
    args = parser.parse_args()

    for _dtype in args.dtype:
        rows = []
        for tp, m_local, (model_dim, inter_dim), experts, topk in itertools.product(
            args.tp, args.batch, args.mnk, args.experts, args.topk
        ):
            rows.append(
                test_tp_token_gemm1(tp, m_local, model_dim, inter_dim, experts, topk)
            )
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "tp token gemm1 summary (markdown):\n%s", df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
