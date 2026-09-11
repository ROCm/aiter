# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Eight-wave MXFP8 prefill adapter for the standard AITER sorted MoE ABI.

Uses the existing G1U1 16x64 weight packing and E8M0 scale layout. Workspace
bounds come from tensor shapes; the valid sorted row count stays on the GPU.
Stage 1 returns sorted FP8 activations and scales; stage 2 reduces sorted BF16
partials in FP32. Neither weight repacking nor host row-count readback is needed.
"""

import functools
import os
import re

import torch

_NAME = re.compile(r"flydsl_moe([12])_mxfp8_8w_t(256x256|128x512)_xcd([0-9]+)")


def kernel_name(stage, tile_m=256, tile_n=256, swizzle=1):
    return f"flydsl_moe{stage}_mxfp8_8w_t{tile_m}x{tile_n}_xcd{swizzle}"


def kernel_params(name):
    match = _NAME.fullmatch(name or "")
    if match is None:
        return None
    stage, tile, swizzle = match.groups()
    tile_m, tile_n = map(int, tile.split("x"))
    if int(swizzle) not in (0, 1, 2, 3, 4, 8):
        raise ValueError(f"Unsupported MXFP8 eight-wave swizzle: {swizzle}")
    if int(stage) == 2 and tile_m != 256:
        raise ValueError("MXFP8 eight-wave stage 2 requires 256x256")
    return {
        "stage": int(stage),
        "tile_m": tile_m,
        "tile_n": tile_n,
        "xcd_swizzle": int(swizzle),
        "mode": "reduce",
        "sort_block_m": 256,
        "a_dtype": "fp8",
        "b_dtype": "fp8",
        "out_dtype": "fp8" if int(stage) == 1 else "bf16",
    }


@functools.lru_cache(maxsize=256)
def _builder(kind, **kwargs):
    from .kernels.mxfp8_moe_8wave.moe import (
        compile_mxfp8_moe_gemm_8w,
        compile_mxfp8_moe_quant,
        compile_mxfp8_moe_reduce,
        compile_mxfp8_moe_unpack_routes,
    )

    return {
        "gemm": compile_mxfp8_moe_gemm_8w,
        "quant": compile_mxfp8_moe_quant,
        "reduce": compile_mxfp8_moe_reduce,
        "routes": compile_mxfp8_moe_unpack_routes,
    }[kind](**kwargs)


def _run(kind, args, **kwargs):
    from .kernels.tensor_shim import _run_compiled

    _run_compiled(_builder(kind, **kwargs), *args)


def _stream():
    return 0 if os.environ.get("COMPILE_ONLY") == "1" else torch.cuda.current_stream()


def _bytes(t):
    return t.view(torch.int8).view(-1)


def _routes(sorted_ids, valid, tokens, topk):
    rows = (sorted_ids.numel() + 255) // 256 * 256
    row_map = torch.empty(rows, dtype=torch.int32, device=sorted_ids.device)
    # Masked/remote routes are zero in the final reduction, including empty EP.
    inverse = torch.full(
        (tokens * topk,), -1, dtype=torch.int32, device=sorted_ids.device
    )
    _run(
        "routes",
        (
            sorted_ids,
            row_map,
            inverse,
            rows,
            tokens,
            valid,
            _stream(),
        ),
        topk=topk,
        dynamic_rows=True,
    )
    return rows, row_map, inverse


def stage1(
    hidden_states,
    w1,
    w2,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    *,
    block_m,
    kernelName,
    a1_scale=None,
    w1_scale=None,
    sorted_weights=None,
    swiglu_limit=None,
):
    params = kernel_params(kernelName)
    if params is None or params["stage"] != 1 or block_m != 256:
        raise ValueError(
            f"Invalid eight-wave stage-1 configuration: {kernelName}, block_m={block_m}"
        )
    if hidden_states.dtype != torch.bfloat16 or sorted_weights is not None:
        raise ValueError(
            "Eight-wave MXFP8 prefill requires BF16 input and stage-2 routing weights"
        )
    tokens, hidden = hidden_states.shape
    inter = w2.shape[-1]
    rows, row_map, inverse = _routes(sorted_ids, num_valid_ids, tokens, topk)
    device = hidden_states.device
    aq = torch.empty((tokens, hidden), dtype=torch.int8, device=device)
    sa = torch.full((rows, hidden // 32), 127, dtype=torch.uint8, device=device)
    act = torch.empty((rows, inter), dtype=torch.bfloat16, device=device)
    kp = (inter + 255) // 256 * 256
    aq2 = torch.empty((rows, kp), dtype=torch.int8, device=device)
    sa2 = torch.empty((rows, kp // 32), dtype=torch.uint8, device=device)
    stream = _stream()
    _run(
        "quant",
        (hidden_states.view(-1), aq.view(-1), sa.view(-1), inverse, tokens, stream),
        K=hidden,
        gather=False,
        scatter_scale_topk=topk,
    )
    _run(
        "gemm",
        (
            aq.view(-1),
            _bytes(w1),
            act.view(-1),
            sa.view(-1),
            _bytes(w1_scale),
            sorted_expert_ids,
            row_map,
            num_valid_ids,
            rows,
            2 * inter,
            stream,
        ),
        K=hidden,
        stage=1,
        gather_a=True,
        dynamic_rows=True,
        tile_m=params["tile_m"],
        tile_n=params["tile_n"],
        xcd_swizzle=params["xcd_swizzle"],
        swiglu_limit=7.0 if swiglu_limit is None else float(swiglu_limit),
    )
    _run(
        "quant",
        (
            act.view(-1),
            aq2.view(-1),
            sa2.view(-1),
            row_map,
            rows,
            num_valid_ids,
            stream,
        ),
        K=inter,
        gather=False,
        dynamic_rows=True,
    )
    return aq2, sa2


def stage2(
    hidden_states,
    w1,
    w2,
    sorted_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    *,
    block_m,
    kernelName,
    a2_scale=None,
    w2_scale=None,
    sorted_weights=None,
):
    params = kernel_params(kernelName)
    if params is None or params["stage"] != 2 or block_m != 256:
        raise ValueError(
            f"Invalid eight-wave stage-2 configuration: {kernelName}, block_m={block_m}"
        )
    tokens, hidden = out.shape
    inter = w2.shape[-1]
    rows, row_map, inverse = _routes(sorted_ids, num_valid_ids, tokens, topk)
    partial = torch.empty((rows, hidden), dtype=torch.bfloat16, device=out.device)
    stream = _stream()
    _run(
        "gemm",
        (
            hidden_states.view(-1),
            _bytes(w2),
            partial.view(-1),
            a2_scale.view(-1),
            _bytes(w2_scale),
            sorted_expert_ids,
            row_map,
            num_valid_ids,
            rows,
            hidden,
            stream,
        ),
        K=hidden_states.shape[-1],
        b_k=inter,
        logical_k=inter,
        stage=2,
        dynamic_rows=True,
        xcd_swizzle=params["xcd_swizzle"],
    )
    _run(
        "reduce",
        (partial.view(-1), out.view(-1), inverse, sorted_weights, tokens, stream),
        N=hidden,
        topk=topk,
        sorted_weights=True,
    )
    return out


stage1._is_mxfp8_8wave_stage1 = True
stage2._is_flydsl_stage2 = True


def precompile(kernelName, token_num, model_dim, inter_dim, experts, topk):
    """Compile through the runtime adapter using FakeTensorMode and COMPILE_ONLY."""
    params = kernel_params(kernelName)
    tokens, hidden, inter = token_num, model_dim, inter_dim
    count = tokens * topk + experts * 256 - topk
    rows = (count + 255) // 256 * 256
    kwargs = {"device": "cpu"}
    w1 = torch.empty((experts, 2 * inter, hidden), dtype=torch.int8, **kwargs)
    w2 = torch.empty((experts, hidden, inter), dtype=torch.int8, **kwargs)
    ids = torch.empty(count, dtype=torch.int32, **kwargs)
    eids = torch.empty(rows // 256, dtype=torch.int32, **kwargs)
    valid = torch.empty(2, dtype=torch.int32, **kwargs)
    if params["stage"] == 1:
        x = torch.empty((tokens, hidden), dtype=torch.bfloat16, **kwargs)
        scale = torch.empty(
            experts * 2 * inter * (hidden // 32), dtype=torch.int8, **kwargs
        )
        stage1(
            x,
            w1,
            w2,
            ids,
            eids,
            valid,
            None,
            topk,
            block_m=256,
            kernelName=kernelName,
            w1_scale=scale,
        )
    else:
        kp = (inter + 255) // 256 * 256
        x = torch.empty((rows, kp), dtype=torch.int8, **kwargs)
        scale = torch.empty(experts * hidden * (kp // 32), dtype=torch.int8, **kwargs)
        ascale = torch.empty((rows, kp // 32), dtype=torch.uint8, **kwargs)
        weights = torch.empty(count, dtype=torch.float32, **kwargs)
        out = torch.empty((tokens, hidden), dtype=torch.bfloat16, **kwargs)
        stage2(
            x,
            w1,
            w2,
            ids,
            eids,
            valid,
            out,
            topk,
            block_m=256,
            kernelName=kernelName,
            w2_scale=scale,
            a2_scale=ascale,
            sorted_weights=weights,
        )
