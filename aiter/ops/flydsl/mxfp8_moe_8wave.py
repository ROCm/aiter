# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP8/A8W4 prefill adapter for the standard AITER sorted MoE ABI.

Uses the existing G1U1 16x64 weight packing and E8M0 scale layout. Workspace
bounds come from tensor shapes; the valid sorted row count stays on the GPU.
Stage 1 returns sorted FP8 activations and scales. A8W4 256x256 and four-wave
128x256 tiles fuse output quantization. Stage 2 reduces sorted BF16 partials in FP32.
A8W4 names select packed E2M1 weights and standard SiLU.
Neither weight repacking nor host row-count readback is needed. A8W4 remains
opt-in: the existing DSV4 tuned kernels remain the default.
"""

import functools
import os
import re

import torch

_NAME = re.compile(
    r"flydsl_moe([12])_(mxfp8|a8w4)_([48])w_t(256x256|128x512|128x256)(?:_persistent([24]))?_xcd([0-9]+)"
)


def is_kernel_name(name):
    """Recognize this adapter's namespace before validating a complete name."""
    return any(
        marker in str(name) for marker in ("_mxfp8_8w_", "_a8w4_8w_", "_a8w4_4w_")
    )


def kernel_name(
    stage, tile_m=256, tile_n=256, swizzle=1, b_dtype="fp8", waves=8, persistent_tiles=0
):
    if b_dtype not in ("fp8", "fp4"):
        raise ValueError(f"Unsupported weight dtype: {b_dtype}")
    if persistent_tiles not in (0, 2, 4):
        raise ValueError(f"Unsupported persistent tile count: {persistent_tiles}")
    family = "a8w4" if b_dtype == "fp4" else "mxfp8"
    suffix = f"_persistent{persistent_tiles}" if persistent_tiles else ""
    return (
        f"flydsl_moe{stage}_{family}_{waves}w_t{tile_m}x{tile_n}{suffix}_xcd{swizzle}"
    )


def kernel_params(name):
    match = _NAME.fullmatch(name or "")
    if match is None:
        return None
    stage, family, waves, tile, persistent, swizzle = match.groups()
    waves = int(waves)
    tile_m, tile_n = map(int, tile.split("x"))
    if int(swizzle) not in (0, 1, 2, 3, 4, 8):
        raise ValueError(f"Unsupported MXFP8 eight-wave swizzle: {swizzle}")
    if persistent and (int(stage), family, waves, tile_m, tile_n) != (
        2,
        "mxfp8",
        8,
        256,
        256,
    ):
        raise ValueError("Persistent prefill requires MXFP8 stage2 tile256x256")
    if waves == 4:
        if family != "a8w4" or (tile_m, tile_n) != (128, 256):
            raise ValueError("Four-wave prefill requires A8W4 tile128x256")
    elif (tile_m, tile_n) not in ((256, 256), (128, 512)):
        raise ValueError("Eight-wave prefill requires tile256x256 or tile128x512")
    if waves == 8 and int(stage) == 2 and tile_m != 256:
        raise ValueError("MXFP8 eight-wave stage 2 requires 256x256")
    return {
        "stage": int(stage),
        "num_waves": waves,
        "persistent_tiles": int(persistent) if persistent else 0,
        "tile_m": tile_m,
        "tile_n": tile_n,
        "xcd_swizzle": int(swizzle),
        "mode": "reduce",
        "sort_block_m": 128 if waves == 4 else 256,
        "a_dtype": "fp8",
        "b_dtype": "fp4" if family == "a8w4" else "fp8",
        "activation_type": "silu" if family == "a8w4" else "swiglu",
        "out_dtype": "fp8" if int(stage) == 1 else "bf16",
    }


@functools.lru_cache(maxsize=256)
def _builder(kind, **kwargs):
    persistent_tiles = kwargs.pop("persistent_tiles", 0)
    if kind == "gemm" and persistent_tiles:
        from .kernels.mxfp8_moe_gemm2_persistent import (
            compile_mxfp8_moe_gemm_persistent,
        )

        return compile_mxfp8_moe_gemm_persistent(m_tiles=persistent_tiles, **kwargs)
    if kind == "gemm" and kwargs.pop("num_waves", 8) == 4:
        from .kernels.mxfp8_moe_4wave import compile_mxfp8_moe_gemm_4w

        return compile_mxfp8_moe_gemm_4w(**kwargs)
    from .kernels.mxfp8_moe_8wave import (
        compile_mxfp8_moe_gemm_8w,
        compile_mxfp8_moe_quant,
        compile_mxfp8_moe_reduce,
        compile_mxfp8_moe_sort_input_scale,
        compile_mxfp8_moe_unpack_routes,
    )

    return {
        "gemm": compile_mxfp8_moe_gemm_8w,
        "quant": compile_mxfp8_moe_quant,
        "reduce": compile_mxfp8_moe_reduce,
        "sort_scale": compile_mxfp8_moe_sort_input_scale,
        "routes": compile_mxfp8_moe_unpack_routes,
    }[kind](**kwargs)


def _run(kind, args, **kwargs):
    from .kernels.tensor_shim import _preload_compiled, _run_compiled

    launch = _builder(kind, **kwargs)
    if os.environ.get("COMPILE_ONLY") == "1":
        _preload_compiled(launch, *args)
    else:
        _run_compiled(launch, *args)


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
    if params is None or params["stage"] != 1 or block_m != params["sort_block_m"]:
        raise ValueError(
            f"Invalid eight-wave stage-1 configuration: {kernelName}, block_m={block_m}"
        )
    if hidden_states.dtype != torch.bfloat16 or sorted_weights is not None:
        raise ValueError(
            "Eight-wave MXFP8 prefill requires BF16 input and stage-2 routing weights"
        )
    tokens, hidden = hidden_states.shape
    inter = w2.shape[-1] * (2 if params["b_dtype"] == "fp4" else 1)
    device = hidden_states.device
    fuse_quant = params["b_dtype"] == "fp4" and (
        params["tile_m"] == 256 or params["num_waves"] == 4
    )
    kp = (inter + 255) // 256 * 256
    if params["b_dtype"] == "fp4":
        rows = (sorted_ids.numel() + 255) // 256 * 256
        row_map = torch.empty(rows, dtype=torch.int32, device=device)
        sa = torch.empty((rows, hidden // 32), dtype=torch.uint8, device=device)
        if os.environ.get("COMPILE_ONLY") == "1":
            aq = torch.empty((tokens, hidden), dtype=torch.int8, device=device)
            scale_per_token = torch.empty(
                (tokens, hidden // 32), dtype=torch.uint8, device=device
            )
        else:
            from aiter.ops.quant import per_1x32_mx_quant_hip

            aq, scale_per_token = per_1x32_mx_quant_hip(
                hidden_states,
                quant_dtype=torch.float8_e4m3fn,
                scale_type=torch.float8_e8m0fnu,
                shuffle=False,
            )
        _run(
            "sort_scale",
            (
                scale_per_token.view(torch.uint8).view(-1),
                sa.view(torch.int32).view(-1),
                sorted_ids,
                row_map,
                num_valid_ids,
                tokens,
                rows,
                _stream(),
            ),
            K=hidden,
            topk=topk,
        )
    else:
        rows, row_map, inverse = _routes(sorted_ids, num_valid_ids, tokens, topk)
        aq = torch.empty((tokens, hidden), dtype=torch.int8, device=device)
        sa = torch.full((rows, hidden // 32), 127, dtype=torch.uint8, device=device)
        _run(
            "quant",
            (
                hidden_states.view(-1),
                aq.view(-1),
                sa.view(-1),
                inverse,
                tokens,
                _stream(),
            ),
            K=hidden,
            gather=False,
            scatter_scale_topk=topk,
        )
    if fuse_quant:
        packed_output = torch.empty(
            rows * (kp + kp // 32), dtype=torch.int8, device=device
        )
        aq2 = packed_output[: rows * kp].view(rows, kp)
        sa2 = packed_output[rows * kp :].view(rows, kp // 32).view(torch.uint8)
        gemm_output = packed_output
    else:
        act = torch.empty((rows, inter), dtype=torch.bfloat16, device=device)
        aq2 = torch.empty((rows, kp), dtype=torch.int8, device=device)
        sa2 = torch.empty((rows, kp // 32), dtype=torch.uint8, device=device)
        gemm_output = act.view(-1)
    stream = _stream()
    _run(
        "gemm",
        (
            _bytes(aq),
            _bytes(w1),
            gemm_output,
            _bytes(sa),
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
        swiglu_limit=None if swiglu_limit is None else float(swiglu_limit),
        activation_type=params["activation_type"],
        b_dtype=params["b_dtype"],
        expert_block_m=block_m,
        fuse_quant=fuse_quant,
        **({"num_waves": 4} if params["num_waves"] == 4 else {}),
    )
    if not fuse_quant:
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
    if params is None or params["stage"] != 2 or block_m != params["sort_block_m"]:
        raise ValueError(
            f"Invalid eight-wave stage-2 configuration: {kernelName}, block_m={block_m}"
        )
    tokens, hidden = out.shape
    inter = w2.shape[-1] * (2 if params["b_dtype"] == "fp4" else 1)
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
        b_dtype=params["b_dtype"],
        expert_block_m=block_m,
        tile_m=params["tile_m"],
        **({"num_waves": 4} if params["num_waves"] == 4 else {}),
        **(
            {"persistent_tiles": params["persistent_tiles"]}
            if params["persistent_tiles"]
            else {}
        ),
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
    if params is None:
        raise ValueError(f"Invalid eight-wave kernel name: {kernelName}")
    tokens, hidden, inter = token_num, model_dim, inter_dim
    pack = 2 if params["b_dtype"] == "fp4" else 1
    block_m = params["sort_block_m"]
    count = tokens * topk + experts * block_m - topk
    rows = (count + 255) // 256 * 256
    kwargs = {"device": "cpu"}
    w1 = torch.empty((experts, 2 * inter, hidden // pack), dtype=torch.int8, **kwargs)
    w2 = torch.empty((experts, hidden, inter // pack), dtype=torch.int8, **kwargs)
    ids = torch.empty(count, dtype=torch.int32, **kwargs)
    eids = torch.empty((count + block_m - 1) // block_m, dtype=torch.int32, **kwargs)
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
            block_m=block_m,
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
            block_m=block_m,
            kernelName=kernelName,
            w2_scale=scale,
            a2_scale=ascale,
            sorted_weights=weights,
        )
