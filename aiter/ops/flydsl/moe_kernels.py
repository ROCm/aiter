# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MOE kernel management: naming, compilation, and high-level API."""

import functools
from typing import Dict, Optional

import torch

_KERNEL_PARAMS: Dict[str, Dict] = {}


def flydsl_kernel_name(
    stage: int,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    mode: str = "",
) -> str:
    """Construct kernel name for the configured stage and tile shape."""
    name = f"flydsl_moe{stage}_a{a_dtype}_w{b_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
    if mode:
        name += f"_{mode}"
    return name


def get_flydsl_kernel_params(name: str) -> Optional[Dict]:
    """Lookup kernel params by name (O(1))."""
    return _KERNEL_PARAMS.get(name)


def get_flydsl_stage1_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported stage1 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"
    tile_ns = [128, 256] if is_fp4 else [128]
    tile_ks = [128, 256] if is_fp4 else [128]
    tile_ms = [16, 32, 64, 128]

    for tm in tile_ms:
        for tn in tile_ns:
            for tk in tile_ks:
                name = flydsl_kernel_name(1, a_dtype, b_dtype, out_dtype, tm, tn, tk)
                kernels[name] = {
                    "stage": 1,
                    "a_dtype": a_dtype,
                    "b_dtype": b_dtype,
                    "out_dtype": out_dtype,
                    "tile_m": tm,
                    "tile_n": tn,
                    "tile_k": tk,
                    "MPerBlock": tm,
                }
    return kernels


def get_flydsl_stage2_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported stage2 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"
    tile_ns = [128, 256] if is_fp4 else [128]
    tile_ks = [128, 256] if is_fp4 else [128]
    tile_ms = [32, 64, 128]
    modes = ["atomic", "reduce"]

    for tm in tile_ms:
        for tn in tile_ns:
            for tk in tile_ks:
                for mode in modes:
                    name = flydsl_kernel_name(
                        2, a_dtype, b_dtype, out_dtype, tm, tn, tk, mode
                    )
                    kernels[name] = {
                        "stage": 2,
                        "a_dtype": a_dtype,
                        "b_dtype": b_dtype,
                        "out_dtype": out_dtype,
                        "tile_m": tm,
                        "tile_n": tn,
                        "tile_k": tk,
                        "mode": mode,
                        "MPerBlock": tm,
                    }
    return kernels


def _register_all_configs():
    """Pre-populate _KERNEL_PARAMS with all supported configs at import time."""
    for a in ("fp8", "fp4", "fp16"):
        for b in ("fp4",):
            for out in ("bf16", "f16"):
                _KERNEL_PARAMS.update(get_flydsl_stage1_kernels(a, b, out))
                _KERNEL_PARAMS.update(get_flydsl_stage2_kernels(a, b, out))


_register_all_configs()


def compile_flydsl_moe_stage1(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage1: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    act: str = "silu",
    persist_m: int = 1,
):
    """Compile stage1 kernel (cached via underlying lru_cache)."""
    if b_dtype == "fp4":
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1

        return compile_mixed_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=doweight_stage1,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            act=act,
            persist_m=persist_m,
        )
    else:
        from .kernels.moe_gemm_2stage import compile_moe_gemm1

        return compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=doweight_stage1,
            in_dtype=a_dtype,
            out_dtype=out_dtype,
        )


def compile_flydsl_moe_stage2(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight_stage2: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    accumulate: bool = True,
    persist_m: int = 1,
):
    """Compile stage2 kernel (cached via underlying lru_cache)."""
    if b_dtype == "fp4":
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm2

        return compile_mixed_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            a_dtype=a_dtype,
            b_dtype=b_dtype,
            out_dtype=out_dtype,
            accumulate=accumulate,
        )
    else:
        from .kernels.moe_gemm_2stage import compile_moe_gemm2

        return compile_moe_gemm2(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage2=doweight_stage2,
            in_dtype=a_dtype,
            out_dtype=out_dtype,
            accumulate=accumulate,
        )


# Private: compiled kernel closures


@functools.cache
def _get_compiled_stage1(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    act: str,
    persist_m: int = 1,
):
    """Compile and cache stage1 kernel, return a tensor_api closure."""
    exe = compile_flydsl_moe_stage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=doweight,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        act=act,
        persist_m=persist_m,
    )
    is_fp4 = b_dtype == "fp4"
    _n_in = inter_dim * 2 if is_fp4 else inter_dim
    _k_in = model_dim

    def tensor_api(
        out: torch.Tensor,
        a: torch.Tensor,
        w: torch.Tensor,
        a_scale: torch.Tensor,
        w_scale: torch.Tensor,
        sorted_ids: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_valid_ids: torch.Tensor,
        token_num: int,
        size_expert_ids_in: int,
    ) -> None:
        if is_fp4:
            empty_bias = torch.empty(0, device=a.device, dtype=torch.float32)
            stream = torch.cuda.current_stream()
            # fp4/e8m0 dtypes are not supported by dlpack; reinterpret as uint8.
            _a = (
                a.view(torch.uint8)
                if a.dtype
                not in (torch.uint8, torch.float16, torch.bfloat16, torch.float32)
                else a
            )
            _w = (
                w.view(torch.uint8)
                if w.dtype
                not in (torch.uint8, torch.float16, torch.bfloat16, torch.float32)
                else w
            )
            _as = (
                a_scale.view(torch.uint8)
                if a_scale is not None
                and a_scale.numel() > 0
                and a_scale.dtype
                not in (torch.uint8, torch.float16, torch.bfloat16, torch.float32)
                else a_scale
            )
            _ws = (
                w_scale.view(torch.uint8)
                if w_scale is not None
                and w_scale.numel() > 0
                and w_scale.dtype
                not in (torch.uint8, torch.float16, torch.bfloat16, torch.float32)
                else w_scale
            )
            exe(
                out,
                _a,
                _w,
                _as,
                _ws,
                sorted_ids,
                sorted_expert_ids,
                topk_weights,
                num_valid_ids,
                empty_bias,
                token_num,
                _n_in,
                _k_in,
                size_expert_ids_in,
                stream,
            )
        else:
            exe(
                out,
                a,
                w,
                a_scale,
                w_scale,
                sorted_ids,
                sorted_expert_ids,
                topk_weights,
                num_valid_ids,
                token_num,
                _n_in,
                _k_in,
                size_expert_ids_in,
            )

    return tensor_api


@functools.cache
def _get_compiled_stage2(
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    doweight: bool,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    accumulate: bool = True,
    persist_m: int = 1,
):
    """Compile and cache stage2 kernel, return a tensor_api closure."""
    exe = compile_flydsl_moe_stage2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=experts,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=doweight,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        accumulate=accumulate,
        persist_m=persist_m,
    )
    is_fp4 = b_dtype == "fp4"
    _n_in = model_dim
    _k_in = inter_dim

    _topk = topk

    def tensor_api(
        out: torch.Tensor,
        a: torch.Tensor,
        w: torch.Tensor,
        a_scale: torch.Tensor,
        w_scale: torch.Tensor,
        sorted_ids: torch.Tensor,
        sorted_expert_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        num_valid_ids: torch.Tensor,
        token_num: int,
        blocks: int,
    ) -> None:
        if accumulate:
            target = out
        else:
            target = torch.empty(
                (out.numel() * topk,),
                device=out.device,
                dtype=out.dtype,
            )

        if is_fp4:
            empty_bias = torch.empty(0, device=a.device, dtype=torch.float32)
            stream = torch.cuda.current_stream()
            # fp4/e8m0 dtypes are not supported by dlpack; cast to uint8
            _a = (
                a.view(torch.uint8)
                if a.dtype
                not in (torch.uint8, torch.float16, torch.bfloat16, torch.float32)
                else a
            )
            _w = (
                w.view(torch.uint8)
                if w.dtype
                not in (torch.uint8, torch.float16, torch.bfloat16, torch.float32)
                else w
            )
            _as = (
                a_scale.view(torch.uint8)
                if a_scale is not None
                and a_scale.numel() > 0
                and a_scale.dtype
                not in (torch.uint8, torch.float16, torch.bfloat16, torch.float32)
                else a_scale
            )
            _ws = (
                w_scale.view(torch.uint8)
                if w_scale is not None
                and w_scale.numel() > 0
                and w_scale.dtype
                not in (torch.uint8, torch.float16, torch.bfloat16, torch.float32)
                else w_scale
            )
            exe(
                target,
                _a,
                _w,
                _as,
                _ws,
                sorted_ids,
                sorted_expert_ids,
                topk_weights,
                num_valid_ids,
                empty_bias,
                token_num,
                _n_in,
                _k_in,
                blocks,
                stream,
            )
        else:
            exe(
                target,
                a,
                w,
                a_scale,
                w_scale,
                sorted_ids,
                sorted_expert_ids,
                topk_weights,
                num_valid_ids,
                token_num,
                _n_in,
                _k_in,
                blocks,
            )

        if not accumulate:
            torch.sum(
                target.view(*out.shape[:-1], _topk, out.shape[-1]),
                dim=-2,
                out=out,
            )

    return tensor_api


# Public API


def flydsl_moe_stage1(
    a: torch.Tensor,
    w1: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 256,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    act: str = "silu",
    w1_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
    persist_m: int = 0,
) -> torch.Tensor:
    """Fused gate+up GEMM (MOE stage1).

    a: (token_num, model_dim), w1: (E, 2*inter_dim, model_dim) pre-shuffled.
    For fp4 stage1, `w1`/`w1_scale` must use the same preshuffle layout as `shuffle_weight(..., (16, 16))`
    and `e8m0_shuffle(...)`.
    Returns (token_num, topk, inter_dim).
    """
    token_num = a.shape[0]
    E = w1.shape[0]
    inter_dim = w1.shape[1] // 2
    model_dim = a.shape[1]

    if a_dtype == "fp4":
        model_dim = model_dim * 2

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16

    if out is None:
        out = torch.empty(
            (token_num, topk, inter_dim), dtype=torch_out_dtype, device=a.device
        )

    dev = a.device
    flat_a_scale = (
        a1_scale.view(-1) if a1_scale is not None else torch.empty(0, device=dev)
    )
    flat_w_scale = (
        w1_scale.view(-1) if w1_scale is not None else torch.empty(0, device=dev)
    )
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )

    # Grid Y = number of M-blocks to launch.
    # Dense upper bound: min(tokens*topk*tile_m, len(sorted_ids)) // tile_m
    # Works when dense (token*topk >= sorted_blocks). But moe_sorting spreads
    # entries across per-expert blocks, so when sparse we need all blocks.
    _all_blks = sorted_expert_ids.shape[0]
    _dense_blks = min(token_num * topk * tile_m, sorted_token_ids.shape[0]) // tile_m
    _grid_y = min(_dense_blks, _all_blks) if _dense_blks >= _all_blks else _all_blks

    # persist_m controls how many M-blocks each WG processes sequentially.
    # For stage1, persist_m=1 is generally optimal: the large K dimension (model_dim)
    # means each WG is already latency-bound on VMEM, and reducing concurrent waves
    # (via persist_m>1) hurts latency hiding more than it helps L2 reuse.
    # persist_m>1 may help only when many M-blocks share the same expert (large batch
    # with few experts), enabling weight tile reuse across consecutive M-blocks.
    _persist_m = persist_m if persist_m > 0 else 1

    tensor_api = _get_compiled_stage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight=(sorted_weights is not None),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        act=act,
        persist_m=_persist_m,
    )

    tensor_api(
        out.view(-1),
        a.view(-1),
        w1.view(-1),
        flat_a_scale,
        flat_w_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sw,
        num_valid_ids,
        token_num,
        _grid_y,
    )

    return out


def flydsl_moe_stage2(
    inter_states: torch.Tensor,
    w2: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    topk: int = 1,
    *,
    tile_m: int = 32,
    tile_n: int = 128,
    tile_k: int = 256,
    a_dtype: str = "fp8",
    b_dtype: str = "fp4",
    out_dtype: str = "bf16",
    mode: str = "atomic",
    w2_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
    valid_m_blocks: Optional[int] = None,
) -> torch.Tensor:
    """Down-projection GEMM (MOE stage2). Supports atomic/reduce modes.

    a: (token_num, topk, inter_dim), w1: (E, model_dim, inter_dim) pre-shuffled.
    Returns (token_num, model_dim).

    valid_m_blocks: pre-computed from moe_sorting's num_valid_ids, trims
        per-expert tile_m padding without GPU sync at call time.
    """

    token_num = inter_states.shape[0]
    E = w2.shape[0]
    model_dim = w2.shape[1]
    inter_dim = inter_states.shape[2]

    accumulate = mode != "reduce"

    if a_dtype == "fp4":
        inter_dim = inter_dim * 2

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16
    if out is None:
        out = torch.empty(
            (token_num, model_dim), dtype=torch_out_dtype, device=inter_states.device
        )

    dev = inter_states.device
    flat_a_scale = (
        a2_scale.view(-1) if a2_scale is not None else torch.empty(0, device=dev)
    )
    flat_w_scale = (
        w2_scale.view(-1) if w2_scale is not None else torch.empty(0, device=dev)
    )
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.empty(sorted_token_ids.shape, dtype=torch.float32, device=dev)
    )

    # Use pre-computed valid_m_blocks from moe_sorting (trims per-expert padding).
    # Fallback: host-side upper bound (no GPU sync).
    if valid_m_blocks is None:
        padded_m_blocks = int(sorted_expert_ids.numel())
        valid_m_blocks = min(token_num * topk, padded_m_blocks)

    _persist_m = 4 if valid_m_blocks > 256 else 1

    tensor_api = _get_compiled_stage2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight=(sorted_weights is not None),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        accumulate=accumulate,
        persist_m=_persist_m,
    )
    tensor_api(
        out,
        inter_states,
        w2,
        flat_a_scale,
        flat_w_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sw,
        num_valid_ids,
        token_num,
        valid_m_blocks,
    )

    return out
