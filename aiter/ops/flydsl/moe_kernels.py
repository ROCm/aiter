# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE kernel management: naming, compilation, and high-level API."""

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
    """Construct kernel name: flydsl_moe{stage}_a{a}_w{b}_{out}_t{M}x{N}x{K}[_{mode}]."""
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
    tile_ns = [256] if is_fp4 else [128]
    tile_ks = [256] if is_fp4 else [128]
    tile_ms = [16, 32, 64]

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
    tile_ks = [256] if is_fp4 else [128]
    tile_ms = [32, 64]
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
            use_cshuffle_epilog=(out_dtype == "fp8"),
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
            stream = torch.cuda.current_stream().cuda_stream
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
    )
    is_fp4 = b_dtype == "fp4"
    _n_in = model_dim
    _k_in = inter_dim

    reduce_exe = None
    if not accumulate:
        from .kernels.moe_gemm_2stage import compile_moe_reduction

        reduce_exe = compile_moe_reduction(
            topk=topk,
            model_dim=model_dim,
            dtype_str=out_dtype,
        )

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
                (token_num * topk * model_dim,),
                device=out.device,
                dtype=out.dtype,
            )

        if is_fp4:
            empty_bias = torch.empty(0, device=a.device, dtype=torch.float32)
            stream = torch.cuda.current_stream().cuda_stream
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
            stream = torch.cuda.current_stream().cuda_stream
            reduce_exe(
                target.view(token_num, topk, model_dim),
                out,
                token_num,
                stream,
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
    w1_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    sorted_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused gate+up GEMM (MOE stage1).

    a: (token_num, model_dim), w1: (E, 2*inter_dim, model_dim) pre-shuffled.
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
        sorted_expert_ids.shape[0],
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
) -> torch.Tensor:
    """Down-projection GEMM (MOE stage2). Supports atomic/reduce modes.

    a: (token_num, topk, inter_dim), w1: (E, model_dim, inter_dim) pre-shuffled.
    Returns (token_num, model_dim).
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
        out = torch.zeros(
            (token_num, model_dim), dtype=torch_out_dtype, device=inter_states.device
        )
    elif accumulate:
        out.zero_()

    dev = inter_states.device
    sw = (
        sorted_weights
        if sorted_weights is not None
        else torch.zeros(sorted_token_ids.shape, dtype=torch.float32, device=dev)
    )

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
    )
    tensor_api(
        out,
        inter_states,
        w2,
        a2_scale,
        w2_scale,
        sorted_token_ids,
        sorted_expert_ids,
        sw,
        num_valid_ids,
        token_num,
        int(sorted_expert_ids.numel()),
    )

    return out
