# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE kernel management: naming, compilation, and high-level API."""

import functools
import os
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
    # fp8xfp8 (non-mixed, same dtype for activation and weight)
    for out in ("bf16", "f16"):
        _KERNEL_PARAMS.update(get_flydsl_stage1_kernels("fp8", "fp8", out))
        _KERNEL_PARAMS.update(get_flydsl_stage2_kernels("fp8", "fp8", out))


_register_all_configs()


def _import_dsl2_kernel_module(module_name: str):
    """Import a kernel module from DSL2_ROOT/kernels/ (FlyDSL repo)."""
    import importlib
    import sys

    dsl2_root = os.environ.get("DSL2_ROOT", "")
    if not dsl2_root:
        raise ImportError(
            "DSL2_ROOT env var not set; required for non-fp4 FlyDSL kernels"
        )
    dsl2_kernels = os.path.join(dsl2_root, "kernels")
    for p in (dsl2_root, dsl2_kernels):
        if p not in sys.path:
            sys.path.insert(0, p)
    return importlib.import_module(module_name)


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
        mod = _import_dsl2_kernel_module("moe_gemm_2stage")
        return mod.compile_moe_gemm1(
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
            use_cshuffle_epilog=(out_dtype == "f16"),
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
        mod = _import_dsl2_kernel_module("moe_gemm_2stage")
        return mod.compile_moe_gemm2(
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


def _find_precompiled_hsaco_stage1(model_dim, inter_dim, experts, topk,
                                   tile_m, tile_n, tile_k, doweight,
                                   a_dtype, out_dtype):
    """Look for a precompiled stage1 .hsaco file."""
    from pathlib import Path
    from aiter.jit.utils.chip_info import get_gfx
    chip = get_gfx()
    dw = "dw1" if doweight else "dw0"
    name = (
        f"moe_gemm1_{a_dtype}_{a_dtype}_{out_dtype}_"
        f"{model_dim}x{inter_dim}_e{experts}_t{topk}_"
        f"tile{tile_m}x{tile_n}x{tile_k}_{dw}_{chip}"
    )
    cache_dir = Path.home() / ".flydsl" / "precompiled"
    hsaco_path = cache_dir / f"{name}.hsaco"
    meta_path = cache_dir / f"{name}.json"
    if hsaco_path.exists():
        return hsaco_path, meta_path
    return None, None


def _make_hip_launcher(hsaco_path, meta_path, kernel_name_default="kernel"):
    """Load a precompiled .hsaco and return (hip, func_ptr, shared_mem)."""
    import ctypes
    import json

    binary = hsaco_path.read_bytes()
    with open(meta_path) as f:
        meta = json.load(f)

    hip = ctypes.CDLL("libamdhip64.so")
    module = ctypes.c_void_p()
    buf = ctypes.create_string_buffer(binary)
    status = hip.hipModuleLoadData(ctypes.byref(module), buf)
    if status != 0:
        raise RuntimeError(f"hipModuleLoadData failed: {status}")

    func_ptr = ctypes.c_void_p()
    kname = meta.get("kernel_name", kernel_name_default).encode()
    status = hip.hipModuleGetFunction(ctypes.byref(func_ptr), module, kname)
    if status != 0:
        raise RuntimeError(f"hipModuleGetFunction failed: {status}")

    shared_mem = meta.get("shared_mem", 8192)
    return hip, func_ptr, shared_mem


def _hip_launch(hip, func_ptr, c_args, gx, gy, shared_mem):
    """Launch a HIP kernel with persistent argument buffer (CUDA graph safe)."""
    import ctypes
    n = len(c_args)
    arg_ptrs = (ctypes.c_void_p * n)()
    for i, a_val in enumerate(c_args):
        arg_ptrs[i] = ctypes.cast(ctypes.pointer(a_val), ctypes.c_void_p)

    stream = torch.cuda.current_stream().cuda_stream
    status = hip.hipModuleLaunchKernel(
        func_ptr,
        ctypes.c_uint(gx), ctypes.c_uint(gy), ctypes.c_uint(1),
        ctypes.c_uint(256), ctypes.c_uint(1), ctypes.c_uint(1),
        ctypes.c_uint(shared_mem),
        ctypes.c_void_p(stream),
        arg_ptrs,
        ctypes.c_void_p(0),
    )
    if status != 0:
        raise RuntimeError(f"hipModuleLaunchKernel failed: {status}")


def _make_hip_tensor_api_stage1(hsaco_path, meta_path, model_dim, inter_dim,
                                topk, tile_n):
    """Create a tensor_api closure for stage1 using HIP to launch precompiled kernel."""
    import ctypes

    hip, func_ptr, shared_mem = _make_hip_launcher(hsaco_path, meta_path, "moe_gemm1_0")
    n_out = inter_dim
    gx = n_out // tile_n

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
        c_args = [
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(w.data_ptr()),
            ctypes.c_void_p(a_scale.data_ptr() if a_scale.numel() > 0 else 0),
            ctypes.c_void_p(w_scale.data_ptr() if w_scale.numel() > 0 else 0),
            ctypes.c_void_p(sorted_ids.data_ptr()),
            ctypes.c_void_p(sorted_expert_ids.data_ptr()),
            ctypes.c_void_p(topk_weights.data_ptr()),
            ctypes.c_void_p(num_valid_ids.data_ptr()),
            ctypes.c_int32(int(token_num)),
            ctypes.c_int32(int(n_out)),
            ctypes.c_int32(int(model_dim)),
            ctypes.c_int32(int(size_expert_ids_in)),
        ]
        _hip_launch(hip, func_ptr, c_args, gx, size_expert_ids_in, shared_mem)

    return tensor_api


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
                token_num,
                _n_in,
                _k_in,
                size_expert_ids_in,
                stream,
            )

    return tensor_api


def _find_precompiled_hsaco(model_dim, inter_dim, experts, topk,
                            tile_m, tile_n, tile_k, doweight,
                            a_dtype, out_dtype, accumulate):
    """Look for a precompiled .hsaco file in ~/.flydsl/precompiled/."""
    from pathlib import Path
    from aiter.jit.utils.chip_info import get_gfx
    chip = get_gfx()
    acc = "acc1" if accumulate else "acc0"
    dw = "dw1" if doweight else "dw0"
    name = (
        f"moe_gemm2_{a_dtype}_{a_dtype}_{out_dtype}_"
        f"{model_dim}x{inter_dim}_e{experts}_t{topk}_"
        f"tile{tile_m}x{tile_n}x{tile_k}_{dw}_{acc}_{chip}"
    )
    cache_dir = Path.home() / ".flydsl" / "precompiled"
    hsaco_path = cache_dir / f"{name}.hsaco"
    meta_path = cache_dir / f"{name}.json"
    if hsaco_path.exists():
        return hsaco_path, meta_path
    return None, None


def _make_hip_tensor_api(hsaco_path, meta_path, model_dim, inter_dim,
                         topk, tile_n, accumulate):
    """Create a tensor_api closure that uses HIP to launch a precompiled kernel."""
    import ctypes

    hip, func_ptr, shared_mem = _make_hip_launcher(hsaco_path, meta_path, "moe_gemm2_0")
    gx = model_dim // tile_n

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

        c_args = [
            ctypes.c_void_p(target.data_ptr()),
            ctypes.c_void_p(a.data_ptr()),
            ctypes.c_void_p(w.data_ptr()),
            ctypes.c_void_p(a_scale.data_ptr() if a_scale is not None else 0),
            ctypes.c_void_p(w_scale.data_ptr() if w_scale is not None else 0),
            ctypes.c_void_p(sorted_ids.data_ptr()),
            ctypes.c_void_p(sorted_expert_ids.data_ptr()),
            ctypes.c_void_p(topk_weights.data_ptr()),
            ctypes.c_void_p(num_valid_ids.data_ptr()),
            ctypes.c_int32(int(token_num)),
            ctypes.c_int32(int(model_dim)),
            ctypes.c_int32(int(inter_dim)),
            ctypes.c_int32(int(blocks)),
        ]
        _hip_launch(hip, func_ptr, c_args, gx, blocks, shared_mem)

        if not accumulate:
            target_view = target.view(token_num, topk, model_dim)
            out.copy_(target_view.sum(dim=1))

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
    # For fp8→bf16: use f32 output + accumulate to avoid bf16 cshuffle bugs
    actual_out_dtype = out_dtype
    actual_accumulate = accumulate
    if b_dtype == "fp8" and out_dtype == "bf16":
        actual_out_dtype = "f32"
        actual_accumulate = True
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
        out_dtype=actual_out_dtype,
        accumulate=actual_accumulate,
    )
    is_fp4 = b_dtype == "fp4"
    _n_in = model_dim
    _k_in = inter_dim

    _use_f32 = (b_dtype == "fp8" and out_dtype == "bf16")

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
        target_dtype = torch.float32 if _use_f32 else out.dtype
        if actual_accumulate:
            if _use_f32:
                target = torch.zeros(out.shape, device=out.device, dtype=torch.float32)
            else:
                target = out
        else:
            target = torch.empty(
                (token_num * topk * model_dim,),
                device=out.device,
                dtype=target_dtype,
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
                token_num,
                _n_in,
                _k_in,
                blocks,
                stream,
            )

        if not actual_accumulate:
            target_view = target.view(token_num, topk, model_dim)
            out.copy_(target_view.sum(dim=1).to(out.dtype))
        elif _use_f32:
            out.copy_(target.to(out.dtype))

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
