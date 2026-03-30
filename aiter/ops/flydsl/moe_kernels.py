# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE kernel management: naming, compilation, and high-level API."""

import functools
import re

from typing import Dict, Optional
from aiter.utility import dtypes

import torch


def _clear_stale_ir_context():
    """Clear stale MLIR IR context left over from a previous JIT compilation.

    When a @flyc.jit function's compilation crashes or a worker process
    reuses a context across tasks, ir.Context.current can remain non-None
    without a matching CompilationContext.  The JIT __call__ then takes
    its 'nested call' fast-path and the kernel body fails because
    CompilationContext.get_current() is None.

    Calling this before exe() ensures the normal compilation path is taken.
    """
    try:
        from flydsl._mlir import ir

        while ir.Context.current is not None:
            ir.Context.current.__exit__(None, None, None)
    except Exception:
        pass


_KERNEL_PARAMS: Dict[str, Dict] = {}

_SUFFIX_RE = re.compile(r"(?P<fq>_fq)?(?:_sbm(?P<sbm>\d+))?$")


def flydsl_kernel_name(
    stage: int,
    a_dtype: str,
    b_dtype: str,
    out_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    mode: str = "",
    sort_block_m: int = 0,
    fuse_fp4_quant: bool = False,
) -> str:
    """Construct kernel name: ``flydsl_moe{stage}_a{a}_w{b}_{out}_t{M}x{N}x{K}[_{mode}][_fq][_sbm{S}]``."""
    name = f"flydsl_moe{stage}_a{a_dtype}_w{b_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
    if mode:
        name += f"_{mode}"
    if fuse_fp4_quant:
        name += "_fq"
    if sort_block_m > 0 and sort_block_m != tile_m:
        name += f"_sbm{sort_block_m}"
    return name


def get_flydsl_kernel_params(name: str) -> Optional[Dict]:
    """Lookup kernel params by name. Strips ``_fq`` / ``_sbm{N}`` suffixes transparently."""
    params = _KERNEL_PARAMS.get(name)
    if params is not None:
        return params
    m = _SUFFIX_RE.search(name)
    if m and m.group(0):
        base_name = name[: m.start()]
        params = _KERNEL_PARAMS.get(base_name)
        if params is not None:
            extra: Dict = {}
            if m.group("fq"):
                extra["fuse_fp4_quant"] = True
            if m.group("sbm") is not None:
                extra["sort_block_m"] = int(m.group("sbm"))
            return {**params, **extra}
    return None


def get_flydsl_stage1_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported stage1 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"

    tile_ns = [32, 64, 128] if is_fp4 else [128]
    tile_ks = [256]
    tile_ms = [16, 32, 64, 128]
    waves_per_eus = [1, 2, 3, 4]
    k_batches = [1, 2, 4, 7, 14]
    b_nts = [0, 2]

    for tm in tile_ms:
        if tm in [16, 32]:
            tile_ns = [32, 64, 128]
        else:
            tile_ns = [64, 128]
        for tn in tile_ns:
            for tk in tile_ks:
                for wpe in waves_per_eus:
                    k_batches = [1]
                    for kb in k_batches:
                        gate_onlys = [False, True] if kb > 1 else [False]
                        for bnt in b_nts:
                            for go in gate_onlys:
                                name = flydsl_kernel_name(
                                    1, a_dtype, b_dtype, out_dtype, tm, tn, tk
                                )
                                if wpe != 1:
                                    name += f"_w{wpe}"
                                if kb != 1:
                                    name += f"_kb{kb}"
                                if bnt != 2:
                                    name += f"_bnt{bnt}"
                                if go:
                                    name += "_go"
                                kernels[name] = {
                                    "stage": 1,
                                    "a_dtype": a_dtype,
                                    "b_dtype": b_dtype,
                                    "out_dtype": out_dtype,
                                    "tile_m": tm,
                                    "tile_n": tn,
                                    "tile_k": tk,
                                    "MPerBlock": tm,
                                    "waves_per_eu": wpe,
                                    "k_batch": kb,
                                    "b_nt": bnt,
                                    "gate_only": go,
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
    tile_ms = [16, 32, 64, 128] if is_fp4 else [32, 64, 128]
    modes = ["atomic", "reduce"]

    for tm in tile_ms:
        for tn in tile_ns:
            for tk in tile_ks:
                for mode in modes:
                    base_name = flydsl_kernel_name(
                        2, a_dtype, b_dtype, out_dtype, tm, tn, tk, mode
                    )
                    base_params = {
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
                    kernels[base_name] = base_params
                    # Persistent variant: round-robin over M tiles, grid_y=cu_num.
                    kernels[base_name + "_persist"] = {
                        **base_params,
                        "persist": True,
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
    fuse_fp4_quant: bool = False,
    fuse_sort_scale: bool = False,
    use_async_copy: bool = False,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    gate_only: bool = False,
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
            fuse_fp4_quant=fuse_fp4_quant,
            fuse_sort_scale=fuse_sort_scale,
            use_async_copy=use_async_copy,
            k_batch=k_batch,
            waves_per_eu=waves_per_eu,
            b_nt=b_nt,
            gate_only=gate_only,
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
    sort_block_m: int = 0,
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
            persist_m=persist_m,
            sort_block_m=sort_block_m,
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
    fuse_fp4_quant: bool = False,
    fuse_sort_scale: bool = False,
    use_async_copy: bool = False,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 0,
    gate_only: bool = False,
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
        fuse_fp4_quant=fuse_fp4_quant,
        fuse_sort_scale=fuse_sort_scale,
        use_async_copy=use_async_copy,
        k_batch=k_batch,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        gate_only=gate_only,
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
        out_scale_sorted: Optional[torch.Tensor] = None,
    ) -> None:
        if gate_only:
            _gx = _n_in // tile_n
        else:
            _gx = _n_in // 2 // tile_n
        _gy = (size_expert_ids_in + persist_m - 1) // persist_m
        _total_wg = _gx * _gy * k_batch
        _clear_stale_ir_context()

        if is_fp4:
            empty_bias = torch.empty(0, device=a.device, dtype=torch.float32)
            empty_scale = torch.empty(0, device=a.device, dtype=torch.float32)
            stream = torch.cuda.current_stream()
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
            _oss = out_scale_sorted if out_scale_sorted is not None else empty_scale
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
                _oss,
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
def _get_compiled_silu_fq(inter_dim: int, topk: int):
    """Compile and cache the fused silu_and_mul + mxfp4 quant + scale-sort kernel."""
    _clear_stale_ir_context()
    from aiter.ops.flydsl.kernels.silu_and_mul_fq import build_silu_and_mul_fq_module

    return build_silu_and_mul_fq_module(inter_dim, topk)


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
    sort_block_m: int = 0,
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
        sort_block_m=sort_block_m,
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
                (token_num * topk * model_dim,),
                device=out.device,
                dtype=out.dtype,
            )

        _clear_stale_ir_context()

        if is_fp4:
            empty_bias = torch.empty(0, device=a.device, dtype=torch.float32)
            stream = torch.cuda.current_stream()
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
            torch.sum(target.view(token_num, _topk, model_dim), dim=1, out=out)

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
    fuse_fp4_quant: bool = False,
    fuse_sort_scale: bool = False,
    use_async_copy: bool = False,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    gate_only: bool = False,
):
    """Fused gate+up GEMM (MOE stage1).

    a: (token_num, model_dim), w1: (E, 2*inter_dim, model_dim) pre-shuffled.
    For fp4 stage1, `w1`/`w1_scale` must use the same preshuffle layout as
    `shuffle_weight(..., (16, 16))` and `e8m0_shuffle(...)`.

    When fuse_sort_scale=True, the kernel writes e8m0 scales in sorted tiled
    layout directly, avoiding a separate moe_mxfp4_sort call.

    When k_batch>1 (split-K), the kernel outputs gate/up partials via atomic
    add into a zeroed buffer, then silu_and_mul fuses activation + reduction.

    When gate_only=True (requires k_batch>1), each workgroup computes only
    one B-tile stream (no gate/up interleaving).  The grid X doubles so
    that by_n naturally covers both gate and up regions.

    Returns:
        Basic:                      out
        fuse_sort_scale:            (out, out_scale_sorted)
    """
    token_num = a.shape[0]
    E = w1.shape[0]
    inter_dim = w1.shape[1] // 2
    model_dim = a.shape[1]

    if a_dtype == "fp4":
        model_dim = model_dim * 2

    torch_out_dtype = dtypes.fp4x2 if fuse_fp4_quant else dtypes.bf16 if out_dtype == "bf16" else dtypes.fp16
    _is_splitk = k_batch > 1

    dev = a.device
    _splitk_fq = _is_splitk and fuse_fp4_quant

    if out is None:
        if _splitk_fq:
            fp4_bytes = token_num * topk * (inter_dim // 2)
            bf16_elems = (fp4_bytes + 1) // 2
            out = torch.empty(bf16_elems, dtype=torch_out_dtype, device=dev)
        else:
            out = torch.empty(
                (token_num, topk, inter_dim), dtype=torch_out_dtype, device=dev
            )

    if _is_splitk:
        torch_tmp_out_dtype = dtypes.bf16 if out_dtype == "bf16" else dtypes.fp16
        tmp_out = torch.zeros(
            (token_num, topk, inter_dim * 2), dtype=torch_tmp_out_dtype, device=dev
        )
    else:
        tmp_out = None

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

    _need_quant = fuse_fp4_quant or _splitk_fq
    _need_sort = _need_quant and (fuse_sort_scale or _splitk_fq)

    _sort_block_m = max(32, tile_m)
    _all_blks = sorted_expert_ids.shape[0]
    _dense_blks = (
        min(token_num * topk * _sort_block_m, sorted_token_ids.shape[0])
        // _sort_block_m
    )
    _grid_y = min(_dense_blks, _all_blks)

    _persist_m = persist_m if persist_m > 0 else 1

    # Allocate sorted-scale buffer with padding for tiled layout
    scale_cols = inter_dim // 32
    sorted_size = max(
        sorted_token_ids.shape[0], sorted_expert_ids.shape[0] * _sort_block_m
    )
    padded_rows = (sorted_size + 255) // 256 * 256
    padded_cols = (scale_cols + 7) // 8 * 8
    out_scale_sorted_flat = (
        torch.empty(padded_rows * padded_cols, dtype=torch.uint8, device=dev)
        if _need_sort
        else torch.empty(0, dtype=torch.uint8, device=dev)
    )

    # split-K GEMM kernel does not fuse quant; the fused silu_and_mul_fq kernel
    # handles activation + quant + scale-sort after the GEMM completes.
    _gemm_fq = fuse_fp4_quant and not _is_splitk
    _gemm_fss = fuse_sort_scale and not _is_splitk

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
        fuse_fp4_quant=_gemm_fq,
        fuse_sort_scale=_gemm_fss,
        use_async_copy=use_async_copy,
        k_batch=k_batch,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        gate_only=gate_only,
    )

    _kernel_out = tmp_out if _is_splitk else out
    tensor_api(
        _kernel_out.view(-1),
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
        out_scale_sorted=out_scale_sorted_flat.view(-1),
    )

    if _splitk_fq:
        _silu_fq = _get_compiled_silu_fq(inter_dim, topk)
        num_sorted_rows = sorted_token_ids.shape[0]
        token_num_t = torch.tensor([token_num], dtype=torch.int32, device=dev)
        _silu_fq(
            tmp_out.view(-1, inter_dim * 2),
            out.view(-1),
            out_scale_sorted_flat,
            sorted_token_ids,
            num_valid_ids,
            token_num_t,
            num_sorted_rows,
        )
    elif _is_splitk:
        from aiter.ops.activation import silu_and_mul

        silu_and_mul(out.view(-1, inter_dim), tmp_out.view(-1, inter_dim * 2))

    if fuse_fp4_quant:
        from aiter.utility.dtypes import fp8_e8m0

        out_scale_sorted = out_scale_sorted_flat.view(fp8_e8m0).view(
            padded_rows, padded_cols
        )
        return out, out_scale_sorted

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
    sort_block_m: int = 0,
    persist: Optional[bool] = None,
) -> torch.Tensor:
    """Down-projection GEMM (MOE stage2). Supports atomic/reduce modes.

    a: (token_num, topk, inter_dim), w1: (E, model_dim, inter_dim) pre-shuffled.
    Returns (token_num, model_dim).

    sort_block_m: block_size used by moe_sorting / stage1. When 0 (default),
        assumed equal to tile_m. When set, stage2 can use a different tile_m
        from sorting/stage1.
    persist: if True, use persistent round-robin mode (grid_y=cu_num);
        if False, use legacy persist_m mode; if None, auto-select.
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
        alloc_fn = torch.zeros if accumulate else torch.empty
        out = alloc_fn(
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

    _sbm = sort_block_m if sort_block_m > 0 else tile_m
    if _sbm == tile_m:
        m_blocks = min(sorted_expert_ids.shape[0], token_num * topk)
    else:
        total_sorted = sorted_expert_ids.shape[0] * _sbm
        m_blocks = (total_sorted + tile_m - 1) // tile_m
    if persist is True:
        _persist_m = -1
    elif persist is False:
        _persist_m = 4 if m_blocks > 256 else 1
    else:
        _persist_m = -1 if m_blocks > 256 else 1

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
        sort_block_m=sort_block_m,
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
        m_blocks,
    )

    return out
