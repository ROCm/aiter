# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE kernel management: naming, compilation, and high-level API."""

import functools
import os
import re
from typing import Dict, Optional

import torch

from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg

_KERNEL_PARAMS: Dict[str, Dict] = {}


def _get_dtypes():
    from aiter.utility import dtypes

    return dtypes


_SUFFIX_RE = re.compile(
    r"(?:_kw(?P<kw>\d+))?(?P<fp4>_fp4)?(?P<fp8>_fp8)?(?:_sbm(?P<sbm>\d+))?$"
)


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
) -> str:
    """Construct kernel name: ``flydsl_moe{stage}_a{a}_w{b}_{out}_t{M}x{N}x{K}[_{mode}][_sbm{S}]``."""
    name = f"flydsl_moe{stage}_a{a_dtype}_w{b_dtype}_{out_dtype}_t{tile_m}x{tile_n}x{tile_k}"
    if mode:
        name += f"_{mode}"
    if sort_block_m > 0 and sort_block_m != tile_m:
        name += f"_sbm{sort_block_m}"
    return name


def get_flydsl_kernel_params(name: str) -> Optional[Dict]:
    """Lookup kernel params by name.

    Strips ``_kw{N}`` / ``_fp4`` / ``_fp8`` / ``_sbm{N}`` suffixes transparently.
    """
    params = _KERNEL_PARAMS.get(name)
    if params is not None:
        return params
    m = _SUFFIX_RE.search(name)
    if m and m.group(0):
        base_name = name[: m.start()]
        params = _KERNEL_PARAMS.get(base_name)
        if params is not None:
            extra: Dict = {}
            if m.group("kw") is not None:
                extra["k_wave"] = int(m.group("kw"))
            if m.group("fp4"):
                extra["out_dtype"] = "fp4"
            if m.group("fp8"):
                extra["out_dtype"] = "fp8"
            if m.group("sbm") is not None:
                extra["sort_block_m"] = int(m.group("sbm"))
            return {**params, **extra}
    return None


def get_flydsl_stage1_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported stage1 configs."""
    kernels = {}
    is_fp4_a = a_dtype == "fp4"
    is_fp4_b = b_dtype == "fp4"

    tile_ns = [32, 64, 128] if is_fp4_b else [128]
    tile_ks = [256]
    tile_ms = [32, 64, 128]

    waves_per_eus = [1, 2, 3, 4]
    k_batches = [1, 2, 4, 7, 14]
    b_nts = [0, 2]
    xcd_swizzles = [0, 4]

    for tm in tile_ms:
        if tm == 32:
            tile_ns = [32, 64, 128]
        else:
            tile_ns = [64, 128] if is_fp4_a else [128, 256]
        for tn in tile_ns:
            for tk in tile_ks:
                for wpe in waves_per_eus:
                    for kb in k_batches if wpe == 3 and tm == 32 and is_fp4_a else [1]:
                        for bnt in b_nts:
                            gate_onlys = (
                                [False, True] if kb > 1 and is_fp4_a else [False]
                            )
                            for go in gate_onlys:
                                for xcd in xcd_swizzles:
                                    base = flydsl_kernel_name(
                                        1, a_dtype, b_dtype, out_dtype, tm, tn, tk
                                    )
                                    if wpe != 1:
                                        base += f"_w{wpe}"
                                    if kb != 1:
                                        base += f"_kb{kb}"
                                    if bnt != 2:
                                        base += f"_bnt{bnt}"
                                    if go:
                                        base += "_go"
                                    if a_dtype == "fp8":
                                        base += "_gui"
                                    if xcd > 0:
                                        base += f"_xcd{xcd}"
                                    # k_wave (intra-block K-slice): only for the
                                    # small-M tile (tile_m==32), no split-K/mock,
                                    # and capped to <=8 total waves (<=512 threads).
                                    num_n_waves = min(4, tn // 32)
                                    k_waves = (
                                        [1, 2, 4]
                                        if (tm == 32 and kb == 1 and not go)
                                        else [1]
                                    )
                                    for kw in k_waves:
                                        if num_n_waves * kw > 8:
                                            continue
                                        if kw > 1 and 4 * tn > tk:
                                            continue
                                        if (
                                            kw > 1
                                            and a_dtype == "fp8"
                                            and num_n_waves < 2
                                        ):
                                            continue
                                        name = base + (f"_kw{kw}" if kw > 1 else "")
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
                                            "gate_mode": (
                                                "mock_gate_only"
                                                if go
                                                else (
                                                    "interleave"
                                                    if a_dtype == "fp8"
                                                    else "separated"
                                                )
                                            ),
                                            "xcd_swizzle": xcd,
                                            "k_wave": kw,
                                        }
    return kernels


def get_flydsl_stage2_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported stage2 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"
    is_fp8 = b_dtype == "fp8"
    tile_ns = [128, 256] if is_fp4 else [128]
    # fp4 stage2 supports tile_k=128 (pack_K=1 scale sub-group shift path) as
    # well as 256.  tile_k=128 cleanly tiles K=inter_dim for TP-sharded shapes
    # whose inter_dim is a multiple of 128 but not 256 (e.g. MiniMax TP4=384).
    tile_ks = [128, 256] if (is_fp4 or is_fp8) else [128]
    tile_ms = [16, 32, 64, 128] if is_fp4 else [32, 64, 128]
    modes = ["atomic", "reduce"]

    b_nts = [0, 2]

    xcd_swizzles = [0, 4]

    for tm in tile_ms:
        for tn in tile_ns:
            for tk in tile_ks:
                for mode in modes:
                    for bnt in b_nts:
                        for xcd in xcd_swizzles:
                            base_name = flydsl_kernel_name(
                                2, a_dtype, b_dtype, out_dtype, tm, tn, tk, mode
                            )
                            if bnt != 0:
                                base_name += f"_bnt{bnt}"
                            if xcd > 0:
                                base_name += f"_xcd{xcd}"
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
                                "b_nt": bnt,
                                "xcd_swizzle": xcd,
                            }
                            kernels[base_name] = base_params
                            kernels[base_name + "_persist"] = {
                                **base_params,
                                "persist": True,
                            }
    _register_production_variants_stage2(kernels, a_dtype, b_dtype, out_dtype)
    return kernels


def _register_production_variants_stage2(
    kernels: Dict[str, Dict], a_dtype: str, b_dtype: str, out_dtype: str
) -> None:
    """Append hand-tuned stage2 variants to ``kernels`` in-place."""
    # (a, b, out, tile_m, tile_n, tile_k, mode, suffix, overrides)
    PRODUCTION_VARIANTS = (
        (
            "fp4",
            "fp4",
            "bf16",
            64,
            128,
            256,
            "atomic",
            "_persist_async_w4_cumul3",
            {
                "persist": True,
                "use_async_copy": True,
                "waves_per_eu": 4,
                "cu_num_mul": 3,
            },
        ),
    )
    for pa, pb, pout, ptm, ptn, ptk, pmode, psuffix, povr in PRODUCTION_VARIANTS:
        if (pa, pb, pout) != (a_dtype, b_dtype, out_dtype):
            continue
        _base = flydsl_kernel_name(2, pa, pb, pout, ptm, ptn, ptk, pmode)
        if _base not in kernels:
            continue
        kernels[_base + psuffix] = {**kernels[_base], **povr}


def get_flydsl_stage1_kernels_int4_bf16(out_dtype: str) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported int4_bf16 stage1 configs."""
    kernels = {}
    a_dtype = "bf16"
    b_dtype = "int4"
    tile_ks = [128, 256]
    tile_ms = [16, 32, 64, 128]
    tile_ns = [64, 128]
    k_batches = [1, 2, 4, 7, 14]

    for tm in tile_ms:
        for tn in tile_ns:
            for tk in tile_ks:
                for kb in k_batches:
                    name = flydsl_kernel_name(
                        1, a_dtype, b_dtype, out_dtype, tm, tn, tk
                    )
                    if kb != 1:
                        name += f"_kb{kb}"
                    kernels[name] = {
                        "stage": 1,
                        "a_dtype": a_dtype,
                        "b_dtype": b_dtype,
                        "out_dtype": out_dtype,
                        "tile_m": tm,
                        "tile_n": tn,
                        "tile_k": tk,
                        "MPerBlock": tm,
                        "in_dtype": "int4_bf16",
                        "k_batch": kb,
                    }
    return kernels


def get_flydsl_stage2_kernels_int4_bf16(out_dtype: str) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported int4_bf16 stage2 configs."""
    kernels = {}
    a_dtype = "bf16"
    b_dtype = "int4"
    tile_ks = [128, 256]
    tile_ms = [16, 32, 64, 128]
    tile_ns = [128]
    # modes = ["atomic", "reduce"]
    modes = ["atomic"]

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
                        "in_dtype": "int4_bf16",
                    }
                    kernels[base_name] = base_params
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
    # mxfp8 (a8w8): fp8 activation + fp8 weight, per-1x32 e8m0 microscale.
    for out in ("bf16", "f16"):
        _KERNEL_PARAMS.update(get_flydsl_stage1_kernels("fp8", "fp8", out))
        _KERNEL_PARAMS.update(get_flydsl_stage2_kernels("fp8", "fp8", out))
    # int4_bf16 (a16wi4) configs
    for out in ("bf16", "f16"):
        _KERNEL_PARAMS.update(get_flydsl_stage1_kernels_int4_bf16(out))
        _KERNEL_PARAMS.update(get_flydsl_stage2_kernels_int4_bf16(out))


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
    use_async_copy: bool = False,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    gate_mode: str = "separated",
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    enable_bias: bool = False,
    a_scale_one: bool = False,
    xcd_swizzle: int = 0,
    k_wave: int = 1,
):
    """Compile stage1 kernel (cached via underlying lru_cache)."""
    if b_dtype in ("fp4", "fp8"):
        from .kernels.mixed_moe_gemm_2stage import compile_mixed_moe_gemm1
        from .moe_common import GateMode

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
            use_async_copy=use_async_copy,
            k_batch=k_batch,
            waves_per_eu=waves_per_eu,
            b_nt=b_nt,
            gate_mode=GateMode(gate_mode),
            model_dim_pad=model_dim_pad,
            inter_dim_pad=inter_dim_pad,
            enable_bias=enable_bias,
            a_scale_one=a_scale_one,
            xcd_swizzle=xcd_swizzle,
            k_wave=k_wave,
        )
    elif a_dtype == "bf16" and b_dtype == "int4":
        # a16wi4: bf16 activations, int4 weights with groupwise scale
        from .kernels.moe_gemm_2stage import compile_moe_gemm1

        # split-K needs cshuffle (None -> auto-enable); non-split-K uses direct epilog
        _use_cshuffle = None if k_batch > 1 else False

        return compile_moe_gemm1(
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            tile_m=tile_m,
            tile_n=tile_n,
            tile_k=tile_k,
            doweight_stage1=doweight_stage1,
            in_dtype="int4_bf16",
            group_size=32,
            out_dtype=out_dtype,
            use_cshuffle_epilog=_use_cshuffle,
            scale_is_bf16=True,
            k_batch=k_batch,
        )
    else:
        raise ValueError(
            f"Unsupported stage1 dtype combination: a_dtype={a_dtype}, b_dtype={b_dtype}"
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
    waves_per_eu: Optional[int] = None,
    use_async_copy: bool = False,
    cu_num_mul: int = 1,
    b_nt: int = 0,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    xcd_swizzle: int = 0,
    enable_bias: bool = False,
):
    """Compile stage2 kernel (cached via underlying lru_cache)."""
    if b_dtype in ("fp4", "fp8"):
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
            waves_per_eu=waves_per_eu,
            use_async_copy=use_async_copy,
            cu_num_mul=cu_num_mul,
            # API parity (reviewer #3): forward `b_nt` and `xcd_swizzle`
            # from the kernel-name parser. They are accepted as ignored
            # kwargs on the fp4xfp4 path so callers parsing the
            # `_bnt{N}` / `_xcd{N}` registry suffixes don't need
            # per-dtype special cases.
            b_nt=b_nt,
            xcd_swizzle=xcd_swizzle,
            model_dim_pad=model_dim_pad,
            inter_dim_pad=inter_dim_pad,
            enable_bias=enable_bias,
        )
    elif a_dtype == "bf16" and b_dtype == "int4":
        # a16wi4: bf16 activations, int4 weights with groupwise scale
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
            in_dtype="int4_bf16",
            group_size=32,
            out_dtype=out_dtype,
            accumulate=accumulate,
            scale_is_bf16=True,
        )
    else:
        raise ValueError(
            f"Unsupported stage2 dtype combination: a_dtype={a_dtype}, b_dtype={b_dtype}"
        )


# Private helpers


_DLPACK_SAFE = (torch.uint8, torch.float16, torch.bfloat16, torch.float32)


def _view_safe(t: torch.Tensor) -> torch.Tensor:
    """View as uint8 if dtype is not dlpack-safe, otherwise return as-is."""
    return (
        t.view(torch.uint8)
        if t is not None and t.numel() > 0 and t.dtype not in _DLPACK_SAFE
        else t
    )


def runtime_swiglu_limit(swiglu_limit: Optional[float], act: str) -> float:
    """Normalize swiglu_limit into the runtime f32 clamp bound passed to kernels.

    The kernels always clamp using this value, so "no clamp" is encoded as +inf:
      - swiglu: defaults to 7.0 when unset (matches the reference ``swiglu()``).
      - silu:   clamps only when a positive limit is configured, else +inf
                (matches the reference's ``if swiglu_limit:`` truthiness).
    """
    if act == "swiglu":
        return float(swiglu_limit) if swiglu_limit else 7.0
    return float(swiglu_limit) if swiglu_limit else float("inf")


def _s1_args_fp4(
    out,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    out_scale_sorted,
    token_num,
    n_in,
    k_in,
    size_expert_ids_in,
    dev,
    bias=None,
    stream=None,
    swiglu_limit=float("inf"),
):
    empty_f32 = torch.empty(0, device=dev, dtype=torch.float32)
    _bias = bias if bias is not None else empty_f32
    if stream is None:
        stream = torch.cuda.current_stream()
    return (
        ptr_arg(out),
        ptr_arg(a),
        ptr_arg(w),
        ptr_arg(a_scale),
        ptr_arg(w_scale),
        ptr_arg(sorted_ids),
        ptr_arg(sorted_expert_ids),
        ptr_arg(sorted_weights),
        ptr_arg(num_valid_ids),
        ptr_arg(_bias),
        ptr_arg(out_scale_sorted),
        token_num,
        n_in,
        k_in,
        size_expert_ids_in,
        float(swiglu_limit),
        stream,
    )


def _s1_args_std(
    out,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    size_expert_ids_in,
    stream=None,
):
    if stream is None:
        stream = torch.cuda.current_stream()
    return (
        ptr_arg(out),
        ptr_arg(a),
        ptr_arg(w),
        ptr_arg(a_scale),
        ptr_arg(w_scale),
        ptr_arg(sorted_ids),
        ptr_arg(sorted_expert_ids),
        ptr_arg(sorted_weights),
        ptr_arg(num_valid_ids),
        token_num,
        n_in,
        k_in,
        size_expert_ids_in,
        stream,
    )


def _s2_args_fp4(
    target,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    blocks,
    dev,
    bias=None,
    stream=None,
):
    _bias = (
        bias.view(-1)
        if bias is not None
        else torch.empty(0, device=dev, dtype=torch.float32)
    )
    if stream is None:
        stream = torch.cuda.current_stream()
    return (
        ptr_arg(target),
        ptr_arg(a),
        ptr_arg(w),
        ptr_arg(a_scale),
        ptr_arg(w_scale),
        ptr_arg(sorted_ids),
        ptr_arg(sorted_expert_ids),
        ptr_arg(sorted_weights),
        ptr_arg(num_valid_ids),
        ptr_arg(_bias),
        token_num,
        n_in,
        k_in,
        blocks,
        stream,
    )


def _s2_args_std(
    target,
    a,
    w,
    a_scale,
    w_scale,
    sorted_ids,
    sorted_expert_ids,
    sorted_weights,
    num_valid_ids,
    token_num,
    n_in,
    k_in,
    blocks,
    stream=None,
):
    if stream is None:
        stream = torch.cuda.current_stream()
    return (
        ptr_arg(target),
        ptr_arg(a),
        ptr_arg(w),
        ptr_arg(a_scale),
        ptr_arg(w_scale),
        ptr_arg(sorted_ids),
        ptr_arg(sorted_expert_ids),
        ptr_arg(sorted_weights),
        ptr_arg(num_valid_ids),
        token_num,
        n_in,
        k_in,
        blocks,
        stream,
    )


def _run_compiled(exe, args):
    """JIT-compile on first call, then dispatch via cached CompiledFunction."""
    import flydsl.compiler as flyc

    cf = getattr(exe, "_cf", None)
    if cf is not None:
        cf(*args)
        return
    try:
        cf = flyc.compile(exe, *args)
        exe._cf = cf
    except Exception:
        # JitFunction.__call__ leaks ir.Context on compilation failure,
        # causing all subsequent JitFunction calls to take a wrong code path
        # (self.func(*args) without CompilationContext → gpu_module_body error).
        # Clean up leaked contexts to isolate failures.
        try:
            from flydsl._mlir import ir

            while ir.Context.current is not None:
                ir.Context.current.__exit__(None, None, None)
        except Exception:
            pass
        raise


def _run_moe_reduction(
    target,
    out,
    token_num,
    topk,
    model_dim,
    expert_mask=None,
    topk_ids=None,
    stream=None,
):
    """Topk reduction epilogue for stage2 reduce mode."""
    use_mask = expert_mask is not None
    if use_mask and topk_ids is None:
        raise ValueError(
            "topk_ids is required when expert_mask is provided for reduce mode"
        )
    # Map torch dtype -> compile_moe_reduction dtype_str
    if out.dtype == torch.float16:
        _reduce_dtype_str = "f16"
    elif out.dtype == torch.bfloat16:
        _reduce_dtype_str = "bf16"
    elif out.dtype == torch.float32:
        _reduce_dtype_str = "f32"
    else:
        _reduce_dtype_str = None

    if _reduce_dtype_str is None:
        # Unsupported dtype for the masked kernel — fall back to torch.sum.
        # This drops the EP mask, so only valid for non-EP runs.
        if use_mask:
            raise NotImplementedError(
                f"Masked moe reduction not supported for dtype {out.dtype}"
            )
        torch.sum(target.view(token_num, topk, model_dim), dim=1, out=out)
        return

    from .kernels.moe_gemm_2stage import compile_moe_reduction

    reduce_exe = compile_moe_reduction(
        topk=topk,
        model_dim=model_dim,
        dtype_str=_reduce_dtype_str,
        use_mask=use_mask,
        # expert_mask is sized by global expert count (≠ w2.shape[0] under EP).
        num_experts=int(expert_mask.numel()) if use_mask else 0,
    )
    X = target.view(token_num, topk, model_dim)
    if use_mask:
        em = expert_mask.to(torch.int32).contiguous()
        tk = topk_ids.to(torch.int32).contiguous()
    else:
        # Placeholders; kernel ignores them when use_mask=False.
        em = torch.empty(0, device=out.device, dtype=torch.int32)
        tk = torch.empty(0, device=out.device, dtype=torch.int32)
    if stream is None:
        stream = torch.cuda.current_stream()
    _run_compiled(
        reduce_exe,
        (
            ptr_arg(X),
            ptr_arg(out),
            ptr_arg(em),
            ptr_arg(tk),
            token_num,
            stream,
        ),
    )


# ---------------------------------------------------------------------------
# gfx1250 MXScale shape-alignment helpers
#
# The FlyDSL mxscale MoE kernels hard-require K (the GEMM contraction dim,
# stage1: model_dim, stage2: inter_dim) be divisible by tile_k (itself a
# multiple of WMMA_K=128), and tile_n to divide N (stage1: 2*inter_dim with
# the stage1 wrapper also requiring inter_dim % tile_n == 0; stage2:
# model_dim). Model shapes like GPT-OSS (2880) break both constraints with
# default tile_n=128 / tile_k=128.
#
# The helpers below let the gfx1250 stage1/stage2 wrappers (a) pick the
# largest legal tile_n that divides the required N dims, and (b) zero-pad
# activations, weights and scales on the K dim to the next multiple of
# tile_k. Zero padding is algebraically safe for mx-quantized GEMM (the
# extra K-slice contributes 0·anything = 0), and is cheap relative to the
# kernel cost (~2% for 2944 vs 2880).
# ---------------------------------------------------------------------------

_MXSCALE_FORMAT_PACK = {
    # in_dtype: (pack_a, pack_b, weight_is_preshuffled)
    "fp4": (2, 2, False),
    "fp8": (1, 1, True),
    "a8w4": (1, 2, True),
}


# Cache padded weight / scale tensors keyed on storage pointer so that
# repeated fused_moe calls with the same W / W_scale don't re-pad +
# re-memcpy ~100MB per invocation. This is the dominant cost for shapes
# whose model_dim is not natively tile_k-aligned (e.g. GPT-OSS 2880 ->
# padded to 2944).
#
# Key:   (data_ptr, numel, element_size, delta_bytes, pad_value, preshuffled)
# Value: padded tensor (strong ref keeps the entry alive).
# Policy: FIFO eviction bounded by _MXSCALE_PAD_CACHE_MAX_BYTES total VRAM
# occupancy (default 512MB) to avoid OOM'ing on multi-GB weight tensors.
# Disable via AITER_GFX1250_DISABLE_PAD_CACHE=1 if memory-constrained.
_MXSCALE_PAD_CACHE: dict = {}
_MXSCALE_PAD_CACHE_BYTES: int = 0
_MXSCALE_PAD_CACHE_MAX_BYTES: int = int(
    os.environ.get("AITER_GFX1250_PAD_CACHE_MAX_BYTES", str(512 * 1024 * 1024))
)
_MXSCALE_PAD_CACHE_ENABLED: bool = not bool(
    int(os.environ.get("AITER_GFX1250_DISABLE_PAD_CACHE", "0"))
)


def _mxscale_pad_cache_key(t: torch.Tensor, delta: int, value: int, preshuffled: bool):
    return (
        int(t.data_ptr()),
        int(t.numel()),
        int(t.element_size()),
        int(delta),
        int(value),
        bool(preshuffled),
    )


def _mxscale_pad_cache_get(key):
    if not _MXSCALE_PAD_CACHE_ENABLED:
        return None
    return _MXSCALE_PAD_CACHE.get(key)


def _mxscale_pad_cache_put(key, value):
    global _MXSCALE_PAD_CACHE_BYTES
    if not _MXSCALE_PAD_CACHE_ENABLED:
        return
    # nbytes of the padded tensor we would cache
    nbytes = int(value.numel()) * int(value.element_size())
    if nbytes > _MXSCALE_PAD_CACHE_MAX_BYTES:
        # Too big to cache without blowing the budget; skip entirely.
        return
    # Evict oldest entries (FIFO) until the new one fits within the byte budget.
    while (
        _MXSCALE_PAD_CACHE_BYTES + nbytes
    ) > _MXSCALE_PAD_CACHE_MAX_BYTES and _MXSCALE_PAD_CACHE:
        oldest_key = next(iter(_MXSCALE_PAD_CACHE))
        evicted = _MXSCALE_PAD_CACHE.pop(oldest_key)
        _MXSCALE_PAD_CACHE_BYTES -= int(evicted.numel()) * int(evicted.element_size())
    _MXSCALE_PAD_CACHE[key] = value
    _MXSCALE_PAD_CACHE_BYTES += nbytes


def _mxscale_align_up(x: int, align: int) -> int:
    return ((int(x) + int(align) - 1) // int(align)) * int(align)


def _mxscale_pick_tile_n(
    default_tile_n: int, *required_divisors: int, in_dtype: str = "fp8", align: int = 16
) -> int:
    """Largest tile_n <= default_tile_n that divides every N dim in
    ``required_divisors`` and is a multiple of ``align`` (bumped to 32 for
    fp4, which uses WMMA_N_EFF=32).

    Matches FlyDSL's own ``bench_resolve_tiles`` heuristic (largest multiple
    of align that divides the N dim). The downstream launch-shape picker
    (`_pick_fp16_single_launch_shape`) will adapt m_warp/n_warp to whatever
    tile_n we pick, falling back to degenerate shapes such as n_warp=1 when
    needed (e.g. tile_n=240 for GPT-OSS fp8).
    """
    if in_dtype == "fp4":
        align = max(align, 32)
    tn = int(default_tile_n)
    while tn >= align:
        if all((int(d) % tn) == 0 for d in required_divisors):
            return tn
        tn -= align
    return align


def _mxscale_zero_pad_last(
    t: torch.Tensor, delta: int, value: int = 0, cache: bool = False
) -> torch.Tensor:
    """Append ``delta`` elements of ``value`` along the last dim (default 0).

    ``torch.nn.functional.pad`` does not implement some 1-byte float dtypes
    (e.g. Float8_e8m0fnu / Float8_e4m3fn / Float4_e2m1fn_x2); operate through
    a uint8 view in that case, then restore the original dtype.

    ``value`` is interpreted as the raw byte/element value (e.g. 0x7F for
    E8M0 = 1.0, 0x00 for E8M0 = 2^-127 / fp8 zero).

    When ``cache=True`` (typical for static weight/scale tensors), the result
    is memoized by the input's storage pointer so repeated calls with the
    same tensor avoid redoing the ~100MB memcpy.
    """
    if int(delta) <= 0:
        return t
    if cache:
        key = _mxscale_pad_cache_key(t, int(delta), int(value), False)
        cached = _mxscale_pad_cache_get(key)
        if cached is not None:
            return cached
    if t.element_size() == 1 and t.dtype not in (torch.uint8, torch.int8):
        orig_dtype = t.dtype
        u8 = t.contiguous().view(torch.uint8)
        padded = torch.nn.functional.pad(u8, (0, int(delta)), value=int(value))
        padded = padded.view(orig_dtype)
    else:
        padded = torch.nn.functional.pad(t.contiguous(), (0, int(delta)), value=value)
    if cache:
        _mxscale_pad_cache_put(key, padded)
    return padded


def _mxscale_pad_weight_k(
    w: torch.Tensor, delta_bytes: int, weight_is_preshuffled: bool, cache: bool = True
) -> torch.Tensor:
    """Zero-pad a weight tensor of shape ``(E, N, K/pack_b)`` on the K-byte
    (last) dim.

    When the caller has already preshuffled the weight
    (fp8 / a8w4 path), a raw ``F.pad`` on the last dim would insert zero
    bytes *inside* each 16-wide shuffled column group, not at the end of
    the virtual K axis. Instead reshape into the underlying 16x16 tile grid
    and append whole zero tiles, which preserves the invariant
    ``preshuffle(pad(W)) == pad_shuffled(preshuffle(W))``.
    """
    if int(delta_bytes) <= 0:
        return w
    if not weight_is_preshuffled:
        return _mxscale_zero_pad_last(w, int(delta_bytes), cache=cache)

    if cache:
        key = _mxscale_pad_cache_key(w, int(delta_bytes), 0, True)
        cached = _mxscale_pad_cache_get(key)
        if cached is not None:
            return cached

    if int(delta_bytes) % 16 != 0:
        raise ValueError(
            f"preshuffled K-pad delta must be a multiple of 16 bytes, got {delta_bytes}"
        )
    E, N, K_old = w.shape
    if N % 16 != 0 or K_old % 16 != 0:
        raise ValueError(
            f"preshuffled weight must have N and K/pack_b divisible by 16, got N={N}, K={K_old}"
        )

    orig_dtype = w.dtype
    w_u8 = w.contiguous()
    if w.element_size() == 1 and w.dtype not in (torch.uint8, torch.int8):
        w_u8 = w_u8.view(torch.uint8)

    # Tile view: (E, N/16, K/16, 16, 16). Append delta_bytes/16 zero
    # tile-columns along the K-tile dim (dim 2).
    tile_view = w_u8.view(E, N // 16, K_old // 16, 16, 16)
    delta_tiles = int(delta_bytes) // 16
    padded = torch.nn.functional.pad(tile_view, (0, 0, 0, 0, 0, delta_tiles))
    padded = padded.contiguous().view(E, N, K_old + int(delta_bytes))
    if padded.dtype != orig_dtype:
        padded = padded.view(orig_dtype)
    if cache:
        _mxscale_pad_cache_put(key, padded)
    return padded


@functools.cache
def _get_compiled_silu_fused(
    inter_dim: int,
    topk: int,
    quant_mode: str = "fp4",
    gui_layout: bool = False,
    act: str = "silu",
    enable_bias: bool = False,
):
    """Compile and cache the fused gate activation + quant + scale-sort kernel."""
    from aiter.ops.flydsl.kernels.silu_and_mul_fq import build_silu_and_mul_fq_module

    return build_silu_and_mul_fq_module(
        inter_dim,
        topk,
        quant_mode,
        gui_layout,
        act=act,
        enable_bias=enable_bias,
    )


@functools.cache
def _get_compiled_swiglu(inter_dim: int):
    """Compile and cache the fused swiglu_and_mul kernel (interleaved input)."""
    from aiter.ops.flydsl.kernels.swiglu_and_mul import build_swiglu_and_mul_module

    return build_swiglu_and_mul_module(inter_dim)


def flydsl_swiglu_and_mul_interleaved(
    input: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Fused swiglu activation for interleaved (gate/up block-interleaved) layout.

    input: (rows, inter_dim*2) bf16, interleaved layout.
    out:   (rows, inter_dim) bf16.
    """
    inter_dim = out.shape[-1]
    num_rows = input.shape[0]
    _swiglu_fn = _get_compiled_swiglu(inter_dim)
    _run_compiled(
        _swiglu_fn,
        (
            input,
            out,
            num_rows,
            torch.cuda.current_stream(),
        ),
    )


def flydsl_silu_and_mul_interleaved(
    input: torch.Tensor,
    out: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    topk: int,
    quant_mode: str = "none",
    gui_layout: bool = True,
) -> None:
    """Fused silu activation for interleaved (gate/up block-interleaved) layout.

    input: (rows, inter_dim*2) bf16, interleaved layout.
    out:   (rows, inter_dim) bf16.
    """
    inter_dim = out.shape[-1]
    num_sorted_rows = sorted_token_ids.shape[0]
    _silu_fn = _get_compiled_silu_fused(
        inter_dim,
        topk,
        quant_mode=quant_mode,
        gui_layout=gui_layout,
        act="silu",
    )
    empty_scale = torch.empty(0, dtype=torch.uint8, device=out.device)
    empty_i32 = torch.empty(0, dtype=torch.int32, device=out.device)
    empty_f32 = torch.empty(0, dtype=torch.float32, device=out.device)
    _run_compiled(
        _silu_fn,
        (
            ptr_arg(input),
            ptr_arg(out),
            ptr_arg(empty_scale),
            ptr_arg(sorted_token_ids),
            ptr_arg(num_valid_ids),
            ptr_arg(empty_i32),
            ptr_arg(empty_f32),
            token_num,
            num_sorted_rows,
            float("inf"),
            torch.cuda.current_stream(),
        ),
    )


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
    use_async_copy: bool = False,
    k_batch: int = 1,
    waves_per_eu: int = 3,
    b_nt: int = 0,
    gate_mode: str = "separated",
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    bias: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    a_scale_one: bool = False,
    xcd_swizzle: int = 0,
    swiglu_limit: Optional[float] = None,
    k_wave: int = 1,
):
    """Fused gate+up GEMM (MOE stage1).

    a: (token_num, model_dim), w1: (E, 2*inter_dim, model_dim) pre-shuffled.
    model_dim and inter_dim INCLUDE padding (model_dim_pad, inter_dim_pad).
    bias: optional (E, 2*inter_dim) f32 bias added before activation.
    For fp4 stage1, `w1`/`w1_scale` must use the same preshuffle layout as
    `shuffle_weight_a16w4(w1, 16, True)` and `shuffle_scale_a16w4(w1_scale, E, True)`.

    When fuse_quant=True, the kernel fuses quantization (fp4/fp8, inferred from
    out_dtype) and writes e8m0 scales in sorted tiled layout directly.

    When k_batch>1 (split-K), the kernel outputs gate/up partials via atomic
    add into a zeroed buffer, then silu_and_mul fuses activation + reduction.

    gate_mode controls the gate/up computation strategy (see GateMode enum).

    Returns:
        Basic:                      out
        fuse_quant:                 (out, out_scale_sorted)
    """
    token_num = a.shape[0]
    E = w1.shape[0]
    inter_dim = w1.shape[1] // 2
    model_dim = a.shape[1]

    if a_dtype == "fp4":
        model_dim = model_dim * 2

    _need_fp4 = out_dtype == "fp4"
    _need_fp8 = out_dtype == "fp8"
    _fuse_any_quant = _need_fp4 or _need_fp8
    _base_out_dtype = "bf16" if _fuse_any_quant else out_dtype
    dtypes = _get_dtypes()

    if _need_fp4:
        torch_out_dtype = dtypes.fp4x2
    elif _need_fp8:
        torch_out_dtype = dtypes.fp8
    else:
        torch_out_dtype = dtypes.bf16 if out_dtype == "bf16" else dtypes.fp16
    _is_splitk = k_batch > 1
    gate_up_interleave = gate_mode == "interleave"

    dev = a.device
    _splitk_fp4 = _is_splitk and _need_fp4
    _gui_sk = gate_up_interleave and _is_splitk
    _gui_sk_fused = _gui_sk and _fuse_any_quant

    if out is None:
        if _need_fp4 or (_gui_sk_fused and _need_fp4):
            out = torch.empty(
                (token_num, topk, inter_dim // 2), dtype=dtypes.fp4x2, device=dev
            )
        elif _need_fp8 or (_gui_sk_fused and _need_fp8):
            out = torch.empty(
                (token_num, topk, inter_dim), dtype=dtypes.fp8, device=dev
            )
        else:
            out = torch.empty(
                (token_num, topk, inter_dim), dtype=torch_out_dtype, device=dev
            )

    if _is_splitk:
        torch_tmp_out_dtype = dtypes.bf16 if _base_out_dtype == "bf16" else dtypes.fp16
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

    _need_quant = _fuse_any_quant or _splitk_fp4 or _gui_sk_fused
    _need_sort = _need_quant

    _sort_block_m = tile_m
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
    _gemm_out_dtype = _base_out_dtype if _is_splitk else out_dtype

    if bias is not None and bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    _kernel_out = tmp_out if _is_splitk else out
    kernel_bias = None if _is_splitk else bias
    # fp4 and fp8 weights both use the MX gemm kernel (bias/out_scale arg builder).
    use_mx_gemm = b_dtype in ("fp4", "fp8")
    _n_in = inter_dim * 2 if use_mx_gemm else inter_dim
    _k_in = model_dim
    _swiglu_limit_val = runtime_swiglu_limit(swiglu_limit, act)

    if use_mx_gemm:
        args = _s1_args_fp4(
            _kernel_out.view(-1),
            a.view(-1),
            w1.view(-1),
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            out_scale_sorted_flat.view(-1),
            token_num,
            _n_in,
            _k_in,
            _grid_y,
            dev,
            bias=(
                kernel_bias.view(-1)
                if kernel_bias is not None
                else torch.empty(0, device=dev)
            ),
            swiglu_limit=_swiglu_limit_val,
        )
    else:
        args = _s1_args_std(
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
            _n_in,
            _k_in,
            _grid_y,
        )

    exe = compile_flydsl_moe_stage1(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage1=(sorted_weights is not None),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=_gemm_out_dtype,
        act=act,
        persist_m=_persist_m,
        use_async_copy=use_async_copy,
        k_batch=k_batch,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        gate_mode=gate_mode,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        enable_bias=(kernel_bias is not None),
        a_scale_one=a_scale_one,
        xcd_swizzle=xcd_swizzle,
        k_wave=k_wave,
    )
    _run_compiled(exe, args)

    num_sorted_rows = sorted_token_ids.shape[0]
    use_splitk_bias = _is_splitk and bias is not None
    if use_splitk_bias and topk_ids is None:
        raise ValueError("topk_ids are required for split-K FlyDSL stage1 bias")
    # sorted_token_ids only gives (token_id, slot_id). Bias is stored per expert,
    # so the post-activation kernel needs topk_ids[token_id * topk + slot_id].
    topk_ids_arg = (
        topk_ids.to(torch.int32).contiguous().view(-1)
        if use_splitk_bias
        else sorted_token_ids.view(-1)
    )
    bias_arg = (
        bias.contiguous().view(-1)
        if use_splitk_bias
        else (
            bias.contiguous().view(-1)[:0]
            if bias is not None
            else torch.empty(0, device=sorted_token_ids.device, dtype=torch.float32)
        )
    )
    if _gui_sk_fused:
        _quant_mode = "fp4" if _need_fp4 else "fp8"
        _silu_fused_k = _get_compiled_silu_fused(
            inter_dim,
            topk,
            _quant_mode,
            gui_layout=True,
            act=act,
            enable_bias=use_splitk_bias,
        )
        _run_compiled(
            _silu_fused_k,
            (
                ptr_arg(tmp_out.view(-1, inter_dim * 2)),
                ptr_arg(out.view(-1).view(torch.uint8)),
                ptr_arg(out_scale_sorted_flat),
                ptr_arg(sorted_token_ids),
                ptr_arg(num_valid_ids),
                ptr_arg(topk_ids_arg),
                ptr_arg(bias_arg),
                token_num,
                num_sorted_rows,
                _swiglu_limit_val,
                torch.cuda.current_stream(),
            ),
        )
    elif _gui_sk:
        _silu_fused_k = _get_compiled_silu_fused(
            inter_dim,
            topk,
            "none",
            gui_layout=True,
            act=act,
            enable_bias=use_splitk_bias,
        )
        _run_compiled(
            _silu_fused_k,
            (
                ptr_arg(tmp_out.view(-1, inter_dim * 2)),
                ptr_arg(out.view(-1).view(torch.uint8)),
                ptr_arg(out_scale_sorted_flat),
                ptr_arg(sorted_token_ids),
                ptr_arg(num_valid_ids),
                ptr_arg(topk_ids_arg),
                ptr_arg(bias_arg),
                token_num,
                num_sorted_rows,
                _swiglu_limit_val,
                torch.cuda.current_stream(),
            ),
        )
    elif _splitk_fp4:
        _silu_fused_k = _get_compiled_silu_fused(
            inter_dim,
            topk,
            act=act,
            enable_bias=use_splitk_bias,
        )
        _run_compiled(
            _silu_fused_k,
            (
                ptr_arg(tmp_out.view(-1, inter_dim * 2)),
                ptr_arg(out.view(-1).view(torch.uint8)),
                ptr_arg(out_scale_sorted_flat),
                ptr_arg(sorted_token_ids),
                ptr_arg(num_valid_ids),
                ptr_arg(topk_ids_arg),
                ptr_arg(bias_arg),
                token_num,
                num_sorted_rows,
                _swiglu_limit_val,
                torch.cuda.current_stream(),
            ),
        )
    elif _is_splitk:
        from aiter.ops.activation import (
            silu_and_mul,
            silu_and_mul_bias,
            swiglu_and_mul,
            swiglu_and_mul_bias,
        )

        post_input = tmp_out.view(-1, inter_dim * 2)
        post_out = out.view(-1, inter_dim)
        post_bias = bias.contiguous() if bias is not None else None
        if bias is not None and act == "swiglu":
            swiglu_and_mul_bias(post_out, post_input, topk_ids_arg, post_bias)
        elif bias is not None and act == "silu":
            silu_and_mul_bias(post_out, post_input, topk_ids_arg, post_bias)
        elif act == "swiglu":
            swiglu_and_mul(post_out, post_input)
        else:
            if bias is not None:
                post_input = post_input + bias[topk_ids.to(torch.long)].view(
                    -1, inter_dim * 2
                )
            silu_and_mul(post_out, post_input)

    if _fuse_any_quant and _need_sort:
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
    waves_per_eu: Optional[int] = None,
    use_async_copy: bool = False,
    cu_num_mul: int = 1,
    b_nt: int = 0,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    xcd_swizzle: int = 0,
    bias: Optional[torch.Tensor] = None,
    return_per_slot: bool = False,
    expert_mask: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Down-projection GEMM (MOE stage2). Supports atomic/reduce modes.

    a: (token_num, topk, inter_dim), w1: (E, model_dim, inter_dim) pre-shuffled.
    Returns (token_num, model_dim) by default.
    bias: optional (E, model_dim) f32 bias added after GEMM.

    sort_block_m: block_size used by moe_sorting / stage1. When 0 (default),
        assumed equal to tile_m. When set, stage2 can use a different tile_m
        from sorting/stage1.
    persist: if True, use persistent round-robin mode (grid_y=cu_num);
        if False, use legacy persist_m mode; if None, auto-select.

    return_per_slot: when True, return the raw per-(token, slot) output as a
        contiguous (token_num, topk, model_dim) tensor without applying the
        topk reduction.

    expert_mask, topk_ids: when both are provided and mode="reduce", the
        post-GEMM reduction fuses the EP validity gather
        ``valid = expert_mask[topk_ids[t, k]] != 0`` and only sums valid
        slots. expert_mask is [num_experts] i32, topk_ids is [token_num, topk] i32.
    """

    token_num = inter_states.shape[0]
    E = w2.shape[0]
    model_dim = w2.shape[1]
    inter_dim = inter_states.shape[2]

    # Debug: force stage2 to use the masked reduce epilogue instead of atomic
    # accumulate. Enabled by default; set AITER_FLYDSL_FORCE_REDUCE=0 to opt out.
    if os.environ.get("AITER_FLYDSL_FORCE_REDUCE", "0") == "1":
        mode = "reduce"

    accumulate = mode != "reduce" and not return_per_slot

    if a_dtype == "fp4":
        inter_dim = inter_dim * 2

    torch_out_dtype = torch.bfloat16 if out_dtype == "bf16" else torch.float16

    if out is None:
        if return_per_slot:
            out = torch.empty(
                (token_num, topk, model_dim),
                dtype=torch_out_dtype,
                device=inter_states.device,
            )
        else:
            alloc_fn = torch.zeros if accumulate else torch.empty
            out = alloc_fn(
                (token_num, model_dim),
                dtype=torch_out_dtype,
                device=inter_states.device,
            )
    # NOTE: when ``accumulate=True`` (atomic mode), the caller is responsible
    # for ensuring ``out`` is zero-initialized. In the standard ``fused_moe``
    # dispatch path this is handled by ``moe_sorting_*_fwd`` which already
    # zeros ``moe_buf`` via ``moe_buf_set_zero_kernel_2d``, so an extra
    # ``out.fill_(0)`` here would be a redundant ~``token_num * model_dim``
    # HBM write (~130us per call at MI355X HBM bw on EP4 prefill shape).

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

    if a_dtype == "fp8":
        _persist_m = 1

    if bias is not None and bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    # fp4 and fp8 weights both use the MX gemm kernel (bias arg builder).
    use_mx_gemm = b_dtype in ("fp4", "fp8")
    _n_in = model_dim
    _k_in = inter_dim

    target = out
    if not accumulate:
        if return_per_slot:
            target = out.view(-1)
        else:
            target = torch.empty(
                (token_num * topk * model_dim,),
                device=out.device,
                dtype=out.dtype,
            )

    if use_mx_gemm:
        args = _s2_args_fp4(
            target,
            inter_states,
            w2,
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            token_num,
            _n_in,
            _k_in,
            m_blocks,
            dev,
            bias=bias,
        )
    else:
        args = _s2_args_std(
            target,
            inter_states,
            w2,
            flat_a_scale,
            flat_w_scale,
            sorted_token_ids,
            sorted_expert_ids,
            sw,
            num_valid_ids,
            token_num,
            _n_in,
            _k_in,
            m_blocks,
        )

    exe = compile_flydsl_moe_stage2(
        model_dim=model_dim,
        inter_dim=inter_dim,
        experts=E,
        topk=topk,
        tile_m=tile_m,
        tile_n=tile_n,
        tile_k=tile_k,
        doweight_stage2=(sorted_weights is not None),
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        out_dtype=out_dtype,
        accumulate=accumulate,
        persist_m=_persist_m,
        sort_block_m=sort_block_m,
        waves_per_eu=waves_per_eu,
        use_async_copy=use_async_copy,
        cu_num_mul=cu_num_mul,
        b_nt=b_nt,
        model_dim_pad=model_dim_pad,
        inter_dim_pad=inter_dim_pad,
        xcd_swizzle=xcd_swizzle,
        enable_bias=(bias is not None),
    )
    _run_compiled(exe, args)

    if not accumulate:
        use_mask = expert_mask is not None
        if use_mask and topk_ids is None:
            raise ValueError(
                "topk_ids is required when expert_mask is provided for reduce mode"
            )
    if not accumulate and not return_per_slot:
        _run_moe_reduction(
            target, out, token_num, topk, model_dim, expert_mask, topk_ids
        )
    return out


# Fused route-map + MX quant + scatter-copy + scale-preshuffle kernels


@functools.cache
def _get_compiled_fused_route_quant_scatter(
    model_dim: int,
    topk: int,
    wmma_rep: int,
    quant_mode: str = "fp4",
    use_expert_row_base: bool = True,
    max_m: int = 0,
):
    """Compile and cache the fused route+quant+scatter+preshuffle kernel."""
    from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
        build_moe_fused_route_quant_scatter_module,
    )

    return build_moe_fused_route_quant_scatter_module(
        model_dim=model_dim,
        topk=topk,
        wmma_rep=wmma_rep,
        quant_mode=quant_mode,
        use_expert_row_base=use_expert_row_base,
        max_m=max_m,
    )


@functools.cache
def _get_compiled_fused_route_quant_scatter_st_ksplit(
    model_dim: int,
    topk: int,
    wmma_rep: int,
    quant_mode: str = "fp4",
    use_expert_row_base: bool = True,
    max_m: int = 0,
):
    from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
        build_moe_fused_route_quant_scatter_st_ksplit_module,
    )

    return build_moe_fused_route_quant_scatter_st_ksplit_module(
        model_dim=model_dim,
        topk=topk,
        wmma_rep=wmma_rep,
        quant_mode=quant_mode,
        use_expert_row_base=use_expert_row_base,
        max_m=max_m,
    )


@functools.cache
def _get_compiled_topids_to_rows():
    from aiter.ops.flydsl.kernels.moe_route_maps import build_moe_topids_to_rows_module

    return build_moe_topids_to_rows_module()


def flydsl_moe_topids_to_rows(
    topk_ids: torch.Tensor,
    E: int,
    max_m: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build masked-layout route rows and per-expert counts."""
    device = topk_ids.device
    token_num, topk = topk_ids.shape
    numel = token_num * topk
    counter = torch.zeros(E, dtype=torch.int32, device=device)
    topids_to_rows = torch.empty(numel, dtype=torch.int32, device=device)

    route_grid = (numel + 255) // 256
    topids_to_rows_kernel = _get_compiled_topids_to_rows()
    topids_to_rows_kernel(
        ptr_arg(topk_ids.to(torch.int32).reshape(-1)),
        ptr_arg(counter),
        ptr_arg(topids_to_rows),
        numel,
        int(max_m),
        route_grid,
        stream=torch.cuda.current_stream(),
    )
    return counter, topids_to_rows.view(token_num, topk)


def flydsl_moe_fused_route_quant_scatter(
    hidden_states: torch.Tensor,  # (token_num, model_dim) bf16
    topk_ids: torch.Tensor,  # (token_num, topk) int32 local expert ids
    E: int,
    max_m: int,
    *,
    wmma_rep: int,
    quant_mode: str = "fp4",
    expert_row_base: Optional[torch.Tensor] = None,  # (E,) int32 dst row base
    out_E: Optional[int] = None,
    out_max_m: Optional[int] = None,
    grouped_a1: Optional[torch.Tensor] = None,  # (out_E, out_max_m, Pb) uint8 out
    grouped_a1_scale: Optional[
        torch.Tensor
    ] = None,  # (out_E, out_max_m//wmma_rep, (model_dim//32)*wmma_rep) uint8 out
):
    """Fused route+MX-quant+scatter+preshuffle in one pass.

    Returns (grouped_a1, grouped_a1_scale, masked_m, topids_to_rows).
    """
    if quant_mode not in ("fp4", "fp8"):
        raise NotImplementedError(
            f"flydsl_moe_fused_route_quant_scatter: quant_mode={quant_mode!r} "
            "unsupported (expected 'fp4' or 'fp8')."
        )
    assert hidden_states.dtype == torch.bfloat16, (
        "fused route+quant kernel currently requires bf16 hidden_states "
        f"(got {hidden_states.dtype})"
    )
    device = hidden_states.device
    token_num, topk = topk_ids.shape
    numel = token_num * topk
    model_dim = hidden_states.shape[-1]
    rows_per_tile = wmma_rep * 16
    assert (
        max_m % rows_per_tile == 0
    ), f"max_m ({max_m}) must be a multiple of wmma_rep*16 ({rows_per_tile})"

    out_E = E if out_E is None else int(out_E)
    out_max_m = max_m if out_max_m is None else int(out_max_m)
    assert out_max_m % rows_per_tile == 0, (
        f"out_max_m ({out_max_m}) must be a multiple of wmma_rep*16 "
        f"({rows_per_tile})"
    )

    payload_bytes_per_row = model_dim if quant_mode == "fp8" else model_dim // 2
    scale_bytes_per_row = model_dim // 32

    use_expert_row_base = expert_row_base is not None
    if use_expert_row_base:
        expert_row_base = expert_row_base.to(device=device, dtype=torch.int32)

    use_routeks_stage1 = (
        token_num > 1 and topk > 1 and quant_mode == "fp4" and not use_expert_row_base
    )
    route_grid = (numel + 255) // 256
    counter = torch.zeros(E, dtype=torch.int32, device=device)
    topids_to_rows = torch.empty(numel, dtype=torch.int32, device=device)
    if grouped_a1 is None:
        grouped_a1 = torch.empty(
            (out_E, out_max_m, payload_bytes_per_row),
            dtype=torch.uint8,
            device=device,
        )
    if grouped_a1_scale is None:
        grouped_a1_scale = torch.empty(
            (out_E, out_max_m // wmma_rep, scale_bytes_per_row * wmma_rep),
            dtype=torch.uint8,
            device=device,
        )

    from aiter.ops.flydsl.kernels.kernels_common import get_warp_size

    wave_size = get_warp_size()
    warps_per_block = 256 // wave_size
    grid_blocks = (numel + warps_per_block - 1) // warps_per_block

    hidden_flat = hidden_states.contiguous().view(-1)
    topk_ids_i32 = topk_ids.to(torch.int32).reshape(-1)
    expert_row_base_arg = (
        expert_row_base.reshape(-1) if use_expert_row_base else counter
    )

    if use_routeks_stage1:
        topids_to_rows_kernel = _get_compiled_topids_to_rows()
        topids_to_rows_kernel(
            ptr_arg(topk_ids_i32),
            ptr_arg(counter),
            ptr_arg(topids_to_rows),
            numel,
            max_m,
            route_grid,
            stream=torch.cuda.current_stream(),
        )
        launch_routeks = _get_compiled_fused_quant_preshuffle_route_ksplit(
            feat_dim=model_dim,
            wmma_rep=wmma_rep,
            quant_mode=quant_mode,
            source_topk=topk,
        )
        launch_routeks(
            ptr_arg(hidden_flat),
            ptr_arg(grouped_a1.view(-1)),
            ptr_arg(grouped_a1_scale.view(-1)),
            ptr_arg(topids_to_rows),
            ptr_arg(counter),  # dummy row_starts; unused because remap_rows=False
            1,
            numel,
            grid_blocks,
            stream=torch.cuda.current_stream(),
        )
        return (
            grouped_a1,
            grouped_a1_scale,
            counter,
            topids_to_rows.view(token_num, topk),
        )

    use_st_ksplit = token_num == 1 and topk > 0 and (topk & (topk - 1)) == 0
    if use_st_ksplit:
        launch = _get_compiled_fused_route_quant_scatter_st_ksplit(
            model_dim=model_dim,
            topk=topk,
            wmma_rep=wmma_rep,
            quant_mode=quant_mode,
            use_expert_row_base=use_expert_row_base,
            max_m=max_m,
        )
    else:
        launch = _get_compiled_fused_route_quant_scatter(
            model_dim=model_dim,
            topk=topk,
            wmma_rep=wmma_rep,
            quant_mode=quant_mode,
            use_expert_row_base=use_expert_row_base,
            max_m=max_m,
        )
    launch(
        ptr_arg(topk_ids_i32),
        ptr_arg(counter),
        ptr_arg(topids_to_rows),
        ptr_arg(hidden_flat),
        ptr_arg(grouped_a1.view(-1)),
        ptr_arg(grouped_a1_scale.view(-1)),
        ptr_arg(expert_row_base_arg),
        numel,
        grid_blocks,
        stream=torch.cuda.current_stream(),
    )
    return (
        grouped_a1,
        grouped_a1_scale,
        counter,
        topids_to_rows.view(token_num, topk),
    )


@functools.cache
def _get_compiled_fused_route_psum_quant_scatter(
    model_dim: int,
    topk: int,
    wmma_rep: int,
    quant_mode: str = "fp4",
):
    """Compile and cache the fully-fused route+psum+quant+scatter kernel."""
    from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
        build_moe_fused_route_psum_quant_scatter_module,
    )

    return build_moe_fused_route_psum_quant_scatter_module(
        model_dim=model_dim,
        topk=topk,
        wmma_rep=wmma_rep,
        quant_mode=quant_mode,
    )


def flydsl_moe_fused_route_psum_quant_scatter(
    hidden_states: torch.Tensor,  # (token_num, model_dim) bf16
    topk_ids: torch.Tensor,  # (token_num, topk) int32 local expert ids
    E: int,
    tile_m: int,
    contiguous_m: int,
    *,
    wmma_rep: int,
    quant_mode: str = "fp4",
):
    """Fully-fused route+psum+quant+scatter for DeepGEMM contiguous-M layout.

    Returns (grouped_a1, grouped_a1_scale, masked_m, topids_to_rows, starts, psum).
    """
    if quant_mode not in ("fp4", "fp8"):
        raise NotImplementedError(
            f"flydsl_moe_fused_route_psum_quant_scatter: quant_mode={quant_mode!r} "
            "unsupported (expected 'fp4' or 'fp8')."
        )
    assert hidden_states.dtype == torch.bfloat16, (
        "fused route+psum+quant kernel currently requires bf16 hidden_states "
        f"(got {hidden_states.dtype})"
    )
    device = hidden_states.device
    token_num, topk = topk_ids.shape
    numel = token_num * topk
    model_dim = hidden_states.shape[-1]
    rows_per_tile = wmma_rep * 16
    contiguous_m = int(contiguous_m)
    assert contiguous_m % rows_per_tile == 0, (
        f"contiguous_m ({contiguous_m}) must be a multiple of wmma_rep*16 "
        f"({rows_per_tile})"
    )
    assert int(tile_m) % rows_per_tile == 0, (
        f"tile_m ({tile_m}) must be a multiple of wmma_rep*16 ({rows_per_tile}) "
        "so tile-aligned starts stay preshuffle-consistent"
    )

    payload_bytes_per_row = model_dim if quant_mode == "fp8" else model_dim // 2
    scale_bytes_per_row = model_dim // 32

    count = torch.zeros(E, dtype=torch.int32, device=device)
    slot_counter = torch.zeros(E, dtype=torch.int32, device=device)
    # Zero-init defensively; in-kernel prefix sum writes these.
    starts = torch.zeros(E, dtype=torch.int32, device=device)
    psum = torch.zeros(E, dtype=torch.int32, device=device)
    barrier = torch.zeros(2, dtype=torch.int32, device=device)
    topids_to_rows = torch.empty(numel, dtype=torch.int32, device=device)

    grouped_a1 = torch.empty(
        (1, contiguous_m, payload_bytes_per_row),
        dtype=torch.uint8,
        device=device,
    )
    grouped_a1_scale = torch.empty(
        (1, contiguous_m // wmma_rep, scale_bytes_per_row * wmma_rep),
        dtype=torch.uint8,
        device=device,
    )

    from aiter.jit.utils.chip_info import get_cu_num

    num_workers = int(get_cu_num())

    hidden_flat = hidden_states.contiguous().view(-1)
    topk_ids_i32 = topk_ids.to(torch.int32).reshape(-1)

    launch = _get_compiled_fused_route_psum_quant_scatter(
        model_dim=model_dim,
        topk=topk,
        wmma_rep=wmma_rep,
        quant_mode=quant_mode,
    )
    launch(
        ptr_arg(topk_ids_i32),
        ptr_arg(count),
        ptr_arg(slot_counter),
        ptr_arg(starts),
        ptr_arg(psum),
        ptr_arg(barrier),
        ptr_arg(topids_to_rows),
        ptr_arg(hidden_flat),
        ptr_arg(grouped_a1.view(-1)),
        ptr_arg(grouped_a1_scale.view(-1)),
        numel,
        int(E),
        int(tile_m),
        num_workers,
        num_workers,
        stream=torch.cuda.current_stream(),
    )
    return (
        grouped_a1,
        grouped_a1_scale,
        count,
        topids_to_rows.view(token_num, topk),
        starts,
        psum,
    )


# Fused grouped MX quant + scale-preshuffle (stage2 input prep)


@functools.cache
def _get_compiled_fused_quant_preshuffle(
    feat_dim: int,
    wmma_rep: int,
    quant_mode: str = "fp4",
    skip_padding: bool = False,
):
    from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
        build_moe_fused_quant_preshuffle_module,
    )

    return build_moe_fused_quant_preshuffle_module(
        feat_dim=feat_dim,
        wmma_rep=wmma_rep,
        quant_mode=quant_mode,
        skip_padding=skip_padding,
    )


@functools.cache
def _get_compiled_fused_quant_preshuffle_route_ksplit(
    feat_dim: int,
    wmma_rep: int,
    quant_mode: str = "fp4",
    source_topk: int = 0,
    remap_rows: bool = False,
):
    from aiter.ops.flydsl.kernels.moe_fused_route_quant_scatter import (
        build_moe_fused_quant_preshuffle_route_ksplit_module,
    )

    return build_moe_fused_quant_preshuffle_route_ksplit_module(
        feat_dim=feat_dim,
        wmma_rep=wmma_rep,
        quant_mode=quant_mode,
        source_topk=source_topk,
        remap_rows=remap_rows,
    )


def flydsl_moe_fused_quant_preshuffle(
    grouped_in: torch.Tensor,  # (E, max_m, feat_dim) or (E*max_m, feat_dim) bf16
    E: int,
    max_m: int,
    *,
    wmma_rep: int,
    quant_mode: str = "fp4",
    masked_m: Optional[torch.Tensor] = None,  # (E,) int32 valid rows per expert
    topids_to_rows: Optional[torch.Tensor] = None,  # route -> global row
    source_topk: int = 0,  # when >0, routeks reads source row = route // source_topk
    row_starts: Optional[torch.Tensor] = None,  # remap masked rows to starts[e]+slot
    route_max_m: int = 0,
    out_payload: Optional[torch.Tensor] = None,  # (E, max_m, Pb) uint8
    out_scale: Optional[torch.Tensor] = None,  # (E, max_m//wmma_rep, Ws*wmma_rep)
):
    """Fused grouped quant + e8m0 scale-preshuffle in one kernel pass.

    Returns (payload, scale_preshuffle). Pass masked_m to skip padding rows.
    """
    if quant_mode not in ("fp4", "fp8"):
        raise NotImplementedError(
            f"flydsl_moe_fused_quant_preshuffle: quant_mode={quant_mode!r} "
            "unsupported (expected 'fp4' or 'fp8')."
        )
    assert grouped_in.dtype == torch.bfloat16, (
        "fused grouped quant+preshuffle requires bf16 input "
        f"(got {grouped_in.dtype})"
    )
    device = grouped_in.device
    feat_dim = grouped_in.shape[-1]
    rows_per_tile = wmma_rep * 16
    assert (
        max_m % rows_per_tile == 0
    ), f"max_m ({max_m}) must be a multiple of wmma_rep*16 ({rows_per_tile})"

    n_rows = E * max_m
    Pb = feat_dim if quant_mode == "fp8" else feat_dim // 2
    Ws = feat_dim // 32
    if out_payload is None:
        out_payload = torch.empty((E, max_m, Pb), dtype=torch.uint8, device=device)
    if out_scale is None:
        out_scale = torch.empty(
            (E, max_m // wmma_rep, Ws * wmma_rep), dtype=torch.uint8, device=device
        )

    skip_padding = masked_m is not None
    if skip_padding:
        masked_m = masked_m.to(device=device, dtype=torch.int32).reshape(-1)
    else:
        # Unused by the kernel (skip_padding=False); a tiny dummy keeps the launch
        # signature uniform without allocating per-row scratch.
        masked_m = torch.empty(max(E, 1), dtype=torch.int32, device=device)

    from aiter.ops.flydsl.kernels.kernels_common import get_warp_size

    wave_size = get_warp_size()
    warps_per_block = 256 // wave_size
    if topids_to_rows is not None:
        topids_to_rows_i32 = topids_to_rows.to(
            device=device, dtype=torch.int32
        ).reshape(-1)
        numel = int(topids_to_rows_i32.numel())
        grid_blocks = (numel + warps_per_block - 1) // warps_per_block
        remap_rows = row_starts is not None
        if remap_rows:
            row_starts_i32 = row_starts.to(device=device, dtype=torch.int32).reshape(-1)
            route_max_m_arg = int(route_max_m)
            if route_max_m_arg <= 0:
                raise ValueError(
                    "route_max_m must be positive when row_starts is provided"
                )
        else:
            row_starts_i32 = masked_m
            route_max_m_arg = 1
        launch = _get_compiled_fused_quant_preshuffle_route_ksplit(
            feat_dim=feat_dim,
            wmma_rep=wmma_rep,
            quant_mode=quant_mode,
            source_topk=source_topk,
            remap_rows=remap_rows,
        )
        launch(
            ptr_arg(grouped_in.contiguous().view(-1)),
            ptr_arg(out_payload.view(-1)),
            ptr_arg(out_scale.view(-1)),
            ptr_arg(topids_to_rows_i32),
            ptr_arg(row_starts_i32),
            route_max_m_arg,
            numel,
            grid_blocks,
            stream=torch.cuda.current_stream(),
        )
        return out_payload, out_scale

    grid_blocks = (n_rows + warps_per_block - 1) // warps_per_block

    launch = _get_compiled_fused_quant_preshuffle(
        feat_dim=feat_dim,
        wmma_rep=wmma_rep,
        quant_mode=quant_mode,
        skip_padding=skip_padding,
    )
    launch(
        ptr_arg(grouped_in.contiguous().view(-1)),
        ptr_arg(out_payload.view(-1)),
        ptr_arg(out_scale.view(-1)),
        ptr_arg(masked_m),
        n_rows,
        max_m,
        grid_blocks,
        stream=torch.cuda.current_stream(),
    )
    return out_payload, out_scale
