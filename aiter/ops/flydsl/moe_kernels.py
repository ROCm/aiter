# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE kernel management: naming, compilation, and high-level API."""

import functools
import os
import re

from typing import Dict, Optional

import torch

_KERNEL_PARAMS: Dict[str, Dict] = {}


def _get_dtypes():
    from aiter.utility import dtypes

    return dtypes


_SUFFIX_RE = re.compile(r"(?P<fp4>_fp4)?(?P<fp8>_fp8)?(?:_sbm(?P<sbm>\d+))?$")


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

    Strips ``_fp4`` / ``_fp8`` / ``_sbm{N}`` suffixes transparently.
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
                                    if a_dtype == "fp8":
                                        name += "_gui"
                                    if xcd > 0:
                                        name += f"_xcd{xcd}"
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
                                    }
    return kernels


def get_flydsl_stage2_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str
) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported stage2 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"
    tile_ns = [128, 256] if is_fp4 else [128]
    # fp4 stage2 supports tile_k=128 (pack_K=1 scale sub-group shift path) as
    # well as 256.  tile_k=128 cleanly tiles K=inter_dim for TP-sharded shapes
    # whose inter_dim is a multiple of 128 but not 256 (e.g. MiniMax TP4=384).
    tile_ks = [128, 256] if is_fp4 else [128]
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
    return kernels


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
    swiglu_limit: float = 0.0,
):
    """Compile stage1 kernel (cached via underlying lru_cache)."""
    if b_dtype == "fp4":
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
            swiglu_limit=swiglu_limit,
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
    b_nt: int = 0,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    xcd_swizzle: int = 0,
    enable_bias: bool = False,
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
            b_nt=b_nt,
            model_dim_pad=model_dim_pad,
            inter_dim_pad=inter_dim_pad,
            xcd_swizzle=xcd_swizzle,
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


def _ptr_view_safe(t: torch.Tensor):
    """Pass only the device data pointer; shape is carried by explicit args."""
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    view = _view_safe(t)
    type_name = type(view).__name__
    module_name = type(view).__module__
    if type_name == "FakeTensor" or "fake_tensor" in module_name:
        return flyc.from_c_void_p(fx.Uint8, 0)
    return flyc.from_c_void_p(fx.Uint8, view.data_ptr())


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
):
    empty_f32 = torch.empty(0, device=dev, dtype=torch.float32)
    _bias = bias if bias is not None else empty_f32
    if stream is None:
        stream = torch.cuda.current_stream()
    return (
        _ptr_view_safe(out),
        _ptr_view_safe(a),
        _ptr_view_safe(w),
        _ptr_view_safe(a_scale),
        _ptr_view_safe(w_scale),
        _ptr_view_safe(sorted_ids),
        _ptr_view_safe(sorted_expert_ids),
        _ptr_view_safe(sorted_weights),
        _ptr_view_safe(num_valid_ids),
        _ptr_view_safe(_bias),
        _ptr_view_safe(out_scale_sorted),
        token_num,
        n_in,
        k_in,
        size_expert_ids_in,
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
        _ptr_view_safe(out),
        _ptr_view_safe(a),
        _ptr_view_safe(w),
        _ptr_view_safe(a_scale),
        _ptr_view_safe(w_scale),
        _ptr_view_safe(sorted_ids),
        _ptr_view_safe(sorted_expert_ids),
        _ptr_view_safe(sorted_weights),
        _ptr_view_safe(num_valid_ids),
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
        _ptr_view_safe(target),
        _ptr_view_safe(a),
        _ptr_view_safe(w),
        _ptr_view_safe(a_scale),
        _ptr_view_safe(w_scale),
        _ptr_view_safe(sorted_ids),
        _ptr_view_safe(sorted_expert_ids),
        _ptr_view_safe(sorted_weights),
        _ptr_view_safe(num_valid_ids),
        _ptr_view_safe(_bias),
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
        _ptr_view_safe(target),
        _ptr_view_safe(a),
        _ptr_view_safe(w),
        _ptr_view_safe(a_scale),
        _ptr_view_safe(w_scale),
        _ptr_view_safe(sorted_ids),
        _ptr_view_safe(sorted_expert_ids),
        _ptr_view_safe(sorted_weights),
        _ptr_view_safe(num_valid_ids),
        token_num,
        n_in,
        k_in,
        blocks,
        stream,
    )


def _run_compiled(exe, args):
    """Call the JitFunction with the given args.
    JitFunction.__call__ handles compilation caching internally.

    Some kernels (e.g. the gfx1250 fp8/a8w4 stage1 path) wrap the compiled
    JIT function in a host-side ``_Stage1GateUpPackedWrapper`` that repacks
    the weight and weight-scale tensors before dispatch. Calling ``exe(*args)``
    works for both ``@flyc.jit`` functions and such host-side wrappers since
    both are callable.
    """
    try:
        exe(*args)
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
    swiglu_limit: float = 0.0,
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
        swiglu_limit=swiglu_limit,
    )


@functools.cache
def _get_compiled_swiglu(inter_dim: int):
    """Compile and cache the fused swiglu_and_mul kernel (interleaved input)."""
    from aiter.ops.flydsl.kernels.swiglu_and_mul import build_swiglu_and_mul_module

    return build_swiglu_and_mul_module(inter_dim)


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
    swiglu_limit: float = 0.0,
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
    is_fp4 = b_dtype == "fp4"
    _n_in = inter_dim * 2 if is_fp4 else inter_dim
    _k_in = model_dim

    if is_fp4:
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
        swiglu_limit=swiglu_limit,
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
            swiglu_limit=swiglu_limit,
        )
        _run_compiled(
            _silu_fused_k,
            (
                _ptr_view_safe(tmp_out.view(-1, inter_dim * 2)),
                _ptr_view_safe(out.view(-1).view(torch.uint8)),
                _ptr_view_safe(out_scale_sorted_flat),
                _ptr_view_safe(sorted_token_ids),
                _ptr_view_safe(num_valid_ids),
                _ptr_view_safe(topk_ids_arg),
                _ptr_view_safe(bias_arg),
                token_num,
                num_sorted_rows,
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
            swiglu_limit=swiglu_limit,
        )
        _run_compiled(
            _silu_fused_k,
            (
                _ptr_view_safe(tmp_out.view(-1, inter_dim * 2)),
                _ptr_view_safe(out.view(-1).view(torch.uint8)),
                _ptr_view_safe(out_scale_sorted_flat),
                _ptr_view_safe(sorted_token_ids),
                _ptr_view_safe(num_valid_ids),
                _ptr_view_safe(topk_ids_arg),
                _ptr_view_safe(bias_arg),
                token_num,
                num_sorted_rows,
                torch.cuda.current_stream(),
            ),
        )
    elif _splitk_fp4:
        _silu_fused_k = _get_compiled_silu_fused(
            inter_dim,
            topk,
            act=act,
            enable_bias=use_splitk_bias,
            swiglu_limit=swiglu_limit,
        )
        _run_compiled(
            _silu_fused_k,
            (
                _ptr_view_safe(tmp_out.view(-1, inter_dim * 2)),
                _ptr_view_safe(out.view(-1).view(torch.uint8)),
                _ptr_view_safe(out_scale_sorted_flat),
                _ptr_view_safe(sorted_token_ids),
                _ptr_view_safe(num_valid_ids),
                _ptr_view_safe(topk_ids_arg),
                _ptr_view_safe(bias_arg),
                token_num,
                num_sorted_rows,
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
    b_nt: int = 0,
    model_dim_pad: int = 0,
    inter_dim_pad: int = 0,
    xcd_swizzle: int = 0,
    bias: Optional[torch.Tensor] = None,
    expert_mask: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Down-projection GEMM (MOE stage2). Supports atomic/reduce modes.

    a: (token_num, topk, inter_dim), w1: (E, model_dim, inter_dim) pre-shuffled.
    Returns (token_num, model_dim).
    bias: optional (E, model_dim) f32 bias added after GEMM.

    sort_block_m: block_size used by moe_sorting / stage1. When 0 (default),
        assumed equal to tile_m. When set, stage2 can use a different tile_m
        from sorting/stage1.
    persist: if True, use persistent round-robin mode (grid_y=cu_num);
        if False, use legacy persist_m mode; if None, auto-select.

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

    if a_dtype == "fp8":
        _persist_m = 1

    if bias is not None and bias.dtype != torch.float32:
        bias = bias.to(torch.float32)
    is_fp4 = b_dtype == "fp4"
    _n_in = model_dim
    _k_in = inter_dim

    target = out
    if not accumulate:
        target = torch.empty(
            (token_num * topk * model_dim,),
            device=out.device,
            dtype=out.dtype,
        )

    if is_fp4:
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
        # Map torch dtype -> compile_moe_reduction dtype_str
        if out.dtype == torch.float16:
            _reduce_dtype_str = "f16"
        elif out.dtype == torch.bfloat16:
            _reduce_dtype_str = "bf16"
        elif out.dtype == torch.float32:
            _reduce_dtype_str = "f32"
        else:
            _reduce_dtype_str = None

        if _reduce_dtype_str is not None:
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
            stream = torch.cuda.current_stream()
            reduce_exe(
                _ptr_view_safe(X),
                _ptr_view_safe(out),
                _ptr_view_safe(em),
                _ptr_view_safe(tk),
                token_num,
                stream,
            )
        else:
            # Unsupported dtype for the masked kernel — fall back to torch.sum.
            # This drops the EP mask, so only valid for non-EP runs.
            if use_mask:
                raise NotImplementedError(
                    f"Masked moe reduction not supported for dtype {out.dtype}"
                )
            torch.sum(target.view(token_num, topk, model_dim), dim=1, out=out)

    return out


# ---------------------------------------------------------------------------
# MoE gather-reduce (weighted) epilogue
#
# Final MoE step: combine the per-expert stage2 output ``grouped_out (E, max_m,
# model_dim)`` into the flat per-token output, weighting each row by its route
# weight and summing the ``topk`` contributions of every token::
#
#     moe_out[t] = sum_k  w(t,k) * grouped_out[expert(t,k), pos(t,k)]
#
# ``flydsl_moe_gather_reduce`` is the one-pass gather-reduce kernel: it builds a
# per-token inverse index map (vectorized host prep) and launches a single
# kernel that produces each output token's row in one pass. The scatter
# reference it is validated against (the per-expert ``index_add_`` loop) lives
# in ``op_tests/test_moe_gather_reduce.py``.
# ---------------------------------------------------------------------------


@functools.cache
def _get_compiled_gather_reduce(model_dim: int, topk: int, out_dtype: str):
    """Compile and cache the one-pass MoE gather-reduce kernel."""
    from aiter.ops.flydsl.kernels.moe_gather_reduce import (
        build_moe_gather_reduce_module,
    )

    return build_moe_gather_reduce_module(model_dim, topk, out_dtype)


def build_topids_to_rows(
    topk_ids: torch.Tensor,  # (token_num, topk) local expert ids in [0, E)
    max_m: int,
    E: int,
) -> torch.Tensor:
    """Per-token gather map: ``topids_to_rows[t,k] = topk_ids[t,k]*max_m + slot``, where
    ``slot`` is token ``t``'s within-expert position in token-major route order
    (matching how the route-gather fills each expert). Returns (token_num, topk)
    int32.

    Argsort-free: the within-expert ``slot`` is a one-hot cumsum (running count
    per expert in route order). Build this once and share it with the
    route-gather (scatter-copy) step instead of recomputing.
    """
    import torch.nn.functional as F

    token_num, topk = topk_ids.shape
    flat_e = topk_ids.reshape(-1).to(torch.long)
    # slot[r] = (# earlier routes to the same expert) = running count - 1
    slot = F.one_hot(flat_e, E).cumsum(0).gather(1, flat_e[:, None]).squeeze(1) - 1
    return (flat_e * max_m + slot).view(token_num, topk).to(torch.int32)


@functools.cache
def _get_compiled_route_maps():
    """Compile and cache the atomic route -> grouped-row map kernel."""
    from aiter.ops.flydsl.kernels.moe_route_maps import build_moe_route_maps_module

    return build_moe_route_maps_module()


def build_route_maps(topk_ids: torch.Tensor, E: int, max_m: int):
    """Per-token route maps via a single atomic-scatter kernel (SGLang-style),
    no host-side argsort / nonzero / one-hot. Returns
    ``(topids_to_rows, rows_to_tokens, masked_m)``:

      topids_to_rows : (token_num, topk) int32  -- route -> grouped row
                 = ``topk_ids[t,k]*max_m + slot`` (gather-reduce input)
      rows_to_tokens  : (E*max_m,)        int32  -- grouped row -> source token
                 (-1 for unused padding rows; scatter-copy input)
      masked_m        : (E,)              int32  -- rows routed to each expert
                 (== bincount(topk_ids), the per-expert GEMM mask)

    The within-expert ``slot`` is claimed by ``atomicAdd(1)`` on a per-expert
    counter initialized to 0; the kernel forms the grouped row in-place as
    ``slot + e*max_m`` (one int mul-add per thread, hidden behind the atomic).
    It writes both maps in one pass (topids_to_rows + its inverse
    rows_to_tokens), and the final counter value is exactly ``counts[e]`` -- so
    ``masked_m`` is the counter itself, no bincount and no host-side
    ``arange``/``clone``/``sub`` to build or strip an offset. Order within an
    expert is atomic-race order (nondeterministic) but self-consistent -- both
    maps come from the same run, and the grouped GEMM is order-agnostic per
    expert.
    """
    device = topk_ids.device
    token_num, topk = topk_ids.shape
    numel = token_num * topk
    topk_ids_i32 = topk_ids.reshape(-1).to(torch.int32).contiguous()
    # Per-expert counter starts at 0; the kernel applies the e*max_m offset, so
    # after the run this buffer holds counts[e] directly == masked_m.
    atomic_buffer = torch.zeros(E, dtype=torch.int32, device=device)
    topids_to_rows = torch.empty(numel, dtype=torch.int32, device=device)
    rows_to_tokens = torch.full((E * max_m,), -1, dtype=torch.int32, device=device)
    grid_blocks = (numel + 255) // 256
    launch = _get_compiled_route_maps()
    launch(
        topk_ids_i32,
        atomic_buffer,
        topids_to_rows,
        rows_to_tokens,
        numel,
        topk,
        max_m,
        grid_blocks,
        stream=torch.cuda.current_stream(),
    )
    # atomic_buffer[e] == counts[e] now; it is masked_m, no further math.
    masked_m = atomic_buffer
    return topids_to_rows.view(token_num, topk), rows_to_tokens, masked_m


def flydsl_moe_gather_reduce(
    grouped_out: torch.Tensor,  # (E, max_m, model_dim) bf16/f16
    topids_to_rows: torch.Tensor,  # (token_num, topk) int32 grouped flat rows
    gather_w: torch.Tensor,  # (token_num, topk) weight, bf16/f16 (== grouped_out dtype)
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """One-pass gather-reduce epilogue. Thin launcher over a *precomputed* gather
    map: ``out[t] = sum_k gather_w[t,k] * grouped_out_flat[topids_to_rows[t,k]]``.

    The caller builds ``topids_to_rows`` once (see ``build_topids_to_rows``,
    argsort-free) and may share it with the route-gather step; this wrapper does
    no host-side map building. ``grouped_out`` and ``gather_w`` must be bf16 or
    f16 (the kernel extends the weight to f32 internally for accumulation)."""
    E, max_m, model_dim = grouped_out.shape
    token_num, topk = topids_to_rows.shape
    device = grouped_out.device
    if grouped_out.dtype == torch.bfloat16:
        out_dtype = "bf16"
    elif grouped_out.dtype == torch.float16:
        out_dtype = "f16"
    else:
        raise ValueError(f"unsupported dtype {grouped_out.dtype}; need bf16/f16")

    # Caller passes topids_to_rows int32 and gather_w bf16/f16 (both contiguous).
    grouped_out_flat = grouped_out.contiguous().view(E * max_m, model_dim)
    if out is None:
        out = torch.empty(
            (token_num, model_dim), dtype=grouped_out.dtype, device=device
        )

    launch = _get_compiled_gather_reduce(model_dim, topk, out_dtype)
    launch(
        grouped_out_flat,
        topids_to_rows,
        gather_w,
        out,
        token_num,
        stream=torch.cuda.current_stream(),
    )
    return out


# ---------------------------------------------------------------------------
# MoE route-gather (scatter-copy) input layout
#
# Pre-stage1 step: copy each token's quantized payload (and per-token scale)
# from the flat per-token layout into the grouped per-expert layout::
#
#     for e in range(E):
#         toks = tokens routed to expert e        # n = counts[e]
#         grouped[e, :n] = a_payload[toks]
#
# ``flydsl_moe_scatter_copy_token`` does the heavy row copies in one kernel pass
# (one block per grouped row, gathered from its source token via a precomputed
# dst->src map) and fills route_tokens/route_weights with cheap host ops. The
# reference loop it is validated against lives in
# ``op_tests/test_moe_scatter_copy_token.py``.
# ---------------------------------------------------------------------------


@functools.cache
def _get_compiled_scatter_copy(row_bytes: int):
    """Compile and cache the one-pass row scatter-copy kernel (per row width)."""
    from aiter.ops.flydsl.kernels.moe_scatter_copy_token import (
        build_moe_scatter_copy_token_module,
    )

    return build_moe_scatter_copy_token_module(row_bytes)


def flydsl_moe_scatter_copy_token(
    a1_payload: torch.Tensor,  # (token_num, Wp) uint8
    a1_scale_token_u8: Optional[torch.Tensor],  # (token_num, Ws) uint8 or None
    rows_to_tokens: torch.Tensor,  # (E*max_m,) int32 grouped row -> token (-1 pad)
    E: int,
    max_m: int,
    grouped_a1: Optional[torch.Tensor] = None,  # (E, max_m, Wp) uint8 out
    a1_scale_raw: Optional[torch.Tensor] = None,  # (E, max_m, Ws) uint8 out
):
    """Copy each token's payload (and per-token scale) into the grouped layout,
    driven by ``rows_to_tokens`` (grouped row -> source token, -1 for padding)
    from ``build_route_maps``. Pure copy -- one kernel per tensor.

    route_tokens/route_weights are NOT produced here: they are needed only by the
    naive epilogue (built in that loop) and, for doweight_stage1, derived on
    demand by the caller from topk_weight + topids_to_rows.

    Output tensors may be passed in (the kernel writes only the mapped/valid
    rows, leaving any pre-existing padding untouched -- e.g. an a1_scale_raw
    pre-filled with 127). When omitted they are allocated zero-filled.

    Returns (grouped_a1, a1_scale_raw)."""
    device = a1_payload.device
    Wp = a1_payload.shape[1]
    num_dst = E * max_m

    if grouped_a1 is None:
        grouped_a1 = torch.zeros((E, max_m, Wp), dtype=torch.uint8, device=device)
    launch_p = _get_compiled_scatter_copy(Wp)
    launch_p(
        a1_payload.contiguous().view(-1, Wp),
        grouped_a1.view(num_dst, Wp),
        rows_to_tokens,
        num_dst,
        stream=torch.cuda.current_stream(),
    )

    if a1_scale_token_u8 is not None:
        Ws = a1_scale_token_u8.shape[1]
        if a1_scale_raw is None:
            a1_scale_raw = torch.zeros((E, max_m, Ws), dtype=torch.uint8, device=device)
        launch_s = _get_compiled_scatter_copy(Ws)
        launch_s(
            a1_scale_token_u8.contiguous().view(-1, Ws),
            a1_scale_raw.view(num_dst, Ws),
            rows_to_tokens,
            num_dst,
            stream=torch.cuda.current_stream(),
        )

    return grouped_a1, a1_scale_raw


@functools.cache
def _get_compiled_scatter_preshuffle_scale(
    row_bytes: int, wmma_rep: int, scale_k_per_tile: int, gather: bool = True
):
    """Compile and cache the WMMA-preshuffle scale kernel (with/without gather)."""
    from aiter.ops.flydsl.kernels.moe_scatter_copy_preshuffle_scale import (
        build_moe_scatter_copy_preshuffle_scale_module,
    )

    return build_moe_scatter_copy_preshuffle_scale_module(
        row_bytes, wmma_rep, scale_k_per_tile, gather=gather
    )


def flydsl_moe_scatter_preshuffle_scale(
    a1_scale_token_u8: torch.Tensor,  # (token_num, Ws) uint8
    rows_to_tokens: torch.Tensor,  # (E*max_m,) int32 grouped row -> token (-1 pad)
    E: int,
    max_m: int,
    *,
    wmma_rep: int,
    scale_k_per_tile: int,
    grouped_a1_scale: Optional[torch.Tensor] = None,  # (E, max_m//wmma_rep, Ws*wmma_rep)
):
    """Route-gather each token's e8m0 scale row AND preshuffle it into the WMMA
    layout in a single kernel pass -- fusing ``flydsl_moe_scatter_copy_token``'s
    scale copy with ``_grouped_a8w4_preshuffle_e8m0_scale``.

    ``max_m`` must be a multiple of ``wmma_rep*16`` (the grouped path pads it to
    a multiple of ``warp_tile_m``). Padding rows (``rows_to_tokens == -1``) are
    left untouched -- the masked GEMM never reads them, matching the previous
    uninitialized ``a1_scale_raw`` behaviour. Returns ``grouped_a1_scale``."""
    device = a1_scale_token_u8.device
    Ws = a1_scale_token_u8.shape[1]
    rows_per_tile = wmma_rep * 16
    assert max_m % rows_per_tile == 0, (
        f"max_m ({max_m}) must be a multiple of wmma_rep*16 ({rows_per_tile})"
    )
    tiles_per_expert = max_m // rows_per_tile

    if grouped_a1_scale is None:
        grouped_a1_scale = torch.empty(
            (E, max_m // wmma_rep, Ws * wmma_rep), dtype=torch.uint8, device=device
        )

    launch = _get_compiled_scatter_preshuffle_scale(
        Ws, wmma_rep, scale_k_per_tile, True
    )
    launch(
        a1_scale_token_u8.contiguous().view(-1, Ws),
        grouped_a1_scale.view(E * (max_m // wmma_rep), Ws * wmma_rep),
        rows_to_tokens,
        max_m,
        E,
        tiles_per_expert,
        stream=torch.cuda.current_stream(),
    )
    return grouped_a1_scale


def flydsl_moe_preshuffle_scale(
    scale_grouped_u8: torch.Tensor,  # (E, max_m, Ws) or (E*max_m, Ws) uint8
    E: int,
    max_m: int,
    *,
    wmma_rep: int,
    scale_k_per_tile: int,
    out: Optional[torch.Tensor] = None,  # (E, max_m//wmma_rep, Ws*wmma_rep)
):
    """Preshuffle an already-grouped row-major e8m0 scale into the WMMA layout in
    one kernel pass -- the in-kernel equivalent of the torch
    ``_grouped_a8w4_preshuffle_e8m0_scale`` permute (used by stage2, where the
    scale is already grouped so no route-gather is needed). Returns ``out``."""
    device = scale_grouped_u8.device
    Ws = scale_grouped_u8.shape[-1]
    rows_per_tile = wmma_rep * 16
    assert max_m % rows_per_tile == 0, (
        f"max_m ({max_m}) must be a multiple of wmma_rep*16 ({rows_per_tile})"
    )
    tiles_per_expert = max_m // rows_per_tile

    if out is None:
        out = torch.empty(
            (E, max_m // wmma_rep, Ws * wmma_rep), dtype=torch.uint8, device=device
        )

    launch = _get_compiled_scatter_preshuffle_scale(
        Ws, wmma_rep, scale_k_per_tile, False
    )
    launch(
        scale_grouped_u8.contiguous().view(E * max_m, Ws),
        out.view(E * (max_m // wmma_rep), Ws * wmma_rep),
        max_m,
        E,
        tiles_per_expert,
        stream=torch.cuda.current_stream(),
    )
    return out
