# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL MOE kernel management: naming, compilation, and high-level API."""

import functools
import os
import re

from typing import Dict, Optional
from aiter.utility import dtypes

import flydsl.compiler as flyc
import torch

_KERNEL_PARAMS: Dict[str, Dict] = {}

_SUFFIX_RE = re.compile(r"(?P<fq>_fq)?(?:_sbm(?P<sbm>\d+))?$")
_KERNEL_NAME_RE = re.compile(
    r"^flydsl_moe(?P<stage>[12])_a(?P<a>.+?)_w(?P<b>.+?)_"
    r"(?P<out>bf16|f16)_t(?P<tm>\d+)x(?P<tn>\d+)x(?P<tk>\d+)"
    r"(?:_(?P<rest>.*))?$"
)

_MFMA16_ALIASES = {"16", "16x16", "16x16x128", "mfma16", "mfma16k128"}
_MFMA32_ALIASES = {"32", "32x32", "32x32x64", "mfma32", "mfma32k64"}


def _stage2_mfma_variant_tag(tile_k: int, a_dtype: str, b_dtype: str) -> str:
    """Return the FP4 stage2 MFMA variant tag needed in kernel names."""
    if a_dtype != "fp4" or b_dtype != "fp4":
        return ""
    if int(tile_k) == 128:
        return "mfma32k64"

    variant = os.environ.get("FLIR_MOE_STAGE2_MFMA", "16x16x128").strip().lower()
    if not variant or variant in _MFMA16_ALIASES:
        return ""
    if variant in _MFMA32_ALIASES:
        return "mfma32k64"
    raise ValueError(
        "FLIR_MOE_STAGE2_MFMA must be '16x16x128' or '32x32x64', "
        f"got {variant!r}"
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


def _parse_flydsl_kernel_name(name: str) -> Optional[Dict]:
    match = _KERNEL_NAME_RE.match(name)
    if match is None:
        return None

    stage = int(match.group("stage"))
    a_dtype = match.group("a")
    b_dtype = match.group("b")
    params: Dict = {
        "stage": stage,
        "a_dtype": a_dtype,
        "b_dtype": b_dtype,
        "out_dtype": match.group("out"),
        "tile_m": int(match.group("tm")),
        "tile_n": int(match.group("tn")),
        "tile_k": int(match.group("tk")),
        "MPerBlock": int(match.group("tm")),
        "use_async_copy": False,
        "waves_per_eu": 1 if b_dtype == "fp4" else 0,
        "b_nt": 2,
    }
    if stage == 1:
        params.update({"k_batch": 1, "gate_only": False})
    else:
        params.update({"mode": "atomic", "sort_block_m": 0, "persist": False})

    for token in (match.group("rest") or "").split("_"):
        if not token:
            continue
        if token == "async":
            params["use_async_copy"] = True
        elif stage == 2 and token in ("atomic", "reduce"):
            params["mode"] = token
        elif token in ("mfma16k128", "mfma32k64"):
            params["mfma_variant"] = token
        elif token.startswith("w") and token[1:].isdigit():
            params["waves_per_eu"] = int(token[1:])
        elif token.startswith("bnt") and token[3:].isdigit():
            params["b_nt"] = int(token[3:])
        elif stage == 1 and token.startswith("kb") and token[2:].isdigit():
            params["k_batch"] = int(token[2:])
        elif stage == 1 and token == "go":
            params["gate_only"] = True
        elif stage == 1 and token == "fq":
            params["fuse_fp4_quant"] = True
        elif stage == 2 and token == "persist":
            params["persist"] = True
        elif token.startswith("sbm") and token[3:].isdigit():
            params["sort_block_m"] = int(token[3:])
        elif token == "fq":
            params["fuse_fp4_quant"] = True
        else:
            return None
    return params


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
    return _parse_flydsl_kernel_name(name)


def _x_load_supported(
    a_dtype: str,
    tile_m: int,
    tile_n: int,
    tile_k: int,
    use_async_copy: bool,
) -> bool:
    """Mirror X-load compile-time constraints before creating tune tasks."""
    elem_bytes = 2 if a_dtype in ("fp16", "bf16", "int4_bf16") else 1
    total_threads = (tile_n // 32) * 64
    bytes_x_per_tile = tile_m * tile_k * elem_bytes
    if bytes_x_per_tile % total_threads != 0:
        return False

    bytes_per_thread_x = bytes_x_per_tile // total_threads
    if a_dtype in ("fp16", "bf16", "int4_bf16"):
        return bytes_per_thread_x % 16 == 0
    if use_async_copy:
        return bytes_per_thread_x % 16 == 0
    return bytes_per_thread_x % 4 == 0


def get_flydsl_stage1_kernels(
    a_dtype: str, b_dtype: str, out_dtype: str, model_dim: Optional[int] = None
) -> Dict[str, Dict]:
    """Return {kernelName: params} for all supported stage1 configs."""
    kernels = {}
    is_fp4 = b_dtype == "fp4"

    tile_ns = [32, 64, 128, 256] if is_fp4 else [128, 256]
    tile_ks = [256] if is_fp4 else [128, 256]
    tile_ms = [16, 32, 64, 128] if is_fp4 else [32, 64, 128]
    waves_per_eus = [1, 2, 3, 4] if is_fp4 else [0, 1, 2, 3, 4]
    k_batches = [1, 2, 4, 8, 16] if is_fp4 else [1]
    b_nts = [0, 2] if is_fp4 else [0, 2]
    # Stage1: fp4 (mixed) and non-fp4 (moe_gemm_2stage) both honor ``use_async_copy``.
    async_copies = [False, True]

    for tm in tile_ms:
        if is_fp4:
            if tm in [16, 32]:
                tile_ns = [32, 64, 128]
            else:
                tile_ns = [64, 128, 256]
        for tn in tile_ns:
            for tk in tile_ks:
                for async_copy in async_copies:
                    if not _x_load_supported(a_dtype, tm, tn, tk, async_copy):
                        continue
                    for wpe in waves_per_eus:
                        for kb in k_batches if wpe == 3 else [1]:
                            if is_fp4:
                                if tk == 512 and kb == 16:
                                    continue
                                if tn // 32 * 64 > 256:
                                    continue
                            # Split-K stage1 requires:
                            #   model_dim % k_batch == 0 and
                            #   (model_dim // k_batch) % tile_k == 0
                            # When model_dim is provided by the tuner, skip
                            # kernels that can never compile for that shape.
                            if model_dim is not None and kb > 1:
                                if model_dim % kb != 0:
                                    continue
                                if (model_dim // kb) % tk != 0:
                                    continue
                            gate_onlys = [False, True] if kb > 1 and is_fp4 else [False]
                            for bnt in b_nts:
                                for go in gate_onlys:
                                    name = flydsl_kernel_name(
                                        1, a_dtype, b_dtype, out_dtype, tm, tn, tk
                                    )
                                    if async_copy:
                                        name += "_async"
                                    if is_fp4 and wpe != 1:
                                        name += f"_w{wpe}"
                                    elif not is_fp4 and wpe > 0:
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
                                        "use_async_copy": async_copy,
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
    tile_ns = [128, 256] if is_fp4 else [128, 256]
    tile_ks = [128, 256] if is_fp4 else [128, 256]
    tile_ms = [16, 32, 64, 128] if is_fp4 else [32, 64, 128]
    modes = ["atomic", "reduce"]
    waves_per_eus = [0] if is_fp4 else [0, 1, 2, 3, 4]
    b_nts = [0] if is_fp4 else [0, 2]
    async_copies = [False, True]

    for tm in tile_ms:
        for tn in tile_ns:
            for tk in tile_ks:
                mfma_variant_tag = _stage2_mfma_variant_tag(tk, a_dtype, b_dtype)
                if is_fp4 and tk == 128:
                    # tile_k=128 uses the dedicated MFMA32/K64 stage2 path.
                    # Keep the candidate set deliberately small until layout
                    # coverage is broadened.
                    if a_dtype != "fp4" or tm not in (32, 64) or tn != 128:
                        continue
                for mode in modes:
                    for async_copy in async_copies:
                        if is_fp4 and tk == 128 and async_copy:
                            continue
                        if not _x_load_supported(a_dtype, tm, tn, tk, async_copy):
                            continue
                        for wpe in waves_per_eus:
                            for bnt in b_nts:
                                base_name = flydsl_kernel_name(
                                    2,
                                    a_dtype,
                                    b_dtype,
                                    out_dtype,
                                    tm,
                                    tn,
                                    tk,
                                    mode,
                                )
                                if async_copy:
                                    base_name += "_async"
                                if mfma_variant_tag:
                                    base_name += f"_{mfma_variant_tag}"
                                if is_fp4 and wpe != 1:
                                    base_name += f"_w{wpe}"
                                elif not is_fp4 and wpe > 0:
                                    base_name += f"_w{wpe}"
                                if bnt != 2:
                                    base_name += f"_bnt{bnt}"
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
                                    "use_async_copy": async_copy,
                                    "waves_per_eu": wpe,
                                    "b_nt": bnt,
                                }
                                if mfma_variant_tag:
                                    base_params["mfma_variant"] = mfma_variant_tag
                                kernels[base_name] = base_params
                                # # Persistent variant: round-robin over M tiles, grid_y=cu_num.
                                # if is_fp4:
                                #     kernels[base_name + "_persist"] = {
                                #         **base_params,
                                #         "persist": True,
                                #     }
    return kernels


def _register_all_configs():
    """Pre-populate _KERNEL_PARAMS with all supported configs at import time."""
    for a in ("fp8", "fp4", "fp16"):
        for b in ("fp4",):
            for out in ("bf16", "f16"):
                _KERNEL_PARAMS.update(get_flydsl_stage1_kernels(a, b, out))
                _KERNEL_PARAMS.update(get_flydsl_stage2_kernels(a, b, out))
    for a in ("fp8", "int8"):
        for b in (a,):
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
    use_g1u1: bool = True,
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
            use_g1u1=use_g1u1,
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
            act=act,
            use_g1u1=use_g1u1,
            use_async_copy=use_async_copy,
            waves_per_eu=waves_per_eu,
            b_nt=b_nt,
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
    use_async_copy: bool = False,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    mfma_variant: Optional[str] = None,
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
            use_async_copy=use_async_copy,
            mfma_variant=mfma_variant,
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
            use_async_copy=use_async_copy,
            waves_per_eu=waves_per_eu,
            b_nt=b_nt,
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
):
    empty_f32 = torch.empty(0, device=dev, dtype=torch.float32)
    return (
        _view_safe(out),
        _view_safe(a),
        _view_safe(w),
        _view_safe(a_scale),
        _view_safe(w_scale),
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        empty_f32,
        out_scale_sorted,
        token_num,
        n_in,
        k_in,
        size_expert_ids_in,
        torch.cuda.current_stream(),
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
):
    return (
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
        torch.cuda.current_stream(),
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
):
    empty_f32 = torch.empty(0, device=dev, dtype=torch.float32)
    return (
        _view_safe(target),
        _view_safe(a),
        _view_safe(w),
        _view_safe(a_scale),
        _view_safe(w_scale),
        sorted_ids,
        sorted_expert_ids,
        sorted_weights,
        num_valid_ids,
        empty_f32,
        token_num,
        n_in,
        k_in,
        blocks,
        torch.cuda.current_stream(),
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
):
    return (
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
        torch.cuda.current_stream(),
    )


def _run_compiled(exe, args):
    """First call: ``flyc.compile(exe, *args)`` compiles **and** executes the kernel.
    Subsequent calls: fast dispatch via the cached ``CompiledFunction``.
    """
    # flydsl>=0.1.2 exposes flydsl.compiler.compile; older versions (e.g.
    # 0.1.1.dev409) only provide jit wrappers and execute via exe(*args).
    if not hasattr(flyc, "compile"):
        exe(*args)
        return

    cf = getattr(exe, "_aiter_cf", None)
    if cf is None:
        cf = flyc.compile(exe, *args)
        exe._aiter_cf = cf
    else:
        cf(*args)


@functools.cache
def _get_compiled_silu_fq(inter_dim: int, topk: int):
    """Compile and cache the fused silu_and_mul + mxfp4 quant + scale-sort kernel."""
    from aiter.ops.flydsl.kernels.silu_and_mul_fq import build_silu_and_mul_fq_module

    return build_silu_and_mul_fq_module(inter_dim, topk)


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
    use_g1u1: Optional[bool] = None,
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
    """Fused MOE stage1 GEMM.

    a: (token_num, model_dim)
    w1:
      - g1u1: (E, 2*inter_dim, model_dim) pre-shuffled
      - g1u0: (E, inter_dim, model_dim) pre-shuffled
    For fp4 stage1, `w1`/`w1_scale` must use the same preshuffle layout as
    `shuffle_weight(..., (16, 16))` and `e8m0_shuffle(...)`.

    When fuse_sort_scale=True, the kernel writes e8m0 scales in sorted tiled
    layout directly, avoiding a separate moe_mxfp4_sort call.

    When k_batch>1 (split-K), the kernel writes partial sums via atomics into
    a zeroed buffer, then runs post-activation:
      - g1u1: reduce gate/up partials with activation+mul
      - g1u0: reduce proj partials then apply unary activation

    When gate_only=True (requires use_g1u1 and k_batch>1), each workgroup computes only
    one B-tile stream (no gate/up interleaving).  The grid X doubles so
    that by_n naturally covers both gate and up regions.

    Returns:
        Basic:                      out
        fuse_sort_scale:            (out, out_scale_sorted)
    """
    token_num = a.shape[0]
    E = w1.shape[0]

    if use_g1u1 is None:
        logical_out_inter_dim = None
        if out is not None:
            logical_out_inter_dim = out.shape[-1] * (2 if fuse_fp4_quant else 1)
        if logical_out_inter_dim is not None:
            if w1.shape[1] == 2 * logical_out_inter_dim:
                use_g1u1 = True
                inter_dim = logical_out_inter_dim
            elif w1.shape[1] == logical_out_inter_dim:
                use_g1u1 = False
                inter_dim = logical_out_inter_dim
            else:
                raise ValueError(
                    f"Unable to infer g1u mode from w1.shape={tuple(w1.shape)} "
                    f"and out.shape={tuple(out.shape)}"
                )
        else:
            # Preserve the historical direct-call behavior: when the caller does
            # not provide enough shape information, assume the legacy g1u1 path.
            if (w1.shape[1] % 2) != 0:
                raise ValueError(
                    f"Unable to infer g1u1 inter_dim from odd w1.shape[1]={w1.shape[1]}"
                )
            use_g1u1 = True
            inter_dim = w1.shape[1] // 2
    else:
        use_g1u1 = bool(use_g1u1)
        if use_g1u1:
            if (w1.shape[1] % 2) != 0:
                raise ValueError(
                    f"g1u1 stage1 expects w1.shape[1] to be even, got {w1.shape[1]}"
                )
            inter_dim = w1.shape[1] // 2
        else:
            inter_dim = w1.shape[1]

    if act == "swiglu" and not use_g1u1:
        raise ValueError("swiglu stage1 requires use_g1u1=True")

    model_dim = a.shape[1]

    if a_dtype == "fp4":
        model_dim = model_dim * 2

    torch_out_dtype = (
        dtypes.fp4x2
        if fuse_fp4_quant
        else dtypes.bf16 if out_dtype == "bf16" else dtypes.fp16
    )
    _is_splitk = k_batch > 1

    dev = a.device
    _splitk_fq = _is_splitk and fuse_fp4_quant

    _splitk_out_cols = inter_dim * (2 if use_g1u1 else 1)
    if _splitk_fq and not use_g1u1:
        raise ValueError("split-K fused fp4 quant currently requires use_g1u1=True")
    if _splitk_fq and act not in ("silu", "swiglu"):
        raise ValueError("split-K fused fp4 quant only supports silu/swiglu stage1")

    if out is None:
        if fuse_fp4_quant:
            out = torch.empty(
                (token_num, topk, inter_dim // 2), dtype=torch_out_dtype, device=dev
            )
        else:
            out = torch.empty(
                (token_num, topk, inter_dim), dtype=torch_out_dtype, device=dev
            )

    if _is_splitk:
        torch_tmp_out_dtype = dtypes.bf16 if out_dtype == "bf16" else dtypes.fp16
        tmp_out = torch.zeros(
            (token_num, topk, _splitk_out_cols), dtype=torch_tmp_out_dtype, device=dev
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

    _kernel_out = tmp_out if _is_splitk else out
    is_fp4 = b_dtype == "fp4"
    _n_in = inter_dim * (2 if (is_fp4 and use_g1u1) else 1)
    # _n_in is used by kernel launch to compute gx.
    # Keep historical behavior:
    # - use_g1u1=True (gated): fp4 path uses 2*inter_dim, non-fp4 uses inter_dim
    # - use_g1u1=False  (non-gated): always inter_dim
    if use_g1u1:
        _n_in = inter_dim * 2 if is_fp4 else inter_dim
    else:
        _n_in = inter_dim
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
        out_dtype=out_dtype,
        act=act,
        use_g1u1=use_g1u1,
        persist_m=_persist_m,
        fuse_fp4_quant=_gemm_fq,
        fuse_sort_scale=_gemm_fss,
        use_async_copy=use_async_copy,
        k_batch=k_batch,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        gate_only=gate_only,
    )
    _run_compiled(exe, args)

    if _splitk_fq:
        _silu_fq = _get_compiled_silu_fq(inter_dim, topk)
        num_sorted_rows = sorted_token_ids.shape[0]
        _run_compiled(
            _silu_fq,
            (
                tmp_out.view(-1, inter_dim * 2),
                out.view(-1).view(torch.uint8),
                out_scale_sorted_flat,
                sorted_token_ids,
                num_valid_ids,
                token_num,
                num_sorted_rows,
                torch.cuda.current_stream(),
            ),
        )
    elif _is_splitk:
        if use_g1u1:
            if act == "gelu":
                from aiter.ops.activation import gelu_and_mul

                gelu_and_mul(out.view(-1, inter_dim), tmp_out.view(-1, inter_dim * 2))
            else:
                from aiter.ops.activation import silu_and_mul

                silu_and_mul(out.view(-1, inter_dim), tmp_out.view(-1, inter_dim * 2))
        else:
            tmp_view = tmp_out.view(-1, inter_dim).to(torch.float32)
            if act == "gelu":
                out.copy_(
                    torch.nn.functional.gelu(tmp_view, approximate="tanh")
                    .to(out.dtype)
                    .view_as(out)
                )
            else:
                out.copy_(
                    torch.nn.functional.silu(tmp_view).to(out.dtype).view_as(out)
                )

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
    use_async_copy: bool = False,
    waves_per_eu: int = 3,
    b_nt: int = 2,
    mfma_variant: Optional[str] = None,
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
    # if persist is True:
    #     _persist_m = -1
    # elif persist is False:
    #     # _persist_m = 4 if m_blocks > 256 else 1
    #     _persist_m = 1  # _persist_m = 1 is better for g1u0
    # else:
    #     _persist_m = -1 if m_blocks > 256 else 1
    _persist_m = 1

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
        use_async_copy=use_async_copy,
        waves_per_eu=waves_per_eu,
        b_nt=b_nt,
        mfma_variant=mfma_variant,
    )
    _run_compiled(exe, args)

    if not accumulate:
        torch.sum(target.view(token_num, topk, model_dim), dim=1, out=out)

    return out
