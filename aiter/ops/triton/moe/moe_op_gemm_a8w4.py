# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_ogs.py

import functools
import itertools
import os
import json
import torch
import triton
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
from aiter.ops.triton._triton_kernels.moe.moe_op_gemm_a8w4 import (
    _moe_gemm_a8w4 as _moe_gemm_a8w4_triton,
)
from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a8w4 import (
    _moe_gemm_a8w4_decode as _moe_gemm_a8w4_decode_gluon,
    _moe_gemm_a8w4_prefill as _moe_gemm_a8w4_prefill_gluon,
)
from aiter.ops.triton.moe.reduce import reduce_grouped
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.gemm_config_utils import pick_gemm_num_stages


@functools.lru_cache
def _get_a8w4_dispatch(arch: str) -> dict:
    """Per-(block_m, N, K) dispatch table for moe_gemm_a8w4. Returns {} if no
    tuned file is shipped for this arch (caller uses the safe-default fallback).
    Mirrors get_moe_configs() in utils/moe_config_utils.py."""
    fpath = f"{AITER_TRITON_CONFIGS_PATH}/moe/{arch}-A8W4.json"
    if os.path.exists(fpath):
        with open(fpath, "r") as f:
            return json.load(f)
    return {}


def can_overflow_int32(tensor: torch.Tensor):
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32


def should_upcast_indices(*args):
    return any(tensor is not None and can_overflow_int32(tensor) for tensor in args)


def allocate_output(
    M,
    N,
    out_dtype,
    reduction_n_matmul,
    reduction_n_reduction,
    routing_data,
    gather_indx,
    scatter_indx,
    block_m,
    split_k,
    device,
):
    # if the activations are gathered, then M is number of gather indices
    if gather_indx is not None:
        M = gather_indx.shape[0]
    # final output
    if routing_data.n_expts_act == 1 or scatter_indx is None:
        y_rows = M
    else:
        y_rows = (
            scatter_indx.shape[0] // routing_data.n_expts_act
        )  # compressed number of rows
    matmul_shape = (split_k, M, N // reduction_n_matmul)
    final_shape = (y_rows, N // reduction_n_matmul // reduction_n_reduction)
    matmul_output = torch.empty(matmul_shape, device=device, dtype=out_dtype)
    if scatter_indx is not None or split_k > 1:
        final_output = torch.empty(final_shape, device=device, dtype=out_dtype)
    else:
        final_output = None
    return matmul_output, final_output


def preshuffle_weights_gfx1250(w: torch.Tensor) -> torch.Tensor:
    assert w.ndim == 3, f"Expected 3D weight tensor (E, K, N), got {w.ndim}"
    E, K, N = w.shape
    assert K % 16 == 0, f"K ({K}) must be divisible by 16 for MFMA preshuffling"
    assert N % 16 == 0, f"N ({N}) must be divisible by 16 for MFMA preshuffling"

    w = w.transpose(1, 2)
    w = w.view(E, N // 16, 16, K // 16, 16)
    w = w.permute(0, 1, 3, 2, 4).contiguous()
    w = w.view(E, N // 16, K * 16)
    w = w.transpose(1, 2)

    return w


def get_kernel_config_triton(m, n, k, routing_data, swizzle_mx_scale=None):
    block_m = routing_data.block_m
    group_m = 4
    num_xcds = 8
    xcd_swizzle = num_xcds
    w_cache_modifier = ".cg" if block_m <= 32 else None
    arch = get_arch()
    split_k = 1

    # Tuned dispatch: per-(block_m, N, K) winners from a sweep tuner.
    # Schema mirrors sister files like gfx950-MOE-FP8_W8A8.json (BLOCK_SIZE_N,
    # BLOCK_SIZE_K, num_warps, …) except BLOCK_SIZE_M is omitted because block_m
    # is the dispatch key, not a tunable (routing decides block_m for the layer).
    tuned = _get_a8w4_dispatch(arch).get(f"bm{block_m}_n{n}_k{k}")
    if tuned is not None:
        return {
            "block_m": block_m,
            "block_n": tuned["BLOCK_SIZE_N"],
            "block_k": tuned["BLOCK_SIZE_K"],
            "num_warps": tuned["num_warps"],
            "num_stages": tuned["num_stages"],
            "group_m": group_m,
            "xcd_swizzle": xcd_swizzle,
            "w_cache_modifier": w_cache_modifier,
            "split_k": split_k,
            "waves_per_eu": tuned.get("waves_per_eu", 0),
            "matrix_instr_nonkdim": tuned.get("matrix_instr_nonkdim", 16),
            "kpack": tuned.get("kpack", 1),
        }

    # Fallback for shapes not in the tuned dispatch JSON.
    # Look for a tuned entry with the same (N, K) but any block_m — the tile
    # geometry and num_stages from that entry are a better starting point than
    # a generic default, and avoid regressing to num_stages=1 on gfx950.
    # Under CDNA4 swizzle, skip BLOCK_K<256 entries since unswizzle can't compile them.
    dispatch = _get_a8w4_dispatch(arch)
    proxy = next(
        (
            v
            for bm in (16, 32, 64, 128)
            if (v := dispatch.get(f"bm{bm}_n{n}_k{k}")) is not None
            and (swizzle_mx_scale != "CDNA4_SCALE" or v.get("BLOCK_SIZE_K", 0) >= 256)
        ),
        None,
    )
    if proxy is not None:
        return {
            "block_m": block_m,
            "block_n": proxy["BLOCK_SIZE_N"],
            "block_k": proxy["BLOCK_SIZE_K"],
            "num_warps": proxy["num_warps"],
            "num_stages": proxy["num_stages"],
            "group_m": group_m,
            "xcd_swizzle": xcd_swizzle,
            "w_cache_modifier": w_cache_modifier,
            "split_k": split_k,
            "waves_per_eu": proxy.get("waves_per_eu", 0),
            "matrix_instr_nonkdim": proxy.get("matrix_instr_nonkdim", 16),
            "kpack": proxy.get("kpack", 1),
        }

    # Last-resort: original shape-based heuristic, gated to gfx942 which has no
    # tuned JSON. Other arches fall back to a conservative safe default.
    block_k = 256
    num_stages = 2

    if arch == "gfx942":
        if block_m == 16:
            block_n = 128
            num_warps = 4
            grid_m = routing_data.n_blocks(m, block_m)
            grid_n = triton.cdiv(n, block_n)
            grid = grid_m * grid_n * split_k
            while block_n >= 64 and grid < 256:
                block_n = block_n // 2
                grid_m = routing_data.n_blocks(m, block_m)
                grid_n = triton.cdiv(n, block_n)
                grid = grid_m * grid_n * split_k
        elif block_m == 32:
            if n <= 1024:
                block_n = 128
                num_warps = 4
            else:
                block_n = 256
                num_warps = 8
        else:
            block_n = 128
            num_warps = 4 if block_m == 128 else 8
    elif arch == "gfx950":
        if block_m == 16:
            block_n = 128
            num_warps = 4

            grid_m = routing_data.n_blocks(m, block_m)
            grid_n = triton.cdiv(n, block_n)
            grid = grid_m * grid_n * split_k
            # Floor at 64 (was 32): out_mx_quant=True with apply_swiglu requires
            # OUT_BLOCK_N = BLOCK_N // 2 >= 32. Loop boundary changed to keep
            # block_n >= 64 for both MX and non-MX paths.
            while block_n >= 128 and grid < get_num_sms():
                block_n = block_n // 2
                grid_m = routing_data.n_blocks(m, block_m)
                grid_n = triton.cdiv(n, block_n)
                grid = grid_m * grid_n * split_k

            if k >= 512:
                block_k = 512

        elif block_m == 32:
            if n <= 1024:
                block_n = 128
                num_warps = 4
            elif n <= 4096:
                block_n = 256
                num_warps = 4
            else:
                block_n = 512
                num_warps = 4

        elif block_m == 64:
            # V4-Flash prefill-tuned (rocprof brute force v2): for block_m=64,
            # (bn=128, nw=4, ns=1) gives 2-4x speedup over the previous bn=512/nw=8
            # default on all four V4-Flash prefill shapes.
            block_n = 128
            num_warps = 4
            num_stages = 1

        else:
            # Cap by N: BN=512 wasted compute on small-N shapes (e.g. N=256
            # → 50% pad, grid_n=1). Tuned shapes bypass this via JSON.
            block_n = min(triton.next_power_of_2(n), 256)
            # routing caps block_m at 128; nw=4 wins ~2x at block_m=128 on gpt-oss
            # shapes (MI355X) but regresses ~7% at block_m=64, so 64 stays at 8.
            num_warps = 4 if block_m == 128 else 8

        # bits_a=8 (fp8), bits_b=4 (mxfp4). Picks ns=2 when the tile fits in LDS,
        # else falls back to ns=1. The previous hardcoded ns=1 silently regressed
        # gpt-oss W4A8 MoE shapes by 30-40% vs the JSON-tuned ns=2 winners; the
        # block_m==64 branch keeps its rocprof-tuned ns=1 override.
        if block_m != 64:
            num_stages = pick_gemm_num_stages(
                arch, block_m, block_n, block_k, 8, 4, use_async_padding=True
            )
    else:
        block_n = 128
        num_warps = 4

    return {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "group_m": group_m,
        "xcd_swizzle": xcd_swizzle,
        "w_cache_modifier": w_cache_modifier,
        "split_k": split_k,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "kpack": 1,
    }


def get_kernel_config_gluon(m, n, k, routing_data):
    block_m = routing_data.block_m
    num_xcds = 1
    w_cache_modifier = ".cg" if block_m <= 32 else None
    num_buffers = 3
    split_k = 1
    block_k = 512

    if block_m == 16:
        block_k = 512
        num_warps = 4
        if get_arch() == "gfx1250":
            # decode (block_m==16): NUM_BUFFERS=3 + block_n=128 restores the
            # software-pipelined expert GEMM; the #3504 decode kernel's
            # block_n=256/NUM_BUFFERS=1 regressed conc-1 decode on gfx1250.
            block_n = 128
            num_stages = 3
        else:
            block_n = 256
            num_stages = 1

    elif block_m == 32:
        if n <= 1024:
            block_n = 128
            num_warps = 4
        else:
            block_n = 256
            num_warps = 4

    else:
        block_n = 256
        block_k = 256
        num_warps = 4

    ret = {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_buffers": num_buffers,
        "xcd_swizzle": num_xcds,
        "split_k": split_k,
        "w_cache_modifier": w_cache_modifier,
        "waves_per_eu": 0,
    }
    return ret


def swizzle_scales_gfx950(data):
    NON_K_PRESHUFFLE_BLOCK_SIZE = 32
    block_shape = data.shape
    SCALE_K = block_shape[-2]
    N = block_shape[-1]
    data = data.transpose(-1, -2)
    data = data.view(-1, N // NON_K_PRESHUFFLE_BLOCK_SIZE, 2, 16, SCALE_K // 8, 2, 4, 1)
    data = data.permute(0, 1, 4, 6, 3, 5, 2, 7).contiguous()
    E = block_shape[0]
    data = data.reshape(E, N // 32, SCALE_K * 32)
    return data.transpose(-1, -2)


def swizzle_scales_gfx1250(data):
    E, K_SCALE, N = data.shape
    preshuffle_factor = 32
    num_chunk_n = N // preshuffle_factor
    SCALE_KWIDTH = 8
    num_chunk_k = K_SCALE // SCALE_KWIDTH

    data = data.transpose(-1, -2)
    data = data.view(E, num_chunk_n, 32, num_chunk_k, SCALE_KWIDTH)
    data = data.permute(0, 1, 3, 2, 4).contiguous()
    data = data.view(E, N // preshuffle_factor, K_SCALE * preshuffle_factor)
    data = data.transpose(-1, -2)

    return data


def swizzle_scales(data):
    """Arch-agnostic scale swizzle for moe_gemm_a8w4.

    Returns (swizzled_data, layout_string) where layout_string is the
    SWIZZLE_MX_SCALE value the kernel expects, or None for unknown arches.
    """
    arch = get_arch()
    if arch == "gfx1250":
        return swizzle_scales_gfx1250(data), "GFX1250_SCALE"
    elif arch == "gfx950":
        return swizzle_scales_gfx950(data), "CDNA4_SCALE"
    else:
        return data, None


# -----------------------------------------------------------------------------
# Triton Implementation
# -----------------------------------------------------------------------------


def moe_gemm_a8w4(
    x,
    w,
    x_scales,
    w_scales,
    x_static_scale=None,
    quant_static_scale=None,
    bias=None,
    routing_data: RoutingData | None = None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    swizzle_mx_scale=None,
    out_dtype=torch.bfloat16,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    swiglu_add_residual=True,
    preshuffled=False,
    unpadded_N=None,
    unpadded_K=None,
    # Idea 1: emit (fp8 e4m3, ue8m0 per-1×32 scale) directly from the GEMM
    # write-back. When out_mx_quant=True, returns (y_fp8, y_scale_ue8m0).
    # Requires SPLIT_K==1 and no scatter_indx (GEMM1-style).
    out_mx_quant: bool = False,
    # External residual to fold into reduce_grouped writeback (saves the
    # standalone routed+shared elementwise add).
    residual=None,
):
    """
    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])
    """
    use_gluon = get_arch() == "gfx1250"
    if preshuffled:
        assert (
            use_gluon
        ), "preshuffled weights are only supported by the gluon (gfx1250) kernel"
    assert w.stride(-2) == 1, "`w` must be column-major when it has data-type mxfp"
    x_has_mx = x_scales is not None
    if x_has_mx:
        assert x.stride(-1) == 1, "'x' must be row-major when it has data-type mxfp"
    if x_has_mx:
        stride_x_mx_m = x_scales.stride(0)
        stride_x_mx_k = x_scales.stride(1)
    else:
        stride_x_mx_m = 0
        stride_x_mx_k = 0
    # determine shapes
    num_tokens = x.shape[-2]
    M = num_tokens if gather_indx is None else gather_indx.shape[0]
    K, N = x.shape[-1], w.shape[-1]
    if preshuffled:
        # preshuffle layout is (E, K_packed*16, N//16); w.shape[-1] = N//16
        N = w.shape[-1] * 16
    # Output buffer must be sized to the PADDED N: the kernel writes full
    # block_n columns per tile (grid_n * block_n cols total), which can exceed
    # unpadded_N when block_n doesn't divide it evenly → OOB on the y buffer.
    padded_N = N
    block_m = routing_data.block_m
    if unpadded_N and block_m == 16:
        N = unpadded_N
    if unpadded_K and block_m == 16:
        K = unpadded_K
    if use_gluon:
        w = w.transpose(1, 2)
        w_scales = w_scales.transpose(1, 2)
    # compute optimization flags
    if use_gluon:
        config = get_kernel_config_gluon(M, N, K, routing_data)
    else:
        config = get_kernel_config_triton(M, N, K, routing_data, swizzle_mx_scale)
    # CDNA4 swizzle requires BLOCK_K % 256 == 0; some tuned small-K entries
    # pick BK<256 for utilization. Clamp only when swizzle is requested so
    # StridedLayout callers keep their tuned BK<256.
    if swizzle_mx_scale == "CDNA4_SCALE" and config["block_k"] < 256:
        config["block_k"] = 256
    # pad x_scales to a whole number of BLOCK_K tiles for async_copy
    if use_gluon and x_has_mx:
        mx_scale_block_k = config["block_k"] // 32
        padded_ks = triton.cdiv(K, config["block_k"]) * mx_scale_block_k
        if padded_ks > x_scales.shape[-1]:
            x_scales = torch.nn.functional.pad(
                x_scales, (0, padded_ks - x_scales.shape[-1])
            )
        stride_x_mx_m = x_scales.stride(0)
        stride_x_mx_k = x_scales.stride(1)
    if apply_swiglu and config["split_k"] > 1:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
        apply_swiglu_reduction = True
        reduction_n_reduction = 2
    elif apply_swiglu:
        apply_swiglu_matmul = True
        reduction_n_matmul = 2
        apply_swiglu_reduction = False
        reduction_n_reduction = 1
    else:
        apply_swiglu_matmul = False
        reduction_n_matmul = 1
        apply_swiglu_reduction = False
        reduction_n_reduction = 1
    # allocate output memory. With out_mx_quant=True, the kernel writes fp8 e4m3
    # into y; otherwise the requested out_dtype (bf16).
    if out_mx_quant:
        assert config["split_k"] == 1, "out_mx_quant requires split_k == 1"
        assert scatter_indx is None, (
            "out_mx_quant currently only supported for GEMM1-style (no scatter); "
            "scatter+combine would need fp8-aware reduce_grouped"
        )
        out_dtype = torch.float8_e4m3fn
    else:
        out_dtype = out_dtype
    y, y_final = allocate_output(
        M,
        padded_N,
        out_dtype,
        reduction_n_matmul,
        reduction_n_reduction,
        routing_data,
        gather_indx,
        scatter_indx,
        config["block_m"],
        config["split_k"],
        x.device,
    )
    # Companion ue8m0 scale buffer for the MXFP8 emit path.
    if out_mx_quant:
        n_out = w.shape[-1] // reduction_n_matmul  # post-swiglu width
        assert n_out % 32 == 0, "out_mx_quant requires N_out % 32 == 0"
        m_out = y.shape[-2]
        y_scale = torch.empty((m_out, n_out // 32), dtype=torch.uint8, device=x.device)
        stride_y_mx_m = y_scale.stride(0)
        stride_y_mx_n = y_scale.stride(1)
    else:
        y_scale = None
        stride_y_mx_m = 0
        stride_y_mx_n = 0
    stride_bias = None if bias is None else bias.stride(0)
    # moe metadata
    expt_data = routing_data.expt_data
    expt_hist = None if expt_data is None else expt_data.hist
    expt_hist_sum = None if expt_data is None else expt_data.token_offs_pad[-1]
    expt_token_offs_raw = None if expt_data is None else expt_data.token_offs_raw
    expt_block_pid_map = None if expt_data is None else expt_data.block_pid_map
    # pid grid
    grid_m = routing_data.n_blocks(M, config["block_m"])
    grid_n = triton.cdiv(N, config["block_n"])
    grid = grid_m * grid_n * config["split_k"]
    # launch kernel
    if use_gluon and block_m == 16:
        _moe_gemm_a8w4_decode_gluon[(grid,)](
            y,
            y.stride(1),
            y.stride(2),
            x,
            x.stride(0),
            x.stride(1),
            x_scales,
            stride_x_mx_m,
            stride_x_mx_k,
            w,
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w_scales,
            w_scales.stride(0),
            w_scales.stride(1),
            w_scales.stride(2),
            x_static_scale,
            quant_static_scale,
            bias,
            stride_bias,
            gammas,
            num_tokens,
            N,
            K,
            gather_indx,
            expt_hist,
            expt_token_offs_raw,
            expt_hist_sum,
            expt_block_pid_map,
            grid_m,
            grid_n,
            apply_swiglu_matmul,
            alpha,
            limit,
            reduction_n_matmul,
            swiglu_add_residual,
            routing_data.n_expts_act,
            config["block_m"],
            config["block_n"],
            config["block_k"],
            XCD_SWIZZLE=config["xcd_swizzle"],
            NUM_BUFFERS=config["num_buffers"],
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            PRESHUFFLED=preshuffled,
            MASK_K_LIMIT=K % config["block_k"],
            W_CACHE_MODIFIER=config["w_cache_modifier"],
            num_warps=config["num_warps"],
            UPCAST_INDICES=should_upcast_indices(x, w, y),
            waves_per_eu=config["waves_per_eu"],
        )
    elif use_gluon:
        _moe_gemm_a8w4_prefill_gluon[(grid,)](
            y,
            y.stride(1),
            y.stride(2),
            x,
            x.stride(0),
            x.stride(1),
            x_scales,
            stride_x_mx_m,
            stride_x_mx_k,
            w,
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w_scales,
            w_scales.stride(0),
            w_scales.stride(1),
            w_scales.stride(2),
            x_static_scale,
            quant_static_scale,
            bias,
            stride_bias,
            gammas,
            num_tokens,
            N,
            K,
            gather_indx,
            expt_hist,
            expt_token_offs_raw,
            expt_hist_sum,
            expt_block_pid_map,
            grid_m,
            grid_n,
            apply_swiglu_matmul,
            alpha,
            limit,
            reduction_n_matmul,
            swiglu_add_residual,
            routing_data.n_expts_act,
            config["block_m"],
            config["block_n"],
            config["block_k"],
            XCD_SWIZZLE=config["xcd_swizzle"],
            NUM_BUFFERS=config["num_buffers"],
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            PRESHUFFLED=preshuffled,
            EVEN_K=K % config["block_k"] == 0,
            MASK_K_LIMIT=K % config["block_k"],
            W_CACHE_MODIFIER=config["w_cache_modifier"],
            num_warps=config["num_warps"],
            UPCAST_INDICES=should_upcast_indices(x, w, y),
            waves_per_eu=config["waves_per_eu"],
        )
    else:
        _moe_gemm_a8w4_triton[(grid,)](
            y,
            y.stride(0),
            y.stride(1),
            y.stride(2),
            x,
            x.stride(0),
            x.stride(1),
            x_scales,
            stride_x_mx_m,
            stride_x_mx_k,
            w,
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w_scales,
            w_scales.stride(0),
            w_scales.stride(1),
            w_scales.stride(2),
            x_static_scale,
            quant_static_scale,
            bias,
            stride_bias,
            gammas,
            N,
            K,
            gather_indx,
            expt_hist,
            expt_token_offs_raw,
            expt_hist_sum,
            expt_block_pid_map,
            grid_m,
            grid_n,
            apply_swiglu_matmul,
            alpha,
            limit,
            reduction_n_matmul,
            swiglu_add_residual,
            routing_data.n_expts_act,
            config["block_m"],
            config["block_n"],
            config["block_k"],
            config["group_m"],
            XCD_SWIZZLE=config["xcd_swizzle"],
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            SPLIT_K=config["split_k"],
            EVEN_K=K % config["block_k"] == 0,
            MASK_K_LIMIT=K % config["block_k"],
            W_CACHE_MODIFIER=config["w_cache_modifier"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
            UPCAST_INDICES=should_upcast_indices(x, w, y),
            waves_per_eu=config["waves_per_eu"],
            matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
            kpack=config["kpack"],
            YMxScale=y_scale,
            stride_y_mx_m=stride_y_mx_m,
            stride_y_mx_n=stride_y_mx_n,
            HAS_MX_OUT=out_mx_quant,
        )

    # MXFP8 emit path: scatter_indx is None and split_k==1, so we bypass
    # reduce_grouped and return (fp8 values, ue8m0 scales) directly.
    if out_mx_quant:
        return y.squeeze(0), y_scale
    # Build grouped reduction inputs in a uniform way
    group_indx = (
        None
        if scatter_indx is None
        else scatter_indx.view(-1, routing_data.n_expts_act)
    )
    # Step 9: external residual fold-in is now wired into reduce_grouped.
    y_final = reduce_grouped(
        y,
        group_indx,
        y_final,
        apply_swiglu_reduction,
        alpha,
        limit,
        reduction_n_reduction,
        out_dtype=out_dtype,
        swiglu_add_residual=swiglu_add_residual,
        residual=residual,
    )
    return y_final


# -----------------------------------------------------------------------------
# Reference Implementation
# -----------------------------------------------------------------------------


def swiglu_torch(a, alpha, limit, add_residual=True):
    a_gelu = a[..., ::2]
    if limit is not None:
        a_gelu = a_gelu.clamp(max=limit)
    a_linear = a[..., 1::2]
    if limit is not None:
        a_linear = a_linear.clamp(min=-limit, max=limit)

    out_gelu = a_gelu * torch.sigmoid(alpha * a_gelu)
    if add_residual:
        out = out_gelu * (a_linear + 1)
    else:
        out = out_gelu * a_linear
    return out


def moe_gemm_torch(
    x,
    w,
    bias,
    routing_data: RoutingData = None,
    gather_indx=None,
    scatter_indx=None,
    gammas=None,
    apply_swiglu=False,
    alpha=1.0,
    limit=1.0,
    add_residual=True,
):
    assert x.dtype.itemsize > 1
    assert w.dtype.itemsize > 1
    if bias is not None and bias.ndim == 1:
        bias = bias.view(1, *bias.shape)
    if w.ndim == 2:
        w = w.view(1, *w.shape)
    n_expts_act = routing_data.n_expts_act
    # memory offsets
    if routing_data.n_expts_tot > 1:
        sizes = routing_data.expt_hist
        off = torch.zeros(sizes.shape[0] + 1, dtype=torch.int32)
        off[1:] = torch.cumsum(sizes, 0)
        offs = list(itertools.pairwise(off))
    else:
        offs = [[0, x.shape[0]] for _ in range(w.shape[0])]
    # compute
    n_rows = x.shape[0] if gather_indx is None else gather_indx.shape[0]
    n_cols = w.shape[-1] // 2 if apply_swiglu else w.shape[-1]
    y = torch.zeros((n_rows, n_cols), device=x.device, dtype=x.dtype)
    for i, (lo, hi) in enumerate(offs):
        if gather_indx is None:
            idx = torch.arange(lo, hi, device=x.device)
        else:
            gather_indx = gather_indx.to(torch.int32)
            idx = gather_indx[lo:hi] // n_expts_act
        out = torch.matmul(x[idx, :].float(), w[i].float())
        if bias is not None:
            out += bias[i, :]
        if apply_swiglu:
            out = swiglu_torch(out, alpha, limit, add_residual)
        if gammas is not None:
            out *= gammas[lo:hi, None]
        y[lo:hi, :] = out
    if scatter_indx is None:
        return y
    # accumulate output from all experts
    scatter_indx = scatter_indx.to(torch.int32)
    n_rows = y.shape[0] // n_expts_act
    out = torch.zeros((n_rows, y.shape[-1]), dtype=torch.float32, device=x.device)
    src_idx = scatter_indx.view(-1, n_expts_act)
    for i in range(n_rows):
        out[i, :] = y[src_idx[i], :].float().sum(0)

    return out


def main():
    import argparse
    from aiter.ops.triton.moe.moe_routing.routing import routing
    from aiter.ops.triton.moe.quant_moe import (
        downcast_to_static_fp8,
        downcast_to_mxfp,
        upcast_from_mxfp,
    )

    parser = argparse.ArgumentParser(description="Run MoE GEMM A8W4 test")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=2880)
    # parser.add_argument("--M", type=int, default=32)
    # parser.add_argument("--N", type=int, default=5760)
    # parser.add_argument("--K", type=int, default=2880)
    # parser.add_argument("--K", type=int, default=512)
    # parser.add_argument("--N", type=int, default=5760)
    # parser.add_argument("--K", type=int, default=2880)
    # parser.add_argument("--N", type=int, default=6144)
    # parser.add_argument("--K", type=int, default=3072)
    parser.add_argument("--E", type=int, default=1, help="Total experts")
    parser.add_argument("--n_expts_act", type=int, default=1, help="Active experts per token")
    parser.add_argument(
        "--do_gather", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--do_scatter", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--has_y_gammas", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--apply_swiglu", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--fused_quant", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--hbm_swizzling", action=argparse.BooleanOptionalAction, default=False,
        help="Enable HBM scale swizzling (default: False).",
    )
    parser.add_argument(
        "--mxfp8_act", action=argparse.BooleanOptionalAction, default=True,
        help="Use mxfp8 microscaled activation instead of static fp8 (default: False).",
    )
    # PRESHUFFLE
    parser.add_argument(
        "--preshuffled", action=argparse.BooleanOptionalAction, default=True,
        help="Use preshuffled weights instead of shuffled weights (default: False).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device is not available")

    arch = get_arch()
    assert arch in ("gfx950", "gfx1250"), (
        f"a8w4 kernel requires gfx950 or gfx1250, got {arch}"
    )

    if args.hbm_swizzling:
        if arch == "gfx950" and (args.N % 32 != 0 or args.K % (32 * 8) != 0):
            raise ValueError(
                f"Shape {args.M}x{args.N}x{args.K} not supported for scale swizzling on gfx950"
            )
        if arch == "gfx1250" and (args.N % 128 != 0 or args.K % (32 * 4) != 0):
            raise ValueError(
                f"Shape {args.M}x{args.N}x{args.K} not supported for scale swizzling on gfx1250"
            )

    print("Testing MoE GEMM A8W4 kernel")
    print(
        f"  M={args.M}, K={args.K}, N={args.N}, E={args.E}, "
        f"n_expts_act={args.n_expts_act}"
    )
    print(
        f"  Flags: gather={args.do_gather}, scatter={args.do_scatter}, "
        f"swiglu={args.apply_swiglu}, fused_quant={args.fused_quant}, "
        f"gammas={args.has_y_gammas}, hbm_swizzling={args.hbm_swizzling}, "
        f"mxfp8_act={args.mxfp8_act}"
    )
    print(f"  Device: {device}, Architecture: {arch}")

    logits = torch.randn((args.M, args.E), dtype=torch.float16, device=device)
    routing_data, gather_idx, scatter_idx = routing(logits, args.n_expts_act)

    config = get_kernel_config_gluon(args.M, args.N, args.K, routing_data)
    pipeline = "decode" if config["block_m"] == 16 else "prefill"
    print(
        f"  Config: block_m={config['block_m']}, block_n={config['block_n']}, "
        f"block_k={config['block_k']}, num_warps={config['num_warps']}, "
        f"num_buffers={config['num_buffers']}"
    )
    print(f"  Pipeline: {pipeline} (gluon)" if arch == "gfx1250" else "  Pipeline: triton")
    routing_data.gate_scal = None
    gather_idx = gather_idx if args.do_gather else None
    scatter_idx = scatter_idx if args.do_scatter else None

    in_m = args.M * (args.n_expts_act if gather_idx is None else 1)

    x_bf16 = torch.randn((in_m, args.K), dtype=torch.bfloat16, device=device) / 10
    w_bf16 = torch.randn((args.E, args.K, args.N), dtype=torch.bfloat16, device=device) / 10
    bias = torch.randn((args.E, args.N), dtype=torch.float32, device=device)
    gammas = (
        2 ** torch.randint(
            -5, 0, (args.M * args.n_expts_act,), device=device, dtype=torch.float32
        )
        if args.has_y_gammas
        else None
    )

    w_tri, w_scale_tri = downcast_to_mxfp(w_bf16, torch.uint8, axis=1)
    w_ref = upcast_from_mxfp(w_tri, w_scale_tri, torch.bfloat16, axis=1)
    if args.preshuffled:
        w_tri = preshuffle_weights_gfx1250(w_tri)

    swizzle_mx_scale = None
    if args.hbm_swizzling:
        if arch == "gfx1250":
            swizzle_mx_scale = "GFX1250_SCALE"
            w_scale_tri = swizzle_scales_gfx1250(w_scale_tri)
        else:
            swizzle_mx_scale = "CDNA4_SCALE"
            w_scale_tri = swizzle_scales_gfx950(w_scale_tri)

    if args.mxfp8_act:
        x_tri, x_mx_scales = downcast_to_mxfp(x_bf16, torch.float8_e4m3fn, axis=1)
        x_ref = upcast_from_mxfp(x_tri, x_mx_scales, torch.bfloat16, axis=1)
        x_static_scale = None
    else:
        x_mx_scales = None
        x_static_scale = x_bf16.abs().max().float() / 448.0
        x_tri = downcast_to_static_fp8(x_bf16, x_static_scale)
        x_ref = x_bf16.clone()

    ref_y = moe_gemm_torch(
        x_ref, w_ref, bias.clone(), routing_data, gather_idx, scatter_idx,
        gammas, args.apply_swiglu,
    )

    block_k = config['block_k']
    block_n = config['block_n']
    K_padded = None
    N_padded = None
    # K_padded = 3072
    # N_padded = args.N

    # xK = x_tri.shape[1]
    # x_pad = torch.zeros((in_m, K_padded), dtype=x_tri.dtype, device=device)
    # x_pad[:, :xK] = x_tri
    # x_tri = x_pad[:, :xK]

    # wK, wN = w_tri.shape[1], w_tri.shape[2]
    # w_pad = torch.zeros((args.E, N_padded, K_padded // 2), dtype=w_tri.dtype, device=device)
    # w_pad = w_pad.transpose(1, 2)
    # w_pad[:, :wK, :wN] = w_tri
    # w_tri = w_pad[:, :wK, :wN]

    # sK, sN = w_scale_tri.shape[1], w_scale_tri.shape[2]
    # ws_pad = torch.zeros((args.E, N_padded, sK), dtype=w_scale_tri.dtype, device=device)
    # ws_pad = ws_pad.transpose(1, 2)
    # ws_pad[:, :sK, :sN] = w_scale_tri
    # w_scale_tri = ws_pad[:, :sK, :sN]

    # bias_pad = torch.zeros((args.E, N_padded), dtype=bias.dtype, device=device)
    # bias_pad[:, :args.N] = bias
    # bias = bias_pad[:, :args.N]

    # print(f"  Stride padding: K {args.K}->{K_padded}, N {args.N}->{N_padded}")
    # print(f"  x_tri       shape={tuple(x_tri.shape)}  stride={x_tri.stride()}")
    # print(f"  w_tri       shape={tuple(w_tri.shape)}  stride={w_tri.stride()}")
    # print(f"  w_scale_tri shape={tuple(w_scale_tri.shape)}  stride={w_scale_tri.stride()}")
    # print(f"  bias        shape={tuple(bias.shape)}  stride={bias.stride()}")

    quant_static_scale = None
    out_dtype = torch.bfloat16
    if args.fused_quant:
        quant_static_scale = ref_y.abs().max().float() / 448.0
        out_dtype = torch.float8_e4m3fn

    print("Preshuffled:", args.preshuffled)
    tri_y = moe_gemm_a8w4(
        x_tri,
        w_tri,
        x_mx_scales,
        w_scale_tri,
        x_static_scale,
        quant_static_scale,
        bias,
        routing_data,
        gather_idx,
        scatter_idx,
        gammas,
        swizzle_mx_scale,
        out_dtype,
        args.apply_swiglu,
        preshuffled=args.preshuffled,
    )
    if args.fused_quant:
        tri_y = (tri_y.float() * quant_static_scale).to(ref_y.dtype)

    print(f"  ref_y shape={tuple(ref_y.shape)} dtype={ref_y.dtype} min={ref_y.float().min().item():.4f} max={ref_y.float().max().item():.4f} nan={ref_y.isnan().sum().item()} inf={ref_y.isinf().sum().item()}")
    print(f"  tri_y shape={tuple(tri_y.shape)} dtype={tri_y.dtype} min={tri_y.float().min().item():.4f} max={tri_y.float().max().item():.4f} nan={tri_y.isnan().sum().item()} inf={tri_y.isinf().sum().item()}")
    print(f"  ref_y[:4,:4]=\n{ref_y[:4,:4]}")
    print(f"  tri_y[:4,:4]=\n{tri_y[:4,:4]}")

    ref_f = ref_y.to(torch.float32).detach()
    tri_f = tri_y.to(torch.float32).detach()
    eps = 1.0e-30
    multiplier = 1.0 / (torch.max(torch.abs(ref_f)) + eps)
    refn = ref_f * multiplier
    trin = tri_f * multiplier
    ref_rms = torch.sqrt(torch.square(refn).mean()) + eps
    rel_err = torch.abs(refn - trin) / torch.maximum(ref_rms, torch.abs(refn))
    max_err = torch.max(rel_err).item()
    rms_err = torch.sqrt(torch.square(rel_err).mean()).item()

    maxtol, rmstol = 4e-1, 4e-2
    print(f"maximum relative error = {max_err} (threshold = {maxtol})")
    print(f"RMS relative error = {rms_err} (threshold = {rmstol})")
    if max_err > maxtol or rms_err > rmstol:
        raise AssertionError("Wrapper test failed against reference")
    print(f"Test completed successfully ({pipeline} pipeline)")
    return 0


if __name__ == "__main__":
    main()