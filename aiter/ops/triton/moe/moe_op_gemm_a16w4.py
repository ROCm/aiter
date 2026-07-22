# adapted from triton_kernels package
# original code https://github.com/triton-lang/triton/blob/main/python/triton_kernels/triton_kernels/matmul_details/_matmul.py

import functools
import itertools
import json
import os
from typing import Optional
import torch
import triton
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
from aiter.ops.triton._triton_kernels.moe.moe_op_gemm_a16w4 import (
    _moe_gemm_a16w4_triton,
)
from aiter.ops.triton._gluon_kernels.gfx1250.moe.moe_op_gemm_a16w4 import (
    _moe_gemm_a16w4_gluon_stage1,
    _moe_gemm_a16w4_gluon_stage2,
    _moe_gemm_a16w4_gluon_stage3,
)
from aiter.ops.triton.moe.reduce import reduce_grouped
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH

# -----------------------------------------------------------------------------
#                    Matrix Multiplication + Outer Gather/Scatter
# -----------------------------------------------------------------------------


def can_overflow_int32(tensor: torch.Tensor):
    max_int32 = (1 << 31) - 1
    offset = 0
    for i in range(tensor.ndim):
        offset += (tensor.shape[i] - 1) * tensor.stride(i)
    return offset > max_int32


def should_upcast_indices(*args):
    return any(tensor is not None and can_overflow_int32(tensor) for tensor in args)


def allocate_output(
    x,
    w,
    out_dtype,
    reduction_n_matmul,
    reduction_n_reduction,
    routing_data,
    gather_indx,
    scatter_indx,
    block_m,
    split_k,
):
    # ---- output ------
    N = w.shape[-1]
    # by default - M is number of rows in the activations
    M = x.shape[-2]
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
    matmul_output = torch.empty(matmul_shape, device=x.device, dtype=out_dtype)
    if scatter_indx is not None or split_k > 1:
        final_output = torch.empty(final_shape, device=x.device, dtype=out_dtype)
    else:
        final_output = None
    return matmul_output, final_output


def get_kernel_config_triton(m, n, k, routing_data):
    """Functional (non-tuned) default triton config; fallback when no tuned JSON
    entry exists (tuned configs live in configs/moe). Safe for any (block_m, N, K):
    the 128-wide N and 256-deep K tiles are masked when N/K are smaller."""
    return {
        "block_m": routing_data.block_m,
        "block_n": 128,
        "block_k": 256,
        "num_warps": 4,
        "num_stages": 1,
        "group_m": 4,
        "xcd_swizzle": 8,
        "w_cache_modifier": None,
        "split_k": 1,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
    }


def get_kernel_config_gluon(m, n, k, routing_data, force_num_buffers=None):
    """Functional (non-tuned) default gluon config; fallback when no tuned JSON
    entry exists (tuned configs live in configs/moe). Safe for any (block_m, N, K):
    single-buffer, 128-wide N tile, 256-deep K tile (satisfies the stage>=2 BLOCK_K
    floor). ``force_num_buffers`` pins the pipeline stage."""
    return {
        "block_m": routing_data.block_m,
        "block_n": 128,
        "block_k": 256,
        "num_warps": 4,
        "num_stages": 2,
        "group_m": 4,
        "xcd_swizzle": 1,
        "w_cache_modifier": None,
        "split_k": 1,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "num_buffers": 1 if force_num_buffers is None else force_num_buffers,
    }


_MOE_A16W4_CONFIG_NAME = "MOE-GEMM-A16W4"


@functools.lru_cache(maxsize=256)
def _load_moe_a16w4_json(variant: str, block_m: int):
    """Load the configs/moe JSON for a (variant, block_m), falling back to the
    block_m-agnostic file. Returns the raw dict (two-level N -> K mapping), or None
    if no file exists (JSON is optional)."""
    arch = get_arch()
    base = f"{AITER_TRITON_CONFIGS_PATH}/moe"
    name = f"{arch}-{_MOE_A16W4_CONFIG_NAME}-{variant}"
    for fname in (f"{name}-BLOCK_M={block_m}.json", f"{name}.json"):
        fpath = f"{base}/{fname}"
        if os.path.exists(fpath):
            with open(fpath, "r") as fh:
                return json.load(fh)
    return None


def _leq_lookup(mapping: dict, prefix: str, val: int):
    """Select an entry from `mapping` by `val` via {prefix}_LEQ_x / {prefix}_GEQ_x
    keys, falling back to a "default" / "any" catch-all. Returns the matched value,
    or None."""
    leq = sorted(
        int(key.rsplit("_", 1)[1])
        for key in mapping
        if key.startswith(prefix + "_LEQ_")
    )
    for bound in leq:
        if val <= bound:
            return mapping[f"{prefix}_LEQ_{bound}"]
    geq = sorted(
        (
            int(key.rsplit("_", 1)[1])
            for key in mapping
            if key.startswith(prefix + "_GEQ_")
        ),
        reverse=True,
    )
    for bound in geq:
        if val >= bound:
            return mapping[f"{prefix}_GEQ_{bound}"]
    if "default" in mapping:
        return mapping["default"]
    if "any" in mapping:
        return mapping["any"]
    return None


def _moe_a16w4_json_entry(variant: str, block_m: int, n: int, k: int):
    """Config entry for a (variant, block_m, N, K) via the two-level N -> K lookup,
    or None if no file/entry matches."""
    cfg = _load_moe_a16w4_json(variant, block_m)
    if cfg is None:
        return None
    sub = _leq_lookup(cfg, "N", n)
    if not isinstance(sub, dict):
        return None
    entry = _leq_lookup(sub, "K", k)
    return dict(entry) if isinstance(entry, dict) else None


def _auto_default(block_m):
    """Functional default for the `auto` variant (no JSON): gfx1250 runs gluon,
    single-buffer. Non-gfx1250 / swizzled scales are forced to triton earlier."""
    return {"backend": "gluon", "num_buffers": 1}


def _get_config(routing_data, m, n, k, config=None, swizzle_mx_scale=None):
    """Resolve the full a16w4 MoE launch config (backend + stage + tiling).

    Backend/stage come from `config` if pinned, else the ``auto`` variant picks
    them per shape. Tiling comes from the variant JSON if present, else the
    functional default; tiling keys in `config` overlay on top.

    Returns ``(config_dict, is_tuned)``; ``config_dict`` always has ``"backend"``
    and (for gluon) ``"num_buffers"``.
    """
    config = dict(config) if config else {}
    block_m = routing_data.block_m
    backend = config.get("backend")
    num_buffers = config.get("num_buffers")
    pinned_gluon = backend == "gluon"

    arch = get_arch()
    # Correctness: the gluon a16w4 kernel is gfx1250-only and supports only
    # compact (non-swizzled) e8m0 scales.
    if swizzle_mx_scale is not None:
        if pinned_gluon and arch == "gfx1250":
            raise ValueError(
                "backend='gluon' cannot honor swizzled MX scales "
                f"(swizzle_mx_scale={swizzle_mx_scale!r}): the Gluon a16w4 kernel "
                "supports only compact e8m0 scales. Use backend='triton'."
            )
        backend = "triton"
    if arch != "gfx1250":
        backend = "triton"

    is_tuned = False
    # Resolve backend and/or gluon stage from `auto` when either is unpinned, so
    # backend="gluon" (no stage) still gets the tuned per-shape stage.
    if backend is None or (backend == "gluon" and num_buffers is None):
        auto = _moe_a16w4_json_entry("auto", block_m, n, k)
        if auto is not None:
            is_tuned = True
        else:
            auto = _auto_default(block_m)
        if backend is None:
            backend = auto["backend"]
        if backend == "gluon" and num_buffers is None:
            num_buffers = auto.get("num_buffers")

    # Tiling for the chosen variant: JSON if present, else functional default.
    if backend == "gluon":
        if num_buffers is None:
            num_buffers = 1
        entry = _moe_a16w4_json_entry(f"gluon-num-stage-{num_buffers}", block_m, n, k)
        if entry is not None:
            entry["block_m"] = block_m
            entry.setdefault("num_buffers", num_buffers)
            params = entry
            is_tuned = True
        else:
            params = get_kernel_config_gluon(
                m, n, k, routing_data, force_num_buffers=num_buffers
            )
    else:
        entry = _moe_a16w4_json_entry("triton", block_m, n, k)
        if entry is not None:
            entry["block_m"] = block_m
            params = entry
            is_tuned = True
        else:
            params = get_kernel_config_triton(m, n, k, routing_data)

    params["backend"] = backend

    # Overlay caller-supplied tiling keys (everything except the control key).
    for key, val in config.items():
        if key != "backend":
            params[key] = val

    return params, is_tuned


# -----------------------------------------------------------------------------
# Triton Implementation
# -----------------------------------------------------------------------------


def moe_gemm_a16w4(
    x,
    w,
    x_scales,  # This argument is for API compatibility with lower-precision data types. For a16, this should be set to None
    w_scales,
    x_static_scale=None,  # This argument is for API compatibility with lower-precision data types. For a16, this should be set to None
    quant_static_scale=None,  # This argument is for API compatibility with lower-precision data types. For a16, this should be set to None
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
    unpadded_N=None,
    unpadded_K=None,
    config: Optional[dict] = None,
):
    """
    Y[:, :] = 0.
    for e in num_experts:
        Y[idxs_y_m(e), :] += matmul(X[idxs_x_m(e), :], W[e, :, :])

    Args:
        config (Optional[dict]): Kernel selection + tuning parameters. May pin
            ``"backend"`` ("triton"/"gluon"), ``"num_buffers"`` (gluon stage
            1/2/3), and tiling keys (block_n, block_k, num_warps, etc.). When
            backend/stage is unpinned, the ``auto`` config resolves it per shape;
            ``block_m`` always comes from ``routing_data``. See
            :func:`_get_config`.
    """
    assert w.stride(-2) == 1, "`w` must be column-major when it has data-type mxfp"
    assert x_scales is None, "x_scales must be none"
    assert x_static_scale is None, "x_static_scale must be none"
    assert quant_static_scale is None, "quant_static_scale must be none"

    # determine shapes
    M = x.shape[-2] if gather_indx is None else gather_indx.shape[0]
    K, N = x.shape[-1], w.shape[-1]
    block_m = routing_data.block_m
    if unpadded_N and block_m == 16:
        N = unpadded_N
    if unpadded_K and block_m == 16:
        K = unpadded_K

    # resolve the launch config (backend + stage + tiling); see _get_config.
    config, _is_tuned = _get_config(
        routing_data, M, N, K, config=config, swizzle_mx_scale=swizzle_mx_scale
    )
    # _get_config always sets block_m from routing_data; keep them in lockstep.
    assert block_m == config["block_m"], (block_m, config["block_m"])
    use_gluon = config["backend"] == "gluon"
    # swiglu runs in the split-k reduction when split_k>1 (matmul writes both
    # halves), otherwise it folds into the matmul itself.
    if apply_swiglu and config["split_k"] > 1:
        apply_swiglu_matmul = False
        apply_swiglu_reduction = True
    else:
        apply_swiglu_matmul = apply_swiglu
        apply_swiglu_reduction = False
    # swiglu halves N (factor-2 reduction) in whichever stage applies it.
    reduction_n_matmul = 2 if apply_swiglu_matmul else 1
    reduction_n_reduction = 2 if apply_swiglu_reduction else 1

    # allocate output memory
    y, y_final = allocate_output(
        x,
        w,
        out_dtype,
        reduction_n_matmul,
        reduction_n_reduction,
        routing_data,
        gather_indx,
        scatter_indx,
        block_m,
        config["split_k"],
    )
    stride_bias = None if bias is None else bias.stride(0)

    # moe metadata. The kernel unconditionally loads hist / token_offs_raw /
    # block_pid_map, so expt_data is required (not optional).
    expt_data = routing_data.expt_data
    assert expt_data is not None, "routing_data.expt_data is required"
    expt_hist = expt_data.hist
    expt_hist_sum = expt_data.token_offs_pad[-1]
    expt_token_offs_raw = expt_data.token_offs_raw
    expt_block_pid_map = expt_data.block_pid_map

    # spmd grid
    grid_m = routing_data.n_blocks(M, block_m)
    grid_n = triton.cdiv(N, config["block_n"])
    grid = grid_m * grid_n * config["split_k"]

    # launch kernel
    if use_gluon:
        w_scales_kernel = w_scales.transpose(1, 2)
        # Each pipeline stage is a separate named kernel; pick by buffer count so
        # profiles attribute time to the pipeline that actually ran.
        num_buffers = max(
            1, min(config["num_buffers"], triton.cdiv(K, config["block_k"]))
        )
        if num_buffers == 1:
            gluon_kernel = _moe_gemm_a16w4_gluon_stage1
        elif num_buffers == 2:
            gluon_kernel = _moe_gemm_a16w4_gluon_stage2
        else:
            gluon_kernel = _moe_gemm_a16w4_gluon_stage3
        gluon_kernel[(grid,)](
            y,
            y.stride(0),
            y.stride(1),
            y.stride(2),
            x,
            x.stride(0),
            x.stride(1),
            w,
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w_scales_kernel,
            w_scales_kernel.stride(0),
            w_scales_kernel.stride(1),
            w_scales_kernel.stride(2),
            bias,
            stride_bias,
            gammas,
            x.shape[-2],
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
            block_m,
            config["block_n"],
            config["block_k"],
            config["group_m"],
            XCD_SWIZZLE=config["xcd_swizzle"],
            NUM_BUFFERS=num_buffers,
            SWIZZLE_MX_SCALE=swizzle_mx_scale,
            SPLIT_K=config["split_k"],
            EVEN_K=K % config["block_k"] == 0,
            W_CACHE_MODIFIER=config["w_cache_modifier"],
            num_warps=config["num_warps"],
            num_stages=config["num_stages"],
            UPCAST_INDICES=should_upcast_indices(x, w, y),
            waves_per_eu=config["waves_per_eu"],
            matrix_instr_nonkdim=config["matrix_instr_nonkdim"],
        )
    else:
        _moe_gemm_a16w4_triton[(grid,)](
            y,
            y.stride(0),
            y.stride(1),
            y.stride(2),
            x,
            x.stride(0),
            x.stride(1),
            w,
            w.stride(0),
            w.stride(1),
            w.stride(2),
            w_scales,
            w_scales.stride(0),
            w_scales.stride(1),
            w_scales.stride(2),
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
            block_m,
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
        )

    # Build grouped reduction inputs in a uniform way
    group_indx = (
        None
        if scatter_indx is None
        else scatter_indx.view(-1, routing_data.n_expts_act)
    )
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
