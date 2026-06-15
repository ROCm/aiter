# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from typing import List, Optional, Tuple

import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.mhc import (
    get_mhc_fused_post_pre_config,
    get_mhc_pre_splitk,
    get_mhc_pre_splitk_large_m,
)

from ..jit.core import compile_ops

MD_NAME = "module_custom_all_reduce"
FUSED_AR_MHC_MD_NAME = "module_fused_ar_mhc"

GFX950_LARGE_M_BOUND = 1024


@compile_ops("module_custom_all_reduce", develop=True)
def init_custom_ar(
    meta_ptr: int,
    rank_data_ptr: int,
    rank_data_sz: int,
    ipc_handle_ptrs: List[int],
    offsets: List[int],
    rank: int,
    fully_connected: bool,
) -> int: ...


@compile_ops("module_custom_all_reduce", develop=True)
def all_reduce(
    _fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    use_new: bool,
    open_fp8_quant: bool,
    reg_inp_ptr: int,
    reg_inp_bytes: int,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def reduce_scatter(
    _fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    m: int,
    n: int,
    k: int,
    split_dim: int,
    reg_ptr: int,
    reg_bytes: int,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def all_gather_reg(
    _fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    dim: int,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def all_gather_unreg(
    _fa: int,
    inp: torch.Tensor,
    reg_buffer: int,
    out: torch.Tensor,
    reg_bytes: int,
    dim: int,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def fused_allreduce_rmsnorm(
    _fa: int,
    inp: torch.Tensor,
    res_inp: torch.Tensor,
    res_out: torch.Tensor,
    out: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    reg_ptr: int,
    reg_bytes: int,
    use_1stage: bool,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def fused_allreduce_rmsnorm_pad(
    _fa: int,
    inp: torch.Tensor,
    res_inp: torch.Tensor,
    res_out: torch.Tensor,
    out: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    reg_ptr: int,
    reg_bytes: int,
    use_1stage: bool,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def fused_allreduce_rmsnorm_quant(
    _fa: int,
    inp: torch.Tensor,
    res_inp: torch.Tensor,
    res_out: torch.Tensor,
    out: torch.Tensor,
    scale_out: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    reg_ptr: int,
    reg_bytes: int,
    use_1stage: bool,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def fused_allreduce_rmsnorm_quant_per_group(
    _fa: int,
    inp: torch.Tensor,
    res_inp: torch.Tensor,
    res_out: torch.Tensor,
    out: torch.Tensor,
    scale_out: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    group_size: int,
    reg_ptr: int,
    reg_bytes: int,
    use_1stage: bool,
    bf16_out_ptr: int = 0,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def fused_allreduce_rmsnorm_mxfp4_quant(
    _fa: int,
    inp: torch.Tensor,
    res_inp: torch.Tensor,
    res_out: torch.Tensor,
    out: torch.Tensor,
    scale_out: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    reg_ptr: int,
    reg_bytes: int,
    use_1stage: bool,
    bf16_out_ptr: int = 0,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def fused_qknorm_allreduce(
    _fa: int,
    qkv_in: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    eps: float,
    reg_ptr: int,
    reg_bytes: int,
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def dispose(_fa: int) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def meta_size() -> int: ...


@compile_ops("module_custom_all_reduce", develop=True)
def register_input_buffer(
    _fa: int, self_ptr: int, ipc_handle_ptrs: List[int], offsets: List[int]
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def register_output_buffer(
    _fa: int, self_ptr: int, ipc_handle_ptrs: List[int], offsets: List[int]
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def get_graph_buffer_count(_fa: int) -> int: ...


@compile_ops("module_custom_all_reduce", develop=True)
def get_graph_buffer_ipc_meta(_fa: int, handle_out: int, offset_out: int) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def register_graph_buffers(
    _fa: int, handle_ptrs: List[int], offset_ptrs: List[int]
) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def allocate_meta_buffer(size: int) -> int: ...


@compile_ops("module_custom_all_reduce", develop=True)
def free_meta_buffer(ptr: int) -> None: ...


@compile_ops("module_custom_all_reduce", develop=True)
def get_meta_buffer_ipc_handle(inp_ptr: int, out_handle_ptr: int) -> None: ...


# ---------------------------------------------------------------------------
# Fused AllReduce + MHC(post+pre) + RMSNorm (separate JIT module, custom AR IPC)
# Same family as fused_allreduce_rmsnorm / fused_qknorm_allreduce above.
# ---------------------------------------------------------------------------


def _parse_splitk_tilek_env(
    env_name: str, hc_hidden_size: int
) -> Optional[tuple[int, int]]:
    forced = os.environ.get(env_name, "").strip()
    if not forced:
        return None
    splitk_s, _, tilek_s = forced.partition(",")
    splitk = int(splitk_s)
    tilek = int(tilek_s) if tilek_s else 64
    if hc_hidden_size % (splitk * tilek) == 0:
        return splitk, tilek
    return None


def get_mhc_fused_post_pre_config_ar_mhc(
    m: int, hidden_size: int
) -> tuple[int, int, int, int]:
    """TP fused post+pre GEMM tiles; may differ from single-card PR #3651 policy."""
    forced = os.environ.get("AITER_AR_MHC_FUSED_POST_PRE_CONFIG", "").strip()
    if forced:
        parts = [int(x.strip()) for x in forced.split(",")]
        if len(parts) == 4:
            return tuple(parts)  # type: ignore[return-value]
    return get_mhc_fused_post_pre_config(m, hidden_size)


def get_mhc_pre_splitk_ar_mhc(m: int, hc_hidden_size: int) -> tuple[int, int]:
    """TP pre GEMM split-K for small/medium M; falls back to single-card policy."""
    parsed = _parse_splitk_tilek_env("AITER_AR_MHC_PRE_SPLITK", hc_hidden_size)
    if parsed is not None:
        return parsed
    return get_mhc_pre_splitk(m, hc_hidden_size)


def get_mhc_pre_splitk_ar_mhc_large_m(m: int, hc_hidden_size: int) -> tuple[int, int]:
    """TP large-M pre GEMM split-K; may differ from single-card PR #3651 policy."""
    parsed = _parse_splitk_tilek_env("AITER_AR_MHC_PRE_LARGE_M_SPLITK", hc_hidden_size)
    if parsed is not None:
        return parsed
    if get_gfx_runtime() == "gfx950" and m >= 8192 and hc_hidden_size % (4 * 64) == 0:
        return 4, 64
    return get_mhc_pre_splitk_large_m(m, hc_hidden_size)


@compile_ops(FUSED_AR_MHC_MD_NAME)
def fused_allreduce_mhc_fused_post_pre_rmsnorm(
    _fa: int,
    inp: torch.Tensor,
    layer_input: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    gemm_out: torch.Tensor,
    gemm_out_sqrsum: torch.Tensor,
    next_residual: torch.Tensor,
    post_mix: torch.Tensor,
    comb_mix: torch.Tensor,
    layer_input_out: torch.Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    norm_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    tile_m: int = 16,
    tile_n: int = 32,
    tile_k: int = 32,
    pre_tile_k: int = 64,
    post_store_nt: int = -1,
    use_large_m: bool = False,
    use_ar_mhc_full_fusion: bool = False,
    use_ar_mhc_post_epilogue: bool = False,
    use_large_m_post_epilogue: bool = False,
    use_new: bool = True,
    open_fp8_quant: bool = False,
    reg_ptr: int = 0,
    reg_bytes: int = 0,
) -> None: ...


def launch_fused_allreduce_mhc_fused_post_pre_rmsnorm(
    _fa: int,
    inp: torch.Tensor,
    residual_in: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
    norm_eps: float = 1e-6,
    use_new: bool = True,
    open_fp8_quant: bool = False,
    reg_ptr: int = 0,
    reg_bytes: int = 0,
    force_fused: bool = True,
    force_full_fusion: bool = False,
    force_large_m_post_epilogue: Optional[bool] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate workspaces and launch native AR+MHC(post+pre)+RMSNorm via custom AR."""
    m = inp.size(0)
    hidden_size = inp.size(-1)
    hc_mult = residual_in.size(1)
    hc_mult3 = fn.size(0)
    hc_hidden_size = hc_mult * hidden_size
    device = inp.device
    arch = get_gfx_runtime()

    if post_layer_mix.ndim == 3:
        post_layer_mix = post_layer_mix.squeeze(-1)

    use_large_m = bool(
        force_fused
        and arch == "gfx950"
        and m > GFX950_LARGE_M_BOUND
        and not force_full_fusion
    )
    use_ar_mhc_full_fusion = bool(
        force_fused
        and hidden_size % 256 == 0
        and (not use_large_m or force_full_fusion)
    )
    use_ar_mhc_post_epilogue = False
    if force_large_m_post_epilogue is None:
        env = os.environ.get("AITER_AR_MHC_LARGE_M_POST_EPILOGUE", "1").strip().lower()
        force_large_m_post_epilogue = env not in ("0", "false", "off", "no")
    use_large_m_post_epilogue = bool(
        use_large_m
        and force_large_m_post_epilogue
        and m in (2048, 8192)
        and hidden_size % (16 // inp.element_size()) == 0
        and hc_mult == 4
        and inp.is_contiguous()
        and residual_in.is_contiguous()
        and post_layer_mix.is_contiguous()
        and comb_res_mix.is_contiguous()
    )

    if use_large_m:
        selected_splitk, pre_tile_k = get_mhc_pre_splitk_ar_mhc_large_m(
            m, hc_hidden_size
        )
        selected_tile_m = selected_tile_n = selected_tile_k = 16
        post_store_nt = 0 if m > 8 * get_cu_num() else -1
    else:
        selected_splitk, selected_tile_m, selected_tile_n, selected_tile_k = (
            get_mhc_fused_post_pre_config_ar_mhc(m, hidden_size)
        )
        _, pre_tile_k = get_mhc_pre_splitk_ar_mhc(m, hc_hidden_size)
        post_store_nt = -1

    n_splits = selected_splitk

    layer_input = torch.empty_like(inp)
    gemm_out_pad = torch.empty(
        n_splits, m, (hc_mult3 + 31) // 32 * 32, dtype=dtypes.fp32, device=device
    )
    gemm_out = gemm_out_pad[:, :, :hc_mult3]
    gemm_out_sqrsum = torch.empty(n_splits, m, dtype=dtypes.fp32, device=device)
    next_residual = torch.empty_like(residual_in)
    post_mix = torch.empty(m, hc_mult, 1, dtype=dtypes.fp32, device=device)
    comb_mix = torch.empty(m, hc_mult, hc_mult, dtype=dtypes.fp32, device=device)
    layer_input_out = torch.empty(m, hidden_size, dtype=dtypes.bf16, device=device)

    fused_allreduce_mhc_fused_post_pre_rmsnorm(
        _fa,
        inp,
        layer_input,
        residual_in,
        post_layer_mix,
        comb_res_mix,
        fn,
        hc_scale,
        hc_base,
        norm_weight,
        gemm_out,
        gemm_out_sqrsum,
        next_residual,
        post_mix,
        comb_mix,
        layer_input_out,
        rms_eps,
        hc_pre_eps,
        hc_sinkhorn_eps,
        norm_eps,
        hc_post_mult_value,
        sinkhorn_repeat,
        selected_tile_m,
        selected_tile_n,
        selected_tile_k,
        pre_tile_k,
        post_store_nt,
        use_large_m,
        use_ar_mhc_full_fusion,
        use_ar_mhc_post_epilogue,
        use_large_m_post_epilogue,
        use_new,
        open_fp8_quant,
        reg_ptr,
        reg_bytes,
    )
    return post_mix, comb_mix, layer_input_out, next_residual
