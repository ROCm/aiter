# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
from ..jit.core import compile_ops, AITER_CSRC_DIR
from .enum import ActivationType, Enum, QuantType
from ..utility import dtypes
import functools

torch.int4 = getattr(torch, "int4", torch.uint32)


@compile_ops("module_moe_asm")
def topk_softmax(
    topk_weights: Tensor,
    topk_indices: Tensor,
    token_expert_indices: Tensor,
    gating_output: Tensor,
    need_renorm: bool,
    num_shared_experts: int = 0,
    shared_expert_scoring_func: str = "",
) -> None: ...


@compile_ops(
    "module_moe_topksoftmax_asm", fc_name="topk_softmax_asm", ffi_type="ctypes"
)
def topk_softmax_asm(
    topk_weights: Tensor,
    topk_indices: Tensor,
    token_expert_indices: Tensor,
    gating_output: Tensor,
    need_renorm: bool,
) -> None: ...


@compile_ops("module_moe_topk")
def topk_sigmoid(
    topk_weights: Tensor, topk_indices: Tensor, gating_output: Tensor
) -> None: ...


@compile_ops("module_moe_asm")
def moe_sum(input: Tensor, output: Tensor) -> None: ...


@compile_ops("module_moe_asm")
def moe_align_block_size(
    topk_ids: Tensor,
    num_experts: int,
    block_size: int,
    sorted_token_ids: Tensor,
    experts_ids: Tensor,
    token_nums: Tensor,
    num_tokens_post_pad: Tensor,
) -> None: ...


@compile_ops("module_moe_fmoe_asm", ffi_type="ctypes")
def fmoe(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
) -> None: ...


@compile_ops("module_moe_fmoe_asm", ffi_type="ctypes")
def fmoe_int8_g1u0(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    input_scale: Tensor,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    fc2_smooth_scale: Tensor,
    activation: Optional[int] = ActivationType.Silu.value,
) -> None: ...


@compile_ops("module_moe_fmoe_asm", ffi_type="ctypes")
def fmoe_g1u1(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    input_scale: Tensor,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    kernelName: Optional[str] = "",
    fc2_smooth_scale: Optional[Tensor] = None,
    activation: Optional[int] = ActivationType.Silu.value,
) -> None: ...


@compile_ops("module_moe_fmoe_asm", ffi_type="ctypes")
def fmoe_g1u1_tkw1(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    input_scale: Tensor,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    kernelName: Optional[str] = "",
    fc2_smooth_scale: Optional[Tensor] = None,
    activation: Optional[int] = ActivationType.Silu.value,
) -> None: ...


@compile_ops("module_moe_fmoe_asm", ffi_type="ctypes")
def fmoe_int8_g1u0_a16(
    out: Tensor,
    input: Tensor,  # bf16
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    fc1_smooth_scale: Tensor,
    fc2_smooth_scale: Tensor,
    activation: Optional[int] = ActivationType.Silu.value,
) -> None: ...


@compile_ops("module_moe_fmoe_asm", ffi_type="ctypes")
def fmoe_g1u1_a16(
    out: Tensor,
    input: Tensor,  # bf16
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    fc1_smooth_scale: Tensor,
    fc2_smooth_scale: Tensor,
    activation: Optional[int] = ActivationType.Silu.value,
) -> None: ...


@compile_ops("module_moe_fmoe_asm", ffi_type="ctypes")
def fmoe_fp8_blockscale_g1u1(
    out: Tensor,
    input: Tensor,
    gate: Tensor,
    down: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    topk: int,
    input_scale: Tensor,
    fc1_scale: Tensor,
    fc2_scale: Tensor,
    kernelName: Optional[str] = "",
    fc_scale_blkn: int = 128,
    fc_scale_blkk: int = 128,
    fc2_smooth_scale: Optional[Tensor] = None,
    activation: Optional[int] = ActivationType.Silu.value,
    block_size_M: int = 32,
) -> None: ...


@compile_ops("module_moe_fmoe_asm", ffi_type="ctypes")
def moe_stage1_g1u1(
    input: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    inter_dim: int,
    kernelName: Optional[str],
    block_m: int,
    ksplit: int = 0,
    activation: Optional[int] = ActivationType.Silu.value,
    quant_type: Optional[int] = QuantType.No.value,
    a1_scale: Optional[Tensor] = None,
    w1_scale: Optional[Tensor] = None,
    sorted_weights: Optional[Tensor] = None,
) -> None: ...


def cmdGenFunc_ck_moe_stage(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    kernelName: Optional[str] = None,
    w1_scale: Optional[Tensor] = None,
    a1_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
    quant_type: int = 0,
    activation: int = 0,
    splitk: int = 1,
    use_non_temporal_load: bool = False,
    dst_type: Optional[str] = None,
    is_shuffled: bool = True,
):

    mul_routed_weight_stage = 2 if sorted_weights is None else 1
    is_splitk = splitk > 1 and not kernelName
    is_splitk = is_splitk or (bool(kernelName) and "Splitk" in kernelName)
    outtype = str2dtype_dict[dst_type] if is_splitk else out.dtype
    md_name, blob_gen_cmd = get_moe_stage_module(
        hidden_states.dtype,
        w1.dtype,
        outtype,
        activation,
        quant_type,
        mul_routed_weight_stage,
        is_shuffled,
        is_splitk,
    )
    return {
        "md_name": md_name,
        "blob_gen_cmd": blob_gen_cmd,
    }


def cmdGenFunc_ck_moe_stage2(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    kernelName: Optional[str] = None,
    w1_scale: Optional[Tensor] = None,
    a1_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
    quant_type: int = 0,
    activation: int = 0,
    splitk: int = 1,
    use_non_temporal_load: bool = False,
    dst_type: Optional[str] = None,
    is_shuffled: bool = True,
):

    mul_routed_weight_stage = 1 if sorted_weights is None else 2
    md_name, blob_gen_cmd = get_moe_stage_module(
        hidden_states.dtype,
        w1.dtype,
        out.dtype,
        activation,
        quant_type,
        mul_routed_weight_stage,
        is_shuffled,
    )
    return {
        "md_name": md_name,
        "blob_gen_cmd": blob_gen_cmd,
    }


@compile_ops("module_moe_ck2stages", gen_func=cmdGenFunc_ck_moe_stage)
def ck_moe_stage1(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    kernelName: Optional[str] = None,
    w1_scale: Optional[Tensor] = None,
    a1_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
    quant_type: int = 0,
    activation: int = 0,
    splitk: Optional[int] = 1,
    use_non_temporal_load: bool = False,
    dst_type: Optional[str] = None,
    is_shuffled: bool = True,
) -> None: ...


@compile_ops("module_moe_ck2stages", gen_func=cmdGenFunc_ck_moe_stage2)
def ck_moe_stage2(
    inter_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    kernelName: Optional[str] = None,
    w2_scale: Optional[Tensor] = None,
    a2_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
    quant_type: int = 0,
    activation: int = 0,
    splitk: int = 1,
    use_non_temporal_load: bool = False,
    dst_type: Optional[str] = None,
    is_shuffled: bool = True,
) -> None: ...


@compile_ops("module_moe_cktile2stages", fc_name="cktile_moe_gemm1")
def moe_cktile2stages_gemm1_ck(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    sorted_ids: Tensor,
    sorted_expert_ids: Tensor,
    max_token_ids: Tensor,
    topk: int,
    n_padded_zeros: Optional[int] = 0,
    k_padded_zeros: Optional[int] = 0,
    topk_weight: Optional[Tensor] = None,
    x_scale: Optional[Tensor] = None,
    w_scale: Optional[Tensor] = None,
    exp_bias: Optional[Tensor] = None,
    activation: Optional[int] = 0,
    block_m: Optional[int] = 32,
    split_k: Optional[int] = 1,
    kernel_name: str = "",
) -> Tensor: ...


def moe_cktile2stages_gemm1(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    sorted_ids: Tensor,
    sorted_expert_ids: Tensor,
    max_token_ids: Tensor,
    topk: int,
    n_padded_zeros: Optional[int] = 0,
    k_padded_zeros: Optional[int] = 0,
    topk_weight: Optional[Tensor] = None,
    x_scale: Optional[Tensor] = None,
    w_scale: Optional[Tensor] = None,
    exp_bias: Optional[Tensor] = None,
    activation: Optional[int] = 0,
    block_m: Optional[int] = 32,
    split_k: Optional[int] = 1,
    kernel_name: str = "",
):
    return moe_cktile2stages_gemm1_ck(
        XQ,
        WQ,
        Y,
        sorted_ids,
        sorted_expert_ids,
        max_token_ids,
        topk,
        n_padded_zeros,
        k_padded_zeros,
        topk_weight,
        x_scale,
        w_scale,
        exp_bias,
        activation,
        block_m,
        split_k,
        kernel_name,
    )


@compile_ops("module_moe_cktile2stages", fc_name="cktile_moe_gemm2")
def moe_cktile2stages_gemm2_ck(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    sorted_ids: Tensor,
    sorted_expert_ids: Tensor,
    max_token_ids: Tensor,
    topk: int,
    n_padded_zeros: Optional[int] = 0,
    k_padded_zeros: Optional[int] = 0,
    topk_weight: Optional[Tensor] = None,
    x_scale: Optional[Tensor] = None,
    w_scale: Optional[Tensor] = None,
    exp_bias: Optional[Tensor] = None,
    activation: Optional[int] = 0,
    block_m: Optional[int] = 32,
    split_k: Optional[int] = 1,
    kernel_name: str = "",
) -> Tensor: ...


def moe_cktile2stages_gemm2(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    sorted_ids: Tensor,
    sorted_expert_ids: Tensor,
    max_token_ids: Tensor,
    topk: int,
    n_padded_zeros: Optional[int] = 0,
    k_padded_zeros: Optional[int] = 0,
    topk_weight: Optional[Tensor] = None,
    x_scale: Optional[Tensor] = None,
    w_scale: Optional[Tensor] = None,
    exp_bias: Optional[Tensor] = None,
    activation: Optional[int] = 0,
    block_m: Optional[int] = 32,
    split_k: Optional[int] = 1,
    kernel_name: str = "",
):
    return moe_cktile2stages_gemm2_ck(
        XQ,
        WQ,
        Y,
        sorted_ids,
        sorted_expert_ids,
        max_token_ids,
        topk,
        n_padded_zeros,
        k_padded_zeros,
        topk_weight,
        x_scale,
        w_scale,
        exp_bias,
        activation,
        block_m,
        split_k,
        kernel_name,
    )


dtype2str_dict = {
    dtypes.fp16: "f16",
    dtypes.bf16: "b16",
    dtypes.fp8: "f8",
    dtypes.i8: "i8",
    dtypes.fp4x2: "fp4x2",
    torch.uint32: "i4",
    torch.int4: "i4",
}

str2dtype_dict = {
    "f16": dtypes.fp16,
    "b16": dtypes.bf16,
}


@functools.lru_cache(maxsize=1024)
def get_moe_stage_module(
    input_dtype,
    weight_dtype,
    output_dtype,
    activation,
    quant_type,
    mul_routed_weight_stage,
    preshuffle_mode=False,
    is_splitk=False,
):
    if isinstance(activation, int):
        activation = ActivationType(activation)
    if isinstance(quant_type, int):
        quant_type = QuantType(quant_type)

    Adtype = dtype2str_dict[input_dtype]
    Bdtype = dtype2str_dict[weight_dtype]
    Cdtype = dtype2str_dict[output_dtype]

    preshuffle_str = ""
    if preshuffle_mode and weight_dtype == dtypes.fp4x2:
        preshuffle_str = "--preshuffle"

    splitk_str = "--issplitk" if is_splitk else ""
    quant_type = (
        QuantType.per_1x128 if quant_type == QuantType.per_128x128 else quant_type
    )
    act = str(activation).split(".")[-1].lower()
    quant_type = str(quant_type).split(".")[-1].lower()

    parts = [
        "module_moe_ck2stages",
        Adtype,
        Bdtype,
        "preshuffle_on" if preshuffle_mode else "preshuffle_off",
        Cdtype,
        act,
        quant_type,
        f"mulWeightStage{mul_routed_weight_stage}",
    ]
    if is_splitk:
        parts.append("splitk")
    md_name = "_".join(parts)
    blob_gen_cmd = [
        f"{AITER_CSRC_DIR}/ck_gemm_moe_2stages_codegen/gen_instances.py -a {Adtype} -b {Bdtype} -c {Cdtype} -q {quant_type} -act {act} -m {mul_routed_weight_stage} {preshuffle_str} {splitk_str} -w {{}}"
    ]

    return md_name, blob_gen_cmd


# ---------------------------------------------------------------------------
# Scale remapping cache for NPerBlock<128 smalltile kernels
# ---------------------------------------------------------------------------
# Key: (data_ptr, shape_tuple, NPerBlock, KPerBlock)
# Value: remapped scale tensor
_w1_scale_remap_cache: dict = {}
_w2_scale_remap_cache: dict = {}


def clear_scale_remap_cache():
    """Clear cached scale remapping tensors.

    Call this when swapping models or weight tensors to avoid stale cache hits.
    """
    _w1_scale_remap_cache.clear()
    _w2_scale_remap_cache.clear()


_CK_STAGE1_KERNEL_NAME_RE = None
_CK_STAGE1_SCALE_BLOCK_N = 128
_CK_STAGE1_SCALE_BLOCK_K = 128


def _parse_stage1_kernel_blocks(kernelName: str):
    """Parse NPerBlock / KPerBlock from a ck2stages stage1 kernel name when present.

    Naming convention (codegen): moe_ck2stages_gemm1_{BlockSize}x{MPerBlock}x{NPerBlock}x{KPerBlock}_...
    Stage1 callers usually pass kernelName="" — the actual instance is then chosen
    inside the C++ moe_stage1_heuristic_dispatch based on (inter_dim, block_m).
    For the explicit-name path we still parse here for completeness.
    """
    global _CK_STAGE1_KERNEL_NAME_RE
    if _CK_STAGE1_KERNEL_NAME_RE is None:
        import re

        _CK_STAGE1_KERNEL_NAME_RE = re.compile(
            r"moe_ck2stages_gemm1_(\d+)x(\d+)x(\d+)x(\d+)_"
        )
    if not kernelName:
        return None, None
    m = _CK_STAGE1_KERNEL_NAME_RE.search(kernelName)
    if m is None:
        return None, None
    return int(m.group(3)), int(m.group(4))


def _stage1_heuristic_nperb_kperb(inter_dim: int, block_m: int):
    """Mirror of C++ moe_stage1_heuristic_dispatch (per_1x128 / F8 / SwigluStep / B16 path).

    See ck2stages_moe_stage1_heuristic_dispatch.hpp:
      - inter_dim % 128 != 0 && inter_dim % 64 == 0  -> NPerBlock=64
            block_m=16 -> KPerBlock = 256/sizeof(F8) = 256
            block_m=32 -> KPerBlock = 128/sizeof(F8) = 128
            block_m=64 -> KPerBlock = 128/sizeof(F8) = 128
      - inter_dim % 128 != 0 && inter_dim % 32 == 0 -> NPerBlock=32, KPerBlock=128
      - else (128-aligned) -> NPerBlock=128
            block_m=16 -> KPerBlock=256, else KPerBlock=128
    Returns (NPerBlock, KPerBlock) or (None, None) if shape is unsupported.
    """
    if inter_dim <= 0 or block_m not in (16, 32, 64):
        return None, None
    if inter_dim % 128 != 0 and inter_dim % 64 == 0:
        kperb = 256 if block_m == 16 else 128
        return 64, kperb
    if inter_dim % 128 != 0 and inter_dim % 64 != 0 and inter_dim % 32 == 0:
        return 32, 128
    if inter_dim % 128 == 0:
        kperb = 256 if block_m == 16 else 128
        return 128, kperb
    return None, None


def _maybe_broadcast_w1_scale_for_smalltile(
    w1_scale: Optional[Tensor],
    kernelName: str,
    quant_type: QuantType,
    w1: Tensor,
    block_m: int,
) -> Optional[Tensor]:
    """Mirror of T29 stage2 helper for stage1.

    Stage1 GEMM dimensions (see gemm_moe_ck2stages_common_blockscale.cuh ck_moe_stage1_gemm):
        w1 physical shape [E, N=2*inter_dim, K=model_dim]
        Scale_Block_M=1, Scale_Block_N=128, Scale_Block_K=128 (hard-coded)
        w1_scale per_1x128 layout: [E, ceil(2*inter_dim/128), ceil(model_dim/128)]

    The dispatcher picks NPerBlock<128 (i.e. 32 or 64) when 2*inter_dim isn't 128-aligned
    (e.g. inter_dim=160 -> NPerBlock=32). CK gridwise b_scale_desc then expects scale tensor
    laid out at NPerBlock granularity, not 128.

    KPerBlock is always >= 128 in stage1 (256 or 128), so k_rep = 1 always — but we keep
    the formula symmetric with stage2 for defensiveness in case future stage1 instances
    add KPerBlock<128.
    """
    if w1_scale is None:
        return w1_scale
    if quant_type != QuantType.per_1x128:
        return w1_scale
    if w1 is None or w1.dim() != 3:
        return w1_scale
    # Resolve (NPerBlock, KPerBlock): explicit kernelName has priority, otherwise
    # mirror the C++ heuristic from (inter_dim, block_m).
    NPerBlock, KPerBlock = _parse_stage1_kernel_blocks(kernelName)
    if NPerBlock is None:
        # w1.shape[1] == 2*inter_dim for the swiglu / g1u1 path
        inter_dim = w1.shape[1] // 2
        NPerBlock, KPerBlock = _stage1_heuristic_nperb_kperb(inter_dim, int(block_m))
        if NPerBlock is None:
            return w1_scale
    n_rep = (
        _CK_STAGE1_SCALE_BLOCK_N // NPerBlock
        if NPerBlock < _CK_STAGE1_SCALE_BLOCK_N
        else 1
    )
    k_rep = (
        _CK_STAGE1_SCALE_BLOCK_K // KPerBlock
        if KPerBlock < _CK_STAGE1_SCALE_BLOCK_K
        else 1
    )
    if n_rep == 1 and k_rep == 1:
        return w1_scale
    # --- Cache lookup ---
    cache_key = (w1_scale.data_ptr(), w1_scale.shape, NPerBlock, KPerBlock)
    cached = _w1_scale_remap_cache.get(cache_key)
    if cached is not None:
        return cached
    # Defensive shape guard — only broadcast when w1_scale is still in per-128
    # layout. This prevents double-broadcasting if some upstream caller has
    # already converted the tensor to per-NPerBlock layout (m1quad analogue,
    # see T29 §5.1 raise).
    if w1_scale.dim() != 3:
        return w1_scale
    E, N_blk, K_blk = w1_scale.shape
    N_phys = w1.shape[1]
    K_phys = w1.shape[2]
    expected_N_blk = (N_phys + _CK_STAGE1_SCALE_BLOCK_N - 1) // _CK_STAGE1_SCALE_BLOCK_N
    expected_K_blk = (K_phys + _CK_STAGE1_SCALE_BLOCK_K - 1) // _CK_STAGE1_SCALE_BLOCK_K
    if N_blk != expected_N_blk or K_blk != expected_K_blk:
        # Already in some other layout — bail out rather than risk corruption.
        return w1_scale
    # CK gridwise b_scale_desc (gridwise_moe_gemm_blockscale.hpp L1165-1254) under
    # NPerBlock<ScaleBlockN expects per-expert layout
    #     (ceil(2*inter_dim, NPerBlock), ceil(K, ScaleBlockK))
    # with per-expert stride = N_dim * K_dim, NOT the previous formulation
    # (ceil(2*inter_dim,128)*n_rep, ceil(K,128)*k_rep). The earlier version
    # over-padded the N axis (e.g. K=160 -> 12 vs CK-expected 10) so per-expert
    # stride was 48 fp32 while CK indexed at 40 fp32 — causing expert i>=1 to read
    # from a region straddling expert i-1's tail-padding (T34 §5 forensic analysis,
    # T31 numerical sanity passed only because random per-expert scales happened
    # to have similar magnitude). Build the N axis by mapping each NPerBlock-tile
    # index k to caller per-ScaleBlockN entry index floor(k*NPerBlock/SBN), then
    # do the same for K when KPerBlock<ScaleBlockK (defensive — stage1 currently
    # has k_rep==1 always).
    device = w1_scale.device
    N_blk_new = (N_phys + NPerBlock - 1) // NPerBlock
    idx_n = (
        torch.arange(N_blk_new, device=device) * NPerBlock
    ) // _CK_STAGE1_SCALE_BLOCK_N
    ws_new = w1_scale.index_select(1, idx_n)
    if k_rep != 1:
        K_blk_new = (K_phys + KPerBlock - 1) // KPerBlock
        idx_k = (
            torch.arange(K_blk_new, device=device) * KPerBlock
        ) // _CK_STAGE1_SCALE_BLOCK_K
        ws_new = ws_new.index_select(2, idx_k)
    result = ws_new.contiguous()
    # --- Cache store ---
    _w1_scale_remap_cache[cache_key] = result
    return result


def ck_moe_stage1_fwd(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    kernelName: str = "",
    w1_scale: Optional[Tensor] = None,
    a1_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
    quant_type: QuantType = QuantType.No,
    activation: ActivationType = ActivationType.Silu,
    splitk: Optional[int] = 1,
    use_non_temporal_load: Optional[bool] = False,
    dst_type: Optional[torch.dtype] = None,
):
    w1_scale_eff = _maybe_broadcast_w1_scale_for_smalltile(
        w1_scale, kernelName, quant_type, w1, block_m if block_m is not None else 32
    )
    ck_moe_stage1(
        hidden_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        kernelName,
        w1_scale_eff,
        a1_scale,
        block_m,
        sorted_weights,
        quant_type.value,
        activation.value,
        int(splitk) if splitk is not None else splitk,
        use_non_temporal_load,
        None if dst_type is None else dtype2str_dict[dst_type],
        is_shuffled=getattr(w1, "is_shuffled", False),
    )
    return out


_CK_STAGE2_KERNEL_NAME_RE = None


def _parse_stage2_kernel_blocks(kernelName: str):
    """Parse NPerBlock / KPerBlock from a ck2stages stage2 kernel name.

    Naming convention (see gemm_moe_ck2stages_common_blockscale.cuh codegen):
        moe_ck2stages_gemm2_{BlockSize}x{MPerBlock}x{NPerBlock}x{KPerBlock}_{MX}x{NX}_...

    Returns (NPerBlock, KPerBlock) or (None, None) if no match.
    """
    global _CK_STAGE2_KERNEL_NAME_RE
    if _CK_STAGE2_KERNEL_NAME_RE is None:
        import re

        _CK_STAGE2_KERNEL_NAME_RE = re.compile(
            r"moe_ck2stages_gemm2_(\d+)x(\d+)x(\d+)x(\d+)_"
        )
    if not kernelName:
        return None, None
    m = _CK_STAGE2_KERNEL_NAME_RE.search(kernelName)
    if m is None:
        return None, None
    return int(m.group(3)), int(m.group(4))


def _stage2_heuristic_nperb_kperb(inter_dim: int, block_m: int):
    """Mirror of C++ moe_stage2_heuristic_dispatch (per_1x128 / F8 / SwigluStep path).

    See ck2stages_moe_stage2_heuristic_dispatch.hpp:
      if inter_dim%128!=0 && inter_dim%64!=0 && inter_dim%32==0 && block_m in (32,64):
          NPerBlock=32, KPerBlock=32  (nopad small-tile path)
      else (pad path):
          NPerBlock=128, KPerBlock=128  (standard path, n_rep==k_rep==1 -> no broadcast)

    Returns (NPerBlock, KPerBlock) or (None, None) if unsupported.
    """
    if inter_dim <= 0 or block_m not in (16, 32, 64):
        return None, None
    if inter_dim % 128 != 0 and inter_dim % 64 != 0 and inter_dim % 32 == 0:
        if block_m in (32, 64):
            return 32, 32  # nopad NPerBlock=32, KPerBlock=32
        # block_m=16 falls through to pad path
    # pad path: NPerBlock=128 -> n_rep=1, no broadcast needed
    return 128, 128


# CK caller-side ScaleBlockN / ScaleBlockK are hard-coded to 128 in
# gemm_moe_ck2stages_common_blockscale.cuh. The per_1x128 stage2 quant produces
# w2_scale of shape [E, N//128, K//128]. When the dispatched kernel has
# NPerBlock < 128, the CK gridwise b_scale descriptor expects the caller to
# provide a tensor laid out at NPerBlock granularity, not 128.
# Note: K axis is always indexed at ScaleBlockK=128 by CK stage2 regardless
# of KPerBlock — so K dim is NEVER broadcast.
# See teammate-T25 / teammate-T26 / T36 analyses.
_CK_STAGE2_SCALE_BLOCK_N = 128
_CK_STAGE2_SCALE_BLOCK_K = 128


def _maybe_broadcast_w2_scale_for_smalltile(
    w2_scale: Optional[Tensor], kernelName: str, quant_type: QuantType,
    w2: Optional[Tensor] = None, block_m: int = 64,
) -> Optional[Tensor]:
    """Mirror of _maybe_broadcast_w1_scale_for_smalltile for stage2.

    Stage2 GEMM dimensions (see gemm_moe_ck2stages_common_blockscale.cuh ck_moe_stage2_gemm):
        w2 physical shape [E, N=model_dim, K=inter_dim]
        Scale_Block_M=1, Scale_Block_N=128, Scale_Block_K=128 (hard-coded)
        w2_scale per_1x128 layout: [E, ceil(model_dim/128), ceil(inter_dim/128)]

    When NPerBlock < 128 (e.g. inter_dim=160 -> NPerBlock=32), CK gridwise
    b_scale_desc expects scale tensor at NPerBlock granularity, not 128.
    K axis: stage2 CK b_scale_desc always indexes K at ScaleBlockK=128 granularity
    regardless of KPerBlock, so K dim is NEVER broadcast.

    Fix (T36): use index_select per-NPerBlock formula (same as T35 stage1 fix).
    Also support kernelName="" (default dispatch) via _stage2_heuristic_nperb_kperb
    so that F-case (default, no explicit kernelName) is also corrected.
    """
    if w2_scale is None:
        return w2_scale
    if quant_type != QuantType.per_1x128:
        return w2_scale
    # Resolve NPerBlock: explicit kernelName has priority, then heuristic.
    NPerBlock, KPerBlock = _parse_stage2_kernel_blocks(kernelName)
    if NPerBlock is None:
        # kernelName is empty or unparseable — use C++ heuristic mirror.
        # Need inter_dim from w2.shape[2] (w2: [E, N=model_dim, K=inter_dim]).
        if w2 is None or w2.dim() != 3:
            return w2_scale
        inter_dim = w2.shape[2]
        NPerBlock, KPerBlock = _stage2_heuristic_nperb_kperb(inter_dim, int(block_m))
        if NPerBlock is None:
            return w2_scale
    n_rep = _CK_STAGE2_SCALE_BLOCK_N // NPerBlock if NPerBlock < _CK_STAGE2_SCALE_BLOCK_N else 1
    # K axis: stage2 CK b_scale_desc always indexes K at ScaleBlockK=128 granularity,
    # regardless of KPerBlock. So k_rep must be 1 — never broadcast K dim.
    k_rep = 1
    if n_rep == 1:
        return w2_scale
    # --- Cache lookup ---
    cache_key = (w2_scale.data_ptr(), w2_scale.shape, NPerBlock, KPerBlock)
    cached = _w2_scale_remap_cache.get(cache_key)
    if cached is not None:
        return cached
    # Expect w2_scale shape [E, N_blk, K_blk] (per-128 layout). Be defensive:
    # only patch the 3-D case; otherwise pass through unchanged.
    if w2_scale.dim() != 3:
        return w2_scale
    E, N_blk, K_blk = w2_scale.shape
    # Derive N_phys (= model_dim) from w2 weight tensor shape [E, N, K].
    # Fall back to inferred N_phys = N_blk * ScaleBlockN when w2 is unavailable
    # (keeps backward-compat with call-sites that don't pass w2).
    if w2 is not None and w2.dim() == 3:
        N_phys = w2.shape[1]
        K_phys = w2.shape[2]
        expected_N_blk = (N_phys + _CK_STAGE2_SCALE_BLOCK_N - 1) // _CK_STAGE2_SCALE_BLOCK_N
        expected_K_blk = (K_phys + _CK_STAGE2_SCALE_BLOCK_K - 1) // _CK_STAGE2_SCALE_BLOCK_K
        if N_blk != expected_N_blk or K_blk != expected_K_blk:
            # Already in some other layout — bail out rather than risk corruption.
            return w2_scale
    else:
        N_phys = N_blk * _CK_STAGE2_SCALE_BLOCK_N
    # CK gridwise b_scale_grid_desc_bn_ak (gridwise_moe_gemm_blockscale.hpp L1166-1170)
    # when NPerBlock < ScaleBlockN: N dim = ceil(problem.N * 2, NPerBlock),
    #                               K dim = ceil(K, ScaleBlockK).
    # For stage2 problem.N = model_dim (down-proj N), hence N * 2 = model_dim * 2.
    # expert_scale_stride = ceil(N*2, NPerBlock) * ceil(K, ScaleBlockK)
    # (gridwise_moe_gemm_blockscale.hpp L1242-1247).
    #
    # CK only accesses block_n_id in [0, ceil(N, NPerBlock)-1] for stage2
    # (IsInputGemm=false). The remaining rows [ceil(N,NPerBlock), ceil(N*2,NPerBlock)-1]
    # are allocated for the expert_scale_stride but never accessed. We must still
    # produce N_blk_new = ceil(N*2, NPerBlock) rows so that the per-expert stride
    # in the caller-allocated buffer matches what CK uses to offset between experts.
    # Rows beyond ceil(N, NPerBlock)-1 are filled by clamping their index_select
    # source index to the last valid entry (safe: not accessed, just padding).
    device = w2_scale.device
    N_blk_new = (N_phys * 2 + NPerBlock - 1) // NPerBlock
    idx_n = (
        torch.arange(N_blk_new, device=device) * NPerBlock
    ) // _CK_STAGE2_SCALE_BLOCK_N
    idx_n = idx_n.clamp(max=N_blk - 1)  # clamp padding rows to last valid scale entry
    ws_new = w2_scale.index_select(1, idx_n)
    # K axis: stage2 ScaleBlockK=128 always >= KPerBlock in current kernels,
    # so k_rep==1. Defensive broadcast kept for future-proofing.
    if k_rep != 1:
        K_phys_val = w2.shape[2] if (w2 is not None and w2.dim() == 3) else K_blk * _CK_STAGE2_SCALE_BLOCK_K
        K_blk_new = (K_phys_val + KPerBlock - 1) // KPerBlock
        idx_k = (
            torch.arange(K_blk_new, device=device) * KPerBlock
        ) // _CK_STAGE2_SCALE_BLOCK_K
        ws_new = ws_new.index_select(2, idx_k)
    result = ws_new.contiguous()
    # --- Cache store ---
    _w2_scale_remap_cache[cache_key] = result
    return result


def ck_moe_stage2_fwd(
    inter_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    topk: int,
    kernelName: str = "",
    w2_scale: Optional[Tensor] = None,
    a2_scale: Optional[Tensor] = None,
    block_m: Optional[int] = 32,
    sorted_weights: Optional[Tensor] = None,
    quant_type: QuantType = QuantType.No,
    activation: ActivationType = ActivationType.Silu,
    use_non_temporal_load: Optional[bool] = False,
):
    w2_scale_eff = _maybe_broadcast_w2_scale_for_smalltile(
        w2_scale, kernelName, quant_type, w2,
        block_m=block_m if block_m is not None else 64,
    )
    ck_moe_stage2(
        inter_states,
        w1,
        w2,
        sorted_token_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        topk,
        kernelName,
        w2_scale_eff,
        a2_scale,
        block_m,
        sorted_weights,
        quant_type.value,
        activation.value,
        use_non_temporal_load=use_non_temporal_load,
        is_shuffled=getattr(w2, "is_shuffled", False),
    )
    return out
