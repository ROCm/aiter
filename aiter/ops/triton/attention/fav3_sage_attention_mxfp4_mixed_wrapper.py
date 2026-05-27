# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Mixed-precision SAGE attention wrapper.

Combines per-block INT8 (high precision) and MXFP4 (low precision) attention.
The block LUT (kv_block_indices, lut_start, lut_count) selects which KV blocks
are computed in HIGH precision. All other KV blocks are computed in LOW precision.

Pipeline (forward):
    1. ``rotation_smooth_qk`` rotates Q/K with a Hadamard matrix and bakes
       ``sm_scale * 1/ln(2)`` into Q_rot (so attention scores are in base-2 units).
       K_rot is also smoothed (K - mean(K)).
    2. Q_rot/K_rot are quantized to *both* INT8 (per-block scalar scale) and
       MXFP4 (per-row, per-32-group E8M0 scale).
    3. V is quantized to FP8 with a per-channel scale.
    4. The mixed kernel reads both representations and selects per KV block
       according to the LUT.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import triton

import aiter
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention import map_dims
from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4_mixed import (
    sage_fwd_mxfp4_mixed,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
    rotation_smooth_qk,
    sage_quant,
)
from aiter.ops.triton.moe.quant_moe import downcast_to_mxfp
from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut


# RCP_LN2 = 1 / ln(2). Multiplying scores by RCP_LN2 lets the kernel use exp2
# instead of exp.
_RCP_LN2 = 1.4426950408889634
_LN2 = 0.6931471805599453  # 1 / _RCP_LN2 -- used to cancel sage_quant's internal


def get_sage_fwd_configs_mxfp4_mixed():
    """Returns tuned config for mixed-precision MXFP4 on supported architectures."""
    arch = arch_info.get_arch()
    if arch != "gfx950":
        raise RuntimeError(f"MXFP4 mixed-precision is not supported on {arch}")
    return {
        "BLOCK_M": 256,
        "BLOCK_N": 128,
        "waves_per_eu": 2,
        "PRE_LOAD_V": False,
        "num_stages": 3,
        "num_warps": 8,
    }


def _quantize_to_mixed(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    BLKQ: int,
    BLKK: int,
    R: torch.Tensor,
    BLOCK_R: int,
    q_smoothing: bool,
    layout: str,
    sm_scale: float,
):
    """Run rotation + quantize Q/K to BOTH int8 (per-block) and mxfp4 (per-32-group)."""
    fp8_dtype = aiter.dtypes.fp8
    fp8_max = torch.finfo(fp8_dtype).max

    # Step 1: rotation_smooth_qk -> Q_rot (with sm_scale*RCP_LN2 baked in),
    # K_rot (smoothed), delta_s (q-smoothing bias).
    q_rot, k_rot, delta_s = rotation_smooth_qk(
        q,
        k,
        BLKQ,
        R=R,
        BLOCK_R=BLOCK_R,
        q_smoothing=q_smoothing,
        layout=layout,
        sm_scale=(sm_scale * _RCP_LN2),
    )

    # Step 2a: per-block INT8 quant on already-rotated/smoothed tensors.
    # Pass sm_scale=ln(2) so sage_quant's internal `sm_scale * RCP_LN2` becomes
    # 1.0 (Q_rot already carries the scale).
    q_int8, q_descale_int8, k_int8, k_descale_int8, v_fp8, v_descale = sage_quant(
        q_rot,
        k_rot,
        v,
        fp8_dtype,
        fp8_max,
        BLKQ=BLKQ,
        BLKK=BLKK,
        sm_scale=_LN2,
        layout=layout,
        smooth_k=False,
    )

    # Step 2b: MXFP4 quant via per-32-group downcast.
    q_fp4, q_descale_fp4 = downcast_to_mxfp(q_rot, torch.uint8, axis=-1)
    k_fp4, k_descale_fp4 = downcast_to_mxfp(k_rot, torch.uint8, axis=-1)

    return (
        q_int8,
        q_descale_int8,
        k_int8,
        k_descale_int8,
        q_fp4,
        q_descale_fp4,
        k_fp4,
        k_descale_fp4,
        v_fp8,
        v_descale,
        delta_s,
    )


def _build_hp_lp_luts(
    hp_block_mask: torch.Tensor,
    num_q_heads: int,
):
    """Build HP and LP (complement) ragged LUTs from an HP block mask.

    hp_block_mask: True = HIGH precision (INT8); False = LOW precision (MXFP4).
        Shape (batch, num_q_blocks, num_kv_blocks) or
              (batch, num_heads, num_q_blocks, num_kv_blocks).
    """
    hp_lut = block_attn_mask_to_ragged_lut(hp_block_mask, num_heads=num_q_heads)
    lp_lut = block_attn_mask_to_ragged_lut(~hp_block_mask, num_heads=num_q_heads)
    return hp_lut, lp_lut


class _FAv3SageMXFP4MixedFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layout: str = "bshd",
        q_smooth: bool = False,
        config: Optional[dict] = None,
        R: torch.Tensor = None,
        BLOCK_R: int = 128,
        hp_block_mask: Optional[torch.Tensor] = None,
        hp_block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
        lp_block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ):
        bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
        _, _, num_q_heads, head_dim = map_dims(q.shape, bshd_map)

        if config is None:
            config = get_sage_fwd_configs_mxfp4_mixed()

        BLKQ = config["BLOCK_M"]
        BLKK = config["BLOCK_N"]
        sm_scale = head_dim**-0.5

        (
            q_int8,
            q_descale_int8,
            k_int8,
            k_descale_int8,
            q_fp4,
            q_descale_fp4,
            k_fp4,
            k_descale_fp4,
            v_fp8,
            v_descale,
            delta_s,
        ) = _quantize_to_mixed(
            q,
            k,
            v,
            BLKQ=BLKQ,
            BLKK=BLKK,
            R=R,
            BLOCK_R=BLOCK_R,
            q_smoothing=q_smooth,
            layout=layout,
            sm_scale=sm_scale,
        )

        # Resolve the HP/LP LUT pair.
        if hp_block_lut is not None and lp_block_lut is not None:
            pass
        elif hp_block_mask is not None:
            hp_block_lut, lp_block_lut = _build_hp_lp_luts(hp_block_mask, num_q_heads)
        else:
            # No HP requested -> everything LP.
            hp_block_lut = None
            lp_block_lut = None  # synthesized inside ``_func`` to "all LP"

        return fav3_sage_mxfp4_mixed_func(
            q_int8=q_int8,
            q_fp4=q_fp4,
            k_int8=k_int8,
            k_fp4=k_fp4,
            v_fp8=v_fp8,
            q_descale_int8=q_descale_int8,
            q_descale_fp4=q_descale_fp4,
            k_descale_int8=k_descale_int8,
            k_descale_fp4=k_descale_fp4,
            v_descale=v_descale,
            bias=delta_s,
            layout=layout,
            config=config,
            hp_block_lut=hp_block_lut,
            lp_block_lut=lp_block_lut,
        )

    @staticmethod
    def backward(ctx, dout: torch.Tensor):
        assert False, "backward not implemented"
        return (None,) * 11


def fav3_sage_mxfp4_mixed_wrapper(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: str = "bshd",
    q_smooth: bool = False,
    config: Optional[dict] = None,
    R: torch.Tensor = None,
    BLOCK_R: int = 128,
    hp_block_mask: Optional[torch.Tensor] = None,
    hp_block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    lp_block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
):
    """High-precision entry point for mixed-precision SAGE attention.

    Args:
        q, k, v: high precision tensors (fp16/bf16/fp32) in ``layout``.
        layout: "bshd" or "bhsd".
        q_smooth: enable Q smoothing (computes a per-block bias delta_s).
        config: optional kernel config dict.
        R: optional Hadamard rotation matrix; if None, generated internally.
        BLOCK_R: rotation block size when R is generated.
        hp_block_mask: optional boolean (B, H, num_q_blocks, num_kv_blocks)
            (or (B, num_q_blocks, num_kv_blocks)) where ``True`` means the
            block is computed in HIGH precision (INT8) and ``False`` in LOW
            precision (MXFP4). Used to build both HP and LP LUTs.
        hp_block_lut, lp_block_lut: pre-built ragged LUTs (kv_block_indices,
            lut_start, lut_count). Provide both to skip mask -> LUT conversion.
            If both are ``None`` and ``hp_block_mask`` is also ``None``, every
            block is computed in LOW precision (matches the dense MXFP4 path).

    Returns:
        out tensor [B, S, Hq, Dv] / [B, Hq, S, Dv] (bf16).
    """
    for tensor, name in zip([q, k, v], ["q", "k", "v"]):
        assert tensor.dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ], f"Expected high-precision for {name}, got {tensor.dtype}"

    return _FAv3SageMXFP4MixedFunc.apply(
        q,
        k,
        v,
        layout,
        q_smooth,
        config,
        R,
        BLOCK_R,
        hp_block_mask,
        hp_block_lut,
        lp_block_lut,
    )


def _make_all_lp_lut(batch, nheads_q, num_q_blocks, num_kv_blocks, device):
    """LUT meaning every (batch, head, q-block) handles every K block in LP."""
    n_q_blocks_total = batch * nheads_q * num_q_blocks
    indices = (
        torch.arange(num_kv_blocks, dtype=torch.int32, device=device)
        .repeat(n_q_blocks_total)
    )
    lut_count = torch.full(
        (n_q_blocks_total,), num_kv_blocks, dtype=torch.int32, device=device
    )
    lut_start = torch.arange(
        0,
        n_q_blocks_total * num_kv_blocks,
        num_kv_blocks,
        dtype=torch.int32,
        device=device,
    )
    return indices, lut_start, lut_count


def _make_empty_lut(batch, nheads_q, num_q_blocks, device):
    n_q_blocks_total = batch * nheads_q * num_q_blocks
    return (
        torch.zeros(1, dtype=torch.int32, device=device),
        torch.zeros(n_q_blocks_total, dtype=torch.int32, device=device),
        torch.zeros(n_q_blocks_total, dtype=torch.int32, device=device),
    )


def fav3_sage_mxfp4_mixed_func(
    q_int8: torch.Tensor,
    q_fp4: torch.Tensor,
    k_int8: torch.Tensor,
    k_fp4: torch.Tensor,
    v_fp8: torch.Tensor,
    q_descale_int8: torch.Tensor,
    q_descale_fp4: torch.Tensor,
    k_descale_int8: torch.Tensor,
    k_descale_fp4: torch.Tensor,
    v_descale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    layout: str = "bshd",
    config: Optional[dict] = None,
    hp_block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    lp_block_lut: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None,
):
    """Direct mixed-precision SAGE forward kernel launcher.

    Q/K must be present in *both* INT8 and MXFP4 form (already
    rotated/smoothed and quantized). V is FP8 with a per-channel descale.

    Provide one or both of:
        hp_block_lut: ragged LUT of HIGH-precision (INT8) KV blocks.
        lp_block_lut: ragged LUT of LOW-precision (MXFP4) KV blocks.
    Defaults: hp=None means no HP work; lp=None means EVERY KV block is LP.
    The two LUTs should be complements of each other for full attention
    coverage.
    """
    bshd_map = [0, 1, 2, 3] if layout == "bshd" else [0, 2, 1, 3]
    batch, seqlen_q, nheads_q, _ = map_dims(q_int8.shape, bshd_map)
    _, seqlen_k, nheads_k, head_size_qk = map_dims(k_int8.shape, bshd_map)
    _, _, _, head_size_v = map_dims(v_fp8.shape, bshd_map)

    assert q_int8.dtype == torch.int8 and k_int8.dtype == torch.int8
    assert q_fp4.dtype == torch.uint8 and k_fp4.dtype == torch.uint8
    assert layout in ("bshd", "bhsd")
    assert nheads_q % nheads_k == 0

    if config is None:
        config = get_sage_fwd_configs_mxfp4_mixed()

    if layout == "bshd":
        out_shape = (q_int8.shape[0], q_int8.shape[1], q_int8.shape[2], v_fp8.shape[-1])
    else:
        out_shape = (q_int8.shape[0], q_int8.shape[1], q_int8.shape[2], v_fp8.shape[-1])
    out = torch.zeros(out_shape, dtype=torch.bfloat16, device=q_int8.device)

    s_q_int8 = map_dims(q_int8.stride(), bshd_map)
    s_q_fp4 = map_dims(q_fp4.stride(), bshd_map)
    s_k_int8 = map_dims(k_int8.stride(), bshd_map)
    s_k_fp4 = map_dims(k_fp4.stride(), bshd_map)
    s_v = map_dims(v_fp8.stride(), bshd_map)
    s_o = map_dims(out.stride(), bshd_map)

    stride_qd_int8_z, stride_qd_int8_h, stride_qd_int8_blk = q_descale_int8.stride()
    stride_kd_int8_z, stride_kd_int8_h, stride_kd_int8_blk = k_descale_int8.stride()

    s_qd_fp4 = map_dims(q_descale_fp4.stride(), bshd_map)
    s_kd_fp4 = map_dims(k_descale_fp4.stride(), bshd_map)

    stride_vd_z, stride_vd_h, _ = v_descale.stride()

    if bias is not None:
        USE_BIAS = True
        stride_bz, stride_bh, stride_bm, stride_bn = bias.stride()
    else:
        USE_BIAS = False
        stride_bz = stride_bh = stride_bm = stride_bn = 0

    padded_d_qk = max(16, 1 << (head_size_qk - 1).bit_length())
    padded_d_v = max(16, 1 << (head_size_v - 1).bit_length())

    BLKQ = config["BLOCK_M"]
    BLKK = config["BLOCK_N"]
    num_q_blocks = (seqlen_q + BLKQ - 1) // BLKQ
    num_kv_blocks = (seqlen_k + BLKK - 1) // BLKK
    device = q_int8.device

    if hp_block_lut is not None:
        hp_kv_block_indices, hp_lut_start, hp_lut_count = hp_block_lut
    else:
        hp_kv_block_indices, hp_lut_start, hp_lut_count = _make_empty_lut(
            batch, nheads_q, num_q_blocks, device
        )

    if lp_block_lut is not None:
        lp_kv_block_indices, lp_lut_start, lp_lut_count = lp_block_lut
    elif hp_block_lut is None:
        lp_kv_block_indices, lp_lut_start, lp_lut_count = _make_all_lp_lut(
            batch, nheads_q, num_q_blocks, num_kv_blocks, device
        )
    else:
        # User supplied HP only: assume LP = empty.
        lp_kv_block_indices, lp_lut_start, lp_lut_count = _make_empty_lut(
            batch, nheads_q, num_q_blocks, device
        )

    def grid(META):
        return (triton.cdiv(seqlen_q, META["BLOCK_M"]), nheads_q, batch)

    sage_fwd_mxfp4_mixed[grid](
        Q_INT8=q_int8,
        Q_FP4=q_fp4,
        K_INT8=k_int8,
        K_FP4=k_fp4,
        V=v_fp8,
        Q_DESCALE_INT8=q_descale_int8,
        Q_DESCALE_FP4=q_descale_fp4,
        K_DESCALE_INT8=k_descale_int8,
        K_DESCALE_FP4=k_descale_fp4,
        V_DESCALE=v_descale,
        BIAS=bias,
        OUT=out,
        stride_q_int8_z=s_q_int8[0],
        stride_q_int8_h=s_q_int8[2],
        stride_q_int8_m=s_q_int8[1],
        stride_q_int8_d=s_q_int8[3],
        stride_q_fp4_z=s_q_fp4[0],
        stride_q_fp4_h=s_q_fp4[2],
        stride_q_fp4_m=s_q_fp4[1],
        stride_q_fp4_d=s_q_fp4[3],
        stride_k_int8_z=s_k_int8[0],
        stride_k_int8_h=s_k_int8[2],
        stride_k_int8_n=s_k_int8[1],
        stride_k_int8_d=s_k_int8[3],
        stride_k_fp4_z=s_k_fp4[0],
        stride_k_fp4_h=s_k_fp4[2],
        stride_k_fp4_n=s_k_fp4[1],
        stride_k_fp4_d=s_k_fp4[3],
        stride_vz=s_v[0],
        stride_vh=s_v[2],
        stride_vk=s_v[1],
        stride_vd=s_v[3],
        stride_qd_int8_z=stride_qd_int8_z,
        stride_qd_int8_h=stride_qd_int8_h,
        stride_qd_int8_blk=stride_qd_int8_blk,
        stride_qd_fp4_z=s_qd_fp4[0],
        stride_qd_fp4_h=s_qd_fp4[2],
        stride_qd_fp4_m=s_qd_fp4[1],
        stride_kd_int8_z=stride_kd_int8_z,
        stride_kd_int8_h=stride_kd_int8_h,
        stride_kd_int8_blk=stride_kd_int8_blk,
        stride_kd_fp4_z=s_kd_fp4[0],
        stride_kd_fp4_h=s_kd_fp4[2],
        stride_kd_fp4_n=s_kd_fp4[1],
        stride_vd_z=stride_vd_z,
        stride_vd_h=stride_vd_h,
        stride_bz=stride_bz,
        stride_bh=stride_bh,
        stride_bm=stride_bm,
        stride_bn=stride_bn,
        stride_oz=s_o[0],
        stride_oh=s_o[2],
        stride_om=s_o[1],
        stride_od=s_o[3],
        hp_kv_block_indices=hp_kv_block_indices,
        hp_lut_start=hp_lut_start,
        hp_lut_count=hp_lut_count,
        lp_kv_block_indices=lp_kv_block_indices,
        lp_lut_start=lp_lut_start,
        lp_lut_count=lp_lut_count,
        HQ=nheads_q,
        HK=nheads_k,
        ACTUAL_BLOCK_DMODEL_QK=head_size_qk,
        ACTUAL_BLOCK_DMODEL_V=head_size_v,
        MAX_SEQLENS_Q=seqlen_q,
        MAX_SEQLENS_K=seqlen_k,
        BLOCK_DMODEL_QK=padded_d_qk,
        BLOCK_DMODEL_V=padded_d_v,
        USE_BIAS=USE_BIAS,
        **config,
    )

    return out
