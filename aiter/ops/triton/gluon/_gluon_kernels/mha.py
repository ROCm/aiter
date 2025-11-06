# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import functools
import json
import torch
import triton
import triton.language as tl
import triton.experimental.gluon as gluon
import triton.experimental.gluon.language as gl

from ...utils._triton import arch_info
from ...utils.core import AITER_TRITON_CONFIGS_PATH
from ...utils._triton.pid_preprocessing import remap_xcd
from ...utils._triton.mha_kernel_utils import _compute_fp8_scaling_factors


@gluon.jit
def _cdiv_fn(x, y):
    return (x + y - 1) // y


@gluon.jit
def _load_fn(ptrs, offset_first, offset_second, boundary_first, boundary_second):
    if offset_first is not None and offset_second is not None:
        mask = (offset_first[:, None] < boundary_first) & (
            offset_second[None, :] < boundary_second
        )
        tensor = gl.load(ptrs, mask=mask, other=0.0)
    elif offset_first is not None:
        mask = offset_first[:, None] < boundary_first
        tensor = gl.load(ptrs, mask=mask, other=0.0)
    elif offset_second is not None:
        mask = offset_second[None, :] < boundary_second
        tensor = gl.load(ptrs, mask=mask, other=0.0)
    else:
        tensor = gl.load(ptrs)
    return tensor


@gluon.jit
def _attn_fwd_inner(
    acc,
    l_i,
    m_i,
    q,
    q_pe,
    k_ptrs,
    k_pe_ptrs,
    v_ptrs,
    stride_kn,
    stride_vk,
    stride_sn,
    start_m,
    seqlen_k,
    seqlen_q,
    dropout_p,
    sd_mask_ptrs,
    dropout_mask_ptrs,
    philox_seed,
    philox_ptrs,
    block_min,
    block_max,
    offs_n_causal,
    masked_blocks,
    n_extra_tokens,
    alibi_slope,
    descale_q,
    descale_k,
    descale_v,
    OFFS_M: gl.constexpr,
    OFFS_N: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    BLOCK_DMODEL_PE: gl.constexpr,  # it's zero or a power of 2
    SM_SCALE: gl.constexpr,
    IS_CAUSAL: gl.constexpr,
    MASK_STEPS: gl.constexpr,
    ENABLE_DROPOUT: gl.constexpr,
    RETURN_SCORES: gl.constexpr,
    PADDED_HEAD: gl.constexpr,
    IS_FP8: gl.constexpr,
    FP8_MAX: gl.constexpr,
    ENABLE_PIPELINING: gl.constexpr,
):
    pass


@gluon.jit
def _attn_fwd(
    q_ptr: torch.Tensor,
    k_ptr: torch.Tensor,
    v_ptr: torch.Tensor,
    descale_q_ptr: torch.Tensor,
    descale_k_ptr: torch.Tensor,
    descale_v_ptr: torch.Tensor,
    out_ptr: torch.Tensor,
    alibi_slopes_ptr: torch.Tensor,
    s_dmask_ptr: torch.Tensor,
    dropout_mask_ptr: torch.Tensor,
    softmax_lse_ptr: torch.Tensor,
    stride_qz_in,
    stride_qh_in,
    stride_qm_in,
    stride_qk_in,
    stride_kz_in,
    stride_kh_in,
    stride_kn_in,
    stride_kk_in,
    stride_vz_in,
    stride_vh_in,
    stride_vn_in,
    stride_vk_in,
    stride_descale_q_z_in,
    stride_descale_k_z_in,
    stride_descale_v_z_in,
    stride_oz_in,
    stride_oh_in,
    stride_om_in,
    stride_on_in,
    stride_alibi_z_in,
    stride_alibi_h_in,
    stride_sd_z_in,
    stride_sd_h_in,
    stride_sd_m_in,
    stride_sd_n_in,
    stride_lse_z_in,
    stride_lse_h_in,
    stride_lse_m_in,
    sm_scale,
    cu_seqlens_q,
    cu_seqlens_k,
    dropout_p,
    philox_seed,
    philox_offset_base_in,
    SEQLEN_Q,
    SEQLEN_K,
    IS_CAUSAL: gl.constexpr,
    NUM_Q_HEADS: gl.constexpr,
    NUM_K_HEADS: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_DMODEL: gl.constexpr,
    BLOCK_DMODEL_POW2: gl.constexpr,
    BLOCK_DMODEL_PE: gl.constexpr,  # it's zero or a power of 2
    RETURN_SCORES: gl.constexpr,
    ENABLE_DROPOUT: gl.constexpr,
    IS_FP8: gl.constexpr,
    FP8_MAX: gl.constexpr,
    VARLEN: gl.constexpr,
    BATCH,
    NUM_XCD: gl.constexpr,
    USE_INT64_STRIDES: gl.constexpr,
):
    if IS_FP8:
        raise NotImplementedError("FP8 is not supported in Gluon MHA yet.")
    if ENABLE_DROPOUT:
        raise NotImplementedError("Dropout is not supported in Gluon MHA yet.")
    if NUM_Q_HEADS > NUM_K_HEADS:
        raise NotImplementedError(
            "Grouped query and multi-query attention not supported yet in Gluon MHA."
        )

    NUM_BLOCKS = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
    # calculate offsets
    wid = tl.program_id(
        0
    )  # workgroup id ranging: 0,1,2,...., (BATCH * NUM_Q_HEADS * NUM_BLOCKS - 1)
    # num blocks along seqlen

    off_q_head = wid % NUM_Q_HEADS  # across num q heads
    off_q_head = remap_xcd(off_q_head, NUM_Q_HEADS, NUM_XCD)
    start_m = (wid // NUM_Q_HEADS) % NUM_BLOCKS
    off_z = (wid // (NUM_BLOCKS * NUM_Q_HEADS)) % BATCH  # across batch size

    # TODO: create layouts (blocked, linear, mfma) for offsets

    # offsets
    offs_m = start_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=...
    )  # across sequence length of q
    offs_n = gl.arange(0, BLOCK_N, layout=...)  # across sequence length of k/v
    offs_d = gl.arange(0, BLOCK_DMODEL_POW2, layout=...)  # across dimension size of v
    HAS_PE: tl.constexpr = BLOCK_DMODEL_PE > 0
    if HAS_PE:
        offs_pe = BLOCK_DMODEL + gl.arange(
            0, BLOCK_DMODEL_PE, layout=...
        )  # across dimension size for positional encoding

    # NOTE:
    # Workaround for int64 strides, In the absence of strides being int64, parts of the offset
    # computation is done in 32 bit and overflows resulting in segfaults
    # If input strides are defined as int64, it disables vectorized loads which drops perf
    # If we define new strides as stride_x = stride_x_in.to(tl.int64), that does not work
    # because strides are tl.constexpr and cannot be upcasted
    # If we define new strides as stride_x: tl.int64 = stride_x_in, segfault remains
    # The permanent solution is to enable upcasting of tl.constexpr
    # In the meantime, the following workaround provides correctness and does not drop perf
    if USE_INT64_STRIDES:
        stride_qz = tl.cast(stride_qz_in, tl.int64)
        stride_qh = tl.cast(stride_qh_in, tl.int64)
        stride_qm = tl.cast(stride_qm_in, tl.int64)
        stride_qk = tl.cast(stride_qk_in, tl.int64)
        stride_kz = tl.cast(stride_kz_in, tl.int64)
        stride_kh = tl.cast(stride_kh_in, tl.int64)
        stride_kn = tl.cast(stride_kn_in, tl.int64)
        stride_kk = tl.cast(stride_kk_in, tl.int64)
        stride_vz = tl.cast(stride_vz_in, tl.int64)
        stride_vh = tl.cast(stride_vh_in, tl.int64)
        stride_vn = tl.cast(stride_vn_in, tl.int64)
        stride_vk = tl.cast(stride_vk_in, tl.int64)

        stride_oz = tl.cast(stride_oz_in, tl.int64)
        stride_oh = tl.cast(stride_oh_in, tl.int64)
        stride_om = tl.cast(stride_om_in, tl.int64)
        stride_on = tl.cast(stride_on_in, tl.int64)
        stride_alibi_z = tl.cast(stride_alibi_z_in, tl.int64)
        stride_alibi_h = tl.cast(stride_alibi_h_in, tl.int64)

        # NOTE: philox offset is need in dropout pointer calculations
        philox_offset_base = tl.cast(philox_offset_base_in, tl.int64)
        stride_sd_z = tl.cast(stride_sd_z_in, tl.int64)
        stride_sd_h = tl.cast(stride_sd_h_in, tl.int64)
        stride_sd_m = tl.cast(stride_sd_m_in, tl.int64)
        stride_sd_n = tl.cast(stride_sd_n_in, tl.int64)
        stride_lse_z = tl.cast(stride_lse_z_in, tl.int64)
        stride_lse_h = tl.cast(stride_lse_h_in, tl.int64)
        stride_lse_m = tl.cast(stride_lse_m_in, tl.int64)
    else:
        stride_qz = stride_qz_in
        stride_qm = stride_qm_in
        stride_qk = stride_qk_in
        stride_qh = stride_qh_in
        stride_kz = stride_kz_in
        stride_kh = stride_kh_in
        stride_kn = stride_kn_in
        stride_kk = stride_kk_in
        stride_vz = stride_vz_in
        stride_vh = stride_vh_in
        stride_vn = stride_vn_in
        stride_vk = stride_vk_in
        stride_descale_q_z = stride_descale_q_z_in
        stride_descale_k_z = stride_descale_k_z_in
        stride_descale_v_z = stride_descale_v_z_in
        stride_oz = stride_oz_in
        stride_oh = stride_oh_in
        stride_om = stride_om_in
        stride_on = stride_on_in
        stride_alibi_z = stride_alibi_z_in
        stride_alibi_h = stride_alibi_h_in
        philox_offset_base = philox_offset_base_in
        stride_sd_z = stride_sd_z_in
        stride_sd_h = stride_sd_h_in
        stride_sd_m = stride_sd_m_in
        stride_sd_n = stride_sd_n_in
        stride_lse_z = stride_lse_z_in
        stride_lse_h = stride_lse_h_in
        stride_lse_m = stride_lse_m_in

    tl.assume(stride_qz_in >= 0)
    tl.assume(stride_qh_in >= 0)
    tl.assume(stride_qm_in >= 0)
    tl.assume(stride_qk_in >= 0)
    tl.assume(stride_kz_in >= 0)
    tl.assume(stride_kh_in >= 0)
    tl.assume(stride_kn_in >= 0)
    tl.assume(stride_kk_in >= 0)
    tl.assume(stride_vz_in >= 0)
    tl.assume(stride_vh_in >= 0)
    tl.assume(stride_vn_in >= 0)
    tl.assume(stride_vk_in >= 0)

    # NOTE: philox offset is need in dropout pointer calculations
    tl.assume(philox_offset_base_in >= 0)
    tl.assume(stride_sd_z_in >= 0)
    tl.assume(stride_sd_h_in >= 0)
    tl.assume(stride_sd_m_in >= 0)
    tl.assume(stride_sd_n_in >= 0)
    tl.assume(stride_lse_z_in >= 0)
    tl.assume(stride_lse_h_in >= 0)
    tl.assume(stride_lse_m_in >= 0)

    if VARLEN:
        cu_seqlens_q_start = gl.amd.cdna4.buffer_load(cu_seqlens_q + off_z)
        cu_seqlens_q_end = gl.amd.cdna4.buffer_load(cu_seqlens_q + off_z + 1)

        seqlen_q = cu_seqlens_q_end - cu_seqlens_q_start
        # We have a one-size-fits-all grid in id(0). Some seqlens might be too
        # small for all start_m so for those we return early.
        if start_m * BLOCK_M > seqlen_q:
            return
        cu_seqlens_k_start = gl.amd.cdna4.buffer_load(cu_seqlens_k + off_z)
        cu_seqlens_k_end = gl.amd.cdna4.buffer_load(cu_seqlens_k + off_z + 1)
        seqlen_k = cu_seqlens_k_end - cu_seqlens_k_start
    else:
        cu_seqlens_q_start = 0
        cu_seqlens_k_start = 0
        seqlen_q = SEQLEN_Q
        seqlen_k = SEQLEN_K

    n_blocks = _cdiv_fn(seqlen_k, BLOCK_N)

    # Now we compute whether we need to exit early due to causal masking.
    # This is because for seqlen_q > seqlen_k, M rows of the attn scores
    # are completely masked, resulting in 0s written to the output, and
    # inf written to LSE. We don't need to do any GEMMs in this case.
    # This block of code determines what N is, and if this WG is operating
    # on those M rows.
    if IS_CAUSAL:
        # TODO: Need to exit early if attention scores are completely masked
        pass

    off_k_head = off_q_head  # for now, no grouped q/k attention


@functools.lru_cache(maxsize=1024)
def _get_config(
    enable_dropout: bool,
    dtype: torch.dtype,
    has_pe: bool = False,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        _get_config._config_dict = {}
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/{dev}-MHA-DEFAULT.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict["default"] = config

    if has_pe and "pe" in _get_config._config_dict["default"]["fwd"]:
        return _get_config._config_dict["default"]["fwd"]["pe"]
    elif enable_dropout or dtype == torch.float32:
        return _get_config._config_dict["default"]["fwd"]["dropout_or_fp32"]
    else:
        return _get_config._config_dict["default"]["fwd"]["default"]
