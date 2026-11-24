# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

_fused_mla_prefill_qkv_fp8_repr = make_kernel_repr(
    "_fused_mla_prefill_qkv_fp8_kernel",
    [
        "NUM_HEADS",
        "NUM_HEADS_OUT",
        "NOPE_DIM",
        "ROPE_DIM",
        "V_DIM",
        "BLOCK_T",
        "NEED_MASK",
        "num_warps",
    ],
)


def _get_config() -> dict:
    config_dir = resolve_config_dir("fusions", "FUSED_MLA_PREFILL_QKV_FP8")
    return dict(load_config_json(f"{config_dir}/DEFAULT.json", required=True)["any"])


@triton.jit(repr=_fused_mla_prefill_qkv_fp8_repr, do_not_specialize=["num_tokens"])
def _fused_mla_prefill_qkv_fp8_kernel(
    q_ptr,
    k_nope_ptr,
    v_ptr,
    k_pe_ptr,
    q_out_ptr,
    k_out_ptr,
    v_out_ptr,
    num_tokens,
    stride_q_t,
    stride_q_h,
    stride_k_nope_t,
    stride_k_nope_h,
    stride_v_t,
    stride_v_h,
    stride_k_pe_t,
    NUM_HEADS: tl.constexpr,
    NUM_HEADS_OUT: tl.constexpr,
    HEADS_OUT_POW2: tl.constexpr,
    NOPE_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    V_DIM: tl.constexpr,
    BLOCK_T: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    """BLOCK_T tokens x all output heads per program; padded head h reads head h % H.

    Offsets stay int32 (the wrapper checks the bound), so memory ops lower to buffer ops.
    """
    tl.assume(stride_q_t > 0)
    tl.assume(stride_q_h > 0)
    tl.assume(stride_k_nope_t > 0)
    tl.assume(stride_k_nope_h > 0)
    tl.assume(stride_v_t > 0)
    tl.assume(stride_v_h > 0)
    tl.assume(stride_k_pe_t > 0)

    rows = tl.arange(0, BLOCK_T * HEADS_OUT_POW2)
    tok = (tl.program_id(0) * BLOCK_T + rows // HEADS_OUT_POW2)[:, None]
    head_out = (rows % HEADS_OUT_POW2)[:, None]
    head = head_out % NUM_HEADS
    mask = None
    if NEED_MASK:
        mask = (tok < num_tokens) & (head_out < NUM_HEADS_OUT)

    QK_DIM: tl.constexpr = NOPE_DIM + ROPE_DIM
    out_row = tok * NUM_HEADS_OUT + head_out
    q_row = q_ptr + tok * stride_q_t + head * stride_q_h

    nope = tl.arange(0, NOPE_DIM)[None, :]
    q_nope = tl.load(q_row + nope, mask=mask)
    k_nope = tl.load(
        k_nope_ptr + tok * stride_k_nope_t + head * stride_k_nope_h + nope, mask=mask
    )
    tl.store(
        q_out_ptr + out_row * QK_DIM + nope,
        q_nope.to(q_out_ptr.dtype.element_ty),
        mask=mask,
    )
    tl.store(
        k_out_ptr + out_row * QK_DIM + nope,
        k_nope.to(k_out_ptr.dtype.element_ty),
        mask=mask,
    )

    if ROPE_DIM > 0:
        rope = tl.arange(0, ROPE_DIM)[None, :]
        q_rope = tl.load(q_row + NOPE_DIM + rope, mask=mask)
        k_rope = tl.load(k_pe_ptr + tok * stride_k_pe_t + rope, mask=mask)
        tl.store(
            q_out_ptr + out_row * QK_DIM + NOPE_DIM + rope,
            q_rope.to(q_out_ptr.dtype.element_ty),
            mask=mask,
        )
        tl.store(
            k_out_ptr + out_row * QK_DIM + NOPE_DIM + rope,
            k_rope.to(k_out_ptr.dtype.element_ty),
            mask=mask,
        )

    v_cols = tl.arange(0, V_DIM)[None, :]
    v = tl.load(v_ptr + tok * stride_v_t + head * stride_v_h + v_cols, mask=mask)
    tl.store(
        v_out_ptr + out_row * V_DIM + v_cols,
        v.to(v_out_ptr.dtype.element_ty),
        mask=mask,
    )
