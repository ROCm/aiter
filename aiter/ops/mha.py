# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os
from typing import Any, Optional, Tuple

import torch
from torch import Generator, Tensor

from ..jit.core import CK_DIR, AITER_META_DIR, ENABLE_CK, compile_ops
from ..jit.utils.chip_info import get_cu_num, get_gfx
from ..jit.utils.torch_guard import torch_compile_guard
from ..jit.utils.mha_recipes import (
    compose_mha_fwd_variant_suffix_and_filter,
    get_mha_varlen_prebuild_variants_by_names,
)
from ..utility import dtypes


def cmdGenFunc_mha_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    sink_size: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    out: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    q_descale: Optional[Tensor] = None,
    k_descale: Optional[Tensor] = None,
    v_descale: Optional[Tensor] = None,
    sink_ptr: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
):
    _, seqlen_q, _, _ = q.shape
    # causal=true is the same as causal=false in this case
    causal = is_causal
    if seqlen_q == 1 and alibi_slopes is None:
        causal = False

    md_name = "mha_fwd"
    filter = "*"
    if q.dtype == dtypes.fp16:
        md_name += "_fp16"
        filter += "_fp16*"
    elif q.dtype == dtypes.bf16:
        md_name += "_bf16"
        filter += "_bf16*"
    elif q.dtype == dtypes.fp8:
        if out is None or out.dtype == dtypes.bf16:
            md_name += "_fp8bf16"
            filter += "_fp8bf16*"
        else:
            raise NotImplementedError("Unsupported output dtype for FP8 MHA")
    if bias is not None:
        md_name += "_bias"
        filter += "_bias*"
    elif alibi_slopes is not None:
        md_name += "_alibi"
        filter += "_alibi*"
    else:
        md_name += "_nbias"
        filter += "_nbias*"
    if not causal and window_size_left == -1 and window_size_right == -1:
        md_name += "_nmask"
        filter += "_nmask*"
    else:
        md_name += "_mask"
        filter += "_m*"
    if return_softmax_lse:
        md_name += "_lse"
        filter += "_lse*"
    else:
        md_name += "_nlse"
        filter += "_nlse*"
    if dropout_p == 0:
        md_name += "_ndropout"
        filter += "_ndropout*"
    else:
        md_name += "_dropout"
        filter += "_dropout*"
    if q_descale is None or k_descale is None or v_descale is None:
        md_name += "_nqscale"
        filter += "_nqscale*"
    else:
        # only support per-tensor quantization for now
        md_name += "_pertensor"
        filter += "_pertensor*"

    blob_gen_cmd = [
        f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd "
        "--receipt 100 --filter {} --output_dir {{}}".format(filter),
    ]
    return {
        "md_name": md_name,
        "blob_gen_cmd": blob_gen_cmd,
    }


def common_mha_fwd_fake_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[torch.Tensor] = None,
):
    batch_size = q.size(0)
    seqlen_q = q.size(1)
    num_heads = q.size(2)
    head_size_v = v.size(3)
    seqlen_k = k.size(1)

    if out is not None:
        if q.dtype != dtypes.fp8:
            assert out.dtype == q.dtype, "Output must have the same dtype as inputs"
        assert out.device == q.device, "Output must be on the same device as inputs"
        assert out.stride(-1) == 1, "Output tensor must have contiguous last dimension"
        assert out.shape == (
            batch_size,
            seqlen_q,
            num_heads,
            head_size_v,
        ), "Output tensor has incorrect shape"
    else:
        out_dtype = dtypes.bf16 if q.dtype == dtypes.fp8 else q.dtype
        out = torch.empty(
            (batch_size, seqlen_q, num_heads, head_size_v),
            dtype=out_dtype,
            device=q.device,
            requires_grad=q.requires_grad,
        )

    if return_softmax_lse:
        softmax_lse = torch.empty(
            (batch_size, num_heads, seqlen_q), dtype=torch.float32, device=q.device
        )
    else:
        softmax_lse = torch.empty((0,), dtype=torch.float32, device=q.device)

    if return_dropout_randval:
        assert dropout_p > 0, "return_dropout_randval requires p_dropout > 0"
        p = torch.empty(
            (batch_size, num_heads, seqlen_q, seqlen_k),
            dtype=torch.uint8,
            device=q.device,
        )
    else:
        p = torch.empty((0,), device=q.device)

    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)

    return out, softmax_lse, p, rng_state


def gen_mha_fwd_fake_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    sink_size: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    q_descale: Optional[Tensor] = None,
    k_descale: Optional[Tensor] = None,
    v_descale: Optional[Tensor] = None,
    sink_ptr: Optional[Tensor] = None,
    gen: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return common_mha_fwd_fake_tensors(
        q, k, v, dropout_p, return_softmax_lse, return_dropout_randval, out
    )


@compile_ops(
    "module_mha_fwd",
    fc_name="mha_fwd",
    gen_func=cmdGenFunc_mha_fwd,
    gen_fake=gen_mha_fwd_fake_tensors,
)
def mha_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    sink_size: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    out: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    q_descale: Optional[Tensor] = None,
    k_descale: Optional[Tensor] = None,
    v_descale: Optional[Tensor] = None,
    sink_ptr: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...


def gen_mha_fwd_native_splitkv_fake_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor],
    softmax_scale: float,
    causal: bool,
    return_lse: bool,
    num_splits: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, seqlen_q, nhead_q, hdim = q.shape
    o = (
        torch.empty(
            (batch_size, seqlen_q, nhead_q, hdim), dtype=q.dtype, device=q.device
        )
        if out is None
        else out
    )
    if return_lse:
        lse = torch.empty(
            (batch_size, nhead_q, seqlen_q), dtype=torch.float32, device=q.device
        )
    else:
        lse = torch.empty((0,), dtype=torch.float32, device=q.device)
    return o, lse


@compile_ops(
    "module_mha_fwd_native_splitkv",
    gen_fake=gen_mha_fwd_native_splitkv_fake_tensors,
)
def mha_fwd_native_splitkv(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Optional[Tensor],
    softmax_scale: float,
    causal: bool,
    return_lse: bool,
    num_splits: int,
) -> Tuple[Tensor, Tensor]: ...


def gen_fmha_v3_fwd_fake_tensors(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    how_v3_bf16_cvt: int,
    out: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    q_descale: Optional[Tensor] = None,
    k_descale: Optional[Tensor] = None,
    v_descale: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return common_mha_fwd_fake_tensors(
        q, k, v, dropout_p, return_softmax_lse, return_dropout_randval, out
    )


@compile_ops(
    "module_fmha_v3_fwd", fc_name="fmha_v3_fwd", gen_fake=gen_fmha_v3_fwd_fake_tensors
)
def fmha_v3_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    how_v3_bf16_cvt: int,
    out: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    q_descale: Optional[Tensor] = None,
    k_descale: Optional[Tensor] = None,
    v_descale: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...


# OPUS gfx950 dense D=128 bf16 forward. In-file @compile_ops stub so mha.py is
# self-contained like the sibling backends (no inline cross-module import). Named
# with a leading underscore, with fc_name pointing at the pybind symbol
# `fmha_fwd_hd128_bf16_opus_fwd` in the `module_fmha_fwd_hd128_bf16_opus` module.
# The kernel writes `out` in place and returns None.
@compile_ops(
    "module_fmha_fwd_hd128_bf16_opus",
    fc_name="fmha_fwd_hd128_bf16_opus_fwd",
    develop=True,
)
def _fmha_fwd_hd128_bf16_opus_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    causal: bool,
    softmax_scale: float,
) -> None: ...


def fmha_fwd_hd128_bf16_opus_fwd(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: float,
    causal: bool,
    out: Optional[Tensor] = None,
) -> Tensor:
    """Public wrapper: allocates `out` if needed and forwards to the OPUS gfx950
    D=128 bf16 kernel entry point (two-layer pattern, mirroring
    `fmha_fwd_with_sink_asm`).

    The kernel applies `softmax_scale` to Q·K^T internally, handles the GQA head
    fan-out, and writes the attention output into `out` in place (no LSE). q/k/v
    are dense bshd ([batch, seq, head, dim]); `out` matches [B, S, H_q, D_v].
    This is the mha.py-local wrapper (plain function, no torch-op registration)
    and the sole owner of the OPUS D=128 binding/public API.
    """
    batch, q_seq_len, q_head_num, qk_head_dim = q.shape
    v_head_dim = v.size(3)

    if out is None:
        out = torch.empty(
            (batch, q_seq_len, q_head_num, v_head_dim),
            dtype=q.dtype,
            device=q.device,
        )

    _fmha_fwd_hd128_bf16_opus_fwd(q, k, v, out, bool(causal), float(softmax_scale))
    return out


# ---------------------------------------------------------------------------
# fmha_fwd_with_sink_asm (gfx1250) — single-shot batched FMHA forward.
#
# API contract: q/k/v are **bshd shape** ([batch, seq, head, dim]); strides are
# read directly from the tensor so non-contiguous bshd-shaped views (e.g. of
# sbhd / bhsd allocations) are accepted.  Only `tensor.stride(-1) == 1` is
# required.  softmax_scale is forwarded to the kernel as-is (the kernel
# applies it internally to Q·K^T before softmax).
#
# Memory-allocation policy: all GPU tensors (out, lse, sink) are allocated on
# the Python side; the C++ entry point performs only pointer + stride
# bookkeeping and kernel launch (no torch dependency).  The public wrapper
# `fmha_fwd_with_sink_asm` below handles allocation and the AITER-post-scale →
# kernel-pre-scale conversion for sink (multiply by sqrt(qk_head_dim)).
# ---------------------------------------------------------------------------
@compile_ops(
    "module_fmha_fwd_with_sink_asm",
    fc_name="fmha_fwd_with_sink_asm",
    ffi_type="ctypes",
)
def _fmha_fwd_with_sink_asm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    lse: Tensor,
    sink: Optional[Tensor],
    softmax_scale: float,
    is_causal: bool,
    return_lse: bool,
) -> None: ...


def fmha_fwd_with_sink_asm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    softmax_scale: float,
    is_causal: bool,
    return_lse: bool,
    sink: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Public wrapper: allocates `out`/`lse` buffers as needed and forwards to
    the ctypes-backed kernel entry point.

    Contract details:
      * `sink` is passed through verbatim — it is the value the kernel
        consumes directly (no host-side scaling). It is optional: pass `None`
        for no sink. Whether the kernel reads it is decided inside the `.co`
        (ENABLE_SINK). When provided it must be a 1-D fp32 tensor of shape
        [q_head_num].
      * The kernel always accesses `ptr_LSE`, so an LSE buffer is always
        allocated even when `return_lse=False`; in that case the contents are
        undefined and callers should ignore the returned `lse`.
    """
    batch, q_seq_len, q_head_num, qk_head_dim = q.shape
    v_head_dim = v.size(3)

    if out is None:
        out = torch.empty(
            (batch, q_seq_len, q_head_num, v_head_dim),
            dtype=q.dtype,
            device=q.device,
        )

    lse = torch.empty(
        (batch, q_head_num, q_seq_len), dtype=torch.float32, device=q.device
    )

    _fmha_fwd_with_sink_asm(
        q,
        k,
        v,
        out,
        lse,
        sink,
        float(softmax_scale),
        bool(is_causal),
        bool(return_lse),
    )
    return out, lse


@compile_ops(
    "module_fmha_fwd_with_sink_varlen_asm",
    fc_name="fmha_fwd_with_sink_varlen_asm",
    ffi_type="ctypes",
)
def _fmha_fwd_with_sink_varlen_asm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    lse: Tensor,
    sink: Optional[Tensor],
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    is_causal: bool,
    return_lse: bool,
) -> None: ...


def fmha_fwd_with_sink_varlen_asm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    softmax_scale: float,
    is_causal: bool,
    return_lse: bool,
    sink: Optional[Tensor] = None,
    out: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Public wrapper: varlen / packed BF16 ASM forward (gfx1250).

    Layout is packed [token, head, dim] (THD), batch folded into the token
    axis; per-batch boundaries come from cumulative-length arrays:
      * q   : (total_q, nheads,   hdim_q)
      * k   : (total_k, nheads_k, hdim_q)
      * v   : (total_k, nheads_k, hdim_v)
      * out : (total_q, nheads,   hdim_v)
      * lse : (total_q, nheads, 1)  fp32  (kernel writes packed [total_q, nheads])
      * cu_seqlens_q/k : int32 [batch+1] cumulative (cu[batch] == total)

    Contract details:
      * The varlen kernel carries NO strides; q/k/v/out MUST be densely packed,
        so this wrapper calls `.contiguous()` defensively.
      * `max_seqlen_q` is the maximum per-batch Q sequence length (caller-
        supplied, e.g. flash_attn_varlen convention) -- it sets the launch tile
        count; the kernel early-exits tiles beyond each batch's actual length.
      * `sink` is passed through verbatim (the value the kernel consumes
        directly, no host-side scaling); optional. Allocation is caller-side.
      * The kernel always accesses `ptr_LSE`, so an LSE buffer is always
        allocated even when `return_lse=False`; in that case ignore the result.
    """
    q, k, v = (x.contiguous() for x in (q, k, v))
    cu_seqlens_q = cu_seqlens_q.to(torch.int32).contiguous()
    cu_seqlens_k = cu_seqlens_k.to(torch.int32).contiguous()

    total_q, q_head_num, qk_head_dim = q.shape
    v_head_dim = v.size(2)

    if out is None:
        out = torch.empty(
            (total_q, q_head_num, v_head_dim), dtype=q.dtype, device=q.device
        )

    lse = torch.empty((total_q, q_head_num, 1), dtype=torch.float32, device=q.device)

    _fmha_fwd_with_sink_varlen_asm(
        q,
        k,
        v,
        out,
        lse,
        sink,
        cu_seqlens_q,
        cu_seqlens_k,
        int(max_seqlen_q),
        float(softmax_scale),
        bool(is_causal),
        bool(return_lse),
    )
    return out, lse


# ---------------------------------------------------------------------------
# fmha_fwd_mxfp8_asm (gfx1250) — dedicated MXFP8 FMHA forward.
#
# This is an intentionally separate integration path from both the bf16
# `fmha_fwd_with_sink_asm` and the shared `fmha_v3` paths.  The MXFP8 kernel
# uses its own slot-padded kernarg ABI carrying q/k/v micro-scaling (e8m0)
# descale pointers, expected to diverge further from the MI350 layout — so it
# gets its own C++ translation unit (csrc/py_itfs_cu/asm_fmha_fwd_mxfp8.cu)
# and its own Python entry point here.
#
# API contract: q/k/v are **bshd shape** ([batch, seq, head, dim]) fp8 (e4m3),
# in bhsd memory order (stride_head > stride_seq); strides are read directly
# from the tensors.  q/k/v_scale are 1-D float8_e8m0fnu descale buffers laid
# out as the kernel expects (block_size=32 along head_dim).  All GPU tensors
# (out, lse, scales) are allocated on the Python side.
# ---------------------------------------------------------------------------
@compile_ops(
    "module_fmha_fwd_mxfp8_asm",
    fc_name="fmha_fwd_mxfp8_asm",
    ffi_type="ctypes",
)
def _fmha_fwd_mxfp8_asm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_scale: Tensor,
    k_scale: Tensor,
    v_scale: Tensor,
    out: Tensor,
    lse: Tensor,
    softmax_scale: float,
    is_causal: bool,
    return_lse: bool,
) -> None: ...


def fmha_fwd_mxfp8_asm(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    q_scale: Tensor,
    k_scale: Tensor,
    v_scale: Tensor,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    return_lse: bool = False,
    out: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    """Public wrapper for the gfx1250 MXFP8 ASM FMHA forward.

    Args:
        q: (batch, seqlen_q, nheads, hdim_q) fp8 (e4m3), bhsd memory layout
        k: (batch, seqlen_k, nheads_k, hdim_q) fp8 (e4m3), bhsd memory layout
        v: (batch, seqlen_k, nheads_k, hdim_v) fp8 (e4m3), bhsd memory layout
        q_scale/k_scale/v_scale: float8_e8m0fnu micro-scaling descale buffers.
        softmax_scale: softmax scaling; defaults to hdim_q ** -0.5.
        is_causal: must be False (causal not supported yet).
        return_lse: whether the returned lse contents are meaningful.
        out: optional preallocated bf16 output [batch, seqlen_q, nheads, hdim_v].

    Returns:
        (out, lse). The kernel always touches the lse buffer; when
        return_lse=False its contents are undefined and should be ignored.
    """
    assert not is_causal, "fmha_fwd_mxfp8_asm: causal masking is not supported yet"

    batch, q_seq_len, q_head_num, qk_head_dim = q.shape
    v_head_dim = v.size(3)

    if softmax_scale is None:
        softmax_scale = qk_head_dim ** (-0.5)

    if out is None:
        # bf16 output in bhsd memory layout, returned as a bshd-shaped view.
        out_bhsd = torch.empty(
            (batch, q_head_num, q_seq_len, v_head_dim),
            dtype=torch.bfloat16,
            device=q.device,
        )
        out = out_bhsd.transpose(1, 2)

    lse = torch.empty(
        (batch, q_head_num, q_seq_len), dtype=torch.float32, device=q.device
    )

    _fmha_fwd_mxfp8_asm(
        q,
        k,
        v,
        q_scale,
        k_scale,
        v_scale,
        out,
        lse,
        float(softmax_scale),
        bool(is_causal),
        bool(return_lse),
    )
    return out, lse


def cmdGenFunc_mha_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    min_seqlen_q: int,
    dropout_p: float,
    softmax_scale: float,
    logits_soft_cap: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    sink_size: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    gen: Optional[torch.Generator] = None,
    cu_seqlens_q_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_padded: Optional[torch.Tensor] = None,
    sink_ptr: Optional[torch.Tensor] = None,
):
    # causal=true is the same as causal=false in this case
    causal = is_causal
    if max_seqlen_q == 1 and alibi_slopes is None:
        causal = False
    if block_table is None:
        if q.dtype == dtypes.fp16:
            dtype_token = "fp16"
        elif q.dtype == dtypes.bf16:
            dtype_token = "bf16"
        elif q.dtype == dtypes.fp8:
            if out is None or out.dtype == dtypes.bf16:
                dtype_token = "fp8bf16"
            else:
                raise NotImplementedError("Unsupported output dtype for FP8 MHA")
        else:
            raise NotImplementedError("Unsupported dtype for MHA")
        logits_positive = 0.0 < logits_soft_cap
        has_bias = bias is not None
        has_alibi = alibi_slopes is not None
        no_mask = (
            (not causal) and (window_size_left == -1) and (window_size_right == -1)
        )
        use_mask = not no_mask
        return_lse = return_softmax_lse
        dropout_zero = dropout_p == 0
        skip_zero = min_seqlen_q == 0
        has_qscale = q_descale is None or k_descale is None or v_descale is None
        suffix, filter_fwd = compose_mha_fwd_variant_suffix_and_filter(
            dtype=dtype_token,
            logits_positive=logits_positive,
            has_bias=has_bias,
            has_alibi=has_alibi,
            use_mask=use_mask,
            return_lse=return_lse,
            dropout_zero=dropout_zero,
            skip_zero=skip_zero,
            has_qscale=has_qscale,
        )
        md_name = f"mha_varlen_fwd{suffix}"
        variants = get_mha_varlen_prebuild_variants_by_names([md_name], CK_DIR)
        blob_gen_cmd = variants[0]["blob_gen_cmd"]
    else:
        md_name = "mha_varlen_fwd"
        filter_fwd_splitkv1 = "*"  # get_fwd_splitkv_combine_blobs()
        filter_fwd_splitkv2 = "*"  # get_fwd_splitkv_blobs()
        if q.dtype == dtypes.fp16:
            md_name += "_fp16"
            filter_fwd_splitkv1 += "_fp16*"
            filter_fwd_splitkv2 += "_fp16*"
        elif q.dtype == dtypes.bf16:
            md_name += "_bf16"
            filter_fwd_splitkv1 += "_bf16*"
            filter_fwd_splitkv2 += "_bf16*"
        if 0.0 < logits_soft_cap:
            md_name += "_logits"
            filter_fwd += "_logits*"
        else:
            md_name += "_nlogits"
            filter_fwd += "_nlogits*"
        if bias is not None:
            md_name += "_bias"
            filter_fwd_splitkv2 += "_bias*"
        elif alibi_slopes is not None:
            md_name += "_alibi"
            filter_fwd_splitkv2 += "_alibi*"
        else:
            md_name += "_nbias"
            filter_fwd_splitkv2 += "_nbias*"
        if not is_causal and window_size_left == -1 and window_size_right == -1:
            md_name += "_nmask"
            filter_fwd_splitkv2 += "_nmask*"
        else:
            md_name += "_mask"
            filter_fwd_splitkv2 += "_m*"
        if return_softmax_lse:
            md_name += "_lse"
            filter_fwd_splitkv1 += "_lse*"
            filter_fwd_splitkv2 += "_lse*"
        else:
            md_name += "_nlse"
            filter_fwd_splitkv1 += "_nlse*"
            filter_fwd_splitkv2 += "_nlse*"
        md_name += "_pagedkv"
        filter_fwd_splitkv2 += "_pagedkv*"
        filter_fwd_splitkv = f"{filter_fwd_splitkv1}@{filter_fwd_splitkv2}"
        blob_gen_cmd = [
            f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd "
            "--receipt 200 --filter {} --output_dir {{}}".format('" "')
        ]
        blob_gen_cmd.append(
            f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d fwd_splitkv "
            "--receipt 200 --filter {} --output_dir {{}}".format(filter_fwd_splitkv)
        )
    return {
        "md_name": md_name,
        "blob_gen_cmd": blob_gen_cmd,
    }


def gen_mha_varlen_fwd_fake_tensor(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    min_seqlen_q: int,
    dropout_p: float,
    softmax_scale: float,
    logits_soft_cap: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    sink_size: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    gen: Optional[torch.Generator] = None,
    cu_seqlens_q_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_padded: Optional[torch.Tensor] = None,
    sink_ptr: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    device = q.device
    dtype = q.dtype

    total_q = q.size(0)
    num_heads = q.size(1)
    head_size_v = v.size(-1)

    if out is not None:
        out_tensor = out
    else:
        out_dtype = dtypes.bf16 if dtype == dtypes.fp8 else dtype
        out_shape = (total_q, num_heads, head_size_v)
        out_tensor = torch.empty(out_shape, device=device, dtype=out_dtype)

    if return_softmax_lse:
        softmax_lse_shape = (num_heads, total_q)
        softmax_lse_tensor = torch.empty(
            softmax_lse_shape, device=device, dtype=torch.float32
        )
    else:
        softmax_lse_tensor = torch.empty((0,), device=device, dtype=torch.float32)

    if return_dropout_randval:
        p_shape = (num_heads, total_q, max_seqlen_k)
        p_tensor = torch.empty(p_shape, device=device, dtype=torch.uint8)
    else:
        p_tensor = torch.empty((0,), device=device)

    rng_state_tensor = torch.empty((2,), device=device, dtype=torch.int64)

    return [out_tensor, softmax_lse_tensor, p_tensor, rng_state_tensor]


@compile_ops(
    "module_mha_varlen_fwd",
    fc_name="mha_varlen_fwd",
    gen_func=cmdGenFunc_mha_varlen_fwd,
    gen_fake=gen_mha_varlen_fwd_fake_tensor,
)
def mha_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    min_seqlen_q: int,
    dropout_p: float,
    softmax_scale: float,
    logits_soft_cap: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    sink_size: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    gen: Optional[torch.Generator] = None,
    cu_seqlens_q_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_padded: Optional[torch.Tensor] = None,
    sink_ptr: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...


def gen_fmha_v3_varlen_fwd_fake_tensor(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    min_seqlen_q: int,
    dropout_p: float,
    softmax_scale: float,
    logits_soft_cap: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    how_v3_bf16_cvt: int,
    out: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    q_descale: Optional[Tensor] = None,
    k_descale: Optional[Tensor] = None,
    v_descale: Optional[Tensor] = None,
    gen: Optional[torch.Generator] = None,
    cu_seqlens_q_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_padded: Optional[torch.Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    device = q.device
    dtype = q.dtype

    total_q = q.size(0)
    num_heads = q.size(1)
    head_size_v = v.size(-1)

    if out is not None:
        out_tensor = out
    else:
        out_shape = (total_q, num_heads, head_size_v)
        out_tensor = torch.empty(out_shape, device=device, dtype=dtype)

    if return_softmax_lse:
        softmax_lse_shape = (num_heads, total_q)
        softmax_lse_tensor = torch.empty(
            softmax_lse_shape, device=device, dtype=torch.float32
        )
    else:
        softmax_lse_tensor = torch.empty((0,), device=device, dtype=torch.float32)

    if return_dropout_randval:
        p_shape = (num_heads, total_q, max_seqlen_k)
        p_tensor = torch.empty(p_shape, device=device, dtype=torch.uint8)
    else:
        p_tensor = torch.empty((0,), device=device)

    rng_state_tensor = torch.empty((2,), device=device, dtype=torch.int64)

    return [out_tensor, softmax_lse_tensor, p_tensor, rng_state_tensor]


@compile_ops(
    "module_fmha_v3_varlen_fwd",
    fc_name="fmha_v3_varlen_fwd",
    gen_fake=gen_fmha_v3_varlen_fwd_fake_tensor,
)
def fmha_v3_varlen_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    min_seqlen_q: int,
    dropout_p: float,
    softmax_scale: float,
    logits_soft_cap: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    how_v3_bf16_cvt: int,
    out: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    q_descale: Optional[Tensor] = None,
    k_descale: Optional[Tensor] = None,
    v_descale: Optional[Tensor] = None,
    gen: Optional[torch.Generator] = None,
    cu_seqlens_q_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_padded: Optional[torch.Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...


def cmdGenFunc_mha_bwd(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    dbias: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
    sink: Optional[Tensor] = None,
    d_sink: Optional[Tensor] = None,
):
    md_name = "mha_bwd"
    filter1 = "*"  # get_bwd_dot_do_o_blobs()
    filter2 = "*"  # get_bwd_convert_dq_blobs()
    filter3 = "*"  # get_bwd_dq_dk_dv_blobs()
    if q.dtype == dtypes.fp16:
        md_name += "_fp16"
        filter1 += "fp16*"
        filter2 += "fp16*"
        filter3 += "fp16*"
    elif q.dtype == dtypes.bf16:
        md_name += "_bf16"
        filter1 += "bf16*"
        filter2 += "bf16*"
        filter3 += "bf16*"
    if bias is not None:
        md_name += "_bias"
        filter3 += "_bias*"
    elif alibi_slopes is not None:
        md_name += "_alibi"
        filter3 += "_alibi*"
    else:
        md_name += "_nbias"
        filter3 += "_nbias*"
    if dbias is not None:
        md_name += "_dbias"
        filter3 += "_dbias*"
    else:
        md_name += "_ndbias"
        filter3 += "_ndbias*"
    if not is_causal and window_size_left == -1 and window_size_right == -1:
        md_name += "_nmask"
        filter3 += "_nmask*"
    else:
        md_name += "_mask"
        filter3 += "_mask*"
    if dropout_p == 0:
        md_name += "_ndropout"
        filter3 += "_ndropout*"
    else:
        md_name += "_dropout"
        filter3 += "_dropout*"
    if deterministic:
        md_name += "_deterministic"
        filter2 += "_deterministic*"
        filter3 += "_deterministic*"
    else:
        md_name += "_ndeterministic"
        filter2 += "_ndeterministic*"
        filter3 += "_ndeterministic*"

    filter = f"{filter1}@{filter2}@{filter3}"

    blob_gen_cmd = [
        f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d bwd "
        "--receipt 300 --filter {} --output_dir {{}}".format(filter),
        f"{AITER_META_DIR}/hsa/codegen.py -m fmha_v3_bwd --output_dir {{}}",
    ]
    return {
        "md_name": md_name,
        "blob_gen_cmd": blob_gen_cmd,
        "flags_extra_cc": ["'-DONLY_FAV3=0'"],
    }


def common_mha_bwd_fake_tensors(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
):
    batch_size = q.size(0)
    seqlen_q = q.size(1)
    num_heads = q.size(2)
    head_size_q = q.size(3)
    head_size_v = v.size(3)
    seqlen_k = k.size(1)
    num_heads_k = k.size(2)

    if dq is None:
        dq = torch.empty_like(q)  # (batch_size, seqlen_q, num_heads, head_size_q)
    else:
        assert dq.dtype == q.dtype, "dq must have the same dtype as q"
        assert dq.device == q.device, "dq must be on the same device as q"
        assert dq.stride(-1) == 1, "dq must have contiguous last dimension"
        assert dq.shape == (
            batch_size,
            seqlen_q,
            num_heads,
            head_size_q,
        ), "dq has incorrect shape"

    if dk is None:
        dk = torch.empty_like(k)  # (batch_size, seqlen_k, num_heads_k, head_size_q)
    else:
        assert dk.dtype == q.dtype, "dk must have the same dtype as q"
        assert dk.device == q.device, "dk must be on the same device as q"
        assert dk.stride(-1) == 1, "dk must have contiguous last dimension"
        assert dk.shape == (
            batch_size,
            seqlen_k,
            num_heads_k,
            head_size_q,
        ), "dk has incorrect shape"

    if dv is None:
        dv = torch.empty_like(v)  # (batch_size, seqlen_k, num_heads_k, head_size_v)
    else:
        assert dv.dtype == q.dtype, "dv must have the same dtype as q"
        assert dv.device == q.device, "dv must be on the same device as q"
        assert dv.stride(-1) == 1, "dv must have contiguous last dimension"
        assert dv.shape == (
            batch_size,
            seqlen_k,
            num_heads_k,
            head_size_v,
        ), "dv has incorrect shape"

    softmax_d = torch.empty(
        (batch_size, num_heads, seqlen_q),  # {batch_size, num_heads, seqlen_q}
        dtype=torch.float32,
        device=q.device,
    )

    return [dq, dk, dv, softmax_d]


def gen_mha_bwd_fake_tensors(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    dbias: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
    sink: Optional[Tensor] = None,
    d_sink: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return common_mha_bwd_fake_tensors(q, k, v, dq, dk, dv)


@compile_ops(
    "module_mha_bwd",
    fc_name="mha_bwd",
    gen_func=cmdGenFunc_mha_bwd,
    gen_fake=gen_mha_bwd_fake_tensors,
)
def mha_bwd(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    dbias: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
    sink: Optional[Tensor] = None,
    d_sink: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...


def gen_fmha_v3_bwd_fake_tensors(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    is_v3_atomic_fp32: bool,
    how_v3_bf16_cvt: int,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return common_mha_bwd_fake_tensors(q, k, v, dq, dk, dv)


@compile_ops(
    "module_fmha_v3_bwd", fc_name="fmha_v3_bwd", gen_fake=gen_fmha_v3_bwd_fake_tensors
)
def fmha_v3_bwd(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    dropout_p: float,
    softmax_scale: float,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    is_v3_atomic_fp32: bool,
    how_v3_bf16_cvt: int,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...


def cmdGenFunc_mha_varlen_bwd(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
    cu_seqlens_q_padded: Optional[Tensor] = None,
    cu_seqlens_k_padded: Optional[Tensor] = None,
    sink: Optional[Tensor] = None,
    d_sink: Optional[Tensor] = None,
) -> dict[str, Any]:
    md_name = "mha_varlen_bwd"
    filter1 = "*"  # get_bwd_dot_do_o_blobs()
    filter2 = "*"  # get_bwd_convert_dq_blobs()
    filter3 = "*"  # get_bwd_dq_dk_dv_blobs()
    if q.dtype == dtypes.fp16:
        md_name += "_fp16"
        filter1 += "fp16*"
        filter2 += "fp16*"
        filter3 += "fp16*"
    elif q.dtype == dtypes.bf16:
        md_name += "_bf16"
        filter1 += "bf16*"
        filter2 += "bf16*"
        filter3 += "bf16*"
    if alibi_slopes is None:
        md_name += "_nbias"
        filter3 += "_nbias*"
    else:
        md_name += "_alibi"
        filter3 += "_alibi*"
    if not is_causal and window_size_left == -1 and window_size_right == -1:
        md_name += "_nmask"
        filter3 += "_nmask*"
    else:
        md_name += "_mask"
        filter3 += "_mask*"
    if dropout_p == 0:
        md_name += "_ndropout"
        filter3 += "_ndropout*"
    else:
        md_name += "_dropout"
        filter3 += "_dropout*"
    if deterministic:
        md_name += "_deterministic"
        filter2 += "_deterministic*"
        filter3 += "_deterministic*"
    else:
        md_name += "_ndeterministic"
        filter2 += "_ndeterministic*"
        filter3 += "_ndeterministic*"
    filter = f"{filter1}@{filter2}@{filter3}"

    blob_gen_cmd = [
        f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d bwd "
        "--receipt 400 --filter {} --output_dir {{}}".format(filter),
        f"{AITER_META_DIR}/hsa/codegen.py -m fmha_v3_bwd --output_dir {{}}",
    ]
    return {
        "md_name": md_name,
        "blob_gen_cmd": blob_gen_cmd,
        "flags_extra_cc": ["'-DONLY_FAV3=0'"],
    }


def cmdGenFunc_mha_batch_prefill(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    kv_indptr: Tensor,
    kv_page_indices: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    logits_soft_cap: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    sink_size: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    # Per-tensor descale for PERTENSOR mode (Q/K/V each have one scale value)
    q_descale: Optional[Tensor] = None,  # [1] per-tensor Q descale
    k_descale: Optional[Tensor] = None,  # [1] per-tensor K descale
    v_descale: Optional[Tensor] = None,  # [1] per-tensor V descale
    # Per-page descale for KV_BLOCKSCALE mode (Q per-tensor, K/V per-page)
    # Mutually exclusive with k_descale/v_descale
    kv_block_descale: Optional[Tensor] = None,  # [num_block, num_kv_head, 2]
    sink_ptr: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
    kv_last_page_lens: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    seqlen_k: Optional[Tensor] = None,
):
    # causal=true is the same as causal=false in this case
    causal = is_causal
    if max_seqlen_q == 1 and alibi_slopes is None:
        causal = False
        if window_size_left >= max_seqlen_k:
            window_size_left = -1
        if window_size_right >= max_seqlen_k:
            window_size_right = -1
    md_name = "mha_batch_prefill"
    filter_fwd = "*"  # get_fwd_blobs()
    if q.dtype == torch.float16:
        md_name += "_fp16"
        filter_fwd += "fp16*"
    elif q.dtype == torch.bfloat16:
        md_name += "_bf16"
        filter_fwd += "bf16*"
    elif q.dtype == dtypes.fp8:
        if out is None or out.dtype == dtypes.bf16:
            md_name += "_fp8bf16"
            filter_fwd += "fp8bf16*"
        else:
            raise NotImplementedError("Unsupported output dtype for FP8 MHA")
    if 0.0 < logits_soft_cap:
        md_name += "_logits"
        filter_fwd += "_logits*"
    else:
        md_name += "_nlogits"
        filter_fwd += "_nlogits*"
    if alibi_slopes is None:
        md_name += "_nbias"
        filter_fwd += "_nbias*"
    else:
        md_name += "_alibi"
        filter_fwd += "_alibi*"
    if not causal and window_size_left == -1 and window_size_right == -1:
        md_name += "_nmask"
        filter_fwd += "_nmask*"
    else:
        md_name += "_mask"
        filter_fwd += "_mask*"
    if return_softmax_lse:
        md_name += "_lse"
        filter_fwd += "_lse*"
    else:
        md_name += "_nlse"
        filter_fwd += "_nlse*"
    if dropout_p == 0:
        md_name += "_ndropout"
        filter_fwd += "_ndropout*"
    else:
        md_name += "_dropout"
        filter_fwd += "_dropout*"
    if kv_block_descale is not None:
        # KV_BLOCKSCALE: Q per-tensor, K/V per-page
        md_name += "_kv_blockscale"
        filter_fwd += "_kv_blockscale*"
    elif q_descale is None or k_descale is None or v_descale is None:
        md_name += "_nqscale"
        filter_fwd += "_nqscale*"
    else:
        # PERTENSOR: per-tensor quantization
        md_name += "_pertensor"
        filter_fwd += "_pertensor*"
    # Sink only applies when there is a causal/window mask; full attention
    # (window_size_left==-1 and window_size_right==-1) ignores sink_size.
    has_effective_sink = sink_size > 0 and (
        causal or not (window_size_left == -1 and window_size_right == -1)
    )
    if has_effective_sink:
        md_name += "_sink"
        filter_fwd += "_sink*"
    else:
        md_name += "_nsink"
        filter_fwd += "_nsink*"
    blob_gen_cmd = [
        f"{CK_DIR}/example/ck_tile/01_fmha/generate.py -d batch_prefill "
        "--receipt 200 --filter {} --output_dir {{}}".format(filter_fwd)
    ]
    return {
        "md_name": md_name,
        "blob_gen_cmd": blob_gen_cmd,
    }


def gen_mha_varlen_bwd_fake_tensors_common(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    max_seqlen_q: int,
    zero_tensors: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
):
    num_heads = q.size(1)

    batch_size = cu_seqlens_q.numel() - 1
    dq_ = torch.empty_like(q)
    dk_ = torch.empty_like(k)
    dv_ = torch.empty_like(v)
    if dq is not None:
        dq_ = dq
    else:
        dq_ = torch.empty_like(q)

    if dk is not None:
        dk_ = dk
    else:
        dk_ = torch.empty_like(k)

    if dv is not None:
        dv_ = dv
    else:
        dv_ = torch.empty_like(v)

    softmax_d = torch.empty(batch_size, num_heads, max_seqlen_q, dtype=torch.float)

    if zero_tensors:
        dq_.zero_()
        softmax_d.zero_()

    return dq_, dk_, dv_, softmax_d


def gen_mha_varlen_bwd_fake_tensors(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
    sink: Optional[Tensor] = None,
    d_sink: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    return gen_mha_varlen_bwd_fake_tensors_common(
        q, k, v, cu_seqlens_q, max_seqlen_q, zero_tensors, dq, dk, dv
    )
    # return common_mha_bwd_fake_tensors(q, k, v, dq, dk, dv)


@compile_ops(
    "module_mha_varlen_bwd",
    fc_name="mha_varlen_bwd",
    gen_func=cmdGenFunc_mha_varlen_bwd,
    gen_fake=gen_mha_varlen_bwd_fake_tensors,
)
def mha_varlen_bwd(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
    cu_seqlens_q_padded: Optional[Tensor] = None,
    cu_seqlens_k_padded: Optional[Tensor] = None,
    sink: Optional[Tensor] = None,
    d_sink: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...


def gen_fmha_v3_varlen_bwd_fake_tensor(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    is_v3_atomic_fp32: bool,
    how_v3_bf16_cvt: int,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
    cu_seqlens_q_padded: Optional[Tensor] = None,
    cu_seqlens_k_padded: Optional[Tensor] = None,
):
    return gen_mha_varlen_bwd_fake_tensors_common(
        q, k, v, cu_seqlens_q, max_seqlen_q, zero_tensors, dq, dk, dv
    )
    # return common_mha_bwd_fake_tensors(q, k, v, dq, dk, dv)


@compile_ops(
    "module_fmha_v3_varlen_bwd",
    fc_name="fmha_v3_varlen_bwd",
    gen_fake=gen_fmha_v3_varlen_bwd_fake_tensor,
)
def fmha_v3_varlen_bwd(
    dout: Tensor,
    q: Tensor,
    k: Tensor,
    v: Tensor,
    out: Tensor,
    softmax_lse: Tensor,
    cu_seqlens_q: Tensor,
    cu_seqlens_k: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    deterministic: bool,
    is_v3_atomic_fp32: bool,
    how_v3_bf16_cvt: int,
    dq: Optional[Tensor] = None,
    dk: Optional[Tensor] = None,
    dv: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    rng_state: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
    cu_seqlens_q_padded: Optional[Tensor] = None,
    cu_seqlens_k_padded: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def _native_splitkv_heuristic(batch, nhead_q, seqlen_q, seqlen_k, num_cu):
    # Pick split-KV group count G for the native D64 kernel; G == 0 falls back to
    # the CK non-split-KV kernel. Tuned on 100 measured shapes
    # (fmha_native scripts/splitkv_heuristic.py).
    # nwg = workgroups (occupancy); skvt = KV tiles (reduction work to split).
    # Tile sizes are the kernel block geometry: kM0=128 query, kN0=64 key.
    SQ_TILE = 128
    KV_TILE = 64

    def snap(x):
        # Largest split in {2,4,8,16} that is <= x (0 if none).
        g = 0
        for c in (2, 4, 8, 16):
            if c <= x:
                g = c
        return g

    sqt = (seqlen_q + SQ_TILE - 1) // SQ_TILE
    skvt = (seqlen_k + KV_TILE - 1) // KV_TILE
    nwg = batch * nhead_q * sqt

    # Cap G so each split keeps enough KV tiles to amortize combine cost.
    kvdiv = 10 if nwg < 24 else 28
    kv_cap = snap(skvt / kvdiv)

    # Regime A -- undersubscribed: split to fill the machine, capped by KV work.
    if nwg < num_cu:
        occ_cap = snap(3.5 * num_cu / nwg)
        return min(occ_cap, kv_cap)

    # Regime B -- saturated: batch >= 2 has ample independent work, split is loss.
    if batch >= 2:
        return 0

    # batch == 1: a small split hides the long per-CU KV reduction, but only if
    # KV is long enough relative to oversubscription.
    over = nwg / num_cu
    if skvt < 10 * over and over < 30:
        return 0

    # Modest split; fewer heads leave more headroom, extreme corner splits more.
    g = 4 if nhead_q <= 8 else 2
    if over >= 30:
        g = max(g, snap(skvt / 160))
    return min(g, kv_cap) if kv_cap > 0 else 0


def _flash_attn_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    sink_size: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    q_descale: Optional[torch.Tensor],
    k_descale: Optional[torch.Tensor],
    v_descale: Optional[torch.Tensor],
    return_lse: bool,
    return_softmax: bool,
    how_v3_bf16_cvt: Optional[int] = 1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    sink_ptr: Optional[Tensor] = None,
    out: Optional[torch.Tensor] = None,
    num_splits: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size, seqlen_q, nhead_q, hdim_q = q.shape
    _, seqlen_k, nhead_k, hdim_v = v.shape
    if sink_ptr is not None:
        assert sink_ptr.device == q.device, "sink_ptr must be on the same device as q"
        assert sink_ptr.shape[0] == nhead_q, "sink_ptr has incorrect shape"
        if sink_ptr.dtype != torch.float32:
            sink_ptr = sink_ptr.to(torch.float32)
    # mask
    window_size_left = -1 if window_size_left >= seqlen_k else window_size_left
    window_size_right = -1 if window_size_right >= seqlen_k else window_size_right
    # mask = causal and window_size_left == -1  # causal mask
    # nmask = not causal and window_size_left == -1 and window_size_right == -1  # no mask
    swa = (window_size_left > 0) or (window_size_right > 0)

    def is_fmha_v3_fp8():
        ret = get_gfx() in ("gfx942", "gfx950")
        ret = ret and (hdim_q == 128)
        ret = ret and (q.dtype == dtypes.fp8)
        ret = ret and (
            q_descale is not None and k_descale is not None and v_descale is not None
        )
        # support per tensor and per head quant scale
        ret = ret and (
            q_descale.shape == (1,) or q_descale.shape == (batch_size, nhead_k)
        )
        ret = ret and (
            q_descale.shape == k_descale.shape and q_descale.shape == v_descale.shape
        )
        return ret

    def can_impl_fmha_v3_fwd():
        # basic
        # fmha v3 is hand-written gfx9 ASM; non-gfx9 must fall back to ck-tile.
        ret = get_gfx() in ("gfx942", "gfx950")
        ret = ret and (alibi_slopes is None)
        ret = ret and (bias is None)
        ret = ret and (dropout_p == 0.0)
        ret = ret and (hdim_v == 128)
        ret = ret and (hdim_q == 128 or hdim_q == 192)
        ret = ret and (nhead_q % nhead_k == 0)
        ret = ret and (not swa)
        ret = ret and (q.dtype == dtypes.bf16 or is_fmha_v3_fp8())
        ret = ret and (cu_seqlens_q is None and cu_seqlens_kv is None)
        # FP8 ASM kernels assemble the GQA-shift from a fixed log2 table
        # (1,2,4,8,16); arbitrary divisor ratios route to CK.
        if is_fmha_v3_fp8():
            gqa_ratio = nhead_q // nhead_k
            ret = ret and ((gqa_ratio & (gqa_ratio - 1)) == 0)
        return ret

    def can_impl_fmha_fwd_with_sink_asm():
        # gfx1250 ASM bf16 forward (fmha_fwd_with_sink_asm).  Single-shot batched
        # (no varlen / dropout / swa / quant / alibi / bias).  Sink logits
        # (per-Q-head fp32) supported; sink-token (sink_size) not supported.
        ret = get_gfx() == "gfx1250"
        ret = ret and (q.dtype == dtypes.bf16)
        ret = ret and (hdim_q in (64, 128))
        ret = ret and (hdim_v == hdim_q)
        ret = ret and (nhead_q % nhead_k == 0)
        ret = ret and (not swa)
        ret = ret and (sink_size == 0)
        ret = ret and (alibi_slopes is None and bias is None)
        ret = ret and (dropout_p == 0.0)
        ret = ret and (cu_seqlens_q is None and cu_seqlens_kv is None)
        ret = ret and (q_descale is None and k_descale is None and v_descale is None)
        # Per-hdim sink eligibility:
        #
        #   D128 kernels (`_rxy`) compile ENABLE_SINK=0 -- the kernel ignores
        #   any sink buffer.  Routing a caller's sink_ptr to it would silently
        #   drop the sink term, so we fall back to CK whenever sink_ptr is set.
        #
        #   D64 kernels (`_rxy_sink`) compile ENABLE_SINK=1 -- the kernel
        #   ALWAYS reads SINK and adds `exp((sink - max) * scale)` to the
        #   softmax denominator.  There is no "skip sink" mode on this binary,
        #   so calling it with sink_ptr=None now forwards a null pointer to the
        #   kernel (the wrapper no longer raises / zero-fills), which the D64
        #   binary would dereference.  To preserve flash_attn_func's documented
        #   `sink_ptr is None` semantics we keep requiring an explicit sink for
        #   D64 here and fall back to CK otherwise.
        if hdim_q == 128:
            ret = ret and (sink_ptr is None)
        elif hdim_q == 64:
            ret = ret and (sink_ptr is not None)
        return ret

    def can_impl_fmha_fwd_mxfp8_asm():
        # gfx1250 ASM MXFP8 forward (fmha_fwd_mxfp8_asm).  Dedicated path: fp8
        # (e4m3) q/k/v with microscaling (e8m0) block-scale descale buffers.
        # The e8m0 descale dtype is what distinguishes MXFP8 from the per-tensor
        # fp8 path (is_fmha_v3_fp8, which uses fp32 descales on gfx942/gfx950).
        ret = get_gfx() == "gfx1250"
        ret = ret and (q.dtype == dtypes.fp8)
        ret = ret and (
            q_descale is not None and k_descale is not None and v_descale is not None
        )
        ret = ret and (
            q_descale.dtype == dtypes.fp8_e8m0
            and k_descale.dtype == dtypes.fp8_e8m0
            and v_descale.dtype == dtypes.fp8_e8m0
        )

        # The MXFP8 ASM kernel is validated only for BHSD memory order: a
        # bshd-shaped [b, s, h, d] tensor whose head stride exceeds its seq
        # stride (i.e. backed by [b, h, s, d] memory), with a contiguous last
        # dim.  Any other layout (e.g. plain contiguous BSHD) is left to the
        # other forward implementations rather than risking a wrong result.
        def _is_bhsd(t):
            return t.dim() == 4 and t.stride(-1) == 1 and t.stride(2) > t.stride(1)

        ret = ret and _is_bhsd(q) and _is_bhsd(k) and _is_bhsd(v)
        # Only the D128 non-causal kernel is registered in fmha_fwd_mxfp8.csv.
        ret = ret and (hdim_q == 128)
        ret = ret and (hdim_v == hdim_q)
        ret = ret and (not causal)
        ret = ret and (seqlen_k % 128 == 0)
        ret = ret and (nhead_q % nhead_k == 0)
        ret = ret and (not swa)
        ret = ret and (sink_size == 0 and sink_ptr is None)
        ret = ret and (alibi_slopes is None and bias is None)
        ret = ret and (dropout_p == 0.0)
        ret = ret and (cu_seqlens_q is None and cu_seqlens_kv is None)
        return ret

    def can_impl_fmha_native():
        # Native hand-written HIP D64 split-K forward. gfx942-only, dense bf16, no
        # bias/alibi/swa/dropout/sink/fp8/varlen. See design doc.
        ret = get_gfx() == "gfx942"
        ret = ret and (
            q.dtype == dtypes.bf16 and k.dtype == dtypes.bf16 and v.dtype == dtypes.bf16
        )
        ret = ret and (q_descale is None and k_descale is None and v_descale is None)
        ret = ret and (hdim_q == 64 and hdim_v == 64)
        ret = ret and (seqlen_q > 0 and seqlen_k > 0)
        ret = ret and (dropout_p == 0.0)
        ret = ret and (bias is None) and (alibi_slopes is None)
        # Native has only two mask modes: full (causal=False) and full causal
        # (causal=True). Require the exact no-window sentinel -- `not swa` is too
        # loose because swa only tests >0, so a finite 0 window (e.g.
        # window_size=(-1, 0), which is semantically causal) would slip through and
        # be computed as unmasked. Any window/sink restriction falls back to CK/ASM.
        ret = ret and (window_size_left == -1 and window_size_right == -1)
        ret = ret and (sink_size == 0)
        ret = ret and (cu_seqlens_q is None and cu_seqlens_kv is None)
        ret = ret and (sink_ptr is None)
        ret = ret and (nhead_q % nhead_k == 0)
        if causal:
            # sq>sk causal would NaN fully-masked rows in attention_ref but combine
            # returns 0 -> divergence; let those fall back to ASM/CK. decode/square
            # always satisfy sk>=sq.
            ret = ret and (seqlen_k >= seqlen_q)
        return ret

    def can_impl_fmha_fwd_hd128_bf16_opus():
        # OPUS gfx950 dense D=128 bf16 forward (hand-written HIP kernel). Env-gated
        # (OFF by default) so it only supersedes the v3/CK path when explicitly
        # enabled via AITER_ENABLE_FMHA_OPUS. Inference-only: the kernel
        # produces the output only (no softmax LSE, no dropout mask), so it must
        # never capture a case that needs LSE / the autograd backward path.
        if int(os.environ.get("AITER_ENABLE_FMHA_OPUS", "0")) == 0:
            return False
        ret = get_gfx() == "gfx950"
        ret = ret and (q.dtype == dtypes.bf16)
        ret = ret and (hdim_q == 128 and hdim_v == 128)
        ret = ret and (nhead_q % nhead_k == 0)
        # dense only (no varlen), and the kernel requires seqlen_q == seqlen_k.
        ret = ret and (cu_seqlens_q is None and cu_seqlens_kv is None)
        ret = ret and (seqlen_q == seqlen_k)
        # no bias / alibi / dropout / sliding-window / sink / quant-descale.
        ret = ret and (bias is None and alibi_slopes is None)
        ret = ret and (dropout_p == 0.0)
        ret = ret and (window_size_left == -1 and window_size_right == -1)
        ret = ret and (sink_size == 0 and sink_ptr is None)
        ret = ret and (q_descale is None and k_descale is None and v_descale is None)
        # inference-only: no LSE => cannot serve return_lse (grad implies return_lse)
        # or the attention-probs path.
        ret = ret and (not return_lse) and (not return_softmax)
        return ret

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    # Validate newly added optional cumulative length / padded arrays if provided.
    # They are currently only plumbed through for future CK support enabling per-batch padding.
    def _validate_cu(name: str, x: Optional[torch.Tensor]):
        if x is None:
            return
        assert x.dim() == 1, f"{name} must be 1D"
        assert x.dtype in (torch.int32, torch.int64), f"{name} must be int32/int64"
        # Lightweight monotonicity / length check deferred until integration point.

    _validate_cu("cu_seqlens_q", cu_seqlens_q)
    _validate_cu("cu_seqlens_kv", cu_seqlens_kv)

    assert num_splits >= 0, f"num_splits must be >= 0 (0=auto), got {num_splits}"
    if can_impl_fmha_native():
        ns = (
            num_splits
            if num_splits >= 1
            else _native_splitkv_heuristic(
                batch_size, nhead_q, seqlen_q, seqlen_k, get_cu_num()
            )
        )
        if ns > 1:
            assert (
                ns <= (seqlen_k + 63) // 64
            ), (  # ceil(seqlen_k/64); don't silently clamp
                f"num_splits={ns} too large for seqlen_k={seqlen_k}"
            )
            out_, softmax_lse = mha_fwd_native_splitkv(
                q, k, v, out, softmax_scale, causal, return_lse, ns
            )
            S_dmask = None
            # grad path needs a real rng_state tensor (dropout=0 -> no-dropout path).
            rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
            return out_, softmax_lse, S_dmask, rng_state
        # ns <= 1 (0 = heuristic fallback, 1 = forced no-split) -> existing dispatch
    # can_impl_fmha_native() False -> num_splits ignored, existing dispatch
    if can_impl_fmha_fwd_with_sink_asm():
        # gfx1250 ASM bf16 path: q/k/v are bshd; kernel reads strides directly,
        # no API-side permute.  softmax_scale is forwarded as-is (kernel applies
        # it internally to Q·K^T).  sink_ptr is passed through verbatim -- it is
        # the value the kernel consumes directly (no host-side scaling); whether
        # the kernel reads it is decided inside the .co.
        #
        # `can_impl_fmha_fwd_with_sink_asm` still enforces the current-binary
        # (hdim, sink_ptr) matrix (D128 requires sink_ptr is None; D64 requires
        # sink_ptr is not None) so we never feed a null sink to a D64 binary that
        # unconditionally reads it -- forward the caller's sink_ptr unmodified.
        out_, softmax_lse = fmha_fwd_with_sink_asm(
            q,
            k,
            v,
            float(softmax_scale),
            bool(causal),
            True,
            sink_ptr,
            out,
        )
        S_dmask = torch.empty((0,), dtype=torch.float32, device=q.device)
        rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    elif can_impl_fmha_fwd_mxfp8_asm():
        # gfx1250 ASM MXFP8 path: fp8 q/k/v with e8m0 block-scale descales.
        # q/k/v are bshd; the kernel reads strides directly (no API-side
        # permute).  Output is bf16; softmax_scale is forwarded as-is.
        out_, softmax_lse = fmha_fwd_mxfp8_asm(
            q,
            k,
            v,
            q_descale,
            k_descale,
            v_descale,
            float(softmax_scale),
            bool(causal),
            True,
            out,
        )
        S_dmask = torch.empty((0,), dtype=torch.float32, device=q.device)
        rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    elif can_impl_fmha_fwd_hd128_bf16_opus():
        # OPUS gfx950 dense D=128 bf16 forward via the public wrapper (two-layer
        # pattern like fmha_fwd_with_sink_asm). q/k/v are dense bshd [B,S,H,D]; the
        # kernel applies softmax_scale to Q·K^T internally and handles the GQA head
        # fan-out. Inference-only, so the LSE / dropout-mask / rng slots are
        # placeholders (the gate guarantees not return_lse / not return_softmax, so
        # the caller never consumes them) -- mirrors the sink-asm branch shape.
        out_ = fmha_fwd_hd128_bf16_opus_fwd(
            q,
            k,
            v,
            softmax_scale=float(softmax_scale),
            causal=bool(causal),
            out=out,
        )
        softmax_lse = torch.empty((0,), dtype=torch.float32, device=q.device)
        S_dmask = torch.empty((0,), dtype=torch.float32, device=q.device)
        rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    elif can_impl_fmha_v3_fwd() and seqlen_q > 128:  # Prefer CK for decode cases
        out_, softmax_lse, S_dmask, rng_state = fmha_v3_fwd(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            return_lse,
            return_softmax,
            how_v3_bf16_cvt,
            out,
            bias,
            alibi_slopes,
            q_descale,
            k_descale,
            v_descale,
            None,
        )
    else:
        out_, softmax_lse, S_dmask, rng_state = mha_fwd(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            sink_size,
            return_lse,
            return_softmax,
            cu_seqlens_q,
            cu_seqlens_kv,
            out,
            bias,
            alibi_slopes,
            q_descale,
            k_descale,
            v_descale,
            sink_ptr,
            None,
            # custom_build_args={"md_name": md_name, "blob_gen_cmd": blob_gen_cmd},
        )
    return out_, softmax_lse, S_dmask, rng_state


# @torch_compile_guard(mutates_args=[])
def can_impl_fmha_v3_bwd(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dbias: Optional[torch.Tensor],
    dropout_p: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    is_v3_atomic_fp32: Optional[bool] = True,
) -> bool:
    _, seqlen_q, nhead_q, hdim_q = q.shape
    _, seqlen_k, nhead_k, hdim_v = v.shape
    batch_stride_q = q.stride(0)
    stride_q = q.stride(1)
    nhead_stride_q = q.stride(2)

    batch_stride_k = k.stride(0)
    stride_k = k.stride(1)
    nhead_stride_k = k.stride(2)

    batch_stride_v = v.stride(0)
    stride_v = v.stride(1)
    nhead_stride_v = v.stride(2)

    batch_stride_do = dout.stride(0)
    stride_do = dout.stride(1)
    nhead_stride_do = dout.stride(2)

    batch_stride_dk = dk.stride(0)
    nhead_stride_dk = dk.stride(2)

    batch_stride_dv = dv.stride(0)
    nhead_stride_dv = dv.stride(2)

    # mask
    window_size_left = -1 if window_size_left >= seqlen_k else window_size_left
    window_size_right = -1 if window_size_right >= seqlen_k else window_size_right
    mask = causal and window_size_left == -1  # causal mask
    nmask = not causal and window_size_left == -1 and window_size_right == -1  # no mask
    swa = (window_size_left > 0) or (window_size_right > 0)

    def np():
        # bwd_hd128_bf16_a16_rtne
        # bwd_hd128_bf16_a16_rtna
        # bwd_hd128_bf16_a16_rtz
        # bwd_hd128_bf16_a32_rtne
        # bwd_hd128_bf16_a32_rtna
        # bwd_hd128_bf16_a32_rtz
        # bwd_hd128_bf16_causal_a16_rtne
        # bwd_hd128_bf16_causal_a16_rtna
        # bwd_hd128_bf16_causal_a16_rtz
        # bwd_hd128_bf16_causal_a32_rtne
        # bwd_hd128_bf16_causal_a32_rtna
        # bwd_hd128_bf16_causal_a32_rtz
        # bwd_hd128_fp16_a16
        # bwd_hd128_fp16_a32
        # bwd_hd128_fp16_causal_a16
        # bwd_hd128_fp16_causal_a32
        # bwd_hd64_bf16_a16_rtne
        # bwd_hd64_bf16_a16_rtna
        # bwd_hd64_bf16_a16_rtz
        # bwd_hd64_bf16_causal_a16_rtne
        # bwd_hd64_bf16_causal_a16_rtna
        # bwd_hd64_bf16_causal_a16_rtz
        # bwd_hd64_fp16_a16
        # bwd_hd64_fp16_causal_a16
        npssk = seqlen_q == seqlen_k
        npssk &= seqlen_k % 64 == 0
        npssk &= stride_q == stride_do
        npssk &= nhead_stride_q == nhead_stride_do
        npssk &= batch_stride_q == batch_stride_do
        npssk &= stride_k == stride_v
        npssk &= nhead_stride_k == nhead_stride_v
        npssk &= batch_stride_k == batch_stride_v
        npssk &= nhead_stride_k == nhead_stride_dk
        npssk &= nhead_stride_v == nhead_stride_dv
        npssk &= (batch_stride_dk / batch_stride_k) == (nhead_q / nhead_k)
        npssk &= (batch_stride_dv / batch_stride_v) == (nhead_q / nhead_k)

        hd128_case = (hdim_q == 128) and npssk
        hd64_case = (hdim_q == 64 and is_v3_atomic_fp32 == False) and npssk
        ret = hd128_case or hd64_case
        ret &= not swa

        return ret

    def pssk():
        # only for hd64 a32 causal/no causal, fp16/bf16-rtne/rtna/rtz cases
        # FIXME: Currently we only support mask_type == mask_enum::no_mask or causal mask with seqlen_q == seqlen_k
        # Because python side only support mask_enum::bottom_right
        # However v3 kernel only support mask_enum::top_left
        # bwd_hd64_bf16_a32_rtne_pssk
        # bwd_hd64_bf16_a32_rtna_pssk
        # bwd_hd64_bf16_a32_rtz_pssk
        # bwd_hd64_bf16_causal_a32_rtne_pssk
        # bwd_hd64_bf16_causal_a32_rtna_pssk
        # bwd_hd64_bf16_causal_a32_rtz_pssk
        # bwd_hd64_fp16_a32_pssk
        # bwd_hd64_fp16_causal_a32_pssk
        # nhead_stride_dq_acc >= stride_dq_acc must be guaranteed
        ret = hdim_q == 64 and is_v3_atomic_fp32 == True
        ret &= not swa

        return ret

    def pddv():
        # only for a16 causal/no causal, fp16/bf16-rtne/rtna/rtz cases
        # bwd_hd128_bf16_a16_rtne_pddv
        # bwd_hd128_bf16_a16_rtna_pddv
        # bwd_hd128_bf16_a16_rtz_pddv
        # bwd_hd128_bf16_causal_a16_rtne_pddv
        # bwd_hd128_bf16_causal_a16_rtna_pddv
        # bwd_hd128_bf16_causal_a16_rtz_pddv
        # bwd_hd128_fp16_a16_pddv
        # bwd_hd128_fp16_causal_a16_pddv
        ret = is_v3_atomic_fp32 == False
        ret &= hdim_q > 64 and hdim_q < 128
        ret &= seqlen_q == seqlen_k
        ret &= seqlen_k % 64 == 0
        ret &= stride_q == stride_do
        ret &= nhead_stride_q == nhead_stride_do
        ret &= batch_stride_q == batch_stride_do
        ret &= stride_k == stride_v
        ret &= nhead_stride_k == nhead_stride_v
        ret &= batch_stride_k == batch_stride_v
        ret &= nhead_stride_k == nhead_stride_dk
        ret &= nhead_stride_v == nhead_stride_dv
        ret &= (batch_stride_dk / batch_stride_k) == (nhead_q / nhead_k)
        ret &= (batch_stride_dv / batch_stride_v) == (nhead_q / nhead_k)
        ret &= not swa

        return ret

    def psskddv():
        # only for a32 causal/no causal, fp16/bf16-rtne/rtna/rtz cases
        # bwd_hd128_bf16_a32_rtne_psskddv
        # bwd_hd128_bf16_a32_rtna_psskddv
        # bwd_hd128_bf16_a32_rtz_psskddv
        # bwd_hd128_bf16_causal_a32_rtne_psskddv
        # bwd_hd128_bf16_causal_a32_rtna_psskddv
        # bwd_hd128_bf16_causal_a32_rtz_psskddv
        # bwd_hd128_fp16_a32_psskddv
        # bwd_hd128_fp16_causal_a32_psskddv
        # bwd_hd192_fp16_a32_psskddv
        # bwd_hd192_fp16_causal_a32_psskddv
        # bwd_hd192_bf16_a32_rtne_psskddv
        # bwd_hd192_bf16_a32_rtna_psskddv
        # bwd_hd192_bf16_a32_rtz_psskddv
        # bwd_hd192_bf16_causal_a32_rtne_psskddv
        # bwd_hd192_bf16_causal_a32_rtna_psskddv
        # bwd_hd192_bf16_causal_a32_rtz_psskddv
        ret = is_v3_atomic_fp32 == True
        ret &= hdim_q > 64 and hdim_q <= 192
        ret &= nmask or mask or (swa and hdim_q > 64 and hdim_q <= 128)

        return ret

    # basic
    ret = get_gfx() == "gfx942"
    ret &= alibi_slopes is None
    ret &= bias is None
    ret &= dbias is None
    ret &= dropout_p == 0.0
    ret &= not deterministic
    ret &= hdim_q == hdim_v
    ret &= nhead_q % nhead_k == 0
    ret &= hdim_q >= 64 and hdim_q <= 192 and hdim_q % 8 == 0
    ret &= np() or pssk() or pddv() or psskddv()
    return ret


def _flash_attn_backward_fake(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dbias: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    is_v3_atomic_fp32: Optional[bool] = True,
    how_v3_bf16_cvt: Optional[int] = 1,
    sink: Optional[Tensor] = None,
    d_sink: Optional[Tensor] = None,
) -> torch.Tensor:
    batch_size = q.size(0)
    seqlen_q = q.size(1)
    num_heads = q.size(2)

    softmax_d = torch.empty(
        (batch_size, num_heads, seqlen_q),  # {batch_size, num_heads, seqlen_q}
        dtype=torch.float32,
        device=q.device,
    )
    return softmax_d


@torch_compile_guard(
    mutates_args=["dq", "dk", "dv"], gen_fake=_flash_attn_backward_fake
)
def _flash_attn_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    dbias: Optional[torch.Tensor],
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    bias: Optional[torch.Tensor],
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    is_v3_atomic_fp32: Optional[bool] = True,
    how_v3_bf16_cvt: Optional[int] = 1,
    sink: Optional[Tensor] = None,
    d_sink: Optional[Tensor] = None,
) -> torch.Tensor:
    # rtna & rtz are deprecated in gfx950
    if get_gfx() == "gfx950" and how_v3_bf16_cvt != 0:
        how_v3_bf16_cvt = 0

    # can_impl_fmha_v3_bwd should before maybe_contiguous to get pure dout, q, k, v, out
    can_impl_fmha_v3_bwd_ = can_impl_fmha_v3_bwd(
        dout,
        q,
        k,
        v,
        dk,
        dv,
        dbias,
        dropout_p,
        causal,
        window_size_left,
        window_size_right,
        bias,
        alibi_slopes,
        deterministic,
        is_v3_atomic_fp32,
    )

    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]

    _, seqlen_q, nhead_q, hdim_q = q.shape
    _, seqlen_k, nhead_k, hdim_v = v.shape
    nmask = not causal and window_size_left == -1 and window_size_right == -1  # no mask
    swa = (window_size_left > 0) or (window_size_right > 0)

    # only 1 block when sk <= 256, thus deterministic
    is_950_1block = (
        get_gfx() == "gfx950"
        and seqlen_k <= 256
        and hdim_q > 64
        and hdim_q <= 128
        and hdim_q % 8 == 0
    )

    def can_impl_fmha_v3_bwd_gfx950():
        ret = get_gfx() == "gfx950"
        ret &= alibi_slopes is None
        ret &= bias is None
        ret &= dbias is None
        ret &= dropout_p == 0.0
        ret &= not deterministic or is_950_1block
        ret &= nhead_q % nhead_k == 0
        ret &= (
            (hdim_q > 64 and hdim_q <= 128) or (hdim_q == 192 and hdim_v == 128)
        ) and hdim_q % 8 == 0
        ret &= not swa
        return ret

    can_impl_fmha_v3_bwd_ |= can_impl_fmha_v3_bwd_gfx950()

    if (
        can_impl_fmha_v3_bwd_ and seqlen_q > 16
    ):  # ck fmha bwd has optimization for seqlen_q <= 16
        if dq is not None:
            dq.zero_()
        (
            dq,
            dk,
            dv,
            softmax_d,
        ) = fmha_v3_bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            False if is_950_1block else deterministic,
            False if is_950_1block else is_v3_atomic_fp32,
            how_v3_bf16_cvt,
            dq,
            dk,
            dv,
            alibi_slopes,
            rng_state,
            None,
        )
    else:
        (
            dq,
            dk,
            dv,
            softmax_d,
        ) = mha_bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            dropout_p,
            softmax_scale,
            causal,
            window_size_left,
            window_size_right,
            deterministic,
            dq,
            dk,
            dv,
            dbias,
            bias,
            alibi_slopes,
            rng_state,
            None,
            sink,
            d_sink,
            # custom_build_args={"md_name": md_name, "blob_gen_cmd": blob_gen_cmd},
        )
    return softmax_d


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        is_grad_enabled,
        is_v3_atomic_fp32: Optional[bool] = True,
        how_v3_bf16_cvt: Optional[int] = 1,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        sink_ptr: Optional[Tensor] = None,
        num_splits: int = 0,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_q_og = q.size(3)
        head_size_v_og = v.size(3)
        if head_size_q_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_q_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_q_og % 8])
        if head_size_v_og % 8 != 0:
            v = torch.nn.functional.pad(v, [0, 8 - head_size_v_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_forward(
            q,
            k,
            v,
            dropout_p,
            softmax_scale,
            causal=causal,
            window_size_left=int(window_size[0]),
            window_size_right=int(window_size[1]),
            sink_size=int(window_size[2]) if len(window_size) == 3 else 0,
            bias=bias,
            alibi_slopes=alibi_slopes,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            return_lse=return_lse,
            return_softmax=return_softmax and dropout_p > 0,
            how_v3_bf16_cvt=how_v3_bf16_cvt,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens_kv,
            sink_ptr=sink_ptr,  # fwd kernel still uses sink_ptr naming
            num_splits=num_splits,
        )
        if is_grad:
            assert return_lse
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.bias = bias
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size_q_og = head_size_q_og
            ctx.is_v3_atomic_fp32 = is_v3_atomic_fp32
            ctx.how_v3_bf16_cvt = how_v3_bf16_cvt
        out = out_padded[..., :head_size_v_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse, rng_state = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        bias = ctx.bias
        dbias = torch.empty_like(bias) if bias is not None else None
        head_size_q_og = ctx.head_size_q_og
        head_size_v_og = dout.size(3)
        dout_padded = dout
        if head_size_v_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_v_og % 8])
        _flash_attn_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            dbias,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            int(ctx.window_size[0]),
            int(ctx.window_size[1]),
            ctx.bias,
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state,
            ctx.is_v3_atomic_fp32,
            ctx.how_v3_bf16_cvt,
            sink=None,
            d_sink=None,
        )
        dq = dq[..., :head_size_q_og]  # We could have padded the head dimension
        dk = dk[..., :head_size_q_og]
        dv = dv[..., :head_size_v_og]
        # Forward positional args order:
        #  1 q
        #  2 k
        #  3 v
        #  4 dropout_p
        #  5 softmax_scale
        #  6 causal
        #  7 window_size (tuple - no grad)
        #  8 bias
        #  9 alibi_slopes
        # 10 deterministic
        # 11 return_lse
        # 12 return_softmax
        # 13 is_grad_enabled
        # 14 is_v3_atomic_fp32
        # 15 how_v3_bf16_cvt
        # 16 cu_seqlens_q
        # 17 cu_seqlens_kv
        # 18 sink_ptr (fwd-only sink scores; not differentiable via autograd.
        #              bwd sink gradient d_sink is computed inside mha_bwd kernel,
        #              not returned here as a positional gradient.)
        # 19 num_splits
        # Need to return exactly 19 gradient entries.
        return (
            dq,  # q
            dk,  # k
            dv,  # v
            None,  # dropout_p
            None,  # softmax_scale
            None,  # causal
            None,  # window_size
            dbias,  # bias
            None,  # alibi_slopes
            None,  # deterministic
            None,  # return_lse
            None,  # return_softmax
            None,  # is_grad_enabled
            None,  # is_v3_atomic_fp32
            None,  # how_v3_bf16_cvt
            None,  # cu_seqlens_q
            None,  # cu_seqlens_kv
            None,  # sink_ptr (not differentiable; bwd uses sink/d_sink args separately)
            None,  # num_splits
        )


def flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1, 0),  # -1 means infinite context window, 0 means no sink
    bias=None,
    alibi_slopes=None,
    deterministic=True,
    return_lse=False,
    return_attn_probs=False,
    how_v3_bf16_cvt=1,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_kv: Optional[torch.Tensor] = None,
    sink_ptr: Optional[Tensor] = None,
    num_splits: int = 0,
):
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim_q)
        k: (batch_size, seqlen, nheads_k, headdim_q)
        v: (batch_size, seqlen, nheads_k, headdim_v)
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim_q).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        bias: (seqlen_q, seqlen_k)
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
        cu_seqlens_q: (batch_size + 1,). The cumulative sequence lengths of the query sequences.
        cu_seqlens_kv: (batch_size + 1,). The cumulative sequence lengths of the key/value sequences.
        num_splits: int. Number of key/value splits for the native split-K forward path.
            0 (default) lets aiter decide via a heuristic; 1 disables split-K (uses the
            standard CK/ASM dispatch); >=2 forces the native split-K kernel with that many
            splits when that path is applicable, otherwise num_splits is ignored.
    Return:
        out: (batch_size, seqlen, nheads, headdim_v).
        softmax_lse [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """
    if not ENABLE_CK:
        from .triton.attention.mha import flash_attn_func as flash_attn_func_triton

        return flash_attn_func_triton(
            q=q,
            k=k,
            v=v,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            bias=bias,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
            sink=sink_ptr,
        )
    return FlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        torch.is_grad_enabled(),
        True,  # is_v3_atomic_fp32
        how_v3_bf16_cvt,
        cu_seqlens_q,
        cu_seqlens_kv,
        sink_ptr,
        num_splits,
    )


def _flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    cu_seqlens_q_padded: Optional[torch.Tensor],
    cu_seqlens_k_padded: Optional[torch.Tensor],
    max_seqlen_q: int,
    max_seqlen_k: int,
    min_seqlen_q: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    logits_soft_cap: float = 0.0,
    window_size_left: int = -1,
    window_size_right: int = -1,
    sink_size: int = 0,
    bias: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    return_softmax: bool = False,
    how_v3_bf16_cvt: Optional[int] = 1,
    block_table: Optional[torch.Tensor] = None,
    out: Optional[torch.Tensor] = None,
    zero_tensors: bool = False,
    sink_ptr: Optional[Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    _, nhead_q, hdim_q = q.shape
    batch_size = cu_seqlens_q.numel() - 1

    nhead_k = v.shape[-2]
    hdim_v = v.shape[-1]
    if sink_ptr is not None:
        assert sink_ptr.device == q.device, "sink_ptr must be on the same device as q"
        assert sink_ptr.shape[0] == nhead_q, "sink_ptr has incorrect shape"
        if sink_ptr.dtype != torch.float32:
            sink_ptr = sink_ptr.to(torch.float32)
    # mask
    window_size_left = -1 if window_size_left >= max_seqlen_k else window_size_left
    window_size_right = -1 if window_size_right >= max_seqlen_k else window_size_right
    sink_size = 0 if sink_size >= max_seqlen_k else sink_size
    mask = causal == True and window_size_left == -1  # causal mask
    nmask = (
        causal == False and window_size_left == -1 and window_size_right == -1
    )  # no mask
    swa = (window_size_left > 0) or (window_size_right > 0)

    def is_fmha_v3_fp8():
        ret = get_gfx() in ("gfx942", "gfx950")
        ret = ret and (hdim_q == 128)
        ret = ret and (q.dtype == dtypes.fp8)
        ret = ret and (
            q_descale is not None and k_descale is not None and v_descale is not None
        )
        # support per tensor and per head quant scale
        ret = ret and (
            q_descale.shape == (1,) or q_descale.shape == (batch_size, nhead_k)
        )
        ret = ret and (
            q_descale.shape == k_descale.shape and q_descale.shape == v_descale.shape
        )
        return ret

    def can_impl_fmha_v3_fwd():
        # basic
        # fmha v3 varlen is hand-written gfx9 ASM; non-gfx9 must fall back to
        # ck-tile (mha_varlen_fwd, the else branch below).
        ret = get_gfx() in ("gfx942", "gfx950")
        ret = ret and (alibi_slopes is None)
        ret = ret and (bias is None)
        ret = ret and (dropout_p == 0.0)
        ret = ret and (hdim_v == 128)
        ret = ret and (hdim_q == 128 or hdim_q == 192)
        ret = ret and (nhead_q % nhead_k == 0)
        ret = ret and (not swa)
        ret = ret and (q.dtype == dtypes.bf16 or is_fmha_v3_fp8())
        ret = ret and logits_soft_cap == 0.0
        # FP8 ASM kernels assemble the GQA-shift from a fixed log2 table
        # (1,2,4,8,16); arbitrary divisor ratios route to CK.
        if is_fmha_v3_fp8():
            gqa_ratio = nhead_q // nhead_k
            ret = ret and ((gqa_ratio & (gqa_ratio - 1)) == 0)
        return ret

    def can_impl_fmha_fwd_with_sink_varlen_asm():
        # gfx1250 ASM bf16 packed/varlen forward (fmha_fwd_with_sink_varlen_asm).
        # Packed THD (batch folded into the token axis); no dropout / swa /
        # quant / alibi / bias / paged (block_table) / logits-soft-cap.  Sink
        # logits (per-Q-head fp32) supported; sink-token (sink_size) not.
        ret = get_gfx() == "gfx1250"
        ret = ret and (q.dtype == dtypes.bf16)
        ret = ret and (hdim_q in (64, 128))
        ret = ret and (hdim_v == hdim_q)
        ret = ret and (nhead_q % nhead_k == 0)
        ret = ret and (not swa)
        ret = ret and (sink_size == 0)
        ret = ret and (alibi_slopes is None and bias is None)
        ret = ret and (dropout_p == 0.0)
        ret = ret and (logits_soft_cap == 0.0)
        ret = ret and (block_table is None)
        # The varlen ASM wrapper carries no physical-padding arrays; route any
        # padded-cu request to CK (mha_varlen_fwd) which understands them.
        ret = ret and (cu_seqlens_q_padded is None and cu_seqlens_k_padded is None)
        ret = ret and (q_descale is None and k_descale is None and v_descale is None)
        # Per-hdim sink eligibility (mirrors the fixed-batch path):
        #   D128 (`_rxy`) binaries compile ENABLE_SINK=0 and ignore the sink
        #   buffer, so routing a caller's sink_ptr to them would silently drop
        #   the sink term -- fall back to CK whenever sink_ptr is set.
        #   D64  (`_rxy_sink`) binaries compile ENABLE_SINK=1 and ALWAYS read
        #   SINK, so calling with sink_ptr=None would dereference a null pointer
        #   -- require an explicit sink for D64 and fall back to CK otherwise.
        if hdim_q == 128:
            ret = ret and (sink_ptr is None)
        elif hdim_q == 64:
            ret = ret and (sink_ptr is not None)
        return ret

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    if can_impl_fmha_fwd_with_sink_varlen_asm():
        # gfx1250 packed/varlen ASM bf16 path.  q/k/v are packed THD; the kernel
        # requires dense packing (the wrapper calls `.contiguous()` defensively)
        # and carries no strides.  softmax_scale is forwarded as-is (the kernel
        # applies it internally to Q·K^T).  sink_ptr is passed through verbatim;
        # `can_impl_fmha_fwd_with_sink_varlen_asm` already enforces the per-hdim
        # (D128→no sink, D64→sink) contract so we never feed a null sink to a
        # D64 binary that unconditionally reads it.
        out, lse_asm = fmha_fwd_with_sink_varlen_asm(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            float(softmax_scale),
            bool(causal),
            True,
            sink_ptr,
            out,
        )
        # The ASM kernel writes packed lse (total_q, nheads, 1); the varlen API
        # convention (mha_varlen_fwd) is (nheads, total_q).  Reshape so callers
        # and the autograd backward see a consistent layout regardless of path.
        softmax_lse = lse_asm.squeeze(-1).transpose(0, 1).contiguous()
        S_dmask = torch.empty((0,), dtype=torch.float32, device=q.device)
        rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)
    elif can_impl_fmha_v3_fwd():
        out, softmax_lse, S_dmask, rng_state = fmha_v3_varlen_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            min_seqlen_q,
            dropout_p,
            softmax_scale,
            logits_soft_cap,
            zero_tensors,
            causal,
            window_size_left,
            window_size_right,
            return_lse,
            return_softmax,
            how_v3_bf16_cvt,
            out,
            block_table,
            bias,
            alibi_slopes,
            q_descale,
            k_descale,
            v_descale,
            None,
            cu_seqlens_q_padded,
            cu_seqlens_k_padded,
            # custom_build_args={"md_name": md_name, "blob_gen_cmd": blob_gen_cmd},
        )
    else:
        # Input validation for padded cumulative arrays if provided
        def _validate(name: str, t: torch.Tensor):
            assert t.dim() == 1, f"{name} must be 1D"
            assert t.is_cuda, f"{name} must be on CUDA"
            assert t.dtype == torch.int32, f"{name} must be int32, actual: {t.dtype}"
            assert t.is_contiguous(), f"{name} must be contiguous"
            assert (
                t.numel() == cu_seqlens_q.numel()
            ), f"{name} length mismatch with batch"
            # light monotonic check (first and last only; deeper check in C++)
            assert t[0].item() == 0, f"{name}[0] must be 0"

        if cu_seqlens_q_padded is not None:
            _validate("cu_seqlens_q_padded", cu_seqlens_q_padded)
        if cu_seqlens_k_padded is not None:
            _validate("cu_seqlens_k_padded", cu_seqlens_k_padded)
        out, softmax_lse, S_dmask, rng_state = mha_varlen_fwd(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            min_seqlen_q,
            dropout_p,
            softmax_scale,
            logits_soft_cap,
            zero_tensors,
            causal,
            window_size_left,
            window_size_right,
            sink_size,
            return_lse,
            return_softmax,
            out=out,
            block_table=block_table,
            bias=bias,
            alibi_slopes=alibi_slopes,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            gen=None,
            cu_seqlens_q_padded=cu_seqlens_q_padded,
            cu_seqlens_k_padded=cu_seqlens_k_padded,
            sink_ptr=sink_ptr,
        )
    return out, softmax_lse, S_dmask, rng_state


def _flash_attn_varlen_backward(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    dq: Optional[torch.Tensor],
    dk: Optional[torch.Tensor],
    dv: Optional[torch.Tensor],
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    window_size_left: int,
    window_size_right: int,
    alibi_slopes: Optional[torch.Tensor],
    deterministic: bool,
    rng_state: Optional[torch.Tensor] = None,
    is_v3_atomic_fp32: Optional[bool] = True,
    how_v3_bf16_cvt: Optional[int] = 1,
    zero_tensors: bool = False,
    cu_seqlens_q_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_padded: Optional[torch.Tensor] = None,
    sink: Optional[Tensor] = None,
    d_sink: Optional[Tensor] = None,
) -> torch.Tensor:
    _, nhead_q, hdim_q = q.shape

    nhead_k = v.shape[-2]
    hdim_v = v.shape[-1]

    # mask
    window_size_left = -1 if window_size_left >= max_seqlen_k else window_size_left
    window_size_right = -1 if window_size_right >= max_seqlen_k else window_size_right
    mask = causal == True and window_size_left == -1  # causal mask
    nmask = (
        causal == False and window_size_left == -1 and window_size_right == -1
    )  # no mask
    swa = (window_size_left > 0) or (window_size_right > 0)

    def pssk():
        # only for hd64 a32 causal/no causal, fp16/bf16-rtne/rtna/rtz cases
        # FIXME: Currently we only support mask_type == mask_enum::no_mask
        # Because python side only support mask_enum::bottom_right
        # However v3 kernel only support mask_enum::top_left
        # bwd_hd64_bf16_a32_rtne_pssk_group
        # bwd_hd64_bf16_a32_rtna_pssk_group
        # bwd_hd64_bf16_a32_rtz_pssk_group
        # bwd_hd64_bf16_causal_a32_rtne_pssk_group
        # bwd_hd64_bf16_causal_a32_rtna_pssk_group
        # bwd_hd64_bf16_causal_a32_rtz_pssk_group
        # bwd_hd64_fp16_a32_pssk_group
        # bwd_hd64_fp16_causal_a32_pssk_group
        # bwd_hd128_bf16_a32_rtne_pssk_group
        # bwd_hd128_bf16_a32_rtna_pssk_group
        # bwd_hd128_bf16_a32_rtz_pssk_group
        # bwd_hd128_bf16_causal_a32_rtne_pssk_group
        # bwd_hd128_bf16_causal_a32_rtna_pssk_group
        # bwd_hd128_bf16_causal_a32_rtz_pssk_group
        # bwd_hd128_fp16_a32_pssk_group
        # bwd_hd128_fp16_causal_a32_pssk_group
        ret = (
            is_v3_atomic_fp32 == True
        )  # nhead_stride_dq_acc >= stride_dq_acc must be guaranteed
        ret &= hdim_q == 64 or hdim_q == 128

        return ret

    def psskddv():
        # bwd_hd128_bf16_a32_rtne_psskddv_group
        # bwd_hd128_bf16_a32_rtna_psskddv_group
        # bwd_hd128_bf16_a32_rtz_psskddv_group
        # bwd_hd128_bf16_causal_a32_rtne_psskddv_group
        # bwd_hd128_bf16_causal_a32_rtna_psskddv_group
        # bwd_hd128_bf16_causal_a32_rtz_psskddv_group
        # bwd_hd128_fp16_a32_psskddv_group
        # bwd_hd128_fp16_causal_a32_psskddv_group
        ret = (
            is_v3_atomic_fp32 == True
        )  # nhead_stride_dq_acc >= stride_dq_acc must be guaranteed
        ret &= hdim_q >= 64 and hdim_q <= 192

        return ret

    def can_impl_fmha_v3_bwd():
        # basic
        ret = get_gfx() == "gfx942"
        ret &= alibi_slopes is None
        # ret &= bias is None
        # ret &= dbias is None
        ret &= dropout_p == 0.0
        ret &= deterministic == False
        ret &= hdim_q == hdim_v
        ret &= nhead_q % nhead_k == 0
        ret &= hdim_q >= 64 and hdim_q <= 192 and hdim_q % 8 == 0
        ret &= not swa
        ret &= pssk() or psskddv()

        return ret

    def can_impl_fmha_v3_bwd_gfx950():
        ret = get_gfx() == "gfx950"
        ret &= alibi_slopes is None
        # ret &= bias is None
        # ret &= dbias is None
        ret &= dropout_p == 0.0
        ret &= deterministic == False
        ret &= hdim_q == hdim_v
        ret &= nhead_q % nhead_k == 0
        ret &= hdim_q > 64 and hdim_q <= 128 and hdim_q % 8 == 0
        ret &= not swa

        return ret

    can_impl_fmha_v3_bwd_ = can_impl_fmha_v3_bwd() or can_impl_fmha_v3_bwd_gfx950()
    # dq, dk, dv are allocated by us so they should already be contiguous
    dout, q, k, v, out = [maybe_contiguous(x) for x in (dout, q, k, v, out)]

    if can_impl_fmha_v3_bwd_:
        (
            dq,
            dk,
            dv,
            softmax_d,
        ) = fmha_v3_varlen_bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            zero_tensors,
            causal,
            window_size_left,
            window_size_right,
            deterministic,
            is_v3_atomic_fp32,
            how_v3_bf16_cvt,
            dq,
            dk,
            dv,
            alibi_slopes,
            rng_state,
            None,
            cu_seqlens_q_padded,
            cu_seqlens_k_padded,
        )
    else:
        (
            dq,
            dk,
            dv,
            softmax_d,
        ) = mha_varlen_bwd(
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p,
            softmax_scale,
            zero_tensors,
            causal,
            window_size_left,
            window_size_right,
            deterministic,
            dq,
            dk,
            dv,
            alibi_slopes,
            rng_state,
            None,
            cu_seqlens_q_padded,
            cu_seqlens_k_padded,
            sink=sink,
            d_sink=d_sink,
            # custom_build_args={"md_name": md_name, "blob_gen_cmd": blob_gen_cmd},
        )
    return softmax_d


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_q,
        dropout_p,
        softmax_scale,
        logits_soft_cap,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_softmax,
        block_table,
        out,
        is_grad_enabled,
        cu_seqlens_q_padded=None,
        cu_seqlens_k_padded=None,
        is_v3_atomic_fp32: Optional[bool] = True,
        how_v3_bf16_cvt: Optional[int] = 1,
        sink_ptr=None,
    ):
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        head_size_q_og = q.size(-1)
        head_size_v_og = v.size(-1)
        if head_size_q_og % 8 != 0:
            q = torch.nn.functional.pad(q, [0, 8 - head_size_q_og % 8])
            k = torch.nn.functional.pad(k, [0, 8 - head_size_q_og % 8])
        if head_size_v_og % 8 != 0:
            v = torch.nn.functional.pad(v, [0, 8 - head_size_v_og % 8])
        out_padded, softmax_lse, S_dmask, rng_state = _flash_attn_varlen_forward(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            cu_seqlens_q_padded,
            cu_seqlens_k_padded,
            max_seqlen_q,
            max_seqlen_k,
            min_seqlen_q,
            dropout_p,
            softmax_scale,
            causal=causal,
            logits_soft_cap=logits_soft_cap,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            sink_size=window_size[2] if len(window_size) > 2 else 0,
            bias=bias,
            alibi_slopes=alibi_slopes,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            return_lse=return_lse,
            return_softmax=return_softmax and dropout_p > 0,
            how_v3_bf16_cvt=how_v3_bf16_cvt,
            block_table=block_table,
            out=out,
            sink_ptr=sink_ptr,
        )
        if is_grad:
            assert return_lse
            ctx.save_for_backward(
                q, k, v, out_padded, softmax_lse, cu_seqlens_q, cu_seqlens_k, rng_state
            )
            ctx.dropout_p = dropout_p
            ctx.max_seqlen_q = max_seqlen_q
            ctx.max_seqlen_k = max_seqlen_k
            ctx.softmax_scale = softmax_scale
            ctx.causal = causal
            ctx.window_size = window_size
            ctx.bias = bias
            ctx.alibi_slopes = alibi_slopes
            ctx.deterministic = deterministic
            ctx.head_size_q_og = head_size_q_og
            ctx.is_v3_atomic_fp32 = is_v3_atomic_fp32
            ctx.how_v3_bf16_cvt = how_v3_bf16_cvt
            ctx.cu_seqlens_q_padded = cu_seqlens_q_padded
            ctx.cu_seqlens_k_padded = cu_seqlens_k_padded

        out = out_padded[..., :head_size_v_og]

        result = [out]
        if return_lse:
            result.append(softmax_lse)
        if return_softmax:
            result.append(S_dmask)

        return result[0] if len(result) == 1 else tuple(result)

    @staticmethod
    def backward(ctx, dout, *args):
        (
            q,
            k,
            v,
            out,
            softmax_lse,
            cu_seqlens_q,
            cu_seqlens_k,
            rng_state,
        ) = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        bias = ctx.bias
        dbias = torch.empty_like(bias) if bias is not None else None
        head_size_q_og = ctx.head_size_q_og
        head_size_v_og = dout.size(2)
        dout_padded = dout
        if head_size_v_og % 8 != 0:
            dout_padded = torch.nn.functional.pad(dout, [0, 8 - head_size_v_og % 8])
        # TODO - dbias
        _flash_attn_varlen_backward(
            dout_padded,
            q,
            k,
            v,
            out,
            softmax_lse,
            dq,
            dk,
            dv,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            ctx.dropout_p,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            ctx.alibi_slopes,
            ctx.deterministic,
            rng_state=rng_state,
            is_v3_atomic_fp32=ctx.is_v3_atomic_fp32,
            how_v3_bf16_cvt=ctx.how_v3_bf16_cvt,
            cu_seqlens_q_padded=ctx.cu_seqlens_q_padded,
            cu_seqlens_k_padded=ctx.cu_seqlens_k_padded,
            sink=None,
            d_sink=None,
        )
        dq = dq[..., :head_size_q_og]  # We could have padded the head dimension
        dk = dk[..., :head_size_q_og]
        dv = dv[..., :head_size_v_og]
        # Forward signature (positional args):
        # q, k, v,
        # cu_seqlens_q, cu_seqlens_k,
        # max_seqlen_q, max_seqlen_k,
        # min_seqlen_q,
        # dropout_p, softmax_scale, logits_soft_cap,
        # causal,
        # window_size,
        # bias,
        # alibi_slopes,
        # deterministic,
        # return_lse, return_softmax,
        # block_table,
        # out,
        # is_grad_enabled,
        # cu_seqlens_q_padded, cu_seqlens_k_padded,
        # is_v3_atomic_fp32, how_v3_bf16_cvt,
        # sink_ptr (fwd-only sink scores; not differentiable via autograd.
        #           bwd sink gradient d_sink is computed inside mha_varlen_bwd kernel,
        #           not returned here as a positional gradient.)
        # We only have gradients for q,k,v (dq,dk,dv) and possibly bias (dbias). Others are None.
        return (
            dq,  # q
            dk,  # k
            dv,  # v
            None,  # cu_seqlens_q
            None,  # cu_seqlens_k
            None,  # max_seqlen_q
            None,  # max_seqlen_k
            None,  # min_seqlen_q
            None,  # dropout_p
            None,  # softmax_scale
            None,  # logits_soft_cap
            None,  # causal
            None,  # window_size (tuple treated as arg)
            dbias,  # bias
            None,  # alibi_slopes
            None,  # deterministic
            None,  # return_lse
            None,  # return_softmax
            None,  # block_table
            None,  # out
            None,  # is_grad_enabled
            None,  # cu_seqlens_q_padded
            None,  # cu_seqlens_k_padded
            None,  # is_v3_atomic_fp32
            None,  # how_v3_bf16_cvt
            None,  # sink_ptr (not differentiable; bwd uses sink/d_sink args separately)
        )


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    min_seqlen_q=0,
    dropout_p=0.0,
    softmax_scale=None,
    logits_soft_cap=0.0,
    causal=False,
    window_size=(-1, -1, 0),  # -1 means infinite context window, 0 means no sink
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    how_v3_bf16_cvt=1,
    block_table=None,
    out=None,
    cu_seqlens_q_padded: Optional[torch.Tensor] = None,
    cu_seqlens_k_padded: Optional[torch.Tensor] = None,
    sink_ptr: Optional[Tensor] = None,
):
    if block_table is not None and (
        cu_seqlens_q_padded is not None or cu_seqlens_k_padded is not None
    ):
        raise NotImplementedError(
            "Paged/Split-KV attention (using block_table) does not currently support "
            "physical sequence padding (cu_seqlens_*_padded)."
        )
    """dropout_p should be set to 0.0 during evaluation
    Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Arguments:
        q: (total_q, nheads, headdim_q), where total_q = total number of query tokens in the batch.
        k: (total_k, nheads_k, headdim_q), where total_k = total number of key tokens in the batch.
        v: (total_k, nheads_k, headdim_v), where total_k = total number of key tokens in the batch.
        cu_seqlens_q: (batch_size + 1,), dtype dtypes.i32. The cumulative sequence lengths
           of the sequences in the batch, used to index into q.
        cu_seqlens_k: (batch_size + 1,), dtype dtypes.i32. The cumulative sequence lengths
           of the sequences in the batch, used to index into kv.
        max_seqlen_q: int. Maximum query sequence length in the batch.
        max_seqlen_k: int. Maximum key sequence length in the batch.
        min_seqlen_q: int. Minimum query sequence length for chunked prefill.
        dropout_p: float. Dropout probability.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim_q).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        bias: (seqlen_q, seqlen_k)
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        deterministic: bool. Whether to use the deterministic implementation of the backward pass,
            which is slightly slower and uses more memory. The forward pass is always deterministic.
        return_attn_probs: bool. Whether to return the attention probabilities. This option is for
           testing only. The returned probabilities are not guaranteed to be correct
           (they might not have the right scaling).
    Return:
        out: (total, nheads, headdim_v).
        softmax_lse [optional, if return_attn_probs=True]: (nheads, total_q_seqlen). The
            logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax
            normalization factor).
        S_dmask [optional, if return_attn_probs=True]: (batch_size, nheads, seqlen, seqlen).
            The output of softmax (possibly with different scaling). It also encodes the dropout
            pattern (negative means that location was dropped, nonnegative means it was kept).
    """

    # Try the PR3039 gfx1250 prefill ASM path before FlyDSL can claim it.
    def can_try_gfx1250_fmha_fwd_with_sink_varlen_asm():
        # Keep this public-router gate intentionally narrow so the PR3039
        # prefill ASM path can be measured without changing decode or other
        # FlyDSL/CK coverage.
        if get_gfx() != "gfx1250" or q.dtype != dtypes.bf16:
            return False
        hdim_q = q.shape[-1]
        hdim_v = v.shape[-1]
        nhead_q = q.shape[-2]
        nhead_k = k.shape[-2]
        if hdim_q not in (64, 128) or hdim_v != hdim_q:
            return False
        if nhead_q % nhead_k != 0:
            return False
        if not causal or dropout_p != 0.0 or logits_soft_cap != 0.0:
            return False
        if window_size[0] != -1 or window_size[1] != -1:
            return False
        sink_size = window_size[2] if len(window_size) > 2 else 0
        if sink_size != 0:
            return False
        if bias is not None or alibi_slopes is not None or block_table is not None:
            return False
        if cu_seqlens_q_padded is not None or cu_seqlens_k_padded is not None:
            return False
        if hdim_q == 64:
            return sink_ptr is not None
        return sink_ptr is None

    if can_try_gfx1250_fmha_fwd_with_sink_varlen_asm():
        return FlashAttnVarlenFunc.apply(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            min_seqlen_q,
            dropout_p,
            softmax_scale,
            logits_soft_cap,
            causal,
            window_size,
            bias,
            alibi_slopes,
            deterministic,
            return_lse,
            return_attn_probs,
            block_table,
            out,
            torch.is_grad_enabled(),
            cu_seqlens_q_padded,
            cu_seqlens_k_padded,
            True,
            how_v3_bf16_cvt,
            sink_ptr,
        )

    # FlyDSL path returns result if supported, None otherwise.
    from .flydsl.fmha_kernels import flydsl_flash_attn_varlen_func

    _flydsl_result = flydsl_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        return_lse=return_lse,
        dropout_p=dropout_p,
        window_size=window_size,
        bias=bias,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_attn_probs=return_attn_probs,
        block_table=block_table,
        out=out,
        sink=sink_ptr,
    )
    if _flydsl_result is not None:
        return _flydsl_result

    if not ENABLE_CK:
        from .triton.attention.mha import (
            flash_attn_varlen_func as flash_attn_varlen_func_triton,
        )

        return flash_attn_varlen_func_triton(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            bias=bias,
            alibi_slopes=alibi_slopes,
            deterministic=deterministic,
            return_lse=return_lse,
            return_attn_probs=return_attn_probs,
            block_table=block_table,
            out=out,
            sink=sink_ptr,
        )
    return FlashAttnVarlenFunc.apply(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_q,
        dropout_p,
        softmax_scale,
        logits_soft_cap,
        causal,
        window_size,
        bias,
        alibi_slopes,
        deterministic,
        return_lse,
        return_attn_probs,
        block_table,
        out,
        torch.is_grad_enabled(),
        cu_seqlens_q_padded,
        cu_seqlens_k_padded,
        True,
        how_v3_bf16_cvt,
        sink_ptr,
    )


def mha_batch_prefill_fake_tensors(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    logits_soft_cap: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    sink_size: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    # Per-tensor descale for PERTENSOR mode
    q_descale: Optional[torch.Tensor] = None,  # [1] per-tensor Q descale
    k_descale: Optional[torch.Tensor] = None,  # [1] per-tensor K descale
    v_descale: Optional[torch.Tensor] = None,  # [1] per-tensor V descale
    # Per-page descale for KV_BLOCKSCALE mode (mutually exclusive with k_descale/v_descale)
    kv_block_descale: Optional[torch.Tensor] = None,  # [num_block, num_kv_head, 2]
    sink_ptr: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
    kv_last_page_lens: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    seqlen_k: Optional[torch.Tensor] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    is_vectorized = k.dim() == 5 and v.dim() == 5
    is_linear = (k.dim() == 4 and v.dim() == 4) or (k.dim() == 3 and v.dim() == 3)
    if not (is_vectorized or is_linear):
        raise ValueError(
            "Batch prefill requires 5D vectorized, 4D linear, or 3D linear (page_size=1) K/V"
            " tensors"
        )
    num_heads = q.size(1)  # num_heads = q.sizes()[1]
    head_size_v = v.size(-2) if is_vectorized else v.size(-1)
    total_q = q.size(0)  # total_q = q.size(0)

    if out is None:
        out_dtype = dtypes.bf16 if q.dtype == dtypes.fp8 else q.dtype
        out = torch.empty(
            (total_q, num_heads, head_size_v),  # {total_q, num_heads, head_size_v}
            dtype=out_dtype,
            device=q.device,
            requires_grad=q.requires_grad,
        )

    if return_softmax_lse:
        softmax_lse = torch.empty(
            (num_heads, total_q),  # {num_heads, total_q}
            dtype=torch.float32,
            device=q.device,
        )
    else:
        softmax_lse = torch.empty((0,), dtype=torch.float32, device=q.device)

    if return_dropout_randval:
        assert dropout_p > 0, "return_dropout_randval requires p_dropout > 0"
        p = torch.empty(
            (num_heads, total_q, max_seqlen_k),  # {num_heads, total_q, max_seqlen_k}
            dtype=torch.uint8,
            device=q.device,
        )
    else:
        p = torch.empty((0,), device=q.device)

    rng_state = torch.empty((2,), dtype=torch.int64, device=q.device)

    return (out, softmax_lse, p, rng_state)


@compile_ops(
    "module_mha_batch_prefill",
    fc_name="mha_batch_prefill",
    gen_func=cmdGenFunc_mha_batch_prefill,
    gen_fake=mha_batch_prefill_fake_tensors,
)
def mha_batch_prefill(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    cu_seqlens_q: Tensor,
    kv_indptr: Tensor,
    kv_page_indices: Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    logits_soft_cap: float,
    zero_tensors: bool,
    is_causal: bool,
    window_size_left: int,
    window_size_right: int,
    sink_size: int,
    return_softmax_lse: bool,
    return_dropout_randval: bool,
    out: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    alibi_slopes: Optional[Tensor] = None,
    # Per-tensor descale for PERTENSOR mode
    q_descale: Optional[torch.Tensor] = None,  # [1] per-tensor Q descale
    k_descale: Optional[torch.Tensor] = None,  # [1] per-tensor K descale
    v_descale: Optional[torch.Tensor] = None,  # [1] per-tensor V descale
    # Per-page descale for KV_BLOCKSCALE mode (mutually exclusive with k_descale/v_descale)
    kv_block_descale: Optional[torch.Tensor] = None,  # [num_block, num_kv_head, 2]
    kv_last_page_lens: Optional[Tensor] = None,
    block_table: Optional[Tensor] = None,
    seqlen_k: Optional[Tensor] = None,
    sink_ptr: Optional[Tensor] = None,
    gen: Optional[Generator] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]: ...


def _mha_batch_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float,
    softmax_scale: float,
    causal: bool,
    logits_soft_cap: float = 0.0,
    window_size_left: int = -1,
    window_size_right: int = -1,
    sink_size: int = 0,
    bias: Optional[torch.Tensor] = None,
    alibi_slopes: Optional[torch.Tensor] = None,
    return_lse: bool = False,
    return_softmax: bool = False,
    zero_tensors: bool = False,
    out: torch.Tensor = None,
    kv_last_page_lens: torch.Tensor = None,
    block_table: torch.Tensor = None,
    seqlen_k: torch.Tensor = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    kv_block_descale: Optional[
        torch.Tensor
    ] = None,  # [num_block, num_kv_head, 2] per-page K/V descales
    sink_ptr: Optional[Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = mha_batch_prefill(
        q,
        k,
        v,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        logits_soft_cap,
        zero_tensors,
        causal,
        window_size_left,
        window_size_right,
        sink_size,
        return_lse,
        return_softmax,
        out,
        bias,
        alibi_slopes,
        q_descale,
        k_descale,
        v_descale,
        kv_block_descale,
        kv_last_page_lens,
        block_table,
        seqlen_k,
        sink_ptr,
        None,
    )
    return out, softmax_lse, S_dmask, rng_state


def mha_batch_prefill_func(
    q,
    k,
    v,
    cu_seqlens_q,
    kv_indptr,
    kv_page_indices,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    logits_soft_cap=0.0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    alibi_slopes=None,
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
    out=None,
    kv_last_page_lens=None,
    block_table=None,
    seqlen_k=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    kv_block_descale=None,  # [num_block, num_kv_head, 2] per-page K/V descales
    sink_ptr=None,
    sink_size: int = 0,
):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if sink_ptr is not None:
        assert sink_ptr.device == q.device, "sink_ptr must be on the same device as q"
        assert sink_ptr.shape[0] == q.size(1), "sink_ptr has incorrect shape"
        if sink_ptr.dtype != torch.float32:
            sink_ptr = sink_ptr.to(torch.float32)
    head_size_q_og = q.size(-1)
    # 16 bytes = 128-bit (dwordx4) vector width assumed by CK kernels.
    k_vector_size = 16 // k.element_size()
    is_vectorized = k.dim() == 5 and v.dim() == 5
    is_linear = (k.dim() == 4 and v.dim() == 4) or (k.dim() == 3 and v.dim() == 3)
    if not (is_vectorized or is_linear):
        raise ValueError(
            "Batch prefill requires 5D vectorized, 4D linear, or 3D linear (page_size=1) K/V"
            " tensors"
        )
    head_size_v_og = v.size(-2) if is_vectorized else v.size(-1)
    if head_size_q_og % k_vector_size != 0 or head_size_v_og % k_vector_size != 0:
        raise ValueError("Batch prefill requires head size divisible by vector size")
    if is_vectorized:
        if k.size(-3) * k_vector_size != head_size_q_og:
            raise ValueError("K vectorized layout does not match Q head size")
        if k.size(-2) % k_vector_size != 0:
            raise ValueError(
                "Vectorized KV requires page size divisible by vector size"
            )
        if v.size(-1) != k_vector_size:
            raise ValueError("Vectorized KV requires last dim equal to vector size")
    else:
        if k.size(-1) != head_size_q_og:
            raise ValueError("K linear layout does not match Q head size")
        if k.size(1) != v.size(1) or k.size(2) != v.size(2):
            raise ValueError("K/V linear layout must match page size and head count")
    if k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("Batch prefill requires K/V with contiguous last dimension")
    out_padded, softmax_lse, S_dmask, rng_state = _mha_batch_prefill(
        q,
        k,
        v,
        cu_seqlens_q,
        kv_indptr,
        kv_page_indices,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        window_size_left=window_size[0],
        window_size_right=window_size[1],
        sink_size=sink_size,
        alibi_slopes=alibi_slopes,
        return_lse=return_lse,
        return_softmax=return_attn_probs and dropout_p > 0,
        out=out,
        kv_last_page_lens=kv_last_page_lens,
        block_table=block_table,
        seqlen_k=seqlen_k,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        kv_block_descale=kv_block_descale,
        sink_ptr=sink_ptr,
    )
    out = out_padded[..., :head_size_v_og]

    result = [out]
    if return_lse:
        result.append(softmax_lse)
    if return_attn_probs:
        result.append(S_dmask)

    return result[0] if len(result) == 1 else tuple(result)


def flash_attn_fp8_pertensor_func(
    q,
    k,
    v,
    q_descale,
    k_descale,
    v_descale,
    causal=False,
    window_size=(-1, -1, 0),  # -1 means infinite context window, 0 means no sink
    softmax_scale=None,
    sink_ptr=None,
):
    if not ENABLE_CK and sink_ptr is None:
        from .triton.attention.mha_v3 import (
            flash_attn_func as flash_attn_func_v3_triton,
        )

        return flash_attn_func_v3_triton(
            q=q,
            k=k,
            v=v,
            softmax_scale=softmax_scale,
            causal=causal,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size=(window_size[0], window_size[1]),
        )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    head_size_q_og = q.size(3)
    head_size_v_og = v.size(3)
    if head_size_q_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_q_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_q_og % 8])
    if head_size_v_og % 8 != 0:
        v = torch.nn.functional.pad(v, [0, 8 - head_size_v_og % 8])
    out_padded, _, _, _ = _flash_attn_forward(
        q,
        k,
        v,
        0.0,
        softmax_scale,
        causal=causal,
        window_size_left=int(window_size[0]),
        window_size_right=int(window_size[1]),
        sink_size=int(window_size[2]) if len(window_size) == 3 else 0,
        bias=None,
        alibi_slopes=None,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        return_lse=False,
        return_softmax=False,
        sink_ptr=sink_ptr,
    )
    out = out_padded[..., :head_size_v_og]
    return out


def flash_attn_varlen_fp8_pertensor_func(
    q,
    k,
    v,
    q_descale,
    k_descale,
    v_descale,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    min_seqlen_q=0,
    logits_soft_cap=0.0,
    causal=False,
    window_size=(-1, -1, 0),  # -1 means infinite context window
    softmax_scale=None,
    sink_ptr=None,
):
    if not ENABLE_CK and sink_ptr is None:
        from .triton.attention.mha_v3 import (
            flash_attn_varlen_func as flash_attn_varlen_func_v3_triton,
        )

        return flash_attn_varlen_func_v3_triton(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            window_size=(window_size[0], window_size[1]),
        )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    head_size_q_og = q.size(-1)
    head_size_v_og = v.size(-1)
    if head_size_q_og % 8 != 0:
        q = torch.nn.functional.pad(q, [0, 8 - head_size_q_og % 8])
        k = torch.nn.functional.pad(k, [0, 8 - head_size_q_og % 8])
    if head_size_v_og % 8 != 0:
        v = torch.nn.functional.pad(v, [0, 8 - head_size_v_og % 8])
    out_padded, _, _, _ = _flash_attn_varlen_forward(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        None,
        None,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_q,
        0.0,
        softmax_scale,
        causal=causal,
        logits_soft_cap=logits_soft_cap,
        window_size_left=int(window_size[0]),
        window_size_right=int(window_size[1]),
        sink_size=int(window_size[2]) if len(window_size) == 3 else 0,
        bias=None,
        alibi_slopes=None,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        return_lse=False,
        return_softmax=False,
        sink_ptr=sink_ptr,
    )
    out = out_padded[..., :head_size_v_og]
    return out
