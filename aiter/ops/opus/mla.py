# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Raw ctypes bindings of the opus MLA kernels.

Each op is the C entry point of the same name in csrc/kernels/mla/opus_mla_{decode,prefill}_fwd.cu.
ctypes binds positionally, so the argument order here must match the C signature one for one.
"""

import torch

from ...jit.core import compile_ops

DECODE_MD_NAME = "module_mla_decode_opus"
PREFILL_MD_NAME = "module_mla_prefill_opus"


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------


@compile_ops(DECODE_MD_NAME, ffi_type="ctypes")
def opus_mla_decode_mxfp8_fwd(
    q_nope: torch.Tensor,  # [total_q, H, D_NOPE]      fp8
    q_scale: torch.Tensor,  # [total_q, H, D_SCALE]     uint8 (E8M0)
    q_rope: torch.Tensor,  # [total_q, H, D_ROPE]      bf16
    kv_nope: torch.Tensor,  # [total_tokens, D_NOPE]    fp8
    kv_scale: torch.Tensor,  # [total_tokens, D_SCALE]   uint8 (E8M0)
    kv_rope: torch.Tensor,  # [total_tokens, D_ROPE]    bf16
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info_set: torch.Tensor,
    page_size: int,
    softmax_scale: float,
    logits: torch.Tensor,  # aiter split_output [num_partials,1,H,D_NOPE] fp32
    attn_lse: torch.Tensor,  # aiter split_lse    [num_partials,1,H,1]      fp32
    out: torch.Tensor,  # final [total_q, H, D_NOPE] bf16
    final_lse: torch.Tensor | None = None,
) -> None: ...


@compile_ops(DECODE_MD_NAME, ffi_type="ctypes")
def opus_mla_decode_fwd(
    q: torch.Tensor,  # [total_q, H, 576] bf16 or fp8 (merged nope+rope)
    kv: torch.Tensor,  # [total_tokens, ..., 576] same dtype as q, page_size == 1
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info_set: torch.Tensor,
    page_size: int,
    softmax_scale: float,
    logits: torch.Tensor,  # aiter split_output [num_partials,1,H,512] fp32
    attn_lse: torch.Tensor,  # aiter split_lse    [num_partials,1,H,1]   fp32
    out: torch.Tensor,  # final [total_q, H, 512] bf16
    final_lse: torch.Tensor | None = None,  # [total_q, H] fp32
    q_scale: torch.Tensor | None = None,  # fp8: float[1] per-tensor descale
    kv_scale: torch.Tensor | None = None,  # fp8: float[1] per-tensor descale
    max_seqlen_q: int = -1,  # fp8: required, picks the kernel; bf16: unused
    causal: bool = True,  # fp8: mask across the max_seqlen_q query tokens
    g_kv_indptr: torch.Tensor | None = None,  # fp8 round-robin CP: GLOBAL kv_indptr
    cp_world_size: int = 1,  # fp8 round-robin CP: number of ranks, 1 = no CP
    cp_rank: int = 0,  # fp8 round-robin CP: this rank
) -> None: ...


@compile_ops(DECODE_MD_NAME, ffi_type="ctypes")
def opus_mla_decode_supported(fp8: bool, nhead: int, max_seqlen_q: int) -> int:
    """1 if :func:`opus_mla_decode_fwd` has a build for ``nhead`` x ``max_seqlen_q``."""


# ---------------------------------------------------------------------------
# Sparse prefill
# ---------------------------------------------------------------------------


@compile_ops(PREFILL_MD_NAME, ffi_type="ctypes")
def opus_mla_v4_prefill_a16w16_gfx950_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,  # [H] fp32
    out: torch.Tensor,
    softmax_scale: float,
    num_splits: int = 1,
    partial_o: torch.Tensor | None = None,  # [N, num_splits, H, 512] fp32
    partial_max: torch.Tensor | None = None,  # [N, num_splits, H] fp32
    partial_sum: torch.Tensor | None = None,  # [N, num_splits, H] fp32
) -> None: ...


@compile_ops(PREFILL_MD_NAME, ffi_type="ctypes")
def opus_mla_v4_prefill_a16w16_gfx1250_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,  # [H] fp32
    out: torch.Tensor,
    softmax_scale: float,
    num_splits: int = 1,
    partial_o: torch.Tensor | None = None,  # [N, num_splits, H, 512] fp32
    partial_max: torch.Tensor | None = None,  # [N, num_splits, H] fp32
    partial_sum: torch.Tensor | None = None,  # [N, num_splits, H] fp32
) -> None: ...


@compile_ops(PREFILL_MD_NAME, ffi_type="ctypes")
def opus_mla_v4_prefill_a8w8_gfx950_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,  # [H] fp32
    out: torch.Tensor,
    softmax_scale: float,
    num_splits: int = 1,
    partial_o: torch.Tensor | None = None,  # [N, num_splits, H, 512] fp32
    partial_max: torch.Tensor | None = None,  # [N, num_splits, H] fp32
    partial_sum: torch.Tensor | None = None,  # [N, num_splits, H] fp32
) -> None: ...


@compile_ops(PREFILL_MD_NAME, ffi_type="ctypes")
def opus_mla_v4_prefill_a8w8_gfx1250_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,  # [H] fp32
    out: torch.Tensor,
    softmax_scale: float,
    num_splits: int = 1,
    partial_o: torch.Tensor | None = None,  # [N, num_splits, H, 512] fp32
    partial_max: torch.Tensor | None = None,  # [N, num_splits, H] fp32
    partial_sum: torch.Tensor | None = None,  # [N, num_splits, H] fp32
) -> None: ...


@compile_ops(PREFILL_MD_NAME, ffi_type="ctypes")
def opus_mla_v41_prefill_a8w8_gfx950_fwd(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    unified_kv: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_scale: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,  # [H] fp32
    out: torch.Tensor,
    softmax_scale: float,
    num_splits: int = 1,
    partial_o: torch.Tensor | None = None,  # [N, num_splits, H, 512] fp32
    partial_max: torch.Tensor | None = None,  # [N, num_splits, H] fp32
    partial_sum: torch.Tensor | None = None,  # [N, num_splits, H] fp32
) -> None: ...
