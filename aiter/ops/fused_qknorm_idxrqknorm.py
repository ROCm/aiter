# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch
from torch import Tensor

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime

FUSED_QKNORM_IDXRQKNORM_SUPPORTS_PACKED_SHUFFLE = True
FUSED_QKNORM_IDXRQKNORM_SUPPORTS_FP8_INDEX_Q = True

_FP8_E4M3_DTYPES = tuple(
    dt
    for dt in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
    )
    if dt is not None
)


# Targets with opus::fp32_to_fp8 (opus.hpp).
_FP8_CVT_GFX = ("gfx942", "gfx950", "gfx1200", "gfx1201", "gfx1250")


def _is_fp8_e4m3_tensor(t: Tensor | None) -> bool:
    return t is not None and t.dtype in _FP8_E4M3_DTYPES


@compile_ops(
    "module_fused_qknorm_idxrqknorm",
    fc_name="fused_qknorm_idxrqknorm",
    develop=True,
)
def _fused_qknorm_idxrqknorm_hip(
    qkv: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    cos_sin_cache: Tensor,
    positions: Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: Tensor | None,
    index_k_norm_weight: Tensor | None,
    num_index_heads: int,
    slot_mapping: Tensor | None,
    kv_cache_k: Tensor | None,
    kv_cache_v: Tensor | None,
    index_cache: Tensor | None,
    block_size: int,
    q_out: Tensor | None,
    index_q_out: Tensor | None,
    index_slot_mapping: Tensor | None,
    kv_cache_dtype: str = "auto",
    index_cache_dtype: str = "auto",
    k_scale: Tensor | None = None,
    v_scale: Tensor | None = None,
    asm_layout: bool = False,
    skip_index_branch: bool = False,
) -> None:
    pass


# ---- NVFP4 raw front ---------------------------------------------------------------------------
# The MiniMax-M3 TP4 NVFP4 insert (16 q / 1 kv / 1 index heads, head_dim 128, rotary 64, bf16) is
# launched eagerly at prefill sizes, where the generic entry's marshalling of 16 tensors through
# aiter_tensor_t costs about 20 us of host time per call. This front checks the operands of that one
# call shape in Python and hands raw device pointers and the stream to a second entry of the same
# module, for every token count; the C++ launcher then picks the prefill variant from
# AITER_FUSED_QKNORM_NVFP4_PREFILL_MIN_TOKENS tokens (default 2) and the generic kernel below it.
# Any other call (and any call the front declines) takes the generic path unchanged.
_NV_PF_ROW = (16 + 2 * 1 + 1 + 1) * 128  # qkv row, elements
_NV_PF_PAGE = 9216  # bytes of one K (or V) side of a 128-token NVFP4 page
_nv_pf_mod = None
_nv_pf_gfx950 = None


def _nvfp4_raw_front(
    qkv,
    q_norm_weight,
    k_norm_weight,
    cos_sin_cache,
    positions,
    eps,
    index_q_norm_weight,
    index_k_norm_weight,
    slot_mapping,
    kv_cache_k,
    kv_cache_v,
    index_cache,
    q_out,
    index_q_out,
    index_slot_mapping,
    k_scale,
    v_scale,
):
    """True when the call was handed to the raw entry."""
    global _nv_pf_mod, _nv_pf_gfx950
    n = qkv.shape[0]
    if n < 1:
        return False
    bf16 = torch.bfloat16
    if (
        qkv.dtype is not bf16
        or qkv.dim() != 2
        or qkv.shape[1] != _NV_PF_ROW
        or not qkv.is_contiguous()
        or cos_sin_cache.dtype is not bf16
        or cos_sin_cache.dim() != 2
        or cos_sin_cache.shape[1] != 64
        or not cos_sin_cache.is_contiguous()
        or positions.dtype is not torch.int64
        or positions.numel() != n
        or not positions.is_contiguous()
        or q_out is None
        or q_out.dtype is not bf16
        or q_out.numel() != n * 2048
        or not q_out.is_contiguous()
        or index_q_out is None
        or index_q_out.dtype is not torch.float8_e4m3fn
        or index_q_out.numel() != n * 128
        or not index_q_out.is_contiguous()
    ):
        return False
    for w in (q_norm_weight, k_norm_weight, index_q_norm_weight, index_k_norm_weight):
        if (
            w is None
            or w.dtype is not bf16
            or w.numel() != 128
            or not w.is_contiguous()
        ):
            return False
    for sl in (slot_mapping, index_slot_mapping):
        if (
            sl is None
            or sl.dtype is not torch.int64
            or sl.numel() != n
            or not sl.is_contiguous()
        ):
            return False
    if (
        kv_cache_k is None
        or kv_cache_v is None
        or kv_cache_k.data_ptr() != kv_cache_v.data_ptr()
        or kv_cache_k.dtype is not torch.uint8
        or kv_cache_k.dim() != 2
        or kv_cache_k.shape[1] != 2 * _NV_PF_PAGE
        or kv_cache_k.stride(1) != 1
        or kv_cache_k.stride(0) < 2 * _NV_PF_PAGE
        or kv_cache_k.stride(0) % 16
        or kv_cache_k.data_ptr() % 16
        or index_cache is None
        or index_cache.dtype is not torch.uint8
        or not index_cache.is_contiguous()
        or index_cache.numel() % _NV_PF_PAGE
        or index_cache.numel() // _NV_PF_PAGE < kv_cache_k.shape[0]
        or k_scale is None
        or v_scale is None
        or k_scale.dtype is not torch.float32
        or v_scale.dtype is not torch.float32
        or k_scale.numel() != 1
        or v_scale.numel() != 1
    ):
        return False
    dev = qkv.get_device()
    if dev < 0 or dev != torch.cuda.current_device():
        return False
    for t in (
        kv_cache_k,
        index_cache,
        k_scale,
        v_scale,
        q_out,
        index_q_out,
        cos_sin_cache,
        positions,
        slot_mapping,
        index_slot_mapping,
        q_norm_weight,
        k_norm_weight,
        index_q_norm_weight,
        index_k_norm_weight,
    ):
        if t.get_device() != dev:
            return False
    if _nv_pf_gfx950 is None:
        _nv_pf_gfx950 = get_gfx_runtime() == "gfx950"
    if not _nv_pf_gfx950:
        return False
    if _nv_pf_mod is None:
        from ..jit import core as jit_core

        if (
            jit_core.AITER_REBUILD
            and "module_fused_qknorm_idxrqknorm" not in jit_core.rebuilded_list
        ):
            return False  # the generic call below does the requested rebuild first
        try:
            module = jit_core.get_module("module_fused_qknorm_idxrqknorm")
        except ModuleNotFoundError:
            return False  # not built yet: the generic call below builds it
        if not hasattr(module, "fused_qknorm_nvfp4_raw"):
            return False  # a build older than the raw entry: the generic call serves it
        _nv_pf_mod = module
    _nv_pf_mod.fused_qknorm_nvfp4_raw(
        qkv.data_ptr(),
        q_out.data_ptr(),
        index_q_out.data_ptr(),
        q_norm_weight.data_ptr(),
        k_norm_weight.data_ptr(),
        index_q_norm_weight.data_ptr(),
        index_k_norm_weight.data_ptr(),
        cos_sin_cache.data_ptr(),
        positions.data_ptr(),
        slot_mapping.data_ptr(),
        index_slot_mapping.data_ptr(),
        kv_cache_k.data_ptr(),
        index_cache.data_ptr(),
        k_scale.data_ptr(),
        v_scale.data_ptr(),
        n,
        kv_cache_k.stride(0),
        eps,
        torch.cuda.current_stream(dev).cuda_stream,
    )
    return True


def fused_qknorm_idxrqknorm(
    qkv: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    cos_sin_cache: Tensor,
    positions: Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    index_q_norm_weight: Tensor | None = None,
    index_k_norm_weight: Tensor | None = None,
    num_index_heads: int = 0,
    slot_mapping: Tensor | None = None,
    kv_cache_k: Tensor | None = None,
    kv_cache_v: Tensor | None = None,
    index_cache: Tensor | None = None,
    block_size: int = 0,
    q_out: Tensor | None = None,
    index_q_out: Tensor | None = None,
    index_slot_mapping: Tensor | None = None,
    kv_cache_dtype: str = "auto",
    index_cache_dtype: str | None = None,
    k_scale: Tensor | None = None,
    v_scale: Tensor | None = None,
    asm_layout: bool = False,
    skip_index_branch: bool = False,
) -> None:
    # The main K/V caches are always passed as separate kv_cache_k / kv_cache_v
    # tensors. asm_layout selects the in-cache addressing: page-16 SHUFFLE
    # (asm_layout=True) vs plain page-128 (asm_layout=False, where kv_cache_k /
    # kv_cache_v are typically the key/value slices of a fused
    # [num_blocks, 2, block_size, num_kv_heads, head_dim] cache).
    # MXFP4 pages are uint8, 8704 B per 128 tokens: [128 x 64 B e2m1][128 x 4 B e8m0].
    # kv_cache_dtype="mxfp4": sparse insert only (pass skip_index_branch=True for a layer that
    #   reuses another layer's top-k); K/V are [num_pages, num_kv_heads, 8704], block_size 128,
    #   no k_scale/v_scale.
    # index_cache_dtype="mxfp4": index_cache is whole contiguous pages; only the index-K write
    #   changes.
    # kv_cache_dtype="nvfp4" + index_cache_dtype="nvfp4" (together; an index-less layer, skip_index_branch,
    #   passes index_cache_dtype "auto"): K, V and index-K are quantized to NVFP4 (e2m1 + one e4m3 scale per
    #   16 elements) straight into the pages, in the same pass as norm/RoPE. kv_cache_k and kv_cache_v are the
    #   same uint8 [num_pages, 2 * num_kv_heads * 9216] pages ([K data | K scale | V data | V scale]; K scale
    #   in MFMA lane order, V data in 32-bit words (8 tokens x 4 bit), V scale [group][token]), block_size 128,
    #   16 B aligned page stride and base; V is written by atomic and/or on 32-bit words, so no two tokens of one call
    #   may share a slot;
    #   index_cache is whole contiguous 9216 B pages ([128 x 64 B e2m1][128 x 8 B e4m3]); k_scale / v_scale are
    #   the fp32 [1] global scales of K and V (the index key has none; they must be positive). The
    #   rounding is that of the torch reference in op_tests/test_fused_qknorm_nvfp4.py, on gfx950's fp8 / fp4
    #   converts; nvfp4 (like mxfp4) is refused off gfx950.
    # q_out may be an e4m3 tensor (fp8 or uint8, insert mode): unit-scale e4m3 cast, saturating
    #   (+-inf too; NaN stays NaN), written instead of the bf16 q (as index_q_out).
    if index_cache_dtype is None:
        index_cache_dtype = (
            "fp8"
            if _is_fp8_e4m3_tensor(index_cache) or _is_fp8_e4m3_tensor(index_q_out)
            else "auto"
        )

    if (
        kv_cache_dtype == "nvfp4"
        and index_cache_dtype == "nvfp4"
        and not skip_index_branch
        and not asm_layout
        and num_heads == 16
        and num_kv_heads == 1
        and num_index_heads == 1
        and rotary_dim == 64
        and block_size == 128
        # data_ptr() / get_device() in _nvfp4_raw_front cannot be traced: a torch.compile(fullgraph=True)
        # call must use the existing guarded HIP entry below (it has a fake/meta implementation for
        # tracing), the same restriction aiter/ops/triton/normalization/rmsnorm.py:83-85 documents.
        and not torch.compiler.is_compiling()
        and _nvfp4_raw_front(
            qkv,
            q_norm_weight,
            k_norm_weight,
            cos_sin_cache,
            positions,
            eps,
            index_q_norm_weight,
            index_k_norm_weight,
            slot_mapping,
            kv_cache_k,
            kv_cache_v,
            index_cache,
            q_out,
            index_q_out,
            index_slot_mapping,
            k_scale,
            v_scale,
        )
    ):
        return

    # opus has no fp4 conversion off gfx950 and no fp8 conversion on gfx1100/gfx115x: the
    # stores would silently write zeros.
    fp4_dtypes = {"mxfp4", "nvfp4"} & {kv_cache_dtype, index_cache_dtype}
    if fp4_dtypes and get_gfx_runtime() != "gfx950":
        raise RuntimeError("fused_qknorm_idxrqknorm mxfp4/nvfp4 caches require gfx950")
    if (
        q_out is not None
        and q_out.dtype != qkv.dtype
        and get_gfx_runtime() not in _FP8_CVT_GFX
    ):
        raise RuntimeError(f"an fp8 q_out requires one of {_FP8_CVT_GFX}")

    use_fp8_kv_cache = (
        kv_cache_k is not None
        and isinstance(kv_cache_dtype, str)
        and kv_cache_dtype.startswith("fp8")
    )
    use_per_token_kv_scale = use_fp8_kv_cache and kv_cache_dtype in (
        "fp8",
        "fp8_e4m3",
    )
    use_static_kv_scale = use_fp8_kv_cache and kv_cache_dtype == "fp8_e4m3_static"
    if use_fp8_kv_cache:
        if not skip_index_branch and index_slot_mapping is None:
            index_slot_mapping = slot_mapping
        assert slot_mapping is not None
        assert kv_cache_v is not None
        if not skip_index_branch:
            assert index_q_norm_weight is not None
            assert index_k_norm_weight is not None
            assert index_cache is not None
            assert index_slot_mapping is not None
        if use_per_token_kv_scale or use_static_kv_scale:
            assert k_scale is not None
            assert v_scale is not None
        else:
            assert k_scale is None
            assert v_scale is None

    _fused_qknorm_idxrqknorm_hip(
        qkv,
        q_norm_weight,
        k_norm_weight,
        cos_sin_cache,
        positions,
        num_heads,
        num_kv_heads,
        rotary_dim,
        eps,
        index_q_norm_weight,
        index_k_norm_weight,
        num_index_heads,
        slot_mapping,
        kv_cache_k,
        kv_cache_v,
        index_cache,
        block_size,
        q_out,
        index_q_out,
        index_slot_mapping,
        kv_cache_dtype,
        index_cache_dtype,
        k_scale,
        v_scale,
        asm_layout,
        skip_index_branch,
    )
