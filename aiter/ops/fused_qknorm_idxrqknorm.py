# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import os

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


# NVFP4 insert (gfx950, M3 TP4 head set): the FlyDSL kernel instead of the HIP one, same bytes.
# AITER_FUSED_QKNORM_NVFP4_FLYDSL=0 keeps the HIP kernel.
_NVFP4_INSERT_FLYDSL = os.environ.get("AITER_FUSED_QKNORM_NVFP4_FLYDSL", "1") != "0"
# Prefill-sized calls only: the FlyDSL kernel merges a V octet by read-modify-write of one run of
# consecutive slots (the HIP kernel uses atomics), so it is kept off the small decode-like calls
# (scattered slots, several tokens of different requests in one octet) the HIP kernel documents.
_NVFP4_INSERT_FLYDSL_MIN_TOKENS = 1024

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


def _take_deferred_front(qkv: Tensor):
    from .gemm_a8w8_qkv_front import take_deferred

    return take_deferred(qkv)


def _run_deferred_front(
    deferred,
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
) -> None:
    """Dispatch a parked fused QKV front. The chain kernel is the TP4 MiniMax-M3.1 head set with NVFP4 K / V / index-K
    pages; anything else cannot consume the parked activation (the QKV was never computed), so it raises.
    """
    from .gemm_a8w8_qkv_front import run_deferred

    if not (
        (num_heads, num_kv_heads, num_index_heads, rotary_dim) == (16, 1, 1, 64)
        and kv_cache_dtype == "nvfp4"
        and index_cache_dtype == "nvfp4"
        and not skip_index_branch
        and not asm_layout
        and block_size == 128
        and q_out is not None
        and index_q_out is not None
        and kv_cache_k is not None
        and kv_cache_v is not None
        and kv_cache_k.data_ptr() == kv_cache_v.data_ptr()
        and index_cache is not None
        and slot_mapping is not None
    ):
        raise RuntimeError(
            "fused_qknorm_idxrqknorm: a deferred QKV front (gemm_a8w8_qkv_front) serves only the 16/1/1-head, rot 64, "
            "NVFP4 K / V / index-K, block 128, non-asm-layout insert with q_out and index_q_out"
        )
    if index_slot_mapping is None:
        index_slot_mapping = slot_mapping
    # ``[pages, 2 * kv heads * 9216]`` uint8 pages (K side then V side) as the kernel's [pages, 2, 128, 72] views
    pages = kv_cache_k
    kv_pages = torch.as_strided(
        pages,
        (pages.shape[0], 2 * num_kv_heads, 128, 72),
        (pages.stride(0), 128 * 72, 72, 1),
    )
    run_deferred(
        deferred,
        q_norm_weight,
        k_norm_weight,
        index_q_norm_weight,
        index_k_norm_weight,
        cos_sin_cache,
        positions,
        slot_mapping,
        index_slot_mapping,
        kv_pages,
        index_cache.view(-1, 128, 72),
        k_scale,
        v_scale,
        q_out,
        index_q_out,
        num_heads,
        num_kv_heads,
        num_index_heads,
        eps,
    )


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
    #   4 B aligned page stride; V is written by atomic and/or on 32-bit words, so no two tokens of one call
    #   may share a slot;
    #   index_cache is whole contiguous 9216 B pages ([128 x 64 B e2m1][128 x 8 B e4m3]); k_scale / v_scale are
    #   the fp32 [1] global scales of K and V (the index key has none). The rounding is integer/compare
    #   arithmetic, not a hardware convert, but nvfp4 (like mxfp4) is refused off gfx950.
    # q_out may be an e4m3 tensor (fp8 or uint8, insert mode): unit-scale e4m3 cast, saturating
    #   (+-inf too; NaN stays NaN), written instead of the bf16 q (as index_q_out).
    if index_cache_dtype is None:
        index_cache_dtype = (
            "fp8"
            if _is_fp8_e4m3_tensor(index_cache) or _is_fp8_e4m3_tensor(index_q_out)
            else "auto"
        )

    # A QKV projection parked by gemm_a8w8_qkv_front (its tuned tier is a fused one): the GEMM runs here, in the
    # same launch as the insert (17-64 rows) or ahead of the reduce-insert (65-128 rows). ``qkv`` is uncomputed.
    deferred = _take_deferred_front(qkv)
    if deferred is not None:
        _run_deferred_front(
            deferred,
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

    if (
        _NVFP4_INSERT_FLYDSL
        and kv_cache_dtype == "nvfp4"
        and index_cache_dtype == "nvfp4"
        and not skip_index_branch
        and not asm_layout
        and qkv.shape[0] >= _NVFP4_INSERT_FLYDSL_MIN_TOKENS
    ):
        if index_slot_mapping is None:
            index_slot_mapping = slot_mapping
        from .flydsl.kernels.fused_qknorm_idxr_nvfp4_insert import (
            fused_qknorm_idxr_nvfp4_insert,
            nvfp4_insert_blocker,
        )

        if (
            nvfp4_insert_blocker(
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
                k_scale,
                v_scale,
            )
            is None
        ):
            fused_qknorm_idxr_nvfp4_insert(
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
                index_cache,
                q_out,
                index_q_out,
                index_slot_mapping,
                k_scale,
                v_scale,
            )
            return

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
