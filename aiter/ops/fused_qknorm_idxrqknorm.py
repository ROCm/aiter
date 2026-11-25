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
    index_scale: Tensor | None = None,
) -> None:
    pass


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
    index_scale: Tensor | None = None,
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
    # kv_cache_dtype="nvfp4" with byte k_scale / v_scale (uint8 or float8_e4m3fn) instead of
    #   the fp32 globals: the NVFP4 row layout of the MiniMax-M3 sparse pool and of the FlyDSL
    #   paged decode's NVFP4 K/V (gfx950 and gfx1250). kv_cache_k / kv_cache_v are the e2m1
    #   payloads [num_pages, num_kv_heads, 128, head_dim // 2] (element 2j in the low nibble of
    #   byte j) and k_scale / v_scale their E4M3 scale planes [num_pages, num_kv_heads, 128,
    #   head_dim // 16], one per 16 elements, no global scale; rows contiguous inside a
    #   (page, head), any base, page and head strides free (the two scale planes share theirs)
    #   as long as no two (page, head)s of a tensor overlap. K (normed, roped) and V are
    #   rounded to the model dtype, then scale = e4m3(amax / 6) (round to nearest even,
    #   saturating at 448) and codes = e2m1(x * (1 / scale)) (half to even, saturating at 6;
    #   the sign is the product's, so +0 in a zero-scale block), IEEE arithmetic: byte for
    #   byte what ATOM's Triton kv_nvfp4_store makes of the rows "auto" writes. Sparse insert
    #   only, as mxfp4; block_size 128; the index cache is "auto" or fp8 as in the FP8 path
    #   (addressed by index_slot_mapping alone), or the NVFP4 index rows below (index_scale),
    #   or absent with skip_index_branch.
    # index_cache_dtype="nvfp4" with index_scale (with the NVFP4 row-layout K/V above): the
    #   index key in the same row layout, quantized in the same pass. index_cache is the e2m1
    #   payload [num_pages, 128, 64] uint8 and index_scale its E4M3 plane [num_pages, 128, 8]
    #   (uint8 or float8_e4m3fn), rows contiguous, page strides free as for the K/V planes;
    #   addressed by index_slot_mapping (slot // 128, slot % 128). The normed, roped key is
    #   rounded to the model dtype, then quantized as the K/V rows (ATOM's index_fp4_store).
    # q_out may be an e4m3 tensor (fp8 or uint8, insert mode): unit-scale e4m3 cast, saturating
    #   (+-inf too; NaN stays NaN), written instead of the bf16 q (as index_q_out).
    if index_cache_dtype is None:
        index_cache_dtype = (
            "fp8"
            if _is_fp8_e4m3_tensor(index_cache) or _is_fp8_e4m3_tensor(index_q_out)
            else "auto"
        )

    # opus has no fp4 conversion off gfx950 and no fp8 conversion on gfx1100/gfx115x: the
    # stores would silently write zeros. The NVFP4 row layout (byte scale planes) is integer
    # and compare arithmetic only, so it also runs on gfx1250; which index caches go with it
    # is the C++ checks' to say, before any launch.
    fp4_dtypes = {"mxfp4", "nvfp4"} & {kv_cache_dtype, index_cache_dtype}
    nvfp4_rows = (
        kv_cache_dtype == "nvfp4"
        and k_scale is not None
        and (k_scale.dtype == torch.uint8 or _is_fp8_e4m3_tensor(k_scale))
    )
    if nvfp4_rows:
        if get_gfx_runtime() not in ("gfx950", "gfx1250"):
            raise RuntimeError(
                "fused_qknorm_idxrqknorm NVFP4 row-layout K/V requires gfx950 or gfx1250"
            )
    elif fp4_dtypes and get_gfx_runtime() != "gfx950":
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
        index_scale,
    )
