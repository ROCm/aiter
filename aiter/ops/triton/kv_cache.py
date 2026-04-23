# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
from aiter.ops.triton._triton_kernels.kv_cache import _cat_and_cache_mla_kernel
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils.types import e4m3_dtype

_LOGGER = AiterTritonLogger()


def cat_and_cache_mla_fake_tensor(
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    apply_scale: bool = True,
    shuffled_kv_cache: bool = False,
) -> None:
    return None


@torch_compile_guard(gen_fake=cat_and_cache_mla_fake_tensor)
def cat_and_cache_mla(
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_scale: torch.Tensor,
    apply_scale: bool = True,
    shuffled_kv_cache: bool = False,
) -> None:
    """
    Perform concat k_nope and k_pe to kv_cache inplace

    Key parameters:
    - k_nope: Matrix X with shape (B_slot, KH, D1).
    - k_pe: Matrix W with shape (B_slot, KH, D2).
    - kv_cache: Matrix W with shape (B_cache, KH, D1 + D2).
    - slot_mapping: Matrix W with shape (B_slot, ).

    B is the number of decode tokens, B_slot is the number of prefill + decode tokens, B_cahce is the max number of tokens of kv_cache
    QH must be multiple of KH

    Returns:
    - kv_cache: The output matrix with shape (B_max, KH, D1 + D2) (inplace).
    """
    _LOGGER.info(
        f"CAT_AND_CACHE_MLA: k_nope={tuple(k_nope.shape)} k_pe={tuple(k_pe.shape)} "
        + f"kv_cache={tuple(kv_cache.shape)} slot_mapping={tuple(slot_mapping.shape)}"
    )

    b, kh, d_nope = k_nope.shape
    bk, kh2, d_rope = k_pe.shape
    kv_cache_dtype = kv_cache.dtype
    assert kv_cache_dtype in [
        torch.bfloat16,
        e4m3_dtype,
        torch.uint8,
    ], "KV cache dtype must be BF16, FP8 or packed FP4"

    block_size = 1
    SCALE_K_WIDTH_NOPE = 4
    SCALE_K_WIDTH_ROPE = 4
    if kv_cache_dtype == torch.uint8:
        assert (
            shuffled_kv_cache == True
        ), "shuffle_kv_cache must be True for FP4 KV cache"
        b_cache, h_cache, block_size, d_cache = kv_cache.shape
        SCALE_K_LORA = d_nope // 16
        SCALE_K_ROPE = d_rope // 16
        SCALE_K_WIDTH_NOPE = (
            min(16, triton.next_power_of_2(SCALE_K_LORA))
            if SCALE_K_LORA >= 4
            else SCALE_K_LORA
        )
        SCALE_K_WIDTH_ROPE = (
            min(16, triton.next_power_of_2(SCALE_K_ROPE))
            if SCALE_K_ROPE >= 4
            else SCALE_K_ROPE
        )
    else:
        if shuffled_kv_cache:
            b_cache, h_cache, block_size, d_cache = kv_cache.shape
        else:
            b_cache, h_cache, d_cache = kv_cache.shape
    (b_slot,) = slot_mapping.shape

    assert (
        b == bk and b_slot == b_slot
    ), "K batch dimensions and slot_mapping should be identical (bk == bk == b_slot)"
    assert kh == kh2 == h_cache, "K head should be identical"
    if kv_cache.dtype == torch.uint8:
        assert (
            (d_nope + d_rope) // 2 + (d_nope + d_rope) // 16
        ) == d_cache, "The D dimension of kv_cache should be (d_nope + d_rope) // 2 + (d_nope + d_rope) // 16 for FP4 KV cache"
    else:
        assert (
            d_nope + d_rope == d_cache
        ), "D dimension of k_nope and k_pe should be summed up to be the D dimension of kv_cache"
    if isinstance(k_scale, torch.Tensor):
        assert k_scale.numel() == 1, "k_scale should be a single-element torch.Tensor"

    if shuffled_kv_cache:
        kv_cache_stride_b = kv_cache.stride(0)
        kv_cache_stride_h = kv_cache.stride(1)
        kv_cache_stride_d = kv_cache.stride(3)
    else:
        kv_cache_stride_b = kv_cache.stride(0)
        kv_cache_stride_h = kv_cache.stride(1)
        kv_cache_stride_d = kv_cache.stride(2)

    assert (
        kv_cache_stride_d == 1
    ), "The stride of the last dimension of KV cache must be 1"

    _cat_and_cache_mla_kernel[(b * kh,)](
        k_nope,
        k_pe,
        kv_cache,
        slot_mapping,
        *k_nope.stride(),
        *k_pe.stride(),
        kv_cache_stride_b,
        kv_cache_stride_h,
        kv_cache_stride_d,
        k_scale_ptr=k_scale,
        KH=kh,
        BLOCK_D_nope=d_nope,
        BLOCK_D_pe=d_rope,
        BLOCK_SIZE=block_size,
        SHUFFLED_KV_CACHE=shuffled_kv_cache,
        SCALE_K_WIDTH_NOPE=SCALE_K_WIDTH_NOPE,
        SCALE_K_WIDTH_ROPE=SCALE_K_WIDTH_ROPE,
        HAVE_K_SCALE=(k_scale is not None and apply_scale),
        num_warps=1,
    )
