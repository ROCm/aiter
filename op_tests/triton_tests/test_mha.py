# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import logging
import numpy as np
import math
from aiter.ops.triton.mha import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_with_kvcache,
    mha_set_use_fused_bwd_kernel,
    mha_set_use_int64_strides,
)
from aiter.ops.triton.mha_v3 import (
    flash_attn_func as flash_attn_func_v3,
    flash_attn_varlen_func as flash_attn_varlen_func_v3,
    flash_attn_with_kvcache as flash_attn_with_kvcache_v3
)
from aiter.ops.triton.utils.mha_kernel_utils import _quantize_bshd, _quantize_thd
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from aiter.test_mha_common import (
    attention_ref,
    generate_random_padding_mask,
    generate_qkv,
)
from typing import Optional

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DEBUG_MODE = False
ATOL_fp8 = 2.5e-1
RTOL_fp8 = 2.5e-1


def pad_rearrange_dropout_mask(
    S_dmask,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqlen_q,
    seqlen_k,
    num_q_heads,
):
    batch_size = cu_seqlens_q.numel() - 1

    padded_dropout_mask = torch.ones(
        (batch_size, num_q_heads, seqlen_q, seqlen_k), device="cuda"
    )
    for b in range(batch_size):
        start_q = cu_seqlens_q[b].item()
        end_q = cu_seqlens_q[b + 1].item()
        start_k = cu_seqlens_k[b].item()
        end_k = cu_seqlens_k[b + 1].item()

        seqlen_q = end_q - start_q
        seqlen_k = end_k - start_k
        for h in range(S_dmask.shape[1]):
            padded_dropout_mask[b, h, :max_seqlen_q, :max_seqlen_k] = S_dmask[
                b, h, :, :
            ]

    return padded_dropout_mask


def fp8_assert_close(
    tensor_a, tensor_b, atol=ATOL_fp8, rtol=RTOL_fp8, max_diff_percentage=0.5
):
    """Assert tensors are close with tolerance for a small percentage of elements.
    Behavior:
      - Each input is inspected independently; if its dtype is not float32 it is cast
        to float32 for comparison stability (covers fp8, fp16, bf16, etc.).
      - A warning is emitted for every tensor that is cast, indicating original dtype.
      - Comparison then proceeds in float32 (or original float32) using mixed absolute & relative criteria
        allowing up to `max_diff_percentage` percent of elements to exceed both tolerances.
    """
    # Cast each tensor individually if not already float32; emit warning for transparency.
    if tensor_a.dtype != torch.float32:
        logger.warning(
            f"fp8_assert_close: casting tensor_a from {tensor_a.dtype} to float32 for comparison"
        )
        a_comp = tensor_a.float()
    else:
        a_comp = tensor_a

    if tensor_b.dtype != torch.float32:
        logger.warning(
            f"fp8_assert_close: casting tensor_b from {tensor_b.dtype} to float32 for comparison"
        )
        b_comp = tensor_b.float()
    else:
        b_comp = tensor_b
    abs_diff = (a_comp - b_comp).abs()
    rel_diff = abs_diff / b_comp.abs().clamp(min=1e-6)

    # calculate elements that exceed tolerance
    abs_check = abs_diff > atol
    rel_check = rel_diff > rtol
    failed_check = torch.logical_and(abs_check, rel_check)

    # calculate percentage of failed elements
    failed_percentage = failed_check.sum().item() / failed_check.numel() * 100

    # if percentage is small enough, test passes
    if failed_percentage <= max_diff_percentage:
        return True

    # Otherwise, provide diagnostic information
    max_abs_idx = torch.argmax(abs_diff).item()
    max_rel_idx = torch.argmax(rel_diff).item()

    flat_to_idx = lambda flat_idx, shape: np.unravel_index(flat_idx, shape)

    max_abs_pos = flat_to_idx(max_abs_idx, tensor_a.shape)
    max_rel_pos = flat_to_idx(max_rel_idx, tensor_a.shape)

    max_abs_diff = abs_diff.flatten()[max_abs_idx].item()
    max_rel_diff = rel_diff.flatten()[max_rel_idx].item()

    raise AssertionError(
        f"Tensors not close enough! {failed_percentage:.6f}% elements exceed tolerance.\n"
        f"Greatest absolute difference: {max_abs_diff} at index {max_abs_pos} (up to {atol} allowed)\n"
        f"Greatest relative difference: {max_rel_diff} at index {max_rel_pos} (up to {rtol} allowed)"
    )


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("BATCH", [1, 4, 57, 128])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (4, 4), (128, 128), (2, 1), (1, 2), (32, 16), (64, 128)],
)
@pytest.mark.parametrize(
    "NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (16, 16), (2, 1), (48, 8)]
)
@pytest.mark.parametrize("HEAD_SZ", [8, 32, 128])
@pytest.mark.parametrize(
    "DROPOUT, RETURN_LSE, RETURN_SOFTMAX, ", [(0.2, True, True), (0.0, False, False)]
)
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
@pytest.mark.parametrize("FP8", [(True), (False)])
def test_mha(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    DROPOUT: float,
    RETURN_LSE: bool,
    RETURN_SOFTMAX: bool,
    CAUSAL: bool,
    FP8: bool,
    version: str,
    dtype=torch.float16,
):
    torch.cuda.empty_cache()
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)

    dropout_mask = None
    if version == "v3":
        if FP8:
            if DROPOUT > 0.0 or RETURN_LSE or RETURN_SOFTMAX:
                pytest.skip(
                    "flash_attn_func_v3 FP8 path doesn't support dropout/lse/attn_probs"
                )
            fp8_dtype = get_fp8_e4m3_dtype()
            group_size = (
                NUM_Q_HEADS // NUM_K_HEADS if NUM_Q_HEADS % NUM_K_HEADS == 0 else None
            )
            k_fp8, k_descale = _quantize_bshd(k, fp8_dtype)
            v_fp8, v_descale = _quantize_bshd(v, fp8_dtype)
            q_fp8, q_descale = _quantize_bshd(q, fp8_dtype, group_size=group_size)
            triton_out = flash_attn_func_v3(
                q_fp8,
                k_fp8,
                v_fp8,
                softmax_scale=None,
                causal=CAUSAL,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )
        else:
            # Non-FP8 v3 path (no dropout/lse/attn_probs support parity check)
            if DROPOUT > 0.0 or RETURN_LSE or RETURN_SOFTMAX:
                pytest.skip(
                    "flash_attn_func_v3 (non-FP8) doesn't expose dropout/lse/attn_probs here"
                )
            triton_out = flash_attn_func_v3(
                q,
                k,
                v,
                softmax_scale=None,
                causal=CAUSAL,
            )
    else:  # v2
        if FP8:
            pytest.skip("FP8 supported only on version 'v3'")
        triton_out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=DROPOUT,
            causal=CAUSAL,
            return_lse=RETURN_LSE,
            return_attn_probs=RETURN_SOFTMAX,
        )

    if version == "v2" and RETURN_LSE:
        assert len(triton_out) > 1
        lse = triton_out[1]
        if DEBUG_MODE:
            print(f"lse.shape={lse.shape}, lse={lse}")

    if version == "v2" and DROPOUT > 0.0 and RETURN_SOFTMAX:
        if RETURN_LSE:
            assert len(triton_out) == 3
            sd_mask = triton_out[2]
        else:
            assert len(triton_out) == 2
            sd_mask = triton_out[1]
        dropout_mask = sd_mask >= 0
        if DEBUG_MODE:
            print(f"sd_mask.shape={sd_mask.shape}, sd_mask={sd_mask}")
            print(
                f"dropout_mask.shape={dropout_mask.shape}, dropout_mask={dropout_mask}"
            )

    if version == "v2" and (RETURN_SOFTMAX or RETURN_LSE):
        triton_out = triton_out[0]
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

    torch_out = attention_ref(
        q, k, v, dropout_p=DROPOUT, dropout_mask=dropout_mask, causal=CAUSAL
    )
    torch_out, attention_scores, _ = torch_out
    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")
        print(
            f"attention_scores.shape={attention_scores.shape}, attention_scores={attention_scores}"
        )

    if FP8:
        fp8_assert_close(triton_out, torch_out, atol=ATOL_fp8, rtol=RTOL_fp8)
    else:
        torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)


# LLaMA 3 405B config
@pytest.mark.parametrize("BATCH", [1])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(128, 8)])
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("CAUSAL", [True])
@pytest.mark.parametrize("DROPOUT", [0.0])
def test_mha_int64_strides(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    DROPOUT: float,
    dtype=torch.float16,
    device="cuda",
    test_backward=True,
):
    """
    In the absence of strides being int64, parts of the offset computation is done in 32 bit and overflows resulting in segfaults.
    """
    torch.cuda.empty_cache()
    torch.manual_seed(20)
    # use int64 strides.
    mha_set_use_int64_strides(
        True
    )  # NOTE: if you set this to false this test case will segfault

    # generate inputs with large strides
    def _generate_input(
        batch: int, seqlen: int, nheads: int, dim_size: int, large_stride: bool = False
    ) -> torch.Tensor:
        seqlens = torch.full((batch,), seqlen)
        cu_seqlens = torch.cat(
            [
                torch.tensor([0], dtype=torch.int32),
                seqlens.cumsum(dim=0, dtype=torch.int32),
            ]
        ).to(device="cuda")
        total_seqlen = cu_seqlens[-1].item()

        if large_stride:
            x_dummy = torch.randn(
                (total_seqlen, nheads, 1024 * 1024 * 64), dtype=dtype, device="cuda"
            ).requires_grad_(True)
            x = x_dummy[:seqlen, :nheads, :dim_size]
        else:
            x = torch.randn(
                (total_seqlen, nheads, dim_size), dtype=dtype, device="cuda"
            ).requires_grad_(True)
        return x, cu_seqlens, seqlen

    # inputs
    q, cu_seqlens_q, max_seqlens_q = _generate_input(
        BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ, large_stride=True
    )
    k, cu_seqlens_k, max_seqlens_k = _generate_input(
        BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ
    )
    v, _, _ = _generate_input(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ)
    do = torch.randn_like(q)

    if DEBUG_MODE:
        print()
        print("q:", q.shape, q.stride())
        print("k:", k.shape, k.stride())
        print("v:", v.shape, v.stride())
        print("cu_seqlens_q:", cu_seqlens_q.shape, cu_seqlens_q.stride())
        print("cu_seqlens_k:", cu_seqlens_k.shape, cu_seqlens_k.stride())

    triton_out, _ = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlens_q,
        max_seqlens_k,
        dropout_p=DROPOUT,
        causal=CAUSAL,
        return_lse=True,
    )
    if test_backward:
        triton_dq, triton_dk, triton_dv = torch.autograd.grad(
            triton_out, (q, k, v), do.clone()
        )

    # NOTE: use fwd output to wait not exit program before kernel finishes
    print("triton_out:", triton_out)
    if test_backward:
        print("triton_dq:", triton_dq.shape, triton_dq.stride())
        print("triton_dk:", triton_dk.shape, triton_dk.stride())
        print("triton_dv:", triton_dv.shape, triton_dv.stride())


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("BATCH", [1, 4, 57, 128])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (4, 4), (128, 128), (2, 1), (1, 2), (32, 16), (64, 128)],
)
@pytest.mark.parametrize(
    "DROPOUT, RETURN_LSE, RETURN_SOFTMAX, ", [(0.0, False, False), (0.2, True, True)]
)
@pytest.mark.parametrize(
    "NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (16, 16), (2, 1), (48, 8)]
)
@pytest.mark.parametrize("HEAD_SZ", [8, 32, 128])
@pytest.mark.parametrize("CAUSAL", [(True), (False)])
@pytest.mark.parametrize("FP8", [(False), (True)])
def test_mha_varlen(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    DROPOUT: float,
    RETURN_LSE: bool,
    RETURN_SOFTMAX: bool,
    CAUSAL: bool,
    FP8: bool,
    version: str,
    dtype=torch.float16,
):
    torch.set_printoptions(threshold=10000)
    torch.cuda.empty_cache()
    torch.manual_seed(20)
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    query_padding_mask = generate_random_padding_mask(
        SEQLEN_Q, BATCH, "cuda", mode="random"
    )
    key_padding_mask = generate_random_padding_mask(
        SEQLEN_K, BATCH, "cuda", mode="random"
    )
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    if DEBUG_MODE:
        print(
            f"query_padding_mask.shape={query_padding_mask.shape} query_padding_mask={query_padding_mask}"
        )
        print(
            f"key_padding_mask.shape={key_padding_mask.shape} key_padding_mask={key_padding_mask}"
        )

        print(f"q.shape={q.shape} q={q}")
        print(f"k.shape={k.shape} k={k}")
        print(f"v.shape={v.shape} v={v}")
        print(f"q_unpad.shape={q_unpad.shape} q_unpad={q_unpad}")
        print(f"k_unpad.shape={k_unpad.shape} k_unpad={k_unpad}")
        print(f"v_unpad.shape={v_unpad.shape} v_unpad={v_unpad}")
        print(f"max_seqlens_q={max_seqlen_q }")
        print(f"max_seqlens_k={max_seqlen_k }")
        print(f"cu_seqlens_q={cu_seqlens_q }")
        print(f"cu_seqlens_k={cu_seqlens_k }")
    if version == "v3":
        if FP8:
            if DROPOUT > 0.0 or RETURN_LSE or RETURN_SOFTMAX:
                pytest.skip(
                    "flash_attn_varlen_func_v3 FP8 path doesn't support dropout/lse/attn_probs"
                )
            fp8_dtype = get_fp8_e4m3_dtype()
            group_size = (
                NUM_Q_HEADS // NUM_K_HEADS if NUM_Q_HEADS % NUM_K_HEADS == 0 else None
            )
            k_fp8, k_descale = _quantize_thd(k_unpad, fp8_dtype, cu_seqlens_k)
            v_fp8, v_descale = _quantize_thd(v_unpad, fp8_dtype, cu_seqlens_k)
            q_fp8, q_descale = _quantize_thd(
                q_unpad, fp8_dtype, cu_seqlens_q, group_size=group_size
            )
            triton_out = flash_attn_varlen_func_v3(
                q_fp8,
                k_fp8,
                v_fp8,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale=None,
                causal=CAUSAL,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
            )
        else:
            if DROPOUT > 0.0 or RETURN_LSE or RETURN_SOFTMAX:
                pytest.skip(
                    "flash_attn_varlen_func_v3 (non-FP8) doesn't expose dropout/lse/attn_probs"
                )
            triton_out = flash_attn_varlen_func_v3(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                softmax_scale=None,
                causal=CAUSAL,
            )
    else:
        if FP8:
            pytest.skip("FP8 supported only on version 'v3'")
        triton_out = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            dropout_p=DROPOUT,
            causal=CAUSAL,
            return_lse=RETURN_LSE,
            return_attn_probs=RETURN_SOFTMAX,
        )

    if version == "v2" and RETURN_LSE:
        assert len(triton_out) > 1
        lse = triton_out[1]
        if DEBUG_MODE:
            print(f"lse.shape={lse.shape}, lse={lse}")

    dropout_mask = None
    if version == "v2" and DROPOUT > 0.0 and RETURN_SOFTMAX:
        if RETURN_LSE:
            assert len(triton_out) == 3
            sd_mask = triton_out[2]
        else:
            assert len(triton_out) == 2
            sd_mask = triton_out[1]
        dropout_mask = sd_mask >= 0
        dropout_mask = pad_rearrange_dropout_mask(
            dropout_mask,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            SEQLEN_Q,
            SEQLEN_K,
            NUM_Q_HEADS,
        )
        dropout_mask = dropout_mask > 0
        if DEBUG_MODE:
            # print(f"sd_mask.shape={sd_mask.shape}, sd_mask={sd_mask}")
            print(
                f"dropout_mask.shape={dropout_mask.shape}, dropout_mask={dropout_mask}"
            )
    if version == "v2" and (RETURN_SOFTMAX or RETURN_LSE):
        triton_out = output_pad_fn(triton_out[0])
    else:
        triton_out = output_pad_fn(triton_out)
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

    torch_out = attention_ref(
        q,
        k,
        v,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        dropout_p=DROPOUT,
        dropout_mask=dropout_mask,
        causal=CAUSAL,
    )
    torch_out, attention_scores, _ = torch_out

    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")
        print(
            f"attention_scores.shape={attention_scores.shape}, attention_scores={attention_scores}"
        )

    if FP8:
        fp8_assert_close(
            triton_out, torch_out, atol=0.25, rtol=10
        )  # Lower tolerance for FP8
    else:
        torch.testing.assert_close(
            triton_out, torch_out.to(triton_out.dtype), atol=1e-1, rtol=1e-1
        )


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("BATCH", [1, 4, 57, 128])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (4, 4), (128, 128), (2, 1), (1, 2), (32, 16), (64, 128)],
)
@pytest.mark.parametrize("DROPOUT, CAUSAL", [(0.0, False), (0.0, True), (0.2, False)])
# @pytest.mark.parametrize('DROPOUT, CAUSAL',[(0.0, False),(0.0, True),(0.2, False),(0.2, True)]) #Debug Causal + Dropout. fails for seq >= 64
@pytest.mark.parametrize(
    "NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (16, 16), (2, 1), (48, 8)]
)
@pytest.mark.parametrize("HEAD_SZ", [8, 32, 128])
@pytest.mark.parametrize("FP8", [False, True])
@pytest.mark.parametrize("FUSED", [False, True])
def test_mha_backward(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    DROPOUT: float,
    CAUSAL: bool,
    FP8: bool,
    FUSED: bool,
    version: str,
    dtype=torch.float16,
):
    torch.cuda.empty_cache()
    torch.manual_seed(20)
    mha_set_use_fused_bwd_kernel(FUSED)
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    do = torch.randn_like(q)

    if DEBUG_MODE:
        print("--------------Triton (setup)----------------")
        print(f"version={version} FP8={FP8} DROPOUT={DROPOUT} CAUSAL={CAUSAL}")
        print(f"q.shape={q.shape} k.shape={k.shape} v.shape={v.shape}")

    # Branch by version similar to test_mha
    if version == "v3":
        if DROPOUT > 0.0:
            pytest.skip("flash_attn_func_v3 backward test: dropout unsupported")
        if FP8 and DROPOUT > 0.0:  # redundant safety
            pytest.skip("Flash Attention v3 FP8 path doesn't support dropout")
        with torch.enable_grad():
            if FP8:
                fp8_dtype = get_fp8_e4m3_dtype()
                group_size = (
                    NUM_Q_HEADS // NUM_K_HEADS
                    if NUM_Q_HEADS % NUM_K_HEADS == 0
                    else None
                )
                k_fp8, k_descale = _quantize_bshd(k, fp8_dtype)
                v_fp8, v_descale = _quantize_bshd(v, fp8_dtype)
                q_fp8, q_descale = _quantize_bshd(q, fp8_dtype, group_size=group_size)
                triton_out = flash_attn_func_v3(
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    softmax_scale=None,
                    causal=CAUSAL,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                lse = sd_mask = dropout_mask = None
                triton_dq, triton_dk, triton_dv = torch.autograd.grad(
                    triton_out, (q_fp8, k_fp8, v_fp8), do.clone()
                )
            else:
                triton_out = flash_attn_func_v3(
                    q,
                    k,
                    v,
                    softmax_scale=None,
                    causal=CAUSAL,
                )
                lse = sd_mask = dropout_mask = None
                triton_dq, triton_dk, triton_dv = torch.autograd.grad(
                    triton_out, (q, k, v), do.clone()
                )
    else:  # version == 'v2'
        pytest.skip("V2 Backward has accuracy issues")
        if FUSED and CAUSAL:
            pytest.skip("FUSED+CAUSAL results in NaNs")
        if FP8:
            pytest.skip("FP8 supported only on version 'v3'")
        with torch.enable_grad():
            triton_out = flash_attn_func(
                q,
                k,
                v,
                dropout_p=DROPOUT,
                causal=CAUSAL,
                return_lse=True,
                return_attn_probs=True,
            )
            assert len(triton_out) == 3
            triton_out, lse, sd_mask = triton_out[0], triton_out[1], triton_out[2]
            if DROPOUT > 0.0:
                dropout_mask = sd_mask >= 0
            else:
                dropout_mask = None
            triton_dq, triton_dk, triton_dv = torch.autograd.grad(
                triton_out, (q, k, v), do.clone()
            )

    if DEBUG_MODE:
        print("--------------Triton (post)----------------")
        print(f"triton_out.shape={triton_out.shape}")
        print(f"dq/dk/dv shapes: {triton_dq.shape} {triton_dk.shape} {triton_dv.shape}")

    # Reference backward (always PyTorch attention_ref)
    with torch.enable_grad():
        torch_out = attention_ref(
            q,
            k,
            v,
            dropout_p=(DROPOUT if version == "v2" else 0.0),
            dropout_mask=(dropout_mask if version == "v2" else None),
            causal=CAUSAL,
        )
    torch_out, attention_scores, _ = torch_out

    torch.testing.assert_close(
        triton_out, torch_out.to(triton_out.dtype), atol=1e-2, rtol=1e-2
    )

    torch_dq, torch_dk, torch_dv = torch.autograd.grad(torch_out, (q, k, v), do)

    if DEBUG_MODE:
        print("--------------Torch (post)----------------")
        print(f"torch_out.shape={torch_out.shape}")

    # Assertions
    if version == "v3":
        if FP8:
            fp8_assert_close(triton_out, torch_out, atol=ATOL_fp8, rtol=RTOL_fp8)
            fp8_assert_close(triton_dq, torch_dq, atol=ATOL_fp8, rtol=RTOL_fp8)
            fp8_assert_close(triton_dk, torch_dk, atol=ATOL_fp8, rtol=RTOL_fp8)
            fp8_assert_close(triton_dv, torch_dv, atol=ATOL_fp8, rtol=RTOL_fp8)
        else:
            torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(triton_dq, torch_dq, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(triton_dk, torch_dk, atol=1e-2, rtol=1e-2)
            torch.testing.assert_close(triton_dv, torch_dv, atol=1e-2, rtol=1e-2)
    else:  # v2 assertions
        torch.testing.assert_close(
            triton_out, torch_out.to(triton_out.dtype), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            triton_dq, torch_dq.to(triton_out.dtype), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            triton_dk, torch_dk.to(triton_out.dtype), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            triton_dv, torch_dv.to(triton_out.dtype), atol=1e-2, rtol=1e-2
        )


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("BATCH", [1, 4, 57, 128])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (4, 4), (128, 128), (2, 1), (1, 2), (32, 16), (64, 128)],
)
@pytest.mark.parametrize("DROPOUT, CAUSAL", [(0.0, False), (0.0, True)])
# @pytest.mark.parametrize('DROPOUT, CAUSAL',[(0.0, False),(0.0, True),(0.2, False),(0.2, True)]) #Debug Causal + Dropout. Fails for seq >=64
@pytest.mark.parametrize(
    "NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (16, 16), (2, 1), (48, 8)]
)
@pytest.mark.parametrize("HEAD_SZ", [8, 32, 128])
@pytest.mark.parametrize("FP8", [False, True])
@pytest.mark.parametrize("FUSED", [False, True])
def test_mha_backward_varlen(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    DROPOUT: float,
    CAUSAL: bool,
    FP8: bool,
    FUSED: bool,
    version: str,
    dtype=torch.float16,
):
    torch.cuda.empty_cache()
    torch.manual_seed(20)

    mha_set_use_fused_bwd_kernel(FUSED)

    # Create padded tensors and derive varlen packed representations
    q = torch.randn(
        (BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ),
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ),
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ),
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    query_padding_mask = generate_random_padding_mask(
        SEQLEN_Q, BATCH, "cuda", mode="random"
    )
    key_padding_mask = generate_random_padding_mask(
        SEQLEN_K, BATCH, "cuda", mode="random"
    )
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q_ref,
        k_ref,
        v_ref,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    q_unpad.requires_grad_(True)
    k_unpad.requires_grad_(True)
    v_unpad.requires_grad_(True)
    # Ensure padded reference tensors track gradients for comparison
    q_ref.requires_grad_(True)
    k_ref.requires_grad_(True)
    v_ref.requires_grad_(True)

    # Unified gradient seed: generate on padded output, then derive unpadded view
    do = torch.randn_like(q_ref)  # shape [B, S_q, H, D]
    # query_padding_mask likely bool; flatten batch*seq then mask to build unpadded gradient seed
    mask_flat = query_padding_mask.reshape(-1).to(torch.bool)
    do_unpad = do.view(BATCH * SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ)[mask_flat]

    # Top-level version branch similar to test_mha_backward
    if version == "v3":
        if DROPOUT > 0.0:
            pytest.skip("flash_attn_varlen_func_v3 backward test: dropout unsupported")
        with torch.enable_grad():
            if FP8:
                fp8_dtype = get_fp8_e4m3_dtype()
                group_size = (
                    NUM_Q_HEADS // NUM_K_HEADS
                    if NUM_Q_HEADS % NUM_K_HEADS == 0
                    else None
                )
                k_fp8, k_descale = _quantize_thd(k_unpad, fp8_dtype, cu_seqlens_k)
                v_fp8, v_descale = _quantize_thd(v_unpad, fp8_dtype, cu_seqlens_k)
                q_fp8, q_descale = _quantize_thd(
                    q_unpad, fp8_dtype, cu_seqlens_q, group_size=group_size
                )
                # Ensure fp8 tensors participate in autograd as leaves
                q_fp8 = q_fp8.detach().requires_grad_()
                k_fp8 = k_fp8.detach().requires_grad_()
                v_fp8 = v_fp8.detach().requires_grad_()
                triton_out = flash_attn_varlen_func_v3(
                    q_fp8,
                    k_fp8,
                    v_fp8,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale=None,
                    causal=CAUSAL,
                    q_descale=q_descale,
                    k_descale=k_descale,
                    v_descale=v_descale,
                )
                triton_dq, triton_dk, triton_dv = torch.autograd.grad(
                    triton_out,
                    (q_fp8, k_fp8, v_fp8),
                    do_unpad.clone(),
                    allow_unused=True,
                )
                if any(g is None for g in (triton_dq, triton_dk, triton_dv)):
                    missing = [
                        name
                        for g, name in zip(
                            (triton_dq, triton_dk, triton_dv), ["dq", "dk", "dv"]
                        )
                        if g is None
                    ]
                    pytest.fail(f"Missing gradients for FP8 varlen inputs: {missing}")
            else:
                triton_out = flash_attn_varlen_func_v3(
                    q_unpad,
                    k_unpad,
                    v_unpad,
                    cu_seqlens_q,
                    cu_seqlens_k,
                    max_seqlen_q,
                    max_seqlen_k,
                    softmax_scale=None,
                    causal=CAUSAL,
                )
                triton_dq, triton_dk, triton_dv = torch.autograd.grad(
                    triton_out,
                    (q_unpad, k_unpad, v_unpad),
                    do_unpad.clone(),
                    allow_unused=True,
                )
                if any(g is None for g in (triton_dq, triton_dk, triton_dv)):
                    missing = [
                        name
                        for g, name in zip(
                            (triton_dq, triton_dk, triton_dv), ["dq", "dk", "dv"]
                        )
                        if g is None
                    ]
                    pytest.fail(
                        f"Missing gradients for v3 varlen inputs (non-FP8): {missing}"
                    )
    else:  # version == 'v2'
        pytest.skip("V2 Backward has accuracy issues")
        if FUSED and CAUSAL:
            pytest.skip("FUSED+CAUSAL results in NaNs")
        if FP8:
            pytest.skip("FP8 supported only on version 'v3'")
        with torch.enable_grad():
            triton_out = flash_attn_varlen_func(
                q_unpad,
                k_unpad,
                v_unpad,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=DROPOUT,
                causal=CAUSAL,
                return_lse=True,
                return_attn_probs=True,
            )
            assert len(triton_out) == 3
            triton_core, lse, sd_mask = triton_out
            if DROPOUT > 0.0:
                dropout_mask = sd_mask >= 0
                dropout_mask = (
                    pad_rearrange_dropout_mask(
                        dropout_mask,
                        cu_seqlens_q,
                        cu_seqlens_k,
                        max_seqlen_q,
                        max_seqlen_k,
                        SEQLEN_Q,
                        SEQLEN_K,
                        NUM_Q_HEADS,
                    )
                    > 0
                )
            else:
                dropout_mask = None
            triton_out = triton_core
            triton_dq, triton_dk, triton_dv = torch.autograd.grad(
                triton_out,
                (q_unpad, k_unpad, v_unpad),
                do_unpad.clone(),
                allow_unused=True,
            )
            if any(g is None for g in (triton_dq, triton_dk, triton_dv)):
                missing = [
                    name
                    for g, name in zip(
                        (triton_dq, triton_dk, triton_dv), ["dq", "dk", "dv"]
                    )
                    if g is None
                ]
                pytest.fail(f"Missing gradients for v2 varlen inputs: {missing}")

    # Pad forward output and gradients back to padded space
    triton_out = output_pad_fn(triton_out)
    triton_dq = dq_pad_fn(triton_dq)
    triton_dk = dk_pad_fn(triton_dk)
    triton_dv = dk_pad_fn(triton_dv)

    # Reference forward & backward always on padded tensors
    with torch.enable_grad():
        torch_out = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            dropout_p=(DROPOUT if version == "v2" else 0.0),
            dropout_mask=(
                dropout_mask if (version == "v2" and DROPOUT > 0.0) else None
            ),
            causal=CAUSAL,
        )
    torch_out, attention_scores, _ = torch_out

    if version == "v3" and FP8:
        fp8_assert_close(triton_out, torch_out, atol=ATOL_fp8, rtol=RTOL_fp8)
    else:
        torch.testing.assert_close(
            triton_out, torch_out.to(triton_out.dtype), atol=1e-2, rtol=1e-2
        )

    # backward reference
    torch_dq, torch_dk, torch_dv = torch.autograd.grad(torch_out, (q, k, v), do)

    # Assertions
    if version == "v3" and FP8:
        fp8_assert_close(triton_dq, torch_dq, atol=ATOL_fp8, rtol=RTOL_fp8)
        fp8_assert_close(triton_dk, torch_dk, atol=ATOL_fp8, rtol=RTOL_fp8)
        fp8_assert_close(triton_dv, torch_dv, atol=ATOL_fp8, rtol=RTOL_fp8)
    else:
        torch.testing.assert_close(
            triton_dq, torch_dq.to(triton_out.dtype), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            triton_dk, torch_dk.to(triton_out.dtype), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            triton_dv, torch_dv.to(triton_out.dtype), atol=1e-2, rtol=1e-2
        )


@pytest.mark.parametrize("version", ["v2", "v3"])
@pytest.mark.parametrize("page_size", [None, 128])
@pytest.mark.parametrize("FP8", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize(
    "seqlen_q, seqlen_k, d",
    [
        (1, 64, 64),
        (8, 128, 64),
        (16, 256, 64),
        (32, 256, 128),
    ],
)
def test_mha_kvcache(
    seqlen_q: int,
    seqlen_k: int,
    d: int,
    causal: bool,
    FP8: bool,
    page_size: Optional[int],
    version: str,
):
    device = "cuda"
    torch.manual_seed(0)

    batch_size = 3
    nheads = 4
    nheads_k = nheads  # Only MHA path exercised

    base_dtype = torch.float16

    if page_size is None:
        k_cache = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=base_dtype
        )
        v_cache = torch.randn(
            batch_size, seqlen_k, nheads_k, d, device=device, dtype=base_dtype
        )
        page_table = None
        k_cache_for_kernel = k_cache
        v_cache_for_kernel = v_cache
    else:
        num_blocks_per_seq = math.ceil(seqlen_k / page_size)
        num_blocks = num_blocks_per_seq * batch_size * 3  # overprovision
        k_cache_blocks = torch.randn(
            num_blocks, page_size, nheads_k, d, device=device, dtype=base_dtype
        )
        v_cache_blocks = torch.randn(
            num_blocks, page_size, nheads_k, d, device=device, dtype=base_dtype
        )
        page_table = torch.randperm(num_blocks, device=device, dtype=torch.int32).view(
            batch_size, -1
        )
        # Gather blocks per batch then flatten
        flat_indices = page_table.flatten()
        gathered_k = k_cache_blocks.index_select(0, flat_indices)
        gathered_v = v_cache_blocks.index_select(0, flat_indices)
        # Reshape to (B, nblocks_per_seq, page_size, H, D) then merge seq dims
        nblocks_total = page_table.shape[1]
        k_cache = gathered_k.view(
            batch_size, nblocks_total, page_size, nheads_k, d
        ).permute(0, 1, 2, 3, 4)
        k_cache = k_cache.reshape(batch_size, nblocks_total * page_size, nheads_k, d)[
            :, :seqlen_k
        ]
        v_cache = gathered_v.view(
            batch_size, nblocks_total, page_size, nheads_k, d
        ).permute(0, 1, 2, 3, 4)
        v_cache = v_cache.reshape(batch_size, nblocks_total * page_size, nheads_k, d)[
            :, :seqlen_k
        ]
        k_cache_for_kernel = k_cache_blocks
        v_cache_for_kernel = v_cache_blocks

    cache_seqlens = torch.full(
        (batch_size,), seqlen_k, dtype=torch.int32, device=device
    )

    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=base_dtype)

    # Optional FP8 path (forward only) – only meaningful for v3 backend.
    # We still produce generic *kernel tensors so names don't imply dtype when FP8 is False.
    if FP8:
        fp8_dtype = get_fp8_e4m3_dtype()
        group_size = nheads // nheads_k if nheads % nheads_k == 0 else None
        if page_size is None:
            k_cache_kernel, k_descale = _quantize_bshd(k_cache_for_kernel, fp8_dtype)
            v_cache_kernel, v_descale = _quantize_bshd(v_cache_for_kernel, fp8_dtype)
        else:
            # Cast logical contiguous tensors then map back to block layout
            k_cache_fp8_logical, k_descale = _quantize_bshd(k_cache, fp8_dtype)
            v_cache_fp8_logical, v_descale = _quantize_bshd(v_cache, fp8_dtype)
            nblocks_total = page_table.shape[1]
            S = k_cache_fp8_logical.shape[1]
            full_blocks_tokens = nblocks_total * page_size
            if S < full_blocks_tokens:
                pad_tokens = full_blocks_tokens - S
                pad_k = torch.zeros(
                    k_cache_fp8_logical.shape[0],
                    pad_tokens,
                    k_cache_fp8_logical.shape[2],
                    k_cache_fp8_logical.shape[3],
                    dtype=k_cache_fp8_logical.dtype,
                    device=k_cache_fp8_logical.device,
                )
                pad_v = torch.zeros_like(pad_k)
                k_padded = torch.cat([k_cache_fp8_logical, pad_k], dim=1)
                v_padded = torch.cat([v_cache_fp8_logical, pad_v], dim=1)
            else:
                k_padded = k_cache_fp8_logical
                v_padded = v_cache_fp8_logical
            k_blocks_ordered = k_padded.view(
                batch_size, nblocks_total, page_size, nheads_k, d
            )
            v_blocks_ordered = v_padded.view(
                batch_size, nblocks_total, page_size, nheads_k, d
            )
            flat_indices = page_table.flatten()
            k_cache_kernel = torch.empty_like(k_cache_for_kernel, dtype=fp8_dtype)
            v_cache_kernel = torch.empty_like(v_cache_for_kernel, dtype=fp8_dtype)
            k_blocks_flat = k_blocks_ordered.reshape(-1, page_size, nheads_k, d)
            v_blocks_flat = v_blocks_ordered.reshape(-1, page_size, nheads_k, d)
            k_cache_kernel[flat_indices] = k_blocks_flat
            v_cache_kernel[flat_indices] = v_blocks_flat
        q_kernel, q_descale = _quantize_bshd(q, fp8_dtype, group_size=group_size)
    else:
        k_cache_kernel = k_cache_for_kernel
        v_cache_kernel = v_cache_for_kernel
        q_kernel = q
        q_descale = k_descale = v_descale = None

    # Reference: clone cache & apply append if needed
    k_cache_ref = k_cache.clone()  # already contiguous logical view
    v_cache_ref = v_cache.clone()
    final_used = cache_seqlens

    arange = torch.arange(seqlen_k, device=device).unsqueeze(0)
    key_padding_mask = arange < final_used.unsqueeze(1)

    max_used = final_used.max().item()
    k_eff = k_cache_ref[:, :max_used]
    v_eff = v_cache_ref[:, :max_used]
    key_mask_eff = key_padding_mask[:, :max_used]

    # Reference runs entirely on original fp16 tensors
    out_ref, _ = attention_ref(
        q,
        k_eff,
        v_eff,
        query_padding_mask=None,
        key_padding_mask=key_mask_eff,
        causal=causal,
        window_size=(-1, -1),
    )

    if version == "v3":
        out_kernel = flash_attn_with_kvcache_v3(
            q_kernel,
            k_cache_kernel,
            v_cache_kernel,
            cache_seqlens=cache_seqlens,
            causal=causal,
            window_size=(-1, -1),
            return_softmax_lse=False,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            page_table=page_table,
        )
    else:  # v2 path
        if FP8:
            pytest.skip(
                "v2 kvcache wrapper currently lacks FP8 descale path; skip FP8 for version=='v2'"
            )
        out_kernel = flash_attn_with_kvcache(
            q_kernel,
            k_cache_kernel,
            v_cache_kernel,
            cache_seqlens=cache_seqlens,
            causal=causal,
            window_size=(-1, -1),
            block_table=page_table,
            return_softmax_lse=False,
        )

    assert out_kernel.shape == out_ref.shape
    # Accuracy assertion with FP8-aware tolerances
    if FP8:
        fp8_assert_close(out_kernel, out_ref, atol=ATOL_fp8, rtol=RTOL_fp8)
    else:
        torch.testing.assert_close(
            out_kernel, out_ref.to(out_kernel.dtype), atol=1e-2, rtol=1e-2
        )
