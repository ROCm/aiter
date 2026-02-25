# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
import torch
import pytest
import logging
import numpy as np
import math
from aiter.test_mha_common import (
    attention_ref,
)
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func
from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
    fav3_sage_mxfp4_wrapper,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DEBUG_MODE = False
ATOL_fp8 = 3.0e-1
RTOL_fp8 = 2.5e-1


def compare_accuracy(current, reference):
    """Print quick statistics comparing FP8 and SageAttn tensors."""
    current_f = current.float()
    reference_f = reference.float()
    abs_diff = torch.abs(reference_f - current_f)

    print("Output Tensor Stats:")
    print(
        f"  Reference ({tuple(reference_f.shape)}): min={reference_f.min().item():.6f}, max={reference_f.max().item():.6f}, "
        f"mean={reference_f.mean().item():.6f}, std={reference_f.std().item():.6f}"
    )
    print(
        f"  Test      ({tuple(current_f.shape)}): min={current_f.min().item():.6f}, max={current_f.max().item():.6f}, "
        f"mean={current_f.mean().item():.6f}, std={current_f.std().item():.6f}"
    )

    print("Correctness Comparison:")
    print(f"  Mean Absolute Error: {abs_diff.mean().item():.6e}")
    print(f"  Max Absolute Error: {abs_diff.max().item():.6e}")
    print(f"  Std Absolute Error: {abs_diff.std().item():.6e}")
    ref_flat = reference_f.reshape(-1)
    test_flat = current_f.reshape(-1)
    cos_sim = torch.nn.functional.cosine_similarity(
        ref_flat.unsqueeze(0), test_flat.unsqueeze(0)
    )
    print(f"  Cosine Similarity: {cos_sim.item():.8f}")


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
    """Assert tensors are close with tolerance for small percentage of elements"""
    # standard comparison
    abs_diff = torch.abs(tensor_a - tensor_b)
    rel_diff = abs_diff / torch.abs(tensor_b.clamp(min=1e-6))

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

    flat_to_idx = lambda flat_idx, shape: np.unravel_index(  # noqa: E731
        flat_idx, shape
    )

    max_abs_pos = flat_to_idx(max_abs_idx, tensor_a.shape)
    max_rel_pos = flat_to_idx(max_rel_idx, tensor_a.shape)

    max_abs_diff = abs_diff.flatten()[max_abs_idx].item()
    max_rel_diff = rel_diff.flatten()[max_rel_idx].item()

    raise AssertionError(
        f"Tensors not close enough! {failed_percentage:.6f}% elements exceed tolerance.\n"
        f"Greatest absolute difference: {max_abs_diff} at index {max_abs_pos} (up to {atol} allowed)\n"
        f"Greatest relative difference: {max_rel_diff} at index {max_rel_pos} (up to {rtol} allowed)"
    )


def _tensor_from_result(result):
    if isinstance(result, torch.Tensor):
        return result
    if isinstance(result, (list, tuple)) and result:
        return _tensor_from_result(result[0])
    raise TypeError(f"Unsupported result type for comparison: {type(result)}")


def check_attention_outputs(
    current,
    reference,
    fp8=False,
    atol=None,
    rtol=None,
    max_diff_percentage=0.5,
):
    current_tensor = _tensor_from_result(current)
    reference_tensor = _tensor_from_result(reference).to(current_tensor.dtype)

    if fp8:
        fp8_assert_close(
            current_tensor,
            reference_tensor,
            atol=atol or ATOL_fp8,
            rtol=rtol or RTOL_fp8,
            max_diff_percentage=max_diff_percentage,
        )
    else:
        torch.testing.assert_close(
            current_tensor,
            reference_tensor,
            atol=atol or 1e-2,
            rtol=rtol or 1e-2,
        )


def input_helper(
    BATCH,
    HQ,
    HK,
    N_CTX_Q,
    N_CTX_K,
    D_HEAD,
    D_HEAD_V,
    dtype,
    layout,
):
    # Generate base inputs in BHSD layout which is the layout used in wan model.
    # Set up tensor shapes based on layout
    if layout == "bhsd":
        q_shape = (BATCH, HQ, N_CTX_Q, D_HEAD)
        k_shape = (BATCH, HK, N_CTX_K, D_HEAD)
        v_shape = (BATCH, HK, N_CTX_K, D_HEAD_V)
    else:  # bshd
        q_shape = (BATCH, N_CTX_Q, HQ, D_HEAD)
        k_shape = (BATCH, N_CTX_K, HK, D_HEAD)
        v_shape = (BATCH, N_CTX_K, HK, D_HEAD_V)

    q = torch.randn(q_shape, device="cuda", dtype=dtype)
    k = torch.randn(k_shape, device="cuda", dtype=dtype)
    v = torch.randn(v_shape, device="cuda", dtype=dtype)
    q.requires_grad = False
    k.requires_grad = False
    v.requires_grad = False

    return q, k, v


@pytest.mark.parametrize("BATCH", [1, 4, 57, 128])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (4, 4), (128, 128), (2, 1), (1, 2), (32, 16), (64, 128)],
)
@pytest.mark.parametrize(
    "NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (16, 16), (2, 1), (48, 8)]
)
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_sage(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    layout: str,
    dtype=torch.bfloat16,
):
    torch.cuda.empty_cache()

    softmax_scale = 1.0 / math.sqrt(HEAD_SZ)

    q, k, v = input_helper(
        BATCH,
        NUM_Q_HEADS,
        NUM_K_HEADS,
        SEQLEN_Q,
        SEQLEN_K,
        HEAD_SZ,
        HEAD_SZ,
        dtype,
        layout,
    )

    triton_out = fav3_sage_wrapper_func(
        q,
        k,
        v,
        softmax_scale,
        causal=False,
        inference_mode=True,
        layout=layout,
    )

    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

    if layout == "bhsd":
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

    torch_out = attention_ref(q, k, v, dropout_p=0.0, dropout_mask=None, causal=False)
    torch_out, attention_scores, _ = torch_out

    if layout == "bhsd":
        torch_out = torch_out.permute(0, 2, 1, 3).contiguous()

    assert torch_out.shape == triton_out.shape

    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")
        print(
            f"attention_scores.shape={attention_scores.shape}, attention_scores={attention_scores}"
        )

    check_attention_outputs(
        triton_out,
        torch_out,
        fp8=True,
        atol=ATOL_fp8,
        rtol=RTOL_fp8,
        max_diff_percentage=0.5,
    )


@pytest.mark.parametrize("BATCH", [1, 4, 57, 128])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (4, 4), (128, 128), (2, 1), (1, 2), (32, 16), (64, 128)],
)
@pytest.mark.parametrize(
    "NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (16, 16), (2, 1), (48, 8)]
)
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("layout", ["bhsd"])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("qsmooth", [True, False])
@pytest.mark.parametrize("hadamard_rotate", [True, False])
def test_sage_mxfp4(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    layout: str,
    causal: bool,
    qsmooth: bool,
    hadamard_rotate: bool,
    dtype=torch.bfloat16,
):

    if not (arch_info.is_fp4_avail()):
        pytest.skip("MXFP4 not supported on this architecture")

    torch.cuda.empty_cache()
    torch.manual_seed(20)

    q, k, v = input_helper(
        BATCH,
        NUM_Q_HEADS,
        NUM_K_HEADS,
        SEQLEN_Q,
        SEQLEN_K,
        HEAD_SZ,
        HEAD_SZ,
        dtype,
        layout,
    )

    triton_out = fav3_sage_mxfp4_wrapper(
        q,
        k,
        v,
        causal=causal,
        layout=layout,
        q_smooth=qsmooth,
        hadamard_rotation=hadamard_rotate,
    )

    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

    if layout == "bhsd":
        q = q.permute(0, 2, 1, 3).contiguous()
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

    torch_out = attention_ref(q, k, v, dropout_p=0.0, dropout_mask=None, causal=causal)
    torch_out, attention_scores, _ = torch_out

    if layout == "bhsd":
        torch_out = torch_out.permute(0, 2, 1, 3).contiguous()

    assert torch_out.shape == triton_out.shape

    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")
        print(
            f"attention_scores.shape={attention_scores.shape}, attention_scores={attention_scores}"
        )

    check_attention_outputs(
        triton_out,
        torch_out,
        fp8=True,
        atol=ATOL_fp8,
        rtol=RTOL_fp8,
        max_diff_percentage=1.5,
    )
