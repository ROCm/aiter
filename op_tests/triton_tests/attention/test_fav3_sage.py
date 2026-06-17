# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
import torch
import pytest
import logging
import numpy as np
import math
from aiter.test_mha_common import (
    attention_ref,
    attention_ref_block_sparse,
)
from aiter.ops.triton.attention.fav3_sage import (
    fav3_sage_wrapper_func,
    get_sage_fwd_configs,
)
from aiter.ops.triton.attention.utils import block_attn_mask_to_ragged_lut
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.attention.fav3_sage_attention_mxfp4_wrapper import (
    fav3_sage_mxfp4_wrapper,
    get_sage_fwd_configs_mxfp4,
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

    torch.manual_seed(20)
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
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_sage(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    layout: str,
    dtype=torch.bfloat16,
):
    HEAD_SZ = 128

    torch.manual_seed(20)
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
        return_lse=False,
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


@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(128, 128), (64, 128)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(2, 2), (16, 16)])
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("layout", ["bshd"])
def test_sage_block_sparse_none(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    layout: str,
    dtype=torch.bfloat16,
):
    """With block_lut=None, output must match non-sparse path (backward compat)."""
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
        return_lse=False,
        layout=layout,
        block_lut=None,
    )
    triton_out_full = fav3_sage_wrapper_func(
        q, k, v, softmax_scale, causal=False, return_lse=False, layout=layout
    )
    check_attention_outputs(
        triton_out,
        triton_out_full,
        fp8=True,
        atol=ATOL_fp8,
        rtol=RTOL_fp8,
        max_diff_percentage=0.5,
    )


@pytest.mark.parametrize("BATCH", [1, 2])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(256, 256)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4)])
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("layout", ["bshd"])
def test_sage_block_sparse_vs_reference(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    layout: str,
    dtype=torch.bfloat16,
):
    """Block-sparse output matches reference that applies the same block mask."""
    torch.cuda.empty_cache()
    config = get_sage_fwd_configs()
    BLOCK_M, BLOCK_N = config["BLOCK_M"], config["BLOCK_N"]
    num_q_blocks = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (SEQLEN_K + BLOCK_N - 1) // BLOCK_N

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
    # Diagonal block mask: Q block qb attends only to KV block qb (and qb-1 for a small band)
    block_attn_mask = torch.zeros(
        BATCH, num_q_blocks, num_kv_blocks, dtype=torch.bool, device="cuda"
    )
    for qb in range(num_q_blocks):
        for kb in range(num_kv_blocks):
            if abs(qb - kb) <= 1:
                block_attn_mask[:, qb, kb] = True

    block_lut = block_attn_mask_to_ragged_lut(block_attn_mask, num_heads=NUM_Q_HEADS)
    softmax_scale = 1.0 / math.sqrt(HEAD_SZ)
    triton_out = fav3_sage_wrapper_func(
        q,
        k,
        v,
        softmax_scale,
        causal=False,
        return_lse=False,
        layout=layout,
        block_lut=block_lut,
    )

    torch_out, _, _ = attention_ref_block_sparse(
        q, k, v, block_attn_mask, BLOCK_M, BLOCK_N
    )
    assert triton_out.shape == torch_out.shape
    check_attention_outputs(
        triton_out,
        torch_out,
        fp8=True,
        atol=ATOL_fp8,
        rtol=RTOL_fp8,
        max_diff_percentage=0.5,
    )


@pytest.mark.parametrize("layout", ["bshd"])
def test_sage_block_sparse_empty_kv_blocks(layout: str, dtype=torch.bfloat16):
    """When a Q block has no KV blocks allowed, that block's output is zero."""
    torch.cuda.empty_cache()
    BATCH, SEQLEN_Q, SEQLEN_K = 1, 256, 256
    NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ = 4, 4, 128
    config = get_sage_fwd_configs()
    BLOCK_M, BLOCK_N = config["BLOCK_M"], config["BLOCK_N"]
    num_q_blocks = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (SEQLEN_K + BLOCK_N - 1) // BLOCK_N

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
    # First Q block attends to nothing; others attend to all KV blocks
    block_attn_mask = torch.ones(
        BATCH, num_q_blocks, num_kv_blocks, dtype=torch.bool, device="cuda"
    )
    block_attn_mask[:, 0, :] = False

    block_lut = block_attn_mask_to_ragged_lut(block_attn_mask, num_heads=NUM_Q_HEADS)
    softmax_scale = 1.0 / math.sqrt(HEAD_SZ)
    triton_out = fav3_sage_wrapper_func(
        q,
        k,
        v,
        softmax_scale,
        causal=False,
        return_lse=False,
        layout=layout,
        block_lut=block_lut,
    )
    torch_out, _, _ = attention_ref_block_sparse(
        q, k, v, block_attn_mask, BLOCK_M, BLOCK_N
    )
    check_attention_outputs(
        triton_out,
        torch_out,
        fp8=True,
        atol=ATOL_fp8,
        rtol=RTOL_fp8,
        max_diff_percentage=0.5,
    )


@pytest.mark.parametrize("layout", ["bshd"])
def test_sage_block_sparse_empty_kv_blocks_lse_is_neg_inf(
    layout: str, dtype=torch.bfloat16
):
    """
    For Q blocks that have no allowed KV blocks the kernel takes the
    `_no_blocks` early-exit path. The returned softmax_lse for those rows must
    be -inf so FA-style ring merges treat the shard as zero-weight; non-empty
    Q blocks must keep finite LSE.
    """
    torch.cuda.empty_cache()
    BATCH, SEQLEN_Q, SEQLEN_K = 1, 256, 256
    NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ = 4, 4, 128
    config = get_sage_fwd_configs()
    BLOCK_M, BLOCK_N = config["BLOCK_M"], config["BLOCK_N"]
    num_q_blocks = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (SEQLEN_K + BLOCK_N - 1) // BLOCK_N

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
    block_attn_mask = torch.ones(
        BATCH, num_q_blocks, num_kv_blocks, dtype=torch.bool, device="cuda"
    )
    block_attn_mask[:, 0, :] = False

    block_lut = block_attn_mask_to_ragged_lut(block_attn_mask, num_heads=NUM_Q_HEADS)
    softmax_scale = 1.0 / math.sqrt(HEAD_SZ)
    _, lse = fav3_sage_wrapper_func(
        q,
        k,
        v,
        softmax_scale,
        causal=False,
        return_lse=True,
        layout=layout,
        block_lut=block_lut,
    )

    assert lse.shape == (BATCH, NUM_Q_HEADS, SEQLEN_Q)
    empty_rows = lse[:, :, :BLOCK_M]
    rest_rows = lse[:, :, BLOCK_M:]
    assert torch.isneginf(
        empty_rows
    ).all(), f"Expected -inf LSE for empty Q block, got {empty_rows}"
    assert torch.isfinite(
        rest_rows
    ).all(), "Expected finite LSE for Q blocks with valid KV"


@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", [(64, 16), (128, 32), (256, 64)])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4), (16, 4)])
def test_sage_causal_above_diagonal_lse_is_neg_inf(
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    layout: str,
    dtype=torch.bfloat16,
):
    """
    With causal masking and seqlen_q > seqlen_k, the first (seqlen_q - seqlen_k)
    Q rows have no valid K positions. Their softmax_lse must be -inf (logsumexp
    over empty set) so FA-style ring merges treat them correctly.
    """
    torch.cuda.empty_cache()
    BATCH, HEAD_SZ = 2, 128
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

    _, lse = fav3_sage_wrapper_func(
        q,
        k,
        v,
        softmax_scale,
        causal=True,
        return_lse=True,
        layout=layout,
    )

    assert lse.shape == (BATCH, NUM_Q_HEADS, SEQLEN_Q)
    n_above = SEQLEN_Q - SEQLEN_K
    above = lse[:, :, :n_above]
    below = lse[:, :, n_above:]
    assert torch.isneginf(
        above
    ).all(), f"Expected -inf LSE for rows above causal diagonal, got {above}"
    assert torch.isfinite(
        below
    ).all(), "Expected finite LSE for rows on/below causal diagonal"


@pytest.mark.parametrize("BATCH", [1, 4, 57, 128])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(1, 1), (4, 4), (128, 128), (2, 1), (1, 2), (32, 16), (64, 128)],
)
@pytest.mark.parametrize(
    "NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (16, 16), (2, 1), (48, 8)]
)
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("qsmooth", [True, False])
def test_sage_mxfp4(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    causal: bool,
    qsmooth: bool,
    dtype=torch.bfloat16,
):
    HEAD_SZ = 128
    layout = "bhsd"
    hadamard_rotate = True  # hadamard expected to be on

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


@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", [(256, 256), (256, 512)])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(2, 2), (16, 16)])
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("layout", ["bhsd"])
def test_sage_mxfp4_block_sparse_none(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    layout: str,
    dtype=torch.bfloat16,
):
    """With block_lut=None, MXFP4 output must match non-sparse path."""
    if not arch_info.is_fp4_avail():
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
        q, k, v, causal=False, layout=layout, hadamard_rotation=True, block_lut=None
    )
    triton_out_full = fav3_sage_mxfp4_wrapper(
        q, k, v, causal=False, layout=layout, hadamard_rotation=True
    )
    check_attention_outputs(
        triton_out,
        triton_out_full,
        fp8=True,
        atol=ATOL_fp8,
        rtol=RTOL_fp8,
        max_diff_percentage=0.5,
    )


@pytest.mark.parametrize("BATCH", [1, 2])
@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", [(512, 512)])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4)])
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("layout", ["bhsd"])
def test_sage_mxfp4_block_sparse_vs_reference(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    layout: str,
    dtype=torch.bfloat16,
):
    """Block-sparse MXFP4 output matches reference that applies the same block mask."""
    if not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 not supported on this architecture")
    torch.cuda.empty_cache()
    torch.manual_seed(20)

    config = get_sage_fwd_configs_mxfp4()
    BLOCK_M, BLOCK_N = config["BLOCK_M"], config["BLOCK_N"]
    num_q_blocks = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (SEQLEN_K + BLOCK_N - 1) // BLOCK_N

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

    # Band mask: Q block qb attends to KV blocks within distance 1
    block_attn_mask = torch.zeros(
        BATCH, num_q_blocks, num_kv_blocks, dtype=torch.bool, device="cuda"
    )
    for qb in range(num_q_blocks):
        for kb in range(num_kv_blocks):
            if abs(qb - kb) <= 1:
                block_attn_mask[:, qb, kb] = True

    block_lut = block_attn_mask_to_ragged_lut(block_attn_mask, num_heads=NUM_Q_HEADS)
    triton_out = fav3_sage_mxfp4_wrapper(
        q,
        k,
        v,
        causal=False,
        layout=layout,
        hadamard_rotation=True,
        block_lut=block_lut,
    )

    # Reference expects bshd
    if layout == "bhsd":
        q_ref = q.permute(0, 2, 1, 3).contiguous()
        k_ref = k.permute(0, 2, 1, 3).contiguous()
        v_ref = v.permute(0, 2, 1, 3).contiguous()
    else:
        q_ref, k_ref, v_ref = q, k, v

    torch_out, _, _ = attention_ref_block_sparse(
        q_ref, k_ref, v_ref, block_attn_mask, BLOCK_M, BLOCK_N
    )
    if layout == "bhsd":
        torch_out = torch_out.permute(0, 2, 1, 3).contiguous()

    assert triton_out.shape == torch_out.shape
    check_attention_outputs(
        triton_out,
        torch_out,
        fp8=True,
        atol=ATOL_fp8,
        rtol=RTOL_fp8,
        max_diff_percentage=1.5,
    )


@pytest.mark.parametrize("layout", ["bhsd"])
def test_sage_mxfp4_block_sparse_empty_kv_blocks(layout: str, dtype=torch.bfloat16):
    """When a Q block has no KV blocks allowed, that block's output is zero."""
    if not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 not supported on this architecture")
    torch.cuda.empty_cache()
    torch.manual_seed(20)

    BATCH, SEQLEN_Q, SEQLEN_K = 1, 512, 512
    NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ = 4, 4, 128
    config = get_sage_fwd_configs_mxfp4()
    BLOCK_M, BLOCK_N = config["BLOCK_M"], config["BLOCK_N"]
    num_q_blocks = (SEQLEN_Q + BLOCK_M - 1) // BLOCK_M
    num_kv_blocks = (SEQLEN_K + BLOCK_N - 1) // BLOCK_N

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

    # First Q block attends to nothing; others attend to all KV blocks
    block_attn_mask = torch.ones(
        BATCH, num_q_blocks, num_kv_blocks, dtype=torch.bool, device="cuda"
    )
    block_attn_mask[:, 0, :] = False

    block_lut = block_attn_mask_to_ragged_lut(block_attn_mask, num_heads=NUM_Q_HEADS)
    triton_out = fav3_sage_mxfp4_wrapper(
        q,
        k,
        v,
        causal=False,
        layout=layout,
        hadamard_rotation=True,
        block_lut=block_lut,
    )

    if layout == "bhsd":
        q_ref = q.permute(0, 2, 1, 3).contiguous()
        k_ref = k.permute(0, 2, 1, 3).contiguous()
        v_ref = v.permute(0, 2, 1, 3).contiguous()
    else:
        q_ref, k_ref, v_ref = q, k, v

    torch_out, _, _ = attention_ref_block_sparse(
        q_ref, k_ref, v_ref, block_attn_mask, BLOCK_M, BLOCK_N
    )
    if layout == "bhsd":
        torch_out = torch_out.permute(0, 2, 1, 3).contiguous()

    check_attention_outputs(
        triton_out,
        torch_out,
        fp8=True,
        atol=ATOL_fp8,
        rtol=RTOL_fp8,
        max_diff_percentage=1.5,
    )


@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", [(256, 256), (512, 512)])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4), (16, 16), (16, 4)])
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_sage_return_lse_matches_reference(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    layout: str,
    dtype=torch.bfloat16,
):
    """
    With smooth_k=True the kernel computes LSE against (K - mean(K)). The
    wrapper's correction term should make the returned LSE match the LSE that
    an un-smoothed K would produce (within sage's int8/fp8 quant noise).
    This is the property that FA-style ring-attention merging relies on.
    """
    HEAD_SZ = 128
    torch.cuda.empty_cache()
    torch.manual_seed(20)
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

    _, triton_lse = fav3_sage_wrapper_func(
        q,
        k,
        v,
        softmax_scale,
        causal=False,
        return_lse=True,
        layout=layout,
        smooth_k=True,
    )

    if layout == "bshd":
        q_b = q.permute(0, 2, 1, 3).contiguous()
        k_b = k.permute(0, 2, 1, 3).contiguous()
    else:
        q_b = q
        k_b = k
    if NUM_Q_HEADS != NUM_K_HEADS:
        assert NUM_Q_HEADS % NUM_K_HEADS == 0
        k_b = k_b.repeat_interleave(NUM_Q_HEADS // NUM_K_HEADS, dim=1)
    qk = (q_b.float() @ k_b.float().transpose(-1, -2)) * softmax_scale
    lse_ref = torch.logsumexp(qk, dim=-1)

    assert (
        triton_lse.shape == lse_ref.shape
    ), f"LSE shape {tuple(triton_lse.shape)} != reference {tuple(lse_ref.shape)}"
    # Pre-PR (no K-mean correction) the LSE is offset by
    # softmax_scale * Q . mean(K)^T, which is O(1)+ in magnitude.
    # With the correction, the residual is bounded by sage's int8/fp8 quant noise.
    torch.testing.assert_close(
        triton_lse,
        lse_ref.to(triton_lse.dtype),
        atol=2e-1,
        rtol=5e-2,
    )


def _fa_merge_partial(out_a, lse_a, out_b, lse_b, layout: str):
    """FA log-sum-exp partial merge. LSE is (B, H_q, S_q); output layout varies."""
    m = torch.maximum(lse_a, lse_b)
    w_a = (lse_a - m).exp()
    w_b = (lse_b - m).exp()
    denom = w_a + w_b
    if layout == "bhsd":
        w_a4 = w_a.unsqueeze(-1)
        w_b4 = w_b.unsqueeze(-1)
        denom4 = denom.unsqueeze(-1)
    else:  # bshd: out is (B, S_q, H_q, D); merge weights into that layout.
        w_a4 = w_a.transpose(1, 2).unsqueeze(-1)
        w_b4 = w_b.transpose(1, 2).unsqueeze(-1)
        denom4 = denom.transpose(1, 2).unsqueeze(-1)
    out = (out_a * w_a4 + out_b * w_b4) / denom4
    lse = m + denom.log()
    return out, lse


@pytest.mark.parametrize("BATCH", [1, 2])
@pytest.mark.parametrize("SEQLEN", [512, 1024])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4), (8, 4)])
@pytest.mark.parametrize("RING_DEGREE", [2, 4])
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
def test_sage_ring_merge_matches_single_call(
    BATCH: int,
    SEQLEN: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    RING_DEGREE: int,
    layout: str,
    dtype=torch.bfloat16,
):
    """
    Mimic ring attention: split K/V along the sequence dim into RING_DEGREE shards,
    run sage per shard with return_lse=True, FA-merge the partial (out, lse) pairs,
    and assert the merged output matches a single-call sage forward (within sage's
    int8/fp8 quant noise). Pre-PR this diverges with RING_DEGREE because each
    shard uses a different K mean; post-PR it stays bounded.
    """
    HEAD_SZ = 128
    assert SEQLEN % RING_DEGREE == 0

    torch.cuda.empty_cache()
    torch.manual_seed(20)
    softmax_scale = 1.0 / math.sqrt(HEAD_SZ)

    q, k, v = input_helper(
        BATCH,
        NUM_Q_HEADS,
        NUM_K_HEADS,
        SEQLEN,
        SEQLEN,
        HEAD_SZ,
        HEAD_SZ,
        dtype,
        layout,
    )

    full_out, _ = fav3_sage_wrapper_func(
        q,
        k,
        v,
        softmax_scale,
        causal=False,
        return_lse=True,
        layout=layout,
        smooth_k=True,
    )

    seq_dim_kv = 2 if layout == "bhsd" else 1
    k_shards = k.chunk(RING_DEGREE, dim=seq_dim_kv)
    v_shards = v.chunk(RING_DEGREE, dim=seq_dim_kv)

    out_acc = None
    lse_acc = None
    for k_s, v_s in zip(k_shards, v_shards):
        out_s, lse_s = fav3_sage_wrapper_func(
            q,
            k_s.contiguous(),
            v_s.contiguous(),
            softmax_scale,
            causal=False,
            return_lse=True,
            layout=layout,
            smooth_k=True,
        )
        out_s = out_s.float()
        lse_s = lse_s.float()
        if out_acc is None:
            out_acc, lse_acc = out_s, lse_s
        else:
            out_acc, lse_acc = _fa_merge_partial(out_acc, lse_acc, out_s, lse_s, layout)

    merged_out = out_acc.to(full_out.dtype)
    assert merged_out.shape == full_out.shape

    check_attention_outputs(
        merged_out,
        full_out,
        fp8=True,
        atol=ATOL_fp8,
        rtol=RTOL_fp8,
        max_diff_percentage=1.0,
    )


@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", [(256, 256), (512, 512)])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4), (16, 16), (16, 4)])
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
@pytest.mark.parametrize("qsmooth", [False, True])
def test_sage_mxfp4_return_lse_matches_reference(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    layout: str,
    qsmooth: bool,
    dtype=torch.bfloat16,
):
    """
    With smooth_k=True the kernel computes LSE against (K - mean(K)). The
    wrapper's correction term should make the returned LSE match the LSE that
    an un-smoothed K would produce (within mxfp4 quant noise). This is the
    property that FA-style ring-attention merging relies on.
    """
    if not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 not supported on this architecture")

    HEAD_SZ = 128
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

    _, triton_lse = fav3_sage_mxfp4_wrapper(
        q,
        k,
        v,
        causal=False,
        layout=layout,
        q_smooth=qsmooth,
        hadamard_rotation=True,
        return_lse=True,
        smooth_k=True,
    )

    softmax_scale = 1.0 / math.sqrt(HEAD_SZ)
    if layout == "bshd":
        q_b = q.permute(0, 2, 1, 3).contiguous()
        k_b = k.permute(0, 2, 1, 3).contiguous()
    else:
        q_b = q
        k_b = k
    if NUM_Q_HEADS != NUM_K_HEADS:
        assert NUM_Q_HEADS % NUM_K_HEADS == 0
        k_b = k_b.repeat_interleave(NUM_Q_HEADS // NUM_K_HEADS, dim=1)
    qk = (q_b.float() @ k_b.float().transpose(-1, -2)) * softmax_scale
    lse_ref = torch.logsumexp(qk, dim=-1)

    assert (
        triton_lse.shape == lse_ref.shape
    ), f"LSE shape {tuple(triton_lse.shape)} != reference {tuple(lse_ref.shape)}"
    # mxfp4 quant noise is larger than int8 sage; use loose tolerances.
    torch.testing.assert_close(
        triton_lse,
        lse_ref.to(triton_lse.dtype),
        atol=4e-1,
        rtol=1e-1,
    )


@pytest.mark.parametrize("BATCH", [1, 2])
@pytest.mark.parametrize("SEQLEN", [512, 1024])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4), (8, 4)])
@pytest.mark.parametrize("RING_DEGREE", [2, 4])
@pytest.mark.parametrize("layout", ["bhsd", "bshd"])
@pytest.mark.parametrize("qsmooth", [False, True])
def test_sage_mxfp4_ring_merge_matches_single_call(
    BATCH: int,
    SEQLEN: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    RING_DEGREE: int,
    layout: str,
    qsmooth: bool,
    dtype=torch.bfloat16,
):
    """
    Mimic ring attention: split K/V along the sequence dim into RING_DEGREE shards,
    run mxfp4 sage per shard with return_lse=True, FA-merge the partial (out, lse)
    pairs, and assert the merged output matches a single-call mxfp4 sage forward
    (within mxfp4 quant noise). Pre-fix this diverges with RING_DEGREE because
    each shard uses a different K mean; post-fix it stays bounded.
    """
    if not arch_info.is_fp4_avail():
        pytest.skip("MXFP4 not supported on this architecture")

    HEAD_SZ = 128
    assert SEQLEN % RING_DEGREE == 0

    torch.cuda.empty_cache()
    torch.manual_seed(20)

    q, k, v = input_helper(
        BATCH,
        NUM_Q_HEADS,
        NUM_K_HEADS,
        SEQLEN,
        SEQLEN,
        HEAD_SZ,
        HEAD_SZ,
        dtype,
        layout,
    )

    full_out, _ = fav3_sage_mxfp4_wrapper(
        q,
        k,
        v,
        causal=False,
        layout=layout,
        q_smooth=qsmooth,
        hadamard_rotation=True,
        return_lse=True,
        smooth_k=True,
    )

    seq_dim_kv = 2 if layout == "bhsd" else 1
    k_shards = k.chunk(RING_DEGREE, dim=seq_dim_kv)
    v_shards = v.chunk(RING_DEGREE, dim=seq_dim_kv)

    out_acc = None
    lse_acc = None
    for k_s, v_s in zip(k_shards, v_shards):
        out_s, lse_s = fav3_sage_mxfp4_wrapper(
            q,
            k_s.contiguous(),
            v_s.contiguous(),
            causal=False,
            layout=layout,
            q_smooth=qsmooth,
            hadamard_rotation=True,
            return_lse=True,
            smooth_k=True,
        )
        out_s = out_s.float()
        lse_s = lse_s.float()
        if out_acc is None:
            out_acc, lse_acc = out_s, lse_s
        else:
            out_acc, lse_acc = _fa_merge_partial(out_acc, lse_acc, out_s, lse_s, layout)

    merged_out = out_acc.to(full_out.dtype)
    assert merged_out.shape == full_out.shape

    check_attention_outputs(
        merged_out,
        full_out,
        fp8=True,
        atol=ATOL_fp8,
        rtol=RTOL_fp8,
        max_diff_percentage=1.5,
    )
