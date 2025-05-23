import torch
import pytest
import logging
import numpy as np
from aiter.ops.triton.mha import (
    flash_attn_func,
    flash_attn_fp8_func,
    flash_attn_varlen_func,
    flash_attn_varlen_fp8_func,
)
from aiter.test_mha_common import (
    attention_ref,
    generate_random_padding_mask,
    generate_qkv,
)

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


@pytest.mark.parametrize("BATCH", [1, 8])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(256, 128), (128, 256), (544, 544)],
)
@pytest.mark.parametrize("DROPOUT, CAUSAL", [(0.0, False), (0.0, True)])
# dropout > 0.0 gives failures
@pytest.mark.parametrize(
    "NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (16, 16), (2, 1), (32, 8)]
)
@pytest.mark.parametrize("HEAD_SZ", [8, 32])
# TODO: Theres no masking for head size in the inner loop of backward kernels so we need to use pow2 and >=16
@pytest.mark.parametrize("FP8", [False])
# @pytest.mark.parametrize('FP8',[(False), (True)]) #TODO Debug FP8
def test_mha_fused_backward(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    DROPOUT: float,
    CAUSAL: bool,
    FP8: bool,
    dtype=torch.float16,
):

    torch.cuda.empty_cache()  # makes the atomic tests pass. TODO: why?
    torch.manual_seed(20)
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    do = torch.randn_like(q)
    with torch.enable_grad():
        if FP8:
            triton_out = flash_attn_fp8_func(
                q,
                k,
                v,
                dropout_p=DROPOUT,
                causal=CAUSAL,
                return_lse=True,
                return_attn_probs=True,
                fused_backward=True,
                onekernel_backward=False,
            )
        else:
            triton_out = flash_attn_func(
                q,
                k,
                v,
                dropout_p=DROPOUT,
                causal=CAUSAL,
                return_lse=True,
                return_attn_probs=True,
                fused_backward=True,
                onekernel_backward=False,
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
    with torch.enable_grad():
        torch_out = attention_ref(
            q, k, v, dropout_p=DROPOUT, dropout_mask=dropout_mask, causal=CAUSAL
        )
    torch_out, attention_scores = torch_out

    torch.testing.assert_close(
        triton_out, torch_out.to(triton_out.dtype), atol=1e-2, rtol=1e-2
    )

    torch_dq, torch_dk, torch_dv = torch.autograd.grad(torch_out, (q, k, v), do)

    if FP8:
        fp8_assert_close(
            triton_dq, torch_dq.to(triton_dq.dtype), atol=ATOL_fp8, rtol=RTOL_fp8
        )
        fp8_assert_close(
            triton_dk, torch_dk.to(triton_dk.dtype), atol=ATOL_fp8, rtol=RTOL_fp8
        )
        fp8_assert_close(
            triton_dv, torch_dv.to(triton_dv.dtype), atol=ATOL_fp8, rtol=RTOL_fp8
        )
    else:
        torch.testing.assert_close(
            triton_dv, torch_dv.to(triton_out.dtype), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            triton_dk, torch_dk.to(triton_out.dtype), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            triton_dq, torch_dq.to(triton_out.dtype), atol=1e-2, rtol=1e-2
        )


@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(256, 128), (256, 128), (555, 555)],
)
@pytest.mark.parametrize("DROPOUT, CAUSAL", [(0.0, True), (0.0, False)])
# dropout > 0.0 gives failures
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(32, 8)])
@pytest.mark.parametrize("HEAD_SZ", [8, 32])
@pytest.mark.parametrize("FP8", [False])
# @pytest.mark.parametrize('FP8',[(False), (True)]) #TODO Debug FP8
def test_mha_fused_backward_varlen(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    DROPOUT: float,
    CAUSAL: bool,
    FP8: bool,
    dtype=torch.float16,
):
    torch.cuda.empty_cache()  # makes the atomic tests pass. TODO: why?
    torch.manual_seed(20)
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

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

    q_unpad.requires_grad = True
    k_unpad.requires_grad = True
    v_unpad.requires_grad = True
    do = torch.randn_like(q)

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
            fused_backward=True,
            onekernel_backward=False,
        )

    assert len(triton_out) == 3
    triton_out, lse, sd_mask = triton_out[0], triton_out[1], triton_out[2]

    if DROPOUT > 0.0:
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
    else:
        dropout_mask = None

    triton_out = output_pad_fn(triton_out)
    triton_dq, triton_dk, triton_dv = torch.autograd.grad(
        triton_out, (q_unpad, k_unpad, v_unpad), do.clone()
    )

    triton_dq = dq_pad_fn(triton_dq)
    triton_dk = dk_pad_fn(triton_dk)
    triton_dv = dk_pad_fn(triton_dv)

    with torch.enable_grad():
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
    torch_out, attention_scores = torch_out

    torch.testing.assert_close(
        triton_out, torch_out.to(triton_out.dtype), atol=1e-2, rtol=1e-2
    )

    torch_dq, torch_dk, torch_dv = torch.autograd.grad(torch_out, (q, k, v), do)

    torch.testing.assert_close(
        triton_dv, torch_dv.to(triton_out.dtype), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        triton_dk, torch_dk.to(triton_out.dtype), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        triton_dq, torch_dq.to(triton_out.dtype), atol=1e-2, rtol=1e-2
    )
