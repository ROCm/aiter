from types import SimpleNamespace

import pytest
import torch

from aiter.ops.triton.attention.mha import gluon_forward_unsupported_reason
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
from aiter.test_mha_common import (
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
    quantize_fp8_per_bh,
)


def skip_if_gluon_unsupported(backend: str, **feature_flags):
    """
    Skip forward tests the Gluon backend can't run.
    """
    if backend != "gluon":
        return
    reason = gluon_forward_unsupported_reason(**feature_flags)
    if reason:
        pytest.skip(reason)


def skip_if_triton_padded_head_miscompiled(
    backend: str, HEAD_SZ: int, CAUSAL: bool, SEQLEN_K: int, NUM_K_HEADS: int
):
    """
    Skip Triton forward tests miscompiled by the pinned ROCm Triton build.

    Remove this function once the Triton compiler pinned by AITER CI is updated.
    """
    if (
        backend == "triton"
        and HEAD_SZ == 33
        and CAUSAL
        and SEQLEN_K >= 2048
        and NUM_K_HEADS == 1
        and arch_info.get_arch() == "gfx950"
    ):
        pytest.skip(
            "gfx950: padded head size 33 is miscompiled by the pinned ROCm "
            "Triton build for causal long-sequence single-KV-head cases; "
            "passes on triton-lang/triton@2074a1b"
        )


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
        (batch_size, num_q_heads, seqlen_q, seqlen_k),
        device="cuda",
        dtype=torch.float32,
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


def make_varlen_fp8_inputs(
    batch: int,
    seqlen_q: int,
    seqlen_k: int,
    num_q_heads: int,
    num_k_heads: int,
    head_sz: int,
    dtype=torch.bfloat16,
):
    """Random padded varlen q/k/v, plus their per-(batch, head) fp8 quantization."""
    torch.manual_seed(20)
    torch.cuda.empty_cache()
    q = torch.randn((batch, seqlen_q, num_q_heads, head_sz), device="cuda", dtype=dtype)
    k = torch.randn((batch, seqlen_k, num_k_heads, head_sz), device="cuda", dtype=dtype)
    v = torch.randn((batch, seqlen_k, num_k_heads, head_sz), device="cuda", dtype=dtype)
    query_padding_mask = generate_random_padding_mask(
        seqlen_q, batch, "cuda", mode="random"
    )
    key_padding_mask = generate_random_padding_mask(
        seqlen_k, batch, "cuda", mode="random"
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
        _,
        _,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)

    # generate_qkv returns grad-tracking tensors; detach so the quantization below
    # doesn't build an autograd graph for this forward-only kernel.
    fp8_dtype = get_fp8_e4m3_dtype()
    q_fp8, q_descale = quantize_fp8_per_bh(q_unpad.detach(), fp8_dtype, cu_seqlens_q)
    k_fp8, k_descale = quantize_fp8_per_bh(k_unpad.detach(), fp8_dtype, cu_seqlens_k)
    v_fp8, v_descale = quantize_fp8_per_bh(v_unpad.detach(), fp8_dtype, cu_seqlens_k)
    return SimpleNamespace(
        q=q,
        k=k,
        v=v,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        q_fp8=q_fp8,
        k_fp8=k_fp8,
        v_fp8=v_fp8,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        output_pad_fn=output_pad_fn,
    )


def varlen_fp8_reference(inputs, causal: bool):
    out, _, _ = attention_ref(
        inputs.q,
        inputs.k,
        inputs.v,
        query_padding_mask=inputs.query_padding_mask,
        key_padding_mask=inputs.key_padding_mask,
        causal=causal,
    )
    return out
