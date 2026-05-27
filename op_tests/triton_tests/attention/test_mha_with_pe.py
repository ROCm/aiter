# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
from aiter.ops.triton.attention.mha import (
    flash_attn_func,
    flash_attn_varlen_func,
    mha_set_use_fused_bwd_kernel,
)
from aiter.test_mha_common import (
    attention_ref,
    generate_random_padding_mask,
    generate_qkv,
)
from op_tests.triton_tests.attention.mha_test_utils import pad_rearrange_dropout_mask

from aiter.ops.triton.utils._triton.arch_info import get_arch

arch = get_arch()


@pytest.mark.parametrize("BATCH", [1, 3])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(16, 48), (4096, 4096)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (64, 8)])
@pytest.mark.parametrize("HEAD_SZ_QK, HEAD_SZ_V", [(128, 64), (192, 128)])
@pytest.mark.parametrize("DROPOUT", [0.0, 0.25])
@pytest.mark.parametrize("CAUSAL", [True, False])
def test_mha_with_pe(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ_QK: int,
    HEAD_SZ_V: int,
    DROPOUT: float,
    CAUSAL: bool,
):
    HAS_DROPOUT: bool = DROPOUT > 0.0
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    # TODO: Enable these test cases once this is fixed
    if arch == "gfx942" and (CAUSAL or HAS_DROPOUT):
        pytest.skip(
            "Causal or Dropout use case isn't currently working with Positional Encoding on gfx942 archictecture."
        )

    # Generate tensors
    torch.cuda.empty_cache()
    torch.manual_seed(20)
    q = torch.randn(
        (BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ_QK), device=device, dtype=dtype
    )
    k = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ_QK), device=device, dtype=dtype
    )
    v = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ_V), device=device, dtype=dtype
    )

    # Triton
    triton_out = flash_attn_func(
        q,
        k,
        v,
        dropout_p=DROPOUT,
        causal=CAUSAL,
        return_lse=HAS_DROPOUT,
        return_attn_probs=HAS_DROPOUT,
    )
    if HAS_DROPOUT:
        assert len(triton_out) == 3
        dropout_mask = triton_out[2] > 0
        triton_out = triton_out[0]
    else:
        dropout_mask = None

    # Torch
    torch_out, _, _ = attention_ref(
        q,
        k,
        v,
        dropout_p=DROPOUT,
        dropout_mask=dropout_mask,
        causal=CAUSAL,
    )

    # Assertion
    torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(16, 1), (64, 128), (4096, 4096)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4), (8, 1)])
@pytest.mark.parametrize("HEAD_SZ_QK, HEAD_SZ_V", [(96, 64), (192, 128)])
@pytest.mark.parametrize("DROPOUT", [0.0, 0.17])
@pytest.mark.parametrize("CAUSAL", [True, False])
def test_mha_varlen_with_pe(
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ_QK: int,
    HEAD_SZ_V: int,
    DROPOUT: float,
    CAUSAL: bool,
):
    BATCH = 5
    HAS_DROPOUT: bool = DROPOUT > 0.0
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    # TODO: Enable these test cases once this is fixed
    if arch == "gfx942" and (CAUSAL or HAS_DROPOUT):
        pytest.skip(
            "Causal or Dropout use case isn't currently working with Positional Encoding on gfx942 archictecture."
        )

    # Generate tensors
    torch.cuda.empty_cache()
    torch.manual_seed(77)
    q = torch.randn(
        (BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ_QK), device=device, dtype=dtype
    )
    k = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ_QK), device=device, dtype=dtype
    )
    v = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ_V), device=device, dtype=dtype
    )
    query_padding_mask = generate_random_padding_mask(SEQLEN_Q, BATCH, device)
    key_padding_mask = generate_random_padding_mask(SEQLEN_K, BATCH, device)
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
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask)

    # Triton
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
        return_lse=HAS_DROPOUT,
        return_attn_probs=HAS_DROPOUT,
    )
    if HAS_DROPOUT:
        assert len(triton_out) == 3
        dropout_mask = (
            pad_rearrange_dropout_mask(
                triton_out[2] > 0,
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
        triton_out = triton_out[0]
    else:
        dropout_mask = None
    triton_out = output_pad_fn(triton_out)

    # Torch
    torch_out, _, _ = attention_ref(
        q,
        k,
        v,
        query_padding_mask=query_padding_mask,
        key_padding_mask=key_padding_mask,
        dropout_p=DROPOUT,
        dropout_mask=dropout_mask,
        causal=CAUSAL,
    )

    # Assertion
    torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(32, 8), (64, 16), (2048, 2048)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4), (32, 4)])
@pytest.mark.parametrize("HEAD_SZ_QK, HEAD_SZ_V", [(32, 16), (192, 128)])
@pytest.mark.parametrize("DROPOUT", [0.0, 0.2])
@pytest.mark.parametrize("CAUSAL", [True, False])
def test_mha_backward_with_pe(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ_QK: int,
    HEAD_SZ_V: int,
    DROPOUT: float,
    CAUSAL: bool,
):
    HAS_DROPOUT: bool = DROPOUT > 0.0

    # TODO: Enable these test cases once this is fixed
    if arch == "gfx942" and (CAUSAL or HAS_DROPOUT):
        pytest.skip(
            "Causal or Dropout use case isn't currently working with Positional Encoding on gfx942 archictecture."
        )

    # Causal + Dropout use case is disabled in `test_mha_backward` and `test_mha_backward_varlen`.
    # FIXME: We should fix it in the base implementation before adding PE to the mix.
    if CAUSAL and HAS_DROPOUT:
        pytest.skip(
            "Causal + Dropout use case isn't supported in backward with Positional Encoding."
        )

    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    # Generate tensors
    torch.cuda.empty_cache()
    torch.manual_seed(63)
    q = torch.randn(
        (BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ_QK),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ_QK),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ_V),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    do = torch.randn((q.shape[:-1] + v.shape[-1:]), dtype=dtype, device=device)

    # Triton forward
    with torch.enable_grad():
        triton_out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=DROPOUT,
            causal=CAUSAL,
            return_lse=HAS_DROPOUT,
            return_attn_probs=HAS_DROPOUT,
        )
    if HAS_DROPOUT:
        assert len(triton_out) == 3
        dropout_mask = triton_out[2] > 0
        triton_out = triton_out[0]
    else:
        dropout_mask = None

    # Torch forward
    with torch.enable_grad():
        torch_out, _, _ = attention_ref(
            q, k, v, dropout_p=DROPOUT, dropout_mask=dropout_mask, causal=CAUSAL
        )

    # Forward assertion
    torch.testing.assert_close(
        triton_out,
        torch_out,
        atol=1e-2,
        rtol=1e-2,
        msg=lambda msg: f"fwd mismatch\n\n{msg}\n",
    )

    # Triton backward
    # PE support isn't implemented in fused backward.
    mha_set_use_fused_bwd_kernel(False)
    triton_dq, triton_dk, triton_dv = torch.autograd.grad(triton_out, (q, k, v), do)

    # Torch backward
    torch_dq, torch_dk, torch_dv = torch.autograd.grad(torch_out, (q, k, v), do)

    # Backward assertions
    bwd_atol = 1e-1
    bwd_rtol = 1e-1
    torch.testing.assert_close(
        triton_dq,
        torch_dq,
        atol=bwd_atol,
        rtol=bwd_rtol,
        msg=lambda msg: f"bwd dq mismatch\n\n{msg}\n",
    )
    torch.testing.assert_close(
        triton_dk,
        torch_dk,
        atol=bwd_atol,
        rtol=bwd_rtol,
        msg=lambda msg: f"bwd dk mismatch\n\n{msg}\n",
    )
    torch.testing.assert_close(
        triton_dv,
        torch_dv,
        atol=bwd_atol,
        rtol=bwd_rtol,
        msg=lambda msg: f"bwd dv mismatch\n\n{msg}\n",
    )


@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(8, 8), (32, 8), (16, 64), (64, 64)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4), (48, 8)])
@pytest.mark.parametrize("HEAD_SZ_QK, HEAD_SZ_V", [(32, 16), (192, 128)])
@pytest.mark.parametrize("DROPOUT", [0.0, 0.2])
@pytest.mark.parametrize("CAUSAL", [True, False])
def test_mha_backward_varlen_with_pe(
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ_QK: int,
    HEAD_SZ_V: int,
    DROPOUT: float,
    CAUSAL: bool,
):
    BATCH = 4
    HAS_DROPOUT: bool = DROPOUT > 0.0

    # TODO: Enable these test cases once this is fixed
    if arch == "gfx942" and (CAUSAL or HAS_DROPOUT):
        pytest.skip(
            "Causal or Dropout use case isn't currently working with Positional Encoding on gfx942 archictecture."
        )

    # Causal + Dropout use case is disabled in `test_mha_backward` and `test_mha_backward_varlen`.
    # FIXME: We should fix it in the base implementation before adding PE to the mix.
    if CAUSAL and HAS_DROPOUT:
        pytest.skip(
            "Causal + Dropout use case isn't supported in backward with Positional Encoding."
        )

    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    # Generate tensors
    torch.cuda.empty_cache()
    torch.manual_seed(133)
    q = torch.randn(
        (BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ_QK),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ_QK),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ_V),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    query_padding_mask = generate_random_padding_mask(SEQLEN_Q, BATCH, device)
    key_padding_mask = generate_random_padding_mask(SEQLEN_K, BATCH, device)
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
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask)
    q_unpad.requires_grad = True
    k_unpad.requires_grad = True
    v_unpad.requires_grad = True
    do = torch.randn((q.shape[:-1] + v.shape[-1:]), dtype=dtype, device=device)

    # Triton forward
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
            return_lse=HAS_DROPOUT,
            return_attn_probs=HAS_DROPOUT,
        )
    if HAS_DROPOUT:
        assert len(triton_out) == 3
        dropout_mask = (
            pad_rearrange_dropout_mask(
                triton_out[2] > 0,
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
        triton_out = triton_out[0]
    else:
        dropout_mask = None
    triton_out = output_pad_fn(triton_out)

    # Torch forward
    with torch.enable_grad():
        torch_out, _, _ = attention_ref(
            q,
            k,
            v,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            dropout_p=DROPOUT,
            dropout_mask=dropout_mask,
            causal=CAUSAL,
        )

    # Forward assertion
    torch.testing.assert_close(
        triton_out,
        torch_out,
        atol=1e-2,
        rtol=1e-2,
        msg=lambda msg: f"fwd mismatch\n\n{msg}\n",
    )

    # Triton backward
    # PE support isn't implemented in fused backward.
    mha_set_use_fused_bwd_kernel(False)
    triton_dq, triton_dk, triton_dv = torch.autograd.grad(
        triton_out, (q_unpad, k_unpad, v_unpad), do
    )
    triton_dq = dq_pad_fn(triton_dq)
    triton_dk = dk_pad_fn(triton_dk)
    triton_dv = dk_pad_fn(triton_dv)

    # Torch backward
    torch_dq, torch_dk, torch_dv = torch.autograd.grad(torch_out, (q, k, v), do)

    # Backward assertions
    bwd_atol = 1e-1
    bwd_rtol = 1e-1
    torch.testing.assert_close(
        triton_dq,
        torch_dq,
        atol=bwd_atol,
        rtol=bwd_rtol,
        msg=lambda msg: f"bwd dq mismatch\n\n{msg}\n",
    )
    torch.testing.assert_close(
        triton_dk,
        torch_dk,
        atol=bwd_atol,
        rtol=bwd_rtol,
        msg=lambda msg: f"bwd dk mismatch\n\n{msg}\n",
    )
    torch.testing.assert_close(
        triton_dv,
        torch_dv,
        atol=bwd_atol,
        rtol=bwd_rtol,
        msg=lambda msg: f"bwd dv mismatch\n\n{msg}\n",
    )
