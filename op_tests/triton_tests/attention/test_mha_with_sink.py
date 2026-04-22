# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest

from aiter.ops.triton.attention.mha import (
    flash_attn_func,
    flash_attn_varlen_func,
    mha_set_use_fused_bwd_kernel,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.test_mha_common import (
    attention_ref,
    generate_qkv,
    generate_random_padding_mask,
)
from op_tests.triton_tests.attention.mha_test_utils import pad_rearrange_dropout_mask

arch = get_arch()


# Run sink and sliding window tests with:
# pytest op_tests/triton_tests/attention/test_mha_with_sink.py -q


@pytest.mark.parametrize("BATCH", [1, 3])
@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", [(128, 64), (32, 128), (1024, 1024)])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(64, 8), (8, 1)])
@pytest.mark.parametrize("HEAD_SZ", [64, 128])
@pytest.mark.parametrize("DROPOUT", [0.0, 0.2])
@pytest.mark.parametrize("CAUSAL", [False, True])
def test_mha_with_sink(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    DROPOUT: float,
    CAUSAL: bool,
):
    HAS_DROPOUT: bool = DROPOUT > 0.0
    # Keep sink coverage aligned with the baseline MHA tests.
    # Causal + dropout backward is still disabled in `test_mha_backward`.
    TEST_BWD: bool = not (CAUSAL and HAS_DROPOUT)
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    torch.cuda.empty_cache()
    torch.manual_seed(0)
    q = torch.randn(
        (BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ),
        device=device,
        dtype=dtype,
        requires_grad=TEST_BWD,
    )
    k = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ),
        device=device,
        dtype=dtype,
        requires_grad=TEST_BWD,
    )
    v = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ),
        device=device,
        dtype=dtype,
        requires_grad=TEST_BWD,
    )
    sink = torch.randn(
        (NUM_Q_HEADS,), device=device, dtype=torch.float32, requires_grad=TEST_BWD
    )

    with torch.set_grad_enabled(TEST_BWD):
        triton_out = flash_attn_func(
            q,
            k,
            v,
            dropout_p=DROPOUT,
            causal=CAUSAL,
            return_lse=HAS_DROPOUT,
            return_attn_probs=HAS_DROPOUT,
            sink=sink,
        )
    if HAS_DROPOUT:
        assert len(triton_out) == 3
        dropout_mask = triton_out[2] > 0
        triton_out = triton_out[0]
    else:
        dropout_mask = None

    with torch.set_grad_enabled(TEST_BWD):
        torch_out, _, _ = attention_ref(
            q,
            k,
            v,
            dropout_p=DROPOUT,
            dropout_mask=dropout_mask,
            causal=CAUSAL,
            sink=sink,
        )

    fwd_atol: float = 1e-2
    fwd_rtol: float = 1e-2
    torch.testing.assert_close(
        triton_out,
        torch_out,
        atol=fwd_atol,
        rtol=fwd_rtol,
        msg=lambda msg: f"fwd mismatch\n\n{msg}\n",
    )

    if not TEST_BWD:
        return

    do = torch.randn_like(q)

    mha_set_use_fused_bwd_kernel(False)
    triton_dq, triton_dk, triton_dv, triton_dsink = torch.autograd.grad(
        triton_out, (q, k, v, sink), do
    )

    torch_dq, torch_dk, torch_dv, torch_dsink = torch.autograd.grad(
        torch_out, (q, k, v, sink), do
    )

    relax_bwd_err_tol: bool = SEQLEN_Q >= 1024 and SEQLEN_K >= 1024
    bwd_atol = 2.5e-2 if relax_bwd_err_tol else 1.5e-2
    bwd_rtol = 2.5e-2 if relax_bwd_err_tol else 1.5e-2
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
    torch.testing.assert_close(
        triton_dsink,
        torch_dsink,
        atol=5e-2,
        rtol=5e-2,
        msg=lambda msg: f"bwd dsink mismatch\n\n{msg}\n",
    )


@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", [(16, 32), (128, 64), (256, 256)])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4), (8, 1)])
@pytest.mark.parametrize("HEAD_SZ", [64, 128])
@pytest.mark.parametrize("DROPOUT", [0.0, 0.2])
@pytest.mark.parametrize("CAUSAL", [False, True])
def test_mha_varlen_with_sink(
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    DROPOUT: float,
    CAUSAL: bool,
):
    BATCH = 2
    HAS_DROPOUT: bool = DROPOUT > 0.0
    # Keep sink coverage aligned with the baseline MHA tests.
    # Dropout backward is still disabled in `test_mha_backward_varlen`.
    TEST_BWD: bool = not HAS_DROPOUT
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    torch.cuda.empty_cache()
    torch.manual_seed(0)
    q = torch.randn(
        (BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ),
        device=device,
        dtype=dtype,
        requires_grad=TEST_BWD,
    )
    k = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ),
        device=device,
        dtype=dtype,
        requires_grad=TEST_BWD,
    )
    v = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ),
        device=device,
        dtype=dtype,
        requires_grad=TEST_BWD,
    )
    sink = torch.randn(
        (NUM_Q_HEADS,), device=device, dtype=torch.float32, requires_grad=TEST_BWD
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
    q_unpad.requires_grad = TEST_BWD
    k_unpad.requires_grad = TEST_BWD
    v_unpad.requires_grad = TEST_BWD

    with torch.set_grad_enabled(TEST_BWD):
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
            sink=sink,
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

    with torch.set_grad_enabled(TEST_BWD):
        torch_out, _, _ = attention_ref(
            q,
            k,
            v,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            dropout_p=DROPOUT,
            dropout_mask=dropout_mask,
            causal=CAUSAL,
            sink=sink,
        )

    fwd_atol: float = 1e-2
    fwd_rtol: float = 1e-2
    torch.testing.assert_close(
        triton_out,
        torch_out,
        atol=fwd_atol,
        rtol=fwd_rtol,
        msg=lambda msg: f"fwd mismatch\n\n{msg}\n",
    )

    if not TEST_BWD:
        return

    do = torch.randn_like(q)

    mha_set_use_fused_bwd_kernel(False)
    triton_dq, triton_dk, triton_dv, triton_dsink = torch.autograd.grad(
        triton_out, (q_unpad, k_unpad, v_unpad, sink), do
    )
    triton_dq = dq_pad_fn(triton_dq)
    triton_dk = dk_pad_fn(triton_dk)
    triton_dv = dk_pad_fn(triton_dv)

    torch_dq, torch_dk, torch_dv, torch_dsink = torch.autograd.grad(
        torch_out, (q, k, v, sink), do
    )

    bwd_atol = 1.5e-2
    bwd_rtol = 1.5e-2
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
    torch.testing.assert_close(
        triton_dsink,
        torch_dsink,
        atol=5e-2,
        rtol=5e-2,
        msg=lambda msg: f"bwd dsink mismatch\n\n{msg}\n",
    )


@pytest.mark.parametrize("BATCH", [1, 2])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K", [(64, 64), (256, 256), (128, 64), (32, 128)]
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(8, 1), (16, 4)])
@pytest.mark.parametrize("HEAD_SZ", [64, 128])
@pytest.mark.parametrize("CAUSAL", [True])
@pytest.mark.parametrize("WINDOW_SIZE_LEFT", [4, 64])
def test_mha_with_sink_sliding_window(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    WINDOW_SIZE_LEFT: int,
):
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    torch.cuda.empty_cache()
    torch.manual_seed(0)
    q = torch.randn(
        (BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    sink = torch.randn(
        (NUM_Q_HEADS,), device=device, dtype=torch.float32, requires_grad=True
    )

    window_size = (WINDOW_SIZE_LEFT, -1)

    with torch.set_grad_enabled(True):
        triton_out = flash_attn_func(
            q,
            k,
            v,
            causal=CAUSAL,
            window_size=window_size,
            sink=sink,
        )

    with torch.set_grad_enabled(True):
        torch_out, _, _ = attention_ref(
            q,
            k,
            v,
            causal=CAUSAL,
            window_size=window_size,
            sink=sink,
        )

    fwd_atol: float = 1e-2
    fwd_rtol: float = 1e-2
    torch.testing.assert_close(
        triton_out,
        torch_out,
        atol=fwd_atol,
        rtol=fwd_rtol,
        msg=lambda msg: f"fwd mismatch\n\n{msg}\n",
    )

    do = torch.randn_like(q)

    mha_set_use_fused_bwd_kernel(False)
    triton_dq, triton_dk, triton_dv, triton_dsink = torch.autograd.grad(
        triton_out, (q, k, v, sink), do
    )

    torch_dq, torch_dk, torch_dv, torch_dsink = torch.autograd.grad(
        torch_out, (q, k, v, sink), do
    )

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
    torch.testing.assert_close(
        triton_dsink,
        torch_dsink,
        atol=5e-2,
        rtol=5e-2,
        msg=lambda msg: f"bwd dsink mismatch\n\n{msg}\n",
    )


@pytest.mark.parametrize("SEQLEN_Q, SEQLEN_K", [(128, 128), (64, 64)])
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(8, 1)])
@pytest.mark.parametrize("HEAD_SZ", [64])
@pytest.mark.parametrize("CAUSAL", [True])
@pytest.mark.parametrize("WINDOW_SIZE_LEFT", [4, 32])
def test_mha_sliding_window_no_sink(
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    WINDOW_SIZE_LEFT: int,
):
    BATCH = 1
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16

    torch.cuda.empty_cache()
    torch.manual_seed(0)
    q = torch.randn(
        (BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        (BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ),
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    window_size = (WINDOW_SIZE_LEFT, -1)

    with torch.set_grad_enabled(True):
        triton_out = flash_attn_func(
            q,
            k,
            v,
            causal=CAUSAL,
            window_size=window_size,
        )

    with torch.set_grad_enabled(True):
        torch_out, _, _ = attention_ref(
            q,
            k,
            v,
            causal=CAUSAL,
            window_size=window_size,
        )

    fwd_atol: float = 1e-2
    fwd_rtol: float = 1e-2
    torch.testing.assert_close(
        triton_out,
        torch_out,
        atol=fwd_atol,
        rtol=fwd_rtol,
        msg=lambda msg: f"fwd mismatch\n\n{msg}\n",
    )

    do = torch.randn_like(q)

    mha_set_use_fused_bwd_kernel(False)
    triton_dq, triton_dk, triton_dv = torch.autograd.grad(triton_out, (q, k, v), do)

    torch_dq, torch_dk, torch_dv = torch.autograd.grad(torch_out, (q, k, v), do)

    bwd_atol = 1.5e-2
    bwd_rtol = 1.5e-2
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


def test_mha_sliding_window_fused_backward_raises():
    q = torch.randn(
        (1, 128, 8, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn(
        (1, 128, 1, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn(
        (1, 128, 1, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    do = torch.randn_like(q)

    mha_set_use_fused_bwd_kernel(True)
    try:
        out = flash_attn_func(q, k, v, causal=True, window_size=(32, -1))
        with pytest.raises(
            ValueError, match="Fused backward doesn't support sliding window attention"
        ):
            torch.autograd.grad(out, (q, k, v), do)
    finally:
        mha_set_use_fused_bwd_kernel(False)


def test_mha_varlen_sliding_window_fused_backward_raises():
    q = torch.randn(
        (1, 64, 8, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    k = torch.randn(
        (1, 64, 1, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    v = torch.randn(
        (1, 64, 1, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = generate_qkv(
        q,
        k,
        v,
        generate_random_padding_mask(64, 1, "cuda", mode="full"),
        generate_random_padding_mask(64, 1, "cuda", mode="full"),
    )
    q_unpad.requires_grad = True
    k_unpad.requires_grad = True
    v_unpad.requires_grad = True
    do = torch.randn_like(q_unpad)

    mha_set_use_fused_bwd_kernel(True)
    try:
        out = flash_attn_varlen_func(
            q_unpad,
            k_unpad,
            v_unpad,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=True,
            window_size=(32, -1),
        )
        with pytest.raises(
            ValueError, match="Fused backward doesn't support sliding window attention"
        ):
            torch.autograd.grad(out, (q_unpad, k_unpad, v_unpad), do)
    finally:
        mha_set_use_fused_bwd_kernel(False)
