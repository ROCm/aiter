# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import pytest
import logging
from aiter.ops.triton.attention.mha import (
    flash_attn_func,
    flash_attn_varlen_func,
    mha_set_use_fused_bwd_kernel,
    mha_set_use_int64_strides,
)
from aiter.test_mha_common import (
    attention_ref,
    generate_random_padding_mask,
    generate_qkv,
)
from op_tests.triton_tests.attention.mha_test_utils import pad_rearrange_dropout_mask

from aiter.ops.triton.utils._triton.arch_info import get_arch

arch = get_arch()

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DEBUG_MODE = False


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
    dtype=torch.float16,
):
    torch.cuda.empty_cache()
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)

    dropout_mask = None
    triton_out = flash_attn_func(
        q,
        k,
        v,
        dropout_p=DROPOUT,
        causal=CAUSAL,
        return_lse=RETURN_LSE,
        return_attn_probs=RETURN_SOFTMAX,
    )

    if RETURN_LSE:
        assert len(triton_out) > 1
        lse = triton_out[1]
        if DEBUG_MODE:
            print(f"lse.shape={lse.shape}, lse={lse}")

    if DROPOUT > 0.0 and RETURN_SOFTMAX:
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

    if RETURN_SOFTMAX or RETURN_LSE:
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

    if RETURN_LSE:
        assert len(triton_out) > 1
        lse = triton_out[1]
        if DEBUG_MODE:
            print(f"lse.shape={lse.shape}, lse={lse}")

    dropout_mask = None
    if DROPOUT > 0.0 and RETURN_SOFTMAX:
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
    if RETURN_SOFTMAX or RETURN_LSE:
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

    torch.testing.assert_close(
        triton_out, torch_out.to(triton_out.dtype), atol=1e-1, rtol=1e-1
    )


# Production shapes based on real models:
#   HQ=32, HK=8:  Llama 3 8B (GQA 4:1)
#   HQ=64, HK=8:  Llama 3 70B (GQA 8:1)
#   HQ=32, HK=32: Llama 2 7B (MHA)
@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize("SEQLEN_Q", [512, 1024, 2048])
@pytest.mark.parametrize("SEQLEN_K", [512, 1024, 2048])
@pytest.mark.parametrize("NUM_Q_HEADS", [32, 64])
@pytest.mark.parametrize("NUM_K_HEADS", [8])
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("CAUSAL", [True, False])
@pytest.mark.parametrize("DROPOUT", [0.0, 0.2])
@pytest.mark.parametrize("FUSED", [False, True])
def test_mha_backward(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    DROPOUT: float,
    FUSED: bool,
    dtype=torch.float16,
):
    HAS_DROPOUT = DROPOUT > 0.0
    torch.cuda.empty_cache()
    torch.manual_seed(20)

    if FUSED and CAUSAL:
        pytest.skip("FUSED+CAUSAL results in NaNs")
    if CAUSAL and HAS_DROPOUT:
        pytest.skip("CAUSAL+DROPOUT backward results in NaNs")

    mha_set_use_fused_bwd_kernel(FUSED)
    q = torch.randn(BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    k = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    v = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True
    do = torch.randn_like(q)

    # Triton forward + backward
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
            dropout_mask = triton_out[2] >= 0
            triton_out = triton_out[0]
        else:
            dropout_mask = None
    triton_dq, triton_dk, triton_dv = torch.autograd.grad(
        triton_out, (q, k, v), do.clone()
    )

    # Reference forward + backward
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        torch_out, _, _ = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            dropout_p=DROPOUT,
            dropout_mask=dropout_mask,
            causal=CAUSAL,
        )
    torch_dq, torch_dk, torch_dv = torch.autograd.grad(
        torch_out, (q_ref, k_ref, v_ref), do.clone()
    )

    # Check quality
    bwd_atol = 1e-1 if HAS_DROPOUT else 1.5e-2
    bwd_rtol = 1e-1 if HAS_DROPOUT else 1e-5
    torch.testing.assert_close(triton_out, torch_out, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(triton_dq, torch_dq, atol=bwd_atol, rtol=bwd_rtol)
    torch.testing.assert_close(triton_dk, torch_dk, atol=bwd_atol, rtol=bwd_rtol)
    torch.testing.assert_close(triton_dv, torch_dv, atol=bwd_atol, rtol=bwd_rtol)


@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize("SEQLEN_Q", [512, 1024, 2048])
@pytest.mark.parametrize("SEQLEN_K", [512, 1024, 2048])
@pytest.mark.parametrize("NUM_Q_HEADS", [32, 64])
@pytest.mark.parametrize("NUM_K_HEADS", [8])
@pytest.mark.parametrize("HEAD_SZ", [128])
@pytest.mark.parametrize("CAUSAL", [True, False])
@pytest.mark.parametrize("DROPOUT", [0.0, 0.2])
@pytest.mark.parametrize("FUSED", [False, True])
def test_mha_backward_varlen(
    BATCH: int,
    SEQLEN_Q: int,
    SEQLEN_K: int,
    NUM_Q_HEADS: int,
    NUM_K_HEADS: int,
    HEAD_SZ: int,
    CAUSAL: bool,
    DROPOUT: float,
    FUSED: bool,
    dtype=torch.float16,
):
    HAS_DROPOUT = DROPOUT > 0.0
    torch.cuda.empty_cache()
    torch.manual_seed(20)

    if FUSED and CAUSAL:
        pytest.skip("FUSED+CAUSAL results in NaNs")
    if CAUSAL and HAS_DROPOUT:
        pytest.skip("CAUSAL+DROPOUT backward results in NaNs")

    mha_set_use_fused_bwd_kernel(FUSED)
    q = torch.randn(BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    k = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
    v = torch.randn(BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ, device="cuda", dtype=dtype)
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

    # Triton varlen forward + backward
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
            dropout_mask = (
                pad_rearrange_dropout_mask(
                    triton_out[2] >= 0,
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
    triton_dq, triton_dk, triton_dv = torch.autograd.grad(
        triton_out, (q_unpad, k_unpad, v_unpad), do.clone()
    )
    triton_dq = dq_pad_fn(triton_dq)
    triton_dk = dk_pad_fn(triton_dk)
    triton_dv = dk_pad_fn(triton_dv)

    # Reference forward + backward
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        torch_out, _, _ = attention_ref(
            q_ref,
            k_ref,
            v_ref,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            dropout_p=DROPOUT,
            dropout_mask=dropout_mask,
            causal=CAUSAL,
        )
    torch_dq, torch_dk, torch_dv = torch.autograd.grad(
        torch_out, (q_ref, k_ref, v_ref), do.clone()
    )

    # Check quality
    bwd_atol = 1e-1 if HAS_DROPOUT else 1.5e-2
    bwd_rtol = 1e-1 if HAS_DROPOUT else 1e-5
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)
    torch.testing.assert_close(triton_dq, torch_dq, atol=bwd_atol, rtol=bwd_rtol)
    torch.testing.assert_close(triton_dk, torch_dk, atol=bwd_atol, rtol=bwd_rtol)
    torch.testing.assert_close(triton_dv, torch_dv, atol=bwd_atol, rtol=bwd_rtol)


# Run PE tests with:
# pytest op_tests/triton_tests/attention/test_mha.py -k with_pe


@pytest.mark.parametrize("BATCH", [1, 3])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(128, 128), (32, 16), (16, 48), (4096, 4096)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(1, 1), (2, 1), (128, 128)])
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


@pytest.mark.parametrize("BATCH", [1, 3])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(16, 16), (32, 16), (64, 128), (4096, 4096)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4), (16, 4), (128, 128)])
@pytest.mark.parametrize("HEAD_SZ_QK, HEAD_SZ_V", [(96, 64), (192, 128)])
@pytest.mark.parametrize("DROPOUT", [0.0, 0.17])
@pytest.mark.parametrize("CAUSAL", [True, False])
def test_mha_varlen_with_pe(
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
    [(16, 16), (32, 8), (64, 16), (2048, 2048)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4), (8, 2), (128, 128)])
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
    # When dropout is active, some cases fail due to less than 1% mismatched elements.
    bwd_atol = 1e-1 if HAS_DROPOUT else 1.5e-2
    bwd_rtol = 1e-1 if HAS_DROPOUT else 1.5e-2
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


@pytest.mark.parametrize("BATCH", [1, 4])
@pytest.mark.parametrize(
    "SEQLEN_Q, SEQLEN_K",
    [(8, 8), (32, 8), (16, 64), (64, 64)],
)
@pytest.mark.parametrize("NUM_Q_HEADS, NUM_K_HEADS", [(4, 4), (8, 2), (128, 128)])
@pytest.mark.parametrize("HEAD_SZ_QK, HEAD_SZ_V", [(32, 16), (192, 128)])
@pytest.mark.parametrize("DROPOUT", [0.0, 0.2])
@pytest.mark.parametrize("CAUSAL", [True, False])
def test_mha_backward_varlen_with_pe(
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
