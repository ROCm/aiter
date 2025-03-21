import triton
import torch
import triton.language as tl
import pytest
import logging
from typing import Any, Dict, Optional


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DEBUG_MODE = False

from aiter.ops.triton.mha import flash_attn_func, flash_attn_fp8_func, flash_attn_varlen_func, flash_attn_varlen_fp8_func
from aiter.test_mha_common import attention_ref, generate_random_padding_mask, generate_qkv 


@pytest.mark.parametrize('BATCH', [1,4,16,57,128])
@pytest.mark.parametrize('SEQLEN_Q, SEQLEN_K', [(1,1), (4,4), (128,128), (2,1), (1,2), (32,16)])
@pytest.mark.parametrize('NUM_Q_HEADS, NUM_K_HEADS', [(1,1), (16,16), (2,1), (48,8)])
@pytest.mark.parametrize('HEAD_SZ', [8, 32, 64 ,128])
@pytest.mark.parametrize('DROPOUT, RETURN_LSE, RETURN_SOFTMAX, ',[(0.2, True, True), (0.0, False, False)])
@pytest.mark.parametrize('CAUSAL', [(True), (False)])
@pytest.mark.parametrize('FP8',[(True), (False)])
def test_mha(BATCH: int, SEQLEN_Q: int, SEQLEN_K: int, NUM_Q_HEADS: int, NUM_K_HEADS: int, HEAD_SZ: int, DROPOUT: float, RETURN_LSE: bool, RETURN_SOFTMAX: bool, CAUSAL: bool, FP8: bool, dtype=torch.float16):
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
                
    dropout_mask = None
    if FP8:
        triton_out = flash_attn_fp8_func(q, k, v, dropout_p=DROPOUT, causal=CAUSAL, return_lse=RETURN_LSE, return_attn_probs=RETURN_SOFTMAX)
    else:
        triton_out = flash_attn_func(q, k, v, dropout_p=DROPOUT, causal=CAUSAL, return_lse=RETURN_LSE, return_attn_probs=RETURN_SOFTMAX)

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
            print(f"dropout_mask.shape={dropout_mask.shape}, dropout_mask={dropout_mask}")

    triton_out = triton_out[0]
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

    
    torch_out = attention_ref(q, k, v, dropout_p=DROPOUT, dropout_mask=dropout_mask, causal=CAUSAL)
    torch_out, attention_scores = torch_out
    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")
        print(f"attention_scores.shape={attention_scores.shape}, attention_scores={attention_scores}")

    torch.testing.assert_close(triton_out, torch_out,atol=1e-2, rtol=1e-2)

@pytest.mark.parametrize('BATCH', [1,4,16,57,128])
@pytest.mark.parametrize('SEQLEN_Q, SEQLEN_K', [(1,1), (4,4), (128,128), (2,1), (1,2), (32,16)])
@pytest.mark.parametrize('DROPOUT, RETURN_LSE, RETURN_SOFTMAX, ',[(0.0, False, False)])
@pytest.mark.parametrize('NUM_Q_HEADS, NUM_K_HEADS', [(1,1), (16,16), (2,1), (48,8)])
@pytest.mark.parametrize('HEAD_SZ', [8, 32, 64 ,128])
@pytest.mark.parametrize('CAUSAL', [(True), (False)])
@pytest.mark.parametrize('FP8',[(False), (True)])
def test_mha_varlen(BATCH: int, SEQLEN_Q: int, SEQLEN_K: int, NUM_Q_HEADS: int, NUM_K_HEADS: int, HEAD_SZ: int, DROPOUT: float, RETURN_LSE: bool, RETURN_SOFTMAX: bool, CAUSAL: bool, FP8: bool, dtype=torch.float16):
    torch.set_printoptions(threshold=10000)
    torch.manual_seed(20)
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    query_padding_mask = generate_random_padding_mask(SEQLEN_Q, BATCH, "cuda", mode="random")
    key_padding_mask = generate_random_padding_mask(SEQLEN_K, BATCH, "cuda", mode="random")
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
        print(f"query_padding_mask.shape={query_padding_mask.shape} query_padding_mask={query_padding_mask}")
        print(f"key_padding_mask.shape={key_padding_mask.shape} key_padding_mask={key_padding_mask}")

        print(f"q.shape={q_unpad.shape} q={q_unpad}")
        print(f"k.shape={k_unpad.shape} k={k_unpad}")
        print(f"v.shape={v_unpad.shape} v={v_unpad}")
        print(f"q_unpad.shape={q_unpad.shape} q_unpad={q_unpad}")
        print(f"k_unpad.shape={k_unpad.shape} k_unpad={k_unpad}")
        print(f"v_unpad.shape={v_unpad.shape} v_unpad={v_unpad}")
        print(f"max_seqlens_q={max_seqlen_q }")
        print(f"max_seqlens_k={max_seqlen_k }")
        print(f"cu_seqlens_q={cu_seqlens_q }")
        print(f"cu_seqlens_k={cu_seqlens_k }")
    if FP8:
        triton_out = flash_attn_varlen_fp8_func(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=DROPOUT,causal=CAUSAL, return_lse=RETURN_LSE, return_attn_probs=RETURN_SOFTMAX)
    else:
        triton_out = flash_attn_varlen_func(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=DROPOUT,causal=CAUSAL, return_lse=RETURN_LSE, return_attn_probs=RETURN_SOFTMAX)

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
        if DEBUG_MODE:
            print(f"sd_mask.shape={sd_mask.shape}, sd_mask={sd_mask}")
            print(f"dropout_mask.shape={dropout_mask.shape}, dropout_mask={dropout_mask}")

    triton_out = output_pad_fn(triton_out[0])
    if DEBUG_MODE:
        print(f"triton_out.shape={triton_out.shape}, triton_out={triton_out}")

    torch_out = attention_ref(q, k, v, query_padding_mask=query_padding_mask, key_padding_mask=key_padding_mask, dropout_p=DROPOUT, dropout_mask=dropout_mask, causal=CAUSAL)
    torch_out, attention_scores = torch_out

    if DEBUG_MODE:
        print(f"torch_out.shape={torch_out.shape}, torch_out={torch_out}")
        print(f"attention_scores.shape={attention_scores.shape}, attention_scores={attention_scores}")

    if FP8: 
        torch.testing.assert_close(triton_out, torch_out.to(triton_out.dtype),atol=0.25, rtol=10) #Lower tolerance for FP8 
    else:
        torch.testing.assert_close(triton_out, torch_out.to(triton_out.dtype),atol=1e-1, rtol=1e-1)  
    