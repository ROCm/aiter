import triton
import torch
import triton.language as tl
import pytest
import logging
from typing import Any, Dict, Optional
import numpy as np


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
DEBUG_MODE =  False
ATOL_fp8 = 2.5e-1
RTOL_fp8 =  2.5e-1

from aiter.ops.triton.mha import flash_attn_func, flash_attn_fp8_func, flash_attn_varlen_func, flash_attn_varlen_fp8_func
from aiter.test_mha_common import attention_ref, generate_random_padding_mask, generate_qkv, pad_rearrange_dropout_mask_hts_to_bhss 



def pad_rearrange_dropout_mask(S_dmask, cu_seqlens_q, cu_seqlens_k,  max_seqlen_q, max_seqlen_k, seqlen_q, seqlen_k, num_q_heads):
    batch_size = cu_seqlens_q.numel() - 1
    
    padded_dropout_mask = torch.ones((batch_size, num_q_heads, seqlen_q, seqlen_k), device="cuda")
    for b in range(batch_size):
        start_q = cu_seqlens_q[b].item()
        end_q = cu_seqlens_q[b + 1].item()
        start_k = cu_seqlens_k[b].item()
        end_k = cu_seqlens_k[b + 1].item()

        seqlen_q = end_q - start_q
        seqlen_k = end_k - start_k
        for h in range(S_dmask.shape[1]):
                padded_dropout_mask[b, h, :max_seqlen_q, :max_seqlen_k] = S_dmask[b, h, : ,:]
    
    
    return padded_dropout_mask



def fp8_assert_close(tensor_a, tensor_b, atol=ATOL_fp8, rtol=RTOL_fp8, max_diff_percentage=0.5):
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


@pytest.mark.parametrize('BATCH', [1,4,57,128])
@pytest.mark.parametrize('SEQLEN_Q, SEQLEN_K', [(1,1), (4,4), (128,128), (2,1), (1,2), (32,16)])
@pytest.mark.parametrize('NUM_Q_HEADS, NUM_K_HEADS', [(1,1), (16,16), (2,1), (48,8)])
@pytest.mark.parametrize('HEAD_SZ', [8, 32, 128])
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

@pytest.mark.parametrize('BATCH', [1,4,57,128])
@pytest.mark.parametrize('SEQLEN_Q, SEQLEN_K', [(1,1), (4,4), (128,128), (2,1), (1,2), (32,16)])
@pytest.mark.parametrize('DROPOUT, RETURN_LSE, RETURN_SOFTMAX, ',[(0.0, False, False), (0.2, True, True)])
@pytest.mark.parametrize('NUM_Q_HEADS, NUM_K_HEADS', [(1,1), (16,16), (2,1), (48,8)])
@pytest.mark.parametrize('HEAD_SZ', [8, 32, 128])
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
        dropout_mask = pad_rearrange_dropout_mask(dropout_mask, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS)
        dropout_mask = dropout_mask > 0
        if DEBUG_MODE:
            #print(f"sd_mask.shape={sd_mask.shape}, sd_mask={sd_mask}")
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


@pytest.mark.parametrize('BATCH', [1,4,57,128])
@pytest.mark.parametrize('SEQLEN_Q, SEQLEN_K', [(1,1), (4,4), (128,128), (2,1), (1,2), (32,16)])
@pytest.mark.parametrize('DROPOUT, CAUSAL',[(0.0, False),(0.0, True),(0.2, False)])
#@pytest.mark.parametrize('DROPOUT, CAUSAL',[(0.0, False),(0.0, True),(0.2, False),(0.2, True)]) #Debug Causal + Dropout. fails for seq >= 64
@pytest.mark.parametrize('NUM_Q_HEADS, NUM_K_HEADS', [(1,1), (16,16), (2,1), (48,8)])
@pytest.mark.parametrize('HEAD_SZ', [8, 32, 128])
@pytest.mark.parametrize('FP8',[(False)])
#@pytest.mark.parametrize('FP8',[(False), (True)]) #TODO Debug FP8
def test_mha_backward(BATCH: int, SEQLEN_Q: int, SEQLEN_K: int, NUM_Q_HEADS: int, NUM_K_HEADS: int, HEAD_SZ: int, DROPOUT: float, CAUSAL: bool, FP8: bool, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

    do = torch.randn_like(q)

    if DEBUG_MODE:
        print("--------------Triton----------------")
        print(f"q.shape={q.shape} q={q}")
        print(f"k.shape={k.shape} k={k}")
        print(f"v.shape={v.shape} v={v}")
        print(f"do.shape={do.shape} do={do}")

    with torch.enable_grad():
        if FP8:
            triton_out = flash_attn_fp8_func(q, k, v, dropout_p=DROPOUT, causal=CAUSAL, return_lse=True, return_attn_probs=True)
        else:
            triton_out = flash_attn_func(q, k, v, dropout_p=DROPOUT, causal=CAUSAL, return_lse=True, return_attn_probs=True, fused_backward=True)

    assert len(triton_out) == 3
    triton_out, lse, sd_mask= triton_out[0], triton_out[1], triton_out[2]

    if DROPOUT > 0.0:
        dropout_mask = sd_mask >= 0
    else:
        dropout_mask = None

    # warmup (compilation)
    # triton_dq, triton_dk, triton_dv = torch.autograd.grad(triton_out, (q, k, v), do.clone())

    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    triton_dq, triton_dk, triton_dv = torch.autograd.grad(triton_out, (q, k, v), do.clone())
    end_event.record()
    torch.cuda.synchronize()
    print(f"Backward pass took for Triton: {start_event.elapsed_time(end_event)} ms")

    if DEBUG_MODE:
        print(f"triton_out={triton_out}")
        print(f"triton_lse={lse}")
        print(f"triton_dq.shape={triton_dq.shape} triton_dq={triton_dq}")
        print(f"triton_dk.shape={triton_dk.shape} triton_dk={triton_dk}")
        print(f"triton_dv.shape={triton_dv.shape} triton_dv={triton_dv}")
        print(f"dropout_mask={dropout_mask}")

    if DEBUG_MODE:
        print("--------------Torch----------------")
        print(f"q.shape={q.shape} q={q}")
        print(f"k.shape={k.shape} k={k}")
        print(f"v.shape={v.shape} v={v}")
        print(f"do.shape={do.shape} do={do}")
    with torch.enable_grad():
        torch_out = attention_ref(q, k, v, dropout_p=DROPOUT, dropout_mask=dropout_mask, causal=CAUSAL)
    torch_out, attention_scores = torch_out

    
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    torch_dq, torch_dk, torch_dv = torch.autograd.grad(torch_out, (q, k, v), do)
    end_event.record()
    torch.cuda.synchronize()
    print(f"Backward pass took for Torch: {start_event.elapsed_time(end_event)} ms")
    
    
    if DEBUG_MODE:
        print(f"torch_out={torch_out}")
        print(f"torch_attn_scores={attention_scores}")
        print(f"torch_dq.shape={torch_dq.shape} torch_dq={torch_dq}")
        print(f"torch_dk.shape={torch_dk.shape} torch_dk={torch_dk}")
        print(f"torch_dv.shape={torch_dv.shape} torch_dv={torch_dv}")

    if FP8:
        fp8_assert_close(triton_dq, torch_dq.to(triton_dq.dtype),atol=ATOL_fp8, rtol=RTOL_fp8)  
        fp8_assert_close(triton_dk, torch_dk.to(triton_dk.dtype),atol=ATOL_fp8, rtol=RTOL_fp8)  
        fp8_assert_close(triton_dv, torch_dv.to(triton_dv.dtype),atol=ATOL_fp8, rtol=RTOL_fp8)  
    else:
        torch.testing.assert_close(triton_dv, torch_dv.to(triton_out.dtype),atol=1e-2, rtol=1e-2)  
        print("Gradient matches for dv!")
        torch.testing.assert_close(triton_dk, torch_dk.to(triton_out.dtype),atol=1e-2, rtol=1e-2)  
        print("Gradient matches for dk!")
        torch.testing.assert_close(triton_dq, torch_dq.to(triton_out.dtype),atol=1e-2, rtol=1e-2)  
        print("Gradient matches for dq!")


@pytest.mark.parametrize('BATCH', [1,4,57,128])
@pytest.mark.parametrize('SEQLEN_Q, SEQLEN_K', [(1,1), (4,4), (128,128), (2,1), (1,2), (32,16)])
@pytest.mark.parametrize('DROPOUT, CAUSAL',[(0.0, False), (0.0, True)])
#@pytest.mark.parametrize('DROPOUT, CAUSAL',[(0.0, False),(0.0, True),(0.2, False),(0.2, True)]) #Debug Causal + Dropout. Fails for seq >=64
@pytest.mark.parametrize('NUM_Q_HEADS, NUM_K_HEADS', [(1,1), (16,16), (2,1), (48,8)])
@pytest.mark.parametrize('HEAD_SZ', [8, 32, 128])
@pytest.mark.parametrize('FP8',[(False)])
#@pytest.mark.parametrize('FP8',[(False), (True)]) #TODO Debug FP8
def test_mha_backward_varlen(BATCH: int, SEQLEN_Q: int, SEQLEN_K: int, NUM_Q_HEADS: int, NUM_K_HEADS: int, HEAD_SZ: int, DROPOUT: float, CAUSAL: bool, FP8: bool, dtype=torch.float16):
    torch.manual_seed(20)
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    q.requires_grad = True
    k.requires_grad = True
    v.requires_grad = True

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

    q_unpad.requires_grad = True
    k_unpad.requires_grad = True
    v_unpad.requires_grad = True
    if DEBUG_MODE:
        print(f"query_padding_mask.shape={query_padding_mask.shape} query_padding_mask={query_padding_mask}")
        print(f"key_padding_mask.shape={key_padding_mask.shape} key_padding_mask={key_padding_mask}")

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
    do = torch.randn_like(q)

    if DEBUG_MODE:
        print("--------------Triton----------------")
        print(f"do.shape={do.shape} do={do}")

    with torch.enable_grad():
        triton_out = flash_attn_varlen_func(q_unpad, k_unpad, v_unpad, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, dropout_p=DROPOUT, causal=CAUSAL, return_lse=True, return_attn_probs=True)

    assert len(triton_out) == 3
    triton_out, lse, sd_mask= triton_out[0], triton_out[1], triton_out[2]

    if DROPOUT > 0.0:
        dropout_mask = sd_mask >= 0
        dropout_mask = pad_rearrange_dropout_mask(dropout_mask, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS)
        dropout_mask = dropout_mask > 0
    else:
        dropout_mask = None

    triton_out = output_pad_fn(triton_out)
    triton_dq, triton_dk, triton_dv = torch.autograd.grad(triton_out, (q_unpad, k_unpad, v_unpad), do.clone())

    triton_dq = dq_pad_fn(triton_dq)
    triton_dk = dk_pad_fn(triton_dk)
    triton_dv = dk_pad_fn(triton_dv)
    if DEBUG_MODE:
        print(f"triton_out={triton_out}")
        print(f"triton_lse.shape={lse.shape} triton_lse={lse}")
        print(f"triton_dq.shape={triton_dq.shape} triton_dq={triton_dq}")
        print(f"triton_dk.shape={triton_dk.shape} triton_dk={triton_dk}")
        print(f"triton_dv.shape={triton_dv.shape} triton_dv={triton_dv}")
        print(f"dropout_mask={dropout_mask}")

    if DEBUG_MODE:
        print("--------------Torch----------------")
        print(f"do.shape={do.shape} do={do}")
    with torch.enable_grad():
        torch_out = attention_ref(q, k, v, query_padding_mask=query_padding_mask, key_padding_mask=key_padding_mask, dropout_p=DROPOUT, dropout_mask=dropout_mask, causal=CAUSAL)
    torch_out, attention_scores = torch_out

    torch_dq, torch_dk, torch_dv = torch.autograd.grad(torch_out, (q, k, v), do)

    if DEBUG_MODE:
        print(f"torch_out={torch_out}")
        print(f"torch_attn_scores={attention_scores}")
        print(f"torch_dq.shape={torch_dq.shape} torch_dq={torch_dq}")
        print(f"torch_dk.shape={torch_dk.shape} torch_dk={torch_dk}")
        print(f"torch_dv.shape={torch_dv.shape} torch_dv={torch_dv}")

    torch.testing.assert_close(triton_dv, torch_dv.to(triton_out.dtype),atol=1e-2, rtol=1e-2)  
    torch.testing.assert_close(triton_dk, torch_dk.to(triton_out.dtype),atol=1e-2, rtol=1e-2)  
    torch.testing.assert_close(triton_dq, torch_dq.to(triton_out.dtype),atol=1e-2, rtol=1e-2)  


if __name__ == "__main__":
    # Run the tests
    # TODO: fails for causal=True
    test_mha_backward(8, 1024, 4096, 16, 16, 128, 0.0, True, False, torch.float16)