import triton
import torch
import triton.language as tl
import pytest
from typing import Any, Dict, Optional


from aiter.ops.triton.mha import flash_attn_func
from aiter.test_mha_common import (
    attention_ref)

@pytest.mark.parametrize('BATCH, SEQLEN_Q, SEQLEN_K, NUM_Q_HEADS, NUM_K_HEADS, HEAD_SZ', [(1, 2, 2, 1, 1, 8)])
def test_mha(BATCH: int, SEQLEN_Q: int, SEQLEN_K: int, NUM_Q_HEADS: int, NUM_K_HEADS: int, HEAD_SZ: int, dtype=torch.float16):

    torch.manual_seed(20)
    q = torch.randn((BATCH, SEQLEN_Q, NUM_Q_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    k = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ ), device="cuda", dtype=dtype)
    v = torch.randn((BATCH, SEQLEN_K, NUM_K_HEADS, HEAD_SZ), device="cuda", dtype=dtype)
    print(f"q.shape={q.shape}, q={q}")
    print(f"k.shape={k.shape}, k={k}")
    print(f"v.shape={v.shape}, v={v}")
    
    triton_out = flash_attn_func(q, k, v)
    print(f"triton_out={triton_out}")
    
    torch_out = attention_ref(q, k, v)
    print(f"torch_out={torch_out}")
