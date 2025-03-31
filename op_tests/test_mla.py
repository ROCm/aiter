# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from op_tests.triton.utils import mla_decode_ref
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter.test_mha_common import attention_ref
from einops import rearrange

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


@benchmark()
def test_mla(
    ctx_lens,
    batch_size,
    nhead,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    kvtype,
    page_size,
    num_kv_splits,
):
    kv_max_sz = 65536  # calculated by rest of mem after weight loaded in frameworks
    num_page = (kv_max_sz + page_size - 1) // page_size

    # for none absorb (mha)
    if batch_size * ctx_lens < 128 * 1024:
        # attention_ref will OOO for big input...
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        q = torch.randn((batch_size, ctx_lens, nhead, qk_head_dim), dtype=dtype)
        k = torch.randn((batch_size, ctx_lens, nhead, qk_head_dim), dtype=dtype)
        v = torch.randn((batch_size, ctx_lens, nhead, v_head_dim), dtype=dtype)
        (out_ref, *_), us_ref = run_perftest(
            attention_ref, q, k, v, causal=True, num_warmup=1, num_iters=2
        )
        out_aiter, us_aiter = run_perftest(
            aiter.flash_attn_func, q, k, v, causal=True
        )
        flop = (
            batch_size
            * nhead
            * 2
            * (ctx_lens * qk_head_dim * ctx_lens + ctx_lens * ctx_lens * v_head_dim)
        )
        checkAllclose(
            out_ref,
            out_aiter,
            msg=f"mha_out     [golden vs aiter_asm]:{us_ref:.2f} us vs {us_aiter:.2f} us...... {flop/us_aiter/1000/1000:.2f} TFlops",
        )

    # for decode (mqa)
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    nhead_kv = 1
    v_head_dim = kv_lora_rank  # for attn_mqa in sglang

    q = torch.randn((batch_size, nhead, qk_head_dim), dtype=dtype)
    kv_buffer = torch.randn(
        (num_page * page_size, nhead_kv, qk_head_dim),  # decode kv head
        dtype=kvtype,
    )

    if qk_head_dim != v_head_dim:
        out_ref = q.new_empty((q.shape[0], nhead, v_head_dim)).fill_(-1)
    else:
        out_ref = torch.empty_like(q)

    sm_scale = 1.0 / (qk_head_dim**0.5)

    seq_lens = torch.tensor([ctx_lens for _ in range(batch_size)], dtype=torch.int)
    kv_indptr = torch.zeros((batch_size + 1,), dtype=torch.int)
    kv_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens, dim=0)
    kv_indices = torch.randint(
        0, num_page, (kv_indptr[-1].item() + 1,), dtype=torch.int
    )
    attn_logits = torch.empty(
        (batch_size, nhead, num_kv_splits, v_head_dim + 1),
        dtype=torch.float32,
    )
    _, us_ref = run_perftest(
        mla_decode_ref.decode_attention_fwd,
        q,
        kv_buffer,
        kv_buffer[..., :kv_lora_rank],
        out_ref,
        kv_indptr,
        kv_indices,
        attn_logits,
        num_kv_splits,
        sm_scale,
    )
    logits_ref, lse_ref = attn_logits.split([v_head_dim, 1], dim=-1)
    logits_ref = rearrange(logits_ref, "bs h sp d -> bs sp h d")
    lse_ref = rearrange(lse_ref, "bs h sp d -> bs sp h d")
    # print(f'{out_ref.view(batch_size, -1)=}')

    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    out_asm = torch.empty((batch_size, nhead, v_head_dim), dtype=dtype).fill_(-1)
    (attn_logits, attn_lse), us_asm = run_perftest(
        aiter.mla.mla_decode_fwd,
        q,
        kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
        out_asm,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        sm_scale,
    )

    # print(f'{out_asm.view(batch_size, -1)=}')
    # checkAllclose(logits_ref, attn_logits,
    #               msg=f'attn_logits [golden vs aiter_asm]')
    # checkAllclose(lse_ref, attn_lse,
    #               msg=f'attn_lse    [golden vs aiter_asm]')
    checkAllclose(
        out_ref,
        out_asm,
        msg=f"attn_out    [golden vs aiter_asm]:{us_ref:.2f} us vs {us_asm:.2f} us......",
    )
    return {"triton": us_ref, "asm": us_asm}


ctx_len = 3200
kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128
nhead = 16  # 128/TP8
block_size = 1
num_kv_splits = 16  # don't why but sglang force 16.... for triton
for dtype, kvtype in [(torch.bfloat16, torch.bfloat16)]:
    for ctx_len in [21, 64, 256, 512, 1024, 3200, 8192][:]:
        for batch_size in [1, 2, 3, 5, 16, 32, 64, 128, 256][:]:
            ret = test_mla(
                ctx_len,
                batch_size,
                nhead,
                kv_lora_rank,
                qk_nope_head_dim,
                qk_rope_head_dim,
                v_head_dim,
                dtype,
                kvtype,
                block_size,
                num_kv_splits,
            )
