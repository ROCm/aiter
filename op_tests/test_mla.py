# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import aiter
from op_tests.triton.utils import mla_decode_ref, mla_extend_ref
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter.test_mha_common import attention_ref
from einops import rearrange
import random

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype,
    is_causal=True,
) -> torch.Tensor:
    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias
    attn_weights = torch.softmax(attn_weights, dim=-1)

    out = torch.einsum("hqk,khd->qhd", attn_weights.float(), value.float())
    return out.to(dtype)


def torch_mha_extend(
    q,  # [total_q, nheads, headdim_q]
    k,  # [num_page * page_size, nhead_kv, qk_head_dim]
    v,  # [num_page * page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    dtype,
):
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    ks = torch.tensor_split(k, kv_indptr.tolist()[1:])
    vs = torch.tensor_split(v, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    for i in range(bs):
        q = qs[i]
        k = ks[i]
        v = vs[i]
        o = ref_masked_attention(q, k, v, sm_scale, dtype)
        os.append(o)
    o = torch.concat(os)
    return o


def torch_mla_extend(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_page * page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
):
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    for i in range(bs):
        kvc = kvs[i]
        q = qs[i]
        k = kvc
        v, _ = torch.split(kvc, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        o = ref_masked_attention(q, k, v, sm_scale, dtype)
        os.append(o)
    o = torch.concat(os)
    return o


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
    varlen,
):
    kv_max_sz = (
        65536 * 32
    )  # calculated by rest of mem after weight loaded in frameworks
    num_page = (kv_max_sz + page_size - 1) // page_size

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_qo = torch.empty(batch_size, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    if varlen:
        for i in range(batch_size):
            seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
            seq_lens_qo[i] = max(
                min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
            )
    else:
        seq_lens_kv.fill_(ctx_lens)
        seq_lens_qo.fill_(ctx_lens)
    kv_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_kv, dim=0)
    kv_indices = torch.randint(
        0, num_page, (kv_indptr[-1].item() + 1,), dtype=torch.int
    )
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    max_seqlen_qo = seq_lens_qo.max().item()
    max_seqlen_kv = seq_lens_kv.max().item()
    total_qo = qo_indptr[-1].item()
    total_kv = kv_indptr[-1].item()
    kv_buffer = torch.randn(
        (num_page * page_size, 1, kv_lora_rank + qk_rope_head_dim),
        dtype=kvtype,
    )
    us_aiter = None
    us_triton = None
    us_asm = None
    # for none absorb (mha)
    if batch_size * ctx_lens < 128 * 8192:
        # attention_ref will OOO for big input...
        qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        sm_scale = 1.0 / (qk_head_dim**0.5)
        # ############################## normal: prefill
        q = torch.randn((total_qo, nhead, qk_head_dim), dtype=dtype)
        k = torch.randn((total_kv, nhead, qk_head_dim), dtype=dtype)
        v = torch.randn((total_kv, nhead, v_head_dim), dtype=dtype)

        out_ref, us_ref = run_perftest(
            torch_mha_extend,
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            dtype=dtype,
            num_iters=3,
            num_warmup=1,
        )
        out_aiter, us_aiter = run_perftest(
            aiter.flash_attn_varlen_func,
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            max_seqlen_qo,
            max_seqlen_kv,
            softmax_scale=sm_scale,
            causal=True,
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
            msg=f"mla_prefill-normal    [torch vs  aiter_ck]:{us_ref:.2f} us vs {us_aiter:>8.2f} us...... {flop/us_aiter/1000/1000:.2f} TFlops",
        )

    # absorb init
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    nhead_kv = 1
    v_head_dim = kv_lora_rank
    sm_scale = 1.0 / (qk_head_dim**0.5)

    # test prefill
    if batch_size * ctx_lens < 32 * 8192:
        # ############################## absorb: prefill
        q = torch.randn((total_qo, nhead, qk_head_dim), dtype=dtype)

        out_torch, us_torch = run_perftest(
            torch_mla_extend,
            q,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=dtype,
            num_iters=3,
            num_warmup=1,
        )

        prefix_indptr = kv_indptr - qo_indptr
        tmp = kv_indptr[1:] - seq_lens_qo
        tmp_inpptr, _ = torch.concat([kv_indptr[1:], tmp]).sort()
        prefix_kv_indices = kv_indices.tensor_split(tmp_inpptr.tolist())
        extend_kv_indices = torch.concat(
            [el for i, el in enumerate(prefix_kv_indices) if i % 2 == 1]
        )
        prefix_kv_indices = torch.concat(
            [el for i, el in enumerate(prefix_kv_indices) if i % 2 == 0]
        )
        extend_kvc = torch.index_select(kv_buffer, 0, extend_kv_indices)
        out_triton = torch.empty((total_qo, nhead, v_head_dim), dtype=dtype).fill_(-1)
        _, us_triton = run_perftest(
            mla_extend_ref.extend_attention_fwd,
            q,
            extend_kvc,
            extend_kvc[..., :kv_lora_rank],
            out_triton,
            kv_buffer,
            kv_buffer[..., :kv_lora_rank],
            qo_indptr,
            prefix_indptr,
            prefix_kv_indices,
            None,
            None,
            max_seqlen_qo,
            sm_scale,
        )
        checkAllclose(
            out_torch,
            out_triton,
            msg=f"mla_prefill-absorb    [torch vs    triton]:{us_torch:.2f} us vs {us_triton:>8.2f} us......",
        )

        out_asm = torch.empty((total_qo, nhead, v_head_dim), dtype=dtype).fill_(-1)
        (attn_logits, attn_lse), us_asm = run_perftest(
            aiter.mla.mla_prefill_fwd,
            q,
            kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            sm_scale,
        )

        checkAllclose(
            out_torch,
            attn_logits,
            msg=f"mla_prefill-absorb    [torch vs aiter_asm]:{us_torch:.2f} us vs {us_asm:>8.2f} us......",
        )

    # ############################## absorb: decode
    seq_lens_qo.fill_(1)
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=dtype)

    # troch implementation
    out_torch_decode, us_torch_decode = run_perftest(
        torch_mla_extend,
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        kv_lora_rank,
        qk_rope_head_dim,
        dtype=dtype,
        num_iters=3,
        num_warmup=1,
    )

    # Triton implementation
    if qk_head_dim != v_head_dim:
        out_ref = q.new_empty((total_q, nhead, v_head_dim)).fill_(-1)
    else:
        out_ref = torch.empty_like(q)

    num_kv_splits = 16
    attn_logits = torch.empty(
        (total_q, nhead, num_kv_splits, v_head_dim + 1),
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
    # logits_ref, lse_ref = attn_logits.split([v_head_dim, 1], dim=-1)
    # logits_ref = rearrange(logits_ref, "bs h sp d -> bs sp h d")
    # lse_ref = rearrange(lse_ref, "bs h sp d -> bs sp h d")
    checkAllclose(
        out_torch_decode,
        out_ref,
        msg=f"mla_decode-absorb    [golden vs    triton]:{us_torch_decode:.2f} us vs {us_ref:.2f} us......",
    )

    # aiter implementation
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
        out_torch_decode,
        out_asm,
        msg=f"mla_decode-absorb    [golden vs aiter_asm]:{us_torch_decode:.2f} us vs {us_asm:.2f} us......",
    )
    return {"ck_576": us_aiter, "triton_576": us_triton, "asm_576": us_asm}


kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128
nhead = 16  # 128/TP8
block_size = 1
df = []
for dtype, kvtype in [(torch.bfloat16, torch.bfloat16)]:
    for ctx_len in [21, 64, 256, 512, 1200, 2200, 3200, 4200, 5200, 6200, 8192][:]:
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
                varlen=False,
            )
            df.append(ret)
import pandas as pd

df = pd.DataFrame(df)
# df.to_csv("mla_prefill.csv")
print(df)
