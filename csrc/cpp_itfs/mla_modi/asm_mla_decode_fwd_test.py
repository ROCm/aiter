# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, benchmark, run_perftest
import random
import itertools
from csrc.cpp_itfs.mla_modi.asm_mla_decode_fwd import asm_mla_decode_fwd
import pandas as pd

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
    is_causal=True,
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
        o = ref_masked_attention(q, k, v, sm_scale, dtype, is_causal=is_causal)
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
    mtp,
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
    kv_indices = torch.randint(0, num_page, (kv_indptr[-1].item(),), dtype=torch.int)
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    max_seqlen_qo = seq_lens_qo.max().item()
    max_seqlen_kv = seq_lens_kv.max().item()
    total_qo = qo_indptr[-1].item()
    total_kv = kv_indptr[-1].item()
    kv_buffer = torch.randn(
        (num_page * page_size, 1, kv_lora_rank + qk_rope_head_dim),
        dtype=kvtype,
    )

    # for none absorb (mha)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    sm_scale = 1.0 / (qk_head_dim**0.5)

    us_aiter = None
    # if batch_size * ctx_lens * nhead < 256 * 8192 * 16:
    #     us_aiter = test_normal_prefill()
    torch.cuda.empty_cache()
    # absorb init
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    nhead_kv = 1
    v_head_dim = kv_lora_rank
    sm_scale = 1.0 / (qk_head_dim**0.5)

    us_asm = None
    torch.cuda.empty_cache()
    # ############################## absorb: decode
    # seq_lens_qo = torch.randint(1, 5, (batch_size,), dtype=torch.int)
    if nhead == 16 and mtp != 1:
        return
    seq_lens_qo.fill_(mtp)

    max_seqlen_qo = seq_lens_qo.max().item()
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=dtype)

    # troch implementation
    out_ref, us_torch_decode = run_perftest(
        torch_mla_extend,
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        kv_lora_rank,
        qk_rope_head_dim,
        is_causal=False,
        dtype=dtype,
        num_iters=2,
        num_warmup=0,
    )

    # aiter implementation
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=dtype).fill_(-1)

    (attn_logits, attn_lse), us_asm_decode = run_perftest(
        asm_mla_decode_fwd,
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

    flops = mtp * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
    bytes = (
        total_kv * nhead_kv * qk_head_dim + total_q * nhead * (qk_head_dim + v_head_dim)
    ) * (torch.finfo(dtype).bits // 8)
    checkAllclose(
        out_ref,
        out_asm,
        msg=f"mla_decode-absorb    [golden vs aiter_asm]:{us_torch_decode:>8.2f} us vs {us_asm_decode:>8.2f} us......",
    )
    return {
        "prefill:ck_192": us_aiter,
        "prefill:asm_576": us_asm,
        "decode:flops": flops,
        "decode:bytes": bytes,
        "decode:asm_576": us_asm_decode,
        "decode:TFLOPS": flops / us_asm_decode / 1e6,
        "decode:TB/s": bytes / us_asm_decode / 1e6,
    }


kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128
block_size = 1
list_dtype = [(torch.bfloat16, torch.bfloat16)]
list_ctx_len = [21, 64, 256, 512, 1200, 3200, 5200, 8192][:1]
list_batch_size = [1, 3, 5, 16, 32, 64, 128, 256][:1]
list_nhead = [(16, 1), (128, 2)][:1]


for nhead, mtp in list_nhead:
    df = []
    for (dtype, kvtype), ctx_len, batch_size in itertools.product(
        list_dtype, list_ctx_len, list_batch_size
    ):
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
            mtp=mtp,
        )
        df.append(ret)
    df = pd.DataFrame(df)
    # df.to_csv(f"mla_mtp{mtp}.csv")
    aiter.logger.info(f"summary:\n{df}")
