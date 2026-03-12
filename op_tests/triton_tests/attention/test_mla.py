# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import random
import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.ops.triton.attention.mla import mla_decode_fwd as triton_mla_decode_fwd

from aiter.ops.triton.utils.types import e4m3_dtype
from typing import Optional

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# current supported case in decode MLA: mtp == 0, 1, 2, 3 (decode_qlen = 1, 2, 3, 4)
# qdtype bf16, kdtype bf16: nhead16, nhead128
# qdtype fp8, kdtype fp8: nhead16, nhead128


def check_support(dtype, kv_dtype, nhead):
    if dtype == dtypes.fp8 and kv_dtype == dtypes.bf16:
        return False
    return True


def cal_diff(
    x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    if use_fp8:
        assert cos_diff < 3e-2
    else:
        assert cos_diff < 1e-5


# def ref_masked_attention(
#     query: torch.Tensor,
#     key: torch.Tensor,
#     value: torch.Tensor,
#     scale: float,
#     dtype,
#     is_causal=True,
# ) -> torch.Tensor:
#     attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
#     if is_causal:
#         s_q = query.shape[0]
#         s_k = key.shape[0]
#         attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
#         temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
#         attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
#         attn_bias.to(query.dtype)
#         attn_weights += attn_bias
#     attn_weights = torch.softmax(attn_weights, dim=-1)

#     out = torch.einsum("hqk,khd->qhd", attn_weights.float(), value.float())
#     return out.to(dtype)


def ref_masked_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    query_len = q.shape[0]
    kv_len = k.shape[0]
    if q.shape[1] != k.shape[1]:
        k = torch.repeat_interleave(k, q.shape[1] // k.shape[1], dim=1)
        v = torch.repeat_interleave(v, q.shape[1] // v.shape[1], dim=1)
    if q.dtype == e4m3_dtype:
        q = q.to(torch.bfloat16)
    k = k.to(q.dtype)
    attn = torch.einsum("qhd,khd->hqk", q, k).float()  # GEMM at q.dtype precision
    attn *= scale
    if q.dtype == e4m3_dtype:
        attn = attn * q_descale
    if k.dtype == e4m3_dtype:
        attn = attn * k_descale
    empty_mask = torch.ones(query_len, kv_len, device=q.device)
    mask = torch.triu(empty_mask, diagonal=kv_len - query_len + 1).bool()

    attn.masked_fill_(mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1).to(v.dtype)

    attn = attn.to(q.dtype)
    v = v.to(q.dtype)
    out = torch.einsum("hqk,khd->qhd", attn, v)  # GEMM at q.dtype precision
    if v.dtype == e4m3_dtype:
        out = out * v_descale

    return out


def torch_mha_extend(
    query,  # [total_q, num_query_heads, qk_lora_rank + qk_rope_head_dim]
    key_cache,  # [num_block, block_size, num_kv_heads, qk_lora_rank + qk_rope_head_dim]
    value_cache,  # [num_block, block_size, num_kv_heads, v_head_dim]
    cu_seqlens_q,
    kv_lens,
    block_tables,
    scale: float,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    o_dtype: Optional[torch.dtype] = torch.bfloat16,
) -> torch.Tensor:
    _, block_size, num_kv_heads, qk_head_dim = key_cache.shape
    v_head_dim = value_cache.shape[-1]
    num_seqs = cu_seqlens_q.shape[0] - 1

    outputs: list[torch.Tensor] = []
    for i in range(num_seqs):
        q = query[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]
        q *= scale

        kv_len = kv_lens[i]
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = key_cache[block_indices].view(-1, num_kv_heads, qk_head_dim)
        k = k[:kv_len]
        v = value_cache[block_indices].view(-1, num_kv_heads, v_head_dim)
        v = v[:kv_len]

        out = ref_masked_attention(q, k, v, q_descale, k_descale, v_descale)

        outputs.append(out)

    return torch.cat(outputs, dim=0).to(o_dtype)


def torch_mla_extend(
    query,  # [total_q, num_query_heads, qk_lora_rank + qk_rope_head_dim]
    kv_buffer,  # [num_block, block_size, num_kv_heads, qk_lora_rank + qk_rope_head_dim]
    cu_seqlens_q,
    seq_lens_kv,
    block_tables,
    qk_lora_rank,
    scale: float,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    o_dtype: Optional[torch.dtype] = torch.bfloat16,
):
    _, block_size, num_kv_heads, qk_head_dim = kv_buffer.shape
    num_seqs = cu_seqlens_q.shape[0] - 1

    outputs: list[torch.Tensor] = []
    for i in range(num_seqs):
        q = query[cu_seqlens_q[i] : cu_seqlens_q[i + 1]]

        kv_len = seq_lens_kv[i]
        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = block_tables[i, :num_kv_blocks]

        k = kv_buffer[block_indices].view(-1, num_kv_heads, qk_head_dim)
        k = k[:kv_len]
        v = k[..., :qk_lora_rank]

        out = ref_masked_attention(q, k, v, scale, q_descale, k_descale, v_descale)

        outputs.append(out)

    return torch.cat(outputs, dim=0).to(o_dtype)


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
    decode_qlen,
    split_per_batch=None,
):
    ret = {}

    kv_max_sz = (
        65536 * 32
    )  # calculated by rest of mem after weight loaded in frameworks
    num_page = (kv_max_sz + page_size - 1) // page_size

    cu_seqlens_q = torch.zeros(batch_size + 1, dtype=torch.int, device="cuda")
    seq_lens_qo = torch.empty(batch_size, dtype=torch.int, device="cuda")
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int, device="cuda")
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int, device="cuda")
    if varlen:
        for i in range(batch_size):
            seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
            seq_lens_qo[i] = max(
                min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
            )
    else:
        seq_lens_kv.fill_(ctx_lens)
        seq_lens_qo.fill_(ctx_lens)
    cu_seqlens_q[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    max_seqlen_qo = seq_lens_qo.max().item()
    max_seqlen_kv = seq_lens_kv.max().item()
    max_num_blocks_per_seq = (max_seqlen_kv + page_size - 1) // page_size
    block_tables = torch.randint(
        0,
        num_page,
        (batch_size, max_num_blocks_per_seq),
        dtype=torch.int32,
        device="cuda",
    )
    total_qo = cu_seqlens_q[-1].item()
    total_kv = seq_lens_kv.sum().item()
    kv_buffer = torch.randn(
        (num_page, page_size, 1, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
        device="cuda",
    )
    out_dtype = torch.bfloat16

    # for none absorb (mha)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    sm_scale = 1.0 / (qk_head_dim**0.5)

    # ############################## normal: prefill
    def test_normal_prefill():
        q = torch.randn(
            (total_qo, nhead, qk_head_dim), dtype=torch.bfloat16, device="cuda"
        )
        k = torch.randn(
            (total_kv, nhead, qk_head_dim), dtype=torch.bfloat16, device="cuda"
        )
        v = torch.randn(
            (total_kv, nhead, v_head_dim), dtype=torch.bfloat16, device="cuda"
        )

        out_ref = torch_mha_extend(
            q,
            k,
            v,
            cu_seqlens_q,
            seq_lens_kv,
            block_tables,
            sm_scale,
            dtype=dtype,
        )

        # out_aiter, us_aiter = run_perftest(
        #     aiter.flash_attn_varlen_func,
        #     q,
        #     k,
        #     v,
        #     cu_seqlens_q,
        #     kv_indptr,
        #     max_seqlen_qo,
        #     max_seqlen_kv,
        #     softmax_scale=sm_scale,
        #     causal=True,
        # )

        # flop = (
        #     batch_size
        #     * nhead
        #     * 2
        #     * (ctx_lens * qk_head_dim * ctx_lens + ctx_lens * ctx_lens * v_head_dim)
        # )
        # checkAllclose(
        #     out_ref.to(torch.float),
        #     out_aiter.to(torch.float),
        #     msg=f"mla_prefill-normal    [torch vs  aiter_ck]: {us_aiter:>8.2f} us...... {flop/us_aiter/1000/1000:>8.2f} TFlops",
        # )
        # return us_aiter

    # us_aiter = None
    # if (
    #     dtype == torch.bfloat16 and kvtype == torch.bfloat16
    # ) and batch_size * ctx_lens * nhead < 256 * 8192 * 16:
    #     us_aiter = test_normal_prefill()
    #     ret["prefill:ck_192"] = us_aiter

    torch.cuda.empty_cache()
    # absorb init
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    nhead_kv = 1
    v_head_dim = kv_lora_rank
    sm_scale = 1.0 / (qk_head_dim**0.5)

    # test prefill
    # ############################## absorb: prefill
    def test_absorb_prefill():
        q = torch.randn(
            (total_qo, nhead, qk_head_dim), dtype=torch.bfloat16, device="cuda"
        )
        out_ref = torch_mla_extend(
            q,  # [total_q, num_query_heads, qk_lora_rank + qk_rope_head_dim]
            kv_buffer,  # [num_block, block_size, num_kv_heads, qk_lora_rank + qk_rope_head_dim]
            cu_seqlens_q,
            seq_lens_kv,
            block_tables,
            kv_lora_rank,
            sm_scale,
            q_descale=None,
            k_descale=None,
            v_descale=None,
            o_dtype=out_dtype,
        )

        # checkAllclose(
        #     out_ref,
        #     out_triton,
        #     msg=f"mla_prefill-absorb    [torch vs aiter_asm]: {us_asm:>8.2f} us......",
        # )
        # return us_asm

    # us_asm = None
    # if (
    #     dtype == torch.bfloat16 and kvtype == torch.bfloat16 and nhead in [16, 128]
    # ) and batch_size * ctx_lens * nhead < 32 * 8192 * 16:
    #     us_asm = test_absorb_prefill()
    #     ret["prefill:asm_576"] = us_asm

    torch.cuda.empty_cache()

    # ############################## absorb: decode
    # seq_lens_qo = torch.randint(1, 5, (batch_size,), dtype=torch.int)
    # if nhead == 16 and decode_qlen != 1:
    #     return
    seq_lens_qo.fill_(decode_qlen)

    max_seqlen_qo = seq_lens_qo.max().item()
    cu_seqlens_q[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = cu_seqlens_q[-1].item()
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16, device="cuda")

    # troch implementation
    out_ref = torch_mla_extend(
        q,  # [total_q, num_query_heads, qk_lora_rank + qk_rope_head_dim]
        kv_buffer,  # [num_block, block_size, num_kv_heads, qk_lora_rank + qk_rope_head_dim]
        cu_seqlens_q,
        seq_lens_kv,
        block_tables,
        kv_lora_rank,
        sm_scale,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        o_dtype=out_dtype,
    )

    def test_absorb_decode():
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_triton = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(
            -1
        )
        _, us_asm_decode = run_perftest(
            triton_mla_decode_fwd,
            q,
            kv_buffer,
            out_triton,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seq_lens_kv,
            max_seqlen_kv=max_seqlen_kv,
            block_tables=block_tables,
            softmax_scale=sm_scale,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            causal=True,
            q_descale=None,
            k_descale=None,
            v_descale=None,
        )

        err = checkAllclose(
            out_ref,
            out_triton,
            msg=f"mla_decode-absorb    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        return err, us_asm_decode

    err = None
    us_asm_decode = 1e12
    if (dtype == torch.bfloat16 and kvtype == torch.bfloat16) and nhead in [
        16,
        32,
        64,
        128,
    ]:
        err, us_asm_decode = test_absorb_decode()
    elif kvtype == dtypes.fp8 and nhead in [16, 128]:
        err, us_asm_decode = test_absorb_decode()

    ret["decode:err"] = err
    ret["decode:asm_576"] = us_asm_decode

    flops = decode_qlen * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
    bytes = (
        total_kv * nhead_kv * qk_head_dim * (torch.finfo(kvtype).bits // 8)
        + total_q * nhead * qk_head_dim * (torch.finfo(dtype).bits // 8)
        + total_q * nhead * v_head_dim * (torch.finfo(out_dtype).bits // 8)
    )

    ret["decode:flops"] = flops
    ret["decode:bytes"] = bytes
    ret["decode:TFLOPS"] = flops / us_asm_decode / 1e6
    ret["decode:TB/s"] = bytes / us_asm_decode / 1e6

    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-k",
    "--kv_lora_rank",
    type=int,
    default=512,
    help="""kv lora rank.
    e.g.: -k 512""",
)
parser.add_argument(
    "-qn",
    "--qk_nope_head_dim",
    type=int,
    default=128,
    help="""qk nope head dim.
    e.g.: -qn 128""",
)
parser.add_argument(
    "-qr",
    "--qk_rope_head_dim",
    type=int,
    default=64,
    help="""qk rope head dim.
    e.g.: -qr 64""",
)
parser.add_argument(
    "-vh",
    "--v_head_dim",
    type=int,
    default=128,
    help="""v head dim.
    e.g.: -vh 128""",
)
parser.add_argument(
    "-blk",
    "--block_size",
    type=int,
    default=64,
    help="""Block size.
    e.g.: -blk 64""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    nargs="*",
    default="bf16,",
    choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
    metavar="{bf16, fp8}",
    help="""Data type of Q.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-kvd",
    "--kv_dtype",
    nargs="*",
    type=dtypes.str2Dtype,
    default="bf16,",
    choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
    metavar="{bf16, fp8}",
    help="""Data type of KV.
    e.g.: -kvd bf16""",
)
parser.add_argument(
    "-c",
    "--ctxLen",
    type=int,
    nargs="*",
    default=[21, 64, 256, 512, 1200, 3200, 5200, 8192],
    help="""Context length.
    e.g.: -c 21""",
)
parser.add_argument(
    "-b",
    "--batchSize",
    type=int,
    nargs="*",
    default=[1, 3, 5, 16, 32, 64, 128, 256],
    help="""Batch size.
    e.g.: -b 16""",
)
parser.add_argument(
    "-n",
    "--nhead",
    type=dtypes.str2tuple,
    choices=[(16, 1), (16, 2), (16, 4), (128, 1), (128, 2)],
    nargs="*",
    const=None,
    default=[(16, 1), (16, 2), (16, 4), (128, 1), (128, 2)],
    help="""Number of nhead and decode_qlen.
    e.g.: -n 16,1""",
)
parser.add_argument(
    "-splits",
    "--split_per_batch",
    type=int,
    nargs="*",
    default=[None],
    help="""kv seqlens split num for per batch.
    e.g.: -ms 32""",
)
parser.add_argument(
    "--varlen",
    action="store_true",
    help="""variable kv seqlens per batch. Default: False.
    --varlen # True""",
)


args = parser.parse_args()

for nhead, decode_qlen in args.nhead:
    df = []
    for dtype, kvtype, ctx_len, batch_size, split_per_batch in itertools.product(
        args.dtype, args.kv_dtype, args.ctxLen, args.batchSize, args.split_per_batch
    ):
        if check_support(dtype, kvtype, nhead):
            ret = test_mla(
                ctx_len,
                batch_size,
                nhead,
                args.kv_lora_rank,
                args.qk_nope_head_dim,
                args.qk_rope_head_dim,
                args.v_head_dim,
                dtype,
                kvtype,
                args.block_size,
                varlen=args.varlen,
                decode_qlen=decode_qlen,
                split_per_batch=split_per_batch,
            )
            df.append(ret)
    df = pd.DataFrame(df)
    # df.to_csv(f"mla_nhead{nhead}decode_qlen{decode_qlen}.csv")
    df_md = df.to_markdown(index=False)
    aiter.logger.info("mla summary (markdown):\n%s", df_md)
