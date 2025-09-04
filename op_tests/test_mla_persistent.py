# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter import dtypes
import random
import itertools
import argparse

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


def cal_diff(
    x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    # if use_fp8:
    #     assert cos_diff < 3e-2
    # else:
    #     assert cos_diff < 1e-5


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype,
    is_causal=True,
    is_fp8=False,
    q_scale=None,
    kv_scale=None,
) -> torch.Tensor:

    if is_fp8:
        scale *= q_scale * kv_scale

    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias

    lse = attn_weights.logsumexp(dim=-1)

    m = attn_weights.max(-1).values

    attn_weights_exp = torch.exp(attn_weights - m.unsqueeze(-1))

    l = attn_weights_exp.sum(-1)

    if is_fp8:
        attn_weights_fp8 = attn_weights_exp.to(torch.float8_e4m3fnuz)
        attn_weights_exp = attn_weights_fp8.to(torch.float)

    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())

    out = out / l.transpose(0, 1).unsqueeze(-1)

    if is_fp8:
        out *= kv_scale
    return out.to(dtype), lse


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
    q_scale=None,
    kv_scale=None,
):
    is_fp8 = q.dtype == torch.float8_e4m3fnuz

    if is_fp8:
        q = q.to(torch.float)
        kvc_cache = kvc_cache.to(torch.float)

    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    lses = []
    for i in range(bs):
        kvc = kvs[i]
        q = qs[i]
        k = kvc
        v, _ = torch.split(kvc, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        o, lse = ref_masked_attention(
            q,
            k,
            v,
            sm_scale,
            dtype,
            is_causal=is_causal,
            is_fp8=is_fp8,
            q_scale=q_scale,
            kv_scale=kv_scale,
        )
        os.append(o)
        lses.append(lse)
    o = torch.concat(os)
    lse = torch.concat(lses).transpose(0, 1)
    return o, lse


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
            # seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
            seq_lens_kv[i] = random.uniform(5, ctx_lens)
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

    us_asm = None
    # if batch_size * ctx_lens * nhead < 32 * 8192 * 16:
    #     us_asm = test_absorb_prefill()
    torch.cuda.empty_cache()
    nhead_kv = 1

    # ############################## absorb: decode
    # seq_lens_qo = torch.randint(1, 5, (batch_size,), dtype=torch.int)
    # if nhead == 16 and mtp != 1:
    #     return
    seq_lens_qo.fill_(mtp)

    max_seqlen_qo = seq_lens_qo.max().item()
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=dtype)

    # troch implementation
    out_ref, lse_ref = torch_mla_extend(
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
    )

    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count

    # aiter implementation
    # the tensor's meaning please refer aiter/ops/attention.py
    work_meta_data = torch.empty([10], dtype=torch.uint64, device="cuda")
    work_indptr = torch.empty([cu_num + 1], dtype=torch.int32, device="cuda")
    work_info_set = torch.empty(
        [batch_size * cu_num, 8], dtype=torch.int32, device="cuda"
    )
    reduce_indptr = torch.empty([batch_size + 1], dtype=torch.int32, device="cuda")
    reduce_final_map = torch.empty([batch_size, 2], dtype=torch.int32, device="cuda")
    reduce_partial_map = torch.empty(
        [batch_size * cu_num], dtype=torch.int32, device="cuda"
    )

    meta = aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        nhead // nhead_kv,
        nhead_kv,
        True,
        work_meta_data,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    )

    def test_absorb_decode():
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=dtype).fill_(-1)

        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q,
            kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            sm_scale,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
        )

        # print(f"{out_ref.view(total_q, -1)=}")
        # print(f"{out_asm.view(total_q, -1)=}")
        # checkAllclose(logits_ref, attn_logits,
        #               msg=f'attn_logits [golden vs aiter_asm]')
        # checkAllclose(lse_ref, attn_lse, msg="attn_lse    [golden vs aiter_asm]")
        flops = mtp * total_kv * nhead * (qk_head_dim + v_head_dim) * 2

        import pdb;pdb.set_trace()

        bytes = (
            total_kv * nhead_kv * qk_head_dim
            + total_q * nhead * (qk_head_dim + v_head_dim)
        ) * (torch.finfo(dtype).bits // 8)
        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        return err, us_asm_decode

    err = None
    us_asm_decode = 10000000000
    err, us_asm_decode = test_absorb_decode()

    def test_absorb_decode_fp8():
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=dtype).fill_(-1)

        q_fp8, q_scale = aiter.per_tensor_quant(q, quant_dtype=torch.float8_e4m3fnuz)
        q_scale = q_scale.to(torch.float)

        kv_buffer_fp8 = kv_buffer.to(torch.float8_e4m3fnuz)
        kv_scale = torch.ones([1], dtype=torch.float, device="cuda")

        out_ref_fp8, lse_ref_fp8 = torch_mla_extend(
            q_fp8,
            kv_buffer_fp8,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=dtype,
            is_causal=True,
            q_scale=q_scale,
            kv_scale=kv_scale,
        )

        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q_fp8,
            kv_buffer_fp8.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            sm_scale,
            q_scale=q_scale,
            kv_scale=kv_scale,
            work_meta_data=work_meta_data,
            work_indptr=work_indptr,
            work_info_set=work_info_set,
            reduce_indptr=reduce_indptr,
            reduce_final_map=reduce_final_map,
            reduce_partial_map=reduce_partial_map,
        )

        cal_diff(out_ref, out_asm, "out", True)

        # print(f"{out_ref.view(total_q, -1)=}")
        # print(f"{out_asm.view(total_q, -1)=}")
        # checkAllclose(logits_ref, attn_logits,
        #               msg=f'attn_logits [golden vs aiter_asm]')
        # checkAllclose(lse_ref, attn_lse, msg="attn_lse    [golden vs aiter_asm]")
        flops = mtp * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
        bytes = (
            total_kv * nhead_kv * qk_head_dim
            + total_q * nhead * (qk_head_dim + v_head_dim)
        ) * (torch.finfo(dtype).bits // 8)
        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        err_fp8 = checkAllclose(
            out_ref_fp8,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden fp8 vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        return err, err_fp8, us_asm_decode

    # err_fp8_fp32, err_fp8_fp8, us_asm_decode_fp8 = test_absorb_decode_fp8()

    # print(f"{out_ref.view(total_q, -1)=}")
    # print(f"{out_asm.view(total_q, -1)=}")
    # checkAllclose(logits_ref, attn_logits,
    #               msg=f'attn_logits [golden vs aiter_asm]')
    # checkAllclose(lse_ref, attn_lse, msg="attn_lse    [golden vs aiter_asm]")
    flops = mtp * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
    bytes = (
        total_kv * nhead_kv * qk_head_dim + total_q * nhead * (qk_head_dim + v_head_dim)
    ) * (torch.finfo(dtype).bits // 8)

    return {
        "decode:flops": flops,
        "decode:bytes": bytes,
        "decode:err": err,
        "decode:asm_576": us_asm_decode,
        "decode:TFLOPS": flops / us_asm_decode / 1e6,
        "decode:TB/s": bytes / us_asm_decode / 1e6,
        # "decode_fp8:err vs fp32": err_fp8_fp32,
        # "decode_fp8:err vs fp8": err_fp8_fp8,
        # "decode_fp8:asm_576": us_asm_decode_fp8,
        # "decode_fp8:TFLOPS": flops / us_asm_decode_fp8 / 1e6,
        # "decode_fp8:TB/s": bytes / us_asm_decode_fp8 / 1e6,
    }


kv_lora_rank = 512
qk_nope_head_dim = 128
qk_rope_head_dim = 64
v_head_dim = 128
block_size = 1
list_dtype = ["bf16"]
l_kv_dtype = ["bf16"]
list_nhead = [(16, 2)]

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
    default=512,
    help="""qk nope head dim.
    e.g.: -qn 512""",
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
    default=512,
    help="""v head dim.
    e.g.: -vh 512""",
)
parser.add_argument(
    "-blk",
    "--block_size",
    type=int,
    default=1,
    help="""Block size.
    e.g.: -blk 1""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=["bf16"],
    nargs="*",
    default=["bf16"],
    help="""Data type of Q.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-kvd",
    "--kv_dtype",
    type=str,
    choices=["bf16"],
    nargs="*",
    default=["bf16"],
    help="""Data type of KV.
    e.g.: -kvd bf16""",
)
parser.add_argument(
    "-c",
    "--ctxLen",
    type=int,
    nargs="*",
    default=[28, 512, 1023, 4888, 12800],  #
    help="""Context length.
    e.g.: -c 21""",
)
parser.add_argument(
    "-b",
    "--batchSize",
    type=int,
    nargs="*",
    default=[i for i in range(1, 80)],  # [41],
    help="""Batch size.
    e.g.: -b 16""",
)
parser.add_argument(
    "-n",
    "--nhead",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""Number of heads.
    e.g.: -n 16,1""",
)

import pandas as pd

args = parser.parse_args()
list_dtype = [dtypes.d_dtypes[key] for key in args.dtype]
l_kv_dtype = [dtypes.d_dtypes[key] for key in args.kv_dtype]
if args.nhead is not None:
    list_nhead = [args.nhead]

for nhead, mtp in list_nhead:
    df = []
    for dtype, kvtype, ctx_len, batch_size in itertools.product(
        list_dtype, l_kv_dtype, args.ctxLen, args.batchSize
    ):
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
            varlen=True,
            mtp=mtp,
        )
        df.append(ret)
    df = pd.DataFrame(df)
    # df.to_csv(f"mla_nhead{nhead}mtp{mtp}.csv")
    aiter.logger.info(f"summary:\n{df}")
