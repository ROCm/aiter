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
# torch.set_printoptions(sci_mode=False, threshold=torch.inf)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# setup_seed(1)

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
            # seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
            seq_lens_kv[i] = random.uniform(4, ctx_lens)
            seq_lens_qo[i] = max(
                min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
            )
    else:
        seq_lens_kv.fill_(ctx_lens)
        seq_lens_qo.fill_(ctx_lens)
    # seq_lens_kv = torch.tensor([3819,9978,784,530,8062,1390,287,1008,5090,5304,7396,2288,2104,4063,3644,5091,6470,4732,7237,430,2777,956,1357,5478,1292,521,6802,1347,2388,5062,443,8560,5049,7235,927,9580,623,4913,2511,8120,1638,4859,600,7289,8278,6693,136,1021,1465,5859,1278,7123,7839,2459,1090,6333,812,9358,6345,8616,2313,6115,6059,4963,
    #     12343, 213, 143, 12312, 12345, 3215, 4444, 5325, 2132, 123, 456, 2135, 135, 2564, 5465, 4362], device="cuda")
    # seq_lens_kv = seq_lens_kv[:batch_size]
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

    # ############################## normal: prefill
    def test_normal_prefill():
        q = torch.randn((total_qo, nhead, qk_head_dim), dtype=dtype)
        k = torch.randn((total_kv, nhead, qk_head_dim), dtype=dtype)
        v = torch.randn((total_kv, nhead, v_head_dim), dtype=dtype)

        out_ref = torch_mha_extend(
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            dtype=dtype,
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
            msg=f"mla_prefill-normal    [torch vs  aiter_ck]: {us_aiter:>8.2f} us...... {flop/us_aiter/1000/1000:>8.2f} TFlops",
        )
        return us_aiter

    us_aiter = None
    # if batch_size * ctx_lens * nhead < 256 * 8192 * 16:
    #     us_aiter = test_normal_prefill()
    torch.cuda.empty_cache()
    # absorb init
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    nhead_kv = 1
    v_head_dim = kv_lora_rank
    sm_scale = 1.0 / (qk_head_dim**0.5)

    # test prefill
    # ############################## absorb: prefill
    def test_absorb_prefill():
        q = torch.randn((total_qo, nhead, qk_head_dim), dtype=dtype)

        out_ref = torch_mla_extend(
            q,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=dtype,
        )

        # #triton version
        # prefix_indptr = kv_indptr - qo_indptr
        # tmp = kv_indptr[1:] - seq_lens_qo
        # tmp_inpptr, _ = torch.concat([kv_indptr[1:], tmp]).sort()
        # prefix_kv_indices = kv_indices.tensor_split(tmp_inpptr.tolist())
        # extend_kv_indices = torch.concat(
        #     [el for i, el in enumerate(prefix_kv_indices) if i % 2 == 1]
        # )
        # prefix_kv_indices = torch.concat(
        #     [el for i, el in enumerate(prefix_kv_indices) if i % 2 == 0]
        # )
        # extend_kvc = torch.index_select(kv_buffer, 0, extend_kv_indices)
        # out_triton = torch.empty((total_qo, nhead, v_head_dim), dtype=dtype).fill_(-1)
        # _, us_triton = run_perftest(
        #     mla_extend_ref.extend_attention_fwd,
        #     q,
        #     extend_kvc,
        #     extend_kvc[..., :kv_lora_rank],
        #     out_triton,
        #     kv_buffer,
        #     kv_buffer[..., :kv_lora_rank],
        #     qo_indptr,
        #     prefix_indptr,
        #     prefix_kv_indices,
        #     None,
        #     None,
        #     max_seqlen_qo,
        #     sm_scale,
        #     num_iters=5,
        # )
        # checkAllclose(
        #     out_ref,
        #     out_triton,
        #     msg=f"mla_prefill-absorb    [torch vs    triton]:{us_torch:>8.2f} us vs {us_triton:>8.2f} us......",
        # )

        out_asm = torch.zeros((total_qo, nhead, v_head_dim), dtype=dtype).fill_(-1)
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
            out_ref,
            out_asm,
            msg=f"mla_prefill-absorb    [torch vs aiter_asm]: {us_asm:>8.2f} us......",
        )
        return us_asm

    us_asm = None
    # if batch_size * ctx_lens * nhead < 32 * 8192 * 16:
    #     us_asm = test_absorb_prefill()
    torch.cuda.empty_cache()

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
    out_ref = torch_mla_extend(
        q,
        kv_buffer,
        qo_indptr,
        kv_indptr,
        kv_indices,
        sm_scale,
        kv_lora_rank,
        qk_rope_head_dim,
        is_causal=True,
        dtype=dtype,
    )

    # Triton implementation
    # if mtp == 1:
    #     if qk_head_dim != v_head_dim:
    #         out_triton = q.new_empty((total_q, nhead, v_head_dim)).fill_(-1)
    #     else:
    #         out_triton = torch.empty_like(q)

    #     num_kv_splits = 16
    #     attn_logits = torch.empty(
    #         (total_q, nhead, num_kv_splits, v_head_dim + 1),
    #         dtype=dtypes.fp32,
    #     )
    #     _, us_ref = run_perftest(
    #         mla_decode_ref.decode_attention_fwd,
    #         q,
    #         kv_buffer,
    #         kv_buffer[..., :kv_lora_rank],
    #         out_triton,
    #         kv_indptr,
    #         kv_indices,
    #         attn_logits,
    #         num_kv_splits,
    #         sm_scale,
    #         num_iters=5,
    #     )
    #     # logits_ref, lse_ref = attn_logits.split([v_head_dim, 1], dim=-1)
    #     # logits_ref = rearrange(logits_ref, "bs h sp d -> bs sp h d")
    #     # lse_ref = rearrange(lse_ref, "bs h sp d -> bs sp h d")
    #     checkAllclose(
    #         out_ref,
    #         out_triton,
    #         msg=f"mla_decode-absorb    [golden vs    triton]:{us_torch_decode:>8.2f} us vs {us_ref:>8.2f} us......",
    #     )

    # aiter implementation
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    out_asm = torch.zeros((total_q, nhead, v_head_dim), dtype=dtype).fill_(-1)
    # max_cu_num = 400
    # batch_split_table = torch.empty(
    #     (max_cu_num), dtype=torch.int32, device="cuda"
    # )
    # split_table = torch.empty(
    #     (max_cu_num), dtype=torch.int32, device="cuda"
    # )
    # num_kv_splits_indptr = torch.empty(
    #     (batch_size + 1), dtype=torch.int32, device="cuda"
    # )
    # kv_seq_les = torch.empty(
    #     (batch_size + 1), dtype=torch.int32, device="cuda"
    # )
    # num_splits = torch.empty(
    #     (1), dtype=torch.int32, device="cuda"
    # )
    #
    # # if varlen == False or mtp == 1 or batch_size < 32 or kv_indptr[batch_size] < 128 * 160:
    # cu_num = torch.ones((1), dtype=torch.int32, device="cuda")
    # if varlen == False or mtp == 1 or kv_indptr[-1] < 16 * 128:
    #     split_table = None
    #     batch_split_table = None
    #     num_kv_splits_indptr = None
    # else:
    #     # num_kv_splits, num_kv_splits_indptr, batch_split_table, split_table, cu_num = aiter.mla.get_meta_param_balanced(
    #     aiter.get_mla_metadata_impl(
    #         kv_indptr,
    #         num_kv_splits_indptr,
    #         batch_split_table,
    #         split_table,
    #         num_splits,
    #     )
    #     if num_kv_splits_indptr[0] == -1:
    #         num_kv_splits_indptr=None
    #         batch_split_table=None
    #         split_table=None

    gpu = torch.cuda.current_device()
    device_properties = torch.cuda.get_device_properties(gpu)
    cu_num = device_properties.multi_processor_count

    # 128 here is the maxmium len in packed_qo (qolen*#heads) can handled by mla main kernel
    # It would be more decent to query this value from aiter.
    max_qo_tiles_per_batch = int(math.ceil(torch.max(seq_lens_qo).item() * nhead / 128))

    # aiter implementation
    # the tensor's meaning please refer aiter/ops/attention.py
    work_meta_data = torch.empty([10], dtype=torch.uint64, device="cuda")
    work_indptr = torch.empty([cu_num + 1], dtype=torch.int32, device="cuda")
    work_info_set = torch.empty(
        [batch_size * max_qo_tiles_per_batch * cu_num, 8],
        dtype=torch.int32,
        device="cuda",
    ).fill_(-1)
    reduce_indptr = torch.empty(
        [batch_size * max_qo_tiles_per_batch + 1], dtype=torch.int32, device="cuda"
    )
    reduce_final_map = torch.empty(
        [batch_size * max_qo_tiles_per_batch, 2], dtype=torch.int32, device="cuda"
    )
    reduce_partial_map = torch.empty(
        [batch_size * max_qo_tiles_per_batch * cu_num], dtype=torch.int32, device="cuda"
    )

    split_params = {
        "kv_granularity": max(page_size, 16),
        "max_seqlen_qo": max_seqlen_qo,
        "uni_seqlen_qo": mtp,
        "fast_mode": 1,
    }
    aiter.get_mla_metadata_v1(
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
        split_params=split_params,
    )

    # print(work_meta_data)
    # print(work_indptr)
    # print(work_info_set)
    # print(reduce_indptr)
    # print(reduce_final_map)
    # print(reduce_partial_map)

    # import pdb; pdb.set_trace()
    # work_meta_data_2     = torch.empty([10], dtype=torch.uint64, device="cuda")
    # work_indptr_2        = torch.empty([81], dtype=torch.int32, device="cuda")
    # work_info_set_2      = torch.empty([batch_size + 80, 8], dtype=torch.int32, device="cuda")
    # reduce_indptr_2      = torch.empty([batch_size * 1 + 1], dtype=torch.int32, device="cuda")
    # reduce_final_map_2   = torch.empty([batch_size * 2, 2], dtype=torch.int32, device="cuda")
    # reduce_partial_map_2 = torch.empty([batch_size * 80], dtype=torch.int32, device="cuda")
    #
    # aiter.get_mla_metadata_v2(
    #     qo_indptr,
    #     kv_indptr,
    #     nhead // nhead_kv,
    #     nhead_kv,
    #     True,
    #     work_meta_data_2,
    #     work_info_set_2,
    #     work_indptr_2,
    #     reduce_indptr_2,
    #     reduce_final_map_2,
    #     reduce_partial_map_2,
    #     # num_reduce_tile,
    # )
    # import pdb; pdb.set_trace()

    (attn_logits, attn_lse), us_asm_decode = run_perftest(
        aiter.mla.mla_decode_fwd_dispatch,
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
        # work_indptr=work_indptr,
        work_info_set=work_info_set,
        reduce_indptr=reduce_indptr,
        reduce_final_map=reduce_final_map,
        reduce_partial_map=reduce_partial_map,
    )

    out_ref_asm = torch.empty_like(out_asm)
    aiter.mla.mla_decode_fwd_dispatch(
        q,
        kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
        out_ref_asm,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_lens,
        max_seqlen_qo,
        sm_scale,
    )
    # print(f"{out_ref.view(total_q, -1)=}")
    # print(f"{out_asm.view(total_q, -1)=}")
    # checkAllclose(logits_ref, attn_logits,
    #               msg=f'attn_logits [golden vs aiter_asm]')
    # checkAllclose(lse_ref, attn_lse,
    #               msg=f'attn_lse    [golden vs aiter_asm]')
    flops = mtp * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
    bytes = (
        total_kv * nhead_kv * qk_head_dim + total_q * nhead * (qk_head_dim + v_head_dim)
    ) * (torch.finfo(dtype).bits // 8)
    err = checkAllclose(
        out_ref_asm,
        out_asm,
        msg=f"mla_decode-absorb    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
    )
    return {
        "prefill:ck_192": us_aiter,
        "prefill:asm_576": us_asm,
        "decode:flops": flops,
        "decode:bytes": bytes,
        "decode:err": err,
        "decode:asm_576": us_asm_decode,
        "decode:TFLOPS": flops / us_asm_decode / 1e6,
        "decode:TB/s": bytes / us_asm_decode / 1e6,
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
    default=[4096, 8192, 16384], #
    help="""Context length.
    e.g.: -c 21""",
)
parser.add_argument(
    "-b",
    "--batchSize",
    type=int,
    nargs="*",
    default=[128], # [41],
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
