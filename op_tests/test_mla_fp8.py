# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import sys
sys.path.insert(0, '/workspace/aiter')
import aiter
from aiter.test_common import checkAllclose, benchmark, run_perftest
from aiter import dtypes
import random
import itertools
import argparse
from typing import Tuple, List, Dict
import numpy as np

import logging

logging.basicConfig(
    format='%(asctime)s %(filename)s:%(lineno)d %(levelname)s %(message)s',
    level=logging.INFO
)

logging.info(f'aiter path {aiter.__path__}')
logging.info(f'aiter spec {aiter.__spec__}')

torch.set_default_device("cuda")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
# setup_seed(1)


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, 
                   k: int = 5, 
                   thresholds: List[float] = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1]) -> Dict:
    """
    Compare two numpy arrays and compute various difference metrics.
    
    Args:
        arr1: First input array (float32)
        arr2: Second input array (float32)
        k: Number of top differences to return
        thresholds: List of thresholds for difference magnitude analysis
        
    Returns:
        Dictionary containing:
        - top_k_diff: Top k absolute differences with their positions
        - threshold_stats: Count and percentage of differences above each threshold
        - nan_info: Information about NaN values in input arrays
    """
    # Check input shapes
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape")
    
    result = {
        'top_k_diff': [],
        'threshold_stats': [],
        'nan_info': {}
    }

    # Check for NaN values
    nan_mask1 = np.isnan(arr1)
    nan_mask2 = np.isnan(arr2)
    
    if np.any(nan_mask1):
        result['nan_info']['arr1_nan_count'] = np.sum(nan_mask1)
        result['nan_info']['arr1_nan_positions'] = np.argwhere(nan_mask1)
        logging.info(f"Warning: arr1 contains {result['nan_info']['arr1_nan_count']} NaN values")
    
    if np.any(nan_mask2):
        result['nan_info']['arr2_nan_count'] = np.sum(nan_mask2)
        result['nan_info']['arr2_nan_positions'] = np.argwhere(nan_mask2)
        logging.info(f"Warning: arr2 contains {result['nan_info']['arr2_nan_count']} NaN values")
    
    # Compute absolute differences
    diff = np.abs(arr1 - arr2)
    total_elements = arr1.size

    max_diff_thr = diff / (1.0 + np.abs(arr2))
    max_diff_thr = max_diff_thr.max()
    logging.info(f"diff.abs.max={diff.max()}")
    logging.info(f"max_diff_thr={max_diff_thr}")

    # Find top k differences
    flat_diff = diff.flatten()
    top_k_indices = np.argpartition(flat_diff, -k)[-k:]
    top_k_indices = top_k_indices[np.argsort(-flat_diff[top_k_indices])]

    # Convert flat indices to multi-dimensional indices
    orig_indices = np.unravel_index(top_k_indices, diff.shape)
    for i in range(k):
        idx = tuple(dim[i] for dim in orig_indices)
        result['top_k_diff'].append({
            'value': diff[idx],
            'position': idx,
            'arr1_value': arr1[idx],
            'arr2_value': arr2[idx]
        })

    # Compute threshold statistics
    for i in range(len(thresholds) - 1):
        lower = thresholds[i]
        upper = thresholds[i + 1]
        mask = (diff >= lower) & (diff < upper)
        count = np.sum(mask)
        result['threshold_stats'].append({
            'range': f"[{lower:.1e}, {upper:.1e})",
            'count': count,
            'percentage': 100 * count / total_elements
        })
    
    # Handle values above the largest threshold
    mask = diff >= thresholds[-1]
    count = np.sum(mask)
    result['threshold_stats'].append({
        'range': f">={thresholds[-1]:.1e}",
        'count': count,
        'percentage': 100 * count / total_elements
    })

    logging.info("\nTop differences:")
    for item in result['top_k_diff']:
        logging.info(f"Position {item['position']}: arr1 = {arr1[item['position']]:.6f}, arr2 = {arr2[item['position']]:.6f}, Diff = {item['value']:.6f}")

    logging.info("\nThreshold statistics:")
    for stat in result['threshold_stats']:
        logging.info(f"{stat['range']}: {stat['count']} ({stat['percentage']:.2f}%)")
    
    logging.info("\nNaN info:")
    logging.info(result['nan_info'])

    return result

def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool=False) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    if use_fp8:
        assert cos_diff < 3e-2
    else:
        assert cos_diff < 1e-5


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

    out = out / l.transpose(0,1).unsqueeze(-1)
    
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
        o, lse = ref_masked_attention(q,
                                      k,
                                      v,
                                      sm_scale,
                                      dtype,
                                      is_causal=is_causal,
                                      is_fp8=is_fp8,
                                      q_scale=q_scale,
                                      kv_scale=kv_scale)
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
            seq_lens_kv[i] = random.uniform(10, ctx_lens)
            seq_lens_qo[i] = max(
                min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
            )
    else:
        seq_lens_kv.fill_(ctx_lens)
        seq_lens_qo.fill_(ctx_lens)

    seq_lens_kv = torch.tensor([3819,9978,784,530,8062,1390,287,1008,5090,5304,7396,2288,2104,4063,3644,5091,6470,4732,7237,430,2777,956,1357,5478,1292,521,6802,1347,2388,5062,443,8560,5049,7235,927,9580,623,4913,2511,8120,1638,4859,600,7289,8278,6693,136,1021,1465,5859,1278,7123,7839,2459,1090,6333,812,9358,6345,8616,2313,6115,6059,4963, 12343, 213, 143, 12312, 12345, 3215, 4444, 5325, 2132, 123, 456, 2135, 135, 2564, 5465, 4362], device="cuda")
    seq_lens_kv = seq_lens_kv[:batch_size]
    aiter.logger.info(f"seq_lens_kv: {seq_lens_kv.numel()}")


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
        is_causal=True,
        dtype=dtype,
    )

    q_fp8, q_scale = aiter.per_tensor_quant(q, quant_dtype=torch.float8_e4m3fnuz)
    q_scale = q_scale.to(torch.float)
    # q_fp8 = q.to(torch.float8_e4m3fnuz)
    # q_scale = torch.ones([1], dtype=torch.float, device="cuda")

    kv_buffer_fp8 = kv_buffer.to(torch.float8_e4m3fnuz)
    # kv_scale = torch.ones([batch_size], dtype=torch.float, device="cuda")
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


    # aiter implementation
    work_meta_data     = torch.empty([10], dtype=torch.uint64, device="cuda")
    work_indptr        = torch.empty([81], dtype=torch.int32, device="cuda")
    work_info_set      = torch.empty([batch_size + 80, 8], dtype=torch.int32, device="cuda")
    reduce_indptr      = torch.empty([batch_size + 1], dtype=torch.int32, device="cuda")
    reduce_final_map   = torch.empty([batch_size, 2], dtype=torch.int32, device="cuda")
    reduce_partial_map = torch.empty([batch_size * 80], dtype=torch.int32, device="cuda")
    # num_reduce_tile    = torch.empty([1], dtype=torch.int32, device="cuda")
    
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
    print(work_indptr)
    print(work_info_set[:work_indptr[-1]])
    print(reduce_indptr)
    print(reduce_final_map)
    print(reduce_partial_map)

    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=dtype).fill_(-1)

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
    # out_asm = out_asm[:total_q]

    # import pdb; pdb.set_trace()

    # print(f"{out_ref.view(total_q, -1)=}")
    # print(f"{out_asm.view(total_q, -1)=}")
    # checkAllclose(logits_ref, attn_logits,
    #               msg=f'attn_logits [golden vs aiter_asm]')
    # checkAllclose(lse_ref, attn_lse, msg="attn_lse    [golden vs aiter_asm]")
    flops = mtp * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
    aiter.logger.info(f"mtp: {mtp}; total_kv: {total_kv}; nhead: {nhead}; {qk_head_dim}/{v_head_dim}")
    bytes = (
        total_kv * nhead_kv * qk_head_dim + total_q * nhead * (qk_head_dim + v_head_dim)
    ) * (torch.finfo(dtype).bits // 8)
    err = checkAllclose(
        out_ref,
        out_asm,
        msg=f"mla_decode-absorb    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
    )
    err_fp8 = checkAllclose(
        out_ref_fp8,
        out_asm,
        msg=f"mla_decode-absorb    [golden fp8 vs aiter_asm]: {us_asm_decode:>8.2f} us......",
    )

    result = compare_arrays(out_asm.float().cpu().numpy(), out_ref.float().cpu().numpy())
    logging.info(f'compare: {result}')

    # cal_diff(out_ref, out_asm, "out", True)

    return {
        "decode:flops": flops,
        "decode:bytes": bytes,
        "decode:err vs float": err,
        "decode:err vs fp8": err_fp8,
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
    # default=[28, 512, 1023, 4888, 12800], #
    default=[16], #
    help="""Context length.
    e.g.: -c 21""",
)
parser.add_argument(
    "-b",
    "--batchSize",
    type=int,
    nargs="*",
    # default=[i for i in range(80, 128, 16)], # [41],
    default=[80], # [41],
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


