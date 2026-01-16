# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import aiter
import argparse
import itertools
import numpy as np
import pandas as pd
import random
import torch

from aiter import dtypes
from aiter import per_tensor_quant
from aiter.test_common import benchmark, checkAllclose, perftest, run_perftest

from typing import Tuple, Optional

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


def calculate_pass_rate(df):
    if "acc result" not in df.columns:
        return

    num_tests = df["acc result"].value_counts().sum()
    if "passed" in df["acc result"].value_counts():
        num_passed = df["acc result"].value_counts()["passed"]
    else:
        num_passed = 0
    if "warning" in df["acc result"].value_counts():
        num_warning = df["acc result"].value_counts()["warning"]
    else:
        num_warning = 0
    if "failed" in df["acc result"].value_counts():
        num_failed = df["acc result"].value_counts()["failed"]
    else:
        num_failed = 0
    aiter.logger.info(
        f"\033[32mpassed {num_passed}/{num_tests}({num_passed / num_tests * 100:.2f}%) \
        \033[33mwarning {num_warning}/{num_tests}({num_warning / num_tests * 100:.2f}%) \
        \033[31mfailed {num_failed}/{num_tests}({num_failed / num_tests * 100:.2f}%) \033[0m"
    )


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype,
    is_causal=True,
    is_fp8_q=False,
    is_fp8_kvc=False,
    q_scale=None,
    kv_scale=None,
):

    if is_fp8_q and q_scale is not None:
        scale *= q_scale
    if is_fp8_kvc and kv_scale is not None:
        scale *= kv_scale

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

    if is_fp8_q:
        attn_weights_fp8 = attn_weights_exp.to(dtype)
        attn_weights_exp = attn_weights_fp8.to(torch.float)

    out = torch.einsum("hqk,khd->qhd", attn_weights_exp.float(), value.float())

    out = out / l.transpose(0, 1).unsqueeze(-1)

    if is_fp8_kvc and kv_scale is not None:
        out *= kv_scale
    return out.to(dtype), lse


def torch_mla_extend(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_block * block_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    softmax_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    is_causal=True,
    q_scale=None,
    kv_scale=None,
):
    is_fp8_q = q.dtype == dtypes.fp8
    is_fp8_kvc = kvc_cache.dtype == dtypes.fp8

    if is_fp8_q:
        q = q.to(torch.float)

    if is_fp8_kvc:
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
            softmax_scale,
            dtype,
            is_causal=is_causal,
            is_fp8_q=is_fp8_q,
            is_fp8_kvc=is_fp8_kvc,
            q_scale=q_scale,
            kv_scale=kv_scale,
        )
        os.append(o)
        lses.append(lse)
    o = torch.concat(os)
    lse = torch.concat(lses).transpose(0, 1)
    return o, lse


@perftest()
def run_aiter_mla_prefill_asm(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    output: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_page_indices: torch.Tensor,
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    max_seqlen_q: int,
    is_causal: bool,
    softmax_scale: float,
    logits: torch.Tensor,
    attn_lse: torch.Tensor,
    q_scale: Optional[torch.Tensor] = None,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    aiter.mla_prefill_ps_asm_fwd(
        Q,
        K,
        V,
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        work_indptr,
        work_info,
        max_seqlen_q,
        softmax_scale,
        is_causal,
        logits,
        attn_lse,
        output,
        q_scale,
        k_scale,
        v_scale,
    )
    return output, logits, attn_lse


@perftest()
def run_aiter_mla_reduce(
    logits: torch.Tensor,
    attn_lse: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    tile_q: int,
    output: torch.Tensor,
    final_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    aiter.mla_reduce_v1(
        logits,
        attn_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        tile_q,
        output,
        final_lse,
    )
    return output, attn_lse


@benchmark()
def test_mla_prefill(
    ctx_lens: int,
    batch_size: int,
    num_head: int,
    qk_head_dim: int,
    v_head_dim: int,
    dtype: torch.dtype,
    kv_dtype: torch.dtype,
    block_size: int,
    varlen: bool = False,
    is_causal: bool = True,
    load_metadata: Optional[bool] = False,
    dump_metadata: Optional[bool] = False,
    profile_ps: Optional[bool] = False,
    skip_reference: Optional[bool] = False,
):
    ret = {}
    out_dtype = torch.bfloat16
    seed = 0
    device = "cuda:0"
    torch.set_default_device(device)
    num_head_q = num_head
    num_head_kv = num_head
    assert num_head_q % num_head_kv == 0
    gqa_ratio = num_head_q // num_head_kv
    softmax_scale = 1.0 / (qk_head_dim**0.5)

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    if varlen:
        for i in range(batch_size):
            seq_lens_kv[i] = max(
                min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
            )
    else:
        seq_lens_kv.fill_(ctx_lens)
    seq_lens_qo = seq_lens_kv.clone()
    max_qlen = seq_lens_qo.max().item()
    max_kvlen = seq_lens_kv.max().item()

    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    actual_blocks = (seq_lens_kv + block_size - 1) // block_size
    kv_indptr[1 : batch_size + 1] = torch.cumsum(actual_blocks, dim=0)
    num_blocks = kv_indptr[-1].item()
    kv_indices = torch.randint(0, num_blocks, (num_blocks,), dtype=torch.int)

    num_tokens = qo_indptr[-1].item()
    Q_bf16 = torch.randn((num_tokens, num_head_q, qk_head_dim), dtype=torch.bfloat16)
    # block_size = 1
    K_bf16 = torch.randn((num_blocks, num_head_kv, qk_head_dim), dtype=torch.bfloat16)
    V_bf16 = K_bf16[:, :, :v_head_dim].contiguous()

    q_quant, q_scale = per_tensor_quant(Q_bf16, quant_dtype=dtype)
    k_quant, k_scale = per_tensor_quant(K_bf16, quant_dtype=kv_dtype)
    v_quant, v_scale = per_tensor_quant(V_bf16, quant_dtype=kv_dtype)

    tile_q = 256
    tile_kv = 128
    qhead_granularity = gqa_ratio
    qlen_granularity = tile_q // qhead_granularity
    # TODO: enhance pre-allocation, current too loose for large context length
    kvlen_granularity = max(tile_kv, block_size)
    (
        (work_meta_data_size, work_meta_data_type),
        (work_indptr_size, work_indptr_type),
        (work_info_size, work_info_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_ps_metadata_info_v1(
        batch_size=batch_size,
        num_head_k=num_head_kv,
        max_qlen=max_qlen,
        qlen_granularity=qlen_granularity,
    )
    work_metadata_ptrs = torch.empty(
        work_meta_data_size, dtype=work_meta_data_type, device=device
    )
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device=device)
    work_info = torch.empty(work_info_size, dtype=work_info_type, device=device)
    reduce_indptr = torch.empty(
        reduce_indptr_size, dtype=reduce_indptr_type, device=device
    )
    reduce_final_map = torch.empty(
        reduce_final_map_size, dtype=reduce_final_map_type, device=device
    )
    reduce_partial_map = torch.empty(
        reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
    )

    metadata_map = {
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "seq_lens_kv": seq_lens_kv,
        "work_indptr": work_indptr,
        "work_info": work_info,
        "reduce_indptr": reduce_indptr,
        "reduce_final_map": reduce_final_map,
        "reduce_partial_map": reduce_partial_map,
    }

    if load_metadata:
        for name, meta in metadata_map.items():
            file_name = f"{name}.bin"
            shape = meta.shape
            array = np.fromfile(file_name, dtype=np.uint32)
            meta = torch.from_numpy(array).reshape(shape)
            torch.set_printoptions(threshold=999999, linewidth=120)
            print(f"==>load {name} from {file_name}:\n{meta}")
    else:
        qo_indptr_cpu = qo_indptr.to("cpu")
        kv_indptr_cpu = kv_indptr.to("cpu")
        seq_lens_kv_cpu = seq_lens_kv.to("cpu")
        # warmup for get_ps_metadata_v1
        aiter.get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_indptr_cpu,
            seq_lens_kv_cpu,
            gqa_ratio,
            num_head_kv,
            work_metadata_ptrs,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            qhead_granularity=qhead_granularity,
            qlen_granularity=qlen_granularity,
            kvlen_granularity=kvlen_granularity,
            block_size=block_size,
            is_causal=is_causal,
        )
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        aiter.get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_indptr_cpu,
            seq_lens_kv_cpu,
            gqa_ratio,
            num_head_kv,
            work_metadata_ptrs,
            work_indptr,
            work_info,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            qhead_granularity=qhead_granularity,
            qlen_granularity=qlen_granularity,
            kvlen_granularity=kvlen_granularity,
            block_size=block_size,
            is_causal=is_causal,
        )
        end_event.record()
        end_event.synchronize()
        us_metadata = start_event.elapsed_time(end_event) * 1000  # ms to us

    if dump_metadata:
        for name, meta in metadata_map.items():
            file_name = f"{name}.bin"
            torch.set_printoptions(threshold=99999999, linewidth=120)
            print(f"==>dump {name} shape {meta.shape} to {file_name}:\n{meta}")
            meta.cpu().numpy().astype(np.uint32).tofile(file_name)

    output = torch.empty((num_tokens, num_head_q, v_head_dim), dtype=torch.bfloat16)

    if profile_ps:
        # pre-allocate final and partial output & lse
        total_s, nhead, v_head_dim = output.shape

        tile_q = 256
        logits = torch.empty(
            (reduce_partial_map.size(0) * tile_q, nhead, v_head_dim),
            dtype=dtypes.fp32,
            device=device,
        )
        attn_lse = torch.empty(
            (reduce_partial_map.size(0) * tile_q, nhead),
            dtype=dtypes.fp32,
            device=device,
        )
        final_lse = torch.empty((total_s, nhead), dtype=dtypes.fp32, device=device)

        out_mla_prefill_asm, us_mla_prefill_asm = run_aiter_mla_prefill_asm(
            q_quant,
            k_quant,
            v_quant,
            output,
            qo_indptr,
            kv_indptr,
            kv_indices,
            work_indptr,
            work_info,
            max_qlen,
            is_causal,
            softmax_scale,
            logits,
            attn_lse,
            q_scale,
            k_scale,
            v_scale,
        )
        output, logits, attn_lse = out_mla_prefill_asm

        out_reduce, us_reduce = run_aiter_mla_reduce(
            logits,
            attn_lse,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            tile_q,
            output,
            final_lse,
        )
        output, final_lse = out_reduce
        output = output.view(total_s, nhead, v_head_dim)

        us_mla_prefill_ps = us_mla_prefill_asm + us_reduce
        # calculate mla_prefill_ps kernel tflops
        # for causal, only take the lower triangle(ops/2)
        g_div = 2 if is_causal else 1
        ops = (
            2.0
            * batch_size
            * num_head_q
            * ctx_len
            * (qk_head_dim * ctx_len + v_head_dim * ctx_len)
        ) / g_div
        tflops_mla_prefill_asm = ops / us_mla_prefill_asm / (1e6)
        # calulate reduce kernel bandwidth
        # input: fp32 partial_out & partial_lse + int32 reduce_indptr, reduce_final_map & reduce_partial_map
        # output: bf16 final_out & final_lse
        allocate_input_bytes = (
            logits.numel() * logits.element_size()
            + attn_lse.numel() * attn_lse.element_size()
            + reduce_indptr.numel() * reduce_indptr.element_size()
            + reduce_final_map.numel() * reduce_final_map.element_size()
            + reduce_partial_map.numel() * reduce_partial_map.element_size()
        )
        allocate_output_bytes = (
            output.numel() * output.element_size()
            + final_lse.numel() * final_lse.element_size()
        )
        allocate_bytes = allocate_input_bytes + allocate_output_bytes

        effective_final_tiles = torch.argmax(reduce_indptr).item()
        effective_partial_tiles = reduce_indptr[-1].item()
        effective_input_bytes = (
            effective_partial_tiles * qlen_granularity * num_head_q * (v_head_dim + 1)
            + (effective_final_tiles + 1)
            + (effective_final_tiles * 2)
            + effective_partial_tiles
        ) * 4
        effective_output_bytes = (
            effective_final_tiles * qlen_granularity * num_head_q * (v_head_dim + 1) * 2
        )
        effective_bytes = effective_input_bytes + effective_output_bytes
        print(
            f"effective_partial_tiles: {effective_partial_tiles}, allocate_partial_tiles: {reduce_partial_map.numel()}"
        )
        print(
            f"effective_final_tiles: {effective_final_tiles}, allocate_final_tiles: {reduce_final_map.numel()}"
        )
        print(
            f"effective_input_bytes: {effective_input_bytes}, allocate_input_bytes: {allocate_input_bytes}"
        )
        print(
            f"effective_output_bytes: {effective_output_bytes}, allocate_output_bytes: {allocate_output_bytes}"
        )
        print(f"effective_bytes: {effective_bytes}, allocate_bytes: {allocate_bytes}")

        reduce_bytes = effective_bytes
        bw_reduce = (reduce_bytes / 1e12) / (us_reduce / (1e6))
        # Store results
        ret["us_metadata"] = us_metadata
        ret["us_mla_prefill_ps"] = us_mla_prefill_ps
        ret["us_mla_prefill_asm"] = us_mla_prefill_asm
        ret["us_mla_prefill_asm_ratio"] = us_mla_prefill_asm / us_mla_prefill_ps
        ret["tflops_mla_prefill_asm"] = tflops_mla_prefill_asm
        ret["us_reduce"] = us_reduce
        ret["us_reduce_ratio"] = us_reduce / us_mla_prefill_ps
        ret["bw_reduce(TB/s)"] = bw_reduce if effective_final_tiles > 0 else 0
    else:
        _, us_aiter_asm = run_perftest(
            aiter.mla.mla_prefill_ps_fwd,
            q_quant,
            k_quant,
            v_quant,
            output,
            qo_indptr,
            kv_indptr,
            kv_indices,
            work_indptr,
            work_info,
            max_qlen,
            is_causal,
            reduce_indptr,
            reduce_final_map,
            reduce_partial_map,
            softmax_scale,
            q_scale,
            k_scale,
            v_scale,
        )

        ret["us_mla_prefill_ps"] = us_aiter_asm

    if not skip_reference:
        # TODO: optimize reference implementation(too slow for large context length)
        kv_buffer = K_bf16.view(-1, num_head_kv, qk_head_dim)
        out_ref, lse_ref = torch_mla_extend(
            Q_bf16,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            softmax_scale,
            kv_lora_rank=v_head_dim,
            qk_rope_head_dim=qk_head_dim - v_head_dim,
            dtype=out_dtype,
            is_causal=is_causal,
        )

        err = checkAllclose(
            out_ref,
            output,
            rtol=5e-2,
            atol=5e-2,
            msg="mla_prefill_ps    [torch vs aiter_asm]: us......",
        )
        if err == 0:
            status = "passed"
        elif 0 < err <= 0.05:
            status = "warning"
        else:
            status = "failed"
        ret["err fp8"] = err
        ret["acc result"] = status

    return ret


l_dtype = ["fp8"]
l_kv_dtype = ["fp8"]
l_num_heads = [1, 16]
l_ctx_len = [
    21,
    64,
    256,
    512,
    1200,
    3200,
    5200,
    8192,
    10000,
    16384,
    # 90000,
]
l_batch_size = [1, 4, 16]
l_block_size = [1]
l_varlen = [False]
l_causal = [True, False]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-qkh",
    "--qk_head_dim",
    type=int,
    default=192,
    help="""qk head dim = kv_lora_rank + qk_rope_head_dim.
    e.g.: -qh 192""",
)
parser.add_argument(
    "-vh",
    "--v_head_dim",
    type=int,
    default=128,
    help="""v head dim = kv_lora_rank.
    e.g.: -vh 128""",
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
    choices=["fp8"],
    nargs="*",
    default=["fp8"],
    help="""Data type of Q.
    e.g.: -d fp8""",
)
parser.add_argument(
    "-kvd",
    "--kv_dtype",
    type=str,
    choices=["fp8"],
    nargs="*",
    default=["fp8"],
    help="""Data type of KV.
    e.g.: -kvd fp8""",
)
parser.add_argument(
    "-c",
    "--ctx_len",
    type=int,
    default=None,
    help="""Context length(for prefill, qo_len = kv_len = context_len).
    e.g.: -c 21""",
)
parser.add_argument(
    "-b",
    "--batch_size",
    type=int,
    default=None,
    help="""Batch size.
    e.g.: -b 16""",
)
parser.add_argument(
    "-n",
    "--num_heads",
    type=int,
    default=None,
    help="""Number of heads(for mla prefill(MHA), num_head_q = num_head_kv).
    e.g.: -n 1""",
)
parser.add_argument(
    "--varlen",
    action="store_true",
    help="""variable kv seqlens per batch. Default: False.
    --varlen # True""",
)
parser.add_argument(
    "--causal",
    action="store_true",
    help="""enable causal mask. Default: False.
    --causal # True""",
)
parser.add_argument(
    "--load_metadata",
    action="store_true",
    help="""load metadata by metadata_map Default: False.
    --load_metadata # True""",
)
parser.add_argument(
    "--dump_metadata",
    action="store_true",
    help="""dump metadata by metadata_map. Default: False.
    --dump_metadata # True""",
)
parser.add_argument(
    "--profile",
    action="store_true",
    help="""Breakdown performance by each operation. Default: False.
    --profile # True""",
)
parser.add_argument(
    "--skip_reference",
    action="store_true",
    help="""skip reference implementation. Default: False.
    --skip_reference # True""",
)

args = parser.parse_args()
if args.dtype is not None:
    l_dtype = [dtypes.d_dtypes[key] for key in args.dtype]
if args.kv_dtype is not None:
    l_kvdtype = [dtypes.d_dtypes[key] for key in args.kv_dtype]
if args.num_heads is not None:
    l_num_heads = [args.num_heads]
if args.ctx_len is not None:
    l_ctx_len = [args.ctx_len]
if args.batch_size is not None:
    l_batch_size = [args.batch_size]
if args.block_size is not None:
    l_block_size = [args.block_size]
if args.varlen is not None:
    l_varlen = [args.varlen]
# if args.causal is not None:
#     l_causal = [args.causal]

if args.profile:
    l_ctx_len = [16384]
    l_batch_size = [4, 16]
    l_num_heads = [1]

df = []
for (
    is_causal,
    num_head,
    dtype,
    kv_dtype,
    ctx_len,
    batch_size,
    block_size,
    varlen,
) in itertools.product(
    l_causal,
    l_num_heads,
    l_dtype,
    l_kvdtype,
    l_ctx_len,
    l_batch_size,
    l_block_size,
    l_varlen,
):
    ret = test_mla_prefill(
        ctx_len,
        batch_size,
        num_head,
        args.qk_head_dim,
        args.v_head_dim,
        dtype,
        kv_dtype,
        block_size,
        varlen,
        is_causal,
        load_metadata=args.load_metadata,
        dump_metadata=args.dump_metadata,
        profile_ps=args.profile,
        skip_reference=args.skip_reference,
    )
    df.append(ret)
df = pd.DataFrame(df)
df_md = df.to_markdown(index=False)
aiter.logger.info("mla_prefill_ps summary (markdown):\n%s", df_md)
df.to_csv("mla_prefill_ps.csv")
calculate_pass_rate(df)
