# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import aiter
import argparse
import itertools
import numpy as np
import pandas as pd
import pytest
import random
import torch

from aiter import dtypes
from aiter.test_common import benchmark, run_perftest

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)


def generate_inputs(
    batch_size: int,
    max_qlen: int,
    ctx_lens: int,
    block_size: int,
    varlen: bool = False,
    is_prefill: bool = False,
):
    seq_lens_qo = torch.zeros(batch_size, dtype=torch.int)
    seq_lens_kv = torch.zeros(batch_size, dtype=torch.int)
    if varlen:
        for i in range(batch_size):
            seq_lens_qo[i] = random.uniform(5, max_qlen)
            seq_lens_kv[i] = (
                seq_lens_qo[i] if is_prefill else random.uniform(5, ctx_lens)
            )
    else:
        seq_lens_qo.fill_(max_qlen)
        if is_prefill:
            seq_lens_kv.fill_(max_qlen)
        else:
            seq_lens_kv.fill_(ctx_lens)

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)

    kv_page_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_page_indptr[1 : batch_size + 1] = torch.cumsum(
        (seq_lens_kv + block_size - 1) // block_size, dim=0
    )
    return qo_indptr, kv_page_indptr, seq_lens_qo, seq_lens_kv


# TODO:
# 1. add reference implementation
# 2. add pa/mla_prefill/mla_decode functional tests
# 3. add performance benchmarks
@benchmark()
def test_attn_ps(
    batch_size: int,
    num_heads: tuple[int, int],
    qo_indptr: torch.Tensor,
    kv_page_indptr: torch.Tensor,
    seq_lens_qo: torch.Tensor,
    seq_lens_kv: torch.Tensor,
    block_size: int,
    tile_q: int = 256,
    tile_kv: int = 128,
    is_causal: bool = True,
    is_prefill: bool = False,
    load_metadata: bool = False,
    dump_metadata: bool = False,
    attn_func: callable = None,
    attn_kwargs: dict = {},
    reduce_func: callable = None,
    reduce_kwargs: dict = {},
):
    ret = {}

    num_head_q, num_head_kv = num_heads
    assert num_head_q % num_head_kv == 0
    gqa_ratio = num_head_q // num_head_kv

    max_qlen = seq_lens_qo.max().item()

    qhead_granularity = gqa_ratio
    qlen_granularity = tile_q // qhead_granularity  # prefill: tile_q, decode: max_qlen
    kvlen_granularity = max(tile_kv, block_size)  # prefill: tile_kv, decode: block_size
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
    work_metadata_ptrs = torch.empty(work_meta_data_size, dtype=work_meta_data_type)
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type)
    work_info = torch.empty(work_info_size, dtype=work_info_type)
    reduce_indptr = torch.empty(reduce_indptr_size, dtype=reduce_indptr_type)
    reduce_final_map = torch.empty(reduce_final_map_size, dtype=reduce_final_map_type)
    reduce_partial_map = torch.empty(
        reduce_partial_map_size, dtype=reduce_partial_map_type
    )

    metadata_map = {
        "qo_indptr": qo_indptr,
        "kv_page_indptr": kv_page_indptr,
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
            print(f"==>load {name} shape {meta.shape} from {file_name}:\n{meta}")
    else:
        qo_indptr_cpu = qo_indptr.to("cpu")
        kv_page_indptr_cpu = kv_page_indptr.to("cpu")
        seq_lens_kv_cpu = seq_lens_kv.to("cpu")
        # warmup
        # TODO: use dummy data
        aiter.get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_page_indptr_cpu,
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
        # benchmark
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        aiter.get_ps_metadata_v1(
            qo_indptr_cpu,
            kv_page_indptr_cpu,
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

        ret["us_metadata"] = us_metadata

    if dump_metadata:
        # TODO: enhance metadata print
        for name, meta in metadata_map.items():
            file_name = f"{name}.bin"
            torch.set_printoptions(threshold=99999999, linewidth=120)
            print(
                f"==>dump {name} shape {meta.shape} to {file_name}:\n{meta}", flush=True
            )
            meta.cpu().numpy().astype(np.uint32).tofile(file_name)

    if attn_func is not None:
        attn_output, us_attn = run_perftest(
            attn_func,
            qo_indptr,
            kv_page_indptr,
            seq_lens_qo,
            seq_lens_kv,
            block_size,
            is_causal,
            **attn_kwargs,
        )
        ret["us_attn"] = us_attn

    if reduce_func is not None:
        reduce_output, us_reduce = run_perftest(
            reduce_func,
            **reduce_kwargs,
        )
        ret["us_reduce"] = us_reduce

    return ret


@pytest.mark.parametrize("batch_size", [416])
@pytest.mark.parametrize("num_heads", [(10, 1)])
@pytest.mark.parametrize("qlen", [4])
@pytest.mark.parametrize("ctx_lens", [30720])
@pytest.mark.parametrize("block_size", [256])
@pytest.mark.parametrize("varlen", [False])
@pytest.mark.parametrize("is_causal", [True])
@pytest.mark.parametrize("is_prefill", [False])
def test_pa_decode_ps(
    batch_size: int,
    num_heads: tuple[int, int],
    qlen: int,
    ctx_lens: int,
    block_size: int,
    varlen: bool,
    is_causal: bool,
    is_prefill: bool,
    load_metadata: bool = False,
    dump_metadata: bool = False,
):
    qo_indptr, kv_page_indptr, seq_lens_qo, seq_lens_kv = generate_inputs(
        batch_size=batch_size,
        max_qlen=max_qlen,
        ctx_lens=ctx_lens,
        block_size=block_size,
        varlen=varlen,
        is_prefill=is_prefill,
    )
    # BD PA case
    seq_lens_kv = torch.tensor([10240] * 415 + [30720], dtype=torch.int)
    kv_page_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_page_indptr[1 : batch_size + 1] = torch.cumsum(
        (seq_lens_kv + block_size - 1) // block_size, dim=0
    )

    ret = test_attn_ps(
        batch_size=batch_size,
        num_heads=num_heads,
        qo_indptr=qo_indptr,
        kv_page_indptr=kv_page_indptr,
        seq_lens_qo=seq_lens_qo,
        seq_lens_kv=seq_lens_kv,
        block_size=block_size,
        is_causal=is_causal,
        is_prefill=is_prefill,
        load_metadata=load_metadata,
        dump_metadata=dump_metadata,
    )
    return ret


@pytest.mark.parametrize("batch_size", [4, 16])
@pytest.mark.parametrize("num_heads", [(1, 1)])
@pytest.mark.parametrize("max_qlen", [16384])
@pytest.mark.parametrize("ctx_lens", [16384])
@pytest.mark.parametrize("block_size", [1])
@pytest.mark.parametrize("varlen", [False])
@pytest.mark.parametrize("is_causal", [True])
@pytest.mark.parametrize("is_prefill", [True])
def test_mla_prefill_ps(
    batch_size: int,
    num_heads: tuple[int, int],
    max_qlen: int,
    ctx_lens: int,
    block_size: int,
    varlen: bool,
    is_causal: bool,
    is_prefill: bool,
    load_metadata: bool = False,
    dump_metadata: bool = False,
):
    qo_indptr, kv_page_indptr, seq_lens_qo, seq_lens_kv = generate_inputs(
        batch_size=batch_size,
        max_qlen=max_qlen,
        ctx_lens=ctx_lens,
        block_size=block_size,
        varlen=varlen,
        is_prefill=is_prefill,
    )
    ret = test_attn_ps(
        batch_size=batch_size,
        num_heads=num_heads,
        qo_indptr=qo_indptr,
        kv_page_indptr=kv_page_indptr,
        seq_lens_qo=seq_lens_qo,
        seq_lens_kv=seq_lens_kv,
        block_size=block_size,
        is_causal=is_causal,
        is_prefill=is_prefill,
        load_metadata=load_metadata,
        dump_metadata=dump_metadata,
    )
    return ret


# TODO: add ut test
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_heads", [(1, 1)])
@pytest.mark.parametrize("max_qlen", [16384])
@pytest.mark.parametrize("ctx_lens", [16384])
@pytest.mark.parametrize("block_size", [1])
@pytest.mark.parametrize("varlen", [False])
@pytest.mark.parametrize("is_causal", [True])
@pytest.mark.parametrize("is_prefill", [False])
def test_mla_decode_ps():
    pass


# TODO: add ut test
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_heads", [(1, 1)])
@pytest.mark.parametrize("max_qlen", [16384])
@pytest.mark.parametrize("ctx_lens", [16384])
@pytest.mark.parametrize("block_size", [1])
@pytest.mark.parametrize("varlen", [False])
@pytest.mark.parametrize("is_causal", [True])
@pytest.mark.parametrize("is_prefill", [False])
def test_sparse_mla_decode_ps():
    pass


l_batch_size = [1]
l_num_heads = [(1, 1), (2, 2)]
l_qlen = [522]
l_ctx_len = [522]
l_block_size = [1, 256]
l_varlen = [False]
l_is_causal = [True]
l_is_prefill = [True, False]


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
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
        type=dtypes.str2tuple,
        default=None,
        help="""Number of heads.
        e.g. -n 8,1""",
    )
    parser.add_argument(
        "-q",
        "--qlen",
        type=int,
        choices=l_qlen,
        default=None,
        help="""Query length.
        e.g. -q 1""",
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
        "-blk",
        "--block_size",
        type=int,
        default=None,
        help="""Block size.
        e.g.: -blk 1""",
    )
    parser.add_argument(
        "-tq",
        "--tile_q",
        type=int,
        default=256,
        help="""Size of Q Tile.
        e.g.: -tq 256""",
    )
    parser.add_argument(
        "-tk",
        "--tile_kv",
        type=int,
        default=128,
        help="""Size of KV Tile.
        e.g.: -tk 128""",
    )
    parser.add_argument(
        "--varlen",
        action="store_true",
        help="""variable kv seqlens per batch. Default: False.
        --varlen # True""",
    )
    parser.add_argument(
        "--is_causal",
        action="store_true",
        help="""True for causal mask, False for none. Default: False.
        --is_causal # True""",
    )
    parser.add_argument(
        "--is_prefill",
        action="store_true",
        help="""True for prefill, False for decode. Default: False.
        --is_prefill # True""",
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.batch_size is not None:
        l_batch_size = [args.batch_size]
    if args.num_heads is not None:
        l_num_heads = [args.num_heads]
    if args.qlen is not None:
        l_qlen = [args.qlen]
    if args.ctx_len is not None:
        l_ctx_len = [args.ctx_len]
    if args.block_size is not None:
        l_block_size = [args.block_size]
    if args.varlen is not None:
        l_varlen = [args.varlen]
    # if args.is_causal is not None:
    #     l_is_causal = [args.is_causal]
    # if args.is_prefill is not None:
    #     l_is_prefill = [args.is_prefill]

    df = []

    ret = test_pa_decode_ps()
    df.append(ret)

    ret = test_mla_prefill_ps()
    df.append(ret)

    ret = test_mla_decode_ps()
    df.append(ret)

    ret = test_sparse_mla_decode_ps()
    df.append(ret)

    # other cases
    for is_prefill in l_is_prefill:
        for is_causal in l_is_causal:
            for (
                batch_size,
                num_heads,
                max_qlen,
                ctx_len,
                block_size,
                varlen,
            ) in itertools.product(
                l_batch_size, l_num_heads, l_qlen, l_ctx_len, l_block_size, l_varlen
            ):
                ret = test_attn_ps(
                    batch_size=batch_size,
                    num_heads=num_heads,
                    max_qlen=max_qlen,
                    ctx_lens=ctx_len,
                    block_size=block_size,
                    is_prefill=is_prefill,
                    is_causal=is_causal,
                    varlen=varlen,
                    load_metadata=args.load_metadata,
                    dump_metadata=args.dump_metadata,
                )
                df.append(ret)

    df = pd.DataFrame(df)
    aiter.logger.info(f"summary:\n{df}")
    df.to_csv("attn_ps.csv")
