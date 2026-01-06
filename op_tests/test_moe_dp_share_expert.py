# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from aiter.test_common import checkAllclose, run_perftest, benchmark
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.int4_utils import *
from aiter import ActivationType, QuantType
from aiter.fused_moe_dp_shared_expert import (
    torch_moe,
    moe_sorting,
    fused_moe_dp_share_expert,
)
import pandas as pd
import aiter
import argparse


@benchmark()
def test_dp_shared_expert_moe(
    token_num,
    model_dim,
    inter_dim,
    share_expert,
    dp_size,
    quant_type=QuantType.per_Token,
    use_smoothquant=True,
    dtype=dtypes.bf16,
    q_dtype=dtypes.i8,
):
    device = torch.device("cuda")
    E = share_expert
    hidden_states = torch.randn((token_num, model_dim), device=device, dtype=dtype)
    w1 = torch.randn((E, inter_dim * 2, model_dim), device=device, dtype=dtype) / 10.0
    w2 = torch.randn((E, model_dim, inter_dim), device=device, dtype=dtype)

    torch_quant = aiter.get_torch_quant(quant_type)
    w1_q, w1_scale = torch_quant(w1, quant_dtype=q_dtype)
    w2_q, w2_scale = torch_quant(w2, quant_dtype=q_dtype)

    if use_smoothquant:
        sm1_scale = torch.randn((E, 1, model_dim), device=device, dtype=torch.float32)
        sm2_scale = torch.randn((E, 1, inter_dim), device=device, dtype=torch.float32)
    else:
        sm1_scale = None
        sm2_scale = None

    topk_ids_list = [list(range(E)) for i in range(token_num)]
    topk_ids = torch.tensor(topk_ids_list, dtype=dtypes.i32, device=device)
    topk_weights = torch.empty((token_num, E), dtype=dtypes.fp32, device=device)
    share_expert_score = 1.0
    topk_weights.fill_(share_expert_score)

    ref = torch_moe(
        hidden_states,
        w1_q,
        w2_q,
        topk_weights,
        topk_ids,
        w1_scale,
        w2_scale,
        sm1_scale,
        sm2_scale,
    )

    w1_q = shuffle_weight(w1_q, layout=(16, 16))
    w2_q = shuffle_weight(w2_q, layout=(16, 16))
    time_list = []
    moe_buf = torch.zeros_like(hidden_states)
    for rank in range(dp_size):
        moe_buf_tmp = torch.zeros_like(hidden_states)
        res, avg_t = run_perftest(
            fused_moe_dp_share_expert,
            hidden_states,
            w1_q,
            w2_q,
            quant_type=quant_type,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=sm1_scale,
            a2_scale=sm2_scale,
            dtype=dtype,
            dp_size=dp_size,
            dp_rank=rank,
            # moe_buf = moe_buf_tmp, # you can use no-shared expert result here, it will atomic add to it
        )
        moe_buf += res
        time_list.append(avg_t)

    avg_t = max(time_list)
    err = checkAllclose(ref, moe_buf, rtol=1e-2, atol=1e-2)
    return {"us": avg_t, "err": err}


list_M = [1, 4, 8, 16, 32, 64, 128, 192, 256, 384, 512, 1024, 8192]
list_E = [2]
list_dim = [(5120, 1536)]
list_q_dtype = [dtypes.i8, dtypes.fp8]

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-dim",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="""Model dimension.
    e.g.: -dim 6144,4096""",
)

parser.add_argument(
    "-t",
    "--tokenNum",
    type=int,
    nargs="?",
    const=None,
    default=None,
    help="""Number of tokens.
    e.g.: -t 1024""",
)

parser.add_argument(
    "-q",
    "--quant",
    type=int,
    choices=range(len(list_q_dtype)),
    default=0,
    help="""select quantization type:
    0 : aiter.QuantType.per_Token, dtypes.i8, dtypes.i8  # a8w8
    1: aiter.QuantType.per_Token, dtypes.fp8, dtypes.fp8  # a8w8
    """,
)
parser.add_argument(
    "-e",
    "--expert",
    type=int,
    default=8,
    help="""Number of experts.
    e.g.: -e 8""",
)

parser.add_argument(
    "-s",
    "--smoothquant",
    action="store_true",
    help="""use smoothquant.""",
)

args = parser.parse_args()

if args.dim is not None:
    list_dim = [args.dim]

if args.tokenNum is not None:
    list_M = [args.tokenNum]

if args.expert is not None:
    list_E = [args.expert]

if args.quant is not None:
    list_q_dtype = [list_q_dtype[args.quant]]

df = []
for q_dtype in list_q_dtype:
    for model_dim, inter_dim in list_dim:
        for E in list_E:
            for M in list_M:
                ret = test_dp_shared_expert_moe(
                    token_num=M,
                    model_dim=model_dim,
                    inter_dim=inter_dim,
                    share_expert=E,
                    dp_size=8,
                    q_dtype=q_dtype,
                    use_smoothquant=args.smoothquant,
                )
                df.append(ret)
    df = pd.DataFrame(df)
    df_md = df.to_markdown(index=False)
    aiter.logger.info("moe_dp_share_expert summary (markdown):\n%s", df_md)
