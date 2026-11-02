# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import logging
import os
from multiprocessing import Pool, freeze_support, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.communication_op import (
    tensor_model_parallel_fused_qknorm_allreduce,
    tensor_model_parallel_fused_qknorm_allreduce_rope,
)
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    graph_capture,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import checkAllclose, perftest

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)


def barrier_before_teardown():
    """Align all ranks before tearing down the distributed groups.

    Drain this rank's GPU work, then join a barrier so no rank starts freeing
    IPC buffers / destroying process groups while a peer is still inside a
    NCCL / custom-all-reduce collective -- that race intermittently hangs when
    these comm UTs run back-to-back in CI. No-op if dist is uninitialized.
    """
    if not dist.is_initialized():
        return
    torch.cuda.synchronize()
    get_tp_group().barrier()
    torch.cuda.synchronize()


COS_SIN_MAX_POS = 16384


def _run_qknorm_case(
    rankID,
    tp_size,
    case_idx,
    shape,
    dtype,
    head_size,
    rotary_dim,
    withGraph,
    graphs,
):
    """Run one fused qknorm all-reduce case on an initialized rank and check
    its q/k/v outputs against the host reference.

    Each rank regenerates every rank's inputs from the same ``case_idx`` seed,
    because the reference norm needs the variance summed over all ranks; the
    reference is computed on the device and only scalars go back to the
    parent. In graph mode the captured graph and every tensor it reads or
    writes are appended to ``graphs`` and must outlive the whole sweep: custom
    all-reduce caches peer IPC addresses by local pointer, so freeing one and
    capturing a later case at the same address would replay with stale peers.
    """
    token_num, hidden_dim_q, hidden_dim_k, hidden_dim_v = shape
    hidden_dim = hidden_dim_q + hidden_dim_k + hidden_dim_v
    gen = torch.Generator(device="cuda").manual_seed(case_idx)
    cos_sin_cache = torch.randn(
        (COS_SIN_MAX_POS, rotary_dim), dtype=dtype, device="cuda", generator=gen
    )
    positions = torch.arange(token_num - 1, -1, -1, dtype=torch.long, device="cuda")
    qkv_ins = []
    q_ws = []
    k_ws = []
    for _ in range(tp_size):
        qkv_ins.append(
            torch.randn(
                (token_num, hidden_dim), dtype=dtype, device="cuda", generator=gen
            )
        )
        q_ws.append(
            torch.randn((hidden_dim_q,), dtype=dtype, device="cuda", generator=gen)
        )
        k_ws.append(
            torch.randn((hidden_dim_k,), dtype=dtype, device="cuda", generator=gen)
        )
    q_refs, k_refs, v_refs = qknorm_allreduce_host(qkv_ins, q_ws, k_ws)
    q_ref = apply_neox_rope_host(
        q_refs[rankID], cos_sin_cache, positions, head_size, rotary_dim
    )
    k_ref = apply_neox_rope_host(
        k_refs[rankID], cos_sin_cache, positions, head_size, rotary_dim
    )
    v_ref = v_refs[rankID]
    qkv_in = qkv_ins[rankID]
    q_w = q_ws[rankID]
    k_w = k_ws[rankID]

    method = (
        tensor_model_parallel_fused_qknorm_allreduce_rope
        if rotary_dim > 0
        else tensor_model_parallel_fused_qknorm_allreduce
    )

    if withGraph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
            q_out, k_out, v_out = method(
                qkv_in,
                q_w,
                k_w,
                cos_sin_cache,
                positions,
                head_size,
                rotary_dim,
                1e-6,
            )
        q_out.fill_(0)
        k_out.fill_(0)
        v_out.fill_(0)
        graphs.append(
            (graph, qkv_in, q_w, k_w, cos_sin_cache, positions, q_out, k_out, v_out)
        )

        @perftest()
        def run_ca():
            graph.replay()

        _, us = run_ca()
    else:

        @perftest()
        def run_ca(qkv_in, q_w, k_w):
            return method(
                qkv_in,
                q_w,
                k_w,
                cos_sin_cache,
                positions,
                head_size,
                rotary_dim,
                1e-6,
            )

        (q_out, k_out, v_out), us = run_ca(qkv_in, q_w, k_w)

    msg = (
        f"test_qknorm_allreduce: rank={rankID} {shape=} {dtype=} {withGraph=} "
        f"{us:>8.2f}"
    )
    err = max(
        checkAllclose(q_ref, q_out.to(q_ref), msg=msg),
        checkAllclose(k_ref, k_out.to(k_ref), msg=msg),
        checkAllclose(v_ref, v_out.to(v_ref), msg=msg),
    )
    return {"us": us, "err": err}


def qknorm_allreduce_sweep(
    tp_size,
    pp_size,
    rankID,
    cases,
    head_size,
    rotary_dim,
    withGraph=False,
    distributed_init_method: str | None = None,
):
    """Run every ``(shape, dtype)`` case on one rank inside a single
    distributed init.

    Setting up the TP group dominates a single case (~30 s at TP8 vs.
    microseconds of kernel time), so the group is created and torn down once.
    Results are returned rather than asserted here: every rank must walk the
    full case list so the collectives stay aligned across ranks.
    """
    device = torch.device(f"cuda:{rankID}")
    torch.cuda.set_device(device)
    # init
    logger.info(f"RANK: {rankID} {tp_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
    )
    ensure_model_parallel_initialized(tp_size, pp_size)

    # warmup and align all gpu
    group = get_tp_group().device_group
    dist.all_reduce(torch.zeros(1).cuda(), group=group)
    torch.cuda.synchronize()

    graphs = []
    results = [
        _run_qknorm_case(
            rankID,
            tp_size,
            case_idx,
            shape,
            dtype,
            head_size,
            rotary_dim,
            withGraph,
            graphs,
        )
        for case_idx, (shape, dtype) in enumerate(cases)
    ]

    # destroy
    if dist.is_initialized():
        barrier_before_teardown()
        destroy_model_parallel()
        destroy_distributed_environment()
        graphs.clear()
        torch.cuda.empty_cache()
    return results


def apply_neox_rope_host(x, cos_sin_cache, positions, head_size, rotary_dim):
    # x: [token_num, hidden_dim]
    # cos_sin_cache: [max_pos, rotary_dim]
    # positions: [token_num]
    token_num = x.shape[0]
    x = x.view(token_num, -1, head_size)  # [token_num, nheads, head_size]
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:]
    cos_sin = cos_sin_cache[positions].to(x.dtype)  # [token_num, rotary_dim]
    cos, sin = cos_sin.chunk(2, dim=-1)
    cos = cos.unsqueeze(-2).to(x.dtype)  # [token_num, 1, rotary_dim/2]
    sin = sin.unsqueeze(-2).to(x.dtype)  # [token_num, 1, rotary_dim/2]
    x1, x2 = x_rot.chunk(2, dim=-1)  # (token_num, nheads, rotary_dim/2) * 2
    out = torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
    if x_pass.numel() > 0:
        out = torch.cat((out, x_pass), dim=-1)
    out = out.view(token_num, -1)
    return out


def qknorm_allreduce_host(qkv_ins, q_ws, k_ws, eps=1e-6):
    tp_size = len(qkv_ins)
    token_num = qkv_ins[0].shape[0]
    hidden_dim_q = q_ws[0].shape[0]
    hidden_dim_k = k_ws[0].shape[0]
    hidden_dim_v = qkv_ins[0].shape[1] - hidden_dim_q - hidden_dim_k
    qs = []
    ks = []
    vs = []
    q_vars = []
    k_vars = []
    for i in range(tp_size):
        qkv_in = qkv_ins[i]
        q, k, v = qkv_in.split([hidden_dim_q, hidden_dim_k, hidden_dim_v], dim=1)
        vs.append(v)
        orig_dtype = q.dtype
        q = q.to(torch.float32)
        k = k.to(torch.float32)
        qs.append(q)
        ks.append(k)
        q_var = q.pow(2).mean(dim=-1, keepdim=True)
        k_var = k.pow(2).mean(dim=-1, keepdim=True)
        q_vars.append(q_var)
        k_vars.append(k_var)
    device = qkv_ins[0].device
    q_var_all = torch.zeros((token_num, 1), dtype=torch.float32, device=device)
    k_var_all = torch.zeros((token_num, 1), dtype=torch.float32, device=device)
    for i in range(tp_size):
        q_var_all += q_vars[i]
        k_var_all += k_vars[i]
    q_var_all = q_var_all / tp_size
    k_var_all = k_var_all / tp_size
    q_outs = []
    k_outs = []
    for i in range(tp_size):
        q = qs[i]
        k = ks[i]
        qw = q_ws[i]
        kw = k_ws[i]
        q = (q * torch.rsqrt(q_var_all + eps) * qw).to(orig_dtype)
        k = (k * torch.rsqrt(k_var_all + eps) * kw).to(orig_dtype)
        q_outs.append(q)
        k_outs.append(k)
    return q_outs, k_outs, vs


def test_qknorm_allreduce(tp_size, pp_size, cases, head_size, rotary_dim, withGraph):
    """Sweep ``(shape, dtype)`` ``cases`` on one TP group and return one
    summary row per case."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "49373"
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp_size) as pool:
        rets = [
            pool.apply_async(
                qknorm_allreduce_sweep,
                args=(
                    tp_size,
                    pp_size,
                    rank,
                    cases,
                    head_size,
                    rotary_dim,
                    withGraph,
                    init_method,
                ),
            )
            for rank in range(tp_size)
        ]
        per_rank = [el.get() for el in rets]
    rows = []
    for i, (shape, dtype) in enumerate(cases):
        all_us = [results[i]["us"] for results in per_rank]
        rows.append(
            {
                "tp_size": tp_size,
                "shape": shape,
                "dtype": dtype,
                "withGraph": withGraph,
                "min_us": min(all_us),
                "max_us": max(all_us),
                "err": max(results[i]["err"] for results in per_rank),
            }
        )
    return rows


l_dtype = ["fp16", "bf16"]
l_shape = [(1, 3072, 512, 1024), (2, 3072, 512, 1024), (16, 3072, 512, 1024)]

# MiniMax-M2 per-rank QKV geometry at TP in {2, 4, 8}, swept across
# token_num to exercise the grid-strided outer loop above kMaxBlocks (=80).
l_T_widen = [1, 16, 32, 64, 80, 128, 256, 512, 1024, 2048]
SHAPE_BY_TP = {
    2: [(T, 3072, 512, 512) for T in l_T_widen],
    4: [(T, 1536, 256, 256) for T in l_T_widen],
    8: [(T, 768, 128, 128) for T in l_T_widen],
}


parser = argparse.ArgumentParser(description="config input of test")
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="data type",
)
parser.add_argument(
    "-s",
    "--shape",
    type=dtypes.str2tuple,
    nargs="?",
    const=None,
    default=None,
    help="shape. e.g. -s 1,3072,512,1024",
)
parser.add_argument(
    "-g",
    "--with-graph",
    type=lambda x: str(x).lower() in ["true", "1", "yes"],
    default=True,
    help="use CUDA graph (default: True). e.g. -g true or -g false",
)
parser.add_argument(
    "--tp-sizes",
    type=lambda s: [int(x) for x in s.split(",") if x.strip()],
    default=[8],
    help="comma-separated TP sizes from {2, 4, 8} (default 8). "
    "Non-default switches to the multi-T SHAPE_BY_TP matrix.",
)


try:
    import pytest

    @pytest.mark.parametrize(
        "tp,shape,dtype_str",
        [
            (tp, shape, d)
            for tp in (2, 4, 8)
            for shape in SHAPE_BY_TP[tp]
            for d in ("bf16", "fp16")
        ],
    )
    def test_widen_multi_t(tp, shape, dtype_str):
        if torch.cuda.device_count() < tp:
            pytest.skip(f"requires >= {tp} GPUs (have {torch.cuda.device_count()})")
        (ret,) = test_qknorm_allreduce(
            tp, 1, [(shape, dtypes.d_dtypes[dtype_str])], 128, 64, withGraph=True
        )
        assert (
            ret["err"] < 1e-2
        ), f"qknorm err={ret['err']} at tp={tp} shape={shape} dtype={dtype_str}"

except ImportError:
    pass


if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    if args.dtype is None:
        l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
    else:
        l_dtype = [dtypes.d_dtypes[args.dtype]]
    # One TP group per tp size; every (dtype, shape) is swept inside it.
    df = []
    for tp in args.tp_sizes:
        if args.shape is not None:
            shapes = [args.shape]
        elif args.tp_sizes == [8]:
            shapes = l_shape
        else:
            shapes = SHAPE_BY_TP.get(tp, l_shape)
        cases = [(shape, dtype) for dtype in l_dtype for shape in shapes]
        df.extend(
            test_qknorm_allreduce(tp, 1, cases, 128, 64, withGraph=args.with_graph)
        )
    df = pd.DataFrame(df)
    show_cols = [
        "tp_size",
        "shape",
        "dtype",
        "withGraph",
        "min_us",
        "max_us",
        "err",
    ]
    show_cols = [c for c in show_cols if c in df.columns]
    logger.info(
        "fused qknorm allreduce summary (markdown):\n%s",
        df[show_cols].to_markdown(index=False),
    )
