# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Graph time of the FlyDSL DCP mailbox and optional vLLM baselines.

Run on one node with 8 GPUs:

    WORLD=8 python3 op_tests/op_benchmarks/flydsl/bench_dcp_mailbox.py
    VLLM_BASELINE=1 TOKENS=8,16 WORLD=8 python3 .../bench_dcp_mailbox.py

The vLLM arm requires commit a2bcb5657a881a002e79d0adab3c6b5e39436121
(``nightly-rocm100-a2bcb565...``). Its ``dcp_a2a_lse_reduce`` is the
pack -> ``all_to_all_single`` -> unpack/combine chain.
"""

import os
import statistics

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aiter.ops.flydsl.dcp_mailbox import (
    CombineWorkspace,
    GatherWorkspace,
    dcp_lse_combine,
    dcp_quant_query_gather,
    dcp_query_gather,
)


class _Group:
    def __init__(self, cpu_group, rank, world):
        self.cpu_group = cpu_group
        self.device_group = dist.group.WORLD
        self.rank_in_group = rank
        self.world_size = world


def _standard_gather(query, world):
    """Match GroupCoordinator's all-gather along the head dimension."""
    shape = tuple(query.shape)
    gathered = torch.empty(
        (world * shape[0],) + shape[1:], dtype=query.dtype, device=query.device
    )
    dist.all_gather_into_tensor(gathered, query)
    return (
        gathered.reshape((world,) + shape)
        .movedim(0, 1)
        .reshape(shape[0], world * shape[1], shape[2])
    )


def _graph_us(fn, sync_group, reps=29, runs=10):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    # Every rank waits on its peers, so align the ranks before the timed replay.
    dist.barrier(group=sync_group)
    samples = []
    for _ in range(runs):
        dist.barrier(group=sync_group)
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        g.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / reps)
        dist.barrier(group=sync_group)
    return statistics.median(samples), min(samples), max(samples)


def _fmt(samples):
    median, low, high = samples
    return f"{median:.2f} [{low:.2f}, {high:.2f}]"


def worker(rank, world):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ.setdefault("MASTER_PORT", "29742")
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    group = _Group(dist.new_group(backend="gloo"), rank, world)
    try:
        baseline = os.environ.get("VLLM_BASELINE") == "1"
        if baseline:
            from vllm.config import VllmConfig, set_current_vllm_config
            from vllm.model_executor.layers.attention.mla_attention import (
                _DecodeConcatQuantFP8,
            )
            from vllm.model_executor.layers.quantization.utils.quant_utils import (
                GroupShape,
            )
            from vllm.v1.attention.ops.dcp import dcp_a2a_lse_reduce

            with set_current_vllm_config(VllmConfig()):
                quant = _DecodeConcatQuantFP8(
                    static=True,
                    group_shape=GroupShape.PER_TENSOR,
                    compile_native=True,
                )

        hl = 12
        token_counts = [
            int(value) for value in os.environ.get("TOKENS", "8,16").split(",")
        ]
        gen = torch.Generator(device="cuda")
        gen.manual_seed(200 + rank)
        for tokens in token_counts:
            max_tokens = max(tokens, 64)
            if baseline:
                dim = 512
                out = torch.randn(
                    tokens,
                    world * hl,
                    dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=gen,
                )
                lse = torch.randn(
                    tokens,
                    world * hl,
                    device="cuda",
                    dtype=torch.float32,
                    generator=gen,
                )
                combine = _graph_us(
                    lambda out=out, lse=lse: dcp_a2a_lse_reduce(
                        out, lse, group, return_lse=True
                    ),
                    group.cpu_group,
                )
                query = torch.randn(
                    tokens,
                    hl,
                    dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=gen,
                )
                gather = _graph_us(
                    lambda query=query: _standard_gather(query, world),
                    group.cpu_group,
                )
                nope_dim, pe_dim = 512, 64
                nope = torch.randn(
                    tokens,
                    hl,
                    nope_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=gen,
                )
                pe = torch.randn(
                    tokens,
                    hl,
                    pe_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=gen,
                )
                scale = torch.tensor([2.0], device="cuda", dtype=torch.float32)
                quant_gather = _graph_us(
                    lambda nope=nope, pe=pe, scale=scale: _standard_gather(
                        quant(nope, pe, scale), world
                    ),
                    group.cpu_group,
                )
                if rank == 0:
                    print(
                        f"DCP_VLLM_BASELINE combine ranks={world} tokens={tokens} "
                        f"heads={hl} dim={dim} a2a={_fmt(combine)} us",
                        flush=True,
                    )
                    print(
                        f"DCP_VLLM_BASELINE gather ranks={world} tokens={tokens} "
                        f"heads={hl} dim={dim} all_gather={_fmt(gather)} us",
                        flush=True,
                    )
                    print(
                        f"DCP_VLLM_BASELINE quant_gather ranks={world} "
                        f"tokens={tokens} heads={hl} dim={nope_dim + pe_dim} "
                        f"quant_all_gather={_fmt(quant_gather)} us",
                        flush=True,
                    )
                continue

            for dim in (128, 512, 576):
                cws = CombineWorkspace(group, torch.device("cuda"), hl, dim, max_tokens)
                out = torch.randn(
                    tokens,
                    world * hl,
                    dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                    generator=gen,
                )
                lse = torch.randn(
                    tokens,
                    world * hl,
                    device="cuda",
                    dtype=torch.float32,
                    generator=gen,
                )
                combine = _graph_us(
                    lambda w=cws, o=out, s=lse: dcp_lse_combine(
                        w, o, s, return_lse=True
                    ),
                    group.cpu_group,
                )
                if rank == 0:
                    print(
                        f"DCP_MAILBOX_BENCH combine ranks={world} tokens={tokens} "
                        f"heads={hl} dim={dim} mailbox={_fmt(combine)} us",
                        flush=True,
                    )
                cws.close()

            dim = 512
            gws = GatherWorkspace(
                group,
                torch.device("cuda"),
                hl,
                dim,
                torch.bfloat16,
                world * hl,
                max_tokens,
            )
            query = torch.randn(
                tokens,
                hl,
                dim,
                device="cuda",
                dtype=torch.bfloat16,
                generator=gen,
            )
            gather = _graph_us(
                lambda gws=gws, query=query: dcp_query_gather(gws, query),
                group.cpu_group,
            )
            if rank == 0:
                print(
                    f"DCP_MAILBOX_BENCH gather ranks={world} tokens={tokens} "
                    f"heads={hl} dim={dim} mailbox={_fmt(gather)} us",
                    flush=True,
                )
            gws.close()

            nope_dim, pe_dim = 512, 64
            qws = GatherWorkspace(
                group,
                torch.device("cuda"),
                hl,
                nope_dim + pe_dim,
                torch.float8_e4m3fn,
                world * hl,
                max_tokens,
            )
            nope = torch.randn(
                tokens,
                hl,
                nope_dim,
                device="cuda",
                dtype=torch.bfloat16,
                generator=gen,
            )
            pe = torch.randn(
                tokens,
                hl,
                pe_dim,
                device="cuda",
                dtype=torch.bfloat16,
                generator=gen,
            )
            scale = torch.tensor([2.0], device="cuda", dtype=torch.float32)
            quant_gather = _graph_us(
                lambda qws=qws, nope=nope, pe=pe, scale=scale: (
                    dcp_quant_query_gather(qws, nope, pe, scale)
                ),
                group.cpu_group,
            )
            if rank == 0:
                print(
                    f"DCP_MAILBOX_BENCH quant_gather ranks={world} tokens={tokens} "
                    f"heads={hl} dim={nope_dim + pe_dim} "
                    f"mailbox={_fmt(quant_gather)} us",
                    flush=True,
                )
            qws.close()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    world = int(os.environ.get("WORLD", "8"))
    mp.spawn(worker, args=(world,), nprocs=world, join=True)
