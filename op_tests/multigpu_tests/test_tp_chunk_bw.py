# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Chunk-size sweep for dest-sharded TP all-gather bandwidth.

Times bulk (one slab) vs chunk_rows. Chunk path uses in-kernel epoch
handshake; host zero is only between cases, not inside the CUDA graph.
Reports XGMI inject GB/s = (world-1) * m_local * (H + H/32) / time.

Run with::

    MORI_SOCKET_IFNAME=lo MORI_SHMEM_HEAP_SIZE=40G PYTHONPATH="$PWD" \\
      torchrun --standalone --nproc-per-node=8 \\
      op_tests/multigpu_tests/test_tp_chunk_bw.py
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")
os.environ.setdefault("MORI_SOCKET_IFNAME", "lo")

import flydsl.expr as fx
import mori.shmem as ms
import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
from aiter.ops.flydsl.kernels.mega_moe.tp_chunk_payload import (
    compile_tp_chunk_push,
    run_tp_chunk_push,
)
from aiter.ops.flydsl.kernels.mega_moe.tp_incremental_push import (
    TpIncrementalWorkspace,
    compile_tp_incremental_push,
    run_tp_incremental_push,
)
from aiter.ops.flydsl.kernels.mega_moe.tp_incremental_schedule import (
    num_publish_chunks,
)

SUPPORTED_GFX = ["gfx950"]
MODEL_DIM = 7168


def _setup_dist():
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return rank, world, device


def _cleanup():
    try:
        ms.shmem_finalize()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _agree(ok: bool, device, msg: str):
    flag = torch.tensor([int(bool(ok))], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if int(flag.item()) == 0:
        raise AssertionError(msg)
    return True


def _barrier():
    torch.cuda.synchronize()
    ms.shmem_barrier_all()


def _fx_stream():
    return fx.Stream(torch.cuda.current_stream().cuda_stream)


def _capture(body):
    _barrier()
    body()
    _barrier()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=stream):
        body()
    for _ in range(5):
        graph.replay()
    _barrier()
    return graph


def _time_graph(graph, iters, device):
    _barrier()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    local_ms = start.elapsed_time(end) / iters
    mean = torch.tensor(local_ms, dtype=torch.float64, device=device)
    maximum = mean.clone()
    dist.all_reduce(mean, op=dist.ReduceOp.SUM)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    return float(mean.item() / dist.get_world_size()), float(maximum.item())


def _row_bytes(model_dim: int) -> int:
    return int(model_dim) + int(model_dim) // 32


def _remote_bytes(world: int, m_local: int, model_dim: int) -> int:
    return (int(world) - 1) * int(m_local) * _row_bytes(model_dim)


def _gbps(nbytes: int, ms: float) -> float:
    return nbytes / (ms * 1e-3) / 1e9


def _all_gather_uint8(local: torch.Tensor, npes: int):
    payload = local.contiguous().view(torch.uint8)
    gathered = torch.empty(
        (npes * payload.shape[0], payload.shape[1]),
        dtype=torch.uint8,
        device=payload.device,
    )
    dist.all_gather_into_tensor(gathered, payload)
    return gathered


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[64, 256],
        help="m_local (tokens per TP rank)",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=int,
        nargs="*",
        default=[MODEL_DIM],
        help="hidden / model_dim",
    )
    parser.add_argument(
        "--chunk-rows",
        type=int,
        nargs="*",
        default=[4, 8, 16, 32, 64, 128, 256],
        help="rows per ordered all-gather send",
    )
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()

    rank, world, device = _setup_dist()
    try:
        gfx = get_gfx()
        if gfx not in SUPPORTED_GFX:
            if rank == 0:
                aiter.logger.warning("tp chunk bw unsupported on %s; skipping", gfx)
            return
        if world < 2:
            if rank == 0:
                aiter.logger.warning("need world_size>=2; got %s", world)
            return

        max_m_local = max(args.batch)
        handshake_src = torch.zeros(1, dtype=torch.int32, device=device)
        handshake_dst = torch.empty(world, dtype=torch.int32, device=device)
        rows = []
        for model_dim in args.mnk:
            if model_dim % 32:
                raise ValueError(f"model_dim={model_dim} must be a multiple of 32")
            workspace = TpIncrementalWorkspace(
                rank=rank,
                npes=world,
                max_m_local=max_m_local,
                model_dim=model_dim,
                num_experts=1,
                device=device,
            )
            if rank == 0:
                aiter.logger.info(
                    "compile chunk/bulk gather model_dim=%s producers=32", model_dim
                )
            compile_tp_chunk_push(
                npes=world,
                model_dim=model_dim,
                row_major=False,
                num_producers=32,
                num_waves=4,
            )
            compile_tp_incremental_push(
                npes=world,
                model_dim=model_dim,
                row_major=False,
                num_producers=32,
                num_waves=4,
            )
            for m_local in args.batch:
                generator = torch.Generator(device=device).manual_seed(0 + rank)
                x = torch.randn(
                    (m_local, model_dim),
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                x_fp8, x_scale = per_1x32_mx_quant(x, quant_mode="fp8")
                nccl_a = _all_gather_uint8(x_fp8, world)
                nccl_scale = _all_gather_uint8(x_scale, world)
                nbytes = _remote_bytes(world, m_local, model_dim)

                def _sync():
                    workspace.zero_handshake()
                    dist.all_gather_into_tensor(handshake_dst, handshake_src)

                def _run_bulk():
                    _sync()
                    run_tp_incremental_push(
                        workspace, x_fp8, x_scale, None, m_local, _fx_stream()
                    )

                workspace.reset()
                _run_bulk()
                torch.cuda.synchronize()
                bulk_ok = torch.equal(
                    workspace.rx_u8[: world * m_local], nccl_a
                ) and torch.equal(workspace.rx_scale[: world * m_local], nccl_scale)
                _agree(bulk_ok, device, f"bulk gather mismatch m_local={m_local}")

                bulk_ms = _time_graph(_capture(_run_bulk), args.iters, device)
                bulk_gbps = _gbps(nbytes, bulk_ms[0])
                rows.append(
                    {
                        "gfx": gfx,
                        "tp": world,
                        "m_local": m_local,
                        "model_dim": model_dim,
                        "chunk_rows": m_local,
                        "chunks": 1,
                        "kind": "bulk",
                        "us": bulk_ms[0] * 1e3,
                        "GB/s": bulk_gbps,
                        "vs bulk": 1.0,
                    }
                )
                if rank == 0:
                    aiter.logger.info(
                        "bulk m_local=%s %.3f us %.2f GB/s",
                        m_local,
                        bulk_ms[0] * 1e3,
                        bulk_gbps,
                    )

                for chunk_rows in args.chunk_rows:
                    n_chunks = num_publish_chunks(m_local, chunk_rows)

                    def _run_chunk(cr=chunk_rows):
                        run_tp_chunk_push(
                            workspace,
                            x_fp8,
                            x_scale,
                            m_local,
                            _fx_stream(),
                            chunk_rows=cr,
                        )

                    workspace.reset()
                    _run_chunk()
                    torch.cuda.synchronize()
                    chunk_ok = torch.equal(
                        workspace.rx_u8[: world * m_local], nccl_a
                    ) and torch.equal(
                        workspace.rx_scale[: world * m_local], nccl_scale
                    )
                    _agree(
                        chunk_ok,
                        device,
                        f"chunk gather mismatch m_local={m_local} "
                        f"chunk_rows={chunk_rows}",
                    )
                    chunk_ms = _time_graph(_capture(_run_chunk), args.iters, device)
                    chunk_gbps = _gbps(nbytes, chunk_ms[0])
                    rows.append(
                        {
                            "gfx": gfx,
                            "tp": world,
                            "m_local": m_local,
                            "model_dim": model_dim,
                            "chunk_rows": chunk_rows,
                            "chunks": n_chunks,
                            "kind": "chunk",
                            "us": chunk_ms[0] * 1e3,
                            "GB/s": chunk_gbps,
                            "vs bulk": chunk_gbps / bulk_gbps if bulk_gbps else 0.0,
                        }
                    )
                    if rank == 0:
                        aiter.logger.info(
                            "chunk m_local=%s chunk_rows=%s C=%s %.3f us "
                            "%.2f GB/s (%.0f%% bulk)",
                            m_local,
                            chunk_rows,
                            n_chunks,
                            chunk_ms[0] * 1e3,
                            chunk_gbps,
                            100.0 * chunk_gbps / bulk_gbps if bulk_gbps else 0.0,
                        )

        if rank == 0:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "tp chunk allgather bw (markdown):\n%s",
                df.to_markdown(index=False, floatfmt=".3f"),
            )
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
