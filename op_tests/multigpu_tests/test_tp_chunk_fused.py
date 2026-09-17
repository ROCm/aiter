# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Chunked TP all-gather + GEMM1: wait-last correctness, early-compute overlap bound.

Wait-last waits for the final send chunk so GEMM1 matches packed NCCL A.
Early-compute starts M-tile 0 after chunk 0; A for those tiles may be
incomplete. The dense rx slab must still match NCCL after the kernel.

Run with::

    MORI_SOCKET_IFNAME=lo MORI_SHMEM_HEAP_SIZE=40G PYTHONPATH="$PWD" \\
      torchrun --standalone --nproc-per-node=8 \\
      op_tests/multigpu_tests/test_tp_chunk_fused.py
"""

from __future__ import annotations

import argparse
import itertools
import os

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")
os.environ.setdefault("MORI_SOCKET_IFNAME", "lo")

import flydsl.expr as fx
import mori.shmem as ms
import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.mega_moe.gemm1 import gemm1_kernel
from aiter.ops.flydsl.kernels.mega_moe.quant import per_1x32_mx_quant
from aiter.ops.flydsl.kernels.mega_moe.tp_chunk_fused import run_tp_chunk_fused
from aiter.ops.flydsl.kernels.mega_moe.tp_incremental_push import TpIncrementalWorkspace
from aiter.ops.flydsl.kernels.mega_moe.tp_incremental_schedule import (
    make_tile_row_base,
    num_publish_chunks,
    pack_rows_by_sorted_ids,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import checkAllclose

SUPPORTED_GFX = ["gfx950"]
SORT_BLOCK_M = 32
TILE_N = 256
TILE_K = 256
CHUNK_ROWS = 32


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


def _all_gather_uint8(local: torch.Tensor, npes: int):
    payload = local.contiguous().view(torch.uint8)
    gathered = torch.empty(
        (npes * payload.shape[0], payload.shape[1]),
        dtype=torch.uint8,
        device=payload.device,
    )
    dist.all_gather_into_tensor(gathered, payload)
    return gathered


def _make_local_inputs(m_local, model_dim, experts, topk, rank, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn(
        (m_local, model_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    scores = torch.randn(
        (m_local, experts), dtype=torch.float32, device=device, generator=generator
    )
    values, ids = torch.topk(scores, topk, dim=-1)
    return (
        x.contiguous(),
        values.softmax(dim=-1).contiguous(),
        ids.to(torch.int32).contiguous(),
    )


def _make_w1(experts, model_dim, inter_dim, device):
    generator = torch.Generator(device=device).manual_seed(999)
    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1 = torch.randn(
        (experts, 2 * inter_dim, model_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w1.mul_(model_dim**-0.25)
    w1_q, w1_scale = quantize(w1, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(experts, 2 * inter_dim, model_dim // 2)
    w1_kernel = shuffle_weight_a16w4(w1_q, 16, True).contiguous()
    w1_scale_kernel = shuffle_scale_a16w4(w1_scale, experts, True).contiguous()
    return w1_kernel, w1_scale_kernel


def _alloc_out(num_valid, inter_dim, device):
    out = torch.zeros((num_valid, inter_dim), dtype=torch.float8_e4m3fn, device=device)
    scale_cols = (inter_dim // 32 + 7) // 8 * 8
    prows = ((num_valid + 255) // 256) * 256
    out_scale = torch.zeros(
        prows * scale_cols + inter_dim, dtype=torch.uint8, device=device
    )
    return out, out_scale


def _pad_dense(payload, scale):
    pad_row = torch.zeros(
        (1, payload.shape[1]), dtype=payload.dtype, device=payload.device
    )
    pad_scale = torch.zeros((1, scale.shape[1]), dtype=scale.dtype, device=scale.device)
    return torch.cat([payload, pad_row], dim=0), torch.cat([scale, pad_scale], dim=0)


def _fx_stream():
    return fx.Stream(torch.cuda.current_stream().cuda_stream)


def _gemm_kwargs(model_dim, inter_dim):
    return {
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "expert_offset": 0,
        "sort_block_m": SORT_BLOCK_M,
        "tile_n": TILE_N,
        "tile_k": TILE_K,
        "num_cu": torch.cuda.get_device_properties(0).multi_processor_count,
        "swiglu_limit": 0.0,
    }


def _rx_ok(workspace, nccl_a, nccl_scale, n_tokens):
    return torch.equal(workspace.rx_u8[:n_tokens], nccl_a) and torch.equal(
        workspace.rx_scale[:n_tokens], nccl_scale
    )


def test_tp_chunk_fused(
    rank,
    world,
    device,
    workspace: TpIncrementalWorkspace,
    w1,
    w1_scale,
    m_local,
    model_dim,
    inter_dim,
    experts,
    topk,
):
    n_tokens = world * m_local
    n_chunks = num_publish_chunks(m_local, CHUNK_ROWS)
    x, topk_weights, topk_ids = _make_local_inputs(
        m_local, model_dim, experts, topk, rank, seed=0, device=device
    )
    x_fp8, x_scale = per_1x32_mx_quant(x, quant_mode="fp8")
    nccl_a = _all_gather_uint8(x_fp8, world)
    nccl_scale = _all_gather_uint8(x_scale, world)
    ids_g = torch.empty((n_tokens, topk), dtype=torch.int32, device=device)
    wts_g = torch.empty((n_tokens, topk), dtype=torch.float32, device=device)
    dist.all_gather_into_tensor(ids_g, topk_ids)
    dist.all_gather_into_tensor(wts_g, topk_weights)

    sorted_ids, _, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        ids_g, wts_g, experts, model_dim, dtypes.bf16, SORT_BLOCK_M
    )
    num_valid = int(num_valid_ids[0].item())
    n_m = num_valid // SORT_BLOCK_M
    tile_row_base = make_tile_row_base(num_valid, SORT_BLOCK_M, device=device)
    expert_ids = sorted_expert_ids[:n_m].contiguous()
    nccl_dense, nccl_scale_dense = _pad_dense(
        nccl_a.view(torch.float8_e4m3fn), nccl_scale
    )
    x_packed = pack_rows_by_sorted_ids(nccl_dense, sorted_ids, num_valid, n_tokens)
    scale_packed = pack_rows_by_sorted_ids(
        nccl_scale_dense, sorted_ids, num_valid, n_tokens
    )
    kw = _gemm_kwargs(model_dim, inter_dim)
    stream = _fx_stream()
    out_ref, os_ref = _alloc_out(num_valid, inter_dim, device)
    gemm1_kernel(
        out_ref,
        x_packed,
        w1,
        scale_packed,
        w1_scale,
        tile_row_base,
        expert_ids,
        os_ref,
        num_valid,
        stream,
        **kw,
    )

    def _run(early_compute, reset=True):
        if reset:
            workspace.reset()
        out, os_ = _alloc_out(num_valid, inter_dim, device)
        run_tp_chunk_fused(
            workspace,
            x_fp8,
            x_scale,
            out,
            w1,
            w1_scale,
            tile_row_base,
            expert_ids,
            sorted_ids,
            os_,
            num_valid,
            n_tokens,
            m_local,
            stream,
            chunk_rows=CHUNK_ROWS,
            early_compute=early_compute,
            **kw,
        )
        torch.cuda.synchronize()
        return out, os_

    out_last, os_last = _run(early_compute=False)
    last_rx = _rx_ok(workspace, nccl_a, nccl_scale, n_tokens)
    last_ready = int(workspace.chunk_ready[n_chunks - 1].item()) == world
    last_err = checkAllclose(
        out_ref.float(),
        out_last.float(),
        rtol=0,
        atol=0,
        printLog=(rank == 0),
        msg="wait-last gemm1 vs packed NCCL A",
    )
    last_scale_err = checkAllclose(
        os_ref.float(),
        os_last.float(),
        rtol=0,
        atol=0,
        printLog=(rank == 0),
        msg="wait-last gemm1 scale vs packed NCCL A",
    )
    _agree(
        last_rx and last_ready and last_err == 0 and last_scale_err == 0,
        device,
        f"wait-last mismatch: rx={last_rx} ready={last_ready} "
        f"out_err={last_err} scale_err={last_scale_err}",
    )

    out_replay, os_replay = _run(early_compute=False, reset=False)
    replay_rx = _rx_ok(workspace, nccl_a, nccl_scale, n_tokens)
    replay_ready = int(workspace.chunk_ready[n_chunks - 1].item()) == 2 * world
    replay_err = checkAllclose(
        out_ref.float(),
        out_replay.float(),
        rtol=0,
        atol=0,
        printLog=(rank == 0),
        msg="wait-last replay without host zero",
    )
    replay_scale_err = checkAllclose(
        os_ref.float(),
        os_replay.float(),
        rtol=0,
        atol=0,
        printLog=(rank == 0),
        msg="wait-last replay scale without host zero",
    )
    _agree(
        replay_rx and replay_ready and replay_err == 0 and replay_scale_err == 0,
        device,
        f"epoch replay mismatch: rx={replay_rx} ready={replay_ready} "
        f"out_err={replay_err} scale_err={replay_scale_err} "
        f"ready={int(workspace.chunk_ready[n_chunks - 1].item())}",
    )

    out_early, os_early = _run(early_compute=True)
    early_rx = _rx_ok(workspace, nccl_a, nccl_scale, n_tokens)
    early_ready = int(workspace.chunk_ready[n_chunks - 1].item()) == world
    early_err = checkAllclose(
        out_ref.float(),
        out_early.float(),
        rtol=0,
        atol=0,
        printLog=False,
        msg="early-compute gemm1 vs packed NCCL A",
    )
    early_scale_err = checkAllclose(
        os_ref.float(),
        os_early.float(),
        rtol=0,
        atol=0,
        printLog=False,
        msg="early-compute gemm1 scale vs packed NCCL A",
    )
    _agree(
        early_rx and early_ready,
        device,
        f"early-compute rx/ready mismatch: rx={early_rx} ready={early_ready}",
    )
    if n_chunks == 1:
        _agree(
            early_err == 0 and early_scale_err == 0,
            device,
            f"C=1 early-compute must match wait-last: out_err={early_err} "
            f"scale_err={early_scale_err}",
        )

    return {
        "gfx": get_gfx(),
        "tp": world,
        "m_local": m_local,
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "experts": experts,
        "topk": topk,
        "tokens": n_tokens,
        "num_valid": num_valid,
        "chunks": n_chunks,
        "wait-last gemm err": last_err,
        "wait-last scale err": last_scale_err,
        "early rx err": 0.0 if early_rx else 1.0,
        "early gemm err": early_err,
    }


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="activation dtype before MXFP8 quant (bf16 only)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 8, 64],
        help="m_local (tokens per TP rank)",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(256, 128)],
        help="model_dim,inter_dim. e.g.: -s 256,128",
    )
    parser.add_argument("--experts", type=int, nargs="*", default=[8])
    parser.add_argument("--topk", type=int, nargs="*", default=[2])
    args = parser.parse_args()

    rank, world, device = _setup_dist()
    try:
        gfx = get_gfx()
        if gfx not in SUPPORTED_GFX:
            if rank == 0:
                aiter.logger.warning("tp chunk fused unsupported on %s; skipping", gfx)
            return
        if world < 2:
            if rank == 0:
                aiter.logger.warning("need world_size>=2; got %s", world)
            return

        max_m_local = max(args.batch)
        rows = []
        workspace = None
        weights = {}
        for _dtype, m_local, (model_dim, inter_dim), experts, topk in itertools.product(
            args.dtype, args.batch, args.mnk, args.experts, args.topk
        ):
            if workspace is None or (
                workspace.model_dim != model_dim or workspace.num_experts != experts
            ):
                workspace = TpIncrementalWorkspace(
                    rank=rank,
                    npes=world,
                    max_m_local=max_m_local,
                    model_dim=model_dim,
                    num_experts=experts,
                    device=device,
                )
            key = (model_dim, experts, inter_dim)
            if key not in weights:
                weights[key] = _make_w1(experts, model_dim, inter_dim, device)
            w1, w1_scale = weights[key]
            if rank == 0:
                aiter.logger.info(
                    "chunk fused case m_local=%s model_dim=%s experts=%s topk=%s",
                    m_local,
                    model_dim,
                    experts,
                    topk,
                )
            rows.append(
                test_tp_chunk_fused(
                    rank,
                    world,
                    device,
                    workspace,
                    w1,
                    w1_scale,
                    m_local,
                    model_dim,
                    inter_dim,
                    experts,
                    topk,
                )
            )
        if rank == 0:
            df = pd.DataFrame(rows)
            aiter.logger.info(
                "tp chunk fused summary (markdown):\n%s",
                df.to_markdown(index=False),
            )
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
