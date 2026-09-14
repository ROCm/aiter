# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Step 3: min-expert TP activation push vs NCCL all-gather.

Each rank publishes every local token once into every peer's dense slab at
``rank * m_local + t``. Consumers still wait for full publish (no overlap).

Run with::

    MORI_SOCKET_IFNAME=lo MORI_SHMEM_HEAP_SIZE=40G PYTHONPATH="$PWD" \\
      torchrun --standalone --nproc-per-node=8 \\
      op_tests/multigpu_tests/test_tp_incremental_push.py
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
from aiter.ops.flydsl.kernels.mega_moe.tp_incremental_push import (
    TpIncrementalWorkspace,
    run_tp_incremental_push,
)
from aiter.ops.flydsl.kernels.mega_moe.tp_incremental_schedule import (
    expected_token_counts,
    make_tile_row_base,
    pack_rows_by_sorted_ids,
)
from aiter.ops.flydsl.kernels.mega_moe.tp_token_gemm1 import gemm1_token_kernel
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import checkAllclose

SUPPORTED_GFX = ["gfx950"]
SORT_BLOCK_M = 32
TILE_N = 256
TILE_K = 256


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


def _mismatch_ratio(got: torch.Tensor, ref: torch.Tensor) -> float:
    if got.numel() == 0:
        return 0.0
    return float((got.view(torch.uint8) != ref.view(torch.uint8)).float().mean().item())


def _row_major_from_rank_major(rank_major: torch.Tensor, npes: int, m_local: int):
    hidden = rank_major.shape[1]
    return (
        rank_major.view(npes, m_local, hidden)
        .permute(1, 0, 2)
        .contiguous()
        .view(npes * m_local, hidden)
    )


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


def _time_us(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) * 1e3)


def _run_push(workspace, local_x, local_scale, local_ids, m_local, row_major=False):
    def _launch():
        run_tp_incremental_push(
            workspace,
            local_x,
            local_scale,
            local_ids,
            m_local,
            _fx_stream(),
            row_major=row_major,
        )

    return _time_us(_launch)


def test_tp_incremental_push(
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
    expected = expected_token_counts(ids_g, experts).to(
        dtype=torch.int32, device=device
    )

    workspace.reset()
    push_us = _run_push(workspace, x_fp8, x_scale, topk_ids, m_local, row_major=False)
    got_a = workspace.rx_u8[:n_tokens]
    got_scale = workspace.rx_scale[:n_tokens]
    a_err = _mismatch_ratio(got_a, nccl_a)
    scale_err = _mismatch_ratio(got_scale, nccl_scale)
    recv_ok = torch.equal(workspace.received, expected)
    done_ok = int(workspace.ranks_done.item()) == world
    layout_ok = a_err == 0.0 and scale_err == 0.0 and recv_ok and done_ok
    _agree(
        layout_ok,
        device,
        f"rank-major push mismatch: a_err={a_err} scale_err={scale_err} "
        f"recv_ok={bool(recv_ok)} ranks_done={int(workspace.ranks_done.item())}",
    )

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
    rx_dense = workspace.rx[: n_tokens + 1]
    rx_scale_dense = workspace.rx_scale[: n_tokens + 1]
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
    out_tok, os_tok = _alloc_out(num_valid, inter_dim, device)
    gemm1_token_kernel(
        out_tok,
        rx_dense,
        w1,
        rx_scale_dense,
        w1_scale,
        tile_row_base,
        expert_ids,
        sorted_ids,
        os_tok,
        num_valid,
        n_tokens,
        stream,
        **kw,
    )
    torch.cuda.synchronize()
    gemm_err = checkAllclose(
        out_ref.float(),
        out_tok.float(),
        rtol=0,
        atol=0,
        printLog=(rank == 0),
        msg="token gemm1 vs packed NCCL A",
    )
    gemm_scale_err = checkAllclose(
        os_ref.float(),
        os_tok.float(),
        rtol=0,
        atol=0,
        printLog=(rank == 0),
        msg="token gemm1 scale vs packed NCCL A",
    )
    _agree(
        gemm_err == 0 and gemm_scale_err == 0,
        device,
        f"gemm1 mismatch: out_err={gemm_err} scale_err={gemm_scale_err}",
    )

    workspace.reset()
    _run_push(workspace, x_fp8, x_scale, topk_ids, m_local, row_major=True)
    rm_a = workspace.rx_u8[:n_tokens]
    rm_scale = workspace.rx_scale[:n_tokens]
    rm_ref_a = _row_major_from_rank_major(nccl_a, world, m_local)
    rm_ref_scale = _row_major_from_rank_major(nccl_scale, world, m_local)
    rm_a_err = _mismatch_ratio(rm_a, rm_ref_a)
    rm_scale_err = _mismatch_ratio(rm_scale, rm_ref_scale)
    vs_nccl = _mismatch_ratio(rm_a, nccl_a)
    _agree(
        rm_a_err == 0.0 and rm_scale_err == 0.0,
        device,
        f"row-major dest mismatch vs permute: a_err={rm_a_err} scale_err={rm_scale_err}",
    )
    if m_local > 1:
        _agree(
            vs_nccl > 0.0,
            device,
            f"row-major dest unexpectedly matched NCCL rank-major (ratio={vs_nccl})",
        )

    nbytes = 2 * n_tokens * (model_dim + model_dim // 32)
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
        "push us": push_us,
        "push TB/s": (nbytes / push_us / 1e6) if push_us else float("nan"),
        "a err": a_err,
        "scale err": scale_err,
        "recv err": 0.0 if recv_ok else 1.0,
        "gemm err": gemm_err,
        "gemm scale err": gemm_scale_err,
        "row-major a err": rm_a_err,
        "row-major vs nccl": vs_nccl,
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
                aiter.logger.warning(
                    "tp incremental push unsupported on %s; skipping", gfx
                )
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
            key = (model_dim, experts)
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
            if key not in weights:
                weights[key] = _make_w1(experts, model_dim, inter_dim, device)
            w1, w1_scale = weights[key]
            if rank == 0:
                aiter.logger.info(
                    "push case m_local=%s model_dim=%s experts=%s topk=%s",
                    m_local,
                    model_dim,
                    experts,
                    topk,
                )
            rows.append(
                test_tp_incremental_push(
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
                "tp incremental push summary (markdown):\n%s",
                df.to_markdown(index=False),
            )
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
