# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""TestWideEpMoe EP16 A4W4 correctness, performance, and profiling test.

Pipeline: packed-FP4 dispatch (MORI InterNodeV1LL) -> AITER fused_moe
(A4W4, per_1x32, real expert_mask) -> BF16 combine (MORI InterNodeV1LL,
same TestWideEpMoe instance).

Launch (2 nodes, one torchrun process per node, 8 local GPUs spawned inside):

  On node_rank 0:
    GPU_PER_NODE=8 torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
        --master_addr=<node0_ip> --master_port=29500 \
        test_wide_ep_moe.py --bs-list 128,512,1024,2048,4096
  On node_rank 1: same with --node_rank=1
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")

import mori
import mori.shmem as ms
import torch
import torch.distributed as dist

import aiter
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_topk, situv2
from aiter.ops.flydsl.test_wide_ep_moe import TestWideEpMoe
from aiter.ops.shuffle import (
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)
from aiter.utility import fp4_utils

# Target EP16 A4W4 pipeline shape. The CSV-compatible result format follows
# aiter/configs/model_configs/kimik3_a4w4_tuned_fmoe.csv, while this benchmark
# intentionally uses the requested larger intermediate dimension (3072).
NETWORK = {
    "model_dim": 3584,
    "inter_dim": 3072,
    "experts": 896,
    "topk": 16,
}
GPU_PER_NODE_DEFAULT = 8


def _setup_dist(rank, world_size, local_rank):
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="cpu:gloo", rank=rank, world_size=world_size
        )
    world_group = dist.group.WORLD
    assert world_group is not None
    torch._C._distributed_c10d._register_process_group("default", world_group)
    ms.shmem_torch_process_group_init("default")
    return device


def _cleanup():
    ms.shmem_finalize()
    if dist.is_initialized():
        dist.destroy_process_group()


def _barrier():
    debug = os.environ.get("AITER_DEBUG_WIDE_EP", "0") == "1"
    rank = dist.get_rank() if dist.is_initialized() else -1
    if debug:
        print(f"[EP16-barrier rank={rank}] cuda sync 1 start", flush=True)
    torch.cuda.synchronize()
    if debug:
        print(f"[EP16-barrier rank={rank}] cuda sync 1 complete; shmem start", flush=True)
    ms.shmem_barrier_all()
    if debug:
        print(f"[EP16-barrier rank={rank}] shmem complete; cuda sync 2 start", flush=True)
    # Agent: shmem_barrier_all may enqueue device work. Drain it here so a
    # caller that records a CUDA event immediately after _barrier() does not
    # accidentally charge the barrier kernel to the following stage.
    torch.cuda.synchronize()
    if debug:
        print(f"[EP16-barrier rank={rank}] cuda sync 2 complete", flush=True)


def _reduce_float(value, op):
    # Process group is cpu:gloo only (no NCCL) -- reduce on CPU.
    result = torch.tensor(float(value), dtype=torch.float32, device="cpu")
    dist.all_reduce(result, op=op)
    return float(result.item())


def _make_local_inputs(
    tokens,
    model_dim,
    experts,
    topk,
    rank,
    seed,
    device,
    routing="random",
    max_tok_anchor=None,
    world_size=None,
    gpu_per_node=None,
):
    """Per-rank random tokens + router topk.

    routing="random" (default): real fused_topk kernel over random router
    logits (test_moe_ep.py style) -- statistically balanced but with natural
    per-rank variance. Duplicate-free within a token by construction (top-k
    selection over one row never repeats a position).

    routing="round_robin": fully-balanced deterministic assignment, same
    formula as mori's tests/python/ops/dispatch_combine_test_utils.py
    gen_test_data(routing="round_robin") (adapted to this EP group's global
    token index: base = (rank * max_tok_anchor + local_token_idx) * topk).
    CAVEAT (found empirically): `topk` *consecutive* ids mod `experts` tend to
    land entirely inside one rank's contiguous expert block (block size 56 >>
    topk 16), so most tokens end up fully local to a single rank -- this
    balances how often each expert is picked, but does NOT spread a given
    token's routing across ranks. Verified: total_recv dropped ~9x vs
    "random" at the same bs.

    routing="cross_node": balances every token across both nodes and all local
    ranks. With 2 nodes, 8 GPUs/node and topk=16, every token sends exactly one
    route to each of the 16 ranks. The local expert rotates with the global
    token index so larger batches also spread work across all local experts.
    """
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn(
        (tokens, model_dim), dtype=torch.bfloat16, device=device, generator=generator
    )
    if routing == "round_robin":
        anchor = max_tok_anchor if max_tok_anchor is not None else tokens
        base = (rank * anchor + torch.arange(tokens, device=device)) * topk
        offsets = torch.arange(topk, device=device)
        topk_ids = ((base.unsqueeze(1) + offsets.unsqueeze(0)) % experts).to(torch.int32)
        raw_weights = torch.rand(
            (tokens, topk), dtype=torch.float32, device=device, generator=generator
        )
        topk_weights = raw_weights.softmax(dim=-1)
    elif routing == "cross_node":
        assert world_size is not None and gpu_per_node is not None
        assert world_size % gpu_per_node == 0, "world_size must be a multiple of gpu_per_node"
        num_nodes = world_size // gpu_per_node
        assert num_nodes == 2, "cross_node routing is written for exactly 2 nodes"
        assert experts % world_size == 0
        local_experts = experts // world_size
        experts_per_node = gpu_per_node * local_experts
        anchor = max_tok_anchor if max_tok_anchor is not None else tokens
        tok_idx = rank * anchor + torch.arange(tokens, device=device)  # (tokens,)
        j = torch.arange(topk, device=device)  # (topk,)
        target_node = j % num_nodes  # (topk,) -- alternates 0/1/0/1/...
        slot_in_node = j // num_nodes  # 0..7 for EP16/topk16
        if topk // num_nodes > gpu_per_node:
            raise ValueError("cross_node routing requires topk/num_nodes <= gpu_per_node")
        target_rank_in_node = (tok_idx.unsqueeze(1) + slot_in_node) % gpu_per_node
        target_local_expert = (
            tok_idx.unsqueeze(1) * (topk // num_nodes) + slot_in_node
        ) % local_experts
        topk_ids = (
            target_node.unsqueeze(0) * experts_per_node
            + target_rank_in_node * local_experts
            + target_local_expert
        ).to(torch.int32)
        raw_weights = torch.rand(
            (tokens, topk), dtype=torch.float32, device=device, generator=generator
        )
        topk_weights = raw_weights.softmax(dim=-1)
    elif routing == "random":
        scores = torch.randn(
            (tokens, experts), dtype=torch.bfloat16, device=device, generator=generator
        )
        topk_ids = torch.empty((tokens, topk), dtype=torch.int32, device=device)
        topk_weights = torch.empty((tokens, topk), dtype=torch.float32, device=device)
        fused_topk(x, scores, topk, True, topk_ids, topk_weights)
    else:
        raise ValueError(
            f"unknown routing: {routing!r} (choose 'random', 'round_robin', or 'cross_node')"
        )
    return x.contiguous(), topk_weights.contiguous(), topk_ids.contiguous()


def _quantize_local_weights(model_dim, inter_dim, local_experts, rank, seed, device):
    """Per-rank local-expert bf16 weights -> a4w4 (per_1x32 mxfp4) quantized +
    shuffled, following test_moe_ep.py's a4w4_mxfp4 branch exactly. Returns both
    the kernel-ready shuffled tensors and the unshuffled quantized tensors (for
    the dequantized reference computation)."""
    generator = torch.Generator(device=device).manual_seed(seed + 1000 + rank)
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    use_mxmoe_w1 = os.environ.get("MORI_MXMOE_W1_LAYOUT", "0") == "1"

    w1 = (
        torch.randn(
            (local_experts, 2 * inter_dim, model_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.1
    )
    if use_mxmoe_w1:
        from aiter.ops.quant import per_1x32_mx_quant_hip

        w1_qt, w1_scale = per_1x32_mx_quant_hip(
            w1.view(-1, model_dim), quant_dtype=dtypes.fp4x2
        )
        w1_qt = w1_qt.view(local_experts, 2 * inter_dim, model_dim // 2)
    else:
        w1_qt, w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
        w1_qt = w1_qt.view(local_experts, 2 * inter_dim, model_dim // 2)

    w2 = (
        torch.randn(
            (local_experts, model_dim, inter_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.1
    )
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
    w2_qt = w2_qt.view(local_experts, model_dim, inter_dim // 2)

    # mxmoe GEMM1 consumes the A16W4 preshuffle, while the previous
    # flydsl_moe1_afp4 baseline consumes the generic layout.  Keep both paths
    # testable without rewriting the weight setup between regression runs.
    w1_a = (
        shuffle_weight_a16w4(w1_qt, 16, False)
        if use_mxmoe_w1
        else shuffle_weight(w1_qt, layout=(16, 16))
    )
    w2_a = (
        shuffle_weight_a16w4(w2_qt, 16, False)
        if use_mxmoe_w1
        else shuffle_weight(w2_qt, layout=(16, 16))
    )
    w1_s = (
        shuffle_scale_a16w4(w1_scale, local_experts, False)
        if use_mxmoe_w1
        else fp4_utils.e8m0_shuffle(w1_scale)
    )
    w2_s = (
        shuffle_scale_a16w4(
            w2_scale.view(-1, inter_dim // 32), local_experts, False
        )
        if use_mxmoe_w1
        else fp4_utils.e8m0_shuffle(w2_scale)
    )
    w1_a.is_shuffled = True
    w2_a.is_shuffled = True

    return (w1_a, w1_s, w2_a, w2_s), (w1_qt, w1_scale, w2_qt, w2_scale)


def _dequant_weight(w_qt, w_scale, orig_shape):
    """mxfp4 -> f32, same formula as test_moe_ep.py's _dequant."""
    wf = fp4_utils.mxfp4_to_f32(w_qt).view(*orig_shape)
    sf = fp4_utils.e8m0_to_f32(w_scale).view(orig_shape[0], orig_shape[1], -1)
    sf = sf.unsqueeze(-1).expand(-1, -1, -1, 32).reshape(*orig_shape)
    return (wf * sf).to(torch.bfloat16)


def _dequant_tokens(tok_fp4, scale, hidden_dim):
    """Dequantize dispatched fp4 tokens back to bf16 (local compute only, never
    crosses the network -- the network transport itself carried fp4)."""
    n = tok_fp4.shape[0]
    wf = fp4_utils.mxfp4_to_f32(tok_fp4).view(n, hidden_dim)
    sf = fp4_utils.e8m0_to_f32(scale).view(n, hidden_dim // 32)
    sf = sf.unsqueeze(-1).expand(-1, -1, 32).reshape(n, hidden_dim)
    return (wf * sf).to(torch.bfloat16)


def _torch_moe_situv2_reference(x, w1, w2, weights, global_ids, expert_mask):
    """Local EP reference following test_moe_2stage's SiTUv2 definition."""
    compute_type = torch.float32
    batch, model_dim = x.shape
    topk = weights.shape[1]
    inter_dim = w2.shape[2]
    local_hash = expert_mask.cumsum(0, dtype=dtypes.i32) - 1
    local_hash[expert_mask == 0] = -1
    local_ids = local_hash[global_ids.long()]
    x_routes = x.to(compute_type).view(batch, 1, model_dim).expand(-1, topk, -1)
    out = torch.zeros((batch, topk, model_dim), dtype=compute_type, device=x.device)
    w1 = w1.to(compute_type)
    w2 = w2.to(compute_type)
    for expert_id in range(w1.shape[0]):
        mask = local_ids == expert_id
        if mask.any():
            gate, up = (x_routes[mask] @ w1[expert_id].transpose(0, 1)).split(
                [inter_dim, inter_dim], dim=-1
            )
            hidden = situv2(gate, up, beta=1.0, linear_beta=1.0)
            out[mask] = hidden @ w2[expert_id].transpose(0, 1)
    return (out * weights.view(batch, topk, 1)).sum(dim=1).to(x.dtype)


def _build_expert_mask(experts, local_expert_start, local_expert_end, device):
    expert_mask = torch.zeros((experts + 1,), dtype=dtypes.i32, device=device)
    expert_mask[local_expert_start:local_expert_end] = 1
    expert_mask[-1] = 0  # fake/padding expert id, never local
    return expert_mask


def _run_comm_only_bs(bs, op, x_fp4, x_scale, weights, ids, model_dim, world_size,
                      iters, stat_iters, rank):
    """Time rank-aligned dispatch-only and combine-only phases."""
    backend = op
    x_fp4 = x_fp4[:bs].contiguous()
    x_scale = x_scale[:bs].contiguous()
    weights = weights[:bs].contiguous()
    ids = ids[:bs].contiguous()
    dispatched = backend.dispatch_prequant(x_fp4, x_scale, weights, ids)
    combine_input = torch.zeros(
        (dispatched.tokens.shape[0], model_dim), dtype=torch.bfloat16,
        device=x_fp4.device,
    )
    backend.combine(combine_input, dispatched)
    torch.cuda.synchronize()

    # Dispatch phase: align ranks before the timed dispatch; consume the
    # routing with an untimed combine so every epoch completes normally.
    dispatch_events = [torch.cuda.Event(enable_timing=True) for _ in range(2 * iters)]
    for i in range(iters):
        torch.cuda.synchronize()
        dist.barrier()
        dispatch_events[2 * i].record()
        dispatched = backend.dispatch_prequant(x_fp4, x_scale, weights, ids)
        dispatch_events[2 * i + 1].record()
        backend.combine(combine_input, dispatched)
    torch.cuda.synchronize()

    # Combine phase: prepare dispatch outside the bracket, drain it locally,
    # then align all ranks before timing only combine.
    combine_events = [torch.cuda.Event(enable_timing=True) for _ in range(2 * iters)]
    for i in range(iters):
        dispatched = backend.dispatch_prequant(x_fp4, x_scale, weights, ids)
        torch.cuda.synchronize()
        dist.barrier()
        combine_events[2 * i].record()
        backend.combine(combine_input, dispatched)
        combine_events[2 * i + 1].record()
    torch.cuda.synchronize()

    keep = max(1, min(stat_iters, iters))
    dispatch_ms = [
        dispatch_events[2*i].elapsed_time(dispatch_events[2*i+1]) for i in range(iters)
    ][-keep:]
    combine_ms = [
        combine_events[2*i].elapsed_time(combine_events[2*i+1]) for i in range(iters)
    ][-keep:]
    d_local = sum(dispatch_ms) / keep
    c_local = sum(combine_ms) / keep
    d_avg = _reduce_float(d_local, dist.ReduceOp.SUM) / world_size
    c_avg = _reduce_float(c_local, dist.ReduceOp.SUM) / world_size
    d_min = _reduce_float(d_local, dist.ReduceOp.MIN)
    d_max = _reduce_float(d_local, dist.ReduceOp.MAX)
    c_min = _reduce_float(c_local, dist.ReduceOp.MIN)
    c_max = _reduce_float(c_local, dist.ReduceOp.MAX)
    d_iter_min = _reduce_float(min(dispatch_ms), dist.ReduceOp.MIN)
    c_iter_min = _reduce_float(min(combine_ms), dist.ReduceOp.MIN)
    if rank == 0:
        print(
            f"[EP16-comm-only] bs={bs} stat={keep}/{iters} "
            f"dispatch={d_avg*1000:.2f}/{d_min*1000:.2f}/{d_max*1000:.2f}us "
            f"combine={c_avg*1000:.2f}/{c_min*1000:.2f}/{c_max*1000:.2f}us "
            f"mean/best/worst min_iter={d_iter_min*1000:.2f}/{c_iter_min*1000:.2f}us",
            flush=True,
        )


def _run_one_bs(
    bs,
    op,
    x_fp4,
    x_scale,
    topk_weights,
    topk_ids,
    w1_a,
    w1_s,
    w2_a,
    w2_s,
    w1_qt,
    w1_scale,
    w2_qt,
    w2_scale,
    expert_mask,
    local_experts,
    model_dim,
    inter_dim,
    world_size,
    iters,
    stat_iters,
    rtol,
    accuracy_max_bs,
    rank,
    perf_out,
    torch_profiler_dir,
    torch_compile_cudagraph,
    profile_warmup_iters,
    profile_iters,
    staged_only,
):
    x_fp4_bs = x_fp4[:bs].contiguous()
    x_scale_bs = x_scale[:bs].contiguous()
    topk_weights_bs = topk_weights[:bs].contiguous()
    topk_ids_bs = topk_ids[:bs].contiguous()

    if os.environ.get("MORI_COMM_ONLY", "0") == "1":
        _run_comm_only_bs(
            bs, op, x_fp4, x_scale, topk_weights, topk_ids, model_dim,
            world_size, iters, stat_iters, rank,
        )
        return

    out = None
    if not staged_only:
        # Public TestWideEpMoe contract: the operator owns the complete
        # inter-node dispatch -> fused_moe -> combine sequence.
        out = op.forward_prequant(
            x_fp4_bs, x_scale_bs, topk_weights_bs, topk_ids_bs
        ).clone()
        torch.cuda.synchronize()
        assert out.shape == (bs, model_dim)
        assert torch.isfinite(out.float()).all(), "TestWideEpMoe output has non-finite values"

        # The public-call check and staged diagnostic are distinct MORI epochs.
        _barrier()

    # Diagnostic-only staged call through the private backend. This preserves
    # per-stage profiling without changing TestWideEpMoe's public forward API.
    backend = op
    debug_wide_ep = os.environ.get("AITER_DEBUG_WIDE_EP", "0") == "1"
    if debug_wide_ep:
        print(f"[EP16-debug rank={rank}] diagnostic dispatch start", flush=True)
    dispatched = backend.dispatch_prequant(x_fp4_bs, x_scale_bs, topk_weights_bs, topk_ids_bs)
    if debug_wide_ep:
        print(f"[EP16-debug rank={rank}] diagnostic dispatch complete", flush=True)
    recv_tok_fp4 = dispatched.tokens
    recv_wts = dispatched.weights
    recv_scale = dispatched.scales
    recv_idx = dispatched.expert_ids
    recv_num_token = dispatched.num_tokens
    torch.cuda.synchronize()
    total_recv = int(recv_num_token[0].item())

    # Capture the two compute launch callables so GEMM1/GEMM2 can be timed
    # independently on the actual post-MORI routing distribution.
    fused_moe_module = importlib.import_module("aiter.fused_moe")
    kernel_calls = []
    fused_moe_module.kernel_bench_callable = kernel_calls
    try:
        moe_out = backend.fused_moe(dispatched)
    finally:
        fused_moe_module.kernel_bench_callable = None
    if debug_wide_ep:
        print(f"[EP16-debug rank={rank}] diagnostic fused_moe complete", flush=True)
        print(
            f"[EP16-debug rank={rank}] combine ABI "
            f"recv_shape={tuple(recv_tok_fp4.shape)} recv_stride={recv_tok_fp4.stride()} "
            f"moe_shape={tuple(moe_out.shape)} moe_stride={moe_out.stride()} "
            f"moe_contiguous={moe_out.is_contiguous()} total_recv={total_recv} "
            f"source_tokens={bs}",
            flush=True,
        )
    if staged_only:
        torch.cuda.synchronize()
        if debug_wide_ep:
            print(f"[EP16-debug rank={rank}] fused_moe GPU sync complete", flush=True)

    # combine()'s indices/weights must be THIS rank's own [tokens, topk]
    # routing passed to dispatch() -- NOT dispatch()'s returned recv_idx/
    # recv_wts (ROCm/mori#475). weights=None: fused_moe already applied
    # topk weighting in stage2 (same convention as
    # test_dispatch_combine_internode.py's run_combine).
    if staged_only and os.environ.get("AITER_DEBUG_COMBINE_WITH_WEIGHTS", "0") == "1":
        # ABI diagnostic only: current MORI's official V1LL test passes the
        # dispatch-returned weights into combine. Production fused_moe already
        # applies them, so this path must not be used for numerical validation.
        combine_out, combine_out_wts = backend.op.combine(
            moe_out, dispatched.weights, dispatched._source_topk_ids
        )
        dispatched._consumed = True
    else:
        combine_out, combine_out_wts = backend.combine(moe_out, dispatched)
    torch.cuda.synchronize()
    if debug_wide_ep:
        print(f"[EP16-debug rank={rank}] diagnostic combine complete", flush=True)
    # Captured atomic GEMM2 calls below intentionally reuse their output buffer
    # for timing and therefore accumulate into ``moe_out``.  Preserve the
    # single-execution result before benchmarking for correctness checks.
    moe_out_correctness = moe_out.clone() if bs <= accuracy_max_bs else None
    diagnostic_out = combine_out[:bs]
    if out is not None and bs <= accuracy_max_bs:
        # GEMM2 uses atomic accumulation, so two otherwise identical launches
        # are not bitwise deterministic. Keep this as a BF16 consistency check.
        torch.testing.assert_close(out, diagnostic_out, rtol=1e-2, atol=2.5e-1)

    def _time_captured_kernel(call):
        for _ in range(3):
            call()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(20)]
        for start, end in zip(starts, ends):
            start.record()
            call()
            end.record()
        torch.cuda.synchronize()
        return sum(start.elapsed_time(end) for start, end in zip(starts, ends)) / 20

    kernel_us = {name: _time_captured_kernel(call) * 1000 for name, call in kernel_calls}
    if debug_wide_ep:
        print(f"[EP16-debug rank={rank}] captured GEMM timing complete", flush=True)
    gemm1_us_local = kernel_us.get("stage1", 0.0)
    gemm2_us_local = kernel_us.get("stage2", 0.0)
    gemm1_us = _reduce_float(gemm1_us_local, dist.ReduceOp.SUM) / world_size
    gemm2_us = _reduce_float(gemm2_us_local, dist.ReduceOp.SUM) / world_size
    gemm_total_us_local = gemm1_us_local + gemm2_us_local
    gemm_total_us = gemm1_us + gemm2_us
    gemm_total_min_us = _reduce_float(gemm_total_us_local, dist.ReduceOp.MIN)
    gemm_total_max_us = _reduce_float(gemm_total_us_local, dist.ReduceOp.MAX)

    # ---- correctness (only below accuracy_max_bs, all-gather ref is O(bs*world)) ----
    rel_l2 = -1.0
    if bs <= accuracy_max_bs:
        if debug_wide_ep:
            print(f"[EP16-debug rank={rank}] torch reference start", flush=True)
        # Reference-only dequantization. The measured production path passes
        # dispatch's packed FP4 activation and E8M0 scale directly to fused_moe.
        recv_tok_bf16 = _dequant_tokens(recv_tok_fp4, recv_scale, model_dim)
        w1_deq = _dequant_weight(w1_qt, w1_scale, (local_experts, 2 * inter_dim, model_dim))
        w2_deq = _dequant_weight(w2_qt, w2_scale, (local_experts, model_dim, inter_dim))
        if os.environ.get("AITER_DEBUG_MX_EP_SYNC", "0") == "1":
            debug_sort = getattr(fused_moe_module, "_mx_ep_debug_sort", None)
            stage1_call = dict(kernel_calls).get("stage1")
            if debug_sort is not None and stage1_call is not None:
                stage1_q, stage1_scale = stage1_call()
                torch.cuda.synchronize()
                post_pad = debug_sort["post_pad"]
                block_m_dbg = debug_sort["block_m"]
                token_rows = debug_sort["sorted_ids"][:post_pad] & 0x00FFFFFF
                valid = token_rows < debug_sort["valid_rows"]
                expert_rows = debug_sort["sorted_expert_ids"][
                    : debug_sort["tile_count"]
                ].repeat_interleave(block_m_dbg)[:post_pad]
                m_rows = debug_sort["m_indices"][:post_pad]
                matches = 0
                elements = 0
                sq_error = 0.0
                sq_reference = 0.0
                gemm2_ref = torch.zeros_like(moe_out, dtype=torch.float32)
                torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
                for expert_id in torch.unique(expert_rows[valid]).tolist():
                    rows = torch.nonzero(valid & (expert_rows == expert_id)).flatten()
                    gate, up = (
                        recv_tok_bf16[m_rows[rows].long()].float()
                        @ w1_deq[int(expert_id)].float().transpose(0, 1)
                    ).split([inter_dim, inter_dim], dim=-1)
                    ref_inter = situv2(gate, up, beta=1.0, linear_beta=1.0)
                    ref_q, _ = torch_quant(
                        ref_inter.to(torch.bfloat16), quant_dtype=dtypes.fp4x2
                    )
                    got = stage1_q[rows]
                    ref_q = ref_q.view(torch.uint8)
                    matches += int((got == ref_q).sum().item())
                    elements += got.numel()
                    # Invert the BM32/BN256 scale layout written by GEMM1 and
                    # read by layout GEMM2. One dword is indexed by
                    # (m_chunk, ku, wave_group, m_lane), with two N-block
                    # halves and two 16-row halves packed into its four bytes.
                    scale_cols = inter_dim // 32
                    c = torch.arange(scale_cols, device=recv_tok_fp4.device)
                    n_block = c // 4
                    wave_group = c % 4
                    ku = n_block // 2
                    ikxdl = n_block % 2
                    rr = rows[:, None]
                    chunk = rr // 32
                    m_lane = rr % 16
                    row_half = (rr % 32) // 16
                    per_chunk_dw = ((inter_dim // 32) // 4 // 2) * 64
                    byte_addr = (
                        (chunk * per_chunk_dw + ku * 64 + wave_group * 16 + m_lane)
                        * 4
                        + ikxdl * 2
                        + row_half
                    )
                    scale_raw = stage1_scale.view(torch.uint8).flatten()[byte_addr]
                    got_f32 = fp4_utils.mxfp4_to_f32(got).view(
                        rows.numel(), inter_dim
                    )
                    scale_f32 = fp4_utils.e8m0_to_f32(scale_raw).repeat_interleave(
                        32, dim=-1
                    )
                    got_f32 = got_f32 * scale_f32
                    diff = got_f32 - ref_inter.float()
                    sq_error += float((diff * diff).sum().item())
                    sq_reference += float((ref_inter.float() ** 2).sum().item())
                    route_out = got_f32 @ w2_deq[int(expert_id)].float().transpose(0, 1)
                    route_out *= debug_sort["sorted_weights"][rows].float().view(-1, 1)
                    gemm2_ref.index_add_(0, m_rows[rows].long(), route_out)
                ratio = matches / max(elements, 1)
                inter_rel_l2 = (sq_error / max(sq_reference, 1e-30)) ** 0.5
                gemm2_rel_l2 = float(
                    torch.linalg.vector_norm(
                        moe_out_correctness[:total_recv].float()
                        - gemm2_ref[:total_recv]
                    )
                    / torch.linalg.vector_norm(gemm2_ref[:total_recv])
                )
                print(
                    f"[AITER_DEBUG_MX_EP] rank={rank} GEMM1 packed-code "
                    f"match={ratio:.6f} ({matches}/{elements}) "
                    f"decoded_inter_relL2={inter_rel_l2:.6f} "
                    f"gemm2_from_inter_relL2={gemm2_rel_l2:.6f}",
                    flush=True,
                )
        ref_moe_out = _torch_moe_situv2_reference(
            recv_tok_bf16[:total_recv],
            w1_deq,
            w2_deq,
            recv_wts[:total_recv],
            recv_idx[:total_recv],
            expert_mask,
        )
        rel_l2 = float(
            torch.linalg.vector_norm(
                (moe_out_correctness[:total_recv] - ref_moe_out).float()
            )
            / torch.linalg.vector_norm(ref_moe_out.float())
        )
        rel_l2 = _reduce_float(rel_l2, dist.ReduceOp.MAX)
        if rel_l2 >= rtol:
            raise AssertionError(f"bs={bs} moe relL2={rel_l2:.6f} exceeds rtol={rtol}")
        assert diagnostic_out.shape == (bs, model_dim)
        assert torch.isfinite(diagnostic_out.float()).all(), "combine output has non-finite values"
        if debug_wide_ep:
            print(f"[EP16-debug rank={rank}] torch reference complete", flush=True)

    if staged_only:
        if rank == 0:
            print(
                f"[EP16-staged-only] bs={bs} relL2={rel_l2:.6f} PASS",
                flush=True,
            )
        return

    # ---- logical GEMM row count: (received row, local expert slot) pairs ----
    # actually computed by fused_moe's grouped GEMM -- exact, not an estimate.
    recv_idx_flat = recv_idx[:total_recv].reshape(-1)
    local_mask = expert_mask[recv_idx_flat.long()] == 1
    local_hits = int(local_mask.sum().item())
    # number of this rank's local experts that actually received >=1 token --
    # matches aiter's own MoE-GEMM benchmark convention (bench_moe_gemm_a4w4_cudagraph.py's
    # `routed = int((rdata.expt_data.hist > 0).sum())`): weight bytes should only be
    # counted for experts that were actually touched, not every local expert, since at
    # small bs some local experts may see zero tokens.
    active_local_experts = int(torch.unique(recv_idx_flat[local_mask]).numel())

    # Agent: emit the per-rank payload metadata used to reproduce MORI's
    # bandwidth convention from rocprof kernel durations.
    print(
        f"[EP16-rank-meta] bs={bs} rank={rank} total_recv={total_recv} "
        f"local_hits={local_hits} active_local_experts={active_local_experts}",
        flush=True,
    )

    _barrier()

    # ---- perf: 3 timing brackets (dispatch / moe / combine) ----
    # CAVEAT (found via rocprofv3 ground truth, not yet resolved): these
    # torch.cuda.Event brackets do NOT necessarily match each op's real GPU
    # completion time. A profiled run (rocprofv3 --kernel-trace) on this exact
    # pipeline showed the real EpDispatchInterNodeV1Kernel/EpCombineInterNodeV1Kernel
    # durations can be far larger (and far more variable, up to ~150ms) than what
    # the bracket here measures (sub-ms), while the real gemm1/gemm2/quant/sorting
    # kernels inside fused_moe summed to only ~0.1ms per call versus a multi-ms
    # "moe" bracket -- i.e. the bracket boundaries likely don't line up with true
    # kernel start/end for these async/persistent-style mori kernels. Treat
    # dispatch_ms/moe_ms/combine_ms (and the derived GB/s/TFLOPS below) as a
    # measure of this script's host-observed critical path, not as validated
    # per-op GPU kernel time, until this is root-caused (check mori's
    # dispatch_combine.py for which stream these kernels run on and whether
    # they're fire-and-forget/polling rather than synchronous per call).
    # Agent: use four independent events per iteration so one iteration's
    # combine end event is never reused as the next iteration's start event.
    # Keep the measured loop barrier-free: per-iteration host/SHMEM barriers
    # perturb V1LL's steady-state pipeline and amplify rank-arrival skew.
    n_events = 4 * iters
    events = [torch.cuda.Event(enable_timing=True) for _ in range(n_events)]
    torch.cuda.synchronize()
    dist.barrier()
    for i in range(iters):
        event_base = 4 * i
        events[event_base].record()
        dispatched = backend.dispatch_prequant(x_fp4_bs, x_scale_bs, topk_weights_bs, topk_ids_bs)
        recv_tok_fp4 = dispatched.tokens
        recv_wts = dispatched.weights
        recv_scale = dispatched.scales
        recv_idx = dispatched.expert_ids
        recv_num_token = dispatched.num_tokens
        events[event_base + 1].record()
        moe_out = backend.fused_moe(dispatched)
        events[event_base + 2].record()
        combine_out, combine_out_wts = backend.combine(moe_out, dispatched)
        events[event_base + 3].record()
    torch.cuda.synchronize()

    # Discard the first (iters - stat_iters) rounds as JIT/cache warmup; average
    # only the trailing stat_iters rounds (user-requested: iters=100, stat over
    # the last 20).
    keep = max(1, min(stat_iters, iters))
    dispatch_ms = [events[4 * i].elapsed_time(events[4 * i + 1]) for i in range(iters)][-keep:]
    moe_ms = [events[4 * i + 1].elapsed_time(events[4 * i + 2]) for i in range(iters)][-keep:]
    combine_ms = [events[4 * i + 2].elapsed_time(events[4 * i + 3]) for i in range(iters)][-keep:]

    # mean, best-rank (min), worst-rank (max) across ranks: dispatch/combine/moe
    # are collective -- the group's real wall-clock latency is bounded by
    # whichever rank is slowest (stragglers are common under EP, since random
    # routing gives ranks uneven local_hits). mean alone hides that spread.
    dispatch_local = sum(dispatch_ms) / keep
    moe_local = sum(moe_ms) / keep
    combine_local = sum(combine_ms) / keep
    dispatch_avg = _reduce_float(dispatch_local, dist.ReduceOp.SUM) / world_size
    dispatch_min = _reduce_float(dispatch_local, dist.ReduceOp.MIN)
    dispatch_max = _reduce_float(dispatch_local, dist.ReduceOp.MAX)
    moe_avg = _reduce_float(moe_local, dist.ReduceOp.SUM) / world_size
    moe_min = _reduce_float(moe_local, dist.ReduceOp.MIN)
    moe_max = _reduce_float(moe_local, dist.ReduceOp.MAX)
    combine_avg = _reduce_float(combine_local, dist.ReduceOp.SUM) / world_size
    combine_min = _reduce_float(combine_local, dist.ReduceOp.MIN)
    combine_max = _reduce_float(combine_local, dist.ReduceOp.MAX)
    total_recv_avg = _reduce_float(float(total_recv), dist.ReduceOp.SUM) / world_size
    local_hits_avg = _reduce_float(float(local_hits), dist.ReduceOp.SUM) / world_size
    local_hits_max = _reduce_float(float(local_hits), dist.ReduceOp.MAX)
    active_experts_avg = (
        _reduce_float(float(active_local_experts), dist.ReduceOp.SUM) / world_size
    )
    active_experts_max = _reduce_float(float(active_local_experts), dist.ReduceOp.MAX)

    # ---- bandwidth / throughput conversions ----
    # dispatch/combine move total_recv rows across the fabric (XGMI+RDMA
    # combined, same convention as test_dispatch_combine_internode.py's
    # disp_total_bytes/comb_total_bytes: total_recv_num_token * hidden * elem_size).
    # GB/s reported for both mean (typical) and max (worst-rank/bottleneck) time.
    # Match AITER's tuning CSV convention: fp4x2 is accounted as one storage
    # byte per logical matrix element. This is an effective-BW convention,
    # not the physical nibble payload size.
    fp4_bytes_per_elem = 1.0
    bf16_bytes_per_elem = 2.0
    dispatch_bytes = total_recv_avg * model_dim * fp4_bytes_per_elem
    combine_bytes = total_recv_avg * model_dim * bf16_bytes_per_elem
    dispatch_bytes_local = total_recv * model_dim * fp4_bytes_per_elem
    combine_bytes_local = total_recv * model_dim * bf16_bytes_per_elem

    def _gbps(nbytes, ms):
        return nbytes / 1e9 / (ms / 1e3) if ms > 0 else 0.0

    # Match MORI's aggregation: compute payload/time for every rank and
    # iteration first, then average. Do not divide average payload by average
    # latency because mean(bytes/time) is not bytes_mean/time_mean.
    dispatch_gbps_local = sum(_gbps(dispatch_bytes_local, ms) for ms in dispatch_ms) / keep
    combine_gbps_local = sum(_gbps(combine_bytes_local, ms) for ms in combine_ms) / keep
    dispatch_gbps = _reduce_float(dispatch_gbps_local, dist.ReduceOp.SUM) / world_size
    dispatch_gbps_best = _reduce_float(dispatch_gbps_local, dist.ReduceOp.MAX)
    dispatch_gbps_worst = _reduce_float(dispatch_gbps_local, dist.ReduceOp.MIN)
    combine_gbps = _reduce_float(combine_gbps_local, dist.ReduceOp.SUM) / world_size
    combine_gbps_best = _reduce_float(combine_gbps_local, dist.ReduceOp.MAX)
    combine_gbps_worst = _reduce_float(combine_gbps_local, dist.ReduceOp.MIN)

    # Profile the public TestWideEpMoe forward API. In compile mode deliberately
    # wrap that public interface directly rather than introducing another
    # fused_moe facade in the implementation.
    profiled_call = op.forward_prequant
    profile_args = (x_fp4_bs, x_scale_bs, topk_weights_bs, topk_ids_bs)
    pipeline_kind = "eager_full_pipeline"
    if torch_compile_cudagraph:
        profiled_call = torch.compile(op.forward_prequant, backend="cudagraphs")
        pipeline_kind = "torch_compile_cudagraph_test_wide_ep_forward"

        # Validate the compiled public callable itself.  The eager correctness
        # checks above do not prove that graph capture/replay preserves output.
        compiled_out = profiled_call(*profile_args).clone()
        torch.cuda.synchronize()
        assert compiled_out.shape == out.shape
        assert torch.isfinite(compiled_out.float()).all()
        torch.testing.assert_close(compiled_out, out, rtol=1e-2, atol=2.5e-1)
        _barrier()
        if rank == 0:
            print(f"[EP16-torch-compile] bs={bs} output check PASS", flush=True)

    if torch_profiler_dir:
        from torch.profiler import ProfilerActivity, profile, record_function

        for _ in range(profile_warmup_iters):
            profiled_call(*profile_args)
        torch.cuda.synchronize()
        _barrier()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as prof:
            for profile_iter in range(profile_iters):
                with record_function(
                    f"megamoe_ep16_{pipeline_kind}_bs{bs}_iter{profile_iter}"
                ):
                    profiled_call(*profile_args)
            torch.cuda.synchronize()
        os.makedirs(torch_profiler_dir, exist_ok=True)
        prof.export_chrome_trace(
            os.path.join(torch_profiler_dir, f"rank{rank}_bs{bs}.json")
        )
        if rank == 0:
            print(
                f"[EP16-torch-profiler] bs={bs} traces={torch_profiler_dir} "
                f"iters={profile_iters} pipeline={pipeline_kind}",
                flush=True,
            )

    # MoE: exact logical-M FLOPs (grouped GEMM1 gate+up, GEMM2 down) -> TFLOPS.
    #
    # Memory bandwidth: "effective bandwidth = minimum required bytes / time"
    # (A + B + C, each counted once per GEMM) -- the same convention aiter's own
    # GEMM/MoE-GEMM benchmarks use (op_tests/test_gemm_a4w4.py: `(x.nbytes + w.nbytes)
    # / us`; op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py: mem_read(x,w,scales)
    # + mem_write(out); op_tests/op_benchmarks/triton/bench_moe_gemm_a4w4_cudagraph.py:
    # per-GEMM activation + `w_bytes` for only the *routed*/active experts + output,
    # with moe1/moe2 summed for the "total" row). fused_moe here is confirmed 2-stage
    # (its own log prints "using 2stage default" -- gemm1 gate+up and gemm2 down are
    # two separate kernel launches), so the intermediate hidden activation genuinely
    # round-trips through HBM: written once by gemm1 (its C), read back once by gemm2
    # (its A) -- not a guess, both gemms' bytes are summed the way aiter's own bench
    # sums moe1_bytes + moe2_bytes for "total":
    #   gemm1: A=recv tokens (model_dim) + B=w1 (active experts only) + C=hidden (inter_dim)
    #   gemm2: A=hidden (inter_dim) + B=w2 (active experts only) + C=output tokens (model_dim)
    # Weight bytes counted ONCE per GEMM (not per M-tile -- that would require
    # assumptions about internal kernel tiling/residency this script can't verify),
    # scoped to `active_experts` (local experts that actually received >=1 token this
    # bs, not all `local_experts` -- some may see zero at small bs).
    def _moe_flops(hits):
        return 2.0 * hits * model_dim * (2 * inter_dim) + 2.0 * hits * inter_dim * model_dim

    def _moe_bytes(received_tokens, _active_experts):
        # Same aggregate convention as gemm_moe_tune.py: input + both local
        # weight matrices + final output, divided by GEMM1+GEMM2 duration.
        return (
            received_tokens * model_dim * fp4_bytes_per_elem
            + local_experts * (2 * inter_dim) * model_dim * fp4_bytes_per_elem
            + local_experts * model_dim * inter_dim * fp4_bytes_per_elem
            + received_tokens * model_dim * bf16_bytes_per_elem
        )

    moe_tflops = _moe_flops(local_hits_avg) / (gemm_total_us * 1e6) if gemm_total_us > 0 else 0.0
    moe_tflops_best = (
        _moe_flops(local_hits_max) / (gemm_total_min_us * 1e6) if gemm_total_min_us > 0 else 0.0
    )
    moe_tflops_worst = (
        _moe_flops(local_hits_max) / (gemm_total_max_us * 1e6) if gemm_total_max_us > 0 else 0.0
    )
    moe_gbps = _gbps(_moe_bytes(total_recv_avg, active_experts_avg), gemm_total_us / 1000)
    moe_gbps_best = _gbps(_moe_bytes(total_recv_avg, active_experts_avg), gemm_total_min_us / 1000)
    moe_gbps_worst = _gbps(_moe_bytes(total_recv_avg, active_experts_avg), gemm_total_max_us / 1000)

    if rank == 0:
        print(
            f"[EP16-a4w4] bs={bs} total_recv~{total_recv_avg:.0f} local_hits~{local_hits_avg:.0f} "
            f"relL2={rel_l2:.6f} (rtol={rtol}, {'checked' if bs <= accuracy_max_bs else 'skipped'}, "
            f"stat over last {keep}/{iters} iters)\n"
            f"  dispatch: {dispatch_avg:.4f}/{dispatch_min:.4f}/{dispatch_max:.4f}ms mean/best/worst  "
            f"{dispatch_gbps:.2f}/{dispatch_gbps_best:.2f}/{dispatch_gbps_worst:.2f} GB/s mean/best/worst\n"
            f"  moe     : {moe_avg:.4f}/{moe_min:.4f}/{moe_max:.4f}ms mean/best/worst  "
            f"{moe_tflops:.2f}/{moe_tflops_best:.2f}/{moe_tflops_worst:.2f} TFLOPS mean/best/worst  "
            f"{moe_gbps:.2f}/{moe_gbps_best:.2f}/{moe_gbps_worst:.2f} GB/s mean/best/worst\n"
            f"  combine : {combine_avg:.4f}/{combine_min:.4f}/{combine_max:.4f}ms mean/best/worst  "
            f"{combine_gbps:.2f}/{combine_gbps_best:.2f}/{combine_gbps_worst:.2f} GB/s mean/best/worst\n"
            f"  kernels : gemm1={gemm1_us:.2f}us gemm2={gemm2_us:.2f}us "
            f"sum={gemm1_us + gemm2_us:.2f}us",
            flush=True,
        )
        if perf_out:
            record = {
                "category": "ep16_a4w4_moe",
                "params": {
                    "world_size": world_size,
                    "bs": bs,
                    "experts": NETWORK["experts"],
                    "local_experts": local_experts,
                    "topk": NETWORK["topk"],
                    "model_dim": model_dim,
                    "inter_dim": inter_dim,
                    "quant_type": "per_1x32_a4w4",
                },
                "stat_iters": keep,
                "total_iters": iters,
                "metrics": {
                    "dispatch_avg_ms": round(dispatch_avg, 4),
                    "dispatch_best_ms": round(dispatch_min, 4),
                    "dispatch_worst_ms": round(dispatch_max, 4),
                    "dispatch_gbps_mean": round(dispatch_gbps, 2),
                    "dispatch_gbps_best": round(dispatch_gbps_best, 2),
                    "dispatch_gbps_worst": round(dispatch_gbps_worst, 2),
                    "moe_avg_ms": round(moe_avg, 4),
                    "moe_best_ms": round(moe_min, 4),
                    "moe_worst_ms": round(moe_max, 4),
                    "moe_tflops_mean": round(moe_tflops, 2),
                    "moe_tflops_best": round(moe_tflops_best, 2),
                    "moe_tflops_worst": round(moe_tflops_worst, 2),
                    "moe_gbps_mean": round(moe_gbps, 2),
                    "moe_gbps_best": round(moe_gbps_best, 2),
                    "moe_gbps_worst": round(moe_gbps_worst, 2),
                    "combine_avg_ms": round(combine_avg, 4),
                    "combine_best_ms": round(combine_min, 4),
                    "combine_worst_ms": round(combine_max, 4),
                    "combine_gbps_mean": round(combine_gbps, 2),
                    "combine_gbps_best": round(combine_gbps_best, 2),
                    "combine_gbps_worst": round(combine_gbps_worst, 2),
                    "total_recv": round(total_recv_avg, 1),
                    "local_hits_mean": round(local_hits_avg, 1),
                    "local_hits_max": round(local_hits_max, 1),
                    "active_experts_mean": round(active_experts_avg, 1),
                    "active_experts_max": round(active_experts_max, 1),
                    "moe_rel_l2": None if rel_l2 < 0 else round(rel_l2, 6),
                    "gemm1_us": round(gemm1_us, 2),
                    "gemm2_us": round(gemm2_us, 2),
                },
                "ts": time.time(),
            }
            os.makedirs(os.path.dirname(os.path.abspath(perf_out)), exist_ok=True)
            with open(perf_out, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, sort_keys=True) + "\n")


def run_ep16_a4w4(
    local_rank,
    bs_list,
    iters,
    stat_iters,
    seed,
    rtol,
    accuracy_max_bs,
    routing,
    gpu_per_node,
    node_rank,
    num_nodes,
    torch_profiler_dir,
    torch_compile_cudagraph,
    profile_warmup_iters,
    profile_iters,
    staged_only,
):
    world_size = num_nodes * gpu_per_node
    rank = node_rank * gpu_per_node + local_rank
    device = _setup_dist(rank, world_size, local_rank)
    perf_out = os.environ.get("MORI_PERF_OUT") if rank == 0 else None

    try:
        network = NETWORK
        experts = network["experts"]
        model_dim = network["model_dim"]
        inter_dim = network["inter_dim"]
        topk = network["topk"]
        if experts % world_size != 0:
            raise ValueError(f"experts={experts} must be divisible by world_size={world_size}")
        local_experts = experts // world_size
        local_expert_start = rank * local_experts
        local_expert_end = local_expert_start + local_experts

        max_bs = max(bs_list)

        # ---- Build all tensors up front, before any dispatch/moe/combine/timing ----
        x, topk_weights, topk_ids = _make_local_inputs(
            max_bs, model_dim, experts, topk, rank, seed, device,
            routing=routing, max_tok_anchor=max_bs,
            world_size=world_size, gpu_per_node=gpu_per_node,
        )
        if os.environ.get("MORI_COMM_ONLY", "0") == "1":
            # The inter-node backend does not touch weights in comm-only mode.
            # Avoid allocating/quantizing the multi-GB MoE weights.
            w1_a = w2_a = torch.empty(1, dtype=dtypes.fp4x2, device=device)
            w1_s = w2_s = torch.empty(1, dtype=torch.uint8, device=device)
            w1_qt = w2_qt = w1_scale = w2_scale = w1_a
        else:
            (w1_a, w1_s, w2_a, w2_s), (w1_qt, w1_scale, w2_qt, w2_scale) = (
                _quantize_local_weights(
                    model_dim, inter_dim, local_experts, rank, seed, device
                )
            )
        expert_mask = _build_expert_mask(experts, local_expert_start, local_expert_end, device)

        if rank == 0:
            print(f"[EP16-a4w4] routing={routing!r}", flush=True)

        torch_quant_act = aiter.get_torch_quant(QuantType.per_1x32)
        x_fp4, x_scale = torch_quant_act(x, quant_dtype=dtypes.fp4x2)
        x_fp4 = x_fp4.view(max_bs, model_dim // 2)

        op = TestWideEpMoe(
            rank=rank,
            world_size=world_size,
            model_dim=model_dim,
            inter_dim=inter_dim,
            experts=experts,
            topk=topk,
            quant="a4w4",
            w1=w1_a,
            w1_scale=w1_s,
            w2=w2_a,
            w2_scale=w2_s,
            max_tok_per_rank=max_bs,
        )

        _barrier()

        for bs in bs_list:
            _run_one_bs(
                bs,
                op,
                x_fp4,
                x_scale,
                topk_weights,
                topk_ids,
                w1_a,
                w1_s,
                w2_a,
                w2_s,
                w1_qt,
                w1_scale,
                w2_qt,
                w2_scale,
                expert_mask,
                local_experts,
                model_dim,
                inter_dim,
                world_size,
                iters,
                stat_iters,
                rtol,
                accuracy_max_bs,
                rank,
                perf_out,
                torch_profiler_dir,
                torch_compile_cudagraph,
                profile_warmup_iters,
                profile_iters,
                staged_only,
            )
    finally:
        _cleanup()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs-list", default="128,512,1024,2048,4096")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument(
        "--stat-iters",
        type=int,
        default=20,
        help="average over only the trailing N of --iters rounds (discard the rest as warmup)",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--rtol", type=float, default=0.15)
    parser.add_argument("--accuracy-max-bs", type=int, default=512)
    parser.add_argument(
        "--torch-profiler-dir",
        default=None,
        help="export one full-pipeline torch.profiler Chrome trace per rank and BS",
    )
    parser.add_argument(
        "--torch-compile-cudagraph",
        action="store_true",
        help=(
            "apply torch.compile(backend='cudagraphs') directly to the public "
            "TestWideEpMoe.forward_prequant interface"
        ),
    )
    parser.add_argument("--profile-warmup-iters", type=int, default=10)
    parser.add_argument("--profile-iters", type=int, default=40)
    parser.add_argument(
        "--staged-only",
        action="store_true",
        help="run one synchronized dispatch/fused_moe/combine diagnostic and stop",
    )
    parser.add_argument(
        "--routing",
        choices=["random", "round_robin", "cross_node"],
        default="random",
        help=(
            "random: real fused_topk over random router logits (statistically "
            "balanced, natural per-rank variance). round_robin: fully-balanced "
            "deterministic assignment (mori's gen_test_data(routing='round_robin') "
            "convention) -- NOTE: tends to cluster a token's whole topk onto one "
            "rank, see docstring. cross_node: each token's topk slots alternate "
            "between node 0's and node 1's expert range (spreads every token "
            "across both nodes), round-robin within each node."
        ),
    )
    args = parser.parse_args()

    bs_list = [int(v) for v in args.bs_list.split(",") if v]
    if not bs_list or min(bs_list) <= 0:
        raise ValueError("--bs-list must contain positive integers")

    gpu_per_node = int(os.environ.get("GPU_PER_NODE", GPU_PER_NODE_DEFAULT))
    num_nodes = int(os.environ["WORLD_SIZE"])
    node_rank = int(os.environ["RANK"])

    torch.multiprocessing.spawn(
        run_ep16_a4w4,
        args=(
            bs_list,
            args.iters,
            args.stat_iters,
            args.seed,
            args.rtol,
            args.accuracy_max_bs,
            args.routing,
            gpu_per_node,
            node_rank,
            num_nodes,
            args.torch_profiler_dir,
            args.torch_compile_cudagraph,
            args.profile_warmup_iters,
            args.profile_iters,
            args.staged_only,
        ),
        nprocs=gpu_per_node,
        join=True,
    )


if __name__ == "__main__":
    main()
