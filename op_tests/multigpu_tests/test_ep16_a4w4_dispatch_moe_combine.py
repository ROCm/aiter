# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""EP16 (2 nodes x 8 GPUs) a4w4 internode dispatch + fusedmoe + combine test.

Pipeline: fp4 dispatch (mori InterNodeV1) -> local fp4->bf16 dequant ->
aiter fused_moe (a4w4_mxfp4, per_1x32 quant, real expert_mask) -> bf16 combine
(mori InterNodeV1, same op instance).

Launch (2 nodes, one torchrun process per node, 8 local GPUs spawned inside):

  On node_rank 0:
    GPU_PER_NODE=8 torchrun --nnodes=2 --node_rank=0 --nproc_per_node=1 \
        --master_addr=<node0_ip> --master_port=29500 \
        test_ep16_a4w4_dispatch_moe_combine.py --bs-list 128,512,1024,2048,4096
  On node_rank 1: same with --node_rank=1
"""

from __future__ import annotations

import argparse
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
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
from aiter.ops.shuffle import shuffle_weight
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
    torch.cuda.synchronize()
    ms.shmem_barrier_all()
    # Agent: shmem_barrier_all may enqueue device work. Drain it here so a
    # caller that records a CUDA event immediately after _barrier() does not
    # accidentally charge the barrier kernel to the following stage.
    torch.cuda.synchronize()


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

    w1 = (
        torch.randn(
            (local_experts, 2 * inter_dim, model_dim),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        * 0.1
    )
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

    w1_a = shuffle_weight(w1_qt, layout=(16, 16))
    w2_a = shuffle_weight(w2_qt, layout=(16, 16))
    w1_s = fp4_utils.e8m0_shuffle(w1_scale)
    w2_s = fp4_utils.e8m0_shuffle(w2_scale)
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
):
    x_fp4_bs = x_fp4[:bs].contiguous()
    x_scale_bs = x_scale[:bs].contiguous()
    topk_weights_bs = topk_weights[:bs].contiguous()
    topk_ids_bs = topk_ids[:bs].contiguous()

    # Public MegaMoEV2 contract: the facade owns the complete inter-node
    # dispatch -> fused_moe -> combine sequence.
    out = op.forward_prequant(x_fp4_bs, x_scale_bs, topk_weights_bs, topk_ids_bs).clone()
    torch.cuda.synchronize()
    assert out.shape == (bs, model_dim)
    assert torch.isfinite(out.float()).all(), "MegaMoEV2 output has non-finite values"

    # Diagnostic-only staged call through the private backend. This preserves
    # per-stage profiling without adding stage methods to MegaMoEV2's public API.
    backend = op._inter_node
    assert backend is not None
    dispatched = backend.dispatch_prequant(x_fp4_bs, x_scale_bs, topk_weights_bs, topk_ids_bs)
    recv_tok_fp4 = dispatched.tokens
    recv_wts = dispatched.weights
    recv_scale = dispatched.scales
    recv_idx = dispatched.expert_ids
    recv_num_token = dispatched.num_tokens
    torch.cuda.synchronize()
    total_recv = int(recv_num_token[0].item())

    moe_out = backend.fused_moe(dispatched)

    # combine()'s indices/weights must be THIS rank's own [tokens, topk]
    # routing passed to dispatch() -- NOT dispatch()'s returned recv_idx/
    # recv_wts (ROCm/mori#475). weights=None: fused_moe already applied
    # topk weighting in stage2 (same convention as
    # test_dispatch_combine_internode.py's run_combine).
    combine_out, combine_out_wts = backend.combine(moe_out, dispatched)
    torch.cuda.synchronize()
    diagnostic_out = combine_out[:bs]
    torch.testing.assert_close(out, diagnostic_out, rtol=0, atol=0)

    # ---- correctness (only below accuracy_max_bs, all-gather ref is O(bs*world)) ----
    rel_l2 = -1.0
    if bs <= accuracy_max_bs:
        # Reference-only dequantization. The measured production path passes
        # dispatch's packed FP4 activation and E8M0 scale directly to fused_moe.
        recv_tok_bf16 = _dequant_tokens(recv_tok_fp4, recv_scale, model_dim)
        w1_deq = _dequant_weight(w1_qt, w1_scale, (local_experts, 2 * inter_dim, model_dim))
        w2_deq = _dequant_weight(w2_qt, w2_scale, (local_experts, model_dim, inter_dim))
        ref_moe_out = _torch_moe_situv2_reference(
            recv_tok_bf16[:total_recv],
            w1_deq,
            w2_deq,
            recv_wts[:total_recv],
            recv_idx[:total_recv],
            expert_mask,
        )
        rel_l2 = float(
            torch.linalg.vector_norm((moe_out[:total_recv] - ref_moe_out).float())
            / torch.linalg.vector_norm(ref_moe_out.float())
        )
        rel_l2 = _reduce_float(rel_l2, dist.ReduceOp.MAX)
        if rel_l2 >= rtol:
            raise AssertionError(f"bs={bs} moe relL2={rel_l2:.6f} exceeds rtol={rtol}")
        assert diagnostic_out.shape == (bs, model_dim)
        assert torch.isfinite(diagnostic_out.float()).all(), "combine output has non-finite values"

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
    fp4_bytes_per_elem = 0.5  # float4_e2m1fn_x2: 2 packed values per byte
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
    per_expert_weight_bytes = (
        w1_a.numel() * w1_a.element_size() + w2_a.numel() * w2_a.element_size()
    ) / local_experts

    def _moe_flops(hits):
        return 2.0 * hits * model_dim * (2 * inter_dim) + 2.0 * hits * inter_dim * model_dim

    def _moe_bytes(hits, active_experts):
        gemm1_bytes = (
            hits * model_dim * bf16_bytes_per_elem  # A: recv tokens read into gemm1
            + per_expert_weight_bytes * active_experts  # B: w1 (+w2, folded in below)
            + hits * inter_dim * bf16_bytes_per_elem  # C: hidden activation written
        )
        gemm2_bytes = (
            hits * inter_dim * bf16_bytes_per_elem  # A: hidden activation read back
            # B (w2) already folded into per_expert_weight_bytes above -- don't double count
            + hits * model_dim * bf16_bytes_per_elem  # C: fused_moe output tokens written
        )
        return gemm1_bytes + gemm2_bytes

    moe_tflops = _moe_flops(local_hits_avg) / 1e12 / (moe_avg / 1e3) if moe_avg > 0 else 0.0
    moe_tflops_best = (
        _moe_flops(local_hits_max) / 1e12 / (moe_min / 1e3) if moe_min > 0 else 0.0
    )
    moe_tflops_worst = (
        _moe_flops(local_hits_max) / 1e12 / (moe_max / 1e3) if moe_max > 0 else 0.0
    )
    moe_gbps = _gbps(_moe_bytes(local_hits_avg, active_experts_avg), moe_avg)
    moe_gbps_best = _gbps(_moe_bytes(local_hits_max, active_experts_max), moe_min)
    moe_gbps_worst = _gbps(_moe_bytes(local_hits_max, active_experts_max), moe_max)

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
            f"{combine_gbps:.2f}/{combine_gbps_best:.2f}/{combine_gbps_worst:.2f} GB/s mean/best/worst",
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
        (w1_a, w1_s, w2_a, w2_s), (w1_qt, w1_scale, w2_qt, w2_scale) = (
            _quantize_local_weights(model_dim, inter_dim, local_experts, rank, seed, device)
        )
        expert_mask = _build_expert_mask(experts, local_expert_start, local_expert_end, device)

        if rank == 0:
            print(f"[EP16-a4w4] routing={routing!r}", flush=True)

        torch_quant_act = aiter.get_torch_quant(QuantType.per_1x32)
        x_fp4, x_scale = torch_quant_act(x, quant_dtype=dtypes.fp4x2)
        x_fp4 = x_fp4.view(max_bs, model_dim // 2)

        op = MegaMoEV2(
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
        ),
        nprocs=gpu_per_node,
        join=True,
    )


if __name__ == "__main__":
    main()
