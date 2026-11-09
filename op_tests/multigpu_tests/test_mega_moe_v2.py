# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
r"""MegaMoEV2 multi-GPU accuracy and performance test.

--suite size (default): one instance per batch size (or one at --max-tok-per-rank),
pinned to the staged route, or to the large-input route with --expect-large-m.
Accuracy against an fp32 reference, explicit and null streams, CUDA graph replay,
per-stage timing, and the opt-in edge cases (pad tail, burst, layer-slot stress,
skewed route switch, adversarial routes, --compare-main).

--suite routes: one instance as production builds it, at --mtpr (default 8192):
the staged, small-input (MTPR 256 and 512) and large-input routes side by side,
each call's route picked by forward(max_tokens=...). Accuracy against the fp32
reference and the pinned staged route (uneven, idle, padded ranks), per-layer
weight rebinding with interleaved staged calls, the 32-bit epoch wrap with zero
rows, CUDA graph replay against eager, and paired timing against the staged route.
The fused routes need an a8w4 layout they support on an 8-XCD gfx950: EP8
DeepSeek-V4-Pro (--network v4_pro, 8 ranks) or EP4 DeepSeek-V4.1-Flash
(--network v41_flash, 4 ranks).

    torchrun --nproc-per-node 8 test_mega_moe_v2.py --bs-list 1,64,128
    torchrun --nproc-per-node 8 test_mega_moe_v2.py --max-tok-per-rank 8192 \
        --expect-large-m --compare-main --bs-list 1,64,513,8192 --accuracy-max-bs 8192
    torchrun --nproc-per-node 8 test_mega_moe_v2.py --suite routes
    # every timed route also checked against the reference:
    torchrun --nproc-per-node 4 test_mega_moe_v2.py --suite routes --network v41_flash --check-time-acc
    # against a same-MTPR staged route, timing only:
    torchrun --nproc-per-node 8 test_mega_moe_v2.py --suite routes \
        --mtpr 256 --small-m-mtprs 256 --time-only --time-bs 140,168,196,224,252

The routes suite takes the flags of the former test_mega_moe_small_m.py; its
--mtpr and --slots are aliases of --max-tok-per-rank and --large-m-state-slots.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")

import mori.shmem as ms
import torch
import torch.distributed as dist
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
    ROUTE_LARGE_M,
    ROUTE_SMALL_M,
    ROUTE_STAGED,
    fused_layout_supported,
    small_m_cu_supported,
)
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility import fp4_utils

NETWORKS = {
    # swiglu_limit 0.0 exercises the operator's default, unclamped SwiGLU.
    "glm52": {
        "model_dim": 6144,
        "inter_dim": 2048,
        "experts": 256,
        "topk": 8,
        "swiglu_limit": 0.0,
    },
    "v4_pro": {
        "model_dim": 7168,
        "inter_dim": 3072,
        "experts": 384,
        "topk": 6,
        "swiglu_limit": 10.0,
    },
    "v41_flash": {
        "model_dim": 5120,
        "inter_dim": 2304,
        "experts": 384,
        "topk": 6,
        "swiglu_limit": 10.0,
    },
    "r1_v3": {
        "model_dim": 7168,
        "inter_dim": 2048,
        "experts": 256,
        "topk": 8,
        "swiglu_limit": 0.0,
    },
    # Kimi-K3 routing/weight geometry.  This exercises topk16 and EP8/epr112 and
    # the real K3 activation: SiTUv2 (hidden_act="situ"), which self-saturates
    # via tanh, so it carries no hard swiglu clamp.  beta=4.0 / linear_beta=25.0
    # are the K3 checkpoint constants the MegaMoE SiTU kernel bakes in (SGLang
    # kimi_k3.py asserts exactly these for the MegaMoE a2a backend).
    "kimi_k3_route": {
        "model_dim": 3584,
        "inter_dim": 3072,
        "experts": 896,
        "topk": 16,
        "swiglu_limit": 0.0,
        "act": "situv2",
        "situ_beta": 4.0,
        "situ_linear_beta": 25.0,
    },
}


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


def _barrier():
    torch.cuda.synchronize()
    ms.shmem_barrier_all()


def _reduce_float(value, device, op):
    result = torch.tensor(float(value), dtype=torch.float32, device=device)
    dist.all_reduce(result, op=op)
    return float(result.item())


def _rel_l2(output, reference):
    """Relative L2 error over all ranks; -1.0 when the reference is all zero."""
    reference = reference.float()
    error_sq = torch.sum((output.float() - reference) ** 2)
    reference_sq = torch.sum(reference**2)
    dist.all_reduce(error_sq)
    dist.all_reduce(reference_sq)
    if not reference_sq.item():
        return -1.0
    return float(torch.sqrt(error_sq / reference_sq))


def _all_ranks(flag, device):
    """Whether ``flag`` holds on every rank."""
    return bool(_reduce_float(bool(flag), device, dist.ReduceOp.MIN))


def _fwd(moe, *args, **kwargs):
    """forward() planned for the instance capacity (the EP-wide row bound of every call
    here): the staged route, or the large-input route when the instance has it."""
    return moe.forward(*args, max_tokens=moe.mtpr, **kwargs)


def _next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()


def _make_inputs(
    tokens,
    model_dim,
    experts,
    topk,
    rank,
    seed,
    device,
    *,
    force_fanout_boundary,
    inject_invalid_route,
    force_padding_boundary,
    concentrate_experts=0,
    zipf=0.0,
):
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn(
        (tokens, model_dim), dtype=torch.bfloat16, device=device, generator=generator
    )
    scores = torch.randn(
        (tokens, experts), dtype=torch.float32, device=device, generator=generator
    )
    if zipf > 0:
        # Zipf expert-popularity skew (realistic MoE load imbalance): bias scores
        # by -zipf*ln(rank+1) over a seed-fixed permutation shared across ranks,
        # so a few experts attract most tokens (spread across expert ids).
        pgen = torch.Generator(device=device).manual_seed(seed)
        perm = torch.randperm(experts, generator=pgen, device=device)
        rank_of = torch.empty(experts, dtype=torch.float32, device=device)
        rank_of[perm] = torch.arange(experts, dtype=torch.float32, device=device)
        scores = scores + (-float(zipf) * torch.log(rank_of + 1.0))[None, :]
    values, ids = torch.topk(scores, topk, dim=-1)
    if concentrate_experts:
        # OOB repro: route every token's top-k slots to experts [0, topk) which
        # all live on rank 0, so rank 0's route-major receive buffer must hold
        # (sum of sender tokens) * topk rows -- the cross-rank skew the local
        # payload-capacity guard cannot see.
        ids = (
            torch.arange(topk, dtype=torch.int64, device=device)
            .expand(tokens, topk)
            .clone()
        )
    small_pair_boundary = (
        force_fanout_boundary
        and not (inject_invalid_route or force_padding_boundary)
        and 2 <= topk <= 8
        and topk % 2 == 0
        and topk // 2 <= dist.get_world_size()
        and experts % dist.get_world_size() == 0
    )
    if small_pair_boundary:
        # Distinct paired experts in multiple destinations force fanout rows
        # and expert padding for the small-topk (e.g. DSv4.1) route format.
        world = dist.get_world_size()
        local_experts = experts // world
        if local_experts < 2:
            raise ValueError("paired fanout requires at least two local experts")
        token = torch.arange(tokens, dtype=torch.int64, device=device)[:, None]
        slot = torch.arange(topk, dtype=torch.int64, device=device)[None, :]
        destination = (token + rank + slot // 2) % world
        ids = destination * local_experts + local_experts - 2 + slot % 2
    elif force_fanout_boundary or inject_invalid_route or force_padding_boundary:
        if topk != 16 or experts % dist.get_world_size():
            raise ValueError("fanout adversarial cases require topk=16 and EP")
        world = dist.get_world_size()
        local_experts = experts // world
        if local_experts < 112:
            raise ValueError("fanout adversarial cases require at least 112 epr")
        token = torch.arange(tokens, dtype=torch.int64, device=device)[:, None]
        slot = torch.arange(topk, dtype=torch.int64, device=device)[None, :]
        if force_padding_boundary:
            if tokens != 1:
                raise ValueError("--force-padding-boundary requires one token")
            local = (rank * topk + slot) % local_experts
            ids = local.expand(tokens, -1).clone()
            if rank == world - 1:
                ids[:, :-2] = torch.arange(
                    1, topk - 1, dtype=torch.int64, device=device
                )
                ids[:, -2] = 0
                ids[:, -1] = local_experts - 1
        else:
            destination = (token + rank) % world
            local = (token * 17 + slot * 7) % (local_experts - 2)
            if inject_invalid_route:
                local = local + 1
            ids = destination * local_experts + local
            if inject_invalid_route:
                ids[:, -2] = -1
                ids[:, -1] = destination[:, 0] * local_experts + local_experts - 1
            else:
                ids[:, -2] = destination[:, 0] * local_experts + local_experts - 2
                ids[:, -1] = destination[:, 0] * local_experts + local_experts - 1
    return (
        x.contiguous(),
        values.softmax(dim=-1).contiguous(),
        ids.to(torch.int32).contiguous(),
    )


def _quantize_weights(model_dim, inter_dim, local_experts, rank, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed + 1000 + rank)
    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x32)

    w1 = torch.randn(
        (local_experts, 2 * inter_dim, model_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w1.mul_(model_dim**-0.25)
    w1_q, w1_scale = quantize(w1, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(local_experts, 2 * inter_dim, model_dim // 2)
    w1_scale_ref = w1_scale.view(local_experts, 2 * inter_dim, model_dim // 32)
    del w1
    w1_kernel = shuffle_weight_a16w4(w1_q, 16, True).contiguous()
    w1_scale_kernel = shuffle_scale_a16w4(w1_scale, local_experts, True).contiguous()

    w2 = torch.randn(
        (local_experts, model_dim, inter_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w2.mul_(inter_dim**-0.25)
    w2_q, w2_scale = quantize(w2, quant_dtype=dtypes.fp4x2)
    w2_q = w2_q.view(local_experts, model_dim, inter_dim // 2)
    w2_scale_ref = w2_scale.view(local_experts, model_dim, inter_dim // 32)
    del w2
    w2_kernel = shuffle_weight_a16w4(w2_q, 16, False).contiguous()
    w2_scale_kernel = shuffle_scale_a16w4(w2_scale, local_experts, False).contiguous()
    torch.cuda.empty_cache()
    return (
        w1_kernel,
        w1_scale_kernel,
        w2_kernel,
        w2_scale_kernel,
        w1_q,
        w1_scale_ref,
        w2_q,
        w2_scale_ref,
    )


def _dequant_expert(weight, scale, rows, cols):
    values = fp4_utils.mxfp4_to_f32(weight).view(rows, cols)
    scales = fp4_utils.e8m0_to_f32(scale).view(rows, cols // 32)
    return values * scales.repeat_interleave(32, dim=-1)


def _all_gather_variable(tensor, token_counts=None):
    world = dist.get_world_size()
    if token_counts is None:
        local_count = torch.tensor(
            tensor.shape[0], dtype=torch.int64, device=tensor.device
        )
        gathered_counts = [torch.empty_like(local_count) for _ in range(world)]
        dist.all_gather(gathered_counts, local_count)
        token_counts = [int(count.item()) for count in gathered_counts]
    max_count = max(token_counts)
    if tensor.shape[0] < max_count:
        padding = torch.empty(
            (max_count - tensor.shape[0], *tensor.shape[1:]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        tensor = torch.cat((tensor, padding))
    gathered = [torch.empty_like(tensor) for _ in range(world)]
    dist.all_gather(gathered, tensor)
    return (
        torch.cat([value[:count] for value, count in zip(gathered, token_counts)]),
        token_counts,
    )


@torch.no_grad()
def _reference(
    x,
    route_weights,
    ids,
    ref_weights,
    rank,
    world,
    model_dim,
    inter_dim,
    experts,
    swiglu_limit,
    act="silu",
    situ_beta=1.0,
    situ_linear_beta=1.0,
):
    x_all, token_counts = _all_gather_variable(x)
    weights_all, _ = _all_gather_variable(route_weights, token_counts)
    ids_all, _ = _all_gather_variable(ids, token_counts)
    partial = torch.zeros(
        (x_all.shape[0], model_dim), dtype=torch.float32, device=x.device
    )
    w1_q, w1_scale, w2_q, w2_scale = ref_weights
    local_experts = experts // world
    expert_start = rank * local_experts
    expert_end = expert_start + local_experts
    active = torch.unique(ids_all)
    active = active[(active >= expert_start) & (active < expert_end)]
    for expert in active.tolist():
        positions = torch.nonzero(ids_all == expert, as_tuple=False)
        rows, slots = positions[:, 0], positions[:, 1]
        local_id = expert - expert_start
        w1 = _dequant_expert(
            w1_q[local_id], w1_scale[local_id], 2 * inter_dim, model_dim
        )
        w2 = _dequant_expert(w2_q[local_id], w2_scale[local_id], model_dim, inter_dim)
        inp = x_all[rows].float()
        gate = inp @ w1[:inter_dim].T
        up = inp @ w1[inter_dim:].T
        if act == "situv2":
            # SiTUv2 (Kimi-K3 hidden_act="situ"), matching SGLang SituAndMul and
            # the operator's no-clamp situv2 epilogue: tanh self-saturates.
            gate = situ_beta * torch.tanh(gate / situ_beta) * torch.sigmoid(gate)
            up = situ_linear_beta * torch.tanh(up / situ_linear_beta)
            hidden = gate * up
        else:
            # MegaMoEV2 treats swiglu_limit <= 0 as "no clamp"; clamping here
            # unconditionally turns up into zeros at the operator's own default of
            # 0.0, so the reference is all-zero and relL2 divides by zero.
            if swiglu_limit > 0:
                gate = gate.clamp(max=swiglu_limit)
                up = up.clamp(-swiglu_limit, swiglu_limit)
            hidden = F.silu(gate) * up
        out = (hidden @ w2.T) * weights_all[rows, slots, None]
        partial.index_add_(0, rows, out)
        del w1, w2, inp, hidden, out
    dist.all_reduce(partial)
    start = sum(token_counts[:rank])
    return partial[start : start + x.shape[0]]


def _moe_reference(moe, x, route_weights, ids, ref_weights):
    """_reference() for the layout and activation of ``moe``."""
    return _reference(
        x,
        route_weights,
        ids,
        ref_weights,
        moe.rank,
        moe.world_size,
        moe.model_dim,
        moe.inter_dim,
        moe.experts,
        moe.swiglu_limit,
        moe.act,
        moe.situ_beta,
        moe.situ_linear_beta,
    )


def _time_graph(fn, device, iters):
    _barrier()
    fn()
    _barrier()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        fn()
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    local_ms = start.elapsed_time(end) / iters
    mean_ms = _reduce_float(local_ms, device, dist.ReduceOp.SUM) / dist.get_world_size()
    max_ms = _reduce_float(local_ms, device, dist.ReduceOp.MAX)
    return mean_ms, max_ms


def _run_size(moe, x, weights, ids, ref_weights, args, rank, world, device):
    tokens = x.shape[0]
    main_output = None
    if args.compare_main:
        if not args.expect_large_m:
            raise ValueError("--compare-main requires --expect-large-m")
        moe.set_large_m_enabled(False)
        try:
            main_output = _fwd(moe, x, weights, ids)[:tokens].clone()
        finally:
            moe.set_large_m_enabled(True)
        _barrier()
    output = _fwd(moe, x, weights, ids)[:tokens]
    if args.expect_large_m:
        runtime = moe._large_m
        if runtime is None:
            raise AssertionError("large-M path was requested but did not activate")
        if runtime._s1_op.indexed_payload:
            raise AssertionError("large-M path must use a non-indexed payload")
        config = runtime._active_config
        if (
            config.stage1.num_waves != 8
            or config.stage2.block_m != 64
            or config.stage2.block_n * (config.stage1.num_waves // 4) != 512
        ):
            raise AssertionError("large-M path did not select its NW8/BN512 emitter")
        if rank == 0:
            print(
                "[LARGE-M] active=PASS payload=non-indexed "
                "emitter=NW8/BN512 bf16_launches=2",
                flush=True,
            )
    if main_output is not None:
        error_sq = torch.sum((output.float() - main_output.float()) ** 2)
        reference_sq = torch.sum(main_output.float() ** 2)
        dist.all_reduce(error_sq)
        dist.all_reduce(reference_sq)
        main_rel_l2 = float(torch.sqrt(error_sq / reference_sq))
        if main_rel_l2 >= args.rtol:
            raise AssertionError(
                f"large-M vs main relL2={main_rel_l2:.6f} exceeds {args.rtol}"
            )
        if rank == 0:
            print(
                f"[LARGE-M] main_relL2={main_rel_l2:.6f} PASS",
                flush=True,
            )
    active_config = getattr(moe, "_active_config", None)
    if active_config is not None and active_config.stage2.aligned_pair:
        aligned_output = output.clone()
        full_output = _fwd(moe, x, weights, ids, slice_output=False)
        expected_shape = (moe.mtpr, moe.model_dim)
        if tuple(full_output.shape) != expected_shape:
            raise AssertionError(
                f"slice_output=False returned {tuple(full_output.shape)}, "
                f"expected {expected_shape}"
            )
        if not torch.equal(full_output[:tokens], aligned_output):
            raise AssertionError("slice_output changed aligned-pair valid rows")
        output = full_output[:tokens]
    _barrier()
    eager_output = output.clone()
    _check_explicit_stream(moe, x, weights, ids, eager_output, device)
    rel_l2 = -1.0
    # Every rank must make the same decision before entering the reference
    # collectives; the reference itself pads variable token counts for gather.
    token_max = torch.tensor(tokens, dtype=torch.int32, device=device)
    dist.all_reduce(token_max, op=dist.ReduceOp.MAX)
    if 0 < int(token_max.item()) <= args.accuracy_max_bs:
        reference = _moe_reference(moe, x, weights, ids, ref_weights)
        rel_l2 = _rel_l2(output, reference)
        if rel_l2 >= args.rtol:
            raise AssertionError(f"bs={tokens} relL2={rel_l2:.6f} exceeds {args.rtol}")
    if args.expect_large_m:
        _check_large_m_edges(
            moe,
            x,
            weights,
            ids,
            eager_output,
            ref_weights,
            args,
            rank,
            world,
            device,
            0 < int(token_max.item()) <= args.accuracy_max_bs,
        )
    if args.smoke_only:
        if args.layer_stress_cycles:
            _run_layer_slot_stress(
                moe,
                x,
                weights,
                ids,
                args.large_m_state_slots,
                args.layer_stress_cycles,
                rank,
            )
        if args.route_switch_rounds:
            _run_skewed_route_switch(
                moe, x, weights, ids, args.route_switch_rounds, args.seed, rank, world
            )
        return

    x_q, scale = moe.quantize(x)
    state = {}

    def stage1():
        moe._run_fused_stage1(x_q, weights, scale, ids)

    def stage2():
        state["output"] = moe._run_stage2(tokens, None, True, moe._active_config)

    def end_to_end():
        state["output"] = _fwd(moe, x, weights, ids)

    stage1_ms = _time_graph(stage1, device, args.iters)
    stage1()
    _barrier()
    stage2_ms = _time_graph(stage2, device, args.iters)
    e2e_ms = _time_graph(end_to_end, device, args.iters)
    graph_output = state["output"][:tokens]
    graph_replay_exact = torch.tensor(
        int(torch.equal(graph_output, eager_output)), dtype=torch.int32, device=device
    )
    dist.all_reduce(graph_replay_exact, op=dist.ReduceOp.MIN)
    if not int(graph_replay_exact.item()):
        raise AssertionError(f"bs={tokens} CUDA Graph replay changed the output")
    sbm = int(moe._s1_active_tile_m)
    gemm2_bm = int(moe._g2_active_block_m)
    p2p_quant = moe._active_config.p2p_quant
    if rank == 0:
        print(
            f"[MEGA-V2] bs={tokens} relL2={rel_l2:.6f} "
            f"path={'fixed' if moe._s1_fixed_slot else 'compact'} "
            f"graph_replay=PASS "
            f"p2p_quant={p2p_quant} SBM={sbm} G2_BM={gemm2_bm} "
            f"stage1={stage1_ms[0]:.4f}/{stage1_ms[1]:.4f}ms "
            f"stage2={stage2_ms[0]:.4f}/{stage2_ms[1]:.4f}ms "
            f"e2e={e2e_ms[0]:.4f}/{e2e_ms[1]:.4f}ms mean/max",
            flush=True,
        )
    if args.pad_tail_rows:
        _check_pad_tail(moe, x, weights, ids, ref_weights, args, rank, world, device)


def _binding_state(moe):
    """Every bound weight view, dispatch table and layer slot of the operator and its runtimes."""
    state = [moe._s1_w1, moe._s1_w1_scale, moe.w2, moe.w2_scale]
    if moe._large_m is not None:
        large = moe._large_m
        state += [large._s1_w1, large._s1_w1_scale, large.w2, large.w2_scale]
        state.append(large._active_layer_slot)
    for runtime in moe._small_m_runtimes:
        state += [runtime._s1_w1, runtime._s1_w1_scale, runtime.w2, runtime.w2_scale]
        state.append(runtime._s1_disp)
    return state


def _same_binding(state, expected):
    return len(state) == len(expected) and all(
        a is b if isinstance(b, torch.Tensor) else a == b
        for a, b in zip(state, expected)
    )


def _check_explicit_stream(moe, x, weights, ids, eager_output, device):
    """Every launch of a call runs on the caller's stream, whatever its type."""
    import flydsl.expr as fx

    tokens = x.shape[0]
    side = torch.cuda.Stream()
    for handle in (side, fx.Stream(side.cuda_stream), side.cuda_stream):
        # A zero-input call leaves other data in the quant and combine buffers.
        _fwd(moe, torch.zeros_like(x), weights, ids)
        # Hold the caller's stream back: a launch left on the current stream
        # would run first and read those buffers before this call fills them.
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            torch.cuda._sleep(50_000_000)
        output = _fwd(moe, x, weights, ids, stream=handle)[:tokens]
        torch.cuda.synchronize()
        if not _reduce_float(
            torch.equal(output, eager_output), device, dist.ReduceOp.MIN
        ):
            raise AssertionError(
                f"bs={tokens} stream={type(handle).__name__} changed the output"
            )
        _barrier()


def _check_large_m_edges(
    moe,
    x,
    weights,
    ids,
    eager_output,
    ref_weights,
    args,
    rank,
    world,
    device,
    check_mask,
):
    """Large-input path: explicit stream, rank-local mask and the SiLU-only gate."""
    import flydsl.expr as fx

    from aiter.ops.flydsl.kernels.mega_moe.mega_moe_large_m import _torch_stream

    tokens = x.shape[0]
    # A zero-input call first, so a quant buffer reused on the current stream holds other data.
    _fwd(moe, torch.zeros_like(x), weights, ids)
    # Only the side stream waits for the inputs; the current stream is held back,
    # so work left on it would run after the kernel.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    torch.cuda._sleep(50_000_000)
    output = _fwd(moe, x, weights, ids, stream=side)[:tokens]
    torch.cuda.synchronize()
    stream_exact = torch.equal(output, eager_output)
    if not _reduce_float(stream_exact, device, dist.ReduceOp.MIN):
        raise AssertionError(f"bs={tokens} explicit stream changed the output")
    _barrier()

    if check_mask:
        # Rank 0 alone masks a third of its slots; the path choice must not depend on it.
        masked_ids = ids.clone()
        mask = rank == 0
        if mask:
            gen = torch.Generator(device=device).manual_seed(args.seed + 17)
            drop = torch.rand(ids.shape, generator=gen, device=device) < 1 / 3
            masked_ids[drop] = -1
        output = _fwd(moe, x, weights, masked_ids, mask_invalid_slots=mask)[
            :tokens
        ].clone()
        _barrier()
        reference = _moe_reference(moe, x, weights, masked_ids, ref_weights)
        rel_l2 = _rel_l2(output, reference)
        if rel_l2 >= args.rtol:
            raise AssertionError(f"bs={tokens} rank-local mask relL2={rel_l2:.6f}")

    # Launches keep the bound pointers (a captured graph replays them): a copy made for a
    # strided weight would be freed after the call, so bind_weights refuses one, as it refuses
    # weights of another layout or an out-of-range slot. A refusal changes no binding anywhere.
    bound = _binding_state(moe)
    w1, w1_scale, w2, w2_scale = moe._s1_w1, moe._s1_w1_scale, moe.w2, moe.w2_scale
    large = moe._large_m
    refusals = [
        (
            "strided",
            lambda: moe.bind_weights(w1, w1_scale, w2.transpose(-1, -2), w2_scale),
        ),
        ("layout", lambda: moe.bind_weights(w1, w1_scale, w2.flatten(), w2_scale)),
        ("slot", lambda: moe.bind_layer(1 << 20, w1, w1_scale, w2.clone(), w2_scale)),
        (
            "runtime_slot",
            lambda: large.bind_layer(1 << 20, w1, w1_scale, w2.clone(), w2_scale),
        ),
    ]
    # Strided weights for another slot need two slots (--large-m-state-slots); one slot checks the range only.
    other_slot = (large._active_layer_slot + 1) % large.state_slots
    if other_slot != large._active_layer_slot:
        refusals.append(
            (
                "runtime_strided",
                lambda: large.bind_layer(
                    other_slot, w1, w1_scale, w2.transpose(-1, -2), w2_scale
                ),
            )
        )
    for label, refused in refusals:
        try:
            refused()
        except ValueError:
            pass
        else:
            raise AssertionError(f"a {label} binding was accepted")
        if not _same_binding(_binding_state(moe), bound):
            raise AssertionError(f"a refused {label} binding changed the bound state")
    slot_refusal = "PASS" if len(refusals) == 5 else "RANGE-ONLY"
    # A null stream handle is the default stream, not a new one the persistent kernel never waits on.
    if any(
        _torch_stream(s) != torch.cuda.default_stream() for s in (0, fx.Stream(None))
    ):
        raise AssertionError("a null stream handle did not map to the default stream")
    null_output = _fwd(moe, x, weights, ids, stream=fx.Stream(None))[:tokens]
    torch.cuda.synchronize()
    if not _reduce_float(
        torch.equal(null_output, eager_output), device, dist.ReduceOp.MIN
    ):
        raise AssertionError(f"bs={tokens} the null stream changed the output")
    # Stage1 reads scales as E8M0 bytes: another 1-byte dtype is refused before any launch.
    x_q, x_scale = moe.quantize(x)
    for label, stage1 in (
        ("main", lambda s: moe._run_fused_stage1(x_q, weights, s, ids)),
        ("large", lambda s: moe._large_m.forward_prequant(x_q, s, weights, ids)),
    ):
        try:
            stage1(x_scale.view(torch.float8_e4m3fn))
        except ValueError:
            pass
        else:
            raise AssertionError(f"{label} Stage1 accepted float8_e4m3fn scales")

    # The instance's own layout (EP8 DeepSeek-V4-Pro or EP4 DeepSeek-V4.1-Flash).
    layout = {
        "arch": "gfx950",
        "world_size": moe.world_size,
        "model_dim": moe.model_dim,
        "inter_dim": moe.inter_dim,
        "experts": moe.experts,
        "topk": moe.topk,
        "quant": "a8w4",
        "swiglu_limit": moe.swiglu_limit,
    }
    if not fused_layout_supported(**layout) or fused_layout_supported(
        act="situv2", **layout
    ):
        raise AssertionError("the large/small-input paths must be gated to SiLU")
    # The small-input kernel shards its work over 8 XCDs: a 4-XCD (128-CU) part is not eligible.
    if not small_m_cu_supported(256) or small_m_cu_supported(128):
        raise AssertionError("the small-input path must be gated to 8-XCD parts")
    if rank == 0:
        print(
            f"[LARGE-M] bs={tokens} explicit_stream=PASS "
            f"rank_local_mask={'PASS' if check_mask else 'SKIP'} silu_gate=PASS "
            f"strided_weights=PASS xcd_gate=PASS slot_refusal={slot_refusal} "
            "scale_dtype=PASS null_stream=PASS",
            flush=True,
        )


def _check_pad_tail(moe, x, weights, ids, ref_weights, args, rank, world, device):
    """Whole -1 rows with mask_invalid_slots=True: exact zeros, real rows intact.

    Runs after the unmasked passes above, so the combine input still holds
    their values; a masked slot that read it would show up as nonzero.
    """
    tokens = x.shape[0]
    real = max(tokens - args.pad_tail_rows, 0)
    masked_ids = ids.clone()
    masked_ids[real:] = -1

    def masked():
        return _fwd(moe, x, weights, masked_ids, mask_invalid_slots=True)[:tokens]

    eager_output = masked().clone()
    _barrier()
    pad_zero = bool((eager_output[real:] == 0).all())
    if not _reduce_float(pad_zero, device, dist.ReduceOp.MIN):
        raise AssertionError(f"bs={tokens} masked pad rows are not zero")
    reference = _moe_reference(moe, x[:real], weights[:real], ids[:real], ref_weights)
    # -1: every row of every rank is padding, so there is nothing to compare.
    rel_l2 = _rel_l2(eager_output[:real], reference)
    if rel_l2 >= args.rtol:
        raise AssertionError(
            f"bs={tokens} pad_tail={args.pad_tail_rows} relL2={rel_l2:.6f} "
            f"exceeds {args.rtol}"
        )
    state = {}

    def capture():
        state["output"] = masked()

    _time_graph(capture, device, 1)
    graph_exact = torch.equal(state["output"], eager_output)
    if not _reduce_float(graph_exact, device, dist.ReduceOp.MIN):
        raise AssertionError(f"bs={tokens} CUDA Graph replay changed masked output")
    if rank == 0:
        print(
            f"[MEGA-V2] bs={tokens} pad_tail={args.pad_tail_rows} "
            f"relL2={rel_l2:.6f} pad_rows=ZERO graph_replay=PASS",
            flush=True,
        )


def _run_burst(moe, x, weights, ids, depth, rank):
    _fwd(moe, x, weights, ids)
    _barrier()
    for _ in range(depth):
        _fwd(moe, x, weights, ids)
    torch.cuda.synchronize()
    if rank == 0:
        print(f"[BURST] completed={depth}/{depth}", flush=True)


def _run_layer_slot_stress(moe, x, weights, ids, slots, cycles, rank):
    """Alternate routes and weight pointers across model-layer slots."""
    if slots < 61 or cycles <= 8:
        raise ValueError("layer-slot stress requires at least 61 slots and 9 cycles")
    if moe._large_m is None:
        raise AssertionError("layer-slot stress requires the large-M path")
    if moe._large_m.state_slots < slots:
        raise AssertionError("large-M state arena is smaller than the stress")

    w1_a, sw1_a = moe._s1_w1, moe._s1_w1_scale
    w2_a, sw2_a = moe.w2, moe.w2_scale
    # Layer B has its own pointers and doubles W2 (E8M0 exponent + 1), so a call that ran
    # on the other layer's weights changes the output.
    w1_b, sw1_b = w1_a.clone(), sw1_a.clone()
    w2_b = w2_a.clone()
    exponent = sw2_a.view(torch.uint8)
    sw2_b = torch.where(exponent < 254, exponent + 1, exponent).view(sw2_a.dtype)
    route_row = torch.arange(ids.shape[0], dtype=torch.int32, device=ids.device)[
        :, None
    ]
    route_slot = torch.arange(moe.topk, dtype=torch.int32, device=ids.device)[None, :]
    ids_b = (
        (route_row * moe.topk + route_slot + rank * (moe.experts // moe.world_size))
        .remainder(moe.experts)
        .contiguous()
    )
    weights_b = weights.flip(0).contiguous()
    tokens = x.shape[0]
    # A few slots per cycle, covering both layers, are checked against the main path on the same
    # binding: the large-M path matches it bit for bit.
    checked_slots = {0, 1, slots - 1}
    checked = 0

    for cycle in range(cycles):
        for slot in range(slots):
            if (cycle + slot) & 1:
                bound = w1_b, sw1_b, w2_b, sw2_b
                route_weights, route_ids = weights_b, ids_b
            else:
                bound = w1_a, sw1_a, w2_a, sw2_a
                route_weights = weights
                route_ids = (
                    (ids + cycle * 37 + slot * 11).remainder(moe.experts).contiguous()
                )
            moe.bind_layer(slot, *bound)
            output = _fwd(moe, x, route_weights, route_ids)[:tokens]
            if slot not in checked_slots:
                continue
            output = output.clone()
            moe.set_large_m_enabled(False)
            try:
                reference = _fwd(moe, x, route_weights, route_ids)[:tokens].clone()
            finally:
                moe.set_large_m_enabled(True)
            mismatched = (output != reference).any(dim=-1).sum().to(torch.int64)
            finite = torch.isfinite(output).all().to(torch.int64)
            dist.all_reduce(mismatched)
            dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if int(mismatched.item()) or not int(finite.item()):
                raise AssertionError(
                    f"layer cycle={cycle} slot={slot}: {int(mismatched.item())} rows "
                    "differ from the main path on the same binding"
                )
            checked += 1
    # The doubled W2 must change the result, or the comparisons above could not see a stale binding.
    moe.bind_layer(0, w1_a, sw1_a, w2_a, sw2_a)
    out_a = _fwd(moe, x, weights, ids)[:tokens].clone()
    moe.bind_layer(0, w1_b, sw1_b, w2_b, sw2_b)
    out_b = _fwd(moe, x, weights, ids)[:tokens].clone()
    differ = (out_a != out_b).any().to(torch.int64)
    dist.all_reduce(differ, op=dist.ReduceOp.MAX)
    if not int(differ.item()):
        raise AssertionError("layer B weights do not change the output")
    moe.bind_layer(0, w1_a, sw1_a, w2_a, sw2_a)
    _barrier()
    if rank == 0:
        print(
            f"[LARGE-M-LAYERS] slots={slots} cycles={cycles} "
            f"calls={slots * cycles} main_checks={checked} route_mutation=PASS "
            "weight_rebind=PASS",
            flush=True,
        )


def _skewed_route(tokens, experts, topk, hot_experts, round_idx, seed, rank, device):
    """Send three of every four rows to each hot expert; weights ignore the bias."""
    generator = torch.Generator(device=device).manual_seed(
        seed + 7919 * (round_idx + 1) + rank
    )
    scores = torch.randn(
        (tokens, experts), dtype=torch.float32, device=device, generator=generator
    )
    rows = torch.arange(tokens, device=device)
    bias = torch.zeros_like(scores)
    hot_rows = ((rows + round_idx) % 4 != 0)[:, None]
    bias[:, hot_experts] = hot_rows * 1.0e4
    ids = torch.topk(scores + bias, topk, dim=-1).indices
    weights = scores.gather(1, ids).softmax(dim=-1)
    return weights.contiguous(), ids.to(torch.int32).contiguous()


def _pair_order_hazard_producers(moe, ids, world):
    """Count producers that copy payload for a destination but skip grouping.

    Mirrors mega_moe_large_m._configure_payload_geometry: a destination
    whose hottest local expert exceeds four payload chunks gets more than four
    producers, which moves some payload producers past the grouping set.
    """
    stage1 = moe._large_m._active_config.stage1
    if stage1.num_dispatch_cu <= 32:
        return 0
    local_experts = moe.experts // world
    counts = torch.bincount(ids[ids >= 0].long(), minlength=moe.experts)
    max_blocks = stage1.num_dispatch_cu // world
    producers = [
        min(max(-(-int(count) // stage1.payload_chunk_rows), 4), max_blocks)
        for count in counts.view(world, local_experts).amax(dim=1).tolist()
    ]
    grouped = sum(producers)
    return sum(
        1
        for slot in range(grouped, stage1.num_dispatch_cu)
        if slot // world < producers[slot % world]
    )


def _run_skewed_route_switch(moe, x, weights, ids, rounds, seed, rank, world):
    """Change a skewed route between large-M launches and compare to main.

    Uniform routes keep four producers per destination, so they never reach
    the producers that skip adaptive grouping; a stale pair order there only
    shows up when the route differs from the previous launch.
    """
    if moe._large_m is None:
        raise AssertionError("skewed route switch requires the large-M path")
    tokens = x.shape[0]
    local_experts = moe.experts // world
    _fwd(moe, x, weights, ids)
    _fwd(moe, x, weights, ids)
    _barrier()
    worst = 0.0
    failures = []
    for round_idx in range(rounds):
        # Like a captured batch: one source rank saturates five
        # destinations while its peers stay near uniform.
        hot_experts = [
            ((rank + round_idx + 3 * hot) % world) * local_experts
            + (round_idx * 7 + hot) % local_experts
            for hot in range(min(5, moe.topk))
            if rank == round_idx % world
        ]
        route_weights, route_ids = _skewed_route(
            tokens, moe.experts, moe.topk, hot_experts, round_idx, seed, rank, x.device
        )
        hazard = torch.tensor(
            _pair_order_hazard_producers(moe, route_ids, world),
            dtype=torch.int32,
            device=x.device,
        )
        dist.all_reduce(hazard)
        if not int(hazard.item()):
            raise AssertionError("skewed route did not exceed four payload producers")
        moe.set_large_m_enabled(False)
        try:
            main_output = _fwd(moe, x, route_weights, route_ids)[:tokens].clone()
        finally:
            moe.set_large_m_enabled(True)
        _barrier()
        output = _fwd(moe, x, route_weights, route_ids)[:tokens].float()
        mismatched = (output != main_output.float()).any(dim=-1).sum().to(torch.int64)
        error_sq = torch.sum((output - main_output.float()) ** 2)
        reference_sq = torch.sum(main_output.float() ** 2)
        for value in (mismatched, error_sq, reference_sq):
            dist.all_reduce(value)
        rel_l2 = float(torch.sqrt(error_sq / reference_sq))
        worst = max(worst, rel_l2)
        # Same kernels and inputs make the large-M path match main bit-for-bit; a stale pair
        # order corrupts whole rows, sometimes only a few per launch.
        if int(mismatched.item()):
            failures.append(
                f"round={round_idx} hazard_producers={int(hazard.item())} "
                f"mismatched_rows={int(mismatched.item())} main_relL2={rel_l2:.6f}"
            )
    if failures:
        raise AssertionError(
            f"skewed route switch failed {len(failures)}/{rounds} rounds: "
            + "; ".join(failures)
        )
    if rank == 0:
        print(
            f"[LARGE-M-ROUTE-SWITCH] rounds={rounds} "
            f"mismatched_rows=0 max_main_relL2={worst:.6f} PASS",
            flush=True,
        )


def _install_config_policy(moe, config_tokens, unify_fields):
    if not config_tokens:
        return
    reference = moe._select_config(config_tokens)
    original_select = moe._select_config
    fields = [field for field in unify_fields.split(",") if field]

    if not fields:

        def select_config(_tokens):
            moe._active_config = reference
            return reference

    else:

        def select_config(tokens):
            local = original_select(tokens)
            stage1_updates = {}
            stage2_updates = {}
            p2p_quant = local.p2p_quant
            for field in fields:
                if field.startswith("stage1."):
                    name = field.removeprefix("stage1.")
                    stage1_updates[name] = getattr(reference.stage1, name)
                elif field.startswith("stage2."):
                    name = field.removeprefix("stage2.")
                    stage2_updates[name] = getattr(reference.stage2, name)
                elif field == "p2p_quant":
                    p2p_quant = reference.p2p_quant
                else:
                    raise ValueError(f"invalid config field {field!r}")
            config = replace(
                local,
                stage1=replace(local.stage1, **stage1_updates),
                stage2=replace(local.stage2, **stage2_updates),
                p2p_quant=p2p_quant,
            )
            moe._active_config = config
            return config

    moe._select_config = select_config


# ---- routes suite: the instance production builds ----------------------------


def _routed(moe, x, weights, ids, max_tokens, expect=None):
    """forward() as production calls it: max_tokens, the call's rows on its largest
    EP rank, picks the route; ``expect`` checks which one."""
    if expect is not None:
        route = moe._route(x.shape[0], max_tokens)[0]
        if route != expect:
            raise AssertionError(
                f"max_tokens={max_tokens} took the {route} route, not {expect}"
            )
    return moe.forward(x, weights, ids, max_tokens=max_tokens)[: x.shape[0]]


def _staged_forward(moe, x, weights, ids):
    """The staged route of the instance (the accuracy and timing reference)."""
    return moe._forward_route((ROUTE_STAGED, moe.mtpr), x, weights, ids)[: x.shape[0]]


def _large_forward(moe, x, weights, ids):
    """The large-input route of the instance."""
    return moe._forward_route((ROUTE_LARGE_M, moe.mtpr), x, weights, ids)[: x.shape[0]]


def _time_replays(fn, calls, trials, device):
    """Median time of one call (us) over graph replays of ``calls`` calls, max over ranks."""
    for _ in range(3):
        fn()
    _barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        for _ in range(calls):
            fn()
    for _ in range(3):
        graph.replay()
    _barrier()
    times = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        _barrier()
        start.record()
        graph.replay()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) * 1000.0 / calls)
    del graph
    times.sort()
    return _reduce_float(times[len(times) // 2], device, dist.ReduceOp.MAX)


def _rank_case(tokens_per_rank, rank, x, w, ids, experts, pad_rows=0):
    """Local inputs for one case; 0 tokens means an idle rank that sends one sentinel row."""
    n = tokens_per_rank[rank]
    if n == 0:
        xi = torch.zeros((1, x.shape[1]), dtype=x.dtype, device=x.device)
        wi = torch.zeros((1, w.shape[1]), dtype=w.dtype, device=w.device)
        ii = torch.full((1, ids.shape[1]), experts, dtype=ids.dtype, device=ids.device)
        return xi, wi, ii, 0
    xi, wi, ii = x[:n].clone(), w[:n].clone(), ids[:n].clone()
    if pad_rows:
        # _fill_padded_rows: rows past num_token_non_padded route to `experts`.
        ii[n - pad_rows :] = experts
    return xi, wi, ii, n - pad_rows


def _check_epoch_wrap_and_zero_rows(moe, bind, case, large_case, device):
    """All ranks with zero rows, then launches across the 32-bit epoch wrap.

    The small-input generation is preset just below 2**31 and 2**32 and the large-input
    combine generation and launch epoch just below 2**31; each following launch must give
    the result it gave before. Returns the number of mismatched rows over all ranks.
    """
    bind(0, 0)
    mism = 0
    empty_x = torch.empty((0, case[0].shape[1]), dtype=case[0].dtype, device=device)
    empty_w = torch.empty((0, case[1].shape[1]), dtype=case[1].dtype, device=device)
    empty_i = torch.empty((0, case[2].shape[1]), dtype=case[2].dtype, device=device)
    if _routed(moe, empty_x, empty_w, empty_i, 0, ROUTE_SMALL_M).shape[0] != 0:
        mism += 1
    if (
        moe._large_m is not None
        and _large_forward(moe, empty_x, empty_w, empty_i).shape[0] != 0
    ):
        mism += 1

    xi, wi, ii, expect, real, mt = case
    large_expect = None
    if large_case is not None:
        large_expect = _large_forward(moe, *large_case[:3])[: large_case[4]].clone()
    for base in (2**31 - 2, 2**32 - 2):
        _barrier()
        for runtime in moe._small_m_runtimes:
            # generation = entry_count // (2 * CUs); the next launches cross the wrap.
            runtime._s1_dispatch_workspace["entry_count"].fill_(
                base * 2 * runtime._s1_num_cu
            )
        if moe._large_m is not None:
            large = moe._large_m
            large._fused_comb_ctr_arena[:, 1].fill_(2**31 - 3)
            large._s1_epoch_expected_arena.fill_(2**31 - 2 * moe.world_size)
        torch.cuda.synchronize()
        _barrier()
        for _ in range(4):
            out = _routed(moe, xi, wi, ii, mt, ROUTE_SMALL_M)[:real]
            mism += int((out != expect[:real]).any(dim=-1).sum().item())
            if large_expect is not None:
                out = _large_forward(moe, *large_case[:3])[: large_case[4]]
                mism += int((out != large_expect).any(dim=-1).sum().item())
    total = torch.tensor(mism, device=device)
    dist.all_reduce(total)
    return int(total.item())


def _routes_instance(args, rank, world, device):
    """The instance with every route its layout supports, and two layers of weights."""
    net = NETWORKS[args.network]
    packs = [
        _quantize_weights(
            net["model_dim"],
            net["inter_dim"],
            net["experts"] // world,
            rank,
            args.seed + 17 * i,
            device,
        )
        for i in range(2)
    ]
    moe = MegaMoEV2(
        rank=rank,
        world_size=world,
        quant="a8w4",
        w1=packs[0][0],
        w1_scale=packs[0][1],
        w2=packs[0][2],
        w2_scale=packs[0][3],
        max_tok_per_rank=args.max_tok_per_rank,
        large_m_state_slots=args.large_m_state_slots,
        small_m_mtprs=tuple(args.small_m_mtprs),
        **net,
    )
    if moe.small_m_mtpr != max(args.small_m_mtprs):
        raise AssertionError(
            f"small-input route not built (small_m_mtpr={moe.small_m_mtpr}); it needs a fused "
            "a8w4 layout (EP8 DeepSeek-V4-Pro on 8 ranks or EP4 DeepSeek-V4.1-Flash on 4 ranks) "
            "on an 8-XCD gfx950"
        )
    x, w, ids = _make_inputs(
        moe.small_m_mtpr,
        net["model_dim"],
        net["experts"],
        net["topk"],
        rank,
        args.seed,
        device,
        force_fanout_boundary=False,
        inject_invalid_route=False,
        force_padding_boundary=False,
    )
    binds = [p[:4] for p in packs]

    def bind(slot, which):
        moe.bind_layer(slot, *binds[which])

    return SimpleNamespace(
        moe=moe,
        net=net,
        rank=rank,
        world=world,
        device=device,
        x=x,
        w=w,
        ids=ids,
        refw=[p[4:] for p in packs],
        bind=bind,
        large_m=moe._large_m is not None,
        failures=[],
    )


def _routes_accuracy(ctx, args):
    """Small-input route against the fp32 reference and the staged route, both layers.

    Returns {(case, layer): (x, w, ids, output, real rows, max_tokens)}.
    """
    moe, rank, world, device = ctx.moe, ctx.rank, ctx.world, ctx.device
    experts = ctx.net["experts"]
    cases = [
        ("uniform1", [1] * world, 0),
        ("uniform8", [8] * world, 0),
        ("uniform32", [32] * world, 0),
        ("uniform51", [51] * world, 0),
        ("uniform64", [64] * world, 0),
        ("uneven", [64, 1, 33, 7, 50, 64, 2, 17][:world], 0),
        ("idle_ranks", [0, 64, 0, 13, 64, 0, 40, 0][:world], 0),
        ("padded", [64, 48, 64, 32, 56, 64, 24, 64][:world], 5),
        ("all_idle", [0] * world, 0),
        ("uniform70", [70] * world, 0),
        ("uniform126", [126] * world, 0),
        ("uniform168", [168] * world, 0),
        ("uniform224", [224] * world, 0),
        ("uniform256", [256] * world, 0),
        ("dec_uneven", [168, 126, 126, 84, 84, 70, 56, 1][:world], 0),
        ("dec_idle", [224, 0, 7, 0, 140, 0, 1, 0][:world], 0),
        ("dec_padded", [224, 224, 224, 224, 224, 224, 224, 224][:world], 37),
        ("uniform280", [280] * world, 0),
        ("uniform364", [364] * world, 0),
        ("uniform448", [448] * world, 0),
        ("uniform512", [512] * world, 0),
        ("c384_uneven", [336, 182, 280, 119, 224, 308, 154, 210][:world], 0),
        ("c384_padded", [448] * world, 160),
        ("c384_idle", [448, 0, 7, 0, 364, 1, 0, 280][:world], 0),
    ]
    cases = [c for c in cases if max(c[1]) <= moe.small_m_mtpr]
    outputs = {}
    for name, tpr, pad in cases:
        for which in range(2):
            ctx.bind(which, which)
            xi, wi, ii, real = _rank_case(
                tpr, rank, ctx.x, ctx.w, ctx.ids, experts, pad
            )
            max_tokens = max(max(tpr), 1)
            out_g = _routed(moe, xi, wi, ii, max_tokens, ROUTE_SMALL_M).clone()
            _barrier()
            out_m = _staged_forward(moe, xi, wi, ii).clone()
            _barrier()
            rows = slice(0, real)
            if max(tpr) == 0:
                rel_staged, rel_ref = 0.0, 0.0
            else:
                ref = _moe_reference(moe, xi[rows], wi[rows], ii[rows], ctx.refw[which])
                rel_ref = _rel_l2(out_g[rows], ref)
                rel_staged = _rel_l2(out_g[rows], out_m[rows])
            # Padded rows are discarded by the caller; they only must not be
            # NaN/Inf. Whether they are exactly zero is reported, not gated.
            pad_ok = _all_ranks(torch.isfinite(out_g).all(), device)
            pad_zero = _all_ranks((out_g[real:] == 0).all(), device) if pad else True
            outputs[(name, which)] = (xi, wi, ii, out_g, real, max_tokens)
            ok = rel_ref < args.rtol and rel_staged < args.rtol and pad_ok
            if rank == 0:
                print(
                    f"[ACC] {name:10s} w{which} tpr={tpr} pad={pad} relL2_vs_ref={rel_ref:.6f} "
                    f"relL2_vs_staged={rel_staged:.6f} finite={pad_ok} pad_rows_zero={pad_zero} "
                    f"{'PASS' if ok else 'FAIL'}",
                    flush=True,
                )
            if not ok:
                ctx.failures.append(f"acc {name} w{which}")
    return outputs


def _routes_edges(ctx, args):
    """Truly empty ranks and rank-local -1 slots, then skewed routes."""
    moe, rank, world, device = ctx.moe, ctx.rank, ctx.world, ctx.device
    x, w, ids = ctx.x, ctx.w, ctx.ids
    ctx.bind(0, 0)
    tpr = [0, 64, 0, 13, 64, 0, 40, 0][:world]
    n = tpr[rank]
    xi, wi, ii = x[:n].clone(), w[:n].clone(), ids[:n].clone()
    if rank % 2:
        # The combine input still holds the rows of the calls above: a -1 slot must not read them.
        gen = torch.Generator(device=device).manual_seed(args.seed + rank)
        ii[torch.rand(ii.shape, generator=gen, device=device) < 1 / 3] = -1
    # forward_small_m: the wrapper existing callers use (forward with mask_invalid_slots=True).
    out = moe.forward_small_m(xi, wi, ii, max_tokens=max(tpr))[:n].clone()
    _barrier()
    rel_ref = _rel_l2(out, _moe_reference(moe, xi, wi, ii, ctx.refw[0]))
    ok = rel_ref < args.rtol and _all_ranks(torch.isfinite(out).all(), device)
    if rank == 0:
        print(
            f"[EDGE] empty_ranks+masked_slots tpr={tpr} relL2_vs_ref={rel_ref:.6f} "
            f"{'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    if not ok:
        ctx.failures.append("empty ranks / masked slots")
    # Skewed routes: the rank holding the hot experts sends FP8 rows and reads cached weights, the others
    # keep BF16 rows, so the combine reads both formats.
    hot = torch.arange(moe.topk, dtype=ids.dtype, device=device)
    for name, n, share in (("hot60", 128, 3), ("one_rank", 64, 5), ("hot60", 448, 3)):
        if n > moe.small_m_mtpr:
            continue
        xi, wi, ii = x[:n].clone(), w[:n].clone(), ids[:n].clone()
        ii[(torch.arange(n, device=device) % 5) < share] = hot
        out = _routed(moe, xi, wi, ii, n, ROUTE_SMALL_M).clone()
        _barrier()
        rel_ref = _rel_l2(out, _moe_reference(moe, xi, wi, ii, ctx.refw[0]))
        ok = rel_ref < args.rtol and _all_ranks(torch.isfinite(out).all(), device)
        if rank == 0:
            print(
                f"[EDGE] {name} tpr={n} relL2_vs_ref={rel_ref:.6f} {'PASS' if ok else 'FAIL'}",
                flush=True,
            )
        if not ok:
            ctx.failures.append(f"skewed {name} {n}")


def _routes_alternation(ctx, args, outputs):
    """Layer rebinding over the state slots, staged calls between small-input calls (eager).

    Every small-input call must repeat its first output bit for bit.
    """
    moe, device = ctx.moe, ctx.device
    gen = torch.Generator().manual_seed(args.seed)
    mism = 0
    staged_calls = 0
    keys = [k for k in outputs if k[0] != "all_idle"]
    for step in range(args.alt_steps):
        slot = step % args.large_m_state_slots
        which = (step + slot // 7) & 1
        ctx.bind(slot, which)
        if torch.rand((), generator=gen).item() < 0.3:
            # A larger staged call on the same bound layer between small-input calls.
            n = 64 + int(torch.randint(0, 200, (), generator=gen).item())
            xm = torch.randn(
                (n, ctx.net["model_dim"]), dtype=torch.bfloat16, device=device
            )
            wm = ctx.w[:1].expand(n, -1).contiguous()
            im = ctx.ids[:1].expand(n, -1).contiguous()
            _staged_forward(moe, xm, wm, im)
            staged_calls += 1
        name = keys[int(torch.randint(0, len(keys), (), generator=gen).item())][0]
        xi, wi, ii, expect, real, mt = outputs[(name, which)]
        out = _routed(moe, xi, wi, ii, mt)[:real]
        # Only real rows: sentinel/idle rows are never reduced, so they hold
        # whatever the combine buffer last had (the caller discards them).
        mism += int((out != expect[:real]).any(dim=-1).sum().item())
    mism_t = torch.tensor(mism, device=device)
    dist.all_reduce(mism_t)
    ok = int(mism_t.item()) == 0
    if ctx.rank == 0:
        print(
            f"[ALT] steps={args.alt_steps} slots={args.large_m_state_slots} "
            f"staged_calls={staged_calls} mismatched_rows_vs_first_call={int(mism_t.item())} "
            f"{'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    if not ok:
        ctx.failures.append("alternation")
    _barrier()
    return keys


def _routes_graph(ctx, args):
    """A CUDA graph of --graph-layers small-input calls over alternating layers, replayed
    on new inputs, against the same calls run eagerly."""
    moe, device, net = ctx.moe, ctx.device, ctx.net
    n = args.graph_bs
    sx, sw, si = ctx.x[:n].clone(), ctx.w[:n].clone(), ctx.ids[:n].clone()
    graph_out = []

    def graph_body():
        graph_out.clear()
        for layer in range(args.graph_layers):
            ctx.bind(layer, layer & 1)
            # The output is the operator's combine buffer (as in main), valid
            # until its next call: copy it out before the next call.
            graph_out.append(_routed(moe, sx, sw, si, n).clone())

    # Warm (eager) so dispatch tables exist before capture.
    graph_body()
    _barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
        graph_body()
    mism = 0
    for rep in range(args.graph_reps):
        gx, gw, gi = _make_inputs(
            n,
            net["model_dim"],
            net["experts"],
            net["topk"],
            ctx.rank,
            args.seed + 100 + rep % 3,
            device,
            force_fanout_boundary=False,
            inject_invalid_route=False,
            force_padding_boundary=False,
        )
        sx.copy_(gx)
        sw.copy_(gw)
        si.copy_(gi)
        _barrier()
        graph.replay()
        torch.cuda.synchronize()
        replay = [o.clone() for o in graph_out]
        _barrier()
        for layer in range(args.graph_layers):
            ctx.bind(layer, layer & 1)
            eager = _routed(moe, gx, gw, gi, n)
            mism += int((eager != replay[layer]).any(dim=-1).sum().item())
        _barrier()
    mism_t = torch.tensor(mism, device=device)
    dist.all_reduce(mism_t)
    ok = int(mism_t.item()) == 0
    if ctx.rank == 0:
        print(
            f"[GRAPH] layers={args.graph_layers} reps={args.graph_reps} bs/rank={n} "
            f"route={moe._route(n, n)[0]} replay_vs_eager_mismatched_rows={int(mism_t.item())} "
            f"{'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    if not ok:
        ctx.failures.append("graph")
    del graph
    _barrier()


def _routes_timing(ctx, args):
    """Per --time-bs: the routed call against the staged route, each a graph of
    --time-calls calls; --check-time-acc first checks every route at that size."""
    moe, rank, device, net = ctx.moe, ctx.rank, ctx.device, ctx.net
    for n in args.time_bs:
        if n <= ctx.x.shape[0]:
            tx, tw, ti = (
                ctx.x[:n].contiguous(),
                ctx.w[:n].contiguous(),
                ctx.ids[:n].contiguous(),
            )
        else:
            tx, tw, ti = _make_inputs(
                n,
                net["model_dim"],
                net["experts"],
                net["topk"],
                rank,
                args.seed + 500,
                device,
                force_fanout_boundary=False,
                inject_invalid_route=False,
                force_padding_boundary=False,
            )
        ctx.bind(0, 0)
        route = moe._route(n, n)[0]
        if args.check_time_acc:
            # The pinned staged and large-input routes and the routed call, against the fp32
            # reference at the timed token count.
            checks = [("staged", lambda: _staged_forward(moe, tx, tw, ti))]
            if ctx.large_m:
                checks.append(("large_m", lambda: _large_forward(moe, tx, tw, ti)))
            checks.append((f"routed:{route}", lambda: _routed(moe, tx, tw, ti, n)))
            ref = _moe_reference(moe, tx, tw, ti, ctx.refw[0])
            for label, fn in checks:
                out = fn()[:n].clone()
                _barrier()
                rel_ref = _rel_l2(out, ref)
                ok = rel_ref < args.rtol and _all_ranks(
                    torch.isfinite(out).all(), device
                )
                if rank == 0:
                    print(
                        f"[ACC-T] bs/rank={n} {label} relL2_vs_ref={rel_ref:.6f} "
                        f"{'PASS' if ok else 'FAIL'}",
                        flush=True,
                    )
                if not ok:
                    ctx.failures.append(f"time-acc {label} {n}")
        paths = [
            ("staged", lambda tx=tx, tw=tw, ti=ti: _staged_forward(moe, tx, tw, ti))
        ]
        if route != ROUTE_STAGED:
            paths.append(
                (route, lambda tx=tx, tw=tw, ti=ti, n=n: _routed(moe, tx, tw, ti, n))
            )
        res = {
            label: _time_replays(fn, args.time_calls, args.time_trials, device)
            for label, fn in paths
        }
        if rank == 0:
            staged = res.pop("staged")
            line = f"[TIME] bs/rank={n:5d} staged={staged:8.1f}us"
            for label, us in res.items():
                line += f" {label}={us:8.1f}us speedup={staged / us:.3f}x"
            print(f"{line} (median of {args.time_trials}, max over ranks)", flush=True)


def _run_routes(args, rank, world, device):
    ctx = _routes_instance(args, rank, world, device)
    moe = ctx.moe
    if rank == 0:
        print(
            f"[SETUP] network={args.network} mtpr={moe.mtpr} small_m_mtpr={moe.small_m_mtpr} "
            f"large_m={ctx.large_m} world={world}",
            flush=True,
        )
    if not args.time_only:
        outputs = _routes_accuracy(ctx, args)
        _routes_edges(ctx, args)
        keys = _routes_alternation(ctx, args, outputs)
        # Zero rows on every rank; 32-bit epochs wrapping.
        case = outputs[(keys[0][0], 0)]
        wrap_mism = _check_epoch_wrap_and_zero_rows(
            moe, ctx.bind, case, case if ctx.large_m else None, device
        )
        if rank == 0:
            print(
                f"[WRAP] zero_rows+epoch_wrap mismatched_rows={wrap_mism} "
                f"large_m={ctx.large_m} {'PASS' if wrap_mism == 0 else 'FAIL'}",
                flush=True,
            )
        if wrap_mism:
            ctx.failures.append("epoch wrap")
        _routes_graph(ctx, args)
    _routes_timing(ctx, args)
    _barrier()
    if ctx.failures:
        raise AssertionError("FAILED: " + ", ".join(ctx.failures))
    if rank == 0:
        print("[RESULT] ALL PASS", flush=True)


def _run_sizes(args, rank, world, device):
    batch_sizes, rank_tokens = args.batch_sizes, args.rank_tokens_list
    network = NETWORKS[args.network]
    if network["experts"] % world:
        raise ValueError(
            f"experts={network['experts']} must be divisible by world={world}"
        )
    if args.max_tok_per_rank is not None and args.max_tok_per_rank < max(batch_sizes):
        raise ValueError("--max-tok-per-rank must cover the largest batch size")
    if args.config_tokens and args.max_tok_per_rank is None:
        raise ValueError("--config-tokens requires --max-tok-per-rank")
    if args.config_tokens < 0 or (
        args.max_tok_per_rank is not None and args.config_tokens > args.max_tok_per_rank
    ):
        raise ValueError("--config-tokens must be between zero and max-tok-per-rank")
    local_experts = network["experts"] // world
    packed = _quantize_weights(
        network["model_dim"],
        network["inter_dim"],
        local_experts,
        rank,
        args.seed,
        device,
    )
    w1, w1_scale, w2, w2_scale, w1_q, w1_ref_scale, w2_q, w2_ref_scale = packed
    if rank_tokens and len(rank_tokens) != world:
        raise ValueError("--rank-tokens must contain one value per rank")
    max_bs = max(batch_sizes + rank_tokens)
    x, weights, ids = _make_inputs(
        max_bs,
        network["model_dim"],
        network["experts"],
        network["topk"],
        rank,
        args.seed,
        device,
        force_fanout_boundary=args.force_fanout_boundary,
        inject_invalid_route=args.inject_invalid_route,
        force_padding_boundary=args.force_padding_boundary,
        concentrate_experts=args.concentrate_experts,
        zipf=args.zipf,
    )
    ref_weights = w1_q, w1_ref_scale, w2_q, w2_ref_scale
    # MegaMoE's producer/consumer wire geometry is rank-invariant. A
    # deliberately skewed test therefore selects one configuration that
    # covers the largest participating rank instead of mixing protocols.
    config_tokens = args.config_tokens
    if rank_tokens and not config_tokens:
        config_tokens = max(1, max(rank_tokens))

    shared_moe = None
    for batch_size in batch_sizes:
        local_batch_size = rank_tokens[rank] if rank_tokens else batch_size
        max_tok_per_rank = args.max_tok_per_rank or max(
            16, _next_power_of_two(batch_size)
        )
        if shared_moe is None or args.max_tok_per_rank is None:
            fanout_masks = ()
            if args.force_fanout_boundary:
                pair_mask = (1 << (local_experts - 2)) | (1 << (local_experts - 1))
                fanout_masks = (pair_mask,) * world
            elif args.inject_invalid_route or args.force_padding_boundary:
                pair_mask = 1 | (1 << (local_experts - 1))
                fanout_masks = (
                    (pair_mask,) + (0,) * (world - 1)
                    if args.force_padding_boundary
                    else (pair_mask,) * world
                )
            shared_moe = MegaMoEV2(
                rank=rank,
                world_size=world,
                quant=args.quant,
                w1=w1,
                w1_scale=w1_scale,
                w2=w2,
                w2_scale=w2_scale,
                max_tok_per_rank=max_tok_per_rank,
                fanout_masks=fanout_masks,
                stage2_p2p_quant=args.stage2_p2p_quant,
                large_m_state_slots=args.large_m_state_slots,
                # Pin the plan: the staged route unless --expect-large-m.
                large_m_enabled=args.expect_large_m,
                small_m_mtprs=(),
                **network,
            )
        moe = shared_moe
        _install_config_policy(moe, config_tokens, args.unify_fields)
        if rank_tokens:
            selected = moe._select_config(local_batch_size)
            configs = [None] * world
            dist.all_gather_object(
                configs,
                (
                    rank,
                    local_batch_size,
                    selected.stage1,
                    selected.stage2,
                    selected.p2p_quant,
                ),
            )
            if rank == 0:
                for config in configs:
                    print(f"[CONFIG] {config}", flush=True)
        local_x = x[:local_batch_size].contiguous()
        local_weights = weights[:local_batch_size].contiguous()
        local_ids = ids[:local_batch_size].contiguous()
        if args.burst_depth:
            _run_burst(moe, local_x, local_weights, local_ids, args.burst_depth, rank)
        else:
            _run_size(
                moe,
                local_x,
                local_weights,
                local_ids,
                ref_weights,
                args,
                rank,
                world,
                device,
            )


def _int_list(text):
    return [int(value) for value in text.split(",") if value]


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--suite",
        choices=("size", "routes"),
        default="size",
        help="size: one pinned-route instance per batch size; routes: the production "
        "instance, its route picked by max_tokens",
    )
    parser.add_argument("--network", choices=NETWORKS, default="v4_pro")
    parser.add_argument("--quant", choices=("a4w4", "a8w4"), default="a8w4")
    parser.add_argument(
        "--seed", type=int, help="default 123 (size suite), 7 (routes suite)"
    )
    parser.add_argument(
        "--rtol",
        type=float,
        help="relL2 gate; default 0.10 (0.25 for a4w4) in the size suite, 0.08 in the routes suite",
    )
    parser.add_argument(
        "--max-tok-per-rank",
        "--mtpr",
        type=int,
        help="instance capacity; size suite default: one instance per batch size, "
        "routes suite default: 8192",
    )
    parser.add_argument(
        "--large-m-state-slots",
        "--slots",
        type=int,
        help="large-input layer state slots; default 2 (size suite, so the edge checks "
        "can bind another slot), 61 (routes suite, a DeepSeek model's layers)",
    )
    size = parser.add_argument_group("size suite")
    size.add_argument(
        "--stage2-p2p-quant",
        choices=("auto", "none", "fp8_blockwise_1x32"),
        default="auto",
    )
    size.add_argument("--bs-list", default="128")
    size.add_argument("--iters", type=int, default=30)
    size.add_argument("--accuracy-max-bs", type=int, default=128)
    size.add_argument("--rank-tokens", default="")
    size.add_argument("--config-tokens", type=int, default=0)
    size.add_argument("--unify-fields", default="")
    size.add_argument("--burst-depth", type=int, default=0)
    size.add_argument("--concentrate-experts", type=int, default=0)
    size.add_argument("--zipf", type=float, default=0.0)
    size.add_argument("--force-fanout-boundary", action="store_true")
    size.add_argument("--inject-invalid-route", action="store_true")
    size.add_argument("--force-padding-boundary", action="store_true")
    size.add_argument(
        "--pad-tail-rows",
        type=int,
        default=0,
        help="Also route each rank's last N rows to -1 (whole rows) and check "
        "forward(mask_invalid_slots=True) returns them as zeros",
    )
    size.add_argument("--expect-large-m", action="store_true")
    size.add_argument("--compare-main", action="store_true")
    size.add_argument("--smoke-only", action="store_true")
    size.add_argument("--layer-stress-cycles", type=int, default=0)
    size.add_argument("--route-switch-rounds", type=int, default=0)
    routes = parser.add_argument_group("routes suite")
    routes.add_argument(
        "--small-m-mtprs",
        type=_int_list,
        default=[256, 512],
        help="small-input runtime MTPRs",
    )
    routes.add_argument("--alt-steps", type=int, default=600)
    routes.add_argument("--graph-layers", type=int, default=8)
    routes.add_argument("--graph-reps", type=int, default=6)
    routes.add_argument("--graph-bs", type=int, default=168)
    routes.add_argument(
        "--time-bs",
        type=_int_list,
        default=[7, 14, 28, 56, 70, 84, 126, 168, 224, 280, 336, 448],
    )
    routes.add_argument("--time-only", action="store_true")
    routes.add_argument(
        "--check-time-acc",
        action="store_true",
        help="check every timed route against the fp32 reference",
    )
    routes.add_argument("--time-calls", type=int, default=50)
    routes.add_argument("--time-trials", type=int, default=15)
    args = parser.parse_args()

    if args.suite == "routes":
        if get_gfx() != "gfx950":
            print(
                f"[SKIP] the small- and large-input routes need gfx950, got {get_gfx()}"
            )
            return
        if args.network not in ("v4_pro", "v41_flash") or args.quant != "a8w4":
            raise ValueError(
                "--suite routes needs --network v4_pro or v41_flash and --quant a8w4"
            )
        args.seed = 7 if args.seed is None else args.seed
        args.rtol = 0.08 if args.rtol is None else args.rtol
        if args.max_tok_per_rank is None:
            args.max_tok_per_rank = 8192
        if args.large_m_state_slots is None:
            args.large_m_state_slots = 61
        run = _run_routes
    else:
        if args.route_switch_rounds and not (args.expect_large_m and args.smoke_only):
            raise ValueError(
                "--route-switch-rounds requires --expect-large-m --smoke-only"
            )
        if (
            sum(
                (
                    args.force_fanout_boundary,
                    args.inject_invalid_route,
                    args.force_padding_boundary,
                )
            )
            > 1
        ):
            raise ValueError("fanout adversarial flags are mutually exclusive")
        if args.rtol is None:
            args.rtol = 0.25 if args.quant == "a4w4" else 0.10
        batch_sizes = [int(value) for value in args.bs_list.split(",")]
        if not batch_sizes or min(batch_sizes) <= 0:
            raise ValueError("--bs-list must contain positive integers")
        rank_tokens = _int_list(args.rank_tokens)
        args.seed = 123 if args.seed is None else args.seed
        if args.large_m_state_slots is None:
            args.large_m_state_slots = 2
        args.batch_sizes, args.rank_tokens_list = batch_sizes, rank_tokens
        run = _run_sizes

    rank, world, device = _setup_dist()
    try:
        run(args, rank, world, device)
    finally:
        _cleanup()


if __name__ == "__main__":
    main()
