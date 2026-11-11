# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import multiprocessing as mp

import pytest
import torch
import torch.distributed as dist

import aiter
from aiter import logger
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.fused_moe import moe_sorting
from aiter.ops.triton.comms.fused.all_reduce_rmsnorm_moe_route import (
    AllReduceRmsnormMoeRoute,
    fused_all_reduce_rmsnorm_moe_route,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import ensure_spawn_method

WORLD_SIZES = [2, 4, 8]
EPS = 1e-6
# hidden, experts, topk, fp32 router, scoring, bias, groups, topk groups, gemma, quant,
# routed scale, renormalize, token counts. The counts cover the one-shot and the two-shot
# protocol, and counts that are not a multiple of the world size.
SHAPES = [
    (
        6144,
        128,
        4,
        True,
        "sigmoid",
        True,
        1,
        1,
        True,
        True,
        2.0,
        True,
        [1, 3, 8, 24, 64, 130],
    ),
    (7168, 256, 8, False, "sigmoid", True, 8, 4, False, True, 2.5, True, [2, 16, 61]),
    (4096, 128, 8, False, "softmax", False, 1, 1, False, False, 1.0, True, [5, 40]),
    (2880, 128, 4, True, "softmax", False, 1, 1, False, False, 1.0, False, [4, 33]),
]
MAX_TOKENS = 192
F32_TINY = torch.finfo(torch.float32).tiny

SKIP = arch_info.get_arch() != "gfx950"
pytestmark = pytest.mark.skipif(SKIP, reason="needs gfx950")


def _reference_norm(parts, residual, weight, gemma, num_warps):
    """h, residual out, normed, x_q and x_scale on the CPU. The CPU gives IEEE fp32 division
    and square root, and the sum of squares has the order of the kernel."""
    tokens, hidden = residual.shape
    h = parts[0].float()
    for p in parts[1:]:
        h = h + p.float()
    s = h.to(torch.bfloat16).float() + residual.float()
    res_out = s.to(torch.bfloat16)
    nv = hidden // 8
    v_ = -(-nv // (64 * num_warps))
    a_ = nv // v_
    v = s.view(tokens, v_, a_, 8).permute(0, 2, 1, 3).reshape(tokens, a_, 8 * v_)
    sq = v[..., 0] * v[..., 0]
    for i in range(1, 8 * v_):
        sq = sq + v[..., i] * v[..., i]
    waves = -(-a_ // 64)
    lanes = torch.zeros((tokens, waves * 64), dtype=torch.float32, device="cpu")
    lanes[:, :a_] = sq
    lanes = lanes.view(tokens, waves, 64)
    lane_ids = torch.arange(64, device="cpu")
    for off in (32, 16, 8, 4, 2, 1):
        lanes = lanes + lanes[..., lane_ids ^ off]
    total = lanes[:, 0, 0]
    for w in range(1, waves):
        total = total + lanes[:, w, 0]
    var = total / torch.tensor(float(hidden), dtype=torch.float32, device="cpu")
    eps32 = torch.tensor(EPS, dtype=torch.float32, device="cpu")
    one = torch.tensor(1.0, dtype=torch.float32, device="cpu")
    rstd = one / torch.sqrt(var + eps32)
    w = weight.float() + (1.0 if gemma else 0.0)
    y = ((s * rstd[:, None]) * w).to(torch.bfloat16)
    groups = y.float().view(tokens, hidden // 32, 32)
    amax = torch.clamp(groups.abs().amax(-1), min=F32_TINY)
    bits = (amax / 448.0).view(torch.int32)
    ex = (bits >> 23) & 0xFF
    scale = torch.clamp(ex + ((bits & 0x7FFFFF) != 0).to(torch.int32), 0, 254)
    scale = torch.where(ex == 0, 0, scale)
    inv = ((254 - scale) << 23).view(torch.float32)
    x_q = (groups * inv[..., None]).view(tokens, hidden).to(torch.float8_e4m3fn)
    return res_out, y, x_q, scale.to(torch.uint8)


def _reference_route(logits, bias, k, scoring, groups, topk_group, renorm, scale):
    """Top-k ids, weights and the gap between the k-th and the next selection value, from
    fp64 logits. Ties go to the lower expert."""
    lf = logits.double()
    sc = torch.sigmoid(lf) if scoring == "sigmoid" else lf
    sel = sc + (bias.double() if bias is not None else 0.0)
    tokens, e = sel.shape
    if groups > 1:
        gv = sel.view(tokens, groups, e // groups)
        if bias is not None:
            gs = gv.topk(2, dim=-1).values.sum(-1)
        else:
            gs = gv.max(-1).values
        keep = torch.sort(gs, dim=-1, descending=True, stable=True).indices[
            :, :topk_group
        ]
        mask = torch.zeros(tokens, groups, dtype=torch.bool, device="cpu")
        mask.scatter_(1, keep, True)
        sel = torch.where(
            mask.repeat_interleave(e // groups, dim=1), sel, -float("inf")
        )
    order = torch.sort(sel, dim=-1, descending=True, stable=True)
    ids = order.indices[:, :k]
    gap = (order.values[:, k - 1] - order.values[:, k]).abs()
    if scoring == "sigmoid":
        w = sc.gather(1, ids)
        if renorm:
            w = w / w.sum(-1, keepdim=True)
    elif renorm:
        w = torch.softmax(lf.gather(1, ids), dim=-1)
    else:
        w = torch.softmax(lf, dim=-1).gather(1, ids)
    return ids, w * scale, gap


def _bytes_differ(a, b):
    return int(
        (
            a.cpu().contiguous().view(torch.uint8) != b.contiguous().view(torch.uint8)
        ).sum()
    )


def _check(out, ref, shape, residual_ok, failures, tag):
    """Compares the outputs of one call with the reference."""
    hidden, e, k, _, scoring, _, groups, topk_group, _, quant, scale, renorm, _ = shape
    res_out, y, x_q, x_s, logits, bias = ref
    bad = _bytes_differ(out["residual_out"], res_out) + _bytes_differ(out["normed"], y)
    if quant:
        bad += _bytes_differ(out["x_q"], x_q) + _bytes_differ(out["x_scale"], x_s)
    if bad:
        failures.append(f"{tag}: {bad} bytes of the norm outputs differ")
    ids_r, w_r, gap = _reference_route(
        logits, bias, k, scoring, groups, topk_group, renorm, scale
    )
    ids = out["topk_ids"].cpu().long()
    same = (ids.sort(-1).values == ids_r.sort(-1).values).all(-1)
    # A set can differ only where the k-th and the next value are a rounding apart.
    if (~same & (gap > 1e-5)).any():
        failures.append(f"{tag}: top-k sets differ in {int((~same).sum())} rows")
    w = out["topk_weights"].cpu().double()
    w_s = w.gather(1, ids.sort(-1).indices)[same]
    w_r_s = w_r.gather(1, ids_r.sort(-1).indices)[same]
    if same.any() and ((w_s - w_r_s).abs() > 1e-5 * w_r_s.abs() + 1e-7).any():
        failures.append(f"{tag}: top-k weights differ")
    sids, sw, seids, nv, _ = moe_sorting(
        out["topk_ids"], out["topk_weights"], e, hidden, torch.bfloat16, block_size=16
    )
    n = int(nv[0])
    if not (
        torch.equal(nv, out["num_valid_ids"])
        and torch.equal(sids[:n], out["sorted_ids"][:n])
        and torch.equal(sw[:n], out["sorted_weights"][:n])
        and torch.equal(seids[: n // 16], out["sorted_expert_ids"][: n // 16])
    ):
        failures.append(f"{tag}: the sort differs from moe_sorting")
    if not bool((out["moe_buf"] == 0).all()):
        failures.append(f"{tag}: moe_buf is not zero")
    if not residual_ok:
        failures.append(f"{tag}: the ranks got different top-k ids")


def _inputs(shape, tokens, seed, rank, device, group, weights, num_warps):
    hidden = shape[0]
    gen = torch.Generator(device=device).manual_seed(1000 * seed + rank)
    x = torch.randn(
        (tokens, hidden), generator=gen, dtype=torch.float32, device=device
    ).to(torch.bfloat16)
    gen.manual_seed(1000 * seed + 999)
    residual = torch.randn(
        (tokens, hidden), generator=gen, dtype=torch.float32, device=device
    ).to(torch.bfloat16)
    if tokens > 2:  # a zero row: all-zero groups, rstd = 1 / sqrt(eps)
        x[1] = 0
        residual[1] = 0
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, x, group=group)
    norm_w, router_w, bias = weights
    res_out, y, x_q, x_s = _reference_norm(
        [p.cpu() for p in parts], residual.cpu(), norm_w.cpu(), shape[8], num_warps
    )
    logits = y.double() @ router_w.cpu().double().T
    ref = (res_out, y, x_q, x_s, logits, bias.cpu() if bias is not None else None)
    return x, residual, ref


def _call(x, residual, weights, shape, comm, out=None):
    norm_w, router_w, bias = weights
    _, _, _, _, scoring, _, groups, topk_group, gemma, quant, scale, renorm, _ = shape
    return fused_all_reduce_rmsnorm_moe_route(
        x,
        residual,
        norm_w,
        EPS,
        router_w,
        comm,
        correction_bias=bias,
        scoring=scoring,
        renormalize=renorm,
        routed_scaling_factor=scale,
        num_expert_group=groups,
        topk_group=topk_group,
        gemma=gemma,
        quant=quant,
        out=out,
    )


def _ranks_agree(ids, group):
    all_ids = [torch.empty_like(ids) for _ in range(dist.get_world_size(group))]
    dist.all_gather(all_ids, ids, group=group)
    return all(torch.equal(all_ids[0], a) for a in all_ids)


def _check_shape(rank, device, tp, shape, failures):
    hidden, e, k, fp32, _, has_bias, _, _, gemma, _, _, _, token_counts = shape
    group = tp.device_group
    gen = torch.Generator(device=device).manual_seed(5)
    norm_w = 0.1 * torch.randn(
        (hidden,), generator=gen, dtype=torch.float32, device=device
    )
    norm_w = (norm_w if gemma else norm_w + 1.0).to(torch.bfloat16)
    router_w = 0.02 * torch.randn(
        (e, hidden), generator=gen, dtype=torch.float32, device=device
    )
    router_w = router_w.to(torch.float32 if fp32 else torch.bfloat16)
    bias = None
    if has_bias:
        bias = 0.01 * torch.randn(
            (e,), generator=gen, dtype=torch.float32, device=device
        )
    weights = (norm_w, router_w, bias)
    comm = AllReduceRmsnormMoeRoute(
        tp.cpu_group, MAX_TOKENS, hidden, e, k, router_fp32=fp32
    )
    tag = f"world {comm.world} rank {rank} hidden {hidden} experts {e} top-{k}"
    try:
        for i, tokens in enumerate(token_counts):
            data = [
                _inputs(
                    shape,
                    tokens,
                    2 * i + j,
                    rank,
                    device,
                    group,
                    weights,
                    comm.num_warps,
                )
                for j in range(2)
            ]
            # 2 launches back to back, with no barrier between them.
            torch.cuda.synchronize()
            dist.barrier(group=tp.cpu_group)
            outs = []
            for x, r, _ in data:
                buf = torch.ones((tokens, hidden), dtype=torch.bfloat16, device=device)
                outs.append(_call(x, r, weights, shape, comm, out={"moe_buf": buf}))
            torch.cuda.synchronize()
            for j, (out, d) in enumerate(zip(outs, data)):
                agree = _ranks_agree(out["topk_ids"], group)
                _check(
                    out,
                    d[2],
                    shape,
                    agree,
                    failures,
                    f"{tag} tokens {tokens} launch {j}",
                )
        _check_graph(rank, device, tp, comm, shape, weights, failures, tag)
        if comm.error() != [0, 0, 0, 0]:
            failures.append(f"{tag}: error {comm.error()}")
    finally:
        comm.close()


def _check_graph(rank, device, tp, comm, shape, weights, failures, tag):
    """3 launches of mixed sizes (both protocols) in a HIP graph, replayed 3 times on new
    data."""
    counts = shape[-1]
    sizes = [counts[-1], counts[0], counts[len(counts) // 2]]
    data = [
        [
            _inputs(
                shape,
                t,
                100 + 3 * rep + j,
                rank,
                device,
                tp.device_group,
                weights,
                comm.num_warps,
            )
            for j, t in enumerate(sizes)
        ]
        for rep in range(3)
    ]
    static = [(x.clone(), r.clone()) for x, r, _ in data[0]]
    bufs = [
        torch.ones((t, shape[0]), dtype=torch.bfloat16, device=device) for t in sizes
    ]
    torch.cuda.synchronize()
    dist.barrier(group=tp.cpu_group)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = [
            _call(x, r, weights, shape, comm, out={"moe_buf": b})
            for (x, r), b in zip(static, bufs)
        ]
    for rep in range(3):
        for (x, r), d, b in zip(static, data[rep], bufs):
            x.copy_(d[0])
            r.copy_(d[1])
            b.fill_(1)
        torch.cuda.synchronize()
        dist.barrier(group=tp.cpu_group)
        graph.replay()
        torch.cuda.synchronize()
        for j, (out, d) in enumerate(zip(outs, data[rep])):
            agree = _ranks_agree(out["topk_ids"], tp.device_group)
            _check(
                out,
                d[2],
                shape,
                agree,
                failures,
                f"{tag}: graph replay {rep} launch {j}",
            )


def _check_timeout(rank, device, tp, failures):
    """The last rank skips a launch. The other ranks must stop at the time limit and write
    the error record of each rank. After reset, a launch is correct again."""
    shape = SHAPES[0]
    hidden, e, k, fp32 = shape[:4]
    gen = torch.Generator(device=device).manual_seed(5)
    weights = (
        (
            0.1
            * torch.randn((hidden,), generator=gen, dtype=torch.float32, device=device)
        ).to(torch.bfloat16),
        0.02
        * torch.randn((e, hidden), generator=gen, dtype=torch.float32, device=device),
        0.01 * torch.randn((e,), generator=gen, dtype=torch.float32, device=device),
    )
    comm = AllReduceRmsnormMoeRoute(
        tp.cpu_group, 64, hidden, e, k, router_fp32=fp32, timeout_s=0.05
    )
    try:
        for tokens in (2, 64):  # one-shot, then two-shot
            x, r, ref = _inputs(
                shape, tokens, 9, rank, device, tp.device_group, weights, comm.num_warps
            )
            torch.cuda.synchronize()
            dist.barrier(group=tp.cpu_group)
            if rank != comm.world - 1:
                _call(x, r, weights, shape, comm)
            torch.cuda.synchronize()
            dist.barrier(group=tp.cpu_group)  # the other ranks stopped
            record = comm.error()
            if record[0] != 1 or record[3] not in (1, 2, 3, 4, 5):
                failures.append(f"rank {rank}: error record {record} after a time-out")
            comm.reset()
            out = _call(x, r, weights, shape, comm, out={"moe_buf": torch.ones_like(x)})
            torch.cuda.synchronize()
            _check(
                out,
                ref,
                shape,
                True,
                failures,
                f"rank {rank} tokens {tokens} after reset",
            )
            if comm.error() != [0, 0, 0, 0]:
                failures.append(f"rank {rank}: error record not clear after reset")
    finally:
        comm.close()


def _worker(rank, world_size, init_method):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(world_size, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    failures = []
    try:
        for shape in SHAPES:
            _check_shape(rank, device, tp, shape, failures)
        _check_timeout(rank, device, tp, failures)
        return failures
    finally:
        aiter.destroy_dist_env()


@pytest.mark.parametrize("world_size", WORLD_SIZES)
def test_all_reduce_rmsnorm_moe_route(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"needs {world_size} GPUs")
    ensure_spawn_method()
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(processes=world_size) as pool:
        results = [
            pool.apply_async(_worker, (rank, world_size, init_method))
            for rank in range(world_size)
        ]
        failures = [f for r in results for f in r.get(timeout=1800)]
    assert not failures, "\n".join(failures)


if __name__ == "__main__":  # CI runs the multi-GPU tests as scripts
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-t",
        "--world-size",
        type=int,
        nargs="+",
        default=[w for w in WORLD_SIZES if w <= torch.cuda.device_count()],
        choices=WORLD_SIZES,
    )
    args = parser.parse_args()
    if SKIP or not args.world_size:
        logger.warning("test_all_reduce_rmsnorm_moe_route skipped: needs gfx950 GPUs")
    for world_size in [] if SKIP else args.world_size:
        test_all_reduce_rmsnorm_moe_route(world_size)
        logger.info(
            "test_all_reduce_rmsnorm_moe_route: world size %d passed", world_size
        )
