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
# routed scale, renormalize, block size, residual, token counts. The counts cover the
# one-shot and the two-shot protocol, and counts that are not a multiple of the world size.
# More than 256 experts keep the selection values in LDS.
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
        16,
        True,
        [1, 3, 8, 24, 64, 130],
    ),
    (
        7168,
        256,
        8,
        False,
        "sigmoid",
        True,
        8,
        4,
        False,
        True,
        2.5,
        True,
        32,
        True,
        [2, 16, 61],
    ),
    (
        4096,
        128,
        8,
        False,
        "softmax",
        False,
        1,
        1,
        False,
        False,
        1.0,
        True,
        64,
        False,
        [5, 40],
    ),
    (
        2880,
        128,
        4,
        True,
        "softmax",
        False,
        1,
        1,
        False,
        False,
        1.0,
        False,
        128,
        True,
        [4, 33],
    ),
    (
        2048,
        384,
        8,
        True,
        "sigmoid",
        True,
        8,
        4,
        True,
        True,
        1.0,
        True,
        16,
        False,
        [3, 50],
    ),
]
MAX_TOKENS = 192

SKIP = arch_info.get_arch() != "gfx950"
pytestmark = pytest.mark.skipif(SKIP, reason="needs gfx950")


def _reference_norm(parts, residual, weight, gemma, num_warps):
    """Residual out and normed on the CPU. The CPU gives IEEE fp32 division and square
    root, and the sum of squares has the order of the kernel."""
    tokens, hidden = parts[0].shape
    h = parts[0].float()
    for p in parts[1:]:
        h = h + p.float()
    s = h.to(torch.bfloat16).float()
    if residual is not None:
        s = s + residual.float()
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
    return res_out, y


def _mxfp8(y):
    """MXFP8 of the fp32 y: e4m3 values and an E8M0 scale for each group of 32. The
    group amax rounds to a power of 2 in fp32, up from 1.75 x 2^e (2^128 is inf). The
    scale is this power / 2^8, clamped to 2^-127 .. 2^127."""
    tokens, hidden = y.shape
    groups = y.view(tokens, hidden // 32, 32).double()
    amax = groups.abs().amax(-1)
    mant, exp = torch.frexp(amax)  # amax = mant * 2^exp, 0.5 <= mant < 1
    pow2 = torch.exp2((exp - 1 + (mant >= 0.875).to(torch.int32)).double())
    pow2 = torch.where(amax > 0, pow2, 0.0).float().double()
    k = (torch.log2(pow2) - 8).clamp(-127, 127)
    x_q = (groups * torch.exp2(-k)[..., None]).float().view(tokens, hidden)
    return x_q.to(torch.float8_e4m3fn), (k + 127).to(torch.uint8)


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


def _steps(a, b):
    """The number of rounding steps between the bf16 values a and b. A value that is
    not finite must have the same bits as the reference."""
    ia, ib = (t.view(torch.int16).int() & 0xFFFF for t in (a, b))
    ka, kb = (torch.where(i >> 15 == 1, -(i & 0x7FFF), i & 0x7FFF) for i in (ia, ib))
    finite = torch.isfinite(a.float()) & torch.isfinite(b.float())
    return torch.where(finite, (ka - kb).abs(), (ia != ib).int() << 16)


def _check(out, ref, shape, residual_ok, failures, tag):
    """Compares the outputs of one call with the reference. A bf16 output can be one
    rounding step from the reference, in 1 of 10,000 values (at least 8). The MXFP8
    outputs and the routing must agree with a reference made from the normed output of
    the kernel."""
    hidden, e, k, _, scoring, _, groups, topk_group, _, quant, scale, renorm, *_ = shape
    bs = shape[12]
    res_out, y, router_w, bias = ref
    normed = out["normed"].cpu()
    for name, o, r in (
        ("residual_out", out["residual_out"].cpu(), res_out),
        ("normed", normed, y),
    ):
        steps = _steps(o, r)
        n, worst = int((steps > 0).sum()), int(steps.max())
        if worst > 1 or n > max(8, steps.numel() // 10000):
            failures.append(f"{tag}: {name}: {n} values differ, by up to {worst} steps")
    if quant:
        x_q, x_s = _mxfp8(normed.float())
        if not torch.equal(out["x_q"].cpu().view(torch.uint8), x_q.view(torch.uint8)):
            failures.append(f"{tag}: x_q differs")
        if not torch.equal(out["x_scale"].cpu(), x_s):
            failures.append(f"{tag}: x_scale differs")
    logits = normed.double() @ router_w.double().T
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
        out["topk_ids"], out["topk_weights"], e, hidden, torch.bfloat16, block_size=bs
    )
    n = int(nv[0])
    if not (
        torch.equal(nv, out["num_valid_ids"])
        and torch.equal(sids[:n], out["sorted_ids"][:n])
        and torch.equal(sw[:n], out["sorted_weights"][:n])
        and torch.equal(seids[: n // bs], out["sorted_expert_ids"][: n // bs])
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
    residual = None
    if shape[13]:
        gen.manual_seed(1000 * seed + 999)
        residual = torch.randn(
            (tokens, hidden), generator=gen, dtype=torch.float32, device=device
        ).to(torch.bfloat16)
    if tokens > 2:  # a zero row: all-zero groups, rstd = 1 / sqrt(eps)
        x[1] = 0
        if residual is not None:
            residual[1] = 0
    parts = [torch.empty_like(x) for _ in range(dist.get_world_size(group))]
    dist.all_gather(parts, x, group=group)
    norm_w, router_w, bias = weights
    res_out, y = _reference_norm(
        [p.cpu() for p in parts],
        residual.cpu() if residual is not None else None,
        norm_w.cpu(),
        shape[8],
        num_warps,
    )
    ref = (res_out, y, router_w.cpu(), bias.cpu() if bias is not None else None)
    return x, residual, ref


def _call(x, residual, weights, shape, comm, out=None):
    norm_w, router_w, bias = weights
    _, _, _, _, scoring, _, groups, topk_group, gemma, quant, scale, renorm, *_ = shape
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
        block_size=shape[12],
        out=out,
    )


def _ranks_agree(ids, group):
    all_ids = [torch.empty_like(ids) for _ in range(dist.get_world_size(group))]
    dist.all_gather(all_ids, ids, group=group)
    return all(torch.equal(all_ids[0], a) for a in all_ids)


def _check_shape(rank, device, tp, shape, failures):
    hidden, e, k, fp32, _, has_bias, _, _, gemma, *_, token_counts = shape
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
    static = [(x.clone(), r if r is None else r.clone()) for x, r, _ in data[0]]
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
            if r is not None:
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
    # The comm size of the first shape: the launches use kernels that each rank has
    # already compiled.
    comm = AllReduceRmsnormMoeRoute(
        tp.cpu_group, MAX_TOKENS, hidden, e, k, router_fp32=fp32
    )
    limit = comm.timeout_us
    try:
        for tokens in (2, 64):  # one-shot, then two-shot
            x, r, ref = _inputs(
                shape, tokens, 9, rank, device, tp.device_group, weights, comm.num_warps
            )
            torch.cuda.synchronize()
            dist.barrier(group=tp.cpu_group)
            comm.timeout_us = 50_000  # 50 ms for the launch that the last rank skips
            if rank != comm.world - 1:
                _call(x, r, weights, shape, comm)
            torch.cuda.synchronize()
            dist.barrier(group=tp.cpu_group)  # the other ranks stopped
            record = comm.error()
            if record[0] != 1 or record[3] not in (1, 2, 3, 4, 5):
                failures.append(f"rank {rank}: error record {record} after a time-out")
            # The launch after reset has the default limit: a rank that starts late
            # must not stop the others.
            comm.timeout_us = limit
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


def _check_routing_edges(rank, device, tp, failures):
    """Softmax routing with logits far apart, and the inputs that the wrapper rejects. Rank 0
    sends 2.0 in the columns 128 r .. 128 r + 127 of row r and the other ranks send zeros, so
    the normed row r is 2.0 in these columns and the logits of row r are exactly
    2 router_weight[:, 128 r]."""
    hidden, e, k = 512, 16, 4
    logits = torch.tensor(
        [
            [300, 200, 100, 0] + [-1000] * 12,
            [0, -88, -89, -150, 7] + [-1000] * 11,
            [10, 9, 5, 0, 4.5] + [-30] * 11,
            [-500, -400, -450, -300] + [-1000] * 12,
        ],
        dtype=torch.float32,
        device=device,
    )
    x = torch.zeros((4, hidden), dtype=torch.bfloat16, device=device)
    router_w = torch.zeros((e, hidden), dtype=torch.float32, device=device)
    for r in range(4):
        if rank == 0:
            x[r, 128 * r : 128 * (r + 1)] = 2.0
        router_w[:, 128 * r] = logits[r] / 2
    norm_w = torch.ones((hidden,), dtype=torch.bfloat16, device=device)
    comm = AllReduceRmsnormMoeRoute(tp.cpu_group, 4, hidden, e, k)
    tag = f"world {comm.world} rank {rank} routing edges"
    try:
        for renorm in (True, False):
            out = fused_all_reduce_rmsnorm_moe_route(
                x,
                None,
                norm_w,
                EPS,
                router_w,
                comm,
                scoring="softmax",
                renormalize=renorm,
            )
            torch.cuda.synchronize()
            ids_r, w_r, _ = _reference_route(
                logits.cpu(), None, k, "softmax", 1, 1, renorm, 1.0
            )
            w = out["topk_weights"].cpu().double()
            if not torch.equal(out["topk_ids"].cpu().long(), ids_r):
                failures.append(f"{tag}: softmax top-k ids differ (renorm {renorm})")
            if not (torch.isfinite(w).all() and torch.allclose(w, w_r, 1e-5, 1e-7)):
                failures.append(
                    f"{tag}: softmax weights {w.tolist()} (renorm {renorm})"
                )
        bias = torch.zeros((e,), dtype=torch.float32, device=device)
        strided = torch.zeros((2 * e,), dtype=torch.float32, device=device)[::2]
        rejected = {
            "softmax with a correction bias": {
                "correction_bias": bias,
                "scoring": "softmax",
            },
            "a strided correction bias": {"correction_bias": strided},
            "kept groups with fewer experts than topk": {
                "num_expert_group": 8,
                "topk_group": 1,
            },
            "groups of one expert with a correction bias": {
                "correction_bias": bias,
                "num_expert_group": 16,
                "topk_group": 4,
            },
        }
        for name, kwargs in rejected.items():
            try:
                fused_all_reduce_rmsnorm_moe_route(
                    x, None, norm_w, EPS, router_w, comm, **kwargs
                )
                failures.append(f"{tag}: {name} was accepted")
            except AssertionError:
                pass
        if comm.error() != [0, 0, 0, 0]:
            failures.append(f"{tag}: error {comm.error()}")
    finally:
        comm.close()


# Group amax values at the edges of the MXFP8 scale rule, and their scale bytes.
EDGE_SCALES = [
    (1.75 * 2.0**-119, 1),  # the smallest amax with a scale above 2^-127
    (4e-36, 1),
    (5.2e-36, 1),
    (1.75 * 2.0**-118, 2),
    (2.0**-100, 19),
    (1.75 / 8, 117),
    (1.7421875, 119),
    (1.75, 120),
    (28.0, 124),
    (448.0, 128),
    (2.0**100, 219),
    (1.75 * 2.0**127, 254),  # amax rounds to 2^128 (inf) in fp32
    (3.3e38, 254),
]


def _check_scales(rank, device, tp, failures):
    """Groups of 32 with an amax from EDGE_SCALES. Rank 0 has rows of +-1, the other
    ranks 0, and there is no residual: then normed = +-norm_weight, and the weight sets
    the amax. The router weights are 0."""
    shape = (6144, 128, 4, True, "sigmoid", False, 1, 1, False, True, 1.0, True, 16)
    shape = shape + (False, [4])
    hidden, e, k = shape[:3]
    tokens = shape[-1][0]
    gen = torch.Generator(device=device).manual_seed(77)
    signs = torch.randint(0, 2, (tokens, hidden), generator=gen, device=device)
    ones = (2 * signs - 1).to(torch.bfloat16)
    x = ones if rank == 0 else torch.zeros_like(ones)
    u = 0.5 + 0.5 * torch.rand(
        (hidden // 32, 32), generator=gen, dtype=torch.float32, device=device
    )
    g = torch.arange(hidden // 32, device=device)
    u[g, g % 32] = 1.0  # one value of each group is the amax
    amax = torch.tensor([a for a, _ in EDGE_SCALES], dtype=torch.float32, device=device)
    norm_w = (amax[g % len(EDGE_SCALES)][:, None] * u).view(hidden).to(torch.bfloat16)
    router_w = torch.zeros((e, hidden), dtype=torch.float32, device=device)
    comm = AllReduceRmsnormMoeRoute(tp.cpu_group, tokens, hidden, e, k)
    parts = [ones.cpu()] + [torch.zeros_like(ones.cpu())] * (comm.world - 1)
    res_out, y = _reference_norm(parts, None, norm_w.cpu(), False, comm.num_warps)
    want = torch.tensor([b for _, b in EDGE_SCALES], dtype=torch.uint8)
    want = want[torch.arange(hidden // 32) % len(EDGE_SCALES)].expand(tokens, -1)
    assert torch.equal(_mxfp8(y.float())[1], want), "EDGE_SCALES or _mxfp8"
    try:
        torch.cuda.synchronize()
        dist.barrier(group=tp.cpu_group)
        out = _call(x, None, (norm_w, router_w, None), shape, comm)
        torch.cuda.synchronize()
        ref = (res_out, y, router_w.cpu(), None)
        _check(out, ref, shape, True, failures, f"rank {rank}: edge scales")
        if not torch.equal(out["x_scale"].cpu(), want):
            failures.append(f"rank {rank}: edge scales: the scale bytes differ")
        if comm.error() != [0, 0, 0, 0]:
            failures.append(f"rank {rank}: edge scales: error {comm.error()}")
    finally:
        comm.close()


def _check_max_tokens(rank, device, tp, failures):
    """The constructor rejects a max_tokens that leaves no free CTA for a call. When it
    accepts a max_tokens, a call with that many tokens is correct. 300 tokens use a
    512-row bucket: at 2 ranks it needs all 256 CTAs. The constructor also rejects a shape
    that needs more LDS than a CU has: 2048 experts at hidden 8192 need 304 KiB."""
    try:
        AllReduceRmsnormMoeRoute(tp.cpu_group, 64, 8192, 2048, 8).close()
        failures.append(f"rank {rank}: a shape that needs 304 KiB of LDS was accepted")
    except AssertionError:
        pass
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
    for max_tokens in (300, 512):
        try:
            comm = AllReduceRmsnormMoeRoute(
                tp.cpu_group, max_tokens, hidden, e, k, router_fp32=fp32
            )
        except AssertionError:
            continue  # rejected: no call can fail later
        tag = f"world {comm.world} rank {rank} max_tokens {max_tokens}"
        try:
            x, r, ref = _inputs(
                shape,
                max_tokens,
                11,
                rank,
                device,
                tp.device_group,
                weights,
                comm.num_warps,
            )
            try:
                out = _call(x, r, weights, shape, comm)
            except Exception as err:  # noqa: BLE001 - every rank raises the same error
                failures.append(f"{tag}: the call failed: {type(err).__name__}: {err}")
                continue
            torch.cuda.synchronize()
            _check(out, ref, shape, True, failures, tag)
        finally:
            comm.close()


def _worker(rank, world_size, init_method):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(world_size, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    failures = []
    # 8 ranks run only the first shape and the time-out check, which launches its kernel.
    # Each other kernel variant compiles on all 8 ranks, which takes most of the CI time.
    # 2 and 4 ranks run all checks.
    full = world_size < 8
    try:
        for shape in SHAPES if full else SHAPES[:1]:
            _check_shape(rank, device, tp, shape, failures)
        _check_timeout(rank, device, tp, failures)
        if full:
            _check_scales(rank, device, tp, failures)
            _check_routing_edges(rank, device, tp, failures)
            _check_max_tokens(rank, device, tp, failures)
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
