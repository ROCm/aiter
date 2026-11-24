# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import itertools
import multiprocessing as mp
import os
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

import aiter
from aiter import logger
from aiter.dist.communication_op import (
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_fused_allreduce_rmsnorm,
    tensor_model_parallel_fused_allreduce_rmsnorm_quant,
)
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.triton.comms import all_reduce as plain
from aiter.ops.triton.comms.all_reduce import (
    all_reduce,
    all_reduce_free,
    all_reduce_init,
    all_reduce_rmsnorm_per_token_quant,
)
from aiter.ops.triton.comms.fused.all_reduce_rmsnorm_quant import (
    all_reduce_rmsnorm_quant,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import ensure_spawn_method

WORLD_SIZE = 4
HIDDEN = 2048
EPS = 1e-6
TOKENS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

SKIP = torch.cuda.device_count() < WORLD_SIZE or arch_info.get_arch() not in (
    "gfx1250",
    "gfx1250-strict",
)
pytestmark = pytest.mark.skipif(SKIP, reason="needs 4 gfx1250 GPUs")


def _inputs(tokens, rank, device):
    """This rank's x, the residual and the weight, and the fp32 references of
    residual_out and out. The 4 inputs are added in rank order, as in the kernel."""
    gen = torch.Generator(device=device).manual_seed(tokens)
    shape = (tokens, HIDDEN)
    xs = [
        torch.randn(shape, generator=gen, dtype=torch.bfloat16, device=device)
        for _ in range(WORLD_SIZE)
    ]
    residual = torch.randn(shape, generator=gen, dtype=torch.bfloat16, device=device)
    w = torch.randn(HIDDEN, generator=gen, dtype=torch.bfloat16, device=device)
    weight = 0.1 * w  # a Gemma weight: 1 + weight is near 1
    s = xs[0].float() + xs[1].float() + xs[2].float() + xs[3].float() + residual.float()
    out = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + EPS) * (1 + weight.float())
    return xs[rank], residual, weight, s, out


def _compare(name, outs, ref_residual, ref_out, group):
    failures = []
    if not torch.equal(outs[1], ref_residual.to(torch.bfloat16)):
        failures.append(f"{name}: residual_out is not bit-exact")
    try:
        torch.testing.assert_close(outs[0], ref_out.to(torch.bfloat16))
    except AssertionError as e:
        failures.append(f"{name}: out: {e}")
    if len(outs) == 4:
        step = outs[3].repeat_interleave(128, dim=1)
        error = (outs[2].float() * step - ref_out).abs()
        # e4m3 has 3 mantissa bits. Its smallest step is 2^-9 of the scale.
        if not bool((error <= 2.0**-4 * ref_out.abs() + 2.0**-9 * step).all()):
            failures.append(f"{name}: fp8 values are wrong")
    for t in outs:
        parts = [torch.empty_like(t.view(torch.uint8)) for _ in range(WORLD_SIZE)]
        dist.all_gather(parts, t.view(torch.uint8), group=group)
        if not all(torch.equal(parts[0], p) for p in parts):
            failures.append(f"{name}: the ranks have different results")
    return failures


def _worker(rank, init_method):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(WORLD_SIZE, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    group = tp.cpu_group
    cases = [_inputs(tokens, rank, device) for tokens in TOKENS]
    inboxes, calls = all_reduce_init(group, max(TOKENS) * HIDDEN * 2)
    failures = []
    try:
        # First, while the inboxes are empty: the call numbers wrap from 2^30 to 1.
        # Calls of 64 to 67 tokens, so that each call has new inputs.
        calls[:-1].fill_(2**30 - 2)
        for i in range(4):
            x, residual, weight, *ref = _inputs(64 + i, rank, device)
            outs = all_reduce_rmsnorm_quant(
                x, residual, weight, EPS, True, i % 2 == 1, group, inboxes, calls
            )
            name = f"rank {rank} call {i} at the counter wrap"
            failures += _compare(name, outs, *ref, tp.device_group)

        for quant in (False, True):
            for tokens, (x, residual, weight, *ref) in zip(TOKENS, cases):
                outs = all_reduce_rmsnorm_quant(
                    x, residual, weight, EPS, True, quant, group, inboxes, calls
                )
                name = f"rank {rank} tokens {tokens} quant {quant}"
                failures += _compare(name, outs, *ref, tp.device_group)

            # All token counts in one HIP graph, replayed 10 times.
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outs = [
                    all_reduce_rmsnorm_quant(
                        x, r, w, EPS, True, quant, group, inboxes, calls
                    )
                    for x, r, w, _, _ in cases
                ]
            for _ in range(10):
                graph.replay()
            for tokens, o, (_, _, _, *ref) in zip(TOKENS, outs, cases):
                name = f"rank {rank} graph tokens {tokens} quant {quant}"
                failures += _compare(name, o, *ref, tp.device_group)
        if calls[-1].item():
            failures.append(f"rank {rank}: {calls[-1].item()} polls timed out")
    finally:
        all_reduce_free(group, inboxes)
        aiter.destroy_dist_env()
    return failures


def test_all_reduce_rmsnorm_quant():
    ensure_spawn_method()
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(processes=WORLD_SIZE) as pool:
        results = [
            pool.apply_async(_worker, (rank, init_method)) for rank in range(WORLD_SIZE)
        ]
        failures = [f for r in results for f in r.get(timeout=1800)]
    assert not failures, "\n".join(failures[:20])


# all_reduce_rmsnorm_per_token_quant: at hidden 6144 the bits of the split path,
# other sizes within 1 bf16 ulp / 1 FP8 code of an fp32 reference. With quant, at
# sizes that are multiples of 16, the bits of the HIP per-token quant of its own bf16 out.
PT_HIDDEN = 6144
PT_CASES = [(t, PT_HIDDEN) for t in (1, 3, 16, 64, 128, 341)]
PT_CASES += [(t, h) for h in (2048, 6152, 16384) for t in (1, 64, (2 << 20) // h)]


def _pt_inputs(tokens, hidden, seed, device, edge=False):
    """The x of all ranks, the residual and a Gemma weight. Edge rows: 0 (scale 0),
    large, tiny, and one that cancels to 0."""
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (tokens, hidden)
    xs = [
        0.5 * torch.randn(shape, generator=gen, dtype=torch.bfloat16, device=device)
        for _ in range(WORLD_SIZE)
    ]
    res = 2 * torch.randn(shape, generator=gen, dtype=torch.bfloat16, device=device)
    w = torch.randn(hidden, generator=gen, dtype=torch.bfloat16, device=device)
    if edge:
        for x in xs:
            x[0].zero_()
            x[1].mul_(3000.0)
            x[3].mul_(1e-12)
        res[0].zero_()
        res[2] = -sum(x[2].float() for x in xs).bfloat16().float()
        res[3].mul_(1e-12)
    return xs, res, 0.1 * w


def _pt_wrong(outs, xs, res, w, quant, rank, ar, gemma=True):
    """Values off the split path's bits (PT_HIDDEN) or off the fp32 reference. The
    ranks must agree on every bit."""
    if res.shape[1] == PT_HIDDEN:  # plain Gluon all-reduce, add_rmsnorm, quant
        res_out, out = torch.empty_like(res), torch.empty_like(res)
        s = all_reduce(xs[rank], *ar)
        aiter.rmsnorm2d_fwd_with_add(out, s, res, res_out, w, EPS, gemma, 0)
        refs = (out, res_out)
        if quant:
            q = aiter.get_hip_quant(aiter.QuantType.per_Token)
            q, scale = q(out, quant_dtype=torch.float8_e4m3fn)
            refs = (q, res_out, scale)
        bad = sum(
            int((o.view(torch.uint8) != r.view(torch.uint8)).sum())
            for o, r in zip(outs, refs)
        )
    else:  # the residual sum is exact; the row sum of squares has another order
        h = sum(x.float() for x in xs).bfloat16().float() + res.float()
        rstd = torch.rsqrt(h.double().square().mean(-1, keepdim=True) + EPS).float()
        y = (h * rstd * (w.float() + gemma)).bfloat16()
        bad = int((outs[1] != h.bfloat16()).sum())
        if quant:
            amax = y.float().abs().amax(-1, keepdim=True)
            if res.shape[1] in (32, 64, 128):  # the HIP group path floors the amax
                amax = amax.clamp_min(1e-10)
            scale = amax / 448.0
            y = (y.float() / scale).clamp(-448, 448).to(outs[0].dtype)
            bad += int((~torch.isclose(outs[2], scale, rtol=2**-7, atol=0)).sum())
        bits = torch.int8 if quant else torch.int16
        bad += int(((outs[0].view(bits).int() - y.view(bits).int()).abs() > 1).sum())
    for o in outs:
        parts = [torch.empty_like(o.view(torch.uint8)) for _ in range(WORLD_SIZE)]
        dist.all_gather(parts, o.view(torch.uint8), group=ar[0])
        bad += sum(int((p != parts[0]).sum()) for p in parts)
    return bad


def _per_token_worker(rank, init_method):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(WORLD_SIZE, rank, distributed_init_method=init_method)
    group = get_tp_group().cpu_group
    inboxes, calls = all_reduce_init(group, 4 << 20)
    ar = (group, inboxes, calls)
    failures = []

    def check(name, tokens, hidden, seed, quant, edge=False, gemma=True):
        xs, res, w = _pt_inputs(tokens, hidden, seed, device, edge)
        op = all_reduce_rmsnorm_per_token_quant
        outs = op(xs[rank], res, w, EPS, gemma, quant, *ar)
        bad = _pt_wrong(outs, xs, res, w, quant, rank, ar, gemma)
        # The HIP per-token quant leaves the last 4 FP8 bytes of each row unwritten
        # when hidden % 16 == 8 and hidden > 2048, so no bit check against it there.
        if quant and hidden % 16 == 0:
            y = op(xs[rank], res, w, EPS, gemma, False, *ar)[0]
            q = aiter.get_hip_quant(aiter.QuantType.per_Token)
            refs = q(y, quant_dtype=torch.float8_e4m3fn)  # out and scale
            for o, r in zip((outs[0], outs[2]), refs):
                bad += int(
                    (o.view(torch.uint8) != r.view(o.shape).view(torch.uint8)).sum()
                )
        if bad:
            failures.append(f"rank {rank} {name} quant {quant}: wrong")

    try:
        # First, while the inboxes are empty: the call numbers wrap from 2^30 to 1.
        calls[:-1].fill_(2**30 - 2)
        for i in range(4):
            check(f"counter wrap {i}", 8, PT_HIDDEN, 400 + i, i % 2 == 0)
        for quant in (True, False):
            for tokens, hidden in PT_CASES:
                check(f"tokens {tokens} hidden {hidden}", tokens, hidden, tokens, quant)
            for h in (PT_HIDDEN, 32, 64, 128):  # 32 to 128: the HIP group path
                check(
                    f"edge rows {h}", 16 if h == PT_HIDDEN else 256, h, 7, quant, True
                )
            check("no gemma", 16, PT_HIDDEN, 8, quant, gemma=False)
        # After plain calls with a partial last chunk, the 2 chunks a wave receives
        # have other call numbers.
        for i, n in enumerate((3 * 6 * plain._LINES + 20, 17 * 6 * plain._LINES)):
            all_reduce(torch.randn(n, dtype=torch.bfloat16, device=device), *ar)
            check(f"after partial call {i}", 4 + i, 6152, 300 + i, i % 2 == 0)

        # Mixed calls in a HIP graph, replayed with new inputs into NaN outputs.
        cases = [(1, False, PT_HIDDEN), (16, True, 4096), (64, True, PT_HIDDEN)]
        cases += [(33, True, 6152), (3, False, 2048)]
        bufs = [_pt_inputs(t, h, 0, device) for t, _, h in cases]
        bufs = [(xs[rank], res, w) for xs, res, w in bufs]
        odd = torch.zeros(
            3 * 6 * plain._LINES + 20, dtype=torch.bfloat16, device=device
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = []
            for (_, quant, _), buf in zip(cases, bufs):
                outs.append(
                    all_reduce_rmsnorm_per_token_quant(*buf, EPS, True, quant, *ar)
                )
                all_reduce(odd, *ar)
        for rep in range(3):
            new = [
                _pt_inputs(t, h, 9 * rep + i, device)
                for i, (t, _, h) in enumerate(cases)
            ]
            for (xs, res, w), buf, o in zip(new, bufs, outs):
                for dst, src in zip(buf, (xs[rank], res, w)):
                    dst.copy_(src)
                for v in o:
                    v.view(torch.uint8).fill_(255)  # NaN
            graph.replay()
            for (t, quant, _), args, o in zip(cases, new, outs):
                if _pt_wrong(o, *args, quant, rank, ar):
                    failures.append(f"rank {rank} graph {rep} tokens {t}: wrong")
        if calls[-1].item():
            failures.append(f"rank {rank}: {calls[-1].item()} polls timed out")
    finally:
        all_reduce_free(group, inboxes)
        aiter.destroy_dist_env()
    return failures


def test_all_reduce_rmsnorm_per_token_quant():
    ensure_spawn_method()
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(processes=WORLD_SIZE) as pool:
        results = [
            pool.apply_async(_per_token_worker, (rank, init_method))
            for rank in range(WORLD_SIZE)
        ]
        failures = [f for r in results for f in r.get(timeout=1800)]
    assert not failures, "\n".join(failures[:20])


def _dispatch_worker(rank, init_method):
    os.environ["AITER_GLUON_FUSED_AR"] = "1"  # read when the TP communicator is made
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(WORLD_SIZE, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    failures = []
    try:
        if tp.device_communicator.gluon_ar is None:
            failures.append(f"rank {rank}: the TP communicator has no Gluon inboxes")
        for tokens in (1, 8, 128, 1024, 2048):  # 2048 rows at hidden 2048 are 8 MiB
            x, residual, weight, *ref = _inputs(tokens, rank, device)
            outs = tensor_model_parallel_fused_allreduce_rmsnorm(
                x, residual, weight, EPS, gemma_norm=True
            )
            name = f"rank {rank} dispatch tokens {tokens}"
            failures += _compare(name, outs, *ref, tp.device_group)

        # The plain all-reduce, up to 4 MiB. Bit-exact only on the Gluon path.
        if tp.device_communicator.gluon_ar_plain is None:
            failures.append(f"rank {rank}: no Gluon inboxes for the plain all-reduce")
        for tokens in (1, 8, 128, 1024):  # 1024 rows at hidden 2048 are 4 MiB
            gen = torch.Generator(device=device).manual_seed(1000 + tokens)
            shape = (tokens, HIDDEN)
            xs = [
                torch.randn(shape, generator=gen, dtype=torch.bfloat16, device=device)
                for _ in range(WORLD_SIZE)
            ]
            ref = xs[0].float() + xs[1].float() + xs[2].float() + xs[3].float()
            out = tensor_model_parallel_all_reduce(xs[rank])
            bad = out.view(torch.int16) != ref.to(torch.bfloat16).view(torch.int16)
            if bool(bad.any()):
                failures.append(f"rank {rank} plain dispatch tokens {tokens}: wrong")

        # all_reduce_rmsnorm_per_token_quant: up to 128 rows where the Lamport
        # one-shot could take the call, up to 4 MiB where it cannot (6152).
        ar = (tp.cpu_group, *tp.device_communicator.gluon_ar_plain)
        for (tokens, hidden), quant in itertools.product(
            ((1, PT_HIDDEN), (128, PT_HIDDEN), (340, 6152)), (True, False)
        ):
            xs, res, w = _pt_inputs(tokens, hidden, 500 + tokens, device)
            with patch.object(
                plain,
                "all_reduce_rmsnorm_per_token_quant",
                wraps=all_reduce_rmsnorm_per_token_quant,
            ) as op:
                if quant:
                    outs = tensor_model_parallel_fused_allreduce_rmsnorm_quant(
                        xs[rank], res, w, EPS, quant_type="per_token", gemma_norm=True
                    )
                else:
                    outs = tensor_model_parallel_fused_allreduce_rmsnorm(
                        xs[rank], res, w, EPS, gemma_norm=True
                    )
            name = f"rank {rank} per-token dispatch tokens {tokens} quant {quant}"
            if not op.called or _pt_wrong(outs, xs, res, w, quant, rank, ar):
                failures.append(f"{name}: another path or wrong")
    finally:
        aiter.destroy_dist_env()
    return failures


def test_fused_allreduce_rmsnorm_dispatch():
    ensure_spawn_method()
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(processes=WORLD_SIZE) as pool:
        results = [
            pool.apply_async(_dispatch_worker, (rank, init_method))
            for rank in range(WORLD_SIZE)
        ]
        failures = [f for r in results for f in r.get(timeout=1800)]
    assert not failures, "\n".join(failures[:20])


if __name__ == "__main__":  # CI runs the multi-GPU tests as scripts
    if SKIP:
        logger.warning("test_all_reduce_rmsnorm_quant skipped: needs 4 gfx1250 GPUs")
    else:
        test_all_reduce_rmsnorm_quant()
        test_all_reduce_rmsnorm_per_token_quant()
        test_fused_allreduce_rmsnorm_dispatch()
