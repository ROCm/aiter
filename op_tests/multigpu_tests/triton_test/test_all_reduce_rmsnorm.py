# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import multiprocessing as mp
import os

import pytest
import torch
import torch.distributed as dist

import aiter
from aiter.dist.communication_op import (
    tensor_model_parallel_fused_allreduce_rmsnorm,
    tensor_model_parallel_fused_allreduce_rmsnorm_quant,
)
from aiter.dist.parallel_state import get_tp_group
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.ops.triton.comms.all_reduce import (
    _LINES,
    all_reduce,
    all_reduce_free,
    all_reduce_init,
    all_reduce_rmsnorm,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import ensure_spawn_method

WORLD_SIZE = 4
EPS = 1e-6
TOKENS = (1, 2, 3, 4, 7, 16, 33, 64, 128, 256)
HIDDEN = 6144  # the bits of the split path; other sizes are near an fp32 reference
OTHER_HIDDEN = (2048, 4096, 5120, 6152, 7168, 8192, 16384)

SKIP = torch.cuda.device_count() < WORLD_SIZE or arch_info.get_arch() != "gfx1250"
pytestmark = pytest.mark.skipif(SKIP, reason="needs 4 gfx1250 GPUs")


def _inputs(tokens, seed, device, edge=False, hidden=HIDDEN):
    """The x of each rank, and the residual and weight of all ranks."""
    gen = torch.Generator(device=device).manual_seed(seed)
    shape = (tokens, hidden)
    xs = [
        torch.randn(shape, generator=gen, dtype=torch.bfloat16, device=device) * 0.5
        for _ in range(WORLD_SIZE)
    ]
    res = torch.randn(shape, generator=gen, dtype=torch.bfloat16, device=device) * 2
    w = torch.randn(hidden, generator=gen, dtype=torch.bfloat16, device=device)
    if edge and tokens >= 4:
        for x in xs:
            x[0].zero_()  # a zero row: the FP8 scale is 0
            x[1].mul_(3000.0)
            x[3].mul_(1e-12)
        res[0].zero_()
        total = xs[0][2].float() + xs[1][2].float() + xs[2][2].float()
        res[2] = (-(total + xs[3][2].float())).to(torch.bfloat16)  # cancellation
        res[3].mul_(1e-12)
    return xs, res, w * 0.1


def _split(x, residual, weight, quant, gemma, group, inboxes, calls):
    """The split path: plain Gluon all-reduce, then the HIP add_rmsnorm(_quant)."""
    ar = all_reduce(x, group, inboxes, calls)
    res_out = torch.empty_like(residual)
    if quant:
        out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
        scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
        aiter.add_rmsnorm_quant(
            out,
            ar,
            residual,
            res_out,
            scale,
            weight,
            EPS,
            gemma_norm=gemma,
            round_before_quant=True,
        )
        return out, res_out, scale
    out = torch.empty_like(residual)
    aiter.rmsnorm2d_fwd_with_add(out, ar, residual, res_out, weight, EPS, gemma, 0)
    return out, res_out


def _far(outs, xs, residual, weight, gemma):
    """Number of values off the fp32 reference: residual_out is exact (the bf16 sum in
    rank order plus the residual), out is 1 bf16 ulp or 1 FP8 code from the reference
    (the row sum of squares has another order), the scale 1 bf16 ulp of the row max.
    Values that the kernel did not write are NaN (sentinel)."""
    h = sum(x.float() for x in xs).bfloat16().float() + residual.float()
    rstd = torch.rsqrt(h.double().square().mean(-1, keepdim=True) + EPS).float()
    y = (h * rstd * (weight.float() + gemma)).bfloat16()
    refs = [y, h.bfloat16()]
    if len(outs) == 3:
        scale = y.float().abs().amax(-1, keepdim=True) * (1.0 / 448.0)
        refs = [(y.float() / scale).clamp(-448, 448).to(outs[0].dtype), refs[1], scale]
    bad = sum(int((~o.float().isfinite()).sum()) for o in outs)
    bad += int((outs[1] != refs[1]).sum())
    if len(outs) == 3:
        bad += int((~torch.isclose(outs[2], refs[2], rtol=2**-7, atol=0)).sum())
    bits = torch.int8 if len(outs) == 3 else torch.int16
    diff = outs[0].view(bits).int() - refs[0].view(bits).int()
    return bad + int((diff.abs() > 1).sum())


def _wrong(outs, xs, res, w, quant, gemma, rank, *ar):
    """Values off the split path's bits (HIDDEN) or off the fp32 reference."""
    if res.shape[1] == HIDDEN:
        return _bad(outs, _split(xs[rank], res, w, quant, gemma, *ar))
    return _far(outs, xs, res, w, gemma)


def _bad(outs, refs):
    """Number of elements whose bits differ."""
    return sum(
        int((o.view(torch.uint8) != r.view(torch.uint8)).sum())
        for o, r in zip(outs, refs)
    )


def _same_on_ranks(outs, group):
    for o in outs:
        o = o.contiguous().view(torch.uint8)
        all_o = [torch.empty_like(o) for _ in range(WORLD_SIZE)]
        dist.all_gather(all_o, o, group=group)
        if any(not torch.equal(a, o) for a in all_o):
            return False
    return True


def _worker(rank, init_method):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    os.environ["AITER_GLUON_FUSED_AR"] = "1"
    aiter.init_dist_env(WORLD_SIZE, rank, distributed_init_method=init_method)
    tp = get_tp_group()
    group = tp.cpu_group
    inboxes, calls = all_reduce_init(group, 4 << 20)
    failures = []

    def check(name, tokens, seed, quant, gemma=True, edge=False, hidden=HIDDEN):
        xs, res, w = _inputs(tokens, seed, device, edge, hidden)
        ar = (group, inboxes, calls)
        outs = all_reduce_rmsnorm(xs[rank], res, w, EPS, gemma, quant, *ar)
        bad = _wrong(outs, xs, res, w, quant, gemma, rank, *ar)
        if bad:
            failures.append(f"rank {rank} {name}: {bad} values differ")
        if not _same_on_ranks(outs, tp.device_group):
            failures.append(f"rank {rank} {name}: the ranks differ")

    try:
        for quant in (True, False):
            for tokens in TOKENS:
                check(f"tokens {tokens} quant {quant}", tokens, tokens, quant)
            check(f"edge rows quant {quant}", 16, 7, quant, edge=True)
            check(f"no gemma quant {quant}", 16, 8, quant, gemma=False)
            for hidden in OTHER_HIDDEN:
                for tokens in (1, 7, 64):
                    name = f"hidden {hidden} tokens {tokens} quant {quant}"
                    check(name, tokens, hidden + tokens, quant, hidden=hidden)

        # Plain calls with a partial last chunk: the 2 chunks of a slab then have
        # other call numbers.
        for i, numel in enumerate((3 * 6 * _LINES + 20, 17 * 6 * _LINES, 600 * 4)):
            all_reduce(
                torch.randn(numel, dtype=torch.bfloat16, device=device),
                group,
                inboxes,
                calls,
            )
            check(f"after partial call {i}", 4 + i, 300 + i, i % 2 == 0)

        # The call numbers wrap from 2^30 to 1.
        calls[:-1].fill_(2**30 - 2)
        for i in range(4):
            check(f"counter wrap {i}", 8, 400 + i, i % 2 == 0)

        # Mixed calls in a HIP graph, replayed with new inputs into NaN outputs.
        cases = [(1, False, HIDDEN), (16, True, 4096), (4, False, 7168)]
        cases += [(64, True, HIDDEN), (33, True, 6152), (3, False, 2048)]
        xs = [
            torch.zeros((t, h), dtype=torch.bfloat16, device=device)
            for t, _, h in cases
        ]
        ress = [torch.zeros_like(x) for x in xs]
        ws = [torch.zeros(h, dtype=torch.bfloat16, device=device) for *_, h in cases]
        odd = torch.zeros(3 * 6 * _LINES + 20, dtype=torch.bfloat16, device=device)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = []
            for (t, quant, h), x, res, w in zip(cases, xs, ress, ws):
                outs.append(
                    all_reduce_rmsnorm(
                        x, res, w, EPS, True, quant, group, inboxes, calls
                    )
                )
                all_reduce(odd, group, inboxes, calls)
        for rep in range(10):
            all_xs = []
            for i, ((t, quant, h), x, res, w) in enumerate(zip(cases, xs, ress, ws)):
                new_xs, new_res, new_w = _inputs(t, 1000 * rep + i, device, hidden=h)
                all_xs.append(new_xs)
                x.copy_(new_xs[rank])
                res.copy_(new_res)
                w.copy_(new_w)
            for o in outs:
                for v in o:
                    v.view(torch.uint8).fill_(255)  # NaN
            graph.replay()
            torch.cuda.synchronize()
            for (t, quant, h), res, w, o, all_x in zip(cases, ress, ws, outs, all_xs):
                ar = (group, inboxes, calls)
                if _wrong(o, all_x, res, w, quant, True, rank, *ar):
                    failures.append(
                        f"rank {rank}: graph replay {rep} tokens {t} hidden {h} is wrong"
                    )

        # The public ops pick all_reduce_rmsnorm and give the bits of the path
        # without it.
        comm = tp.device_communicator
        if not comm.gluon_ar_norm:
            failures.append(
                f"rank {rank}: all_reduce_rmsnorm is off in the communicator"
            )
        else:
            for tokens in (1, 16, 64):
                xs, res, w = _inputs(tokens, 500 + tokens, device)
                x = xs[rank]
                for quant in (True, False):
                    if quant:
                        op = tensor_model_parallel_fused_allreduce_rmsnorm_quant
                        kw = {"quant_type": "per_token", "gemma_norm": True}
                    else:
                        op = tensor_model_parallel_fused_allreduce_rmsnorm
                        kw = {"gemma_norm": True}
                    fused = op(x, res, w, EPS, **kw)
                    comm.gluon_ar_norm = False
                    try:
                        split = op(x, res, w, EPS, **kw)
                    finally:
                        comm.gluon_ar_norm = True
                    if _bad(fused, split):
                        failures.append(
                            f"rank {rank}: public op tokens {tokens} quant {quant} differs"
                        )
            if comm.gluon_ar_plain[1][-1].item():
                failures.append(f"rank {rank}: communicator polls timed out")
        if calls[-1].item():
            failures.append(f"rank {rank}: {calls[-1].item()} polls timed out")
        return failures
    finally:
        all_reduce_free(group, inboxes)
        aiter.destroy_dist_env()


def test_all_reduce_rmsnorm():
    ensure_spawn_method()
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(processes=WORLD_SIZE) as pool:
        results = [
            pool.apply_async(_worker, (rank, init_method)) for rank in range(WORLD_SIZE)
        ]
        failures = [f for r in results for f in r.get(timeout=1800)]
    assert not failures, "\n".join(failures)


if __name__ == "__main__":  # CI runs the multi-GPU tests as scripts
    test_all_reduce_rmsnorm()
