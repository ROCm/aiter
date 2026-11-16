# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check CustomAllreduce.custom_dcp_merge, the gfx1250 decode context parallel
merge: every rank holds o [B, H, 128] bf16 and lse [B, H] fp32 over its own KV
shard, and gets its H / n heads merged over all the shards by their LSE.

Against an fp32 torch reference on every rank (each rank rebuilds every rank's
inputs from their seeds), for:
- token counts from 1 up to what the LL scratch holds, with rows whose LSE is
  -inf on some ranks or on all of them, owned_counts with zeros, and return_lse;
  o and lse contiguous, and strided (o a [..., :128] view of [B, H, 192], lse a
  transposed view);
- no tokens (B = 0), followed by a regular call;
- a CUDA graph replayed with new inputs and new owned_counts;
- calls interleaved with LL all-reduces and all-gathers, which share its
  scratch and epochs, with one rank (the same choice on every rank) arriving
  late each round.

Run with: python3 op_tests/multigpu_tests/test_dcp_merge.py -t 4
"""

import argparse
import random
from multiprocessing import freeze_support

import torch
import torch.distributed as dist

from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

D = 128


def _inputs(src, b, h, seed, device):
    """Rank src's o and lse for case `seed`; the same on every rank."""
    g = torch.Generator(device=device).manual_seed(1000003 * src + 7919 * seed + b)
    o = torch.randn((b, h, D), generator=g, device=device).to(torch.bfloat16)
    lse = torch.randn((b, h), generator=g, device=device) * 2.0 + 4.0
    rows = torch.arange(b, device=device)
    lse[rows % 7 == src % 7] = float("-inf")  # no KV for these rows on this rank
    if b > 1:
        lse[1] = float("-inf")  # no KV for row 1 on any rank
    if b > 2:
        lse[2, 0] = float("nan")  # counts as no KV
    return o, lse


def _owned(src, b, device, seed=0):
    """owned_counts of rank src: 0 for every fifth row, offset by the rank and
    the seed."""
    rows = torch.arange(b, device=device)
    return ((rows + src + seed) % 5 != 0).to(torch.int32) * 3


def _strided(o, lse):
    """The same values as o and lse, as an o [..., :128] view of [B, H, 192]
    and a transposed lse view."""
    b, h, _ = o.shape
    wide = torch.full((b, h, D + 64), float("nan"), dtype=o.dtype, device=o.device)
    wide[..., :D] = o
    lse_t = torch.empty((h, b), dtype=lse.dtype, device=lse.device).t()
    lse_t.copy_(lse)
    return wide[..., :D], lse_t


def _reference(world, rank, b, h, seed, device, with_owned, owned_seed=0):
    hl = h // world
    os_, ls = [], []
    for s in range(world):
        o, lse = _inputs(s, b, h, seed, device)
        if with_owned:
            owned = _owned(s, b, device, owned_seed)
            lse = lse.masked_fill(owned[:, None] == 0, float("-inf"))
        os_.append(o[:, rank * hl : (rank + 1) * hl].float())
        ls.append(lse[:, rank * hl : (rank + 1) * hl])
    o, lse = torch.stack(os_), torch.stack(ls)
    lse = torch.where(torch.isnan(lse) | (lse == float("inf")), float("-inf"), lse)
    m = lse.max(0).values
    m = torch.where(m == float("-inf"), torch.zeros_like(m), m)
    g = torch.log(torch.exp(lse - m).sum(0)) + m
    f = torch.exp(lse - g)
    f = torch.where(torch.isnan(f), torch.zeros_like(f), f)
    v = torch.where(f[..., None] == 0, torch.zeros_like(o), o)
    return (v * f[..., None]).sum(0), g


def _check(out, out_lse, ref, ref_lse, context):
    # One bf16 ulp of the row's largest value: an element that nearly cancels
    # has no meaningful element-relative error.
    tol = ref.abs().amax(dim=-1, keepdim=True) * 2.0**-8 + 1e-30
    err = (out.float() - ref).abs()
    assert bool((err <= tol).all()), f"{context}: {int((err > tol).sum())} elements off"
    if out_lse is not None:
        torch.testing.assert_close(
            out_lse, ref_lse, rtol=1e-5, atol=1e-5, equal_nan=True, msg=context
        )


def _merge_worker(
    rank, world, heads, sizes, iterations, init_method, device_index=None
):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    local = rank if device_index is None else device_index
    torch.cuda.set_device(local)
    device = torch.device("cuda", local)
    dist.init_process_group(
        "gloo", world_size=world, rank=rank, init_method=init_method
    )
    ca = CustomAllreduce(dist.group.WORLD, device)
    try:
        assert not ca.disabled, "custom allreduce is disabled"
        o0, l0 = _inputs(rank, 1, heads, 0, device)
        if not ca.should_custom_dcp_merge(o0, l0):
            if rank == 0:
                print("custom_dcp_merge does not run here (gfx1250 LL only): skipped")
            return
        hl = heads // world
        max_b = ca._dcp_merge_max_rows // hl

        # sizes, edge rows, owned_counts and return_lse, contiguous and strided
        for b in [x for x in sizes if x <= max_b] + [max_b]:
            for with_owned in (False, True):
                for strided in (False, True):
                    o, lse = _inputs(rank, b, heads, 1, device)
                    if strided:
                        o, lse = _strided(o, lse)
                        assert ca.should_custom_dcp_merge(o, lse)
                    owned = _owned(rank, b, device) if with_owned else None
                    out, out_lse = ca.custom_dcp_merge(o, lse, owned, return_lse=True)
                    ref, ref_lse = _reference(
                        world, rank, b, heads, 1, device, with_owned
                    )
                    context = f"{rank=} {b=} {with_owned=} {strided=}"
                    _check(out, out_lse, ref, ref_lse, context)
        if rank == 0:
            print(f"TP{world} H{heads}: sizes {sizes} and {max_b} tokens pass")

        # no tokens, then a regular call (the epochs stay in step)
        o, lse = _inputs(rank, 4, heads, 3, device)
        out, out_lse = ca.custom_dcp_merge(o[:0], lse[:0], return_lse=True)
        assert out.shape == (0, hl, D) and out_lse.shape == (0, hl), f"{rank=} B=0"
        out = ca.custom_dcp_merge(o, lse)
        ref, _ = _reference(world, rank, 4, heads, 3, device, False)
        _check(out, None, ref, None, f"{rank=} after B=0")
        if rank == 0:
            print(f"TP{world}: B = 0 passes")

        # a CUDA graph, replayed with new inputs and new owned_counts
        b = sizes[len(sizes) // 2]
        o, lse = _inputs(rank, b, heads, 2, device)
        owned = _owned(rank, b, device)
        ca.custom_dcp_merge(o, lse)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with ca.capture(), torch.cuda.graph(graph, stream=stream):
            gout = ca.custom_dcp_merge(o, lse)
            gown = ca.custom_dcp_merge(o, lse, owned)
        torch.cuda.current_stream().wait_stream(stream)
        for it in range(iterations):
            no, nl = _inputs(rank, b, heads, 100 + it, device)
            o.copy_(no)
            lse.copy_(nl)
            owned.copy_(_owned(rank, b, device, it))
            gout.fill_(float("nan"))
            gown.fill_(float("nan"))
            graph.replay()
            ref, _ = _reference(world, rank, b, heads, 100 + it, device, False)
            _check(gout, None, ref, None, f"{rank=} graph {it=}")
            ref, _ = _reference(world, rank, b, heads, 100 + it, device, True, it)
            _check(gown, None, ref, None, f"{rank=} graph owned {it=}")
        if rank == 0:
            print(f"TP{world}: {iterations} graph replays pass")

        # interleaved with the LL all-reduce and all-gather (shared scratch and
        # epochs); each round one rank, the same on every rank, arrives late
        cases = [(b, s) for s, b in enumerate(sizes)]
        for it in range(iterations):
            for b, seed in cases:
                o, lse = _inputs(rank, b, heads, 200 + it + seed, device)
                x = torch.full(
                    (b, 7168), float(rank + 1), dtype=torch.bfloat16, device=device
                )
                ar = ca.custom_all_reduce(x)
                ag = ca.custom_all_gather(x[:1].contiguous(), 0)
                if rank == random.Random(1009 * it + seed).randrange(world):
                    torch.cuda._sleep(2_000_000)
                out = ca.custom_dcp_merge(o, lse)
                ref, _ = _reference(
                    world, rank, b, heads, 200 + it + seed, device, False
                )
                _check(out, None, ref, None, f"{rank=} interleaved {it=} {b=}")
                expected = float(world * (world + 1) // 2)
                assert ar is not None and bool(
                    (ar == expected).all()
                ), f"{rank=} AR {it=}"
                assert ag is not None and [int(v) for v in ag[:, 0].tolist()] == list(
                    range(1, world + 1)
                ), f"{rank=} AG {it=}"
        if rank == 0:
            print(f"TP{world}: {iterations} interleaved rounds pass")
        dist.barrier()
    finally:
        ca.close()
        dist.destroy_process_group()


def test_dcp_merge(
    world, heads=96, sizes=(1, 3, 8, 54, 115, 168, 320, 640), iterations=20
):
    torch.multiprocessing.spawn(
        _merge_worker,
        args=(
            world,
            heads,
            list(sizes),
            iterations,
            get_distributed_init_method(get_ip(), get_open_port()),
        ),
        nprocs=world,
        join=True,
    )


parser = argparse.ArgumentParser(description="gfx1250 DCP merge (custom_dcp_merge)")
parser.add_argument("-t", "--tp-size", type=int, choices=[2, 4, 6, 8], default=8)
parser.add_argument(
    "--heads", type=int, default=96, help="query heads a rank attends (default 96)"
)
parser.add_argument("--iterations", type=int, default=20)

if __name__ == "__main__":
    freeze_support()
    args = parser.parse_args()
    test_dcp_merge(args.tp_size, args.heads, iterations=args.iterations)
