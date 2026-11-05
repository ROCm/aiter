# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness of the FlyDSL intra-node EP dispatch / combine over HIP IPC
symmetric buffers (no mori).

Each rank dispatches MXFP8 rows (fp8 + e8m0 scales) with its top-k ids and
weights, then combines bf16 rows that are a function of what it received.

* dispatch: the received count equals the number of (source token, this rank)
  pairs with at least one valid expert here; every received row, its scales,
  weights and ids equal the source token's (looked up through the reported
  source position).
* combine: equal, bit for bit (+0 / -0 aside), to an f32 sum over the token's
  distinct destination ranks in top-k slot order, rounded to bf16. Negative ids
  are skipped, duplicates of a rank count once.
* the same pair captured in a CUDA graph and replayed several times gives the
  same result (launch epochs advance on the device).

Routings: random, all tokens to one rank, one rank receiving nothing, all
top-k slots on one rank, masked (-1) slots, ranks with no tokens.

    python op_tests/multigpu_tests/test_flydsl_ep_intranode.py [--world 8]
"""

import argparse
import os
import socket
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROUTINGS = ("random", "onerank", "skip0", "dupe", "neg", "empty")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_ids(routing, T, K, E, epr, world, rank, g):
    if T == 0:
        return torch.zeros(0, K, dtype=torch.int32)
    if routing in ("random", "empty", "neg"):
        ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])
        if routing == "neg":
            ids[torch.rand(T, K, generator=g) < 0.25] = -1
            ids[0] = -1
    elif routing == "onerank":
        dst = world - 1
        ids = torch.stack(
            [torch.randperm(epr, generator=g)[:K] + dst * epr for _ in range(T)]
        )
    elif routing == "skip0":
        ids = torch.stack(
            [torch.randperm(E - epr, generator=g)[:K] + epr for _ in range(T)]
        )
    elif routing == "dupe":
        ids = torch.stack(
            [
                torch.randperm(epr, generator=g)[:K]
                + int(torch.randint(0, world, (1,), generator=g)) * epr
                for _ in range(T)
            ]
        )
    else:
        raise ValueError(routing)
    return ids.to(torch.int32)


def _bitdiff(x, y):
    d = (x.view(torch.int16) != y.view(torch.int16)) & ~((x == 0) & (y == 0))
    return int(d.sum().item())


def _reference(x_fp8, ids, epr):
    """f32 sum over distinct destination ranks in top-k order -> bf16."""
    T, K = ids.shape
    acc = torch.zeros(T, x_fp8.shape[1], dtype=torch.float32, device=x_fp8.device)
    dest = torch.where(ids >= 0, torch.div(ids, epr, rounding_mode="floor"), -1)
    xf = x_fp8.float()
    for k in range(K):
        first = dest[:, k] >= 0
        for j in range(k):
            first &= dest[:, j] != dest[:, k]
        scale = 1.0 + dest[:, k].clamp(min=0).float() / 8.0
        contrib = (xf * scale[:, None]).to(torch.bfloat16).float()
        acc += torch.where(first[:, None], contrib, torch.zeros_like(contrib))
    return acc.to(torch.bfloat16)


def _worker(rank, world, port, shapes, tokens, routings, replays, result_q):
    from aiter.ops.flydsl.kernels.flydsl_dispatch_combine_intranode_op import (
        FlyDSLDispatchCombineConfig,
        FlyDSLDispatchCombineIntraNodeOp,
    )

    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl", rank=rank, world_size=world, device_id=dev
    )
    group = dist.new_group(list(range(world)), backend="gloo")
    failures = []
    fp8 = torch.float8_e4m3fn
    try:
        for hidden, E, K, max_tok in shapes:
            epr = E // world
            sd = hidden // 32
            op = FlyDSLDispatchCombineIntraNodeOp(
                FlyDSLDispatchCombineConfig(
                    rank=rank,
                    world_size=world,
                    hidden_dim=hidden,
                    max_num_inp_token_per_rank=max_tok,
                    num_experts_per_rank=epr,
                    num_experts_per_token=K,
                    dispatch_dtype=fp8,
                    combine_dtype=torch.bfloat16,
                    scale_dim=sd,
                    scale_type_size=1,
                ),
                group=group,
            )
            g = torch.Generator().manual_seed(1234 + rank)
            for routing in routings:
                for T0 in tokens:
                    T = 0 if (routing == "empty" and rank % 2 == 1) else T0
                    ids_c = _make_ids(routing, T, K, E, epr, world, rank, g)
                    ids = ids_c.to(dev)
                    w = torch.rand(T, K, generator=g).to(dev)
                    x = (
                        (torch.randn(T, hidden, generator=g) * 8)
                        .clamp(-448, 448)
                        .to(fp8)
                        .to(dev)
                    )
                    sc = torch.randint(
                        100, 140, (T, sd), generator=g, dtype=torch.uint8
                    ).to(dev)
                    all_rows = [None] * world
                    dist.all_gather_object(
                        all_rows,
                        (ids_c, x.view(torch.uint8).cpu(), sc.cpu(), w.cpu()),
                        group=group,
                    )
                    n_in = max(1, world * max(int(r[0].shape[0]) for r in all_rows))

                    def step(op=op, x=x, w=w, sc=sc, ids=ids, n_in=n_in):
                        out = op.dispatch(x, w, sc, ids)
                        rows = min(out[0].shape[0], n_in)
                        cin = (out[0][:rows].float() * (1.0 + rank / 8.0)).to(
                            torch.bfloat16
                        )
                        return out, op.combine(cin, None, ids)[0]

                    out, res = step()
                    torch.cuda.synchronize()
                    tag = f"rank{rank} hidden={hidden} E={E} K={K} {routing} T={T0}"
                    # dispatch
                    n = int(out[4].reshape(-1)[0].item())
                    expect = sum(
                        int(
                            ((r[0].long() >= 0) & (r[0].long() // epr == rank))
                            .any(1)
                            .sum()
                        )
                        for r in all_rows
                        if r[0].numel()
                    )
                    if n != expect:
                        failures.append(f"{tag}: received {n}, expected {expect}")
                    src = op.get_dispatch_src_token_pos()[:n].long().cpu()
                    rx = out[0][:n].view(torch.uint8).cpu()
                    rs = out[2][:n].reshape(n, -1)[:, :sd].contiguous().cpu()
                    rw, ri = out[1][:n].cpu(), out[3][:n].cpu()
                    for i in range(n):
                        s_r, s_t = int(src[i]) // max_tok, int(src[i]) % max_tok
                        ids_s, x_s, sc_s, w_s = all_rows[s_r]
                        if not (
                            torch.equal(rx[i], x_s[s_t])
                            and torch.equal(rs[i].view(torch.uint8), sc_s[s_t])
                            and torch.equal(rw[i], w_s[s_t])
                            and torch.equal(ri[i], ids_s[s_t])
                        ):
                            failures.append(f"{tag}: received row {i} differs")
                            break
                    # combine
                    ref = _reference(x, ids, epr)
                    bad = _bitdiff(res[:T], ref) if T else 0
                    if bad:
                        failures.append(f"{tag}: combine {bad} elements differ")
                    # CUDA graph capture + replays
                    dist.barrier(group=group)
                    s = torch.cuda.Stream()
                    with torch.cuda.stream(s):
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=s):
                            _, res_g = step()
                    torch.cuda.synchronize()
                    dist.barrier(group=group)
                    for _ in range(replays):
                        graph.replay()
                    torch.cuda.synchronize()
                    bad = _bitdiff(res_g[:T], ref) if T else 0
                    if bad:
                        failures.append(f"{tag}: graph replay {bad} elements differ")
                    del graph
                    dist.barrier(group=group)
            op._heap.close()
    except Exception as e:  # noqa: BLE001
        failures.append(f"rank{rank}: {type(e).__name__}: {e}")
    result_q.put((rank, failures))
    dist.barrier(group=group)
    dist.destroy_process_group()


def run(world=None, shapes=None, tokens=None, routings=ROUTINGS, replays=3):
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() not in ("gfx942", "gfx950"):
        print(
            f"skip: FlyDSL intra-node EP kernels target gfx942/gfx950, not {get_gfx()}"
        )
        return True
    n_dev = torch.cuda.device_count()
    world = min(8, n_dev) if world is None else world
    if world < 2 or world > n_dev:
        print(f"skip: needs 2..8 GPUs, have {n_dev}")
        return True
    # (hidden, experts, top-k, max tokens per rank)
    shapes = shapes or [(5120, 48 * world, 6, 4096), (2048, 16 * world, 3, 1024)]
    tokens = tokens or [1, 7, 32, 192, 1000]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(
            target=_worker,
            args=(r, world, port, shapes, tokens, routings, replays, q),
        )
        for r in range(world)
    ]
    for p in procs:
        p.start()
    results = dict(q.get() for _ in procs)
    for p in procs:
        p.join()
    failures = [f for r in sorted(results) for f in results[r]]
    for f in failures[:50]:
        print("FAIL", f)
    print(
        f"flydsl intranode EP: world={world} shapes={shapes} tokens={tokens} "
        f"routings={list(routings)}: {'FAILED' if failures else 'passed'}"
    )
    return not failures


def test_flydsl_ep_intranode():
    assert run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--world", type=int, default=None)
    ap.add_argument("--tokens", type=str, default=None, help="e.g. 1,32,192")
    ap.add_argument("--routings", type=str, default=",".join(ROUTINGS))
    ap.add_argument("--replays", type=int, default=3)
    a = ap.parse_args()
    ok = run(
        world=a.world,
        tokens=[int(t) for t in a.tokens.split(",")] if a.tokens else None,
        routings=a.routings.split(","),
        replays=a.replays,
    )
    sys.exit(0 if ok else 1)
