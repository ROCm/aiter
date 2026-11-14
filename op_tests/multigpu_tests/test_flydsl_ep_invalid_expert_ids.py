# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL intra-node EP dispatch / combine with invalid expert ids.

A top-k slot whose expert id is negative (-1 padding, other negatives) or
>= world * experts_per_rank must not be dispatched, must not count as an
earlier slot's destination rank, and must add nothing to the combine result.
Each rank routes bf16 rows through ``FlyDSLDispatchCombineIntraNodeOp`` and
checks, against a reference that skips those slots:

* dispatch: the received count equals the number of (source token, this rank)
  pairs with at least one valid expert here; every received row and its ids
  and weights equal the source token's (found through the source position).
* ``enable_std_moe=True``: ``packed_recv_count[e]`` equals the number of slots
  routed to local expert ``e``; combine (expert-major partials, each local
  expert's rows scaled by its own factor) equals the weighted sum over the
  valid slots only.
* ``enable_std_moe=False``: combine of a rank-dependent value equals the sum
  over the token's distinct valid destination ranks.

Routings mix -1 padding, other negative ids, ids >= world * experts_per_rank (including
the first out-of-range id) and tokens with no valid slot.

    python op_tests/multigpu_tests/test_flydsl_ep_invalid_expert_ids.py [--world 8]
"""

import argparse
import queue
import socket
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROUTINGS = ("valid", "pad", "neg", "oob", "mixed")
MODES = ("std_moe", "plain")


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _make_ids(routing, T, K, E, g):
    """Distinct valid ids per token, then some slots replaced by invalid ids."""
    if T == 0:
        return torch.zeros(0, K, dtype=torch.int64)
    ids = torch.stack([torch.randperm(E, generator=g)[:K] for _ in range(T)])
    if routing == "valid":
        return ids
    hit = torch.rand(T, K, generator=g) < 0.3
    if routing == "pad":
        bad = torch.full((T, K), -1)
    elif routing == "neg":
        bad = torch.where(
            torch.rand(T, K, generator=g) < 0.5,
            torch.full((T, K), -1),
            -torch.randint(2, 1 << 20, (T, K), generator=g),
        )
    elif routing == "oob":
        bad = E + torch.randint(0, 3 * E, (T, K), generator=g)
        bad[:, 0] = E  # first out-of-range id
    elif routing == "mixed":
        bad = torch.where(
            torch.rand(T, K, generator=g) < 0.5,
            torch.full((T, K), -1),
            E + torch.randint(0, 1 << 20, (T, K), generator=g),
        )
        ids[0] = bad[0]  # a token with no valid slot
    else:
        raise ValueError(routing)
    return torch.where(hit, bad, ids)


def _expert_scale(local_e, epr):
    return 1.0 + local_e.float() / epr


def _reference(x, ids, w, E, epr, world, std_moe):
    """Combine result with invalid slots skipped (f32 sums, bf16 per rank)."""
    T = ids.shape[0]
    xf = x.float()
    acc = torch.zeros(T, x.shape[1], dtype=torch.float32)
    vld = (ids >= 0) & (ids < E)
    dest = torch.where(vld, torch.div(ids, epr, rounding_mode="floor"), -1)
    for r in range(world):
        on_r = dest == r
        if std_moe:
            coef = torch.where(on_r, w * _expert_scale(ids % epr, epr), 0.0).sum(1)
            part = (xf * coef[:, None]).to(torch.bfloat16).float()
        else:
            part = (xf * (1.0 + r / 8.0)).to(torch.bfloat16).float()
        acc += torch.where(on_r.any(1)[:, None], part, torch.zeros_like(part))
    return acc.to(torch.bfloat16)


def _close(got, ref):
    if got.numel() == 0:
        return True
    return torch.allclose(got.float(), ref.float(), rtol=2e-2, atol=2e-2)


def _worker(rank, world, port, cfg, result_q):
    import mori

    from aiter.ops.flydsl.kernels.flydsl_dispatch_combine_intranode_op import (
        FlyDSLDispatchCombineConfig,
        FlyDSLDispatchCombineIntraNodeOp,
    )

    hidden, epr, K, max_tok, tokens, routings, modes = cfg
    E = epr * world
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group(
        "cpu:gloo,cuda:nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
        device_id=dev,
    )
    torch._C._distributed_c10d._register_process_group("default", dist.group.WORLD)
    mori.shmem.shmem_torch_process_group_init("default")
    group = dist.new_group(list(range(world)), backend="gloo")
    failures = []
    try:
        for mode in modes:
            std_moe = mode == "std_moe"
            op = FlyDSLDispatchCombineIntraNodeOp(
                FlyDSLDispatchCombineConfig(
                    rank=rank,
                    world_size=world,
                    hidden_dim=hidden,
                    max_num_inp_token_per_rank=max_tok,
                    num_experts_per_rank=epr,
                    num_experts_per_token=K,
                    data_type=torch.bfloat16,
                    enable_std_moe=std_moe,
                )
            )
            mr = op.cfg.max_recv
            prx = (
                torch.empty(epr * mr, hidden, dtype=torch.bfloat16, device=dev)
                if std_moe
                else None
            )
            g = torch.Generator().manual_seed(4321 + rank)
            for routing in routings:
                for T in tokens:
                    ids_c = _make_ids(routing, T, K, E, g)
                    w_c = torch.rand(T, K, generator=g)
                    x_c = torch.randn(T, hidden, generator=g).to(torch.bfloat16)
                    ids = ids_c.to(torch.int32).to(dev)
                    all_rows = [None] * world
                    dist.all_gather_object(all_rows, (ids_c, x_c, w_c), group=group)
                    tag = f"rank{rank} {mode} {routing} T={T}"

                    out = op.dispatch(x_c.to(dev), w_c.to(dev), None, ids, prx)
                    torch.cuda.synchronize()
                    # Read counts before combine.
                    n = int(out[4].reshape(-1)[0].item())
                    src = op.get_dispatch_src_token_pos()[:n].long().cpu()
                    in_range = [(r[0] >= 0) & (r[0] < E) for r in all_rows]
                    expect = sum(
                        int((v & (r[0] // epr == rank)).any(1).sum())
                        for r, v in zip(all_rows, in_range)
                    )
                    if n != expect:
                        failures.append(f"{tag}: received {n}, expected {expect}")
                    elif n:
                        s_r, s_t = src // max_tok, src % max_tok
                        got_src = sorted(zip(s_r.tolist(), s_t.tolist()))
                        want_src = sorted(
                            (r_i, t)
                            for r_i, (r, v) in enumerate(zip(all_rows, in_range))
                            for t in range(r[0].shape[0])
                            if bool((v[t] & (r[0][t] // epr == rank)).any())
                        )
                        if got_src != want_src:
                            failures.append(f"{tag}: received source set differs")
                        else:
                            rows = torch.stack(
                                [all_rows[a][1][b] for a, b in zip(s_r, s_t)]
                            )
                            rids = torch.stack(
                                [all_rows[a][0][b] for a, b in zip(s_r, s_t)]
                            )
                            rwts = torch.stack(
                                [all_rows[a][2][b] for a, b in zip(s_r, s_t)]
                            )
                            if not torch.equal(out[0][:n].cpu(), rows):
                                failures.append(f"{tag}: received rows differ")
                            if not torch.equal(out[3][:n].cpu().long(), rids.long()):
                                failures.append(f"{tag}: received ids differ")
                            if not torch.equal(out[1][:n].cpu(), rwts):
                                failures.append(f"{tag}: received weights differ")
                    if std_moe:
                        cnt = out[5].cpu().long()
                        want_cnt = torch.zeros(epr, dtype=torch.long)
                        for r in all_rows:
                            loc = r[0][(r[0] >= rank * epr) & (r[0] < (rank + 1) * epr)]
                            want_cnt += torch.bincount(loc - rank * epr, minlength=epr)
                        if not torch.equal(cnt, want_cnt):
                            failures.append(
                                f"{tag}: packed_recv_count {cnt.tolist()} != "
                                f"{want_cnt.tolist()}"
                            )
                        scale = _expert_scale(torch.arange(epr, device=dev), epr)
                        pv = prx.view(epr, mr, hidden)
                        pv.copy_((pv.float() * scale[:, None, None]).to(pv.dtype))
                        cin = out[0][: max(n, 1)]
                        res = op.combine(cin, None, ids, packed_recv_x=prx)[0]
                    else:
                        cin = (out[0][: max(n, 1)].float() * (1.0 + rank / 8.0)).to(
                            torch.bfloat16
                        )
                        res = op.combine(cin, None, ids)[0]
                    torch.cuda.synchronize()
                    ref = _reference(x_c, ids_c, w_c, E, epr, world, std_moe)
                    if not _close(res[:T].cpu(), ref):
                        err = (res[:T].cpu().float() - ref.float()).abs().max().item()
                        failures.append(f"{tag}: combine differs (max abs {err:.3g})")
                    dist.barrier(group=group)
            del op
    except Exception as e:  # noqa: BLE001 (reported as a failure)
        failures.append(f"rank{rank}: {type(e).__name__}: {e}")
    result_q.put((rank, failures))
    try:
        dist.barrier(group=group)
        mori.shmem.shmem_finalize()
    finally:
        dist.destroy_process_group()


def run(world=None, routings=ROUTINGS, modes=MODES, tokens=None, timeout=900):
    from aiter.jit.utils.chip_info import get_gfx

    if get_gfx() not in ("gfx942", "gfx950"):
        print(f"skip: FlyDSL intra-node EP targets gfx942/gfx950, not {get_gfx()}")
        return True
    n_dev = torch.cuda.device_count()
    world = min(8, n_dev) if world is None else world
    if world < 2 or world > n_dev:
        print(f"skip: needs 2..8 GPUs, have {n_dev}")
        return True
    # (hidden, experts per rank, top-k, max tokens per rank, token counts, ...)
    cfg = (
        2048,
        8,
        6,
        256,
        tokens or [1, 7, 64, 256],
        tuple(routings),
        tuple(modes),
    )
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = _free_port()
    procs = [
        ctx.Process(target=_worker, args=(r, world, port, cfg, q)) for r in range(world)
    ]
    for p in procs:
        p.start()
    results = {}
    deadline = time.monotonic() + timeout
    while len(results) < world:
        try:
            r, f = q.get(timeout=5)
            results[r] = f
            continue
        except queue.Empty:
            pass
        dead = [
            i
            for i, p in enumerate(procs)
            if i not in results and p.exitcode not in (None, 0)
        ]
        if dead:  # a crashed rank leaves its peers waiting
            results[-1] = [f"rank(s) {dead} exited without reporting"]
            break
        if time.monotonic() > deadline:
            results[-1] = [f"timeout after {timeout}s"]
            break
    for p in procs:
        p.join(timeout=60)
        if p.is_alive():
            p.kill()
    failures = [f for r in sorted(results) for f in results[r]]
    if len(results) < world and -1 not in results:
        failures.append("a rank exited without reporting")
    for f in failures[:50]:
        print("FAIL", f)
    print(
        f"flydsl intranode EP invalid expert ids: world={world} "
        f"modes={list(modes)} routings={list(routings)}: "
        f"{'FAILED' if failures else 'passed'}"
    )
    return not failures


def test_flydsl_ep_invalid_expert_ids():
    assert run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--world", type=int, default=None)
    ap.add_argument("--routings", type=str, default=",".join(ROUTINGS))
    ap.add_argument("--modes", type=str, default=",".join(MODES))
    ap.add_argument("--tokens", type=str, default=None, help="e.g. 1,64,256")
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    ok = run(
        world=a.world,
        routings=a.routings.split(","),
        modes=a.modes.split(","),
        tokens=[int(t) for t in a.tokens.split(",")] if a.tokens else None,
        timeout=a.timeout,
    )
    sys.exit(0 if ok else 1)
