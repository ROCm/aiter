# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness of the FlyDSL intra-node EP dispatch / combine over HIP IPC
symmetric buffers (no mori).

Each rank dispatches rows with e8m0 scales, its top-k ids and weights, then
combines bf16 rows that are a function of what it received. Variants:

* ``fp8``: MXFP8 rows (fp8 + e8m0 scales), bf16 combine.
* ``fp8_direct_cast``: the same dispatch; combine keeps bf16 at the API and
  moves fp8 between ranks (``combine_quant_type="fp8_direct_cast"``).
* ``fp4``: MXFP4 rows (packed e2m1 + e8m0 scales), bf16 combine.

* dispatch: the received count equals the number of (source token, this rank)
  pairs with at least one valid expert here; every received row, its scales,
  weights and ids equal the source token's (looked up through the reported
  source position).
* combine: equal, bit for bit (+0 / -0 aside), to an f32 sum over the token's
  distinct destination ranks in top-k slot order, rounded to bf16 (each
  contribution rounded to fp8 first under ``fp8_direct_cast``). Negative ids
  are skipped, duplicates of a rank count once.
* the same pair captured in a CUDA graph and replayed several times: before
  each replay every dispatch and combine output is overwritten with garbage
  (on all ranks), and after it the dispatch and combine checks above pass
  again (launch epochs advance on the device).

Routings: random, all tokens to one rank, one rank receiving nothing, all
top-k slots on one rank, masked (-1) slots, ranks with no tokens.

    python op_tests/multigpu_tests/test_flydsl_ep_intranode.py [--world 8]
"""

import argparse
import gc
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROUTINGS = ("random", "onerank", "skip0", "dupe", "neg", "empty")
VARIANTS = ("fp8", "fp8_direct_cast", "fp4")

# e2m1 code -> value; the low nibble of a byte is the first element.
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1 = _E2M1 + tuple(-v for v in _E2M1)


_LUTS = {}


def _as_float(t):
    """Rows as f32; packed fp4 rows are decoded to twice as many columns.

    The decode table is uploaded on the first (eager) call per device, so the
    same call can be captured in a CUDA graph later."""
    if t.dtype != torch.float4_e2m1fn_x2:
        return t.float()
    lut = _LUTS.get(t.device)
    if lut is None:
        lut = _LUTS[t.device] = torch.tensor(
            _E2M1, dtype=torch.float32, device=t.device
        )
    b = t.view(torch.uint8).long()
    return torch.stack((lut[b & 15], lut[b >> 4]), dim=-1).flatten(-2)


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


def _stacked(all_rows, t_max, j, cols, dtype):
    """Field j of every rank's gathered rows as one [world, t_max, cols] tensor."""
    out = torch.zeros(len(all_rows), t_max, cols, dtype=dtype)
    for r_i, r in enumerate(all_rows):
        out[r_i, : r[j].shape[0]] = r[j].reshape(r[j].shape[0], cols)
    return out


def poison(*tensors):
    """Overwrite output buffers with NaN / all-ones bytes, so that a check
    after a graph replay cannot pass on what an earlier launch left there."""
    for t in tensors:
        if t is not None:
            t.view(torch.uint8).fill_(0xFF)


def _reference(xf, ids, epr, wire_fp8=False):
    """Combine reference for source rows ``xf`` (f32): destination rank r
    returns its received row times ``1 + r / 8`` in bf16; f32 sum over the
    distinct valid destination ranks in top-k order -> bf16. ``wire_fp8``:
    each returned row travels as fp8 (``fp8_direct_cast``)."""
    T, K = ids.shape
    acc = torch.zeros(T, xf.shape[1], dtype=torch.float32, device=xf.device)
    dest = torch.where(ids >= 0, torch.div(ids, epr, rounding_mode="floor"), -1)
    for k in range(K):
        first = dest[:, k] >= 0
        for j in range(k):
            first &= dest[:, j] != dest[:, k]
        scale = 1.0 + dest[:, k].clamp(min=0).float() / 8.0
        contrib = (xf * scale[:, None]).to(torch.bfloat16).float()
        if wire_fp8:
            contrib = contrib.to(torch.float8_e4m3fn).float()
        acc += torch.where(first[:, None], contrib, torch.zeros_like(contrib))
    return acc.to(torch.bfloat16)


def _worker(rank, world, port, cases, tokens, routings, replays, result_q):
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
    fp4 = torch.float4_e2m1fn_x2
    try:
        for variant, (hidden, E, K, max_tok) in cases:
            epr = E // world
            sd = hidden // 32
            d_dtype = fp4 if variant == "fp4" else fp8
            wire_fp8 = variant == "fp8_direct_cast"
            xb = hidden // 2 if variant == "fp4" else hidden  # row bytes
            op = FlyDSLDispatchCombineIntraNodeOp(
                FlyDSLDispatchCombineConfig(
                    rank=rank,
                    world_size=world,
                    hidden_dim=hidden,
                    max_num_inp_token_per_rank=max_tok,
                    num_experts_per_rank=epr,
                    num_experts_per_token=K,
                    dispatch_dtype=d_dtype,
                    combine_dtype=torch.bfloat16,
                    combine_quant_type="fp8_direct_cast" if wire_fp8 else "none",
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
                    if variant == "fp4":
                        x = (
                            torch.randint(0, 256, (T, xb), generator=g)
                            .to(torch.uint8)
                            .view(fp4)
                            .to(dev)
                        )
                    else:
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

                    def combine(out, op=op, ids=ids, n_in=n_in):
                        rows = min(out[0].shape[0], n_in)
                        cin = (_as_float(out[0][:rows]) * (1.0 + rank / 8.0)).to(
                            torch.bfloat16
                        )
                        return op.combine(cin, None, ids)[0]

                    tag = (
                        f"rank{rank} {variant} hidden={hidden} E={E} K={K} "
                        f"{routing} T={T0}"
                    )

                    def check_dispatch(
                        out,
                        src,
                        when,
                        tag=tag,
                        all_rows=all_rows,
                        geom=(epr, max_tok, sd, xb, K),
                    ):
                        epr, max_tok, sd, xb, K = geom
                        # out[4] / src: the count and source positions as
                        # dispatch left them (combine resets the count)
                        n = int(out[4].reshape(-1)[0].item())
                        src = src[:n].long().cpu()
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
                            failures.append(
                                f"{tag}: {when} received {n}, expected {expect}"
                            )
                        if not n or n != expect:
                            return
                        # every received row against its source token
                        t_max = max(int(r[0].shape[0]) for r in all_rows)
                        s_r, s_t = src // max_tok, src % max_tok
                        got = (
                            out[0][:n].view(torch.uint8).cpu(),
                            out[2][:n].reshape(n, -1)[:, :sd].contiguous().cpu(),
                            out[1][:n].cpu(),
                            out[3][:n].cpu(),
                        )
                        want = (
                            _stacked(all_rows, t_max, 1, xb, torch.uint8)[s_r, s_t],
                            _stacked(all_rows, t_max, 2, sd, torch.uint8)[s_r, s_t],
                            _stacked(all_rows, t_max, 3, K, torch.float32)[s_r, s_t],
                            _stacked(all_rows, t_max, 0, K, torch.int32)[s_r, s_t],
                        )
                        for name, g_t, w_t in zip(
                            ("rows", "scales", "weights", "ids"), got, want
                        ):
                            if not torch.equal(g_t.view(w_t.dtype), w_t):
                                failures.append(f"{tag}: {when} received {name} differ")

                    out = op.dispatch(x, w, sc, ids)
                    torch.cuda.synchronize()
                    check_dispatch(out, op.shmem_tok_id_to_src, "dispatch")
                    # combine
                    res = combine(out)
                    torch.cuda.synchronize()
                    ref = _reference(_as_float(x), ids, epr, wire_fp8)
                    bad = _bitdiff(res[:T], ref) if T else 0
                    if bad:
                        failures.append(f"{tag}: combine {bad} elements differ")
                    # CUDA graph capture + replays. The graph's outputs are the
                    # op's buffers, which still hold the eager results: poison
                    # them on every rank before each replay.
                    dist.barrier(group=group)
                    s = torch.cuda.Stream()
                    with torch.cuda.stream(s):
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=s):
                            out_g = op.dispatch(x, w, sc, ids)
                            # combine resets the count
                            n_g = out_g[4].clone()
                            src_g = op.shmem_tok_id_to_src.clone()
                            res_g = combine(out_g)
                    torch.cuda.synchronize()
                    for i in range(replays):
                        poison(*out_g[:4], n_g, src_g, res_g, op.shmem_tok_id_to_src)
                        torch.cuda.synchronize()
                        dist.barrier(group=group)
                        graph.replay()
                        torch.cuda.synchronize()
                        dist.barrier(group=group)
                        check_dispatch((*out_g[:4], n_g), src_g, f"replay {i}")
                        bad = _bitdiff(res_g[:T], ref) if T else 0
                        if bad:
                            failures.append(
                                f"{tag}: graph replay {i} combine {bad} elements differ"
                            )
                    del graph
                    dist.barrier(group=group)
            op.close()
            # Drop every view of the op's buffers here: IPC memory is freed when
            # its last tensor dies, and that must not happen inside the next
            # variant's graph capture.
            op = out = out_g = res = res_g = n_g = src_g = combine = None
            gc.collect()
    except Exception as e:  # noqa: BLE001
        failures.append(f"rank{rank}: {type(e).__name__}: {e}")
    result_q.put((rank, failures))
    dist.barrier(group=group)
    dist.destroy_process_group()


def run(
    world=None,
    shapes=None,
    tokens=None,
    routings=ROUTINGS,
    replays=3,
    variants=VARIANTS,
):
    from aiter.dist.utils import get_open_port
    from aiter.jit.utils.chip_info import get_gfx_runtime

    gfx = get_gfx_runtime()
    if gfx not in ("gfx942", "gfx950"):
        print(f"skip: FlyDSL intra-node EP kernels target gfx942/gfx950, not {gfx}")
        return True
    n_dev = torch.cuda.device_count()
    world = min(8, n_dev) if world is None else world
    if world < 2 or world > min(8, n_dev):
        print(f"skip: world {world} needs 2..8 GPUs, have {n_dev}")
        return True
    # (hidden, experts, top-k, max tokens per rank)
    shapes = shapes or [(5120, 48 * world, 6, 4096), (2048, 16 * world, 3, 1024)]
    tokens = tokens or [1, 7, 32, 192, 1000]
    # fp8 runs every shape; the other variants run the last one
    cases = [
        (v, sh) for v in variants for sh in (shapes if v == "fp8" else shapes[-1:])
    ]
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = get_open_port()
    procs = [
        ctx.Process(
            target=_worker,
            args=(r, world, port, cases, tokens, routings, replays, q),
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
        f"routings={list(routings)} variants={list(variants)}: "
        f"{'FAILED' if failures else 'passed'}"
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
    ap.add_argument("--variants", type=str, default=",".join(VARIANTS))
    a = ap.parse_args()
    ok = run(
        world=a.world,
        tokens=[int(t) for t in a.tokens.split(",")] if a.tokens else None,
        routings=a.routings.split(","),
        replays=a.replays,
        variants=a.variants.split(","),
    )
    sys.exit(0 if ok else 1)
