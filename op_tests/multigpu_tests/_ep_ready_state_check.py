# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Check the per-layer seeding of the tile-ready combine against a torch model.

`begin_fused_combine` has to agree exactly with the predicate a peer's ep_rowmap
builder uses, otherwise the consumer either waits for a contribution nobody will
send (hang) or sums a slot nobody wrote (wrong result). This replays dispatch with
random routing, seeds the ready state, and compares combine_meta element-wise with
a CPU reference over tok_map / topk_ids / topk_weights.

    torchrun --standalone --nproc_per_node=4 _ep_ready_state_check.py
"""
import argparse
import os

import torch
import torch.distributed as dist

from aiter.ops.flydsl.dispatch_combine_v2 import (
    EpDispatchCombineConfig,
    EpDispatchCombineOp,
)


def reference_meta(tok_map, ids, wts, *, topk, experts_per_rank, sentinel, max_tok):
    """group index + f32 weight bits every produced (token, k) should resolve to."""
    ntok = ids.shape[0]
    zero_group = topk * max_tok
    dest_pe = ids // experts_per_rank
    group = torch.full((ntok, topk), zero_group, dtype=torch.int32)
    wbits = torch.zeros((ntok, topk), dtype=torch.int32)
    w_wire = wts.to(torch.bfloat16).float()
    raw_bits = wts.contiguous().view(torch.int32)
    for t in range(ntok):
        for k in range(topk):
            leader = k
            for j in range(k):
                if dest_pe[t, j] == dest_pe[t, k]:
                    leader = j
                    break
            published = int(tok_map[t, leader]) != sentinel
            if published and float(w_wire[t, k]) != 0.0:
                group[t, k] = k * max_tok + t
                wbits[t, k] = raw_bits[t, k]
    return group, wbits


def main():
    ap = argparse.ArgumentParser(description="tile-ready combine seeding check")
    ap.add_argument("-bs", "--tokens", type=int, default=96)
    ap.add_argument("-hd", "--hidden", type=int, default=2048)
    ap.add_argument("-e", "--expert", type=int, default=64)
    ap.add_argument("-k", "--topk", type=int, default=6)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--zero_frac", type=float, default=0.1,
                    help="fraction of route weights forced to exactly 0")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    torch.cuda.set_device(local_rank)
    dev = torch.device("cuda", local_rank)

    from mori.cco import Communicator

    objs = [Communicator.get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(objs, src=0)
    comm = Communicator.init(world, rank, objs[0])

    epr = args.expert // world
    cfg = EpDispatchCombineConfig(
        rank=rank,
        world_size=world,
        hidden_dim=args.hidden,
        max_num_inp_token_per_rank=args.tokens,
        num_experts_per_rank=epr,
        num_experts_per_token=args.topk,
        data_type=torch.bfloat16,
        combine_mode="scatter_fused",
        fused_combine=True,
        fused_combine_tile_bytes=512,
    )
    op = EpDispatchCombineOp(cfg, comm)
    comm.barrier()

    sentinel = world * op._recv_cap
    bad = 0
    for layer in range(args.layers):
        g = torch.Generator(device="cpu").manual_seed(1234 + layer * 97 + rank)
        ids = torch.stack(
            [torch.randperm(args.expert, generator=g)[: args.topk] for _ in range(args.tokens)]
        ).to(torch.int32)
        wts = torch.rand((args.tokens, args.topk), generator=g, dtype=torch.float32)
        zero_mask = torch.rand((args.tokens, args.topk), generator=g) < args.zero_frac
        wts[zero_mask] = 0.0
        ids_d, wts_d = ids.to(dev), wts.to(dev)
        x = torch.randn((args.tokens, args.hidden), device=dev, dtype=torch.bfloat16)

        op.dispatch(x, wts_d, None, ids_d)
        op.begin_fused_combine(ids_d, wts_d, args.tokens)
        torch.cuda.synchronize()

        tok_map = op.token_dest_map[: args.tokens * args.topk].view(args.tokens, args.topk).cpu()
        meta = op.combine_meta.view(-1, 2)[: args.tokens * args.topk].cpu()
        got_group = meta[:, 0].view(args.tokens, args.topk)
        got_wbits = meta[:, 1].view(args.tokens, args.topk)

        exp_group, exp_wbits = reference_meta(
            tok_map, ids, wts,
            topk=args.topk, experts_per_rank=epr, sentinel=sentinel, max_tok=args.tokens,
        )
        n_g = int((got_group != exp_group).sum())
        n_w = int((got_wbits != exp_wbits).sum())
        epoch = int(op.combine_epoch.cpu()[0])
        if n_g or n_w or epoch != layer + 1:
            bad += 1
            print(
                f"[rank{rank}] layer{layer} MISMATCH group={n_g} wbits={n_w} "
                f"epoch={epoch} (want {layer + 1})",
                flush=True,
            )
        comm.barrier()

    total = torch.tensor([bad], dtype=torch.int64)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    if rank == 0:
        produced = int((op.combine_meta.view(-1, 2)[:, 1] != 0).sum())
        if int(total.item()) == 0:
            print(
                f"PASS: ready-state seeding matches reference over {args.layers} layers "
                f"(world={world} topk={args.topk} produced_routes={produced})",
                flush=True,
            )
        else:
            print(f"FAIL: {int(total.item())} layer/rank mismatches", flush=True)
    op.arena.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
