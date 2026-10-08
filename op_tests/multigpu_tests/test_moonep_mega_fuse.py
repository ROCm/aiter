# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
"""8-GPU check: MegaMoEV2 with MoonEP placement fused into prepare.

Runs the same skewed routing through a plain MegaMoEV2 and through a
``moonep_slots=B`` instance whose weight window is ``[EPR home | B slots]``.
The slots are filled in ``after_prepare`` from the ``placed`` table prepare
wrote, the way a production weight pool would.  Outputs must match.

    torchrun --nproc-per-node 8 op_tests/multigpu_tests/test_moonep_mega_fuse.py
"""

import argparse
import os

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "40G")

import mori.shmem as ms  # noqa: E402
import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2  # noqa: E402

from test_mega_moe_v2 import _barrier, _cleanup, _quantize_weights, _setup_dist  # noqa: E402

MODEL_DIM, INTER_DIM, EXPERTS, TOPK, SWIGLU = 7168, 3072, 384, 6, 10.0


def _routing(tokens, rank, step, device, *, hot):
    gen = torch.Generator(device=device).manual_seed(1234 + 97 * step + rank)
    scores = torch.randn((tokens, EXPERTS), device=device, generator=gen)
    if hot:
        # Two hot ranks with many similar experts: several prefetches per rank.
        bias = torch.zeros(EXPERTS, device=device)
        hot_ranks = [(step % 8), (step + 3) % 8]
        for h in hot_ranks:
            bias[h * 48:(h + 1) * 48] = 1.5
        scores = scores + bias
    values, ids = torch.topk(scores, TOPK, dim=-1)
    x = torch.randn((tokens, MODEL_DIM), dtype=torch.bfloat16, device=device, generator=gen)
    return x, values.softmax(-1).contiguous(), ids.to(torch.int32).contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--skip-prefetch", action="store_true",
                        help="negative control: leave the slots stale")
    args = parser.parse_args()
    rank, world, device = _setup_dist()
    epr = EXPERTS // world
    B = args.slots
    try:
        everyone = [_quantize_weights(MODEL_DIM, INTER_DIM, epr, r, 7, device)[:4] for r in range(world)]
        w1, w1s, w2, w2s = everyone[rank]
        plain = MegaMoEV2(
            rank=rank, world_size=world, model_dim=MODEL_DIM, inter_dim=INTER_DIM,
            experts=EXPERTS, topk=TOPK, quant="a8w4", w1=w1, w1_scale=w1s, w2=w2,
            w2_scale=w2s, max_tok_per_rank=args.tokens, swiglu_limit=SWIGLU,
        )

        def per_expert(t):
            return t.contiguous().view(torch.uint8).view(epr, -1)

        # All experts, flattened per expert, in global order.
        bank = [torch.cat([per_expert(e[i]) for e in everyone]) for i in range(4)]
        windows = [
            torch.cat([per_expert(t), torch.zeros_like(per_expert(t)[:B])]).contiguous()
            for t in (w1, w1s, w2, w2s)
        ]
        views = [win.view(t.dtype) for win, t in zip(windows, (w1, w1s, w2, w2s))]
        fused = MegaMoEV2(
            rank=rank, world_size=world, model_dim=MODEL_DIM, inter_dim=INTER_DIM,
            experts=world * (epr + B), topk=TOPK, quant="a8w4",
            w1=views[0], w1_scale=views[1], w2=views[2], w2_scale=views[3],
            max_tok_per_rank=args.tokens, swiglu_limit=SWIGLU, moonep_slots=B,
        )
        placed, prev = fused.moonep_slot_tables()

        def prefetch():
            if args.skip_prefetch:
                return
            want = placed[rank].long()
            copy = (want >= 0) & (want != prev[rank].long())
            src = want.clamp(min=0)
            for win, all_experts in zip(windows, bank):
                rows = win[epr:]
                rows.copy_(torch.where(copy[:, None], all_experts.index_select(0, src), rows))

        ok = True
        for step in range(args.steps):
            for balance in ((True, False) if step == 0 else (True,)):
                x, wts, ids = _routing(args.tokens, rank, step, device, hot=step % 2 == 0)
                ref = plain.forward(x, wts, ids).clone()
                _barrier()
                out = fused.forward(x, wts, ids, moonep_balance=balance, after_prepare=prefetch).clone()
                _barrier()
                diff = (out.float() - ref.float()).abs()
                scale = ref.float().abs().max().clamp(min=1e-6)
                max_rel = float(diff.max() / scale)
                exact = bool(torch.equal(out, ref))
                prefetched = int((placed >= 0).sum())
                stats = torch.tensor([max_rel, float(exact), prefetched], device=device)
                gathered = [torch.zeros_like(stats) for _ in range(world)]
                dist.all_gather(gathered, stats)
                if rank == 0:
                    rels = [float(g[0]) for g in gathered]
                    exacts = sum(int(g[1]) for g in gathered)
                    step_ok = max(rels) < 2e-2
                    ok &= step_ok
                    print(f"step={step} balance={balance} hot={step % 2 == 0} "
                          f"placed={int(gathered[0][2])} exact_ranks={exacts}/{world} "
                          f"max_rel={max(rels):.3e} -> {'OK' if step_ok else 'FAIL'}", flush=True)
        if rank == 0:
            print("ALL_OK" if ok else "SOME_FAILED", flush=True)
    finally:
        _barrier()
        _cleanup()


if __name__ == "__main__":
    main()
