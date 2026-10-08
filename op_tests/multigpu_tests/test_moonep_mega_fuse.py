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
    parser.add_argument("--layers", type=int, default=2,
                        help="weight windows sharing one instance, like model layers")
    parser.add_argument("--skip-prefetch", action="store_true",
                        help="negative control: leave the slots stale")
    parser.add_argument("--shared-state", action="store_true",
                        help="negative control: all layers share one slot state")
    parser.add_argument("--mtpr", type=int, default=0,
                        help="instance capacity (default: --tokens), e.g. decode batches "
                             "far below a prefill-sized instance")
    parser.add_argument("--graph", action="store_true",
                        help="run the fused instance from one CUDA graph per layer")
    args = parser.parse_args()
    rank, world, device = _setup_dist()
    epr = EXPERTS // world
    B = args.slots
    try:

        def per_expert(t):
            return t.contiguous().view(torch.uint8).view(epr, -1)

        layers = []
        for layer in range(args.layers):
            everyone = [
                _quantize_weights(MODEL_DIM, INTER_DIM, epr, r, 7 + 31 * layer, device)[:4]
                for r in range(world)
            ]
            home = everyone[rank]
            # All experts, flattened per expert, in global order.
            bank = [torch.cat([per_expert(e[i]) for e in everyone]) for i in range(4)]
            windows = [
                torch.cat([per_expert(t), torch.zeros_like(per_expert(t)[:B])]).contiguous()
                for t in home
            ]
            views = [win.view(t.dtype) for win, t in zip(windows, home)]
            layers.append(dict(home=home, bank=bank, windows=windows, views=views))
            del everyone

        def bind_plain(moe, layer):
            w1, w1s, w2, w2s = layer["home"]
            moe._s1_w1, moe._s1_w1_scale = w1.view(torch.uint8), w1s.view(torch.uint8)
            moe.w2, moe.w2_scale = w2, w2s

        def bind_fused(moe, layer):
            w1, w1s, w2, w2s = layer["views"]
            moe._s1_w1, moe._s1_w1_scale = w1.view(torch.uint8), w1s.view(torch.uint8)
            moe.w2, moe.w2_scale = w2, w2s
            moe.bind_moonep_slot_state(layer["state"])

        first = layers[0]
        plain = MegaMoEV2(
            rank=rank, world_size=world, model_dim=MODEL_DIM, inter_dim=INTER_DIM,
            experts=EXPERTS, topk=TOPK, quant="a8w4", w1=first["home"][0],
            w1_scale=first["home"][1], w2=first["home"][2], w2_scale=first["home"][3],
            max_tok_per_rank=args.mtpr or args.tokens, swiglu_limit=SWIGLU,
        )
        fused = MegaMoEV2(
            rank=rank, world_size=world, model_dim=MODEL_DIM, inter_dim=INTER_DIM,
            experts=world * (epr + B), topk=TOPK, quant="a8w4",
            w1=first["views"][0], w1_scale=first["views"][1], w2=first["views"][2],
            w2_scale=first["views"][3], max_tok_per_rank=args.mtpr or args.tokens,
            swiglu_limit=SWIGLU, moonep_slots=B,
        )
        shared = fused.new_moonep_slot_state()
        for layer in layers:
            layer["state"] = shared if args.shared_state else fused.new_moonep_slot_state()

        def prefetch(layer):
            if args.skip_prefetch:
                return
            placed, prev = fused.moonep_slot_tables()
            want = placed[rank].long()
            copy = (want >= 0) & (want != prev[rank].long())
            src = want.clamp(min=0)
            for win, all_experts in zip(layer["windows"], layer["bank"]):
                rows = win[epr:]
                rows.copy_(torch.where(copy[:, None], all_experts.index_select(0, src), rows))

        graphs = {}

        def replay(index, layer, x, wts, ids):
            """Capture this layer once (prepare, slot fill and all), then replay."""
            if index not in graphs:
                static = {"x": x.clone(), "wts": wts.clone(), "ids": ids.clone()}

                def run():
                    return fused.forward(
                        static["x"], static["wts"], static["ids"],
                        after_prepare=lambda: prefetch(layer),
                    )

                run()
                _barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
                    static["out"] = run()
                graphs[index] = (graph, static)
            graph, static = graphs[index]
            static["x"].copy_(x)
            static["wts"].copy_(wts)
            static["ids"].copy_(ids)
            graph.replay()
            return static["out"].clone()

        ok = True
        for step in range(args.steps):
            for index, layer in enumerate(layers):
                for balance in ((True, False) if step == 0 and not args.graph else (True,)):
                    # Layers see different hot experts, so their slots diverge.
                    x, wts, ids = _routing(args.tokens, rank, step + 5 * index, device, hot=step % 2 == 0)
                    bind_plain(plain, layer)
                    ref = plain.forward(x, wts, ids).clone()
                    _barrier()
                    bind_fused(fused, layer)
                    if args.graph:
                        out = replay(index, layer, x, wts, ids)
                    else:
                        out = fused.forward(
                            x, wts, ids, moonep_balance=balance,
                            after_prepare=lambda layer=layer: prefetch(layer),
                        ).clone()
                    _barrier()
                    diff = (out.float() - ref.float()).abs()
                    scale = ref.float().abs().max().clamp(min=1e-6)
                    max_rel = float(diff.max() / scale)
                    exact = bool(torch.equal(out, ref))
                    placed, prev = fused.moonep_slot_tables()
                    prefetched = int((placed >= 0).sum())
                    kept = int(((placed >= 0) & (placed == prev)).sum())
                    stats = torch.tensor([max_rel, float(exact), prefetched, kept], device=device)
                    gathered = [torch.zeros_like(stats) for _ in range(world)]
                    dist.all_gather(gathered, stats)
                    if rank == 0:
                        rels = [float(g[0]) for g in gathered]
                        exacts = sum(int(g[1]) for g in gathered)
                        step_ok = max(rels) < 2e-2
                        ok &= step_ok
                        print(f"step={step} layer={index} balance={balance} hot={step % 2 == 0} "
                              f"placed={int(gathered[0][2])} kept={int(gathered[0][3])} "
                              f"exact_ranks={exacts}/{world} max_rel={max(rels):.3e} "
                              f"-> {'OK' if step_ok else 'FAIL'}", flush=True)
        if rank == 0:
            print("ALL_OK" if ok else "SOME_FAILED", flush=True)
    finally:
        _barrier()
        _cleanup()


if __name__ == "__main__":
    main()
