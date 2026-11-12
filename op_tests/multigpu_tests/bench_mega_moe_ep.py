# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused EP MegaMoE layer perf (MXFP4 weights; ``--network`` shape as in
``test_mega_moe_ep``) over token counts and routings, against ``MegaMoEV2`` and the
unfused mori dispatch -> ``fused_moe`` -> mori combine on the same shape::

    torchrun --nproc_per_node=8 op_tests/multigpu_tests/bench_mega_moe_ep.py \\
        --engine ep -q a8w4 --mtpr 8192 -t 1024 2048 4096 8192 -r uniform zipf:1 spike

One engine per run (``--engine ep | v2 | mori``): EP uses a hipIpc symmetric arena, V2
and mori use mori shmem.  ``v2`` takes whichever ``MegaMoEV2`` is on ``PYTHONPATH``; if it
has a small-m path (``small_m_mtpr``) the token counts it covers go through
``forward_small_m``.  ``mori`` is the online EP MoE of ATOM's ``MoriPrepareAndFinalize``:
bf16 mori dispatch, ``fused_moe`` (gate|up interleaved MXFP4 weights, MXFP8
activations; a8w4 only), mori combine, with ATOM's (blocks, warps) per phase and its
decode receive-buffer trim (``--mori-phase``; ``auto``: decode at mtpr <= 512).

The time is the mean over ``--iters`` CUDA-graph replays of one forward (``--eager``:
eager calls), wall clock, taken as the worst rank.  The table also lists, per case,
``bal`` = mean / max routed rows per rank and ``hot`` = max / mean routed rows per
expert.  Routings: see ``test_mega_moe_ep.route_ids``.  The weights are random, so the
outputs are not checked here (``test_mega_moe_ep.py`` does that).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_mega_moe_ep import NETWORKS, SITU_BETA, route_ids

MORI_DECODE_MAX_MTPR = 512


def _weights(rank, device, H, I, epr):
    qz = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    g = torch.Generator(device=device).manual_seed(100 + rank)
    out = []
    for rows, cols in ((2 * I, H), (H, I)):
        w = torch.randn(
            epr, rows, cols, dtype=torch.bfloat16, device=device, generator=g
        )
        q, s = qz(w * cols**-0.25, quant_dtype=dtypes.fp4x2)
        out += [q.view(epr, rows, cols // 2).contiguous(), s]
    return out


class MoriFusedMoE:
    """mori dispatch -> ``fused_moe`` -> mori combine as ATOM runs it online."""

    def __init__(self, common, weights, phase, device):
        import mori

        from aiter.fused_moe import fused_moe
        from aiter.ops.flydsl.moe_common import GateMode

        rank, world = common["rank"], common["world_size"]
        epr = common["experts"] // world
        self.fused_moe = fused_moe
        self.gate_mode = GateMode.INTERLEAVE.value
        self.weights, self.world = weights, world
        self.swiglu_limit = common["swiglu_limit"] or None
        self.phase = phase
        self.decode = phase == "decode"
        cu = torch.cuda.get_device_properties(device).multi_processor_count
        # ATOM MoriPrepareAndFinalize._get_dispatch_config()
        self.launch = (min(64, cu), 4) if self.decode else (min(128, cu), 16)
        self.op = mori.ops.EpDispatchCombineOp(
            mori.ops.EpDispatchCombineConfig(
                data_type=torch.bfloat16,
                rank=rank,
                world_size=world,
                hidden_dim=common["model_dim"],
                scale_dim=0,
                scale_type_size=torch.float32.itemsize,
                max_token_type_size=torch.bfloat16.itemsize,
                max_num_inp_token_per_rank=common["max_tok_per_rank"],
                num_experts_per_rank=epr,
                num_experts_per_token=common["topk"],
                warp_num_per_block=16,
                block_num=80,
                rdma_block_num=0,
                kernel_type=mori.ops.EpDispatchCombineKernelType.IntraNode,
                gpu_per_node=world,
            )
        )
        # the trailing zero is the sentinel fused_moe maps non-local experts to
        self.expert_mask = torch.zeros(
            common["experts"] + 1, dtype=torch.int32, device=device
        )
        self.expert_mask[rank * epr : (rank + 1) * epr] = 1

    def __call__(self, x, wts, ids):
        tokens = x.shape[0]
        blocks, warps = self.launch
        recv, recv_wts, _, recv_ids, recv_tokens = self.op.dispatch(
            x, wts, None, ids, block_num=blocks, warp_per_block=warps
        )
        if self.decode:
            # mori sends a token once per destination rank, so equal-batch decode
            # receives at most tokens * world rows (ATOM's static graph trim)
            rows = tokens * self.world
            recv, recv_wts, recv_ids = recv[:rows], recv_wts[:rows], recv_ids[:rows]
        w1, w1_scale, w2, w2_scale = self.weights
        out = self.fused_moe(
            recv,
            w1,
            w2,
            recv_wts,
            recv_ids,
            self.expert_mask,
            quant_type=aiter.QuantType.per_1x32,
            num_local_tokens=recv_tokens,
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            dtype=torch.bfloat16,
            swiglu_limit=self.swiglu_limit,
            gate_mode=self.gate_mode,
            quant_dtype_a=dtypes.fp8,
        )
        return self.op.combine(out, None, ids, block_num=blocks, warp_per_block=warps)[
            0
        ][:tokens]


def build(engine, network, quant, mtpr, rank, world, device, mori_phase="auto"):
    H, I, K, swiglu_limit, epr, act = NETWORKS[network]
    w1q, w1s, w2q, w2s = _weights(rank, device, H, I, epr)
    common = {
        "rank": rank,
        "world_size": world,
        "model_dim": H,
        "inter_dim": I,
        "experts": epr * world,
        "topk": K,
        "max_tok_per_rank": mtpr,
        "quant": quant,
        "swiglu_limit": swiglu_limit,
        "act": act,
    }
    if act == "situv2":
        common.update(situ_beta=SITU_BETA[0], situ_linear_beta=SITU_BETA[1])
    if engine == "ep":
        from aiter.ops.flydsl.mega_moe_ep import MegaMoeEP, MegaMoeEPConfig

        layer = MegaMoeEP(
            MegaMoeEPConfig(**common),
            w1=w1q,
            w1_scale=w1s.view(epr, 2 * I, H // 32).contiguous(),
            w2=w2q,
            w2_scale=w2s.view(epr, H, I // 32).contiguous(),
            device=device,
        )
        layer.prepare()
        return layer
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    shuffled = {
        "w1": shuffle_weight_a16w4(w1q, 16, True).contiguous(),
        "w1_scale": shuffle_scale_a16w4(w1s, epr, True).contiguous(),
        "w2": shuffle_weight_a16w4(w2q, 16, False).contiguous(),
        "w2_scale": shuffle_scale_a16w4(w2s, epr, False).contiguous(),
    }
    if engine == "mori":
        if act != "silu":
            raise ValueError(f"--engine mori: {network} ({act}) is not a SwiGLU MoE")
        # fused_moe with MXFP4 activations and interleaved gate|up is wrong
        if quant != "a8w4":
            raise ValueError("--engine mori supports -q a8w4 only")
        if mori_phase == "auto":
            mori_phase = "decode" if mtpr <= MORI_DECODE_MAX_MTPR else "prefill"
        weights = [shuffled[k] for k in ("w1", "w1_scale", "w2", "w2_scale")]
        return MoriFusedMoE(common, weights, mori_phase, device)
    from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2

    return MegaMoEV2(**common, **shuffled)


def balance(ids, world, epr):
    rk = torch.bincount(ids.flatten().long() // epr, minlength=world).float()
    ex = torch.bincount(ids.flatten().long(), minlength=epr * world).float()
    dist.all_reduce(rk)
    dist.all_reduce(ex)
    return (rk.mean() / rk.max()).item(), (ex.max() / ex.mean()).item()


def bench(layer, x, wts, ids, args, device):
    tokens = x.shape[0]
    if tokens <= getattr(layer, "small_m_mtpr", 0):
        call = lambda: layer.forward_small_m(x, wts, ids, max_tokens=tokens)
    else:
        call = lambda: layer(x, wts, ids)
    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    dist.barrier()
    step = call
    if not args.eager:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for _ in range(args.warmup):
            graph.replay()
        step = graph.replay
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    ms = torch.tensor([(time.perf_counter() - t0) / args.iters * 1e3], device=device)
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    return ms.item()


def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument("--engine", default="ep", choices=["ep", "v2", "mori"])
    parser.add_argument(
        "--mori-phase",
        default="auto",
        choices=["auto", "decode", "prefill"],
        help="ATOM phase of --engine mori (auto: decode at mtpr <= 512).",
    )
    parser.add_argument("--network", choices=NETWORKS, default="v4_pro")
    parser.add_argument("-q", "--quant", default="a8w4", choices=["a8w4", "a4w4"])
    parser.add_argument("--mtpr", type=int, default=8192, help="max_tok_per_rank")
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[1, 64, 512, 1024, 2048, 4096, 8192],
        help="Tokens per rank (cases above mtpr are skipped).",
    )
    parser.add_argument(
        "-r",
        "--routes",
        nargs="*",
        default=["uniform", "zipf:0.5", "zipf:1", "zipf:2", "spike"],
    )
    parser.add_argument("--eager", action="store_true", help="Time eager calls.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    device = torch.device("cuda", lr)
    dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    if args.engine == "mori":
        # ATOM passes the phase geometry to every call; MORI's AUTO rules would
        # override it
        os.environ["MORI_EP_LAUNCH_CONFIG_MODE"] = "MANUAL"
        os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", str(16 << 30))
    if args.engine in ("v2", "mori"):
        import mori.shmem as mori_shmem
        import torch._C._distributed_c10d as c10d

        c10d._register_process_group("default", dist.group.WORLD)
        mori_shmem.shmem_torch_process_group_init("default")

    H, _, K, _, epr, _ = NETWORKS[args.network]
    layer = build(
        args.engine,
        args.network,
        args.quant,
        args.mtpr,
        rank,
        world,
        device,
        args.mori_phase,
    )
    rows = []
    for route in args.routes:
        for tokens in (t for t in args.tokens if t <= args.mtpr):
            x, wts, ids = route_ids(
                route, tokens, epr * world, world, rank, 0, device, H, K
            )
            bal, hot = balance(ids, world, epr)
            ms = bench(layer, x, wts, ids, args, device)
            rows.append(
                {
                    "engine": args.engine
                    + (f":{layer.phase}" if args.engine == "mori" else ""),
                    "network": args.network,
                    "quant": args.quant,
                    "mtpr": args.mtpr,
                    "route": route,
                    "tokens": tokens,
                    "bal": f"{bal:.3f}",
                    "hot": f"{hot:.1f}",
                    "ms": f"{ms:.3f}",
                }
            )
            if rank == 0:
                aiter.logger.info("%s", rows[-1])
            dist.barrier()
    if rank == 0:
        mode = "eager" if args.eager else "graph"
        aiter.logger.info(
            "MegaMoE EP bench ep%d (%s, worst rank):\n%s",
            world,
            mode,
            pd.DataFrame(rows).to_markdown(index=False),
        )
    del layer
    if args.engine in ("v2", "mori"):
        mori_shmem.shmem_finalize()
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
