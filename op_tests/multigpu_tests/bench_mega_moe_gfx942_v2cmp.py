# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MegaMoEGfx942 measured the way ``test_mega_moe_v2.py`` measures MegaMoEV2, for an
apples-to-apples comparison of the two on gfx942.

Same inputs (seed + rank: bf16 x ~ randn, top-k of randn scores, softmax weights; the
rows of a batch are the first ``bs`` of ``max(--bs-list)``), same MXFP4 experts (seed +
1000 + rank, per_1x32 quant of randn * H^-1/4 / I^-1/4; only the preshuffle differs:
MegaMoEGfx942 takes w1 with gate_up=False), same fp32 reference and relL2 (over all
ranks), same timing: the call captured in a CUDA graph on a side stream, 10 replays to
warm up, then ``--iters`` replays between two CUDA events, the mean / max over the ranks.
``--bs-list`` holds tokens per rank. Each path of MegaMoEGfx942 is timed on its own:

- main:  ``forward`` of an instance built with large_M off (the main path)
- large: ``forward`` of an instance built with large_M on
- small: ``forward_small_m`` (the small kernel) of the large_M instance, for batches up
  to ``--small-mtpr`` tokens per rank

    HIP_VISIBLE_DEVICES=2,3,4,5 torchrun --nproc_per_node=4 \\
        op_tests/multigpu_tests/bench_mega_moe_gfx942_v2cmp.py --network v41_flash \\
        --max-tok-per-rank 8192 --bs-list 1,16,128,512,2048,8192 --iters 10

Elsewhere (not gfx942, no mori, not under torchrun, or a layer MegaMoEGfx942 does not
take) it prints why and exits 0.
"""

import argparse
import os

os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "16G")

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.utility import fp4_utils

try:
    import mori.shmem as ms
except ImportError:  # _skip_reason reports it
    ms = None

NETWORKS = {
    "v41_flash": {
        "model_dim": 5120,
        "inter_dim": 2304,
        "experts": 384,
        "topk": 6,
        "swiglu_limit": 10.0,
    },
    "v4_flash": {
        "model_dim": 4096,
        "inter_dim": 2048,
        "experts": 256,
        "topk": 6,
        "swiglu_limit": 10.0,
    },
}
PATHS = ("main", "large", "small")


def _skip_reason(network):
    """Why the bench cannot run here (None: it can)."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    gfx = get_gfx_runtime()
    if gfx != "gfx942":
        return f"MegaMoEGfx942 runs on gfx942, this is {gfx}"
    if ms is None:
        return "needs mori"
    if "WORLD_SIZE" not in os.environ:
        return "launch it with torchrun (see the module docstring)"
    from aiter.ops.flydsl.kernels.mega_moe_gfx942 import eligible

    net, world = NETWORKS[network], int(os.environ["WORLD_SIZE"])
    if not eligible(
        world_size=world,
        model_dim=net["model_dim"],
        inter_dim=net["inter_dim"],
        experts=net["experts"],
        topk=net["topk"],
    ):
        return f"MegaMoEGfx942 does not run {network} over {world} ranks here"
    return None


def _setup_dist():
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    import torch._C._distributed_c10d as c10d

    c10d._register_process_group("default", dist.group.WORLD)
    ms.shmem_torch_process_group_init("default")
    return rank, dist.get_world_size(), device


def _barrier():
    torch.cuda.synchronize()
    ms.shmem_barrier_all()


def _reduce_float(value, device, op):
    result = torch.tensor(float(value), dtype=torch.float32, device=device)
    dist.all_reduce(result, op=op)
    return float(result.item())


def _rel_l2(output, reference):
    """Relative L2 error over all ranks (test_mega_moe_v2's)."""
    error_sq = torch.sum((output.float() - reference) ** 2)
    reference_sq = torch.sum(reference**2)
    dist.all_reduce(error_sq)
    dist.all_reduce(reference_sq)
    return float(torch.sqrt(error_sq / reference_sq))


def _max_row_err(output, reference, device):
    """max |out - ref| / rms(reference row) over all ranks (test_mega_moe_gfx942's)."""
    rms = reference.pow(2).mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-30)
    worst = ((output.float() - reference) / rms).abs().max()
    return _reduce_float(worst, device, dist.ReduceOp.MAX)


def _make_inputs(tokens, model_dim, experts, topk, rank, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    x = torch.randn(
        (tokens, model_dim), dtype=torch.bfloat16, device=device, generator=generator
    )
    scores = torch.randn(
        (tokens, experts), dtype=torch.float32, device=device, generator=generator
    )
    values, ids = torch.topk(scores, topk, dim=-1)
    return (
        x.contiguous(),
        values.softmax(dim=-1).contiguous(),
        ids.to(torch.int32).contiguous(),
    )


def _quantize_weights(model_dim, inter_dim, local_experts, rank, seed, device):
    """test_mega_moe_v2's experts, w1 preshuffled with gate_up=False."""
    generator = torch.Generator(device=device).manual_seed(seed + 1000 + rank)
    quantize = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    w1 = torch.randn(
        (local_experts, 2 * inter_dim, model_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w1.mul_(model_dim**-0.25)
    w1_q, w1_scale = quantize(w1, quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(local_experts, 2 * inter_dim, model_dim // 2)
    w1_scale_ref = w1_scale.view(local_experts, 2 * inter_dim, model_dim // 32)
    del w1
    w1_kernel = shuffle_weight_a16w4(w1_q, 16, False).contiguous()
    w1_scale_kernel = shuffle_scale_a16w4(w1_scale, local_experts, False).contiguous()
    w2 = torch.randn(
        (local_experts, model_dim, inter_dim),
        dtype=torch.bfloat16,
        device=device,
        generator=generator,
    )
    w2.mul_(inter_dim**-0.25)
    w2_q, w2_scale = quantize(w2, quant_dtype=dtypes.fp4x2)
    w2_q = w2_q.view(local_experts, model_dim, inter_dim // 2)
    w2_scale_ref = w2_scale.view(local_experts, model_dim, inter_dim // 32)
    del w2
    w2_kernel = shuffle_weight_a16w4(w2_q, 16, False).contiguous()
    w2_scale_kernel = shuffle_scale_a16w4(w2_scale, local_experts, False).contiguous()
    torch.cuda.empty_cache()
    kernel = (w1_kernel, w1_scale_kernel, w2_kernel, w2_scale_kernel)
    return kernel, (w1_q, w1_scale_ref, w2_q, w2_scale_ref)


def _dequant_expert(weight, scale, rows, cols):
    values = fp4_utils.mxfp4_to_f32(weight).view(rows, cols)
    scales = fp4_utils.e8m0_to_f32(scale).view(rows, cols // 32)
    return values * scales.repeat_interleave(32, dim=-1)


def _all_gather(tensor):
    """Every rank's rows (all ranks hold the same count here)."""
    parts = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(parts, tensor.contiguous())
    return torch.cat(parts)


@torch.no_grad()
def _reference(x, route_weights, ids, ref_weights, rank, world, net):
    """test_mega_moe_v2's fp32 reference (silu with the swiglu clamp)."""
    H, I, E = net["model_dim"], net["inter_dim"], net["experts"]
    limit = net["swiglu_limit"]
    x_all, weights_all, ids_all = (_all_gather(t) for t in (x, route_weights, ids))
    partial = torch.zeros((x_all.shape[0], H), dtype=torch.float32, device=x.device)
    w1_q, w1_scale, w2_q, w2_scale = ref_weights
    local_experts = E // world
    start = rank * local_experts
    active = torch.unique(ids_all)
    active = active[(active >= start) & (active < start + local_experts)]
    for expert in active.tolist():
        rows, slots = torch.nonzero(ids_all == expert, as_tuple=True)
        local = expert - start
        w1 = _dequant_expert(w1_q[local], w1_scale[local], 2 * I, H)
        w2 = _dequant_expert(w2_q[local], w2_scale[local], H, I)
        inp = x_all[rows].float()
        gate = inp @ w1[:I].T
        up = inp @ w1[I:].T
        if limit > 0:
            gate = gate.clamp(max=limit)
            up = up.clamp(-limit, limit)
        out = (F.silu(gate) * up) @ w2.T * weights_all[rows, slots, None]
        partial.index_add_(0, rows, out)
    dist.all_reduce(partial)
    return partial[rank * x.shape[0] : (rank + 1) * x.shape[0]]


def _time_graph(fn, device, iters, warmup):
    """test_mega_moe_v2's: (mean, max) over the ranks of the ms per graph replay."""
    _barrier()
    fn()
    _barrier()
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        fn()
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    local_ms = start.elapsed_time(end) / iters
    mean_ms = _reduce_float(local_ms, device, dist.ReduceOp.SUM) / dist.get_world_size()
    max_ms = _reduce_float(local_ms, device, dist.ReduceOp.MAX)
    del graph
    return mean_ms, max_ms


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument("--network", choices=NETWORKS, default="v41_flash")
    parser.add_argument("--max-tok-per-rank", type=int, default=8192)
    parser.add_argument("--bs-list", default="1,16,128,512,2048,8192")
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=1, help="timings per path / bs")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--accuracy-max-bs", type=int, default=8192)
    parser.add_argument("--paths", default=",".join(PATHS))
    parser.add_argument("--small-mtpr", type=int, default=128)
    parser.add_argument("--csv", help="rank 0 writes the results here")
    args = parser.parse_args()
    reason = _skip_reason(args.network)
    if reason:
        aiter.logger.warning("bench_mega_moe_gfx942_v2cmp skipped: %s", reason)
        return
    bs_list = [int(v) for v in args.bs_list.split(",")]
    paths = [p for p in args.paths.split(",") if p]
    assert set(paths) <= set(PATHS), paths
    assert max(bs_list) <= args.max_tok_per_rank

    rank, world, device = _setup_dist()
    try:
        from aiter.ops.flydsl.kernels.mega_moe_gfx942 import MegaMoEGfx942

        net = NETWORKS[args.network]
        H, E = net["model_dim"], net["experts"]
        kernel_w, ref_w = _quantize_weights(
            H, net["inter_dim"], E // world, rank, args.seed, device
        )
        w1, w1_scale, w2, w2_scale = MegaMoEGfx942.prepare_weights(
            *kernel_w, w1_gate_up_interleaved=False, quant="a8w4"
        )
        del kernel_w
        x, wts, ids = _make_inputs(
            max(bs_list), H, E, net["topk"], rank, args.seed, device
        )
        common = {
            "rank": rank,
            "world_size": world,
            "model_dim": H,
            "inter_dim": net["inter_dim"],
            "experts": E,
            "topk": net["topk"],
            "quant": "a8w4",
            "w1": w1,
            "w1_scale": w1_scale,
            "w2": w2,
            "w2_scale": w2_scale,
            "max_tok_per_rank": args.max_tok_per_rank,
            "swiglu_limit": net["swiglu_limit"],
        }
        layers = {}
        if "main" in paths:
            layers["main"] = MegaMoEGfx942(
                **common, small_m_enabled=False, large_m_enabled=False
            )
        if {"large", "small"} & set(paths):
            layers["large"] = MegaMoEGfx942(
                **common,
                small_m_mtprs=(args.small_mtpr,),
                small_m_enabled="small" in paths,
                large_m_enabled=True,
            )
        rows = []
        for bs in bs_list:
            lx, lw, li = (
                x[:bs].contiguous(),
                wts[:bs].contiguous(),
                ids[:bs].contiguous(),
            )
            ref = None
            if bs <= args.accuracy_max_bs:
                ref = _reference(lx, lw, li, ref_w, rank, world, net)
            row = {"bs": bs}
            for path in paths:
                if path == "small" and bs > args.small_mtpr:
                    continue
                mega = layers["main" if path == "main" else "large"]
                if path == "small":

                    def call(mega=mega, bs=bs, lx=lx, lw=lw, li=li):
                        return mega.forward_small_m(lx, lw, li, max_tokens=bs)

                else:

                    def call(mega=mega, lx=lx, lw=lw, li=li):
                        return mega.forward(lx, lw, li)

                eager = call().clone()
                _barrier()
                state = {}

                def e2e(call=call, state=state):
                    state["out"] = call()

                times = [
                    _time_graph(e2e, device, args.iters, args.warmup)
                    for _ in range(args.repeat)
                ]
                graph_out = state["out"][:bs].clone()
                same = torch.tensor(
                    int(torch.equal(graph_out, eager)), dtype=torch.int32, device=device
                )
                dist.all_reduce(same, op=dist.ReduceOp.MIN)
                rel = gre = worst = -1.0
                if ref is not None:
                    rel = _rel_l2(eager, ref)
                    gre = _rel_l2(graph_out, ref)
                    worst = _max_row_err(eager, ref, device)
                mean = sorted(t[0] for t in times)
                row[f"{path} relL2"] = rel
                row[f"{path} max_err"] = worst
                row[f"{path} ms"] = mean[len(mean) // 2]
                row[f"{path} ms max"] = sorted(t[1] for t in times)[len(times) // 2]
                if rank == 0:
                    print(
                        f"[MEGA-GFX942] world={world} bs={bs} path={path} "
                        f"relL2={rel:.6f} graph_relL2={gre:.6f} max_err={worst:.3f} "
                        f"graph==eager={bool(int(same))} e2e mean/max ms: "
                        + " ".join(f"{a:.4f}/{b:.4f}" for a, b in times),
                        flush=True,
                    )
            rows.append(row)
        if rank == 0:
            df = pd.DataFrame(rows)
            print(df.to_markdown(index=False, floatfmt=".4f"), flush=True)
            if args.csv:
                df.to_csv(args.csv, index=False)
    finally:
        _barrier()
        ms.shmem_finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
