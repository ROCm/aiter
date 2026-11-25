# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Decode attention output path at TP, four ways, in CUDA graphs:

  unfused  flydsl pa_decode -> per-token FP8 quant -> gemm_a8w8_bpreshuffle -> custom all-reduce + Gemma RMSNorm
  oar      flydsl pa_decode -> per-token FP8 quant -> OprojAllreduceRmsnorm (o_proj + AR + residual + RMSNorm fused)
  oarq     flydsl pa_decode -> OprojAllreduceRmsnorm on bf16 x (the quant fused in too)
  mega     MegaPagedAttention (everything in one persistent kernel)

usage: python bench_oproj_ar_vs_mega.py [--world 4] [--rows 1 8 32 64 128] [--ctx 2048] [--json out.jsonl]
       python bench_oproj_ar_vs_mega.py --trace [--rows ...]   # phase times of oar, oarq and mega instead

Per row count: each chain's time (20 calls a graph, --iters replays, max over ranks), the unfused chain's
cumulative parts, the oar kernels alone, and the max |out| / |residual_out| difference of oar, oarq and mega
from unfused. --trace builds the three persistent kernels with phase stamps and prints, per row count, when
(us after the kernel's first CTA started; latest CTA, rank 0) the last CTA starts, phase 0 (mega: phases 1-2)
is done, x is ready, o_proj is done and the kernel ends.
MiniMax-M3 attention at TP4: 16 query heads and one KV head a rank, head_dim 128, FP8 per-token KV, page 16,
o_proj into hidden 6144. The visible GPUs are the ranks; run with aiter (JIT modules built) and FlyDSL on
PYTHONPATH.
"""

import argparse
import json
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def run(rank, args, port):
    from aiter import dtypes
    from aiter.dist.communication_op import (
        tensor_model_parallel_fused_allreduce_rmsnorm,
    )
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        set_custom_all_reduce,
    )
    from aiter.ops.flydsl.mega_paged_attention import MegaPagedAttention
    from aiter.ops.flydsl.oproj_allreduce_rmsnorm import OprojAllreduceRmsnorm
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_bpreshuffle
    from aiter.ops.quant import per_token_quant_hip, pertoken_quant
    from aiter.ops.shuffle import shuffle_weight

    world = args.world
    dev = torch.device("cuda", rank)
    torch.cuda.set_device(dev)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world, rank=rank, distributed_init_method=f"tcp://127.0.0.1:{port}"
    )
    ensure_model_parallel_initialized(world, 1)
    tp = get_tp_group()
    G, D, PAGE, N, EPS = 16, 128, 16, 6144, 1e-6
    K = G * D
    fp8 = dtypes.fp8
    gen = torch.Generator(device=dev).manual_seed(100 + rank)
    w = torch.randn(N, K, device=dev, generator=gen) * 0.02
    w8, w_scale = pertoken_quant(w, quant_dtype=fp8)
    w_scale = w_scale.reshape(N).contiguous()
    w_shuf = shuffle_weight(w8, layout=(16, 16))
    norm_w = (
        torch.randn(N, device=dev, generator=torch.Generator(device=dev).manual_seed(7))
        * 0.1
    ).to(torch.bfloat16)

    def bench_rows(T):
        max_rows = -(-T // 16) * 16
        mpa = MegaPagedAttention(
            rank=rank,
            world_size=world,
            device=dev,
            group=tp.cpu_group,
            head_dim=D,
            query_group_size=G,
            block_size=PAGE,
            max_rows=max_rows,
            max_context_length=args.ctx,
            hidden=N,
            kv_dtype=fp8,
            softmax_scale=D**-0.5,
            eps=EPS,
            max_tiles_per_task=args.tiles_per_task,
            block_n=args.block_n,
            occupancy=args.grid_per_cu,
            two_shot=not args.one_shot,
            trace=args.trace,
        )
        oar, oarq = (
            OprojAllreduceRmsnorm(
                rank=rank,
                world_size=world,
                device=dev,
                group=tp.cpu_group,
                row_k=K,
                max_rows=max_rows,
                hidden=N,
                eps=EPS,
                x_dtype=x_dtype,
                block_n=args.block_n,
                occupancy=args.grid_per_cu,
                two_shot=not args.one_shot,
                trace=args.trace,
            )
            for x_dtype in (fp8, torch.bfloat16)
        )
        pages_per_seq = args.ctx // PAGE
        num_pages = T * pages_per_seq
        cpu = torch.Generator().manual_seed(11)
        table = (
            torch.randperm(num_pages, generator=cpu)
            .to(torch.int32)
            .reshape(T, pages_per_seq)
            .to(dev)
        )
        context = torch.full((T,), args.ctx, dtype=torch.int32, device=dev)
        residual = torch.randn(T, N, generator=cpu).to(torch.bfloat16).to(dev)
        query = torch.empty(T, G, D, dtype=torch.bfloat16, device=dev).uniform_(
            -1, 1, generator=gen
        )
        key = torch.empty(num_pages, 1, PAGE, D, device=dev).uniform_(
            -1, 1, generator=gen
        )
        value = torch.empty_like(key).uniform_(-1, 1, generator=gen)
        k8, ks = pertoken_quant(key, quant_dtype=fp8)
        v8, vs = pertoken_quant(value, quant_dtype=fp8)
        key_cache = (
            k8.reshape(num_pages, 1, PAGE, D // 16, 16)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
        value_cache = (
            v8.reshape(num_pages, 1, PAGE // 16, 16, D)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )
        k_scale = ks.reshape(num_pages, 1, PAGE).contiguous()
        v_scale = vs.reshape(num_pages, 1, PAGE).contiguous()

        plan = plan_pa_decode(context, 1)
        attn = torch.empty(T, G, D, dtype=torch.bfloat16, device=dev)
        out_m, res_m = (
            torch.empty(T, N, dtype=torch.bfloat16, device=dev) for _ in range(2)
        )
        out_o, res_o = (
            torch.empty(T, N, dtype=torch.bfloat16, device=dev) for _ in range(2)
        )
        out_q, res_q = (
            torch.empty(T, N, dtype=torch.bfloat16, device=dev) for _ in range(2)
        )
        state = {}

        def pa():
            pa_decode(
                attn,
                query,
                key_cache,
                value_cache,
                context,
                table,
                D**-0.5,
                1,
                compute_type=fp8,
                key_scale=k_scale.view(num_pages, 1, PAGE, 1),
                value_scale=v_scale.view(num_pages, 1, PAGE, 1),
                work_plan=plan,
                max_context_length=args.ctx,
            )

        def quant():
            state["xq"], state["xs"] = per_token_quant_hip(
                attn.view(T, K), quant_dtype=fp8
            )

        def gemm():
            state["y"] = gemm_a8w8_bpreshuffle(
                state["xq"],
                w_shuf,
                state["xs"],
                w_scale.view(N, 1),
                dtype=torch.bfloat16,
            )

        def ar():
            state["out"], state["res"] = tensor_model_parallel_fused_allreduce_rmsnorm(
                state["y"], residual, norm_w, EPS, gemma_norm=True
            )

        def oar_only():
            oar(
                state["xq"],
                state["xs"],
                w_shuf,
                w_scale,
                residual,
                norm_w,
                out=out_o,
                residual_out=res_o,
            )

        def oarq_only():
            oarq(
                attn.view(T, K),
                None,
                w_shuf,
                w_scale,
                residual,
                norm_w,
                out=out_q,
                residual_out=res_q,
            )

        def unfused():
            pa()
            quant()
            gemm()
            ar()

        def partial():
            pa()
            quant()
            oar_only()

        def partial_q():
            pa()
            oarq_only()

        def mega():
            mpa(
                query,
                key_cache,
                value_cache,
                k_scale,
                v_scale,
                table,
                context,
                w_shuf,
                w_scale,
                residual,
                norm_w,
                out=out_m,
                residual_out=res_m,
            )

        for _ in range(3):
            unfused()
            partial()
            partial_q()
            mega()
        torch.cuda.synchronize()
        dist.barrier()

        def timed(fns):
            graph = torch.cuda.CUDAGraph()
            with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
                for _ in range(args.inner):
                    for fn in fns:
                        fn()
            torch.cuda.synchronize()
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            dist.barrier()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            for _ in range(args.iters):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            us = start.elapsed_time(end) * 1000 / (args.iters * args.inner)
            t = torch.tensor([us], device=dev)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            return t.item()

        if args.trace:
            # Phase stamps of the last of 20 calls a graph replay, like the timed runs.
            phases = {}
            for name, fns, stamps in (
                ("oar", [partial], oar.trace),
                ("oarq", [partial_q], oarq.trace),
                ("mega", [mega], mpa.trace),
            ):
                timed(fns)
                t = stamps[:, :5].double().cpu() / 100.0
                start = t[:, 0].min()
                ready = t[:, 2][t[:, 2] > 0]
                phases[name] = [
                    (t[:, 0].max() - start).item(),
                    (t[:, 1].max() - start).item(),
                    (ready.max() - start).item() if ready.numel() else float("nan"),
                    (t[:, 3].max() - start).item(),
                    (t[:, 4].max() - start).item(),
                ]
            if rank == 0:
                print(
                    f"T={T:3d} | "
                    + " | ".join(
                        f"{n} last-start {p[0]:4.1f} p0 {p[1]:5.1f} x-ready {p[2]:5.1f} o_proj {p[3]:5.1f} "
                        f"end {p[4]:5.1f}"
                        for n, p in phases.items()
                    ),
                    flush=True,
                )
                if args.json:
                    with open(args.json, "a") as f:
                        f.write(
                            json.dumps(
                                {
                                    "rows": T,
                                    "ctx": args.ctx,
                                    "world": world,
                                    "trace": phases,
                                }
                            )
                            + "\n"
                        )
            mpa.close()
            oar.close()
            oarq.close()
            return

        # Alternate the chains over --rounds so drift hits them alike; keep each one's best round.
        best = {}
        for _ in range(args.rounds):
            for name, fns in (
                ("pa", [pa]),
                ("quant", [pa, quant]),
                ("gemm", [pa, quant, gemm]),
                ("unfused", [unfused]),
                ("oar_kernel", [oar_only]),
                ("partial", [partial]),
                ("oarq_kernel", [oarq_only]),
                ("partial_q", [partial_q]),
                ("mega", [mega]),
            ):
                us = timed(fns)
                best[name] = min(best.get(name, us), us)
        r = best
        # Outputs of one plain (eager) call of each chain, from the same inputs.
        unfused()
        partial()
        partial_q()
        mega()
        torch.cuda.synchronize()

        def diff(a, b):
            return (a.float() - b.float()).abs().max().item()

        d_o, d_or = diff(state["out"], out_o), diff(state["res"], res_o)
        d_q, d_qr = diff(state["out"], out_q), diff(state["res"], res_q)
        d_m, d_mr = diff(state["out"], out_m), diff(state["res"], res_m)
        row = {
            "rows": T,
            "ctx": args.ctx,
            "world": world,
            "unfused_us": r["unfused"],
            "partial_us": r["partial"],
            "partial_q_us": r["partial_q"],
            "mega_us": r["mega"],
            "oar_kernel_us": r["oar_kernel"],
            "oarq_kernel_us": r["oarq_kernel"],
            "pa_us": r["pa"],
            "quant_us": r["quant"] - r["pa"],
            "gemm_us": r["gemm"] - r["quant"],
            "ar_norm_us": r["unfused"] - r["gemm"],
            "partial_vs_unfused": r["unfused"] / r["partial"],
            "partial_q_vs_unfused": r["unfused"] / r["partial_q"],
            "mega_vs_unfused": r["unfused"] / r["mega"],
            "mega_vs_partial": r["partial"] / r["mega"],
            "mega_vs_partial_q": r["partial_q"] / r["mega"],
            "oar_out_diff": d_o,
            "oar_res_diff": d_or,
            "oarq_out_diff": d_q,
            "oarq_res_diff": d_qr,
            "mega_out_diff": d_m,
            "mega_res_diff": d_mr,
        }
        if rank == 0:
            if args.json:
                with open(args.json, "a") as f:
                    f.write(json.dumps(row) + "\n")
            print(
                f"T={T:3d} | unfused {r['unfused']:6.1f} (pa {r['pa']:5.1f} quant {row['quant_us']:4.1f} "
                f"gemm {row['gemm_us']:5.1f} ar+norm {row['ar_norm_us']:5.1f}) | oar chain {r['partial']:6.1f} "
                f"(kernel {r['oar_kernel']:5.1f}) {row['partial_vs_unfused']:.2f}x | oarq chain "
                f"{r['partial_q']:6.1f} (kernel {r['oarq_kernel']:5.1f}) {row['partial_q_vs_unfused']:.2f}x | "
                f"mega {r['mega']:6.1f} {row['mega_vs_unfused']:.2f}x | max|diff| vs unfused: "
                f"oar {d_o:.2e}/{d_or:.2e} oarq {d_q:.2e}/{d_qr:.2e} mega {d_m:.2e}/{d_mr:.2e}",
                flush=True,
            )
        mpa.close()
        oar.close()
        oarq.close()

    for T in args.rows:
        bench_rows(T)
    dist.barrier()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--json")
    p.add_argument("--world", type=int, default=4)
    p.add_argument(
        "--rows", type=int, nargs="+", default=[1, 8, 16, 32, 48, 64, 96, 128]
    )
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--inner", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--tiles-per-task", type=int, default=4)
    p.add_argument("--block-n", type=int, default=48)
    p.add_argument("--grid-per-cu", type=int, default=2)
    p.add_argument("--one-shot", action="store_true")
    p.add_argument("--trace", action="store_true")
    args = p.parse_args()
    port = 29500 + os.getpid() % 1000
    mp.spawn(run, args=(args, port), nprocs=args.world, join=True)
