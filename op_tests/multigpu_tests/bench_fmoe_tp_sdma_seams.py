# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Paired CUDA-Graph benchmark for Upgraded Regular-TP MX seams (L0 vs S1 vs S3 vs S1+S3)."""

import argparse
import os
import socket
import statistics
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_REPO_ROOT = os.path.abspath(f"{__file__}/../../..")
sys.path.insert(0, _REPO_ROOT)

EPS = 1e-6
# (E, k, D, i_tp2, i_tp4+, bf16_epi, q_epi, a_dt, act, n_downstream)
PRESETS = {
    "q38": (
        512,
        10,
        8192,
        256,
        256,
        "add_rmsnorm",
        "add_rmsnorm_fp4",
        "fp4",
        "silu",
        2048,
    ),
    "k3": (
        896,
        16,
        3584,
        384,
        384,
        "rmsnorm_only",
        "rmsnorm_only_fp8",
        "fp8",
        "situ",
        1792,
    ),
    "v4_flash": (
        256,
        6,
        4096,
        256,
        256,
        "add_rmsnorm",
        "add_rmsnorm_fp4",
        "fp4",
        "silu",
        2048,
    ),
    "flash_next": (
        512,
        10,
        2560,
        384,
        256,
        "add_rmsnorm",
        "add_rmsnorm_fp4",
        "fp4",
        "silu",
        1536,
    ),
}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _bench_rank(rank: int, tp: int, init_m: str, cfg: dict) -> None:
    import aiter as ai
    import aiter.fused_moe as fm
    from aiter import dtypes
    from aiter.ops.asm import (
        TpAsmContext,
        TpMoeCall,
        prepare_downstream_mx_inputs,
        quantize_moe_input,
        tp_moe_layer_asm,
    )
    from aiter.ops.flydsl.quick_allreduce import FlyQuickAllReduceRMSNorm
    from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight, shuffle_weight

    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", init_m, world_size=tp, rank=rank, device_id=dev)
    gloo = dist.new_group(backend="gloo")
    sync = lambda: (torch.cuda.synchronize(dev), dist.barrier(group=gloo))

    E, k, D, i2, i_o, bf16_epi, q_epi, a_dt, act, N_dn = PRESETS[cfg["shape"]]
    T_ = int(cfg["tokens"])
    I = i2 if tp == 2 else i_o
    is_add = bf16_epi.startswith("add_rmsnorm")

    gen_r = torch.Generator(device=dev).manual_seed(77 + 37 * rank + D)
    gen_s = torch.Generator(device=dev).manual_seed(888 + D)
    inp = torch.randn(T_, D, generator=gen_r, device=dev, dtype=dtypes.bf16) * 0.4
    res0 = torch.randn(T_, D, generator=gen_s, device=dev, dtype=dtypes.bf16) * 0.4
    res1 = torch.randn(T_, D, generator=gen_s, device=dev, dtype=dtypes.bf16) * 0.4
    nw0 = 1.0 + 0.2 * torch.randn(D, generator=gen_s, device=dev, dtype=dtypes.bf16)
    nw1 = 1.0 + 0.2 * torch.randn(D, generator=gen_s, device=dev, dtype=dtypes.bf16)
    rw = torch.randn(E, D, generator=gen_s, device=dev, dtype=dtypes.bf16) * (D**-0.5)

    def q4(r: int, c: int, sc: int):
        q, s = ai.per_1x32_f4_quant(
            torch.randn(E, r, c, generator=gen_r, device=dev, dtype=dtypes.bf16)
            * (sc**-0.5)
        )
        return q, s.view(E * r, c // 32).view(torch.float8_e8m0fnu)

    w1_q, w1_s = q4(2 * I, D, D)
    w2_q, w2_s = q4(D, I, I * tp)
    w1, w2 = moe_shuffle_weight(w1_q), moe_shuffle_weight(w2_q)
    w1_ss, w2_ss = moe_shuffle_scale(w1_s, E), moe_shuffle_scale(w2_s, E)

    w_dn = torch.randn(N_dn, D, generator=gen_s, device=dev, dtype=dtypes.bf16) * (
        D**-0.5
    )
    if a_dt == "fp4":
        wd_q, wd_s = ai.per_1x32_f4_quant_hip(w_dn, shuffle=True)
    else:
        wd_q, wd_s = ai.per_1x32_mx_quant_hip(
            w_dn, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
        wd_s = wd_s.view(N_dn, D // 32)
    wd_sh = shuffle_weight(wd_q, (16, 16))

    qr = FlyQuickAllReduceRMSNorm(
        group=gloo, device=dev, rank=rank, world_size=tp, algorithm="mesh"
    )
    qr.preload(D)
    qr.preload(D, quant_dtype=a_dt, scale_shuffle=False, return_norm_bf16=True)
    ctx = TpAsmContext(gloo, dev, T_, D, k, E, I)

    # Pre-allocate buffers per arm
    q_cols = D // 2 if a_dt == "fp4" else D
    q_dt = dtypes.fp4x2 if a_dt == "fp4" else dtypes.fp8
    x_bf16 = torch.empty((T_, D), dtype=dtypes.bf16, device=dev)
    res0_out = torch.empty((T_, D), dtype=dtypes.bf16, device=dev)
    xq_s3 = torch.empty((T_, q_cols), dtype=q_dt, device=dev)
    xs_s3 = torch.empty((T_, D // 32), dtype=dtypes.fp8_e8m0, device=dev)
    moe_bf16 = torch.empty((T_, D), dtype=dtypes.bf16, device=dev)
    moe_q = torch.empty((T_, q_cols), dtype=q_dt, device=dev)
    moe_s = torch.empty((T_, D // 32), dtype=dtypes.fp8_e8m0, device=dev)
    res1_out = torch.empty((T_, D), dtype=dtypes.bf16, device=dev) if is_add else None
    flush_buf = torch.empty(32 * 1024 * 1024, dtype=torch.int32, device=dev)  # 128 MB

    def run_downstream_from_bf16(t_bf16: torch.Tensor) -> torch.Tensor:
        if a_dt == "fp4":
            dq, ds = ai.per_1x32_f4_quant_hip(t_bf16, shuffle=True)
            return ai.gemm_a4w4(dq, wd_sh, ds, wd_s, dtype=dtypes.bf16)
        dq, ds = ai.per_1x32_mx_quant_hip(
            t_bf16, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
        return ai.gemm_a8w8_mxfp8_bpreshuffle(
            dq, wd_sh, ds.view(T_, D // 32), wd_s, dtype=dtypes.bf16
        )

    def run_downstream_from_mx(q_t: torch.Tensor, s_t: torch.Tensor) -> torch.Tensor:
        dq, ds = prepare_downstream_mx_inputs(q_t, s_t, a_dt)
        if a_dt == "fp4":
            return ai.gemm_a4w4(dq, wd_sh, ds, wd_s, dtype=dtypes.bf16)
        return ai.gemm_a8w8_mxfp8_bpreshuffle(dq, wd_sh, ds, wd_s, dtype=dtypes.bf16)

    def arm_l0() -> torch.Tensor:
        qr.allreduce_rmsnorm(inp, res0, nw0, EPS, out=x_bf16, residual_out=res0_out)
        topk_w, topk_ids = fm.fused_topk(x_bf16, x_bf16 @ rw.t(), k, True)
        xq, xs = quantize_moe_input(x_bf16, a_dt)
        call = TpMoeCall(
            xq,
            topk_ids,
            topk_w,
            w1,
            w2,
            w1_ss,
            w2_ss,
            res1 if is_add else None,
            nw1,
            EPS,
            x_scale=xs,
        )
        m = tp_moe_layer_asm(
            call,
            ctx,
            bf16_epi,
            moe_bf16,
            None,
            residual_out=res1_out,
            a_dtype=a_dt,
            act=act,
        )
        return run_downstream_from_bf16(m.out)

    def arm_s1() -> torch.Tensor:
        qr.allreduce_rmsnorm(inp, res0, nw0, EPS, out=x_bf16, residual_out=res0_out)
        topk_w, topk_ids = fm.fused_topk(x_bf16, x_bf16 @ rw.t(), k, True)
        xq, xs = quantize_moe_input(x_bf16, a_dt)
        call = TpMoeCall(
            xq,
            topk_ids,
            topk_w,
            w1,
            w2,
            w1_ss,
            w2_ss,
            res1 if is_add else None,
            nw1,
            EPS,
            x_scale=xs,
        )
        m = tp_moe_layer_asm(
            call, ctx, q_epi, moe_q, moe_s, residual_out=res1_out, a_dtype=a_dt, act=act
        )
        return run_downstream_from_mx(m.out, m.out_scale)

    def arm_s3() -> torch.Tensor:
        qr.allreduce_rmsnorm_mx(
            inp,
            res0,
            nw0,
            EPS,
            xq_s3,
            xs_s3,
            quant_dtype=a_dt,
            scale_shuffle=False,
            return_norm_bf16=True,
            out=x_bf16,
            residual_out=res0_out,
        )
        topk_w, topk_ids = fm.fused_topk(x_bf16, x_bf16 @ rw.t(), k, True)
        call = TpMoeCall(
            xq_s3,
            topk_ids,
            topk_w,
            w1,
            w2,
            w1_ss,
            w2_ss,
            res1 if is_add else None,
            nw1,
            EPS,
            x_scale=xs_s3,
        )
        m = tp_moe_layer_asm(
            call,
            ctx,
            bf16_epi,
            moe_bf16,
            None,
            residual_out=res1_out,
            a_dtype=a_dt,
            act=act,
        )
        return run_downstream_from_bf16(m.out)

    def arm_s1_s3() -> torch.Tensor:
        qr.allreduce_rmsnorm_mx(
            inp,
            res0,
            nw0,
            EPS,
            xq_s3,
            xs_s3,
            quant_dtype=a_dt,
            scale_shuffle=False,
            return_norm_bf16=True,
            out=x_bf16,
            residual_out=res0_out,
        )
        topk_w, topk_ids = fm.fused_topk(x_bf16, x_bf16 @ rw.t(), k, True)
        call = TpMoeCall(
            xq_s3,
            topk_ids,
            topk_w,
            w1,
            w2,
            w1_ss,
            w2_ss,
            res1 if is_add else None,
            nw1,
            EPS,
            x_scale=xs_s3,
        )
        m = tp_moe_layer_asm(
            call, ctx, q_epi, moe_q, moe_s, residual_out=res1_out, a_dtype=a_dt, act=act
        )
        return run_downstream_from_mx(m.out, m.out_scale)

    arms = {
        "L0_bf16_seams": arm_l0,
        "S1_only": arm_s1,
        "S3_only": arm_s3,
        "S1_plus_S3": arm_s1_s3,
    }
    graphs, samples = {}, {a: [] for a in arms}
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    dt_t = torch.empty(1, dtype=torch.float64, device=dev)

    try:
        for name, fn in arms.items():
            for _ in range(2):
                fn()
            sync()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
            graphs[name] = g

        names = list(arms)
        iters = int(cfg["iters"])
        rounds = int(cfg["rounds"])
        for r in range(rounds):
            rot = r % len(names)
            for name in names[rot:] + names[:rot]:
                rep_ms = []
                for _ in range(iters):
                    flush_buf.zero_()
                    sync()
                    e0.record()
                    graphs[name].replay()
                    e1.record()
                    torch.cuda.synchronize(dev)
                    rep_ms.append(e0.elapsed_time(e1))
                dt_t.fill_(statistics.median(rep_ms) * 1000.0)
                dist.all_reduce(dt_t, op=dist.ReduceOp.MAX, group=gloo)
                samples[name].append(float(dt_t.item()))

        if rank == 0:
            l0_s = samples["L0_bf16_seams"]
            med = {n: statistics.median(v) for n, v in samples.items()}
            d_s1 = statistics.median([b - a for a, b in zip(l0_s, samples["S1_only"])])
            d_s3 = statistics.median([b - a for a, b in zip(l0_s, samples["S3_only"])])
            d_both = statistics.median(
                [b - a for a, b in zip(l0_s, samples["S1_plus_S3"])]
            )
            pct = (
                100.0
                * (med["L0_bf16_seams"] - med["S1_plus_S3"])
                / med["L0_bf16_seams"]
            )
            print(
                f"RESULT shape={cfg['shape']} tp={tp} T={T_} | "
                f"L0={med['L0_bf16_seams']:.1f}us "
                f"S1={med['S1_only']:.1f}us ({d_s1:+.1f}us) "
                f"S3={med['S3_only']:.1f}us ({d_s3:+.1f}us) "
                f"S1+S3={med['S1_plus_S3']:.1f}us ({d_both:+.1f}us, {pct:+.1f}%)"
            )
        ctx.check_error()
    finally:
        for g in graphs.values():
            g.reset()
        ctx.close()
        qr.close()
        sync()
        dist.destroy_process_group()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, default=4, choices=(2, 4))
    p.add_argument("--shape", default="q38", choices=tuple(PRESETS))
    p.add_argument("--tokens", type=int, default=4096)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--i-scale", default="shard", choices=("shard",))
    args = p.parse_args()
    cfg = vars(args)
    mp.spawn(
        _bench_rank,
        args=(args.tp, f"tcp://127.0.0.1:{_free_port()}", cfg),
        nprocs=args.tp,
        join=True,
    )


if __name__ == "__main__":
    main()
