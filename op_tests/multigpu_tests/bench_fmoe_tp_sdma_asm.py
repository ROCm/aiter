# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""CUDA-Graph benchmark comparing stock ``fused_moe`` + QuickAllReduce vs ``tp_moe_layer_asm``."""

import argparse
import glob
import os
import socket
import statistics
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

EPS = 1e-6
PRESETS = {
    "q38": (512, 10, 8192, 256, 256, "add_rmsnorm", "fp4", "silu"),
    "flash_next": (512, 10, 2560, 384, 256, "raw_bf16", "fp4", "silu"),
    "v4_flash": (256, 6, 4096, 256, 256, "raw_bf16", "fp4", "silu"),
    "k3": (896, 16, 3584, 384, 384, "rmsnorm_only", "fp8", "situ"),
}


def _bench_rank(rank: int, tp: int, init_m: str, cfg: dict) -> None:
    sys.path.insert(0, cfg["root"])
    import aiter as ai
    import aiter.fused_moe as fm
    from aiter.ops.asm import TpAsmContext, TpMoeCall, tp_moe_layer_asm
    from aiter.ops.flydsl import FlyQuickAllReduce
    from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", init_m, world_size=tp, rank=rank, device_id=dev)
    gloo = dist.new_group(backend="gloo")
    sync = lambda: (torch.cuda.synchronize(dev), dist.barrier(group=gloo))
    E, k, D, i2, i_o, def_epi, a_dt, act = PRESETS[cfg["shape"]]
    T_, I = int(cfg["tokens"]), (i2 if tp == 2 else i_o)
    epi = def_epi if cfg["epi_mode"] == "auto" else cfg["epi_mode"]
    gen = torch.Generator(device=dev).manual_seed(77 + 37 * rank + D)
    rn = lambda *s: torch.randn(*s, generator=gen, device=dev, dtype=torch.bfloat16)
    x, res, nw = rn(T_, D), rn(T_, D) * 0.5, 1.0 + 0.2 * rn(D)
    topk_w, topk_ids = fm.fused_topk(x, rn(T_, E), k, True)

    def q4(r: int, c: int, sc: int):
        q, s = ai.per_1x32_f4_quant(rn(E, r, c) * (sc**-0.5))
        return q, s.view(E * r, c // 32).view(torch.float8_e8m0fnu)

    (w1_q, w1_s), (w2_q, w2_s) = q4(2 * I, D, D), q4(D, I, I * tp)
    w1, w2 = moe_shuffle_weight(w1_q), moe_shuffle_weight(w2_q)
    w1_ss, w2_ss = moe_shuffle_scale(w1_s, E), moe_shuffle_scale(w2_s, E)
    if a_dt == "fp8":
        os.environ["AITER_SITUV2_A8W4"] = "1"
        w1_st = moe_shuffle_weight(w1_q, E, True, True)
        w1_ss_st = moe_shuffle_scale(w1_s, E, True, True)
        moe_kw = {"activation": ai.ActivationType.Situv2, "beta": 4.0}
        moe_kw.update(linear_beta=25.0, gate_mode=fm.GateMode.INTERLEAVE)
    else:
        w1_st, w1_ss_st, moe_kw = w1, w1_ss, {"activation": ai.ActivationType.Silu}
    moe_kw.update(quant_type=ai.QuantType.per_1x32, w1_scale=w1_ss_st, w2_scale=w2_ss)

    qr = FlyQuickAllReduce(group=gloo, device=dev, rank=rank, world_size=tp)
    if epi == "add_rmsnorm":
        qr.compile_rmsnorm(D)
    ctx = TpAsmContext(gloo, dev, T_, D, k, E, I)
    out_s, res_s, ar_s, out_a = (torch.empty_like(x) for _ in range(4))
    res_a = torch.empty_like(x) if epi == "add_rmsnorm" else None
    call = TpMoeCall(x, topk_ids, topk_w, w1, w2, w1_ss, w2_ss, res, nw, EPS)

    def run_stock() -> None:
        mo = fm.fused_moe(x, w1_st, w2, topk_w, topk_ids, **moe_kw)
        if epi == "raw_bf16":
            qr.allreduce(mo, out_s)
        elif epi == "rmsnorm_only":
            qr.allreduce(mo, ar_s)
            out_s.copy_(ai.rmsnorm2d_fwd(ar_s, nw, EPS))
        else:
            qr.allreduce_rmsnorm(mo, res, nw, EPS, out_s, res_s)

    asm_kw = {"residual_out": res_a, "a_dtype": a_dt, "act": act}
    run_asm = lambda: tp_moe_layer_asm(call, ctx, epi, out_a, **asm_kw)
    arms = {"stock": (run_stock, out_s), "fmoe_tp_sdma": (run_asm, out_a)}
    graphs, samples = {}, {a: [] for a in arms}
    ev = lambda: torch.cuda.Event(enable_timing=True)
    e0, e1, dt_t = ev(), ev(), torch.empty(1, dtype=torch.float64, device=dev)
    try:
        for name, (fn, _) in arms.items():
            for _ in range(2):
                fn()
            sync()
            graphs[name] = g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
        names, iters = list(arms), int(cfg["iters"])
        for r in range(int(cfg["rounds"])):
            for name in names[r % 2 :] + names[: r % 2]:
                sync()
                e0.record()
                for _ in range(iters):
                    graphs[name].replay()
                e1.record()
                torch.cuda.synchronize(dev)
                dt_t[0] = e0.elapsed_time(e1) * 1000.0 / iters
                dist.all_reduce(dt_t, op=dist.ReduceOp.MAX)
                samples[name].append(float(dt_t.item()))
        ctx.check_error()
        if rank == 0:
            hdr = f"arm ({cfg['shape']} T{T_} TP{tp} {epi})"
            print(f"\n| {hdr} | med us | p95 us | min us | vs stock | ok |")
            print("|---|---:|---:|---:|---:|---|")
            for a, (_, buf) in arms.items():
                xs, bs = sorted(samples[a]), samples["stock"]
                med, p95 = statistics.median(xs), xs[round(0.95 * (len(xs) - 1))]
                m = statistics.median(v / b for v, b in zip(samples[a], bs))
                vs = "-" if a == "stock" else f"{100.0 * (m - 1.0):+.1f}%"
                ok = bool(torch.isfinite(buf).all())
                print(f"| {a} | {med:.1f} | {p95:.1f} | {xs[0]:.1f} | {vs} | {ok} |")
        del graphs
        sync()
    finally:
        ctx.close()
        qr.close()
        dist.destroy_process_group()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for f, kw in (
        ("--tp", {"type": int, "choices": (2, 4, 8), "default": 2}),
        ("--shape", {"choices": list(PRESETS), "default": "q38"}),
        ("--tokens", {"type": int, "default": 16384}),
        ("--epi-mode", {"default": "auto"}),
        ("--rounds", {"type": int, "default": 7}),
        ("--iters", {"type": int, "default": 3}),
    ):
        ap.add_argument(f, **kw)
    a, root = ap.parse_args(), os.path.abspath(f"{__file__}/../../..")
    if not os.environ.get("AITER_CONFIG_FMOE"):
        pat = f"{root}/aiter/configs/model_configs/*tuned_fmoe*.csv"
        mc = sorted(p for p in glob.glob(pat) if "untuned" not in os.path.basename(p))
        base = f"{root}/aiter/configs/tuned_fmoe.csv"
        os.environ["AITER_CONFIG_FMOE"] = os.pathsep.join([base, *mc])
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        init = f"tcp://127.0.0.1:{sock.getsockname()[1]}"
    mp.spawn(_bench_rank, args=(a.tp, init, {**vars(a), "root": root}), nprocs=a.tp)


if __name__ == "__main__":
    main()
