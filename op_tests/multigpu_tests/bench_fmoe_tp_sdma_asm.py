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
    from aiter.ops.asm import (
        EPI_MODES,
        TpAsmContext,
        TpMoeCall,
        quantize_moe_input,
        tp_moe_layer_asm,
    )
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
    if epi not in EPI_MODES:
        raise ValueError(f"unsupported epi_mode={epi!r}, expected one of {EPI_MODES}")
    tm_arg = None if cfg["tile_m"] == "auto" else int(cfg["tile_m"])
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

    is_add = epi.startswith("add_rmsnorm")
    is_norm = not epi.startswith("raw_bf16")
    is_fp4_o, is_fp8_o = epi.endswith("_fp4"), epi.endswith("_fp8")

    def alloc_out():
        if is_fp4_o:
            o = torch.empty((T_, D // 2), dtype=ai.dtypes.fp4x2, device=dev)
            s = torch.empty((T_, D // 32), dtype=ai.dtypes.fp8_e8m0, device=dev)
        elif is_fp8_o:
            o = torch.empty((T_, D), dtype=ai.dtypes.fp8, device=dev)
            s = torch.empty((T_, D // 32), dtype=ai.dtypes.fp8_e8m0, device=dev)
        else:
            o, s = torch.empty_like(x), None
        return o, s, (torch.empty_like(x) if is_add else None)

    qr = FlyQuickAllReduce(group=gloo, device=dev, rank=rank, world_size=tp)
    qr_norm = is_add and qr.supports_rmsnorm(D, T_ * D * 2)
    if qr_norm:
        qr.compile_rmsnorm(D)
    qr_f4 = (
        torch.empty(qr.mxfp4_scale_shape(T_, D), dtype=torch.uint8, device=dev)
        if epi == "add_rmsnorm_fp4" and qr_norm and getattr(qr, "supports_mxfp4", False)
        else None
    )
    ctx = TpAsmContext(gloo, dev, T_, D, k, E, I)
    ar_s = torch.empty_like(x)
    (out_s, sc_s, res_s), (out_a, sc_a, res_a), (out_p, sc_p, res_p) = (
        alloc_out() for _ in range(3)
    )
    res_in, nw_in = (res if is_add else None), (nw if is_norm else None)
    xq, xs = quantize_moe_input(x, a_dt)
    call = TpMoeCall(x, topk_ids, topk_w, w1, w2, w1_ss, w2_ss, res_in, nw_in, EPS)
    call_pq = TpMoeCall(
        xq, topk_ids, topk_w, w1, w2, w1_ss, w2_ss, res_in, nw_in, EPS, x_scale=xs
    )

    def run_stock() -> None:
        mo = fm.fused_moe(x, w1_st, w2, topk_w, topk_ids, **moe_kw)
        if epi == "raw_bf16":
            qr.allreduce(mo, out_s)
        elif epi == "rmsnorm_only":
            qr.allreduce(mo, ar_s)
            out_s.copy_(ai.rmsnorm2d_fwd(ar_s, nw, EPS))
        elif epi == "add_rmsnorm":
            if qr_norm:
                qr.allreduce_rmsnorm(mo, res, nw, EPS, out_s, res_s)
            else:
                qr.allreduce(mo, ar_s)
                ai.rmsnorm2d_fwd_with_add(out_s, ar_s, res, res_s, nw, EPS)
        elif is_add and qr_f4 is not None:
            qr.allreduce_rmsnorm_mxfp4(
                mo, res, nw, EPS, ar_s, res_s, out_s.view(torch.uint8), qr_f4
            )
        elif is_add:
            qr.allreduce(mo, ar_s)
            ai.add_rmsnorm_quant(out_s, ar_s, res, res_s, sc_s, nw, EPS, 32, False)
        else:
            qr.allreduce(mo, ar_s)
            ai.rmsnorm_quant(out_s, ar_s, sc_s, nw, EPS, 32, False)

    base_kw = {"a_dtype": a_dt, "act": act, "tile_m": tm_arg}
    run_asm = lambda: tp_moe_layer_asm(
        call, ctx, epi, out_a, sc_a, residual_out=res_a, **base_kw
    )
    run_pq = lambda: tp_moe_layer_asm(
        call_pq, ctx, epi, out_p, sc_p, residual_out=res_p, **base_kw
    )
    arms = {
        "stock": (run_stock, out_s, qr_f4 if qr_f4 is not None else sc_s),
        "fmoe_tp_sdma": (run_asm, out_a, sc_a),
        "fmoe_tp_sdma_prequant": (run_pq, out_p, sc_p),
    }
    graphs, samples = {}, {a: [] for a in arms}
    ev = lambda: torch.cuda.Event(enable_timing=True)
    e0, e1, dt_t = ev(), ev(), torch.empty(1, dtype=torch.float64, device=dev)
    try:
        for name, (fn, _, _) in arms.items():
            for _ in range(2):
                fn()
            sync()
            graphs[name] = g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                fn()
        names, iters = list(arms), int(cfg["iters"])
        for r in range(int(cfg["rounds"])):
            rot = r % len(names)
            for name in names[rot:] + names[:rot]:
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
            tm_tag = f"bm{cfg['tile_m']}"
            hdr = f"arm ({cfg['shape']} T{T_} TP{tp} {epi} {tm_tag})"
            print(f"\n| {hdr} | med us | p95 us | min us | vs stock | ok |")
            print("|---|---:|---:|---:|---:|---|")
            for a, (_, buf, sc) in arms.items():
                xs_s, bs = sorted(samples[a]), samples["stock"]
                med, p95 = statistics.median(xs_s), xs_s[round(0.95 * (len(xs_s) - 1))]
                m = statistics.median(v / b for v, b in zip(samples[a], bs))
                vs = "-" if a == "stock" else f"{100.0 * (m - 1.0):+.1f}%"
                ok = (
                    True
                    if buf.dtype == ai.dtypes.fp4x2
                    else bool(
                        torch.isfinite(
                            buf.float() if buf.element_size() == 1 else buf
                        ).all()
                    )
                )
                if sc is not None:
                    ok = ok and bool((sc.view(torch.uint8) != 0xFF).all())
                print(f"| {a} | {med:.1f} | {p95:.1f} | {xs_s[0]:.1f} | {vs} | {ok} |")
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
        ("--tile-m", {"choices": ("auto", "32", "64"), "default": "auto"}),
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
