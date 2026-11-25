#!/usr/bin/env python3
"""Reproducer: gfx1250 MXFP8 GEMM (aiter/ops/flydsl/kernels/gemm_a8w8_gfx1250.py) C-store race.

The kernel stages the bf16 C tile in LDS with ds_store (over the last k-tiles' FP8 A operand),
passes s_wait_dscnt 0 + s_barrier, and each wave TDM-stores (tensor_store_from_lds) a box of
whole rows. Now and then a 16-byte chunk leaves with what LDS held before the ds_store: FP8
activation bytes, which read as bf16 are 1e24..1e38 (never NaN/Inf). It shows up when another
kernel runs beside the GEMM on a second HIP graph queue (MiniMax-M3.1 TP4 shared-expert
down_proj next to the routed MoE).

Each graph replay marks the outputs, forks a background GEMM onto stream B, runs --na small
GEMMs (the M3.1 shared down_proj, N=6144 K=768) on stream A, joins, and counts outputs that
differ from a clean eager run. All scales are 1.0 and |A| is 64..448, so a stale A chunk is
unmistakable.
  python repro_gemm_a8w8_tdm_race.py                                  # unfixed: bad > 0
  DEBUG_HIP_FORCE_GRAPH_QUEUES=1 python repro_gemm_a8w8_tdm_race.py   # one graph queue: 0
  python repro_gemm_a8w8_tdm_race.py --bg none                        # nothing beside it: 0
"""

import argparse
import time

import torch
from aiter.utility import dtypes
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.flydsl.batched_gemm_a8w8_gfx1250 import (
    pick_bmm_mx32_kernel_name,
    run_bmm_a8w8_mxfp8_32_gfx1250,
)
from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
    run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250,
)


def bmm(x8, wq, xs, ws, out, kernel_name=None):
    """[M,1,K] x [1,N,K] -> [M,1,N]; a flydsl_mxfp8_32_bpreshuffle_* name (e.g. split-K) runs the 2-D GEMM."""
    if kernel_name is None or kernel_name.startswith("flydsl_bmm"):
        return run_bmm_a8w8_mxfp8_32_gfx1250(
            x8, wq, xs, ws, out, kernel_name=kernel_name
        )
    e8m0 = dtypes.fp8_e8m0
    run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250(
        x8[:, 0], wq[0], xs[:, 0].view(e8m0), ws[0].view(e8m0), out[:, 0], kernel_name
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", default="8,64", help="GEMM rows (decode tokens)")
    p.add_argument("--n", type=int, default=6144)
    p.add_argument("--k", type=int, default=768)
    # per --m entry (or one for all): t16x64x128 is the M3.1 TP4 shared down_proj kernel at M=8 (the
    # one that fired in the model), t32x128x128 the one at M=128; "auto" = aiter's heuristic pick
    p.add_argument("--kernel", default="t16x64x128_mw1_nw4_nb4,t32x128x128_mw1_nw4_nb4")
    p.add_argument(
        "--bg",
        default="moe",
        choices=["moe", "gemm", "mm", "none"],
        help="kernel on stream B",
    )
    p.add_argument(
        "--moe-t", type=int, default=8, help="tokens of the routed MXFP4 MoE (--bg moe)"
    )
    p.add_argument("--bg-m", type=int, default=4096, help="rows of the background GEMM")
    p.add_argument("--bg-n", type=int, default=6144)
    p.add_argument("--bg-k", type=int, default=768)
    p.add_argument(
        "--bg-kernel",
        default="flydsl_bmm_mxfp8_32_wmma_t32x64x128_mw1_nw4_nb4_sk1_cm1_cn1",
    )
    p.add_argument("--na", type=int, default=64, help="small GEMM launches per replay")
    p.add_argument("--nb", type=int, default=6, help="background launches per replay")
    p.add_argument(
        "--nw",
        type=int,
        default=16,
        help="distinct A/W operand sets cycled over the launches",
    )
    p.add_argument("--replays", type=int, default=10**7)
    p.add_argument("--seconds", type=float, default=60.0, help="time cap per M")
    p.add_argument(
        "--eager", action="store_true", help="two plain streams instead of a CUDA graph"
    )
    a = p.parse_args()
    torch.manual_seed(0)
    dev = "cuda"

    def operands(m, n, k):
        mag = torch.empty(m, 1, k, device=dev).uniform_(
            64, 448
        )  # FP8 bytes 0x68..0x7e (+sign)
        x8 = (mag * (torch.randint(0, 2, mag.shape, device=dev) * 2 - 1)).to(dtypes.fp8)
        w8 = (torch.randn(1, n, k, device=dev) * 0.02).to(dtypes.fp8)
        xs = torch.full((m, 1, k // 32), 127, dtype=torch.uint8, device=dev)  # e8m0 2^0
        ws = torch.full((1, n, k // 32), 127, dtype=torch.uint8, device=dev)
        return x8, w8, shuffle_weight(w8), xs, ws

    def make_bg():
        if a.bg == "gemm":
            x8, _, wq, xs, ws = operands(a.bg_m, a.bg_n, a.bg_k)
            out = torch.empty(a.bg_m, 1, a.bg_n, dtype=torch.bfloat16, device=dev)
            return lambda: bmm(x8, wq, xs, ws, out, kernel_name=a.bg_kernel)
        if (
            a.bg == "moe"
        ):  # M3.1 TP4 routed MoE: 128 experts top-4, 6144 -> 768, a8w4 grouped TDM GEMMs
            from aiter import ActivationType, QuantType
            from aiter.fused_moe import fused_moe
            from aiter.ops.flydsl.moe_common import GateMode
            from aiter.ops.shuffle import moe_shuffle_scale, moe_shuffle_weight

            E, topk, D, I, T = 128, 4, 6144, 768, a.moe_t
            u8 = lambda *sh, lo=0, hi=256: torch.randint(
                lo, hi, sh, dtype=torch.uint8, device=dev
            )
            w1 = moe_shuffle_weight(
                u8(E, 2 * I, D // 2), experts_cnt=E, is_guinterleave=True, gate_up=True
            )
            w2 = moe_shuffle_weight(u8(E, D, I // 2), experts_cnt=E)
            s1 = moe_shuffle_scale(
                u8(E, 2 * I, D // 32, lo=118, hi=124),
                experts_cnt=E,
                is_guinterleave=True,
                gate_up=True,
            )
            s2 = moe_shuffle_scale(u8(E, D, I // 32, lo=118, hi=124), experts_cnt=E)
            h = torch.randn(T, D, dtype=torch.bfloat16, device=dev)
            ids = torch.stack(
                [torch.randperm(E, device=dev)[:topk] for _ in range(T)]
            ).int()
            wts = torch.full((T, topk), 1.0 / topk, device=dev)
            return lambda: fused_moe(
                h,
                w1,
                w2,
                wts,
                ids,
                activation=ActivationType.Swiglu,
                quant_type=QuantType.per_1x32,
                w1_scale=s1,
                w2_scale=s2,
                gate_mode=GateMode.INTERLEAVE.value,
                dtype=torch.bfloat16,
            )
        if a.bg == "mm":
            x = torch.randn(a.bg_m, 4096, dtype=torch.bfloat16, device=dev)
            w = torch.randn(4096, 4096, dtype=torch.bfloat16, device=dev)
            return lambda: torch.mm(x, w)
        return None

    bg = make_bg()
    print(
        f"bg={a.bg} ({a.bg_kernel if a.bg == 'gemm' else ''}) bg_mnk={a.bg_m}x{a.bg_n}x{a.bg_k} na={a.na} nb={a.nb} N={a.n} K={a.k}",
        flush=True,
    )
    total_bad = 0
    ms, kns = [int(v) for v in a.m.split(",")], a.kernel.split(",")
    for m, kn in zip(ms, kns * len(ms) if len(kns) == 1 else kns):
        kn = pick_bmm_mx32_kernel_name(1, m, a.n, a.k) if kn == "auto" else kn
        kn = (
            kn
            if kn.startswith("flydsl_")
            else f"flydsl_bmm_mxfp8_32_wmma_{kn}_sk1_cm1_cn1"
        )
        ops = [
            operands(m, a.n, a.k) for _ in range(a.nw)
        ]  # distinct weights per launch: cold like a model's layers
        refs = torch.empty(a.nw, m, 1, a.n, dtype=torch.bfloat16, device=dev)
        for j, (x8, w8, wq, xs, ws) in enumerate(ops):
            bmm(x8, wq, xs, ws, refs[j], kernel_name=kn)
        torch.cuda.synchronize()
        x8, w8 = ops[0][:2]
        exact = (x8.float()[:, 0] @ w8.float()[0].T)[:, None]
        rel = ((refs[0].float() - exact).norm() / exact.norm()).item()
        assert (
            rel < 1e-2 and refs.float().abs().max() < 1e5
        ), f"eager result wrong: rel {rel}"
        ops = [(x8, wq, xs, ws) for x8, _, wq, xs, ws in ops]
        ref = refs[torch.arange(a.na, device=dev) % a.nw]
        outs = torch.empty(a.na, m, 1, a.n, dtype=torch.bfloat16, device=dev)
        rep_bad = torch.zeros((), dtype=torch.int64, device=dev)
        launch_bad = torch.zeros((), dtype=torch.int64, device=dev)
        row_bad = torch.zeros(m, dtype=torch.int64, device=dev)
        vmax = torch.zeros((), dtype=torch.float32, device=dev)
        s0, s1 = torch.cuda.Stream(), torch.cuda.Stream()
        with torch.cuda.stream(s0):  # compile + warm up outside the capture
            for i in range(2):
                bmm(*ops[i % a.nw], outs[i], kernel_name=kn)
                if bg:
                    bg()
        torch.cuda.synchronize()

        def step():
            outs.fill_(float("nan"))  # an unwritten chunk counts as bad too
            s1.wait_stream(s0)
            with torch.cuda.stream(s1):
                for _ in range(a.nb if bg else 0):
                    bg()
            for i in range(a.na):
                bmm(*ops[i % a.nw], outs[i], kernel_name=kn)
            s0.wait_stream(s1)
            diff = (outs != ref).flatten(2).any(-1)  # [na, m]
            rep_bad.add_(diff.any())
            launch_bad.add_(diff.any(-1).sum())
            row_bad.add_(diff.sum(0))
            torch.maximum(
                vmax,
                torch.where(outs != ref, outs.float().abs(), 0).nan_to_num(3e38).amax(),
                out=vmax,
            )

        g = None if a.eager else torch.cuda.CUDAGraph()
        if g is not None:
            with torch.cuda.graph(g, stream=s0):
                step()
        torch.cuda.synchronize()
        t0, done = time.time(), 0
        while done < a.replays and time.time() - t0 < a.seconds:
            for _ in range(min(100, a.replays - done)):
                if g is not None:
                    g.replay()
                else:
                    with torch.cuda.stream(s0):
                        step()
                done += 1
            torch.cuda.synchronize()
        dt = time.time() - t0
        nb_rep = int(rep_bad)
        total_bad += nb_rep
        rows = {r: int(c) for r, c in enumerate(row_bad.tolist()) if c}
        print(
            f"M={m:4d} {kn}: bad replays: {nb_rep} / {done}  (bad launches {int(launch_bad)} / "
            f"{done * a.na}, rows hit {rows}, max |bad| {float(vmax):.3g}, {dt:.1f} s, "
            f"{dt / done * 1e6:.0f} us/replay, eager rel err {rel:.1e})",
            flush=True,
        )
        del g
    print("RESULT", "BAD" if total_bad else "CLEAN", f"total bad replays {total_bad}")


if __name__ == "__main__":
    main()
