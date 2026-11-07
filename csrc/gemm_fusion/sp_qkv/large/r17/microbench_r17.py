#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""R18 r13 / large/r17 standalone microbench: FlyDSL preshuffle GEMM with the destination-major BF16 epilogue (S[dst][row][F]) vs the stock row-major epilogue.

UNCOMPILED / UNTESTED source (written without a GPU or FlyDSL). Run on one MI355X (gfx950) with aiter + flydsl installed:
    python microbench_r17.py --m 384 450 512 700 --n 9216 9856 --tile 477 236 --iters 100 [--exact] [--no-sys-release] [--no-check]

For each (tile, M, N): times (a) BASE  = the same kernel with dst_map="none" (row-major [M, N] bf16, = the stock FlyDSL number to compare with, ~41.5 us at M=512 for tile 477)
                              (b) DST   = dst_map = dense (N 9216) / indexer (N 9856): epilogue writes bf16((acc*Ws)*As) into S[4][M][F] through the field map (IK replicated x4 for the indexer),
and checks (b) against a torch reference mapped through mirror_dstmap.py: --exact uses small-integer fp8 data and power-of-two scales so the fp32 accumulation is order-independent and the
check is BYTE-EXACT; the default random data is checked to within 1 bf16 ulp (accumulation-order differences only). Completion = kernel boundary ("all done"; no arrival counters).
Tiles (aiter a8w8 bpreshuffle tuned names): 477 = flydsl_bpreshuflle_112x192x256_F8_F8_B16_1x1x4x2 (async copy 1, waves_per_eu 1, xcd_swizzle 4, lds_stage 2);
                                            236 = flydsl_bpreshuflle_128x192x256_F8_F8_B16_1x0x4x2 (async copy 1, waves_per_eu unset, xcd_swizzle 4, lds_stage 2).
"""

import argparse
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flydsl.expr as fx
import mirror_dstmap as mm
from flydsl_dstmap_gemm import compile_preshuffle_gemm

from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled
from aiter.ops.shuffle import shuffle_weight

K = 6144
TILES = {
    477: {
        "tile_m": 112,
        "tile_n": 192,
        "tile_k": 256,
        "use_async_copy": True,
        "waves_per_eu": 1,
        "xcd_swizzle": 4,
        "lds_stage": 2,
    },
    236: {
        "tile_m": 128,
        "tile_n": 192,
        "tile_k": 256,
        "use_async_copy": True,
        "waves_per_eu": None,
        "xcd_swizzle": 4,
        "lds_stage": 2,
    },
}


def make_inputs(m, n, exact, seed):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    if exact:
        # values in {-2,-1,0,1,2}: |sum| <= 4*K < 2^24 so every fp32 accumulation order is exact; power-of-two scales keep the products exact
        xi = torch.randint(-2, 3, (m, K), device="cuda", generator=g).float()
        wi = torch.randint(-2, 3, (n, K), device="cuda", generator=g).float()
        xs = torch.pow(
            2.0, torch.randint(-4, 3, (m,), device="cuda", generator=g).float()
        )
        ws = torch.pow(
            2.0, torch.randint(-4, 3, (n,), device="cuda", generator=g).float()
        )
        xq, wq = xi.to(torch.float8_e4m3fn), wi.to(torch.float8_e4m3fn)
    else:
        xq = (torch.randn(m, K, device="cuda", generator=g) * 0.5).to(
            torch.float8_e4m3fn
        )
        wq = (torch.randn(n, K, device="cuda", generator=g) * 0.5).to(
            torch.float8_e4m3fn
        )
        xs = torch.rand(m, device="cuda", generator=g) * 0.02 + 0.005
        ws = torch.rand(n, device="cuda", generator=g) * 0.02 + 0.005
    return xq, wq, xs.float().contiguous(), ws.float().contiguous()


def reference_S(xq, wq, xs, ws, m, n, indexer):
    cf = xq.float() @ wq.float().t()  # fp32 (fp8 -> fp32 is exact)
    ref = ((cf * ws[None, :]) * xs[:, None]).to(
        torch.bfloat16
    )  # r13 ptpc order (acc * Ws) * As, one RNE
    f = mm.f_of(indexer)
    s = torch.zeros(4, m, f, dtype=torch.bfloat16, device="cuda")
    tab = mm.column_tables(indexer)
    n_idx = torch.tensor([t[0] for t in tab], device="cuda")
    d_idx = torch.tensor([t[1] for t in tab], device="cuda")
    pk_idx = torch.tensor([t[2] for t in tab], device="cuda")
    s.permute(0, 2, 1)[d_idx, pk_idx] = ref.t()[
        n_idx
    ]  # every (dst, packed) cell exactly once
    return s, ref


def timed(fn, iters, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(5):  # 5 batches of iters/5 launches -> median of batch means
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
            enable_timing=True
        )
        e0.record()
        for _ in range(max(1, iters // 5)):
            fn()
        e1.record()
        torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1) * 1000.0 / max(1, iters // 5))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, nargs="+", default=[384, 450, 512, 700])
    ap.add_argument("--n", type=int, nargs="+", default=[9216, 9856])
    ap.add_argument("--tile", type=int, nargs="+", default=[477, 236])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument(
        "--exact", action="store_true", help="small-integer data: byte-exact check"
    )
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument(
        "--no-sys-release",
        action="store_true",
        help="A/B: plain (write-back) S stores without the system-scope release (BG-1 option sys_release=False)",
    )
    a = ap.parse_args()
    rows = []
    for tile in a.tile:
        cfg = TILES[tile]
        for n in a.n:
            indexer = n == 9856
            dmap = "indexer" if indexer else "dense"
            f = mm.f_of(indexer)
            exe_base = compile_preshuffle_gemm(
                N=n,
                K=K,
                in_dtype="fp8",
                out_dtype="bf16",
                split_k=1,
                dst_map="none",
                **cfg,
            )
            exe_dst = compile_preshuffle_gemm(
                N=n,
                K=K,
                in_dtype="fp8",
                out_dtype="bf16",
                split_k=1,
                dst_map=dmap,
                sys_release=not a.no_sys_release,
                **cfg,
            )
            for m in a.m:
                xq, wq, xs, ws = make_inputs(m, n, a.exact, seed=m * 1000 + n)
                wq_sh = shuffle_weight(wq, layout=(16, 16))
                out_base = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
                s_buf = torch.zeros(4 * m * f, dtype=torch.bfloat16, device="cuda")
                sem = torch.empty(0, dtype=torch.int32, device="cuda")
                bias = torch.empty(0, dtype=torch.bfloat16, device="cuda")
                stream = fx.Stream(torch.cuda.current_stream())
                xq8, wq8 = xq.view(torch.int8).contiguous().view(-1), wq_sh.view(
                    torch.int8
                ).contiguous().view(-1)

                def run_base(
                    exe_base=exe_base,
                    out_base=out_base,
                    sem=sem,
                    xq8=xq8,
                    wq8=wq8,
                    xs=xs,
                    ws=ws,
                    bias=bias,
                    m=m,
                    n=n,
                    stream=stream,
                ):  # bound per iteration (the closures run inside it)
                    _run_compiled(
                        exe_base,
                        out_base.view(-1),
                        out_base.view(-1),
                        sem,
                        xq8,
                        wq8,
                        xs,
                        ws,
                        bias,
                        m,
                        n,
                        stream,
                    )

                def run_dst(
                    exe_dst=exe_dst,
                    s_buf=s_buf,
                    sem=sem,
                    xq8=xq8,
                    wq8=wq8,
                    xs=xs,
                    ws=ws,
                    bias=bias,
                    m=m,
                    n=n,
                    stream=stream,
                ):  # arg_c and arg_out are both S (arg_c is only used to build the unused default-store views)
                    _run_compiled(
                        exe_dst, s_buf, s_buf, sem, xq8, wq8, xs, ws, bias, m, n, stream
                    )

                t_base = timed(run_base, a.iters)
                t_dst = timed(run_dst, a.iters)
                status = "unchecked"
                if not a.no_check:
                    s_buf.zero_()
                    run_dst()
                    torch.cuda.synchronize()
                    s_ref, _ref = reference_S(xq, wq, xs, ws, m, n, indexer)
                    got = s_buf.view(4, m, f)
                    exact = bool(
                        torch.equal(got.view(torch.int16), s_ref.view(torch.int16))
                    )
                    if exact:
                        status = "BYTE-EXACT"
                    else:
                        d = (got.float() - s_ref.float()).abs()
                        tol = s_ref.float().abs() * (2.0**-7) + 1e-30
                        bad = int((d > tol).sum())
                        status = f"{'FAIL' if bad else 'OK(<=1ulp)'} mismatches>{1}ulp={bad} byte-diffs={int((got.view(torch.int16) != s_ref.view(torch.int16)).sum())} maxabs={float(d.max()):.3e}"
                        if a.exact:
                            status = "FAIL(exact mode, must be byte-equal) " + status
                rows.append((tile, m, n, t_base, t_dst, status))
                print(
                    f"tile {tile} M {m:4d} N {n}: base {t_base:7.2f} us  dst-map {t_dst:7.2f} us  (+{t_dst - t_base:5.2f})  {status}",
                    flush=True,
                )
    print("\ntile,M,N,base_us,dstmap_us,delta_us,check")
    for r in rows:
        print(f"{r[0]},{r[1]},{r[2]},{r[3]:.2f},{r[4]:.2f},{r[4] - r[3]:.2f},{r[5]}")
    bad = [r for r in rows if r[5].startswith("FAIL")]
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
