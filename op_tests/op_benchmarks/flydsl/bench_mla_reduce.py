"""Standalone benchmark for the FlyDSL MLA reduce kernel.

Builds a uniform synthetic workload (same layout as test_flydsl_mla_reduce.py) and
reports kernel-only device time plus achieved HBM bandwidth.
"""

import argparse

import torch
from aiter.test_common import run_perftest
from op_tests.flydsl_mla_reduce_common import (
    bench_cudagraph,
    build_inputs,
    make_runner,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--Dv", type=int, default=512)
    ap.add_argument("--tiles", type=int, default=256, help="num_reduce_tile")
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument(
        "--M", type=int, default=1, help="max_seqlen_q (q-positions per token group)"
    )
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--lse", action="store_true", help="write final LSE (always on in prod)")
    args = ap.parse_args()

    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    H, Dv, T_, S, M = args.H, args.Dv, args.tiles, args.splits, args.M

    po, pl, indptr, fmap, pmap, fout, flse = build_inputs(T_, S, H, Dv, out_dtype, M=M)
    fout.zero_()
    flse.zero_()

    run = make_runner(
        po, pl, indptr, pmap, fmap, fout, flse, H, Dv, args.dtype, args.lse, M
    )

    _, kernel_us = run_perftest(run, num_warmup=25, num_iters=100)
    graph_us = bench_cudagraph(run) * 1e3

    bytes_partial_o = T_ * S * H * Dv * 4
    bytes_partial_lse = T_ * S * H * 4
    bytes_final_o = T_ * H * Dv * out_dtype.itemsize
    bytes_final_lse = T_ * H * 4
    total = bytes_partial_o + bytes_partial_lse + bytes_final_o + bytes_final_lse
    gbps = total / (kernel_us * 1e-6) / 1e9
    path = "massive" if S >= 4 else "simple"
    print(
        f"[bench] H={H} Dv={Dv} tiles={T_} splits={S} M={M} path={path} "
        f"kernel={kernel_us:.1f}us graph={graph_us:.1f}us BW={gbps:.0f}GB/s "
        f"({gbps/5300*100:.0f}% of 5.3TB/s)"
    )


if __name__ == "__main__":
    main()
