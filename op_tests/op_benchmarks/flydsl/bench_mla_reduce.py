"""Standalone benchmark for the FlyDSL MLA reduce kernel.

Three modes:
  --mode uniform   (default) uniform synthetic workload; all tiles have the
                   same num_splits. Keeps old PR tables reproducible.
  --mode irregular per-tile split list via --splits-per-tile "8,32,0,..."
                   or --splits-list-file; exercises mixed-tier serving grids.
  --mode replay    load real metadata from a .pt file (from dump_decode_metadata.py)
                   and bench the exact production distribution.
"""

import argparse
import torch
from aiter.test_common import run_perftest
from op_tests.flydsl_mla_reduce_common import (
    bench_cudagraph,
    build_inputs,
    build_irregular_inputs,
    make_runner,
    max_splits_from_indptr,
)
from aiter.ops.flydsl.kernels.mla_reduce import select_tier


def _traffic_bytes(indptr, H, Dv, out_dtype, num_partial_rows):
    """Accurate byte model for irregular / replay workloads."""
    diffs = indptr[1:] - indptr[:-1]
    total_splits = int(diffs.sum().item())
    active_tiles = int((diffs > 1).sum().item())
    bytes_partial_o = total_splits * H * Dv * 4
    bytes_partial_lse = total_splits * H * 4
    bytes_final_o = active_tiles * H * Dv * torch.finfo(out_dtype).bits // 8
    bytes_final_lse = active_tiles * H * 4
    return bytes_partial_o + bytes_partial_lse + bytes_final_o + bytes_final_lse


def _bench_and_print(run, po, indptr, H, Dv, out_dtype, M, label):
    _, kernel_us = run_perftest(run, num_warmup=25, num_iters=100)
    graph_us = bench_cudagraph(run) * 1e3
    total = _traffic_bytes(indptr, H, Dv, out_dtype, po.shape[0])
    gbps = total / (kernel_us * 1e-6) / 1e9
    max_s = max_splits_from_indptr(indptr)
    tier = select_tier(max_s)
    print(
        f"[FlyDSL {label}] H={H} Dv={Dv} M={M} max_splits={max_s} "
        f"tier={tier.value} "
        f"kernel={kernel_us:.1f}us graph={graph_us:.1f}us "
        f"BW={gbps:.0f}GB/s ({gbps/5300*100:.0f}% of 5.3TB/s)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["uniform", "irregular", "replay"], default="uniform")
    # uniform / irregular shared
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--Dv", type=int, default=512)
    ap.add_argument("--M", type=int, default=1, help="max_seqlen_q")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--lse", action="store_true")
    # uniform
    ap.add_argument("--tiles", type=int, default=256)
    ap.add_argument("--splits", type=int, default=8)
    # irregular
    ap.add_argument(
        "--splits-per-tile",
        default=None,
        help='comma-separated per-tile split counts, e.g. "32,0,32,8"',
    )
    ap.add_argument("--splits-list-file", default=None, help="one int per line")
    ap.add_argument("--gap-stride", type=int, default=1)
    ap.add_argument("--pool-slack", type=int, default=0)
    # replay
    ap.add_argument("--meta", default=None, help="path to .pt from dump_decode_metadata.py")
    args = ap.parse_args()

    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if args.mode == "uniform":
        H, Dv, T, S, M = args.H, args.Dv, args.tiles, args.splits, args.M
        po, pl, indptr, fmap, pmap, fout, flse = build_inputs(T, S, H, Dv, out_dtype, M=M)
        fout.zero_(); flse.zero_()
        run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, args.dtype, args.lse, M)
        _bench_and_print(run, po, indptr, H, Dv, out_dtype, M, "uniform")

    elif args.mode == "irregular":
        H, Dv, M = args.H, args.Dv, args.M
        if args.splits_list_file:
            with open(args.splits_list_file) as f:
                splits_per_tile = [int(l.strip()) for l in f if l.strip()]
        elif args.splits_per_tile:
            splits_per_tile = [int(x) for x in args.splits_per_tile.split(",")]
        else:
            ap.error("--mode irregular requires --splits-per-tile or --splits-list-file")
        po, pl, indptr, fmap, pmap, fout, flse = build_irregular_inputs(
            splits_per_tile, H, Dv, out_dtype, M=M,
            gap_stride=args.gap_stride, pool_slack=args.pool_slack,
        )
        fout.zero_(); flse.zero_()
        run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, args.dtype, args.lse, M)
        _bench_and_print(run, po, indptr, H, Dv, out_dtype, M, "irregular")

    else:  # replay
        if not args.meta:
            ap.error("--mode replay requires --meta <path.pt>")
        meta = torch.load(args.meta, map_location="cpu")
        H = meta.get("H", args.H)
        Dv = meta.get("Dv", args.Dv)
        num_partial_rows = meta["num_partial_rows"]
        num_final_rows = meta["num_final_rows"]
        M = args.M  # replay always uses the dumped metadata's M=1 decode

        indptr = meta["reduce_indptr"].cuda()
        fmap   = meta["reduce_final_map"].cuda()
        pmap   = meta["reduce_partial_map"].cuda()

        g = torch.Generator(device="cuda").manual_seed(0)
        po = torch.randn(num_partial_rows, H, Dv, dtype=torch.float32, device="cuda", generator=g)
        pl = torch.randn(num_partial_rows, H, dtype=torch.float32, device="cuda", generator=g) * 2.0
        fout = torch.zeros(num_final_rows, H, Dv, dtype=out_dtype, device="cuda")
        flse = torch.zeros(num_final_rows, H, dtype=torch.float32, device="cuda")

        run = make_runner(
            po, pl, indptr, pmap, fmap, fout, flse, H, Dv, args.dtype, args.lse, M,
            num_partial_rows=num_partial_rows, num_final_rows=num_final_rows,
        )
        _bench_and_print(run, po, indptr, H, Dv, out_dtype, M, f"replay:{args.meta}")


if __name__ == "__main__":
    main()
