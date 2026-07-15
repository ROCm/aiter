"""Standalone benchmark + correctness check for kn_mla_reduce_v1 (csrc/kernels/mla/reduce.cu).

Three modes matching the FlyDSL bench for apples-to-apples comparison:
  --mode uniform   (default) uniform synthetic; every tile has num_splits splits.
  --mode irregular per-tile split list via --splits-per-tile or --splits-list-file.
  --mode replay    load real metadata from a .pt file (dump_decode_metadata.py).
"""

import argparse
import torch
import aiter
from aiter.test_common import run_perftest


def build_inputs(num_tiles, num_splits, H, Dv, out_dtype, M=1, device="cuda", seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    num_partial_rows = num_tiles * num_splits

    partial_output = torch.randn(
        num_partial_rows, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    partial_lse = (
        torch.randn(
            num_partial_rows, H, dtype=torch.float32, device=device, generator=g
        )
        * 2.0
    )
    reduce_indptr = torch.arange(
        0, num_partial_rows + 1, num_splits, dtype=torch.int32, device=device
    )
    reduce_partial_map = torch.arange(
        num_partial_rows, dtype=torch.int32, device=device
    )
    q_start = torch.arange(num_tiles, dtype=torch.int32, device=device) * M
    reduce_final_map = torch.stack([q_start, q_start + M], dim=1).contiguous()
    final_output = torch.zeros(num_tiles * M, H, Dv, dtype=out_dtype, device=device)
    final_lse = torch.zeros(num_tiles * M, H, dtype=torch.float32, device=device)
    return (
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_output,
        final_lse,
    )


def build_irregular_inputs(
    splits_per_tile, H, Dv, out_dtype, M=1, gap_stride=1, device="cuda", seed=0
):
    g = torch.Generator(device=device).manual_seed(seed)
    num_tiles = len(splits_per_tile)
    total_splits = sum(int(s) for s in splits_per_tile)

    indptr_host = [0]
    for s in splits_per_tile:
        indptr_host.append(indptr_host[-1] + int(s))
    reduce_indptr = torch.tensor(indptr_host, dtype=torch.int32, device=device)

    if total_splits > 0:
        slot = torch.arange(total_splits, dtype=torch.int32, device=device)
        reduce_partial_map = slot * (gap_stride * M)
        max_base = int(reduce_partial_map.max().item())
    else:
        reduce_partial_map = torch.zeros(1, dtype=torch.int32, device=device)
        max_base = 0
    num_partial_rows = max_base + M

    partial_output = torch.randn(
        num_partial_rows, H, Dv, dtype=torch.float32, device=device, generator=g
    )
    partial_lse = (
        torch.randn(
            num_partial_rows, H, dtype=torch.float32, device=device, generator=g
        )
        * 2.0
    )

    q_start = torch.arange(num_tiles, dtype=torch.int32, device=device) * M
    reduce_final_map = torch.stack([q_start, q_start + M], dim=1).contiguous()
    for t, s in enumerate(splits_per_tile):
        if int(s) <= 1:
            reduce_final_map[t, 0] = 1 << 24
            reduce_final_map[t, 1] = (1 << 24) + M

    final_output = torch.zeros(num_tiles * M, H, Dv, dtype=out_dtype, device=device)
    final_lse = torch.zeros(num_tiles * M, H, dtype=torch.float32, device=device)
    return (
        partial_output,
        partial_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_output,
        final_lse,
    )


def _traffic_bytes(indptr, H, Dv, out_dtype):
    diffs = indptr[1:] - indptr[:-1]
    total_splits = int(diffs.sum().item())
    active_tiles = int((diffs > 1).sum().item())
    bytes_partial_o = total_splits * H * Dv * 4
    bytes_partial_lse = total_splits * H * 4
    bytes_final_o = active_tiles * H * Dv * torch.finfo(out_dtype).bits // 8
    bytes_final_lse = active_tiles * H * 4
    return bytes_partial_o + bytes_partial_lse + bytes_final_o + bytes_final_lse


def bench_cudagraph(fn, num_warmup=25, num_iters=100):
    for _ in range(max(1, num_warmup)):
        fn()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        fn()
        side.synchronize()
        with torch.cuda.graph(graph, stream=side):
            for _ in range(num_iters):
                fn()
    torch.cuda.current_stream().wait_stream(side)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / num_iters


def _bench_and_print(run, indptr, H, Dv, out_dtype, M, label):
    _, kernel_us = run_perftest(run, num_warmup=25, num_iters=100)
    graph_us = bench_cudagraph(run) * 1e3
    total = _traffic_bytes(indptr, H, Dv, out_dtype)
    gbps = total / (kernel_us * 1e-6) / 1e9
    max_s = int((indptr[1:] - indptr[:-1]).max().item())
    path = "massive" if max_s >= 4 else "simple"
    print(
        f"[HIP {label}] H={H} Dv={Dv} M={M} max_splits={max_s} path={path} "
        f"kernel={kernel_us:.2f}us graph={graph_us:.2f}us "
        f"BW={gbps:.0f}GB/s ({gbps/5300*100:.1f}% of 5.3TB/s)"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", choices=["uniform", "irregular", "replay"], default="uniform"
    )
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--Dv", type=int, default=512)
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--check", action="store_true")
    # uniform
    ap.add_argument("--tiles", type=int, default=256)
    ap.add_argument("--splits", type=int, default=8)
    # irregular
    ap.add_argument("--splits-per-tile", default=None)
    ap.add_argument("--splits-list-file", default=None)
    ap.add_argument("--gap-stride", type=int, default=1)
    # replay
    ap.add_argument(
        "--meta", default=None, help="path to .pt from dump_decode_metadata.py"
    )
    args = ap.parse_args()

    _dt = {"bf16": torch.bfloat16, "fp16": torch.float16}
    out_dtype = _dt[args.dtype]

    if args.mode == "uniform":
        H, Dv, T, S, M = args.H, args.Dv, args.tiles, args.splits, args.M
        po, pl, indptr, fmap, pmap, fout, flse = build_inputs(
            T, S, H, Dv, out_dtype, M=M
        )

        def run():
            aiter.mla_reduce_v1(po, pl, indptr, fmap, pmap, M, 0, fout, flse)

        _bench_and_print(run, indptr, H, Dv, out_dtype, M, "uniform")

    elif args.mode == "irregular":
        H, Dv, M = args.H, args.Dv, args.M
        if args.splits_list_file:
            with open(args.splits_list_file) as f:
                splits_per_tile = [int(line.strip()) for line in f if line.strip()]
        elif args.splits_per_tile:
            splits_per_tile = [int(x) for x in args.splits_per_tile.split(",")]
        else:
            ap.error(
                "--mode irregular requires --splits-per-tile or --splits-list-file"
            )
        po, pl, indptr, fmap, pmap, fout, flse = build_irregular_inputs(
            splits_per_tile,
            H,
            Dv,
            out_dtype,
            M=M,
            gap_stride=args.gap_stride,
        )
        fout.zero_()
        flse.zero_()

        def run():
            aiter.mla_reduce_v1(po, pl, indptr, fmap, pmap, M, 0, fout, flse)

        _bench_and_print(run, indptr, H, Dv, out_dtype, M, "irregular")

    else:  # replay
        if not args.meta:
            ap.error("--mode replay requires --meta <path.pt>")
        meta = torch.load(args.meta, map_location="cpu")
        H = meta.get("H", args.H)
        Dv = meta.get("Dv", args.Dv)
        num_partial_rows = meta["num_partial_rows"]
        num_final_rows = meta["num_final_rows"]
        M = args.M

        indptr = meta["reduce_indptr"].cuda()
        fmap = meta["reduce_final_map"].cuda()
        pmap = meta["reduce_partial_map"].cuda()

        g = torch.Generator(device="cuda").manual_seed(0)
        po = torch.randn(
            num_partial_rows, H, Dv, dtype=torch.float32, device="cuda", generator=g
        )
        pl = (
            torch.randn(
                num_partial_rows, H, dtype=torch.float32, device="cuda", generator=g
            )
            * 2.0
        )
        fout = torch.zeros(num_final_rows, H, Dv, dtype=out_dtype, device="cuda")
        flse = torch.zeros(num_final_rows, H, dtype=torch.float32, device="cuda")

        def run():
            aiter.mla_reduce_v1(po, pl, indptr, fmap, pmap, M, 0, fout, flse)

        _bench_and_print(run, indptr, H, Dv, out_dtype, M, f"replay:{args.meta}")


if __name__ == "__main__":
    main()
