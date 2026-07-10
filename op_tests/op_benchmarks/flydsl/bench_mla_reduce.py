"""FlyDSL MLA reduce benchmark.

Default (no ``--mode``): GLM-5.2 serving scoreboard — production wrapper path
with ``final_output=[bs, H, Dv]``, ``actual_max_splits``, adaptive launch, and
DA split-K. Backends: ``hip``, ``wrapper-daoff``, ``wrapper-daon``.

Synthetic / replay modes (``--mode``):
  uniform    uniform synthetic workload; all tiles share ``num_splits``
  irregular  per-tile split list via ``--splits-per-tile`` or ``--splits-list-file``
  replay     load real metadata from a ``.pt`` file (``dump_decode_metadata.py``)
"""

import argparse
import os
import sys

import torch
import aiter

from aiter.test_common import run_perftest
from aiter.ops.flydsl.kernels.mla_reduce import plan_splitk_capture_safe, select_tier
from op_tests.flydsl_mla_reduce_common import (
    bench_cudagraph,
    build_inputs,
    build_irregular_inputs,
    make_runner,
    max_splits_from_indptr,
)

# --- GLM-5.2 serving scoreboard ------------------------------------------------

SERVING_H, SERVING_DV = 16, 512
SERVING_NUM_REDUCE_TILE = 16384
SERVING_PARTIAL_POOL = 606
SERVING_OUT_DTYPE = torch.bfloat16

SERVING_SCENARIOS = [
    ("b1_s128", 1, 128),
    ("b8_s32", 8, 32),
    ("b8_s26", 8, 26),
    ("b8_s13", 8, 13),
    ("b8_s6", 8, 6),
    ("b8_s5", 8, 5),
    ("b8_s3", 8, 3),
    ("b8_s2", 8, 2),
]

SERVING_BACKENDS = ("hip", "wrapper-daoff", "wrapper-daon")


def _serving_build(active_tiles, splits):
    active_splits = active_tiles * splits
    pool_slack = max(0, SERVING_PARTIAL_POOL - active_splits)
    splits_per_tile = [splits] * active_tiles + [0] * (
        SERVING_NUM_REDUCE_TILE - active_tiles
    )
    po, pl, indptr, fmap, pmap, fout, flse = build_irregular_inputs(
        splits_per_tile,
        SERVING_H,
        SERVING_DV,
        SERVING_OUT_DTYPE,
        M=1,
        gap_stride=1,
        pool_slack=pool_slack,
    )
    fout = fout[:active_tiles].contiguous()
    flse = flse[:active_tiles].contiguous()
    return po, pl, indptr, fmap, pmap, fout, flse


def _make_serving_hip_runner(po, pl, indptr, fmap, pmap, fout, flse, M=1):
    def run():
        aiter.mla_reduce_v1(po, pl, indptr, fmap, pmap, M, 0, fout, flse)

    return run


def _make_serving_wrapper_runner(po, pl, indptr, fmap, pmap, fout, flse, splits):
    from aiter.ops.flydsl import flydsl_mla_reduce_v1

    def run():
        flydsl_mla_reduce_v1(
            po,
            pl,
            indptr,
            fmap,
            pmap,
            1,
            fout,
            flse,
            num_kv_splits=splits,
            actual_max_splits=splits,
        )

    return run


def run_serving(backend: str) -> None:
    if backend == "wrapper-daoff":
        os.environ["AITER_MLA_REDUCE_DA_SPLITK"] = "0"
    elif backend == "wrapper-daon":
        os.environ["AITER_MLA_REDUCE_DA_SPLITK"] = "1"

    num_cu = torch.cuda.get_device_properties(0).multi_processor_count
    print(
        f"# GLM-5.2 serving mla_reduce  H={SERVING_H} Dv={SERVING_DV} out=bf16 M=1 "
        f"grid={SERVING_NUM_REDUCE_TILE} pool={SERVING_PARTIAL_POOL} "
        f"num_cu={num_cu} backend={backend}"
    )
    for label, active, splits in SERVING_SCENARIOS:
        po, pl, indptr, fmap, pmap, fout, flse = _serving_build(active, splits)
        fout.zero_()
        flse.zero_()
        if backend == "hip":
            run = _make_serving_hip_runner(po, pl, indptr, fmap, pmap, fout, flse)
            plan = ""
        else:
            run = _make_serving_wrapper_runner(
                po, pl, indptr, fmap, pmap, fout, flse, splits
            )
            engage, K, slots = plan_splitk_capture_safe(
                num_final_rows=int(fout.size(0)),
                H=SERVING_H,
                max_seqlen_q=1,
                num_kv_splits=splits,
                num_cu=num_cu,
            )
            plan = (
                f"splitk K={K} slots={slots} grid_p={slots * K}"
                if (engage and backend == "wrapper-daon")
                else "single-kernel"
            )
        _, kernel_us = run_perftest(run, num_warmup=25, num_iters=100)
        graph_us = bench_cudagraph(run) * 1e3
        print(
            f"[{backend}] {label:8s} active={active} splits={splits:3d} "
            f"kernel={kernel_us:6.1f}us graph={graph_us:6.1f}us  {plan}"
        )


# --- uniform / irregular / replay --------------------------------------------


def _traffic_bytes(indptr, H, Dv, out_dtype, num_partial_rows):
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


def run_synthetic(args: argparse.Namespace) -> None:
    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    if args.mode == "uniform":
        H, Dv, T, S, M = args.H, args.Dv, args.tiles, args.splits, args.M
        po, pl, indptr, fmap, pmap, fout, flse = build_inputs(
            T, S, H, Dv, out_dtype, M=M
        )
        fout.zero_()
        flse.zero_()
        run = make_runner(
            po, pl, indptr, pmap, fmap, fout, flse, H, Dv, args.dtype, args.lse, M
        )
        _bench_and_print(run, po, indptr, H, Dv, out_dtype, M, "uniform")

    elif args.mode == "irregular":
        H, Dv, M = args.H, args.Dv, args.M
        if args.splits_list_file:
            with open(args.splits_list_file) as f:
                splits_per_tile = [int(line.strip()) for line in f if line.strip()]
        elif args.splits_per_tile:
            splits_per_tile = [int(x) for x in args.splits_per_tile.split(",")]
        else:
            raise SystemExit(
                "--mode irregular requires --splits-per-tile or --splits-list-file"
            )
        po, pl, indptr, fmap, pmap, fout, flse = build_irregular_inputs(
            splits_per_tile,
            H,
            Dv,
            out_dtype,
            M=M,
            gap_stride=args.gap_stride,
            pool_slack=args.pool_slack,
        )
        fout.zero_()
        flse.zero_()
        run = make_runner(
            po, pl, indptr, pmap, fmap, fout, flse, H, Dv, args.dtype, args.lse, M
        )
        _bench_and_print(run, po, indptr, H, Dv, out_dtype, M, "irregular")

    else:  # replay
        if not args.meta:
            raise SystemExit("--mode replay requires --meta <path.pt>")
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

        run = make_runner(
            po,
            pl,
            indptr,
            pmap,
            fmap,
            fout,
            flse,
            H,
            Dv,
            args.dtype,
            args.lse,
            M,
            num_partial_rows=num_partial_rows,
            num_final_rows=num_final_rows,
        )
        _bench_and_print(run, po, indptr, H, Dv, out_dtype, M, f"replay:{args.meta}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in SERVING_BACKENDS:
        run_serving(sys.argv[1])
        return

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "backend",
        nargs="?",
        default="wrapper-daon",
        choices=SERVING_BACKENDS,
        help="serving-path backend (default when no --mode)",
    )
    ap.add_argument(
        "--mode",
        choices=["uniform", "irregular", "replay"],
        default=None,
        help="synthetic/replay mode; omit for GLM-5.2 serving scoreboard",
    )
    ap.add_argument("--H", type=int, default=128)
    ap.add_argument("--Dv", type=int, default=512)
    ap.add_argument("--M", type=int, default=1, help="max_seqlen_q")
    ap.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    ap.add_argument("--lse", action="store_true")
    ap.add_argument("--tiles", type=int, default=256)
    ap.add_argument("--splits", type=int, default=8)
    ap.add_argument(
        "--splits-per-tile",
        default=None,
        help='comma-separated per-tile split counts, e.g. "32,0,32,8"',
    )
    ap.add_argument("--splits-list-file", default=None, help="one int per line")
    ap.add_argument("--gap-stride", type=int, default=1)
    ap.add_argument("--pool-slack", type=int, default=0)
    ap.add_argument("--meta", default=None, help="path to .pt from dump_decode_metadata.py")
    args = ap.parse_args()

    if args.mode is None:
        run_serving(args.backend)
    else:
        run_synthetic(args)


if __name__ == "__main__":
    main()
