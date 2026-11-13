"""Config sweep for the gfx1250 gluon A4W4 MoE kernels.

Reuses bench_moe_gemm_a4w4_cudagraph's setup and timing, and overrides the
config lookup (get_kernel_config_gluon) for one projection at a time.

  python op_tests/op_benchmarks/triton/tune_moe_a4w4_gluon.py \
      --shape 2048 1024 --experts 256 8 --M 256 16384 --layer moe1 moe2 \
      [--preshuffle] [--fused-quant] --out tune.csv
"""

import argparse
import contextlib
import csv
import fcntl
import itertools
import os
import sys
import time
import traceback

import torch
import triton

import aiter.ops.triton.moe.moe_op_gemm_a4w4 as moe_mod
from op_tests.op_benchmarks.triton import bench_moe_gemm_a4w4_cudagraph as bench

_orig_get_cfg = moe_mod.get_kernel_config_gluon
_override = {}  # (n, k) -> dict of overrides


def _patched_get_cfg(m, n, k, routing_data):
    cfg = _orig_get_cfg(m, n, k, routing_data)
    ov = _override.get((n, k))
    if ov:
        cfg = dict(cfg)
        cfg.update(ov)
        cfg["num_buffers"] = min(cfg["num_buffers"], triton.cdiv(k, cfg["block_k"]))
    return cfg


moe_mod.get_kernel_config_gluon = _patched_get_cfg


def layer_nk(layer, dim1, dim2):
    # moe1: N = dim2, K = dim1; moe2: N = dim1, K = dim2 / 2
    return (dim2, dim1) if layer == "moe1" else (dim1, dim2 // 2)


@contextlib.contextmanager
def gpu_lock():
    # Hold the shared node GPU lock per candidate, not for the whole sweep, so
    # other users' jobs can interleave. Set GPU_LOCK_FILE to enable.
    path = os.environ.get("GPU_LOCK_FILE")
    if not path:
        yield
        return
    with open(path) as fd:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            torch.cuda.synchronize()
            fcntl.flock(fd, fcntl.LOCK_UN)
    # flock is not FIFO: re-locking immediately can starve queued jobs, so
    # leave a window for any waiter to take the lock before the next candidate.
    time.sleep(2)


def run_one(args, batch, layer, ov):
    with gpu_lock():
        return _run_one(args, batch, layer, ov)


def _run_one(args, batch, layer, ov):
    n, k = layer_nk(layer, args.shape[0], args.shape[1])
    _override.clear()
    if ov is not None:
        _override[(n, k)] = ov
    torch.manual_seed(0)
    res = bench.bench_mlp_single_weight_init(
        batch,
        args.shape[0],
        args.shape[1],
        args.experts[0],
        args.experts[1],
        "mx4",
        "mx4",
        1,
        args.preshuffle,
        "gluon",
        None,
        args.rep,
        layers=(layer,),
        fused_quant=args.fused_quant,
    )
    lp = res["layers"][layer]
    return lp["latency_ms"] * 1e3, lp, res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shape", type=int, nargs=2, required=True)
    p.add_argument("--experts", type=int, nargs=2, required=True)
    p.add_argument("--M", type=int, nargs="+", required=True)
    p.add_argument("--layer", nargs="+", default=["moe1", "moe2"])
    p.add_argument("--preshuffle", action="store_true")
    p.add_argument("--fused-quant", action="store_true")
    p.add_argument("--rep", type=int, default=20)
    p.add_argument("--block-n", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--block-k", type=int, nargs="+", default=[128, 256, 512])
    p.add_argument("--num-buffers", type=int, nargs="+", default=[1, 2, 3, 4])
    p.add_argument("--num-warps", type=int, nargs="+", default=[4, 8])
    p.add_argument("--num-ctas", type=int, nargs="+", default=[1])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    fields = ["batch", "layer", "block_m", "block_n", "block_k", "num_buffers",
              "num_warps", "num_ctas", "latency_us", "tflops", "tbps", "status"]
    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()

    for batch, layer in itertools.product(args.M, args.layer):
        base_us, lp, res = run_one(args, batch, layer, None)
        bm = res["block_m"]
        print(f"[{batch} {layer} bm{bm}] default: {base_us:.2f} us", flush=True)
        w.writerow(dict(batch=batch, layer=layer, block_m=bm, block_n="default",
                        latency_us=base_us,
                        tflops=lp["flops"] / lp["latency_ms"] * 1e-9,
                        tbps=lp["bytes"] / lp["latency_ms"] * 1e-9, status="ok"))
        best = (base_us, "default")
        n, k = layer_nk(layer, args.shape[0], args.shape[1])
        seen = set()
        for bn, bk, nb, nw, nc in itertools.product(
                args.block_n, args.block_k, args.num_buffers, args.num_warps,
                args.num_ctas):
            if bk > k or bn > n:
                continue
            nb_eff = min(nb, triton.cdiv(k, bk))
            key = (bn, bk, nb_eff, nw, nc)
            if key in seen:
                continue
            seen.add(key)
            ov = dict(block_n=bn, block_k=bk, num_buffers=nb_eff, num_warps=nw,
                      num_ctas=nc)
            row = dict(batch=batch, layer=layer, block_m=bm, block_n=bn,
                       block_k=bk, num_buffers=nb_eff, num_warps=nw,
                       num_ctas=nc)
            try:
                us, lp, _ = run_one(args, batch, layer, ov)
                row.update(latency_us=us,
                           tflops=lp["flops"] / lp["latency_ms"] * 1e-9,
                           tbps=lp["bytes"] / lp["latency_ms"] * 1e-9, status="ok")
                if us < best[0]:
                    best = (us, key)
                print(f"[{batch} {layer} bm{bm}] {key}: {us:.2f} us", flush=True)
            except Exception as e:  # invalid config (LDS overflow, asserts, ...)
                msg = f"{type(e).__name__}: {str(e).splitlines()[0][:120] if str(e) else ''}"
                row.update(status=msg)
                print(f"[{batch} {layer} bm{bm}] {key}: FAIL {msg}", flush=True)
                torch.cuda.synchronize()
            w.writerow(row)
            f.flush()
        print(f"BEST [{batch} {layer} bm{bm}] {best}", flush=True)
    f.close()


if __name__ == "__main__":
    sys.exit(main())
