#!/usr/bin/env python3
"""Launcher for FlyDSL A8W8 FP8 GEMM kernel — used by perf_a8w8.sh for roccap capture.

Can use either auto-config (from JSON configs) or manual tile params.

Examples:
    # Auto-config (looks up best config for shape):
    python gemm_a8w8_launch.py 32 7168 4096

    # Manual config override:
    python gemm_a8w8_launch.py 64 5120 2880 --tile-m 64 --tile-n 256 --tile-k 256 --num-buffers 3
"""

import os
import sys

# Ensure aiter is importable
_AITER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
if _AITER_ROOT not in sys.path:
    sys.path.insert(0, _AITER_ROOT)

import flydsl  # noqa: E402,F401 — preload comgr before torch/HIP
import torch

from aiter.ops.flydsl.kernels.gemm_a8w8_gfx1250 import compile_gemm_a8w8
from aiter.ops.flydsl.gemm_config_utils import get_gemm_config


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Launch FlyDSL A8W8 FP8 GEMM kernel")
    parser.add_argument("M", type=int)
    parser.add_argument("N", type=int)
    parser.add_argument("K", type=int)
    parser.add_argument("--tile-m", type=int, default=None)
    parser.add_argument("--tile-n", type=int, default=None)
    parser.add_argument("--tile-k", type=int, default=None)
    parser.add_argument("--m-warp", type=int, default=None)
    parser.add_argument("--n-warp", type=int, default=None)
    parser.add_argument("--num-buffers", type=int, default=None)
    parser.add_argument("--cluster-m", type=int, default=None)
    parser.add_argument("--cluster-n", type=int, default=None)
    parser.add_argument("--waves-per-eu", type=int, default=None)
    parser.add_argument("--no-tdm-load", action="store_true", default=False)
    parser.add_argument("--out-dtype", type=str, default="bf16", choices=["f16", "bf16", "f32"])
    parser.add_argument("--has-bias", action="store_true", default=False)
    parser.add_argument("--no-auto-config", action="store_true",
                        help="Skip auto-config lookup, use defaults/manual args only")
    args = parser.parse_args()

    M, N, K = args.M, args.N, args.K

    # Load auto-config unless disabled
    config = {}
    if not args.no_auto_config:
        try:
            config, is_tuned = get_gemm_config("GEMM-A8W8", M, N=N, K=K)
            label = "tuned" if is_tuned else "default"
            print(f"Auto-config ({label}): {config}")
        except (FileNotFoundError, KeyError) as e:
            print(f"Auto-config failed ({e}), using defaults")

    # CLI args override auto-config
    tile_m = args.tile_m or config.get("tile_m", 128)
    tile_n = args.tile_n or config.get("tile_n", 256)
    tile_k = args.tile_k or config.get("tile_k", 128)
    m_warp = args.m_warp or config.get("m_warp", 2)
    n_warp = args.n_warp or config.get("n_warp", 4)
    num_buffers = args.num_buffers or config.get("num_buffers", 2)
    cluster_m = args.cluster_m if args.cluster_m is not None else config.get("cluster_m", 1)
    cluster_n = args.cluster_n if args.cluster_n is not None else config.get("cluster_n", 1)
    waves_per_eu = args.waves_per_eu if args.waves_per_eu is not None else config.get("waves_per_eu", None)
    l2_prefetch_distance = config.get("l2_prefetch_distance", 0)
    use_tdm_load = not args.no_tdm_load and config.get("use_tdm_load", True)
    has_bias = args.has_bias

    fp8_dtype = torch.float8_e4m3fn
    _out_torch = {"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[args.out_dtype]

    # Pad dimensions
    mpad = (M + tile_m - 1) // tile_m * tile_m
    npad = (N + tile_n - 1) // tile_n * tile_n
    kpad = (K + tile_k - 1) // tile_k * tile_k

    a_gpu = torch.zeros((mpad, kpad), dtype=fp8_dtype, device="cuda")
    b_gpu = torch.zeros((npad, kpad), dtype=fp8_dtype, device="cuda")
    a_scale_gpu = torch.ones(mpad, dtype=torch.float32, device="cuda")
    b_scale_gpu = torch.ones(npad, dtype=torch.float32, device="cuda")
    bias_gpu = torch.zeros(npad, dtype=torch.float32, device="cuda")
    c_gpu = torch.zeros((mpad, npad), dtype=_out_torch, device="cuda")

    print(f"Compiling A8W8 GEMM: M={M}({mpad}), N={N}({npad}), K={K}({kpad}), "
          f"tiles=({tile_m},{tile_n},{tile_k}), warps=({m_warp},{n_warp}), "
          f"bufs={num_buffers}, cluster=({cluster_m},{cluster_n}), "
          f"tdm_load={use_tdm_load}, out={args.out_dtype}, bias={has_bias}")

    launch_fn = compile_gemm_a8w8(
        K=kpad,
        tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
        m_warp=m_warp, n_warp=n_warp,
        num_buffers=num_buffers,
        cluster_m=cluster_m, cluster_n=cluster_n,
        waves_per_eu=waves_per_eu,
        l2_prefetch_distance=l2_prefetch_distance,
        use_tdm_load=use_tdm_load,
        out_dtype=args.out_dtype,
        has_bias=has_bias,
    )

    print("Launching kernel...")
    launch_fn(
        c_gpu.contiguous().view(-1),
        a_gpu.contiguous().view(-1),
        b_gpu.contiguous().view(-1),
        a_scale_gpu.contiguous().view(-1),
        b_scale_gpu.contiguous().view(-1),
        bias_gpu.contiguous().view(-1),
        mpad, npad,
        torch.cuda.current_stream(),
    )
    torch.cuda.synchronize()
    print("Done.")


if __name__ == "__main__":
    main()
