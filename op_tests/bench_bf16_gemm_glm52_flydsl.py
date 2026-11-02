#!/usr/bin/env python
"""Config-driven driver for FlyDSL BF16 split-K GEMM (GLM-5.2 decode, N=6144 K=3072).

C = A @ B^T, A:(M,K) bf16, B:(N,K) bf16, out:(M,N) bf16, fp32 accumulate.

Reads kernel config from the KF_CONFIG env var (JSON), e.g.:
  KF_CONFIG='{"tile_m":32,"tile_n":128,"tile_k":64,"split_k":4,
              "block_m_warps":2,"block_n_warps":2,"b_to_lds":true,
              "waves_per_eu":0,"async_copy":true}'

Modes:
  --mode test   -> prints 'SNR: XX.XX dB'
  --mode bench  -> prints 'median_ms: X.XXXX' and per-iter 'wall_ms: X.XXXX'

Prints 'flops:' and 'hbm_bytes:' for roofline / oracle tooling.
"""
import argparse
import json
import os
import statistics

import torch

from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm

N_DEFAULT = 6144
K_DEFAULT = 3072
SEED = 20260401


# Default = sweep winner at M=128, confirmed 15.13us device (rocprofv3).
# split_k=1 (no reduction kernel); tile_k=256 maximizes B-reuse.
DEFAULT_CFG = {
    "tile_m": 64,
    "tile_n": 64,
    "tile_k": 256,
    "split_k": 1,
    "block_m_warps": 1,
    "block_n_warps": 4,
    "b_to_lds": True,
    "waves_per_eu": 0,
    "async_copy": None,  # None -> arch default (True on gfx950)
}


# Per-M dispatch table (Phase C), device_us measured by rocprofv3 (kernel-trace).
# All entries: b_to_lds=True, waves_per_eu=0 (no-op on generic HGEMM), async default.
# Small M uses tile_m=16 + split-K (spread the constant 37.7MB B-read across CUs);
# large M uses big tiles + tile_k=64 (MFMA-pipeline fill, more n-tiles for occupancy).
DISPATCH = {
    1:    {"tile_m": 16, "tile_n": 64,  "tile_k": 128, "split_k": 4, "block_m_warps": 1, "block_n_warps": 2, "b_to_lds": True},
    8:    {"tile_m": 16, "tile_n": 64,  "tile_k": 128, "split_k": 4, "block_m_warps": 1, "block_n_warps": 2, "b_to_lds": True},
    32:   {"tile_m": 16, "tile_n": 64,  "tile_k": 128, "split_k": 2, "block_m_warps": 1, "block_n_warps": 2, "b_to_lds": True},
    64:   {"tile_m": 16, "tile_n": 64,  "tile_k": 128, "split_k": 1, "block_m_warps": 1, "block_n_warps": 2, "b_to_lds": True},
    128:  {"tile_m": 64, "tile_n": 64,  "tile_k": 256, "split_k": 1, "block_m_warps": 1, "block_n_warps": 4, "b_to_lds": True},
    256:  {"tile_m": 32, "tile_n": 128, "tile_k": 64,  "split_k": 1, "block_m_warps": 1, "block_n_warps": 4, "b_to_lds": True},
    512:  {"tile_m": 64, "tile_n": 128, "tile_k": 64,  "split_k": 1, "block_m_warps": 1, "block_n_warps": 4, "b_to_lds": True},
    1024: {"tile_m": 128,"tile_n": 128, "tile_k": 64,  "split_k": 1, "block_m_warps": 2, "block_n_warps": 2, "b_to_lds": True},
    4096: {"tile_m": 256,"tile_n": 128, "tile_k": 64,  "split_k": 1, "block_m_warps": 4, "block_n_warps": 2, "b_to_lds": True},
}


def get_config(cli_json="", M=None, use_dispatch=False):
    cfg = dict(DEFAULT_CFG)
    # precedence: --config CLI > KF_CONFIG env > per-M dispatch table > default
    if use_dispatch and M in DISPATCH:
        cfg.update(DISPATCH[M])
    env = os.environ.get("KF_CONFIG", "")
    if env.strip():
        cfg.update(json.loads(env))
    if cli_json and cli_json.strip():
        cfg.update(json.loads(cli_json))
    return cfg


def run_kernel(a, b, cfg, out=None):
    kwargs = dict(
        tile_m=cfg["tile_m"],
        tile_n=cfg["tile_n"],
        tile_k=cfg["tile_k"],
        split_k=cfg["split_k"],
        block_m_warps=cfg["block_m_warps"],
        block_n_warps=cfg["block_n_warps"],
        b_to_lds=cfg["b_to_lds"],
        waves_per_eu=cfg.get("waves_per_eu", 0),
    )
    if cfg.get("async_copy") is not None:
        kwargs["async_copy"] = cfg["async_copy"]
    return flydsl_hgemm(a, b, out=out, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["test", "bench", "profile"], default="test")
    ap.add_argument("--M", type=int, default=128)
    ap.add_argument("--N", type=int, default=N_DEFAULT)
    ap.add_argument("--K", type=int, default=K_DEFAULT)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--inner", type=int, default=20)
    ap.add_argument("--bench-mode", action="store_true")
    ap.add_argument("--config", type=str, default="", help="JSON config override")
    ap.add_argument("--dispatch", action="store_true",
                    help="auto-select config from the per-M DISPATCH table")
    args, _ = ap.parse_known_args()
    if args.bench_mode:
        args.mode = "bench"

    torch.cuda.set_device(0)
    dev = "cuda"
    cfg = get_config(args.config, M=args.M, use_dispatch=args.dispatch)
    print(f"config: {json.dumps(cfg)}")
    print(f"shape: M={args.M} N={args.N} K={args.K}")

    gen = torch.Generator(device=dev)
    gen.manual_seed(SEED)
    a = torch.rand((args.M, args.K), generator=gen, device=dev, dtype=torch.bfloat16)
    b = torch.rand((args.N, args.K), generator=gen, device=dev, dtype=torch.bfloat16)

    M, N, K = args.M, args.N, args.K
    flops = 2 * M * N * K
    hbm_bytes = (M * K + N * K + M * N) * 2  # bf16 in/out
    print(f"flops: {flops}")
    print(f"hbm_bytes: {hbm_bytes}")

    # correctness
    out = run_kernel(a, b, cfg)
    torch.cuda.synchronize()
    ref = torch.mm(a.float(), b.float().t())
    err = ref - out.float()
    sig = (ref**2).sum().item()
    noise = (err**2).sum().item()
    snr = 10.0 * (torch.log10(torch.tensor(sig / max(noise, 1e-30)))).item()
    print(f"SNR: {snr:.2f} dB")

    if args.mode == "test":
        return

    if args.mode == "profile":
        # Pure device-time capture for rocprofv3 --kernel-trace: no per-iter
        # events/sync so the profiler records clean kernel durations.
        out_buf = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
        for _ in range(args.warmup):
            run_kernel(a, b, cfg, out=out_buf)
        torch.cuda.synchronize()
        for _ in range(args.iters):
            run_kernel(a, b, cfg, out=out_buf)
        torch.cuda.synchronize()
        print(f"profile_iters: {args.iters}")
        return

    # bench: batch `inner` back-to-back launches between one event pair to
    # keep the GPU queue full (hide host dispatch latency -> steady-state
    # device time), report per-call time.
    inner = args.inner
    out_buf = torch.empty((M, N), dtype=torch.bfloat16, device=dev)
    for _ in range(args.warmup):
        run_kernel(a, b, cfg, out=out_buf)
    torch.cuda.synchronize()

    times = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.iters):
        start.record()
        for _ in range(inner):
            run_kernel(a, b, cfg, out=out_buf)
        end.record()
        end.synchronize()
        ms = start.elapsed_time(end) / inner
        times.append(ms)
        print(f"wall_ms: {ms:.5f}")
    med = statistics.median(times)
    print(f"median_ms: {med:.5f}")
    print(f"median_us: {med*1000:.3f}")


if __name__ == "__main__":
    main()
