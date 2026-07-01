# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Profiling driver for the sparse paged-decode attention kernels.

Two Triton kernels are exercised:
  _pa_decode_sparse        — split-K main kernel (grid: T x n_head_blocks x KV_SPLITS)
  _pa_decode_sparse_reduce — log-sum-exp combine  (grid: T x H)

Usage examples
--------------
# Default DSv4 shapes, bf16 KV, print table (with verification):
  python bench_pa_decode_sparse.py

# Compare Triton and FlyDSL side-by-side:
  python bench_pa_decode_sparse.py --backend both

# Skip verification (e.g. before rocprof):
  python bench_pa_decode_sparse.py --no-verify

# FlyDSL only:
  python bench_pa_decode_sparse.py --backend flydsl

# FP8 KV cache path (auto-dispatched in FlyDSL):
  python bench_pa_decode_sparse.py --kv_dtype fp8

# Single shape:
  python bench_pa_decode_sparse.py --T 32 --H 128 --D 576 --kv_len 2048

# Override split parameters:
  python bench_pa_decode_sparse.py --block_k 32 --kv_splits 8

# Profile main kernel only (skip the reduce step):
  python bench_pa_decode_sparse.py --skip_reduce

# rocprof wrapper:
  rocprof --stats python bench_pa_decode_sparse.py --T 32 --H 128 --D 576 --kv_len 2048 --no-verify

Kernels
-------
  - _pa_decode_sparse:        inner split-K attention loop over gathered KV tiles
  - _pa_decode_sparse_reduce: log-sum-exp combine of KV_SPLITS partial softmax states
"""

import os
import sys

# Make the script runnable directly (python path/to/bench.py) by putting the
# repo root on sys.path, so `import aiter` / `import op_tests` resolve without
# requiring PYTHONPATH or an installed aiter package.
_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import argparse
import sys

import torch
import triton

from aiter.ops.triton.attention.pa_decode_sparse import pa_decode_sparse
from aiter.ops.flydsl.kernels.pa_decode_sparse import flydsl_pa_decode_sparse
from _bench_timing import MeasureConfig, bench_graph, TimingStats

_ATOL = 5e-3
_RTOL = 5e-3

# ---------------------------------------------------------------------------
# DSv4 representative shapes
# ---------------------------------------------------------------------------

DSV4_SHAPES = [
    #(label,       T,    H,    D,   kv_len)

    # DS V3.2 shapes
    ("T1-kv2k-d576",      1,  128,  576,   2048),
    ("T2-kv2k-d576",      2,  128,  576,   2048),
    ("T4-kv2k-d576",      4,  128,  576,   2048),
    ("T16-kv2k-d576",    16,  128,  576,   2048),
    ("T32-kv2k-d576",    32,  128,  576,   2048),
    ("T64-kv2k-d576",    64,  128,  576,   2048),
    ("T128-kv2k-d576",  128,  128,  576,   2048),
    ("T256-kv2k-d576",  256,  128,  576,   2048),
    ("T32-kv1k-d576",    32,  128,  576,   1024),
    ("T32-kv4k-d576",    32,  128,  576,   4096),
    ("T32-kv8k-d576",    32,  128,  576,   8192),

    # DS V4 shapes
    ("T1-kv2k-d512",      1,  128,  512,   2048),
    ("T2-kv2k-d512",      2,  128,  512,   2048),
    ("T4-kv2k-d512",      4,  128,  512,   2048),
    ("T16-kv2k-d512",    16,  128,  512,   2048),
    ("T32-kv2k-d512",    32,  128,  512,   2048),
    ("T64-kv2k-d512",    64,  128,  512,   2048),
    ("T128-kv2k-d512",  128,  128,  512,   2048),
    ("T256-kv2k-d512",  256,  128,  512,   2048),
    ("T32-kv1k-d512",    32,  128,  512,   1024),
    ("T32-kv4k-d512",    32,  128,  512,   4096),
    ("T32-kv8k-d512",    32,  128,  512,   8192),
]

_FP8_DTYPE = torch.float8_e4m3fnuz
_FP8_GROUP_SIZE = 64


# ---------------------------------------------------------------------------
# Input builders
# ---------------------------------------------------------------------------

def make_inputs_bf16(T, H, D, kv_len, seed=42):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    total_pages = T * kv_len

    q          = torch.randn(T, H, D, dtype=torch.bfloat16, device=device) * 0.5
    unified_kv = torch.randn(total_pages, D, dtype=torch.bfloat16, device=device) * 0.5
    attn_sink  = torch.zeros(H, dtype=torch.float32, device=device)

    kv_lens = torch.full((T,), kv_len, dtype=torch.int64, device=device)
    indptr  = torch.zeros(T + 1, dtype=torch.int32, device=device)
    indptr[1:] = kv_lens.cumsum(0).to(torch.int32)

    total_indices = int(indptr[-1].item())
    indices = torch.randint(0, total_pages, (total_indices,), dtype=torch.int32, device=device)

    return q, unified_kv, None, indices, indptr, attn_sink, float(D ** -0.5)


def make_inputs_fp8(T, H, D, kv_len, seed=42):
    q, unified_kv_bf16, _, indices, indptr, attn_sink, scale = make_inputs_bf16(
        T, H, D, kv_len, seed
    )
    total_pages = unified_kv_bf16.shape[0]
    num_groups  = D // _FP8_GROUP_SIZE

    kv_f32  = unified_kv_bf16.float().view(total_pages, num_groups, _FP8_GROUP_SIZE)
    amax    = kv_f32.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    fp8_max = torch.finfo(_FP8_DTYPE).max
    scales  = (amax / fp8_max).squeeze(-1).to(torch.float32)
    kv_fp8  = (kv_f32 / amax * fp8_max).view(total_pages, D).to(_FP8_DTYPE)

    return q, kv_fp8, scales, indices, indptr, attn_sink, scale


# ---------------------------------------------------------------------------
# Bandwidth helpers
# ---------------------------------------------------------------------------

def _bytes_bf16(T, H, D, kv_len):
    return T*H*D*2 + T*kv_len*D*2 + T*H*D*2 + T*kv_len*4


def _bytes_fp8(T, H, D, kv_len):
    return T*H*D*2 + T*kv_len*D*1 + T*kv_len*(D//_FP8_GROUP_SIZE)*4 + T*H*D*2 + T*kv_len*4


# ---------------------------------------------------------------------------
# Torch reference
# ---------------------------------------------------------------------------

def _torch_reference(q, unified_kv, indices, indptr, sink, scale):
    """Pure-torch sparse attention reference. Returns bf16 [T, H, D]."""
    T, H, D   = q.shape
    device    = q.device
    indptr64  = indptr.to(torch.int64)
    out       = torch.zeros(T, H, D, dtype=torch.float32, device=device)
    for t in range(T):
        s, e  = int(indptr64[t].item()), int(indptr64[t + 1].item())
        kv_t  = unified_kv[indices[s:e].long()].float()
        q_t   = q[t].float()
        scores_aug = torch.cat(
            [sink.float().unsqueeze(1), (q_t @ kv_t.T) * scale], dim=1
        )
        probs = torch.softmax(scores_aug, dim=-1)
        v_aug = torch.cat(
            [torch.zeros(1, D, dtype=torch.float32, device=device), kv_t], dim=0
        )
        out[t] = probs @ v_aug
    return out.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Per-backend run / bench helpers
# ---------------------------------------------------------------------------

def _run_triton(q, unified_kv, kv_scales, indices, indptr, sink, scale, kv_splits):
    return pa_decode_sparse(
        q, unified_kv, indices, indptr, sink, scale,
        kv_scales=kv_scales, block_h=None, kv_splits=kv_splits,
        has_invalid=False, skip_reduce=False,
    )


def _run_flydsl(q, unified_kv, kv_scales, indices, indptr, sink, scale, kv_splits,
               mfma_shape="16x16x16", n_waves=4):
    return flydsl_pa_decode_sparse(
        q, unified_kv, indices, indptr, sink, scale,
        kv_scales=kv_scales, kv_splits=kv_splits, skip_reduce=False,
        mfma_shape=mfma_shape, n_waves=n_waves,
    )


def _time_ms(closure, cfg: MeasureConfig) -> float:
    """Return median latency in ms using HIP-graph replay."""
    stats: TimingStats = bench_graph(closure, cfg)
    return stats.median_us / 1000.0


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

_CHK_PASS  = "PASS"
_CHK_FAIL  = "FAIL"
_CHK_SKIP  = "N/A"   # --no-verify
_CHK_UNSUP = "N/A"   # dtype/D not supported


def _allclose(out, ref):
    max_err = (out.float() - ref.float()).abs().max().item()
    return torch.allclose(out.float(), ref.float(), atol=_ATOL, rtol=_RTOL), max_err


def _verify_backend(backend_name, out, ref, triton_out=None):
    """Print per-check lines; return overall pass/fail."""
    ok_total = True

    if ref is not None:
        passed, max_err = _allclose(out, ref)
        ok_total = ok_total and passed
        print(f"    {backend_name} vs torch-ref: {'PASS' if passed else 'FAIL'}  maxErr={max_err:.2e}")
    else:
        print(f"    {backend_name} vs torch-ref: SKIP (no torch ref for fp8)")

    if backend_name == "flydsl" and triton_out is not None:
        passed2, max_err2 = _allclose(out, triton_out)
        ok_total = ok_total and passed2
        print(f"    flydsl vs triton:     {'PASS' if passed2 else 'FAIL'}  maxErr={max_err2:.2e}")

    return ok_total


# ---------------------------------------------------------------------------
# Combined bench + verify per shape
# ---------------------------------------------------------------------------

def run_shape(label, T, H, D, kv_len, backend, kv_dtype, block_k, kv_splits,
              skip_reduce, cfg, do_verify, mfma_shape="16x16x16", n_waves=4):
    """Benchmark and optionally verify one shape.

    Returns:
        bench_results : dict  backend -> (ms, GB/s, TFLOPS) or None
        chk_results   : dict  backend -> _CHK_* string
    """
    if kv_dtype == "fp8":
        q, unified_kv, kv_scales, indices, indptr, sink, scale = make_inputs_fp8(T, H, D, kv_len)
        bw_bytes = _bytes_fp8(T, H, D, kv_len)
    else:
        q, unified_kv, kv_scales, indices, indptr, sink, scale = make_inputs_bf16(T, H, D, kv_len)
        bw_bytes = _bytes_bf16(T, H, D, kv_len)

    # block_k for triton; for flydsl it's auto-selected by mfma_shape
    bk     = block_k if block_k else (16 if D >= 256 else 32)
    n_tiles = T * triton.cdiv(kv_len, bk)
    flops  = 2.0 * n_tiles * H * D * bk + 2.0 * n_tiles * H * bk

    bench_results: dict = {}
    chk_results:   dict = {}

    if mfma_shape is not None:
        from aiter.ops.flydsl.kernels.pa_decode_sparse import _MFMA_MAP
        mfma = _MFMA_MAP[mfma_shape]
        d_per_wave = D // n_waves if D % n_waves == 0 else 1
        flydsl_ok = (
            D % 64 == 0
            and H % mfma.MFMA_M == 0
            and D % n_waves == 0
            and d_per_wave % mfma.MFMA_N == 0
        )
    else:
        flydsl_ok = True  # mfma_shape=None means auto-select; assume it's valid

    # --- verification ---
    triton_out = flydsl_out = ref = None
    if do_verify:
        print(f"  [verify] {label}  T={T} H={H} D={D} kv_len={kv_len}")
        if backend in ("triton", "both"):
            triton_out = _run_triton(q, unified_kv, kv_scales, indices, indptr, sink, scale,
                                     kv_splits)
            if kv_dtype != "fp8":
                ref = _torch_reference(q, unified_kv, indices, indptr, sink, scale)
                ok = _verify_backend("triton", triton_out, ref)
            else:
                ok = True  # fp8 Triton is reference; skip torch comparison
            chk_results["triton"] = _CHK_PASS if ok else _CHK_FAIL

        if backend in ("flydsl", "both"):
            if not flydsl_ok:
                chk_results["flydsl"] = _CHK_UNSUP
            else:
                if ref is None and kv_dtype != "fp8":
                    ref = _torch_reference(q, unified_kv, indices, indptr, sink, scale)
                flydsl_out = _run_flydsl(q, unified_kv, kv_scales, indices, indptr, sink, scale,
                                         kv_splits, mfma_shape=mfma_shape, n_waves=n_waves)
                ok = _verify_backend("flydsl", flydsl_out, ref,
                                     triton_out=triton_out if backend == "both" else None)
                chk_results["flydsl"] = _CHK_PASS if ok else _CHK_FAIL
    else:
        if backend in ("triton", "both"):
            chk_results["triton"] = _CHK_SKIP
        if backend in ("flydsl", "both"):
            chk_results["flydsl"] = _CHK_SKIP if flydsl_ok else _CHK_UNSUP

    # --- benchmark ---
    if backend in ("triton", "both"):
        def fn_triton():
            pa_decode_sparse(
                q, unified_kv, indices, indptr, sink, scale,
                kv_scales=kv_scales, block_h=None, kv_splits=kv_splits,
                has_invalid=False, skip_reduce=skip_reduce,
            )
        ms = _time_ms(fn_triton, cfg)
        bench_results["triton"] = (ms, bw_bytes / (ms * 1e-3) * 1e-9, flops / (ms * 1e-3) * 1e-12)

    if backend in ("flydsl", "both"):
        if not flydsl_ok:
            bench_results["flydsl"] = None
        else:
            def fn_flydsl():
                flydsl_pa_decode_sparse(
                    q, unified_kv, indices, indptr, sink, scale,
                    kv_scales=kv_scales, kv_splits=kv_splits, skip_reduce=skip_reduce,
                    mfma_shape=mfma_shape, n_waves=n_waves,
                )
            ms = _time_ms(fn_flydsl, cfg)
            bench_results["flydsl"] = (ms, bw_bytes / (ms * 1e-3) * 1e-9,
                                       flops / (ms * 1e-3) * 1e-12)

    return bench_results, chk_results


# ---------------------------------------------------------------------------
# Table formatting
# ---------------------------------------------------------------------------

_HDR  = f"{'Shape':<14} {'T':>4} {'H':>4} {'D':>4} {'kv_len':>7}   {'ms':>8} {'GB/s':>8} {'TFLOPS':>8}   verification"
_SEP  = "-" * len(_HDR)


def _fmt_perf(result):
    if result is None:
        return f"{'N/A':>8} {'N/A':>8} {'N/A':>8}"
    ms, bw, tflops = result
    return f"{ms:>8.3f} {bw:>8.1f} {tflops:>8.3f}"


def _fmt_row(label, T, H, D, kv_len, perf, chk):
    return (
        f"{label:<14} {T:>4} {H:>4} {D:>4} {kv_len:>7}"
        f"   {_fmt_perf(perf)}   {chk}"
    )


def run_benchmark(args):
    if args.T and args.H and args.D and args.kv_len:
        shapes = [("custom", args.T, args.H, args.D, args.kv_len)]
    else:
        shapes = DSV4_SHAPES

    backend     = args.backend
    kv_dtype    = args.kv_dtype
    block_k     = args.block_k if args.block_k else None
    kv_splits   = args.kv_splits if args.kv_splits else None
    skip_reduce = args.skip_reduce
    do_verify   = not args.no_verify
    mfma_shape  = args.mfma_shape
    n_waves     = args.n_waves

    cfg = MeasureConfig(
        warmup_iters=args.warmup,
        bench_iters=args.bench_iters,
        graph_replay_iters=args.rep,
    )

    backends = ["triton", "flydsl"] if backend == "both" else [backend]

    # When mfma_shape == "all", benchmark all three FlyDSL variants:
    # 16x16x16/4w, 32x32x8/4w (D=512 only), 32x32x8/6w (D=576 only)
    _ALL_FLYDSL_VARIANTS = [
        ("16x16x16", 4),
        ("32x32x8",  4),   # D=512: D_PER_WAVE=128, 4 chunks
        ("32x32x8",  6),   # D=576: D_PER_WAVE=96,  3 chunks
    ]

    if mfma_shape == "all" and "flydsl" in backends:
        # Run triton once + all three FlyDSL variants; display multi-variant comparison
        all_bench_triton: dict = {}
        all_bench_variants: dict = {}  # (mfma_shape, n_waves) -> {label: (T,H,D,kv_len,res,chk)}
        all_pass = True

        for label, T, H, D, kv_len in shapes:
            # Triton
            if "triton" in backends:
                bench_res, chk_res = run_shape(
                    label, T, H, D, kv_len,
                    backend="triton", kv_dtype=kv_dtype,
                    block_k=block_k, kv_splits=kv_splits,
                    skip_reduce=skip_reduce, cfg=cfg,
                    do_verify=do_verify,
                )
                all_bench_triton[label] = (T, H, D, kv_len, bench_res, chk_res)
                for v in chk_res.values():
                    if v == _CHK_FAIL:
                        all_pass = False

            # FlyDSL variants
            for ms_shape, nw in _ALL_FLYDSL_VARIANTS:
                key = (ms_shape, nw)
                if key not in all_bench_variants:
                    all_bench_variants[key] = {}
                bench_res, chk_res = run_shape(
                    label, T, H, D, kv_len,
                    backend="flydsl", kv_dtype=kv_dtype,
                    block_k=block_k, kv_splits=kv_splits,
                    skip_reduce=skip_reduce, cfg=cfg,
                    do_verify=do_verify,
                    mfma_shape=ms_shape, n_waves=nw,
                )
                all_bench_variants[key][label] = (T, H, D, kv_len, bench_res, chk_res)
                for v in chk_res.values():
                    if v == _CHK_FAIL:
                        all_pass = False

        # Print comparison table
        _HDR_ALL = (
            f"{'Shape':<14} {'T':>4} {'H':>4} {'D':>4} {'kv_len':>7}"
            f"   {'triton':>8} {'fl-16/4w':>9} {'fl-32/4w':>9} {'fl-32/6w':>9}"
            f"   {'fl-16/tr':>9} {'fl-32/4w/tr':>11} {'fl-32/6w/tr':>11}"
        )
        _SEP_ALL = "-" * len(_HDR_ALL)
        print(f"\n[FlyDSL mfma_shape=all vs triton — TFLOPS]")
        print(_HDR_ALL)
        print(_SEP_ALL)
        for label, T, H, D, kv_len in shapes:
            tr_res = all_bench_triton.get(label, (T, H, D, kv_len, {}, {}))[4].get("triton")
            fl_results = {}
            for ms_shape, nw in _ALL_FLYDSL_VARIANTS:
                key = (ms_shape, nw)
                fl_results[key] = all_bench_variants[key][label][4].get("flydsl")

            tr_tf = f"{tr_res[2]:>8.3f}" if tr_res else f"{'N/A':>8}"
            fl_tfs = []
            for ms_shape, nw in _ALL_FLYDSL_VARIANTS:
                r = fl_results[(ms_shape, nw)]
                fl_tfs.append(f"{r[2]:>9.3f}" if r else f"{'N/A':>9}")
            ratios = []
            for ms_shape, nw in _ALL_FLYDSL_VARIANTS:
                r = fl_results[(ms_shape, nw)]
                if r and tr_res:
                    ratios.append(f"{r[2]/tr_res[2]:>9.3f}×")
                else:
                    ratios.append(f"{'N/A':>9}")
            print(
                f"{label:<14} {T:>4} {H:>4} {D:>4} {kv_len:>7}"
                f"   {tr_tf} {' '.join(fl_tfs)}"
                f"   {' '.join(ratios)}"
            )

        if do_verify:
            print()
            print("Verification:", "ALL PASS" if all_pass else "SOME FAILURES")
        return

    # --- standard single-variant path ---
    # Run each shape once (verification + benchmark for all active backends),
    # then display a separate table per backend.
    all_bench: dict = {}   # label -> (bench_res, chk_res)
    all_pass = True
    for label, T, H, D, kv_len in shapes:
        bench_res, chk_res = run_shape(
            label, T, H, D, kv_len,
            backend=backend, kv_dtype=kv_dtype,
            block_k=block_k, kv_splits=kv_splits,
            skip_reduce=skip_reduce, cfg=cfg,
            do_verify=do_verify,
            mfma_shape=mfma_shape, n_waves=n_waves,
        )
        all_bench[label] = (T, H, D, kv_len, bench_res, chk_res)
        for v in chk_res.values():
            if v == _CHK_FAIL:
                all_pass = False

    for b in backends:
        print(f"\n[{b}]")
        print(_HDR)
        print(_SEP)
        for label, T, H, D, kv_len in shapes:
            _, _, _, _, bench_res, chk_res = all_bench[label]
            chk = chk_res.get(b, _CHK_SKIP)
            print(_fmt_row(label, T, H, D, kv_len, bench_res.get(b), chk))

    if backend == "both":
        _HDR_CMP = (
            f"{'Shape':<14} {'T':>4} {'H':>4} {'D':>4} {'kv_len':>7}"
            f"   {'TFLOPS(fl)':>10} {'TFLOPS(tr)':>10}"
            f"   {'verify(fl)':>10} {'verify(tr)':>10}"
            f"   {'fl/tr':>6}"
        )
        _SEP_CMP = "-" * len(_HDR_CMP)
        print(f"\n[comparison: flydsl ({mfma_shape}/{n_waves}w) vs triton]")
        print(_HDR_CMP)
        print(_SEP_CMP)
        for label, T, H, D, kv_len in shapes:
            _, _, _, _, bench_res, chk_res = all_bench[label]
            fl = bench_res.get("flydsl")
            tr = bench_res.get("triton")
            fl_tflops = f"{fl[2]:>10.3f}" if fl else f"{'N/A':>10}"
            tr_tflops = f"{tr[2]:>10.3f}" if tr else f"{'N/A':>10}"
            fl_chk = f"{chk_res.get('flydsl', _CHK_SKIP):>10}"
            tr_chk = f"{chk_res.get('triton', _CHK_SKIP):>10}"
            if fl and tr:
                ratio = f"{fl[2] / tr[2]:>6.3f}"
            else:
                ratio = f"{'N/A':>6}"
            print(
                f"{label:<14} {T:>4} {H:>4} {D:>4} {kv_len:>7}"
                f"   {fl_tflops} {tr_tflops}"
                f"   {fl_chk} {tr_chk}"
                f"   {ratio}"
            )

    if do_verify:
        print()
        print("Verification:", "ALL PASS" if all_pass else "SOME FAILURES")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark _pa_decode_sparse + _pa_decode_sparse_reduce",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--T",      type=int, default=None, help="Number of decode tokens")
    p.add_argument("--H",      type=int, default=None, help="Number of query heads")
    p.add_argument("--D",      type=int, default=None, help="Head dimension")
    p.add_argument("--kv_len", type=int, default=None, help="KV slots per token")

    p.add_argument("--backend", choices=["triton", "flydsl", "both"], default="both",
                   help="Kernel backend to benchmark (default: triton)")

    p.add_argument("--block_k",   type=int, default=0,
                   help="Override BLOCK_K (default: 16 for D>=256, else 32)")
    p.add_argument("--kv_splits", type=int, default=0,
                   help="Override KV_SPLITS (default: auto)")
    p.add_argument("--skip_reduce", action="store_true",
                   help="Profile main kernel only, skip the reduce step")

    p.add_argument("--kv_dtype", choices=["bf16", "fp8"], default="bf16",
                   help="KV cache dtype (default: bf16); fp8 is Triton-only")

    p.add_argument("--mfma_shape", choices=["16x16x16", "32x32x8", "all"], default=None,
                   help="MFMA instruction shape for FlyDSL (default: 16x16x16); "
                        "'all' benchmarks 16x16x16/4w + 32x32x8/4w(D=512) + 32x32x8/6w(D=576)")
    p.add_argument("--n_waves", type=int, default=None,
                   help="Number of wavefronts per CTA for FlyDSL (default: 4); "
                        "ignored when --mfma_shape all")

    p.add_argument("--no-verify", action="store_true",
                   help="Disable correctness checks (useful for profiling)")

    p.add_argument("--warmup",      type=int, default=10,
                   help="Graph warmup iterations (default: 10)")
    p.add_argument("--bench-iters", type=int, default=20,
                   help="Outer timing samples (default: 20)")
    p.add_argument("--rep",         type=int, default=50,
                   help="Graph replays per sample (default: 50)")

    return p.parse_args()


def main():
    if not torch.cuda.is_available():
        print("ERROR: CUDA/HIP device required", file=sys.stderr)
        return 1
    args = parse_args()
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
