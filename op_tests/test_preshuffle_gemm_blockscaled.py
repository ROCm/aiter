"""Tests for preshuffle_gemm_blockscaled. Style mirrors op_tests/test_mhc.py."""

import argparse

import pandas as pd
import torch

import aiter.test_common as testc
from aiter.ops.flydsl.kernels.preshuffle_gemm_blockscaled import (
    compile_preshuffle_gemm_blockscaled,
    compile_preshuffle_gemm_blockscaled_auto,
    compile_preshuffle_gemm_blockscaled_fp32,
)


# ────────────────────────────────────────────────────────────────────────────
# Packed-UE8M0 helpers (DeepGEMM SM100 ABI)
# ────────────────────────────────────────────────────────────────────────────
def _pack_ue8m0_bytes_to_i32(bytes_t: torch.Tensor) -> torch.Tensor:
    """4 UE8M0 bytes (one per consecutive 128-K-col group) → 1 i32, little-endian."""
    assert bytes_t.dtype == torch.uint8 and bytes_t.shape[-1] == 4
    return (
        bytes_t[..., 0].to(torch.int32)
        | (bytes_t[..., 1].to(torch.int32) << 8)
        | (bytes_t[..., 2].to(torch.int32) << 16)
        | (bytes_t[..., 3].to(torch.int32) << 24)
    )


def _fp32_to_ue8m0_byte(x_fp32: torch.Tensor) -> torch.Tensor:
    """Round abs amax up to next power-of-2 and encode the exponent as uint8."""
    bits = x_fp32.float().abs().view(torch.int32)
    exp = ((bits + 0x400000) & 0xFF800000) >> 23
    return exp.clamp(0, 255).to(torch.uint8)


def build_inputs(M: int, N: int, K: int, device: str = "cuda", seed: int = 0):
    """Build (a_fp8, b_preshuf_fp8, sa_i32, sb_i32, c_ref_bf16) for one shape."""
    g = torch.Generator(device=device).manual_seed(seed)
    a_f32 = torch.randn(M, K, device=device, generator=g) * 0.3
    b_f32 = torch.randn(N, K, device=device, generator=g) * 0.3

    # A: per-row 1×128 UE8M0 scale (one scale per (M, K//128) tile of A)
    sa_amax = a_f32.abs().reshape(M, K // 128, 128).amax(-1)
    sa_byte = _fp32_to_ue8m0_byte(sa_amax)
    sa_pow2 = (2.0 ** (sa_byte.to(torch.int32) - 127)).float()
    a_scaled = a_f32 / sa_pow2.repeat_interleave(128, dim=-1).clamp_min(1e-30)
    a_fp8 = a_scaled.clamp(-448, 448).to(torch.float8_e4m3fn)
    sa_i32 = _pack_ue8m0_bytes_to_i32(sa_byte.reshape(M, K // 512, 4))

    # B: 128×128 UE8M0 scale (one scalar per (N//128, K//128) block)
    b_blk = b_f32.reshape(N // 128, 128, K // 128, 128).permute(0, 2, 1, 3)
    sb_amax = b_blk.abs().amax(dim=(-1, -2))
    sb_byte = _fp32_to_ue8m0_byte(sb_amax)
    sb_pow2 = (2.0 ** (sb_byte.to(torch.int32) - 127)).float()
    b_scaled = b_f32 / (
        sb_pow2.repeat_interleave(128, dim=0)
        .repeat_interleave(128, dim=1)
        .clamp_min(1e-30)
    )
    b_fp8_rowmajor = b_scaled.clamp(-448, 448).to(torch.float8_e4m3fn)
    sb_i32 = _pack_ue8m0_bytes_to_i32(sb_byte.reshape(N // 128, K // 512, 4))

    # Preshuffle B from (N, K) row-major into (N//16, K//64, 4, 16, 16) for the kernel.
    # Inner (16, 16) = (klane, kpack) per the kernel's stride layout:
    #   stride_nlane = 16 (innermost)
    #   stride_klane = 256
    #   stride_k0    = 1024
    #   stride_n0    = (K/64) * 1024
    # So (n0, k0, klane, nlane, kpack) accesses element
    #   B[n0*16 + nlane, k0*64 + klane*16 + kpack].
    b_view = b_fp8_rowmajor.reshape(N // 16, 16, K // 64, 4, 16)
    # axes after reshape: (n0, nlane, k0, klane, kpack)
    b_preshuf = b_view.permute(0, 2, 3, 1, 4).contiguous()
    # axes: (n0, k0, klane, nlane, kpack)

    # fp32 reference: dequant both, matmul, cast.
    a_dq = a_fp8.float() * sa_pow2.repeat_interleave(128, dim=-1)
    b_dq = b_fp8_rowmajor.float() * (
        sb_pow2.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    )
    c_ref = (a_dq @ b_dq.T).bfloat16()
    return a_fp8, b_preshuf, sa_i32, sb_i32, c_ref


# ────────────────────────────────────────────────────────────────────────────
# Tests
# ────────────────────────────────────────────────────────────────────────────
def test_build_inputs_shapes():
    """Sanity: build_inputs produces tensors of the shapes the kernel expects."""
    M, N, K = 256, 256, 1024
    a, b, sa, sb, c = build_inputs(M, N, K)
    assert a.shape == (M, K) and a.dtype == torch.float8_e4m3fn, (a.shape, a.dtype)
    assert b.shape == (N // 16, K // 64, 4, 16, 16) and b.dtype == torch.float8_e4m3fn, (
        b.shape,
        b.dtype,
    )
    assert sa.shape == (M, K // 512) and sa.dtype == torch.int32, (sa.shape, sa.dtype)
    assert sb.shape == (N // 128, K // 512) and sb.dtype == torch.int32, (sb.shape, sb.dtype)
    assert c.shape == (M, N) and c.dtype == torch.bfloat16, (c.shape, c.dtype)


def build_inputs_fp32(M: int, N: int, K: int, device: str = "cuda", seed: int = 0):
    """Like build_inputs, but returns sa/sb as fp32 power-of-2 scalars rather
    than packed-UE8M0 i32. Use for the fp32-scale launcher.

    Returns (a_fp8, b_preshuf_fp8, sa_fp32, sb_fp32, c_ref_bf16).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    a_f32 = torch.randn(M, K, device=device, generator=g) * 0.3
    b_f32 = torch.randn(N, K, device=device, generator=g) * 0.3

    # A scale: per-row 1×128 UE8M0, but stored as fp32 power-of-2.
    sa_amax = a_f32.abs().reshape(M, K // 128, 128).amax(-1)
    sa_byte = _fp32_to_ue8m0_byte(sa_amax)
    sa_pow2 = (2.0 ** (sa_byte.to(torch.int32) - 127)).float()  # fp32 (M, K//128)
    a_scaled = a_f32 / sa_pow2.repeat_interleave(128, dim=-1).clamp_min(1e-30)
    a_fp8 = a_scaled.clamp(-448, 448).to(torch.float8_e4m3fn)

    # B scale: 128×128 UE8M0, stored as fp32.
    b_blk = b_f32.reshape(N // 128, 128, K // 128, 128).permute(0, 2, 1, 3)
    sb_amax = b_blk.abs().amax(dim=(-1, -2))
    sb_byte = _fp32_to_ue8m0_byte(sb_amax)
    sb_pow2 = (2.0 ** (sb_byte.to(torch.int32) - 127)).float()  # fp32 (N//128, K//128)
    b_scaled = b_f32 / (
        sb_pow2.repeat_interleave(128, dim=0)
        .repeat_interleave(128, dim=1)
        .clamp_min(1e-30)
    )
    b_fp8_rowmajor = b_scaled.clamp(-448, 448).to(torch.float8_e4m3fn)

    # Same preshuffle as build_inputs.
    b_view = b_fp8_rowmajor.reshape(N // 16, 16, K // 64, 4, 16)
    b_preshuf = b_view.permute(0, 2, 3, 1, 4).contiguous()

    # fp32 reference (identical to ue8m0 path — scales are the same powers-of-2).
    a_dq = a_fp8.float() * sa_pow2.repeat_interleave(128, dim=-1)
    b_dq = b_fp8_rowmajor.float() * (
        sb_pow2.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
    )
    c_ref = (a_dq @ b_dq.T).bfloat16()
    return a_fp8, b_preshuf, sa_pow2.contiguous(), sb_pow2.contiguous(), c_ref


def _run_kernel(M, N, K, *, tile_m=128, tile_n=128, tile_k=128):
    """Compile + launch + return (c_out, c_ref) on cuda."""
    a, b, sa, sb, c_ref = build_inputs(M, N, K)
    c_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    launch = compile_preshuffle_gemm_blockscaled(
        M=M, N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
    )
    stream = torch.cuda.current_stream().cuda_stream
    launch(c_out, a, b, sa, sb, M, stream)
    torch.cuda.synchronize()
    return c_out, c_ref


def _check_close(c_out, c_ref, *, rel_tol=0.05, abs_tol=0.1):
    """Per-element check: err < max(abs_tol, rel_tol * |ref|).

    This is the correct metric for fp8 blockscaled GEMM:
      - rel_tol guards meaningful values (fp8 quant noise ~1/(2*448) ≈ 0.1%
        per element, accumulated over K → a few % at K~thousands).
      - abs_tol absorbs near-zero ref values where any tiny perturbation
        produces a large relative error (denominator is tiny).
    Returns dict with stats + a bool `passed`.
    """
    err = (c_out.float() - c_ref.float()).abs()
    ref_abs = c_ref.float().abs()
    rel = err / (ref_abs + 1e-6)
    # Per-element tolerance: whichever bound is looser is the binding one.
    tol = torch.maximum(
        torch.full_like(ref_abs, abs_tol),
        rel_tol * ref_abs,
    )
    fail_mask = err > tol
    n_fail = fail_mask.sum().item()
    n_total = err.numel()
    return {
        "max_abs": err.max().item(),
        "max_rel": rel.max().item(),
        "median_rel": rel.median().item(),
        "n_fail": n_fail,
        "n_total": n_total,
        "fail_frac": n_fail / n_total,
        "passed": n_fail == 0,
    }


def test_blockscaled_gemm_correctness_small():
    """End-to-end numerical check at the smallest valid shape."""
    M, N, K = 256, 256, 1024
    c_out, c_ref = _run_kernel(M, N, K)
    stats = _check_close(c_out, c_ref)
    print(
        f"\n[{M=} {N=} {K=}] max_abs={stats['max_abs']:.4f} "
        f"max_rel={stats['max_rel']:.4f} median_rel={stats['median_rel']:.4f} "
        f"n_fail={stats['n_fail']}/{stats['n_total']}"
    )
    assert stats["passed"], (
        f"{stats['n_fail']} of {stats['n_total']} elements failed "
        f"abs_tol=0.1 or rel_tol=5% — likely a layout/scale-packing bug; "
        f"suspects in order: B preshuffle perm, MFMA opsel byte index, "
        f"sa/sb LDS slot indexing."
    )


def _run_kernel_fp32(M, N, K, *, tile_m=128, tile_n=128, tile_k=128):
    """Compile+launch fp32-scale variant. Returns (c_out, c_ref)."""
    a, b, sa, sb, c_ref = build_inputs_fp32(M, N, K)
    c_out = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    launch = compile_preshuffle_gemm_blockscaled_fp32(
        M=M, N=N, K=K, tile_m=tile_m, tile_n=tile_n, tile_k=tile_k,
    )
    stream = torch.cuda.current_stream().cuda_stream
    launch(c_out, a, b, sa, sb, M, stream)
    torch.cuda.synchronize()
    return c_out, c_ref


def test_fp32_scale_correctness_small():
    """fp32-scale variant should match the UE8M0 path bit-for-bit (when scales
    are already powers of 2 — which they always are coming from build_inputs)."""
    M, N, K = 256, 256, 1024
    c_out, c_ref = _run_kernel_fp32(M, N, K)
    stats = _check_close(c_out, c_ref)
    print(
        f"\n[fp32-scale {M=} {N=} {K=}] max_abs={stats['max_abs']:.4f} "
        f"median_rel={stats['median_rel']:.4f} "
        f"n_fail={stats['n_fail']}/{stats['n_total']}"
    )
    assert stats["passed"], (
        f"{stats['n_fail']} of {stats['n_total']} elements failed; likely a bug "
        f"in the in-kernel fp32→UE8M0 conversion or the vec-4 fp32 load layout."
    )


def test_fp32_vs_ue8m0_bit_equivalence():
    """The fp32 path converts to UE8M0 in-kernel. When the host already has
    power-of-2 fp32 scales, the in-kernel conversion should produce the same
    UE8M0 bytes as the host's pre-packing, so kernel outputs should be EXACTLY
    bitwise identical between the two paths.
    """
    M, N, K = 256, 256, 1024
    # Use the same seed for both — same A/B data, same scale values.
    c_ue8m0, _ = _run_kernel(M, N, K)
    c_fp32, _ = _run_kernel_fp32(M, N, K)
    n_diff = (c_ue8m0 != c_fp32).sum().item()
    n_total = c_ue8m0.numel()
    max_diff = (c_ue8m0.float() - c_fp32.float()).abs().max().item()
    print(
        f"\n[fp32 vs ue8m0 bit-equivalence {M=} {N=} {K=}] "
        f"n_diff={n_diff}/{n_total}  max_abs_diff={max_diff:.6f}"
    )
    # max_abs_diff should be exactly 0 if the in-kernel conversion is correct.
    assert max_diff == 0.0, (
        f"fp32 and ue8m0 paths produced different outputs (max_abs_diff="
        f"{max_diff}, n_diff={n_diff}/{n_total}). The in-kernel fp32→UE8M0 "
        f"conversion is not bit-identical to the host-side packing."
    )


def test_blockscaled_gemm_correctness_sweep():
    """Verify a handful of shapes to catch shape-dependent regressions."""
    shapes = [
        (256, 256, 1024),
        (256, 256, 2048),
        (1024, 256, 1024),
        (256, 1024, 1024),
        (1024, 1024, 2048),
    ]
    failures = []
    for M, N, K in shapes:
        c_out, c_ref = _run_kernel(M, N, K)
        stats = _check_close(c_out, c_ref)
        print(
            f"[{M=:5d} {N=:5d} {K=:5d}] max_abs={stats['max_abs']:.4f} "
            f"median_rel={stats['median_rel']:.4f} "
            f"n_fail={stats['n_fail']}/{stats['n_total']} "
            f"({stats['fail_frac']*100:.3f}%)"
        )
        if not stats["passed"]:
            failures.append((M, N, K, stats))
    assert not failures, f"{len(failures)} shape(s) failed: " + "; ".join(
        f"M={M} N={N} K={K} n_fail={s['n_fail']}" for (M, N, K, s) in failures
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("-n", type=int, nargs="+", default=[256, 1024])
    parser.add_argument("-k", type=int, nargs="+", default=[1024, 2048])
    args = parser.parse_args()
    rows = []
    for M in args.m:
        for N in args.n:
            for K in args.k:
                if M % 128 or N % 128 or K % 1024:
                    continue
                c_out, c_ref = _run_kernel(M, N, K)
                stats = _check_close(c_out, c_ref)
                rows.append({
                    "M": M, "N": N, "K": K,
                    "max_abs": round(stats["max_abs"], 4),
                    "median_rel": round(stats["median_rel"], 4),
                    "n_fail": stats["n_fail"],
                    "fail_pct": round(stats["fail_frac"] * 100, 3),
                })
    print(pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
