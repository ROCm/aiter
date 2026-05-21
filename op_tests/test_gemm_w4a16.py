# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# AIESW-32176: smoke test + microbench for the CK WMMA W4A16 b_scale GEMM.
#
# Builds AWQ-style packed weights (vLLM ExLlama [N, K//8] int32 layout),
# repacks to CK pk_i4_v3 b_scale layout [K0, N, K1//2] int8 (byte-equivalent
# verified in /scratch/marcusr/aiesw32176/verify_layout.py), and compares the
# CK kernel output against a torch fp16/bf16 reference computed from the
# dequantized weights.
#
# Cases: (sym, asym) x (fp16, bf16). bf16 is skipped if the build does not
# include the bf16 kernel template instantiation.

import sys
import argparse

import torch

import aiter

torch.manual_seed(0)
DEVICE = "cuda"

# Default shape: AIESW-32176 target (Qwen3-4B gate_up_proj, G=128). The G=32
# variant uses the Qwen3-VL-4B gate_up_proj shape (same N=19456, K=2560)
# from AIESW-32282 — K is divisible by both 32 and 128 so we keep N/K
# identical across G to make the bench numbers directly comparable.
M, N, K = 2048, 19456, 2560
KPerBlock = 32

# fp16 GEMM with K=2560 + ~0.05 magnitude products has per-output sigma ~2.5;
# 1e-2 relative agreement is realistic for fp16 accumulation in a fused dequant
# path.
TOL_REL = 5e-3


def _exllama_pack(nibbles: torch.Tensor) -> torch.Tensor:
    """vLLM ExLlama pack: nibbles [N, K] int -> [N, K//8] int32, shifts
    [0, 16, 4, 20, 8, 24, 12, 28] (interleave low/high halves).
    """
    N_, K_ = nibbles.shape
    shifts = torch.tensor(
        [(k // 2) * 4 + (k % 2) * 16 for k in range(8)],
        dtype=torch.int64,
        device=nibbles.device,
    )
    nib_8 = nibbles.view(N_, K_ // 8, 8).to(torch.int64)
    return ((nib_8 & 0xF) << shifts.view(1, 1, 8)).sum(dim=-1).to(torch.int32)


def _repack_vllm_to_ck_b_scale(w_q_vllm: torch.Tensor, kperblock: int = KPerBlock):
    """vLLM ExLlama [N, K//8] int32 -> CK pk_i4_v3 b_scale [K0, N, K1//2] int8.

    Pure metadata + axis swap; no nibble reshuffle. Verified byte-identical to
    CK's expected layout in /scratch/marcusr/aiesw32176/verify_layout.py.
    """
    N_, K_div_8 = w_q_vllm.shape
    K_ = K_div_8 * 8
    K0 = K_ // kperblock
    return (
        w_q_vllm.reshape(N_, K0, kperblock // 8)
        .permute(1, 0, 2)
        .contiguous()
        .view(torch.int8)
    )


def _build_case(asym: bool, dtype: torch.dtype, G: int):
    """Returns (a, w_q_ck, w_s, scaled_zp_or_None, w_dequant_for_ref)."""
    # fp16 quantization math is fine; cast to dtype at the end.
    w = (torch.randn(N, K, dtype=torch.float16, device=DEVICE) * 0.1).contiguous()
    w_grp = w.view(N, K // G, G)
    if asym:
        w_min = w_grp.amin(dim=-1)
        w_max = w_grp.amax(dim=-1)
        scale = ((w_max - w_min) / 15.0).clamp(min=1e-4).contiguous()
        zp = (-w_min / scale).round().clamp(0, 15).to(torch.int32)
        nibbles = (
            (w_grp / scale.unsqueeze(-1) + zp.unsqueeze(-1).float())
            .round()
            .to(torch.int32)
            .clamp(0, 15)
            .view(N, K)
        )
        w_dequant = (
            (nibbles.float() - zp.float().repeat_interleave(G, dim=1))
            * scale.float().repeat_interleave(G, dim=1)
        )
        scaled_zp = ((zp.float() - 8.0) * scale.float()).to(dtype).contiguous()
    else:
        scale = (w_grp.abs().amax(dim=-1) / 7.0).clamp(min=1e-4).contiguous()
        nibbles = (w_grp / scale.unsqueeze(-1)).round().to(torch.int32) + 8
        nibbles = nibbles.clamp(0, 15).view(N, K)
        w_dequant = (
            (nibbles.float() - 8.0)
            * scale.float().repeat_interleave(G, dim=1)
        )
        scaled_zp = None

    w_q_vllm = _exllama_pack(nibbles)
    w_q_ck = _repack_vllm_to_ck_b_scale(w_q_vllm, KPerBlock)
    w_s = scale.to(dtype).contiguous()
    a = (torch.randn(M, K, dtype=dtype, device=DEVICE) * 0.5).contiguous()
    return a, w_q_ck, w_s, scaled_zp, w_dequant.to(dtype)


def _run_one(asym: bool, dtype: torch.dtype, G: int,
             pre_dequant_to_lds: bool = False,
             truncate_bf16_round: bool = False) -> tuple[bool, float]:
    pdl_tag = "PDL " if pre_dequant_to_lds else "    "
    tbt_tag = "TRUNC" if truncate_bf16_round else "RTE  "
    label = (
        f"G={G:<3d} {'asym' if asym else 'sym '} "
        f"{str(dtype).split('.')[-1]:7s} {pdl_tag}{tbt_tag}"
    )
    try:
        a, w_q_ck, w_s, scaled_zp, w_dequant = _build_case(asym=asym, dtype=dtype, G=G)
        Y = torch.empty((M, N), dtype=dtype, device=DEVICE)
        aiter.gemm_w4a16(
            a, w_q_ck, w_s, Y, G, scaled_zp,
            pre_dequant_to_lds=pre_dequant_to_lds,
            truncate_bf16_round=truncate_bf16_round,
        )
        torch.cuda.synchronize()
    except Exception as exc:  # noqa: BLE001
        print(f"  [{label}] SKIP — {type(exc).__name__}: {exc}")
        return False, float("nan")

    out_ref = torch.nn.functional.linear(a.float(), w_dequant.float()).to(dtype)
    diff = (Y.float() - out_ref.float()).abs()
    rel = diff.max().item() / (out_ref.float().abs().max().item() + 1e-9)
    ok = rel < TOL_REL
    print(
        f"  [{label}] max_abs={diff.max().item():.4f} "
        f"mean_abs={diff.mean().item():.4f} "
        f"max_abs/max_ref={rel:.3e} {'PASS' if ok else 'FAIL'}"
    )
    return ok, rel


def _bench_one(asym: bool, dtype: torch.dtype, G: int, reps: int = 100,
               pre_dequant_to_lds: bool = False,
               truncate_bf16_round: bool = False):
    pdl_tag = "PDL " if pre_dequant_to_lds else "    "
    tbt_tag = "TRUNC" if truncate_bf16_round else "RTE  "
    label = (
        f"G={G:<3d} {'asym' if asym else 'sym '} "
        f"{str(dtype).split('.')[-1]:7s} {pdl_tag}{tbt_tag}"
    )
    try:
        a, w_q_ck, w_s, scaled_zp, _ = _build_case(asym=asym, dtype=dtype, G=G)
        Y = torch.empty((M, N), dtype=dtype, device=DEVICE)
        # warmup
        for _ in range(10):
            aiter.gemm_w4a16(
                a, w_q_ck, w_s, Y, G, scaled_zp,
                pre_dequant_to_lds=pre_dequant_to_lds,
                truncate_bf16_round=truncate_bf16_round,
            )
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(reps):
            aiter.gemm_w4a16(
                a, w_q_ck, w_s, Y, G, scaled_zp,
                pre_dequant_to_lds=pre_dequant_to_lds,
                truncate_bf16_round=truncate_bf16_round,
            )
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / reps
        flops = 2.0 * M * N * K
        tflops = flops / (ms * 1e-3) / 1e12
        print(f"  [{label}] {ms:.3f} ms/iter   {tflops:.1f} TFLOPS  "
              f"(M={M} N={N} K={K})")
    except Exception as exc:  # noqa: BLE001
        print(f"  [{label}] SKIP — {type(exc).__name__}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="aiter CK W4A16 b_scale smoke test")
    parser.add_argument("--no-bf16", action="store_true",
                        help="Skip bf16 cases (use if CK was built fp16-only).")
    parser.add_argument("--reps", type=int, default=100)
    parser.add_argument(
        "--groups",
        type=int,
        nargs="+",
        default=[128, 32],
        help="AWQ group_size values to test (default: 128 32).",
    )
    parser.add_argument(
        "--pre-dequant",
        action="store_true",
        help=(
            "Also exercise the PreDequantToLDS=true path. Currently STUBBED "
            "and expected to TORCH_CHECK at runtime — see TODO(AIESW-32282)."
        ),
    )
    parser.add_argument(
        "--truncate-bf16",
        action="store_true",
        help=(
            "Also exercise the TruncateBf16Round=true path. True runtime "
            "template flag — both flavors live in the same .so as distinct "
            "template-mangled symbols. fp16 cases pass through unchanged."
        ),
    )
    args = parser.parse_args()

    dtypes = [torch.float16] + ([] if args.no_bf16 else [torch.bfloat16])
    base_cases = [
        (asym, dt, G)
        for G in args.groups
        for dt in dtypes
        for asym in (False, True)
    ]
    # PDL flag dimension: always include false; include true iff --pre-dequant.
    pdl_flags = (False,) + ((True,) if args.pre_dequant else ())
    # TruncateBf16Round flag dimension: always include false; include true iff
    # --truncate-bf16.
    tbt_flags = (False,) + ((True,) if args.truncate_bf16 else ())
    cases = [
        (asym, dt, G, pdl, tbt)
        for asym, dt, G in base_cases
        for pdl in pdl_flags
        for tbt in tbt_flags
    ]

    print(f"== Correctness (M={M} N={N} K={K} G={args.groups}) ==")
    all_ok = True
    for asym, dt, G, pdl, tbt in cases:
        ok, _ = _run_one(asym=asym, dtype=dt, G=G,
                         pre_dequant_to_lds=pdl,
                         truncate_bf16_round=tbt)
        # PDL=True is a known-stub failure; don't gate the overall result on
        # it. TruncateBf16Round=true is correctness-equivalent to RTE
        # within the 5e-3 op-test tolerance, so it IS gated.
        if not pdl:
            all_ok &= ok

    print(f"\n== Timing ({args.reps} reps each) ==")
    for asym, dt, G, pdl, tbt in cases:
        _bench_one(asym=asym, dtype=dt, G=G, reps=args.reps,
                   pre_dequant_to_lds=pdl,
                   truncate_bf16_round=tbt)

    print(f"\nResult: {'PASS' if all_ok else 'FAIL'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
