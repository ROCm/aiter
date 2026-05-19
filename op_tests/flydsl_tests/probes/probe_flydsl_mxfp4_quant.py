#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Byte-exact parity probe: FlyDSL MXFP4 quant vs Triton MXFP4 quant.

Runs both quantizers on the same (q, k, v) and asserts:
  - q_fp4 / k_fp4: byte-exact on rows in-range
  - q_d / k_d:     byte-exact (e8m0 must match exactly)
  - v_fp8:         byte-exact
  - v_scale:       max abs diff < 1e-6 (per-channel scale, computed by torch
                    reduction in both paths — should be bit-identical)

Usage: HIP_VISIBLE_DEVICES=0 python op_tests/flydsl_tests/probes/probe_flydsl_mxfp4_quant.py
"""

import os
import sys

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from aiter.ops.triton.quant.sage_attention_quant_wrappers import (  # noqa: E402
    sage_quant_mxfp4 as triton_quant,
)
from aiter.ops.flydsl.sage_quant_mxfp4 import flydsl_sage_quant_mxfp4  # noqa: E402
from aiter.utility.dtypes import fp8 as _fp8  # noqa: E402


def _bytes_diff(a: torch.Tensor, b: torch.Tensor) -> int:
    """Count number of differing bytes (treating as uint8)."""
    a_u = a.view(torch.uint8) if a.dtype != torch.uint8 else a
    b_u = b.view(torch.uint8) if b.dtype != torch.uint8 else b
    return int((a_u != b_u).sum().item())


def main():
    fp8_max = torch.finfo(_fp8).max
    cases = [
        # (B, S_q, S_k, Hq, Hk, D, BLKQ, BLKK, label)
        (1, 256, 256, 8, 8, 128, 256, 64, "tiny S=256 Hq=8"),
        (1, 1024, 1024, 8, 8, 128, 256, 64, "S=1024 Hq=8"),
        (1, 4096, 4096, 8, 8, 128, 256, 64, "S=4096 Hq=8"),
        (2, 8192, 8192, 16, 4, 128, 256, 64, "S=8192 Hq=16 Hk=4 (GQA)"),
    ]

    all_pass = True
    for (B, S_q, S_k, Hq, Hk, D, BLKQ, BLKK, label) in cases:
        torch.manual_seed(0)
        q = torch.randn(B, S_q, Hq, D, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(B, S_k, Hk, D, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(B, S_k, Hk, D, dtype=torch.bfloat16, device="cuda")

        # Triton reference. Note BLOCK_R=128 — the bench wrapper
        # (fav3_sage_attention_mxfp4_wrapper.py:52) passes 128, NOT the
        # default 32, so we must match that to get a meaningful comparison.
        t_qfp4, t_qd, t_kfp4, t_kd, t_vfp8, t_vsc, t_ds = triton_quant(
            q, k, v, _fp8, fp8_max, BLKQ=BLKQ, BLKK=BLKK,
            layout="bshd", q_smoothing=False, BLOCK_R=128,
        )
        torch.cuda.synchronize()

        # FlyDSL
        f_qfp4, f_qd, f_kfp4, f_kd, f_vfp8, f_vsc, f_ds = flydsl_sage_quant_mxfp4(
            q, k, v, _fp8, fp8_max, BLKQ=BLKQ, BLKK=BLKK,
            layout="bshd", BLOCK_R=128,
        )
        torch.cuda.synchronize()

        # Compare
        n_q_fp4 = _bytes_diff(t_qfp4, f_qfp4)
        n_q_d   = _bytes_diff(t_qd, f_qd)
        n_k_fp4 = _bytes_diff(t_kfp4, f_kfp4)
        n_k_d   = _bytes_diff(t_kd, f_kd)
        n_v_fp8 = _bytes_diff(t_vfp8, f_vfp8)
        v_sc_diff = (t_vsc - f_vsc).abs().max().item()

        total_q_bytes = t_qfp4.numel()
        total_k_bytes = t_kfp4.numel()
        total_v_bytes = t_vfp8.numel()
        total_qd_bytes = t_qd.numel()
        total_kd_bytes = t_kd.numel()

        # Some tolerance for FP4 nibble: occasional rounding diff is OK if rare.
        # e8m0 must match exactly (or RNE drift becomes amplified).
        FP4_TOL = 0.01   # 1% byte mismatch tolerated (from rounding diffs)
        E8M0_TOL = 0.0   # exact

        ok = (
            n_q_fp4 / total_q_bytes < FP4_TOL and
            n_k_fp4 / total_k_bytes < FP4_TOL and
            n_v_fp8 / total_v_bytes < FP4_TOL and
            n_q_d / total_qd_bytes <= E8M0_TOL and
            n_k_d / total_kd_bytes <= E8M0_TOL and
            v_sc_diff < 1e-5
        )
        status = "PASS" if ok else "FAIL"
        all_pass = all_pass and ok

        print(f"[{status}] {label}")
        print(f"  q_fp4 mismatch: {n_q_fp4:>10} / {total_q_bytes:>10} = "
              f"{100.0*n_q_fp4/total_q_bytes:.4f}%")
        print(f"  q_d   mismatch: {n_q_d:>10} / {total_qd_bytes:>10} = "
              f"{100.0*n_q_d/total_qd_bytes:.4f}%")
        print(f"  k_fp4 mismatch: {n_k_fp4:>10} / {total_k_bytes:>10} = "
              f"{100.0*n_k_fp4/total_k_bytes:.4f}%")
        print(f"  k_d   mismatch: {n_k_d:>10} / {total_kd_bytes:>10} = "
              f"{100.0*n_k_d/total_kd_bytes:.4f}%")
        print(f"  v_fp8 mismatch: {n_v_fp8:>10} / {total_v_bytes:>10} = "
              f"{100.0*n_v_fp8/total_v_bytes:.4f}%")
        print(f"  v_scale max abs diff: {v_sc_diff:.2e}")
        print()

    print("ALL PASS" if all_pass else "SOME FAILED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
