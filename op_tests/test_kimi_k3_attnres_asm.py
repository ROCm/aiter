# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and performance tests for Kimi-K3 AttnResidual ASM kernels.

Tests both score and combine against their Triton reference implementations
at the Kimi-K3 CONC=64 decode shape (T=64, NVB=8, H=7168).

Run:
    python op_tests/test_kimi_k3_attnres_asm.py

Skip condition: test is a no-op on non-gfx950 (MI300X and earlier).
"""

import math
import unittest

import torch
import triton
import triton.language as tl

from aiter.test_common import perftest

# ---------------------------------------------------------------------------
# Shape constants (Kimi-K3 CONC=64 decode)
# ---------------------------------------------------------------------------
T = 64
NVB = 8
H = 7168
BLOCK_H = 1024
MAX_ROWS = 16
EPS = 1e-6


# ---------------------------------------------------------------------------
# Triton reference kernels (from sglang AttnResidual)
# ---------------------------------------------------------------------------


@triton.jit
def _score_kernel_ref(
    prefix_ptr,
    bank_ptr,
    cw_ptr,
    scores_ptr,
    NVB,
    EPS: tl.constexpr,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_j = tl.program_id(1)

    offs_h = tl.arange(0, BLOCK_H)
    sumsq = tl.zeros([BLOCK_H], tl.float32)
    dotv = tl.zeros([BLOCK_H], tl.float32)

    for bh in range(0, tl.cdiv(H, BLOCK_H)):
        h0 = bh * BLOCK_H
        if pid_j < NVB:
            v = tl.load(bank_ptr + pid_t * stride_bm + pid_j * stride_bb + h0 + offs_h)
        else:
            v = tl.load(prefix_ptr + pid_t * stride_pm + h0 + offs_h)
        vf = v.to(tl.float32)
        cw = tl.load(cw_ptr + h0 + offs_h)
        sumsq += vf * vf
        dotv += vf * cw

    ss = tl.sum(sumsq, 0)
    dv = tl.sum(dotv, 0)
    rrms = tl.rsqrt(ss / H + EPS)
    score = dv * rrms
    tl.store(scores_ptr + pid_t * stride_sm + pid_j, score)


@triton.jit
def _combine_kernel_ref(
    prefix_ptr,
    bank_ptr,
    scores_ptr,
    out_ptr,
    NVB,
    stride_pm,
    stride_bm,
    stride_bb,
    stride_sm,
    stride_om,
    BLOCK_H: tl.constexpr,
    MAX_ROWS: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    h0 = pid_h * BLOCK_H
    offs_b = tl.arange(0, MAX_ROWS)
    mask_b = offs_b <= NVB
    raw = tl.load(
        scores_ptr + pid_t * stride_sm + offs_b, mask=mask_b, other=float("-inf")
    )
    m = tl.max(raw, axis=0)
    e = tl.where(mask_b, tl.exp(raw - m), 0.0)
    p = e / tl.sum(e, axis=0)
    offs_h = h0 + tl.arange(0, BLOCK_H)
    acc = tl.zeros([BLOCK_H], tl.float32)
    for j in range(0, NVB + 1):
        if j < NVB:
            v = tl.load(bank_ptr + pid_t * stride_bm + j * stride_bb + offs_h).to(
                tl.float32
            )
        else:
            v = tl.load(prefix_ptr + pid_t * stride_pm + offs_h).to(tl.float32)
        p_j = tl.sum(tl.where(offs_b == j, p, 0.0), axis=0)
        acc += p_j * v
    tl.store(out_ptr + pid_t * stride_om + offs_h, acc.to(out_ptr.dtype.element_ty))


def _run_score_ref(prefix, bank, cw, scores):
    scores.zero_()
    _score_kernel_ref[(T, NVB + 1)](
        prefix,
        bank,
        cw,
        scores,
        NVB,
        EPS,
        prefix.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        H=H,
        BLOCK_H=BLOCK_H,
    )
    torch.cuda.synchronize()


def _run_combine_ref(prefix, bank, scores, out):
    out.zero_()
    n_h = H // BLOCK_H
    _combine_kernel_ref[(T, n_h)](
        prefix,
        bank,
        scores,
        out,
        NVB,
        prefix.stride(0),
        bank.stride(0),
        bank.stride(1),
        scores.stride(0),
        out.stride(0),
        BLOCK_H=BLOCK_H,
        MAX_ROWS=MAX_ROWS,
    )
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_inputs(seed=42):
    torch.manual_seed(seed)
    device = "cuda"
    prefix = torch.randn(T, H, dtype=torch.bfloat16, device=device) * 0.1
    bank = torch.randn(T, NVB, H, dtype=torch.bfloat16, device=device) * 0.1
    cw = torch.randn(H, dtype=torch.float32, device=device) * 0.1
    scores = torch.zeros(T, MAX_ROWS, dtype=torch.float32, device=device)
    out = torch.zeros(T, H, dtype=torch.bfloat16, device=device)
    return prefix, bank, cw, scores, out


def _snr_db(ref, cand):
    diff = (ref.float() - cand.float()).norm()
    sig = ref.float().norm()
    if sig == 0:
        return float("inf")
    return 20 * math.log10((sig / (diff + 1e-12)).item())


# ---------------------------------------------------------------------------
# @perftest wrappers
# ---------------------------------------------------------------------------


@perftest()
def _bench_score_ref(prefix, bank, cw, scores):
    _run_score_ref(prefix, bank, cw, scores)
    return scores


@perftest()
def _bench_score_asm(prefix, bank, cw, scores):
    from aiter.ops.flydsl.kernels.kimi_k3_attnres import attnres_score_asm

    attnres_score_asm(prefix, bank, cw, scores)
    return scores


@perftest()
def _bench_combine_ref(prefix, bank, scores, out):
    _run_combine_ref(prefix, bank, scores, out)
    return out


@perftest()
def _bench_combine_asm(prefix, bank, scores, out):
    from aiter.ops.flydsl.kernels.kimi_k3_attnres import attnres_combine_asm

    attnres_combine_asm(prefix, bank, scores, out)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def _skip_if_not_supported():
    from aiter.ops.flydsl.kernels.kimi_k3_attnres import (
        is_kimi_k3_attnres_asm_supported,
    )

    return not is_kimi_k3_attnres_asm_supported()


@unittest.skipIf(not torch.cuda.is_available(), "no GPU")
@unittest.skipIf(_skip_if_not_supported(), "gfx950 + .co files required")
class TestAttnresScore(unittest.TestCase):
    def setUp(self):
        self.prefix, self.bank, self.cw, self.scores, self.out = _make_inputs()

    def test_correctness(self):
        from aiter.ops.flydsl.kernels.kimi_k3_attnres import attnres_score_asm

        _run_score_ref(self.prefix, self.bank, self.cw, self.scores)
        ref = self.scores.clone()

        self.scores.zero_()
        attnres_score_asm(self.prefix, self.bank, self.cw, self.scores)

        snr = _snr_db(ref, self.scores)
        self.assertGreater(snr, 35.0, f"score SNR too low: {snr:.1f} dB")

    def test_deterministic(self):
        from aiter.ops.flydsl.kernels.kimi_k3_attnres import attnres_score_asm

        attnres_score_asm(self.prefix, self.bank, self.cw, self.scores)
        r1 = self.scores.clone()
        attnres_score_asm(self.prefix, self.bank, self.cw, self.scores)
        r2 = self.scores.clone()
        self.assertTrue(torch.equal(r1, r2), "score kernel not deterministic")


@unittest.skipIf(not torch.cuda.is_available(), "no GPU")
@unittest.skipIf(_skip_if_not_supported(), "gfx950 + .co files required")
class TestAttnresCombine(unittest.TestCase):
    def setUp(self):
        self.prefix, self.bank, self.cw, self.scores, self.out = _make_inputs()
        # Populate scores with ref values
        _run_score_ref(self.prefix, self.bank, self.cw, self.scores)

    def test_correctness(self):
        from aiter.ops.flydsl.kernels.kimi_k3_attnres import attnres_combine_asm

        _run_combine_ref(self.prefix, self.bank, self.scores, self.out)
        ref = self.out.clone()

        self.out.zero_()
        attnres_combine_asm(self.prefix, self.bank, self.scores, self.out)

        snr = _snr_db(ref, self.out)
        self.assertGreater(snr, 40.0, f"combine SNR too low: {snr:.1f} dB")

    def test_deterministic(self):
        from aiter.ops.flydsl.kernels.kimi_k3_attnres import attnres_combine_asm

        attnres_combine_asm(self.prefix, self.bank, self.scores, self.out)
        r1 = self.out.clone()
        attnres_combine_asm(self.prefix, self.bank, self.scores, self.out)
        r2 = self.out.clone()
        self.assertTrue(torch.equal(r1, r2), "combine kernel not deterministic")


# ---------------------------------------------------------------------------
# Performance summary (run standalone)
# ---------------------------------------------------------------------------


def _run_perf():
    from aiter.ops.flydsl.kernels.kimi_k3_attnres import (
        is_kimi_k3_attnres_asm_supported,
    )

    if not is_kimi_k3_attnres_asm_supported():
        print("SKIP: gfx950 not detected or .co files missing")
        return

    prefix, bank, cw, scores, out = _make_inputs()
    _run_score_ref(prefix, bank, cw, scores)  # warm scores for combine

    _, ref_score_us = _bench_score_ref(prefix, bank, cw, scores)
    _, asm_score_us = _bench_score_asm(prefix, bank, cw, scores)
    _, ref_comb_us = _bench_combine_ref(prefix, bank, scores, out)
    _, asm_comb_us = _bench_combine_asm(prefix, bank, scores, out)

    score_speedup = (ref_score_us - asm_score_us) / ref_score_us * 100
    comb_speedup = (ref_comb_us - asm_comb_us) / ref_comb_us * 100

    header = f"{'Kernel':<20} {'Ref (us)':>10} {'ASM (us)':>10} {'Speedup':>10}"
    sep = "-" * len(header)
    print(f"\nAttnResidual ASM vs Triton  [gfx950, T={T} NVB={NVB} H={H}]")
    print(sep)
    print(header)
    print(sep)
    print(
        f"{'score':<20} {ref_score_us:>10.2f} {asm_score_us:>10.2f} {score_speedup:>+9.1f}%"
    )
    print(
        f"{'combine':<20} {ref_comb_us:>10.2f} {asm_comb_us:>10.2f} {comb_speedup:>+9.1f}%"
    )
    print(sep)


if __name__ == "__main__":
    _run_perf()
    unittest.main(exit=False)
