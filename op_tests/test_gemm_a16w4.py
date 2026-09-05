# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Dense a16w4 GEMM benchmarks on gfx1201 (Radeon AI PRO R9700).

TWO SWEEPS, TWO TABLES
======================
The op has two tiles with opposite characters, so they get separate sweeps
rather than one table that averages them into nonsense:

    prefill   M >= 512   compute bound, BM128 BN512 BK64, no split-K
    decode    M <=  64   memory bound,  BM16 BN128 BK64, grid split-K=2

M in [65, 511] is DELIBERATELY NOT SWEPT. Neither tile covers it: prefill's
grid is (M/128, N/512), so below M=512 it runs too few workgroups to fill the
part, and decode's cost grows linearly in M because BM=16. The C++ dispatch
splits the range at M=256 (kPrefillMinM) as damage control, but both halves
are off their tuned regime -- at M=128 the vLLM Triton W4A16 kernel is still
1.25x faster. A third tile is the fix; until it lands, benchmarking that range
would only publish a number nobody has optimised.

WHAT THE CANDIDATES MEAN
========================
    a16w4                  the op under test
    vllm_triton_w4a16      THE COMPARISON THAT MATTERS -- vLLM's Triton W4A16
                           kernel, the thing gemm_a16w4 replaces in
                           production. Vendored verbatim (see the attribution
                           block below); same nibbles, same symmetric zero
                           point, same output dtype, so the only differences
                           are the weight packing and the kernel itself.
    torch_a16w16_ceiling   F.linear on the SAME weights, dequantized to 16-bit
                           beforehand -> hipBLASLt. A CEILING, NOT A
                           COMPETITOR: it reads 4x the weight bytes and does
                           no dequant at all, so at large M -- where the GEMM
                           is compute bound -- matching it is the best a
                           weight-only kernel can do, since it is doing
                           strictly more work for the same FLOPs. Kept only to
                           bound the remaining headroom.
    triton_a16w16          aiter's Triton dense GEMM, when a tuned config for
                           the running arch exists (there is none for gfx1201
                           today, so it is normally skipped)

`TB/s` is reported per candidate against its own byte count: 16-bit weights
for the ceiling, 4-bit + scales for Triton (HAS_ZP=False folds the zero point
into a constant), 4-bit + scales + zeros for a16w4.

The whole file is gated on gfx1201 and exits early elsewhere: the op is
RDNA4-only, and a table of just the baselines on another card would say
nothing about it.

QUANTIZATION MODEL (identical for every candidate and the reference)
====================================================================
    nibbles [K, N]        uint8, values 0..15
    scales  [K/G, N]      fp16/bf16
    zeros   [K/G, N]      fp16/bf16, constant ZP=8 (symmetric), matching what
                          an AWQ checkpoint with a symmetric zero point ships
    W_deq[k, n] = (nibbles[k, n] - zeros[k//G, n]) * scales[k//G, n]

WEIGHT PREPARATION
==================
Goes through aiter.prepare_a16w4_weight(), the same call a serving stack makes
in process_weights_after_loading() -- packing [K, N] nibbles into [K/8, N]
int32 along K, plus the MAGIC nibble permutation and +1024 zero bias on the
fp16 path. It runs once per shape here, outside the timed region, exactly as
it would in production.

RUN
===
    python3 op_tests/test_gemm_a16w4.py                    # both tables
    python3 op_tests/test_gemm_a16w4.py -d fp16 bf16
    python3 op_tests/test_gemm_a16w4.py -s 1,5120,5120     # decode table only
"""

import argparse
import functools

import pandas as pd
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.gemm_op_a16w4 import (
    GROUP_SIZE,
    PACK_K,
    SUPPORTED_GFX,
    gemm_a16w4,
    gemm_a16w4_workspace_elems,
    prepare_a16w4_weight,
)
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

torch.set_default_device("cuda")

# Symmetric zero point, the common AWQ case. The vLLM Triton kernel below
# folds exactly this value in as a compile-time constant when HAS_ZP=False,
# so both candidates compute bit-for-bit the same C = A @ ((W - 8) * scale).
ZP = 8

# Sweep bounds. These are NOT the C++ dispatch threshold (kPrefillMinM = 256 in
# csrc/gemm_a16w4/gemm_a16w4.cu) -- they are deliberately narrower, so that
# each table only contains shapes the corresponding tile was actually tuned
# for. M in [65, 511] is skipped: prefill runs too few workgroups there
# (its grid is (M/128, N/512)) and decode's cost grows linearly in M because
# BM=16, so neither number would say anything useful until a third tile lands.
PREFILL_MIN_M = 512
DECODE_MAX_M = 64


# ──────────────────────────────────────────────────────────────────────
# Reference baseline: vLLM's Triton W4A16 kernel
# ──────────────────────────────────────────────────────────────────────
#
# THIRD-PARTY CODE. Reproduced verbatim from vLLM 0.21.0,
#   vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py
# which is Apache-2.0. It is vendored rather than imported so the benchmark
# does not require vLLM to be installed, and it is NOT part of the aiter
# library -- it lives here only so `gemm_a16w4` is measured against the
# kernel it actually replaces in production rather than against a
# dequantized-fp16 upper bound.
#
# Kept byte-identical on purpose: the point is to measure what a user gets
# today, so it is not autotuned, not re-tiled, and not given num_warps /
# num_stages overrides that vLLM does not pass either.


@triton.jit
def _vllm_triton_w4a16_kernel(
    a_ptr, b_ptr, scales_ptr, zeros_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    group_size,
    HAS_ZP: tl.constexpr,
    ZP_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_bn = pid_n * (BLOCK_N // 8) + tl.arange(0, BLOCK_N // 8)

    shifts_row = tl.arange(0, 8) * 4
    shifts_1d_2d = tl.broadcast_to(shifts_row[None, :], (BLOCK_N // 8, 8))
    shifts_1d = tl.reshape(shifts_1d_2d, (BLOCK_N,))
    shifts = tl.broadcast_to(shifts_1d[None, :], (BLOCK_K, BLOCK_N))

    offs_sn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        mask_a = (offs_m[:, None] < M) & mask_k[None, :]
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
        mask_b = mask_k[:, None] & (offs_bn[None, :] < N // 8)
        b_packed = tl.load(b_ptrs, mask=mask_b, other=0)

        b = tl.interleave(b_packed, b_packed)
        b = tl.interleave(b, b)
        b = tl.interleave(b, b)
        b = (b >> shifts) & 0xF

        g_idx = (k_start * BLOCK_K) // group_size

        scale_offset = g_idx * N + offs_sn
        scale_mask = offs_sn < N
        scales = tl.load(scales_ptr + scale_offset, mask=scale_mask, other=1.0)
        scales = tl.broadcast_to(scales[None, :], (BLOCK_K, BLOCK_N))

        if HAS_ZP:
            zero_offset = g_idx * (N // 8) + offs_bn
            zero_mask = offs_bn < N // 8
            z_packed = tl.load(zeros_ptr + zero_offset, mask=zero_mask, other=0)
            z = tl.interleave(z_packed, z_packed)
            z = tl.interleave(z, z)
            z = tl.interleave(z, z)
            z = (z >> shifts_1d) & 0xF
            z = tl.broadcast_to(z[None, :], (BLOCK_K, BLOCK_N))
        else:
            z = tl.full((BLOCK_K, BLOCK_N), ZP_BIAS, dtype=tl.int32)

        b_fp = (b - z).to(a.dtype) * scales
        accumulator += tl.dot(a, b_fp, out_dtype=tl.float32)

    c = accumulator.to(c_ptr.type.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask_c)


def vllm_triton_config(m, group_size=GROUP_SIZE):
    """vLLM's production tile table, verbatim.

    gfx1201 takes the _ON_GFX1X branch because vllm.platforms.rocm sets
        _ON_GFX1X = any(arch in _GCN_ARCH for arch in ["gfx11", "gfx12"])
    so RDNA4 inherits a table whose own comment says it was tuned for gfx1151.
    There is no RDNA4 entry. That is what production runs today, so that is
    what is measured -- these numbers are a statement about vLLM's current
    defaults on this card, not about Triton as a compiler.
    """
    if m <= 32:
        block_m, block_n, block_k = 32, 32, 64
    elif m <= 64:
        block_m, block_n, block_k = 64, 64, 32
    else:
        block_m, block_n, block_k = 128, 32, 64
    if group_size < block_k:
        block_k = group_size
    return block_m, block_n, block_k


def pack_int4_along_n(nibbles):
    """[K, N] uint8 nibbles -> [K, N/8] int32, nibble n%8 at bit 4*(n%8).

    The AWQ/GPTQ checkpoint layout, which is what the Triton kernel consumes
    directly. `prepare_a16w4_weight()` produces the other packing (along K)
    for gemm_a16w4; both are derived from the same nibbles here, so the two
    candidates are fed identical numbers.

    Worth noting for integration: gemm_a16w4 cannot read the checkpoint
    layout natively and Triton can, so adopting it costs one repack in
    process_weights_after_loading(). That is a one-off, not per token, and is
    outside the timed region here.
    """
    k, n = nibbles.shape
    v = nibbles.to(torch.int64).view(k, n // PACK_K, PACK_K)
    shifts = (torch.arange(PACK_K, device=nibbles.device, dtype=torch.int64) * 4).view(
        1, 1, PACK_K
    )
    return (v << shifts).sum(dim=-1).to(torch.int32)


def generate_gemm_a16w4_inputs(m, n, k, gsize, dtype):
    """Build one int4 weight set plus every layout the candidates need.

    Magnitudes are chosen so the fp32 reference and the 16-bit kernels agree
    to ~1e-2 over K up to 16k; plain randn weights would make the comparison
    dominated by fp16 accumulation error rather than by kernel bugs.
    """
    torch.manual_seed(0)
    x = (torch.randn((m, k), dtype=dtypes.fp32) * 0.1).to(dtype)

    nibbles = torch.randint(0, 16, (k, n), dtype=torch.uint8)
    scales = (torch.rand((k // gsize, n), dtype=dtypes.fp32) * 0.01 + 0.005).to(dtype)
    zeros = torch.full((k // gsize, n), float(ZP), dtype=dtype)

    # [K, N] fp32 dequantized weights, then [N, K] for the 16-bit GEMM APIs.
    w_deq = (
        nibbles.to(dtypes.fp32) - zeros.to(dtypes.fp32).repeat_interleave(gsize, dim=0)
    ) * scales.to(dtypes.fp32).repeat_interleave(gsize, dim=0)
    w_nk = w_deq.T.contiguous().to(dtype)

    # Two packings of the SAME nibbles: [K, N/8] along N is the checkpoint
    # layout the Triton kernel reads; prepare_a16w4_weight() gives [K/8, N]
    # along K (plus the MAGIC permutation and +1024 zero bias on fp16).
    # Both are load-time work, timed nowhere.
    b_tri = pack_int4_along_n(nibbles)
    w_q, s_q, z_q = prepare_a16w4_weight(nibbles, scales, zeros, dtype)

    return x, b_tri, w_q, s_q, z_q, w_deq, w_nk


def run_torch(x, w_deq, dtype):
    """Reference only: fp32 dequant + matmul. Not timed, not in the table."""
    return (x.to(dtypes.fp32) @ w_deq).to(dtype)


@functools.lru_cache(maxsize=None)
def triton_a16w16_available(m, n, k):
    """True when aiter ships a tuned GEMM-A16W16 config for this arch.

    As of writing there is none for gfx1201 -- configs/gfx1201/triton/gemm/
    has a8w8 and conv families only -- so get_gemm_config() raises and the
    Triton baseline is simply unavailable on this card. Probing through the
    resolver rather than stat()ing a hand-built path keeps this working if the
    family migrates between the legacy and nested config layouts.
    """
    try:
        get_gemm_config("GEMM-A16W16", m, n, k)
        return True
    except AssertionError:
        return False


# ──────────────────────────────────────────────────────────────────────
# Benchmarks
# ──────────────────────────────────────────────────────────────────────


def run_candidates(m, n, k, gsize, dtype):
    """Time every candidate for one shape and return the table row.

    Shared by both sweeps: prefill and decode differ only in which shapes
    reach them, not in how a shape is measured.
    """
    x, b_tri, w_q, s_q, z_q, w_deq, w_nk = generate_gemm_a16w4_inputs(
        m, n, k, gsize, dtype
    )
    ref = run_torch(x, w_deq, dtype)

    esize = x.element_size()
    flops = 2 * m * n * k
    # Per-candidate byte counts. The 16-bit path reads N*K 2-byte weights; the
    # two 4-bit paths read N*K/2 bytes plus per-group metadata -- Triton takes
    # scales only (HAS_ZP=False folds the zero point into a constant), a16w4
    # takes scales and zeros.
    bytes_a16w16 = (m * k + n * k + m * n) * esize
    bytes_meta = (k // gsize) * n * esize
    bytes_triton = (m * k + m * n) * esize + n * k // 2 + bytes_meta
    bytes_a16w4 = (m * k + m * n) * esize + n * k // 2 + 2 * bytes_meta

    # Preallocated outputs, as a linear layer would pass them. One buffer per
    # candidate so a kernel is never timed writing into another's cache lines.
    y = torch.empty((m, n), dtype=dtype)
    y_q = torch.empty((m, n), dtype=dtype)
    y_tri = torch.empty((m, n), dtype=dtype)

    block_m, block_n, block_k = vllm_triton_config(m)
    tri_grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    # Triton reads scales in the activation dtype, which is what
    # prepare_a16w4_weight() returns; on the fp16 path it only casts them,
    # the MAGIC permutation and +1024 bias touch the weights and zeros only.
    scales_arg = s_q

    def run_vllm_triton():
        _vllm_triton_w4a16_kernel[tri_grid](
            x, b_tri, scales_arg, b_tri, y_tri, m, n, k,
            x.stride(0), x.stride(1), b_tri.stride(0), b_tri.stride(1),
            y_tri.stride(0), y_tri.stride(1),
            gsize, HAS_ZP=False, ZP_BIAS=ZP,
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k,
        )
        # run_perftest hands the return value to checkAllclose, and a raw
        # Triton launch returns None.
        return y_tri

    candidates = {
        # THE COMPARISON THAT MATTERS: the kernel gemm_a16w4 replaces in
        # production. Same nibbles, same symmetric zero point, same output
        # dtype -- only the weight packing and the kernel differ.
        "vllm_triton_w4a16": (run_vllm_triton, bytes_triton),
        # CEILING, NOT A COMPETITOR: same weights, dequantized to 16-bit
        # beforehand, so it reads 4x the weight bytes and does no dequant.
        # Kept because it bounds how much of the gap to dense fp16 is left,
        # but it is not what a16w4 is competing against.
        "torch_a16w16_ceiling": (lambda: F.linear(x, w_nk), bytes_a16w16),
    }
    if triton_a16w16_available(m, n, k):
        candidates["triton_a16w16"] = (
            lambda: gemm_a16w16(x, w_nk, None, dtype, y),
            bytes_a16w16,
        )
    else:
        aiter.logger.info(
            "triton_a16w16 skipped: no tuned GEMM-A16W16 config for %s", get_gfx()
        )

    reason = aiter.gemm_a16w4_unsupported_reason(m, n, k, dtype == dtypes.fp16)
    if reason:
        # Not a failure: prefill wants M % 128 == 0 and N % 512 == 0, decode
        # wants K % 256 == 0, so some shapes land outside both tiles. Leaving
        # the cells nan is more honest than silently reshaping the case.
        aiter.logger.info("a16w4 skipped for m=%d n=%d k=%d: %s", m, n, k, reason)
    else:
        # BOTH buffers are hoisted, which is what a serving stack does and what
        # the numbers depend on: at M=1 the kernel is ~24 us, so allocating a
        # fresh output and split-K scratch per call adds ~45 us and would show
        # a 1.4x win where the kernel actually delivers 3.2x.
        ws = torch.empty(
            gemm_a16w4_workspace_elems(m, n, k, dtype == dtypes.fp16),
            dtype=dtypes.fp32,
        )
        candidates["a16w4"] = (
            lambda: gemm_a16w4(x, w_q, s_q, z_q, out=y_q, workspace=ws),
            bytes_a16w4,
        )

    ret = {"gfx": get_gfx()}
    for name, (fn, nbytes) in candidates.items():
        # use_cuda_event=True, not the default torch-profiler path. On gfx1201
        # (ROCm 7.2.2) the profiler records no CUDA device events -- roctracer
        # logs "produced duplicate flow start" and every kernel is attributed
        # to the CPU -- so get_trace_perf() drops every row and reports
        # "no valida data after post process!" with us=0. That happens for a
        # bare torch.matmul too, so it is the harness on this arch rather than
        # any candidate here. cuda.Event timing does not depend on roctracer.
        out, us = run_perftest(fn, use_cuda_event=True)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: a16w4 gemm m={m} n={n} k={k}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_gemm_a16w4_prefill(m, n, k, gsize, dtype):
    """Compute-bound sweep, M >= 512. Takes the BM128 BN512 BK64 tile."""
    return run_candidates(m, n, k, gsize, dtype)


@benchmark()
def test_gemm_a16w4_decode(m, n, k, gsize, dtype):
    """Memory-bound sweep, M <= 64. Takes the BM16 BN128 BK64 split-K tile."""
    return run_candidates(m, n, k, gsize, dtype)


def summarize(name, rows, dtype):
    if not rows:
        return
    aiter.logger.info(
        "a16w4 %s summary (dtype=%s):\n%s",
        name,
        dtype,
        pd.DataFrame(rows).to_markdown(index=False),
    )


def main():
    # Whole-op arch gate lives here, not inside the @benchmark fns: @benchmark
    # always returns the call-args dict, so an in-fn return would still emit a
    # meaningless args-only row.
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "a16w4 gemm is %s only; skipping on %s",
            "/".join(SUPPORTED_GFX),
            get_gfx(),
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["fp16"], dtypes.d_dtypes["bf16"]],
        nargs="*",
        # A string default is passed through `type`, a list default is not.
        default="fp16,",
        metavar="{fp16,bf16}",
        help="""Data type of activations, scales and output.
        e.g.: -d fp16 bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            # Qwen3-32B linear layers as deployed (hidden 5120, intermediate
            # 25600). Every (N, K) here satisfies both tiles: prefill needs
            # N % 512 and K % 128, decode needs N % 128 and K % 256.
            #   5120 x  4096   o_proj
            #   5120 x  5120   qkv / o_proj
            #   5120 x 12800   down_proj
            #  25600 x  5120   gate / up_proj
            # Each M is routed to the prefill or decode table by the bounds
            # above; nothing in [65, 511] is listed, because no tile is tuned
            # for it yet.
            *[(m, 5120, 4096) for m in (1, 8, 32, 64, 512, 1024, 4096)],
            *[(m, 5120, 5120) for m in (1, 8, 32, 64, 512, 1024, 4096)],
            *[(m, 5120, 12800) for m in (1, 8, 32, 64, 512, 1024, 4096)],
            *[(m, 25600, 5120) for m in (1, 8, 32, 64, 512, 1024, 4096)],
        ],
        help="""Shape of mnk. Routed to the prefill table when m >= 512 and to
        the decode table when m <= 64; anything between is skipped.
        e.g.:   -s 1,5120,5120
                --mnk 4096,25600,5120""",
    )
    args = parser.parse_args()

    # GROUP_SIZE is fixed by the kernel, not a sweep axis: it is baked into the
    # LDS layout and the K-step, so there is no -g flag to vary it.
    for dtype in args.dtype:
        prefill_rows, decode_rows = [], []
        for m, n, k in args.mnk:
            if m >= PREFILL_MIN_M:
                prefill_rows.append(
                    test_gemm_a16w4_prefill(m, n, k, GROUP_SIZE, dtype)
                )
            elif m <= DECODE_MAX_M:
                decode_rows.append(test_gemm_a16w4_decode(m, n, k, GROUP_SIZE, dtype))
            else:
                aiter.logger.warning(
                    "m=%d skipped: [%d, %d] has no tuned tile yet, so a number "
                    "there would not mean anything",
                    m,
                    DECODE_MAX_M + 1,
                    PREFILL_MIN_M - 1,
                )
        summarize("prefill", prefill_rows, dtype)
        summarize("decode", decode_rows, dtype)


if __name__ == "__main__":
    main()
