# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Test / verification harness for the dedicated gfx1250 MXFP8 ASM FMHA forward
# path (aiter.fmha_fwd_mxfp8_asm).
#
# Usage:
#   Correctness:  python3 op_tests/test_fmha_fwd_mxfp8_asm.py
#   Performance:  python3 op_tests/test_fmha_fwd_mxfp8_asm.py --perf
#
# Default config matches poc_kl/mi400/fmha_fwd_mxfp8/new_run.sh `test`:
#   batch=1, kv_head_num=1, gqa=1, seq_len=512, head_dim=128, mask=0.

import argparse

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.test_common import run_perftest
from aiter.test_mha_common import attention_ref

BLOCK_SIZE = 32
SUB_Q = 256
SUB_K = 128


def _is_gfx1250_host() -> bool:
    """True only on a gfx1250 GPU host.

    The MXFP8 ASM kernel is only shipped in hsa/gfx1250/fmha_fwd_mxfp8/*.co —
    there are no gfx942 / gfx950 / etc. binaries.  On any other arch the call
    raises 'no kernel for arch=...' at launch, so this guard skips (not fails)
    the tests on non-gfx1250 hosts.  Short-circuits on no-GPU and swallows any
    probe error.
    """
    if not torch.cuda.is_available():
        return False
    try:
        return get_gfx() == "gfx1250"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _is_gfx1250_host(),
    reason=(
        "fmha_fwd_mxfp8_asm ASM kernel is only shipped for gfx1250 "
        "(hsa/gfx1250/fmha_fwd_mxfp8/*.co); no GPU or a different arch — skip"
    ),
)


def align_to_tile(original, tile_size):
    return (original + tile_size - 1) // tile_size * tile_size


def create_mxfp8_scale_buffer(
    batch,
    head_num,
    seq_len,
    head_dim,
    block_size,
    sub_tile,
    device="cuda",
    fill_value=1.0,
    extra_tiles=0,
):
    """MXFP8 block-scale buffer (float8_e8m0fnu).

    Layout mirrors the poc host (fmha_fwd_mxfp8.cpp): the buffer is flat with
    `align(batch*seq_len, sub_tile) * head_dim * head_num / block_size` bytes.
    fill_value=1.0 maps to E8M0 byte 0x7F (2^0, i.e. no scaling).

    `extra_tiles` pads the (already tile-aligned) seq dimension by N additional
    `sub_tile`-sized tiles.  This is a workaround for a known bug in the current
    K-scale kernel, which over-reads the K block-scale buffer by 2 tiles; pass
    extra_tiles=2 for k_scale to keep that read in-bounds.
    """
    total_seq = batch * seq_len
    aligned_seq = align_to_tile(total_seq, sub_tile) + extra_tiles * sub_tile
    num = aligned_seq * head_dim * head_num // block_size
    return torch.full((num,), fill_value, dtype=torch.float8_e8m0fnu, device=device)


def make_inputs(batch, nheads, nheads_k, seqlen_q, seqlen_k, d):
    """Build fp8 q/k/v as BSHD-shaped views over BHSD memory + e8m0 scales."""
    torch.random.manual_seed(0)
    d_v = d

    # BHSD memory (batch, head, seq, dim); transpose(1,2) gives a BSHD-shaped
    # view whose strides satisfy stride_head > stride_seq (bhsd memory order).
    q_bhsd = torch.randn(
        batch, nheads, seqlen_q, d, device="cuda", dtype=torch.bfloat16
    )
    k_bhsd = torch.randn(
        batch, nheads_k, seqlen_k, d, device="cuda", dtype=torch.bfloat16
    )
    v_bhsd = torch.randn(
        batch, nheads_k, seqlen_k, d_v, device="cuda", dtype=torch.bfloat16
    )

    q_fp8 = q_bhsd.to(dtypes.fp8)
    k_fp8 = k_bhsd.to(dtypes.fp8)
    v_fp8 = v_bhsd.to(dtypes.fp8)

    q_in = q_fp8.transpose(1, 2)
    k_in = k_fp8.transpose(1, 2)
    v_in = v_fp8.transpose(1, 2)

    q_scale = create_mxfp8_scale_buffer(batch, nheads, seqlen_q, d, BLOCK_SIZE, SUB_Q)
    # K-scale kernel bug workaround: over-reads by 2 tiles -> pad with 2 tiles.
    k_scale = create_mxfp8_scale_buffer(
        batch, nheads_k, seqlen_k, d, BLOCK_SIZE, SUB_K, extra_tiles=2
    )
    v_scale = create_mxfp8_scale_buffer(
        batch, nheads_k, seqlen_k, d_v, BLOCK_SIZE, SUB_K
    )

    return q_in, k_in, v_in, q_scale, k_scale, v_scale, q_fp8, k_fp8, v_fp8


def run_one(batch, nheads, nheads_k, seqlen_q, seqlen_k, d, check=True):
    (
        q_in,
        k_in,
        v_in,
        q_scale,
        k_scale,
        v_scale,
        q_fp8,
        k_fp8,
        v_fp8,
    ) = make_inputs(batch, nheads, nheads_k, seqlen_q, seqlen_k, d)

    out, lse = aiter.fmha_fwd_mxfp8_asm(
        q_in,
        k_in,
        v_in,
        q_scale,
        k_scale,
        v_scale,
        return_lse=True,
    )

    if not check:
        print("[test] kernel returned (no correctness check requested).")
        return

    q_ref = q_fp8.to(torch.bfloat16).transpose(1, 2)
    k_ref = k_fp8.to(torch.bfloat16).transpose(1, 2)
    v_ref = v_fp8.to(torch.bfloat16).transpose(1, 2)
    out_ref, _, _ = attention_ref(q_ref, k_ref, v_ref, causal=False, upcast=True)

    out_fp32 = out.float()
    out_ref_fp32 = out_ref.float()
    max_diff = (out_fp32 - out_ref_fp32).abs().max().item()
    print(f"[test] max_diff={max_diff:.5f}")
    assert max_diff < 0.05, f"max_diff={max_diff} exceeds threshold 0.05"
    print("[test] PASS")


@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("d", [128])
@pytest.mark.parametrize("seqlen", [256, 384, 512, 1024, 8192])
@pytest.mark.parametrize("nheads, nheads_k", [(5, 5), (8, 2)])
@pytest.mark.parametrize("batch", [3])
def test_fmha_fwd_mxfp8_asm_correctness(batch, nheads, nheads_k, seqlen, d, causal):
    """MXFP8 ASM FMHA forward correctness across batch/head/seqlen shapes."""
    (
        q_in,
        k_in,
        v_in,
        q_scale,
        k_scale,
        v_scale,
        q_fp8,
        k_fp8,
        v_fp8,
    ) = make_inputs(batch, nheads, nheads_k, seqlen, seqlen, d)

    out, _ = aiter.fmha_fwd_mxfp8_asm(
        q_in,
        k_in,
        v_in,
        q_scale,
        k_scale,
        v_scale,
        is_causal=causal,
        return_lse=True,
    )

    q_ref = q_fp8.to(torch.bfloat16).transpose(1, 2)
    k_ref = k_fp8.to(torch.bfloat16).transpose(1, 2)
    v_ref = v_fp8.to(torch.bfloat16).transpose(1, 2)
    out_ref, _, _ = attention_ref(q_ref, k_ref, v_ref, causal=causal, upcast=True)

    max_diff = (out.float() - out_ref.float()).abs().max().item()
    print(
        f"[corr b={batch} hq={nheads} hk={nheads_k} seq={seqlen} d={d} "
        f"causal={causal}] max_diff={max_diff:.5f}"
    )
    assert max_diff < 0.05, f"max_diff={max_diff} exceeds threshold 0.05"


def run_perf(batch, nheads, nheads_k, seqlen_q, seqlen_k, d):
    (
        q_in,
        k_in,
        v_in,
        q_scale,
        k_scale,
        v_scale,
        *_,
    ) = make_inputs(batch, nheads, nheads_k, seqlen_q, seqlen_k, d)
    d_v = d

    (out, lse), us = run_perftest(
        aiter.fmha_fwd_mxfp8_asm,
        q_in,
        k_in,
        v_in,
        q_scale,
        k_scale,
        v_scale,
        return_lse=True,
    )

    fwd_flop = (
        batch * nheads * (seqlen_q * seqlen_k * d * 2 + seqlen_q * seqlen_k * d_v * 2)
    )
    tflops = fwd_flop / 1.0e6 / us
    quant_bytes = (
        batch
        * nheads
        * 1
        * (seqlen_q * d + seqlen_k * d + seqlen_k * d_v + seqlen_q * d_v)
    )
    gb_per_s = quant_bytes / 1.0e3 / us
    print(
        f"[perf] b={batch} nheads={nheads} nheads_k={nheads_k} "
        f"seqlen_q={seqlen_q} seqlen_k={seqlen_k} d={d}"
    )
    print(f"[perf] latency={us:.2f} us  tflops={tflops:.2f}  gb/s={gb_per_s:.2f}")


parser = argparse.ArgumentParser(description="MXFP8 ASM FMHA forward test (gfx1250)")
parser.add_argument("-b", "--batch", type=int, default=1)
parser.add_argument("-n", "--nheads", type=int, default=1)
parser.add_argument("-nk", "--nheads_k", type=int, default=1)
parser.add_argument("-q", "--seqlen_q", type=int, default=512)
parser.add_argument("-k", "--seqlen_k", type=int, default=512)
parser.add_argument("-d", "--head_dim", type=int, default=128)
parser.add_argument(
    "--no-check",
    action="store_true",
    help="run the kernel without the correctness check",
)
parser.add_argument(
    "--perf",
    action="store_true",
    help="run a performance measurement instead of the correctness check",
)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.perf:
        run_perf(
            args.batch,
            args.nheads,
            args.nheads_k,
            args.seqlen_q,
            args.seqlen_k,
            args.head_dim,
        )
    else:
        run_one(
            args.batch,
            args.nheads,
            args.nheads_k,
            args.seqlen_q,
            args.seqlen_k,
            args.head_dim,
            check=not args.no_check,
        )
