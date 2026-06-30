# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Correctness + performance test for the dedicated gfx1250 MXFP8 ASM FMHA
# forward path (aiter.fmha_fwd_mxfp8_asm).  Follows the aiter op_test standard:
# a @benchmark sweep fn whose call args are the table columns, candidates timed
# with run_perftest + checked with checkAllclose against a torch reference, and
# one markdown summary table emitted from main().

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.test_mha_common import attention_ref

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx1250"]  # MXFP8 ASM kernel is only shipped for gfx1250

BLOCK_SIZE = 32
SUB_Q = 256
SUB_K = 128


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
    """Build fp8 q/k/v as BSHD-shaped views over BHSD memory + e8m0 scales.

    Reproduces the real call layout: the MXFP8 kernel consumes bshd-shaped
    tensors backed by bhsd memory (head stride > seq stride), so q/k/v are
    transposed views of contiguous [b, h, s, d] tensors (not fresh contiguous
    [b, s, h, d] tensors).
    """
    torch.random.manual_seed(0)
    d_v = d

    q_bhsd = torch.randn(batch, nheads, seqlen_q, d, dtype=torch.bfloat16)
    k_bhsd = torch.randn(batch, nheads_k, seqlen_k, d, dtype=torch.bfloat16)
    v_bhsd = torch.randn(batch, nheads_k, seqlen_k, d_v, dtype=torch.bfloat16)

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


def run_torch(q_fp8, k_fp8, v_fp8, causal):
    """Reference only: bf16 attention over the dequantized fp8 inputs (fp32
    math inside attention_ref).  Not timed, not in the table."""
    q_ref = q_fp8.to(torch.bfloat16).transpose(1, 2)
    k_ref = k_fp8.to(torch.bfloat16).transpose(1, 2)
    v_ref = v_fp8.to(torch.bfloat16).transpose(1, 2)
    out_ref, _, _ = attention_ref(q_ref, k_ref, v_ref, causal=causal, upcast=True)
    return out_ref


@benchmark()
def test_fmha_fwd_mxfp8(batch, nheads, nheads_k, seqlen, d, causal):
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
    d_v = d

    ref = run_torch(q_fp8, k_fp8, v_fp8, causal)

    candidates = {
        "asm": lambda: aiter.fmha_fwd_mxfp8_asm(
            q_in,
            k_in,
            v_in,
            q_scale,
            k_scale,
            v_scale,
            is_causal=causal,
            return_lse=True,
        )[0],
    }

    # forward attention FLOPs: QK^T (sq*sk*d) + P*V (sq*sk*d_v), 2 per MAC.
    flops = 2 * batch * nheads * seqlen * seqlen * (d + d_v)
    # element traffic (bytes): fp8 q/k/v in + bf16 out.
    nbytes = (
        batch * nheads * seqlen * d  # q   (fp8, 1B)
        + batch * nheads_k * seqlen * d  # k   (fp8, 1B)
        + batch * nheads_k * seqlen * d_v  # v   (fp8, 1B)
        + batch * nheads * seqlen * d_v * 2  # out (bf16, 2B)
    )

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=2e-2,
            msg=f"{name}: fmha_fwd_mxfp8",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("fmha_fwd_mxfp8 unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="MXFP8 ASM FMHA forward test (gfx1250)",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[3],
        help="batch sizes to sweep",
    )
    parser.add_argument(
        "-hk",
        "--hqk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(5, 5), (8, 2)],
        help="(head_num_q, head_num_kv) pairs, e.g. 5,5 8,2",
    )
    parser.add_argument(
        "-s",
        "--seqlen",
        type=int,
        nargs="*",
        default=[256, 384, 512, 1024, 8192],
        help="sequence lengths to sweep (seqlen_q == seqlen_k)",
    )
    parser.add_argument(
        "-d",
        "--head_dim",
        type=int,
        nargs="*",
        default=[128],
        help="head dims to sweep",
    )
    parser.add_argument(
        "-c",
        "--causal",
        type=dtypes.str2bool,
        nargs="*",
        default=[False],
        help="causal flags to sweep (kernel currently supports False only)",
    )
    args = parser.parse_args()

    df = []
    for (nheads, nheads_k), batch, seqlen, d, causal in itertools.product(
        args.hqk, args.batch, args.seqlen, args.head_dim, args.causal
    ):
        df.append(test_fmha_fwd_mxfp8(batch, nheads, nheads_k, seqlen, d, causal))
    df = pd.DataFrame(df)
    aiter.logger.info(f"fmha_fwd_mxfp8 summary:\n{df.to_markdown(index=False)}")


if __name__ == "__main__":
    main()
