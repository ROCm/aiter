# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# fp8 block-scale UNIFORM (Route B fp8, 4-wave full-tile, direct store) batched
# GEMM for the DeepSeek-V4 output-LoRA (wo_a). Exercises both layout surfaces:
#   * bmajor: O/wo_a/Y = [G,M,K]/[G,N,K]/[G,M,N] (batch-major, contiguous).
#   * mmajor: O/Y = [M,G,*] (transposed view of the batch-major tensors) -- the
#     zero-copy DSV4 path (o=[num_tokens, n_groups, K] feeds with no transpose).
# Per-token A scale x_scale (GROUP_M=1), 128x128 block B scale w_scale. Sweeps
# Y dtype {fp32,bf16}. Compares uniform_scale kids (700=128x128, 701=256x128)
# against a dequant einsum reference (and a8w8_scale_mmajor where applicable).
#
# v1 kernel constraints: N % 128 == 0, K % 128 == 0, M % B_M == 0 (no store OOB
# masking). DSV4-Pro TP=8 wo_a maps to G=2, M=512, N=1024, K=4096.

import argparse
import itertools

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.opus.bmm_op import (
    _opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_raw,
    _opus_bmm_a8w8_mxscale_mmajor_raw,
    _opus_bmm_a8w8_mxscale_splitk_mmajor_raw,
    _opus_bmm_a8w8_uniform_scale_raw,
    _opus_bmm_a8w8_uniform_scale_mmajor_raw,
)

try:
    from aiter.ops.opus.bmm_op import _opus_bmm_a8w8_scale_mmajor_raw
except ImportError:  # pragma: no cover
    _opus_bmm_a8w8_scale_mmajor_raw = None

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]  # fp8 block-scale uniform is gfx950-only
GROUP = 128  # GROUP_N == GROUP_K == 128; GROUP_M == 1 (per-token)
_DT = {"fp32": dtypes.fp32, "bf16": dtypes.bf16}


def _quant_per_token(x_bf16):
    """x_bf16 [G,M,K] -> fp8 O + x_scale [G,M,K/128] (per-token, per-128-K)."""
    G, M, K = x_bf16.shape
    xb = x_bf16.to(dtypes.fp32).view(G, M, K // GROUP, GROUP)
    scale = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
    q = (xb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return q.view(G, M, K), scale.squeeze(-1).to(dtypes.fp32)


def _quant_block(w_bf16):
    """w_bf16 [G,N,K] -> fp8 W + w_scale [G,N/128,K/128] (128x128 blocks)."""
    G, N, K = w_bf16.shape
    wb = w_bf16.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
    scale = wb.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-8) / 448.0
    q = (wb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return q.view(G, N, K), scale.view(G, N // GROUP, K // GROUP).to(dtypes.fp32)


def _to_e8m0_scale(scale):
    # Round up to a power-of-two scale so quantized fp8 values stay in range.
    e = torch.ceil(torch.log2(scale.to(dtypes.fp32))).to(torch.int32) + 127
    e = torch.clamp(e, 0, 255).to(torch.uint8)
    scale_pow2 = torch.exp2(e.to(dtypes.fp32) - 127.0)
    return e, scale_pow2


def _quant_per_token_e8m0(x_bf16):
    G, M, K = x_bf16.shape
    xb = x_bf16.to(dtypes.fp32).view(G, M, K // GROUP, GROUP)
    raw_scale = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
    e8m0, scale = _to_e8m0_scale(raw_scale)
    q = (xb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return q.view(G, M, K), e8m0.squeeze(-1), scale.squeeze(-1)


def _quant_block_e8m0(w_bf16):
    G, N, K = w_bf16.shape
    wb = w_bf16.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
    raw_scale = wb.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-8) / 448.0
    e8m0, scale = _to_e8m0_scale(raw_scale)
    q = (wb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return (
        q.view(G, N, K),
        e8m0.view(G, N // GROUP, K // GROUP),
        scale.view(G, N // GROUP, K // GROUP),
    )


def run_torch(O_fp8, W_fp8, x_scale, w_scale):
    """Reference: dequant fp8 -> fp32 einsum -> [G,M,N]. Not timed."""
    G, M, K = O_fp8.shape
    N = W_fp8.shape[1]
    act = O_fp8.to(dtypes.fp32).view(G, M, K // GROUP, GROUP)
    act = (act * x_scale.unsqueeze(-1)).view(G, M, K)
    W = W_fp8.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
    W = (W * w_scale.view(G, N // GROUP, 1, K // GROUP, 1)).view(G, N, K)
    return torch.einsum("gmk,gnk->gmn", act, W).to(dtypes.fp32)


@benchmark()
def test_uniform_scale(m, n, k, g, layout, out_dtype):
    ydt = _DT[out_dtype]
    # Canonical batch-major tensors [G,M,K] / [G,N,K].
    O_bf16 = (torch.rand((g, m, k), dtype=dtypes.fp32) / 10).to(dtypes.bf16)
    W_bf16 = (torch.rand((g, n, k), dtype=dtypes.fp32) / 10).to(dtypes.bf16)
    O_fp8, x_scale = _quant_per_token(O_bf16)  # [g,m,k], [g,m,k/128]
    W_fp8, w_scale = _quant_block(W_bf16)  # [g,n,k], [g,n/128,k/128]

    ref_bm = run_torch(O_fp8, W_fp8, x_scale, w_scale)  # [g,m,n]

    if layout == "bmajor":
        O_in, xs_in = O_fp8, x_scale  # [g,m,k], [g,m,k/128] contiguous
        ref = ref_bm
        y_shape = (g, m, n)
        run = _opus_bmm_a8w8_uniform_scale_raw
        bf16_ref = torch.einsum(
            "gmk,gnk->gmn", O_bf16.to(dtypes.fp32), W_bf16.to(dtypes.fp32)
        )
    else:  # mmajor: transposed views (dim0=M, dim1=batch=G)
        O_in = O_fp8.transpose(0, 1)  # [m,g,k] view
        xs_in = x_scale.transpose(0, 1)  # [m,g,k/128] view
        ref = ref_bm.transpose(0, 1)  # [m,g,n]
        y_shape = (m, g, n)
        run = _opus_bmm_a8w8_uniform_scale_mmajor_raw
        bf16_ref = torch.einsum(
            "mgk,gnk->mgn",
            O_bf16.transpose(0, 1).to(dtypes.fp32),
            W_bf16.to(dtypes.fp32),
        )

    def _call(kid):
        Y = torch.empty(y_shape, dtype=ydt)
        run(O_in, W_fp8, Y, xs_in, w_scale, kid)
        return Y

    candidates = {
        "uniform700_128x128": (lambda: _call(700), ref),
        "uniform701_256x128": (lambda: _call(701), ref),
    }
    # a8w8_scale (8-wave quad) baseline: mmajor only, supports fp32/bf16 Y.
    if layout == "mmajor" and _opus_bmm_a8w8_scale_mmajor_raw is not None:

        def _a8w8():
            Y = torch.empty(y_shape, dtype=ydt)
            _opus_bmm_a8w8_scale_mmajor_raw(O_in, W_fp8, Y, xs_in, w_scale)
            return Y

        candidates["a8w8_scale"] = (_a8w8, ref)

        O_mx, xs_mx, xs_mx_fp32 = _quant_per_token_e8m0(O_bf16)
        W_mx, ws_mx, ws_mx_fp32 = _quant_block_e8m0(W_bf16)
        O_mx_in = O_mx.transpose(0, 1)
        xs_mx_in = xs_mx.transpose(0, 1)
        ref_mx = run_torch(O_mx, W_mx, xs_mx_fp32, ws_mx_fp32).transpose(0, 1)

        def _a8w8_mx():
            Y = torch.empty(y_shape, dtype=ydt)
            _opus_bmm_a8w8_mxscale_mmajor_raw(O_mx_in, W_mx, Y, xs_mx_in, ws_mx)
            return Y

        candidates["a8w8_mxscale"] = (_a8w8_mx, ref_mx)

        def _a8w8_mx_sk(split_k):
            Y = torch.empty(y_shape, dtype=ydt)
            _opus_bmm_a8w8_mxscale_splitk_mmajor_raw(
                O_mx_in, W_mx, Y, xs_mx_in, ws_mx, split_k
            )
            return Y

        # The 8-wave splitK pipeline requires at least 2 128-K tiles per split.
        for split_k in (2, 4, 6, 8, 16):
            if k // GROUP >= 2 * split_k:
                candidates[f"a8w8_mxscale_sk{split_k}"] = (
                    lambda split_k=split_k: _a8w8_mx_sk(split_k),
                    ref_mx,
                )

        def _a8w8_mx_flatmm_sk(split_k, kernel_id=0):
            Y = torch.empty(y_shape, dtype=ydt)
            _opus_bmm_a8w8_mxscale_flatmm_splitk_mmajor_raw(
                O_mx_in, W_mx, Y, xs_mx_in, ws_mx, split_k, kernel_id
            )
            return Y

        # Direct-store K256 variants bypass the split-K workspace and reduce
        # kernel.
        if k % (2 * GROUP) == 0:
            total_tiles_k256 = k // (2 * GROUP)
            if total_tiles_k256 >= 3:
                if m % 64 == 0 and n % 32 == 0:
                    candidates["a8w8_mxscale_flatmm_m64n32k256_sk1"] = (
                        lambda: _a8w8_mx_flatmm_sk(1, 320),
                        ref_mx,
                    )
                    candidates["a8w8_mxscale_flatmm_m64n32k256_wg1_sk1"] = (
                        lambda: _a8w8_mx_flatmm_sk(1, 322),
                        ref_mx,
                    )
                if m % 32 == 0 and n % 64 == 0:
                    candidates["a8w8_mxscale_flatmm_m32n64k256_sk1"] = (
                        lambda: _a8w8_mx_flatmm_sk(1, 640),
                        ref_mx,
                    )
                    candidates["a8w8_mxscale_flatmm_m32n64k256_selfload_sk1"] = (
                        lambda: _a8w8_mx_flatmm_sk(1, 646),
                        ref_mx,
                    )
                    candidates["a8w8_mxscale_flatmm_m32n64k256_wg1_sk1"] = (
                        lambda: _a8w8_mx_flatmm_sk(1, 642),
                        ref_mx,
                    )
                total_tiles_k128 = k // GROUP
                if total_tiles_k128 >= 3:
                    if m % 64 == 0 and n % 64 == 0:
                        candidates["a8w8_mxscale_flatmm_m64n64k128_sk1"] = (
                            lambda: _a8w8_mx_flatmm_sk(1, 650),
                            ref_mx,
                        )
                    if m % 64 == 0 and n % 128 == 0:
                        candidates["a8w8_mxscale_flatmm64_sk1"] = (
                            lambda: _a8w8_mx_flatmm_sk(1, 64),
                            ref_mx,
                        )
                    if m % 128 == 0 and n % 128 == 0:
                        candidates["a8w8_mxscale_flatmm_m128n128k128_wg1_sk1"] = (
                            lambda: _a8w8_mx_flatmm_sk(1, 128),
                            ref_mx,
                        )
                        candidates[
                            "a8w8_mxscale_flatmm_m128n128k128_persistent_mouter_wg1_sk1"
                        ] = (
                            lambda: _a8w8_mx_flatmm_sk(1, 131),
                            ref_mx,
                        )
                    if m % 128 == 0 and n % 256 == 0:
                        candidates["a8w8_mxscale_flatmm_m128n256k128_wave8n2_sk1"] = (
                            lambda: _a8w8_mx_flatmm_sk(1, 132),
                            ref_mx,
                        )
                        candidates[
                            "a8w8_mxscale_flatmm_m128n256k128_wave4n2_selfload_sk1"
                        ] = (
                            lambda: _a8w8_mx_flatmm_sk(1, 133),
                            ref_mx,
                        )
                    if m % 256 == 0 and n % 128 == 0:
                        candidates[
                            "a8w8_mxscale_flatmm_m256n128k128_wave4m2_selfload_sk1"
                        ] = (
                            lambda: _a8w8_mx_flatmm_sk(1, 134),
                            ref_mx,
                        )
                    if m % 64 == 0 and n % 256 == 0:
                        candidates["a8w8_mxscale_flatmm_m64n256k128_nphase_sk1"] = (
                            lambda: _a8w8_mx_flatmm_sk(1, 129),
                            ref_mx,
                        )

        # Base flatmm uses B_K=128 with a 3-slot prefetch pipeline; keep at
        # least 3 K tiles per split.
        for split_k in (2, 3, 4, 5, 8):
            total_tiles = k // GROUP
            iters_full = (total_tiles + split_k - 1) // split_k
            last_loops = total_tiles - (split_k - 1) * iters_full
            if last_loops >= 3:
                candidates[f"a8w8_mxscale_flatmm_sk{split_k}"] = (
                    lambda split_k=split_k: _a8w8_mx_flatmm_sk(split_k),
                    ref_mx,
                )
                if split_k == 2:
                    candidates["a8w8_mxscale_flatmm_fused_sk2"] = (
                        lambda: _a8w8_mx_flatmm_sk(2, 100),
                        ref_mx,
                    )
                if m % 64 == 0:
                    candidates[f"a8w8_mxscale_flatmm64_sk{split_k}"] = (
                        lambda split_k=split_k: _a8w8_mx_flatmm_sk(split_k, 64),
                        ref_mx,
                    )
                if n % 256 == 0:
                    candidates[f"a8w8_mxscale_flatmm256_sk{split_k}"] = (
                        lambda split_k=split_k: _a8w8_mx_flatmm_sk(split_k, 256),
                        ref_mx,
                    )
            if k % (2 * GROUP) == 0:
                total_tiles_k256 = k // (2 * GROUP)
                iters_full_k256 = (total_tiles_k256 + split_k - 1) // split_k
                last_loops_k256 = total_tiles_k256 - (split_k - 1) * iters_full_k256
                if last_loops_k256 >= 3:
                    if m % 64 == 0 and n % 32 == 0:
                        candidates[f"a8w8_mxscale_flatmm_m64n32k256_sk{split_k}"] = (
                            lambda split_k=split_k: _a8w8_mx_flatmm_sk(split_k, 320),
                            ref_mx,
                        )
                    if m % 32 == 0 and n % 64 == 0:
                        candidates[f"a8w8_mxscale_flatmm_m32n64k256_sk{split_k}"] = (
                            lambda split_k=split_k: _a8w8_mx_flatmm_sk(split_k, 640),
                            ref_mx,
                        )

    def _bf16_einsum():
        if layout == "mmajor":
            return torch.einsum("mgk,gnk->mgn", O_bf16.transpose(0, 1), W_bf16)
        return torch.einsum("gmk,gnk->gmn", O_bf16, W_bf16)

    candidates["bf16_einsum"] = (_bf16_einsum, bf16_ref)

    flops = 2 * m * g * n * k
    nbytes = (m * g * k + g * n * k + m * g * n) * O_fp8.element_size()

    ret = {"gfx": get_gfx()}
    for name, (fn, ref_i) in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            ref_i,
            out.to(dtypes.fp32),
            rtol=5e-2,
            atol=5e-2,
            msg=f"{name}: uniform_scale {layout}/{out_dtype}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("uniform_scale unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[2],
        help="G = n_local_groups (DSV4 wo_a batch); DSV4-Pro TP=8 -> G=2",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (512, 1024, 4096),
        ],
        help="(M=num_tokens, N=head_dim, K=o_lora_rank); DSV4-Pro TP=8 wo_a uses (512,1024,4096)",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        nargs="*",
        default=["mmajor"],
        help="tensor layout: mmajor (DSV4 zero-copy) or bmajor (batch-major)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        nargs="*",
        default=["fp32", "bf16"],
        help="Y output dtype: fp32 or bf16",
    )
    args = parser.parse_args()

    df = []
    for layout, out_dtype, g, (m, n, k) in itertools.product(
        args.layout, args.dtype, args.batch, args.mnk
    ):
        df.append(test_uniform_scale(m, n, k, g, layout, out_dtype))
    df = pd.DataFrame(df)
    aiter.logger.info(
        "uniform_scale wo_a summary (markdown):\n%s", df.to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
