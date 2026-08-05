# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.quant import per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import (
    shuffle_scale_a16w4,
    shuffle_scale_blockscale_a,
    shuffle_scale_blockscale_b,
    shuffle_weight,
)
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)
from aiter.utility import fp4_utils

torch.set_default_device("cuda")

# MFMA scaled 16x16x128 + the FlyDSL preshuffle layout are gfx950-only.
SUPPORTED_GFX = ["gfx950"]


def run_torch(codes_a, scale_a, codes_b, scale_b, group, dtype=dtypes.bf16):
    """Dequantize both operands to fp32 and do a plain linear.

    ``group`` is the scale's K granularity (32 for MX per-1x32, 128 for the coarse
    blockscale), so one reference serves both modes.
    """
    a_deq = codes_a.to(dtypes.fp32) * fp4_utils.e8m0_to_f32(
        scale_a.repeat_interleave(group, dim=1)
    )
    b_deq = codes_b.to(dtypes.fp32) * fp4_utils.e8m0_to_f32(
        scale_b.repeat_interleave(group, dim=1)
    )
    return F.linear(a_deq, b_deq).to(dtype)


@benchmark()
def test_gemm_a8w8_mxscale_preshuffle(m, n, k, dtype):
    # A is [M, K] fp8 codes, NOT preshuffled; B is shuffle_weight'd. Both scale
    # axes are padded to 32 rows before quant, the way the op wants them.
    ma, na = (m + 31) // 32 * 32, (n + 31) // 32 * 32
    a_f = torch.zeros(ma, k, dtype=dtypes.fp32)
    b_f = torch.zeros(na, k, dtype=dtypes.fp32)
    a_f[:m] = torch.randn(m, k)
    b_f[:n] = torch.randn(n, k)
    a_q, sa = per_1x32_f8_scale_f8_quant(
        a_f, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    b_q, sb = per_1x32_f8_scale_f8_quant(
        b_f, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    a_codes, b_codes = a_q[:m], b_q[:n]
    b_shuf = shuffle_weight(b_codes, layout=(16, 16))

    # Coarse blockscale (A 1x128, B 128x128) from the per-1x32 scale: block max so
    # the codes stay in range. A real blockscale quantizer emits sa_128 directly.
    sa_u, sb_u = sa.view(torch.uint8), sb.view(torch.uint8)
    sa_128 = sa_u.view(ma, k // 128, 4).amax(dim=2).contiguous()
    sb_128 = sb_u[:n].view(n // 128, 128, k // 128, 4).amax(dim=(1, 3)).contiguous()

    # Scale prep is caller-side and hoisted out of the timed region, matching the
    # model: B is weight-prep (once), A is per-call but a plain reshape+permute.
    scales = {
        # (a_scale, b_scale, reference operands, scale bytes read by the kernel)
        "blockscale": (
            shuffle_scale_blockscale_a(sa_128, k),
            shuffle_scale_blockscale_b(sb_128, n, k),
            (sa_128[:m], sb_128.repeat_interleave(128, dim=0), 128),
        ),
        "mx1x32": (
            shuffle_scale_a16w4(sa, 1, False),
            shuffle_scale_a16w4(sb, 1, False),
            (sa[:m], sb[:n], 32),
        ),
    }

    # A x B^T with fp8 codes: FLOPs = 2*M*N*K; bytes = codes + out + this mode's
    # scale traffic (blockscale reads 4x less A-scale and 256x less B-scale, which
    # is the whole point of the mode -- so count it per candidate).
    flops = 2 * m * n * k
    code_out_bytes = m * k + n * k + m * n * torch.empty(0, dtype=dtype).element_size()

    ret = {"gfx": get_gfx()}
    for name, (a_scale, b_scale, (ref_sa, ref_sb, group)) in scales.items():
        ref = run_torch(a_codes, ref_sa, b_codes, ref_sb, group, dtype)
        bs = name == "blockscale"
        out, us = run_perftest(
            lambda a_scale=a_scale, b_scale=b_scale, bs=bs: (
                aiter.gemm_a8w8_mxscale_preshuffle(
                    a_codes, b_shuf, a_scale, b_scale, dtype, blockscale=bs
                )
            )
        )
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: gemm a8w8 mxscale preshuffle",
        )
        nbytes = code_out_bytes + a_scale.numel() + b_scale.numel()
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    # Whole-op gate: the kernel is gfx950-only and needs the flydsl toolchain.
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "gemm_a8w8_mxscale_preshuffle unsupported on %s; skipping", get_gfx()
        )
        return
    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        aiter.logger.warning("flydsl not installed; skipping")
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16, fp16}",
        help="""Output data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        # deepseek-v4 a8w8 shapes, M sweeping decode -> prefill. Both modes need
        # K%128==0; blockscale additionally needs N%128==0.
        default=[
            (1, 2048, 7168),
            (16, 2048, 7168),
            (64, 2048, 7168),
            (256, 2048, 7168),
            (1024, 2048, 7168),
            (4096, 2048, 7168),
            (1, 7168, 3072),
            (64, 7168, 3072),
            (1024, 7168, 3072),
            (1, 6144, 7168),
            (64, 6144, 7168),
            (1024, 6144, 7168),
        ],
        help="""M,N,K of the gemm.
        e.g.: -s 64,2048,7168""",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for ((m, n, k),) in itertools.product(args.mnk):
            df.append(test_gemm_a8w8_mxscale_preshuffle(m, n, k, dtype))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "gemm_a8w8_mxscale_preshuffle summary (markdown):\n%s",
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
