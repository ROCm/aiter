# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Functional test for gfx1250 F4GEMM (preload SGPR mode):
#   gemm_mxfp4_asm: D = A * B^T with e8m0 per-32 scales (intype=7)
#   gemm_nvfp4_asm: D = A * B^T with e4m3 per-32 scales + GlobalScaleA/B (intype=8)

import argparse
import torch

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_weight, shuffle_scale_f4
from aiter.utility import fp4_utils
from aiter.test_common import checkAllclose

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)
SCALE_BLOCK = 32


def _e4m3_to_f32(scale_u8: torch.Tensor) -> torch.Tensor:
    return scale_u8.view(torch.float8_e4m3fn).to(torch.float32)


def _rand_fp4_packed(rows: int, k: int, device="cuda") -> torch.Tensor:
    # Packed fp4: each uint8 carries two 4-bit nibbles → shape (rows, k/2).
    assert k % 2 == 0
    return torch.randint(0, 256, (rows, k // 2), dtype=torch.uint8, device=device)


def _rand_e8m0_scale(rows: int, k: int, device="cuda") -> torch.Tensor:
    # e8m0 = unsigned 8-bit exponent with bias 127. 0x7F = 1.0.
    return torch.randint(0x70, 0x88, (rows, k // SCALE_BLOCK), dtype=torch.uint8, device=device)


def _rand_e4m3_scale(rows: int, k: int, device="cuda") -> torch.Tensor:
    # Random small positive e4m3 values; reject NaN encoding (0x7F / 0xFF).
    raw = torch.randint(0x20, 0x50, (rows, k // SCALE_BLOCK), dtype=torch.uint8, device=device)
    return raw


def mxfp4_ref_matmul(A_fp4, B_fp4, sA_e8m0, sB_e8m0, M, N, K):
    A_f32 = fp4_utils.mxfp4_to_f32(A_fp4)[:M]                                   # (M, K)
    B_f32 = fp4_utils.mxfp4_to_f32(B_fp4)[:N]                                   # (N, K)
    sA = fp4_utils.e8m0_to_f32(sA_e8m0).repeat_interleave(SCALE_BLOCK, dim=1)   # (M, K)
    sB = fp4_utils.e8m0_to_f32(sB_e8m0).repeat_interleave(SCALE_BLOCK, dim=1)   # (N, K)
    return (A_f32 * sA) @ (B_f32 * sB).T


def nvfp4_ref_matmul(A_fp4, B_fp4, sA_e4m3, sB_e4m3, gA, gB, M, N, K):
    A_f32 = fp4_utils.mxfp4_to_f32(A_fp4)[:M]
    B_f32 = fp4_utils.mxfp4_to_f32(B_fp4)[:N]
    sA = _e4m3_to_f32(sA_e4m3).repeat_interleave(SCALE_BLOCK, dim=1)
    sB = _e4m3_to_f32(sB_e4m3).repeat_interleave(SCALE_BLOCK, dim=1)
    return float(gA) * float(gB) * (A_f32 * sA) @ (B_f32 * sB).T


def run_one(intype: str, M: int, N: int, K: int, apre: int):
    assert K % SCALE_BLOCK == 0, "K must be a multiple of 32"
    assert intype in ("mxfp4", "nvfp4")
    is_mx = intype == "mxfp4"

    A = _rand_fp4_packed(M, K)
    B = _rand_fp4_packed(N, K)
    if is_mx:
        sA = _rand_e8m0_scale(M, K)
        sB = _rand_e8m0_scale(N, K)
        ref = mxfp4_ref_matmul(A, B, sA, sB, M, N, K).to(dtypes.bf16)
    else:
        sA = _rand_e4m3_scale(M, K)
        sB = _rand_e4m3_scale(N, K)
        gA = 0.5
        gB = 0.5
        ref = nvfp4_ref_matmul(A, B, sA, sB, gA, gB, M, N, K).to(dtypes.bf16)

    # Preshuffle: B is always preshuffled; A optional per `apre`.
    A_dev = shuffle_weight(A, layout=(16, 16)) if apre else A
    B_dev = shuffle_weight(B, layout=(16, 16))
    intype_id = 7 if is_mx else 8
    sA_dev = shuffle_scale_f4(sA, intype_id)
    sB_dev = shuffle_scale_f4(sB, intype_id)

    if is_mx:
        out = aiter.gemm_mxfp4_asm(
            A_dev, B_dev, sA_dev, sB_dev,
            dtype=dtypes.bf16, a_preshuffle=bool(apre),
        )
    else:
        out = aiter.gemm_nvfp4_asm(
            A_dev, B_dev, sA_dev, sB_dev, gA, gB,
            dtype=dtypes.bf16, a_preshuffle=bool(apre),
        )

    err = checkAllclose(ref, out, rtol=1e-1, atol=1.0,
                        msg=f"{intype} M={M} N={N} K={K} apre={apre}")
    return err


def _str2tuple(s: str):
    return tuple(int(x) for x in s.split(","))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test gfx1250 F4GEMM (MXFP4 / NVFP4) ASM kernels")
    parser.add_argument("--intype", choices=["mxfp4", "nvfp4", "both"], default="both")
    parser.add_argument("--apre", type=int, choices=[0, 1], default=1,
                        help="A-preshuffle: 1 to preshuffle A, 0 to send row-major")
    parser.add_argument("-s", "--shape", type=_str2tuple, nargs="*",
                        default=[(256, 256, 256), (512, 512, 512)],
                        help="(M,N,K) tuples, e.g. -s 256,256,256 512,512,512")
    args = parser.parse_args()

    intypes = ["mxfp4", "nvfp4"] if args.intype == "both" else [args.intype]
    for it in intypes:
        for M, N, K in args.shape:
            run_one(it, M, N, K, args.apre)
