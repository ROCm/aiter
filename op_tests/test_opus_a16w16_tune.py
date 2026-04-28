# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
import os
import torch
import argparse

# opus kernel metadata + codegen live under csrc/opus_gemm/ (alongside C++
# sources). Import opus_gemm_common directly from there via sys.path so
# the aiter/ops/opus/ Python surface stays thin and metadata has a single
# source of truth.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "csrc", "opus_gemm"))
from opus_gemm_common import (
    a16w16_kernels_list,
    a16w16_flatmm_kernels_list,
    a16w16_flatmm_splitk_kernels_list,
)

from aiter import dtypes
from aiter.test_common import checkAllclose
# opus_gemm_a16w16_tune is the user-facing pybind binding, stays under
# aiter/ops/opus/ (not the shim in aiter/ops/deepgemm.py, which prints a
# DeprecationWarning).
from aiter.ops.opus.gemm_op_a16w16 import opus_gemm_a16w16_tune


# 33 kids total across three pipelines:
#   * 6 split-barrier a16w16 (ids 4..9)           - ignore splitK
#   * 16 a16w16_flatmm (4-wave warp-spec) (100..115) - ignore splitK
#   * 11 a16w16_flatmm_splitk (fp32 workspace + reduce) (200..210) - splitK=2 (KBatch=2)
ALL_KERNELS = {
    **a16w16_kernels_list,
    **a16w16_flatmm_kernels_list,
    **a16w16_flatmm_splitk_kernels_list,
}


def _flatmm_pfk(k):
    """Compute prefetch_k_iter for a flatmm / flatmm_splitk instance.

    Mirrors the traits formula so we can size K >= pfk * B_K before calling
    the kernel (which TORCH_CHECKs on the same bound).
    """
    sizeof_da = 2  # bf16 locked for flatmm family
    LOAD_GROUP_M = 64 if k.W_M >= 32 else 32
    LOAD_GROUP_N = 64 if k.W_N >= 32 else 32
    LOAD_GROUP_K = k.W_K * 2
    num_m = k.B_M // LOAD_GROUP_M
    num_n = k.B_N // LOAD_GROUP_N
    num_k = k.B_K // LOAD_GROUP_K
    smem_linear = 64 * 16 // sizeof_da  # WARP_SIZE=64
    smem_sub = smem_linear // LOAD_GROUP_K
    slots = LOAD_GROUP_M // smem_sub
    padding = 16 // sizeof_da if k.W_M >= 32 else 2 * 16 // sizeof_da
    per_glsz = slots * (smem_linear + padding) * sizeof_da
    per_iter = (num_m + num_n) * num_k * per_glsz
    lds_total = 163840
    return max(lds_total // max(k.WG_PER_CU, 1), 1) // max(per_iter, 1)


def _splitK_for_kid(k):
    """Default splitK to test per kid. splitk kids use KBatch=2 to exercise the
    split path; non-splitk kids get splitK=0 (ignored)."""
    if k.kernel_tag == "a16w16_flatmm_splitk":
        return 2
    return 0


def _min_k_for_kid(k):
    """Return minimum K that satisfies the pipeline's host-side constraint.

    Split-barrier a16w16: loops >= 2 and even -> K >= 2 * B_K.
    Flatmm (non-splitk):  loops >= pfk        -> K >= pfk * B_K.
    Flatmm splitk:        split_k * pfk * B_K <= K. We test with splitK=2, so
                          need K >= 2 * pfk * B_K.
    """
    if k.kernel_tag == "a16w16_flatmm":
        return _flatmm_pfk(k) * k.B_K
    if k.kernel_tag == "a16w16_flatmm_splitk":
        return 2 * _flatmm_pfk(k) * k.B_K
    return 2 * k.B_K


def torch_ref(XQ, WQ, out_dtype):
    return torch.bmm(XQ.float(), WQ.float().transpose(-1, -2)).to(out_dtype)


def test_one_kernel(kid, k, batch=1, out_dtype=torch.bfloat16):
    M = max(k.B_M, 64)
    N = max(k.B_N, 64)
    K = _min_k_for_kid(k)
    splitK = _splitK_for_kid(k)

    XQ = torch.randn(batch, M, K, device="cuda", dtype=dtypes.bf16)
    WQ = torch.randn(batch, N, K, device="cuda", dtype=dtypes.bf16)
    Y = torch.zeros(batch, M, N, device="cuda", dtype=out_dtype)

    ref = torch_ref(XQ, WQ, out_dtype)

    opus_gemm_a16w16_tune(XQ, WQ, Y, kid, splitK)

    err = checkAllclose(Y, ref, msg=f"kid={kid} splitK={splitK}", rtol=0.03, atol=0.03)
    if err > 0.001:
        raise RuntimeError(
            f"kid={kid} splitK={splitK} accuracy check failed: err={err:.6f} > 0.001 "
            f"(rtol=0.03, atol=0.03)"
        )
    return err


def main():
    parser = argparse.ArgumentParser(
        description="Test all a16w16 + a16w16_flatmm + a16w16_flatmm_splitk tune kernel instances"
    )
    parser.add_argument("--kid", type=int, default=None, help="Test a single kernel ID")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=["bf16", "fp32"],
        help="Output dtype (splitk kids now support both bf16 and fp32 via "
             "splitk_reduce_kernel<D_OUT>)",
    )
    parser.add_argument(
        "--tag", type=str, default=None,
        choices=["a16w16", "a16w16_flatmm", "a16w16_flatmm_splitk"],
        help="Filter kids by pipeline tag",
    )
    args = parser.parse_args()

    out_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    if args.kid is not None:
        kids = {args.kid: ALL_KERNELS[args.kid]}
    elif args.tag == "a16w16":
        kids = a16w16_kernels_list
    elif args.tag == "a16w16_flatmm":
        kids = a16w16_flatmm_kernels_list
    elif args.tag == "a16w16_flatmm_splitk":
        kids = a16w16_flatmm_splitk_kernels_list
    else:
        kids = ALL_KERNELS

    passed, failed, errored = 0, 0, 0
    for kid in sorted(kids):
        k = kids[kid]
        # splitk kids now accept both bf16 and fp32 Y (reduce kernel chooses
        # D_OUT at launch time), so honor --dtype for every kid.
        kid_out_dtype = out_dtype
        K = _min_k_for_kid(k)
        M = max(k.B_M, 64)
        N = max(k.B_N, 64)
        splitK = _splitK_for_kid(k)
        tag = (
            f"[{kid:3d}] {k.name}  (M={M} N={N} K={K} splitK={splitK} "
            f"out={str(kid_out_dtype).split('.')[-1]})"
        )
        try:
            err = test_one_kernel(kid, k, batch=args.batch, out_dtype=kid_out_dtype)
            print(f"  [PASS] {tag}  err={err:.4f}")
            passed += 1
        except RuntimeError as e:
            err_msg = str(e).split("\n")[0]
            print(f"  [FAIL] {tag}  {err_msg}")
            failed += 1
        except Exception as e:
            print(f"  [ERR ] {tag}  {type(e).__name__}: {e}")
            errored += 1

    total = passed + failed + errored
    print(f"\nSummary: {passed}/{total} passed, {failed} failed, {errored} errors")
    return 1 if (failed + errored) > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
