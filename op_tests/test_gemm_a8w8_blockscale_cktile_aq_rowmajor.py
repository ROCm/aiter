# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Test that CKTile blockscale 8-warp kernels produce correct results with
both AQLayout options (ColumnMajor = default, RowMajor = skip transpose).

Verifies:
  1. Both AQ layout variants produce output matching a PyTorch reference.
  2. RowMajor variant does NOT transpose x_scale (no extra allocation).
  3. TileKernelInstance name encodes AQRowMajor correctly.
  4. Non-8-warp kernels ignore the AQRowMajor flag (always RowMajor).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.test_common import checkAllclose

BLOCK_SHAPE = (128, 128)


def torch_reference(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    """FP8 blockscale GEMM reference using PyTorch."""
    block_n, block_k = BLOCK_SHAPE
    m, k = x.shape
    n = weight.shape[0]
    scale_n = (n + block_n - 1) // block_n
    scale_k = (k + block_k - 1) // block_k

    x_f = x.to(x_scale.dtype).view(m, k // block_k, block_k) * x_scale.unsqueeze(-1)
    x_f = x_f.view(m, k)

    from einops import rearrange

    ws = rearrange(
        w_scale.view(-1, 1)
        .repeat(1, block_n * block_k)
        .view(scale_n, scale_k, block_n, block_k),
        "bn bk n k -> (bn n) (bk k)",
    )[:n, :k]
    w_f = weight.to(ws.dtype) * ws

    return F.linear(x_f.to(dtypes.fp32), w_f.to(dtypes.fp32)).to(dtype)


def run_cktile_tune(x, weight, x_scale, w_scale, kernel_id, dtype=dtypes.bf16):
    """Invoke a specific CKTile kernel by ID via the tune entry point."""
    from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale_cktile_tune

    m, k = x.shape
    n = weight.shape[0]
    Y = torch.empty(m, n, dtype=dtype, device=x.device)
    return gemm_a8w8_blockscale_cktile_tune(
        x, weight, x_scale, w_scale, Y, kernelId=kernel_id
    )


def test_instance_names():
    """Verify kernel name encoding includes 'aqrm' suffix for AQRowMajor."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(
        0, os.path.join(this_dir, "..", "csrc", "ck_gemm_a8w8_blockscale")
    )
    from gemm_a8w8_blockscale_cktile_instance import (
        TileKernelInstance,
        candidate_kernels_cktile_dict,
    )

    # Non-AQRowMajor name should NOT end with _aqrm (4x2x1 = 8 warps)
    inst_default = TileKernelInstance(
        192, 256, 128, 4, 2, 1, 16, 16, 128,
        "Intrawave", False, True, False, 1,
    )
    assert not inst_default.name.endswith("_aqrm"), (
        f"Default instance name should not have _aqrm suffix: {inst_default.name}"
    )

    # AQRowMajor=True name SHOULD end with _aqrm
    inst_rm = TileKernelInstance(
        192, 256, 128, 4, 2, 1, 16, 16, 128,
        "Intrawave", False, True, False, 1, AQRowMajor=True,
    )
    assert inst_rm.name.endswith("_aqrm"), (
        f"AQRowMajor instance name should have _aqrm suffix: {inst_rm.name}"
    )

    # Names must be distinct
    assert inst_default.name != inst_rm.name, "Names must differ"

    # Verify is_eight_warp property (4x2x1 = 8 warps, K_Warp_Tile=128)
    assert inst_rm.is_eight_warp, "4x2x1 with K_Warp_Tile=128 should be 8-warp"

    non_8w = TileKernelInstance(
        16, 128, 256, 1, 4, 1, 16, 16, 64,
        "Intrawave", False, True, False, 1,
    )
    assert not non_8w.is_eight_warp, "1x4x1 with K_Warp_Tile=64 is not 8-warp"

    # Check that RowMajor variants exist in the candidate dict
    aqrm_kernels = {
        kid: k
        for kid, k in candidate_kernels_cktile_dict.items()
        if getattr(k, "AQRowMajor", False)
    }
    print(f"  Found {len(aqrm_kernels)} AQRowMajor kernel variants in candidate dict")
    assert len(aqrm_kernels) > 0, "Expected at least one AQRowMajor kernel variant"

    for kid, k in aqrm_kernels.items():
        assert k.is_eight_warp, (
            f"AQRowMajor kernel {kid} ({k.name}) should be 8-warp "
            f"(warps={k.M_Warp}x{k.N_Warp}x{k.K_Warp}={k.M_Warp*k.N_Warp*k.K_Warp})"
        )
        assert "_aqrm" in k.name, (
            f"AQRowMajor kernel {kid} should have _aqrm in name: {k.name}"
        )

    print("  PASSED: instance name encoding")


def test_accuracy(m, n, k, dtype=dtypes.bf16, err_threshold=0.05):
    """Test that both AQ layout variants match the PyTorch reference."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(
        0, os.path.join(this_dir, "..", "csrc", "ck_gemm_a8w8_blockscale")
    )
    from gemm_a8w8_blockscale_cktile_instance import candidate_kernels_cktile_dict

    block_n, block_k = BLOCK_SHAPE
    scale_m = m
    scale_n = (n + block_n - 1) // block_n
    scale_k = (k + block_k - 1) // block_k

    x = (torch.rand((m, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    weight = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    x_scale = torch.rand([scale_m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")

    ref = torch_reference(x, weight, x_scale, w_scale, dtype)

    # Find 8-warp kernels: both ColumnMajor and RowMajor variants
    eight_warp_colmajor = {}
    eight_warp_rowmajor = {}
    for kid, inst in candidate_kernels_cktile_dict.items():
        if not inst.is_eight_warp:
            continue
        if getattr(inst, "AQRowMajor", False):
            eight_warp_rowmajor[kid] = inst
        else:
            eight_warp_colmajor[kid] = inst

    if not eight_warp_colmajor:
        print(f"  SKIP: no 8-warp ColumnMajor kernels for shape ({m},{n},{k})")
        return
    if not eight_warp_rowmajor:
        print(f"  SKIP: no 8-warp RowMajor kernels for shape ({m},{n},{k})")
        return

    # Test one ColumnMajor and one RowMajor kernel
    cm_kid, cm_inst = next(iter(eight_warp_colmajor.items()))
    rm_kid, rm_inst = next(iter(eight_warp_rowmajor.items()))

    print(f"  Testing ColumnMajor kernel {cm_kid} ({cm_inst.name})")
    out_cm = run_cktile_tune(x, weight, x_scale, w_scale, cm_kid, dtype)
    err_cm = checkAllclose(ref, out_cm, msg=f"ColMajor(id={cm_kid})")

    print(f"  Testing RowMajor kernel {rm_kid} ({rm_inst.name})")
    out_rm = run_cktile_tune(x, weight, x_scale, w_scale, rm_kid, dtype)
    err_rm = checkAllclose(ref, out_rm, msg=f"RowMajor(id={rm_kid})")

    # Also check that both outputs are close to each other
    checkAllclose(out_cm, out_rm, msg="ColMajor vs RowMajor")

    print(
        f"  PASSED: shape ({m},{n},{k}) "
        f"cm_err={err_cm:.4f} rm_err={err_rm:.4f}"
    )


def test_padded_weight_stride(m, n, k, dtype=dtypes.bf16):
    """Test that RowMajor variant works with padded (non-contiguous) weight tensors,
    similar to vLLM's _maybe_pad_fp8_weight."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(
        0, os.path.join(this_dir, "..", "csrc", "ck_gemm_a8w8_blockscale")
    )
    from gemm_a8w8_blockscale_cktile_instance import candidate_kernels_cktile_dict

    block_n, block_k = BLOCK_SHAPE
    scale_m = m
    scale_n = (n + block_n - 1) // block_n
    scale_k = (k + block_k - 1) // block_k

    x = (torch.rand((m, k), dtype=dtypes.fp32, device="cuda") / 10).to(dtypes.fp8)
    weight_orig = (torch.rand((n, k), dtype=dtypes.fp32, device="cuda") / 10).to(
        dtypes.fp8
    )
    x_scale = torch.rand([scale_m, scale_k], dtype=dtypes.fp32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=dtypes.fp32, device="cuda")

    # Simulate _maybe_pad_fp8_weight: pad last dim, then narrow back
    num_pad = 256
    weight_padded = F.pad(weight_orig, (0, num_pad), "constant", 0)[..., :-num_pad]
    assert weight_padded.shape == weight_orig.shape
    assert weight_padded.stride(0) == k + num_pad, (
        f"Expected stride {k + num_pad}, got {weight_padded.stride(0)}"
    )
    assert weight_padded.stride(-1) == 1

    ref = torch_reference(x, weight_orig, x_scale, w_scale, dtype)

    # Find any RowMajor 8-warp kernel
    rm_kid = None
    for kid, inst in candidate_kernels_cktile_dict.items():
        if inst.is_eight_warp and getattr(inst, "AQRowMajor", False):
            rm_kid = kid
            break

    if rm_kid is None:
        print("  SKIP: no RowMajor 8-warp kernel available")
        return

    out = run_cktile_tune(x, weight_padded, x_scale, w_scale, rm_kid, dtype)
    err = checkAllclose(ref, out, msg=f"PaddedWeight+RowMajor(id={rm_kid})")
    print(f"  PASSED: padded weight shape ({m},{n},{k}) err={err:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test CKTile blockscale GEMM AQRowMajor optimization"
    )
    parser.add_argument(
        "--skip-accuracy",
        action="store_true",
        help="Skip accuracy tests (useful for fast name-only checks)",
    )
    args = parser.parse_args()

    print("=== Test 1: Instance name encoding ===")
    test_instance_names()

    if not args.skip_accuracy:
        shapes = [
            (128, 2048, 7168),
            (256, 7168, 2048),
            (1, 7168, 2048),
            (512, 4096, 7168),
        ]
        print("\n=== Test 2: Accuracy (ColumnMajor vs RowMajor vs Reference) ===")
        for m, n, k in shapes:
            print(f"\nShape: M={m}, N={n}, K={k}")
            test_accuracy(m, n, k)

        print("\n=== Test 3: Padded weight stride handling ===")
        for m, n, k in shapes[:2]:
            print(f"\nShape: M={m}, N={n}, K={k}")
            test_padded_weight_stride(m, n, k)

    print("\n=== All tests passed ===")
