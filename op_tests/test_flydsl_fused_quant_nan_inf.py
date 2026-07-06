# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""NaN/Inf robustness of the fused grouped quant + e8m0 scale-preshuffle kernel.

The per-block amax reduction uses maxnumf (drops NaN) + minnumf(amax, FLT_MAX)
(clamps +/-Inf to finite). Net effect being verified here:

  * No e8m0 scale byte is ever the 0xFF NaN encoding -- so a bad activation
    never writes a NaN scale that the downstream GEMM would propagate.
  * The damage never crosses the 32-wide MX block that holds the bad element:
    other rows and other blocks in the same row are bit-identical to a finite
    baseline (rows are independent, warp-per-row).
  * NaN is dropped: even the bad element's own block scale + finite mates are
    identical to baseline (full isolation).
  * Inf is clamped to a finite max: its own block's scale rises but stays a
    valid exponent; the Inf element saturates to the fp8/fp4 max in HW.

Runs on the portable (non-pk8) path on gfx942 and the pk8 path on gfx1250.
"""

import pytest
import torch

from aiter.ops.flydsl.moe_kernels import flydsl_moe_fused_quant_preshuffle

MX_BLOCK = 32  # elements per e8m0 block


def _run(grouped_in, *, quant_mode, wmma_rep=1):
    E, max_m, feat_dim = grouped_in.shape
    return flydsl_moe_fused_quant_preshuffle(
        grouped_in, E, max_m, wmma_rep=wmma_rep, quant_mode=quant_mode
    )


@pytest.mark.parametrize("quant_mode", ["fp4", "fp8"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_bad_element_does_not_poison_beyond_its_block(quant_mode, bad_value):
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    device = "cuda"
    E, max_m, feat_dim = 1, 16, 128  # feat_dim % 128 == 0 -> scale dwords aligned
    is_nan = bad_value != bad_value

    # Constant 0.5 so each block's finite amax is 0.5 regardless of the injected
    # element -> for NaN (dropped) the finite amax is unchanged.
    base = torch.full((E, max_m, feat_dim), 0.5, dtype=torch.bfloat16, device=device)
    payload_ref, scale_ref = _run(base.clone(), quant_mode=quant_mode)

    bad = base.clone()
    bad_col = 5  # inside MX block 0 (cols 0..31) of row 0
    bad[0, 0, bad_col] = bad_value
    payload_bad, scale_bad = _run(bad, quant_mode=quant_mode)

    tag = f"{quant_mode}/{bad_value}"

    # (1) Core guarantee: no e8m0 scale byte is the 0xFF NaN encoding.
    assert (scale_bad != 0xFF).all(), f"{tag}: emitted an 0xFF (NaN) e8m0 scale byte"

    # (2) Isolation across rows: rows 1.. are bit-identical to the baseline
    #     (payload is plain row-major, so this is a direct slice compare).
    assert torch.equal(payload_bad[:, 1:, :], payload_ref[:, 1:, :]), (
        f"{tag}: payload of untouched rows changed"
    )

    # (3) Isolation across blocks within row 0: the bad element sits in block 0
    #     (cols 0..31); every later block's payload must be unchanged.
    per_byte = 2 if quant_mode == "fp4" else 1  # fp4 packs 2 elems/byte
    block0_bytes = MX_BLOCK // per_byte
    assert torch.equal(
        payload_bad[0, 0, block0_bytes:], payload_ref[0, 0, block0_bytes:]
    ), f"{tag}: payload of later blocks in the same row changed"

    if is_nan:
        # (4a) NaN is dropped: the whole scale tensor and the finite mates of
        #      block 0 are bit-identical to the finite baseline.
        assert torch.equal(scale_bad, scale_ref), f"{tag}: scale changed (NaN not dropped)"
        bad_byte = bad_col // per_byte
        mate_bytes = [b for b in range(block0_bytes) if b != bad_byte]
        assert torch.equal(
            payload_bad[0, 0, mate_bytes], payload_ref[0, 0, mate_bytes]
        ), f"{tag}: finite block-mates changed (NaN not isolated)"
    else:
        # (4b) Inf is clamped to a finite max: block 0's amax rises, so its scale
        #      differs from baseline but is still a valid exponent (covered by
        #      (1)). Confirm the clamp actually engaged.
        assert not torch.equal(scale_bad, scale_ref), (
            f"{tag}: expected Inf to raise its block's scale via the FLT_MAX clamp"
        )


@pytest.mark.parametrize("quant_mode", ["fp4", "fp8"])
def test_finite_input_valid_scales(quant_mode):
    """Sanity: ordinary finite activations only ever produce valid e8m0 bytes."""
    if not torch.cuda.is_available():
        pytest.skip("needs GPU")
    device = "cuda"
    E, max_m, feat_dim = 2, 32, 256
    torch.manual_seed(1)
    x = torch.randn((E, max_m, feat_dim), dtype=torch.bfloat16, device=device)
    _, scale = _run(x, quant_mode=quant_mode)
    assert (scale != 0xFF).all()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
