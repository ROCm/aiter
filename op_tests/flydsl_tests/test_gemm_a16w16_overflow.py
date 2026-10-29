# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Regression tests for ROCm/aiter issue #6402 & gfx1250 FlyDSL 4 GiB wraparound:
[gfx1250] BF16 GEMM: FlyDSL gemm_a16w16 output stores wrap around at 2^32 bytes,
corrupting row 0 onward when padded output extent exceeds 4 GiB.

Root Cause:
----------
In `aiter.ops.flydsl.kernels.gemm_a16w16_kernel_gfx1250`, the output tensor C (or Y) was
accessed through a monolithic buffer descriptor `gY` created at the base address `arg_y`:
    num_records_bytes = M * ld_y * elem_bytes_d

Store operations in `store_final` computed a monolithic element offset from the matrix origin:
    c_off = row * ld_y + col_base
There was no software bounds check on the M dimension (no `row < M` guard), as the kernel
relied on the hardware buffer descriptor's `NUM_RECORDS` field to drop writes beyond M.

For shapes whose padded grid along M exceeds 2^32 bytes (such as M=14081, N=151936, BF16,
where M pads to 14336 with tile_m=256):
    Row stride in bytes = 151936 * 2 = 303,872 bytes
    Row 14134 ends at byte: 14135 * 303,872 - 1 = 4,295,230,719 (> 2^32)
    Row 14135 starts at byte: 14135 * 303,872 = 4,295,230,720 = 2^32 + 263,424 bytes
    4,295,230,720 % 2^32 = 263,424 bytes

AMD GPU buffer store instructions (BUFFER_STORE_DWORDX4 / BUFFER_STORE_SHORT) use a
32-bit voffset VGPR. When the offset exceeds 2^32 - 1, it wraps modulo 2^32 in unsigned
arithmetic:
    Wrapped byte offset: 263,424 bytes
    Destination row: 263,424 // 303,872 = 0 (Row 0!)
    Destination column: (263,424 % 303,872) // 2 = 131,712 (Column 131,712)

Because the wrapped offset (263,424) is strictly less than `num_records_bytes` (~4.28 GB),
the hardware bounds check evaluates to TRUE. The out-of-bounds padded threads in the final
M tile execute the write to low memory addresses, silently overwriting Row 0 onward!

Fix:
---
1. Re-anchor the output pointer per tile using 64-bit integer arithmetic:
       tile_c_base_elems = blk_m64 * ldy64
       tile_y_ptr = fx.add_offset(arg_y, tile_c_base_elems)
2. Bound the buffer descriptor to the valid rows remaining in this tile:
       valid_tile_rows = min(tile_m, max(0, M - blk_m))
       tile_num_records_bytes = valid_tile_rows * ldy64 * elem_bytes_d
   This guarantees that `NUM_RECORDS` never exceeds tile_m * N * 2 (~77.8 MB << 4 GiB).
3. Compute epilogue offsets locally:
       c_off = local_row * ld_y + col_base
   Local offsets never exceed ~77.8 MB, completely eliminating 32-bit voffset wraparound.
4. Add software guard `row_in_bounds = fx.Int32(global_row) < i32_m` so that padded threads
   in the final tile do not issue stores.
"""

import unittest

import pytest

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    torch = None
    F = None
    HAS_TORCH = False


class TestFlyDSLGemmOverflowArithmeticUnit(unittest.TestCase):
    """Unit tests verifying address arithmetic, boundary calculations, and wraparound math.

    Runs on CPU without requiring GPU hardware or AMD ROCm drivers.
    """

    def test_triton_2gb_boundary_arithmetic(self):
        """Verify exact byte offsets for M=7068, N=151936 BF16 (Triton 2 GiB boundary)."""
        M = 7068
        N = 151936
        elem_bytes = 2
        stride_bytes = N * elem_bytes  # 303,872 bytes

        total_bytes = M * stride_bytes
        self.assertEqual(total_bytes, 2147767296)  # 0x80045400
        self.assertGreater(total_bytes, 2**31)

        # Row 7067 starts below 2^31
        row7067_start = 7067 * stride_bytes
        self.assertEqual(row7067_start, 2147463424)
        self.assertLess(row7067_start, 2**31)

        # Column 10111 is the last byte before 2^31
        col10111_start = row7067_start + 10111 * elem_bytes
        self.assertEqual(col10111_start, 2147483646)
        self.assertLess(col10111_start, 2**31)

        # Column 10112 hits exactly 2^31
        col10112_start = row7067_start + 10112 * elem_bytes
        self.assertEqual(col10112_start, 2147483648)
        self.assertEqual(col10112_start, 2**31)

        # Padded rows for tile_m=256: 28 tiles -> 7168 rows
        m_padded = ((M + 255) // 256) * 256
        self.assertEqual(m_padded, 7168)
        padded_bytes = m_padded * stride_bytes
        self.assertEqual(padded_bytes, 2178154496)
        # Verify padded bytes do NOT exceed 2^32 (FlyDSL does not wrap for this shape)
        self.assertLess(padded_bytes, 2**32)

    def test_flydsl_4gb_wrap_arithmetic(self):
        """Verify exact byte offsets and wraparound for M=14081, N=151936 BF16."""
        M = 14081
        N = 151936
        elem_bytes = 2
        stride_bytes = N * elem_bytes  # 303,872 bytes

        # Logical output size is strictly below 2^32
        logical_bytes = M * stride_bytes
        self.assertEqual(logical_bytes, 4278821632)  # 0xff09a300
        self.assertLess(logical_bytes, 2**32)
        self.assertEqual(2**32 - logical_bytes, 16145664)  # 16.14 MB below 2^32

        # Padded grid with tile_m=256
        tile_m = 256
        tiles_m = (M + tile_m - 1) // tile_m
        self.assertEqual(tiles_m, 56)
        m_padded = tiles_m * tile_m
        self.assertEqual(m_padded, 14336)

        padded_bytes = m_padded * stride_bytes
        self.assertEqual(padded_bytes, 4356308992)  # 0x103a80000
        self.assertGreater(padded_bytes, 2**32)
        self.assertEqual(padded_bytes - 2**32, 61341696)  # 61.34 MB above 2^32

        # Final tile starts at row 14080 (the only valid row in tile 55)
        final_tile_start = 55 * tile_m
        self.assertEqual(final_tile_start, 14080)
        self.assertEqual(final_tile_start * stride_bytes, 4278517760)

        # First row crossing 2^32:
        # row_threshold = 2^32 / 303,872 = 14134.138
        row14134_start = 14134 * stride_bytes
        self.assertEqual(row14134_start, 4294926848)
        self.assertLess(row14134_start, 2**32)

        # Column 20224 in row 14134 crosses 2^32
        col20224_start = row14134_start + 20224 * elem_bytes
        self.assertEqual(col20224_start, 2**32)

        # Row 14135 is the first row entirely >= 2^32
        row14135_start = 14135 * stride_bytes
        self.assertEqual(row14135_start, 4295230720)
        self.assertGreaterEqual(row14135_start, 2**32)

        # 32-bit wraparound modulo 2^32
        wrapped_offset = row14135_start % (2**32)
        self.assertEqual(wrapped_offset, 263424)

        # Destination landing address:
        target_row = wrapped_offset // stride_bytes
        target_col = (wrapped_offset % stride_bytes) // elem_bytes
        self.assertEqual(target_row, 0)  # Row 0!
        self.assertEqual(target_col, 131712)  # Column 131,712

        # Verify that wrapped offset passes hardware descriptor bounds check if unpatched:
        # In unpatched code, num_records_bytes was 4,278,821,632 (clamped to 0xFFFFFFFF)
        self.assertLess(wrapped_offset, logical_bytes)

    def test_rebased_tile_local_bounds(self):
        """Verify that tile-local rebasing strictly bounds offsets and descriptor bounds."""
        M = 14081
        N = 151936
        elem_bytes = 2
        tile_m = 256
        stride_bytes = N * elem_bytes

        # Maximum local offset for ANY tile:
        # local_row in [0, 255], col in [0, 151935]
        max_local_row = tile_m - 1
        max_col = N - 1
        max_local_elems = max_local_row * N + max_col
        max_local_bytes = max_local_elems * elem_bytes

        # Maximum local bytes is ~77.8 MB, which is << 2^31 (2 GiB) and << 2^32 (4 GiB)
        self.assertLess(max_local_bytes, 78 * 1024 * 1024)
        self.assertLess(max_local_bytes, 2**31)

        # Final tile (Tile 55, blk_m = 14080):
        blk_m = 14080
        valid_tile_rows = min(tile_m, max(0, M - blk_m))
        self.assertEqual(valid_tile_rows, 1)  # Only row 14080 is valid

        tile_num_records_bytes = valid_tile_rows * stride_bytes
        self.assertEqual(tile_num_records_bytes, 303872)

        # Padded row 14081 has local_row = 1:
        padded_local_row = 1
        padded_start_bytes = padded_local_row * stride_bytes
        self.assertEqual(padded_start_bytes, 303872)
        # In hardware, padded_start_bytes >= tile_num_records_bytes -> OUT OF BOUNDS!
        self.assertGreaterEqual(padded_start_bytes, tile_num_records_bytes)


# ---------------------------------------------------------------------------
# Hardware Execution Tests (Require gfx1250 GPU)
# ---------------------------------------------------------------------------


def is_gfx1250_available():
    if not HAS_TORCH or not torch.cuda.is_available():
        return False
    try:
        from aiter.jit.utils.chip_info import get_gfx

        return get_gfx() == "gfx1250"
    except (KeyError, RuntimeError, AssertionError, OSError):
        return False


@pytest.mark.skipif(
    not is_gfx1250_available(), reason="Requires gfx1250 (MI350/MI450) hardware"
)
def test_gemm_a16w16_flydsl_control():
    """Control shape: M=4096, N=16384, K=128 BF16 (~128 MB output)."""
    from aiter.ops.flydsl.gemm_a16w16_gfx1250 import gemm_a16w16

    M, N, K = 4096, 16384, 128
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    y = torch.full((M, N), float("nan"), dtype=torch.bfloat16, device="cuda")

    gemm_a16w16(x, w, y=y)

    # Verify first 16 rows and last 16 rows against sliced reference
    for rows in (slice(0, 16), slice(M - 16, M)):
        ref = F.linear(x[rows, :], w)
        torch.testing.assert_close(y[rows, :], ref, atol=1e-1, rtol=1e-2)
        assert (
            not torch.isnan(y[rows, :]).any().item()
        ), "Output contains unwritten NaNs!"


@pytest.mark.skipif(
    not is_gfx1250_available(), reason="Requires gfx1250 (MI350/MI450) hardware"
)
def test_gemm_a16w16_flydsl_2gb_boundary():
    """2 GiB boundary test: M=7068, N=151936, K=64 BF16 (2.148 GB output)."""
    from aiter.ops.flydsl.gemm_a16w16_gfx1250 import gemm_a16w16

    M, N, K = 7068, 151936, 64
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    y = torch.full((M, N), float("nan"), dtype=torch.bfloat16, device="cuda")

    gemm_a16w16(x, w, y=y)

    # Check row 0, row 7066, and boundary row 7067
    for row_idx in (0, 1, 7066, 7067):
        rows = slice(row_idx, row_idx + 1)
        ref = F.linear(x[rows, :], w)
        torch.testing.assert_close(y[rows, :], ref, atol=1e-1, rtol=1e-2)
        assert (
            not torch.isnan(y[rows, :]).any().item()
        ), f"Row {row_idx} contains unwritten NaNs!"


@pytest.mark.skipif(
    not is_gfx1250_available(), reason="Requires gfx1250 (MI350/MI450) hardware"
)
def test_gemm_a16w16_flydsl_4gb_wrap_anti_corruption():
    """4 GiB wraparound test: M=14081, N=151936, K=64 BF16 (4.279 GB output, padded to 4.356 GB).

    Verifies that:
    1. Padded rows from the final tile do NOT wrap around and corrupt Row 0 onward.
    2. The last valid row (Row 14080) is written correctly without remaining NaNs.
    """
    from aiter.ops.flydsl.gemm_a16w16_gfx1250 import gemm_a16w16

    M, N, K = 14081, 151936, 64
    x = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
    y = torch.full((M, N), float("nan"), dtype=torch.bfloat16, device="cuda")

    gemm_a16w16(x, w, y=y)

    # Assertion 1: Anti-Corruption Check on Row 0 through Row 15
    # If 32-bit wraparound occurred, Row 14135 overwrote Row 0!
    ref_row0 = F.linear(x[0:16, :], w)
    torch.testing.assert_close(y[0:16, :], ref_row0, atol=1e-1, rtol=1e-2)

    # Assertion 2: Tail Correctness on Row 14080 (the only valid row in tile 55)
    ref_tail = F.linear(x[14080:14081, :], w)
    torch.testing.assert_close(y[14080:14081, :], ref_tail, atol=1e-1, rtol=1e-2)
    assert (
        not torch.isnan(y[14080, :]).any().item()
    ), "Tail row 14080 contains unwritten NaNs!"


if __name__ == "__main__":
    unittest.main()
