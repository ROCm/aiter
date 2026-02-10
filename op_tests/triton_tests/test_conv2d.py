# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Comprehensive tests for the Triton Conv2d kernel.

Every test compares the Triton output against ``torch.nn.functional.conv2d``
as the ground-truth reference (computed in float32 for tighter tolerance).

Coverage:
  - kernel_size: 1x1 through 11x11, square and non-square
  - stride: 1 through 4, symmetric and asymmetric
  - padding: 0 through large values, symmetric and asymmetric
  - dilation: 1 through 4, symmetric and asymmetric
  - groups: 1, 2, 4, 8, 16, depthwise, depthwise-with-multiplier
  - bias: True / False
  - batch: 1 through 64
  - in_channels: 1 through 512
  - out_channels: 1 through 512
  - spatial sizes: 1x1 through 224x224, square and non-square
  - dtypes: float32, float16, bfloat16
  - unbatched (3-D) input
  - int scalar args vs tuple args
  - stride + dilation + padding combos
  - real-world architecture shapes (ResNet, VGG, MobileNet)
"""

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.conv2d import conv2d as triton_conv2d, conv2d_nhwc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ref_conv2d(input, weight, bias, stride, padding, dilation, groups):
    """Reference conv2d in float32."""
    return F.conv2d(
        input.float(), weight.float(),
        bias.float() if bias is not None else None,
        stride=stride, padding=padding, dilation=dilation, groups=groups,
    )


def _tols(dtype):
    """Return (atol, rtol) appropriate for *dtype*.

    fp32 uses 5e-4 / 1e-4 to accommodate accumulation error for large K
    (e.g. 512 channels * 9 kernel elements = 4608 FMAs).
    """
    if dtype == torch.float32:
        return 5e-4, 1e-4
    elif dtype == torch.float16:
        return 1e-2, 1e-2
    else:  # bfloat16
        return 2e-2, 2e-2


def _run_conv2d(
    batch, in_channels, in_h, in_w,
    out_channels, kernel_size, stride, padding, dilation, groups,
    bias, dtype,
    atol=None, rtol=None,
):
    """Shared test driver."""
    torch.cuda.empty_cache()
    if atol is None or rtol is None:
        _a, _r = _tols(dtype)
        atol = atol or _a
        rtol = rtol or _r

    x = torch.randn(batch, in_channels, in_h, in_w, dtype=dtype, device="cuda")
    w = torch.randn(
        out_channels, in_channels // groups, kernel_size[0], kernel_size[1],
        dtype=dtype, device="cuda",
    )
    b = torch.randn(out_channels, dtype=dtype, device="cuda") if bias else None

    ref = _ref_conv2d(x, w, b, stride, padding, dilation, groups).to(dtype)
    out = triton_conv2d(x, w, b, stride=stride, padding=padding,
                        dilation=dilation, groups=groups)

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# ===================================================================
#  Section 1: dtype and bias (fundamental correctness)
# ===================================================================

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [True, False])
class TestDtypeBias:
    def test_3x3(self, dtype, bias):
        _run_conv2d(2, 16, 32, 32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1, bias, dtype)

    def test_1x1(self, dtype, bias):
        _run_conv2d(2, 64, 16, 16, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias, dtype)

    def test_5x5_stride2(self, dtype, bias):
        _run_conv2d(2, 16, 32, 32, 32, (5, 5), (2, 2), (2, 2), (1, 1), 1, bias, dtype)

    def test_depthwise(self, dtype, bias):
        _run_conv2d(2, 32, 16, 16, 32, (3, 3), (1, 1), (1, 1), (1, 1), 32, bias, dtype)


# ===================================================================
#  Section 2: Kernel sizes — square
# ===================================================================

@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 9, 11])
def test_square_kernel(k):
    """Square kernel from 1x1 to 11x11."""
    h_in = max(k + 4, 16)  # ensure valid output
    _run_conv2d(1, 8, h_in, h_in, 16, (k, k), (1, 1), (0, 0), (1, 1), 1, True, torch.float32)


# ===================================================================
#  Section 3: Kernel sizes — non-square / asymmetric
# ===================================================================

@pytest.mark.parametrize("kh,kw", [
    (1, 3), (3, 1), (1, 5), (5, 1), (1, 7), (7, 1),
    (3, 5), (5, 3), (3, 7), (7, 3), (2, 4), (4, 2),
    (1, 11), (11, 1), (2, 3), (3, 2),
])
def test_nonsquare_kernel(kh, kw):
    h_in = max(kh + 4, 16)
    w_in = max(kw + 4, 16)
    _run_conv2d(1, 8, h_in, w_in, 16, (kh, kw), (1, 1), (0, 0), (1, 1), 1, True, torch.float32)


# ===================================================================
#  Section 4: Strides — square
# ===================================================================

@pytest.mark.parametrize("s", [1, 2, 3, 4])
def test_square_stride(s):
    _run_conv2d(2, 16, 32, 32, 16, (3, 3), (s, s), (1, 1), (1, 1), 1, True, torch.float32)


# ===================================================================
#  Section 5: Strides — asymmetric
# ===================================================================

@pytest.mark.parametrize("sh,sw", [
    (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (4, 1),
])
def test_asymmetric_stride(sh, sw):
    _run_conv2d(2, 16, 32, 32, 16, (3, 3), (sh, sw), (1, 1), (1, 1), 1, True, torch.float32)


# ===================================================================
#  Section 6: Padding — square
# ===================================================================

@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5])
def test_square_padding(p):
    _run_conv2d(1, 8, 16, 16, 8, (3, 3), (1, 1), (p, p), (1, 1), 1, False, torch.float32)


# ===================================================================
#  Section 7: Padding — asymmetric
# ===================================================================

@pytest.mark.parametrize("ph,pw", [
    (0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1), (2, 4), (4, 2),
])
def test_asymmetric_padding(ph, pw):
    _run_conv2d(1, 8, 16, 16, 8, (3, 3), (1, 1), (ph, pw), (1, 1), 1, True, torch.float32)


# ===================================================================
#  Section 8: Large padding (padding > kernel_size // 2)
# ===================================================================

@pytest.mark.parametrize("p", [3, 5, 7])
def test_large_padding(p):
    """Padding larger than half the kernel to exercise heavy zero-pad regions."""
    _run_conv2d(1, 4, 8, 8, 4, (3, 3), (1, 1), (p, p), (1, 1), 1, True, torch.float32)


# ===================================================================
#  Section 9: Dilation — square
# ===================================================================

@pytest.mark.parametrize("d", [1, 2, 3, 4])
def test_square_dilation(d):
    h = max(32, 3 + 2 * d)  # ensure valid output
    _run_conv2d(1, 8, h, h, 8, (3, 3), (1, 1), (0, 0), (d, d), 1, True, torch.float32)


# ===================================================================
#  Section 10: Dilation — asymmetric
# ===================================================================

@pytest.mark.parametrize("dh,dw", [
    (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2),
])
def test_asymmetric_dilation(dh, dw):
    _run_conv2d(1, 8, 32, 32, 8, (3, 3), (1, 1), (0, 0), (dh, dw), 1, True, torch.float32)


# ===================================================================
#  Section 11: Dilation + padding combo
# ===================================================================

@pytest.mark.parametrize("d,p", [
    (2, 2), (3, 3), (2, 4), (4, 4), (3, 6),
])
def test_dilation_with_padding(d, p):
    _run_conv2d(1, 8, 32, 32, 8, (3, 3), (1, 1), (p, p), (d, d), 1, True, torch.float32)


# ===================================================================
#  Section 12: Groups — various
# ===================================================================

@pytest.mark.parametrize("in_c,out_c,groups", [
    # Standard grouped conv
    (16, 16, 1),
    (16, 16, 2),
    (16, 16, 4),
    (16, 16, 8),
    (16, 16, 16),   # depthwise
    (32, 32, 1),
    (32, 32, 2),
    (32, 32, 4),
    (32, 32, 8),
    (32, 32, 16),
    (32, 32, 32),   # depthwise
    (64, 64, 1),
    (64, 64, 4),
    (64, 64, 16),
    (64, 64, 64),   # depthwise
    # Groups with different in/out channels
    (16, 32, 2),
    (16, 32, 4),
    (16, 32, 8),
    (16, 64, 4),
    (32, 64, 8),
    (64, 128, 16),
    # Depthwise with channel multiplier > 1
    (16, 32, 16),   # multiplier 2
    (16, 48, 16),   # multiplier 3
    (32, 64, 32),   # multiplier 2
    (8, 32, 8),     # multiplier 4
])
def test_groups_varied(in_c, out_c, groups):
    _run_conv2d(2, in_c, 16, 16, out_c, (3, 3), (1, 1), (1, 1), (1, 1), groups, True, torch.float32)


# ===================================================================
#  Section 13: Groups + stride + dilation
# ===================================================================

@pytest.mark.parametrize("groups,stride,dilation", [
    (4, (2, 2), (1, 1)),
    (4, (1, 1), (2, 2)),
    (4, (2, 2), (2, 2)),
    (16, (2, 2), (1, 1)),
    (16, (1, 1), (2, 2)),
    (32, (2, 2), (1, 1)),
])
def test_groups_with_stride_dilation(groups, stride, dilation):
    in_c = 32
    out_c = 32
    _run_conv2d(2, in_c, 32, 32, out_c, (3, 3), stride, (1, 1), dilation, groups, True, torch.float32)


# ===================================================================
#  Section 14: Channel counts — small
# ===================================================================

@pytest.mark.parametrize("in_c,out_c", [
    (1, 1), (1, 2), (1, 4), (1, 8), (1, 16),
    (2, 1), (2, 2), (2, 4),
    (3, 1), (3, 8), (3, 16), (3, 32), (3, 64),  # RGB-like input
    (4, 4), (4, 8), (4, 16),
])
def test_small_channel_counts(in_c, out_c):
    _run_conv2d(1, in_c, 16, 16, out_c, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


# ===================================================================
#  Section 15: Channel counts — medium to large
# ===================================================================

@pytest.mark.parametrize("in_c,out_c", [
    (16, 32), (32, 64), (64, 128), (128, 256), (256, 512),
    (32, 16), (64, 32), (128, 64), (256, 128),  # reduction
    (64, 64), (128, 128), (256, 256),            # same
    (512, 512),
])
def test_medium_large_channels(in_c, out_c):
    # Use smaller spatial for large channel counts to keep memory reasonable
    h = 8 if in_c >= 256 else 14
    _run_conv2d(1, in_c, h, h, out_c, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


# ===================================================================
#  Section 16: Spatial sizes — small
# ===================================================================

@pytest.mark.parametrize("h,w,k", [
    (1, 1, 1),   # 1x1 spatial, 1x1 kernel
    (2, 2, 1),
    (3, 3, 1),
    (3, 3, 3),
    (4, 4, 3),
    (5, 5, 3),
    (5, 5, 5),
    (6, 6, 3),
    (7, 7, 3),
    (7, 7, 5),
    (7, 7, 7),
])
def test_small_spatial(h, w, k):
    pad = k // 2
    _run_conv2d(1, 8, h, w, 8, (k, k), (1, 1), (pad, pad), (1, 1), 1, True, torch.float32)


# ===================================================================
#  Section 17: Spatial sizes — medium / standard
# ===================================================================

@pytest.mark.parametrize("h,w", [
    (8, 8), (14, 14), (16, 16), (28, 28), (32, 32), (56, 56),
])
def test_medium_spatial(h, w):
    _run_conv2d(1, 16, h, w, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


# ===================================================================
#  Section 18: Spatial sizes — large
# ===================================================================

@pytest.mark.parametrize("h,w", [
    (112, 112), (128, 128), (224, 224),
])
def test_large_spatial(h, w):
    _run_conv2d(1, 3, h, w, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


# ===================================================================
#  Section 19: Spatial sizes — non-square
# ===================================================================

@pytest.mark.parametrize("h,w", [
    (7, 13), (13, 7), (16, 32), (32, 16), (28, 56), (56, 28),
    (1, 16), (16, 1), (3, 32), (32, 3), (14, 7), (7, 14),
    (100, 50), (50, 100),
])
def test_nonsquare_spatial(h, w):
    _run_conv2d(1, 8, h, w, 8, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


# ===================================================================
#  Section 20: Batch sizes
# ===================================================================

@pytest.mark.parametrize("batch", [1, 2, 3, 4, 7, 8, 15, 16, 32, 64])
def test_batch_sizes(batch):
    _run_conv2d(batch, 16, 16, 16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


# ===================================================================
#  Section 21: Unbatched (3-D) input
# ===================================================================

@pytest.mark.parametrize("in_c,out_c,k,p", [
    (8, 16, 3, 1),
    (3, 32, 5, 2),
    (16, 16, 1, 0),
    (1, 8, 3, 1),
])
def test_unbatched(in_c, out_c, k, p):
    """3-D input (C, H, W) should produce 3-D output."""
    x = torch.randn(in_c, 16, 16, dtype=torch.float32, device="cuda")
    w = torch.randn(out_c, in_c, k, k, dtype=torch.float32, device="cuda")
    b = torch.randn(out_c, dtype=torch.float32, device="cuda")

    ref = F.conv2d(x.unsqueeze(0).float(), w.float(), b.float(), padding=p).squeeze(0)
    out = triton_conv2d(x, w, b, padding=p)

    assert out.dim() == 3, f"Expected 3-D output, got {out.dim()}-D"
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-5)


# ===================================================================
#  Section 22: Scalar int arguments (not tuples)
# ===================================================================

def test_int_stride():
    """stride=2 rather than (2,2)."""
    x = torch.randn(1, 8, 16, 16, dtype=torch.float32, device="cuda")
    w = torch.randn(8, 8, 3, 3, dtype=torch.float32, device="cuda")
    ref = F.conv2d(x, w, stride=2, padding=1)
    out = triton_conv2d(x, w, stride=2, padding=1)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-5)


def test_int_padding():
    """padding=1 rather than (1,1)."""
    x = torch.randn(1, 8, 16, 16, dtype=torch.float32, device="cuda")
    w = torch.randn(8, 8, 3, 3, dtype=torch.float32, device="cuda")
    ref = F.conv2d(x, w, padding=1)
    out = triton_conv2d(x, w, padding=1)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-5)


def test_int_dilation():
    """dilation=2 rather than (2,2)."""
    x = torch.randn(1, 8, 32, 32, dtype=torch.float32, device="cuda")
    w = torch.randn(8, 8, 3, 3, dtype=torch.float32, device="cuda")
    ref = F.conv2d(x, w, dilation=2)
    out = triton_conv2d(x, w, dilation=2)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-5)


def test_int_kernel_all_args():
    """All args as plain ints."""
    x = torch.randn(2, 16, 32, 32, dtype=torch.float32, device="cuda")
    w = torch.randn(32, 16, 3, 3, dtype=torch.float32, device="cuda")
    b = torch.randn(32, dtype=torch.float32, device="cuda")
    ref = F.conv2d(x, w, b, stride=2, padding=1, dilation=1, groups=1)
    out = triton_conv2d(x, w, b, stride=2, padding=1, dilation=1, groups=1)
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-5)


# ===================================================================
#  Section 23: Edge cases
# ===================================================================

def test_single_pixel_input():
    """1x1 spatial input with 1x1 kernel."""
    _run_conv2d(1, 8, 1, 1, 8, (1, 1), (1, 1), (0, 0), (1, 1), 1, True, torch.float32)


def test_single_pixel_input_3x3_padded():
    """1x1 spatial input with 3x3 kernel and padding=1 -> 1x1 output."""
    _run_conv2d(1, 4, 1, 1, 4, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


def test_single_channel_in_out():
    """1 in_channel, 1 out_channel."""
    _run_conv2d(1, 1, 16, 16, 1, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


def test_single_pixel_output_from_stride():
    """Stride so large that output is 1x1."""
    _run_conv2d(1, 8, 7, 7, 8, (3, 3), (5, 5), (0, 0), (1, 1), 1, True, torch.float32)


def test_kernel_equals_input():
    """Kernel exactly matches input spatial size -> 1x1 output."""
    for sz in [3, 5, 7]:
        _run_conv2d(1, 4, sz, sz, 4, (sz, sz), (1, 1), (0, 0), (1, 1), 1, True, torch.float32)


def test_wide_output_channels():
    """Many more output channels than input."""
    _run_conv2d(1, 4, 8, 8, 256, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


def test_narrow_output_channels():
    """Many more input channels than output (bottleneck)."""
    _run_conv2d(1, 256, 8, 8, 4, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


def test_odd_channel_counts():
    """Odd (non-power-of-2) channel counts."""
    _run_conv2d(1, 3, 16, 16, 7, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)
    _run_conv2d(1, 5, 16, 16, 13, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)
    _run_conv2d(1, 7, 16, 16, 11, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


def test_even_kernel_sizes():
    """Even-sized kernels (2x2, 4x4, 6x6) which are less common but valid."""
    for k in [2, 4, 6]:
        _run_conv2d(1, 8, 16, 16, 8, (k, k), (1, 1), (0, 0), (1, 1), 1, True, torch.float32)


# ===================================================================
#  Section 24: Comprehensive stride + dilation + padding combos
# ===================================================================

@pytest.mark.parametrize("stride,dilation,padding,kernel_size", [
    ((2, 2), (1, 1), (1, 1), (3, 3)),
    ((1, 1), (2, 2), (2, 2), (3, 3)),
    ((2, 2), (2, 2), (2, 2), (3, 3)),
    ((2, 1), (1, 2), (0, 1), (3, 3)),
    ((1, 2), (2, 1), (1, 0), (3, 3)),
    ((3, 3), (1, 1), (1, 1), (3, 3)),
    ((2, 2), (1, 1), (0, 0), (5, 5)),
    ((2, 2), (2, 2), (4, 4), (5, 5)),
    ((1, 1), (3, 3), (3, 3), (3, 3)),
    ((2, 2), (1, 1), (3, 3), (7, 7)),
    ((2, 3), (1, 2), (1, 2), (3, 5)),
    ((1, 1), (1, 1), (0, 0), (1, 1)),
    ((4, 4), (1, 1), (0, 0), (3, 3)),
])
def test_stride_dilation_padding_kernel_combos(stride, dilation, padding, kernel_size):
    _run_conv2d(
        2, 16, 32, 32, 16, kernel_size, stride, padding, dilation, 1,
        True, torch.float32,
    )


# ===================================================================
#  Section 25: Full combo — groups + stride + dilation + padding + kernel + bias
# ===================================================================

@pytest.mark.parametrize("groups,bias", [(1, True), (2, True), (4, False), (16, True)])
@pytest.mark.parametrize("stride,padding", [((1, 1), (1, 1)), ((2, 2), (0, 0))])
def test_full_combo(groups, bias, stride, padding):
    _run_conv2d(
        2, 16, 32, 32, 32, (3, 3), stride, padding, (1, 1), groups,
        bias, torch.float32,
    )


# ===================================================================
#  Section 26: Real-world architecture shapes
# ===================================================================

class TestResNet:
    """Shapes from ResNet-50."""

    def test_stem(self):
        """Conv1: 3->64, 7x7, stride=2, pad=3."""
        _run_conv2d(1, 3, 224, 224, 64, (7, 7), (2, 2), (3, 3), (1, 1), 1, True, torch.float32)

    def test_block1_3x3(self):
        _run_conv2d(1, 64, 56, 56, 64, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)

    def test_block1_1x1_expand(self):
        _run_conv2d(1, 64, 56, 56, 256, (1, 1), (1, 1), (0, 0), (1, 1), 1, True, torch.float32)

    def test_block1_1x1_reduce(self):
        _run_conv2d(1, 256, 56, 56, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, True, torch.float32)

    def test_block2_downsample(self):
        _run_conv2d(1, 128, 28, 28, 128, (3, 3), (2, 2), (1, 1), (1, 1), 1, True, torch.float32)

    def test_block3_3x3(self):
        _run_conv2d(1, 256, 14, 14, 256, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)

    def test_block4_3x3(self):
        _run_conv2d(1, 512, 7, 7, 512, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)

    def test_batched_block1(self):
        _run_conv2d(8, 64, 56, 56, 64, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


class TestVGG:
    """Shapes from VGG-16."""

    def test_conv1_1(self):
        _run_conv2d(1, 3, 224, 224, 64, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)

    def test_conv2_1(self):
        _run_conv2d(1, 64, 112, 112, 128, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)

    def test_conv3_1(self):
        _run_conv2d(1, 128, 56, 56, 256, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)

    def test_conv4_1(self):
        _run_conv2d(1, 256, 28, 28, 512, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)

    def test_conv5_1(self):
        _run_conv2d(1, 512, 14, 14, 512, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


class TestMobileNet:
    """Depthwise separable conv shapes from MobileNet."""

    def test_dw_112(self):
        """Depthwise 3x3 at 112x112."""
        _run_conv2d(1, 32, 112, 112, 32, (3, 3), (1, 1), (1, 1), (1, 1), 32, True, torch.float32)

    def test_pw_112(self):
        """Pointwise 1x1 at 112x112."""
        _run_conv2d(1, 32, 112, 112, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, True, torch.float32)

    def test_dw_stride2(self):
        """Depthwise 3x3 stride-2 downsampling."""
        _run_conv2d(1, 64, 56, 56, 64, (3, 3), (2, 2), (1, 1), (1, 1), 64, True, torch.float32)

    def test_pw_56(self):
        _run_conv2d(1, 64, 56, 56, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, True, torch.float32)

    def test_dw_28(self):
        _run_conv2d(1, 128, 28, 28, 128, (3, 3), (1, 1), (1, 1), (1, 1), 128, True, torch.float32)

    def test_dw_14(self):
        _run_conv2d(1, 256, 14, 14, 256, (3, 3), (1, 1), (1, 1), (1, 1), 256, True, torch.float32)

    def test_dw_7(self):
        _run_conv2d(1, 512, 7, 7, 512, (3, 3), (1, 1), (1, 1), (1, 1), 512, True, torch.float32)


class TestEfficientNet:
    """Shapes inspired by EfficientNet (various kernel sizes, dilation)."""

    def test_k5_dw(self):
        """5x5 depthwise."""
        _run_conv2d(1, 40, 28, 28, 40, (5, 5), (1, 1), (2, 2), (1, 1), 40, True, torch.float32)

    def test_k5_stride2_dw(self):
        _run_conv2d(1, 80, 14, 14, 80, (5, 5), (2, 2), (2, 2), (1, 1), 80, True, torch.float32)

    def test_k3_dw_dilation2(self):
        """Dilated depthwise conv."""
        _run_conv2d(1, 64, 28, 28, 64, (3, 3), (1, 1), (2, 2), (2, 2), 64, True, torch.float32)


# ===================================================================
#  Section 27: Stress tests — larger configurations
# ===================================================================

@pytest.mark.parametrize("batch,in_c,h,out_c", [
    (8, 64, 56, 64),
    (16, 128, 28, 128),
    (32, 64, 28, 64),
    (4, 256, 14, 256),
    (2, 512, 7, 512),
    (64, 16, 16, 16),
])
def test_stress(batch, in_c, h, out_c):
    _run_conv2d(batch, in_c, h, h, out_c, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


# ===================================================================
#  Section 28: Half-precision stress (fp16 and bf16 at various sizes)
# ===================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("in_c,out_c,h,k", [
    (3, 64, 32, 7),
    (64, 64, 16, 3),
    (128, 128, 8, 3),
    (16, 16, 32, 5),
    (32, 32, 16, 3),  # depthwise-like but groups=1
])
def test_half_precision_sizes(dtype, in_c, out_c, h, k):
    p = k // 2
    _run_conv2d(2, in_c, h, h, out_c, (k, k), (1, 1), (p, p), (1, 1), 1, True, dtype)


# ===================================================================
#  Section 29: No-bias with all parameter combos
# ===================================================================

@pytest.mark.parametrize("k,s,p,d,g", [
    (1, 1, 0, 1, 1),
    (3, 1, 1, 1, 1),
    (3, 2, 1, 1, 1),
    (5, 1, 2, 1, 1),
    (3, 1, 0, 2, 1),
    (3, 1, 1, 1, 4),
    (3, 2, 1, 1, 16),
    (1, 1, 0, 1, 16),
])
def test_no_bias_combos(k, s, p, d, g):
    _run_conv2d(
        2, 16, 32, 32, 16, (k, k), (s, s), (p, p), (d, d), g,
        False, torch.float32,
    )


# ===================================================================
#  Section 30: Output shape correctness (no value check, just shape)
# ===================================================================

@pytest.mark.parametrize("N,Ci,H,W,Co,kH,kW,sH,sW,pH,pW,dH,dW,g", [
    (1, 3, 224, 224, 64, 7, 7, 2, 2, 3, 3, 1, 1, 1),
    (2, 16, 32, 32, 32, 3, 3, 1, 1, 1, 1, 1, 1, 1),
    (1, 64, 56, 56, 256, 1, 1, 1, 1, 0, 0, 1, 1, 1),
    (4, 128, 14, 14, 128, 3, 3, 2, 2, 1, 1, 1, 1, 1),
    (1, 16, 16, 16, 16, 3, 3, 1, 1, 2, 2, 2, 2, 1),
    (1, 32, 7, 7, 32, 3, 3, 1, 1, 1, 1, 1, 1, 32),
    (1, 8, 10, 20, 16, 3, 5, 2, 3, 1, 2, 1, 1, 1),
    (3, 6, 15, 15, 12, 5, 5, 1, 1, 0, 0, 1, 1, 3),
])
def test_output_shape(N, Ci, H, W, Co, kH, kW, sH, sW, pH, pW, dH, dW, g):
    x = torch.randn(N, Ci, H, W, dtype=torch.float32, device="cuda")
    w = torch.randn(Co, Ci // g, kH, kW, dtype=torch.float32, device="cuda")
    ref = F.conv2d(x, w, stride=(sH, sW), padding=(pH, pW), dilation=(dH, dW), groups=g)
    out = triton_conv2d(x, w, stride=(sH, sW), padding=(pH, pW), dilation=(dH, dW), groups=g)
    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"


# ===================================================================
#  Section 31: Large kernel with dilation (atrous convolution)
# ===================================================================

@pytest.mark.parametrize("k,d", [
    (3, 2), (3, 3), (3, 4), (3, 6),
    (5, 2), (5, 3),
    (7, 2),
])
def test_atrous_conv(k, d):
    """Atrous / dilated convolution with large effective receptive field."""
    p = d * (k // 2)  # 'same'-like padding
    h = max(32, k + 2 * d * (k - 1))
    _run_conv2d(1, 8, h, h, 8, (k, k), (1, 1), (p, p), (d, d), 1, True, torch.float32)


# ===================================================================
#  Section 32: Regression-style sweep (auto-generated combos)
# ===================================================================

def _sweep_configs():
    """Generate a variety of (batch, Ci, H, W, Co, k, s, p, d, g) tuples."""
    configs = []
    for k in [1, 3, 5]:
        for s in [1, 2]:
            for d in [1, 2]:
                p = d * (k // 2)  # enough to keep output > 0
                for g in [1, 4]:
                    Ci = 16
                    Co = 16
                    H = max(16, k + 2 * d * (k - 1))
                    configs.append((1, Ci, H, H, Co, k, s, p, d, g))
    return configs


@pytest.mark.parametrize(
    "batch,Ci,H,W,Co,k,s,p,d,g",
    _sweep_configs(),
    ids=[f"k{c[5]}s{c[6]}d{c[8]}g{c[9]}" for c in _sweep_configs()],
)
def test_sweep(batch, Ci, H, W, Co, k, s, p, d, g):
    _run_conv2d(batch, Ci, H, W, Co, (k, k), (s, s), (p, p), (d, d), g, True, torch.float32)


# ###################################################################
#  NHWC LAYOUT TESTS
# ###################################################################

# ---------------------------------------------------------------------------
# NHWC helpers
# ---------------------------------------------------------------------------

def _ref_conv2d_for_nhwc(x_nhwc, weight, bias, stride, padding, dilation, groups):
    """Run reference conv2d in NCHW float32 and return NHWC result."""
    x_nchw = x_nhwc.float().permute(0, 3, 1, 2)  # NHWC -> NCHW
    ref_nchw = F.conv2d(
        x_nchw, weight.float(),
        bias.float() if bias is not None else None,
        stride=stride, padding=padding, dilation=dilation, groups=groups,
    )
    return ref_nchw.permute(0, 2, 3, 1)  # NCHW -> NHWC


def _run_nhwc(
    batch, in_channels, in_h, in_w,
    out_channels, kernel_size, stride, padding, dilation, groups,
    bias, dtype, atol=None, rtol=None,
):
    """Shared NHWC test driver."""
    torch.cuda.empty_cache()
    if atol is None or rtol is None:
        _a, _r = _tols(dtype)
        atol = atol or _a
        rtol = rtol or _r

    # Create NHWC input
    x = torch.randn(batch, in_h, in_w, in_channels, dtype=dtype, device="cuda")
    w = torch.randn(
        out_channels, in_channels // groups, kernel_size[0], kernel_size[1],
        dtype=dtype, device="cuda",
    )
    b = torch.randn(out_channels, dtype=dtype, device="cuda") if bias else None

    ref = _ref_conv2d_for_nhwc(x, w, b, stride, padding, dilation, groups).to(dtype)
    out = triton_conv2d(x, w, b, stride=stride, padding=padding,
                        dilation=dilation, groups=groups, layout="nhwc")

    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


# ===================================================================
#  NHWC Section 1: dtype and bias
# ===================================================================

@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [True, False])
class TestNHWCDtypeBias:
    def test_3x3(self, dtype, bias):
        _run_nhwc(2, 16, 32, 32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1, bias, dtype)

    def test_1x1(self, dtype, bias):
        _run_nhwc(2, 64, 16, 16, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias, dtype)

    def test_5x5_stride2(self, dtype, bias):
        _run_nhwc(2, 16, 32, 32, 32, (5, 5), (2, 2), (2, 2), (1, 1), 1, bias, dtype)

    def test_depthwise(self, dtype, bias):
        _run_nhwc(2, 32, 16, 16, 32, (3, 3), (1, 1), (1, 1), (1, 1), 32, bias, dtype)


# ===================================================================
#  NHWC Section 2: Kernel sizes
# ===================================================================

@pytest.mark.parametrize("k", [1, 2, 3, 5, 7, 9])
def test_nhwc_square_kernel(k):
    h = max(k + 4, 16)
    _run_nhwc(1, 8, h, h, 16, (k, k), (1, 1), (0, 0), (1, 1), 1, True, torch.float32)


@pytest.mark.parametrize("kh,kw", [(1, 3), (3, 1), (3, 5), (5, 3), (1, 7), (7, 1)])
def test_nhwc_nonsquare_kernel(kh, kw):
    h, w = max(kh + 4, 16), max(kw + 4, 16)
    _run_nhwc(1, 8, h, w, 16, (kh, kw), (1, 1), (0, 0), (1, 1), 1, True, torch.float32)


# ===================================================================
#  NHWC Section 3: Strides
# ===================================================================

@pytest.mark.parametrize("sh,sw", [(1, 1), (2, 2), (3, 3), (1, 2), (2, 1), (2, 3)])
def test_nhwc_strides(sh, sw):
    _run_nhwc(2, 16, 32, 32, 16, (3, 3), (sh, sw), (1, 1), (1, 1), 1, True, torch.float32)


# ===================================================================
#  NHWC Section 4: Padding
# ===================================================================

@pytest.mark.parametrize("ph,pw", [(0, 0), (1, 1), (2, 2), (0, 1), (3, 1), (5, 5)])
def test_nhwc_padding(ph, pw):
    _run_nhwc(1, 8, 16, 16, 8, (3, 3), (1, 1), (ph, pw), (1, 1), 1, True, torch.float32)


# ===================================================================
#  NHWC Section 5: Dilation
# ===================================================================

@pytest.mark.parametrize("dh,dw", [(1, 1), (2, 2), (3, 3), (1, 2), (2, 1)])
def test_nhwc_dilation(dh, dw):
    _run_nhwc(1, 8, 32, 32, 8, (3, 3), (1, 1), (0, 0), (dh, dw), 1, True, torch.float32)


@pytest.mark.parametrize("d,p", [(2, 2), (3, 3), (2, 4)])
def test_nhwc_dilation_with_padding(d, p):
    _run_nhwc(1, 8, 32, 32, 8, (3, 3), (1, 1), (p, p), (d, d), 1, True, torch.float32)


# ===================================================================
#  NHWC Section 6: Groups
# ===================================================================

@pytest.mark.parametrize("in_c,out_c,groups", [
    (16, 16, 1), (16, 16, 2), (16, 16, 4), (16, 16, 8), (16, 16, 16),
    (32, 64, 8), (16, 32, 16), (64, 64, 64),
    (8, 32, 8),   # depthwise with multiplier 4
    (32, 32, 32),  # depthwise
])
def test_nhwc_groups(in_c, out_c, groups):
    _run_nhwc(2, in_c, 16, 16, out_c, (3, 3), (1, 1), (1, 1), (1, 1), groups, True, torch.float32)


@pytest.mark.parametrize("groups,stride,dilation", [
    (4, (2, 2), (1, 1)), (4, (1, 1), (2, 2)),
    (16, (2, 2), (1, 1)), (32, (1, 1), (2, 2)),
])
def test_nhwc_groups_with_stride_dilation(groups, stride, dilation):
    _run_nhwc(2, 32, 32, 32, 32, (3, 3), stride, (1, 1), dilation, groups, True, torch.float32)


# ===================================================================
#  NHWC Section 7: Channel counts
# ===================================================================

@pytest.mark.parametrize("in_c,out_c", [
    (1, 1), (1, 8), (3, 16), (3, 64),
    (16, 32), (64, 128), (128, 256), (256, 512),
    (64, 32), (256, 128),
    (5, 13), (7, 11),  # odd counts
])
def test_nhwc_channels(in_c, out_c):
    h = 8 if in_c >= 256 else 16
    _run_nhwc(1, in_c, h, h, out_c, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


# ===================================================================
#  NHWC Section 8: Spatial sizes
# ===================================================================

@pytest.mark.parametrize("h,w", [
    (3, 3), (7, 7), (14, 14), (28, 28), (56, 56), (112, 112),
    (7, 13), (16, 32), (1, 16), (16, 1),
])
def test_nhwc_spatial(h, w):
    pad = (1, 1) if min(h, w) >= 3 else (0, 0)
    k = (3, 3) if min(h, w) >= 3 else (1, 1)
    _run_nhwc(1, 8, h, w, 8, k, (1, 1), pad, (1, 1), 1, True, torch.float32)


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32])
def test_nhwc_batch_sizes(batch):
    _run_nhwc(batch, 16, 16, 16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


# ===================================================================
#  NHWC Section 9: Unbatched (3-D) input
# ===================================================================

@pytest.mark.parametrize("in_c,out_c,k,p", [
    (8, 16, 3, 1), (3, 32, 5, 2), (16, 16, 1, 0),
])
def test_nhwc_unbatched(in_c, out_c, k, p):
    """3-D NHWC input (H, W, C) should produce 3-D output."""
    x = torch.randn(16, 16, in_c, dtype=torch.float32, device="cuda")
    w = torch.randn(out_c, in_c, k, k, dtype=torch.float32, device="cuda")
    b = torch.randn(out_c, dtype=torch.float32, device="cuda")

    # Reference via NCHW
    x_nchw = x.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    ref_nchw = F.conv2d(x_nchw, w, b, padding=p).squeeze(0)  # (C, Ho, Wo)
    ref_nhwc = ref_nchw.permute(1, 2, 0)  # (Ho, Wo, C)

    out = triton_conv2d(x, w, b, padding=p, layout="nhwc")
    assert out.dim() == 3, f"Expected 3-D, got {out.dim()}-D"
    torch.testing.assert_close(out, ref_nhwc, atol=5e-4, rtol=1e-4)


# ===================================================================
#  NHWC Section 10: conv2d_nhwc convenience function
# ===================================================================

def test_conv2d_nhwc_shorthand():
    """conv2d_nhwc should produce identical results to conv2d(..., layout='nhwc')."""
    x = torch.randn(2, 16, 16, 32, dtype=torch.float32, device="cuda")
    w = torch.randn(64, 32, 3, 3, dtype=torch.float32, device="cuda")
    b = torch.randn(64, dtype=torch.float32, device="cuda")

    out1 = triton_conv2d(x, w, b, padding=1, layout="nhwc")
    out2 = conv2d_nhwc(x, w, b, padding=1)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)


# ===================================================================
#  NHWC Section 11: Cross-layout consistency
# ===================================================================

@pytest.mark.parametrize("batch,in_c,h,out_c,k,s,p,d,g", [
    (1, 16, 16, 16, 3, 1, 1, 1, 1),
    (2, 32, 32, 32, 3, 2, 1, 1, 1),
    (1, 64, 14, 128, 1, 1, 0, 1, 1),
    (2, 16, 32, 16, 3, 1, 0, 2, 1),
    (1, 32, 16, 32, 3, 1, 1, 1, 32),
    (2, 16, 32, 32, 5, 2, 2, 1, 4),
    (4, 3, 56, 64, 7, 2, 3, 1, 1),
])
def test_cross_layout_consistency(batch, in_c, h, out_c, k, s, p, d, g):
    """NCHW and NHWC Triton kernels must agree after permutation."""
    x_nchw = torch.randn(batch, in_c, h, h, dtype=torch.float32, device="cuda")
    w = torch.randn(out_c, in_c // g, k, k, dtype=torch.float32, device="cuda")
    b = torch.randn(out_c, dtype=torch.float32, device="cuda")

    out_nchw = triton_conv2d(x_nchw, w, b, stride=s, padding=p, dilation=d,
                             groups=g, layout="nchw")

    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
    out_nhwc = triton_conv2d(x_nhwc, w, b, stride=s, padding=p, dilation=d,
                             groups=g, layout="nhwc")

    # Convert NHWC output to NCHW for comparison
    out_nhwc_as_nchw = out_nhwc.permute(0, 3, 1, 2)
    torch.testing.assert_close(out_nchw, out_nhwc_as_nchw, atol=5e-4, rtol=1e-4)


# ===================================================================
#  NHWC Section 12: Real architectures
# ===================================================================

class TestNHWCResNet:
    def test_stem(self):
        _run_nhwc(1, 3, 224, 224, 64, (7, 7), (2, 2), (3, 3), (1, 1), 1, True, torch.float32)

    def test_block1(self):
        _run_nhwc(1, 64, 56, 56, 64, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)

    def test_block2_downsample(self):
        _run_nhwc(1, 128, 28, 28, 128, (3, 3), (2, 2), (1, 1), (1, 1), 1, True, torch.float32)

    def test_block3(self):
        _run_nhwc(1, 256, 14, 14, 256, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)

    def test_block4(self):
        _run_nhwc(1, 512, 7, 7, 512, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)

    def test_pointwise(self):
        _run_nhwc(1, 64, 56, 56, 256, (1, 1), (1, 1), (0, 0), (1, 1), 1, True, torch.float32)

    def test_batched(self):
        _run_nhwc(8, 64, 56, 56, 64, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


class TestNHWCMobileNet:
    def test_dw_112(self):
        _run_nhwc(1, 32, 112, 112, 32, (3, 3), (1, 1), (1, 1), (1, 1), 32, True, torch.float32)

    def test_pw_112(self):
        _run_nhwc(1, 32, 112, 112, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, True, torch.float32)

    def test_dw_stride2(self):
        _run_nhwc(1, 64, 56, 56, 64, (3, 3), (2, 2), (1, 1), (1, 1), 64, True, torch.float32)

    def test_dw_7(self):
        _run_nhwc(1, 512, 7, 7, 512, (3, 3), (1, 1), (1, 1), (1, 1), 512, True, torch.float32)


# ===================================================================
#  NHWC Section 13: Stride+dilation+padding+kernel combos
# ===================================================================

@pytest.mark.parametrize("stride,dilation,padding,kernel_size", [
    ((2, 2), (1, 1), (1, 1), (3, 3)),
    ((1, 1), (2, 2), (2, 2), (3, 3)),
    ((2, 2), (2, 2), (2, 2), (3, 3)),
    ((2, 1), (1, 2), (0, 1), (3, 3)),
    ((2, 2), (1, 1), (0, 0), (5, 5)),
    ((1, 1), (3, 3), (3, 3), (3, 3)),
    ((2, 2), (1, 1), (3, 3), (7, 7)),
    ((1, 1), (1, 1), (0, 0), (1, 1)),
    ((4, 4), (1, 1), (0, 0), (3, 3)),
])
def test_nhwc_param_combos(stride, dilation, padding, kernel_size):
    _run_nhwc(2, 16, 32, 32, 16, kernel_size, stride, padding, dilation, 1, True, torch.float32)


# ===================================================================
#  NHWC Section 14: Full combo (groups + stride + bias)
# ===================================================================

@pytest.mark.parametrize("groups,bias", [(1, True), (2, True), (4, False), (16, True)])
@pytest.mark.parametrize("stride,padding", [((1, 1), (1, 1)), ((2, 2), (0, 0))])
def test_nhwc_full_combo(groups, bias, stride, padding):
    _run_nhwc(2, 16, 32, 32, 32, (3, 3), stride, padding, (1, 1), groups, bias, torch.float32)


# ===================================================================
#  NHWC Section 15: Half-precision
# ===================================================================

@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("in_c,out_c,h,k", [
    (3, 64, 32, 7), (64, 64, 16, 3), (128, 128, 8, 3), (32, 32, 16, 3),
])
def test_nhwc_half_precision(dtype, in_c, out_c, h, k):
    p = k // 2
    _run_nhwc(2, in_c, h, h, out_c, (k, k), (1, 1), (p, p), (1, 1), 1, True, dtype)


# ===================================================================
#  NHWC Section 16: Sweep (auto-generated combos)
# ===================================================================

@pytest.mark.parametrize(
    "batch,Ci,H,W,Co,k,s,p,d,g",
    _sweep_configs(),
    ids=[f"nhwc_k{c[5]}s{c[6]}d{c[8]}g{c[9]}" for c in _sweep_configs()],
)
def test_nhwc_sweep(batch, Ci, H, W, Co, k, s, p, d, g):
    _run_nhwc(batch, Ci, H, W, Co, (k, k), (s, s), (p, p), (d, d), g, True, torch.float32)
