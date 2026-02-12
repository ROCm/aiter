# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Comprehensive tests for the Triton Conv2d kernels (NCHW & NHWC).

Every test compares the Triton output against ``torch.nn.functional.conv2d``
(computed in float32) as the ground-truth reference.

Coverage axes:
  layout       NCHW, NHWC, cross-layout consistency
  kernel_size  1x1 .. 11x11, square & non-square
  stride       1 .. 4, symmetric & asymmetric
  padding      0 .. large, symmetric & asymmetric
  dilation     1 .. 4, symmetric & asymmetric
  groups       1, 2, 4, 8, 16, depthwise, depthwise x multiplier
  bias         True / False
  batch        1 .. 64
  channels     1 .. 512, odd counts
  spatial      1x1 .. 224x224, square & non-square
  dtypes       float32, float16, bfloat16
  unbatched    3-D input
  scalar args  int vs tuple
  architectures  ResNet, VGG, MobileNet, EfficientNet
"""

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.conv2d import conv2d as triton_conv2d, conv2d_nhwc


# ── Helpers ──────────────────────────────────────────────────────────────────

def _tols(dtype):
    """Return (atol, rtol) appropriate for *dtype*."""
    if dtype == torch.float32:
        return 5e-4, 1e-4
    if dtype == torch.float16:
        return 1e-2, 1e-2
    return 2e-2, 2e-2  # bfloat16


def _run(
    batch, in_c, h, w, out_c, ks, stride, padding, dilation, groups,
    bias, dtype, layout="nchw", atol=None, rtol=None,
):
    """Unified test driver for NCHW and NHWC layouts."""
    torch.cuda.empty_cache()
    _a, _r = _tols(dtype)
    atol = atol if atol is not None else _a
    rtol = rtol if rtol is not None else _r

    x = (torch.randn(batch, in_c, h, w, dtype=dtype, device="cuda")
         if layout == "nchw"
         else torch.randn(batch, h, w, in_c, dtype=dtype, device="cuda"))
    wt = torch.randn(out_c, in_c // groups, ks[0], ks[1], dtype=dtype, device="cuda")
    b = torch.randn(out_c, dtype=dtype, device="cuda") if bias else None

    # Reference: always NCHW float32
    x_ref = x.float() if layout == "nchw" else x.float().permute(0, 3, 1, 2)
    ref = F.conv2d(
        x_ref, wt.float(), b.float() if b is not None else None,
        stride=stride, padding=padding, dilation=dilation, groups=groups,
    )
    if layout == "nhwc":
        ref = ref.permute(0, 2, 3, 1)
    ref = ref.to(dtype)

    out = triton_conv2d(
        x, wt, b, stride=stride, padding=padding,
        dilation=dilation, groups=groups, layout=layout,
    )
    assert out.shape == ref.shape, f"Shape mismatch: {out.shape} vs {ref.shape}"
    torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)


def _sweep_configs():
    """Generate (batch, Ci, H, W, Co, k, s, p, d, g) combos."""
    cfgs = []
    for k in [1, 3, 5]:
        for s in [1, 2]:
            for d in [1, 2]:
                p = d * (k // 2)
                for g in [1, 4]:
                    H = max(16, k + 2 * d * (k - 1))
                    cfgs.append((1, 16, H, H, 16, k, s, p, d, g))
    return cfgs


# ── 1. Dtype x Bias fundamentals (both layouts) ─────────────────────────────

@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [True, False])
class TestDtypeBias:
    def test_3x3(self, layout, dtype, bias):
        _run(2, 16, 32, 32, 32, (3, 3), (1, 1), (1, 1), (1, 1), 1, bias, dtype, layout)

    def test_1x1(self, layout, dtype, bias):
        _run(2, 64, 16, 16, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, bias, dtype, layout)

    def test_5x5_stride2(self, layout, dtype, bias):
        _run(2, 16, 32, 32, 32, (5, 5), (2, 2), (2, 2), (1, 1), 1, bias, dtype, layout)

    def test_depthwise(self, layout, dtype, bias):
        _run(2, 32, 16, 16, 32, (3, 3), (1, 1), (1, 1), (1, 1), 32, bias, dtype, layout)


# ── 2. Kernel sizes ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6, 7, 9, 11])
def test_nchw_square_kernel(k):
    _run(1, 8, max(k + 4, 16), max(k + 4, 16), 16,
         (k, k), (1, 1), (0, 0), (1, 1), 1, True, torch.float32)


@pytest.mark.parametrize("kh,kw", [
    (1, 3), (3, 1), (1, 5), (5, 1), (1, 7), (7, 1),
    (3, 5), (5, 3), (3, 7), (7, 3), (2, 4), (4, 2),
    (1, 11), (11, 1), (2, 3), (3, 2),
])
def test_nchw_nonsquare_kernel(kh, kw):
    _run(1, 8, max(kh + 4, 16), max(kw + 4, 16), 16,
         (kh, kw), (1, 1), (0, 0), (1, 1), 1, True, torch.float32)


@pytest.mark.parametrize("k", [1, 2, 3, 5, 7, 9])
def test_nhwc_square_kernel(k):
    _run(1, 8, max(k + 4, 16), max(k + 4, 16), 16,
         (k, k), (1, 1), (0, 0), (1, 1), 1, True, torch.float32, "nhwc")


@pytest.mark.parametrize("kh,kw", [(1, 3), (3, 1), (3, 5), (5, 3), (1, 7), (7, 1)])
def test_nhwc_nonsquare_kernel(kh, kw):
    _run(1, 8, max(kh + 4, 16), max(kw + 4, 16), 16,
         (kh, kw), (1, 1), (0, 0), (1, 1), 1, True, torch.float32, "nhwc")


# ── 3. Strides ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("sh,sw", [
    (1, 1), (2, 2), (3, 3), (4, 4),                                   # square
    (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (1, 4), (4, 1),  # asymmetric
])
def test_nchw_strides(sh, sw):
    _run(2, 16, 32, 32, 16, (3, 3), (sh, sw), (1, 1), (1, 1), 1, True, torch.float32)


@pytest.mark.parametrize("sh,sw", [(1, 1), (2, 2), (3, 3), (1, 2), (2, 1), (2, 3)])
def test_nhwc_strides(sh, sw):
    _run(2, 16, 32, 32, 16, (3, 3), (sh, sw), (1, 1), (1, 1), 1, True, torch.float32, "nhwc")


# ── 4. Padding ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("p", [0, 1, 2, 3, 4, 5])
def test_nchw_square_padding(p):
    _run(1, 8, 16, 16, 8, (3, 3), (1, 1), (p, p), (1, 1), 1, False, torch.float32)


@pytest.mark.parametrize("ph,pw", [
    (0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1), (2, 4), (4, 2),
])
def test_nchw_asymmetric_padding(ph, pw):
    _run(1, 8, 16, 16, 8, (3, 3), (1, 1), (ph, pw), (1, 1), 1, True, torch.float32)


@pytest.mark.parametrize("p", [3, 5, 7])
def test_nchw_large_padding(p):
    _run(1, 4, 8, 8, 4, (3, 3), (1, 1), (p, p), (1, 1), 1, True, torch.float32)


@pytest.mark.parametrize("ph,pw", [(0, 0), (1, 1), (2, 2), (0, 1), (3, 1), (5, 5)])
def test_nhwc_padding(ph, pw):
    _run(1, 8, 16, 16, 8, (3, 3), (1, 1), (ph, pw), (1, 1), 1, True, torch.float32, "nhwc")


# ── 5. Dilation ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("dh,dw", [
    (1, 1), (2, 2), (3, 3), (4, 4),                        # square
    (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2),       # asymmetric
])
def test_nchw_dilation(dh, dw):
    _run(1, 8, 32, 32, 8, (3, 3), (1, 1), (0, 0), (dh, dw), 1, True, torch.float32)


@pytest.mark.parametrize("d,p", [(2, 2), (3, 3), (2, 4), (4, 4), (3, 6)])
def test_nchw_dilation_with_padding(d, p):
    _run(1, 8, 32, 32, 8, (3, 3), (1, 1), (p, p), (d, d), 1, True, torch.float32)


@pytest.mark.parametrize("dh,dw", [(1, 1), (2, 2), (3, 3), (1, 2), (2, 1)])
def test_nhwc_dilation(dh, dw):
    _run(1, 8, 32, 32, 8, (3, 3), (1, 1), (0, 0), (dh, dw), 1, True, torch.float32, "nhwc")


@pytest.mark.parametrize("d,p", [(2, 2), (3, 3), (2, 4)])
def test_nhwc_dilation_with_padding(d, p):
    _run(1, 8, 32, 32, 8, (3, 3), (1, 1), (p, p), (d, d), 1, True, torch.float32, "nhwc")


# ── 6. Groups ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("in_c,out_c,groups", [
    (16, 16, 1), (16, 16, 2), (16, 16, 4), (16, 16, 8), (16, 16, 16),
    (32, 32, 1), (32, 32, 2), (32, 32, 4), (32, 32, 8), (32, 32, 16), (32, 32, 32),
    (64, 64, 1), (64, 64, 4), (64, 64, 16), (64, 64, 64),
    (16, 32, 2), (16, 32, 4), (16, 32, 8), (16, 64, 4), (32, 64, 8), (64, 128, 16),
    (16, 32, 16), (16, 48, 16), (32, 64, 32), (8, 32, 8),
])
def test_nchw_groups(in_c, out_c, groups):
    _run(2, in_c, 16, 16, out_c, (3, 3), (1, 1), (1, 1), (1, 1), groups, True, torch.float32)


@pytest.mark.parametrize("groups,stride,dilation", [
    (4, (2, 2), (1, 1)), (4, (1, 1), (2, 2)), (4, (2, 2), (2, 2)),
    (16, (2, 2), (1, 1)), (16, (1, 1), (2, 2)), (32, (2, 2), (1, 1)),
])
def test_nchw_groups_with_stride_dilation(groups, stride, dilation):
    _run(2, 32, 32, 32, 32, (3, 3), stride, (1, 1), dilation, groups, True, torch.float32)


@pytest.mark.parametrize("in_c,out_c,groups", [
    (16, 16, 1), (16, 16, 2), (16, 16, 4), (16, 16, 8), (16, 16, 16),
    (32, 64, 8), (16, 32, 16), (64, 64, 64), (8, 32, 8), (32, 32, 32),
])
def test_nhwc_groups(in_c, out_c, groups):
    _run(2, in_c, 16, 16, out_c, (3, 3), (1, 1), (1, 1), (1, 1), groups, True, torch.float32, "nhwc")


@pytest.mark.parametrize("groups,stride,dilation", [
    (4, (2, 2), (1, 1)), (4, (1, 1), (2, 2)),
    (16, (2, 2), (1, 1)), (32, (1, 1), (2, 2)),
])
def test_nhwc_groups_with_stride_dilation(groups, stride, dilation):
    _run(2, 32, 32, 32, 32, (3, 3), stride, (1, 1), dilation, groups, True, torch.float32, "nhwc")


# ── 7. Channel counts ──────────────────────────────────────────────────────

@pytest.mark.parametrize("in_c,out_c", [
    (1, 1), (1, 2), (1, 4), (1, 8), (1, 16),
    (2, 1), (2, 2), (2, 4),
    (3, 1), (3, 8), (3, 16), (3, 32), (3, 64),
    (4, 4), (4, 8), (4, 16),
])
def test_nchw_small_channels(in_c, out_c):
    _run(1, in_c, 16, 16, out_c, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


@pytest.mark.parametrize("in_c,out_c", [
    (16, 32), (32, 64), (64, 128), (128, 256), (256, 512),
    (32, 16), (64, 32), (128, 64), (256, 128),
    (64, 64), (128, 128), (256, 256), (512, 512),
])
def test_nchw_medium_large_channels(in_c, out_c):
    h = 8 if in_c >= 256 else 14
    _run(1, in_c, h, h, out_c, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


@pytest.mark.parametrize("in_c,out_c", [
    (1, 1), (1, 8), (3, 16), (3, 64),
    (16, 32), (64, 128), (128, 256), (256, 512),
    (64, 32), (256, 128), (5, 13), (7, 11),
])
def test_nhwc_channels(in_c, out_c):
    h = 8 if in_c >= 256 else 16
    _run(1, in_c, h, h, out_c, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32, "nhwc")


# ── 8. Spatial sizes ────────────────────────────────────────────────────────

@pytest.mark.parametrize("h,w,k", [
    (1, 1, 1), (2, 2, 1), (3, 3, 1), (3, 3, 3), (4, 4, 3),
    (5, 5, 3), (5, 5, 5), (6, 6, 3), (7, 7, 3), (7, 7, 5), (7, 7, 7),
])
def test_nchw_small_spatial(h, w, k):
    _run(1, 8, h, w, 8, (k, k), (1, 1), (k // 2, k // 2), (1, 1), 1, True, torch.float32)


@pytest.mark.parametrize("h,w", [(8, 8), (14, 14), (16, 16), (28, 28), (32, 32), (56, 56)])
def test_nchw_medium_spatial(h, w):
    _run(1, 16, h, w, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


@pytest.mark.parametrize("h,w", [(112, 112), (128, 128), (224, 224)])
def test_nchw_large_spatial(h, w):
    _run(1, 3, h, w, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


@pytest.mark.parametrize("h,w", [
    (7, 13), (13, 7), (16, 32), (32, 16), (28, 56), (56, 28),
    (1, 16), (16, 1), (3, 32), (32, 3), (14, 7), (7, 14), (100, 50), (50, 100),
])
def test_nchw_nonsquare_spatial(h, w):
    _run(1, 8, h, w, 8, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


@pytest.mark.parametrize("h,w", [
    (3, 3), (7, 7), (14, 14), (28, 28), (56, 56), (112, 112),
    (7, 13), (16, 32), (1, 16), (16, 1),
])
def test_nhwc_spatial(h, w):
    pad = (1, 1) if min(h, w) >= 3 else (0, 0)
    k = (3, 3) if min(h, w) >= 3 else (1, 1)
    _run(1, 8, h, w, 8, k, (1, 1), pad, (1, 1), 1, True, torch.float32, "nhwc")


# ── 9. Batch sizes ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("batch", [1, 2, 3, 4, 7, 8, 15, 16, 32, 64])
def test_nchw_batch_sizes(batch):
    _run(batch, 16, 16, 16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


@pytest.mark.parametrize("batch", [1, 2, 4, 8, 16, 32])
def test_nhwc_batch_sizes(batch):
    _run(batch, 16, 16, 16, 16, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32, "nhwc")


# ── 10. Unbatched (3-D) input ───────────────────────────────────────────────

@pytest.mark.parametrize("in_c,out_c,k,p", [
    (8, 16, 3, 1), (3, 32, 5, 2), (16, 16, 1, 0), (1, 8, 3, 1),
])
def test_nchw_unbatched(in_c, out_c, k, p):
    """3-D NCHW input (C, H, W) -> 3-D output."""
    x = torch.randn(in_c, 16, 16, dtype=torch.float32, device="cuda")
    w = torch.randn(out_c, in_c, k, k, dtype=torch.float32, device="cuda")
    b = torch.randn(out_c, dtype=torch.float32, device="cuda")
    ref = F.conv2d(x.unsqueeze(0).float(), w.float(), b.float(), padding=p).squeeze(0)
    out = triton_conv2d(x, w, b, padding=p)
    assert out.dim() == 3
    torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-5)


@pytest.mark.parametrize("in_c,out_c,k,p", [
    (8, 16, 3, 1), (3, 32, 5, 2), (16, 16, 1, 0),
])
def test_nhwc_unbatched(in_c, out_c, k, p):
    """3-D NHWC input (H, W, C) -> 3-D output."""
    x = torch.randn(16, 16, in_c, dtype=torch.float32, device="cuda")
    w = torch.randn(out_c, in_c, k, k, dtype=torch.float32, device="cuda")
    b = torch.randn(out_c, dtype=torch.float32, device="cuda")
    x_nchw = x.permute(2, 0, 1).unsqueeze(0)
    ref = F.conv2d(x_nchw, w, b, padding=p).squeeze(0).permute(1, 2, 0)
    out = triton_conv2d(x, w, b, padding=p, layout="nhwc")
    assert out.dim() == 3
    torch.testing.assert_close(out, ref, atol=5e-4, rtol=1e-4)


# ── 11. Scalar int arguments ────────────────────────────────────────────────

class TestScalarIntArgs:
    """Verify that plain-int stride / padding / dilation work (not only tuples)."""

    def test_stride(self):
        x = torch.randn(1, 8, 16, 16, dtype=torch.float32, device="cuda")
        w = torch.randn(8, 8, 3, 3, dtype=torch.float32, device="cuda")
        torch.testing.assert_close(
            triton_conv2d(x, w, stride=2, padding=1),
            F.conv2d(x, w, stride=2, padding=1), atol=1e-4, rtol=1e-5)

    def test_padding(self):
        x = torch.randn(1, 8, 16, 16, dtype=torch.float32, device="cuda")
        w = torch.randn(8, 8, 3, 3, dtype=torch.float32, device="cuda")
        torch.testing.assert_close(
            triton_conv2d(x, w, padding=1),
            F.conv2d(x, w, padding=1), atol=1e-4, rtol=1e-5)

    def test_dilation(self):
        x = torch.randn(1, 8, 32, 32, dtype=torch.float32, device="cuda")
        w = torch.randn(8, 8, 3, 3, dtype=torch.float32, device="cuda")
        torch.testing.assert_close(
            triton_conv2d(x, w, dilation=2),
            F.conv2d(x, w, dilation=2), atol=1e-4, rtol=1e-5)

    def test_all_int(self):
        x = torch.randn(2, 16, 32, 32, dtype=torch.float32, device="cuda")
        w = torch.randn(32, 16, 3, 3, dtype=torch.float32, device="cuda")
        b = torch.randn(32, dtype=torch.float32, device="cuda")
        torch.testing.assert_close(
            triton_conv2d(x, w, b, stride=2, padding=1, dilation=1, groups=1),
            F.conv2d(x, w, b, stride=2, padding=1, dilation=1, groups=1),
            atol=1e-4, rtol=1e-5)


# ── 12. Edge cases ──────────────────────────────────────────────────────────

_EDGE_CASES = [
    # (batch, ic, h, w, oc, ks, stride, pad, dilation, groups, bias, id)
    (1, 8, 1, 1, 8, (1, 1), (1, 1), (0, 0), (1, 1), 1, True, "1x1_input"),
    (1, 4, 1, 1, 4, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, "1x1_input_3x3_pad"),
    (1, 1, 16, 16, 1, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, "single_channel"),
    (1, 8, 7, 7, 8, (3, 3), (5, 5), (0, 0), (1, 1), 1, True, "1x1_output_stride"),
    (1, 4, 3, 3, 4, (3, 3), (1, 1), (0, 0), (1, 1), 1, True, "kernel_eq_3"),
    (1, 4, 5, 5, 4, (5, 5), (1, 1), (0, 0), (1, 1), 1, True, "kernel_eq_5"),
    (1, 4, 7, 7, 4, (7, 7), (1, 1), (0, 0), (1, 1), 1, True, "kernel_eq_7"),
    (1, 4, 8, 8, 256, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, "wide_output"),
    (1, 256, 8, 8, 4, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, "narrow_output"),
    (1, 3, 16, 16, 7, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, "odd_3to7"),
    (1, 5, 16, 16, 13, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, "odd_5to13"),
    (1, 7, 16, 16, 11, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, "odd_7to11"),
    (1, 8, 16, 16, 8, (2, 2), (1, 1), (0, 0), (1, 1), 1, True, "even_k2"),
    (1, 8, 16, 16, 8, (4, 4), (1, 1), (0, 0), (1, 1), 1, True, "even_k4"),
    (1, 8, 16, 16, 8, (6, 6), (1, 1), (0, 0), (1, 1), 1, True, "even_k6"),
]


@pytest.mark.parametrize(
    "batch,ic,h,w,oc,ks,s,p,d,g,bias,label", _EDGE_CASES,
    ids=[c[-1] for c in _EDGE_CASES],
)
def test_edge_case(batch, ic, h, w, oc, ks, s, p, d, g, bias, label):
    _run(batch, ic, h, w, oc, ks, s, p, d, g, bias, torch.float32)


# ── 13. Multi-parameter combos (both layouts) ───────────────────────────────

# Shared between NCHW and NHWC
_SHARED_PARAM_COMBOS = [
    ((2, 2), (1, 1), (1, 1), (3, 3)),
    ((1, 1), (2, 2), (2, 2), (3, 3)),
    ((2, 2), (2, 2), (2, 2), (3, 3)),
    ((2, 1), (1, 2), (0, 1), (3, 3)),
    ((2, 2), (1, 1), (0, 0), (5, 5)),
    ((1, 1), (3, 3), (3, 3), (3, 3)),
    ((2, 2), (1, 1), (3, 3), (7, 7)),
    ((1, 1), (1, 1), (0, 0), (1, 1)),
    ((4, 4), (1, 1), (0, 0), (3, 3)),
]

# NCHW-only extras
_NCHW_PARAM_COMBOS_EXTRA = [
    ((1, 2), (2, 1), (1, 0), (3, 3)),
    ((3, 3), (1, 1), (1, 1), (3, 3)),
    ((2, 2), (2, 2), (4, 4), (5, 5)),
    ((2, 3), (1, 2), (1, 2), (3, 5)),
]


@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@pytest.mark.parametrize("stride,dilation,padding,kernel_size", _SHARED_PARAM_COMBOS)
def test_param_combos(layout, stride, dilation, padding, kernel_size):
    _run(2, 16, 32, 32, 16, kernel_size, stride, padding, dilation, 1, True, torch.float32, layout)


@pytest.mark.parametrize("stride,dilation,padding,kernel_size", _NCHW_PARAM_COMBOS_EXTRA)
def test_nchw_param_combos_extra(stride, dilation, padding, kernel_size):
    _run(2, 16, 32, 32, 16, kernel_size, stride, padding, dilation, 1, True, torch.float32)


# ── 14. Full combo: groups x stride x bias (both layouts) ───────────────────

@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@pytest.mark.parametrize("groups,bias", [(1, True), (2, True), (4, False), (16, True)])
@pytest.mark.parametrize("stride,padding", [((1, 1), (1, 1)), ((2, 2), (0, 0))])
def test_full_combo(layout, groups, bias, stride, padding):
    _run(2, 16, 32, 32, 32, (3, 3), stride, padding, (1, 1), groups, bias, torch.float32, layout)


# ── 15. Architecture shapes (data-driven) ───────────────────────────────────

# (N, Ci, H, W, Co, ks, stride, pad, dilation, groups, layouts, id)
_ARCH_SHAPES = [
    # ResNet-50
    (1, 3, 224, 224, 64, (7, 7), (2, 2), (3, 3), (1, 1), 1, "both", "resnet_stem"),
    (1, 64, 56, 56, 64, (3, 3), (1, 1), (1, 1), (1, 1), 1, "both", "resnet_block1"),
    (1, 64, 56, 56, 256, (1, 1), (1, 1), (0, 0), (1, 1), 1, "both", "resnet_expand"),
    (1, 256, 56, 56, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, "nchw", "resnet_reduce"),
    (1, 128, 28, 28, 128, (3, 3), (2, 2), (1, 1), (1, 1), 1, "both", "resnet_block2_ds"),
    (1, 256, 14, 14, 256, (3, 3), (1, 1), (1, 1), (1, 1), 1, "both", "resnet_block3"),
    (1, 512, 7, 7, 512, (3, 3), (1, 1), (1, 1), (1, 1), 1, "both", "resnet_block4"),
    (8, 64, 56, 56, 64, (3, 3), (1, 1), (1, 1), (1, 1), 1, "both", "resnet_batched"),
    # VGG-16
    (1, 3, 224, 224, 64, (3, 3), (1, 1), (1, 1), (1, 1), 1, "nchw", "vgg_conv1"),
    (1, 64, 112, 112, 128, (3, 3), (1, 1), (1, 1), (1, 1), 1, "nchw", "vgg_conv2"),
    (1, 128, 56, 56, 256, (3, 3), (1, 1), (1, 1), (1, 1), 1, "nchw", "vgg_conv3"),
    (1, 256, 28, 28, 512, (3, 3), (1, 1), (1, 1), (1, 1), 1, "nchw", "vgg_conv4"),
    (1, 512, 14, 14, 512, (3, 3), (1, 1), (1, 1), (1, 1), 1, "nchw", "vgg_conv5"),
    # MobileNet (depthwise separable)
    (1, 32, 112, 112, 32, (3, 3), (1, 1), (1, 1), (1, 1), 32, "both", "mobile_dw_112"),
    (1, 32, 112, 112, 64, (1, 1), (1, 1), (0, 0), (1, 1), 1, "both", "mobile_pw_112"),
    (1, 64, 56, 56, 64, (3, 3), (2, 2), (1, 1), (1, 1), 64, "both", "mobile_dw_s2"),
    (1, 64, 56, 56, 128, (1, 1), (1, 1), (0, 0), (1, 1), 1, "nchw", "mobile_pw_56"),
    (1, 128, 28, 28, 128, (3, 3), (1, 1), (1, 1), (1, 1), 128, "nchw", "mobile_dw_28"),
    (1, 256, 14, 14, 256, (3, 3), (1, 1), (1, 1), (1, 1), 256, "nchw", "mobile_dw_14"),
    (1, 512, 7, 7, 512, (3, 3), (1, 1), (1, 1), (1, 1), 512, "both", "mobile_dw_7"),
    # EfficientNet
    (1, 40, 28, 28, 40, (5, 5), (1, 1), (2, 2), (1, 1), 40, "nchw", "effnet_k5_dw"),
    (1, 80, 14, 14, 80, (5, 5), (2, 2), (2, 2), (1, 1), 80, "nchw", "effnet_k5_s2"),
    (1, 64, 28, 28, 64, (3, 3), (1, 1), (2, 2), (2, 2), 64, "nchw", "effnet_k3_d2"),
]


def _expand_arch():
    out = []
    for N, Ci, H, W, Co, ks, s, p, d, g, layouts, label in _ARCH_SHAPES:
        for lay in (["nchw", "nhwc"] if layouts == "both" else [layouts]):
            out.append((N, Ci, H, W, Co, ks, s, p, d, g, lay, f"{label}_{lay}"))
    return out


@pytest.mark.parametrize(
    "N,Ci,H,W,Co,ks,s,p,d,g,layout,label", _expand_arch(),
    ids=[e[-1] for e in _expand_arch()],
)
def test_architecture(N, Ci, H, W, Co, ks, s, p, d, g, layout, label):
    _run(N, Ci, H, W, Co, ks, s, p, d, g, True, torch.float32, layout)


# ── 16. Stress (larger configs, NCHW) ────────────────────────────────────────

@pytest.mark.parametrize("batch,in_c,h,out_c", [
    (8, 64, 56, 64), (16, 128, 28, 128), (32, 64, 28, 64),
    (4, 256, 14, 256), (2, 512, 7, 512), (64, 16, 16, 16),
])
def test_stress(batch, in_c, h, out_c):
    _run(batch, in_c, h, h, out_c, (3, 3), (1, 1), (1, 1), (1, 1), 1, True, torch.float32)


# ── 17. Half precision (both layouts) ────────────────────────────────────────

@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("in_c,out_c,h,k", [
    (3, 64, 32, 7), (64, 64, 16, 3), (128, 128, 8, 3), (32, 32, 16, 3),
])
def test_half_precision(layout, dtype, in_c, out_c, h, k):
    _run(2, in_c, h, h, out_c, (k, k), (1, 1), (k // 2, k // 2), (1, 1), 1, True, dtype, layout)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_nchw_half_precision_extra(dtype):
    """NCHW-only extra config: 16->16, k=5."""
    _run(2, 16, 32, 32, 16, (5, 5), (1, 1), (2, 2), (1, 1), 1, True, dtype)


# ── 18. No-bias combos (NCHW) ───────────────────────────────────────────────

@pytest.mark.parametrize("k,s,p,d,g", [
    (1, 1, 0, 1, 1), (3, 1, 1, 1, 1), (3, 2, 1, 1, 1), (5, 1, 2, 1, 1),
    (3, 1, 0, 2, 1), (3, 1, 1, 1, 4), (3, 2, 1, 1, 16), (1, 1, 0, 1, 16),
])
def test_no_bias_combos(k, s, p, d, g):
    _run(2, 16, 32, 32, 16, (k, k), (s, s), (p, p), (d, d), g, False, torch.float32)


# ── 19. Output shape correctness (NCHW) ─────────────────────────────────────

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


# ── 20. Atrous / dilated convolution ────────────────────────────────────────

@pytest.mark.parametrize("k,d", [
    (3, 2), (3, 3), (3, 4), (3, 6), (5, 2), (5, 3), (7, 2),
])
def test_atrous_conv(k, d):
    p = d * (k // 2)
    h = max(32, k + 2 * d * (k - 1))
    _run(1, 8, h, h, 8, (k, k), (1, 1), (p, p), (d, d), 1, True, torch.float32)


# ── 21. Auto-generated sweep (both layouts) ─────────────────────────────────

@pytest.mark.parametrize("layout", ["nchw", "nhwc"])
@pytest.mark.parametrize(
    "batch,Ci,H,W,Co,k,s,p,d,g", _sweep_configs(),
    ids=[f"k{c[5]}s{c[6]}d{c[8]}g{c[9]}" for c in _sweep_configs()],
)
def test_sweep(layout, batch, Ci, H, W, Co, k, s, p, d, g):
    _run(batch, Ci, H, W, Co, (k, k), (s, s), (p, p), (d, d), g, True, torch.float32, layout)


# ── 22. Cross-layout consistency & NHWC shorthand ───────────────────────────

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
    torch.testing.assert_close(out_nchw, out_nhwc.permute(0, 3, 1, 2), atol=5e-4, rtol=1e-4)


def test_conv2d_nhwc_shorthand():
    """conv2d_nhwc must produce identical results to conv2d(..., layout='nhwc')."""
    x = torch.randn(2, 16, 16, 32, dtype=torch.float32, device="cuda")
    w = torch.randn(64, 32, 3, 3, dtype=torch.float32, device="cuda")
    b = torch.randn(64, dtype=torch.float32, device="cuda")
    out1 = triton_conv2d(x, w, b, padding=1, layout="nhwc")
    out2 = conv2d_nhwc(x, w, b, padding=1)
    torch.testing.assert_close(out1, out2, atol=0, rtol=0)
