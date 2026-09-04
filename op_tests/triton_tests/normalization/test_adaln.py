# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
import pytest

from aiter.test_common import checkAllclose
from aiter.ops.triton.utils.types import str_to_torch_dtype
from aiter.ops.triton.normalization.adaln_zero import adaln_zero


def run_torch(x, scale, shift, eps):
    """Eager AdaLN-Zero reference: LayerNorm (no affine) + scale/shift modulation."""
    H = x.shape[-1]
    x_norm = F.layer_norm(x, (H,), eps=eps)
    return x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def run_triton(x, scale, shift, eps):
    return adaln_zero(x, scale, shift, eps)


def get_vals():
    # (B, seq, hidden). Mix of real diffusion-DiT shapes (FLUX/SD3.5/CogVideoX/
    # HunyuanVideo/Wan2.1) and generic/edge shapes (odd hidden to exercise masking).
    return [
        (2, 128, 768),
        (1, 1, 4096),
        (3, 333, 257),
        (1, 4096, 3072),  # FLUX.1-dev / HunyuanVideo single block
        (2, 4096, 3072),
        (4, 2560, 3072),  # CogVideoX-5B
        (1, 2592, 2432),  # SD3.5-Large (odd hidden)
        (2, 4096, 5120),  # Wan2.1-14B short
        (1, 16384, 3072),  # HunyuanVideo 480p
        (1, 29700, 5120),  # Wan2.1-14B 720p
    ]


@pytest.mark.parametrize("dtype_str", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("B, S, H", get_vals())
def test_adaln_zero(B, S, H, dtype_str, eps=1e-5):
    dtype = str_to_torch_dtype[dtype_str]
    torch.manual_seed(0)
    x = torch.randn(B, S, H, device="cuda", dtype=dtype)
    scale = torch.randn(B, H, device="cuda", dtype=dtype)
    shift = torch.randn(B, H, device="cuda", dtype=dtype)

    y_torch = run_torch(x, scale, shift, eps)
    y_triton = run_triton(x, scale, shift, eps)

    # The fused kernel accumulates the LayerNorm stats in fp32 and rounds once;
    # the eager reference rounds every intermediate in the working dtype. For
    # bf16/fp16 the two therefore differ by ~1 ULP on a small fraction of
    # elements, so we allow a tiny error ratio (aiter's checkAllclose convention)
    # rather than a strict elementwise compare. fp32 must match tightly.
    if dtype in (torch.float16, torch.bfloat16):
        atol, rtol, max_err_ratio = 1e-2, 1e-2, 0.02
    else:
        atol, rtol, max_err_ratio = 1e-5, 1e-5, 0.0

    err = checkAllclose(
        y_torch,
        y_triton,
        atol=atol,
        rtol=rtol,
        msg=f"adaln_zero {B}x{S}x{H} {dtype_str}",
    )
    assert err <= max_err_ratio, f"error ratio {err} exceeds {max_err_ratio}"


def _perf_sweep():
    """Eager vs fused perf on the diffusion-DiT shapes. Run as a script:

    python op_tests/triton_tests/normalization/test_adaln.py
    """
    from aiter.test_common import run_perftest

    dtype = torch.bfloat16
    eps = 1e-5
    shapes = [
        ("FLUX.1-dev/single", 3072, 4096),
        ("SD3.5-Large/mmdit", 2432, 2592),
        ("CogVideoX-5B/dit", 3072, 2560),
        ("Wan2.1-14B/short", 5120, 4096),
        ("HunyuanVideo/480p", 3072, 16384),
        ("HunyuanVideo/720p", 3072, 29700),
        ("Wan2.1-14B/720p", 5120, 29700),
    ]
    print(
        f"{'shape':22} {'B':>2} {'seq':>6} {'H':>5} "
        f"{'eager_us':>9} {'fused_us':>9} {'uplift':>7} {'fused_GB/s':>10}"
    )
    for name, H, S in shapes:
        for B in (1, 2, 4):
            x = torch.randn(B, S, H, device="cuda", dtype=dtype)
            scale = torch.randn(B, H, device="cuda", dtype=dtype)
            shift = torch.randn(B, H, device="cuda", dtype=dtype)
            _, us_eager = run_perftest(run_torch, x, scale, shift, eps)
            _, us_fused = run_perftest(adaln_zero, x, scale, shift, eps)
            gbps = (2 * B * S * H * x.element_size()) / us_fused / 1e3  # GB/s
            print(
                f"{name:22} {B:>2} {S:>6} {H:>5} "
                f"{us_eager:>9.2f} {us_fused:>9.2f} {us_eager / us_fused:>6.2f}x "
                f"{gbps:>10.1f}"
            )


if __name__ == "__main__":
    _perf_sweep()
