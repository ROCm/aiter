# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL-only unit test for gemm_a8w8_blockscale on gfx1250.

Intentionally isolated from the triton test harness: does NOT import
`aiter.ops.flydsl` or `aiter.ops.triton` at top level, because those
trigger a build of `module_aiter_core` (via `aiter.ops.enum`) which is
orthogonal to the flydsl kernel under test. The kernel file is loaded
directly by file path.
"""

import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F

BLOCK_SHAPE = (128, 128)  # (scale_block_n, scale_block_k)

_KERNEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "aiter",
        "ops",
        "flydsl",
        "kernels",
        "gemm_a8w8_blockscale_gfx1250.py",
    )
)


def _get_gpu_arch():
    if not torch.cuda.is_available():
        return None
    return getattr(torch.cuda.get_device_properties(0), "gcnArchName", None)


def _flydsl_available():
    if importlib.util.find_spec("flydsl") is None:
        return False
    arch = _get_gpu_arch()
    return arch is not None and arch.startswith("gfx1250")


if not _flydsl_available():
    pytest.skip(
        "FlyDSL blockscale tests require gfx1250 and the flydsl package.",
        allow_module_level=True,
    )


def _load_kernel():
    """Load the kernel module by file path to skip aiter.ops.flydsl.__init__."""
    spec = importlib.util.spec_from_file_location(
        "_flydsl_a8w8_blockscale_kernel", _KERNEL_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.gemm_a8w8_blockscale


gemm_a8w8_blockscale = _load_kernel()


def _get_fp8_dtype():
    """gfx1250 / MI350 uses OCP FP8 E4M3FN."""
    return torch.float8_e4m3fn


def _generate_inputs(M, N, K, block_shape_n=128, block_shape_k=128):
    """Build FP8 X/W and f32 block scales at the given shape."""
    torch.manual_seed(0)
    fp8 = _get_fp8_dtype()

    x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(fp8)
    w = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(fp8)

    scale_k = (K + block_shape_k - 1) // block_shape_k
    scale_n = (N + block_shape_n - 1) // block_shape_n

    x_scale = torch.rand((M, scale_k), dtype=torch.float32, device="cuda")
    w_scale = torch.rand((scale_n, scale_k), dtype=torch.float32, device="cuda")

    return x, w, x_scale, w_scale


def _reference(
    x, w, x_scale, w_scale, dtype=torch.bfloat16, block_shape_n=128, block_shape_k=128
):
    """Torch reference: broadcast block scales, dequantize, matmul, cast."""
    M, K = x.shape
    N = w.shape[0]

    xs = x_scale.repeat_interleave(block_shape_k, dim=1)[:M, :K]
    x_deq = x.to(xs.dtype) * xs

    ws = (
        w_scale.repeat_interleave(block_shape_n, dim=0).repeat_interleave(
            block_shape_k, dim=1
        )
    )[:N, :K]
    w_deq = w.to(ws.dtype) * ws

    out = F.linear(x_deq.to(torch.float32), w_deq.to(torch.float32))
    return out.to(dtype)


def get_x_vals():
    """K must be a multiple of 128 (kernel constraint); M and N are free."""
    return [
        # Aligned baselines
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (128, 256, 512),
        (256, 512, 256),
        # Tiny M / N (tile-edge masking)
        (1, 1, 128),
        (1, 16, 128),
        (16, 1, 128),
        (1, 3, 128),
        # Unaligned M/N, aligned K
        (3, 5, 128),
        (7, 9, 128),
        (17, 33, 128),
        (63, 127, 128),
        (65, 129, 128),
        # Unaligned M/N, multi-tile K
        (17, 33, 512),
        (63, 127, 256),
        (129, 31, 512),
        # Asymmetric rectangular
        (32, 256, 128),
        (256, 32, 128),
        (128, 128, 1024),
        (1024, 128, 128),
        (1535, 512, 768),
        (1536, 511, 768),
        # bench shapes
        # (32, 5120, 2944),
        # (2048, 5120, 2944),
    ]


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_gemm_a8w8_blockscale(M, N, K, dtype):
    torch.cuda.empty_cache()

    if K % 128 != 0:
        pytest.skip(f"K={K} not a multiple of 128 (kernel constraint)")

    block_shape_n, block_shape_k = BLOCK_SHAPE

    x, w, x_scale, w_scale = _generate_inputs(M, N, K, block_shape_n, block_shape_k)
    ref = _reference(
        x,
        w,
        x_scale,
        w_scale,
        dtype=dtype,
        block_shape_n=block_shape_n,
        block_shape_k=block_shape_k,
    )

    out = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=dtype)

    torch.testing.assert_close(out, ref, atol=1e-2, rtol=1e-2)
