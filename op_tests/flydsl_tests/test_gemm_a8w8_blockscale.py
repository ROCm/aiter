# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL unit tests for A8W8 FP8 blockscale GEMM on gfx1250.

Intentionally isolated from the triton test harness: this file does not import
`aiter.ops.flydsl` or `aiter.ops.triton` at top-level because those can trigger
an unrelated `module_aiter_core` build.
"""

import argparse
import importlib.util
import os

import pytest
import torch

SCALE_BLOCK_N = 128
SCALE_BLOCK_K = 128

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
    """Load kernel module by file path to bypass aiter.ops.flydsl package init."""
    spec = importlib.util.spec_from_file_location(
        "_flydsl_a8w8_blockscale_kernel", _KERNEL_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.gemm_a8w8_blockscale


gemm_a8w8_blockscale = _load_kernel()


def _check_gfx1250():
    arch = _get_gpu_arch()
    if arch is None or not arch.startswith("gfx1250"):
        pytest.skip(f"gemm_a8w8_blockscale requires gfx1250, got {arch}")


def _check_shape_compat(M, N, K, tile_k=128, num_buffers=2):
    """Kernel requires num_k_tiles >= num_buffers - 1."""
    _ = M
    _ = N
    num_k_tiles = K // tile_k
    if num_k_tiles < num_buffers - 1:
        pytest.skip(
            f"{num_buffers}-stage pipeline requires num_k_tiles >= {num_buffers - 1}, "
            f"got K={K} (num_k_tiles={num_k_tiles})"
        )


def _get_fp8_dtype():
    """gfx1250 / MI350 uses OCP FP8 E4M3FN."""
    return torch.float8_e4m3fn


def _generate_inputs(
    M,
    N,
    K,
    scale_block_n=SCALE_BLOCK_N,
    scale_block_k=SCALE_BLOCK_K,
):
    """Build FP8 X/W plus f32 block scales."""
    torch.manual_seed(0)
    fp8 = _get_fp8_dtype()

    x = (torch.rand((M, K), dtype=torch.float32, device="cuda") / 10).to(fp8)
    w = (torch.rand((N, K), dtype=torch.float32, device="cuda") / 10).to(fp8)

    scale_k = (K + scale_block_k - 1) // scale_block_k
    scale_n = (N + scale_block_n - 1) // scale_block_n

    x_scale = torch.rand((M, scale_k), dtype=torch.float32, device="cuda")
    w_scale = torch.rand((scale_n, scale_k), dtype=torch.float32, device="cuda")

    return x, w, x_scale, w_scale


def _reference_output(
    x_fp8,
    w_fp8,
    x_scale,
    w_scale,
    scale_block_n=SCALE_BLOCK_N,
    scale_block_k=SCALE_BLOCK_K,
    dtype=torch.bfloat16,
):
    """Broadcast scales over tiles, dequantize, matmul in f32, cast."""
    M, K = x_fp8.shape
    N = w_fp8.shape[0]

    xs_broadcast = x_scale.repeat_interleave(scale_block_k, dim=1)[:M, :K]
    x_deq = x_fp8.to(xs_broadcast.dtype) * xs_broadcast

    ws_broadcast = (
        w_scale.repeat_interleave(scale_block_n, dim=0).repeat_interleave(
            scale_block_k, dim=1
        )
    )[:N, :K]
    w_deq = w_fp8.to(ws_broadcast.dtype) * ws_broadcast

    out = torch.matmul(x_deq.float(), w_deq.float().T)
    return out.to(dtype)


def _assert_close(out, ref, *, rtol=1e-2, atol=1e-2):
    torch.testing.assert_close(
        out.cpu().to(torch.float32),
        ref.cpu().to(torch.float32),
        rtol=rtol,
        atol=atol,
    )


def get_basic_shapes():
    return [
        (128, 128, 128),
        (128, 256, 256),
        (256, 128, 256),
        (128, 512, 128),
        (512, 128, 128),
        (128, 128, 512),
        (128, 128, 1024),
        (128, 1536, 7168),
        (128, 7168, 1536),
    ]


def get_large_shapes():
    return [
        (256, 1024, 1024),
        (512, 2048, 2048),
    ]


def get_tdm_store_shapes():
    return [
        (128, 128, 128),
        (256, 256, 256),
        (128, 512, 256),
        (128, 256, 1024),
        (512, 1024, 1024),
    ]


@pytest.mark.parametrize("M, N, K", get_basic_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_a8w8_blockscale_basic(M, N, K, dtype):
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=dtype)
    out = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=dtype)
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M, N, K", [(128, 256, 256), (256, 512, 512)])
@pytest.mark.parametrize("num_buffers", [2, 3, 4])
def test_gemm_a8w8_blockscale_num_buffers(M, N, K, num_buffers):
    _check_gfx1250()
    _check_shape_compat(M, N, K, num_buffers=num_buffers)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    out = gemm_a8w8_blockscale(
        x,
        w,
        x_scale,
        w_scale,
        dtype=torch.bfloat16,
        num_buffers=num_buffers,
    )
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M, N, K", [(128, 256, 256)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gemm_a8w8_blockscale_dtype(M, N, K, dtype):
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=dtype)
    out = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=dtype)

    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    atol = 1e-3 if dtype == torch.float32 else 1e-2
    _assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (256, 256, 256)])
def test_gemm_a8w8_blockscale_preallocated_output(M, N, K):
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    y = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)

    out = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=torch.bfloat16, y=y)
    assert out.data_ptr() == y.data_ptr(), "Output should reuse pre-allocated y"
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "M, N, K",
    [
        (128, 256, 256),
        (256, 128, 256),
        (128, 128, 512),
        (128, 128, 1024),
        (1024, 1024, 1024),
    ],
)
def test_gemm_a8w8_blockscale_scales_per_tile(M, N, K):
    _check_gfx1250()
    _check_shape_compat(M, N, K, tile_k=256)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    out = gemm_a8w8_blockscale(
        x,
        w,
        x_scale,
        w_scale,
        dtype=torch.bfloat16,
        tile_k=256,
    )
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M, N, K", get_large_shapes())
def test_gemm_a8w8_blockscale_large(M, N, K):
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    out = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M, N, K", get_tdm_store_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_a8w8_blockscale_tdm_store_basic(M, N, K, dtype):
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=dtype)
    out = gemm_a8w8_blockscale(
        x,
        w,
        x_scale,
        w_scale,
        dtype=dtype,
        use_tdm_store=True,
    )
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M, N, K", [(128, 256, 256)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gemm_a8w8_blockscale_tdm_store_dtype(M, N, K, dtype):
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=dtype)
    out = gemm_a8w8_blockscale(
        x,
        w,
        x_scale,
        w_scale,
        dtype=dtype,
        use_tdm_store=True,
    )

    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    atol = 1e-3 if dtype == torch.float32 else 1e-2
    _assert_close(out, ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("M, N, K", [(128, 256, 256), (256, 512, 512)])
@pytest.mark.parametrize("num_buffers", [2, 3, 4])
def test_gemm_a8w8_blockscale_tdm_store_num_buffers(M, N, K, num_buffers):
    _check_gfx1250()
    _check_shape_compat(M, N, K, num_buffers=num_buffers)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    out = gemm_a8w8_blockscale(
        x,
        w,
        x_scale,
        w_scale,
        dtype=torch.bfloat16,
        num_buffers=num_buffers,
        use_tdm_store=True,
    )
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M, N, K", [(128, 128, 128), (256, 256, 256)])
def test_gemm_a8w8_blockscale_tdm_store_preallocated_output(M, N, K):
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    y = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)

    out = gemm_a8w8_blockscale(
        x,
        w,
        x_scale,
        w_scale,
        dtype=torch.bfloat16,
        y=y,
        use_tdm_store=True,
    )
    assert out.data_ptr() == y.data_ptr(), "Output should reuse pre-allocated y"
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-M", type=int, default=128)
    parser.add_argument("-N", type=int, default=256)
    parser.add_argument("-K", type=int, default=256)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "f32"],
    )
    parser.add_argument("--num-buffers", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument(
        "--tdm-store",
        action="store_true",
        help="Use the LDS-staged TDM-store epilogue.",
    )
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "f32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    _check_gfx1250()
    _check_shape_compat(args.M, args.N, args.K, num_buffers=args.num_buffers)

    x, w, x_scale, w_scale = _generate_inputs(args.M, args.N, args.K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=dtype)
    out = gemm_a8w8_blockscale(
        x,
        w,
        x_scale,
        w_scale,
        dtype=dtype,
        num_buffers=args.num_buffers,
        use_tdm_store=args.tdm_store,
    )

    torch.cuda.synchronize()
    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    atol = 1e-3 if dtype == torch.float32 else 1e-2
    _assert_close(out, ref, rtol=rtol, atol=atol)
    print(
        f"PASSED M={args.M} N={args.N} K={args.K} dtype={args.dtype} "
        f"num_buffers={args.num_buffers} tdm_store={args.tdm_store}"
    )
