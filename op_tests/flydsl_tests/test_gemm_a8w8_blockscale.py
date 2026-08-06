# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL unit tests for A8W8 FP8 blockscale GEMM on gfx1250."""

import argparse
import importlib.util
import os

import pytest
import torch

from aiter.ops.shuffle import shuffle_weight_gfx1250

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
        "gemm_a8w8_blockscale_f32_gfx1250.py",
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
        "_flydsl_a8w8_blockscale_wrapper", _KERNEL_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.gemm_a8w8_blockscale


_raw_gemm_a8w8_blockscale = _load_kernel()


def gemm_a8w8_blockscale(*args, **kwargs):
    # Unit-test default: 3 buffers. An explicit num_buffers in a test
    # (e.g. the num_buffers sweep) still overrides this.
    kwargs.setdefault("num_buffers", 3)
    return _raw_gemm_a8w8_blockscale(*args, **kwargs)


def _check_gfx1250():
    arch = _get_gpu_arch()
    if arch is None or not arch.startswith("gfx1250"):
        pytest.skip(f"gemm_a8w8_blockscale requires gfx1250, got {arch}")


def _padded_k(K, tile_k=128):
    """The wrapper zero-pads K up to a multiple of tile_k before launching."""
    return ((K + tile_k - 1) // tile_k) * tile_k


def _check_shape_compat(M, N, K, tile_k=128, num_buffers=3):
    """Kernel requires num_k_tiles >= num_buffers - 1."""
    _ = M
    _ = N
    num_k_tiles = _padded_k(K, tile_k) // tile_k
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


@pytest.mark.parametrize("M, N, K", get_basic_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_gemm_a8w8_blockscale_basic(M, N, K, dtype):
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=dtype)
    w = shuffle_weight_gfx1250(w)
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
    w = shuffle_weight_gfx1250(w)
    out = gemm_a8w8_blockscale(
        x,
        w,
        x_scale,
        w_scale,
        dtype=torch.bfloat16,
        num_buffers=num_buffers,
    )
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "M, N, K", [(128, 256, 256), (256, 512, 512), (128, 128, 1024)]
)
@pytest.mark.parametrize("variant", ["compute_bound", "memory_bound"])
def test_gemm_a8w8_blockscale_variant(M, N, K, variant):
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    w = shuffle_weight_gfx1250(w)
    out = gemm_a8w8_blockscale(
        x, w, x_scale, w_scale, dtype=torch.bfloat16, variant=variant
    )
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


_M_SWEEP = [1, 32, 64, 128]
_NK_SWEEP = [
    (8192, 1024),
    (4096, 8192),
    (4096, 4096),
    (4096, 2048),
    (1536, 4096),
    (32768, 1024),
]


@pytest.mark.parametrize("N, K", _NK_SWEEP)
@pytest.mark.parametrize("M", _M_SWEEP)
@pytest.mark.parametrize("variant", ["compute_bound", "memory_bound"])
def test_gemm_a8w8_blockscale_m_sweep(variant, M, N, K):
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    w = shuffle_weight_gfx1250(w)
    out = gemm_a8w8_blockscale(
        x, w, x_scale, w_scale, dtype=torch.bfloat16, variant=variant
    )
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize(
    "M, N, K", [(128, 256, 1024), (256, 512, 2048), (128, 128, 4096)]
)
@pytest.mark.parametrize("split_k", [2, 4, 8])
def test_gemm_a8w8_blockscale_split_k(M, N, K, split_k):
    _check_gfx1250()
    tile_k = 128
    num_buffers = 3
    if K % (split_k * tile_k) != 0:
        pytest.skip(f"K={K} not divisible by split_k*tile_k ({split_k}*{tile_k})")
    if (K // split_k) // tile_k < num_buffers - 1:
        pytest.skip(f"per-split num_k_tiles < num_buffers-1 (split_k={split_k}, K={K})")
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    w = shuffle_weight_gfx1250(w)
    out = gemm_a8w8_blockscale(
        x,
        w,
        x_scale,
        w_scale,
        dtype=torch.bfloat16,
        variant="memory_bound",
        split_k=split_k,
    )
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


def test_gemm_a8w8_blockscale_split_k_requires_memory_bound():
    _check_gfx1250()
    M, N, K = 128, 256, 1024
    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    w = shuffle_weight_gfx1250(w)
    with pytest.raises(ValueError):
        gemm_a8w8_blockscale(
            x,
            w,
            x_scale,
            w_scale,
            dtype=torch.bfloat16,
            variant="compute_bound",
            split_k=2,
        )


@pytest.mark.parametrize("M, N, K", [(128, 256, 256)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gemm_a8w8_blockscale_dtype(M, N, K, dtype):
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=dtype)
    w = shuffle_weight_gfx1250(w)
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
    w = shuffle_weight_gfx1250(w)

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
    w = shuffle_weight_gfx1250(w)
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
    w = shuffle_weight_gfx1250(w)
    out = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


_RAGGED_M = [
    7,
    15,
    17,
    31,
    33,
    63,
    65,
    127,
    129,
    136,
    144,
    191,
    193,
    200,
    255,
    257,
]

_RAGGED_N = [16, 48, 112, 144, 240, 272, 400]

_RAGGED_K = [
    (96, 128),
    (160, 128),
    (288, 128),
    (544, 128),
    (1056, 128),
    (384, 256),
    (1152, 256),
]


@pytest.mark.parametrize("M", _RAGGED_M)
def test_gemm_a8w8_blockscale_ragged_m(M):
    _check_gfx1250()
    N, K = 256, 512
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    w = shuffle_weight_gfx1250(w)
    out = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    assert out.shape == (M, N)
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M", [129, 200])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_gemm_a8w8_blockscale_ragged_m_dtype(M, dtype):
    _check_gfx1250()
    N, K = 256, 512
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=dtype)
    w = shuffle_weight_gfx1250(w)
    out = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=dtype)

    tol = 1e-3 if dtype == torch.float32 else 1e-2
    _assert_close(out, ref, rtol=tol, atol=tol)


@pytest.mark.parametrize("N", _RAGGED_N)
def test_gemm_a8w8_blockscale_ragged_n(N):
    _check_gfx1250()
    M, K = 128, 512
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    w = shuffle_weight_gfx1250(w)
    out = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    assert out.shape == (M, N)
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M, N, K", [(129, 144, 512), (200, 240, 512), (257, 272, 512)])
@pytest.mark.parametrize("variant", ["compute_bound", "memory_bound"])
def test_gemm_a8w8_blockscale_ragged_mn_variant(M, N, K, variant):
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    w = shuffle_weight_gfx1250(w)
    out = gemm_a8w8_blockscale(
        x, w, x_scale, w_scale, dtype=torch.bfloat16, variant=variant
    )
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("num_buffers", [2, 3, 4])
def test_gemm_a8w8_blockscale_ragged_m_num_buffers(num_buffers):
    _check_gfx1250()
    M, N, K = 200, 256, 512
    _check_shape_compat(M, N, K, num_buffers=num_buffers)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    w = shuffle_weight_gfx1250(w)
    out = gemm_a8w8_blockscale(
        x, w, x_scale, w_scale, dtype=torch.bfloat16, num_buffers=num_buffers
    )
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M", [129, 200, 255])
@pytest.mark.parametrize("split_k", [2, 4])
def test_gemm_a8w8_blockscale_ragged_m_split_k(M, split_k):
    _check_gfx1250()
    N, K = 256, 1024
    tile_k = 128
    num_buffers = 3
    if (K // split_k) // tile_k < num_buffers - 1:
        pytest.skip(f"per-split num_k_tiles < num_buffers-1 (split_k={split_k})")
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    w = shuffle_weight_gfx1250(w)
    out = gemm_a8w8_blockscale(
        x,
        w,
        x_scale,
        w_scale,
        dtype=torch.bfloat16,
        variant="memory_bound",
        split_k=split_k,
    )
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M, N", [(128, 144), (200, 272), (129, 16)])
def test_gemm_a8w8_blockscale_ragged_n_split_k(M, N):
    _check_gfx1250()
    K, split_k = 1024, 2
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    w = shuffle_weight_gfx1250(w)
    out = gemm_a8w8_blockscale(
        x,
        w,
        x_scale,
        w_scale,
        dtype=torch.bfloat16,
        variant="memory_bound",
        split_k=split_k,
    )
    assert out.shape == (M, N)
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("K, tile_k", _RAGGED_K)
@pytest.mark.parametrize("M", [128, 200])
def test_gemm_a8w8_blockscale_ragged_k(M, K, tile_k):
    _check_gfx1250()
    N = 256
    num_buffers = 2
    _check_shape_compat(M, N, K, tile_k=tile_k, num_buffers=num_buffers)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    w = shuffle_weight_gfx1250(w)
    out = gemm_a8w8_blockscale(
        x,
        w,
        x_scale,
        w_scale,
        dtype=torch.bfloat16,
        tile_k=tile_k,
        num_buffers=num_buffers,
    )
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("M", [7, 129, 136, 191, 200, 255])
def test_gemm_a8w8_blockscale_no_oob_row_writes(M):
    _check_gfx1250()
    N, K = 256, 512
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    canary_rows = tile_m = 128
    big = torch.full(
        (M + canary_rows, N), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    y = big[:M]

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    w = shuffle_weight_gfx1250(w)
    out = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=torch.bfloat16, y=y)

    assert out.data_ptr() == y.data_ptr(), "kernel did not write our buffer in place"
    _assert_close(out, ref, rtol=1e-2, atol=1e-2)

    clobbered = ~big[M:].isnan()
    n_clobbered = int(clobbered.sum())
    assert n_clobbered == 0, (
        f"M={M} (tile_m={tile_m}): kernel wrote {n_clobbered} element(s) past row "
        f"{M - 1}; first clobbered row offset "
        f"{int(clobbered.any(dim=1).nonzero()[0])}"
    )


@pytest.mark.parametrize("M, N, K", [(128, 144, 512), (200, 272, 512), (129, 16, 512)])
def test_gemm_a8w8_blockscale_ragged_n_preallocated_output(M, N, K):
    _check_gfx1250()
    _check_shape_compat(M, N, K)
    torch.cuda.empty_cache()

    x, w, x_scale, w_scale = _generate_inputs(M, N, K)
    y = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
    ref = _reference_output(x, w, x_scale, w_scale, dtype=torch.bfloat16)
    w = shuffle_weight_gfx1250(w)

    out = gemm_a8w8_blockscale(x, w, x_scale, w_scale, dtype=torch.bfloat16, y=y)
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
    parser.add_argument(
        "--num-buffers", type=int, default=2, choices=[2, 3, 4, 5, 6, 7, 8]
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="compute_bound",
        choices=["compute_bound", "memory_bound"],
    )
    parser.add_argument("--tile-m", type=int, default=None)
    parser.add_argument("--tile-n", type=int, default=None)
    parser.add_argument("--tile-k", type=int, default=None)
    parser.add_argument("--m-warp", type=int, default=None)
    parser.add_argument("--n-warp", type=int, default=None)
    parser.add_argument("--split-k", type=int, default=None)
    args = parser.parse_args()

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "f32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    _check_gfx1250()
    tile_k_compat = args.tile_k if args.tile_k is not None else 128
    _check_shape_compat(
        args.M, args.N, args.K, tile_k=tile_k_compat, num_buffers=args.num_buffers
    )

    tuned = {}
    for name, val in (
        ("tile_m", args.tile_m),
        ("tile_n", args.tile_n),
        ("tile_k", args.tile_k),
        ("m_warp", args.m_warp),
        ("n_warp", args.n_warp),
        ("split_k", args.split_k),
    ):
        if val is not None:
            tuned[name] = val

    x, w, x_scale, w_scale = _generate_inputs(args.M, args.N, args.K)
    ref = _reference_output(x, w, x_scale, w_scale, dtype=dtype)
    w = shuffle_weight_gfx1250(w)
    out = gemm_a8w8_blockscale(
        x,
        w,
        x_scale,
        w_scale,
        dtype=dtype,
        num_buffers=args.num_buffers,
        variant=args.variant,
        **tuned,
    )

    torch.cuda.synchronize()
    rtol = 1e-3 if dtype == torch.float32 else 1e-2
    atol = 1e-3 if dtype == torch.float32 else 1e-2
    _assert_close(out, ref, rtol=rtol, atol=atol)
    print(
        f"PASSED M={args.M} N={args.N} K={args.K} dtype={args.dtype} "
        f"num_buffers={args.num_buffers} "
        f"variant={args.variant} " + " ".join(f"{k}={v}" for k, v in tuned.items())
    )
