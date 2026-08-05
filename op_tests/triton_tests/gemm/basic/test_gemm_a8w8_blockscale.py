# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale,
    gemm_a8w8_blockscale_preshuffle,
)
from aiter.ops.triton.gluon.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale as gluon_gfx950_gemm_a8w8_blockscale,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.types import get_fp8_dtypes, str_to_torch_dtype

block_shape = (128, 128)
DEVICE_ARCH = arch_info.get_arch()


def run_torch(x, weight, x_scale, w_scale, dtype=torch.bfloat16):
    block_shape_n, block_shape_k = block_shape
    m, k = x.shape
    n = weight.shape[0]
    x_scale = x_scale.repeat_interleave(block_shape_k, dim=1)
    x = x.to(x_scale.dtype) * x_scale[:m, :k]
    x = x.view(m, k)
    w_scale = w_scale.repeat_interleave(block_shape_n, dim=0)
    w_scale = w_scale.repeat_interleave(block_shape_k, dim=1)
    w_scale = w_scale[:n, :k]
    weight = weight.to(w_scale.dtype) * w_scale

    out = F.linear(x.to(torch.float32), weight.to(torch.float32))

    return out.to(dtype)


def run_triton(x, weight, x_scale, w_scale, dtype=torch.bfloat16, y=None, impl=None):
    return impl(x, weight, x_scale, w_scale, dtype, y)


e5m2_type, e4m3_type = get_fp8_dtypes()

DSR1_M32_CASES = (
    pytest.param("gluon_dsr1_m32_n2112_k7168", 32, 2112, 7168, id="n2112-k7168"),
    pytest.param("gluon_dsr1_m32_n3072_k1536", 32, 3072, 1536, id="n3072-k1536"),
    pytest.param("gluon_dsr1_m32_n4608_k7168", 32, 4608, 7168, id="n4608-k7168"),
    pytest.param("gluon_dsr1_m32_n7168_k2048", 32, 7168, 2048, id="n7168-k2048"),
    pytest.param("gluon_dsr1_m32_n7168_k2304", 32, 7168, 2304, id="n7168-k2304"),
)


def get_x_vals():
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
    # GPT-OSS-120B attention projections
    x_vals += [(v, 106496, 16384) for v in (256, 4096)]  # LL3 405B FC1
    x_vals += [(v, 9216, 7168) for v in (128, 192, 4096, 8000)]
    x_vals += [(v, 7168, 4608) for v in (128, 192, 4096, 8000)]
    x_vals += [(v, 8192, 512) for v in (128, 192, 4096, 8000)]
    # Small-K shapes that exercise the gluon wind-down's num_k_iter guards
    # (BLOCK_SIZE_K=128; K in {128,192,256,320} -> num_k_iter in {1,2,2,3}).
    # K<BLOCK_SIZE_K isn't supported by the gluon wrapper (GROUP_K assert).
    x_vals += [(512, 512, K) for K in (128, 192, 256, 320)]
    x_vals += [(v, 8192, 1024) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 4096, 8192) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 4096, 4096) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 4096, 2048) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 1536, 4096) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 32768, 1024) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 8192, 1536) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 7168, 4096) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 1536, 7168) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 7168, 768) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 2048, 7168) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 16384, 1536) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 65536, 1536) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 7168, 16384) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 6144, 7168) for v in (1, 32, 64, 128, 256, 1024)]
    x_vals += [(v, 7168, 3072) for v in (1, 32, 64, 128, 256, 1024)]
    return x_vals


def generate_gemm_a8w8_blockscale_inputs(
    M: int,
    N: int,
    K: int,
    block_shape_n: int,
    block_shape_k: int,
    dtype=torch.bfloat16,
    layout: str = "TN",
    output: bool = False,
    shuffle: bool = False,
):
    """
    The GEMM kernel expects:
    - x: (M, K) -> row-major format
    - w: (N, K) -> column-major format
    """
    torch.manual_seed(0)
    scale_n = (N + block_shape_n - 1) // block_shape_n
    scale_k = (K + block_shape_k - 1) // block_shape_k

    if layout[0] == "T":
        x = (torch.rand((M, K), dtype=torch.float16, device="cuda") / 10).to(e4m3_type)
    else:
        x = (
            (torch.rand((K, M), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .T
        )

    if layout[1] == "N":
        weight = (torch.rand((N, K), dtype=torch.float16, device="cuda") / 10).to(
            e4m3_type
        )
    else:
        weight = (
            (torch.rand((K, N), dtype=torch.float16, device="cuda") / 10)
            .to(e4m3_type)
            .T
        )

    x_scale = torch.rand([M, scale_k], dtype=torch.float32, device="cuda")
    w_scale = torch.rand([scale_n, scale_k], dtype=torch.float32, device="cuda")

    if shuffle:
        weight_shuffle_layout = (16, 16)
        weight_shuffled = shuffle_weight(weight, weight_shuffle_layout).reshape(
            weight.shape[0] // weight_shuffle_layout[0],
            weight.shape[1] * weight_shuffle_layout[0],
        )
        x_scale_shuffled = x_scale.transpose(0, 1).contiguous().view(*x_scale.shape)
    else:
        weight_shuffled = weight
        x_scale_shuffled = x_scale

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda").cuda()

    return x, weight, weight_shuffled, x_scale, x_scale_shuffled, w_scale, y


@pytest.mark.skipif(
    DEVICE_ARCH != "gfx950", reason="DeepSeek-R1 M=32 kernels require gfx950."
)
@pytest.mark.parametrize("selector,M,N,K", DSR1_M32_CASES)
def test_gfx950_dsr1_m32_public_dispatch(
    monkeypatch, selector: str, M: int, N: int, K: int
):
    import aiter.ops.triton.gluon.gemm_a8w8_blockscale_dsr1_m32 as dispatcher
    from aiter.ops import gemm_op_a8w8

    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    x, weight, _, x_scale, _, w_scale, _ = generate_gemm_a8w8_blockscale_inputs(
        M,
        N,
        K,
        *block_shape,
        dtype=torch.bfloat16,
        layout="TN",
    )
    reference = run_torch(x, weight, x_scale, w_scale, torch.bfloat16)

    config_calls = []

    def get_config(m, n, k, tuned_file):
        config_calls.append((m, n, k, tuned_file))
        assert (m, n, k) == (M, N, K)
        return {"libtype": "gluon", "kernelName": selector, "splitK": 0}

    def fail_ck_fallback(*args, **kwargs):
        pytest.fail("exact-shape dispatch unexpectedly fell back to CK")

    real_load_kernel = dispatcher._load_kernel
    kernel_calls = []

    def load_kernel_spy(shape):
        real_kernel = real_load_kernel(shape)

        def kernel_spy(a, b, a_scale, b_scale, out):
            kernel_calls.append(
                {
                    "shape": shape,
                    "kernel": real_kernel,
                    "out_ptr": out.data_ptr(),
                }
            )
            result = real_kernel(a, b, a_scale, b_scale, out)
            kernel_calls[-1]["result_ptr"] = result.data_ptr()
            return result

        return kernel_spy

    monkeypatch.setattr(gemm_op_a8w8, "_hip_blockscale_supported", lambda: True)
    monkeypatch.setattr(gemm_op_a8w8, "get_CKGEMM_config", get_config)
    monkeypatch.setattr(gemm_op_a8w8, "gemm_a8w8_blockscale_ck", fail_ck_fallback)
    monkeypatch.setattr(dispatcher, "_load_kernel", load_kernel_spy)

    result = gemm_op_a8w8.gemm_a8w8_blockscale(
        x,
        weight,
        x_scale,
        w_scale,
        dtype=torch.bfloat16,
    )

    assert len(config_calls) == 1
    assert len(kernel_calls) == 1
    call = kernel_calls[0]
    assert call["shape"] == (M, N, K)
    assert call["kernel"].__name__ == f"block_scaled_mm_n{N}_k{K}_m32"
    assert call["kernel"].__module__.endswith(f".n{N}_k{K}_m32")
    assert call["result_ptr"] == call["out_ptr"] == result.data_ptr()
    torch.testing.assert_close(reference, result, atol=0.01, rtol=1e-2)


@pytest.mark.skipif(
    DEVICE_ARCH != "gfx950", reason="DeepSeek-R1 M=32 kernels require gfx950."
)
def test_gfx950_dsr1_m32_dispatch_rejects_invalid_contracts(monkeypatch):
    import aiter.ops.triton.gluon.gemm_a8w8_blockscale_dsr1_m32 as dispatcher

    M, N, K = 32, 3072, 1536
    selector = "gluon_dsr1_m32_n3072_k1536"
    device = "cuda"
    x = torch.empty((M, K), dtype=torch.float8_e4m3fn, device=device)
    weight = torch.empty((N, K), dtype=torch.float8_e4m3fn, device=device)
    x_scale = torch.empty((M, K // 128), dtype=torch.float32, device=device)
    weight_scale = torch.empty((N // 128, K // 128), dtype=torch.float32, device=device)
    out = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    x_m33 = torch.empty((M + 1, K), dtype=x.dtype, device=device)
    x_scale_m33 = torch.empty((M + 1, K // 128), dtype=x_scale.dtype, device=device)
    out_m33 = torch.empty((M + 1, N), dtype=out.dtype, device=device)
    x_noncontiguous = torch.empty((K, M), dtype=x.dtype, device=device).T
    out_noncontiguous = torch.empty((N, M), dtype=out.dtype, device=device).T

    invalid_cases = {
        "m31": (x[: M - 1], weight, x_scale[: M - 1], weight_scale, out[: M - 1]),
        "m33": (x_m33, weight, x_scale_m33, weight_scale, out_m33),
        "fp16-output": (
            x,
            weight,
            x_scale,
            weight_scale,
            torch.empty_like(out, dtype=torch.float16),
        ),
        "fnuz-input": (
            torch.empty_like(x, dtype=torch.float8_e4m3fnuz),
            weight,
            x_scale,
            weight_scale,
            out,
        ),
        "noncontiguous-input": (
            x_noncontiguous,
            weight,
            x_scale,
            weight_scale,
            out,
        ),
        "noncontiguous-output": (
            x,
            weight,
            x_scale,
            weight_scale,
            out_noncontiguous,
        ),
        "wrong-scale-shape": (
            x,
            weight,
            torch.empty((M, K // 128 - 1), dtype=x_scale.dtype, device=x_scale.device),
            weight_scale,
            out,
        ),
    }

    def fail_load_kernel(shape):
        pytest.fail(f"invalid contract unexpectedly loaded exact kernel for {shape}")

    monkeypatch.setattr(dispatcher, "_load_kernel", fail_load_kernel)

    for case, tensors in invalid_cases.items():
        result = dispatcher.try_gemm_a8w8_blockscale_dsr1_m32(
            *tensors,
            kernel_name=selector,
            gfx="gfx950",
        )
        assert result is None, case

    selector_mismatch = dispatcher.try_gemm_a8w8_blockscale_dsr1_m32(
        x,
        weight,
        x_scale,
        weight_scale,
        out,
        kernel_name="gluon_dsr1_m32_n2112_k7168",
        gfx="gfx950",
    )
    assert selector_mismatch is None


@pytest.mark.skipif(
    DEVICE_ARCH != "gfx950", reason="DeepSeek-R1 M=32 kernels require gfx950."
)
@pytest.mark.parametrize(
    "M,dtype",
    (
        pytest.param(31, torch.bfloat16, id="padded-m31"),
        pytest.param(32, torch.float16, id="fp16-output"),
    ),
)
def test_gfx950_dsr1_m32_public_dispatch_preserves_tuned_ck_fallback(
    monkeypatch, M: int, dtype: torch.dtype
):
    from aiter.ops import gemm_op_a8w8
    from aiter.ops.triton.gluon.gemm_a8w8_blockscale_dsr1_m32 import (
        DSR1_M32_CK_FALLBACK_CONFIGS,
    )

    N, K = 3072, 1536
    selector = "gluon_dsr1_m32_n3072_k1536"
    x = torch.empty((M, K), dtype=torch.float8_e4m3fn, device="cuda")
    weight = torch.empty((N, K), dtype=torch.float8_e4m3fn, device="cuda")
    x_scale = torch.empty((M, K // 128), dtype=torch.float32, device="cuda")
    weight_scale = torch.empty((N // 128, K // 128), dtype=torch.float32, device="cuda")
    ck_calls = []

    def get_config(m, n, k, tuned_file):
        assert (m, n, k) == (M, N, K)
        return {"libtype": "gluon", "kernelName": selector, "splitK": 0}

    def fake_ck(a, b, a_scale, b_scale, out, *, splitK=0, kernelName=""):
        ck_calls.append((splitK, kernelName, out))
        return out

    monkeypatch.setattr(gemm_op_a8w8, "_hip_blockscale_supported", lambda: True)
    monkeypatch.setattr(gemm_op_a8w8, "get_CKGEMM_config", get_config)
    monkeypatch.setattr(gemm_op_a8w8, "gemm_a8w8_blockscale_ck", fake_ck)

    result = gemm_op_a8w8.gemm_a8w8_blockscale(
        x,
        weight,
        x_scale,
        weight_scale,
        dtype=dtype,
    )

    assert len(ck_calls) == 1
    split_k, kernel_name, out = ck_calls[0]
    assert (split_k, kernel_name) == DSR1_M32_CK_FALLBACK_CONFIGS[selector]
    assert result is out and result.dtype == dtype and result.shape == (M, N)


@pytest.mark.parametrize(
    "dtype, M, N, K, layout, output",
    [
        (dtype, *shape, layout, output)
        for output in [True]
        for dtype in ["bf16"]
        for layout in ["TN"]
        for shape in get_x_vals()
    ],
)
@pytest.mark.parametrize("backend", ["gluon", "triton"])
@pytest.mark.parametrize("shuffle", [True, False])
def test_gemm(dtype, M, N, K, layout, output, backend, shuffle):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests
    torch.cuda.synchronize()

    block_shape_n, block_shape_k = block_shape

    if backend == "gluon":
        if shuffle:
            if DEVICE_ARCH not in ("gfx1250"):
                pytest.skip("Gluon + shuffle implementation requires gfx1250.")
        elif DEVICE_ARCH not in ("gfx950", "gfx1250"):
            pytest.skip("Gluon implementation requires gfx950 or gfx1250.")

    if shuffle and (N % 16 > 0 or K % 32 > 0):
        pytest.skip(
            "N has to be multiple of 16 and K has to be multiple of 32 for preshuffle cases"
        )

    if backend not in ("gluon",) and K < 512:
        pytest.skip("Small-K shapes exercise gluon-only paths.")

    dtype = str_to_torch_dtype[dtype]
    x, weight, weight_triton, x_scale, x_scale_shuffled, w_scale, y = (
        generate_gemm_a8w8_blockscale_inputs(
            M,
            N,
            K,
            block_shape_n,
            block_shape_k,
            dtype=dtype,
            layout=layout,
            output=output,
            shuffle=shuffle,
        )
    )

    a = run_torch(x, weight, x_scale, w_scale, dtype)

    if not shuffle and backend == "gluon" and DEVICE_ARCH == "gfx950":
        impl = gluon_gfx950_gemm_a8w8_blockscale
    else:
        if shuffle:

            def impl(x, w, xs, ws, dt, y):
                return gemm_a8w8_blockscale_preshuffle(
                    x, w, xs, ws, dt, y, backend=backend
                )

        else:

            def impl(x, w, xs, ws, dt, y):
                return gemm_a8w8_blockscale(x, w, xs, ws, dt, y, backend=backend)

    b = run_triton(x, weight_triton, x_scale_shuffled, w_scale, dtype, y, impl)

    torch.testing.assert_close(a, b, atol=0.01, rtol=1e-2)
