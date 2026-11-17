# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.gemm.basic.gemm_a16w16 import _is_gluon_available, gemm_a16w16
from aiter.ops.triton.gemm.basic.gemm_a16w16_atomic import gemm_a16w16_atomic
from op_tests.triton_tests.utils.types import str_to_torch_dtype


def is_gluon_supported():
    """gluon a16w16 kernels are only available on supported archs (gfx1250)."""
    return _is_gluon_available()


def _skip_if_triton_on_gfx1250(backend):
    """gfx1250 only ships gluon-format a16w16 configs, so the triton backend has
    no usable config there; skip the triton backend on gfx1250."""
    if backend != "triton":
        return
    from aiter.ops.triton.utils._triton.arch_info import get_arch

    if "gfx1250" in (get_arch() or ""):
        pytest.skip("triton backend has no gfx1250 a16w16 config (gluon-only arch)")


def generate_gemm_a16w16_inputs(M, N, K, dtype, layout="TN", output=True, bias=False):
    torch.manual_seed(0)
    if isinstance(dtype, str):
        dtype = str_to_torch_dtype[dtype]

    # TN is default layout
    if layout[0] == "T":
        x = torch.randn((M, K), dtype=dtype, device="cuda")
    else:
        x = torch.randn((K, M), dtype=dtype, device="cuda").T

    if layout[1] == "T":
        weight = torch.randn((K, N), dtype=dtype, device="cuda").T
    else:
        weight = torch.randn((N, K), dtype=dtype, device="cuda")

    bias_tensor = None
    if bias:
        bias_tensor = torch.empty((N), dtype=dtype, device="cuda")

    y = None
    if output:
        y = torch.empty((M, N), dtype=dtype, device="cuda")
        out_dtype = (None,)
    else:
        out_dtype = dtype

    return x, weight, bias_tensor, out_dtype, y


def get_x_vals():
    x_vals = [(1, 1, 1)]  # minimal case
    x_vals += [(3, 5, 2)]  # irregular shape
    x_vals += [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 5, 8)]
    x_vals += [(2**i, 256, 7168) for i in range(5, 9)]  # DSR1 router GEMM
    # GPT-OSS-120B attention projections
    x_vals += [(2**i, 5120, 2880) for i in range(5, 9)]  # GPTOSS QKV input projection
    x_vals += [(2048, 5120, 2880)]  # GPTOSS QKV @ M=2048
    x_vals += [(8192, 5120, 2880)]  # GPTOSS QKV @ M=8192
    x_vals += [(2**i, 2880, 4096) for i in range(5, 9)]  # output projection
    x_vals += [(2**i, 128, 2880) for i in range(5, 9)]  # Router GEMM
    x_vals += [(v, 106496, 16384) for v in (256, 4096)]  # LL3 405B FC1
    return x_vals


# Test plain BF16 GEMMs - the most common types.
@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("kernel_type", ["bandwidth_bound", "compute_bound"])
def test_gemm_a16_w16(M: int, N: int, K: int, backend, kernel_type):
    if backend == "triton" and kernel_type != "bandwidth_bound":
        pytest.skip("kernel_type only applies to the gluon backend")
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    _skip_if_triton_on_gfx1250(backend)

    x, w, _, _out_dtype, _y = generate_gemm_a16w16_inputs(
        M,
        N,
        K,
        dtype=torch.bfloat16,
        output=False,
    )

    torch_out = F.linear(x, w, bias=None)

    triton_out = gemm_a16w16(x, w, backend=backend, kernel_type=kernel_type)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-2)


# Output larger than 2**31 bytes (Wan2.2 FFN up-projection at 720p x 193 frames).
@pytest.mark.parametrize("M, N, K", [(90000, 13824, 5120)])
def test_gemm_a16_w16_large_output(M: int, N: int, K: int):
    if not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    assert M * N * 2 > 2**31

    x, w, _, _, _ = generate_gemm_a16w16_inputs(
        M, N, K, dtype=torch.bfloat16, output=False
    )
    # NaN-fill so any row the kernel never writes fails the comparison.
    y = torch.full((M, N), float("nan"), dtype=torch.bfloat16, device="cuda")
    gemm_a16w16(x, w, y=y, backend="gluon")

    boundary_row = 2**31 // (N * 2)
    for rows in (
        slice(0, 1024),
        slice(boundary_row - 512, boundary_row + 512),
        slice(M - 1024, M),
    ):
        torch.testing.assert_close(y[rows], F.linear(x[rows], w), atol=1e-1, rtol=1e-2)


# Smaller set for testing activations, setting the output tensor and dtype
def get_fewer_x_vals():
    x_vals = [(16, 1024, 1024)]
    x_vals += [(128, 8192, 512)]
    x_vals += [(256, 512, 8192)]
    x_vals += [(1024 * v, 1024 * v, 1024 * v) for v in (1, 5, 8)]
    return x_vals


# A smaller set of shapes that tests fused activations, different dtypes
# and output tensor arg. We don't want the larger set above to test
# all these combinations.
@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "silu"])
@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("kernel_type", ["bandwidth_bound", "compute_bound"])
def test_gemm_a16_w16_activation(
    M: int, N: int, K: int, dtype, output, activation, backend, kernel_type
):
    if backend == "triton" and kernel_type != "bandwidth_bound":
        pytest.skip("kernel_type only applies to the gluon backend")
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    _skip_if_triton_on_gfx1250(backend)

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(
        M,
        N,
        K,
        dtype,
        output=output,
    )

    torch_out = F.linear(x, w, bias=None)
    if activation == "gelu":
        torch_out = F.gelu(torch_out)
    elif activation == "gelu_tanh":
        torch_out = F.gelu(torch_out, approximate="tanh")
    elif activation == "silu":
        torch_out = F.silu(torch_out)

    triton_out = gemm_a16w16(
        x,
        w,
        None,
        out_dtype,
        y,
        activation=activation,
        backend=backend,
        kernel_type=kernel_type,
    )

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("layout", ["TT", "NN", "NT"])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("kernel_type", ["bandwidth_bound", "compute_bound"])
def test_gemm_a16_w16_layout(M: int, N: int, K: int, layout, backend, kernel_type):
    if backend == "triton" and kernel_type != "bandwidth_bound":
        pytest.skip("kernel_type only applies to the gluon backend")
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    _skip_if_triton_on_gfx1250(backend)

    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, _, out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, layout=layout, output=False
    )

    torch_out = F.linear(x, w, bias=None)

    triton_out = gemm_a16w16(
        x, w, None, out_dtype, y, backend=backend, kernel_type=kernel_type
    )

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("output", [True, False])
def test_gemm_a16_w16_atomic(M: int, N: int, K: int, output):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, _, _out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=output
    )

    torch_out = F.linear(x, w, bias=None)

    # Accumulation in bf16/fp16 leads to precision loss, cast y to fp32 to prevent that
    if output:
        y = y.to(torch.float32).zero_()
        triton_out = gemm_a16w16_atomic(x, w, torch.float32, y).to(torch.bfloat16)
    else:
        triton_out = gemm_a16w16_atomic(x, w, dtype=torch.float32).to(torch.bfloat16)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("layout", ["TT", "NN", "NT"])
def test_gemm_a16_w16_atomic_layout(M: int, N: int, K: int, layout):
    torch.cuda.empty_cache()  # Helps avoid hangs in large tests

    x, w, _, _out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, layout=layout, output=True
    )

    torch_out = F.linear(x, w, bias=None)

    y = y.to(torch.float32).zero_()
    triton_out = gemm_a16w16_atomic(x, w, torch.float32, y).to(torch.bfloat16)

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("strided", [False, True])
def test_gemm_a16_w16_atomic_accumulate(M: int, N: int, K: int, strided):
    """accumulate=True adds X @ W^T into y in place, like torch.addmm_."""
    torch.cuda.empty_cache()

    x, w, _, _out_dtype, _y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, output=False
    )
    base = torch.randn((M, 2 * N if strided else N), dtype=torch.float32, device="cuda")
    y = base[:, :N]
    torch_out = torch.addmm(y, x.float(), w.float().T)

    triton_out = gemm_a16w16_atomic(x, w, torch.float32, y, accumulate=True)

    assert triton_out.data_ptr() == y.data_ptr()
    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize("layout", ["TN", "TT"])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
def test_gemm_a16w16_persistent_output(M: int, N: int, K: int, layout, output, backend):
    if backend == "gluon" and not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    torch.cuda.empty_cache()

    x, w, _, _out_dtype, y = generate_gemm_a16w16_inputs(
        M, N, K, torch.bfloat16, layout=layout, output=output
    )

    torch_out = F.linear(x, w, bias=None)

    if output:
        triton_out = gemm_a16w16(
            x, w, dtype=torch.bfloat16, y=y, backend=backend, persistent=True
        )
    else:
        triton_out = gemm_a16w16(
            x, w, dtype=torch.bfloat16, backend=backend, persistent=True
        )

    torch.testing.assert_close(triton_out, torch_out, atol=1e-1, rtol=1e-1)


# gfx1250 gluon compute_bound epilogues (REGISTER_STORE): 0 = LDS + TDM store, 1 = register store from the WMMA
# layout, 2 = coalesced register store.  Shapes cover the gpt-oss bucket edges (1024/1025, 2048/2049, 3072/3073) and
# M/N tails that a 256x256 or 128x128 tile leaves partial.
_STORE_CONFIG = {
    "BLOCK_M": 128,
    "BLOCK_N": 128,
    "BLOCK_K": 64,
    "NUM_BUFFERS": 4,
    "num_warps": 4,
    "waves_per_eu": 1,
    "kernel_type": "compute_bound",
}


@pytest.mark.parametrize(
    "M, N, K",
    [
        (1, 128, 2880),
        (97, 2880, 4096),
        (1025, 5120, 2880),
        (2049, 2560, 2880),
        (3073, 2880, 2048),
        (300, 200, 2880),
    ],
)
@pytest.mark.parametrize("register_store", [0, 1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bias", [False, True])
def test_gemm_a16w16_gfx1250_register_store(M, N, K, register_store, dtype, bias):
    if not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=dtype, device="cuda")
    w = torch.randn((N, K), dtype=dtype, device="cuda") / K**0.5
    b = torch.randn((N,), dtype=dtype, device="cuda") if bias else None
    config = dict(_STORE_CONFIG, REGISTER_STORE=register_store)
    out = gemm_a16w16(x, w, bias=b, dtype=dtype, config=config, backend="gluon")
    torch.testing.assert_close(out, F.linear(x, w, bias=b), atol=1e-1, rtol=1e-2)


def test_gemm_a16w16_gfx1250_store_env_override(monkeypatch):
    """AITER_GFX1250_A16W16_STORE overrides the entry's REGISTER_STORE (A/B only) and rejects other values."""
    if not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    torch.manual_seed(0)
    x = torch.randn((257, 2880), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((640, 2880), dtype=torch.bfloat16, device="cuda") / 2880**0.5
    ref = F.linear(x, w)
    for mode in ("0", "1", "2"):
        monkeypatch.setenv("AITER_GFX1250_A16W16_STORE", mode)
        out = gemm_a16w16(
            x, w, config=dict(_STORE_CONFIG, REGISTER_STORE=0), backend="gluon"
        )
        torch.testing.assert_close(out, ref, atol=1e-1, rtol=1e-2)
    monkeypatch.setenv("AITER_GFX1250_A16W16_STORE", "3")
    with pytest.raises(ValueError):
        gemm_a16w16(x, w, config=dict(_STORE_CONFIG), backend="gluon")
    monkeypatch.delenv("AITER_GFX1250_A16W16_STORE")
    with pytest.raises(ValueError):
        gemm_a16w16(x, w, config=dict(_STORE_CONFIG, REGISTER_STORE=5), backend="gluon")


# The gpt-oss entries are tuned on a 96-CU DPX partition and live in configs/gfx1250/96cu/: a 96-CU device must pick
# them (not the 96cu DEFAULT's bandwidth_bound rows), and every bucket they select must compute the right result.
_GPTOSS_SHAPES = [
    (5120, 2880),  # TP1 QKV
    (2880, 4096),  # TP1 O
    (128, 2880),  # router
    (2560, 2880),  # TP2 QKV
    (2880, 2048),  # TP2 O
    (201088, 2880),  # LM head, TP1
    (100544, 2880),  # LM head, TP2 (partial last N tile)
]


@pytest.mark.parametrize(
    "N, K", _GPTOSS_SHAPES[:5]
)  # the LM heads are tuned for decode M only
def test_gemm_a16w16_gfx1250_gptoss_96cu_lookup(N, K, monkeypatch):
    """Read configs/gfx1250/96cu. The arch is pinned because this only reads
    JSON: on an MI35X runner get_arch() is gfx950 and the gluon default there
    does not exist."""
    from aiter.ops.triton.utils import gemm_config_utils
    from aiter.ops.triton.utils._triton import arch_info

    monkeypatch.setattr(arch_info, "get_arch", lambda: "gfx1250")
    # The loader cache key does not include the arch.
    gemm_config_utils._get_gemm_config_cached.cache_clear()
    try:
        for M in (4096, 8192, 16384, 32768):
            config, tuned = gemm_config_utils.get_gemm_config(
                "GEMM-A16W16", M, N, K, backend="gluon", cu_num=96
            )
            assert tuned and config["kernel_type"] == "compute_bound", (
                M,
                N,
                K,
                config,
            )
    finally:
        gemm_config_utils._get_gemm_config_cached.cache_clear()


@pytest.mark.parametrize("N, K", _GPTOSS_SHAPES)
@pytest.mark.parametrize("M", [1, 64, 1000, 2049, 3073, 8192])
def test_gemm_a16w16_gfx1250_gptoss_tuned(M, N, K):
    if not is_gluon_supported():
        pytest.skip("Gluon not supported on this architecture")
    torch.manual_seed(0)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    w = torch.randn((N, K), dtype=torch.bfloat16, device="cuda") / K**0.5
    torch.testing.assert_close(
        gemm_a16w16(x, w, backend="gluon"), F.linear(x, w), atol=1e-1, rtol=1e-2
    )
