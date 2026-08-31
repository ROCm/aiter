# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F

_KERNEL_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "aiter",
        "ops",
        "flydsl",
        "gemm_a16w16_gfx1250.py",
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
        "FlyDSL a16w16 tests require gfx1250 and the flydsl package.",
        allow_module_level=True,
    )


def _load_kernel():
    """Load the kernel module by file path to skip aiter.ops.flydsl.__init__."""
    spec = importlib.util.spec_from_file_location(
        "_flydsl_a16w16_wrapper", _KERNEL_PATH
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.gemm_a16w16


gemm_a16w16 = _load_kernel()


def _generate_inputs(M, N, K, dtype, layout="TN", output=False, bias=False):
    if layout[0] == "T":
        x = torch.randn((M, K), dtype=dtype, device="cuda")
    else:
        x = torch.randn((K, M), dtype=dtype, device="cuda").T

    if layout[1] == "T":
        w = torch.randn((K, N), dtype=dtype, device="cuda").T
    else:
        w = torch.randn((N, K), dtype=dtype, device="cuda")

    bias_tensor = None
    if bias:
        bias_tensor = torch.randn((N,), dtype=dtype, device="cuda")

    y = torch.empty((M, N), dtype=dtype, device="cuda") if output else None
    return x, w, bias_tensor, y


def get_x_vals():
    return [
        # Aligned
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        # Multi-block
        (128, 256, 512),
        (256, 512, 256),
        # Asymmetric
        (32, 256, 128),
        (256, 32, 128),
        # Large K (drives the flat main loop + drain)
        (128, 128, 1024),
        (1024, 128, 128),
    ]


def get_fewer_x_vals():
    return [
        (64, 64, 64),
        (128, 256, 512),
        (256, 512, 256),
        (128, 128, 1024),
        (1024, 128, 128),
    ]


@pytest.mark.parametrize("M, N, K", get_x_vals())
def test_gemm_a16_w16(M, N, K):
    torch.cuda.empty_cache()
    x, w, _, _ = _generate_inputs(M, N, K, torch.bfloat16)

    torch_out = F.linear(x, w, bias=None)
    kernel_out = gemm_a16w16(x, w, dtype=torch.bfloat16, num_buffers=3)

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("num_buffers", [2, 3])
def test_gemm_a16_w16_compute_bound(M, N, K, num_buffers):
    torch.cuda.empty_cache()
    x, w, _, _ = _generate_inputs(M, N, K, torch.bfloat16)

    torch_out = F.linear(x, w, bias=None)
    kernel_out = gemm_a16w16(
        x, w, dtype=torch.bfloat16, num_buffers=num_buffers, variant="compute_bound"
    )

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("activation", ["gelu", "gelu_tanh", "silu", "silu_exp2"])
@pytest.mark.parametrize("M, N, K", get_fewer_x_vals())
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("output", [True, False])
def test_gemm_a16_w16_activation(M, N, K, dtype, output, activation):
    torch.cuda.empty_cache()
    x, w, _, y = _generate_inputs(M, N, K, dtype, output=output)

    torch_out = F.linear(x, w, bias=None)
    if activation == "gelu":
        torch_out = F.gelu(torch_out)
    elif activation == "gelu_tanh":
        torch_out = F.gelu(torch_out, approximate="tanh")
    elif activation in ("silu", "silu_exp2"):
        torch_out = F.silu(torch_out)

    kernel_out = gemm_a16w16(
        x, w, dtype=dtype, y=y, activation=activation, num_buffers=3
    )

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("layout", ["TN", "TT", "NN", "NT"])
def test_gemm_a16_w16_layout(M, N, K, layout):
    torch.cuda.empty_cache()
    x, w, _, _ = _generate_inputs(M, N, K, torch.bfloat16, layout=layout)

    torch_out = F.linear(x, w, bias=None)
    kernel_out = gemm_a16w16(x, w, dtype=torch.bfloat16, num_buffers=3)

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-1)


# 32x32 output tile, deep K (tile_k=128); tile_n=32 needs n_warp=2, K multiple of 128.
@pytest.mark.parametrize(
    "M, N, K", [(32, 32, 256), (128, 128, 256), (256, 256, 512), (64, 64, 1024)]
)
def test_gemm_a16_w16_tile_32x32x128(M, N, K):
    torch.cuda.empty_cache()
    x, w, _, _ = _generate_inputs(M, N, K, torch.bfloat16)

    torch_out = F.linear(x, w, bias=None)
    kernel_out = gemm_a16w16(
        x,
        w,
        dtype=torch.bfloat16,
        num_buffers=3,
        tile_m=32,
        tile_n=32,
        tile_k=128,
        m_warp=2,
        n_warp=2,
    )

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-2)


# Split-K: partition K across split_k grid-z workgroups, atomic-add into a pre-zeroed C.
def _splitk_configs():
    # (M, N, K, tile_m, tile_n, tile_k, m_warp, n_warp, num_buffers, split_k) — all valid.
    return [
        (128, 128, 512, 64, 64, 128, 2, 2, 2, 2),  # 4 tiles, 2/split
        (128, 128, 512, 64, 64, 128, 2, 2, 2, 4),  # 4 tiles, 1/split
        (64, 64, 768, 64, 64, 128, 2, 2, 2, 3),  # 6 tiles, 2/split
        (64, 64, 768, 64, 64, 128, 2, 2, 2, 6),  # 6 tiles, 1/split
        (128, 256, 1024, 64, 64, 128, 2, 2, 2, 4),  # 8 tiles, 2/split
        (256, 128, 1024, 32, 32, 128, 2, 2, 2, 8),  # 8 tiles, 1/split (max split)
        (128, 128, 1024, 64, 64, 128, 2, 2, 3, 4),  # nb=3, 2/split
    ]


@pytest.mark.parametrize("M,N,K,tm,tn,tk,mw,nw,nb,sk", _splitk_configs())
def test_gemm_a16_w16_split_k(M, N, K, tm, tn, tk, mw, nw, nb, sk):
    """Primary split-K correctness: fp32 output (lossless) -> split_k=N == split_k=1."""
    torch.cuda.empty_cache()
    x, w, _, _ = _generate_inputs(M, N, K, torch.bfloat16)

    common = {
        "dtype": torch.float32,
        "tile_m": tm,
        "tile_n": tn,
        "tile_k": tk,
        "m_warp": mw,
        "n_warp": nw,
        "num_buffers": nb,
    }
    out_single = gemm_a16w16(x, w, split_k=1, **common).clone()
    out_split = gemm_a16w16(x, w, split_k=sk, **common).clone()

    torch.testing.assert_close(out_split, out_single, atol=1e-2, rtol=1e-3)


@pytest.mark.parametrize("M,N,K,tm,tn,tk,mw,nw,nb,sk", _splitk_configs())
def test_gemm_a16_w16_split_k_bf16(M, N, K, tm, tn, tk, mw, nw, nb, sk):
    """bf16-output split-K at O(1) magnitudes (scaled inputs) must match split_k=1."""
    torch.cuda.empty_cache()
    x, w, _, _ = _generate_inputs(M, N, K, torch.bfloat16)
    x = (x.float() * 0.1).to(torch.bfloat16)
    w = (w.float() * 0.1).to(torch.bfloat16)

    common = {
        "dtype": torch.bfloat16,
        "tile_m": tm,
        "tile_n": tn,
        "tile_k": tk,
        "m_warp": mw,
        "n_warp": nw,
        "num_buffers": nb,
    }
    out_single = gemm_a16w16(x, w, split_k=1, **common).clone()
    out_split = gemm_a16w16(x, w, split_k=sk, **common).clone()

    torch.testing.assert_close(out_split, out_single, atol=1e-1, rtol=2e-2)


@pytest.mark.parametrize("sk", [2, 4])
def test_gemm_a16_w16_split_k_output_buffer_zeroed(sk):
    """split_k>1 atomic-ADDS into C; the wrapper must zero it each call (no doubling)."""
    M, N, K = 128, 128, 512
    torch.cuda.empty_cache()
    x, w, _, _ = _generate_inputs(M, N, K, torch.bfloat16)
    y = torch.empty((M, N), dtype=torch.float32, device="cuda")

    common = {
        "dtype": torch.float32,
        "y": y,
        "tile_m": 64,
        "tile_n": 64,
        "tile_k": 128,
        "m_warp": 2,
        "n_warp": 2,
        "num_buffers": 2,
        "split_k": sk,
    }
    out1 = gemm_a16w16(x, w, **common).clone()
    out2 = gemm_a16w16(x, w, **common).clone()  # reuse y -> must re-zero, not 2x

    torch.testing.assert_close(out2, out1, atol=1e-2, rtol=1e-3)


def _oob_configs():
    return [
        (100, 100, 256, 128, 128, 32, 2, 4, 3),
        (129, 257, 512, 128, 128, 32, 2, 4, 3),
        (65, 190, 256, 128, 128, 32, 2, 4, 3),
        (250, 120, 512, 64, 64, 128, 2, 2, 2),
        (100, 128, 256, 128, 128, 32, 2, 4, 3),
        (128, 100, 256, 128, 128, 32, 2, 4, 3),
        (33, 65, 256, 32, 32, 128, 2, 2, 3),
    ]


@pytest.mark.parametrize("M,N,K,tm,tn,tk,mw,nw,nb", _oob_configs())
def test_gemm_a16_w16_oob_tile_edges(M, N, K, tm, tn, tk, mw, nw, nb):
    torch.cuda.empty_cache()
    x, w, _, _ = _generate_inputs(M, N, K, torch.bfloat16)

    torch_out = F.linear(x, w, bias=None)
    kernel_out = gemm_a16w16(
        x,
        w,
        dtype=torch.bfloat16,
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        m_warp=mw,
        n_warp=nw,
        num_buffers=nb,
    )

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize(
    "M,N,K,tm,tn,tk,mw,nw,nb",
    [
        (100, 128, 256, 128, 128, 32, 2, 4, 3),
        (65, 64, 512, 64, 64, 128, 2, 2, 2),
    ],
)
def test_gemm_a16_w16_oob_guard_region(M, N, K, tm, tn, tk, mw, nw, nb):
    torch.cuda.empty_cache()
    x, w, _, _ = _generate_inputs(M, N, K, torch.bfloat16)
    sentinel = 777.0
    parent = torch.full((M + tm, N), sentinel, dtype=torch.bfloat16, device="cuda")
    y = parent[:M]

    torch_out = F.linear(x, w, bias=None)
    kernel_out = gemm_a16w16(
        x,
        w,
        dtype=torch.bfloat16,
        y=y,
        tile_m=tm,
        tile_n=tn,
        tile_k=tk,
        m_warp=mw,
        n_warp=nw,
        num_buffers=nb,
    )

    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-2)
    assert torch.all(parent[M:] == sentinel)


@pytest.mark.parametrize("K, tile_k", [(48, 32), (100, 32), (130, 64), (1000, 64)])
@pytest.mark.parametrize("layout", ["TN", "NN", "TT", "NT"])
def test_gemm_a16_w16_ragged_k(K, tile_k, layout, monkeypatch):
    torch.cuda.empty_cache()
    M, N = 128, 192
    x, w, _, _ = _generate_inputs(M, N, K, torch.bfloat16, layout=layout)

    def _no_pad(*args, **kwargs):
        raise AssertionError("host-side pad/copy in gemm_a16w16 wrapper")

    monkeypatch.setattr(F, "pad", _no_pad)
    torch_out = (x.float() @ w.float().T).to(torch.bfloat16)
    kernel_out = gemm_a16w16(x, w, dtype=torch.bfloat16, tile_k=tile_k)
    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-2)


def test_gemm_a16_w16_ragged_k_split_k():
    torch.cuda.empty_cache()
    M, N, K, sk = 64, 64, 576, 2
    x, w, _, _ = _generate_inputs(M, N, K, torch.bfloat16)
    torch_out = x.float() @ w.float().T
    kernel_out = gemm_a16w16(
        x,
        w,
        dtype=torch.float32,
        tile_m=64,
        tile_n=64,
        tile_k=64,
        m_warp=2,
        n_warp=2,
        num_buffers=2,
        split_k=sk,
    )
    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-2)


@pytest.mark.parametrize("N", [100, 101, 190, 257])
@pytest.mark.parametrize("otype", [torch.bfloat16, torch.float32])
def test_gemm_a16_w16_ragged_n_direct_store(N, otype):
    torch.cuda.empty_cache()
    M, K = 129, 256
    x, w, _, _ = _generate_inputs(M, N, K, torch.bfloat16)
    sentinel = 777.0
    parent = torch.full((M + 32, N + 64), sentinel, dtype=otype, device="cuda")
    y = parent[:M, :N]

    torch_out = (x.float() @ w.float().T).to(otype)
    out = gemm_a16w16(x, w, dtype=otype, y=y)

    assert out is y
    torch.testing.assert_close(out, torch_out, atol=1e-1, rtol=1e-2)
    assert torch.all(parent[:, N:] == sentinel)
    assert torch.all(parent[M:] == sentinel)


@pytest.mark.parametrize("N", [100, 190])
def test_gemm_a16_w16_ragged_n_split_k(N):
    torch.cuda.empty_cache()
    M, K, sk = 64, 512, 2
    x, w, _, _ = _generate_inputs(M, N, K, torch.bfloat16)
    sentinel = 777.0
    parent = torch.full((M + 8, N + 32), sentinel, dtype=torch.float32, device="cuda")
    y = parent[:M, :N]

    torch_out = x.float() @ w.float().T
    out = gemm_a16w16(
        x,
        w,
        dtype=torch.float32,
        y=y,
        tile_m=64,
        tile_n=64,
        tile_k=64,
        m_warp=2,
        n_warp=2,
        num_buffers=2,
        split_k=sk,
    )

    assert out is y
    torch.testing.assert_close(out, torch_out, atol=1e-1, rtol=1e-2)
    assert torch.all(parent[:, N:] == sentinel)
    assert torch.all(parent[M:] == sentinel)


@pytest.mark.parametrize("M, N, K", [(100, 190, 256), (65, 257, 512)])
def test_gemm_a16_w16_ragged_n_bias_activation(M, N, K):
    torch.cuda.empty_cache()
    x, w, bias, _ = _generate_inputs(M, N, K, torch.bfloat16, bias=True)

    torch_out = F.relu(F.linear(x, w, bias=bias))
    kernel_out = gemm_a16w16(x, w, bias=bias, dtype=torch.bfloat16, activation="relu")
    torch.testing.assert_close(kernel_out, torch_out, atol=1e-1, rtol=1e-2)
