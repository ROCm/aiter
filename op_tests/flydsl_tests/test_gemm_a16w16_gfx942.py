# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Oracle-gated correctness tests for the native gfx942 (CDNA3) bf16 GEMM.

Compares the kernel against an f32 matmul oracle (bf16 inputs upcast to f32)
within bf16 tolerance, across every ticket regime -- HC down (large-K/narrow-N),
HC up (short-K/wide-N), decode/GEMV (tiny M), square compute-bound -- plus
non-divisible-M boundary shapes. Exercised both directly (``gemm_a16w16``) and
through the public ``flydsl_hgemm`` dispatch (per-shape auto-config), with the
fp32-output and explicit split-K paths spot-checked.

Skips unless running on real gfx942 hardware.
"""

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
from aiter.ops.flydsl.kernels.gemm_a16w16_gfx942 import gemm_a16w16

RTOL = ATOL = 1e-2


def _require_gfx942():
    gfx = get_gfx()
    if gfx != "gfx942":
        pytest.skip(f"requires gfx942, got {gfx!r}")


def _inputs(m, n, k):
    torch.manual_seed(0)
    x = (torch.randn(m, k, device="cuda") * k**-0.5).to(torch.bfloat16)
    w = (torch.randn(n, k, device="cuda") * k**-0.5).to(torch.bfloat16)  # [N, K]
    ref = x.float() @ w.float().t()  # f32 oracle
    return x, w, ref


def _assert_close(y, ref, msg):
    y = y.float()
    rel = (y - ref).abs().max().item() / (ref.abs().max().item() + 1e-9)
    assert (
        torch.allclose(y, ref, rtol=RTOL, atol=ATOL) or rel < 2e-2
    ), f"{msg}: relerr={rel:.3e} exceeds bf16 tolerance"


# ticket regimes: (M, N, K, regime)
SHAPES = [
    (1, 384, 10240, "hc-down"),
    (4, 384, 10240, "hc-down"),
    (512, 384, 10240, "hc-down"),
    (1024, 384, 10240, "hc-down-specialized"),
    (2048, 384, 10240, "hc-down"),
    (8192, 384, 10240, "hc-down"),
    (1, 10240, 320, "hc-up"),
    (4, 10240, 320, "hc-up"),
    (512, 10240, 320, "hc-up"),
    (2048, 10240, 320, "hc-up"),
    (8192, 10240, 320, "hc-up"),
    (1, 4096, 4096, "decode"),
    (4096, 4096, 4096, "square"),
    (8192, 8192, 8192, "square"),
    # non-divisible-M boundary cases
    (100, 384, 10240, "boundary-M"),
    (1000, 10240, 320, "boundary-M"),
    (4095, 4096, 4096, "boundary-M"),
    # non-divisible-N: the real gated-residual down projection (N=336)
    (1024, 336, 10240, "boundary-N"),
    (4096, 336, 10240, "boundary-N"),
    (1000, 336, 10240, "boundary-MN"),
]


@pytest.mark.parametrize(
    "m,n,k,regime", SHAPES, ids=[f"{r}-M{m}" for m, n, k, r in SHAPES]
)
def test_gemm_a16w16_gfx942(m, n, k, regime):
    """Direct kernel call vs f32 oracle (kernel's own default config)."""
    _require_gfx942()
    x, w, ref = _inputs(m, n, k)
    y = gemm_a16w16(x, w.t(), layout="nt", out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    _assert_close(y, ref, f"{regime} M{m} N{n} K{k}")


@pytest.mark.parametrize(
    "m,n,k,regime", SHAPES, ids=[f"{r}-M{m}" for m, n, k, r in SHAPES]
)
def test_flydsl_hgemm_gfx942_autoconfig(m, n, k, regime):
    """Public dispatch with per-shape auto-config vs f32 oracle.

    ``flydsl_hgemm`` takes AITER's B[N, K] convention directly (no transpose).
    """
    _require_gfx942()
    x, w, ref = _inputs(m, n, k)
    y = flydsl_hgemm(x, w, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    _assert_close(y, ref, f"{regime} M{m} N{n} K{k} (dispatch)")


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_gemm_a16w16_gfx942_split_k(out_dtype):
    """Explicit split-K: fp32 output uses native atomics; bf16 goes via f32 accum."""
    _require_gfx942()
    m, n, k = 512, 384, 10240
    x, w, ref = _inputs(m, n, k)
    y = gemm_a16w16(
        x,
        w.t(),
        layout="nt",
        out_dtype=out_dtype,
        user_kwargs={
            "block_m": 32,
            "block_n": 32,
            "block_k": 64,
            "stages": 2,
            "split_k": 4,
            "m_waves": 2,
            "n_waves": 2,
        },
    )
    torch.cuda.synchronize()
    _assert_close(y, ref, f"split-K out={out_dtype}")


@pytest.mark.parametrize("split_k", [2, 3, 4])
def test_gemm_a16w16_gfx942_buffered_split_k(split_k):
    """The 128-bit global-to-register load path honors each split's K offset."""
    _require_gfx942()
    m, n, k = 128, 128, 1536
    x, w, ref = _inputs(m, n, k)
    y = gemm_a16w16(
        x,
        w.t(),
        layout="nt",
        out_dtype=torch.bfloat16,
        user_kwargs={
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "stages": 1,
            "split_k": split_k,
            "m_waves": 2,
            "n_waves": 2,
            "buffered_load": True,
            "async_load_bytes": 16,
        },
    )
    torch.cuda.synchronize()
    _assert_close(y, ref, f"buffered split-K={split_k}")


@pytest.mark.parametrize("split_k", [1, 3])
@pytest.mark.parametrize("m,n", [(1000, 384), (1000, 336), (960, 344)])
def test_gemm_a16w16_gfx942_pingpong_edge_tiles(m, n, split_k):
    """Pingpong edge tiles: bounded buffered loads, and direct stores past M."""
    _require_gfx942()
    k = 4096
    x, w, ref = _inputs(m, n, k)
    out = torch.full((m + 64, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    y = gemm_a16w16(
        x,
        w.t(),
        out=out[:m],
        layout="nt",
        user_kwargs={
            "block_m": 64,
            "block_n": 64,
            "block_k": 128,
            "stages": 1,
            "split_k": split_k,
            "m_waves": 2,
            "n_waves": 2,
            "buffered_load": True,
            "async_load_bytes": 16,
            "pingpong": True,
            "split_k_reduce": split_k > 1,
            "direct_store": n % 64 == 0,
        },
    )
    torch.cuda.synchronize()
    _assert_close(y, ref, f"pingpong edge M{m} N{n} split_k={split_k}")
    assert torch.isnan(out[m:]).all(), "stores past M overwrote the guard rows"


@pytest.mark.parametrize(
    "m,n,k,tile",
    [
        (
            1000,
            336,
            10240,
            {"block_m": 64, "block_n": 64, "block_k": 128, "split_k": 3},
        ),
        (960, 344, 4096, {"block_m": 128, "block_n": 96, "block_k": 128, "split_k": 2}),
        (2048, 4096, 1024, {"block_m": 256, "block_n": 128, "block_k": 64}),
        (1024, 768, 128, {"block_m": 128, "block_n": 192, "block_k": 64}),
        (
            1000,
            1024,
            2048,
            {"block_m": 64, "block_n": 128, "block_k": 128, "k_waves": 2},
        ),
        (1000, 10240, 320, {"block_m": 128, "block_n": 128, "block_k": 64}),
        (
            1000,
            336,
            10240,
            {
                "block_m": 128,
                "block_n": 112,
                "block_k": 128,
                "m_waves": 2,
                "n_waves": 1,
                "k_waves": 2,
                "split_k": 3,
                "direct_store": True,
            },
        ),
        (250, 384, 10240, {"block_m": 32, "block_n": 96, "block_k": 128, "split_k": 8}),
        (
            500,
            336,
            10240,
            {
                "block_m": 32,
                "block_n": 128,
                "block_k": 128,
                "m_waves": 1,
                "n_waves": 4,
                "split_k": 6,
            },
        ),
        (
            1000,
            336,
            10240,
            {
                "block_m": 64,
                "block_n": 64,
                "block_k": 128,
                "m_waves": 2,
                "n_waves": 1,
                "k_waves": 2,
                "split_k": 3,
            },
        ),
        # C tile over 64 KB: staged through LDS in row chunks.
        (
            1000,
            10240,
            320,
            {"block_m": 256, "block_n": 160, "block_k": 64, "m_waves": 4, "n_waves": 1},
        ),
        (
            1000,
            1000,
            1024,
            {
                "block_m": 256,
                "block_n": 160,
                "block_k": 64,
                "m_waves": 4,
                "n_waves": 1,
                "split_k": 2,
            },
        ),
    ],
)
def test_gemm_a16w16_gfx942_stream_operands(m, n, k, tile):
    """Streamed-operand pingpong: edge tiles, split-K, k_waves and K=2 tiles."""
    _require_gfx942()
    x, w, ref = _inputs(m, n, k)
    out = torch.full((m + 64, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    split_k = tile.get("split_k", 1)
    cfg = {
        "stages": 1,
        "m_waves": 2,
        "n_waves": 2,
        "buffered_load": True,
        "async_load_bytes": 16,
        "pingpong": True,
        "stream_operands": True,
        "split_k_reduce": split_k > 1,
        "direct_store": False,
    }
    cfg.update(tile)
    if cfg.get("k_waves", 1) > 1 and "m_waves" not in tile:
        cfg["m_waves"] = 1
    y = gemm_a16w16(x, w.t(), out=out[:m], layout="nt", user_kwargs=cfg)
    torch.cuda.synchronize()
    _assert_close(y, ref, f"stream M{m} N{n} K{k} {tile}")
    assert torch.isnan(out[m:]).all(), "stores past M overwrote the guard rows"


def test_gemm_a16w16_gfx942_tuned_table(tmp_path, monkeypatch):
    """A tuned-table row overrides the heuristic; misses and the env switch use it."""
    _require_gfx942()
    import aiter.ops.flydsl.kernels.gemm_a16w16_gfx942 as g

    m, n, k = 1000, 344, 4096
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    row = {
        "gfx": props.gcnArchName.split(":")[0],
        "cu_num": props.multi_processor_count,
        "M": m,
        "N": n,
        "K": k,
        "out_dtype": "bf16",
        "block_m": 64,
        "block_n": 64,
        "block_k": 128,
        "stages": 1,
        "split_k": 2,
        "m_waves": 2,
        "n_waves": 1,
        "k_waves": 2,
        "group_m": 4,
        "async_load_bytes": 16,
        "buffered_load": True,
        "direct_store": False,
        "pingpong": True,
        "split_k_reduce": True,
        "prefetch_depth": 1,
        "stream_operands": True,
    }
    table = tmp_path / "table.csv"
    table.write_text(
        ",".join(row) + "\n" + ",".join(str(v) for v in row.values()) + "\n"
    )
    monkeypatch.setenv("AITER_CONFIG_FLYDSL_GEMM_A16W16_GFX942", str(table))
    g.reload_tuned_table()
    try:
        expected = {key: row[key] for key in g.TUNED_CONFIG_COLUMNS}
        assert g.gfx942_default_config(m, n, k) == expected
        assert g.gfx942_tuned_config(m + 8, n, k) is None
        assert g.gfx942_tuned_config(m, n, k, torch.float32) is None
        monkeypatch.setenv("AITER_FLYDSL_GEMM_GFX942_HEURISTIC", "1")
        assert g.gfx942_default_config(m, n, k) == g.gfx942_auto_config(m, n, k)
        monkeypatch.delenv("AITER_FLYDSL_GEMM_GFX942_HEURISTIC")
        x, w, ref = _inputs(m, n, k)
        y = gemm_a16w16(x, w.t(), layout="nt")
        torch.cuda.synchronize()
        _assert_close(y, ref, f"tuned-table M{m} N{n} K{k}")
    finally:
        monkeypatch.delenv("AITER_CONFIG_FLYDSL_GEMM_A16W16_GFX942")
        g.reload_tuned_table()


def test_gemm_a16w16_gfx942_fp32_out():
    """fp32 output path on a compute-bound square shape."""
    _require_gfx942()
    m, n, k = 4096, 4096, 4096
    x, w, ref = _inputs(m, n, k)
    y = gemm_a16w16(x, w.t(), layout="nt", out_dtype=torch.float32)
    torch.cuda.synchronize()
    _assert_close(y, ref, f"fp32-out M{m} N{n} K{k}")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
