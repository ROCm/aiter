# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Oracle-gated correctness tests for the native gfx942 (CDNA3) bf16 GEMM.

Compares the kernel against an f32 matmul oracle (bf16 inputs upcast to f32)
within bf16 tolerance. Shapes cover down projections (large K, narrow N), up
projections (short K, wide N), decode (tiny M), square compute-bound GEMMs and
M/N boundary tiles, through both ``gemm_a16w16`` and the public
``flydsl_hgemm`` dispatch (per-shape default config). Explicit configs cover every
main-loop pipeline, split-K mode and output dtype.

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


OUT_DTYPES = [torch.bfloat16, torch.float32]
OUT_DTYPE_IDS = ["bf16", "fp32"]


def _guarded_out(m, n, dtype):
    """Output with NaN guard rows past M; the caller passes ``out[:m]``."""
    return torch.full((m + 64, n), float("nan"), dtype=dtype, device="cuda")


# (M, N, K, regime)
SHAPES = [
    (1, 384, 10240, "down"),
    (4, 384, 10240, "down"),
    (512, 384, 10240, "down"),
    (1024, 384, 10240, "down"),
    (2048, 384, 10240, "down"),
    (8192, 384, 10240, "down"),
    (1, 10240, 320, "up"),
    (4, 10240, 320, "up"),
    (512, 10240, 320, "up"),
    (2048, 10240, 320, "up"),
    (8192, 10240, 320, "up"),
    (1, 4096, 4096, "decode"),
    (4096, 4096, 4096, "square"),
    (8192, 8192, 8192, "square"),
    # non-divisible-M boundary cases
    (100, 384, 10240, "boundary-M"),
    (1000, 10240, 320, "boundary-M"),
    (4095, 4096, 4096, "boundary-M"),
    # non-divisible-N: N=336 and N=344 are not multiples of 64
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
    """Public dispatch with the per-shape default config vs f32 oracle.

    ``flydsl_hgemm`` takes AITER's B[N, K] convention directly (no transpose).
    """
    _require_gfx942()
    x, w, ref = _inputs(m, n, k)
    y = flydsl_hgemm(x, w, out_dtype=torch.bfloat16)
    torch.cuda.synchronize()
    _assert_close(y, ref, f"{regime} M{m} N{n} K{k} (dispatch)")


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"k_waves": 2},
        {
            "block_m": 128,
            "block_n": 128,
            "block_k": 64,
            "stages": 2,
            "split_k": 1,
            "m_waves": 2,
            "n_waves": 2,
            "group_m": 0,
        },
    ],
    ids=["unset", "k_waves-only", "explicit-default-tile"],
)
def test_flydsl_hgemm_gfx942_forwards_overrides(overrides, monkeypatch):
    """flydsl_hgemm passes exactly the tile arguments it was given to the kernel."""
    _require_gfx942()
    import aiter.ops.flydsl.kernels.gemm_a16w16_gfx942 as g

    compiled = {}
    make_param = g.make_gemm_a16w16_gfx942_param

    def recording_make_param(**kwargs):
        compiled.update(kwargs)
        return make_param(**kwargs)

    monkeypatch.setattr(g, "make_gemm_a16w16_gfx942_param", recording_make_param)
    m, n, k = 1024, 1024, 1024
    x, w, ref = _inputs(m, n, k)
    y = flydsl_hgemm(x, w, **overrides)
    torch.cuda.synchronize()
    _assert_close(y, ref, f"flydsl_hgemm {overrides}")
    if overrides.keys() & g._STRUCTURAL_KEYS:
        assert {key: compiled[key] for key in overrides} == overrides
        assert not compiled.get("buffered_load", False)
    else:
        expected = {**g.gfx942_default_config(m, n, k), **overrides}
        assert {key: compiled[key] for key in expected} == expected


@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_gemm_a16w16_gfx942_split_k(out_dtype):
    """Explicit async split-K: fp32 output uses native atomics; bf16 goes via f32 accum."""
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


@pytest.mark.parametrize("stages", [1, 2])
@pytest.mark.parametrize("split_k", [2, 3, 4])
def test_gemm_a16w16_gfx942_buffered_split_k(split_k, stages):
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
            "stages": stages,
            "split_k": split_k,
            "m_waves": 2,
            "n_waves": 2,
            "buffered_load": True,
            "async_load_bytes": 16,
        },
    )
    torch.cuda.synchronize()
    _assert_close(y, ref, f"buffered stages={stages} split-K={split_k}")


@pytest.mark.parametrize("out_dtype", OUT_DTYPES, ids=OUT_DTYPE_IDS)
@pytest.mark.parametrize("split_k", [1, 3])
@pytest.mark.parametrize("m,n", [(1000, 384), (1000, 336), (960, 344)])
def test_gemm_a16w16_gfx942_pingpong_edge_tiles(m, n, split_k, out_dtype):
    """Pingpong edge tiles: bounded buffered loads, and direct stores past M."""
    _require_gfx942()
    k = 4096
    x, w, ref = _inputs(m, n, k)
    out = _guarded_out(m, n, out_dtype)
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
    _assert_close(y, ref, f"pingpong edge M{m} N{n} split_k={split_k} {out_dtype}")
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
@pytest.mark.parametrize("out_dtype", OUT_DTYPES, ids=OUT_DTYPE_IDS)
def test_gemm_a16w16_gfx942_pingpong(m, n, k, tile, out_dtype):
    """Pingpong: edge tiles, split-K, k_waves and two-K-tile shapes."""
    _require_gfx942()
    x, w, ref = _inputs(m, n, k)
    out = _guarded_out(m, n, out_dtype)
    split_k = tile.get("split_k", 1)
    cfg = {
        "stages": 1,
        "m_waves": 2,
        "n_waves": 2,
        "buffered_load": True,
        "async_load_bytes": 16,
        "pingpong": True,
        "split_k_reduce": split_k > 1,
        "direct_store": False,
    }
    cfg.update(tile)
    if cfg.get("k_waves", 1) > 1 and "m_waves" not in tile:
        cfg["m_waves"] = 1
    y = gemm_a16w16(x, w.t(), out=out[:m], layout="nt", user_kwargs=cfg)
    torch.cuda.synchronize()
    _assert_close(y, ref, f"pingpong M{m} N{n} K{k} {tile} {out_dtype}")
    assert torch.isnan(out[m:]).all(), "stores past M overwrote the guard rows"


@pytest.mark.parametrize("out_dtype", OUT_DTYPES, ids=OUT_DTYPE_IDS)
@pytest.mark.parametrize("stages", [2, 3, 4])
@pytest.mark.parametrize(
    "m,n,k,split_k,split_k_reduce",
    [
        (768, 768, 128, 1, False),  # fewer K tiles than stages
        (768, 768, 192, 1, False),
        (1000, 336, 1024, 1, False),  # M and N edge tiles
        (500, 384, 2048, 2, False),  # atomic split-K
        (1000, 336, 4096, 3, True),  # workspace split-K
    ],
)
def test_gemm_a16w16_gfx942_async_stages(
    m, n, k, split_k, split_k_reduce, stages, out_dtype
):
    """Async DMA pipeline; stages >= 3 adds the register double buffer."""
    _require_gfx942()
    x, w, ref = _inputs(m, n, k)
    out = _guarded_out(m, n, out_dtype)
    cfg = {
        "block_m": 64,
        "block_n": 64,
        "block_k": 64,
        "stages": stages,
        "split_k": split_k,
        "split_k_reduce": split_k_reduce,
        "m_waves": 2,
        "n_waves": 2,
        "async_load_bytes": 4,
    }
    if k // 64 // split_k < stages - 1:
        with pytest.raises(ValueError, match="needs at least"):
            gemm_a16w16(x, w.t(), out=out[:m], layout="nt", user_kwargs=cfg)
        return
    y = gemm_a16w16(x, w.t(), out=out[:m], layout="nt", user_kwargs=cfg)
    torch.cuda.synchronize()
    _assert_close(y, ref, f"async stages={stages} M{m} N{n} K{k} split_k={split_k}")
    assert torch.isnan(out[m:]).all(), "stores past M overwrote the guard rows"


def test_gemm_a16w16_gfx942_tuned_table(tmp_path, monkeypatch):
    """A tuned-table row overrides the fallback; misses and the env switch use it."""
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
        monkeypatch.setenv("AITER_FLYDSL_GEMM_GFX942_SKIP_TABLE", "1")
        assert g.gfx942_default_config(m, n, k) == g.gfx942_fallback_config(m, n, k)
        monkeypatch.delenv("AITER_FLYDSL_GEMM_GFX942_SKIP_TABLE")
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


@pytest.mark.parametrize(
    "store",
    [
        {"direct_store": True},
        {"direct_store": False},
        {"split_k": 4, "split_k_reduce": True},
        {"split_k": 4, "pingpong": False},  # atomic
    ],
    ids=["direct", "cshuffle", "workspace-split-k", "atomic-split-k"],
)
def test_gemm_a16w16_gfx942_fp32_out_precision(store):
    """fp32 output keeps fp32 accuracy on every store path (no bf16 rounding)."""
    _require_gfx942()
    m, n, k = 512, 512, 4096
    x, w, ref = _inputs(m, n, k)
    cfg = {
        "block_m": 64,
        "block_n": 64,
        "block_k": 128,
        "stages": 1,
        "split_k": 1,
        "m_waves": 2,
        "n_waves": 2,
        "k_waves": 1,
        "buffered_load": True,
        "async_load_bytes": 16,
        "pingpong": True,
        **store,
    }
    y = gemm_a16w16(x, w.t(), layout="nt", out_dtype=torch.float32, user_kwargs=cfg)
    torch.cuda.synchronize()
    rel = ((y - ref).abs().max() / ref.abs().max()).item()
    assert rel < 1e-4, f"fp32 output {store}: relerr={rel:.3e} (bf16 rounding?)"


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 384, 320),
        (300, 1024, 320),
        (100, 7168, 1024),
        (6000, 2560, 320),
        (77, 200, 64),
    ],
)
def test_gemm_a16w16_gfx942_fallback(m, n, k, monkeypatch):
    """Cost-model fallback on untuned shapes, down to the K=64 async last resort."""
    _require_gfx942()
    monkeypatch.setenv("AITER_FLYDSL_GEMM_GFX942_SKIP_TABLE", "1")
    x, w, ref = _inputs(m, n, k)
    y = gemm_a16w16(x, w.t(), layout="nt")
    torch.cuda.synchronize()
    _assert_close(y, ref, f"fallback M{m} N{n} K{k}")


@pytest.mark.parametrize("out_dtype", OUT_DTYPES, ids=OUT_DTYPE_IDS)
@pytest.mark.parametrize(
    "split_k",
    [{}, {"split_k": 2, "split_k_reduce": True}],
    ids=["default", "workspace-split-k"],
)
@pytest.mark.parametrize("n", [337, 339])
def test_gemm_a16w16_gfx942_ragged_n(n, split_k, out_dtype):
    """N not a multiple of the store vector: the last vector of each row stops at N."""
    _require_gfx942()
    m, k = 200, 512
    x, w, ref = _inputs(m, n, k)
    out = _guarded_out(m, n, out_dtype)
    cfg = {"block_m": 64, "block_n": 64, "block_k": 64, **split_k} if split_k else {}
    y = gemm_a16w16(x, w.t(), out=out[:m], layout="nt", user_kwargs=cfg)
    torch.cuda.synchronize()
    _assert_close(y, ref, f"ragged N{n} {split_k} {out_dtype}")
    assert torch.isnan(out[m:]).all(), "stores past N overwrote the guard rows"


@pytest.mark.parametrize(
    "cfg,match",
    [
        ({"split_k": 2, "split_k_reduce": False}, "multiple of 4"),
        ({"block_k": 16}, "multiple of 32"),
    ],
    ids=["atomic-split-k-ragged-n", "block_k-16"],
)
def test_gemm_a16w16_gfx942_rejects_unsupported(cfg, match):
    _require_gfx942()
    x, w, _ = _inputs(64, 337, 256)
    tile = {"block_m": 32, "block_n": 32, "block_k": 64, "m_waves": 1, "n_waves": 1}
    with pytest.raises(ValueError, match=match):
        gemm_a16w16(x, w.t(), layout="nt", user_kwargs={**tile, **cfg})


@pytest.mark.parametrize("layout", ["nn", "tn", "tt"])
def test_gemm_a16w16_gfx942_rejects_transposed_layouts(layout):
    _require_gfx942()
    x, w, _ = _inputs(32, 64, 128)
    with pytest.raises(ValueError, match="'nt' layout only"):
        gemm_a16w16(x, w.t(), layout=layout)


def test_gemm_a16w16_gfx942_split_k_launch_sequence():
    """Atomic and workspace split-K launches with one split_k, interleaved on a
    stream: each keeps its own semaphores, so no launch sees a stale tag."""
    _require_gfx942()
    m, n, k = 256, 256, 2048
    x, w, ref = _inputs(m, n, k)
    base = {"block_m": 64, "block_n": 64, "block_k": 64, "stages": 2, "split_k": 4}
    for i in range(3):
        for reduce in (False, True):
            y = gemm_a16w16(
                x, w.t(), layout="nt", user_kwargs={**base, "split_k_reduce": reduce}
            )
            torch.cuda.synchronize()
            _assert_close(y, ref, f"launch {i} split_k_reduce={reduce}")


def test_gemm_a16w16_gfx942_split_k_workspace_growth(monkeypatch):
    """Exact first size, at least doubling growth, outgrown buffers kept alive."""
    _require_gfx942()
    import aiter.ops.flydsl.kernels.gemm_a16w16_gfx942 as g

    monkeypatch.setattr(g, "_SPLIT_K_WORKSPACE", {})
    monkeypatch.setattr(g, "_RETIRED_SPLIT_K_WORKSPACES", [])
    first = g.get_split_k_workspace("cuda", 1000)
    assert first.numel() == 1000
    assert g.get_split_k_workspace("cuda", 600) is first
    grown = g.get_split_k_workspace("cuda", 1500)
    assert grown.numel() == 2000
    assert g._RETIRED_SPLIT_K_WORKSPACES == [first]
    assert g.get_split_k_workspace("cuda", 5000).numel() == 5000


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
