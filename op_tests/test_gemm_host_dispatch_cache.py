# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Host-side dispatch caches of the GEMM wrappers must not change any answer.

* the batched mxfp8_32 heuristic memoised per M class picks exactly what the
  per-M heuristic picks;
* the mxscale bpreshuffle lookup's N/K pre-check answers like the padded-M walk
  and reports an untuned weight shape once;
* the nested custom op called through its undecorated body (gemm_a8w8_blockscale
  -> gemm_a8w8_blockscale_bpreshuffle) returns what the registered torch.ops
  dispatch returns (torch_compile_guard's _eager_body), and a rebound module
  attribute still wins.
* the mxfp8_32 kernelName suffixes ``_pl`` / ``_nopl`` (batched and 2-D) pick
  the software-pipelined / single-fence K loop, a name without either gets
  default_pipelined_k_loop's pick, and the kernel rejects the pipelined loop,
  or any value other than 0 / 1, where that loop does not apply.

The first two, the ``_pl`` checks and the routing contract run on CPU; the
numerical checks run on any GPU whose arch has the kernel and skip elsewhere.
"""

import itertools
import logging

import pytest
import torch

from aiter import dtypes
from aiter.ops import gemm_op_a8w8
from aiter.ops.flydsl import batched_gemm_a8w8_gfx1250 as bmm
from aiter.ops.gemm_op_common import find_padded_m_row

# DeepSeek-V4.1 projection shapes at TP4/TP8 plus generic ones; K = 576 is the
# ragged shared-expert down projection.
_MX32_NK = [
    (512, 7168),
    (1024, 7168),
    (2048, 7168),
    (7168, 576),
    (7168, 1024),
    (7168, 2048),
    (4096, 4096),
    (1536, 512),
]


def _mx32_ms():
    """Every M up to 1024 and, above it, each tile boundary +-1 of every
    candidate tile_m up to well past the M-block limit."""
    ms = set(range(1, 1025))
    params, _ = bmm._mx32_m_class_params()
    for tm, _sat in params:
        for j in range(1, 80):
            ms.update((tm * j - 1, tm * j, tm * j + 1))
    ms.update(range(1025, 20000, 97))
    return sorted(m for m in ms if m > 0)


@pytest.fixture
def mx32_pick(monkeypatch):
    def setup(cu_num):
        monkeypatch.setattr(bmm, "get_cu_num", lambda: cu_num)
        bmm.pick_bmm_mx32_kernel_name.cache_clear()

    yield setup
    bmm.pick_bmm_mx32_kernel_name.cache_clear()


def _exact_or_error(b, m, n, k, w32):
    try:
        return bmm._pick_bmm_mx32_kernel_name_exact.__wrapped__(b, m, n, k, w32)
    except RuntimeError as e:
        return type(e)


def _cached_or_error(b, m, n, k, w32):
    try:
        return bmm.pick_bmm_mx32_kernel_name(b, m, n, k, w32)
    except RuntimeError as e:
        return type(e)


@pytest.mark.parametrize("cu_num", [256, 64])
@pytest.mark.parametrize("b", [1, 2, 8])
@pytest.mark.parametrize("w32", [False, True])
def test_mx32_pick_by_m_class_matches_exact(mx32_pick, cu_num, b, w32):
    mx32_pick(cu_num)
    ms = _mx32_ms()
    for n, k in _MX32_NK:
        # Ascending and descending M so each M class is first filled from both
        # of its ends.
        for m in itertools.chain(ms, reversed(ms)):
            assert _cached_or_error(b, m, n, k, w32) == _exact_or_error(
                b, m, n, k, w32
            ), (b, m, n, k, w32, cu_num)
    # A few hundred classes per weight shape, not one entry per M.
    assert len(bmm._MX32_PICK_BY_M_CLASS) < len(_MX32_NK) * len(ms) // 4


def test_mx32_parse_cache_returns_private_copies():
    name = bmm.bmm_mx32_kernel_name(64, 128, 256, 2, 2, 3)
    first = bmm.parse_bmm_mx32_kernel_name(name)
    first["tile_m"] = -1
    assert bmm.parse_bmm_mx32_kernel_name(name)["tile_m"] == 64
    assert bmm.parse_bmm_mx32_kernel_name("not_a_kernel") is None


def test_mx32_pl_suffix_round_trip():
    from aiter.ops.flydsl import mxfp8_bpreshuffle_gemm_gfx1250 as dense

    # t192x128x256 mw2 nw2 meets the default rule, t192x256x256 mw2 nw2
    # does not (VGPR estimate); the suffixes override it either way.
    for tile_n, default in ((128, 1), (256, 0)):
        name = bmm.bmm_mx32_kernel_name(192, tile_n, 256, 2, 2, 3)
        plain = bmm.parse_bmm_mx32_kernel_name(name)
        piped = bmm.parse_bmm_mx32_kernel_name(name + "_pl")
        single = bmm.parse_bmm_mx32_kernel_name(name + "_nopl")
        assert [c.pop("pipelined") for c in (plain, piped, single)] == [default, 1, 0]
        assert plain == piped == single
    for bad in ("_pl_pl", "_nopl_pl", "_pl_nopl"):
        assert bmm.parse_bmm_mx32_kernel_name(name + bad) is None

    suffix = "t64x128x256_mw2_nw2_nb3_sk1_cm1_cn1"
    parse32 = dense.parse_mxfp8_32_wmma_kernel_name
    name = f"{dense.MX32_WMMA_NAME_PREFIX}_{suffix}"
    plain, piped = parse32(name), parse32(name + "_pl")
    single = parse32(name + "_nopl")
    # The 2-D name leaves the default to the resolved launch config.
    assert [c.pop("pipelined") for c in (plain, piped, single)] == [
        dense.PIPELINED_DEFAULT,
        1,
        0,
    ]
    assert plain == piped == single
    aspre = parse32(name + "_aspre_pl")
    assert aspre["pipelined"] == 1 and aspre["a_scale_preshuffled"]
    assert parse32(name + "_aspre_nopl")["pipelined"] == 0
    # Only the generic mxfp8_32 kernel has the pipelined loop.
    for tail in ("_pl", "_nopl"):
        assert parse32(f"{dense.MX32_COMPUTE_WMMA_NAME_PREFIX}_{suffix}{tail}") is None
        name128 = f"{dense.WMMA_NAME_PREFIX}_{suffix}"
        assert dense.parse_wmma_kernel_name(name128 + tail) is None
    assert dense.parse_wmma_kernel_name(name128)["pipelined"] == 0
    compute = parse32(f"{dense.MX32_COMPUTE_WMMA_NAME_PREFIX}_{suffix}")
    assert compute["pipelined"] == 0


_DEFAULT_OK = {
    "mx32": True,
    "preload": False,
    "use_cluster": False,
    "split_k": 1,
    "tile_m": 64,
    "tile_n": 128,
    "tile_k": 256,
    "m_warp": 2,
    "n_warp": 2,
}


def test_default_pipelined_k_loop():
    from aiter.ops.flydsl.mxfp8_bpreshuffle_gemm_gfx1250 import (
        default_pipelined_k_loop as pick,
    )

    assert pick(**_DEFAULT_OK) == 1
    assert pick(**{**_DEFAULT_OK, "tile_k": 512}) == 1
    for key, value in (
        ("tile_k", 128),  # one k-step per K-tile
        ("mx32", False),
        ("preload", True),
        ("use_cluster", True),
        ("split_k", 2),
    ):
        assert pick(**{**_DEFAULT_OK, key: value}) == 0, key
    # VGPRs per lane: accumulators (8 each) plus two fragment sets
    # (16 per WMMA row or column) must fit 512. 6x4 reps is exactly 512.
    for tile_m, tile_n, m_warp, n_warp, want in (
        (192, 128, 2, 2, 1),  # 6x4 reps: 192 + 320
        (128, 128, 2, 2, 1),  # 4x4: 128 + 256
        (128, 64, 4, 1, 1),  # 2x4: 64 + 192
        (128, 256, 2, 2, 0),  # 4x8: 256 + 384
        (192, 256, 2, 2, 0),  # 6x8: 384 + 448
        (256, 256, 2, 2, 0),  # 8x8: 512 + 512
    ):
        cfg = {
            **_DEFAULT_OK,
            "tile_m": tile_m,
            "tile_n": tile_n,
            "m_warp": m_warp,
            "n_warp": n_warp,
        }
        assert pick(**cfg) == want, cfg


def test_mx32_2d_default_k_loop_after_cluster_fit():
    from aiter.ops.flydsl import mxfp8_bpreshuffle_gemm_gfx1250 as dense

    def resolved(name, m):
        return dict(dense._resolved_wmma_config(name, m, True, True))["pipelined"]

    prefix = dense.MX32_WMMA_NAME_PREFIX
    assert resolved(f"{prefix}_t64x128x256_mw2_nw2_nb3_sk1_cm1_cn1", 64) == 1
    assert resolved(f"{prefix}_t64x128x128_mw2_nw2_nb3_sk1_cm1_cn1", 64) == 0
    assert resolved(f"{prefix}_t64x128x256_mw2_nw2_nb3_sk2_cm1_cn1", 64) == 0
    assert resolved(f"{prefix}_t64x128x256_mw2_nw2_nb3_sk1_cm1_cn2", 64) == 0
    assert resolved(f"{prefix}_t128x256x256_mw2_nw2_nb3_sk1_cm1_cn1", 128) == 0
    # cluster_m is fitted to M first: one M tile drops cm2 to cm1.
    clustered = f"{prefix}_t64x128x256_mw2_nw2_nb3_sk1_cm2_cn1"
    assert resolved(clustered, 128) == 0
    assert resolved(clustered, 64) == 1
    # The suffixes are kept as asked.
    plain = f"{prefix}_t64x128x256_mw2_nw2_nb3_sk1_cm1_cn1"
    assert resolved(plain + "_nopl", 64) == 0
    assert resolved(f"{prefix}_t128x256x256_mw2_nw2_nb3_sk1_cm1_cn1_pl", 128) == 1


_PIPE_OK = {
    "mx32": True,
    "preload": False,
    "use_cluster": False,
    "split_k": 1,
    "tile_k": 256,
}


def test_pipelined_k_loop_check():
    from aiter.ops.flydsl.kernels.gemm_a8w8_gfx1250 import check_pipelined_k_loop

    assert check_pipelined_k_loop(1, **_PIPE_OK) is True
    assert check_pipelined_k_loop(0, **_PIPE_OK) is False
    # 0 never asks anything of the config.
    assert check_pipelined_k_loop(0, **{**_PIPE_OK, "tile_k": 128}) is False
    with pytest.raises(ValueError, match="tile_k >= 256"):
        check_pipelined_k_loop(1, **{**_PIPE_OK, "tile_k": 128})
    for key, value in (
        ("mx32", False),
        ("preload", True),
        ("use_cluster", True),
        ("split_k", 2),
    ):
        with pytest.raises(ValueError, match="pipelined K loop needs"):
            check_pipelined_k_loop(1, **{**_PIPE_OK, key: value})
    for bad in (-1, 2):
        with pytest.raises(ValueError, match="must be 0 or 1"):
            check_pipelined_k_loop(bad, **_PIPE_OK)


@pytest.fixture
def mxscale_table(monkeypatch):
    table = {
        ("gfx1250", 256, 64, 7168, 1024, "32x32"): {"kernelName": "m64"},
        ("gfx1250", 256, 256, 7168, 1024, "32x32"): {"kernelName": "m256"},
        ("gfx1250", 256, 128, 2048, 7168, "1x32"): {"kernelName": "m128"},
    }
    monkeypatch.setattr(gemm_op_a8w8, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(gemm_op_a8w8, "get_cu_num", lambda: 256)
    monkeypatch.setattr(gemm_op_a8w8, "_load_mxscale_bpreshuffle_tuned", lambda: table)
    monkeypatch.setattr(gemm_op_a8w8, "_MXSCALE_UNTUNED_REPORTED", set())
    gemm_op_a8w8._mxscale_bpreshuffle_tuned_nk.cache_clear()
    gemm_op_a8w8.get_mxscale_bpreshuffle_config.cache_clear()
    yield table
    gemm_op_a8w8._mxscale_bpreshuffle_tuned_nk.cache_clear()
    gemm_op_a8w8.get_mxscale_bpreshuffle_config.cache_clear()


def test_mxscale_config_precheck_matches_padded_walk(mxscale_table):
    shapes = [(7168, 1024, "32x32"), (2048, 7168, "1x32"), (2048, 7168, "32x32")]
    shapes += [(4096, 4096, "32x32")]
    for (n, k, block), m in itertools.product(shapes, range(1, 1100)):
        want, _ = find_padded_m_row(
            mxscale_table,
            lambda pm, n=n, k=k, block=block: ("gfx1250", 256, pm, n, k, block),
            m,
            n,
            k,
        )
        assert gemm_op_a8w8.get_mxscale_bpreshuffle_config(m, n, k, block) == want


def test_mxscale_untuned_reported_once_per_weight_shape(mxscale_table, monkeypatch):
    warnings = []
    monkeypatch.setattr(
        gemm_op_a8w8.logger, "warning", lambda msg, *a, **kw: warnings.append(msg)
    )
    for m in range(1, 300):
        assert (
            gemm_op_a8w8.get_mxscale_bpreshuffle_config(m, 4096, 4096, "32x32") is None
        )
        assert (
            gemm_op_a8w8.get_mxscale_bpreshuffle_config(m, 4096, 4096, "1x32") is None
        )
    assert len(warnings) == 2
    assert all("untuned" in w for w in warnings)


def test_blockscale_bpreshuffle_nested_body_and_rebind(monkeypatch):
    calls = []

    def body(XQ, WQ, x_scale, w_scale, dtype=dtypes.bf16, out=None):
        calls.append((x_scale.dtype, w_scale.dtype))
        return torch.zeros(XQ.shape[0], WQ.shape[0], dtype=dtype)

    op = gemm_op_a8w8.gemm_a8w8_blockscale_bpreshuffle
    assert op._eager_body.__name__ == "gemm_a8w8_blockscale_bpreshuffle"
    monkeypatch.setattr(op, "_eager_body", body)
    x = torch.zeros(4, 64, dtype=dtypes.fp8)
    w = torch.zeros(32, 64, dtype=dtypes.fp8)
    xs = torch.zeros(4, 2, dtype=torch.uint8)
    ws = torch.zeros(1, 2, dtype=torch.uint8)
    y = gemm_op_a8w8.gemm_a8w8_blockscale(x, w, xs, ws, isBpreshuffled=True)
    assert y.shape == (4, 32)
    assert calls == [(dtypes.fp8_e8m0, dtypes.fp8_e8m0)]

    rebound = []

    def patched(XQ, WQ, x_scale, w_scale, dtype=dtypes.bf16, out=None):
        rebound.append(True)
        return torch.ones(XQ.shape[0], WQ.shape[0], dtype=dtype)

    monkeypatch.setattr(gemm_op_a8w8, "gemm_a8w8_blockscale_bpreshuffle", patched)
    y = gemm_op_a8w8.gemm_a8w8_blockscale(x, w, xs, ws, isBpreshuffled=True)
    assert rebound == [True] and len(calls) == 1
    assert bool((y == 1).all())


def _gpu():
    return torch.cuda.is_available()


@pytest.mark.skipif(not _gpu(), reason="needs a GPU")
@pytest.mark.parametrize("m", [1, 37, 512, 1758])
@pytest.mark.parametrize("n,k", [(7168, 1024), (2048, 7168)])
@pytest.mark.parametrize("w_rows", [1, 32])
def test_blockscale_bpreshuffle_nested_matches_op(m, n, k, w_rows):
    from aiter import benchmark_data_init as bench_init
    from aiter.ops.flydsl.batched_gemm_a8w8 import bmm_a8w8_mxfp8_supported
    from aiter.ops.shuffle import shuffle_weight

    if not bmm_a8w8_mxfp8_supported(f"{w_rows}x32"):
        pytest.skip(f"no group32 {w_rows}x32 bpreshuffle GEMM on this arch")
    gen = bench_init.make_generator(0)
    x = bench_init.fill_fp8((m, k), "uniform", gen, dtype=dtypes.fp8)
    w = bench_init.fill_fp8((n, k), "uniform", gen, dtype=dtypes.fp8)
    xs = bench_init.fill_scale_e8m0((m, k // 32), "auto", gen).view(dtypes.fp8_e8m0)
    ws = bench_init.fill_scale_e8m0((-(-n // w_rows), k // 32), "auto", gen).view(
        dtypes.fp8_e8m0
    )
    w_shuf = shuffle_weight(w, layout=(16, 16))
    nested = gemm_op_a8w8.gemm_a8w8_blockscale(
        x, w_shuf, xs, ws, dtypes.bf16, isBpreshuffled=True
    )
    via_op = torch.ops.aiter.gemm_a8w8_blockscale_bpreshuffle(
        x, w_shuf, xs, ws, dtypes.bf16
    )
    assert torch.equal(nested, via_op)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    raise SystemExit(pytest.main([__file__, "-v"]))
