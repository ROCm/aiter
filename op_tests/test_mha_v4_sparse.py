# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Block-sparse and Sol-Attn MHA v4: LUT handling, work-table ordering, tile-count scaling,
per-recipe geometry, LSE, and the pooled Sol-Attn pass.

Split out of test_mha_v4.py, which had grown past 2600 lines with the sparse cases forming
one contiguous block of it. Absorbs the former test_mha_v4_sparse_tile_scaling.py.
"""

import argparse
import csv
import functools
import itertools
import math
import os
import subprocess
import sys
from collections.abc import Callable
from typing import NamedTuple

import pandas as pd
import pytest
import torch
import torch._dynamo

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_ROOT_DIR
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.mha_v4 import (
    MHA_V4_SOL_MODE,
    MHA_V4_SPARSE_MODE,
    AttentionFormat,
    AttentionPack,
    AttentionScaleMode,
    _resolve_raw_recipe,
    mha_v4,
    mha_v4_block_tile,
    mha_v4_block_tiles,
    mha_v4_kv_tile,
    mha_v4_kv_tile_for_q_tile,
    mha_v4_operands,
    mha_v4_packed,
    mha_v4_ragged_kv,
    mha_v4_sol,
    mha_v4_sparse_work_table,
    native_fp8_format,
    scale_modes_for_formats,
)
from aiter.ops.mha_v4_quant import (
    MHA_V4_LOG2E,
    mha_v4_q_multiplier,
    mxfp4_k_view,
    mxfp4_v_view,
    mxfp6_k_view,
    quantize_fp8,
    quantize_fp8_rotated,
    quantize_int8,
    quantize_mxfp4_k,
    quantize_mxfp4_q,
    quantize_mxfp6_k,
    quantize_mxfp6_q,
    quantize_mxfp8_k,
    quantize_mxfp8_q,
    quantize_v_fp8,
    quantize_v_mxfp4_fp6_p,
    quantize_v_mxfp6_fp6_p,
    rotate_activation_hd128,
)
from aiter.ops.triton.attention.utils import (
    SOL_ATTN_TS_KV,
    SOL_ATTN_TS_QO,
    _e8m0_dequantize,
    _e8m0_quantize,
    _sol_attn_block_mean,
    _sol_attn_block_variance,
    _sol_attn_pool_mx,
    _sol_attn_pool_reuse_descale,
    block_attn_mask_to_ragged_lut,
    sol_prepare,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest


@pytest.fixture(autouse=True)
def isolate_dynamo_cache():
    """Keep each test's ``torch.compile`` behaviour independent of the tests that ran before it.

    Dynamo caches compiled entries per code object, and this file compiles ``mha_v4`` under many
    format combinations. Sharing that cache across tests means the suite drifts toward the recompile
    limit and whichever test compiles last fails under ``fullgraph=True`` -- a failure that reports
    against a kernel while actually depending on how many earlier tests got far enough to compile.
    """
    torch._dynamo.reset()
    yield


_MHA_V4_SPARSE_ARCH = get_gfx() in ("gfx942", "gfx950")


FP8 = native_fp8_format()
_MX_SCALES = {
    "q_scale_mode": AttentionScaleMode.E8M0_PER_1X32,
    "k_scale_mode": AttentionScaleMode.E8M0_PER_1X32,
    "v_scale_mode": AttentionScaleMode.F32_PER_TENSOR,
}

# (q/k format, v format, kwargs). K takes Q's format, as the manifest rows do.
SPARSE_RECIPES = {
    "bf16": (AttentionFormat.BF16, AttentionFormat.BF16, {}),
    "bf16fp8": (AttentionFormat.BF16, FP8, {}),
    "i8fp8": (AttentionFormat.INT8, FP8, {}),
    "fp8": (FP8, FP8, {}),
    "mxfp8": (FP8, FP8, _MX_SCALES),
    "f8f6": (FP8, AttentionFormat.MXFP6, {}),
    "f6f8": (AttentionFormat.MXFP6, FP8, {}),
    "mxfp6": (AttentionFormat.MXFP6, AttentionFormat.MXFP6, {}),
    "f6f4": (AttentionFormat.MXFP6, AttentionFormat.MXFP4, {}),
    "mxfp4": (AttentionFormat.MXFP4, AttentionFormat.MXFP4, {}),
}

# The finer geometry, which gfx950 serves for these recipes in both the sparse and Sol-Attn modes.
_MHA_V4_FINE_TILE = (64, 64)


def _recipe_operands(q_format, v_format, kwargs=None):
    """The manifest key a raw mha_v4 block-sparse call with these formats dispatches on."""
    kwargs = kwargs or {}
    formats = (q_format, q_format, v_format)
    recipe = _resolve_raw_recipe(
        *formats,
        kwargs.get("q_scale_mode"),
        kwargs.get("k_scale_mode"),
        kwargs.get("v_scale_mode"),
        sparse=True,
    )
    return mha_v4_operands(*formats, *recipe.scale_modes, recipe.v_pack)


def _sparse_tiles(name):
    """The block-sparse geometries the manifest serves a recipe at, or () if it has none."""
    if not _MHA_V4_SPARSE_ARCH:
        return ()
    q_format, v_format, kwargs = SPARSE_RECIPES[name]
    return mha_v4_block_tiles(
        _recipe_operands(q_format, v_format, kwargs), MHA_V4_SPARSE_MODE
    )


def _requires_sparse_row(name, block_tile=None):
    """Skip unless the manifest has a block-sparse row for the recipe, at block_tile if given."""
    tiles = _sparse_tiles(name)
    if block_tile is None:
        return pytest.mark.skipif(
            not tiles, reason=f"no {name} block-sparse row on {get_gfx()}"
        )
    q_tile, kv_tile = block_tile
    return pytest.mark.skipif(
        block_tile not in tiles,
        reason=f"no {q_tile}x{kv_tile} {name} block-sparse row on {get_gfx()}",
    )


def _sparse_recipe_params(names=None, block_tile=None):
    """pytest params of (q/k format, v format, kwargs), each skipped where its row is not shipped."""
    return [
        pytest.param(
            *SPARSE_RECIPES[name],
            marks=_requires_sparse_row(name, block_tile),
            id=name,
        )
        for name in (SPARSE_RECIPES if names is None else names)
    ]


def _sparse_recipe_tile_params():
    """_sparse_recipe_params() plus a block_tile: None for each recipe's default, then 64x64."""
    return [
        pytest.param(*p.values, None, marks=p.marks, id=p.id)
        for p in _sparse_recipe_params()
    ] + [
        pytest.param(*p.values, _MHA_V4_FINE_TILE, marks=p.marks, id=f"{p.id}-64x64")
        for p in _sparse_recipe_params(block_tile=_MHA_V4_FINE_TILE)
    ]


def _recipe_name_params(names):
    """pytest params of recipe names, each skipped where its row is not shipped."""
    return [
        pytest.param(name, marks=_requires_sparse_row(name), id=name) for name in names
    ]


def _v_pack(q_format, v_format):
    """The V packing a raw call with these formats dispatches with."""
    return _resolve_raw_recipe(
        q_format, q_format, v_format, None, None, None, sparse=True
    ).v_pack


def _default_kv_tile(q_format=None, v_format=None, mode=MHA_V4_SPARSE_MODE):
    """The KV tile the block-sparse rows route at for these formats.

    Named rather than left to the arch because gfx950's rows no longer agree at a 256-row query
    tile: BF16 and BF16/FP8 route on a 64-token block and every other recipe on 128, so an
    operand-blind query raises. Defaults to the per-tensor FP8 row, which is what most of the
    tests here build, and takes the formats where a test is parametrized over them.
    """
    q_format = native_fp8_format() if q_format is None else q_format
    v_format = native_fp8_format() if v_format is None else v_format
    return mha_v4_kv_tile(
        mha_v4_operands(
            q_format,
            q_format,
            v_format,
            *scale_modes_for_formats(q_format, q_format, v_format),
            _v_pack(q_format, v_format),
        ),
        mode,
    )


_FP6_P_FORMAT_PARAMS = [
    pytest.param(
        q_format,
        v_format,
        marks=_requires_sparse_row(name),
        id=f"{name}_fp6p",
    )
    for name, q_format, v_format in (
        ("mxfp6", AttentionFormat.MXFP6, AttentionFormat.MXFP6),
        ("f8f6", FP8, AttentionFormat.MXFP6),
        ("f6f4", AttentionFormat.MXFP6, AttentionFormat.MXFP4),
    )
]


def _sol_attn_raw_inputs(sequence_k, heads=2, sequence_q=256, batch=1, seed=0):
    """The BF16 operands the sparse and Sol-Attn tests quantize, so recipes see the same data."""
    torch.manual_seed(seed)
    q = torch.randn(
        (batch, sequence_q, heads, 128), device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        (batch, sequence_k, heads, 128), device="cuda", dtype=torch.bfloat16
    )
    return q, k, torch.randn_like(k)


def test_mha_v4_packed_rejects_partial_lut():
    dummy = torch.empty(0)
    with pytest.raises(ValueError, match="all be set or all omitted"):
        mha_v4_packed(
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            AttentionFormat.INT8,
            AttentionFormat.INT8,
            AttentionFormat.FP8,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            kv_block_indices=dummy,
        )


def test_mha_v4_rejects_wrong_block_mask_shape():
    q = torch.zeros((1, 256, 2, 128), dtype=torch.bfloat16)
    mask = torch.ones((1, 2, 1, 1), dtype=torch.bool)
    with pytest.raises(ValueError, match="block_mask must have shape"):
        mha_v4(
            q,
            q,
            q,
            FP8,
            FP8,
            FP8,
            block_mask=mask,
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse rows")
def test_mha_v4_block_mask_without_a_sparse_row_names_the_recipe():
    """The other arch's FP8 has no row here, so there is no KV tile to cut the mask to."""
    foreign = (
        AttentionFormat.FP8_E4M3
        if FP8 == AttentionFormat.FP8_E4M3_FNUZ
        else AttentionFormat.FP8_E4M3_FNUZ
    )
    q = torch.zeros((1, 256, 2, 128), dtype=torch.bfloat16)
    mask = torch.ones((1, 2, 1, 1), dtype=torch.bool)
    with pytest.raises(NotImplementedError, match="has no .* row for"):
        mha_v4(q, q, q, foreign, foreign, foreign, block_mask=mask)

    packed = q.cuda()
    lut = torch.zeros(2, device="cuda", dtype=torch.int32)
    per_tensor = AttentionScaleMode.F32_PER_TENSOR
    with pytest.raises(NotImplementedError, match="has no .* row for"):
        mha_v4_packed(
            packed,
            packed,
            packed,
            packed,
            packed,
            packed,
            foreign,
            foreign,
            foreign,
            per_tensor,
            per_tensor,
            per_tensor,
            kv_block_indices=lut,
            lut_start=lut,
            lut_count=lut,
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse schema")
def test_mha_v4_sparse_schema_mutates_only_out():
    dense = str(torch.ops.aiter.mha_v4_fwd_launch.default._schema)
    assert "Tensor kv_block_indices" not in dense
    assert "Tensor(a6!) out" in dense

    sparse = str(torch.ops.aiter.mha_v4_fwd_sparse_launch.default._schema)
    assert "Tensor kv_block_indices" in sparse
    assert "Tensor lut_start" in sparse
    assert "Tensor lut_count" in sparse
    assert "Tensor(a6!) out" in sparse
    assert sparse.endswith("-> ()")


def _work_table_counts(total, pattern):
    """LUT lengths covering the tie structures the ordering has to get right."""
    if pattern == "uniform":
        return torch.full((total,), 7, device="cuda", dtype=torch.int32)
    if pattern == "zeros":
        return torch.zeros((total,), device="cuda", dtype=torch.int32)
    if pattern == "random":
        return torch.randint(0, 64, (total,), device="cuda", dtype=torch.int32)
    if pattern == "wide_random":
        return torch.randint(0, 8192, (total,), device="cuda", dtype=torch.int32)
    if pattern == "two_values":
        alternating = torch.arange(total, device="cuda") % 3 == 0
        return torch.where(alternating, 9, 4).to(torch.int32)
    if pattern == "descending":
        return torch.arange(total, 0, -1, device="cuda", dtype=torch.int32)
    return torch.arange(1, total + 1, device="cuda", dtype=torch.int32)


def _unpack_work_table(table, nhead, q_tiles):
    q_idx = (table & 0xFFFF).long()
    h_idx = ((table >> 16) & 0xFF).long()
    b_idx = ((table >> 24) & 0xFF).long()
    return (b_idx * nhead + h_idx) * q_tiles + q_idx


# The table has one entry per (batch, head, query tile). 8192 is the point where the builder hands
# the sort to ATen, so straddle it, and include sizes that are not multiples of a wave or workgroup.
@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.parametrize(
    ("batch", "nhead", "q_tiles"),
    [
        (1, 1, 1),
        (1, 8, 3),
        (2, 5, 7),
        (1, 16, 32),
        (1, 32, 32),
        (1, 5, 296),
        (4, 16, 64),
        (8, 16, 64),
        (8, 32, 64),
    ],
)
@pytest.mark.parametrize(
    "pattern",
    [
        "uniform",
        "zeros",
        "random",
        "wide_random",
        "two_values",
        "descending",
        "ascending",
    ],
)
def test_mha_v4_sparse_work_table_is_longest_lut_first(batch, nhead, q_tiles, pattern):
    torch.manual_seed(7)
    total = batch * nhead * q_tiles
    counts = _work_table_counts(total, pattern)

    table = mha_v4_sparse_work_table(counts, batch, nhead, q_tiles)
    visited = _unpack_work_table(table, nhead, q_tiles)

    # Every tile exactly once. This is the part a wrong table would turn into a wrong result.
    assert torch.equal(visited.sort().values, torch.arange(total, device="cuda"))

    # Longest LUT first, so no heavy tile straggles behind the rest.
    ordered = counts[visited]
    assert bool((ordered[:-1] >= ordered[1:]).all())

    # Ties keep raster order, which is what leaves uniform counts spatially coherent. A stable
    # reference sort pins the whole permutation, not just the two properties above.
    expected = torch.argsort(counts, descending=True, stable=True)
    assert torch.equal(visited, expected)


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.parametrize(
    "batch,nhead,q_tiles", [(1, 16, 32), (1, 5, 296), (8, 16, 64), (8, 32, 64)]
)
def test_mha_v4_sparse_work_table_leaves_uniform_counts_in_raster_order(
    batch, nhead, q_tiles
):
    """Top-k sparsity gives every tile the same LUT length, and that case must not be shuffled."""
    total = batch * nhead * q_tiles
    counts = torch.full((total,), 5, device="cuda", dtype=torch.int32)

    table = mha_v4_sparse_work_table(counts, batch, nhead, q_tiles)

    visited = _unpack_work_table(table, nhead, q_tiles)
    assert torch.equal(visited, torch.arange(total, device="cuda"))


# Either side of the 8192-entry hand-off to ATen, which builds each order separately.
@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.parametrize(("batch", "nhead", "q_tiles"), [(2, 5, 7), (8, 32, 64)])
@pytest.mark.parametrize("pattern", ["random", "wide_random", "uniform"])
@pytest.mark.parametrize("order", ["lpt", "head", "raster"])
def test_mha_v4_sparse_work_table_follows_the_work_order(
    monkeypatch, batch, nhead, q_tiles, pattern, order
):
    """AITER_MHA_V4_WORK_ORDER picks the whole permutation, ties included."""
    torch.manual_seed(7)
    total = batch * nhead * q_tiles
    counts = _work_table_counts(total, pattern)
    monkeypatch.setenv("AITER_MHA_V4_WORK_ORDER", order)

    table = mha_v4_sparse_work_table(counts, batch, nhead, q_tiles)

    lengths = counts.tolist()
    key = {
        "lpt": lambda slot: (-lengths[slot], slot),
        # (batch, head) groups in raster order, longest LUT first within each.
        "head": lambda slot: (slot // q_tiles, -lengths[slot], slot),
        "raster": lambda slot: slot,
    }[order]
    visited = _unpack_work_table(table, nhead, q_tiles).tolist()
    assert visited == sorted(range(total), key=key)


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
def test_mha_v4_sparse_work_table_rejects_an_unknown_work_order(monkeypatch):
    monkeypatch.setenv("AITER_MHA_V4_WORK_ORDER", "fastest")
    counts = torch.ones(4, device="cuda", dtype=torch.int32)
    with pytest.raises(RuntimeError, match="must be lpt, head or raster"):
        mha_v4_sparse_work_table(counts, 1, 1, 4)


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.parametrize(
    ("q_format", "v_format", "kwargs", "block_tile"), _sparse_recipe_tile_params()
)
def test_mha_v4_sparse_all_true_mask_matches_dense(
    q_format, v_format, kwargs, block_tile
):
    torch.manual_seed(41)
    q = torch.randn((1, 511, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 512, 5, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    if block_tile is None:
        block_tile = mha_v4_block_tile(
            _recipe_operands(q_format, v_format, kwargs), MHA_V4_SPARSE_MODE
        )
    q_tile, kv_tile = block_tile
    mask = torch.ones(
        (1, 5, -(-511 // q_tile), 512 // kv_tile),
        device="cuda",
        dtype=torch.bool,
    )
    formats = (q_format, q_format, v_format)
    dense = mha_v4(q, k, v, *formats, **kwargs)
    sparse = mha_v4(q, k, v, *formats, block_mask=mask, block_tile=block_tile, **kwargs)
    torch.cuda.synchronize()
    _assert_sparse_matches_dense(sparse, dense)


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 rows are ragged")
@pytest.mark.parametrize("tail", [44, 5])
@pytest.mark.parametrize(
    ("q_format", "v_format", "kwargs", "block_tile"), _sparse_recipe_tile_params()
)
def test_mha_v4_sparse_masks_a_ragged_last_block(
    q_format, v_format, kwargs, block_tile, tail
):
    """A selected short last block attends to the keys it has and to none past seqlen_k.

    The dense row masks its own tail, so it is the reference over exactly the selected keys. The
    last block is selected alongside others, alone (the single-block schedule), and with every
    other block, so both loop exits and the lone-block seed are covered.
    """
    torch.manual_seed(tail)
    formats = (q_format, q_format, v_format)
    recipe = _resolve_raw_recipe(
        *formats,
        kwargs.get("q_scale_mode"),
        kwargs.get("k_scale_mode"),
        kwargs.get("v_scale_mode"),
        sparse=True,
    )
    operands = mha_v4_operands(*formats, *recipe.scale_modes, recipe.v_pack)
    if block_tile is None:
        block_tile = mha_v4_block_tile(operands, MHA_V4_SPARSE_MODE)
    assert mha_v4_ragged_kv(operands, MHA_V4_SPARSE_MODE, block_tile)
    q_tile, kv_tile = block_tile
    blocks = 4
    sequence_q = 511
    sequence_k = (blocks - 1) * kv_tile + tail
    q = torch.randn((1, sequence_q, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, sequence_k, 5, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    tail_keys = torch.arange((blocks - 1) * kv_tile, sequence_k, device="cuda")
    q_tiles = -(-sequence_q // q_tile)
    for selected in ([0, 2, 3], [3], [0, 1, 2, 3]):
        mask = torch.zeros((1, 5, q_tiles, blocks), device="cuda", dtype=torch.bool)
        mask[..., selected] = True
        keys = torch.cat(
            [
                torch.arange(b * kv_tile, (b + 1) * kv_tile, device="cuda")
                for b in selected[:-1]
            ]
            + [tail_keys]
        )
        dense = mha_v4(q, k[:, keys], v[:, keys], *formats, **kwargs)
        sparse = mha_v4(
            q, k, v, *formats, block_mask=mask, block_tile=block_tile, **kwargs
        )
        torch.cuda.synchronize()
        _assert_sparse_matches_dense(sparse, dense, f"blocks {selected}")


def _assert_sparse_matches_dense(sparse, dense, message=None):
    """Compare code objects that use different softmax reduction schedules."""
    cosine = torch.nn.functional.cosine_similarity(
        sparse.float().flatten(), dense.float().flatten(), dim=0
    )
    assert cosine > 0.99, message
    assert torch.isfinite(sparse).all()


@_requires_sparse_row("mxfp4")
def test_mha_v4_f4f4_sparse_all_true_mask_matches_dense():
    """Sparse F4F4 remains close to dense on an all-true mask; both run FP6-P."""
    torch.manual_seed(41)
    q = torch.randn((1, 511, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 512, 5, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mxfp4 = AttentionFormat.MXFP4
    mask = torch.ones(
        (1, 5, 2, 512 // _default_kv_tile(mxfp4, mxfp4)),
        device="cuda",
        dtype=torch.bool,
    )
    args = (AttentionFormat.MXFP4,) * 3
    dense = mha_v4(q, k, v, *args)
    sparse = mha_v4(q, k, v, *args, block_mask=mask)
    torch.cuda.synchronize()

    _assert_sparse_matches_dense(sparse, dense)


@_requires_sparse_row("mxfp6")
def test_mha_v4_mxfp6_accepts_block_mask():
    """MXFP6 Q/K/V has a sorted-sparse row, so a block mask is dispatched, not rejected."""
    q = torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)
    mxfp6 = AttentionFormat.MXFP6
    mask = torch.ones(
        (1, 2, 1, 256 // _default_kv_tile(mxfp6, mxfp6)),
        device="cuda",
        dtype=torch.bool,
    )
    out = mha_v4(
        q,
        q,
        q,
        AttentionFormat.MXFP6,
        AttentionFormat.MXFP6,
        AttentionFormat.MXFP6,
        block_mask=mask,
    )
    assert out.shape == q.shape
    assert torch.isfinite(out).all()


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@_requires_sparse_row("fp8")
def test_mha_v4_sparse_block_mask_compiles_without_graph_breaks():
    """The mask path derives its geometry from host state, which Dynamo cannot trace.

    _default_kv_tile() reads the manifest and get_gfx() shells out to rocminfo, so both sit behind
    torch_compile_guard. Without that the sparse mask path costs graph breaks per trace and fails
    under fullgraph, which no other test in this file would notice.
    """
    torch.manual_seed(41)
    kv_tile = _default_kv_tile()
    q = torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 4 * kv_tile, 2, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.ones((1, 2, 1, 4), device="cuda", dtype=torch.bool)
    fp8_format = native_fp8_format()

    def call():
        return mha_v4(q, k, v, fp8_format, fp8_format, fp8_format, block_mask=mask)

    explained = torch._dynamo.explain(call)()
    assert explained.break_reasons == [], [
        str(reason.reason) for reason in explained.break_reasons
    ]

    eager = call()
    compiled = torch.compile(call, fullgraph=True)()
    torch.cuda.synchronize()

    assert torch.equal(eager, compiled)


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.parametrize(
    ("q_format", "v_format"),
    [
        pytest.param(*p.values[:2], marks=p.marks, id=p.id)
        for p in _sparse_recipe_params(("bf16", "bf16fp8", "fp8", "mxfp4"))
    ]
    + _FP6_P_FORMAT_PARAMS,
)
def test_mha_v4_sparse_gqa_all_true_mask_matches_repeated_kv(q_format, v_format):
    torch.manual_seed(41)
    query_heads = 8
    kv_heads = 2
    gqa_ratio = query_heads // kv_heads
    q = torch.randn((1, 256, query_heads, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 256, kv_heads, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    kv_tiles = 256 // _default_kv_tile(q_format, v_format)
    mask = torch.ones((1, query_heads, 1, kv_tiles), device="cuda", dtype=torch.bool)
    k_repeated = k.repeat_interleave(gqa_ratio, dim=2)
    v_repeated = v.repeat_interleave(gqa_ratio, dim=2)

    gqa_sparse = mha_v4(q, k, v, q_format, q_format, v_format, block_mask=mask)
    mha_sparse = mha_v4(
        q,
        k_repeated,
        v_repeated,
        q_format,
        q_format,
        v_format,
        block_mask=mask,
    )
    torch.cuda.synchronize()

    assert torch.equal(gqa_sparse, mha_sparse)
    if q_format != AttentionFormat.MXFP4:
        gqa_dense = mha_v4(q, k, v, q_format, q_format, v_format)
        mha_dense = mha_v4(q, k_repeated, v_repeated, q_format, q_format, v_format)
        assert torch.equal(gqa_dense, mha_dense)
        _assert_sparse_matches_dense(gqa_sparse, gqa_dense)
    assert torch.equal(gqa_sparse, mha_sparse)


class _Operand(NamedTuple):
    """A quantized MHA v4 operand and the descale it was produced with."""

    quantized: torch.Tensor
    descale: torch.Tensor
    # The pre-quantization tensor, kept only by the recipes whose stored codes are not element
    # addressable: those cannot be pooled or dequantized from `quantized` at all.
    source: torch.Tensor | None = None


def _sparse_fp8_operands(sequence_k, heads=2, sequence_q=256, batch=1, seed=0):
    """Quantize once so sparse and reference runs share descales exactly.

    Re-quantizing a KV slice would pick a different per-tensor amax, which shifts every
    value and hides whether the kernel read the KV blocks the LUT named.
    """
    q, k, v = _sol_attn_raw_inputs(sequence_k, heads, sequence_q, batch, seed)
    return (
        _Operand(*quantize_fp8_rotated(q)),
        _Operand(*quantize_fp8_rotated(k)),
        _Operand(*quantize_fp8(v)),
    )


def _sparse_fp8_launch(q, k, v, block_mask=None):
    lut = {}
    if block_mask is not None:
        indices, start, count = block_attn_mask_to_ragged_lut(
            block_mask,
            num_heads=block_mask.shape[1],
            return_none_if_dense=False,
        )
        lut = {
            "kv_block_indices": indices,
            "lut_start": start,
            "lut_count": count,
        }
    fp8_format = native_fp8_format()
    return mha_v4_packed(
        q.quantized,
        k.quantized,
        v.quantized,
        q.descale,
        k.descale,
        v.descale,
        fp8_format,
        fp8_format,
        fp8_format,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        AttentionScaleMode.F32_PER_TENSOR,
        **lut,
    )


def _gather_kv_tiles(operand, tiles):
    """Concatenate the named KV tiles, leaving the quantized bytes and descale untouched."""
    kv_tile = _default_kv_tile()
    gathered = torch.cat(
        [operand.quantized[:, tile * kv_tile : (tile + 1) * kv_tile] for tile in tiles],
        dim=1,
    )
    return _Operand(gathered.contiguous(), operand.descale)


def _tile_mask(heads, kv_tiles, tiles, q_tiles=1, batch=1):
    mask = torch.zeros(
        (batch, heads, q_tiles, kv_tiles), device="cuda", dtype=torch.bool
    )
    for tile in tiles:
        mask[:, :, :, tile] = True
    return mask


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@_requires_sparse_row("fp8")
@pytest.mark.parametrize("tiles", [(0,), (1,), (3,), (0, 2), (1, 2, 3)])
def test_mha_v4_sparse_reads_only_the_kv_tiles_the_lut_names(tiles):
    """A kernel that ignored kv_block_indices would pass every all-True test."""
    heads = 2
    kv_tile = _default_kv_tile()
    kv_tiles = 4
    q, k, v = _sparse_fp8_operands(sequence_k=kv_tiles * kv_tile, heads=heads)

    mask = _tile_mask(heads, kv_tiles, tiles)
    sparse = _sparse_fp8_launch(q, k, v, block_mask=mask)
    # Dense over exactly the selected tiles: same quantized bytes, same descales, so the
    # only difference is which KV blocks take part.
    dense = _sparse_fp8_launch(
        q, _gather_kv_tiles(k, tiles), _gather_kv_tiles(v, tiles)
    )
    torch.cuda.synchronize()

    _assert_sparse_matches_dense(sparse, dense)


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@_requires_sparse_row("fp8")
def test_mha_v4_sparse_distinct_kv_tiles_give_distinct_results():
    """Guards the reference itself: selecting different tiles must change the output."""
    heads = 2
    kv_tile = _default_kv_tile()
    kv_tiles = 4
    q, k, v = _sparse_fp8_operands(sequence_k=kv_tiles * kv_tile, heads=heads)

    outputs = [
        _sparse_fp8_launch(q, k, v, block_mask=_tile_mask(heads, kv_tiles, (tile,)))
        for tile in range(kv_tiles)
    ]
    torch.cuda.synchronize()

    for tile in range(1, kv_tiles):
        assert not torch.equal(outputs[0], outputs[tile]), (
            f"kv tile 0 and kv tile {tile} produced identical output, so the kernel is "
            "not reading kv_block_indices"
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@_requires_sparse_row("fp8")
def test_mha_v4_sparse_gives_each_head_its_own_kv_tiles():
    """4-D masks may give heads different KV lists; each head must follow its own row."""
    heads = 3
    kv_tile = _default_kv_tile()
    kv_tiles = 4
    per_head = ((0,), (3,), (1, 2))
    q, k, v = _sparse_fp8_operands(sequence_k=kv_tiles * kv_tile, heads=heads)

    mask = torch.zeros((1, heads, 1, kv_tiles), device="cuda", dtype=torch.bool)
    for head, tiles in enumerate(per_head):
        for tile in tiles:
            mask[:, head, :, tile] = True
    sparse = _sparse_fp8_launch(q, k, v, block_mask=mask)
    torch.cuda.synchronize()

    for head, tiles in enumerate(per_head):
        dense = _sparse_fp8_launch(
            q, _gather_kv_tiles(k, tiles), _gather_kv_tiles(v, tiles)
        )
        torch.cuda.synchronize()
        _assert_sparse_matches_dense(
            sparse[:, :, head],
            dense[:, :, head],
            f"head {head} did not attend to tiles {tiles}",
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@_requires_sparse_row("fp8")
def test_mha_v4_sparse_follows_the_lut_across_query_tiles():
    """Multiple query tiles exercise the work table on a real launch, not just its ordering."""
    heads = 2
    kv_tile = _default_kv_tile()
    kv_tiles = 4
    q_tiles = 2
    q, k, v = _sparse_fp8_operands(
        sequence_k=kv_tiles * kv_tile, heads=heads, sequence_q=256 * q_tiles
    )

    mask = torch.zeros((1, heads, q_tiles, kv_tiles), device="cuda", dtype=torch.bool)
    mask[:, :, 0, 0] = True
    mask[:, :, 1, 3] = True
    sparse = _sparse_fp8_launch(q, k, v, block_mask=mask)
    torch.cuda.synchronize()

    for q_tile, tiles in ((0, (0,)), (1, (3,))):
        dense = _sparse_fp8_launch(
            q, _gather_kv_tiles(k, tiles), _gather_kv_tiles(v, tiles)
        )
        torch.cuda.synchronize()
        rows = slice(q_tile * 256, (q_tile + 1) * 256)
        _assert_sparse_matches_dense(
            sparse[:, rows],
            dense[:, rows],
            f"query tile {q_tile} did not attend to tiles {tiles}",
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@_requires_sparse_row("fp8")
@pytest.mark.parametrize("tail_rows", [64, 128, 200])
def test_mha_v4_sparse_partial_query_tile_follows_the_lut(tail_rows):
    """A trailing partial query tile has to select and zero the same way a full one does.

    Production sequence lengths are not multiples of the 256-row query tile, and the empty-row
    no-op is built on the same tail masking the partial tile uses, so the two belong in one
    case: the short tile must still read the KV blocks its row names, and an all-False row on
    that tile must come back zero rather than reading the masked-off remainder.
    """
    heads = 2
    kv_tile = _default_kv_tile()
    kv_tiles = 4
    q_tiles = 3
    per_tile = ((0, (0,)), (1, (1, 2)), (2, (3,)))
    q, k, v = _sparse_fp8_operands(
        sequence_k=kv_tiles * kv_tile,
        heads=heads,
        sequence_q=256 * (q_tiles - 1) + tail_rows,
    )

    mask = torch.zeros((1, heads, q_tiles, kv_tiles), device="cuda", dtype=torch.bool)
    for q_tile, tiles in per_tile:
        for tile in tiles:
            mask[:, :, q_tile, tile] = True
    mask[:, 1, q_tiles - 1, :] = False  # head 1's partial-tile row selects nothing
    sparse = _sparse_fp8_launch(q, k, v, block_mask=mask)
    torch.cuda.synchronize()

    for q_tile, tiles in per_tile:
        dense = _sparse_fp8_launch(
            q, _gather_kv_tiles(k, tiles), _gather_kv_tiles(v, tiles)
        )
        torch.cuda.synchronize()
        rows = slice(q_tile * 256, min((q_tile + 1) * 256, sparse.shape[1]))
        live_heads = 1 if q_tile == q_tiles - 1 else heads
        _assert_sparse_matches_dense(
            sparse[:, rows, :live_heads],
            dense[:, rows, :live_heads],
            f"query tile {q_tile} did not attend to tiles {tiles}",
        )

    tail = slice((q_tiles - 1) * 256, sparse.shape[1])
    assert torch.equal(
        sparse[:, tail, 1], torch.zeros_like(sparse[:, tail, 1])
    ), "empty row on the partial query tile is not zero"
    # Without this the case would also pass on a kernel that skipped the short tile entirely,
    # since both sides of the comparison above would then be zero.
    assert (
        sparse[:, tail, 0].abs().max() > 0
    ), "live partial-tile row came back degenerate"
    assert torch.isfinite(sparse).all(), "partial query tile leaked NaN or infinity"


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.parametrize(("q_format", "v_format", "kwargs"), _sparse_recipe_params())
def test_mha_v4_sparse_empty_row_writes_zeros(q_format, v_format, kwargs):
    """An all-False row selects no KV block, so its output tile must be zero, not garbage.

    Its lut_start also sits one past the last kv_block_indices entry, which is what used to send
    the row's reads far out of bounds and fault the kernel.
    """
    heads = 2
    kv_tiles = 4
    selected = (0, 1)
    torch.manual_seed(0)
    q = torch.randn((1, 256, heads, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn(
        (1, kv_tiles * _default_kv_tile(q_format, v_format), heads, 128),
        device="cuda",
        dtype=torch.bfloat16,
    )
    v = torch.randn_like(k)

    def launch(q, k, v, mask):
        return mha_v4(q, k, v, q_format, q_format, v_format, block_mask=mask, **kwargs)

    mask = torch.zeros((1, heads, 1, kv_tiles), device="cuda", dtype=torch.bool)
    for tile in selected:
        mask[:, 0, :, tile] = True  # head 0 selects two tiles; head 1 stays all-False
    out = launch(q, k, v, mask)
    torch.cuda.synchronize()

    assert torch.equal(
        out[:, :, 1], torch.zeros_like(out[:, :, 1])
    ), "empty row is not zero"
    assert torch.isfinite(out).all(), "empty row leaked NaN or infinity"
    assert out[:, :, 0].abs().max() > 0, "live row came back degenerate"

    # The empty row must not perturb the row that does select tiles: give head 1 a tile and head 0's
    # output has to stay bit-identical, since each workgroup owns one (batch, head, query tile).
    mask[:, 1, :, 0] = True
    populated = launch(q, k, v, mask)
    torch.cuda.synchronize()
    assert torch.equal(
        out[:, :, 0], populated[:, :, 0]
    ), "empty row disturbed the live row"


def test_mha_v4_rejects_non_bool_block_mask():
    """Counts come from a sum but the fill uses truthiness, so non-bool masks disagree."""
    q = torch.zeros((1, 256, 2, 128), dtype=torch.bfloat16)
    mask = torch.ones((1, 2, 1, 256 // _default_kv_tile()), dtype=torch.int32)
    with pytest.raises(ValueError, match="block_mask must be a bool tensor"):
        mha_v4(
            q,
            q,
            q,
            FP8,
            FP8,
            FP8,
            block_mask=mask,
        )


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two GPUs to mismatch devices"
)
def test_mha_v4_rejects_block_mask_on_another_device():
    q = torch.zeros((1, 256, 2, 128), dtype=torch.bfloat16, device="cuda:0")
    mask = torch.ones(
        (1, 2, 1, 256 // _default_kv_tile()), dtype=torch.bool, device="cuda:1"
    )
    with pytest.raises(ValueError, match="block_mask must be on the same device"):
        mha_v4(
            q,
            q,
            q,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            AttentionFormat.FP8,
            block_mask=mask,
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@_requires_sparse_row("fp8")
def test_mha_v4_sparse_rejects_empty_kv_block_indices():
    """Rows may be empty, but the ASM still dereferences the row base, so the buffer cannot be."""
    heads = 2
    kv_tile = _default_kv_tile()
    kv_tiles = 4
    q, k, v = _sparse_fp8_operands(sequence_k=kv_tiles * kv_tile, heads=heads)
    rows = heads  # batch 1, one query tile
    device = q.quantized.device
    fp8_format = native_fp8_format()
    with pytest.raises(RuntimeError, match="must be non-empty"):
        mha_v4_packed(
            q.quantized,
            k.quantized,
            v.quantized,
            q.descale,
            k.descale,
            v.descale,
            fp8_format,
            fp8_format,
            fp8_format,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            kv_block_indices=torch.zeros(0, dtype=torch.int32, device=device),
            lut_start=torch.zeros(rows, dtype=torch.int32, device=device),
            lut_count=torch.ones(rows, dtype=torch.int32, device=device),
        )


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 BF16 sparse")
def test_mha_v4_sparse_bf16_rejects_lut_beyond_lds_capacity():
    max_tiles = 8192
    bf16 = AttentionFormat.BF16
    kv_tile = _default_kv_tile(bf16, bf16)
    q = torch.zeros((1, 1, 1, 128), device="cuda", dtype=torch.bfloat16)
    backing = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
    kv = backing.as_strided((1, (max_tiles + 1) * kv_tile, 1, 128), (0, 0, 0, 1))
    indices = torch.zeros(1, device="cuda", dtype=torch.int32)
    row = torch.zeros(1, device="cuda", dtype=torch.int32)

    with pytest.raises(RuntimeError, match="holds 8192 LUT entries"):
        mha_v4_packed(
            q,
            kv,
            kv,
            q,
            kv,
            kv,
            AttentionFormat.BF16,
            AttentionFormat.BF16,
            AttentionFormat.BF16,
            AttentionScaleMode.NONE,
            AttentionScaleMode.NONE,
            AttentionScaleMode.NONE,
            kv_block_indices=indices,
            lut_start=row,
            lut_count=row,
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@_requires_sparse_row("fp8")
@pytest.mark.parametrize(
    "mutation,message",
    [
        pytest.param(
            "indices.fill_(9999)",
            "outside",
            id="index_out_of_range",
        ),
        pytest.param(
            "start.fill_(-1)",
            "negative",
            id="negative_start",
        ),
    ],
)
def test_mha_v4_sparse_validation_rejects_malformed_lut(mutation, message):
    """Enable opt-in validation before AITER loads, without slowing the parent test process."""
    probe = f"""
from op_tests.test_mha_v4_sparse import (
    AttentionFormat,
    AttentionScaleMode,
    _default_kv_tile,
    _sparse_fp8_operands,
    _tile_mask,
    block_attn_mask_to_ragged_lut,
    mha_v4_packed,
    native_fp8_format,
)

heads = 2
kv_tiles = 4
q, k, v = _sparse_fp8_operands(
    sequence_k=kv_tiles * _default_kv_tile(), heads=heads
)
mask = _tile_mask(heads, kv_tiles, (0, 1))
indices, start, count = block_attn_mask_to_ragged_lut(
    mask, num_heads=heads, return_none_if_dense=False
)
{mutation}
fp8_format = native_fp8_format()
mha_v4_packed(
    q.quantized,
    k.quantized,
    v.quantized,
    q.descale,
    k.descale,
    v.descale,
    fp8_format,
    fp8_format,
    fp8_format,
    AttentionScaleMode.F32_PER_TENSOR,
    AttentionScaleMode.F32_PER_TENSOR,
    AttentionScaleMode.F32_PER_TENSOR,
    kv_block_indices=indices,
    lut_start=start,
    lut_count=count,
)
"""
    env = {**os.environ, "AITER_MHA_V4_VALIDATE_LUT": "1"}
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=AITER_ROOT_DIR,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode != 0
    assert message in result.stderr


# Two tiles is the baseline the growing counts are judged against; the rest span the region the
# suite never covered. 32 tiles is where the shipped mxfp8 object had lost a quarter of its cosine.
BASELINE_TILES = 2
GROWN_TILES = (4, 7, 8, 12, 16, 32)

# Each row carries its own quantization error, so the bar is that accuracy does not DEGRADE with
# tile count. Only a modest absolute floor is applied on top, to catch a row that is broken outright.
MAX_DEGRADATION = 0.01
ABSOLUTE_FLOOR = 0.95


def _reference(q, k, v):
    qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))
    scores = torch.matmul(qf, kf.transpose(-1, -2)) * (qf.shape[-1] ** -0.5)
    return torch.matmul(torch.softmax(scores, dim=-1), vf).transpose(1, 2)


def _cosine(a, b):
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


def _run(recipe_name, tiles, heads=5, sequence_q=256, all_true=True):
    q_format, v_format, kwargs = SPARSE_RECIPES[recipe_name]
    sequence_k = tiles * _default_kv_tile(q_format, v_format)
    torch.manual_seed(41)
    q = torch.randn((1, sequence_q, heads, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, sequence_k, heads, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.zeros(
        (1, heads, -(-sequence_q // 256), tiles), device="cuda", dtype=torch.bool
    )
    mask[:] = all_true
    if not all_true:
        mask[:, :, :, ::2] = True
    out = mha_v4(q, k, v, q_format, q_format, v_format, block_mask=mask, **kwargs)
    torch.cuda.synchronize()
    return out, _reference(q, k, v), mask


@pytest.mark.parametrize("recipe_name", _recipe_name_params(sorted(SPARSE_RECIPES)))
def test_mha_v4_sparse_accuracy_holds_as_tile_count_grows(recipe_name):
    """A row's own small-tile-count accuracy is its baseline; growing the count must not erode it."""
    baseline_out, baseline_ref, _ = _run(recipe_name, BASELINE_TILES)
    baseline = _cosine(baseline_out, baseline_ref)
    assert torch.isfinite(
        baseline_out
    ).all(), f"{recipe_name}: non-finite at {BASELINE_TILES} tiles"
    assert baseline > ABSOLUTE_FLOOR, f"{recipe_name}: baseline cosine {baseline:.5f}"

    for tiles in GROWN_TILES:
        out, ref, _ = _run(recipe_name, tiles)
        assert torch.isfinite(out).all(), f"{recipe_name}: non-finite at {tiles} tiles"
        cosine = _cosine(out, ref)
        assert (
            cosine > ABSOLUTE_FLOOR
        ), f"{recipe_name}: cosine {cosine:.5f} at {tiles} tiles"
        assert cosine > baseline - MAX_DEGRADATION, (
            f"{recipe_name}: accuracy degrades with tile count -- "
            f"{baseline:.5f} at {BASELINE_TILES} tiles, {cosine:.5f} at {tiles}"
        )


@pytest.mark.parametrize("recipe_name", _recipe_name_params(sorted(SPARSE_RECIPES)))
def test_mha_v4_sparse_skipping_lut_holds_as_tile_count_grows(recipe_name):
    """Same sweep with a LUT that actually skips, so the walk has to jump rather than run affine."""
    for tiles in (4, 8, 16, 32):
        out, _, mask = _run(recipe_name, tiles, all_true=False)
        assert torch.isfinite(out).all(), f"{recipe_name}: non-finite at {tiles} tiles"
        assert mask.sum() > 0
        assert not bool(
            (out == 0).all()
        ), f"{recipe_name}: all-zero output at {tiles} tiles"


@pytest.mark.parametrize(
    "recipe_name", _recipe_name_params(["bf16", "i8fp8", "fp8", "mxfp8", "f6f8"])
)
def test_mha_v4_sparse_all_true_lut_matches_dense_bitwise(recipe_name):
    """An all-true LUT selects every tile, so the walk must reduce exactly to the dense one.

    Restricted to the rows whose sparse and dense objects share a P quantization. The MX-V sparse
    rows are FP6-P builds of their own, and the BF16/FP8 sparse row scales its FP8 P differently
    from the dense one, so those are compared by cosine instead.
    """
    q_format, v_format, kwargs = SPARSE_RECIPES[recipe_name]
    for tiles in (2, 8, 16):
        sequence_k = tiles * _default_kv_tile(q_format, v_format)
        torch.manual_seed(41)
        q = torch.randn((1, 256, 5, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn((1, sequence_k, 5, 128), device="cuda", dtype=torch.bfloat16)
        v = torch.randn_like(k)
        mask = torch.ones((1, 5, 1, tiles), device="cuda", dtype=torch.bool)
        sparse = mha_v4(
            q, k, v, q_format, q_format, v_format, block_mask=mask, **kwargs
        )
        dense = mha_v4(q, k, v, q_format, q_format, v_format, **kwargs)
        torch.cuda.synchronize()
        assert torch.equal(sparse, dense), (
            f"{recipe_name}: all-true LUT differs from dense at {tiles} tiles "
            f"(cosine {_cosine(sparse, dense):.7f})"
        )


_SOL_ATTN_SOFTMAX_SCALE = 128**-0.5


class _SolAttnRecipe(NamedTuple):
    """One mode-2 manifest row: its formats, its quantizers, and how to dequantize it again.

    The recipes split on a single question -- whether an operand's scale varies along the sequence
    axis that pooling reduces. A per-tensor descale does not, so the pooled operand reuses it
    untouched and the kernarg's scale slots stay NULL. An E8M0 1x32 scale does, so that operand has
    to pool in dequantized space and hand the kernel a pooled scale of its own, which is the only
    reason those slots exist.
    """

    id: str
    qk_format: AttentionFormat
    qk_scale_mode: AttentionScaleMode
    quantize_q: Callable
    quantize_k: Callable
    quantize_v: Callable = quantize_fp8
    v_format: AttentionFormat = native_fp8_format()
    v_scale_mode: AttentionScaleMode = AttentionScaleMode.F32_PER_TENSOR
    # Named when the operands' stored codes are sub-byte and permuted, so neither pooling nor a
    # reference can read them back; see SOL_ATTN_PACKED_FORMATS.
    packed_format: str | None = None
    # V's packed format when it differs from K's, which it does wherever V's name also carries its
    # layout (the *_fp6_p ones), and the only packed operand of f8f6, whose FP8 Q/K are addressable.
    # Left None, V inherits K's packing only if it has K's format.
    v_packed_format: str | None = None
    v_pack: AttentionPack = AttentionPack.DEFAULT
    oracle_floor: float = 0.999
    # How far the corrected oracle has to beat the uncorrected one. Any margin at all proves the
    # correction reaches the output; a row whose exact pass is lossy enough to leave the correction
    # a large share of the total can demand more.
    keep_or_drop_margin: float = 0.0

    @property
    def k_carries_pooled_scale(self) -> bool:
        return self.qk_scale_mode == AttentionScaleMode.E8M0_PER_1X32

    @property
    def v_carries_pooled_scale(self) -> bool:
        return self.v_scale_mode == AttentionScaleMode.E8M0_PER_1X32

    @property
    def v_packed(self) -> str | None:
        if self.v_packed_format is not None:
            return self.v_packed_format
        return self.packed_format if self.v_format == self.qk_format else None

    @property
    def ref_softmax_scale(self):
        """The MX quantizers fold ``softmax_scale * log2(e)`` into Q.

        A reference built from the dequantized operands therefore has to divide that back out
        through its own softmax scale, rather than by rescaling the operand the kernel read.
        """
        return 1.0 / MHA_V4_LOG2E if self.k_carries_pooled_scale else None

    def kv_tile(self):
        """The KV tile this recipe's Sol-Attn row routes at.

        Built from the recipe's own scale modes rather than derived from its formats, because
        that is the whole difference between two rows here: FP8 and MXFP8 name the same three
        formats and are told apart only by the scale modes, so deriving them would ask about
        the wrong row.
        """
        return mha_v4_kv_tile(self.manifest_operands(), MHA_V4_SOL_MODE)

    def manifest_operands(self):
        """The manifest key this recipe's Sol-Attn row dispatches on."""
        return mha_v4_operands(
            self.qk_format,
            self.qk_format,
            self.v_format,
            self.qk_scale_mode,
            self.qk_scale_mode,
            self.v_scale_mode,
            self.v_pack,
        )

    def block_tile(self):
        """The (q_tile, kv_tile) this recipe's Sol-Attn row dispatches at."""
        return (256, self.kv_tile())

    def operands(self, sequence_k, heads=2, sequence_q=256, batch=1, seed=0):
        return self.quantize(
            *_sol_attn_raw_inputs(sequence_k, heads, sequence_q, batch, seed)
        )

    def quantize(self, q, k, v):
        # A packed operand carries its BF16 source along: pooling and the reference both have to
        # work from what the packer was given, because the codes it produced cannot be read back.
        keep_qk = self.packed_format is not None
        keep_v = self.v_packed is not None
        return (
            _Operand(*self.quantize_q(q), q if keep_qk else None),
            _Operand(*self.quantize_k(k), k if keep_qk else None),
            _Operand(*self.quantize_v(v), v if keep_v else None),
        )

    def prepare(
        self,
        q,
        k,
        v,
        beta,
        heads,
        block_tile=None,
        force_block_mask=None,
        k_variance=False,
    ):
        """Route and pool. K's scale goes in only when pooling cannot preserve it."""
        packed = self.packed_format
        tile_m, tile_n = self.block_tile() if block_tile is None else block_tile
        return sol_prepare(
            # Routing scores Q, and a packed Q is not element addressable either, so a packed
            # recipe routes on its source. Scale invariance is what makes that equivalent.
            q.source if packed is not None else q.quantized,
            k.quantized,
            v.quantized,
            beta=beta,
            num_heads=heads,
            # sol_prepare defaults BLOCK_N to the gfx950 tile. Pooling has to match the row
            # the launcher will pick or the pooled tensors are the wrong height outright, so take
            # it from the manifest the way mha_v4_sol() does.
            BLOCK_M=tile_m,
            BLOCK_N=tile_n,
            # A packed operand pools from its source and is quantized again, so it has no stored
            # scale to pool and passing one is an error.
            k_scale=(
                k.descale if self.k_carries_pooled_scale and packed is None else None
            ),
            v_scale=(
                v.descale
                if self.v_carries_pooled_scale and self.v_packed is None
                else None
            ),
            k_source=k.source,
            v_source=v.source,
            k_packed_format=packed,
            v_packed_format=self.v_packed,
            force_block_mask=force_block_mask,
            k_variance=k_variance,
        )

    @staticmethod
    def dequantize(quantized, descale):
        """A per-tensor descale multiplies; an E8M0 image expands over channel groups first."""
        if descale.dtype == torch.uint8:
            return _e8m0_dequantize(quantized, descale)
        if descale.dim() == 3:
            # A per-channel descale is [batch, heads, d], shared by every token.
            descale = descale.unsqueeze(1)
        return quantized.float() * descale.float()

    def reference_operands(self, q, k, v):
        """The exact-pass operands a reference should score, in the space the kernel reads them.

        An addressable recipe dequantizes the very codes the kernel was handed, which is what makes
        a disagreement with its own oracle mean something. A packed one cannot: the codes are
        sub-byte and permuted, and V's logical view is an aliased descriptor rather than an
        indexable tensor. It scores the BF16 source instead, with Q carrying the
        multiplier the packer folded into it -- for the addressable MX rows that multiplier comes
        back out of the codes on dequantization, so only this branch reapplies it.
        """
        v_ref = (
            self.dequantize(v.quantized, v.descale)
            if self.v_packed is None
            else v.source.float()
        )
        if self.packed_format is None:
            return (
                self.dequantize(q.quantized, q.descale),
                self.dequantize(k.quantized, k.descale),
                v_ref,
            )
        return (
            q.source.float() * mha_v4_q_multiplier(_SOL_ATTN_SOFTMAX_SCALE),
            k.source.float(),
            v_ref,
        )

    def dequantize_pooled(self, plan, name, source):
        """Dequantize a pooled operand with its own scale, or the source's when it has none."""
        if (self.packed_format if name == "mean_k" else self.v_packed) is not None:
            # Same unreadability as the exact-pass operands, so sol_prepare hands back the
            # values it quantized rather than expecting them to be recovered from the packed pair.
            return plan[f"{name}_pooled"].float()
        pooled_scale = plan[f"{name}_scale"]
        return self.dequantize(
            plan[name], source.descale if pooled_scale is None else pooled_scale
        )


def _fp8_recipe():
    return _SolAttnRecipe(
        id="fp8",
        qk_format=native_fp8_format(),
        qk_scale_mode=AttentionScaleMode.F32_PER_TENSOR,
        quantize_q=quantize_fp8_rotated,
        quantize_k=quantize_fp8_rotated,
    )


# Built once at module scope rather than per call. _sol_attn_launch falls back to it from inside a
# compiled region, and constructing the recipe there would make Dynamo materialize the field
# defaults with no source to guard on -- which it cannot do for the torch custom-op objects the
# quantizer fields hold.
_FP8_SOL_ATTN_RECIPE = _fp8_recipe()


def _quantize_bf16(t):
    """Store a BF16 operand as-is.

    The kernel's NONE scale mode makes the descale a placeholder it never reads, but the recipe's
    dequantize() path multiplies by it unconditionally, so hand back a unit scalar rather than
    special-casing the reference.
    """
    return t, torch.ones((), dtype=torch.float32, device=t.device)


def _mxfp8_quantize_q(q):
    return quantize_mxfp8_q(q, mha_v4_q_multiplier(_SOL_ATTN_SOFTMAX_SCALE))


def _mxfp4_quantize_q(q):
    return quantize_mxfp4_q(q, mha_v4_q_multiplier(_SOL_ATTN_SOFTMAX_SCALE))


# The MXFP4 K and V packers emit a flat backing buffer plus its scale, and the kernel wants the
# strided view over that buffer, so the recipe's quantizer hands back the view.
def _mxfp4_quantize_k(k):
    raw, scale = quantize_mxfp4_k(k)
    return mxfp4_k_view(raw, scale), scale


def _mxfp4_quantize_v_fp6_p(v):
    raw, scale = quantize_v_mxfp4_fp6_p(v)
    return mxfp4_v_view(raw, scale, v.shape[1]), scale


def _mxfp6_quantize_q(q):
    return quantize_mxfp6_q(q, mha_v4_q_multiplier(_SOL_ATTN_SOFTMAX_SCALE))


def _mxfp6_quantize_k(k):
    raw, scale = quantize_mxfp6_k(k)
    return mxfp6_k_view(raw, scale, *k.shape[:3])


_SOL_ATTN_RECIPES = [
    _FP8_SOL_ATTN_RECIPE,
    # The two BF16 Q/K rows. Neither pools anything it cannot reuse a source scale for -- bf16 has
    # no scale at all and bf16fp8's V descale is per-tensor -- so both leave the kernarg's pooled
    # scale slots NULL. They also tile KV at 64 while routing on 128-token blocks, which is the
    # manifest's business rather than this test's: everything here reads the tile back from it.
    _SolAttnRecipe(
        id="bf16",
        qk_format=AttentionFormat.BF16,
        qk_scale_mode=AttentionScaleMode.NONE,
        quantize_q=_quantize_bf16,
        quantize_k=_quantize_bf16,
        quantize_v=_quantize_bf16,
        v_format=AttentionFormat.BF16,
        v_scale_mode=AttentionScaleMode.NONE,
    ),
    _SolAttnRecipe(
        id="bf16fp8",
        qk_format=AttentionFormat.BF16,
        qk_scale_mode=AttentionScaleMode.NONE,
        quantize_q=_quantize_bf16,
        quantize_k=_quantize_bf16,
    ),
    _SolAttnRecipe(
        id="i8fp8",
        qk_format=AttentionFormat.INT8,
        qk_scale_mode=AttentionScaleMode.F32_PER_TENSOR,
        quantize_q=quantize_int8,
        quantize_k=quantize_int8,
    ),
    _SolAttnRecipe(
        id="mxfp8",
        qk_format=native_fp8_format(),
        qk_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
        quantize_q=_mxfp8_quantize_q,
        quantize_k=quantize_mxfp8_k,
    ),
    # The FP6-P rows: all three operands E8M0, P quantized to FP6 and V packed in the token order
    # that P contracts over, so V names its layout as well as its format. Every one of them
    # declares LSE, ragged KV, KV ranges, the Jensen term and sorted dispatch. With all three
    # E8M0, neither pooled operand can inherit a source descale, so both kernarg scale slots are
    # exercised at once.
    _SolAttnRecipe(
        id="mxfp4_fp6p",
        qk_format=AttentionFormat.MXFP4,
        qk_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
        quantize_q=_mxfp4_quantize_q,
        quantize_k=_mxfp4_quantize_k,
        quantize_v=_mxfp4_quantize_v_fp6_p,
        v_format=AttentionFormat.MXFP4,
        v_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
        packed_format="mxfp4",
        v_packed_format="mxfp4_fp6_p",
        v_pack=AttentionPack.V_FOR_FP6_P,
        # FP4 carries eight magnitude levels, so the exact pass alone only reaches ~0.98 of the
        # oracle before Sol-Attn contributes anything; what separates working from broken on this
        # row is the correction's distance to keep-or-drop, not the absolute cosine.
        oracle_floor=0.97,
        keep_or_drop_margin=0.05,
    ),
    _SolAttnRecipe(
        id="mxfp6_fp6p",
        qk_format=AttentionFormat.MXFP6,
        qk_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
        quantize_q=_mxfp6_quantize_q,
        quantize_k=_mxfp6_quantize_k,
        quantize_v=quantize_v_mxfp6_fp6_p,
        v_format=AttentionFormat.MXFP6,
        v_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
        packed_format="mxfp6",
        v_packed_format="mxfp6_fp6_p",
        v_pack=AttentionPack.V_FOR_FP6_P,
        oracle_floor=0.995,
        keep_or_drop_margin=0.05,
    ),
    # f8f6's per-tensor FP8 Q/K pool and route on their codes like the fp8 row, so only V carries
    # a pooled scale.
    _SolAttnRecipe(
        id="f8f6_fp6p",
        qk_format=AttentionFormat.FP8,
        qk_scale_mode=AttentionScaleMode.F32_PER_TENSOR,
        quantize_q=quantize_fp8_rotated,
        quantize_k=quantize_fp8_rotated,
        quantize_v=quantize_v_mxfp6_fp6_p,
        v_format=AttentionFormat.MXFP6,
        v_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
        v_packed_format="mxfp6_fp6_p",
        v_pack=AttentionPack.V_FOR_FP6_P,
        oracle_floor=0.995,
        keep_or_drop_margin=0.05,
    ),
    _SolAttnRecipe(
        id="f6f4_fp6p",
        qk_format=AttentionFormat.MXFP6,
        qk_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
        quantize_q=_mxfp6_quantize_q,
        quantize_k=_mxfp6_quantize_k,
        quantize_v=_mxfp4_quantize_v_fp6_p,
        v_format=AttentionFormat.MXFP4,
        v_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
        packed_format="mxfp6",
        v_packed_format="mxfp4_fp6_p",
        v_pack=AttentionPack.V_FOR_FP6_P,
        # Only V is FP4, so it sits between the mxfp4 and mxfp6 floors (measured ~0.992).
        oracle_floor=0.985,
        keep_or_drop_margin=0.05,
    ),
]

# Recipes whose only Sol-Attn row is the 64x64 one, so they have no default geometry to test at.
# f6f8's MXFP6 Q/K pool like the mxfp6 row's; its V's per-channel FP8 descale does not vary along
# the sequence, so pooled V reuses it the way a per-tensor one is reused.
_SOL_ATTN_FINE_ONLY_RECIPES = [
    _SolAttnRecipe(
        id="f6f8",
        qk_format=AttentionFormat.MXFP6,
        qk_scale_mode=AttentionScaleMode.E8M0_PER_1X32,
        quantize_q=_mxfp6_quantize_q,
        quantize_k=_mxfp6_quantize_k,
        quantize_v=quantize_v_fp8,
        v_scale_mode=AttentionScaleMode.F32_PER_CHANNEL,
        packed_format="mxfp6",
        oracle_floor=0.995,
    ),
]


def _sol_attn_geometry_params(ids=None):
    """(recipe, block_tile) params: each recipe at its default geometry (None), then at 64x64
    where it has a row there."""
    recipes = [r for r in _SOL_ATTN_RECIPES if ids is None or r.id in ids]
    fine_only = [r for r in _SOL_ATTN_FINE_ONLY_RECIPES if ids is None or r.id in ids]
    return [pytest.param(r, None, id=r.id) for r in recipes] + [
        pytest.param(
            r,
            _MHA_V4_FINE_TILE,
            marks=_requires_sol_row(r, _MHA_V4_FINE_TILE),
            id=f"{r.id}-64x64",
        )
        for r in recipes + fine_only
    ]


# mha_v4_sol() quantizes for you, and knows the per-tensor recipes and the FP6-P MX ones; the
# other MX rows go through mha_v4_packed with operands the caller quantized.
_SOL_ATTN_RAW_RECIPES = [
    r
    for r in _SOL_ATTN_RECIPES
    if not r.k_carries_pooled_scale or r.v_pack == AttentionPack.V_FOR_FP6_P
]


# gfx942 carries mode-2 rows for the two per-tensor recipes only; the MX ones skip themselves on
# the per-recipe manifest check below. Everything else in this section reads its geometry from
# its recipe's own geometry, so it retiles from 256x128 to 256x64 without further arch branching.
_MHA_V4_SOL_ATTN_ARCH = get_gfx() in ("gfx942", "gfx950")

_SOL_ATTN_FP8 = next(r for r in _SOL_ATTN_RECIPES if r.id == "fp8")


def _sol_attn_tiles(recipe):
    """The Sol-Attn geometries the manifest serves a recipe at, or () if it has none."""
    if not _MHA_V4_SOL_ATTN_ARCH:
        return ()
    return mha_v4_block_tiles(recipe.manifest_operands(), MHA_V4_SOL_MODE)


def _sol_attn_row_missing(recipe, block_tile=None):
    """Why the manifest cannot run `recipe` at block_tile (its default if None), or None if it can."""
    tiles = _sol_attn_tiles(recipe)
    if block_tile is None:
        return None if tiles else f"no {recipe.id} Sol-Attn row on {get_gfx()}"
    if tuple(block_tile) in tiles:
        return None
    q_tile, kv_tile = block_tile
    return f"no {q_tile}x{kv_tile} {recipe.id} Sol-Attn row on {get_gfx()}"


def _requires_sol_row(recipe, block_tile=None):
    reason = _sol_attn_row_missing(recipe, block_tile)
    return pytest.mark.skipif(reason is not None, reason=reason or "")


def _skip_without_sol_row(recipe, block_tile=None):
    reason = _sol_attn_row_missing(recipe, block_tile)
    if reason is not None:
        pytest.skip(reason)


def _sol_attn_launch(
    q,
    k,
    v,
    plan,
    mean_k=None,
    mean_v=None,
    block_bitmap=None,
    recipe=None,
    block_tile=None,
    return_lse=False,
    sorted_dispatch=None,
    kv_range_tokens=0,
    jensen=False,
):
    """Launch the Sol-Attn row over a sol_prepare() plan, overriding tensors if asked.

    jensen passes the plan's mean_k_var (and its scale), so the plan must have been prepared with
    k_variance=True.
    """
    recipe = _FP8_SOL_ATTN_RECIPE if recipe is None else recipe
    return mha_v4_packed(
        q.quantized,
        k.quantized,
        v.quantized,
        q.descale,
        k.descale,
        v.descale,
        recipe.qk_format,
        recipe.qk_format,
        recipe.v_format,
        recipe.qk_scale_mode,
        recipe.qk_scale_mode,
        recipe.v_scale_mode,
        kv_block_indices=plan["kv_block_indices"],
        lut_start=plan["lut_start"],
        lut_count=plan["lut_count"],
        mean_k=plan["mean_k"] if mean_k is None else mean_k,
        mean_v=plan["mean_v"] if mean_v is None else mean_v,
        block_bitmap=plan["block_bitmap"] if block_bitmap is None else block_bitmap,
        mean_k_scale=plan["mean_k_scale"],
        mean_v_scale=plan["mean_v_scale"],
        block_tile=block_tile,
        return_lse=return_lse,
        sorted_dispatch=sorted_dispatch,
        kv_range_tokens=kv_range_tokens,
        mean_k_var=plan["mean_k_var"] if jensen else None,
        mean_k_var_scale=plan["mean_k_var_scale"] if jensen else None,
        v_pack=recipe.v_pack,
    )


def _select_all_plan(plan, batch, heads, q_tiles, kv_tiles, device):
    """Replace a routed plan's selection with "every block exact", keeping its pooled tensors.

    Sol-Attn then has to reduce to its own exact pass: every column of the approximate pass is
    masked off, so the pooled tensors cannot reach the output at all.
    """
    mask = torch.ones(
        (batch, heads, q_tiles, kv_tiles), device=device, dtype=torch.bool
    )
    indices, start, count = block_attn_mask_to_ragged_lut(
        mask, num_heads=heads, return_none_if_dense=False
    )
    rows, bitmap_ds = plan["block_bitmap"].shape
    return {
        **plan,
        "kv_block_indices": indices,
        "lut_start": start,
        "lut_count": count,
        # All bits set: every block already computed exactly, including the padding bits above
        # num_kv_blocks that clip the last tile's overhang.
        "block_bitmap": torch.full(
            (rows, bitmap_ds), 0xFFFFFFFF, device=device, dtype=torch.uint32
        ),
        "block_attn_mask": mask,
    }


@pytest.mark.skipif(not _MHA_V4_SOL_ATTN_ARCH, reason="Sol-Attn validation")
@_requires_sol_row(_SOL_ATTN_FP8)
def test_mha_v4_sol_select_all_ignores_the_pooled_tensors():
    """The strongest host-path check that needs no tolerance and no oracle.

    Under an all-True selection the approximate pass is masked column by column, so the output
    must not depend on the pooled tensors at all. Any wrong kernarg offset, pooled stride, or
    bitmap row pitch breaks that: the correction leaks in and the two runs diverge.
    """
    heads, batch = 2, 1
    kv_tile = _default_kv_tile(mode=MHA_V4_SOL_MODE)
    kv_tiles = 4
    q, k, v = _sparse_fp8_operands(
        sequence_k=kv_tiles * kv_tile, heads=heads, batch=batch
    )
    routed = sol_prepare(
        q.quantized,
        k.quantized,
        v.quantized,
        beta=0.4,
        num_heads=heads,
        BLOCK_N=_default_kv_tile(mode=MHA_V4_SOL_MODE),
    )
    plan = _select_all_plan(
        routed, batch, heads, routed["num_q_tiles"], kv_tiles, q.quantized.device
    )

    with_pooled = _sol_attn_launch(q, k, v, plan)
    # Same launch, but pooled K/V that would move the output by a lot if they were ever read.
    scrambled = _sol_attn_launch(
        q,
        k,
        v,
        plan,
        mean_k=torch.full_like(plan["mean_k"], 4.0),
        mean_v=torch.full_like(plan["mean_v"], -4.0),
    )
    torch.cuda.synchronize()

    assert torch.equal(with_pooled, scrambled)
    assert torch.isfinite(with_pooled).all()
    assert with_pooled.abs().sum() > 0


@pytest.mark.skipif(not _MHA_V4_SOL_ATTN_ARCH, reason="Sol-Attn validation")
@_requires_sol_row(_SOL_ATTN_FP8)
@_requires_sparse_row("fp8")
def test_mha_v4_sol_select_all_is_no_less_accurate_than_the_sparse_row():
    """Cross-row check on the exact pass, against the oracle rather than against each other.

    Under an all-True mask both rows compute plain dense attention over the same quantized bytes,
    so the two are interchangeable in principle -- but not bit-for-bit: the deployed fp8 sparse
    code object predates the PKNORM softmax the Sol-Attn one is built with, so they approximate
    exp2 differently and land ~6e-4 of cosine apart. Asserting they agree to some tighter figure
    would only be measuring that gap. What actually matters is that neither approximation is worse
    than the other, so both are compared to the fp32 oracle instead; the shared ~4e-2 relative
    error against it is fp8 quantization, which both rows carry equally.
    """
    from aiter.test_mha_common import sol_attn_ref

    heads, batch = 2, 1
    kv_tile = _default_kv_tile(mode=MHA_V4_SOL_MODE)
    kv_tiles = 4
    q, k, v = _sparse_fp8_operands(
        sequence_k=kv_tiles * kv_tile, heads=heads, batch=batch
    )
    routed = sol_prepare(
        q.quantized,
        k.quantized,
        v.quantized,
        beta=0.4,
        num_heads=heads,
        BLOCK_N=_default_kv_tile(mode=MHA_V4_SOL_MODE),
    )
    plan = _select_all_plan(
        routed, batch, heads, routed["num_q_tiles"], kv_tiles, q.quantized.device
    )

    sol = _sol_attn_launch(q, k, v, plan)
    sparse = _sparse_fp8_launch(q, k, v, block_mask=plan["block_attn_mask"])
    torch.cuda.synchronize()

    oracle, _ = sol_attn_ref(
        q.quantized.float() * q.descale.float(),
        k.quantized.float() * k.descale.float(),
        v.quantized.float() * v.descale.float(),
        plan["block_attn_mask"],
        plan["mean_k"].float() * k.descale.float(),
        plan["mean_v"].float() * v.descale.float(),
        BLOCK_M=SOL_ATTN_TS_QO,
        BLOCK_N=kv_tile,
    )

    def error(actual):
        return ((actual.float() - oracle.float()).norm() / oracle.float().norm()).item()

    sol_error, sparse_error = error(sol), error(sparse)
    assert sol_error < 0.05, f"Sol-Attn relative error {sol_error}"
    assert sol_error < 1.1 * sparse_error, (
        f"Sol-Attn is less accurate than the sparse row on the same mask: "
        f"{sol_error} vs {sparse_error}"
    )


@pytest.mark.skipif(not _MHA_V4_SOL_ATTN_ARCH, reason="Sol-Attn validation")
@pytest.mark.parametrize(("recipe", "block_tile"), _sol_attn_geometry_params())
@pytest.mark.parametrize("tail", [0, 37])
@pytest.mark.parametrize("beta", [0.4, 1.0])
def test_mha_v4_sol_matches_the_oracle_and_beats_keep_or_drop(
    beta, tail, recipe, block_tile
):
    """The correction has to both track the oracle and be worth having.

    sol_attn_ref on the SAME routed mask is the accuracy target; the same oracle with
    correction=False is plain block-sparse attention over that mask, which is what Sol-Attn is
    supposed to improve on. Checking only the first would pass on a kernel that quietly dropped
    the correction, since a well-routed mask is already close on its own. A nonzero tail leaves
    the last KV block short, which the exact and pooled passes both have to bound.
    """
    _skip_without_sol_row(recipe, block_tile)
    from aiter.test_mha_common import sol_attn_ref

    heads, batch = 2, 1
    q_tile, kv_tile = recipe.block_tile() if block_tile is None else block_tile
    kv_tiles = 16
    q, k, v = recipe.operands(
        sequence_k=kv_tiles * kv_tile + tail, heads=heads, sequence_q=512, batch=batch
    )
    plan = recipe.prepare(q, k, v, beta=beta, heads=heads, block_tile=block_tile)
    fraction = plan["block_attn_mask"].float().mean().item()
    assert 0.02 < fraction < 0.9, f"degenerate routing at beta={beta}: {fraction}"

    sol = _sol_attn_launch(q, k, v, plan, recipe=recipe, block_tile=block_tile)
    torch.cuda.synchronize()

    # Score the oracle on the very values the kernel was handed, pooled tensors included: a
    # per-tensor recipe reuses the source descale, a block-granular one reads the pooled scale
    # instead, and getting that wrong here shows up as the kernel disagreeing with its own oracle.
    ref_args = (
        *recipe.reference_operands(q, k, v),
        plan["block_attn_mask"],
        recipe.dequantize_pooled(plan, "mean_k", k),
        recipe.dequantize_pooled(plan, "mean_v", v),
    )
    ref_kwargs = {
        "BLOCK_M": q_tile,
        "BLOCK_N": kv_tile,
        "softmax_scale": recipe.ref_softmax_scale,
    }
    reference, _ = sol_attn_ref(*ref_args, **ref_kwargs)
    keep_or_drop, _ = sol_attn_ref(*ref_args, **ref_kwargs, correction=False)

    def cosine(a, b):
        return torch.nn.functional.cosine_similarity(
            a.float().flatten(), b.float().flatten(), dim=0
        ).item()

    to_reference = cosine(sol, reference)
    to_keep_or_drop = cosine(sol, keep_or_drop)
    assert to_reference > recipe.oracle_floor, f"cosine to oracle {to_reference}"
    # The kernel is nearer the corrected oracle than the uncorrected one, i.e. it really does
    # carry the dropped blocks' zeroth-order mass rather than discarding it.
    assert (
        to_reference > to_keep_or_drop + recipe.keep_or_drop_margin
    ), f"correction not observable: {to_reference} vs {to_keep_or_drop}"


def test_mha_v4_packed_rejects_pooled_without_lut():
    dummy = torch.empty(0)
    fp8_format = native_fp8_format()
    with pytest.raises(ValueError, match="needs the ragged LUT triple"):
        mha_v4_packed(
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            fp8_format,
            fp8_format,
            fp8_format,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            mean_k=dummy,
            mean_v=dummy,
            block_bitmap=dummy,
        )


def test_mha_v4_packed_rejects_partial_pooled_triple():
    dummy = torch.empty(0)
    fp8_format = native_fp8_format()
    with pytest.raises(ValueError, match="all be set or all omitted"):
        mha_v4_packed(
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            dummy,
            fp8_format,
            fp8_format,
            fp8_format,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            AttentionScaleMode.F32_PER_TENSOR,
            mean_k=dummy,
            mean_v=dummy,
        )


@pytest.mark.skipif(not _MHA_V4_SOL_ATTN_ARCH, reason="Sol-Attn validation")
@_requires_sol_row(_SOL_ATTN_FP8)
@pytest.mark.skipif(
    get_gfx() == "gfx942",
    reason="gfx942 Sol-Attn writes zeros for an empty row instead of the pooled-only softmax: "
    "with no exact block the row never establishes the online-softmax state the correction pass "
    "normalizes against. sol_prepare keeps at least one exact block per row, so this is "
    "unreachable through the routed path; a caller hand-building an empty row gets zeros.",
)
def test_mha_v4_sol_empty_lut_rows_fall_back_to_the_pooled_softmax():
    """A row that selects nothing must degrade to the pooled-only answer, not to zeros or NaN.

    Such a row leaves the exact pass with no running softmax max, and the approximate pass has to
    recover one for the row to normalize at all. Every row is emptied here, so the whole output is
    the fallback and no amount of correct rows can hide a broken one.
    """
    from aiter.test_mha_common import sol_attn_ref

    heads, batch = 2, 1
    kv_tile = _default_kv_tile(mode=MHA_V4_SOL_MODE)
    kv_tiles = 8
    q, k, v = _sparse_fp8_operands(
        sequence_k=kv_tiles * kv_tile, heads=heads, sequence_q=512, batch=batch
    )
    plan = sol_prepare(
        q.quantized,
        k.quantized,
        v.quantized,
        beta=0.4,
        num_heads=heads,
        BLOCK_N=_default_kv_tile(mode=MHA_V4_SOL_MODE),
    )
    # Empty every row, keeping the bitmap consistent with it: nothing is exact, so nothing is
    # masked out of the approximate pass. The tail bits above num_kv_blocks stay set.
    plan["block_attn_mask"] = torch.zeros_like(plan["block_attn_mask"])
    plan["lut_count"] = torch.zeros_like(plan["lut_count"])
    plan["lut_start"] = torch.zeros_like(plan["lut_start"])
    rows, bitmap_ds = plan["block_bitmap"].shape
    device = q.quantized.device
    padding = (torch.arange(bitmap_ds * 32, device=device) >= kv_tiles).to(torch.int64)
    packed = (padding.reshape(bitmap_ds, 32) << torch.arange(32, device=device)).sum(
        dim=-1
    )
    plan["block_bitmap"] = (
        (packed & 0xFFFFFFFF).to(torch.uint32).expand(rows, bitmap_ds).contiguous()
    )

    sol = _sol_attn_launch(q, k, v, plan)
    torch.cuda.synchronize()

    def dequant(operand):
        return operand.quantized.float() * operand.descale.float()

    reference, _ = sol_attn_ref(
        dequant(q),
        dequant(k),
        dequant(v),
        plan["block_attn_mask"],
        plan["mean_k"].float() * k.descale.float(),
        plan["mean_v"].float() * v.descale.float(),
        BLOCK_M=SOL_ATTN_TS_QO,
        BLOCK_N=kv_tile,
    )
    assert torch.isfinite(sol).all(), "an empty row produced a non-finite output"
    assert (
        sol.abs().sum() > 0
    ), "an empty row produced a zero tile instead of the fallback"
    cosine = torch.nn.functional.cosine_similarity(
        sol.float().flatten(), reference.float().flatten(), dim=0
    ).item()
    assert cosine > 0.999, f"cosine to the pooled-only oracle {cosine}"


@pytest.mark.skipif(not _MHA_V4_SOL_ATTN_ARCH, reason="Sol-Attn validation")
@_requires_sol_row(_SOL_ATTN_FP8)
def test_mha_v4_sol_compiles_without_graph_breaks():
    """Routing and launch have to trace as one graph, which is why sol_prepare exists.

    Every shape it returns is a function of the input shapes and no host-side branch reads device
    data, so a caller can compile an attention layer around Sol-Attn without wrapping the routing
    in an opaque custom op of their own. A graph break here would take that away.
    """
    heads, batch = 2, 1
    kv_tile = _default_kv_tile(mode=MHA_V4_SOL_MODE)
    kv_tiles = 4
    q, k, v = _sparse_fp8_operands(
        sequence_k=kv_tiles * kv_tile, heads=heads, batch=batch
    )

    def call():
        plan = sol_prepare(
            q.quantized,
            k.quantized,
            v.quantized,
            beta=0.4,
            num_heads=heads,
            BLOCK_N=_default_kv_tile(mode=MHA_V4_SOL_MODE),
        )
        return _sol_attn_launch(q, k, v, plan)

    explained = torch._dynamo.explain(call)()
    assert explained.break_reasons == [], [
        str(reason.reason) for reason in explained.break_reasons
    ]


@pytest.mark.skipif(not _MHA_V4_SOL_ATTN_ARCH, reason="Sol-Attn validation")
@pytest.mark.parametrize("recipe", _SOL_ATTN_RAW_RECIPES, ids=lambda r: r.id)
@pytest.mark.parametrize("beta", [0.4, 1.0])
def test_mha_v4_sol_raw_matches_quantizing_and_routing_by_hand(beta, recipe):
    """All the raw entry point adds over the packed one is quantize-then-route, so pin exactly that.

    Deliberately not compared against dense attention: on random Gaussian operands there is no
    structure for any block mask to exploit, so every block carries similar mass and Sol-Attn sits
    around 0.78 cosine of dense at beta=0.4 -- a fact about the data, not about the kernel. The
    tests that do bound accuracy compare against sol_attn_ref on the mask the kernel was given.
    """
    _skip_without_sol_row(recipe)
    heads, batch = 2, 1
    kv_tile = recipe.kv_tile()
    q, k, v = _sol_attn_raw_inputs(
        sequence_k=16 * kv_tile, heads=heads, sequence_q=512, batch=batch, seed=3
    )

    sol = mha_v4_sol(
        q, k, v, recipe.qk_format, recipe.qk_format, recipe.v_format, beta=beta
    )

    operands = recipe.quantize(q, k, v)
    plan = recipe.prepare(*operands, beta=beta, heads=heads)
    by_hand = _sol_attn_launch(*operands, plan, recipe=recipe)
    torch.cuda.synchronize()

    assert torch.equal(sol, by_hand)
    assert torch.isfinite(sol).all()
    # Routing has to be doing something: neither degenerate all-exact nor all-approximate.
    fraction = plan["block_attn_mask"].float().mean().item()
    assert 0.02 < fraction < 0.9, f"degenerate routing at beta={beta}: {fraction}"


@pytest.mark.skipif(not _MHA_V4_SOL_ATTN_ARCH, reason="Sol-Attn validation")
@pytest.mark.parametrize("recipe", _SOL_ATTN_RAW_RECIPES, ids=lambda r: r.id)
def test_mha_v4_sol_raw_compile_parity(recipe):
    """The raw entry point picks its quantizers off the format enums, which Dynamo has to fold away.

    Those are host-side branches on Python values, so they specialize rather than break the graph --
    but only as long as nothing in them reads a tensor, which is exactly what would regress if a
    future recipe needed device-side routing to choose.
    """
    _skip_without_sol_row(recipe)
    heads, batch = 2, 1
    kv_tile = recipe.kv_tile()
    q, k, v = _sol_attn_raw_inputs(
        sequence_k=4 * kv_tile, heads=heads, batch=batch, seed=5
    )
    formats = (recipe.qk_format, recipe.qk_format, recipe.v_format)

    eager = mha_v4_sol(q, k, v, *formats, beta=0.4)
    compiled = torch.compile(mha_v4_sol, fullgraph=True)(q, k, v, *formats, beta=0.4)
    torch.cuda.synchronize()

    assert torch.equal(eager, compiled)


@pytest.mark.skipif(not _MHA_V4_SOL_ATTN_ARCH, reason="Sol-Attn validation")
@pytest.mark.parametrize("beta", [0.4, 1.0])
def test_mha_v4_sol_mxfp4_fills_both_pooled_scale_slots(beta):
    """mxfp4_fp6p has Q, K and V all E8M0, so neither pooled operand can inherit a source descale
    and both kernarg scale slots are live at once.

    Accuracy is covered by the recipe-parametrized oracle test. What is only checkable here is that
    BOTH slots are read: a NULL one is not an error state, it just means "keep reading the source
    image", so a fill that dropped either would still run and still look about as accurate. Only
    perturbing one at a time separates them.
    """
    recipe = next(r for r in _SOL_ATTN_RECIPES if r.id == "mxfp4_fp6p")
    _skip_without_sol_row(recipe)

    heads, batch = 2, 1
    q, k, v = recipe.operands(
        sequence_k=16 * recipe.kv_tile(), heads=heads, sequence_q=512, batch=batch
    )
    plan = recipe.prepare(q, k, v, beta=beta, heads=heads)
    assert (
        plan["mean_k_scale"] is not None and plan["mean_v_scale"] is not None
    ), "both operands are block-granular, so both slots should be filled"

    baseline = _sol_attn_launch(q, k, v, plan, recipe=recipe)
    torch.cuda.synchronize()
    assert torch.isfinite(baseline).all()

    def bumped(scale):
        """Bump every exponent, keeping the slack the V scale is read into.

        A plain elementwise copy would hand back a tightly sized allocation, which the V-scale read
        can run past, so the perturbed run and not the kernel would be what fails.
        """
        backing = scale.new_zeros((scale.numel() + 512,))
        backing[: scale.numel()] = (
            (scale.reshape(-1).int() + 1).clamp(max=255).to(torch.uint8)
        )
        return torch.as_strided(backing, scale.shape, scale.stride())

    for slot in ("mean_k_scale", "mean_v_scale"):
        perturbed = _sol_attn_launch(
            q, k, v, {**plan, slot: bumped(plan[slot])}, recipe=recipe
        )
        torch.cuda.synchronize()
        assert not torch.equal(baseline, perturbed), f"{slot} is not being read"


@pytest.mark.skipif(not _MHA_V4_SOL_ATTN_ARCH, reason="Sol-Attn validation")
def test_mha_v4_sol_reads_the_pooled_scale_it_was_given():
    """A NULL pooled-scale slot is not an error state, it just means "keep reading K's own scale".

    So a block-granular recipe that failed to pass one, or a kernarg fill that dropped it, would
    still run and still look plausible -- the approximate pass would simply rescale its pooled K by
    the wrong exponents. Perturbing the scale and requiring the output to move is what distinguishes
    a slot that is read from one that is merely populated.
    """
    recipe = next(
        r
        for r in _SOL_ATTN_RECIPES
        if r.k_carries_pooled_scale and not r.v_carries_pooled_scale
    )
    _skip_without_sol_row(recipe)
    heads = 2
    q, k, v = recipe.operands(sequence_k=8 * recipe.kv_tile(), heads=heads)
    plan = recipe.prepare(q, k, v, beta=0.4, heads=heads)
    assert plan["mean_k_scale"] is not None
    assert (
        plan["mean_v_scale"] is None
    ), "this recipe's V is per-tensor and pools for free"

    baseline = _sol_attn_launch(q, k, v, plan, recipe=recipe)
    bumped = dict(plan)
    bumped["mean_k_scale"] = (
        (plan["mean_k_scale"].int() + 1).clamp(max=255).to(torch.uint8)
    )
    perturbed = _sol_attn_launch(q, k, v, bumped, recipe=recipe)
    torch.cuda.synchronize()

    assert torch.isfinite(baseline).all()
    assert not torch.equal(baseline, perturbed), (
        "doubling every pooled K exponent changed nothing, so the kernel is not reading "
        "mean_k_scale"
    )


@pytest.mark.skipif(not _MHA_V4_SOL_ATTN_ARCH, reason="Sol-Attn validation")
@pytest.mark.parametrize("recipe", _SOL_ATTN_RECIPES, ids=lambda r: r.id)
def test_mha_v4_sol_pooled_scales_must_match_the_scale_modes(recipe):
    """Which operands need a pooled scale follows from the scale modes, so it is not the caller's
    to choose: supplying one for a per-tensor operand describes a read the kernel never does, and
    omitting one for a block-granular operand leaves it reading unpooled exponents.
    """
    _skip_without_sol_row(recipe)
    heads = 2
    q, k, v = recipe.operands(sequence_k=8 * recipe.kv_tile(), heads=heads)
    plan = recipe.prepare(q, k, v, beta=0.4, heads=heads)

    def launch(**scales):
        return mha_v4_packed(
            q.quantized,
            k.quantized,
            v.quantized,
            q.descale,
            k.descale,
            v.descale,
            recipe.qk_format,
            recipe.qk_format,
            recipe.v_format,
            recipe.qk_scale_mode,
            recipe.qk_scale_mode,
            recipe.v_scale_mode,
            kv_block_indices=plan["kv_block_indices"],
            lut_start=plan["lut_start"],
            lut_count=plan["lut_count"],
            mean_k=plan["mean_k"],
            mean_v=plan["mean_v"],
            block_bitmap=plan["block_bitmap"],
            v_pack=recipe.v_pack,
            **scales,
        )

    # Both directions of the contract, and both operands: whether a slot is required follows from
    # that operand's scale mode alone, which is exactly what the plan already encodes -- a pooled
    # scale is present when and only when the mode is block-granular. So the wrong call is to omit
    # one the plan filled, or to supply one the plan left empty.
    correct = {slot: plan[slot] for slot in ("mean_k_scale", "mean_v_scale")}
    for slot, value in correct.items():
        wrong = dict(correct)
        wrong[slot] = None if value is not None else plan["mean_k"]
        with pytest.raises(RuntimeError, match=slot):
            launch(**wrong)
            torch.cuda.synchronize()


def test_mha_v4_sol_rejects_recipes_without_a_manifest_row():
    dummy = torch.empty((1, 256, 1, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(NotImplementedError, match="f8f6 and f6f4 recipes only"):
        mha_v4_sol(
            dummy,
            dummy,
            dummy,
            AttentionFormat.MXFP6,
            AttentionFormat.MXFP6,
            native_fp8_format(),
        )


def _e8m0_operand(batch, seqlen, heads, head_dim, decades=12.0, seed=0):
    """(raw, data, scale) for an E8M0 operand whose exponent changes from token to token.

    A flat-magnitude operand would hide the whole problem: pooling stored codes is only wrong
    because the scale varies along the axis being pooled, so the test data has to vary along it.
    """
    torch.manual_seed(seed)
    raw = torch.randn(batch, seqlen, heads, head_dim, device="cuda")
    stagger = torch.exp2(
        torch.arange(seqlen, device="cuda").float().remainder(decades) - decades / 2
    )
    raw = raw * stagger.view(1, seqlen, 1, 1)
    data, scale = _e8m0_quantize(raw, dtypes.fp8)
    return raw, data, scale


def test_sol_attn_e8m0_quantization_round_trips():
    raw, data, scale = _e8m0_operand(2, 128, 3, 128)
    assert data.dtype == dtypes.fp8 and scale.dtype == torch.uint8
    assert scale.shape == (2, 128, 3, 4)
    # 255 is the E8M0 NaN encoding and must never be produced.
    assert int(scale.max()) <= 254

    back = _e8m0_dequantize(data, scale)
    assert torch.isfinite(back).all()
    cosine = torch.nn.functional.cosine_similarity(
        back.flatten(), raw.flatten(), dim=0
    ).item()
    assert cosine > 0.999, f"round trip lost too much: cosine {cosine}"

    # Iterating has to CONVERGE rather than drift. The first requantize is allowed to differ: it
    # sees the rounded values, so a group whose max landed well below the format limit gets a
    # tighter exponent than the raw data asked for, which is an improvement. The second cannot,
    # because that group's max is now within a factor of two of the limit by construction.
    once_data, once_scale = _e8m0_quantize(back, dtypes.fp8)
    twice_data, twice_scale = _e8m0_quantize(
        _e8m0_dequantize(once_data, once_scale), dtypes.fp8
    )
    assert torch.equal(twice_scale, once_scale)
    assert torch.equal(twice_data.view(torch.uint8), once_data.view(torch.uint8))


def test_sol_attn_pooling_a_block_granular_scale_must_not_pool_the_codes():
    """The reason the mode-2 kernarg carries pooled scales at all.

    For a per-tensor or per-channel descale, pooling the stored codes and reusing the descale is
    exact, because neither varies along the sequence axis being pooled. An E8M0 1x32 scale does, so
    the codes of one pooled row are not on a common footing and averaging them is meaningless --
    measured here at roughly half the cosine of doing it properly.
    """
    _, data, scale = _e8m0_operand(1, 1024, 2, 128)
    oracle = _sol_attn_block_mean(_e8m0_dequantize(data, scale), SOL_ATTN_TS_KV)
    cos = lambda a, b: torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()

    _, _, pooled = _sol_attn_pool_mx(data, scale, SOL_ATTN_TS_KV)
    proper = cos(pooled, oracle)
    naive = cos(_sol_attn_pool_reuse_descale(data, SOL_ATTN_TS_KV), oracle)

    assert proper > 0.99, f"dequantize-pool-requantize only reached {proper}"
    assert naive < 0.9, (
        f"code pooling scored {naive} on a per-token scale, so this test is no longer "
        "demonstrating why pooled scales are needed"
    )


def test_sol_attn_pooling_an_integer_operand_must_round_not_truncate():
    """i8fp8 stores K as int8, and casting a float mean to an integer dtype truncates toward zero.

    The float8 operands round on cast, so this only bites the integer recipes, and it bites them
    quietly: every pooled row creeps toward zero, which shrinks the approximate pass's scores rather
    than breaking anything outright.
    """
    codes = torch.randint(-128, 128, (1, 512, 2, 128), dtype=torch.int8, device="cuda")
    exact = _sol_attn_block_mean(codes.float(), SOL_ATTN_TS_KV)
    pooled = _sol_attn_pool_reuse_descale(codes, SOL_ATTN_TS_KV)

    assert pooled.dtype == codes.dtype
    assert torch.equal(pooled, exact.round().to(torch.int8))

    rounded_err = (pooled.float() - exact).abs().mean().item()
    truncated_err = (exact.to(torch.int8).float() - exact).abs().mean().item()
    assert rounded_err < 0.75 * truncated_err, (
        f"pooled codes are no better than truncation ({rounded_err} vs {truncated_err}), "
        "so the rounding step is not doing anything"
    )


def test_sol_prepare_pools_scales_only_for_the_operands_that_need_them():
    """K block-granular and V per-tensor is exactly the mxfp8 case."""
    _, k_data, k_scale = _e8m0_operand(1, 512, 2, 128)
    v_data = torch.randn(1, 512, 2, 128, device="cuda").to(dtypes.fp8)
    q = torch.randn(1, 512, 2, 128, device="cuda", dtype=torch.bfloat16)

    plan = sol_prepare(q, k_data, v_data, 0.5, num_heads=2, k_scale=k_scale)

    num_kv_blocks = 512 // SOL_ATTN_TS_KV
    assert plan["mean_k_scale"].shape == (1, num_kv_blocks, 2, 4)
    assert plan["mean_k_scale"].dtype == torch.uint8
    assert plan["mean_k"].shape == (1, num_kv_blocks, 2, 128)
    # V reuses its source descale, so it must NOT get one, and the kernarg slot stays null.
    assert plan["mean_v_scale"] is None

    # Every operand per-tensor is the fp8 recipe, which must be unaffected by any of this.
    per_tensor = sol_prepare(q, k_data, v_data, 0.5, num_heads=2)
    assert per_tensor["mean_k_scale"] is None
    assert per_tensor["mean_v_scale"] is None


def test_sol_prepare_with_pooled_scales_compiles_without_graph_breaks():
    _, k_data, k_scale = _e8m0_operand(1, 512, 2, 128)
    v_data = torch.randn(1, 512, 2, 128, device="cuda").to(dtypes.fp8)
    q = torch.randn(1, 512, 2, 128, device="cuda", dtype=torch.bfloat16)

    def routed(q, k, v, s):
        plan = sol_prepare(q, k, v, 0.5, num_heads=2, k_scale=s)
        return plan["mean_k"], plan["mean_k_scale"], plan["block_bitmap"]

    eager = routed(q, k_data, v_data, k_scale)
    traced = torch.compile(routed, fullgraph=True)(q, k_data, v_data, k_scale)
    for got, want in zip(traced, eager):
        assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))


# (id, Q/K format, V format, dense tolerance in nats). None is this GPU's FP8 encoding, which is
# gfx-dependent, so it cannot be spelled at module scope.
_LSE_ROWS = (
    ("bf16", AttentionFormat.BF16, AttentionFormat.BF16, 0.05),
    ("bf16fp8", AttentionFormat.BF16, None, 0.05),
    # The FP8 row's denominator is summed from the same FP8 P that PV consumes, so its LSE is an
    # order of magnitude looser than the BF16 rows' -- the residual is set by how peaked each
    # row's softmax is, not by a constant offset.
    ("fp8", None, None, 0.09),
    # INT8 and MXFP8 feed PV the same FP8 P as the FP8 row, and so share its tolerance.
    ("i8fp8", AttentionFormat.INT8, None, 0.09),
    ("mxfp8", None, None, 0.09),
    ("f6f8", AttentionFormat.MXFP6, None, 0.09),
    # The FP6-P rows, scored against the BF16 sources the way every row here is, so the residual
    # is mostly their own Q/K rounding: FP4's is the coarsest in the file.
    ("mxfp4_fp6p", AttentionFormat.MXFP4, AttentionFormat.MXFP4, 0.2),
    ("mxfp6_fp6p", AttentionFormat.MXFP6, AttentionFormat.MXFP6, 0.09),
    ("f8f6_fp6p", None, AttentionFormat.MXFP6, 0.09),
    ("f6f4_fp6p", AttentionFormat.MXFP6, AttentionFormat.MXFP4, 0.09),
)


# Rows whose sorted-sparse code object also carries the store. It is a separate build of the
# source and a separate manifest column, so it is listed rather than inferred from the dense rows.
_LSE_FP6_P_IDS = ("mxfp4_fp6p", "mxfp6_fp6p", "f8f6_fp6p", "f6f4_fp6p")


_LSE_SPARSE_IDS = ("bf16", "bf16fp8", "fp8", "i8fp8", "mxfp8", "f6f8", *_LSE_FP6_P_IDS)


_LSE_GFX950 = pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 LSE")


def _lse_ids(ids):
    """MHA v4 exports a validated LSE on gfx950 only."""
    return [pytest.param(i, marks=_LSE_GFX950) for i in ids]


def _lse_sparse_tile_params():
    """(row id, block_tile) for every sparse LSE row: None for its default tile, then 64x64."""
    return [pytest.param(i, None, marks=_LSE_GFX950, id=i) for i in _LSE_SPARSE_IDS] + [
        pytest.param(
            i,
            _MHA_V4_FINE_TILE,
            marks=[
                _LSE_GFX950,
                _requires_sparse_row(i.removesuffix("_fp6p"), _MHA_V4_FINE_TILE),
            ],
            id=f"{i}-64x64",
        )
        for i in _LSE_SPARSE_IDS
    ]


def _lse_row(row_id):
    """((q, k, v) formats, tolerance) for an LSE row, resolving None to this GPU's FP8 encoding."""
    _, qk_format, v_format, tolerance = next(r for r in _LSE_ROWS if r[0] == row_id)
    fp8 = native_fp8_format()
    qk = fp8 if qk_format is None else qk_format
    return (qk, qk, fp8 if v_format is None else v_format), tolerance


def _lse_scale_modes(row_id):
    formats, _ = _lse_row(row_id)
    if row_id != "mxfp8":
        return scale_modes_for_formats(*formats)
    e8m0 = AttentionScaleMode.E8M0_PER_1X32
    return e8m0, e8m0, AttentionScaleMode.F32_PER_TENSOR


def _lse_launch(row_id, q, k, v, **kwargs):
    """MXFP8 shares FP8's formats, so raw mha_v4 reaches it through its scale modes."""
    if row_id == "mxfp8":
        return mha_v4(q, k, v, FP8, FP8, FP8, **_MX_SCALES, **kwargs)
    formats, _ = _lse_row(row_id)
    return mha_v4(q, k, v, *formats, **kwargs)


def _reference_scores(q, k, softmax_scale):
    qf, kf = (t.float().permute(0, 2, 1, 3) for t in (q, k))
    return qf @ kf.transpose(-1, -2) * softmax_scale


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 manifest")
@pytest.mark.parametrize("row_id", _lse_ids(row[0] for row in _LSE_ROWS))
def test_mha_v4_dense_lse_matches_logsumexp(row_id):
    """The natural log of the softmax denominator, not log2 and not the raw running max."""
    torch.manual_seed(23)
    batch, sequence, heads, head_dim = 2, 512, 3, 128
    q = torch.randn(
        (batch, sequence, heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    k, v = torch.randn_like(q), torch.randn_like(q)
    scale = head_dim**-0.5
    _, tolerance = _lse_row(row_id)

    _out, lse = _lse_launch(row_id, q, k, v, softmax_scale=scale, return_lse=True)
    torch.cuda.synchronize()

    assert lse.shape == (batch, heads, sequence)
    assert lse.dtype == torch.float32
    reference = torch.logsumexp(_reference_scores(q, k, scale), dim=-1)
    # Absolute, not relative: an LSE is a log, so the scale a relative bound would divide by is
    # the arbitrary one of the exponent, and a merge consumes the difference between two of them.
    assert (lse - reference).abs().max() < tolerance


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 manifest")
@pytest.mark.parametrize("row_id", _lse_ids(row[0] for row in _LSE_ROWS))
def test_mha_v4_asking_for_the_lse_does_not_change_the_output(row_id):
    """Bitwise, so the LSE store cannot be paid for by a different softmax path."""
    torch.manual_seed(29)
    q = torch.randn((1, 512, 2, 128), device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)

    plain = _lse_launch(row_id, q, k, v)
    with_lse, _ = _lse_launch(row_id, q, k, v, return_lse=True)
    torch.cuda.synchronize()
    assert torch.equal(plain, with_lse)


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 manifest")
@pytest.mark.parametrize(("row_id", "block_tile"), _lse_sparse_tile_params())
def test_mha_v4_sparse_lse_covers_only_the_selected_blocks(row_id, block_tile):
    """The property a ring merge rests on.

    An LSE over all keys would be wrong here in a way no output comparison catches: the outputs
    would still agree, because O is a ratio in which the unselected mass cancels, while the weights
    two ranks were merged with would be drawn from different distributions.
    """
    torch.manual_seed(31)
    batch, sequence, heads, head_dim = 2, 1024, 2, 128
    formats, tolerance = _lse_row(row_id)
    q = torch.randn(
        (batch, sequence, heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    k, v = torch.randn_like(q), torch.randn_like(q)
    scale = head_dim**-0.5
    if block_tile is None:
        block_tile = mha_v4_block_tile(
            mha_v4_operands(
                *formats,
                *_lse_scale_modes(row_id),
                _v_pack(formats[0], formats[2]),
            ),
            MHA_V4_SPARSE_MODE,
        )
    q_tile, kv_tile = block_tile

    mask = torch.zeros(
        (batch, heads, -(-sequence // q_tile), sequence // kv_tile),
        dtype=torch.bool,
        device="cuda",
    )
    mask[..., ::2] = True
    _, lse = _lse_launch(
        row_id,
        q,
        k,
        v,
        softmax_scale=scale,
        block_mask=mask,
        block_tile=block_tile,
        return_lse=True,
    )
    torch.cuda.synchronize()

    keep = mask[:, :, 0, :].repeat_interleave(kv_tile, dim=-1)
    scores = _reference_scores(q, k, scale)
    selected = torch.logsumexp(
        scores.masked_fill(~keep[:, :, None, :], -math.inf), dim=-1
    )
    assert (lse - selected).abs().max() < tolerance
    # And it is not the all-keys LSE, which the selection makes a strictly larger number.
    assert (torch.logsumexp(scores, dim=-1) - lse).min() > 0.1


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 manifest")
@pytest.mark.parametrize(
    ("recipe_id", "tolerance"),
    [
        ("bf16", 0.05),
        ("bf16fp8", 0.05),
        ("fp8", 0.09),
        ("i8fp8", 0.09),
        pytest.param(
            "mxfp8",
            0.09,
            marks=pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MXFP8"),
        ),
        # Scored against the BF16 sources, so FP4's own Q/K rounding is in the residual.
        pytest.param(
            "mxfp4_fp6p",
            0.2,
            marks=pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 FP6-P rows"),
        ),
        *(
            pytest.param(
                recipe_id,
                tolerance,
                marks=pytest.mark.skipif(
                    get_gfx() != "gfx950", reason="gfx950 FP6-P rows"
                ),
            )
            for recipe_id, tolerance in (
                ("mxfp6_fp6p", 0.09),
                ("f8f6_fp6p", 0.09),
                ("f6f4_fp6p", 0.09),
            )
        ),
        # Its only Sol-Attn row is 64x64.
        pytest.param(
            "f6f8",
            0.09,
            marks=pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 64x64 rows"),
        ),
    ],
)
def test_mha_v4_sol_lse_is_the_joint_softmax_denominator(recipe_id, tolerance):
    """It counts the pooled proxy columns as well as the exact ones, which is what makes it merge.

    Checked against the reference's own joint logsumexp rather than against attention over the
    selected blocks, because those are different numbers here and only the first one composes: a
    ring merge adds numerators and denominators separately, so the proxy has to be in the
    denominator for the same reason it is in the numerator. An LSE over the exact columns alone
    would still look plausible and would silently misweight every merge.
    """
    from aiter.test_mha_common import sol_attn_ref

    torch.manual_seed(37)
    fine_only = {r.id for r in _SOL_ATTN_FINE_ONLY_RECIPES}
    recipe = next(
        r for r in _SOL_ATTN_RECIPES + _SOL_ATTN_FINE_ONLY_RECIPES if r.id == recipe_id
    )
    block_tile = _MHA_V4_FINE_TILE if recipe_id in fine_only else None
    _skip_without_sol_row(recipe, block_tile)
    heads, sequence = 2, 1024
    q_tile, kv_tile = block_tile or recipe.block_tile()
    operands = recipe.operands(sequence_k=sequence, heads=heads, sequence_q=sequence)
    plan = recipe.prepare(*operands, beta=1.0, heads=heads, block_tile=block_tile)
    fraction = plan["block_attn_mask"].float().mean().item()
    assert 0.02 < fraction < 0.9, f"degenerate routing: {fraction}"

    _, lse = _sol_attn_launch(
        *operands, plan, recipe=recipe, block_tile=block_tile, return_lse=True
    )
    torch.cuda.synchronize()

    q_op, k_op, v_op = operands
    _, reference = sol_attn_ref(
        *recipe.reference_operands(q_op, k_op, v_op),
        plan["block_attn_mask"],
        recipe.dequantize_pooled(plan, "mean_k", k_op),
        recipe.dequantize_pooled(plan, "mean_v", v_op),
        BLOCK_M=q_tile,
        BLOCK_N=kv_tile,
        softmax_scale=recipe.ref_softmax_scale,
    )
    assert lse.shape == reference.shape
    assert (lse - reference.float()).abs().max() < tolerance


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 rows declare sorted")
@pytest.mark.parametrize(("recipe", "block_tile"), _sol_attn_geometry_params())
def test_mha_v4_sol_sorted_dispatch_is_bitwise_raster(recipe, block_tile):
    """Heavy-first dispatch reorders the workgroups, never their work.

    The last query tile of every (batch, head) is forced all-exact, which puts its LUT a full
    level above the routed rows, so the table genuinely permutes the grid. A decode that got a
    head or batch field wrong would compute some tile into another's rows and break equality.
    The raster launch is the one an XCD-swizzling row relabels, so this also pins the table
    decode against the swizzle it bypasses.
    """
    _skip_without_sol_row(recipe, block_tile)
    torch.manual_seed(41)
    batch, heads, sequence = 2, 4, 1024
    q_tile, kv_tile = recipe.block_tile() if block_tile is None else block_tile
    q_tiles, kv_tiles = sequence // q_tile, sequence // kv_tile
    operands = recipe.operands(
        sequence_k=sequence, heads=heads, sequence_q=sequence, batch=batch
    )
    heavy = torch.zeros((1, 1, q_tiles, kv_tiles), dtype=torch.bool, device="cuda")
    heavy[..., -1, :] = True
    plan = recipe.prepare(
        *operands, beta=1.0, heads=heads, block_tile=block_tile, force_block_mask=heavy
    )
    counts = plan["lut_count"].view(batch, heads, q_tiles)
    assert (counts[..., -1] == kv_tiles).all()
    assert counts[..., :-1].float().mean() < 0.5 * kv_tiles, "degenerate routing"

    launch = functools.partial(
        _sol_attn_launch,
        *operands,
        plan,
        recipe=recipe,
        block_tile=block_tile,
        return_lse=True,
    )
    raster = launch(sorted_dispatch=False)
    ordered = launch(sorted_dispatch=True)
    default = launch()
    for launched in (ordered, default):
        assert torch.equal(launched[0], raster[0])
        assert torch.equal(launched[1], raster[1])


# Every gfx950 Sol row declares the Jensen term, and all but the BF16 ones take KV ranges.
_SOL_ATTN_GFX950_PARAMS = [
    pytest.param(
        *p.values,
        marks=[
            *p.marks,
            pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 Sol rows"),
        ],
        id=p.id,
    )
    for p in _sol_attn_geometry_params()
]


def _sol_attn_row_declares(recipe, block_tile, capability: str) -> bool:
    """Whether the Sol-Attn row `recipe` dispatches at `block_tile` declares a manifest column.

    Read from the CSV for the reason _mha_v4_lut_capacity gives: the launcher enforces these
    capabilities, and there is no Python accessor to ask.
    """
    q_tile, kv_tile = recipe.block_tile() if block_tile is None else block_tile
    asm_dir = os.environ.get("AITER_ASM_DIR", os.path.join(AITER_ROOT_DIR, "hsa"))
    manifest = os.path.join(asm_dir, get_gfx(), "fmha_v4_fwd", "fmha_v4_fwd.csv")
    wanted = {
        "q_format": recipe.qk_format,
        "k_format": recipe.qk_format,
        "v_format": recipe.v_format,
        "v_pack": recipe.v_pack,
        "q_scale_mode": recipe.qk_scale_mode,
        "k_scale_mode": recipe.qk_scale_mode,
        "v_scale_mode": recipe.v_scale_mode,
        "mode": MHA_V4_SOL_MODE,
        "ts_qo": q_tile,
        "ts_kv": kv_tile,
    }
    with open(manifest, newline="") as handle:
        for row in csv.DictReader(
            filter(lambda line: not line.startswith("#"), handle)
        ):
            if all(int(row[column]) == int(value) for column, value in wanted.items()):
                return bool(int(row[capability]))
    raise LookupError(f"no Sol-Attn row for {recipe.id} at {q_tile}x{kv_tile}")


def _rotated(x):
    out = torch.empty_like(x)
    rotate_activation_hd128(out, x.contiguous())
    return out


def _sol_attn_jensen_ref(
    q, k, v, mask, mean_k, mean_v, block_tile, softmax_scale, jensen
):
    """sol_attn_ref over whole blocks, with the Jensen term on every pooled logit.

    jensen is (q_rot, var_rot): Q and the per-block K variance in the basis the kernel holds them,
    so the term is 0.5 * softmax_scale^2 * sum_d q_rot_d^2 * var_rot[d]. Returns (out, lse).
    """
    from aiter.test_mha_common import block_attn_mask_to_token_mask

    q, k, v, mean_k, mean_v = (t.float() for t in (q, k, v, mean_k, mean_v))
    block_m, block_n = block_tile
    sequence_q, sequence_k = q.shape[1], k.shape[1]
    assert sequence_k % block_n == 0, "whole blocks only"
    exact = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)
    allowed = block_attn_mask_to_token_mask(
        mask, sequence_q, sequence_k, block_m, block_n, q.device
    )
    exact = exact.masked_fill(~allowed, float("-inf"))
    q_rot, var_rot = jensen
    pooled = (
        torch.einsum("bthd,bjhd->bhtj", q * softmax_scale, mean_k)
        + math.log(block_n)
        + 0.5
        * softmax_scale**2
        * torch.einsum("bthd,bjhd->bhtj", q_rot.float() ** 2, var_rot.float())
    )
    q_tile = torch.arange(sequence_q, device=q.device) // block_m
    pooled = pooled.masked_fill(mask[:, :, q_tile, :], float("-inf"))
    joint = torch.cat([exact, pooled], dim=-1)
    weights = torch.softmax(joint, dim=-1)
    out = torch.einsum("bhts,bshd->bthd", weights, torch.cat([v, mean_v], dim=1))
    return out, torch.logsumexp(joint, dim=-1)


@pytest.mark.parametrize(("recipe", "block_tile"), _SOL_ATTN_GFX950_PARAMS)
def test_mha_v4_sol_kv_ranges_merge_to_the_one_range_answer(recipe, block_tile):
    """KV ranges split the softmax, not the result: each range's exact and pooled columns are
    normalized on their own and merged by LSE, which is the joint softmax again. So the ranged
    launch has to land on the one-range one, and both on the oracle, with a ragged last range.
    A row that does not declare ranges has to refuse one rather than run unsplit."""
    from aiter.test_mha_common import sol_attn_ref

    _skip_without_sol_row(recipe, block_tile)
    heads = 2
    q_tile, kv_tile = recipe.block_tile() if block_tile is None else block_tile
    operands = recipe.operands(
        sequence_k=80 * kv_tile + 37, heads=heads, sequence_q=512, seed=3
    )
    plan = recipe.prepare(*operands, beta=1.0, heads=heads, block_tile=block_tile)
    launch = functools.partial(
        _sol_attn_launch, *operands, plan, recipe=recipe, block_tile=block_tile
    )
    if not _sol_attn_row_declares(recipe, block_tile, "kv_range"):
        with pytest.raises(RuntimeError, match="has no in-kernel KV range reset"):
            launch(kv_range_tokens=32 * kv_tile)
        return
    one, one_lse = launch(return_lse=True)
    ranged, ranged_lse = launch(return_lse=True, kv_range_tokens=32 * kv_tile)
    torch.cuda.synchronize()

    reference, reference_lse = sol_attn_ref(
        *recipe.reference_operands(*operands),
        plan["block_attn_mask"],
        recipe.dequantize_pooled(plan, "mean_k", operands[1]),
        recipe.dequantize_pooled(plan, "mean_v", operands[2]),
        BLOCK_M=q_tile,
        BLOCK_N=kv_tile,
        softmax_scale=recipe.ref_softmax_scale,
    )
    assert torch.isfinite(ranged).all() and torch.isfinite(ranged_lse).all()
    # Not bitwise: each range quantizes its P against its own running max.
    assert _cosine(ranged, one) > 0.998
    assert _cosine(ranged, reference) > recipe.oracle_floor
    assert (ranged_lse - one_lse).abs().max() < 0.05
    assert (ranged_lse - reference_lse.float()).abs().max() < 0.2


@pytest.mark.parametrize(("recipe", "block_tile"), _SOL_ATTN_GFX950_PARAMS)
def test_mha_v4_sol_jensen_term_is_the_block_k_variance(recipe, block_tile):
    """sol_prepare(k_variance=True) hands the kernel the block variance of K in the basis the
    kernel holds K in: rotated for the FP8 and MX recipes, as stored for BF16 and INT8. The kernel
    output has to match the oracle's second-order term at its own weight rather than at none or
    double it -- which is what a variance taken in the wrong basis, or read through the wrong
    slot, would look like. All the row's options then go on at once: sorted dispatch and, where
    the row takes them, KV ranges must not disturb it."""
    _skip_without_sol_row(recipe, block_tile)
    heads = 2
    q_tile, kv_tile = recipe.block_tile() if block_tile is None else block_tile
    operands = recipe.operands(
        sequence_k=64 * kv_tile, heads=heads, sequence_q=512, seed=4
    )
    plan = recipe.prepare(
        *operands, beta=1.0, heads=heads, block_tile=block_tile, k_variance=True
    )
    assert plan["mean_k_var"].shape == plan["mean_k"].shape
    if recipe.k_carries_pooled_scale:
        assert plan["mean_k_var_scale"].shape == plan["mean_k_scale"].shape
    else:
        assert plan["mean_k_var_scale"] is None
    launch = functools.partial(
        _sol_attn_launch, *operands, plan, recipe=recipe, block_tile=block_tile
    )

    plain = launch()
    corrected, corrected_lse = launch(jensen=True, return_lse=True)
    torch.cuda.synchronize()
    assert not torch.equal(plain, corrected)

    if recipe.packed_format is None:
        # Addressable codes dequantize straight into the kernel's basis, along with any multiplier
        # the quantizer folded into Q, which ref_softmax_scale then divides back out.
        q_op, k_op = operands[0], operands[1]
        q_rot = recipe.dequantize(q_op.quantized, q_op.descale)
        var_rot = _sol_attn_block_variance(
            recipe.dequantize(k_op.quantized, k_op.descale), kv_tile
        )
        softmax_scale = recipe.ref_softmax_scale or _SOL_ATTN_SOFTMAX_SCALE
    else:
        q_source, k_source = operands[0].source, operands[1].source
        q_rot = _rotated(q_source).float() * mha_v4_q_multiplier(
            _SOL_ATTN_SOFTMAX_SCALE
        )
        var_rot = _sol_attn_block_variance(_rotated(k_source).float(), kv_tile)
        softmax_scale = recipe.ref_softmax_scale
    ref_args = (
        *recipe.reference_operands(*operands),
        plan["block_attn_mask"],
        recipe.dequantize_pooled(plan, "mean_k", operands[1]),
        recipe.dequantize_pooled(plan, "mean_v", operands[2]),
        (q_tile, kv_tile),
        softmax_scale,
    )
    cosine = {
        factor: _cosine(
            corrected, _sol_attn_jensen_ref(*ref_args, (q_rot, var_rot * factor))[0]
        )
        for factor in (0.0, 1.0, 2.0)
    }
    assert cosine[1.0] > max(cosine[0.0], cosine[2.0]), cosine
    assert cosine[1.0] > recipe.oracle_floor, cosine
    # Half or one and a half times the term already misses by about 0.33 nats.
    reference_lse = _sol_attn_jensen_ref(*ref_args, (q_rot, var_rot))[1]
    assert (corrected_lse - reference_lse).abs().max() < 0.15

    everything = {"jensen": True, "return_lse": True}
    if _sol_attn_row_declares(recipe, block_tile, "kv_range"):
        everything["kv_range_tokens"] = 32 * kv_tile
    raster = launch(sorted_dispatch=False, **everything)
    ordered = launch(sorted_dispatch=True, **everything)
    torch.cuda.synchronize()
    assert torch.equal(raster[0], ordered[0]) and torch.equal(raster[1], ordered[1])
    assert _cosine(ordered[0], corrected) > 0.999


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 manifest")
def test_mha_v4_sol_sorted_dispatch_needs_a_row_that_declares_it():
    """Insisting on sorted dispatch fails loudly where it cannot be honoured.

    Every gfx950 Sol row declares it, so only a non-Sol launch has none to honour.
    """
    q = torch.randn((1, 512, 2, 128), device="cuda", dtype=torch.bfloat16)
    bf16, none = AttentionFormat.BF16, AttentionScaleMode.NONE
    with pytest.raises(ValueError, match="sorted_dispatch orders the Sol-Attn launch"):
        mha_v4_packed(
            q, q, q, q, q, q, bf16, bf16, bf16, none, none, none, sorted_dispatch=True
        )


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 manifest")
def test_mha_v4_block_tile_default_is_the_256_row_of_the_named_recipe():
    """Adding geometries must not move a recipe's default: its callers' masks are shaped for it.

    Asked per recipe because that is the only way the question has an answer once an arch serves
    more than one KV tile at this Q tile, which gfx950 now does.
    """
    fp8 = native_fp8_format()
    per_tensor = AttentionScaleMode.F32_PER_TENSOR
    operands = mha_v4_operands(fp8, fp8, fp8, per_tensor, per_tensor, per_tensor)
    q_tile, kv_tile = mha_v4_block_tile(operands, MHA_V4_SPARSE_MODE)
    assert q_tile == 256
    assert kv_tile == mha_v4_kv_tile(operands, MHA_V4_SPARSE_MODE)
    assert mha_v4_kv_tile_for_q_tile(q_tile, operands, MHA_V4_SPARSE_MODE) == kv_tile
    assert (q_tile, kv_tile) in mha_v4_block_tiles(operands, MHA_V4_SPARSE_MODE)


@pytest.mark.skipif(
    get_gfx() != "gfx950",
    reason="only gfx950 serves two KV tiles at a 256-row query tile",
)
def test_mha_v4_block_tile_refuses_to_guess_when_the_rows_disagree():
    """BF16 routes on a 64-token block here and every other recipe on 128.

    There is no arch-wide answer to give, and a mask cut for the wrong block does not miss by a
    little -- it addresses the wrong keys -- so the operand-blind query raises rather than
    returning one of the two.
    """
    with pytest.raises(ValueError, match="disagree on ts_kv"):
        mha_v4_block_tile()


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 manifest")
@pytest.mark.parametrize(
    "q_format, k_format, v_format, mode, expected",
    [
        ("fp8", "fp8", "fp8", MHA_V4_SPARSE_MODE, True),
        ("fp8", "fp8", "fp8", MHA_V4_SOL_MODE, True),
        ("BF16", "BF16", "BF16", MHA_V4_SOL_MODE, True),
        ("MXFP4", "MXFP4", "MXFP4", MHA_V4_SOL_MODE, True),
        ("FP8", "FP8", "MXFP6", MHA_V4_SOL_MODE, True),
        ("MXFP6", "MXFP6", "fp8", MHA_V4_SPARSE_MODE, True),
    ],
)
def test_mha_v4_ragged_kv_answers_per_row(q_format, k_format, v_format, mode, expected):
    """The capability belongs to a row, so a caller deciding whether to pad must name one."""

    def fmt(name):
        return native_fp8_format() if name == "fp8" else getattr(AttentionFormat, name)

    q_format, k_format, v_format = fmt(q_format), fmt(k_format), fmt(v_format)
    recipe = _resolve_raw_recipe(
        q_format, k_format, v_format, None, None, None, sparse=True
    )
    operands = mha_v4_operands(
        q_format, k_format, v_format, *recipe.scale_modes, recipe.v_pack
    )
    assert mha_v4_ragged_kv(operands, mode) is expected


@pytest.mark.skipif(get_gfx() != "gfx942", reason="gfx942 manifest")
def test_mha_v4_ragged_kv_is_false_on_gfx942():
    fp8 = native_fp8_format()
    per_tensor = AttentionScaleMode.F32_PER_TENSOR
    operands = mha_v4_operands(fp8, fp8, fp8, per_tensor, per_tensor, per_tensor)
    assert not mha_v4_ragged_kv(operands, MHA_V4_SOL_MODE)


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 manifest")
def test_mha_v4_kv_tile_for_q_tile_is_zero_when_unserved():
    """0 rather than a raise, so a caller can offer a geometry and fall back."""
    assert mha_v4_kv_tile_for_q_tile(48) == 0


@_requires_sparse_row("fp8", _MHA_V4_FINE_TILE)
def test_mha_v4_rejects_a_geometry_with_no_row():
    q = torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    fp8 = native_fp8_format()
    mask = torch.ones((1, 2, 2, 2), device="cuda", dtype=torch.bool)
    with pytest.raises(ValueError, match="has no 128x128 kernel"):
        mha_v4(q, k, k, fp8, fp8, fp8, block_mask=mask, block_tile=(128, 128))


@_requires_sparse_row("fp8", _MHA_V4_FINE_TILE)
def test_mha_v4_sparse_64x64_all_true_mask_matches_dense():
    """Not bit-equality, unlike the 256x128 sibling: a 64-row tile accumulates its online softmax
    in a different order than the dense row's 256, so the two differ by reassociation alone.
    """
    torch.manual_seed(41)
    q_tile, kv_tile = _MHA_V4_FINE_TILE
    q = torch.randn((1, 512, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(k)
    fp8 = native_fp8_format()
    mask = torch.ones(
        (1, 2, 512 // q_tile, 512 // kv_tile), device="cuda", dtype=torch.bool
    )
    dense = mha_v4(q, k, v, fp8, fp8, fp8)
    sparse = mha_v4(
        q, k, v, fp8, fp8, fp8, block_mask=mask, block_tile=_MHA_V4_FINE_TILE
    )
    torch.cuda.synchronize()
    assert torch.isfinite(sparse).all()
    assert _cosine(sparse, dense) > 0.999


@_requires_sparse_row("fp8", _MHA_V4_FINE_TILE)
def test_mha_v4_sparse_64x64_follows_the_lut():
    """The LUT has to be read in units of the dispatched kv_tile. If the kernel and the host
    disagreed about that, naming disjoint block sets would not give disjoint results."""
    torch.manual_seed(7)
    _q_tile, kv_tile = _MHA_V4_FINE_TILE
    q = torch.randn((1, 64, 1, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 256, 1, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    fp8 = native_fp8_format()
    outs = []
    for block in range(256 // kv_tile):
        mask = torch.zeros((1, 1, 1, 256 // kv_tile), device="cuda", dtype=torch.bool)
        mask[..., block] = True
        outs.append(
            mha_v4(
                q, k, v, fp8, fp8, fp8, block_mask=mask, block_tile=_MHA_V4_FINE_TILE
            )
        )
    torch.cuda.synchronize()
    for i, out in enumerate(outs):
        assert torch.isfinite(out).all()
        for j in range(i + 1, len(outs)):
            assert not torch.equal(out, outs[j]), f"blocks {i} and {j} gave one result"


@_requires_sparse_row("fp8", _MHA_V4_FINE_TILE)
@pytest.mark.parametrize("reversed_order", [False, True])
def test_mha_v4_sparse_64x64_honours_a_mask_that_varies_per_query_tile(reversed_order):
    """Every other sparse case here names one block set and gives it to every query tile, which
    is the one shape of mask that cannot separate the two KV pointers: if K ignored the LUT and
    stayed on block 0 while V followed it, a mask whose blocks are all block 0 still reads the
    right K. Only a mask that differs per query tile pulls them apart. Both orders run because a
    tile index that happens to equal its block index is its own coincidence.
    """
    torch.manual_seed(5)
    q_tile, kv_tile = _MHA_V4_FINE_TILE
    seqlen, heads, head_dim = 1024, 2, 128
    q = torch.randn((1, seqlen, heads, head_dim), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(k)
    q_tiles, kv_tiles = seqlen // q_tile, seqlen // kv_tile

    tiles = torch.arange(q_tiles, device="cuda")
    blocks = tiles % kv_tiles
    if reversed_order:
        blocks = (kv_tiles - 1) - blocks
    mask = torch.zeros((1, heads, q_tiles, kv_tiles), device="cuda", dtype=torch.bool)
    mask[0, :, tiles, blocks] = True

    out = mha_v4(
        q,
        k,
        v,
        AttentionFormat.BF16,
        AttentionFormat.BF16,
        AttentionFormat.BF16,
        block_mask=mask,
        block_tile=_MHA_V4_FINE_TILE,
    )
    torch.cuda.synchronize()

    qf, kf, vf = (t.float().transpose(1, 2) for t in (q, k, v))
    scores = torch.einsum("bhqd,bhkd->bhqk", qf, kf) * head_dim**-0.5
    tokens = mask.repeat_interleave(q_tile, 2).repeat_interleave(kv_tile, 3)
    weights = scores.masked_fill(~tokens, float("-inf")).softmax(-1)
    reference = torch.einsum("bhqk,bhkd->bhqd", weights, vf).transpose(1, 2)

    assert torch.isfinite(out).all()
    assert _cosine(out, reference) > 0.999


@_requires_sol_row(_SOL_ATTN_FP8, _MHA_V4_FINE_TILE)
def test_mha_v4_sol_64x64_select_all_matches_dense():
    """Selecting every block leaves the pooled pass with nothing to correct, so Sol-Attn at 64x64
    has to reduce to its own exact pass. This is what catches a bitmap grouped for the wrong tile:
    a misgrouped row would unmask blocks the exact pass already covered and double-count them.
    """
    torch.manual_seed(11)
    q = torch.randn((1, 512, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(k)
    fp8 = native_fp8_format()
    dense = mha_v4(q, k, v, fp8, fp8, fp8)
    # beta=-inf keeps every block, since the threshold is mean + beta * std.
    sol = mha_v4_sol(
        q, k, v, fp8, fp8, fp8, beta=float("-inf"), block_tile=_MHA_V4_FINE_TILE
    )
    torch.cuda.synchronize()
    assert torch.isfinite(sol).all()
    assert _cosine(sol, dense) > 0.999


@_requires_sol_row(_SOL_ATTN_FP8, _MHA_V4_FINE_TILE)
def test_mha_v4_sol_64x64_compiles_without_graph_breaks():
    """As for the default geometry: naming a non-default tile must not reintroduce the manifest
    read onto the traced path."""
    torch.manual_seed(17)
    q = torch.randn((1, 512, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    v = torch.randn_like(k)
    fp8 = native_fp8_format()

    def routed(q, k, v):
        return mha_v4_sol(
            q, k, v, fp8, fp8, fp8, beta=1.0, block_tile=_MHA_V4_FINE_TILE
        )

    eager = routed(q, k, v)
    traced = torch.compile(routed, fullgraph=True)(q, k, v)
    torch.cuda.synchronize()
    assert _cosine(traced, eager) > 0.999


def _mha_v4_lut_capacity(mode: int, q_tile: int, kv_tile: int) -> int:
    """The lut_max the manifest declares for one block-sparse geometry, or 0 if it has no row.

    Read from the CSV rather than from a Python accessor because there is deliberately none: the
    launcher is what enforces the ceiling, and a caller that needed the number to chunk a sequence
    would be working around the split the error message asks for.
    """
    asm_dir = os.environ.get("AITER_ASM_DIR", os.path.join(AITER_ROOT_DIR, "hsa"))
    manifest = os.path.join(asm_dir, get_gfx(), "fmha_v4_fwd", "fmha_v4_fwd.csv")
    with open(manifest, newline="") as handle:
        for row in csv.DictReader(
            filter(lambda line: not line.startswith("#"), handle)
        ):
            if (
                int(row["mode"]) == mode
                and int(row["ts_qo"]) == q_tile
                and int(row["ts_kv"]) == kv_tile
                and int(row["q_format"]) == int(native_fp8_format())
            ):
                return int(row["lut_max"])
    return 0


@_requires_sol_row(_SOL_ATTN_FP8, _MHA_V4_FINE_TILE)
def test_mha_v4_sol_rejects_a_key_length_past_the_lut_capacity():
    """The launcher refuses a key length past the row's lut_max, and accepts one exactly at it.

    The boundary is checked from both sides, because the interesting failure is an off-by-one that
    rejects a length the kernel handles: the capacity is exactly reachable, a row may select every
    block, and the pass at capacity is what says so. This row walks its LUT from memory, so lut_max
    is the manifest's placeholder rather than a staging capacity, and the pass at it is also the
    longest walk the row can be handed.

    Bounded by the shape rather than by the row's real maximum on purpose -- finding the latter
    means reading lut_count back from the device on every launch -- so a sequence is refused when
    any row COULD overrun, whatever this particular mask selected. That is also why one query tile
    is enough: the bound is on the key length, and self-attention at the placeholder would build a
    block mask of 2^32 entries.
    """
    q_tile, kv_tile = _MHA_V4_FINE_TILE
    capacity = _mha_v4_lut_capacity(MHA_V4_SOL_MODE, q_tile, kv_tile)
    assert capacity, "the 64x64 Sol-Attn row must declare a lut_max"

    fp8 = native_fp8_format()

    def run(blocks):
        q = torch.randn((1, q_tile, 1, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn(
            (1, blocks * kv_tile, 1, 128), device="cuda", dtype=torch.bfloat16
        )
        try:
            out = mha_v4_sol(
                q, k, k, fp8, fp8, fp8, beta=1.0, block_tile=_MHA_V4_FINE_TILE
            )
            torch.cuda.synchronize()
            return out
        finally:
            del q, k
            torch.cuda.empty_cache()

    assert torch.isfinite(
        run(capacity)
    ).all(), f"{capacity} blocks is exactly the declared capacity and has to run"

    with pytest.raises(RuntimeError, match=f"holds {capacity} LUT entries"):
        run(capacity + 1)


_SPARSE_BENCH_GFX = ["gfx950"]
_SPARSE_BENCH_Q_TILE = 256


def run_torch_mha_v4_sparse(q, k, v, block_mask, softmax_scale, kv_tile):
    """Masked dense attention in FP32: the KV blocks a LUT row drops never reach the softmax."""
    scores = (
        torch.matmul(
            q.transpose(1, 2).float(), k.transpose(1, 2).float().transpose(-1, -2)
        )
        * softmax_scale
    )
    keep = block_mask.repeat_interleave(_SPARSE_BENCH_Q_TILE, dim=2)[
        :, :, : q.shape[1], :
    ].repeat_interleave(kv_tile, dim=3)[..., : k.shape[1]]
    scores = scores.masked_fill(~keep, float("-inf"))
    return torch.matmul(
        torch.softmax(scores, dim=-1), v.transpose(1, 2).float()
    ).transpose(1, 2)


@benchmark()
def benchmark_mha_v4_sparse(batch, sequence_q, sequence_k, heads, density, dtype):
    """Benchmark the sorted block-sparse BF16 path against masked Torch attention."""
    head_dim = 128
    kv_tile = _default_kv_tile(AttentionFormat.BF16, AttentionFormat.BF16)
    softmax_scale = head_dim**-0.5
    torch.manual_seed(batch + sequence_q + sequence_k + heads)
    q = torch.randn((batch, sequence_q, heads, head_dim), device="cuda", dtype=dtype)
    k = torch.randn((batch, sequence_k, heads, head_dim), device="cuda", dtype=dtype)
    v = torch.randn_like(k)

    q_tiles = (sequence_q + _SPARSE_BENCH_Q_TILE - 1) // _SPARSE_BENCH_Q_TILE
    kv_tiles = (sequence_k + kv_tile - 1) // kv_tile
    keep = max(1, round(kv_tiles * density))
    block_mask = torch.zeros(
        (batch, heads, q_tiles, kv_tiles), device="cuda", dtype=torch.bool
    )
    # Keep the first `keep` tiles of every row: a fixed prefix makes the selected count exact, so
    # the FLOP and byte counts below describe the work the kernel really did.
    block_mask[..., :keep] = True
    reference = run_torch_mha_v4_sparse(q, k, v, block_mask, softmax_scale, kv_tile)

    formats = (AttentionFormat.BF16, AttentionFormat.BF16, AttentionFormat.BF16)
    candidates = {
        "sparse": lambda: mha_v4(
            q, k, v, *formats, softmax_scale=softmax_scale, block_mask=block_mask
        )
    }
    if keep == kv_tiles:
        # Dense reads every tile, so it only answers the same question as the reference when the
        # mask selects everything; at lower density it would be fast and wrong.
        candidates["dense"] = lambda: mha_v4(
            q, k, v, *formats, softmax_scale=softmax_scale
        )

    visited_k = keep * kv_tile
    flops = 4 * batch * heads * sequence_q * visited_k * head_dim
    elements = batch * heads * head_dim * (sequence_q * 2 + visited_k * 2)
    nbytes = elements * q.element_size()

    ret = {"gfx": get_gfx(), "kv_tiles": kv_tiles, "kept": keep}
    for name, candidate in candidates.items():
        output, us = run_perftest(candidate)
        err = checkAllclose(
            reference,
            output.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"{name}: block-sparse BF16",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in _SPARSE_BENCH_GFX:
        aiter.logger.warning(
            "MHA v4 block-sparse benchmark unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Benchmark sorted block-sparse BF16 MHA v4",
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[1])
    parser.add_argument("--sequence-q", type=int, nargs="*", default=[256, 512])
    parser.add_argument("--sequence-k", type=int, nargs="*", default=[1024, 2048])
    parser.add_argument("--heads", type=int, nargs="*", default=[8])
    parser.add_argument(
        "--density",
        type=float,
        nargs="*",
        default=[0.25, 0.5, 1.0],
        help="fraction of KV tiles each LUT row selects",
    )
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default=[dtypes.bf16]
    )
    args = parser.parse_args()

    rows = []
    for batch, sequence_q, sequence_k, heads, density, dtype in itertools.product(
        args.batch,
        args.sequence_q,
        args.sequence_k,
        args.heads,
        args.density,
        args.dtype,
    ):
        if dtype != dtypes.bf16:
            aiter.logger.warning("MHA v4 sparse benchmark skips dtype %s", dtype)
            continue
        rows.append(
            benchmark_mha_v4_sparse(
                batch, sequence_q, sequence_k, heads, density, dtype
            )
        )
    if rows:
        frame = pd.DataFrame(rows)
        aiter.logger.info(
            "MHA v4 block-sparse BF16 summary (markdown):\n%s",
            frame.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
