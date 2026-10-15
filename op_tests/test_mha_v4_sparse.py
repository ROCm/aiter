# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Block-sparse MHA v4: LUT handling, work-table ordering, tile-count scaling, per-recipe
geometry, and LSE.

Split out of test_mha_v4.py, which had grown past 2600 lines with the sparse cases forming
one contiguous block of it. Absorbs the former test_mha_v4_sparse_tile_scaling.py.
"""

import argparse
import csv
import glob
import itertools
import math
import os
import subprocess
import sys
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
    MHA_V4_SPARSE_MODE,
    AttentionFormat,
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
    mha_v4_sparse_work_table,
    native_fp8_format,
    scale_modes_for_formats,
)
from aiter.ops.mha_v4_quant import (
    quantize_fp8,
    quantize_fp8_rotated,
)
from aiter.ops.triton.attention.utils import (
    block_attn_mask_to_ragged_lut,
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


def _mha_v4_co_available(co_stem: str) -> bool:
    """Whether this arch deploys fwd_hd128_<co_stem>.co, with an _<ts_qo>x<ts_kv> suffix or none.

    The gfx942 objects carry no tile suffix: that arch ships a single geometry.
    """
    asm_dir = os.environ.get("AITER_ASM_DIR", os.path.join(AITER_ROOT_DIR, "hsa"))
    fwd_dir = os.path.join(asm_dir, get_gfx(), "fmha_v4_fwd")
    if get_gfx() == "gfx942":
        fwd_dir = os.path.join(fwd_dir, "MI300")
    return os.path.isfile(os.path.join(fwd_dir, f"fwd_hd128_{co_stem}.co")) or bool(
        glob.glob(os.path.join(fwd_dir, f"fwd_hd128_{co_stem}_[0-9]*x[0-9]*.co"))
    )


def _mha_v4_sparse_co_available() -> bool:
    return _mha_v4_co_available("fp8_sparse")


_MHA_V4_SPARSE_ARCH = get_gfx() in ("gfx942", "gfx950")


def _mha_v4_mxfp6_sparse_co_available() -> bool:
    """MXFP6 Q/K/V sparse is a gfx950-only row, and runs the FP6-P object."""
    return get_gfx() == "gfx950" and _mha_v4_co_available("mxfp6_fp6p_sparse")


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

# Every sparse row exists on gfx950; gfx942 ships the two per-tensor ones.
_GFX942_SPARSE_RECIPES = ("fp8", "i8fp8")

# The finer geometry, which gfx950 also serves for these recipes in the sparse mode.
_MHA_V4_FINE_TILE = (64, 64)
_FINE_TILE_RECIPES = (
    "fp8",
    "i8fp8",
    "bf16",
    "bf16fp8",
    "f8f6",
    "mxfp6",
    "f6f4",
    "mxfp4",
    "mxfp8",
    "f6f8",
)
_FINE_TILE_MARK = pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 64x64 rows")


def _sparse_recipe_params(names=None):
    """pytest params of (q/k format, v format, kwargs), each skipped where its row is not shipped."""
    params = []
    for name in SPARSE_RECIPES if names is None else names:
        marks = ()
        if name not in _GFX942_SPARSE_RECIPES:
            marks = pytest.mark.skipif(
                get_gfx() != "gfx950", reason=f"gfx950 {name} sparse"
            )
        params.append(pytest.param(*SPARSE_RECIPES[name], marks=marks, id=name))
    return params


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
        marks=pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 FP6-P rows"),
        id=f"{name}_fp6p",
    )
    for name, q_format, v_format in (
        ("mxfp6", AttentionFormat.MXFP6, AttentionFormat.MXFP6),
        ("f8f6", AttentionFormat.FP8, AttentionFormat.MXFP6),
        ("f6f4", AttentionFormat.MXFP6, AttentionFormat.MXFP4),
    )
]


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


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 sparse validation")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
@pytest.mark.parametrize(("q_format", "v_format", "kwargs"), _sparse_recipe_params())
def test_mha_v4_sparse_all_true_mask_matches_dense(q_format, v_format, kwargs):
    torch.manual_seed(41)
    q = torch.randn((1, 511, 5, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 512, 5, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = torch.ones(
        (1, 5, 2, 512 // _default_kv_tile(q_format, v_format)),
        device="cuda",
        dtype=torch.bool,
    )
    formats = (q_format, q_format, v_format)
    dense = mha_v4(q, k, v, *formats, **kwargs)
    sparse = mha_v4(q, k, v, *formats, block_mask=mask, **kwargs)
    torch.cuda.synchronize()
    _assert_sparse_matches_dense(sparse, dense)


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 rows are ragged")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
@pytest.mark.parametrize("tail", [44, 5])
@pytest.mark.parametrize(
    ("q_format", "v_format", "kwargs", "block_tile"),
    [
        pytest.param(*p.values, None, marks=p.marks, id=p.id)
        for p in _sparse_recipe_params()
    ]
    + [
        pytest.param(
            *p.values, _MHA_V4_FINE_TILE, marks=_FINE_TILE_MARK, id=f"{p.id}-64x64"
        )
        for p in _sparse_recipe_params(_FINE_TILE_RECIPES)
    ],
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


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MX sparse")
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
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


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MXFP6 validation")
@pytest.mark.skipif(
    not _mha_v4_mxfp6_sparse_co_available(),
    reason="sorted-sparse MXFP6 code object is not deployed",
)
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
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
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
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
@pytest.mark.parametrize(
    ("q_format", "v_format"),
    [
        pytest.param(
            AttentionFormat.BF16,
            AttentionFormat.BF16,
            marks=pytest.mark.skipif(
                get_gfx() != "gfx950", reason="gfx950 BF16 sparse"
            ),
            id="bf16",
        ),
        pytest.param(
            AttentionFormat.BF16,
            native_fp8_format(),
            marks=pytest.mark.skipif(
                get_gfx() != "gfx950", reason="gfx950 BF16 sparse"
            ),
            id="bf16fp8",
        ),
        pytest.param(
            native_fp8_format(),
            native_fp8_format(),
            id="fp8",
        ),
        pytest.param(
            AttentionFormat.MXFP4,
            AttentionFormat.MXFP4,
            marks=pytest.mark.skipif(
                get_gfx() != "gfx950", reason="gfx950 MXFP4 sparse"
            ),
            id="mxfp4",
        ),
        *_FP6_P_FORMAT_PARAMS,
    ],
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


def _sparse_fp8_operands(sequence_k, heads=2, sequence_q=256, batch=1, seed=0):
    """Quantize once so sparse and reference runs share descales exactly.

    Re-quantizing a KV slice would pick a different per-tensor amax, which shifts every
    value and hides whether the kernel read the KV blocks the LUT named.
    """
    torch.manual_seed(seed)
    q = torch.randn(
        (batch, sequence_q, heads, 128), device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn(
        (batch, sequence_k, heads, 128), device="cuda", dtype=torch.bfloat16
    )
    v = torch.randn_like(k)
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
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
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
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
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
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
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
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
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
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
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
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
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
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
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
@pytest.mark.skipif(
    not _mha_v4_sparse_co_available(),
    reason="sorted-sparse MHA v4 code object is not deployed",
)
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

requires_sparse = pytest.mark.skipif(
    get_gfx() != "gfx950" or not _mha_v4_sparse_co_available(),
    reason="gfx950 sorted-sparse MHA v4 code object is not deployed",
)


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


@requires_sparse
@pytest.mark.parametrize("recipe_name", sorted(SPARSE_RECIPES))
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


@requires_sparse
@pytest.mark.parametrize("recipe_name", sorted(SPARSE_RECIPES))
def test_mha_v4_sparse_skipping_lut_holds_as_tile_count_grows(recipe_name):
    """Same sweep with a LUT that actually skips, so the walk has to jump rather than run affine."""
    for tiles in (4, 8, 16, 32):
        out, _, mask = _run(recipe_name, tiles, all_true=False)
        assert torch.isfinite(out).all(), f"{recipe_name}: non-finite at {tiles} tiles"
        assert mask.sum() > 0
        assert not bool(
            (out == 0).all()
        ), f"{recipe_name}: all-zero output at {tiles} tiles"


@requires_sparse
@pytest.mark.parametrize("recipe_name", ["bf16", "i8fp8", "fp8", "mxfp8", "f6f8"])
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


def _lse_ids(ids):
    """MHA v4 exports a validated LSE on gfx950 only."""
    gfx950 = pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 LSE")
    return [pytest.param(i, marks=gfx950) for i in ids]


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
@pytest.mark.parametrize("row_id", _lse_ids(_LSE_SPARSE_IDS))
def test_mha_v4_sparse_lse_covers_only_the_selected_blocks(row_id):
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
    q_tile, kv_tile = mha_v4_block_tile(
        mha_v4_operands(
            *formats,
            *_lse_scale_modes(row_id),
            _v_pack(formats[0], formats[2]),
        ),
        MHA_V4_SPARSE_MODE,
    )

    mask = torch.zeros(
        (batch, heads, -(-sequence // q_tile), sequence // kv_tile),
        dtype=torch.bool,
        device="cuda",
    )
    mask[..., ::2] = True
    _, lse = _lse_launch(
        row_id, q, k, v, softmax_scale=scale, block_mask=mask, return_lse=True
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


def _mha_v4_fine_tile_available() -> bool:
    # Asked about the per-tensor FP8 row, which is the only one the tests below launch. A blind
    # query would raise here rather than answer: this enumerates one KV tile per Q tile, and at
    # 256 rows gfx950 serves two of them depending on the recipe.
    fp8 = native_fp8_format()
    per_tensor = AttentionScaleMode.F32_PER_TENSOR
    operands = mha_v4_operands(fp8, fp8, fp8, per_tensor, per_tensor, per_tensor)
    return _MHA_V4_FINE_TILE in mha_v4_block_tiles(operands)


_MHA_V4_FINE_TILE_REASON = "no 64x64 block-sparse MHA v4 row on this GPU"


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
        ("BF16", "BF16", "BF16", MHA_V4_SPARSE_MODE, True),
        ("MXFP4", "MXFP4", "MXFP4", MHA_V4_SPARSE_MODE, True),
        ("FP8", "FP8", "MXFP6", MHA_V4_SPARSE_MODE, True),
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
    assert not mha_v4_ragged_kv(operands, MHA_V4_SPARSE_MODE)


@pytest.mark.skipif(not _MHA_V4_SPARSE_ARCH, reason="gfx942/gfx950 manifest")
def test_mha_v4_kv_tile_for_q_tile_is_zero_when_unserved():
    """0 rather than a raise, so a caller can offer a geometry and fall back."""
    assert mha_v4_kv_tile_for_q_tile(48) == 0


@pytest.mark.skipif(not _mha_v4_fine_tile_available(), reason=_MHA_V4_FINE_TILE_REASON)
def test_mha_v4_rejects_a_geometry_with_no_row():
    q = torch.randn((1, 256, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn_like(q)
    fp8 = native_fp8_format()
    mask = torch.ones((1, 2, 2, 2), device="cuda", dtype=torch.bool)
    with pytest.raises(ValueError, match="has no 128x128 kernel"):
        mha_v4(q, k, k, fp8, fp8, fp8, block_mask=mask, block_tile=(128, 128))


@pytest.mark.skipif(not _mha_v4_fine_tile_available(), reason=_MHA_V4_FINE_TILE_REASON)
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


@pytest.mark.skipif(not _mha_v4_fine_tile_available(), reason=_MHA_V4_FINE_TILE_REASON)
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


@pytest.mark.skipif(not _mha_v4_fine_tile_available(), reason=_MHA_V4_FINE_TILE_REASON)
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
