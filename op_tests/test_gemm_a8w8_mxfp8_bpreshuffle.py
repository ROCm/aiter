# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP8 (1x32 e8m0 scales) bpreshuffle GEMM.

Runs every tuned (M, N, K) of this arch by default (an untuned gfx950 falls
back to a default shape list), or the shapes given by -mnk, through
gemm_a8w8_mxfp8_bpreshuffle and checks it against an fp32 reference.

Both gfx950 and gfx1250 read row-major x_scale [M, K/32]. gfx950 reads
row-major w_scale [N, K/32]; gfx1250 reads 32x4 shuffled w_scale, as laid out
by shuffle_mxfp8_bpreshuffle_scale (whose layout test_w_scale_layout checks).
"""

import argparse
import re
from pathlib import Path
from unittest import mock

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops import shuffle
from aiter.ops.gemm_op_a8w8 import _get_mxfp8_bpreshuffle_config
from aiter.ops.quant import per_group_quant_hip
from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale, shuffle_weight
from aiter.test_common import assertAllclose, benchmark, checkAllclose, perftest

# Shapes for a gfx950 without tuned rows: DeepSeek-V3 dense projections, ragged M included.
DEFAULT_MNK_GFX950 = [
    (m, n, k)
    for n, k in ((2112, 7168), (7168, 2048), (24576, 1536), (7168, 16384))
    for m in (1, 16, 37, 64, 300, 1024, 4096)
]

# gfx950, untuned K that 128 does not divide: no mxpsh tile runs it, so the
# entry falls back to the batched GEMM's heuristic (DeepSeek-V4.1's
# shared-expert down projection at TP8).
RAGGED_K_MNK_GFX950 = [(6, 5120, 288), (2048, 5120, 288)]


def run_torch(x, x_scale, weight, w_scale, w_block=(128, 128), x_block=32):
    """fp32 reference from row-major 1 x x_block A-scales and w_block B-scales."""
    m, k = x.shape
    n = weight.shape[0]
    x_scale = torch.exp2(x_scale.view(torch.uint8).float() - 127)
    x = (x.float().view(m, k // x_block, x_block) * x_scale[..., None]).view(m, k)
    w_scale = torch.exp2(w_scale.view(torch.uint8).float() - 127)
    w_scale = w_scale.repeat_interleave(w_block[0], 0)[:n]
    w_scale = w_scale.repeat_interleave(w_block[1], 1)
    return x @ (weight.float() * w_scale).T


@perftest()
def run_gemm(x, weight, x_scale, w_scale, dtype=dtypes.bf16):
    return aiter.gemm_a8w8_mxfp8_bpreshuffle(x, weight, x_scale, w_scale, dtype)


@benchmark()
def test_gemm(m, n, k, dtype=dtypes.bf16):
    x = torch.randn(m, k, dtype=dtypes.bf16)
    weight = (torch.randn(n, k) * 0.02).to(dtypes.fp8)
    w_scale = torch.randint(122, 127, (n, k // 32), dtype=torch.uint8)
    w_scale = w_scale.view(dtypes.fp8_e8m0)
    xq, x_scale = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, group_size=32, scale_type=dtypes.fp8_e8m0
    )
    ref = run_torch(xq, x_scale, weight, w_scale, (1, 32))
    out, us = run_gemm(
        xq,
        shuffle_weight(weight, layout=(16, 16)),
        x_scale,
        shuffle_mxfp8_bpreshuffle_scale(w_scale),
        dtype,
    )
    err = checkAllclose(ref, out.float(), msg=f"M={m} N={n} K={k}: ")
    libtype, kernel_name, splitk = _get_mxfp8_bpreshuffle_config(m, n, k)
    return {
        "libtype": libtype,
        "splitK": splitk,
        "us": round(us, 2),
        "TFLOPS": round(2 * m * n * k / us / 1e6, 1),
        "TB/s": round((m * k + n * k + m * n * 2) / us / 1e6, 2),
        "err": err,
        "kernelName": kernel_name,
    }


def test_w_scale_layout(n=64, kb=16):
    """[N, K/32] e8m0: gfx1250 family at n32k4 offsets, gfx950 unchanged; bad shapes raise."""
    w_scale = torch.randint(0, 255, (n, kb), dtype=torch.uint8, device="cpu")
    r = torch.arange(n, device="cpu")[:, None]
    c = torch.arange(kb, device="cpu")[None, :]
    n32k4 = (r // 32 * (kb // 4) + c // 4) * 128 + r % 32 * 4 + c % 4
    for gfx, offset in (
        ("gfx1250", n32k4),
        ("gfx1250-strict", n32k4),
        ("gfx950", None),
    ):
        with mock.patch.object(shuffle, "get_gfx_runtime", return_value=gfx):
            out = shuffle_mxfp8_bpreshuffle_scale(w_scale.view(dtypes.fp8_e8m0))
            for bad in ((48, 16), (64, 6)) if offset is not None else ():
                try:
                    shuffle_mxfp8_bpreshuffle_scale(torch.zeros(bad, device="cpu"))
                    raise AssertionError(f"{gfx}: [N, K/32]={bad} must be refused")
                except ValueError:
                    pass
        assert out.shape == w_scale.shape and out.dtype == dtypes.fp8_e8m0, gfx
        flat = out.view(torch.uint8).flatten()
        got = flat.view(n, kb) if offset is None else flat[offset]
        assert torch.equal(got, w_scale), f"{gfx}: wrong w_scale layout"
    aiter.logger.info("shuffle_mxfp8_bpreshuffle_scale layout: OK")


# (scale block, kernelName without _fsk, N, K): the generic kernel at split_k 2/4/8,
# a one-row tile, clusters along M and N next to the K-split cluster, and the _cp4
# and _qe_nlb modes of the K3 rows. The last six have no row-stripe epilogue (split_k
# 3/5/6/7, a 16-row tile at 8, two waves) and reduce on split 0; K tiles that split_k
# does not divide (28 over 5, 8 or 6, 33 over 5) split unevenly and have no unfused twin.
FUSED_SPLITK_CASES = (
    (
        128,
        "flydsl_mxfp8_128_bpreshuffle_wmma_t32x64x256_mw1_nw4_nb3_sk2_cm1_cn1",
        1536,
        7168,
    ),
    (
        128,
        "flydsl_mxfp8_128_bpreshuffle_wmma_t16x128x256_mw1_nw4_nb3_sk4_cm1_cn1",
        2176,
        7168,
    ),
    (
        128,
        "flydsl_mxfp8_128_bpreshuffle_wmma_t64x128x128_mw2_nw2_nb4_sk8_cm1_cn1",
        6400,
        7168,
    ),
    (
        128,
        "flydsl_mxfp8_128_bpreshuffle_wmma_t64x128x256_mw2_nw2_nb3_sk2_cm2_cn1",
        1536,
        4096,
    ),
    (
        128,
        "flydsl_mxfp8_128_bpreshuffle_wmma_t32x64x256_mw1_nw4_nb3_sk4_cm1_cn2",
        1536,
        4096,
    ),
    (
        128,
        "flydsl_mxfp8_128_bpreshuffle_wmma_t64x64x256_mw2_nw2_nb4_sk4_cm2_cn2_cp4",
        1536,
        7168,
    ),
    (
        128,
        "flydsl_mxfp8_128_bpreshuffle_wmma_t64x128x256_mw1_nw4_nb4_sk4_cm1_cn1_qe_nlb",
        1536,
        7168,
    ),
    (
        32,
        "flydsl_mxfp8_32_bpreshuffle_wmma_t32x64x256_mw1_nw4_nb3_sk4_cm1_cn1",
        1536,
        7168,
    ),
    (
        32,
        "flydsl_mxfp8_32_bpreshuffle_wmma_t64x128x128_mw2_nw2_nb4_sk2_cm2_cn1",
        1536,
        4096,
    ),
    (
        128,
        "flydsl_mxfp8_128_bpreshuffle_wmma_t16x64x256_mw1_nw4_nb4_sk7_cm1_cn1",
        1536,
        7168,
    ),
    (
        128,
        "flydsl_mxfp8_128_bpreshuffle_wmma_t16x64x256_mw1_nw4_nb4_sk5_cm1_cn1",
        1536,
        7168,
    ),
    (
        128,
        "flydsl_mxfp8_128_bpreshuffle_wmma_t64x32x128_mw2_nw2_nb4_sk5_cm1_cn2",
        7168,
        4224,
    ),
    (
        128,
        "flydsl_mxfp8_128_bpreshuffle_wmma_t16x64x256_mw1_nw4_nb3_sk8_cm1_cn1",
        1536,
        7168,
    ),
    (
        128,
        "flydsl_mxfp8_128_bpreshuffle_wmma_t32x64x256_mw1_nw2_nb4_sk2_cm1_cn1",
        1536,
        7168,
    ),
    (
        32,
        "flydsl_mxfp8_32_bpreshuffle_wmma_t32x64x256_mw1_nw4_nb3_sk6_cm1_cn1",
        1536,
        7168,
    ),
)
# M of each case: a single row, ragged tiles, and a tile count cluster_m divides or not.
FUSED_SPLITK_M = (1, 7, 33, 300, 640)
# (kernelName, N, K): the sk1 _qe kernels of the K3 rows.
MX128_DIRECT_CASES = (
    (
        "flydsl_mxfp8_128_bpreshuffle_wmma_t64x128x512_mw1_nw4_nb3_sk1_cm1_cn1_qe",
        2176,
        7168,
    ),
    (
        "flydsl_mxfp8_128_bpreshuffle_wmma_t128x128x256_mw2_nw2_nb4_sk1_cm1_cn2_qe",
        7168,
        1536,
    ),
)


def _fused_splitk_operands(block, m, n, k):
    """(xq, x_scale, weight, w_scale) with a 1 x block A scale, and a 128 x 128
    (block 128) or 32 x 32 (block 32) weight scale."""
    x = torch.randn(m, k, dtype=dtypes.bf16)
    xq, x_scale = per_group_quant_hip(
        x, quant_dtype=dtypes.fp8, group_size=block, scale_type=dtypes.fp8_e8m0
    )
    weight = (torch.randn(n, k) * 0.02).to(dtypes.fp8)
    w_scale = torch.randint(122, 127, (n // block, k // block), dtype=torch.uint8)
    return xq, x_scale, weight, w_scale.view(dtypes.fp8_e8m0)


@perftest()
def run_flydsl(entry, xq, x_scale, weight_shuffle, w_scale, out, kernel_name, **kw):
    return entry(xq, weight_shuffle, x_scale, w_scale, out, kernel_name, **kw)


@benchmark()
def test_fused_splitk(block, kernel_name, m, n, k):
    """The generic kernel's fused split-K gives the bytes of its unfused twin (where K
    divides evenly), and really skips the reduce kernel, eagerly and under graph replay.
    """
    from aiter.ops.flydsl import mxfp8_bpreshuffle_gemm_gfx1250 as flydsl_gemm

    flydsl_gemm._lazy_import()
    mx128 = block == 128
    entry = (
        flydsl_gemm.run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250
        if mx128
        else flydsl_gemm.run_gemm_a8w8_mxfp8_32_bpreshuffle_gfx1250
    )
    kw = {} if mx128 else {"w_scale_32x32": True}
    xq, x_scale, weight, w_scale = _fused_splitk_operands(block, m, n, k)
    ref = run_torch(xq, x_scale, weight, w_scale, (block, block), x_block=block)
    if mx128:  # the kernel reads the A scale as K-major storage of an (M, K/128) view
        x_scale = x_scale.t().contiguous().view(*x_scale.shape)
    w_shuf = shuffle_weight(weight, layout=(16, 16))
    reduces = {}

    def launch(name):
        out = torch.full((m, n), float("nan"), dtype=dtypes.bf16)
        with mock.patch.object(
            flydsl_gemm,
            "_compile_splitk_reduce",
            wraps=flydsl_gemm._compile_splitk_reduce,
        ) as reduce:
            res, us = run_flydsl(entry, xq, x_scale, w_shuf, w_scale, out, name, **kw)
        reduces[name] = reduce.call_count
        return res, us

    fused, us_fused = launch(kernel_name + "_fsk")
    assert reduces[kernel_name + "_fsk"] == 0, reduces
    err = assertAllclose(
        ref, fused.float(), tol_err_ratio=0, msg=f"M={m} N={n} K={k} fused: "
    )
    tile_k, split_k = map(int, re.search(r"x(\d+)_mw.*_sk(\d+)_", kernel_name).groups())
    us_unfused = None
    if k % (split_k * tile_k) == 0:
        unfused, us_unfused = launch(kernel_name)
        assert reduces[kernel_name] > 0, reduces
        assert torch.equal(fused, unfused), f"{kernel_name}: fused != unfused at M={m}"

    out = torch.full((m, n), float("nan"), dtype=dtypes.bf16)
    entry(xq, w_shuf, x_scale, w_scale, out, kernel_name + "_fsk", **kw)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        entry(xq, w_shuf, x_scale, w_scale, out, kernel_name + "_fsk", **kw)
    out.fill_(float("nan"))
    graph.replay()
    assert torch.equal(out, fused), f"{kernel_name}: graph replay differs at M={m}"
    flops, nbytes = 2 * m * n * k, m * k + n * k + m * n * 2
    return {
        "unfused us": us_unfused and round(us_unfused, 2),
        "fused us": round(us_fused, 2),
        "fused TFLOPS": round(flops / us_fused / 1e6, 1),
        "fused TB/s": round(nbytes / us_fused / 1e6, 2),
        "fused err": err,
    }


# (kernelName, N, K, M list): compute kernels whose last group of N tiles is partial
# (bounded_n; ps2 at nb2 / nb4 stores C per wave), clusters of 3 / 5 M tiles, M below
# the tile, A scale rows off a 16-byte boundary (M 77, 253, 301, 777), and generic
# tiles of 160 / 192 rows.
BOUNDED_CASES = (
    (
        "flydsl_mxfp8_128_bpreshuffle_compute_wmma_t256x256x128_mw2_nw2_nb4_sk1_cm1_cn2",
        1152,
        1536,
        (256, 300, 777),
    ),
    (
        "flydsl_mxfp8_128_bpreshuffle_compute_wmma_t256x256x128_mw2_nw2_nb2_sk1_cm2_cn1_ps2",
        1152,
        1536,
        (300, 777),
    ),
    (
        "flydsl_mxfp8_128_bpreshuffle_compute_wmma_t128x256x128_mw2_nw2_nb3_sk1_cm1_cn1_ps2",
        1152,
        1536,
        (77, 128, 300),
    ),
    (
        "flydsl_mxfp8_128_bpreshuffle_compute_wmma_t256x256x128_mw2_nw2_nb4_sk1_cm1_cn4",
        2176,
        7168,
        (100, 253, 256, 1000),
    ),
    (
        "flydsl_mxfp8_128_bpreshuffle_compute_wmma_t128x128x128_mw2_nw2_nb4_sk1_cm3_cn1",
        2176,
        7168,
        (300, 301, 384),
    ),
    (
        "flydsl_mxfp8_128_bpreshuffle_wmma_t32x128x128_mw2_nw2_nb2_sk1_cm5_cn1",
        12288,
        128,
        (160, 33),
    ),
    (
        "flydsl_mxfp8_128_bpreshuffle_wmma_t64x128x256_mw2_nw2_nb3_sk1_cm3_cn1",
        2048,
        1536,
        (192, 300),
    ),
    (
        "flydsl_mxfp8_128_bpreshuffle_wmma_t160x256x128_mw2_nw2_nb4_sk1_cm1_cn1",
        2048,
        7168,
        (129, 160, 300),
    ),
    (
        "flydsl_mxfp8_128_bpreshuffle_wmma_t192x256x128_mw2_nw2_nb4_sk1_cm1_cn1",
        2048,
        7168,
        (161, 192),
    ),
)


def test_mx128_kernel_direct(kernel_name, m, n, k):
    """An explicitly named sk1 mx128 kernel matches the reference, writes nothing outside
    its M x N output, launches no reduce kernel, and replays bitwise under a CUDA graph.
    """
    from aiter.ops.flydsl import mxfp8_bpreshuffle_gemm_gfx1250 as flydsl_gemm

    flydsl_gemm._lazy_import()
    entry = flydsl_gemm.run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250
    xq, x_scale, weight, w_scale = _fused_splitk_operands(128, m, n, k)
    ref = run_torch(xq, x_scale, weight, w_scale, (128, 128), x_block=128)
    x_scale = x_scale.t().contiguous().view(*x_scale.shape)
    w_shuf = shuffle_weight(weight, layout=(16, 16))

    # A store past column N lands in the next row, past the last row in a guard row.
    guarded = torch.full((m + 2, n), float("nan"), dtype=dtypes.bf16)
    out = guarded[1 : m + 1]
    with mock.patch.object(
        flydsl_gemm,
        "_compile_splitk_reduce",
        wraps=flydsl_gemm._compile_splitk_reduce,
    ) as reduce:
        entry(xq, w_shuf, x_scale, w_scale, out, kernel_name)
    assert reduce.call_count == 0, f"{kernel_name}: sk1 launched a reduce kernel"
    err = assertAllclose(
        ref, out.float(), tol_err_ratio=0, msg=f"M={m} N={n} K={k} {kernel_name}: "
    )
    assert guarded[[0, -1]].isnan().all(), f"{kernel_name}: wrote outside M={m} N={n}"

    eager = out.clone()
    out.fill_(float("nan"))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        entry(xq, w_shuf, x_scale, w_scale, out, kernel_name)
    graph.replay()
    assert torch.equal(out, eager), f"{kernel_name}: graph replay differs at M={m}"
    return {"err": err}


def test_fused_splitk_whole_row_passes():
    """The epilogue stores whole passes of rows_per_iter * unroll rows: a 192x256 tile with
    two splits would leave 32 of 96 rows of every stripe unwritten, so it must not take
    the row-stripe epilogue (the 96 rows of a split are not a whole number of 64-row
    passes); split 0 sums the tile instead.
    """
    from aiter.ops.flydsl import mxfp8_bpreshuffle_gemm_gfx1250 as flydsl_gemm
    from aiter.ops.flydsl.kernels import splitk_fused_epilogue_gfx1250 as epilogue

    flydsl_gemm._lazy_import()
    assert (flydsl_gemm._FUSED_UNROLL, flydsl_gemm._FUSED_MAX_PARTIAL_VECTORS) == (
        epilogue.UNROLL,
        epilogue.MAX_PARTIAL_VECTORS,
    )
    assert not flydsl_gemm.fused_splitk_ok(192, 256, 2, 2, 1, 1, 2)
    # The epilogue sizes its load batch per 128 threads: two waves must not fuse.
    assert not flydsl_gemm.fused_splitk_ok(32, 64, 1, 2, 1, 1, 2)
    m, n, k = 192, 256, 1024
    name = "flydsl_mxfp8_128_bpreshuffle_wmma_t192x256x128_mw2_nw2_nb2_sk2_cm1_cn1_fsk"
    xq, x_scale, weight, w_scale = _fused_splitk_operands(128, m, n, k)
    ref = run_torch(xq, x_scale, weight, w_scale, (128, 128), x_block=128)
    x_scale = x_scale.t().contiguous().view(*x_scale.shape)
    out = torch.full((m, n), float("nan"), dtype=dtypes.bf16)
    flydsl_gemm.run_gemm_a8w8_mxfp8_128_bpreshuffle_gfx1250(
        xq, shuffle_weight(weight, layout=(16, 16)), x_scale, w_scale, out, name
    )
    assertAllclose(ref, out.float(), tol_err_ratio=0, msg=f"{name}: ")


def tuned_shapes():
    table = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_MXFP8_BPRESHUFFLE_FILE)
    table = table[(table["gfx"] == get_gfx()) & (table["cu_num"] == get_cu_num())]
    shapes = list(table[["M", "N", "K"]].itertuples(index=False, name=None))
    if not shapes and get_gfx() == "gfx950":
        return DEFAULT_MNK_GFX950  # not tuned on this machine yet
    return shapes


def test_untuned_fallback_pick():
    """The pick for a shape without a tuned row is unfused, unsplit, at most 256 wide, of
    power-of-two geometry, fits the shape and underfills no compute tile."""
    from aiter.ops.flydsl.gemm_tune.flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common import (
        is_compute_kernel,
        is_pow2_geometry,
        kernel_fits_shape,
    )
    from aiter.ops.gemm_op_a8w8 import _flydsl_mxfp8_fallback_kernel as pick

    for m, n, k in (
        (1, 1536, 7168),
        (48, 6400, 7168),
        (100, 7168, 8192),
        (333, 4096, 1536),
        (4097, 7168, 4224),
    ):
        ki = pick(m, n, k)
        assert ki is not None and kernel_fits_shape(ki, m, n, k), (m, n, k)
        assert ki.split_k == 1 and not ki.fused_splitk and ki.tile_n <= 256, ki.name
        assert is_pow2_geometry(ki), ki.name
        assert not (is_compute_kernel(ki) and m < ki.tile_m), ki.name
        # mx32 needs 32-aligned tiles and plain A; an odd M has no A-preshuffle pairing.
        k32 = pick(m, n, k, mx32=True)
        assert k32 is None or (not k32.a_preshuffle and k32.tile_n % 32 == 0), (m, n, k)
        assert pick(m | 1, n, k, a_preshuffle=True) is None


K3_TABLE = "kimik3_a8w8_blockscale_mxscale_bpreshuffle_tuned_gemm.csv"


def test_tuner_catalog():
    """Fused split-K candidates fit by the row-stripe or the rank0 epilogue, rank0 only
    needs K % tile_k == 0, a cluster along N needs no second M tile, a partial N group
    fits only with n_tail, and the K3 table's kernelIds are the catalog's."""
    from aiter.ops.flydsl.gemm_tune import (
        flydsl_gemm_mxfp8_128_bpreshuffle_wmma_common as catalog,
    )

    kernel = {ki.name: ki for ki in catalog.kernels_list.values()}
    prefix = "flydsl_mxfp8_128_bpreshuffle_wmma_"
    ki = kernel[prefix + "t16x64x256_mw1_nw4_nb4_sk5_cm1_cn1_fsk"]
    fits = lambda k: catalog.kernel_fits_shape(ki, 16, 1536, k)
    # 28 K tiles split unevenly over 5; not a tile_k multiple; one K tile per split
    assert fits(7168) and not fits(7168 + 128) and not fits(5 * 256)
    n_cluster = kernel[prefix + "t16x64x128_mw1_nw4_nb4_sk6_cm1_cn2_fsk"]
    m_cluster = kernel[prefix + "t16x64x256_mw1_nw4_nb4_sk4_cm2_cn1_fsk"]
    assert catalog.kernel_fits_shape(n_cluster, 1, 7168, 4224)
    assert not catalog.kernel_fits_shape(m_cluster, 16, 7168, 4224)
    compute = prefix.replace("wmma", "compute_wmma")
    # M below the tile fits a compute kernel that clusters no M rows.
    under = kernel[compute + "t256x256x128_mw2_nw2_nb4_sk1_cm1_cn4"]
    assert catalog.kernel_fits_shape(under, 100, 12288, 7168)
    assert not catalog.kernel_fits_shape(
        kernel[compute + "t256x256x128_mw2_nw2_nb4_sk1_cm2_cn4"], 100, 12288, 7168
    )
    # A partial last group of N tiles fits a split_k=1 compute kernel for the tuner only.
    tail = kernel[compute + "t256x256x128_mw2_nw2_nb4_sk1_cm4_cn4"]
    assert catalog.kernel_fits_shape(tail, 4096, 49408, 7168, n_tail=True)
    assert not catalog.kernel_fits_shape(tail, 4096, 49408, 7168)
    assert not catalog.kernel_fits_shape(
        kernel[compute + "t256x256x128_mw2_nw2_nb4_sk2_cm2_cn2_fsk"],
        4096,
        49408,
        7168,
        n_tail=True,
    )
    # Clusters of 3 M tiles and 160-row tiles are tuner candidates the heuristics skip.
    for name in (
        compute + "t128x256x128_mw2_nw2_nb3_sk1_cm3_cn1",
        prefix + "t160x256x128_mw2_nw2_nb4_sk1_cm1_cn1",
    ):
        assert not catalog.is_pow2_geometry(kernel[name])
    table = Path(aiter.__file__).parent / "configs/model_configs" / K3_TABLE
    for row in pd.read_csv(table).query("gfx == 'gfx1250'").itertuples():
        assert catalog.kernels_list[row.kernelId].name == row.kernelName, row


if __name__ == "__main__":
    torch.set_default_device("cuda")
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=None,
        help="""M,N,K shapes; default: every tuned shape of this arch.
        e.g.: -mnk 512,7168,16384 1000,65536,1536""",
    )
    args = parser.parse_args()

    test_w_scale_layout()
    if get_gfx() not in ("gfx950", "gfx1250"):
        print(f"skip: gfx950 / gfx1250 only, got {get_gfx()}")
    else:
        shapes = args.mnk or tuned_shapes()
        if args.mnk is None and get_gfx() == "gfx950":
            shapes = shapes + RAGGED_K_MNK_GFX950
        df = pd.DataFrame([test_gemm(*mnk) for mnk in shapes])
        aiter.logger.info(f"gemm_a8w8_mxfp8_bpreshuffle summary:\n{df.to_markdown()}")
    if get_gfx() == "gfx1250":
        test_untuned_fallback_pick()
        test_tuner_catalog()
        test_fused_splitk_whole_row_passes()
        df = pd.DataFrame(
            [
                test_fused_splitk(block, name, m, n, k)
                for block, name, n, k in FUSED_SPLITK_CASES
                for m in FUSED_SPLITK_M
            ]
        )
        aiter.logger.info(f"fused split-K summary:\n{df.to_markdown(index=False)}")
        for name, n, k in MX128_DIRECT_CASES:
            for m in FUSED_SPLITK_M:
                test_mx128_kernel_direct(name, m, n, k)
        for name, n, k, ms in BOUNDED_CASES:
            for m in ms:
                test_mx128_kernel_direct(name, m, n, k)
