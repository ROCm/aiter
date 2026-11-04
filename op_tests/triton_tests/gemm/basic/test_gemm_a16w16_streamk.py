# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from aiter.ops.triton.gemm.basic.gemm_a16w16_streamk import (
    _SPLIT_FLOOR_M,
    _SPLIT_FLOOR_N,
    _SUPPORTED_ARCHS,
    _plan_k_pipeline,
    _plan_tiles,
    gemm_a16w16_streamk,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from op_tests.triton_tests.gemm.basic.test_gemm_a16w16 import (
    generate_gemm_a16w16_inputs,
)


def is_supported():
    return any(a in (get_arch() or "") for a in _SUPPORTED_ARCHS)


pytestmark = pytest.mark.skipif(
    not is_supported(), reason=f"gemm_a16w16_streamk needs one of {_SUPPORTED_ARCHS}"
)


def generate_inputs(M, N, K, dtype, layout="TN"):
    """x and w only, from the shared gemm_a16w16 generator.

    layout[1] == 'T' gives an N-major weight, which takes the native-B path. This kernel makes
    its own output, so the generator runs with output=False.
    """
    x, w, _, _, _ = generate_gemm_a16w16_inputs(
        M, N, K, dtype, layout=layout, output=False
    )
    return x, w


#: The CU count that the shipped DEFAULT.json buckets were tuned for. gfx1250 also has a 96cu
#: config tree, which selects other tiles and gives another worker count.
_TUNED_CU_NUM = 256


def skip_unless_tuned_cu_count():
    """The coverage guards pin fixed shapes to the buckets of the tuned CU count.

    The shapes stay correct on any CU count, and the correctness tests still run them there.
    Only the guards, which assert which path a shape takes, depend on the config tree.
    """
    if get_num_sms() != _TUNED_CU_NUM:
        pytest.skip(
            f"coverage shapes are picked for the {_TUNED_CU_NUM}-CU config tree, "
            f"this device reports {get_num_sms()}"
        )


def assert_matches_torch(out, x, w):
    """One tolerance for every correctness test, so they cannot drift apart."""
    expect = F.linear(x.float(), w.float(), bias=None).to(out.dtype)
    torch.testing.assert_close(out, expect, atol=1e-1, rtol=1e-2)


def plan_for(M, N, K):
    """The tile plan the wrapper will really use for this shape."""
    config, _ = get_gemm_config("GEMM-A16W16-STREAMK", M, N, K, backend="gluon")
    plan = _plan_tiles(
        M,
        N,
        config["BLOCK_M"],
        config["BLOCK_N"],
        config["CTAS_M"],
        config["CTAS_N"],
    )
    return config, plan


def get_x_vals():
    x_vals = [(1024 * v, 1024 * v, 1024 * v) for v in (1, 2, 4, 8)]
    x_vals += [(64, 4096, 4096), (128, 4096, 4096), (512, 4096, 4096)]
    x_vals += [(8192, 5120, 2880)]  # K % BLOCK_K == BLOCK_K / 2, takes the KTAIL drain
    x_vals += [(4096, 2880, 4096)]  # N not a multiple of the cluster tile
    x_vals += [(3000, 2048, 2048)]  # M not a multiple of the cluster tile
    x_vals += [(3000, 2880, 2048)]  # M and N both ragged
    x_vals += get_small_k_x_vals()
    x_vals += get_k_tail_x_vals()
    x_vals += get_narrow_n_x_vals()
    x_vals += get_remainder_x_vals()
    return x_vals


def get_k_tail_x_vals():
    """A K that does not divide BLOCK_K, on the fused single-CTA path.

    One CTA has no multicast, so it issues a fused load and moves the descriptor itself. That
    update must keep the descriptor bounds, or the last K tile reads past K instead of
    zero-filling. Every one of these shapes was wrong before `clamp_bounds=True`.

    The suite could not see it: the only single-CTA shapes had a K that divides BLOCK_K, or one
    K tile, where the descriptor is still at its own bounds.
    """
    return [
        (64, 64, 288),  # one tail step after a whole step
        (64, 64, 320),
        (64, 64, 1056),  # several whole steps, then a tail
        (64, 4096, 4128),  # a wide N with a tail
        (32, 64, 288),  # M below the tile
    ]


def get_small_k_x_vals():
    """Shapes with one K tile only, where the steady loop of the pipeline is empty.

    NUM_BUFFERS has a floor of 2, so `cdiv(K, BLOCK_K) - (NUM_BUFFERS - 1)` is 0 here. The
    kernel must not then state that the trip count is above 0. It did, and the UB miscompiled
    the kernel: 64x64x256 returned NaN and 64x64x32 returned garbage.
    """
    return [
        (64, 64, 32),  # a part K tile, below one wmma K step
        (64, 64, 128),  # a half K tile, which also takes the KTAIL drain
        (64, 64, 256),  # exactly one K tile
        (256, 256, 64),
        (1024, 1024, 128),
    ]


def get_narrow_n_x_vals():
    """Shapes whose N is narrower than the bucket tile, so there is one tile column.

    The `any` bucket uses a 1024 wide tile, so an N of 1024 or less gives a single column and
    the TDM bounds masking carries the rest. These are slow (see Limits in the PR), but they
    must stay correct.
    """
    return [
        (3072, 1024, 2048),  # N equals the tile width
        (3072, 512, 2048),  # N is half the tile width
    ]


def get_remainder_x_vals():
    """Shapes that leave tiles over, so the StreamK remainder phase runs.

    The remainder cuts each leftover tile into SPLIT_M x SPLIT_N sub-tiles and gives one
    sub-tile to each cluster. Each shape here selects a different split, cluster and tile,
    because the sub-tile size decides the layouts that the phase compiles.
    """
    return [
        (512, 6144, 2048),  # split 2x1, cluster 2x2, tile 128x256
        (256, 10240, 2048),  # split 2x2, cluster 2x2, tile 128x256
        (1024, 1536, 2048),  # split 2x1, cluster 4x4, tile 256x256
        (1536, 4096, 2048),  # split 2x1, cluster 4x4, tile 512x512
        (2048, 2560, 2048),  # split 4x1, cluster 4x4, tile 512x512
        (1536, 2880, 2048),  # split 4x2, cluster 4x4, tile 512x512, both axes
        (3072, 7168, 2048),  # split 2x1, cluster 4x4, tile 1024x1024
        (4096, 5120, 2048),  # split 4x1, cluster 4x4, tile 1024x1024
        (3072, 6144, 2048),  # split 8x1, cluster 4x4, tile 1024x1024
    ]


@pytest.mark.parametrize("M, N, K", get_x_vals())
@pytest.mark.parametrize("layout", ["TN", "TT"])
def test_gemm_a16w16_streamk(M: int, N: int, K: int, layout):
    x, w = generate_inputs(M, N, K, torch.bfloat16, layout=layout)

    assert_matches_torch(gemm_a16w16_streamk(x, w), x, w)


@pytest.mark.parametrize("M, N, K", [(2048, 2048, 2048), (3000, 2880, 2048)])
@pytest.mark.parametrize("in_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("out_dtype", [None, torch.float32])
@pytest.mark.parametrize("layout", ["TN", "TT"])
def test_gemm_a16w16_streamk_dtypes(M, N, K, in_dtype, out_dtype, layout):
    """A 2-byte C must accumulate in fp32. An fp16 accumulator does not compile.

    out_dtype None keeps the input dtype, which also turns on the aliased C store. fp32 is
    4 bytes, so it takes the masked store instead.
    """
    x, w = generate_inputs(M, N, K, in_dtype, layout=layout)

    out = gemm_a16w16_streamk(x, w, dtype=out_dtype)

    assert out.dtype == (out_dtype or in_dtype)
    assert_matches_torch(out, x, w)


@pytest.mark.parametrize("M, N, K", [(2048, 2048, 2048)])
def test_gemm_a16w16_streamk_out_tensor(M: int, N: int, K: int):
    """A caller-supplied y must be written in place and returned."""
    x, w = generate_inputs(M, N, K, torch.bfloat16)
    y = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")

    out = gemm_a16w16_streamk(x, w, y=y)

    assert out.data_ptr() == y.data_ptr()
    assert_matches_torch(y, x, w)


def test_gemm_a16w16_streamk_rejects_bad_inputs():
    """The wrapper must refuse what the kernel cannot do, and refuse it before the launch.

    A y that is too small is the important one: the kernel writes a whole (M, N) tile grid, so
    it goes past the end of the allocation instead of failing.
    """
    M = N = K = 512
    x, w = generate_inputs(M, N, K, torch.bfloat16)

    for bad_y in (
        torch.empty((1, N), dtype=torch.bfloat16, device="cuda"),
        torch.empty((M, 1), dtype=torch.bfloat16, device="cuda"),
        torch.empty((M, N), dtype=torch.float32, device="cuda"),
        torch.empty((N, M), dtype=torch.bfloat16, device="cuda").T,
    ):
        with pytest.raises(AssertionError):
            gemm_a16w16_streamk(x, w, y=bad_y)

    with pytest.raises(AssertionError):
        gemm_a16w16_streamk(x.float(), w.float())
    with pytest.raises(AssertionError):
        gemm_a16w16_streamk(x, w, dtype=torch.float64)
    with pytest.raises(AssertionError):
        gemm_a16w16_streamk(x, w.half())


@pytest.mark.parametrize("M, N, K", get_k_tail_x_vals())
def test_gemm_a16w16_streamk_k_tail_uses_the_fused_path(M: int, N: int, K: int):
    """Each K-tail shape must really take the fused single-CTA path, and really have a tail.

    The fused path runs only at a cluster of one CTA. If a config change gives these shapes a
    larger cluster, they stop covering the descriptor-bounds bug and nothing else fails.
    """
    skip_unless_tuned_cu_count()
    config, _ = plan_for(M, N, K)

    assert (
        config["CTAS_M"] * config["CTAS_N"] == 1
    ), "not the single-CTA path, so the fused load is not covered"
    assert K % config["BLOCK_K"], f"K={K} divides BLOCK_K, so there is no tail to clamp"
    assert (
        K > config["BLOCK_K"]
    ), f"K={K} is one tile, where the bounds are still its own"


@pytest.mark.parametrize("M, N, K", get_small_k_x_vals())
def test_gemm_a16w16_streamk_small_k_empties_the_steady_loop(M: int, N: int, K: int):
    """Each small-K shape must really leave the steady loop empty.

    These shapes exist to cover the empty steady loop, where the kernel must not state that the
    trip count is above 0. Their result is correct on any path, so without this assert a config
    change can take them off that path and the coverage goes away unnoticed.
    """
    skip_unless_tuned_cu_count()
    config, _ = plan_for(M, N, K)
    _, _, steady_nonempty = _plan_k_pipeline(
        K, config["BLOCK_K"], config["NUM_BUFFERS"]
    )

    assert (
        not steady_nonempty
    ), f"K={K} still fills the steady loop, so it covers nothing"


@pytest.mark.parametrize("M, N, K", get_remainder_x_vals())
def test_gemm_a16w16_streamk_remainder_runs(M: int, N: int, K: int):
    """Each remainder shape must really reach phase 2, and with a split above 1x1.

    The result of a remainder shape is correct even when phase 2 does not run, because phase 1
    computes every whole tile. A config change can therefore take the remainder out of the test
    coverage and no other test fails. This test holds the coverage.
    """
    skip_unless_tuned_cu_count()
    config, (num_tiles, num_sms, streamk_tiles, split_m, split_n) = plan_for(M, N, K)

    assert (
        streamk_tiles > 0
    ), f"{num_tiles} tiles on {num_sms} workers leave nothing over"
    assert split_m * split_n > 1, "a 1x1 split keeps the tile on the phase-1 path"
    # The sub-tiles must fit one wave; two half-size waves are slower than one partial wave.
    assert streamk_tiles * split_m * split_n <= num_sms
    # The N floor is a measured cliff, not a layout limit: a per-CTA N of 32 makes a bf16 row
    # of 64 bytes, and that measured -11.2% against 64 at an equal area and sub-tile count.
    per_cta_m = config["BLOCK_M"] // split_m // config["CTAS_M"]
    per_cta_n = config["BLOCK_N"] // split_n // config["CTAS_N"]
    assert per_cta_n >= _SPLIT_FLOOR_N, f"sub-tile too narrow on N: {per_cta_n}"
    assert per_cta_m >= _SPLIT_FLOOR_M, f"sub-tile too short on M: {per_cta_m}"


@pytest.mark.parametrize(
    "per_cta, ctas, expect_a_interval, expect_b_interval",
    [
        ((256, 256, 128), (4, 4), 128, 256),
        ((128, 128, 128), (2, 2), 128, 128),
        ((64, 256, 128), (1, 1), 128, 256),
    ],
)
def test_layout_builders(per_cta, ctas, expect_a_interval, expect_b_interval):
    """The shared layouts must pad on the per-CTA innermost extent, under the TDM cap of 256.

    An over-cap interval makes the TDM descriptor fail to encode, and a CGA-wide interval breaks
    the C-stage reinterpret, so pin both here rather than wait for a compile error.
    """
    from aiter.ops.triton._gluon_kernels.gfx1250.layouts import (
        TDM_PAD_INTERVAL_MAX,
        operand_bundle_cga,
        wmma_layout,
    )

    BM, BN, BK = per_cta
    wmma = wmma_layout((BM, BN), ctas, 4)
    _, shared_a = operand_bundle_cga(wmma, 0, (BM, BK), ctas)
    _, shared_b = operand_bundle_cga(wmma, 1, (BK, BN), ctas)

    a_interval = shared_a.interval_padding_pairs[0][0]
    b_interval = shared_b.interval_padding_pairs[0][0]
    assert a_interval == expect_a_interval
    assert b_interval == expect_b_interval
    assert b_interval <= TDM_PAD_INTERVAL_MAX

    # A splits over M only and B over N only, so each gets zero bases on the other axis.
    assert all(row[1] == 0 for row in shared_a.cga_layout)
    assert all(row[0] == 0 for row in shared_b.cga_layout)

    # with_identity_for needs the cluster-wide shape, not the per-CTA tile.
    assert shared_a.shape == [BM * ctas[0], BK]
    assert shared_b.shape == [BK, BN * ctas[1]]
