# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import copy
import sys

import pytest

# Support direct execution with the same parametrization as the pytest runner.
if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, *sys.argv[1:]]))

import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops import gemm_op_a8w8
from aiter.ops.gemm_op_a8w8 import gemm_a8w8_blockscale
from aiter.ops.shuffle import shuffle_mxfp8fp4_scale, shuffle_weight
from aiter.ops.triton.gemm.basic import gemm_afp8wfp8 as afp8wfp8_op
from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale_group32 import (
    gemm_a8w8_blockscale_group32,
)
from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import (
    _is_gluon_available,
    gemm_afp8wfp8,
    gemm_afp8wfp8_block128_preshuffle,
    gemm_afp8wfp8_preshuffle,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton.utils.shuffle import shuffle_scale_gemm_128row
from aiter.test_common import checkAllclose

SCALE_GROUP_SIZE = 32  # A: 1x32 e8m0 scale group
W_SCALE_K_GROUP = 128  # B: 128 in K direction
W_SCALE_N_GROUP = 128  # B: 128 in N direction
FP8_MAX = 448.0  # e4m3 max

# Activations here are signed (randn), so a K-long fp8 dot product produces a
# few heavily cancelled outputs whose magnitude is orders below the partial
# sums. On those, a per-element absolute tolerance is meaningless -- the kernel
# and the fp32 reference sit within one bf16 ULP of an fp64 reference, but their
# fp32 accumulators differ by more than any sane atol. checkAllclose's outlier
# ratio is the right criterion (and the aiter house default); catastrophic_check
# still fails hard on NaN/Inf or delta > ref_max/2, so real bugs are caught.
TOL_ERR_RATIO = 0.05


def e8m0_to_f32(x: torch.Tensor) -> torch.Tensor:
    """Decode unsigned-biased e8m0 (uint8) to fp32. Bias 127, value = 2^(b-127)."""
    return torch.exp2((x.to(torch.int32) - 127).to(torch.float32))


def generate_inputs(
    M: int,
    N: int,
    K: int,
    shuffle: bool = False,
    x_scale_group_size: int = SCALE_GROUP_SIZE,
    transpose_x_scale: bool = False,
):
    """Returns ``(x_fp8, w_fp8, w_kernel, x_scales, x_scales_kernel, w_scales)``.

    ``w_fp8`` is always the unshuffled weight (for use by the fp32 reference).
    ``w_kernel`` is the weight to pass to the kernel: identical to ``w_fp8``
    when ``shuffle=False``, or shuffled via ``shuffle_weight(layout=(16, 16))``
    when ``shuffle=True``.

    ``x_scales`` is the logical ``(M, K // x_scale_group_size)`` scale the
    reference dequants with. ``x_scales_kernel`` is what the kernel is handed:
    the same tensor normally, or a byte-transposed buffer when
    ``transpose_x_scale=True`` — i.e. ``.shape`` still reads ``(M, Kg)`` but the
    storage is column-major, which is what
    ``per_group_quant_hip(transpose_scale=True)`` produces.
    """
    # Small random fp32 → fp8 e4m3fn, kept inside e4m3 range so the cast is exact-ish.
    x_f32 = torch.randn((M, K), dtype=torch.float32, device="cuda")
    w_f32 = torch.randn((N, K), dtype=torch.float32, device="cuda")
    x_f32 = torch.clamp(x_f32, -FP8_MAX, FP8_MAX)
    w_f32 = torch.clamp(w_f32, -FP8_MAX, FP8_MAX)
    x_fp8 = x_f32.to(torch.float8_e4m3fn)
    w_fp8 = w_f32.to(torch.float8_e4m3fn)

    # e8m0 scales near 127 (== 1.0) so the dequant has unit-ish magnitude.
    x_scales = torch.randint(
        125, 130, (M, K // x_scale_group_size), dtype=torch.uint8, device="cuda"
    )
    w_scales = torch.randint(
        125,
        130,
        (N // W_SCALE_N_GROUP, K // W_SCALE_K_GROUP),
        dtype=torch.uint8,
        device="cuda",
    )

    if transpose_x_scale:
        # Same bytes, laid out (Kg, M) row-major, then reinterpreted as (M, Kg).
        # The wrapper recovers the real strides from is_x_scale_transposed=True.
        x_scales_kernel = x_scales.T.contiguous().reshape(M, K // x_scale_group_size)
    else:
        x_scales_kernel = x_scales

    if shuffle:
        # shuffle_weight operates on raw bytes; view as uint8 to avoid dtype quirks.
        w_kernel = shuffle_weight(w_fp8.view(torch.uint8), layout=(16, 16))
    else:
        w_kernel = w_fp8

    return x_fp8, w_fp8, w_kernel, x_scales, x_scales_kernel, w_scales


def run_torch_gemm_afp8wfp8(
    x_fp8: torch.Tensor,
    w_fp8: torch.Tensor,
    x_scales: torch.Tensor,
    w_scales: torch.Tensor,
    out_dtype: torch.dtype,
    x_scale_group_size: int = SCALE_GROUP_SIZE,
) -> torch.Tensor:
    """Reference: dequant both operands to fp32 and run torch.mm."""
    M, K = x_fp8.shape
    N, _ = w_fp8.shape

    x_view = x_fp8 if x_fp8.dtype != torch.uint8 else x_fp8.view(torch.float8_e4m3fn)
    w_view = w_fp8 if w_fp8.dtype != torch.uint8 else w_fp8.view(torch.float8_e4m3fn)
    x_f32 = x_view.to(torch.float32)
    w_f32 = w_view.to(torch.float32)

    x_s_f32 = e8m0_to_f32(x_scales).repeat_interleave(x_scale_group_size, dim=1)
    assert x_s_f32.shape == (M, K)

    w_s_f32 = e8m0_to_f32(w_scales)
    w_s_f32 = w_s_f32.repeat_interleave(W_SCALE_N_GROUP, dim=0).repeat_interleave(
        W_SCALE_K_GROUP, dim=1
    )
    assert w_s_f32.shape == (N, K)

    x_dq = x_f32 * x_s_f32
    w_dq = w_f32 * w_s_f32
    return torch.mm(x_dq, w_dq.T).to(out_dtype)


# (x_scale_group_size, transpose_x_scale). 128/True is what ATOM's per_1x128
# quant emits; 32/False is MX activations.
SCALE_MODES = [(128, False), (128, True), (32, False), (32, True)]


def get_shapes():
    # (M, N, K), with N % 128 == 0 and K % 128 == 0 to fit the 128x128 W-scale layout.
    return [
        (m, n, k)
        for m in [16, 32, 64, 512]
        for n, k in [
            (2048, 7168),
            (65536, 1536),
            (16384, 1536),
            (8192, 1536),
            (7168, 16384),
            (7168, 4096),
            (6144, 7168),
            (7168, 3072),
            (1536, 7168),
            (7168, 768),
        ]
    ]


@pytest.mark.parametrize("M, N, K", get_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("x_scale_group_size, transpose_x_scale", SCALE_MODES)
def test_gemm_afp8wfp8(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    x_scale_group_size: int,
    transpose_x_scale: bool,
):
    torch.manual_seed(0)
    if not arch_info.is_fp8_avail():
        pytest.skip("MXFP8 GEMM requires FP8-capable arch")
    torch.cuda.empty_cache()

    x_fp8, w_fp8, w_kernel, x_scales, x_scales_kernel, w_scales = generate_inputs(
        M,
        N,
        K,
        shuffle=False,
        x_scale_group_size=x_scale_group_size,
        transpose_x_scale=transpose_x_scale,
    )

    torch_out = run_torch_gemm_afp8wfp8(
        x_fp8, w_fp8, x_scales, w_scales, dtype, x_scale_group_size
    )
    triton_out = gemm_afp8wfp8(
        x_fp8,
        w_kernel,
        x_scales_kernel,
        w_scales,
        dtype=dtype,
        x_scale_group_size=x_scale_group_size,
        is_x_scale_transposed=transpose_x_scale,
    )

    err = checkAllclose(
        torch_out,
        triton_out,
        atol=1e-2,
        rtol=1e-2,
        msg=f"afp8wfp8 M={M} N={N} K={K} gs={x_scale_group_size} T={transpose_x_scale} ",
        catastrophic_check=True,
    )
    assert err < TOL_ERR_RATIO, f"{err:.2%} of elements mismatch"


def _gluon_available() -> bool:
    """The gluon MXFP8 block128 preshuffle kernel is gfx1250-only."""
    try:
        return "gfx1250" in arch_info.get_arch()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.parametrize("M, N, K", get_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("x_scale_group_size, transpose_x_scale", SCALE_MODES)
def test_gemm_afp8wfp8_block128_preshuffle(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    backend: str,
    x_scale_group_size: int,
    transpose_x_scale: bool,
):
    torch.manual_seed(0)
    if not arch_info.is_fp8_avail():
        pytest.skip("MXFP8 GEMM requires FP8-capable arch")
    if N % 16 != 0 or K % 32 != 0:
        pytest.skip("Preshuffle requires N % 16 == 0 and K % 32 == 0")
    if backend == "gluon":
        if not _gluon_available():
            pytest.skip("Gluon MXFP8 block128 preshuffle requires gfx1250")
        # The gluon kernel tiles K with the [16, 16, 128] wmma shape.
        if K % 128 != 0 or N % 128 != 0:
            pytest.skip("Gluon kernel requires N % 128 == 0 and K % 128 == 0")
    torch.cuda.empty_cache()

    x_fp8, w_fp8, w_kernel, x_scales, x_scales_kernel, w_scales = generate_inputs(
        M,
        N,
        K,
        shuffle=True,
        x_scale_group_size=x_scale_group_size,
        transpose_x_scale=transpose_x_scale,
    )

    torch_out = run_torch_gemm_afp8wfp8(
        x_fp8, w_fp8, x_scales, w_scales, dtype, x_scale_group_size
    )
    out = gemm_afp8wfp8_block128_preshuffle(
        x_fp8,
        w_kernel,
        x_scales_kernel,
        w_scales,
        dtype=dtype,
        x_scale_group_size=x_scale_group_size,
        is_x_scale_transposed=transpose_x_scale,
        backend=backend,
    )

    err = checkAllclose(
        torch_out,
        out,
        atol=1e-2,
        rtol=1e-2,
        msg=(
            f"afp8wfp8 block128 preshuffle [{backend}] M={M} N={N} K={K} "
            f"gs={x_scale_group_size} T={transpose_x_scale} "
        ),
        catastrophic_check=True,
    )
    assert err < TOL_ERR_RATIO, f"{err:.2%} of elements mismatch"


@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("ctas_m, ctas_n", [(2, 1), (1, 2), (2, 2), (4, 1), (1, 4)])
def test_gluon_block128_preshuffle_cga_multicast(
    ctas_m: int, ctas_n: int, ragged: bool
):
    """CTA-cluster (CGA) operand multicast reproduces the single-CTA result.

    BLOCK_SIZE_M / BLOCK_SIZE_N become the cluster tile, so a CTAS_M x CTAS_N
    cluster covers the same output tile with the same per-CTA footprint and each
    operand fetch multicast to the CTAs that share it. The result must be
    bit-identical to CTAS 1x1: multicast changes who reads a byte, never which
    byte or the order it is accumulated in.
    """
    if not _gluon_available():
        pytest.skip("Gluon MXFP8 block128 preshuffle CGA multicast requires gfx1250")

    from triton._C.libtriton.gluon_ir import make_cga_layout

    from aiter.ops.triton._gluon_kernels.gfx1250.gemm.basic.gemm_mxfp8 import (
        cga_bases,
    )

    # The kernel cannot call make_cga_layout (nanobind, rejected by gluon's
    # dependency walker), so it reimplements it -- pin the two together.
    assert cga_bases(ctas_m, ctas_n) == make_cga_layout(
        [ctas_m, ctas_n], [ctas_m, ctas_n], [0, 1]
    )

    torch.manual_seed(0)
    x_scale_group_size, transpose_x_scale = 128, True
    config, _ = get_gemm_config(
        "GEMM-AFP8WFP8_BLOCK128_PRESHUFFLED", 1024, 1024, 1536, backend="gluon"
    )
    # Enough tiles that the cluster grid has more than one entry in both dims.
    M = config["BLOCK_SIZE_M"] * ctas_m * 2
    N = config["BLOCK_SIZE_N"] * ctas_n * 2
    K = 1536
    if ragged:
        # A cluster tile that does not divide M: the tail is handled purely by
        # the TDM descriptor bounds (zero-fill on load, clip on store), same as
        # the single-CTA path -- but the cluster tile is CTAS_M times larger, so
        # a CGA turns shapes that used to be exact into ragged ones. N stays
        # aligned because the preshuffled B descriptor is indexed in N//16.
        M -= config["BLOCK_SIZE_M"] // 2

    x_fp8, _, w_kernel, _, x_scales_kernel, w_scales = generate_inputs(
        M,
        N,
        K,
        shuffle=True,
        x_scale_group_size=x_scale_group_size,
        transpose_x_scale=transpose_x_scale,
    )

    def run(cm, cn):
        cfg = copy.deepcopy(config)
        cfg["CTAS_M"], cfg["CTAS_N"] = cm, cn
        return gemm_afp8wfp8_block128_preshuffle(
            x_fp8,
            w_kernel,
            x_scales_kernel,
            w_scales,
            dtype=torch.bfloat16,
            config=cfg,
            x_scale_group_size=x_scale_group_size,
            is_x_scale_transposed=transpose_x_scale,
            backend="gluon",
        )

    single = run(1, 1)
    clustered = run(ctas_m, ctas_n)
    assert torch.equal(single, clustered), (
        f"CTAS {ctas_m}x{ctas_n} differs from 1x1: max abs delta "
        f"{(single.float() - clustered.float()).abs().max().item()}"
    )


@pytest.mark.parametrize("M, N, K", get_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("x_scale_group_size, transpose_x_scale", SCALE_MODES)
def test_gemm_afp8wfp8_block128_preshuffle_gluon_matches_triton(
    M: int,
    N: int,
    K: int,
    dtype: torch.dtype,
    x_scale_group_size: int,
    transpose_x_scale: bool,
):
    """Both backends consume the same shuffled weight and compact 128x128 scales,
    so they must agree far more tightly than either agrees with the fp32 reference."""
    torch.manual_seed(0)
    if not arch_info.is_fp8_avail():
        pytest.skip("MXFP8 GEMM requires FP8-capable arch")
    if not _gluon_available():
        pytest.skip("Gluon MXFP8 block128 preshuffle requires gfx1250")
    if K % 128 != 0 or N % 128 != 0:
        pytest.skip("Gluon kernel requires N % 128 == 0 and K % 128 == 0")
    torch.cuda.empty_cache()

    x_fp8, _, w_kernel, _, x_scales_kernel, w_scales = generate_inputs(
        M,
        N,
        K,
        shuffle=True,
        x_scale_group_size=x_scale_group_size,
        transpose_x_scale=transpose_x_scale,
    )

    kwargs = {
        "dtype": dtype,
        "x_scale_group_size": x_scale_group_size,
        "is_x_scale_transposed": transpose_x_scale,
    }
    triton_out = gemm_afp8wfp8_block128_preshuffle(
        x_fp8, w_kernel, x_scales_kernel, w_scales, backend="triton", **kwargs
    )
    gluon_out = gemm_afp8wfp8_block128_preshuffle(
        x_fp8, w_kernel, x_scales_kernel, w_scales, backend="gluon", **kwargs
    )

    # Deliberately strict, unlike the reference comparisons above: both backends
    # consume identical inputs and (today) agree bitwise, so there is no
    # cancellation tail to tolerate here and any drift is a real divergence.
    torch.testing.assert_close(gluon_out, triton_out, atol=1e-2, rtol=1e-2)


@pytest.fixture
def _require_gfx950():
    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        pytest.skip("CDNA4 scaled MFMA requires gfx950")


def _execution_config(packed=False, fused=False, splits=3, n_first=False):
    config = {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 64,
        "BLOCK_SIZE_K": 128,
        "GROUP_SIZE_M": 1,
        "REDUCE_BLOCK_SIZE_M": 32,
        "REDUCE_BLOCK_SIZE_N": 32,
        "NUM_KSPLIT": splits,
        "N_FIRST": n_first,
        "FUSED_SPLITK": fused,
        "num_warps": 4,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
        "cache_modifier": ".cg",
    }
    if packed:
        config["packed"] = dict(config, BLOCK_SIZE_M=4, K_PACK=4)
    return config


def _compact_scale_inputs(m, n, k, a_group, b_group):
    torch.manual_seed(m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    w = torch.randn(n, k, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    xs = torch.randint(124, 131, (m, k // a_group), device="cuda", dtype=torch.uint8)
    ws = torch.randint(
        124,
        131,
        ((n + b_group[0] - 1) // b_group[0], k // b_group[1]),
        device="cuda",
        dtype=torch.uint8,
    )
    return x, w, xs, ws, _compact_scale_reference(x, w, xs, ws, a_group, b_group)


def _compact_scale_reference(x, w, xs, ws, a_group, b_group):
    n = w.shape[0]
    a = x.double() * torch.exp2(xs.double() - 127).repeat_interleave(a_group, -1)
    b = w.double() * torch.exp2(ws.double() - 127).repeat_interleave(b_group[0], 0)[
        :n
    ].repeat_interleave(b_group[1], -1)
    return a @ b.T


def _assert_compact_scale_close(actual, expected):
    torch.testing.assert_close(
        actual.float(),
        expected.to(actual.dtype).float(),
        rtol=0.016 if actual.dtype == torch.bfloat16 else 3e-5,
        atol=5e-5 * expected.abs().max().item(),
    )


@pytest.mark.parametrize("a_group", [32, 128])
@pytest.mark.parametrize("b_group", [(1, 32), (32, 32), (128, 128)])
@pytest.mark.parametrize(
    "packed,fused", [(False, False), (False, True), (True, False), (True, True)]
)
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.usefixtures("_require_gfx950")
def test_compact_scale_execution_variants(a_group, b_group, packed, fused, transposed):
    # Tail in M and N, and K=1152 has a tail in the packed 512-element step.
    x, w, xs, ws, expected = _compact_scale_inputs(
        7 if packed else 19, 131, 1152, a_group, b_group
    )
    if transposed:
        xs = xs.T.contiguous().reshape(xs.shape)
        # Independently exercise ordinary tensor strides on the weight scales.
        ws = ws.T.contiguous().T
    config = _execution_config(packed, fused, n_first=transposed)
    original = copy.deepcopy(config)
    y = torch.empty(expected.shape, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):  # Fused arrival counters must be reusable.
        actual = gemm_afp8wfp8(
            x,
            w,
            xs,
            ws,
            dtype=torch.bfloat16,
            y=y,
            config=config,
            x_scale_group_size=a_group,
            w_scale_group_size=b_group,
            is_x_scale_transposed=transposed,
        )
        assert actual is y
        _assert_compact_scale_close(actual, expected)
    assert config == original


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("b_group", [(32, 32), (128, 128)])
@pytest.mark.parametrize("splits", [3, 64])
@pytest.mark.usefixtures("_require_gfx950")
def test_split_k_skip_reduce_and_override(packed, b_group, splits):
    x, w, xs, ws, expected = _compact_scale_inputs(3, 67, 640, 128, b_group)
    config = _execution_config(packed, fused=True)
    original = copy.deepcopy(config)
    partials = gemm_afp8wfp8(
        x,
        w,
        xs,
        ws,
        dtype=torch.bfloat16,
        config=config,
        x_scale_group_size=128,
        w_scale_group_size=b_group,
        split_k=splits,
        skip_reduce=True,
    )
    assert partials.ndim == 3
    assert partials.dtype == torch.float32
    assert tuple(partials.shape[1:]) == tuple(expected.shape)
    assert 1 <= partials.shape[0] <= splits
    _assert_compact_scale_close(partials.sum(0), expected)
    actual = gemm_afp8wfp8(
        x,
        w,
        xs,
        ws,
        dtype=torch.bfloat16,
        config=config,
        x_scale_group_size=128,
        w_scale_group_size=b_group,
        split_k=splits,
    )
    _assert_compact_scale_close(actual, expected)
    assert config == original


@pytest.mark.parametrize("rows", [1, 32])
@pytest.mark.usefixtures("_require_gfx950")
def test_compact_scales_with_uint8_operands(rows):
    x, w, xs, ws, expected = _compact_scale_inputs(3, 131, 1152, 32, (rows, 32))
    config = _execution_config()
    actual = gemm_afp8wfp8(
        x.view(torch.uint8),
        w.view(torch.uint8),
        xs,
        ws,
        dtype=torch.bfloat16,
        config=config,
        x_scale_group_size=32,
        w_scale_group_size=(rows, 32),
        split_k=3,
    )
    _assert_compact_scale_close(actual, expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.usefixtures("_require_gfx950")
def test_fused_split_k_graph_replay_reads_live_scales(packed):
    a_group, b_group = 128, (128, 128)
    x, w, xs, ws, _ = _compact_scale_inputs(3, 131, 1152, a_group, b_group)
    config = _execution_config(packed=packed, fused=True)
    kwargs = {
        "dtype": torch.bfloat16,
        "config": config,
        "x_scale_group_size": a_group,
        "w_scale_group_size": b_group,
    }
    # Initialize per-stream counters before capture, then ensure each replay
    # consumes current operands/scales and leaves the counters reusable.
    gemm_afp8wfp8(x, w, xs, ws, **kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gemm_afp8wfp8(x, w, xs, ws, **kwargs)
    for factor in (2.0, 0.5):
        x.copy_((x.float() * factor).to(x.dtype))
        xs.add_(1)
        ws.add_(1)
        graph.replay()
        _assert_compact_scale_close(
            actual, _compact_scale_reference(x, w, xs, ws, a_group, b_group)
        )


@pytest.mark.parametrize("a_group", [32, 128])
@pytest.mark.parametrize("b_group", [(1, 32), (32, 32), (128, 128)])
@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.usefixtures("_require_gfx950")
def test_compact_scale_partial_k_tile(a_group, b_group, fused):
    # K is divisible by every scale group, but not by the 256-element tile.
    # This must compile and exercise masked E4M3 loads (EVEN_K=False).
    x, w, xs, ws, expected = _compact_scale_inputs(19, 131, 640, a_group, b_group)
    config = _execution_config(fused=fused, splits=3)
    config["BLOCK_SIZE_K"] = 256
    actual = gemm_afp8wfp8(
        x,
        w,
        xs,
        ws,
        dtype=torch.bfloat16,
        config=config,
        x_scale_group_size=a_group,
        w_scale_group_size=b_group,
    )
    _assert_compact_scale_close(actual, expected)


def generate_group32_inputs(m, n, k, weight_group_rows=32):
    torch.manual_seed(m + n + k)
    x = torch.randn(m, k, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    weight = torch.randn(n, k, device="cuda", dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    xs = torch.randint(122, 131, (m, k // 32), device="cuda", dtype=torch.uint8)
    ws = torch.randint(
        122,
        131,
        (-(-n // weight_group_rows), k // 32),
        device="cuda",
        dtype=torch.uint8,
    )
    return x, weight, xs.view(torch.float8_e8m0fnu), ws.view(torch.float8_e8m0fnu)


def run_group32_reference(x, weight, xs, ws, weight_group_rows=32):
    # FP64 is independent of the MFMA's internal block accumulation and of
    # the kernel's K reduction order.
    a = x.double() * xs.double().repeat_interleave(32, -1)
    b = weight.double() * ws.double().repeat_interleave(weight_group_rows, 0)[
        : weight.shape[0]
    ].repeat_interleave(32, -1)
    return a @ b.T


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 5120, 576),
        (3, 2053, 1280),
        (4, 8192, 1280),
        (8, 5120, 1152),
        (16, 4096, 1280),
        (31, 4096, 1280),
        (3, 5120, 2048),
        (8, 5120, 2304),
        (4, 5120, 4096),
        (3, 2304, 5120),
        (1, 5120, 8192),
        # Untuned geometries exercise the default configuration with tails.
        (63, 8193, 1280),
        (129, 4097, 576),
        (255, 16385, 1152),
        (1023, 4097, 1280),
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.usefixtures("_require_gfx950")
def test_group32_projection_panel_scales_and_tails(m, n, k, dtype):
    x, weight, xs, ws = generate_group32_inputs(m, n, k)
    actual = gemm_a8w8_blockscale_group32(x, weight, xs, ws, dtype=dtype)
    expected = run_group32_reference(x, weight, xs, ws)
    peak = expected.abs().max().item()
    # Compare rounded BF16 outputs; near cancellation uses a peak-relative floor.
    torch.testing.assert_close(
        actual.float(),
        expected.to(dtype).float(),
        rtol=0.016,
        atol=5e-5 * peak,
    )


@pytest.mark.parametrize("m,packed", [(3, True), (129, False)])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.usefixtures("_require_gfx950")
def test_tuned_variants_use_production_config(m, packed, dtype):
    n, k = 8192, 1280
    config, tuned = get_gemm_config("GEMM-AFP8WFP8_A32_W32X32", m, n, k)
    assert tuned
    assert ("packed" in config) == packed
    if not packed:
        assert config["N_FIRST"]
    x, w, xs, ws = generate_group32_inputs(m, n, k)
    actual = gemm_a8w8_blockscale_group32(x, w, xs, ws, dtype=dtype)
    expected = run_group32_reference(x, w, xs, ws)
    torch.testing.assert_close(
        actual.float(),
        expected.to(dtype).float(),
        rtol=0.016,
        atol=5e-5 * expected.abs().max().item(),
    )


@pytest.mark.parametrize(
    "a_code,b_code", [(0, 254), (254, 0), (128, 0), (255, 127), (127, 255)]
)
@pytest.mark.usefixtures("_require_gfx950")
def test_group32_projection_extreme_scale_codes(a_code, b_code):
    x = torch.ones(3, 1280, device="cuda", dtype=torch.float32).to(torch.float8_e4m3fn)
    weight = torch.ones(2048, 1280, device="cuda", dtype=torch.float32).to(
        torch.float8_e4m3fn
    )
    xs = torch.full((3, 40), a_code, device="cuda", dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    ws = torch.full((64, 40), b_code, device="cuda", dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    actual = gemm_a8w8_blockscale_group32(x, weight, xs, ws, dtype=torch.bfloat16)
    if 255 in (a_code, b_code):
        assert actual.isnan().all()
    else:
        expected = torch.full_like(actual, 1280 * 2.0 ** (a_code + b_code - 254))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "rows,n,k,split_k", [(3, 4096, 1280, None), (33, 8193, 576, 3)]
)
@pytest.mark.usefixtures("_require_gfx950")
def test_group32_projection_graph_reads_live_inputs_and_scales(rows, n, k, split_k):
    x, weight, xs, ws = generate_group32_inputs(2 * rows, n, k)
    gemm_a8w8_blockscale_group32(
        x, weight, xs, ws, dtype=torch.bfloat16, split_k=split_k
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gemm_a8w8_blockscale_group32(
            x, weight, xs, ws, dtype=torch.bfloat16, split_k=split_k
        )
    for factor in (2, 0.5):
        x.copy_((x.float() * factor).to(x.dtype))
        xs.view(torch.uint8).add_(1)
        graph.replay()
        expected = run_group32_reference(x, weight, xs, ws)
        torch.testing.assert_close(
            actual.float(),
            expected.to(torch.bfloat16).float(),
            rtol=0.016,
            atol=5e-5 * expected.abs().max().item(),
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.usefixtures("_require_gfx950")
def test_row_scaled_fp8_projection_with_token_and_column_tails(dtype):
    x, weight, xs, ws = generate_group32_inputs(97, 8193, 576)
    # Per-row scales share the scheduling kernel but not the compact weight grid.
    ws = ws.repeat_interleave(32, 0)[: weight.shape[0]].contiguous()
    actual = gemm_a8w8_blockscale_group32(
        x, weight, xs, ws, weight_group_rows=1, dtype=dtype, split_k=3
    )
    expected = (x.double() * xs.double().repeat_interleave(32, -1)) @ (
        weight.double() * ws.double().repeat_interleave(32, -1)
    ).T
    torch.testing.assert_close(
        actual.float(),
        expected.to(dtype).float(),
        rtol=0.016,
        atol=5e-5 * expected.abs().max().item(),
    )


@pytest.mark.parametrize(
    "m,n,k", [(0, 65, 64), (3, 2053, 1280), (66, 8193, 576), (512, 4096, 1280)]
)
@pytest.mark.parametrize("group_n", [1, 32])
@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("raw_views", [False, True])
@pytest.mark.usefixtures("_require_gfx950")
def test_public_group32_dispatch(m, n, k, group_n, configured, raw_views, monkeypatch):
    if configured:
        monkeypatch.setattr(
            gemm_op_a8w8, "get_CKGEMM_config", lambda *args: {"libtype": "triton"}
        )
    x, w, xs, ws = generate_group32_inputs(m, n, k)
    if group_n == 1:
        ws = ws.repeat_interleave(32, 0)[:n].contiguous()
    expected = gemm_a8w8_blockscale_group32(x, w, xs, ws, weight_group_rows=group_n)
    operands = (x, w, xs, ws)
    if raw_views:
        operands = tuple(t.view(torch.uint8) for t in operands)
    actual = gemm_a8w8_blockscale(*operands)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("split_k", [None, 1, 3, 100])
@pytest.mark.usefixtures("_require_gfx950")
def test_raw_views_and_preallocated_output(split_k):
    x, w, xs, ws = generate_group32_inputs(63, 8193, 576)
    output = torch.empty((63, 8193), dtype=torch.bfloat16, device="cuda")
    actual = gemm_a8w8_blockscale_group32(
        x.view(torch.uint8),
        w.view(torch.uint8),
        xs.view(torch.uint8),
        ws.view(torch.uint8),
        y=output,
        split_k=split_k,
    )
    assert actual is output
    expected = run_group32_reference(x, w, xs, ws).to(output.dtype)
    checkAllclose(
        expected,
        actual,
        rtol=0.016,
        atol=5e-5 * expected.abs().max().item(),
        catastrophic_check=True,
    )


@pytest.mark.parametrize(
    "invalid", ["stride", "scale_shape", "scale_dtype", "output", "split"]
)
@pytest.mark.usefixtures("_require_gfx950")
def test_invalid_group32_contract(invalid):
    x, w, xs, ws = generate_group32_inputs(3, 2053, 1280)
    kwargs = {}
    if invalid == "stride":
        x = x.T.contiguous().T
    elif invalid == "scale_shape":
        ws = ws[:-1]
    elif invalid == "scale_dtype":
        xs = xs.float()
    elif invalid == "output":
        kwargs["y"] = torch.empty((3, 2052), dtype=torch.bfloat16, device="cuda")
    else:
        kwargs["split_k"] = 0
    with pytest.raises(AssertionError):
        gemm_a8w8_blockscale_group32(x, w, xs, ws, **kwargs)


@pytest.mark.parametrize("split_k", [None, 3])
@pytest.mark.usefixtures("_require_gfx950")
def test_public_group32_compile_dynamic_rows(split_k):
    def forward(x, w, xs, ws):
        return gemm_a8w8_blockscale(x, w, xs, ws, split_k=split_k)

    compiled = torch.compile(forward, fullgraph=True, dynamic=True)
    for m in (3, 7, 65):
        x, w, xs, ws = generate_group32_inputs(m, 4096, 1280)
        torch.testing.assert_close(
            compiled(x, w, xs, ws), forward(x, w, xs, ws), rtol=0, atol=0
        )


@pytest.mark.parametrize("configured", [False, True])
@pytest.mark.parametrize("split_k", [None, 3])
@pytest.mark.usefixtures("_require_gfx950")
def test_public_group32_graph_replay(configured, split_k, monkeypatch):
    if configured:
        monkeypatch.setattr(
            gemm_op_a8w8, "get_CKGEMM_config", lambda *args: {"libtype": "triton"}
        )
    x, w, xs, ws = generate_group32_inputs(3, 4096, 1280)
    gemm_a8w8_blockscale(x, w, xs, ws, split_k=split_k)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gemm_a8w8_blockscale(x, w, xs, ws, split_k=split_k)
    x.copy_((x.float() * 0.5).to(x.dtype))
    xs.view(torch.uint8).add_(1)
    graph.replay()
    expected = gemm_a8w8_blockscale_group32(x, w, xs, ws, split_k=split_k)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("n", [1, 2, 16, 31, 32, 33])
@pytest.mark.parametrize("group_n", [1, 32])
@pytest.mark.parametrize("raw_views", [False, True])
@pytest.mark.usefixtures("_require_gfx950")
def test_public_small_n_scale_layouts(n, group_n, raw_views):
    x, w, xs, ws = generate_group32_inputs(3, n, 128, weight_group_rows=group_n)
    expected = run_group32_reference(x, w, xs, ws, weight_group_rows=group_n)
    operands = (x, w, xs, ws)
    if raw_views:
        operands = tuple(t.view(torch.uint8) for t in operands)
    actual = gemm_a8w8_blockscale(*operands, dtype=torch.bfloat16)
    torch.testing.assert_close(
        actual.float(),
        expected.to(torch.bfloat16).float(),
        rtol=0.016,
        atol=5e-5 * expected.abs().max().item(),
    )


_FUSED_TILE = {
    "BLOCK_SIZE_M": 16,
    "BLOCK_SIZE_N": 32,
    "BLOCK_SIZE_K": 256,
    "GROUP_SIZE_M": 1,
    "cache_modifier": "",
    "num_warps": 2,
    "num_stages": 2,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16,
    "NUM_KSPLIT": 5,
    "N_FIRST": False,
    "REDUCE_BLOCK_SIZE_M": 32,
    "REDUCE_BLOCK_SIZE_N": 32,
    "FUSED_SPLITK": True,
}
_FUSED_PACKED = dict(
    _FUSED_TILE,
    packed={
        "BLOCK_SIZE_M": 8,
        "BLOCK_SIZE_N": 16,
        "BLOCK_SIZE_K": 256,
        "K_PACK": 2,
        "cache_modifier": "",
        "NUM_KSPLIT": 5,
        "num_warps": 2,
        "num_stages": 2,
        "waves_per_eu": 0,
        "matrix_instr_nonkdim": 16,
    },
)
_FUSED = {"tile": _FUSED_TILE, "packed": _FUSED_PACKED}


def _assert_matches_reference(actual, x, w, xs, ws, dtype, weight_group_rows=32):
    expected = run_group32_reference(x, w, xs, ws, weight_group_rows)
    torch.testing.assert_close(
        actual.float(),
        expected.to(dtype).float(),
        rtol=0.016,
        atol=5e-5 * expected.abs().max().item(),
    )


@pytest.mark.parametrize("variant", ["tile", "packed"])
@pytest.mark.parametrize(
    "m,n,k", [(1, 512, 5120), (3, 2053, 1280), (16, 1152, 5120), (37, 5120, 2304)]
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.usefixtures("_require_gfx950")
def test_fused_split_k_matches_reference(variant, m, n, k, dtype):
    x, w, xs, ws = generate_group32_inputs(m, n, k)
    actual = gemm_a8w8_blockscale_group32(
        x, w, xs, ws, dtype=dtype, config=_FUSED[variant]
    )
    _assert_matches_reference(actual, x, w, xs, ws, dtype)


@pytest.mark.parametrize("variant", ["tile", "packed"])
@pytest.mark.parametrize("m,n,k", [(3, 2053, 1280), (37, 5120, 2336)])
@pytest.mark.usefixtures("_require_gfx950")
def test_weight_cache_modifier_keeps_results(variant, m, n, k):
    x, w, xs, ws = generate_group32_inputs(m, n, k)
    config = _FUSED[variant]
    cached = dict(config, cache_modifier=".cg")
    if variant == "packed":
        cached["packed"] = dict(config["packed"], cache_modifier=".cg")
    run = gemm_a8w8_blockscale_group32
    expected = run(x, w, xs, ws, dtype=torch.bfloat16, config=config)
    actual = run(x, w, xs, ws, dtype=torch.bfloat16, config=cached)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.usefixtures("_require_gfx950")
def test_fused_split_k_row_scales():
    x, w, xs, ws = generate_group32_inputs(5, 2053, 1280, weight_group_rows=1)
    actual = gemm_a8w8_blockscale_group32(
        x, w, xs, ws, weight_group_rows=1, dtype=torch.bfloat16, config=_FUSED_TILE
    )
    _assert_matches_reference(actual, x, w, xs, ws, torch.bfloat16, 1)


@pytest.mark.parametrize("variant", ["tile", "packed"])
@pytest.mark.usefixtures("_require_gfx950")
def test_fused_split_k_is_deterministic_and_rezeroes_counters(variant):
    x, w, xs, ws = generate_group32_inputs(4, 1152, 5120)
    run = lambda: gemm_a8w8_blockscale_group32(
        x, w, xs, ws, dtype=torch.bfloat16, config=_FUSED[variant]
    )
    first = run()
    # A stale counter or an early partial read would change a later sum.
    assert all(torch.equal(first, run()) for _ in range(50))
    torch.cuda.synchronize()
    assert not afp8wfp8_op._split_counters(x.device).any()


@pytest.mark.parametrize("variant", ["tile", "packed"])
@pytest.mark.usefixtures("_require_gfx950")
def test_fused_split_k_graph_replay_on_warmed_stream(variant):
    x, w, xs, ws = generate_group32_inputs(6, 1152, 5120)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        gemm_a8w8_blockscale_group32(
            x, w, xs, ws, dtype=torch.bfloat16, config=_FUSED[variant]
        )
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = gemm_a8w8_blockscale_group32(
            x, w, xs, ws, dtype=torch.bfloat16, config=_FUSED[variant]
        )
    for factor in (2, 0.5):
        x.copy_((x.float() * factor).to(x.dtype))
        xs.view(torch.uint8).add_(1)
        graph.replay()
        torch.cuda.synchronize()
        _assert_matches_reference(actual, x, w, xs, ws, torch.bfloat16)


@pytest.mark.parametrize("variant", ["tile", "packed"])
@pytest.mark.usefixtures("_require_gfx950")
def test_fused_split_k_first_use_inside_capture(variant, monkeypatch):
    # Isolate the counter cache so earlier tests cannot warm the capture stream.
    torch.cuda.synchronize()
    monkeypatch.setattr(afp8wfp8_op, "_split_counters_by_stream", {})
    # Before torch 2.10 counters cannot be allocated mid-capture: runs unfused.
    x, w, xs, ws = generate_group32_inputs(6, 1152, 5120)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = gemm_a8w8_blockscale_group32(
            x, w, xs, ws, dtype=torch.bfloat16, config=_FUSED[variant]
        )
    graph.replay()
    torch.cuda.synchronize()
    _assert_matches_reference(actual, x, w, xs, ws, torch.bfloat16)


@pytest.mark.parametrize("variant", ["tile", "packed"])
@pytest.mark.usefixtures("_require_gfx950")
def test_fused_split_k_concurrent_streams(variant):
    inputs = [generate_group32_inputs(m, 1152, 5120) for m in (3, 5)]
    expected = [
        gemm_a8w8_blockscale_group32(*t, dtype=torch.bfloat16, config=_FUSED[variant])
        for t in inputs
    ]
    torch.cuda.synchronize()
    streams = [torch.cuda.Stream() for _ in inputs]
    outputs = [[], []]
    # Unsynchronized interleaving keeps both streams' reductions in flight.
    for _ in range(40):
        for i, (stream, operands) in enumerate(zip(streams, inputs)):
            with torch.cuda.stream(stream):
                outputs[i].append(
                    gemm_a8w8_blockscale_group32(
                        *operands, dtype=torch.bfloat16, config=_FUSED[variant]
                    )
                )
    torch.cuda.synchronize()
    for want, got in zip(expected, outputs):
        assert all(torch.equal(want, y) for y in got)


# ---------------------------------------------------------------------------
# gemm_afp8wfp8_preshuffle: the gfx1250 gluon MXFP8 GEMM (e4m3 x e4m3, 1x32 e8m0
# scales on both operands; also 1x128 x and 32x32 / 128x128 w scales).
# ---------------------------------------------------------------------------

PRESHUFFLE_SCALE_GROUP = 32

# Applied per test, not as a module pytestmark, since the rest of this file
# covers other arches.
_requires_gluon_preshuffle = pytest.mark.skipif(
    not _is_gluon_available(),
    reason="gemm_afp8wfp8_preshuffle is gfx1250 (gluon) only",
)


def preshuffle_generate_inputs(M, N, K, seed=0):
    """e4m3 x/w in [-2, 2) and e8m0 scales in [2^-3, 2^1]; returns raw tensors."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.rand((M, K), device="cuda", generator=g) * 4 - 2).to(torch.float8_e4m3fn)
    w = (torch.rand((N, K), device="cuda", generator=g) * 4 - 2).to(torch.float8_e4m3fn)
    xs = torch.randint(
        124,
        129,
        (M, K // PRESHUFFLE_SCALE_GROUP),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    ws = torch.randint(
        124,
        129,
        (N, K // PRESHUFFLE_SCALE_GROUP),
        dtype=torch.uint8,
        device="cuda",
        generator=g,
    )
    return x, w, xs, ws


def preshuffle_layout_scales(s, scale_layout):
    if scale_layout == "128row":
        return shuffle_scale_gemm_128row(s)
    return shuffle_mxfp8fp4_scale(s.contiguous()).view(
        -1, s.shape[1] * PRESHUFFLE_SCALE_GROUP
    )


def preshuffle_prepare(x, w, xs, ws, scale_layout, x_preshuffled):
    """Kernel-side inputs: preshuffled w (N // 16, K * 16), optional preshuffled x, laid-out scales."""
    M, K = x.shape
    N = w.shape[0]
    w_k = shuffle_weight(w.view(torch.uint8)).reshape(N // 16, K * 16)
    x_k = shuffle_weight(x.view(torch.uint8)) if x_preshuffled else x
    # x scales of a CTA with fewer than 32 rows are read un-shuffled (row-major)
    xs_k = xs.contiguous() if M < 32 else preshuffle_layout_scales(xs, scale_layout)
    return x_k, w_k, xs_k, preshuffle_layout_scales(ws, scale_layout)


def preshuffle_reference(x, w, xs, ws):
    xd = x.float() * torch.exp2(xs.float() - 127).repeat_interleave(
        PRESHUFFLE_SCALE_GROUP, 1
    )
    wd = w.float() * torch.exp2(ws.float() - 127).repeat_interleave(
        PRESHUFFLE_SCALE_GROUP, 1
    )
    return xd @ wd.T


def preshuffle_check(y, ref):
    err = ((y.float() - ref).norm() / ref.norm()).item()
    assert err < 1e-2, f"rel_l2 {err:.3e}"
    # elementwise: within a couple of bf16 ULPs of the value range
    tol = 2e-2 * ref.abs().max().item()
    assert (y.float() - ref).abs().max().item() <= tol


PRESHUFFLE_SHAPES = [
    (1, 1024, 256),
    (16, 4096, 1024),
    (31, 7168, 512),
    (32, 1024, 1024),
    (150, 2112, 1024),
    (256, 8192, 2048),
    (1024, 1024, 1024),
    (2048, 7168, 4096),
    (4096, 4096, 4096),
    (8192, 8192, 1024),
]


@_requires_gluon_preshuffle
@pytest.mark.parametrize("M, N, K", PRESHUFFLE_SHAPES)
@pytest.mark.parametrize("scale_layout", ["n32k4", "128row"])
@pytest.mark.parametrize("x_preshuffled", [False, True])
def test_gemm_afp8wfp8_preshuffle(M, N, K, scale_layout, x_preshuffled):
    if x_preshuffled and M % 16:
        pytest.skip("x_preshuffled needs M % 16 == 0")
    x, w, xs, ws = preshuffle_generate_inputs(M, N, K)
    x_k, w_k, xs_k, ws_k = preshuffle_prepare(x, w, xs, ws, scale_layout, x_preshuffled)
    y = gemm_afp8wfp8_preshuffle(
        x_k,
        w_k,
        xs_k,
        ws_k,
        torch.bfloat16,
        x_preshuffled=x_preshuffled,
        scale_layout=scale_layout,
    )
    preshuffle_check(y, preshuffle_reference(x, w, xs, ws))


PRESHUFFLE_CLUSTER_CONFIG = dict(
    BLOCK_SIZE_M=1024,
    BLOCK_SIZE_N=1024,
    BLOCK_SIZE_K=128,
    num_warps=4,
    NUM_BUFFERS=3,
    num_ctas=16,
    GROUP_M=8,
    B2B_WMMA=1,
    NUM_KSUB=1,
    EPI_SPLIT=1,
)


@_requires_gluon_preshuffle
@pytest.mark.parametrize("EPI_SPLIT", [1, 2])
@pytest.mark.parametrize("x_preshuffled", [False, True])
def test_gemm_afp8wfp8_preshuffle_cluster_config(EPI_SPLIT, x_preshuffled):
    """Explicit 16-CTA cluster config, incl. the sliced epilogue and a ragged N."""
    M, N, K = 2048, 3072, 2048
    x, w, xs, ws = preshuffle_generate_inputs(M, N, K, seed=1)
    x_k, w_k, xs_k, ws_k = preshuffle_prepare(x, w, xs, ws, "n32k4", x_preshuffled)
    y = gemm_afp8wfp8_preshuffle(
        x_k,
        w_k,
        xs_k,
        ws_k,
        torch.bfloat16,
        config=dict(PRESHUFFLE_CLUSTER_CONFIG, EPI_SPLIT=EPI_SPLIT),
        x_preshuffled=x_preshuffled,
    )
    preshuffle_check(y, preshuffle_reference(x, w, xs, ws))


@_requires_gluon_preshuffle
def test_gemm_afp8wfp8_preshuffle_rejects_layout_config_keys():
    x, w, xs, ws = preshuffle_generate_inputs(256, 256, 256)
    x_k, w_k, xs_k, ws_k = preshuffle_prepare(x, w, xs, ws, "n32k4", False)
    with pytest.raises(ValueError):
        gemm_afp8wfp8_preshuffle(
            x_k, w_k, xs_k, ws_k, config=dict(PRESHUFFLE_CLUSTER_CONFIG, SCALE_MODE=2)
        )


# --- fp8 element formats and coarser e8m0 scale granularities ------------------------

PRESHUFFLE_FP8_DTYPES = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}


def preshuffle_generate_inputs_fmt(M, N, K, a_fmt, b_fmt, x_group, w_block, seed=0):
    """fp8 x/w of the given formats, x scales per (1, x_group), w scales per w_block."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.rand((M, K), device="cuda", generator=g) * 4 - 2).to(
        PRESHUFFLE_FP8_DTYPES[a_fmt]
    )
    w = (torch.rand((N, K), device="cuda", generator=g) * 4 - 2).to(
        PRESHUFFLE_FP8_DTYPES[b_fmt]
    )
    xs = torch.randint(
        124, 129, (M, K // x_group), dtype=torch.uint8, device="cuda", generator=g
    )
    bn, bk = w_block
    ws = torch.randint(
        124, 129, (-(-N // bn), K // bk), dtype=torch.uint8, device="cuda", generator=g
    )
    return x, w, xs, ws


def preshuffle_reference_fmt(x, w, xs, ws, x_group, w_block):
    N = w.shape[0]
    bn, bk = w_block
    xd = x.float() * torch.exp2(xs.float() - 127).repeat_interleave(x_group, 1)
    wsd = (
        torch.exp2(ws.float() - 127)
        .repeat_interleave(bn, 0)[:N]
        .repeat_interleave(bk, 1)
    )
    return xd @ (w.float() * wsd).T


def preshuffle_run_fmt(
    M, N, K, a_fmt, b_fmt, x_group, w_block, scale_layout="n32k4", x_preshuffled=False
):
    x, w, xs, ws = preshuffle_generate_inputs_fmt(
        M, N, K, a_fmt, b_fmt, x_group, w_block
    )
    if x_group == 128:
        xs_k = xs.T.contiguous()  # compact column-major (K // 128, M)
    elif M < 32:
        xs_k = xs.contiguous()
    else:
        xs_k = preshuffle_layout_scales(xs, scale_layout)
    # MX 1x32 w scales are laid out; coarse blocks go in compact, as in the checkpoint
    ws_k = preshuffle_layout_scales(ws, scale_layout) if w_block == (1, 32) else ws
    w_k = shuffle_weight(w.view(torch.uint8)).reshape(N // 16, K * 16).view(w.dtype)
    x_k = shuffle_weight(x.view(torch.uint8)).view(x.dtype) if x_preshuffled else x
    y = gemm_afp8wfp8_preshuffle(
        x_k,
        w_k,
        xs_k,
        ws_k,
        torch.bfloat16,
        x_preshuffled=x_preshuffled,
        scale_layout=scale_layout,
        x_scale_group_size=x_group,
        w_scale_block_size=w_block,
    )
    preshuffle_check(y, preshuffle_reference_fmt(x, w, xs, ws, x_group, w_block))


@_requires_gluon_preshuffle
@pytest.mark.parametrize(
    "a_fmt, b_fmt",
    [("e4m3", "e4m3"), ("e5m2", "e5m2"), ("e4m3", "e5m2"), ("e5m2", "e4m3")],
)
@pytest.mark.parametrize(
    "M",
    [
        1,
        7,
        16,
        31,
        32,
        61,
        64,
        127,
        128,
        192,
        256,
        511,
        512,
        1000,
        1024,
        1536,
        2048,
        3072,
        8193,
        16384,
    ],
)
@pytest.mark.parametrize(
    "N, K",
    [
        (4096, 1024),
        (2112, 2048),
        (4096, 2048),
        (9216, 2048),
        (12352, 2048),
        (2048, 4096),
    ],
)
def test_gemm_afp8wfp8_preshuffle_formats(M, N, K, a_fmt, b_fmt):
    preshuffle_run_fmt(M, N, K, a_fmt, b_fmt, 32, (1, 32))


@_requires_gluon_preshuffle
@pytest.mark.parametrize("w_block", [(1, 32), (32, 32), (128, 128)])
@pytest.mark.parametrize(
    "M, N, K",
    [
        (1, 1024, 256),
        (31, 7168, 512),
        (150, 2112, 1024),
        (256, 8192, 2048),
        (4096, 4096, 4096),
    ],
)
@pytest.mark.parametrize("x_preshuffled", [False, True])
def test_gemm_afp8wfp8_preshuffle_x1x128(M, N, K, w_block, x_preshuffled):
    """DeepSeek-V3.1-style UE8M0: 1x128 x scales (compact column-major) x coarse w scales."""
    if x_preshuffled and M % 16:
        pytest.skip("x_preshuffled needs M % 16 == 0")
    preshuffle_run_fmt(
        M, N, K, "e4m3", "e4m3", 128, w_block, x_preshuffled=x_preshuffled
    )


@_requires_gluon_preshuffle
@pytest.mark.parametrize("w_block", [(32, 32), (128, 128)])
@pytest.mark.parametrize("scale_layout", ["n32k4", "128row"])
@pytest.mark.parametrize("M, N, K", [(31, 1024, 1024), (2048, 7168, 4096)])
def test_gemm_afp8wfp8_preshuffle_w_blocks(M, N, K, w_block, scale_layout):
    preshuffle_run_fmt(M, N, K, "e4m3", "e4m3", 32, w_block, scale_layout=scale_layout)


@_requires_gluon_preshuffle
def test_gemm_afp8wfp8_preshuffle_rejects_bad_args():
    x, w, xs, ws = preshuffle_generate_inputs_fmt(
        256, 256, 256, "e4m3", "e4m3", 128, (128, 128)
    )
    w_k = shuffle_weight(w.view(torch.uint8)).reshape(16, 256 * 16)
    xs_k = xs.T.contiguous()
    with pytest.raises(ValueError):
        gemm_afp8wfp8_preshuffle(
            x, w_k, xs_k, ws, x_scale_group_size=64, w_scale_block_size=(128, 128)
        )
    with pytest.raises(AssertionError):  # 1x128 x scales must be (K // 128, M)
        gemm_afp8wfp8_preshuffle(
            x, w_k, xs, ws, x_scale_group_size=128, w_scale_block_size=(128, 128)
        )
    with pytest.raises(ValueError):  # unsupported w block
        gemm_afp8wfp8_preshuffle(
            x, w_k, xs_k, ws, x_scale_group_size=128, w_scale_block_size=(64, 64)
        )
    with pytest.raises(ValueError):  # compact w scales of the wrong shape
        gemm_afp8wfp8_preshuffle(
            x,
            w_k,
            xs_k,
            ws.T.contiguous()[:1],
            x_scale_group_size=128,
            w_scale_block_size=(128, 128),
        )
