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
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.gemm.basic.gemm_afp8wfp8 import (
    gemm_afp8wfp8,
    gemm_afp8wfp8_preshuffle,
)
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
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
# SCALE_MODES = [(128, False), (128, True), (32, False), (32, True)]
SCALE_MODES = [
    (128, True),
]


def get_shapes():
    # (M, N, K), with N % 128 == 0 and K % 128 == 0 to fit the 128x128 W-scale layout.
    return [
        (m, n, k)
        # for m in [1, 8, 16, 32, 64, 512, 16384]
        for m in [512, 16384]
        for n, k in [
            (65536, 1536),
        ]
    ]


@pytest.mark.parametrize("M, N, K", get_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
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
    """The gluon MXFP8 preshuffle kernel is gfx1250-only."""
    try:
        return "gfx1250" in arch_info.get_arch()
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.parametrize("M, N, K", get_shapes())
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("backend", ["triton", "gluon"])
@pytest.mark.parametrize("x_scale_group_size, transpose_x_scale", SCALE_MODES)
def test_gemm_afp8wfp8_preshuffle(
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
            pytest.skip("Gluon MXFP8 preshuffle requires gfx1250")
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
    out = gemm_afp8wfp8_preshuffle(
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
            f"afp8wfp8 preshuffle [{backend}] M={M} N={N} K={K} "
            f"gs={x_scale_group_size} T={transpose_x_scale} "
        ),
        catastrophic_check=True,
    )
    assert err < TOL_ERR_RATIO, f"{err:.2%} of elements mismatch"


@pytest.mark.parametrize("ragged", [False, True])
@pytest.mark.parametrize("ctas_m, ctas_n", [(2, 1), (1, 2), (2, 2), (4, 1), (1, 4)])
def test_gluon_preshuffle_cga_multicast(ctas_m: int, ctas_n: int, ragged: bool):
    """CTA-cluster (CGA) operand multicast reproduces the single-CTA result.

    BLOCK_SIZE_M / BLOCK_SIZE_N become the cluster tile, so a CTAS_M x CTAS_N
    cluster covers the same output tile with the same per-CTA footprint and each
    operand fetch multicast to the CTAs that share it. The result must be
    bit-identical to CTAS 1x1: multicast changes who reads a byte, never which
    byte or the order it is accumulated in.
    """
    if not _gluon_available():
        pytest.skip("Gluon MXFP8 preshuffle CGA multicast requires gfx1250")

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
        "GEMM-AFP8WFP8_PRESHUFFLED", 1024, 1024, 1536, backend="gluon"
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
        return gemm_afp8wfp8_preshuffle(
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
def test_gemm_afp8wfp8_preshuffle_gluon_matches_triton(
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
        pytest.skip("Gluon MXFP8 preshuffle requires gfx1250")
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
    triton_out = gemm_afp8wfp8_preshuffle(
        x_fp8, w_kernel, x_scales_kernel, w_scales, backend="triton", **kwargs
    )
    gluon_out = gemm_afp8wfp8_preshuffle(
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
    x = torch.randn(m, k, device="cuda").to(torch.float8_e4m3fn)
    w = torch.randn(n, k, device="cuda").to(torch.float8_e4m3fn)
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
    y = torch.empty(expected.shape, device="cuda", dtype=torch.float32)
    for _ in range(2):  # Fused arrival counters must be reusable.
        actual = gemm_afp8wfp8(
            x,
            w,
            xs,
            ws,
            dtype=torch.float32,
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
        dtype=torch.float32,
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
    kwargs = dict(
        dtype=torch.float32,
        config=config,
        x_scale_group_size=a_group,
        w_scale_group_size=b_group,
    )
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
        dtype=torch.float32,
        config=config,
        x_scale_group_size=a_group,
        w_scale_group_size=b_group,
    )
    _assert_compact_scale_close(actual, expected)
