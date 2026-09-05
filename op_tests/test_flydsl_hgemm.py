# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf for FlyDSL A16W16 HGEMM (gfx950).

The model path (``tuned_gemm.flydsl_gemm``) is BF16 x BF16:

    a    : [M, K]  contiguous
    w    : [N, K]  contiguous
    out  : [M, N]  preallocated
    y    = flydsl_hgemm(a, w, out=out, ...)   # kernel sees w.t() as NT

``gemm_a16w16`` is the same kernel with explicit layout (nn/nt/tn/tt). Each
layout character is a stride only: N = row-major, T = a ``.t()`` view of a
contiguous tensor — the same non-contiguous views the kernel is asked to
accept. Logical shapes never change.

Run:
    python op_tests/test_flydsl_hgemm.py
    python op_tests/test_flydsl_hgemm.py -s 128,4096,4096 -l nt --policy hti
    python op_tests/test_flydsl_hgemm.py -l nn nt tn tt
"""

import argparse
import functools
import itertools
from types import SimpleNamespace

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import flydsl_hgemm
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

torch.set_default_device("cuda")

# Public wrapper currently supports gfx950 only. Positive allow-list: an unknown
# new card must not silently run an unbuilt kernel.
SUPPORTED_GFX = ["gfx950"]
SEED = 0


@functools.lru_cache(maxsize=1)
def _a16w16():
    """Lazy gfx950 kernel symbols. NT @benchmark path uses flydsl_hgemm."""
    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950 import (
        GEMM_A16W16_DTYPE_BF16,
        GEMM_A16W16_DTYPE_FP16,
        GEMM_A16W16_DTYPE_FP32,
        gemm_a16w16,
        make_gemm_a16w16_param_and_validate,
    )
    from aiter.ops.flydsl.kernels.gemm_a16w16_gfx950_utils import GFX950_DMA_BYTES

    return SimpleNamespace(
        gemm_a16w16=gemm_a16w16,
        validate=make_gemm_a16w16_param_and_validate,
        dma_bytes=GFX950_DMA_BYTES,
        dtype_bf16=GEMM_A16W16_DTYPE_BF16,
        dtype_fp16=GEMM_A16W16_DTYPE_FP16,
        dtype_fp32=GEMM_A16W16_DTYPE_FP32,
    )


def _in_dtype_id(dtype):
    kern = _a16w16()
    return kern.dtype_fp16 if dtype is dtypes.fp16 else kern.dtype_bf16


def _out_dtype_id(in_dtype, out_dtype):
    if out_dtype is dtypes.fp32:
        return _a16w16().dtype_fp32
    return _in_dtype_id(in_dtype)


def _tile(
    block_m,
    block_n,
    block_k,
    stages,
    m_waves,
    n_waves,
    k_waves,
    group_m,
    split_k,
    use_hti,
):
    return {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "stages": stages,
        "split_k": split_k,
        "m_waves": m_waves,
        "n_waves": n_waves,
        "k_waves": k_waves,
        "group_m": group_m,
        "use_half_tile_interleaved": use_hti,
    }


# Exact policies from the previous pytest cases that generic fallbacks miss
# (split-K, slice-K / k_waves, fp32 C, group_m swizzle).
_KNOWN_TILES = {
    (32, 384, 7168, "ft", 8): _tile(32, 64, 64, 5, 2, 2, 1, 0, 8, False),
    (3, 5120, 2880, "ft", 1): _tile(64, 64, 64, 5, 2, 2, 1, 0, 1, False),
    (3, 5120, 2880, "ft", 3): _tile(16, 64, 64, 7, 1, 2, 1, 0, 3, False),
    (800, 384, 7168, "ft", 1): _tile(32, 64, 128, 6, 1, 2, 2, 0, 1, False),
    (64, 64, 512, "hti", 2): _tile(64, 64, 64, 2, 2, 2, 1, 0, 2, True),
    (1280, 64, 4096, "hti", 8): _tile(64, 64, 256, 2, 2, 2, 1, 0, 8, True),
    (64, 128, 1024, "ft", 1): _tile(32, 64, 128, 4, 1, 2, 4, 0, 1, False),
    (64, 128, 2048, "ft", 1): _tile(32, 64, 128, 4, 1, 2, 2, 0, 1, False),
    (64, 64, 768, "ft", 1): _tile(64, 64, 64, 2, 2, 2, 1, 0, 1, False),
    (64, 64, 768, "hti", 3): _tile(64, 64, 64, 2, 2, 2, 1, 0, 3, True),
    (320, 4096, 256, "ft", 1): _tile(64, 64, 64, 2, 2, 2, 1, 4, 1, False),
    (320, 4032, 256, "ft", 1): _tile(64, 64, 64, 2, 2, 2, 1, 4, 1, False),
}


def _fallback_tiles(m, n, k, policy, split_k):
    # (block_m, block_n, block_k, stages, m_waves, n_waves, k_waves, group_m)
    if policy in ("ht", "hti"):
        hti_opts = [
            (256, 256, 64, 2, 2, 4, 1, 0),
            (256, 256, 64, 2, 2, 4, 1, 4),
            (128, 128, 64, 2, 2, 2, 1, 0),
            (64, 64, 256, 2, 2, 2, 1, 0),
            (64, 128, 128, 2, 2, 4, 1, 0),
            (64, 64, 64, 2, 2, 2, 1, 0),
        ]
        return [_tile(*opt, split_k, True) for opt in hti_opts]
    ft_opts = [
        (256, 256, 64, 2, 2, 4, 1, 4),
        (128, 256, 64, 3, 4, 4, 1, 4),
        (128, 128, 64, 4, 2, 4, 1, 4),
        (64, 128, 64, 6, 2, 4, 1, 4),
        (64, 64, 64, 4, 2, 2, 1, 0),
        (64, 64, 64, 5, 2, 2, 1, 0),
        (32, 64, 128, 6, 1, 2, 2, 0),
        (32, 64, 128, 4, 1, 2, 4, 0),
        (32, 64, 64, 5, 2, 2, 1, 0),
        (32, 32, 64, 8, 2, 1, 1, 0),
        (16, 64, 64, 7, 1, 2, 1, 0),
        (16, 32, 256, 4, 1, 2, 2, 4),
        (16, 16, 256, 4, 1, 1, 2, 4),
        (16, 16, 128, 8, 1, 1, 1, 4),
        (16, 16, 64, 8, 1, 1, 1, 4),
    ]

    def _score(opt):
        block_m, block_n, *_rest = opt
        return (
            int(block_m > m) + int(block_n > n),
            abs(block_m - m) + abs(block_n - n),
        )

    return [_tile(*opt, split_k, False) for opt in sorted(ft_opts, key=_score)]


def _validate_kwargs(tile, dtype, out_dtype, layout, has_bias):
    c_dtype = dtype if out_dtype is None else out_dtype
    kw = dict(tile)
    kw.update(
        in_dtype_id=_in_dtype_id(dtype),
        out_dtype_id=_out_dtype_id(dtype, c_dtype),
        a_is_transposed=layout[0] == "t",
        b_is_transposed=layout[1] == "t",
        has_bias=bool(has_bias),
    )
    return kw


def _pick_tile(m, n, k, dtype, out_dtype, layout, has_bias, policy, split_k):
    """Return a launchable tile dict, or None if this config is unsupported."""
    known = _KNOWN_TILES.get((m, n, k, policy, split_k))
    tiles = [known] if known is not None else []
    tiles.extend(_fallback_tiles(m, n, k, policy, split_k))
    for tile in tiles:
        if tile is None:
            continue
        if tile["split_k"] != split_k:
            continue
        if tile["use_half_tile_interleaved"] != (policy in ("ht", "hti")):
            continue
        kw = _validate_kwargs(tile, dtype, out_dtype, layout, has_bias)
        if _a16w16().validate(m, n, k, kw) is not None:
            return tile
    return None


def run_torch(a, b, bias, out_dtype):
    # Reference only: fp32 math, cast back. Not timed, not in the table.
    if bias is None:
        ref = torch.mm(a.to(dtypes.fp32), b.to(dtypes.fp32))
    else:
        ref = torch.addmm(bias.to(dtypes.fp32), a.to(dtypes.fp32), b.to(dtypes.fp32))
    return ref.to(out_dtype)


def _empty_layout_matrix(nrow, ncol, dtype, transposed):
    # T is a transposed view of a contiguous tensor (non-contiguous leading
    # stride), matching how the kernel is fed column-major operands.
    if transposed:
        return torch.empty((ncol, nrow), dtype=dtype, device="cuda").t()
    return torch.empty((nrow, ncol), dtype=dtype, device="cuda")


def _create_inputs(m, n, k, dtype, layout, has_bias):
    torch.manual_seed(SEED)
    a = _empty_layout_matrix(m, k, dtype, layout[0] == "t")
    b = _empty_layout_matrix(k, n, dtype, layout[1] == "t")
    a.uniform_(-1, 1)
    b.uniform_(-1, 1)
    bias = None
    if has_bias:
        bias = torch.empty((n,), dtype=dtype, device="cuda")
        bias.uniform_(10, 20)
    return a, b, bias


def _atol_rtol(k, split_k, k_waves, has_bias, dtype):
    k_scale = (k / 8192) ** 0.5
    k_scale *= split_k * k_waves
    atol_scale = 1.5 if has_bias else 1.0
    if dtype is dtypes.bf16:
        return 2e-1 * k_scale * atol_scale, 2e-1
    return 5e-2 * k_scale * atol_scale, 5e-2


@benchmark()
def test_hgemm(m, n, k, dtype, layout, has_bias, policy, split_k, out_dtype):
    c_dtype = dtype if out_dtype is None else out_dtype
    tile = _pick_tile(m, n, k, dtype, out_dtype, layout, has_bias, policy, split_k)
    # main() filters unsupported configs before calling.
    assert tile is not None, "unsupported gemm_a16w16_gfx950 shape/config"

    a, b, bias = _create_inputs(m, n, k, dtype, layout, has_bias)
    # Dirty preallocated C, as the model passes out= a buffer it owns.
    out = torch.randn((m, n), dtype=c_dtype, device="cuda")
    ref = run_torch(a, b, bias, c_dtype)

    def _run_flydsl():
        # NT is the tuned_gemm / flydsl_hgemm path: A[M,K] @ W[N,K]^T.
        # nn/tn/tt are real kernel layouts the public wrapper does not expose,
        # so those rows drive gemm_a16w16 directly (same kernel, explicit layout).
        if layout == "nt":
            weight = b.t()  # [N, K], the flydsl_hgemm / tuned_gemm operand
            return flydsl_hgemm(
                a,
                weight,
                out=out,
                bias=bias,
                block_m=tile["block_m"],
                block_n=tile["block_n"],
                block_k=tile["block_k"],
                stages=tile["stages"],
                split_k=tile["split_k"],
                m_waves=tile["m_waves"],
                n_waves=tile["n_waves"],
                k_waves=tile["k_waves"],
                group_m=tile["group_m"],
                policy=policy,
                out_dtype=c_dtype,
            )
        return _a16w16().gemm_a16w16(
            a,
            b,
            out,
            bias=bias,
            user_kwargs=tile,
            layout=layout,
            out_dtype=c_dtype,
        )

    candidates = {"flydsl": _run_flydsl}
    # torch.mm is a competing kernel only when C shares the operand dtype;
    # fp32 C is a kernel epilogue, not a torch.mm out= path.
    if c_dtype == a.dtype:
        out_mm = torch.empty_like(out)

        def _run_torch_mm():
            if bias is None:
                torch.mm(a, b, out=out_mm)
            else:
                torch.addmm(bias, a, b, out=out_mm)
            return out_mm

        candidates["torch_mm"] = _run_torch_mm

    flops = 2 * m * n * k
    nbytes = (
        m * k * a.element_size() + n * k * b.element_size() + m * n * out.element_size()
    )
    if bias is not None:
        nbytes += n * bias.element_size()

    atol, rtol = _atol_rtol(k, split_k, tile["k_waves"], has_bias, dtype)
    ret = {
        "gfx": get_gfx(),
        "tile": (
            f"{tile['block_m']}x{tile['block_n']}x{tile['block_k']}"
            f"s{tile['stages']}"
        ),
        "waves": f"{tile['m_waves']}x{tile['n_waves']}x{tile['k_waves']}",
    }
    for name, fn in candidates.items():
        # Not `out`: `_run_flydsl` closes over the preallocated buffer.
        y, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            y.to(dtypes.fp32),
            rtol=rtol,
            atol=atol,
            msg=f"{name}: flydsl hgemm",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def _expect_error(fn, exc_type, match):
    try:
        fn()
    except exc_type as exc:
        if match not in str(exc):
            raise AssertionError(f"expected {match!r} in {exc!r}") from exc
        return
    raise AssertionError(f"expected {exc_type.__name__} matching {match!r}")


def check_padded_stride_and_storage_offset():
    """Non-contiguous padded views with a non-zero storage offset."""
    m = n = 64
    k = 256
    dtype = dtypes.bf16
    layout = "nt"
    col_offset = 8
    a_storage = torch.empty((m + 1, k + 32), dtype=dtype, device="cuda")
    a_storage.uniform_(-1, 1)
    a = a_storage[1:, col_offset : col_offset + k]
    b_storage = torch.empty((n + 1, k + 48), dtype=dtype, device="cuda")
    b_storage.uniform_(-1, 1)
    b = b_storage[1:, col_offset : col_offset + k].t()
    assert a.shape == (m, k) and b.shape == (k, n)
    assert not a.is_contiguous() and not b.is_contiguous()
    assert a.storage_offset() > 0 and b.storage_offset() > 0

    bias = torch.empty((n,), dtype=dtype, device="cuda").uniform_(-1, 1)
    out = torch.empty((m, n), dtype=dtype, device="cuda")
    tile = _pick_tile(m, n, k, dtype, None, layout, True, "ft", 1)
    assert tile is not None
    result = _a16w16().gemm_a16w16(
        a, b, out, bias=bias, user_kwargs=tile, layout=layout
    )
    ref = run_torch(a, b, bias, dtype)
    assert result.data_ptr() == out.data_ptr()
    checkAllclose(
        ref.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=2e-1,
        atol=3e-2,
        msg="padded stride: flydsl hgemm",
    )


def check_host_bias_contiguity_fallback():
    m = n = 64
    k = 256
    dtype = dtypes.bf16
    a = torch.randn((m, k), dtype=dtype, device="cuda")
    b = _empty_layout_matrix(k, n, dtype, transposed=True)
    b.normal_()
    bias = torch.randn((n * 2,), dtype=dtype, device="cuda")[::2]
    out = torch.empty((m, n), dtype=dtype, device="cuda")
    tile = _pick_tile(m, n, k, dtype, None, "nt", True, "ft", 1)
    assert tile is not None
    _a16w16().gemm_a16w16(a, b, out, bias=bias, user_kwargs=tile, layout="nt")
    ref = run_torch(a, b, bias, dtype)
    assert not bias.is_contiguous()
    checkAllclose(
        ref.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=2e-1,
        atol=3e-2,
        msg="non-contiguous bias: flydsl hgemm",
    )


def check_allocates_output_with_default_policy():
    m = n = 64
    k = 256
    dtype = dtypes.bf16
    a = torch.randn((m, k), dtype=dtype, device="cuda")
    b = _empty_layout_matrix(k, n, dtype, transposed=True)
    b.normal_()
    out = _a16w16().gemm_a16w16(a, b)
    ref = torch.mm(a, b)
    assert out.shape == (m, n) and out.dtype == dtype
    checkAllclose(
        ref.to(dtypes.fp32),
        out.to(dtypes.fp32),
        rtol=2e-1,
        atol=3e-2,
        msg="allocated out: flydsl hgemm",
    )


def check_split_k_stream_local_sync_buffers():
    m = n = 64
    k = 512
    dtype = dtypes.bf16
    layout = "nt"
    split_k = 2
    tile = _pick_tile(m, n, k, dtype, None, layout, True, "ft", split_k)
    assert tile is not None
    a, b, bias = _create_inputs(m, n, k, dtype, layout, True)
    out0 = torch.randn((m, n), dtype=dtype, device="cuda")
    out1 = torch.randn((m, n), dtype=dtype, device="cuda")
    stream0 = torch.cuda.Stream()
    stream1 = torch.cuda.Stream()
    gemm = _a16w16().gemm_a16w16
    gemm(a, b, out0, bias=bias, user_kwargs=tile, layout=layout)
    torch.cuda.synchronize()
    with torch.cuda.stream(stream0):
        gemm(
            a,
            b,
            out0,
            bias=bias,
            user_kwargs=tile,
            layout=layout,
            stream=stream0,
        )
    with torch.cuda.stream(stream1):
        gemm(
            a,
            b,
            out1,
            bias=bias,
            user_kwargs=tile,
            layout=layout,
            stream=stream1,
        )
    stream0.synchronize()
    stream1.synchronize()
    ref = run_torch(a, b, bias, dtype)
    checkAllclose(
        ref.to(dtypes.fp32),
        out0.to(dtypes.fp32),
        rtol=2e-1,
        atol=2e-1,
        msg="split-k stream0: flydsl hgemm",
    )
    checkAllclose(
        ref.to(dtypes.fp32),
        out1.to(dtypes.fp32),
        rtol=2e-1,
        atol=2e-1,
        msg="split-k stream1: flydsl hgemm",
    )


def check_rejects_unsupported_inputs():
    m = n = 64
    k = 256
    dtype = dtypes.bf16
    kern = _a16w16()
    gemm = kern.gemm_a16w16
    a = _empty_layout_matrix(m, k, dtype, transposed=False)
    b = _empty_layout_matrix(k, n, dtype, transposed=True)
    a.normal_()
    b.normal_()
    # Row-major A with a strided inner dim is not a valid NT/NN load.
    bad_a = torch.randn((m, k * 2), dtype=dtype, device="cuda")[:, ::2]
    _expect_error(
        lambda: gemm(bad_a, b, layout="nt"),
        ValueError,
        "A does not satisfy",
    )
    # Column-major A whose M is not a DMA vector multiple.
    a_vec = kern.dma_bytes // dtype.itemsize
    padded_m = 2 * a_vec
    a_tn = torch.randn((k, padded_m), dtype=dtype, device="cuda")[:, : a_vec + 1].t()
    b_tn = torch.randn((k, n), dtype=dtype, device="cuda")
    _expect_error(
        lambda: gemm(a_tn, b_tn, layout="tn"),
        ValueError,
        f"M divisible by {a_vec}",
    )
    tile = _tile(64, 64, 64, 2, 2, 2, 1, 0, 1, False)
    a_tail = torch.randn((m, 480), dtype=dtype, device="cuda")
    b_tail = _empty_layout_matrix(480, n, dtype, transposed=True)
    b_tail.normal_()
    _expect_error(
        lambda: gemm(a_tail, b_tail, user_kwargs=tile, layout="nt"),
        AssertionError,
        "K-tail is unsupported",
    )
    # k=64 / block_k=64 → one K tile; HTI needs at least two.
    a_hti = torch.randn((m, 64), dtype=dtype, device="cuda")
    b_hti = _empty_layout_matrix(64, n, dtype, transposed=True)
    b_hti.normal_()
    hti = _tile(64, 64, 64, 2, 2, 2, 1, 0, 1, True)
    _expect_error(
        lambda: gemm(a_hti, b_hti, user_kwargs=hti, layout="nt"),
        AssertionError,
        "HTI requires at least two",
    )


def check_split_k_slice_k_and_fp32_epilogue():
    """Configs the default argparse product does not cover (split-K, slice-K,
    four-wave, fp32 C, group_m swizzle). Same shape as dcp's check_modes: run
    the @benchmark fn for correctness, do not merge into the summary table.
    """
    dtype = dtypes.bf16
    for m, n, k, layout, has_bias, policy, split_k, out_dtype in [
        (32, 384, 7168, "nt", False, "ft", 8, None),
        (64, 64, 512, "nt", True, "hti", 2, None),
        (1280, 64, 4096, "nt", False, "hti", 8, None),
        (64, 128, 1024, "nt", True, "ft", 1, None),
        (800, 384, 7168, "nt", False, "ft", 1, None),
        (64, 64, 768, "nt", True, "ft", 1, dtypes.fp32),
        (64, 64, 768, "nt", False, "hti", 3, dtypes.fp32),
        (320, 4096, 256, "nt", False, "ft", 1, None),
        (320, 4032, 256, "nt", False, "ft", 1, None),
    ]:
        if (
            _pick_tile(m, n, k, dtype, out_dtype, layout, has_bias, policy, split_k)
            is None
        ):
            continue
        test_hgemm(m, n, k, dtype, layout, has_bias, policy, split_k, out_dtype)


def summarize(title, rows):
    df = pd.DataFrame(rows)
    if df.empty:
        return
    aiter.logger.info("%s summary (markdown):\n%s", title, df.to_markdown(index=False))


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("flydsl hgemm unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16,fp16}",
        help="""Data type.
        e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (32, 4096, 4096),
            (128, 4096, 4096),
            (512, 4096, 4096),
            (2048, 2048, 2048),
            (8192, 8192, 8192),
            (3, 5120, 2880),
            (32, 384, 7168),
            (8160, 8160, 8192),
        ],
        help="""Shape of mnk.
        e.g.:   -s 128,4096,4096
                --mnk 128,4096,4096""",
    )
    parser.add_argument(
        "-l",
        "--layout",
        type=str,
        choices=["nn", "nt", "tn", "tt"],
        nargs="*",
        default=["nt"],
        help="""A/B layout (N=row-major, T=transposed view).
        nt is the tuned_gemm / flydsl_hgemm model path (the default).
        nn/tn/tt exercise the kernel's other stride combos.
        e.g.: -l nt""",
    )
    parser.add_argument(
        "--bias",
        type=dtypes.str2bool,
        nargs="*",
        default=[False, True],
        help="""Fused bias.
        e.g.: --bias 0 1""",
    )
    parser.add_argument(
        "--policy",
        type=str,
        choices=["ft", "ht", "hti"],
        nargs="*",
        default=["ft", "hti"],
        help="""Kernel policy: ft (full-tile); ht/hti (half-tile interleaved).
        e.g.: --policy hti""",
    )
    parser.add_argument(
        "--split-k",
        type=int,
        nargs="*",
        default=[1],
        help="""Split-K partitions.
        e.g.: --split-k 1 8""",
    )
    parser.add_argument(
        "--out-dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[None],
        help="""Output dtype (none = same as input).
        e.g.: --out-dtype none,fp32""",
    )
    args = parser.parse_args()

    # Paths the argparse product does not cover. Same pattern as
    # test_flydsl_dcp_topk_merge.check_modes: raise on failure, stay out of
    # the summary table (every CLI flag remains a swept list).
    check_padded_stride_and_storage_offset()
    check_host_bias_contiguity_fallback()
    check_allocates_output_with_default_policy()
    check_split_k_stream_local_sync_buffers()
    check_rejects_unsupported_inputs()
    check_split_k_slice_k_and_fp32_epilogue()
    aiter.logger.info("flydsl hgemm: correctness checks passed")

    for dtype in args.dtype:
        rows = []
        for (
            layout,
            has_bias,
            policy,
            split_k,
            out_dtype,
            (m, n, k),
        ) in itertools.product(
            args.layout,
            args.bias,
            args.policy,
            args.split_k,
            args.out_dtype,
            args.mnk,
        ):
            tile = _pick_tile(
                m, n, k, dtype, out_dtype, layout, has_bias, policy, split_k
            )
            if tile is None:
                # Unsupported on this arch/config (e.g. HTI needs even K tiles,
                # column-major A needs M % DMA vec). Skip before @benchmark.
                continue
            rows.append(
                test_hgemm(m, n, k, dtype, layout, has_bias, policy, split_k, out_dtype)
            )
        summarize(f"flydsl_hgemm {dtype}", rows)


if __name__ == "__main__":
    main()
