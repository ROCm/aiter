# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Runtime correctness checks and perf sweep for the public BF16 decode GEMM.

``python3`` this file: it runs the validity checks first, then a
``@benchmark`` / ``run_perftest`` sweep ending in a markdown table. aiter
op_tests are plain scripts -- there is no pytest here, and the checks are
called from ``main()`` so a single invocation covers both.
"""

from __future__ import annotations

import argparse

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
import aiter.ops.flydsl.gemm_kernels as flydsl_gemm_kernels
from aiter import dtypes
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.flydsl.gemm_kernels import (
    ActivationSource,
    BlockMfmaDecodeConfig,
    ContractionMode,
    ReductionMode,
    WaveDecodeConfig,
    gemm_decode_bf16,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.tuned_gemm import (
    get_GEMM_A16W16_config,
    get_GEMM_A16W16_config_,
    is_flydsl_decode_config,
    tgemm,
)

ARCH = get_gfx_runtime()
SUPPORTED_ARCHS = ("gfx942", "gfx950")
ATOL = 0.125
RTOL = 0.01


def _inputs(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    row = torch.arange(m, device="cuda", dtype=torch.int32)[:, None]
    col = torch.arange(n, device="cuda", dtype=torch.int32)[:, None]
    red = torch.arange(k, device="cuda", dtype=torch.int32)[None, :]
    a = (((row * 7 + red * 3) % 23) - 11).to(torch.float32).div_(16).bfloat16()
    b = (((col * 5 + red * 7) % 29) - 14).to(torch.float32).div_(16).bfloat16()
    return a, b


def _bias(n: int) -> torch.Tensor:
    return (
        (((torch.arange(n, device="cuda", dtype=torch.int32) * 3) % 17) - 8)
        .to(torch.float32)
        .div_(16)
        .bfloat16()
    )


def run_torch(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    product = a.float() @ b.float().T
    if bias is not None:
        product = product + bias.float()
    return product.bfloat16()


def _output(m: int, n: int) -> torch.Tensor:
    return torch.full((m, n), torch.nan, device="cuda", dtype=torch.bfloat16)


def _assert_output(
    output: torch.Tensor,
    reference: torch.Tensor,
) -> None:
    assert torch.isfinite(output).all(), "decode GEMM produced non-finite values"
    # checkAllclose reports and returns the mismatching fraction; it never
    # raises. That is right for the sweep below, where the number becomes an
    # `err` column, but these are validity gates: without the assert a broken
    # kernel would log an error and still exit 0, leaving CI green.
    err = checkAllclose(
        reference.to(dtypes.fp32),
        output.to(dtypes.fp32),
        rtol=RTOL,
        atol=ATOL,
        msg="gemm_decode_bf16 output",
    )
    assert (
        err == 0
    ), f"gemm_decode_bf16 mismatched the reference on {err:.2%} of elements"


def _block_config(
    source: ActivationSource,
    *,
    columns_per_wave: int,
    persistent_n: bool = False,
) -> BlockMfmaDecodeConfig:
    return BlockMfmaDecodeConfig(
        waves_per_workgroup=4,
        columns_per_wave=columns_per_wave,
        activation_source=source,
        b_load_width=8,
        k_unroll=2,
        prefetch_stages=2 if ARCH == "gfx950" else 1,
        persistent_n=persistent_n,
        workgroups_per_cu=1,
        waves_per_eu=2 if ARCH == "gfx950" else 0,
    )


def _wave_config(m: int, k: int) -> WaveDecodeConfig:
    return WaveDecodeConfig(
        m_per_wave=m,
        n_per_wave=1,
        kvec=2,
        prefetch_depth=0,
        waves_per_eu=4,
        reduction=ReductionMode.DPP if k % 2 == 0 else ReductionMode.BPERMUTE,
        contraction=(
            ContractionMode.DOT2_BF16
            if ARCH == "gfx950"
            else ContractionMode.SCALAR_F32
        ),
    )


def _run_config(
    m: int,
    n: int,
    k: int,
    config: BlockMfmaDecodeConfig,
    *,
    with_bias: bool = False,
) -> None:
    a, b = _inputs(m, n, k)
    bias = _bias(n) if with_bias else None
    output = _output(m, n)
    returned = gemm_decode_bf16(
        a,
        b,
        output,
        config,
        bias=bias,
    )
    torch.cuda.synchronize()
    assert returned is output
    _assert_output(output, run_torch(a, b, bias))


def check_wave_no_bias() -> None:
    m, n, k = 1, 64, 128
    a, b = _inputs(m, n, k)
    output = _output(m, n)
    returned = gemm_decode_bf16(a, b, output, _wave_config(m, k))
    torch.cuda.synchronize()
    assert returned is output
    _assert_output(output, run_torch(a, b))


def check_wave_bias_and_odd_tails() -> None:
    m, n, k = 5, 65, 257
    a, b = _inputs(m, n, k)
    bias = _bias(n)
    output = _output(m, n)
    returned = gemm_decode_bf16(
        a,
        b,
        output,
        _wave_config(m, k),
        bias=bias,
    )
    torch.cuda.synchronize()
    assert returned is output
    _assert_output(output, run_torch(a, b, bias))


def check_block_mfma_global() -> None:
    _run_config(
        3,
        65,
        257,
        _block_config(ActivationSource.GLOBAL, columns_per_wave=2),
        with_bias=True,
    )


def check_block_mfma_lds_k_padding() -> None:
    _run_config(
        5,
        17,
        129,
        _block_config(ActivationSource.FULL_LDS, columns_per_wave=1),
    )


def check_block_mfma_persistent_n() -> None:
    _run_config(
        3,
        5001,
        257,
        _block_config(
            ActivationSource.FULL_LDS,
            columns_per_wave=1,
            persistent_n=True,
        ),
        with_bias=True,
    )


@benchmark()
def test_gemm_decode(m, n, k, dtype):
    """Time the deployed path against the previous default, rotating inputs.

    `tgemm.mm` is what callers run: it looks up the tuned row and launches the
    decode kernel it names. `F.linear` is what these shapes ran before the
    decode rows existed. Tensors are passed to `run_perftest` so it can rotate
    copies of them and the weight is not served from cache on every iteration.
    Each candidate is timed with eager launches and under CUDA-graph replay, and
    must match the reference with the same tolerance as the correctness cases.
    """
    a, b = _inputs(m, n, k)
    ref = run_torch(a, b).to(dtypes.fp32)
    row = get_GEMM_A16W16_config(m, n, k, False, str(dtype), str(dtype))
    candidates = {"aiter_tuned": tgemm.mm, "torch_linear": F.linear}
    flops = 2 * m * n * k
    nbytes = (m * k + n * k + m * n) * a.element_size()

    ret = {"gfx": ARCH, "aiter row": row.get("kernelName") or row.get("libtype")}
    for name, fn in candidates.items():
        out, us = run_perftest(fn, a, b, num_warmup=10)
        _, us_graph = run_perftest(fn, a, b, num_warmup=10, testGraph=True)
        err = checkAllclose(
            ref,
            out.to(dtypes.fp32),
            rtol=RTOL,
            atol=ATOL,
            msg=f"{name}: {m}x{n}x{k} {dtype}",
        )
        assert err == 0, f"{name} mismatched the reference on {err:.2%} of elements"
        ret[f"{name} us"] = us
        ret[f"{name} graph us"] = us_graph
        ret[f"{name} TB/s"] = nbytes / us / 1e6 if us else 0
    ret["TFLOPS (aiter graph)"] = (
        flops / ret["aiter_tuned graph us"] / 1e6 if ret["aiter_tuned graph us"] else 0
    )
    return ret


def _shipped_decode_rows() -> list[tuple]:
    """Every decode-kernel key in the merged tuned config for this card."""
    cu_num = get_cu_num()
    return sorted(
        key
        for key, row in get_GEMM_A16W16_config_().items()
        if is_flydsl_decode_config(row) and key[0] == ARCH and int(key[1]) == cu_num
    )


def check_tuned_rows_dispatch() -> None:
    """Every shipped decode row is selected by the dispatcher and runs.

    The cases above call `gemm_decode_bf16` with hand-built configs, so they
    never touch the tuned CSV. The dispatcher drops a decode row
    silently -- no log, no error -- when its kernelName fails to parse or names
    another shape, and falls through to a different backend. A typo in any
    shipped row would therefore pass every other check here. This one asks the
    dispatcher about each row, then drives a few through `tgemm.mm`.
    """
    rows = _shipped_decode_rows()
    if not rows:
        aiter.logger.warning(
            "no decode rows for %s/cu_num=%s; skipping", ARCH, get_cu_num()
        )
        return

    dropped = []
    for key in rows:
        _, _, m, n, k, bias, dtype, otype, scale_ab, bpreshuffle = key
        cfg = get_GEMM_A16W16_config(m, n, k, bias, dtype, otype, scale_ab, bpreshuffle)
        shipped = get_GEMM_A16W16_config_()[key]["kernelName"]
        if (
            cfg is None
            or not is_flydsl_decode_config(cfg)
            or cfg.get("kernelName") != shipped
        ):
            dropped.append((m, n, k, None if cfg is None else cfg.get("libtype")))
    assert not dropped, (
        f"{len(dropped)} of {len(rows)} decode rows are not selected by "
        f"the dispatcher (M, N, K, got): {dropped[:10]}"
    )

    # End to end through the public entry point: the smallest shape per M, so
    # the check stays cheap while covering every M the kernel supports.
    smallest = {}
    for key in rows:
        m, n, k = key[2], key[3], key[4]
        if key[5] is False and (m not in smallest or n * k < smallest[m][0]):
            smallest[m] = (n * k, n, k)
    calls = {"n": 0}
    real = flydsl_gemm_kernels.gemm_decode_bf16

    def counted(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    flydsl_gemm_kernels.gemm_decode_bf16 = counted
    try:
        for m, (_, n, k) in sorted(smallest.items()):
            a, b = _inputs(m, n, k)
            before = calls["n"]
            out = tgemm.mm(a, b)
            torch.cuda.synchronize()
            assert (
                calls["n"] == before + 1
            ), f"tgemm.mm did not route M={m} ({n},{k}) to the decode kernel"
            _assert_output(out, run_torch(a, b))
    finally:
        flydsl_gemm_kernels.gemm_decode_bf16 = real
    aiter.logger.info(
        "dispatcher selected all %d decode rows; ran %d through tgemm.mm",
        len(rows),
        len(smallest),
    )


def _strided_layouts(a: torch.Tensor) -> dict[str, torch.Tensor]:
    """The same values as ``a`` in layouts a caller can legally pass.

    A column slice of a wider tensor (for example a fused projection output)
    has a row stride above K; a size-1 row may carry any row stride.
    """
    m, k = a.shape
    wide = torch.zeros((m, 2 * k + 3), device=a.device, dtype=a.dtype)
    wide[:, 5 : 5 + k] = a
    layouts = {"column slice": wide[:, 5 : 5 + k]}
    if m == 1:
        layouts["transposed column vector"] = a.reshape(k, 1).T
    return layouts


def check_strided_activation() -> None:
    """A with a row stride above K gives the same C as packed A, bit for bit.

    Covers both policies directly and the tuned path through `tgemm.mm`, with
    no copy of A: the result must not depend on how A is laid out.
    """
    m_wave, m_block, k, n = 2, 3, 1536, 1536
    cases = [
        (m_wave, _wave_config(m_wave, k)),
        (1, _wave_config(1, k)),
        (m_block, _block_config(ActivationSource.GLOBAL, columns_per_wave=1)),
        (m_block, _block_config(ActivationSource.FULL_LDS, columns_per_wave=1)),
    ]
    for m, config in cases:
        a, b = _inputs(m, n, k)
        packed = gemm_decode_bf16(a, b, _output(m, n), config)
        _assert_output(packed, run_torch(a, b))
        for name, view in _strided_layouts(a).items():
            strided = gemm_decode_bf16(view, b, _output(m, n), config)
            assert torch.equal(
                strided, packed
            ), f"{type(config).__name__} M={m}: {name} A differs from packed A"

    rows = _shipped_decode_rows()
    for key in rows[:1] + rows[-1:]:
        m, n, k = key[2], key[3], key[4]
        a, b = _inputs(m, n, k)
        packed = tgemm.mm(a, b)
        for name, view in _strided_layouts(a).items():
            assert torch.equal(
                tgemm.mm(view, b), packed
            ), f"tgemm.mm M={m} ({n},{k}): {name} A differs from packed A"


CORRECTNESS_CASES = (
    check_wave_no_bias,
    check_wave_bias_and_odd_tails,
    check_block_mfma_global,
    check_block_mfma_lds_k_padding,
    check_block_mfma_persistent_n,
    check_tuned_rows_dispatch,
    check_strided_activation,
)


def _run_correctness_cases() -> None:
    """Run every validity case before the timed sweep.

    CI runs op_tests files as scripts, so the checks have to be reachable from
    `main()` -- there is no collector that would find them otherwise. They take
    no arguments, so calling them in order is the whole mechanism.
    """
    for case in CORRECTNESS_CASES:
        aiter.logger.info("running %s", case.__name__)
        case()


def _default_sweep_shapes() -> list[tuple[int, int, int]]:
    """For each M, the smallest and the largest shipped decode shape."""
    by_m = {}
    for key in _shipped_decode_rows():
        if key[5] is False:
            by_m.setdefault(key[2], set()).add((key[2], key[3], key[4]))
    shapes = []
    for m in sorted(by_m):
        ordered = sorted(by_m[m], key=lambda s: s[1] * s[2])
        shapes += sorted({ordered[0], ordered[-1]})
    return shapes


def main():
    if ARCH not in SUPPORTED_ARCHS:
        aiter.logger.warning("gemm_decode_bf16 unsupported on %s; skipping", ARCH)
        return

    torch.set_default_device("cuda")
    _run_correctness_cases()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16}",
        help="""Data type.
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=None,
        help="""Shape of mnk. Default: for each M, the smallest and the largest
    shape that has a tuned decode row on this GPU.
    e.g.:   -s 1,896,7168""",
    )
    args = parser.parse_args()

    mnk = args.mnk or _default_sweep_shapes()
    for dtype in args.dtype:
        df = []
        for m, n, k in mnk:
            if not 1 <= m <= 5:
                aiter.logger.warning(
                    "gemm_decode_bf16 supports M in [1, 5]; skipping m=%s", m
                )
                continue
            df.append(test_gemm_decode(m, n, k, dtype))
        if df:
            df = pd.DataFrame(df)
            aiter.logger.info(
                "gemm_decode_bf16 summary (markdown):\n%s",
                df.to_markdown(index=False),
            )


if __name__ == "__main__":
    main()
