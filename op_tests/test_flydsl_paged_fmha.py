# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""Public paged-FP8 operator correctness and performance sweep on gfx950."""

import argparse
import itertools
from functools import partial

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest


def _pair(value):
    try:
        pair = tuple(int(x) for x in value.replace(":", ",").split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected two positive integers") from error
    if len(pair) != 2 or min(pair) <= 0:
        raise argparse.ArgumentTypeError("expected two positive integers")
    return pair


def _causal_pairs(query_length, kv_length):
    active_rows = min(query_length, kv_length)
    return (
        active_rows * (active_rows + 1) // 2
        + max(kv_length - query_length, 0) * active_rows
    )


@benchmark()
def benchmark_paged(
    batch, query_length, kv_length, head_dim, value_dim, page_size, dtype, backends
):
    from aiter.ops.flydsl import flydsl_flash_attn_paged_fp8_func
    from aiter.ops.mha import _mha_batch_prefill, mha_batch_prefill_func
    from op_tests.flydsl_tests.test_flydsl_paged_fmha import (
        csr_metadata,
        make_case,
        reference,
    )

    layout = "linear3d" if page_size == 1 else "vectorized"
    case = make_case(
        page_size,
        layout,
        head_dim,
        value_dim,
        qlens=(query_length,) * batch,
        klens=(kv_length,) * batch,
        heads=(16, 1),
    )
    case.maxkv = kv_length
    metadata = csr_metadata(case)
    indptr, indices = metadata["kv_indptr"], metadata["kv_page_indices"]
    last = metadata["kv_last_page_lens"] if page_size > 1 else None
    scale = head_dim**-0.5
    expected = reference(case)
    descales = {"q_descale": case.qs, "k_descale": case.ks, "v_descale": case.vs}
    candidates = {}
    if "flydsl" in backends:
        output = torch.empty_like(case.out)
        candidates["flydsl"] = partial(
            flydsl_flash_attn_paged_fp8_func,
            case.q,
            case.k,
            case.v,
            case.cuq,
            query_length,
            kv_length,
            kv_indptr=indptr,
            kv_page_indices=indices,
            kv_last_page_lens=last,
            softmax_scale=scale,
            out=output,
            **descales,
        )
    if "aiter" in backends:
        output = torch.empty_like(case.out)
        candidates["aiter"] = partial(
            mha_batch_prefill_func,
            case.q,
            case.k,
            case.v,
            case.cuq,
            indptr,
            indices,
            query_length,
            kv_length,
            causal=True,
            softmax_scale=scale,
            kv_last_page_lens=last,
            out=output,
            **descales,
        )
    if "ck" in backends:
        known_rocm72_fault = (torch.version.hip or "").startswith("7.2") and (
            (page_size == 16 and (head_dim, value_dim) == (192, 128))
            or (head_dim == 192 and batch >= 4)
        )
        if page_size == 64:
            aiter.logger.warning("CK page 64 has no matching D128/D192 kernel")
        elif known_rocm72_fault:
            aiter.logger.warning(
                "CK case excluded: recorded ROCm 7.2/gfx950 D192 fault or unqualified batch"
            )
        else:
            ck_output = torch.empty_like(case.out)
            ck_key = case.k.unsqueeze(1) if page_size == 1 else case.k
            ck_value = case.v.unsqueeze(1) if page_size == 1 else case.v

            def ck():
                # Bypass the new high-level dispatch to measure the CK backend.
                return _mha_batch_prefill(
                    case.q,
                    ck_key,
                    ck_value,
                    case.cuq,
                    indptr,
                    indices,
                    query_length,
                    kv_length,
                    0.0,
                    scale,
                    True,
                    kv_last_page_lens=last,
                    out=ck_output,
                    **descales,
                )[0]

            candidates["ck"] = ck
    flops = (
        2 * batch * 16 * _causal_pairs(query_length, kv_length) * (head_dim + value_dim)
    )
    # Native cache/Q/O/metadata footprint, not a measured HBM-traffic counter.
    nbytes = sum(
        t.numel() * t.element_size()
        for t in (case.q, case.k, case.v, case.out, case.cuq, indptr, indices)
    )
    if last is not None:
        nbytes += last.numel() * last.element_size()
    ret = {"gfx": get_gfx(), "layout": layout}
    for name, fn in candidates.items():
        actual, us = run_perftest(
            fn, num_iters=30, num_warmup=10, num_rotate_args=1, use_cuda_event=True
        )
        assert bool(actual.isfinite().all()), f"{name} output is not finite"
        err = checkAllclose(
            expected.float(),
            actual.float(),
            rtol=0.02,
            atol=0.02,
            tol_err_ratio=0,
            msg=f"{name}: paged FP8",
        )
        assert err == 0, f"{name} numerical gate failed: {err}"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-d", "--dtype", nargs="+", choices=["fp8"], default=["fp8"])
    parser.add_argument("-b", "--batch", nargs="+", type=int, default=[1])
    parser.add_argument(
        "-s", "--seqlens", nargs="+", type=_pair, default=[(257, 513), (4096, 8192)]
    )
    parser.add_argument(
        "--head-dims",
        nargs="+",
        type=_pair,
        default=[(128, 128), (192, 128), (192, 192)],
    )
    parser.add_argument(
        "--page-sizes",
        nargs="+",
        type=int,
        choices=[1, 16, 64, 1024],
        default=[1, 16, 64, 1024],
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["flydsl", "aiter", "ck"],
        default=["flydsl", "aiter"],
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or get_gfx() != "gfx950":
        aiter.logger.warning("native paged FP8 requires gfx950; skipping")
        return
    if any(batch <= 0 for batch in args.batch):
        parser.error("batch sizes must be positive")
    if any(dims not in ((128, 128), (192, 128), (192, 192)) for dims in args.head_dims):
        parser.error("supported head pairs are 128,128; 192,128; 192,192")
    rows = []
    for dtype, batch, (q, kv), (d, dv), page in itertools.product(
        args.dtype, args.batch, args.seqlens, args.head_dims, args.page_sizes
    ):
        rows.append(
            benchmark_paged(batch, q, kv, d, dv, page, dtype, tuple(args.backends))
        )
    aiter.logger.info(
        "paged FP8 summary (markdown):\n%s", pd.DataFrame(rows).to_markdown(index=False)
    )


if __name__ == "__main__":
    main()
