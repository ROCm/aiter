# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

r"""Compare native paged FP8 entry points on gfx950 (16 Q heads / 1 KV head).

Backends share FP8 Q/K/V, shuffled physical pages, scalar descales and
preallocated BF16 outputs:
  flydsl: flydsl_flash_attn_paged_prefill_func with native CSR metadata.
  ck: the existing CK MHA batch-prefill path.

python3 -m op_tests.op_benchmarks.flydsl.bench_paged_fmha
python3 -m op_tests.op_benchmarks.flydsl.bench_paged_fmha \
        -b 1 -s 65536,131072 --page-sizes 64 --backends flydsl \
        --iters 100 --warmup 10 -o paged.csv
"""

import argparse
import hashlib
import itertools
import json
import subprocess
from functools import partial
from pathlib import Path

import pandas as pd
import torch

BACKENDS = ("flydsl", "ck")
FP8_RTOL = 0.02
FP8_ATOL = 0.02
# Bound each FP32 score tensor, independently of the full Q length. Softmax
# needs another tensor of this size; reconstructed K/V and the output are linear.
REFERENCE_SCORE_BYTES = 256 * 1024**2


def reference_chunked(case, score_bytes=REFERENCE_SCORE_BYTES):
    """Full-output FP32 oracle, gathering each request's logical K/V only once."""
    output = torch.empty_like(case.out, dtype=torch.float32)
    offset = 0
    for b, (qlen, klen) in enumerate(zip(case.qlens, case.klens)):
        if klen == 0:
            output[offset : offset + qlen].zero_()
            offset += qlen
            continue
        # Reject impossible score budgets before materializing the request's K/V.
        rows = min(256, score_bytes // (case.hq * klen * 4))
        if rows < 1:
            raise ValueError("one reference query row exceeds the score-memory budget")
        physical = case.table[b, : (klen + case.page - 1) // case.page].long()
        k, v = case.k[physical], case.v[physical]
        if case.layout == "vectorized":
            k, v = k.permute(0, 3, 1, 2, 4), v.permute(0, 2, 4, 1, 3)
        k = k.reshape(-1, case.hkv, case.d)[:klen].float() * case.ks
        v = v.reshape(-1, case.hkv, case.dv)[:klen].float() * case.vs
        k = k.repeat_interleave(case.hq // case.hkv, dim=1).transpose(0, 1)
        v = v.repeat_interleave(case.hq // case.hkv, dim=1).transpose(0, 1)
        key_positions = torch.arange(klen, device=case.q.device)
        for start in range(0, qlen, rows):
            end = min(start + rows, qlen)
            q = case.q[offset + start : offset + end].float() * case.qs
            scores = q.transpose(0, 1) @ k.transpose(-1, -2)
            scores.mul_(case.d**-0.5)
            query_positions = torch.arange(start, end, device=case.q.device)
            allowed = key_positions[None, :] <= query_positions[:, None] + klen - qlen
            scores.masked_fill_(~allowed, float("-inf"))
            probability = torch.softmax(scores, dim=-1).nan_to_num_(0)
            output[offset + start : offset + end] = (probability @ v).transpose(0, 1)
            del scores, probability
        offset += qlen
    return output


def mean_relative_error(reference, actual):
    """Normalized mean absolute error, matching the backward benchmark's mean_rel."""
    reference, actual = reference.float(), actual.float()
    return (
        (actual - reference).abs().mean() / reference.abs().mean().clamp_min(1e-12)
    ).item()


def ck_skip_reason(batch, query_length, kv_length, head_dim, value_dim, page_size):
    """Skip recorded CK failures and configurations not requalified after faults."""
    if page_size == 64:
        return "page size 64: no matching D128/D192 kernel in the qualified CK MHA configuration"
    if (torch.version.hip or "").startswith("7.2"):
        # Page 1024 passed the B=1/2/3 tail/prefill and B=1 long-context rechecks.
        if (head_dim, value_dim) == (192, 128) and page_size in (1, 16):
            return (
                f"ROCm 7.2 D192/V128, page size {page_size}: recorded non-finite output"
            )
        if head_dim == 192 and batch >= 4:
            return (
                "ROCm 7.2 D192 batch >= 4: not requalified after a recorded "
                "B4/page size 16 D192/V192 GPU memory fault"
            )
        if (head_dim, value_dim) == (192, 192) and (
            batch,
            page_size,
            query_length,
            kv_length,
        ) == (1, 1, 16384, 32768):
            return "ROCm 7.2 B1/page size 1 D192/V192 Q16K/KV32K: recorded non-finite output"
    return ""


def _pair(value):
    try:
        pair = tuple(int(x) for x in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "expected two comma-separated positive integers"
        ) from error
    if len(pair) != 2 or min(pair) <= 0:
        raise argparse.ArgumentTypeError(
            "expected two comma-separated positive integers"
        )
    return pair


def _causal_pairs(query_length, kv_length):
    active_rows = min(query_length, kv_length)
    return (
        active_rows * (active_rows + 1) // 2
        + max(kv_length - query_length, 0) * active_rows
    )


def benchmark_paged(
    batch,
    query_length,
    kv_length,
    head_dim,
    value_dim,
    page_size,
    dtype,
    backends,
    iters=100,
    warmup=10,
    check=True,
):
    import aiter
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.ops.flydsl import flydsl_flash_attn_paged_prefill_func
    from aiter.ops.mha import _mha_batch_prefill
    from aiter.test_common import checkAllclose, run_perftest
    from op_tests.test_flydsl_paged_fmha import csr_metadata, make_case

    layout = "linear3d" if page_size == 1 else "vectorized"
    ret = {"gfx": get_gfx(), "layout": layout}
    selected = list(backends)
    if "ck" in selected:
        reason = ck_skip_reason(
            batch, query_length, kv_length, head_dim, value_dim, page_size
        )
        if reason:
            selected.remove("ck")
            ret.update({"ck status": "SKIP", "ck reason": reason})
            aiter.logger.warning("CK skipped: %s", reason)
    if not selected:
        return ret
    case = make_case(
        page_size,
        layout,
        head_dim,
        value_dim,
        qlens=(query_length,) * batch,
        klens=(kv_length,) * batch,
        heads=(16, 1),
    )
    metadata = csr_metadata(case)
    indptr, indices = metadata["kv_indptr"], metadata["kv_page_indices"]
    last = metadata["kv_last_page_lens"] if page_size > 1 else None
    scale = head_dim**-0.5
    descales = {"q_descale": case.qs, "k_descale": case.ks, "v_descale": case.vs}
    candidates = {}
    if "flydsl" in selected:
        output = torch.empty_like(case.out)
        candidates["flydsl"] = partial(
            flydsl_flash_attn_paged_prefill_func,
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
            causal=True,
            out=output,
            **descales,
        )
    if "ck" in selected:
        ck_output = torch.empty_like(case.out)
        ck_key = case.k.unsqueeze(1) if page_size == 1 else case.k
        ck_value = case.v.unsqueeze(1) if page_size == 1 else case.v

        def ck():
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
    # Respect the requested order, including when checking the reverse order.
    candidates = {name: candidates[name] for name in selected}
    # Both entry points may import/compile on their first invocation.
    # Complete every backend's first-use work before warming or timing any one.
    for fn in candidates.values():
        fn()
    expected = reference_chunked(case) if check else None
    flops = (
        2 * batch * 16 * _causal_pairs(query_length, kv_length) * (head_dim + value_dim)
    )
    measurements = {
        name: run_perftest(fn, num_iters=iters, num_warmup=warmup, num_rotate_args=1)
        for name, fn in candidates.items()
    }
    # Each backend owns a distinct output buffer. Check them only after all
    # timings, so reductions/copies from accuracy checks cannot perturb a peer.
    for name, (actual, avg_us) in measurements.items():
        assert bool(actual.isfinite().all()), f"{name} output is not finite"
        err = float("nan")
        if check:
            err = checkAllclose(
                expected,
                actual.float(),
                rtol=FP8_RTOL,
                atol=FP8_ATOL,
                tol_err_ratio=0,
                msg=f"{name}: paged FP8",
            )
            assert err == 0, f"{name} numerical gate failed: {err}"
        us = float(avg_us)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} err"] = err
        ret[f"{name} status"] = "PASS" if check else "SKIP"
        ret[f"{name} accuracy"] = (
            mean_relative_error(expected, actual) if check else float("nan")
        )
        if not check:
            ret[f"{name} reason"] = "accuracy check disabled by --no-check"
    return ret


def runtime_metadata():
    import flydsl

    script = Path(__file__).resolve()
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(script.parents[3]), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    return {
        "gpu": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "flydsl_version": flydsl.__version__,
        "aiter_revision": revision,
        "harness_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "timing": "aiter-run-perftest-gpu-average",
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("-d", "--dtype", nargs="+", choices=["fp8"], default=["fp8"])
    parser.add_argument("-b", "--batch", nargs="+", type=int, default=[1])
    parser.add_argument(
        "-s",
        "--seqlens",
        nargs="+",
        type=_pair,
        metavar="Q,KV",
        default=[(257, 513), (4096, 8192)],
    )
    parser.add_argument(
        "--head-dims",
        nargs="+",
        type=_pair,
        metavar="QK,V",
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
        choices=BACKENDS,
        default=list(BACKENDS),
        help="entry points to compare, in measurement order",
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="profiled calls per backend (minimum 2)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="equal untimed calls per backend after all first-use compilation/setup",
    )
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="disable the full-output FP32 oracle; accuracy is unavailable and status is SKIP",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="write full results and runtime metadata to CSV",
    )
    args = parser.parse_args()
    if args.iters < 2:
        parser.error("iters must be at least 2 for run_perftest")
    if args.warmup < 1:
        parser.error("warmup must be positive")
    if any(batch <= 0 for batch in args.batch):
        parser.error("batch sizes must be positive")
    if any(dims not in ((128, 128), (192, 128), (192, 192)) for dims in args.head_dims):
        parser.error("supported head pairs are 128,128; 192,128; 192,192")
    # AITER import probes the GPU on some revisions. Keep --help and argument
    # validation usable without a GPU or an installed FlyDSL runtime.
    if not torch.cuda.is_available():
        print("native paged FP8 requires a gfx950 GPU; skipping")
        return 0
    import aiter
    from aiter.jit.utils.chip_info import get_gfx
    from aiter.test_common import benchmark

    if get_gfx() != "gfx950":
        aiter.logger.warning("native paged FP8 requires gfx950; skipping")
        return 0
    run_benchmark = benchmark()(benchmark_paged)
    backends = tuple(dict.fromkeys(args.backends))
    runtime = runtime_metadata()
    aiter.logger.info("paged FP8 runtime: %s", json.dumps(runtime))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for dtype, batch, (q, kv), (d, dv), page in itertools.product(
        args.dtype, args.batch, args.seqlens, args.head_dims, args.page_sizes
    ):
        result = run_benchmark(
            batch,
            q,
            kv,
            d,
            dv,
            page,
            dtype,
            backends,
            iters=args.iters,
            warmup=args.warmup,
            check=args.check,
        )
        for name in backends:
            row = {
                **runtime,
                "gfx": result["gfx"],
                "layout": result["layout"],
                "batch": batch,
                "query_length": q,
                "kv_length": kv,
                "head_dim": d,
                "value_dim": dv,
                "page_size": page,
                "query_heads": 16,
                "kv_heads": 1,
                "dtype": dtype,
                "backend": name,
                "iters": args.iters,
                "warmup": args.warmup,
                "avg_us": result.get(f"{name} us", float("nan")),
                "status": result[f"{name} status"],
                "accuracy": result.get(f"{name} accuracy", float("nan")),
                "skip_reason": result.get(f"{name} reason", ""),
            }
            for metric in ("TFLOPS", "err"):
                row[metric] = result.get(f"{name} {metric}", float("nan"))
            rows.append(row)
        # Preserve completed shapes if a subsequent backend fails or is interrupted.
        if args.output:
            pd.DataFrame(rows).to_csv(args.output, index=False)
    columns = [
        "batch",
        "query_length",
        "kv_length",
        "head_dim",
        "value_dim",
        "page_size",
        "backend",
        "avg_us",
        "TFLOPS",
        "accuracy",
        "status",
    ]
    aiter.logger.info(
        "paged FP8 summary (markdown):\n%s",
        pd.DataFrame(rows)[columns].to_markdown(
            index=False,
            floatfmt=tuple(
                ".3e" if column == "accuracy" else ".4f" for column in columns
            ),
        ),
    )
    if args.output:
        aiter.logger.info("results saved to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
