# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FP4 MQA shared-pool addressing regression tests (gfx950).

The original constant 8192-logit regression at byte offset 4,608,000,000
(4.608 GB, about 4.292 GiB) is retained. Random tests additionally compare
shared-pool pages to same-byte contiguous pages AND an independent dequantized
FP4 oracle. Physical pages straddle 2 GiB and 4 GiB, with shuffled mappings,
padded block-table rows, nonzero storage offsets, and tail/empty windows.
Pools are allocated sequentially; the largest backing store is about 4.61 GB.

Usage:
    python op_tests/test_flydsl_pa_mqa_logits_fp4_strided_pages.py --graph
    python op_tests/test_flydsl_pa_mqa_logits_fp4_strided_pages.py --case small \
        --graph --sensitivity --benchmark --json-output results.json

Run the same file with separate before/after AITER source mounts and distinct
--label values. Benchmark numbers time the AITER public API, not vLLM gather.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
from pathlib import Path

import torch

from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
    flydsl_pa_mqa_logits_fp4,
)
from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
    flydsl_pa_mqa_logits_fp4_prefill,
)
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.test_common import run_perftest

try:
    from test_flydsl_pa_mqa_logits_fp4_prefill import (
        fp4_dequant_e2m1_with_e8m0,
        fp4_quant_e2m1_with_e8m0,
        indexer_k_fp4_paged_preshuffle,
        quant_q_fp4_preshuffle,
        ref_prefill_logits,
    )
except ModuleNotFoundError as exc:
    if exc.name != "test_flydsl_pa_mqa_logits_fp4_prefill":
        raise
    from op_tests.test_flydsl_pa_mqa_logits_fp4_prefill import (
        fp4_dequant_e2m1_with_e8m0,
        fp4_quant_e2m1_with_e8m0,
        indexer_k_fp4_paged_preshuffle,
        quant_q_fp4_preshuffle,
        ref_prefill_logits,
    )

dev = "cuda"
HEAD_DIM = 128
KV_BLOCK_SIZE = 64
PAGE_BYTES = KV_BLOCK_SIZE * 68
BLOCK_K = 256
HEADS = 64
EXPECTED_LOGIT = 8192.0
GUARD = 123456.0
SEED = 5518
# At each power-of-two boundary, used pages are on both sides of the boundary.
CASES = {
    "small": (37, 8192),
    "2gib": ((1 << 31) // 8192 - 4, 8192),
    "4gib": ((1 << 32) // 8192 - 4, 8192),
    "original": (20000, 230400),
}


def vllm_fp4_cache_views(
    kv_cache: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split vLLM (num_blocks, 64, 68) layout into AITER preshuffle views."""
    assert kv_cache.ndim == 3 and kv_cache.dtype == torch.uint8
    num_blocks, block_size, row_bytes = kv_cache.shape
    assert block_size == KV_BLOCK_SIZE and row_bytes == 68
    assert kv_cache.stride(2) == 1 and kv_cache.stride(1) == 68
    page_stride = kv_cache.stride(0)
    assert page_stride >= block_size * row_bytes
    values = torch.as_strided(
        kv_cache,
        (num_blocks, 1, 4, block_size, 16),
        (page_stride, block_size * 64, block_size * 16, 16, 1),
    )
    scales = torch.as_strided(
        kv_cache,
        (num_blocks, 1, 4, block_size),
        (page_stride, block_size * 4, block_size, 1),
        storage_offset=kv_cache.storage_offset() + block_size * 64,
    )
    return values, scales


def padded_table(entries: torch.Tensor) -> torch.Tensor:
    """Poison padding and add both a row stride and a nonzero storage offset."""
    rows, columns = entries.shape
    storage = torch.full(
        (rows + 1, columns + 11), 2147483647, dtype=torch.int32, device=dev
    )
    table = storage[1:, 3 : 3 + columns]
    table.copy_(entries)
    assert table.stride(0) > columns and table.storage_offset() > 0
    return table


def allocate_pool(page_base, page_stride, dense_cache):
    """Copy only used page bytes, never initialize the multi-GB unused pool."""
    pages = dense_cache.shape[0]
    layer_offset = 3 * PAGE_BYTES + 128
    pool_pages = page_base + pages
    storage_bytes = layer_offset + (pool_pages - 1) * page_stride + PAGE_BYTES
    storage = torch.empty(storage_bytes, dtype=torch.uint8, device=dev)
    cache = storage.as_strided(
        (pool_pages, KV_BLOCK_SIZE, 68), (page_stride, 68, 1), layer_offset
    )
    # Page zero is a valid masked-load target, and a deterministic wrong-address
    # target; the correct result never depends on uninitialized page-zero bytes.
    cache[0].zero_()
    cache[page_base : page_base + pages].copy_(dense_cache)
    return cache, storage_bytes


def make_inputs(rows, context, constant=False):
    """Three requests, distinct per-block scales and signed random FP4 data."""
    generator = torch.Generator(device=dev).manual_seed(SEED)
    requests = 3
    # Kernel loads page metadata in four-page chunks, including masked tails.
    pages_per_request = (context + BLOCK_K - 1) // BLOCK_K * 4
    tokens = pages_per_request * KV_BLOCK_SIZE
    pages = requests * pages_per_request
    logical_pages = torch.randperm(pages, generator=generator, device=dev).reshape(
        requests, pages_per_request
    )
    table = padded_table(logical_pages.to(torch.int32))
    dense_cache = torch.zeros((pages, KV_BLOCK_SIZE, 68), dtype=torch.uint8, device=dev)
    dense_values, dense_scales = vllm_fp4_cache_views(dense_cache)
    if constant:
        q_source = torch.ones((rows + 7, HEADS, HEAD_DIM), device=dev)
        k_source = torch.ones((requests, tokens, HEAD_DIM), device=dev)
        weights = torch.ones((rows, HEADS), dtype=torch.bfloat16, device=dev)
    else:

        def random_scaled(shape):
            values = torch.randn(shape, generator=generator, device=dev)
            exponents = torch.randint(
                -2,
                3,
                (*shape[:-1], HEAD_DIM // 32, 1),
                generator=generator,
                device=dev,
            )
            return (values.reshape(*shape[:-1], 4, 32) * (2.0**exponents)).reshape(
                shape
            )

        q_source = random_scaled((rows + 7, HEADS, HEAD_DIM))
        k_source = random_scaled((requests, tokens, HEAD_DIM))
        weights = (
            0.25 + torch.rand((rows, HEADS), generator=generator, device=dev)
        ).to(torch.bfloat16)
    q_source = q_source.to(torch.bfloat16)
    k_source = k_source.to(torch.bfloat16)
    q_storage, qs_storage = quant_q_fp4_preshuffle(q_source)
    q, qs = q_storage[7:], qs_storage[7:]
    _, q_e8 = fp4_quant_e2m1_with_e8m0(q_source[7:])
    q_dq = fp4_dequant_e2m1_with_e8m0(q, q_e8)
    k_packed, k_e8 = fp4_quant_e2m1_with_e8m0(k_source)
    k_dq = fp4_dequant_e2m1_with_e8m0(k_packed, k_e8)
    slots = (
        logical_pages[:, :, None] * KV_BLOCK_SIZE
        + torch.arange(KV_BLOCK_SIZE, device=dev)
    ).reshape(-1)
    indexer_k_fp4_paged_preshuffle(
        k_source.reshape(-1, HEAD_DIM),
        slots,
        dense_values,
        dense_scales,
        KV_BLOCK_SIZE,
    )
    if constant:
        # Keep the original literal 0x22 / E8M0=127 byte pattern, not merely
        # another quantization of the same real-valued input.
        dense_values.fill_(0x22)
        dense_scales.fill_(127)
        q.fill_(0x22)
        qs.fill_(127)
    else:
        assert q_e8.unique().numel() > 1 and k_e8.unique().numel() > 1
    mapping = torch.arange(rows, device=dev, dtype=torch.int32) % requests
    return dense_cache, table, q, qs, weights, q_dq, k_dq, mapping


def guarded_output(rows, context):
    backing = torch.full((rows + 2, context + 19), GUARD, device=dev)
    out = backing[1:-1, 7 : 7 + context]
    return backing, out


def verify_output(out, backing, reference, *, exact=False):
    valid = torch.isfinite(reference)
    assert bool(torch.isneginf(out[~valid]).all()), "invalid logits must be -inf"
    torch.testing.assert_close(
        out[valid],
        reference[valid],
        rtol=0 if exact else 2e-4,
        atol=0 if exact else 2e-3,
    )
    assert bool((backing[0] == GUARD).all())
    assert bool((backing[-1] == GUARD).all())
    assert bool((backing[1:-1, :7] == GUARD).all())
    assert bool((backing[1:-1, out.shape[1] + 7 :] == GUARD).all())


def run_case(
    name,
    *,
    graph=False,
    sensitivity=False,
    benchmark=False,
    rows=12,
    context=333,
    iters=50,
    warmup=10,
    constant=False,
    paths=("prefill", "decode"),
    layouts=("dense", "strided"),
):
    page_base, page_stride = CASES[name]
    dense_cache, dense_table, q, qs, weights, q_dq, k_dq, mapping = make_inputs(
        rows, context, constant
    )
    dense_values, dense_scales = vllm_fp4_cache_views(dense_cache)
    # Stock kernels expect separately contiguous K and scale arrays. Preserve
    # every payload byte while removing only the shared-pool storage layout.
    dense_values = dense_values.contiguous()
    dense_scales = dense_scales.contiguous()
    dense_table = dense_table.contiguous()
    cache = values = scales = strided_table = None
    storage_bytes = 0
    if "strided" in layouts:
        cache, storage_bytes = allocate_pool(page_base, page_stride, dense_cache)
        strided_table = padded_table(dense_table + page_base)
        values, scales = vllm_fp4_cache_views(cache)
        assert torch.equal(
            values[page_base : page_base + len(dense_cache)], dense_values
        )
        assert torch.equal(
            scales[page_base : page_base + len(dense_cache)], dense_scales
        )
    records = []
    for path in paths:
        ends_list = [context, context - 1, min(65, context), 0]
        ends = torch.tensor(
            [ends_list[r % 4] for r in range(rows)], dtype=torch.int32, device=dev
        )
        starts = torch.zeros_like(ends)
        if path == "prefill":
            starts[1::4] = min(17, context - 1)
            starts[2::4] = ends[2::4]  # nonzero empty windows
            table_dense, table_strided = dense_table, strided_table
        else:
            table_dense = dense_table[mapping.long()].contiguous()
            table_strided = (
                padded_table(strided_table[mapping.long()])
                if strided_table is not None
                else None
            )
        reference = ref_prefill_logits(
            q_dq, k_dq, weights, mapping, starts, ends, context
        )
        if constant:
            assert bool((reference[torch.isfinite(reference)] == EXPECTED_LOGIT).all())
        outputs = {}
        for layout, kv, ks, table in (
            ("dense", dense_values, dense_scales, table_dense),
            ("strided", values, scales, table_strided),
        ):
            if layout not in layouts:
                continue
            backing, out = guarded_output(rows, context)

            def launch(
                path=path,
                kv=kv,
                ks=ks,
                table=table,
                starts=starts,
                ends=ends,
                out=out,
            ):
                if path == "prefill":
                    return flydsl_pa_mqa_logits_fp4_prefill(
                        q,
                        qs,
                        kv,
                        ks,
                        table,
                        weights,
                        mapping,
                        starts,
                        ends,
                        context,
                        parallel_unit_num=max(512, rows),
                        out=out,
                    )
                return flydsl_pa_mqa_logits_fp4(
                    q[:, None],
                    qs[:, None],
                    kv,
                    ks,
                    table,
                    weights,
                    ends,
                    context,
                    kv_block_size=KV_BLOCK_SIZE,
                    block_k=BLOCK_K,
                    parallel_unit_num=max(512, rows),
                    out=out,
                )

            launch()
            torch.cuda.synchronize()
            verify_output(out, backing, reference, exact=constant)
            outputs[layout] = out.clone()
            modes = [("eager", launch)]
            if graph:
                graph_obj = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph_obj):
                    launch()
                # Poison the entire logical output between replays, so a no-op
                # or stale captured output cannot pass either masks or values.
                for _ in range(2):
                    out.fill_(GUARD)
                    graph_obj.replay()
                    torch.cuda.synchronize()
                    verify_output(out, backing, reference, exact=constant)
                modes.append(("graph_replay", graph_obj.replay))
            if sensitivity and not constant and layout == "strided":
                valid = torch.isfinite(reference)
                saved_table = table.clone()
                table.copy_(table.roll(1, dims=1))
                try:
                    launch()
                    torch.cuda.synchronize()
                    assert not torch.allclose(
                        out[valid], reference[valid], rtol=2e-4, atol=2e-3
                    ), "page-map mutation was undetected"
                finally:
                    table.copy_(saved_table)
                used_scales = scales[page_base : page_base + len(dense_cache)]
                saved_scales = used_scales.clone()
                used_scales.copy_(used_scales.roll(1, dims=-1))
                try:
                    launch()
                    torch.cuda.synchronize()
                    assert not torch.allclose(
                        out[valid], reference[valid], rtol=2e-4, atol=2e-3
                    ), "scale-permutation mutation was undetected"
                finally:
                    used_scales.copy_(saved_scales)
                launch()
                torch.cuda.synchronize()
                verify_output(out, backing, reference)
            for mode, call in modes:
                result = {
                    "case": name,
                    "pattern": "constant" if constant else "random",
                    "path": path,
                    "layout": layout,
                    "mode": mode,
                    "rows": rows,
                    "requests": 3,
                    "context": context,
                    "heads": HEADS,
                    "head_dim": HEAD_DIM,
                    "kv_block_size": KV_BLOCK_SIZE,
                    "block_k": BLOCK_K,
                    "q_shape": list(q.shape),
                    "kv_shape": list(kv.shape),
                    "page_stride_bytes": kv.stride(0),
                    "scale_page_stride_bytes": ks.stride(0),
                    "block_table_shape": list(table.shape),
                    "block_table_stride": table.stride(0),
                    "block_table_storage_offset": table.storage_offset(),
                    "q_storage_offset_bytes": q.storage_offset(),
                    "q_scale_storage_offset_bytes": qs.storage_offset(),
                    "output_stride": out.stride(0),
                    "output_storage_offset": out.storage_offset(),
                    "cache_storage_offset_bytes": (
                        cache.storage_offset() if layout == "strided" else 0
                    ),
                    "scale_storage_offset_bytes": ks.storage_offset(),
                    "physical_page_offset_min_bytes": (
                        page_base * page_stride if layout == "strided" else 0
                    ),
                    "physical_page_offset_max_bytes": (
                        (page_base + len(dense_cache) - 1) * page_stride
                        if layout == "strided"
                        else (len(dense_cache) - 1) * kv.stride(0)
                    ),
                    "pool_allocation_bytes": (
                        storage_bytes if layout == "strided" else dense_cache.numel()
                    ),
                    "correctness": "passed",
                    "sensitivity_checked": sensitivity
                    and not constant
                    and layout == "strided",
                }
                if benchmark:
                    _, latency = run_perftest(
                        call,
                        num_iters=iters,
                        num_warmup=warmup,
                        num_rotate_args=1,
                        use_cuda_event=True,
                    )
                    result["latency_us"] = float(latency)
                    verify_output(out, backing, reference, exact=constant)
                records.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
            # Release graphs before moving to another layout/large allocation.
            if graph:
                del graph_obj
            del modes, call
        if len(outputs) == 2:
            torch.testing.assert_close(
                outputs["strided"], outputs["dense"], rtol=0, atol=0
            )
    return records


def test_small_stride_prefill_decode(graph: bool = False) -> None:
    run_case("small", graph=graph, sensitivity=True)


def test_large_offset_prefill_decode(graph: bool = False) -> None:
    """Original 4.608 GB (4.292 GiB) regression plus power-of-two boundaries."""
    for name in ("2gib", "4gib", "original"):
        run_case(name, graph=graph)
        gc.collect()
        torch.cuda.empty_cache()
    run_case("original", graph=graph, rows=512, context=2048, constant=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--graph",
        action="store_true",
        help="Capture and replay BOTH prefill and decode",
    )
    parser.add_argument("--case", choices=("all", *CASES), default="all")
    parser.add_argument("--path", choices=("both", "prefill", "decode"), default="both")
    parser.add_argument(
        "--layout",
        choices=("both", "dense", "strided"),
        default="both",
        help="Dense is stock-compatible and avoids shared-pool allocation; both compares identical payload bytes",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Require wrong page-map and scale permutations to disagree",
    )
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--rows", type=int, default=12)
    parser.add_argument("--context", type=int, default=333)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--label", default="unlabelled")
    parser.add_argument(
        "--source-revision",
        help="Host-verified SHA when mount git metadata is unavailable",
    )
    parser.add_argument(
        "--runtime-label",
        help="Container image/build provenance supplied by the runner",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.rows < 4 or args.context < 65 or args.iters < 1 or args.warmup < 1:
        parser.error("rows >= 4, context >= 65, iters >= 1, warmup >= 1 required")
    if get_arch() != "gfx950":
        raise SystemExit(f"gfx950 required; current architecture: {get_arch()}")
    source = Path(
        __import__(flydsl_pa_mqa_logits_fp4.__module__, fromlist=["__file__"]).__file__
    ).resolve()
    revision = subprocess.run(
        ["git", "-C", str(source.parent), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "metadata": {
            "label": args.label,
            "torch": torch.__version__,
            "hip": torch.version.hip,
            "python": platform.python_version(),
            "gpu": torch.cuda.get_device_name(),
            "arch": get_arch(),
            "source": str(source),
            "revision": args.source_revision or revision.stdout.strip() or None,
            "revision_origin": (
                "runner"
                if args.source_revision
                else ("git" if revision.returncode == 0 else "unavailable")
            ),
            "runtime_label": args.runtime_label,
            "seed": SEED,
            "iterations": args.iters,
            "warmup": args.warmup,
            "timer": "aiter.test_common.run_perftest CUDA events; fixed inputs, no rotation",
            "timing_scope": "public API including internal schedule and output reset; graph mode is one captured API replay",
            "rtol": 2e-4,
            "atol": 2e-3,
        },
        "results": [],
    }
    for name in CASES if args.case == "all" else (args.case,):
        report["results"].extend(
            run_case(
                name,
                graph=args.graph,
                sensitivity=args.sensitivity,
                benchmark=args.benchmark,
                rows=args.rows,
                context=args.context,
                iters=args.iters,
                warmup=args.warmup,
                paths=("prefill", "decode") if args.path == "both" else (args.path,),
                layouts=(
                    ("dense", "strided") if args.layout == "both" else (args.layout,)
                ),
            )
        )
        gc.collect()
        torch.cuda.empty_cache()
    if args.case == "all":
        report["results"].extend(
            run_case(
                "original",
                graph=args.graph,
                rows=512,
                context=2048,
                constant=True,
                paths=("prefill", "decode") if args.path == "both" else (args.path,),
                layouts=(
                    ("dense", "strided") if args.layout == "both" else (args.layout,)
                ),
            )
        )
    if args.json_output:
        args.json_output.write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: all requested FP4 shared-pool comparisons and guards")


if __name__ == "__main__":
    main()
