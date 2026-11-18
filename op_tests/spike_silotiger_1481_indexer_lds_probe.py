# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""SILOTIGER-1481 phase-1 candidate-state probe on the 16-row indexer tile.

The zero-byte copy must match the unmodified prefill scores on ``p32k`` and
stay within 5% of that session's unmodified ``T_score``. Candidate arrays of
16, 32 and 64 KiB are timed at ``p128k`` and ``p256k`` on the Phase 0
168-page table, and again at ``p128k`` on the trimmed 84-page prefix. The
trimmed prefix also times Arm A's top-k plus expand, which is the budget a
64 KiB array has to fit under.

Host compile, with every GPU hidden, before the timed gpuq job::

    HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \\
    GPU_ARCHS=gfx950 AITER_AOT_REPLAY=1 \\
    /workspaces/aiter/flydsl_venv/bin/python \\
        op_tests/spike_silotiger_1481_indexer_lds_probe.py --host-compile

Timed run, from the repo root, after that compile::

    export GPUQ_USER=<owner>
    gpuq --num-gpus=1 /workspaces/aiter/flydsl_venv/bin/python \\
        op_tests/spike_silotiger_1481_indexer_lds_probe.py
"""

from __future__ import annotations

import argparse
import functools
import os
import statistics
import sys
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(__file__))

import bench_qsa_indexer_select as bench

from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import qsa_paged as paged
from aiter.ops.flydsl.kernels.qsa.indexer_select import (
    build_qsa_indexer_prefill_scores_module,
)

# The production builder is not cached. Rebuilding it on every chunk puts
# the module construction inside T_score. The probe builder is cached.
_production_scores = functools.lru_cache(maxsize=16)(
    build_qsa_indexer_prefill_scores_module
)
from aiter.ops.flydsl.kernels.qsa.indexer_select_lds_probe import (
    build_qsa_indexer_prefill_scores_lds_probe,
)
from aiter.ops.qsa import qsa_expand_block_indices

WARMUP = 5
ITERS = 10
REPEATS = 2
HEADS = 4
PAGE = bench.PAGE
TOKENS_PER_PAGE = PAGE * bench.COMPRESS
CANDIDATES = (0, 16 * 1024, 32 * 1024, 64 * 1024)
# Pre-fill for the match check only. A score past a row's length is never
# written, so a value that survives the launch is an unwritten entry.
SENTINEL = 12345.0
# Compute units per arch for the host compile. gfx942 is MI300X (304 CUs).
CU_NUM = {"gfx950": 256, "gfx942": 304}


def _trim_pages(context: int) -> int:
    return (context + TOKENS_PER_PAGE - 1) // TOKENS_PER_PAGE


def _built(name: str, index_k: torch.Tensor):
    case = next(case for case in bench._cases() if case.name == name)
    built = bench._build(case, index_k)
    live = 0
    for req in case.requests:
        live += (req.context // bench.COMPRESS + PAGE - 1) // PAGE
    if index_k[:live].numel():
        index_k[:live].copy_(
            torch.randn(
                live, *index_k.shape[1:], dtype=index_k.dtype, device=index_k.device
            )
        )
    return built


def _table_for(built, pages: int | None):
    table = built[2]
    if pages is None:
        return table
    if pages > table.shape[1]:
        raise ValueError(f"trim {pages} exceeds the padded table {table.shape[1]}")
    return table[:, :pages]


def _columns(table, index_k) -> int:
    return int(table.shape[1]) * int(index_k.shape[1])


def _launch_shape(q, n_columns: int, batch_rows: int, n_req: int):
    n_heads = int(q.shape[1])
    rows = int(q.shape[0])
    fold, tiles = paged._indexer_rows_scorer_layout(
        n_req, batch_rows, n_heads, n_columns, row_bounded=False
    )
    if rows != batch_rows and not fold:
        tiles = paged._indexer_rows_scorer_layout(
            n_req, rows, n_heads, n_columns, row_bounded=False
        )[1]
    tile_rows = 16 // n_heads if fold else 16
    col_tiles = paged.ceildiv(n_columns, paged._ROWS_BLOCK_N)
    row_tiles = paged.ceildiv(rows, tile_rows)
    tile_groups = paged.ceildiv(col_tiles, tiles)
    return fold, tiles, row_tiles, tile_groups


def _workgroups(q, n_columns: int, batch_rows: int, n_req: int) -> int:
    _fold, _tiles, row_tiles, tile_groups = _launch_shape(
        q, n_columns, batch_rows, n_req
    )
    return int(row_tiles) * int(tile_groups)


def _sink_for(q, n_columns: int, batch_rows: int, n_req: int):
    return torch.empty(
        _workgroups(q, n_columns, batch_rows, n_req), dtype=torch.int32, device=q.device
    )


def _score_once(
    plan,
    candidate: int | None,
    q,
    index_k,
    table,
    token_to_req,
    positions,
    context,
    n_columns: int,
    batch_rows: int,
    scores,
    row_lens,
    sink,
) -> None:
    arch = paged._qsa_arch()
    use_k32 = arch == "gfx950"
    wide = paged.span_bytes(index_k) > (1 << 32)
    strides = tuple(int(s) for s in index_k.stride()[:3]) if wide else None
    rows = int(q.shape[0])
    n_req = int(context.shape[0])
    fold, tiles, row_tiles, tile_groups = _launch_shape(q, n_columns, batch_rows, n_req)
    page_size = int(index_k.shape[1])
    n_heads = int(q.shape[1])
    stream = torch.cuda.current_stream(q.device)
    common = (
        q,
        index_k,
        table,
        token_to_req,
        positions,
        context,
        scores,
        row_lens,
        int(n_columns),
        rows,
        n_req,
        float(paged._SCORE_SCALE),
        int(index_k.shape[0]),
        row_tiles,
        tile_groups,
        tiles,
    )
    if candidate is None:
        exe = plan(page_size, use_k32, n_heads, wide, strides, fold_heads=fold)
        paged._run_compiled(exe, *common, stream)
        return
    exe = plan(
        page_size,
        use_k32,
        n_heads,
        wide,
        strides,
        fold_heads=fold,
        candidate_bytes=candidate,
    )
    need = int(row_tiles) * int(tile_groups)
    if int(sink.numel()) < need:
        raise RuntimeError(f"sink has {int(sink.numel())} slots, launch needs {need}")
    paged._run_compiled(exe, *common, sink, stream)


def _chunks(rows: int, n_columns: int):
    step = paged._score_workspace_rows(n_columns)
    for start in range(0, rows, step):
        yield start, min(rows, start + step)


def _run_scores(plan, candidate, built, table, index_k, into=None, limit=None):
    q, _q_gqa, _table, token_to_req, positions, context, *_rest = built
    n_columns = _columns(table, index_k)
    batch = int(q.shape[0])
    n_req = int(context.shape[0])
    out = []
    for index, (start, end) in enumerate(_chunks(batch, n_columns)):
        if limit is not None and index >= limit:
            break
        rows = end - start
        scores = torch.full(
            (rows, n_columns), SENTINEL, dtype=torch.float32, device=q.device
        )
        row_lens = torch.full((rows,), -7, dtype=torch.int32, device=q.device)
        sink = _sink_for(q[start:end], n_columns, batch, n_req)
        _score_once(
            plan,
            candidate,
            q[start:end],
            index_k,
            table,
            token_to_req[start:end],
            positions[start:end],
            context,
            n_columns,
            batch,
            scores,
            row_lens,
            sink,
        )
        if into is not None:
            out.append((scores, row_lens))
    return out


def _time_scores(plan, candidate, built, table, index_k) -> float:
    q, _q_gqa, _table, token_to_req, positions, context, *_rest = built
    n_columns = _columns(table, index_k)
    batch = int(q.shape[0])
    n_req = int(context.shape[0])
    sink = torch.empty(
        max(
            _workgroups(q[start:end], n_columns, batch, n_req)
            for start, end in _chunks(batch, n_columns)
        ),
        dtype=torch.int32,
        device=q.device,
    )
    slices = [
        (
            q[start:end],
            token_to_req[start:end],
            positions[start:end],
            torch.empty(end - start, n_columns, dtype=torch.float32, device=q.device),
            torch.empty(end - start, dtype=torch.int32, device=q.device),
        )
        for start, end in _chunks(batch, n_columns)
    ]

    def once() -> float:
        pairs = []
        for query, req, pos, scores, row_lens in slices:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _score_once(
                plan,
                candidate,
                query,
                index_k,
                table,
                req,
                pos,
                context,
                n_columns,
                batch,
                scores,
                row_lens,
                sink,
            )
            end.record()
            pairs.append((start, end))
        torch.cuda.synchronize()
        return sum(a.elapsed_time(b) for a, b in pairs) * 1000.0

    for _ in range(WARMUP):
        once()
    return statistics.median(once() for _ in range(ITERS))


def _match(built, table, index_k, candidate: int = 0) -> None:
    ref = _run_scores(_production_scores, None, built, table, index_k, into=True)
    got = _run_scores(
        build_qsa_indexer_prefill_scores_lds_probe,
        candidate,
        built,
        table,
        index_k,
        into=True,
    )
    torch.cuda.synchronize()
    for index, ((ref_scores, ref_lens), (got_scores, got_lens)) in enumerate(
        zip(ref, got)
    ):
        if not torch.equal(ref_lens, got_lens):
            raise AssertionError(f"p32k chunk {index}: row_lens differ")
        # The scorer writes only each row's visible prefix (columns below
        # row_lens). Past it the buffer keeps whatever memory held before, so
        # the comparison is restricted to the prefix, and every visible score
        # must have been written.
        cols = torch.arange(ref_scores.shape[1], device=ref_scores.device)
        vis = cols.unsqueeze(0) < ref_lens.to(torch.int64).unsqueeze(1)
        for side, scores in (("ref", ref_scores), ("got", got_scores)):
            unwritten = int((vis & (scores == SENTINEL)).sum())
            if unwritten:
                raise AssertionError(
                    f"p32k chunk {index}: {side} left {unwritten} visible scores unwritten"
                )
        ref_v = ref_scores[vis]
        got_v = got_scores[vis]
        ref_nan = torch.isnan(ref_v)
        got_nan = torch.isnan(got_v)
        finite = torch.isfinite(ref_v) & torch.isfinite(got_v)
        gap = torch.where(finite, (ref_v - got_v).abs(), torch.zeros_like(ref_v))
        one_nan = ref_nan ^ got_nan
        # Gaps under 1e-6 are fp32 noise. A real divergence is orders larger.
        gap_max = float(gap.max()) if gap.numel() else 0.0
        if one_nan.any() or gap_max > 1e-6:
            raise AssertionError(
                f"p32k chunk {index}: visible mismatch max abs {gap_max:.3e},"
                f" one-sided nan {int(one_nan.sum())},"
                f" ref nan {int(ref_nan.sum())} got nan {int(got_nan.sum())}"
            )
    print(
        f"p32k: candidate {candidate} matches unmodified visible scores on "
        f"{len(ref)} chunks",
        flush=True,
    )


def _time_trimmed_budget(built, table, index_k) -> tuple[float, float, float]:
    q, _q_gqa, _table, token_to_req, positions, context, *_rest = built
    block_ids = torch.empty(q.shape[0], 512, dtype=torch.int32, device=q.device)
    token_ids = torch.empty(
        q.shape[0],
        bench.TOKEN_TOPK + bench.COMPRESS - 1,
        dtype=torch.int32,
        device=q.device,
    )

    def once(probe: bench._StageProbe):
        probe.reset()
        paged.qsa_indexer_block_ids(
            q,
            index_k,
            table,
            token_to_req,
            positions,
            context,
            out=block_ids,
            heads=(HEADS,),
        )
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        qsa_expand_block_indices(
            block_ids,
            positions,
            context,
            token_to_req,
            bench.COMPRESS,
            bench.TOKEN_TOPK,
            token_ids,
        )
        end.record()
        torch.cuda.synchronize()
        return (
            probe._us(probe.score),
            probe._us(probe.topk),
            start.elapsed_time(end) * 1000.0,
        )

    with bench._StageProbe() as probe:
        for _ in range(WARMUP):
            once(probe)
        samples = [once(probe) for _ in range(ITERS)]
    return tuple(statistics.median(sample[i] for sample in samples) for i in range(3))


def _host_compile() -> None:
    for var in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        if os.environ.get(var, None) != "":
            raise SystemExit(f"--host-compile needs {var}='' (every GPU hidden)")
    arch = os.environ.get("GPU_ARCHS", "")
    if arch not in CU_NUM:
        raise SystemExit(f"--host-compile needs GPU_ARCHS in {tuple(CU_NUM)}")
    from aiter.aot.flydsl import dry_run as dry_run_module

    cases = ("p32k", "p128k", "p256k")
    with dry_run_module.dry_run(arch, CU_NUM[arch]):
        index_k = bench._wide_indexer_cache()
        for name in cases:
            built = _built(name, index_k)
            print(f"{name}: host compile production scorer", flush=True)
            _run_scores(
                _production_scores,
                None,
                built,
                built[2],
                index_k,
                limit=1,
            )
        widths = {
            "p128k": _built("p128k", index_k),
            "p256k": _built("p256k", index_k),
        }
        tables = {
            "p128k:pad": (widths["p128k"], widths["p128k"][2]),
            "p128k:trim": (
                widths["p128k"],
                _table_for(widths["p128k"], _trim_pages(131072)),
            ),
            "p256k:pad": (widths["p256k"], widths["p256k"][2]),
        }
        for label, (built, table) in tables.items():
            for candidate in CANDIDATES:
                print(f"{label}: host compile candidate {candidate}", flush=True)
                _run_scores(
                    build_qsa_indexer_prefill_scores_lds_probe,
                    candidate,
                    built,
                    table,
                    index_k,
                    limit=1,
                )
            print(f"{label}: host compile block ids", flush=True)
            q, _gqa, _table, token_to_req, positions, context, *_rest = built
            # Top-k's launcher asks for the current device before it compiles.
            # The GPUs are hidden, so answer device 0 and let the compile proceed.
            with mock.patch.object(torch._C, "_cuda_getDevice", lambda: 0):
                paged.qsa_indexer_block_ids(
                    q,
                    index_k,
                    table,
                    token_to_req,
                    positions,
                    context,
                    heads=(HEADS,),
                )
    print("COMPILE_DONE", flush=True)


def _launch_check(candidate: int) -> None:
    """Fit check in its own process. A launch that fails leaves the HIP
    context in an error state, so each oversized candidate runs alone."""
    index_k = bench._wide_indexer_cache()
    p32 = _built("p32k", index_k)
    try:
        _match(p32, p32[2], index_k, candidate)
    except Exception as exc:  # noqa: BLE001
        print(
            f"LAUNCH_CHECK candidate={candidate} FAIL {type(exc).__name__}", flush=True
        )
        print(f"LAUNCH_CHECK_MSG {str(exc).splitlines()[0]}", flush=True)
        raise SystemExit(3)
    print(f"LAUNCH_CHECK candidate={candidate} OK", flush=True)


def main() -> None:
    global CANDIDATES
    parser = argparse.ArgumentParser(description="SILOTIGER-1481 phase-1 LDS probe")
    parser.add_argument("--host-compile", action="store_true")
    parser.add_argument(
        "--candidates",
        default="16384,32768,65536",
        help="comma list of nonzero candidate sizes to run in this process",
    )
    parser.add_argument(
        "--launch-check",
        type=int,
        default=None,
        help="check one candidate size's launch and scores on p32k, then exit",
    )
    args = parser.parse_args()
    if args.host_compile:
        _host_compile()
        return
    if args.launch_check is not None:
        bench._print_device()
        _launch_check(args.launch_check)
        return
    CANDIDATES = (0,) + tuple(int(x) for x in args.candidates.split(",") if x)

    bench._print_device()
    bench._assert_idle()
    if get_gfx_runtime() not in bench.SUPPORTED_GFX:
        raise SystemExit(2)

    index_k = bench._wide_indexer_cache()
    print(
        f"indexer pool pages={index_k.shape[0]} "
        f"stride={tuple(int(s) for s in index_k.stride())} "
        f"span_bytes={paged.span_bytes(index_k)}",
        flush=True,
    )
    p32 = _built("p32k", index_k)
    print("p32k: faithfulness", flush=True)
    _match(p32, p32[2], index_k)
    faith = {}
    for repeat in (1, 2):
        print(f"p32k: unmodified repeat {repeat}", flush=True)
        unmodified = _time_scores(_production_scores, None, p32, p32[2], index_k)
        print(f"p32k: candidate 0 repeat {repeat}", flush=True)
        zero = _time_scores(
            build_qsa_indexer_prefill_scores_lds_probe, 0, p32, p32[2], index_k
        )
        ratio = zero / unmodified if unmodified else float("nan")
        faith[repeat] = (unmodified, zero, ratio)
        print(
            f"p32k repeat {repeat}: unmodified {unmodified:.1f} "
            f"dummy0 {zero:.1f} ratio {ratio:.4f}",
            flush=True,
        )
    faithful = all(abs(ratio - 1.0) <= 0.05 for _u, _z, ratio in faith.values())
    print(f"FAITHFUL={faithful}", flush=True)

    # The candidate arrays only add a dead read, so every size must reproduce
    # the unmodified scores. Launch failures print without raising, so a size
    # that did not launch would show here as a mismatch, and is not timed.
    launch_ok = {}
    for candidate in CANDIDATES:
        if candidate == 0:
            continue
        print(f"p32k: candidate {candidate} match", flush=True)
        try:
            _match(p32, p32[2], index_k, candidate)
            launch_ok[candidate] = True
        except Exception as exc:  # noqa: BLE001
            launch_ok[candidate] = f"fail: {type(exc).__name__}: {exc}"
            print(f"p32k candidate {candidate} {launch_ok[candidate]}", flush=True)
    print(f"LAUNCH_OK={launch_ok}", flush=True)

    results = {}
    for name, context in (("p128k", 131072), ("p256k", bench.MAX_MODEL_LEN)):
        built = _built(name, index_k)
        widths = {"pad": built[2]}
        if name == "p128k":
            widths["trim"] = _table_for(built, _trim_pages(context))
        for label, table in widths.items():
            key = f"{name}:{label}"
            results[key] = {}
            for repeat in (1, 2):
                results[key][repeat] = {}
                print(f"{key}: unmodified repeat {repeat}", flush=True)
                base = _time_scores(_production_scores, None, built, table, index_k)
                results[key][repeat]["base"] = base
                print(f"{key} repeat {repeat}: base {base:.1f}", flush=True)
                for candidate in CANDIDATES:
                    if candidate == 0:
                        continue
                    print(f"{key}: candidate {candidate} repeat {repeat}", flush=True)
                    try:
                        got = _time_scores(
                            build_qsa_indexer_prefill_scores_lds_probe,
                            candidate,
                            built,
                            table,
                            index_k,
                        )
                    except Exception as exc:  # noqa: BLE001
                        got = f"fail: {type(exc).__name__}: {exc}"
                        print(f"{key} candidate {candidate} {got}", flush=True)
                    results[key][repeat][candidate] = got
                    if isinstance(got, float):
                        delta = got - base
                        print(
                            f"{key} repeat {repeat}: {candidate} {got:.1f} "
                            f"delta {delta:.1f}",
                            flush=True,
                        )
            for repeat in (1, 2):
                print(f"{key}: arm A budget repeat {repeat}", flush=True)
                score, topk, expand = _time_trimmed_budget(built, table, index_k)
                results[key][repeat]["budget"] = (
                    score,
                    topk,
                    expand,
                    topk + expand,
                )
                step = score + topk + expand
                share = (topk + expand) / step if step else float("nan")
                print(
                    f"{key} repeat {repeat}: T_score {score:.1f} T_topk {topk:.1f} "
                    f"T_expand {expand:.1f} budget {topk + expand:.1f} "
                    f"share {share:.3f}",
                    flush=True,
                )
        del built
        torch.cuda.empty_cache()

    print("case | bytes | T_score | delta | arm A budget | under", flush=True)
    for name in ("p128k", "p256k"):
        for label in ("pad", "trim"):
            key = f"{name}:{label}"
            if key not in results:
                continue
            base = results[key][2]["base"]
            _score, _topk, _expand, budget = results[key][2]["budget"]
            for candidate in CANDIDATES:
                if candidate == 0:
                    continue
                got = results[key][2][candidate]
                if isinstance(got, float):
                    delta = got - base
                    print(
                        f"{key} | {candidate} | {got:.1f} | {delta:.1f} | "
                        f"{budget:.1f} | {delta < budget}",
                        flush=True,
                    )
                else:
                    print(f"{key} | {candidate} | {got}", flush=True)
    print(
        f"p32k faithfulness ratios "
        f"{faith[1][2]:.4f} {faith[2][2]:.4f} faithful={faithful}",
        flush=True,
    )


if __name__ == "__main__":
    main()
