# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and benchmark suite for experimental paged-MQA Stage A."""

import argparse
import itertools
from dataclasses import dataclass

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import (
    flydsl_fp8_paged_mqa_local_topk,
    flydsl_fp8_paged_mqa_topk,
)
from aiter.ops.flydsl.split_topk_merge import (
    alloc_split_topk_merge_workspace,
    split_topk_merge,
)
from aiter.ops.flydsl.topk_per_row import flydsl_top_k_per_row_decode
from aiter.ops.shuffle import shuffle_weight
from aiter.ops.triton.attention.pa_mqa_logits import deepgemm_fp8_paged_mqa_logits
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ("gfx950",)
HEADS = 32
HEAD_DIM = 128
_ORACLE_CHUNK = 16384
_EDGE_MODES = (
    "empty",
    "k_tail",
    "packed_page64",
    "invalid_page",
    "short_dirty",
    "merge_short",
    "graph",
)


@dataclass
class Case:
    q: torch.Tensor
    kv: torch.Tensor
    scales: torch.Tensor
    weights: torch.Tensor
    lengths: torch.Tensor
    block_tables: torch.Tensor


def _make_case(
    rows,
    length,
    page_size,
    *,
    seed=17,
    ragged=False,
):
    """Build independent physical KV pages for each request."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    pages_per_request = max(1, (length + page_size - 1) // page_size)
    pages = pages_per_request * rows
    q = (
        torch.randn(
            rows,
            1,
            HEADS,
            HEAD_DIM,
            device=device,
            generator=generator,
        )
        * 0.25
    ).to(torch.float8_e4m3fn)
    kv = (
        torch.randn(pages, page_size, HEAD_DIM, device=device, generator=generator)
        * 0.25
    ).to(torch.float8_e4m3fn)
    scales = torch.rand(
        pages, page_size, device=device, generator=generator, dtype=torch.float32
    )
    weights = torch.randn(
        rows, HEADS, device=device, generator=generator, dtype=torch.float32
    )
    lengths = torch.full((rows,), length, device=device, dtype=torch.int32)
    if ragged and rows > 1:
        lengths[0] = 0
        lengths[-1] = max(0, length - 3)
    block_tables = torch.stack(
        [
            torch.randperm(pages_per_request, device=device, generator=generator)
            + request * pages_per_request
            for request in range(rows)
        ]
    ).to(torch.int32)
    return Case(q, kv, scales, weights, lengths, block_tables)


def _preshuffle_kv(kv):
    return shuffle_weight(kv, layout=(16, 16)).contiguous()


def _pack_kv(kv, scales):
    pages, page_size, dim = kv.shape
    raw = torch.empty(
        (pages, page_size * (dim + 4)),
        dtype=torch.uint8,
        device=kv.device,
    )
    raw[:, : page_size * dim] = kv.view(torch.uint8).reshape(pages, -1)
    raw[:, page_size * dim :] = scales.view(torch.uint8).reshape(pages, -1)
    return raw.view(pages, page_size, 1, dim + 4)


def run_torch(case):
    """Independent FP32 oracle with explicit page mapping and epilogue order."""
    rows = case.q.shape[0]
    q = case.q.reshape(rows, HEADS, HEAD_DIM)
    page_size = case.kv.shape[1]
    flat_kv = case.kv.reshape(-1, HEAD_DIM).float()
    flat_scales = case.scales.reshape(-1)
    outputs = []
    for row in range(rows):
        length = int(case.lengths[row].item())
        logical = torch.arange(length, device=case.q.device)
        physical = (
            case.block_tables[row, logical // page_size] * page_size
            + logical % page_size
        )
        row_out = torch.empty(length, dtype=torch.float32, device=case.q.device)
        for begin in range(0, length, _ORACLE_CHUNK):
            span = physical[begin : begin + _ORACLE_CHUNK]
            keys = flat_kv[span]
            dots = torch.sum(q[row].float()[:, None, :] * keys[None, :, :], dim=-1)
            activated = torch.relu(dots) * flat_scales[span][None, :]
            row_out[begin : begin + span.numel()] = torch.sum(
                case.weights[row, :, None] * activated, dim=0
            )
        outputs.append(row_out)
    return outputs


def _assert_topk_values(population, selected, count, *, msg):
    expected = torch.topk(population.float(), count, sorted=True).values
    got = torch.sort(selected.float(), descending=True).values[:count]
    return checkAllclose(
        got,
        expected,
        rtol=2e-4,
        atol=2e-4,
        tol_err_ratio=0,
        msg=msg,
    )


def _assert_candidates(case, scores, positions, counts, *, k, splits):
    reference = run_torch(case)
    err = 0.0
    for row, row_scores in enumerate(reference):
        length = row_scores.numel()
        union = set()
        for split in range(splits):
            begin = length * split // splits
            end = length * (split + 1) // splits
            count = min(k, end - begin)
            assert int(counts[row, split]) == count
            got_positions = positions[row, split, :count].long()
            assert got_positions.unique().numel() == count
            assert torch.all((got_positions >= begin) & (got_positions < end))
            if count:
                err = max(
                    err,
                    _assert_topk_values(
                        row_scores[begin:end],
                        row_scores[got_positions],
                        count,
                        msg=f"row={row} split={split}",
                    ),
                    checkAllclose(
                        row_scores[got_positions].float(),
                        scores[row, split, :count].float(),
                        rtol=2e-4,
                        atol=2e-4,
                        msg=f"row={row} split={split} selected scores",
                    ),
                )
                union.update(got_positions.cpu().tolist())
            assert torch.all(positions[row, split, count:] == -1)
            assert torch.all(torch.isneginf(scores[row, split, count:]))

        global_count = min(k, length)
        if global_count:
            union_positions = torch.tensor(sorted(union), device=row_scores.device)
            err = max(
                err,
                _assert_topk_values(
                    row_scores,
                    row_scores[union_positions],
                    global_count,
                    msg=f"row={row} union recall",
                ),
            )
    return err


def _assert_compact_topk(case, scores, positions, k):
    reference = run_torch(case)
    err = 0.0
    for row, row_scores in enumerate(reference):
        count = min(k, row_scores.numel())
        assert torch.all(positions[row, count:] == -1)
        assert torch.all(torch.isneginf(scores[row, count:]))
        if count:
            got = positions[row, :count].long()
            assert got.unique().numel() == count
            err = max(
                err,
                _assert_topk_values(
                    row_scores,
                    row_scores[got],
                    count,
                    msg=f"row={row} compact TopK",
                ),
                checkAllclose(
                    row_scores[got].float(),
                    scores[row, :count].float(),
                    rtol=2e-4,
                    atol=2e-4,
                    msg=f"row={row} compact TopK scores",
                ),
            )
    return err


def _tensor_bytes(*tensors):
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


def _edge_result(us, flops, nbytes, err):
    return {
        "gfx": get_gfx(),
        "kernel us": us,
        "kernel TFLOPS": flops / us / 1e6,
        "kernel TB/s": nbytes / us / 1e6,
        "kernel err": err,
    }


@benchmark()
def test_indexer_edges(mode):
    """Run one distinct correctness boundary through its public entry point."""
    if mode == "empty":
        rows, length, k, splits = 1, 0, 128, 1
        case = _make_case(rows, length, 16, seed=11)

        def candidate():
            return flydsl_fp8_paged_mqa_local_topk(
                case.q,
                _preshuffle_kv(case.kv),
                case.scales,
                case.weights,
                case.lengths,
                case.block_tables,
                k=k,
                num_splits=splits,
            )

        outputs, us = run_perftest(candidate)
        err = _assert_candidates(case, *outputs, k=k, splits=splits)
        nbytes = _tensor_bytes(
            case.q,
            case.kv,
            case.scales,
            case.weights,
            case.lengths,
            case.block_tables,
            *outputs,
        )
        return _edge_result(us, 0, nbytes, err)

    if mode == "k_tail":
        rows, length, k, splits = 2, 129, 128, 4
        case = _make_case(rows, length, 16, seed=13, ragged=True)

        def candidate():
            return flydsl_fp8_paged_mqa_local_topk(
                case.q,
                _preshuffle_kv(case.kv),
                case.scales,
                case.weights,
                case.lengths,
                case.block_tables,
                k=k,
                num_splits=splits,
            )

        outputs, us = run_perftest(candidate)
        err = _assert_candidates(case, *outputs, k=k, splits=splits)
        flops = 2 * sum(case.lengths.tolist()) * HEADS * HEAD_DIM
        nbytes = _tensor_bytes(
            case.q,
            case.kv,
            case.scales,
            case.weights,
            case.lengths,
            case.block_tables,
            *outputs,
        )
        return _edge_result(us, flops, nbytes, err)

    if mode in ("packed_page64", "short_dirty", "graph"):
        length = 32 if mode == "short_dirty" else 512
        rows, k, splits = 2, 128, 4
        case = _make_case(rows, length, 64, seed=17)
        packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
        workspace = alloc_split_topk_merge_workspace(case.q.device, rows)
        out_scores = torch.full((rows, k), 7.0, dtype=torch.float32, device="cuda")
        out_positions = torch.full((rows, k), 123, dtype=torch.int32, device="cuda")

        def candidate():
            return flydsl_fp8_paged_mqa_topk(
                case.q,
                packed,
                None,
                case.weights,
                case.lengths,
                case.block_tables,
                k=k,
                num_splits=splits,
                workspace=workspace,
                out_scores=out_scores if mode == "short_dirty" else None,
                out_positions=out_positions if mode == "short_dirty" else None,
            )

        if mode == "graph":
            candidate()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                scores, positions = candidate()

            def replay():
                graph.replay()
                return scores, positions

            outputs, us = run_perftest(replay)
        else:
            outputs, us = run_perftest(candidate)
        scores, positions = outputs
        if mode == "short_dirty":
            assert scores.data_ptr() == out_scores.data_ptr()
            assert positions.data_ptr() == out_positions.data_ptr()
        err = _assert_compact_topk(case, scores, positions, k)
        flops = 2 * rows * length * HEADS * HEAD_DIM
        nbytes = _tensor_bytes(
            case.q,
            packed,
            case.weights,
            case.lengths,
            case.block_tables,
            scores,
            positions,
        )
        return _edge_result(us, flops, nbytes, err)

    if mode == "invalid_page":
        rows, length, k = 1, 128, 128
        case = _make_case(rows, length, 64, seed=19)
        case.block_tables[0, 0] = -1
        packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)

        def candidate():
            return flydsl_fp8_paged_mqa_topk(
                case.q,
                packed,
                None,
                case.weights,
                case.lengths,
                case.block_tables,
                k=k,
                num_splits=1,
            )

        (scores, positions), us = run_perftest(candidate)
        live_mask = scores[0] > -float("inf")
        live = positions[0, live_mask]
        assert live.numel() == 64
        assert set(live.cpu().tolist()) == set(range(64, 128))
        valid_case = Case(
            case.q,
            case.kv,
            case.scales,
            case.weights,
            torch.full_like(case.lengths, 64),
            case.block_tables[:, 1:2],
        )
        expected = run_torch(valid_case)[0]
        actual = scores[0, live_mask]
        err = checkAllclose(
            actual.sort().values,
            expected.sort().values,
            rtol=2e-4,
            atol=2e-4,
            msg=mode,
        )
        nbytes = _tensor_bytes(
            case.q,
            packed,
            case.weights,
            case.lengths,
            case.block_tables,
            scores,
            positions,
        )
        return _edge_result(us, 2 * 64 * HEADS * HEAD_DIM, nbytes, err)

    if mode == "merge_short":
        rows, splits, k, live = 1, 2, 128, 32
        scores = torch.full(
            (rows, splits, k), -float("inf"), dtype=torch.float32, device="cuda"
        )
        positions = torch.full((rows, splits, k), -1, dtype=torch.int32, device="cuda")
        counts = torch.full((rows, splits), live, dtype=torch.int32, device="cuda")
        for split in range(splits):
            begin = split * live
            scores[0, split, :live] = torch.arange(
                begin, begin + live, dtype=torch.float32, device="cuda"
            )
            positions[0, split, :live] = torch.arange(
                begin, begin + live, dtype=torch.int32, device="cuda"
            )
        out_scores = torch.full((rows, k), 7.0, dtype=torch.float32, device="cuda")
        out_positions = torch.full((rows, k), 123, dtype=torch.int32, device="cuda")

        def candidate():
            return split_topk_merge(
                scores,
                positions,
                counts,
                k=k,
                out_scores=out_scores,
                out_positions=out_positions,
            )

        (selected_scores, selected_positions), us = run_perftest(candidate)
        assert set(selected_positions[0, : 2 * live].cpu().tolist()) == set(
            range(2 * live)
        )
        assert torch.all(selected_positions[0, 2 * live :] == -1)
        assert torch.all(torch.isneginf(selected_scores[0, 2 * live :]))
        err = checkAllclose(
            selected_scores[0, : 2 * live].sort().values,
            torch.arange(2 * live, dtype=torch.float32, device="cuda"),
            rtol=0,
            atol=0,
            msg=mode,
        )
        nbytes = _tensor_bytes(
            scores,
            positions,
            counts,
            selected_scores,
            selected_positions,
        )
        return _edge_result(us, 0, nbytes, err)

    raise ValueError(f"unknown mode {mode!r}")


@benchmark()
def test_fp8_paged_mqa_topk(rows, length):
    """Benchmark compact TopK against materialized logits plus TopK."""
    case = _make_case(rows, length, 64, seed=73)
    reference = run_torch(case)
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
    k = 2048
    logits = torch.empty((rows, length), dtype=torch.float32, device=case.q.device)
    materialized_positions = torch.empty(
        (rows, k), dtype=torch.int32, device=case.q.device
    )
    materialized_scores = torch.empty(
        (rows, k), dtype=torch.float32, device=case.q.device
    )

    def compact():
        return flydsl_fp8_paged_mqa_topk(
            case.q,
            packed,
            None,
            case.weights,
            case.lengths,
            case.block_tables,
            k=k,
            num_splits=None,
        )

    def logits_plus_topk():
        deepgemm_fp8_paged_mqa_logits(
            case.q,
            packed,
            case.weights,
            logits,
            case.lengths,
            case.block_tables,
            length,
            Preshuffle=True,
            KVBlockSize=64,
            ChunkK=256,
        )
        flydsl_top_k_per_row_decode(
            logits,
            1,
            case.lengths,
            materialized_positions,
            rows,
            logits.stride(0),
            1,
            k,
            False,
            materialized_scores,
        )
        return materialized_scores, materialized_positions

    flops = 2 * rows * length * HEADS * HEAD_DIM
    shared_bytes = _tensor_bytes(
        case.q,
        packed,
        case.weights,
        case.lengths,
        case.block_tables,
        materialized_scores,
        materialized_positions,
    )
    candidate_bytes = {
        "compact": shared_bytes,
        "logits_topk": shared_bytes + 2 * logits.numel() * logits.element_size(),
    }
    ret = {"gfx": get_gfx()}
    for name, candidate in (
        ("compact", compact),
        ("logits_topk", logits_plus_topk),
    ):
        (scores, positions), us = run_perftest(candidate)
        err = 0.0
        for row, row_scores in enumerate(reference):
            got = positions[row].long()
            assert got.unique().numel() == k, f"{name}: row {row} duplicate positions"
            err = max(
                err,
                _assert_topk_values(
                    row_scores,
                    row_scores[got],
                    k,
                    msg=f"{name} row={row}",
                ),
            )
        err = max(
            err,
            checkAllclose(
                reference[0][positions[0].long()].float(),
                scores[0].float(),
                rtol=2e-4,
                atol=2e-4,
                msg=name,
            ),
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = candidate_bytes[name] / us / 1e6
        ret[f"{name} err"] = err
    return ret


def _summarize(name, rows):
    summary = pd.DataFrame(rows)
    aiter.logger.info(
        "%s summary (markdown):\n%s", name, summary.to_markdown(index=False)
    )


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "fp8_paged_mqa_local_topk unsupported on %s; skipping", get_gfx()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Experimental paged-MQA Stage A correctness and benchmark sweep",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        choices=_EDGE_MODES,
        default=list(_EDGE_MODES),
    )
    parser.add_argument("-b", "--batch", type=int, nargs="*", default=[8, 32])
    parser.add_argument(
        "-l",
        "--length",
        type=int,
        nargs="*",
        default=[16384, 65536],
    )
    args = parser.parse_args()

    _summarize(
        "fp8_paged_mqa_local_topk edges",
        [test_indexer_edges(mode) for mode in args.modes],
    )
    _summarize(
        "fp8_paged_mqa_topk",
        [
            test_fp8_paged_mqa_topk(rows, length)
            for rows, length in itertools.product(args.batch, args.length)
        ],
    )


if __name__ == "__main__":
    main()
