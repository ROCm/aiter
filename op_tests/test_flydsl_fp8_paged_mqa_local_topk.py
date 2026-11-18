# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness and benchmark suite for experimental paged-MQA Stage A."""

import argparse
from dataclasses import dataclass

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl import (
    flydsl_fp8_paged_mqa_local_topk,
    flydsl_fp8_paged_mqa_topk,
)
from aiter.ops.flydsl.split_topk_merge import (
    alloc_split_topk_merge_workspace,
    split_topk_merge,
)
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ("gfx950",)
HEADS = 32
HEAD_DIM = 128
_ORACLE_CHUNK = 16384
_EDGE_MODES = (
    "ragged_128_split",
    "reservoir_compaction",
    "packed_abi",
    "merge_exceptions",
    "graph_streams",
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
    lengths=None,
    physical_pages=None,
):
    """Build independent physical KV pages for each request."""
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    if lengths is None:
        lengths = torch.full((rows,), length, device=device, dtype=torch.int32)
    else:
        lengths = torch.tensor(lengths, device=device, dtype=torch.int32)
        length = max(length, int(lengths.max().item()))
    assert lengths.numel() == rows
    pages_per_request = max(1, (length + page_size - 1) // page_size)
    pages = max(pages_per_request * rows, physical_pages or 0)
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
    scales = torch.ones(
        pages, page_size, device=device, generator=generator, dtype=torch.float32
    )
    weights = torch.randn(
        rows, HEADS, device=device, generator=generator, dtype=torch.float32
    )
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
        table_span = case.block_tables.shape[1] * page_size
        length = min(int(case.lengths[row].item()), table_span)
        logical = torch.arange(length, device=case.q.device)
        physical_pages = case.block_tables[row, logical // page_size]
        valid_page = (physical_pages >= 0) & (physical_pages < case.kv.shape[0])
        physical = physical_pages.clamp(0, case.kv.shape[0] - 1) * page_size
        physical += logical % page_size
        row_out = torch.empty(length, dtype=torch.float32, device=case.q.device)
        for begin in range(0, length, _ORACLE_CHUNK):
            span = physical[begin : begin + _ORACLE_CHUNK]
            valid = valid_page[begin : begin + _ORACLE_CHUNK]
            keys = flat_kv[span]
            dots = torch.sum(q[row].float()[:, None, :] * keys[None, :, :], dim=-1)
            activated = torch.relu(dots)
            scores = (
                torch.sum(case.weights[row, :, None] * activated, dim=0)
                * flat_scales[span]
            )
            row_out[begin : begin + span.numel()] = torch.where(
                valid, scores, float("-inf")
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


def _ordered_float_keys(values):
    bits = values.view(torch.int32).long()
    return torch.where(values.isnan(), 0x7FFFFFFF, bits ^ ((bits >> 31) & 0x7FFFFFFF))


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
    assert err == 0, f"kernel produced {err:.2%} elements outside tolerance"
    return {
        "gfx": get_gfx_runtime(),
        "kernel us": us,
        "kernel TFLOPS": flops / us / 1e6,
        "kernel TB/s": nbytes / us / 1e6,
        "kernel err": err,
    }


def _compact_case(
    case,
    packed,
    *,
    k,
    splits,
    out_scores=None,
    out_positions=None,
    workspace=None,
):
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
        out_scores=out_scores,
        out_positions=out_positions,
    )


def _run_graph_stream_case():
    k, splits = 2048, 4
    case = _make_case(2, 513, 64, seed=17, physical_pages=4096)
    case.block_tables[0, 0] = -1
    case.block_tables[0, 1] = case.kv.shape[0] + 4096
    case.block_tables[1, 2] = case.kv.shape[0] - 1
    packed = _pack_kv(_preshuffle_kv(case.kv), case.scales).squeeze(2)
    workspace = alloc_split_topk_merge_workspace(case.q.device, case.q.shape[0])
    out_scores = torch.full(
        (case.q.shape[0], k), float("nan"), dtype=torch.float32, device="cuda"
    )
    out_positions = torch.full(
        (case.q.shape[0], k), 123456, dtype=torch.int32, device="cuda"
    )
    changed_lengths = (
        torch.tensor([128, 513], dtype=torch.int32, device="cuda"),
        torch.tensor([600, 576], dtype=torch.int32, device="cuda"),
    )
    stream = torch.cuda.Stream()
    producer = torch.cuda.current_stream()
    ready = torch.cuda.Event()
    ready.record(producer)
    stream.wait_event(ready)

    with torch.cuda.stream(stream):
        warm = _compact_case(
            case,
            packed,
            k=k,
            splits=splits,
            workspace=workspace,
            out_scores=out_scores,
            out_positions=out_positions,
        )
    stream.synchronize()
    warm_err = _assert_compact_topk(case, *warm, k)
    assert torch.count_nonzero(workspace[0]).item() == 0

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        captured = _compact_case(
            case,
            packed,
            k=k,
            splits=splits,
            workspace=workspace,
            out_scores=out_scores,
            out_positions=out_positions,
        )

    def ordered_replay(lengths):
        with torch.cuda.stream(stream):
            case.lengths.copy_(lengths)
            out_scores.fill_(7.0)
            out_positions.fill_(123456)
            graph.replay()
        stream.synchronize()
        err = _assert_compact_topk(case, *captured, k)
        assert torch.count_nonzero(workspace[0]).item() == 0
        return err

    err = max(warm_err, *(ordered_replay(lengths) for lengths in changed_lengths))

    def timed_replay():
        with torch.cuda.stream(stream):
            graph.replay()
        return captured

    _, us = run_perftest(timed_replay, num_iters=10, num_warmup=2)
    nbytes = _tensor_bytes(
        case.q,
        case.kv,
        case.scales,
        case.weights,
        case.lengths,
        case.block_tables,
        *captured,
    )
    return _edge_result(us, 0, nbytes, err)


@benchmark()
def test_indexer_edges(mode):
    """Run one non-redundant Stage A / compact-merge boundary."""
    if mode == "ragged_128_split":
        case = _make_case(
            3,
            129,
            64,
            seed=11,
            lengths=(0, 127, 129),
        )
        packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)

        def candidate():
            return _compact_case(
                case,
                packed,
                k=128,
                splits=128,
            )

        (scores, positions), us = run_perftest(candidate)
        err = _assert_compact_topk(case, scores, positions, 128)
        return _edge_result(
            us, 0, _tensor_bytes(case.q, case.kv, scores, positions), err
        )

    if mode == "reservoir_compaction":
        case = _make_case(1, 14_000, 64, seed=13)
        pattern = torch.tensor([0.125, 0.5, 2.0, 4.0], device=case.scales.device)
        case.scales.copy_(
            pattern.repeat(case.scales.numel() // 4).reshape_as(case.scales)
        )
        shuffled = _preshuffle_kv(case.kv)

        def candidate():
            return flydsl_fp8_paged_mqa_local_topk(
                case.q,
                shuffled,
                case.scales,
                case.weights,
                case.lengths,
                case.block_tables,
                k=512,
                num_splits=1,
            )

        outputs, us = run_perftest(candidate)
        err = _assert_candidates(case, *outputs, k=512, splits=1)
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

    if mode == "packed_abi":
        rows, k, splits = 2, 1024, 4
        case = _make_case(rows, 32, 64, seed=19)
        packed = _pack_kv(_preshuffle_kv(case.kv), case.scales)
        packed_forms = (packed, packed.squeeze(2))
        workspace = alloc_split_topk_merge_workspace(case.q.device, rows)
        out_scores = torch.full((rows, k), 7.0, dtype=torch.float32, device="cuda")
        out_positions = torch.full((rows, k), 123, dtype=torch.int32, device="cuda")
        call = 0

        def candidate():
            nonlocal call
            packed_form = packed_forms[call & 1]
            call += 1
            out_scores.fill_(7.0)
            out_positions.fill_(123)
            return _compact_case(
                case,
                packed_form,
                k=k,
                splits=splits,
                workspace=workspace,
                out_scores=out_scores,
                out_positions=out_positions,
            )

        (scores, positions), us = run_perftest(candidate)
        assert scores.data_ptr() == out_scores.data_ptr()
        assert positions.data_ptr() == out_positions.data_ptr()
        err = _assert_compact_topk(case, scores, positions, k)
        nbytes = _tensor_bytes(
            case.q,
            case.kv,
            case.scales,
            case.weights,
            case.lengths,
            case.block_tables,
            scores,
            positions,
        )
        return _edge_result(us, 2 * rows * 32 * HEADS * HEAD_DIM, nbytes, err)

    if mode == "merge_exceptions":
        rows, splits, k = 1, 3, 128
        counts = torch.tensor([[-9, k + 17, 7]], dtype=torch.int32, device="cuda")
        scores = torch.full(
            (rows, splits, k), -float("inf"), dtype=torch.float32, device="cuda"
        )
        positions = torch.full((rows, splits, k), -1, dtype=torch.int32, device="cuda")
        exceptional = torch.tensor(
            [float("nan"), float("inf"), 3.0, 3.0, -0.0, 0.0, float("-inf")],
            device="cuda",
        )
        scores[0, 1] = torch.arange(k, dtype=torch.float32, device="cuda")
        positions[0, 1] = torch.arange(k, dtype=torch.int32, device="cuda")
        scores[0, 1, : exceptional.numel()] = exceptional
        scores[0, 2, :7] = torch.arange(7, dtype=torch.float32, device="cuda")
        positions[0, 2, :7] = torch.arange(k, k + 7, dtype=torch.int32, device="cuda")
        out_scores = torch.full((rows, k), 7.0, dtype=torch.float32, device="cuda")
        out_positions = torch.full((rows, k), 123, dtype=torch.int32, device="cuda")

        def candidate():
            out_scores.fill_(7.0)
            out_positions.fill_(123)
            return split_topk_merge(
                scores,
                positions,
                counts,
                k=k,
                out_scores=out_scores,
                out_positions=out_positions,
            )

        (selected_scores, selected_positions), us = run_perftest(candidate)
        assert selected_scores.data_ptr() == out_scores.data_ptr()
        assert selected_positions.data_ptr() == out_positions.data_ptr()
        assert selected_positions[0].unique().numel() == k
        expected = torch.cat((scores[0, 1], scores[0, 2, :7])).topk(k).values
        err = checkAllclose(
            _ordered_float_keys(selected_scores[0]).sort().values.to(torch.float64),
            _ordered_float_keys(expected).sort().values.to(torch.float64),
            rtol=0,
            atol=0,
            tol_err_ratio=0,
            msg=mode,
        )
        return _edge_result(
            us,
            0,
            _tensor_bytes(
                scores,
                positions,
                counts,
                selected_scores,
                selected_positions,
            ),
            err,
        )

    if mode == "graph_streams":
        return _run_graph_stream_case()

    raise ValueError(f"unknown mode {mode!r}")


def _summarize(rows):
    summary = pd.DataFrame(rows)
    aiter.logger.info(
        "fp8_paged_mqa_local_topk summary (markdown):\n%s",
        summary.to_markdown(index=False),
    )


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "fp8_paged_mqa_local_topk unsupported on %s; skipping",
            get_gfx_runtime(),
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Paged-MQA Stage A correctness and compact-merge boundaries",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        choices=_EDGE_MODES,
        default=list(_EDGE_MODES),
    )
    args = parser.parse_args()
    _summarize([test_indexer_edges(mode) for mode in args.modes])


if __name__ == "__main__":
    main()
