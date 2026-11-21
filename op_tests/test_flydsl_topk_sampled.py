# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correct sampled-decode TopK coverage against a torch ordered-key oracle."""

import argparse

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.topk.topk_per_row import flydsl_sampled_topk_decode
from aiter.ops.topk import _select_flydsl_topk_decode_path, top_k_per_row_decode
from aiter.test_common import benchmark, checkAllclose, run_perftest

K = 2048
_MODES = ("ordinary_graph", "adversarial", "underfill", "short_ragged")


def ordered_keys(values):
    bits = values.view(torch.int32).long()
    ordered = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    return torch.where(values.isnan(), 0x7FFFFFFF, ordered)


def check_selection(logits, seq, next_n, output):
    lengths = seq[:, None] - next_n + torch.arange(next_n, device="cuda") + 1
    lengths = lengths.clamp(0, logits.shape[1]).flatten()
    err = 0.0
    for row, length in enumerate(lengths.tolist()):
        idx = output[row].long()
        valid = idx >= 0
        count = min(length, K)
        assert valid.sum().item() == count
        assert torch.all(idx[valid] < length)
        assert idx[valid].unique().numel() == count
        if length <= K:
            expected_positions = torch.arange(K, device="cuda")
            expected_positions[length:] = -1
            assert torch.equal(idx, expected_positions)
        elif count:
            keys = ordered_keys(logits[row, :length])
            expected = keys.topk(count).values.sort().values
            selected = keys[idx[valid]].sort().values
            err = max(
                err,
                checkAllclose(
                    selected.to(torch.float64),
                    expected.to(torch.float64),
                    rtol=0,
                    atol=0,
                    tol_err_ratio=0,
                    msg=f"sampled row={row} length={length}",
                ),
            )
    return err


def sampled_columns(row, length):
    """Host mirror of the public kernel's deterministic sample positions."""
    mask = 0xFFFFFFFF
    stratum = max(2048, ((length + 4095) // 4096) * 32)
    chunks = (length + stratum - 1) // stratum
    cols = []
    for chunk in range(min(chunks, 128)):
        begin = chunk * stratum
        extent = min(stratum, length - begin)
        hashed = (chunk + row * 4099 + 17) & mask
        hashed = ((hashed ^ (hashed >> 16)) * 0x7FEB352D) & mask
        hashed = ((hashed ^ (hashed >> 15)) * 0x846CA68B) & mask
        hashed ^= hashed >> 16
        base = begin + (hashed % max(extent // 32, 1)) * 32
        cols += [base + t * 4 + j for t in range(8) for j in range(4)]
    return [column for column in cols if column < length]


def _make_case(mode):
    torch.manual_seed(42)
    if mode == "ordinary_graph":
        logits = torch.randn((4, 128000), device="cuda")
        return logits, torch.full((1,), 128000, device="cuda", dtype=torch.int32), 4
    if mode == "adversarial":
        values = torch.tensor(
            [
                float("nan"),
                float("inf"),
                float("-inf"),
                1.0,
                1.0,
                -0.0,
                0.0,
                -1.0,
            ],
            device="cuda",
        )
        logits = values.repeat(4, 16000)
        return logits, torch.full((1,), 128000, device="cuda", dtype=torch.int32), 4
    if mode == "underfill":
        width = 128000
        logits = torch.randn((4, width), device="cuda")
        for row in range(4):
            columns = torch.tensor(sampled_columns(row, width), device="cuda")
            high = columns[: K // 2]
            logits[row, high] = 100.0 + torch.rand(high.numel(), device="cuda")
        return logits, torch.full((1,), width, device="cuda", dtype=torch.int32), 4
    if mode == "short_ragged":
        seq = torch.tensor(
            [0, 7, K - 1, K, K + 3, 8192, 11000],
            device="cuda",
            dtype=torch.int32,
        )
        logits = torch.randn((seq.numel() * 4, 11000), device="cuda")
        return logits, seq, 4
    raise ValueError(f"unknown mode {mode!r}")


@benchmark()
def test_sampled_topk(mode):
    logits, seq, next_n = _make_case(mode)
    rows = logits.shape[0]
    if mode == "ordinary_graph":
        assert (
            _select_flydsl_topk_decode_path(
                "gfx950", False, logits.shape[1], rows, next_n, K
            )
            == "sampled"
        )
    output = torch.full((rows, K), -99, dtype=torch.int32, device="cuda")
    report_counts = mode in ("adversarial", "underfill")
    counts = (
        torch.full((rows,), -99, dtype=torch.int32, device="cuda")
        if report_counts
        else None
    )

    def candidate():
        if mode == "ordinary_graph":
            top_k_per_row_decode(
                logits,
                next_n,
                seq,
                output,
                rows,
                logits.stride(0),
                logits.stride(1),
                K,
            )
        else:
            flydsl_sampled_topk_decode(
                logits,
                next_n,
                seq,
                output,
                candidate_counts=counts,
            )
        return output if counts is None else (output, counts)

    if mode == "ordinary_graph":
        stream = torch.cuda.Stream()
        producer = torch.cuda.current_stream()
        ready = torch.cuda.Event()
        ready.record(producer)
        stream.wait_event(ready)
        with torch.cuda.stream(stream):
            candidate()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            candidate()

        def replay():
            with torch.cuda.stream(stream):
                graph.replay()
            return output, counts

        _, us = run_perftest(replay, num_iters=10, num_warmup=2)
    else:
        _, us = run_perftest(candidate, num_iters=10, num_warmup=2)

    err = check_selection(logits, seq, next_n, output)
    assert err == 0, f"sampled TopK mismatch: {err:.2%}"
    if mode == "adversarial":
        assert torch.all(
            counts > 9000
        ), "tie plateau did not use count-overflow fallback"
    elif mode == "underfill":
        assert torch.all(counts < K), "threshold did not use count-underflow fallback"
    nbytes = logits.numel() * logits.element_size() + output.numel() * 4
    return {
        "gfx": get_gfx_runtime(),
        "sampled us": us,
        "sampled TFLOPS": 0,
        "sampled TB/s": nbytes / us / 1e6,
        "sampled err": err,
    }


def main():
    if get_gfx_runtime() != "gfx950":
        aiter.logger.warning(
            "sampled decode TopK unsupported on %s; skipping", get_gfx_runtime()
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Sampled decode TopK correctness boundaries",
    )
    parser.add_argument("--modes", nargs="*", choices=_MODES, default=list(_MODES))
    args = parser.parse_args()
    summary = pd.DataFrame([test_sampled_topk(mode) for mode in args.modes])
    aiter.logger.info(
        "sampled decode TopK summary (markdown):\n%s",
        summary.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
