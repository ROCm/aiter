# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import statistics
import time

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops import topk as public_topk_impl
from aiter.ops.flydsl import topk_per_row as topk_per_row_impl
from aiter.ops.flydsl.kernels.topk_per_row_decode import (
    topk_per_row_decode_workspace_shapes,
)
from aiter.test_common import benchmark, checkAllclose

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
E2E_WARMUP_ITERS = 20
E2E_TIMED_ITERS = 101


def test_flydsl_topk_three_pass_workspace_geometry():
    assert topk_per_row_decode_workspace_shapes(26, True)[0][1] == 16
    assert topk_per_row_decode_workspace_shapes(128, False)[0][1] == 16


@pytest.mark.parametrize(
    "arch,stable,top_k,rows,width,expected",
    [
        ("gfx942", True, 2048, 1, 65_536, False),
        ("gfx950", True, 128, 1, 65_536, False),
        ("gfx950", True, 256, 1, 65_536, False),
        ("gfx950", True, 512, 1, 65_536, True),
        ("gfx950", True, 1024, 1, 65_536, True),
        ("gfx950", True, 2048, 1, 65_536, True),
        ("gfx950", True, 4096, 1, 65_536, True),
        ("gfx950", False, 2048, 1, 131_071, False),
        ("gfx950", False, 2048, 32, 131_072, True),
        ("gfx950", False, 2048, 33, 131_072, False),
        ("gfx950", True, 2048, 128, 20_000, True),
        ("gfx950", True, 2048, 129, 20_000, False),
        ("gfx950", True, 2048, 1, 32_767, False),
        ("gfx950", True, 2048, 16, 32_768, True),
        ("gfx950", True, 2048, 17, 32_768, False),
        ("gfx950", True, 2048, 32, 65_536, True),
        ("gfx950", True, 2048, 33, 65_536, False),
    ],
)
def test_public_topk_decode_gate(
    monkeypatch, arch, stable, top_k, rows, width, expected
):
    from aiter.ops.flydsl import utils as flydsl_utils

    public_topk_impl._flydsl_topk_decode_available.cache_clear()
    monkeypatch.setattr(public_topk_impl, "get_gfx", lambda: arch)
    monkeypatch.setattr(public_topk_impl, "_FLYDSL_TOPK_DECODE_DISABLED", False)
    monkeypatch.setattr(flydsl_utils, "is_flydsl_available", lambda: True)
    logits = torch.empty((rows, width), dtype=torch.float32, device="cuda")
    assert (
        public_topk_impl._should_use_flydsl_topk_decode(
            logits,
            rows,
            top_k,
            stable,
        )
        is expected
    )


@pytest.mark.parametrize("stable", [False, True])
def test_public_topk_decode_dispatch_matches_hip(monkeypatch, stable):
    from aiter.ops.flydsl import utils as flydsl_utils

    public_topk_impl._flydsl_topk_decode_available.cache_clear()
    monkeypatch.setattr(public_topk_impl, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(public_topk_impl, "_FLYDSL_TOPK_DECODE_DISABLED", False)
    monkeypatch.setattr(flydsl_utils, "is_flydsl_available", lambda: True)
    rows, width, top_k = 4, 20_000, 2048
    logits = torch.randn((rows, width), dtype=torch.float32, device="cuda")
    seq_lens = torch.full((rows,), width, dtype=torch.int32, device="cuda")
    output = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    reference = torch.empty_like(output)
    public_topk_impl.top_k_per_row_decode(
        logits,
        1,
        seq_lens,
        output,
        rows,
        logits.stride(0),
        logits.stride(1),
        top_k,
        stable,
    )
    _run_hip(logits, 1, seq_lens, reference, top_k, stable)
    torch.cuda.synchronize()
    if not stable:
        output = torch.sort(output, dim=-1).values
        reference = torch.sort(reference, dim=-1).values
    torch.testing.assert_close(output, reference, rtol=0, atol=0)


def test_flydsl_topk_workspace_cache_reuses_allocation():
    rows = 4
    context_len = topk_per_row_impl._ONE_WORKGROUP_MAX_ROW_WIDTH + 1
    top_k = 2048
    logits = torch.randn((rows, context_len), dtype=torch.float32, device="cuda")
    seq_lens = torch.full((rows,), context_len, dtype=torch.int32, device=logits.device)
    output = torch.empty((rows, top_k), dtype=torch.int32, device=logits.device)

    topk_per_row_impl.clear_topk_per_row_decode_workspace_cache()
    try:
        _run_flydsl(logits, 1, seq_lens, output, top_k, True)
        _run_flydsl(logits, 1, seq_lens, output, top_k, True)
        torch.cuda.synchronize()
        cache_info = topk_per_row_impl._get_cached_workspace.cache_info()
        assert cache_info.misses == 1
        assert cache_info.hits >= 1
    finally:
        topk_per_row_impl.clear_topk_per_row_decode_workspace_cache()


@pytest.mark.parametrize("stable", [False, True])
def test_flydsl_topk_dynamic_n_reuses_launcher(stable: bool):
    rows, top_k = 4, 2048
    topk_per_row_impl.build_topk_per_row_decode_module.cache_clear()
    topk_per_row_impl.clear_topk_per_row_decode_workspace_cache()
    try:
        for context_len in (20_001, 40_003):
            logits = torch.randn(
                (rows, context_len), dtype=torch.float32, device="cuda"
            )
            seq_lens = torch.full(
                (rows,), context_len, dtype=torch.int32, device="cuda"
            )
            output = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
            reference = torch.empty_like(output)
            _run_flydsl(logits, 1, seq_lens, output, top_k, stable)
            _run_hip(logits, 1, seq_lens, reference, top_k, stable)
            torch.cuda.synchronize()
            if stable:
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
            else:
                torch.testing.assert_close(
                    torch.sort(output, dim=1).values,
                    torch.sort(reference, dim=1).values,
                    rtol=0,
                    atol=0,
                )

        cache_info = topk_per_row_impl.build_topk_per_row_decode_module.cache_info()
        assert cache_info.misses == 1
        assert cache_info.hits >= 1
    finally:
        topk_per_row_impl.build_topk_per_row_decode_module.cache_clear()
        topk_per_row_impl.clear_topk_per_row_decode_workspace_cache()


def test_flydsl_topk_rejects_strided_seq_lens():
    rows, context_len, top_k = 4, 20_001, 2048
    logits = torch.randn((rows, context_len), dtype=torch.float32, device="cuda")
    seq_lens = torch.full((rows * 2,), context_len, dtype=torch.int32, device="cuda")[
        ::2
    ]
    output = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    with pytest.raises(ValueError, match="seq_lens must be contiguous"):
        _run_flydsl(logits, 1, seq_lens, output, top_k, True)


def run_torch(
    logits: torch.Tensor,
    context_lens: list[int],
    next_n: int,
    top_k: int,
    stable: bool,
) -> torch.Tensor:
    rows = []
    for row in range(logits.shape[0]):
        row_len = min(
            logits.shape[1],
            max(
                0,
                context_lens[row // next_n] - next_n + row % next_n + 1,
            ),
        )
        valid_k = min(row_len, top_k)
        padded = torch.full((top_k,), -1, dtype=torch.int64, device=logits.device)
        if stable:
            selected = torch.argsort(
                logits[row, :row_len], descending=True, stable=True
            )[:valid_k]
            padded[:valid_k] = torch.sort(selected).values
        else:
            if valid_k > 0:
                padded[:valid_k] = torch.topk(
                    logits[row, :row_len], valid_k, sorted=False
                ).indices
        rows.append(padded)
    return torch.stack(rows).to(torch.int32)


def _run_flydsl(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    top_k: int,
    stable: bool,
) -> torch.Tensor:
    aiter.flydsl_top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        top_k,
        stable,
    )
    return indices


def _run_hip(
    logits: torch.Tensor,
    next_n: int,
    seq_lens: torch.Tensor,
    indices: torch.Tensor,
    top_k: int,
    stable: bool,
) -> torch.Tensor:
    public_topk_impl._hip_top_k_per_row_decode(
        logits,
        next_n,
        seq_lens,
        indices,
        logits.shape[0],
        logits.stride(0),
        logits.stride(1),
        top_k,
        stable,
    )
    return indices


def _run_blocking_e2e(candidates):
    """Measure Host call through post-call GPU synchronization."""
    names = tuple(candidates)
    outputs = {}
    for _ in range(E2E_WARMUP_ITERS):
        for name in names:
            outputs[name] = candidates[name]()
    torch.cuda.synchronize()

    samples = {name: [] for name in names}
    for iteration in range(E2E_TIMED_ITERS):
        order = names if iteration % 2 == 0 else names[::-1]
        for name in order:
            start_ns = time.perf_counter_ns()
            output = candidates[name]()
            torch.cuda.synchronize()
            samples[name].append((time.perf_counter_ns() - start_ns) / 1000.0)
            outputs[name] = output

    return outputs, {
        name: statistics.median(candidate_samples)
        for name, candidate_samples in samples.items()
    }


@pytest.mark.parametrize("stable", [False, True])
def test_flydsl_topk_short_rows_are_minus_one_padded(stable: bool):
    top_k = 2048
    context_len = 4096
    next_n = 1
    valid_lens = [0, 1, top_k - 1, top_k]
    num_rows = len(valid_lens)

    torch.manual_seed(123)
    logits = torch.randn(num_rows, context_len, dtype=torch.float32)
    seq_lens = torch.tensor(valid_lens, dtype=torch.int32, device=logits.device)
    expected = run_torch(logits, valid_lens, next_n, top_k, stable)

    eager_output = torch.full((num_rows, top_k), -777, dtype=torch.int32)
    _run_flydsl(logits, next_n, seq_lens, eager_output, top_k, stable)
    torch.cuda.synchronize()

    # Capture with full rows, then replay with shorter rows. This verifies that
    # stale indices from the captured execution are overwritten by -1.
    graph_seq_lens = torch.full(
        (num_rows,), context_len, dtype=torch.int32, device=logits.device
    )
    graph_output = torch.full((num_rows, top_k), -777, dtype=torch.int32)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_flydsl(logits, next_n, graph_seq_lens, graph_output, top_k, stable)
    graph_seq_lens.copy_(seq_lens)
    graph.replay()
    torch.cuda.synchronize()

    expected_compare = expected if stable else torch.sort(expected, dim=-1).values
    for name, output in (("eager", eager_output), ("graph", graph_output)):
        output_compare = output if stable else torch.sort(output, dim=-1).values
        torch.testing.assert_close(
            output_compare,
            expected_compare,
            rtol=0,
            atol=0,
            msg=f"{name} output mismatch",
        )
        for row, valid_len in enumerate(valid_lens):
            assert torch.all(output[row, valid_len:] == -1), (
                f"{name} row {row} did not overwrite its padded tail"
            )


def test_flydsl_topk_single_row_dynamic_multi_graph():
    top_k = 2048
    context_len = 65536
    logits = torch.randn((1, context_len), dtype=torch.float32, device="cuda")
    seq_lens = torch.full((1,), context_len, dtype=torch.int32, device="cuda")
    output = torch.empty((1, top_k), dtype=torch.int32, device="cuda")
    reference = torch.empty_like(output)

    topk_per_row_impl.build_topk_per_row_decode_module.cache_clear()
    # Long single-row calls deliberately use the safe multi-launch path.
    _run_flydsl(logits, 1, seq_lens, output, top_k, True)
    _run_hip(logits, 1, seq_lens, reference, top_k, True)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, reference, rtol=0, atol=0)
    assert topk_per_row_impl.build_topk_per_row_decode_module.cache_info().misses == 1

    graph_lens = seq_lens.clone()
    graph_output = torch.full_like(output, -777)
    _run_flydsl(logits, 1, graph_lens, graph_output, top_k, True)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_flydsl(logits, 1, graph_lens, graph_output, top_k, True)

    # Exercise stable ties, then transition to the direct-fill path on replay.
    logits.zero_()
    graph.replay()
    _run_hip(logits, 1, seq_lens, reference, top_k, True)
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, reference, rtol=0, atol=0)

    graph_lens.fill_(1)
    graph.replay()
    _run_hip(logits, 1, graph_lens, reference, top_k, True)
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, reference, rtol=0, atol=0)


@pytest.mark.parametrize("num_rows", [2, 32])
def test_flydsl_topk_short_multi_row_one_workgroup_graph(num_rows: int):
    top_k = 2048
    context_len = 20_000
    torch.manual_seed(20260828 + num_rows)
    logits = torch.randint(
        -8,
        9,
        (num_rows, context_len),
        dtype=torch.int32,
        device="cuda",
    ).to(torch.float32)
    seq_lens = torch.full((num_rows,), context_len, dtype=torch.int32, device="cuda")
    graph_output = torch.empty((num_rows, top_k), dtype=torch.int32, device="cuda")
    reference = torch.empty_like(graph_output)

    _run_flydsl(logits, 1, seq_lens, graph_output, top_k, True)
    _run_hip(logits, 1, seq_lens, reference, top_k, True)
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, reference, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run_flydsl(logits, 1, seq_lens, graph_output, top_k, True)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, reference, rtol=0, atol=0)

    # Reuse the captured graph with shorter rows to exercise dynamic row bounds.
    seq_lens.fill_(top_k + 17)
    graph.replay()
    _run_hip(logits, 1, seq_lens, reference, top_k, True)
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, reference, rtol=0, atol=0)


def test_flydsl_topk_three_pass_matches_hip():
    if get_gfx() != "gfx950":
        pytest.skip("three-pass radix dispatch is tuned for gfx950")

    top_k = 2048
    context_len = 5 << 18
    valid_lens = [
        0,
        1,
        top_k - 1,
        top_k,
        context_len // 4,
        context_len - 17,
        context_len,
    ]
    num_rows = len(valid_lens)

    torch.manual_seed(321)
    logits = torch.randint(
        -8,
        9,
        (num_rows, context_len),
        dtype=torch.int32,
        device="cuda",
    ).to(torch.float32)
    seq_lens = torch.tensor(valid_lens, dtype=torch.int32, device="cuda")
    flydsl_output = torch.full(
        (num_rows, top_k), -777, dtype=torch.int32, device="cuda"
    )
    hip_output = torch.full_like(flydsl_output, -777)

    _run_flydsl(logits, 1, seq_lens, flydsl_output, top_k, True)
    _run_hip(logits, 1, seq_lens, hip_output, top_k, True)
    torch.cuda.synchronize()

    torch.testing.assert_close(flydsl_output, hip_output, rtol=0, atol=0)


@benchmark()
def test_flydsl_topk_decode(
    batch_size: int,
    context_len: int,
    top_k: int,
    next_n: int,
    row_padding: int,
    stable: bool,
) -> dict:
    if top_k > context_len:
        raise ValueError("top_k must not exceed the logits row width")

    num_rows = batch_size * next_n
    torch.manual_seed(42)
    if stable:
        logits_storage = torch.randint(
            -8,
            9,
            (num_rows, context_len + row_padding),
            dtype=torch.int32,
        ).to(torch.float32)
    else:
        logits_storage = torch.randn(
            num_rows, context_len + row_padding, dtype=torch.float32
        )
    logits = logits_storage[:, :context_len]
    min_seq_len = top_k + next_n - 1
    context_lens = [
        max(min_seq_len, context_len - (request * 17) % 257)
        for request in range(batch_size)
    ]
    seq_lens = torch.tensor(context_lens, dtype=torch.int32, device=logits.device)
    ref = run_torch(logits, context_lens, next_n, top_k, stable)

    outputs = {
        "flydsl": torch.empty(num_rows, top_k, dtype=torch.int32),
        "hip": torch.empty(num_rows, top_k, dtype=torch.int32),
    }
    candidates = {
        "flydsl": lambda: _run_flydsl(
            logits, next_n, seq_lens, outputs["flydsl"], top_k, stable
        ),
        "hip": lambda: _run_hip(
            logits, next_n, seq_lens, outputs["hip"], top_k, stable
        ),
    }

    # Stable mode adds deterministic count and ordered-write passes.
    flops = 0
    nbytes = (6 if stable else 5) * logits.numel() * logits.element_size() + outputs[
        "flydsl"
    ].numel() * outputs["flydsl"].element_size()
    ref_compare = (ref if stable else torch.sort(ref, dim=-1).values).to(dtypes.fp32)
    timed_outputs, e2e_us = _run_blocking_e2e(candidates)
    ret = {"gfx": get_gfx(), "timing": "Host call + GPU sync"}
    for name in candidates:
        out = timed_outputs[name]
        us = e2e_us[name]
        out_compare = (out if stable else torch.sort(out, dim=-1).values).to(
            dtypes.fp32
        )
        err = checkAllclose(
            ref_compare,
            out_compare,
            rtol=0,
            atol=0,
            msg=f"{name}: top_k_per_row_decode",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


test_flydsl_topk_decode.__test__ = False


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "FlyDSL top_k_per_row_decode unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="FlyDSL per-row decode TopK correctness and performance",
    )
    parser.add_argument("-b", "--batch-size", type=int, nargs="*", default=[2, 4])
    parser.add_argument(
        "-c", "--context-len", type=int, nargs="*", default=[4101, 131072]
    )
    parser.add_argument("-k", "--top-k", type=int, nargs="*", default=[1, 2048, 4096])
    parser.add_argument("-n", "--next-n", type=int, nargs="*", default=[1, 4])
    parser.add_argument("--row-padding", type=int, nargs="*", default=[0, 17])
    parser.add_argument(
        "--stable", type=dtypes.str2bool, nargs="*", default=[False, True]
    )
    args = parser.parse_args()

    rows = []
    for (
        context_len,
        top_k,
        next_n,
        stable,
        batch_size,
        row_padding,
    ) in itertools.product(
        args.context_len,
        args.top_k,
        args.next_n,
        args.stable,
        args.batch_size,
        args.row_padding,
    ):
        rows.append(
            test_flydsl_topk_decode(
                batch_size, context_len, top_k, next_n, row_padding, stable
            )
        )
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "FlyDSL top_k_per_row_decode summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
