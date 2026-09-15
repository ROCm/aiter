# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-block radix selection across row bounds, distributions and output modes.

Run with ``python -m pytest op_tests/test_flydsl_topk_one_block.py``.
Launch directly to exercise both short-row block sizes and the streaming path.
"""

import numpy as np
import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.kernels_common import get_warp_size
from aiter.ops.flydsl.kernels.radix_topk_one_block import (
    build_radix_topk_one_block_module,
)
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled


def _keys(values):
    bits = values.view(np.uint32)
    keys = bits ^ np.where(bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
    # Preserve the one-block kernel's existing bitwise NaN/zero ordering.
    return keys


@pytest.mark.parametrize(
    "width,block_threads", [(4096, 256), (4096, 1024), (65527, 1024)]
)
@pytest.mark.parametrize("k", [1, 512, 2000, 2048, 4096])
@pytest.mark.parametrize("stable", [False, True])
@pytest.mark.parametrize(
    "is_decode,write_values", [(False, False), (False, True), (True, True)]
)
def test_one_block_selection(width, block_threads, k, stable, is_decode, write_values):
    arch = get_gfx()
    if arch not in ("gfx942", "gfx950", "gfx1250"):
        pytest.skip("FlyDSL top-k requires a supported AMD GPU")
    rows, next_n = 18, 3
    rng = np.random.default_rng(473)
    host = rng.standard_normal((rows, width + 17), dtype=np.float32)
    host[5:8] = host[5:8] * np.float32(2.7) + np.float32(68)
    host[8] = np.float32(68)
    host[9] = np.round(host[9])
    host[10] = rng.uniform(0.9, 1.0, width + 17).astype(np.float32)
    host[11] = -np.abs(host[11])
    host[12] = np.resize(np.array([-0.0, 0.0], dtype=np.float32), width + 17)
    host[13] = np.resize(
        np.array(
            [0x7F800000, 0xFF800000, 0x7FC00001, 0xFFC00002, 0x3F800000],
            dtype=np.uint32,
        ).view(np.float32),
        width + 17,
    )
    # Values separated only by the last radix digit, with a partial tie at k.
    host[14] = (
        np.uint32(0x3F810000) + rng.integers(0, 16, width + 17, dtype=np.uint32)
    ).view(np.float32)
    host[15:] = np.float32(-68)
    starts = (
        np.zeros(rows, dtype=np.int32)
        if is_decode
        else np.arange(rows, dtype=np.int32) % 4
    )
    if is_decode:
        seq_lens = np.array(
            [0, k + 1, width - 2, width, width + next_n, k], dtype=np.int32
        )
        ends = np.clip(
            seq_lens[np.arange(rows) // next_n] - next_n + np.arange(rows) % next_n + 1,
            0,
            width,
        )
        device_ends = torch.from_numpy(seq_lens).cuda()
    else:
        lengths = np.array(
            [0, 1, max(k - 1, 0), k, k + 1]
            + [width - 8 - r % 4 for r in range(rows - 5)],
            dtype=np.int32,
        )
        ends = np.minimum(starts + lengths, width)
        device_ends = torch.from_numpy(ends).cuda()
    # Slice retains a padded row stride, and prefill starts need not be aligned.
    logits = torch.from_numpy(host).cuda()[:, :width]
    device_starts = torch.from_numpy(starts).cuda()
    indices = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    values = torch.empty((rows, k), dtype=torch.float32, device="cuda")
    launcher = build_radix_topk_one_block_module(
        k,
        block_threads=block_threads,
        stable=stable,
        wave_size=get_warp_size(arch),
        write_values=write_values,
        is_decode=is_decode,
        short_rows=width <= 4096,
        arch=arch,
        lds_budget_bytes=78 * 1024 if arch == "gfx950" else 0,
    )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    def run():
        _run_compiled(
            launcher,
            logits,
            device_starts,
            device_ends,
            indices,
            values if write_values else logits,
            width,
            next_n if is_decode else 1,
            rows,
            stream,
        )

    references = []
    for row in range(rows):
        lo, hi = int(starts[row]), int(ends[row])
        keys = _keys(host[row, lo:hi])
        positions = np.arange(hi - lo)
        selected = np.lexsort((positions, -keys.astype(np.int64)))[:k]
        references.append((np.sort(selected) + lo, np.sort(keys[selected])))

    with torch.cuda.stream(stream):
        # Poison output before each call, then exercise repeated graph replay.
        for _ in range(3):
            indices.fill_(-123456789)
            run()
            stream.synchronize()
            result = indices.cpu().numpy()
            result_values = values.cpu().numpy() if write_values else None
            for row, (expected, selected_keys) in enumerate(references):
                count = len(expected)
                got = result[row, :count]
                assert np.all(result[row, count:] == -1)
                assert np.all((got >= starts[row]) & (got < ends[row]))
                assert len(np.unique(got)) == count
                if stable:
                    np.testing.assert_array_equal(got, expected)
                else:
                    np.testing.assert_array_equal(
                        np.sort(_keys(host[row, got])), selected_keys
                    )
                if write_values:
                    np.testing.assert_array_equal(
                        result_values[row, :count].view(np.uint32),
                        host[row, got].view(np.uint32),
                    )
                    assert np.all(np.isneginf(result_values[row, count:]))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
        saved = indices.clone()
        for _ in range(5):
            graph.replay()
        stream.synchronize()
        if stable:
            assert torch.equal(indices, saved)


@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("write_values", [False, True])
def test_bitmap_capacity_boundary(delta, write_values):
    # k=2048 uses 8427 key words under the gfx950 78 KiB LDS budget;
    # other architectures retain the 4096-word default candidate buffer.
    capacity = 8427 if get_gfx() == "gfx950" else 4096
    test_one_block_selection(
        capacity * 32 + delta, 1024, 2048, True, False, write_values
    )


def test_non_power_of_two_bitonic_fallback():
    test_one_block_selection(300001, 1024, 2000, True, False, True)


@pytest.mark.parametrize("delta", [-1, 0, 1])
def test_bitmap_work_boundary(delta):
    # 128 indices use 28 bitonic stages, or 1792 pair comparisons.
    test_one_block_selection(1792 * 32 + delta, 1024, 128, True, False, True)
