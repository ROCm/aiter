# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch


def test_partial_record_geometry():
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import partial_record_format

    f1024 = partial_record_format(1024)
    assert f1024.payload_bytes == 2048
    assert f1024.source_offset == 2048
    assert f1024.record_bytes == 2304
    assert f1024.padding_bytes == 252

    f3584 = partial_record_format(3584)
    assert f3584.payload_bytes == 7168
    assert f3584.source_offset == 7168
    assert f3584.record_bytes == 7424
    assert f3584.padding_bytes == 252

    f7168 = partial_record_format(7168)
    assert f7168.payload_bytes == 14336
    assert f7168.source_offset == 14336
    assert f7168.record_bytes == 14592
    assert f7168.padding_bytes == 252


@pytest.mark.parametrize("hidden_dim", [1024, 3584, 7168])
def test_partial_record_pack_unpack_bitwise(hidden_dim):
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        pytest.skip("FlyDSL required")
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import build_partial_record_module

    torch.manual_seed(41 + hidden_dim)
    dev = torch.device("cuda", 0)
    stream = torch.cuda.current_stream(dev)
    max_sources = 5
    record_count = 3
    partial = torch.randn(
        max_sources, hidden_dim, dtype=torch.bfloat16, device=dev
    )
    source_ids = torch.tensor([3, 0, 4], dtype=torch.int32, device=dev)
    module = build_partial_record_module(hidden_dim)
    fmt = module.format
    records = torch.full(
        (record_count * fmt.record_bytes,), 0xA5, dtype=torch.uint8, device=dev
    )
    rows_out = torch.empty(
        record_count, hidden_dim, dtype=torch.bfloat16, device=dev
    )
    sources_out = torch.empty(record_count, dtype=torch.int32, device=dev)
    error = torch.zeros(1, dtype=torch.int32, device=dev)

    module.launch_pack(
        partial.data_ptr(),
        source_ids.data_ptr(),
        records.data_ptr(),
        record_count,
        max_sources,
        error.data_ptr(),
        stream=stream,
    )
    torch.cuda.synchronize(dev)

    expected = partial[source_ids.to(torch.int64)]
    packed_rows = torch.stack(
        [
            records[
                i * fmt.record_bytes : i * fmt.record_bytes + fmt.payload_bytes
            ].view(torch.bfloat16)
            for i in range(record_count)
        ]
    )
    dwords = records.view(torch.int32).view(record_count, fmt.record_bytes // 4)
    assert torch.equal(packed_rows, expected)
    assert torch.equal(dwords[:, fmt.source_offset // 4], source_ids)
    for i in range(record_count):
        padding = records[
            i * fmt.record_bytes
            + fmt.source_offset
            + 4 : (i + 1) * fmt.record_bytes
        ]
        assert torch.count_nonzero(padding).item() == 0
    assert error.item() == 0

    module.launch_unpack(
        records.data_ptr(),
        rows_out.data_ptr(),
        sources_out.data_ptr(),
        record_count,
        max_sources,
        error.data_ptr(),
        stream=stream,
    )
    torch.cuda.synchronize(dev)
    assert torch.equal(rows_out, expected)
    assert torch.equal(sources_out, source_ids)
    assert error.item() == 0

    # Source validation is device-side; an invalid record is reported and its
    # output row is deterministically zeroed instead of indexing out of bounds.
    dwords[0, fmt.source_offset // 4] = max_sources
    rows_out.fill_(1)
    error.zero_()
    module.launch_unpack(
        records.data_ptr(),
        rows_out.data_ptr(),
        sources_out.data_ptr(),
        record_count,
        max_sources,
        error.data_ptr(),
        stream=stream,
    )
    torch.cuda.synchronize(dev)
    assert error.item() >= 1
    assert torch.count_nonzero(rows_out[0]).item() == 0
