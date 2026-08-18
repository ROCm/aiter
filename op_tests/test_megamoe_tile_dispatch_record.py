# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest
import torch


def test_dispatch_record_parameterized_geometry():
    from aiter.ops.flydsl.kernels.megamoe_tile.runtime import HierCcoArenaLayout
    from aiter.ops.flydsl.kernels.megamoe_tile.wire import K3DispatchWireLayout

    small = K3DispatchWireLayout(hidden_bytes=512, scale_bytes=32, topk=2)
    k3 = K3DispatchWireLayout(hidden_bytes=1792, scale_bytes=112, topk=16)
    final = K3DispatchWireLayout(hidden_bytes=3584, scale_bytes=224, topk=16)
    assert (small.record_bytes, small.records_per_64k) == (2048, 32)
    assert (k3.record_bytes, k3.records_per_64k) == (2048, 32)
    assert final.route_mask_offset + 8 == 3952
    assert (final.record_bytes, final.records_per_64k) == (4096, 16)
    arena = HierCcoArenaLayout.create(
        max_m_tiles=8,
        max_source_tokens=64,
        max_fanout_records=32,
        fanout_record_bytes=final.record_bytes,
    )
    assert arena.region("fanout_inbox").shape == (32, 4096)


@pytest.mark.parametrize(
    "hidden_dim,topk,record_bytes",
    [(1024, 2, 2048), (3584, 16, 2048), (7168, 16, 4096)],
)
def test_dispatch_record_device_matches_host_reference(
    hidden_dim, topk, record_bytes
):
    if not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    from aiter.ops.flydsl.utils import is_flydsl_available

    if not is_flydsl_available():
        pytest.skip("FlyDSL required")
    from aiter.ops.flydsl.kernels.megamoe_tile.kernels import build_dispatch_record_module
    from aiter.ops.flydsl.kernels.megamoe_tile.wire import (
        K3DispatchWireLayout,
        pack_dispatch_records,
        unpack_dispatch_records,
    )

    torch.manual_seed(53 + hidden_dim + topk)
    dev = torch.device("cuda", 0)
    stream = torch.cuda.current_stream(dev)
    token_count = 5
    record_count = 3
    num_experts = 32
    max_source_tokens = 128
    layout = K3DispatchWireLayout(
        hidden_bytes=hidden_dim // 2,
        scale_bytes=hidden_dim // 32,
        topk=topk,
    )
    module = build_dispatch_record_module(hidden_dim, topk)
    assert module.layout == layout
    assert layout.record_bytes == record_bytes

    hidden = torch.randint(
        0, 256, (token_count, layout.hidden_bytes), dtype=torch.uint8, device=dev
    )
    scales = torch.randint(
        0, 256, (token_count, layout.scale_bytes), dtype=torch.uint8, device=dev
    )
    ids = torch.randint(
        0, num_experts, (token_count, topk), dtype=torch.int32, device=dev
    )
    weights = torch.randn(token_count, topk, dtype=torch.float32, device=dev)
    sources = torch.tensor([9, 11, 13, 15, 17], dtype=torch.int32, device=dev)
    masks = torch.tensor(
        [(1 << ((i % topk) + 1)) - 1 for i in range(token_count)],
        dtype=torch.int64,
        device=dev,
    )
    row_ids = torch.tensor([3, 0, 4], dtype=torch.int32, device=dev)
    records = torch.full(
        (record_count, layout.record_bytes), 0xA5, dtype=torch.uint8, device=dev
    )
    error = torch.zeros(1, dtype=torch.int32, device=dev)

    module.launch_pack(
        hidden.data_ptr(),
        scales.data_ptr(),
        ids.data_ptr(),
        weights.data_ptr(),
        sources.data_ptr(),
        masks.data_ptr(),
        row_ids.data_ptr(),
        records.data_ptr(),
        record_count,
        token_count,
        num_experts,
        max_source_tokens,
        error.data_ptr(),
        stream=stream,
    )
    torch.cuda.synchronize(dev)

    rows = row_ids.to(torch.int64)
    reference = pack_dispatch_records(
        hidden[rows],
        scales[rows],
        ids[rows],
        weights[rows],
        sources[rows],
        masks[rows],
        layout=layout,
    )
    assert torch.equal(records, reference)
    assert error.item() == 0

    hidden_out = torch.empty(
        record_count, layout.hidden_bytes, dtype=torch.uint8, device=dev
    )
    scales_out = torch.empty(
        record_count, layout.scale_bytes, dtype=torch.uint8, device=dev
    )
    ids_out = torch.empty(record_count, topk, dtype=torch.int32, device=dev)
    weights_out = torch.empty(record_count, topk, dtype=torch.float32, device=dev)
    sources_out = torch.empty(record_count, dtype=torch.int32, device=dev)
    masks_out = torch.empty(record_count, dtype=torch.int64, device=dev)
    module.launch_unpack(
        records.data_ptr(),
        hidden_out.data_ptr(),
        scales_out.data_ptr(),
        ids_out.data_ptr(),
        weights_out.data_ptr(),
        sources_out.data_ptr(),
        masks_out.data_ptr(),
        record_count,
        num_experts,
        max_source_tokens,
        error.data_ptr(),
        stream=stream,
    )
    torch.cuda.synchronize(dev)

    fields = unpack_dispatch_records(reference, layout=layout)
    assert torch.equal(hidden_out, fields["hidden_fp4_u8"])
    assert torch.equal(scales_out, fields["scales_e8m0_u8"])
    assert torch.equal(ids_out, fields["topk_ids"])
    assert torch.equal(weights_out, fields["topk_weights"])
    assert torch.equal(sources_out, fields["source_flat_token"])
    assert torch.equal(masks_out, fields["destination_route_mask"])
    assert error.item() == 0

    # Validate source/id/reserved/padding checks without relying on host code.
    dwords = records.view(torch.int32).flatten()
    stride = layout.record_bytes // 4
    dwords[layout.source_offset // 4] = max_source_tokens
    dwords[stride + (layout.hidden_bytes + layout.scale_bytes) // 4] = num_experts
    dwords[2 * stride + layout.source_offset // 4 + 1] = 1
    error.zero_()
    module.launch_unpack(
        records.data_ptr(),
        hidden_out.data_ptr(),
        scales_out.data_ptr(),
        ids_out.data_ptr(),
        weights_out.data_ptr(),
        sources_out.data_ptr(),
        masks_out.data_ptr(),
        record_count,
        num_experts,
        max_source_tokens,
        error.data_ptr(),
        stream=stream,
    )
    torch.cuda.synchronize(dev)
    assert error.item() >= 3
