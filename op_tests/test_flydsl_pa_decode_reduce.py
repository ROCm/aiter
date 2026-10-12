# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exercise runtime partition buckets with packed rows and poisoned padding."""

import importlib.util

import pytest
import torch


@pytest.mark.parametrize(
    "head_dim,max_parts,compact",
    [
        (64, 17, False),
        (128, 64, False),
        (128, 80, True),
        (128, 80, False),
        (192, 33, False),
    ],
)
@pytest.mark.parametrize("use_sinks", [False, True])
def test_pa_decode_reduce_partition_buckets(head_dim, max_parts, compact, use_sinks):
    if not torch.cuda.is_available() or importlib.util.find_spec("flydsl") is None:
        pytest.skip("ROCm and FlyDSL are required")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(device)
    if properties.gcnArchName.split(":", 1)[0] not in ("gfx942", "gfx950"):
        pytest.skip("PA decode requires gfx942 or gfx950")
    if max_parts > properties.multi_processor_count:
        pytest.skip("partition cap exceeds device CU count")

    from aiter.ops.flydsl.pa_decode import launch_pa_decode_ps_reduce

    torch.manual_seed(0)
    counts = [0, 1, 2, 3, 4, 5, 8, 9, 16, 17, 31, 32, 33, max_parts, 0]
    counts = [min(count, max_parts) for count in counts]
    batch, query_length, kv_heads, group = len(counts), 2, 2, 4
    rows = query_length * group
    capacity = max(batch, sum(counts)) if compact else batch * max_parts
    sums = torch.full((kv_heads, capacity, rows), float("nan"), device=device)
    maxima = torch.full_like(sums, float("nan"))
    partials = torch.full(
        (kv_heads, capacity, rows, head_dim),
        float("nan"),
        dtype=torch.bfloat16,
        device=device,
    )
    output = torch.empty(
        (batch, query_length, kv_heads, group, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    info = torch.empty((batch, 2), dtype=torch.int32, device=device)
    sinks = None
    if use_sinks:
        sinks = torch.linspace(-2, 2, kv_heads * group, device=device)
        sinks[0], sinks[-1] = float("-inf"), float("inf")

    def populate(counts):
        sums.fill_(float("nan"))
        maxima.fill_(float("nan"))
        partials.fill_(float("nan"))
        reference = torch.zeros_like(output, dtype=torch.float32)
        records, cursor = [], 0
        for seq, count in enumerate(counts):
            records.append((cursor, count))
            if count:
                part_sums = torch.rand((kv_heads, count, rows), device=device) + 0.1
                part_maxes = torch.randn_like(part_sums)
                part_outputs = torch.randn(
                    (kv_heads, count, rows, head_dim),
                    dtype=torch.bfloat16,
                    device=device,
                )
                sums[:, cursor : cursor + count] = part_sums
                maxima[:, cursor : cursor + count] = part_maxes
                partials[:, cursor : cursor + count] = part_outputs
                lse = part_maxes + part_sums.log()
                denominator = torch.logsumexp(lse, dim=1)
                if sinks is not None:
                    sink_rows = sinks.reshape(kv_heads, group).repeat(1, query_length)
                    denominator = torch.logaddexp(denominator, sink_rows)
                weights = (lse - denominator[:, None]).exp()
                result = (part_outputs.float() * weights[..., None]).sum(dim=1)
                reference[seq] = result.reshape(
                    kv_heads, query_length, group, head_dim
                ).permute(1, 0, 2, 3)
                cursor += count
        info.copy_(torch.tensor(records, dtype=torch.int32, device=device))
        return reference

    def launch():
        launch_pa_decode_ps_reduce(
            output,
            sums,
            maxima,
            partials,
            sinks,
            *output.stride()[:4],
            sums.stride(0),
            sums.stride(1),
            partials.stride(0),
            partials.stride(1),
            partials.stride(2),
            query_seq_len=query_length,
            query_group_size=group,
            head_size=head_dim,
            context_partition_num=max_parts,
            stream=torch.cuda.current_stream(),
            reduce_info=info,
        )

    reference = populate(counts)
    launch()
    torch.testing.assert_close(output.float(), reference, atol=0.008, rtol=0.008)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        launch()
    torch.cuda.current_stream().wait_stream(side)
    for replay_counts in (list(reversed(counts)), [0] * batch):
        reference = populate(replay_counts)
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output.float(), reference, atol=0.008, rtol=0.008)
