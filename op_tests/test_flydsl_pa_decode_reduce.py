# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check packed PA reduction against an independent FP32 softmax oracle."""

import importlib.util

import pytest
import torch


@pytest.mark.parametrize(
    "head,batch,kv_heads,qlen,group,bound,dtype,sink_dtype",
    [
        (64, 6, 1, 1, 8, 8, torch.float32, None),
        (128, 6, 2, 4, 8, 48, torch.bfloat16, torch.float32),
        (128, 32, 1, 4, 8, 256, torch.bfloat16, None),
        (192, 6, 2, 1, 16, 80, torch.float32, torch.float16),
        (256, 6, 1, 4, 8, 256, torch.bfloat16, None),
        (256, 32, 1, 4, 8, 256, torch.bfloat16, None),
        (256, 128, 1, 4, 8, 256, torch.bfloat16, torch.bfloat16),
        (256, 6, 2, 4, 8, 65, torch.float16, torch.float32),
        (256, 6, 1, 4, 8, 80, torch.bfloat16, None),
        (512, 6, 2, 3, 8, 80, torch.float16, torch.float32),
        (1024, 6, 1, 4, 8, 64, torch.bfloat16, None),
        (1024, 6, 1, 4, 8, 80, torch.float32, torch.float32),
    ],
)
def test_packed_reduce(head, batch, kv_heads, qlen, group, bound, dtype, sink_dtype):
    if not torch.cuda.is_available() or importlib.util.find_spec("flydsl") is None:
        pytest.skip("requires ROCm and FlyDSL")
    props = torch.cuda.get_device_properties(0)
    if props.gcnArchName.split(":")[0] not in ("gfx942", "gfx950"):
        pytest.skip("PA decode supports gfx942/gfx950")
    if bound > props.multi_processor_count:
        pytest.skip("partition bound exceeds the device CU count")

    from aiter.ops.flydsl.pa_decode import launch_pa_decode_ps_reduce

    torch.manual_seed(5949)
    rows = qlen * group
    counts = [0, 1, 2, 4, 8, 16, 32, 33, 64, bound]
    counts = [min(counts[i % len(counts)], bound) for i in range(batch)]
    # Include both the largest LDS rung and the short register branch.
    counts[-1] = bound
    starts, cursor = [], 0
    for count in counts:
        starts.append(cursor)
        cursor += count
    capacity = cursor + bound
    sums = torch.full((kv_heads, capacity, rows), float("nan"), device="cuda")
    maxes = torch.full_like(sums, float("nan"))
    partials = torch.full(
        (kv_heads, capacity, rows, head), float("nan"), dtype=dtype, device="cuda"
    )
    sums[:, :cursor] = torch.rand((kv_heads, cursor, rows), device="cuda") * 3 + 0.1
    maxes[:, :cursor] = torch.randn((kv_heads, cursor, rows), device="cuda")
    partials[:, :cursor] = torch.randn(
        (kv_heads, cursor, rows, head), dtype=dtype, device="cuda"
    )
    info = torch.tensor(list(zip(starts, counts)), dtype=torch.int32, device="cuda")
    output = torch.full(
        (batch, qlen, kv_heads, group, head),
        float("nan"),
        dtype=dtype,
        device="cuda",
    )
    sinks = None
    if sink_dtype is not None:
        sinks = torch.linspace(-2, 2, kv_heads * group, dtype=sink_dtype, device="cuda")
        sinks[0], sinks[-1] = -torch.inf, torch.inf

    def reference():
        result = torch.zeros_like(output, dtype=torch.float32)
        for seq, (first, count) in enumerate(info.cpu().tolist()):
            if count == 0:
                continue
            m = maxes[:, first : first + count]
            s = sums[:, first : first + count]
            maximum = m.amax(1)
            sink = None
            if sinks is not None:
                sink = sinks.float().reshape(kv_heads, group).repeat(1, qlen)
                maximum = torch.maximum(maximum, sink)
            weights = s * torch.exp(m - maximum[:, None, :])
            denominator = weights.sum(1)
            if sink is not None:
                # A +inf sink has unit mass at the shared max and suppresses KV.
                shift = torch.where(sink == maximum, 0.0, sink - maximum)
                denominator += torch.exp(shift)
            weights /= denominator[:, None, :]
            values = partials[:, first : first + count].float()
            reduced = (values * weights[..., None]).sum(1)
            result[seq] = reduced.reshape(kv_heads, qlen, group, head).transpose(0, 1)
        return result

    def launch():
        launch_pa_decode_ps_reduce(
            output,
            sums,
            maxes,
            partials,
            sinks,
            output.stride(0),
            output.stride(1),
            output.stride(2),
            output.stride(3),
            maxes.stride(0),
            maxes.stride(1),
            partials.stride(0),
            partials.stride(1),
            partials.stride(2),
            query_seq_len=qlen,
            query_group_size=group,
            head_size=head,
            partition_bound=bound,
            max_partitions=bound,
            reduce_info=info,
            stream=torch.cuda.current_stream(),
        )

    def check():
        expected = reference()
        atol, rtol = {
            torch.float32: (1e-5, 1e-4),
            torch.float16: (2e-3, 2e-3),
            torch.bfloat16: (0.015, 0.01),
        }[dtype]
        torch.testing.assert_close(output.float(), expected, atol=atol, rtol=rtol)
        assert torch.isfinite(output).all()

    launch()
    check()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        launch()
    torch.cuda.current_stream().wait_stream(stream)
    # Reuse one specialization and graph after all live counts become small.
    # Empty sequences may legally point at the end of the scratch allocation.
    updated = info.clone()
    updated[:, 1].clamp_(max=4)
    updated[0] = torch.tensor([capacity, 0], dtype=torch.int32, device="cuda")
    info.copy_(updated)
    output.fill_(float("nan"))
    graph.replay()
    check()
