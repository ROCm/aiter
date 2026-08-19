#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Test Iris AllGather and ReduceScatter on four gfx950 GPUs.

Run with:

    torchrun --standalone --nproc-per-node=4 \
        op_tests/multigpu_tests/triton_test/test_iris_agrs.py
"""

import inspect
import json
import os

import torch
import torch.distributed as dist
from aiter.ops.triton.comms import IrisCommContext, all_gather, reduce_scatter

WORLD_SIZE = 4
AG_SHAPES = (
    ("hidden", 4, 4096, torch.bfloat16),
    ("weights", 4, 6, torch.float32),
    ("ids", 4, 6, torch.int32),
)
RS_SHAPE = (16, 4096, torch.bfloat16)
HAS_AG_OUTPUT = "output" in inspect.signature(all_gather).parameters
HAS_RS_OUTPUT = "output" in inspect.signature(reduce_scatter).parameters


def make_input(
    rows: int,
    columns: int,
    dtype: torch.dtype,
    rank: int,
    replay: int,
    device: torch.device,
) -> torch.Tensor:
    values = torch.arange(rows * columns, dtype=torch.int64, device=device).reshape(
        rows, columns
    )
    if dtype == torch.int32:
        return ((values % 97) + rank * 101 + replay * 1009).to(dtype)
    values = ((values % 17) - 8).to(torch.float32) * 0.03125
    return (values + rank * 0.125 + replay * 2.0).to(dtype)


def call_all_gather(
    input_tensor: torch.Tensor,
    ctx: IrisCommContext,
    output: torch.Tensor,
) -> torch.Tensor:
    if HAS_AG_OUTPUT:
        return all_gather(input_tensor, ctx, output=output)
    return all_gather(input_tensor, ctx)


def call_reduce_scatter(
    input_tensor: torch.Tensor,
    ctx: IrisCommContext,
    output: torch.Tensor,
) -> torch.Tensor:
    if HAS_RS_OUTPUT:
        return reduce_scatter(input_tensor, ctx, output=output)
    return reduce_scatter(input_tensor, ctx)


def assert_exact(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    if actual.dtype == torch.bfloat16:
        equal = torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    elif actual.dtype == torch.float32:
        equal = torch.equal(actual.view(torch.int32), expected.view(torch.int32))
    else:
        equal = torch.equal(actual, expected)
    if not equal:
        mismatch = actual != expected
        raise AssertionError(
            f"{name} has {int(mismatch.sum().item())} wrong values out of "
            f"{actual.numel()}"
        )


def expected_all_gather(
    rows: int,
    columns: int,
    dtype: torch.dtype,
    replay: int,
    device: torch.device,
) -> torch.Tensor:
    return torch.cat(
        [
            make_input(rows, columns, dtype, rank, replay, device)
            for rank in range(WORLD_SIZE)
        ],
        dim=0,
    )


def expected_reduce_scatter(
    rank: int,
    replay: int,
    device: torch.device,
) -> torch.Tensor:
    rows, columns, dtype = RS_SHAPE
    inputs = [
        make_input(rows, columns, dtype, source, replay, device)
        for source in range(WORLD_SIZE)
    ]
    reduced = sum(value.float() for value in inputs).to(dtype)
    return reduced.chunk(WORLD_SIZE, dim=0)[rank]


def check_outputs(
    ag_outputs: list[torch.Tensor],
    rs_output: torch.Tensor,
    rank: int,
    replay: int,
    device: torch.device,
) -> None:
    for output, (name, rows, columns, dtype) in zip(ag_outputs, AG_SHAPES):
        assert_exact(
            output,
            expected_all_gather(rows, columns, dtype, replay, device),
            f"{name} AllGather replay {replay}",
        )
    torch.testing.assert_close(
        rs_output,
        expected_reduce_scatter(rank, replay, device),
        rtol=0.02,
        atol=0.02,
    )


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"This test requires {WORLD_SIZE} ranks")

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", device_id=device)

    with IrisCommContext(heap_size=128 << 20) as ctx:
        ag_inputs = [
            ctx.iris_ctx.empty((rows, columns), dtype=dtype)
            for _, rows, columns, dtype in AG_SHAPES
        ]
        rs_input = ctx.iris_ctx.empty(RS_SHAPE[:2], dtype=RS_SHAPE[2])
        ag_outputs = [
            ctx.iris_ctx.empty((rows * WORLD_SIZE, columns), dtype=dtype)
            for _, rows, columns, dtype in AG_SHAPES
        ]
        rs_output = ctx.iris_ctx.empty(
            (RS_SHAPE[0] // WORLD_SIZE, RS_SHAPE[1]), dtype=RS_SHAPE[2]
        )

        local_base = int(ctx.get_heap_bases()[rank].item())
        local_offsets = torch.tensor(
            [tensor.data_ptr() - local_base for tensor in [*ag_inputs, rs_input]],
            dtype=torch.int64,
            device=device,
        )
        gathered_offsets = torch.empty(
            world_size * local_offsets.numel(), dtype=torch.int64, device=device
        )
        dist.all_gather_into_tensor(gathered_offsets, local_offsets)
        offset_rows = gathered_offsets.reshape(world_size, -1)
        if not torch.all(offset_rows == offset_rows[0]):
            raise AssertionError(f"Iris heap offsets differ: {offset_rows.cpu().tolist()}")

        for tensor, (_, rows, columns, dtype) in zip(ag_inputs, AG_SHAPES):
            tensor.copy_(make_input(rows, columns, dtype, rank, 0, device))
        rs_input.copy_(
            make_input(*RS_SHAPE, rank=rank, replay=0, device=device)
        )
        eager_ag = [
            call_all_gather(input_tensor, ctx, output)
            for input_tensor, output in zip(ag_inputs, ag_outputs)
        ]
        eager_rs = call_reduce_scatter(rs_input, ctx, rs_output)
        torch.cuda.synchronize()
        check_outputs(eager_ag, eager_rs, rank, 0, device)

        legacy_ag = all_gather(ag_inputs[1], ctx)
        legacy_rs = reduce_scatter(rs_input, ctx)
        torch.cuda.synchronize()
        assert_exact(
            legacy_ag,
            expected_all_gather(4, 6, torch.float32, 0, device),
            "legacy AllGather",
        )
        torch.testing.assert_close(
            legacy_rs,
            expected_reduce_scatter(rank, 0, device),
            rtol=0.02,
            atol=0.02,
        )

        zero_input = ctx.iris_ctx.empty((0, 4096), dtype=torch.bfloat16)
        zero_output = ctx.iris_ctx.empty((0, 4096), dtype=torch.bfloat16)
        gathered_zero = call_all_gather(zero_input, ctx, zero_output)
        if gathered_zero.shape != (0, 4096):
            raise AssertionError(f"Unexpected zero-token shape: {gathered_zero.shape}")

        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for input_tensor, output in zip(ag_inputs, ag_outputs):
                call_all_gather(input_tensor, ctx, output)
            call_reduce_scatter(rs_input, ctx, rs_output)
        capture_stream.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            graph_ag = [
                call_all_gather(input_tensor, ctx, output)
                for input_tensor, output in zip(ag_inputs, ag_outputs)
            ]
            graph_rs = call_reduce_scatter(rs_input, ctx, rs_output)

        for replay in range(1, 4):
            with torch.cuda.stream(capture_stream):
                for tensor, (_, rows, columns, dtype) in zip(ag_inputs, AG_SHAPES):
                    tensor.copy_(
                        make_input(rows, columns, dtype, rank, replay, device)
                    )
                rs_input.copy_(
                    make_input(*RS_SHAPE, rank=rank, replay=replay, device=device)
                )
                graph.replay()
            capture_stream.synchronize()
            check_outputs(graph_ag, graph_rs, rank, replay, device)

        if rank == 0:
            print(
                "IRIS_AGRS_TEST_RESULT="
                + json.dumps(
                    {
                        "all_gather_dtypes": ["bfloat16", "float32", "int32"],
                        "graph_replays": 3,
                        "heap_offsets": offset_rows[0].cpu().tolist(),
                        "reduce_scatter": "bfloat16_sum",
                        "status": "pass",
                        "uniform_zero_tokens": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
