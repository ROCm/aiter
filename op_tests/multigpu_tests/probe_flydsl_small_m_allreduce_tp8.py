# SPDX-License-Identifier: MIT
"""Compare the standalone FlyDSL small-M BF16 AR with custom AllReduce."""

from __future__ import annotations

import argparse
import json
import statistics

import flydsl.expr as fx
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from mori.cco import Communicator

from aiter.dist.parallel_state import get_tp_group
from aiter.ops.flydsl.kernels.comm_fused_moe import small_m_allreduce
from aiter.ops.flydsl.kernels.comm_fused_moe.sync import FLAT_VA_RANK_STRIDE
from aiter.ops.flydsl.kernels.tensor_shim import ptr_arg
from aiter.ops.flydsl.moe_kernels import _run_compiled
from op_tests.multigpu_tests import moe_tp_stage2_test_utils as fixtures


TP = 8
H = 7168


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=(1, 2))
    parser.add_argument("--threads", type=int, default=512, choices=(256, 512))
    parser.add_argument("--block-limit", type=int)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    return parser.parse_args()


def register_windows(tp_group, rank: int, tensors):
    uid = Communicator.get_unique_id() if rank == 0 else None
    comm = Communicator.init(
        TP,
        rank,
        tp_group.broadcast_object(uid),
        per_rank_vmm=FLAT_VA_RANK_STRIDE,
    )
    windows = tuple(
        comm.register_external_window(tensor.data_ptr(), tensor.nbytes)
        for tensor in tensors
    )
    bases = tuple(
        window.local_ptr - rank * FLAT_VA_RANK_STRIDE for window in windows
    )
    return comm, windows, bases


def rank_max(local_us: float, group) -> float:
    local = torch.tensor([local_us], dtype=torch.float64, device="cuda")
    values = torch.empty(TP, dtype=torch.float64, device="cuda")
    dist.all_gather_into_tensor(values, local, group=group)
    return float(values.max().item())


def capture(body, group):
    fixtures.barrier(group)
    result = body()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with get_tp_group().graph_capture() as capture_context:
        with torch.cuda.graph(graph, stream=capture_context.stream):
            result = body()
    for _ in range(3):
        graph.replay()
    fixtures.barrier(group)
    return graph, result


def measure(graph, rounds: int, iterations: int, group):
    samples = []
    for _ in range(rounds):
        fixtures.barrier(group)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(
            rank_max(start.elapsed_time(end) * 1000.0 / iterations, group)
        )
    return {
        "median_rank_max_us": statistics.median(samples),
        "min_rank_max_us": min(samples),
        "max_rank_max_us": max(samples),
        "samples_us": samples,
    }


def error(actual, expected, group):
    diff = actual.float() - expected.float()
    values = torch.stack(
        (
            diff.abs().max(),
            diff.norm() / expected.float().norm().clamp_min(1.0e-12),
        )
    )
    dist.all_reduce(values, op=dist.ReduceOp.MAX, group=group)
    return {"max_abs": float(values[0]), "rel_l2": float(values[1])}


def fill_input(tensor: torch.Tensor, rank: int) -> None:
    values = torch.arange(
        tensor.numel(), dtype=torch.float32, device=tensor.device
    ).reshape_as(tensor)
    values = (values.remainder(127) - 63.0) / 64.0 + (rank + 1) / 32.0
    tensor.copy_(values.to(torch.bfloat16))


def main() -> None:
    cli = parse_args()
    rank, _world, _local_rank, device, group = fixtures.setup_distributed(TP)
    tp_group = get_tp_group()
    ca_comm = tp_group.device_communicator.ca_comm
    if ca_comm is None or ca_comm.disabled:
        raise RuntimeError("custom AllReduce is unavailable")
    keepalive = []
    results = []

    for m in cli.tokens:
        input_buffer = symm_mem.empty(
            (m, H), dtype=torch.bfloat16, device=device
        )
        state = symm_mem.empty(
            (
                small_m_allreduce.state_bytes(
                    m, H, cli.threads, cli.block_limit
                ),
            ),
            dtype=torch.uint8,
            device=device,
        )
        output = torch.empty_like(input_buffer)
        state.zero_()
        fill_input(input_buffer, rank)

        comm, windows, bases = register_windows(
            tp_group, rank, (input_buffer, state)
        )
        keepalive.append((comm, windows))
        input_flat_base, state_flat_base = bases
        launcher = small_m_allreduce.compile_bf16_one_stage(
            m, H, cli.threads, cli.block_limit
        )

        def flydsl_body():
            stream = torch.cuda.current_stream(device)
            _run_compiled(
                launcher,
                (
                    fx.Int64(input_flat_base),
                    ptr_arg(state),
                    fx.Int64(state_flat_base),
                    ptr_arg(output),
                    rank,
                    stream,
                ),
            )
            return output

        custom_input = input_buffer.clone()
        if not ca_comm.should_custom_ar(custom_input):
            raise RuntimeError(f"custom AllReduce rejected M={m} BF16 input")

        def custom_body():
            result = ca_comm.custom_all_reduce(
                custom_input, use_new=True, open_fp8_quant=False
            )
            if result is None:
                raise RuntimeError("custom AllReduce unexpectedly returned None")
            return result

        gathered = torch.empty(
            (TP * m, H), dtype=torch.bfloat16, device=device
        )
        dist.all_gather_into_tensor(gathered, input_buffer, group=group)
        expected = gathered.view(TP, m, H).float().sum(dim=0).to(torch.bfloat16)

        flydsl_body()
        custom_result = custom_body()
        torch.cuda.synchronize(device)
        flydsl_error = error(output, expected, group)
        custom_error = error(custom_result, expected, group)
        if flydsl_error["max_abs"] != 0.0 or custom_error["max_abs"] != 0.0:
            raise AssertionError(
                f"M={m} AllReduce mismatch: "
                f"flydsl={flydsl_error}, custom={custom_error}"
            )

        flydsl_graph, flydsl_result = capture(flydsl_body, group)
        custom_graph, custom_result = capture(custom_body, group)
        flydsl_timing = measure(
            flydsl_graph, cli.rounds, cli.iterations, group
        )
        custom_timing = measure(
            custom_graph, cli.rounds, cli.iterations, group
        )
        torch.cuda.synchronize(device)
        graph_errors = {
            "flydsl": error(flydsl_result, expected, group),
            "custom": error(custom_result, expected, group),
        }
        result = {
            "m": m,
            "bytes": input_buffer.nbytes,
            "blocks": small_m_allreduce.grid_blocks(
                m, H, cli.threads, cli.block_limit
            ),
            "threads": cli.threads,
            "custom_fully_connected": bool(ca_comm.fully_connected),
            "flydsl": flydsl_timing,
            "custom": custom_timing,
            "ratio": (
                flydsl_timing["median_rank_max_us"]
                / custom_timing["median_rank_max_us"]
            ),
            "errors": graph_errors,
        }
        results.append(result)
        if rank == 0:
            print(
                "FLYDSL_SMALL_M_AR_RESULT " + json.dumps(result, sort_keys=True),
                flush=True,
            )

        del flydsl_graph, custom_graph, custom_input, output, state, input_buffer
        fixtures.barrier(group)

    if rank == 0:
        print(
            "FLYDSL_SMALL_M_AR_SUMMARY " + json.dumps(results, sort_keys=True),
            flush=True,
        )
    keepalive.clear()
    fixtures.cleanup_distributed(rank)


if __name__ == "__main__":
    main()
