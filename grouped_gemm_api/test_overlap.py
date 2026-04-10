"""
Overlap grouped GEMM (default stream) with NCCL all_gather on a second stream.

Runs one process per GPU: init NCCL, then a fixed sequence — 10 cycles of
all_gather only, 10 cycles of grouped_gemm_fprop only, 10 cycles of overlapped
grouped_gemm_fprop (default stream) plus all_gather (second stream) with no
per-iteration sync inside that phase.

PyTorch profiler (Chrome trace): set env TEST_OVERLAP_TRACE_DIR to a directory;
each rank writes overlap_rank{r}.json, then they are merged into overlap_merged.json
(pid/tid offset per rank so all GPUs show as distinct processes; chrome://tracing or
perfetto.dev).
"""

from __future__ import annotations

import json
import os
import socket
import sys

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from grouped_gemm_ops import grouped_gemm_fprop

# Per-rank contribution to all_gather (bfloat16 elements).
ALL_GATHER_NUMEL = 3_564_800  # 222_800 * 16

# Cycles per phase: (1) all_gather only, (2) fprop only, (3) overlapped fprop + all_gather.
PHASE_CYCLES = 10


def _trace_dir() -> str | None:
    d = os.environ.get("TEST_OVERLAP_TRACE_DIR", "").strip()
    return d if d else None


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


# Separate Chrome trace processes per rank after merge (avoid collapsed timelines).
_MERGE_PID_STRIDE = 1_000_000
_MERGE_TID_STRIDE = 65_536


def _load_chrome_trace_events(path: str) -> tuple[list, dict | None]:
    """Return (traceEvents list, optional wrapper dict with other keys)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data, None
    if isinstance(data, dict) and "traceEvents" in data:
        te = data["traceEvents"]
        if not isinstance(te, list):
            raise ValueError(f"{path}: traceEvents is not a list")
        rest = {k: v for k, v in data.items() if k != "traceEvents"}
        return te, rest
    raise ValueError(f"{path}: unsupported Chrome trace JSON (expected list or dict with traceEvents)")


def _offset_chrome_event_pid_tid(event: dict, rank: int) -> dict:
    """Copy event with pid/tid shifted so merged traces do not collide across ranks."""
    e = dict(event)
    pid = e.get("pid")
    if isinstance(pid, int):
        e["pid"] = pid + rank * _MERGE_PID_STRIDE
    tid = e.get("tid")
    if isinstance(tid, int):
        e["tid"] = tid + rank * _MERGE_TID_STRIDE
    return e


def merge_chrome_traces_from_ranks(trace_dir: str, world_size: int, merged_name: str = "overlap_merged.json") -> str:
    """Combine overlap_rank0..overlap_rank{world_size-1}.json into one Chrome trace file.

    Returns path to the merged JSON. Call from the parent process after all ranks have
    finished writing their per-rank traces.
    """
    merged_events: list = []
    wrapper_meta: dict | None = None

    for rank in range(world_size):
        path = os.path.join(trace_dir, f"overlap_rank{rank}.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing per-rank trace: {path}")
        events, wrapper = _load_chrome_trace_events(path)
        if wrapper_meta is None and wrapper is not None:
            wrapper_meta = dict(wrapper)
        merged_events.extend(_offset_chrome_event_pid_tid(ev, rank) if isinstance(ev, dict) else ev for ev in events)

    merged_events.sort(key=lambda e: e.get("ts", 0) if isinstance(e, dict) else 0)

    out_path = os.path.join(trace_dir, merged_name)
    if wrapper_meta is not None:
        wrapper_meta["traceEvents"] = merged_events
        payload = wrapper_meta
    else:
        payload = {"schemaVersion": 1, "traceEvents": merged_events}

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    return out_path


def _worker(rank: int, world_size: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)

    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=world_size)
    device = torch.device(f"cuda:{rank}")

    G, M, N, K = 8, 256, 4096, 2048
    split_sizes = torch.full((G,), M, dtype=torch.int64, device=device)
    x = torch.randn(G * M, K, device=device, dtype=torch.bfloat16)
    w = torch.randn(G, N, K, device=device, dtype=torch.bfloat16)

    comm_stream = torch.cuda.Stream(device=device)

    # Independent payload so all_gather does not depend on GEMM output (safe overlap).
    local = torch.full((ALL_GATHER_NUMEL,), float(rank), device=device, dtype=torch.bfloat16)
    tensor_list = [
        torch.empty(ALL_GATHER_NUMEL, device=device, dtype=torch.bfloat16) for _ in range(world_size)
    ]

    def _run_test_sequence() -> torch.Tensor:
        o: torch.Tensor | None = None

        with torch.profiler.record_function("phase1_all_gather_only"):
            for _ in range(PHASE_CYCLES):
                with torch.profiler.record_function("nccl_all_gather"):
                    with torch.cuda.stream(comm_stream):
                        dist.all_gather(tensor_list, local)
        torch.cuda.current_stream(device).synchronize()
        comm_stream.synchronize()

        with torch.profiler.record_function("phase2_fprop_only"):
            for _ in range(PHASE_CYCLES):
                with torch.profiler.record_function("grouped_gemm_fprop"):
                    o = grouped_gemm_fprop(x, w, split_sizes)
        assert o is not None
        torch.cuda.current_stream(device).synchronize()
        comm_stream.synchronize()

        with torch.profiler.record_function("phase3_overlapped"):
            for _ in range(PHASE_CYCLES):
                with torch.profiler.record_function("grouped_gemm_fprop"):
                    o = grouped_gemm_fprop(x, w, split_sizes)
                with torch.profiler.record_function("nccl_all_gather"):
                    with torch.cuda.stream(comm_stream):
                        dist.all_gather(tensor_list, local)
        return o

    trace_dir = _trace_dir()
    if trace_dir is not None:
        os.makedirs(trace_dir, exist_ok=True)
        trace_path = os.path.join(trace_dir, f"overlap_rank{rank}.json")
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            with_stack=False,
        ) as prof:
            with torch.profiler.record_function("test_sequence_total"):
                out = _run_test_sequence()
        prof.export_chrome_trace(trace_path)
        if rank == 0:
            print(
                f"Wrote per-rank PyTorch Chrome traces under {trace_dir} (overlap_rank0..{world_size - 1}.json); "
                "parent merges to overlap_merged.json after all ranks finish.",
                file=sys.stderr,
            )
    else:
        out = _run_test_sequence()

    torch.cuda.current_stream(device).synchronize()
    comm_stream.synchronize()

    assert out is not None
    assert out.shape == (G * M, N)
    for i, t in enumerate(tensor_list):
        assert t.shape == (ALL_GATHER_NUMEL,)
        assert torch.all(t == float(i)), f"rank {rank}: slice from rank {i} mismatch"

    dist.destroy_process_group()


def test_grouped_gemm_fprop_overlap_nccl_all_gather() -> None:
    world_size = torch.cuda.device_count()
    if world_size < 2:
        pytest.skip("Need at least 2 CUDA devices for multi-GPU NCCL overlap test")

    port = _find_free_port()
    mp.spawn(_worker, args=(world_size, port), nprocs=world_size, join=True)
    trace_dir = _trace_dir()
    if trace_dir is not None:
        merged = merge_chrome_traces_from_ranks(trace_dir, world_size)
        print(f"Merged profiler Chrome trace: {merged}", file=sys.stderr)


if __name__ == "__main__":
    ws = torch.cuda.device_count()
    if ws < 2:
        print("Need at least 2 CUDA devices.", file=sys.stderr)
        sys.exit(0)
    mp.spawn(_worker, args=(ws, _find_free_port()), nprocs=ws, join=True)
    trace_dir = _trace_dir()
    if trace_dir is not None:
        merged = merge_chrome_traces_from_ranks(trace_dir, ws)
        print(f"Merged profiler Chrome trace: {merged}", file=sys.stderr)
    print("OK: grouped_gemm_fprop + NCCL all_gather overlap smoke test passed.")
