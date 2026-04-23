#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
#
# Reproducer: RCCL all-gather overlapped with Triton grouped GEMM slowdown on ROCm.
#
# Usage:
#   torchrun --nproc_per_node=2 bench_overlap.py
#   torchrun --nproc_per_node=8 bench_overlap.py --grid-dims 128,256 --ag-size-mb 128
#   torchrun --nproc_per_node=2 bench_overlap.py --profile

import argparse
import os
import sys
import statistics

import torch
import torch.distributed as dist
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton JIT helpers (inlined from AITER pid_preprocessing.py)
# ---------------------------------------------------------------------------

@triton.jit
def remap_xcd(pid, GRID_MN, NUM_XCDS: tl.constexpr = 8):
    """Redistribute program IDs across XCDs for better L2 cache utilization."""
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
    if xcd < tall_xcds:
        pid = xcd * pids_per_xcd + local_pid
    else:
        pid = (
            tall_xcds * pids_per_xcd
            + (xcd - tall_xcds) * (pids_per_xcd - 1)
            + local_pid
        )
    return pid


@triton.jit
def pid_grid(pid: int, num_pid_m: int, num_pid_n: int, GROUP_SIZE_M: tl.constexpr = 1):
    """Map 1D pid to 2D grid coordinates (pid_m, pid_n)."""
    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@triton.jit
def _remap_xcd_tile_grid(
    tile_in_mm, num_row_tiles, num_col_tiles,
    GROUP_SIZE: tl.constexpr = 1,
    NUM_XCDS: tl.constexpr = 8,
):
    return pid_grid(
        remap_xcd(tile_in_mm, num_row_tiles * num_col_tiles, NUM_XCDS=NUM_XCDS),
        num_row_tiles,
        num_col_tiles,
        GROUP_SIZE_M=GROUP_SIZE,
    )


# ---------------------------------------------------------------------------
# Triton grouped GEMM kernel (persistent, based on AITER gmm_kernel)
# ---------------------------------------------------------------------------

@triton.jit
def grouped_gemm_kernel(
    lhs_ptr, rhs_ptr, group_sizes_ptr, out_ptr,
    M: int, K: int, N: int, G: int,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    GRID_DIM: tl.constexpr,
    NUM_XCDS: tl.constexpr,
):
    tl.assume(M > 0)
    tl.assume(K > 0)
    tl.assume(N > 0)
    tl.assume(G > 0)

    num_n_tiles = tl.cdiv(N, BLOCK_N)
    tile = tl.program_id(0)
    last_mm_tile = 0
    last_m = 0

    for g in range(G):
        m = tl.load(group_sizes_ptr + g)
        num_m_tiles = tl.cdiv(m, BLOCK_M)
        num_tiles = num_m_tiles * num_n_tiles

        while tile >= last_mm_tile and tile < last_mm_tile + num_tiles:
            tile_in_mm = tile - last_mm_tile
            tile_m, tile_n = _remap_xcd_tile_grid(
                tile_in_mm, num_m_tiles, num_n_tiles,
                GROUP_SIZE=GROUP_SIZE, NUM_XCDS=NUM_XCDS,
            )

            offs_lhs_m = (tile_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)) % m
            offs_rhs_n = (tile_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)) % N
            offs_k = tl.arange(0, BLOCK_K).to(tl.int64)

            lhs_ptrs = lhs_ptr + (last_m + offs_lhs_m[:, None]) * K + offs_k[None, :]
            rhs_ptrs = (
                rhs_ptr
                + g.to(tl.int64) * K * N
                + offs_k[:, None] * N
                + offs_rhs_n[None, :]
            )

            acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

            for _k in range(0, tl.cdiv(K, BLOCK_K)):
                lhs_tile = tl.load(lhs_ptrs)
                rhs_tile = tl.load(rhs_ptrs)
                acc += tl.dot(lhs_tile, rhs_tile, input_precision="ieee")
                lhs_ptrs += BLOCK_K
                rhs_ptrs += BLOCK_K * N

            acc = acc.to(out_ptr.type.element_ty)

            offs_out_m = tile_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
            offs_out_n = tile_n.to(tl.int64) * BLOCK_N + tl.arange(0, BLOCK_N)
            out_ptrs = out_ptr + (last_m + offs_out_m[:, None]) * N + offs_out_n[None, :]
            tl.store(
                out_ptrs, acc,
                mask=(offs_out_m[:, None] < m) & (offs_out_n[None, :] < N),
            )

            tile += GRID_DIM

        last_mm_tile += num_tiles
        last_m += m


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def run_grouped_gemm_triton(lhs, rhs, group_sizes, out, M, K, N, G, grid_dim, num_xcds):
    num_n_tiles = triton.cdiv(N, 128)
    num_m_tiles_per_group = (group_sizes + 128 - 1) // 128
    total_tiles = int((num_m_tiles_per_group * num_n_tiles).sum().item())
    num_programs = min(grid_dim, total_tiles)
    grouped_gemm_kernel[(num_programs,)](
        lhs, rhs, group_sizes, out,
        M, K, N, G,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=64,
        GROUP_SIZE=1, GRID_DIM=grid_dim, NUM_XCDS=num_xcds,
    )
    return out


def run_grouped_gemm_primus(lhs, rhs, group_lens, out, M, K, N, G, num_cu, _num_xcds):
    from primus_turbo.pytorch.ops import grouped_gemm as pt_grouped_gemm
    return pt_grouped_gemm(lhs, rhs, group_lens, trans_b=_primus_trans_b, num_cu=num_cu)


# Default — overridden by main() based on --backend
run_grouped_gemm = run_grouped_gemm_triton


# ---------------------------------------------------------------------------
# Distributed setup
# ---------------------------------------------------------------------------

def setup_distributed():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, dist.get_world_size(), dist.get_rank()


def cleanup():
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def measure_allgather_alone(ag_input, ag_output, comm_stream, warmup, iters):
    """Measure all-gather time alone for overlap verification."""
    for _ in range(warmup):
        with torch.cuda.stream(comm_stream):
            dist.all_gather_into_tensor(ag_output, ag_input)
        torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        with torch.cuda.stream(comm_stream):
            start_events[i].record(comm_stream)
            dist.all_gather_into_tensor(ag_output, ag_input)
            end_events[i].record(comm_stream)
        torch.cuda.synchronize()
        dist.barrier()

    return [s.elapsed_time(e) for s, e in zip(start_events, end_events)]


def benchmark_gemm_only(lhs, rhs, group_sizes, out, M, K, N, G,
                        grid_dim, num_xcds, warmup, iters):
    """Scenario 1: grouped gemm alone."""
    for _ in range(warmup):
        run_grouped_gemm(lhs, rhs, group_sizes, out, M, K, N, G, grid_dim, num_xcds)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        start_events[i].record()
        run_grouped_gemm(lhs, rhs, group_sizes, out, M, K, N, G, grid_dim, num_xcds)
        end_events[i].record()
        torch.cuda.synchronize()
        dist.barrier()

    return [s.elapsed_time(e) for s, e in zip(start_events, end_events)]


def benchmark_sequential(lhs, rhs, group_sizes, out, M, K, N, G,
                         ag_inputs, ag_outputs, comm_streams, ag_groups,
                         grid_dim, num_xcds, warmup, iters):
    """Scenario 2: all-gather then grouped gemm (sequential)."""
    for _ in range(warmup):
        for ag_in, ag_out, cs, pg in zip(ag_inputs, ag_outputs, comm_streams, ag_groups):
            with torch.cuda.stream(cs):
                dist.all_gather_into_tensor(ag_out, ag_in, group=pg)
        torch.cuda.synchronize()
        run_grouped_gemm(lhs, rhs, group_sizes, out, M, K, N, G, grid_dim, num_xcds)
    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        for ag_in, ag_out, cs, pg in zip(ag_inputs, ag_outputs, comm_streams, ag_groups):
            with torch.cuda.stream(cs):
                dist.all_gather_into_tensor(ag_out, ag_in, group=pg)
        torch.cuda.synchronize()
        start_events[i].record()
        run_grouped_gemm(lhs, rhs, group_sizes, out, M, K, N, G, grid_dim, num_xcds)
        end_events[i].record()
        torch.cuda.synchronize()
        dist.barrier()

    return [s.elapsed_time(e) for s, e in zip(start_events, end_events)]


def benchmark_overlap(lhs, rhs, group_sizes, out, M, K, N, G,
                      ag_inputs, ag_outputs, comm_streams, ag_groups,
                      grid_dim, num_xcds, warmup, iters):
    """Scenario 3: all-gather + grouped gemm overlapped. Returns (gemm_times, wall_times)."""
    compute_stream = torch.cuda.default_stream()

    for _ in range(warmup):
        for ag_in, ag_out, cs, pg in zip(ag_inputs, ag_outputs, comm_streams, ag_groups):
            with torch.cuda.stream(cs):
                dist.all_gather_into_tensor(ag_out, ag_in, group=pg, async_op=True)
        run_grouped_gemm(lhs, rhs, group_sizes, out, M, K, N, G, grid_dim, num_xcds)
        torch.cuda.synchronize()

    gemm_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    gemm_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    wall_start = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    wall_end = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]

    for i in range(iters):
        # Record wall start on compute stream before anything
        wall_start[i].record(compute_stream)

        # Launch all-gathers on separate comm streams with separate PGs
        for ag_in, ag_out, cs, pg in zip(ag_inputs, ag_outputs, comm_streams, ag_groups):
            with torch.cuda.stream(cs):
                dist.all_gather_into_tensor(ag_out, ag_in, group=pg, async_op=True)

        # Launch gemm on compute stream (no wait_stream — run concurrently)
        gemm_start[i].record(compute_stream)
        run_grouped_gemm(lhs, rhs, group_sizes, out, M, K, N, G, grid_dim, num_xcds)
        gemm_end[i].record(compute_stream)

        # Wait for all comm streams to finish
        for cs in comm_streams:
            compute_stream.wait_stream(cs)
        wall_end[i].record(compute_stream)

        torch.cuda.synchronize()
        dist.barrier()

    gemm_times = [s.elapsed_time(e) for s, e in zip(gemm_start, gemm_end)]
    wall_times = [s.elapsed_time(e) for s, e in zip(wall_start, wall_end)]
    return gemm_times, wall_times


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report_results(grid_dim, nccl_max_nchannels, results, ag_alone_times, rank):
    """Print human-readable table (stderr) and CSV (stdout) on rank 0."""
    if rank != 0:
        return

    gemm_only = results["gemm_only"]
    sequential = results["sequential"]
    overlap_gemm = results["overlap_gemm"]
    overlap_wall = results["overlap_wall"]

    baseline_mean = statistics.mean(gemm_only)

    nccl_str = nccl_max_nchannels if nccl_max_nchannels else "default"

    print(f"\n=== GRID_DIM={grid_dim}, NCCL_MAX_NCHANNELS={nccl_str} ===",
          file=sys.stderr)
    print(f"{'Scenario':<20s} | {'Mean (ms)':>10s} | {'Min (ms)':>10s} | "
          f"{'Max (ms)':>10s} | {'Slowdown':>10s}", file=sys.stderr)
    print("-" * 72, file=sys.stderr)

    for name, times in [("Gemm only", gemm_only),
                        ("Sequential", sequential),
                        ("Overlap (gemm)", overlap_gemm)]:
        mean_t = statistics.mean(times)
        min_t = min(times)
        max_t = max(times)
        slowdown = mean_t / baseline_mean if baseline_mean > 0 else float('inf')
        print(f"{name:<20s} | {mean_t:>10.3f} | {min_t:>10.3f} | "
              f"{max_t:>10.3f} | {slowdown:>9.2f}x", file=sys.stderr)

    # Overlap wall clock for verification
    wall_mean = statistics.mean(overlap_wall)
    ag_mean = statistics.mean(ag_alone_times)
    gemm_mean = statistics.mean(gemm_only)
    sum_time = ag_mean + gemm_mean
    overlapped = "YES" if wall_mean < sum_time else "NO"
    print(f"{'Overlap (wall)':<20s} | {wall_mean:>10.3f} | {'':>10s} | "
          f"{'':>10s} | overlap={overlapped}", file=sys.stderr)
    print(f"{'All-gather alone':<20s} | {ag_mean:>10.3f} | {min(ag_alone_times):>10.3f} | "
          f"{max(ag_alone_times):>10.3f} |", file=sys.stderr)
    print(f"  (gemm+ag sum={sum_time:.3f}ms, wall={wall_mean:.3f}ms)", file=sys.stderr)

    # CSV output on stdout
    for name, times in [("gemm_only", gemm_only),
                        ("sequential", sequential),
                        ("overlap_gemm", overlap_gemm),
                        ("overlap_wall", overlap_wall),
                        ("allgather_alone", ag_alone_times)]:
        mean_t = statistics.mean(times)
        min_t = min(times)
        max_t = max(times)
        print(f"{grid_dim},{nccl_str},{name},{mean_t:.4f},{min_t:.4f},{max_t:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RCCL + Triton grouped GEMM overlap slowdown reproducer")
    parser.add_argument("--G", type=int, default=8, help="Number of expert groups")
    parser.add_argument("--M", type=int, default=4096, help="Total tokens (rows)")
    parser.add_argument("--K", type=int, default=4096, help="Hidden dimension")
    parser.add_argument("--N", type=int, default=4096, help="Output dimension")
    parser.add_argument("--ag-size-mb", type=int, default=64,
                        help="All-gather tensor size in MB")
    parser.add_argument("--grid-dims", type=str, default="256",
                        help="Comma-separated GRID_DIM values to sweep")
    parser.add_argument("--num-xcds", type=int, default=8, help="XCD count")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Measurement iterations")
    parser.add_argument("--profile", action="store_true",
                        help="Enable PyTorch profiler trace export")
    parser.add_argument("--num-ag", type=int, default=1,
                        help="Number of concurrent all-gathers to launch during overlap")
    parser.add_argument("--backend", type=str, default="triton",
                        choices=["triton", "primus"],
                        help="Grouped GEMM backend: 'triton' (built-in) or 'primus' (Primus-Turbo CK)")
    parser.add_argument("--trans-b", action="store_true",
                        help="Use transposed weight layout [G, N, K] (Primus backend only)")
    args = parser.parse_args()

    local_rank, world_size, rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")

    G, M, K, N = args.G, args.M, args.K, args.N
    grid_dims = [int(x.strip()) for x in args.grid_dims.split(",")]
    backend = args.backend

    global run_grouped_gemm, _primus_trans_b
    _primus_trans_b = args.trans_b
    if backend == "triton":
        # Ensure K is divisible by BLOCK_K=64
        assert K % 64 == 0, f"K={K} must be divisible by 64"
        assert not args.trans_b, "--trans-b is only supported with --backend primus"
        run_grouped_gemm = run_grouped_gemm_triton
    else:
        run_grouped_gemm = run_grouped_gemm_primus

    nccl_max_nchannels = os.environ.get("NCCL_MAX_NCHANNELS", "")

    if rank == 0:
        print(f"Config: G={G}, M={M}, K={K}, N={N}, world_size={world_size}",
              file=sys.stderr)
        print(f"Backend: {backend}", file=sys.stderr)
        print(f"Grid dims to sweep: {grid_dims}", file=sys.stderr)
        print(f"NCCL_MAX_NCHANNELS={nccl_max_nchannels or 'default'}", file=sys.stderr)
        print(f"All-gather size: {args.ag_size_mb} MB", file=sys.stderr)
        # CSV header
        print("grid_dim,nccl_max_nchannels,scenario,mean_ms,min_ms,max_ms")

    # Allocate tensors
    lhs = torch.randn(M, K, dtype=torch.bfloat16, device=device)
    if args.trans_b:
        rhs = torch.randn(G, N, K, dtype=torch.bfloat16, device=device)
    else:
        rhs = torch.randn(G, K, N, dtype=torch.bfloat16, device=device)
    group_size_val = M // G
    remainder = M % G
    gs_list = [group_size_val] * G
    if remainder > 0:
        gs_list[-1] += remainder
    gs_dtype = torch.int64 if backend == "primus" else torch.int32
    group_sizes = torch.tensor(gs_list, dtype=gs_dtype, device=device)
    out = torch.empty(M, N, dtype=torch.bfloat16, device=device)

    ag_numel = args.ag_size_mb * 1024 * 1024 // 2  # bf16 = 2 bytes
    num_ag = args.num_ag

    # Allocate separate buffers, streams, and process groups for each concurrent all-gather.
    # Separate PGs get independent RCCL communicators, so their collectives run concurrently.
    ag_inputs = [torch.randn(ag_numel, dtype=torch.bfloat16, device=device) for _ in range(num_ag)]
    ag_outputs = [torch.empty(world_size * ag_numel, dtype=torch.bfloat16, device=device) for _ in range(num_ag)]
    comm_streams = [torch.cuda.Stream(device=device) for _ in range(num_ag)]
    if num_ag > 1:
        all_ranks = list(range(world_size))
        ag_groups = [dist.new_group(all_ranks) for _ in range(num_ag)]
    else:
        ag_groups = [None]  # use default PG

    if rank == 0:
        print(f"Concurrent all-gathers: {num_ag} (separate process groups)", file=sys.stderr)

    # Measure all-gather alone (for overlap verification, use first buffer/stream)
    ag_alone_times = measure_allgather_alone(
        ag_inputs[0], ag_outputs[0], comm_streams[0], args.warmup, args.iters)

    for grid_dim in grid_dims:
        if rank == 0:
            print(f"\n--- Running with GRID_DIM={grid_dim} ---", file=sys.stderr)

        if args.profile:
            from torch.profiler import profile, ProfilerActivity
            trace_dir = f"./traces/rank{rank}_grid{grid_dim}"
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(trace_dir),
            ) as prof:
                results = _run_all_scenarios(
                    lhs, rhs, group_sizes, out, M, K, N, G,
                    ag_inputs, ag_outputs, comm_streams, ag_groups,
                    grid_dim, args.num_xcds, args.warmup, args.iters,
                )
            if rank == 0:
                print(f"  Profiler trace saved to {trace_dir}/", file=sys.stderr)
        else:
            results = _run_all_scenarios(
                lhs, rhs, group_sizes, out, M, K, N, G,
                ag_inputs, ag_outputs, comm_streams, ag_groups,
                grid_dim, args.num_xcds, args.warmup, args.iters,
            )

        report_results(grid_dim, nccl_max_nchannels, results, ag_alone_times, rank)

    cleanup()


def _run_all_scenarios(lhs, rhs, group_sizes, out, M, K, N, G,
                       ag_inputs, ag_outputs, comm_streams, ag_groups,
                       grid_dim, num_xcds, warmup, iters):
    gemm_only = benchmark_gemm_only(
        lhs, rhs, group_sizes, out, M, K, N, G,
        grid_dim, num_xcds, warmup, iters)

    sequential = benchmark_sequential(
        lhs, rhs, group_sizes, out, M, K, N, G,
        ag_inputs, ag_outputs, comm_streams, ag_groups,
        grid_dim, num_xcds, warmup, iters)

    overlap_gemm, overlap_wall = benchmark_overlap(
        lhs, rhs, group_sizes, out, M, K, N, G,
        ag_inputs, ag_outputs, comm_streams, ag_groups,
        grid_dim, num_xcds, warmup, iters)

    return {
        "gemm_only": gemm_only,
        "sequential": sequential,
        "overlap_gemm": overlap_gemm,
        "overlap_wall": overlap_wall,
    }


if __name__ == "__main__":
    main()
