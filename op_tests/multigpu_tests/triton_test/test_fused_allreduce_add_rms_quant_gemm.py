# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Test fused allreduce + rmsnorm + FP8 quant + scaled GEMM.

Compares the fused (iris) implementation against the reference (NCCL)
implementation for correctness in both eager and CUDA graph modes.

Usage:
    pytest test_fused_allreduce_add_rms_quant_gemm.py -v -s
    pytest test_fused_allreduce_add_rms_quant_gemm.py -k "graph and M4"
"""

import logging
import multiprocessing as mp
import os
import socket
import sys
import traceback

import pytest
import torch
import torch.distributed as dist

from aiter.test_common import checkAllclose, ensure_spawn_method
from aiter.ops.triton.comms.fused_allreduce_add_rms_quant_gemm import (
    fused_allreduce_add_rms_quant_gemm,
)
from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype

logger = logging.getLogger("aiter")

FP8_DTYPE = get_fp8_e4m3_dtype()


def _find_free_port():
    """Find a free TCP port by binding to port 0."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _run_on_gpu(
    tp_size, gpu_id, M, N, dtype, use_residual, impl, graph, num_runs, input_type
):
    """Run one variant on one GPU.

    The worker is impl-agnostic: `impl` is passed straight through to
    `fused_allreduce_add_rms_quant_gemm`.

    Returns list of (gemm_cpu, res_cpu) tuples of length `num_runs`.
    """
    try:
        device = torch.device(f"cuda:{gpu_id}")
        torch.cuda.set_device(device)
        torch.set_default_device(device)

        from aiter.dist.parallel_state import (
            init_distributed_environment,
            ensure_model_parallel_initialized,
            set_custom_all_reduce,
        )

        set_custom_all_reduce(True)
        init_distributed_environment(
            world_size=tp_size,
            rank=gpu_id,
            distributed_init_method="env://",
        )
        ensure_model_parallel_initialized(tp_size, 1)

        # Weights (same seed on all GPUs)
        torch.manual_seed(42)
        rms_weight = torch.ones(N, dtype=dtype, device=device)
        gemm_weight = (
            torch.rand(
                N,
                N,
                dtype=torch.float32,
                device=device,
            )
            .to(FP8_DTYPE)
            .contiguous()
            .t()
        )
        weight_scale = torch.tensor(
            1.0,
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)

        def make_input(input_seed, residual_seed):
            if input_type == "ones":
                inp = torch.ones(M, N, dtype=dtype, device=device)
                res = (
                    torch.ones(M, N, dtype=dtype, device=device)
                    if use_residual
                    else None
                )
            else:
                torch.manual_seed(input_seed)
                inp = torch.randn(M, N, dtype=dtype, device=device)
                res = None
                if use_residual:
                    # Residual must be the same on all ranks (matches
                    # vLLM contract: residual = hidden_states from
                    # before the TP layer, identical post-allreduce).
                    torch.manual_seed(residual_seed)
                    res = torch.randn(M, N, dtype=dtype, device=device)
            return inp, res

        def run(inp, res):
            g, r = fused_allreduce_add_rms_quant_gemm(
                inp,
                rms_weight,
                1e-6,
                FP8_DTYPE,
                "",
                gemm_weight,
                weight_scale,
                dtype,
                residual=res,
                impl=impl,
            )
            torch.cuda.synchronize()
            return g.cpu(), r.cpu() if r is not None else None

        def input_seed_i(i):
            return 1000 + gpu_id * num_runs + i

        def residual_seed_i(i):
            return 5000 + i  # same on all GPUs

        if not graph:
            results = []
            for i in range(num_runs):
                inp, res = make_input(input_seed_i(i), residual_seed_i(i))
                gemm, res_out = run(inp, res)
                results.append((gemm, res_out))
            return results

        # Graph mode: warmup + capture on a dedicated stream
        capture_stream = torch.cuda.Stream()

        with torch.cuda.stream(capture_stream):
            for wi in range(3):
                inp, res = make_input(200 + gpu_id, 5200)
                run(inp, res)
        capture_stream.synchronize()

        # Capture
        input_cap, res_cap = make_input(300 + gpu_id, 5300)
        cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(cuda_graph, stream=capture_stream):
            cap_gemm, cap_res = fused_allreduce_add_rms_quant_gemm(
                input_cap,
                rms_weight,
                1e-6,
                FP8_DTYPE,
                "",
                gemm_weight,
                weight_scale,
                dtype,
                residual=res_cap,
            )

        # Replay
        results = []
        for i in range(num_runs):
            inp, res = make_input(input_seed_i(i), residual_seed_i(i))

            with torch.cuda.stream(capture_stream):
                input_cap.copy_(inp)
                if res_cap is not None and res is not None:
                    res_cap.copy_(res)
                cuda_graph.replay()
            capture_stream.synchronize()

            gemm = cap_gemm.clone().cpu()
            res_out = cap_res.clone().cpu() if cap_res is not None else None
            results.append((gemm, res_out))

        return results

    except Exception as e:
        logger.error(
            f"\n-->[Error on GPU {gpu_id}]: {str(e)}\n"
            f"-->[Traceback]: {''.join(traceback.format_exception(*sys.exc_info()))}"
        )
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_pool(
    tp_size, impl, M, N, dtype, use_residual, graph, num_runs, input_type, port
):
    """Launch one variant across all GPUs via mp.Pool."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    ensure_spawn_method()

    pool = mp.Pool(processes=tp_size)
    async_results = []
    for i in range(tp_size):
        async_results.append(
            pool.apply_async(
                _run_on_gpu,
                args=(
                    tp_size,
                    i,
                    M,
                    N,
                    dtype,
                    use_residual,
                    impl,
                    graph,
                    num_runs,
                    input_type,
                ),
            )
        )
    pool.close()
    pool.join()
    return [r.get() for r in async_results]


@pytest.mark.parametrize("use_residual", [False, True])
@pytest.mark.parametrize("N", [8192])
@pytest.mark.parametrize("M", [1, 4, 32, 128, 512, 576, 1024])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mode", ["eager", "graph"])
@pytest.mark.parametrize("tp_size", [8])
@pytest.mark.parametrize("graph_replays", [5])
@pytest.mark.parametrize("input_type", ["rand"])
def test_correctness(
    M, N, dtype, use_residual, mode, tp_size, graph_replays, input_type
):
    """Compare fused (iris) vs ref (NCCL) for correctness.

    Each variant runs in its own process pool to avoid contamination.

    In eager mode, runs both variants once with the same input.
    In graph mode, iris captures a CUDA graph and replays multiple times.
    Ref always runs eagerly. Both use deterministic inputs so results
    match. Graph mode verifies barrier epochs advance correctly across
    repeated replays.
    """
    if torch.cuda.device_count() < tp_size:
        pytest.skip(f"Need {tp_size} GPUs, only {torch.cuda.device_count()} available")
    graph = mode == "graph"
    num_runs = graph_replays if graph else 1

    # Ref always runs eager (NCCL can't be graph-captured)
    ref_results = _run_pool(
        tp_size,
        "ref",
        M,
        N,
        dtype,
        use_residual,
        graph=False,
        num_runs=num_runs,
        input_type=input_type,
        port=_find_free_port(),
    )
    iris_results = _run_pool(
        tp_size,
        "iris",
        M,
        N,
        dtype,
        use_residual,
        graph=graph,
        num_runs=num_runs,
        input_type=input_type,
        port=_find_free_port(),
    )
    failures = []
    for gpu_id in range(tp_size):
        ref_gpu = ref_results[gpu_id]
        iris_gpu = iris_results[gpu_id]
        assert len(ref_gpu) == len(iris_gpu)

        for check_i, ((ref_gemm, ref_res), (fused_gemm, fused_res)) in enumerate(
            zip(ref_gpu, iris_gpu)
        ):
            tag = f"replay {check_i}" if graph else "eager"

            # Check residual first (isolates allreduce correctness)
            if use_residual:
                msg = f"GPU {gpu_id}, {tag}: Residual, M={M}, N={N}"
                err = checkAllclose(
                    ref_res,
                    fused_res,
                    msg=msg,
                    atol=1e-3,
                    rtol=1e-3,
                )
                if err != 0:
                    failures.append(msg)

            msg = (
                f"GPU {gpu_id}, {tag}: GEMM, " f"M={M}, N={N}, residual={use_residual}"
            )
            err = checkAllclose(
                ref_gemm,
                fused_gemm,
                msg=msg,
                atol=6,
                rtol=1e-1,
            )
            if err != 0:
                failures.append(msg)

    assert not failures, f"{len(failures)} checks failed:\n" + "\n".join(failures)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
