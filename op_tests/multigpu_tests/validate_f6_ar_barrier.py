# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""TP4 validation of the gfx1250 two-stage fused AR + RMSNorm barriers.

Run only on four idle gfx1250 GPUs, under an outer timeout:

    python op_tests/multigpu_tests/validate_f6_ar_barrier.py
    python op_tests/multigpu_tests/validate_f6_ar_barrier.py --stress-steps 3000 --near-wrap

Default run: for M=2517 and 4536 at hidden 6144, compare the two-stage path
bitwise with the one-stage fixed-rank-order path, in eager calls under rank
arrival skew and in HIP-graph replays, and check that every rank holds the
same bits.

--stress-steps N adds N back-to-back eager calls whose row counts are drawn
at random from the two-stage range (342..5400 rows), quantized or not, each
checked bitwise against the one-stage path; then graphs that each hold
several random-size calls, replayed with fresh inputs. --near-wrap first sets
every barrier epoch of every rank to just below 2^32 so the run wraps them.
--check-kernels profiles one call and expects start_sync_kernel twice,
ar_rmsnorm_reduce_scatter_gfx1250 and ar_rmsnorm_gfx1250 once and no
end_sync_kernel (needs a profiler that records device kernels).

A process-pool timeout turns a barrier hang into a failing run.
"""

import argparse
import hashlib
import json
import logging
import random
import statistics
import time
from multiprocessing import TimeoutError, get_context

import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

LOG = logging.getLogger("aiter.f6")
TP = 4
HIDDEN = 6144
DEFAULT_ROWS = (2517, 4536)
# Two-stage at TP4, bf16, hidden 6144: above the 4 MiB LL limit and within the
# communicator's 64 MiB input buffer.
STRESS_ROWS = (342, 5400)
EPS = 1e-6
EXPECTED_KERNELS = {
    "start_sync_kernel": 2,
    "ar_rmsnorm_reduce_scatter_gfx1250": 1,
    "ar_rmsnorm_gfx1250": 1,
    "end_sync_kernel": 0,
}
# Signal layout (custom_all_reduce_gfx1250.cuh): start, end: uint32[512][64];
# _flag: uint32[512]; each array 128-byte aligned.
SIGNAL_EPOCH_WORDS = 512 * 64 * 2 + 512
NEAR_WRAP_EPOCH = 0xFFFFFF00


def _is_gfx1250(device: torch.device) -> bool:
    return "gfx1250" in torch.cuda.get_device_properties(device).gcnArchName


def _skew(rank: int, step: int) -> None:
    # Vary host arrival so barriers see ranks in different orders.
    time.sleep(((rank * 17 + step * 29) % 8) * 0.0002)


def _buffers(rows: int, quant: bool, device: torch.device):
    shape = (rows, HIDDEN)
    residual_out = torch.empty(shape, dtype=torch.bfloat16, device=device)
    if quant:
        return (
            torch.empty(shape, dtype=dtypes.fp8, device=device),
            residual_out,
            torch.empty((rows, 1), dtype=torch.float32, device=device),
        )
    return torch.empty(shape, dtype=torch.bfloat16, device=device), residual_out


def _call(ca, x, residual, weight, buffers, *, use_1stage: bool, quant: bool):
    kwargs = {
        "res_out": buffers[1],
        "out": buffers[0],
        "w": weight,
        "eps": EPS,
        "registered": False,
        "use_1stage": use_1stage,
        "post_per_token_quant": quant,
        "gemma_norm": True,
    }
    if quant:
        kwargs["scale_out"] = buffers[2]
    result = ca.fused_ar_rms(x, residual, **kwargs)
    assert all(actual is expected for actual, expected in zip(result, buffers))
    return result


def _assert_bitwise(label: str, expected, actual) -> None:
    assert len(expected) == len(actual)
    for index, (want, got) in enumerate(zip(expected, actual)):
        want_bits = want.contiguous().view(torch.uint8)
        got_bits = got.contiguous().view(torch.uint8)
        if not torch.equal(want_bits, got_bits):
            mismatches = torch.count_nonzero(want_bits != got_bits).item()
            raise AssertionError(f"{label} tensor {index} differs in {mismatches} bytes")


def _digest(tensors) -> str:
    value = hashlib.sha256()
    for tensor in tensors:
        value.update(tensor.contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return value.hexdigest()


def _assert_all_rank_bits(label: str, tensors) -> None:
    digests = [None] * TP
    dist.all_gather_object(digests, _digest(tensors))
    if len(set(digests)) != 1:
        raise AssertionError(f"{label} differs across ranks: {digests}")


def _capture(ca, fn):
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with ca.capture(), torch.cuda.graph(graph, stream=stream):
        outputs = fn()
    torch.cuda.current_stream().wait_stream(stream)
    dist.barrier()
    return graph, outputs


def _set_epochs(ca, value: int) -> None:
    """Set every start/end/_flag barrier epoch of this rank to `value`.

    Every rank must call this while no collective is in flight anywhere; all
    ranks then agree that each slot's last barrier had epoch `value`.
    """
    import ctypes

    import numpy as np

    from aiter.jit.utils.hip_runtime import load_hip_runtime

    meta = ca._pool["meta"]
    ptr = meta.data_ptr() if callable(meta.data_ptr) else meta.data_ptr
    host = np.full(SIGNAL_EPOCH_WORDS, value, dtype=np.uint32)
    torch.cuda.synchronize()
    err = load_hip_runtime().hipMemcpy(
        ctypes.c_void_p(ptr),
        ctypes.c_void_p(host.ctypes.data),
        ctypes.c_size_t(host.nbytes),
        ctypes.c_int(1),  # hipMemcpyHostToDevice
    )
    assert err == 0, f"hipMemcpy returned {err}"
    torch.cuda.synchronize()


def _profile_kernel_count(fn) -> dict[str, int]:
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    names = [event.name for event in prof.events()]
    counts = {k: sum(k in name for name in names) for k in EXPECTED_KERNELS}
    if counts != EXPECTED_KERNELS:
        raise AssertionError(f"kernel count {counts}, expected {EXPECTED_KERNELS}")
    return counts


def _inputs(rows: int, device: torch.device, rank: int, seed: int):
    common = torch.Generator(device=device).manual_seed(seed)
    own = torch.Generator(device=device).manual_seed(seed + 1000 + rank)
    shape = (rows, HIDDEN)
    residual = torch.empty(shape, dtype=torch.bfloat16, device=device)
    x = torch.empty(shape, dtype=torch.bfloat16, device=device)
    residual.uniform_(-0.25, 0.25, generator=common)
    x.uniform_(-0.25, 0.25, generator=own)
    return x, residual, common, own


def _run_case(rank, ca, rows, quant, eager_iters, graph_iters, device, check_kernels):
    x, residual, common, own = _inputs(rows, device, rank, 393_000 + rows)
    weight = torch.empty(HIDDEN, dtype=torch.bfloat16, device=device)
    weight.uniform_(-0.1, 0.1, generator=common)

    eager_ref = _buffers(rows, quant, device)
    eager_fused = _buffers(rows, quant, device)
    for step in range(eager_iters):
        residual.uniform_(-0.25, 0.25, generator=common)
        x.uniform_(-0.25, 0.25, generator=own)
        _skew(rank, step)
        ref = _call(ca, x, residual, weight, eager_ref, use_1stage=True, quant=quant)
        fused = _call(ca, x, residual, weight, eager_fused, use_1stage=False, quant=quant)
        torch.cuda.synchronize()
        _assert_bitwise(f"eager M={rows} step={step}", ref, fused)
    _assert_all_rank_bits(f"eager M={rows}", eager_fused)

    graph_ref_buf = _buffers(rows, quant, device)
    graph_fused_buf = _buffers(rows, quant, device)
    ref_graph, ref_out = _capture(
        ca,
        lambda: _call(ca, x, residual, weight, graph_ref_buf, use_1stage=True, quant=quant),
    )
    fused_graph, fused_out = _capture(
        ca,
        lambda: _call(ca, x, residual, weight, graph_fused_buf, use_1stage=False, quant=quant),
    )
    for step in range(graph_iters):
        residual.uniform_(-0.25, 0.25, generator=common)
        x.uniform_(-0.25, 0.25, generator=own)
        _skew(rank, eager_iters + step)
        ref_graph.replay()
        fused_graph.replay()
        torch.cuda.synchronize()
        _assert_bitwise(f"graph M={rows} replay={step}", ref_out, fused_out)
    _assert_all_rank_bits(f"graph M={rows}", fused_out)

    result = {"rows": rows, "quant": quant, "eager": eager_iters, "graph": graph_iters}
    if check_kernels:
        dist.barrier()
        result["kernels"] = _profile_kernel_count(
            lambda: _call(ca, x, residual, weight, eager_fused, use_1stage=False, quant=quant)
        )
    dist.barrier()
    return result


def _stress(rank, ca, steps, device, batch=8, graph_calls=6, graph_replays=200):
    """Random-size back-to-back calls, each checked against the one-stage path."""
    rng = random.Random(393_006)  # same plan on every rank
    lo, hi = STRESS_ROWS
    seed = 393_500
    x_pool, r_pool, common, own = _inputs(hi, device, rank, seed)
    weight = torch.empty(HIDDEN, dtype=torch.bfloat16, device=device)
    weight.uniform_(-0.1, 0.1, generator=common)
    out = {q: [_buffers(hi, q, device) for _ in range(batch)] for q in (False, True)}
    ref = {q: [_buffers(hi, q, device) for _ in range(batch)] for q in (False, True)}

    def view(bufs, rows):
        return tuple(b[:rows] for b in bufs)

    checked = 0
    for start in range(0, steps, batch):
        plan = [(rng.randint(lo, hi), rng.random() < 0.5) for _ in range(min(batch, steps - start))]
        x_pool.uniform_(-0.25, 0.25, generator=own)
        r_pool.uniform_(-0.25, 0.25, generator=common)
        _skew(rank, start // batch)
        fused = []
        for i, (rows, quant) in enumerate(plan):
            # Distinct row offsets so consecutive calls read different data.
            off = (i * 37) % (hi - rows + 1)
            x, residual = x_pool[off : off + rows], r_pool[off : off + rows]
            fused.append(
                _call(ca, x, residual, weight, view(out[quant][i], rows), use_1stage=False, quant=quant)
            )
        refs = []
        for i, (rows, quant) in enumerate(plan):
            off = (i * 37) % (hi - rows + 1)
            x, residual = x_pool[off : off + rows], r_pool[off : off + rows]
            refs.append(
                _call(ca, x, residual, weight, view(ref[quant][i], rows), use_1stage=True, quant=quant)
            )
        torch.cuda.synchronize()
        for i, (rows, quant) in enumerate(plan):
            _assert_bitwise(f"stress step={start + i} M={rows} quant={quant}", refs[i], fused[i])
            checked += 1
        if (start // batch) % 50 == 0:
            _assert_all_rank_bits(f"stress step={start}", fused[-1])

    # Graphs holding several random-size calls, replayed with fresh inputs.
    graphs_checked = 0
    for g in range(3):
        plan = [(rng.randint(lo, hi), rng.random() < 0.5) for _ in range(graph_calls)]

        def body(use_1stage, bufs):
            outs = []
            for i, (rows, quant) in enumerate(plan):
                off = (i * 53 + g) % (hi - rows + 1)
                outs.append(
                    _call(
                        ca,
                        x_pool[off : off + rows],
                        r_pool[off : off + rows],
                        weight,
                        view(bufs[quant][i], rows),
                        use_1stage=use_1stage,
                        quant=quant,
                    )
                )
            return outs

        fused_graph, fused_out = _capture(ca, lambda: body(False, out))
        ref_graph, ref_out = _capture(ca, lambda: body(True, ref))
        for rep in range(graph_replays):
            x_pool.uniform_(-0.25, 0.25, generator=own)
            r_pool.uniform_(-0.25, 0.25, generator=common)
            if rep % 20 == 0:
                _skew(rank, rep)
            fused_graph.replay()
            fused_graph.replay()  # back-to-back replays of the same graph
            ref_graph.replay()
            torch.cuda.synchronize()
            for i in range(graph_calls):
                _assert_bitwise(f"graph {g} replay {rep} call {i}", ref_out[i], fused_out[i])
            graphs_checked += 1
        _assert_all_rank_bits(f"graph {g}", fused_out[-1])
        del fused_graph, ref_graph
    dist.barrier()
    return {"stress_calls_checked": checked, "graph_replays_checked": graphs_checked}


def _worker(rank, init_method, rows, eager_iters, graph_iters, stress_steps, near_wrap, check_kernels):
    from aiter.dist.device_communicators.custom_all_reduce import CustomAllreduce

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    if not _is_gfx1250(device):
        raise RuntimeError(f"rank {rank}: F6 validation requires gfx1250")
    dist.init_process_group("gloo", world_size=TP, rank=rank, init_method=init_method)
    ca = CustomAllreduce(dist.group.WORLD, device)
    if ca is None or ca.disabled:
        raise RuntimeError(f"rank {rank}: gfx1250 custom all-reduce is disabled")
    try:
        if near_wrap:
            torch.cuda.synchronize()
            dist.barrier()
            _set_epochs(ca, NEAR_WRAP_EPOCH)
            dist.barrier()
        results = []
        for m in rows:
            for quant in (False, True):
                results.append(
                    _run_case(rank, ca, m, quant, eager_iters, graph_iters, device, check_kernels)
                )
        if stress_steps:
            results.append(_stress(rank, ca, stress_steps, device))
        torch.cuda.synchronize()
        dist.barrier()
        return {"rank": rank, "results": results}
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        ca.close()
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", nargs="+", type=int, default=list(DEFAULT_ROWS))
    parser.add_argument("--eager-iters", type=int, default=20)
    parser.add_argument("--graph-iters", type=int, default=100)
    parser.add_argument("--stress-steps", type=int, default=0)
    parser.add_argument("--near-wrap", action="store_true")
    parser.add_argument("--check-kernels", action="store_true")
    parser.add_argument(
        "--timeout",
        type=int,
        default=1200,
        help="whole TP4 run timeout in seconds; expiry is treated as a deadlock",
    )
    parser.add_argument("--json", default="", help="optional summary output path")
    args = parser.parse_args()
    if any(m <= 0 for m in args.rows):
        raise ValueError("--rows must be positive")

    init_method = get_distributed_init_method(get_ip(), get_open_port())
    ctx = get_context("spawn")
    pool = ctx.Pool(processes=TP)
    t0 = time.time()
    try:
        pending = pool.starmap_async(
            _worker,
            [
                (
                    rank,
                    init_method,
                    tuple(args.rows),
                    args.eager_iters,
                    args.graph_iters,
                    args.stress_steps,
                    args.near_wrap,
                    args.check_kernels,
                )
                for rank in range(TP)
            ],
        )
        per_rank = pending.get(timeout=args.timeout)
        pool.close()
        pool.join()
    except TimeoutError as error:
        pool.terminate()
        pool.join()
        raise RuntimeError(f"TP4 F6 validation exceeded {args.timeout}s; possible deadlock") from error
    except BaseException:
        pool.terminate()
        pool.join()
        raise

    summary = {
        "tp": TP,
        "hidden": HIDDEN,
        "args": vars(args),
        "seconds": round(time.time() - t0, 1),
        "ranks": per_rank,
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    print("PASS")
    if args.json:
        with open(args.json, "w") as output:
            output.write(text + "\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
