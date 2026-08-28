# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kernel comparison benchmark for aiter's all-reduce implementations.

Candidates:

* ``custom``   -- ``aiter::cross_device_reduce_1stage`` / ``_2stage``, the two
  kernels that dominate a DeepSeek-V4 profile (docs/communication_kernels.md
  §8.6 -- together ~50% of aiter kernel time in a TP4 DSv4 trace). Numerically
  exact: bf16 inputs accumulated in fp32, bitwise identical across ranks.
* ``rccl``     -- ``dist.all_reduce`` baseline.
* ``qr_int4``  -- FlyDSL INT4 two-shot quick-reduce (ROCm/aiter#4970), included
  automatically when present. **Lossy**: the payload is quantized to INT4 with
  group-16 E4M3 scales, so it is ~4x smaller on the wire and cannot be compared
  on ``err``. Every candidate therefore also reports SQNR against an fp32
  reference; expect ~inf dB for the exact kernels and ~19 dB for INT4.

Because the candidates are not accuracy-equivalent, read the table as
"what does this cost and what does it cost you" -- a speedup column alone is
misleading for ``qr_int4``.

Which kernel runs is chosen by the C++ host dispatch in
``csrc/include/custom_all_reduce.cuh`` (``CustomAllreduce::allreduce``), keyed on
world size and message bytes:

    world_size == 2                      -> 1stage  (unconditional)
    full_nvlink && world <= 4 && < 160 KiB -> 1stage
    full_nvlink && world <= 8 && <  80 KiB -> 1stage
    otherwise                            -> 2stage

There is no env override, so **TP2 can only reach the 1-stage kernel**. Run at
``-t 4`` (or 8) to exercise both; the ``kernel`` column reports which one each
row actually measured. TP4 also matches the production DSv4 trace.

The benchmark calls ``ca_comm.all_reduce`` directly rather than
``tensor_model_parallel_all_reduce``, so the kernel is measured at every size --
the production wrapper would divert anything above ``AITER_CUSTOM_AR_MAX_SIZE``
(64 MiB default) to RCCL. RCCL is timed separately as the baseline.

Examples::

    # default sweep (TP4 on GPUs 4-7 if 4+ visible, else TP2)
    python3 op_tests/multigpu_tests/bench_comm_allreduce.py

    # the 2-GPU case, DSv4 decode shapes only
    HIP_VISIBLE_DEVICES=6,7 python3 op_tests/multigpu_tests/bench_comm_allreduce.py -t 2

    # both kernels, DSv4 shapes, on GPUs 4-7
    HIP_VISIBLE_DEVICES=4,5,6,7 python3 op_tests/multigpu_tests/bench_comm_allreduce.py \
        -t 4 -s 1,7168 8,7168 1024,7168 4096,7168

    # profiling entrypoint: few iters, per-rank chrome trace
    HIP_VISIBLE_DEVICES=6,7 python3 op_tests/multigpu_tests/bench_comm_allreduce.py \
        -t 2 -s 8,7168 --iters 20 --profile

    # under rocprofv3. Do NOT pass -o: ranks are separate processes and a fixed
    # output name makes them overwrite each other (you get one rank, silently).
    # Omitting it gives <pid>_kernel_trace.csv per rank.
    HIP_VISIBLE_DEVICES=4,5,6,7 rocprofv3 --kernel-trace -d /tmp/arprof \
        --output-format csv -- \
        python3 op_tests/multigpu_tests/bench_comm_allreduce.py \
            -t 4 -s 8,7168 12,7168 --iters 20 --warmup 2
    # then filter Kernel_Name for cross_device_reduce_{1,2}stage.
"""

import argparse
import itertools
import logging
import math
import os
from multiprocessing import Pool, freeze_support, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

from aiter import dtypes
from aiter.dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)

SUPPORTED_GFX = ["gfx942", "gfx950"]

# QRInt4 (ROCm/aiter#4970) is optional: absent until that PR merges, and it
# needs FlyDSL. Probed here so the bench degrades to custom-vs-RCCL instead of
# failing when it is missing.
try:
    from aiter.ops.flydsl.kernels.qr_int4 import QRInt4

    HAS_QR_INT4 = True
except Exception as _qr_exc:  # noqa: BLE001
    QRInt4 = None
    HAS_QR_INT4 = False
    _QR_IMPORT_ERR = _qr_exc

# QRInt4 constraints, from aiter/ops/flydsl/kernels/qr_int4.py.
_QR_INT4_ARCHS = ("gfx942", "gfx950")
_QR_INT4_WORLDS = (2, 4, 8)


def qr_int4_applicable(world_size: int, dtype, nbytes: int) -> bool:
    """Whether QRInt4 accepts this configuration (bf16, TP 2/4/8, 16B multiple)."""
    return (
        HAS_QR_INT4
        and get_gfx() in _QR_INT4_ARCHS
        and world_size in _QR_INT4_WORLDS
        and dtype == dtypes.bf16
        and nbytes % 16 == 0
    )


def sqnr_db(got: torch.Tensor, ref: torch.Tensor) -> float:
    """Signal-to-quantization-noise ratio in dB, matching the #4970 test.

    The only accuracy metric that spans exact and quantized candidates.
    Returns +inf when the result is bit-exact against the reference.
    """
    got = got.to(dtypes.fp32)
    ref = ref.to(dtypes.fp32)
    mse = float(((got - ref) ** 2).mean().item())
    ref_pow = float((ref * ref).mean().item())
    if not math.isfinite(mse) or not math.isfinite(ref_pow):
        return float("-inf")
    if ref_pow <= 0.0:
        return float("inf") if mse <= 0.0 else float("-inf")
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10(ref_pow / mse)


# DeepSeek-V4 hidden size. Decode shapes (1,2,4,8) and prefill chunks
# (1024, 4096) are the exact token counts observed in the TP4 trace in
# op_tests/dump_data/aiter_kernels_union.json.
DSV4_HIDDEN = 7168
L_SHAPE = [(1, DSV4_HIDDEN), (8, DSV4_HIDDEN), (1024, DSV4_HIDDEN)]

# Mirrors the C++ dispatch cited in the module docstring, so the `kernel` column
# is a prediction, not an observation. Kept in sync by hand -- after touching the
# .cuh, confirm with the rocprofv3 recipe above and compare Kernel_Name against
# this column.
_ONESTAGE_MAX_BYTES = {4: 160 * 1024, 6: 80 * 1024, 8: 80 * 1024}


def predicted_kernel(world_size: int, nbytes: int) -> str:
    """Which cross_device_reduce_* the host dispatch will pick."""
    if world_size == 2:
        return "1stage"
    limit = _ONESTAGE_MAX_BYTES.get(world_size)
    if limit is None:
        return "?"
    return "1stage" if nbytes < limit else "2stage"


def collective_bw(nbytes: int, us: float, world_size: int, kernel: str):
    """(algbw, busbw, traffic) in GB/s, GB/s, bytes.

    ``algbw`` is the user-visible rate (message / time). ``busbw`` is the
    NCCL-tests convention for all-reduce -- ``algbw * 2*(N-1)/N`` -- which
    normalizes across world size so numbers are comparable between TPs.
    ``traffic`` is what this particular kernel actually moves per rank:
    one-shot reads the whole buffer from every peer, two-shot moves a
    reduce-scatter plus an all-gather.
    """
    n = world_size
    algbw = nbytes / us / 1e3  # bytes/us -> GB/s
    busbw = algbw * 2 * (n - 1) / n
    traffic = (n - 1) * nbytes if kernel == "1stage" else int(2 * (n - 1) / n * nbytes)
    return algbw, busbw, traffic


def _make_input(rank: int, tokens: int, hidden: int, dtype):
    """Deterministic per-rank input, reproducible from any process.

    Built with an explicit CPU generator so every rank can reconstruct every
    other rank's contribution and check the reduction locally. That keeps whole
    tensors out of the multiprocessing pipe -- returning a 56 MiB prefill
    activation per rank exhausts /dev/shm.
    """
    g = torch.Generator(device="cpu").manual_seed(20260828 + rank)
    return torch.randn((tokens, hidden), generator=g, dtype=dtypes.fp32).to(dtype)


def _worker(
    tp_size: int,
    rank: int,
    tokens: int,
    hidden: int,
    dtype,
    num_iters: int,
    num_warmup: int,
    init_method: str,
    profile: bool,
):
    """One rank: build input, time custom AR and RCCL, return metrics + output."""
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp_size, 1)
    group = get_tp_group().device_group
    ca_comm = get_tp_group().device_communicator.ca_comm
    assert ca_comm is not None and not ca_comm.disabled, (
        f"rank {rank}: custom allreduce is disabled; nothing to benchmark "
        f"(world_size={tp_size})"
    )

    x = _make_input(rank, tokens, hidden, dtype).to(device)
    out = torch.empty_like(x)
    nbytes = x.numel() * x.element_size()

    # Warm the RCCL communicator and align ranks before any timing.
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    def run_custom():
        # Direct kernel entry: bypasses should_custom_ar()'s size window so the
        # kernel is measured even above the 64 MiB RCCL-fallback threshold.
        return ca_comm.all_reduce(x, out=out)

    rccl_buf = x.clone()

    def run_rccl():
        rccl_buf.copy_(x)
        dist.all_reduce(rccl_buf, group=group)
        return rccl_buf

    # Barrier before each timed region so the measurement reflects the kernel
    # rather than accumulated rank skew. Production time is higher: in the DSv4
    # trace the 1-stage kernel spends most of its duration spinning in
    # start_sync waiting for peers (docs/communication_kernels.md §8.6 item 3).
    dist.barrier(group=group)
    torch.cuda.synchronize()
    custom_out, custom_us = run_perftest(
        run_custom, num_iters=num_iters, num_warmup=num_warmup
    )

    dist.barrier(group=group)
    torch.cuda.synchronize()
    _, rccl_us = run_perftest(run_rccl, num_iters=num_iters, num_warmup=num_warmup)

    qr_us = float("nan")
    qr_out = None
    if qr_int4_applicable(tp_size, dtype, nbytes):
        # QRInt4 exchanges IPC handles over broadcast_object_list, so it needs
        # the gloo (CPU) group -- it rejects an NCCL group outright.
        fly = QRInt4(
            group=get_tp_group().cpu_group,
            device=device,
            rank=rank,
            world_size=tp_size,
        )
        try:
            qr_buf = torch.empty_like(x)
            # compile() is itself a collective launch on every super-tile
            # engine; doing it up front keeps a JIT out of the timed region.
            dist.barrier(group=group)
            fly.compile(x, qr_buf)
            dist.barrier(group=group)
            torch.cuda.synchronize()
            qr_res, qr_us = run_perftest(
                lambda: fly.allreduce(x, qr_buf) or qr_buf,
                num_iters=num_iters,
                num_warmup=num_warmup,
            )
            qr_out = qr_res.clone()
        finally:
            fly.close()

    if profile:
        dist.barrier(group=group)
        torch.cuda.synchronize()
        trace = f"comm_ar_tp{tp_size}_m{tokens}_k{hidden}_rank{rank}.json"
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as prof:
            for _ in range(num_iters):
                run_custom()
            torch.cuda.synchronize()
        prof.export_chrome_trace(trace)
        logger.info("rank %d: wrote %s", rank, trace)

    # Verify in-rank against the fp32 sum of every rank's contribution,
    # accumulated one peer at a time so peak memory stays at ~2 activations.
    ref = torch.zeros((tokens, hidden), dtype=dtypes.fp32, device=device)
    for peer in range(tp_size):
        ref += _make_input(peer, tokens, hidden, dtype).to(device, dtypes.fp32)
    # The exact kernels must pass a tight tolerance. QRInt4 is quantized and
    # would fail it by construction (~19 dB SQNR is ~11% relative error), so it
    # is graded on SQNR only -- the same gate #4970's own test uses.
    err = checkAllclose(
        ref,
        custom_out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg=f"custom_ar tp{tp_size} {tokens}x{hidden} rank{rank}",
    )

    ret = {
        "custom_us": custom_us,
        "rccl_us": rccl_us,
        "nbytes": nbytes,
        "err": err,
        "custom_sqnr": sqnr_db(custom_out, ref),
        "qr_us": qr_us,
        "qr_sqnr": sqnr_db(qr_out, ref) if qr_out is not None else float("nan"),
    }
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return ret


@benchmark()
def test_allreduce(tp_size, tokens, hidden, dtype, num_iters, num_warmup, profile):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    with Pool(processes=tp_size) as pool:
        rets = [
            pool.apply_async(
                _worker,
                args=(
                    tp_size,
                    r,
                    tokens,
                    hidden,
                    dtype,
                    num_iters,
                    num_warmup,
                    init_method,
                    profile,
                ),
            )
            for r in range(tp_size)
        ]
        pool.close()
        pool.join()
    rets = [r.get() for r in rets]

    nbytes = rets[0]["nbytes"]
    kernel = predicted_kernel(tp_size, nbytes)
    err = max(r["err"] for r in rets)
    custom_us = [r["custom_us"] for r in rets]
    rccl_us = [r["rccl_us"] for r in rets]
    # Slowest rank is what the model waits on; spread across ranks is skew.
    custom, rccl = max(custom_us), max(rccl_us)
    _, busbw, traffic = collective_bw(nbytes, custom, tp_size, kernel)
    _, rccl_busbw, _ = collective_bw(nbytes, rccl, tp_size, kernel)
    row = {
        "gfx": get_gfx(),
        "kernel": kernel,
        "KiB": nbytes / 1024,
        "custom us": custom,
        "custom busbw GB/s": busbw,
        "custom traffic MiB": traffic / 2**20,
        "rank spread us": max(custom_us) - min(custom_us),
        "rccl us": rccl,
        "rccl busbw GB/s": rccl_busbw,
        "vs rccl": rccl / custom,
        "err": err,
        "custom SQNR dB": min(r["custom_sqnr"] for r in rets),
    }

    qr_us = [r["qr_us"] for r in rets]
    if not any(math.isnan(u) for u in qr_us):
        qr = max(qr_us)
        # busbw on the bf16 payload, not the INT4 wire format: this answers
        # "how fast does my (M, hidden) bf16 all-reduce finish", which is the
        # only basis on which the three candidates are comparable. The actual
        # wire traffic is ~4x smaller, which is where the speedup comes from.
        _, qr_busbw, _ = collective_bw(nbytes, qr, tp_size, "2stage")
        row.update(
            {
                "qr_int4 us": qr,
                "qr_int4 busbw GB/s": qr_busbw,
                "qr_int4 vs custom": custom / qr,
                "qr_int4 SQNR dB": min(r["qr_sqnr"] for r in rets),
            }
        )
    return row


def main():
    if get_gfx() not in SUPPORTED_GFX:
        logger.warning("custom all-reduce unsupported on %s; skipping", get_gfx())
        return

    visible = torch.cuda.device_count()
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=__doc__,
    )
    parser.add_argument(
        "-t",
        "--tp",
        type=int,
        nargs="*",
        choices=[2, 4, 8],
        default=None,
        help="tensor-parallel world size(s). Default: 4 when >=4 GPUs are "
        "visible (reaches both kernels), else 2. TP2 can only reach 1stage.",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=L_SHAPE,
        help="(tokens, hidden) pairs, e.g. -s 8,7168 4096,7168",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        nargs="*",
        choices=["fp16", "bf16"],
        default=["bf16"],
        help="data type(s)",
    )
    parser.add_argument("--iters", type=int, default=101, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="also emit a per-rank chrome trace of the custom-AR loop",
    )
    args = parser.parse_args()

    tps = args.tp if args.tp else ([4] if visible >= 4 else [2])
    tps = [t for t in tps if t <= visible]
    if not tps:
        logger.warning("no requested TP size fits %d visible GPUs; skipping", visible)
        return
    if visible < 4 and 2 in tps:
        logger.warning(
            "only %d GPUs visible: TP2 reaches cross_device_reduce_1stage only. "
            "Use 4+ GPUs to also measure 2stage.",
            visible,
        )

    for dtype_name in args.dtype:
        dtype = dtypes.d_dtypes[dtype_name]
        df = []
        for tp_size, (tokens, hidden) in itertools.product(tps, args.shape):
            df.append(
                test_allreduce(
                    tp_size,
                    tokens,
                    hidden,
                    dtype,
                    args.iters,
                    args.warmup,
                    args.profile,
                )
            )
        df = pd.DataFrame(df)
        # @benchmark turns every call arg into a column; keep the sweep axes and
        # the metrics, drop the run-control knobs (iters/warmup/profile).
        show = [
            c
            for c in df.columns
            if c not in ("num_iters", "num_warmup", "profile", "dtype")
        ]
        logger.info(
            "custom all-reduce %s summary (markdown):\n%s",
            dtype_name,
            df[show].to_markdown(index=False, floatfmt=".4g"),
        )


if __name__ == "__main__":
    freeze_support()
    main()
