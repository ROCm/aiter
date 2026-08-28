# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kernel comparison benchmark for aiter's all-reduce implementations.

Candidates:

* ``aiter_cross_device_reduce`` -- the in-tree HIP kernels
  ``aiter::cross_device_reduce_1stage`` / ``_2stage``, which dominate a
  DeepSeek-V4 profile (docs/communication_kernels.md §8.6 -- together ~50% of
  aiter kernel time in a TP4 DSv4 trace). Numerically exact: bf16 inputs
  accumulated in fp32, bitwise identical across ranks.
* ``rccl`` -- ``dist.all_reduce`` baseline.
* ``qr_int4`` -- FlyDSL INT4 two-shot quick-reduce (ROCm/aiter#4970). **Lossy**:
  the payload is quantized to INT4 with group-16 E4M3 scales, so it is ~4x
  smaller on the wire. bf16 / TP∈{2,4,8} / gfx942 / gfx950 only; skipped
  (columns absent) elsewhere.

The table carries one column group per candidate -- ``us``, ``busbw GB/s``,
``SQNR dB`` -- so they read side by side at each shape. No ratio columns: divide
two ``us`` cells.

Because the candidates are **not accuracy-equivalent**, every one is graded on
SQNR against a common fp32 reference. The exact kernels land at the bf16
rounding floor (~55 dB; RCCL a little lower since it reduces in bf16 rather than
accumulating in fp32), INT4 at ~19 dB. Each has its own floor in
``SQNR_FLOOR_DB`` and the run asserts on it, so a real regression fails
regardless of which accuracy class the candidate is in. Speed alone is a
misleading read for ``qr_int4``.

``busbw`` is always computed on the bf16 payload, including for ``qr_int4``:
the question the table answers is "how fast does my (M, hidden) bf16 all-reduce
finish", not "how efficiently is the wire used".

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

    # save a report to diff against after a kernel change
    HIP_VISIBLE_DEVICES=4,5,6,7 python3 op_tests/multigpu_tests/bench_comm_allreduce.py \
        -t 4 -o /tmp/ar_before.md
    #   ... change the kernel, rebuild, then -o /tmp/ar_after.md and diff the two.

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
import sys
from datetime import datetime
from multiprocessing import Pool, freeze_support, set_start_method
from pathlib import Path

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
from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.test_common import benchmark, checkAllclose, run_perftest

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)

SUPPORTED_GFX = ["gfx942", "gfx950"]

# Candidate key for the in-tree HIP kernels (aiter::cross_device_reduce_1stage /
# _2stage). Named after the kernel symbol so the table column and a rocprofv3
# Kernel_Name line up.
AITER_CDR = "aiter_cross_device_reduce"

# QRInt4 rides on FlyDSL, which aiter itself treats as optional and gates behind
# is_flydsl_available() (it is unavailable on archs outside flydsl's
# SMEM_CAPACITY_MAP). Mirror that gate rather than a bare try/except so this
# bench reports the same availability aiter does.
if is_flydsl_available():
    from aiter.ops.flydsl import QRInt4

    HAS_QR_INT4 = True
else:
    QRInt4 = None
    HAS_QR_INT4 = False

# QRInt4 constraints, from aiter/ops/flydsl/kernels/qr_int4.py.
_QR_INT4_ARCHS = ("gfx942", "gfx950")
_QR_INT4_WORLDS = (2, 4, 8)

# Per-candidate accuracy floor in dB. The exact kernels sit at the bf16 rounding
# floor (~55 dB against an fp32 reference), so 40 dB catches a real regression
# without tripping on rounding. QRInt4 quantizes to INT4; 18 dB is the gate
# #4970's own test uses.
SQNR_FLOOR_DB = {AITER_CDR: 40.0, "rccl": 40.0, "qr_int4": 18.0}
EXACT_CANDIDATES = {AITER_CDR: True, "rccl": True, "qr_int4": False}


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
    """One rank: build the input, time every applicable candidate, return scalars.

    Only scalars cross the process boundary -- returning a 56 MiB prefill
    activation per rank exhausts /dev/shm.
    """
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
    nbytes = x.numel() * x.element_size()

    # Warm the RCCL communicator and align ranks before any timing.
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    # --- candidates -------------------------------------------------------
    # Each entry is a zero-arg thunk returning the all-reduced tensor. Every
    # candidate owns its output buffer so none of them alias.
    candidates = {}
    teardown = []

    cdr_buf = torch.empty_like(x)
    # Direct kernel entry: bypasses should_custom_ar()'s size window, so the
    # kernel is measured even above the 64 MiB RCCL-fallback threshold.
    candidates[AITER_CDR] = lambda: ca_comm.all_reduce(x, out=cdr_buf)

    rccl_buf = torch.empty_like(x)

    def _rccl():
        rccl_buf.copy_(x)
        dist.all_reduce(rccl_buf, group=group)
        return rccl_buf

    candidates["rccl"] = _rccl

    if qr_int4_applicable(tp_size, dtype, nbytes):
        # QRInt4 exchanges IPC handles via broadcast_object_list, so it needs
        # the gloo (CPU) group -- it rejects an NCCL group outright.
        fly = QRInt4(
            group=get_tp_group().cpu_group,
            device=device,
            rank=rank,
            world_size=tp_size,
        )
        teardown.append(fly.close)
        qr_buf = torch.empty_like(x)
        # compile() is itself a collective launch on every super-tile engine;
        # doing it up front keeps a JIT out of the timed region.
        dist.barrier(group=group)
        fly.compile(x, qr_buf)

        def _qr():
            fly.allreduce(x, qr_buf)
            return qr_buf

        candidates["qr_int4"] = _qr

    # --- reference --------------------------------------------------------
    # fp32 sum of every rank's contribution, accumulated one peer at a time so
    # peak memory stays at ~2 activations.
    ref = torch.zeros((tokens, hidden), dtype=dtypes.fp32, device=device)
    for peer in range(tp_size):
        ref += _make_input(peer, tokens, hidden, dtype).to(device, dtypes.fp32)

    # --- time + grade -----------------------------------------------------
    ret = {"nbytes": nbytes}
    try:
        for name, fn in candidates.items():
            # Barrier before each timed region so the measurement reflects the
            # kernel rather than accumulated rank skew. Production time is
            # higher: in the DSv4 trace the 1-stage kernel spends most of its
            # duration spinning in start_sync waiting for peers
            # (docs/communication_kernels.md §8.6 item 3).
            dist.barrier(group=group)
            torch.cuda.synchronize()
            got, us = run_perftest(fn, num_iters=num_iters, num_warmup=num_warmup)

            sqnr = sqnr_db(got, ref)
            floor = SQNR_FLOOR_DB[name]
            assert sqnr >= floor, (
                f"{name} tp{tp_size} {tokens}x{hidden} rank{rank}: "
                f"SQNR {sqnr:.2f} dB below the {floor} dB floor"
            )
            if EXACT_CANDIDATES[name]:
                # Exact candidates additionally get the usual tight tolerance.
                # QRInt4 is skipped here by construction: ~19 dB SQNR is ~11%
                # relative error, so checkAllclose would fail and log a scary
                # failure for a kernel that is behaving exactly as designed.
                checkAllclose(
                    ref,
                    got.to(dtypes.fp32),
                    rtol=1e-2,
                    atol=1e-2,
                    msg=f"{name} tp{tp_size} {tokens}x{hidden} rank{rank}",
                )
            ret[f"{name}_us"] = us
            ret[f"{name}_sqnr"] = sqnr

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
                for fn in candidates.values():
                    for _ in range(num_iters):
                        fn()
                torch.cuda.synchronize()
            prof.export_chrome_trace(trace)
            logger.info("rank %d: wrote %s", rank, trace)
    finally:
        for close in teardown:
            close()

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
    row = {
        "gfx": get_gfx(),
        "kernel": kernel,
        "KiB": nbytes / 1024,
    }

    # One column group per candidate, side by side. busbw is always computed on
    # the bf16 payload -- including for qr_int4, whose wire format is ~4x
    # smaller. That is the only basis on which the candidates are comparable:
    # it answers "how fast does my (M, hidden) bf16 all-reduce finish".
    for name in EXACT_CANDIDATES:
        key = f"{name}_us"
        if key not in rets[0]:
            continue  # candidate not applicable to this config
        per_rank = [r[key] for r in rets]
        # Slowest rank is what the model waits on.
        us = max(per_rank)
        _, busbw, _ = collective_bw(nbytes, us, tp_size, kernel)
        row[f"{name} us"] = us
        row[f"{name} busbw GB/s"] = busbw
        row[f"{name} SQNR dB"] = min(r[f"{name}_sqnr"] for r in rets)
        if name == AITER_CDR:
            # Rank spread is the skew indicator, and skew is a property of the
            # barrier rather than of any one candidate -- reporting it once for
            # the kernel under study keeps the table readable.
            row[f"{AITER_CDR} spread us"] = us - min(per_rank)
    return row


def device_description() -> str:
    """Marketing name plus enough detail to pin the SKU when it is generic.

    ``get_device_name`` is the marketing string the driver reports -- on a
    properly provisioned card that is e.g. "AMD Instinct MI355X", but many
    hosts report a generic "AMD Radeon Graphics". CU count and memory
    disambiguate the SKU in that case (256 CU / 288 GiB is an MI355X), which is
    the whole point of putting it in a provenance header.

    ``pci_device_id`` is deliberately not used: torch reports 0 for it on ROCm,
    so it would look like real provenance while carrying none. ``rocm-smi
    --showproductname`` has the real one (Card Model) if you need it.
    """
    p = torch.cuda.get_device_properties(0)
    return (
        f"{torch.cuda.get_device_name(0)} "
        f"[{p.gcnArchName}, {p.multi_processor_count} CU, "
        f"{p.total_memory / 2**30:.0f} GiB]"
    )


def add_baseline_ratios(df, baseline: str):
    """Append a ``{cand} vs {baseline}`` speedup column per candidate.

    Ratio is ``baseline_us / candidate_us``, so **> 1.0 means the candidate is
    faster than the baseline**. The baseline gets no column of its own (it
    would be 1.0 everywhere). The baseline name is in the column header so a
    saved report stays self-describing when the flag changes.

    Speed is only half the comparison here -- ``qr_int4`` buys its ratio with
    ~36 dB of SQNR. Read the ratio next to the SQNR columns, not alone.
    """
    base_col = f"{baseline} us"
    if base_col not in df.columns:
        logger.warning(
            "baseline %r produced no results (not applicable to any row in "
            "this sweep); skipping speedup columns",
            baseline,
        )
        return df
    for name in EXACT_CANDIDATES:
        col = f"{name} us"
        if name == baseline or col not in df.columns:
            continue
        df[f"{name} vs {baseline}"] = df[base_col] / df[col]
    return df


def _write_report(path, sections, args, visible: int) -> None:
    """Write the summary tables to *path* with enough provenance to diff runs.

    The point of saving a report is comparing a later run against it, so the
    header records everything that changes the numbers: arch, visible GPU
    count, iteration counts, and the exact command. Without those a saved
    table is unfalsifiable.
    """
    stamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [
        "# aiter all-reduce benchmark",
        "",
        f"- generated: {stamp}",
        f"- device: {device_description()}",
        f"- arch: {get_gfx()} ({visible} GPU(s) visible)",
        f"- iters: {args.iters} (warmup {args.warmup})",
        f"- baseline: {args.baseline}",
        f"- qr_int4 available: {HAS_QR_INT4}",
        "",
        "Candidates are not accuracy-equivalent --",
        "read the ratio alongside `SQNR dB`, never alone.",
        "",
    ]
    for dtype_name, table in sections:
        lines += [f"## {dtype_name}", "", table, ""]

    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines))
    logger.info("wrote %s", out)


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
        help="also emit a per-rank chrome trace of every candidate's loop",
    )
    parser.add_argument(
        "-b",
        "--baseline",
        choices=list(EXACT_CANDIDATES),
        default=AITER_CDR,
        help="candidate to measure the others against. Adds a\n"
        "'<cand> vs <baseline>' column per candidate, where > 1.0 means the\n"
        f"candidate is faster than the baseline. Default: {AITER_CDR}, i.e.\n"
        "the table answers 'is anything beating the kernel we ship?'. Pass\n"
        "'rccl' for the 'are we beating the library?' framing instead.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="also write the summary tables to PATH as markdown, with a\n"
        "provenance header (arch, iters, command). Overwrites PATH.",
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

    sections = []
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
        # Ratios are derived from the per-candidate `us` columns, so they are a
        # post-processing step -- appended last, giving one comparison block at
        # the right edge of the table.
        table = add_baseline_ratios(df[show], args.baseline).to_markdown(
            index=False, floatfmt=".4g"
        )
        logger.info("all-reduce %s summary (markdown):\n%s", dtype_name, table)
        sections.append((dtype_name, table))

    if args.output:
        _write_report(args.output, sections, args, visible)


if __name__ == "__main__":
    freeze_support()
    main()
