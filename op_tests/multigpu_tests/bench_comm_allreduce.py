# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kernel comparison benchmark for aiter's all-reduce implementations.

Every implementation reachable from the plain (unfused) all-reduce path is a
candidate here, so one sweep answers "which of the things aiter can do to an
all-reduce is fastest at this shape, and what does it cost in accuracy".

| column     | what runs                              | wire       | exact |
|------------|----------------------------------------|------------|-------|
| ``cdr``    | ``aiter::cross_device_reduce_{1,2}stage`` | bf16/fp16  | yes |
| ``cdr_naive`` | the same call with ``use_new=False`` -> ``*_naive`` kernels | bf16/fp16 | yes |
| ``cdr_fp8``   | ``CustomAllreduce::runFp8QuantKernel``  | fp8      | no  |
| ``qr_fp``     | quick-reduce, no quantization           | fp16     | no  |
| ``qr_fp8``    | quick-reduce, E4M3 codec                | fp8      | no  |
| ``qr_int6``   | quick-reduce, 6-bit codec               | int6     | no  |
| ``qr_int4``   | quick-reduce, 4-bit codec               | int4     | no  |
| ``qr_int3``   | quick-reduce, 3-bit codec (TP2 only)    | int3     | no  |
| ``fly_int4``  | FlyDSL two-shot INT4 (ROCm/aiter#4970)  | int4     | no  |
| ``rccl``      | ``dist.all_reduce``                     | bf16/fp16| yes |

Candidates are skipped (an ``n/a`` cell in the latency/accuracy/busbw/roofline
tables, or no column at all when nothing in the sweep could run them) where
they do not apply:

* ``cdr_fp8`` is **fp16-only and only above 128*2048 elements** -- below that
  ``custom_all_reduce.cu:90`` silently runs the plain kernel instead, so timing
  it there would report the same number twice under two names.
* ``qr_*`` need TP in {2, 4, 8} and fp16/bf16; ``qr_int3`` is additionally
  TP2-only (it is disabled on larger worlds in
  ``quick_all_reduce.py:212-224``).
* ``fly_int4`` needs bf16, TP in {2, 4, 8} and gfx942/gfx950.

The first table printed is the ``summary``: per shape, the fastest candidate
clearing an accuracy floor (``fastest collective``), the fastest bit-accurate
one (``fastest exact collective``), and each one's ratio against the
production path (``prod collective``, ``prod time (us)``). The floor defaults
to ``DEFAULT_MIN_SQNR`` and exists so a codec that is fast only because it
barely transmits anything cannot win the column; ``--min-sqnr`` moves it, and
``fastest collective SQNR dB`` prints what the winning choice actually cost.
The per-candidate latency and accuracy tables below it are the full picture
the summary collapses -- a candidate below the floor still appears there.

Because the candidates are **not accuracy-equivalent**, every one is graded on
SQNR against a common fp32 reference and asserted against its own floor in
``CANDIDATES``, so a real regression fails regardless of which accuracy class
the candidate is in. The exact kernels land at the bf16 rounding floor (~55 dB;
RCCL a little lower since it reduces in bf16 rather than accumulating in fp32),
the quantized ones at their codec's floor. Speed alone is a misleading read for
everything in the "exact = no" rows above -- the table prints ``us`` and
``SQNR dB`` as two aligned tables for exactly that reason.

``busbw`` (``--busbw``) is always computed on the payload dtype, including for
the quantizing candidates whose wire format is several times smaller: the
question the table answers is "how fast does my (M, hidden) all-reduce finish",
not "how efficiently is the wire used".

Every ``us`` is hipEvent wall time bracketing the call, not summed per-kernel
device time, so it includes the peer-wait that dominates the 1-stage kernel.
The torch profiler is not usable here -- see the note in ``_bench_shape``.

Which ``cross_device_reduce_*`` runs is chosen by the C++ host dispatch in
``csrc/include/custom_all_reduce.cuh`` (``CustomAllreduce::allreduce``), keyed on
world size and message bytes. There is no env override, so the only way to reach
a given kernel is to pick a shape:

    use_new=true  (``cdr``)          | use_new=false (``cdr_naive``)
    world == 2            -> 1stage  | world == 2            -> 1stage
    world <= 4, < 160 KiB -> 1stage  | world <= 4, < 512 KiB -> 1stage_naive
    world <= 8, <  80 KiB -> 1stage  | world <= 8, < 256 KiB -> 1stage_naive
    otherwise             -> 2stage  | otherwise             -> 2stage_naive

**TP2 can only reach the 1-stage kernel**; the default shape list straddles both
TP4 boundaries (M=11/12) and the TP8 one (M=5/6). The ``kernel`` / ``naive``
columns report the prediction for each row.

The benchmark calls the kernels directly rather than going through
``tensor_model_parallel_all_reduce``, so each is measured at every size even
where production would not select it. The ``prod path`` column reports what
``CudaCommunicator.all_reduce`` *would* dispatch for that row under the
environment you launched with -- so a row can read ``prod path = rccl`` while
still carrying custom-AR timings. That is the point: it shows what the
production gates leave on the table.

Examples::

    # default sweep: TP4 only, DSv4 shapes plus every dispatch boundary
    python3 op_tests/multigpu_tests/bench_comm_allreduce.py

    # also cover the 1stage-only TP2 case, decode shapes only
    HIP_VISIBLE_DEVICES=6,7 python3 op_tests/multigpu_tests/bench_comm_allreduce.py \
        -tp 2 -s 1,7168 8,7168

    # just the two kernels we ship, against RCCL
    python3 op_tests/multigpu_tests/bench_comm_allreduce.py -c cdr rccl

    # "what is the fastest thing I could actually ship?" -- the summary table's
    # `fastest collective` column, restricted to candidates clearing 25 dB SQNR
    python3 op_tests/multigpu_tests/bench_comm_allreduce.py --min-sqnr 25

    # fp16, where the fp8-quantized custom AR becomes available
    python3 op_tests/multigpu_tests/bench_comm_allreduce.py -d fp16

    # add the fabric ceiling: how much of what TransferBench can move in the
    # same pattern is each candidate actually getting? Needs the TransferBench
    # binary -- see transferbench_roofline.py.
    python3 op_tests/multigpu_tests/bench_comm_allreduce.py --roofline

    # save a report to diff against after a kernel change
    python3 op_tests/multigpu_tests/bench_comm_allreduce.py -o /tmp/ar_before.md
    #   ... change the kernel, rebuild, then -o /tmp/ar_after.md and diff the two.

    # profiling entrypoint: few iters, per-rank chrome trace
    HIP_VISIBLE_DEVICES=6,7 python3 op_tests/multigpu_tests/bench_comm_allreduce.py \
        -tp 2 -s 8,7168 --iters 20 --profile

    # under rocprofv3. Do NOT pass -o: ranks are separate processes and a fixed
    # output name makes them overwrite each other (you get one rank, silently).
    # Omitting it gives <pid>_kernel_trace.csv per rank.
    HIP_VISIBLE_DEVICES=4,5,6,7 rocprofv3 --kernel-trace -d /tmp/arprof \
        --output-format csv -- \
        python3 op_tests/multigpu_tests/bench_comm_allreduce.py \
            -tp 4 -s 8,7168 12,7168 --iters 20 --warmup 2
    # then filter Kernel_Name for cross_device_reduce_{1,2}stage.
"""

import argparse
import logging
import math
import os
import subprocess
import sys
from dataclasses import dataclass
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
from aiter.test_common import checkAllclose, run_perftest

# Sibling module rather than a package import: this directory is not a package,
# and the ranks are spawned (not forked), so they re-import this file and need
# the same path fix.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transferbench_roofline as tbr

logger = logging.getLogger("aiter")

set_start_method("spawn", force=True)

SUPPORTED_GFX = ["gfx942", "gfx950"]

# Quick reduce is gated at *import* time on AITER_QUICK_REDUCE_QUANTIZATION
# naming a valid regime (quick_all_reduce.py:30-38); with the var unset the
# module never probes the JIT and every QuickAllReduce comes back disabled. The
# bench drives the regime per candidate instead, so it forces a valid one into
# the environment before the ranks are spawned (see main()) and remembers what
# the user actually had for the `prod path` column.
_QR_ENV = "AITER_QUICK_REDUCE_QUANTIZATION"
_QR_ENABLING_REGIME = "FP"

# QuickAllReduce constraints, from aiter/dist/device_communicators/quick_all_reduce.py.
_QR_WORLDS = (2, 4, 8)
_QR_DTYPES = (dtypes.fp16, dtypes.bf16)

# QRInt4 (FlyDSL) constraints, from aiter/ops/flydsl/kernels/qr_int4.py.
_FLY_ARCHS = ("gfx942", "gfx950")
_FLY_WORLDS = (2, 4, 8)

# runFp8QuantKernel is only reached for fp16 inputs of at least this many
# elements; below it custom_all_reduce.cu:90 falls through to the plain kernel.
_FP8_MIN_NUMEL = 128 * 2048

# QRInt4 rides on FlyDSL, which aiter itself treats as optional and gates behind
# is_flydsl_available() (it is unavailable on archs outside flydsl's
# SMEM_CAPACITY_MAP). Mirror that gate rather than a bare try/except so this
# bench reports the same availability aiter does.
if is_flydsl_available():
    from aiter.ops.flydsl import QRInt4
    from aiter.ops.flydsl.kernels.qr_int4 import (
        ALGORITHMS,
        MIN_PAYLOAD_BYTES,
        has_xgmi_peer_links,
    )

    _FLY_MIN_BYTES = {n: a.min_bytes for n, a in ALGORITHMS.items()}
    HAS_FLY_INT4 = True
else:
    QRInt4 = None
    MIN_PAYLOAD_BYTES = 0
    _FLY_MIN_BYTES = {}
    has_xgmi_peer_links = None
    HAS_FLY_INT4 = False


def _peer_link_type() -> str:
    """GPU-to-GPU link type for the provenance header.

    Shares QRInt4's KFD probe rather than reimplementing it, so the report can
    never disagree with the dispatch decision the kernel actually made. Says so
    plainly when flydsl is absent and the probe is unavailable, rather than
    guessing -- a wrong link type here would misattribute a whole class of
    performance difference.
    """
    if has_xgmi_peer_links is None:
        return "unknown (flydsl unavailable)"
    return "xGMI" if has_xgmi_peer_links() else "PCIe (no xGMI)"


def _gpu_numa_map() -> str:
    """``cuda:i -> NUMA node`` for every visible GPU, for the provenance header.

    *Which* GPUs a run used is not cosmetic on a multi-socket PCIe host. A GPU
    hangs off one socket's root complex, so a pair on one node reaches each
    other through that socket's switch while a pair spanning nodes also crosses
    the inter-socket link. Picking devices 1,2,3,4 rather than 0,1,2,3 on the
    reference box moves 3 of 6 pairs across that boundary to 4 of 6, and
    measured 21% on the roofline -- a swing large enough that two reports
    without this line are simply not comparable.

    Read from sysfs via the BDF torch reports, since neither torch nor the HIP
    runtime exposes the NUMA node directly. Degrades to a plain "unknown"
    rather than guessing.
    """
    try:
        parts = []
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            bdf = f"{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0"
            node = Path(f"/sys/bus/pci/devices/{bdf}/numa_node")
            parts.append(f"{i}:{node.read_text().strip() if node.exists() else '?'}")
        return ", ".join(parts) if parts else "none"
    except (OSError, AttributeError, RuntimeError):
        return "unknown"


@dataclass(frozen=True)
class Candidate:
    """One thing that can perform the all-reduce, plus how to grade it.

    ``sqnr_floor`` is the accuracy gate in dB, per candidate rather than global:
    an exact kernel and a 3-bit codec cannot share one. ``exact`` additionally
    subjects the candidate to the usual tight ``checkAllclose``; the quantizing
    ones are graded on SQNR alone, since ~19 dB is ~11% relative error and
    ``checkAllclose`` would log a scary failure for a kernel behaving exactly as
    designed.
    """

    key: str
    family: str  # cdr | qr | fly | rccl -- selects the launcher
    sqnr_floor: float
    exact: bool
    quant: str | None = None  # QuickReduceRegime name, family == "qr"
    use_new: bool = True  # family == "cdr"
    fp8: bool = False  # family == "cdr"
    algorithm: str = "two_shot"  # QRInt4 schedule, family == "fly"
    # QRInt4 tuning knobs, family == "fly". None means "leave the constructor
    # default alone"; a value makes this candidate a distinct engine with its
    # own IPC inbox, so two rows can differ only in tuning.
    super_tile: int | None = None
    grid_cap: int | None = None

    @property
    def fly_cfg(self) -> tuple:
        """Identity of the QRInt4 engine this candidate needs."""
        return (self.algorithm, self.super_tile, self.grid_cap)


# Floors sit ~5 dB below what each candidate measures on a healthy gfx950 build
# (the parenthesised bf16 / fp16 numbers), which catches a real regression
# without tripping on rounding. The exact kernels are at the payload dtype's
# rounding floor, not at any property of the collective, which is why they share
# one number. fly_int4 measures 19.2 dB where #4970's own test gates at 18.
CANDIDATES = (
    Candidate("cdr", "cdr", 40.0, True),  # 55 / 73
    Candidate("cdr_naive", "cdr", 40.0, True, use_new=False),  # 55 / 73
    Candidate("cdr_fp8", "cdr", 26.0, False, fp8=True),  # n/a / 33
    Candidate("qr_fp", "qr", 40.0, False, quant="FP"),  # 55 / 69
    Candidate("qr_fp8", "qr", 24.0, False, quant="FP8"),  # 29.5 / 29.5
    Candidate("qr_int6", "qr", 24.0, False, quant="INT6"),  # 30.4 / 30.4
    Candidate("qr_int4", "qr", 14.0, False, quant="INT4"),  # 18.3 / 18.3
    Candidate("qr_int3", "qr", 8.0, False, quant="INT3"),  # 12.2 / 12.2
    Candidate("fly_int4", "fly", 15.0, False),  # 19.2 / n/a
    # Same kernel family, ring schedule. Its floor is lower than fly_int4's
    # because the ring's reduce-scatter lap requantizes N-1 times where two-shot
    # requantizes once; measured 18.7 dB at TP4 (against 19.2), and *better*
    # than two-shot at TP2 (22.2 dB) where the all-gather lap's verbatim
    # forwarding dominates. 14 dB leaves the usual ~5 dB of headroom.
    # Auto: no pinned super_tile, so QRInt4 walks RING_ST_LADDER and picks by
    # payload size at launch. This is what production gets.
    Candidate("fly_int4_ring", "fly", 14.0, False, algorithm="ring"),  # 18.7 / n/a
    # Super-tile variants of the ring with the ladder *disabled* -- pinning
    # super_tile fixes one value for every size. Kept as separate rows so a
    # sweep is one bench run rather than a rebuild, and so `fly_int4_ring`
    # (auto) can be checked against the best pinned row at each shape. ST sets how many tiles a block batches behind
    # one publish; publishes per rank are `num_tiles / ST * 2(N-1)` and are
    # *independent of the block count*, so ST is the only knob that reduces
    # them -- and it pays in parallelism, because `_grid_x` derives the block
    # count from `num_tiles / ST`. Measured on MI350P TP4 bf16 hidden 7168:
    # ST=8 wins at 14 MiB, ST=16 is 1.24x at 56 MiB and 1.27x at 114 MiB, ST=32
    # is slightly behind 16. Accuracy is identical at every ST.
    #
    # Each carries its own grid_cap because the wire buffer is
    # `2(N-1) * grid * (ST * rank_atoms * 1152 + 64)` bytes -- ST=16 at the
    # default cap of 1216 is ~269 MB per rank, against 28 MB at cap 128, and
    # they measure the same (671 vs 673 us).
    Candidate(
        "fly_int4_ring_st8",
        "fly",
        14.0,
        False,
        algorithm="ring",
        super_tile=8,
        grid_cap=128,
    ),
    Candidate(
        "fly_int4_ring_st16",
        "fly",
        14.0,
        False,
        algorithm="ring",
        super_tile=16,
        grid_cap=128,
    ),
    Candidate(
        "fly_int4_ring_st32",
        "fly",
        14.0,
        False,
        algorithm="ring",
        super_tile=32,
        grid_cap=128,
    ),
    Candidate("rccl", "rccl", 40.0, True),  # 51 / 69
)
CANDIDATE_KEYS = [c.key for c in CANDIDATES]
PRIMARY = "cdr"  # the kernel we ship, and the default baseline

# Accuracy floor, in dB, for the summary table's `fastest collective` column.
# Ranked on speed alone it would name the widest-error codec in the sweep at
# nearly every shape -- qr_int3 at ~12 dB is ~25% relative error -- so the
# default excludes the codecs that are fast only because they barely transmit
# anything.
#
# 15 dB is not a judgement about what is shippable; it is the *widest gap*
# between adjacent accuracy classes above, so the default cannot flip a winner
# on measurement wobble:
#
#     12.2  qr_int3          <- excluded
#     ---- 15.0 dB ----      <- 6 dB of clear air, no candidate lands here
#     18.3  qr_int4          <- admitted
#     19.2  fly_int4
#     29.5  qr_fp8
#     30.4  qr_int6
#     33    cdr_fp8 (fp16)
#     51-73 the exact kernels
#
# Raise it with --min-sqnr to ask a real deployment question ("fastest thing
# above 25 dB"); pass 0 to rank on speed alone, which still drops a candidate
# whose error exceeds its signal.
DEFAULT_MIN_SQNR = 15.0


def applicable(cand: Candidate, world_size: int, dtype, numel: int, nbytes: int):
    """Whether *cand* can legally run this configuration.

    Mirrors each implementation's own gate. A candidate that is not applicable
    is not run at all, so its cell reads ``n/a`` in the latency, accuracy,
    busbw and roofline-efficiency tables (or its column is absent when nothing
    in the sweep could run it): ``n/a`` always means "cannot run here", never
    "ran and was slow". A bare ``nan`` elsewhere in those tables means
    something else entirely -- a roofline measurement TransferBench could not
    make, or a summary winner excluded by the accuracy floor -- and is left as
    ``nan`` on purpose so the two are not confused.
    """
    if nbytes % 16 != 0:
        # Every custom path requires 16B-aligned payloads; only RCCL survives.
        return cand.family == "rccl"
    if cand.family == "cdr":
        if cand.fp8:
            return dtype == dtypes.fp16 and numel >= _FP8_MIN_NUMEL
        return True
    if cand.family == "qr":
        if world_size not in _QR_WORLDS or dtype not in _QR_DTYPES:
            return False
        # INT3 on TP4/TP8 is disabled upstream for poor kernel performance, not
        # for correctness -- benchmarking it there would advertise a path
        # production refuses to take.
        return not (cand.quant == "INT3" and world_size != 2)
    if cand.family == "fly":
        # QRInt4.allreduce refuses payloads under its floor. The two schedules
        # have different ones -- two-shot's is an accuracy policy, the ring's a
        # speed one (it only overtakes two-shot past ~10.5 MiB on PCIe) -- so
        # mirror whichever applies rather than letting the call raise.
        floor = _FLY_MIN_BYTES[cand.algorithm]
        return (
            HAS_FLY_INT4
            and get_gfx() in _FLY_ARCHS
            and world_size in _FLY_WORLDS
            and dtype == dtypes.bf16
            and nbytes >= floor
        )
    return True  # rccl


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


# DeepSeek-V4 hidden size. Decode shapes (1, 2, 4, 8) and prefill chunks
# (1024, 4096) are the exact token counts observed in the TP4 trace in
# op_tests/dump_data/aiter_kernels_union.json.
DSV4_HIDDEN = 7168

# Every M below is either a production shape or a dispatch boundary; at bf16 x
# 7168 a token is 14336 B, which is what puts the boundaries where they are:
#   1, 2, 4, 8   DSv4 decode (M=8 is the most frequent 1-stage shape in the trace)
#   5, 6         TP8 1stage/2stage crossover (80 KiB)
#   11, 12       TP4 1stage/2stage crossover (160 KiB)
#   128          mid-size, and the smallest default shape where cdr_fp8 applies
#   1024, 4096   DSv4 prefill chunks
#   4681         the 64 MiB AITER_CUSTOM_AR_MAX_SIZE cutoff, to the token
#   8192         past the cutoff: production diverts to RCCL here, this row
#                measures what that costs
L_SHAPE = [
    (m, DSV4_HIDDEN) for m in (1, 2, 4, 5, 6, 8, 11, 12, 128, 1024, 4096, 4681, 8192)
]

# Mirrors the C++ dispatch cited in the module docstring, so the `kernel` column
# is a prediction, not an observation. Kept in sync by hand -- after touching the
# .cuh, confirm with the rocprofv3 recipe above and compare Kernel_Name against
# this column.
_ONESTAGE_MAX_BYTES = {4: 160 * 1024, 6: 80 * 1024, 8: 80 * 1024}
_ONESTAGE_MAX_BYTES_NAIVE = {4: 512 * 1024, 6: 256 * 1024, 8: 256 * 1024}


def predicted_kernel(world_size: int, nbytes: int, use_new: bool = True) -> str:
    """Which cross_device_reduce_* the host dispatch will pick."""
    if not use_new:
        # The legacy branch keeps the vectorized 1stage kernel at TP2 (only the
        # block count differs from use_new=True) and uses the naive kernels
        # everywhere else, with a wider 1stage window.
        if world_size == 2:
            return "1stage"
        limit = _ONESTAGE_MAX_BYTES_NAIVE.get(world_size)
        if limit is None:
            return "?"
        return "1stage_naive" if nbytes < limit else "2stage_naive"
    if world_size == 2:
        return "1stage"
    limit = _ONESTAGE_MAX_BYTES.get(world_size)
    if limit is None:
        return "?"
    if nbytes < limit:
        return "1stage"
    # The 2stage dispatch drops to the naive kernel when the payload cannot be
    # split into ngpus*16B chunks, and unconditionally on world size 6.
    vectorizable = world_size != 6 and nbytes % (world_size * 16) == 0
    return "2stage" if vectorizable else "2stage_naive"


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
    traffic = (
        (n - 1) * nbytes
        if kernel.startswith("1stage")
        else int(2 * (n - 1) / n * nbytes)
    )
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


def production_path(ca_comm, qr_comm, x, world_size: int, prod_regime: str | None):
    """What CudaCommunicator.all_reduce would dispatch for *x*.

    Evaluated with the *user's* AITER_QUICK_REDUCE_QUANTIZATION, not the one
    this bench forces into the environment to keep the QR candidates alive, so
    the column describes the deployment rather than the benchmark.
    """
    from aiter.dist.device_communicators.quick_all_reduce import QuickReduceRegime

    qr_usable = (
        qr_comm is not None
        and not qr_comm.disabled
        and prod_regime in QuickReduceRegime.__members__
        and prod_regime != "NONE"
        and not (prod_regime == "INT3" and world_size != 2)
    )
    if qr_usable:
        saved = qr_comm.qr_quant_level
        qr_comm.qr_quant_level = QuickReduceRegime[prod_regime]
        try:
            if qr_comm.should_quick_allreduce(x):
                return f"qr:{prod_regime.lower()}"
        finally:
            qr_comm.qr_quant_level = saved
    if ca_comm is not None and not ca_comm.disabled and ca_comm.should_custom_ar(x):
        nbytes = x.numel() * x.element_size()
        return f"cdr:{predicted_kernel(world_size, nbytes)}"
    return "rccl"


def _build_thunks(cands, *, ca_comm, qr_comm, fly, group, x):
    """Zero-arg thunks, one per candidate, each returning the all-reduced tensor.

    Every candidate owns its output buffer so none of them alias, and the QR
    quantization level is set inside the thunk rather than around the timed
    loop, so candidates sharing one QuickAllReduce cannot leak state into each
    other.
    """
    from aiter.dist.device_communicators.quick_all_reduce import QuickReduceRegime

    thunks = {}
    buffers = []
    for cand in cands:
        out = torch.empty_like(x)
        buffers.append(out)
        if cand.family == "cdr":
            # Direct kernel entry: bypasses should_custom_ar()'s size window, so
            # the kernel is measured even above the 64 MiB RCCL-fallback cutoff.
            def _cdr(o=out, c=cand):
                return ca_comm.all_reduce(
                    x, out=o, use_new=c.use_new, open_fp8_quant=c.fp8
                )

            thunks[cand.key] = _cdr
        elif cand.family == "qr":

            def _qr(o=out, lvl=QuickReduceRegime[cand.quant]):
                qr_comm.qr_quant_level = lvl
                return qr_comm.quick_all_reduce(x, out=o)

            thunks[cand.key] = _qr
        elif cand.family == "fly":

            def _fly(o=out, eng=fly[cand.fly_cfg]):
                eng.allreduce(x, o)
                return o

            thunks[cand.key] = _fly
        else:

            def _rccl(o=out):
                o.copy_(x)
                dist.all_reduce(o, group=group)
                return o

            thunks[cand.key] = _rccl
    return thunks, buffers


def _bench_shape(
    *,
    tp_size,
    rank,
    tokens,
    hidden,
    dtype,
    num_iters,
    num_warmup,
    profile,
    group,
    ca_comm,
    qr_comm,
    fly,
    keys,
    prod_regime,
):
    """Time and grade every applicable candidate at one shape. Scalars only."""
    device = torch.device(f"cuda:{rank}")
    x = _make_input(rank, tokens, hidden, dtype).to(device)
    nbytes = x.numel() * x.element_size()

    cands = [
        c
        for c in CANDIDATES
        if c.key in keys
        and applicable(c, tp_size, dtype, x.numel(), nbytes)
        and not (c.family == "qr" and qr_comm is None)
        and not (c.family == "fly" and c.fly_cfg not in fly)
    ]
    thunks, buffers = _build_thunks(
        cands, ca_comm=ca_comm, qr_comm=qr_comm, fly=fly, group=group, x=x
    )

    # fp32 sum of every rank's contribution, accumulated one peer at a time so
    # peak memory stays at ~2 activations.
    ref = torch.zeros((tokens, hidden), dtype=dtypes.fp32, device=device)
    for peer in range(tp_size):
        ref += _make_input(peer, tokens, hidden, dtype).to(device, dtypes.fp32)

    ret = {
        "nbytes": nbytes,
        "prod": production_path(ca_comm, qr_comm, x, tp_size, prod_regime),
    }
    for cand in cands:
        # Barrier before each timed region so the measurement reflects the
        # kernel rather than accumulated rank skew. Production time is higher:
        # in the DSv4 trace the 1-stage kernel spends most of its duration
        # spinning in start_sync waiting for peers
        # (docs/communication_kernels.md §8.6 item 3).
        dist.barrier(group=group)
        torch.cuda.synchronize()
        # hipEvent timing rather than run_perftest's default torch-profiler
        # path. Ranks are spawned children, and once the parent has initialized
        # HIP -- which `import aiter` does at module scope -- some ROCm builds
        # hand the children a profiler that records CPU ops but no GPU activity.
        # That is silent for RCCL (an aten op with 0 device time) and fatal for
        # the custom-AR candidates, which register no aten op at all: the event
        # table comes back empty and get_trace_perf() raises on the missing
        # host_time_sum column. Events are also the honest metric here -- they
        # bracket the whole collective, including the start_sync spin the
        # profiler's per-kernel device time hides.
        got, us = run_perftest(
            thunks[cand.key],
            num_iters=num_iters,
            num_warmup=num_warmup,
            use_cuda_event=True,
        )

        sqnr = sqnr_db(got, ref)
        assert sqnr >= cand.sqnr_floor, (
            f"{cand.key} tp{tp_size} {tokens}x{hidden} rank{rank}: "
            f"SQNR {sqnr:.2f} dB below the {cand.sqnr_floor} dB floor"
        )
        if cand.exact:
            checkAllclose(
                ref,
                got.to(dtypes.fp32),
                rtol=1e-2,
                atol=1e-2,
                msg=f"{cand.key} tp{tp_size} {tokens}x{hidden} rank{rank}",
            )
        ret[f"{cand.key}_us"] = us
        ret[f"{cand.key}_sqnr"] = sqnr

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
            for fn in thunks.values():
                for _ in range(num_iters):
                    fn()
            torch.cuda.synchronize()
        prof.export_chrome_trace(trace)
        logger.info("rank %d: wrote %s", rank, trace)

    # The 8192-token row is 112 MiB per buffer and there are up to ten of them;
    # drop them before the next shape rather than letting the caching allocator
    # hold every shape's working set at once.
    del thunks, buffers, ref, x
    torch.cuda.empty_cache()
    return ret


def _worker(
    tp_size,
    rank,
    shapes,
    dtype,
    num_iters,
    num_warmup,
    init_method,
    profile,
    keys,
    prod_regime,
):
    """One rank: join the group once, then sweep every shape.

    The whole shape list runs inside a single process because the setup this
    amortizes is not small -- an RCCL communicator, the custom-AR IPC pool, the
    quick-reduce IPC buffer and a FlyDSL JIT, per rank. Only scalars cross the
    process boundary: returning a 56 MiB prefill activation per rank exhausts
    /dev/shm.
    """
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp_size, 1)
    tp_group = get_tp_group()
    group = tp_group.device_group
    ca_comm = tp_group.device_communicator.ca_comm
    qr_comm = tp_group.device_communicator.qr_comm
    assert ca_comm is not None and not ca_comm.disabled, (
        f"rank {rank}: custom allreduce is disabled; nothing to benchmark "
        f"(world_size={tp_size})"
    )
    if qr_comm is None or qr_comm.disabled:
        logger.warning(
            "rank %d: quick allreduce is disabled (%s=%s); its columns will be absent",
            rank,
            _QR_ENV,
            os.environ.get(_QR_ENV),
        )
        qr_comm = None

    # Warm the RCCL communicator and align ranks before any timing.
    dist.all_reduce(torch.zeros(1, device=device), group=group)
    torch.cuda.synchronize()

    fly = {}  # QRInt4 schedule name -> engine
    # One engine per distinct (schedule, super_tile, grid_cap): each owns its
    # own IPC inbox, whose layout depends on all three. Sorted so every rank
    # performs its handle exchanges in the same sequence -- the exchange is a
    # collective, so a differing order across ranks deadlocks.
    # ``None`` means "constructor default" and does not order against an int,
    # so sort on a total key rather than the tuple itself.
    wanted_cfgs = sorted(
        {c.fly_cfg for c in CANDIDATES if c.family == "fly" and c.key in keys},
        key=lambda c: (
            c[0],
            -1 if c[1] is None else c[1],
            -1 if c[2] is None else c[2],
        ),
    )
    if (
        wanted_cfgs
        and HAS_FLY_INT4
        and get_gfx() in _FLY_ARCHS
        and tp_size in _FLY_WORLDS
        and dtype == dtypes.bf16
    ):
        for cfg in wanted_cfgs:
            algo, st, cap = cfg
            # QRInt4 exchanges IPC handles via broadcast_object_list, so it
            # needs the gloo (CPU) group -- it rejects an NCCL group outright.
            kw = {}
            if st is not None:
                kw["super_tile"] = st
            if cap is not None:
                kw["grid_cap"] = cap
            fly[cfg] = QRInt4(
                group=tp_group.cpu_group,
                device=device,
                rank=rank,
                world_size=tp_size,
                algorithm=algo,
                **kw,
            )
        # compile() is itself a collective launch on every super-tile engine and
        # builds all of them, so one call at any shape keeps the JIT out of
        # every timed region below.
        warm = torch.zeros((8, DSV4_HIDDEN), dtype=dtypes.bf16, device=device)
        for cfg in wanted_cfgs:
            dist.barrier(group=group)
            fly[cfg].compile(warm, torch.empty_like(warm))
        del warm

    try:
        rows = [
            _bench_shape(
                tp_size=tp_size,
                rank=rank,
                tokens=tokens,
                hidden=hidden,
                dtype=dtype,
                num_iters=num_iters,
                num_warmup=num_warmup,
                profile=profile,
                group=group,
                ca_comm=ca_comm,
                qr_comm=qr_comm,
                fly=fly,
                keys=keys,
                prod_regime=prod_regime,
            )
            for tokens, hidden in shapes
        ]
    finally:
        for eng in fly.values():
            eng.close()
        if dist.is_initialized():
            destroy_model_parallel()
            destroy_distributed_environment()
            torch.cuda.empty_cache()
    return rows


def dtype2str(dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _row(tp_size, tokens, hidden, dtype, rank_rets):
    """Collapse one shape's per-rank scalars into a table row."""
    nbytes = rank_rets[0]["nbytes"]
    row = {
        "gfx": get_gfx(),
        "dtype": dtype2str(dtype),
        "TP": tp_size,
        "M": tokens,
        "K": hidden,
        "payload size (KiB)": nbytes / 1024,
        # Carried for the roofline, which needs the exact byte count rather
        # than the rounded KiB the tables print. Not in ID_COLUMNS, so it never
        # reaches a printed table.
        "_nbytes": nbytes,
        "kernel": predicted_kernel(tp_size, nbytes),
        "naive": predicted_kernel(tp_size, nbytes, use_new=False),
        "prod path": rank_rets[0]["prod"],
    }
    for cand in CANDIDATES:
        key = f"{cand.key}_us"
        if key not in rank_rets[0]:
            continue  # candidate not applicable to this config
        per_rank = [r[key] for r in rank_rets]
        # Slowest rank is what the model waits on.
        us = max(per_rank)
        _, busbw, _ = collective_bw(nbytes, us, tp_size, row["kernel"])
        row[f"{cand.key} us"] = us
        row[f"{cand.key} busbw GB/s"] = busbw
        row[f"{cand.key} SQNR dB"] = min(r[f"{cand.key}_sqnr"] for r in rank_rets)
        if cand.key == PRIMARY:
            # Rank spread is the skew indicator, and skew is a property of the
            # barrier rather than of any one candidate -- reporting it once for
            # the kernel under study keeps the table readable.
            row[f"{PRIMARY} spread us"] = us - min(per_rank)
    return row


def run_sweep(tp_size, shapes, dtype, args, keys, prod_regime):
    """Spawn one process per rank and sweep every shape inside them."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    logger.info(
        "TP%d %s: %d shape(s), %d iters",
        tp_size,
        dtype2str(dtype),
        len(shapes),
        args.iters,
    )
    with Pool(processes=tp_size) as pool:
        rets = [
            pool.apply_async(
                _worker,
                args=(
                    tp_size,
                    r,
                    shapes,
                    dtype,
                    args.iters,
                    args.warmup,
                    init_method,
                    args.profile,
                    keys,
                    prod_regime,
                ),
            )
            for r in range(tp_size)
        ]
        pool.close()
        pool.join()
    per_rank = [r.get() for r in rets]
    return [
        _row(tp_size, tokens, hidden, dtype, [pr[i] for pr in per_rank])
        for i, (tokens, hidden) in enumerate(shapes)
    ]


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


ID_COLUMNS = [
    "gfx",
    "dtype",
    "TP",
    "M",
    "payload size (KiB)",
    "kernel",
    "naive",
    "prod path",
]

# The summary table answers "what should I use", not "how was this row
# dispatched" -- gfx/dtype are constant for a whole report section and
# kernel/naive are dispatch predictions that `prod path` already summarizes,
# so they are dropped rather than repeated on every row.
SUMMARY_ID_COLUMNS = [
    c for c in ID_COLUMNS if c not in ("gfx", "dtype", "kernel", "naive")
]


def _mark_na(out, cols):
    """Turn NaN into ``None`` in *cols*, so ``to_markdown(missingval="n/a")``
    renders it as ``n/a`` instead of ``nan``.

    Only for columns whose NaN can *only* mean "candidate not applicable to
    this row" (see ``applicable()``) -- a plain float NaN elsewhere (a failed
    roofline measurement, a floor-excluded winner) is left alone, since that is
    a different thing than "this kernel does not run here" and should keep
    reading ``nan``. Needs an explicit ``None`` rather than a float NaN because
    tabulate's ``missingval`` only fires on the former; converting the column
    to ``object`` first is what makes that stick without disturbing the
    ``floatfmt`` of the real numbers still in it.
    """
    for col in cols:
        if col in out.columns:
            out[col] = out[col].astype(object).where(out[col].notna(), None)


def latency_table(df, baseline: str, keys):
    """``us`` per candidate plus a speedup column against *baseline*.

    Ratio is ``baseline_us / candidate_us``, so **> 1.0 means the candidate is
    faster than the baseline**. The baseline gets no column of its own (it would
    be 1.0 everywhere). The baseline name is in the column header so a saved
    report stays self-describing when the flag changes.

    Speed is only half the comparison -- the quantizing candidates buy their
    ratio with tens of dB of SQNR. Read this table next to the accuracy one.
    """
    cols = [c for c in ID_COLUMNS if c in df] + [
        f"{k} us" for k in keys if f"{k} us" in df
    ]
    if f"{PRIMARY} spread us" in df:
        cols.append(f"{PRIMARY} spread us")
    out = df[cols].copy()
    base_col = f"{baseline} us"
    if base_col not in df.columns:
        logger.warning(
            "baseline %r produced no results (not applicable to any row in "
            "this sweep); skipping speedup columns",
            baseline,
        )
        _mark_na(out, [f"{k} us" for k in keys])
        return out
    for k in keys:
        col = f"{k} us"
        if k == baseline or col not in df.columns:
            continue
        out[f"{k} vs {baseline}"] = df[base_col] / df[col]
    _mark_na(out, [f"{k} us" for k in keys] + [f"{k} vs {baseline}" for k in keys])
    return out


def metric_table(df, suffix: str, keys):
    cols = [c for c in ID_COLUMNS if c in df] + [
        f"{k} {suffix}" for k in keys if f"{k} {suffix}" in df
    ]
    out = df[cols].copy()
    _mark_na(out, [f"{k} {suffix}" for k in keys])
    return out


def _roof(row, key, measured):
    """TransferBench ceiling for *key*'s wire bytes in this row, or None.

    Shared by ``roofline_table`` and ``summary_table`` so both grade a
    candidate against the bytes it actually sends rather than the payload it
    was handed. *key* of ``None`` means the payload itself.

    Keyed on ``(TP, bytes)`` only. It used to also key on the candidate's own
    algorithm, which meant a candidate was graded against a ceiling built from
    the same algorithm it had chosen -- so picking a better one than the model
    put ``eff`` above 1.0. The roof is now the best algorithm for that many
    bytes, which is a bound a candidate cannot legitimately beat.
    """
    nbytes = int(row["_nbytes"])
    return measured.get(
        (int(row["TP"]), tbr.wire_bytes(nbytes, key) if key else nbytes)
    )


def _roof_us(row, key, measured):
    """``_roof`` reduced to microseconds, or None when it was not measured."""
    roof = _roof(row, key, measured)
    return roof.us if roof is not None else None


def _eff(row, key, us, measured):
    """``roof us / us`` for *key*, or nan if the candidate or roofline is missing."""
    if key is None or not pd.notna(us):
        return float("nan")
    roof = _roof_us(row, key, measured)
    return roof / us if roof is not None else float("nan")


def _prod_candidate_key(prod_path: str) -> str | None:
    """Map a ``prod path`` string to the ``CANDIDATES`` key with its timing.

    ``production_path()`` emits exactly three shapes: ``"rccl"``,
    ``"cdr:<kernel>"`` (e.g. ``"cdr:2stage"``), and ``"qr:<regime>"`` (e.g.
    ``"qr:int4"``). The ``cdr:`` prefix always means the shipped kernel --
    production has no lever to select ``cdr_naive``. Returns ``None`` for a
    format this does not recognize, which the caller treats as "no production
    reference for this row" rather than guessing.
    """
    if prod_path == "rccl":
        return "rccl"
    if prod_path.startswith("cdr:"):
        return "cdr"
    if prod_path.startswith("qr:"):
        return f"qr_{prod_path.split(':', 1)[1]}"
    return None


def summary_table(df, keys, min_sqnr: float = DEFAULT_MIN_SQNR, roofline=None):
    """One winner per shape, and what it buys over the row's production path.

    The headline answer: for this shape, what is the fastest thing aiter can do
    to an all-reduce, and how much faster is it than what production actually
    dispatches here?

    **"Production" is derived per row from ``prod path``, not from a fixed
    candidate.** An earlier version used *every* row's ``prod time (us)`` from
    a single CLI-selected baseline (``cdr`` by default) while labelling the
    column with the per-row ``prod path`` string -- so a row where production
    falls back to RCCL (large messages past the custom-AR cutoff, or QR
    disabled) printed ``prod collective = rccl`` next to ``cdr``'s time and
    efficiency under that name. The two must always describe the same
    collective: ``_prod_candidate_key`` parses ``prod path`` (``"rccl"``,
    ``"cdr:<kernel>"``, ``"qr:<regime>"``) back into the ``CANDIDATES`` key
    that was actually timed, and every ``prod *`` column below comes from that
    key's own row -- never from an unrelated fixed baseline.

    A row can still show ``prod collective`` with no timing: if ``prod path``
    names a candidate the sweep did not measure (``-c`` excluded it, or the
    env implied a candidate the sweep never enabled), ``prod time (us)`` is
    NaN and a rendered ``-``. That is reported once per candidate rather than
    silently substituting a different collective's number -- see the log line
    this emits.

    **Ranked on speed alone this table would be a trap**, which is why the
    winner is accuracy-gated and why there are two of them:

    * ``fastest collective`` is the fastest candidate clearing *min_sqnr*
      (default ``DEFAULT_MIN_SQNR``), with ``fastest collective SQNR dB``
      printed beside it so the cost of the choice is never off-screen. Without
      a floor the winner would be the widest-error codec in the sweep at
      nearly every shape -- ``qr_int3`` at ~12 dB is ~25% relative error and
      beats everything on speed.
    * ``fastest exact collective`` is the fastest of the bit-accurate
      candidates (``Candidate.exact``), i.e. the fastest option that does not
      change the model's numerics at all. Omitted when every candidate in the
      sweep is already exact, since it would just repeat ``fastest
      collective``.

    A candidate excluded by the floor is not hidden: it keeps its column in the
    latency table, and the count of rows where the floor changed the winner is
    logged, so the default can never silently bury a result.

    Both ratios are ``prod time (us) / fastest time (us)``, so **> 1.0 means
    faster than production**, matching ``latency_table``. A row whose winner
    *is* the production path reads 1.0, which is the useful answer that
    nothing beat it.

    *roofline*, when given the ``measured`` lookup from ``measure_roofline``,
    adds a ``prod eff`` / ``fastest eff`` / ``fastest exact eff`` column next
    to each winner -- the same ``roof us / cand us`` ratio as
    ``roofline_table``, graded on that candidate's own wire bytes, so a
    quantizing winner is not held to the exact candidates' ceiling. This is
    also why a faster winner can show a *lower* eff than a slower one: e.g.
    ``fly_int4`` sends 1/4 the bytes of ``rccl``, so its roof is a quarter the
    size, and the same fixed quantize/dequantize and launch overhead is a
    larger fraction of a smaller roof. Faster-but-less-efficient is the
    signature of a candidate that is winning on payload reduction rather than
    on using the fabric well.
    """
    exact_keys = {c.key for c in CANDIDATES if c.exact}
    live = [k for k in keys if f"{k} us" in df.columns]
    # Only worth a separate column when the sweep actually mixes accuracy
    # classes; with -c cdr rccl every candidate is exact and it would duplicate.
    want_exact = any(k not in exact_keys for k in live)

    def _pick(row, pool, floor):
        """Fastest candidate in *pool* that ran here and clears *floor*.

        A candidate with no SQNR recorded is excluded rather than admitted:
        an ungraded result cannot be shown to clear the floor, and defaulting
        it in would be the one way this table could recommend something whose
        accuracy nobody checked.
        """
        best_k, best_us = None, float("inf")
        for k in pool:
            us = row.get(f"{k} us")
            if us is None or not pd.notna(us) or us >= best_us:
                continue
            if floor is not None:
                sqnr = row.get(f"{k} SQNR dB")
                if sqnr is None or not pd.notna(sqnr) or sqnr < floor:
                    continue
            best_k, best_us = k, us
        return best_k, (best_us if best_k else float("nan"))

    rows = []
    gated = {}  # candidate -> how many rows it would have won but for the floor
    unresolved_prod = {}  # candidate -> rows where prod path named it unmeasured
    id_rename = {"prod path": "prod collective"}
    for _, r in df.iterrows():
        out = {id_rename.get(c, c): r[c] for c in SUMMARY_ID_COLUMNS if c in df}

        prod_key = _prod_candidate_key(r["prod path"])
        prod_us = r.get(f"{prod_key} us") if prod_key else None
        if prod_key is not None and (prod_us is None or not pd.notna(prod_us)):
            unresolved_prod[prod_key] = unresolved_prod.get(prod_key, 0) + 1
            prod_us = None
        out["prod time (us)"] = prod_us if prod_us is not None else float("nan")
        if roofline is not None:
            out["prod eff"] = (
                _eff(r, prod_key, prod_us, roofline)
                if prod_key is not None and prod_us is not None
                else float("nan")
            )

        k, us = _pick(r, live, min_sqnr)
        out["fastest collective"] = k or "-"
        out["fastest time (us)"] = us
        out["fastest vs prod"] = (
            prod_us / us if prod_us is not None and pd.notna(us) else float("nan")
        )
        if roofline is not None:
            out["fastest eff"] = _eff(r, k, us, roofline)
        out["fastest collective SQNR dB"] = (
            r.get(f"{k} SQNR dB", float("nan")) if k else float("nan")
        )

        # What the floor cost, so it is visible rather than merely applied.
        ungated, _ = _pick(r, live, None)
        if ungated is not None and ungated != k:
            gated[ungated] = gated.get(ungated, 0) + 1

        if want_exact:
            k, us = _pick(r, [x for x in live if x in exact_keys], min_sqnr)
            out["fastest exact collective"] = k or "-"
            out["fastest exact time (us)"] = us
            out["fastest exact vs prod"] = (
                prod_us / us if prod_us is not None and pd.notna(us) else float("nan")
            )
            if roofline is not None:
                out["fastest exact eff"] = _eff(r, k, us, roofline)
        rows.append(out)

    for k, n in sorted(gated.items(), key=lambda kv: -kv[1]):
        logger.info(
            "summary: %s was fastest on %d/%d row(s) but is below the "
            "%g dB floor; see the latency table for its timings",
            k,
            n,
            len(df),
            min_sqnr,
        )

    for k, n in sorted(unresolved_prod.items(), key=lambda kv: -kv[1]):
        logger.warning(
            "summary: prod path named %s on %d/%d row(s), but it was not "
            "measured in this sweep (excluded by -c, or implied by an env "
            "var this sweep did not enable); prod time/eff are blank there",
            k,
            n,
            len(df),
        )
    return pd.DataFrame(rows)


def measure_roofline(df, keys, *, binary, cus, iters, warmup):
    """Run TransferBench once per TP and return the ``(tp, wire_bytes,
    pattern) -> us`` lookup that ``roofline_table`` and ``summary_table`` both
    grade candidates against.

    One TransferBench process per TP; the ranks have already been joined by
    then, so the GPUs are free. A TP whose measurement fails is warned about
    and left blank rather than taking the whole run down. Returns ``None`` if
    every TP failed, so callers can skip the roofline entirely.
    """
    live = [k for k in keys if f"{k} us" in df.columns]

    # (tp, wire_bytes, pattern) -> us. Collect every distinct request first so
    # each TP costs exactly one process launch no matter how many shapes and
    # candidates map onto the same measurement.
    measured = {}
    for tp_size, sub in df.groupby("TP"):
        requests = set()
        for _, r in sub.iterrows():
            nbytes = int(r["_nbytes"])
            requests.add(nbytes)
            for k in live:
                if pd.notna(r[f"{k} us"]):
                    requests.add(tbr.wire_bytes(nbytes, k))
        logger.info(
            "TransferBench: TP%d, %d distinct byte count(s) x %d algorithm(s)",
            tp_size,
            len(requests),
            len(tbr._ALGOS),
        )
        try:
            got = tbr.measure(
                int(tp_size),
                requests,
                binary=binary,
                cus=cus,
                iters=iters,
                warmup=warmup,
            )
        except (RuntimeError, OSError, subprocess.SubprocessError) as e:
            logger.warning("TransferBench: TP%d roofline unavailable: %s", tp_size, e)
            continue
        for nbytes, roof in got.items():
            measured[(int(tp_size), nbytes)] = roof

    return measured or None


def roofline_table(df, keys, measured):
    """Fabric ceiling per row, and what fraction of it each candidate reached.

    ``roof us`` is the fastest way TransferBench could move this row's *payload*
    bytes, over one-shot, two-shot and ring; ``roof algo`` names the winner.
    Each ``<cand> eff`` uses the same best-over-algorithms roof but at that
    candidate's own wire size (``transferbench_roofline.wire_bytes``), so a
    quantizing candidate is graded on the bytes it really sends rather than the
    ones it was handed.

    ``eff`` is ``roof us / cand us``, so **1.0 means the candidate is at the
    ceiling and values above 1.0 should not occur**. Two ways to read it:

    * **Well below 1.0 at small sizes is expected, not a finding.** The
      roofline has no peer handshake, and the 1-stage kernel is dominated by
      the ``start_sync`` spin there. The gap is the sync cost, not waste.
    * **Above 1.0 is a bug in the roofline**, not a fast kernel. It means some
      algorithm the candidate can reach is not in ``_ALGOS``, so the "ceiling"
      is really the cost of an algorithm the candidate beat. This is exactly
      what a two-shot-only model did to ``rccl`` on a NUMA-split PCIe host
      (``eff`` 1.18, because RCCL rings and the model did not). Add the missing
      pattern rather than explaining the number away.

    ``roof algo`` is worth reading next to the ``kernel`` column: where they
    disagree, the dispatch picked an algorithm this fabric does not favour.

    *measured* is the lookup from ``measure_roofline``.
    """
    out = df[[c for c in ID_COLUMNS if c in df]].copy()
    live = [k for k in keys if f"{k} us" in df.columns]

    roofs = [_roof(r, None, measured) for _, r in df.iterrows()]
    out["roof us"] = [x.us if x is not None else None for x in roofs]
    out["roof GB/s"] = [
        float("nan") if u is None or not u else n / u / 1e3
        for u, n in zip(out["roof us"], df["_nbytes"])
    ]
    out["roof algo"] = [x.algo if x is not None else "-" for x in roofs]
    for k in live:
        # None (-> "n/a") when the candidate does not apply to this row;
        # float nan (-> "nan") when it does but the roofline point for it
        # could not be measured. _eff() collapses that distinction, so it is
        # remade here rather than by blanket-marking the column afterwards.
        effs = [
            _eff(r, k, r[f"{k} us"], measured) if pd.notna(r[f"{k} us"]) else None
            for _, r in df.iterrows()
        ]
        out[f"{k} eff"] = pd.Series(effs, dtype=object, index=out.index)
    return out


def _write_report(
    path, sections, args, visible: int, prod_regime, roofline_cus=None
) -> None:
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
        # Arch does not identify the fabric -- MI350X (xGMI) and MI350P
        # (PCIe-only) both report gfx950, and peer bandwidth between them
        # differs by an order of magnitude. Without this line two reports from
        # the two machines are indistinguishable on the axis that explains
        # most of the gap between them.
        f"- peer links: {_peer_link_type()}",
        # Which devices, not just how many: on a PCIe host the NUMA split of
        # the chosen subset is worth ~21% on the roofline (see _gpu_numa_map).
        f"- HIP_VISIBLE_DEVICES: {os.environ.get('HIP_VISIBLE_DEVICES', '(unset)')}",
        f"- GPU NUMA node: {_gpu_numa_map()}",
        f"- iters: {args.iters} (warmup {args.warmup})",
        f"- baseline: {args.baseline}",
        f"- fly_int4 available: {HAS_FLY_INT4}",
        (
            "- bf16 cast to fp16 on the QR wire: "
            f"{os.environ.get('AITER_QUICK_REDUCE_CAST_BF16_TO_FP16', '1')}"
        ),
        f"- `prod path` evaluated with {_QR_ENV}={prod_regime!r}",
        f"- summary `fastest collective` accuracy floor: {args.min_sqnr} dB",
    ]
    # Only describe the roofline when one was actually measured: --roofline
    # degrades to a warning when the binary is missing, and a header promising
    # a table the report does not contain is worse than no header line.
    if roofline_cus is not None:
        lines.append(f"- roofline CU sweep: {roofline_cus}")
    lines += [
        f"- command: {' '.join(sys.argv)}",
        "",
        "Candidates are not accuracy-equivalent --",
        "read the latency table alongside the SQNR one, never alone.",
        "The summary table's `fastest collective` column is the fastest candidate",
        "clearing the accuracy floor above, with `fastest collective SQNR dB`",
        "beside it showing what that choice costs; `fastest exact collective` is",
        "the fastest option that leaves the model's numerics untouched.",
        "Candidates below the floor are excluded from `fastest collective` only --",
        "their timings are still in the latency table.",
        "",
    ]
    if roofline_cus is not None:
        lines += [
            "`eff` in the roofline table is `roof us / cand us`, where the roof is",
            "the fastest of one-shot / two-shot / ring moving that candidate's wire",
            "bytes **with no peer handshake**; `roof algo` names the winner. Below",
            "1.0 at small sizes is the sync cost, not waste. Above 1.0 should not",
            "happen and means the roof is missing an algorithm the candidate used,",
            "not that the kernel was fast.",
            "",
        ]
    for title, table in sections:
        lines += [f"## {title}", "", table, ""]

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
        "-tp",
        "--tp",
        type=int,
        nargs="*",
        choices=[2, 4, 6, 8],
        default=None,
        help="tensor-parallel world size(s). Default: TP4 only. Pass -tp 2 to\n"
        "also cover the 1stage-only TP2 case, or -tp 2 4 8 for the full sweep\n"
        "(each still capped to the visible GPUs). TP6 is custom-AR only\n"
        "(quick reduce rejects it).",
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
        choices=["fp16", "bf16", "fp32"],
        default=["bf16"],
        help="data type(s). fp16 is the only one where cdr_fp8 applies.\n"
        "fp32 reaches only cdr/cdr_naive/rccl (every codec is fp16/bf16-only),\n"
        "and is the apples-to-apples case for --roofline: TransferBench is\n"
        "itself fp32, so the roof then matches on element type as well as on\n"
        "byte count. Note a token is twice the bytes, so the 1stage/2stage\n"
        "dispatch boundaries land at half the M.",
    )
    parser.add_argument(
        "-c",
        "--candidates",
        nargs="*",
        choices=CANDIDATE_KEYS,
        default=None,
        help="restrict the candidate set (default: everything applicable)",
    )
    parser.add_argument("--iters", type=int, default=101, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    parser.add_argument(
        "--busbw",
        action="store_true",
        help="also print the busbw table (derived from the us table)",
    )
    parser.add_argument(
        "--min-sqnr",
        type=float,
        default=DEFAULT_MIN_SQNR,
        help="accuracy floor in dB for the summary table's 'best' column, so a\n"
        f"fast but very inaccurate codec cannot win it. Default: "
        f"{DEFAULT_MIN_SQNR} dB,\n"
        "which admits int4 (~18 dB) and excludes int3 (~12 dB, ~25%% relative\n"
        "error). Raise it to ask a deployment question ('fastest thing above\n"
        "25 dB'); pass 0 to rank on speed alone. Excluded candidates keep\n"
        "their columns in the latency and accuracy tables either way.",
    )
    parser.add_argument(
        "--roofline",
        action="store_true",
        help="also print a roofline table: TransferBench moving the same bytes\n"
        "in the same pattern with no peer handshake, and each candidate's\n"
        "efficiency against it. Needs the TransferBench binary (not in a\n"
        "default ROCm install) -- see transferbench_roofline.py. Adds one\n"
        "TransferBench process per TP size, after that TP's ranks have exited.",
    )
    parser.add_argument(
        "--roofline-bin",
        metavar="PATH",
        default=None,
        help="path to the TransferBench binary. Default: $TRANSFERBENCH, then\n"
        "PATH, then the usual ROCmValidationSuite locations.",
    )
    parser.add_argument(
        "--roofline-cus",
        type=int,
        nargs="*",
        default=list(tbr.DEFAULT_CUS),
        help="CU counts to try per roofline point; the best one wins, across\n"
        "every modelled algorithm. Measured effect is small (<1%% between 8-32\n"
        f"and 4-128 on gfx950). Default: {list(tbr.DEFAULT_CUS)}",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="also emit a per-rank chrome trace of every candidate's loop.\n"
        "On ROCm builds where a spawned rank's profiler sees no GPU activity\n"
        "(the reason the table itself is timed with hipEvents -- see\n"
        "_bench_shape) the trace carries CPU rows only.",
    )
    parser.add_argument(
        "-b",
        "--baseline",
        choices=CANDIDATE_KEYS,
        default=PRIMARY,
        help="candidate to measure the others against. Adds a\n"
        "'<cand> vs <baseline>' column per candidate, where > 1.0 means the\n"
        f"candidate is faster than the baseline. Default: {PRIMARY}, i.e.\n"
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

    tps = args.tp if args.tp else [4]
    tps = [t for t in tps if t <= visible]
    if not tps:
        logger.warning("no requested TP size fits %d visible GPUs; skipping", visible)
        return
    if max(tps) < 4:
        logger.warning(
            "only %d GPUs visible: TP2 reaches cross_device_reduce_1stage only. "
            "Use 4+ GPUs to also measure 2stage.",
            visible,
        )

    keys = args.candidates or CANDIDATE_KEYS

    # Resolve the binary before spawning anything: a missing TransferBench
    # should cost a warning at startup, not a full sweep followed by one.
    roofline_bin = None
    if args.roofline:
        roofline_bin = tbr.find_binary(args.roofline_bin)
        if roofline_bin is None:
            logger.warning(
                "--roofline requested but the TransferBench binary was not found; "
                "build it from https://github.com/ROCm/TransferBench and set "
                "$TRANSFERBENCH or pass --roofline-bin. Continuing without it."
            )
        else:
            logger.info("TransferBench: using %s", roofline_bin)
    # Remember what the deployment would do before overriding the environment
    # for our own QR candidates; `prod path` is reported against this value.
    prod_regime = os.environ.get(_QR_ENV)
    if any(c.family == "qr" for c in CANDIDATES if c.key in keys):
        os.environ[_QR_ENV] = _QR_ENABLING_REGIME
        # Build the quick-reduce JIT module here rather than letting every
        # spawned rank race for the same first build.
        import aiter as ops

        ops.qr_max_size()

    sections = []
    roofline_cus = None  # set once a roofline table actually lands in a section
    for dtype_name in args.dtype:
        dtype = dtypes.d_dtypes[dtype_name]
        rows = []
        for tp_size in tps:
            rows += run_sweep(tp_size, args.shape, dtype, args, keys, prod_regime)
        df = pd.DataFrame(rows)

        measured = None
        if roofline_bin is not None:
            measured = measure_roofline(
                df,
                keys,
                binary=roofline_bin,
                cus=args.roofline_cus,
                iters=args.iters,
                warmup=args.warmup,
            )

        tables = [
            (
                f"{dtype_name} summary",
                summary_table(df, keys, args.min_sqnr, measured),
            ),
            (f"{dtype_name} latency", latency_table(df, args.baseline, keys)),
            (f"{dtype_name} accuracy", metric_table(df, "SQNR dB", keys)),
        ]
        if args.busbw:
            tables.append((f"{dtype_name} busbw", metric_table(df, "busbw GB/s", keys)))
        if measured is not None:
            tables.append(
                (f"{dtype_name} roofline", roofline_table(df, keys, measured))
            )
            roofline_cus = args.roofline_cus
        for title, table in tables:
            md = table.to_markdown(index=False, floatfmt=".4g", missingval="n/a")
            logger.info("all-reduce %s (markdown):\n%s", title, md)
            sections.append((title, md))

    if args.output:
        _write_report(args.output, sections, args, visible, prod_regime, roofline_cus)


if __name__ == "__main__":
    freeze_support()
    main()
