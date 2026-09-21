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
| ``fly_int4``  | FlyDSL mesh INT4 (ROCm/aiter#4970)      | int4     | no  |
| ``fly_int4_ring`` | FlyDSL ring, same two-shot volume   | int4/int6| no  |
| ``fly_1stage``| FlyDSL exact one-shot                   | bf16     | yes |
| ``rccl``      | ``dist.all_reduce``                     | bf16/fp16| yes |

The three ``fly_*`` families are the ones with a dispatch question open: which
of them wins is a function of payload size, and so is which variant wins inside
each. Every ``fly_*`` key above is joined by pinned tuning rows
(``fly_int4_ring_st16``, ``fly_1stage_b256_a4_g64``, ...) whose only purpose is to be
swept against the auto rows. A key names a *policy*, not a binary -- the auto
rows walk a size ladder -- so the ``variant`` column and the ``kernel variants``
table report the JIT symbol that actually ran at each shape, super-tile and
block count included.

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

``--fusion ar_rmsnorm`` swaps the whole candidate set for the **fused epilogue**
question instead: all-reduce + residual add + RMSNorm.

| column          | what runs                                   | fuses |
|-----------------|---------------------------------------------|-------|
| ``fused_fly_1stage``  | FlyDSL one-shot with the epilogue fused | yes |
| ``fused_fly_1stage_b<block>_g<cap>[_ss]`` | same, pinned block x grid cap x self-skip | yes |
| ``fused_cdr_1stage``  | ``allreduce_fusion_kernel_1stage`` -- the traced kernel | yes |
| ``fused_cdr_2stage``  | ``allreduce_fusion_kernel_2stage``      | yes |
| ``fused_qr_fp8``/``_int4`` | ``qr_all_reduce_rmsnorm`` per codec | yes |
| ``fused_fly_ring``    | FlyDSL quantized **ring** with the epilogue fused | yes |
| ``fused_fly_mesh``    | same, mesh schedule                     | yes |
| ``separate_cdr``      | ``cross_device_reduce`` + ``rmsnorm2d_fwd_with_add`` | no |
| ``separate_rccl``     | ``dist.all_reduce`` + ditto             | no |
| ``separate_fly1s``    | plain FlyDSL one-shot + ditto           | no |
| ``separate_flyring``  | plain FlyDSL ring + ditto               | no |
| ``separate_flymesh``  | plain FlyDSL mesh + ditto               | no |
| ``separate_qr_int4``  | ``quick_all_reduce`` + ditto            | no |

``fused_fly_ring`` against ``separate_flyring`` is the pair this fusion exists
for: the same schedule, one launch and ``4S`` of HBM traffic against two and
``6S``. It is a prefill row -- the quantized schedules are gated below by
accuracy, and the ring is where the two-shot family wins at size. Its block is
sized to the token row (``hidden/8`` threads), so it runs at widths that are a
multiple of 1024 up to 8192 and reports ``n/a`` elsewhere, where
``fused_qr_*`` needs the row to tile a 32 KiB tile evenly instead.

The ``fused_fly_1stage_b*`` rows pin the **block**, which in a fused build is
not the free knob it is in the plain one. The row has to fit one workgroup, so
``block * atoms * 8 == hidden``: pinning the block pins the atoms, and which
blocks exist at all is a function of the width. hidden=7168 admits only 896 and
448; hidden=8192 admits 1024, 512 and 256. A row whose block this width cannot
produce is **skipped**, not failed -- expect a different subset of the ``b*``
rows to report at each shape, which is why they are generated from
``one_shot_allreduce.fused_block_options`` rather than listed by hand.

The prefix describes the **kernel**, not the mode: ``fused_`` only where the
kernel genuinely fuses, ``separate_`` for the two-launch baselines. Both run
under ``--fusion``, graded against the same reference.

A ``separate_*`` row is two launches against a ``fused_*`` row's one. Under the
default graph timing both are measured with the launch path removed, so the
column compares kernels rather than host paths -- which is what makes the
fused-vs-separate pair meaningful. Fusing does also remove a real launch, but
this harness cannot size that: see the ``--timing`` note above.

The first table printed is the ``summary``: per shape, the fastest candidate
clearing an accuracy floor (``fastest collective``), the fastest bit-accurate
one (``fastest exact collective``), and each one's ratio against the
production path (``prod collective``, ``prod time (us)``). The floor defaults
to ``DEFAULT_MIN_SQNR`` and exists so a codec that is fast only because it
barely transmits anything cannot win the column; ``--min-sqnr`` moves it, and
``fastest collective SQNR dB`` prints what the winning choice actually cost.
The ``latency & accuracy by case`` tables below it are the full picture the
summary collapses: one small table per shape, one row per candidate, latency
(``us``, speedup vs the baseline) and accuracy (``SQNR dB``) side by side --
a candidate below the floor still appears there.

Because the candidates are **not accuracy-equivalent**, every one is graded on
SQNR against a common fp32 reference and asserted against its own floor in
``CANDIDATES``, so a real regression fails regardless of which accuracy class
the candidate is in. The exact kernels land at the bf16 rounding floor (~55 dB;
RCCL a little lower since it reduces in bf16 rather than accumulating in fp32),
the quantized ones at their codec's floor. Speed alone is a misleading read for
everything in the "exact = no" rows above -- each case's table prints ``us``
next to ``SQNR dB`` for exactly that reason.

``busbw`` (``--busbw``) is always computed on the payload dtype, including for
the quantizing candidates whose wire format is several times smaller: the
question the table answers is "how fast does my (M, hidden) all-reduce finish",
not "how efficiently is the wire used".

**``us`` is HIP-graph replay time by default** (``--timing graph``): capture the
collective, replay it, divide. That is the metric a captured deployment sees --
decode is captured -- and it is the only fair kernel-to-kernel comparison here.

``--timing eager`` switches to hipEvent wall time around the Python call. Read
that number knowing what it contains: ``run_perftest`` brackets a loop of
back-to-back calls, so once host cost per call exceeds device time the GPU
starves and the measurement *is* the host cost -- at TP2/M=1 a 5.7 us kernel
reads as ~20 us. That cost also differs per candidate family (a ``separate_*``
row makes two Python op calls, ``cdr``/``qr`` go through pybind, the FlyDSL
rows through ``_run_compiled``, ``rccl`` through an aten op plus a ``copy_``),
so eager partly ranks candidates by how much Python sits in their bench thunk
-- a property of this harness, not of the kernel. Every boundary in
``allreduce_policy`` is a crossover *between* families, so that bias lands
straight on the shipped thresholds. Either way the peer-wait that dominates the
1-stage kernel is included, and the torch profiler is not usable here -- see
the note in ``_bench_shape``.

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

    # the fused epilogue at the exact shapes the Qwen3-235B decode trace runs
    HIP_VISIBLE_DEVICES=0,1 python3 op_tests/multigpu_tests/bench_comm_allreduce.py \
        -tp 2 --fusion ar_rmsnorm -s 1,4096 32,4096

    # just the pair that answers "was fusing worth it": same all-reduce, same
    # arch, differing only in whether the norm is fused
    python3 op_tests/multigpu_tests/bench_comm_allreduce.py -tp 2 \
        --fusion ar_rmsnorm -s 1,4096 \
        -c fused_fly_1stage separate_fly1s fused_cdr_1stage

    # add the fabric ceiling: how much of what TransferBench can move in the
    # same pattern is each candidate actually getting? Needs the TransferBench
    # binary -- see transferbench_roofline.py.
    python3 op_tests/multigpu_tests/bench_comm_allreduce.py --roofline

    # save a report to diff against after a kernel change
    python3 op_tests/multigpu_tests/bench_comm_allreduce.py -o /tmp/ar_before.md
    #   ... change the kernel, rebuild, then -o /tmp/ar_after.md and diff the two.

    # dispatch-threshold sweep: a byte ladder too long for a command line, and
    # a CSV of the raw numbers to fit against. The default shape list jumps
    # 168 KiB -> 1.75 MiB -> 14 MiB and both family crossovers hide in those
    # gaps -- pinned fly_1stage* rows survive across the whole ladder by
    # default (see _FLY1S_DEFAULT_CEILING); AITER_BENCH_FLY1S_MAX_KB would
    # only be needed to narrow the window instead. See
    # op_tests/multigpu_tests/shapes/README.md.
    python3 op_tests/multigpu_tests/bench_comm_allreduce.py -tp 4 \
        -c fly_int4 fly_int4_ring fly_1stage fly_1stage_b256_a4_g64 \
        --shape-csv op_tests/multigpu_tests/shapes/ar_sweep_a_small.csv \
        -o /tmp/ar_a_small_tp4.md --output-csv /tmp/ar_a_small_tp4.csv

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

    # hardware counters (PMC), which the recipe above cannot give you. Counter
    # collection has to be joined across several passes, and a multi-process
    # counter set has no stable cross-pass rank key -- start-order ranking would
    # pair one rank's counters with another's -- so the tooling refuses it. Run
    # one rank per process and profile only that one. The peers go first and
    # --repeat is the number of passes the profiler will make (4 on gfx950):
    INIT=tcp://127.0.0.1:29500
    ARGS="-c fly_int4 --shape 1024,7168 --warmup 20 --iters 50"
    for r in 1 2 3; do
        HIP_VISIBLE_DEVICES=0,1,2,3 python3 \
            op_tests/multigpu_tests/bench_comm_allreduce.py \
            $ARGS --rank $r --init-method $INIT --repeat 4 &
    done
    HIP_VISIBLE_DEVICES=0,1,2,3 rocprofv3 -i pmc_recipe.txt -f csv -d /tmp/arpmc \
        -o pass0_%pid% -- \
        python3 op_tests/multigpu_tests/bench_comm_allreduce.py \
            $ARGS --rank 0 --init-method $INIT
    # Keep the JIT cache on (do not set FLYDSL_RUNTIME_ENABLE_CACHE=0): every
    # pass must launch the identical kernel set or the join has nothing to
    # match on. Warm up properly too -- a cold first rendezvous spins for
    # milliseconds waiting on a peer that is still building, and that one
    # outlier dominates any counter divided by GRBM_GUI_ACTIVE.
"""

import argparse
import logging
import math
import os
import re
import subprocess
import sys
import time
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

# The two knobs that move production's fused-AR 1stage/2stage split
# (communicator_cuda.py:22-25). Read for the `prod path` column, never set by
# this bench -- the fused_cdr_* rows call the kernel directly and pick their own
# schedule, so they are measured at every size regardless of what production
# would choose.
_AR_1STAGE_ENV = "AITER_AR_1STAGE"
_AR_1STAGE_MAX_KB_ENV = "AITER_AR_1STAGE_MAX_KB"
_QR_ENABLING_REGIME = "FP"

# QuickAllReduce constraints, from aiter/dist/device_communicators/quick_all_reduce.py.
_QR_WORLDS = (2, 4, 8)
_QR_DTYPES = (dtypes.fp16, dtypes.bf16)

# QuickAllReduceInt4 (FlyDSL) constraints, from aiter/ops/flydsl/quick_allreduce_int4.py.
_FLY_ARCHS = ("gfx942", "gfx950")
_FLY_WORLDS = (2, 4, 8)

# runFp8QuantKernel is only reached for fp16 inputs of at least this many
# elements; below it custom_all_reduce.cu:90 falls through to the plain kernel.
_FP8_MIN_NUMEL = 128 * 2048

# The FlyDSL schedules ride on flydsl, which aiter treats as an optional
# dependency and version-checks at import (it is unavailable on archs outside
# flydsl's SMEM_CAPACITY_MAP). A failed import here is that gate -- there is no
# separate availability predicate to mirror.
try:
    from aiter.dist.device_communicators.flydsl_all_reduce import (
        FlyDSLAllReduce,
        FlyDSLAllReduceRMSNorm,
    )
    from aiter.ops.flydsl import QuickAllReduceInt4
    from aiter.ops.flydsl import allreduce_policy as policy
    from aiter.ops.flydsl.kernels.one_shot_allreduce import (
        fused_block_options as _fused_block_options,
    )
    from aiter.ops.flydsl.kernels.one_shot_allreduce import (
        fused_hidden_supported as _fused_hidden_supported,
    )
    from aiter.ops.flydsl.kernels.quick_allreduce_fusions import (
        quick_reduce_hidden_supported as _flyqr_hidden_supported,
    )
    from aiter.ops.flydsl.kernels.quick_allreduce_fusions import (
        quick_reduce_row_block_options as _flyqr_block_options,
    )
    from aiter.ops.flydsl.one_shot_allreduce import (
        OneShotAllReduce,
        OneShotAllReduceRMSNorm,
    )
    from aiter.ops.flydsl.quick_allreduce_int4 import (
        ALGORITHMS,
        MIN_PAYLOAD_BYTES,
        QuickAllReduceInt4RMSNorm,
        _resolve_codecs,
        has_xgmi_peer_links,
    )

    HAS_FLY_INT4 = True
except Exception:  # noqa: BLE001
    QuickAllReduceInt4 = None
    QuickAllReduceInt4RMSNorm = None
    OneShotAllReduce = None
    OneShotAllReduceRMSNorm = None
    _fused_hidden_supported = None
    _fused_block_options = None
    _flyqr_hidden_supported = None
    _flyqr_block_options = None
    MIN_PAYLOAD_BYTES = 0
    ALGORITHMS = {}
    _resolve_codecs = None
    has_xgmi_peer_links = None
    FlyDSLAllReduce = None
    FlyDSLAllReduceRMSNorm = None
    policy = None
    HAS_FLY_INT4 = False

# FlyDSLAllReduce is opt-in and self-disabling; the bench turns it on for its
# own `fly_auto` row the same way it forces a quick-reduce regime for the qr_*
# rows, and for the same reason -- measuring a path production can reach
# requires enabling it.
_FLY_ENV = "AITER_FLY_AR"

# AITER_FLY_AR_ACCURACY defaults to "exact", which never opens the mesh/ring
# window at all -- past oneshot_max_exact the policy has no window and
# should_fly_all_reduce simply declines the payload. `fly_auto`'s mode is a
# bench CLI flag (--fly-accuracy, default "fast") rather than whatever the
# launching shell happens to export, so a report's accuracy regime is always
# what its own command line says.
_FLY_ACCURACY_ENV = "AITER_FLY_AR_ACCURACY"
_FLY_ACCURACY_CHOICES = ("fast", "exact")
_FLY_ACCURACY_DEFAULT = "fast"

# How `us` is measured. Recorded in the report header and in every CSV row, so
# a fit can refuse to mix regimes -- the two are not comparable and the
# difference is largest exactly where the decode thresholds live.
_TIMING_CHOICES = ("graph", "eager")
_TIMING_DEFAULT = "graph"
_GRAPH_INNER_DEFAULT = 10


def _bench_fly_accuracy_mode() -> str:
    """The bench's own accuracy regime: ``--fly-accuracy``, as seen by a
    worker process via ``_FLY_ACCURACY_ENV``.

    Read by ``applicable()`` so pinned FlyDSL rows are gated the same way
    ``fly_auto`` is -- "exact" has no mesh/ring window in production, so a
    pinned mesh/ring row is not "relevant for the mode" there either.
    """
    return os.environ.get(_FLY_ACCURACY_ENV, _FLY_ACCURACY_DEFAULT)


def _aiter_origin() -> str:
    """Filesystem root of the ``aiter`` package this process actually imported."""
    try:
        import aiter as _a

        return str(Path(_a.__file__).resolve().parent)
    except Exception:  # noqa: BLE001
        return "unknown"


def _peer_link_type() -> str:
    """GPU-to-GPU link type for the provenance header.

    Shares QuickAllReduceInt4's KFD probe rather than reimplementing it, so the report can
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
    algorithm: str = "mesh"  # QuickAllReduceInt4 schedule, family == "fly"
    # QuickAllReduceInt4 tuning knobs, family == "fly". None means "leave the constructor
    # default alone"; a value makes this candidate a distinct engine with its
    # own IPC inbox, so two rows can differ only in tuning.
    super_tile: int | None = None
    grid_cap: int | None = None
    # Wire format per lap, family == "fly". None means QuickAllReduceInt4's own per-world
    # default, which is *not* constant across the sweep: the ring's
    # reduce-scatter lap widens to INT6 at TP8 (_RS_INT6_MIN_WORLD) because it
    # requantizes N-1 times. That is the right production default and the wrong
    # thing to leave floating in a crossover sweep -- a TP4-vs-TP8 comparison
    # would then differ in schedule *and* codec at once, and neither column
    # would say which moved the number. Pin these to hold the wire constant.
    rs_codec: str | None = None
    ag_codec: str | None = None
    # OneShotAllReduce tuning knobs, family == "fly1s". Same rule: a distinct
    # value means a distinct engine with its own inbox.
    atoms: int | None = None
    fanout: str | None = None
    # Drop this rank's own trip through its own inbox, family == "fly1s". None
    # leaves it to the rung; False and True both pin it, and pinning it True
    # specialises the binary per rank.
    skip_self: bool | None = None
    # Threads per block, family == "fly1s". Also the tile width, so it is the
    # knob that sets how many blocks a decode payload gets. None leaves it to
    # the rung.
    block: int | None = None
    # Rows that only exist under --fusion ar_rmsnorm. A fused candidate is never
    # run in plain mode and vice versa: the two modes compute different things
    # and are graded against different references, so mixing them in one table
    # would put incomparable numbers in one column.
    fusion: bool = False
    # For family == "separate": which all-reduce carries the two-launch
    # baseline. The norm is always aiter's rmsnorm2d_fwd_with_add.
    sep_ar: str | None = None
    # For family == "fused_cdr": 1stage vs 2stage. Distinct from `use_new`,
    # which selects the plain kernel family rather than the fused schedule.
    use_1stage: bool = True

    @property
    def fly_cfg(self) -> tuple:
        """Identity of the QuickAllReduceInt4 engine this candidate needs."""
        return (
            self.algorithm,
            self.super_tile,
            self.grid_cap,
            self.rs_codec,
            self.ag_codec,
        )

    @property
    def fly1s_cfg(self) -> tuple:
        """Identity of the OneShotAllReduce engine this candidate needs."""
        return (self.atoms, self.grid_cap, self.fanout, self.skip_self, self.block)

    def fly1s_rung(self, min_bytes: int) -> tuple:
        """This candidate as an ``ONESHOT_LADDER`` rung, for the fit's paste."""
        if self.family != "fly1s":
            raise ValueError(f"{self.key} is not a one-shot candidate")
        unpinned = [
            n
            for n in ("atoms", "grid_cap", "fanout", "block", "skip_self")
            if getattr(self, n) is None
        ]
        if unpinned:
            raise ValueError(
                f"{self.key} leaves {', '.join(unpinned)} to the OneShotAllReduce "
                "default, so it has no ONESHOT_LADDER rung. Pin every knob (see "
                "_FLY1S_GRID) or keep the row out of the ladder fit."
            )
        return (
            int(min_bytes),
            self.atoms,
            self.grid_cap,
            self.fanout,
            self.block,
            self.skip_self,
        )

    @property
    def flyqr_rms_cfg(self) -> tuple:
        """Identity of the QuickAllReduceInt4RMSNorm engine this candidate needs.

        Same shape as ``fly_cfg`` plus ``block``, and like ``fly1s_rms_cfg``
        deliberately not keyed on hidden: one object serves every width,
        building a per-hidden inbox on demand. The fused geometry *is* a
        function of hidden -- the block is sized to the row -- so keying here
        would build one engine per (config, shape) in a sweep and exhaust the
        IPC heap. ``block`` is safe to key on because it is a *policy* (pin this
        width, or take the widest), resolved per hidden inside the engine.
        """
        return (
            self.algorithm,
            self.super_tile,
            self.grid_cap,
            self.rs_codec,
            self.ag_codec,
            self.block,
        )

    @property
    def fly1s_rms_cfg(self) -> tuple:
        """Identity of the OneShotAllReduceRMSNorm engine this candidate needs.

        Deliberately the same shape as ``fly1s_cfg`` and *not* keyed on hidden:
        one engine object serves every width, building a per-hidden inbox on
        demand. Keying on hidden here would build one engine per (config, shape)
        in the sweep and exhaust the IPC heap.

        ``atoms`` and ``block`` are the same knob in a fused build -- the row
        has to fit one workgroup, so one picks the other -- and the fused rows
        pin ``block``, leaving ``atoms`` None for the engine to resolve per
        width. Both are in the key anyway: they are engine-identifying, and a
        row that pinned ``atoms`` instead must not collide with one of these.
        """
        return (self.atoms, self.grid_cap, self.fanout, self.skip_self, self.block)


_FLY1S_GRID = (
    # block, atoms, grid_cap, fanout   tile
    (64, 1, 64, "peer"),  # 1 KiB
    (64, 1, 256, "peer"),  # 1 KiB
    (128, 1, 128, "peer"),  # 2 KiB
    (256, 1, 64, "peer"),  # 4 KiB
    (256, 1, 128, "peer"),  # 4 KiB
    (128, 2, 64, "peer"),  # 4 KiB
    (64, 4, 64, "peer"),  # 4 KiB
    (256, 2, 64, "peer"),  # 8 KiB
    (256, 2, 64, "atom"),  # 8 KiB
    (256, 4, 64, "peer"),  # 16 KiB
    (256, 4, 64, "atom"),  # 16 KiB
    (256, 4, 128, "peer"),  # 16 KiB
)


def _fly1s_grid_rows():
    """``_FLY1S_GRID`` x self-skip as Candidates."""
    rows = []
    for skip_self in (False, True):
        for block, atoms, cap, fanout in _FLY1S_GRID:
            key = f"fly_1stage_b{block}_a{atoms}_g{cap}"
            if atoms > 1 and fanout == "atom":
                key += "_fa"
            if skip_self:
                key += "_ss"
            rows.append(
                Candidate(
                    key,
                    "fly1s",
                    40.0,  # min acceptable SQNR value
                    True,
                    atoms=atoms,
                    grid_cap=cap,
                    fanout=fanout,
                    block=block,
                    skip_self=skip_self,
                )
            )
    return tuple(rows)


# Widths the fused grid is generated for. Which blocks exist is a function of
# hidden -- the row has to fit one workgroup -- so there is no width-independent
# block list to sweep. Take the union over the widths this bench actually sees
# and let `_fused_hidden_ok` drop the rows that do not apply at the shape being
# run; a row keyed on a block that width cannot produce is skipped, not failed.
_FUSED_GRID_HIDDENS = (2048, 3072, 4096, 5120, 6144, 7168, 8192)
# Grid cap is the only knob that moves the *block count* in fused mode: the tile
# is one token row, so the tile size, the wire volume and the flag count are all
# fixed by the shape. Block moves the thread/work split within a row and the LDS
# partial count.
_FUSED_FLY1S_CAPS = (8, 32, 64, 128)


def _fused_fly1s_grid_rows():
    """Legal fused blocks x grid cap x self-skip as Candidates.

    The block list comes from the kernel module rather than from a literal here,
    because it *is* the kernel's constraint: BLOCK = hidden/(8*atoms) has to be a
    whole number of waves and no wider than 1024.
    """
    if _fused_block_options is None:
        return ()
    blocks = sorted(
        {b for h in _FUSED_GRID_HIDDENS for b, _ in _fused_block_options(h)},
        reverse=True,
    )
    rows = []
    for skip_self in (False, True):
        for block in blocks:
            for cap in _FUSED_FLY1S_CAPS:
                key = f"fused_fly_1stage_b{block}_g{cap}"
                if skip_self:
                    key += "_ss"
                rows.append(
                    Candidate(
                        key,
                        "fused_fly1s",
                        40.0,  # min acceptable SQNR value
                        True,
                        fusion=True,
                        grid_cap=cap,
                        block=block,
                        skip_self=skip_self,
                    )
                )
    return tuple(rows)


def _fused_flyqr_blocks():
    """Blocks worth pinning on the quantized fused rows.

    Stricter geometry than the one-shot's -- a rank-tile also has to land on the
    64 B sector grid, so the block is a multiple of 128 -- which leaves most
    widths with no choice: 7168 is always 896, at every world size. Only widths
    with two or more options contribute; the unpinned rows already cover the
    forced ones.
    """
    if _flyqr_block_options is None:
        return ()
    blocks = set()
    for h in _FUSED_GRID_HIDDENS:
        for ws in _FLY_WORLDS:
            opts = _flyqr_block_options(h, ws)
            if len(opts) > 1:
                blocks |= {b for b, _ in opts}
    return tuple(sorted(blocks, reverse=True))


def _fused_flyqr_grid_rows():
    """Pinned-block rows for the quantized fused schedules."""
    rows = []
    for algorithm, floor in (("ring", 10.0), ("mesh", 15.0)):
        for block in _fused_flyqr_blocks():
            rows.append(
                Candidate(
                    f"fused_fly_{algorithm}_b{block}",
                    "fused_flyqr",
                    floor,  # same floors as the unpinned rows of each schedule
                    False,
                    fusion=True,
                    algorithm=algorithm,
                    block=block,
                )
            )
    return tuple(rows)


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
    # Mesh tuning rows. The mesh is the one schedule with *no* size ladder --
    # `_Algorithm.st_ladder` is empty for it, so ST=8 (falling back to 1 when a
    # payload has fewer than 8 tiles) runs at every size, and the default grid
    # cap of 1216 is never revisited. These two rows are what decides whether
    # that is right or merely untested: `st1` pins the fallback at every size,
    # `g128` holds ST at the default and moves only the block ceiling, to the
    # same 128 the ring's rungs use.
    Candidate("fly_int4_st1", "fly", 15.0, False, super_tile=1),
    Candidate("fly_int4_g128", "fly", 15.0, False, grid_cap=128),
    # Exact FlyDSL one-shot: no codec, fp32 accumulate, one rounding, so it
    # lands at the same bf16 floor as cdr and shares its 40 dB gate and its
    # `exact=True` checkAllclose. Decode-only -- it pushes the whole payload to
    # every peer, so it is gated *above* by MAX_PAYLOAD_BYTES rather than below
    # like the quantized rows.
    # Auto: no pinned knobs, so OneShotAllReduce walks ONESHOT_LADDER and picks
    # by payload size at launch. This is what production gets; the pinned rows
    # below are what it is fitted against.
    Candidate("fly_1stage", "fly1s", 40.0, True),  # 55 / n/a
    # Pinned rows: the knob grid the ladder is fitted over. See _FLY1S_GRID.
    *_fly1s_grid_rows(),
    # Same kernel family, ring schedule. Its floor is lower than fly_int4's
    # because the ring's reduce-scatter lap requantizes N-1 times where the mesh
    # requantizes once; measured 18.7 dB at TP4 (against 19.2), and *better*
    # than the mesh at TP2 (22.2 dB) where the all-gather lap's verbatim
    # forwarding dominates. At TP8 INT4 would land ~15 dB, which is why the
    # ring defaults to an INT6 reduce-scatter lap there (~21 dB); see
    # QuickAllReduceInt4's rs_codec. 14 dB leaves the usual ~5 dB of headroom.
    # Auto: no pinned super_tile, so QuickAllReduceInt4 walks RING_ST_LADDER and picks by
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
    # The same two rungs with the reduce-scatter lap pinned to INT6. The rows
    # above leave `rs_codec=None`, i.e. QuickAllReduceInt4's per-world default, which is
    # INT4 below TP8 and INT6 at TP8 -- so the TP4 and TP8 reports are not
    # comparing the same wire, and the TP8 ring's 21.6 dB against TP4's 18.7 is
    # a codec difference reported as a schedule difference. These rows hold the
    # wire constant across world sizes; read them against the INT4 rows at the
    # same ST to price what the wider RS lap costs in latency. Same 14 dB floor
    # -- INT6 only ever lands above INT4, so it cannot be the row that trips.
    Candidate(
        "fly_int4_ring_st8_int6",
        "fly",
        14.0,
        False,
        algorithm="ring",
        super_tile=8,
        grid_cap=128,
        rs_codec="int6",
        ag_codec="int4",
    ),
    Candidate(
        "fly_int4_ring_st32_int6",
        "fly",
        14.0,
        False,
        algorithm="ring",
        super_tile=32,
        grid_cap=128,
        rs_codec="int6",
        ag_codec="int4",
    ),
    # Production dispatch: FlyDSLAllReduce picking a family per payload size.
    # Its accuracy mode is the bench's own --fly-accuracy flag (default
    # "fast", see `_FLY_ACCURACY_ENV`), not whatever AITER_FLY_AR_ACCURACY the
    # launching shell happens to export -- at the shipped default of
    # accuracy=exact this row's window is capped at oneshot_max_exact and the
    # mesh/ring rungs are never reached at all (see allreduce_policy.resolve).
    # The acceptance test for the whole heuristic -- it must stay within ~10%
    # of the best pinned row at every shape, which is what
    # `fit_allreduce_policy.py --audit-auto` checks (run with --fly-accuracy
    # fast to exercise the full three-family policy). Its accuracy floor has
    # to be the *quantized* one even in fast mode even though it is bit-exact
    # at decode sizes: one row spans both accuracy classes because the
    # schedule changes underneath it, which is exactly the thing being tested.
    Candidate(
        "fly_auto", "flyauto", 14.0, False
    ),  # 55 at decode / 18.7 at prefill (fast)
    Candidate("rccl", "rccl", 40.0, True),  # 51 / 69
    # ---- --fusion ar_rmsnorm rows ------------------------------------------
    # All-reduce + residual add + RMSNorm, i.e. what
    # `aiter::allreduce_fusion_kernel_1stage<T, T, N, false>` computes. These
    # never run in plain mode: `out` here is a *normalized* activation, graded
    # against a different reference, so putting it in the same column as a raw
    # sum would be meaningless.
    #
    # `fused_` means the kernel genuinely fuses; `separate_` is a two-launch
    # baseline (all-reduce, then aiter's rmsnorm2d_fwd_with_add). The prefix
    # describes the kernel, not the mode -- both families run under --fusion.
    #
    # The new FlyDSL kernel, ladder-driven. The row this whole mode exists for.
    Candidate("fused_fly_1stage", "fused_fly1s", 40.0, True, fusion=True),
    # ... plus the pinned block x grid-cap x self-skip grid, see
    # `_fused_fly1s_grid_rows`.
    *_fused_fly1s_grid_rows(),
    # The shipped fused dispatcher, routed through  FlyDSLAllReduceRMSNorm.
    Candidate("fused_fly_auto", "fused_flyauto", 10.0, False, fusion=True),
    # The incumbent: the kernel the Qwen3-235B MXFP4 decode trace spends 3.72 s
    # in, 7.1x the next kernel. Beating this is the point.
    Candidate(
        "fused_cdr_1stage", "fused_cdr", 40.0, True, fusion=True, use_1stage=True
    ),
    Candidate(
        "fused_cdr_2stage", "fused_cdr", 40.0, True, fusion=True, use_1stage=False
    ),
    # Quantized wire, so the same codec floors as the plain qr_* rows.
    Candidate("fused_qr_fp8", "fused_qr", 24.0, False, fusion=True, quant="FP8"),
    Candidate("fused_qr_int4", "fused_qr", 14.0, False, fusion=True, quant="INT4"),
    # Two-launch baselines. `separate_fly1s` is the one that isolates what the
    # fusion itself buys: same all-reduce as `fused_fly_1stage`, same arch,
    # differing only in whether the norm is fused.
    # Quantized fused rows: the two-shot schedules with the epilogue fused.
    #
    # The floor is *not* the plain schedule's own -- fusing through RMSNorm
    # changes what the metric measures, not just its value. The plain ring's
    # SQNR is over the raw all-reduce output; the fused SQNR is over
    # ``x * rsqrt(mean(x^2) + eps) * w``, and dividing by a row's own norm
    # amplifies whatever quantization variance that row already had. That
    # variance only averages out over enough rows.
    Candidate(
        "fused_fly_ring", "fused_flyqr", 10.0, False, fusion=True, algorithm="ring"
    ),
    Candidate(
        "fused_fly_mesh", "fused_flyqr", 15.0, False, fusion=True, algorithm="mesh"
    ),
    # ... plus their pinned-block rows, see `_fused_flyqr_grid_rows`.
    *_fused_flyqr_grid_rows(),
    Candidate("separate_cdr", "separate", 40.0, True, fusion=True, sep_ar="cdr"),
    Candidate("separate_rccl", "separate", 40.0, True, fusion=True, sep_ar="rccl"),
    # The incumbent quick-reduce's own two-launch baseline, so "does fusing
    # pay?" is answerable for the kernel we ship as well as for the new ones.
    Candidate(
        "separate_qr_int4",
        "separate",
        14.0,
        False,
        fusion=True,
        sep_ar="qr",
        quant="INT4",
    ),
    Candidate("separate_fly1s", "separate", 40.0, True, fusion=True, sep_ar="fly1s"),
    # The pair that makes `fused_fly_ring` mean something: the same schedule,
    # two launches (all-reduce, then aiter's rmsnorm2d_fwd_with_add) against
    # one. 6S of HBM traffic against 4S, plus the launch.
    Candidate(
        "separate_flyring",
        "separate",
        14.0,  # matches `fly_int4_ring`'s floor, not the mesh's. Flat
        # 19.5-19.7 dB across M=8..8192 at TP4/hidden=7168 (measured) -- the
        # plain kernel's fixed 32 KiB tile does not hit the few-tile regime
        # `fused_fly_ring` does, so it does not need that candidate's lower
        # floor.
        False,
        fusion=True,
        sep_ar="flyring",
        algorithm="ring",
    ),
    # Same for the mesh, so each schedule is compared against itself rather
    # than against the other one's two-launch baseline.
    Candidate(
        "separate_flymesh",
        "separate",
        15.0,
        False,
        fusion=True,
        sep_ar="flyring",
        algorithm="mesh",
    ),
)
CANDIDATE_KEYS = [c.key for c in CANDIDATES]
PRIMARY = "cdr"  # the kernel we ship, and the default baseline
# The fused-mode counterpart: the traced kernel, i.e. what production runs today.
FUSION_PRIMARY = "fused_cdr_1stage"
FUSION_MODES = ("none", "ar_rmsnorm")
# RMSNorm epsilon, matching the value in the Qwen3-235B trace's call signature.
FUSION_EPS = 1e-6

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


def _fused_hidden_ok(hidden: int, cand: Candidate) -> bool:
    """Whether the FlyDSL fused kernel has a build for this width.

    One block covers one token row there, so BLOCK = hidden/(8*atoms) has to be
    a wave64 multiple no wider than 1024. Unlike every other gate in this file
    that is a *compile-time* limit, not a policy, which is why it asks the
    kernel module rather than restating the arithmetic.

    A block-pinned row asks the question the other way round -- "does this width
    admit this block" -- and the answer is no for most (width, block) pairs:
    hidden=7168 has only 896 and 448, so a b256 row is *skipped* here rather
    than failing later. That is the same shape of answer as the atoms form, and
    it is why these rows can be generated across widths and left to gate.
    """
    if _fused_hidden_supported is None:
        return False
    if cand.block is not None:
        if _fused_block_options is None:
            return False
        return any(b == cand.block for b, _ in _fused_block_options(int(hidden)))
    return _fused_hidden_supported(int(hidden), 1 if cand.atoms is None else cand.atoms)


def _flyqr_block_ok(hidden: int, world_size: int, cand: Candidate) -> bool:
    """Whether a pinned block exists for the quantized fused schedules here.

    Unpinned rows take whatever ``quick_reduce_row_block`` picks, so they only
    need the width to fuse at all. A pinned row needs that *specific* width of
    workgroup, which most (hidden, world_size) pairs do not offer -- the
    two-shot geometry also has to land a rank-tile on the 64 B sector grid, and
    at TP8 that leaves exactly one block per width.
    """
    if _flyqr_hidden_supported is None or not _flyqr_hidden_supported(
        int(hidden), int(world_size)
    ):
        return False
    if cand.block is None:
        return True
    if _flyqr_block_options is None:
        return False
    opts = _flyqr_block_options(int(hidden), int(world_size))
    return any(b == cand.block for b, _ in opts)


def applicable(
    cand: Candidate,
    world_size: int,
    dtype,
    numel: int,
    nbytes: int,
    fusion: str = "none",
    hidden: int | None = None,
):
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
    # A fused candidate only runs under --fusion, and a plain one only outside
    # it. The two modes compute different functions and are graded against
    # different references; one table cannot hold both.
    if cand.fusion != (fusion != "none"):
        return False
    if nbytes % 16 != 0:
        # Every custom path requires 16B-aligned payloads; only RCCL survives.
        return cand.family in ("rccl", "separate") and cand.sep_ar in (None, "rccl")
    if cand.family == "fused_cdr":
        # can_use_fuse_ar_rms in communicator_cuda.py, plus the kernel's own
        # blockDim = out_hidden/8 <= 1024 limit.
        return (
            dtype in (dtypes.fp16, dtypes.bf16)
            and world_size != 6
            and hidden is not None
            and hidden <= 8192
        )
    if cand.family == "fused_qr":
        if world_size not in _QR_WORLDS or dtype not in _QR_DTYPES:
            return False
        # should_quick_allreduce_rmsnorm: a hidden row must tile a 32 KiB QR
        # tile evenly. At hidden=7168 bf16 (14336 B) it does not, and the host
        # falls back to unfused rather than launching the fused kernel.
        if hidden is None:
            return False
        row = hidden * (2 if dtype != dtypes.fp32 else 4)
        return row <= 32768 and 32768 % row == 0
    if cand.family == "fused_flyqr":
        # Gated on the fused geometry rather than on a payload floor: the block
        # is hidden/(8*atoms_per_row) threads, so only widths that land on the
        # 64 B sector grid can build at all. The engines here disable the size
        # floor, as the plain fly rows do.
        return (
            HAS_FLY_INT4
            and get_gfx() in _FLY_ARCHS
            and world_size in _FLY_WORLDS
            and dtype == dtypes.bf16
            and hidden is not None
            and _flyqr_block_ok(hidden, world_size, cand)
        )
    if cand.family == "fused_fly1s":
        return (
            HAS_FLY_INT4
            and get_gfx() in _FLY_ARCHS
            and world_size in _FLY_WORLDS
            and dtype == dtypes.bf16
            and hidden is not None
            and _fused_hidden_ok(hidden, cand)
            and nbytes <= _fly1s_ceiling(world_size)
        )
    if cand.family == "separate":
        if cand.sep_ar == "rccl":
            return True
        if cand.sep_ar == "cdr":
            return dtype in (dtypes.fp16, dtypes.bf16)
        if cand.sep_ar == "qr":
            return world_size in _QR_WORLDS and dtype in _QR_DTYPES
        if cand.sep_ar == "flyring":
            return (
                HAS_FLY_INT4
                and get_gfx() in _FLY_ARCHS
                and world_size in _FLY_WORLDS
                and dtype == dtypes.bf16
            )
        return (
            HAS_FLY_INT4
            and get_gfx() in _FLY_ARCHS
            and world_size in _FLY_WORLDS
            and dtype == dtypes.bf16
            and nbytes <= _fly1s_ceiling(world_size)
        )
    if cand.family == "cdr":
        if cand.fp8:
            return dtype == dtypes.fp16 and numel >= _FP8_MIN_NUMEL
        return True
    if cand.family == "qr":
        if world_size not in _QR_WORLDS or dtype not in _QR_DTYPES:
            return False
        # INT3 on TP4/TP8 is disabled upstream for poor kernel performance, not
        # for correctness -- benchmarking it there would advertise a path
        # production refuses to take. Unconditional: a stricter accuracy regime
        # must not be the thing that re-enables it.
        if cand.quant == "INT3" and world_size != 2:
            return False
        # Same rule as the `fly` mesh/ring rows below: an exact-mode deployment
        # ships no lossy kernel, so a quantizing quick-reduce level is not
        # "relevant for the mode" there. `FP` is the one level that is not a
        # codec -- it lands at the bf16 rounding floor alongside cdr.
        return cand.quant == "FP" or _bench_fly_accuracy_mode() == "fast"
    if cand.family == "fly":
        # Deliberately *not* gated on QuickAllReduceInt4's own payload floor, which the
        # engines here disable with min_bytes=0.
        return (
            HAS_FLY_INT4
            and get_gfx() in _FLY_ARCHS
            and world_size in _FLY_WORLDS
            and dtype == dtypes.bf16
            and _bench_fly_accuracy_mode() == "fast"
        )
    if cand.family in ("flyauto", "fused_flyauto"):
        # Gated by the dispatcher's own policy rather than by a constant here:
        # the whole point of the row is that its window is the shipped one.
        return (
            HAS_FLY_INT4
            and get_gfx() in _FLY_ARCHS
            and world_size in _FLY_WORLDS
            and dtype == dtypes.bf16
        )
    if cand.family == "fly1s":
        # Gated from above, not below: OneShotAllReduce.allreduce refuses
        # payloads over its ceiling because wire volume is (N-1)x the message.
        return (
            HAS_FLY_INT4
            and get_gfx() in _FLY_ARCHS
            and world_size in _FLY_WORLDS
            and dtype == dtypes.bf16
            and nbytes <= _fly1s_ceiling(world_size)
        )
    return True  # rccl


# The pinned fly1s rows default to this rather than to MAX_PAYLOAD_BYTES_BY_WORLD
# (the live oneshot_max_exact): that constant is a *policy* -- where fly_auto
# should stop dispatching to the one-shot -- not a correctness limit, since the
# kernel is exact at any size.
#
# 64 MiB rather than unbounded. The one-shot pushes (N-1)*S and handshakes once
# per tile, so at TP8/114 MiB it is ~7300 sequential round trips per call --
# minutes of wall clock per candidate under graph replay, to measure a size no
# policy would route here. 64 MiB is far past every shipped ceiling (the widest
# is 1.5 MiB), so a fit can still see the crossover and a long way beyond it,
# and it matches AITER_CUSTOM_AR_MAX_SIZE -- the largest payload any custom path
# handles. AITER_BENCH_FLY1S_MAX_KB moves it either way for one run.
_FLY1S_DEFAULT_CEILING = 64 << 20


def _fly1s_ceiling(world_size: int) -> int:
    """Payload ceiling for the one-shot rows, overridable for the sweep.

    Independent of world size and of whatever ``oneshot_max_exact`` is shipped
    today -- see the module constant's comment. ``AITER_BENCH_FLY1S_MAX_KB``
    moves it for a single run without editing this function, e.g. to focus a
    dispatch-ladder sweep on a specific window or to look past 64 MiB.
    """
    kb = os.environ.get("AITER_BENCH_FLY1S_MAX_KB")
    if kb:
        return int(kb) << 10
    # HAS_FLY_INT4, not `_fly1s_max_bytes is not None`: the ceiling no longer
    # derives from the shipped policy, so the import is only an availability
    # probe and saying so directly is clearer.
    return _FLY1S_DEFAULT_CEILING if HAS_FLY_INT4 else 0


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


def load_shapes_csv(path: str) -> list[tuple[int, int]]:
    """``(M, K)`` pairs from a CSV, for sweeps too long to put on a command line.

    ``M`` and ``K`` columns, uppercase, any extra columns ignored -- the same
    contract as ``test_gemm_a8w8_blockscale.py``'s ``--csv``, so a shape file is
    readable across the op_tests. An extra ``label`` column is conventional here
    for naming what a row is probing; it is carried nowhere and exists for
    whoever reads the file.

    The point is the dispatch sweeps: the crossovers this benchmark exists to
    find sit between the shapes ``L_SHAPE`` measures, and bracketing them takes
    ~50 sizes per world size. That is a file, not an argument list, and it wants
    to be committed next to the report it produced so the run is reproducible.

    Duplicates are dropped preserving order rather than silently timed twice.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"shape CSV not found: {p}")
    df = pd.read_csv(p)
    missing = {"M", "K"} - set(df.columns)
    if missing:
        raise ValueError(
            f"{p}: missing column(s) {sorted(missing)}; expected M,K "
            f"(got {list(df.columns)})"
        )
    shapes = []
    for i, row in df.iterrows():
        m, k = int(row["M"]), int(row["K"])
        if m < 1 or k < 1:
            raise ValueError(f"{p} row {i}: M and K must be positive, got {m},{k}")
        # Every custom path requires a 16 B-aligned payload and would be
        # silently dropped to the rccl-only row by `applicable()`. Refuse the
        # shape instead: in a shape file that is a typo, not a request.
        nbytes = m * k * 2
        if nbytes % 16 != 0:
            raise ValueError(
                f"{p} row {i}: {m}x{k} is {nbytes} B at 2 B/element, not a "
                "multiple of 16; every candidate but rccl would be skipped"
            )
        shapes.append((m, k))
    return list(dict.fromkeys(shapes))


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
    one-shot reads the whole buffer from every peer, a two-shot moves a
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


def _fused_1stage_policy(world_size: int):
    """``(use_1stage_override, limit_bytes)`` for the fused AR, from the env.

    The same two knobs ``CudaCommunicator`` reads at class-definition time
    (communicator_cuda.py:22-25). Read here rather than hardcoded so the
    ``prod path`` column describes the *deployment* -- if a shell exports
    either of these, production's 1stage/2stage split moves and a column that
    ignored them would quietly report the wrong kernel.
    """
    override = {"1": True, "0": False}.get(os.environ.get(_AR_1STAGE_ENV, ""))
    max_kb = int(os.environ.get(_AR_1STAGE_MAX_KB_ENV, "-1"))
    # The default limit is a fixed 1.75 MiB (128 tokens x DSv4's 7168 x bf16)
    # divided by the world size -- so it scales with TP but *not* with hidden.
    # At TP2 that is 896 KiB and at TP8 224 KiB, which is why the same
    # (M=32, hidden=4096) shape is 1stage at TP2 and 2stage at TP8.
    limit = max_kb * 1024 if max_kb >= 0 else 128 * 7168 * 2 // world_size
    return override, limit


def production_fused_path(ca_comm, qr_comm, x, weight, world_size: int, prod_regime):
    """What CudaCommunicator.fused_allreduce_rmsnorm would dispatch for *x*.

    Mirrors the gate chain in communicator_cuda.py: quick-reduce's fused kernel
    first (2-stage sizes only), then the custom fused AR, then the unfused
    all-reduce + rmsnorm split. Reported so a row can say `prod path = separate`
    while still carrying fused timings -- which is the point of the column.
    """
    from aiter.dist.device_communicators.quick_all_reduce import QuickReduceRegime

    hidden = weight.numel()
    total_bytes = x.numel() * x.element_size()
    can_use = hidden <= 16384 and total_bytes < 8 * 1024 * 8192 and world_size != 6
    override, limit = _fused_1stage_policy(world_size)
    use_1stage = override if override is not None else (total_bytes <= limit)

    if not use_1stage and qr_comm is not None and not qr_comm.disabled:
        saved = qr_comm.qr_quant_level
        try:
            if prod_regime in QuickReduceRegime.__members__ and prod_regime != "NONE":
                qr_comm.qr_quant_level = QuickReduceRegime[prod_regime]
                if qr_comm.should_quick_allreduce_rmsnorm(x, x, weight, hidden):
                    return f"qr_fused:{prod_regime.lower()}"
        finally:
            qr_comm.qr_quant_level = saved
    if (
        can_use
        and ca_comm is not None
        and not ca_comm.disabled
        and ca_comm.should_custom_ar(x)
    ):
        return f"cdr_fused:{'1stage' if use_1stage else '2stage'}"
    return "separate"


def _variant_of(
    cand: Candidate,
    fly,
    fly1s,
    flyauto,
    nbytes: int,
    *,
    fly1s_rms=None,
    flyqr_rms=None,
    fused_flyauto=None,
    hidden=None,
) -> str | None:
    """The kernel *cand* would actually run at *nbytes*, or None.

    Only the flydsl families can answer this: their engines expose the JIT
    symbol their factory stamped on the launch wrapper, which names every
    compile-time knob. The other families dispatch inside C++ or inside RCCL and
    have nothing equivalent to report, so they get ``n/a`` rather than a guess.
    """
    if cand.family == "fused_fly1s":
        # The fused engine's variant is a function of hidden as well as bytes:
        # the tile is one token row, so the width is baked into the symbol.
        eng = (fly1s_rms or {}).get(cand.fly1s_rms_cfg)
        return eng.variant(int(hidden), int(nbytes)) if eng is not None else None
    if cand.family == "fused_flyauto":
        # `<family>:<symbol>`, so the report says which family the shipped
        # fused policy picked here as well as which binary it ran.
        return (
            fused_flyauto.variant(int(hidden), int(nbytes))
            if fused_flyauto is not None
            else None
        )
    if cand.family == "fused_flyqr":
        # Same reason, for the same reason: a fused two-shot build sizes its
        # block to the row, so hidden reaches the symbol.
        eng = (flyqr_rms or {}).get(cand.flyqr_rms_cfg)
        return eng.variant(int(hidden), int(nbytes)) if eng is not None else None
    if cand.family == "separate":
        if cand.sep_ar == "fly1s":
            eng = fly1s.get(cand.fly1s_cfg)
        elif cand.sep_ar == "flyring":
            eng = fly.get(cand.fly_cfg)
        else:
            eng = None
        return eng.variant(int(nbytes)) if eng is not None else None
    eng = (
        fly.get(cand.fly_cfg)
        if cand.family == "fly"
        else (
            fly1s.get(cand.fly1s_cfg)
            if cand.family == "fly1s"
            else flyauto if cand.family == "flyauto" else None
        )
    )
    return eng.variant(int(nbytes)) if eng is not None else None


def _ran_exact(cand: Candidate, flyauto, nbytes: int) -> bool:
    """Whether *cand* is bit-accurate **at this shape**.

    For every other family exactness is a property of the candidate, because
    the candidate names one kernel. ``fly_auto`` names a *policy*: it dispatches
    to the bit-exact one-shot below its ceiling and to a quantized mesh/ring
    above it, so a single ``Candidate.exact`` flag cannot describe it.

    ``Candidate.exact`` stays the static answer and still drives the accuracy
    *floor*: ``fly_auto``'s floor has to stay the quantized one, since one
    column spans both accuracy classes and the floor has to admit the worst of
    them.
    """
    if cand.family != "flyauto":
        return cand.exact
    # The policy object is the only thing that knows which family this payload
    # reaches; "oneshot" is the exact one (bf16 widened to fp32, accumulated in
    # rank order, one rounding on output).
    return flyauto is not None and flyauto.family_for(int(nbytes)) == "oneshot"


# The ring bakes its rank into the kernel at compile time, so its symbol carries
# an ``_r<n>_`` field and every rank legitimately reports a different string for
# the same variant. Collapse that one field before comparing.
_RANK_FIELD = re.compile(r"_r\d+_")


def _agree_variant(per_rank) -> str | None:
    """One variant string for a row, or a flag that the ranks disagreed.

    Every rank must be running the same variant of the same kernel; if they are
    not, a latency taken as ``max`` over ranks is comparing two different
    binaries and the row is meaningless. That is not hypothetical -- the ring
    compiles per rank -- so it is checked rather than assumed, and a
    disagreement is reported in the cell instead of being averaged away.
    """
    seen = {_RANK_FIELD.sub("_r*_", v) for v in per_rank if v is not None}
    if not seen:
        return None
    if len(seen) > 1:
        logger.warning("ranks disagree on the kernel variant: %s", sorted(seen))
        return "MIXED: " + " | ".join(sorted(seen))
    return seen.pop()


def _cfg_order(cfg: tuple) -> tuple:
    """Total order over engine-config tuples, for a deterministic build order.

    Every engine does its own IPC handle exchange, which is a collective, so
    ranks disagreeing on the construction order deadlock. The tuples mix
    ``None`` ("constructor default") with ints and strings, and ``None`` does
    not order against either, so sort on ``(is-set, string form)`` per field
    rather than on the tuple itself. String form because a single key has to
    cover ``super_tile`` (int) and ``rs_codec`` (str) in the same position
    across the two families -- the order only has to be *stable*, not
    numerically meaningful.
    """
    return tuple(x for f in cfg for x in ((f is not None), f"{f}"))


def _fly_kwargs(cfg: tuple, names: tuple) -> dict:
    """Non-``None`` fields of *cfg* as constructor kwargs, named by *names*.

    ``None`` means "leave the constructor default alone", which is not the same
    as passing the default explicitly: ``QuickAllReduceInt4`` distinguishes an unset
    ``super_tile`` (walk the ladder) from a pinned one (this value at every
    size), and an unset codec from a pinned one.
    """
    return {n: v for n, v in zip(names, cfg) if v is not None}


def _fusion_inputs(tokens: int, hidden: int, dtype, device):
    """Residual and gain for the fused mode, identical on every rank.

    Rank-independent on purpose: the residual is a transformer block's input,
    which TP replicates, and the norm gain is a weight. Seeding them per rank
    would make the reference below wrong in a way that looks like a kernel bug.
    """
    residual = _make_input(4242, tokens, hidden, dtype).to(device)
    weight = _make_input(9999, 1, hidden, dtype).reshape(hidden).to(device)
    return residual, weight


def _fusion_reference(tp_size, tokens, hidden, dtype, device, residual, weight):
    """fp32 reference for ``(out, residual_out)``, matching the C++ contract.

    The ``.to(dtype).to(fp32)`` round-trip after the sum is deliberate and is
    what ``allreduce_fusion_kernel_1stage`` does: the unfused path all-reduces
    into a bf16 tensor before the residual add, and the fused kernel is a
    drop-in, so it discards the same mantissa bits. Leaving it out would put
    this reference ~1 ULP from *every* candidate rather than from none.
    """
    ar = torch.zeros((tokens, hidden), dtype=dtypes.fp32, device=device)
    for peer in range(tp_size):
        ar += _make_input(peer, tokens, hidden, dtype).to(device, dtypes.fp32)
    ar = ar.to(dtype).to(dtypes.fp32)
    s = ar + residual.to(dtypes.fp32)
    rstd = torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + FUSION_EPS)
    out_ref = (s * rstd * weight.to(dtypes.fp32)).to(dtype)
    return out_ref.to(dtypes.fp32), s.to(dtype).to(dtypes.fp32)


def _build_fused_thunks(
    cands,
    *,
    ca_comm,
    qr_comm,
    fly,
    fly1s,
    fly1s_rms,
    flyqr_rms,
    fused_flyauto,
    group,
    x,
    residual,
    weight,
):
    """Zero-arg thunks returning ``(out, residual_out)`` for the fused mode.

    Every thunk preallocates **both** outputs and passes them in. That is not
    tidiness: ``ca_comm.fused_ar_rms`` allocates whatever it is not given, and
    an allocation inside a HIP-graph capture is at best a surprise.
    """
    from aiter import rmsnorm2d_fwd_with_add
    from aiter.dist.device_communicators.quick_all_reduce import QuickReduceRegime

    thunks = {}
    buffers = []
    hidden = x.shape[-1]
    for cand in cands:
        out = torch.empty_like(x)
        res_out = torch.empty_like(x)
        buffers += [out, res_out]
        if cand.family == "fused_cdr":

            def _f(o=out, ro=res_out, c=cand):
                ca_comm.fused_ar_rms(
                    x,
                    residual,
                    res_out=ro,
                    out=o,
                    w=weight,
                    eps=FUSION_EPS,
                    registered=False,
                    use_1stage=c.use_1stage,
                )
                return o, ro

            thunks[cand.key] = _f
        elif cand.family == "fused_qr":

            def _f(o=out, ro=res_out, lvl=QuickReduceRegime[cand.quant], h=hidden):
                qr_comm.qr_quant_level = lvl
                a, b = qr_comm.quick_all_reduce_rmsnorm(
                    x, residual, weight, FUSION_EPS, h
                )
                o.copy_(a)
                ro.copy_(b)
                return o, ro

            thunks[cand.key] = _f
        elif cand.family == "fused_fly1s":

            def _f(o=out, ro=res_out, eng=fly1s_rms[cand.fly1s_rms_cfg]):
                eng.allreduce_rmsnorm(
                    x, residual, weight, FUSION_EPS, out=o, residual_out=ro
                )
                return o, ro

            thunks[cand.key] = _f
        elif cand.family == "fused_flyqr":

            def _f(o=out, ro=res_out, eng=flyqr_rms[cand.flyqr_rms_cfg]):
                eng.allreduce_rmsnorm(
                    x, residual, weight, FUSION_EPS, out=o, residual_out=ro
                )
                return o, ro

            thunks[cand.key] = _f
        elif cand.family == "fused_flyauto":
            def _f(o=out, ro=res_out, comm=fused_flyauto):
                comm.fly_fused_ar_rms(
                    x, residual, weight, FUSION_EPS, out=o, res_out=ro
                )
                return o, ro

            thunks[cand.key] = _f
        else:  # separate: all-reduce, then a standalone norm
            ar_buf = torch.empty_like(x)
            buffers.append(ar_buf)

            if cand.sep_ar == "cdr":

                def _ar(b=ar_buf):
                    return ca_comm.all_reduce(x, out=b)

            elif cand.sep_ar == "rccl":

                def _ar(b=ar_buf):
                    b.copy_(x)
                    dist.all_reduce(b, group=group)
                    return b

            elif cand.sep_ar == "qr":

                def _ar(b=ar_buf, lvl=QuickReduceRegime[cand.quant]):
                    qr_comm.qr_quant_level = lvl
                    return qr_comm.quick_all_reduce(x, out=b)

            elif cand.sep_ar == "flyring":

                def _ar(b=ar_buf, eng=fly[cand.fly_cfg]):
                    eng.allreduce(x, b)
                    return b

            else:

                def _ar(b=ar_buf, eng=fly1s[cand.fly1s_cfg]):
                    eng.allreduce(x, b)
                    return b

            def _f(o=out, ro=res_out, ar=_ar):
                got = ar()
                rmsnorm2d_fwd_with_add(o, got, residual, ro, weight, FUSION_EPS)
                return o, ro

            thunks[cand.key] = _f
    return thunks, buffers


def _bench_graph(thunk, *, num_iters, num_warmup, inner, group, label="candidate"):
    """hipEvent-timed HIP-graph replay. Returns ``(output, us_per_call)``.

    Same ``(data, time)`` order ``run_perftest`` returns, so the two timing
    paths in ``_bench_shape`` unpack identically.

    Why not ``run_perftest(testGraph=True)``: it times the replay through the
    torch profiler, and this bench cannot use the profiler at all -- spawned
    ranks get one that records CPU ops but no GPU activity, and the custom-AR
    and FlyDSL candidates register no aten op, so the event table comes back
    empty and ``get_trace_perf`` raises on the missing ``host_time_sum``.

    ``graph_capture()`` is required, not cosmetic. It enters ``ca_comm.capture()``,
    whose exit flushes the buffer addresses the graph recorded
    (``custom_all_reduce.py:capture``) and whose ``_IS_CAPTURING`` routes
    ``custom_fused_ar_rms`` down its capture branch, and it owns the side stream
    RCCL capture needs. ``stream=gc.stream`` is what puts the capture on *that*
    stream rather than on ``torch.cuda.graph``'s own class-level one. Capturing
    without either records a different code path than the one that replays.

    ``inner`` calls per graph, so the replay is back-to-back collectives with no
    host in between -- the run-ahead case the double-buffered inbox is built
    for, and a closer model of production than eager is.

    What comes back is whatever the thunk returns -- a tensor in plain mode, the
    ``(out, residual_out)`` pair in fused mode -- and it is what a **replay**
    produced, not what a subsequent eager call produced. Every returned buffer
    is poisoned and the graph replayed once more before it is read, so a capture
    that dropped a launch or replayed a stale buffer fails the SQNR gate.
    Grading an eager call instead would score that capture clean, because every
    thunk writes into the same preallocated output and the eager call would
    simply overwrite the evidence.
    """
    from aiter.dist.parallel_state import graph_capture

    for _ in range(max(1, num_warmup)):
        thunk()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    graph = torch.cuda.CUDAGraph()
    out = None
    try:
        with graph_capture() as gc, torch.cuda.graph(graph, stream=gc.stream):
            for _ in range(inner):
                out = thunk()
    except Exception as exc:
        # Not every candidate is guaranteed capturable -- flyauto in particular
        # is unverified. Say which one and how to get a number anyway, rather
        # than dying on a bare HIP error.
        raise RuntimeError(
            f"{label}: HIP graph capture failed ({exc}). Re-run with "
            "--timing eager to measure this candidate on the host path instead."
        ) from exc
    torch.cuda.synchronize()
    dist.barrier(group=group)

    reps = max(1, num_iters // inner)
    graph.replay()  # one untimed replay: the first is cold
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    us = start.elapsed_time(end) * 1000.0 / (reps * inner)

    # Barriered on both sides of the poison: a peer still replaying into this
    # rank's inbox would otherwise race the fill_, and every rank must reach
    # the grading replay having issued the same number of colour increments.
    dist.barrier(group=group)
    for buf in out if isinstance(out, tuple) else (out,):
        buf.fill_(float("nan"))
    torch.cuda.synchronize()
    dist.barrier(group=group)
    graph.replay()
    torch.cuda.synchronize()
    return out, us


def _build_thunks(cands, *, ca_comm, qr_comm, fly, fly1s, flyauto, group, x):
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
        elif cand.family == "fly1s":

            def _fly1s(o=out, eng=fly1s[cand.fly1s_cfg]):
                eng.allreduce(x, o)
                return o

            thunks[cand.key] = _fly1s
        elif cand.family == "flyauto":

            def _flyauto(o=out, comm=flyauto):
                comm.fly_all_reduce(x, out=o)
                return o

            thunks[cand.key] = _flyauto
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
    fly1s,
    flyauto,
    keys,
    prod_regime,
    timing,
    graph_inner,
    fusion="none",
    fly1s_rms=None,
    flyqr_rms=None,
    fused_flyauto=None,
):
    """Time and grade every applicable candidate at one shape. Scalars only."""
    device = torch.device(f"cuda:{rank}")
    x = _make_input(rank, tokens, hidden, dtype).to(device)
    nbytes = x.numel() * x.element_size()
    fused = fusion != "none"
    residual = weight = res_ref = None
    if fused:
        residual, weight = _fusion_inputs(tokens, hidden, dtype, device)

    cands = [
        c
        for c in CANDIDATES
        if c.key in keys
        and applicable(c, tp_size, dtype, x.numel(), nbytes, fusion, hidden)
        and not (c.family in ("qr", "fused_qr") and qr_comm is None)
        and not (c.family == "separate" and c.sep_ar == "qr" and qr_comm is None)
        and not (c.family == "fly" and c.fly_cfg not in fly)
        and not (c.family == "fly1s" and c.fly1s_cfg not in fly1s)
        and not (c.family == "fused_fly1s" and c.fly1s_rms_cfg not in (fly1s_rms or {}))
        and not (c.family == "fused_flyqr" and c.flyqr_rms_cfg not in (flyqr_rms or {}))
        and not (
            c.family == "separate" and c.sep_ar == "fly1s" and c.fly1s_cfg not in fly1s
        )
        and not (
            c.family == "separate" and c.sep_ar == "flyring" and c.fly_cfg not in fly
        )
        # flyauto's window is dynamic (depends on AITER_FLY_AR_ACCURACY, only
        # known once the object exists), unlike every other family's static
        # applicable() check -- and under the default accuracy=exact policy it
        # is a real "n/a" above oneshot_max, since that policy builds no
        # mesh/ring engine at all. should_fly_all_reduce is the one predicate
        # that already knows this; calling fly_all_reduce without checking it
        # first is a KeyError on any shape past the ceiling.
        and not (
            c.family == "flyauto"
            and (flyauto is None or not flyauto.should_fly_all_reduce(x))
        )
        and not (
            c.family == "fused_flyauto"
            and (
                fused_flyauto is None
                or not fused_flyauto.should_fly_fused_ar_rms(x, residual, weight)
            )
        )
    ]
    if fused:
        thunks, buffers = _build_fused_thunks(
            cands,
            ca_comm=ca_comm,
            qr_comm=qr_comm,
            fly=fly,
            fly1s=fly1s,
            fly1s_rms=fly1s_rms,
            flyqr_rms=flyqr_rms,
            fused_flyauto=fused_flyauto,
            group=group,
            x=x,
            residual=residual,
            weight=weight,
        )
        ref, res_ref = _fusion_reference(
            tp_size, tokens, hidden, dtype, device, residual, weight
        )
    else:
        thunks, buffers = _build_thunks(
            cands,
            ca_comm=ca_comm,
            qr_comm=qr_comm,
            fly=fly,
            fly1s=fly1s,
            flyauto=flyauto,
            group=group,
            x=x,
        )
        # fp32 sum of every rank's contribution, accumulated one peer at a time
        # so peak memory stays at ~2 activations.
        ref = torch.zeros((tokens, hidden), dtype=dtypes.fp32, device=device)
        for peer in range(tp_size):
            ref += _make_input(peer, tokens, hidden, dtype).to(device, dtypes.fp32)

    ret = {
        "nbytes": nbytes,
        "prod": (
            production_fused_path(ca_comm, qr_comm, x, weight, tp_size, prod_regime)
            if fused
            else production_path(ca_comm, qr_comm, x, tp_size, prod_regime)
        ),
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
        if timing == "graph":
            # `_bench_graph` warms, captures, times the replay and hands back
            # the output a replay produced -- so the SQNR/allclose gates below
            # cover the captured path rather than an eager re-run.
            got, us = _bench_graph(
                thunks[cand.key],
                num_iters=num_iters,
                num_warmup=num_warmup,
                inner=graph_inner,
                group=group,
                label=f"{cand.key} tp{tp_size} {tokens}x{hidden}",
            )
        else:
            got, us = run_perftest(
                thunks[cand.key],
                num_iters=num_iters,
                num_warmup=num_warmup,
                use_cuda_event=True,
            )
        # In fused mode a thunk returns (out, residual_out); `out` carries the
        # headline columns and residual_out is checked separately below.
        res_got = None
        if fused:
            got, res_got = got

        sqnr = sqnr_db(got, ref)
        assert sqnr >= cand.sqnr_floor, (
            f"{cand.key} tp{tp_size} {tokens}x{hidden} rank{rank}: "
            f"SQNR {sqnr:.2f} dB below the {cand.sqnr_floor} dB floor"
        )
        # Per shape, not per candidate: fly_auto is exact only where its policy
        # reaches the one-shot.
        ran_exact = _ran_exact(cand, flyauto, nbytes)
        if ran_exact:
            checkAllclose(
                ref,
                got.to(dtypes.fp32),
                rtol=1e-2,
                atol=1e-2,
                msg=f"{cand.key} tp{tp_size} {tokens}x{hidden} rank{rank}",
            )
        if fused and cand.exact:
            # The second output is graded too. An epilogue can get `out` right
            # and `residual_out` wrong -- they come from different points in the
            # dataflow -- and a fused kernel that corrupts the residual poisons
            # every later layer while looking fine here.
            res_sqnr = sqnr_db(res_got, res_ref)
            assert res_sqnr >= cand.sqnr_floor, (
                f"{cand.key} tp{tp_size} {tokens}x{hidden} rank{rank}: "
                f"residual_out SQNR {res_sqnr:.2f} dB below the "
                f"{cand.sqnr_floor} dB floor"
            )
        ret[f"{cand.key}_us"] = us
        ret[f"{cand.key}_sqnr"] = sqnr
        ret[f"{cand.key}_exact"] = ran_exact
        # Resolved after the run, not before: for a ladder-driven engine the
        # variant is a function of the payload, and asking the engine is the
        # only way to learn which rung this size took.
        ret[f"{cand.key}_variant"] = _variant_of(
            cand,
            fly,
            fly1s,
            flyauto,
            nbytes,
            fly1s_rms=fly1s_rms,
            flyqr_rms=flyqr_rms,
            fused_flyauto=fused_flyauto,
            hidden=hidden,
        )

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
    del thunks, buffers, ref, x, residual, weight, res_ref
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
    timing,
    graph_inner,
    fusion="none",
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
    logger.info("rank %d: aiter package %s", rank, _aiter_origin())
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

    fly = {}  # QuickAllReduceInt4 config tuple -> engine
    # One engine per distinct (schedule, super_tile, grid_cap, rs_codec,
    # ag_codec): each owns its own IPC inbox, whose layout depends on all five.
    # Sorted so every rank performs its handle exchanges in the same sequence --
    # the exchange is a collective, so a differing order across ranks deadlocks.
    # ``None`` means "constructor default" and does not order against an int,
    # so sort on a total key rather than the tuple itself.
    wanted_cfgs = sorted(
        {
            c.fly_cfg
            for c in CANDIDATES
            if c.key in keys
            # separate_flyring runs the *plain* ring and then a standalone
            # norm, so it draws from this pool rather than from the fused one.
            and (
                c.family == "fly" or (c.family == "separate" and c.sep_ar == "flyring")
            )
        },
        key=_cfg_order,
    )
    if (
        wanted_cfgs
        and HAS_FLY_INT4
        and get_gfx() in _FLY_ARCHS
        and tp_size in _FLY_WORLDS
        and dtype == dtypes.bf16
        and _bench_fly_accuracy_mode() == "fast"
    ):
        for cfg in wanted_cfgs:
            # QuickAllReduceInt4 exchanges IPC handles via broadcast_object_list, so it
            # needs the gloo (CPU) group -- it rejects an NCCL group outright.
            fly[cfg] = QuickAllReduceInt4(
                group=tp_group.cpu_group,
                device=device,
                rank=rank,
                world_size=tp_size,
                algorithm=cfg[0],
                # Measure every size the sweep asks for.
                min_bytes=0,
                **_fly_kwargs(
                    cfg[1:], ("super_tile", "grid_cap", "rs_codec", "ag_codec")
                ),
            )
        # compile() JIT-compiles every super-tile engine without launching any
        # of them (quick_allreduce_int4.compile_only), so one call at any shape keeps every
        # timed region below free of a first-call JIT stall.
        warm = torch.zeros((8, DSV4_HIDDEN), dtype=dtypes.bf16, device=device)
        for cfg in wanted_cfgs:
            dist.barrier(group=group)
            fly[cfg].compile_and_launch(warm, torch.empty_like(warm))
        del warm

    fly1s = {}  # fly1s_cfg tuple -> OneShotAllReduce engine
    # Same rules as the QuickAllReduceInt4 engines above: one per distinct config, each with
    # its own IPC inbox, constructed in a total order because the handle
    # exchange is a collective.
    wanted_1s = sorted(
        {
            c.fly1s_cfg
            for c in CANDIDATES
            if c.key in keys
            # separate_fly1s runs the *plain* one-shot and then a standalone
            # norm, so it needs an engine from this pool rather than a fused one.
            and (
                c.family == "fly1s" or (c.family == "separate" and c.sep_ar == "fly1s")
            )
        },
        key=_cfg_order,
    )
    if (
        wanted_1s
        and HAS_FLY_INT4
        and get_gfx() in _FLY_ARCHS
        and tp_size in _FLY_WORLDS
        and dtype == dtypes.bf16
    ):
        for cfg in wanted_1s:
            kw = _fly_kwargs(cfg, ("atoms", "grid_cap", "fanout", "skip_self", "block"))
            fly1s[cfg] = OneShotAllReduce(
                group=tp_group.cpu_group,
                device=device,
                rank=rank,
                world_size=tp_size,
                max_bytes=_fly1s_ceiling(tp_size),
                **kw,
            )
        warm = torch.zeros((8, DSV4_HIDDEN), dtype=dtypes.bf16, device=device)
        for cfg in wanted_1s:
            dist.barrier(group=group)
            fly1s[cfg].compile_and_launch(warm, torch.empty_like(warm))
        del warm

    fly1s_rms = {}  # (atoms, grid_cap, fanout) -> OneShotAllReduceRMSNorm engine
    # One object per distinct config, each building a per-hidden IPC inbox on
    # demand (the fused tile is one token row, so the wire layout depends on the
    # width). Same total-order rule as above: every build is a collective.
    wanted_rms = sorted(
        {
            c.fly1s_rms_cfg
            for c in CANDIDATES
            if c.family == "fused_fly1s" and c.key in keys
        },
        key=_cfg_order,
    )
    if (
        wanted_rms
        and HAS_FLY_INT4
        and get_gfx() in _FLY_ARCHS
        and tp_size in _FLY_WORLDS
        and dtype == dtypes.bf16
    ):
        for cfg in wanted_rms:
            kw = _fly_kwargs(cfg, ("atoms", "grid_cap", "fanout", "skip_self", "block"))
            fly1s_rms[cfg] = OneShotAllReduceRMSNorm(
                group=tp_group.cpu_group,
                device=device,
                rank=rank,
                world_size=tp_size,
                max_bytes=_fly1s_ceiling(tp_size),
                **kw,
            )
        # Warm every (config, hidden) this sweep will touch, before any timing
        # and well before any graph capture: a FlyDSL JIT compile inside a
        # capture is fatal, and a lazily-built engine is a collective.
        for hidden in sorted({h for _, h in shapes}):
            warm = torch.zeros((8, hidden), dtype=dtypes.bf16, device=device)
            w = torch.zeros(hidden, dtype=dtypes.bf16, device=device)
            for cfg, eng in fly1s_rms.items():
                if not eng.supports_hidden(hidden):
                    continue
                dist.barrier(group=group)
                eng.compile_and_launch(warm, warm.clone(), w, FUSION_EPS)
            del warm, w

    flyqr_rms = {}  # (algorithm, st, grid_cap, rs, ag) -> QuickAllReduceInt4RMSNorm
    # One object per distinct config, each building a per-hidden IPC inbox on
    # demand: a fused two-shot build sizes its block to the token row, so the
    # whole geometry -- tile, wire slots, codec offsets -- depends on the width.
    # Same total-order rule as every pool above; each build is a collective.
    wanted_qr_rms = sorted(
        {
            c.flyqr_rms_cfg
            for c in CANDIDATES
            if c.family == "fused_flyqr" and c.key in keys
        },
        key=_cfg_order,
    )
    if (
        wanted_qr_rms
        and HAS_FLY_INT4
        and get_gfx() in _FLY_ARCHS
        and tp_size in _FLY_WORLDS
        and dtype == dtypes.bf16
    ):
        for cfg in wanted_qr_rms:
            flyqr_rms[cfg] = QuickAllReduceInt4RMSNorm(
                group=tp_group.cpu_group,
                device=device,
                rank=rank,
                world_size=tp_size,
                algorithm=cfg[0],
                **_fly_kwargs(
                    cfg[1:],
                    ("super_tile", "grid_cap", "rs_codec", "ag_codec", "block"),
                ),
            )
            # Measure every size the sweep asks for, as the plain fly rows do.
            flyqr_rms[cfg].min_bytes = 0
        # Warm every (config, hidden) this sweep will touch, before any timing
        # and well before any graph capture: a FlyDSL JIT compile inside a
        # capture is fatal, and a lazily-built engine is a collective.
        for hidden in sorted({h for _, h in shapes}):
            warm = torch.zeros((8, hidden), dtype=dtypes.bf16, device=device)
            w = torch.zeros(hidden, dtype=dtypes.bf16, device=device)
            for cfg, eng in flyqr_rms.items():
                if not eng.supports_hidden(hidden):
                    continue
                dist.barrier(group=group)
                eng.compile_and_launch(warm, warm.clone(), w, FUSION_EPS)
            del warm, w

    # Production dispatch, built last so its three internal engines exchange
    # handles after every pinned one -- the exchange is a collective and the
    # order has to match across ranks. It self-disables unless AITER_FLY_AR is
    # set, which main() does when this row is in the sweep, alongside
    # AITER_FLY_AR_ACCURACY from --fly-accuracy.
    flyauto = None
    if (
        any(c.family == "flyauto" and c.key in keys for c in CANDIDATES)
        and HAS_FLY_INT4
        and get_gfx() in _FLY_ARCHS
        and tp_size in _FLY_WORLDS
        and dtype == dtypes.bf16
    ):
        dist.barrier(group=group)
        comm = FlyDSLAllReduce(group=tp_group.cpu_group, device=device)
        if comm.disabled:
            logger.warning(
                "rank %d: fly_auto requested but FlyDSLAllReduce disabled "
                "itself; its column will be absent",
                rank,
            )
        else:
            flyauto = comm
            # One warm call per reachable family, so nothing JITs inside a
            # timed region. A family is only compiled by a payload that reaches
            # it, so the probes are sited just inside each window: at the
            # one-shot ceiling, one token above it, and one token above the
            # mesh ceiling.
            tok = DSV4_HIDDEN * 2
            reachable = policy.families_reachable(flyauto.policy)
            # Probe sites: just inside each window boundary. Ring probe is only
            # computed when mesh_max is finite (otherwise ring is not reachable
            # and mesh_max=None would make the arithmetic nonsensical).
            probes = {
                "oneshot": max(1, flyauto.policy.oneshot_max // tok),
                "mesh": max(1, flyauto.policy.oneshot_max // tok + 1),
            }
            if flyauto.policy.mesh_max is not None:
                probes["ring"] = max(1, flyauto.policy.mesh_max // tok + 1)
            for family, m in probes.items():
                # Everything decidable from the byte count is decided before
                # the allocation.
                nbytes = m * tok
                if (
                    family not in reachable
                    or flyauto.family_for(nbytes) != family
                ):
                    continue  # window too narrow, or this policy never reaches `family`
                t = torch.zeros((m, DSV4_HIDDEN), dtype=dtypes.bf16, device=device)
                # The residual tensor-shaped checks (dtype, contiguity, and
                # the family actually having an engine) only
                # should_fly_all_reduce can make.
                if not flyauto.should_fly_all_reduce(t):
                    del t
                    continue
                dist.barrier(group=group)
                flyauto.fly_all_reduce(t, out=torch.empty_like(t))
                del t

    fused_flyauto = None
    if (
        any(c.family == "fused_flyauto" and c.key in keys for c in CANDIDATES)
        and HAS_FLY_INT4
        and get_gfx() in _FLY_ARCHS
        and tp_size in _FLY_WORLDS
        and dtype == dtypes.bf16
    ):
        dist.barrier(group=group)
        comm = FlyDSLAllReduceRMSNorm(group=tp_group.cpu_group, device=device)
        if comm.disabled:
            logger.warning(
                "rank %d: fused_fly_auto requested but FlyDSLAllReduceRMSNorm "
                "disabled itself; its column will be absent",
                rank,
            )
        else:
            fused_flyauto = comm
            for tokens, hidden in shapes:
                warm = torch.zeros((tokens, hidden), dtype=dtypes.bf16, device=device)
                w = torch.zeros(hidden, dtype=dtypes.bf16, device=device)
                dist.barrier(group=group)
                if fused_flyauto.should_fly_fused_ar_rms(warm, warm, w):
                    fused_flyauto.fly_fused_ar_rms(warm, warm.clone(), w, FUSION_EPS)
                del warm, w

    # Every fly candidate is a distinct engine with a distinct IPC inbox, and
    # they are all live at once for the whole sweep. A wide tuning sweep is
    # therefore holding a fixed cost on the device before a single payload is
    # allocated -- and the inbox scales with ST * grid, so the ring's high rungs
    # dominate it. Report it here, where it is attributable to the candidate
    # list, rather than letting it surface as an OOM on the largest shape.
    if fly or fly1s or fly1s_rms or flyqr_rms or flyauto or fused_flyauto:
        per_engine = sorted(
            [(f"fly{cfg}", eng.inbox_bytes) for cfg, eng in fly.items()]
            + [(f"fly1s{cfg}", eng.inbox_bytes) for cfg, eng in fly1s.items()]
            + [(f"fly1s_rms{cfg}", eng.inbox_bytes) for cfg, eng in fly1s_rms.items()]
            + [(f"flyqr_rms{cfg}", eng.inbox_bytes) for cfg, eng in flyqr_rms.items()]
            + ([("fly_auto", flyauto.inbox_bytes)] if flyauto else [])
            + (
                [("fused_fly_auto", fused_flyauto.inbox_bytes)] if fused_flyauto else []
            ),
            key=lambda kv: -kv[1],
        )
        total = sum(b for _, b in per_engine)
        logger.info(
            "rank %d: %d flydsl engine(s), %.1f MiB of IPC inbox; largest: %s",
            rank,
            len(per_engine),
            total / 2**20,
            ", ".join(f"{k} {b / 2**20:.1f} MiB" for k, b in per_engine[:3]),
        )

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
                fly1s=fly1s,
                flyauto=flyauto,
                keys=keys,
                prod_regime=prod_regime,
                timing=timing,
                graph_inner=graph_inner,
                fusion=fusion,
                fly1s_rms=fly1s_rms,
                flyqr_rms=flyqr_rms,
                fused_flyauto=fused_flyauto,
            )
            for tokens, hidden in shapes
        ]
    finally:
        # Every engine built above, not just the two-shot ones: each holds an
        # IPC inbox and a peer mapping per rank, and a wide sweep builds tens
        # of them.
        for eng in (*fly.values(), *fly1s.values()):
            eng.close()
        for eng in flyqr_rms.values():
            eng.close()
        for eng in fly1s_rms.values():
            eng.close()
        if flyauto is not None:
            flyauto.close()
        if fused_flyauto is not None:
            fused_flyauto.close()
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
        # Median across ranks, for fitting dispatch thresholds. `us` above is
        # the right *reporting* metric -- the model waits on the slowest rank --
        # but it is also the noisiest, since it takes the worst of N samples and
        # a single straggler moves it by tens of microseconds (see `spread us`).
        # A threshold fitted against that noise lands in the wrong place; a
        # threshold is a question about the kernel, not about arrival skew.
        #
        # Upper median on an even rank count, deliberately -- it errs towards
        # the metric above rather than away from it. At TP2 that makes this
        # column identical to `us` by construction; there are only two samples
        # and nothing to reject.
        row[f"{cand.key} median us"] = sorted(per_rank)[len(per_rank) // 2]
        row[f"{cand.key} variant"] = _agree_variant(
            [r.get(f"{cand.key}_variant") for r in rank_rets]
        )
        row[f"{cand.key} SQNR dB"] = min(r[f"{cand.key}_sqnr"] for r in rank_rets)
        # Per shape, because fly_auto's accuracy class is a function of the
        # payload.
        row[f"{cand.key} exact"] = all(
            r.get(f"{cand.key}_exact", False) for r in rank_rets
        )
        # Rank spread, per candidate. Reported for every row rather than only
        # for PRIMARY: skew is mostly a property of the barrier, but not
        # entirely, and a candidate that compiles a *different kernel per rank*
        # -- the ring bakes rank into its cache key where the mesh passes it as
        # a runtime argument -- can in principle land one rank with worse code
        # than the others. That shows up here and nowhere else in the report.
        #
        # Read it knowing what it cannot see: these collectives are
        # barrier-synchronised, so a slow rank stalls its peers in the
        # handshake and they all retire together. A near-zero spread therefore
        # means "no skew *outside* the collective", not "every rank did equal
        # work". A large spread is still worth chasing; a small one does not
        # by itself acquit a straggler.
        row[f"{cand.key} spread us"] = us - min(per_rank)
    return row


def run_sweep(tp_size, shapes, dtype, args, keys, prod_regime):
    """Spawn one process per rank and sweep every shape inside them."""
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    init_method = get_distributed_init_method(get_ip(), get_open_port())
    logger.info(
        "TP%d %s: %d shape(s), %d iters, timing %s",
        tp_size,
        dtype2str(dtype),
        len(shapes),
        args.iters,
        (f"graph (inner {args.graph_inner})" if args.timing == "graph" else "eager"),
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
                    args.timing,
                    args.graph_inner,
                    args.fusion,
                ),
            )
            for r in range(tp_size)
        ]
        pool.close()
        # Not pool.join(): every rank's _worker shares one process group, so a
        # `dist.barrier()` a few lines into any candidate needs all tp_size
        # ranks to reach it. If one rank returns early -- success or exception,
        # fewer collective calls than its peers either way -- the barrier
        # sequence desyncs permanently and the remaining ranks spin in that
        # barrier forever. pool.join() waits for every worker process to
        # exit, so it would hang right along with them, silently sitting on
        # top of a result (or exception). Poll instead, and the moment any one 
        # rank's result is ready, fetch it -- an exception surfaces immediately 
        # instead of waiting behind peers that will now never finish.
        pending = set(range(len(rets)))
        while pending:
            for i in sorted(pending):
                if not rets[i].ready():
                    continue
                pending.discard(i)
                try:
                    rets[i].get()
                except Exception:
                    stuck = sorted(pending)
                    logger.error(
                        "rank %d failed (see traceback below); rank(s) %s were "
                        "still running and will now be killed -- a per-rank "
                        "failure desyncs the barrier sequence, so they were "
                        "never going to finish on their own",
                        i,
                        stuck,
                    )
                    pool.terminate()
                    pool.join()
                    raise
            if pending:
                time.sleep(1.0)
        pool.join()
    per_rank = [r.get() for r in rets]
    return [
        _row(tp_size, tokens, hidden, dtype, [pr[i] for pr in per_rank])
        for i, (tokens, hidden) in enumerate(shapes)
    ]


def run_single_rank(
    tp_size, rank, shapes, dtype, args, keys, prod_regime, init_method, repeat
):
    """Run one rank in *this* process, re-joining the group ``repeat`` times.

    The profiling counterpart to ``run_sweep``. rocprofv3 instruments the
    process it launches and every descendant, so the ``Pool`` above puts all
    four ranks under the profiler -- and WaveScope then refuses the resulting
    PMC data outright, because a multi-process counter set has no stable
    cross-pass rank key and start-order ranking would silently pair one rank's
    counters with another's. One rank per process is what makes the profiler
    wrap rank 0 alone while the peers run outside it.

    ``repeat`` is for the peers, not the profiled rank. A multi-pass PMC
    capture re-runs the application once per counter recipe -- four times on
    gfx950 -- and the profiled rank is a fresh process each time. The peers are
    not, so they must re-join once per pass. ``_worker`` already destroys the
    process group and the model-parallel state in its own ``finally``, so each
    iteration here starts clean; between iterations rank 0's TCPStore is gone
    and the peers simply block in ``init_process_group`` until the next rank-0
    process binds the port.

    Returns the first join's per-shape scalars. There is only one rank here, so
    there is nothing to collapse with ``_row``.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    rows = None
    for i in range(repeat):
        logger.info(
            "TP%d %s rank %d: %d shape(s), %d iters, join %d/%d via %s",
            tp_size,
            dtype2str(dtype),
            rank,
            len(shapes),
            args.iters,
            i + 1,
            repeat,
            init_method,
        )
        got = _worker(
            tp_size,
            rank,
            shapes,
            dtype,
            args.iters,
            args.warmup,
            init_method,
            args.profile,
            keys,
            prod_regime,
            args.timing,
            args.graph_inner,
            args.fusion,
        )
        if rows is None:
            rows = got
    return rows


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


def case_tables(df, keys, baseline: str):
    """One latency/accuracy table per case (shape x TP x dtype), candidates as rows.

    The predecessor of this function, ``latency_table``, stacked every case
    into one wide table with a column per candidate, and accuracy sat in a
    second wide table of its own keyed the same way. With the full candidate
    set that grid is wider than a screen, so comparing implementations at one
    shape meant scanning across a giant row in one table, then finding the
    matching row in another. This stacks the other way: one small table per
    case -- its title carries the identity (TP, dtype, M, payload size,
    predicted kernel, prod path) that used to be repeated as leading columns
    on every row -- with one row per candidate that actually ran here, latency
    and SQNR side by side, so comparing implementations is reading down a
    short column instead.

    Ratio is ``baseline_us / candidate_us``, so **> 1.0 means the candidate is
    faster than the baseline**; the baseline's own row reads 1.0.
    """
    live = [k for k in keys if f"{k} us" in df.columns]
    base_col = f"{baseline} us"
    if base_col not in df.columns:
        logger.warning(
            "baseline %r produced no results (not applicable to any row in "
            "this sweep); skipping speedup columns",
            baseline,
        )

    tables = []
    for _, r in df.iterrows():
        bits = [
            f"TP{int(r['TP'])}",
            r["dtype"],
            f"M={int(r['M'])}",
            f"{r['payload size (KiB)']:.4g} KiB",
            f"kernel={r['kernel']}",
        ]
        if r["naive"] != r["kernel"]:
            bits.append(f"naive={r['naive']}")
        bits.append(f"prod={r['prod path']}")
        title = ", ".join(bits)

        base_us = r.get(base_col)
        base_us = base_us if pd.notna(base_us) else None

        rows = []
        absent = []
        for k in live:
            us = r.get(f"{k} us")
            if us is None or not pd.notna(us):
                absent.append(k)
                rows.append({"candidate": k, "us": float("nan")})
                continue
            row = {"candidate": k, "us": us}
            if base_us is not None:
                row[f"vs {baseline}"] = base_us / us
            row["SQNR dB"] = r.get(f"{k} SQNR dB", float("nan"))
            row["busbw GB/s"] = r.get(f"{k} busbw GB/s", float("nan"))
            spread = r.get(f"{k} spread us")
            if spread is not None and pd.notna(spread):
                row["spread us"] = spread
            # The kernel that actually ran, where the candidate can say. A
            # candidate key like `fly_int4_ring` names a *policy*, not a binary:
            # it walks a size ladder and picks a different super-tile per shape.
            # Without this column the report cannot distinguish "the auto row
            # chose well here" from "the auto row happened to agree with a
            # pinned row", which is the whole question a ladder fit asks.
            variant = r.get(f"{k} variant")
            if variant is not None and pd.notna(variant):
                row["variant"] = variant
            rows.append(row)
        if not rows:
            rows = [{"candidate": "-", "us": float("nan")}]
        cdf = pd.DataFrame(rows)
        _mark_na(cdf, ["spread us", "variant"])
        if absent:
            mask = cdf["candidate"].isin(absent)
            for col in cdf.columns:
                if col == "candidate":
                    continue
                cdf[col] = cdf[col].astype(object)
                cdf.loc[mask, col] = None
        tables.append((title, cdf))
    return tables


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
    # --fusion ar_rmsnorm shapes, from production_fused_path.
    if prod_path == "separate":
        return "separate_cdr"
    if prod_path.startswith("cdr_fused:"):
        return f"fused_cdr_{prod_path.split(':', 1)[1]}"
    if prod_path.startswith("qr_fused:"):
        return f"fused_qr_{prod_path.split(':', 1)[1]}"
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
      candidates, i.e. the fastest option that does not change the model's
      numerics at all. Membership is decided **per shape**, not per candidate:
      ``fly_auto`` dispatches to the exact one-shot below its policy ceiling
      and to a quantized schedule above it, so it belongs in this column on
      some rows and not others. Omitted when every candidate in the sweep is
      exact at every shape, since it would just repeat ``fastest collective``.

    A candidate excluded by the floor is not hidden: it keeps its row in that
    shape's ``latency & accuracy by case`` table, and the count of rows where
    the floor changed the winner is logged, so the default can never silently
    bury a result.

    Both ratios are ``prod time (us) / fastest time (us)``, so **> 1.0 means
    faster than production**, matching ``case_tables``'s ``vs <baseline>``
    convention. A row whose winner *is* the production path reads 1.0, which
    is the useful answer that nothing beat it.

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
    live = [k for k in keys if f"{k} us" in df.columns]

    def _exact_here(row, k) -> bool:
        """Whether *k* was bit-accurate on *row*'s shape.

        Read per row from the ``<k> exact`` column rather than from
        ``Candidate.exact``, because ``fly_auto`` changes accuracy class with
        the payload.
        """
        v = row.get(f"{k} exact")
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return next((c.exact for c in CANDIDATES if c.key == k), False)
        return bool(v)

    # Only worth a separate column when the sweep actually mixes accuracy
    # classes; with -c cdr rccl every candidate is exact and it would duplicate.
    want_exact = any(
        not _exact_here(r, k) for _, r in df.iterrows() for k in live if pd.notna(r.get(f"{k} us"))
    )

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
            k, us = _pick(r, [x for x in live if _exact_here(r, x)], min_sqnr)
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
            "%g dB floor; see the latency & accuracy by case tables for its timings",
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


def _fused_1stage_note(world_sizes) -> str:
    """One line describing what decides `cdr_fused:1stage` vs `:2stage`.

    Worth recording in a saved report: the boundary moves with world size, so
    two reports of the same shape can legitimately name different kernels, and
    the env knobs can move it further.
    """
    override = os.environ.get(_AR_1STAGE_ENV, "")
    if override in ("0", "1"):
        return f"{_AR_1STAGE_ENV}={override} (forced {'1stage' if override == '1' else '2stage'})"
    max_kb = int(os.environ.get(_AR_1STAGE_MAX_KB_ENV, "-1"))
    if max_kb >= 0:
        return f"{_AR_1STAGE_MAX_KB_ENV}={max_kb} KiB (all world sizes)"
    parts = [
        f"tp{ws} {128 * 7168 * 2 // ws / 1024:.0f} KiB"
        for ws in sorted(set(world_sizes))
    ]
    return "default 128*7168*2/world_size -- " + ", ".join(parts)


def _fly_floor_note(world_sizes) -> str:
    """``QuickAllReduceInt4.allreduce``'s own size floor per (schedule, world size)."""
    if not HAS_FLY_INT4:
        return "n/a"
    parts = []
    for algorithm in sorted(ALGORITHMS):
        for ws in sorted(set(world_sizes)):
            if ws not in _FLY_WORLDS:
                continue
            algo = ALGORITHMS[algorithm]
            rs_codec, ag_codec = _resolve_codecs(algo, ws, None, None)
            # World-keyed since the ring floor was measured; the codec is still
            # printed because it is what the wire actually carries at this TP.
            floor = algo.floor_bytes(ws)
            parts.append(
                f"{algorithm}/tp{ws} {floor >> 10} KiB (rs={rs_codec} ag={ag_codec})"
            )
    return "; ".join(parts) or "n/a"


def _write_raw_csv(path, df, dtype_name: str, per_dtype: bool, args=None) -> None:
    """Dump the un-collapsed dataframe for one dtype.

    The markdown tables answer "which candidate won"; this answers "what were
    all the numbers", which is what fitting a dispatch threshold needs -- the
    losers matter as much as the winner, because the threshold sits where two
    curves cross and both have to be in hand to find it. ``_nbytes`` rides along
    (it is deliberately absent from ``ID_COLUMNS``, so no printed table carries
    the exact byte count) because that is the axis a threshold is expressed on.

    One file per dtype: the frame is rebuilt per dtype, and a single path would
    have the last one silently overwrite the rest.
    """
    out = Path(path)
    if per_dtype:
        out = out.with_name(f"{out.stem}_{dtype_name}{out.suffix or '.csv'}")
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    if args is not None:
        df = df.copy()
        df["timing"] = args.timing
        df["graph inner"] = args.graph_inner if args.timing == "graph" else 0
        df["fly accuracy"] = args.fly_accuracy
    df.to_csv(out, index=False)
    logger.info("wrote %s (%d row(s) x %d column(s))", out, len(df), len(df.columns))


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
        f"- aiter package: {_aiter_origin()}",
        # What the `us` column means. There is exactly one time per candidate,
        # so a report is unreadable without this line.
        (
            f"- timing: **{args.timing}** -- `us` is "
            + (
                f"HIP-graph replay ({args.graph_inner} collectives per capture)"
                if args.timing == "graph"
                else "eager hipEvent wall time, host path included"
            )
        ),
        f"- fusion: {args.fusion}",
        f"- FlyDSL accuracy regime: {args.fly_accuracy}",
        f"- baseline: {args.baseline}",
        f"- fly_int4 available: {HAS_FLY_INT4}",
        (
            "- QuickAllReduceInt4 deployment floors (not enforced here): "
            f"{_fly_floor_note(args.tp if args.tp else [4])}"
        ),
        (
            "- bf16 cast to fp16 on the QR wire: "
            f"{os.environ.get('AITER_QUICK_REDUCE_CAST_BF16_TO_FP16', '1')}"
        ),
        f"- `prod path` evaluated with {_QR_ENV}={prod_regime!r}",
        (
            "- `prod path` fused 1stage/2stage split: "
            + _fused_1stage_note(args.tp if args.tp else [4])
        ),
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
        "Candidates are not accuracy-equivalent -- each `latency & accuracy by",
        "case` table below prints `us` next to `SQNR dB` for exactly that reason.",
        "The summary table's `fastest collective` column is the fastest candidate",
        "clearing the accuracy floor above, with `fastest collective SQNR dB`",
        "beside it showing what that choice costs; `fastest exact collective` is",
        "the fastest option that leaves the model's numerics untouched.",
        "Candidates below the floor are excluded from `fastest collective` only --",
        "their timings are still in the per-case tables.",
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
        "--shape-csv",
        metavar="PATH",
        default=None,
        help="read (tokens, hidden) pairs from a CSV with M,K columns instead\n"
        "of -s/--shape. For the dispatch sweeps, whose ~50 sizes per world size\n"
        "do not fit on a command line; see op_tests/multigpu_tests/shapes/.",
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
    parser.add_argument(
        "--fly-accuracy",
        choices=_FLY_ACCURACY_CHOICES,
        default=_FLY_ACCURACY_DEFAULT,
        help="AITER_FLY_AR_ACCURACY for every FlyDSL candidate in the sweep\n"
        "(ignored if none are). 'fast' (default) opens the mesh/ring window\n"
        "past the one-shot ceiling, so `fly_auto` exercises the full\n"
        "three-family policy at every shape, and pinned fly_int4*/\n"
        "fly_int4_ring* (quantized) rows run alongside the one-shot rows.\n"
        "'exact' matches the shipped production default: only the one-shot\n"
        "is ever reachable, `fly_auto` reads n/a above oneshot_max_exact\n"
        "rather than quantizing, and the quantized fly_int4*/fly_int4_ring*\n"
        "rows are excluded entirely (n/a) rather than advertising a lossy\n"
        "kernel an exact-mode deployment would never dispatch to.",
    )
    parser.add_argument("--iters", type=int, default=101, help="timed iterations")
    parser.add_argument("--warmup", type=int, default=5, help="warmup iterations")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="run only this rank in this process instead of spawning a pool of\n"
        "them, and skip the summary tables (they need every rank's scalars).\n"
        "Requires --init-method, and every rank of the group must be launched\n"
        "separately with the same one. This exists so a profiler can wrap one\n"
        "rank: rocprofv3 instruments the process it launches and all of its\n"
        "descendants, so the default pool launch profiles all four ranks at\n"
        "once and WaveScope refuses that PMC data as multi-process. See\n"
        "--repeat for the peer side.",
    )
    parser.add_argument(
        "--init-method",
        metavar="URL",
        default=None,
        help="rendezvous for --rank, e.g. tcp://127.0.0.1:29500. Taken\n"
        "verbatim, unlike the pool path which picks its own free port -- the\n"
        "point is that separately launched ranks agree on it.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="with --rank, join the group and run this many times in sequence.\n"
        "For the peers of a multi-pass PMC capture: the capture re-runs the\n"
        "application once per counter recipe (four times on gfx950) with a\n"
        "fresh profiled rank each pass, so the peers have to re-join that many\n"
        "times. Leave it at 1 for the profiled rank itself.",
    )
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
        "their rows in the latency & accuracy by case tables either way.",
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
        "--fusion",
        choices=FUSION_MODES,
        default="none",
        help="switch the candidate set to a fused epilogue.\n"
        "'ar_rmsnorm' measures all-reduce + residual add + RMSNorm -- what\n"
        "aiter::allreduce_fusion_kernel_1stage computes, and the most expensive\n"
        "kernel in the Qwen3-235B MXFP4 decode trace. The fused_* rows are the\n"
        "kernels that genuinely fuse; the separate_* rows are the two-launch\n"
        "baselines (all-reduce, then rmsnorm2d_fwd_with_add) they must beat.\n"
        "Default 'none' leaves every existing invocation untouched.",
    )
    parser.add_argument(
        "--timing",
        choices=_TIMING_CHOICES,
        default=_TIMING_DEFAULT,
        help="how every candidate is timed. Exactly one time is measured and\n"
        "the `us` column holds it; the report header and the CSV record which.\n"
        "'graph' (default) captures a HIP graph and times the replay. That is\n"
        "the metric production sees -- decode is captured -- and it is the only\n"
        "fair kernel-to-kernel comparison, because eager timing here is\n"
        "host-bound: run_perftest brackets back-to-back Python calls, so once\n"
        "host cost per call exceeds device time the GPU starves and the number\n"
        "*is* the host cost. That cost also differs per candidate family (a\n"
        "separate_* row makes two Python op calls, cdr and qr go through\n"
        "pybind, the FlyDSL rows through _run_compiled, rccl through an aten op\n"
        "plus a copy_), so eager partly ranks candidates by how much Python\n"
        "sits in their bench thunk -- a property of this harness, not of the\n"
        "kernel. Since every family boundary in allreduce_policy is a\n"
        "*crossover between families*, that bias lands directly on the shipped\n"
        "thresholds.\n"
        "'eager' is kept for when the host path is what you want to see.",
    )
    parser.add_argument(
        "--graph-inner",
        type=int,
        default=_GRAPH_INNER_DEFAULT,
        help="collectives captured per HIP graph (--timing graph). Replay is\n"
        "back-to-back with no host in between, which is the run-ahead case the\n"
        "double-buffered inbox is designed for. Pass 1 to price a capture that\n"
        "cannot run ahead.",
    )
    parser.add_argument(
        "-b",
        "--baseline",
        choices=CANDIDATE_KEYS,
        default=None,
        help="candidate to measure the others against. Adds a\n"
        "'<cand> vs <baseline>' column per candidate, where > 1.0 means the\n"
        f"candidate is faster than the baseline. Default: {PRIMARY}, or\n"
        f"{FUSION_PRIMARY} under --fusion -- either way the kernel production\n"
        "runs today, so the table answers 'is anything beating what we ship?'.\n"
        "Pass 'rccl' for the 'are we beating the library?' framing instead.",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="PATH",
        default=None,
        help="also write the summary tables to PATH as markdown, with a\n"
        "provenance header (arch, iters, command). Overwrites PATH.",
    )
    parser.add_argument(
        "--output-csv",
        metavar="PATH",
        default=None,
        help="also write the raw per-shape dataframe to PATH as CSV: every\n"
        "candidate's us, median us, SQNR, busbw, spread and kernel variant,\n"
        "plus the exact byte count. This is the machine-readable form the\n"
        "dispatch-threshold fitting reads; the markdown is for humans. One file\n"
        "per dtype -- PATH gets the dtype inserted before its suffix when the\n"
        "sweep covers more than one.",
    )
    args = parser.parse_args()
    # Mode-dependent default, resolved here so --help can describe both modes.
    if args.baseline is None:
        args.baseline = FUSION_PRIMARY if args.fusion != "none" else PRIMARY
    if args.graph_inner < 1:
        parser.error("--graph-inner must be positive")
    fused_keys = {c.key for c in CANDIDATES if c.fusion}
    if args.candidates:
        wrong = [
            k for k in args.candidates if (k in fused_keys) != (args.fusion != "none")
        ]
        if wrong:
            parser.error(
                f"--fusion {args.fusion} cannot run {', '.join(wrong)}: fused and "
                "plain candidates compute different things and are graded against "
                "different references, so they never share a sweep."
            )
    if (args.baseline in fused_keys) != (args.fusion != "none"):
        parser.error(
            f"--baseline {args.baseline} does not belong to --fusion {args.fusion}"
        )
    if args.shape_csv is not None:
        if args.shape is not L_SHAPE:
            parser.error("--shape-csv and -s/--shape are mutually exclusive")
        args.shape = load_shapes_csv(args.shape_csv)
        sizes = [m * k * 2 for m, k in args.shape]
        logger.info(
            "loaded %d shape(s) from %s, %.4g KiB to %.4g KiB",
            len(args.shape),
            args.shape_csv,
            min(sizes) / 1024,
            max(sizes) / 1024,
        )

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

    # Default to the half of the candidate list that matches the mode.
    keys = args.candidates or sorted(
        k for k in CANDIDATE_KEYS if (k in fused_keys) == (args.fusion != "none")
    )

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
    if any(c.family == "flyauto" for c in CANDIDATES if c.key in keys):
        # FlyDSLAllReduce is opt-in; set it before the ranks are spawned so the
        # children inherit it. Unlike _QR_ENV this does not change `prod path`,
        # which reports the custom-AR/quick-reduce dispatch only.
        os.environ[_FLY_ENV] = "1"
    # --fly-accuracy, not whatever accuracy mode the launching shell happens to
    # have exported -- a report's accuracy regime should be exactly what its own
    # command line says. Set unconditionally, not just when a FlyDSL family is
    # in the sweep: applicable() reads this to gate the quantized qr rows too,
    # and a `-c cdr qr_int4 rccl --fly-accuracy exact` sweep would otherwise
    # keep them, having never exported the variable the gate reads.
    os.environ[_FLY_ACCURACY_ENV] = args.fly_accuracy
    prod_regime = os.environ.get(_QR_ENV)
    if any(
        c.family in ("qr", "fused_qr") or (c.family == "separate" and c.sep_ar == "qr")
        for c in CANDIDATES
        if c.key in keys
    ):
        os.environ[_QR_ENV] = _QR_ENABLING_REGIME
        # Build the quick-reduce JIT module here rather than letting every
        # spawned rank race for the same first build.
        import aiter as ops

        ops.qr_max_size()

    # Single-rank mode branches here, after the environment preamble above, so
    # a profiled rank sees exactly what a pooled one would.
    if args.rank is not None:
        if args.init_method is None:
            parser.error("--rank requires --init-method; every rank must agree on it")
        if args.repeat < 1:
            parser.error("--repeat must be at least 1")
        if len(tps) != 1 or len(args.dtype) != 1:
            parser.error(
                "--rank runs one rank of one TP size at one dtype; pass a single "
                "-tp and a single --dtype"
            )
        tp_size = tps[0]
        if not 0 <= args.rank < tp_size:
            parser.error(f"--rank {args.rank} is out of range for TP{tp_size}")
        rows = run_single_rank(
            tp_size,
            args.rank,
            args.shape,
            dtypes.d_dtypes[args.dtype[0]],
            args,
            keys,
            prod_regime,
            args.init_method,
            args.repeat,
        )
        for (tokens, hidden), ret in zip(args.shape, rows):
            logger.info(
                "rank %d %dx%d: %s",
                args.rank,
                tokens,
                hidden,
                {k: v for k, v in ret.items() if k.endswith("_us")},
            )
        return

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
        ]
        for title, table in tables:
            md = table.to_markdown(index=False, floatfmt=".4g", missingval="n/a")
            logger.info("all-reduce %s (markdown):\n%s", title, md)
            sections.append((title, md))

        case_title = f"{dtype_name} latency & accuracy by case"
        case_md = "\n\n".join(
            f"### {title}\n\n"
            + cdf.to_markdown(index=False, floatfmt=".4g", missingval="n/a")
            for title, cdf in case_tables(df, keys, args.baseline)
        )
        logger.info("all-reduce %s (markdown):\n%s", case_title, case_md)
        sections.append((case_title, case_md))

        tables = []
        # Which binary each candidate ran, per shape. Only the flydsl families
        # can report it, so the table is skipped entirely when none are in the
        # sweep rather than printed as a wall of `n/a`.
        if any(f"{k} variant" in df.columns for k in keys):
            tables.append(
                (f"{dtype_name} kernel variants", metric_table(df, "variant", keys))
            )
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

        if args.output_csv:
            _write_raw_csv(args.output_csv, df, dtype_name, len(args.dtype) > 1, args)

    if args.output:
        _write_report(args.output, sections, args, visible, prod_regime, roofline_cus)


if __name__ == "__main__":
    freeze_support()
    main()
