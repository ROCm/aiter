# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""TransferBench-backed bandwidth roofline for ``bench_comm_allreduce.py``.

``bench_comm_allreduce.py`` compares aiter's all-reduce candidates against each
other and against RCCL. Both of those are *floors*: they tell you whether a
kernel beats another kernel, not how much of the fabric any of them is
actually using. This module supplies the ceiling, by asking TransferBench
(https://github.com/ROCm/TransferBench) to move the same bytes in the same
pattern with none of the collective's semantics attached.

The one TransferBench feature that makes this possible: a Transfer is defined
as "an Executor reads and **adds** values from source memory, then writes the
sum to destination memory", and source/destination locations concatenate. So
``G0G1G2G3->G0->G0`` is a GPU-0 kernel that reads all four peer buffers, sums
them, and writes GPU 0's copy -- exactly the data movement of a one-shot
all-reduce, minus the peer handshake.

What the roofline is and is not
===============================

The roof for a byte count is the **best over algorithms**, not the throughput of
one. That distinction is load-bearing: an earlier version modelled only the
direct two-shot below, and on a NUMA-split PCIe host that pattern is ~19% slower
than a ring -- so RCCL, which rings, measured *above* the "ceiling" at
``eff = 1.18``. A ceiling a real candidate can beat is not a ceiling. Every
algorithm here is a legitimate way to all-reduce the same bytes, so the fastest
of them is the honest bound::

    one-shot  N parallel transfers, GPU i reduces all N buffers into its own:
              -N (G0..G(N-1) Gi Gi <cus> <bytes>)  for each i

    two-shot  reduce-scatter then all-gather, as two tests whose times are
              summed (TransferBench has no ordering between transfers within a
              test, so they cannot share one):
              RS: -N (G0..G(N-1) Gi Gi <cus> <bytes/N>)         for each i
              AG: -N (Gi Gi G0..G(N-1) except Gi <cus> <bytes/N>) for each i

    ring      one step, each rank sending its chunk to its successor, scaled by
              the 2(N-1) steps a ring all-reduce takes:
              -N (Gi Gi G(i+1 mod N) <cus> <bytes/N>)  for each i

Every step of a ring all-reduce drives the identical link pattern -- only the
chunk being carried differs -- so one step is measured and multiplied rather
than emitting 2(N-1) tests and paying 2(N-1) launch overheads.

TransferBench cannot express the chunk *offsets* that a real reduce-scatter
reads, but the bytes moved per link are the same, which is all a bandwidth
roofline depends on.

Only the ring survives above ``MAX_FANIN``: it needs one source and one
destination per transfer, where the other two need N. Above that world size the
roof is the ring alone rather than absent.

Deliberately not modelled, all of which make the roofline optimistic:

* **No peer handshake.** Ranks barrier on the host before the timed loop; there
  is no in-kernel ``start_sync``/``end_sync``. The 1-stage kernel spends most
  of its duration spinning there, so at small sizes the measured kernel will
  sit far below this ceiling and that is not a fabric problem.
* **No quantize/dequantize ALU cost.** The quantizing candidates are rooflined
  by shrinking the byte count (see ``WIRE_RATIO``), which prices their wire
  saving but not the codec.
* **Scale bytes are ignored.** The quick-reduce codecs also ship per-group
  scales, so their true wire is a little larger than ``WIRE_RATIO`` says and
  their roofline is correspondingly a little too fast.
* **fp32 elements.** TransferBench is hardcoded to ``float`` throughout
  (``numBytes / sizeof(float)``). Only the byte count is comparable, which for
  a bandwidth ceiling is the part that matters.

Because of the first point especially, treat a low efficiency at small sizes as
"sync-bound, as expected" and a low efficiency at large sizes as a real
question about the kernel.

One parsing constraint worth knowing, since it decides how the numbers are
read back: TransferBench prints durations as ``%8.3f ms``, a 1 us quantum, and
a small-shape all-reduce roofline is *sub*-microsecond -- it reads back as a
flat ``0.000``. Durations are therefore re-derived from the bandwidth column
where that is the more precise of the two; see ``_duration_ms``. Rows where
both columns bottom out are dropped rather than reported as zero.

Usage
=====

Requires the ``TransferBench`` binary, which is not part of a default ROCm
install -- it ships with ROCmValidationSuite, or build it from source
(https://github.com/ROCm/TransferBench). Point ``$TRANSFERBENCH`` at it or pass
``--roofline-bin``. When it cannot be found the caller gets an empty result and
a warning rather than an exception.

Standalone, to see what would be run and to check the parser::

    python3 op_tests/multigpu_tests/transferbench_roofline.py --dry-run -t 4
    python3 op_tests/multigpu_tests/transferbench_roofline.py --self-test
    python3 op_tests/multigpu_tests/transferbench_roofline.py -t 4 -b 114688
"""

import logging
import math
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger("aiter")

# Wire bytes per payload byte, per bench_comm_allreduce.py candidate key. This
# is what the candidate actually puts on the fabric relative to its (bf16/fp16)
# input, and it is the only thing that distinguishes one candidate's roofline
# from another's -- the pattern and the CU count are shared.
#
# The int/fp8 ratios are codec width / 16, and ignore the per-group scales the
# quick-reduce codecs also send, so they are slightly optimistic. qr_fp stays at
# 1.0 on bf16 input because AITER_QUICK_REDUCE_CAST_BF16_TO_FP16 casts to fp16,
# which is the same 2 bytes per element.
WIRE_RATIO = {
    "cdr": 1.0,
    "cdr_naive": 1.0,
    "cdr_fp8": 0.5,
    "qr_fp": 1.0,
    "qr_fp8": 0.5,
    "qr_int6": 6.0 / 16.0,
    "qr_int4": 4.0 / 16.0,
    "qr_int3": 3.0 / 16.0,
    "fly_int4": 4.0 / 16.0,
    "fly_int4_ring": 4.0 / 16.0,
    "rccl": 1.0,
}

# Which traffic pattern each candidate actually runs, independent of the shape.
# Only the cross_device_reduce candidates switch pattern with size, and for
# those the caller passes the host dispatch's own prediction instead.
#
#   qr_*      always two-shot -- every regime lands in
#             allreduce_prototype_twoshot (csrc/include/quick_all_reduce.cuh)
#   fly_int4  two-shot by construction (ROCm/aiter#4970)
#   fly_int4_ring
#             the same kernel family on a 2(N-1)-hop ring; identical wire
#             volume to two-shot, so it shares the ratio and differs only in
#             which pattern it drives
#   rccl      a ring all-reduce moves 2(N-1)/N x nbytes per rank, which is the
#             same per-link traffic as reduce-scatter plus all-gather
_FIXED_PATTERN = {
    "qr_fp": "2stage",
    "qr_fp8": "2stage",
    "qr_int6": "2stage",
    "qr_int4": "2stage",
    "qr_int3": "2stage",
    "fly_int4": "2stage",
    "fly_int4_ring": "ring",
    "rccl": "2stage",
}


def pattern(cand_key: str, predicted: str) -> str:
    """Which algorithm *cand_key* itself runs, for reporting.

    **This no longer selects the roof.** It used to, and that was the bug: a
    candidate graded against its own algorithm can beat the "ceiling" simply by
    picking a better one, which is how ``rccl`` reached ``eff = 1.18``. The roof
    is now the best over every algorithm in ``_ALGOS``. What survives here is
    descriptive: comparing a candidate's own pattern against the ``roof algo``
    that won says whether it chose well for this fabric.

    *predicted* is the ``cross_device_reduce_*`` the host dispatch would pick
    for this shape, and is used only for the candidates that actually follow it.
    """
    fixed = _FIXED_PATTERN.get(base_key(cand_key))
    if fixed is not None:
        return fixed
    return "1stage" if predicted.startswith("1stage") else "2stage"


# TransferBench caps a Transfer at 8 sources and 8 destinations (MAX_SRCS /
# MAX_DSTS in src/header/TransferBench.hpp). A one-shot all-reduce needs one
# source per rank and an all-gather one destination per rank, so TP8 sits
# exactly at the limit and TP16 cannot be expressed at all.
MAX_FANIN = 8

# CU counts to try per measurement; the roofline is the best of them. A single
# count would report whatever that count happens to achieve rather than what the
# fabric can do, and the best count moves with both size and world size.
DEFAULT_CUS = (8, 16, 32)

_ENV_BIN = "TRANSFERBENCH"


def find_binary(explicit: str | None = None) -> str | None:
    """Locate the TransferBench executable, or return None.

    Checked in order: the explicit argument, ``$TRANSFERBENCH``, ``$PATH``, and
    the two paths ROCmValidationSuite installs it under.
    """
    for cand in (explicit, os.environ.get(_ENV_BIN)):
        if cand:
            p = Path(cand).expanduser()
            if p.is_file() and os.access(p, os.X_OK):
                return str(p)
            logger.warning("TransferBench: %r is not an executable file", cand)
            return None
    found = shutil.which("TransferBench")
    if found:
        return found
    for p in ("/opt/rocm/bin/TransferBench", "/opt/rocm/libexec/rvs/TransferBench"):
        if os.access(p, os.X_OK):
            return p
    return None


def round16(nbytes) -> int:
    """Largest multiple of 16 not exceeding *nbytes*, floored at 16.

    TransferBench requires a multiple of 4; 16 additionally matches the
    alignment every custom all-reduce path in aiter demands, so a rooflined
    byte count is one the real kernels could also have run.
    """
    return max(16, (int(nbytes) // 16) * 16)


_ST_SUFFIX = re.compile(r"_st\d+$")


def base_key(cand_key: str) -> str:
    """Strip a ``_st<N>`` tuning suffix; variants share their base's wire shape.

    ``fly_int4_ring_st16`` puts exactly the bytes on the wire that
    ``fly_int4_ring`` does and drives the same pattern -- the super-tile changes
    how many tiles a block batches behind one publish, never the wire format.

    Resolving by prefix matters because both lookups below fall back to a
    *silent* default (ratio 1.0, pattern two-shot). A variant added in
    ``bench_comm_allreduce.py`` and forgotten here would then be graded against
    a 4x-too-large roof and quietly report a quarter of its real efficiency.
    """
    return _ST_SUFFIX.sub("", cand_key)


def wire_bytes(payload_bytes: int, cand_key: str) -> int:
    """Bytes *cand_key* puts on the wire for a *payload_bytes* all-reduce."""
    return round16(payload_bytes * WIRE_RATIO.get(base_key(cand_key), 1.0))


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def _advanced(transfers) -> str:
    """One config line in advanced mode: ``-N (src exe dst SEs Bytes) ...``.

    Advanced mode (negative transfer count, explicit per-transfer bytes) rather
    than simple mode, because it lets every size in a sweep live in one config
    file -- simple mode takes its size from the command line, which would mean
    one process launch per size.
    """
    body = " ".join(f"({s} {e} {d} {cu} {nb})" for s, e, d, cu, nb in transfers)
    return f"-{len(transfers)} {body}"


def _one_shot(tp: int, nbytes: int, cus: int) -> list[str]:
    """Every GPU reduces every rank's buffer into its own. One test."""
    peers = "".join(f"G{i}" for i in range(tp))
    return [_advanced([(peers, f"G{i}", f"G{i}", cus, nbytes) for i in range(tp)])]


def _two_shot(tp: int, nbytes: int, cus: int) -> list[str]:
    """Reduce-scatter then all-gather, as two tests to be summed.

    The two phases are separate config lines because transfers within one line
    run in parallel with no dependency between them, and an all-gather that
    starts before its reduce-scatter finishes is not a two-shot all-reduce.
    Summing two tests slightly over-counts: it pays a launch and a barrier
    twice where the real kernel pays them once.
    """
    chunk = round16(nbytes // tp)
    peers = "".join(f"G{i}" for i in range(tp))
    # RS keeps every source including the local one: a real reduce-scatter does
    # read its own contribution. AG deliberately excludes the local destination
    # -- a rank already holds its own chunk, and writing it to itself is local
    # traffic no all-gather performs. Leaving it in inflated the two-shot roof
    # enough to push measured `eff` above 1.0 at TP2.
    others = ["".join(f"G{j}" for j in range(tp) if j != i) for i in range(tp)]
    rs = _advanced([(peers, f"G{i}", f"G{i}", cus, chunk) for i in range(tp)])
    ag = _advanced([(f"G{i}", f"G{i}", others[i], cus, chunk) for i in range(tp)])
    return [rs, ag]


def _ring(tp: int, nbytes: int, cus: int) -> list[str]:
    """One step of a ring all-reduce; the caller scales by ``_ring_steps``.

    Each rank sends its chunk to its successor and receives from its
    predecessor -- point-to-point only, never a fan-in. That is exactly why a
    ring survives an unfavourable NUMA split that the direct patterns do not:
    it needs N peer links rather than all N(N-1) of them, and the ring order
    can be chosen to cross the socket boundary as few times as possible.
    """
    chunk = round16(nbytes // tp)
    return [
        _advanced(
            [(f"G{i}", f"G{i}", f"G{(i + 1) % tp}", cus, chunk) for i in range(tp)]
        )
    ]


def _ring_steps(tp: int) -> int:
    """A ring all-reduce is 2(N-1) steps: N-1 to reduce-scatter, N-1 to gather."""
    return 2 * (tp - 1)


# name -> (emit, steps(tp), fan-in(tp)). ``fan-in`` is the largest source or
# destination list the pattern builds, checked against MAX_FANIN.
_ALGOS = (
    ("one-shot", _one_shot, lambda tp: 1, lambda tp: tp),
    ("two-shot", _two_shot, lambda tp: 1, lambda tp: tp),
    ("ring", _ring, _ring_steps, lambda tp: 1),
)


@dataclass(frozen=True)
class _Plan:
    """One (byte count, algorithm, CU count) measurement.

    ``steps`` scales the measured time: 1 for the patterns emitted in full, and
    2(N-1) for the ring, of which only one representative step is run.
    """

    nbytes: int
    algo: str
    cus: int
    steps: int
    lines: list[str] = field(compare=False)


@dataclass(frozen=True)
class Roof:
    """The best time any modelled algorithm achieved, and which one it was."""

    us: float
    algo: str


def build_plans(tp: int, byte_counts, cus=DEFAULT_CUS, algos=None) -> list[_Plan]:
    """A plan per (byte count, algorithm, CU count), in test order.

    Keyed on bytes alone: the roof is the best algorithm for that many bytes,
    so a candidate's own choice of algorithm no longer selects its ceiling.
    *algos* restricts which are considered; the default is all of them.
    """
    plans = []
    for nbytes in sorted(set(byte_counts)):
        for name, emit, steps, fanin in _ALGOS:
            if algos is not None and name not in algos:
                continue
            if fanin(tp) > MAX_FANIN:
                continue
            for cu in cus:
                plans.append(_Plan(nbytes, name, cu, steps(tp), emit(tp, nbytes, cu)))
    return plans


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

# TransferBench prints a bordered 5-column table per test (Utilities.hpp
# PrintResults). OUTPUT_TO_CSV=1 is not used even though it looks easier to
# parse: it emits that same 5-column table comma-separated under an 11-column
# header (Test#,Transfer#,NumBytes,...) that no longer matches it, and it drops
# the "Test N:" line that delimits one test from the next.
#
# Anchoring on the units rather than on the column separator keeps these working
# whether or not SHOW_BORDERS drew the '|' glyphs.
_SEP = r"\s*[│|]?\s*"
_TEST_RE = re.compile(r"^Test\s+(\d+):")
_EXEC_RE = re.compile(
    r"Executor:\s+(?:Rank\s+\d+\s+)?(\S+)\s+(\d+)"
    rf"{_SEP}([\d.]+)\s*GB/s{_SEP}([\d.]+)\s*ms{_SEP}(\d+)\s*bytes"
)
_AGG_RE = re.compile(
    r"Aggregate\s+\(CPU\)"
    rf"{_SEP}([\d.]+)\s*GB/s{_SEP}([\d.]+)\s*ms{_SEP}(\d+)\s*bytes"
    # Overhead is Test time minus the slowest Executor's, and goes slightly
    # negative when an Executor's own timer outruns the CPU bracket around it.
    rf"{_SEP}Overhead\s+(-?[\d.]+)\s*ms"
)
_ERR_RE = re.compile(r"^\s*\[ERROR\]\s*(.*)")


def _duration_ms(gbps: float, ms: float, nbytes: int) -> float:
    """Best available duration for one row, in ms.

    The printed ms column alone is not good enough. TransferBench formats it as
    ``%8.3f ms``, a 1 us quantum: a small-shape all-reduce roofline is
    sub-microsecond and reads back as a flat ``0.000``, and a 10 us one is
    quantized by 10%. Bandwidth is printed at the same three decimals but on a
    value in the 1-1000 range, so for GPU transfers it carries several more
    significant digits, and it inverts exactly:

        avgBandwidthGbPerSec = (numBytes / 1e6) / avgDurationMsec
        -- TransferBench.hpp:6191, and identically for transfers at :6205

    Neither channel is always better: a slow CPU transfer prints 0.027 GB/s
    against 77.492 ms, where the time column is the precise one. Since both are
    quantized to the same 5e-4 absolute, the one with the larger magnitude has
    the better relative precision, so pick that one.

    Returns nan when both read as zero, which means the row was below the
    resolution of the text interface entirely.
    """
    if gbps > 0.0 and nbytes > 0 and gbps >= ms:
        return (nbytes / 1e6) / gbps
    if ms > 0.0:
        return ms
    return float("nan")


@dataclass
class TestResult:
    """One TransferBench test: per-executor times plus the aggregate row."""

    num: int
    exec_ms: dict = field(default_factory=dict)  # "GPU 00" -> ms
    total_ms: float = float("nan")  # CPU wall clock over all executors
    overhead_ms: float = float("nan")

    @property
    def slowest_exec_ms(self) -> float:
        """The executor the collective would actually wait on.

        Matches bench_comm_allreduce.py's ``max(per_rank)``, and unlike
        ``total_ms`` it excludes the host-side barrier and launch overhead that
        the benchmark's own hipEvent bracket does not contain either.

        One unresolvable executor makes the whole test unresolvable: the
        collective waits on the slowest, and a row that did not resolve could
        have been it. ``max`` is not asked to rank a nan, whose comparisons are
        all false and would silently return whichever value came first.
        """
        vals = list(self.exec_ms.values())
        if not vals or any(math.isnan(v) for v in vals):
            return float("nan")
        return max(vals)


def parse_tests(text: str) -> list[TestResult]:
    """Every test in a TransferBench run, in output order."""
    tests: list[TestResult] = []
    for line in text.splitlines():
        m = _TEST_RE.match(line)
        if m:
            tests.append(TestResult(num=int(m.group(1))))
            continue
        if not tests:
            continue
        m = _EXEC_RE.search(line)
        if m:
            tests[-1].exec_ms[f"{m.group(1)} {m.group(2)}"] = _duration_ms(
                float(m.group(3)), float(m.group(4)), int(m.group(5))
            )
            continue
        m = _AGG_RE.search(line)
        if m:
            tests[-1].total_ms = _duration_ms(
                float(m.group(1)), float(m.group(2)), int(m.group(3))
            )
            tests[-1].overhead_ms = float(m.group(4))
    return tests


# ---------------------------------------------------------------------------
# Driving the binary
# ---------------------------------------------------------------------------


def _run(binary: str, config: str, *, iters: int, warmup: int, timeout: float) -> str:
    """Run one config file and return stdout, or raise RuntimeError."""
    env = dict(os.environ)
    env.update(
        {
            "NUM_ITERATIONS": str(iters),
            "NUM_WARMUPS": str(warmup),
            "USE_HIP_EVENTS": "1",
            # Validation walks every element on the host and costs far more than
            # the transfers at prefill sizes. The roofline is a timing number;
            # correctness of the sum is not what is under test here.
            "ALWAYS_VALIDATE": "-1",
            "OUTPUT_TO_CSV": "0",
            "HIDE_ENV": "1",
        }
    )
    with tempfile.NamedTemporaryFile(
        "w", suffix=".cfg", prefix="ar_roofline_", delete=False
    ) as fh:
        fh.write(config)
        cfg_path = fh.name
    try:
        # The size argument is unused -- every transfer carries its own byte
        # count in advanced mode -- but the client requires a multiple of 4.
        proc = subprocess.run(
            [binary, cfg_path, "16"],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            # Checked below instead, to surface TransferBench's own [ERROR]
            # lines rather than a bare CalledProcessError.
            check=False,
        )
    finally:
        os.unlink(cfg_path)
    if proc.returncode != 0:
        errs = _ERR_RE.findall(proc.stdout) + _ERR_RE.findall(proc.stderr)
        detail = "; ".join(errs) or (proc.stderr or proc.stdout)[-400:].strip()
        raise RuntimeError(f"TransferBench exited {proc.returncode}: {detail}")
    return proc.stdout


def measure(
    tp_size: int,
    byte_counts,
    *,
    binary: str,
    cus=DEFAULT_CUS,
    algos=None,
    iters: int = 20,
    warmup: int = 3,
    timeout: float = 900.0,
) -> dict:
    """``{bytes: Roof}`` -- the fastest modelled all-reduce of that many bytes.

    Every byte count, algorithm and CU count goes into a single config file and
    so a single process launch: the tests are independent, and a launch costs
    more than most of the tests do. The winner is the minimum over *both* the
    algorithm and the CU count, because a ceiling is what the fabric can do,
    not what one arbitrary choice of either achieves.
    """
    plans = build_plans(tp_size, byte_counts, cus, algos)
    if not plans:
        return {}
    config = "\n".join(line for p in plans for line in p.lines) + "\n"
    text = _run(binary, config, iters=iters, warmup=warmup, timeout=timeout)
    tests = parse_tests(text)

    expected = sum(len(p.lines) for p in plans)
    if len(tests) != expected:
        raise RuntimeError(
            f"TransferBench returned {len(tests)} test(s), expected {expected}; "
            "the output format may have changed"
        )

    best: dict = {}
    unresolved: set = set()
    pos = 0
    for plan in plans:
        chunk = tests[pos : pos + len(plan.lines)]
        pos += len(plan.lines)
        # Sum across the phases of a multi-test plan (two-shot), scale by the
        # step count (the ring runs one representative step), then keep the
        # fastest algorithm/CU pair for this byte count.
        us = sum(t.slowest_exec_ms for t in chunk) * 1e3 * plan.steps
        # A row below the text interface's resolution comes back nan (see
        # _duration_ms). Dropping it leaves the caller with no entry, which
        # renders as a blank cell -- reporting the 0 would instead read as an
        # infinitely fast fabric and take every efficiency to 0.
        if not math.isfinite(us) or us <= 0.0:
            unresolved.add(plan.nbytes)
            continue
        prev = best.get(plan.nbytes)
        if prev is None or us < prev.us:
            best[plan.nbytes] = Roof(us, plan.algo)

    for nbytes in sorted(unresolved - set(best)):
        logger.warning(
            "TransferBench: TP%d %d B is below the resolution of the "
            "reported table; leaving it blank",
            tp_size,
            nbytes,
        )
    return best


# ---------------------------------------------------------------------------
# Standalone entrypoint: dry-run the config, self-test the parser, or measure
# ---------------------------------------------------------------------------

# Captured from the TransferBench timing documentation, which is the only
# published sample of the exact table this module parses. Two executors, four
# transfers, one test -- enough to pin the executor rows, the aggregate row and
# the slowest-executor rule.
_SAMPLE = """Test 1:
-------------------┬--------------┬------------┬-------------------┬--------------------
 Executor: CPU 00   │  0.027 GB/s  │ 77.492 ms │    2097152 bytes  │  4.489 GB/s (sum)
-------------------┼--------------┼------------┼-------------------┼--------------------
    Transfer 0     │  4.476 GB/s  │   0.234 ms │    1048576 bytes  │  C0 -> C0:4 -> N
    Transfer 1     │  0.014 GB/s  │  77.359 ms │    1048576 bytes  │  G0 -> C0:4 -> N
-------------------┼--------------┼------------┼-------------------┼--------------------
 Executor: GPU 00   │ 97.436 GB/s  │   0.689 ms │   67108864 bytes  │ 129.692 GB/s (sum)
-------------------┼--------------┼------------┼-------------------┼--------------------
    Transfer 2     │ 80.886 GB/s  │   0.415 ms │   33554432 bytes  │  G0 -> G0:4 -> G0
    Transfer 3     │ 48.807 GB/s  │   0.687 ms │   33554432 bytes  │  G0 -> G0:4 -> G1
-------------------┼--------------┼------------┼-------------------┼--------------------
Aggregate (CPU)    │  0.891 GB/s  │  77.688 ms │   69206016 bytes  │  Overhead 0.197 ms
-------------------┴--------------┴------------┴-------------------┴--------------------
"""


def _self_test() -> None:
    """Check the parser against the captured sample, borders on and off."""
    for label, text in (
        ("bordered", _SAMPLE),
        # SHOW_BORDERS=0 drops the glyphs and leaves the columns space-separated.
        ("plain", _SAMPLE.replace("│", " ")),
    ):
        tests = parse_tests(text)
        assert len(tests) == 1, f"{label}: got {len(tests)} tests"
        t = tests[0]
        assert t.num == 1, f"{label}: test number {t.num}"
        # CPU row: 0.027 GB/s against 77.492 ms, so the time column wins and is
        # taken verbatim. GPU row: 97.436 GB/s against 0.689 ms, so the
        # bandwidth column wins and the duration is re-derived from it.
        assert set(t.exec_ms) == {"CPU 00", "GPU 00"}, f"{label}: {t.exec_ms}"
        assert t.exec_ms["CPU 00"] == 77.492, f"{label}: {t.exec_ms}"
        assert abs(t.exec_ms["GPU 00"] - 67108864 / 1e6 / 97.436) < 1e-9, t.exec_ms
        assert abs(t.exec_ms["GPU 00"] - 0.689) < 5e-4, "should agree to the printed ms"
        assert t.slowest_exec_ms == 77.492, f"{label}: {t.slowest_exec_ms}"
        assert t.overhead_ms == 0.197, f"{label}: {t.overhead_ms}"
        print(f"parser self-test ({label}): ok")

    # Border rows must not be mistaken for data, and text before the first
    # "Test N:" (the version banner, the env var dump) must be ignored.
    assert parse_tests("TransferBench v1.65\nAggregate (CPU) | 1 GB/s | 2 ms |") == []
    print("parser self-test (preamble): ok")

    # The whole reason durations are re-derived from bandwidth: a small-shape
    # roofline is sub-microsecond, and the ms column bottoms out at 0.000.
    sub_us = (
        "Test 1:\n"
        " Executor: GPU 00   │ 300.000 GB/s  │    0.000 ms │       114688 bytes  │\n"
    )
    got = parse_tests(sub_us)[0].slowest_exec_ms
    assert abs(got - 114688 / 1e6 / 300.0) < 1e-12, got
    assert abs(got * 1e3 - 0.382) < 1e-3, f"{got * 1e3} us"
    # Only when both columns are zero is the row genuinely unresolvable.
    dead = " Executor: GPU 00   │ 0.000 GB/s  │    0.000 ms │       114688 bytes  │\n"
    assert math.isnan(parse_tests("Test 1:\n" + dead)[0].slowest_exec_ms)
    print("parser self-test (sub-microsecond): ok")

    assert round16(114688) == 114688
    assert round16(114688 * 3 / 16) == 21504
    assert round16(8) == 16
    assert wire_bytes(114688, "qr_int4") == 28672
    assert wire_bytes(114688, "rccl") == 114688
    print("sizing self-test: ok")

    # cdr follows the host dispatch; everything else is pinned to its own
    # algorithm regardless of what cdr would have done at this shape.
    assert pattern("cdr", "1stage") == "1stage"
    assert pattern("cdr_naive", "2stage_naive") == "2stage"
    assert pattern("qr_int4", "1stage") == "2stage"
    assert pattern("rccl", "1stage") == "2stage"
    print("pattern self-test: ok")

    # Every byte count is planned against all three algorithms, so the roof is
    # a bound over algorithms rather than the cost of one.
    plans = build_plans(4, [4096], cus=(8,))
    assert [p.algo for p in plans] == ["one-shot", "two-shot", "ring"], plans
    by_algo = {p.algo: p for p in plans}

    assert len(by_algo["one-shot"].lines) == 1
    assert by_algo["one-shot"].steps == 1
    assert by_algo["one-shot"].lines[0] == (
        "-4 (G0G1G2G3 G0 G0 8 4096) (G0G1G2G3 G1 G1 8 4096) "
        "(G0G1G2G3 G2 G2 8 4096) (G0G1G2G3 G3 G3 8 4096)"
    ), by_algo["one-shot"].lines[0]

    two = by_algo["two-shot"]
    assert len(two.lines) == 2 and two.steps == 1, two
    # RS reads every rank including itself; AG writes only to the peers.
    assert two.lines[0] == (
        "-4 (G0G1G2G3 G0 G0 8 1024) (G0G1G2G3 G1 G1 8 1024) "
        "(G0G1G2G3 G2 G2 8 1024) (G0G1G2G3 G3 G3 8 1024)"
    ), two.lines[0]
    assert two.lines[1] == (
        "-4 (G0 G0 G1G2G3 8 1024) (G1 G1 G0G2G3 8 1024) "
        "(G2 G2 G0G1G3 8 1024) (G3 G3 G0G1G2 8 1024)"
    ), two.lines[1]

    # One representative step, scaled by 2(N-1) rather than emitted 6 times.
    ring = by_algo["ring"]
    assert len(ring.lines) == 1 and ring.steps == 6, ring
    assert ring.lines[0] == (
        "-4 (G0 G0 G1 8 1024) (G1 G1 G2 8 1024) (G2 G2 G3 8 1024) (G3 G3 G0 8 1024)"
    ), ring.lines[0]
    assert _ring_steps(2) == 2 and _ring_steps(8) == 14

    # Above the fan-in cap only the ring is expressible, and it still is.
    wide = build_plans(16, [4096], cus=(8,))
    assert [p.algo for p in wide] == ["ring"], wide
    print("config self-test: ok")


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-t", "--tp", type=int, default=4, help="world size")
    p.add_argument(
        "-b",
        "--bytes",
        type=int,
        nargs="*",
        default=[114688],
        help="payload byte counts (default: one bf16 8x7168 activation)",
    )
    p.add_argument(
        "-a",
        "--algo",
        nargs="*",
        default=None,
        choices=[name for name, *_ in _ALGOS],
        help="restrict the algorithms considered. Default: all of them, and\n"
        "the roof is the fastest -- which is the point. Pin one to compare\n"
        "patterns directly (e.g. ring vs two-shot on a NUMA-split host).",
    )
    p.add_argument("--cus", type=int, nargs="*", default=list(DEFAULT_CUS))
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--bin", default=None, help="path to the TransferBench binary")
    p.add_argument(
        "--dry-run", action="store_true", help="print the config that would run"
    )
    p.add_argument(
        "--self-test", action="store_true", help="check the parser and config builder"
    )
    args = p.parse_args()

    if args.self_test:
        _self_test()
        return

    byte_counts = [round16(b) for b in args.bytes]
    if args.dry_run:
        for plan in build_plans(args.tp, byte_counts, args.cus, args.algo):
            steps = f" x{plan.steps} steps" if plan.steps != 1 else ""
            print(f"# {plan.algo} {plan.nbytes}B x {plan.cus} CUs{steps}")
            print("\n".join(plan.lines))
        return

    binary = find_binary(args.bin)
    if binary is None:
        # Self-skip rather than exit non-zero. CI runs `python3 <file>` over
        # every .py under multigpu_tests/ (.github/scripts/aiter_test.sh), so a
        # hard failure here would fail the whole multi-GPU job on every runner
        # that lacks an optional third-party binary.
        logger.warning(
            "TransferBench not found; nothing to do. Build it from "
            "https://github.com/ROCm/TransferBench and set $TRANSFERBENCH "
            "or pass --bin."
        )
        return
    got = measure(
        args.tp,
        byte_counts,
        binary=binary,
        cus=args.cus,
        algos=args.algo,
        iters=args.iters,
        warmup=args.warmup,
    )
    for nbytes, roof in sorted(got.items()):
        print(
            f"{nbytes:>12,d} B  {roof.us:9.2f} us  "
            f"{nbytes / roof.us / 1e3:7.1f} GB/s  via {roof.algo}"
        )


if __name__ == "__main__":
    main()
