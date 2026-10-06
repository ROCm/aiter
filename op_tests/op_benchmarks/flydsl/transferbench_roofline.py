# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""TransferBench-backed bandwidth roofline for ``bench_comm_allreduce.py``.

``bench_comm_allreduce.py`` compares aiter's all-reduce candidates against each
other and against RCCL. This module supplies the roofline, by asking TransferBench
(https://github.com/ROCm/TransferBench) to move the same bytes in the same
pattern with none of the collective's semantics attached.

The one TransferBench feature that makes this possible: a Transfer is defined
as "an Executor reads and **adds** values from source memory, then writes the
sum to destination memory", and source/destination locations concatenate.

What the roofline is
===============================

The roof for a byte count is the **best over algorithms**, not the throughput of
one. 

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

Usage
=====

Requires the ``TransferBench`` binary, which is not part of a default ROCm
install -- it ships with ROCmValidationSuite, or build it from source
(https://github.com/ROCm/TransferBench). Point ``$TRANSFERBENCH`` at it or pass
``--roofline-bin``. When it cannot be found the caller gets an empty result and
a warning rather than an exception.

Standalone, to see what would be run and to check the parser::

    python3 op_tests/op_benchmarks/flydsl/transferbench_roofline.py --dry-run -t 4
    python3 op_tests/op_benchmarks/flydsl/transferbench_roofline.py --self-test
    python3 op_tests/op_benchmarks/flydsl/transferbench_roofline.py -t 4 -b 114688
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
# input.
SCALE_RATIO = 128.0 / 4096.0
WIRE_RATIO = {
    "cdr": 1.0,
    "cdr_naive": 1.0,
    "cdr_fp8": 0.5,
    "qr_fp": 1.0,
    "qr_fp8": 8.0 / 16.0 + SCALE_RATIO,
    "qr_int6": 6.0 / 16.0 + SCALE_RATIO,
    "qr_int4": 4.0 / 16.0 + SCALE_RATIO,
    "qr_int3": 3.0 / 16.0 + SCALE_RATIO,
    "fly_int4": 4.0 / 16.0 + SCALE_RATIO,
    "fly_int4_ring": 4.0 / 16.0 + SCALE_RATIO,
    # Follows its reported variant (``wire_ratio``); this is only the fallback
    # for a row with none, and is the exact one-shot's ratio.
    "fly_auto": 1.0,
    # Exact, bf16 on the wire -- no codec, so the payload dtype is the wire
    # dtype.
    "fly_1stage": 1.0,
    "rccl": 1.0,
    # ---- --fusion ar_rmsnorm rows -----------------------------------------
    # Fusing an RMSNorm epilogue changes what happens to the bytes after they
    # land, never how many cross the fabric, so every ratio here is its plain
    # counterpart's.
    "fused_cdr_1stage": 1.0,
    "fused_cdr_2stage": 1.0,
    "fused_fly_1stage": 1.0,
    "fused_qr_fp8": 8.0 / 16.0 + SCALE_RATIO,
    "fused_qr_int4": 4.0 / 16.0 + SCALE_RATIO,
    "fused_fly_ring": 4.0 / 16.0 + SCALE_RATIO,
    "fused_fly_mesh": 4.0 / 16.0 + SCALE_RATIO,
    # Follows its reported variant, as ``fly_auto`` does.
    "fused_fly_auto": 1.0,
    # Two-launch baselines: the all-reduce is the plain kernel, and the norm
    # that follows it is local -- no fabric traffic at all.
    "separate_cdr": 1.0,
    "separate_rccl": 1.0,
    "separate_fly1s": 1.0,
    "separate_qr_int4": 4.0 / 16.0 + SCALE_RATIO,
    "separate_flyring": 4.0 / 16.0 + SCALE_RATIO,
    "separate_flymesh": 4.0 / 16.0 + SCALE_RATIO,
    "separate_fly_auto": 1.0,
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
    "fly_1stage": "1stage",
    "rccl": "2stage",
}


def pattern(cand_key: str, predicted: str) -> str:
    """Which algorithm *cand_key* itself runs, for reporting."""
    m = _RS_CODEC_KEY.match(cand_key)
    fixed = _FIXED_PATTERN.get(base_key(m.group(1) if m else cand_key))
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


# Tuning suffixes that never change the wire shape: ``_b<N>`` (block size),
# ``_st<N>`` (two-shot/ring super-tile), ``_g<N>`` (grid cap), ``_a<N>`` (atoms
# per thread), ``_fa`` (fanout order), ``_k<N>`` (fused one-shot row split). All
# of them change how the bytes are scheduled, none of them change how many there
# are or which pattern is driven.
_TUNING_SUFFIX = re.compile(r"_(?:b\d+|st\d+|g\d+|a\d+|fa|k\d+)$")

# A pinned reduce-scatter codec, ``fly_int4_ring_st8_int6``: the RS lap runs at
# that width and the AG lap stays int4. Anchored on the ring family because
# ``qr_int6`` and ``fly_int4`` also end in ``_int<N>`` and mean something else.
_RS_CODEC_KEY = re.compile(r"^(fly_int4_ring.*)_int(\d+)$")
_AG_CODEC_BITS = 4

# The codec(s) a FlyDSL engine reports in ``variant()``: mesh names one
# (``..._uncached_int4_b256/grid_x16``), ring names the RS and AG laps
# (``..._finegrained_int6_int4_b256/...``). One-shot variants name none.
_VARIANT_CODEC = re.compile(r"_(?:uncached|finegrained)_(int\d+(?:_int\d+)?)_b\d+")
_VARIANT_EXACT = ("one_shot_allreduce", "oneshot:")


def base_key(cand_key: str) -> str:
    """Strip every tuning suffix; variants share their base's wire shape."""
    while True:
        stripped = _TUNING_SUFFIX.sub("", cand_key)
        if stripped == cand_key:
            return cand_key
        cand_key = stripped


def variant_wire_ratio(variant) -> float | None:
    """Wire bytes per payload byte for a reported kernel *variant*, or None.

    Returns None for anything it does not recognise (including NaN, ``n/a`` and
    a cross-rank disagreement flag), leaving the caller to use the key.
    """
    if not isinstance(variant, str):
        return None
    m = _VARIANT_CODEC.search(variant)
    if m:
        bits = [int(b) for b in re.findall(r"int(\d+)", m.group(1))]
        return sum(bits) / len(bits) / 16.0 + SCALE_RATIO
    if any(tag in variant for tag in _VARIANT_EXACT):
        return 1.0
    return None


_warned_keys: set = set()


def wire_ratio(cand_key: str, variant=None) -> float:
    """Wire bytes per payload byte for *cand_key*."""

    # ``fly_*``, ``fused_fly_*`` and ``separate_fly_auto``: every FlyDSL row
    # reports the kernel it ran, and the ``*fly_auto`` policies have no static
    # wire at all -- it is whichever family they picked at this shape.
    if "fly_" in cand_key:
        ratio = variant_wire_ratio(variant)
        if ratio is not None:
            return ratio
    m = _RS_CODEC_KEY.match(cand_key)
    if m:
        return (int(m.group(2)) + _AG_CODEC_BITS) / 32.0 + SCALE_RATIO
    base = base_key(cand_key)
    if base in WIRE_RATIO:
        return WIRE_RATIO[base]
    if cand_key not in _warned_keys:
        _warned_keys.add(cand_key)
        logger.warning(
            "TransferBench: no wire ratio for candidate %r (resolved to %r); "
            "grading it at 1.0 -- add it to WIRE_RATIO",
            cand_key,
            base,
        )
    return 1.0


def has_wire_ratio(cand_key: str) -> bool:
    """Whether *cand_key* resolves through the key alone, without the fallback."""
    return bool(_RS_CODEC_KEY.match(cand_key)) or base_key(cand_key) in WIRE_RATIO


def wire_bytes(payload_bytes: int, cand_key: str, variant=None) -> int:
    """Bytes *cand_key* puts on the wire for a *payload_bytes* all-reduce."""
    return round16(payload_bytes * wire_ratio(cand_key, variant))


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------


def _advanced(transfers) -> str:
    """One config line in advanced mode: ``-N (src exe dst SEs Bytes) ...``."""

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
    # traffic no all-gather performs.
    others = ["".join(f"G{j}" for j in range(tp) if j != i) for i in range(tp)]
    rs = _advanced([(peers, f"G{i}", f"G{i}", cus, chunk) for i in range(tp)])
    ag = _advanced([(f"G{i}", f"G{i}", others[i], cus, chunk) for i in range(tp)])
    return [rs, ag]


def _ring(tp: int, nbytes: int, cus: int) -> list[str]:
    """One step of a ring all-reduce; the caller scales by ``_ring_steps``.

    Each rank sends its chunk to its successor and receives from its
    predecessor -- point-to-point only, never a fan-in. 
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


# name -> (emit, steps(tp), fan-in(tp), split(tp)). ``fan-in`` is the largest
# source or destination list the pattern builds, checked against MAX_FANIN.
# ``split`` is how many ways the pattern divides its byte count: the bytes each
# Transfer carries are ``nbytes / split``, which is what the launch-floor probe
# below needs to pin at one minimal Transfer.
_ALGOS = (
    ("one-shot", _one_shot, lambda tp: 1, lambda tp: tp, lambda tp: 1),
    ("two-shot", _two_shot, lambda tp: 1, lambda tp: tp, lambda tp: tp),
    ("ring", _ring, _ring_steps, lambda tp: 1, lambda tp: tp),
)

# Bytes per Transfer in the launch-floor probe.
FLOOR_BYTES = 1024
FLOOR_ITERS = 200
FLOOR_WARMUP = 10

# Byte counts up to this are measured at ``FLOOR_ITERS`` rather than the caller's
# count, in their own process.
SMALL_BYTES = 4 << 20

# A test whose time exceeds its floor by less than this has no resolvable
# bandwidth term.
MIN_RESOLVED_US = 0.15


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
        for name, emit, steps, fanin, _split in _ALGOS:
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
    """Best available duration for one row, in ms. """
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
        """The executor the collective would actually wait on."""
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
            # the transfers at prefill sizes.
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


def measure_floors(
    tp_size: int,
    *,
    binary: str,
    cus=DEFAULT_CUS,
    algos=None,
    iters: int = FLOOR_ITERS,
    warmup: int = FLOOR_WARMUP,
    timeout: float = 900.0,
) -> dict:
    """``{(algo, cus): [ms per test]}`` -- what a minimal test costs to run. """
    keys, config = [], []
    for name, emit, _steps, fanin, split in _ALGOS:
        if algos is not None and name not in algos:
            continue
        if fanin(tp_size) > MAX_FANIN:
            continue
        for cu in cus:
            lines = emit(tp_size, FLOOR_BYTES * split(tp_size), cu)
            keys.append(((name, cu), len(lines)))
            config.extend(lines)
    if not keys:
        return {}
    tests = parse_tests(
        _run(
            binary,
            "\n".join(config) + "\n",
            iters=iters,
            warmup=warmup,
            timeout=timeout,
        )
    )
    if len(tests) != len(config):
        raise RuntimeError(
            f"TransferBench returned {len(tests)} floor test(s), "
            f"expected {len(config)}; the output format may have changed"
        )
    floors, pos = {}, 0
    for key, n in keys:
        floors[key] = [t.slowest_exec_ms for t in tests[pos : pos + n]]
        pos += n
    return floors


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
    subtract_floor: bool = True,
) -> dict:
    """``{bytes: Roof}`` -- the fastest modelled all-reduce of that many bytes."""
    plans = build_plans(tp_size, byte_counts, cus, algos)
    if not plans:
        return {}
    floors = (
        measure_floors(
            tp_size, binary=binary, cus=cus, algos=algos, timeout=timeout
        )
        if subtract_floor
        else {}
    )
    if floors:
        logger.info(
            "TransferBench: TP%d launch floor (us): %s",
            tp_size,
            ", ".join(
                f"{a}@{c}CU {'+'.join(f'{x * 1e3:.2f}' for x in v)}"
                for (a, c), v in sorted(floors.items())
            ),
        )
    small = [p for p in plans if p.nbytes <= SMALL_BYTES]
    large = [p for p in plans if p.nbytes > SMALL_BYTES]
    best: dict = {}
    unresolved: set = set()
    for group, group_iters, group_warmup in (
        (small, max(iters, FLOOR_ITERS), max(warmup, FLOOR_WARMUP)),
        (large, iters, warmup),
    ):
        if not group:
            continue
        config = "\n".join(line for p in group for line in p.lines) + "\n"
        text = _run(
            binary, config, iters=group_iters, warmup=group_warmup, timeout=timeout
        )
        tests = parse_tests(text)

        expected = sum(len(p.lines) for p in group)
        if len(tests) != expected:
            raise RuntimeError(
                f"TransferBench returned {len(tests)} test(s), expected "
                f"{expected}; the output format may have changed"
            )

        pos = 0
        for plan in group:
            chunk = tests[pos : pos + len(plan.lines)]
            pos += len(plan.lines)
            floor = floors[(plan.algo, plan.cus)] if floors else [0.0] * len(chunk)
            # Per test: take off its own launch floor, and sum across the phases
            # of a multi-test plan (two-shot).
            bw_us = sum(t.slowest_exec_ms - f for t, f in zip(chunk, floor)) * 1e3
            if not math.isfinite(bw_us) or bw_us < (
                MIN_RESOLVED_US if floors else 0.0
            ):
                unresolved.add(plan.nbytes)
                continue
            # Scale by the step count (the ring runs one representative step),
            # then keep the fastest algorithm/CU pair for this byte count.
            us = bw_us * plan.steps
            if us <= 0.0:
                unresolved.add(plan.nbytes)
                continue
            prev = best.get(plan.nbytes)
            if prev is None or us < prev.us:
                best[plan.nbytes] = Roof(us, plan.algo)

    for nbytes in sorted(unresolved - set(best)):
        logger.warning(
            "TransferBench: TP%d %d B does not clear the launch floor or the "
            "resolution of the reported table; leaving it blank",
            tp_size,
            nbytes,
        )
    return best


# ---------------------------------------------------------------------------
# Standalone entrypoint: dry-run the config, self-test the parser, or measure
# ---------------------------------------------------------------------------

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
    p.add_argument(
        "--no-floor",
        action="store_true",
        help="report raw TransferBench times instead of subtracting each\n"
        "test's fixed launch cost (the default, bandwidth-only roof)",
    )
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

    byte_counts = [round16(b) for b in args.bytes]
    if args.dry_run:
        for plan in build_plans(args.tp, byte_counts, args.cus, args.algo):
            steps = f" x{plan.steps} steps" if plan.steps != 1 else ""
            print(f"# {plan.algo} {plan.nbytes}B x {plan.cus} CUs{steps}")
            print("\n".join(plan.lines))
        if not args.no_floor:
            print(f"# launch floor probe, {FLOOR_BYTES} B per Transfer")
            for name, emit, _steps, fanin, split in _ALGOS:
                if (args.algo and name not in args.algo) or fanin(args.tp) > MAX_FANIN:
                    continue
                for cu in args.cus:
                    print(f"# {name} x {cu} CUs")
                    print("\n".join(emit(args.tp, FLOOR_BYTES * split(args.tp), cu)))
        return

    binary = find_binary(args.bin)
    if binary is None:
        # Warn and return rather than exit non-zero: TransferBench is an
        # optional third-party binary, and its absence is not an error.
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
        subtract_floor=not args.no_floor,
    )
    for nbytes, roof in sorted(got.items()):
        print(
            f"{nbytes:>12,d} B  {roof.us:9.2f} us  "
            f"{nbytes / roof.us / 1e3:7.1f} GB/s  via {roof.algo}"
        )


if __name__ == "__main__":
    main()
