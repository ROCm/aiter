# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which FlyDSL all-reduce schedule to run, as a function of payload size.

There are three schedules and the fastest one is a function of how many bytes
are being reduced:

* **one-shot** (``OneShotAllReduce``) -- one round, no grid-wide barrier, wire
  volume ``(N-1)*S``. Exact: fp32 accumulate, one bf16 rounding, ~55 dB.
* **mesh** (``QRInt4(algorithm="mesh")``) -- two-shot, fanout to all ``N-1``
  peers twice, wire volume ``2(N-1)/N*S``, INT4 on the wire, ~19 dB.
* **ring** (``QRInt4(algorithm="ring")``) -- two-shot, ``2(N-1)`` hops, same
  wire volume as the mesh traded for per-destination locality.

They are ordered by size on every fabric measured, so the whole family decision
is **two thresholds per (link type, world size)** -- which is what this module
is. Deliberately no runtime search, no probing, no first-call calibration: a
collective on the critical path cannot afford to discover its own policy, and a
table that a reviewer can read is worth more than a percent of latency.

This module imports nothing from the kernels. It is data plus arithmetic, so
the tables can be tested without a GPU, without flydsl, and without an IPC
rendezvous -- see ``op_tests/flydsl_tests/test_flydsl_ar_policy.py``. The caller
resolves the link type (``qr_int4.has_xgmi_peer_links``) and passes it in.

Why the one-shot boundary has to be keyed on world size
-------------------------------------------------------

The one-shot moves ``(N-1)*S`` where a two-shot moves ``2(N-1)/N*S`` -- a ratio
of ``N/2``. At TP2 the two are *equal*, so the one-shot's single round and
absent barrier win far up the size range; at TP8 it is pushing 4x the bytes and
loses almost immediately. A single world-independent ceiling is wrong in both
directions at once and cannot be fixed by moving it, which is what the 192 KiB
constant this replaces was: 2.7x too low at TP2, 6x too high at TP8, and worth
2.08x worst-case at TP8 against a per-shape oracle.

Provenance
----------

The PCIe rows are fitted from a 52-point byte ladder (14 KiB .. 112 MiB, four
points per octave) on an MI350P, by exhaustive search over round thresholds
minimising worst-case regret against a per-shape oracle. Worst case 1.002x
(TP2), 1.016x (TP4), 1.066x (TP8). Held out against hidden sizes 4096 and 8192
at 1.000x, which is what justifies keying the table on bytes rather than on
shape. See ``op_tests/dump_data/sweep/RESULTS.md`` and regenerate with
``op_tests/multigpu_tests/fit_allreduce_policy.py``.

**The xGMI rows are not measured.** See ``_XGMI_UNMEASURED`` below.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("aiter")

# Fabric between GPUs. Arch does not determine it: an MI350X (xGMI) and an
# MI350P (PCIe-only) both report gfx950 and want different answers here, the
# same reason ``inbox_memory="auto"`` reads the KFD topology.
LINKS = ("pcie", "xgmi")

SUPPORTED_WORLDS = (2, 4, 8)

# No ceiling. The ring's inbox is a fixed ring of wire slots sized by
# ``ST * grid``, not by the payload, so a 112 MiB reduce costs no more memory
# than a 1 MiB one and still beats RCCL by 2.8-4.0x at the top of the sweep.
NO_MAX = 1 << 62


@dataclass(frozen=True)
class FamilyPolicy:
    """Family boundaries for one ``(link, world_size)``, in payload bytes.

    ``nbytes <= oneshot_max``  -> one-shot
    ``nbytes <= mesh_max``     -> mesh
    otherwise                  -> ring

    ``oneshot_max_exact`` is the same boundary under the default
    exactness-preferring policy: the largest payload at which the bit-exact
    one-shot is still within ``EXACT_SLACK`` of the fastest quantized schedule.
    It is always ``>= oneshot_max`` -- widening the window is the only thing
    preferring exactness can do -- and on PCIe it is *equal* at TP2 and TP4,
    i.e. keeping ~55 dB instead of ~19 dB costs nothing at all there.

    ``min_bytes`` is where this whole path starts being worth taking; below it
    the caller should fall through to ``cross_device_reduce``.
    """

    oneshot_max: int
    oneshot_max_exact: int
    mesh_max: int
    min_bytes: int = 0
    max_bytes: int = NO_MAX

    def __post_init__(self):
        if not self.oneshot_max_exact >= self.oneshot_max:
            raise ValueError(
                f"oneshot_max_exact ({self.oneshot_max_exact}) must be >= "
                f"oneshot_max ({self.oneshot_max}): preferring the exact "
                "schedule can only widen its window"
            )
        if self.mesh_max < self.oneshot_max_exact:
            raise ValueError(
                f"mesh_max ({self.mesh_max}) must be >= oneshot_max_exact "
                f"({self.oneshot_max_exact}); the families partition by size"
            )


# How much latency the default policy will give up to keep the exact schedule.
# The one-shot is bit-comparable with ``cross_device_reduce`` (~55 dB); the
# quantized schedules are ~19 dB, which is ~11% relative error. 10% of a
# microsecond-scale collective is a cheaper thing to spend than that.
EXACT_SLACK = 1.10

# Placeholder marker for the unmeasured rows, so the log line and the tests can
# find them without duplicating the list.
_XGMI_UNMEASURED = True

FAMILY_POLICY: dict[tuple[str, int], FamilyPolicy] = {
    # --- PCIe: measured, MI350P, 2026-09-10 ------------------------------
    # TP2: the one-shot and the two-shots move *identical* wire volume
    # ((N-1)*S == 2(N-1)/N*S at N=2), so the exact schedule stays ahead on its
    # single round all the way to 512 KiB, and exactness is free.
    ("pcie", 2): FamilyPolicy(
        oneshot_max=512 << 10, oneshot_max_exact=512 << 10, mesh_max=4 << 20
    ),
    ("pcie", 4): FamilyPolicy(
        oneshot_max=96 << 10, oneshot_max_exact=96 << 10, mesh_max=12 << 20
    ),
    # TP8 is the only row where preferring exactness costs anything: 32 -> 48 KiB
    # for a worst case of 1.066x.
    ("pcie", 8): FamilyPolicy(
        oneshot_max=32 << 10, oneshot_max_exact=48 << 10, mesh_max=12 << 20
    ),
    # --- xGMI: NOT MEASURED, conservative placeholder --------------------
    #
    # Deliberately conservative in the one direction that cannot regress
    # anything: keep the one-shot inside its historical 192 KiB ceiling, and set
    # ``mesh_max`` so the **ring is never auto-selected**. The mesh is the
    # documented default on a meshed fabric and the ring is structurally worse
    # there -- it trades fanout for per-destination locality, which is what a
    # PCIe host wants and an xGMI host does not.
    #
    # The one-shot boundary above is the part most likely to be wrong. Its
    # driver, the ``N/2`` wire-volume ratio, is fabric-independent, but the
    # constant is not: xGMI peer bandwidth is an order of magnitude higher, which
    # moves where a launch-bound schedule stops winning. Replace these three rows
    # from an MI350X sweep -- the fitting script and shape files already exist.
    ("xgmi", 2): FamilyPolicy(
        oneshot_max=192 << 10, oneshot_max_exact=192 << 10, mesh_max=NO_MAX
    ),
    ("xgmi", 4): FamilyPolicy(
        oneshot_max=192 << 10, oneshot_max_exact=192 << 10, mesh_max=NO_MAX
    ),
    ("xgmi", 8): FamilyPolicy(
        oneshot_max=192 << 10, oneshot_max_exact=192 << 10, mesh_max=NO_MAX
    ),
}

# --- environment ---------------------------------------------------------
#
# The existing AITER_AR_1STAGE* / AITER_AR_QUANT_* variables belong to the fused
# all-reduce + RMSNorm + mxfp4 path in communicator_cuda.py and name a different
# kernel's 1-stage; they are deliberately not reused. AITER_ALL_REDUCE_CODEC
# (qr_int4.py) still applies and is unchanged -- it selects the wire format,
# which is orthogonal to the schedule chosen here.

ENABLE_VAR = "AITER_FLY_AR"
ACCURACY_VAR = "AITER_FLY_AR_ACCURACY"
ONESHOT_MAX_VAR = "AITER_FLY_AR_ONESHOT_MAX_BYTES"
MESH_MAX_VAR = "AITER_FLY_AR_MESH_MAX_BYTES"

ACCURACY_MODES = ("exact", "fast")
DEFAULT_ACCURACY = "exact"


def _env_int(name: str) -> int | None:
    """A non-negative override from *name*, or None. ``-1`` means "use the table".

    Matches the house sentinel convention (``AITER_CUSTOM_AR_MAX_SIZE``). A value
    that does not parse warns and is ignored rather than raising: a typo in an
    environment variable should not take a model down at import.
    """
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        val = int(raw)
    except ValueError:
        logger.warning("QRInt4: ignoring %s=%r, expected an integer", name, raw)
        return None
    return None if val < 0 else val


def enabled() -> bool | None:
    """Tristate ``AITER_FLY_AR``: True forces on, False forces off, None auto.

    Same idiom as ``AITER_AR_1STAGE`` in ``communicator_cuda.py``.
    """
    return {"1": True, "0": False}.get(os.environ.get(ENABLE_VAR, "").strip())


def accuracy_mode() -> str:
    """``"exact"`` (default) or ``"fast"``, from ``AITER_FLY_AR_ACCURACY``."""
    raw = os.environ.get(ACCURACY_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_ACCURACY
    mode = raw.strip().lower()
    if mode not in ACCURACY_MODES:
        logger.warning(
            "QRInt4: ignoring %s=%r, expected one of %s",
            ACCURACY_VAR,
            raw,
            ACCURACY_MODES,
        )
        return DEFAULT_ACCURACY
    return mode


def resolve(link: str, world_size: int, mode: str | None = None) -> FamilyPolicy:
    """The policy in force for a rank, environment overrides applied.

    *mode* defaults to ``accuracy_mode()``. In ``"exact"`` the one-shot window is
    ``oneshot_max_exact``; in ``"fast"`` it is ``oneshot_max``. Either way the
    returned record has both fields set to the resolved value, so callers never
    have to re-apply the mode.
    """
    if link not in LINKS:
        raise ValueError(f"link must be one of {LINKS}, got {link!r}")
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    base = FAMILY_POLICY[(link, int(world_size))]
    mode = accuracy_mode() if mode is None else mode
    if mode not in ACCURACY_MODES:
        raise ValueError(f"mode must be one of {ACCURACY_MODES}, got {mode!r}")

    one = base.oneshot_max_exact if mode == "exact" else base.oneshot_max
    mesh = base.mesh_max
    override_one = _env_int(ONESHOT_MAX_VAR)
    override_mesh = _env_int(MESH_MAX_VAR)
    if override_one is not None:
        one = override_one
    if override_mesh is not None:
        mesh = override_mesh
    # An override can invert the ordering the families depend on. Clamp rather
    # than raise -- someone pinning the one-shot ceiling above the ring floor
    # means "give me the one-shot up to here", not "crash".
    mesh = max(mesh, one)
    return FamilyPolicy(
        oneshot_max=one,
        oneshot_max_exact=one,
        mesh_max=mesh,
        min_bytes=base.min_bytes,
        max_bytes=base.max_bytes,
    )


def pick_family(nbytes: int, policy: FamilyPolicy) -> str:
    """``"oneshot"`` | ``"mesh"`` | ``"ring"`` for a payload of *nbytes*."""
    if nbytes <= policy.oneshot_max:
        return "oneshot"
    return "mesh" if nbytes <= policy.mesh_max else "ring"


def families_reachable(policy: FamilyPolicy) -> tuple[str, ...]:
    """Families a *policy* can ever select, in size order.

    Engine construction is a collective IPC handle exchange and so cannot be
    lazy -- every engine a rank might need has to be built up front, in an order
    every rank agrees on. Building the ones the table can never reach would cost
    an inbox and a JIT for nothing, which on xGMI (where ``mesh_max`` is
    unbounded) is the whole ring.
    """
    out = []
    if policy.oneshot_max >= policy.min_bytes:
        out.append("oneshot")
    if policy.mesh_max > policy.oneshot_max:
        out.append("mesh")
    if policy.max_bytes > policy.mesh_max:
        out.append("ring")
    return tuple(out)
