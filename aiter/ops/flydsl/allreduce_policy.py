# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which FlyDSL all-reduce schedule to run, as a function of payload size.

There are three schedules and the fastest one is a function of how many bytes
are being reduced:

* **one-shot** (``OneShotAllReduce``) -- one round, no grid-wide barrier, wire
  volume ``(N-1)*S``. Exact: fp32 accumulate, one bf16 rounding.
* **mesh** (``QuickAllReduceInt4(algorithm="mesh")``) -- two-shot, fanout to all
  ``N-1`` peers twice, wire volume ``2(N-1)/N*S``, INT4 on the wire.
* **ring** (``QuickAllReduceInt4(algorithm="ring")``) -- two-shot, ``2(N-1)``
  hops, same wire volume as the mesh, traded for per-destination locality.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

logger = logging.getLogger("aiter")

# Fabric between GPUs.
LINKS = ("pcie", "xgmi")

SUPPORTED_WORLDS = (2, 4, 8)

# No ceiling. The ring's inbox is a fixed ring of wire slots sized by ``ST * grid``.
NO_MAX = 1 << 62


@dataclass(frozen=True)
class FamilyPolicy:
    """Family boundaries for one ``(link, world_size)``, in payload bytes.

    ``nbytes <= oneshot_max``  -> one-shot
    ``nbytes <= mesh_max``     -> mesh
    otherwise                  -> ring

    ``min_bytes`` is where this whole path starts being worth taking; below it
    the caller should fall through to other alternatives.
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


FAMILY_POLICY: dict[tuple[str, int], FamilyPolicy] = {
    # --- PCIe: Policy from measurements --------------------
    ("pcie", 2): FamilyPolicy(
        oneshot_max=512 << 10, oneshot_max_exact=512 << 10, mesh_max=4 << 20
    ),
    ("pcie", 4): FamilyPolicy(
        oneshot_max=96 << 10, oneshot_max_exact=96 << 10, mesh_max=12 << 20
    ),
    ("pcie", 8): FamilyPolicy(
        oneshot_max=32 << 10, oneshot_max_exact=48 << 10, mesh_max=12 << 20
    ),
    # --- xGMI: Not yet measured, conservative placeholder --------------------
    #
    # Set ``mesh_max`` so the ring is never auto-selected. The mesh is the
    # default algorithm on a meshed fabric and the ring is structurally worse
    # there -- it trades fanout for per-destination locality, which is what a
    # PCIe host wants and an xGMI host does not.
    #
    # The one-shot boundary above is the part most likely to be wrong. Its
    # driver, the ``N/2`` wire-volume ratio, is fabric-independent, but the
    # constant is not: xGMI peer bandwidth is an order of magnitude higher.
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

# --- environment variables ---------------------------------------------------------
#
# TODO: can we re-use the existing AITER_AR_1STAGE* / AITER_AR_QUANT_* variables?

ENABLE_VAR = "AITER_FLY_AR"
ACCURACY_VAR = "AITER_FLY_AR_ACCURACY"
ONESHOT_MAX_VAR = "AITER_FLY_AR_ONESHOT_MAX_BYTES"
MESH_MAX_VAR = "AITER_FLY_AR_MESH_MAX_BYTES"

ACCURACY_MODES = ("exact", "fast")
DEFAULT_ACCURACY = "exact"


def _env_int(name: str) -> int | None:
    """A non-negative override from *name*, or None. ``-1`` means "use the table"."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        val = int(raw)
    except ValueError:
        logger.warning("FlyDSL QR: ignoring %s=%r, expected an integer", name, raw)
        return None
    return None if val < 0 else val


def enabled() -> bool:
    """Whether ``AITER_FLY_AR`` opts in to the FlyDSL all-reduce path.

    Opt-in only, and only ``"1"`` opts in -- unset, ``"0"`` and anything else
    means disabled.
    """
    return os.environ.get(ENABLE_VAR, "").strip() == "1"


def accuracy_mode() -> str:
    """``"exact"`` (default) or ``"fast"``, from ``AITER_FLY_AR_ACCURACY``."""
    raw = os.environ.get(ACCURACY_VAR)
    if raw is None or not raw.strip():
        return DEFAULT_ACCURACY
    mode = raw.strip().lower()
    if mode not in ACCURACY_MODES:
        logger.warning(
            "FlyDSL QR: ignoring %s=%r, expected one of %s",
            ACCURACY_VAR,
            raw,
            ACCURACY_MODES,
        )
        return DEFAULT_ACCURACY
    return mode


def resolve(link: str, world_size: int, mode: str | None = None) -> FamilyPolicy:
    """The policy in force for a rank, environment overrides applied.

    *mode* defaults to ``accuracy_mode()``, and picks between two different
    policies, not just two boundaries:

    * ``"fast"`` -- the full three-family policy. One-shot up to
      ``oneshot_max``, mesh/ring (quantized) beyond it.
    * ``"exact"`` (default) -- **only** the one-shot is ever reachable, at its
      widened ``oneshot_max_exact`` ceiling. Above that, this returns a policy
      with no mesh/ring window at all (``mesh_max == max_bytes ==
      oneshot_max``), so ``should_fly_all_reduce`` declines the payload and the
      caller falls through to whatever it would otherwise dispatch to
      (``cross_device_reduce``/RCCL) rather than silently quantizing.

    ``AITER_FLY_AR_ONESHOT_MAX_BYTES`` applies in both modes -- it only moves
    where the one-shot's own ceiling sits. ``AITER_FLY_AR_MESH_MAX_BYTES`` is
    ignored (with a warning) in ``"exact"`` mode: honouring it would reopen the
    mesh/ring window ``"exact"`` exists to close.
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

    override_one = _env_int(ONESHOT_MAX_VAR)

    if mode == "exact":
        one = base.oneshot_max_exact if override_one is None else override_one
        if _env_int(MESH_MAX_VAR) is not None:
            logger.warning(
                "FlyDSL QR: ignoring %s in accuracy=exact mode -- exact mode "
                "has no mesh/ring window to widen. Set %s=fast to use it.",
                MESH_MAX_VAR,
                ACCURACY_VAR,
            )
        return FamilyPolicy(
            oneshot_max=one,
            oneshot_max_exact=one,
            mesh_max=one,
            min_bytes=base.min_bytes,
            max_bytes=one,
        )

    one = base.oneshot_max if override_one is None else override_one
    mesh = base.mesh_max
    override_mesh = _env_int(MESH_MAX_VAR)
    if override_mesh is not None:
        mesh = override_mesh
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
    """Families a *policy* can ever select, in size order."""
    out = []
    if policy.oneshot_max >= policy.min_bytes:
        out.append("oneshot")
    if policy.mesh_max > policy.oneshot_max:
        out.append("mesh")
    if policy.max_bytes > policy.mesh_max:
        out.append("ring")
    return tuple(out)
