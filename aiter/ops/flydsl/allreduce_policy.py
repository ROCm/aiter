# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Which FlyDSL all-reduce schedule to run, as a function of payload size.

There are three schedules and the fastest one is a function of how many bytes
are being reduced:

* **one-shot** (``OneShotAllReduce``) -- one round, no grid-wide barrier, wire
  volume ``(N-1)*S``. Exact: fp32 accumulate, one bf16 rounding.
* **mesh** (``FlyQuickAllReduce(algorithm="mesh")``) -- two-shot, fanout to all
  ``N-1`` peers twice, wire volume ``2(N-1)/N*S``, INT4 on the wire.
* **ring** (``FlyQuickAllReduce(algorithm="ring")``) -- two-shot, ``2(N-1)``
  hops, same wire volume as the mesh, traded for per-destination locality.

They do not share a dispatcher. Each lives in the aiter slot whose accuracy
contract it already matches, and this module hands each slot its own view of
one shared table row:

* ``resolve_oneshot`` -> ``CustomAllreduce``, which is exact.
* ``resolve_quant``   -> ``QuickAllReduce``, which is allowed to quantize.

The fused all-reduce+RMSNorm path is different: all three of its families live
in one slot (``QuickAllReduce``), so it keeps a single unified view
(``FusedPolicy`` / ``resolve_fused``) rather than the split above. Its accuracy
default is ``exact`` -- fused output feeds the next layer, so quantizing is
opt-in via ``AITER_FLY_AR_ACCURACY=fast``.
"""

from __future__ import annotations

import functools
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

    ``min_bytes`` is where this whole path starts being worth taking; below it
    the caller should fall through to other alternatives.

    The two one-shot ceilings are measured against *different* alternatives and
    so do not order against each other:

    * ``oneshot_max`` is where the quantized **mesh** overtakes the one-shot.
      It is the quick-reduce slot's exclusive floor.
    * ``oneshot_max_exact`` is where ``cross_device_reduce``/RCCL -- what the
      payload reaches if every FlyDSL family declines -- overtakes it. It is the
      custom-all-reduce slot's ceiling.
    """

    oneshot_max: int
    oneshot_max_exact: int
    mesh_max: int
    min_bytes: int = 0
    max_bytes: int = NO_MAX

    def __post_init__(self):
        if self.oneshot_max <= 0 or self.oneshot_max_exact <= 0:
            raise ValueError(
                f"oneshot_max ({self.oneshot_max}) and oneshot_max_exact "
                f"({self.oneshot_max_exact}) must be positive"
            )
        if self.mesh_max < self.oneshot_max:
            raise ValueError(
                f"mesh_max ({self.mesh_max}) must be >= oneshot_max "
                f"({self.oneshot_max}); the families partition by size"
            )
        if self.min_bytes < 0:
            raise ValueError(f"min_bytes ({self.min_bytes}) must be non-negative")


FAMILY_POLICY: dict[tuple[str, int], FamilyPolicy] = {
    # --- PCIe: Policy from measurements (on gfx950/MI350P) --------------------
    ("pcie", 2): FamilyPolicy(
        oneshot_max=384 << 10,
        oneshot_max_exact=64 << 20,
        mesh_max=NO_MAX,
        min_bytes=24 << 10,
    ),
    ("pcie", 4): FamilyPolicy(
        oneshot_max=64 << 10, oneshot_max_exact=(160 << 10) - 1, mesh_max=16 << 20
    ),
    ("pcie", 8): FamilyPolicy(
        oneshot_max=16 << 10, oneshot_max_exact=(80 << 10) - 1, mesh_max=24 << 20
    ),
    # --- xGMI: Policy from measurements (on gfx942) --------------------
    ("xgmi", 2): FamilyPolicy(
        oneshot_max=384 << 10,
        oneshot_max_exact=1536 << 10,
        mesh_max=128 << 20,
        min_bytes=56 << 10,
    ),
    ("xgmi", 4): FamilyPolicy(
        oneshot_max=384 << 10,
        oneshot_max_exact=384 << 10,
        mesh_max=128 << 20,
        min_bytes=48 << 10,
    ),
    ("xgmi", 8): FamilyPolicy(
        oneshot_max=256 << 10,
        oneshot_max_exact=168 << 10,
        mesh_max=128 << 20,
        min_bytes=112 << 10,
    ),
}


@dataclass(frozen=True)
class FusedPolicy:
    """Fused all-reduce+RMSNorm boundaries for one ``(link, world_size)``.

    Unlike the plain path -- whose one-shot and quantized families are split
    across two aiter slots -- the fused families all live in ``QuickAllReduce``,
    so this stays a single unified view. ``min_bytes`` is where the fused path
    starts being worth taking; below it it declines.

    * ``oneshot_max`` is where the quantized **mesh/ring** overtakes the one-shot.
    * ``oneshot_max_exact`` is where the **fallback** the caller would otherwise
      use (``cross_device_reduce``/RCCL) overtakes it, since exact mode declines
      rather than quantizing.
    """

    oneshot_max: int
    oneshot_max_exact: int
    mesh_max: int | None
    ring_max: int | None = None
    min_bytes: int = 0

    @property
    def max_bytes(self) -> int | None:
        """Upper bound of the dispatch range, or ``None`` if unbounded.

        ``None`` means at least one active family has no size ceiling.
        Callers that need an integer for range comparisons should treat
        ``None`` as infinity.
        """
        if self.mesh_max is None or self.ring_max is None:
            return None
        return max(self.oneshot_max, self.oneshot_max_exact, self.mesh_max, self.ring_max)

    def __post_init__(self):
        if self.oneshot_max <= 0 or self.oneshot_max_exact <= 0:
            raise ValueError(
                f"oneshot_max ({self.oneshot_max}) and oneshot_max_exact "
                f"({self.oneshot_max_exact}) must be positive"
            )
        if self.mesh_max is not None and self.mesh_max > 0 and self.mesh_max < self.oneshot_max:
            raise ValueError(
                f"mesh_max ({self.mesh_max}) must be >= oneshot_max "
                f"({self.oneshot_max}); the families partition by size"
            )


FUSED_FAMILY_POLICY: dict[tuple[str, int], FusedPolicy] = {
    # --- PCIe: Policy from measurements (on gfx950/MI350P) --------------------
    ("pcie", 2): FusedPolicy(
        oneshot_max=1792 << 10, oneshot_max_exact=32 << 20, mesh_max=1792 << 10, ring_max=None
    ),
    ("pcie", 4): FusedPolicy(
        oneshot_max=256 << 10, oneshot_max_exact=1344 << 10, mesh_max=8 << 20, ring_max=None
    ),
    ("pcie", 8): FusedPolicy(
        oneshot_max=64 << 10, oneshot_max_exact=2 << 20, mesh_max=128 << 20, ring_max=None
    ),
    # --- xGMI: Policy from measurements (on gfx942/MI300X) --------------------
    # No ring algorithm, mesh is always better above oneshot_max.
    ("xgmi", 2): FusedPolicy(
        oneshot_max=2 << 20, oneshot_max_exact=32 << 20, mesh_max=None,
    ),
    ("xgmi", 4): FusedPolicy(
        oneshot_max=1376256, oneshot_max_exact=512 << 10, mesh_max=None,
    ),
    ("xgmi", 8): FusedPolicy(
        oneshot_max=448 << 10, oneshot_max_exact=128 << 10, mesh_max=None,
    ),
}

# --- environment variables ---------------------------------------------------------
#
# TODO: can we re-use the existing AITER_AR_1STAGE* / AITER_AR_QUANT_* variables?

ENABLE_VAR = "AITER_FLY_AR"
ACCURACY_VAR = "AITER_FLY_AR_ACCURACY"
ONESHOT_MAX_VAR = "AITER_FLY_AR_ONESHOT_MAX_BYTES"
ONESHOT_MIN_VAR = "AITER_FLY_AR_ONESHOT_MIN_BYTES"
MESH_MAX_VAR = "AITER_FLY_AR_MESH_MAX_BYTES"
# Fused overrides. Separate from the plain ones because the boundaries differ;
# ENABLE_VAR and ACCURACY_VAR are shared -- one FlyDSL all-reduce family.
FUSED_ONESHOT_MAX_VAR = "AITER_FLY_AR_FUSED_ONESHOT_MAX_BYTES"
FUSED_MESH_MAX_VAR = "AITER_FLY_AR_FUSED_MESH_MAX_BYTES"
FUSED_MIN_VAR = "AITER_FLY_AR_FUSED_MIN_BYTES"
# Comma-separated hidden sizes to build the fused engines for at startup.
# The fused wire layout depends on the width, so a width cannot be built until
# it is known -- and building inside a HIP graph capture is not possible.
# Declaring the model's widths here removes the question.
FUSED_HIDDENS_VAR = "AITER_FLY_AR_FUSED_HIDDENS"
# Whether a hidden dim with no native row geometry may run on a wider workgroup
# with the lanes past the real row masked off.
FUSED_PAD_VAR = "AITER_FLY_AR_FUSED_PAD"

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


@functools.lru_cache(maxsize=1)
def detect_link() -> str:
    """``"xgmi"`` or ``"pcie"`` for this host, probed once per process.

    Host-wide, not per group; see ``has_xgmi_peer_links`` for the uniform-node
    assumption that makes that safe.
    """

    from .quick_allreduce import has_xgmi_peer_links

    return "xgmi" if has_xgmi_peer_links() else "pcie"


# --- plain all-reduce: split views for the two aiter slots -------------------------


def _base(link: str, world_size: int) -> FamilyPolicy:
    if link not in LINKS:
        raise ValueError(f"link must be one of {LINKS}, got {link!r}")
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    return FAMILY_POLICY[(link, int(world_size))]


def _oneshot_boundary(base: FamilyPolicy) -> int:
    """The measured one-shot/mesh crossover, ``ONESHOT_MAX_VAR`` applied."""

    override = _env_int(ONESHOT_MAX_VAR)
    return base.oneshot_max if override is None else override


@dataclass(frozen=True)
class OneShotPolicy:
    """The exact one-shot's window, as the custom-all-reduce slot sees it."""

    max_bytes: int
    min_bytes: int = 0


@dataclass(frozen=True)
class QuantPolicy:
    """The quantized families' window, as the quick-reduce slot sees it.

    ``floor`` is **exclusive**: dispatch only when ``nbytes > floor``. At or
    below it the exact one-shot is faster, and declining is what lets the
    payload reach the custom-all-reduce slot that hosts it.
    """

    floor: int
    mesh_max: int
    max_bytes: int


def resolve_oneshot(link: str, world_size: int) -> OneShotPolicy:
    """The one-shot's window for a rank, environment overrides applied.

    ``ONESHOT_MIN_VAR`` overrides the small-payload floor below which the
    one-shot declines (the custom-AR slot then falls through to ``cdr``);
    ``ONESHOT_MAX_VAR`` overrides the ceiling. ``-1`` on either means "use the
    table".
    """

    base = _base(link, world_size)
    max_override = _env_int(ONESHOT_MAX_VAR)
    min_override = _env_int(ONESHOT_MIN_VAR)
    return OneShotPolicy(
        max_bytes=base.oneshot_max_exact if max_override is None else max_override,
        min_bytes=base.min_bytes if min_override is None else min_override,
    )


def resolve_quant(link: str, world_size: int) -> QuantPolicy:
    """The mesh/ring window for a rank, environment overrides applied."""

    base = _base(link, world_size)
    floor = _oneshot_boundary(base)
    mesh = base.mesh_max
    override_mesh = _env_int(MESH_MAX_VAR)
    if override_mesh is not None:
        mesh = override_mesh
    # The families partition by size; an override must not invert them.
    mesh = max(mesh, floor)
    return QuantPolicy(floor=floor, mesh_max=mesh, max_bytes=base.max_bytes)


def pick_quant_family(nbytes: int, policy: QuantPolicy) -> str:
    """``"mesh"`` | ``"ring"`` for a payload of *nbytes*.

    Assumes ``nbytes > policy.floor``; below that the caller should have
    declined so the exact one-shot gets the payload.
    """
    return "mesh" if nbytes <= policy.mesh_max else "ring"


def quant_family_range(family: str, policy: QuantPolicy) -> tuple[int, int]:
    """Payload bytes (inclusive) ``pick_quant_family`` sends to *family*."""
    if family == "mesh":
        return policy.floor + 1, min(policy.mesh_max, policy.max_bytes)
    if family == "ring":
        # The ring algorithm is beneficial for large messages, i.e.,
        # it comes after the mesh with increasing message size.
        return policy.mesh_max + 1, policy.max_bytes
    raise ValueError(f"family must be 'mesh' or 'ring', got {family!r}")


def quant_families_reachable(policy: QuantPolicy) -> tuple[str, ...]:
    """Quantized families a *policy* can ever select, in size order."""
    out = []
    if policy.mesh_max > policy.floor:
        out.append("mesh")
    if policy.max_bytes > policy.mesh_max:
        out.append("ring")
    return tuple(out)


# --- fused all-reduce+RMSNorm: one unified view (all families in one slot) ---------


def fused_hiddens(extra: tuple[int, ...] = ()) -> tuple[int, ...]:
    """Widths to build the fused engines for up front, sorted and deduped.

    ``AITER_FLY_AR_FUSED_HIDDENS`` plus whatever the caller passed. Sorted so
    every rank builds in the same order -- each build is a collective.
    """
    out = {int(h) for h in extra if int(h) > 0}
    raw = os.environ.get(FUSED_HIDDENS_VAR, "")
    for tok in raw.replace(" ", "").split(","):
        if not tok:
            continue
        try:
            val = int(tok)
        except ValueError:
            logger.warning(
                "FlyDSL QR: ignoring %r in %s, expected an integer",
                tok,
                FUSED_HIDDENS_VAR,
            )
            continue
        if val > 0:
            out.add(val)
    return tuple(sorted(out))


def fused_pad_enabled() -> bool:
    """Whether padded fused builds are allowed. On unless ``…_PAD`` is ``"0"``."""
    return os.environ.get(FUSED_PAD_VAR, "").strip() != "0"


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


def _base_fused(link: str, world_size: int) -> FusedPolicy:
    if link not in LINKS:
        raise ValueError(f"link must be one of {LINKS}, got {link!r}")
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    return FUSED_FAMILY_POLICY[(link, int(world_size))]


def resolve_fused(link: str, world_size: int, mode: str | None = None) -> FusedPolicy:
    """The fused policy in force for a rank, environment overrides applied.

    *mode* defaults to ``accuracy_mode()`` and picks between two policies:

    * ``"fast"`` -- the full three-family policy. One-shot up to
      ``oneshot_max``, mesh/ring (quantized) beyond it.
    * ``"exact"`` (default) -- **only** the one-shot is ever reachable, at its
      widened ``oneshot_max_exact`` ceiling. Above that the fused path declines
      and the caller falls through rather than silently quantizing.

    The extra boundary versus the plain path is ``min_bytes``: below it the
    fused path declines.
    """
    base = _base_fused(link, world_size)
    mode = accuracy_mode() if mode is None else mode
    if mode not in ACCURACY_MODES:
        raise ValueError(f"mode must be one of {ACCURACY_MODES}, got {mode!r}")

    override_one = _env_int(FUSED_ONESHOT_MAX_VAR)

    if mode == "exact":
        one = base.oneshot_max_exact if override_one is None else override_one
        if _env_int(FUSED_MESH_MAX_VAR) is not None:
            logger.warning(
                "FlyDSL QR: ignoring %s in accuracy=exact mode -- exact mode "
                "has no mesh/ring window to widen. Set %s=fast to use it.",
                FUSED_MESH_MAX_VAR,
                ACCURACY_VAR,
            )
        # mesh_max=0 collapses the mesh window to zero; ring_max=0 means no ring
        # either. Only one-shot is reachable.
        policy = FusedPolicy(
            oneshot_max=one,
            oneshot_max_exact=one,
            mesh_max=0,
            ring_max=0,
            min_bytes=base.min_bytes,
        )
    else:
        one = base.oneshot_max if override_one is None else override_one
        mesh = base.mesh_max
        override_mesh = _env_int(FUSED_MESH_MAX_VAR)
        if override_mesh is not None:
            mesh = override_mesh
        if mesh is not None:
            mesh = max(mesh, one)
        policy = FusedPolicy(
            oneshot_max=one,
            oneshot_max_exact=one,
            mesh_max=mesh,
            ring_max=base.ring_max,
            min_bytes=base.min_bytes,
        )

    floor = _env_int(FUSED_MIN_VAR)
    if floor is None:
        return policy
    return FusedPolicy(
        oneshot_max=policy.oneshot_max,
        oneshot_max_exact=policy.oneshot_max_exact,
        mesh_max=policy.mesh_max,
        ring_max=policy.ring_max,
        min_bytes=floor,
    )


def pick_fused_family(nbytes: int, policy: FusedPolicy) -> str:
    """``"oneshot"`` | ``"mesh"`` | ``"ring"`` for a payload of *nbytes*."""
    if nbytes <= policy.oneshot_max:
        return "oneshot"
    if policy.mesh_max is None or nbytes <= policy.mesh_max:
        return "mesh"
    return "ring"


def fused_family_range(family: str, policy: FusedPolicy) -> tuple[int, int]:
    """Payload bytes (inclusive) the fused dispatcher sends to *family*.

    ``pick_fused_family`` clipped to ``min_bytes`` below and ``max_bytes``
    above; an unbounded policy is capped at the 4 GiB buffer window every
    FlyDSL schedule enforces.
    """
    top = 0xFFFFFFFF if policy.max_bytes is None else policy.max_bytes
    if family == "oneshot":
        lo, hi = 0, policy.oneshot_max
    elif family == "mesh":
        lo = policy.oneshot_max + 1
        hi = top if policy.mesh_max is None else policy.mesh_max
    elif family == "ring":
        if policy.mesh_max is None:
            # An unbounded mesh takes everything above the one-shot.
            return 1, 0
        lo, hi = max(policy.oneshot_max, policy.mesh_max) + 1, top
    else:
        raise ValueError(f"family must be 'oneshot', 'mesh' or 'ring', got {family!r}")
    return max(lo, policy.min_bytes), min(hi, top)


def fused_families_reachable(policy: FusedPolicy) -> tuple[str, ...]:
    """Fused families a *policy* can ever select, in size order."""
    out = []
    if policy.oneshot_max >= policy.min_bytes:
        out.append("oneshot")
    # Mesh is reachable when its window is non-empty: either mesh_max is None
    # (unbounded) or mesh_max > oneshot_max. Also needs to be reachable above
    # min_bytes -- but if oneshot_max >= min_bytes that is already guaranteed.
    if policy.mesh_max is None or policy.mesh_max > policy.oneshot_max:
        out.append("mesh")
    if policy.mesh_max is not None:
        ring_floor = max(policy.oneshot_max, policy.mesh_max)
        if policy.ring_max is None or policy.ring_max > ring_floor:
            out.append("ring")
    return tuple(out)
