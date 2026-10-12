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

They do not share a dispatcher. Each lives in the aiter slot whose accuracy
contract it already matches, and this module hands each slot its own view of
one shared table row:

* ``resolve_oneshot`` -> ``CustomAllreduce``, which is exact.
* ``resolve_quant``   -> ``QuickAllReduce``, which is allowed to quantize.
"""

from __future__ import annotations

import functools
import logging
import os

from .kernels.comm.allreduce_shared import (
    NO_MAX,
    FamilyPolicy,
    OneShotPolicy,
    QuantPolicy,
)

logger = logging.getLogger("aiter")

# Fabric between GPUs.
LINKS = ("pcie", "xgmi")

SUPPORTED_WORLDS = (2, 4, 8)

FAMILY_POLICY: dict[tuple[str, int], FamilyPolicy] = {
    # --- PCIe: Policy from measurements (on gfx950/MI350P) --------
    ("pcie", 2): FamilyPolicy(
        oneshot_max=384 << 10,
        oneshot_max_exact=64 << 20,
        mesh_max=NO_MAX,
    ),
    ("pcie", 4): FamilyPolicy(
        oneshot_max=128 << 10, oneshot_max_exact=1 << 20, mesh_max=16 << 20
    ),
    ("pcie", 8): FamilyPolicy(
        oneshot_max=96 << 10, oneshot_max_exact=1 << 20, mesh_max=24 << 20
    ),
    # --- xGMI: Policy from measurements (on gfx942) --------------------
    ("xgmi", 2): FamilyPolicy(
        oneshot_max=384 << 10,
        oneshot_max_exact=1536 << 10,
        mesh_max=128 << 20,
    ),
    ("xgmi", 4): FamilyPolicy(
        oneshot_max=384 << 10,
        oneshot_max_exact=384 << 10,
        mesh_max=128 << 20,
    ),
    ("xgmi", 8): FamilyPolicy(
        oneshot_max=256 << 10,
        oneshot_max_exact=168 << 10,
        mesh_max=128 << 20,
    ),
}

# Rows measured on one arch, keyed ``(arch, link, world_size)``. On that arch
# they replace the ``FAMILY_POLICY`` row for the same ``(link, world_size)``.
#
# gfx950/xGMI (MI355X), measured with the ``bench`` scenario of the FlyDSL
# all-reduce op tests:
#
# * The ring never wins: it is 1.03x-1.16x slower than the mesh at TP2 up to
#   512 MiB, and 2.2x-5.7x slower at TP4 and TP8. ``mesh_max`` is unbounded.
# * ``oneshot_max_exact`` is where ``cross_device_reduce`` overtakes the
#   one-shot. At TP4 and TP8 that is where it switches to its 2-stage kernel
#   (160 KiB and 80 KiB), which is fast on this fabric.
# * ``oneshot_max`` is where the mesh overtakes the fastest exact kernel. At
#   TP4 and TP8 that kernel is ``cross_device_reduce``'s 2-stage, not the
#   one-shot, so the mesh starts well above the one-shot ceiling and the payloads
#   between the two stay exact.
ARCH_FAMILY_POLICY: dict[tuple[str, str, int], FamilyPolicy] = {
    ("gfx950", "xgmi", 2): FamilyPolicy(
        oneshot_max=448 << 10,
        oneshot_max_exact=3 << 20,
        mesh_max=NO_MAX,
    ),
    ("gfx950", "xgmi", 4): FamilyPolicy(
        oneshot_max=960 << 10,
        # Every payload below the 160 KiB 2-stage switch: payloads are whole
        # multiples of 16 B.
        oneshot_max_exact=(160 << 10) - 16,
        mesh_max=NO_MAX,
    ),
    ("gfx950", "xgmi", 8): FamilyPolicy(
        oneshot_max=1600 << 10,
        oneshot_max_exact=96 << 10,
        mesh_max=NO_MAX,
    ),
}


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
    return os.environ.get("AITER_FLY_AR", "").strip() == "1"


@functools.lru_cache(maxsize=1)
def detect_arch() -> str | None:
    """This host's ``gfx`` arch, or None if it cannot be read."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    try:
        return get_gfx_runtime()
    except (KeyError, RuntimeError):
        return None


@functools.lru_cache(maxsize=1)
def detect_link() -> str:
    """``"xgmi"`` or ``"pcie"`` for this host, probed once per process.

    Host-wide, not per group; see ``has_xgmi_peer_links`` for the uniform-node
    assumption that makes that safe.
    """

    from .quick_allreduce_int4 import has_xgmi_peer_links

    return "xgmi" if has_xgmi_peer_links() else "pcie"


def family_policy(link: str, world_size: int, arch: str | None = None) -> FamilyPolicy:
    """The table row for *(arch, link, world_size)*, without overrides.

    *arch* defaults to this host's. An arch with no row of its own for the
    cell gets the ``FAMILY_POLICY`` row.
    """
    if link not in LINKS:
        raise ValueError(f"link must be one of {LINKS}, got {link!r}")
    if world_size not in SUPPORTED_WORLDS:
        raise ValueError(
            f"world_size must be one of {SUPPORTED_WORLDS}, got {world_size}"
        )
    if arch is None:
        arch = detect_arch()
    row = ARCH_FAMILY_POLICY.get((arch, link, int(world_size)))
    return FAMILY_POLICY[(link, int(world_size))] if row is None else row


def _oneshot_boundary(base: FamilyPolicy) -> int:
    """The measured one-shot/mesh crossover, ``AITER_FLY_AR_ONESHOT_MAX_BYTES``
    applied."""

    override = _env_int("AITER_FLY_AR_ONESHOT_MAX_BYTES")
    return base.oneshot_max if override is None else override


def resolve_oneshot(
    link: str, world_size: int, arch: str | None = None
) -> OneShotPolicy:
    """The one-shot's window for a rank, environment overrides applied.

    *arch* defaults to this host's; refer to ``family_policy``.

    ``AITER_FLY_AR_ONESHOT_MIN_BYTES`` overrides the small-payload floor below
    which the one-shot declines (the custom-AR slot then falls through to
    ``cdr``); ``AITER_FLY_AR_ONESHOT_MAX_BYTES`` overrides the ceiling. ``-1``
    on either means "use the table".
    """

    base = family_policy(link, world_size, arch)
    max_override = _env_int("AITER_FLY_AR_ONESHOT_MAX_BYTES")
    min_override = _env_int("AITER_FLY_AR_ONESHOT_MIN_BYTES")
    return OneShotPolicy(
        max_bytes=base.oneshot_max_exact if max_override is None else max_override,
        min_bytes=base.min_bytes if min_override is None else min_override,
    )


def resolve_quant(link: str, world_size: int, arch: str | None = None) -> QuantPolicy:
    """The mesh/ring window for a rank, environment overrides applied.

    *arch* defaults to this host's; refer to ``family_policy``.
    """

    base = family_policy(link, world_size, arch)
    floor = _oneshot_boundary(base)
    mesh = base.mesh_max
    override_mesh = _env_int("AITER_FLY_AR_MESH_MAX_BYTES")
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
