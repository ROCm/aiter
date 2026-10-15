# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""Stable scheduling classes for dynamic expert-major SonicMoE routes.

The local route count is data dependent under expert parallelism.  It is a
valid runtime extent, but it is a poor compiler or launch-topology key: nearby
ranks can otherwise select different kernels immediately before a collective.
This module deliberately separates the *policy size* used for coarse tuning
from every allocation bound and from the exact runtime route count.

Callers which coordinate expert-parallel ranks may pass the same
``route_policy_size`` on every rank (normally a conservative nominal per-rank
route count).  The hint is a policy floor rather than an unsafe upper bound:
an unexpectedly larger local shard is promoted automatically.  Without a
hint we place the local count in one of four broad capacity classes.  The
final class is open ended, so arbitrarily large legal route counts do not
create further policy or JIT variants.
"""

from __future__ import annotations

from enum import IntEnum


class E16RoutePolicy(IntEnum):
    """Finite performance-policy classes for the Qwen3 E16 route path."""

    SMALL = 0
    MEDIUM = 1
    LARGE = 2
    XLARGE = 3


_SMALL_MAX_ROUTES = 4 * 1024
_MEDIUM_MAX_ROUTES = 16 * 1024
_LARGE_MAX_ROUTES = 32 * 1024

# Representative work bounds used only for choosing finite tile, cache, and
# algorithm families.  They are intentionally not allocation or runtime-grid
# bounds and need not be greater than the current local route count.  In
# particular, XLARGE is saturated so R>64K continues to reuse exactly the same
# compiled family.
_POLICY_ROUTE_REPRESENTATIVES = {
    E16RoutePolicy.SMALL: 4 * 1024,
    E16RoutePolicy.MEDIUM: 16 * 1024,
    E16RoutePolicy.LARGE: 32 * 1024,
    E16RoutePolicy.XLARGE: 64 * 1024,
}

E16_ROUTE_POLICY_SIZES = tuple(_POLICY_ROUTE_REPRESENTATIVES.values())


def select_e16_route_policy(
    routes: int,
    route_policy_size: int | None = None,
) -> E16RoutePolicy:
    """Map a dynamic route count or a rank-shared hint to a finite class.

    ``route_policy_size`` is a performance-policy floor, not a storage-capacity
    promise.  The actual ``routes`` value remains authoritative for bounds and
    allocations and promotes the policy when it exceeds the shared hint.  This
    lets ordinary imbalanced EP ranks use one shared policy while unexpected
    outliers remain on an appropriately sized, still finite kernel family.
    """

    if isinstance(routes, bool) or not isinstance(routes, int):
        raise TypeError("routes must be an integer")
    if routes < 0:
        raise ValueError(f"routes must be non-negative, got {routes}")
    if route_policy_size is None:
        policy_size = routes
    else:
        if isinstance(route_policy_size, bool) or not isinstance(
            route_policy_size, int
        ):
            raise TypeError("route_policy_size must be None or an integer")
        if route_policy_size <= 0:
            raise ValueError(
                "route_policy_size must be positive when supplied, got "
                f"{route_policy_size}"
            )
        policy_size = max(routes, route_policy_size)

    if policy_size <= _SMALL_MAX_ROUTES:
        return E16RoutePolicy.SMALL
    if policy_size <= _MEDIUM_MAX_ROUTES:
        return E16RoutePolicy.MEDIUM
    if policy_size <= _LARGE_MAX_ROUTES:
        return E16RoutePolicy.LARGE
    return E16RoutePolicy.XLARGE


def e16_route_policy_representative(policy: E16RoutePolicy | int) -> int:
    """Return the bounded geometry representative for ``policy``."""

    try:
        normalized = E16RoutePolicy(policy)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid E16 route policy {policy!r}") from error
    return _POLICY_ROUTE_REPRESENTATIVES[normalized]


def canonical_e16_route_policy_size(shared_max_routes: int) -> int:
    """Coarsen an EP-wide maximum route count to one finite policy size.

    Callers should compute ``shared_max_routes`` from metadata already known
    for the current all-to-all (or from a safe static upper bound), then pass
    the returned value to every rank's forward call.  Exact route counts never
    become compiler keys; values above 64K deliberately remain in the open
    ended XLARGE family.
    """

    return e16_route_policy_representative(select_e16_route_policy(shared_max_routes))


__all__ = [
    "E16_ROUTE_POLICY_SIZES",
    "E16RoutePolicy",
    "canonical_e16_route_policy_size",
    "e16_route_policy_representative",
    "select_e16_route_policy",
]
