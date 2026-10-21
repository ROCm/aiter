# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""call_id of a fused launch and its static uniqueness check (R12 design card r2, D2.2 and D-8(d)).

The tag a producer stores and a consumer expects is E = (STEP << 32) | call_id, so two DIFFERENT launches that can meet in one handshake must carry different call_ids.
call_id = ``decision_key(site, tokens, vid, n_fused)``: the process-independent, rank-uniform 31-bit key ATOM already uses for its rank consensus
(atom.model_ops.gemm_fusion.consensus.decision_key; this copy is pinned to it by test vectors). Because the key is a 31-bit hash, uniqueness is not automatic: all
registered keys of one profile must map to DISTINCT ids, checked statically (`check_call_ids` / `check_profile_call_ids`) before any launch; on a collision the colliding
sites must run natively (refuse fusion) and the collision is logged.

call_id 0 is also refused: a zero call_id at STEP 0 makes the expected tag 0, which equals the content of a freshly zeroed window (a peer that never wrote its tag would
"match").
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

log = logging.getLogger("aiter.gemm_fusion")

CallKey = tuple[str, int, str, int]  # (site, tokens, vid, n_fused)
KeyFn = Callable[[str, int, str, int], int]


def decision_key(site: str, tokens: int, vid: str, n_fused: int) -> int:
    """Process-independent 31-bit key (python ``hash`` is salted per process). Identical to ATOM's consensus.decision_key."""
    blob = f"{site}|{tokens}|{vid}|{n_fused}".encode()
    return (
        int.from_bytes(hashlib.blake2b(blob, digest_size=4).digest(), "little")
        & 0x7FFFFFFF
    )


@dataclass
class CallIdReport:
    keys: int = 0  # distinct (site, tokens, vid, n_fused) keys examined
    collisions: dict[int, list[CallKey]] = field(
        default_factory=dict
    )  # call_id -> the >= 2 distinct keys that share it
    zero_id: list[CallKey] = field(default_factory=list)  # keys whose call_id is 0
    unresolved: list[str] = field(
        default_factory=list
    )  # profile entries the check could not expand (unknown variant)

    @property
    def refused_keys(self) -> frozenset[CallKey]:
        out = {k for ks in self.collisions.values() for k in ks}
        out.update(self.zero_id)
        return frozenset(out)

    @property
    def refused_sites(self) -> frozenset[str]:
        return frozenset(k[0] for k in self.refused_keys)

    @property
    def ok(self) -> bool:
        return not self.collisions and not self.zero_id


def check_call_ids(
    keys: Iterable[CallKey], key_fn: KeyFn = decision_key
) -> CallIdReport:
    """Static uniqueness check over a set of keys (duplicates of the SAME key are one key, never a collision)."""
    distinct = sorted({tuple(k) for k in keys})
    by_id: dict[int, list[CallKey]] = {}
    for k in distinct:
        by_id.setdefault(key_fn(*k), []).append(k)
    rep = CallIdReport(keys=len(distinct))
    for cid, ks in sorted(by_id.items()):
        if len(ks) > 1:
            rep.collisions[cid] = ks
    rep.zero_id = list(by_id.get(0, []))
    return rep


def profile_call_id_keys(
    profile, sites: Iterable[str], world: int | None = None
) -> tuple[list[CallKey], list[str]]:
    """Every (site, tokens, vid, n_fused) the profile can dispatch for these sites: each exact row (tokens = m * world) and every m an m-range rule covers.
    n_fused = the stage count of the row's variant. -> (keys, unresolved): entries whose variant could not be loaded are reported, not guessed.
    """
    from . import catalog
    from .match import _load_builtin

    sites = list(sites)
    entries: dict[tuple[str, int, int], None] = (
        {}
    )  # (vid, tokens, n_fused), insertion ordered, deduped
    unresolved: list[str] = []
    for e in profile.entries():
        if world is not None and int(e.world) != int(world):
            continue
        if e.vid not in catalog._CATALOG:
            _load_builtin(e.vid)
        info = catalog._CATALOG.get(e.vid)
        if info is None:
            unresolved.append(f"{e.vid}: variant not registered")
            continue
        n_fused = len(info.stage_kinds)
        ms = (
            [e.m]
            if hasattr(e, "m")
            else range(
                -(-e.m_min // e.m_multiple) * e.m_multiple, e.m_max + 1, e.m_multiple
            )
        )
        for m in ms:
            entries[(e.vid, int(m) * int(e.world), n_fused)] = None
    keys = [(s, tokens, vid, n) for s in sites for (vid, tokens, n) in entries]
    return keys, sorted(set(unresolved))


def check_profile_call_ids(
    profile,
    sites: Iterable[str],
    world: int | None = None,
    key_fn: KeyFn = decision_key,
) -> CallIdReport:
    """The KEY gate: all keys of one profile x the model's fused sites map to distinct non-zero call_ids. Logs every collision."""
    keys, unresolved = profile_call_id_keys(profile, sites, world)
    rep = check_call_ids(keys, key_fn)
    rep.unresolved = unresolved
    for cid, ks in rep.collisions.items():
        log.error(
            "event=call_id collision call_id=%d keys=%s: fusion refused for sites %s",
            cid,
            ks,
            sorted({k[0] for k in ks}),
        )
    for k in rep.zero_id:
        log.error(
            "event=call_id zero call_id key=%s: fusion refused for site %s", k, k[0]
        )
    return rep


class CallIdRegistry:
    """Collects the keys of every site x bucket the engine will register, then freezes them: `freeze()` runs the static check once; afterwards `call_id(key)` returns the
    id of an accepted key and raises for a refused one (the caller then runs that site natively, on every rank: the keys and the check are rank-uniform).
    """

    def __init__(self, key_fn: KeyFn = decision_key):
        self._key_fn = key_fn
        self._keys: set[CallKey] = set()
        self.report: CallIdReport | None = None

    def add(self, site: str, tokens: int, vid: str, n_fused: int) -> None:
        if self.report is not None:
            raise RuntimeError(
                "CallIdRegistry is frozen: register every site before freeze()"
            )
        self._keys.add((site, int(tokens), vid, int(n_fused)))

    def freeze(self) -> CallIdReport:
        if self.report is None:
            self.report = check_call_ids(self._keys, self._key_fn)
            for cid, ks in self.report.collisions.items():
                log.error(
                    "event=call_id collision call_id=%d keys=%s: fusion refused for sites %s",
                    cid,
                    ks,
                    sorted({k[0] for k in ks}),
                )
            for k in self.report.zero_id:
                log.error(
                    "event=call_id zero call_id key=%s: fusion refused for site %s",
                    k,
                    k[0],
                )
        return self.report

    def is_refused(self, site: str, tokens: int, vid: str, n_fused: int) -> bool:
        rep = self.freeze()
        return (site, int(tokens), vid, int(n_fused)) in rep.refused_keys

    def call_id(self, site: str, tokens: int, vid: str, n_fused: int) -> int:
        rep = self.freeze()
        key = (site, int(tokens), vid, int(n_fused))
        if key not in self._keys:
            raise KeyError(f"call_id key {key} was never registered")
        if key in rep.refused_keys:
            raise ValueError(
                f"call_id key {key} is refused (collision or zero id): run site {site!r} natively"
            )
        return self._key_fn(*key)
