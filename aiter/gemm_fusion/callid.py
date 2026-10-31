# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""call_id of a fused launch and its static uniqueness check (R12 design card r2, D2.2 and D-8(d)).

The tag a producer stores and a consumer expects is E = (STEP << 32) | call_id, so two DIFFERENT launches that can meet in one handshake must carry different call_ids.
call_id = ``decision_key(site, tokens, vid, n_fused)``: the process-independent, rank-uniform 31-bit key ATOM already uses for its rank consensus
(atom.model_ops.gemm_fusion.consensus.decision_key; this copy is pinned to it by test vectors). Because the key is a 31-bit hash, uniqueness is not automatic: all
registered keys of one profile must map to DISTINCT ids, checked statically (`check_call_ids` / `check_profile_call_ids`) before any launch; on a collision the colliding
sites must run natively (refuse fusion) and the collision is logged.

R16 / large r13: the tag is E = (STEP << 32) | ((m_real - 1) << 19) | (call_id & 0x7FFFF): only the LOW 19 bits of call_id reach the tag, next to the real rows. Two launches that can
meet in one handshake share STEP and the token count (one forward pass), so for r13 the ids must be distinct on those 19 bits WITHIN one token count (``tag_bits=19``: grouped by
tokens; keys of different token counts differ in the m_real field of the tag). A zero low field is refused as well (m_real = 1 at STEP 0 would give the expected tag 0).

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
    # R16 (tag_bits): low-bits value -> the >= 2 distinct keys of ONE token count that share it; and the keys whose low field is 0
    low_collisions: dict[tuple[int, int], list[CallKey]] = field(
        default_factory=dict
    )  # (tokens, call_id & mask) -> keys
    zero_low: list[CallKey] = field(default_factory=list)
    tag_bits: int | None = None
    unresolved: list[str] = field(
        default_factory=list
    )  # profile entries the check could not expand (unknown variant)

    @property
    def refused_keys(self) -> frozenset[CallKey]:
        out = {k for ks in self.collisions.values() for k in ks}
        out.update(self.zero_id)
        out.update(k for ks in self.low_collisions.values() for k in ks)
        out.update(self.zero_low)
        return frozenset(out)

    @property
    def refused_sites(self) -> frozenset[str]:
        return frozenset(k[0] for k in self.refused_keys)

    @property
    def ok(self) -> bool:
        return (
            not self.collisions
            and not self.zero_id
            and not self.low_collisions
            and not self.zero_low
        )


def check_call_ids(
    keys: Iterable[CallKey],
    key_fn: KeyFn = decision_key,
    tag_bits: int | None = None,
    tag_vids: Iterable[str] | None = None,
    per_tokens: bool = False,
    tag_bits_by_token: dict[int, int] | Callable[[int], int | None] | None = None,
) -> CallIdReport:
    """Static uniqueness check over a set of keys (duplicates of the SAME key are one key, never a collision).
    per_tokens (R16): only keys of ONE token count can meet in a handshake (one forward pass), so require distinct ids within each token count instead of across the whole set (over the
    pad-M universe of every real m, global 31-bit uniqueness is impossible: ~480k keys collide ~50 times by the birthday bound, and was never needed).
    tag_bits (R16, large r13: 19): ALSO require the low tag_bits of the ids of the keys whose variant is in tag_vids (default: all) to be distinct per token count and non-zero: the r13
    tag carries only those bits of call_id next to the real rows.
    tag_bits_by_token (ROUTE A): the tag width is per TOKEN COUNT instead (r13/r18 19 low bits, r19 18: a profile may carry both families, one family per token count); a dict or a
    tokens -> width callable; None skips that token count."""
    distinct = sorted({tuple(k) for k in keys})
    by_id: dict[int, list[CallKey]] = {}
    for k in distinct:
        by_id.setdefault(key_fn(*k), []).append(k)
    rep = CallIdReport(keys=len(distinct), tag_bits=tag_bits)
    for cid, ks in sorted(by_id.items()):
        if per_tokens:
            by_tok: dict[int, list[CallKey]] = {}
            for k in ks:
                by_tok.setdefault(k[1], []).append(k)
            ks = [k for g in by_tok.values() if len(g) > 1 for k in g]
        if len(ks) > 1:
            rep.collisions[cid] = ks
    rep.zero_id = list(by_id.get(0, []))
    if tag_bits is not None or tag_bits_by_token is not None:
        vids = None if tag_vids is None else set(tag_vids)

        def _bits(tokens: int) -> int | None:
            if tag_bits_by_token is None:
                return tag_bits
            if callable(tag_bits_by_token):
                return tag_bits_by_token(tokens)
            return tag_bits_by_token.get(tokens, tag_bits)

        by_low: dict[tuple[int, int], list[CallKey]] = {}
        for cid, ks in by_id.items():
            for k in ks:
                if vids is None or k[2] in vids:
                    bits = _bits(k[1])
                    if bits is None:
                        continue
                    mask = (1 << bits) - 1
                    by_low.setdefault((k[1], cid & mask), []).append(k)
        for (tokens, low), ks in sorted(by_low.items()):
            if low == 0:
                rep.zero_low.extend(ks)
            if len(ks) > 1:
                rep.low_collisions[(tokens, low)] = ks
    return rep


def tag_unique_ids(
    keys: Iterable[CallKey],
    key_fn: KeyFn = decision_key,
    tag_bits: int = 19,
    tag_bits_by_token: dict[int, int] | Callable[[int], int | None] | None = None,
) -> dict[CallKey, int]:
    """call_ids whose LOW tag_bits are distinct and non-zero within every token count BY CONSTRUCTION (R16, large r13): low = 1 + the ordinal of (site, vid, n_fused) in the sorted
    set of distinct triples, the high bits from key_fn (so the id stays a 31-bit key with the hash's entropy above the tag field). Rank-uniform (a pure function of the key set,
    which every rank builds identically). Raises if there are not enough ordinals for the sites.
    """
    distinct = sorted({tuple(k) for k in keys})
    triples = sorted({(k[0], k[2], k[3]) for k in distinct})

    def _bits(tokens: int) -> int:
        if tag_bits_by_token is None:
            return tag_bits
        if callable(tag_bits_by_token):
            got = tag_bits_by_token(tokens)
            return tag_bits if got is None else got
        return tag_bits_by_token.get(tokens, tag_bits)

    narrowest = min({_bits(k[1]) for k in distinct} or {tag_bits})
    if len(triples) >= (1 << narrowest):
        raise ValueError(
            f"{len(triples)} (site, vid, n_fused) triples do not fit {narrowest} tag bits"
        )
    ordinal = {t: i + 1 for i, t in enumerate(triples)}
    out = {}
    for k in distinct:
        mask = (1 << _bits(k[1])) - 1
        out[k] = ((key_fn(*k) & ~mask) & 0x7FFFFFFF) | ordinal[(k[0], k[2], k[3])]
    return out


def profile_call_id_keys(
    profile, sites: Iterable[str], world: int | None = None, pad_m: bool = False
) -> tuple[list[CallKey], list[str]]:
    """Every (site, tokens, vid, n_fused) the profile can dispatch for these sites: each exact row (tokens = m * world) and every m an m-range rule covers.
    n_fused = the stage count of the row's variant. -> (keys, unresolved): entries whose variant could not be loaded are reported, not guessed.
    pad_m (R16): ALSO every REAL local m a prepared bucket can serve (match.bucket_in_band over every m of the band: the key carries the real token count, the bucket's variant and its
    stage count), e.g. every m 1025..1280 for the r13 bucket 1280.
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
    if pad_m:
        from .match import bucket_in_band

        bands = {}
        for r in profile.rows():
            if world is None or int(r.world) == int(world):
                bands.setdefault((r.sig, r.arch, r.cu, r.world), []).append(r.m)
        for (sig, arch, cu, wd), ms_ in bands.items():
            for m in range(1, max(ms_) + 1):
                got = bucket_in_band(profile, sig, arch, cu, wd, m)
                if got is None:
                    continue
                vid = got[1]
                info = catalog._CATALOG.get(vid)
                if info is not None:
                    entries[(vid, m * int(wd), len(info.stage_kinds))] = None
    keys = [(s, tokens, vid, n) for s in sites for (vid, tokens, n) in entries]
    return keys, sorted(set(unresolved))


def check_profile_call_ids(
    profile,
    sites: Iterable[str],
    world: int | None = None,
    key_fn: KeyFn = decision_key,
    tag_bits: int | None = None,
    pad_m: bool = False,
    tag_vids: Iterable[str] | None = None,
    per_tokens: bool | None = None,
    tag_bits_by_token: dict[int, int] | Callable[[int], int | None] | None = None,
) -> CallIdReport:
    """The KEY gate: all keys of one profile x the model's fused sites map to distinct non-zero call_ids. Logs every collision. R16: tag_bits=19 also checks the low bits per token
    count (large r13 tags); pad_m=True covers every real m a bucket can serve."""
    keys, unresolved = profile_call_id_keys(profile, sites, world, pad_m=pad_m)
    rep = check_call_ids(
        keys,
        key_fn,
        tag_bits=tag_bits,
        tag_vids=tag_vids,
        per_tokens=(
            pad_m if per_tokens is None else per_tokens
        ),  # the pad-M universe is checked per token count
        tag_bits_by_token=tag_bits_by_token,
    )
    rep.unresolved = unresolved
    for (tokens, low), ks in rep.low_collisions.items():
        bits = tag_bits
        if bits is None and tag_bits_by_token is not None:
            bits = (
                tag_bits_by_token(tokens)
                if callable(tag_bits_by_token)
                else tag_bits_by_token.get(tokens)
            )
        log.error(
            "event=call_id low_collision tokens=%d low%d=%d keys=%s: fusion refused for sites %s",
            tokens,
            bits or 0,
            low,
            ks,
            sorted({k[0] for k in ks}),
        )
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

    def __init__(
        self,
        key_fn: KeyFn = decision_key,
        tag_bits: int | None = None,
        tag_vids: Iterable[str] | None = None,
        per_tokens: bool = False,
    ):
        self._key_fn = key_fn
        self._tag_bits = tag_bits  # R16 (large r13): also check the low tag bits per token count (for the variants in tag_vids)
        self._tag_vids = None if tag_vids is None else tuple(tag_vids)
        self._per_tokens = per_tokens
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
            self.report = check_call_ids(
                self._keys,
                self._key_fn,
                tag_bits=self._tag_bits,
                tag_vids=self._tag_vids,
                per_tokens=self._per_tokens,
            )
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
