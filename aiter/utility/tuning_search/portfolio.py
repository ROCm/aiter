# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A portfolio of strategies under a bandit, with warm starts.

No single strategy wins everywhere: differential evolution is robust on
rugged spaces, TPE is sample-efficient on smooth ones and can use a
selector's prior, a hill climber finishes a good start quickly. The
portfolio runs several and lets a discounted Thompson-sampling bandit give
each ask to the member most likely to improve the incumbent right now:

- reward 1 when a member's fresh full-fidelity observation beats the best
  seen so far by more than ``min_gain`` (relative), else 0; cache hits and
  low-fidelity observations do not move the bandit;
- every arm's evidence decays by ``discount`` per rewarded observation
  (idle arms included, so an arm that was productive long ago does not
  keep a stale, confident posterior), and allocation follows whichever
  member is productive *now*;
- every member is asked at least once before the bandit decides;
- every observation goes to every member, so a member immediately exploits
  another's find (DE immigrates it, TPE models it, the climber moves there).

Warm starts are evaluated before any member asks: the dispatcher's pick,
the selector's top-k, neighbor problems' winners, other archs' winners
(:class:`WarmStart`; see :mod:`.db` for the transfer queries). Configs that
the space refuses are repaired to the nearest valid one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .params import SearchSpace
from .strategy import Strategy
from .trial import Observation, Trial

__all__ = ["Portfolio", "WarmStart"]


@dataclass(frozen=True)
class WarmStart:
    """A config to evaluate first, and where it came from (``dispatcher``,
    ``selector``, ``neighbor:<problem>``, ``arch:<gfx>``, ...)."""

    config: Mapping[str, object]
    source: str

    def to_json(self) -> dict:
        return {"config": dict(self.config), "source": self.source}

    @classmethod
    def from_json(cls, d: Mapping) -> WarmStart:
        return cls(config=dict(d["config"]), source=str(d["source"]))


class Portfolio(Strategy):
    name = "portfolio"

    def __init__(
        self,
        space: SearchSpace,
        seed: int = 0,
        *,
        members: Sequence[Strategy],
        warm: Sequence[WarmStart] = (),
        discount: float = 0.9,
        min_gain: float = 0.001,
        cost_aware: bool = False,
    ):
        super().__init__(space, seed)
        if not members:
            raise ValueError("Portfolio needs at least one member strategy")
        names = [m.name for m in members]
        if len(set(names)) != len(names):
            raise ValueError(f"Portfolio member names repeat: {names!r}")
        if not 0.0 < discount <= 1.0:
            raise ValueError(f"Portfolio discount={discount}, expected 0 < d <= 1")
        self.members = list(members)
        self.warm = list(warm)
        self.discount = float(discount)
        self.min_gain = float(min_gain)
        self.cost_aware = bool(cost_aware)
        self.spent = [0.0] * len(self.members)
        self.alpha = [1.0] * len(self.members)
        self.beta = [1.0] * len(self.members)
        self.asks = [0] * len(self.members)
        self.wins = [0] * len(self.members)
        self.owner: dict[str, int] = {}
        self.exhausted: set[int] = set()
        self._warm_queue: list[tuple[list[int], str]] | None = None
        self.warm_results: dict[str, str] = {}

    # ---------------------------------------------------------------- warm
    def _build_warm(self) -> None:
        queue: list[tuple[list[int], str]] = []
        keys: set[str] = set()
        for w in self.warm:
            g = self.space.encode(w.config)
            r = self.space.repair(g, self.rng)
            if r is None:
                self.warm_results[w.source] = "unrepairable"
                continue
            k = self.space.key(r)
            if k in keys:
                continue
            keys.add(k)
            queue.append((list(r), w.source if r == g else f"{w.source}+repaired"))
        self._warm_queue = queue

    # -------------------------------------------------------------- bandit
    def _pick(self, blocked: set[int]) -> int | None:
        live = [
            i
            for i in range(len(self.members))
            if i not in self.exhausted and i not in blocked
        ]
        if not live:
            return None
        fresh = [i for i in live if self.asks[i] == 0]
        if fresh:
            return fresh[0]
        draws = []
        mean_cost = sum(self.spent) / max(1, sum(self.asks))
        for i in live:
            score = self.rng.betavariate(self.alpha[i], self.beta[i])
            if self.cost_aware and mean_cost > 0 and self.spent[i] > 0:
                # improvements per second, not per ask: an expensive member
                # (slow compiles, big configs) must earn its share
                score /= (self.spent[i] / self.asks[i]) / mean_cost
            draws.append((score, -i))
        return -max(draws)[1]

    def _reward(self, i: int, success: bool) -> None:
        d = self.discount
        for j in range(len(self.members)):
            self.alpha[j] = 1.0 + d * (self.alpha[j] - 1.0)
            self.beta[j] = 1.0 + d * (self.beta[j] - 1.0)
        self.alpha[i] += 1.0 if success else 0.0
        self.beta[i] += 0.0 if success else 1.0
        if success:
            self.wins[i] += 1

    # ------------------------------------------------------------ protocol
    def ask(self, n: int = 1) -> list[Trial]:
        if self._warm_queue is None:
            self._build_warm()
        out: list[Trial] = []
        taken = self.pending_keys()
        while self._warm_queue and len(out) < int(n):
            g, source = self._warm_queue.pop(0)
            k = self.space.key(g)
            if k in self.history or k in taken:
                continue
            taken.add(k)
            self._uid += 1
            t = Trial(
                genome=tuple(g),
                uid=f"{self.uid_prefix}:{self._uid}",
                origin=f"warm:{source}",
            )
            self.pending[t.uid] = t.genome
            out.append(t)
        if out or any(u not in self.owner for u in self.pending):
            return out  # members start once every warm start is known
        blocked: set[int] = set()
        while len(out) < int(n):
            i = self._pick(blocked)
            if i is None:
                break
            got = self.members[i].ask(1)
            if not got:
                if self.members[i].done:
                    self.exhausted.add(i)
                else:
                    blocked.add(i)  # waiting on its own pending results
                continue
            t = got[0]
            self.asks[i] += 1
            self.owner[t.uid] = i
            self.pending[t.uid] = t.genome
            out.append(t)
        return out

    def tell(self, observations: Sequence[Observation]) -> None:
        for ob in observations:
            _k, before = self.best()
            if ob.trial.origin.startswith("warm:"):
                self.warm_results[ob.trial.origin[5:]] = ob.status
            super().tell([ob])
            for m in self.members:
                m.tell([ob])
            i = self.owner.pop(ob.trial.uid, None)
            if i is None or ob.cached or not ob.full:
                continue
            self.spent[i] += ob.compile_s + ob.measure_s
            gain = self.gain_threshold(self.min_gain, len(ob.samples))
            success = ob.ok and (
                not math.isfinite(before) or ob.value < before * (1.0 - gain)
            )
            self._reward(i, success)

    @property
    def done(self) -> bool:
        return (
            self._warm_queue is not None
            and not self._warm_queue
            and len(self.exhausted) == len(self.members)
        )

    def attach_noise(self, noise) -> None:
        super().attach_noise(noise)
        for m in self.members:
            m.attach_noise(noise)

    def allocation(self) -> dict[str, dict]:
        """Per member: asks, wins (incumbent improvements), posterior mean."""
        return {
            m.name: {
                "asks": self.asks[i],
                "wins": self.wins[i],
                "posterior": self.alpha[i] / (self.alpha[i] + self.beta[i]),
                "seconds": self.spent[i],
                "exhausted": i in self.exhausted,
            }
            for i, m in enumerate(self.members)
        }

    # --------------------------------------------------------------- state
    def _state(self) -> dict:
        return {
            "members": [m.state_dict() for m in self.members],
            "alpha": self.alpha,
            "beta": self.beta,
            "asks": self.asks,
            "wins": self.wins,
            "spent": self.spent,
            "owner": self.owner,
            "exhausted": sorted(self.exhausted),
            "warm_queue": self._warm_queue,
            "warm_results": self.warm_results,
        }

    def _load(self, d: Mapping) -> None:
        if len(d["members"]) != len(self.members):
            raise ValueError(
                f"portfolio state - members={len(d['members'])}, expected "
                f"{len(self.members)} (the same member list)"
            )
        for m, s in zip(self.members, d["members"]):
            m.load_state_dict(s)
        self.alpha = [float(x) for x in d["alpha"]]
        self.beta = [float(x) for x in d["beta"]]
        self.asks = [int(x) for x in d["asks"]]
        self.wins = [int(x) for x in d["wins"]]
        self.spent = [float(x) for x in d["spent"]]
        self.owner = {k: int(v) for k, v in d["owner"].items()}
        self.exhausted = {int(x) for x in d["exhausted"]}
        self._warm_queue = (
            None
            if d["warm_queue"] is None
            else [(list(g), str(s)) for g, s in d["warm_queue"]]
        )
        self.warm_results = dict(d["warm_results"])
