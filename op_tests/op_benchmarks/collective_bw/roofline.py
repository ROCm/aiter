# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The TransferBench rooflines of an op's ``Traffic`` and of its RCCL calls.

The op's roof is the fastest way TransferBench finds to move the op's bytes,
in the op's own pattern and memory type: at the op's own grid or at a fixed
copy width, lap after lap or (for an op that overlaps its laps) all at once.
See ``patterns`` for the candidates. The roof is optimistic in every other
way: it has no handshakes, no quantization or layout work, no cross-GPU
ordering, and it is timed inside the kernel, so the host launch is left out.

The RCCL roof moves RCCL's own bytes (unquantized, ``patterns.rccl_traffic``)
and also tries every memory type, as RCCL's internals are not modelled.

Both are the best of a stock copy kernel, not the link's physical peak. A
library with a better copy loop can beat them: RCCL's bf16 all-reduce does by
~10% at large sizes on MI350X, where no TransferBench executor (GFX or DMA,
push or pull, any memory type, width or unroll) gets past ~58 GB/s per xGMI
direction while RCCL sustains ~65 GB/s.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import patterns
import tb_text

from aiter.ops.flydsl.comm_traffic import Traffic

logger = logging.getLogger("aiter")


@dataclass(frozen=True)
class Roof:
    us: float  # whole call, the fastest candidate
    phase_us: tuple[float, ...]  # that candidate, per test, repeat included
    label: str = ""  # that candidate's label


class Roofline:
    """Measures and remembers roofs. ``available`` is False when no
    TransferBench binary was found; ``measure`` then returns nothing."""

    def __init__(
        self,
        binary: str | None = None,
        *,
        iters: int = 20,
        warmup: int = 3,
        timeout: float = 900.0,
    ):
        self.binary = tb_text.find_binary(binary)
        self.iters, self.warmup, self.timeout = iters, warmup, timeout
        self.version: str | None = None
        self._roofs: dict[tuple[str, Traffic], Roof] = {}
        if self.binary is None:
            logger.warning(
                "TransferBench not found ($%s, PATH): no roofline. Build it from "
                "https://github.com/ROCm/TransferBench and set $%s.",
                tb_text.ENV_BIN,
                tb_text.ENV_BIN,
            )

    @property
    def available(self) -> bool:
        return self.binary is not None

    def measure(self, traffics: Iterable[Traffic]) -> dict[Traffic, Roof]:
        """The op roofs of *traffics*."""
        wanted = list(dict.fromkeys(traffics))
        return self._solve("op", {t: patterns.candidates(t) for t in wanted})

    def measure_rccl(self, traffics: Iterable[Traffic]) -> dict[Traffic, Roof]:
        """The RCCL roofs of *traffics*, keyed by ``patterns.rccl_traffic``."""
        wanted = list(dict.fromkeys(patterns.rccl_traffic(t) for t in traffics))
        return self._solve("rccl", {t: patterns.rccl_candidates(t) for t in wanted})

    def _solve(self, kind: str, specs: dict[Traffic, list]) -> dict[Traffic, Roof]:
        """The fastest candidate of each traffic in *specs*. *kind* keeps
        the op and RCCL roofs of one traffic apart in the cache."""
        if not self.available:
            return {}
        todo = {
            (kind, t): c for t, c in specs.items() if (kind, t) not in self._roofs
        }
        # Each distinct test runs once, and TransferBench once per environment.
        tests = list(
            dict.fromkeys(t for cands in todo.values() for c in cands for t in c.tests)
        )
        by_env: dict[tuple, list[patterns.TbTest]] = defaultdict(list)
        for test in tests:
            by_env[test.env].append(test)
        ms: dict[tuple[str, tuple], float] = {}
        for env, group in by_env.items():
            lines = list(dict.fromkeys(t.line for t in group))
            results, self.version = tb_text.run(
                self.binary,
                lines,
                dict(env),
                iters=self.iters,
                warmup=self.warmup,
                timeout=self.timeout,
            )
            for line, res in zip(lines, results):
                ms[(line, env)] = res.kernel_ms
        for key, cands in todo.items():
            best = None
            for c in cands:
                us = tuple(ms[(t.line, t.env)] * 1e3 * t.repeat for t in c.tests)
                if best is None or sum(us) < best.us:
                    best = Roof(us=sum(us), phase_us=us, label=c.label)
            self._roofs[key] = best
        return {t: self._roofs[(kind, t)] for t in specs}
