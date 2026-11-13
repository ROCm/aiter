# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The TransferBench roofline of an op's ``Traffic``.

The roof of a call is the time TransferBench needs to move the same bytes, in
the op's own pattern, into the op's own memory type, with the op's own launch
geometry. Each phase is run as its own test and the phase times are summed, so
the roof does not see any overlap between phases that the op may get. The
roof is optimistic in every other way: it has no handshakes, no quantization
or layout work, no cross-GPU ordering, and its launch cost is amortized over
sub-iterations.

Run it only while no benchmark process holds the GPUs.
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
    us: float  # whole call
    phase_us: tuple[float, ...]  # per phase, repeat included


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
        self._roofs: dict[Traffic, Roof] = {}
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
        """Roofs of *traffics*. TransferBench runs once per distinct
        environment (block size, sub-iterations) over all phases that share it."""
        wanted = list(dict.fromkeys(traffics))
        if not self.available:
            return {}
        todo = [t for t in wanted if t not in self._roofs]
        by_env: dict[tuple, list[tuple[Traffic, int, patterns.TbTest]]] = defaultdict(
            list
        )
        for traffic in todo:
            for i, test in enumerate(patterns.tb_tests(traffic)):
                by_env[test.env].append((traffic, i, test))
        phase_us: dict[Traffic, list[float]] = {
            t: [float("nan")] * len(t.phases) for t in todo
        }
        for env, entries in by_env.items():
            results, self.version = tb_text.run(
                self.binary,
                [test.line for _t, _i, test in entries],
                dict(env),
                iters=self.iters,
                warmup=self.warmup,
                timeout=self.timeout,
            )
            for (traffic, i, test), res in zip(entries, results):
                phase_us[traffic][i] = res.slowest_ms * 1e3 * test.repeat
        for traffic, us in phase_us.items():
            self._roofs[traffic] = Roof(us=sum(us), phase_us=tuple(us))
        return {t: self._roofs[t] for t in wanted}
