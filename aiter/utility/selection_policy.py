# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""How a tune picks its winner and whether it reads package power.

A tuner builds one policy from ``--select`` and asks it for every
latency-or-power decision: the rank of a candidate, the order of a result
frame, the cost of a multi-stage pair, how many candidates the accurate pass
times, and the package-power sample for a kernel call. Tuners do not branch on
the select mode themselves, and they do not read ``AITER_SELECT``. That
variable only picks the tables ``jit/core`` loads at runtime.

``mp_tuner`` has no ``args``; it reads the policy that ``TunerCommon.run``
activates for the duration of the tune.
"""

import contextlib
import math

from aiter.utility.tune_result import (
    attach_power_columns,
    candidate_beats,
    measure_all_accurate,
    normalize_select,
    power_columns,
    selection_rank,
    sort_frame_for_selection,
    stage_energy,
)


class SelectionPolicy:
    """Latency selection: lowest ``us``, no power reads."""

    name = "latency"
    measures_power = False

    def __reduce__(self):
        return (make_policy, (self.name,))

    def rank(self, us, watts=float("nan")):
        """Sort key for one candidate. Lower is better."""
        return selection_rank(us, watts, self.measures_power)

    def rank_of(self, candidate):
        """Sort key for a dict or attribute-style candidate with ``us``/``watts``."""
        if isinstance(candidate, dict):
            return self.rank(
                candidate.get("us", float("nan")),
                candidate.get("watts", float("nan")),
            )
        return self.rank(
            getattr(candidate, "us", float("nan")),
            getattr(candidate, "watts", float("nan")),
        )

    def beats(self, current, challenger, latency_margin_us: float = 0.0) -> bool:
        """True when challenger replaces current. Equal rank keeps current.

        ``latency_margin_us`` is a noise floor for latency mode only: the
        challenger must be faster by more than that margin.
        """
        if current is None or self.measures_power or not latency_margin_us:
            return candidate_beats(current, challenger, self.measures_power)
        return (
            self.rank_of(challenger)[1] + latency_margin_us < self.rank_of(current)[1]
        )

    def best(self, candidates, latency_choice=None):
        """Winner among candidates.

        ``latency_choice`` is a tuner's own latency rule (for example a timing
        gate with an accuracy tie-break). Latency mode uses it when given;
        power mode always ranks by energy.
        """
        candidates = list(candidates)
        if not candidates:
            return None
        if latency_choice is not None and not self.measures_power:
            return latency_choice(candidates)
        return min(candidates, key=self.rank_of)

    def sort_frame(self, frame):
        """Rows in preference order; the winner is first."""
        return sort_frame_for_selection(frame, self.measures_power)

    def pair_energy(self, us1, watts1, us2, watts2) -> float:
        """Package energy of a two-stage pair. Stage 2 counts only when it ran."""
        return stage_energy(us1, watts1) + stage_energy(us2, watts2)

    def accurate_count(self, shortlist: int, qualified: int) -> int:
        """How many accuracy-qualified candidates the accurate pass times."""
        return measure_all_accurate(self.measures_power, shortlist, qualified)

    def measure(self, call, device_index=None, kernel_us=None) -> float:
        """Package watts while ``call`` runs. NaN when not measured.

        ``kernel_us`` is the timed GPU time of one ``call``; pass it so the
        sample discounts host launch gaps between calls.
        """
        return float("nan")

    def measure_timed(
        self, func, args, kwargs, device_index=None, kernel_us=None
    ) -> float:
        """Package watts for a kernel the timer ran as ``run_perftest(func, *args, **kwargs)``."""
        return float("nan")

    def record(self, target: dict, key, call, device_index=None, kernel_us=None):
        """Store the power sample for ``call`` under ``target[key]`` when measuring.

        Latency mode leaves ``target`` untouched, so a latency result frame
        never gains a ``watts`` column.
        """

    def attach_columns(self, frame, results):
        """Frame with a ``watts`` column on a power tune. Latency frames are unchanged."""
        return frame

    def result_metrics(self, result, tflops):
        """Extra result-row columns and log text for one timed result."""
        return {}, ""

    def watt_text(self, watts) -> str:
        """`` watts=...`` for a candidate log line; empty in latency mode."""
        return ""

    def profile_file(self, tune_file: str) -> str:
        """Default ranked profile path when a tuner writes every pair; empty for latency."""
        return ""


class PowerSelectionPolicy(SelectionPolicy):
    """Power selection: lowest package energy (``us * PPT watts``)."""

    name = "power"
    measures_power = True

    def measure(self, call, device_index=None, kernel_us=None) -> float:
        from aiter.utility.package_power import measure_kernel

        return measure_kernel(call, device_index=device_index, kernel_us=kernel_us)

    def measure_timed(
        self, func, args, kwargs, device_index=None, kernel_us=None
    ) -> float:
        from aiter.utility.package_power import kernel_call

        return self.measure(
            kernel_call(func, args, kwargs),
            device_index=device_index,
            kernel_us=kernel_us,
        )

    def record(self, target: dict, key, call, device_index=None, kernel_us=None):
        target[key] = self.measure(call, device_index=device_index, kernel_us=kernel_us)

    def attach_columns(self, frame, results):
        return attach_power_columns(frame, results)

    def result_metrics(self, result, tflops):
        watts, tflops_per_w = power_columns(result, tflops)
        return (
            {"watts": [watts]},
            f", watts={watts} W, tflops_per_w={tflops_per_w} TFLOPS/W",
        )

    def watt_text(self, watts) -> str:
        try:
            watts_f = float(watts)
        except (TypeError, ValueError):
            watts_f = float("nan")
        if math.isnan(watts_f):
            return " watts=nan"
        return f" watts={watts_f:.1f}"

    def profile_file(self, tune_file: str) -> str:
        import os

        root, ext = os.path.splitext(tune_file)
        return (root if ext else tune_file) + ".ranked.csv"


LATENCY = SelectionPolicy()
POWER = PowerSelectionPolicy()


def make_policy(select) -> SelectionPolicy:
    """Policy for a ``--select`` value or an ``args`` namespace that carries one."""
    if select is not None and not isinstance(select, str):
        select = getattr(select, "select", None)
    return POWER if normalize_select(select) == "power" else LATENCY


_active = LATENCY


def active_policy() -> SelectionPolicy:
    """Policy of the tune that is running in this process (latency when none is)."""
    return _active


@contextlib.contextmanager
def use_policy(policy: SelectionPolicy):
    """Make ``policy`` the one ``mp_tuner`` and other arg-less helpers see."""
    global _active
    previous = _active
    _active = policy
    try:
        yield policy
    finally:
        _active = previous
