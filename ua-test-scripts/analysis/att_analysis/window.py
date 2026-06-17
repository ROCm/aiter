"""Pick a short, representative steady-state window of a few loop iterations.

The FA4 main loop issues a block barrier each iteration, so barrier cycles are a
natural iteration ruler. We take the median barrier spacing in the middle of the
wave (steady state, away from prologue/epilogue) and return a window spanning
``n_iters`` of them.
"""
from __future__ import annotations

import statistics
from .model import Wave, Trace


def barrier_cycles(trace: Trace, wave: Wave) -> list[int]:
    """Absolute cycles of s_barrier instructions in this wave's stream."""
    bar_linenos = {c.lineno for c in trace.code
                   if (m := c.mnemonic) and m.startswith("s_barrier")}
    if not bar_linenos:
        return []
    return [i.cycle for i in wave.instructions if i.lineno in bar_linenos]


def pick_window(trace: Trace, wave: Wave, n_iters: int = 3,
                center_frac: float = 0.5) -> tuple[int, int, list[int]]:
    """Return (lo_cycle, hi_cycle, iteration_boundary_cycles) for n_iters.

    Falls back to a cycle-length heuristic if no barriers are found.
    """
    begin, end = wave.begin, wave.end
    bars = barrier_cycles(trace, wave)

    if len(bars) >= 4:
        # Collapse barriers that are within a few cycles (multiple s_barrier in
        # one sync point) into a single boundary.
        merged = [bars[0]]
        for c in bars[1:]:
            if c - merged[-1] > 32:
                merged.append(c)
        if len(merged) >= n_iters + 1:
            deltas = [b - a for a, b in zip(merged, merged[1:])]
            period = statistics.median(deltas)
            center = begin + int((end - begin) * center_frac)
            # first boundary at/after center
            start_i = next((k for k, c in enumerate(merged) if c >= center), 0)
            start_i = min(start_i, len(merged) - 1 - n_iters)
            start_i = max(start_i, 0)
            lo = merged[start_i]
            hi = merged[min(start_i + n_iters, len(merged) - 1)]
            bounds = merged[start_i:start_i + n_iters + 1]
            return lo, hi, bounds

    # Fallback: estimate period from total instructions / barrier-ish density.
    span = end - begin
    lo = begin + int(span * center_frac)
    hi = min(end, lo + int(span * 0.06) * max(1, n_iters))
    return lo, hi, []
