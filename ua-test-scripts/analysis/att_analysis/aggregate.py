"""Per-wave / per-phase cycle rollups and a warp-group imbalance metric.

Given the two co-resident waves, we summarise -- over a chosen window -- how
many cycles each spends executing vs stalling vs waiting, broken down by phase.
The headline numbers:

* ``phase_cycles``  : per phase, the busy/stall cycles attributed by instruction.
* ``state_cycles``  : per state (exec/stall/wait/idle) from the wave timeline.
* ``overlap``       : fraction of the window where WG0 is in MATRIX while WG1 is
                      in SOFTMAX (and vice-versa) -- the thing the FA4 ping-pong
                      is trying to maximise.
* ``imbalance``     : |matrix_busy(WG0) - matrix_busy(WG1)| style deltas.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .model import Trace, Wave, STATE_NAME
from .phases import PhaseTagger, MATRIX, SOFTMAX


@dataclass
class WaveStats:
    wid: int
    slot: int
    window: tuple[int, int]
    state_cycles: dict[str, int] = field(default_factory=dict)
    phase_busy: dict[str, int] = field(default_factory=dict)   # exec+stall by phase
    phase_stall: dict[str, int] = field(default_factory=dict)  # stall only by phase
    n_inst: int = 0

    @property
    def total(self) -> int:
        return self.window[1] - self.window[0]

    def state_frac(self, st: str) -> float:
        return self.state_cycles.get(st, 0) / max(self.total, 1)


def wave_stats(trace: Trace, tagger: PhaseTagger, wave: Wave,
               lo: int, hi: int) -> WaveStats:
    s = WaveStats(wid=wave.wid, slot=wave.slot, window=(lo, hi))
    for a, b, st in wave.state_segments():
        if b < lo or a > hi:
            continue
        dur = min(b, hi) - max(a, lo)
        if dur > 0:
            name = STATE_NAME.get(st, "idle")
            s.state_cycles[name] = s.state_cycles.get(name, 0) + dur
    for i in wave.insts_in_window(lo, hi):
        ph = tagger.phase_of_lineno(i.lineno)
        s.phase_busy[ph] = s.phase_busy.get(ph, 0) + i.total
        s.phase_stall[ph] = s.phase_stall.get(ph, 0) + i.stall
        s.n_inst += 1
    return s


def phase_timeline(trace: Trace, tagger: PhaseTagger, wave: Wave,
                   lo: int, hi: int) -> list[tuple[int, int, str]]:
    """Coalesced (start, end, phase) segments from instruction tokens."""
    segs: list[tuple[int, int, str]] = []
    for i in wave.insts_in_window(lo, hi):
        ph = tagger.phase_of_lineno(i.lineno)
        end = i.cycle + max(i.total, 1)
        if segs and segs[-1][2] == ph and i.cycle <= segs[-1][1] + 2:
            segs[-1] = (segs[-1][0], max(segs[-1][1], end), ph)
        else:
            segs.append((i.cycle, end, ph))
    return segs


def overlap_fraction(seg0, seg1, lo: int, hi: int) -> dict[str, float]:
    """Fraction of [lo,hi] where one wave is MATRIX and the other SOFTMAX."""
    def sample(segs, c):
        for a, b, ph in segs:
            if a <= c < b:
                return ph
        return None
    step = max((hi - lo) // 2000, 1)
    good = both = total = 0
    for c in range(lo, hi, step):
        p0, p1 = sample(seg0, c), sample(seg1, c)
        total += 1
        if {p0, p1} == {MATRIX, SOFTMAX}:
            good += 1
        if p0 == p1 == MATRIX or p0 == p1 == SOFTMAX:
            both += 1
    return {"matrix_softmax_overlap": good / max(total, 1),
            "same_phase_collision": both / max(total, 1)}


def summarize(trace: Trace, lo: int, hi: int, se: int = 0, simd: int = 0):
    w0, w1 = trace.coresident_pair(se=se, simd=simd)
    tagger = PhaseTagger(trace)
    s0 = wave_stats(trace, tagger, w0, lo, hi)
    s1 = wave_stats(trace, tagger, w1, lo, hi)
    seg0 = phase_timeline(trace, tagger, w0, lo, hi)
    seg1 = phase_timeline(trace, tagger, w1, lo, hi)
    ov = overlap_fraction(seg0, seg1, lo, hi)
    return {
        "window": (lo, hi),
        "phase_mode": tagger.mode,
        "wg0": s0,
        "wg1": s1,
        "overlap": ov,
    }
