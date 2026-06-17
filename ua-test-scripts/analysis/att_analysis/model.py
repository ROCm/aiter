"""Data model + loader for rocprofv3 ATT ``ui_output_*`` trace directories.

This is the headless equivalent of what ROCprof Compute Viewer ingests. A
``ui_output_agent_{N}_dispatch_{N}`` directory contains:

* ``filenames.json``  -- maps [SE][SIMD][slot][wid] -> (per-wave file, begin, end)
* ``code.json``       -- the ISA listing + per-line cost, header:
                         ``ISA, _, LineNumber, Source, Codeobj, Vaddr, Hit, Latency, Stall, Idle``
* ``se{SE}_sm{SIMD}_sl{slot}_wv{wid}.json`` -- one decoded wave, with:
      wave.instructions : list of [cycle, exec, stall, total, code_line]
      wave.timeline     : list of [state, duration]   state 1=idle 2=exec 3=wait 4=stall
      wave.waitcnt      : list of [tok_idx, [[dep_tok, kind], ...]]
      wave.begin/end/cu/simd/slot/id

All cycles are absolute in the SIMD clock domain, so two waves co-resident on
one SIMD share the same axis.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import Iterator

# Wave-state codes used by the ATT decoder (confirmed by summing to wave dur).
STATE_IDLE, STATE_EXEC, STATE_WAIT, STATE_STALL = 1, 2, 3, 4
STATE_NAME = {STATE_IDLE: "idle", STATE_EXEC: "exec", STATE_WAIT: "wait", STATE_STALL: "stall"}

# code.json row indices (per its header).
C_ISA, C_LINENO, C_SOURCE, C_CODEOBJ, C_VADDR, C_HIT, C_LAT, C_STALL, C_IDLE = 0, 2, 3, 4, 5, 6, 7, 8, 9


@dataclass(frozen=True)
class CodeLine:
    """One row of code.json: an ISA instruction (or a ``;`` section comment)."""
    lineno: int          # index referenced by instruction tokens (field 4)
    isa: str             # disassembled text, or "; <symbol>" for section headers
    source: str          # "file:line" when built with debug line tables, else ""
    vaddr: int
    hit: int
    latency: int         # aggregate latency cycles over the whole trace
    stall: int           # aggregate stall cycles

    @property
    def is_comment(self) -> bool:
        return self.isa.lstrip().startswith(";")

    @property
    def mnemonic(self) -> str:
        s = self.isa.strip()
        return s.split()[0] if s and not self.is_comment else ""


@dataclass
class Inst:
    """One executed instruction token in a wave's stream."""
    __slots__ = ("cycle", "exec", "stall", "total", "lineno")
    cycle: int
    exec: int
    stall: int
    total: int
    lineno: int


@dataclass
class Wave:
    se: int
    simd: int
    slot: int
    wid: int
    path: str
    _raw: dict = field(default=None, repr=False)

    @cached_property
    def _wave(self) -> dict:
        if self._raw is None:
            with open(self.path) as f:
                self._raw = json.load(f)
        return self._raw["wave"]

    @property
    def begin(self) -> int:
        return self._wave["begin"]

    @property
    def end(self) -> int:
        return self._wave["end"]

    @property
    def cu(self) -> int:
        return self._wave["cu"]

    @cached_property
    def instructions(self) -> list[Inst]:
        return [Inst(c, e, s, t, ln) for c, e, s, t, ln in self._wave["instructions"]]

    @cached_property
    def timeline(self) -> list[tuple[int, int]]:
        """State segments as (state_code, duration), in order from wave.begin."""
        return [(s, d) for s, d in self._wave["timeline"]]

    def state_segments(self) -> Iterator[tuple[int, int, int]]:
        """Yield (start_cycle, end_cycle, state_code) absolute segments."""
        t = self.begin
        for state, dur in self.timeline:
            yield t, t + dur, state
            t += dur

    def insts_in_window(self, lo: int, hi: int) -> list[Inst]:
        return [i for i in self.instructions if lo <= i.cycle <= hi]

    def __repr__(self) -> str:
        return f"Wave(se{self.se} sm{self.simd} sl{self.slot} wv{self.wid})"


class Trace:
    """A single decoded ATT dispatch (``ui_output_*`` directory)."""

    def __init__(self, ui_dir: str):
        self.dir = ui_dir
        with open(os.path.join(ui_dir, "filenames.json")) as f:
            self.filenames = json.load(f)
        self._code: list[CodeLine] | None = None
        self._code_by_lineno: dict[int, CodeLine] | None = None

    # ----- code.json (ISA + source) -------------------------------------
    @property
    def code(self) -> list[CodeLine]:
        if self._code is None:
            with open(os.path.join(self.dir, "code.json")) as f:
                raw = json.load(f)["code"]
            self._code = [
                CodeLine(
                    lineno=r[C_LINENO], isa=r[C_ISA], source=r[C_SOURCE],
                    vaddr=r[C_VADDR], hit=r[C_HIT], latency=r[C_LAT], stall=r[C_STALL],
                )
                for r in raw
            ]
            self._code_by_lineno = {c.lineno: c for c in self._code}
        return self._code

    def code_line(self, lineno: int) -> CodeLine | None:
        if self._code is None:
            _ = self.code
        return self._code_by_lineno.get(lineno)

    @property
    def has_source(self) -> bool:
        """True if the trace carries ISA->source mapping (line-tables build)."""
        return any(c.source for c in self.code if not c.is_comment)

    # ----- waves ---------------------------------------------------------
    def waves(self) -> list[Wave]:
        out = []
        wf = self.filenames["wave_filenames"]
        for se, simds in wf.items():
            for simd, slots in simds.items():
                for slot, wids in slots.items():
                    for wid, entry in wids.items():
                        fname = entry[0]
                        out.append(Wave(int(se), int(simd), int(slot), int(wid),
                                        os.path.join(self.dir, fname)))
        return out

    def wave(self, se: int, simd: int, slot: int, wid: int) -> Wave:
        entry = self.filenames["wave_filenames"][str(se)][str(simd)][str(slot)][str(wid)]
        return Wave(se, simd, slot, wid, os.path.join(self.dir, entry[0]))

    def wave_window(self, se: int, simd: int, slot: int, wid: int) -> tuple[int, int]:
        entry = self.filenames["wave_filenames"][str(se)][str(simd)][str(slot)][str(wid)]
        return entry[1], entry[2]

    def coresident_pair(self, se: int = 0, simd: int = 0) -> tuple[Wave, Wave]:
        """Find a slot0 wave and a slot1 wave on (se,simd) with maximal overlap.

        These are the two waves time-sharing the SIMD -- on the FA4 pipeline one
        belongs to each warp group, which is exactly the overlap we want to see.
        """
        wf = self.filenames["wave_filenames"][str(se)][str(simd)]

        def slot_ranges(slot: str):
            return [(int(w), e[1], e[2], e[0]) for w, e in wf.get(slot, {}).items()]

        s0, s1 = slot_ranges("0"), slot_ranges("1")
        if not s0 or not s1:
            raise ValueError(f"SIMD {simd} on SE {se} does not have two occupied slots")
        best, pair = -1, None
        for w0, b0, e0, f0 in s0:
            for w1, b1, e1, f1 in s1:
                ov = min(e0, e1) - max(b0, b1)
                if ov > best:
                    best, pair = ov, ((w0, f0), (w1, f1))
        (w0, f0), (w1, f1) = pair
        return (Wave(se, simd, 0, w0, os.path.join(self.dir, f0)),
                Wave(se, simd, 1, w1, os.path.join(self.dir, f1)))


def find_ui_dir(run_or_ui_dir: str) -> str:
    """Accept a run dir, its att/ subdir, or a ui_output_* dir; return the ui dir."""
    if os.path.isfile(os.path.join(run_or_ui_dir, "filenames.json")):
        return run_or_ui_dir
    for root, dirs, _ in os.walk(run_or_ui_dir):
        for d in dirs:
            if d.startswith("ui_output_agent_"):
                return os.path.join(root, d)
    raise FileNotFoundError(f"no ui_output_* directory under {run_or_ui_dir}")
