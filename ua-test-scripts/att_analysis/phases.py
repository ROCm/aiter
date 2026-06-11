"""Classify each instruction into an FA4 pipeline phase.

Accurate path (trace built with -gline-tables-only): every ISA line carries a
``file:line``. We classify with this precedence:

  1. control opcodes      s_barrier -> BARRIER, s_waitcnt -> MEMWAIT
  2. unambiguous compute   v_mfma -> MATRIX, v_exp/v_rcp/... -> SOFTMAX
  3. source file           warp_gemm*->MATRIX, hip_math/math/masking->SOFTMAX,
                           buffer_addressing->LOAD, magic_div/coordinate/kernel->ADDR
  4. pipeline lambda range  fa4_matrix/gemm->MATRIX, fmha_alu*/fa4_softmax->SOFTMAX,
                           K/V_mem_load->LOAD, K/V_lds_load->LDS, refresh_*->ADDR
                           (ranges parsed from the .hpp so they track edits)
  5. mnemonic fallback     ds_read/ds_write->LDS, buffer/global_load->LOAD
  6. OTHER

Fallback path (no source): mnemonic only.
"""
from __future__ import annotations

import os
import re

from .model import CodeLine, Trace

MATRIX = "matrix"
SOFTMAX = "softmax"
LOAD = "load"        # global memory (K/V/Q DRAM) loads
LDS = "lds"          # ds_read/ds_write (incl. the hoisted V transpose reads)
ADDR = "addr"        # address calc / offset refresh / kernel setup
BARRIER = "barrier"
MEMWAIT = "memwait"  # s_waitcnt
OTHER = "other"

PHASE_COLORS = {
    MATRIX:   "#1f77b4",   # blue
    SOFTMAX:  "#ff7f0e",   # orange
    LOAD:     "#8c564b",   # brown   -- global mem
    LDS:      "#17becf",   # cyan    -- ds_read/write
    ADDR:     "#9467bd",   # purple  -- addressing
    BARRIER:  "#d62728",   # red
    MEMWAIT:  "#7f7f7f",   # gray
    OTHER:    "#dddddd",
}
PHASE_ORDER = [MATRIX, SOFTMAX, LDS, LOAD, ADDR, BARRIER, MEMWAIT, OTHER]

# ---- mnemonic rules -----------------------------------------------------
_MFMA = re.compile(r"^v_mfma|^v_smfmac")
_TRANSC = re.compile(r"^v_exp|^v_rcp|^v_log|^v_rsq|^v_sqrt")
_LDS = re.compile(r"^ds_(read|write|load|store)")
_GLOAD = re.compile(r"^buffer_load|^global_load|^buffer_store|^global_store")


def _mnemonic_phase(c: CodeLine) -> str | None:
    m = c.mnemonic
    if not m:
        return None
    if m.startswith("s_barrier"):
        return BARRIER
    if m.startswith("s_waitcnt"):
        return MEMWAIT
    if _MFMA.match(m):
        return MATRIX
    if _TRANSC.match(m):
        return SOFTMAX
    if _LDS.match(m):
        return LDS
    if _GLOAD.match(m):
        return LOAD
    return None


def classify_mnemonic(c: CodeLine) -> str:
    return _mnemonic_phase(c) or OTHER


# ---- source-file rules --------------------------------------------------
_FILE_PHASE = [
    (re.compile(r"warp_gemm"), MATRIX),
    (re.compile(r"__clang_hip_math\.h|/math\.hpp|math_v2|block_fmha_fwd_v3_pipeline"), SOFTMAX),
    (re.compile(r"block_masking"), SOFTMAX),
    (re.compile(r"amd_buffer_addressing"), LOAD),
    (re.compile(r"magic_div|coordinate_transform|unified_attention_kernel|tensor_coordinate"), ADDR),
]


def _file_phase(filename: str) -> str | None:
    for pat, ph in _FILE_PHASE:
        if pat.search(filename):
            return ph
    return None


# ---- pipeline lambda line ranges (parsed from the .hpp) -----------------
_LAMBDA_PHASE = {
    "fa4_matrix": MATRIX, "gemm": MATRIX, "cl_calc": MATRIX, "fa4_post_process": MATRIX,
    "fmha_alu0": SOFTMAX, "fmha_alu1": SOFTMAX, "fmha_alu_D_upd": SOFTMAX,
    "fmha_mask_at": SOFTMAX, "fa4_softmax": SOFTMAX, "fmha_post_process": SOFTMAX,
    "K_mem_load": LOAD, "V_mem_load": LOAD, "V_mem_load_rt": LOAD,
    "K_lds_load": LDS, "V_lds_load": LDS, "V_lds_load_rt": LDS,
    "refresh_k_offsets": ADDR, "refresh_v_offsets": ADDR,
}
_LAMBDA_RE = re.compile(r"^\s*(?:\[\[maybe_unused\]\]\s*)?auto\s+(\w+)\s*=\s*\[")


def parse_lambda_ranges(hpp_path: str) -> list[tuple[int, int, str]]:
    """Return [(start_line, end_line, phase)] for known pipeline lambdas.

    A lambda spans from its definition line to the next lambda definition
    (good enough: we only consult these ranges for instructions whose deepest
    source is the pipeline .hpp itself, i.e. pipeline-local code).
    """
    starts: list[tuple[int, str]] = []
    with open(hpp_path) as f:
        for n, line in enumerate(f, 1):
            m = _LAMBDA_RE.match(line)
            if m:
                starts.append((n, m.group(1)))
    ranges = []
    for i, (ln, name) in enumerate(starts):
        end = starts[i + 1][0] - 1 if i + 1 < len(starts) else ln + 400
        if name in _LAMBDA_PHASE:
            ranges.append((ln, end, _LAMBDA_PHASE[name]))
    return ranges


class PhaseTagger:
    """Caches lineno -> phase for a trace, choosing source or mnemonic mode."""

    PIPELINE_HPP = "unified_attention_pipeline.hpp"

    def __init__(self, trace: Trace, prefer_source: bool | None = None):
        self.trace = trace
        self.use_source = trace.has_source if prefer_source is None else prefer_source
        self._cache: dict[int, str] = {}
        self._ranges: list[tuple[int, int, str]] = []
        if self.use_source:
            self._ranges = self._load_pipeline_ranges()

    def _load_pipeline_ranges(self):
        # Find the pipeline .hpp path from the trace's Source strings.
        for c in self.trace.code:
            if c.source and self.PIPELINE_HPP in c.source:
                path = c.source.rsplit(":", 1)[0]
                if os.path.isfile(path):
                    try:
                        return parse_lambda_ranges(path)
                    except OSError:
                        return []
        return []

    @property
    def mode(self) -> str:
        return "source" if self.use_source else "mnemonic"

    def _pipeline_line_phase(self, lineno_src: int) -> str | None:
        for lo, hi, ph in self._ranges:
            if lo <= lineno_src <= hi:
                return ph
        return None

    def _classify_source(self, c: CodeLine) -> str:
        # 1. unambiguous opcodes win over the source file: a ds_read is an LDS
        #    op even though its builtin lives in amd_buffer_addressing*.hpp, and
        #    s_barrier/s_waitcnt/mfma/exp are self-evident regardless of file.
        mp = _mnemonic_phase(c)
        if mp is not None:
            return mp
        filename, _, line_s = c.source.rpartition(":")
        # 2. source file (for generic VALU/scalar ops whose file is meaningful)
        fp = _file_phase(filename)
        if fp is not None:
            return fp
        # 3. pipeline lambda range
        if self.PIPELINE_HPP in filename and line_s.isdigit():
            pp = self._pipeline_line_phase(int(line_s))
            if pp is not None:
                return pp
        return OTHER

    def phase_of_lineno(self, lineno: int) -> str:
        if lineno in self._cache:
            return self._cache[lineno]
        c = self.trace.code_line(lineno)
        if c is None:
            ph = OTHER
        elif self.use_source and c.source:
            ph = self._classify_source(c)
        else:
            ph = classify_mnemonic(c)
        self._cache[lineno] = ph
        return ph
