# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""SILOTIGER-1481 phase-0 harness for the logits select loop.

Builds the six shapes in design-spike.md, calls ``qsa_select_paged_tokens``
with ``allow_flydsl=True``, and times score, top-k, and expand separately by
wrapping those three calls. CUDA events are summed across the row chunks of
one step. The default protocol is 5 warmup steps and the median of 10.

``qsa_indexer_block_ids`` is in this tree, and these shapes pass its gate, so
the public select would not run the three calls. The spike's probe base is
the logits chunk loop, so the call is made with that gate rejecting. Timings
are that loop, not the indexer.

Correctness smoke on ``s8k`` and ``s32k``: each row's selected block set
matches an fp32 torch top-512 over the visible window the top-k read. A block
whose score equals the 512th may be in or out. That guards this harness. It
is not the ticket's fp64 set gate.

Launch from the repo root, in flydsl_venv, through gpuq (see gpuq.md)::

    export GPUQ_USER=<owner>
    gpuq --num-gpus=1 /workspaces/aiter/flydsl_venv/bin/python \\
        op_tests/spike_silotiger_1481_select_probe.py
"""

import argparse
import ctypes
import ctypes.util
import os
import statistics
import sys
from dataclasses import dataclass

import torch

import aiter.ops.qsa as qsa_mod
from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.ops.qsa import (
    DEFAULT_LOGITS_WORKSPACE_BYTES,
    qsa_select_paged_tokens,
)

COMPRESS_RATIO = 4
PAGE_SIZE = 16
TOKEN_TOPK = 2048
BLOCK_TOPK = TOKEN_TOPK // COMPRESS_RATIO
HEADS = 4
DIM = 128
SMOKE_CASES = ("s8k", "s32k")
SUPPORTED_GFX = ("gfx942", "gfx950")


@dataclass(frozen=True)
class Request:
    kind: str
    rows: int
    columns: int


@dataclass(frozen=True)
class Geometry:
    rows: int
    columns: int
    requests: int
    rows_per_chunk: int
    chunks: int
    score_bytes: int
    unique_k_bytes: int


@dataclass(frozen=True)
class Inputs:
    q: torch.Tensor
    k_cache: torch.Tensor
    page_table: torch.Tensor
    token_to_request: torch.Tensor
    query_positions: torch.Tensor
    context_lens: torch.Tensor


@dataclass
class StepTimes:
    score_us: float
    topk_us: float
    expand_us: float
    flydsl_chunks: int
    loop_chunks: int


def _requests(*specs: tuple[str, int, int]) -> tuple[Request, ...]:
    return tuple(Request(kind, rows, columns) for kind, rows, columns in specs)


# mix128k is a mixed step: decode rows first (one row per request), then the
# prefill rows. Each request has its own pages at the same column count.
CASES: dict[str, tuple[Request, ...]] = {
    "s8k": _requests(("prefill", 8192, 2048)),
    "s32k": _requests(("prefill", 16384, 8192)),
    "s128k": _requests(("prefill", 16384, 32768)),
    "s256k": _requests(("prefill", 16384, 65536)),
    "two128k": _requests(("prefill", 8192, 32768), ("prefill", 8192, 32768)),
    "mix128k": _requests(
        *(("decode", 1, 32768),) * 64,
        ("prefill", 16384, 32768),
    ),
}

EXPECTED_CHUNKS = {
    "s8k": 1,
    "s32k": 4,
    "s128k": 16,
    "s256k": 32,
    "two128k": 16,
    "mix128k": 17,
}
EXPECTED_SCORE_BYTES = {
    "s128k": 2 * 1024**3,
    "s256k": 4 * 1024**3,
}


def geometry(requests: tuple[Request, ...]) -> Geometry:
    """Chunk count and fp32 score traffic for one step.

    Page-table width is the shared column count divided by the compressed
    page size. The workspace cap is ``DEFAULT_LOGITS_WORKSPACE_BYTES``.
    """
    columns = {req.columns for req in requests}
    if len(columns) != 1:
        raise ValueError("a step has one page-table width")
    width_columns = requests[0].columns
    if width_columns % PAGE_SIZE:
        raise ValueError("columns must be a multiple of the compressed page")
    rows = sum(req.rows for req in requests)
    rows_per_chunk = max(1, DEFAULT_LOGITS_WORKSPACE_BYTES // (width_columns * 4))
    chunks = (rows + rows_per_chunk - 1) // rows_per_chunk
    return Geometry(
        rows=rows,
        columns=width_columns,
        requests=len(requests),
        rows_per_chunk=rows_per_chunk,
        chunks=chunks,
        score_bytes=rows * width_columns * 4,
        unique_k_bytes=sum(req.columns * DIM * 2 for req in requests),
    )


def check_catalog() -> None:
    """The shape table in design-spike.md, checked without launching."""
    for name, requests in CASES.items():
        geo = geometry(requests)
        if geo.chunks != EXPECTED_CHUNKS[name]:
            raise AssertionError(
                f"{name}: chunks {geo.chunks} != {EXPECTED_CHUNKS[name]}"
            )
        expected = EXPECTED_SCORE_BYTES.get(name)
        if expected is not None and geo.score_bytes != expected:
            raise AssertionError(f"{name}: score bytes {geo.score_bytes} != {expected}")


def build_inputs(requests: tuple[Request, ...], seed: int) -> Inputs:
    """Wide-end prefill: every row's visible groups equal that request's columns.

    K is one contiguous cache. Request ``r`` owns pages ``[r * width, (r + 1) * width)``.
    """
    geo = geometry(requests)
    width = geo.columns // PAGE_SIZE
    torch.manual_seed(seed)
    q = torch.randn(geo.rows, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    k_cache = torch.randn(
        geo.requests * width,
        PAGE_SIZE,
        1,
        DIM,
        dtype=torch.bfloat16,
        device="cuda",
    )
    page_table = torch.arange(
        geo.requests * width, dtype=torch.int32, device="cuda"
    ).view(geo.requests, width)
    token_to_request = torch.empty(geo.rows, dtype=torch.int32, device="cuda")
    query_positions = torch.empty(geo.rows, dtype=torch.int32, device="cuda")
    context_lens = torch.empty(geo.requests, dtype=torch.int32, device="cuda")
    offset = 0
    for req_id, req in enumerate(requests):
        context = req.columns * COMPRESS_RATIO
        token_to_request[offset : offset + req.rows] = req_id
        query_positions[offset : offset + req.rows] = context - 1
        context_lens[req_id] = context
        offset += req.rows
    return Inputs(
        q=q,
        k_cache=k_cache,
        page_table=page_table,
        token_to_request=token_to_request,
        query_positions=query_positions,
        context_lens=context_lens,
    )


def _reject_indexer(*_args, **_kwargs) -> str:
    return "spike probe base is the logits chunk loop, not qsa_indexer_block_ids"


class _Probe:
    """Wrap the three loop calls, and keep the select off the indexer."""

    def __init__(self, time_stages: bool, check_blocks: bool):
        self.time_stages = time_stages
        self.check_blocks = check_blocks
        self.events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] = {
            "score": [],
            "topk": [],
            "expand": [],
        }
        self.flydsl_chunks = 0
        self.loop_chunks = 0
        self.expand_chunks = 0
        self._logits: torch.Tensor | None = None
        self._saved: list[object] = []

    def __enter__(self) -> "_Probe":
        qsa = qsa_mod
        paged = qsa._qsa_paged
        if paged is None:
            raise RuntimeError("FlyDSL qsa_paged is not importable")
        real_flydsl = qsa.qsa_paged_mqa_logits_flydsl
        real_score = qsa.qsa_paged_mqa_logits
        real_topk = qsa.top_k_per_row_prefill
        real_expand = qsa.qsa_expand_block_indices
        real_gate = paged.qsa_indexer_selection_supported
        self._saved = [
            (qsa, "qsa_paged_mqa_logits_flydsl", real_flydsl),
            (qsa, "qsa_paged_mqa_logits", real_score),
            (qsa, "top_k_per_row_prefill", real_topk),
            (qsa, "qsa_expand_block_indices", real_expand),
            (paged, "qsa_indexer_selection_supported", real_gate),
        ]
        probe = self

        def flydsl(*args, **kwargs):
            probe.flydsl_chunks += 1
            return real_flydsl(*args, **kwargs)

        def score(*args, **kwargs):
            def run():
                logits, visible = real_score(*args, **kwargs)
                if probe.check_blocks:
                    probe._logits = logits
                return logits, visible

            return probe._timed("score", run)

        def topk(*args, **kwargs):
            def run():
                real_topk(*args, **kwargs)
                probe.loop_chunks += 1
                if probe.check_blocks:
                    logits = probe._logits
                    probe._logits = None
                    if logits is None:
                        raise AssertionError("top-k ran without a score chunk")
                    _assert_top512(args[3], logits, args[1], args[2])

            return probe._timed("topk", run)

        def expand(*args, **kwargs):
            def run():
                probe.expand_chunks += 1
                return real_expand(*args, **kwargs)

            return probe._timed("expand", run)

        try:
            qsa.qsa_paged_mqa_logits_flydsl = flydsl
            qsa.qsa_paged_mqa_logits = score
            qsa.top_k_per_row_prefill = topk
            qsa.qsa_expand_block_indices = expand
            paged.qsa_indexer_selection_supported = _reject_indexer
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for owner, name, original in self._saved:
            setattr(owner, name, original)
        self._saved = []
        self._logits = None

    def _timed(self, stage: str, run):
        if not self.time_stages:
            return run()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = run()
        end.record()
        self.events[stage].append((start, end))
        return out

    def totals_us(self) -> dict[str, float]:
        torch.cuda.synchronize()
        totals = {}
        for stage, pairs in self.events.items():
            totals[stage] = (
                sum(start.elapsed_time(end) for start, end in pairs) * 1000.0
            )
        return totals


def _assert_top512(
    selected: torch.Tensor,
    logits: torch.Tensor,
    row_starts: torch.Tensor,
    row_ends: torch.Tensor,
) -> None:
    """Selected ids are a top-``BLOCK_TOPK`` of ``logits`` on ``[start, end)``.

    Ids scoring above the kth score are required. Ids scoring below it are
    rejected. Ids tied with the kth score may be either side of the cut.
    """
    rows, k = selected.shape
    if k != BLOCK_TOPK or logits.shape[0] != rows:
        raise AssertionError(
            f"selected {tuple(selected.shape)} vs logits {tuple(logits.shape)}"
        )
    cols = logits.shape[1]
    column = torch.arange(cols, device=logits.device)
    window = (column >= row_starts.to(torch.int64).unsqueeze(1)) & (
        column < row_ends.to(torch.int64).unsqueeze(1)
    )
    scores = logits.masked_fill(~window, float("-inf"))
    finite = torch.isfinite(scores)
    if not torch.equal(finite, window):
        raise AssertionError("visible window holds a non-finite score")
    if int(window.sum(dim=1).min().item()) < k:
        raise AssertionError("visible window is shorter than 512")
    kth = torch.topk(scores, k, dim=1).values[:, -1]
    chosen = selected.to(torch.int64)
    if not torch.all((chosen >= 0) & (chosen < cols)):
        raise AssertionError("selected block id is out of range")
    ordered = torch.sort(chosen, dim=1).values
    if torch.any(ordered[:, 1:] == ordered[:, :-1]):
        raise AssertionError("selected block ids repeat in a row")
    if not torch.all(torch.gather(window, 1, chosen)):
        raise AssertionError("selected block is outside the visible window")
    chosen_score = torch.gather(scores, 1, chosen)
    below = chosen_score < kth.unsqueeze(1)
    if torch.any(below):
        bad = int(below.any(dim=1).sum().item())
        raise AssertionError(f"{bad} rows selected a block below the 512th score")
    better = scores > kth.unsqueeze(1)
    if not torch.equal(better.sum(dim=1), (chosen_score > kth.unsqueeze(1)).sum(dim=1)):
        missed = int(
            (better.sum(dim=1) != (chosen_score > kth.unsqueeze(1)).sum(dim=1))
            .sum()
            .item()
        )
        raise AssertionError(f"{missed} rows missed a block above the 512th score")


def run_select(inputs: Inputs, probe: _Probe) -> torch.Tensor:
    with probe:
        return qsa_select_paged_tokens(
            inputs.q,
            inputs.k_cache,
            inputs.page_table,
            inputs.token_to_request,
            inputs.query_positions,
            inputs.context_lens,
            TOKEN_TOPK,
            COMPRESS_RATIO,
            allow_flydsl=True,
        )


def _times_of(probe: _Probe) -> StepTimes:
    totals = probe.totals_us()
    return StepTimes(
        score_us=totals["score"],
        topk_us=totals["topk"],
        expand_us=totals["expand"],
        flydsl_chunks=probe.flydsl_chunks,
        loop_chunks=probe.loop_chunks,
    )


def _require_loop(name: str, geo: Geometry, probe: _Probe) -> None:
    if probe.loop_chunks != geo.chunks or probe.expand_chunks != geo.chunks:
        raise AssertionError(
            f"{name}: top-k chunks {probe.loop_chunks}, expand chunks "
            f"{probe.expand_chunks}, expected {geo.chunks}"
        )
    if name in SMOKE_CASES and probe.flydsl_chunks != geo.chunks:
        raise AssertionError(
            f"{name}: FlyDSL score chunks {probe.flydsl_chunks} != {geo.chunks}"
        )


def run_case(name: str, requests: tuple[Request, ...], warmup: int, iters: int) -> dict:
    geo = geometry(requests)
    inputs = build_inputs(requests, seed=list(CASES).index(name))
    written = geo.score_bytes
    read = geo.score_bytes
    print(
        f"{name}: chunks={geo.chunks} rows_per_chunk={geo.rows_per_chunk} "
        f"requests={geo.requests} rows={geo.rows} columns={geo.columns} "
        f"score_bytes_written={written} score_bytes_read={read} "
        f"unique_k_bytes={geo.unique_k_bytes}",
        flush=True,
    )

    smoke = "n/a"
    last_flydsl = 0
    if name in SMOKE_CASES:
        probe = _Probe(time_stages=False, check_blocks=True)
        run_select(inputs, probe)
        _require_loop(name, geo, probe)
        last_flydsl = probe.flydsl_chunks
        smoke = "pass"
        print(
            f"{name}: smoke pass (fp32 torch top-512, ties at the 512th allowed)",
            flush=True,
        )

    samples: list[StepTimes] = []
    for _ in range(warmup):
        probe = _Probe(time_stages=False, check_blocks=False)
        run_select(inputs, probe)
        _require_loop(name, geo, probe)
        last_flydsl = probe.flydsl_chunks
    for _ in range(iters):
        probe = _Probe(time_stages=True, check_blocks=False)
        run_select(inputs, probe)
        _require_loop(name, geo, probe)
        samples.append(_times_of(probe))
        last_flydsl = probe.flydsl_chunks

    def median(attr: str) -> float | None:
        if not samples:
            return None
        return statistics.median(getattr(step, attr) for step in samples)

    score_us = median("score_us")
    topk_us = median("topk_us")
    expand_us = median("expand_us")
    flydsl_chunks = samples[-1].flydsl_chunks if samples else last_flydsl
    return {
        "case": name,
        "chunks": geo.chunks,
        "score_bytes_written": written,
        "score_bytes_read": read,
        "flydsl_chunks": flydsl_chunks,
        "smoke": smoke,
        "T_score_us": score_us,
        "T_topk_us": topk_us,
        "T_expand_us": expand_us,
        "T_step_us": None if score_us is None else score_us + topk_us + expand_us,
    }


def _pci_bus_id(device: int = 0) -> str:
    libname = ctypes.util.find_library("amdhip64") or "libamdhip64.so"
    lib = ctypes.CDLL(libname)
    lib.hipDeviceGetPCIBusId.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    lib.hipDeviceGetPCIBusId.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(64)
    err = lib.hipDeviceGetPCIBusId(buf, 64, device)
    if err != 0:
        return f"hipDeviceGetPCIBusId error {err}"
    return buf.value.decode()


def report_device() -> None:
    props = torch.cuda.get_device_properties(0)
    arch_name = getattr(props, "gcnArchName", "")
    print(
        "DEVICE "
        f"get_gfx={get_gfx()} "
        f"get_gfx_runtime={get_gfx_runtime()} "
        f"pci={_pci_bus_id()} "
        f"name={props.name} "
        f"gcn={arch_name} "
        f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '')}",
        flush=True,
    )


def _cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.1f}"
    return str(value)


def print_summary(rows: list[dict]) -> None:
    header = (
        "case",
        "chunks",
        "score_bytes_written",
        "score_bytes_read",
        "flydsl_chunks",
        "smoke",
        "T_score_us",
        "T_topk_us",
        "T_expand_us",
        "T_step_us",
    )
    body = [[_cell(row[key]) for key in header] for row in rows]
    widths = [len(name) for name in header]
    for line in body:
        widths = [max(width, len(cell)) for width, cell in zip(widths, line)]

    def fmt(line: list[str]) -> str:
        return (
            "| "
            + " | ".join(cell.rjust(width) for cell, width in zip(line, widths))
            + " |"
        )

    print(fmt(list(header)))
    print("| " + " | ".join("-" * width for width in widths) + " |")
    for line in body:
        print(fmt(line))


def main() -> None:
    check_catalog()
    parser = argparse.ArgumentParser(description="SILOTIGER-1481 select-path probe")
    parser.add_argument(
        "--cases",
        nargs="*",
        default=list(CASES),
        choices=list(CASES),
        help="shapes to build; default is all six",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()
    if args.warmup < 0 or args.iters < 0:
        raise SystemExit("warmup and iters must be >= 0")

    report_device()
    arch = get_gfx_runtime()
    if arch not in SUPPORTED_GFX:
        print(f"{arch} is outside {SUPPORTED_GFX}; not running the probe", flush=True)
        raise SystemExit(2)

    rows = [run_case(name, CASES[name], args.warmup, args.iters) for name in args.cases]
    print_summary(rows)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("probe failed", flush=True)
        raise
    sys.stdout.flush()
