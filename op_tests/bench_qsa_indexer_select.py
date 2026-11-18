# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""SILOTIGER-1481 phase-0 matched baseline.

Times aiter#5996 ``qsa_indexer_block_ids`` plus ``qsa_expand_block_indices``
on the ticket's geometry: compressed page 392, page tables padded to
``max_model_len`` 262144, and an indexer cache whose span exceeds 4 GiB.
Sparse GQA (Flash-Next TP2: 12 heads, dim 256, 1 KV head) is timed on the
expanded ids and is not part of ``T_step``.

A 1M prompt is not built. ``max_model_len`` is 262144.

This script times arm A only. It does not launch the vLLM arms.

``--skip-gqa`` compiles and times the indexer and expand only. The sparse
GQA kernel does not compile for ``mix64``, and that launch is not part of
arm A.

Launch from the repo root, in flydsl_venv, through gpuq. Compile first::

    gpuq --num-gpus=1 /workspaces/aiter/flydsl_venv/bin/python \\
        op_tests/bench_qsa_indexer_select.py --compile-only --skip-gqa

Then queue the timed run. Kernels must already be compiled.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
import re
import statistics
import subprocess
from dataclasses import dataclass
from typing import Self

import torch

from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.ops.flydsl import qsa_paged as paged
from aiter.ops.qsa import qsa_expand_block_indices

WARMUP = 5
ITERS = 10
REPEATS = 2
MAX_MODEL_LEN = 262144
PAGE = 392
PAD_PAGES = 168  # 168 * 392 = 65856 columns
COLUMNS = PAD_PAGES * PAGE
HEADS = 4
DIM = 128
COMPRESS = 4
TOKEN_TOPK = 2048
GQA_HEADS = 12
GQA_DIM = 256
GQA_PAGE = 16
CHUNK_ROWS = 16384
# Larger than the 256 MiB last-level cache the spike used.
FLUSH_BYTES = 1 << 30
SUPPORTED_GFX = ("gfx942", "gfx950")


@dataclass(frozen=True)
class Request:
    rows: int
    context: int
    pos0: int


@dataclass(frozen=True)
class Case:
    name: str
    requests: tuple[Request, ...]


def _tail(rows: int, context: int) -> Request:
    return Request(rows, context, context - rows)


def _cases() -> tuple[Case, ...]:
    """One step per case. A prompt longer than one chunk uses its last chunk."""
    c128 = 131072
    return (
        Case("p8k", (_tail(8192, 8192),)),
        Case("p32k", (_tail(CHUNK_ROWS, 32768),)),
        Case("p128k", (_tail(CHUNK_ROWS, c128),)),
        Case("p256k", (_tail(CHUNK_ROWS, MAX_MODEL_LEN),)),
        Case("two128k", (_tail(8192, c128), _tail(8192, c128))),
        Case(
            "four128k",
            tuple(_tail(4096, c128) for _ in range(4)),
        ),
        Case(
            "mix3",
            tuple(Request(1, c128, c128 - 1) for _ in range(3))
            + (_tail(CHUNK_ROWS - 3, c128),),
        ),
        Case(
            "mix16",
            tuple(Request(1, c128, c128 - 1) for _ in range(16))
            + (_tail(CHUNK_ROWS - 16, c128),),
        ),
        Case(
            "mix64",
            tuple(Request(1, c128, c128 - 1) for _ in range(64))
            + (_tail(CHUNK_ROWS - 64, c128),),
        ),
        Case(
            "mtp",
            (Request(4, c128, c128 - 4), _tail(CHUNK_ROWS - 4, c128)),
        ),
    )


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


def _wide_indexer_cache() -> torch.Tensor:
    """One indexer layer as a view of a block pool, spanning past 4 GiB.

    vLLM binds the compressed indexer cache as ``[blocks, page, 1, D]`` taken
    from a larger pool. The page stride is the pool pitch, not the packed
    page, and that span is what selects the wide gather. A contiguous tensor
    of the same shape stays on the dense specialization even when its numel
    passes 4 GiB. 8192 pages covers the bench's largest live set (mix64).
    """
    n_pages = 8192
    page_elems = PAGE * DIM
    gaps = n_pages - 1
    # One past the last element must sit beyond 4 GiB. bf16 is 2 bytes.
    need = (1 << 32) // 2 + 1
    page_stride = max(page_elems, (need - page_elems + gaps - 1) // gaps)
    if page_stride % 8:
        page_stride += 8 - (page_stride % 8)
    parent = torch.empty(
        gaps * page_stride + page_elems, dtype=torch.bfloat16, device="cuda"
    )
    return parent.as_strided(
        (n_pages, PAGE, 1, DIM),
        (page_stride, DIM, DIM, 1),
    )


class _StageProbe:
    """Split the indexer's internal score and select launches."""

    def __init__(self) -> None:
        self.score: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.topk: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._score_ids: set[int] = set()
        self._saved: list[tuple[object, str, object]] = []

    def __enter__(self) -> Self:
        probe = self

        def mark(fn):
            def wrapped(*args, **kwargs):
                exe = fn(*args, **kwargs)
                probe._score_ids.add(id(exe))
                return exe

            return wrapped

        def run_compiled(exe, *args, **kwargs):
            bucket = probe.score if id(exe) in probe._score_ids else probe.topk
            return probe._bracket(bucket, real_run, exe, *args, **kwargs)

        def topk(*args, **kwargs):
            return probe._bracket(probe.topk, real_topk, *args, **kwargs)

        real_run = paged._run_compiled
        real_topk = paged.flydsl_top_k_per_row_decode
        real_prefill = paged._prefill_scores_plan
        real_decode = paged._scores_plan
        self._saved = [
            (paged, "_run_compiled", real_run),
            (paged, "flydsl_top_k_per_row_decode", real_topk),
            (paged, "_prefill_scores_plan", real_prefill),
            (paged, "_scores_plan", real_decode),
        ]
        paged._prefill_scores_plan = mark(real_prefill)
        paged._scores_plan = mark(real_decode)
        paged._run_compiled = run_compiled
        paged.flydsl_top_k_per_row_decode = topk
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for owner, name, original in self._saved:
            setattr(owner, name, original)
        self._saved = []

    def reset(self) -> None:
        self.score.clear()
        self.topk.clear()

    @staticmethod
    def _bracket(bucket, fn, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = fn(*args, **kwargs)
        end.record()
        bucket.append((start, end))
        return out

    @staticmethod
    def _us(pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]]) -> float:
        return sum(start.elapsed_time(end) for start, end in pairs) * 1000.0


def _build(case: Case, index_k: torch.Tensor):
    rows = sum(req.rows for req in case.requests)
    n_req = len(case.requests)
    q = torch.randn(rows, HEADS, DIM, dtype=torch.bfloat16, device="cuda")
    q_gqa = torch.randn(rows, GQA_HEADS, GQA_DIM, dtype=torch.bfloat16, device="cuda")
    table = torch.zeros(n_req, PAD_PAGES, dtype=torch.int32, device="cuda")
    token_to_req = torch.empty(rows, dtype=torch.int32, device="cuda")
    positions = torch.empty(rows, dtype=torch.int32, device="cuda")
    context = torch.empty(n_req, dtype=torch.int32, device="cuda")
    gqa_pages = 0
    cursor = 0
    physical = 0
    for req_id, req in enumerate(case.requests):
        if req.context > MAX_MODEL_LEN or req.pos0 < 0 or req.rows < 1:
            raise ValueError(f"{case.name} request {req_id} does not fit")
        live_cols = req.context // COMPRESS
        live_pages = (live_cols + PAGE - 1) // PAGE
        if physical + live_pages >= index_k.shape[0]:
            raise ValueError(f"{case.name} indexer pool cannot hold the live pages")
        if live_pages:
            table[req_id, :live_pages] = torch.arange(
                physical, physical + live_pages, dtype=torch.int32, device="cuda"
            )
        physical += live_pages
        token_to_req[cursor : cursor + req.rows] = req_id
        positions[cursor : cursor + req.rows] = torch.arange(
            req.pos0, req.pos0 + req.rows, dtype=torch.int32, device="cuda"
        )
        context[req_id] = req.context
        gqa_pages += (req.context + GQA_PAGE - 1) // GQA_PAGE
        cursor += req.rows
    # Uninitialized live pages made top-k several times slower than filled
    # pages. Fill the pages this case's table names.
    if physical:
        index_k[:physical].copy_(
            torch.randn(
                physical,
                *index_k.shape[1:],
                dtype=index_k.dtype,
                device=index_k.device,
            )
        )
    gqa_width = max((req.context + GQA_PAGE - 1) // GQA_PAGE for req in case.requests)
    kv = torch.empty(
        gqa_pages, GQA_PAGE, 1, GQA_DIM, dtype=torch.bfloat16, device="cuda"
    )
    gqa_table = torch.zeros(n_req, gqa_width, dtype=torch.int32, device="cuda")
    base = 0
    for req_id, req in enumerate(case.requests):
        pages = (req.context + GQA_PAGE - 1) // GQA_PAGE
        gqa_table[req_id, :pages] = torch.arange(
            base, base + pages, dtype=torch.int32, device="cuda"
        )
        base += pages
    block_ids = torch.empty(rows, 512, dtype=torch.int32, device="cuda")
    token_ids = torch.empty(
        rows, TOKEN_TOPK + COMPRESS - 1, dtype=torch.int32, device="cuda"
    )
    return (
        q,
        q_gqa,
        table,
        token_to_req,
        positions,
        context,
        kv,
        gqa_table,
        block_ids,
        token_ids,
    )


def _flush(buf: torch.Tensor) -> None:
    buf.fill_(1)
    torch.cuda.synchronize()


def _time_case(
    case: Case,
    built,
    index_k: torch.Tensor,
    flush: torch.Tensor | None,
    run_gqa: bool,
) -> dict:
    (
        q,
        q_gqa,
        table,
        token_to_req,
        positions,
        context,
        kv,
        gqa_table,
        block_ids,
        token_ids,
    ) = built
    chunks = (
        q.shape[0] + paged._score_workspace_rows(COLUMNS) - 1
    ) // paged._score_workspace_rows(COLUMNS)
    score_bytes = q.shape[0] * COLUMNS * 4

    def once(probe: _StageProbe) -> tuple[float, float, float, float]:
        probe.reset()
        if flush is not None:
            _flush(flush)
        paged.qsa_indexer_block_ids(
            q,
            index_k,
            table,
            token_to_req,
            positions,
            context,
            out=block_ids,
            heads=(HEADS,),
        )
        expand_start = torch.cuda.Event(enable_timing=True)
        expand_end = torch.cuda.Event(enable_timing=True)
        expand_start.record()
        qsa_expand_block_indices(
            block_ids,
            positions,
            context,
            token_to_req,
            COMPRESS,
            TOKEN_TOPK,
            token_ids,
        )
        expand_end.record()
        if run_gqa:
            gqa_start = torch.cuda.Event(enable_timing=True)
            gqa_end = torch.cuda.Event(enable_timing=True)
            gqa_start.record()
            paged.flydsl_qsa_paged_gqa(
                q_gqa, kv, kv, token_ids, gqa_table, token_to_req
            )
            gqa_end.record()
        torch.cuda.synchronize()
        gqa_us = gqa_start.elapsed_time(gqa_end) * 1000.0 if run_gqa else float("nan")
        return (
            probe._us(probe.score),
            probe._us(probe.topk),
            expand_start.elapsed_time(expand_end) * 1000.0,
            gqa_us,
        )

    with _StageProbe() as probe:
        for _ in range(WARMUP):
            once(probe)
        samples = [once(probe) for _ in range(ITERS)]
    medians = [statistics.median(sample[i] for sample in samples) for i in range(4)]
    score, topk, expand, gqa = medians
    step = score + topk + expand
    return {
        "chunks": chunks,
        "score_bytes": score_bytes,
        "score": score,
        "topk": topk,
        "expand": expand,
        "gqa": gqa,
        "step": step,
        "share": (topk + expand) / step if step else float("nan"),
    }


def _assert_idle() -> None:
    """Exit 86 when the gpuq-assigned GPU is already busy.

    ``hipDeviceGetPCIBusId`` on device 0 is this process's GPU. rocm-smi's
    index is a different numbering. This process's own pid is ignored.
    """
    bus = _pci_bus_id().lower()
    bus_txt = subprocess.check_output(["rocm-smi", "--showbus"], text=True)
    use_txt = subprocess.check_output(["rocm-smi", "--showuse"], text=True)
    pid_txt = subprocess.check_output(["rocm-smi", "--showpids"], text=True)
    buses = {
        found.lower(): int(idx)
        for idx, found in re.findall(r"GPU\[(\d+)\].*PCI Bus:\s*(\S+)", bus_txt)
    }
    if bus not in buses:
        raise SystemExit(f"assigned bus {bus} is not in rocm-smi")
    idx = buses[bus]
    uses = {
        int(gpu): int(use)
        for gpu, use in re.findall(r"GPU\[(\d+)\].*GPU use \(%\):\s*(\d+)", use_txt)
    }
    use = uses.get(idx)
    others = []
    for line in pid_txt.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit():
            continue
        pid = int(parts[0])
        gpus = [int(tok) for tok in parts[2].split(",") if tok.isdigit()]
        if idx in gpus and pid != os.getpid():
            others.append(pid)
    print(
        f"IDLE rocm={idx} bus={bus} use={use} other_pids={others or 'none'}",
        flush=True,
    )
    if use != 0 or others:
        raise SystemExit(86)


def _print_device() -> None:
    props = torch.cuda.get_device_properties(0)
    print(
        "DEVICE "
        f"get_gfx={get_gfx()} get_gfx_runtime={get_gfx_runtime()} "
        f"pci={_pci_bus_id()} name={props.name} "
        f"gcn={getattr(props, 'gcnArchName', '')} "
        f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '')}",
        flush=True,
    )


def _compile(cases: tuple[Case, ...], run_gqa: bool) -> None:
    index_k = _wide_indexer_cache()
    print(
        f"indexer pool pages={index_k.shape[0]} "
        f"stride={tuple(int(s) for s in index_k.stride())} "
        f"span_bytes={paged.span_bytes(index_k)}",
        flush=True,
    )
    for case in cases:
        print(f"{case.name}: compile launch", flush=True)
        built = _build(case, index_k)
        (
            q,
            q_gqa,
            table,
            token_to_req,
            positions,
            context,
            kv,
            gqa_table,
            block_ids,
            token_ids,
        ) = built
        paged.qsa_indexer_block_ids(
            q,
            index_k,
            table,
            token_to_req,
            positions,
            context,
            out=block_ids,
            heads=(HEADS,),
        )
        qsa_expand_block_indices(
            block_ids, positions, context, token_to_req, COMPRESS, TOKEN_TOPK, token_ids
        )
        if run_gqa:
            paged.flydsl_qsa_paged_gqa(
                q_gqa, kv, kv, token_ids, gqa_table, token_to_req
            )
        torch.cuda.synchronize()
        del built
    print("COMPILE_DONE", flush=True)


def _fmt_bytes(n: int) -> str:
    if n >= 1 << 30 and n % (1 << 30) == 0:
        return f"{n // (1 << 30)} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.1f} MiB"
    return str(n)


def main() -> None:
    parser = argparse.ArgumentParser(description="SILOTIGER-1481 phase-0 baseline")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Launch each case once so the kernels are compiled, then exit.",
    )
    parser.add_argument(
        "--skip-gqa",
        action="store_true",
        help="Do not launch sparse GQA. Arm A is the indexer and expand.",
    )
    args = parser.parse_args()
    run_gqa = not args.skip_gqa
    _print_device()
    _assert_idle()
    if get_gfx_runtime() not in SUPPORTED_GFX:
        raise SystemExit(2)
    print(
        "1M prompt skipped: max_model_len is 262144. "
        "This run is arm A only. "
        + (
            "Sparse GQA is not launched."
            if args.skip_gqa
            else "Sparse GQA is timed and is not part of T_step."
        ),
        flush=True,
    )
    cases = _cases()
    if args.compile_only:
        _compile(cases, run_gqa)
        return

    index_k = _wide_indexer_cache()
    flush = torch.empty(FLUSH_BYTES, dtype=torch.uint8, device="cuda")
    print(
        f"indexer pool pages={index_k.shape[0]} "
        f"stride={tuple(int(s) for s in index_k.stride())} "
        f"span_bytes={paged.span_bytes(index_k)}",
        flush=True,
    )
    rows = []
    for case in cases:
        built = _build(case, index_k)
        for repeat, cold in (("warm-1", False), ("warm-2", False), ("cold", True)):
            print(f"{case.name}: timing {repeat}", flush=True)
            got = _time_case(case, built, index_k, flush if cold else None, run_gqa)
            got["case"] = case.name
            got["repeat"] = repeat
            gqa_txt = f"{got['gqa']:.1f}" if got["gqa"] == got["gqa"] else "skipped"
            print(
                f"{case.name} {repeat}: T_score {got['score']:.1f} "
                f"T_topk {got['topk']:.1f} T_expand {got['expand']:.1f} "
                f"T_gqa {gqa_txt} T_step {got['step']:.1f} "
                f"share {got['share']:.3f} chunks {got['chunks']}",
                flush=True,
            )
            rows.append(got)
        del built
        torch.cuda.empty_cache()

    print(
        "case | arm | chunks | score bytes | T_score | T_topk | T_expand | "
        "T_step | top-k+expand share | T_gqa | repeat",
        flush=True,
    )
    for got in rows:
        gqa_txt = f"{got['gqa']:.1f}" if got["gqa"] == got["gqa"] else "skipped"
        print(
            f"{got['case']} | aiter#5996 | {got['chunks']} | {_fmt_bytes(got['score_bytes'])} | "
            f"{got['score']:.1f} | {got['topk']:.1f} | {got['expand']:.1f} | "
            f"{got['step']:.1f} | {got['share']:.3f} | {gqa_txt} | {got['repeat']}",
            flush=True,
        )
    decision = [
        got
        for got in rows
        if got["case"] in ("p128k", "p256k") and got["repeat"] == "warm-2"
    ]
    fired = all(got["share"] < 0.15 for got in decision)
    print(f"GATE5 profile_only_close={fired}", flush=True)


if __name__ == "__main__":
    main()
