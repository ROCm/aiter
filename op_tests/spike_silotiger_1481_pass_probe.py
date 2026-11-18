# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""SILOTIGER-1481 phase-3 second-score probe.

Launches the unmodified logits scorer twice back to back on each shape.
Pass 2 is queued immediately after pass 1, with no sync between them, so it
sees pass 1's cache. Each pass sums CUDA-event time across the same chunks
as the phase-1 select loop. The reported number is the median of 10 after
5 warmup steps.

Launch from the repo root, in flydsl_venv, through gpuq::

    export GPUQ_USER=<owner>
    gpuq --num-gpus=1 /workspaces/aiter/flydsl_venv/bin/python \\
        op_tests/spike_silotiger_1481_pass_probe.py
"""

import ctypes
import ctypes.util
import os
import statistics

import spike_silotiger_1481_select_probe as shapes
import torch

from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.ops.qsa import qsa_paged_mqa_logits

WARMUP = 5
ITERS = 10
# Phase-1 medians. Option 1's estimate is the slowed scorer plus T_topk.
PHASE1 = {
    "s8k": {"step": 203.2, "topk": 61.7},
    "s32k": {"step": 937.5, "topk": 411.4},
    "s128k": {"step": 2453.5, "topk": 809.3, "score_slow": 1573.4},
    "s256k": {"step": 4753.7, "topk": 1539.8, "score_slow": 3262.1},
    "two128k": {"step": 2485.9, "topk": 828.5},
    "mix128k": {"step": 2847.4, "topk": 820.3},
}
DECISION = ("s128k", "s256k")


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


def _chunks(inputs, columns: int):
    rows = inputs.q.shape[0]
    rows_per_chunk = max(1, shapes.DEFAULT_LOGITS_WORKSPACE_BYTES // (columns * 4))
    for start in range(0, rows, rows_per_chunk):
        end = min(start + rows_per_chunk, rows)
        sl = slice(start, end)
        yield (
            inputs.q[sl],
            inputs.k_cache,
            inputs.page_table,
            inputs.token_to_request[sl],
            inputs.query_positions[sl],
            inputs.context_lens,
        )


def _score(chunk, columns: int) -> None:
    q, k_cache, page_table, token_to_request, query_positions, context_lens = chunk
    row_starts = torch.empty(q.shape[0], dtype=torch.int32, device=q.device)
    qsa_paged_mqa_logits(
        q,
        k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        compress_ratio=shapes.COMPRESS_RATIO,
        max_columns=columns,
        row_starts=row_starts,
        allow_flydsl=True,
    )


def _time(inputs, columns: int) -> tuple[float, float]:
    chunks = list(_chunks(inputs, columns))

    def once() -> tuple[float, float]:
        recorded = []
        for _pass in range(2):
            pairs = []
            for chunk in chunks:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                _score(chunk, columns)
                end.record()
                pairs.append((start, end))
            recorded.append(pairs)
        torch.cuda.synchronize()
        totals = []
        for pairs in recorded:
            totals.append(sum(start.elapsed_time(end) for start, end in pairs) * 1000.0)
        return totals[0], totals[1]

    for _ in range(WARMUP):
        once()
    samples = [once() for _ in range(ITERS)]
    return (
        statistics.median(sample[0] for sample in samples),
        statistics.median(sample[1] for sample in samples),
    )


def main() -> None:
    props = torch.cuda.get_device_properties(0)
    print(
        "DEVICE "
        f"get_gfx={get_gfx()} get_gfx_runtime={get_gfx_runtime()} "
        f"pci={_pci_bus_id()} name={props.name} "
        f"gcn={getattr(props, 'gcnArchName', '')} "
        f"HIP_VISIBLE_DEVICES={os.environ.get('HIP_VISIBLE_DEVICES', '')}",
        flush=True,
    )
    if get_gfx_runtime() not in shapes.SUPPORTED_GFX:
        raise SystemExit(2)

    names = list(shapes.CASES)
    built = {
        name: shapes.build_inputs(shapes.CASES[name], seed=names.index(name))
        for name in names
    }
    columns = {name: shapes.geometry(shapes.CASES[name]).columns for name in names}

    measured = {}
    for name in names:
        print(f"{name}: timing two back-to-back score passes", flush=True)
        pass1, pass2 = _time(built[name], columns[name])
        measured[name] = (pass1, pass2)
        step = PHASE1[name]["step"]
        print(
            f"{name}: T_pass1 {pass1:.1f} us T_pass2 {pass2:.1f} us "
            f"sum {pass1 + pass2:.1f} us phase1 T_step {step:.1f} us "
            f"over={pass1 + pass2 > step}",
            flush=True,
        )

    over = []
    for name in DECISION:
        pass1, pass2 = measured[name]
        over.append(pass1 + pass2 > PHASE1[name]["step"])
    option2_out = all(over)
    print(f"RULE2 option2_out={option2_out}", flush=True)

    print(
        "case | T_pass1 us | T_pass2 us | sum us | phase1 T_step us | over",
        flush=True,
    )
    for name in names:
        pass1, pass2 = measured[name]
        step = PHASE1[name]["step"]
        print(
            f"{name} | {pass1:.1f} | {pass2:.1f} | {pass1 + pass2:.1f} | "
            f"{step:.1f} | {pass1 + pass2 > step}",
            flush=True,
        )

    # Rule 5 compares full step estimates. Histogram cost is not in these yet.
    print(
        "pre-histogram estimates (option1 = slowed score + phase1 T_topk)", flush=True
    )
    close = []
    for name in DECISION:
        est1 = PHASE1[name]["score_slow"] + PHASE1[name]["topk"]
        est2 = measured[name][0] + measured[name][1]
        step = PHASE1[name]["step"]
        gap = abs(est1 - est2)
        close.append(gap <= 0.10 * step)
        print(
            f"{name}: est1 {est1:.1f} us est2 {est2:.1f} us "
            f"|est1-est2|/T_step {gap / step:.4f} within_10pct_of_step={gap <= 0.10 * step}",
            flush=True,
        )
    print(
        f"RULE5 histogram_due={all(close) and not option2_out} "
        f"estimates_within_10pct_of_T_step={all(close)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
