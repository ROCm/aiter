# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""SILOTIGER-1481 phase-2 LDS occupancy probe.

Times the spike scorer copy. Dummy 0 must match the unmodified logits on
``s32k`` and stay within 5% of that case's phase-1 ``T_score``. Dummy 32768
is timed on ``s128k`` and ``s256k``. Dummy 65536 is attempted on ``s128k``
and a launch failure is recorded rather than retried.

Launch from the repo root, in flydsl_venv, through gpuq::

    export GPUQ_USER=<owner>
    gpuq --num-gpus=1 /workspaces/aiter/flydsl_venv/bin/python \\
        op_tests/spike_silotiger_1481_lds_probe.py
"""

import ctypes
import ctypes.util
import os
import statistics
import traceback

import spike_silotiger_1481_select_probe as shapes
import torch

from aiter.jit.utils.chip_info import get_gfx, get_gfx_runtime
from aiter.ops.flydsl.kernels.mqa_logits.qsa_flydsl_layout import (
    qsa_paged_mqa_logits_flydsl,
)
from aiter.ops.flydsl.kernels.mqa_logits.qsa_flydsl_layout_lds_probe import (
    qsa_paged_mqa_logits_flydsl_lds_probe,
)

WARMUP = 5
ITERS = 10
DIVISOR = 128**0.5
# Phase-1 medians on the unmodified scorer. The budget is T_topk + T_expand.
PHASE1 = {
    "s32k": {"score": 394.1, "budget": 411.4 + 132.0},
    "s128k": {"score": 1461.4, "budget": 809.3 + 182.7},
    "s256k": {"score": 2972.5, "budget": 1539.8 + 241.5},
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
            rows_per_chunk,
        )


def _score(fn, chunk, columns: int, dummy_bytes: int | None):
    q, k_cache, page_table, token_to_request, query_positions, context_lens, _ = chunk
    rows = q.shape[0]
    logits = torch.empty((rows, columns), dtype=torch.float32, device=q.device)
    visible = torch.empty(rows, dtype=torch.int32, device=q.device)
    row_starts = torch.empty(rows, dtype=torch.int32, device=q.device)
    args = (
        q,
        k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        shapes.COMPRESS_RATIO,
        DIVISOR,
        logits,
        visible,
        row_starts,
    )
    if dummy_bytes is None:
        fn(*args)
    else:
        fn(*args, dummy_bytes=dummy_bytes)
    return logits, visible, row_starts


def _time(fn, inputs, columns: int, dummy_bytes: int | None) -> float:
    chunks = list(_chunks(inputs, columns))

    def once() -> float:
        pairs = []
        for chunk in chunks:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            _score(fn, chunk, columns, dummy_bytes)
            end.record()
            pairs.append((start, end))
        torch.cuda.synchronize()
        return sum(start.elapsed_time(end) for start, end in pairs) * 1000.0

    for _ in range(WARMUP):
        once()
    return statistics.median(once() for _ in range(ITERS))


def _match(name: str, inputs, columns: int) -> None:
    for index, chunk in enumerate(_chunks(inputs, columns)):
        ref, ref_vis, ref_starts = _score(
            qsa_paged_mqa_logits_flydsl, chunk, columns, None
        )
        got, got_vis, got_starts = _score(
            qsa_paged_mqa_logits_flydsl_lds_probe, chunk, columns, 0
        )
        if not torch.equal(ref_vis, got_vis):
            raise AssertionError(f"{name} chunk {index}: visible_groups differ")
        if not torch.equal(ref_starts, got_starts):
            raise AssertionError(f"{name} chunk {index}: row_starts differ")
        if not torch.equal(ref, got):
            gap = (ref - got).abs()
            finite = torch.isfinite(ref) & torch.isfinite(got)
            worst = (
                float(gap[finite].max().item()) if bool(finite.any()) else float("nan")
            )
            raise AssertionError(
                f"{name} chunk {index}: logits differ, max abs {worst}"
            )
    print(f"{name}: dummy 0 logits match the unmodified scorer", flush=True)


def _try_launch(inputs, columns: int, dummy_bytes: int) -> str:
    chunk = next(_chunks(inputs, columns))
    try:
        _score(qsa_paged_mqa_logits_flydsl_lds_probe, chunk, columns, dummy_bytes)
        torch.cuda.synchronize()
    except Exception as exc:
        print(
            f"dummy {dummy_bytes} launch failed: {type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        return f"fail: {type(exc).__name__}: {exc}"
    print(f"dummy {dummy_bytes} launch succeeded", flush=True)
    return "ok"


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

    built = {
        name: shapes.build_inputs(
            shapes.CASES[name], seed=list(shapes.CASES).index(name)
        )
        for name in ("s32k", "s128k", "s256k")
    }
    columns = {name: shapes.geometry(shapes.CASES[name]).columns for name in built}

    print("faithfulness: compiling dummy 0 and the unmodified scorer", flush=True)
    _match("s32k", built["s32k"], columns["s32k"])
    # Pair each copy with the unmodified scorer so the two see the same clocks.
    print("s32k: timing unmodified scorer", flush=True)
    unmodified = {
        "s32k": _time(qsa_paged_mqa_logits_flydsl, built["s32k"], columns["s32k"], None)
    }
    print(f"s32k: unmodified T_score {unmodified['s32k']:.1f} us", flush=True)
    print("s32k: timing dummy 0", flush=True)
    dummy0 = _time(
        qsa_paged_mqa_logits_flydsl_lds_probe, built["s32k"], columns["s32k"], 0
    )
    print(f"s32k: dummy 0 T_score {dummy0:.1f} us", flush=True)

    dummy32 = {}
    for name in ("s128k", "s256k"):
        print(f"{name}: timing unmodified scorer", flush=True)
        unmodified[name] = _time(
            qsa_paged_mqa_logits_flydsl, built[name], columns[name], None
        )
        print(f"{name}: unmodified T_score {unmodified[name]:.1f} us", flush=True)
        print(f"{name}: timing dummy 32768", flush=True)
        dummy32[name] = _time(
            qsa_paged_mqa_logits_flydsl_lds_probe, built[name], columns[name], 32768
        )
        print(f"{name}: dummy 32768 T_score {dummy32[name]:.1f} us", flush=True)

    print("s128k: attempting dummy 65536", flush=True)
    launch_64 = _try_launch(built["s128k"], columns["s128k"], 65536)
    dummy64 = None
    if launch_64 == "ok":
        print("s128k: timing dummy 65536", flush=True)
        dummy64 = _time(
            qsa_paged_mqa_logits_flydsl_lds_probe,
            built["s128k"],
            columns["s128k"],
            65536,
        )
        print(f"s128k: dummy 65536 T_score {dummy64:.1f} us", flush=True)

    phase1 = PHASE1["s32k"]["score"]
    ratio_phase1 = dummy0 / phase1
    ratio_session = dummy0 / unmodified["s32k"]
    faithful = abs(ratio_phase1 - 1.0) <= 0.05 and abs(ratio_session - 1.0) <= 0.05
    print(
        f"s32k faithfulness: dummy0/phase1={ratio_phase1:.4f} "
        f"dummy0/session={ratio_session:.4f} faithful={faithful}",
        flush=True,
    )
    print(
        "case | unmodified us | dummy 0 us | dummy 32KiB us | dummy 64KiB | "
        "delta 32KiB us | phase1 budget us | option1 out",
        flush=True,
    )
    option1_out = False
    for name in ("s32k", "s128k", "s256k"):
        got32 = dummy32.get(name)
        delta = None if got32 is None else got32 - unmodified[name]
        budget = PHASE1[name]["budget"]
        eliminated = bool(delta is not None and delta > budget)
        if name in ("s128k", "s256k") and eliminated:
            option1_out = True
        print(
            f"{name} | {unmodified[name]:.1f} | "
            f"{f'{dummy0:.1f}' if name == 's32k' else ''} | "
            f"{'' if got32 is None else f'{got32:.1f}'} | "
            f"{launch_64 if name == 's128k' else ''} | "
            f"{'' if delta is None else f'{delta:.1f}'} | {budget:.1f} | {eliminated}",
            flush=True,
        )
    if dummy64 is not None:
        print(f"s128k dummy 64KiB T_score {dummy64:.1f} us", flush=True)
    print(f"RULE3 option1_out={option1_out} faithful={faithful}", flush=True)


if __name__ == "__main__":
    main()
