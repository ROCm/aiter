#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT pre-compilation for the gfx950 hyper-connection seam.

``hc_seam_prepared`` picks its kernels per token count M and per (combine,
next gate) variant: two skinny kernels for M <= 3, three deferred launches up
to M = 128 whose K1/K2 geometry and structural knobs follow M, and four
pre-normalized launches above, whose K1/K2 geometry comes from the tuned seam
rows or the heuristic and whose reduce kernel is keyed by the padded row
extent. A serving process would JIT each combination on the request path. This
module replays the public entry with its default arguments (no knob overrides)
on FakeTensors inside :func:`aiter.aot.flydsl.dry_run.dry_run` (see
``replay.py``) for every (combine, next gate, M) in ``DEFAULT_ROWS``, plus one
``hc_gated_combine`` call. Only BF16 norm weights and the test's eps are
covered.

Usage:
    python -m aiter.aot.flydsl.hc_seam
    python -m aiter.aot.flydsl.hc_seam --check
"""

from __future__ import annotations

import argparse
import itertools

from aiter.aot.flydsl import replay as _replay
from aiter.aot.flydsl.common import dedupe_jobs

LABEL = "HC_SEAM"
# No tuned CSV: the seam reads hc_gated_residual_tuned.json at dispatch, and
# the jobs come from DEFAULT_ROWS.
DEFAULT_CSVS: list[str] = []
# hc_seam is gfx950-only; the job tag is the arch gate.
AOT_ARCH_DEFAULT = "gfx950"

# One M per distinct compile key reachable under default dispatch (see
# tuned.py). Edges of the selectors, M -> path:
#   1..3      skinny (chunk_m differs for odd/even M)
#   4..128    deferred; K1 geometry edges at 32 and 96, K2 edges at 32, 64, 96;
#             structural knobs on for 64..128 (combine and next gate); the
#             padded extent is 64 up to M=64 and 128 above
#   129..256  four-launch; tuned seam rows 128/192/256 (nearest in log space
#             switch at 156/221) and the heuristic tile (32 below 256, 64 at
#             256); the padded extent steps at 129, 161, 193, 225 for tile 32
#             and at 129, 193 for tile 64 (reduce keyed by padded * n_pad)
DEFAULT_ROWS = (
    1, 2, 3, 4, 16, 32, 33, 64, 65, 96, 97, 128,
    129, 160, 161, 192, 193, 224, 225, 256,
)  # fmt: skip
_COMBINE_M = 16
_HIDDEN = 10240
_LOWRANK = 320
_HC = 4
# The seam test's eps; part of the deferred reduce kernel's compile key.
_EPS = 1e-6


def seam_jobs() -> list[dict]:
    jobs = [
        {
            "kernel_name": f"hc_seam_c{combine}_n{next_gate}",
            "gfx": AOT_ARCH_DEFAULT,
            "op": "seam",
            "combine": combine,
            "next_gate": next_gate,
            "m": m,
        }
        for combine, next_gate, m in itertools.product((0, 1), (0, 1), DEFAULT_ROWS)
    ]
    jobs.append(
        {
            "kernel_name": "hc_gated_combine",
            "gfx": AOT_ARCH_DEFAULT,
            "op": "combine",
            "combine": 1,
            "next_gate": 0,
            "m": _COMBINE_M,
        }
    )
    return dedupe_jobs(jobs)


def replay(job: dict) -> None:
    import importlib
    from contextlib import ExitStack
    from unittest import mock

    import torch

    # aiter.ops.flydsl re-exports a function named hc_seam, which shadows the
    # submodule in a from-import.
    op = importlib.import_module("aiter.ops.flydsl.hc_seam")

    def empty(*shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="cuda")

    # Two limits of the dry run's FakeTensors, which the seam's host code meets:
    # it asks HIP for the current raw stream directly, and its overlap guards
    # compare byte spans of fake "addresses" that are storage ids a few
    # hundred bytes apart, so every pair of tensors looks overlapping.
    with ExitStack() as stack:
        stack.enter_context(
            mock.patch.object(torch._C, "_cuda_getCurrentRawStream", lambda _i: 0)
        )
        for name in ("k1", "k2"):
            module = importlib.import_module(
                f"aiter.ops.flydsl.kernels.hyper_connection_gated_residual.{name}"
            )
            stack.enter_context(
                mock.patch.object(module, "check_no_overlap", lambda *_a, **_k: None)
            )
        _call(job, op, empty)


def _call(job: dict, op, empty) -> None:
    import torch

    m = job["m"]
    residual = empty(m, _HIDDEN)
    y = empty(m, 2560)
    gate_prev = empty(m, _HC, dtype=torch.float32)
    if job["op"] == "combine":
        op.hc_gated_combine(y, residual, gate_prev)
        return
    # Packs the weights as a caller does once; the packed merged tensor's row
    # count (320 or 384) is part of the dispatch.
    prepared = op.hc_seam_prepare_weights(
        empty(_HIDDEN),
        empty(_LOWRANK, _HIDDEN),
        empty(_HIDDEN, _LOWRANK),
        empty(_HC, _HIDDEN) if job["next_gate"] else None,
    )
    combine = job["combine"]
    op.hc_seam_prepared(
        y if combine else None,
        residual,
        gate_prev if combine else None,
        prepared,
        _EPS,
    )


def describe(job: dict) -> str:
    if job["op"] == "combine":
        return f"hc_gated_combine M={job['m']}"
    return (
        f"hc_seam_prepared combine={job['combine']} "
        f"next_gate={job['next_gate']} M={job['m']}"
    )


def collect_jobs(_csv_paths: list[str], _args: argparse.Namespace | None) -> list:
    return seam_jobs()


def configure(_parser: argparse.ArgumentParser) -> None:
    """No options of its own."""


def main() -> None:
    _replay.main(
        "aiter.aot.flydsl.hc_seam",
        "AOT pre-compile the gfx950 hyper-connection seam kernels",
    )


if __name__ == "__main__":
    main()
