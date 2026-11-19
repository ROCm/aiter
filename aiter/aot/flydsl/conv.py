#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT for the FlyDSL implicit-GEMM convolution.

Each tuned row is replayed through ``flydsl_conv_implicit`` itself on
FakeTensors inside :func:`aiter.aot.flydsl.dry_run.dry_run` (see
``replay.py``): the runtime looks the row up, pads channels, resolves split-K
and the variable-resolution (``AITER_CONV3D_DYN_HW``) path, and compiles the
conv kernel and the NCDHW->NDHWC pre-transpose it launches first. The output
layout is part of the compile key and the caller's choice, so each row is
replayed with both; an NCDHW input reaches the pre-transpose, and an NDHWC
input the same conv kernel without it.

``AITER_CONV3D_DYN_HW`` is read by the replayed runtime, so build and
serving agree as long as both see the same value.

Usage::

    python -m aiter.aot.flydsl.conv
    python -m aiter.aot.flydsl.conv --csv /path/to/bf16_tuned_conv3d.csv
    python -m aiter.aot.flydsl.conv --check
"""

from __future__ import annotations

import csv
import os

from aiter.aot.flydsl import replay as _replay
from aiter.jit.core import AITER_CONFIGS

LABEL = "CONV"
DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_CONV3D_BF16_FILE]
AOT_ARCH_DEFAULT = "gfx950"

_SHAPE_COLS = (
    "N",
    "C",
    "D",
    "H",
    "W",
    "K",
    "kT",
    "kH",
    "kW",
    "stride_d",
    "stride_h",
    "stride_w",
    "pad_d",
    "pad_h",
    "pad_w",
    "dil_d",
    "dil_h",
    "dil_w",
    "groups",
)
_OUTPUT_LAYOUTS = ("NCDHW", "NDHWC")


def parse_csv(csv_path: str) -> list[dict]:
    """One job per FlyDSL row of a tuned conv table and output layout."""
    from aiter.ops.flydsl.conv_kernels import (
        LIBTYPE_FLYDSL,
        TUNED_LIBTYPE_COLUMN,
        _parse_tuned_bool,
    )

    jobs = []
    with open(csv_path, newline="") as f:
        for raw in csv.DictReader(f):
            row = {k.strip(): (v or "").strip() for k, v in raw.items() if k}
            # Empty libtype means FlyDSL (pre-libtype tables).
            libtype = row.get(TUNED_LIBTYPE_COLUMN, "")
            if libtype and libtype != LIBTYPE_FLYDSL:
                continue
            try:
                shape = {c: int(row[c]) for c in _SHAPE_COLS}
                has_bias = _parse_tuned_bool(row.get("bias"))
            except (KeyError, ValueError) as exc:
                print(f"  [WARN] {csv_path}: unparsable row ({exc}), skipping")
                continue
            for layout in _OUTPUT_LAYOUTS:
                jobs.append(
                    {
                        "kernel_name": row.get("kernelName") or "conv3d_implicit",
                        "gfx": row.get("gfx", ""),
                        "cu_num": int(row.get("cu_num") or 0),
                        "bias": has_bias,
                        "output_layout": layout,
                        **shape,
                    }
                )
    return jobs


def replay(job: dict) -> None:
    import torch

    from aiter.ops.flydsl.conv_kernels import flydsl_conv_implicit

    def empty(*shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="cuda")

    c, k, groups = job["C"], job["K"], job["groups"]
    flydsl_conv_implicit(
        empty(job["N"], c, job["D"], job["H"], job["W"]),
        empty(k, c // groups, job["kT"], job["kH"], job["kW"]),
        bias=empty(k) if job["bias"] else None,
        stride=(job["stride_d"], job["stride_h"], job["stride_w"]),
        padding=(job["pad_d"], job["pad_h"], job["pad_w"]),
        dilation=(job["dil_d"], job["dil_h"], job["dil_w"]),
        groups=groups,
        input_layout="NCDHW",
        output_layout=job["output_layout"],
    )


def describe(job: dict) -> str:
    shape = "x".join(str(job[c]) for c in ("N", "C", "D", "H", "W"))
    kernel = "x".join(str(job[c]) for c in ("K", "kT", "kH", "kW"))
    return (
        f"conv3d in={shape} w={kernel} stride=({job['stride_d']},{job['stride_h']},"
        f"{job['stride_w']}) groups={job['groups']} bias={job['bias']} "
        f"out={job['output_layout']}"
    )


def config_env(csv_paths: list[str]) -> dict[str, str]:
    return {"AITER_CONFIG_CONV3D_BF16": os.pathsep.join(csv_paths)}


def main() -> None:
    _replay.main(
        "aiter.aot.flydsl.conv",
        "AOT pre-compile the FlyDSL conv kernels of tuned conv3d configs",
    )


if __name__ == "__main__":
    main()
