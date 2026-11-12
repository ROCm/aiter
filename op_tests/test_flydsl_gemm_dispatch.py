# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Dispatch checks for the FlyDSL A16W16 rows (hgemm and decode).

``python3`` this file. aiter op_tests are plain scripts, so every check is called
from ``main()``.

  - decode tuner candidates parse back to the config they were named from;
  - every FlyDSL row of the merged A16W16 tuned config for this GPU is selected
    by the dispatcher, and a sample of rows runs through ``tgemm.mm`` against an
    FP32 reference.
"""

from __future__ import annotations

import collections

import torch

import aiter
from aiter.jit.utils.chip_info import get_cu_num, get_gfx
from aiter.ops.flydsl.gemm_a16w16_policy import get_flydsl_a16w16_decode_configs
from aiter.ops.flydsl.gemm_kernels import get_flydsl_decode_kernel_params
from aiter.tuned_gemm import (
    get_GEMM_A16W16_config,
    get_GEMM_A16W16_config_,
    is_flydsl_decode_config,
    tgemm,
)

ARCH = get_gfx()
MAX_REL_ERR = 0.02
PROBE_SHAPES = ((1, 896, 7168), (4, 1536, 7168), (5, 2112, 7168))


def check_decode_name_round_trip() -> None:
    """A decode candidate's kernel name parses back to its own config."""
    checked = 0
    for m, n, k in PROBE_SHAPES:
        for has_bias in (False, True):
            candidates = get_flydsl_a16w16_decode_configs(
                m, n, k, torch.bfloat16, has_bias
            )
            if ARCH in ("gfx942", "gfx950"):
                assert candidates, f"no decode candidates for M={m} ({n},{k})"
            for _, name, config in candidates:
                params = get_flydsl_decode_kernel_params(name)
                assert params is not None, f"cannot parse {name}"
                got = (params["arch"], params["m"], params["n"], params["k"])
                assert got == (ARCH, m, n, k), f"{name}: parsed {got}"
                assert params["config"] == config, f"{name}: config differs"
                assert params["has_bias"] == has_bias, f"{name}: bias differs"
                checked += 1
    aiter.logger.info("decode name round trip: %d candidates", checked)


def check_tuned_rows_dispatch() -> None:
    """Every FlyDSL row for this GPU is selected by the dispatcher and runs."""
    cu_num = get_cu_num()
    rows = sorted(
        (key, row)
        for key, row in get_GEMM_A16W16_config_().items()
        if row.get("libtype") == "flydsl" and key[0] == ARCH and int(key[1]) == cu_num
    )
    if not rows:
        aiter.logger.warning("no FlyDSL A16W16 rows for %s; skipping", ARCH)
        return
    dropped = []
    per_kind = collections.Counter()
    sample = {}
    for key, row in rows:
        _, _, m, n, k, bias, dtype, otype, scale_ab, bpreshuffle = key
        kind = "decode" if is_flydsl_decode_config(row) else "hgemm"
        per_kind[kind] += 1
        cfg = get_GEMM_A16W16_config(m, n, k, bias, dtype, otype, scale_ab, bpreshuffle)
        if cfg is None or cfg.get("kernelName") != row["kernelName"]:
            dropped.append((m, n, k, kind, None if cfg is None else cfg.get("libtype")))
            continue
        # One cheap row per (kind, M, bias, out dtype) to run end to end.
        slot = (kind, m, bias, otype)
        if n * k <= 16384 * 7168 and (slot not in sample or n * k < sample[slot][0]):
            sample[slot] = (n * k, m, n, k, bias, otype)
    assert not dropped, (
        f"{len(dropped)} of {len(rows)} FlyDSL rows are not selected by the "
        f"dispatcher (M, N, K, kind, got): {dropped[:10]}"
    )

    torch.manual_seed(0)
    worst = 0.0
    for _, m, n, k, bias, otype in sample.values():
        a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        b = torch.randn(n, device="cuda", dtype=torch.bfloat16) if bias else None
        out = tgemm.mm(a, w, b, otype=eval(otype))
        ref = a.float() @ w.float().T + (0 if b is None else b.float())
        err = ((out.float() - ref).abs().max() / ref.abs().max()).item()
        assert err < MAX_REL_ERR, f"M={m} ({n},{k}) bias={bias}: rel err {err:.2e}"
        worst = max(worst, err)
    aiter.logger.info(
        "dispatcher selected all %d FlyDSL rows %s; ran %d, worst rel err %.1e",
        len(rows),
        dict(per_kind),
        len(sample),
        worst,
    )


CHECKS = (check_decode_name_round_trip, check_tuned_rows_dispatch)


def main() -> None:
    for check in CHECKS:
        aiter.logger.info("running %s", check.__name__)
        check()


if __name__ == "__main__":
    main()
