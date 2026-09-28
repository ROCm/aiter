# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Checks for the FlyDSL GEMM family registry (``aiter.flydsl_gemm_registry``).

``python3`` this file. aiter op_tests are plain scripts, so every check is called
from ``main()``. The checks are generic over the registry: a newly registered
family is covered without editing this file.

  - every family's kernel names survive parse(name) for its own candidates;
  - every family that claims tuner support offers candidates on this GPU;
  - every FlyDSL row of the merged A16W16 tuned config for this GPU belongs to a
    registered family, is selected by the dispatcher, and a sample of rows runs
    through ``tgemm.mm`` against an FP32 reference.
"""

from __future__ import annotations

import collections

import torch

import aiter
from aiter import flydsl_gemm_registry as registry
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.tuned_gemm import get_GEMM_A16W16_config, get_GEMM_A16W16_config_, tgemm

ARCH = get_gfx_runtime()
A16W16 = registry.A16W16
MAX_REL_ERR = 0.02

# Small set of shapes that every A16W16 family should be able to offer
# candidates for on a GPU it supports (decode takes M<=5, hgemm any M).
PROBE_SHAPES = ((1, 896, 7168), (4, 1536, 7168), (64, 2112, 7168))


def _problem(m: int, n: int, k: int) -> registry.GemmProblem:
    return registry.GemmProblem(
        m=m,
        n=n,
        k=k,
        arch=ARCH,
        in_dtype=torch.bfloat16,
        out_dtype=torch.bfloat16,
        cu_num=get_cu_num(),
    )


def _families_here() -> list[registry.FlydslGemmFamily]:
    return registry.families(A16W16, ARCH)


def check_name_round_trip() -> None:
    """A candidate's kernel name finds its family again and parses back."""
    checked = 0
    for family in _families_here():
        if family.candidates is None:
            continue
        for shape in PROBE_SHAPES:
            for cand in family.candidates(_problem(*shape), "bounded")[:20]:
                found = registry.family_for_kernel(A16W16, cand.kernel_name, ARCH)
                assert found is family, (
                    f"{cand.kernel_name} resolves to "
                    f"{None if found is None else found.name}, not {family.name}"
                )
                params = family.parse(cand.kernel_name)
                assert (
                    params is not None
                ), f"{family.name} cannot parse {cand.kernel_name}"
                if isinstance(cand.params, dict) and isinstance(params, dict):
                    shared = set(cand.params) & set(params)
                    diff = {k for k in shared if cand.params[k] != params[k]}
                    assert (
                        not diff
                    ), f"{cand.kernel_name}: parse differs on {sorted(diff)}"
                checked += 1
    aiter.logger.info("name round trip: %d candidates", checked)


def check_families_offer_candidates() -> None:
    """A family that claims tuner support must offer something on this GPU."""
    for family in _families_here():
        if family.candidates is None:
            aiter.logger.info("%s: no tuner candidates (declared)", family.name)
            continue
        counts = [len(family.candidates(_problem(*s), "bounded")) for s in PROBE_SHAPES]
        assert any(counts), f"{family.name} offers no candidates on {ARCH}: {counts}"
        aiter.logger.info("%s: candidates per probe shape %s", family.name, counts)


def _flydsl_rows() -> list[tuple[tuple, dict]]:
    cu_num = get_cu_num()
    return sorted(
        (key, row)
        for key, row in get_GEMM_A16W16_config_().items()
        if row.get("libtype") == registry.FLYDSL_LIBTYPE
        and key[0] == ARCH
        and int(key[1]) == cu_num
    )


def check_tuned_rows_dispatch() -> None:
    """Every FlyDSL row for this GPU is registered, selected and runs."""
    rows = _flydsl_rows()
    if not rows:
        aiter.logger.warning("no FlyDSL A16W16 rows for %s; skipping", ARCH)
        return
    unregistered, dropped = [], []
    per_family = collections.Counter()
    sample = {}
    for key, row in rows:
        _, _, m, n, k, bias, dtype, otype, scale_ab, bpreshuffle = key
        family = registry.family_for_kernel(A16W16, row["kernelName"], ARCH)
        if family is None or family.parse(row["kernelName"]) is None:
            unregistered.append(row["kernelName"])
            continue
        per_family[family.name] += 1
        cfg = get_GEMM_A16W16_config(m, n, k, bias, dtype, otype, scale_ab, bpreshuffle)
        if cfg is None or cfg.get("kernelName") != row["kernelName"]:
            dropped.append(
                (m, n, k, family.name, None if cfg is None else cfg.get("libtype"))
            )
            continue
        # One cheap row per (family, M, bias, out dtype) to run end to end.
        slot = (family.name, m, bias, otype)
        if n * k <= 16384 * 7168 and (slot not in sample or n * k < sample[slot][0]):
            sample[slot] = (n * k, m, n, k, bias, otype)
    assert (
        not unregistered
    ), f"{len(unregistered)} FlyDSL rows name no registered family: {unregistered[:5]}"
    assert not dropped, (
        f"{len(dropped)} of {len(rows)} FlyDSL rows are not selected by the "
        f"dispatcher (M, N, K, family, got): {dropped[:10]}"
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
        dict(per_family),
        len(sample),
        worst,
    )


CHECKS = (
    check_name_round_trip,
    check_families_offer_candidates,
    check_tuned_rows_dispatch,
)


def main() -> None:
    aiter.logger.info(
        "families on %s: %s", ARCH, [family.name for family in _families_here()]
    )
    for check in CHECKS:
        aiter.logger.info("running %s", check.__name__)
        check()


if __name__ == "__main__":
    main()
