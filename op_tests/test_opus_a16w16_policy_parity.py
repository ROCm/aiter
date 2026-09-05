# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only parity coverage for the migrated OPUS A16W16 policy."""

from __future__ import annotations

from functools import cache
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from aiter.ops.opus import policy
from csrc.opus_gemm.opus_gemm_common import (
    GFX942_BF16WS_EXACT_N,
    get_kernel_instance,
)

# These are literal control-flow translations of the three headers at the
# pre-PR merge base. Keep this reference independent of policy.py.
_PRE_PR_REF = "ded4e3e8eee11f56853054c4ed4bdf2790545e5d"
_PRE_PR_HEADERS = {
    "gfx950": "4c8f03542e4459b51a17c0bd9fe224533af0c594",
    "gfx942": "7906b9ba0e32a1c89d0752961f321fb9b6c26dd9",
    "gfx1250": "ff4b83daa5e074d8cc2ebdc3c29e1d8cfaa8e02c",
}


def _pre_pr_gfx950(M: int, N: int, K: int, has_bias: bool, _output: str) -> int:
    split_barrier_ok = N % 16 == 0 and K % 64 == 0 and (K // 64) % 2 == 0
    if M <= 4:
        if M % 64 == 0 and N % 64 == 0 and K % 128 == 0:
            return 1208
        return 208
    if M <= 64:
        if M % 64 == 0 and N % 32 == 0 and K % 128 == 0:
            return 1206
        return 206
    if M <= 128:
        if M % 64 == 0 and N % 64 == 0 and K % 64 == 0:
            return 1200
        return 200
    if split_barrier_ok and not has_bias:
        if M % 256 == 0 and N % 256 == 0 and K % 64 == 0:
            return 1300
        return 300
    if M % 64 == 0 and N % 64 == 0 and K % 64 == 0:
        return 1200
    return 200


def _pre_pr_gfx1250(M: int, N: int, _K: int, _has_bias: bool, _output: str) -> int:
    if M % 32 == 0:
        if N % 128 == 0:
            return 20007
        if N % 64 == 0:
            return 20006
        if N % 32 == 0:
            return 20005
    if N % 128 == 0:
        return 20004
    if N % 64 == 0:
        return 20003
    return 20000


def _pre_pr_gfx942_bf16(M: int, N: int, K: int) -> int:
    k64_ok = K % 64 == 0
    k32_ok = K % 32 == 0
    wkc_bk64_ok = K >= 4096 and K % 512 == 0
    p1_ok = K % 128 == 0
    loops = (K + 63) // 64
    sb_ok = N % 16 == 0 and K % 64 == 0 and loops >= 2 and loops % 2 == 0

    if K == 4096:
        if p1_ok and M in (48, 64) and N == 1024:
            return 10213
        if p1_ok and ((M == 128 and N == 512) or (M == 256 and N == 256)):
            return 10213
        if p1_ok and M == 512 and N == 256:
            return 10203
        if M in (48, 64) and 1536 <= N <= 2048:
            return 10205
        if (M == 128 and N == 1024) or (M == 256 and N == 512):
            return 10205
        if (
            (M == 128 and 1536 <= N <= 2048)
            or (M == 256 and N == 1024)
            or (M == 512 and N == 512)
        ):
            return 10200

    if K >= 1024 and k32_ok and N >= 1536 and M <= 32:
        if M <= 4 and N >= 4096:
            return 10300
        if M <= 16:
            return 10305 if wkc_bk64_ok else 10301
        return 10305 if M == 32 and K == 4096 and wkc_bk64_ok else 10303

    if (
        K >= 512
        and k64_ok
        and (N <= 64 or (M <= 128 and N <= 1024) or (M <= 8 and N <= 1536))
    ):
        if N <= 64 and M > 128:
            return 10302
        if N <= 256 or M <= 8 or (M <= 16 and N <= 800):
            return 10300
        return 10302

    bf16ws_band = (
        K >= 4096 and K % 64 == 0 and 104 <= M <= 608 and (N == 256 or 512 <= N <= 2048)
    )
    if bf16ws_band:
        return 10210

    if N == 384 and K >= 4096:
        if M <= 128:
            return 10302
        if M <= 224:
            return 10201
        if 392 <= M <= 512:
            return 10204
        return 10200

    if k64_ok and N >= 4096 and K <= 3200:
        if K <= 640 and M <= 128:
            return 10001
        return 10000
    if sb_ok and M >= 128:
        return 10000
    if N <= 256 and p1_ok:
        return 10201
    return 10200


def _pre_pr_gfx942(M: int, N: int, K: int, has_bias: bool, output: str) -> int:
    if output == "bf16" and not has_bias:
        return _pre_pr_gfx942_bf16(M, N, K)
    if N <= 256 and K % 128 == 0:
        return 10201
    return 10200


_PRE_PR_HEURISTICS = {
    "gfx950": _pre_pr_gfx950,
    "gfx942": _pre_pr_gfx942,
    "gfx1250": _pre_pr_gfx1250,
}


def _around(*boundaries: int) -> tuple[int, ...]:
    return tuple(
        sorted(
            value
            for boundary in boundaries
            for value in (boundary - 1, boundary, boundary + 1)
            if value > 0
        )
    )


_M_SWEEP = _around(4, 8, 16, 32, 48, 64, 104, 128, 224, 256, 392, 512, 608)
_N_SWEEP = _around(16, 32, 64, 128, 256, 384, 512, 768, 800, 1024, 1536, 2048, 4096)
_K_SWEEP = _around(32, 64, 128, 512, 640, 1000, 1024, 2048, 3200, 4096, 5120, 7168)


def test_python_heuristics_match_pre_pr_cpp_boundary_sweep():
    mismatches = []
    checked = 0
    for arch, reference in _PRE_PR_HEURISTICS.items():
        for M, N, K, has_bias, output in product(
            _M_SWEEP, _N_SWEEP, _K_SWEEP, (False, True), ("bf16", "fp32")
        ):
            expected = reference(M, N, K, has_bias, output)
            actual = policy.select_a16w16_heuristic_kid(
                arch=arch,
                M=M,
                N=N,
                K=K,
                batch=1,
                has_bias=has_bias,
                output_dtype=output,
            )
            checked += 1
            if actual != expected:
                mismatches.append((arch, M, N, K, has_bias, output, expected, actual))
                if len(mismatches) == 20:
                    break
        if mismatches:
            break

    assert checked > 500_000
    assert not mismatches, (
        f"Python heuristic differs from pre-PR {_PRE_PR_REF} header blobs "
        f"{_PRE_PR_HEADERS}: {mismatches}"
    )


@cache
def _shipped_opus_rows() -> pd.DataFrame:
    root = Path(__file__).resolve().parents[1]
    paths = [root / "aiter/configs/bf16_tuned_gemm.csv"]
    paths.extend(
        sorted((root / "aiter/configs/model_configs").glob("*_bf16_tuned_gemm.csv"))
    )
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        if "libtype" in frame.columns:
            frame = frame[frame["libtype"].eq("opus")]
        if not frame.empty:
            frames.append(frame)
    assert frames, "the shipped BF16 tuning files contain no OPUS rows"
    return pd.concat(frames, ignore_index=True).drop_duplicates()


def _pre_pr_python_tuned_map(rows: pd.DataFrame) -> dict[tuple, dict]:
    # aiter/ops/opus/common.py on main used this key (notably without gfx).
    columns = policy._A16W16_TUNED_KEY_COLUMNS[1:]
    result = {}
    for _, row in rows.sort_values("us", kind="stable", na_position="last").iterrows():
        key = tuple(row[column] for column in columns)
        result.setdefault(key, row.to_dict())
    return result


def _pre_pr_codegen_tuned_map(rows: pd.DataFrame) -> dict[tuple, int]:
    # opus_gemm_lookup.h had one arch-specific (M,N,K,outdtype) table. The
    # generated entry carried a launcher pointer, represented here by its kid.
    result = {}
    for _, row in rows.iterrows():
        kid = int(row["solidx"])
        kid_arches = [
            arch
            for arch in _PRE_PR_HEURISTICS
            if get_kernel_instance(arch, "a16w16", kid) is not None
        ]
        assert kid_arches == [str(row["gfx"])]
        key = (
            kid_arches[0],
            int(row["M"]),
            int(row["N"]),
            int(row["K"]),
            str(row["outdtype"]),
        )
        result[key] = kid
    return result


def _dtype(value: object) -> torch.dtype:
    return {
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
    }[str(value)]


_EXPECTED_GFX_KEY_FIXES = {
    ("gfx950", 256, 1536, 7168): ((20399, 5), (1214, 4)),
    ("gfx950", 1024, 512, 7168): ((20400, 3), (1208, 2)),
    ("gfx950", 64, 8448, 7168): ((20341, 3), (1216, 2)),
    ("gfx950", 8192, 1024, 4096): ((21177, 0), (1401, 0)),
    ("gfx950", 16384, 512, 4096): ((21177, 0), (1401, 0)),
    ("gfx950", 8192, 1024, 7168): ((21177, 0), (1401, 0)),
    ("gfx950", 16384, 512, 7168): ((21177, 0), (1401, 0)),
}


def test_shipped_tuned_selection_diff_is_exhaustive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    rows = _shipped_opus_rows()
    merged = tmp_path / "bf16_tuned_gemm.csv"
    rows.to_csv(merged, index=False)
    monkeypatch.setattr(
        policy,
        "AITER_CONFIGS",
        SimpleNamespace(AITER_CONFIG_GEMM_BF16_FILE=str(merged)),
    )
    policy._load_a16w16_opus_tuned.cache_clear()
    policy.lookup_a16w16_opus_config.cache_clear()
    try:
        current = policy._load_a16w16_opus_tuned()
    finally:
        policy._load_a16w16_opus_tuned.cache_clear()
        policy.lookup_a16w16_opus_config.cache_clear()

    pre_pr_python = _pre_pr_python_tuned_map(rows)
    pre_pr_codegen = _pre_pr_codegen_tuned_map(rows)
    differences = {}
    invalid_current_rows = []
    redirected_current_rows = []

    for key, config in current.items():
        fields = dict(zip(policy._A16W16_TUNED_KEY_COLUMNS, key))
        arch = str(fields["gfx"])
        shape = (arch, int(fields["M"]), int(fields["N"]), int(fields["K"]))
        pair = (int(config["solidx"]), int(config["splitK"]))

        # The removed C++ table and the runtime table select the same kid for
        # every shipped full-key row. splitK was not stored in the C++ table.
        codegen_key = (*shape, str(fields["outdtype"]))
        assert pre_pr_codegen[codegen_key] == pair[0]

        old_row = pre_pr_python[key[1:]]
        old_pair = (int(old_row["solidx"]), int(old_row["splitK"]))
        if old_pair != pair:
            assert str(old_row["gfx"]) != arch
            assert get_kernel_instance(arch, "a16w16", old_pair[0]) is None
            assert shape not in differences
            differences[shape] = (old_pair, pair)

        plan = policy.resolve_a16w16_tuned_candidate(
            arch=arch,
            M=int(fields["M"]),
            N=int(fields["N"]),
            K=int(fields["K"]),
            batch=1,
            cu_num=int(fields["cu_num"]),
            has_bias=bool(fields["bias"]),
            input_dtype=_dtype(fields["dtype"]),
            output_dtype=_dtype(fields["outdtype"]),
            requested_kid=pair[0],
            requested_split_k=pair[1],
        )
        if plan is None:
            invalid_current_rows.append((*shape, pair))
        elif plan.resolved_kid != pair[0]:
            redirected_current_rows.append((*shape, pair, plan.resolved_kid))

    assert differences == _EXPECTED_GFX_KEY_FIXES
    assert invalid_current_rows == []
    assert redirected_current_rows == []


@pytest.mark.parametrize(
    ("arch", "cu_num", "M", "N", "K", "has_bias", "output", "expected"),
    (
        ("gfx950", 256, 1, 17, 130, False, torch.bfloat16, (208, 0)),
        ("gfx950", 256, 5, 33, 130, False, torch.bfloat16, (206, 0)),
        ("gfx950", 256, 64, 64, 128, False, torch.bfloat16, (1206, 0)),
        ("gfx950", 256, 65, 64, 64, False, torch.float32, (200, 0)),
        ("gfx950", 256, 129, 16, 128, False, torch.bfloat16, (300, 0)),
        ("gfx950", 256, 256, 256, 128, False, torch.bfloat16, (1300, 0)),
        ("gfx950", 256, 256, 256, 128, True, torch.bfloat16, (1200, 0)),
        ("gfx1250", 256, 31, 127, 4098, False, torch.bfloat16, (20000, 0)),
        ("gfx1250", 256, 32, 128, 4098, False, torch.bfloat16, (20007, 0)),
        ("gfx1250", 256, 33, 64, 4098, True, torch.float32, (20003, 0)),
        # main selected 10210 here, whose generated launcher redirected to
        # 10200 before launch. The Python policy now resolves the same final id.
        ("gfx942", 80, 256, 768, 7168, False, torch.bfloat16, (10200, 7)),
        ("gfx942", 80, 257, 1024, 7168, False, torch.bfloat16, (10210, 4)),
        ("gfx942", 80, 4, 4097, 1024, False, torch.bfloat16, (10300, 0)),
        ("gfx942", 80, 32, 1537, 2048, False, torch.bfloat16, (10303, 0)),
        ("gfx942", 80, 128, 257, 4096, False, torch.bfloat16, (10302, 0)),
        ("gfx942", 80, 32, 256, 1024, False, torch.float32, (10201, 8)),
    ),
)
def test_untuned_shape_final_selection_matches_pre_pr(
    arch, cu_num, M, N, K, has_bias, output, expected
):
    rows = _shipped_opus_rows()
    full_key = (
        arch,
        cu_num,
        M,
        N,
        K,
        has_bias,
        str(torch.bfloat16),
        str(output),
        False,
        False,
    )
    current_keys = {
        tuple(row[column] for column in policy._A16W16_TUNED_KEY_COLUMNS)
        for _, row in rows.iterrows()
    }
    assert full_key not in current_keys

    raw_pre_pr = _PRE_PR_HEURISTICS[arch](
        M, N, K, has_bias, "bf16" if output == torch.bfloat16 else "fp32"
    )
    plan = policy.resolve_a16w16_heuristic_candidate(
        arch=arch,
        M=M,
        N=N,
        K=K,
        batch=1,
        cu_num=cu_num,
        has_bias=has_bias,
        input_dtype=torch.bfloat16,
        output_dtype=output,
    )
    assert plan is not None
    assert (
        policy.select_a16w16_heuristic_kid(
            arch=arch,
            M=M,
            N=N,
            K=K,
            batch=1,
            has_bias=has_bias,
            output_dtype=output,
        )
        == raw_pre_pr
    )
    assert (plan.resolved_kid, plan.abi_split_k) == expected


@pytest.mark.parametrize(
    ("requested", "expected"),
    (
        (10210, 10200),
        (10213, 10203),
        (10216, None),
        (10200, 10200),
        (10300, 10300),
    ),
)
def test_gfx942_non_exact_n_matches_pre_pr_generated_launcher(requested, expected):
    # main redirected the two paired BF16-workspace launchers and AITER_CHECKed
    # 10216, which has no FP32-workspace sibling. ``None`` is that rejection at
    # policy time; unrelated gfx942 kids must remain unchanged.
    plan = policy.resolve_a16w16_tuned_candidate(
        arch="gfx942",
        M=256,
        N=768,
        K=4096,
        batch=1,
        cu_num=80,
        has_bias=False,
        input_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        requested_kid=requested,
        requested_split_k=1,
    )
    assert (None if plan is None else plan.resolved_kid) == expected


def test_gfx942_exact_n_keeps_each_bf16_workspace_kid():
    for N, requested in product(sorted(GFX942_BF16WS_EXACT_N), (10210, 10213, 10216)):
        plan = policy.resolve_a16w16_tuned_candidate(
            arch="gfx942",
            M=256,
            N=N,
            K=4096,
            batch=1,
            cu_num=80,
            has_bias=False,
            input_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            requested_kid=requested,
            requested_split_k=1,
        )
        assert plan is not None
        assert plan.resolved_kid == requested
