# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL split-K HGEMM precision regressions.

Usage:
    python aiter/ops/flydsl/test_flydsl_splitk_hgemm.py
    pytest -q aiter/ops/flydsl/test_flydsl_splitk_hgemm.py
"""

from __future__ import annotations

import os

import pytest
import torch

from aiter.ops.flydsl.utils import is_flydsl_available
from aiter.ops.shuffle import shuffle_weight

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL HGEMM tests.", allow_module_level=True
    )

try:
    from aiter.ops.flydsl.gemm_kernels import (
        FIXED_C_TO_LDS,
        FIXED_STAGE,
        KERNEL_ASYNC_COPY,
        flydsl_hgemm,
        flydsl_kernel_name,
    )
except ImportError as exc:
    pytest.skip(
        f"Unable to import FlyDSL HGEMM kernels: {exc}", allow_module_level=True
    )

torch.set_default_device("cuda")

DEFAULT_ATOL = 1e-2
DEFAULT_RTOL = 1e-2
DEFAULT_PASS_PCT = 99.9
DEFAULT_INPUT_SEED = 20260401

SPLITK_PRECISION_CASES = [
    {
        "name": "splitk8_tile32_m104_n384_k7168",
        "m": 104,
        "n": 384,
        "k": 7168,
        "tile_k": 128,
        "tile_m": 32,
        "tile_n": 64,
        "pack_n": 1,
        "split_k": 8,
        "b_preshuffle": False,
    },
    {
        "name": "splitk4_tile16_m1_n7168_k512",
        "m": 1,
        "n": 7168,
        "k": 512,
        "tile_k": 128,
        "tile_m": 16,
        "tile_n": 128,
        "pack_n": 1,
        "split_k": 4,
        "b_preshuffle": False,
    },
    {
        "name": "splitk16_tile32_m1_n2112_k7168_warp2x2_blds",
        "m": 1,
        "n": 2112,
        "k": 7168,
        "tile_k": 64,
        "tile_m": 32,
        "tile_n": 64,
        "pack_n": 1,
        "split_k": 16,
        "block_m_warps": 2,
        "block_n_warps": 2,
        "b_to_lds": True,
        "b_preshuffle": False,
        "pass_pct": 99.0,
        "max_delta_limit": 32.0,
    },
    {
        "name": "splitk8_tile32_m1_n3072_k1536_warp2x2_blds",
        "m": 1,
        "n": 3072,
        "k": 1536,
        "tile_k": 64,
        "tile_m": 32,
        "tile_n": 64,
        "pack_n": 1,
        "split_k": 8,
        "block_m_warps": 2,
        "block_n_warps": 2,
        "b_to_lds": True,
        "b_preshuffle": False,
        "pass_pct": 99.0,
        "max_delta_limit": 8.0,
    },
]

# Repeated launches catch inter-kernel split-K synchronization regressions, but
# stay opt-in because they are slower than the default precision cases.
SPLITK_STRESS_ENV = "AITER_RUN_FLYDSL_SPLITK_STRESS"
SPLITK_STRESS_REPEAT = 20
SPLITK_STRESS_MAX_DELTA_LIMIT = 10.0

SPLITK_STRESS_CASES = [
    {
        "name": "gptoss_generic_splitk4_bias_repeated_launch",
        "m": 32,
        "n": 2880,
        "k": 2048,
        "kernel_family": "hgemm",
        "tile_m": 32,
        "tile_n": 64,
        "tile_k": 256,
        "split_k": 4,
        "block_m_warps": 1,
        "block_n_warps": 4,
        "b_to_lds": False,
        "b_preshuffle": False,
        "has_bias": True,
    },
    {
        "name": "gptoss_small_m_blds_block_k_loops1_splitk8_bias",
        "m": 8,
        "n": 2880,
        "k": 2048,
        "kernel_family": "small_m",
        "tile_m": 16,
        "tile_n": 64,
        "tile_k": 256,
        "split_k": 8,
        "block_m_warps": 1,
        "block_n_warps": 2,
        "b_to_lds": True,
        "b_preshuffle": False,
        "waves_per_eu": 4,
        "b_to_lds_unroll": 8,
        "has_bias": True,
    },
]


def run_torch_acc(
    a: torch.Tensor, b: torch.Tensor, dtype=torch.float32
) -> torch.Tensor:
    return torch.mm(a.to(torch.float32), b.to(torch.float32).t()).to(dtype)


def make_inputs(
    m: int,
    n: int,
    k: int,
    torch_dtype: torch.dtype,
    *,
    seed: int = DEFAULT_INPUT_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    a = torch.rand((m, k), generator=gen, device="cuda", dtype=torch_dtype)
    b = torch.rand((n, k), generator=gen, device="cuda", dtype=torch_dtype)
    return a, b


def _check_output(
    ref: torch.Tensor,
    out: torch.Tensor,
    label: str,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    pass_pct: float = DEFAULT_PASS_PCT,
    max_delta_limit: float | None = None,
) -> tuple[bool, float, float]:
    ref_f = ref.float()
    out_f = out.float()
    close_mask = torch.isclose(ref_f, out_f, atol=atol, rtol=rtol)
    pct_close = close_mask.float().mean().item() * 100.0
    max_delta = (ref_f - out_f).abs().max().item()
    passed = pct_close >= pass_pct
    if max_delta_limit is not None:
        passed = passed and max_delta <= max_delta_limit
    print(
        f"  [{label}] max_delta={max_delta:.4f}, {pct_close:.4f}% close "
        f"(atol={atol}, rtol={rtol})"
    )
    print(f"  ref  sample: {ref.reshape(-1)[:8]}")
    print(f"  test sample: {out.reshape(-1)[:8]}")
    print(f"  --> {'PASS' if passed else 'FAIL'}")
    return passed, max_delta, pct_close


def run_splitk_precision_case(
    case: dict,
    *,
    seed: int = DEFAULT_INPUT_SEED,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
    pass_pct: float = DEFAULT_PASS_PCT,
) -> tuple[bool, float, float]:
    print("=" * 80)
    print(
        "[flydsl] split-K HGEMM precision regression "
        f"case={case['name']} shape=({case['m']}, {case['n']}, {case['k']}) "
        f"tile=({case['tile_m']}, {case['tile_n']}, {case['tile_k']}) "
        f"split_k={case['split_k']}"
    )
    print("=" * 80)

    torch_dtype = torch.bfloat16
    a, b = make_inputs(case["m"], case["n"], case["k"], torch_dtype, seed=seed)
    ref = run_torch_acc(a, b, dtype=torch_dtype)
    b_shuf = (
        shuffle_weight(b, layout=(16 * case["pack_n"], 16))
        if case["b_preshuffle"]
        else b
    )

    out = flydsl_hgemm(
        a,
        b_shuf,
        tile_k=case["tile_k"],
        tile_m=case["tile_m"],
        tile_n=case["tile_n"],
        pack_n=case["pack_n"],
        split_k=case["split_k"],
        block_m_warps=case.get("block_m_warps", 1),
        block_n_warps=case.get("block_n_warps", 4),
        b_to_lds=case.get("b_to_lds", False),
        b_preshuffle=case["b_preshuffle"],
    )
    torch.cuda.synchronize()

    return _check_output(
        ref,
        out,
        case["name"],
        atol=case.get("atol", atol),
        rtol=case.get("rtol", rtol),
        pass_pct=case.get("pass_pct", pass_pct),
        max_delta_limit=case.get("max_delta_limit"),
    )


@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case["name"]) for case in SPLITK_PRECISION_CASES],
)
def test_flydsl_splitk_hgemm_precision_regressions(case: dict):
    passed, _, _ = run_splitk_precision_case(case)
    assert passed


@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case["name"]) for case in SPLITK_STRESS_CASES],
)
def test_flydsl_splitk_repeated_launch_stress(case: dict):
    if os.environ.get(SPLITK_STRESS_ENV) != "1":
        pytest.skip(f"set {SPLITK_STRESS_ENV}=1 to run repeated-launch stress tests")

    torch_dtype = torch.bfloat16
    a, b = make_inputs(case["m"], case["n"], case["k"], torch_dtype)
    bias = (
        torch.rand((case["n"],), device="cuda", dtype=torch_dtype)
        if case.get("has_bias", False)
        else None
    )
    ref = run_torch_acc(a, b, dtype=torch.float32)
    if bias is not None:
        ref = ref + bias.float()
    ref = ref.to(torch_dtype)

    max_deltas = []
    for _ in range(SPLITK_STRESS_REPEAT):
        out = flydsl_hgemm(
            a,
            b,
            bias=bias,
            kernel_family=case.get("kernel_family"),
            tile_k=case["tile_k"],
            tile_m=case["tile_m"],
            tile_n=case["tile_n"],
            split_k=case["split_k"],
            block_m_warps=case.get("block_m_warps", 1),
            block_n_warps=case.get("block_n_warps", 4),
            waves_per_eu=case.get("waves_per_eu", 0),
            b_to_lds_unroll=case.get("b_to_lds_unroll", 0),
            b_to_lds=case.get("b_to_lds", False),
            b_preshuffle=case["b_preshuffle"],
        )
        torch.cuda.synchronize()
        max_deltas.append((ref.float() - out.float()).abs().max().item())

    max_delta = max(max_deltas)
    assert max_delta <= SPLITK_STRESS_MAX_DELTA_LIMIT, (
        f"{case['name']} max_delta={max_delta:.4f}, "
        f"all repeated max_delta values={max_deltas}"
    )


def _make_kernel_name_kwargs(**overrides) -> dict:
    kwargs = {
        "stage": FIXED_STAGE,
        "dtype": "bf16",
        "out_dtype": "bf16",
        "tile_m": 32,
        "tile_n": 64,
        "tile_k": 64,
        "split_k": 8,
        "block_m_warp": 2,
        "block_n_warp": 2,
        "async_copy": KERNEL_ASYNC_COPY,
        "b_to_lds": True,
        "b_preshuffle": False,
        "c_to_lds": FIXED_C_TO_LDS,
    }
    kwargs.update(overrides)
    return kwargs


def test_flydsl_kernel_name_rejects_legacy_metadata():
    with pytest.raises(ValueError, match="stage="):
        flydsl_kernel_name(**_make_kernel_name_kwargs(stage=1))

    with pytest.raises(ValueError, match="async_copy"):
        flydsl_kernel_name(**_make_kernel_name_kwargs(async_copy=not KERNEL_ASYNC_COPY))

    with pytest.raises(ValueError, match="c_to_lds"):
        flydsl_kernel_name(**_make_kernel_name_kwargs(c_to_lds=True))

    with pytest.raises(ValueError, match="b_to_lds=False"):
        flydsl_kernel_name(**_make_kernel_name_kwargs(b_to_lds=True, b_preshuffle=True))


def print_summary(results: list[tuple[str, str, float, float]]) -> None:
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for name, status, max_delta, pct_close in results:
        print(
            f"  {status:>5s}  {name:<35s}  max_delta={max_delta:>8.4f}  "
            f"close={pct_close:>6.2f}%"
        )

    n_pass = sum(1 for _, status, _, _ in results if status == "PASS")
    print(f"\n  {n_pass}/{len(results)} passed")


def main() -> int:
    results: list[tuple[str, str, float, float]] = []

    for case in SPLITK_PRECISION_CASES:
        try:
            passed, max_delta, pct_close = run_splitk_precision_case(case)
            results.append(
                (
                    case["name"],
                    "PASS" if passed else "FAIL",
                    max_delta,
                    pct_close,
                )
            )
        except Exception:
            import traceback

            traceback.print_exc()
            results.append((case["name"], "ERROR", 0.0, 0.0))

    print_summary(results)
    return 1 if any(status in ("FAIL", "ERROR") for _, status, _, _ in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
