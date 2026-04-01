# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for FlyDSL split-K HGEMM precision regressions.

Usage:
    python aiter/ops/flydsl/test_flydsl_splitk_hgemm.py
    pytest -q aiter/ops/flydsl/test_flydsl_splitk_hgemm.py
"""

from __future__ import annotations

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
    from aiter.ops.flydsl.gemm_kernels import flydsl_hgemm
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
        "name": "splitk2_tile48_m136_n384_k7168",
        "m": 136,
        "n": 384,
        "k": 7168,
        "tile_k": 128,
        "tile_m": 48,
        "tile_n": 64,
        "pack_n": 1,
        "split_k": 2,
        "b_preshuffle": False,
    },
    {
        "name": "splitk8_tile48_m136_n384_k7168",
        "m": 136,
        "n": 384,
        "k": 7168,
        "tile_k": 128,
        "tile_m": 48,
        "tile_n": 64,
        "pack_n": 1,
        "split_k": 8,
        "b_preshuffle": False,
    },
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
        "name": "splitk8_tile48_m48_n1024_k7168",
        "m": 48,
        "n": 1024,
        "k": 7168,
        "tile_k": 128,
        "tile_m": 48,
        "tile_n": 64,
        "pack_n": 1,
        "split_k": 8,
        "b_preshuffle": False,
    },
    {
        "name": "splitk8_tile48_m48_n2112_k7168",
        "m": 48,
        "n": 2112,
        "k": 7168,
        "tile_k": 128,
        "tile_m": 48,
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
) -> tuple[bool, float, float]:
    ref_f = ref.float()
    out_f = out.float()
    close_mask = torch.isclose(ref_f, out_f, atol=atol, rtol=rtol)
    pct_close = close_mask.float().mean().item() * 100.0
    max_delta = (ref_f - out_f).abs().max().item()
    passed = pct_close >= pass_pct
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
        b_preshuffle=case["b_preshuffle"],
    )
    torch.cuda.synchronize()

    return _check_output(
        ref,
        out,
        case["name"],
        atol=atol,
        rtol=rtol,
        pass_pct=pass_pct,
    )


@pytest.mark.parametrize(
    "case",
    [pytest.param(case, id=case["name"]) for case in SPLITK_PRECISION_CASES],
)
def test_flydsl_splitk_hgemm_precision_regressions(case: dict):
    passed, _, _ = run_splitk_precision_case(case)
    assert passed


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
