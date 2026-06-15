# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the FlyDSL *biased* grouped-topk gating kernel.

Validates ``flydsl_biased_grouped_topk`` (the biased routing variant, used when
a per-expert ``correction_bias`` is present) against two references:
  * the HIP op ``aiter.biased_grouped_topk_hip``;
  * the high-precision torch reference ``biased_grouped_topk_torch``.

Correctness criteria (same tie-robust approach as the non-biased test):
  * fp32: exact expert-id match + routing weights equal to ~1e-4.
  * bf16/fp16: low-precision scores collapse into ties that FlyDSL / HIP / torch
    break differently, so we compare the routing weights sorted by *value*.

Usage:
    python aiter/ops/flydsl/test_flydsl_biased_grouped_topk.py
    pytest -q aiter/ops/flydsl/test_flydsl_biased_grouped_topk.py
"""

import pytest
import torch

from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL biased grouped-topk tests.",
        allow_module_level=True,
    )

try:
    from aiter.ops.flydsl.moe_kernels import flydsl_biased_grouped_topk
    from aiter.ops.topk import biased_grouped_topk_hip, biased_grouped_topk_torch
except ImportError as exc:
    pytest.skip(
        f"Unable to import FlyDSL biased grouped-topk kernel: {exc}",
        allow_module_level=True,
    )

torch.set_default_device("cuda")

# (num_tokens, num_experts, num_expert_group, topk_group, topk)
BIASED_CASES = [
    # Production-like routing: no grouping, sigmoid + bias, renorm.
    {"token": 64, "expert": 192, "group": 1, "topk_group": 1, "topk": 8},
    {"token": 1, "expert": 192, "group": 1, "topk_group": 1, "topk": 8},
    {"token": 34, "expert": 192, "group": 1, "topk_group": 1, "topk": 8},
    {"token": 256, "expert": 192, "group": 1, "topk_group": 1, "topk": 8},
    # Grouped biased routing (top-2 group score). These configs avoid the
    # 256/8/4/8 special case that the dispatch intentionally leaves to the HIP op.
    {"token": 64, "expert": 256, "group": 8, "topk_group": 4, "topk": 6},
    {"token": 1, "expert": 256, "group": 8, "topk_group": 4, "topk": 6},
    {"token": 128, "expert": 128, "group": 8, "topk_group": 4, "topk": 6},
]

RENORM = [True, False]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def _run_case(
    token: int,
    expert: int,
    group: int,
    topk_group: int,
    topk: int,
    need_renorm: bool,
    dtype: torch.dtype,
    scale_factor: float = 1.0,
    seed: int = 0,
):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    gating = torch.randn((token, expert), generator=gen, device="cuda", dtype=dtype)
    correction_bias = torch.randn(expert, generator=gen, device="cuda", dtype=dtype)

    # HIP reference op (in-place into strided outputs).
    w_hip = torch.empty_strided(
        (token, topk), (topk + 10, 1), dtype=torch.float32, device="cuda"
    )
    id_hip = torch.empty_strided(
        (token, topk), (topk + 10, 1), dtype=torch.int32, device="cuda"
    )
    biased_grouped_topk_hip(
        gating, correction_bias, w_hip, id_hip,
        group, topk_group, need_renorm, scale_factor,
    )

    # FlyDSL op.
    w_fly = torch.empty_strided(
        (token, topk), (topk + 10, 1), dtype=torch.float32, device="cuda"
    )
    id_fly = torch.empty_strided(
        (token, topk), (topk + 10, 1), dtype=torch.int32, device="cuda"
    )
    flydsl_biased_grouped_topk(
        gating, correction_bias, w_fly, id_fly,
        group, topk_group, need_renorm, scale_factor,
    )
    torch.cuda.synchronize()

    ids_hip_s, perm = torch.sort(id_hip, dim=-1)
    ids_fly_s, perm_f = torch.sort(id_fly, dim=-1)
    id_match = torch.equal(ids_fly_s.to(torch.int32), ids_hip_s.to(torch.int32))
    w_delta = (
        w_fly.gather(1, perm_f).float() - w_hip.gather(1, perm).float()
    ).abs().max().item()

    # Tie-robust: compare weights sorted by value (vs HIP).
    w_delta_val = (
        torch.sort(w_fly.float(), dim=-1).values
        - torch.sort(w_hip.float(), dim=-1).values
    ).abs().max().item()

    # High-precision torch reference. FlyDSL stores weights scaled by
    # `scale_factor` (renorm -> sum == scale_factor; otherwise de-biased sigmoid
    # * scale_factor), whereas the torch ref uses scale_factor == 1, so we divide
    # it out before the (tie-robust, value-sorted) comparison.
    w_torch, _ = biased_grouped_topk_torch(
        gating, correction_bias, topk, need_renorm, group, topk_group
    )
    w_delta_torch = (
        torch.sort(w_fly.float() / scale_factor, dim=-1).values
        - torch.sort(w_torch.float(), dim=-1).values
    ).abs().max().item()

    tol = 1e-4 if dtype == torch.float32 else 5e-3
    if dtype == torch.float32:
        passed = id_match and w_delta <= tol and w_delta_torch <= tol
    else:
        passed = w_delta_val <= tol and w_delta_torch <= tol

    label = (
        f"t{token}_e{expert}_g{group}_tg{topk_group}_k{topk}_"
        f"renorm{int(need_renorm)}_sf{scale_factor}_{str(dtype).split('.')[-1]}"
    )
    print(
        f"  [{label}] vs HIP id_match={id_match} w_delta={w_delta:.3e} "
        f"tie_robust_wdelta={w_delta_val:.3e} | vs torch wdelta={w_delta_torch:.3e} "
        f"--> {'PASS' if passed else 'FAIL'}"
    )
    return passed, w_delta_val


@pytest.mark.parametrize(
    "case",
    [
        pytest.param(c, id=f"t{c['token']}_e{c['expert']}_g{c['group']}")
        for c in BIASED_CASES
    ],
)
@pytest.mark.parametrize("need_renorm", RENORM)
@pytest.mark.parametrize("dtype", DTYPES, ids=[str(d).split(".")[-1] for d in DTYPES])
def test_flydsl_biased_grouped_topk(case: dict, need_renorm: bool, dtype):
    passed, _ = _run_case(
        case["token"], case["expert"], case["group"],
        case["topk_group"], case["topk"], need_renorm, dtype,
        scale_factor=2.826,
    )
    assert passed


def main() -> int:
    results = []
    for dtype in DTYPES:
        for case in BIASED_CASES:
            for need_renorm in RENORM:
                for sf in (1.0, 2.826):
                    try:
                        passed, _ = _run_case(
                            case["token"], case["expert"], case["group"],
                            case["topk_group"], case["topk"], need_renorm, dtype, sf,
                        )
                        status = "PASS" if passed else "FAIL"
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        status = "ERROR"
                    results.append(status)

    n_pass = sum(1 for s in results if s == "PASS")
    print(f"\n{'='*60}\n  {n_pass}/{len(results)} passed "
          f"(FlyDSL vs HIP biased_grouped_topk)\n{'='*60}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
