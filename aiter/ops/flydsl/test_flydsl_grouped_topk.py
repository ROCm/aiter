# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the FlyDSL grouped-topk gating kernel.

Validates ``flydsl_grouped_topk`` against two references:
  * the HIP op ``aiter.grouped_topk``;
  * the high-precision torch reference ``grouped_topk_torch`` (fp32 internal).

Correctness criteria:
  * fp32: exact expert-id match + routing weights equal to ~1e-4.
  * bf16/fp16: softmax/max are monotonic, so expert *selection* depends only on
    the raw-logit ordering. In low precision the raw logits collapse to equal
    values (ties), and FlyDSL / HIP / torch break those ties differently. We
    therefore compare the routing weights sorted by *value* (tie-robust): tied
    experts carry equal weights, so the value-sorted weight vectors must agree.

Usage:
    python aiter/ops/flydsl/test_flydsl_grouped_topk.py
    pytest -q aiter/ops/flydsl/test_flydsl_grouped_topk.py
"""

import pytest
import torch

from aiter.ops.flydsl.utils import is_flydsl_available

if not torch.cuda.is_available():
    pytest.skip("ROCm not available. Skipping GPU tests.", allow_module_level=True)
if not is_flydsl_available():
    pytest.skip(
        "flydsl is not installed. Skipping FlyDSL grouped-topk tests.",
        allow_module_level=True,
    )

try:
    from aiter.ops.flydsl.moe_kernels import flydsl_grouped_topk
    from aiter.ops.topk import grouped_topk as grouped_topk_hip
    from aiter.ops.topk import grouped_topk_torch
except ImportError as exc:
    pytest.skip(
        f"Unable to import FlyDSL grouped-topk kernel: {exc}",
        allow_module_level=True,
    )

torch.set_default_device("cuda")

# (num_tokens, num_experts, num_expert_group, topk_group, topk)
GROUPED_TOPK_CASES = [
    # Group-limited routing, 256 experts / 8 groups.
    {"token": 64, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
    {"token": 1, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
    {"token": 4096, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
    {"token": 200, "expert": 128, "group": 8, "topk_group": 4, "topk": 6},
    # No-grouping degenerate config (group==1 -> plain topk).
    {"token": 33, "expert": 128, "group": 1, "topk_group": 1, "topk": 4},
]

SCORING = ["softmax", "sigmoid"]
RENORM = [True, False]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def _sorted_by_id(ids, w):
    ids_s, perm = torch.sort(ids, dim=-1)
    return ids_s.to(torch.int32), w.gather(1, perm).float()


def _run_case(
    token: int,
    expert: int,
    group: int,
    topk_group: int,
    topk: int,
    scoring_func: str,
    need_renorm: bool,
    dtype: torch.dtype,
    scale_factor: float = 1.0,
    seed: int = 0,
):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    gating = torch.randn((token, expert), generator=gen, device="cuda", dtype=dtype)
    is_softmax = scoring_func == "softmax"

    # HIP reference op (in-place into strided outputs).
    w_hip = torch.empty_strided(
        (token, topk), (topk + 10, 1), dtype=torch.float32, device="cuda"
    )
    id_hip = torch.empty_strided(
        (token, topk), (topk + 10, 1), dtype=torch.int32, device="cuda"
    )
    grouped_topk_hip(
        gating, w_hip, id_hip, group, topk_group, need_renorm, is_softmax, scale_factor
    )

    # FlyDSL op.
    w_fly = torch.empty_strided(
        (token, topk), (topk + 10, 1), dtype=torch.float32, device="cuda"
    )
    id_fly = torch.empty_strided(
        (token, topk), (topk + 10, 1), dtype=torch.int32, device="cuda"
    )
    flydsl_grouped_topk(
        gating, w_fly, id_fly, group, topk_group, need_renorm, is_softmax, scale_factor
    )
    torch.cuda.synchronize()

    id_hip_s, w_hip_s = _sorted_by_id(id_hip, w_hip)
    id_fly_s, w_fly_s = _sorted_by_id(id_fly, w_fly)
    id_match = torch.equal(id_fly_s, id_hip_s)
    w_delta = (w_fly_s - w_hip_s).abs().max().item()

    # Tie-robust: compare weights sorted by value (vs HIP).
    w_delta_val = (
        torch.sort(w_fly.float(), dim=-1).values
        - torch.sort(w_hip.float(), dim=-1).values
    ).abs().max().item()

    # High-precision torch reference (fp32 internal). FlyDSL stores weights
    # scaled by `scale_factor` (renorm -> sum == scale_factor; otherwise raw
    # score * scale_factor), whereas the torch ref uses scale_factor == 1, so we
    # divide it out before the (tie-robust, value-sorted) comparison.
    w_torch, _ = grouped_topk_torch(
        gating, topk, need_renorm, group, topk_group, scoring_func=scoring_func
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
        f"{scoring_func}_renorm{int(need_renorm)}_{str(dtype).split('.')[-1]}"
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
        for c in GROUPED_TOPK_CASES
    ],
)
@pytest.mark.parametrize("scoring_func", SCORING)
@pytest.mark.parametrize("need_renorm", RENORM)
@pytest.mark.parametrize(
    "dtype", DTYPES, ids=[str(d).split(".")[-1] for d in DTYPES]
)
def test_flydsl_grouped_topk(case: dict, scoring_func: str, need_renorm: bool, dtype):
    passed, _ = _run_case(
        case["token"],
        case["expert"],
        case["group"],
        case["topk_group"],
        case["topk"],
        scoring_func,
        need_renorm,
        dtype,
        scale_factor=1.0,
    )
    assert passed


def main() -> int:
    results = []
    for dtype in DTYPES:
        for case in GROUPED_TOPK_CASES:
            for scoring_func in SCORING:
                for need_renorm in RENORM:
                    try:
                        passed, md = _run_case(
                            case["token"],
                            case["expert"],
                            case["group"],
                            case["topk_group"],
                            case["topk"],
                            scoring_func,
                            need_renorm,
                            dtype,
                        )
                        status = "PASS" if passed else "FAIL"
                    except Exception:
                        import traceback

                        traceback.print_exc()
                        status, md = "ERROR", 0.0
                    results.append(status)

    n_pass = sum(1 for s in results if s == "PASS")
    print(f"\n{'='*60}\n  {n_pass}/{len(results)} passed "
          f"(FlyDSL vs HIP grouped_topk)\n{'='*60}")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
