# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Planted defects that today's acceptance gates accept, and the gates that catch them.

Every test here asserts *both* halves of a claim:

1. a deliberately broken result passes the gate AITER ships today, and
2. the same broken result fails the gate this branch adds.

Asserting the first half matters as much as the second. It is what makes the
claim falsifiable, and it means that if the shipped gate is ever tightened
elsewhere, these tests fail and say so rather than quietly becoming vacuous.

Three escapes are covered, one per acceptance mechanism:

* ``test_mismatch_does_not_fail_the_process`` -- the verdict is computed and
  discarded, so a wrong kernel exits 0.
* ``test_planted_offset_passes_the_hardcoded_atol`` -- the MoE ``atol=100`` is
  looser than the arithmetic justifies at that output magnitude.
* ``test_dropped_rows_pass_the_tuner_cosine_gate`` -- the tuner's whole-tensor
  score dilutes a defect confined to a few rows.

CPU only. No kernels, no GPU, no build.
"""

import os
import subprocess
import sys
from pathlib import Path

import torch

from aiter.test_common import checkAllclose
from aiter.utility.cos_diff import (
    COS_DIFF_THRESHOLD,
    _whole,
    combined_cos_diff,
    worst_row_cos_diff,
)
from aiter.utility.tolerance import derive_tolerance

REPO_ROOT = Path(__file__).resolve().parents[1]

# What op_tests/test_moe.py compares against, before this branch.
SHIPPED_MOE_RTOL, SHIPPED_MOE_ATOL = 0.01, 100

# checkAllclose accepts up to this fraction of mismatching elements.
TOL_ERR_RATIO = 0.05


# --------------------------------------------------------------------------
# Escape 1: a failing comparison does not fail the build.
# --------------------------------------------------------------------------

# A miniature op_test with the shape of a real one: build tensors, compare,
# fall off the end of the script. Every element is wrong by 10x.
_MINIATURE_OP_TEST = """
import torch
from aiter.test_common import checkAllclose

ref = torch.ones(64, 128)
out = ref * 10.0
checkAllclose(out, ref, msg="planted defect ")
print("op test body completed")
"""


def _run_miniature_op_test(strict):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["AITER_STRICT_ALLCLOSE"] = "1" if strict else "0"
    return subprocess.run(
        [sys.executable, "-c", _MINIATURE_OP_TEST],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_mismatch_does_not_fail_the_process():
    """A kernel that is wrong in every element still exits 0.

    This is the escape that makes the other two reachable: even when the
    comparison does print ``failed!``, nothing downstream observes it. Of the
    ~700 ``checkAllclose`` call sites under ``op_tests/``, the overwhelming
    majority discard the returned ratio, so the only record of the failure is a
    line in a log that no CI job reads.

    Spawns two interpreters, so it is the slow test in this file.
    """
    stock = _run_miniature_op_test(strict=False)
    assert "op test body completed" in stock.stdout, stock.stderr
    assert stock.returncode == 0, (
        "the shipped behaviour is expected to exit 0 here; if this now fails, "
        "the escape has been fixed and this test should be retired"
    )

    gated = _run_miniature_op_test(strict=True)
    assert gated.returncode != 0, gated.stdout + gated.stderr
    assert "op test body completed" not in gated.stdout


def test_strict_flag_raises_without_touching_the_call_site():
    """The same switch, in-process, via the explicit argument."""
    ref = torch.ones(64, 128)
    out = ref * 10.0

    # Default: returns the verdict instead of acting on it.
    assert checkAllclose(out, ref, printLog=False) == 1.0

    try:
        checkAllclose(out, ref, strict=True)
    except AssertionError:
        pass
    else:
        raise AssertionError("strict=True should have raised")


def test_strict_leaves_a_passing_comparison_alone():
    """Opt-in must not turn correct results into failures."""
    ref = torch.ones(64, 128)
    assert checkAllclose(ref, ref.clone(), strict=True) == 0


def test_strict_does_not_disturb_the_tuner_ranking_path():
    """``printLog=False`` is the autotuner asking for a number, not a verdict.

    Raising there would abort tuning instead of rejecting one candidate, so the
    environment switch deliberately does not reach it.
    """
    ref = torch.ones(64, 128)
    out = ref * 10.0
    os.environ["AITER_STRICT_ALLCLOSE"] = "1"
    try:
        assert checkAllclose(out, ref, printLog=False) == 1.0
    finally:
        os.environ.pop("AITER_STRICT_ALLCLOSE", None)


# --------------------------------------------------------------------------
# Escape 2: the hardcoded MoE tolerance is not tied to the arithmetic.
# --------------------------------------------------------------------------


def _moe_like_reference(max_abs, rows=512, cols=128, seed=0):
    """bf16 activations whose largest magnitude is ``max_abs``."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(rows, cols, generator=g, dtype=torch.float32)
    x = x / x.abs().max() * max_abs
    return x.to(torch.bfloat16)


def test_planted_offset_passes_the_hardcoded_atol():
    """A constant 99.0 bias on every output element is invisible to atol=100.

    ``atol=100`` is a magic number: it is not derived from the dtype, the
    accumulator, or the reduction depth, so it has no relationship to the error
    the pipeline can actually produce. At this output magnitude the fp8 error
    model justifies 64, and the gap between 64 and 100 is free space a broken
    kernel can sit in.
    """
    ref = _moe_like_reference(max_abs=1928.0)
    broken = (ref.float() + 99.0).to(torch.bfloat16)

    shipped = checkAllclose(
        broken, ref, rtol=SHIPPED_MOE_RTOL, atol=SHIPPED_MOE_ATOL, printLog=False
    )
    assert shipped == 0, f"expected the shipped gate to accept this, got {shipped}"

    rtol, atol = derive_tolerance(
        torch.float8_e4m3fnuz,
        torch.bfloat16,
        num_accumulations=3840,
        max_value=ref.abs().max().item(),
    )
    assert atol == 64.0, atol

    derived = checkAllclose(broken, ref, rtol=rtol, atol=atol, printLog=False)
    assert (
        derived > TOL_ERR_RATIO
    ), f"expected the derived gate to reject this, got {derived}"


def test_derived_tolerance_is_not_merely_tighter():
    """At a larger output magnitude the derivation is *looser* than 100.

    Worth asserting, because it shows the method is calibrating the gate rather
    than cranking it down. The same fp8 algorithm lands at 64 or 256 depending
    only on which binary exponent band ``max|ref|`` falls in, which is why one
    constant cannot be right for every shape.
    """
    bands = {
        m: derive_tolerance(
            torch.float8_e4m3fnuz,
            torch.bfloat16,
            num_accumulations=3840,
            max_value=m,
        )[1]
        for m in (1928.0, 2048.0, 6304.0)
    }
    assert bands == {1928.0: 64.0, 2048.0: 128.0, 6304.0: 256.0}, bands
    assert min(bands.values()) < SHIPPED_MOE_ATOL < max(bands.values())


# --------------------------------------------------------------------------
# Escape 3: the tuner's cosine score dilutes a localised defect.
# --------------------------------------------------------------------------


def test_dropped_rows_pass_the_tuner_cosine_gate():
    """A kernel that drops 17% of its output rows is still selected as best.

    The tuner flattens both tensors into one long vector, so the energy of the
    correct rows pays for the missing ones. Zeroing a fraction ``f`` of rows
    scores ``f / (2 - f)``, which stays under the 0.1 threshold until ``f``
    reaches 18%.
    """
    g = torch.Generator().manual_seed(0)
    ref = torch.randn(512, 128, generator=g, dtype=torch.float64)
    # Equal row energy, so the dilution matches the closed form exactly.
    ref = ref / ref.norm(dim=1, keepdim=True)

    dropped = int(0.17 * ref.shape[0])
    out = ref.clone()
    out[:dropped] = 0.0

    f = dropped / ref.shape[0]
    whole = _whole(ref.flatten(), out.flatten())
    assert abs(whole - f / (2 - f)) < 1e-9, (whole, f / (2 - f))
    assert (
        whole < COS_DIFF_THRESHOLD
    ), f"expected the shipped metric to accept, got {whole}"

    worst = worst_row_cos_diff(ref, out)
    assert worst == 1.0, worst
    assert worst > COS_DIFF_THRESHOLD


def test_rowwise_metric_accepts_a_uniformly_good_result():
    """Per-row scoring must not reject small errors spread over every row."""
    g = torch.Generator().manual_seed(1)
    ref = torch.randn(256, 64, generator=g, dtype=torch.float64)
    out = ref + torch.randn(256, 64, generator=g, dtype=torch.float64) * 1e-3

    assert _whole(ref.flatten(), out.flatten()) < COS_DIFF_THRESHOLD
    assert worst_row_cos_diff(ref, out) < COS_DIFF_THRESHOLD


def test_rowwise_scoring_is_opt_in_and_can_only_tighten():
    """``combined_cos_diff`` is what the tuner calls; check the switch and the
    monotonicity it promises."""
    g = torch.Generator().manual_seed(3)
    ref = torch.randn(128, 32, generator=g, dtype=torch.float64)
    ref = ref / ref.norm(dim=1, keepdim=True)
    out = ref.clone()
    out[:16] = 0.0
    whole = _whole(ref.flatten(), out.flatten())

    os.environ.pop("AITER_MOE_COS_DIFF_ROWWISE", None)
    assert combined_cos_diff(ref, out, whole) == whole

    os.environ["AITER_MOE_COS_DIFF_ROWWISE"] = "1"
    try:
        tightened = combined_cos_diff(ref, out, whole)
        assert tightened == 1.0, tightened
        assert tightened >= whole
        # Mismatched shapes fall back rather than guessing a row axis.
        assert combined_cos_diff(ref, out[:, :16], whole) == whole
        assert combined_cos_diff(None, out, whole) == whole
    finally:
        os.environ.pop("AITER_MOE_COS_DIFF_ROWWISE", None)


def test_rowwise_metric_ignores_all_zero_padding_rows():
    """Padding rows in the sorted MoE layout carry no signal to compare."""
    g = torch.Generator().manual_seed(2)
    ref = torch.randn(64, 32, generator=g, dtype=torch.float64)
    ref[32:] = 0.0
    out = ref.clone()

    assert worst_row_cos_diff(ref, out) == 0.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all acceptance-escape tests passed")
