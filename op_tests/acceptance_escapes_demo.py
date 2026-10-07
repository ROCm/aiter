# SPDX-License-Identifier: MIT
"""Four acceptance escapes, measured against UNMODIFIED AITER.

This script imports nothing that the PR adds. Point it at a stock checkout and
it reports what the shipped gates do with deliberately broken results:

    PYTHONPATH=/path/to/stock/aiter python stock_escapes.py

Each case prints the number the shipped gate actually produced, so the claim
can be checked rather than taken on trust. CPU only.
"""

import os
import subprocess
import sys

import torch

from aiter.test_common import checkAllclose

# The MoE gate as it appears in op_tests/test_moe.py:199 and
# op_tests/test_moe_tkw1.py:194 on upstream main.
MOE_RTOL, MOE_ATOL = 0.01, 100

# The tuner's acceptance threshold, csrc/.../gemm_moe_tune.py:116.
COS_DIFF_THRESHOLD = 1e-1


def rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


# ---------------------------------------------------------------------------
# 1. A kernel that is wrong in every element exits 0.
# ---------------------------------------------------------------------------

_MINIATURE_OP_TEST = """
import torch
from aiter.test_common import checkAllclose
ref = torch.ones(64, 128)
checkAllclose(ref * 10.0, ref, msg="planted defect ")
print("op test body completed")
"""


def escape_1_exit_code():
    rule("1. checkAllclose computes a verdict and discards it")
    proc = subprocess.run(
        [sys.executable, "-c", _MINIATURE_OP_TEST],
        capture_output=True,
        text=True,
        check=False,
    )
    printed_failed = "failed!" in (proc.stdout + proc.stderr)
    print("  every element wrong by 10x")
    print(f"  the comparison printed 'failed!' : {printed_failed}")
    print(f"  process exit code                : {proc.returncode}")
    print(f"  -> CI is green: {proc.returncode == 0}")
    return proc.returncode == 0 and printed_failed


# ---------------------------------------------------------------------------
# 2. The hardcoded MoE atol=100 absorbs a large constant bias.
# ---------------------------------------------------------------------------


def escape_2_magic_tolerance():
    rule("2. MoE atol=100 is not tied to the arithmetic")
    g = torch.Generator().manual_seed(0)
    ref = torch.randn(512, 128, generator=g, dtype=torch.float32)
    ref = (ref / ref.abs().max() * 1928.0).to(torch.bfloat16)
    broken = (ref.float() + 99.0).to(torch.bfloat16)

    ratio = checkAllclose(broken, ref, rtol=MOE_RTOL, atol=MOE_ATOL, printLog=False)
    print(f"  max|ref|                         : {ref.abs().max().item():.1f}")
    print("  planted bias on every element    : +99.0")
    print(f"  shipped gate rtol={MOE_RTOL} atol={MOE_ATOL}")
    print(f"  mismatching elements             : {ratio:.1%}")
    print(f"  -> accepted: {ratio == 0}")
    return ratio == 0


# ---------------------------------------------------------------------------
# 3. The tuner's whole-tensor cos_diff dilutes a localised defect.
# ---------------------------------------------------------------------------


def escape_3_cosine_dilution():
    rule("3. Tuner cos_diff is a whole-tensor average")
    g = torch.Generator().manual_seed(0)
    ref = torch.randn(512, 128, generator=g, dtype=torch.float64)
    ref = ref / ref.norm(dim=1, keepdim=True)

    dropped = int(0.17 * ref.shape[0])
    out = ref.clone()
    out[:dropped] = 0.0

    # Verbatim from csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py:286.
    x, y = ref.flatten(), out.flatten()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)

    f = dropped / ref.shape[0]
    print(f"  output rows zeroed               : {dropped}/{ref.shape[0]} ({f:.0%})")
    print(f"  closed form f/(2-f)              : {f / (2 - f):.6f}")
    print(f"  tuner cos_diff                   : {cos_diff:.6f}")
    print(f"  acceptance threshold             : {COS_DIFF_THRESHOLD}")
    print(f"  -> accepted as best kernel: {cos_diff < COS_DIFF_THRESHOLD}")
    return cos_diff < COS_DIFF_THRESHOLD


# ---------------------------------------------------------------------------
# 4. A gate that cannot fail, already in the shipped suite.
# ---------------------------------------------------------------------------


def escape_4_disabled_gate():
    rule("4. op_tests/test_gemm_a8w8.py:189 -- tuned fp8 shapes")
    g = torch.Generator().manual_seed(0)
    ref = torch.randn(256, 256, generator=g)
    garbage = torch.randn(256, 256, generator=g) * 1000.0

    # The exact arguments used for shapes that have a tuned config, i.e. the
    # configs that ship.
    ratio = checkAllclose(
        garbage, ref, rtol=1e-1, atol=1e-1, tol_err_ratio=1.0, printLog=False
    )
    print("  comparing noise against the reference")
    print("  gate: rtol=1e-1 atol=1e-1 tol_err_ratio=1.0 printLog=False")
    print(f"  mismatching elements             : {ratio:.1%}")
    print("  threshold it is judged against   : 100.0%")
    print("  nothing is printed (printLog=False), nothing is raised")
    print(f"  -> cannot fail by construction: {ratio <= 1.0}")
    return ratio <= 1.0


# ---------------------------------------------------------------------------
# The other half: the same three defects, judged by the gates the PR adds.
# Skipped automatically on a stock checkout, where they do not exist.
# ---------------------------------------------------------------------------


def gates_available():
    import inspect

    try:
        import aiter.utility.cos_diff
        import aiter.utility.tolerance  # noqa: F401
    except ImportError:
        return False
    return "strict" in inspect.signature(checkAllclose).parameters


def catches():
    from aiter.utility.cos_diff import worst_row_cos_diff
    from aiter.utility.tolerance import derive_tolerance

    rule("Same three defects, judged by the gates this branch adds")

    env = dict(os.environ, AITER_STRICT_ALLCLOSE="1")
    proc = subprocess.run(
        [sys.executable, "-c", _MINIATURE_OP_TEST],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    print(f"  1. AITER_STRICT_ALLCLOSE=1       -> exit code {proc.returncode}")

    g = torch.Generator().manual_seed(0)
    ref = torch.randn(512, 128, generator=g, dtype=torch.float32)
    ref = (ref / ref.abs().max() * 1928.0).to(torch.bfloat16)
    broken = (ref.float() + 99.0).to(torch.bfloat16)
    rtol, atol = derive_tolerance(
        torch.float8_e4m3fnuz,
        torch.bfloat16,
        num_accumulations=3840,
        max_value=ref.abs().max().item(),
    )
    ratio = checkAllclose(broken, ref, rtol=rtol, atol=atol, printLog=False)
    print(f"  2. derived rtol={rtol} atol={atol}  -> {ratio:.1%} of elements flagged")

    g = torch.Generator().manual_seed(0)
    ref = torch.randn(512, 128, generator=g, dtype=torch.float64)
    ref = ref / ref.norm(dim=1, keepdim=True)
    out = ref.clone()
    out[: int(0.17 * ref.shape[0])] = 0.0
    print(f"  3. worst-row cos_diff            -> {worst_row_cos_diff(ref, out):.6f}")

    return (
        proc.returncode != 0
        and ratio > 0.05
        and worst_row_cos_diff(ref, out) > COS_DIFF_THRESHOLD
    )


if __name__ == "__main__":
    print(f"python   : {sys.version.split()[0]}")
    print(f"torch    : {torch.__version__}")
    print(f"aiter    : {os.environ.get('PYTHONPATH', '(not set)')}")

    results = {
        "exit code": escape_1_exit_code(),
        "magic tolerance": escape_2_magic_tolerance(),
        "cosine dilution": escape_3_cosine_dilution(),
        "disabled gate": escape_4_disabled_gate(),
    }

    rule("Summary")
    for name, escaped in results.items():
        print(f"  {name:<20} defect went undetected: {escaped}")

    caught = None
    if gates_available():
        caught = catches()
        print(f"\n  all three caught by the new gates: {caught}")
    else:
        print("\n  (stock checkout: the gates this PR adds are not present)")

    sys.exit(0 if all(results.values()) and caught is not False else 1)
