# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Every routing op that writes 32-bit ids must reject a wider id buffer.

These kernels write ids through a raw pointer that is reinterpreted as
``int32_t*`` (and weights as ``float*``). Handed an ``int64`` buffer they wrote
four bytes per element and left the rest holding whatever the caller allocated
-- a tensor fully populated in shape and dtype, and mostly garbage in value.
Nothing raised, so the corruption surfaced far away: as an out-of-bounds write
inside the consumer, or as duplicate expert ids tripping an assert in a
different library.

The positive behaviour is covered elsewhere; what was missing is the guarantee
that the *wrong* dtype cannot get through silently. Each case below is an entry
point separately reachable from Python, so a check on one says nothing about
the others -- ``topk_softmax`` alone has a HIP path, an ASM path and a sigmoid
sibling, and ``biased_grouped_topk`` re-dispatches to ``moe_fused_gate`` purely
on token count.

Each case runs in a subprocess. ``AITER_CHECK`` in these translation units
calls ``aiter_detail::check_fail``, which prints to stderr and then either
throws or calls ``abort()`` depending on ``g_aiter_can_throw`` -- and none of
these files opt into the ctypes error translation that sets it, so today they
abort. That is pre-existing behaviour shared with the ~34 AITER_CHECKs already
in these same files, not something these checks introduce, but it does mean
``pytest.raises`` cannot see them. Asserting on the message the process died
with is correct under either mode.
"""

import subprocess
import sys
import textwrap

import pytest

PREAMBLE = """
import torch
from aiter import dtypes
from aiter.jit.utils.chip_info import get_cu_num
from aiter.ops.moe_op import topk_sigmoid, topk_softmax, topk_softmax_asm
from aiter.ops.topk import biased_grouped_topk, grouped_topk, moe_fused_gate

torch.set_default_device("cuda")

TOKENS, EXPERTS, TOPK, GROUPS = 64, 32, 4, 4
gating = lambda tokens=TOKENS: torch.randn(tokens, EXPERTS, dtype=dtypes.bf16)
bias = lambda: torch.randn(EXPERTS, dtype=dtypes.fp32)
weights = lambda tokens=TOKENS, dtype=dtypes.fp32: torch.empty(tokens, TOPK, dtype=dtype)
ids = lambda dtype, tokens=TOKENS: torch.empty(tokens, TOPK, dtype=dtype)
"""


def run_case(body: str):
    """Run one call in a fresh interpreter; return (returncode, stderr)."""
    proc = subprocess.run(
        [sys.executable, "-c", PREAMBLE + textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=1800,
    )
    return proc.returncode, proc.stderr


def assert_rejected(body: str, expected: str):
    rc, err = run_case(body)
    assert rc != 0, f"call was accepted; it should have been rejected\nstderr:\n{err}"
    assert expected in err, f"rejected, but not for the expected reason\nstderr:\n{err}"


def test_topk_softmax_rejects_int64_indices():
    assert_rejected(
        "topk_softmax(weights(), ids(torch.int64), ids(dtypes.i32), gating(), False)",
        "topk_indices must be int32",
    )


def test_topk_softmax_rejects_int64_token_expert_indices():
    assert_rejected(
        "topk_softmax(weights(), ids(dtypes.i32), ids(torch.int64), gating(), False)",
        "token_expert_indices must be int32",
    )


def test_topk_softmax_rejects_non_fp32_weights():
    assert_rejected(
        "topk_softmax(weights(dtype=dtypes.bf16), ids(dtypes.i32), ids(dtypes.i32),"
        " gating(), False)",
        "topk_weights must be float32",
    )


@pytest.mark.skipif(
    __import__("aiter.jit.utils.chip_info", fromlist=["get_gfx"]).get_gfx()
    not in ("gfx942", "gfx950"),
    reason="the asm topk_softmax is only built for gfx942/gfx950",
)
def test_topk_softmax_asm_rejects_int64_indices():
    # This path scales its output stride by a hard-coded 4 bytes per element,
    # so it is the one most easily missed when only the HIP entry point is
    # guarded.
    assert_rejected(
        "topk_softmax_asm(weights(), ids(torch.int64), ids(dtypes.i32), gating(), False)",
        "topk_indices must be int32",
    )


def test_topk_sigmoid_rejects_int64_indices():
    assert_rejected(
        "topk_sigmoid(weights(), ids(torch.int64), gating())",
        "topk_indices must be int32",
    )


def test_grouped_topk_rejects_int64_ids():
    assert_rejected(
        "grouped_topk(gating(), weights(), ids(torch.int64), GROUPS, 2, False)",
        "topk_ids must be int32",
    )


def test_biased_grouped_topk_rejects_int64_ids():
    assert_rejected(
        "biased_grouped_topk(gating(), bias(), weights(), ids(torch.int64),"
        " GROUPS, 2, True)",
        "topk_ids must be int32",
    )


def test_biased_grouped_topk_rejects_int64_ids_on_the_fused_gate_route():
    # biased_grouped_topk dispatches to moe_fused_gate -- a different launcher,
    # separately unguarded -- once the token count passes cu_num * 212 and the
    # group is small enough. Guarding only the HIP path leaves this reachable
    # from the same public call, so the threshold is computed, not assumed.
    assert_rejected(
        """
        n = get_cu_num() * 212 + 1
        biased_grouped_topk(gating(n), bias(), weights(n), ids(torch.int64, n),
                            GROUPS, 2, True)
        """,
        "topk_ids must be int32",
    )


def test_moe_fused_gate_rejects_int64_ids():
    assert_rejected(
        "moe_fused_gate(gating(), bias(), weights(), ids(torch.int64),"
        " GROUPS, 2, TOPK, 0)",
        "topk_ids must be int32",
    )


if __name__ == "__main__":
    # The op-test CI runs these files as `python3 <file>` rather than through
    # pytest (.github/scripts/aiter_test.sh:85). Without this the module would
    # define nine tests, run none of them, and exit 0 -- reported as a pass.
    # Which is the same failure mode this whole file exists to rule out.
    sys.exit(pytest.main([__file__, "-v"]))
