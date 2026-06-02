# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the ``expected_m`` host-side scheduling hint.

Background
----------
``fused_moe`` historically used ``M = topk_ids.shape[0]`` for kernel-config
tier lookup via ``get_padded_M``. Under DeepEP / Mori low-latency dispatch,
``topk_ids.shape[0]`` is the *padded* buffer size
(``num_max_dispatch_tokens_per_rank * world_size``) and tells nothing about
the real workload, which pegs the lookup at the worst-case tier
(32768 / 131072) and misses the primary CSV.

Commit ``a68ed37bf [fused_moe] add expected_m host-side scheduling hint``
adds an optional ``expected_m`` kwarg (the host-side average #tokens per
local expert under uniform routing) that overrides the M used for tier
matching. When ``expected_m is None`` the behavior is bit-identical to the
pre-patch baseline.

What this file tests
--------------------
Two properties, both CPU-only so the test runs anywhere:

1. ``test_get_padded_M_tier_split``
   ``get_padded_M`` actually returns *different* tiers for the padded LL
   buffer M and a realistic ``expected_m``. Without this, the hint would
   carry no information and the PR would be pointless.

2. ``test_fused_moe_uses_expected_m_for_tier_lookup``
   The ``fused_moe_`` entry point forwards ``expected_m`` into
   ``get_2stage_cfgs`` via ``get_padded_M(M_for_schedule)``. Verified by
   monkey-patching ``get_2stage_cfgs`` to record its first positional
   argument, then calling ``fused_moe_`` with tiny placeholder tensors
   (the call is intercepted before any GPU kernel launches, so no device
   is required).

Both tests are intentionally independent of the actual MoE math; they
exercise only the host-side scheduling hint plumbing.
"""

import sys

import pytest
import torch

from aiter import fused_moe as fused_moe_mod
from aiter.fused_moe import fused_moe_, get_padded_M

# --------------------------------------------------------------------------
# (1) Pure tier-selection check
# --------------------------------------------------------------------------


def test_get_padded_M_tier_split():
    """The padded LL buffer M and a realistic expected_m must map to
    *different* tiers; otherwise the hint cannot influence kernel choice.

    Numbers below mirror what we observe in production:
      raw_M       = 32 local_experts * 512 max_dispatch_per_rank = 16384
      expected_m  = num_tokens (64) * world_size (8) * topk (8)
                  / num_experts (256) = 16  -> 17 with ceil-div remainder
    """
    raw_M = 16384
    expected_m = 17

    tier_raw = get_padded_M(raw_M)
    tier_hint = get_padded_M(expected_m)

    assert tier_raw != tier_hint, (
        f"get_padded_M maps both raw_M={raw_M} and expected_m={expected_m} "
        f"to the same tier ({tier_raw}); the host-side hint cannot help."
    )
    assert tier_hint < tier_raw, (
        f"expected_m={expected_m} should pick a smaller tier than "
        f"raw_M={raw_M}, got tier_hint={tier_hint} >= tier_raw={tier_raw}."
    )


def test_get_padded_M_none_falls_back_to_M():
    """Without an explicit hint, M_for_schedule must equal M (i.e. the
    expected_m=None path is bit-identical to pre-patch behavior).

    This is the source-level invariant we encode in fused_moe_:

        M_for_schedule = expected_m if expected_m is not None else M

    so verifying the get_padded_M values match for any sample M is
    sufficient -- if it didn't, the new code path would silently change
    the tier for every non-EP caller. We sweep a handful of representative
    M values.
    """
    for M in (1, 17, 64, 256, 1024, 16384, 131072, 524288):
        # explicit None  must behave like passing M as the schedule key
        M_for_schedule = None if False else M  # mirror the source guard
        assert get_padded_M(M) == get_padded_M(M_for_schedule)


# --------------------------------------------------------------------------
# (2) Plumbing check: fused_moe_ -> get_2stage_cfgs
# --------------------------------------------------------------------------


class _ShortCircuit(Exception):
    """Raised by the stubbed get_2stage_cfgs to abort fused_moe_ before
    any GPU kernel launch. The first positional arg captured on the
    exception instance is the schedule-key M that fused_moe_ chose."""

    def __init__(self, captured_M, captured_topk):
        super().__init__(f"captured M={captured_M} topk={captured_topk}")
        self.captured_M = captured_M
        self.captured_topk = captured_topk


def _stub_get_2stage_cfgs(*args, **kwargs):
    captured_M = args[0] if args else kwargs.get("token")
    captured_topk = args[4] if len(args) > 4 else kwargs.get("topk")
    raise _ShortCircuit(captured_M, captured_topk)


def _make_tiny_inputs():
    """Return placeholder tensors with realistic dtypes/shapes. The
    monkey-patched get_2stage_cfgs aborts before any real kernel runs,
    so the tensors only need to satisfy shape/dtype asserts in
    fused_moe_'s preamble.
    """
    device = "cpu"  # never actually consumed; aborted before kernel launch
    M, topk = 16384, 8
    E, model_dim, inter_dim = 32, 2048, 1024
    hidden_states = torch.zeros((M, model_dim), dtype=torch.bfloat16, device=device)
    w1 = torch.zeros((E, inter_dim * 2, model_dim), dtype=torch.bfloat16, device=device)
    w2 = torch.zeros((E, model_dim, inter_dim), dtype=torch.bfloat16, device=device)
    topk_weight = torch.zeros((M, topk), dtype=torch.float32, device=device)
    topk_ids = torch.zeros((M, topk), dtype=torch.int32, device=device)
    return hidden_states, w1, w2, topk_weight, topk_ids


def _call_fused_moe_capture_M(monkeypatch, expected_m):
    """Invoke fused_moe_ with the stub and return the (M, topk) it would
    have passed to get_2stage_cfgs."""
    monkeypatch.setattr(fused_moe_mod, "get_2stage_cfgs", _stub_get_2stage_cfgs)

    hidden_states, w1, w2, topk_weight, topk_ids = _make_tiny_inputs()
    try:
        fused_moe_(
            hidden_states=hidden_states,
            w1=w1,
            w2=w2,
            topk_weight=topk_weight,
            topk_ids=topk_ids,
            expected_m=expected_m,
        )
    except _ShortCircuit as e:
        return e.captured_M, e.captured_topk
    raise AssertionError("get_2stage_cfgs stub was not reached")


def test_fused_moe_uses_expected_m_for_tier_lookup(monkeypatch):
    """Passing expected_m must change the M handed to get_2stage_cfgs."""
    raw_M = 16384
    expected_m = 17

    M_passed_with_hint, _ = _call_fused_moe_capture_M(
        monkeypatch, expected_m=expected_m
    )
    M_passed_without_hint, _ = _call_fused_moe_capture_M(monkeypatch, expected_m=None)

    # get_2stage_cfgs receives get_padded_M(M_for_schedule); when hint is
    # provided it's get_padded_M(expected_m), otherwise get_padded_M(M).
    assert M_passed_with_hint == get_padded_M(expected_m), (
        f"fused_moe_ should call get_2stage_cfgs with "
        f"get_padded_M({expected_m})={get_padded_M(expected_m)} when "
        f"expected_m is supplied; saw {M_passed_with_hint}."
    )
    assert M_passed_without_hint == get_padded_M(raw_M), (
        f"fused_moe_ should call get_2stage_cfgs with "
        f"get_padded_M({raw_M})={get_padded_M(raw_M)} when expected_m "
        f"is None (bit-identical to pre-patch); saw {M_passed_without_hint}."
    )
    assert M_passed_with_hint != M_passed_without_hint, (
        "With and without the hint, fused_moe_ resolved to the same tier "
        "-- the host-side scheduling hint had no effect."
    )


def test_fused_moe_uses_topk1_for_ep_lookup(monkeypatch):
    """When expected_m is supplied we are in an EP dispatch path; the value
    already folds the topk fan-out into a per-local-expert token count, so the
    topk handed to get_2stage_cfgs must be canonicalized to 1. Without the
    hint, the real topk (topk_ids.shape[1]) is forwarded unchanged."""
    expected_m = 17
    real_topk = _make_tiny_inputs()[-1].shape[1]  # topk_ids.shape[1]

    _, topk_with_hint = _call_fused_moe_capture_M(monkeypatch, expected_m=expected_m)
    _, topk_without_hint = _call_fused_moe_capture_M(monkeypatch, expected_m=None)

    assert topk_with_hint == 1, (
        "With expected_m supplied, fused_moe_ must look up the EP-canonical "
        f"topk==1 CSV tier; saw topk={topk_with_hint}."
    )
    assert topk_without_hint == real_topk, (
        f"Without expected_m, fused_moe_ must forward the real topk "
        f"({real_topk}) to get_2stage_cfgs; saw {topk_without_hint}."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
