# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter.spec_verify_pick off the GPU: the torch definition of the contract.

CPU tensors take ``_spec_verify_pick_ref``, so these run without a GPU. The
kernel side of the same rules is op_tests/test_spec_verify_pick.py.
"""

import pytest
import torch

from aiter.ops import sample

EPS = 1e-10
V = 16


def _call(logits, temps, target, bonus, cu, expo_t=None, expo_b=None, **kw):
    n_t, n_req = len(target), len(bonus)
    expo_t = torch.ones(n_t, V) if expo_t is None else expo_t
    expo_b = torch.ones(n_req, V) if expo_b is None else expo_b
    out_t = torch.full((n_t,), -1, dtype=torch.int32)
    out_b = torch.full((n_req,), -1, dtype=torch.int32)
    sample.spec_verify_pick(
        out_t,
        out_b,
        logits,
        expo_t,
        expo_b,
        temps,
        torch.tensor(target, dtype=torch.int64),
        torch.tensor(bonus, dtype=torch.int64),
        cu,
        **kw,
    )
    return out_t.tolist(), out_b.tolist()


def test_eps_and_greedy_eps_are_keyword_only():
    logits = torch.zeros(1, V)
    args = (
        torch.zeros(0, dtype=torch.int32),
        torch.zeros(1, dtype=torch.int32),
        logits,
        torch.ones(0, V),
        torch.ones(1, V),
        torch.zeros(1),
        torch.zeros(0, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int64),
        torch.zeros(1, dtype=torch.int32),
    )
    # The routed op's schema puts them after ``*``; the dispatcher rejects a
    # positional call as a TypeError or RuntimeError depending on the path.
    with pytest.raises((TypeError, RuntimeError), match="positional"):
        sample.spec_verify_pick(*args, EPS, EPS)
    sample.spec_verify_pick(*args, eps=EPS, greedy_eps=EPS)


def test_greedy_key_order():
    """tie="low", -0.0 == 0.0, NaN above +inf, an all -inf row picks 0."""
    logits = torch.full((4, V), float("-inf"))
    logits[0, 7], logits[0, 5] = 0.0, -0.0
    logits[1, 3], logits[1, 9] = float("inf"), float("nan")
    logits[2, 3], logits[2, 2] = 1.0, 1.0
    temps = torch.zeros(4)
    cu = torch.zeros(4, dtype=torch.int32)
    _, ob = _call(logits, temps, [], [0, 1, 2, 3], cu, greedy_eps=EPS)
    assert ob == [5, 9, 2, 0]


def test_temperature_zero_is_greedy_on_every_row():
    logits = torch.randn(5, V, generator=torch.Generator().manual_seed(0))
    noise = torch.rand(3, V) + 0.01  # would move a sampled pick
    ot, ob = _call(
        logits,
        torch.zeros(2),
        [0, 1, 3],
        [2, 4],
        torch.tensor([2, 3], dtype=torch.int32),
        expo_t=noise,
        expo_b=noise[:2],
        eps=EPS,
        greedy_eps=EPS,
    )
    want = logits.argmax(dim=-1).tolist()
    assert ot == [want[0], want[1], want[3]] and ob == [want[2], want[4]]


@pytest.mark.parametrize(
    "temps, cu, target, bonus",
    [
        # Verification rows with no requests to own them.
        (torch.zeros(0), torch.zeros(0, dtype=torch.int32), [0, 1], []),
        # A 0-dim temperature.
        (torch.tensor(0.0), torch.tensor([1], dtype=torch.int32), [0], [1]),
        # One temperature for two requests.
        (torch.zeros(1), torch.tensor([1, 1], dtype=torch.int32), [0], [1, 2]),
        # Prefix sums that do not match the requests.
        (torch.zeros(2), torch.tensor([1], dtype=torch.int32), [0], [1, 2]),
    ],
)
def test_malformed_temperatures_raise_value_error(temps, cu, target, bonus):
    with pytest.raises(ValueError, match="spec_verify_pick"):
        _call(torch.zeros(3, V), temps, target, bonus, cu, greedy_eps=EPS)


def test_temperature_below_the_clamp_samples_at_the_clamp():
    """T in (greedy_eps, 1e-5] samples at T = 1e-5 (the kernel clamps the
    same way), so the pick equals the T = 1e-5 pick and, on a row whose top
    two logits are far apart against T, the argmax."""
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(3, V, generator=g)
    noise = torch.empty(3, V).exponential_(1, generator=g)
    picks = {}
    for t in (1e-9, 1e-7, 1e-5):
        picks[t] = _call(
            logits,
            torch.full((3,), t),
            [],
            [0, 1, 2],
            torch.zeros(3, dtype=torch.int32),
            expo_b=noise,
            eps=EPS,
            greedy_eps=1e-10,
        )[1]
    assert picks[1e-9] == picks[1e-7] == picks[1e-5]
    assert picks[1e-9] == logits.argmax(dim=-1).tolist()


def test_sampled_rows_follow_the_documented_law():
    """argmax(softmax(x / max(T, 1e-5)) / (expo + eps)), the owner's
    temperature on a verification row."""
    g = torch.Generator().manual_seed(2)
    logits = torch.randn(5, V, generator=g) * 3
    expo = torch.empty(5, V).exponential_(1, generator=g)
    temps = torch.tensor([0.7, 1.3])
    cu = torch.tensor([2, 3], dtype=torch.int32)
    target, bonus = [0, 1, 3], [2, 4]
    ot, ob = _call(
        logits,
        temps,
        target,
        bonus,
        cu,
        expo_t=expo[:3],
        expo_b=expo[3:],
        eps=EPS,
        greedy_eps=EPS,
    )
    t_rows = temps[torch.tensor([0, 0, 1])]

    def law(rows, t, e):
        p = torch.softmax(logits[rows] / t.unsqueeze(-1), dim=-1)
        return (p / (e + EPS)).argmax(dim=-1).tolist()

    assert ot == law(target, t_rows, expo[:3])
    assert ob == law(bonus, temps, expo[3:])
