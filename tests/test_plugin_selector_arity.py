# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The arity cross-check the kernel-plugin skill tells provider authors to run.

A plugin's ``select()`` is handed the contract's full positional list, defaults
already applied, so the number of names it unpacks *is* the contract. Getting
that number wrong is not a crash: AITER's candidate check compares parameter
counts, so a selector that unpacks the wrong arity for an op whose contract has
the same count passes every gate and reads the wrong argument.

Checking it is a dozen lines of ``ast`` and ``json`` -- no torch, no GPU, no
provider installed. Those dozen lines live in the skill, which makes them
advice. They live here too, which makes them tested: the fixtures below prove
the check catches a real mismatch, so a provider team copying it is copying
something that has been run.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

BASELINE = Path(__file__).with_name("plugin_contracts.json")


def contract_arity(baseline: dict) -> dict[str, int]:
    """Positional parameters per routed op, read from the committed baseline."""

    return {
        op: len([p for p in c["signature"].split(" -> ")[0].split(", ") if p.strip()])
        for op, c in baseline.items()
    }


def selector_mismatches(source: str, arity: dict[str, int]) -> list[str]:
    """Every selector in ``_SELECTORS`` whose unpacked arity is not its contract's."""

    tree = ast.parse(source)
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    table = next(
        (
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and getattr(node.targets[0], "id", "") == "_SELECTORS"
        ),
        None,
    )
    if table is None:
        return ["no _SELECTORS table found"]

    problems = []
    for op_node, fn_node in zip(table.keys, table.values):
        op = op_node.value
        if op not in arity:
            problems.append(f"{op}: not a routed op in the baseline")
            continue
        body = functions[fn_node.id].body
        unpack = next(
            (
                s
                for s in body
                if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Tuple)
            ),
            None,
        )
        got = len(unpack.targets[0].elts) if unpack is not None else None
        if got != arity[op]:
            problems.append(
                f"{op}: contract takes {arity[op]}, {fn_node.id} unpacks {got}"
            )
    return problems


_FOUR = {
    "demo.norm": {
        "signature": "a:positional, b:positional, c:positional, d:positional, -> Tensor"
    }
}

_GOOD = """
def _select_norm(args):
    value, weight, epsilon, sensitive = args
    return None

_SELECTORS = {"demo.norm": _select_norm}
"""

_SHORT = """
def _select_norm(args):
    value, weight, epsilon = args
    return None

_SELECTORS = {"demo.norm": _select_norm}
"""


def test_a_selector_matching_its_contract_reports_nothing():
    assert selector_mismatches(_GOOD, contract_arity(_FOUR)) == []


def test_a_selector_short_by_one_is_named_with_both_counts():
    """The message has to say which op and both numbers, or it is not actionable."""

    (problem,) = selector_mismatches(_SHORT, contract_arity(_FOUR))
    assert "demo.norm" in problem
    assert "takes 4" in problem and "unpacks 3" in problem


def test_an_op_outside_the_baseline_is_reported_rather_than_skipped():
    """Serving an op AITER does not route is a defect, not a silent no-op."""

    (problem,) = selector_mismatches(_GOOD, {})
    assert "not a routed op" in problem


@pytest.mark.skipif(not BASELINE.is_file(), reason="no committed baseline")
def test_the_committed_baseline_still_parses_the_way_the_check_assumes():
    """The check reads `signature` by splitting on ' -> ' and ', '.

    If the baseline's shape ever changes, every provider team's copy of this
    silently reports nothing. Fail here instead.
    """

    baseline = json.loads(BASELINE.read_text())
    for op, contract in baseline.items():
        assert "signature" in contract, f"{op}: baseline entry has no signature"
        assert (
            contract_arity({op: contract})[op] > 0
        ), f"{op}: parsed as zero parameters"
