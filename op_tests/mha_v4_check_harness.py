# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Minimal runner for the mha_v4 op_tests.

CI runs op_tests files as `python3 <file>`, so these checks have to execute without pytest.
`parametrize` and `skipif` keep the per-check case tables declarative; `CheckRegistry.run`
collects every test_* in the module and runs it.
"""

import contextlib
import itertools
import re
import traceback

from aiter import logger


class Param:
    """One parametrize entry, optionally disabled by its own condition."""

    __slots__ = ("id", "skip", "values")

    def __init__(self, values, skip=False, id=None):
        self.values = values
        self.skip = bool(skip)
        self.id = id


def param(*values, skip=False, id=None):
    return Param(values, skip, id)


def parametrize(argnames, argvalues, ids=None):
    """Attach a case table. Stacked decorators multiply, and values bind by argument name."""
    names = (
        [name.strip() for name in argnames.split(",")]
        if isinstance(argnames, str)
        else list(argnames)
    )

    def wrap(fn):
        table = []
        for entry in argvalues:
            skip = False
            if isinstance(entry, Param):
                skip = entry.skip
                entry = entry.values if len(names) > 1 else entry.values[0]
            values = tuple(entry) if len(names) > 1 else (entry,)
            table.append((dict(zip(names, values)), skip))
        fn.case_tables = [table, *getattr(fn, "case_tables", [])]
        return fn

    return wrap


def skipif(condition, reason=""):
    """Disable a check when `condition` holds, evaluated at import like pytest's."""

    def wrap(fn):
        fn.skip_reasons = [*getattr(fn, "skip_reasons", []), (bool(condition), reason)]
        return fn

    return wrap


@contextlib.contextmanager
def patched_attr(obj, name, value):
    original = getattr(obj, name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        setattr(obj, name, original)


@contextlib.contextmanager
def expect_raises(exc, match=None):
    try:
        yield
    except exc as caught:
        if match is not None and not re.search(match, str(caught)):
            raise AssertionError(
                f"expected {match!r} in {type(caught).__name__}: {caught}"
            ) from None
        return
    raise AssertionError(f"expected {exc.__name__} but nothing was raised")


class CheckRegistry:
    """Runs every test_* in a module. `setup`/`teardown` replace the autouse fixtures."""

    def __init__(self, setup=None, teardown=None):
        self._setup = setup
        self._teardown = teardown

    @staticmethod
    def cases(fn):
        """Expand a check's stacked case tables into (kwargs, skipped) pairs."""
        tables = getattr(fn, "case_tables", None)
        if not tables:
            return [({}, False)]
        expanded = []
        for combo in itertools.product(*tables):
            values = {}
            skipped = False
            for case, case_skipped in combo:
                values.update(case)
                skipped = skipped or case_skipped
            expanded.append((values, skipped))
        return expanded

    def run(self, namespace):
        failures, ran, skipped = [], 0, 0
        for name, fn in sorted(namespace.items()):
            if not name.startswith("test_") or not callable(fn):
                continue
            gated = any(cond for cond, _ in getattr(fn, "skip_reasons", []))
            for case, case_skipped in self.cases(fn):
                if gated or case_skipped:
                    skipped += 1
                    continue
                shown = ",".join(f"{key}={value!r}" for key, value in case.items())
                label = f"{name}({shown})"
                if self._setup is not None:
                    self._setup()
                try:
                    fn(**case)
                    ran += 1
                # One bad check must not hide the rest; they are re-raised in aggregate.
                # The traceback is the whole report: a bare assert carries no message.
                except Exception:  # noqa: BLE001
                    failures.append(label)
                    logger.error("check %s failed:\n%s", label, traceback.format_exc())
                finally:
                    if self._teardown is not None:
                        self._teardown()
        logger.info(
            "checks: %d ran, %d skipped, %d failed", ran, skipped, len(failures)
        )
        return failures
