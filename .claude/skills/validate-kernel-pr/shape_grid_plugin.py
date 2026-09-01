"""Validator-owned pytest plugin: deliver the S1 shape grid to a parametrized target.

WHY A THIRD CHANNEL
-------------------
The two existing channels require the target to have been written for them: an environment
variable it reads (`--shape-env`), or a CLI flag it parses (`--shape-arg`). Measured on
ROCm/aiter, that leaves the stage inert on the repository's dominant test shape. Zero of the
seven files in `op_tests/flydsl_tests/` expose either channel; every one of them declares its
shapes as literals inside `@pytest.mark.parametrize`. Four consecutive real FlyDSL kernel PRs
reached `INCONCLUSIVE` for this reason alone, and the skip text blamed the kernel for a limit
that belonged to the injector.

Parametrization is pytest's own shape channel, so this plugin uses it rather than asking
kernel authors to add a hook for the validator's benefit.

WHAT IT DOES NOT CHANGE
-----------------------
The burden of proof is identical to the other two channels. The executor still runs the
invalid-grid probe first: if feeding this plugin a deliberately unusable grid does NOT make
the target fail, the target is not consuming the grid, and `correctness_s1_grid` is not
credited. A plugin that quietly failed to override would therefore produce a skip, never a
pass. The plugin is generated per run with the grid baked in, so the tested PR cannot reach
it or forge its effect.
"""

import pytest

# Filled in by the executor when this module is generated.
_VALIDATION_SHAPE_ARGNAMES: tuple = ()
_VALIDATION_SHAPE_GRID: list = []


def _bound_names(mark):
    """Argument names a parametrize mark binds, in either accepted spelling."""
    if not mark.args:
        return []
    argnames = mark.args[0]
    if isinstance(argnames, str):
        return [name.strip() for name in argnames.split(",") if name.strip()]
    return [str(name) for name in argnames]


@pytest.hookimpl(tryfirst=True)
def pytest_generate_tests(metafunc):
    names = list(_VALIDATION_SHAPE_ARGNAMES)
    if not names or not _VALIDATION_SHAPE_GRID:
        return
    if not set(names) <= set(metafunc.fixturenames):
        # This test does not take the named shape arguments. Leaving it alone is correct: the
        # invalid-grid probe decides whether the grid was delivered, so a target this plugin
        # cannot reach produces a skip rather than an unearned pass.
        return

    # Drop the target's own parametrization of these names before the builtin hook runs,
    # otherwise pytest raises on duplicate parametrization. tryfirst puts this ahead of
    # _pytest.python's implementation.
    #
    # A mark that binds the requested names TOGETHER WITH others cannot be stripped: removing
    # it would leave those other names as unfilled fixtures and the target would error for a
    # reason that has nothing to do with the kernel. Observed on a real target declaring
    # `@pytest.mark.parametrize(("num_heads", "with_sink"), [...])` when only `num_heads` was
    # named. Refuse instead, so the invalid-grid probe leaves the stage uncredited and the
    # reason tells the caller to name the whole tuple.
    kept = []
    for mark in metafunc.definition.own_markers:
        if mark.name == "parametrize":
            bound = set(_bound_names(mark))
            if bound & set(names):
                if not bound <= set(names):
                    return
                continue
        kept.append(mark)
    metafunc.definition.own_markers = kept

    rows = [tuple(row) for row in _VALIDATION_SHAPE_GRID]
    if len(names) == 1:
        rows = [row[0] for row in rows]
    metafunc.parametrize(
        names,
        rows,
        ids=[f"s1grid{index}" for index in range(len(rows))],
    )
