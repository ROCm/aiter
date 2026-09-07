"""Run a script target under the validator's own route profiler.

The profiling in ``validation_probe`` has nothing to do with pytest —
``pytest_configure``/``pytest_sessionfinish`` are only the injection points, and the work is
``sys.setprofile`` plus writing the receipt. A script target driven by a ``__main__`` guard
therefore has no reason to forgo a receipt; it just had no place to install the hook.

This shim supplies that place. It deliberately calls the probe's own hooks rather than
re-implementing them, so ``producer`` in the receipt stays truthful: the receipt is still
produced by ``validate-kernel-pr.validation_probe`` and the tested PR still cannot forge one.

    run_script_with_probe.py <probe-module-name> <target-file> [args...]

Exit code and stdout/stderr are the target's own, so the caller's existing pass/fail
attribution is unchanged.
"""

from __future__ import annotations

import importlib
import runpy
import sys


class _Config:
    """Stands in for pytest's ``config`` object.

    The probe only ever uses it as a place to hang ``_validation_probe``, so an empty
    namespace is a complete substitute rather than a partial fake.
    """


class _Session:
    def __init__(self, config):
        self.config = config


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: run_script_with_probe.py <probe-module> <target-file> [args...]",
            file=sys.stderr,
        )
        return 2
    probe_module, target = argv[0], argv[1]
    target_argv = argv[1:]

    probe = importlib.import_module(probe_module)
    config = _Config()
    probe.pytest_configure(config)

    status = 0
    saved_argv = sys.argv[:]
    sys.argv = target_argv
    try:
        runpy.run_path(target, run_name="__main__")
    except SystemExit as exit_request:
        code = exit_request.code
        if code is None:
            status = 0
        elif isinstance(code, int):
            status = code
        else:
            print(code, file=sys.stderr)
            status = 1
    finally:
        sys.argv = saved_argv
        # Always write the receipt, including on failure: "the route never ran" is exactly
        # the evidence a reviewer needs when a target fails, and losing it would make a red
        # run indistinguishable from an unprofiled one.
        probe.pytest_sessionfinish(_Session(config), status)
    return status


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
