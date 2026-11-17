# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Differential test of aiter.moev2 against the legacy MoE code (steps doc 1.3).

CPU (`--check`): for every key of the legacy snapshot that a registered v2 path claims,
the v2 plan (in the snapshot record format) must equal the legacy golden record. There
is no allowlist: the revamp keeps today's behavior (design doc principle 8), and a bug
fix lands in the legacy code first, with the snapshot regenerated.

    python tests/moe_dispatch/differential.py --check [--arch gfx950]

GPU: `run_differential(path_name, call)` runs one call through AITER_MOEV2=off and
AITER_MOEV2=<path> and compares the outputs (see tests/test_moev2_differential.py).
"""

import argparse
import contextlib
import json
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent


def compare(arch):
    """Return (claimed {label: path name}, problems [str]) for one simulated arch."""
    from aiter.moev2 import paths
    from moe_dispatch import arch_sim, record, snapshot

    golden = snapshot.read(arch)
    checkers = [p for p in paths.registered().values() if p.dispatch_record is not None]
    claimed, problems = {}, []
    with arch_sim.simulated_arch(arch) as (gfx, cu_num):
        for key in snapshot.dispatch_keys(gfx, cu_num):
            label = key.label()
            if label in claimed:
                continue
            for path in checkers:
                try:
                    plan = path.dispatch_record(key.args, key.call_kwargs())
                except Exception as exc:  # noqa: BLE001
                    plan = record.error_record(exc)
                if plan is None:
                    continue
                if label in claimed:
                    problems.append(
                        f"{label}: claimed by {claimed[label]} and {path.name}"
                    )
                    continue
                claimed[label] = path.name
                got = plan if isinstance(plan, dict) else record.normalize(plan)
                expected = snapshot.resolve_same_as_base(golden, label)
                if got != expected:
                    problems.append(
                        f"{label} [{path.name}]\n"
                        f"    legacy: {json.dumps(expected, sort_keys=True)}\n"
                        f"    v2:     {json.dumps(got, sort_keys=True)}"
                    )
    return claimed, problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", required=True)
    parser.add_argument("--arch", action="append", help="simulated arch (default: all)")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(_HERE.parent))
    from moe_dispatch import snapshot

    snapshot._setup()
    from moe_dispatch import arch_sim

    failed = False
    for arch in args.arch or list(arch_sim.ARCHS):
        claimed, problems = compare(arch)
        by_path = {}
        for name in claimed.values():
            by_path[name] = by_path.get(name, 0) + 1
        summary = ", ".join(f"{k}={v}" for k, v in sorted(by_path.items())) or "none"
        print(
            f"{arch}: {len(claimed)} keys claimed by v2 ({summary}), {len(problems)} problems"
        )
        for line in problems[:40]:
            print("  " + line)
        failed |= bool(problems)
    return 1 if failed else 0


@contextlib.contextmanager
def moev2_mode(value):
    """Run with AITER_MOEV2=value, restoring the previous setting afterwards."""
    from aiter.moev2 import envs

    old = os.environ.get(envs.AITER_MOEV2)
    os.environ[envs.AITER_MOEV2] = value
    envs.reload()
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(envs.AITER_MOEV2, None)
        else:
            os.environ[envs.AITER_MOEV2] = old
        envs.reload()


def run_differential(path_name, call, *, bitwise=False, atol=1e-2, rtol=1e-2):
    """Run `call` (legacy fused_moe arguments by name) on legacy and on `path_name`.

    Pass `bitwise=True` when the v2 plan equals the legacy plan (same kernels and
    parameters); otherwise the outputs are compared with a tolerance.
    """
    import inspect

    import torch

    import aiter.fused_moe as fm
    from aiter.moev2 import paths

    bound = inspect.signature(fm.fused_moe).bind(**call)
    bound.apply_defaults()
    with moev2_mode("off"):
        ref = fm.fused_moe(**call)
    with moev2_mode(path_name):
        # The legacy hook passes every parameter, defaults included; check the same mapping.
        routed = paths.route_to_v2(bound.arguments)
        if routed is None or routed.name != path_name:
            raise AssertionError(f"path {path_name} does not claim this call")
        out = fm.fused_moe(**call)
    if bitwise:
        assert torch.equal(ref, out), f"{path_name}: output differs from legacy"
    else:
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)
    return ref, out


if __name__ == "__main__":
    sys.exit(main())
