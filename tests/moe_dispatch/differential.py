# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Differential test of aiter.moev2 against the legacy MoE code.

CPU (`--check`): for every key of the legacy snapshot that a registered v2 path takes,
the v2 plan (in the snapshot record format) must equal the legacy golden record, and the
log lines v2 dispatch emits must equal the ones `get_2stage_cfgs` emits for the key. There
is no allowlist: the revamp keeps today's behavior, and a bug
fix lands in the legacy code first, with the snapshot regenerated.

    python tests/moe_dispatch/differential.py --check [--arch gfx950]

GPU: `run_differential(path_name, call)` runs one call through AITER_MOEV2=off and
AITER_MOEV2=<path> and compares the kernel launches and outputs (see
tests/moe_dispatch/gpu_differential.py).

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import argparse
import contextlib
import dataclasses
import json
import os
import pathlib
import sys

_HERE = pathlib.Path(__file__).resolve().parent


@contextlib.contextmanager
def captured_logs():
    """Collect `(level, message)` of the aiter logger at INFO, printing nothing.

    Every record is kept except one: the AITER_MOEV2 selection line that
    `aiter.moev2.paths` logs once per selection (after start-up or a reload), e.g.
    "AITER_MOEV2='a8w8_pt': a8w8_pt on aiter.moev2, every other call on the legacy
    code ...". `paths` tags that record with the attribute `SELECTION_RECORD`.

    The differential compares the lines of a legacy run (AITER_MOEV2=off) with those of
    a v2 run (AITER_MOEV2=<path>). The selection line names the AITER_MOEV2 value, so it
    can never match between the two, and it says nothing about how a call was
    dispatched. Compared, it fails every case. Dispatch lines ("using 2stage ...",
    "run_1stage = ...", warnings) are still compared exactly. Lines logged under
    AITER_MOE_EXPLAIN=1 are not tagged and are kept; the harness does not set it.
    """
    import logging

    from aiter.moev2.paths import SELECTION_RECORD

    logger = logging.getLogger("aiter")
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            # The selection line, not a dispatch line; see the docstring.
            if not getattr(record, SELECTION_RECORD, False):
                records.append((record.levelno, record.getMessage()))

    handler = _Collect(logging.DEBUG)
    saved = logger.handlers[:], logger.level, logger.propagate
    logger.handlers[:] = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        yield records
    finally:
        logger.handlers[:], level, logger.propagate = saved[0], saved[1], saved[2]
        logger.setLevel(level)


def _legacy_logs(key):
    import aiter.fused_moe as fm

    with captured_logs() as logs:
        try:
            fm.get_2stage_cfgs.__wrapped__(*key.args, **key.call_kwargs())
        except Exception:  # noqa: BLE001, S110
            pass
    return logs


def compare(arch, check_logs=True):
    """Return (claimed {label: path name}, problems [str]) for one simulated arch."""
    from aiter.moev2 import paths
    from aiter.moev2.dispatch import tuned
    from moe_dispatch import arch_sim, record, snapshot

    golden = snapshot.read(arch)
    checkers = [p for p in paths.registered().values() if p.dispatch_record is not None]
    claimed, problems = {}, []
    with arch_sim.simulated_arch(arch) as (gfx, cu_num):
        keys = snapshot.dispatch_keys(gfx, cu_num)
        # Load the tuned table once up front so no key's logs include the load warnings.
        with captured_logs():
            if keys:
                _legacy_logs(keys[0])
            tuned._default_table()
        for key in keys:
            label = key.label()
            if label in claimed:
                continue
            for path in checkers:
                with captured_logs() as v2_logs:
                    try:
                        plan = path.dispatch_record(key.args, key.call_kwargs())
                    except Exception as exc:  # noqa: BLE001
                        plan = record.error_record(exc)
                if plan is None:
                    continue
                if check_logs:
                    legacy_logs = _legacy_logs(key)
                    if v2_logs != legacy_logs:
                        problems.append(
                            f"{label} [{path.name}] log lines differ\n"
                            f"    legacy: {legacy_logs}\n"
                            f"    v2:     {v2_logs}"
                        )
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
    parser.add_argument(
        "--no-logs", action="store_true", help="skip the log comparison"
    )
    args = parser.parse_args(argv)

    # `tests` so `moe_dispatch` imports, and the repository root so `aiter` is
    # this checkout rather than an installed copy, even one earlier on the path.
    front = [str(_HERE.parent), str(_HERE.parents[1])]
    sys.path[:] = front + [p for p in sys.path if p not in front]
    from moe_dispatch import snapshot

    snapshot._setup()
    from moe_dispatch import arch_sim

    failed = False
    for arch in args.arch or list(arch_sim.ARCHS):
        claimed, problems = compare(arch, check_logs=not args.no_logs)
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


def _bits(t):
    """`t` with floating-point values as integers of the same width: equal NaNs compare
    equal, and -0.0 differs from 0.0."""
    import torch

    if t.is_floating_point():
        width = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}
        return t.view(width[t.element_size()])
    return t.view(torch.uint8) if t.element_size() == 1 else t


def _expand(launch):
    """`{where: value}` of a recorded launch, dataclass arguments by field."""
    pairs = [(i, x) for i, x in enumerate(launch.args)]
    pairs += list(launch.keywords.items())
    expanded = []
    for where, x in pairs:
        if dataclasses.is_dataclass(x) and not isinstance(x, type):
            expanded.extend(
                (f"{where}.{field.name}", getattr(x, field.name))
                for field in dataclasses.fields(x)
            )
        else:
            expanded.append((where, x))
    return dict(expanded)


def _float_gap(x, y, noise, z=None):
    """Why floating-point `x` (legacy) and `y` (v2) are not close, or None.

    Close: where either value is non-finite, `y` has the bits of `x` or of `z` (legacy's
    second run, if any); elsewhere within 1 ulp of their own dtype at their magnitude (a
    rounding flip) or within twice `noise` (how far legacy's two runs differ on the same
    argument).
    """
    import torch

    xf, yf = x.float(), y.float()
    finite = xf.isfinite() & yf.isfinite()
    odd = ~finite
    if odd.any():
        match = _bits(x)[odd] == _bits(y)[odd]
        if z is not None:
            match |= _bits(z)[odd] == _bits(y)[odd]
        if not bool(match.all()):
            return "non-finite values differ"
    if not finite.any():
        return None
    gap = (xf - yf).abs()[finite]
    info = torch.finfo(x.dtype)
    peak = torch.maximum(xf.abs(), yf.abs())[finite].clamp_min(info.tiny)
    mantissa = {
        torch.float32: 23,
        torch.float16: 10,
        torch.bfloat16: 7,
        torch.float8_e5m2: 2,
    }.get(x.dtype, 3)
    ulp = torch.exp2(peak.log2().floor() - mantissa)
    allowed = torch.maximum(ulp, torch.full_like(ulp, 2 * noise))
    if bool((gap <= allowed).all()):
        return None
    worst = int((gap - allowed).argmax())
    return (
        f"differ beyond 1 ulp and twice legacy's own spread ({noise:.3g}): "
        f"{int((gap > allowed).sum())} of {gap.numel()} values, worst "
        f"{gap[worst].item():.3g} at magnitude {peak[worst].item():.3g}"
    )


def compare_launches(legacy, v2, call, out_legacy, out_v2, legacy2=None):
    """`{(stage, argument): problem}` between two `kernel_bench_callable` recordings.

    Same stages, same functions, equal non-tensor arguments, and tensors over what the
    kernels define: the sort buffers up to `num_valid_ids` (the rest is `torch.empty`),
    `[token, topk, inter]` buffers at the routed slots that are local (stage 1 skips the
    others), and nothing for the output buffer. The sort buffers and every integer
    tensor must be bitwise equal. Floating-point tensors (activations, intermediates and
    their scales) may differ by 1 ulp of their dtype, or by twice what the same
    argument differs between `legacy` and `legacy2`, a second legacy run: some stage-1
    kernels are not repeatable. A non-finite value must equal one of legacy's runs.
    """
    import torch

    if [n for n, _ in legacy] != [n for n, _ in v2]:
        return {
            ("", ""): f"stages differ: {[n for n, _ in legacy]} vs {[n for n, _ in v2]}"
        }
    if legacy2 is not None and [n for n, _ in legacy2] != [n for n, _ in legacy]:
        legacy2 = None
    roles = {}
    block_m = num_valid = None
    if "stage1" in dict(legacy):
        stage1 = dict(legacy)["stage1"]
        v2_stage1 = dict(v2)["stage1"]
        # Executor and legacy both call stage 1 as (a1, w1, w2, sorted_ids,
        # sorted_expert_ids, num_valid_ids, out, topk, block_m=..., ...).
        block_m = int(stage1.keywords["block_m"])
        num_valid = int(stage1.args[5][0])
        for a, b, role in zip(stage1.args[3:5], v2_stage1.args[3:5], ("ids", "blocks")):
            roles[a.data_ptr()], roles[b.data_ptr()] = role, role
        for _, launch in legacy + v2:
            if launch.keywords.get("sorted_weights") is not None:
                roles[launch.keywords["sorted_weights"].data_ptr()] = "ids"
    topk_ids, expert_mask = call["topk_ids"], call.get("expert_mask")
    slots = None if expert_mask is None else expert_mask[topk_ids.long()].bool()
    skip = {out_legacy.data_ptr(), out_v2.data_ptr()}

    def routed(t):
        """`t` at the local routed slots if it is a `[token, topk, ...]` buffer."""
        if t.dim() == 3 and t.shape[:2] == topk_ids.shape and slots is not None:
            return _bits(t)[slots].view(t.dtype)
        return t

    def spread(x, z):
        """How much legacy's two runs differ in one floating-point argument, over the
        values finite in both (`_float_gap` checks the others)."""
        xf, zf = x.float(), z.float()
        finite = xf.isfinite() & zf.isfinite()
        return (xf - zf).abs()[finite].max().item() if finite.any() else 0.0

    problems = {}
    seconds = [None] * len(legacy) if legacy2 is None else [l for _, l in legacy2]
    for (name, a), (_, b), second in zip(legacy, v2, seconds):
        second = {} if second is None else _expand(second)
        if a.func != b.func:
            problems[name, ""] = f"{name}: {a.func} vs {b.func}"
            continue
        if len(a.args) != len(b.args) or set(a.keywords) != set(b.keywords):
            problems[name, ""] = f"{name}: arguments differ: {a} vs {b}"
            continue
        pairs = [(i, x, y) for i, (x, y) in enumerate(zip(a.args, b.args))]
        pairs += [(k, a.keywords[k], b.keywords[k]) for k in a.keywords]
        expanded = []
        for where, x, y in pairs:
            if dataclasses.is_dataclass(x) and dataclasses.is_dataclass(y):
                if type(x) is not type(y):
                    problems[name, where] = (
                        f"{name}[{where}]: {type(x).__name__} vs {type(y).__name__}"
                    )
                    continue
                expanded.extend(
                    (
                        f"{where}.{field.name}",
                        getattr(x, field.name),
                        getattr(y, field.name),
                    )
                    for field in dataclasses.fields(x)
                )
            else:
                expanded.append((where, x, y))
        pairs = expanded
        for where, x, y in pairs:
            if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
                if x != y:
                    problems[name, where] = f"{name}[{where}]: {x!r} vs {y!r}"
                continue
            if x.data_ptr() in skip:
                continue
            if x.dtype != y.dtype or x.shape != y.shape:
                problems[name, where] = (
                    f"{name}[{where}]: {x.dtype}{tuple(x.shape)} vs {y.dtype}{tuple(y.shape)}"
                )
                continue
            role = roles.get(x.data_ptr())
            if role is None and x.is_floating_point():
                reason = None
                if not torch.equal(_bits(routed(x)), _bits(routed(y))):
                    z = second.get(where)
                    like_x = isinstance(z, torch.Tensor) and z.dtype == x.dtype
                    z = routed(z) if like_x and z.shape == x.shape else None
                    noise = 0.0 if z is None else spread(routed(x), z)
                    reason = _float_gap(routed(x), routed(y), noise, z)
                if reason is not None:
                    problems[name, where] = (
                        f"{name}[{where}]: {x.dtype}{tuple(x.shape)} {reason}"
                    )
                continue
            if role == "ids":
                x, y = x[:num_valid], y[:num_valid]
            elif role == "blocks":
                x, y = x[: num_valid // block_m], y[: num_valid // block_m]
            else:
                x, y = routed(x), routed(y)
            if not torch.equal(_bits(x), _bits(y)):
                problems[name, where] = (
                    f"{name}[{where}]: {x.dtype}{tuple(x.shape)} values differ"
                )
    return problems


def _ulp(t):
    import torch

    finite = t.float()[t.isfinite()]
    peak = finite.abs().max().item() if finite.numel() else 0.0
    if peak == 0:
        return 0.0
    mantissa = {torch.bfloat16: 7, torch.float16: 10}.get(t.dtype, 23)
    return 2.0 ** (int(torch.tensor(peak).log2().floor()) - mantissa)


def run_differential(path_name, call, *, bitwise=False):
    """Run `call` (legacy fused_moe arguments by name) on legacy and on `path_name`.

    The path must claim and run the call (not decline it), launch the same kernels with
    the same arguments (`compare_launches`, against both legacy runs: exact but for
    floating-point tensors), and produce the same output: bitwise with
    `bitwise=True`, else within the larger of 4 ulps of the output's peak and twice the
    difference between two legacy runs (atomic accumulation in GEMM2 is not
    deterministic).
    """
    import dataclasses
    import inspect

    import torch

    import aiter.fused_moe as fm
    from aiter.moev2 import api, paths

    path = paths.registered()[path_name]
    bound = inspect.signature(fm.fused_moe).bind(**call)
    bound.apply_defaults()
    # The legacy hooks pass every parameter, defaults included; check the same mapping.
    entry_call = (
        dict(bound.arguments) if path.entry == "api" else api.op_call(bound.arguments)
    )

    def launch(mode):
        bench = []
        fm.kernel_bench_callable = bench
        if call.get("output") is not None:
            # Legacy and v2 share the caller's buffer: what one leaves unwritten must
            # not pass as the other's result.
            call["output"].fill_(float("nan"))
        try:
            with moev2_mode(mode):
                out = fm.fused_moe(**call)
        finally:
            fm.kernel_bench_callable = None
        torch.cuda.synchronize()
        if call.get("output") is not None and out is not call["output"]:
            raise AssertionError(f"{mode}: the result is not the caller's output=")
        return out, bench

    ref_raw, legacy_launches = launch("off")
    ref = ref_raw.clone()
    ref2_raw, legacy2_launches = launch("off")
    ref2 = ref2_raw.clone()
    results = []

    def run(c):
        out = path.run(c)
        results.append(out)
        return out

    paths._REGISTRY[path_name] = dataclasses.replace(path, run=run)
    try:
        with moev2_mode(path_name):
            routed = paths.route_to_v2(entry_call, path.entry)
            if routed is None or routed.name != path_name:
                raise AssertionError(f"path {path_name} does not claim this call")
        out_raw, v2_launches = launch(path_name)
        out = out_raw.clone()
    finally:
        paths._REGISTRY[path_name] = path
        paths.reload()
    if len(results) != 1 or results[0] is paths.DECLINED:
        raise AssertionError(f"path {path_name} did not run this call: {results}")
    problems = list(
        compare_launches(
            legacy_launches, v2_launches, call, ref_raw, out_raw, legacy2_launches
        ).values()
    )
    if problems:
        raise AssertionError(
            f"{path_name}: launches differ: " + "; ".join(problems[:6])
        )
    if bitwise:
        assert torch.equal(ref, out), f"{path_name}: output differs from legacy"
    else:
        finite = ref.isfinite()
        assert torch.equal(finite, out.isfinite()) and torch.equal(
            ref[~finite], out[~finite]
        ), f"{path_name}: non-finite outputs differ from legacy"

        def gap(x):
            return (x.float()[finite] - ref.float()[finite]).abs().max().item()

        noise, err = gap(ref2), gap(out)
        tol = max(4 * _ulp(ref), 2 * noise)
        assert (
            err <= tol
        ), f"{path_name}: max |v2 - legacy| {err} > {tol} (legacy noise {noise})"
    return ref, out


if __name__ == "__main__":
    sys.exit(main())
