# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Serial post-tune verification.

Re-runs each shape's top-1 picked kernel on a single GPU and applies a
strict per-element relative-error check that the multi-process tune
hot path does not run (a global max-magnitude heuristic is used there
to keep the hot path free of extra GPU/CPU syncs).

On failure the candidate is demoted (its ``err_ratio`` is bumped to
``1.0``) so the caller's existing top-1 selection naturally falls
through to top-2/3. The number of fallback attempts per shape is
capped by ``max_fallback``.

This module deliberately lives outside ``mp_tuner`` so that the
multi-process tune path stays untouched. Verification runs serially on
a single GPU; its synchronizations therefore cannot perturb any other
worker's timing measurements.
"""

from collections import defaultdict

import torch

# kwargs reserved by run_perftest in test_common.py; they configure
# benchmarking and must not be forwarded to the kernel function.
_PERFTEST_KWARGS = frozenset(
    {"num_iters", "num_warmup", "testGraph", "num_rotate_args", "needTrace"}
)


def _max_rel_err(actual: torch.Tensor, ref: torch.Tensor, eps: float = 1e-6) -> float:
    """Maximum per-element |a-b|/|b| over the full tensor, in fp32."""
    a = actual.float()
    b = ref.float()
    return ((a - b).abs() / b.abs().clamp(min=eps)).max().item()


def _materialize_args(task, device):
    """Re-generate fresh input data and reconstruct the kernel/ref args.

    Mirrors the dict-keyed convention used by ``mp_tuner.work_group``:
    ``args[0]`` is a tuple of keys into the dict returned by
    ``gen_data(*gen_args, device=...)``; remaining positional args are
    passed through as-is. ``ref_args[0]`` follows the same convention.
    """
    (
        _info,
        gen_data,
        gen_args,
        _func,
        args,
        _kwargs,
        ref_func,
        ref_args,
        ref_kwargs,
        ref,
        *_rest,
    ) = task

    if gen_data is None:
        raise NotImplementedError(
            "post_verify currently requires gen_data; "
            "pre-materialized in_data path is not yet supported"
        )

    data = gen_data(*gen_args, device=device)
    real_args = tuple(data[k] for k in args[0]) + tuple(args[1:])

    if ref is None and ref_func is not None:
        keys, *rest = ref_args
        ref = ref_func(*tuple(data[k] for k in keys), *rest, **ref_kwargs)

    return real_args, ref


def _run_and_compare(task, device) -> float:
    """Re-execute one task once on ``device`` and return max per-element rel err.

    Note: ``task[5]`` (``kwargs``) is the dict that ``mp_tuner.worker``
    forwards to ``run_perftest``. ``run_perftest`` peels off its own
    benchmarking keys (``num_warmup``, ``num_iters`` ...) and passes any
    remaining keys through to the kernel. We do the same split here so
    that kernels which legitimately receive extra kwargs continue to
    work.
    """
    real_args, ref = _materialize_args(task, device)
    _func = task[3]
    raw_kwargs = task[5] or {}
    func_kwargs = {k: v for k, v in raw_kwargs.items() if k not in _PERFTEST_KWARGS}

    out = _func(*real_args, **func_kwargs)
    torch.cuda.synchronize()

    refs = ref if isinstance(ref, (list, tuple)) else [ref]
    outs = out if isinstance(out, (list, tuple)) else [out]
    tensor_pairs = [
        (o, r)
        for o, r in zip(outs, refs)
        if isinstance(o, torch.Tensor) and isinstance(r, torch.Tensor)
    ]
    if not tensor_pairs:
        return 0.0
    return max(_max_rel_err(o, r) for o, r in tensor_pairs)


def verify_top1(
    tasks,
    result,
    *,
    rel_tol: float = 10.0,
    max_fallback: int = 3,
    gpu_id: int = 0,
    verbose: bool = False,
):
    """Demote top-1 picks that fail strict per-element check.

    Args:
        tasks: full task list passed to ``mp_tuner``; each task's
            ``task[0]`` is the unique ``info`` tuple used as key.
        result: list of ``(info, us, err_ratio)`` returned by
            ``mp_tuner``; mutated in place.
        rel_tol: maximum allowed per-element relative error
            (``|a-b| / |b|``). The default of ``10.0`` is the same
            "10x off" threshold reviewers requested for wide-range
            outputs; smaller values mean stricter checks.
        max_fallback: max number of candidates per shape to verify
            (top-1, top-2, ..., top-max_fallback). Demoted entries
            have ``err_ratio`` set to ``1.0`` so the caller's sort
            naturally skips them.
        gpu_id: GPU index to run verification on (serial).
        verbose: print one line per demotion / late accept.

    Returns:
        The same ``result`` list, with demoted entries' ``err_ratio``
        bumped to ``1.0``.
    """
    task_by_info = {t[0]: t for t in tasks}

    by_shape: "dict[object, list[int]]" = defaultdict(list)
    for idx, (info, _us, _err) in enumerate(result):
        by_shape[info[0]].append(idx)

    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    n_shapes = len(by_shape)
    n_demoted = 0
    n_late_accept = 0

    for shape_key, idxs in by_shape.items():
        valid = sorted(
            (i for i in idxs if result[i][1] > 0 and result[i][2] < 1.0),
            key=lambda i: result[i][1],
        )
        for attempt, i in enumerate(valid[:max_fallback]):
            info = result[i][0]
            task = task_by_info.get(info)
            if task is None:
                if verbose:
                    print(
                        f"[post-verify] {shape_key}: no task for info={info}; skipping"
                    )
                break
            try:
                rel_err = _run_and_compare(task, device)
            except Exception as e:
                if verbose:
                    print(f"[post-verify] {info}: verify crashed ({e}); demoting")
                rel_err = float("inf")

            if rel_err <= rel_tol:
                if attempt > 0:
                    n_late_accept += 1
                    if verbose:
                        print(
                            f"[post-verify] {shape_key}: accepted top-{attempt + 1} "
                            f"after {attempt} demotion(s) (max_rel_err={rel_err:.3g})"
                        )
                break

            n_demoted += 1
            us = result[i][1]
            result[i] = (info, us, 1.0)
            if verbose:
                print(
                    f"[post-verify] {shape_key}: top-{attempt + 1} demoted "
                    f"(max_rel_err={rel_err:.3g} > {rel_tol})"
                )

    print(
        f"[post-verify] verified {n_shapes} shape(s); "
        f"{n_demoted} candidate(s) demoted, {n_late_accept} shape(s) accepted via fallback"
    )
    return result
