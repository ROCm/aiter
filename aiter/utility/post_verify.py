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


def _max_slack(
    actual: torch.Tensor,
    ref: torch.Tensor,
    atol: float,
    rtol: float,
    tol_floor: float = 1e-12,
) -> float:
    """Maximum per-element ``|a-b| / (atol + rtol*|b|)`` over the tensor.

    The denominator matches what ``torch.isclose(a, b, atol=atol, rtol=rtol)``
    uses as its element-wise tolerance, so the returned ratio is "by how
    many times does the worst element exceed the kernel's own
    isclose-tolerance":

    * value <= 1.0  -- every element is within isclose tolerance (the
      kernel would already have errRatio=0 with these tols).
    * value > 1.0   -- some element is over tolerance; how much is the
      catastrophic-ness we want to gate on.

    Anchoring to the local tolerance avoids the well-known pitfall of
    naive ``|a-b| / |b|``: for small ``|b|`` (e.g. quantized outputs that
    are legitimately near zero), the denominator collapses to ``eps``
    and *correct* kernels look catastrophic. Using ``atol + rtol*|b|``
    rejects this entire false-positive class while still flagging both
    (i) a kernel that writes a wrong large value at a tiny-reference
    position (the reviewer's wide-range example), and (ii) a kernel
    that writes 0 where the reference is large.
    """
    a = actual.float()
    b = ref.float()
    local_tol = (b.abs() * rtol).add_(atol).clamp_(min=tol_floor)
    return ((a - b).abs() / local_tol).max().item()


def _materialize_args(task, device):
    """Re-generate fresh input data and reconstruct the kernel/ref args.

    Mirrors the dict-keyed convention used by ``mp_tuner.work_group``:
    ``args[0]`` is a tuple of keys into the dict returned by
    ``gen_data(*gen_args, device=...)``; remaining positional args are
    passed through as-is. ``ref_args[0]`` follows the same convention.

    Returns ``(real_args, ref, arg_key_list)`` where ``arg_key_list`` is
    the ordered list of keys used to look up tensors in ``data`` (or
    ``None`` if the task does not use ``gen_data``). It is needed by
    callers that want to NaN-fill output tensors via ``output_keys``.
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
    real_args = list(data[k] for k in args[0]) + list(args[1:])
    arg_key_list = list(args[0])

    if ref is None and ref_func is not None:
        keys, *rest = ref_args
        ref = ref_func(*tuple(data[k] for k in keys), *rest, **ref_kwargs)

    return real_args, ref, arg_key_list


def _fill_nan_sentinel(real_args, arg_key_list, output_keys):
    """Fill output tensors with NaN so any unwritten region is detectable.

    Mirrors the NaN-sentinel logic in ``mp_tuner.worker``; needed in
    ``--fast_tune`` mode where the worker skips reference comparison and
    therefore never observes a NaN it would have rejected. Without this,
    a partial-write kernel produces non-NaN garbage from
    ``torch.empty`` and can sneak past the per-element relative-error
    check (small-magnitude garbage is, by definition, close to small
    reference values).
    """
    if not output_keys or arg_key_list is None:
        return
    for key in output_keys:
        if key not in arg_key_list:
            continue
        idx = arg_key_list.index(key)
        if idx < len(real_args) and isinstance(real_args[idx], torch.Tensor):
            real_args[idx].fill_(float("nan"))
    torch.cuda.synchronize()


def _run_and_compare(task, device) -> float:
    """Re-execute one task once on ``device`` and return max per-element slack.

    Two failure modes are detected:

    1. *Partial-write* kernels: output tensors are NaN-filled before the
       kernel runs (mirroring ``mp_tuner.worker``); any surviving NaN /
       Inf in the output is treated as an infinite slack.
    2. *Wide-range* / wrong-value errors: per-element
       ``|a-b| / (atol + rtol*|b|)`` is computed in fp32; the maximum
       value is returned. The (atol, rtol) used are the ones the tune
       task itself passed to ``checkAllclose`` (extracted from
       ``task[10]``/``task[11]``), so this metric is on the same scale
       as the tuner's tolerance: 1.0 means "exactly at the tune-time
       isclose threshold", 10.0 means "10x past it".

    Note: ``task[5]`` (``kwargs``) is the dict that ``mp_tuner.worker``
    forwards to ``run_perftest``. ``run_perftest`` peels off its own
    benchmarking keys (``num_warmup``, ``num_iters`` ...) and passes any
    remaining keys through to the kernel. We do the same split here so
    that kernels which legitimately receive extra kwargs continue to
    work.
    """
    real_args, ref, arg_key_list = _materialize_args(task, device)
    _func = task[3]
    raw_kwargs = task[5] or {}
    func_kwargs = {k: v for k, v in raw_kwargs.items() if k not in _PERFTEST_KWARGS}
    rtol = task[10] if len(task) > 10 and isinstance(task[10], (int, float)) else 1e-2
    atol = task[11] if len(task) > 11 and isinstance(task[11], (int, float)) else 1e-2
    output_keys = (
        task[14] if len(task) > 14 and isinstance(task[14], (list, tuple)) else None
    )

    _fill_nan_sentinel(real_args, arg_key_list, output_keys)

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

    for o, _ in tensor_pairs:
        if not torch.isfinite(o).all():
            return float("inf")

    return max(_max_slack(o, r, atol=atol, rtol=rtol) for o, r in tensor_pairs)


def verify_top1(
    tasks,
    result,
    *,
    slack_tol: float = 10.0,
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
        slack_tol: maximum allowed per-element
            ``|a-b| / (atol + rtol*|b|)`` (see ``_max_slack``). A value
            of ``1.0`` is exactly what ``torch.isclose`` already allows;
            anything above ``1.0`` is "by how many times past
            tune-time tolerance is the worst element". The default of
            ``10.0`` is a reasonable "10x past tolerance is suspicious"
            gate that catches both wide-range and large-value
            wrong-write failure modes without false-positives on
            kernels that already pass tune-time isclose.
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
                slack = _run_and_compare(task, device)
            except Exception as e:
                if verbose:
                    print(f"[post-verify] {info}: verify crashed ({e}); demoting")
                slack = float("inf")

            if slack <= slack_tol:
                if verbose:
                    suffix = (
                        f" (top-1 accepted, max_slack={slack:.3g})"
                        if attempt == 0
                        else (
                            f" (top-{attempt + 1} accepted after "
                            f"{attempt} demotion(s), max_slack={slack:.3g})"
                        )
                    )
                    print(f"[post-verify] {shape_key}:{suffix}")
                if attempt > 0:
                    n_late_accept += 1
                break

            n_demoted += 1
            us = result[i][1]
            result[i] = (info, us, 1.0)
            if verbose:
                print(
                    f"[post-verify] {shape_key}: top-{attempt + 1} demoted "
                    f"(max_slack={slack:.3g} > {slack_tol})"
                )

    print(
        f"[post-verify] verified {n_shapes} shape(s); "
        f"{n_demoted} candidate(s) demoted, {n_late_accept} shape(s) accepted via fallback"
    )
    return result
