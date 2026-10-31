# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`aiter.topk_select`: the dispatched per-row top-k, against its five backends.

Two halves. The sweep times `topk_select` beside every backend that can serve
the shape, so the dispatch can be read off the table rather than trusted; a row
where `topk_select` is far off the best column is a threshold that has moved.

The invariants are the other half, and they are the part a value comparison
cannot reach. Three bugs in this entry were silent -- an offset applied before
the gather, a row range handed to the backend that reads it as a whole-row
selection, and a `sorted_index` that reordered the indices without the values --
and all three returned plausible tensors. Each has an assertion here.

    python op_tests/test_topk_select.py
"""

import argparse
import ast
import itertools
import os
import subprocess
import sys
from contextlib import contextmanager
from unittest import mock

import pandas as pd
import torch

import aiter
import aiter.ops.flydsl.kernels.topk.topk_per_row_radix_stream as st
from aiter import dtypes
from aiter.jit.utils.chip_info import _LDS_CAPACITY_BYTES as LDS_CAPACITY
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.tensor_shim import wave_size_of
from aiter.ops.topk import _device_arch, _sampled_on_device, _sampled_supports_cached
from aiter.ops.topk_select import (
    _available,
    _choose,
    topk_select,
    topk_select_backend,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]


def run_torch(x, row_lens, k):
    """Reference: mask each row past its length, then torch's own selector."""
    masked = x.to(dtypes.fp32).clone()
    lens = row_lens.tolist()
    for r, n in enumerate(lens):
        masked[r, n:] = float("-inf")
    return torch.sort(
        torch.topk(masked, k, dim=1).values, dim=1, descending=True
    ).values


def _sorted_values(x, idx):
    """The selected scores, padding included, in descending order."""
    got = torch.where(
        idx >= 0,
        x.gather(1, idx.clamp_min(0).long()),
        torch.tensor(float("-inf"), device=x.device),
    )
    return torch.sort(got.to(dtypes.fp32), dim=1, descending=True).values


@benchmark()
def test_topk_select(m, n, k, tie, deterministic):
    x = torch.randn(m, n, dtype=dtypes.fp32)
    row_lens = torch.full((m,), n, dtype=dtypes.i32)
    ref = run_torch(x, row_lens, k)
    serving = _serving(m, n, k, wave_size_of(x.device.index), False, x.device.index)

    candidates = {
        "topk_select": lambda: topk_select(x, k, tie=tie, deterministic=deterministic)[
            1
        ],
    }
    # Every backend that can serve this shape, so the dispatch is visible rather
    # than asserted. Driven through the entry with the others withheld, which is
    # the same path the dispatcher takes.
    for backend in sorted(serving):
        candidates[backend] = lambda b=backend: _run_single_backend(x, row_lens, k, b)

    # One pass of the row plus the k written back.
    nbytes = (m * n + m * k) * x.element_size()
    ret = {"gfx": get_gfx(), "picked": topk_select_backend(m, n, k, serving)}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = checkAllclose(
            ref, _sorted_values(x, out), rtol=0, atol=0, msg=f"{name}: topk_select"
        )
    return ret


@benchmark()
def test_topk_argmax_half(m, n, dtype):
    """k=1 at a half format, against the two things a caller would do instead.

    `torch` is what a caller holding bf16 logits runs today. `cast+select` is
    what adopting this entry cost before the reduction took half formats, and it
    is here because it is the arm that says whether taking the dtype natively
    was worth doing -- not as an alternative anyone should pick.

    Exactness is against `torch.argmax` at the tensor's own dtype, with no
    tolerance: widening a half format to fp32 is lossless, so "close" would be
    the wrong bar and would hide a column picked one place over.
    """
    x = torch.randn(m, n, dtype=dtype)
    ref = x.argmax(dim=-1)
    nbytes = m * n * x.element_size() + m * 4

    ret = {"gfx": get_gfx(), "dtype": str(dtype)}
    arms = {
        "torch": lambda: x.argmax(dim=-1),
        "cast+select": lambda: topk_select(x.to(dtypes.fp32), 1, tie="low")[1],
        "select": lambda: topk_select(x, 1, tie="low")[1],
    }
    for name, fn in arms.items():
        out, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} match"] = bool(torch.equal(out.view(-1).long(), ref))
    ret["select vs torch"] = ret["torch us"] / ret["select us"]
    return ret


def test_zero_rows():
    """An empty batch selects nothing and leaves no launch error behind.

    Every backend puts rows on a grid dimension, so a zero-row call used to
    launch a zero-sized grid: HIP rejected it, and the next kernel on the
    stream raised. The probe after each call is that next kernel. Each shape is
    first run at one row, so the empty call takes the cached launch a server
    hits rather than the first-call compile.
    """
    i32 = torch.int32
    cases = [
        # (name, dtype, width, k, kwargs)
        *(
            (f"argmax {dt} n={n}", dt, n, 1, {"tie": "low"})
            for dt in (dtypes.fp32, dtypes.bf16, dtypes.fp16)
            # 2048 takes the single-pass launch, 129280 the split one.
            for n in (2048, 129280)
        ),
        ("small_k", dtypes.fp32, 2048, 16, {}),
        ("stream", dtypes.fp32, 32768, 512, {}),
        # The form a relaxed-acceptance rejection sampler asks for.
        (
            "decode sorted+values",
            dtypes.fp32,
            129280,
            10,
            {"sorted": True, "tie": "low", "return_value": True},
        ),
        (
            "output_idx+offset+end",
            dtypes.fp32,
            2048,
            16,
            {
                "output_idx": torch.empty(0, 16, dtype=i32),
                "output_idx_offset": torch.empty(0, dtype=i32),
                "end": torch.empty(0, dtype=i32),
                "sorted_index": True,
            },
        ),
    ]
    failures = []
    for name, dtype, n, k, kwargs in cases:
        try:
            warm = {kk: v for kk, v in kwargs.items() if not torch.is_tensor(v)}
            topk_select(torch.randn(1, n, dtype=dtype), k, **warm)
            torch.cuda.synchronize()
            vals, idx = topk_select(torch.empty(0, n, dtype=dtype), k, **kwargs)
            torch.ones(1).add_(1)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            failures.append(f"{name}: {e!r}"[:200])
            continue
        if idx.shape != (0, k):
            failures.append(f"{name}: idx shape {tuple(idx.shape)}")
        if "output_idx" in kwargs and idx.data_ptr() != kwargs["output_idx"].data_ptr():
            failures.append(f"{name}: output_idx not returned")
        if kwargs.get("return_value") and (vals is None or vals.shape != (0, k)):
            got = None if vals is None else tuple(vals.shape)
            failures.append(f"{name}: values {got}")
    return failures


def test_half_invariants(m, n):
    """What the half-format reduction promises beyond agreeing on random rows.

    Random rows never tie, and a tie is the whole reason `tie="low"` exists; the
    documented NaN rule and the signed zeros are likewise invisible without rows
    built to hold them.
    """
    failures = []

    def want(cond, label):
        if not cond:
            failures.append(label)

    for dtype in (dtypes.bf16, dtypes.fp16):
        tag = str(dtype).rsplit(".", 1)[-1]
        x = torch.randn(4, n, dtype=dtype)
        # Every element equal: column 0, or the tie-break is not the lowest.
        x[0] = 1.0
        # A maximum repeated at three columns: the first of them.
        x[1] = -1.0
        x[1, [7, 8, n // 2]] = 5.0
        # -0.0 and 0.0 are one score with two bit patterns and must not become
        # two keys; the earlier column wins whichever pattern it holds.
        x[2] = -3.0
        x[2, 5] = -0.0
        x[2, 9] = 0.0
        # NaN outranks +inf here, which `torch.argmax` does not promise -- so
        # this row is checked against the rule, and the three above against
        # torch as well.
        x[3] = 0.0
        x[3, 1] = float("inf")
        x[3, n - 2] = float("nan")

        got = topk_select(x, 1, tie="low")[1].view(-1).long()
        torch.cuda.synchronize()
        want(got.tolist() == [0, 7, 5, n - 2], f"{tag}: ties, zeros and NaN")
        want(
            torch.equal(got[:3], x[:3].contiguous().argmax(dim=-1)),
            f"{tag}: agrees with torch where torch promises an order",
        )

        # A row bound still bounds a half format, and an empty row answers -1.
        end = torch.tensor([n, 16, 1, 0], dtype=dtypes.i32)
        _, i = topk_select(x, 1, end=end, tie="low")
        torch.cuda.synchronize()
        want(int(i[3]) == -1, f"{tag}: an empty row yields -1")
        want(
            all(int(i[r]) < int(end[r]) for r in range(3)),
            f"{tag}: end= bounds every index",
        )

        # Past k=1 there is no half-format build, and saying so is the point:
        # falling through to "no backend serves" would read as a geometry
        # problem and send the caller looking in the wrong place.
        try:
            topk_select(x, 8)
        except NotImplementedError:
            pass
        else:
            failures.append(f"{tag}: k>1 was accepted without a build")

    del m
    for label in failures:
        aiter.logger.error("INVARIANT FAILED: %s", label)
    return failures


def _serving(rows, width, k, wave, ragged, device):
    """The backends that serve this shape on this device.

    `_available` plus the row-count half of sampled's predicate, which it cannot
    ask; off gfx950 that half declines every shape.
    """
    served = set(
        _available(
            width,
            k,
            wave,
            ragged,
            True,
            _sampled_on_device(device),
            _device_arch(device),
        )
    )
    if "sampled" in served and not _sampled_supports_cached(rows, width, k, device):
        served.discard("sampled")
    return frozenset(served)


class _NotServed(AssertionError):
    """The backend asked for does not serve the shape on this device."""


@contextmanager
def _only_backend(backend):
    """Hide every backend but `backend` from the entry's untied dispatch."""
    import aiter.ops.topk_select as ts

    keep = ts._BACKENDS_BY_TIE
    narrowed = dict(keep)
    narrowed[None] = (backend,)
    ts._BACKENDS_BY_TIE = narrowed
    # The dispatch is memoized on the call shape, which reads these tables. They
    # are constants everywhere but here, so withholding a backend means dropping
    # the answers taken while it was visible -- both on the way in and out.
    ts._choose.cache_clear()
    try:
        yield
    finally:
        ts._BACKENDS_BY_TIE = keep
        ts._choose.cache_clear()


def _run_single_backend(x, row_lens, k, backend, end=None, output_idx=None):
    """Call one backend through the entry by hiding the others from it.

    Asserts that the entry really used it. It did not, once: the dispatch tail
    could name a backend outside the set it was given, so hiding the others made
    `small_k` fall through to decode and this function timed decode under the
    small-k column -- 210.7us for a selector that runs in 40.2.
    """
    import aiter.ops.topk_select as ts

    with _only_backend(backend):
        rows, width = x.shape
        wave = wave_size_of(x.device.index)
        dev = x.device.index
        served = _serving(rows, width, k, wave, end is not None, dev) & {backend}
        if not served:
            raise _NotServed(
                f"{backend} does not serve {rows}x{width} k={k} on {get_gfx()}"
            )
        picked = ts.topk_select_backend(rows, width, k, served, device=dev)
        if picked != backend:
            raise AssertionError(f"asked for {backend}, the dispatch chose {picked}")
        return topk_select(x, k, end=end, output_idx=output_idx)[1]


# One shape per route, each forced onto the backend named.
_STRIDED_ROUTES = (
    ("argmax", 8, 4096, 1, dtypes.fp32),
    ("argmax", 8, 4096, 1, dtypes.bf16),
    ("small_k", 8, 4096, 16, dtypes.fp32),
    ("plain", 8, 4096, 2048, dtypes.fp32),  # register copy, 4 per lane
    ("plain", 64, 8192, 2048, dtypes.fp32),  # register copy, wide first digit
    ("plain", 8, 16384, 2048, dtypes.fp32),  # LDS tail with the sampled stage
    ("plain", 8, 32768, 2048, dtypes.fp32),  # LDS tail
    ("plain", 64, 32768, 512, dtypes.fp32),  # generic radix
    ("sampled", 64, 32768, 2048, dtypes.fp32),
    ("stream", 64, 32768, 512, dtypes.fp32),
    ("stream", 8, 2050, 2048, dtypes.fp32),  # small reject on gfx950
    ("decode", 4, 32768, 2048, dtypes.fp32),
)


def test_strided_layouts():
    """Row-strided `input`, `end` and `output_idx` on every route.

    The entry checks their shapes and dtypes and nothing about their strides.
    `plain` read a column slice of a wider tensor as dense rows, several
    backends wrote dense rows into a strided `output_idx` or read a strided
    `end` as dense, and all of them returned plausible tensors.
    """
    failures, not_served = [], set()
    for backend, m, n, k, dtype in _STRIDED_ROUTES:
        for layout in ("input", "input+end", "output", "end"):
            torch.manual_seed(0)
            x = torch.randn(m, n, dtype=dtype)
            if layout.startswith("input"):
                x = torch.zeros(m, n + 64, dtype=dtype)[:, :n].copy_(x)
            ragged = layout in ("input+end", "end")
            lens = (
                torch.randint(k, n + 1, (m,), dtype=dtypes.i32)
                if ragged
                else torch.full((m,), n, dtype=dtypes.i32)
            )
            end = None
            if ragged:
                end = (
                    lens
                    if layout == "input+end"
                    else torch.stack([lens, lens], 1)[:, 0]
                )
            guard, out = None, None
            if layout == "output":
                guard = torch.full((m, k + 32), -7, dtype=dtypes.i32)
                out = guard[:, :k]
            label = f"{backend} {m}x{n} k={k} {dtype} strided {layout}"
            try:
                idx = _run_single_backend(x, lens, k, backend, end=end, output_idx=out)
                torch.cuda.synchronize()
            except _NotServed:
                # A route this device does not have, e.g. stream at these widths
                # on gfx942; reported below rather than counted as a failure.
                not_served.add(f"{backend} {m}x{n} k={k} {layout}")
                continue
            except Exception as e:  # noqa: BLE001
                failures.append(f"{label}: {type(e).__name__}: {e}")
                continue
            if not torch.equal(_sorted_values(x, idx), run_torch(x, lens, k)):
                failures.append(f"{label}: wrong selection")
            if guard is not None and not (
                bool((guard[:, k:] == -7).all()) and idx.data_ptr() == out.data_ptr()
            ):
                failures.append(f"{label}: not written into output_idx in place")
    for label in failures:
        aiter.logger.error("STRIDED LAYOUT FAILED: %s", label)
    if not_served:
        aiter.logger.warning(
            "strided layouts: %d routes not served on %s, skipped: %s",
            len(not_served),
            get_gfx(),
            sorted(not_served),
        )
    return failures


def test_row_pitch():
    """Rows whose pitch is not the width, through the dispatch.

    A backend that reads rows in place must be routed on the pitch it is
    launched at. `sampled` was routed on the width and launched at `stride(0)`:
    a broadcast row (pitch 0) aborted the process, one row with a pitch of 1
    came back as one index and -1s, and overlapping rows were read wrong. Each
    shape here is wide enough for `sampled` on gfx950.
    """
    m, n, k = 64, 131072, 1024
    torch.manual_seed(0)
    flat = torch.randn(m * n // 2 + n)
    layouts = {
        "one row, pitch 1": torch.randn(n, 1).T,
        "broadcast rows, pitch 0": torch.randn(n).expand(m, n),
        "overlapping rows, pitch n/2": flat.as_strided((m, n), (n // 2, 1)),
        "column slice, pitch n+64": torch.randn(m, n + 64)[:, :n],
    }
    failures = []
    for layout, x in layouts.items():
        rows = x.shape[0]
        for ragged in (False, True):
            lens = (
                torch.randint(k, n + 1, (rows,), dtype=dtypes.i32)
                if ragged
                else torch.full((rows,), n, dtype=dtypes.i32)
            )
            label = f"{layout}{', ragged' if ragged else ''}"
            try:
                idx = topk_select(x, k, end=lens if ragged else None)[1]
                torch.cuda.synchronize()
            except Exception as e:  # noqa: BLE001
                failures.append(f"{label}: {type(e).__name__}: {e}")
                continue
            ordered = idx.sort(dim=1).values
            if not bool((ordered[:, 0] >= 0).all()) or bool(
                (ordered[:, 1:] == ordered[:, :-1]).any()
            ):
                failures.append(f"{label}: an index is -1 or repeated")
            elif not torch.equal(_sorted_values(x, idx), run_torch(x, lens, k)):
                failures.append(f"{label}: wrong selection")
    for label in failures:
        aiter.logger.error("ROW PITCH FAILED: %s", label)
    return failures


def test_pitch_routing():
    """On gfx950, rows at a pitch past their width route as their dense twin
    wherever plain reads them in place, and away from plain wherever it would
    copy them first; plain's copy measured slower there than the backends that
    read in place."""
    import aiter.ops.topk_select as ts

    failures = []
    if get_gfx() != "gfx950":
        return failures
    dev = torch.cuda.current_device()
    wave = wave_size_of(dev)
    for rows in (8, 64, 192, 256, 4096):
        for width in (3000, 8192, 32768, 81920, 98304, 131072, 200000):
            for k in (1024, 2048):
                route = [
                    ts._choose.__wrapped__(
                        rows, width, pitch, k, wave, False, None, False, True, dev, True
                    )
                    for pitch in (width, width + 64)
                ]
                in_place = ts._plain_reads_pitch(rows, width, k, dev)
                if in_place and route[1] != route[0]:
                    failures.append(
                        f"{rows}x{width} k={k}: {route[1]}, dense {route[0]}"
                    )
                if not in_place and route[1] == "plain":
                    failures.append(f"{rows}x{width} k={k}: plain copies the rows")
    for label in failures:
        aiter.logger.error("PITCH ROUTING FAILED: %s", label)
    return failures


def test_plain_pitch_in_place():
    """Rows at a pitch past their width that are routed to plain are read in
    place, not copied, wherever its uniform kernels or its ranged form read
    that pitch (`_plain_reads_pitch`, `_plain_reads_pitch_ranged`).

    Off gfx950 plain copied every such call first, doubling the input's memory.
    """
    import aiter.ops.topk_select as ts

    dev = torch.cuda.current_device()
    wave = wave_size_of(dev)
    failures, checked, copies = [], 0, []
    if _device_arch(dev) not in ts._PLAIN_ARCHES:
        return failures
    real = torch.Tensor.contiguous

    def spy(t, *args, **kwargs):
        out = real(t, *args, **kwargs)
        if out.data_ptr() != t.data_ptr():
            copies.append(tuple(t.shape))
        return out

    for rows in (8, 256, 4096):
        for width in (4097, 8192, 8193, 32768, 65536):
            for k in (1024, 2048):
                for pitch in (width + 64, 2 * width):
                    route = ts._choose.__wrapped__(
                        rows, width, pitch, k, wave, False, None, False, True, dev, True
                    )
                    if route != "plain" or not (
                        ts._plain_reads_pitch(rows, width, k, dev)
                        or ts._plain_reads_pitch_ranged(rows, width, pitch, k, dev)
                    ):
                        continue
                    checked += 1
                    torch.manual_seed(rows + width + k)
                    x = torch.randn(rows, pitch)[:, :width]
                    lens = torch.full((rows,), width, dtype=dtypes.i32)
                    label = f"{rows}x{width} k={k} pitch={pitch}"
                    copies.clear()
                    torch.Tensor.contiguous = spy
                    try:
                        idx = topk_select(x, k)[1]
                        torch.cuda.synchronize()
                    finally:
                        torch.Tensor.contiguous = real
                    if copies:
                        failures.append(f"{label}: copied the rows")
                    if not torch.equal(_sorted_values(x, idx), run_torch(x, lens, k)):
                        failures.append(f"{label}: wrong selection")
    if not checked and get_gfx() in ("gfx942", "gfx950"):
        failures.append("no shape reached plain at a pitch it reads in place")
    for label in failures:
        aiter.logger.error("PLAIN PITCH IN PLACE FAILED: %s", label)
    return failures


# One shape per FlyDSL backend, with the first call that puts a stride of 1 on
# the leading axis: a (1, n) row, or for k = 1 an (m, 1) column, strides (1, 1).
_UNIT_STRIDE_ROUTES = (
    ("decode", 64, 131072, 1024, "row"),
    ("stream", 64, 32768, 256, "row"),
    ("small_k", 64, 4096, 16, "row"),
    ("argmax", 64, 4096, 1, "row"),
    ("argmax", 64, 4096, 1, "column"),
)
# Where every route above is served; elsewhere one that is not is reported.
_UNIT_STRIDE_ARCHES = ("gfx942", "gfx950")


def test_unit_stride_first():
    """A first call whose stride-1 axis is not the last, then dense rows, on
    each FlyDSL backend.

    FlyDSL holds a tensor's first stride-1 axis static and the other strides
    dynamic, so a (1, n) row or an (m, 1) column with strides (1, 1) builds a
    kernel that reads rows at a pitch of 1. A launcher that reused that build
    for the dense rows after it returned 63 of 64 of them wrong. The odd call has
    to be each launcher's first, so this runs in a process of its own.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    script = (
        f"import sys; sys.path.insert(0, {here!r}); import test_topk_select as t; "
        "print('UNIT_STRIDE', repr(t._unit_stride_failures()))"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("UNIT_STRIDE ")]
    if done.returncode != 0 or not lines:
        tail = (done.stderr or done.stdout).strip().splitlines()[-3:]
        return [f"exit {done.returncode}: {' | '.join(tail)}"]
    failures, not_served = ast.literal_eval(lines[-1][len("UNIT_STRIDE ") :])
    for label in failures:
        aiter.logger.error("UNIT STRIDE FIRST FAILED: %s", label)
    if not_served:
        aiter.logger.warning(
            "unit stride first: %d routes not served on %s, skipped: %s",
            len(not_served),
            get_gfx(),
            not_served,
        )
    return failures


def _unit_stride_failures():
    failures, not_served = [], []
    for backend, m, n, k, first in _UNIT_STRIDE_ROUTES:
        torch.manual_seed(m * n + k)
        odd = torch.randn(n, 1).T if first == "row" else torch.randn(m, 1)
        for label, x in (
            (f"{tuple(odd.shape)} strides {odd.stride()}", odd),
            (f"{m}x{n} after it", torch.randn(m, n)),
        ):
            lens = torch.full((x.shape[0],), x.shape[1], dtype=dtypes.i32)
            try:
                idx = _run_single_backend(x, lens, k, backend)
            except _NotServed:
                if get_gfx() in _UNIT_STRIDE_ARCHES:
                    failures.append(f"{backend} {label}: not served")
                else:
                    not_served.append(f"{backend} {label}")
                break
            torch.cuda.synchronize()
            if not torch.equal(_sorted_values(x, idx), run_torch(x, lens, k)):
                failures.append(f"{backend} {label}: wrong selection")
    return failures, not_served


def test_rejections():
    """A wrong rank or dtype is refused with the ValueError that names the rule.

    The entry's `sorted` flag keeps DeepSelect's name, so the message has to
    reach the builtin some other way.
    """
    failures = []
    for shape, dtype in (
        ((4096,), dtypes.fp32),
        ((2, 8, 4096), dtypes.fp32),
        ((8, 4096), dtypes.i32),
    ):
        label = f"{shape} {dtype}"
        try:
            topk_select(torch.zeros(shape, dtype=dtype), 16)
        except ValueError as e:
            if "2-D" not in str(e):
                failures.append(f"{label}: unexpected message: {e}")
        except Exception as e:  # noqa: BLE001
            failures.append(f"{label}: {type(e).__name__}: {e}")
        else:
            failures.append(f"{label}: accepted")
    for label in failures:
        aiter.logger.error("REJECTION FAILED: %s", label)


def _serves(backend, rows, width, k, ragged):
    """Would the entry, with only `backend` visible, really dispatch to it?"""
    import aiter.ops.topk_select as ts

    dev = torch.cuda.current_device()
    served = _serving(rows, width, k, wave_size_of(dev), ragged, dev) & {backend}
    return (
        bool(served)
        and ts.topk_select_backend(rows, width, k, served, ragged=ragged, device=dev)
        == backend
    )


# One case per launch shape a backend has, as (backend, width, k).
_LAUNCH_SHAPES = (
    ("argmax", 2048, 1),  # one workgroup per row
    ("argmax", 129280, 1),  # split, then fold
    # Widths off the vector and off the bucket: the only ones with columns past
    # the row inside the last load, which is what an `end` past the row reaches.
    ("argmax", 2050, 1),
    ("small_k", 2048, 16),
    ("small_k", 3000, 16),
    ("plain", 32768, 1024),
    ("decode", 8192, 512),  # one workgroup, persistent
    ("decode", 8190, 512),  # the same, off the vector
    ("decode", 32768, 512),  # one block
    ("decode", 262144, 512),  # split into parts
    ("stream", 32768, 512),
    ("stream", 262144, 16),  # split, then merge
)
# More row counts than the 8-entry cache `end=None` used to read from.
_GRAPH_ROWS = tuple(range(1, 11))


def _capture_at_row_counts(n, k, capture):
    """One graph per row count in `_GRAPH_ROWS`, each over its own buffers.

    Every call is warmed on the capture stream first: the first one compiles,
    and the split decode allocates its stream-local counter there.
    """
    graphs = []
    for m in _GRAPH_ROWS:
        x = torch.randn(m, n, dtype=dtypes.fp32)
        out = torch.empty(m, k, dtype=dtypes.i32)
        capture.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture):
            topk_select(x, k, output_idx=out)
        torch.cuda.current_stream().wait_stream(capture)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture):
            topk_select(x, k, output_idx=out)
        graphs.append((graph, x, out))
    return graphs


def test_full_rows_graphs():
    """`end=None` reads no length tensor, so a graph at any row count holds.

    It used to read one from an 8-entry LRU keyed on the row count. Capturing a
    ninth count evicted -- and freed -- the tensor an earlier graph had baked
    in, and replaying that graph read whatever had reused the memory: wrong
    draft tokens, then a memory fault (ROCm/ATOM#2663). Each launch shape is
    captured at more counts than that cache held, and every graph is replayed on
    fresh scores only once all are captured.
    """
    failures = []
    capture = torch.cuda.Stream()
    for backend, n, k in _LAUNCH_SHAPES:
        tag = f"{backend} n={n} k={k}"
        missed = [m for m in _GRAPH_ROWS if not _serves(backend, m, n, k, False)]
        if missed:
            failures.append(f"{tag}: not dispatched to {backend} at m={missed}")
            continue
        with _only_backend(backend):
            graphs = _capture_at_row_counts(n, k, capture)
        for graph, x, out in graphs:
            x.copy_(torch.randn_like(x))
            out.fill_(-2)
            graph.replay()
            full = torch.full((x.shape[0],), n, dtype=dtypes.i32)
            if not torch.equal(_sorted_values(x, out), run_torch(x, full, k)):
                failures.append(f"{tag} m={x.shape[0]}: graph replay")
        torch.cuda.synchronize()

    for label in failures:
        aiter.logger.error("FULL-ROWS GRAPH FAILED: %s", label)
    return failures


# The backends that bound a row's length by the row width in the kernel.
_CLAMPS_END = frozenset({"argmax", "small_k", "decode", "stream"})


def test_ragged_end_per_backend():
    """An explicit `end` on every launch shape that takes ragged rows.

    The sweep passes no `end`, so it only reaches the full-rows builds; these
    are the builds that read a length per row. Where the kernel bounds a length
    by the width, an `end` past the row must select as the full row does rather
    than read the zeros past it: all-negative rows are where those would win.
    """
    failures = []
    for backend, n, k in _LAUNCH_SHAPES:
        with _only_backend(backend):
            for m in _GRAPH_ROWS:
                if not _serves(backend, m, n, k, True):
                    continue
                x = torch.randn(m, n, dtype=dtypes.fp32)
                end = torch.randint(k, n + 1, (m,), dtype=dtypes.i32)
                got = topk_select(x, k, end=end)[1]
                if not torch.equal(_sorted_values(x, got), run_torch(x, end, k)):
                    failures.append(f"{backend} n={n} k={k} m={m}: ragged end")
                if backend not in _CLAMPS_END:
                    continue
                x = -x.abs() - 1
                past = torch.full((m,), n + 64, dtype=dtypes.i32)
                got = topk_select(x, k, end=past)[1]
                full = torch.full((m,), n, dtype=dtypes.i32)
                # Checked before the gather, which would fault on such a column.
                if bool((got >= n).any()) or not torch.equal(
                    _sorted_values(x, got), run_torch(x, full, k)
                ):
                    failures.append(f"{backend} n={n} k={k} m={m}: end past row")
        torch.cuda.synchronize()

    for label in failures:
        aiter.logger.error("RAGGED END FAILED: %s", label)
    return failures


def test_lds_sizing():
    """The streaming selector's LDS sizing, on cards this box does not have.

    `_SUPPORTED_GFX` names gfx942 and the routing rule has no arch term, so the
    streaming selector is what every shape falls through to on a 64 KiB CU as
    well as a 160 KiB one. A build that claims more LDS than the card holds is
    chosen at dispatch and dies at launch -- loudly, but after `serves` has
    already told the router the geometry was fine.

    Arithmetic, so the card it is about does not have to be present. Three
    properties, and the first is what keeps the other two from being a rewrite
    of the code they check:

      gfx950 is unchanged     the budgets replaced two literals, and the
                              prefetch depths were tuned at those literals.
      servable implies fits   whatever `serves` accepts has a footprint inside
                              the card, on every arch the capacity table names.
      a plan is a preference  `_resolve_lds` returns the planned pair only when
                              it fits, and its own search otherwise. A table
                              fitted on one card must not become a launch
                              failure on another.
    """
    failures = []

    def want(cond, label):
        if not cond:
            failures.append(label)

    def budgets_for(cap):
        """The kernel's own derivation, not a copy of it."""
        return cap // 2 - st._LDS_RESIDENT_SLACK, cap - st._LDS_SOLO_SLACK

    want(
        budgets_for(LDS_CAPACITY["gfx950"]) == (76 * 1024, 148 * 1024),
        "gfx950 budgets reproduce the literals they replaced",
    )

    for arch, cap in LDS_CAPACITY.items():
        pair = budgets_for(cap)
        for k in (16, 256, 1024, 2048, 4096):
            for bt in (512, 1024):
                if 2048 % bt:
                    continue
                resolved = st._resolve_lds(k, bt, pair, st._VEC)
                served = st.topk_per_row_radix_stream_serves(k, 64, bt, pair) is None
                want(
                    served == (resolved is not None),
                    f"{arch} k={k} bt={bt}: serves agrees with _resolve_lds",
                )
                if resolved is None:
                    continue
                unroll, soft = resolved
                capacity = k + soft + unroll * bt * st._VEC
                # `topk_stream_config['lds_bytes']` leaves out `scan` and
                # `state`; the assembler does not.
                declared = (
                    (capacity + k) * 8
                    + st._NUM_BUCKETS * 4
                    + (bt // 64) * 4
                    + st._ST_SLOTS * 4
                )
                want(declared <= cap, f"{arch} k={k} bt={bt}: {declared}B fits {cap}B")

                # Both sides of the plan's width test, since above k=1024 it
                # declines on one of them and offers a pair on the other.
                for width in (4 * k, 64 * k):
                    plan = st.topk_per_row_radix_stream_lds_plan(4096, width, k)
                    got = st._resolve_lds(k, bt, pair, st._VEC, plan=plan)
                    if plan is None:
                        want(
                            got == resolved,
                            f"{arch} k={k} bt={bt} n={width}: declining is a no-op",
                        )
                        continue
                    p_capacity = k + plan[1] + plan[0] * bt * st._VEC
                    p_declared = (p_capacity + k) * 8 + st._NUM_BUCKETS * 4 + 4096
                    want(
                        (
                            tuple(got) == plan
                            if p_declared <= pair[1]
                            else tuple(got) == tuple(resolved)
                        ),
                        f"{arch} k={k} bt={bt} n={width}: "
                        f"the plan is taken only when it fits",
                    )

    for label in failures:
        aiter.logger.error("LDS SIZING FAILED: %s", label)
    return failures


def test_terminal_window():
    """`stream` drops its loop past the initial window (`terminal`) only for
    rows that fit it, and the build for a width decides that from the window it
    builds. This holds the window to the buffer, the decision to the edge, and
    checks the selection on both sides of it."""
    import aiter.ops.topk_select as ts

    failures = []
    if get_gfx() != "gfx950":
        return failures
    k, dev = 2048, torch.cuda.current_device()
    budgets = ts._lds_budgets(_device_arch(dev))
    for rows in (8, 64, 512):
        bt = st.topk_per_row_radix_stream_block_threads(rows, k)
        plan = st.topk_per_row_radix_stream_lds_plan(rows, 4 * k, k)
        cap = st.topk_per_row_radix_stream_window_cap(
            k, bt, budgets, lds_plan=plan, terminal=True
        )
        cfg = st.build_topk_per_row_radix_stream_module(
            k, 64, terminal=True, block_threads=bt, lds_budgets=budgets, lds_plan=plan
        ).topk_stream_config
        whole = cfg["capacity"] // cfg["vec"] * cfg["vec"]
        if not cap == cfg["window_cap"] == whole:
            failures.append(
                f"{rows} rows: the window is {cap}, the build holds {cfg['window_cap']}, "
                f"its buffer {cfg['capacity']} rounds to {whole}"
            )
        for width in (cap, cap + st._VEC):
            if st.topk_per_row_radix_stream_lds_plan(rows, width, k) != plan:
                continue  # another plan, another window; not this edge
            built = st.build_topk_per_row_radix_stream_module_for_width(
                k, 64, width, True, block_threads=bt, lds_budgets=budgets, lds_plan=plan
            ).topk_stream_config
            if built["terminal"] != (width <= cap):
                failures.append(
                    f"{rows}x{width} (cap {cap}): terminal={built['terminal']}"
                )
            torch.manual_seed(width)
            x = torch.randn(rows, width, dtype=dtypes.fp32)
            lens = torch.full((rows,), width, dtype=dtypes.i32)
            try:
                idx = _run_single_backend(x, lens, k, "stream")
            except _NotServed:
                # This test runs on gfx950 only, where stream serves these.
                failures.append(f"{rows}x{width} (cap {cap}): not served")
                continue
            if not torch.equal(_sorted_values(x, idx), run_torch(x, lens, k)):
                failures.append(f"{rows}x{width} (cap {cap}): wrong selection")
    for label in failures:
        aiter.logger.error("TERMINAL WINDOW FAILED: %s", label)
    return failures


# One shape per kernel family, each forced onto the backend named; `True` gives
# the rows an `end`.
_NAN_ROUTES = (
    ("argmax", 8, 4096, 1, False),
    ("small_k", 8, 4096, 16, False),
    ("stream", 64, 32768, 512, False),
    ("stream", 8, 2050, 2048, False),
    # gfx950 k=2048 rows a few columns past k, one per row band of
    # `_stream_small_reject`, and rows of exactly k.
    ("stream_reject", 8, 2050, 2048, False),
    ("stream_reject", 1024, 2055, 2048, False),
    ("stream_reject", 4096, 2056, 2048, False),
    ("whole_row", 1024, 2048, 2048, False),
    ("decode", 4, 32768, 2048, False),
    ("sampled", 64, 32768, 2048, False),
    ("sampled", 64, 32768, 2048, True),
    ("plain", 600, 6000, 2048, False),  # register copy, 512 threads
    ("plain", 64, 4500, 2048, False),  # register copy, 11+11+10 bits
    ("plain", 8, 4096, 2048, False),  # register copy, 4 per lane
    ("plain", 64, 8192, 2048, False),  # register copy, wide first digit
    ("plain", 8, 16384, 2048, False),  # LDS tail with the sampled stage
    ("plain", 8, 32768, 2048, False),  # LDS tail
    ("plain", 256, 100000, 2048, False),  # LDS tail, wide rows
    ("plain", 64, 65536, 1024, False),  # generic radix, compacted rows
    ("plain", 64, 8192, 1024, False),  # generic radix
    ("plain", 64, 32768, 2048, True),  # generic radix, ranged
    ("plain", 2, 262144, 2048, False),  # multi-block
    ("plain", 2, 262144, 2048, True),  # multi-block, ranged
)
# Quiet and signalling NaNs of both signs, a payload, and all-ones.
_NAN_BITS = (0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FC00123, 0xFFFFFFFF)


def _nan_rows(m, n, k, ends, seed):
    """Rows holding fewer NaNs than `k`, more than `k`, only negative ones, and
    none, among infinities and signed zeros; NaNs past a row's end too."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(m, n, dtype=dtypes.fp32, generator=g)
    bits = x.view(dtypes.i32)
    pattern = torch.tensor(
        [b - (1 << 32) if b >= 1 << 31 else b for b in _NAN_BITS], dtype=dtypes.i32
    )

    def cols(count, live):
        return torch.randperm(live, generator=g, device="cuda")[:count]

    for r in range(m):
        live = int(ends[r])
        kind = r % 4
        if kind == 0:
            count = max(1, k // 4)
        elif kind == 1:
            count = min(live, k + max(1, k // 2))
        elif kind == 2:
            count = max(1, k // 2)
            x[r, :live] = float("-inf")
        else:
            count = 0
        if count:
            at = cols(count, live)
            if kind == 2:
                bits[r, at] = pattern[1]
            else:
                bits[r, at] = pattern[torch.arange(count, device="cuda") % len(pattern)]
        others = cols(min(live, 8), live)
        x[r, others[:2]] = float("inf")
        x[r, others[2:4]] = float("-inf")
        x[r, others[4:6]] = -0.0
        x[r, others[6:8]] = 0.0
        if live < n:
            bits[r, live:] = pattern[0]
    return x


def _nan_high_key(x):
    """An int64 key whose order is the documented one: NaN of either sign
    above +inf, -0.0 equal to 0.0."""
    bits = x.view(dtypes.i32).long() & 0xFFFFFFFF
    mag = bits & 0x7FFFFFFF
    key = torch.where(bits >= 1 << 31, (1 << 31) - 1 - mag, bits + (1 << 31))
    key = torch.where(mag == 0, torch.full_like(key, 1 << 31), key)
    return torch.where(mag > 0x7F800000, torch.full_like(key, 1 << 33), key)


def test_nan_order():
    """Every backend ranks a NaN of either sign above +inf.

    Checked as the property rather than against torch: indices in range and
    distinct, and no unselected live element ranked above a selected one.
    """
    failures, not_served = [], []
    for backend, m, n, k, ragged in _NAN_ROUTES:
        tag = f"{backend} {m}x{n} k={k}{' end' if ragged else ''}"
        if ragged:
            g = torch.Generator(device="cuda").manual_seed(n + m)
            ends = torch.randint(
                n - n // 4, n + 1, (m,), dtype=dtypes.i32, generator=g, device="cuda"
            )
        else:
            ends = torch.full((m,), n, dtype=dtypes.i32)
        x = _nan_rows(m, n, k, ends, seed=m * n + k)
        try:
            idx = _run_single_backend(
                x, ends, k, backend, end=ends if ragged else None
            ).long()
        except _NotServed:
            # The routes are gfx950 shapes; elsewhere some are not served.
            if get_gfx() == "gfx950":
                failures.append(f"{tag}: not served")
            else:
                not_served.append(tag)
            continue
        torch.cuda.synchronize()
        live = torch.arange(n).view(1, n) < ends.view(m, 1)
        if not bool(((idx >= 0) & (idx < ends.view(m, 1).long())).all()):
            failures.append(f"{tag}: index outside the row")
            continue
        ordered = torch.sort(idx, dim=1).values
        if k > 1 and not bool((ordered[:, 1:] != ordered[:, :-1]).all()):
            failures.append(f"{tag}: an index repeats")
            continue
        key = _nan_high_key(x)
        picked = torch.zeros_like(live).scatter_(1, idx, True)
        floor = key.gather(1, idx).min(dim=1).values
        rest = torch.where(live & ~picked, key, torch.full_like(key, -1))
        bad = (rest.max(dim=1).values > floor).nonzero().flatten().tolist()
        if bad:
            failures.append(f"{tag}: rows {bad[:4]} rank a NaN or a value wrongly")
    for label in failures:
        aiter.logger.error("NAN ORDER FAILED: %s", label)
    if not_served:
        aiter.logger.warning(
            "nan order: %d routes not served on %s, skipped: %s",
            len(not_served),
            get_gfx(),
            not_served,
        )
    return failures


# One shape per route on gfx950: argmax, small_k, plain, decode, stream, sampled,
# whole rows and the small-reject stream.
_CROSS_DEVICE_SHAPES = (
    (8, 4096, 1),
    (8, 4096, 16),
    (600, 6000, 2048),
    (4, 32768, 2048),
    (64, 32768, 512),
    (64, 131072, 1024),
    (64, 2048, 2048),
    (64, 2050, 2048),
)


def test_cross_device():
    """The input on a GPU that is not the current one, in a process that has
    run the same shapes on the current one, which runs them again after.

    Every launch has to land on the input's GPU, under every output option,
    and the call has to leave the current device as it found it. A launcher
    used on both GPUs has to keep serving each.
    """
    if torch.cuda.device_count() < 2:
        return []
    failures = _cross_device_failures()
    for label in failures:
        aiter.logger.error("CROSS DEVICE FAILED: %s", label)
    return failures


def _on_current_failures(m, n, k, when):
    x = torch.randn(m, n, dtype=dtypes.fp32)
    lens = torch.full((m,), n, dtype=dtypes.i32)
    try:
        idx = topk_select(x, k)[1]
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        return [f"{m}x{n} k={k} on the current GPU {when}: {type(e).__name__}: {e}"]
    if not torch.equal(_sorted_values(x, idx), run_torch(x, lens, k)):
        return [f"{m}x{n} k={k} on the current GPU {when}: wrong selection"]
    return []


def _cross_device_failures():
    failures = []
    home = torch.cuda.current_device()
    other = (home + 1) % torch.cuda.device_count()
    on = torch.device("cuda", other)
    for m, n, k in _CROSS_DEVICE_SHAPES:
        torch.manual_seed(m * n + k)
        failures += _on_current_failures(m, n, k, "before")
        x = torch.randn(m, n, dtype=dtypes.fp32, device=on)
        off = torch.arange(m, dtype=dtypes.i32, device=on) * n
        for ragged in (False, True):
            end = (
                torch.randint(k, n + 1, (m,), dtype=dtypes.i32, device=on)
                if ragged
                else None
            )
            lens = end if ragged else torch.full((m,), n, dtype=dtypes.i32, device=on)
            ref = run_torch(x, lens, k)
            options = (
                {},
                {"return_value": True},
                {"sorted": True, "return_value": True},
                {"sorted_index": True, "return_value": True},
                {"output_idx_offset": off, "return_value": True},
                {"output_idx": torch.empty((m, k), dtype=dtypes.i32, device=on)},
            )
            for opts in options:
                label = f"{m}x{n} k={k}{' end' if ragged else ''} {sorted(opts)}"
                try:
                    v, i = topk_select(x, k, end=end, **opts)
                    torch.cuda.synchronize(on)
                except Exception as e:  # noqa: BLE001
                    failures.append(f"{label}: {type(e).__name__}: {e}")
                    continue
                if torch.cuda.current_device() != home:
                    failures.append(
                        f"{label}: left cuda:{torch.cuda.current_device()} current"
                    )
                    torch.cuda.set_device(home)
                if i.device != on or (v is not None and v.device != on):
                    failures.append(f"{label}: output on the wrong GPU")
                    continue
                if (
                    "output_idx" in opts
                    and i.data_ptr() != opts["output_idx"].data_ptr()
                ):
                    failures.append(f"{label}: not written to output_idx")
                local = i - off.view(m, 1) if "output_idx_offset" in opts else i
                if not torch.equal(_sorted_values(x, local), ref):
                    failures.append(f"{label}: wrong selection")
                elif v is not None and not torch.equal(v, x.gather(1, local.long())):
                    failures.append(f"{label}: values do not pair with idx")
                elif opts.get("sorted") and not bool((v[:, :-1] >= v[:, 1:]).all()):
                    failures.append(f"{label}: values not descending")
                elif opts.get("sorted_index") and not bool(
                    (i[:, :-1] <= i[:, 1:]).all()
                ):
                    failures.append(f"{label}: indices not ascending")
        failures += _on_current_failures(m, n, k, "after")
    return failures


class _NoFlyDSLState:
    """A launcher with none of the state a TopK launcher copies per GPU."""


# What a further GPU's launcher has to start from: FlyDSL 0.4.0's in-process
# caches, and the target it resolves with the signature on a first call.
_FRESH_PER_DEVICE = {
    "_mem_cache": {},
    "_call_state_cache": {},
    "_last_compiled": None,
    "cache_manager": None,
    "manager_key": None,
    "_sig": None,
    "_backend_target": None,
}


def test_launcher_per_device():
    """A launcher asked for on a further GPU is a copy that starts afresh, and
    one whose FlyDSL state cannot be reset raises rather than being shared.

    `test_cross_device` needs two GPUs and CI has one, so this asks for a device
    id nothing is launched on: the copy is made, never run.
    """
    from aiter.ops.flydsl.kernels.topk.topk_launch import _launcher_on
    from aiter.ops.flydsl.kernels.topk.topk_per_row_radix_stream import (
        build_topk_per_row_radix_stream_module,
    )

    failures = []
    dev = torch.cuda.current_device()
    other = dev + 4096
    # A build of its own, past the builder's cache, with a signature, a target
    # and an in-process cache entry, as one that has run would have.
    exe = build_topk_per_row_radix_stream_module.__wrapped__(16, wave_size_of(dev))
    exe._ensure_sig()
    exe._mem_cache["probe"] = object()
    if _launcher_on(exe, dev) is not exe:
        failures.append("the first GPU did not get the launcher itself")
    own = _launcher_on(exe, other)
    if own is exe:
        failures.append("a further GPU shared the first one's launcher")
    else:
        for name, fresh in _FRESH_PER_DEVICE.items():
            if getattr(own, name, "missing") != fresh:
                failures.append(f"the copy kept {name}")
        if own._mem_cache is exe._mem_cache:
            failures.append("the copy shares the in-process cache")
        if exe._sig is None or "probe" not in exe._mem_cache:
            failures.append("making the copy reset the original")
        if _launcher_on(exe, other) is not own:
            failures.append("a second ask on the same GPU made another copy")
    stub = _NoFlyDSLState()
    _launcher_on(stub, dev)
    try:
        _launcher_on(stub, other)
        failures.append("a launcher it cannot copy was shared with a further GPU")
    except RuntimeError:
        pass
    for label in failures:
        aiter.logger.error("LAUNCHER PER DEVICE FAILED: %s", label)
    return failures


def test_arch_decisions():
    """The arch-specific choices take the arch they are given, which the
    dispatch derives from the input's GPU: at k=2048 the streaming selector fits
    gfx950's LDS and not gfx942's, and the whole-row shortcut is gfx950's."""
    failures = []
    if _device_arch(0) != "gfx950":
        aiter.logger.info("arch decisions: device 0 is not gfx950, skipped")
        return failures
    n, k = 131072, 2048
    for arch, served in (("gfx950", True), ("gfx942", False)):
        if ("stream" in _available(n, k, 64, False, True, False, arch)) != served:
            failures.append(f"stream at {n}x{k} on {arch}: expected {served}")
        if ("whole_row" in _available(k, k, 64, False, True, False, arch)) != served:
            failures.append(f"whole_row at {k}x{k} on {arch}: expected {served}")
    got = _choose.__wrapped__(4096, k, k, k, 64, False, None, False, True, 0, True)
    if got != "whole_row":
        failures.append(f"4096x{k} k={k} on gfx950 chose {got}, not whole_row")
    for label in failures:
        aiter.logger.error("ARCH DECISIONS FAILED: %s", label)
    return failures


class _PlainAsked(Exception):
    pass


def test_plain_off_its_arches():
    """Off `_PLAIN_ARCHES` -- gfx1250, wave32, where plain's module does not
    build -- the dispatch neither picks plain nor asks its module anything:
    asking builds the module, and that build fails. Run as gfx1250 on whatever
    GPU this is, since the routing is host code."""
    import aiter.ops.topk as tk
    import aiter.ops.topk_plain as tp
    import aiter.ops.topk_select as ts

    failures, asked = [], []

    def refuse(name):
        def ask(*args, **kwargs):
            asked.append(name)
            raise _PlainAsked(name)

        return ask

    def clear():
        for mod in (ts, tp):
            for v in vars(mod).values():
                if hasattr(v, "cache_clear"):
                    v.cache_clear()

    names = (
        "_topk_plain",
        "topk_plain",
        "topk_plain_batches_ragged_rows",
        "topk_plain_reads_pitch",
        "topk_plain_use_mulblocks",
        "topk_plain_uses_radix",
        "topk_plain_width",
        "topk_plain_workspace_size",
    )
    patches = [
        mock.patch.object(mod, name, refuse(name))
        for mod in (tp, ts)
        for name in names
        if hasattr(mod, name)
    ]
    patches += [
        mock.patch.object(mod, "_device_arch", lambda device: "gfx1250")
        for mod in (tk, ts)
    ]
    dev = torch.cuda.current_device()
    clear()
    try:
        for p in patches:
            p.start()
        for rows in (1, 64, 4096):
            for width in (2048, 4096, 32768, 131072, 1048576):
                for k in (1, 16, 512, 1024, 2048):
                    for ragged in (False, True):
                        for tie, det in ((None, False), ("low", False), (None, True)):
                            label = f"{rows}x{width} k={k} ragged={ragged} tie={tie}"
                            try:
                                got = ts._choose.__wrapped__(
                                    rows,
                                    width,
                                    width,
                                    k,
                                    32,
                                    ragged,
                                    tie,
                                    det,
                                    True,
                                    dev,
                                    True,
                                )
                            except _PlainAsked as e:
                                failures.append(f"{label} det={det}: asked {e}")
                                continue
                            if got == "plain":
                                failures.append(f"{label} det={det}: chose plain")
    finally:
        for p in reversed(patches):
            p.stop()
        clear()
    for label in failures[:8]:
        aiter.logger.error("PLAIN OFF ITS ARCHES FAILED: %s", label)
    return failures


def test_invariants(m, n, k):
    """The properties a value comparison cannot see."""
    failures = []
    x = torch.randn(m, n, dtype=dtypes.fp32)

    def want(cond, label):
        if not cond:
            failures.append(label)

    # values[j] is the score at idx[j], under every ordering flag. The pairing
    # broke once when `sorted_index` reordered the indices on their own.
    # `return_value` is off by default, so every check that reads the values
    # asks for them; the ones that only read indices deliberately do not, which
    # is also what keeps the default path covered.
    for flags in ({}, {"sorted": True}, {"sorted_index": True}):
        v, i = topk_select(x, k, return_value=True, **flags)
        torch.cuda.synchronize()
        want(torch.equal(v, x.gather(1, i.long())), f"values pair with idx {flags}")
    want(topk_select(x, k)[0] is None, "return_value defaults off")
    want(
        torch.equal(
            topk_select(x, k, sorted=True, return_value=True)[0],
            torch.sort(
                topk_select(x, k, return_value=True)[0], dim=1, descending=True
            ).values,
        ),
        "sorted=True orders the values",
    )
    # `sorted` has to order the indices even when the values are dropped, which
    # it can only do by gathering them anyway. It silently did nothing once.
    #
    # Tested as the property rather than against the `return_value=True` call:
    # two calls return the same set but not the same slot order, and where two
    # selected scores are exactly equal -- 2 slots in 64 rows of gaussian noise,
    # often enough to matter -- an unstable `torch.sort` puts that pair either
    # way round. Descending values is what `sorted` promises; identical tensors
    # is not.
    by_value = x.gather(1, topk_select(x, k, sorted=True)[1].long())
    want(
        bool((by_value[:, :-1] >= by_value[:, 1:]).all()),
        "sorted=True orders the indices without return_value",
    )
    v, i = topk_select(x, k, sorted_index=True)
    want(bool((i[:, :-1] <= i[:, 1:]).all()), "sorted_index orders the indices")

    # A row range must bound the answer, not be read as "no range given".
    end = torch.randint(k, n, (m,), dtype=dtypes.i32)
    _, i = topk_select(x, k, end=end)
    torch.cuda.synchronize()
    want(bool((i < end.view(m, 1)).all()), "end= bounds every index")
    want(
        torch.equal(_sorted_values(x, i), run_torch(x, end, k)),
        "end= selects the right values",
    )

    # The offset renumbers the output; the values still come from this tensor.
    # Compared under `deterministic=True` and on sorted indices, because the
    # default leaves both the slot order and -- on `plain` -- the selected set
    # free to move between calls, and comparing two calls element-wise would be
    # testing that freedom rather than the offset.
    off = torch.arange(m, dtype=dtypes.i32) * n
    base_v, base_i = topk_select(
        x, k, deterministic=True, sorted_index=True, return_value=True
    )
    shift_v, shifted = topk_select(
        x,
        k,
        deterministic=True,
        sorted_index=True,
        output_idx_offset=off,
        return_value=True,
    )
    torch.cuda.synchronize()
    want(torch.equal(shifted, base_i + off.view(m, 1)), "offset shifts the indices")
    want(torch.equal(shift_v, base_v), "offset leaves the values alone")

    # `tie` and `deterministic` promise the selected *set* does not move, not the
    # order it comes back in: the streaming selector takes its output slots from
    # a shared counter, so a row with far more ties at the cut than places
    # permutes between calls while selecting the same columns every time. Sorted
    # here for that reason, and `sorted_index=True` below is the stronger promise
    # for callers who need the tensor itself to be reproducible.
    tied = x.clone()
    tied[:, : 4 * k] = 7.0
    for kw in ({"tie": "low"}, {"deterministic": True}):
        first = torch.sort(topk_select(tied, k, **kw)[1], dim=1).values
        again = torch.sort(topk_select(tied, k, **kw)[1], dim=1).values
        torch.cuda.synchronize()
        want(torch.equal(first, again), f"selects the same set under {kw}")
        pinned = topk_select(tied, k, sorted_index=True, **kw)[1]
        repeat = topk_select(tied, k, sorted_index=True, **kw)[1]
        torch.cuda.synchronize()
        want(torch.equal(pinned, repeat), f"sorted_index is reproducible under {kw}")

    for label in failures:
        aiter.logger.error("INVARIANT FAILED: %s", label)
    return failures


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("topk_select unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-m", "--rows", type=int, nargs="*", default=[1, 64, 1024], help="rows per call"
    )
    parser.add_argument(
        "-n",
        "--width",
        type=int,
        nargs="*",
        default=[2048, 32768, 262144],
        help="columns per row",
    )
    parser.add_argument(
        "-k",
        "--topk",
        type=int,
        nargs="*",
        default=[16, 512, 2048],
        help="elements selected per row",
    )
    parser.add_argument(
        "-t",
        "--tie",
        type=str,
        nargs="*",
        default=["none"],
        choices=["none", "low", "high"],
    )
    parser.add_argument(
        "--include-large",
        action="store_true",
        help="also run fp32 shapes of 4 GiB or more (m * n * 4 >= 2**32),\n"
        "which the sweep skips by default; each such shape is timed on\n"
        "every backend that serves it, like any other shape",
    )
    args = parser.parse_args()

    bad = test_lds_sizing()
    aiter.logger.info(
        "LDS sizing: %s", "all hold" if not bad else f"{len(bad)} FAILED: {bad}"
    )
    bad = test_terminal_window()
    aiter.logger.info(
        "terminal window: %s", "all hold" if not bad else f"{len(bad)} FAILED: {bad}"
    )
    assert not bad, f"terminal window: {len(bad)} failed"
    bad = test_invariants(64, 32768, 512)
    aiter.logger.info(
        "invariants: %s", "all hold" if not bad else f"{len(bad)} FAILED: {bad}"
    )
    bad = test_zero_rows()
    aiter.logger.info(
        "zero rows: %s", "all hold" if not bad else f"{len(bad)} FAILED: {bad}"
    )
    bad = test_half_invariants(64, 32768)
    aiter.logger.info(
        "half-format invariants: %s",
        "all hold" if not bad else f"{len(bad)} FAILED: {bad}",
    )
    bad = test_strided_layouts()
    aiter.logger.info(
        "strided layouts: %s", "all hold" if not bad else f"{len(bad)} FAILED: {bad}"
    )
    assert not bad, f"strided layouts: {len(bad)} failed"
    bad = test_row_pitch()
    aiter.logger.info(
        "row pitch: %s", "all hold" if not bad else f"{len(bad)} FAILED: {bad}"
    )
    assert not bad, f"row pitch: {len(bad)} failed"
    bad = test_pitch_routing()
    aiter.logger.info(
        "pitch routing: %s", "all hold" if not bad else f"{len(bad)} FAILED: {bad}"
    )
    assert not bad, f"pitch routing: {len(bad)} failed"
    bad = test_plain_pitch_in_place()
    aiter.logger.info(
        "plain pitch in place: %s",
        "all hold" if not bad else f"{len(bad)} FAILED: {bad}",
    )
    assert not bad, f"plain pitch in place: {len(bad)} failed"
    bad = test_unit_stride_first()
    aiter.logger.info(
        "unit stride first: %s",
        "all hold" if not bad else f"{len(bad)} FAILED: {bad}",
    )
    assert not bad, f"unit stride first: {len(bad)} failed"
    bad = test_rejections()
    aiter.logger.info(
        "rejections: %s", "all hold" if not bad else f"{len(bad)} FAILED: {bad}"
    )
    assert not bad, f"rejections: {len(bad)} failed"
    bad = test_nan_order()
    aiter.logger.info(
        "NaN order: %s", "all hold" if not bad else f"{len(bad)} FAILED: {bad}"
    )
    assert not bad, f"NaN order: {len(bad)} failed"
    bad = test_cross_device()
    aiter.logger.info(
        "cross device: %s", "all hold" if not bad else f"{len(bad)} FAILED: {bad}"
    )
    assert not bad, f"cross device: {len(bad)} failed"
    bad = test_launcher_per_device()
    aiter.logger.info(
        "launcher per device: %s",
        "all hold" if not bad else f"{len(bad)} FAILED: {bad}",
    )
    assert not bad, f"launcher per device: {len(bad)} failed"
    bad = test_arch_decisions()
    aiter.logger.info(
        "arch decisions: %s", "all hold" if not bad else f"{len(bad)} FAILED: {bad}"
    )
    assert not bad, f"arch decisions: {len(bad)} failed"
    bad = test_plain_off_its_arches()
    aiter.logger.info(
        "plain off its arches: %s",
        "all hold" if not bad else f"{len(bad)} FAILED: {bad[:4]}",
    )
    assert not bad, f"plain off its arches: {len(bad)} failed"
    bad = test_full_rows_graphs()
    aiter.logger.info(
        "full-rows graphs: %s", "all hold" if not bad else f"{len(bad)} FAILED: {bad}"
    )
    bad = test_ragged_end_per_backend()
    aiter.logger.info(
        "ragged end: %s", "all hold" if not bad else f"{len(bad)} FAILED: {bad}"
    )

    df = [
        test_topk_argmax_half(m, n, dtype)
        for dtype in (dtypes.bf16, dtypes.fp16)
        for m, n in itertools.product(args.rows, args.width)
    ]
    aiter.logger.info(
        "k=1 half-format summary:\n%s", pd.DataFrame(df).to_markdown(index=False)
    )

    for tie in args.tie:
        df = []
        for m, n, k in itertools.product(args.rows, args.width, args.topk):
            if k > n or (m * n * 4 >= 2**32 and not args.include_large):
                continue
            df.append(test_topk_select(m, n, k, None if tie == "none" else tie, False))
        df = pd.DataFrame(df)
        aiter.logger.info(
            "topk_select summary (tie=%s):\n%s", tie, df.to_markdown(index=False)
        )


if __name__ == "__main__":
    main()
