# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""uBench data and JSON helpers, ported from gfx1250/microbench.

Source commit: e566d682a90108ca1babf9d8f5ba3ec13c32aa10
Only the dense floating-point data helpers needed by gfx950 are included.
"""

import json

import pandas as pd
import torch

DATA_DISTS = ("zero", "constant", "uniform", "norm")
_STAGE_ELEMS = 1 << 28


def print_json_table(name, rows, keep=None):
    """Print benchmark rows as one record-oriented JSON object.

    A single-line object is intentional: parent benchmark drivers can validate
    and forward it without parsing pandas' human-readable table formats.
    """
    if isinstance(rows, pd.DataFrame):
        df = rows.copy()
    else:
        df = pd.DataFrame([row for row in rows if row is not None])
    if not df.empty:
        df = df.replace("", pd.NA).dropna(axis=1, how="all")
        if keep is not None:
            cols = [column for column in keep if column in df.columns]
            cols += [
                column
                for column in df.columns
                if "err_msg" in column and column not in cols
            ]
            df = df[cols]
    records = json.loads(df.to_json(orient="records"))
    print(json.dumps({"name": name, "rows": records}), flush=True)


def make_generator(seed, device="cuda"):
    """Seeded ``torch.Generator`` -- same seed => bit-identical buffers."""
    return torch.Generator(device=device).manual_seed(int(seed))


def _row_chunks(rows, cols):
    """Row slices whose f32 staging stays around _STAGE_ELEMS elements."""
    step = max(_STAGE_ELEMS // max(cols, 1), 1)
    for start in range(0, rows, step):
        yield start, min(start + step, rows)


def _canon_dist(dist, allowed):
    if dist == "gaussian":
        dist = "norm"
    if dist not in allowed:
        raise ValueError(f"dist {dist!r}; choose from {allowed}")
    return dist


def _sample_data_f32(shape, dist, gen, *, lo, hi, device):
    if dist == "uniform":
        return torch.empty(shape, dtype=torch.float32, device=device).uniform_(
            lo, hi, generator=gen
        )
    if dist == "norm":
        return torch.empty(shape, dtype=torch.float32, device=device).normal_(
            0.0, 1.0, generator=gen
        )
    raise ValueError(f"data dist {dist!r} is not continuous; use fill dispatch")


def _fill_sampled(shape, dist, gen, *, dtype, device, uniform, constant, sample_fn):
    if dist == "zero":
        return torch.zeros(shape, dtype=dtype, device=device)
    if dist == "constant":
        return torch.full(shape, constant, dtype=dtype, device=device)
    lo, hi = uniform
    if len(shape) != 2:
        return sample_fn(shape, dist, gen, lo=lo, hi=hi, device=device).to(dtype)
    rows, cols = shape
    out = torch.empty(shape, dtype=dtype, device=device)
    for r0, r1 in _row_chunks(rows, cols):
        v = sample_fn((r1 - r0, cols), dist, gen, lo=lo, hi=hi, device=device)
        out[r0:r1] = v.to(dtype)
        del v
    return out


def fill(
    shape,
    dist,
    gen,
    *,
    dtype=torch.float32,
    device="cuda",
    uniform=(-1.0, 1.0),
    constant=1.0,
):
    """Return a ``dtype`` DATA tensor of ``shape``.

    ``dist`` in {zero, constant, uniform, norm}. ``uniform`` is U(lo, hi);
    ``norm`` / ``gaussian`` is N(0, 1). ``zero`` / ``constant`` ignore ``gen``.
    """
    dist = _canon_dist(dist, DATA_DISTS)
    return _fill_sampled(
        shape,
        dist,
        gen,
        dtype=dtype,
        device=device,
        uniform=uniform,
        constant=constant,
        sample_fn=_sample_data_f32,
    )
