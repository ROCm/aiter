# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""CPU-only checks of the RDNA attention tables.

The tables of csrc/kernels/rdna_attn are data: one entry per build of a kernel,
per GPU family. A launch test only sees the family of the GPU it runs on, so
the rules every table must keep are checked here, on the headers themselves and
for every family: an entry that breaks one would otherwise surface as a missing
or ambiguous build on hardware CI does not have.
"""

import itertools
import re
from pathlib import Path

import pytest

_CSRC = Path(__file__).resolve().parents[1] / "csrc" / "kernels" / "rdna_attn"
FAMILIES = ("Gfx1151", "Gfx1150", "Gfx1201")
HEAD_DIMS = (32, 48, 64, 72, 80, 96, 128, 256, 512)
# DecodeConfig, PrefillConfig and PrefillBatchConfig, in order.
DECODE = (
    "head_dim",
    "num_q_heads",
    "num_kv_heads",
    "window",
    "max_query_len",
    "dtype",
    "page_size",
    "max_segments",
    "row_groups",
    "min_segment_blocks",
    "waves",
    "head_dim_split",
    "prefetch",
    "v_in_lds",
    "dot_product",
)
PREFILL = (
    "head_dim",
    "num_q_heads",
    "num_kv_heads",
    "window",
    "dtype",
    "page_size",
    "min_query_len",
    "waves",
    "key_tile",
    "head_dim_split",
    "v_t",
    "double_buffer",
    "key_waves",
    "max_segments",
    "head_group",
    "causal",
    "row_tiles",
)
PREFILL_BATCH = (
    "head_dim",
    "num_q_heads",
    "num_kv_heads",
    "window",
    "page_size",
    "min_mean_m",
    "min_keys",
    "min_query_len",
    "head_group",
    "causal",
)
# Fields an entry may leave out, with their values then.
DEFAULTS = {"causal": 1, "row_tiles": 1}
# The cells' lower bounds (aiter/ops/rdna_attn.py).
BATCH_MEAN_M = (1, 16, 64, 256, 1024, 4096)
BATCH_KEYS = (0, 512, 2048, 8192)
MAX_M = 8

_ENTRY = re.compile(r"^\s*\{([\w:, ]+)\},$")
_ENUMS = {"Dtype::kFp16": 0, "Dtype::kBf16": 1}


def _table(kind: str, family: str, names: tuple[str, ...]) -> list[dict[str, int]]:
    """The entries of a family's table in rdna_<kind>_configs.h."""
    lines = (_CSRC / f"rdna_{kind}_configs.h").read_text().splitlines()
    start = 1 + next(
        i
        for i, line in enumerate(lines)
        if re.search(rf"\bk{family}\w*Configs\[\] = \{{$", line)
    )
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("};"))
    entries = []
    for line in lines[start:end]:
        m = _ENTRY.match(line)
        assert m, f"rdna_{kind}_configs.h: not one entry per line: {line!r}"
        values = [
            _ENUMS[t] if t in _ENUMS else int(t.removeprefix("HeadDim::k"))
            for t in map(str.strip, m[1].split(","))
        ]
        tail = [DEFAULTS[n] for n in names[len(values) :]]
        entries.append(dict(zip(names, values + tail, strict=True)))
    assert entries, f"{family} has no {kind} entries"
    return entries


def _shape(e: dict[str, int]) -> tuple[int, int, int, int]:
    return (e["num_q_heads"], e["num_kv_heads"], e["head_dim"], e["window"])


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("kind", ["decode", "decode_batch"])
def test_decode_tables_are_well_formed(kind, family):
    """No two decode builds share a shape, and every shape covers every M in
    both dtypes; a batch never runs the dot-product path."""
    entries = _table(kind, family, DECODE)
    keys = [
        (*_shape(e), e["page_size"], e["max_query_len"], e["dtype"]) for e in entries
    ]
    assert len(set(keys)) == len(keys), "two builds share a shape"
    for e in entries:
        assert e["head_dim"] in HEAD_DIMS
        assert e["num_q_heads"] % e["num_kv_heads"] == 0
        assert e["page_size"] % 16 == 0
        assert 1 <= e["max_query_len"] <= MAX_M
        assert not (kind == "decode_batch" and e["dot_product"]), e
    served = set(keys)
    for *shape, page in {k[:5] for k in keys}:
        for m, dtype in itertools.product(range(1, MAX_M + 1), (0, 1)):
            assert (*shape, page, m, dtype) in served, (kind, shape, page, m, dtype)


@pytest.mark.parametrize("family", FAMILIES)
def test_batched_decode_serves_the_shapes_of_decode(family):
    one = {(*_shape(e), e["page_size"]) for e in _table("decode", family, DECODE)}
    batch = {
        (*_shape(e), e["page_size"]) for e in _table("decode_batch", family, DECODE)
    }
    assert one == batch


@pytest.mark.parametrize("family", FAMILIES)
def test_prefill_tables_are_well_formed(family):
    """No two prefill builds share a shape and step, each shape has the same
    steps in both dtypes, and every decode shape has causal prefill builds."""
    entries = _table("prefill", family, PREFILL)
    keys = [
        (*_shape(e), e["page_size"], e["causal"], e["dtype"], e["min_query_len"])
        for e in entries
    ]
    assert len(set(keys)) == len(keys), "two builds share a step"
    for e in entries:
        assert e["head_dim"] in HEAD_DIMS
        assert e["num_q_heads"] % e["num_kv_heads"] == 0
        assert e["key_tile"] % 16 == 0
        assert e["waves"] % (e["head_dim_split"] * e["key_waves"]) == 0, e
        # Non-causal: no window, and only it may run two row tiles a wave.
        assert e["causal"] or e["window"] == 0, e
        assert e["row_tiles"] == 1 or not e["causal"], e
    steps: dict[tuple, set[int]] = {}
    for k in keys:
        steps.setdefault(k[:7], set()).add(k[7])
    for (*shape, dtype), own in steps.items():
        assert steps.get((*shape, 1 - dtype)) == own, (shape, "dtypes differ")
    causal = {k[:5] for k in keys if k[5]}
    for e in _table("decode", family, DECODE):
        if e["page_size"] == 16:
            assert (*_shape(e), 16) in causal, (_shape(e), "has no prefill build")


@pytest.mark.parametrize("family", FAMILIES)
def test_prefill_batch_tables_are_well_formed(family):
    """Every cell names a prefill build of its shape, in both dtypes, and a
    head group that divides its kv heads; each shape has the first cell."""
    builds = {
        (
            *_shape(e),
            e["page_size"],
            e["causal"],
            e["dtype"],
            e["min_query_len"],
        )
        for e in _table("prefill", family, PREFILL)
    }
    seen = set()
    for c in _table("prefill_batch", family, PREFILL_BATCH):
        config = (*_shape(c), c["page_size"], c["causal"])
        key = (*config, c["min_mean_m"], c["min_keys"])
        assert key not in seen, key
        seen.add(key)
        assert c["min_mean_m"] in BATCH_MEAN_M
        assert c["min_keys"] in BATCH_KEYS
        assert c["head_group"] == 0 or c["num_kv_heads"] % c["head_group"] == 0
        for dtype in (0, 1):
            assert (*config, dtype, c["min_query_len"]) in builds, key
    for config in {k[:6] for k in seen}:
        assert (*config, BATCH_MEAN_M[0], BATCH_KEYS[0]) in seen, config


@pytest.mark.parametrize(
    "kind, names",
    [
        ("decode", DECODE),
        ("decode_batch", DECODE),
        ("prefill", PREFILL),
        ("prefill_batch", PREFILL_BATCH),
    ],
)
def test_gfx1150_serves_the_causal_shapes_of_gfx1151(kind, names):
    """gfx1150's tables serve the shapes gfx1151's serve; the non-causal
    (encoder) builds may differ."""
    shapes = {
        family: {
            (*_shape(e), e["page_size"])
            for e in _table(kind, family, names)
            if e.get("causal", 1)
        }
        for family in ("Gfx1151", "Gfx1150")
    }
    assert shapes["Gfx1151"] == shapes["Gfx1150"]
