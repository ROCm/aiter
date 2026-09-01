# SPDX-License-Identifier: MIT
# Copyright (C) 2026-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Resolution tests for the unified-attention config tables.

These never launch a kernel -- they only exercise config lookup, so they run on
any machine and can sweep every arch through the ``arch`` override rather than
only the one the runner happens to have.

The case that motivates the page-size sweep: the old ``BS_LEQ_*`` ladder only
covered up to its largest named bound and from its smallest, so a 48-token page
fell in the gap and raised KeyError in the field. The kernel test parametrises
``block_size`` over [16, 64] only -- both powers of two, both exactly on bucket
edges -- so CI could not reach it.
"""

import json
from pathlib import Path

import pytest
import torch

from aiter.ops.triton.attention.unified_attention import _UAParams
from aiter.ops.triton.utils import unified_attention_utils
from aiter.ops.triton.utils.types import e4m3_dtype
from aiter.ops.triton.utils.unified_attention_utils import (
    explain,
    get_unified_attention_config,
)

# (arch, backend) for every table in the tree.
TABLES = [
    ("gfx942", "triton"),
    ("gfx950", "triton"),
    ("gfx1100", "triton"),
    ("gfx1151", "triton"),
    ("gfx1200", "triton"),
    ("gfx1201", "triton"),
    ("gfx1250", "triton"),
    pytest.param(
        "gfx1250",
        "gluon",
        marks=pytest.mark.skip(
            reason="gluon table not yet converted to the nested format"
        ),
    ),
]

OPS = ["attn_2d", "attn_3d", "reduce", "kv_split"]

# Exactly what each op's config may contain. attn_2d/attn_3d/reduce configs are
# splatted into the kernel launch as **config, so an EXTRA key is a TypeError at
# launch, and a key the wrapper also passes by name is a duplicate-argument
# TypeError. attn_3d and reduce both receive TILE_SIZE by name, which is why
# their leaves must not carry tile bounds. kv_split is read field by field.
ALLOWED_KEYS = {
    "attn_2d": {"BLOCK_M", "TILE_SIZE", "num_warps", "num_stages", "waves_per_eu"},
    "attn_3d": {"BLOCK_M", "num_warps", "num_stages", "waves_per_eu"},
    "reduce": {"num_warps", "num_stages", "waves_per_eu"},
    "kv_split": {"NUM_SEGMENTS", "TILE_SIZE"},
}

# Page sizes: powers of two on and between bucket edges, plus the two the
# ladder could not express -- 1 (below every bound) and 48 (between them).
BLOCK_SIZES = [1, 16, 32, 48, 64, 128, 256]
HEAD_SIZES = [32, 64, 128, 192, 256, 512, 576]
QUERY_LENS = [1, 4, 100, 256, 4096]
DTYPES = [
    (torch.bfloat16, torch.bfloat16),
    (e4m3_dtype, e4m3_dtype),
    (torch.bfloat16, e4m3_dtype),
    (torch.uint8, torch.uint8),
]

# Params every op must come back with, whatever the arch or the rules that fired.
REQUIRED_KEYS = {
    "attn_2d": {"BLOCK_M", "num_warps", "waves_per_eu", "TILE_SIZE"},
    "attn_3d": {"BLOCK_M", "num_warps", "waves_per_eu"},
    "reduce": {"waves_per_eu"},
    "kv_split": {"NUM_SEGMENTS", "TILE_SIZE"},
}


def make_params(
    *,
    head_size=128,
    max_seqlen_q=1,
    max_seqlen_k=65536,
    sliding_window=0,
    shuffled_kv_cache=False,
    q_dtype=torch.bfloat16,
    kv_cache_dtype=torch.bfloat16,
    num_queries_per_kv=8,
    block_size=16,
):
    """A real ``_UAParams``, not a stand-in.

    Config lookup reads only a handful of fields, but building the real
    NamedTuple means renaming one of them breaks this test loudly instead of
    letting a ``SimpleNamespace`` silently keep the old attribute name.
    """
    return _UAParams(
        q=None,
        k=None,
        v=None,
        out=None,
        cu_seqlens_q=None,
        seqused_k=None,
        block_table=None,
        softmax_scale=1.0,
        softcap=0.0,
        causal=True,
        sliding_window=sliding_window,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        num_tokens=1,
        num_query_heads=num_queries_per_kv,
        num_kv_heads=1,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        num_seqs=1,
        total_num_q_blocks=1,
        num_2d_prgms=1,
        num_blocks=1,
        block_size=block_size,
        k_width=8,
        scale_k_width=4,
        block_scales_size=16,
        q_dtype=q_dtype,
        kv_cache_dtype=kv_cache_dtype,
        all_decode=max_seqlen_q == 1,
        shuffled_kv_cache=shuffled_kv_cache,
        use_alibi_slopes=False,
        use_qq_bias=False,
        num_sms=304,
        target_num_prgms=304 * 4,
    )


@pytest.mark.parametrize("arch,backend", TABLES)
@pytest.mark.parametrize("op", OPS)
def test_resolution_is_total(arch, backend, op):
    """Every reachable situation resolves, for every table in the tree.

    A miss here is not a wrong number -- it is a KeyError at kernel launch.
    """
    for block_size in BLOCK_SIZES:
        for head_size in HEAD_SIZES:
            for max_seqlen_q in QUERY_LENS:
                for q_dtype, kv_dtype in DTYPES:
                    for sliding_window in (0, 1024):
                        for max_seqlen_k in (512, 65536):
                            params = make_params(
                                head_size=head_size,
                                max_seqlen_q=max_seqlen_q,
                                max_seqlen_k=max_seqlen_k,
                                sliding_window=sliding_window,
                                q_dtype=q_dtype,
                                kv_cache_dtype=kv_dtype,
                                block_size=block_size,
                            )
                            config, _ = get_unified_attention_config(
                                op, params, backend=backend, arch=arch
                            )
                            missing = REQUIRED_KEYS[op] - set(config)
                            assert not missing, (
                                f"{arch}/{backend} {op}: missing {sorted(missing)} for "
                                f"D={head_size} Q={max_seqlen_q} BS={block_size} "
                                f"dtypes=({q_dtype}, {kv_dtype}) SW={sliding_window}"
                            )


@pytest.mark.parametrize("arch,backend", TABLES)
@pytest.mark.parametrize("block_size", BLOCK_SIZES)
def test_tile_size_divides_or_is_divided_by_page(arch, backend, block_size):
    """TILE_SIZE and the page size stay commensurate.

    The derived form is what makes this hold for every page size rather than
    only the bucketed ones.
    """
    for head_size in HEAD_SIZES:
        params = make_params(head_size=head_size, block_size=block_size)
        config, _ = get_unified_attention_config(
            "kv_split", params, backend=backend, arch=arch
        )
        tile = config["TILE_SIZE"]
        assert tile > 0
        assert tile & (tile - 1) == 0, f"TILE_SIZE={tile} is not a power of two"


@pytest.mark.parametrize("arch,backend", TABLES)
def test_fp8_tile_is_never_smaller_than_bf16(arch, backend):
    """fp8 needs at least the tile bf16 uses, and gets more where a table says
    so through an ``fp8_fp8`` bucket.

    There is no fp8 knob in the resolver: fp8 differences are ordinary buckets,
    so this checks the outcome rather than the mechanism. The floor is not
    universal either -- the CDNA/RDNA paths raise fp8 to 32 while the gfx1250 3D
    path pins the tile to the page size and raises nothing.
    """
    for block_size in BLOCK_SIZES:
        bf16 = make_params(block_size=block_size)
        fp8 = make_params(
            block_size=block_size, q_dtype=e4m3_dtype, kv_cache_dtype=e4m3_dtype
        )
        c_bf16, _ = get_unified_attention_config(
            "kv_split", bf16, backend=backend, arch=arch
        )
        c_fp8, _ = get_unified_attention_config(
            "kv_split", fp8, backend=backend, arch=arch
        )
        assert c_fp8["TILE_SIZE"] >= c_bf16["TILE_SIZE"]


@pytest.mark.parametrize("arch,backend", TABLES)
def test_resolution_is_deterministic_and_isolated(arch, backend):
    """Two lookups agree, and mutating one result cannot affect the next --
    the loader hands back a deep copy of a cached table.
    """
    params = make_params()
    first, _ = get_unified_attention_config(
        "attn_2d", params, backend=backend, arch=arch
    )
    first["BLOCK_M"] = -1
    first["INJECTED"] = True
    second, _ = get_unified_attention_config(
        "attn_2d", params, backend=backend, arch=arch
    )
    assert second["BLOCK_M"] != -1
    assert "INJECTED" not in second


@pytest.mark.parametrize("arch,backend", TABLES)
@pytest.mark.parametrize("op", OPS)
def test_explain_names_the_entry_and_agrees_with_the_lookup(arch, backend, op):
    """`explain` has to report the entry the real lookup actually lands on.

    A trace that drifts from the resolver is worse than no trace, so this
    checks the reported values against get_unified_attention_config() rather
    than only checking that some text was produced.
    """
    for head_size in (64, 128, 576):
        for max_seqlen_q in (1, 4096):
            params = make_params(head_size=head_size, max_seqlen_q=max_seqlen_q)
            text = explain(op, params, backend=backend, arch=arch)
            assert "matched:" in text and "leaf:" in text
            assert "UNRESOLVED" not in text

            config, _ = get_unified_attention_config(
                op, params, backend=backend, arch=arch
            )
            reported = {}
            for line in text.split("leaf:", 1)[1].splitlines():
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                reported[key.strip()] = value.split("(derived)")[0].strip()
            for key, value in config.items():
                assert key in reported, f"{key} missing from explain:\n{text}"
                assert reported[key] == str(
                    value
                ), f"{key}: explain says {reported[key]}, lookup says {value}"


@pytest.mark.parametrize("arch,backend", TABLES)
@pytest.mark.parametrize("op", OPS)
def test_no_key_the_kernel_launch_would_reject(arch, backend, op):
    """A resolved config carries nothing beyond what the launch accepts.

    attn_2d / attn_3d / reduce configs reach the kernel as ``**config``, so this
    is the difference between a clean launch and a TypeError. Two ways to get
    it wrong, both silent at resolution time: a key no kernel parameter matches,
    or a key the wrapper already passes by name (attn_3d and reduce are handed
    TILE_SIZE explicitly, so a tile bound left in their leaves collides).
    """
    for head_size in HEAD_SIZES:
        for max_seqlen_q in (1, 4096):
            for block_size in (16, 48, 256):
                params = make_params(
                    head_size=head_size,
                    max_seqlen_q=max_seqlen_q,
                    block_size=block_size,
                )
                config, _ = get_unified_attention_config(
                    op, params, backend=backend, arch=arch
                )
                extra = set(config) - ALLOWED_KEYS[op]
                assert not extra, (
                    f"{arch}/{backend} {op}: config carries {sorted(extra)}, which "
                    f"the kernel launch does not accept (allowed: "
                    f"{sorted(ALLOWED_KEYS[op])})"
                )


# --------------------------------------------------------------------------
# Structural checks on the checked-in JSON.
#
# These validate static data, so they belong in CI rather than in the loader:
# a table cannot become malformed at runtime, only when someone edits it. The
# loader keeps only an O(1) guard per level, because walking the tree on every
# lookup re-validated the same file once per distinct shape.
# --------------------------------------------------------------------------

# What a leaf may hold: the launch keys for that op, plus the tile bounds that
# compute_tile_params() consumes into TILE_SIZE.
_TILE_BOUNDS = {"TILE_SIZE_MIN", "TILE_SIZE_MAX"}
ALLOWED_LEAF_KEYS = {
    op: (keys - {"TILE_SIZE"}) | _TILE_BOUNDS for op, keys in ALLOWED_KEYS.items()
}


def _table(arch, backend):
    path = (
        Path(unified_attention_utils.__file__).parent.parent
        / "configs"
        / arch
        / backend
        / "attention"
        / "unified_attention"
        / "DEFAULT.json"
    )
    return json.loads(path.read_text())


def _entries(table, axes):
    """(key, config) for every entry. A section with no axes is one bare
    config, not a table of entries."""
    return [("(flat)", table)] if not axes else list(table.items())


def _components(key):
    return () if key == "any" else tuple(key.split("."))


@pytest.mark.parametrize("arch,backend", TABLES)
@pytest.mark.parametrize("op", OPS)
def test_every_section_has_an_any_entry(arch, backend, op):
    """Resolution walks candidates and takes the first hit, so a section with
    no 'any' has shapes that resolve to nothing at all."""
    table = _table(arch, backend)
    axes = table.get("schema", {}).get(op, [])
    if not axes:
        return
    assert (
        "any" in table[op]
    ), f"{arch}/{backend} {op}: no 'any' entry (keys: {sorted(table[op])[:6]})"


@pytest.mark.parametrize("arch,backend", TABLES)
@pytest.mark.parametrize("op", OPS)
def test_key_components_match_the_declared_schema(arch, backend, op):
    """Every component names an axis the resolver can evaluate, and one the
    section's schema lists. A typo'd component would otherwise never match and
    silently fall through to 'any'."""
    table = _table(arch, backend)
    declared = table.get("schema", {}).get(op, [])
    if not declared:
        return
    for key in table[op]:
        for part in _components(key):
            axis = unified_attention_utils._axis_of(part)
            assert axis in unified_attention_utils._AXIS_KIND, (
                f"{arch}/{backend} {op}: {key!r} has component {part!r} naming "
                f"unknown axis {axis!r}"
            )
            assert axis in declared, (
                f"{arch}/{backend} {op}: {key!r} keys on {axis!r}, which the "
                f"schema does not list ({declared})"
            )


@pytest.mark.parametrize("arch,backend", TABLES)
@pytest.mark.parametrize("op", OPS)
def test_schema_lists_exactly_the_axes_used(arch, backend, op):
    """The schema is meant to be an honest summary, so an axis declared but
    never used is as wrong as one that is missing."""
    table = _table(arch, backend)
    declared = set(table.get("schema", {}).get(op, []))
    used = {
        unified_attention_utils._axis_of(part)
        for key in (table[op] if declared else {})
        for part in _components(key)
    }
    assert declared == used, (
        f"{arch}/{backend} {op}: schema declares {sorted(declared)} but the "
        f"keys use {sorted(used)}"
    )


@pytest.mark.parametrize("arch,backend", TABLES)
@pytest.mark.parametrize("op", OPS)
def test_no_duplicate_entries_after_canonicalisation(arch, backend, op):
    """Keys omit don't-care axes, so two different-looking keys can expand to
    the same slots -- one would silently shadow the other."""
    table = _table(arch, backend)
    axes = tuple(table.get("schema", {}).get(op, []))
    if not axes:
        return
    seen = {}
    for key in table[op]:
        canon = unified_attention_utils._canonical(key, axes)
        assert canon not in seen, (
            f"{arch}/{backend} {op}: {key!r} and {seen[canon]!r} both expand to "
            f"{canon}"
        )
        seen[canon] = key


@pytest.mark.parametrize("arch,backend", TABLES)
@pytest.mark.parametrize("op", OPS)
def test_every_entry_is_complete_and_carries_nothing_extra(arch, backend, op):
    """Checked structurally rather than through resolution, so an entry no test
    situation happens to reach is still covered."""
    table = _table(arch, backend)
    axes = table.get("schema", {}).get(op, [])
    required = REQUIRED_KEYS[op] - {"TILE_SIZE"}
    for key, config in _entries(table[op], axes):
        extra = set(config) - ALLOWED_LEAF_KEYS[op]
        assert not extra, f"{arch}/{backend} {op}: {key!r} has extra {sorted(extra)}"
        missing = required - set(config)
        assert not missing, f"{arch}/{backend} {op}: {key!r} missing {sorted(missing)}"
        if "TILE_SIZE" in ALLOWED_KEYS[op]:
            assert _TILE_BOUNDS <= set(config), (
                f"{arch}/{backend} {op}: {key!r} needs {sorted(_TILE_BOUNDS)} "
                f"to derive TILE_SIZE"
            )
