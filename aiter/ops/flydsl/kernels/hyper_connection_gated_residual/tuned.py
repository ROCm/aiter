# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Select measured kernel configurations by architecture and tensor shape.

Untuned token counts use the nearest measured count in log space. Missing
architecture or kernel tables return ``None`` so callers can use heuristics.

Table schema (``aiter/configs/model_configs/hc_gated_residual_tuned.json``)::

    {
      "gfx950": {
        "up_gate_mix": {"256": {"block_m": 64, "block_n": 32,
                                 "m_waves": 1, "n_waves": 2, "_us": ...}, ...},
        "up_gate_mix_n336": {"256": {"block_m": 32, "block_n": 16,
                                      "m_waves": 1, "n_waves": 1, ...}, ...},
        "k1":          {"256": {"method": "splitk", ...}, ...},
        "k1_n336":     {"256": {"method": "splitk", ...}, ...}
      }
    }

The ``_us`` field is provenance only (the measured latency) and is ignored at
runtime.
"""

import functools
import json
import math
import os

_TABLE_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "..",
        "..",
        "configs",
        "model_configs",
        "hc_gated_residual_tuned.json",
    )
)


@functools.lru_cache(maxsize=1)
def _table() -> dict:
    """Load and cache the tuned-config table; empty dict if absent/invalid."""
    try:
        with open(_TABLE_PATH) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _nearest_tokens(shape_map: dict, tokens: int):
    """Tuned token count closest to ``tokens`` in log space (exact if present)."""
    keys = sorted(int(k) for k in shape_map)
    if not keys:
        return None
    if tokens in keys:
        return tokens
    ref = math.log(max(tokens, 1))
    return min(keys, key=lambda k: abs(math.log(k) - ref))


def _entry(arch: str, kernel: str, tokens: int):
    tbl = _table().get(arch, {}).get(kernel)
    if not tbl:
        return None
    key = _nearest_tokens(tbl, tokens)
    return tbl[str(key)] if key is not None else None


def up_gate_mix_config(arch: str, tokens: int, packed_width: int | None = None):
    """Tuned ``{block_m, block_n, m_waves, n_waves}`` or ``None``."""
    arch_table = _table().get(arch, {})
    specific = (
        arch_table.get(f"up_gate_mix_n{packed_width}")
        if packed_width is not None
        else None
    )
    # K1's packed row stride changes K2's memory access pattern. Apply
    # stride-specific entries only to measured token counts.
    e = specific.get(str(tokens)) if specific else None
    if not e:
        e = _entry(arch, "up_gate_mix", tokens)
    if not e:
        return None
    return {k: e[k] for k in ("block_m", "block_n", "m_waves", "n_waves")}


def k1_plan(arch: str, tokens: int, n_pad: int | None = None):
    """Tuned fused-K1 (combine+norm+down) plan or ``None``.

    Entry is a flat dict: ``method`` (``"decouple"`` or ``"splitk"``) plus the
    kwargs :func:`flydsl_k1_combine_norm_down` consumes -- ``split_k`` (split-K
    only), ``stages``, ``block_k``, ``sk_block_m`` (split-K partial tile height),
    ``dn_block_n``/``dn_block_m``/``dn_m_waves``/``dn_n_waves``. Missing keys
    keep the kernel's heuristic default. ``_us`` is provenance only.

    Counts below the smallest measured entry return ``None`` to preserve the
    low-token split-K heuristic.
    """
    arch_table = _table().get(arch, {})
    # A shape-specific table must remain active outside its measured range
    # because the generic table may use incompatible N-wave geometry.
    specific = arch_table.get(f"k1_n{n_pad}") if n_pad is not None else None
    tbl = specific or arch_table.get("k1")
    if not tbl:
        return None
    keys = sorted(int(k) for k in tbl)
    if tokens < keys[0]:
        return None
    key = _nearest_tokens(tbl, tokens)
    return tbl[str(key)] if key is not None else None
