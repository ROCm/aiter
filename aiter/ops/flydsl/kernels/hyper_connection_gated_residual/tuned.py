# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Data-driven kernel-config selection from the offline tuning sweep.

Loads a distilled, per-architecture lookup table (produced by
``op_tests/flydsl_tests/tune.py --export``, see also the scratch tuner) and
returns the best-measured block config for a given ``(kernel, token count)``.
Shapes that were not tuned resolve to the nearest tuned token count in log
space, which is the natural interpolation for GEMM-like latency curves.

This replaces hand-tuned dispatch thresholds (e.g. a fixed split-K vs pipeline
cutover) with the actual measurements. When the table is missing an arch or
kernel the lookups return ``None`` so callers fall back to their heuristic
defaults, keeping the kernels usable on untuned hardware.

Table schema (``tuned_configs.json``)::

    {
      "gfx950": {
        "up_gate_mix": {"256": {"block_m": 64, "block_n": 32,
                                 "m_waves": 1, "n_waves": 2, "_us": 13.14}, ...},
        "down":        {"256": {"method": "splitk",
                                 "config": {...}, "_us": 18.88}, ...}
      }
    }

The ``_us`` field is provenance only (the measured latency) and is ignored at
runtime.
"""
import functools
import json
import math
import os

_TABLE_PATH = os.path.join(os.path.dirname(__file__), "tuned_configs.json")


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


def up_gate_mix_config(arch: str, tokens: int):
    """Tuned ``{block_m, block_n, m_waves, n_waves}`` or ``None``."""
    e = _entry(arch, "up_gate_mix", tokens)
    if not e:
        return None
    return {k: e[k] for k in ("block_m", "block_n", "m_waves", "n_waves")}


def k1_plan(arch: str, tokens: int):
    """Tuned fused-K1 (combine+norm+down) plan or ``None``.

    Entry is a flat dict: ``method`` (``"decouple"`` or ``"splitk"``) plus the
    kwargs :func:`flydsl_k1_combine_norm_down` consumes -- ``split_k`` (split-K
    only), ``block_k``, ``sk_block_m`` (split-K partial tile height),
    ``dn_block_n``/``dn_block_m``/``dn_m_waves``/``dn_n_waves``.     Missing keys let
    the kernel keep its heuristic default for that dimension, so a partial entry
    is valid. ``_us`` is provenance only.

    Unlike the other tables this does **not** extrapolate below its smallest
    tuned token: the low/mid-M split-K heuristic is already tuned and must
    not be overwritten by a nearest-snap onto a large-M decouple entry. Below the
    tuned range this returns ``None`` (kernel keeps its heuristic).
    """
    tbl = _table().get(arch, {}).get("k1")
    if not tbl:
        return None
    keys = sorted(int(k) for k in tbl)
    if tokens < keys[0]:
        return None
    key = _nearest_tokens(tbl, tokens)
    return tbl[str(key)] if key is not None else None
