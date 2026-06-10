# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Per-shape JSON dispatch layer for the FlyDSL ``jagged_dense_bmm_gen`` kernel.

This is a forward-compatible companion to the sibling-branch dispatch
(``jagged_dense_bmm_dispatch.py``), wrapping OUR generalized kernel in
``kernels/jagged_dense_bmm_gen.py`` rather than the sibling's
``kernels/jagged_dense_bmm.py``. It is named ``_v2`` so the two can coexist if
the sibling branch ever merges.

Design (mirrors the sibling):

    JSON  : arch-keyed-v1 schema:
              {schema, by_arch: {<arch>: {gfx, source, fallback, winners}}}
            The loader picks the section matching the detected ROCm arch
            (flydsl.runtime.device.get_rocm_arch, overridable via
            FLYDSL_JAGGED_DENSE_BMM_ARCH). winners maps a per-shape id -> a
            config dict. gfx942 (MI300X) and gfx950 (MI355X) have SEPARATE
            sections because gfx942 has no 16x16x32 bf16 atom -- the gfx950
            winners force use_mfma_k32=true, which core-dumps on gfx942. A flat
            (legacy, non-by_arch) JSON is still accepted as-is, so an env-var
            override file (FLYDSL_JAGGED_DENSE_BMM_DISPATCH_V2_JSON) may use
            either schema.
    py    : _shape_id(...), _load_dispatch_table() (cached, arch-selected),
            _resolve_dispatch(...) = explicit-override > exact-match > heuristic,
            config normalize/validate, and a public wrapper that resolves the
            config then calls jagged_dense_bmm() with the chosen knobs.

Shape id (mirrors sibling naming, with the confusing HSTU mapping):

    B{n_groups}D{reduction_K}K{output_N}N{max_seq_len}

    where reduction_K is the bench's "D" (== our kernel's reduction K, the
    dense ``(B*N, K)`` second dim) and output_N is the bench's "K" (== our
    kernel's output N, the dense first dim / n_groups). ``max_seq_len`` (Mi) is
    the per-group sequence length used to size the M grid envelope.

CRITICAL scoping (what config space this layer tunes over TODAY)

    The generalized kernel currently exposes exactly these per-call tunables:
        xcd_c        : chiplet (XCD) remap chunk C    (1 disables the remap)
        xcd_w        : chiplet (XCD) remap window W
        use_mfma_k32 : CDNA4 16x16x32 bf16 atom (gfx950 default True)
        block_k      : shape-derived (128 if reduction_K<=256 else 64); a winner
                       MAY force the non-default value via the "block_k" key.
    BLOCK_M / BLOCK_N / STAGES_A / THREADS are module constants in
    jagged_dense_bmm_gen.py and are NOT per-call tunable yet, so this layer does
    NOT attempt to set them.

Forward compatibility (not-yet-merged warp / waves_per_eu / tile knobs)

    Two sibling clones (jagged_dense_bmm_warps.py, jagged_dense_bmm_wpe.py) are
    concurrently adding warp-layout (m_warps/n_warps/tile_n) and waves_per_eu
    knobs. To let their winners land in THIS same table without a schema change,
    the config schema below already carries those keys with inert defaults:
        m_warps, n_warps, tile_m, tile_n, tile_k, stages, waves_per_eu, b_to_lds
    They are NORMALIZED and stored but IGNORED by the apply path today (only
    xcd_c/xcd_w/use_mfma_k32/block_k are forwarded to jagged_dense_bmm). When the
    clones merge, the apply path can begin forwarding the extra keys; existing
    JSON winners stay valid.

Usage::

    from aiter.ops.flydsl.jagged_dense_bmm_dispatch_v2 import (
        jagged_dense_bmm_dispatched,
    )
    out = jagged_dense_bmm_dispatched(
        C, A, B, BIAS, SEQ_OFFSETS, n_groups=B, max_seq_len=Mi, stream=st,
    )
"""

from __future__ import annotations

import functools
import json
import os
from pathlib import Path
from typing import Optional

from .kernels.jagged_dense_bmm_gen import jagged_dense_bmm
from .kernels.jagged_dense_bmm_persist_dev import jagged_dense_bmm as jagged_dense_bmm_persist

__all__ = [
    "jagged_dense_bmm_dispatched",
    "resolve_config",
    "shape_id",
]

_DISPATCH_TABLE: Optional[dict] = None
_DISPATCH_CACHE: dict[tuple, dict] = {}

# Inert defaults for the full forward-compatible config schema. The keys the
# apply path forwards TODAY are xcd_c / xcd_w / use_mfma_k32 / block_k; the rest
# are reserved for the not-yet-merged warp / waves_per_eu / tile clones and are
# normalized + stored but ignored on apply.
_SCHEMA_DEFAULTS = {
    # --- live knobs (forwarded to jagged_dense_bmm today) ---
    "xcd_c": None,  # None -> kernel picks weight-size-dependent default
    "xcd_w": None,  # None -> kernel default (8 for uniform_seqlen)
    "use_mfma_k32": None,  # None -> auto (True on gfx950)
    "block_k": None,  # None -> shape-derived (128 if reduction_K<=256 else 64)
    # --- reserved / forward-compatible (ignored on apply today) ---
    "tile_m": 128,  # == BLOCK_M module constant
    "tile_n": 128,  # == BLOCK_N module constant
    "tile_k": None,  # mirrors block_k; reserved for the future warp clone
    "stages": 2,  # == STAGES_A module constant
    "m_warps": 4,  # current tiled_mma is (1,4,1) -> 4 N-warps; reserved
    "n_warps": 1,
    "waves_per_eu": 0,  # 0 == compiler default; reserved for the wpe clone
    "b_to_lds": False,  # gen kernel streams B to regs (g2r), not LDS; reserved
}


def _dispatch_json_paths() -> tuple[Path, ...]:
    env = os.environ.get("FLYDSL_JAGGED_DENSE_BMM_DISPATCH_V2_JSON")
    if env:
        return (Path(env),)
    return (Path(__file__).resolve().parent / "jagged_dense_bmm_dispatch_v2.json",)


def _detect_arch() -> Optional[str]:
    """Detected ROCm arch (e.g. ``gfx942``), or None if unavailable. Overridable
    via ``FLYDSL_JAGGED_DENSE_BMM_ARCH`` for testing a non-native table."""
    env = os.environ.get("FLYDSL_JAGGED_DENSE_BMM_ARCH")
    if env:
        return env
    try:
        from flydsl.runtime.device import get_rocm_arch

        return get_rocm_arch()
    except Exception:
        return None


def _select_arch_section(data: dict, arch: Optional[str]) -> dict:
    """Pick the per-arch sub-table from an ``arch-keyed-v1`` JSON. Falls back to
    an exact ``gfx`` match, else the lone section if only one exists, else an
    empty table. A flat (legacy, non-arch-keyed) JSON is returned as-is."""
    by_arch = data.get("by_arch")
    if not isinstance(by_arch, dict):
        return data  # legacy flat schema (e.g. an env-var override file)
    if arch and arch in by_arch:
        return by_arch[arch]
    if arch:
        for sect in by_arch.values():
            if sect.get("gfx") == arch:
                return sect
    if len(by_arch) == 1:
        return next(iter(by_arch.values()))
    return {}


def _load_dispatch_table() -> dict:
    global _DISPATCH_TABLE
    if _DISPATCH_TABLE is not None:
        return _DISPATCH_TABLE
    arch = _detect_arch()
    for path in _dispatch_json_paths():
        if path.is_file():
            data = json.loads(path.read_text())
            section = _select_arch_section(data, arch)
            _DISPATCH_TABLE = {
                "gfx": section.get("gfx"),
                "winners": dict(section.get("winners") or {}),
                "fallback": dict(section.get("fallback") or {}),
            }
            return _DISPATCH_TABLE
    _DISPATCH_TABLE = {"gfx": None, "winners": {}, "fallback": {}}
    return _DISPATCH_TABLE


def shape_id(*, n_groups: int, reduction_k: int, output_n: int, max_seq_len: int) -> str:
    """Per-shape key. ``reduction_k`` is bench-D, ``output_n`` is bench-K."""
    return f"B{n_groups}D{reduction_k}K{output_n}N{max_seq_len}"


def _coerce(v, kind):
    if v is None:
        return None
    if kind is bool:
        return bool(v)
    return int(v)


def _normalize_cfg(cfg: dict) -> dict:
    """Coerce a raw (JSON or kwarg) config to the full schema with defaults.

    Unknown keys are dropped; missing keys take their schema default. Optional
    knobs (xcd_c/xcd_w/use_mfma_k32/block_k/tile_k) stay ``None`` so the kernel's
    own shape-derived defaults still apply when a winner omits them.
    """
    out = {}
    for key, default in _SCHEMA_DEFAULTS.items():
        raw = cfg.get(key, default)
        if key == "use_mfma_k32":
            out[key] = _coerce(raw, bool)
        elif key == "b_to_lds":
            out[key] = bool(raw) if raw is not None else False
        else:
            out[key] = _coerce(raw, int)
    return out


def _config_valid(cfg: dict, *, reduction_k: int, output_n: int, n_groups: int) -> bool:
    """Cheap validity gate for the live knobs.

    - output_n must tile by BLOCK_N (128) so the n-block grid is exact.
    - reduction_k must tile by the effective block_k (must also leave >=2 K-tiles
      for the 2-stage double-buffer; that is why block_k<=reduction_k//2).
    - the XCD remap is a bijection for any C/W (the kernel identity-maps the tail),
      so no batch-divisibility constraint is needed here, unlike the sibling.
    """
    block_n = cfg.get("tile_n") or 128
    if output_n % block_n != 0:
        return False
    bk = cfg.get("block_k")
    if bk is None:
        bk = 128 if reduction_k <= 256 else 64
    if reduction_k % bk != 0 or reduction_k // bk < 2:
        return False
    return True


def _d_bucket(reduction_k: int) -> str:
    if reduction_k <= 256:
        return "d_le_256"
    if reduction_k <= 512:
        return "d_le_512"
    if reduction_k <= 1024:
        return "d_le_1024"
    return "d_big"


def _heuristic_dispatch(*, reduction_k: int, output_n: int, n_groups: int) -> dict:
    """D-bucketed fallback for shapes absent from ``winners``.

    Reads the JSON ``fallback`` (global + by_d_bucket) when present, else uses
    the kernel's own shape-derived defaults (xcd_c/xcd_w left None). Per prior
    sweeps the robust live-knob default is the kernel's auto choice
    (xcd_c=32 for reduction_K<=256, 60 otherwise; xcd_w=8), so the heuristic
    simply leaves those None and lets the kernel pick unless the JSON overrides.
    """
    rules = _load_dispatch_table().get("fallback") or {}
    base = dict(rules.get("global") or {})
    by_bucket = rules.get("by_d_bucket") or {}
    bucket = _d_bucket(reduction_k)
    if isinstance(by_bucket.get(bucket), dict):
        base.update(by_bucket[bucket].get("config") or {})
    return _normalize_cfg(base)


def resolve_config(
    *,
    n_groups: int,
    reduction_k: int,
    output_n: int,
    max_seq_len: int,
    xcd_c: Optional[int] = None,
    xcd_w: Optional[int] = None,
    use_mfma_k32: Optional[bool] = None,
    block_k: Optional[int] = None,
) -> dict:
    """Full config for a shape: explicit override > table exact-match > heuristic.

    An explicit kwarg (any of the live knobs) always wins for that knob. The
    remaining knobs come from the table winner (exact shape_id match) or, absent
    that, the D-bucketed heuristic. Returns a normalized full-schema dict.
    """
    explicit = {
        "xcd_c": xcd_c,
        "xcd_w": xcd_w,
        "use_mfma_k32": use_mfma_k32,
        "block_k": block_k,
    }
    any_explicit = any(v is not None for v in explicit.values())

    key = (n_groups, reduction_k, output_n, max_seq_len, xcd_c, xcd_w, use_mfma_k32, block_k)
    cached = _DISPATCH_CACHE.get(key)
    if cached is not None:
        return dict(cached)

    table = _load_dispatch_table()
    sid = shape_id(n_groups=n_groups, reduction_k=reduction_k, output_n=output_n, max_seq_len=max_seq_len)
    entry = table["winners"].get(sid)
    if entry is not None:
        cfg = _normalize_cfg(entry)
        if not _config_valid(cfg, reduction_k=reduction_k, output_n=output_n, n_groups=n_groups):
            cfg = _heuristic_dispatch(reduction_k=reduction_k, output_n=output_n, n_groups=n_groups)
    else:
        cfg = _heuristic_dispatch(reduction_k=reduction_k, output_n=output_n, n_groups=n_groups)

    # Explicit kwargs override the resolved config for those specific knobs.
    if any_explicit:
        for k, v in explicit.items():
            if v is not None:
                cfg[k] = v if k == "use_mfma_k32" else int(v)

    _DISPATCH_CACHE[key] = dict(cfg)
    return dict(cfg)


def jagged_dense_bmm_dispatched(
    C,
    A,
    B,
    BIAS,
    SEQ_OFFSETS,
    n_groups: int,
    max_seq_len: int,
    stream=None,
    uniform_seqlen: bool = True,
    # explicit overrides (None -> dispatch table / heuristic / kernel default)
    xcd_c: Optional[int] = None,
    xcd_w: Optional[int] = None,
    use_mfma_k32: Optional[bool] = None,
    block_k: Optional[int] = None,
):
    """Dispatched public wrapper around ``jagged_dense_bmm``.

    Same tensor inputs as the kernel's public entry. Resolves the per-shape
    config from the JSON table (or heuristic fallback), then calls the kernel
    with the chosen live knobs. Tensor layout / construction is the caller's
    responsibility (identical to the bench convention: tall pre-transposed dense
    ``(B*N, K)``, flat bias, padded output, A/C ``mark_layout_dynamic``).
    """
    # OUR kernel derives N (output) and K (reduction) from the dense matrix.
    output_n = B.shape[0] // n_groups  # bench-K
    reduction_k = B.shape[1]  # bench-D
    cfg = resolve_config(
        n_groups=n_groups,
        reduction_k=reduction_k,
        output_n=output_n,
        max_seq_len=max_seq_len,
        xcd_c=xcd_c,
        xcd_w=xcd_w,
        use_mfma_k32=use_mfma_k32,
        block_k=block_k,
    )

    import flydsl.expr as fx

    if stream is None:
        stream = fx.Stream(None)

    # --- Skew gate: pick the kernel variant (ARCH-DEPENDENT) ---
    # Under SKEW the right default is the STATIC kernel with the XCD remap turned
    # OFF (round-robin), which spreads the surviving tiles evenly across all 8 XCDs
    # (the gen kernel does this automatically for uniform_seqlen=False).
    #
    # The persistent problem-visitor variant (on-device CUM prefix, no host sync)
    # was a win ONLY on gfx950 for the small-weight/large-B cell (output_n<=256 AND
    # n_groups>=1024, e.g. B1024_D256): there it beat static remap-off ~1.13x.
    # On gfx942 (MI300X) that does NOT hold -- re-measured 2026-06-10
    # (op_tests/flydsl_tests/_regime_gates.py, do_bench cold-L2): for B1024_D256
    # skew, persist=1.045ms LOSES to static remap-off=0.904ms (1.16x) and to static
    # remap-ON=0.857ms (1.22x). Persist loses on every gfx942 skew shape, so it is
    # gated to gfx95x only. (gfx942 routing of B1024_D256 skew -> remap-ON is set
    # via _SKEW_REMAP_ON below.)
    arch = _detect_arch()
    is_gfx95x = bool(arch) and arch.lower().startswith("gfx95")
    total_out = n_groups * output_n * max_seq_len
    use_persist = (
        is_gfx95x
        and (not uniform_seqlen)
        and output_n <= 256
        and n_groups >= 1024
        and total_out >= (256 * 256 * 4096)
    )
    if use_persist:
        return jagged_dense_bmm_persist(C, A, B, BIAS, SEQ_OFFSETS, n_groups, max_seq_len, stream=stream)

    # Skew + gfx942: the small-weight/large-B cell (B1024_D256) prefers the XCD
    # remap ON even under skew (0.857 vs 0.904 remap-off, 1.05x), because with many
    # groups the L2 reuse outweighs the chiplet imbalance there. Every other skew
    # shape prefers remap-off (D512: remap-on is 1.2-1.6x SLOWER). Re-measured
    # 2026-06-10 on gfx942.
    skew_remap_on = (
        (not uniform_seqlen)
        and (not is_gfx95x)
        and output_n <= 256
        and n_groups >= 1024
    )
    if skew_remap_on:
        return jagged_dense_bmm(
            C, A, B, BIAS, SEQ_OFFSETS, n_groups, max_seq_len, stream=stream,
            xcd_c=32, xcd_w=8, use_mfma_k32=cfg["use_mfma_k32"], uniform_seqlen=False,
        )

    # block_k is shape-derived inside the kernel; it has no public-entry arg, so
    # a forced block_k from the table cannot be applied without editing the
    # kernel. It is therefore validated/stored but only the auto value is used.
    # (Reserved for when the warp clone exposes block_k as a public knob.)
    return jagged_dense_bmm(
        C,
        A,
        B,
        BIAS,
        SEQ_OFFSETS,
        n_groups,
        max_seq_len,
        stream=stream,
        xcd_c=cfg["xcd_c"] if uniform_seqlen else None,
        xcd_w=cfg["xcd_w"] if uniform_seqlen else None,
        use_mfma_k32=cfg["use_mfma_k32"],
        uniform_seqlen=uniform_seqlen,
    )
