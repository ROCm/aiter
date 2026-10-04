#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
MHA forward tuning driver (productionized version).

End-to-end pipeline:

    mha_untune_<gid>_<sig>.csv
        (<sig> = <mode>_<dtype>_hq<HQ>_hv<HV>_mask<M>_nh<NQ>_nhk<NK>_<logits>_
                 <bias>_<lse>_<dropout>_<skip>_<qscale>_<sink>)
        |
        |  1) enum      : enumerate legal FmhaFwdTileSize candidates for each
        |                 (tune-hdim-q, tune-hdim-v) pair
        |
        |  2) build     : cmake configure + `cmake --build` each candidate
        |                 (each pair has its own build sub-dir; each tile has
        |                  its own build_<name>/ under that sub-dir)
        |
        |  3) bench     : for each max_seqlen row in the untune CSV, run every
        |                 built binary in group mode with -b=1 -s=M -s_k=M,
        |                 pick the top-1 (TFlops desc, time asc). Only
        |                 mode=group untune CSVs are accepted. Q/K lengths
        |                 are always equal (seqlen_q != seqlen_k pending).
        |
        v
    mha_tuned_<gid>_<sig>.csv (each row: signature + bench_variant + metrics
                               + human-readable tile_expr)

Sub-commands:

    enum   : only step 1 (produce tile_candidates.json under work-dir)
    build  : step 1 (if needed) + step 2
    bench  : step 3; auto-runs step 1 (enum) and/or step 2 (build) when
             their products are missing, so `bench` can double as a
             one-shot entry point equivalent to `run`.
    run    : full pipeline in one shot

Build reuse (bench only):

    Every successful tile build writes <build_dir>/.mha_tune_build_stamp.json
    recording hdim_q/hdim_v, tile, the tune-config JSON (incl. signature
    filters), cmake args and build target. `bench` reuses an existing binary
    only when its stamp matches the current run; a missing or different
    stamp is a hard error (exit 2) -- use a new --work-dir or delete the
    listed build_<tile> dirs. The stamp does NOT track CK source, compiler
    or arch changes; clean the work-dir yourself after changing those.

Typical usage:

    python mha_tune.py run \\
        -i /path/to/mha_untune_1_group_bf16_hq256_hv256_mask2_nh16_nhk2_nlogits_nbias_nlse_ndropout_skip_nqscale_nsink.csv \\
        --ck-root  /path/to/composable_kernel \\
        --work-dir /tmp/mha_tune_hq256 \\
        --tune-hdim-q 256 --tune-hdim-v 256 \\
        --jobs 64 --workers 4

-------------------------------------------------------------------------
Which options are used by which sub-command?
-------------------------------------------------------------------------

  Common (all sub-commands): -i / --work-dir / --ck-root /
                             --tune-hdim-q / --tune-hdim-v /
                             --allow-mfma-16 / --occupancy

  build / bench / run only:  --build-target / -j / -w / --cmake-opt /
                             --no-fresh / --stop-on-error / --dry-run
                             --warmup / --repeat

Group signature (NOT command-line options):

  nhead_q / nhead_k, mask, bias, lse, dropout, skip_min_seqlen_q, ... are
  read from the untune CSV (every row carries every signature column and
  the filename encodes the same values; parse_untune_csv() rejects any
  disagreement, missing column or legacy 5-field name). They drive both
  the CustomTuneFactory `filters` (so codegen only compiles the matching
  pipeline variant) and the bench CLI (`-h= -h_k= -mask= -bias= -lse=`).

  skip_min_seqlen_q=1 groups: tile_example_fmha_fwd cannot dispatch skip
  kernels, so the nskip variant is built + benched instead, a [WARN] is
  printed and the tuned CSV records bench_variant=nskip.

  --allow-mfma-16       : extends the enumerable bf16 mfma set from the
                          default [(32,32,16)] to also include
                          [(16,16,16),(16,16,32)]. REQUIRED whenever
                          tune-hdim-q or tune-hdim-v is not a multiple
                          of 32 (e.g. hdim=80), otherwise every candidate
                          is filtered out at enum layer-1 and the tile
                          list is empty.

NOTE: All logic that touches CK-tile codegen (env vars
CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE / CK_TILE_FMHA_FWD_CUSTOM_FACTORY) is
inherited from the reference gen_tune_configs.py POC and MUST be kept in sync
with 3rdparty/composable_kernel's fmha codegen.

Each enumerated tile is materialized to a stand-alone tune-config JSON file
(see `<work-dir>/hq<H>_hv<HV>/tune_configs/<tile_name>.json`) that CK's
codegen reads through CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# ===========================================================================
# 0. Constants
# ===========================================================================

ARCH = "gfx942"

# Mapping from the mask_type stored in mha_untune_*.csv (integer enum, see
# csrc/include/ck_tile_shim.h::mask_enum) to the -mask= value accepted by
# tile_example_fmha_fwd's mask_info::decode
# (3rdparty/composable_kernel/example/ck_tile/01_fmha/mask.hpp).
#
# NOTE: decode() only accepts these bare (no ':') tokens:
#     "0"        -> no_mask
#     "1" / "t"  -> mask_top_left  (default full causal)
#     "2" / "b"  -> mask_bottom_right (default full causal)
# The literal "n" / "g" that this project historically used are ONLY valid
# for `serialize()` output (or as codegen filter names), NOT for `-mask=`
# input. Passing "n" makes the runner exit with
#     Invalid argument: invalid mask value: n
# so we must map to numeric strings.
#
# `window_generic` (mask_type == 3) cannot be represented without the (y, x)
# window; leave it unmapped so the caller aborts with a clear error instead
# of silently benching a wrong shape.
MASK_TYPE_TO_LETTER: dict[int, str | None] = {
    0: "0",  # no_mask
    1: "1",  # mask_top_left      (top-left causal)
    2: "2",  # mask_bottom_right  (bottom-right causal)
    3: None,  # window_generic: needs y,x -> caller must reject
}

# Mapping from the mask_type integer to CK codegen's pipeline `F_mask` key
# (used in CustomTuneFactory's `filters.mask` list). The exact key set depends
# on CK's `--mask` codegen option:
#
#   * simplified (CK's default; what our build path uses): F_mask in {"s_no",
#     "s_mask"} — only distinguishes "any mask" vs "no mask".
#   * generic: F_mask in {"no", "causal", "generic"}.
#
# Keep in sync with
# 3rdparty/composable_kernel/example/ck_tile/01_fmha/codegen/cpp_symbol_map.py
# (get_mask_map).
MASK_TYPE_TO_CK_NAME_SIMPLIFIED: dict[int, str] = {
    0: "s_no",  # no_mask
    1: "s_mask",  # mask_top_left
    2: "s_mask",  # mask_bottom_right
    3: "s_mask",  # window_generic
}
MASK_TYPE_TO_CK_NAME_GENERIC: dict[int, str] = {
    0: "no",
    1: "causal",  # mask_top_left     -> causal
    2: "causal",  # mask_bottom_right -> causal
    3: "generic",  # window_generic
}
# CK codegen's default `--mask` value is "simplified"; our CMake path doesn't
# override it, so pick the simplified key set here.
DEFAULT_MASK_IMPL = "simplified"


def _mask_type_to_ck_name(
    mask_type: int, mask_impl: str = DEFAULT_MASK_IMPL
) -> str | None:
    if mask_impl == "simplified":
        return MASK_TYPE_TO_CK_NAME_SIMPLIFIED.get(int(mask_type))
    if mask_impl == "generic":
        return MASK_TYPE_TO_CK_NAME_GENERIC.get(int(mask_type))
    return None


# Mapping from tile_example_fmha_fwd's -bias= letter to CK codegen's BIAS_MAP
# key. Keep in sync with
# 3rdparty/composable_kernel/example/ck_tile/01_fmha/codegen/cpp_symbol_map.py.
BIAS_LETTER_TO_CK_NAME: dict[str, str] = {
    "n": "no",
    "e": "bias",  # elementwise
    "a": "alibi",
}

# bias_type integer (bias_enum, as dumped / stored in the untune CSV) ->
# tile_example_fmha_fwd `-bias=` letter.
BIAS_TYPE_TO_LETTER: dict[int, str] = {0: "n", 1: "e", 2: "a"}

# ---------------------------------------------------------------------------
# Group signature (single source of truth for mha_tune.py).
#
# Must stay identical to mha_count_shape.GROUP_COLS and
# mha_gen_runtime_json.SIGNATURE_COLS (the tools directory is not a package,
# so each script keeps its own copy; the unit tests assert they agree).
#
# The untune CSV carries every column on every row, and its filename
#   mha_untune_<gid>_<mode>_<dtype>_hq<HQ>_hv<HV>_mask<M>_nh<NQ>_nhk<NK>_
#     <logits>_<bias>_<lse>_<dropout>_<skip>_<qscale>_<sink>.csv
# encodes the same values; parse_untune_csv() requires both to agree. There
# are NO command-line overrides for these fields.
# ---------------------------------------------------------------------------
SIGNATURE_COLS: tuple[str, ...] = (
    "mode",
    "dtype",
    "hdim_q",
    "hdim_v",
    "mask_type",
    "nhead_q",
    "nhead_k",
    "has_logits_soft_cap",
    "bias_type",
    "has_lse",
    "has_dropout",
    "skip_min_seqlen_q",
    "qscale_type",
    "has_sink",
)
_STR_SIGNATURE_COLS = frozenset({"mode", "dtype"})

# Columns of mha_tuned_*.csv: the full signature (consumed by
# mha_gen_runtime_json.py), the kernel variant actually benched, then the
# per-row winner.
TUNED_CSV_COLS: tuple[str, ...] = (
    "max_seqlen",
    *SIGNATURE_COLS,
    "bench_variant",
    "best_hdim_q",
    "best_hdim_v",
    "best_time_ms",
    "best_tflops",
    "best_gbps",
    "best_kname",
    "best_tile_name",
    "best_tile_expr",
    "status",
)

# Filename token -> value, per signature field (mirrors mha_count_shape's
# _BOOL_TOKENS / _BIAS_TOKENS / _QSCALE_TOKENS, which follow CK-tile
# FmhaFwdPipeline.name).
_SIG_BOOL_TOKENS: dict[str, dict[str, int]] = {
    "has_logits_soft_cap": {"nlogits": 0, "logits": 1},
    "has_lse": {"nlse": 0, "lse": 1},
    "has_dropout": {"ndropout": 0, "dropout": 1},
    "skip_min_seqlen_q": {"nskip": 0, "skip": 1},
    "has_sink": {"nsink": 0, "sink": 1},
}
_SIG_BIAS_TOKENS: dict[str, int] = {"nbias": 0, "bias": 1, "alibi": 2}
_SIG_QSCALE_TOKENS: dict[str, int] = {
    "nqscale": 0,
    "pertensor": 1,
    "blockscale": 2,
    "kv_blockscale": 3,
    "mx": 4,
}


def _alt(tokens) -> str:
    # Longest first so that e.g. `kv_blockscale` wins over `blockscale`.
    return "|".join(re.escape(t) for t in sorted(tokens, key=len, reverse=True))


# (field, regex for one filename token, converter) in filename order.
_SIG_TOKEN_SPECS: tuple[tuple[str, str, Any], ...] = (
    ("mode", r"[a-zA-Z0-9]+", str),
    ("dtype", r"[a-zA-Z0-9]+", str),
    ("hdim_q", r"hq(\d+)", int),
    ("hdim_v", r"hv(\d+)", int),
    ("mask_type", r"mask(\d+)", int),
    ("nhead_q", r"nh(\d+)", int),
    ("nhead_k", r"nhk(\d+)", int),
    (
        "has_logits_soft_cap",
        f"({_alt(_SIG_BOOL_TOKENS['has_logits_soft_cap'])})",
        _SIG_BOOL_TOKENS["has_logits_soft_cap"].__getitem__,
    ),
    ("bias_type", f"({_alt(_SIG_BIAS_TOKENS)})", _SIG_BIAS_TOKENS.__getitem__),
    *(
        (c, f"({_alt(_SIG_BOOL_TOKENS[c])})", _SIG_BOOL_TOKENS[c].__getitem__)
        for c in ("has_lse", "has_dropout", "skip_min_seqlen_q")
    ),
    ("qscale_type", f"({_alt(_SIG_QSCALE_TOKENS)})", _SIG_QSCALE_TOKENS.__getitem__),
    (
        "has_sink",
        f"({_alt(_SIG_BOOL_TOKENS['has_sink'])})",
        _SIG_BOOL_TOKENS["has_sink"].__getitem__,
    ),
)
assert tuple(s[0] for s in _SIG_TOKEN_SPECS) == SIGNATURE_COLS


def parse_signature(sig: str) -> dict[str, Any]:
    """Parse a group signature string (the `<sig>` part of
    `mha_untune_<gid>_<sig>.csv`) into {SIGNATURE_COLS: value}.

    Every field is mandatory and must appear in SIGNATURE_COLS order;
    missing / reordered / unknown tokens and trailing garbage raise
    ValueError naming the first offending field and the remaining ones.
    """
    rest = sig
    out: dict[str, Any] = {}
    for i, (field, frag, conv) in enumerate(_SIG_TOKEN_SPECS):
        m = re.match(rf"(?:{frag})(?=_|$)", rest)
        if not m:
            remaining = [s[0] for s in _SIG_TOKEN_SPECS[i:]]
            raise ValueError(
                f"signature {sig!r}: cannot parse field {field!r} at "
                f"{rest!r} (expected /{frag}/); missing or invalid fields: "
                f"{remaining}"
            )
        token = m.group(1) if m.groups() else m.group(0)
        out[field] = conv(token)
        rest = rest[m.end() :]
        rest = rest.removeprefix("_")
    if rest:
        raise ValueError(f"signature {sig!r}: unexpected trailing tokens {rest!r}")
    return out


# CK-tile fmha default binary target.
DEFAULT_BUILD_TARGET = "tile_example_fmha_fwd"

# Default CMake configure options (mirrors gen_tune_configs.py).
# NOTE: `-DFMHA_FWD_GEN_OPTDIM=<hdim_q>` and `-DFMHA_FWD_GEN_FILTER=<glob>`
# (derived from the untune CSV signature, see _gen_filter_from_meta) are
# appended per-pair at runtime.
DEFAULT_CMAKE_OPTIONS: list[str] = [
    "-G",
    "Ninja",
    "-DCMAKE_BUILD_TYPE=Release",
    f"-DGPU_TARGETS={ARCH}",
    "-DBUILD_DEV=OFF",
    "-DBUILD_TESTING=OFF",
    "-DDL_KERNELS=OFF",
    "-DBUILD_MHA_LIB=OFF",
    "-DFMHA_FWD_GEN_RECEIPT=200",
]

# tile_example_fmha_fwd (CK example runner) never sets
# fmha_fwd_traits::skip_min_seqlen_q (fmha_fwd_runner.hpp init_traits; the
# field defaults to false in fmha_fwd.hpp), so it can only dispatch the
# `_nskip` kernel variant. Groups whose signature has skip_min_seqlen_q=1
# are therefore built AND benched with the nskip variant and the tuned CSV
# records bench_variant=nskip. Flip this once the runner can select skip.
BENCH_SKIP_SUPPORTED = False

# Regex used to parse the perf line printed by tile_example_fmha_fwd, e.g.
#   ..., 2.897 ms, 37.69 TFlops, 101.68 GB/s
_PERF_RE = re.compile(
    r"([-+]?\d+(?:\.\d+)?)\s*ms\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*TFlops\s*,\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*GB/s",
    re.IGNORECASE,
)
_KNAME_RE = re.compile(r"(fmha_fwd_[A-Za-z0-9_]+)")

# Untune CSV filename pattern: `mha_untune_<gid>_<sig>.csv`, where <sig> is
# the full group signature written by mha_count_shape.py and parsed by
# parse_signature(). Legacy 5-field names (ending at `mask<M>`) are rejected
# by parse_untune_csv().
_UNTUNE_NAME_RE = re.compile(r"^mha_untune_(?P<gid>\d+)_(?P<sig>.+)\.csv$")


# ===========================================================================
# 1. Tile enumeration (ported from enum_fmha_tile_256_bf16.py)
# ===========================================================================
#
# Filter layers:
#   Layer 0: hardware/semantic constants for the given (hdim, hdim_v).
#   Layer 1: self-consistency (block divisible by warp * mfma, warp totals
#            of both gemms match).
#   Layer 2: Gfx9 codegen rule `check_hdim_tile` - for hdim != (128,128) the
#            qr/qr_async/qs pipelines require F_bm0 == 128.
#   Layer 3: LDS-size coarse upper bound.
#
# This is a *duplicated* rule set; the source of truth is
# 3rdparty/composable_kernel/example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py

BM0_CANDIDATES = [64, 128, 256]
BN0_CANDIDATES = [32, 64, 128, 256]
BK0_CANDIDATES = [16, 32, 64]
BK1_CANDIDATES = [16, 32, 64]

# bf16 mfma on gfx942 (CDNA3): (wm, wn, wk)
MFMA_BF16_DEFAULT: list[tuple[int, int, int]] = [(32, 32, 16)]
MFMA_BF16_EXTRA: list[tuple[int, int, int]] = [(16, 16, 16), (16, 16, 32)]

WARP_TOTAL_CANDIDATES = [4]
WARP_LAYOUTS = {
    4: [(4, 1, 1)],
    8: [(8, 1, 1), (4, 2, 1), (2, 4, 1), (1, 8, 1)],
}
ALIGN_GEMM1_WITH_GEMM0 = True
ALIGN_MFMA_GEMM1_WITH_GEMM0 = True

LDS_LIMIT_BYTES = 60 * 1024
BF16_ELEM_BYTES = 2


def _next_pow2(x: int) -> int:
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _bk0max_candidates(hdim: int) -> list[int]:
    """Allowed F_bk0max values, deduced from CK's official tile table.

    * Most (hdim, hdim_v) rows: bk0max == hdim.
    * Some (e.g. (192, 128)):  bk0max == next_pow2(hdim).
    So we allow both (deduped, exact-first, then padded).
    """
    exact = int(hdim)
    padded = _next_pow2(exact)
    if padded == exact:
        return [exact]
    return [exact, padded]


@dataclass(frozen=True)
class TileSize:
    F_bm0: int
    F_bn0: int
    F_bk0: int
    F_bn1: int
    F_bk1: int
    F_bk0max: int
    F_rm0: int
    F_rn0: int
    F_rk0: int
    F_rm1: int
    F_rn1: int
    F_rk1: int
    F_wm0: int
    F_wn0: int
    F_wk0: int
    F_wm1: int
    F_wn1: int
    F_wk1: int
    F_occupancy: int = -1

    # 19 fields in the order CK's FmhaFwdTileSize expects.
    _ORDERED_FIELDS = (
        "F_bm0",
        "F_bn0",
        "F_bk0",
        "F_bn1",
        "F_bk1",
        "F_bk0max",
        "F_rm0",
        "F_rn0",
        "F_rk0",
        "F_rm1",
        "F_rn1",
        "F_rk1",
        "F_wm0",
        "F_wn0",
        "F_wk0",
        "F_wm1",
        "F_wn1",
        "F_wk1",
        "F_occupancy",
    )

    @property
    def name(self) -> str:
        base = (
            f"b{self.F_bm0}x{self.F_bn0}x{self.F_bk0}x{self.F_bn1}x"
            f"{self.F_bk1}x{self.F_bk0max}"
            f"_r{self.F_rm0}x{self.F_rn0}x{self.F_rk0}"
            f"_r{self.F_rm1}x{self.F_rn1}x{self.F_rk1}"
            f"_w{self.F_wm0}x{self.F_wn0}x{self.F_wk0}"
            f"_w{self.F_wm1}x{self.F_wn1}x{self.F_wk1}"
        )
        if self.F_occupancy != -1:
            base += f"_o{self.F_occupancy}"
        return base

    def as_args(self) -> list[int]:
        return [getattr(self, k) for k in self._ORDERED_FIELDS]

    def fields_dict(self) -> dict[str, int]:
        return {k: int(getattr(self, k)) for k in self._ORDERED_FIELDS}

    def as_ck_expr(self) -> str:
        """Human-readable expression matching the FmhaFwdTileSize(...) form.

        Example:
            FmhaFwdTileSize( 64,  64,  32, 256,  32, 256,  4, 1, 1,  4, 1, 1,
                             16, 16, 32,  16, 16, 32,  -1)
        """
        vals = self.as_args()
        return (
            f"FmhaFwdTileSize("
            f"{vals[0]:3d}, {vals[1]:3d}, {vals[2]:3d}, {vals[3]:3d}, "
            f"{vals[4]:3d}, {vals[5]:3d}, "
            f"{vals[6]:2d}, {vals[7]:d}, {vals[8]:d}, "
            f"{vals[9]:2d}, {vals[10]:d}, {vals[11]:d}, "
            f"{vals[12]:3d}, {vals[13]:2d}, {vals[14]:2d}, "
            f"{vals[15]:3d}, {vals[16]:2d}, {vals[17]:2d}, "
            f"{vals[18]:3d})"
        )


def _layer0_semantic(t: TileSize, hdim: int, hdim_v: int) -> bool:
    if t.F_bn1 != hdim_v:
        return False
    if t.F_bk0max not in _bk0max_candidates(hdim):
        return False
    return t.F_rk0 == 1 and t.F_rk1 == 1


def _layer1_self_consistent(t: TileSize) -> bool:
    w0 = t.F_rm0 * t.F_rn0 * t.F_rk0
    w1 = t.F_rm1 * t.F_rn1 * t.F_rk1
    if w0 != w1:
        return False
    if w0 not in WARP_TOTAL_CANDIDATES:
        return False
    # gemm0
    if (t.F_rm0 * t.F_wm0) == 0 or t.F_bm0 % (t.F_rm0 * t.F_wm0) != 0:
        return False
    if (t.F_rn0 * t.F_wn0) == 0 or t.F_bn0 % (t.F_rn0 * t.F_wn0) != 0:
        return False
    if t.F_wk0 == 0 or t.F_bk0 % t.F_wk0 != 0:
        return False
    # gemm1
    if (t.F_rm1 * t.F_wm1) == 0 or t.F_bm0 % (t.F_rm1 * t.F_wm1) != 0:
        return False
    if (t.F_rn1 * t.F_wn1) == 0 or t.F_bn1 % (t.F_rn1 * t.F_wn1) != 0:
        return False
    return t.F_wk1 != 0 and t.F_bk1 % t.F_wk1 == 0


def _layer2_gfx9_check_hdim_tile(t: TileSize, hdim: int, hdim_v: int) -> bool:
    # Mirror of CompatibilityRuleFactoryGfx9.check_hdim_tile for bf16 +
    # qr/qr_async/qs pipelines.
    if (hdim, hdim_v) == (128, 128):
        if t.F_bn0 != 128:
            return False
    else:
        if t.F_bm0 != 128:
            return False
    return True


def _estimate_lds_bytes(t: TileSize) -> int:
    k_bytes = t.F_bn0 * t.F_bk0 * BF16_ELEM_BYTES
    v_bytes = t.F_bn1 * t.F_bk1 * BF16_ELEM_BYTES
    return 2 * (k_bytes + v_bytes)  # 2x for async double-buffer


def _layer3_lds(t: TileSize) -> bool:
    return _estimate_lds_bytes(t) <= LDS_LIMIT_BYTES


def enumerate_tiles(
    hdim: int,
    hdim_v: int,
    occupancies: Sequence[int],
    mfma_list: Sequence[tuple[int, int, int]],
) -> tuple[list[TileSize], dict[str, int]]:
    """Enumerate legal FmhaFwdTileSize candidates for one (hdim, hdim_v)."""
    stats = {
        "total_enumerated": 0,
        "after_layer0_semantic": 0,
        "after_layer1_self_consistent": 0,
        "after_layer2_gfx9_check_hdim_tile": 0,
        "after_layer3_lds": 0,
    }

    seen: set = set()
    tiles: list[TileSize] = []

    bk0max_choices = _bk0max_candidates(hdim)

    for bm0, bn0, bk0, bk1, bk0max in itertools.product(
        BM0_CANDIDATES,
        BN0_CANDIDATES,
        BK0_CANDIDATES,
        BK1_CANDIDATES,
        bk0max_choices,
    ):
        for warp_total in WARP_TOTAL_CANDIDATES:
            for rm0, rn0, rk0 in WARP_LAYOUTS[warp_total]:
                gemm1_layouts = (
                    [(rm0, rn0, rk0)]
                    if ALIGN_GEMM1_WITH_GEMM0
                    else WARP_LAYOUTS[warp_total]
                )
                for rm1, rn1, rk1 in gemm1_layouts:
                    for mfma0 in mfma_list:
                        mfma1_iter = (
                            [mfma0] if ALIGN_MFMA_GEMM1_WITH_GEMM0 else mfma_list
                        )
                        for mfma1 in mfma1_iter:
                            wm0, wn0, wk0 = mfma0
                            wm1, wn1, wk1 = mfma1

                            probe = TileSize(
                                F_bm0=bm0,
                                F_bn0=bn0,
                                F_bk0=bk0,
                                F_bn1=hdim_v,
                                F_bk1=bk1,
                                F_bk0max=bk0max,
                                F_rm0=rm0,
                                F_rn0=rn0,
                                F_rk0=rk0,
                                F_rm1=rm1,
                                F_rn1=rn1,
                                F_rk1=rk1,
                                F_wm0=wm0,
                                F_wn0=wn0,
                                F_wk0=wk0,
                                F_wm1=wm1,
                                F_wn1=wn1,
                                F_wk1=wk1,
                                F_occupancy=-1,
                            )

                            stats["total_enumerated"] += 1
                            if not _layer0_semantic(probe, hdim, hdim_v):
                                continue
                            stats["after_layer0_semantic"] += 1
                            if not _layer1_self_consistent(probe):
                                continue
                            stats["after_layer1_self_consistent"] += 1
                            if not _layer2_gfx9_check_hdim_tile(
                                probe,
                                hdim,
                                hdim_v,
                            ):
                                continue
                            stats["after_layer2_gfx9_check_hdim_tile"] += 1
                            if not _layer3_lds(probe):
                                continue
                            stats["after_layer3_lds"] += 1

                            for occ in occupancies:
                                tile = TileSize(
                                    F_bm0=bm0,
                                    F_bn0=bn0,
                                    F_bk0=bk0,
                                    F_bn1=hdim_v,
                                    F_bk1=bk1,
                                    F_bk0max=bk0max,
                                    F_rm0=rm0,
                                    F_rn0=rn0,
                                    F_rk0=rk0,
                                    F_rm1=rm1,
                                    F_rn1=rn1,
                                    F_rk1=rk1,
                                    F_wm0=wm0,
                                    F_wn0=wn0,
                                    F_wk0=wk0,
                                    F_wm1=wm1,
                                    F_wn1=wn1,
                                    F_wk1=wk1,
                                    F_occupancy=occ,
                                )
                                if tile.name in seen:
                                    continue
                                seen.add(tile.name)
                                tiles.append(tile)

    tiles.sort(
        key=lambda t: (
            t.F_bm0,
            t.F_bn0,
            t.F_bk0,
            t.F_bk1,
            t.F_bk0max,
            t.F_rm0 * t.F_rn0 * t.F_rk0,
            t.F_rm0,
            t.F_rn0,
            t.F_wm0,
            t.F_wn0,
            t.F_wk0,
            t.F_wm1,
            t.F_wn1,
            t.F_wk1,
            t.F_occupancy,
        )
    )
    return tiles, stats


# ===========================================================================
# 2. Untune CSV IO
# ===========================================================================

# Only CK-tile group (varlen) fwd is tuned; batch / splitkv / appendkv /
# pagedkv are rejected (mha_count_shape.py drops them before grouping).
SUPPORTED_MODES = ("group",)


def _require_supported_mode(mode: str, where: str) -> None:
    if str(mode).strip().lower() not in SUPPORTED_MODES:
        raise ValueError(
            f"{where}: unsupported mode={mode!r}; only "
            f"{'/'.join(SUPPORTED_MODES)} (varlen fwd) is supported by the "
            f"MHA tuning workflow"
        )


@dataclass
class UntuneMeta:
    """Group-level metadata parsed from an untune CSV.

    The SIGNATURE_COLS fields form the "group signature" that pins exactly
    one CK-tile fmha kernel variant (mode/dtype/hdim/mask + traits) and the
    bench shape (nhead_q/nhead_k). They are read from the CSV columns and
    cross-checked against the filename; there are no CLI overrides.
    """

    gid: int  # group id from filename
    mode: str  # always 'group' (only group/varlen fwd is supported)
    dtype: str  # 'bf16' / 'fp16'
    hdim_q: int
    hdim_v: int
    mask_type: int  # 0/1/2 (3 = window_generic is rejected at bench time)
    nhead_q: int
    nhead_k: int
    has_logits_soft_cap: int  # always 0 (supported scope)
    bias_type: int  # 0 = none, 1 = elementwise, 2 = alibi
    has_lse: int
    has_dropout: int  # always 0 (dropout is dropped by mha_count_shape)
    skip_min_seqlen_q: int  # 1 when the dumped min_seqlen_q != 0
    qscale_type: int  # always 0 (supported scope)
    has_sink: int  # always 0 (supported scope)
    input_stem: str  # basename without extension (for output naming)
    input_path: Path

    def signature(self) -> dict[str, Any]:
        return {c: getattr(self, c) for c in SIGNATURE_COLS}


# Signature values mha_tune.py can build + bench. Anything else is outside
# the supported scope (mha_count_shape.py `group` already drops those
# records; this guards against hand-edited CSVs).
_SUPPORTED_SIGNATURE_VALUES: dict[str, tuple[Any, ...]] = {
    "mode": SUPPORTED_MODES,
    "has_logits_soft_cap": (0,),
    "has_dropout": (0,),  # dump has no p_drop; pending future support
    "qscale_type": (0,),
    "has_sink": (0,),
    "bias_type": tuple(BIAS_TYPE_TO_LETTER),
    "has_lse": (0, 1),
    "skip_min_seqlen_q": (0, 1),
}


def _sig_value(col: str, raw: Any, where: str) -> Any:
    s = str(raw).strip()
    if col in _STR_SIGNATURE_COLS:
        return s
    try:
        return int(s)
    except ValueError:
        raise ValueError(f"{where}: column {col}={raw!r} is not an integer") from None


def parse_untune_csv(path: Path) -> tuple[UntuneMeta, list[int]]:
    """Read mha_untune_<gid>_<sig>.csv and return (meta, max_seqlen_list).

    Strict checks (ValueError on failure; no legacy fallback):
      * the filename matches `mha_untune_<gid>_<sig>.csv` and <sig> holds
        every SIGNATURE_COLS field (see parse_signature);
      * the CSV has `max_seqlen` + every SIGNATURE_COLS column;
      * every row carries the same signature values;
      * the CSV signature equals the filename signature;
      * the signature is inside the supported scope (group mode, no
        dropout / soft-cap / qscale / sink).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"untune csv not found: {path}")

    m = _UNTUNE_NAME_RE.match(path.name)
    if not m:
        raise ValueError(
            f"untune csv filename {path.name!r} does not match "
            f"mha_untune_<gid>_<signature>.csv; regenerate it with "
            f"`mha_count_shape.py generate_tune_range`"
        )
    gid = int(m.group("gid"))
    try:
        name_sig = parse_signature(m.group("sig"))
    except ValueError as e:
        raise ValueError(
            f"untune csv filename {path.name!r}: {e}; legacy 5-field names "
            f"are no longer supported, re-run `mha_count_shape.py group` + "
            f"`generate_tune_range`"
        ) from None

    with path.open("r", encoding="utf-8", newline="") as fp:
        reader = csv.DictReader(fp)
        header = list(reader.fieldnames or [])
        rows = list(reader)

    missing = [c for c in ("max_seqlen", *SIGNATURE_COLS) if c not in header]
    if missing:
        raise ValueError(
            f"untune csv {path} missing required columns: {missing}; "
            f"re-run `mha_count_shape.py generate_tune_range`"
        )
    if not rows:
        raise ValueError(f"untune csv is empty (no data rows): {path}")

    csv_sig: dict[str, Any] | None = None
    for i, r in enumerate(rows, start=2):  # header is line 1
        where = f"untune csv {path} line {i}"
        sig = {c: _sig_value(c, r[c], where) for c in SIGNATURE_COLS}
        if csv_sig is None:
            csv_sig = sig
        elif sig != csv_sig:
            diff = [c for c in SIGNATURE_COLS if sig[c] != csv_sig[c]]
            raise ValueError(
                f"{where} has a different group signature than the first "
                f"row (fields {diff}); refusing to tune a mixed group."
            )
    assert csv_sig is not None

    diff = [c for c in SIGNATURE_COLS if csv_sig[c] != name_sig[c]]
    if diff:
        detail = ", ".join(
            f"{c}: filename={name_sig[c]!r} csv={csv_sig[c]!r}" for c in diff
        )
        raise ValueError(
            f"untune csv {path}: filename signature disagrees with the CSV "
            f"columns ({detail})"
        )

    _require_supported_mode(csv_sig["mode"], f"untune csv {path}")
    for c, allowed in _SUPPORTED_SIGNATURE_VALUES.items():
        if csv_sig[c] not in allowed:
            raise ValueError(
                f"untune csv {path}: unsupported {c}={csv_sig[c]!r} "
                f"(supported: {list(allowed)})"
            )

    try:
        max_seqlens = [int(r["max_seqlen"]) for r in rows]
    except ValueError as e:
        raise ValueError(f"untune csv {path}: bad max_seqlen value: {e}") from None

    meta = UntuneMeta(
        gid=gid,
        **csv_sig,
        input_stem=path.stem,
        input_path=path,
    )
    return meta, max_seqlens


def tuned_csv_path(meta: UntuneMeta, work_dir: Path) -> Path:
    """Derive `mha_tuned_*.csv` path from the input untune csv stem."""
    stem = meta.input_stem
    if stem.startswith("mha_untune_"):
        new_stem = "mha_tuned_" + stem[len("mha_untune_") :]
    else:
        new_stem = "mha_tuned_" + stem
    return work_dir / f"{new_stem}.csv"


# ===========================================================================
# 3. CMake configure + build (ported from gen_tune_configs.py)
# ===========================================================================


def _build_tune_config_payload(
    dtype: str,
    hdim: int,
    hdim_v: int,
    tile: TileSize,
    *,
    target: str = ARCH,
    filters: dict[str, list[str]] | None = None,
    disable_check_hdim_tile: bool = True,
) -> dict[str, Any]:
    """Build the CustomTuneFactory JSON payload (v1 schema).

    Layout expected by
    `3rdparty/composable_kernel/example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`:

        {
          "schema_version": 1,
          "target": "gfx942",
          "dtypes": ["bf16"],
          "tiles": {"bf16": {"80,96": [ {F_bm0:...} ]}},
          "filters": {"mode":[...], "mask":[...], ...},
          "relax_rules": {"disable_check_hdim_tile": true}
        }

    Only a single tile is embedded (one JSON per tile → one build dir).
    """
    hkey = f"{int(hdim)},{int(hdim_v)}"
    payload: dict[str, Any] = {
        "schema_version": 1,
        "target": str(target),
        "dtypes": [dtype],
        "tiles": {
            dtype: {
                hkey: [tile.fields_dict()],
            },
        },
        "relax_rules": {
            "disable_check_hdim_tile": bool(disable_check_hdim_tile),
        },
    }
    if filters:
        payload["filters"] = filters
    return payload


def _write_tune_config_file(path: Path, payload: dict[str, Any]) -> str:
    """Write payload to disk, return its serialized (pretty) form."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=False)
    path.write_text(text)
    return text


def _tune_configs_dir(pair_sub_dir: Path) -> Path:
    return pair_sub_dir / "tune_configs"


def _tune_config_path(pair_sub_dir: Path, tile: TileSize) -> Path:
    return _tune_configs_dir(pair_sub_dir) / f"{tile.name}.json"


def _bool_filter(v: int) -> list[str]:
    return ["t" if int(v) else "f"]


def bench_variant(meta: UntuneMeta) -> str:
    """Kernel variant actually built and benched for `meta`: 'skip' or 'nskip'.

    Always 'nskip' while BENCH_SKIP_SUPPORTED is False (see its comment).
    """
    if meta.skip_min_seqlen_q and BENCH_SKIP_SUPPORTED:
        return "skip"
    return "nskip"


def _warn_if_skip_downgraded(meta: UntuneMeta, stage: str) -> None:
    if meta.skip_min_seqlen_q and bench_variant(meta) == "nskip":
        print(
            f"[WARN] {stage}: group needs skip_min_seqlen_q=1 but "
            f"{DEFAULT_BUILD_TARGET} cannot dispatch skip kernels; "
            f"building/benching the nskip variant (bench_variant=nskip)",
            file=sys.stderr,
        )


def _filters_from_meta(meta: UntuneMeta) -> dict[str, list[str]]:
    """Assemble the CustomTuneFactory `filters` block from the untune CSV
    signature, so codegen only emits the kernel variant the group needs.

    `skip` follows bench_variant() (nskip while the CK example runner cannot
    dispatch skip kernels); every other trait mirrors the signature.
    """
    _require_supported_mode(meta.mode, "_filters_from_meta")
    filters: dict[str, list[str]] = {
        "mode": [str(meta.mode)],
        "vlayout": ["row"],
    }
    mask_name = _mask_type_to_ck_name(int(meta.mask_type), DEFAULT_MASK_IMPL)
    if mask_name is not None:
        filters["mask"] = [mask_name]
    bias_letter = BIAS_TYPE_TO_LETTER.get(int(meta.bias_type))
    if bias_letter is None:
        raise ValueError(f"_filters_from_meta: unsupported bias_type={meta.bias_type}")
    filters["bias"] = [BIAS_LETTER_TO_CK_NAME[bias_letter]]
    filters["lse"] = _bool_filter(meta.has_lse)
    filters["dropout"] = _bool_filter(meta.has_dropout)
    filters["logits"] = _bool_filter(meta.has_logits_soft_cap)
    if int(meta.qscale_type) != 0:
        raise ValueError(
            f"_filters_from_meta: unsupported qscale_type={meta.qscale_type}"
        )
    filters["qscale"] = ["no"]
    filters["skip"] = ["t" if bench_variant(meta) == "skip" else "f"]
    filters["sink"] = _bool_filter(meta.has_sink)
    return filters


# CK-tile FmhaFwdPipeline.name tokens used by the FMHA_FWD_GEN_FILTER glob.
_GEN_FILTER_BIAS = {0: "nbias", 1: "bias", 2: "alibi"}


def _gen_filter_from_meta(meta: UntuneMeta) -> str:
    """`-DFMHA_FWD_GEN_FILTER=` glob matching the kernel names of `meta`.

    Kernel names look like
      fmha_fwd_d<H>_<dtype>_<mode>_<tile>_<pipeline>_..._nlogits_nbias_mask_
      nlse_ndropout_nskip_nqscale_...
    The glob is a coarse pre-filter (CustomTuneFactory `filters` do the
    precise selection); it must never exclude the variant we need.
    """
    lse = "lse" if meta.has_lse else "nlse"
    drop = "dropout" if meta.has_dropout else "ndropout"
    bias = _GEN_FILTER_BIAS[int(meta.bias_type)]
    return f"*{meta.dtype}*_{bias}*_{lse}*_{drop}*"


def _which_hipcc() -> str | None:
    from shutil import which

    return which("hipcc")


def _format_cmd(cmd: list[str], env_overrides: dict[str, str]) -> str:
    env_part = " ".join(f"{k}={shlex.quote(v)}" for k, v in env_overrides.items())
    cmd_part = " ".join(shlex.quote(x) for x in cmd)
    return (env_part + " " + cmd_part) if env_part else cmd_part


def _run_cmd(
    cmd: list[str],
    env_overrides: dict[str, str],
    cwd: str,
    dry_run: bool,
    log: list[str] | None = None,
) -> int:
    """Run a subprocess. If `log` is provided we capture combined output into
    it (concurrent-friendly); else we stream directly to stdout.
    """
    header1 = f"  [cmd] cwd={cwd}"
    header2 = f"        {_format_cmd(cmd, env_overrides)}"
    if log is not None:
        log.append(header1)
        log.append(header2)
    else:
        print(header1)
        print(header2)

    if dry_run:
        return 0

    env = os.environ.copy()
    env.update(env_overrides)
    try:
        if log is not None:
            completed = subprocess.run(
                cmd,
                cwd=cwd,
                env=env,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            for line in (completed.stdout or "").rstrip("\n").splitlines():
                log.append(f"    | {line}")
            return completed.returncode
        completed = subprocess.run(cmd, cwd=cwd, env=env, check=False)
        return completed.returncode
    except FileNotFoundError as e:
        msg = f"  [error] command not found: {e}"
        if log is not None:
            log.append(msg)
        else:
            print(msg, file=sys.stderr)
        return 127


def _run_cmd_capture(
    cmd: list[str],
    env_overrides: dict[str, str],
    cwd: str,
    dry_run: bool,
) -> tuple[int, str]:
    """Run a subprocess, capture combined stdout+stderr, mirror to console."""
    print(f"  [cmd] cwd={cwd}")
    print(f"        {_format_cmd(cmd, env_overrides)}")
    if dry_run:
        return 0, ""
    env = os.environ.copy()
    env.update(env_overrides)
    try:
        completed = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except FileNotFoundError as e:
        print(f"  [error] command not found: {e}", file=sys.stderr)
        return 127, ""
    out = completed.stdout or ""
    for line in out.rstrip("\n").splitlines():
        print(f"    | {line}")
    return completed.returncode, out


def _do_configure(
    build_dir: str,
    cfg_json_path: str,
    hipcc: str | None,
    extra_cmake_opts: list[str],
    ck_root: str,
    dry_run: bool,
    fresh: bool,
    log: list[str] | None,
) -> int:
    """`cmake -S <ck_root> -B <build_dir> ...` with tune config injected.

    Must be called with CWD == ck_root so that CK-tile fmha's codegen picks
    up CMakeLists.txt at `.`.
    """

    def _emit(msg: str) -> None:
        if log is not None:
            log.append(msg)
        else:
            print(msg)

    if fresh:
        if os.path.isdir(build_dir):
            _emit(f"  [fresh] purging existing build dir: {build_dir}")
            if not dry_run:
                shutil.rmtree(build_dir, ignore_errors=True)
        else:
            _emit(f"  [fresh] build dir does not exist yet: {build_dir}")
    else:
        _emit(f"  [fresh] SKIPPED (per --no-fresh): {build_dir}")

    cmd: list[str] = ["cmake", "-S", ".", "-B", build_dir]
    if hipcc:
        cmd += [
            f"-DCMAKE_CXX_COMPILER={hipcc}",
            f"-DCMAKE_C_COMPILER={hipcc}",
        ]
    cmd += list(DEFAULT_CMAKE_OPTIONS)
    cmd += list(extra_cmake_opts)

    env_overrides = {
        "CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE": cfg_json_path,
        "CK_TILE_FMHA_FWD_CUSTOM_FACTORY": "0",
    }
    _emit(
        "  [env] CK_TILE_FMHA_FWD_CUSTOM_FACTORY="
        f"{env_overrides['CK_TILE_FMHA_FWD_CUSTOM_FACTORY']}"
        f", CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE={cfg_json_path}"
    )
    return _run_cmd(cmd, env_overrides, cwd=ck_root, dry_run=dry_run, log=log)


def _verify_blob_list(
    build_dir: str,
    expected_tile_token: str,
    dry_run: bool,
) -> tuple[bool, str]:
    """After configure, sanity-check that fwd_blob_list.txt contains
    the expected tile token (i.e. CustomTuneFactory did take effect).
    """
    blob_list = os.path.join(
        build_dir, "example", "ck_tile", "01_fmha", "fwd_blob_list.txt"
    )
    if dry_run:
        return True, f"(dry-run) would check {blob_list}"
    if not os.path.isfile(blob_list):
        return False, f"blob-list not found: {blob_list}"
    with open(blob_list, "r", encoding="utf-8") as f:
        content = f.read()
    total = content.count(".cpp")
    hit = content.count(expected_tile_token)
    sample = "\n    ".join(content.splitlines()[:3])
    if hit == 0:
        return False, (
            f"blob-list has {total} cpp(s) but NONE contains "
            f"'{expected_tile_token}'.\n  first entries:\n    {sample}"
        )
    return True, (
        f"blob-list ok: {hit}/{total} cpp(s) match '{expected_tile_token}'.\n"
        f"  first entries:\n    {sample}"
    )


def _do_make(
    build_dir: str,
    target: str,
    jobs: int,
    cfg_json_path: str,
    ck_root: str,
    dry_run: bool,
    log: list[str] | None,
) -> int:
    """`cmake --build <build_dir> --target <target> -j <jobs>`

    NOTE: CK-tile fmha runs codegen (Python) at BUILD time via
    add_custom_command, so tune-config env vars must be present here too.
    """
    cmd: list[str] = [
        "cmake",
        "--build",
        build_dir,
        "--target",
        target,
        "-j",
        str(jobs),
    ]
    env_overrides = {
        "CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE": cfg_json_path,
        "CK_TILE_FMHA_FWD_CUSTOM_FACTORY": "0",
    }
    return _run_cmd(cmd, env_overrides, cwd=ck_root, dry_run=dry_run, log=log)


def _binary_path(build_dir: str, target: str) -> str:
    return os.path.join(build_dir, "bin", target)


# ---------------------------------------------------------------------------
# Build stamp: records the MHA config a build_<tile>/ dir was compiled with,
# so `bench` can refuse to reuse a binary built for a different signature.
# Deliberately does NOT track CK source / compiler / arch changes.
# ---------------------------------------------------------------------------

BUILD_STAMP_SCHEMA = 1
BUILD_STAMP_FILENAME = ".mha_tune_build_stamp.json"


def _stamp_path(build_dir: str | Path) -> Path:
    return Path(build_dir) / BUILD_STAMP_FILENAME


def _build_stamp(
    plan: TilePlan,
    extra_cmake_opts: Sequence[str],
    build_target: str,
) -> dict[str, Any]:
    """Everything that decides the codegen / compiled kernel of one tile."""
    return {
        "schema": BUILD_STAMP_SCHEMA,
        "hdim_q": plan.pair.hdim_q,
        "hdim_v": plan.pair.hdim_v,
        "tile": plan.tile.name,
        "tune_config": json.loads(plan.cfg_json_text),
        "cmake_args": list(DEFAULT_CMAKE_OPTIONS) + list(extra_cmake_opts),
        "build_target": build_target,
    }


def _read_stamp(build_dir: str | Path) -> dict[str, Any] | None:
    """Return the stamp dict, or None when missing / unreadable."""
    try:
        data = json.loads(_stamp_path(build_dir).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _write_stamp(build_dir: str | Path, stamp: dict[str, Any]) -> None:
    """Atomically write the stamp (tmp file + os.replace)."""
    path = _stamp_path(build_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(stamp, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _remove_stamp(build_dir: str | Path) -> None:
    try:
        _stamp_path(build_dir).unlink()
    except FileNotFoundError:
        pass


def _stamp_diff(old: Any, new: Any, prefix: str = "") -> list[str]:
    """Dotted key paths whose values differ between two stamps."""
    if isinstance(old, dict) and isinstance(new, dict):
        out: list[str] = []
        for k in sorted(set(old) | set(new), key=str):
            key = f"{prefix}.{k}" if prefix else str(k)
            if k not in old or k not in new:
                out.append(key)
            else:
                out.extend(_stamp_diff(old[k], new[k], key))
        return out
    return [] if old == new else [prefix or "<root>"]


# ===========================================================================
# 4. Bench (per-tile invocation of tile_example_fmha_fwd)
# ===========================================================================


def _parse_perf(stdout: str) -> dict[str, Any] | None:
    """Return the LAST perf triplet parsed from stdout, or None."""
    last: dict[str, Any] | None = None
    for raw in stdout.splitlines():
        m = _PERF_RE.search(raw)
        if not m:
            continue
        km = _KNAME_RE.search(raw)
        last = {
            "time_ms": float(m.group(1)),
            "tflops": float(m.group(2)),
            "gbps": float(m.group(3)),
            "kname": km.group(1) if km else "",
            "line": raw.strip(),
        }
    return last


def bench_args_for_meta(
    meta: UntuneMeta,
    *,
    hdim_q_bench: int,
    hdim_v_bench: int,
    max_seqlen: int,
    warmup: int,
    repeat: int,
) -> list[str]:
    """tile_example_fmha_fwd CLI for one max_seqlen row of `meta`'s group.

    Every shape/trait argument (heads, mask, bias, lse, dropout) comes from
    the untune CSV signature; only the compiled hdim and timing knobs come
    from the command line.
    """
    mask_letter = MASK_TYPE_TO_LETTER.get(int(meta.mask_type))
    if mask_letter is None:
        raise ValueError(
            f"mask_type={meta.mask_type} cannot be mapped to a "
            f"tile_example_fmha_fwd -mask= value (window_generic needs an "
            f"explicit y,x window and is not supported by this tuner)."
        )
    if int(meta.has_dropout):
        raise ValueError(
            "has_dropout=1 groups are not supported (the dump has no p_drop)"
        )
    return _build_bench_args(
        dtype=meta.dtype,
        hdim_q_bench=hdim_q_bench,
        hdim_v_bench=hdim_v_bench,
        nhead_q=meta.nhead_q,
        nhead_k=meta.nhead_k,
        max_seqlen=max_seqlen,
        mask_letter=mask_letter,
        lse=int(meta.has_lse),
        p_drop=0.0,
        bias=BIAS_TYPE_TO_LETTER[int(meta.bias_type)],
        warmup=warmup,
        repeat=repeat,
        mode=meta.mode,
    )


def _build_bench_args(
    dtype: str,
    hdim_q_bench: int,
    hdim_v_bench: int,
    nhead_q: int,
    nhead_k: int,
    max_seqlen: int,
    mask_letter: str,
    lse: int,
    p_drop: float,
    bias: str,
    warmup: int,
    repeat: int,
    mode: str,
) -> list[str]:
    """Assemble CLI for tile_example_fmha_fwd (b=1, group mode).

    The kernel instances emitted by CustomTuneFactory are strictly filtered on
    `filters.mode` (see JSON), which comes from the untune CSV (always
    `group`; batch is rejected). The example runner's dispatcher looks
    up instances by `fmha_fwd_traits`, so `-mode` must match what we compiled
    or the runner exits with ", not supported yet".

    In group mode with `-b=1 -s=M -s_k=M`, the runner treats the single batch
    as one variable-length sequence of length M.

    Known limitation: Q and K always have the same length (`-s_k` == `-s`).
    Calls dumped with seqlen_q != seqlen_k (e.g. prefix-cached prefill) are
    tuned by their Q length only (`mha_count_shape.py group` warns about
    them); full seqlen_q != seqlen_k support is pending.
    """
    _require_supported_mode(mode, "_build_bench_args")
    mode_int = 1  # tile_example_fmha_fwd: 0=batch, 1=group
    return [
        f"-prec={dtype}",
        f"-mode={mode_int}",
        "-b=1",  # single sequence
        f"-h={nhead_q}",
        f"-h_k={nhead_k}",
        f"-d={hdim_q_bench}",
        f"-d_v={hdim_v_bench}",
        f"-s={max_seqlen}",
        f"-s_k={max_seqlen}",
        f"-mask={mask_letter}",
        "-scale_s=0",
        f"-bias={bias}",
        f"-lse={lse}",
        f"-p_drop={p_drop}",
        # Disable example runner's Pack-GQA folding path: when h/h_k > 1 and
        # mask=no_mask it tries to fold `nhead_ratio` Q-heads into seqlen_q
        # to reuse an MHA kernel, but the CK version we build against exits
        # with "not supported yet". Real GQA is handled by the kernel itself
        # via nhead_q/nhead_k, so this only affects the runner's fast path.
        "-pack_gqa=0",
        "-vlayout=r",
        "-v=0",
        f"-warmup={warmup}",
        f"-repeat={repeat}",
        "-timer=gpu",
        "-kname=1",
    ]


def _do_bench(
    binary: str,
    bench_args: list[str],
    ck_root: str,
    dry_run: bool,
) -> tuple[str, dict[str, Any] | None]:
    """Run one built binary once; return (status, parsed_perf).

    status in {"ok", "skipped", "run_failed", "no_perf"}.
    """
    if not os.path.isfile(binary):
        print(f"  [skip] binary not found: {binary}")
        return "skipped", None
    if not os.access(binary, os.X_OK):
        print(f"  [skip] binary not executable: {binary}")
        return "skipped", None

    cmd = [binary] + list(bench_args)
    rc, out = _run_cmd_capture(cmd, env_overrides={}, cwd=ck_root, dry_run=dry_run)
    if dry_run:
        return "ok", None
    if rc != 0:
        print(f"  [fail] bench exited rc={rc}", file=sys.stderr)
        return "run_failed", None
    parsed = _parse_perf(out)
    if parsed is None:
        print("  [warn] no perf line detected in binary output.", file=sys.stderr)
        return "no_perf", None
    print(
        f"  [perf] time={parsed['time_ms']:.3f} ms, "
        f"tflops={parsed['tflops']:.2f}, bw={parsed['gbps']:.2f} GB/s"
    )
    return "ok", parsed


# ===========================================================================
# 5. Work item plumbing
# ===========================================================================


@dataclass
class PairPlan:
    """One (tune_hdim_q, tune_hdim_v) configuration.

    Each pair gets its own sub-directory under work_dir to keep the
    `-DFMHA_FWD_GEN_OPTDIM=<hdim_q>` cmake macro separated per hdim.
    """

    hdim_q: int
    hdim_v: int
    sub_dir: Path  # work_dir / f"hq{hq}_hv{hv}"
    tiles_json: Path  # sub_dir / "tile_candidates.json"
    build_root: Path  # sub_dir  (each tile becomes build_root/build_<name>)


@dataclass
class TilePlan:
    """One concrete tile within a pair (post-enum)."""

    pair: PairPlan
    tile: TileSize
    build_dir: Path  # pair.build_root / f"build_{tile.name}"
    cfg_json_path: Path  # pair.sub_dir / "tune_configs" / f"{tile.name}.json"
    cfg_json_text: str  # in-memory snapshot (debug print only)


def _pair_plans(
    work_dir: Path,
    tune_hdim_q: int,
    tune_hdim_v: int,
) -> list[PairPlan]:
    """Materialize per-pair sub-directories.

    Currently we only support a single (hdim_q, hdim_v) per invocation
    (user chose single-value CLI for --tune-hdim-q/v), but keep a list
    return type so multi-pair extensions stay drop-in.
    """
    pair_dir = work_dir / f"hq{tune_hdim_q}_hv{tune_hdim_v}"
    return [
        PairPlan(
            hdim_q=tune_hdim_q,
            hdim_v=tune_hdim_v,
            sub_dir=pair_dir,
            tiles_json=pair_dir / "tile_candidates.json",
            build_root=pair_dir,
        )
    ]


def _write_tiles_json(
    path: Path,
    dtype: str,
    hdim_q: int,
    hdim_v: int,
    tiles: list[TileSize],
    stats: dict[str, int],
) -> None:
    payload = {
        "arch": ARCH,
        "dtype": dtype,
        "hdim": hdim_q,
        "hdim_v": hdim_v,
        "stats": stats,
        "strict_legal": [
            {
                "name": t.name,
                "args": t.as_args(),
                "lds_bytes_estimate": _estimate_lds_bytes(t),
                "warp_total": t.F_rm0 * t.F_rn0 * t.F_rk0,
                "fields": asdict(t),
            }
            for t in tiles
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _read_tiles_json(path: Path) -> tuple[str, int, int, list[TileSize]]:
    data = json.loads(path.read_text())
    dtype = data["dtype"]
    hdim_q = int(data["hdim"])
    hdim_v = int(data["hdim_v"])
    tiles: list[TileSize] = []
    for entry in data.get("strict_legal", []):
        fields = entry["fields"]
        tiles.append(TileSize(**{k: int(fields[k]) for k in TileSize._ORDERED_FIELDS}))
    return dtype, hdim_q, hdim_v, tiles


# ===========================================================================
# 6. Sub-command implementations
# ===========================================================================


def _resolve_occupancies(arg: str | None) -> list[int]:
    """Parse `--occupancy` CLI (comma-separated or single value)."""
    if arg is None or arg == "":
        return [1, 2, 3, 4, 5]
    parts = [x.strip() for x in arg.split(",") if x.strip() != ""]
    if not parts:
        return [1, 2, 3, 4, 5]
    out: list[int] = []
    for p in parts:
        v = int(p)
        if v not in (-1, 1, 2, 3, 4, 5):
            raise ValueError(f"--occupancy value {v} not in {{-1,1,2,3,4,5}}")
        out.append(v)
    return out


def _mfma_list_for(dtype: str, allow_mfma_16: bool) -> list[tuple[int, int, int]]:
    if dtype != "bf16":
        # Currently our enumeration mirrors bf16 rules only.
        print(
            f"[warn] dtype={dtype} is not bf16; enumeration mirrors bf16 "
            f"rules and may miss legal tiles.",
            file=sys.stderr,
        )
    mfma = list(MFMA_BF16_DEFAULT)
    if allow_mfma_16:
        mfma += list(MFMA_BF16_EXTRA)
    return mfma


def cmd_enum(args: argparse.Namespace) -> int:
    """Enumerate tile candidates for one (hdim_q, hdim_v) pair.

    Reads the untune csv only to log the group signature; the tile enum
    itself uses --tune-hdim-q / --tune-hdim-v (the actual hdim the built
    kernels will target). CSV's own hdim_q/hdim_v are NOT enforced to equal
    tune-hdim-q/v (per user's decision: silent mismatch is OK).
    """
    meta, _ = parse_untune_csv(args.input_csv)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    if meta.hdim_q != args.tune_hdim_q or meta.hdim_v != args.tune_hdim_v:
        print(
            f"[note] untune csv reports (hdim_q={meta.hdim_q}, "
            f"hdim_v={meta.hdim_v}) but tune-hdim-q/v = "
            f"({args.tune_hdim_q}, {args.tune_hdim_v}); "
            f"proceeding with the tune values (user override)."
        )

    occs = _resolve_occupancies(args.occupancy)
    mfma_list = _mfma_list_for(meta.dtype, args.allow_mfma_16)

    if args.tune_hdim_v % 32 != 0 and not args.allow_mfma_16:
        print(
            f"[warn] tune-hdim-v={args.tune_hdim_v} is not a multiple of 32; "
            f"with the default mfma set {MFMA_BF16_DEFAULT} layer1 will "
            f"reject every candidate. Pass --allow-mfma-16."
        )

    limit = int(getattr(args, "limit", 0) or 0)

    plans = _pair_plans(work_dir, args.tune_hdim_q, args.tune_hdim_v)
    for pp in plans:
        tiles, stats = enumerate_tiles(
            hdim=pp.hdim_q,
            hdim_v=pp.hdim_v,
            occupancies=occs,
            mfma_list=mfma_list,
        )
        full_count = len(tiles)
        if limit > 0 and full_count > limit:
            tiles = tiles[:limit]
            stats = dict(stats)
            stats["limited_to"] = limit
            print(
                f"[note] --limit {limit} applied: keeping first {limit} "
                f"of {full_count} enumerated tiles for debug."
            )
        _write_tiles_json(pp.tiles_json, meta.dtype, pp.hdim_q, pp.hdim_v, tiles, stats)

        # Also drop one stand-alone tune-config JSON per tile, so downstream
        # `build`/`bench` (and manual invocations) can simply set
        # CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE=<path> without further work.
        # The filters come from the untune CSV signature, so the enum output
        # is already complete (build/bench rewrite the same content).
        enum_filters = _filters_from_meta(meta)

        cfg_dir = _tune_configs_dir(pp.sub_dir)
        cfg_dir.mkdir(parents=True, exist_ok=True)
        for t in tiles:
            payload = _build_tune_config_payload(
                meta.dtype,
                pp.hdim_q,
                pp.hdim_v,
                t,
                target=ARCH,
                filters=enum_filters,
                disable_check_hdim_tile=True,
            )
            _write_tune_config_file(_tune_config_path(pp.sub_dir, t), payload)

        print("=" * 72)
        print(f"pair (hdim_q, hdim_v) = ({pp.hdim_q}, {pp.hdim_v})")
        print(f"dtype        = {meta.dtype}")
        print(f"mfma set     = {mfma_list}")
        print(f"occupancies  = {occs}")
        for k, v in stats.items():
            print(f"  {k:40s} = {v}")
        print(f"strict_legal count = {len(tiles)}")
        print(f"wrote: {pp.tiles_json}")
        print(f"wrote: {len(tiles)} tune-config json(s) under {cfg_dir}")
    return 0


def _tile_plans_for(
    pair: PairPlan,
    tiles: list[TileSize],
    dtype: str,
    *,
    filters: dict[str, list[str]] | None = None,
    disable_check_hdim_tile: bool = True,
) -> list[TilePlan]:
    """Materialize TilePlan for each tile, writing its stand-alone tune JSON.

    If `filters` is provided (build / run cmds), the on-disk JSON is
    overwritten to contain the full v1 payload; if not (enum cmd), only the
    `tiles` + `relax_rules` blocks are written (build stage will overwrite).
    """
    plans: list[TilePlan] = []
    for t in tiles:
        payload = _build_tune_config_payload(
            dtype,
            pair.hdim_q,
            pair.hdim_v,
            t,
            target=ARCH,
            filters=filters,
            disable_check_hdim_tile=disable_check_hdim_tile,
        )
        cfg_path = _tune_config_path(pair.sub_dir, t)
        text = _write_tune_config_file(cfg_path, payload)
        bd = pair.build_root / f"build_{t.name}"
        plans.append(
            TilePlan(
                pair=pair,
                tile=t,
                build_dir=bd,
                cfg_json_path=cfg_path,
                cfg_json_text=text,
            )
        )
    return plans


def _configure_and_build_one(
    plan: TilePlan,
    hipcc: str | None,
    ck_root: str,
    extra_cmake_opts: list[str],
    do_configure: bool,
    do_make: bool,
    build_target: str,
    jobs: int,
    fresh: bool,
    dry_run: bool,
    buffered: bool,
) -> dict[str, Any]:
    log: list[str] | None = [] if buffered else None

    def _emit(msg: str) -> None:
        if log is not None:
            log.append(msg)
        else:
            print(msg)

    _emit(f"[{plan.tile.name}] pair=(hq={plan.pair.hdim_q}, " f"hv={plan.pair.hdim_v})")
    _emit(f"  build_dir={plan.build_dir}")
    _emit(f"  CK_TILE_FMHA_FWD_CUSTOM_TUNE_CONFIG_FILE={plan.cfg_json_path}")

    result: dict[str, Any] = {
        "tile_name": plan.tile.name,
        "pair": (plan.pair.hdim_q, plan.pair.hdim_v),
        "build_dir": str(plan.build_dir),
        "configure_ok": True,
        "build_ok": True,
        "did_configure": False,
        "did_make": False,
        "log": log,
    }

    # Drop any previous stamp first so an interrupted / failed rebuild can
    # never leave "old binary + valid stamp" behind.
    if (do_configure or do_make) and not dry_run:
        _remove_stamp(plan.build_dir)

    if do_configure:
        result["did_configure"] = True
        rc = _do_configure(
            build_dir=str(plan.build_dir),
            cfg_json_path=str(plan.cfg_json_path),
            hipcc=hipcc,
            extra_cmake_opts=extra_cmake_opts,
            ck_root=ck_root,
            dry_run=dry_run,
            fresh=fresh,
            log=log,
        )
        if rc != 0:
            _emit(f"  [fail] configure rc={rc}")
            result["configure_ok"] = False
            result["build_ok"] = False
            return result

        ok_blob, msg_blob = _verify_blob_list(
            build_dir=str(plan.build_dir),
            expected_tile_token=plan.tile.name,
            dry_run=dry_run,
        )
        _emit(f"  [verify] {msg_blob}")
        if not ok_blob:
            _emit(
                "  [fail] blob-list verify failed; "
                "CustomTuneFactory did NOT take effect."
            )
            result["configure_ok"] = False
            result["build_ok"] = False
            return result

    if do_make:
        result["did_make"] = True
        rc = _do_make(
            build_dir=str(plan.build_dir),
            target=build_target,
            jobs=jobs,
            cfg_json_path=str(plan.cfg_json_path),
            ck_root=ck_root,
            dry_run=dry_run,
            log=log,
        )
        if rc != 0:
            _emit(f"  [fail] build rc={rc}")
            result["build_ok"] = False
            return result
        if not dry_run:
            _write_stamp(
                plan.build_dir, _build_stamp(plan, extra_cmake_opts, build_target)
            )
            _emit(f"  [stamp] wrote {_stamp_path(plan.build_dir)}")

    return result


def _extra_cmake_opts_for_pair(
    pair: PairPlan,
    meta: UntuneMeta,
    user_cmake_opts: Sequence[str],
) -> list[str]:
    """Per-pair -DFMHA_FWD_GEN_OPTDIM=<hdim_q> plus the signature-derived
    -DFMHA_FWD_GEN_FILTER. Other --cmake-opt user overrides are kept as-is;
    they win over defaults (CMake takes the last -D<var>=<val>).

    Shared by the build stage and bench's build-stamp check so both see
    exactly the same arguments.
    """
    return [
        f"-DFMHA_FWD_GEN_OPTDIM={pair.hdim_q}",
        f"-DFMHA_FWD_GEN_FILTER={_gen_filter_from_meta(meta)}",
    ] + list(user_cmake_opts)


def _classify_existing_builds(
    plans: list[TilePlan],
    meta: UntuneMeta,
    args: argparse.Namespace,
) -> tuple[list[TilePlan], list[TilePlan], list[tuple[TilePlan, list[str]]]]:
    """Split plans into (reusable, to_build, stale).

    - to_build : binary missing -> build normally.
    - reusable : binary present and its build stamp matches this run.
    - stale    : binary present but stamp missing / unreadable / different;
                 the second item lists the differing stamp keys.
    """
    reusable: list[TilePlan] = []
    to_build: list[TilePlan] = []
    stale: list[tuple[TilePlan, list[str]]] = []
    for p in plans:
        if not os.path.isfile(_binary_path(str(p.build_dir), args.build_target)):
            to_build.append(p)
            continue
        new = _build_stamp(
            p,
            _extra_cmake_opts_for_pair(p.pair, meta, args.cmake_opt),
            args.build_target,
        )
        old = _read_stamp(p.build_dir)
        if old is None:
            stale.append((p, ["<no build stamp found>"]))
        elif old != new:
            stale.append((p, _stamp_diff(old, new)))
        else:
            reusable.append(p)
    return reusable, to_build, stale


def _build_stage(
    plans: list[TilePlan],
    args: argparse.Namespace,
    do_configure: bool,
    do_make: bool,
    *,
    meta: UntuneMeta,
) -> tuple[int, list[TilePlan]]:
    """Configure and/or build every tile plan.

    `meta` (the untune CSV signature) determines the codegen
    `-DFMHA_FWD_GEN_FILTER` glob.

    Returns (exit_code, successfully_built_plans).
    """
    _warn_if_skip_downgraded(meta, "build")
    hipcc = _which_hipcc() if do_configure else None
    if do_configure and not hipcc:
        print(
            "[warn] hipcc not in PATH; CMAKE_{C,CXX}_COMPILER unset.", file=sys.stderr
        )

    extra_by_pair: dict[tuple[int, int], list[str]] = {}
    for p in plans:
        key = (p.pair.hdim_q, p.pair.hdim_v)
        if key not in extra_by_pair:
            extra_by_pair[key] = _extra_cmake_opts_for_pair(
                p.pair, meta, args.cmake_opt
            )

    ck_root = str(Path(args.ck_root).resolve())
    if do_configure and not os.path.isfile(os.path.join(ck_root, "CMakeLists.txt")):
        print(
            f"[error] --ck-root does not look like a CK source root "
            f"(missing CMakeLists.txt): {ck_root}",
            file=sys.stderr,
        )
        return 2, []

    workers = max(1, int(args.workers))
    concurrent = workers > 1 and (do_configure or do_make)

    ok_plans: list[TilePlan] = []
    configure_failures: list[str] = []
    build_failures: list[str] = []

    def _fold(res: dict[str, Any], src_plan: TilePlan) -> int | None:
        buf = res.get("log")
        if buf:
            print("\n".join(buf))
        if res["did_configure"] and not res["configure_ok"]:
            configure_failures.append(res["tile_name"])
            return 2 if args.stop_on_error else None
        if res["did_make"] and not res["build_ok"]:
            build_failures.append(res["tile_name"])
            return 1 if args.stop_on_error else None
        ok_plans.append(src_plan)
        return None

    if concurrent:
        print(f"# workers = {workers} (concurrent configure+build)")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _configure_and_build_one,
                    p,
                    hipcc,
                    ck_root,
                    extra_by_pair[(p.pair.hdim_q, p.pair.hdim_v)],
                    do_configure,
                    do_make,
                    args.build_target,
                    args.jobs,
                    not args.no_fresh,
                    args.dry_run,
                    True,
                ): p
                for p in plans
            }
            for fut in as_completed(futures):
                p = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(
                        f"[fatal] worker raised for {p.tile.name}: {e}", file=sys.stderr
                    )
                    configure_failures.append(p.tile.name)
                    if args.stop_on_error:
                        for f in futures:
                            f.cancel()
                        return 3, ok_plans
                    continue
                rc = _fold(res, p)
                if rc is not None:
                    for f in futures:
                        f.cancel()
                    return rc, ok_plans
    else:
        for p in plans:
            res = _configure_and_build_one(
                p,
                hipcc,
                ck_root,
                extra_by_pair[(p.pair.hdim_q, p.pair.hdim_v)],
                do_configure,
                do_make,
                args.build_target,
                args.jobs,
                not args.no_fresh,
                args.dry_run,
                False,
            )
            rc = _fold(res, p)
            if rc is not None:
                return rc, ok_plans

    print()
    print("# ==== build summary ====")
    print(f"# ok                  : {len(ok_plans)}")
    print(f"# configure failures  : {len(configure_failures)}")
    print(f"# build failures      : {len(build_failures)}")
    for n in configure_failures:
        print(f"#   [cfg-fail] {n}")
    for n in build_failures:
        print(f"#   [make-fail] {n}")
    return 0, ok_plans


def _load_pair_plans_from_disk(
    work_dir: Path,
    tune_hdim_q: int,
    tune_hdim_v: int,
    dtype: str,
    *,
    filters: dict[str, list[str]] | None = None,
    limit: int = 0,
) -> list[TilePlan]:
    """Reload tile candidates from `tile_candidates.json` under each pair.

    If `filters` is provided, every per-tile tune-config JSON is (re)written
    with the full filter block; otherwise the JSON files written by enum are
    reused as-is. `limit > 0` truncates the reloaded tile list to the first
    N entries per pair (matches `cmd_enum`'s --limit behavior).
    """
    pair_plans = _pair_plans(work_dir, tune_hdim_q, tune_hdim_v)
    all_tile_plans: list[TilePlan] = []
    for pp in pair_plans:
        if not pp.tiles_json.is_file():
            raise FileNotFoundError(
                f"expected tile candidates at {pp.tiles_json} "
                f"(run `mha_tune.py enum` first)"
            )
        _, hq, hv, tiles = _read_tiles_json(pp.tiles_json)
        # Fix pair hdim from json (should match).
        assert hq == pp.hdim_q and hv == pp.hdim_v, f"pair mismatch in {pp.tiles_json}"
        if limit > 0 and len(tiles) > limit:
            print(
                f"[note] --limit {limit} applied to build/bench: "
                f"keeping first {limit} of {len(tiles)} tiles from "
                f"{pp.tiles_json}."
            )
            tiles = tiles[:limit]
        all_tile_plans.extend(
            _tile_plans_for(
                pp,
                tiles,
                dtype,
                filters=filters,
                disable_check_hdim_tile=True,
            )
        )
    return all_tile_plans


def cmd_build(args: argparse.Namespace) -> int:
    """enum (idempotent) + configure + build."""
    meta, _ = parse_untune_csv(args.input_csv)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # Run enum first so build has fresh tile_candidates.json.
    rc = cmd_enum(args)
    if rc != 0:
        return rc

    plans = _load_pair_plans_from_disk(
        work_dir,
        args.tune_hdim_q,
        args.tune_hdim_v,
        meta.dtype,
        filters=_filters_from_meta(meta),
        limit=int(getattr(args, "limit", 0) or 0),
    )
    if not plans:
        print(
            "[error] no tile candidates enumerated; nothing to build.", file=sys.stderr
        )
        return 1
    rc, _ok = _build_stage(plans, args, do_configure=True, do_make=True, meta=meta)
    return rc


def cmd_bench(args: argparse.Namespace) -> int:
    """Benchmark stage. Auto-runs `enum` and/or `build` first if their
    products are missing, so `bench` can be used as a single entry point
    for the full pipeline while still short-circuiting to just-bench when
    everything is already on disk."""
    meta, max_seqlens = parse_untune_csv(args.input_csv)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # ---- Auto step 1: enum ------------------------------------------------
    # `_load_pair_plans_from_disk` raises FileNotFoundError when any pair's
    # tile_candidates.json is missing. In that case, run `enum` once and
    # retry the load; if the load still fails, propagate the error.
    try:
        plans = _load_pair_plans_from_disk(
            work_dir,
            args.tune_hdim_q,
            args.tune_hdim_v,
            meta.dtype,
            filters=_filters_from_meta(meta),
            limit=int(getattr(args, "limit", 0) or 0),
        )
    except FileNotFoundError as e:
        print(
            f"[bench] tile_candidates.json not found ({e}); " f"running `enum` first.",
            file=sys.stderr,
        )
        rc = cmd_enum(args)
        if rc != 0:
            return rc
        plans = _load_pair_plans_from_disk(
            work_dir,
            args.tune_hdim_q,
            args.tune_hdim_v,
            meta.dtype,
            filters=_filters_from_meta(meta),
            limit=int(getattr(args, "limit", 0) or 0),
        )
    if not plans:
        print("[error] no tile candidates enumerated.", file=sys.stderr)
        return 1

    # ---- Auto step 2: build ----------------------------------------------
    # Reuse a tile binary only when its build stamp matches this run's MHA
    # config (hdim / tile / tune-config filters / cmake args / target). A
    # binary with a missing or different stamp is never reused nor rebuilt
    # in place: bail out and let the user pick a new --work-dir or delete
    # the stale build dirs. Tiles without a binary are built normally.
    reusable, missing, stale = _classify_existing_builds(plans, meta, args)
    if stale:
        for p, diff in stale:
            print(
                f"[error] stale build dir {p.build_dir}: "
                f"differs in {', '.join(diff)}",
                file=sys.stderr,
            )
        print(
            f"[error] {len(stale)}/{len(plans)} tile build dir(s) were built "
            f"with a different MHA config (or before build stamps existed); "
            f"refusing to reuse them. Use a new --work-dir, or remove the "
            f"build_<tile> dir(s) listed above and rerun.",
            file=sys.stderr,
        )
        return 2
    if missing:
        print(
            f"[bench] {len(missing)}/{len(plans)} tile binaries missing; "
            f"running `build` for them first.",
            file=sys.stderr,
        )
        rc, ok_plans = _build_stage(
            missing, args, do_configure=True, do_make=True, meta=meta
        )
        if rc != 0:
            return rc
        keep = {id(p) for p in reusable} | {id(p) for p in ok_plans}
        plans = [p for p in plans if id(p) in keep]
        if not plans:
            print("[error] no tiles built successfully; cannot bench.", file=sys.stderr)
            return 1
    elif reusable:
        print(
            f"[bench] reusing {len(reusable)} tile binaries (build stamps match).",
            file=sys.stderr,
        )

    # ---- Step 3: bench ---------------------------------------------------
    return _bench_stage_and_dump(plans, meta, max_seqlens, args)


def cmd_run(args: argparse.Namespace) -> int:
    """Full pipeline: enum -> build -> bench -> dump tuned csv."""
    meta, max_seqlens = parse_untune_csv(args.input_csv)
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. enum
    rc = cmd_enum(args)
    if rc != 0:
        return rc

    # 2. build
    plans = _load_pair_plans_from_disk(
        work_dir,
        args.tune_hdim_q,
        args.tune_hdim_v,
        meta.dtype,
        filters=_filters_from_meta(meta),
        limit=int(getattr(args, "limit", 0) or 0),
    )
    if not plans:
        print("[error] no tile candidates enumerated.", file=sys.stderr)
        return 1
    rc, ok_plans = _build_stage(plans, args, do_configure=True, do_make=True, meta=meta)
    if rc != 0:
        return rc

    # 3. bench (only use successfully-built plans)
    if not ok_plans:
        print("[error] no tiles built successfully; cannot bench.", file=sys.stderr)
        return 1
    return _bench_stage_and_dump(ok_plans, meta, max_seqlens, args)


# ===========================================================================
# 7. Bench-stage driver + tuned CSV emitter
# ===========================================================================


def _bench_stage_and_dump(
    plans: list[TilePlan],
    meta: UntuneMeta,
    max_seqlens: list[int],
    args: argparse.Namespace,
) -> int:
    """For each max_seqlen row: sweep all plans, keep top-1, emit tuned csv.

    Bench must run serially (GPU is exclusive).
    """
    ck_root = str(Path(args.ck_root).resolve())
    try:
        # Validate mask / dropout once before touching any binary.
        bench_args_for_meta(
            meta,
            hdim_q_bench=args.tune_hdim_q,
            hdim_v_bench=args.tune_hdim_v,
            max_seqlen=1,
            warmup=args.warmup,
            repeat=args.repeat,
        )
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    mask_letter = MASK_TYPE_TO_LETTER[int(meta.mask_type)]
    variant = bench_variant(meta)
    _warn_if_skip_downgraded(meta, "bench")

    # Pre-collect the (binary, plan) pairs so we don't repeatedly stat().
    ready: list[tuple[str, TilePlan]] = []
    for p in plans:
        b = _binary_path(str(p.build_dir), args.build_target)
        if not os.path.isfile(b):
            print(f"[skip] binary not found for {p.tile.name}: {b}", file=sys.stderr)
            continue
        ready.append((b, p))

    if not ready:
        print("[error] no built binaries available for bench.", file=sys.stderr)
        return 1

    # Per-shape results (also archived to disk for debug).
    per_shape_dir = Path(args.work_dir).resolve() / "bench"
    per_shape_dir.mkdir(parents=True, exist_ok=True)

    tuned_rows: list[dict[str, Any]] = []

    for row_idx, M in enumerate(max_seqlens):
        bench_args_list = bench_args_for_meta(
            meta,
            hdim_q_bench=args.tune_hdim_q,
            hdim_v_bench=args.tune_hdim_v,
            max_seqlen=M,
            warmup=args.warmup,
            repeat=args.repeat,
        )
        print()
        print(
            f"# ==== bench row {row_idx}: max_seqlen={M} " f"(mask={mask_letter}) ===="
        )
        print(f"# args: {' '.join(shlex.quote(x) for x in bench_args_list)}")

        per_tile_results: list[dict[str, Any]] = []
        for binary, p in ready:
            print(f"[bench max_s={M}] tile={p.tile.name}")
            status, perf = _do_bench(
                binary=binary,
                bench_args=bench_args_list,
                ck_root=ck_root,
                dry_run=args.dry_run,
            )
            if status == "ok" and perf is not None:
                per_tile_results.append(
                    {
                        "tile_name": p.tile.name,
                        "pair_hdim_q": p.pair.hdim_q,
                        "pair_hdim_v": p.pair.hdim_v,
                        "tile_expr": p.tile.as_ck_expr(),
                        "tile_args": p.tile.as_args(),
                        "time_ms": perf["time_ms"],
                        "tflops": perf["tflops"],
                        "gbps": perf["gbps"],
                        "kname": perf["kname"],
                    }
                )

        if not per_tile_results:
            print(
                f"[warn] no successful bench for max_seqlen={M}; " f"skipping row.",
                file=sys.stderr,
            )
            tuned_rows.append(
                {
                    "max_seqlen": M,
                    **meta.signature(),
                    "bench_variant": variant,
                    "best_hdim_q": "",
                    "best_hdim_v": "",
                    "best_time_ms": "",
                    "best_tflops": "",
                    "best_gbps": "",
                    "best_kname": "",
                    "best_tile_name": "",
                    "best_tile_expr": "",
                    "status": "no_perf",
                }
            )
            continue

        per_tile_results.sort(key=lambda r: (-float(r["tflops"]), float(r["time_ms"])))
        best = per_tile_results[0]

        # Archive per-shape json for debug.
        per_shape_json = per_shape_dir / f"bench_s{M}.json"
        try:
            per_shape_json.write_text(
                json.dumps(
                    {
                        "max_seqlen": M,
                        "bench_args": bench_args_list,
                        "results": per_tile_results,
                    },
                    indent=2,
                )
            )
        except OSError as e:
            print(f"[warn] failed to write {per_shape_json}: {e}", file=sys.stderr)

        # Build tile_expr in the form `(hq, hv) : [FmhaFwdTileSize(...)]`.
        expr_full = (
            f"({best['pair_hdim_q']}, {best['pair_hdim_v']}) : "
            f"[{best['tile_expr']}]"
        )
        tuned_rows.append(
            {
                "max_seqlen": M,
                **meta.signature(),
                "bench_variant": variant,
                "best_hdim_q": best["pair_hdim_q"],
                "best_hdim_v": best["pair_hdim_v"],
                "best_time_ms": f"{best['time_ms']:.4f}",
                "best_tflops": f"{best['tflops']:.4f}",
                "best_gbps": f"{best['gbps']:.4f}",
                "best_kname": best["kname"],
                "best_tile_name": best["tile_name"],
                "best_tile_expr": expr_full,
                "status": "ok",
            }
        )

        print(
            f"  -> best: {best['tile_name']}   "
            f"time={best['time_ms']:.3f} ms  TFlops={best['tflops']:.2f}  "
            f"BW={best['gbps']:.2f} GB/s"
        )
        print(f"  -> tile_expr: {expr_full}")

    # ---- write tuned csv ----
    out_csv = tuned_csv_path(meta, Path(args.work_dir).resolve())
    fieldnames = list(TUNED_CSV_COLS)
    try:
        with out_csv.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            for row in tuned_rows:
                writer.writerow(row)
    except OSError as e:
        print(f"[error] failed to write tuned csv {out_csv}: {e}", file=sys.stderr)
        return 1

    print()
    print(f"[done] tuned csv written to: {out_csv}")
    print(f"[done] per-shape bench json under: {per_shape_dir}")
    return 0


# ===========================================================================
# 8. CLI
# ===========================================================================


def _add_common_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "-i",
        "--input-csv",
        type=Path,
        required=True,
        help="[all cmds] Path to mha_untune_<gid>_<signature>.csv produced "
        "by `mha_count_shape.py generate_tune_range`. The signature "
        "(mode, dtype, hdim, mask, nhead_q/k, bias/lse/dropout/skip/"
        "...) is read from the CSV columns and must match the "
        "filename; each row's max_seqlen drives one bench shape.",
    )
    sp.add_argument(
        "--work-dir",
        type=Path,
        required=True,
        help="[all cmds] Directory to hold per-pair `hq<H>_hv<HV>/` "
        "sub-directories (tune_configs/*.json, build_<tile>/ dirs, "
        "per-shape bench json) and the final mha_tuned_*.csv. Will "
        "be created if missing.",
    )
    sp.add_argument(
        "--ck-root",
        type=Path,
        required=True,
        help="[all cmds] Path to the composable_kernel source root (the "
        "directory containing the top-level CMakeLists.txt). Used "
        "as the source dir for cmake configure.",
    )
    sp.add_argument(
        "--tune-hdim-q",
        type=int,
        required=True,
        help="[all cmds] Q head dimension the built kernels will actually "
        "target (single integer, no list). Does NOT have to equal "
        "the hdim_q in the untune CSV: for CSV hdim_q=72 typical "
        "choices are 80 or 96 (CK's next 16-aligned tile-table rows). "
        "Used both as the tiles-dict key in the tune-config JSON "
        "and as `-DFMHA_FWD_GEN_OPTDIM=<val>` when building.",
    )
    sp.add_argument(
        "--tune-hdim-v",
        type=int,
        required=True,
        help="[all cmds] V head dimension the built kernels will actually "
        "target (single integer, no list). Same rules as "
        "--tune-hdim-q.",
    )
    sp.add_argument(
        "--allow-mfma-16",
        action="store_true",
        help="[all cmds] Extend the bf16 mfma enumeration set from the "
        "default [(32,32,16)] to also include "
        "[(16,16,16),(16,16,32)]. REQUIRED when tune-hdim-q or "
        "tune-hdim-v is not a multiple of 32 (e.g. 72, 80): with "
        "only (32,32,16) available, layer-1 divisibility filter "
        "rejects every candidate and the tile list becomes empty.",
    )
    sp.add_argument(
        "--occupancy",
        default=None,
        help="[all cmds] Comma-separated occupancy values to sweep in the "
        "tile enum, e.g. '1,2,3,4,5' (default) or '-1' for auto-only. "
        "Maps to FmhaFwdTileSize.F_occupancy.",
    )
    sp.add_argument(
        "--limit",
        type=int,
        default=0,
        help="[all cmds, DEBUG] If > 0, keep only the first N tile "
        "candidates during enum, and reload only the first N tiles "
        "in build/bench/run. Useful for smoke-testing the pipeline "
        "without waiting for a full sweep. 0 (default) = no limit.",
    )


def _add_build_args(sp: argparse.ArgumentParser) -> None:
    sp.add_argument(
        "--build-target",
        default=DEFAULT_BUILD_TARGET,
        help="[build/bench/run] Target passed to `cmake --build`, and "
        "the executable name looked up under each build_<tile>/ "
        "at bench time (default: %(default)s).",
    )
    sp.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=64,
        help="[build/run] Parallelism for `cmake --build -j` PER tile "
        "(default: %(default)s). Total in-flight jobs is roughly "
        "workers*jobs.",
    )
    sp.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="[build/run] Number of tiles to configure+build "
        "concurrently (default: %(default)s). Watch out for "
        "workers*jobs vs available CPU cores.",
    )
    sp.add_argument(
        "--cmake-opt",
        action="append",
        default=[],
        help="[build/run] Extra option appended to every cmake configure "
        "command (may be repeated). Example: "
        "--cmake-opt=-DCMAKE_HIP_ARCHITECTURES=gfx942.",
    )
    sp.add_argument(
        "--no-fresh",
        action="store_true",
        help="[build/run] Do NOT purge existing build dir before "
        "configure. Default wipes it because CK-tile fmha's "
        "fwd_blob_list.txt is only regenerated on the first "
        "configure and stale blobs would ignore the new "
        "CustomTuneFactory payload.",
    )
    sp.add_argument(
        "--stop-on-error",
        action="store_true",
        help="[build/bench/run] Abort the whole run if configure, build, "
        "or bench of any tile fails. Default is best-effort: skip "
        "the failing tile and keep going.",
    )
    sp.add_argument(
        "--dry-run",
        action="store_true",
        help="[build/bench/run] Print what would be executed (cmake / "
        "ninja / bench command lines plus resolved env vars) but "
        "do not actually run them. Useful for reviewing the "
        "pipeline before a long build.",
    )


def _add_bench_args(sp: argparse.ArgumentParser) -> None:
    # NOTE: there are intentionally no --nhead-q/--nhead-k/--bias/--lse/
    # --p-drop options: head counts and kernel traits are read from the
    # untune CSV (columns + filename signature) so they cannot drift from
    # the dumped workload. Passing them is an argparse error.
    sp.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="[build/bench/run] Warm-up iterations before timing. "
        "Passed to `-warmup=<n>` (default: %(default)s).",
    )
    sp.add_argument(
        "--repeat",
        type=int,
        default=50,
        help="[build/bench/run] Timed iterations used to compute "
        "averaged ms / TFlops / GB/s. Passed to `-repeat=<n>` "
        "(default: %(default)s).",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sp_enum = sub.add_parser(
        "enum",
        help="Only enumerate legal FmhaFwdTileSize candidates "
        "(no configure/build/bench).",
    )
    _add_common_args(sp_enum)
    sp_enum.set_defaults(func=cmd_enum)

    sp_build = sub.add_parser(
        "build",
        help="Enumerate + cmake configure + `cmake --build` every tile.",
    )
    _add_common_args(sp_build)
    _add_build_args(sp_build)
    _add_bench_args(sp_build)  # build shares the bench parser surface
    sp_build.set_defaults(func=cmd_build)

    sp_bench = sub.add_parser(
        "bench",
        help="Run the bench stage; automatically runs `enum` and/or "
        "`build` first when their products are missing on disk. "
        "When everything is already built, this is a zero-overhead "
        "bench-only run. Effectively a superset of `run` that can "
        "also short-circuit for iterative bench-only workflows.",
    )
    _add_common_args(sp_bench)
    _add_build_args(sp_bench)  # need --build-target etc. to locate binaries
    _add_bench_args(sp_bench)
    sp_bench.set_defaults(func=cmd_bench)

    sp_run = sub.add_parser(
        "run",
        help="Full pipeline: enum + build + bench -> mha_tuned_*.csv.",
    )
    _add_common_args(sp_run)
    _add_build_args(sp_run)
    _add_bench_args(sp_run)
    sp_run.set_defaults(func=cmd_run)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args) or 0)
    except FileNotFoundError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2
    except ValueError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
