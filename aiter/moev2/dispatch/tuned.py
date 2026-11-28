# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuned-table lookup.

`read_table` is the CSV reader the legacy `get_2stage_cfgs` also uses, so both see the
same rows; `lookup` reproduces the legacy `_lookup_cfg` with its fallbacks.

Every cache in v2 dispatch that depends on tuned tables, environment variables or the
arch is declared with `@cached` (the default-parameter helpers in `segments/defaults.py`
are cleared by name), so `clear_cache()` resets all of them at once; tests call it after
changing AITER_CONFIG_FMOE or patching the arch, and the CK tuner after it rewrites a
table.

NOTE: aiter.moev2 is under active development (MoE v2 migration, not graduated).
The legacy MoE path calls `read_table` and `index_columns` (factored out of its
`get_cfg_2stages`), `share_default_table` and `clear_cache` through `moev2_tuned`:
changes to them must keep legacy behavior
(`python tests/moe_dispatch/snapshot.py --check`). Please coordinate any change in this
module with the moev2 owners (@qli88). See aiter/moev2/README.md.
"""

import functools

from aiter import logger
from aiter.jit.utils.chip_info import gfx_from_cu_num
from aiter.ops.flydsl.mxfp4_kname import _is_mxfp4_kname

from ..shapes import _PADDED_M_TIERS

# Prefill-range rows only: a table row at a token count that is not a
# get_padded_M bucket becomes a lookup tier (`extra_m_tiers`) above this many
# tokens. Below it the existing off-bucket rows (decode-size rows of other
# models) stay unreachable, as before tiers existed.
EXTRA_TIER_MIN_TOKENS = 1024


@functools.cache
def extra_m_tiers(tune_file: str) -> tuple[int, ...]:
    """Token counts at which ``tune_file`` has tuned rows between two
    ``get_padded_M`` buckets, in the prefill range: the tiers ``tuned_M`` may key
    a lookup on. A tier serves only the shapes that have a row there; any other
    shape is looked up at its bucket (``get_2stage_cfgs``)."""
    import pandas as pd

    tokens = pd.read_csv(tune_file, usecols=["token"])["token"].dropna()
    return tuple(
        sorted(
            {
                int(t)
                for t in tokens
                if EXTRA_TIER_MIN_TOKENS < int(t) < _PADDED_M_TIERS[0]
                and int(t) & (int(t) - 1)
            }
        )
    )


_CACHES: list = []


def cached(fn=None, *, maxsize=None):
    """`functools.lru_cache` that `clear_cache()` also clears."""

    def wrap(f):
        wrapped = functools.lru_cache(maxsize=maxsize)(f)
        _CACHES.append(wrapped)
        return wrapped

    return wrap(fn) if fn is not None else wrap


def clear_cache():
    """Clear every v2 dispatch cache and re-read the moev2 environment variables."""
    from .. import envs
    from ..segments import defaults

    _LAST_READ.clear()
    for cache in _CACHES:
        cache.cache_clear()
    # Plain lru_caches: moved verbatim from the legacy code, which shares them.
    for helper in (defaults.get_block_size_M, defaults.use_nt, defaults.get_ksplit):
        helper.cache_clear()
    envs.reload()


_ACT_TYPE_DISABLED_KEY = "__ignore__"

INDEX_COLUMNS = (
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
)
# A dedicated table (`config_file=`, the FHMoE path) is also keyed on these.
DEDICATED_INDEX_COLUMNS = (
    "shared_expert_id",
    "hidden_pad",
    "intermediate_pad",
    "gate_mode",
)


def index_columns(dedicated: bool) -> list:
    return list(INDEX_COLUMNS + (DEDICATED_INDEX_COLUMNS if dedicated else ()))


def ensure_gfx_column(df):
    """Guarantee a usable `gfx` column, migrating legacy cu_num-only CSVs."""
    if "gfx" not in df.columns:
        df = df.copy()
        df["gfx"] = df["cu_num"].map(gfx_from_cu_num)
        return df
    # Backfill placeholder/missing gfx (e.g. 0 filled when merging a config
    # that lacks the column against ones that have it).
    bad = df["gfx"].isna() | df["gfx"].astype(str).isin(["0", "", "nan", "None"])
    if bad.any():
        df = df.copy()
        df.loc[bad, "gfx"] = df.loc[bad, "cu_num"].map(gfx_from_cu_num)
    return df


# The tables most recently read per (file, dedicated), shared with the legacy reader so
# a process reads (and warns about) each table once whichever side loads it first.
_LAST_READ: dict = {}


def read_table(tune_file, _INDEX_COLS, dedicated, tag=None):
    """Read one tuned CSV into `(primary, fallback)` dicts keyed on `_INDEX_COLS`.

    `fallback` ignores `act_type` and is empty for a dedicated table. A `tag` keeps
    only the rows with that `_tag` (rows tuned with a swiglu_limit); such a read is
    not shared through `_LAST_READ`.
    """
    import pandas as pd

    df = pd.read_csv(tune_file)
    df = ensure_gfx_column(df)
    if tag is not None:
        tags = df["_tag"] if "_tag" in df.columns else pd.Series("", index=df.index)
        df = df[tags.fillna("").astype(str) == tag]
    elif "_tag" in df.columns:
        df = df[df["_tag"].fillna("") != "flydsl_fallback"]

    # Primary dict: keep original act_type for exact-match lookup.
    df_primary = df.copy()
    dup_mask = df_primary.duplicated(subset=_INDEX_COLS, keep="first")
    if dup_mask.any():
        if dedicated:
            raise ValueError(f"Duplicate dedicated FHMoE rows in {tune_file}")
        logger.warning(
            f"[fused_moe] duplicate tuned rows (primary) in {tune_file}; "
            f"keeping first match for {int(dup_mask.sum())} rows"
        )
        df_primary = df_primary.loc[~dup_mask]
    primary = df_primary.set_index(_INDEX_COLS).to_dict("index")

    if dedicated:
        result = (primary, {})
        if tag is None:
            _LAST_READ[(tune_file, dedicated)] = result
        return result

    # Fallback dict: disable act_type so any activation can match.
    df_fallback = df.copy()
    if "kernelName1" in df_fallback.columns:
        is_ck2stages = (
            df_fallback["kernelName1"]
            .fillna("")
            .astype(str)
            .str.contains("moe_ck2stages")
        )
        df_fallback = df_fallback.loc[~is_ck2stages]
    for column in ("kernelName1", "kernelName2"):
        if column in df_fallback.columns:
            is_mxfp4 = df_fallback[column].map(_is_mxfp4_kname)
            df_fallback = df_fallback.loc[~is_mxfp4]
    if "act_type" in df_fallback.columns:
        df_fallback["act_type"] = _ACT_TYPE_DISABLED_KEY
    dup_mask = df_fallback.duplicated(subset=_INDEX_COLS, keep="first")
    if dup_mask.any():
        logger.warning(
            f"[fused_moe] duplicate tuned rows after disabling act_type in {tune_file}; "
            f"keeping first match for {int(dup_mask.sum())} rows"
        )
        df_fallback = df_fallback.loc[~dup_mask]
    fallback = df_fallback.set_index(_INDEX_COLS).to_dict("index")

    result = (primary, fallback)
    if tag is None:
        _LAST_READ[(tune_file, dedicated)] = result
    return result


def default_table_file() -> str:
    from aiter.jit.core import AITER_CONFIGS

    return AITER_CONFIGS.AITER_CONFIG_FMOE_FILE


_shared_default_table = None


def share_default_table(source):
    """Read the default table through `source()` instead of from the CSV.

    aiter.fused_moe passes its `cfg_2stages` loader, so tests and tuners that assign or
    reset that global change what v2 reads too.
    """
    global _shared_default_table
    _shared_default_table = source


def _default_table():
    if _shared_default_table is not None:
        return _shared_default_table()
    return _own_default_table()


@cached
def _own_default_table():
    # Read once per process, like the legacy `cfg_2stages` global.
    tune_file = default_table_file()
    return _LAST_READ.get((tune_file, False)) or read_table(
        tune_file, index_columns(False), False
    )


@cached
def _dedicated_table(tune_file):
    return _LAST_READ.get((tune_file, True)) or read_table(
        tune_file, index_columns(True), True
    )


def table_for(problem):
    tune_file = problem.extras.get("tuned_table")
    return _default_table() if tune_file is None else _dedicated_table(tune_file)


def lookup_keys(problem):
    """`(keys, keys_disabled)` the legacy lookup uses; `keys` is also the log key."""
    keys = problem.log_keys()
    keys_disabled = keys[:7] + (_ACT_TYPE_DISABLED_KEY,) + keys[8:]
    return keys, keys_disabled


def lookup(problem, tables=None):
    """The tuned row for `problem` (the legacy `_lookup_cfg`), or None.

    Exact key first, then the activation-agnostic table (default table only), then, for a
    token count above the smallest tier, the smaller tiers in descending order.
    """
    c2s = table_for(problem) if tables is None else tables
    if not c2s:
        return None
    keys, keys_disabled = lookup_keys(problem)
    dedicated = problem.extras.get("tuned_table") is not None
    token = problem.shape.m
    primary, fallback = c2s
    lookup_keys_ = keys[:7] + (str(problem.activation.activation),) + keys[8:]
    result = primary.get(lookup_keys_, None)
    if result is None and not dedicated:
        result = fallback.get(keys_disabled, None)
    # Tier fallback: if current tier not found, try smaller tiers in descending order
    if result is None and not dedicated and token > _PADDED_M_TIERS[0]:
        tier_idx = _PADDED_M_TIERS.index(token) if token in _PADDED_M_TIERS else -1
        for fallback_tier in reversed(_PADDED_M_TIERS[:tier_idx]):
            # keys layout: (gfx, cu_num, token, ...); replace token (idx 2).
            keys_fb = lookup_keys_[:2] + (fallback_tier,) + lookup_keys_[3:]
            keys_fb_disabled = keys_disabled[:2] + (fallback_tier,) + keys_disabled[3:]
            result = primary.get(keys_fb, None)
            if result is None:
                result = fallback.get(keys_fb_disabled, None)
            if result is not None:
                break
    return result
