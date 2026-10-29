# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuner result that still unpacks as (info, us, err_ratio) and carries package watts."""

import math
import os

_POWER_ALIASES = {"power", "efficiency", "power_efficiency"}
_LATENCY_ALIASES = {"latency", "perf", "us", "performance"}

# Written on every row of a power tune. A row without ``select`` is a latency
# row; the other columns record the power-management state the watts came from.
SELECT_COLUMN = "select"
POWER_CONDITION_COLUMNS = ("perf_level", "power_cap_w", "sclk_max_mhz")
POWER_ROW_COLUMNS = (SELECT_COLUMN,) + POWER_CONDITION_COLUMNS
UNKNOWN_CONDITION = "unknown"
MIXED_CONDITION = "mixed"
# Power-tuned tables live here so the model_configs/ shape merge never ranks a
# power row against a latency row.
POWER_CONFIG_DIRNAME = "power_configs"


def normalize_select(value) -> str:
    raw = str(value if value is not None else "latency").strip().lower()
    if raw in _POWER_ALIASES:
        return "power"
    if raw in _LATENCY_ALIASES or raw == "":
        return "latency"
    return raw


def power_selection_enabled(explicit=None) -> bool:
    raw = (
        explicit if explicit is not None else os.environ.get("AITER_SELECT", "latency")
    )
    return normalize_select(raw) == "power"


def row_select_mode(value) -> str:
    """``power`` for a power-tuned row. Blank, NaN, or a merge fill of 0 is latency."""
    if value is None:
        return "latency"
    try:
        if isinstance(value, float) and math.isnan(value):
            return "latency"
    except TypeError:
        pass
    text = str(value).strip()
    if text in ("", "0", "nan", "None"):
        return "latency"
    return normalize_select(text)


def efficiency_cost(us, watts) -> float:
    """Lower is better. For a fixed shape, TFLOP/s/W is proportional to 1/(us * watts)."""
    try:
        us_f = float(us)
        watts_f = float(watts)
    except (TypeError, ValueError):
        return math.inf
    if us_f <= 0 or math.isinf(us_f) or math.isnan(watts_f) or watts_f <= 0:
        return math.inf
    return us_f * watts_f


def stage_energy(us, watts) -> float:
    """Energy of one stage. A stage that did not run (us <= 0) costs 0.

    Unmeasured watts on a stage that did run are NaN, so the pair sorts last.
    """
    try:
        us_f = float(us)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(us_f):
        return float("nan")
    if us_f <= 0:
        return 0.0
    cost = efficiency_cost(us_f, watts)
    if math.isinf(cost):
        return float("nan")
    return cost


def measure_all_accurate(select_power: bool, shortlist: int, qualified: int) -> int:
    """How many accuracy-qualified candidates the accurate pass should time.

    Latency mode keeps the shortlist. Power mode times every qualified
    candidate, because a cooler kernel can sit outside the latency window.
    An empty qualified set leaves the shortlist alone.
    """
    if select_power and qualified > 0:
        return qualified
    return shortlist


def candidate_beats(current, challenger, select_power: bool) -> bool:
    """True when challenger replaces current. Equal rank keeps current."""
    if current is None:
        return True

    def _field(row, name):
        if isinstance(row, dict):
            return row.get(name, float("nan"))
        return getattr(row, name, float("nan"))

    current_rank = selection_rank(
        _field(current, "us"), _field(current, "watts"), select_power
    )
    challenger_rank = selection_rank(
        _field(challenger, "us"), _field(challenger, "watts"), select_power
    )
    return challenger_rank < current_rank


def sort_by_energy_then_latency(frame, us_col="us", watts_col="watts"):
    """Order by package energy, then by latency.

    Unmeasured watts sort as infinite cost, so they stay behind any measured
    row and, among themselves, keep the lowest ``us``. Dedup must run after
    this sort: a cost-only sort leaves every NaN row tied and a stable order
    then keeps whichever candidate arrived first.
    """
    cost = [
        efficiency_cost(us, watts)
        for us, watts in zip(frame[us_col], frame[watts_col], strict=True)
    ]
    return (
        frame.assign(_energy_cost=cost)
        .sort_values(["_energy_cost", us_col], kind="mergesort")
        .drop(columns="_energy_cost")
    )


def selection_rank(us, watts, select_power: bool):
    """Sort key. Latency mode is lowest us. Power mode is us*watts, with unmeasured rows last by us."""
    try:
        us_f = float(us)
    except (TypeError, ValueError):
        us_f = math.inf
    if not select_power:
        return (0, us_f)
    cost = efficiency_cost(us_f, watts)
    if math.isinf(cost):
        return (1, us_f)
    return (0, cost)


def power_columns(result, tflops):
    """Return (watts, tflops_per_w) for a tune result. NaN when power was not measured."""
    watts = getattr(result, "watts", float("nan"))
    try:
        watts_f = float(watts)
    except (TypeError, ValueError):
        watts_f = float("nan")
    try:
        tflops_f = float(tflops)
    except (TypeError, ValueError):
        tflops_f = float("nan")
    if math.isnan(watts_f) or watts_f <= 0 or math.isnan(tflops_f) or tflops_f <= 0:
        return float("nan"), float("nan")
    return round(watts_f, 3), round(tflops_f / watts_f, 6)


class TuneResult(tuple):
    """3-tuple plus package power in watts, so existing unpacking keeps working."""

    def __new__(cls, info, us, err_ratio, watts=float("nan")):
        obj = tuple.__new__(cls, (info, us, err_ratio))
        try:
            obj.watts = float(watts)
        except (TypeError, ValueError):
            obj.watts = float("nan")
        return obj

    def __reduce__(self):
        return (self.__class__, (self[0], self[1], self[2], self.watts))


def attach_power_columns(df, results):
    """Add watts on a power tune. Latency frames stay unchanged.

    ``tflops_per_w`` is added later by ``TunerCommon.power_metrics``, so a
    tuner with its own formula (MoE stage energy) is not overwritten here.
    """
    if not power_selection_enabled():
        return df
    if df is None or not hasattr(df, "columns"):
        return df
    if len(df) != len(list(results)):
        return df
    watts = []
    for el in results:
        raw = getattr(el, "watts", float("nan"))
        try:
            watts_f = float(raw)
        except (TypeError, ValueError):
            watts_f = float("nan")
        if math.isnan(watts_f) or watts_f <= 0:
            watts.append(float("nan"))
        else:
            watts.append(round(watts_f, 3))
    if "watts" not in df.columns:
        df = df.copy()
        df["watts"] = watts
    return df


def sort_frame_for_selection(df, select_power=None):
    """Order rows so the preferred kernel is first.

    Power mode uses ``us * watts`` (the same key as ``selection_rank``). A
    frame that recorded ``tflops_per_w`` and not watts falls back to that
    column. Latency mode, and a power frame with neither column, uses ``us``.
    """
    if select_power is None:
        select_power = power_selection_enabled()
    if select_power and "watts" in df.columns and df["watts"].notna().any():
        return sort_by_energy_then_latency(df)
    if (
        select_power
        and "tflops_per_w" in df.columns
        and df["tflops_per_w"].notna().any()
    ):
        return df.sort_values("tflops_per_w", ascending=False, kind="mergesort")
    if "us" in df.columns:
        return df.sort_values("us", kind="mergesort")
    return df
