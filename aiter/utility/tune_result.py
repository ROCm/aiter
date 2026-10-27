# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tuner result that still unpacks as (info, us, err_ratio) and carries package watts."""

import math
import os

_POWER_ALIASES = {"power", "efficiency", "power_efficiency"}
_LATENCY_ALIASES = {"latency", "perf", "us", "performance"}


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
    """Add watts and tflops_per_w when a tuner result frame does not already have them."""
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
    out = df
    if "watts" not in out.columns:
        out = out.copy()
        out["watts"] = watts
    if "tflops_per_w" not in out.columns and "tflops" in out.columns:
        if out is df:
            out = out.copy()
        per_w = []
        for tflops, watts_f in zip(out["tflops"], out["watts"]):
            try:
                tflops_f = float(tflops)
                watts_v = float(watts_f)
            except (TypeError, ValueError):
                per_w.append(float("nan"))
                continue
            if (
                math.isnan(tflops_f)
                or math.isnan(watts_v)
                or tflops_f <= 0
                or watts_v <= 0
            ):
                per_w.append(float("nan"))
            else:
                per_w.append(round(tflops_f / watts_v, 6))
        out["tflops_per_w"] = per_w
    return out


def sort_frame_for_selection(df):
    """Order rows so the preferred kernel is first. Power mode uses TFLOP/s/W when present."""
    if (
        power_selection_enabled()
        and "tflops_per_w" in df.columns
        and df["tflops_per_w"].notna().any()
    ):
        return df.sort_values("tflops_per_w", ascending=False, kind="stable")
    if "us" in df.columns:
        return df.sort_values("us", kind="stable")
    return df
