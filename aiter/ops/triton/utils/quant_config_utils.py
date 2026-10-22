# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Config loader for the Gluon quant kernels.

Reads ``configs/<arch>/gluon/quant/<config_name>/DEFAULT.json``.
"""

import functools

from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)


@functools.cache
def _axes_from_keys(keys: tuple) -> tuple:
    orders = {
        tuple(part.split("_")[0] for part in key.split("."))
        for key in keys
        if "." in key
    }
    if len(orders) != 1:
        raise ValueError(
            f"bucket keys must share one axis order, found {sorted(orders)}"
        )
    return orders.pop()


def table_axes(table: dict) -> tuple:
    """Axis order of the composite bucket keys: ``N_LEQ_32.M_LEQ_1`` gives
    ``("N", "M")``. The leftmost axis wins ties, so the table sets the order."""
    return _axes_from_keys(tuple(table))


def get_quant_config(config_name: str, **values) -> dict:
    """Launch config of a Gluon quant kernel for the running arch.

    ``config_name`` is e.g. ``"MXFP4"``. ``values`` give one value per axis of
    the table (see ``table_axes``). The result is a fresh dict.
    """
    cfg_dir = resolve_config_dir("quant", config_name, backend="gluon")
    tuned = load_config_json(f"{cfg_dir}/DEFAULT.json")
    return select_leq_config(tuned, axes=table_axes(tuned), **values)
