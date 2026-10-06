# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Config loader for the Gluon quant kernels.

Reads ``configs/<arch>/gluon/quant/<config_name>/DEFAULT.json``.
"""

from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)


def get_quant_config(config_name: str, axes: tuple, **values) -> dict:
    """Launch config of a Gluon quant kernel for the running arch.

    ``config_name`` is e.g. ``"MXFP4"``. ``axes`` and ``values`` pick the bucket,
    as in ``select_leq_config``. The result is a fresh dict.
    """
    cfg_dir = resolve_config_dir("quant", config_name, backend="gluon")
    tuned = load_config_json(f"{cfg_dir}/DEFAULT.json")
    return select_leq_config(tuned, axes=axes, **values)
