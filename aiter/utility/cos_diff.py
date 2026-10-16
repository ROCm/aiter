# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Row-wise companion to the MoE tuner's whole-tensor ``cos_diff``.

The tuner scores a candidate kernel with

    cos_diff = 1 - 2 * <x, y> / (<x, x> + <y, y>)

over the flattened tensor. That is a normalised squared error on one long
vector, so a defect confined to a few output rows is divided by the energy of
every other row. The dilution is exact and easy to state: if a kernel zeroes a
fraction ``f`` of the output rows and is correct everywhere else, it scores

    f / (2 - f)

which stays under the 0.1 acceptance threshold until ``f`` reaches 18%. In MoE
terms a kernel can silently drop the output of nearly one expert in six and
still be selected as the best candidate.

Scoring each row on its own and taking the worst removes the dilution: a zeroed
row scores 1.0 regardless of how many correct rows surround it.

Rows whose reference energy is below ``COS_DIFF_ROW_ENERGY_FLOOR`` relative to
the mean row energy are skipped. Padding rows in the sorted MoE layout are
all-zero by construction, and a normalised metric on an all-zero reference is
meaningless rather than wrong.
"""

import os

import torch

from aiter import logger

COS_DIFF_THRESHOLD = 1e-1

COS_DIFF_ROW_ENERGY_FLOOR = 1e-3

_ROWWISE_ENV = "AITER_MOE_COS_DIFF_ROWWISE"


def rowwise_enabled():
    """Whether to fold the worst-row score into the tuner's verdict.

    Opt-in, and read at call time rather than import time. Turning it on can
    only reject candidates the whole-tensor score would have accepted, so it
    changes which kernel the tuner picks.
    """
    return os.environ.get(_ROWWISE_ENV, "0") == "1"


def _whole(x, y):
    num = 2.0 * (x * y).sum().item()
    den = max((x * x + y * y).sum().item(), 1e-12)
    return 1.0 - num / den


def worst_row_cos_diff(x, y, energy_floor=COS_DIFF_ROW_ENERGY_FLOOR):
    """Largest per-row ``cos_diff`` over rows carrying real signal.

    ``x`` and ``y`` are 2-D float64 ``[rows, cols]``. Returns 0.0 when no row
    clears the energy floor.
    """
    ref_energy = (x * x).sum(dim=1)
    live = ref_energy > ref_energy.mean() * energy_floor
    if not bool(live.any()):
        return 0.0

    x, y = x[live], y[live]
    num = 2.0 * (x * y).sum(dim=1)
    den = torch.clamp((x * x).sum(dim=1) + (y * y).sum(dim=1), min=1e-12)
    return float((1.0 - num / den).max().item())


def combined_cos_diff(x_rows, y_rows, whole_tensor):
    """The tuner's verdict, tightened by the worst row when enabled.

    ``whole_tensor`` is the score the tuner already computed. Taking the max is
    monotone: the result is never smaller, so this can only reject a candidate,
    never accept one that was previously rejected.
    """
    if not rowwise_enabled():
        return whole_tensor
    if x_rows is None or y_rows is None or x_rows.shape != y_rows.shape:
        # Says so rather than quietly degrading to the whole-tensor score: a
        # packed layout compared against an unpacked reference lands here, and
        # silently falling back would look like row-wise scoring was applied.
        logger.warning(
            "cos_diff: row-wise scoring requested but skipped, shapes "
            f"{None if x_rows is None else tuple(x_rows.shape)} vs "
            f"{None if y_rows is None else tuple(y_rows.shape)}"
        )
        return whole_tensor
    return max(whole_tensor, worst_row_cos_diff(x_rows, y_rows))
