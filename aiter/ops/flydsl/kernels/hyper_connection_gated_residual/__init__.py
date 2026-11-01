# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Two-stage Gated-Residual (Hyper-Connections) ``combine_and_mix``.

The public fused two-stage kernel and its shared numerical reference.
``K1`` fuses combine + grouped-RMSNorm + down; ``K2`` fuses silu + up + gated
mean and re-forms ``xn`` from ``r2`` on the fly, so ``xn``/``gate`` never touch
HBM. The operator runs as a K1+K2 pair, with an additional K1 reduction when
required by its split-K or decoupled schedule.
"""

from .k1 import flydsl_k1_combine_norm_down
from .k2 import flydsl_up_gate_mix_norm
from .op import (
    flydsl_gr_two_stage_combine,
    flydsl_gr_two_stage_combine_and_mix,
    flydsl_gr_two_stage_mix,
    fold_norm_weight,
    merge_gr_two_stage_weight,
)
from .reference import (
    gr_combine,
    gr_combine_and_mix,
    gr_grouped_rmsnorm,
    gr_mix,
    gr_mix_body,
)

__all__ = [
    "flydsl_gr_two_stage_combine",
    "flydsl_gr_two_stage_combine_and_mix",
    "flydsl_gr_two_stage_mix",
    "flydsl_k1_combine_norm_down",
    "flydsl_up_gate_mix_norm",
    "fold_norm_weight",
    "gr_combine",
    "gr_combine_and_mix",
    "gr_grouped_rmsnorm",
    "gr_mix",
    "gr_mix_body",
    "merge_gr_two_stage_weight",
]
