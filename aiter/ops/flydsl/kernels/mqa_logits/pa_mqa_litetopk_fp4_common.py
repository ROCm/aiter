# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import flydsl.expr as fx
from flydsl.expr import Array, Int32

FP4_LITETOPK_SUPPORTED_TOPKS = (512, 1024)


@fx.struct
class LiteTopKScanStorage:
    histogram: Array[Int32, 256, 16]
    scan: Array[Int32, 9, 16]
    state: Array[Int32, 3, 4]


@fx.struct
class LiteTopKSeedStorage:
    scores: Array[fx.Float32, 8192, 16]
    histogram: Array[Int32, 256, 16]
    scan: Array[Int32, 9, 16]
    maxima: Array[fx.Float32, 8, 16]
    neg_minima: Array[fx.Float32, 8, 16]
    finite_counts: Array[Int32, 8, 16]
    nonfinite_counts: Array[Int32, 8, 16]
    calibration: Array[fx.Float32, 2, 16]
    state: Array[Int32, 2, 4]
