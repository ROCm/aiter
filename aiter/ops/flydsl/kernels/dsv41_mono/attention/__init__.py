# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The mono layer's attention stages: ``front`` (K1: wqkv, q / kv norms and the
KV insert, wq_b) and ``back`` (K2: split attention, combine, wo_a, wo_b)."""
