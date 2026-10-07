# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono layer kernels and the stages they are built from.

Each stage reproduces the rounding of the original kernel it replaces (the
reduction trees, the contractions, the quantization rules), as recorded in
each module; only the split of a long reduction may differ.
"""
