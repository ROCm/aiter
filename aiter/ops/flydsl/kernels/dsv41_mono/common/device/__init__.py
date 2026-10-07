# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM, MIT License) at 97359d6:
# atom/mono/device/__init__.py
"""Device-side mechanisms every mono kernel shares, one module each: ``ops``
(wave / math / packing primitives) and ``sync`` (the tagged mailbox)."""
