# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from ROCm/ATOM (https://github.com/ROCm/ATOM, MIT License) at 97359d6:
# atom/mono/__init__.py
"""The mono kernels' shared mechanisms: ``runtime`` (host: TP agreement, peer
memory, kernel argument tables), ``plan`` (plain Python: the execution model,
build keys, layouts, TP shards) and ``device`` (traced inside kernels: the
tagged mailbox, wave / math primitives)."""
