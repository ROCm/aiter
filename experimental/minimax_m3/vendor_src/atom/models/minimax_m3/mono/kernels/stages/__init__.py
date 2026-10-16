# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The fused layer kernel's stages (``post_attn``), a module per group.

Each ``<group>_defs(k4ctx)`` defines a group's functions over the kernel's names
``k4ctx`` (the builder's, the kernel body's so far and earlier groups'
functions), binding every name it reads at its top, and returns the functions;
the kernel body calls them in program order. The parameter is ``k4ctx`` and no
stage assigns that name: flydsl carries a name a dynamic loop assigns as loop
state when the enclosing function has it too."""
