# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The V4.1 mono kernels' deployment constants (ATOM's ``config``, less its
engine's deployment checks: vLLM's integration decides what it routes)."""

# a DSpark verify: every request's anchor token and its 5 drafted ones
SPEC_TOKENS = 5
# the requests a step serves at most
MAX_REQUESTS = 8
# the target's rows a step: the buffers' width, eight verifies
MAX_ROWS = MAX_REQUESTS * (SPEC_TOKENS + 1)
# a token's keys at most: 64 splits of one 16-key tile each
ATTENTION_KEYS_MAX = 64 * 16
