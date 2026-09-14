# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Per-expert payload handshake for TP incremental activation all-gather.

Step 2 host-prefill: ``payload_ready[e] == expected[e]`` before launch.
Later steps increment ``payload_ready`` from producers.
"""

from __future__ import annotations

import flydsl.expr as fx

from .. import communication_ops_utils as comm_ops
from .gemm_util import _buffer_load


def wait_expert_payload(ready_base_i64, expected_rsrc, expert_i32):
    """Spin until ``payload_ready[expert] == expected[expert]`` (agent scope)."""
    want = _buffer_load(expected_rsrc, expert_i32, fx.Int32)
    addr = fx.Int64(ready_base_i64) + fx.Int64(expert_i32) * fx.Int64(4)
    comm_ops.spin_until_eq_i32(addr, want)
