# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Modifications Copyright (C) 2026 Advanced Micro Devices, Inc.

"""gfx1250 cluster MCAST mask helper, vendored into aiter.

``compute_mcast_masks`` moved from ``flydsl.expr.rocdl.cluster`` to flydsl's
repo-level ``kernels/common/``, which its wheel does not ship. This is pure
index math with no ROCDL primitives; the remaining cluster primitives
(``cluster_barrier`` / ``compute_cluster_position`` / ``cluster_wait``) still
live in flydsl and are imported from there.

Upstream: FlyDSL ``kernels/common/gfx1250_cluster.py`` @ ROCm/FlyDSL#880.
"""

from __future__ import annotations

from flydsl.expr import arith as _arith_ext
from flydsl.expr.meta import dsl_loc_tracing
from flydsl.expr.typing import T


@dsl_loc_tracing
def compute_mcast_masks(local_x, local_y, cluster_m: int, cluster_n: int):
    """Compute MCAST workgroup_mask values for A and B matrices.

    Hardware flat WG index within a cluster uses X-inner ordering
    (gfx1250 Shader Programming, TTMP6 layout, section 3.5.5.1):

        flat_wg_id = wg_x + wg_y * nwg_x = local_x + local_y * cluster_m

    where cluster_dims = (cluster_m, cluster_n, 1), so nwg_x = cluster_m.

    A mask: WGs sharing the same M-tile row (same local_x, varying local_y).
        Bits: {local_x + ly * cluster_m : ly in 0..cluster_n-1}
    B mask: WGs sharing the same N-tile column (same local_y, varying local_x).
        Bits: {lx + local_y * cluster_m : lx in 0..cluster_m-1}

    Args:
        local_x: WG row within cluster (MLIR index, 0..cluster_m-1).
        local_y: WG column within cluster (MLIR index, 0..cluster_n-1).
        cluster_m: Cluster rows (Python int).
        cluster_n: Cluster columns (Python int).

    Returns:
        (a_mask, b_mask) as MLIR i32 values for TDM workgroup_mask.
    """
    local_x_i32 = _arith_ext.index_cast(T.i32, local_x)
    local_y_i32 = _arith_ext.index_cast(T.i32, local_y)
    cluster_m_i32 = _arith_ext.constant(cluster_m, type=T.i32)

    # A mask: pattern has bits at strides of cluster_m, shifted by local_x.
    a_pattern_val = 0
    for ly in range(cluster_n):
        a_pattern_val |= 1 << (ly * cluster_m)
    a_pattern = _arith_ext.constant(a_pattern_val, type=T.i32)
    a_mask = _arith_ext.shli(a_pattern, local_x_i32)

    # B mask: cluster_m contiguous low bits, shifted by local_y * cluster_m.
    b_pattern = _arith_ext.constant((1 << cluster_m) - 1, type=T.i32)
    col_base = _arith_ext.muli(local_y_i32, cluster_m_i32)
    b_mask = _arith_ext.shli(b_pattern, col_base)

    return a_mask, b_mask


__all__ = ["compute_mcast_masks"]
