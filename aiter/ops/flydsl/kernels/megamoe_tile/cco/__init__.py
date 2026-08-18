# SPDX-License-Identifier: MIT
"""AITER-local CCO GDA FlyDSL bridge used by hierarchical MegaMoE."""

from .host import (
    TransportResources,
    clear_hip_last_error,
    create_transport_resources,
    fill_window_bytes,
    read_window_u32,
    read_window_u64,
    read_window_bytes,
    write_window_u32,
    write_window_u64,
    zero_window,
)
from .ops import (
    TEAM_RAIL,
    TEAM_WORLD,
    flush_async,
    lsa_ptr,
    put,
    put_value,
    wait_ready,
    wait_request,
)
from .smoke import TransportSmoke, build_transport_smoke
from .stage1_sidecar import (
    CcoStage1Sidecar,
    Stage1SidecarModule,
    build_stage1_sidecar_module,
)
from .stage2_sidecar import (
    DEFAULT_PARTIAL_RECORD_BYTES,
    CcoStage2ReturnSidecar,
    Stage2SidecarModule,
    build_stage2_sidecar_module,
)

__all__ = [
    "TransportResources",
    "TransportSmoke",
    "CcoStage1Sidecar",
    "Stage1SidecarModule",
    "CcoStage2ReturnSidecar",
    "DEFAULT_PARTIAL_RECORD_BYTES",
    "Stage2SidecarModule",
    "TEAM_RAIL",
    "TEAM_WORLD",
    "create_transport_resources",
    "fill_window_bytes",
    "clear_hip_last_error",
    "read_window_u32",
    "read_window_u64",
    "read_window_bytes",
    "write_window_u32",
    "write_window_u64",
    "zero_window",
    "build_transport_smoke",
    "build_stage1_sidecar_module",
    "build_stage2_sidecar_module",
    "flush_async",
    "lsa_ptr",
    "put",
    "put_value",
    "wait_ready",
    "wait_request",
]
