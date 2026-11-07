# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Variant registry: variant id <-> contract <-> source digest <-> Plan class."""

from __future__ import annotations

from dataclasses import dataclass

_CATALOG: dict[str, VariantInfo] = {}


@dataclass(frozen=True)
class Contract:
    arch: tuple[str, ...]  # ("gfx950",)
    world: int  # 4
    topology: str  # "single_node_xgmi_p2p"
    graph: str  # "capture_safe_epoch_state"
    requires: tuple[str, ...] = ()  # ("mori_cco", "sdma", "gfx950_sc_bypass")
    m_multiple: int = 1  # local M divisibility assumed by the kernel
    mutates: tuple[str, ...] = ()  # caches written by the variant


@dataclass(frozen=True)
class VariantInfo:
    vid: str
    family: str
    stage_kinds: tuple[str, ...]  # the stage prefix this variant covers
    contract: Contract
    source_sha: str
    factory: type  # Plan class: factory(spec, params, ctx)


def register_variant(*, vid, family, stage_kinds, contract, source_sha):
    def deco(cls):
        if vid in _CATALOG:
            raise ValueError(f"duplicate gemm_fusion variant id {vid!r}")
        _CATALOG[vid] = VariantInfo(
            vid, family, tuple(stage_kinds), contract, source_sha, cls
        )
        return cls

    return deco


def get_variant(vid: str) -> VariantInfo:
    return _CATALOG[vid]
