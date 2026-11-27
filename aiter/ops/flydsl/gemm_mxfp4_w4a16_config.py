# SPDX-License-Identifier: MIT

"""Measured shape configs and a validated fallback for RDNA MXFP4 GEMM."""

import json
from dataclasses import dataclass, fields, replace
from functools import lru_cache
from pathlib import Path

SUPPORTED_ARCHS = ("gfx1151",)
_CONFIG_DIR = Path(__file__).with_name("configs")
_CONFIG_NAME = "GEMM-MXFP4-W4A16"
_TUNED_CU_COUNTS = {"gfx1151": 40}


@dataclass(frozen=True)
class MXFP4W4A16Config:
    group_tile_m: int = 16
    group_tile_n: int = 64
    stage_k: int = 64
    waves_per_group_m: int = 1
    waves_per_group_n: int = 2
    traversal_group_m: int = 1
    split_k: int = 4
    wave_groups_m: int = 1
    wave_groups_n: int = 1
    k_unroll: int = 8
    right_cache_modifier: int = 2
    left_cache_modifier: int = 0
    left_in_registers: bool = True
    right_in_registers: bool = False

    @property
    def wave_groups(self):
        return self.wave_groups_m * self.wave_groups_n

    @property
    def compute_threads(self):
        return 32 * self.waves_per_group_m * self.waves_per_group_n

    @property
    def threads(self):
        return self.compute_threads * self.wave_groups

    @property
    def lds_bytes(self):
        left_values = (
            0 if self.left_in_registers else self.group_tile_m * (self.stage_k + 8)
        )
        right_values = (
            0 if self.right_in_registers else self.stage_k * self.group_tile_n
        )
        return 2 * self.wave_groups * (left_values + right_values)


def validate_config(config: MXFP4W4A16Config, shape: tuple[int, int, int]):
    bm, bn, bk = config.group_tile_m, config.group_tile_n, config.stage_k
    if any(value <= 0 for value in (bm, bn, bk)):
        raise ValueError("Group tile dimensions and stage_k must be positive")
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("Shape must contain positive M,N,K")
    if config.split_k not in (1, 2, 4):
        raise ValueError("split_k must be 1, 2, or 4")
    if (
        config.waves_per_group_m < 1
        or config.waves_per_group_n < 1
        or config.threads not in (32, 64, 128, 256)
    ):
        raise ValueError("Use one, two, four, or eight wave32 waves per block")
    if (
        bm % (16 * config.waves_per_group_m)
        or bn % (16 * config.waves_per_group_n)
        or bk % 16
    ):
        raise ValueError(
            "Group tile and stage_k must contain whole WMMA tiles and K fragments"
        )
    if any(
        value not in (1, 2, 4) for value in (config.wave_groups_m, config.wave_groups_n)
    ):
        raise ValueError("Each wave-group dimension must be 1, 2, or 4")
    if config.wave_groups > 1 and config.compute_threads != 32:
        raise ValueError("Independent groups each contain exactly one wave32")
    if config.k_unroll not in (1, 2, 4, 8, 16):
        raise ValueError("k_unroll must be 1, 2, 4, 8, or 16")
    if config.right_cache_modifier not in range(
        8
    ) or config.left_cache_modifier not in range(8):
        raise ValueError("gfx11 cache modifiers must fit three bits")
    staging_tiles = []
    if not config.left_in_registers:
        staging_tiles.append((bm, bk))
    if not config.right_in_registers:
        staging_tiles.append((bk, bn))
    for rows, columns in staging_tiles:
        chunks = columns // 8
        if (
            columns % 8
            or chunks > config.compute_threads
            or config.compute_threads % chunks
        ):
            raise ValueError("Copy geometry must distribute aligned eight-BF16 chunks")
        if rows % (config.compute_threads // chunks):
            raise ValueError(
                "Every cooperative staging thread must own complete chunks"
            )
    if config.traversal_group_m < 1:
        raise ValueError("traversal_group_m must be positive")
    if config.lds_bytes > 65536:
        raise ValueError("LDS allocation exceeds the 64 KiB limit")
    return config


@lru_cache(maxsize=128)
def _read_tuned_configs(arch: str, n: int, k: int):
    path = (
        _CONFIG_DIR
        / arch
        / "gemm"
        / "gemm_mxfp4_w4a16"
        / f"{_CONFIG_NAME}-N={n}-K={k}.json"
    )
    try:
        with path.open(encoding="utf-8") as handle:
            rows = json.load(handle)
    except FileNotFoundError:
        return {}

    # The Qwen sweep measured exact M rows. Other M values use the heuristic.
    configs = {}
    schema = {field.name: field.type for field in fields(MXFP4W4A16Config)}
    for key, values in rows.items():
        prefix, sep, value = key.rpartition("_")
        if prefix != "M_EQ" or not sep or not value.isdecimal() or int(value) < 1:
            raise ValueError(f"{path}: invalid row key {key!r}")
        if (
            not isinstance(values, dict)
            or values.keys() != schema.keys()
            or any(type(values[name]) is not kind for name, kind in schema.items())
        ):
            raise ValueError(f"{path}: invalid config fields for {key}")
        m = int(value)
        configs[m] = validate_config(MXFP4W4A16Config(**values), (m, n, k))
    return configs


def _ceildiv(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _heuristic_config(m: int, n: int, k: int, cu_count: int) -> MXFP4W4A16Config:
    if m <= 16:
        tiles = (
            (16, 256, 1, 4),
            (16, 128, 1, 2),
            (16, 64, 1, 1),
            (16, 32, 1, 1),
            (16, 16, 1, 1),
        )
    elif m < 128 or n < 128:
        tiles = ((32, 128, 1, 4), (32, 64, 1, 2), (16, 32, 1, 1), (16, 16, 1, 1))
    else:
        tiles = (
            (128, 128, 2, 2),
            (64, 128, 2, 2),
            (32, 64, 1, 2),
            (16, 32, 1, 1),
            (16, 16, 1, 1),
        )

    for bm, bn, wm, wn in tiles:
        workgroups = _ceildiv(m, bm) * _ceildiv(n, bn)
        if bn <= n and workgroups >= cu_count:
            break

    stage_k = 16 if k <= 16 else 32
    k_stages = _ceildiv(k, stage_k)
    split_k = 1
    for splits in (2, 4):
        if workgroups * split_k >= cu_count or splits > k_stages:
            break
        split_k = splits
    return validate_config(
        replace(
            MXFP4W4A16Config(),
            group_tile_m=bm,
            group_tile_n=bn,
            stage_k=stage_k,
            waves_per_group_m=wm,
            waves_per_group_n=wn,
            split_k=split_k,
            k_unroll=2 if m < 128 else 4,
            left_in_registers=bm == 16,
        ),
        (m, n, k),
    )


@lru_cache(maxsize=256)
def select_mxfp4_w4a16_config(
    arch: str, m: int, n: int, k: int, cu_count: int
) -> MXFP4W4A16Config:
    """Pick an exact measured row or choose a legal tile for the live device."""
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Unsupported MXFP4 W4A16 target {arch!r}")
    if min(m, n, k, cu_count) < 1:
        raise ValueError("M, N, K and the compute-unit count must be positive")
    if cu_count == _TUNED_CU_COUNTS.get(arch):
        measured = _read_tuned_configs(arch, n, k).get(m)
        if measured is not None:
            return measured
    return _heuristic_config(m, n, k, cu_count)
