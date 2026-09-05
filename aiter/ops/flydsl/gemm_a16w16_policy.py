# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shape-aware policy selection for the FlyDSL A16W16 GEMM."""

import itertools
from dataclasses import dataclass

import torch

from aiter.jit.utils.chip_info import get_gfx

from .kernels.gemm_a16w16_gfx950 import (
    GEMM_A16W16_DTYPE_BF16,
    GEMM_A16W16_DTYPE_FP16,
    GEMM_A16W16_DTYPE_FP32,
    make_gemm_a16w16_param_and_validate,
)

__all__ = ["GemmConfigPruner", "get_flydsl_a16w16_configs"]


@dataclass(frozen=True)
class GemmConfigPruner:
    """Prune A16W16 policies using tile efficiency and estimated occupancy."""

    m: int
    n: int
    k: int
    device_props: object
    element_bytes: int
    target_waves_per_cu: int = 8
    max_split_grid_rounds: int = 2
    max_tile_grid_ratio: int = 4

    @staticmethod
    def _ceil_div(value, divisor):
        return (value + divisor - 1) // divisor

    def _tiles(self, config):
        return self._ceil_div(self.m, config["block_m"]) * self._ceil_div(
            self.n, config["block_n"]
        )

    def _iou(self, config):
        split_k = config["split_k"]
        padded_m = self._ceil_div(self.m, config["block_m"]) * config["block_m"]
        padded_n = self._ceil_div(self.n, config["block_n"]) * config["block_n"]
        part_k = self.k // split_k
        padded_k = (
            self._ceil_div(part_k, config["block_k"]) * config["block_k"] * split_k
        )
        return self.m * self.n * self.k / (padded_m * padded_n * padded_k)

    def _occupancy(self, config):
        props = self.device_props
        waves = config["m_waves"] * config["n_waves"] * config["k_waves"]
        lds = max(
            config["stages"]
            * (config["block_m"] + config["block_n"])
            * config["block_k"]
            * self.element_bytes,
            config["k_waves"]
            * config["block_m"]
            * config["block_n"]
            * self.element_bytes,
        )
        lds_per_cu = getattr(
            props,
            "shared_memory_per_multiprocessor",
            props.shared_memory_per_block,
        )
        resident = (
            min(
                props.max_threads_per_multi_processor // props.warp_size // waves,
                lds_per_cu // lds,
            )
            * waves
        )
        grid = (
            self._tiles(config)
            * config["split_k"]
            * waves
            / props.multi_processor_count
        )
        return min(self.target_waves_per_cu, resident, grid)

    def prune(self, configs):
        if not configs:
            return configs
        ious = [self._iou(config) for config in configs]
        keep_ratio = max(0.625, 1 - self.m / 160, 1 - 120 / self.m)
        min_iou = max(ious) * keep_ratio
        num_cus = self.device_props.multi_processor_count
        max_tiles = (
            max(num_cus, min(self._tiles(config) for config in configs))
            * self.max_tile_grid_ratio
        )
        kept = []
        for config, iou in zip(configs, ious):
            tiles = self._tiles(config)
            max_split_k = (
                1
                if tiles >= num_cus
                else self._ceil_div(self.max_split_grid_rounds * num_cus, tiles)
            )
            if (
                iou >= min_iou
                and tiles <= max_tiles
                and config["split_k"] <= max_split_k
                and (config["group_m"] == 0 or (tiles >= num_cus and tiles % 8 == 0))
            ):
                kept.append(config)

        best = {}
        keep = set()
        for index, config in sorted(
            enumerate(kept), key=lambda item: (item[1]["k_waves"], item[0])
        ):
            key = tuple(
                (name, value) for name, value in config.items() if name != "k_waves"
            )
            occupancy = self._occupancy(config)
            if occupancy > best.get(key, -1.0) + 1e-9:
                best[key] = occupancy
                keep.add(index)
        return [config for index, config in enumerate(kept) if index in keep]


def get_flydsl_a16w16_configs(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    out_dtype: torch.dtype,
    has_bias: bool,
):
    """Generate and validate the shape-aware policy catalog used by tuning."""

    if make_gemm_a16w16_param_and_validate is None:
        return []
    if get_gfx() != "gfx950":
        return []
    if dtype not in (torch.float16, torch.bfloat16):
        return []
    if out_dtype not in (dtype, torch.float32):
        return []

    split_k_candidates = [1]
    split_k_candidates.extend(split_k for split_k in range(2, 10) if k % split_k == 0)
    selections = {
        "block_m": [16, 32, 48, 64, 80, 96, 128, 256],
        "block_n": [16, 32, 64, 80, 96, 128, 256],
        "block_k": [64, 128, 256],
        "stages": list(range(2, 10)),
        "split_k": split_k_candidates,
        "m_waves": [1, 2, 4],
        "n_waves": [1, 2, 4],
        "k_waves": [1, 2],
        "group_m": [0, 4],
        "use_half_tile_interleaved": [False, True],
    }
    configs = [
        dict(zip(selections, combo))
        for combo in itertools.product(*selections.values())
    ]
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    configs = GemmConfigPruner(
        m,
        n,
        k,
        device_props,
        2,
    ).prune(configs)

    in_dtype_id = (
        GEMM_A16W16_DTYPE_FP16 if dtype == torch.float16 else GEMM_A16W16_DTYPE_BF16
    )
    out_dtype_id = GEMM_A16W16_DTYPE_FP32 if out_dtype == torch.float32 else in_dtype_id
    valid_configs = []
    is_large_gemm = m >= 4096 and n >= 4096 and k >= 4096
    for config in configs:
        if is_large_gemm:
            if not (
                config["use_half_tile_interleaved"]
                and config["block_m"] == 256
                and config["block_n"] == 256
                and config["block_k"] == 64
                and config["stages"] == 2
                and config["split_k"] == 1
                and config["m_waves"] == 2
                and config["n_waves"] == 4
                and config["k_waves"] == 1
            ):
                continue
        elif not config["use_half_tile_interleaved"]:
            mma_m_iters = config["block_m"] // config["m_waves"] // 16
            mma_n_iters = config["block_n"] // config["n_waves"] // 16
            if mma_m_iters > 4 or mma_n_iters > 4:
                continue

        validation_config = {
            **config,
            "in_dtype_id": in_dtype_id,
            "out_dtype_id": out_dtype_id,
            "a_is_transposed": False,
            "b_is_transposed": True,
            "has_bias": has_bias,
        }
        if (
            make_gemm_a16w16_param_and_validate(
                m,
                n,
                k,
                validation_config,
            )
            is not None
        ):
            valid_configs.append(config)
    return valid_configs
