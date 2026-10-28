# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shape-aware policy selection for the FlyDSL A16W16 GEMM."""

import itertools
from dataclasses import dataclass

import torch

from aiter.jit.utils.chip_info import get_gfx, get_lds_capacity_bytes

from .kernels.gemm_a16w16_gfx950 import (
    GEMM_A16W16_DTYPE_BF16,
    GEMM_A16W16_DTYPE_FP16,
    GEMM_A16W16_DTYPE_FP32,
    make_gemm_a16w16_param_and_validate,
)
from .kernels.gemm_a16w16_kernel_gfx1250 import LDS_PAD_A, LDS_PAD_B

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
        # A workgroup may occupy the whole per-CU LDS here, so the capacity the
        # chip table reports per workgroup is also the per-CU figure.
        lds_per_cu = get_lds_capacity_bytes()
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
        # Large-M tuning must retain wider N tiles for ragged model dimensions
        # such as 2112 and 2880. Their 6-9% N padding can be cheaper than the
        # poor MFMA efficiency of an exactly-dividing BN64 tile.
        if self.m >= 4096:
            keep_ratio = min(keep_ratio, 0.90)
        min_iou = max(ious) * keep_ratio
        num_cus = self.device_props.multi_processor_count
        max_tiles = (
            max(num_cus, min(self._tiles(config) for config in configs))
            * self.max_tile_grid_ratio
        )
        kept = []
        for config, iou in zip(configs, ious):
            tiles = self._tiles(config)
            near_cu_rect = (
                iou >= 0.90 and tiles * 4 >= num_cus * 3 and tiles * 4 <= num_cus * 5
            )
            max_split_k = (
                1
                if tiles >= num_cus
                else self._ceil_div(self.max_split_grid_rounds * num_cus, tiles)
            )
            if (
                (iou >= min_iou or near_cu_rect)
                and tiles <= max_tiles
                and config["split_k"] <= max_split_k
                and (
                    config["group_m"] == 0
                    or (
                        config["block_m"],
                        config["block_n"],
                        config["group_m"],
                    )
                    in (
                        (320, 224, 2),
                        (320, 224, 4),
                        (320, 224, 16),
                        (224, 320, 8),
                        (160, 192, 16),
                    )
                    or (tiles >= num_cus and tiles % 8 == 0)
                )
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


# Tiles of the gfx1250 kernel (gemm_a16w16_gfx1250) as (block_m, block_n, block_k,
# m_waves, n_waves). They hold the config of every gfx1250 FlyDSL row in the BF16
# tuned CSVs, which come from sweeps over BF16 shapes from M = 1 to 16384.
_GFX1250_TILES = (
    (16, 16, 256, 1, 1),
    (16, 64, 256, 1, 2),
    (16, 128, 256, 1, 2),
    (64, 64, 128, 4, 2),
    (64, 64, 256, 4, 2),
    (64, 128, 128, 2, 2),
    (128, 128, 32, 2, 4),
    (128, 128, 64, 2, 2),
    (128, 256, 64, 2, 4),
    (256, 128, 64, 4, 2),
    (256, 256, 64, 2, 4),
    (256, 256, 128, 2, 2),
    (256, 256, 128, 2, 4),
)


def gfx1250_lds_bytes(block_m, block_n, block_k, stages, split_k):
    """LDS of gemm_a16w16_gfx1250: every stage holds a padded A and B tile."""
    a_elems = block_m * (block_k + LDS_PAD_A) + LDS_PAD_A
    b_elems = block_n * (block_k + LDS_PAD_B) + LDS_PAD_B
    # Split-K adds a 16-byte-aligned flag.
    return stages * (a_elems + b_elems) * 2 + (16 if split_k > 1 else 0)


def _gfx1250_configs(m: int, n: int, k: int):
    lds_bytes = get_lds_capacity_bytes("gfx1250")
    split_k_candidates = [1]
    split_k_candidates.extend(split_k for split_k in range(2, 9) if k % split_k == 0)
    configs = []
    for (
        (block_m, block_n, block_k, m_waves, n_waves),
        stages,
        split_k,
        half,
    ) in itertools.product(
        _GFX1250_TILES, range(2, 7), split_k_candidates, (False, True)
    ):
        if gfx1250_lds_bytes(block_m, block_n, block_k, stages, split_k) > lds_bytes:
            continue
        # The split-K reduction code grows with the accumulators per wave: with 64 of
        # them (256x256 on 2x2 waves) one candidate compiled for minutes, and no gfx1250
        # row uses split-K with more than 2.
        if split_k > 1 and (block_m // m_waves // 16) * (block_n // n_waves // 16) > 4:
            continue
        # The prologue fills stages - 1 buffers from each split's K range.
        if -(-(k // split_k) // block_k) < stages - 1:
            continue
        configs.append(
            {
                "block_m": block_m,
                "block_n": block_n,
                "block_k": block_k,
                "stages": stages,
                "split_k": split_k,
                "m_waves": m_waves,
                "n_waves": n_waves,
                # The gfx1250 kernel has no slice-K and ignores group_m.
                "k_waves": 1,
                "group_m": 0,
                "use_half_tile_interleaved": half,
            }
        )
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    return GemmConfigPruner(m, n, k, device_props, 2).prune(configs)


def get_flydsl_a16w16_configs(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    out_dtype: torch.dtype,
    has_bias: bool,
    *,
    use_large_gemm_fast_path: bool = True,
):
    """Generate and validate the shape-aware policy catalog used by tuning."""

    if dtype not in (torch.float16, torch.bfloat16):
        return []
    if out_dtype not in (dtype, torch.float32):
        return []
    if get_gfx() in ("gfx1250", "gfx1250-strict"):
        return _gfx1250_configs(m, n, k)
    if make_gemm_a16w16_param_and_validate is None:
        return []
    if get_gfx() != "gfx950":
        return []

    split_k_candidates = [1]
    split_k_candidates.extend(split_k for split_k in range(2, 10) if k % split_k == 0)
    selections = {
        "block_m": [16, 32, 48, 64, 80, 96, 128, 192, 208, 224, 256],
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
    if m <= 320 and k % 5 == 0:
        configs.append(
            {
                "block_m": 320,
                "block_n": 192,
                "block_k": 64,
                "stages": 2,
                "split_k": 5,
                "m_waves": 4,
                "n_waves": 2,
                "k_waves": 1,
                "group_m": 0,
                "use_half_tile_interleaved": False,
            }
        )
    for block_m, block_n, stages, m_waves, n_waves, group_m in (
        (256, 192, 2, 4, 4, 0),
        (288, 192, 2, 2, 4, 0),
        (320, 224, 2, 4, 2, 0),
        (320, 224, 2, 4, 2, 2),
        (320, 224, 2, 4, 2, 4),
        (320, 224, 2, 4, 2, 16),
        (224, 320, 2, 2, 4, 8),
        (224, 192, 3, 2, 6, 0),
        (160, 192, 3, 2, 4, 16),
        (192, 192, 3, 3, 4, 0),
    ):
        configs.append(
            {
                "block_m": block_m,
                "block_n": block_n,
                "block_k": 64,
                "stages": stages,
                "split_k": 1,
                "m_waves": m_waves,
                "n_waves": n_waves,
                "k_waves": 1,
                "group_m": group_m,
                "use_half_tile_interleaved": False,
            }
        )
    is_large_gemm = use_large_gemm_fast_path and m >= 4096 and n >= 4096 and k >= 4096
    if is_large_gemm:
        configs = [
            config
            for config in configs
            if (
                config["use_half_tile_interleaved"]
                and config["block_m"] == 256
                and config["block_n"] == 256
                and config["block_k"] == 64
                and config["stages"] == 2
                and config["split_k"] == 1
                and config["m_waves"] == 2
                and config["n_waves"] == 4
                and config["k_waves"] == 1
            )
        ]
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    if not is_large_gemm:
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
    for config in configs:
        # The small-BM half-tile path corrupts ragged-N output because its
        # second N half uses a full-tile async-load layout. Keep ragged N on
        # full-tile or BM>=128 policies until that kernel path is fixed.
        if (
            config["use_half_tile_interleaved"]
            and config["block_m"] < 128
            and n % config["block_n"] != 0
        ):
            continue
        if not is_large_gemm and not config["use_half_tile_interleaved"]:
            mma_m_iters = config["block_m"] // config["m_waves"] // 16
            mma_n_iters = config["block_n"] // config["n_waves"] // 16
            if (mma_m_iters > 4 or mma_n_iters > 4) and (
                config["block_m"],
                config["block_n"],
            ) not in (
                (208, 256),
                (224, 256),
                (320, 192),
                (256, 192),
                (288, 192),
                (320, 224),
                (224, 320),
                (224, 192),
                (160, 192),
                (192, 192),
            ):
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
