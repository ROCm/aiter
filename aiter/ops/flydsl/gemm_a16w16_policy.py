# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Shape-aware policy selection for the FlyDSL A16W16 GEMM."""

import itertools
from dataclasses import dataclass

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime, get_lds_capacity_bytes

from .gemm_a16w16_band import band_stream_config
from .kernels.gemm_a16w16_gfx950 import (
    GEMM_A16W16_DTYPE_BF16,
    GEMM_A16W16_DTYPE_FP16,
    GEMM_A16W16_DTYPE_FP32,
    SPLIT_K_WORKSPACE_MAX_ELEMS,
    make_gemm_a16w16_param_and_validate,
    split_k_workspace_supported,
)
from .kernels.gemm_a16w16_kernel_gfx1250 import LDS_PAD_A, LDS_PAD_B
from .kernels.gemm_a16w16_skinny_gfx950 import skinny_config_error

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
    tdm: bool = False
    b_k_contiguous: bool = True

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
        if self.tdm:
            pad = 8
            a_stage = config["block_m"] * (config["block_k"] + pad)
            b_stage = (
                config["block_n"] * (config["block_k"] + pad)
                if self.b_k_contiguous
                else config["block_k"] * (config["block_n"] + pad)
            )
            c_pad = 8 if config.get("interleave", False) else 0
            lds = (
                max(
                    config["stages"] * (a_stage + b_stage),
                    config["block_m"] * (config["block_n"] + c_pad),
                )
                * self.element_bytes
            )
        # A workgroup may occupy the whole per-CU LDS here, so the capacity the
        # chip table reports per workgroup is also the per-CU figure.
        lds_per_cu = (
            get_lds_capacity_bytes("gfx1250") if self.tdm else get_lds_capacity_bytes()
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
        # Large-M tuning must retain wider N tiles for ragged model dimensions
        # such as 2112 and 2880. Their 6-9% N padding can be cheaper than the
        # poor MFMA efficiency of an exactly-dividing BN64 tile.
        if self.m >= 4096:
            keep_ratio = min(keep_ratio, 0.90)
        min_iou = max(ious) * keep_ratio
        num_cus = self.device_props.multi_processor_count
        # Hand-added tiles wider than the 256x256 grid must not tighten the limit.
        grid_tiles = [
            self._tiles(config)
            for config in configs
            if config["block_m"] <= 256 and config["block_n"] <= 256
        ] or [self._tiles(config) for config in configs]
        max_tiles = max(num_cus, min(grid_tiles)) * self.max_tile_grid_ratio
        kept = []
        for config, iou in zip(configs, ious):
            tiles = self._tiles(config)
            validated_half_tile = (
                config["use_half_tile_interleaved"]
                and config["block_m"] == 256
                and config["block_n"] == 256
                and config["block_k"] == 64
                and config["stages"] == 2
                and config["split_k"] == 1
                and config["m_waves"] == 2
                and config["n_waves"] == 4
                and config["k_waves"] == 1
                and config["group_m"] in (0, 1, 2, 4, 8, 16)
            )
            validated_grouped_half_tile = (
                config["use_half_tile_interleaved"]
                and config["group_m"] == 4
                and (
                    config["block_m"],
                    config["block_n"],
                    config["block_k"],
                    config["m_waves"],
                    config["n_waves"],
                )
                == (192, 128, 128, 2, 4)
            )
            validated_half_tile_workspace = (
                config["use_half_tile_interleaved"]
                and config["block_m"] == 256
                and config["block_n"] == 256
                and config["block_k"] == 64
                and config["stages"] == 2
                and config["split_k"] > 1
                and config["m_waves"] == 2
                and config["n_waves"] == 4
                and config["k_waves"] == 1
                and config["group_m"] in (0, 4, 8, 16)
                and config["split_k"] * self.m * self.n <= SPLIT_K_WORKSPACE_MAX_ELEMS
            )
            near_cu_rect = (
                iou >= 0.90 and tiles * 4 >= num_cus * 3 and tiles * 4 <= num_cus * 5
            )
            max_split_k = (
                1
                if tiles >= num_cus
                else self._ceil_div(self.max_split_grid_rounds * num_cus, tiles)
            )
            if (
                (
                    iou >= min_iou
                    or near_cu_rect
                    or validated_half_tile
                    or validated_grouped_half_tile
                    or validated_half_tile_workspace
                )
                and tiles <= max_tiles
                and config["split_k"] <= max_split_k
                and (
                    config["group_m"] == 0
                    or validated_half_tile
                    or validated_grouped_half_tile
                    or validated_half_tile_workspace
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
                        (128, 128, 1),
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


def _gfx1250_tdm_configs(m, n, k, dtype, layout="nt"):
    from .kernels.gemm_a16w16_tdm_gfx1250 import (
        GEMM_DTYPE_BF16,
        GEMM_DTYPE_FP16,
        validate_gemm_gfx1250_config,
    )

    split_k_candidates = [1]
    split_k_candidates.extend(split_k for split_k in range(2, 10) if k % split_k == 0)
    selections = {
        "block_m": [16, 32, 48, 64, 80, 96, 128, 256],
        "block_n": [16, 32, 64, 80, 96, 128, 160, 256],
        "block_k": [64, 128, 256, 512],
        "stages": [i for i in range(2, 10)],
        "split_k": split_k_candidates,
        "m_waves": [1, 2, 4],
        "n_waves": [1, 2, 4],
        "group_m": [0, 4],
        "mma_m": [16],
        "mma_n": [16],
        "mma_k": [32],
        # NN keeps the original schedule; the validator checks NT quadrant fit.
        "interleave": [False, True] if layout == "nt" else [False],
    }
    if layout == "nt" and min(m, n, k) >= 4096:
        # Pin large NT GEMMs while retaining the group_m search.
        selections.update(
            {
                "block_m": [256],
                "block_n": [256],
                "block_k": [128],
                "stages": [2],
                "m_waves": [2],
                "n_waves": [2],
                "interleave": [True],
                "split_k": [1],
            }
        )
    keys = selections.keys()
    values = selections.values()
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    configs = [
        config
        for config in configs
        if (
            not config["interleave"]
            or (config["m_waves"], config["n_waves"]) not in ((2, 4), (4, 2), (4, 4))
        )
    ]
    for config in configs:
        config["k_waves"] = 1
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    b_k_contiguous = layout == "nt"
    pruner = GemmConfigPruner(
        m,
        n,
        k,
        device_props,
        2,
        tdm=True,
        b_k_contiguous=b_k_contiguous,
    )
    configs = pruner.prune(configs)
    dtype_id = GEMM_DTYPE_FP16 if dtype is torch.float16 else GEMM_DTYPE_BF16
    valid_configs = []
    for config in configs:
        try:
            validate_gemm_gfx1250_config(
                m,
                n,
                k,
                config["block_m"],
                config["block_n"],
                config["block_k"],
                config["stages"],
                config["m_waves"],
                config["n_waves"],
                config["group_m"],
                dtype_id,
                config["mma_m"],
                config["mma_n"],
                config["mma_k"],
                b_k_contiguous,
                config["split_k"],
                interleave=config["interleave"],
            )
            valid_configs.append(config)
        except ValueError:
            # Invalid tile/partition combinations are expected during enumeration.
            continue
    for config in valid_configs:
        config.pop("mma_m")
        config.pop("mma_n")
        config.pop("mma_k")
        config["policy"] = "tdmi" if config["interleave"] else "tdm"
        config["use_half_tile_interleaved"] = False
    return valid_configs


def get_flydsl_a16w16_configs(
    m: int,
    n: int,
    k: int,
    dtype: torch.dtype,
    out_dtype: torch.dtype,
    has_bias: bool,
    *,
    use_large_gemm_fast_path: bool = True,
    apply_policy_pruning: bool = True,
):
    """Generate and validate the shape-aware policy catalog used by tuning."""

    if dtype not in (torch.float16, torch.bfloat16):
        return []
    if out_dtype not in (dtype, torch.float32):
        return []
    if get_gfx_runtime() == "gfx1250" and out_dtype == dtype:
        # Re-add the original gfx1250 FT/HT backend to the tuner race next to
        # TDM. Their split-K partials and K-tail handling differ, so each
        # backend is validated by its own catalog.
        return _gfx1250_tdm_configs(m, n, k, dtype) + _gfx1250_configs(m, n, k)
    if get_gfx_runtime() in ("gfx1250", "gfx1250-strict"):
        return _gfx1250_configs(m, n, k)
    if make_gemm_a16w16_param_and_validate is None:
        return []
    if get_gfx_runtime() != "gfx950":
        return []

    # FT may distribute whole K tiles unevenly. The kernel validator retains
    # the stricter HTI/alignment requirements for each candidate.
    split_k_candidates = list(range(1, min(k, 9) + 1))
    selections = {
        "block_m": [16, 32, 48, 64, 80, 96, 128, 192, 208, 224, 256],
        "block_n": [16, 32, 64, 80, 96, 128, 160, 256],
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
    for group_m in (1, 2, 8, 16):
        configs.append(
            {
                "block_m": 256,
                "block_n": 256,
                "block_k": 64,
                "stages": 2,
                "split_k": 1,
                "m_waves": 2,
                "n_waves": 4,
                "k_waves": 1,
                "group_m": group_m,
                "use_half_tile_interleaved": True,
            }
        )
    for split_k in split_k_candidates[1:]:
        if split_k * m * n <= SPLIT_K_WORKSPACE_MAX_ELEMS:
            for group_m in (8, 16):
                configs.append(
                    {
                        "block_m": 256,
                        "block_n": 256,
                        "block_k": 64,
                        "stages": 2,
                        "split_k": split_k,
                        "m_waves": 2,
                        "n_waves": 4,
                        "k_waves": 1,
                        "group_m": group_m,
                        "use_half_tile_interleaved": True,
                    }
                )
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
    if k % 2 == 0 and 2 * m * n <= SPLIT_K_WORKSPACE_MAX_ELEMS:
        configs.append(
            {
                "block_m": 320,
                "block_n": 192,
                "block_k": 64,
                "stages": 2,
                "split_k": 2,
                "m_waves": 4,
                "n_waves": 4,
                "k_waves": 1,
                "group_m": 0,
                "use_half_tile_interleaved": False,
            }
        )
    if k % 4 == 0 and 4 * m * n <= SPLIT_K_WORKSPACE_MAX_ELEMS:
        configs.append(
            {
                "block_m": 320,
                "block_n": 192,
                "block_k": 64,
                "stages": 2,
                "split_k": 4,
                "m_waves": 4,
                "n_waves": 4,
                "k_waves": 1,
                "group_m": 0,
                "use_half_tile_interleaved": False,
            }
        )
    if 384 < m <= 400 and k % 4 == 0:
        configs.append(
            {
                "block_m": 400,
                "block_n": 128,
                "block_k": 64,
                "stages": 2,
                "split_k": 4,
                "m_waves": 5,
                "n_waves": 2,
                "k_waves": 1,
                "group_m": 0,
                "use_half_tile_interleaved": False,
            }
        )
    if m <= 320 and k % 7 == 0:
        configs.append(
            {
                "block_m": 320,
                "block_n": 192,
                "block_k": 64,
                "stages": 2,
                "split_k": 7,
                "m_waves": 4,
                "n_waves": 4,
                "k_waves": 1,
                "group_m": 0,
                "use_half_tile_interleaved": False,
            }
        )
    if k % 7 == 0 and 7 * m * n <= SPLIT_K_WORKSPACE_MAX_ELEMS:
        for m_waves, n_waves in ((4, 2), (2, 4)):
            configs.append(
                {
                    "block_m": 256,
                    "block_n": 256,
                    "block_k": 64,
                    "stages": 2,
                    "split_k": 7,
                    "m_waves": m_waves,
                    "n_waves": n_waves,
                    "k_waves": 1,
                    "group_m": 0,
                    "use_half_tile_interleaved": False,
                }
            )
    if 320 < m <= 384 and k % 7 == 0:
        configs.append(
            {
                "block_m": 384,
                "block_n": 192,
                "block_k": 64,
                "stages": 2,
                "split_k": 7,
                "m_waves": 8,
                "n_waves": 1,
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
    for m_waves, n_waves, k_waves in ((2, 2, 2), (4, 4, 1)):
        configs.append(
            {
                "block_m": 128,
                "block_n": 128,
                "block_k": 64,
                "stages": 3,
                "split_k": 1,
                "m_waves": m_waves,
                "n_waves": n_waves,
                "k_waves": k_waves,
                "group_m": 1,
                "use_half_tile_interleaved": False,
            }
        )
    is_large_gemm = (
        apply_policy_pruning
        and use_large_gemm_fast_path
        and m >= 4096
        and n >= 4096
        and k >= 4096
    )
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
    if apply_policy_pruning and not is_large_gemm:
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
    memory_configs = []
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
        if (
            apply_policy_pruning
            and not is_large_gemm
            and not config["use_half_tile_interleaved"]
        ):
            mma_m_iters = config["block_m"] // config["m_waves"] // 16
            mma_n_iters = config["block_n"] // config["n_waves"] // 16
            # Decode tiles can stream a wider N panel per wave. The old
            # four-repeat cutoff excluded 128x96 / 4x1-wave candidates.
            max_n_iters = 8 if m <= 128 and m <= config["block_m"] else 4
            if (mma_m_iters > 4 or mma_n_iters > max_n_iters) and (
                (config["block_m"], config["block_n"])
                not in (
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
                    (384, 192),
                    (256, 256),
                    (400, 128),
                )
                # Past 64 accumulator tiles per wave the kernel spills to
                # scratch, which can exhaust device memory at dispatch.
                or mma_m_iters * mma_n_iters > 64
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
            # Atomic and workspace reductions trade launch count for traffic;
            # keep both so tuning picks the faster one per shape.
            if split_k_workspace_supported(
                m, n, config["split_k"], out_dtype_id, has_bias
            ) and (
                make_gemm_a16w16_param_and_validate(
                    m, n, k, {**validation_config, "use_split_k_workspace": True}
                )
                is not None
            ):
                valid_configs.append({**config, "use_split_k_workspace": True})
            # Stagger is a measured option, never a global default. At small
            # M, a single row tile streams B; preserve the original catalog
            # and let tuning race stagger and B cache policy against the original.
            if m <= 128 and m <= config["block_m"]:
                memory_configs.extend(
                    {**config, "k_stagger": stride, "b_non_temporal": nt}
                    for stride in (0, 1, 3)
                    for nt in (False, True)
                    if stride or nt
                )
    skinny_configs = []
    if dtype == out_dtype == torch.bfloat16 and not has_bias and 0 < m <= 128:
        # The public wrapper and tuned-name parser already support skinny,
        # but the tuner previously generated only FT/HTI candidates.
        block_m = -(-m // 8) * 8
        for block_n, block_k, stages, k_waves in itertools.product(
            (16, 24, 32, 48, 64, 72, 80, 96), (32, 64, 128), (2, 3, 4, 6), (2, 4, 8)
        ):
            if skinny_config_error(n, k, block_m, block_n, block_k, stages, k_waves):
                continue
            skinny_configs.append(
                {
                    "block_m": block_m,
                    "block_n": block_n,
                    "block_k": block_k,
                    "stages": stages,
                    "split_k": 1,
                    "m_waves": 1,
                    "n_waves": 1,
                    "k_waves": k_waves,
                    "group_m": 0,
                    "use_half_tile_interleaved": False,
                    "policy": "sk",
                }
            )
    band_configs = []
    if dtype == out_dtype == torch.bfloat16 and not has_bias:
        config = band_stream_config(m, n, k)
        if config is not None and device_props.multi_processor_count == 256:
            band_configs.append(config)
    return valid_configs + memory_configs + skinny_configs + band_configs
