# SPDX-License-Identifier: MIT
# Ported from xytpai/flydsl-examples @ 5017f4c16fc46badb5fce58c485e9bd090ffe325.
"""Original gfx1250 TDM configuration space; no replacement of existing policies."""

import itertools
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .kernels.gemm_a16w16_tdm_gfx1250 import (
    GEMM_DTYPE_BF16,
    GEMM_DTYPE_FP16,
    validate_gemm_gfx1250_config,
)


@dataclass(frozen=True)
class GemmConfigPruner:
    m: int
    n: int
    k: int
    device_props: object
    element_bytes: int
    target_waves_per_cu: int = 8
    max_split_grid_rounds: int = 2
    max_tile_grid_ratio: int = 4
    arch: str = "gfx950"
    b_k_contiguous: bool = True

    @staticmethod
    def _ceil_div(value, divisor):
        return (value + divisor - 1) // divisor

    def _tiles(self, config):
        return self._ceil_div(self.m, config["block_m"]) * self._ceil_div(
            self.n, config["block_n"]
        )

    def _iou(self, config):
        split_k = config.get("split_k", 1)
        padded_m = self._ceil_div(self.m, config["block_m"]) * config["block_m"]
        padded_n = self._ceil_div(self.n, config["block_n"]) * config["block_n"]
        part_k = self.k // split_k
        padded_k = (
            self._ceil_div(part_k, config["block_k"]) * config["block_k"] * split_k
        )
        return self.m * self.n * self.k / (padded_m * padded_n * padded_k)

    def _lds_bytes(self, config):
        if self.arch != "gfx1250":
            return max(
                config["stages"]
                * (config["block_m"] + config["block_n"])
                * config["block_k"]
                * self.element_bytes,
                config["k_waves"]
                * config["block_m"]
                * config["block_n"]
                * self.element_bytes,
            )
        pad = 8
        block_k = config["block_k"]
        a_stage = config["block_m"] * (block_k + pad)
        if self.b_k_contiguous:
            b_stage = config["block_n"] * (block_k + pad)
        else:
            b_stage = block_k * (config["block_n"] + pad)
        elements = max(
            config["stages"] * (a_stage + b_stage),
            config["block_m"] * config["block_n"],
        )
        return elements * self.element_bytes

    def _occupancy(self, config):
        props = self.device_props
        waves = config["m_waves"] * config["n_waves"] * config.get("k_waves", 1)
        lds = self._lds_bytes(config)
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
            * config.get("split_k", 1)
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
            split_k = config.get("split_k", 1)
            max_split_k = (
                1
                if tiles >= num_cus
                else self._ceil_div(self.max_split_grid_rounds * num_cus, tiles)
            )
            if (
                iou >= min_iou
                and tiles <= max_tiles
                and split_k <= max_split_k
                and (config["group_m"] == 0 or (tiles >= num_cus and tiles % 8 == 0))
            ):
                kept.append(config)
        best = {}
        keep = set()
        for index, config in sorted(
            enumerate(kept), key=lambda item: (item[1].get("k_waves", 1), item[0])
        ):
            key = tuple(
                (name, value) for name, value in config.items() if name != "k_waves"
            )
            occupancy = self._occupancy(config)
            if occupancy > best.get(key, -1.0) + 1e-9:
                best[key] = occupancy
                keep.add(index)
        return [config for index, config in enumerate(kept) if index in keep]


def get_gfx1250_tdm_configs(m, n, k, dtype, *, layout="nt", enable_split_k=True):
    """Exact source config space and pruning, returned in AITER policy schema."""
    if dtype not in (torch.float16, torch.bfloat16):
        return []
    args = SimpleNamespace(
        m=m, n=n, k=k, dtype=dtype, layout=layout, enable_split_k=enable_split_k
    )
    split_k_candidates = [1]
    if args.enable_split_k:
        split_k_candidates.extend(
            split_k for split_k in range(2, 10) if args.k % split_k == 0
        )
    selections = {
        "block_m": [16, 32, 48, 64, 80, 96, 128, 256],
        "block_n": [16, 32, 64, 80, 96, 128, 256],
        "block_k": [64, 128, 256, 512],
        "stages": [i for i in range(2, 10)],
        "split_k": split_k_candidates,
        "m_waves": [1, 2, 4],
        "n_waves": [1, 2, 4],
        "group_m": [0, 4],
        "mma_m": [16],
        "mma_n": [16],
        "mma_k": [32],
    }
    keys = selections.keys()
    values = selections.values()
    configs = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
    device_props = torch.cuda.get_device_properties(torch.cuda.current_device())
    b_k_contiguous = args.layout == "nt"
    pruner = GemmConfigPruner(
        args.m,
        args.n,
        args.k,
        device_props,
        2,
        arch="gfx1250",
        b_k_contiguous=b_k_contiguous,
    )
    configs = pruner.prune(configs)
    dtype_id = GEMM_DTYPE_FP16 if args.dtype is torch.float16 else GEMM_DTYPE_BF16
    valid_configs = []
    for config in configs:
        try:
            validate_gemm_gfx1250_config(
                args.m,
                args.n,
                args.k,
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
            )
            config = {
                name: value
                for name, value in config.items()
                if not name.startswith("mma_")
            }
            config.update(k_waves=1, use_half_tile_interleaved=False, policy="tdm")
            valid_configs.append(config)
        except ValueError:
            continue
    return valid_configs
