# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Load validated GPT-OSS IQ2R tensors from a compiled checkpoint.

The loader understands the stable IQ2R tensor ABI, not any particular
compiler implementation.  It supports expert slicing so a single real expert
can serve as a lightweight correctness fixture during kernel bring-up.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .ops.iq2r_format import (
    IQ2R_ACTIVATION_BASIS,
    IQ2R_FORMAT_NAME,
    IQ2R_FORMAT_VERSION,
    IQ2R_GPT_OSS_EXPERTS,
    IQ2RMetadata,
    iq2r_gpt_oss_metadata,
    iq2r_validate_expert_weights,
)

IQ2R_CHECKPOINT_TENSOR_RANK = 0
IQ2R_CHECKPOINT_SCHEMA = "aiter-gpt-oss-iq2r"
IQ2R_CHECKPOINT_SCHEMA_VERSION = 1
IQ2R_GATE_UP_TN = 128
IQ2R_DOWN_TN = 64


def _require_safetensors():
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("IQ2R checkpoint loading requires safetensors") from error
    return safe_open


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return result


def iq2r_compiled_tensor_keys(
    layer_index: int, projection: str, *, tensor_rank: int = IQ2R_CHECKPOINT_TENSOR_RANK
) -> dict[str, str]:
    """Return the stable compiled-checkpoint keys for one projection."""

    if isinstance(layer_index, bool) or not isinstance(layer_index, int):
        raise TypeError("layer_index must be an int")
    if layer_index < 0:
        raise ValueError("layer_index must be non-negative")
    if projection == "gate_up":
        projection_name = "gate_up_proj"
    elif projection == "down":
        projection_name = "down_proj"
    else:
        raise ValueError(f"projection must be 'gate_up' or 'down', got {projection!r}")
    prefix = f"model.layers.{layer_index}.mlp.experts.{projection_name}.{tensor_rank}"
    return {
        "data": f"{prefix}.iq2r_data",
        "auxiliary": f"{prefix}.iq2r_auxiliary",
        "bias": f"{prefix}.bias",
        "tile_n": f"{prefix}.iq2r_tN",
    }


def iq2r_gpt_oss_source_keys(layer_index: int) -> dict[str, str]:
    """Return the six canonical Hugging Face GPT-OSS expert tensor names."""

    if isinstance(layer_index, bool) or not isinstance(layer_index, int):
        raise TypeError("layer_index must be an int")
    if layer_index < 0:
        raise ValueError("layer_index must be non-negative")
    prefix = f"model.layers.{layer_index}.mlp.experts"
    return {
        "gate_up_blocks": f"{prefix}.gate_up_proj_blocks",
        "gate_up_scales": f"{prefix}.gate_up_proj_scales",
        "gate_up_bias": f"{prefix}.gate_up_proj_bias",
        "down_blocks": f"{prefix}.down_proj_blocks",
        "down_scales": f"{prefix}.down_proj_scales",
        "down_bias": f"{prefix}.down_proj_bias",
    }


def _checkpoint_shards(model_dir: Path, config: dict[str, Any]) -> list[Path]:
    declared = config.get("file_manifest")
    if isinstance(declared, list):
        paths = [
            model_dir / name for name in declared if str(name).endswith(".safetensors")
        ]
    else:
        paths = sorted(model_dir.glob("*.safetensors"))
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "IQ2R checkpoint declares missing safetensor shards: " + ", ".join(missing)
        )
    if not paths:
        raise FileNotFoundError(f"no safetensor shards found in {model_dir}")
    return paths


def _validate_model_config(config: dict[str, Any]) -> None:
    iq2r = config.get("iq2r")
    if not isinstance(iq2r, dict):
        raise ValueError(
            "checkpoint config does not contain an IQ2R format declaration"
        )
    actual = (
        iq2r.get("format"),
        iq2r.get("version"),
        iq2r.get("activation_basis"),
    )
    expected = (IQ2R_FORMAT_NAME, IQ2R_FORMAT_VERSION, IQ2R_ACTIVATION_BASIS)
    if actual != expected:
        raise ValueError(
            f"unsupported IQ2R checkpoint identity {actual!r}; expected {expected!r}"
        )
    if int(config.get("compiled_tensor_parallel_size", -1)) != 1:
        raise ValueError("initial IQ2R integration requires compiled TP1")
    if int(config.get("compiled_expert_parallel_size", -1)) != 1:
        raise ValueError("initial IQ2R integration requires compiled EP1")
    if int(config.get("num_local_experts", -1)) != IQ2R_GPT_OSS_EXPERTS:
        raise ValueError("initial IQ2R integration requires 128 GPT-OSS experts")


def _find_projection_shard(shards: list[Path], keys: dict[str, str]) -> Path:
    safe_open = _require_safetensors()
    required = set(keys.values())
    for path in shards:
        with safe_open(path, framework="pt", device="cpu") as handle:
            if required.issubset(handle.keys()):
                return path
    raise KeyError(
        f"checkpoint shards do not contain the complete IQ2R projection: {sorted(required)}"
    )


def _load_projection_slice(
    path: Path,
    keys: dict[str, str],
    metadata: IQ2RMetadata,
    *,
    expert_start: int,
    expert_count: int,
) -> tuple[Tensor, Tensor, Tensor, int]:
    safe_open = _require_safetensors()
    stop = expert_start + expert_count
    with safe_open(path, framework="pt", device="cpu") as handle:
        data_shape = tuple(handle.get_slice(keys["data"]).get_shape())
        auxiliary_shape = tuple(handle.get_slice(keys["auxiliary"]).get_shape())
        bias_shape = tuple(handle.get_slice(keys["bias"]).get_shape())
        if data_shape != (IQ2R_GPT_OSS_EXPERTS, metadata.data_bytes):
            raise ValueError(
                f"{keys['data']} has shape {data_shape}, expected "
                f"({IQ2R_GPT_OSS_EXPERTS},{metadata.data_bytes})"
            )
        if auxiliary_shape != (IQ2R_GPT_OSS_EXPERTS, metadata.auxiliary_bytes):
            raise ValueError(
                f"{keys['auxiliary']} has shape {auxiliary_shape}, expected "
                f"({IQ2R_GPT_OSS_EXPERTS},{metadata.auxiliary_bytes})"
            )
        if bias_shape != (IQ2R_GPT_OSS_EXPERTS, metadata.logical_n):
            raise ValueError(
                f"{keys['bias']} has shape {bias_shape}, expected "
                f"({IQ2R_GPT_OSS_EXPERTS},{metadata.logical_n})"
            )
        data = handle.get_slice(keys["data"])[expert_start:stop].contiguous()
        auxiliary = handle.get_slice(keys["auxiliary"])[expert_start:stop].contiguous()
        bias = handle.get_slice(keys["bias"])[expert_start:stop].contiguous()
        tile_n_tensor = handle.get_tensor(keys["tile_n"])

    iq2r_validate_expert_weights(data, auxiliary, metadata, expert_count=expert_count)
    if bias.dtype != torch.bfloat16:
        raise TypeError(f"{keys['bias']} must be bfloat16, got {bias.dtype}")
    if tile_n_tensor.dtype != torch.int32 or tuple(tile_n_tensor.shape) != (1,):
        raise ValueError(f"{keys['tile_n']} must be int32 [1]")
    return data, auxiliary, bias, int(tile_n_tensor.item())


@dataclass(frozen=True, slots=True)
class IQ2RLayerCheckpoint:
    layer_index: int
    expert_start: int
    expert_count: int
    gate_up_data: Tensor
    gate_up_auxiliary: Tensor
    gate_up_bias: Tensor
    down_data: Tensor
    down_auxiliary: Tensor
    down_bias: Tensor
    gate_up_tile_n: int
    down_tile_n: int
    gate_up_metadata: IQ2RMetadata
    down_metadata: IQ2RMetadata
    source_shards: tuple[str, ...]


def load_iq2r_layer_checkpoint(
    model_dir: str | os.PathLike[str],
    layer_index: int,
    *,
    expert_start: int = 0,
    expert_count: int = IQ2R_GPT_OSS_EXPERTS,
    device: torch.device | str = "cpu",
) -> IQ2RLayerCheckpoint:
    """Load one GPT-OSS IQ2R layer, optionally slicing its expert range."""

    if expert_start < 0 or expert_count <= 0:
        raise ValueError("expert_start must be non-negative and expert_count positive")
    if expert_start + expert_count > IQ2R_GPT_OSS_EXPERTS:
        raise ValueError(
            "requested IQ2R expert slice exceeds the 128-expert checkpoint"
        )
    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing IQ2R checkpoint config: {config_path}")
    config = _read_json(config_path)
    _validate_model_config(config)
    total_layers = int(config.get("num_hidden_layers", -1))
    if not (0 <= layer_index < total_layers):
        raise ValueError(f"layer_index {layer_index} is outside [0,{total_layers})")

    shards = _checkpoint_shards(model_dir, config)
    gate_up_keys = iq2r_compiled_tensor_keys(layer_index, "gate_up")
    down_keys = iq2r_compiled_tensor_keys(layer_index, "down")
    gate_up_shard = _find_projection_shard(shards, gate_up_keys)
    down_shard = _find_projection_shard(shards, down_keys)
    gate_up_metadata = iq2r_gpt_oss_metadata("gate_up")
    down_metadata = iq2r_gpt_oss_metadata("down")
    gate_data, gate_auxiliary, gate_bias, gate_tile_n = _load_projection_slice(
        gate_up_shard,
        gate_up_keys,
        gate_up_metadata,
        expert_start=expert_start,
        expert_count=expert_count,
    )
    down_data, down_auxiliary, down_bias, down_tile_n = _load_projection_slice(
        down_shard,
        down_keys,
        down_metadata,
        expert_start=expert_start,
        expert_count=expert_count,
    )
    if gate_tile_n != IQ2R_GATE_UP_TN:
        raise ValueError(
            f"GPT-OSS gate/up IQ2R tile N must be {IQ2R_GATE_UP_TN}, got {gate_tile_n}"
        )
    if down_tile_n != IQ2R_DOWN_TN:
        raise ValueError(
            f"GPT-OSS down IQ2R tile N must be {IQ2R_DOWN_TN}, got {down_tile_n}"
        )

    return IQ2RLayerCheckpoint(
        layer_index=layer_index,
        expert_start=expert_start,
        expert_count=expert_count,
        gate_up_data=gate_data.to(device=device),
        gate_up_auxiliary=gate_auxiliary.to(device=device),
        gate_up_bias=gate_bias.to(device=device),
        down_data=down_data.to(device=device),
        down_auxiliary=down_auxiliary.to(device=device),
        down_bias=down_bias.to(device=device),
        gate_up_tile_n=gate_tile_n,
        down_tile_n=down_tile_n,
        gate_up_metadata=gate_up_metadata,
        down_metadata=down_metadata,
        source_shards=tuple(dict.fromkeys((gate_up_shard.name, down_shard.name))),
    )


__all__ = [
    "IQ2R_CHECKPOINT_TENSOR_RANK",
    "IQ2R_CHECKPOINT_SCHEMA",
    "IQ2R_CHECKPOINT_SCHEMA_VERSION",
    "IQ2R_DOWN_TN",
    "IQ2R_GATE_UP_TN",
    "IQ2RLayerCheckpoint",
    "iq2r_compiled_tensor_keys",
    "iq2r_gpt_oss_source_keys",
    "load_iq2r_layer_checkpoint",
]
