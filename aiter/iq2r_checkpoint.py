# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Load validated IQ2R tensors from a Redline compiled checkpoint.

The loader understands the stable IQ2R tensor ABI, not any particular
compiler implementation. GPT-OSS v1 checkpoints remain supported, while the
model-neutral path also handles bias-free DeepSeek-shaped MoE checkpoints such
as GLM-5.3. Expert slicing lets one real expert serve as a lightweight
correctness fixture during kernel bring-up.
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
# Legacy Redline/GPT-OSS shard metadata. Compiled model config validation is
# based on the public ``iq2r`` format declaration and accepts both schemas.
IQ2R_CHECKPOINT_SCHEMA = "aiter-gpt-oss-iq2r"
IQ2R_CHECKPOINT_SCHEMA_VERSION = 1
IQ2R_GENERIC_CHECKPOINT_SCHEMA = "aiter-iq2r"
IQ2R_GENERIC_CHECKPOINT_SCHEMA_VERSION = 2
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
        raise TypeError(f"expected a JSON object in {path}")
    return result


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


def _config_value(config: dict[str, Any], *names: str, default: Any = None) -> Any:
    text = _text_config(config)
    for name in names:
        if config.get(name) is not None:
            return config[name]
        if text.get(name) is not None:
            return text[name]
    return default


def _config_int(config: dict[str, Any], *names: str, default: int | None = None) -> int:
    value = _config_value(config, *names, default=default)
    if isinstance(value, bool) or not isinstance(value, int):
        joined = "/".join(names)
        raise TypeError(f"checkpoint config has no integer {joined}")
    return value


def _architectures(config: dict[str, Any]) -> tuple[str, ...]:
    value = _config_value(config, "architectures", default=())
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        return ()
    return tuple(value)


@dataclass(frozen=True, slots=True)
class _CheckpointLayout:
    model_family: str
    total_layers: int
    first_moe_layer: int
    expert_count: int
    hidden_size: int
    intermediate_size: int
    gate_up_module: str
    down_module: str
    has_bias: bool


def _checkpoint_layout(config: dict[str, Any]) -> _CheckpointLayout:
    architectures = _architectures(config)
    model_type = str(_config_value(config, "model_type", default=""))
    iq2r = config.get("iq2r")
    assert isinstance(iq2r, dict)

    is_gpt_oss = model_type == "gpt_oss" or any(
        "GptOss" in architecture or "GPTOSS" in architecture
        for architecture in architectures
    )
    is_glm = model_type in ("glm5", "glm5_next", "glm_moe_dsa") or any(
        architecture.startswith(("Glm5", "GlmMoe")) for architecture in architectures
    )

    total_layers = _config_int(config, "num_hidden_layers")
    expert_count = _config_int(config, "num_local_experts", "n_routed_experts")
    hidden_size = _config_int(config, "hidden_size")
    intermediate_size = _config_int(
        config, "moe_intermediate_size", "intermediate_size"
    )

    if is_gpt_oss:
        return _CheckpointLayout(
            model_family="gpt_oss",
            total_layers=total_layers,
            first_moe_layer=0,
            expert_count=expert_count,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gate_up_module="experts.gate_up_proj",
            down_module="experts.down_proj",
            has_bias=True,
        )
    if is_glm:
        return _CheckpointLayout(
            model_family="glm5",
            total_layers=total_layers,
            first_moe_layer=_config_int(config, "first_k_dense_replace", default=0),
            expert_count=expert_count,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            gate_up_module="up_gate_proj",
            down_module="down_proj",
            has_bias=False,
        )

    projection_modules = iq2r.get("expert_projection_modules")
    if not isinstance(projection_modules, dict):
        raise TypeError(
            "unsupported IQ2R checkpoint architecture; model-neutral checkpoints "
            "must declare iq2r.expert_projection_modules"
        )
    gate_up_module = projection_modules.get("gate_up")
    down_module = projection_modules.get("down")
    if not isinstance(gate_up_module, str) or not isinstance(down_module, str):
        raise TypeError(
            "iq2r.expert_projection_modules must contain gate_up and down strings"
        )
    return _CheckpointLayout(
        model_family=model_type or "generic",
        total_layers=total_layers,
        first_moe_layer=_config_int(config, "first_k_dense_replace", default=0),
        expert_count=expert_count,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        gate_up_module=gate_up_module,
        down_module=down_module,
        has_bias=bool(iq2r.get("expert_bias", False)),
    )


def iq2r_compiled_tensor_keys(
    layer_index: int,
    projection: str,
    *,
    tensor_rank: int = IQ2R_CHECKPOINT_TENSOR_RANK,
    module_name: str | None = None,
) -> dict[str, str]:
    """Return the stable compiled-checkpoint keys for one projection.

    ``module_name`` defaults to the legacy GPT-OSS path. Redline's
    DeepSeek-shaped loader uses ``up_gate_proj`` and ``down_proj`` instead.
    """

    if isinstance(layer_index, bool) or not isinstance(layer_index, int):
        raise TypeError("layer_index must be an int")
    if layer_index < 0:
        raise ValueError("layer_index must be non-negative")
    if projection == "gate_up":
        projection_name = module_name or "experts.gate_up_proj"
    elif projection == "down":
        projection_name = module_name or "experts.down_proj"
    else:
        raise ValueError(f"projection must be 'gate_up' or 'down', got {projection!r}")
    prefix = f"model.layers.{layer_index}.mlp.{projection_name}.{tensor_rank}"
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


def iq2r_glm5_source_keys(
    layer_index: int,
    expert_index: int,
    *,
    root: str = "model.language_model",
) -> dict[str, str]:
    """Return GLM-5 block-FP8 source keys for one routed expert."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (layer_index, expert_index)
    ):
        raise ValueError("layer_index and expert_index must be non-negative ints")
    prefix = f"{root}.layers.{layer_index}.mlp.experts.{expert_index}"
    return {
        f"{projection}_{kind}": f"{prefix}.{projection}.{kind}"
        for projection in ("gate_proj", "up_proj", "down_proj")
        for kind in ("weight", "weight_scale_inv")
    }


def iq2r_glm5_overlay_keys(
    layer_index: int, *, root: str = "model.language_model"
) -> dict[str, str]:
    """Return fused IQ2R overlay keys consumed by ATOM's GLM adapter."""

    if isinstance(layer_index, bool) or not isinstance(layer_index, int):
        raise TypeError("layer_index must be an int")
    if layer_index < 0:
        raise ValueError("layer_index must be non-negative")
    prefix = f"{root}.layers.{layer_index}.mlp.experts"
    return {
        "gate_up_data": f"{prefix}.iq2r_gate_up_data",
        "gate_up_auxiliary": f"{prefix}.iq2r_gate_up_auxiliary",
        "down_data": f"{prefix}.iq2r_down_data",
        "down_auxiliary": f"{prefix}.iq2r_down_auxiliary",
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


def _validate_model_config(config: dict[str, Any]) -> _CheckpointLayout:
    iq2r = config.get("iq2r")
    if not isinstance(iq2r, dict):
        raise TypeError("checkpoint config does not contain an IQ2R format declaration")
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
    layout = _checkpoint_layout(config)
    if layout.expert_count <= 0 or layout.expert_count > 512:
        raise ValueError("IQ2R checkpoint must contain between 1 and 512 experts")
    if layout.model_family == "gpt_oss" and layout.expert_count != IQ2R_GPT_OSS_EXPERTS:
        raise ValueError("legacy GPT-OSS IQ2R checkpoints require 128 experts")
    return layout


def _find_projection_shard(
    shards: list[Path], keys: dict[str, str], *, has_bias: bool
) -> Path:
    safe_open = _require_safetensors()
    required_names = ("data", "auxiliary", "tile_n") + (("bias",) if has_bias else ())
    required = {keys[name] for name in required_names}
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
    total_experts: int,
    expert_start: int,
    expert_count: int,
    has_bias: bool,
) -> tuple[Tensor, Tensor, Tensor | None, int]:
    safe_open = _require_safetensors()
    stop = expert_start + expert_count
    with safe_open(path, framework="pt", device="cpu") as handle:
        data_shape = tuple(handle.get_slice(keys["data"]).get_shape())
        auxiliary_shape = tuple(handle.get_slice(keys["auxiliary"]).get_shape())
        if data_shape != (total_experts, metadata.data_bytes):
            raise ValueError(
                f"{keys['data']} has shape {data_shape}, expected "
                f"({total_experts},{metadata.data_bytes})"
            )
        if auxiliary_shape != (total_experts, metadata.auxiliary_bytes):
            raise ValueError(
                f"{keys['auxiliary']} has shape {auxiliary_shape}, expected "
                f"({total_experts},{metadata.auxiliary_bytes})"
            )
        data = handle.get_slice(keys["data"])[expert_start:stop].contiguous()
        auxiliary = handle.get_slice(keys["auxiliary"])[expert_start:stop].contiguous()
        if has_bias:
            bias_shape = tuple(handle.get_slice(keys["bias"]).get_shape())
            if bias_shape != (total_experts, metadata.logical_n):
                raise ValueError(
                    f"{keys['bias']} has shape {bias_shape}, expected "
                    f"({total_experts},{metadata.logical_n})"
                )
            bias = handle.get_slice(keys["bias"])[expert_start:stop].contiguous()
        else:
            bias = None
        tile_n_tensor = handle.get_tensor(keys["tile_n"])

    iq2r_validate_expert_weights(data, auxiliary, metadata, expert_count=expert_count)
    if bias is not None and bias.dtype != torch.bfloat16:
        raise TypeError(f"{keys['bias']} must be bfloat16, got {bias.dtype}")
    if tile_n_tensor.dtype != torch.int32 or tuple(tile_n_tensor.shape) != (1,):
        raise ValueError(f"{keys['tile_n']} must be int32 [1]")
    return data, auxiliary, bias, int(tile_n_tensor.item())


@dataclass(frozen=True, slots=True)
class IQ2RLayerCheckpoint:
    layer_index: int
    expert_start: int
    expert_count: int
    total_experts: int
    model_family: str
    gate_up_data: Tensor
    gate_up_auxiliary: Tensor
    gate_up_bias: Tensor | None
    down_data: Tensor
    down_auxiliary: Tensor
    down_bias: Tensor | None
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
    expert_count: int | None = None,
    device: torch.device | str = "cpu",
) -> IQ2RLayerCheckpoint:
    """Load one IQ2R MoE layer, optionally slicing its expert range."""

    model_dir = Path(model_dir)
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"missing IQ2R checkpoint config: {config_path}")
    config = _read_json(config_path)
    layout = _validate_model_config(config)
    if expert_count is None:
        expert_count = layout.expert_count
    if expert_start < 0 or expert_count <= 0:
        raise ValueError("expert_start must be non-negative and expert_count positive")
    if expert_start + expert_count > layout.expert_count:
        raise ValueError(
            "requested IQ2R expert slice exceeds the checkpoint expert count "
            f"({layout.expert_count})"
        )
    if not (layout.first_moe_layer <= layer_index < layout.total_layers):
        raise ValueError(
            f"layer_index {layer_index} is outside the MoE layer range "
            f"[{layout.first_moe_layer},{layout.total_layers})"
        )

    shards = _checkpoint_shards(model_dir, config)
    gate_up_keys = iq2r_compiled_tensor_keys(
        layer_index, "gate_up", module_name=layout.gate_up_module
    )
    down_keys = iq2r_compiled_tensor_keys(
        layer_index, "down", module_name=layout.down_module
    )
    gate_up_shard = _find_projection_shard(
        shards, gate_up_keys, has_bias=layout.has_bias
    )
    down_shard = _find_projection_shard(shards, down_keys, has_bias=layout.has_bias)
    if layout.model_family == "gpt_oss":
        gate_up_metadata = iq2r_gpt_oss_metadata("gate_up")
        down_metadata = iq2r_gpt_oss_metadata("down")
    else:
        gate_up_metadata = IQ2RMetadata(
            logical_n=2 * layout.intermediate_size,
            logical_k=layout.hidden_size,
        )
        down_metadata = IQ2RMetadata(
            logical_n=layout.hidden_size,
            logical_k=layout.intermediate_size,
        )
    gate_data, gate_auxiliary, gate_bias, gate_tile_n = _load_projection_slice(
        gate_up_shard,
        gate_up_keys,
        gate_up_metadata,
        total_experts=layout.expert_count,
        expert_start=expert_start,
        expert_count=expert_count,
        has_bias=layout.has_bias,
    )
    down_data, down_auxiliary, down_bias, down_tile_n = _load_projection_slice(
        down_shard,
        down_keys,
        down_metadata,
        total_experts=layout.expert_count,
        expert_start=expert_start,
        expert_count=expert_count,
        has_bias=layout.has_bias,
    )
    if layout.model_family == "gpt_oss":
        if gate_tile_n != IQ2R_GATE_UP_TN:
            raise ValueError(
                f"GPT-OSS gate/up IQ2R tile N must be {IQ2R_GATE_UP_TN}, "
                f"got {gate_tile_n}"
            )
        if down_tile_n != IQ2R_DOWN_TN:
            raise ValueError(
                f"GPT-OSS down IQ2R tile N must be {IQ2R_DOWN_TN}, got {down_tile_n}"
            )
    else:
        for projection, tile_n, metadata in (
            ("gate/up", gate_tile_n, gate_up_metadata),
            ("down", down_tile_n, down_metadata),
        ):
            if tile_n not in (64, 128) or metadata.logical_n % tile_n:
                raise ValueError(
                    f"{projection} IQ2R tile N {tile_n} is incompatible with "
                    f"logical N={metadata.logical_n}"
                )

    return IQ2RLayerCheckpoint(
        layer_index=layer_index,
        expert_start=expert_start,
        expert_count=expert_count,
        total_experts=layout.expert_count,
        model_family=layout.model_family,
        gate_up_data=gate_data.to(device=device),
        gate_up_auxiliary=gate_auxiliary.to(device=device),
        gate_up_bias=None if gate_bias is None else gate_bias.to(device=device),
        down_data=down_data.to(device=device),
        down_auxiliary=down_auxiliary.to(device=device),
        down_bias=None if down_bias is None else down_bias.to(device=device),
        gate_up_tile_n=gate_tile_n,
        down_tile_n=down_tile_n,
        gate_up_metadata=gate_up_metadata,
        down_metadata=down_metadata,
        source_shards=tuple(dict.fromkeys((gate_up_shard.name, down_shard.name))),
    )


__all__ = [
    "IQ2R_CHECKPOINT_SCHEMA",
    "IQ2R_CHECKPOINT_SCHEMA_VERSION",
    "IQ2R_CHECKPOINT_TENSOR_RANK",
    "IQ2R_DOWN_TN",
    "IQ2R_GATE_UP_TN",
    "IQ2R_GENERIC_CHECKPOINT_SCHEMA",
    "IQ2R_GENERIC_CHECKPOINT_SCHEMA_VERSION",
    "IQ2RLayerCheckpoint",
    "iq2r_compiled_tensor_keys",
    "iq2r_glm5_overlay_keys",
    "iq2r_glm5_source_keys",
    "iq2r_gpt_oss_source_keys",
    "load_iq2r_layer_checkpoint",
]
