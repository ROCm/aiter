# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compile a block-FP8 GLM-5.3 checkpoint into the packed IQ2R checkpoint.

For each MoE layer the routed experts, plus the shared expert fused as the
last expert, are encoded to IQ2R on the GPU, packed with
``aiter.iq2r_glm53.iq2r_glm53_pack`` and written as
``iq2r-layer-NNNN-{gate-up,down}.safetensors``. A run over every MoE layer
then copies the remaining FP8 tensors into new shards and writes the index,
config.json and tokenizer files, so the output directory is a complete
checkpoint. The MTP layer keeps its FP8 experts.
A calibration artifact from ``aiter.iq2r_glm5_calibrate`` is required for a
production build.
Uniform importance is available only behind an explicit diagnostic flag and is
recorded as not O0-quality in every layer file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor

from .iq2r_glm53 import iq2r_glm53_gate_bytes, iq2r_glm53_pack
from .ops.iq2r import iq2r_encode_device
from .ops.iq2r_encoder import iq2r_learn_codebook
from .ops.iq2r_format import (
    IQ2R_ACTIVATION_BASIS,
    IQ2R_FORMAT_NAME,
    IQ2RMetadata,
)

_CALIBRATION_FORMAT = "iq2r-calibration"
_CALIBRATION_VERSION = 1
_CALIBRATION_SCHEME = "iq2r-diagonal-second-moment"
_TARGET_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.(experts|shared_experts)\."
    r"(gate_up|down)_proj\.weight$"
)
LAYOUT = "glm53-packed-v1"
_SHARD_BYTES = 5 << 30
_PROJECTIONS = {"gate_up": "gate-up", "down": "down"}
_MODEL_FILES = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "LICENSE",
)


def iq2r_compiled_tensor_keys(layer_index: int, projection: str) -> dict[str, str]:
    """Return the checkpoint keys for one stacked projection."""

    if projection == "gate_up":
        module_name = "up_gate_proj"
    elif projection == "down":
        module_name = "down_proj"
    else:
        raise ValueError(f"projection must be 'gate_up' or 'down', got {projection!r}")
    prefix = f"model.layers.{layer_index}.mlp.{module_name}.0"
    return {
        "data": f"{prefix}.iq2r_data",
        "auxiliary": f"{prefix}.iq2r_auxiliary",
        "tile_n": f"{prefix}.iq2r_tN",
    }


def _layer_file(layer: int, projection: str) -> str:
    return f"iq2r-layer-{layer:04d}-{_PROJECTIONS[projection]}.safetensors"


def _glm5_expert_keys(prefix: str) -> dict[str, str]:
    return {
        f"{projection}_{kind}": f"{prefix}.{projection}.{kind}"
        for projection in ("gate_proj", "up_proj", "down_proj")
        for kind in ("weight", "weight_scale_inv")
    }


def iq2r_glm5_source_keys(layer_index: int, expert_index: int) -> dict[str, str]:
    """Return GLM-5 block-FP8 source keys for one routed expert."""

    return _glm5_expert_keys(f"model.layers.{layer_index}.mlp.experts.{expert_index}")


def iq2r_glm5_shared_source_keys(layer_index: int) -> dict[str, str]:
    """Return GLM-5 block-FP8 source keys for its single shared expert."""

    return _glm5_expert_keys(f"model.layers.{layer_index}.mlp.shared_experts")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


@dataclass(frozen=True, slots=True)
class GLM5Layout:
    layer_count: int
    first_moe_layer: int
    expert_count: int
    hidden_size: int
    intermediate_size: int
    block_n: int
    block_k: int

    @property
    def moe_layers(self) -> int:
        return self.layer_count - self.first_moe_layer

    @property
    def compiled_experts(self) -> int:
        # The shared expert is fused as the last expert.
        return self.expert_count + 1


@dataclass(frozen=True, slots=True)
class GLM5Importance:
    """Per-MoE-layer importance: routed [layers, experts, K], shared [layers, K]."""

    gate_up: Tensor
    down: Tensor
    shared_gate_up: Tensor
    shared_down: Tensor
    first_moe_layer: int
    quality: str

    def for_projection(self, layer: int, projection: str, expert: int) -> Tensor:
        index = layer - self.first_moe_layer
        routed = self.gate_up if projection == "gate_up" else self.down
        if expert < routed.shape[1]:
            importance = routed[index, expert].float()
        else:
            shared = (
                self.shared_gate_up if projection == "gate_up" else self.shared_down
            )
            importance = shared[index].float()
        return (importance / importance.mean().clamp_min(1e-12)).clamp_min(1e-6)


def glm5_source_layout(config: dict[str, Any]) -> GLM5Layout:
    """Return the checkpoint layout for GLM-5.3."""

    architectures = config.get("architectures") or []
    if not (
        config.get("model_type") == "glm_moe_dsa"
        or any(
            isinstance(name, str) and name.startswith("GlmMoeDsa")
            for name in architectures
        )
    ):
        raise ValueError("IQ2R GLM compiler requires a GLM MoE DSA checkpoint")
    quantization = config.get("quantization_config")
    if not isinstance(quantization, dict) or quantization.get("quant_method") != "fp8":
        raise ValueError("GLM-5 IQ2R source must use block-FP8 weights")
    block_size = quantization.get("weight_block_size")
    if (
        not isinstance(block_size, list)
        or len(block_size) != 2
        or not all(isinstance(value, int) and value > 0 for value in block_size)
    ):
        raise ValueError("GLM-5 FP8 config has no valid weight_block_size")
    layout = GLM5Layout(
        layer_count=int(config.get("num_hidden_layers", -1)),
        first_moe_layer=int(config.get("first_k_dense_replace", -1)),
        expert_count=int(config.get("n_routed_experts", -1)),
        hidden_size=int(config.get("hidden_size", -1)),
        intermediate_size=int(config.get("moe_intermediate_size", -1)),
        block_n=block_size[0],
        block_k=block_size[1],
    )
    if not (0 <= layout.first_moe_layer < layout.layer_count):
        raise ValueError("GLM-5 config has an invalid routed-MoE layer range")
    if not (0 < layout.expert_count <= 512):
        raise ValueError("GLM-5 config has an invalid routed expert count")
    if config.get("n_shared_experts") != 1:
        raise ValueError("GLM-5 IQ2R requires exactly one shared expert")
    if layout.hidden_size <= 0 or layout.intermediate_size <= 0:
        raise ValueError("GLM-5 config has invalid expert dimensions")
    return layout


def _validate_importance(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if value.dtype != torch.float32 or tuple(value.shape) != shape:
        raise ValueError(
            f"IQ2R {name} importance is {value.dtype} {tuple(value.shape)}, "
            f"expected torch.float32 {shape}"
        )
    if not bool(torch.isfinite(value).all()) or bool(torch.lt(value, 0).any()):
        raise ValueError(f"IQ2R {name} importance has non-finite or negative values")


def _load_calibration_artifact(
    payload: dict[str, Any], layout: GLM5Layout
) -> GLM5Importance:
    identity = (
        payload.get("format"),
        payload.get("version"),
        payload.get("scheme"),
        payload.get("basis"),
    )
    expected = (
        _CALIBRATION_FORMAT,
        _CALIBRATION_VERSION,
        _CALIBRATION_SCHEME,
        IQ2R_ACTIVATION_BASIS,
    )
    if identity != expected:
        raise ValueError(
            f"unsupported calibration identity {identity!r}; expected {expected!r}"
        )
    targets = payload.get("targets")
    if not isinstance(targets, dict):
        raise TypeError("calibration artifact has no targets object")

    # found[module, projection][layer] is the target's [groups, K] importance.
    found: dict[tuple[str, str], dict[int, Tensor]] = {}
    for name, target in targets.items():
        match = _TARGET_PATTERN.fullmatch(name)
        if match is None or not isinstance(target, dict):
            continue
        importance = target.get("importance")
        if not isinstance(importance, Tensor):
            raise TypeError(f"calibration target {name!r} has no importance tensor")
        layer, module, projection = match.groups()
        found.setdefault((module, projection), {})[int(layer)] = importance

    layers = range(layout.first_moe_layer, layout.layer_count)
    widths = {"gate_up": layout.hidden_size, "down": layout.intermediate_size}
    stacked = {}
    for module, groups in (("experts", layout.expert_count), ("shared_experts", 1)):
        for projection, width in widths.items():
            by_layer = found.get((module, projection), {})
            if set(by_layer) != set(layers):
                raise ValueError(
                    f"calibration {module} {projection} targets cover layers "
                    f"{sorted(by_layer)}, expected {layers.start}..{layers.stop - 1}"
                )
            value = torch.stack([by_layer[layer] for layer in layers]).contiguous()
            _validate_importance(
                f"{module} {projection}", value, (len(layers), groups, width)
            )
            stacked[module, projection] = value
    return GLM5Importance(
        stacked["experts", "gate_up"],
        stacked["experts", "down"],
        stacked["shared_experts", "gate_up"][:, 0],
        stacked["shared_experts", "down"][:, 0],
        layout.first_moe_layer,
        "calibrated-o0",
    )


def load_glm5_importance(
    path: str | os.PathLike[str] | None,
    layout: GLM5Layout,
    *,
    diagnostic_uniform_importance: bool = False,
) -> GLM5Importance:
    """Load calibration, or explicitly create a diagnostic uniform cache."""

    if path is None:
        if not diagnostic_uniform_importance:
            raise ValueError(
                "production IQ2R compilation requires a calibration cache from "
                "aiter.iq2r_glm5_calibrate; "
                "use --diagnostic-uniform-importance only for kernel bring-up"
            )
        layers = layout.moe_layers
        return GLM5Importance(
            torch.ones(layers, layout.expert_count, layout.hidden_size),
            torch.ones(layers, layout.expert_count, layout.intermediate_size),
            torch.ones(layers, layout.hidden_size),
            torch.ones(layers, layout.intermediate_size),
            layout.first_moe_layer,
            "diagnostic-uniform-not-o0-quality",
        )
    if diagnostic_uniform_importance:
        raise ValueError(
            "choose either --calibration-cache or --diagnostic-uniform-importance"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("IQ2R calibration cache must contain a dictionary")
    return _load_calibration_artifact(payload, layout)


class _TensorReader:
    def __init__(self, model_dir: Path, weight_map: dict[str, str]) -> None:
        self.model_dir = model_dir
        self.weight_map = weight_map
        self.stack = ExitStack()
        self.handles: dict[str, Any] = {}

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        self.stack.close()

    def get(self, name: str) -> Tensor:
        try:
            shard = self.weight_map[name]
        except KeyError as error:
            raise KeyError(f"source index is missing {name!r}") from error
        handle = self.handles.get(shard)
        if handle is None:
            path = self.model_dir / shard
            if not path.is_file():
                raise FileNotFoundError(f"source checkpoint shard is missing: {path}")
            handle = self.stack.enter_context(
                safe_open(path, framework="pt", device="cpu")
            )
            self.handles[shard] = handle
        return handle.get_tensor(name)


def dequantize_block_fp8(
    weight: Tensor,
    scale_inv: Tensor,
    *,
    block_n: int,
    block_k: int,
    device: torch.device | str,
) -> Tensor:
    """Materialize one 2-D block-FP8 matrix as contiguous GPU FP32."""

    if weight.ndim != 2 or scale_inv.ndim != 2:
        raise ValueError("block-FP8 weight and scale must both be 2-D")
    n, k = weight.shape
    padded_n = ((n + block_n - 1) // block_n) * block_n
    padded_k = ((k + block_k - 1) // block_k) * block_k
    expected_scale_shape = (padded_n // block_n, padded_k // block_k)
    if tuple(scale_inv.shape) != expected_scale_shape:
        raise ValueError(
            f"block-FP8 scale shape {tuple(scale_inv.shape)} does not match "
            f"{expected_scale_shape} for weight {tuple(weight.shape)}"
        )
    value = weight.to(device=device, dtype=torch.float32)
    if padded_n != n or padded_k != k:
        value = torch.nn.functional.pad(value, (0, padded_k - k, 0, padded_n - n))
    value = value.reshape(
        padded_n // block_n,
        block_n,
        padded_k // block_k,
        block_k,
    )
    scale = scale_inv.to(device=device, dtype=torch.float32).reshape(
        padded_n // block_n, 1, padded_k // block_k, 1
    )
    return (value * scale).reshape(padded_n, padded_k)[:n, :k].contiguous()


def interleave_gate_up(gate: Tensor, up: Tensor) -> Tensor:
    """Return ``gate0,up0,gate1,up1,...`` rows for AITER's SwiGLU ABI."""

    if gate.shape != up.shape or gate.ndim != 2:
        raise ValueError("gate and up weights must have identical [N,K] shapes")
    return torch.stack((gate, up), dim=1).reshape(2 * gate.shape[0], gate.shape[1])


def _encode_with_scale_retry(
    weight: Tensor,
    importance: Tensor,
    *,
    iterations: int,
    sample_vectors: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    codebook = iq2r_learn_codebook(
        weight,
        importance,
        iterations=iterations,
        sample_vectors=sample_vectors,
        seed=seed,
    )
    for exponent_radius in range(17):
        try:
            return iq2r_encode_device(
                weight,
                importance,
                codebook,
                exponent_radius=exponent_radius,
            )
        except ValueError as error:
            if (
                "scale exponent range exceeds" not in str(error)
                or exponent_radius == 16
            ):
                raise


def _projection_metadata(layout: GLM5Layout, projection: str) -> IQ2RMetadata:
    if projection == "gate_up":
        return IQ2RMetadata(
            logical_n=2 * layout.intermediate_size,
            logical_k=layout.hidden_size,
        )
    if projection == "down":
        return IQ2RMetadata(
            logical_n=layout.hidden_size,
            logical_k=layout.intermediate_size,
        )
    raise ValueError(f"unknown projection {projection!r}")


def _layer_file_metadata(layer: int, projection: str, quality: str) -> dict[str, str]:
    return {
        "format": "pt",
        "iq2r_format": IQ2R_FORMAT_NAME,
        "iq2r_activation_basis": IQ2R_ACTIVATION_BASIS,
        "iq2r_layer": str(layer),
        "iq2r_projection": projection,
        "iq2r_quality": quality,
        "iq2r_layout": LAYOUT,
    }


def _source_projection(
    reader: _TensorReader,
    layout: GLM5Layout,
    layer: int,
    expert: int,
    projection: str,
    device: torch.device | str,
) -> Tensor:
    if expert == layout.expert_count:
        names = iq2r_glm5_shared_source_keys(layer)
    else:
        names = iq2r_glm5_source_keys(layer, expert)
    if projection == "gate_up":
        gate = dequantize_block_fp8(
            reader.get(names["gate_proj_weight"]),
            reader.get(names["gate_proj_weight_scale_inv"]),
            block_n=layout.block_n,
            block_k=layout.block_k,
            device=device,
        )
        up = dequantize_block_fp8(
            reader.get(names["up_proj_weight"]),
            reader.get(names["up_proj_weight_scale_inv"]),
            block_n=layout.block_n,
            block_k=layout.block_k,
            device=device,
        )
        return interleave_gate_up(gate, up).contiguous()
    return dequantize_block_fp8(
        reader.get(names["down_proj_weight"]),
        reader.get(names["down_proj_weight_scale_inv"]),
        block_n=layout.block_n,
        block_k=layout.block_k,
        device=device,
    )


def _validate_projection_shard(
    shard_path: Path,
    layout: GLM5Layout,
    layer: int,
    projection: str,
    quality: str,
) -> None:
    """Validate a packed layer file for ``--resume``."""

    metadata = _projection_metadata(layout, projection)
    data_bytes = (
        iq2r_glm53_gate_bytes(metadata.logical_n)
        if projection == "gate_up"
        else metadata.data_bytes
    )
    keys = iq2r_compiled_tensor_keys(layer, projection)
    compiled_experts = layout.compiled_experts
    expected_tensors = {
        keys["data"]: ([compiled_experts, data_bytes], "U8"),
        keys["auxiliary"]: (
            [compiled_experts, metadata.auxiliary_bytes],
            "U8",
        ),
        keys["tile_n"]: ([1], "I32"),
    }
    expected_metadata = _layer_file_metadata(layer, projection, quality)
    try:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            actual_keys = set(handle.keys())
            if actual_keys != set(expected_tensors):
                raise ValueError(
                    f"tensor keys {sorted(actual_keys)} do not match "
                    f"{sorted(expected_tensors)}"
                )
            for name, (expected_shape, expected_dtype) in expected_tensors.items():
                tensor_slice = handle.get_slice(name)
                actual_shape = tensor_slice.get_shape()
                actual_dtype = tensor_slice.get_dtype()
                if actual_shape != expected_shape or actual_dtype != expected_dtype:
                    raise ValueError(
                        f"tensor {name!r} is {actual_dtype} {actual_shape}, expected "
                        f"{expected_dtype} {expected_shape}"
                    )
            actual_metadata = handle.metadata() or {}
            for name, expected in expected_metadata.items():
                if actual_metadata.get(name) != expected:
                    raise ValueError(
                        f"metadata {name!r} is {actual_metadata.get(name)!r}, "
                        f"expected {expected!r}"
                    )
            tile_n = handle.get_tensor(keys["tile_n"])
            if tile_n.item() != 128:
                raise ValueError(f"tile_n is {tile_n.item()}, expected 128")
    except Exception as error:
        raise ValueError(f"invalid compiled shard {shard_path}: {error}") from error


def _encode_projection(
    reader: _TensorReader,
    layout: GLM5Layout,
    importance: GLM5Importance,
    layer: int,
    projection: str,
    *,
    device: torch.device | str,
    iterations: int,
    sample_vectors: int,
) -> tuple[Tensor, Tensor]:
    metadata = _projection_metadata(layout, projection)
    compiled_experts = layout.compiled_experts
    data = torch.empty((compiled_experts, metadata.data_bytes), dtype=torch.uint8)
    auxiliary = torch.empty(
        (compiled_experts, metadata.auxiliary_bytes), dtype=torch.uint8
    )
    for expert in range(compiled_experts):
        weight = _source_projection(reader, layout, layer, expert, projection, device)
        expected = (metadata.logical_n, metadata.logical_k)
        if tuple(weight.shape) != expected:
            raise ValueError(
                f"layer {layer} expert {expert} {projection} has shape "
                f"{tuple(weight.shape)}, expected {expected}"
            )
        expert_importance = (
            importance.for_projection(layer, projection, expert)
            .to(device=device, dtype=torch.float32, non_blocking=True)
            .contiguous()
        )
        encoded_data, encoded_auxiliary = _encode_with_scale_retry(
            weight,
            expert_importance,
            iterations=iterations,
            sample_vectors=sample_vectors,
            seed=0x10A0 + layer * 1024 + expert * 2 + (projection == "down"),
        )
        data[expert].copy_(encoded_data.cpu())
        auxiliary[expert].copy_(encoded_auxiliary.cpu())
        del weight, expert_importance, encoded_data, encoded_auxiliary
        if (expert + 1) % 16 == 0 or expert + 1 == compiled_experts:
            print(
                json.dumps(
                    {
                        "layer": layer,
                        "projection": projection,
                        "experts_complete": expert + 1,
                        "experts_total": compiled_experts,
                    }
                ),
                flush=True,
            )
    return data, auxiliary


def _compile_layer(
    reader: _TensorReader,
    layout: GLM5Layout,
    importance: GLM5Importance,
    layer: int,
    output_dir: Path,
    *,
    device: torch.device,
    iterations: int,
    sample_vectors: int,
) -> None:
    encoded = [
        _encode_projection(
            reader,
            layout,
            importance,
            layer,
            projection,
            device=device,
            iterations=iterations,
            sample_vectors=sample_vectors,
        )
        for projection in _PROJECTIONS
    ]
    # Returns (gate_quad, gate_auxiliary, down_data, down_auxiliary).
    packed = iq2r_glm53_pack(
        *(tensor.to(device) for pair in encoded for tensor in pair),
        intermediate_size=layout.intermediate_size,
    )
    for projection, data, auxiliary in zip(_PROJECTIONS, packed[::2], packed[1::2]):
        keys = iq2r_compiled_tensor_keys(layer, projection)
        path = output_dir / _layer_file(layer, projection)
        temporary = path.with_suffix(path.suffix + ".tmp")
        save_file(
            {
                keys["data"]: data.cpu().contiguous(),
                keys["auxiliary"]: auxiliary.cpu().contiguous(),
                keys["tile_n"]: torch.tensor([128], dtype=torch.int32),
            },
            temporary,
            metadata=_layer_file_metadata(layer, projection, importance.quality),
        )
        os.replace(temporary, path)


def _validate_device(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise ValueError("GLM IQ2R compilation requires a ROCm GPU device")
    if torch.version.hip is None:
        raise ValueError("GLM IQ2R compilation requires a ROCm PyTorch build")
    properties = torch.cuda.get_device_properties(device)
    architecture = str(getattr(properties, "gcnArchName", "")).split(":", 1)[0]
    if architecture != "gfx950":
        raise ValueError(
            f"GLM IQ2R compilation requires gfx950, got {architecture or properties.name}"
        )


def _tensor_bytes(path: Path) -> dict[str, int]:
    """Return the byte size of each tensor in a safetensors file."""

    with path.open("rb") as handle:
        header = json.loads(handle.read(int.from_bytes(handle.read(8), "little")))
    return {
        name: info["data_offsets"][1] - info["data_offsets"][0]
        for name, info in header.items()
        if name != "__metadata__"
    }


def _write_checkpoint_files(
    model_dir: Path,
    output_dir: Path,
    config: dict[str, Any],
    weight_map: dict[str, str],
    layout: GLM5Layout,
) -> Path:
    """Write the FP8 shards, index, config.json and tokenizer files.

    Every source tensor except the compiled experts is copied into new shards
    of about 5 GiB. Returns the config path.
    """

    layers = range(layout.first_moe_layer, layout.layer_count)
    compiled = {
        name
        for layer in layers
        for keys in (
            iq2r_glm5_shared_source_keys(layer),
            *(
                iq2r_glm5_source_keys(layer, expert)
                for expert in range(layout.expert_count)
            ),
        )
        for name in keys.values()
    }
    missing = compiled - weight_map.keys()
    if missing:
        raise KeyError(f"source index is missing {min(missing)!r}")

    # Fill the shards in source-shard order so each reads few source shards.
    groups: list[list[str]] = [[]]
    group_bytes = 0
    sizes: dict[str, dict[str, int]] = {}
    for shard, name in sorted(
        (shard, name) for name, shard in weight_map.items() if name not in compiled
    ):
        if shard not in sizes:
            sizes[shard] = _tensor_bytes(model_dir / shard)
        if groups[-1] and group_bytes + sizes[shard][name] > _SHARD_BYTES:
            groups.append([])
            group_bytes = 0
        groups[-1].append(name)
        group_bytes += sizes[shard][name]

    output_map: dict[str, str] = {}
    with _TensorReader(model_dir, weight_map) as reader:
        for index, names in enumerate(groups):
            shard = f"model-{index + 1:05d}-of-{len(groups):05d}.safetensors"
            path = output_dir / shard
            temporary = path.with_suffix(path.suffix + ".tmp")
            save_file(
                {name: reader.get(name) for name in names},
                temporary,
                metadata={"format": "pt"},
            )
            os.replace(temporary, path)
            output_map.update(dict.fromkeys(names, shard))
    for layer in layers:
        for projection in _PROJECTIONS:
            keys = iq2r_compiled_tensor_keys(layer, projection).values()
            output_map.update(dict.fromkeys(keys, _layer_file(layer, projection)))
    total_size = sum(
        sum(_tensor_bytes(output_dir / shard).values())
        for shard in set(output_map.values())
    )
    _atomic_write_json(
        output_dir / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": output_map},
    )

    output_config = dict(config)
    output_config["quantization_config"] = {
        "quant_method": "iq2r",
        "base_quantization_config": config["quantization_config"],
        # ATOM fnmatches patterns that contain "*". One pattern per compiled
        # layer keeps the MTP layer's experts on the FP8 path.
        "iq2r_modules": [
            f"model.layers.{layer}.mlp.{module}*"
            for module in ("experts", "shared_experts")
            for layer in layers
        ],
        "iq2r_layout": LAYOUT,
    }
    config_path = output_dir / "config.json"
    _atomic_write_json(config_path, output_config)
    for name in _MODEL_FILES:
        if (model_dir / name).is_file():
            shutil.copyfile(model_dir / name, output_dir / name)
    return config_path


def compile_glm5_iq2r(
    model_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    calibration_cache: str | os.PathLike[str] | None = None,
    diagnostic_uniform_importance: bool = False,
    device: torch.device | str = "cuda:0",
    layer_indices: list[int] | None = None,
    iterations: int = 4,
    sample_vectors: int = 65536,
    force: bool = False,
    resume: bool = False,
) -> Path | None:
    """Compile the selected MoE layers into ``output_dir``.

    A run over every MoE layer also writes the rest of the checkpoint and
    returns its config path. A run over a subset returns None.
    """

    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir).resolve()
    index_path = model_dir / "model.safetensors.index.json"
    config = _read_json(model_dir / "config.json")
    layout = glm5_source_layout(config)
    source_index = _read_json(index_path)
    weight_map = source_index.get("weight_map")
    if not isinstance(weight_map, dict) or not all(
        isinstance(name, str) and isinstance(shard, str)
        for name, shard in weight_map.items()
    ):
        raise ValueError(f"{index_path} does not contain a string weight_map object")
    if iterations <= 0 or sample_vectors <= 0:
        raise ValueError("iterations and sample_vectors must be positive")

    importance = load_glm5_importance(
        calibration_cache,
        layout,
        diagnostic_uniform_importance=diagnostic_uniform_importance,
    )
    moe_layers = list(range(layout.first_moe_layer, layout.layer_count))
    selected_layers = (
        moe_layers if layer_indices is None else sorted(set(layer_indices))
    )
    invalid = [layer for layer in selected_layers if layer not in moe_layers]
    if invalid or not selected_layers:
        raise ValueError(
            f"selected layers must be inside [{layout.first_moe_layer},"
            f"{layout.layer_count}); invalid={invalid}"
        )

    resolved_device = torch.device(device)
    _validate_device(resolved_device)
    output_dir.mkdir(parents=True, exist_ok=True)
    for layer in selected_layers:
        paths = {
            projection: output_dir / _layer_file(layer, projection)
            for projection in _PROJECTIONS
        }
        existing = [path for path in paths.values() if path.exists()]
        if existing and not force:
            if not resume:
                raise FileExistsError(
                    f"refusing to overwrite {existing[0]}; pass --resume or --force"
                )
            if len(existing) == len(paths):
                for projection, path in paths.items():
                    _validate_projection_shard(
                        path, layout, layer, projection, importance.quality
                    )
                continue
        with _TensorReader(model_dir, weight_map) as reader:
            _compile_layer(
                reader,
                layout,
                importance,
                layer,
                output_dir,
                device=resolved_device,
                iterations=iterations,
                sample_vectors=sample_vectors,
            )

    if selected_layers != moe_layers:
        return None
    return _write_checkpoint_files(model_dir, output_dir, config, weight_map, layout)


def _parse_layers(value: str) -> list[int]:
    result: list[int] = []
    for item in value.split(","):
        if "-" in item:
            start, stop = (int(part) for part in item.split("-", 1))
            if stop < start:
                raise argparse.ArgumentTypeError(f"invalid layer range {item!r}")
            result.extend(range(start, stop + 1))
        else:
            result.append(int(item))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--calibration-cache", type=Path)
    parser.add_argument("--diagnostic-uniform-importance", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--layers", type=_parse_layers)
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--sample-vectors", type=int, default=65536)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config_path = compile_glm5_iq2r(
        args.model_dir,
        args.output_dir,
        calibration_cache=args.calibration_cache,
        diagnostic_uniform_importance=args.diagnostic_uniform_importance,
        device=args.device,
        layer_indices=args.layers,
        iterations=args.iterations,
        sample_vectors=args.sample_vectors,
        force=args.force,
        resume=args.resume,
    )
    print(json.dumps({"config": config_path and str(config_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GLM5Importance",
    "GLM5Layout",
    "compile_glm5_iq2r",
    "dequantize_block_fp8",
    "glm5_source_layout",
    "interleave_gate_up",
    "load_glm5_importance",
]
