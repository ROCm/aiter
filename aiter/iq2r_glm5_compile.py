# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compile GLM-5.3 block-FP8 routed experts into AITER IQ2R tensors.

This tool intentionally writes an expert-only intermediate checkpoint. Use
``aiter.iq2r_overlay`` afterward to combine it with the immutable base model.
A Redline IQ2R calibration artifact/cache is required for a production build.
Uniform importance is available only behind an explicit diagnostic flag and is
recorded as not O0-quality in the generated config.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import torch
from torch import Tensor

from .iq2r_checkpoint import (
    iq2r_compiled_tensor_keys,
    iq2r_glm5_shared_source_keys,
    iq2r_glm5_source_keys,
)
from .ops.iq2r import iq2r_encode_device
from .ops.iq2r_encoder import iq2r_learn_codebook
from .ops.iq2r_format import (
    IQ2R_ACTIVATION_BASIS,
    IQ2R_FORMAT_NAME,
    IQ2R_FORMAT_VERSION,
    IQ2RMetadata,
)

_REDLINE_CALIBRATION_FORMAT = "redline-calibration"
_REDLINE_CALIBRATION_VERSION = 1
_REDLINE_CALIBRATION_SCHEME = "iq2r-diagonal-second-moment"
_TARGET_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.(experts|shared_experts)\."
    r"(gate_up_proj|up_gate_proj|down_proj)\.weight$"
)


def _require_safetensors():
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("GLM IQ2R compilation requires safetensors") from error
    return safe_open, save_file


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
    model_family: str = "glm5_next"
    source_root: str = "model.language_model"
    shared_expert_count: int = 0

    @property
    def moe_layers(self) -> int:
        return self.layer_count - self.first_moe_layer


@dataclass(frozen=True, slots=True)
class GLM5Importance:
    gate_up: Tensor
    down: Tensor
    metadata: dict[str, Any]
    quality: str
    shared_gate_up: Tensor | None = None
    shared_down: Tensor | None = None

    def for_projection(self, layer: int, projection: str, expert: int) -> Tensor:
        cache_layer = layer - int(self.metadata["first_moe_layer"])
        source = self.gate_up if projection == "gate_up" else self.down
        if expert < source.shape[1]:
            importance = source[cache_layer, expert].float()
        else:
            shared = (
                self.shared_gate_up if projection == "gate_up" else self.shared_down
            )
            if shared is None or expert != source.shape[1]:
                raise ValueError(
                    f"no calibrated importance for fused shared expert {expert} "
                    f"in layer {layer} projection {projection}"
                )
            importance = shared[cache_layer].float()
        return (importance / importance.mean().clamp_min(1e-12)).clamp_min(1e-6)


def glm5_source_layout(config: dict[str, Any]) -> GLM5Layout:
    """Return the checkpoint layout for Flash or plain GLM-5.3."""

    architectures = config.get("architectures") or []
    is_flash = config.get("model_type") == "glm5_next" or any(
        isinstance(name, str) and name.startswith("Glm5Next") for name in architectures
    )
    is_plain = config.get("model_type") == "glm_moe_dsa" or any(
        isinstance(name, str) and name.startswith("GlmMoeDsa") for name in architectures
    )
    if not (is_flash or is_plain):
        raise ValueError(
            "IQ2R GLM compiler requires a GLM-5 Next or GLM MoE DSA checkpoint"
        )
    if is_flash:
        text = config.get("text_config")
        if not isinstance(text, dict):
            raise TypeError("GLM-5 Next config does not contain a text_config object")
        model_family = "glm5_next"
        source_root = "model.language_model"
    else:
        text = config
        model_family = "glm_moe_dsa"
        source_root = "model"
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
        layer_count=int(text.get("num_hidden_layers", -1)),
        first_moe_layer=int(text.get("first_k_dense_replace", -1)),
        expert_count=int(text.get("n_routed_experts", -1)),
        hidden_size=int(text.get("hidden_size", -1)),
        intermediate_size=int(text.get("moe_intermediate_size", -1)),
        block_n=block_size[0],
        block_k=block_size[1],
        model_family=model_family,
        source_root=source_root,
        shared_expert_count=int(text.get("n_shared_experts", 0) or 0),
    )
    if not (0 <= layout.first_moe_layer < layout.layer_count):
        raise ValueError("GLM-5 config has an invalid routed-MoE layer range")
    if not (0 < layout.expert_count <= 512):
        raise ValueError("GLM-5 config has an invalid routed expert count")
    if layout.shared_expert_count not in (0, 1):
        raise ValueError("GLM-5 IQ2R currently supports at most one shared expert")
    if layout.hidden_size <= 0 or layout.intermediate_size <= 0:
        raise ValueError("GLM-5 config has invalid expert dimensions")
    return layout


def _validate_coverage(metadata: dict[str, Any]) -> None:
    missing = int(
        metadata.get(
            "unobserved_target_groups",
            metadata.get("unobserved_layer_experts", 0),
        )
        or 0
    )
    policy = metadata.get("unobserved_policy", "error")
    if missing and policy not in ("group-mean", "layer-mean"):
        raise ValueError(
            f"IQ2R calibration has {missing} unobserved target groups with policy "
            f"{policy!r}"
        )


def _validate_importance_shapes(
    gate_up: Tensor, down: Tensor, layout: GLM5Layout
) -> None:
    expected_gate = (
        layout.moe_layers,
        layout.expert_count,
        layout.hidden_size,
    )
    expected_down = (
        layout.moe_layers,
        layout.expert_count,
        layout.intermediate_size,
    )
    if gate_up.dtype != torch.float32 or down.dtype != torch.float32:
        raise ValueError("IQ2R importance diagonals must be float32")
    if tuple(gate_up.shape) != expected_gate or tuple(down.shape) != expected_down:
        raise ValueError(
            "IQ2R importance cache shape mismatch: "
            f"gate_up={tuple(gate_up.shape)}, down={tuple(down.shape)}, "
            f"expected_gate={expected_gate}, expected_down={expected_down}"
        )
    if not bool(torch.isfinite(gate_up).all()) or not bool(torch.isfinite(down).all()):
        raise ValueError("IQ2R importance cache contains non-finite values")
    if bool(torch.lt(gate_up, 0).any()) or bool(torch.lt(down, 0).any()):
        raise ValueError("IQ2R importance cache contains negative values")


def _validate_shared_importance_shapes(
    gate_up: Tensor | None,
    down: Tensor | None,
    layout: GLM5Layout,
) -> None:
    if gate_up is None and down is None:
        return
    if gate_up is None or down is None:
        raise ValueError(
            "IQ2R shared-expert importance must contain paired gate-up and down targets"
        )
    expected_gate = (layout.moe_layers, layout.hidden_size)
    expected_down = (layout.moe_layers, layout.intermediate_size)
    if gate_up.dtype != torch.float32 or down.dtype != torch.float32:
        raise ValueError("IQ2R shared-expert importance diagonals must be float32")
    if tuple(gate_up.shape) != expected_gate or tuple(down.shape) != expected_down:
        raise ValueError(
            "IQ2R shared-expert importance shape mismatch: "
            f"gate_up={tuple(gate_up.shape)}, down={tuple(down.shape)}, "
            f"expected_gate={expected_gate}, expected_down={expected_down}"
        )
    if not bool(torch.isfinite(gate_up).all()) or not bool(torch.isfinite(down).all()):
        raise ValueError("IQ2R shared-expert importance contains non-finite values")
    if bool(torch.lt(gate_up, 0).any()) or bool(torch.lt(down, 0).any()):
        raise ValueError("IQ2R shared-expert importance contains negative values")


def _load_direct_importance(
    payload: dict[str, Any], layout: GLM5Layout
) -> GLM5Importance:
    if (
        payload.get("format") != IQ2R_FORMAT_NAME
        or payload.get("version") != IQ2R_FORMAT_VERSION
    ):
        raise ValueError("unsupported IQ2R importance cache identity")
    metadata = dict(payload.get("metadata") or {})
    expected = (IQ2R_FORMAT_NAME, IQ2R_FORMAT_VERSION, IQ2R_ACTIVATION_BASIS)
    actual = (
        metadata.get("format"),
        metadata.get("version"),
        metadata.get("activation_basis"),
    )
    if actual != expected:
        raise ValueError(
            f"importance metadata identity {actual!r} does not match {expected!r}"
        )
    _validate_coverage(metadata)
    gate_up = payload["gate_up"].contiguous()
    down = payload["down"].contiguous()
    _validate_importance_shapes(gate_up, down, layout)
    shared_gate_up = payload.get("shared_gate_up")
    shared_down = payload.get("shared_down")
    if isinstance(shared_gate_up, Tensor):
        shared_gate_up = shared_gate_up.contiguous()
    if isinstance(shared_down, Tensor):
        shared_down = shared_down.contiguous()
    _validate_shared_importance_shapes(shared_gate_up, shared_down, layout)
    metadata["first_moe_layer"] = layout.first_moe_layer
    return GLM5Importance(
        gate_up,
        down,
        metadata,
        "calibrated-o0",
        shared_gate_up,
        shared_down,
    )


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
        _REDLINE_CALIBRATION_FORMAT,
        _REDLINE_CALIBRATION_VERSION,
        _REDLINE_CALIBRATION_SCHEME,
        IQ2R_ACTIVATION_BASIS,
    )
    if identity != expected:
        raise ValueError(
            f"unsupported Redline calibration identity {identity!r}; expected {expected!r}"
        )
    metadata = dict(payload.get("metadata") or {})
    _validate_coverage(metadata)
    targets = payload.get("targets")
    if not isinstance(targets, dict):
        raise TypeError("Redline calibration artifact has no targets object")

    projections: dict[str, dict[int, Tensor]] = {"gate_up": {}, "down": {}}
    shared_projections: dict[str, dict[int, Tensor]] = {
        "gate_up": {},
        "down": {},
    }
    for name, target in targets.items():
        match = _TARGET_PATTERN.fullmatch(name)
        if match is None or not isinstance(target, dict):
            continue
        layer = int(match.group(1))
        module = match.group(2)
        projection = "down" if match.group(3) == "down_proj" else "gate_up"
        importance = target.get("importance")
        if not isinstance(importance, Tensor):
            raise TypeError(f"calibration target {name!r} has no importance tensor")
        destination = shared_projections if module == "shared_experts" else projections
        destination[projection][layer] = importance

    gate_layers = set(projections["gate_up"])
    if not gate_layers or gate_layers != set(projections["down"]):
        raise ValueError(
            "Redline calibration must contain paired gate-up and down targets"
        )
    actual_layers = set(range(layout.first_moe_layer, layout.layer_count))
    compact_layers = set(range(layout.moe_layers))
    if gate_layers == actual_layers:
        ordered_layers = range(layout.first_moe_layer, layout.layer_count)
    elif gate_layers == compact_layers:
        ordered_layers = range(layout.moe_layers)
    else:
        raise ValueError(
            "Redline calibration layers do not match GLM routed layers: "
            f"got {sorted(gate_layers)}"
        )
    gate_up = torch.stack(
        [projections["gate_up"][layer] for layer in ordered_layers]
    ).contiguous()
    down = torch.stack(
        [projections["down"][layer] for layer in ordered_layers]
    ).contiguous()
    _validate_importance_shapes(gate_up, down, layout)
    shared_gate_layers = set(shared_projections["gate_up"])
    shared_down_layers = set(shared_projections["down"])
    if shared_gate_layers != shared_down_layers:
        raise ValueError(
            "Redline calibration must contain paired shared gate-up and down targets"
        )
    shared_gate_up = None
    shared_down = None
    if shared_gate_layers:
        if shared_gate_layers == actual_layers:
            shared_ordered_layers = range(layout.first_moe_layer, layout.layer_count)
        elif shared_gate_layers == compact_layers:
            shared_ordered_layers = range(layout.moe_layers)
        else:
            raise ValueError(
                "Redline shared-expert calibration layers do not match GLM layers: "
                f"got {sorted(shared_gate_layers)}"
            )
        shared_gate_up = torch.stack(
            [shared_projections["gate_up"][layer] for layer in shared_ordered_layers]
        ).contiguous()
        shared_down = torch.stack(
            [shared_projections["down"][layer] for layer in shared_ordered_layers]
        ).contiguous()
    _validate_shared_importance_shapes(shared_gate_up, shared_down, layout)
    metadata.update(
        {
            "calibration_format": _REDLINE_CALIBRATION_FORMAT,
            "calibration_version": _REDLINE_CALIBRATION_VERSION,
            "calibration_scheme": _REDLINE_CALIBRATION_SCHEME,
            "first_moe_layer": layout.first_moe_layer,
        }
    )
    return GLM5Importance(
        gate_up,
        down,
        metadata,
        "calibrated-o0",
        shared_gate_up,
        shared_down,
    )


def load_glm5_importance(
    path: str | os.PathLike[str] | None,
    layout: GLM5Layout,
    *,
    diagnostic_uniform_importance: bool = False,
) -> GLM5Importance:
    """Load Redline calibration, or explicitly create a diagnostic uniform cache."""

    if path is None:
        if not diagnostic_uniform_importance:
            raise ValueError(
                "production IQ2R compilation requires a Redline calibration cache; "
                "use --diagnostic-uniform-importance only for kernel bring-up"
            )
        gate_up = torch.ones(
            layout.moe_layers,
            layout.expert_count,
            layout.hidden_size,
            dtype=torch.float32,
        )
        down = torch.ones(
            layout.moe_layers,
            layout.expert_count,
            layout.intermediate_size,
            dtype=torch.float32,
        )
        return GLM5Importance(
            gate_up,
            down,
            {
                "first_moe_layer": layout.first_moe_layer,
                "calibration_scheme": "diagnostic-uniform",
                "warning": "not O0 quality",
            },
            "diagnostic-uniform-not-o0-quality",
            (
                torch.ones(
                    layout.moe_layers,
                    layout.hidden_size,
                    dtype=torch.float32,
                )
                if layout.shared_expert_count
                else None
            ),
            (
                torch.ones(
                    layout.moe_layers,
                    layout.intermediate_size,
                    dtype=torch.float32,
                )
                if layout.shared_expert_count
                else None
            ),
        )
    if diagnostic_uniform_importance:
        raise ValueError(
            "choose either --calibration-cache or --diagnostic-uniform-importance"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError("IQ2R calibration cache must contain a dictionary")
    if payload.get("format") == _REDLINE_CALIBRATION_FORMAT:
        return _load_calibration_artifact(payload, layout)
    return _load_direct_importance(payload, layout)


class _TensorReader:
    def __init__(
        self,
        model_dir: Path,
        weight_map: dict[str, str],
        safe_open,
    ) -> None:
        self.model_dir = model_dir
        self.weight_map = weight_map
        self.safe_open = safe_open
        self.stack = ExitStack()
        self.handles: dict[str, Any] = {}

    def __enter__(self) -> Self:
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
                self.safe_open(path, framework="pt", device="cpu")
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
    raise AssertionError("unreachable")


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


def _compiled_expert_count(layout: GLM5Layout, fuse_shared_expert: bool) -> int:
    if not fuse_shared_expert:
        return layout.expert_count
    if layout.model_family != "glm_moe_dsa" or layout.shared_expert_count != 1:
        raise ValueError(
            "fused IQ2R shared-expert compilation currently requires plain "
            "GLM MoE DSA with exactly one shared expert"
        )
    return layout.expert_count + 1


def _source_projection(
    reader: _TensorReader,
    layout: GLM5Layout,
    layer: int,
    expert: int,
    projection: str,
    device: torch.device | str,
    *,
    fuse_shared_expert: bool = False,
) -> Tensor:
    if expert == layout.expert_count and fuse_shared_expert:
        names = iq2r_glm5_shared_source_keys(layer, root=layout.source_root)
    else:
        names = iq2r_glm5_source_keys(layer, expert, root=layout.source_root)
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
    safe_open,
    *,
    fuse_shared_expert: bool = False,
) -> None:
    """Validate an existing projection shard before accepting ``--resume``."""

    metadata = _projection_metadata(layout, projection)
    module_name = "up_gate_proj" if projection == "gate_up" else "down_proj"
    keys = iq2r_compiled_tensor_keys(layer, projection, module_name=module_name)
    compiled_experts = _compiled_expert_count(layout, fuse_shared_expert)
    expected_tensors = {
        keys["data"]: ([compiled_experts, metadata.data_bytes], "U8"),
        keys["auxiliary"]: (
            [compiled_experts, metadata.auxiliary_bytes],
            "U8",
        ),
        keys["tile_n"]: ([1], "I32"),
    }
    expected_metadata = {
        "format": "pt",
        "iq2r_format": IQ2R_FORMAT_NAME,
        "iq2r_format_version": str(IQ2R_FORMAT_VERSION),
        "iq2r_activation_basis": IQ2R_ACTIVATION_BASIS,
        "iq2r_layer": str(layer),
        "iq2r_projection": projection,
        "iq2r_quality": quality,
    }
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
        if isinstance(error, ValueError) and str(error).startswith(
            f"invalid resume shard {shard_path}:"
        ):
            raise
        raise ValueError(f"invalid resume shard {shard_path}: {error}") from error


def _compile_projection(
    reader: _TensorReader,
    layout: GLM5Layout,
    importance: GLM5Importance,
    layer: int,
    projection: str,
    output_dir: Path,
    *,
    device: torch.device | str,
    iterations: int,
    sample_vectors: int,
    force: bool,
    resume: bool,
    safe_open,
    save_file,
    fuse_shared_expert: bool = False,
) -> str:
    suffix = "gate-up" if projection == "gate_up" else "down"
    shard_name = f"iq2r-layer-{layer:04d}-{suffix}.safetensors"
    shard_path = output_dir / shard_name
    if shard_path.exists():
        if resume:
            _validate_projection_shard(
                shard_path,
                layout,
                layer,
                projection,
                importance.quality,
                safe_open,
                fuse_shared_expert=fuse_shared_expert,
            )
            return shard_name
        if not force:
            raise FileExistsError(f"refusing to overwrite {shard_path}; pass --force")

    metadata = _projection_metadata(layout, projection)
    compiled_experts = _compiled_expert_count(layout, fuse_shared_expert)
    data = torch.empty((compiled_experts, metadata.data_bytes), dtype=torch.uint8)
    auxiliary = torch.empty(
        (compiled_experts, metadata.auxiliary_bytes), dtype=torch.uint8
    )
    for expert in range(compiled_experts):
        weight = _source_projection(
            reader,
            layout,
            layer,
            expert,
            projection,
            device,
            fuse_shared_expert=fuse_shared_expert,
        )
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

    module_name = "up_gate_proj" if projection == "gate_up" else "down_proj"
    keys = iq2r_compiled_tensor_keys(layer, projection, module_name=module_name)
    tensors = {
        keys["data"]: data,
        keys["auxiliary"]: auxiliary,
        keys["tile_n"]: torch.tensor([128], dtype=torch.int32),
    }
    temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    save_file(
        tensors,
        temporary,
        metadata={
            "format": "pt",
            "iq2r_format": IQ2R_FORMAT_NAME,
            "iq2r_format_version": str(IQ2R_FORMAT_VERSION),
            "iq2r_activation_basis": IQ2R_ACTIVATION_BASIS,
            "iq2r_layer": str(layer),
            "iq2r_projection": projection,
            "iq2r_quality": importance.quality,
        },
    )
    os.replace(temporary, shard_path)
    return shard_name


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


def _compiled_config(
    source_config: dict[str, Any],
    layout: GLM5Layout,
    importance: GLM5Importance,
    selected_layers: list[int],
    file_manifest: list[str],
    source_model: str,
    source_config_sha256: str,
    fuse_shared_expert: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "architectures": list(source_config.get("architectures") or []),
        "model_type": layout.model_family,
        "compiled_tensor_parallel_size": 1,
        "compiled_expert_parallel_size": 1,
        "storage_format": "safetensors",
        "profile": "o0" if importance.quality == "calibrated-o0" else "diagnostic",
        "file_manifest": file_manifest,
        "iq2r": {
            "format": IQ2R_FORMAT_NAME,
            "version": IQ2R_FORMAT_VERSION,
            "activation_basis": IQ2R_ACTIVATION_BASIS,
            "expert_projection_modules": {
                "gate_up": "up_gate_proj",
                "down": "down_proj",
            },
            "expert_bias": False,
            "gate_up_row_order": "gate-up-interleaved",
            "model_family": layout.model_family,
            "source_root": layout.source_root,
            "quality": importance.quality,
            "compiled_expert_count": _compiled_expert_count(layout, fuse_shared_expert),
            "fused_shared_expert": fuse_shared_expert,
            "compiled_layers": selected_layers,
            "source_model": source_model,
            "source_config_sha256": source_config_sha256,
            "calibration": importance.metadata,
        },
    }
    dimensions = {
        "num_hidden_layers": layout.layer_count,
        "first_k_dense_replace": layout.first_moe_layer,
        "n_routed_experts": layout.expert_count,
        "hidden_size": layout.hidden_size,
        "moe_intermediate_size": layout.intermediate_size,
    }
    if layout.model_family == "glm5_next":
        result["text_config"] = {
            **dict(source_config["text_config"]),
            **dimensions,
        }
    else:
        result.update(dimensions)
    return result


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
    fuse_shared_expert: bool = False,
    diagnostic_shared_importance: bool = False,
) -> Path:
    """Compile GLM routed experts and return the intermediate config path."""

    safe_open, save_file = _require_safetensors()
    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir).resolve()
    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config = _read_json(config_path)
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
    _compiled_expert_count(layout, fuse_shared_expert)
    if fuse_shared_expert and (
        importance.shared_gate_up is None or importance.shared_down is None
    ):
        if not diagnostic_shared_importance:
            raise ValueError(
                "fused shared-expert compilation requires shared-expert "
                "calibration; use --diagnostic-shared-importance only for "
                "performance bring-up"
            )
        importance = GLM5Importance(
            importance.gate_up,
            importance.down,
            {
                **importance.metadata,
                "shared_expert_importance": "diagnostic-uniform",
                "warning": "shared expert is not O0 calibrated",
            },
            "calibrated-routed-diagnostic-shared-not-o0-quality",
            torch.ones(layout.moe_layers, layout.hidden_size, dtype=torch.float32),
            torch.ones(
                layout.moe_layers, layout.intermediate_size, dtype=torch.float32
            ),
        )
    selected_layers = (
        list(range(layout.first_moe_layer, layout.layer_count))
        if layer_indices is None
        else sorted(set(layer_indices))
    )
    invalid = [
        layer
        for layer in selected_layers
        if not (layout.first_moe_layer <= layer < layout.layer_count)
    ]
    if invalid or not selected_layers:
        raise ValueError(
            f"selected layers must be inside [{layout.first_moe_layer},"
            f"{layout.layer_count}); invalid={invalid}"
        )

    resolved_device = torch.device(device)
    _validate_device(resolved_device)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_manifest: list[str] = []
    for layer in selected_layers:
        with _TensorReader(model_dir, weight_map, safe_open) as reader:
            file_manifest.append(
                _compile_projection(
                    reader,
                    layout,
                    importance,
                    layer,
                    "gate_up",
                    output_dir,
                    device=resolved_device,
                    iterations=iterations,
                    sample_vectors=sample_vectors,
                    force=force,
                    resume=resume,
                    safe_open=safe_open,
                    save_file=save_file,
                    fuse_shared_expert=fuse_shared_expert,
                )
            )
            file_manifest.append(
                _compile_projection(
                    reader,
                    layout,
                    importance,
                    layer,
                    "down",
                    output_dir,
                    device=resolved_device,
                    iterations=iterations,
                    sample_vectors=sample_vectors,
                    force=force,
                    resume=resume,
                    safe_open=safe_open,
                    save_file=save_file,
                    fuse_shared_expert=fuse_shared_expert,
                )
            )

    source_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    compiled_config = _compiled_config(
        config,
        layout,
        importance,
        selected_layers,
        file_manifest,
        str(model_dir),
        source_config_sha256,
        fuse_shared_expert,
    )
    output_config = output_dir / "config.json"
    if output_config.exists() and not (force or resume):
        raise FileExistsError(
            f"refusing to overwrite {output_config}; pass --force or --resume"
        )
    _atomic_write_json(output_config, compiled_config)
    return output_config


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
    parser.add_argument("--fuse-shared-expert", action="store_true")
    parser.add_argument("--diagnostic-shared-importance", action="store_true")
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
        fuse_shared_expert=args.fuse_shared_expert,
        diagnostic_shared_importance=args.diagnostic_shared_importance,
    )
    print(json.dumps({"config": str(config_path)}))
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
