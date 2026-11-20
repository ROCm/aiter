# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compile a block-FP8 GLM-5.3 checkpoint into the packed IQ2R-v2 checkpoint.

For each MoE layer the 256 routed experts are encoded in IQ2R-v2
(``aiter/ops/iq2r_v2.py``) on the GPU, packed with
``aiter.iq2r_glm53.iq2r_glm53_pack`` and written as
``iq2r-layer-NNNN-{gate-up,down}.safetensors``. The shared expert and the MTP
layer keep their block-FP8 weights. A run over every MoE layer then copies the
remaining FP8 tensors into new shards and writes the index, config.json and
tokenizer files, so the output directory is a complete checkpoint.

Importance is the router-weighted second moment from the calibration artifact
of ``aiter.quantization.iq2r.glm5.calibrate``; an expert that fewer than 256
corpus tokens reach uses its weight's column energy instead.
``docs/iq2r_glm53.md`` describes the recipe. Build:

    python3 -m aiter.quantization.iq2r.glm5.calibrate --model-dir $FP8 \\
        --text-file corpus.jsonl --output calibration.pt
    # one process per GPU, each on its own MoE layers (e.g. 3-12, 13-22, ...)
    HIP_VISIBLE_DEVICES=$i python3 -m aiter.quantization.iq2r.glm5.compile \\
        --model-dir $FP8 --output-dir $OUT --calibration-cache calibration.pt \\
        --layers ...
    # then one run without --layers writes the shards, index and config
    python3 -m aiter.quantization.iq2r.glm5.compile --model-dir $FP8 \\
        --output-dir $OUT --calibration-cache calibration.pt

``--codebook-entries 1024`` builds the 1,024-entry variant (10-bit indices,
layout ``glm53-packed-v3``); the default is 512 entries.

``--ldlq --hessian-dir DIR`` re-chooses the gate/up codewords and signs by LDLQ
(``iq2r_v2_ldlq``) after the normal encode, with each layer's Hessians from
``calibrate --hessian-dir``: ``H = H_all / tr + 1.0 * H_decision / tr``,
damped by ``1.0 * mean(diag)``; the gains are fitted to the result. Scales,
format and kernels are unchanged; down is not re-encoded.

Each process encodes on its first visible GPU; select it with
HIP_VISIBLE_DEVICES. Layer files already in the output directory are checked
and kept if they were built from the same calibration and recipe; any other
existing layer file is an error, and has to be deleted to be rebuilt.
"""

from __future__ import annotations

import argparse
import hashlib
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

from aiter.iq2r_glm53 import (
    EXPERTS,
    HIDDEN,
    TOPK,
    iq2r_glm53_aux_bytes,
    iq2r_glm53_gate_bytes,
    iq2r_glm53_pack,
    iq2r_glm53_unpack,
)
from aiter.ops.iq2r_format import IQ2R_ACTIVATION_BASIS, IQ2RMetadata
from aiter.ops.iq2r_v2 import (
    IQ2R_V2_LDLQ_DAMP,
    IQ2R_V2_LEVELS_C4,
    IQ2RV2LdlqFactor,
    iq2r_v2_format_name,
    iq2r_v2_gains,
    iq2r_v2_ldlq,
    iq2r_v2_ldlq_factor,
    iq2r_v2_learn_codebook,
    iq2r_v2_materialize,
    iq2r_v2_write_auxiliary,
    iq2r_v2_write_data,
)

_CALIBRATION_FORMAT = "iq2r-calibration"
_CALIBRATION_VERSION = 1
_CALIBRATION_SCHEME = "iq2r-diagonal-second-moment"
# An artifact may also hold shared-expert targets; they are hashed but not used
# (see _load_calibration_artifact).
_TARGET_PATTERN = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.(experts|shared_experts)\."
    r"(gate_up|down)_proj\.weight$"
)
LAYOUT = "glm53-packed-v2"
# Layout of a checkpoint per codebook size.
LAYOUTS = {512: LAYOUT, 1024: "glm53-packed-v3"}
# The recipe: the kernels' level set, the reconstruction gain <W, Wq> / <W, W>,
# the routed-token count below which an expert's importance falls back to its
# weight energy, and the codebook learner's rounds and sample size.
_LEVELS = IQ2R_V2_LEVELS_C4
FORMAT = iq2r_v2_format_name(_LEVELS)
_CODEBOOK_ENTRIES = 512
_TARGET_GAIN = 0.98
_MIN_ROUTED_TOKENS = 256
_ITERATIONS = 4
_SAMPLE_VECTORS = 65536
# 256-weight blocks per encoder pass; the result does not depend on it.
_CHUNK_BLOCKS = 65536
_DEVICE = "cuda:0"

# The serving kernels are built for this shape. ATOM slices the 2048-wide
# experts into the TP4 (512) and TP8 (256) shards they cover.
_GLM53_SHAPE = {
    "hidden_size": HIDDEN,
    "moe_intermediate_size": 2048,
    "n_routed_experts": EXPERTS,
    "num_experts_per_tok": TOPK,
}
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


def iq2r_glm5_source_keys(layer_index: int, expert_index: int) -> dict[str, str]:
    """Return GLM-5 block-FP8 source keys for one routed expert."""

    prefix = f"model.layers.{layer_index}.mlp.experts.{expert_index}"
    return {
        f"{projection}_{kind}": f"{prefix}.{projection}.{kind}"
        for projection in ("gate_proj", "up_proj", "down_proj")
        for kind in ("weight", "weight_scale_inv")
    }


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


@dataclass(frozen=True, slots=True)
class GLM5Importance:
    """Per-MoE-layer routed-expert importance [layers, experts, K].

    ``sha256`` identifies the calibration ("uniform" for the diagnostic mode).
    """

    gate_up: Tensor
    down: Tensor
    first_moe_layer: int
    quality: str
    sha256: str
    # Corpus tokens that reached each routed expert [layers, experts] (forced
    # fills excluded); None for the diagnostic uniform importance.
    routed_hits: Tensor | None = None

    def routed_tokens(self, layer: int, expert: int) -> int | None:
        if self.routed_hits is None:
            return None
        return int(self.routed_hits[layer - self.first_moe_layer, expert])

    def for_projection(self, layer: int, projection: str, expert: int) -> Tensor:
        routed = self.gate_up if projection == "gate_up" else self.down
        importance = routed[layer - self.first_moe_layer, expert].float()
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
    if config.get("n_shared_experts") != 1:
        raise ValueError("GLM-5 IQ2R requires exactly one shared expert")
    mismatched = {
        name: config.get(name)
        for name, value in _GLM53_SHAPE.items()
        if config.get(name) != value
    }
    if mismatched:
        raise ValueError(
            f"the GLM-5.3 IQ2R kernels need {_GLM53_SHAPE}; this config has "
            f"{mismatched}"
        )
    return layout


# LDLQ (--ldlq): weight of the decision bucket and the damping.
_LDLQ_BETA = 1.0
_LDLQ_DAMP = IQ2R_V2_LDLQ_DAMP
_HESSIAN_FORMAT = "iq2r-layer-hessian"
_HESSIAN_VERSION = 1
_HESSIAN_MANIFEST = "hessians.json"


@dataclass(frozen=True)
class GLM5Hessians:
    """Per-MoE-layer input Hessians from ``calibrate --hessian-dir``.

    ``sha256`` identifies the captures (the manifest's file digests).
    """

    directory: Path
    manifest: dict[str, Any]
    sha256: str

    def factor(self, layer: int, device: torch.device | str) -> IQ2RV2LdlqFactor:
        name = f"hessian-layer-{layer:04d}.pt"
        payload = torch.load(
            self.directory / name, map_location="cpu", weights_only=True
        )
        digest = hashlib.sha256()
        for bucket in sorted(self.manifest["buckets"]):
            digest.update(payload[bucket].numpy().tobytes())
        if digest.hexdigest() != self.manifest["files"][name]:
            raise ValueError(f"{name} does not match the digest in {_HESSIAN_MANIFEST}")
        all_ = payload["all"].to(device=device, dtype=torch.float64)
        hessian = all_ / all_.trace()
        if "decision" in payload:
            decision = payload["decision"].to(device=device, dtype=torch.float64)
            hessian = hessian + _LDLQ_BETA * decision / decision.trace()
        return iq2r_v2_ldlq_factor(hessian, _LDLQ_DAMP)


def load_glm5_hessians(
    directory: str | os.PathLike[str], layout: GLM5Layout
) -> GLM5Hessians:
    directory = Path(directory)
    manifest = _read_json(directory / _HESSIAN_MANIFEST)
    if (manifest.get("format"), manifest.get("version")) != (
        _HESSIAN_FORMAT,
        _HESSIAN_VERSION,
    ):
        raise ValueError(f"{directory} does not hold {_HESSIAN_FORMAT} captures")
    if "all" not in manifest.get("buckets", ()):
        raise ValueError("the Hessian captures have no 'all' bucket")
    expected = {
        f"hessian-layer-{layer:04d}.pt"
        for layer in range(layout.first_moe_layer, layout.layer_count)
    }
    if set(manifest.get("files", {})) != expected:
        raise ValueError(
            f"the Hessian captures cover {sorted(manifest.get('files', {}))}, "
            f"expected layers {layout.first_moe_layer}..{layout.layer_count - 1}"
        )
    digest = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True).encode()
    ).hexdigest()
    return GLM5Hessians(directory, manifest, digest)


def _validate_importance(name: str, value: Tensor, shape: tuple[int, ...]) -> None:
    if value.dtype != torch.float32 or tuple(value.shape) != shape:
        raise ValueError(
            f"IQ2R {name} importance is {value.dtype} {tuple(value.shape)}, "
            f"expected torch.float32 {shape}"
        )
    if not bool(torch.isfinite(value).all()) or bool(torch.lt(value, 0).any()):
        raise ValueError(f"IQ2R {name} importance has non-finite or negative values")


def _importance_sha256(tensors: list[Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.contiguous().view(torch.uint8).numpy())
    return digest.hexdigest()


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
    hits: dict[int, Tensor] = {}
    for name, target in targets.items():
        match = _TARGET_PATTERN.fullmatch(name)
        if match is None or not isinstance(target, dict):
            continue
        importance = target.get("importance")
        if not isinstance(importance, Tensor):
            raise TypeError(f"calibration target {name!r} has no importance tensor")
        layer, module, projection = match.groups()
        found.setdefault((module, projection), {})[int(layer)] = importance
        if module == "experts" and projection == "gate_up":
            natural = target.get("hit_count")
            if isinstance(natural, Tensor):
                forced = target.get("forced_hit_count")
                hits[int(layer)] = natural - (
                    forced if isinstance(forced, Tensor) else 0
                )

    layers = range(layout.first_moe_layer, layout.layer_count)
    widths = {"gate_up": layout.hidden_size, "down": layout.intermediate_size}
    routed = {}
    for projection, width in widths.items():
        by_layer = found.get(("experts", projection), {})
        if set(by_layer) != set(layers):
            raise ValueError(
                f"calibration experts {projection} targets cover layers "
                f"{sorted(by_layer)}, expected {layers.start}..{layers.stop - 1}"
            )
        value = torch.stack([by_layer[layer] for layer in layers]).contiguous()
        _validate_importance(
            f"experts {projection}", value, (len(layers), layout.expert_count, width)
        )
        routed[projection] = value
    # Shared-expert targets are not used, but an artifact that has them hashes
    # them after the routed ones, as the layer files built from it record.
    shared = [
        found[key][layer]
        for key in (("shared_experts", "gate_up"), ("shared_experts", "down"))
        for layer in sorted(found.get(key, ()))
    ]
    return GLM5Importance(
        routed["gate_up"],
        routed["down"],
        layout.first_moe_layer,
        "calibrated-o0",
        _importance_sha256([routed["gate_up"], routed["down"], *shared]),
        (
            torch.stack([hits[layer] for layer in layers])
            if set(hits) == set(layers)
            else None
        ),
    )


def load_glm5_importance(
    path: str | os.PathLike[str] | None,
    layout: GLM5Layout,
    *,
    diagnostic_uniform_importance: bool = False,
) -> GLM5Importance:
    """Load calibration, or (tests only) create a diagnostic uniform importance."""

    if path is None:
        if not diagnostic_uniform_importance:
            raise ValueError(
                "IQ2R compilation requires a calibration cache from "
                "aiter.quantization.iq2r.glm5.calibrate"
            )
        layers = layout.moe_layers
        return GLM5Importance(
            torch.ones(layers, layout.expert_count, layout.hidden_size),
            torch.ones(layers, layout.expert_count, layout.intermediate_size),
            layout.first_moe_layer,
            "diagnostic-uniform-not-o0-quality",
            "uniform",
        )
    if diagnostic_uniform_importance:
        raise ValueError("choose either a calibration cache or uniform importance")
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
    """Materialize one 2-D block-FP8 matrix as contiguous FP32 on ``device``."""

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


def _projection_metadata(
    layout: GLM5Layout, projection: str, entries: int = _CODEBOOK_ENTRIES
) -> IQ2RMetadata:
    if projection == "gate_up":
        return IQ2RMetadata(
            logical_n=2 * layout.intermediate_size,
            logical_k=layout.hidden_size,
            codebook_entries=entries,
        )
    if projection == "down":
        return IQ2RMetadata(
            logical_n=layout.hidden_size,
            logical_k=layout.intermediate_size,
            codebook_entries=entries,
        )
    raise ValueError(f"unknown projection {projection!r}")


def _encoding_metadata(
    importance: GLM5Importance,
    iterations: int,
    sample_vectors: int,
    entries: int = _CODEBOOK_ENTRIES,
    hessians: GLM5Hessians | None = None,
) -> dict[str, str]:
    """What decides the encoded bytes besides the source weights.

    Every layer file records it, so a later run keeps only layers built the
    same way. 512-entry builds without LDLQ keep the shipped keys.
    """

    extra = {} if entries == 512 else {"iq2r_codebook_entries": str(entries)}
    if hessians is not None:
        extra["iq2r_ldlq"] = json.dumps(
            {
                "projections": ["gate_up"],
                "beta_decision": _LDLQ_BETA,
                "damp": _LDLQ_DAMP,
                "buckets": sorted(hessians.manifest["buckets"]),
                "hessians_sha256": hessians.sha256,
            },
            sort_keys=True,
        )
    return {
        **extra,
        "iq2r_quality": importance.quality,
        "iq2r_calibration_sha256": importance.sha256,
        "iq2r_codebook_iterations": str(iterations),
        "iq2r_codebook_sample_vectors": str(sample_vectors),
        "iq2r_v2_levels": _LEVELS.name,
        "iq2r_v2_target_gain": repr(_TARGET_GAIN),
        "iq2r_v2_min_routed_tokens": str(_MIN_ROUTED_TOKENS),
        "iq2r_v2_shared_expert": "fp8",
    }


def _encoding_entries(encoding: dict[str, str]) -> int:
    return int(encoding.get("iq2r_codebook_entries", 512))


def _layer_file_metadata(
    layer: int, projection: str, encoding: dict[str, str]
) -> dict[str, str]:
    entries = _encoding_entries(encoding)
    return {
        "format": "pt",
        "iq2r_format": iq2r_v2_format_name(_LEVELS, entries),
        "iq2r_activation_basis": IQ2R_ACTIVATION_BASIS,
        "iq2r_layer": str(layer),
        "iq2r_projection": projection,
        **encoding,
        "iq2r_layout": LAYOUTS[entries],
    }


def _source_projection(
    reader: _TensorReader,
    layout: GLM5Layout,
    layer: int,
    expert: int,
    projection: str,
    device: torch.device | str,
) -> Tensor:
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
    encoding: dict[str, str],
) -> None:
    """Check that an existing layer file is the one this run would write."""

    entries = _encoding_entries(encoding)
    metadata = _projection_metadata(layout, projection, entries)
    data_bytes = (
        iq2r_glm53_gate_bytes(metadata.logical_n, entries)
        if projection == "gate_up"
        else metadata.data_bytes
    )
    auxiliary_bytes = iq2r_glm53_aux_bytes(
        metadata.logical_n,
        metadata.logical_k,
        _LEVELS.classes,
        gate=projection == "gate_up",
        codebook_entries=entries,
    )
    keys = iq2r_compiled_tensor_keys(layer, projection)
    expected_tensors = {
        keys["data"]: ([layout.expert_count, data_bytes], "U8"),
        keys["auxiliary"]: ([layout.expert_count, auxiliary_bytes], "U8"),
        keys["tile_n"]: ([1], "I32"),
    }
    expected_metadata = _layer_file_metadata(layer, projection, encoding)
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
        raise ValueError(
            f"existing layer file {shard_path} does not match this build ({error}); "
            "delete it to rebuild the layer"
        ) from error


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
    entries: int = _CODEBOOK_ENTRIES,
    ldlq: IQ2RV2LdlqFactor | None = None,
) -> tuple[Tensor, Tensor]:
    """Canonical IQ2R-v2 buffers ``(data, auxiliary)`` of every routed expert.

    ``ldlq``: the layer's factor; gate/up codewords are re-chosen with it.
    """
    metadata = _projection_metadata(layout, projection, entries)
    data_rows, aux_rows = [], []
    for expert in range(layout.expert_count):
        # gate/up is the interleaved matrix (gate0, up0, ...) with one codebook;
        # its gains are fitted to the gate and the up rows separately.
        weight = _source_projection(reader, layout, layer, expert, projection, device)
        tokens = importance.routed_tokens(layer, expert)
        if tokens is not None and tokens < _MIN_ROUTED_TOKENS:
            channel = weight.square().sum(dim=0)
            expert_importance = (channel / channel.mean().clamp_min(1e-12)).clamp_min(
                1e-6
            )
        else:
            expert_importance = importance.for_projection(layer, projection, expert).to(
                device=device, dtype=torch.float32, non_blocking=True
            )
        expert_importance = expert_importance.float().contiguous()
        # The learner returns the matrix's encoding with the codebook it keeps.
        book, enc = iq2r_v2_learn_codebook(
            weight,
            expert_importance,
            _LEVELS,
            rounds=iterations,
            sample_vectors=sample_vectors,
            seed=(0x10A0 + layer * 1024 + expert * 2 + (projection == "down")) * 2,
            chunk_blocks=_CHUNK_BLOCKS,
            entries=entries,
        )
        if ldlq is not None and projection == "gate_up":
            enc = iq2r_v2_ldlq(weight, enc, book, ldlq, _LEVELS)
        gains = iq2r_v2_gains(
            weight,
            enc.reconstruction,
            _TARGET_GAIN,
            interleaved=projection == "gate_up",
        )
        data = iq2r_v2_write_data(
            enc.indices, enc.negative, enc.levels, _LEVELS, entries
        )
        aux = iq2r_v2_write_auxiliary(book[None], enc.d, gains)
        # Every matrix: the records decode to the encoding, bit for bit.
        decoded = iq2r_v2_materialize(
            data,
            aux,
            metadata.logical_n,
            metadata.logical_k,
            _LEVELS,
            apply_gain=False,
            codebook_entries=entries,
        )
        if not torch.equal(decoded, enc.reconstruction):
            raise RuntimeError(
                f"layer {layer} expert {expert} {projection}: records do not "
                "decode to the encoding"
            )
        del weight, enc, decoded
        if tuple(data.shape) != (metadata.data_bytes,):
            raise ValueError(f"layer {layer} expert {expert} {projection}: data bytes")
        data_rows.append(data.cpu())
        aux_rows.append(aux.cpu())
        if (expert + 1) % 16 == 0 or expert + 1 == layout.expert_count:
            print(
                json.dumps(
                    {
                        "layer": layer,
                        "projection": projection,
                        "experts_complete": expert + 1,
                        "experts_total": layout.expert_count,
                    }
                ),
                flush=True,
            )
    return torch.stack(data_rows), torch.stack(aux_rows)


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
    entries: int = _CODEBOOK_ENTRIES,
    hessians: GLM5Hessians | None = None,
) -> None:
    """Encode and pack one layer; write its layer files that do not exist yet."""

    factor = hessians.factor(layer, device) if hessians is not None else None
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
            entries=entries,
            ldlq=factor,
        )
        for projection in _PROJECTIONS
    ]
    # Returns (gate_quad, gate_auxiliary, down_data, down_auxiliary).
    canonical = [tensor.to(device) for pair in encoded for tensor in pair]
    packed = iq2r_glm53_pack(
        *canonical, intermediate_size=layout.intermediate_size, levels=_LEVELS
    )
    # The packed layer round-trips: packing its unpacked records gives it back.
    repacked = iq2r_glm53_pack(
        *iq2r_glm53_unpack(
            *packed, intermediate_size=layout.intermediate_size, levels=_LEVELS
        ),
        intermediate_size=layout.intermediate_size,
        levels=_LEVELS,
    )
    if not all(torch.equal(a, b) for a, b in zip(repacked, packed)):
        raise RuntimeError(f"layer {layer}: the packed layer does not round-trip")
    del canonical, repacked
    metadata = _encoding_metadata(
        importance, iterations, sample_vectors, entries, hessians
    )
    for projection, data, auxiliary in zip(_PROJECTIONS, packed[::2], packed[1::2]):
        path = output_dir / _layer_file(layer, projection)
        if path.exists():
            continue
        keys = iq2r_compiled_tensor_keys(layer, projection)
        temporary = path.with_suffix(path.suffix + ".tmp")
        save_file(
            {
                keys["data"]: data.cpu().contiguous(),
                keys["auxiliary"]: auxiliary.cpu().contiguous(),
                keys["tile_n"]: torch.tensor([128], dtype=torch.int32),
            },
            temporary,
            metadata=_layer_file_metadata(layer, projection, metadata),
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
    entries: int = _CODEBOOK_ENTRIES,
) -> Path:
    """Write the FP8 shards, index, config.json and tokenizer files.

    Every source tensor except the routed experts of the compiled layers
    (including the shared experts) is copied into new shards of about 5 GiB.
    Returns the config path.
    """

    layers = range(layout.first_moe_layer, layout.layer_count)
    compiled = {
        name
        for layer in layers
        for expert in range(layout.expert_count)
        for name in iq2r_glm5_source_keys(layer, expert).values()
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
        # layer keeps the MTP layer's experts and the shared experts on the
        # FP8 path.
        "iq2r_modules": [f"model.layers.{layer}.mlp.experts*" for layer in layers],
        "iq2r_layout": LAYOUTS[entries],
        "iq2r_format": iq2r_v2_format_name(_LEVELS, entries),
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
    layer_indices: list[int] | None = None,
    diagnostic_uniform_importance: bool = False,
    iterations: int = _ITERATIONS,
    sample_vectors: int = _SAMPLE_VECTORS,
    codebook_entries: int = _CODEBOOK_ENTRIES,
    ldlq_hessian_dir: str | os.PathLike[str] | None = None,
) -> Path | None:
    """Compile the selected MoE layers into ``output_dir`` on ``cuda:0``.

    Valid layer files already in ``output_dir`` are kept; an invalid one is an
    error. A run over every MoE layer also writes the rest of the checkpoint
    and returns its config path. A run over a subset returns None.
    ``codebook_entries`` (512 or 1024) selects the codebook size.
    ``ldlq_hessian_dir`` (Hessians from ``calibrate --hessian-dir``) turns on
    LDLQ for gate/up.
    ``diagnostic_uniform_importance``, ``iterations`` and ``sample_vectors``
    exist for tests; production builds keep the defaults.
    """

    model_dir = Path(model_dir).resolve()
    output_dir = Path(output_dir).resolve()
    # Writing into the source would replace its index and config.json.
    if (
        output_dir == model_dir
        or model_dir in output_dir.parents
        or output_dir in model_dir.parents
    ):
        raise ValueError(
            f"the output dir {output_dir} and the model dir {model_dir} must not "
            "be the same or nested"
        )
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
    if codebook_entries not in LAYOUTS:
        raise ValueError(
            f"codebook_entries must be 512 or 1024, got {codebook_entries}"
        )

    importance = load_glm5_importance(
        calibration_cache,
        layout,
        diagnostic_uniform_importance=diagnostic_uniform_importance,
    )
    if importance.routed_hits is None and importance.quality == "calibrated-o0":
        raise ValueError(
            "the calibration artifact has no routed hit counts, which the "
            "low-token fallback needs; re-run calibrate"
        )
    hessians = (
        load_glm5_hessians(ldlq_hessian_dir, layout)
        if ldlq_hessian_dir is not None
        else None
    )
    encoding = _encoding_metadata(
        importance, iterations, sample_vectors, codebook_entries, hessians
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

    device = torch.device(_DEVICE)
    _validate_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    for layer in selected_layers:
        existing = 0
        for projection in _PROJECTIONS:
            path = output_dir / _layer_file(layer, projection)
            if path.exists():
                _validate_projection_shard(path, layout, layer, projection, encoding)
                existing += 1
        if existing == len(_PROJECTIONS):
            continue
        with _TensorReader(model_dir, weight_map) as reader:
            _compile_layer(
                reader,
                layout,
                importance,
                layer,
                output_dir,
                device=device,
                iterations=iterations,
                sample_vectors=sample_vectors,
                entries=codebook_entries,
                hessians=hessians,
            )

    if selected_layers != moe_layers:
        return None
    return _write_checkpoint_files(
        model_dir, output_dir, config, weight_map, layout, codebook_entries
    )


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
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-dir", type=Path, required=True, help="GLM-5.3 block-FP8 checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="IQ2R-v2 checkpoint directory (not inside --model-dir)",
    )
    parser.add_argument(
        "--calibration-cache",
        type=Path,
        required=True,
        help="artifact written by aiter.quantization.iq2r.glm5.calibrate",
    )
    parser.add_argument(
        "--layers",
        type=_parse_layers,
        help=(
            "MoE layers to encode, e.g. 3-20,40, to split the work over GPUs; "
            "without it every layer is encoded (or kept) and the checkpoint "
            "files are written"
        ),
    )
    parser.add_argument(
        "--codebook-entries",
        type=int,
        choices=sorted(LAYOUTS),
        default=_CODEBOOK_ENTRIES,
        help="codebook size: 512 (glm53-packed-v2) or 1024 (glm53-packed-v3)",
    )
    parser.add_argument(
        "--ldlq",
        action="store_true",
        help="re-choose the gate/up codewords by LDLQ (needs --hessian-dir)",
    )
    parser.add_argument(
        "--hessian-dir",
        type=Path,
        help="layer Hessians written by aiter.quantization.iq2r.glm5.calibrate --hessian-dir",
    )
    args = parser.parse_args(argv)
    if args.ldlq != (args.hessian_dir is not None):
        parser.error("--ldlq and --hessian-dir go together")
    extra = {} if not args.ldlq else {"ldlq_hessian_dir": args.hessian_dir}
    config_path = compile_glm5_iq2r(
        args.model_dir,
        args.output_dir,
        calibration_cache=args.calibration_cache,
        layer_indices=args.layers,
        codebook_entries=args.codebook_entries,
        **extra,
    )
    print(json.dumps({"config": config_path and str(config_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GLM5Hessians",
    "GLM5Importance",
    "GLM5Layout",
    "compile_glm5_iq2r",
    "dequantize_block_fp8",
    "glm5_source_layout",
    "interleave_gate_up",
    "load_glm5_hessians",
    "load_glm5_importance",
]
