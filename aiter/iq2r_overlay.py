# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Build Hugging Face-compatible IQ2R checkpoint overlays.

An overlay keeps the base model shards as symlinks and makes its generated
index authoritative. GPT-OSS v1 reuses the model's fused MXFP4 tensor names.
The model-neutral v2 schema replaces per-expert tensors with four explicitly
named fused IQ2R tensors while preserving the base quantization configuration
for attention, dense MLPs, and shared experts.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .iq2r_checkpoint import (
    IQ2R_CHECKPOINT_SCHEMA,
    IQ2R_CHECKPOINT_SCHEMA_VERSION,
    IQ2R_GENERIC_CHECKPOINT_SCHEMA,
    IQ2R_GENERIC_CHECKPOINT_SCHEMA_VERSION,
    iq2r_compiled_tensor_keys,
    iq2r_glm5_overlay_keys,
    iq2r_glm5_source_keys,
    iq2r_gpt_oss_source_keys,
    load_iq2r_layer_checkpoint,
)
from .iq2r_glm5_compile import glm5_source_layout
from .ops.iq2r_format import (
    IQ2R_GPT_OSS_DOWN_N,
    IQ2R_GPT_OSS_GATE_UP_N,
    IQ2R_GPT_OSS_K,
)

IQ2R_OVERLAY_SCHEMA = "aiter-gpt-oss-iq2r-overlay"
IQ2R_OVERLAY_SCHEMA_VERSION = 1
IQ2R_GENERIC_OVERLAY_SCHEMA = "aiter-iq2r-overlay"
IQ2R_GENERIC_OVERLAY_SCHEMA_VERSION = 2


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


def _replace_with_symlink(source: Path, destination: Path, *, force: bool) -> None:
    if destination.is_symlink() and destination.resolve() == source.resolve():
        return
    if destination.exists() or destination.is_symlink():
        if not force:
            raise FileExistsError(f"refusing to replace {destination}; pass --force")
        destination.unlink()
    destination.symlink_to(source.resolve())


def _overlay_tensor_names(layer_index: int) -> dict[str, str]:
    # Reuse GPT-OSS's standard names so the existing model mapping remains the
    # sole checkpoint-to-module adapter. In an IQ2R overlay, *_blocks are the
    # packed data buffers and *_scales are the IQ2R auxiliary buffers.
    return iq2r_gpt_oss_source_keys(layer_index)


def create_gpt_oss_iq2r_overlay(
    model_dir: str | os.PathLike[str],
    compiled_iq2r_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    force: bool = False,
) -> Path:
    """Create a complete GPT-OSS IQ2R overlay and return its manifest path."""

    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("IQ2R overlay creation requires safetensors") from error

    model_dir = Path(model_dir).resolve()
    compiled_iq2r_dir = Path(compiled_iq2r_dir).resolve()
    output_dir = Path(output_dir).resolve()

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config = _read_json(config_path)
    source_index = _read_json(index_path)
    source_weight_map = source_index.get("weight_map")
    if not isinstance(source_weight_map, dict):
        raise TypeError(f"{index_path} does not contain a weight_map object")
    layer_count = int(config.get("num_hidden_layers", -1))
    if layer_count <= 0:
        raise ValueError("GPT-OSS config has no positive num_hidden_layers")

    # Validate the compiled checkpoint before touching the destination. Loading
    # layer zero verifies the model-level format declaration and full expert
    # shapes; all remaining layers are checked as their shards are emitted.
    first_checkpoint = load_iq2r_layer_checkpoint(compiled_iq2r_dir, 0)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_shards = sorted(set(source_weight_map.values()))
    for shard in source_shards:
        source = model_dir / shard
        if not source.is_file():
            raise FileNotFoundError(f"source checkpoint shard is missing: {source}")
        _replace_with_symlink(source, output_dir / Path(shard).name, force=force)
    for source in model_dir.iterdir():
        if (
            not source.is_file()
            or source.name == "config.json"
            or source.name == "model.safetensors.index.json"
            or source.suffix == ".safetensors"
        ):
            continue
        _replace_with_symlink(source, output_dir / source.name, force=force)

    weight_map = dict(source_weight_map)
    layer_records: list[dict[str, Any]] = []
    total_iq2r_bytes = 0
    total_iq2r_payload_bytes = 0
    for layer_index in range(layer_count):
        checkpoint = (
            first_checkpoint
            if layer_index == 0
            else load_iq2r_layer_checkpoint(compiled_iq2r_dir, layer_index)
        )
        source_names = iq2r_gpt_oss_source_keys(layer_index)
        for source_name in source_names.values():
            if source_name not in weight_map:
                raise KeyError(f"source index is missing {source_name!r}")
            del weight_map[source_name]

        names = _overlay_tensor_names(layer_index)
        tensors = {
            names["gate_up_blocks"]: checkpoint.gate_up_data.contiguous(),
            names["gate_up_scales"]: checkpoint.gate_up_auxiliary.contiguous(),
            names["gate_up_bias"]: checkpoint.gate_up_bias.contiguous(),
            names["down_blocks"]: checkpoint.down_data.contiguous(),
            names["down_scales"]: checkpoint.down_auxiliary.contiguous(),
            names["down_bias"]: checkpoint.down_bias.contiguous(),
        }
        shard_name = f"iq2r-model-layer-{layer_index:04d}.safetensors"
        shard_path = output_dir / shard_name
        if shard_path.exists() and not force:
            raise FileExistsError(f"refusing to overwrite {shard_path}; pass --force")
        temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
        if temporary.exists():
            temporary.unlink()
        save_file(
            tensors,
            temporary,
            metadata={
                "format": "pt",
                "iq2r_schema": IQ2R_CHECKPOINT_SCHEMA,
                "iq2r_schema_version": str(IQ2R_CHECKPOINT_SCHEMA_VERSION),
                "iq2r_layer": str(layer_index),
            },
        )
        os.replace(temporary, shard_path)
        for name in tensors:
            weight_map[name] = shard_name
        shard_bytes = shard_path.stat().st_size
        payload_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in tensors.values()
        )
        total_iq2r_bytes += shard_bytes
        total_iq2r_payload_bytes += payload_bytes
        layer_records.append(
            {
                "layer_index": layer_index,
                "overlay_shard": shard_name,
                "overlay_shard_bytes": shard_bytes,
                "payload_bytes": payload_bytes,
                "compiled_source_shards": list(checkpoint.source_shards),
                "gate_up_tile_n": checkpoint.gate_up_tile_n,
                "down_tile_n": checkpoint.down_tile_n,
                "gate_up_metadata": checkpoint.gate_up_metadata.to_dict(),
                "down_metadata": checkpoint.down_metadata.to_dict(),
            }
        )
        del tensors, checkpoint

    overlay_config = dict(config)
    overlay_config["quantization_config"] = {
        "quant_method": "iq2r",
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head",
        ],
        "schema": IQ2R_OVERLAY_SCHEMA,
        "schema_version": IQ2R_OVERLAY_SCHEMA_VERSION,
    }
    _atomic_write_json(output_dir / "config.json", overlay_config)

    overlay_index = {
        "metadata": {
            **(source_index.get("metadata") or {}),
            "iq2r_schema": IQ2R_OVERLAY_SCHEMA,
            "iq2r_schema_version": IQ2R_OVERLAY_SCHEMA_VERSION,
            "iq2r_layer_count": layer_count,
            "iq2r_bytes": total_iq2r_bytes,
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    _atomic_write_json(output_dir / "model.safetensors.index.json", overlay_index)

    source_mxfp4_payload_bytes = (
        layer_count
        * 128
        * (IQ2R_GPT_OSS_GATE_UP_N + IQ2R_GPT_OSS_DOWN_N)
        * (IQ2R_GPT_OSS_K // 32)
        * 17
    )
    manifest = {
        "schema": IQ2R_OVERLAY_SCHEMA,
        "schema_version": IQ2R_OVERLAY_SCHEMA_VERSION,
        "base_model": str(model_dir),
        "compiled_iq2r_fixture": str(compiled_iq2r_dir),
        "layer_count": layer_count,
        "iq2r_bytes": total_iq2r_bytes,
        "iq2r_payload_bytes": total_iq2r_payload_bytes,
        "source_mxfp4_payload_bytes": source_mxfp4_payload_bytes,
        "iq2r_to_mxfp4_payload_ratio": (
            total_iq2r_payload_bytes / source_mxfp4_payload_bytes
        ),
        "source_shards": source_shards,
        "layers": layer_records,
    }
    manifest_path = output_dir / "iq2r-overlay-manifest.json"
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


def _is_glm5(config: dict[str, Any]) -> bool:
    architectures = config.get("architectures") or []
    return config.get("model_type") in ("glm5_next", "glm_moe_dsa") or any(
        isinstance(name, str) and name.startswith(("Glm5Next", "GlmMoeDsa"))
        for name in architectures
    )


def _link_base_model_files(
    model_dir: Path,
    output_dir: Path,
    source_weight_map: dict[str, str],
    *,
    force: bool,
) -> list[str]:
    source_shards = sorted(set(source_weight_map.values()))
    for shard in source_shards:
        source = model_dir / shard
        if not source.is_file():
            raise FileNotFoundError(f"source checkpoint shard is missing: {source}")
        _replace_with_symlink(source, output_dir / Path(shard).name, force=force)
    for source in model_dir.iterdir():
        if (
            not source.is_file()
            or source.name == "config.json"
            or source.name == "model.safetensors.index.json"
            or source.suffix == ".safetensors"
        ):
            continue
        _replace_with_symlink(source, output_dir / source.name, force=force)
    return source_shards


def create_glm5_iq2r_overlay(
    model_dir: str | os.PathLike[str],
    compiled_iq2r_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    force: bool = False,
    reuse_compiled_shards: bool = False,
) -> Path:
    """Create a GLM-5.3 IQ2R overlay from a Redline TP1/EP1 O0 checkpoint."""

    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as error:
        raise RuntimeError("IQ2R overlay creation requires safetensors") from error

    model_dir = Path(model_dir).resolve()
    compiled_iq2r_dir = Path(compiled_iq2r_dir).resolve()
    output_dir = Path(output_dir).resolve()

    config_path = model_dir / "config.json"
    index_path = model_dir / "model.safetensors.index.json"
    config = _read_json(config_path)
    if not _is_glm5(config):
        raise ValueError(
            "create_glm5_iq2r_overlay requires a GLM-5 Next or GLM MoE DSA model"
        )
    layout = glm5_source_layout(config)
    if reuse_compiled_shards and layout.model_family != "glm_moe_dsa":
        raise ValueError(
            "compiled-shard reuse is currently supported only for plain glm_moe_dsa"
        )
    source_index = _read_json(index_path)
    source_weight_map = source_index.get("weight_map")
    if not isinstance(source_weight_map, dict) or not all(
        isinstance(name, str) and isinstance(shard, str)
        for name, shard in source_weight_map.items()
    ):
        raise ValueError(f"{index_path} does not contain a string weight_map object")

    layer_count = layout.layer_count
    first_moe_layer = layout.first_moe_layer
    expert_count = layout.expert_count
    hidden_size = layout.hidden_size
    intermediate_size = layout.intermediate_size

    # Validate the compiled model-level identity and first full layer before
    # creating any output. Remaining layers are checked as they are emitted.
    first_checkpoint = load_iq2r_layer_checkpoint(compiled_iq2r_dir, first_moe_layer)
    if first_checkpoint.model_family != "glm5":
        raise ValueError(
            "compiled IQ2R checkpoint is not identified as a GLM-5 checkpoint"
        )
    if first_checkpoint.total_experts != expert_count:
        raise ValueError(
            "compiled IQ2R and source GLM expert counts differ: "
            f"{first_checkpoint.total_experts} != {expert_count}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    source_shards = _link_base_model_files(
        model_dir, output_dir, source_weight_map, force=force
    )

    weight_map = dict(source_weight_map)
    layer_records: list[dict[str, Any]] = []
    total_iq2r_bytes = 0
    total_iq2r_payload_bytes = 0
    removed_source_tensor_count = 0
    for layer_index in range(first_moe_layer, layer_count):
        checkpoint = (
            first_checkpoint
            if layer_index == first_moe_layer
            else load_iq2r_layer_checkpoint(compiled_iq2r_dir, layer_index)
        )
        if checkpoint.gate_up_bias is not None or checkpoint.down_bias is not None:
            raise ValueError("GLM-5 routed experts must not contain projection biases")
        if checkpoint.expert_count != expert_count:
            raise ValueError(
                f"compiled layer {layer_index} contains "
                f"{checkpoint.expert_count} experts, expected {expert_count}"
            )

        for expert_index in range(expert_count):
            for source_name in iq2r_glm5_source_keys(
                layer_index, expert_index, root=layout.source_root
            ).values():
                if source_name not in weight_map:
                    raise KeyError(f"source index is missing {source_name!r}")
                del weight_map[source_name]
                removed_source_tensor_count += 1

        if reuse_compiled_shards:
            projection_keys = {
                "gate_up": iq2r_compiled_tensor_keys(
                    layer_index, "gate_up", module_name="up_gate_proj"
                ),
                "down": iq2r_compiled_tensor_keys(
                    layer_index, "down", module_name="down_proj"
                ),
            }
            projection_shards: dict[str, str] = {}
            for projection, keys in projection_keys.items():
                for candidate in checkpoint.source_shards:
                    source = compiled_iq2r_dir / candidate
                    if not source.is_file():
                        raise FileNotFoundError(
                            f"compiled IQ2R shard is missing: {source}"
                        )
                    with safe_open(source, framework="pt", device="cpu") as handle:
                        if {keys["data"], keys["auxiliary"]}.issubset(handle.keys()):
                            projection_shards[projection] = candidate
                            break
                else:
                    raise KeyError(
                        f"compiled layer {layer_index} has no {projection} data shard"
                    )
            for candidate in dict.fromkeys(projection_shards.values()):
                _replace_with_symlink(
                    compiled_iq2r_dir / candidate,
                    output_dir / candidate,
                    force=force,
                )
            for projection, keys in projection_keys.items():
                shard = projection_shards[projection]
                weight_map[keys["data"]] = shard
                weight_map[keys["auxiliary"]] = shard
            shard_names = list(dict.fromkeys(projection_shards.values()))
            shard_bytes = sum(
                (output_dir / name).stat().st_size for name in shard_names
            )
            payload_bytes = sum(
                tensor.numel() * tensor.element_size()
                for tensor in (
                    checkpoint.gate_up_data,
                    checkpoint.gate_up_auxiliary,
                    checkpoint.down_data,
                    checkpoint.down_auxiliary,
                )
            )
        else:
            names = iq2r_glm5_overlay_keys(layer_index, root=layout.source_root)
            tensors = {
                names["gate_up_data"]: checkpoint.gate_up_data.contiguous(),
                names["gate_up_auxiliary"]: checkpoint.gate_up_auxiliary.contiguous(),
                names["down_data"]: checkpoint.down_data.contiguous(),
                names["down_auxiliary"]: checkpoint.down_auxiliary.contiguous(),
            }
            shard_name = f"iq2r-model-layer-{layer_index:04d}.safetensors"
            shard_path = output_dir / shard_name
            if shard_path.exists() and not force:
                raise FileExistsError(
                    f"refusing to overwrite {shard_path}; pass --force"
                )
            temporary = shard_path.with_suffix(shard_path.suffix + ".tmp")
            if temporary.exists():
                temporary.unlink()
            save_file(
                tensors,
                temporary,
                metadata={
                    "format": "pt",
                    "iq2r_schema": IQ2R_GENERIC_CHECKPOINT_SCHEMA,
                    "iq2r_schema_version": str(IQ2R_GENERIC_CHECKPOINT_SCHEMA_VERSION),
                    "iq2r_layer": str(layer_index),
                },
            )
            os.replace(temporary, shard_path)
            for name in tensors:
                weight_map[name] = shard_name
            shard_names = [shard_name]
            shard_bytes = shard_path.stat().st_size
            payload_bytes = sum(
                tensor.numel() * tensor.element_size() for tensor in tensors.values()
            )
        total_iq2r_bytes += shard_bytes
        total_iq2r_payload_bytes += payload_bytes
        layer_record = {
            "layer_index": layer_index,
            "overlay_shards": shard_names,
            "overlay_shard_bytes": shard_bytes,
            "payload_bytes": payload_bytes,
            "compiled_source_shards": list(checkpoint.source_shards),
            "gate_up_tile_n": checkpoint.gate_up_tile_n,
            "down_tile_n": checkpoint.down_tile_n,
            "gate_up_metadata": checkpoint.gate_up_metadata.to_dict(),
            "down_metadata": checkpoint.down_metadata.to_dict(),
        }
        if len(shard_names) == 1:
            # Preserve the original manifest field for consumers written
            # before compiled projection shards could be reused independently.
            layer_record["overlay_shard"] = shard_names[0]
        layer_records.append(layer_record)
        checkpoint = None

    base_quantization_config = config.get("quantization_config")
    if not isinstance(base_quantization_config, dict):
        raise TypeError("GLM-5 source config has no base quantization_config")
    overlay_config = dict(config)
    overlay_config["quantization_config"] = {
        "quant_method": "iq2r",
        "schema": IQ2R_GENERIC_OVERLAY_SCHEMA,
        "schema_version": IQ2R_GENERIC_OVERLAY_SCHEMA_VERSION,
        "base_quantization_config": base_quantization_config,
        # Both GLM checkpoint roots are normalized to ``model.layers`` by the
        # ATOM model adapters before quantization method selection.
        "iq2r_modules": ["model.layers.*.mlp.experts"],
    }
    _atomic_write_json(output_dir / "config.json", overlay_config)

    moe_layer_count = layer_count - first_moe_layer
    overlay_index = {
        "metadata": {
            **(source_index.get("metadata") or {}),
            "iq2r_schema": IQ2R_GENERIC_OVERLAY_SCHEMA,
            "iq2r_schema_version": IQ2R_GENERIC_OVERLAY_SCHEMA_VERSION,
            "iq2r_layer_count": moe_layer_count,
            "iq2r_first_layer": first_moe_layer,
            "iq2r_bytes": total_iq2r_bytes,
        },
        "weight_map": dict(sorted(weight_map.items())),
    }
    _atomic_write_json(output_dir / "model.safetensors.index.json", overlay_index)

    source_fp8_weight_bytes = (
        moe_layer_count * expert_count * 3 * hidden_size * intermediate_size
    )
    manifest = {
        "schema": IQ2R_GENERIC_OVERLAY_SCHEMA,
        "schema_version": IQ2R_GENERIC_OVERLAY_SCHEMA_VERSION,
        "model_family": layout.model_family,
        "source_root": layout.source_root,
        "base_model": str(model_dir),
        "compiled_iq2r_checkpoint": str(compiled_iq2r_dir),
        "first_moe_layer": first_moe_layer,
        "layer_count": moe_layer_count,
        "expert_count": expert_count,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "iq2r_bytes": total_iq2r_bytes,
        "iq2r_payload_bytes": total_iq2r_payload_bytes,
        "source_fp8_weight_bytes": source_fp8_weight_bytes,
        "iq2r_to_source_fp8_weight_ratio": (
            total_iq2r_payload_bytes / source_fp8_weight_bytes
        ),
        "removed_source_tensor_count": removed_source_tensor_count,
        "reused_compiled_shards": reuse_compiled_shards,
        "source_shards": source_shards,
        "layers": layer_records,
    }
    manifest_path = output_dir / "iq2r-overlay-manifest.json"
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


def create_iq2r_overlay(
    model_dir: str | os.PathLike[str],
    compiled_iq2r_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    *,
    force: bool = False,
    reuse_compiled_shards: bool = False,
) -> Path:
    """Create the architecture-appropriate IQ2R overlay."""

    config = _read_json(Path(model_dir) / "config.json")
    if _is_glm5(config):
        return create_glm5_iq2r_overlay(
            model_dir,
            compiled_iq2r_dir,
            output_dir,
            force=force,
            reuse_compiled_shards=reuse_compiled_shards,
        )
    architectures = config.get("architectures") or []
    if config.get("model_type") == "gpt_oss" or "GptOssForCausalLM" in architectures:
        return create_gpt_oss_iq2r_overlay(
            model_dir, compiled_iq2r_dir, output_dir, force=force
        )
    raise ValueError(
        "unsupported IQ2R overlay source architecture: "
        f"model_type={config.get('model_type')!r}, architectures={architectures!r}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--compiled-iq2r-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--reuse-compiled-shards",
        action="store_true",
        help=(
            "Reference the plain-GLM compiled projection shards directly instead "
            "of duplicating their payload into combined overlay shards"
        ),
    )
    args = parser.parse_args(argv)
    manifest = create_iq2r_overlay(
        args.model_dir,
        args.compiled_iq2r_dir,
        args.output_dir,
        force=args.force,
        reuse_compiled_shards=args.reuse_compiled_shards,
    )
    print(json.dumps({"manifest": str(manifest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "IQ2R_GENERIC_OVERLAY_SCHEMA",
    "IQ2R_GENERIC_OVERLAY_SCHEMA_VERSION",
    "IQ2R_OVERLAY_SCHEMA",
    "IQ2R_OVERLAY_SCHEMA_VERSION",
    "create_glm5_iq2r_overlay",
    "create_gpt_oss_iq2r_overlay",
    "create_iq2r_overlay",
]
