# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import aiter.iq2r_overlay as overlay_module
from aiter.iq2r_checkpoint import (
    iq2r_compiled_tensor_keys,
    iq2r_glm5_overlay_keys,
    iq2r_glm5_source_keys,
    iq2r_gpt_oss_source_keys,
)
from aiter.iq2r_overlay import (
    create_glm5_iq2r_overlay,
    create_gpt_oss_iq2r_overlay,
    create_iq2r_overlay,
)


class _FakeMetadata:
    def __init__(self, matrix: str):
        self.matrix = matrix

    def to_dict(self):
        return {"matrix": self.matrix, "format_version": 4}


def _write_source_model(path, layers: int) -> None:
    path.mkdir()
    tensors = {"model.embed_tokens.weight": torch.arange(4, dtype=torch.float32)}
    for layer in range(layers):
        for index, name in enumerate(iq2r_gpt_oss_source_keys(layer).values()):
            tensors[name] = torch.tensor([layer, index], dtype=torch.float32)
    save_file(tensors, path / "model-00001-of-00001.safetensors")
    weight_map = {name: "model-00001-of-00001.safetensors" for name in tensors}
    (path / "config.json").write_text(
        json.dumps({"model_type": "gpt_oss", "num_hidden_layers": layers})
    )
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 123}, "weight_map": weight_map})
    )
    (path / "tokenizer.json").write_text("{}")


def _fake_checkpoint(layer: int):
    return SimpleNamespace(
        layer_index=layer,
        expert_start=0,
        expert_count=128,
        gate_up_data=torch.tensor([[layer, 1]], dtype=torch.uint8),
        gate_up_auxiliary=torch.tensor([[layer, 2]], dtype=torch.uint8),
        gate_up_bias=torch.tensor([[layer, 3]], dtype=torch.bfloat16),
        down_data=torch.tensor([[layer, 4]], dtype=torch.uint8),
        down_auxiliary=torch.tensor([[layer, 5]], dtype=torch.uint8),
        down_bias=torch.tensor([[layer, 6]], dtype=torch.bfloat16),
        gate_up_tile_n=128,
        down_tile_n=64,
        gate_up_metadata=_FakeMetadata("gate_up"),
        down_metadata=_FakeMetadata("down"),
        source_shards=(f"compiled-{layer}.safetensors",),
    )


def _write_glm_source_model(path, layers: int, experts: int) -> None:
    path.mkdir()
    tensors = {"model.language_model.embed_tokens.weight": torch.arange(4)}
    # Include one MTP layer after num_hidden_layers. It must remain base FP8 and
    # must not be rewritten into the runtime's routed-expert overlay.
    for layer in range(3, layers + 1):
        for expert in range(experts):
            for index, name in enumerate(iq2r_glm5_source_keys(layer, expert).values()):
                tensors[name] = torch.tensor([layer, expert, index])
    save_file(tensors, path / "model-00001-of-00001.safetensors")
    weight_map = {name: "model-00001-of-00001.safetensors" for name in tensors}
    config = {
        "architectures": ["Glm5NextForConditionalGeneration"],
        "model_type": "glm5_next",
        "text_config": {
            "num_hidden_layers": layers,
            "first_k_dense_replace": 3,
            "n_routed_experts": experts,
            "hidden_size": 128,
            "moe_intermediate_size": 64,
        },
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 456}, "weight_map": weight_map})
    )
    (path / "tokenizer.json").write_text("{}")


def _write_plain_glm_source_model(path, layers: int, experts: int) -> None:
    path.mkdir()
    tensors = {
        "model.embed_tokens.weight": torch.arange(4),
        "model.layers.0.mlp.gate_proj.weight": torch.tensor([10]),
        "model.layers.3.mlp.shared_experts.gate_proj.weight": torch.tensor([11]),
        "model.layers.3.self_attn.q_proj.weight": torch.tensor([12]),
    }
    # Include one MTP layer after num_hidden_layers. It must remain base FP8.
    for layer in range(3, layers + 1):
        for expert in range(experts):
            for index, name in enumerate(
                iq2r_glm5_source_keys(layer, expert, root="model").values()
            ):
                tensors[name] = torch.tensor([layer, expert, index])
    save_file(tensors, path / "model-00001-of-00001.safetensors")
    weight_map = {name: "model-00001-of-00001.safetensors" for name in tensors}
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": layers,
        "first_k_dense_replace": 3,
        "n_routed_experts": experts,
        "hidden_size": 128,
        "moe_intermediate_size": 64,
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 789}, "weight_map": weight_map})
    )
    (path / "tokenizer.json").write_text("{}")


def _fake_glm_checkpoint(layer: int, experts: int):
    return SimpleNamespace(
        layer_index=layer,
        expert_start=0,
        expert_count=experts,
        total_experts=experts,
        model_family="glm5",
        gate_up_data=torch.full((experts, 2), layer, dtype=torch.uint8),
        gate_up_auxiliary=torch.zeros((experts, 2), dtype=torch.uint8),
        gate_up_bias=None,
        down_data=torch.full((experts, 2), layer + 1, dtype=torch.uint8),
        down_auxiliary=torch.zeros((experts, 2), dtype=torch.uint8),
        down_bias=None,
        gate_up_tile_n=128,
        down_tile_n=128,
        gate_up_metadata=_FakeMetadata("gate_up"),
        down_metadata=_FakeMetadata("down"),
        source_shards=(
            f"compiled-{layer}-gate.safetensors",
            f"compiled-{layer}-down.safetensors",
        ),
    )


def test_overlay_is_complete_and_index_authoritative(tmp_path, monkeypatch):
    source = tmp_path / "source"
    compiled = tmp_path / "compiled"
    output = tmp_path / "overlay"
    _write_source_model(source, layers=2)
    compiled.mkdir()

    monkeypatch.setattr(
        overlay_module,
        "load_iq2r_layer_checkpoint",
        lambda _path, layer: _fake_checkpoint(layer),
    )
    manifest_path = create_gpt_oss_iq2r_overlay(source, compiled, output)

    assert (output / "model-00001-of-00001.safetensors").is_symlink()
    assert (output / "tokenizer.json").is_symlink()
    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"] == {
        "quant_method": "iq2r",
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head",
        ],
        "schema": "aiter-gpt-oss-iq2r-overlay",
        "schema_version": 1,
    }

    index = json.loads((output / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    assert weight_map["model.embed_tokens.weight"] == "model-00001-of-00001.safetensors"
    for layer in range(2):
        names = iq2r_gpt_oss_source_keys(layer)
        replacement = f"iq2r-model-layer-{layer:04d}.safetensors"
        assert all(weight_map[name] == replacement for name in names.values())
        with safe_open(output / replacement, framework="pt", device="cpu") as shard:
            assert set(shard.keys()) == set(names.values())
            assert shard.get_tensor(names["gate_up_bias"]).dtype == torch.bfloat16
            assert shard.get_tensor(names["down_bias"]).dtype == torch.bfloat16

    manifest = json.loads(manifest_path.read_text())
    assert manifest["layer_count"] == 2
    assert len(manifest["layers"]) == 2
    assert manifest["iq2r_payload_bytes"] == 32
    assert manifest["iq2r_to_mxfp4_payload_ratio"] > 0
    assert index["metadata"]["iq2r_layer_count"] == 2
    assert index["metadata"]["iq2r_bytes"] == manifest["iq2r_bytes"]


def test_glm_overlay_replaces_only_runtime_routed_experts(tmp_path, monkeypatch):
    source = tmp_path / "source"
    compiled = tmp_path / "compiled"
    output = tmp_path / "overlay"
    experts = 2
    _write_glm_source_model(source, layers=5, experts=experts)
    compiled.mkdir()

    monkeypatch.setattr(
        overlay_module,
        "load_iq2r_layer_checkpoint",
        lambda _path, layer: _fake_glm_checkpoint(layer, experts),
    )
    manifest_path = create_glm5_iq2r_overlay(source, compiled, output)

    config = json.loads((output / "config.json").read_text())
    assert config["quantization_config"] == {
        "quant_method": "iq2r",
        "schema": "aiter-iq2r-overlay",
        "schema_version": 2,
        "base_quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
        "iq2r_modules": ["model.layers.*.mlp.experts"],
    }

    index = json.loads((output / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    for layer in (3, 4):
        for expert in range(experts):
            assert all(
                name not in weight_map
                for name in iq2r_glm5_source_keys(layer, expert).values()
            )
        replacement = f"iq2r-model-layer-{layer:04d}.safetensors"
        names = iq2r_glm5_overlay_keys(layer)
        assert all(weight_map[name] == replacement for name in names.values())
        with safe_open(output / replacement, framework="pt", device="cpu") as shard:
            assert set(shard.keys()) == set(names.values())

    # Layer 5 is the checkpoint's MTP draft layer and is outside
    # text_config.num_hidden_layers, so its original FP8 tensors remain indexed.
    assert all(name in weight_map for name in iq2r_glm5_source_keys(5, 0).values())
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "aiter-iq2r-overlay"
    assert manifest["first_moe_layer"] == 3
    assert manifest["layer_count"] == 2
    assert manifest["removed_source_tensor_count"] == 24
    assert manifest["layers"][0]["overlay_shard"] == (
        "iq2r-model-layer-0003.safetensors"
    )
    assert manifest["layers"][0]["overlay_shards"] == [
        "iq2r-model-layer-0003.safetensors"
    ]


def test_plain_glm_overlay_uses_model_root_and_preserves_non_routed_tensors(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    compiled = tmp_path / "compiled"
    output = tmp_path / "overlay"
    experts = 2
    _write_plain_glm_source_model(source, layers=5, experts=experts)
    compiled.mkdir()

    monkeypatch.setattr(
        overlay_module,
        "load_iq2r_layer_checkpoint",
        lambda _path, layer: _fake_glm_checkpoint(layer, experts),
    )
    manifest_path = create_iq2r_overlay(source, compiled, output)

    config = json.loads((output / "config.json").read_text())
    assert config["model_type"] == "glm_moe_dsa"
    assert config["quantization_config"]["iq2r_modules"] == [
        "model.layers.*.mlp.experts"
    ]

    index = json.loads((output / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    for layer in (3, 4):
        for expert in range(experts):
            assert all(
                name not in weight_map
                for name in iq2r_glm5_source_keys(layer, expert, root="model").values()
            )
        names = iq2r_glm5_overlay_keys(layer, root="model")
        replacement = f"iq2r-model-layer-{layer:04d}.safetensors"
        assert all(weight_map[name] == replacement for name in names.values())

    assert "model.layers.0.mlp.gate_proj.weight" in weight_map
    assert "model.layers.3.mlp.shared_experts.gate_proj.weight" in weight_map
    assert "model.layers.3.self_attn.q_proj.weight" in weight_map
    assert all(
        name in weight_map
        for name in iq2r_glm5_source_keys(5, 0, root="model").values()
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["model_family"] == "glm_moe_dsa"
    assert manifest["source_root"] == "model"
    assert manifest["removed_source_tensor_count"] == 24


def test_plain_glm_overlay_can_reuse_compiled_projection_shards(tmp_path, monkeypatch):
    source = tmp_path / "source"
    compiled = tmp_path / "compiled"
    output = tmp_path / "overlay"
    experts = 2
    _write_plain_glm_source_model(source, layers=5, experts=experts)
    compiled.mkdir()
    for layer in (3, 4):
        for projection, suffix, module_name in (
            ("gate_up", "gate", "up_gate_proj"),
            ("down", "down", "down_proj"),
        ):
            keys = iq2r_compiled_tensor_keys(layer, projection, module_name=module_name)
            save_file(
                {
                    keys["data"]: torch.zeros((experts, 2), dtype=torch.uint8),
                    keys["auxiliary"]: torch.zeros((experts, 2), dtype=torch.uint8),
                    keys["tile_n"]: torch.tensor([128], dtype=torch.int32),
                },
                compiled / f"compiled-{layer}-{suffix}.safetensors",
            )

    monkeypatch.setattr(
        overlay_module,
        "load_iq2r_layer_checkpoint",
        lambda _path, layer: _fake_glm_checkpoint(layer, experts),
    )
    manifest_path = create_glm5_iq2r_overlay(
        source,
        compiled,
        output,
        reuse_compiled_shards=True,
    )

    index = json.loads((output / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    for layer in (3, 4):
        gate_keys = iq2r_compiled_tensor_keys(
            layer, "gate_up", module_name="up_gate_proj"
        )
        down_keys = iq2r_compiled_tensor_keys(layer, "down", module_name="down_proj")
        assert weight_map[gate_keys["data"]] == f"compiled-{layer}-gate.safetensors"
        assert (
            weight_map[gate_keys["auxiliary"]] == f"compiled-{layer}-gate.safetensors"
        )
        assert weight_map[down_keys["data"]] == f"compiled-{layer}-down.safetensors"
        assert (
            weight_map[down_keys["auxiliary"]] == f"compiled-{layer}-down.safetensors"
        )
        assert (output / f"compiled-{layer}-gate.safetensors").is_symlink()
        assert (output / f"compiled-{layer}-down.safetensors").is_symlink()
        assert not (output / f"iq2r-model-layer-{layer:04d}.safetensors").exists()

    manifest = json.loads(manifest_path.read_text())
    assert manifest["reused_compiled_shards"] is True
