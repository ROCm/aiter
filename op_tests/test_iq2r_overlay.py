# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import save_file

import aiter.iq2r_overlay as overlay_module
from aiter.iq2r_checkpoint import iq2r_gpt_oss_source_keys
from aiter.iq2r_overlay import create_gpt_oss_iq2r_overlay


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
