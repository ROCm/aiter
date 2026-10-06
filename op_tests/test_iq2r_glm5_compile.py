# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import fnmatch
import functools
import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import aiter.iq2r_glm5_compile as compile_module
from aiter.iq2r_glm5_compile import (
    LAYOUT,
    GLM5Layout,
    _projection_metadata,
    _validate_projection_shard,
    _write_checkpoint_files,
    compile_glm5_iq2r,
    dequantize_block_fp8,
    glm5_source_layout,
    interleave_gate_up,
    iq2r_compiled_tensor_keys,
    iq2r_glm5_shared_source_keys,
    iq2r_glm5_source_keys,
    load_glm5_importance,
)
from aiter.iq2r_glm53 import iq2r_glm53_gate_bytes
from aiter.ops.iq2r_format import (
    IQ2R_ACTIVATION_BASIS,
    IQ2R_FORMAT_NAME,
    IQ2RMetadata,
)

_HIDDEN = 6144
_INTERMEDIATE = 2048
_EXPERTS = 2
_LAYERS = 5
_QUALITY = "diagnostic-uniform-not-o0-quality"


def _layout() -> GLM5Layout:
    return GLM5Layout(
        layer_count=5,
        first_moe_layer=3,
        expert_count=2,
        hidden_size=4,
        intermediate_size=2,
        block_n=2,
        block_k=2,
    )


def test_source_layout_reads_plain_glm53_config():
    quantization_config = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
    }
    dimensions = {
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "hidden_size": 6144,
        "moe_intermediate_size": 2048,
    }
    plain = glm5_source_layout(
        {
            "architectures": ["GlmMoeDsaForCausalLM"],
            "model_type": "glm_moe_dsa",
            **dimensions,
            "quantization_config": quantization_config,
        }
    )

    assert plain.model_family == "glm_moe_dsa"
    assert plain.source_root == "model"
    assert plain.layer_count == 78
    assert plain.expert_count == 256
    assert plain.shared_expert_count == 1
    assert plain.hidden_size == 6144
    assert plain.intermediate_size == 2048
    with pytest.raises(ValueError, match="GLM MoE DSA"):
        glm5_source_layout(
            {
                "model_type": "llama",
                **dimensions,
                "quantization_config": quantization_config,
            }
        )


def test_plain_glm53_source_keys_use_top_level_model_root():
    names = iq2r_glm5_source_keys(3, 255, root="model")
    assert names["gate_proj_weight"] == (
        "model.layers.3.mlp.experts.255.gate_proj.weight"
    )
    assert names["down_proj_weight_scale_inv"] == (
        "model.layers.3.mlp.experts.255.down_proj.weight_scale_inv"
    )
    shared = iq2r_glm5_shared_source_keys(3, root="model")
    assert shared["gate_proj_weight"] == (
        "model.layers.3.mlp.shared_experts.gate_proj.weight"
    )


@pytest.mark.skipif(
    not Path("/models/zai-org/GLM-5.3/config.json").is_file(),
    reason="plain GLM-5.3 checkpoint is not mounted",
)
def test_real_plain_glm53_index_matches_compiler_contract():
    model_dir = Path("/models/zai-org/GLM-5.3")
    config = json.loads((model_dir / "config.json").read_text())
    layout = glm5_source_layout(config)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]

    assert layout.model_family == "glm_moe_dsa"
    assert layout.source_root == "model"
    assert layout.layer_count == 78
    assert layout.first_moe_layer == 3
    assert layout.expert_count == 256
    for layer, expert in ((3, 0), (77, 255), (78, 0)):
        assert all(
            name in weight_map
            for name in iq2r_glm5_source_keys(
                layer, expert, root=layout.source_root
            ).values()
        )


def test_interleave_gate_up_uses_aiter_swiglu_row_order():
    gate = torch.tensor([[1, 2], [3, 4]])
    up = torch.tensor([[10, 20], [30, 40]])
    assert torch.equal(
        interleave_gate_up(gate, up),
        torch.tensor([[1, 2], [10, 20], [3, 4], [30, 40]]),
    )


def test_block_fp8_dequantization_applies_each_2d_scale_block():
    weight = torch.ones((3, 4), dtype=torch.float32)
    scale = torch.tensor([[2.0, 3.0], [5.0, 7.0]])
    actual = dequantize_block_fp8(
        weight,
        scale,
        block_n=2,
        block_k=2,
        device="cpu",
    )
    assert torch.equal(
        actual,
        torch.tensor(
            [
                [2.0, 2.0, 3.0, 3.0],
                [2.0, 2.0, 3.0, 3.0],
                [5.0, 5.0, 7.0, 7.0],
            ]
        ),
    )


def test_loads_compact_glm_calibration_artifact(tmp_path):
    layout = _layout()
    targets = {}
    for layer in range(layout.moe_layers):
        targets[f"model.layers.{layer}.mlp.experts.gate_up_proj.weight"] = {
            "importance": torch.full(
                (layout.expert_count, layout.hidden_size), layer + 1.0
            )
        }
        targets[f"model.layers.{layer}.mlp.experts.down_proj.weight"] = {
            "importance": torch.full(
                (layout.expert_count, layout.intermediate_size), layer + 2.0
            )
        }
    path = tmp_path / "calibration.pt"
    torch.save(
        {
            "format": "iq2r-calibration",
            "version": 1,
            "scheme": "iq2r-diagonal-second-moment",
            "basis": "native",
            "targets": targets,
            "metadata": {
                "unobserved_target_groups": 0,
                "unobserved_policy": "error",
            },
        },
        path,
    )

    loaded = load_glm5_importance(path, layout)
    assert loaded.quality == "calibrated-o0"
    assert loaded.gate_up.shape == (2, 2, 4)
    assert loaded.down.shape == (2, 2, 2)
    # Each per-expert vector is normalized to mean 1.
    assert torch.equal(loaded.for_projection(3, "gate_up", 0), torch.ones(4))


@pytest.mark.parametrize("grouped", [False, True])
def test_loads_shared_expert_calibration_targets(tmp_path, grouped):
    layout = GLM5Layout(
        layer_count=5,
        first_moe_layer=3,
        expert_count=2,
        hidden_size=4,
        intermediate_size=2,
        block_n=2,
        block_k=2,
        model_family="glm_moe_dsa",
        source_root="model",
        shared_expert_count=1,
    )
    targets = {}
    for layer in range(layout.moe_layers):
        targets[f"model.layers.{layer}.mlp.experts.gate_up_proj.weight"] = {
            "importance": torch.ones(layout.expert_count, layout.hidden_size)
        }
        targets[f"model.layers.{layer}.mlp.experts.down_proj.weight"] = {
            "importance": torch.ones(layout.expert_count, layout.intermediate_size)
        }
        groups = (1,) if grouped else ()
        targets[f"model.layers.{layer}.mlp.shared_experts.gate_up_proj.weight"] = {
            "importance": torch.full(groups + (layout.hidden_size,), layer + 3.0)
        }
        targets[f"model.layers.{layer}.mlp.shared_experts.down_proj.weight"] = {
            "importance": torch.full(groups + (layout.intermediate_size,), layer + 4.0)
        }
    path = tmp_path / "calibration.pt"
    torch.save(
        {
            "format": "iq2r-calibration",
            "version": 1,
            "scheme": "iq2r-diagonal-second-moment",
            "basis": "native",
            "targets": targets,
            "metadata": {
                "unobserved_target_groups": 0,
                "unobserved_policy": "error",
            },
        },
        path,
    )

    loaded = load_glm5_importance(path, layout)
    assert loaded.shared_gate_up is not None
    assert loaded.shared_down is not None
    assert loaded.shared_gate_up.shape == (2, 4)
    assert loaded.shared_down.shape == (2, 2)
    assert torch.equal(loaded.for_projection(3, "gate_up", 2), torch.ones(4))


def test_uniform_importance_requires_explicit_diagnostic_mode():
    layout = _layout()
    with pytest.raises(ValueError, match="production IQ2R compilation requires"):
        load_glm5_importance(None, layout)

    loaded = load_glm5_importance(None, layout, diagnostic_uniform_importance=True)
    assert loaded.quality == "diagnostic-uniform-not-o0-quality"
    assert loaded.metadata["warning"] == "not O0 quality"


def _has_gfx950() -> bool:
    return torch.cuda.is_available() and "gfx950" in (
        torch.cuda.get_device_properties(0).gcnArchName
    )


def _write_source_model(path: Path) -> dict[str, torch.Tensor]:
    """Write a GLM-5.3-shaped FP8 model with placeholder tensors."""
    path.mkdir()
    shards = {
        "model-00001-of-00002.safetensors": {
            "model.embed_tokens.weight": torch.arange(4, dtype=torch.bfloat16),
            "model.layers.0.mlp.gate_proj.weight": torch.tensor([10]),
            "model.layers.3.self_attn.q_proj.weight": torch.tensor([12]),
        },
        "model-00002-of-00002.safetensors": {},
    }
    experts = shards["model-00002-of-00002.safetensors"]
    # Layer 5 is the MTP layer after num_hidden_layers. It must stay FP8.
    for layer in range(3, _LAYERS + 1):
        for index, name in enumerate(iq2r_glm5_shared_source_keys(layer).values()):
            experts[name] = torch.tensor([layer, -1, index])
        for expert in range(_EXPERTS):
            for index, name in enumerate(iq2r_glm5_source_keys(layer, expert).values()):
                experts[name] = torch.tensor([layer, expert, index])
    weight_map = {}
    for shard, tensors in shards.items():
        save_file(tensors, path / shard)
        weight_map.update(dict.fromkeys(tensors, shard))
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": _LAYERS,
        "first_k_dense_replace": 3,
        "n_routed_experts": _EXPERTS,
        "n_shared_experts": 1,
        "hidden_size": _HIDDEN,
        "moe_intermediate_size": _INTERMEDIATE,
        "quantization_config": {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
        },
    }
    (path / "config.json").write_text(json.dumps(config))
    (path / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": 0}, "weight_map": weight_map})
    )
    (path / "tokenizer.json").write_text("{}")
    (path / "README.md").write_text("source model card")
    return {
        name: value for tensors in shards.values() for name, value in tensors.items()
    }


def _compiled_source_keys(layers) -> set[str]:
    return {
        name
        for layer in layers
        for keys in (
            iq2r_glm5_shared_source_keys(layer),
            *(iq2r_glm5_source_keys(layer, expert) for expert in range(_EXPERTS)),
        )
        for name in keys.values()
    }


def test_checkpoint_files_replace_compiled_experts(tmp_path, monkeypatch):
    source, output = tmp_path / "source", tmp_path / "out"
    tensors = _write_source_model(source)
    output.mkdir()
    for layer in (3, 4):
        for projection, suffix in (("gate_up", "gate-up"), ("down", "down")):
            keys = iq2r_compiled_tensor_keys(layer, projection)
            save_file(
                {name: torch.zeros(2, dtype=torch.uint8) for name in keys.values()},
                output / f"iq2r-layer-{layer:04d}-{suffix}.safetensors",
            )
    # One tensor per shard, so the shards cross the source-file boundary.
    monkeypatch.setattr(compile_module, "_SHARD_BYTES", 1)
    config = json.loads((source / "config.json").read_text())
    _write_checkpoint_files(
        source,
        output,
        config,
        json.loads((source / "model.safetensors.index.json").read_text())["weight_map"],
        glm5_source_layout(config),
        safe_open,
        save_file,
    )

    index = json.loads((output / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    compiled = _compiled_source_keys((3, 4))
    kept = {name: value for name, value in tensors.items() if name not in compiled}
    # Embedding, dense MLP, attention and the 3 MTP-layer experts.
    assert len(kept) == 3 + 6 * (_EXPERTS + 1)
    assert not compiled & set(weight_map)
    assert len({weight_map[name] for name in kept}) == len(kept)
    for name, value in kept.items():
        with safe_open(output / weight_map[name], "pt") as handle:
            assert torch.equal(handle.get_tensor(name), value)
    for layer in (3, 4):
        for projection, suffix in (("gate_up", "gate-up"), ("down", "down")):
            for name in iq2r_compiled_tensor_keys(layer, projection).values():
                assert weight_map[name] == (
                    f"iq2r-layer-{layer:04d}-{suffix}.safetensors"
                )
    # The 12 placeholder layer tensors hold 2 bytes each.
    assert index["metadata"]["total_size"] == (
        sum(value.nbytes for value in kept.values()) + 12 * 2
    )

    quantization = json.loads((output / "config.json").read_text())[
        "quantization_config"
    ]
    assert quantization["quant_method"] == "iq2r"
    assert quantization["base_quantization_config"] == config["quantization_config"]
    assert quantization["iq2r_layout"] == LAYOUT
    patterns = quantization["iq2r_modules"]
    # ATOM matches a pattern without "*" as a substring of the module name.
    assert all("*" in pattern for pattern in patterns)

    def is_iq2r(module: str) -> bool:
        return any(fnmatch.fnmatch(module, pattern) for pattern in patterns)

    for layer in (3, 4):
        assert is_iq2r(f"model.layers.{layer}.mlp.experts")
        assert is_iq2r(f"model.layers.{layer}.mlp.shared_experts")
    for module in (
        "model.layers.5.mlp.experts",
        "model.layers.30.mlp.experts",
        "model.layers.3.self_attn.q_proj",
    ):
        assert not is_iq2r(module)
    assert (output / "tokenizer.json").read_text() == "{}"
    assert not (output / "README.md").exists()


def test_resume_validates_existing_layer_file(tmp_path):
    layout = GLM5Layout(
        layer_count=4,
        first_moe_layer=3,
        expert_count=2,
        hidden_size=32,
        intermediate_size=32,
        block_n=16,
        block_k=16,
        shared_expert_count=1,
    )
    metadata = IQ2RMetadata(logical_n=64, logical_k=32)
    keys = iq2r_compiled_tensor_keys(3, "gate_up")
    shard = tmp_path / "iq2r-layer-0003-gate-up.safetensors"
    file_metadata = {
        "format": "pt",
        "iq2r_format": IQ2R_FORMAT_NAME,
        "iq2r_activation_basis": IQ2R_ACTIVATION_BASIS,
        "iq2r_layer": "3",
        "iq2r_projection": "gate_up",
        "iq2r_quality": _QUALITY,
        "iq2r_layout": LAYOUT,
    }
    tensors = {
        keys["data"]: torch.zeros((3, iq2r_glm53_gate_bytes(64)), dtype=torch.uint8),
        keys["auxiliary"]: torch.zeros(
            (3, metadata.auxiliary_bytes), dtype=torch.uint8
        ),
        keys["tile_n"]: torch.tensor([128], dtype=torch.int32),
    }
    save_file(tensors, shard, metadata=file_metadata)
    _validate_projection_shard(shard, layout, 3, "gate_up", _QUALITY, safe_open)

    # Unpacked gate/up records have a different size.
    unpacked = dict(tensors)
    unpacked[keys["data"]] = torch.zeros((3, metadata.data_bytes), dtype=torch.uint8)
    save_file(unpacked, shard, metadata=file_metadata)
    with pytest.raises(ValueError, match="invalid compiled shard.*expected U8"):
        _validate_projection_shard(shard, layout, 3, "gate_up", _QUALITY, safe_open)

    save_file(tensors, shard, metadata={**file_metadata, "iq2r_quality": "other"})
    with pytest.raises(ValueError, match="metadata 'iq2r_quality'"):
        _validate_projection_shard(shard, layout, 3, "gate_up", _QUALITY, safe_open)


@pytest.mark.skipif(not _has_gfx950(), reason="requires gfx950")
def test_compile_packs_layers_and_writes_checkpoint(tmp_path, monkeypatch):
    from aiter.iq2r_glm53 import iq2r_glm53_pack

    source, output = tmp_path / "source", tmp_path / "out"
    _write_source_model(source)
    generator = torch.Generator().manual_seed(0x5310)
    encoded, calls = {}, []

    def fake_encode(reader, layout, importance, layer, projection, **kwargs):
        # Random records stand in for the encoder, which test_iq2r_hip covers.
        metadata = _projection_metadata(layout, projection)
        encoded[layer, projection] = tuple(
            torch.randint(
                0, 256, (_EXPERTS + 1, size), generator=generator, dtype=torch.uint8
            )
            for size in (metadata.data_bytes, metadata.auxiliary_bytes)
        )
        calls.append(layer)
        return encoded[layer, projection]

    monkeypatch.setattr(compile_module, "_encode_projection", fake_encode)
    run = functools.partial(
        compile_glm5_iq2r, source, output, diagnostic_uniform_importance=True
    )
    assert run(layer_indices=[3]) is None
    assert not (output / "config.json").exists()
    with pytest.raises(FileExistsError, match="--resume"):
        run()
    config_path = run(resume=True)
    # The resumed run validates layer 3 and only encodes layer 4.
    assert calls == [3, 3, 4, 4]

    for layer in (3, 4):
        expected = iq2r_glm53_pack(
            *(
                tensor.cuda()
                for projection in ("gate_up", "down")
                for tensor in encoded[layer, projection]
            ),
            intermediate_size=_INTERMEDIATE,
        )
        for projection, suffix, values in (
            ("gate_up", "gate-up", expected[:2]),
            ("down", "down", expected[2:]),
        ):
            keys = iq2r_compiled_tensor_keys(layer, projection)
            path = output / f"iq2r-layer-{layer:04d}-{suffix}.safetensors"
            with safe_open(path, "pt") as handle:
                assert handle.metadata()["iq2r_layout"] == LAYOUT
                assert handle.metadata()["iq2r_quality"] == _QUALITY
                assert torch.equal(handle.get_tensor(keys["data"]), values[0].cpu())
                assert torch.equal(
                    handle.get_tensor(keys["auxiliary"]), values[1].cpu()
                )
    weight_map = json.loads((output / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    assert all((output / name).is_file() for name in set(weight_map.values()))
    config = json.loads(config_path.read_text())
    assert config["quantization_config"]["iq2r_layout"] == LAYOUT


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
