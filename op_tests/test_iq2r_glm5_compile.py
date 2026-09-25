# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from aiter.iq2r_checkpoint import (
    iq2r_compiled_tensor_keys,
    iq2r_glm5_shared_source_keys,
    iq2r_glm5_source_keys,
)
from aiter.iq2r_glm5_compile import (
    GLM5Importance,
    GLM5Layout,
    _compiled_config,
    _validate_projection_shard,
    dequantize_block_fp8,
    glm5_source_layout,
    interleave_gate_up,
    load_glm5_importance,
)
from aiter.ops.iq2r_format import (
    IQ2R_ACTIVATION_BASIS,
    IQ2R_FORMAT_NAME,
    IQ2R_FORMAT_VERSION,
    IQ2RMetadata,
)


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


def test_source_layout_supports_flash_and_plain_glm53():
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
    flash = glm5_source_layout(
        {
            "architectures": ["Glm5NextForConditionalGeneration"],
            "model_type": "glm5_next",
            "text_config": dimensions,
            "quantization_config": quantization_config,
        }
    )
    plain = glm5_source_layout(
        {
            "architectures": ["GlmMoeDsaForCausalLM"],
            "model_type": "glm_moe_dsa",
            **dimensions,
            "quantization_config": quantization_config,
        }
    )

    assert flash.model_family == "glm5_next"
    assert flash.source_root == "model.language_model"
    assert plain.model_family == "glm_moe_dsa"
    assert plain.source_root == "model"
    assert plain.layer_count == 78
    assert plain.expert_count == 256
    assert plain.shared_expert_count == 1
    assert plain.hidden_size == 6144
    assert plain.intermediate_size == 2048


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


def test_plain_glm53_compiled_config_keeps_dimensions_top_level():
    layout = GLM5Layout(
        layer_count=78,
        first_moe_layer=3,
        expert_count=256,
        hidden_size=6144,
        intermediate_size=2048,
        block_n=128,
        block_k=128,
        model_family="glm_moe_dsa",
        source_root="model",
        shared_expert_count=1,
    )
    importance = GLM5Importance(
        gate_up=torch.empty(0),
        down=torch.empty(0),
        metadata={"calibration_scheme": "diagnostic-uniform"},
        quality="diagnostic-uniform-not-o0-quality",
    )
    config = _compiled_config(
        {
            "architectures": ["GlmMoeDsaForCausalLM"],
            "model_type": "glm_moe_dsa",
        },
        layout,
        importance,
        [3],
        ["iq2r-layer-0003-gate-up.safetensors"],
        "/models/zai-org/GLM-5.3",
        "abc123",
    )

    assert "text_config" not in config
    assert config["model_type"] == "glm_moe_dsa"
    assert config["num_hidden_layers"] == 78
    assert config["n_routed_experts"] == 256
    assert config["hidden_size"] == 6144
    assert config["moe_intermediate_size"] == 2048
    assert config["iq2r"]["model_family"] == "glm_moe_dsa"
    assert config["iq2r"]["source_root"] == "model"

    fused = _compiled_config(
        {
            "architectures": ["GlmMoeDsaForCausalLM"],
            "model_type": "glm_moe_dsa",
        },
        layout,
        importance,
        [3],
        ["iq2r-layer-0003-gate-up.safetensors"],
        "/models/zai-org/GLM-5.3",
        "abc123",
        True,
    )
    assert fused["n_routed_experts"] == 256
    assert fused["iq2r"]["compiled_expert_count"] == 257
    assert fused["iq2r"]["fused_shared_expert"] is True


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


def test_loads_compact_redline_glm_calibration_artifact(tmp_path):
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
            "format": "redline-calibration",
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
    # Per-expert vectors are normalized exactly as Redline's compiler does.
    assert torch.equal(loaded.for_projection(3, "gate_up", 0), torch.ones(4))


def test_loads_shared_expert_redline_calibration_targets(tmp_path):
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
        targets[f"model.layers.{layer}.mlp.shared_experts.gate_up_proj.weight"] = {
            "importance": torch.full((layout.hidden_size,), layer + 3.0)
        }
        targets[f"model.layers.{layer}.mlp.shared_experts.down_proj.weight"] = {
            "importance": torch.full((layout.intermediate_size,), layer + 4.0)
        }
    path = tmp_path / "calibration.pt"
    torch.save(
        {
            "format": "redline-calibration",
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


def test_resume_validates_existing_projection_shard(tmp_path):
    layout = GLM5Layout(
        layer_count=4,
        first_moe_layer=3,
        expert_count=2,
        hidden_size=32,
        intermediate_size=32,
        block_n=16,
        block_k=16,
    )
    metadata = IQ2RMetadata(logical_n=64, logical_k=32)
    keys = iq2r_compiled_tensor_keys(3, "gate_up", module_name="up_gate_proj")
    data_key = keys["data"]
    auxiliary_key = keys["auxiliary"]
    tile_key = keys["tile_n"]
    shard = tmp_path / "iq2r-layer-0003-gate-up.safetensors"
    file_metadata = {
        "format": "pt",
        "iq2r_format": IQ2R_FORMAT_NAME,
        "iq2r_format_version": str(IQ2R_FORMAT_VERSION),
        "iq2r_activation_basis": IQ2R_ACTIVATION_BASIS,
        "iq2r_layer": "3",
        "iq2r_projection": "gate_up",
        "iq2r_quality": "diagnostic-uniform-not-o0-quality",
    }
    save_file(
        {
            data_key: torch.zeros((2, metadata.data_bytes), dtype=torch.uint8),
            auxiliary_key: torch.zeros(
                (2, metadata.auxiliary_bytes), dtype=torch.uint8
            ),
            tile_key: torch.tensor([128], dtype=torch.int32),
        },
        shard,
        metadata=file_metadata,
    )

    _validate_projection_shard(
        shard,
        layout,
        3,
        "gate_up",
        "diagnostic-uniform-not-o0-quality",
        safe_open,
    )

    save_file(
        {
            data_key: torch.zeros((1, metadata.data_bytes), dtype=torch.uint8),
            auxiliary_key: torch.zeros(
                (2, metadata.auxiliary_bytes), dtype=torch.uint8
            ),
            tile_key: torch.tensor([128], dtype=torch.int32),
        },
        shard,
        metadata=file_metadata,
    )
    with pytest.raises(ValueError, match="invalid resume shard.*expected U8"):
        _validate_projection_shard(
            shard,
            layout,
            3,
            "gate_up",
            "diagnostic-uniform-not-o0-quality",
            safe_open,
        )
