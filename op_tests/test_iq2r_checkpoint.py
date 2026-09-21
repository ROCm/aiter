# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from aiter.iq2r_checkpoint import (
    iq2r_compiled_tensor_keys,
    load_iq2r_layer_checkpoint,
)
from aiter.ops.iq2r_format import (
    IQ2R_FORMAT_NAME,
    IQ2R_FORMAT_VERSION,
    IQ2RMetadata,
)


def test_compiled_tensor_keys_are_unambiguous():
    keys = iq2r_compiled_tensor_keys(7, "gate_up")
    assert keys["data"] == ("model.layers.7.mlp.experts.gate_up_proj.0.iq2r_data")
    assert keys["auxiliary"].endswith(".iq2r_auxiliary")
    with pytest.raises(ValueError, match="projection"):
        iq2r_compiled_tensor_keys(0, "sideways")

    glm_keys = iq2r_compiled_tensor_keys(3, "gate_up", module_name="up_gate_proj")
    assert glm_keys["data"] == ("model.layers.3.mlp.up_gate_proj.0.iq2r_data")


def test_loader_rejects_wrong_basis(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    config = {
        "num_hidden_layers": 1,
        "num_local_experts": 128,
        "compiled_tensor_parallel_size": 1,
        "compiled_expert_parallel_size": 1,
        "iq2r": {
            "format": IQ2R_FORMAT_NAME,
            "version": IQ2R_FORMAT_VERSION,
            "activation_basis": "hadamard",
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="checkpoint identity"):
        load_iq2r_layer_checkpoint(tmp_path, 0, expert_count=1)


@pytest.mark.parametrize(
    ("architecture", "model_type", "nested_config"),
    [
        ("Glm5NextForCausalLM", "glm5_next", True),
        ("GlmMoeDsaForCausalLM", "glm_moe_dsa", False),
    ],
)
def test_loads_bias_free_glm_checkpoint_and_slices_experts(
    tmp_path, architecture, model_type, nested_config
):
    experts = 2
    hidden_size = 128
    intermediate_size = 64
    gate_metadata = IQ2RMetadata(logical_n=2 * intermediate_size, logical_k=hidden_size)
    down_metadata = IQ2RMetadata(logical_n=hidden_size, logical_k=intermediate_size)
    gate_keys = iq2r_compiled_tensor_keys(3, "gate_up", module_name="up_gate_proj")
    down_keys = iq2r_compiled_tensor_keys(3, "down", module_name="down_proj")
    tensors = {
        gate_keys["data"]: torch.zeros(
            experts, gate_metadata.data_bytes, dtype=torch.uint8
        ),
        gate_keys["auxiliary"]: torch.zeros(
            experts, gate_metadata.auxiliary_bytes, dtype=torch.uint8
        ),
        gate_keys["tile_n"]: torch.tensor([128], dtype=torch.int32),
        down_keys["data"]: torch.zeros(
            experts, down_metadata.data_bytes, dtype=torch.uint8
        ),
        down_keys["auxiliary"]: torch.zeros(
            experts, down_metadata.auxiliary_bytes, dtype=torch.uint8
        ),
        down_keys["tile_n"]: torch.tensor([128], dtype=torch.int32),
    }
    save_file(tensors, tmp_path / "flywheel_model.0.safetensors")
    dimensions = {
        "num_hidden_layers": 5,
        "first_k_dense_replace": 3,
        "n_routed_experts": experts,
        "hidden_size": hidden_size,
        "moe_intermediate_size": intermediate_size,
    }
    config = {
        "architectures": [architecture],
        "model_type": model_type,
        "compiled_tensor_parallel_size": 1,
        "compiled_expert_parallel_size": 1,
        "file_manifest": ["flywheel_model.0.safetensors"],
        "iq2r": {
            "format": IQ2R_FORMAT_NAME,
            "version": IQ2R_FORMAT_VERSION,
            "activation_basis": "native",
        },
    }
    if nested_config:
        config["text_config"] = dimensions
    else:
        config.update(dimensions)
    (tmp_path / "config.json").write_text(json.dumps(config))

    loaded = load_iq2r_layer_checkpoint(tmp_path, 3, expert_start=1, expert_count=1)
    assert loaded.model_family == "glm5"
    assert loaded.total_experts == experts
    assert loaded.gate_up_data.shape == (1, gate_metadata.data_bytes)
    assert loaded.down_data.shape == (1, down_metadata.data_bytes)
    assert loaded.gate_up_bias is None
    assert loaded.down_bias is None

    with pytest.raises(ValueError, match="MoE layer range"):
        load_iq2r_layer_checkpoint(tmp_path, 2, expert_count=1)


@pytest.mark.skipif(
    not Path("/models/openai/gpt-oss-120b-o0-e132-profile/config.json").is_file(),
    reason="compiled GPT-OSS O0 fixture is not mounted",
)
def test_real_compiled_o0_checkpoint_slice():
    loaded = load_iq2r_layer_checkpoint(
        "/models/openai/gpt-oss-120b-o0-e132-profile",
        0,
        expert_start=0,
        expert_count=1,
    )
    assert loaded.gate_up_data.shape == (1, 4_945_920)
    assert loaded.down_data.shape == (1, 2_472_960)
    assert torch.count_nonzero(loaded.gate_up_auxiliary[:, :8]).item() == 0
    assert torch.count_nonzero(loaded.down_auxiliary[:, :8]).item() == 0
    assert loaded.gate_up_bias.dtype == loaded.down_bias.dtype == torch.bfloat16
