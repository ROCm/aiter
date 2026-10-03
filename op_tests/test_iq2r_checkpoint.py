# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from pathlib import Path

import pytest
import torch

from aiter.iq2r_checkpoint import (
    iq2r_compiled_tensor_keys,
    load_iq2r_layer_checkpoint,
)
from aiter.ops.iq2r_format import (
    IQ2R_FORMAT_NAME,
    IQ2R_FORMAT_VERSION,
)


def test_compiled_tensor_keys_are_unambiguous():
    keys = iq2r_compiled_tensor_keys(7, "gate_up")
    assert keys["data"] == ("model.layers.7.mlp.experts.gate_up_proj.0.iq2r_data")
    assert keys["auxiliary"].endswith(".iq2r_auxiliary")
    with pytest.raises(ValueError, match="projection"):
        iq2r_compiled_tensor_keys(0, "sideways")


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
