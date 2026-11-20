# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import fnmatch
import functools
import hashlib
import json
import re
from pathlib import Path

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import aiter.quantization.iq2r.glm5.compile as compile_module
from aiter.iq2r_glm53 import (
    iq2r_glm53_aux_bytes,
    iq2r_glm53_gate_bytes,
    iq2r_glm53_pack,
)
from aiter.ops.iq2r_format import IQ2R_ACTIVATION_BASIS, IQ2RMetadata
from aiter.quantization.iq2r.glm5.compile import (
    FORMAT,
    LAYOUT,
    GLM5Importance,
    GLM5Layout,
    _encode_projection,
    _validate_projection_shard,
    _write_checkpoint_files,
    compile_glm5_iq2r,
    dequantize_block_fp8,
    glm5_source_layout,
    interleave_gate_up,
    iq2r_compiled_tensor_keys,
    iq2r_glm5_source_keys,
    load_glm5_importance,
    main,
)

_HIDDEN = 6144
_INTERMEDIATE = 2048
_EXPERTS = 2
_LAYERS = 5
_QUALITY = "diagnostic-uniform-not-o0-quality"
# Every layer file records these; a later run keeps only matching files.
_ENCODING = {
    "iq2r_quality": _QUALITY,
    "iq2r_calibration_sha256": "uniform",
    "iq2r_codebook_iterations": "4",
    "iq2r_codebook_sample_vectors": "65536",
    "iq2r_v2_levels": "c4e4q",
    "iq2r_v2_target_gain": "0.98",
    "iq2r_v2_min_routed_tokens": "256",
    "iq2r_v2_shared_expert": "fp8",
}


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


def _shared_source_keys(layer: int) -> list[str]:
    prefix = f"model.layers.{layer}.mlp.shared_experts"
    return [
        f"{prefix}.{projection}.{kind}"
        for projection in ("gate_proj", "up_proj", "down_proj")
        for kind in ("weight", "weight_scale_inv")
    ]


def test_source_layout_reads_glm53_config():
    config = {
        "architectures": ["GlmMoeDsaForCausalLM"],
        "model_type": "glm_moe_dsa",
        "num_hidden_layers": 78,
        "first_k_dense_replace": 3,
        "n_routed_experts": 256,
        "n_shared_experts": 1,
        "num_experts_per_tok": 8,
        "hidden_size": 6144,
        "moe_intermediate_size": 2048,
        "quantization_config": {
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
        },
    }
    layout = glm5_source_layout(config)
    assert layout == GLM5Layout(78, 3, 256, 6144, 2048, 128, 128)
    assert layout.moe_layers == 75
    with pytest.raises(ValueError, match="GLM MoE DSA"):
        glm5_source_layout({**config, "architectures": [], "model_type": "llama"})
    with pytest.raises(ValueError, match="one shared expert"):
        glm5_source_layout({**config, "n_shared_experts": 2})
    # Other GLM MoE DSA shapes would compile but not serve.
    for name, value in (
        ("hidden_size", 5120),
        ("moe_intermediate_size", 1536),
        ("n_routed_experts", 160),
        ("num_experts_per_tok", 6),
    ):
        with pytest.raises(ValueError, match=re.escape(f"has {{{name!r}: {value}}}")):
            glm5_source_layout({**config, name: value})


def test_source_and_compiled_keys_match_glm53_checkpoint_names():
    names = iq2r_glm5_source_keys(3, 255)
    assert names["gate_proj_weight"] == (
        "model.layers.3.mlp.experts.255.gate_proj.weight"
    )
    assert names["down_proj_weight_scale_inv"] == (
        "model.layers.3.mlp.experts.255.down_proj.weight_scale_inv"
    )
    assert len(names) == 6
    assert iq2r_compiled_tensor_keys(7, "gate_up") == {
        "data": "model.layers.7.mlp.up_gate_proj.0.iq2r_data",
        "auxiliary": "model.layers.7.mlp.up_gate_proj.0.iq2r_auxiliary",
        "tile_n": "model.layers.7.mlp.up_gate_proj.0.iq2r_tN",
    }
    assert iq2r_compiled_tensor_keys(7, "down")["data"] == (
        "model.layers.7.mlp.down_proj.0.iq2r_data"
    )
    assert LAYOUT == "glm53-packed-v2"
    assert FORMAT == "iq2r-512-fullsign-c4e4q-e8m0d256-native-v2"


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


def _artifact(layout: GLM5Layout, *, shared: bool, hits: bool = True) -> dict:
    """A calibration artifact; ``shared`` adds the targets earlier versions wrote."""

    widths = {"gate_up": layout.hidden_size, "down": layout.intermediate_size}
    modules = [("experts", layout.expert_count, 1.0)]
    if shared:
        modules.append(("shared_experts", 1, 3.0))
    targets = {}
    for layer in range(layout.first_moe_layer, layout.layer_count):
        for module, groups, offset in modules:
            for projection, width in widths.items():
                target = {"importance": torch.full((groups, width), layer + offset)}
                if hits:
                    target["hit_count"] = torch.tensor([300, 40][:groups]) + layer
                    target["forced_hit_count"] = torch.tensor([0, 30][:groups])
                targets[
                    f"model.layers.{layer}.mlp.{module}.{projection}_proj.weight"
                ] = target
    return {
        "format": "iq2r-calibration",
        "version": 1,
        "scheme": "iq2r-diagonal-second-moment",
        "basis": "native",
        "targets": targets,
    }


def _hash(tensors) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.contiguous().view(torch.uint8).numpy())
    return digest.hexdigest()


def test_loads_calibration_artifact(tmp_path):
    layout = _layout()
    path = tmp_path / "calibration.pt"
    artifact = _artifact(layout, shared=False)
    torch.save(artifact, path)

    loaded = load_glm5_importance(path, layout)
    assert loaded.quality == "calibrated-o0"
    assert torch.equal(loaded.gate_up[:, 0, 0], torch.tensor([4.0, 5.0]))
    assert loaded.down.shape == (2, 2, 2)
    # Each vector is normalized to mean 1.
    for expert in range(layout.expert_count):
        assert torch.equal(loaded.for_projection(3, "gate_up", expert), torch.ones(4))
    targets = artifact["targets"]
    stacked = [
        torch.stack(
            [
                targets[f"model.layers.{layer}.mlp.experts.{projection}_proj.weight"][
                    "importance"
                ]
                for layer in (3, 4)
            ]
        )
        for projection in ("gate_up", "down")
    ]
    assert loaded.sha256 == _hash(stacked)

    # Any change to the importance changes the identity the layer files record.
    targets["model.layers.3.mlp.experts.down_proj.weight"]["importance"][1, 0] += 1
    torch.save(artifact, path)
    assert load_glm5_importance(path, layout).sha256 != loaded.sha256

    del targets["model.layers.4.mlp.experts.down_proj.weight"]
    torch.save(artifact, path)
    with pytest.raises(ValueError, match="experts down targets cover"):
        load_glm5_importance(path, layout)


def test_shared_expert_targets_are_ignored_but_hashed_as_before(tmp_path):
    """Artifacts that also hold shared-expert targets load to the same routed
    importance, and keep the hash the layer files built from them record."""

    layout = _layout()
    routed_path, shared_path = tmp_path / "routed.pt", tmp_path / "shared.pt"
    torch.save(_artifact(layout, shared=False), routed_path)
    artifact = _artifact(layout, shared=True)
    torch.save(artifact, shared_path)
    routed = load_glm5_importance(routed_path, layout)
    loaded = load_glm5_importance(shared_path, layout)
    assert torch.equal(loaded.gate_up, routed.gate_up)
    assert torch.equal(loaded.down, routed.down)
    assert torch.equal(loaded.routed_hits, routed.routed_hits)
    # The earlier hash: routed gate/up, routed down, shared gate/up, shared down,
    # each stacked over the MoE layers.
    targets = artifact["targets"]
    previous = _hash(
        torch.stack(
            [
                targets[f"model.layers.{layer}.mlp.{module}.{projection}_proj.weight"][
                    "importance"
                ]
                for layer in (3, 4)
            ]
        )
        for module in ("experts", "shared_experts")
        for projection in ("gate_up", "down")
    )
    assert loaded.sha256 == previous
    assert routed.sha256 != previous


def test_calibration_hit_counts_exclude_forced_fills(tmp_path):
    layout = _layout()
    path = tmp_path / "calibration.pt"
    torch.save(_artifact(layout, shared=True), path)
    loaded = load_glm5_importance(path, layout)
    assert [loaded.routed_tokens(4, expert) for expert in (0, 1)] == [304, 14]
    assert (
        load_glm5_importance(
            None, layout, diagnostic_uniform_importance=True
        ).routed_tokens(3, 0)
        is None
    )


def test_uniform_importance_requires_explicit_diagnostic_mode():
    layout = _layout()
    with pytest.raises(ValueError, match="requires a calibration cache"):
        load_glm5_importance(None, layout)

    loaded = load_glm5_importance(None, layout, diagnostic_uniform_importance=True)
    assert loaded.quality == _QUALITY
    assert loaded.sha256 == "uniform"
    assert loaded.down.shape == (2, 2, 2)


def test_compile_refuses_to_write_into_the_source(tmp_path):
    source = tmp_path / "source"
    for output in (source, source / "out", tmp_path):
        with pytest.raises(ValueError, match="same or nested"):
            compile_glm5_iq2r(source, output, diagnostic_uniform_importance=True)


@pytest.fixture
def two_experts(monkeypatch):
    # Two routed experts instead of 256 keep the checkpoint tests small.
    monkeypatch.setitem(compile_module._GLM53_SHAPE, "n_routed_experts", _EXPERTS)


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
        for index, name in enumerate(_shared_source_keys(layer)):
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
        "num_experts_per_tok": 8,
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


def test_compile_requires_calibration_hit_counts(tmp_path, two_experts):
    source = tmp_path / "source"
    _write_source_model(source)
    path = tmp_path / "calibration.pt"
    layout = glm5_source_layout(json.loads((source / "config.json").read_text()))
    artifact = _artifact(
        GLM5Layout(_LAYERS, 3, _EXPERTS, _HIDDEN, _INTERMEDIATE, 128, 128),
        shared=False,
        hits=False,
    )
    torch.save(artifact, path)
    # The artifact loads; compiling from it needs the hit counts.
    assert load_glm5_importance(path, layout).routed_hits is None
    with pytest.raises(ValueError, match="no routed hit counts"):
        compile_glm5_iq2r(source, tmp_path / "out", calibration_cache=path)
    assert not (tmp_path / "out").exists()


def test_checkpoint_keeps_shared_expert_fp8(tmp_path, monkeypatch, two_experts):
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
    )

    index = json.loads((output / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    compiled = {
        name
        for layer in (3, 4)
        for expert in range(_EXPERTS)
        for name in iq2r_glm5_source_keys(layer, expert).values()
    }
    kept = {name: value for name, value in tensors.items() if name not in compiled}
    # Embedding, dense MLP, attention, the shared experts of layers 3 and 4 and
    # the routed and shared experts of the MTP layer stay FP8.
    assert len(kept) == 3 + 6 * 2 + 6 * (_EXPERTS + 1)
    for layer in (3, 4):
        assert set(_shared_source_keys(layer)) <= set(weight_map)
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
    assert quantization == {
        "quant_method": "iq2r",
        "base_quantization_config": config["quantization_config"],
        "iq2r_modules": ["model.layers.3.mlp.experts*", "model.layers.4.mlp.experts*"],
        "iq2r_layout": "glm53-packed-v2",
        "iq2r_format": "iq2r-512-fullsign-c4e4q-e8m0d256-native-v2",
    }
    patterns = quantization["iq2r_modules"]
    # ATOM matches a pattern without "*" as a substring of the module name.
    assert all("*" in pattern for pattern in patterns)

    def is_iq2r(module: str) -> bool:
        return any(fnmatch.fnmatch(module, pattern) for pattern in patterns)

    for layer in (3, 4):
        assert is_iq2r(f"model.layers.{layer}.mlp.experts")
        assert not is_iq2r(f"model.layers.{layer}.mlp.shared_experts")
    for module in (
        "model.layers.5.mlp.experts",
        "model.layers.30.mlp.experts",
        "model.layers.3.self_attn.q_proj",
    ):
        assert not is_iq2r(module)
    assert (output / "tokenizer.json").read_text() == "{}"
    assert not (output / "README.md").exists()


@pytest.mark.skipif(not _has_gfx950(), reason="requires gfx950")
def test_encodes_routed_experts_with_low_token_fallback(monkeypatch):
    layout = GLM5Layout(
        layer_count=4,
        first_moe_layer=3,
        expert_count=2,
        hidden_size=256,
        intermediate_size=256,
        block_n=128,
        block_k=128,
    )
    generator = torch.Generator(device="cuda").manual_seed(0x5320)
    weights = [
        torch.randn((256, 256), generator=generator, device="cuda") * (0.01 + 0.01 * e)
        for e in range(2)
    ]
    channel = torch.linspace(0.5, 2.0, 256)
    importance = GLM5Importance(
        torch.stack([channel, channel])[None],
        torch.stack([channel, channel])[None],
        3,
        "calibrated-o0",
        "test",
        torch.tensor([[1000, 255]]),
    )
    seen = []
    learn = compile_module.iq2r_v2_learn_codebook

    def record(weight, expert_importance, *args, **kwargs):
        seen.append(expert_importance.cpu())
        return learn(weight, expert_importance, *args, **kwargs)

    monkeypatch.setattr(compile_module, "iq2r_v2_learn_codebook", record)
    monkeypatch.setattr(
        compile_module,
        "_source_projection",
        lambda reader, layout, layer, expert, projection, device: weights[expert],
    )
    data, auxiliary = _encode_projection(
        None,
        layout,
        importance,
        3,
        "down",
        device="cuda",
        iterations=1,
        sample_vectors=4096,
    )
    # Routed experts only: the shared expert stays FP8.
    assert data.shape[0] == auxiliary.shape[0] == 2
    assert torch.allclose(seen[0], channel / channel.mean())
    energy = weights[1].square().sum(dim=0).cpu()
    assert torch.allclose(seen[1], (energy / energy.mean()).clamp_min(1e-6))


def _layer_tensors(layout: GLM5Layout, layer: int, projection: str) -> dict:
    metadata = IQ2RMetadata(
        logical_n=(
            2 * layout.intermediate_size
            if projection == "gate_up"
            else layout.hidden_size
        ),
        logical_k=(
            layout.hidden_size if projection == "gate_up" else layout.intermediate_size
        ),
    )
    data_bytes = (
        iq2r_glm53_gate_bytes(metadata.logical_n)
        if projection == "gate_up"
        else metadata.data_bytes
    )
    aux_bytes = iq2r_glm53_aux_bytes(
        metadata.logical_n, metadata.logical_k, 4, gate=projection == "gate_up"
    )
    keys = iq2r_compiled_tensor_keys(layer, projection)
    return {
        keys["data"]: torch.zeros((layout.expert_count, data_bytes), dtype=torch.uint8),
        keys["auxiliary"]: torch.zeros(
            (layout.expert_count, aux_bytes), dtype=torch.uint8
        ),
        keys["tile_n"]: torch.tensor([128], dtype=torch.int32),
    }


def _file_metadata(layer: int, projection: str, entries: int = 512) -> dict[str, str]:
    extra = {} if entries == 512 else {"iq2r_codebook_entries": str(entries)}
    return {
        "format": "pt",
        "iq2r_format": (
            FORMAT if entries == 512 else "iq2r-1024-fullsign-c4e4q-e8m0d256-native-v2"
        ),
        "iq2r_activation_basis": IQ2R_ACTIVATION_BASIS,
        "iq2r_layer": str(layer),
        "iq2r_projection": projection,
        **extra,
        **_ENCODING,
        "iq2r_layout": LAYOUT if entries == 512 else "glm53-packed-v3",
    }


def test_existing_layer_file_validation(tmp_path):
    layout = GLM5Layout(
        layer_count=4,
        first_moe_layer=3,
        expert_count=2,
        hidden_size=_HIDDEN,
        intermediate_size=256,
        block_n=128,
        block_k=128,
    )
    for projection in ("gate_up", "down"):
        shard = tmp_path / f"{projection}.safetensors"
        tensors = _layer_tensors(layout, 3, projection)
        metadata = _file_metadata(3, projection)
        save_file(tensors, shard, metadata=metadata)
        _validate_projection_shard(shard, layout, 3, projection, _ENCODING)

        # A file with the shared expert fused as a third expert is rejected.
        keys = iq2r_compiled_tensor_keys(3, projection)
        fused = dict(tensors)
        fused[keys["data"]] = torch.zeros(
            (3, tensors[keys["data"]].shape[1]), dtype=torch.uint8
        )
        save_file(fused, shard, metadata=metadata)
        with pytest.raises(ValueError, match=r"expected U8 \[2, .*delete it"):
            _validate_projection_shard(shard, layout, 3, projection, _ENCODING)

        # A layer encoded another way is rejected.
        for name in (*_ENCODING, "iq2r_format", "iq2r_layout"):
            save_file(tensors, shard, metadata={**metadata, name: "other"})
            with pytest.raises(ValueError, match=f"metadata '{name}'"):
                _validate_projection_shard(shard, layout, 3, projection, _ENCODING)


def test_compile_keeps_valid_layer_files_and_rejects_others(
    tmp_path, monkeypatch, two_experts
):
    source, output = tmp_path / "source", tmp_path / "out"
    _write_source_model(source)
    calls = []

    def fake_compile_layer(reader, layout, importance, layer, output_dir, **kwargs):
        calls.append(layer)
        for projection, suffix in (("gate_up", "gate-up"), ("down", "down")):
            path = output_dir / f"iq2r-layer-{layer:04d}-{suffix}.safetensors"
            if not path.exists():
                save_file(
                    _layer_tensors(layout, layer, projection),
                    path,
                    metadata=_file_metadata(layer, projection),
                )

    monkeypatch.setattr(compile_module, "_validate_device", lambda device: None)
    monkeypatch.setattr(compile_module, "_compile_layer", fake_compile_layer)
    run = functools.partial(
        compile_glm5_iq2r, source, output, diagnostic_uniform_importance=True
    )
    assert run(layer_indices=[3]) is None
    assert not (output / "config.json").exists()
    # Layer 3 is kept, layer 4 is encoded, then the checkpoint files follow.
    assert run() == output / "config.json"
    assert calls == [3, 4]
    assert run() == output / "config.json"
    assert calls == [3, 4]

    # A missing projection file is rebuilt, the valid other one is kept.
    (output / "iq2r-layer-0004-down.safetensors").unlink()
    assert run(layer_indices=[4]) is None
    assert calls == [3, 4, 4]

    # Files built another way are never overwritten.
    with pytest.raises(ValueError, match="iq2r_codebook_iterations.*delete it"):
        run(iterations=5)
    gate = output / "iq2r-layer-0003-gate-up.safetensors"
    gate.write_bytes(b"not a layer file")
    with pytest.raises(ValueError, match="0003-gate-up.*delete it"):
        run(layer_indices=[3])
    assert gate.read_bytes() == b"not a layer file"
    assert calls == [3, 4, 4]


def test_cli_options(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_compile(model_dir, output_dir, **kwargs):
        calls.append((model_dir, output_dir, kwargs))

    monkeypatch.setattr(compile_module, "compile_glm5_iq2r", fake_compile)
    paths = [
        "--model-dir",
        "fp8",
        "--output-dir",
        "out",
        "--calibration-cache",
        "calibration.pt",
    ]
    assert main(paths) == 0
    assert main([*paths, "--layers", "3-5,40"]) == 0
    assert main([*paths, "--codebook-entries", "1024"]) == 0
    assert calls == [
        (
            Path("fp8"),
            Path("out"),
            {
                "calibration_cache": Path("calibration.pt"),
                "layer_indices": None,
                "codebook_entries": 512,
            },
        ),
        (
            Path("fp8"),
            Path("out"),
            {
                "calibration_cache": Path("calibration.pt"),
                "layer_indices": [3, 4, 5, 40],
                "codebook_entries": 512,
            },
        ),
        (
            Path("fp8"),
            Path("out"),
            {
                "calibration_cache": Path("calibration.pt"),
                "layer_indices": None,
                "codebook_entries": 1024,
            },
        ),
    ]
    capsys.readouterr()
    # The calibration cache is required.
    with pytest.raises(SystemExit):
        main(paths[:4])
    with pytest.raises(SystemExit):
        main([*paths, "--layers", "5-3"])
    with pytest.raises(SystemExit):
        main([*paths, "--codebook-entries", "768"])
    assert len(calls) == 3
    # LDLQ needs the layer Hessians, and the Hessians are only for LDLQ.
    assert main([*paths, "--ldlq", "--hessian-dir", "h"]) == 0
    assert calls[-1][2]["ldlq_hessian_dir"] == Path("h")
    capsys.readouterr()
    for extra in (["--ldlq"], ["--hessian-dir", "h"]):
        with pytest.raises(SystemExit):
            main([*paths, *extra])
    assert len(calls) == 4


@pytest.mark.skipif(not _has_gfx950(), reason="requires gfx950")
@pytest.mark.parametrize("entries", [512, 1024])
def test_compile_packs_layers_and_writes_checkpoint(
    tmp_path, monkeypatch, two_experts, entries
):
    source, output = tmp_path / "source", tmp_path / "out"
    _write_source_model(source)
    generator = torch.Generator(device="cuda").manual_seed(0x5310)
    # Random weights stand in for the placeholder FP8 tensors.
    weights = {
        (layer, expert, projection): torch.randn(
            shape, generator=generator, device="cuda"
        )
        * 0.02
        for layer in (3, 4)
        for expert in range(_EXPERTS)
        for projection, shape in (
            ("gate_up", (2 * _INTERMEDIATE, _HIDDEN)),
            ("down", (_HIDDEN, _INTERMEDIATE)),
        )
    }
    monkeypatch.setattr(
        compile_module,
        "_source_projection",
        lambda reader, layout, layer, expert, projection, device: weights[
            layer, expert, projection
        ],
    )
    encoded = {}
    encode = compile_module._encode_projection

    def spy(reader, layout, importance, layer, projection, **kwargs):
        encoded[layer, projection] = encode(
            reader, layout, importance, layer, projection, **kwargs
        )
        return encoded[layer, projection]

    monkeypatch.setattr(compile_module, "_encode_projection", spy)
    run = functools.partial(
        compile_glm5_iq2r,
        source,
        output,
        diagnostic_uniform_importance=True,
        iterations=1,
        sample_vectors=4096,
        codebook_entries=entries,
    )
    encoding = {
        **_ENCODING,
        "iq2r_codebook_iterations": "1",
        "iq2r_codebook_sample_vectors": "4096",
    }
    assert run(layer_indices=[3]) is None
    assert set(encoded) == {(3, "gate_up"), (3, "down")}
    config_path = run()
    # The second run keeps layer 3 and encodes only layer 4.
    assert set(encoded) == {(3, "gate_up"), (3, "down"), (4, "gate_up"), (4, "down")}
    assert config_path == output / "config.json"

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
                assert handle.metadata() == {
                    **_file_metadata(layer, projection, entries),
                    **encoding,
                }
                assert torch.equal(handle.get_tensor(keys["data"]), values[0].cpu())
                assert torch.equal(
                    handle.get_tensor(keys["auxiliary"]), values[1].cpu()
                )
                assert handle.get_tensor(keys["tile_n"]).tolist() == [128]
    config = json.loads(config_path.read_text())["quantization_config"]
    assert config["iq2r_layout"] == (
        "glm53-packed-v2" if entries == 512 else "glm53-packed-v3"
    )
    # A 1,024-entry layer decodes with 10-bit indices.
    if entries == 1024:
        _, aux = encoded[3, "down"]
        assert aux.shape[1] == 1024 * 8 + _HIDDEN * (_INTERMEDIATE // 256) + 8
    # test_checkpoint_keeps_shared_expert_fp8 checks the other files.


def _write_hessians(path: Path, layers, seed: int) -> None:
    """Random correlated layer Hessians in the calibrate --hessian-dir format."""
    import aiter.quantization.iq2r.glm5.calibrate as calibrate_module

    generator = torch.Generator().manual_seed(seed)
    buckets = {}
    for name in ("all", "decision"):
        bucket = calibrate_module.LayerHessianAccumulator(layers)
        for layer in layers:
            basis = torch.randn((_HIDDEN, 32), generator=generator)
            x = torch.randn((256, 32), generator=generator) @ basis.T
            bucket.observe(layer, x + 0.3 * torch.randn(x.shape, generator=generator))
        buckets[name] = bucket
    calibrate_module.save_hessians(path, buckets, {})


@pytest.mark.skipif(not _has_gfx950(), reason="requires gfx950")
def test_compile_ldlq_reencodes_gate_up_only(tmp_path, monkeypatch, two_experts):
    source, plain, ldlq = tmp_path / "source", tmp_path / "plain", tmp_path / "ldlq"
    _write_source_model(source)
    hessians = tmp_path / "hessians"
    _write_hessians(hessians, range(3, _LAYERS), 0x53D0)
    generator = torch.Generator(device="cuda").manual_seed(0x5311)
    weights = {
        (layer, expert, projection): torch.randn(
            shape, generator=generator, device="cuda"
        )
        * 0.02
        for layer in (3, 4)
        for expert in range(_EXPERTS)
        for projection, shape in (
            ("gate_up", (2 * _INTERMEDIATE, _HIDDEN)),
            ("down", (_HIDDEN, _INTERMEDIATE)),
        )
    }
    monkeypatch.setattr(
        compile_module,
        "_source_projection",
        lambda reader, layout, layer, expert, projection, device: weights[
            layer, expert, projection
        ],
    )
    run = functools.partial(
        compile_glm5_iq2r,
        source,
        diagnostic_uniform_importance=True,
        iterations=1,
        sample_vectors=4096,
        codebook_entries=1024,
        layer_indices=[3],
    )
    run(plain)
    run(ldlq, ldlq_hessian_dir=hessians)
    layout = compile_module.glm5_source_layout(
        json.loads((source / "config.json").read_text())
    )
    factor = compile_module.load_glm5_hessians(hessians, layout).factor(3, "cuda")
    for suffix, projection in (("gate-up", "gate_up"), ("down", "down")):
        keys = iq2r_compiled_tensor_keys(3, projection)
        name = f"iq2r-layer-0003-{suffix}.safetensors"
        with safe_open(plain / name, "pt") as a, safe_open(ldlq / name, "pt") as b:
            meta = b.metadata()
            recipe = json.loads(meta["iq2r_ldlq"])
            assert recipe["projections"] == ["gate_up"]
            assert recipe["beta_decision"] == 1.0 and recipe["damp"] == 1.0
            assert "iq2r_ldlq" not in a.metadata()
            same = torch.equal(a.get_tensor(keys["data"]), b.get_tensor(keys["data"]))
            # Down is not re-encoded; gate/up records change.
            assert same == (projection == "down")
    # Rebuilding into the LDLQ directory with other Hessians is refused.
    _write_hessians(tmp_path / "other", range(3, _LAYERS), 0x53D1)
    with pytest.raises(ValueError, match="does not match"):
        run(ldlq, ldlq_hessian_dir=tmp_path / "other")
    assert factor.u.shape == (_HIDDEN, _HIDDEN)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
