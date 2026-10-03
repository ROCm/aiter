# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""GLM-5.3 IQ2R checkpoint packer on a tiny synthetic overlay checkpoint."""

import json

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

import aiter.iq2r_glm53_pack_checkpoint as pack_checkpoint
from aiter.ops.iq2r_format import IQ2RMetadata

_HIDDEN = 6144
_INTERMEDIATE = 2048
_EXPERTS = 2
_LAYER = 3


def _has_gfx950() -> bool:
    return torch.cuda.is_available() and "gfx950" in (
        torch.cuda.get_device_properties(0).gcnArchName
    )


def _write_source(root):
    """Write an overlay-shaped checkpoint: two base shards and one IQ2R layer."""
    generator = torch.Generator().manual_seed(0x5310)
    base = {
        "a.safetensors": {
            "model.embed_tokens.weight": torch.randn(
                64, 32, generator=generator
            ).bfloat16(),
            "model.norm.weight": torch.randn(32, generator=generator).bfloat16(),
        },
        "b.safetensors": {
            "lm_head.weight": torch.randn(64, 32, generator=generator).bfloat16(),
            "model.layers.3.mlp.gate.weight": torch.randn(8, 32, generator=generator),
        },
    }
    gate_meta = IQ2RMetadata(logical_n=2 * _INTERMEDIATE, logical_k=_HIDDEN)
    down_meta = IQ2RMetadata(logical_n=_HIDDEN, logical_k=_INTERMEDIATE)

    def random_bytes(columns):
        return torch.randint(
            0, 256, (_EXPERTS, columns), generator=generator, dtype=torch.uint8
        )

    prefix = f"model.layers.{_LAYER}.mlp.experts"
    iq2r = {
        "gate-up": {
            f"{prefix}.up_gate_proj.0.iq2r_data": random_bytes(gate_meta.data_bytes),
            f"{prefix}.up_gate_proj.0.iq2r_auxiliary": random_bytes(
                gate_meta.auxiliary_bytes
            ),
            f"{prefix}.up_gate_proj.0.iq2r_tN": torch.tensor([128], dtype=torch.int32),
        },
        "down": {
            f"{prefix}.down_proj.0.iq2r_data": random_bytes(down_meta.data_bytes),
            f"{prefix}.down_proj.0.iq2r_auxiliary": random_bytes(
                down_meta.auxiliary_bytes
            ),
            f"{prefix}.down_proj.0.iq2r_tN": torch.tensor([128], dtype=torch.int32),
        },
    }
    weight_map = {}
    for name, tensors in base.items():
        save_file(tensors, root / name)
        weight_map.update({key: name for key in tensors})
    for proj, tensors in iq2r.items():
        name = f"iq2r-layer-{_LAYER:04d}-{proj}.safetensors"
        save_file(tensors, root / name, metadata={"iq2r_quality": "test"})
        weight_map.update({key: name for key in tensors})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": weight_map})
    )
    config = {
        "moe_intermediate_size": _INTERMEDIATE,
        "quantization_config": {"quant_method": "iq2r"},
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "tokenizer.json").write_text("{}")
    return base, iq2r


def test_base_and_meta_stages(tmp_path, monkeypatch):
    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    base, iq2r = _write_source(src)
    # One tensor per shard, so the grouping crosses the source-file boundary.
    monkeypatch.setattr(pack_checkpoint, "SHARD_BYTES", 1)
    pack_checkpoint.main([str(src), str(out), "base"])
    # The IQ2R files come from the iq2r stage; copy them as-is for this CPU test.
    for proj in iq2r:
        name = f"iq2r-layer-{_LAYER:04d}-{proj}.safetensors"
        (out / name).write_bytes((src / name).read_bytes())
    pack_checkpoint.main([str(src), str(out), "meta"])

    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]
    expected = {k: v for tensors in base.values() for k, v in tensors.items()}
    assert len({f for k, f in index.items() if k in expected}) == len(expected)
    for key, value in expected.items():
        with safe_open(str(out / index[key]), "pt") as f:
            assert torch.equal(f.get_tensor(key), value)
    for proj, tensors in iq2r.items():
        for key in tensors:
            assert index[key] == f"iq2r-layer-{_LAYER:04d}-{proj}.safetensors"
    config = json.loads((out / "config.json").read_text())
    assert config["quantization_config"]["iq2r_layout"] == pack_checkpoint.LAYOUT
    assert (out / "tokenizer.json").exists()


@pytest.mark.skipif(not _has_gfx950(), reason="requires gfx950")
def test_iq2r_stage_matches_direct_pack(tmp_path):
    from aiter.iq2r_glm53 import iq2r_glm53_pack

    src, out = tmp_path / "src", tmp_path / "out"
    src.mkdir()
    _, iq2r = _write_source(src)
    pack_checkpoint.main([str(src), str(out), "iq2r"])

    prefix = f"model.layers.{_LAYER}.mlp.experts"
    sources = (
        iq2r["gate-up"][f"{prefix}.up_gate_proj.0.iq2r_data"],
        iq2r["gate-up"][f"{prefix}.up_gate_proj.0.iq2r_auxiliary"],
        iq2r["down"][f"{prefix}.down_proj.0.iq2r_data"],
        iq2r["down"][f"{prefix}.down_proj.0.iq2r_auxiliary"],
    )
    expected = iq2r_glm53_pack(
        *(t.cuda() for t in sources), intermediate_size=_INTERMEDIATE
    )
    names = (
        ("gate-up", f"{prefix}.up_gate_proj.0.iq2r_data"),
        ("gate-up", f"{prefix}.up_gate_proj.0.iq2r_auxiliary"),
        ("down", f"{prefix}.down_proj.0.iq2r_data"),
        ("down", f"{prefix}.down_proj.0.iq2r_auxiliary"),
    )
    for (proj, key), value in zip(names, expected):
        with safe_open(
            str(out / f"iq2r-layer-{_LAYER:04d}-{proj}.safetensors"), "pt"
        ) as f:
            assert f.metadata()["iq2r_layout"] == pack_checkpoint.LAYOUT
            assert f.metadata()["iq2r_quality"] == "test"
            assert torch.equal(f.get_tensor(key), value.cpu())
            tile_key = key.rsplit(".", 1)[0] + ".iq2r_tN"
            assert torch.equal(f.get_tensor(tile_key), iq2r[proj][tile_key])

    # A packed checkpoint is not a valid source.
    for name in ("model.safetensors.index.json", "config.json"):
        (out / name).write_bytes((src / name).read_bytes())
    with pytest.raises(SystemExit, match="already packed"):
        pack_checkpoint.main([str(out), str(tmp_path / "again"), "iq2r"])
