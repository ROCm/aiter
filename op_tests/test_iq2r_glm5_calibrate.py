# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from types import SimpleNamespace

import pytest
import torch

from aiter.iq2r_glm5_calibrate import (
    DiagonalSecondMomentAccumulator,
    TargetSpec,
    capture_glm5_calibration,
    glm5_target_specs,
    load_texts,
    main,
    routed_target,
    shared_target,
    token_batches,
)
from aiter.iq2r_glm5_compile import GLM5Layout, load_glm5_importance


class _CharTokenizer:
    eos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens
        return [2 + ord(char) % 60 for char in text]


def _tiny_config():
    from transformers import GlmMoeDsaConfig

    values = {
        "vocab_size": 64,
        "hidden_size": 32,
        "intermediate_size": 64,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 3,
        "first_k_dense_replace": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "n_shared_experts": 1,
        "n_routed_experts": 8,
        "n_group": 2,
        "topk_group": 1,
        "num_experts_per_tok": 2,
        "kv_lora_rank": 16,
        "q_lora_rank": 16,
        "qk_rope_head_dim": 8,
        "v_head_dim": 8,
        "qk_nope_head_dim": 8,
        "index_topk": 8,
        "index_head_dim": 16,
        "index_n_heads": 2,
        "max_position_embeddings": 128,
        "eos_token_id": 1,
    }
    return GlmMoeDsaConfig(**values)


def _tiny_model(dormant_expert=None):
    pytest.importorskip("transformers.models.glm_moe_dsa")
    from transformers import GlmMoeDsaForCausalLM

    torch.manual_seed(0)
    config = _tiny_config()
    config._experts_implementation = "eager"
    model = GlmMoeDsaForCausalLM(config).eval()
    if dormant_expert is not None:
        for layer in model.model.layers[config.first_k_dense_replace :]:
            layer.mlp.gate.e_score_correction_bias[dormant_expert] = -100.0
    return model


_TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "pack my box with five dozen liquor jugs",
    "sphinx of black quartz, judge my vow",
]


def _record_moe_inputs(model):
    """Return each MoE layer's routed-expert call and shared-expert inputs from one run."""

    records = {}
    handles = []
    for index, layer in enumerate(model.model.layers):
        if not hasattr(layer.mlp, "experts"):
            continue
        record = records.setdefault(index, {"routed": [], "gate_up": [], "down": []})

        def routed(module, args, record=record):
            record["routed"].append(tuple(arg.detach().clone() for arg in args))

        def shared(projection, record=record):
            return lambda module, args: record[projection].append(args[0].detach())

        handles.append(layer.mlp.experts.register_forward_pre_hook(routed))
        shared_experts = layer.mlp.shared_experts
        handles.append(
            shared_experts.gate_proj.register_forward_pre_hook(shared("gate_up"))
        )
        handles.append(
            shared_experts.down_proj.register_forward_pre_hook(shared("down"))
        )
    return records, handles


def _weighted_moment(values, weights):
    weights2 = weights.double().square()
    return (values.double().square() * weights2[:, None]).sum(0) / weights2.sum()


def test_accumulator_weights_rows_by_squared_route():
    spec = TargetSpec("target", groups=2, width=3)
    accumulator = DiagonalSecondMomentAccumulator([spec])
    values = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    route = torch.tensor([0.5, 2.0])
    accumulator.observe("target", 1, values, route)
    accumulator.observe("target", 1, values[:1], forced=True)

    target = accumulator.targets()["target"]
    weights2 = torch.tensor([0.25, 4.0, 1.0], dtype=torch.float64)
    rows = torch.cat([values, values[:1]]).double()
    expected = (rows.square() * weights2[:, None]).sum(0) / weights2.sum()
    torch.testing.assert_close(target["importance"][1], expected.float())
    assert target["importance"].dtype == torch.float32
    assert target["denominator"].tolist() == [0.0, 5.25]
    assert target["hit_count"].tolist() == [0, 3]
    assert target["forced_hit_count"].tolist() == [0, 1]
    assert torch.equal(target["importance"][0], torch.zeros(3))
    assert accumulator.missing_count() == 1

    with pytest.raises(ValueError, match="expected width 3"):
        accumulator.observe("target", 0, torch.ones(2, 4))
    with pytest.raises(IndexError):
        accumulator.observe("target", 2, values)


def test_token_batches_append_eos_and_cycle_texts():
    tokenizer = _CharTokenizer()
    texts = ["ab", "c"]
    batches = list(token_batches(tokenizer, texts, sequence_length=4, max_tokens=10))
    a, b, c = (tokenizer.encode(char)[0] for char in "abc")
    stream = [a, b, 1, c, 1, a, b, 1, c, 1]
    assert [batch.tolist() for batch in batches] == [
        [stream[:4]],
        [stream[4:8]],
        [stream[8:]],
    ]
    assert all(batch.dtype == torch.long for batch in batches)


def test_load_texts_reads_jsonl_fields_and_plain_lines(tmp_path):
    jsonl = tmp_path / "corpus.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps(row)
            for row in ({"text": "one"}, {"prompt": "two"}, {"content": "three"})
        )
        + "\n\n",
        encoding="utf-8",
    )
    assert load_texts(jsonl) == ["one", "two", "three"]

    plain = tmp_path / "corpus.txt"
    plain.write_text("  first line \n\nsecond\n", encoding="utf-8")
    assert load_texts(plain) == ["first line", "second"]

    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"body": "x"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="no text/prompt/content"):
        load_texts(bad)


def test_target_specs_cover_routed_and_shared_experts_per_layer():
    config = SimpleNamespace(
        first_k_dense_replace=3,
        num_hidden_layers=5,
        num_local_experts=4,
        n_shared_experts=1,
        hidden_size=6,
        moe_intermediate_size=2,
    )
    specs = glm5_target_specs(config)
    assert [spec.name for spec in specs[:4]] == [
        routed_target(3, "gate_up"),
        routed_target(3, "down"),
        shared_target(3, "gate_up"),
        shared_target(3, "down"),
    ]
    assert [(spec.groups, spec.width) for spec in specs[:4]] == [
        (4, 6),
        (4, 2),
        (1, 6),
        (1, 2),
    ]
    assert len(specs) == 8
    config.n_shared_experts = 2
    with pytest.raises(ValueError, match="one shared expert"):
        glm5_target_specs(config)


def test_capture_matches_routed_and_shared_references():
    model = _tiny_model()
    tokens = 48
    reference_input = next(token_batches(_CharTokenizer(), _TEXTS, tokens, tokens))
    records, handles = _record_moe_inputs(model)
    with torch.inference_mode():
        reference_logits = model(input_ids=reference_input, use_cache=False).logits
    for handle in handles:
        handle.remove()

    payload = capture_glm5_calibration(
        model,
        _CharTokenizer(),
        _TEXTS,
        sequence_length=tokens,
        max_tokens=tokens,
        device="cpu",
    )
    # The observers are removed again and left the model's forward unchanged.
    with torch.inference_mode():
        logits = model(input_ids=reference_input, use_cache=False).logits
    torch.testing.assert_close(logits, reference_logits, rtol=0, atol=0)

    targets = payload["targets"]
    config = model.config
    for layer in range(config.first_k_dense_replace, config.num_hidden_layers):
        hidden, top_k_index, top_k_weights = records[layer]["routed"][0]
        experts = model.model.layers[layer].mlp.experts
        gate_up = targets[routed_target(layer, "gate_up")]
        down = targets[routed_target(layer, "down")]
        assert int(gate_up["hit_count"].sum()) == tokens * config.num_experts_per_tok
        assert torch.equal(gate_up["hit_count"], down["hit_count"])
        for expert in range(config.num_local_experts):
            token, slot = torch.where(top_k_index == expert)
            if token.numel() == 0:
                continue
            values = hidden[token]
            route = top_k_weights[token, slot]
            torch.testing.assert_close(
                gate_up["importance"][expert],
                _weighted_moment(values, route).float(),
            )
            activated = experts._apply_gate(values @ experts.gate_up_proj[expert].T)
            torch.testing.assert_close(
                down["importance"][expert],
                _weighted_moment(activated, route).float(),
            )

        ones = torch.ones(tokens)
        for projection in ("gate_up", "down"):
            shared = targets[shared_target(layer, projection)]
            assert shared["hit_count"].tolist() == [tokens]
            assert shared["denominator"].tolist() == [float(tokens)]
            values = records[layer][projection][0].reshape(tokens, -1)
            torch.testing.assert_close(
                shared["importance"][0], _weighted_moment(values, ones).float()
            )

    metadata = payload["metadata"]
    assert metadata["token_count"] == tokens
    assert metadata["forced_layer_input_tokens"] == 0
    assert metadata["forced_layer_experts"] == []


def test_capture_forces_dormant_experts_from_layer_inputs():
    dormant = 7
    tokens = 64
    model = _tiny_model(dormant_expert=dormant)
    with pytest.raises(RuntimeError, match="unobserved"):
        capture_glm5_calibration(
            model,
            _CharTokenizer(),
            _TEXTS,
            sequence_length=32,
            max_tokens=tokens,
            force_unobserved_tokens=0,
            device="cpu",
        )

    payload = capture_glm5_calibration(
        model,
        _CharTokenizer(),
        _TEXTS,
        sequence_length=32,
        max_tokens=tokens,
        force_unobserved_tokens=32,
        device="cpu",
    )
    config = model.config
    moe_layers = range(config.first_k_dense_replace, config.num_hidden_layers)
    metadata = payload["metadata"]
    assert metadata["forced_layer_input_tokens"] == 32
    assert metadata["forced_layer_experts"] == [
        [layer, dormant] for layer in moe_layers
    ]
    for layer in moe_layers:
        gate_up = payload["targets"][routed_target(layer, "gate_up")]
        assert gate_up["hit_count"][dormant] == 32
        assert gate_up["forced_hit_count"][dormant] == 32
        assert int(gate_up["forced_hit_count"].sum()) == 32
        # The shared expert keeps natural-token statistics only.
        shared = payload["targets"][shared_target(layer, "gate_up")]
        assert shared["hit_count"].tolist() == [tokens]


def test_cli_artifact_loads_into_iq2r_compiler(tmp_path):
    model = _tiny_model()
    tokenizers = pytest.importorskip("tokenizers")
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "</s>": 1}
    for word in " ".join(_TEXTS).replace(",", "").split():
        vocab.setdefault(word, len(vocab))
    backend = tokenizers.Tokenizer(
        tokenizers.models.WordLevel(vocab, unk_token="[UNK]")
    )
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    model_dir = tmp_path / "model"
    PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", eos_token="</s>"
    ).save_pretrained(model_dir)
    model.save_pretrained(model_dir)
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        "".join(json.dumps({"text": text}) + "\n" for text in _TEXTS),
        encoding="utf-8",
    )
    output = tmp_path / "out" / "calibration.pt"

    assert (
        main(
            [
                "--model-dir",
                str(model_dir),
                "--text-file",
                str(corpus),
                "--output",
                str(output),
                "--sequence-length",
                "16",
                "--max-tokens",
                "96",
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    # The atomic save leaves no temporary file behind.
    assert [entry.name for entry in output.parent.iterdir()] == ["calibration.pt"]
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert payload["format"] == "iq2r-calibration"
    assert len(payload["targets"]) == 8
    metadata = payload["metadata"]
    assert metadata["token_count"] == 96
    assert metadata["model_type"] == "glm_moe_dsa"
    assert len(metadata["model_config_sha256"]) == 64

    config = model.config
    layout = GLM5Layout(
        layer_count=config.num_hidden_layers,
        first_moe_layer=config.first_k_dense_replace,
        expert_count=config.num_local_experts,
        hidden_size=config.hidden_size,
        intermediate_size=config.moe_intermediate_size,
        block_n=16,
        block_k=16,
    )
    loaded = load_glm5_importance(output, layout)
    assert loaded.quality == "calibrated-o0"
    targets = payload["targets"]
    assert torch.equal(
        loaded.gate_up[1], targets[routed_target(2, "gate_up")]["importance"]
    )
    assert torch.equal(
        loaded.shared_down[0], targets[shared_target(1, "down")]["importance"][0]
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
