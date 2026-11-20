# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

import aiter.quantization.iq2r.glm5.calibrate as calibrate_module
from aiter.quantization.iq2r.glm5.calibrate import (
    ChatTokens,
    DiagonalSecondMomentAccumulator,
    LayerHessianAccumulator,
    MergedHessians,
    TargetSpec,
    _dequantized_fp8_linear,
    capture_glm5_calibration,
    capture_glm5_hessians,
    decision_kind_weights,
    glm5_target_specs,
    hessian_file,
    load_texts,
    load_texts_by_kind,
    main,
    parse_decision_weights,
    routed_target,
    save_calibration,
    save_hessians,
    scored_positions,
    token_batches,
)
from aiter.quantization.iq2r.glm5.compile import (
    GLM5Layout,
    load_glm5_hessians,
    load_glm5_importance,
)


class _CharTokenizer:
    """Rows already carry the chat template's special tokens."""

    eos_token_id = 1

    def encode(self, text, add_special_tokens=True):
        assert not add_special_tokens
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
    """Return each MoE layer's routed-expert calls from one run."""

    records = {}
    handles = []
    for index, layer in enumerate(model.model.layers):
        if not hasattr(layer.mlp, "experts"):
            continue
        record = records.setdefault(index, [])

        def routed(module, args, record=record):
            record.append(tuple(arg.detach().clone() for arg in args))

        handles.append(layer.mlp.experts.register_forward_pre_hook(routed))
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


def test_token_batches_feed_each_row_truncated_without_special_tokens():
    tokenizer = _CharTokenizer()
    texts = ["abcde", "f", "", "gh"]
    batches = list(token_batches(tokenizer, texts, sequence_length=3, max_tokens=5))
    a, b, c, f, g = (2 + ord(char) % 60 for char in "abcfg")
    # One pass: the token budget cuts the last row, the empty row is skipped.
    assert [batch.tolist() for batch in batches] == [[[a, b, c]], [[f]], [[g]]]
    assert all(batch.dtype == torch.long for batch in batches)
    batches = list(token_batches(tokenizer, texts, sequence_length=8, max_tokens=99))
    assert [batch.shape[1] for batch in batches] == [5, 1, 2]


def test_token_batches_can_keep_the_end_of_long_rows():
    tokenizer = _CharTokenizer()
    texts = ["abcde", "f", "gh"]
    c, d, e, f, g, h = (2 + ord(char) % 60 for char in "cdefgh")
    batches = token_batches(
        tokenizer, texts, sequence_length=3, max_tokens=99, keep="end"
    )
    # A long row keeps its last tokens (its last reply and stop decision).
    assert [batch.tolist() for batch in batches] == [[[c, d, e]], [[f]], [[g, h]]]
    # The token budget also cuts from the front.
    batches = token_batches(
        tokenizer, texts, sequence_length=3, max_tokens=4, keep="end"
    )
    assert [batch.tolist() for batch in batches] == [[[c, d, e]], [[f]]]
    with pytest.raises(ValueError, match="keep"):
        list(
            token_batches(tokenizer, texts, sequence_length=3, max_tokens=9, keep="mid")
        )


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


def test_target_specs_cover_routed_experts_per_layer():
    config = SimpleNamespace(
        first_k_dense_replace=3,
        num_hidden_layers=5,
        num_local_experts=4,
        n_shared_experts=1,
        hidden_size=6,
        moe_intermediate_size=2,
    )
    specs = glm5_target_specs(config)
    assert [spec.name for spec in specs] == [
        routed_target(3, "gate_up"),
        routed_target(3, "down"),
        routed_target(4, "gate_up"),
        routed_target(4, "down"),
    ]
    assert [(spec.groups, spec.width) for spec in specs[:2]] == [(4, 6), (4, 2)]


def test_capture_matches_routed_references():
    model = _tiny_model()
    tokens = 32
    texts = _TEXTS[:1]
    reference_input = next(token_batches(_CharTokenizer(), texts, tokens, tokens))
    assert reference_input.shape == (1, tokens)
    records, handles = _record_moe_inputs(model)
    with torch.inference_mode():
        reference_logits = model(input_ids=reference_input, use_cache=False).logits
    for handle in handles:
        handle.remove()

    payload = capture_glm5_calibration(
        model,
        _CharTokenizer(),
        texts,
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
    moe_layers = range(config.first_k_dense_replace, config.num_hidden_layers)
    # Routed experts only: the shared expert stays FP8 in IQ2R-v2.
    assert set(targets) == {
        routed_target(layer, projection)
        for layer in moe_layers
        for projection in ("gate_up", "down")
    }
    for layer in moe_layers:
        hidden, top_k_index, top_k_weights = records[layer][0]
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

    metadata = payload["metadata"]
    assert metadata["token_count"] == tokens
    assert metadata["sequence_length"] == tokens
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


def test_artifact_loads_into_iq2r_compiler(tmp_path):
    model = _tiny_model()
    payload = capture_glm5_calibration(
        model,
        _CharTokenizer(),
        _TEXTS,
        sequence_length=16,
        max_tokens=96,
        device="cpu",
    )
    output = tmp_path / "out" / "calibration.pt"
    save_calibration(payload, output)
    # The atomic save leaves no temporary file behind.
    assert [entry.name for entry in output.parent.iterdir()] == ["calibration.pt"]
    payload = torch.load(output, map_location="cpu", weights_only=True)
    assert payload["format"] == "iq2r-calibration"
    assert len(payload["targets"]) == 4
    assert payload["metadata"]["token_count"] == 48

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
        loaded.routed_hits[0], targets[routed_target(1, "gate_up")]["hit_count"]
    )


def test_cli_takes_only_the_corpus_paths(monkeypatch, capsys):
    calls = []

    def fake_calibrate(model_dir, text_file, output, **kwargs):
        calls.append((model_dir, text_file, output, kwargs))
        return {"format": "iq2r-calibration", "targets": {}, "metadata": {}}

    monkeypatch.setattr(calibrate_module, "calibrate_glm5", fake_calibrate)
    paths = ["--model-dir", "m", "--text-file", "c.jsonl", "--output", "o.pt"]
    assert main(paths) == 0
    assert main([*paths, "--fp8-kernel-dir", "k"]) == 0
    assert calls == [
        ("m", "c.jsonl", "o.pt", {"fp8_kernel_dir": None}),
        ("m", "c.jsonl", "o.pt", {"fp8_kernel_dir": "k"}),
    ]
    capsys.readouterr()
    with pytest.raises(SystemExit):
        main(paths[:4])
    assert len(calls) == 2
    # The recipe is fixed.
    assert calibrate_module.SEQUENCE_LENGTH == 16384
    assert calibrate_module.MAX_TOKENS == 1_000_000
    assert calibrate_module.FORCE_UNOBSERVED_TOKENS == 512


class _ChatTokenizer(_CharTokenizer):
    """Characters stand for the chat template's special tokens."""

    special: ClassVar[dict[str, tuple[str, int]]] = {
        "<|assistant|>": ("@", 50),
        "<|user|>": ("#", 51),
        "<|observation|>": ("&", 52),
        "<|system|>": ("~", 53),
        "<|endoftext|>": ("$", 1),
        "</think>": ("%", 54),
        "<tool_call>": ("^", 55),
    }
    unk_token_id = 0

    def __init__(self, overrides=None):
        self.ids = {name: tid for name, (_, tid) in self.special.items()}
        self.ids.update(overrides or {})
        self.chars = {c: tid for c, tid in self.special.values()}

    def encode(self, text, add_special_tokens=True):
        assert not add_special_tokens
        return [self.chars.get(c, 2 + ord(c) % 40) for c in text]

    def convert_tokens_to_ids(self, names):
        return [self.ids.get(name, 0) for name in names]


_CHATS = [
    "~sys#user asks@assistant says$#more?@sure%done^call&obs@ok$",
    "#question@answer text here$#again@more text$",
]


def test_scored_positions_predict_assistant_text_and_its_end_of_turn():
    tokens = ChatTokens.from_tokenizer(_ChatTokenizer())
    ids = _ChatTokenizer().encode("#u@ab$#v@c&o@d$", add_special_tokens=False)
    # Scored next tokens: <|assistant|> and the text after it up to the next
    # role (both $), plus the end-of-turn token right after that text (the #
    # after $, the & after c).
    expected = [False] * len(ids)
    for target in (2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14):
        expected[target - 1] = True
    assert scored_positions(ids, tokens).tolist() == expected


def test_chat_tokens_need_every_special_token():
    with pytest.raises(ValueError, match="no single token"):
        ChatTokens.from_tokenizer(_ChatTokenizer({"</think>": 0}))


def _layer_inputs(model, input_ids):
    """Each MoE layer's input [T, H] and the logits of one run."""
    records, handles = {}, []
    for index, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "experts"):

            def hook(module, args, index=index):
                records[index] = args[0].reshape(-1, args[0].shape[-1]).clone()

            handles.append(layer.mlp.register_forward_pre_hook(hook))
    with torch.inference_mode():
        logits = model(input_ids=input_ids, use_cache=False).logits[0]
    for handle in handles:
        handle.remove()
    return records, logits


def test_hessian_capture_matches_layer_inputs():
    model = _tiny_model()
    tokenizer = _ChatTokenizer()
    tokens = ChatTokens.from_tokenizer(tokenizer)
    runs = [
        (torch.tensor([tokenizer.encode(text, add_special_tokens=False)]),)
        + _layer_inputs(
            model, torch.tensor([tokenizer.encode(text, add_special_tokens=False)])
        )
        for text in _CHATS
    ]
    # Make the decision bucket non-empty: </think> is the most frequent top-1.
    top1 = torch.cat(
        [
            logits[scored_positions(ids[0].tolist(), tokens)].argmax(-1)
            for ids, _, logits in runs
        ]
    )
    frequent = int(top1.mode().values)
    tokenizer = _ChatTokenizer({"</think>": frequent})
    tokens = ChatTokens.from_tokenizer(tokenizer)

    accumulators = {
        decision: capture_glm5_hessians(
            model, tokenizer, _CHATS, decision=decision, device="cpu"
        )
        for decision in (False, True)
    }
    for (bucket, fed), decision in zip(accumulators.values(), (False, True)):
        assert fed == sum(len(text) for text in _CHATS)
        means = bucket.means()
        for layer in range(1, 3):
            rows = []
            for ids, inputs, logits in runs:
                scored = scored_positions(ids[0].tolist(), tokens)
                keep = scored
                if decision:
                    chosen = torch.isin(
                        logits[scored].argmax(-1), torch.tensor(sorted(tokens.decision))
                    )
                    keep = scored.nonzero().flatten()[chosen]
                rows.append(inputs[layer][keep].float())
            x = torch.cat(rows)
            assert bucket.counts[layer] == x.shape[0] > 0
            torch.testing.assert_close(means[layer], x.T @ x / x.shape[0])


def test_hessians_save_and_load_into_the_compiler(tmp_path):
    model = _tiny_model()
    tokenizer = _ChatTokenizer()
    buckets = {
        "all": capture_glm5_hessians(
            model, tokenizer, _CHATS, decision=False, device="cpu"
        )[0],
        "decision": capture_glm5_hessians(
            model, tokenizer, _CHATS[:1], decision=False, device="cpu"
        )[0],
    }
    manifest = save_hessians(tmp_path, buckets, {"all_corpus_sha256": "x"})
    assert manifest["buckets"] == ["all", "decision"]
    assert sorted(manifest["files"]) == [hessian_file(1), hessian_file(2)]
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
    hessians = load_glm5_hessians(tmp_path, layout)
    factor = hessians.factor(2, "cpu")
    all_ = buckets["all"].means()[2].double()
    decision = buckets["decision"].means()[2].double()
    h = all_ / all_.trace() + decision / decision.trace()
    h = h + h.diagonal().mean() * torch.eye(h.shape[0], dtype=torch.float64)
    u = factor.u.double()
    torch.testing.assert_close(u.T @ u, torch.linalg.inv(h), rtol=1e-4, atol=1e-4)
    # A changed file no longer matches its digest.
    payload = torch.load(tmp_path / hessian_file(2), weights_only=True)
    payload["all"] = payload["all"] * 2
    torch.save(payload, tmp_path / hessian_file(2))
    with pytest.raises(ValueError, match="digest"):
        hessians.factor(2, "cpu")


def test_cli_hessian_options(monkeypatch, capsys):
    calls = []

    def fake_calibrate(model_dir, text_file, output, **kwargs):
        calls.append((output, kwargs))
        return {"hessians": {"files": {}, "buckets": ["all"]}}

    monkeypatch.setattr(calibrate_module, "calibrate_glm5", fake_calibrate)
    base = ["--model-dir", "m", "--text-file", "c.jsonl"]
    assert main([*base, "--hessian-dir", "h", "--decision-text-file", "d.jsonl"]) == 0
    assert calls == [
        (
            None,
            {
                "fp8_kernel_dir": None,
                "hessian_dir": "h",
                "decision_text_file": "d.jsonl",
                "decision_weights": None,
                "decision_keep": "end",
            },
        )
    ]
    capsys.readouterr()
    weights = "chat=0.21, code=0.13"
    args = [*base, "--hessian-dir", "h", "--decision-text-file", "d.jsonl"]
    assert main([*args, "--decision-weights", weights, "--decision-keep", "start"]) == 0
    assert calls[-1][1]["decision_weights"] == {"chat": 0.21, "code": 0.13}
    assert calls[-1][1]["decision_keep"] == "start"
    calls.pop()
    capsys.readouterr()
    for bad, message in (
        ([*base, "--hessian-dir", "h", "--decision-weights", weights], "needs"),
        ([*args, "--decision-weights", "chat"], "kind=weight"),
    ):
        with pytest.raises(SystemExit):
            main(bad)
        assert message in capsys.readouterr().err
    # The decision corpus only feeds the Hessians: without --hessian-dir it is
    # an error, not silently ignored.
    with pytest.raises(SystemExit):
        main([*base, "--output", "o.pt", "--decision-text-file", "d.jsonl"])
    assert "--decision-text-file needs --hessian-dir" in capsys.readouterr().err
    assert len(calls) == 1


def test_dequantized_fp8_linear_uses_128_blocks_with_partial_edges():
    """576 x 200 has partial last blocks; each scale covers 128 x 128 weights."""
    generator = torch.Generator().manual_seed(0)
    weight = torch.randn((576, 200), generator=generator).to(torch.float8_e4m3fn)
    scale = torch.rand((5, 2), generator=generator) + 0.5
    x = torch.randn((3, 200), generator=generator).to(torch.bfloat16)
    grid = scale.repeat_interleave(128, 0)[:576].repeat_interleave(128, 1)[:, :200]
    expected = torch.nn.functional.linear(x, (weight.float() * grid).to(torch.bfloat16))
    assert torch.equal(_dequantized_fp8_linear(x, weight, scale, (128, 128)), expected)
    assert torch.equal(_dequantized_fp8_linear(x, weight, scale), expected)


def test_decision_corpus_groups_by_kind(tmp_path):
    path = tmp_path / "decisions.jsonl"
    rows = [("swe", "a"), ("math", "b"), ("swe", "c")]
    path.write_text("".join(json.dumps({"kind": k, "text": t}) + "\n" for k, t in rows))
    assert load_texts_by_kind(path) == {"swe": ["a", "c"], "math": ["b"]}
    plain = tmp_path / "plain.txt"
    plain.write_text("x\ny\n")
    assert load_texts_by_kind(plain) == {"": ["x", "y"]}


def test_decision_kind_weights_given_or_token_share():
    kinds, tokens = ["chat", "code", "math"], [600, 300, 100]
    assert decision_kind_weights(kinds, tokens, None) == (
        [0.6, 0.3, 0.1],
        "token share",
    )
    weights, source = decision_kind_weights(
        kinds, tokens, {"math": 1.0, "chat": 2.0, "code": 1.0}
    )
    assert weights == [0.5, 0.25, 0.25] and source == "given"
    with pytest.raises(ValueError, match="name kinds"):
        decision_kind_weights(kinds, tokens, {"chat": 1.0, "code": 1.0})
    with pytest.raises(ValueError, match="positive"):
        decision_kind_weights(kinds, tokens, {"chat": 1.0, "code": 0.0, "math": 1.0})
    assert parse_decision_weights("a=1, b=0.5") == {"a": 1.0, "b": 0.5}
    for bad in ("a", "a=1,a=2", "=1"):
        with pytest.raises(ValueError):
            parse_decision_weights(bad)


def test_merged_hessians_weight_trace_normalized_kinds():
    generator = torch.Generator().manual_seed(1)
    parts = []
    for rows, scale in ((50, 1.0), (7, 30.0)):
        part = LayerHessianAccumulator(range(1, 3))
        for layer in part.layers:
            part.observe(layer, scale * torch.randn((rows, 16), generator=generator))
        parts.append(part)
    for weights, (wa, wb) in ((None, (0.5, 0.5)), ([300.0, 100.0], (0.75, 0.25))):
        merged = MergedHessians(parts, weights)
        assert merged.counts == {1: 57, 2: 57}
        for layer, h in merged.means().items():
            a, b = (p.means()[layer].double() for p in parts)
            expected = wa * a / a.trace() + wb * b / b.trace()
            torch.testing.assert_close(h.double(), expected, rtol=1e-6, atol=1e-7)
            assert abs(float(h.double().trace()) - 1.0) < 1e-5


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
