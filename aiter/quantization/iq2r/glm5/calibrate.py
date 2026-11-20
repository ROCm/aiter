# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Capture the GLM-5.3 importance statistics that IQ2R-v2 compilation weights with.

The block-FP8 checkpoint runs prefill-only in Hugging Face Transformers over a
text corpus, spread over every visible GPU (Accelerate device map
"balanced"). Each corpus row (one text, or one whole conversation rendered
with the chat template) is its own sequence of up to 16,384 tokens, fed as
rendered (no special tokens added), over at most 1,000,000 tokens. Input
channel k of every routed expert projection in layers
``first_k_dense_replace`` onwards accumulates a router-weighted diagonal
second moment

    importance[k] = sum_t r_t^2 * x_t[k]^2 / sum_t r_t^2

over the tokens t routed to the expert, with routing weight r_t. x_t is the
projection input: the MoE input for gate/up and act(gate) * up for down.
Experts that no corpus token reaches get one forced pass over up to 512
tokens of their layer's input. The output is an ``iq2r-calibration`` artifact
for ``aiter.quantization.iq2r.glm5.compile --calibration-cache``:

    python3 -m aiter.quantization.iq2r.glm5.calibrate --model-dir $FP8 \\
        --text-file corpus.jsonl --output calibration.pt

``--hessian-dir`` also captures, for LDLQ (``compile --ldlq``), each MoE
layer's input second moment H = sum x x^T / n (x: the MoE input, the gate/up
input of every expert), in two buckets:

- ``all``: the scored positions of ``--text-file``, i.e. the positions that
  predict an assistant token or the end-of-turn token right after assistant
  text (the "stop / hand over" decision);
- ``decision``: the scored positions of ``--decision-text-file`` (a separate
  corpus, kept apart from the calibration corpus and from evaluation text)
  where the FP8 model's top-1 next token is end of turn (``<|endoftext|>``,
  ``<|user|>``, ``<|observation|>``), ``</think>`` or ``<tool_call>``. JSONL
  rows with a ``kind`` field (for example SWE agent, math reasoning, terminal
  agent, chat) give one Hessian per kind; the bucket is their mean after each
  is normalized by its trace, weighted by the kind's share of the tokens fed
  (after the 16,384-token cut), so a kind weighs by its text, not by its
  position count.

They are written as ``hessian-layer-NNNN.pt`` plus ``hessians.json`` (corpus
hashes, counts and each file's digest); ``--output`` is then optional:

    python3 -m aiter.quantization.iq2r.glm5.calibrate --model-dir $FP8 \\
        --text-file corpus.jsonl --decision-text-file decisions.jsonl \\
        --hessian-dir hessians/

Needs Transformers with GLM MoE DSA support; tested with 5.16.1. The BF16
fallback below, ``--fp8-kernel-dir`` and a 5.16 workaround patch private
Transformers internals, so other versions may need changes. The FP8 matmul
comes from the ``kernels`` package (``kernels-community/finegrained-fp8``);
without it, the block-FP8 weights are dequantized to BF16 for each matmul
instead (BF16 activations times the exact weights, slower).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
import types
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from aiter.ops.iq2r_format import IQ2R_ACTIVATION_BASIS

from .compile import (
    _CALIBRATION_FORMAT,
    _CALIBRATION_SCHEME,
    _CALIBRATION_VERSION,
)

_ROUTED_EXPERTS = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")
# The recipe.
SEQUENCE_LENGTH = 16384
MAX_TOKENS = 1_000_000
FORCE_UNOBSERVED_TOKENS = 512
# Which part of a text longer than SEQUENCE_LENGTH is fed.
KEEP_CHOICES = ("start", "end")
_DEVICE_MAP = "balanced"
_INPUT_DEVICE = "cuda:0"


def routed_target(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{projection}_proj.weight"


@dataclass(frozen=True)
class TargetSpec:
    name: str
    groups: int
    width: int


def glm5_target_specs(config) -> list[TargetSpec]:
    """Return the routed-expert targets (one group per expert) per MoE layer."""

    widths = {"gate_up": config.hidden_size, "down": config.moe_intermediate_size}
    specs = []
    for layer in range(config.first_k_dense_replace, config.num_hidden_layers):
        for projection, width in widths.items():
            specs.append(
                TargetSpec(
                    routed_target(layer, projection), config.num_local_experts, width
                )
            )
    return specs


class DiagonalSecondMomentAccumulator:
    """Accumulate sum r^2 x^2 and sum r^2 per target group in float64 on the host."""

    def __init__(self, specs: list[TargetSpec]):
        if not specs:
            raise ValueError("at least one calibration target is required")
        self.specs = {spec.name: spec for spec in specs}
        if len(self.specs) != len(specs):
            raise ValueError("calibration target names must be unique")
        self.sums = {
            spec.name: torch.zeros(spec.groups, spec.width, dtype=torch.float64)
            for spec in specs
        }
        self.denominators = {
            spec.name: torch.zeros(spec.groups, dtype=torch.float64) for spec in specs
        }
        self.hit_counts = {
            spec.name: torch.zeros(spec.groups, dtype=torch.int64) for spec in specs
        }
        self.forced_hit_counts = {
            spec.name: torch.zeros(spec.groups, dtype=torch.int64) for spec in specs
        }

    def observe(
        self,
        target: str,
        group: int,
        values: Tensor,
        weights: Tensor | None = None,
        *,
        forced: bool = False,
    ) -> None:
        spec = self.specs[target]
        if not 0 <= group < spec.groups:
            raise IndexError(f"calibration target {target!r} has no group {group}")
        values = values.reshape(-1, values.shape[-1])
        if values.shape[-1] != spec.width:
            raise ValueError(
                f"calibration target {target!r} expected width {spec.width}, "
                f"got {values.shape[-1]}"
            )
        rows = values.shape[0]
        if rows == 0:
            return
        if weights is None:
            weights2 = torch.ones(rows, dtype=torch.float32, device=values.device)
        else:
            weights2 = weights.reshape(-1).float().square()
            if weights2.shape[0] != rows:
                raise ValueError(
                    f"calibration target {target!r} has {rows} rows but "
                    f"{weights2.shape[0]} weights"
                )
        weighted = values.float().square() * weights2.unsqueeze(1)
        self.sums[target][group].add_(weighted.sum(dim=0).double().cpu())
        self.denominators[target][group] += weights2.sum().double().cpu()
        self.hit_counts[target][group] += rows
        if forced:
            self.forced_hit_counts[target][group] += rows

    def missing(self) -> dict[str, Tensor]:
        return {name: counts == 0 for name, counts in self.hit_counts.items()}

    def missing_count(self) -> int:
        return sum(int(mask.sum()) for mask in self.missing().values())

    def targets(self) -> dict[str, dict[str, Any]]:
        """Return the artifact targets: importance [groups, K] plus counts."""

        targets = {}
        for name in self.specs:
            denominator = self.denominators[name]
            importance = self.sums[name] / denominator.clamp_min(1e-30).unsqueeze(1)
            targets[name] = {
                "importance": importance.float().contiguous(),
                "denominator": denominator.clone(),
                "hit_count": self.hit_counts[name].clone(),
                "forced_hit_count": self.forced_hit_counts[name].clone(),
            }
        return targets


class GLM5CalibrationObserver:
    """Record the expert projection inputs of a Transformers GLM MoE DSA model.

    Each routed experts module runs the per-expert loop of Transformers' eager
    implementation and records the gate/up and down inputs on the way.
    ``force`` limits recording to experts with no hits so far and, for those
    still without hits, records every token of the layer's input with weight 1.
    """

    def __init__(self, model, accumulator: DiagonalSecondMomentAccumulator):
        config = model.config
        self.model = model
        self.accumulator = accumulator
        self.layers = range(config.first_k_dense_replace, config.num_hidden_layers)
        self.capture_mask = torch.zeros(
            config.num_hidden_layers, config.num_local_experts, dtype=torch.bool
        )
        self.capture_mask[config.first_k_dense_replace :] = True
        self.force_mask = torch.zeros_like(self.capture_mask)
        self._forwards: list[tuple[torch.nn.Module, Any]] = []

    def install(self) -> None:
        # The routed experts module is GlmMoeDsaExperts, or FP8Experts for the
        # block-FP8 checkpoint; the patterns match the module path only.
        routed = {}
        for name, module in self.model.named_modules():
            match = _ROUTED_EXPERTS.search(name)
            if match is not None:
                routed[int(match.group(1))] = module
        if sorted(routed) != list(self.layers):
            raise RuntimeError(
                f"found GLM routed experts for layers {sorted(routed)}, "
                f"expected {self.layers.start}..{self.layers.stop - 1}"
            )
        for layer in self.layers:
            module = routed[layer]
            self._forwards.append((module, module.forward))
            module.forward = types.MethodType(self._routed_forward(layer), module)

    def restore(self) -> None:
        for module, forward in self._forwards:
            module.forward = forward
        self._forwards.clear()

    def force(self, missing: dict[str, Tensor]) -> None:
        for layer in self.layers:
            unobserved = (
                missing[routed_target(layer, "gate_up")]
                | missing[routed_target(layer, "down")]
            )
            self.capture_mask[layer] = unobserved
            self.force_mask[layer] = unobserved

    def run(self, input_ids: Tensor) -> None:
        self.model.model(input_ids=input_ids, use_cache=False)

    @staticmethod
    def _project(module, projection: str, expert: int, values: Tensor) -> Tensor:
        weight = getattr(module, f"{projection}_proj")[expert]
        scale_inv = getattr(module, f"{projection}_proj_scale_inv", None)
        if scale_inv is None:
            return F.linear(values, weight)
        activation_scale = (
            getattr(module, f"{projection}_proj_activation_scale")[expert]
            if module.activation_scheme == "static"
            else None
        )
        return module.linear(
            values, weight, scale_inv[expert], activation_scale=activation_scale
        )

    def _routed_forward(self, layer: int):
        observer = self
        gate_up_name = routed_target(layer, "gate_up")
        down_name = routed_target(layer, "down")

        def forward(module, hidden_states, top_k_index, top_k_weights):
            fp8 = hasattr(module, "gate_up_proj_scale_inv")
            accumulation_dtype = torch.float32 if fp8 else hidden_states.dtype
            final_hidden_states = torch.zeros_like(
                hidden_states, dtype=accumulation_dtype
            )
            with torch.no_grad():
                expert_mask = F.one_hot(
                    top_k_index, num_classes=module.num_experts + 1
                ).permute(2, 1, 0)
                expert_hit = (
                    torch.greater(expert_mask.sum(dim=(-1, -2)), 0)
                    .nonzero(as_tuple=False)
                    .view(-1)
                )
            accumulator = observer.accumulator
            for expert_tensor in expert_hit:
                expert = int(expert_tensor)
                if expert == module.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert])
                current = hidden_states[token_idx]
                route = top_k_weights[token_idx, top_k_pos]
                capture = bool(observer.capture_mask[layer, expert])
                if capture:
                    accumulator.observe(gate_up_name, expert, current, route)
                activated = module._apply_gate(
                    observer._project(module, "gate_up", expert, current)
                )
                if capture:
                    accumulator.observe(down_name, expert, activated, route)
                projected = observer._project(module, "down", expert, activated)
                final_hidden_states.index_add_(
                    0, token_idx, (projected * route[:, None]).to(accumulation_dtype)
                )

            for expert_tensor in observer.force_mask[layer].nonzero():
                expert = int(expert_tensor[0])
                if accumulator.hit_counts[gate_up_name][expert] != 0:
                    continue
                route = torch.ones(
                    hidden_states.shape[0],
                    dtype=torch.float32,
                    device=hidden_states.device,
                )
                accumulator.observe(
                    gate_up_name, expert, hidden_states, route, forced=True
                )
                activated = module._apply_gate(
                    observer._project(module, "gate_up", expert, hidden_states)
                )
                accumulator.observe(down_name, expert, activated, route, forced=True)

            return final_hidden_states.to(hidden_states.dtype)

        return forward


def load_texts(path: str | os.PathLike[str]) -> list[str]:
    """Read one text per nonempty line; JSONL rows use their text/prompt/content field."""

    texts = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{"):
            row = json.loads(line)
            for key in ("text", "prompt", "content"):
                if key in row:
                    texts.append(str(row[key]))
                    break
            else:
                raise ValueError(
                    f"JSONL row has no text/prompt/content field: {line[:80]}"
                )
        else:
            texts.append(line)
    if not texts:
        raise ValueError(f"calibration corpus {path} is empty")
    return texts


def load_texts_by_kind(path: str | os.PathLike[str]) -> dict[str, list[str]]:
    """``load_texts`` grouped by the JSONL rows' ``kind`` field, in first-seen
    order; a corpus without kinds is one group named ""."""

    texts = load_texts(path)
    kinds = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line:
            kinds.append(
                str(json.loads(line).get("kind", "")) if line.startswith("{") else ""
            )
    groups: dict[str, list[str]] = {}
    for kind, text in zip(kinds, texts):
        groups.setdefault(kind, []).append(text)
    return groups


def corpus_digest(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def token_batches(
    tokenizer,
    texts: list[str],
    sequence_length: int,
    max_tokens: int,
    keep: str = "start",
) -> Iterator[Tensor]:
    """Feed each text as its own sequence, as rendered (no special tokens added).

    A text longer than ``sequence_length`` keeps its first (``keep="start"``)
    or last (``keep="end"``) tokens; one pass over the texts or until
    ``max_tokens``. A corpus of whole chat-template conversations so lets every
    token attend to its own conversation only, as in serving. ``"end"`` keeps a
    long conversation's last reply and its stop decision.
    """

    if keep not in KEEP_CHOICES:
        raise ValueError(f"keep must be one of {KEEP_CHOICES}, not {keep!r}")
    total = 0
    for text in texts:
        if total >= max_tokens:
            return
        encoded = tokenizer.encode(text, add_special_tokens=False)
        length = min(sequence_length, max_tokens - total)
        encoded = encoded[-length:] if keep == "end" else encoded[:length]
        if encoded:
            total += len(encoded)
            yield torch.tensor(encoded, dtype=torch.long).unsqueeze(0)


def capture_glm5_calibration(
    model,
    tokenizer,
    texts: list[str],
    *,
    sequence_length: int = SEQUENCE_LENGTH,
    max_tokens: int = MAX_TOKENS,
    force_unobserved_tokens: int = FORCE_UNOBSERVED_TOKENS,
    device: torch.device | str | None = None,
) -> dict[str, Any]:
    """Run ``texts`` through ``model`` and return the calibration artifact payload.

    Experts that no corpus token reaches get one forced pass of up to
    ``force_unobserved_tokens`` tokens; any that remain unobserved are an error.
    """

    if sequence_length <= 0 or max_tokens <= 0 or force_unobserved_tokens < 0:
        raise ValueError(
            "sequence_length and max_tokens must be positive, "
            "force_unobserved_tokens non-negative"
        )
    device = torch.device(device) if device is not None else model.device
    accumulator = DiagonalSecondMomentAccumulator(glm5_target_specs(model.config))
    observer = GLM5CalibrationObserver(model, accumulator)
    natural_tokens = 0
    forced_tokens = 0
    observer.install()
    try:
        with torch.inference_mode():
            for input_ids in token_batches(
                tokenizer, texts, sequence_length, max_tokens
            ):
                observer.run(input_ids.to(device))
                natural_tokens += input_ids.numel()
                print(f"captured {natural_tokens} tokens", flush=True)
            if accumulator.missing_count() and force_unobserved_tokens:
                observer.force(accumulator.missing())
                for input_ids in token_batches(
                    tokenizer, texts, sequence_length, force_unobserved_tokens
                ):
                    observer.run(input_ids.to(device))
                    forced_tokens += input_ids.numel()
                    remaining = accumulator.missing_count()
                    print(
                        f"forced-fill {forced_tokens} tokens, "
                        f"unobserved target/groups={remaining}",
                        flush=True,
                    )
                    if remaining == 0:
                        break
    finally:
        observer.restore()

    missing = accumulator.missing()
    missing_count = accumulator.missing_count()
    if missing_count:
        examples = [
            [name, group]
            for name, mask in missing.items()
            for group in mask.nonzero().flatten().tolist()
        ][:16]
        raise RuntimeError(
            f"{missing_count} calibration target/groups were unobserved; "
            f"first={examples}. Use more tokens or a broader corpus."
        )
    targets = accumulator.targets()
    hit_counts = torch.cat([target["hit_count"] for target in targets.values()])
    metadata = {
        "corpus_sha256": corpus_digest(texts),
        "sequence_length": sequence_length,
        "token_count": natural_tokens,
        "forced_layer_input_tokens": forced_tokens,
        "forced_layer_experts": observer.force_mask.nonzero().tolist(),
        "hit_count_min": int(hit_counts.min()),
        "hit_count_max": int(hit_counts.max()),
    }
    return {
        "format": _CALIBRATION_FORMAT,
        "version": _CALIBRATION_VERSION,
        "scheme": _CALIBRATION_SCHEME,
        "basis": IQ2R_ACTIVATION_BASIS,
        "targets": targets,
        "metadata": metadata,
    }


# --------------------------------------------------------------------------
# Layer Hessians (LDLQ)
# --------------------------------------------------------------------------

HESSIAN_FORMAT = "iq2r-layer-hessian"
HESSIAN_VERSION = 1
HESSIAN_MANIFEST = "hessians.json"
# GLM chat template roles; generated text runs from <|assistant|> to the next role.
_ASSISTANT = "<|assistant|>"
_ROLES = ("<|user|>", "<|observation|>", "<|system|>")
END_OF_TURN_TOKENS = ("<|endoftext|>", "<|user|>", "<|observation|>")
DECISION_TOKENS = (*END_OF_TURN_TOKENS, "</think>", "<tool_call>")


def hessian_file(layer: int) -> str:
    return f"hessian-layer-{layer:04d}.pt"


@dataclass(frozen=True)
class ChatTokens:
    """Token ids of the chat template's roles and the decision tokens."""

    assistant: int
    roles: frozenset[int]
    end_of_turn: frozenset[int]
    decision: frozenset[int]

    @classmethod
    def from_tokenizer(cls, tokenizer) -> ChatTokens:
        names = (_ASSISTANT, *_ROLES, *DECISION_TOKENS)
        ids = dict(zip(names, tokenizer.convert_tokens_to_ids(list(names))))
        unknown = getattr(tokenizer, "unk_token_id", None)
        missing = [n for n, i in ids.items() if i is None or i == unknown]
        if missing:
            raise ValueError(f"the tokenizer has no single token for {missing}")
        return cls(
            ids[_ASSISTANT],
            frozenset(ids[n] for n in _ROLES),
            frozenset(ids[n] for n in END_OF_TURN_TOKENS),
            frozenset(ids[n] for n in DECISION_TOKENS),
        )


def scored_positions(ids: list[int], tokens: ChatTokens) -> Tensor:
    """Bool [T]: positions whose next token is assistant text or the end of
    turn right after it, the positions a model is scored on. The last position
    has no next token, so a row's final stop decision counts only if the row
    ends with the end-of-turn token after its last reply."""

    generated, on = [], False
    for token in ids:
        if token == tokens.assistant:
            on = True
        elif token in tokens.roles:
            on = False
        generated.append(on)
    target = torch.tensor(generated, dtype=torch.bool)
    ends = torch.tensor([t in tokens.end_of_turn for t in ids], dtype=torch.bool)
    target[1:] |= ends[1:] & target[:-1]
    # Position t predicts token t + 1.
    return torch.cat([target[1:], torch.zeros(1, dtype=torch.bool)])


class LayerHessianAccumulator:
    """Sum x x^T (FP32, on each layer's device) and the row count per layer."""

    def __init__(self, layers: range):
        self.layers = layers
        self.sums: dict[int, Tensor] = {}
        self.counts = {layer: 0 for layer in layers}

    def observe(self, layer: int, rows: Tensor) -> None:
        if rows.shape[0] == 0:
            return
        x = rows.float()
        if layer not in self.sums:
            self.sums[layer] = torch.zeros(
                x.shape[1], x.shape[1], dtype=torch.float32, device=x.device
            )
        self.sums[layer].addmm_(x.T, x)
        self.counts[layer] += x.shape[0]

    def means(self) -> dict[int, Tensor]:
        missing = [layer for layer in self.layers if not self.counts[layer]]
        if missing:
            raise RuntimeError(f"no positions reached layers {missing}")
        return {
            layer: (self.sums[layer] / self.counts[layer]).cpu()
            for layer in self.layers
        }


class MergedHessians:
    """Per-layer weighted mean of several buckets' Hessians, each normalized by
    its trace (the counts add up); saves like a LayerHessianAccumulator.
    ``weights`` (default equal) are normalized to sum to 1."""

    def __init__(
        self, parts: list[LayerHessianAccumulator], weights: list[float] | None = None
    ):
        self.layers = parts[0].layers
        self.parts = parts
        weights = weights or [1.0] * len(parts)
        self.weights = [w / sum(weights) for w in weights]
        self.counts = {
            layer: sum(part.counts[layer] for part in parts) for layer in self.layers
        }

    def means(self) -> dict[int, Tensor]:
        means = [part.means() for part in self.parts]
        merged = {}
        for layer in self.layers:
            total = None
            for m, weight in zip(means, self.weights):
                h = m[layer].double()
                h = weight * h / h.trace()
                total = h if total is None else total + h
            merged[layer] = total.float()
        return merged


def decision_kind_weights(
    kinds: list[str], token_counts: list[int], given: dict[str, float] | None
) -> tuple[list[float], str]:
    """The decision bucket's per-kind weights and how they were set: ``given``
    (one positive weight per kind, normalized) or, without it, each kind's
    share of the tokens fed."""

    if given is None:
        total = sum(token_counts)
        return [count / total for count in token_counts], "token share"
    if set(given) != set(kinds):
        raise ValueError(
            f"decision weights name kinds {sorted(given)}, the corpus has {sorted(kinds)}"
        )
    if any(not weight > 0 for weight in given.values()):
        raise ValueError("decision weights must be positive")
    total = sum(given.values())
    return [given[kind] / total for kind in kinds], "given"


def parse_decision_weights(text: str) -> dict[str, float]:
    """``"chat=0.21,code=0.13"`` -> ``{"chat": 0.21, "code": 0.13}``."""

    weights = {}
    for item in text.split(","):
        kind, sep, value = item.partition("=")
        if not sep or not kind.strip() or kind.strip() in weights:
            raise ValueError(f"expected kind=weight pairs, got {text!r}")
        weights[kind.strip()] = float(value)
    return weights


def _moe_layers(model) -> range:
    config = model.config
    return range(config.first_k_dense_replace, config.num_hidden_layers)


def capture_glm5_hessians(
    model,
    tokenizer,
    texts: list[str],
    *,
    decision: bool,
    sequence_length: int = SEQUENCE_LENGTH,
    max_tokens: int = MAX_TOKENS,
    keep: str = "start",
    device: torch.device | str | None = None,
) -> tuple[LayerHessianAccumulator, int]:
    """Accumulate every MoE layer's input Hessian over the scored positions of
    ``texts`` (``decision``: only those whose FP8 top-1 next token is a decision
    token). Returns the accumulator and the corpus tokens fed."""

    device = torch.device(device) if device is not None else model.device
    tokens = ChatTokens.from_tokenizer(tokenizer)
    decision_ids = torch.tensor(sorted(tokens.decision))
    layers = _moe_layers(model)
    accumulator = LayerHessianAccumulator(layers)
    fed = 0
    with torch.inference_mode():
        for input_ids in token_batches(
            tokenizer, texts, sequence_length, max_tokens, keep
        ):
            scored = scored_positions(input_ids[0].tolist(), tokens)
            fed += input_ids.numel()
            if not scored.any():
                continue
            captured: dict[int, Tensor] = {}
            handles = []
            for layer in layers:

                def hook(module, args, layer=layer, scored=scored, captured=captured):
                    x = args[0].reshape(-1, args[0].shape[-1])
                    rows = x[scored.to(x.device)]
                    if decision:
                        captured[layer] = rows
                    else:
                        accumulator.observe(layer, rows)

                handles.append(
                    model.model.layers[layer].mlp.register_forward_pre_hook(hook)
                )
            try:
                if decision:
                    logits = model(
                        input_ids=input_ids.to(device), use_cache=False
                    ).logits
                    top1 = logits[0][scored.to(logits.device)].argmax(-1).cpu()
                    chosen = torch.isin(top1, decision_ids)
                else:
                    model.model(input_ids=input_ids.to(device), use_cache=False)
            finally:
                for handle in handles:
                    handle.remove()
            if decision:
                for layer, rows in captured.items():
                    accumulator.observe(layer, rows[chosen.to(rows.device)])
            print(
                f"hessian {'decision' if decision else 'all'}: {fed} tokens, "
                f"{accumulator.counts[layers.start]} positions",
                flush=True,
            )
    return accumulator, fed


def save_hessians(
    output_dir: str | os.PathLike[str],
    buckets: dict[str, LayerHessianAccumulator],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write one file per layer and the manifest (atomically, manifest last)."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    means = {name: bucket.means() for name, bucket in buckets.items()}
    layers = next(iter(buckets.values())).layers
    files = {}
    for layer in layers:
        payload = {
            "format": HESSIAN_FORMAT,
            "version": HESSIAN_VERSION,
            "layer": layer,
            **{name: means[name][layer].contiguous() for name in buckets},
            **{f"n_{name}": bucket.counts[layer] for name, bucket in buckets.items()},
        }
        digest = hashlib.sha256()
        for name in sorted(buckets):
            digest.update(payload[name].numpy().tobytes())
        save_calibration(payload, output_dir / hessian_file(layer))
        files[hessian_file(layer)] = digest.hexdigest()
    manifest = {
        "format": HESSIAN_FORMAT,
        "version": HESSIAN_VERSION,
        "layers": [layers.start, layers.stop - 1],
        "buckets": sorted(buckets),
        "counts": {
            name: bucket.counts[layers.start] for name, bucket in buckets.items()
        },
        "files": files,
        **metadata,
    }
    temporary = output_dir / f".{HESSIAN_MANIFEST}.tmp"
    temporary.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    os.replace(temporary, output_dir / HESSIAN_MANIFEST)
    return manifest


def save_calibration(payload: dict[str, Any], path: str | os.PathLike[str]) -> None:
    """Write the artifact atomically so an interrupted capture leaves no partial file."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def config_fingerprint(config) -> str:
    """Hash the architecture fields of a config, ignoring paths and versions."""

    ignored = {"_commit_hash", "_name_or_path", "transformers_version", "torch_dtype"}
    values = {
        key: value
        for key, value in config.to_dict().items()
        if key not in ignored and not key.startswith("_")
    }
    encoded = json.dumps(
        values, sort_keys=True, separators=(",", ":"), default=str
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _dequantized_fp8_linear(
    input: Tensor,
    weight: Tensor,
    weight_scale_inv: Tensor,
    block_size=None,
    activation_scale=None,
    bias: Tensor | None = None,
    **_,
) -> Tensor:
    """Transformers' ``fp8_linear`` as dequantize-then-matmul in plain torch."""
    scale = weight_scale_inv.float()
    if scale.numel() == 1:
        dense = weight.float() * scale
    else:
        # Scale blocks are block_size (128 x 128); the last block of a
        # dimension that is not a multiple of it is partial.
        block_rows, block_columns = tuple(block_size) if block_size else (128, 128)
        rows, columns = weight.shape[-2:]
        grid = scale.repeat_interleave(block_rows, -2)[..., :rows, :]
        grid = grid.repeat_interleave(block_columns, -1)[..., :columns]
        dense = weight.float() * grid
    return F.linear(input, dense.to(input.dtype), bias)


def _dequantized_fp8_experts_forward(module, hidden_states, top_k_index, top_k_weights):
    """Routed FP8 experts as a per-expert loop over ``module.linear`` (which
    then runs ``_dequantized_fp8_linear``), for every experts implementation."""
    final = torch.zeros_like(hidden_states, dtype=torch.float32)
    with torch.no_grad():
        mask = F.one_hot(top_k_index, num_classes=module.num_experts + 1).permute(
            2, 1, 0
        )
        hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero(as_tuple=False).view(-1)
    for expert_tensor in hit:
        expert = int(expert_tensor)
        if expert == module.num_experts:
            continue
        slot, token = torch.where(mask[expert])
        gate_up = module.linear(
            hidden_states[token],
            module.gate_up_proj[expert],
            module.gate_up_proj_scale_inv[expert],
        )
        down = module.linear(
            module._apply_gate(gate_up),
            module.down_proj[expert],
            module.down_proj_scale_inv[expert],
        )
        final.index_add_(0, token, (down * top_k_weights[token, slot, None]).float())
    return final.to(hidden_states.dtype)


def _use_dequantized_fp8_if_needed() -> bool:
    """Route Transformers' FP8 matmuls through ``_dequantized_fp8_linear`` when
    the finegrained-fp8 kernel cannot be loaded; returns whether it did."""
    try:
        from transformers.integrations import finegrained_fp8
    except ImportError:
        return False
    try:
        finegrained_fp8.load_finegrained_fp8_kernel()
        return False
    except ImportError:
        finegrained_fp8.fp8_linear = _dequantized_fp8_linear
        # MoE layers dispatch through the experts-implementation registry.
        mapping = finegrained_fp8.FP8ExpertsInterface._global_mapping
        for name in list(mapping):
            mapping[name] = _dequantized_fp8_experts_forward
        print(
            "finegrained-fp8 kernel unavailable: FP8 matmuls dequantize to BF16",
            flush=True,
        )
        return True


def calibrate_glm5(
    model_dir: str | os.PathLike[str],
    text_file: str | os.PathLike[str],
    output: str | os.PathLike[str] | None,
    *,
    fp8_kernel_dir: str | os.PathLike[str] | None = None,
    hessian_dir: str | os.PathLike[str] | None = None,
    decision_text_file: str | os.PathLike[str] | None = None,
    decision_weights: dict[str, float] | None = None,
    decision_keep: str = "end",
) -> dict[str, Any]:
    """Load the checkpoint, capture the corpus statistics and write the artifact
    (``output``) and/or the layer Hessians (``hessian_dir``).

    ``decision_weights`` sets the decision bucket's per-kind weights (default:
    each kind's share of the tokens fed); ``decision_keep`` is the part of a
    decision text longer than SEQUENCE_LENGTH that is fed (default its end,
    where the reply's last decisions are)."""

    if output is None and hessian_dir is None:
        raise ValueError("nothing to write: give an output and/or a hessian dir")
    if decision_text_file is not None and hessian_dir is None:
        raise ValueError("the decision corpus is for the layer Hessians (hessian dir)")
    if decision_weights is not None and decision_text_file is None:
        raise ValueError("decision weights need a decision corpus")
    if decision_keep not in KEEP_CHOICES:
        raise ValueError(f"decision_keep must be one of {KEEP_CHOICES}")

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if fp8_kernel_dir is not None:
        # Offline hosts: register a local build of kernels-community/finegrained-fp8
        # where Transformers would cache the Hub download.
        import kernels
        from transformers.integrations import hub_kernels

        hub_kernels._KERNEL_MODULE_MAPPING["finegrained-fp8"] = (
            kernels.get_local_kernel(Path(fp8_kernel_dir))
        )

    try:
        from transformers.integrations.finegrained_fp8 import FP8Experts
    except ImportError:
        pass
    else:
        # Transformers 5.16 looks up a TP-layer rewrite for the experts
        # implementation while preparing an FP8 checkpoint, even with none set.
        overrides = getattr(FP8Experts, "_impl_tp_layer_overrides", None)
        if isinstance(overrides, dict):
            overrides.setdefault(None, {})

    dequantized_fp8 = fp8_kernel_dir is None and _use_dequantized_fp8_if_needed()
    texts = load_texts(text_file)
    config = AutoConfig.from_pretrained(model_dir)
    if getattr(config, "model_type", None) != "glm_moe_dsa":
        raise ValueError(f"{model_dir} is not a GLM MoE DSA checkpoint")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        device_map=_DEVICE_MAP,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    ).eval()
    model_metadata = {
        "model_path": str(Path(model_dir).resolve()),
        "model_revision": getattr(config, "_commit_hash", None),
        "model_type": config.model_type,
        "architectures": list(getattr(config, "architectures", None) or ()),
        "model_config_sha256": config_fingerprint(config),
        "device_map": _DEVICE_MAP,
        "fp8_matmul": "dequantized-bf16" if dequantized_fp8 else "finegrained-fp8",
    }
    payload: dict[str, Any] = {}
    if output is not None:
        payload = capture_glm5_calibration(
            model, tokenizer, texts, device=_INPUT_DEVICE
        )
        payload["metadata"].update(model_metadata)
        save_calibration(payload, output)
    if hessian_dir is not None:
        buckets = {}
        corpora = {"all": (texts, False)}
        hessian_metadata = {**model_metadata, "sequence_length": SEQUENCE_LENGTH}
        for name, (bucket_texts, decision) in corpora.items():
            buckets[name], fed = capture_glm5_hessians(
                model, tokenizer, bucket_texts, decision=decision, device=_INPUT_DEVICE
            )
            hessian_metadata[f"{name}_corpus_sha256"] = corpus_digest(bucket_texts)
            hessian_metadata[f"{name}_token_count"] = fed
        if decision_text_file is not None:
            groups = load_texts_by_kind(decision_text_file)
            kinds, parts = {}, []
            for kind, kind_texts in groups.items():
                part, fed = capture_glm5_hessians(
                    model,
                    tokenizer,
                    kind_texts,
                    decision=True,
                    keep=decision_keep,
                    device=_INPUT_DEVICE,
                )
                parts.append(part)
                kinds[kind] = {
                    "positions": part.counts[part.layers.start],
                    "token_count": fed,
                    "corpus_sha256": corpus_digest(kind_texts),
                }
            tokens = [k["token_count"] for k in kinds.values()]
            weights, weight_source = decision_kind_weights(
                list(kinds), tokens, decision_weights
            )
            buckets["decision"] = (
                parts[0] if len(parts) == 1 else MergedHessians(parts, weights)
            )
            hessian_metadata["decision_keep"] = decision_keep
            hessian_metadata["decision_corpus_sha256"] = corpus_digest(
                load_texts(decision_text_file)
            )
            hessian_metadata["decision_token_count"] = sum(
                k["token_count"] for k in kinds.values()
            )
            if len(parts) > 1:
                for kind, weight in zip(kinds.values(), weights):
                    kind["weight"] = weight
                hessian_metadata["decision_kinds"] = kinds
                hessian_metadata["decision_merge"] = (
                    f"weighted mean of trace-normalized kinds ({weight_source} weights)"
                )
        payload["hessians"] = save_hessians(hessian_dir, buckets, hessian_metadata)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-dir", required=True, help="GLM-5.3 block-FP8 checkpoint"
    )
    parser.add_argument(
        "--text-file",
        required=True,
        help="UTF-8 corpus: one text per line, or JSONL with a text/prompt/content field",
    )
    parser.add_argument("--output", help="calibration artifact to write (.pt)")
    parser.add_argument(
        "--hessian-dir",
        help="also write the per-MoE-layer input Hessians for LDLQ to this directory",
    )
    parser.add_argument(
        "--decision-text-file",
        help=(
            "decision corpus for the Hessians' decision bucket (same format as "
            "--text-file; keep it apart from the calibration and evaluation text)"
        ),
    )
    parser.add_argument(
        "--decision-weights",
        help=(
            "per-kind weights of the decision bucket, e.g. 'chat=0.21,code=0.13,...' "
            "(one per kind in the decision corpus; normalized). Default: each kind's "
            "share of the tokens fed"
        ),
    )
    parser.add_argument(
        "--decision-keep",
        choices=KEEP_CHOICES,
        default="end",
        help=(
            f"part of a decision text longer than {SEQUENCE_LENGTH} tokens that is fed "
            "(default: end, which keeps the last reply and its stop decision)"
        ),
    )
    parser.add_argument(
        "--fp8-kernel-dir",
        default=None,
        help=(
            "local build of kernels-community/finegrained-fp8 for hosts without "
            "Hugging Face Hub access"
        ),
    )
    args = parser.parse_args(argv)
    if args.output is None and args.hessian_dir is None:
        parser.error("give --output and/or --hessian-dir")
    if args.decision_text_file is not None and args.hessian_dir is None:
        parser.error("--decision-text-file needs --hessian-dir (it feeds the Hessians)")
    if args.decision_weights is not None and args.decision_text_file is None:
        parser.error("--decision-weights needs --decision-text-file")
    weights = None
    if args.decision_weights is not None:
        try:
            weights = parse_decision_weights(args.decision_weights)
        except ValueError as error:
            parser.error(f"--decision-weights: {error}")
    extra = {}
    if args.hessian_dir is not None:
        extra = {
            "hessian_dir": args.hessian_dir,
            "decision_text_file": args.decision_text_file,
        }
        if args.decision_text_file is not None:
            extra["decision_weights"] = weights
            extra["decision_keep"] = args.decision_keep
    payload = calibrate_glm5(
        args.model_dir,
        args.text_file,
        args.output,
        fp8_kernel_dir=args.fp8_kernel_dir,
        **extra,
    )
    summary = {key: value for key, value in payload.items() if key != "targets"}
    summary["targets"] = len(payload.get("targets", {}))
    if "hessians" in summary:
        summary["hessians"] = {
            k: v for k, v in summary["hessians"].items() if k != "files"
        }
    print(json.dumps(summary, sort_keys=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
