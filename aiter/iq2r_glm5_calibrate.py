# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Capture the GLM-5.3 importance statistics that IQ2R compilation weights with.

The block-FP8 checkpoint runs prefill-only in Hugging Face Transformers over a
text corpus. Input channel k of every routed and shared expert projection in
layers ``first_k_dense_replace`` onwards accumulates a weighted diagonal
second moment

    importance[k] = sum_t r_t^2 * x_t[k]^2 / sum_t r_t^2

x_t is the projection input of token t: the MoE input for gate/up and
act(gate) * up for down. A routed expert sees the tokens routed to it, weighted
by their routing weight r_t; the shared expert sees every token with r_t = 1.
The output is an ``iq2r-calibration`` artifact for
``aiter.iq2r_glm5_compile --calibration-cache``.

Needs Transformers with GLM MoE DSA support (5.16 or newer) and, for the FP8
checkpoint, the ``kernels`` package.
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

from .iq2r_glm5_compile import (
    _CALIBRATION_FORMAT,
    _CALIBRATION_SCHEME,
    _CALIBRATION_VERSION,
)
from .ops.iq2r_format import IQ2R_ACTIVATION_BASIS

_ROUTED_EXPERTS = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.experts$")
_SHARED_EXPERTS = re.compile(r"(?:^|\.)layers\.(\d+)\.mlp\.shared_experts$")


def routed_target(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.experts.{projection}_proj.weight"


def shared_target(layer: int, projection: str) -> str:
    return f"model.layers.{layer}.mlp.shared_experts.{projection}_proj.weight"


@dataclass(frozen=True)
class TargetSpec:
    name: str
    groups: int
    width: int


def glm5_target_specs(config) -> list[TargetSpec]:
    """Return the routed (one group per expert) and shared (one group) targets."""

    if getattr(config, "n_shared_experts", None) != 1:
        raise ValueError("GLM-5 IQ2R calibration requires exactly one shared expert")
    widths = {"gate_up": config.hidden_size, "down": config.moe_intermediate_size}
    specs = []
    for layer in range(config.first_k_dense_replace, config.num_hidden_layers):
        for projection, width in widths.items():
            specs.append(
                TargetSpec(
                    routed_target(layer, projection), config.num_local_experts, width
                )
            )
        for projection, width in widths.items():
            specs.append(TargetSpec(shared_target(layer, projection), 1, width))
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
    implementation and records the gate/up and down inputs on the way; forward
    pre-hooks record the shared expert's inputs. ``force`` limits recording to
    experts with no hits so far and, for those still without hits, records
    every token of the layer's input with weight 1.
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
        self.capture_shared = True
        self._forwards: list[tuple[torch.nn.Module, Any]] = []
        self._handles = []

    def install(self) -> None:
        # The routed experts module is GlmMoeDsaExperts, or FP8Experts for the
        # block-FP8 checkpoint; the patterns match the module path only.
        found = {"routed": {}, "shared": {}}
        for name, module in self.model.named_modules():
            for kind, pattern in (
                ("routed", _ROUTED_EXPERTS),
                ("shared", _SHARED_EXPERTS),
            ):
                match = pattern.search(name)
                if match is not None:
                    found[kind][int(match.group(1))] = module
        for kind, modules in found.items():
            if sorted(modules) != list(self.layers):
                raise RuntimeError(
                    f"found GLM {kind} experts for layers {sorted(modules)}, "
                    f"expected {self.layers.start}..{self.layers.stop - 1}"
                )
        routed, shared = found["routed"], found["shared"]
        for layer in self.layers:
            module = routed[layer]
            self._forwards.append((module, module.forward))
            module.forward = types.MethodType(self._routed_forward(layer), module)
            for projection, linear in (
                ("gate_up", shared[layer].gate_proj),
                ("down", shared[layer].down_proj),
            ):
                self._handles.append(
                    linear.register_forward_pre_hook(
                        self._shared_hook(shared_target(layer, projection))
                    )
                )

    def restore(self) -> None:
        for module, forward in self._forwards:
            module.forward = forward
        self._forwards.clear()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def force(self, missing: dict[str, Tensor]) -> None:
        # The shared expert keeps natural-corpus statistics only.
        self.capture_shared = False
        for layer in self.layers:
            unobserved = (
                missing[routed_target(layer, "gate_up")]
                | missing[routed_target(layer, "down")]
            )
            self.capture_mask[layer] = unobserved
            self.force_mask[layer] = unobserved

    def run(self, input_ids: Tensor) -> None:
        self.model.model(input_ids=input_ids, use_cache=False)

    def _shared_hook(self, target: str):
        def hook(module, args):
            del module
            if self.capture_shared:
                self.accumulator.observe(target, 0, args[0].detach())

        return hook

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


def corpus_digest(texts: list[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def token_batches(
    tokenizer, texts: list[str], sequence_length: int, max_tokens: int
) -> Iterator[Tensor]:
    """Tokenize the texts in order, EOS after each, cycling until ``max_tokens``."""

    tokens: list[int] = []
    index = 0
    while len(tokens) < max_tokens:
        encoded = tokenizer.encode(texts[index % len(texts)], add_special_tokens=True)
        if not encoded and tokenizer.eos_token_id is None:
            raise ValueError("tokenizer produced no calibration tokens")
        tokens.extend(encoded)
        if tokenizer.eos_token_id is not None:
            tokens.append(tokenizer.eos_token_id)
        index += 1
    tokens = tokens[:max_tokens]
    for start in range(0, len(tokens), sequence_length):
        chunk = tokens[start : start + sequence_length]
        yield torch.tensor(chunk, dtype=torch.long).unsqueeze(0)


def capture_glm5_calibration(
    model,
    tokenizer,
    texts: list[str],
    *,
    sequence_length: int = 512,
    max_tokens: int = 32768,
    force_unobserved_tokens: int = 512,
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


def calibrate_glm5(
    model_dir: str | os.PathLike[str],
    text_file: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    sequence_length: int = 512,
    max_tokens: int = 32768,
    force_unobserved_tokens: int = 512,
    device: str = "cuda:0",
    device_map: str | None = None,
    fp8_kernel_dir: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Load the checkpoint, capture the corpus statistics and write the artifact."""

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

    texts = load_texts(text_file)
    config = AutoConfig.from_pretrained(model_dir)
    if getattr(config, "model_type", None) != "glm_moe_dsa":
        raise ValueError(f"{model_dir} is not a GLM MoE DSA checkpoint")
    target = torch.device(device)
    if device_map is None:
        index = target.index if target.type == "cuda" else None
        device_map = {"": target if index is None else index}
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        config=config,
        device_map=device_map,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
    ).eval()
    payload = capture_glm5_calibration(
        model,
        tokenizer,
        texts,
        sequence_length=sequence_length,
        max_tokens=max_tokens,
        force_unobserved_tokens=force_unobserved_tokens,
        device=target,
    )
    payload["metadata"].update(
        {
            "model_path": str(Path(model_dir).resolve()),
            "model_revision": getattr(config, "_commit_hash", None),
            "model_type": config.model_type,
            "architectures": list(getattr(config, "architectures", None) or ()),
            "model_config_sha256": config_fingerprint(config),
            "device_map": device_map if isinstance(device_map, str) else None,
        }
    )
    save_calibration(payload, output)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capture the IQ2R calibration artifact (per-expert diagonal second "
            "moments) of a GLM-5.3 block-FP8 checkpoint"
        )
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument(
        "--text-file",
        required=True,
        help="UTF-8 corpus: one text per line, or JSONL with a text/prompt/content field",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument(
        "--force-unobserved-tokens",
        type=int,
        default=512,
        help="tokens of layer input recorded for experts the corpus never reaches",
    )
    parser.add_argument("--device", default="cuda:0", help="device of the input tokens")
    parser.add_argument(
        "--device-map",
        choices=("auto", "balanced", "balanced_low_0", "sequential"),
        default=None,
        help="Accelerate device map for a checkpoint larger than one GPU",
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
    payload = calibrate_glm5(
        args.model_dir,
        args.text_file,
        args.output,
        sequence_length=args.sequence_length,
        max_tokens=args.max_tokens,
        force_unobserved_tokens=args.force_unobserved_tokens,
        device=args.device,
        device_map=args.device_map,
        fp8_kernel_dir=args.fp8_kernel_dir,
    )
    summary = {key: value for key, value in payload.items() if key != "targets"}
    summary["targets"] = len(payload["targets"])
    print(json.dumps(summary, sort_keys=True, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
