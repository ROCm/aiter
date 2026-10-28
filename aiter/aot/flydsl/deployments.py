# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MoE deployments the AOT replays compile for.

``moe_deployments.json`` records each MoE model's architecture once, and each
deployment of it by its knobs: EP size, EPLB redundant experts, shared-expert
handling, quant recipe and token range. What one rank hands the runtime is
derived here, with the convention ATOM and vLLM share:

- ``fused_moe``: the ``fused_moe`` call behind an EP dispatch, which ``moe.py``
  replays and ``op_tests/test_moe_ep.py -t aot_ep_profiles`` checks;
- ``mega_moe``: the ``MegaMoEV2`` each rank builds, which ``mega_moe.py``
  replays.

Usage:
    # Validate deployment files and list what they expand to
    python -m aiter.aot.flydsl.deployments [deployments.json ...]
"""

import itertools
import json
import os
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

DEFAULT_DEPLOYMENTS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe_deployments.json")
]
SCHEMA_VERSION = 1
# MXFP4 (per_1x32) expert weights with FP8 or FP4 activations.
QUANTS = ("a8w4", "a4w4")
ACTIVATIONS = ("Silu", "Situv2", "Swiglu")
# separate: the shared experts run outside fused_moe; fused: every rank runs
# them as extra local experts, as ATOM and vLLM fuse them.
SHARED_EXPERT_MODES = ("separate", "fused")
_SECTION_ARCHS = {"fused_moe": ("gfx950", "gfx1250"), "mega_moe": ("gfx950",)}
_SECTION_ACTIVATIONS = {"fused_moe": ACTIVATIONS, "mega_moe": ("Silu", "Situv2")}
_MODEL_ID = re.compile(r"[a-z0-9][a-z0-9.\-]*")

_FILE_KEYS = {"schema_version", "models", *_SECTION_ARCHS}
_MODEL_REQUIRED = {
    "name",
    "model_dim",
    "inter_dim",
    "routed_experts",
    "top_k",
    "shared_experts",
    "activation",
    "source",
}
_MODEL_OPTIONAL = {"swiglu_limit", "situ_beta", "situ_linear_beta", "note"}
_SITU_PARAMS = ("situ_beta", "situ_linear_beta")
_DEPLOYMENT_REQUIRED = {"model", "gfx", "cu_num", "quant", "ep", "source"}
_DEPLOYMENT_OPTIONAL = {"redundant_experts", "note"}
_SECTION_KEYS = {
    "fused_moe": ({"max_tokens"}, {"shared_expert_mode"}),
    "mega_moe": ({"max_tokens_per_rank"}, set()),
}


@dataclass(frozen=True)
class Model:
    """A MoE model's architecture, as its ``config.json`` gives it."""

    id: str
    name: str
    model_dim: int
    inter_dim: int
    routed_experts: int
    top_k: int
    shared_experts: int
    activation: str
    swiglu_limit: float
    situ_beta: float
    situ_linear_beta: float
    source: str


@dataclass(frozen=True)
class _Deployment:
    id: str
    path: str
    model: Model
    gfx: str
    cu_num: int
    quant: str
    ep: int
    redundant_experts: int
    source: str

    @property
    def physical_experts(self) -> int:
        """Expert slots of the EP group: routed experts plus EPLB replicas."""
        return self.model.routed_experts + self.redundant_experts


@dataclass(frozen=True)
class FusedMoECall(_Deployment):
    """What one rank passes ``fused_moe`` behind an EP dispatch."""

    shared_expert_mode: str
    max_tokens: int

    @property
    def fused_shared(self) -> int:
        return self.model.shared_experts if self.shared_expert_mode == "fused" else 0

    @property
    def local_experts(self) -> int:
        """``w1.shape[0]``: the rank's slots, then the fused shared experts."""
        return self.physical_experts // self.ep + self.fused_shared

    @property
    def topk_cols(self) -> int:
        """``topk_ids.shape[1]``. Fused shared experts take a column each, and
        a sentinel column masks them on the tokens of other ranks."""
        if not self.fused_shared:
            return self.model.top_k
        return self.model.top_k + self.fused_shared + 1

    @property
    def mask_size(self) -> int:
        """``expert_mask.numel()``: every slot, the fused shared experts and
        a sentinel."""
        return self.physical_experts + self.fused_shared + 1


@dataclass(frozen=True)
class MegaMoEProfile(_Deployment):
    """The ``MegaMoEV2`` each rank of a deployment builds, at each of its
    token capacities."""

    max_tokens_per_rank: tuple[int, ...]

    @property
    def experts_per_rank(self) -> int:
        return self.physical_experts // self.ep


class Deployments(NamedTuple):
    fused_moe: list[FusedMoECall]
    mega_moe: list[MegaMoEProfile]


def _fail(where: str, message: str):
    raise ValueError(f"{where}: {message}")


def _keys(where: str, entry, required: set, optional: set) -> dict:
    if not isinstance(entry, dict):
        _fail(where, f"expected an object, got {type(entry).__name__}")
    unknown = sorted(set(entry) - required - optional)
    if unknown:
        _fail(where, f"unknown keys {unknown}")
    missing = sorted(required - set(entry))
    if missing:
        _fail(where, f"missing keys {missing}")
    return entry


def _int(where: str, key: str, value, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _fail(where, f"{key} must be an integer >= {minimum}, got {value!r}")
    return value


def _float(where: str, key: str, value, positive: bool = False) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or value < 0
        or (positive and value == 0)
    ):
        bound = "> 0" if positive else ">= 0"
        _fail(where, f"{key} must be a number {bound}, got {value!r}")
    return float(value)


def _text(where: str, key: str, value) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(where, f"{key} must be a non-empty string")
    return value


def _choice(where: str, key: str, value, choices: Sequence[str]) -> str:
    if value not in choices:
        _fail(where, f"{key} must be one of {list(choices)}, got {value!r}")
    return value


def _values(where: str, entry: dict, key: str, default) -> list:
    """A knob given as one value or as a list of the values to expand."""
    value = entry.get(key, default)
    if not isinstance(value, list):
        return [value]
    if not value or len(set(value)) != len(value):
        _fail(where, f"{key} must list distinct values")
    return value


def _model(where: str, model_id: str, entry) -> Model:
    if not _MODEL_ID.fullmatch(model_id):
        _fail(where, "a model id is lower-case letters, digits, '.' and '-'")
    entry = _keys(where, entry, _MODEL_REQUIRED, _MODEL_OPTIONAL)
    activation = _choice(where, "activation", entry["activation"], ACTIVATIONS)
    # The betas are checkpoint constants MegaMoE compiles in; leaving them to a
    # default would build kernels serving never asks for.
    for key in _SITU_PARAMS:
        if (key in entry) != (activation == "Situv2"):
            _fail(where, f"{key} is given exactly for Situv2 models")
    model = Model(
        id=model_id,
        name=_text(where, "name", entry["name"]),
        model_dim=_int(where, "model_dim", entry["model_dim"]),
        inter_dim=_int(where, "inter_dim", entry["inter_dim"]),
        routed_experts=_int(where, "routed_experts", entry["routed_experts"]),
        top_k=_int(where, "top_k", entry["top_k"]),
        shared_experts=_int(where, "shared_experts", entry["shared_experts"], 0),
        activation=activation,
        swiglu_limit=_float(where, "swiglu_limit", entry.get("swiglu_limit", 0.0)),
        situ_beta=_float(where, "situ_beta", entry.get("situ_beta", 1.0), True),
        situ_linear_beta=_float(
            where, "situ_linear_beta", entry.get("situ_linear_beta", 1.0), True
        ),
        source=_text(where, "source", entry["source"]),
    )
    if model.top_k > model.routed_experts:
        _fail(where, "top_k exceeds routed_experts")
    return model


def _expand(section: str, where: str, path: str, entry, models: dict) -> list:
    """The deployments one entry of ``section`` lists, one per combination of
    its list-valued knobs."""
    required, optional = _SECTION_KEYS[section]
    entry = _keys(
        where,
        entry,
        _DEPLOYMENT_REQUIRED | required,
        _DEPLOYMENT_OPTIONAL | optional,
    )
    model = models.get(entry["model"])
    if model is None:
        _fail(where, f"unknown model {entry['model']!r}")
    _choice(where, "activation", model.activation, _SECTION_ACTIVATIONS[section])
    gfx = _choice(where, "gfx", entry["gfx"], _SECTION_ARCHS[section])
    common = {
        "path": path,
        "model": model,
        "gfx": gfx,
        "cu_num": _int(where, "cu_num", entry["cu_num"]),
        "source": _text(where, "source", entry["source"]),
    }
    if section == "fused_moe":
        mode = _choice(
            where,
            "shared_expert_mode",
            entry.get("shared_expert_mode", "separate"),
            SHARED_EXPERT_MODES,
        )
        if mode == "fused" and not model.shared_experts:
            _fail(where, f"{model.id} has no shared experts to fuse")
        common["shared_expert_mode"] = mode
        common["max_tokens"] = _int(where, "max_tokens", entry["max_tokens"])
    else:
        common["max_tokens_per_rank"] = tuple(
            _int(where, "max_tokens_per_rank", m)
            for m in _values(where, entry, "max_tokens_per_rank", None)
        )
    deployments = []
    for quant, ep, redundant in itertools.product(
        _values(where, entry, "quant", None),
        _values(where, entry, "ep", None),
        _values(where, entry, "redundant_experts", 0),
    ):
        _choice(where, "quant", quant, QUANTS)
        _int(where, "ep", ep, 2)
        _int(where, "redundant_experts", redundant, 0)
        # ATOM refuses an EP group its expert slots do not split over evenly.
        if (model.routed_experts + redundant) % ep:
            _fail(
                where,
                f"{model.routed_experts} routed + {redundant} redundant experts "
                f"do not split over EP{ep}",
            )
        name = f"{model.id}/{gfx}/{quant}/ep{ep}"
        name += f"-r{redundant}" if redundant else ""
        if common.get("shared_expert_mode") == "fused":
            name += "-fused"
        kind = FusedMoECall if section == "fused_moe" else MegaMoEProfile
        deployments.append(
            kind(id=name, quant=quant, ep=ep, redundant_experts=redundant, **common)
        )
    return deployments


def _read(path: str) -> dict:
    with open(path) as f:
        document = json.load(f)
    _keys(path, document, {"schema_version"}, _FILE_KEYS)
    if document["schema_version"] != SCHEMA_VERSION:
        _fail(path, f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(document.get("models", {}), dict):
        _fail(path, "models must map model ids to models")
    for section in _SECTION_ARCHS:
        if not isinstance(document.get(section, []), list):
            _fail(path, f"{section} must be a list of deployments")
    return document


def load(paths: Sequence[str] = DEFAULT_DEPLOYMENTS) -> Deployments:
    """Every deployment the files of ``paths`` list, in file order. A
    deployment names a model of its own file or of the default one."""
    paths = list(dict.fromkeys(os.path.abspath(path) for path in paths))
    defaults = [os.path.abspath(path) for path in DEFAULT_DEPLOYMENTS]
    documents = {path: _read(path) for path in dict.fromkeys(defaults + paths)}
    models = {}
    for path, document in documents.items():
        for model_id, entry in document.get("models", {}).items():
            model = _model(f"{path}: models.{model_id}", model_id, entry)
            if models.setdefault(model_id, model) != model:
                _fail(path, f"model {model_id!r} differs from its other definition")
    sections = {section: {} for section in _SECTION_ARCHS}
    for path in paths:
        for section, found in sections.items():
            for index, entry in enumerate(documents[path].get(section, [])):
                where = f"{path}: {section}[{index}]"
                for deployment in _expand(section, where, path, entry, models):
                    if deployment.id in found:
                        _fail(where, f"{deployment.id} is listed twice")
                    found[deployment.id] = deployment
    return Deployments(*(list(found.values()) for found in sections.values()))


def main(argv: list[str] | None = None) -> int:
    paths = (sys.argv[1:] if argv is None else argv) or DEFAULT_DEPLOYMENTS
    try:
        deployments = load(paths)
    except (OSError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(
        f"{'fused_moe':42} {'expert':>6} {'topk':>4} {'mask':>4} {'max_M':>7}  source"
    )
    for call in deployments.fused_moe:
        print(
            f"{call.id:42} {call.local_experts:6} {call.topk_cols:4} "
            f"{call.mask_size:4} {call.max_tokens:7}  {call.source}"
        )
    print(f"\n{'mega_moe':42} {'E/rank':>6} {'topk':>4}  max tokens per rank; source")
    for profile in deployments.mega_moe:
        mtprs = ", ".join(map(str, profile.max_tokens_per_rank))
        print(
            f"{profile.id:42} {profile.experts_per_rank:6} "
            f"{profile.model.top_k:4}  {mtprs}; {profile.source}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
