#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Reduce MXFP4 tuning candidates with an OpenAI recommendation.

The model is only a preselector. Candidate legality, correctness validation,
benchmarking, and winner selection remain owned by ``Mxfp4FlydslTuner``.
Every model-provided ID is checked against the existing legal candidate set;
any API, cache, or response error falls back to that complete set.

Preselection follows ``--tune-stage``: a joint sweep is reduced over
(GEMM1, GEMM2) pairs, and a ``gemm1``/``gemm2`` sweep over the one stage that
actually varies. See ``Mxfp4FlydslTuner`` for the stage semantics.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import numbers
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, "") and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

try:  # pragma: no cover - depends on which GEMM1 launcher this tree ships
    from aiter.ops.flydsl.mxfp4_gemm1_kernels import _effective_use_nt
except ImportError:  # launcher without the BM32 streaming-load override

    def _effective_use_nt(*, n_tokens, topk, NE, BM, use_nt, inline_quant):
        """Identity fallback.

        This only describes what the runtime will actually do with the
        candidate's ``use_nt``. A launcher that does not override it is
        described correctly by returning it unchanged.
        """
        return use_nt


from aiter.ops.flydsl.mxfp4_kname import (
    _parse_mxfp4_g1_kname,
    _parse_mxfp4_g2_kname,
    _select_mxfp4_block_m,
)

try:
    from . import gemm_moe_tune
except ImportError:
    import gemm_moe_tune

KEY_COLUMNS = gemm_moe_tune.FMOE_KEY_COLUMNS
RESULT_COLUMNS = gemm_moe_tune.FMOE_RESULT_COLUMNS
Mxfp4FlydslTuner = gemm_moe_tune.Mxfp4FlydslTuner
# Re-exported: the registry GEMM2 parser shim now lives with the base tuner.
_install_modern_gemm2_parser = gemm_moe_tune._install_modern_gemm2_parser


_PROMPT_SCHEMA_VERSION = 1
_CACHE_SCHEMA_VERSION = 1
_DEFAULT_TOP_K = 4
_DEFAULT_MAX_CANDIDATES = 256
_DEFAULT_CACHE_PATH = Path.home() / ".cache/aiter/mxfp4_openai_recommendations.json"
_POLICY_PATH = _REPO_ROOT / "docs" / "mxfp4_gemm1_tuning_best_practices.md"
_SHAPE_FIELDS = (
    "gfx",
    "cu_num",
    "token",
    "model_dim",
    "inter_dim",
    "expert",
    "topk",
    "act_type",
    "dtype",
    "q_dtype_a",
    "q_dtype_w",
    "q_type",
    "use_g1u1",
    "doweight_stage1",
    "block_m",
)


class RecommendationError(RuntimeError):
    """An expected recommendation/cache validation failure."""


@dataclass
class SelectionResult:
    candidates: list[dict]
    source: str
    full_count: int


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise ValueError("value must be positive")
    return value


def _positive_float(value):
    value = float(value)
    if value <= 0:
        raise ValueError("value must be positive")
    return value


def _json_scalar(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        result = float(value)
        return None if math.isnan(result) else result
    if hasattr(value, "item"):
        return _json_scalar(value.item())
    return str(value)


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(value):
    if not isinstance(value, str):
        value = _canonical_json(value)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _shape_payload(row):
    return {
        field: _json_scalar(row[field])
        for field in _SHAPE_FIELDS
        if field in row and _json_scalar(row[field]) is not None
    }


def _candidate_id(candidate):
    identity = {
        "block_m": int(candidate["block_m"]),
        "gemm1": str(candidate["kernelName1"]),
        "gemm2": str(candidate["kernelName2"]),
    }
    return f"cfg_{_sha256(identity)[:20]}"


def _g1_features(row, candidate):
    parsed = _parse_mxfp4_g1_kname(str(candidate["kernelName1"]))
    effective_use_nt = _effective_use_nt(
        n_tokens=int(row["token"]),
        topk=int(row["topk"]),
        NE=int(row["expert"]),
        BM=int(parsed["BM"]),
        use_nt=bool(parsed["use_nt"]),
        inline_quant=bool(parsed["inline_quant"]),
    )
    return {
        "BM": int(parsed["BM"]),
        "BN": int(parsed["BN"]),
        "BK": int(parsed["BK"]),
        "use_nt": bool(effective_use_nt),
        "inline_quant": bool(parsed["inline_quant"]),
        "xcd_swizzle": int(parsed["xcd_swizzle"]),
        "a_dtype": str(parsed["a_dtype"]),
        "out_dtype": str(parsed["out_dtype"]),
        "activation": str(parsed["act"]),
        "interleave": bool(parsed["interleave"]),
        "enable_bias": bool(parsed["enable_bias"]),
        # Wave shape. These must be part of the feature set: the deduplication
        # below treats candidates with equal features as the same kernel, so
        # omitting them silently collapsed every k_wave/num_waves variant onto
        # its 4-wave k_wave=1 sibling and the axis was never benchmarked.
        "num_waves": int(parsed.get("num_waves", 4)),
        "k_wave": int(parsed.get("k_wave", 1)),
    }


def _g2_features(candidate):
    name = str(candidate["kernelName2"])
    try:
        from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

        params = get_flydsl_kernel_params(name)
    except Exception:  # noqa: BLE001 - feature extraction must not break tuning
        params = None

    if params is not None:
        allowed = (
            "a_dtype",
            "b_dtype",
            "out_dtype",
            "tile_m",
            "tile_n",
            "tile_k",
            "mode",
            "sort_block_m",
            "persist",
            "b_nt",
            "xcd_swizzle",
            "k_wave",
            "waves_per_eu",
        )
        return {
            "family": "layout",
            **{
                key: _json_scalar(params[key])
                for key in allowed
                if key in params and _json_scalar(params[key]) is not None
            },
        }

    try:
        parsed = _parse_mxfp4_g2_kname(name)
    except (KeyError, TypeError, ValueError):
        # The opaque identity is kept locally in the candidate hash. No kernel
        # name is disclosed to the API.
        return {"family": "locked"}
    return {
        "family": "native",
        "BM": int(parsed["BM"]),
        "BN": int(parsed["BN"]),
        "BK": int(parsed["BK"]),
        "atomic": bool(parsed["atomic"]),
        "use_nt": bool(parsed["use_nt"]),
        "mxfp4out": bool(parsed["mxfp4out"]),
        "cshuffle": bool(parsed["cshuffle"]),
        "xcd_swizzle": int(parsed["xcd_swizzle"]),
    }


def _candidate_descriptor(row, candidate):
    return {
        "id": _candidate_id(candidate),
        "gemm1": _g1_features(row, candidate),
        "gemm2": _g2_features(candidate),
    }


def _deduplicate_effective_candidates(row, candidates):
    """Drop candidates that dispatch to the same effective GEMM1/GEMM2 pair."""

    deduplicated = []
    positions = {}
    requested_nt = {}
    for candidate in candidates:
        copied = dict(candidate)
        try:
            descriptor = _candidate_descriptor(row, copied)
            parsed = _parse_mxfp4_g1_kname(str(copied["kernelName1"]))
            key = _canonical_json(
                {
                    "block_m": int(copied["block_m"]),
                    "gemm1": descriptor["gemm1"],
                    # Keep different GEMM2 implementations distinct even when
                    # feature extraction cannot parse a locked kernel.
                    "gemm2_hash": _sha256(str(copied["kernelName2"])),
                }
            )
            candidate_requested_nt = bool(parsed["use_nt"])
        except (KeyError, TypeError, ValueError):
            # Existing candidate generation is authoritative. If a future name
            # cannot be parsed, preserve it instead of incorrectly collapsing it.
            key = _canonical_json(
                {
                    "block_m": copied.get("block_m"),
                    "gemm1_hash": _sha256(str(copied.get("kernelName1"))),
                    "gemm2_hash": _sha256(str(copied.get("kernelName2"))),
                }
            )
            candidate_requested_nt = False

        if key not in positions:
            positions[key] = len(deduplicated)
            requested_nt[key] = candidate_requested_nt
            deduplicated.append(copied)
        elif requested_nt[key] and not candidate_requested_nt:
            # Prefer the name that already encodes the effective cache policy.
            deduplicated[positions[key]] = copied
            requested_nt[key] = False
    return deduplicated


def _g1_baseline_score(row, candidate):
    """Rank GEMM1 candidates by distance from the hand-written dispatch defaults."""
    target_bm = _select_mxfp4_block_m(
        token=int(row["token"]),
        expert=int(row["expert"]),
        topk=int(row["topk"]),
    )
    average_rows = (
        int(row["token"]) * int(row["topk"]) + int(row["expert"]) - 1
    ) // int(row["expert"])
    target_bn = 128 if average_rows <= 32 else 256
    target_interleave = str(row.get("gate_mode", "")).lower() == "interleave"
    baseline_kernel = row.get("_baseline_kernelName1") or row.get("kernelName1")
    if baseline_kernel:
        try:
            target_interleave = bool(
                _parse_mxfp4_g1_kname(str(baseline_kernel))["interleave"]
            )
        except (KeyError, TypeError, ValueError):
            pass

    features = _g1_features(row, candidate)
    total_m_blocks = (int(row["token"]) * int(row["topk"]) + features["BM"] - 1) // (
        features["BM"]
    )
    target_nt = (
        features["BM"] in (16, 32, 64)
        and total_m_blocks < int(row["expert"])
        and not (features["a_dtype"] == "fp8" and features["BM"] == 64)
    )
    if features["BM"] == 16:
        target_nt = True
    return (
        features["BM"] != target_bm,
        features["BN"] != target_bn,
        features["use_nt"] != target_nt,
        features["interleave"] != target_interleave,
        features["xcd_swizzle"] != 0,
    )


def _g2_baseline_score(row, candidate):
    """Rank GEMM2 candidates by distance from the dispatch tile defaults.

    ``family: "locked"`` means feature extraction could not read the kernel, so
    it sorts last rather than silently becoming the safety baseline.
    """
    features = _g2_features(candidate)
    family = features.get("family")
    if family == "locked":
        return (True,) * 6
    model_dim = int(row["model_dim"])
    inter_dim = int(row["inter_dim"])
    target_tn = 256 if model_dim % 256 == 0 else 128
    target_tk = 256 if inter_dim % 256 == 0 else 128
    if family == "native":
        tile_n, tile_k = features["BN"], features["BK"]
        atomic = features["atomic"]
        persist = False
        xcd_swizzle = features["xcd_swizzle"]
        b_nt = 2 if features["use_nt"] else 0
    else:
        tile_n = int(features.get("tile_n", 0) or 0)
        tile_k = int(features.get("tile_k", 0) or 0)
        atomic = str(features.get("mode", "atomic")) == "atomic"
        persist = bool(features.get("persist", False))
        xcd_swizzle = int(features.get("xcd_swizzle", 0) or 0)
        b_nt = int(features.get("b_nt", 0) or 0)
    return (
        tile_n != target_tn,
        tile_k != target_tk,
        not atomic,
        b_nt != 2,
        persist,
        xcd_swizzle != 0,
    )


def _select_baseline(row, candidates, stage="both"):
    """Pick the candidate that is always benchmarked, whatever the model says.

    It is the closest thing to the shipped dispatch heuristic for the stage
    being swept, so a bad recommendation can never leave the shape worse off
    than the untuned default.
    """
    if stage == "gemm2":

        def score(candidate):
            return (*_g2_baseline_score(row, candidate), _candidate_id(candidate))

    elif stage == "gemm1":

        def score(candidate):
            return (*_g1_baseline_score(row, candidate), _candidate_id(candidate))

    else:

        def score(candidate):
            return (
                *_g1_baseline_score(row, candidate),
                *_g2_baseline_score(row, candidate),
                _candidate_id(candidate),
            )

    return min(candidates, key=score)


class RecommendationCache:
    def __init__(self, path):
        self.path = Path(path).expanduser()
        self._data = None

    def _load(self):
        if self._data is not None:
            return self._data
        if not self.path.exists():
            self._data = {"version": _CACHE_SCHEMA_VERSION, "entries": {}}
            return self._data
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RecommendationError("corrupt recommendation cache") from exc
        if (
            not isinstance(data, dict)
            or data.get("version") != _CACHE_SCHEMA_VERSION
            or not isinstance(data.get("entries"), dict)
        ):
            raise RecommendationError("invalid recommendation cache schema")
        self._data = data
        return data

    def get(self, key):
        entry = self._load()["entries"].get(key)
        if entry is not None and not isinstance(entry, dict):
            raise RecommendationError("invalid recommendation cache entry")
        return entry

    def put(self, key, entry):
        data = self._load()
        updated = {
            "version": data["version"],
            "entries": {**data["entries"], key: entry},
        }
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp_path = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
            temp_path.write_text(
                json.dumps(updated, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.replace(temp_path, self.path)
        except OSError as exc:
            raise RecommendationError("failed to write recommendation cache") from exc
        self._data = updated


def _default_client_factory(**kwargs):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RecommendationError(
            "OpenAI SDK is unavailable; install requirements-tuning.txt"
        ) from exc
    return OpenAI(**kwargs)


class OpenAICandidateSelector:
    def __init__(
        self,
        *,
        model,
        base_url,
        api_key,
        top_k,
        timeout,
        max_candidates,
        cache,
        policy_path=_POLICY_PATH,
        refresh=False,
        default_headers=None,
        client_factory: Callable[..., Any] | None = None,
    ):
        self.model = str(model or "").strip()
        self.base_url = str(base_url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.top_k = int(top_k)
        self.timeout = float(timeout)
        self.max_candidates = int(max_candidates)
        self.cache = cache
        self.policy_path = Path(policy_path)
        self.refresh = bool(refresh)
        self.default_headers = dict(default_headers or {})
        self.client_factory = client_factory or _default_client_factory

    def _policy(self):
        try:
            policy = self.policy_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise RecommendationError("cannot read tuning policy") from exc
        return policy, _sha256(policy)

    def _cache_key(self, row, candidate_ids, policy_hash, stage):
        return _sha256(
            {
                "prompt_schema": _PROMPT_SCHEMA_VERSION,
                "policy_hash": policy_hash,
                "model": self.model,
                "base_url_hash": _sha256(self.base_url or "official"),
                "shape": _shape_payload(row),
                "candidate_ids": sorted(candidate_ids),
                "top_k": self.top_k,
                "stage": stage,
            }
        )

    @staticmethod
    def _validate_ids(ids, allowed_ids, expected_count):
        if (
            not isinstance(ids, list)
            or len(ids) != expected_count
            or any(not isinstance(candidate_id, str) for candidate_id in ids)
            or len(set(ids)) != len(ids)
            or any(candidate_id not in allowed_ids for candidate_id in ids)
        ):
            raise RecommendationError("invalid candidate IDs in model response")
        return ids

    def _request_ids(
        self,
        *,
        policy,
        shape,
        baseline,
        candidates,
        expected_count,
        stage,
    ):
        if not self.api_key:
            raise RecommendationError("OPENAI_API_KEY is not set")
        client_kwargs = {
            "api_key": self.api_key,
            "timeout": self.timeout,
            "max_retries": 0,
        }
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        if self.default_headers:
            client_kwargs["default_headers"] = self.default_headers
        client = self.client_factory(**client_kwargs)
        payload = {
            "task": "select_mxfp4_benchmark_candidates",
            # Which half of the pipeline actually varies across `candidates`.
            # In a single-stage sweep the other half is identical everywhere and
            # carries no signal.
            "tune_stage": stage,
            "required_count": expected_count,
            "shape": shape,
            # The baseline is already forced into the final set. Hide its ID so
            # the model cannot spend one of its additional slots selecting it.
            "safety_baseline_features": {
                key: value for key, value in baseline.items() if key != "id"
            },
            "candidates": candidates,
        }
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Select benchmark candidates, not a presumed final winner. "
                        "Follow the tuning policy below. Return one JSON object with "
                        "exactly one key, candidate_ids. candidate_ids must contain "
                        "the requested number of unique IDs from the supplied list. "
                        "Do not invent IDs.\n\n"
                        f"{policy}"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(payload, sort_keys=True),
                },
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=256,
        )
        try:
            content = response.choices[0].message.content
            parsed = json.loads(content)
        except (AttributeError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RecommendationError("malformed model response") from exc
        if not isinstance(parsed, dict) or set(parsed) != {"candidate_ids"}:
            raise RecommendationError("unexpected model response schema")
        allowed_ids = {candidate["id"] for candidate in candidates}
        return self._validate_ids(parsed["candidate_ids"], allowed_ids, expected_count)

    def select(self, row, candidates, stage="both"):
        full = [dict(candidate) for candidate in candidates]
        if not full:
            return SelectionResult([], "no_candidates", 0)
        if len(full) <= self.top_k:
            return SelectionResult(full, "within_budget", len(full))
        if len(full) > self.max_candidates:
            return SelectionResult(full, "candidate_limit_fallback", len(full))
        if not self.model:
            return SelectionResult(full, "missing_model_fallback", len(full))

        try:
            policy, policy_hash = self._policy()
            descriptors = [_candidate_descriptor(row, candidate) for candidate in full]
            by_id = {
                descriptor["id"]: candidate
                for descriptor, candidate in zip(descriptors, full)
            }
            descriptor_by_id = {
                descriptor["id"]: descriptor for descriptor in descriptors
            }
            if len(by_id) != len(full):
                raise RecommendationError("candidate ID collision")

            baseline = _select_baseline(row, full, stage)
            baseline_id = _candidate_id(baseline)
            model_descriptors = [
                descriptor
                for descriptor in descriptors
                if descriptor["id"] != baseline_id
            ]
            expected_model_count = min(self.top_k - 1, len(model_descriptors))
            final_count = 1 + expected_model_count
            cache_key = self._cache_key(row, by_id, policy_hash, stage)
            cached = self.cache.get(cache_key)
            if cached is not None and not self.refresh:
                selected_ids = self._validate_ids(
                    cached.get("selected_ids"), set(by_id), final_count
                )
                if baseline_id not in selected_ids:
                    raise RecommendationError("cached selection omits baseline")
                return SelectionResult(
                    [dict(by_id[candidate_id]) for candidate_id in selected_ids],
                    "cache",
                    len(full),
                )

            if expected_model_count == 0:
                selected_ids = [baseline_id]
            else:
                model_ids = self._request_ids(
                    policy=policy,
                    shape=_shape_payload(row),
                    baseline=descriptor_by_id[baseline_id],
                    candidates=model_descriptors,
                    expected_count=expected_model_count,
                    stage=stage,
                )
                selected_ids = [baseline_id, *model_ids]

            self.cache.put(
                cache_key,
                {
                    "model": self.model,
                    "policy_hash": policy_hash,
                    "selected_ids": selected_ids,
                    "universe_size": len(full),
                },
            )
            return SelectionResult(
                [dict(by_id[candidate_id]) for candidate_id in selected_ids],
                "api",
                len(full),
            )
        except Exception as exc:  # noqa: BLE001 - safe exhaustive fallback is required
            return SelectionResult(
                full,
                f"{type(exc).__name__}_fallback",
                len(full),
            )


class OpenAIMxfp4FlydslTuner(Mxfp4FlydslTuner):
    """MXFP4 tuner with parent-side OpenAI candidate preselection."""

    def _setup_specific_arguments(self):
        super()._setup_specific_arguments()
        self.parser.add_argument(
            "--openai-model",
            default=os.environ.get("OPENAI_MODEL", ""),
            help="OpenAI model name (or OPENAI_MODEL).",
        )
        self.parser.add_argument(
            "--openai-base-url",
            default=os.environ.get("OPENAI_BASE_URL", ""),
            help="Optional OpenAI-compatible endpoint (or OPENAI_BASE_URL).",
        )
        self.parser.add_argument(
            "--openai-user",
            default=os.environ.get("OPENAI_USER") or os.environ.get("AMD_NTID", ""),
            help="Optional gateway user/NTID header (or OPENAI_USER/AMD_NTID).",
        )
        self.parser.add_argument(
            "--openai-top-k",
            type=_positive_int,
            default=_DEFAULT_TOP_K,
            help="Final candidate budget per shape, including one safety baseline.",
        )
        self.parser.add_argument(
            "--openai-timeout",
            type=_positive_float,
            default=30.0,
            help="OpenAI request timeout in seconds.",
        )
        self.parser.add_argument(
            "--openai-max-candidates",
            type=_positive_int,
            default=_DEFAULT_MAX_CANDIDATES,
            help="Fall back to exhaustive tuning above this prompt candidate count.",
        )
        self.parser.add_argument(
            "--openai-cache",
            default=os.environ.get(
                "AITER_OPENAI_TUNER_CACHE", str(_DEFAULT_CACHE_PATH)
            ),
            help="Validated recommendation cache path.",
        )
        self.parser.add_argument(
            "--openai-refresh",
            action="store_true",
            help="Ignore a matching cache entry and request a fresh recommendation.",
        )
        self.parser.add_argument(
            "--openai-plan-only",
            action="store_true",
            help=(
                "Run candidate preselection, write the recommendation cache, and "
                "exit without benchmarking. Lets a host that can reach the API "
                "produce the plan for a GPU host that cannot: copy the cache over "
                "and rerun with the same --openai-model/--openai-top-k."
            ),
        )

    @staticmethod
    def _make_selector(args):
        base_url = str(args.openai_base_url or "")
        amd_gateway = "llm-api.amd.com" in base_url.lower()
        gateway_key = os.environ.get("AMD_LLM_GATEWAY_KEY") or os.environ.get(
            "LLM_GATEWAY_KEY", ""
        )
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if amd_gateway and not api_key:
            api_key = gateway_key
        default_headers = {}
        if amd_gateway and gateway_key:
            default_headers["Ocp-Apim-Subscription-Key"] = gateway_key
        if amd_gateway and args.openai_user:
            default_headers["user"] = str(args.openai_user)
        return OpenAICandidateSelector(
            model=args.openai_model,
            base_url=base_url,
            api_key=api_key,
            top_k=args.openai_top_k,
            timeout=args.openai_timeout,
            max_candidates=args.openai_max_candidates,
            cache=RecommendationCache(args.openai_cache),
            refresh=args.openai_refresh,
            default_headers=default_headers or None,
        )

    def _candidate_rows(self, row, stage="auto"):
        active = getattr(self, "_openai_active_candidates", None)
        if active is not None:
            return [dict(candidate) for candidate in active]
        return super()._candidate_rows(row, stage)

    def _plan_rows(self, rows, args):
        """Preselect candidates per shape, returning (row, selected, full, stage).

        Shared by tune() and --openai-plan-only, so the plan a disconnected GPU
        host replays from cache is produced by exactly the code that runs there.
        """
        selector = self._make_selector(args)
        plans = []
        for row in rows:
            stage = self._resolve_tune_stage(row, args)
            legal = Mxfp4FlydslTuner._candidate_rows(self, row, stage)
            full = _deduplicate_effective_candidates(row, legal)
            decision = selector.select(row, full, stage)
            print(
                f"[openai-mxfp4] token={row['token']} inter={row['inter_dim']} "
                f"expert={row['expert']} topk={row['topk']} stage={stage} "
                f"candidates={decision.full_count}->{len(decision.candidates)} "
                f"source={decision.source}",
                flush=True,
            )
            plans.append((row, decision.candidates, full, stage, decision.source))
        return plans

    def plan_only(self, args):
        """Populate the recommendation cache without touching the GPU."""
        self.apply_stage_defaults(args)
        untunedf = self.get_untuned_gemm_list(args.untune_file)
        rows = [series.to_dict() for _, series in untunedf.iterrows()]
        plans = self._plan_rows(rows, args)

        sources = {}
        for *_rest, source in plans:
            sources[source] = sources.get(source, 0) + 1
        print(
            f"[openai-mxfp4] planned {len(plans)} shapes -> {args.openai_cache}",
            flush=True,
        )
        print(
            "[openai-mxfp4] sources: "
            + ", ".join(f"{name}={count}" for name, count in sorted(sources.items())),
            flush=True,
        )
        fallbacks = sum(
            count for source, count in sources.items() if source.endswith("_fallback")
        )
        if fallbacks:
            print(
                f"[openai-mxfp4] WARNING: {fallbacks} shapes fell back to the full "
                "candidate set; those would be tuned exhaustively",
                flush=True,
            )
        return sources

    def _torch_ref(self, data, topk, dtype, activation):
        if getattr(self, "_openai_stage1_only", False):
            return None
        return Mxfp4FlydslTuner._torch_ref(data, topk, dtype, activation)

    @staticmethod
    def _metric_value(value):
        text = str(value).strip()
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
        return float(text)

    def _stage1_context(self, row, data):
        cache_key = (
            int(row["token"]),
            int(row["inter_dim"]),
            int(row["block_m"]),
            str(row["act_type"]),
        )
        cached = data.get("_openai_stage1_context")
        if cached is not None and cached["cache_key"] == cache_key:
            return cached

        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import moe_sorting
        from aiter.ops.flydsl.mxfp4_v2_tune_utils import (
            v2_stage1_sorted_ref,
        )

        if str(row["act_type"]).endswith("Situv2"):
            activation = ActivationType.Situv2
        elif str(row["act_type"]).endswith("Swiglu"):
            activation = ActivationType.Swiglu
        else:
            activation = ActivationType.Silu

        topk = int(row["topk"])
        block_m = int(row["block_m"])
        token = int(row["token"])
        model_dim = int(row["model_dim"])
        inter_dim = int(row["inter_dim"])
        experts = int(row["expert"])
        ref1 = self.run_torch_moe_stage1(
            data["a1_qt"],
            data["w1_qt"],
            data["w2_qt"],
            data["topk_weights"],
            data["topk_ids"],
            data["a1_scale"],
            data["w1_scale"],
            w1_bias=data["bias1"],
            dtype=dtypes.bf16,
            activation=activation,
            quant_type=QuantType.per_1x32,
            doweight_stage1=False,
            topk=topk,
        )
        (
            sorted_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            moe_buf,
            m_indices,
            _reverse_sorted,
        ) = moe_sorting(
            data["topk_ids"],
            data["topk_weights"],
            experts,
            model_dim,
            dtypes.bf16,
            block_size=block_m,
            accumulate=False,
            output_aux=True,
        )
        sorted_ref = v2_stage1_sorted_ref(
            ref1,
            data["topk_ids"],
            sorted_ids,
            sorted_expert_ids,
            sorted_ids.shape[0],
            token=token,
            inter_dim=inter_dim,
            bm_s1=block_m,
            max_sorted=sorted_ids.shape[0],
        )
        context = {
            "cache_key": cache_key,
            "activation": activation,
            "sorted_ids": sorted_ids,
            "sorted_weights": sorted_weights,
            "sorted_expert_ids": sorted_expert_ids,
            "num_valid_ids": num_valid_ids,
            "moe_buf": moe_buf,
            "m_indices": m_indices,
            "sorted_ref": sorted_ref,
        }
        data["_openai_stage1_context"] = context
        return context

    def _run_stage1_candidate(self, row, candidate, args, data):
        from aiter.fused_moe import _mxfp4_a4w4_stage1_fw
        from aiter.ops.flydsl.mxfp4_v2_tune_utils import (
            v2_stage1_dequant_cosine_err,
        )
        from aiter.test_common import run_perftest

        context = self._stage1_context(row, data)
        kernel_name = str(candidate["kernelName1"])
        parsed = _parse_mxfp4_g1_kname(kernel_name)
        topk = int(row["topk"])
        inter_dim = int(row["inter_dim"])
        w1_a16, w1s_a16 = self._a4w4_stage1_inputs(data, bool(parsed["interleave"]))

        def run():
            return _mxfp4_a4w4_stage1_fw(
                data["input"],
                w1_a16,
                data["w2_a16"],
                context["sorted_ids"],
                context["sorted_expert_ids"],
                context["num_valid_ids"],
                None,
                topk,
                block_m=int(candidate["block_m"]),
                w1_scale=w1s_a16,
                kernelName1=kernel_name,
                m_indices=context["m_indices"],
                moe_buf=context["moe_buf"],
                interleave=bool(parsed["interleave"]),
                bias1=data["bias1"],
                swiglu_limit=float(parsed.get("swiglu_limit", 7.0)),
                situ_beta=float(parsed.get("situ_beta", 1.0)),
                situ_linear_beta=float(parsed.get("situ_linear_beta", 1.0)),
            )

        output, _output_scale = run()
        error = v2_stage1_dequant_cosine_err(
            context["sorted_ref"],
            output,
            msg=f"stage1[{kernel_name}]",
            inter_dim=inter_dim,
            adtype="fp4",
        )
        if not math.isfinite(float(error)) or float(error) > args.errRatio:
            raise RuntimeError(f"stage1 cosine err_ratio {error} > {args.errRatio}")

        _, elapsed_us, kernel_times = run_perftest(
            run,
            num_warmup=int(args.warmup),
            num_iters=int(args.iters),
            return_kernel_times=True,
        )
        us1 = round(
            sum(
                float(value)
                for name, value in kernel_times.items()
                if any(marker in str(name) for marker in self.STAGE1_KERNEL_MARKERS)
            ),
            4,
        )
        if us1 <= 0:
            raise RuntimeError("profiler did not report GEMM1 target kernel time")
        us2 = self._metric_value(row["_baseline_us2"])
        candidate.update(
            {
                "us1": us1,
                "us2": us2,
                "us": round(us1 + us2, 4),
                # err1 here is stage1 output vs a bf16-rounded sorted reference.
                # For an fp4-emitting GEMM1 that comparison carries a floor of a
                # few tenths of a percent from the reference's own bf16 rounding,
                # not from the kernel, so do not read it as kernel error.
                "err1": self._format_err(error),
                "err2": self._format_err(self._metric_value(row["_baseline_err2"])),
            }
        )
        candidate["tflops"], candidate["bw"] = self._calculate_candidate_performance(
            row, candidate, candidate["us"]
        )
        print(
            f"[openai-mxfp4-g1] token={row['token']} "
            f"inter={row['inter_dim']} {kernel_name} "
            f"us1={us1} stage1_err={candidate['err1']}",
            flush=True,
        )
        return round(float(elapsed_us), 4)

    @staticmethod
    def _isolated_stage1(row, stage):
        """True when GEMM1 can be benchmarked without running GEMM2 at all.

        --baseline-config supplies the locked GEMM2's measured time, so a
        GEMM1 sweep only has to launch stage 1 and check it against a sorted
        stage-1 reference. That is both faster and a sharper correctness signal
        than an end-to-end cosine, which a fast GEMM2 can mask.
        """
        return stage == "gemm1" and "_baseline_us2" in row

    def _run_candidate(
        self,
        row,
        candidate,
        args,
        data=None,
        ref=None,
        model_dim_pad=0,
        stage="both",
    ):
        if not self._isolated_stage1(row, stage):
            return super()._run_candidate(
                row,
                candidate,
                args,
                data=data,
                ref=ref,
                model_dim_pad=model_dim_pad,
                stage=stage,
            )
        if data is None:
            raise RuntimeError("GEMM1-only tuning requires prepared data")
        return self._run_stage1_candidate(row, candidate, args, data)

    @staticmethod
    def _invalid_result(candidate):
        if candidate is None:
            return True
        value = candidate.get("us", Mxfp4FlydslTuner.INVALID_TIME)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return True
        return (
            not math.isfinite(value)
            or value == Mxfp4FlydslTuner.INVALID_TIME
            or value == Mxfp4FlydslTuner.INF_TIME
        )

    def _tune_preselected_shape(self, row, args, selected, full, stage="both"):
        if not selected:
            raise RuntimeError("no legal MXFP4 candidates for shape")
        stage1_only = self._isolated_stage1(row, stage)
        self._openai_stage1_only = stage1_only
        self._openai_active_candidates = [dict(candidate) for candidate in selected]
        try:
            best, profiles, rejects = super()._tune_one_shape(row, args)
        finally:
            self._openai_active_candidates = None
            self._openai_stage1_only = False

        if not self._invalid_result(best):
            return best, profiles, rejects

        selected_ids = {_candidate_id(candidate) for candidate in selected}
        remaining = [
            dict(candidate)
            for candidate in full
            if _candidate_id(candidate) not in selected_ids
        ]
        if not remaining:
            return best, profiles, rejects

        print(
            "[openai-mxfp4] selected candidates all failed; "
            f"retrying {len(remaining)} remaining candidates",
            flush=True,
        )
        self._openai_stage1_only = stage1_only
        self._openai_active_candidates = remaining
        try:
            retry_best, retry_profiles, retry_rejects = super()._tune_one_shape(
                row, args
            )
        finally:
            self._openai_active_candidates = None
            self._openai_stage1_only = False
        # Keep the rejects from the preselected pass too: "the model's picks
        # all failed" is exactly what a post-mortem needs to see.
        return (
            retry_best,
            [*profiles, *retry_profiles],
            [*rejects, *retry_rejects],
        )

    def _failure_candidate(self, row, candidates, exc):
        if candidates:
            candidate = dict(candidates[0])
        else:
            candidate = self._candidate_row(
                row,
                int(row.get("block_m", 0) or 0),
                "",
                str(row.get("kernelName2", "")),
            )
        candidate["us1"] = self.INVALID_TIME
        candidate["us2"] = self.INVALID_TIME
        candidate["us"] = self.INVALID_TIME
        candidate["kernelName1"] = (f"FAILED: {type(exc).__name__}: {exc!s}")[:240]
        return candidate

    def _run_shape_safely(self, row, args, selected, full, stage="both"):
        try:
            return self._tune_preselected_shape(row, args, selected, full, stage)
        except Exception as exc:  # noqa: BLE001 - preserve other shapes
            print(
                "[openai-mxfp4] shape failed: " f"{type(exc).__name__}: {exc!s}",
                flush=True,
            )
            return (
                self._failure_candidate(row, full or selected, exc),
                [],
                [self._reject_row(row, None, stage, exc)],
            )

    def tune(self, untunedf, tunedf, args):
        del tunedf
        rows = [row.to_dict() for _, row in untunedf.iterrows()]
        self._profile_rows = []
        self._reject_rows = []
        if not rows:
            return []

        plans = [plan[:4] for plan in self._plan_rows(rows, args)]

        mp_num = int(getattr(args, "mp", 1) or 1)
        try:
            ngpu = torch.cuda.device_count()
        except Exception:  # noqa: BLE001
            ngpu = 1
        mp_num = max(1, min(mp_num, ngpu, len(plans)))

        if mp_num <= 1:
            shape_results = [
                self._run_shape_safely(row, args, selected, full, stage)
                for row, selected, full, stage in plans
            ]
        else:
            print(
                f"[openai-mxfp4] tuning {len(plans)} shapes across {mp_num} GPUs",
                flush=True,
            )
            ctx = multiprocessing.get_context("spawn")
            with ctx.Manager() as manager:
                gpu_queue = manager.Queue()
                for gpu in range(mp_num):
                    gpu_queue.put(gpu)
                payloads = [
                    (self.keys, row, args, selected, full, stage, gpu_queue)
                    for row, selected, full, stage in plans
                ]
                with ctx.Pool(processes=mp_num, maxtasksperchild=1) as pool:
                    shape_results = pool.map(
                        _openai_mxfp4_shape_worker, payloads, chunksize=1
                    )

        bests = []
        for best, profiles, rejects in shape_results:
            bests.append(best)
            self._profile_rows.extend(profiles)
            self._reject_rows.extend(rejects)
        return bests


def _openai_mxfp4_shape_worker(payload):
    keys, row, args, selected, full, stage, gpu_queue = payload
    gpu = gpu_queue.get()
    tuner = OpenAIMxfp4FlydslTuner.__new__(OpenAIMxfp4FlydslTuner)
    tuner.keys = keys
    try:
        torch.cuda.set_device(gpu)
        print(
            f"[openai-mxfp4] token={row['token']} inter={row['inter_dim']} "
            f"expert={row['expert']} topk={row['topk']} stage={stage} -> GPU{gpu}",
            flush=True,
        )
        return tuner._run_shape_safely(row, args, selected, full, stage)
    except Exception as exc:  # noqa: BLE001 - preserve other worker results
        print(
            f"[openai-mxfp4] shape failed on GPU{gpu}: "
            f"{type(exc).__name__}: {exc!s}",
            flush=True,
        )
        return (
            tuner._failure_candidate(row, full or selected, exc),
            [],
            [tuner._reject_row(row, None, stage, exc)],
        )
    finally:
        gpu_queue.put(gpu)


def main():
    tuner = OpenAIMxfp4FlydslTuner(
        "openaiMxfp4FlydslTuner",
        KEY_COLUMNS,
        RESULT_COLUMNS,
        "OpenAI-guided replacement MXFP4 FlyDSL MoE tuner",
    )
    args = tuner.parse_args()
    if args.openai_plan_only:
        tuner.plan_only(args)
        return
    tuner.run(args, False)


if __name__ == "__main__":
    main()
