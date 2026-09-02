#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Recommend MXFP4 a4w4 tuning candidates with an LLM, as a CSV the tuner reads.

`Mxfp4FlydslTuner` enumerates its whole legal space per shape -- 1470 (g1, g2)
pairs for Kimi-K3 and 2928 for Kimi-K2 -- and benchmarks all of it, which is
hours of GPU time per model config. This script does the "decide what to try"
half on the CPU and writes the answer to a CSV that
``gemm_moe_tune.py --mxfp4-flydsl --candidate-csv`` consumes as its config list.

Per shape:

1. Enumerate the legal candidates through ``Mxfp4FlydslTuner._candidate_rows``,
   so the recommendation can never name a kernel the runtime rejects.
2. Drop candidates that dispatch to the same effective kernel pair.
3. Rank by distance from the shipped dispatch heuristic and keep the best
   ``--max-candidates`` as the prompt budget.
4. Ask the model to choose ``--top-k`` of them.
5. Pin the heuristic baseline into the output so a bad recommendation can never
   leave a shape worse off than the untuned default.

There is deliberately no heuristic-only fallback: if the model is unreachable or
answers with anything unexpected, the run fails and writes nothing rather than
emitting a silently degraded CSV that looks like a recommendation.

No GPU is touched -- only kernel-name parsers and the kernel registries.

Usage:
    export OPENAI_API_KEY=... OPENAI_MODEL=gpt-5.5
    python csrc/ck_gemm_moe_2stages_codegen/recommend_mxfp4_candidates.py \\
        -i aiter/configs/model_configs/kimik3_a4w4_untuned_fmoe.csv \\
        -o /tmp/kimik3_candidates.csv --gfx gfx950 --cu-num 256 --top-k 8
"""

import argparse
import csv
import hashlib
import json
import math
import numbers
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gemm_moe_tune import Mxfp4FlydslTuner

from aiter.ops.flydsl.mxfp4_kname import (
    _parse_mxfp4_g1_kname,
    _parse_mxfp4_g2_kname,
)

_CACHE_SCHEMA_VERSION = 1
_DEFAULT_TOP_K = 8
_DEFAULT_MAX_CANDIDATES = 256
# Reasoning models burn a few hundred hidden tokens before emitting this small
# JSON object; a 256-token cap returned empty with finish_reason="length".
_DEFAULT_MAX_COMPLETION_TOKENS = 4096
_RETRY_BACKOFF_S = 5.0
_DEFAULT_CACHE_PATH = Path.home() / ".cache/aiter/mxfp4_candidate_recommendations.json"

# Mirrors the `key` list in gemm_moe_tune.py's __main__ block: this is the shape
# identity the tuner indexes rows by, so the CSV has to carry all of it.
TUNER_KEYS = (
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
)
CANDIDATE_COLUMNS = (
    "rank",
    "block_m",
    "ksplit",
    "kernelName1",
    "kernelName2",
    "candidate_id",
    "source",
    "reason",
)
CSV_COLUMNS = TUNER_KEYS + CANDIDATE_COLUMNS

_DEFAULT_POLICY = """\
MXFP4 a4w4 GEMM1/GEMM2 candidate selection policy.

You are picking which kernel configurations get *benchmarked*, not guessing a
winner. Spend the budget on configurations that are individually plausible and
collectively diverse -- a slate of near-identical kernels wastes the sweep.

Background that matters:
- block_m should track routed rows per expert (token*topk/expert). Small token
  counts want block_m 16; large ones want 64 or 128.
- block_m 16 is the only inline-quant ("f16in") GEMM1 variant, and it is the only
  one that can carry hidden prefetch ("hpf"). hpf helps most at very small token
  counts and is roughly neutral above token ~16.
- BN 256 suits wide output tiles; BN 128 suits few routed rows. BN 64 only exists
  for block_m 32 non-inline separated.
- xcd_swizzle spreads blocks across XCDs and tends to matter once there are many
  more blocks than CUs.
- k_wave > 1 splits the K loop across waves; it only pays when K tiles per wave
  stay large enough to hide latency.
- GEMM2 tile_n should divide model_dim and tile_k should divide inter_dim.
  atomic epilogs avoid a separate reduction; reduce epilogs need one.

Prefer covering distinct regions of the space (different block_m, BN, epilog)
over several neighbours of one point.
"""


class RecommendationError(RuntimeError):
    """Raised when a usable recommendation cannot be produced."""


@dataclass
class ShapePlan:
    row: dict
    selected: list
    baseline: dict
    full_count: int
    pruned_count: int
    source: str


def _positive_int(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _positive_float(value):
    value = float(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
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


def _default_block_m(*, token, expert, topk):
    """The block_m the shipped dispatch heuristic would pick.

    Reproduced here because `_select_mxfp4_block_m` was removed from
    aiter/ops/flydsl/mxfp4_kname.py as dead code; this is only a ranking prior,
    never a correctness gate.
    """
    token, expert, topk = int(token), int(expert), int(topk)
    average_rows = (token * topk + expert - 1) // expert
    if token == 1:
        # BM16's fused inline quantization has excessive error for a single token.
        return 32
    if token <= 128:
        return 16
    if average_rows <= 32:
        return 32
    if average_rows <= 64:
        return 64
    return 128


def _shape_payload(row):
    return {
        field: _json_scalar(row[field])
        for field in TUNER_KEYS
        if field in row and _json_scalar(row[field]) is not None
    }


def _candidate_id(candidate):
    identity = {
        "block_m": int(candidate["block_m"]),
        "gemm1": str(candidate["kernelName1"]),
        "gemm2": str(candidate["kernelName2"]),
    }
    return f"cfg_{_sha256(identity)[:20]}"


def _g1_features(candidate):
    parsed = _parse_mxfp4_g1_kname(str(candidate["kernelName1"]))
    return {
        "BM": int(parsed["BM"]),
        "BN": int(parsed["BN"]),
        "BK": int(parsed["BK"]),
        "use_nt": bool(parsed["use_nt"]),
        "inline_quant": bool(parsed["inline_quant"]),
        "prefetch_hidden": bool(parsed.get("prefetch_hidden", False)),
        "xcd_swizzle": int(parsed["xcd_swizzle"]),
        "a_dtype": str(parsed["a_dtype"]),
        "out_dtype": str(parsed["out_dtype"]),
        "activation": str(parsed["act"]),
        "interleave": bool(parsed["interleave"]),
        "enable_bias": bool(parsed["enable_bias"]),
        # Wave shape must be part of the feature set: dedup treats equal features
        # as the same kernel, so omitting these collapses every k_wave/num_waves
        # variant onto its 4-wave k_wave=1 sibling and the axis is never swept.
        "num_waves": int(parsed.get("num_waves", 4)),
        "k_wave": int(parsed.get("k_wave", 1)),
    }


def _g2_features(candidate):
    name = str(candidate["kernelName2"])
    try:
        from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

        params = get_flydsl_kernel_params(name)
    except Exception:  # noqa: BLE001 - feature extraction must not break planning
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


def _candidate_descriptor(candidate):
    return {
        "id": _candidate_id(candidate),
        "gemm1": _g1_features(candidate),
        "gemm2": _g2_features(candidate),
    }


def _deduplicate_effective_candidates(candidates):
    """Drop candidates that dispatch to the same effective GEMM1/GEMM2 pair."""
    deduplicated = []
    seen = set()
    for candidate in candidates:
        copied = dict(candidate)
        try:
            descriptor = _candidate_descriptor(copied)
            key = _canonical_json(
                {
                    "block_m": int(copied["block_m"]),
                    "gemm1": descriptor["gemm1"],
                    # Keep distinct GEMM2 implementations apart even when feature
                    # extraction cannot read a locked kernel.
                    "gemm2_hash": _sha256(str(copied["kernelName2"])),
                }
            )
        except (KeyError, TypeError, ValueError):
            # Candidate generation is authoritative. If a future name cannot be
            # parsed, keep it rather than incorrectly collapsing it.
            key = _canonical_json(
                {
                    "block_m": copied.get("block_m"),
                    "gemm1_hash": _sha256(str(copied.get("kernelName1"))),
                    "gemm2_hash": _sha256(str(copied.get("kernelName2"))),
                }
            )
        if key not in seen:
            seen.add(key)
            deduplicated.append(copied)
    return deduplicated


def _g1_baseline_score(row, candidate):
    """Rank GEMM1 candidates by distance from the hand-written dispatch defaults."""
    target_bm = _default_block_m(
        token=row["token"], expert=row["expert"], topk=row["topk"]
    )
    average_rows = (
        int(row["token"]) * int(row["topk"]) + int(row["expert"]) - 1
    ) // int(row["expert"])
    target_bn = 128 if average_rows <= 32 else 256

    features = _g1_features(candidate)
    total_m_blocks = (int(row["token"]) * int(row["topk"]) + features["BM"] - 1) // (
        features["BM"]
    )
    target_nt = features["BM"] == 16 or (
        features["BM"] in (32, 64) and total_m_blocks < int(row["expert"])
    )
    return (
        features["BM"] != target_bm,
        features["BN"] != target_bn,
        features["use_nt"] != target_nt,
        features["interleave"],
        features["xcd_swizzle"] != 0,
        features["k_wave"] != 1,
        features["num_waves"] != 4,
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
    target_tn = 256 if int(row["model_dim"]) % 256 == 0 else 128
    target_tk = 256 if int(row["inter_dim"]) % 256 == 0 else 128
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


def _rank_key(row, candidate):
    return (
        *_g1_baseline_score(row, candidate),
        *_g2_baseline_score(row, candidate),
        _candidate_id(candidate),
    )


def _select_baseline(row, candidates):
    """The candidate always benchmarked, whatever the model says.

    It is the closest thing to the shipped dispatch heuristic, so a bad
    recommendation can never leave the shape worse off than the untuned default.
    """
    return min(candidates, key=lambda c: _rank_key(row, c))


def _strata(row, candidate):
    """The (block_m, BN, xcd_swizzle, GEMM2 epilog) cell a candidate belongs to.

    Every axis here is one whose prior is known wrong on real configs, so each
    must be covered rather than predicted:
      - block_m: the shipped heuristic returns 32 where kimi-k3 tunes 16, and
        128 where it tunes 64.
      - BN: the "few routed rows -> BN128" rule is backwards; kimi-k3 tokens
        3-32 ship BN256.
      - xcd_swizzle: `_rank_key` penalises xcd != 0, yet a majority of kimi-k3's
        tuned rows use _xcd2/_xcd4. Unstratified, xcd=0 filled every cell and all
        seven GEMM1 misses at top_k=16 were xcd variants.
      - prefetch_hidden: `_rank_key` does not score it at all, so within a cell
        hpf-vs-not fell to the candidate-id tiebreak -- arbitrary, on the axis
        the policy calls out as the biggest small-token win.
      - k_wave: 7 of the 9 GEMM1 misses across glm5/kimi-k2/qwen were `_kw2`
        configs. kimi-k3 cannot expose this (3584/256 = 14 is not divisible by
        4, so its k_wave space is thin), which is exactly why an axis must be
        covered rather than predicted from one config's behaviour.
      - use_nt: the last outstanding miss (kimi-k2 token 2048, inter 256,
        `32x256x256_nt_xcd4`) sat at index 32 of a 64-candidate cell because
        `_rank_key` penalises `use_nt` against its target, so no prompt budget
        reached it. Every GEMM1 axis the name encodes is now a stratum.
    """
    g1 = _g1_features(candidate)
    g2 = _g2_features(candidate)
    epilog = g2.get("mode") or ("atomic" if g2.get("atomic", True) else "reduce")
    return (
        g1["BM"],
        g1["BN"],
        g1["xcd_swizzle"],
        g1["prefetch_hidden"],
        g1["k_wave"],
        g1["use_nt"],
        str(epilog),
    )


def _prune(row, candidates, limit):
    """Keep `limit` candidates, round-robin across (block_m, epilog) strata.

    A global sort cannot be used here. `_rank_key`'s first component is
    `BM != target_bm`, so sorting by it puts every candidate of the heuristic's
    preferred block_m ahead of every other -- and with thousands of candidates
    against a 256 budget the prompt ends up holding exactly one block_m tier.
    Measured on kimi-k3: at token 4096/8192/32768 the prompt was 100% block_m
    128 while the shipped config is block_m 64, so the model could not have
    recommended it at any temperature. block_m is also the axis that dominates
    MoE performance, which makes it the worst one to silently collapse.

    Round-robin instead: every legal stratum contributes its locally best
    candidates, so a wrong prior costs ordering, never coverage.
    """
    if len(candidates) <= limit:
        return list(candidates)
    buckets = {}
    for candidate in candidates:
        buckets.setdefault(_strata(row, candidate), []).append(candidate)
    for cell in buckets.values():
        cell.sort(key=lambda c: _rank_key(row, c))
    # Visit strata in heuristic-preference order so ties favour the prior.
    order = sorted(buckets, key=lambda k: _rank_key(row, buckets[k][0]))
    kept, depth = [], 0
    while len(kept) < limit:
        progressed = False
        for cell in order:
            if depth < len(buckets[cell]):
                kept.append(buckets[cell][depth])
                progressed = True
                if len(kept) == limit:
                    break
        if not progressed:
            break
        depth += 1
    return kept


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
                json.dumps(updated, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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
            "OpenAI SDK is unavailable; pip install openai"
        ) from exc
    return OpenAI(**kwargs)


class CandidateAgent:
    """Asks a model which candidates to benchmark. Never falls back."""

    def __init__(
        self,
        *,
        model,
        api_key=None,
        base_url=None,
        user=None,
        timeout=60.0,
        top_k=_DEFAULT_TOP_K,
        max_candidates=_DEFAULT_MAX_CANDIDATES,
        policy=_DEFAULT_POLICY,
        cache=None,
        refresh=False,
        retries=2,
        extra_headers=None,
        client_factory=None,
    ):
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.user = user or os.environ.get("OPENAI_USER") or os.environ.get("AMD_NTID")
        self.timeout = timeout
        self.top_k = top_k
        self.max_candidates = max_candidates
        self.policy = policy
        self.cache = cache
        self.refresh = refresh
        self.retries = int(retries)
        self.extra_headers = dict(extra_headers or {})
        # Resolved at call time, not bound as a default, so tests can swap the
        # module-level factory without reaching into the instance.
        self.client_factory = client_factory

    @property
    def default_headers(self):
        """Headers beyond the SDK's `Authorization: Bearer <api_key>`.

        Some gateways authenticate on their own header instead: AMD's APIM
        rejects a bearer-only request with 401 "missing subscription key", so
        --header carries whatever the endpoint needs.
        """
        headers = dict(self.extra_headers)
        if self.user:
            headers.setdefault("user", self.user)
        return headers or None

    def _cache_key(self, row, candidate_ids):
        return _sha256(
            {
                "version": _CACHE_SCHEMA_VERSION,
                "model": self.model,
                "top_k": self.top_k,
                "policy": _sha256(self.policy),
                "shape": _shape_payload(row),
                "candidates": sorted(candidate_ids),
            }
        )

    @staticmethod
    def _validate_ids(ids, allowed_ids, expected_count):
        if (
            not isinstance(ids, list)
            or len(ids) != expected_count
            or any(not isinstance(cid, str) for cid in ids)
            or len(set(ids)) != len(ids)
            or any(cid not in allowed_ids for cid in ids)
        ):
            raise RecommendationError(
                "invalid candidate IDs in model response "
                f"(expected {expected_count} unique IDs from the supplied list)"
            )
        return ids

    def _request_ids(self, *, shape, baseline, descriptors, expected_count):
        if not self.model:
            raise RecommendationError(
                "no model configured; pass --model or set OPENAI_MODEL"
            )
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
        factory = self.client_factory or _default_client_factory
        client = factory(**client_kwargs)
        payload = {
            "task": "select_mxfp4_benchmark_candidates",
            "required_count": expected_count,
            "shape": shape,
            # The baseline is already pinned into the output. Hide its ID so the
            # model cannot spend one of its slots re-selecting it.
            "safety_baseline_features": {
                key: value for key, value in baseline.items() if key != "id"
            },
            "candidates": descriptors,
        }
        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an GPU kernel expert."
                            "Select benchmark candidates, not a presumed final "
                            "winner. Follow the tuning policy below. Return one "
                            "JSON object with exactly one key, candidate_ids. "
                            "candidate_ids must contain the requested number of "
                            "unique IDs from the supplied list. Do not invent "
                            f"IDs.\n\n{self.policy}"
                        ),
                    },
                    {"role": "user", "content": json.dumps(payload, sort_keys=True)},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=_DEFAULT_MAX_COMPLETION_TOKENS,
            )
        except RecommendationError:
            raise
        except Exception as exc:
            raise RecommendationError(f"model request failed: {exc}") from exc
        try:
            parsed = json.loads(response.choices[0].message.content)
        except (AttributeError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RecommendationError("malformed model response") from exc
        if not isinstance(parsed, dict) or set(parsed) != {"candidate_ids"}:
            raise RecommendationError("unexpected model response schema")
        allowed = {d["id"] for d in descriptors}
        return self._validate_ids(parsed["candidate_ids"], allowed, expected_count)

    def select(self, row, candidates):
        """Return (selected_candidates, baseline_candidate, pruned_count, source)."""
        if not candidates:
            raise RecommendationError("no legal candidates for this shape")
        baseline = _select_baseline(row, candidates)
        pruned = _prune(row, candidates, self.max_candidates)
        # Keep the baseline visible to the ranking even if pruning dropped it.
        if all(_candidate_id(c) != _candidate_id(baseline) for c in pruned):
            pruned.append(baseline)

        others = [c for c in pruned if _candidate_id(c) != _candidate_id(baseline)]
        want = max(0, self.top_k - 1)
        if want == 0 or not others:
            return [], baseline, len(pruned), "baseline_only"
        if len(others) <= want:
            return others, baseline, len(pruned), "within_budget"

        descriptors = [_candidate_descriptor(c) for c in others]
        by_id = {d["id"]: c for d, c in zip(descriptors, others)}
        shape = _shape_payload(row)
        baseline_descriptor = _candidate_descriptor(baseline)

        cache_key = self._cache_key(row, list(by_id))
        if self.cache is not None and not self.refresh:
            entry = self.cache.get(cache_key)
            if entry is not None:
                try:
                    ids = self._validate_ids(entry.get("ids"), set(by_id), want)
                except RecommendationError:
                    ids = None
                if ids is not None:
                    return [by_id[i] for i in ids], baseline, len(pruned), "cache"

        # Retry the *same* request on a malformed or out-of-set answer. This is
        # not a fallback -- the model still makes every decision -- but a single
        # bad completion should not discard a whole multi-shape run. Auth and
        # configuration errors are not retried.
        last = None
        for attempt in range(self.retries + 1):
            try:
                ids = self._request_ids(
                    shape=shape,
                    baseline=baseline_descriptor,
                    descriptors=descriptors,
                    expected_count=want,
                )
                break
            except RecommendationError as exc:
                if "OPENAI_API_KEY" in str(exc) or "no model configured" in str(exc):
                    raise
                last = exc
                print(
                    f"[recommend] attempt {attempt + 1}/{self.retries + 1} failed: "
                    f"{exc}",
                    flush=True,
                )
                if attempt < self.retries:
                    # Gateways return transient 5xx under load; retrying with no
                    # pause just burns the budget inside the same bad window.
                    time.sleep(_RETRY_BACKOFF_S * (2**attempt))
        else:
            raise RecommendationError(f"model failed after retries: {last}")
        if self.cache is not None:
            self.cache.put(cache_key, {"ids": ids, "model": self.model})
        return [by_id[i] for i in ids], baseline, len(pruned), "model"


def enumerate_candidates(row):
    """Every legal (GEMM1, GEMM2) pair for a shape, via the tuner's own filter."""
    tuner = Mxfp4FlydslTuner.__new__(Mxfp4FlydslTuner)
    tuner.keys = [k for k in TUNER_KEYS if k in row]
    return tuner._candidate_rows(row)


def _expand_gemm2(row, chosen, legal, per_g1):
    """Give each chosen GEMM1 its `per_g1` best GEMM2 partners.

    A slate is a (GEMM1, GEMM2) product, but a fixed top_k spends nearly all its
    slots on distinct GEMM1s -- measured on kimi-k3, 4 of 10 losing shapes held
    the *right* GEMM1 and lost purely on its GEMM2 partner. Expanding after
    selection keeps the model in charge of which GEMM1s are worth trying while
    making sure each one actually gets searched.
    """
    if per_g1 == 1:
        return chosen
    by_g1 = {}
    for candidate in legal:
        by_g1.setdefault(str(candidate["kernelName1"]), []).append(candidate)
    out, seen = [], set()
    for candidate in chosen:
        partners = by_g1.get(str(candidate["kernelName1"]), [candidate])
        # Spread the partners across GEMM2 shapes instead of taking the top-N by
        # rank. Rank-ordering collapses here exactly as it did for the prompt: at
        # kimi-k3 token 1024 the top-3-by-rank partners all missed the tuned
        # GEMM2, leaving a reproducible 1.5% loss (0.985 over three repeats)
        # even though the right GEMM1 was in the slate.
        cells = {}
        for partner in partners:
            g2 = _g2_features(partner)
            cell = (
                g2.get("tile_n"),
                g2.get("tile_k"),
                g2.get("mode") or ("atomic" if g2.get("atomic", True) else "reduce"),
                bool(g2.get("persist", False)),
                bool(g2.get("b_nt", 0) or g2.get("use_nt", False)),
            )
            cells.setdefault(cell, []).append(partner)
        for cell in cells.values():
            cell.sort(key=lambda c: _rank_key(row, c))
        order = sorted(cells, key=lambda k: _rank_key(row, cells[k][0]))
        # per_g1 == 0 means "every legal partner". For a fixed GEMM1 the layout
        # family offers only ~16, and they are one-per-cell, so stratifying
        # degenerates to ranking and a partial take can still miss the tuned
        # pair. Full coverage is the only guarantee, and it makes exact-pair
        # containment equal to GEMM1 containment.
        limit = len(partners) if per_g1 == 0 else per_g1 + 1
        picked = [candidate]
        depth = 0
        while len(picked) < limit and depth < max(len(v) for v in cells.values()):
            for cell in order:
                if depth < len(cells[cell]):
                    picked.append(cells[cell][depth])
                    if len(picked) >= limit:
                        break
            depth += 1
        for extra in picked:
            key = _candidate_id(extra)
            if key not in seen:
                seen.add(key)
                out.append(extra)
    return out


def plan_shape(row, agent, g2_per_g1=1):
    legal = enumerate_candidates(row)
    unique = _deduplicate_effective_candidates(legal)
    selected, baseline, pruned_count, source = agent.select(row, unique)
    selected = _expand_gemm2(row, selected, unique, g2_per_g1)
    return ShapePlan(
        row=row,
        selected=selected,
        baseline=baseline,
        full_count=len(unique),
        pruned_count=pruned_count,
        source=source,
    )


def plan_to_rows(plan):
    """Baseline first, then the model's picks in rank order."""
    rows = []
    ordered = [(plan.baseline, "baseline")] + [(c, "model") for c in plan.selected]
    for rank, (candidate, source) in enumerate(ordered):
        entry = {key: _json_scalar(plan.row.get(key)) for key in TUNER_KEYS}
        entry.update(
            {
                "rank": rank,
                "block_m": int(candidate["block_m"]),
                "ksplit": int(candidate.get("ksplit", 0) or 0),
                "kernelName1": str(candidate["kernelName1"]),
                "kernelName2": str(candidate["kernelName2"]),
                "candidate_id": _candidate_id(candidate),
                "source": source if source == "baseline" else plan.source,
                "reason": "",
            }
        )
        rows.append(entry)
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_shapes(untuned_file, gfx, cu_num):
    import pandas as pd

    frame = pd.read_csv(untuned_file).drop_duplicates().reset_index(drop=True)
    rows = []
    for _, series in frame.iterrows():
        row = {k: _json_scalar(v) for k, v in series.to_dict().items()}
        # The tuner injects the *runtime* gfx/cu_num before tuning. This script
        # may run on a different host, so they are explicit inputs and the tuner
        # validates them when it loads the CSV.
        row.setdefault("gfx", gfx)
        row.setdefault("cu_num", cu_num)
        row["gfx"] = gfx
        row["cu_num"] = cu_num
        rows.append(row)
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Recommend MXFP4 a4w4 tuning candidates into a CSV."
    )
    parser.add_argument("-i", "--untune_file", required=True, help="untuned shape CSV")
    parser.add_argument("-o", "--out", required=True, help="candidate CSV to write")
    parser.add_argument(
        "--gfx",
        required=True,
        help="target arch the tuner will run on, e.g. gfx950 (tagged into the CSV)",
    )
    parser.add_argument(
        "--cu-num",
        type=_positive_int,
        required=True,
        help="target CU count, e.g. 256 (tagged into the CSV)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", ""),
        help="model name (or OPENAI_MODEL)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", ""),
        help="OpenAI-compatible endpoint (or OPENAI_BASE_URL)",
    )
    parser.add_argument(
        "--user",
        default="",
        help="gateway user/NTID header (or OPENAI_USER/AMD_NTID)",
    )
    parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=_DEFAULT_TOP_K,
        help="candidates per shape, including the pinned baseline",
    )
    parser.add_argument(
        "--max-candidates",
        type=_positive_int,
        default=_DEFAULT_MAX_CANDIDATES,
        help="prompt budget: heuristically prune to this many before asking",
    )
    parser.add_argument(
        "--timeout", type=_positive_float, default=60.0, help="request timeout (s)"
    )
    parser.add_argument(
        "--g2-per-g1",
        dest="g2_per_g1",
        type=int,
        default=1,
        help=(
            "after selection, also benchmark this many GEMM2 partners per chosen "
            "GEMM1; 0 means every legal partner (makes exact-pair containment "
            "equal to GEMM1 containment)"
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="extra attempts when the model returns a malformed/out-of-set answer",
    )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "extra HTTP header, repeatable. Needed for gateways that do not "
            "authenticate on the bearer token, e.g. "
            "--header Ocp-Apim-Subscription-Key=$AMD_LLM_GATEWAY_KEY"
        ),
    )
    parser.add_argument("--policy", default="", help="path to a policy markdown file")
    parser.add_argument("--cache", default=str(_DEFAULT_CACHE_PATH))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--refresh", action="store_true", help="ignore cached recommendations"
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    policy = _DEFAULT_POLICY
    if args.policy:
        policy = Path(args.policy).read_text(encoding="utf-8")

    extra_headers = {}
    for item in args.header:
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            raise RecommendationError(f"--header must be NAME=VALUE, got {item!r}")
        extra_headers[name.strip()] = value

    agent = CandidateAgent(
        model=args.model,
        base_url=args.base_url or None,
        user=args.user or None,
        timeout=args.timeout,
        top_k=args.top_k,
        max_candidates=args.max_candidates,
        policy=policy,
        cache=None if args.no_cache else RecommendationCache(args.cache),
        refresh=args.refresh,
        retries=args.retries,
        extra_headers=extra_headers,
    )

    shapes = _load_shapes(args.untune_file, args.gfx, args.cu_num)
    all_rows = []
    sources = {}
    for row in shapes:
        plan = plan_shape(row, agent, args.g2_per_g1)
        rows = plan_to_rows(plan)
        all_rows.extend(rows)
        sources[plan.source] = sources.get(plan.source, 0) + 1
        print(
            f"[recommend] token={row['token']} inter={row['inter_dim']} "
            f"expert={row['expert']} topk={row['topk']} "
            f"legal={plan.full_count} pruned={plan.pruned_count} "
            f"-> {len(rows)} rows ({plan.source})",
            flush=True,
        )

    # Written only after every shape succeeded: a partial CSV would look like a
    # complete recommendation to the tuner.
    write_csv(args.out, all_rows)
    print(
        f"[recommend] wrote {len(all_rows)} candidate rows for {len(shapes)} shapes "
        f"to {args.out}",
        flush=True,
    )
    print(
        "[recommend] sources: "
        + ", ".join(f"{k}={v}" for k, v in sorted(sources.items())),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except RecommendationError as exc:
        print(f"[recommend] ERROR: {exc}", file=sys.stderr)
        sys.exit(2)
