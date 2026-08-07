# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from csrc.ck_gemm_moe_2stages_codegen import (
    tune_mxfp4_flydsl_openai as openai_tuner,
)
from csrc.ck_gemm_moe_2stages_codegen.gemm_moe_tune import Mxfp4FlydslTuner
from csrc.ck_gemm_moe_2stages_codegen.tune_mxfp4_flydsl_openai import (
    KEY_COLUMNS,
    RESULT_COLUMNS,
    OpenAICandidateSelector,
    OpenAIMxfp4FlydslTuner,
    RecommendationCache,
    _candidate_descriptor,
    _candidate_id,
    _deduplicate_effective_candidates,
    _install_modern_gemm2_parser,
    _openai_mxfp4_shape_worker,
    _select_baseline,
)


def _shape(token=32, block_m=64):
    return {
        "gfx": "gfx950",
        "cu_num": 256,
        "token": token,
        "model_dim": 6144,
        "inter_dim": 512,
        "expert": 256,
        "topk": 8,
        "act_type": "ActivationType.Silu",
        "dtype": "torch.bfloat16",
        "q_dtype_a": "torch.float4_e2m1fn_x2",
        "q_dtype_w": "torch.float4_e2m1fn_x2",
        "q_type": "QuantType.per_1x32",
        "use_g1u1": 1,
        "doweight_stage1": 0,
        "block_m": block_m,
        "kernelName2": (f"flydsl_mxmoe_g2_a4w4_{block_m}x256x256_atomic"),
        "run_1stage": 0,
    }


def _base_tuner():
    tuner = Mxfp4FlydslTuner.__new__(Mxfp4FlydslTuner)
    tuner.keys = KEY_COLUMNS
    return tuner


def _candidate(row, *, bn, use_nt, xcd=0, interleave=False):
    tuner = _base_tuner()
    bm = int(row["block_m"])
    name = tuner._g1_kname(
        bm,
        bn,
        use_nt,
        False,
        xcd,
        interleave=interleave,
    )
    return tuner._candidate_row(row, bm, name, row["kernelName2"])


def _candidates(row):
    return [
        _candidate(row, bn=128, use_nt=False),
        _candidate(row, bn=128, use_nt=True),
        _candidate(row, bn=256, use_nt=False),
        _candidate(row, bn=256, use_nt=True),
        _candidate(row, bn=128, use_nt=False, xcd=2),
        _candidate(row, bn=256, use_nt=False, xcd=4),
    ]


class _FakeClientFactory:
    def __init__(self, content):
        self.content = content
        self.client_kwargs = []
        self.requests = []

    def __call__(self, **kwargs):
        self.client_kwargs.append(kwargs)
        factory = self

        class Completions:
            @staticmethod
            def create(**request):
                factory.requests.append(request)
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=factory.content)
                        )
                    ]
                )

        return SimpleNamespace(chat=SimpleNamespace(completions=Completions()))


def _selector(
    tmp_path,
    factory,
    *,
    api_key="secret-key",
    top_k=3,
    max_candidates=256,
    refresh=False,
):
    policy = tmp_path / "policy.md"
    if not policy.exists():
        policy.write_text("Prefer legal, diverse MXFP4 candidates.", encoding="utf-8")
    return OpenAICandidateSelector(
        model="test-model",
        base_url="https://example.invalid/v1",
        api_key=api_key,
        top_k=top_k,
        timeout=5,
        max_candidates=max_candidates,
        cache=RecommendationCache(tmp_path / "recommendations.json"),
        policy_path=policy,
        refresh=refresh,
        client_factory=factory,
    )


def _recommended_model_ids(row, candidates, top_k=3):
    baseline_id = _candidate_id(_select_baseline(row, candidates))
    pool_ids = [
        _candidate_id(candidate)
        for candidate in candidates
        if _candidate_id(candidate) != baseline_id
    ]
    return pool_ids[: top_k - 1]


def test_effective_use_nt_candidates_are_deduplicated():
    row = _shape(token=1024, block_m=32)
    cached = _candidate(row, bn=256, use_nt=False)
    non_temporal = _candidate(row, bn=256, use_nt=True)

    candidates = _deduplicate_effective_candidates(row, [non_temporal, cached])

    assert len(candidates) == 1
    assert candidates[0]["kernelName1"] == cached["kernelName1"]


def test_safety_baseline_preserves_baseline_interleave_layout():
    row = _shape()
    separated = _candidate(row, bn=128, use_nt=False)
    interleaved = _candidate(row, bn=128, use_nt=False, interleave=True)
    row["_baseline_kernelName1"] = interleaved["kernelName1"]

    selected = _select_baseline(row, [separated, interleaved])

    assert selected["kernelName1"] == interleaved["kernelName1"]


def test_valid_recommendation_uses_opaque_ids_and_cache(tmp_path):
    row = _shape()
    candidates = _candidates(row)
    model_ids = _recommended_model_ids(row, candidates)
    factory = _FakeClientFactory(json.dumps({"candidate_ids": model_ids}))
    selector = _selector(tmp_path, factory)

    result = selector.select(row, candidates)

    assert result.source == "api"
    assert len(result.candidates) == 3
    assert len(factory.requests) == 1
    request_text = json.dumps(factory.requests[0], sort_keys=True)
    assert "secret-key" not in request_text
    assert candidates[0]["kernelName1"] not in request_text
    assert candidates[0]["kernelName2"] not in request_text

    user_payload = json.loads(factory.requests[0]["messages"][1]["content"])
    assert set(user_payload) == {
        "task",
        "required_count",
        "shape",
        "safety_baseline_features",
        "candidates",
    }
    assert "id" not in user_payload["safety_baseline_features"]
    assert all(
        set(candidate) == {"id", "gemm1", "gemm2"}
        for candidate in user_payload["candidates"]
    )

    cache_text = (tmp_path / "recommendations.json").read_text(encoding="utf-8")
    assert "secret-key" not in cache_text
    assert candidates[0]["kernelName1"] not in cache_text
    assert candidates[0]["kernelName2"] not in cache_text

    def fail_if_called(**_kwargs):
        raise AssertionError("cache hit must not construct an API client")

    cached_selector = _selector(tmp_path, fail_if_called)
    cached_result = cached_selector.select(row, candidates)
    assert cached_result.source == "cache"
    assert [_candidate_id(candidate) for candidate in cached_result.candidates] == [
        _candidate_id(candidate) for candidate in result.candidates
    ]


def test_policy_change_invalidates_cache(tmp_path):
    row = _shape()
    candidates = _candidates(row)
    model_ids = _recommended_model_ids(row, candidates)
    first_factory = _FakeClientFactory(json.dumps({"candidate_ids": model_ids}))
    _selector(tmp_path, first_factory).select(row, candidates)

    (tmp_path / "policy.md").write_text(
        "Updated MXFP4 tuning policy.", encoding="utf-8"
    )
    second_factory = _FakeClientFactory(json.dumps({"candidate_ids": model_ids}))
    result = _selector(tmp_path, second_factory).select(row, candidates)

    assert result.source == "api"
    assert len(second_factory.requests) == 1


@pytest.mark.parametrize(
    "content",
    [
        "not json",
        json.dumps({"candidate_ids": ["unknown", "unknown"]}),
        json.dumps({"candidate_ids": ["unknown"]}),
        json.dumps({"candidate_ids": [], "extra": True}),
    ],
)
def test_invalid_model_response_falls_back_to_full_search(tmp_path, content):
    row = _shape()
    candidates = _candidates(row)
    result = _selector(tmp_path, _FakeClientFactory(content)).select(row, candidates)

    assert result.source.endswith("_fallback")
    assert len(result.candidates) == len(candidates)


def test_missing_api_key_falls_back_without_constructing_client(tmp_path):
    row = _shape()
    candidates = _candidates(row)

    def fail_if_called(**_kwargs):
        raise AssertionError("missing key must not construct a client")

    result = _selector(tmp_path, fail_if_called, api_key="").select(row, candidates)

    assert result.source == "RecommendationError_fallback"
    assert len(result.candidates) == len(candidates)


def test_corrupt_cache_falls_back_without_api_call(tmp_path):
    row = _shape()
    candidates = _candidates(row)
    (tmp_path / "recommendations.json").write_text("{", encoding="utf-8")

    def fail_if_called(**_kwargs):
        raise AssertionError("corrupt cache must fail safely before API use")

    result = _selector(tmp_path, fail_if_called).select(row, candidates)

    assert result.source == "RecommendationError_fallback"
    assert len(result.candidates) == len(candidates)


def test_cache_write_failure_does_not_retain_recommendation(tmp_path, monkeypatch):
    row = _shape()
    candidates = _candidates(row)
    model_ids = _recommended_model_ids(row, candidates)
    factory = _FakeClientFactory(json.dumps({"candidate_ids": model_ids}))
    selector = _selector(tmp_path, factory)

    def fail_replace(*_args):
        raise OSError("read-only cache")

    monkeypatch.setattr(openai_tuner.os, "replace", fail_replace)
    first = selector.select(row, candidates)
    second = selector.select(row, candidates)

    assert first.source == second.source == "RecommendationError_fallback"
    assert len(first.candidates) == len(second.candidates) == len(candidates)
    assert len(factory.requests) == 2


def test_sdk_error_falls_back_to_full_search(tmp_path):
    row = _shape()
    candidates = _candidates(row)

    def timeout_factory(**_kwargs):
        raise TimeoutError("API timeout")

    result = _selector(tmp_path, timeout_factory).select(row, candidates)

    assert result.source == "TimeoutError_fallback"
    assert len(result.candidates) == len(candidates)


def test_candidate_limit_and_budget_skip_api(tmp_path):
    row = _shape()
    candidates = _candidates(row)

    def fail_if_called(**_kwargs):
        raise AssertionError("API call was not expected")

    limited = _selector(tmp_path, fail_if_called, max_candidates=5).select(
        row, candidates
    )
    assert limited.source == "candidate_limit_fallback"

    within_budget = _selector(tmp_path, fail_if_called, top_k=len(candidates)).select(
        row, candidates
    )
    assert within_budget.source == "within_budget"


def test_cli_overrides_environment_and_client_is_lazy(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "environment-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://environment.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "environment-key")
    tuner = OpenAIMxfp4FlydslTuner(
        "test",
        KEY_COLUMNS,
        RESULT_COLUMNS,
        "test",
    )
    args = tuner.parser.parse_args(
        [
            "--openai-model",
            "cli-model",
            "--openai-base-url",
            "https://cli.invalid/v1",
            "--openai-cache",
            str(tmp_path / "cache.json"),
        ]
    )

    selector = tuner._make_selector(args)

    assert selector.model == "cli-model"
    assert selector.base_url == "https://cli.invalid/v1"
    assert selector.api_key == "environment-key"
    assert selector.cache.path == tmp_path / "cache.json"


def test_amd_gateway_uses_subscription_header(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("AMD_LLM_GATEWAY_KEY", "gateway-key")
    tuner = OpenAIMxfp4FlydslTuner(
        "test",
        KEY_COLUMNS,
        RESULT_COLUMNS,
        "test",
    )
    args = tuner.parser.parse_args(
        [
            "--openai-model",
            "GPT-5.6-sol",
            "--openai-base-url",
            "https://llm-api.amd.com/Unified/v1",
            "--openai-user",
            "sixifang",
            "--openai-cache",
            str(tmp_path / "cache.json"),
        ]
    )

    selector = tuner._make_selector(args)

    assert selector.api_key == "gateway-key"
    assert selector.default_headers == {
        "Ocp-Apim-Subscription-Key": "gateway-key",
        "user": "sixifang",
    }


def test_raw_input_can_inherit_locked_baseline_columns(tmp_path):
    raw_path = tmp_path / "untuned.csv"
    baseline_path = tmp_path / "baseline.csv"
    row = _shape()
    pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key
                not in (
                    "gfx",
                    "cu_num",
                    "block_m",
                    "kernelName2",
                    "run_1stage",
                )
            }
        ]
    ).to_csv(raw_path, index=False)
    pd.DataFrame(
        [
            {
                **{key: row[key] for key in KEY_COLUMNS},
                "block_m": row["block_m"],
                "kernelName1": _candidate(row, bn=256, use_nt=False)["kernelName1"],
                "kernelName2": row["kernelName2"],
                "run_1stage": 0,
                "us2": 12.5,
                "err2": "1.2500%",
                "_tag": "baseline-only",
            }
        ]
    ).to_csv(baseline_path, index=False)

    tuner = OpenAIMxfp4FlydslTuner.__new__(OpenAIMxfp4FlydslTuner)
    tuner.keys = KEY_COLUMNS
    tuner._openai_baseline_config = str(baseline_path)
    tuner.get_gfx = lambda: "gfx950"
    tuner.get_cu_num = lambda: 256

    merged = tuner.get_untuned_gemm_list(str(raw_path))

    assert len(merged) == 1
    assert merged.loc[0, "block_m"] == row["block_m"]
    assert merged.loc[0, "kernelName2"] == row["kernelName2"]
    assert merged.loc[0, "run_1stage"] == 0
    assert merged.loc[0, "_baseline_us2"] == 12.5
    assert merged.loc[0, "_baseline_err2"] == "1.2500%"
    assert merged.loc[0, "_source_index"] == 0
    assert "_tag" not in merged.columns


def test_openai_tuner_installs_registered_gemm2_parser():
    from aiter import fused_moe
    from csrc.ck_gemm_moe_2stages_codegen import gemm_moe_tune

    name = "flydsl_moe2_afp4_wfp4_bf16_" "t64x128x256_reduce_bnt2_persist"

    _install_modern_gemm2_parser()
    parsed = fused_moe.parse_g2_kname_any(name)
    parsed_v2 = fused_moe.parse_flydsl_v2_gemm2_kernel(name)

    assert parsed["v2"]
    assert parsed["BM"] == 64
    assert not parsed["atomic"]
    assert parsed["use_nt"]
    assert parsed_v2["tile_m"] == 64
    assert parsed_v2["tile_n"] == 128
    assert parsed_v2["tile_k"] == 256
    assert parsed_v2["epilog"] == "reduce"
    assert parsed_v2["persist"]
    assert gemm_moe_tune.parse_g2_kname_any is fused_moe.parse_g2_kname_any


def test_all_selected_failures_retry_unselected_candidates(monkeypatch):
    row = _shape()
    full = _candidates(row)[:2]
    selected = full[:1]
    calls = []

    def fake_tune_one_shape(self, _row, _args):
        active = self._candidate_rows(_row)
        calls.append([_candidate_id(candidate) for candidate in active])
        if len(calls) == 1:
            return {"us": self.INVALID_TIME}, []
        best = dict(active[0])
        best["us"] = 5.0
        return best, [{"kernelName1": best["kernelName1"]}]

    monkeypatch.setattr(Mxfp4FlydslTuner, "_tune_one_shape", fake_tune_one_shape)
    tuner = OpenAIMxfp4FlydslTuner.__new__(OpenAIMxfp4FlydslTuner)
    tuner.keys = KEY_COLUMNS

    best, profiles = tuner._tune_preselected_shape(
        row, SimpleNamespace(), selected, full
    )

    assert calls == [
        [_candidate_id(selected[0])],
        [_candidate_id(full[1])],
    ]
    assert best["kernelName1"] == full[1]["kernelName1"]
    assert len(profiles) == 1


def test_worker_receives_preselected_candidates_without_api(
    monkeypatch,
):
    row = _shape()
    full = _candidates(row)[:2]
    selected = full[:1]
    observed = {}

    class Queue:
        def __init__(self):
            self.returned = []

        @staticmethod
        def get():
            return 3

        def put(self, value):
            self.returned.append(value)

    queue = Queue()
    monkeypatch.setattr(
        "csrc.ck_gemm_moe_2stages_codegen."
        "tune_mxfp4_flydsl_openai.torch.cuda.set_device",
        lambda gpu: observed.setdefault("gpu", gpu),
    )

    def fake_run(self, worker_row, _args, worker_selected, worker_full):
        observed["row"] = worker_row
        observed["selected"] = worker_selected
        observed["full"] = worker_full
        return worker_selected[0], []

    monkeypatch.setattr(OpenAIMxfp4FlydslTuner, "_run_shape_safely", fake_run)

    best, profiles = _openai_mxfp4_shape_worker(
        (
            KEY_COLUMNS,
            row,
            SimpleNamespace(),
            selected,
            full,
            queue,
        )
    )

    assert observed == {
        "gpu": 3,
        "row": row,
        "selected": selected,
        "full": full,
    }
    assert best == selected[0]
    assert profiles == []
    assert queue.returned == [3]


def test_worker_gpu_failure_returns_failed_shape_and_releases_gpu(monkeypatch):
    row = _shape()
    full = _candidates(row)[:2]

    class Queue:
        def __init__(self):
            self.returned = []

        @staticmethod
        def get():
            return 2

        def put(self, value):
            self.returned.append(value)

    queue = Queue()

    def fail_set_device(_gpu):
        raise RuntimeError("GPU unavailable")

    monkeypatch.setattr(openai_tuner.torch.cuda, "set_device", fail_set_device)
    best, profiles = _openai_mxfp4_shape_worker(
        (
            KEY_COLUMNS,
            row,
            SimpleNamespace(),
            full[:1],
            full,
            queue,
        )
    )

    assert best["us"] == Mxfp4FlydslTuner.INVALID_TIME
    assert best["kernelName1"].startswith("FAILED: RuntimeError")
    assert profiles == []
    assert queue.returned == [2]


def test_descriptor_contains_features_but_not_kernel_names():
    row = _shape()
    candidate = _candidates(row)[0]

    descriptor = _candidate_descriptor(row, candidate)
    serialized = json.dumps(descriptor, sort_keys=True)

    assert set(descriptor) == {"id", "gemm1", "gemm2"}
    assert candidate["kernelName1"] not in serialized
    assert candidate["kernelName2"] not in serialized
