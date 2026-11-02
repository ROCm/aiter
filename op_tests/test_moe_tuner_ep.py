# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unit tests for the EP-aware paths of the 2-stage fused-MoE tuner.

Covers the four fixes in this PR:
  * build_ep_routing            -- EP geometry for run_config
  * FmoeTuner._ep_rescore_topn  -- two-phase candidate scoring
  * FmoeTuner.calculate         -- non-positive pair time guard
  * test_common.perftest        -- AITER_USE_CUDA_EVENT

Routing tests run on CPU tensors: they check the geometry the tuner hands the
kernel, not the kernel. Run with:
    pytest op_tests/test_moe_tuner_ep.py
"""

import importlib.util
import os
import sys

import pandas as pd
import pytest
import torch

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TUNER_PATH = os.path.join(
    REPO_DIR, "csrc", "ck_gemm_moe_2stages_codegen", "gemm_moe_tune.py"
)


def _load_tuner_module():
    """Import gemm_moe_tune.py by path; it is a script, not an installed module."""
    spec = importlib.util.spec_from_file_location("gemm_moe_tune", TUNER_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["gemm_moe_tune"] = module
    spec.loader.exec_module(module)
    return module


tuner_mod = _load_tuner_module()
build_ep_routing = tuner_mod.build_ep_routing
FmoeTuner = tuner_mod.FmoeTuner

# DeepSeek-V4.1-Flash at DEP8: 384 global experts, 48 per rank, topk 6.
EXPERT, TOPK, EP = 48, 6, 8
GLOBAL_EXPERTS = EXPERT * EP


@pytest.mark.parametrize("token", [512, 1024, 2048])
def test_ep_routing_geometry(token):
    """Shapes, dtypes and the mask convention the kernel relies on."""
    topk_ids, topk_weights, expert_mask, num_local_tokens, recv = build_ep_routing(
        token, TOPK, EXPERT, EP, device="cpu"
    )

    # The buffer keeps the full padded row count; only `recv` of it is live.
    assert topk_ids.shape == (token, TOPK)
    assert topk_weights.shape == (token, TOPK)
    assert topk_ids.dtype == torch.int32
    assert 0 < recv < token
    assert num_local_tokens.tolist() == [recv]

    # vLLM's determine_expert_map convention: 1 over the local globals, 0 for
    # everything remote, plus a trailing 0 for the sentinel slot.
    assert expert_mask.shape == (GLOBAL_EXPERTS + 1,)
    assert expert_mask[:EXPERT].eq(1).all()
    assert expert_mask[EXPERT:].eq(0).all()


@pytest.mark.parametrize("token", [512, 2048])
def test_ep_routing_prefix_invariant(token):
    """num_local_tokens is a prefix length, so live rows must be packed first.

    If a row past `recv` carried a local expert the kernel would silently skip
    its work; if a row before `recv` carried none, the kernel would process a
    row the server never delivered.
    """
    topk_ids, _, _, _, recv = build_ep_routing(token, TOPK, EXPERT, EP, device="cpu")

    live, tail = topk_ids[:recv], topk_ids[recv:]
    assert live.lt(EXPERT).any(dim=1).all(), "a delivered row has no local expert"
    assert tail.eq(GLOBAL_EXPERTS).all(), "padded tail is not all sentinel"


@pytest.mark.parametrize("token", [512, 2048])
def test_ep_routing_ids_are_global(token):
    """Routes span the global expert space, not just this rank's slice.

    Local-only ids would make every row deliverable and put ~EP times too much
    work per expert, which is the dense batch this fix exists to replace.
    """
    topk_ids, _, _, _, recv = build_ep_routing(token, TOPK, EXPERT, EP, device="cpu")

    live = topk_ids[:recv]
    assert live.max().item() < GLOBAL_EXPERTS
    assert live.ge(EXPERT).any(), "no remote routes: ids look local-only"
    # topk picks are distinct experts.
    assert all(len(set(row.tolist())) == TOPK for row in live)


def test_ep_routing_work_is_ep_times_sparser():
    """The point of the fix: pairs per expert drop by ~EP versus dense.

    Dense gives token*topk/expert rows per expert; EP gives that over EP. The
    bound is loose because routing is sampled, not constructed to hit the mean.
    """
    token = 2048
    topk_ids, _, _, _, recv = build_ep_routing(token, TOPK, EXPERT, EP, device="cpu")

    local_pairs = int(topk_ids[:recv].lt(EXPERT).sum())
    dense_pairs = token * TOPK
    assert local_pairs == pytest.approx(dense_pairs / EP, rel=0.1)
    assert local_pairs < dense_pairs / 4


def test_ep_routing_is_deterministic():
    """Same shape must give the same draw, or two runs are not comparable."""
    a = build_ep_routing(512, TOPK, EXPERT, EP, device="cpu")
    b = build_ep_routing(512, TOPK, EXPERT, EP, device="cpu")
    assert a[4] == b[4]
    assert torch.equal(a[0], b[0])

    different = build_ep_routing(512, TOPK, EXPERT, EP, device="cpu", seed=1)
    assert not torch.equal(a[0], different[0])


# -- calculate() guard --------------------------------------------------------

_KEY = (
    "gfx950",
    256,
    2048,
    5120,
    2304,
    EXPERT,
    TOPK,
    tuner_mod.ActivationType.Silu,
    torch.bfloat16,
    torch.float4_e2m1fn_x2,
    torch.float4_e2m1fn_x2,
    tuner_mod.QuantType.per_1x32,
    1,
    0,
)


@pytest.mark.parametrize(
    "us", [0, -0.5, FmoeTuner.INVALID_TIME, FmoeTuner.INF_TIME, float("-inf")]
)
def test_calculate_rejects_nonpositive_us(us):
    """post_process ranks by us1+us2, and INVALID_TIME is -1, so a pair with one
    invalid stage sums to <=0 and used to reach the division as a real time."""
    tuner = FmoeTuner.__new__(FmoeTuner)
    assert tuner.calculate((_KEY, "", "kn1", 64, us, 0.01)) == (0, 0)


def test_calculate_accepts_positive_us():
    tuner = FmoeTuner.__new__(FmoeTuner)
    tflops, bw = tuner.calculate((_KEY, "", "kn1", 64, 150.0, 0.01))
    assert tflops > 0 and bw > 0


# -- two-phase rescoring gate -------------------------------------------------


def _profile_df():
    return pd.DataFrame(
        [
            {
                "kernelName1": "kn1_a",
                "kernelName2": "kn2_a",
                "block_m": 64,
                "us": 150.0,
            },
            {
                "kernelName1": "kn1_b",
                "kernelName2": "kn2_b",
                "block_m": 128,
                "us": 120.0,
            },
        ]
    )


@pytest.mark.parametrize(
    ("ep_size", "topn"),
    [(None, None), ("8", None), (None, "24"), ("1", "24"), ("8", "0")],
)
def test_ep_rescore_is_opt_in(monkeypatch, ep_size, topn):
    """Off unless both env vars ask for it, so non-EP tuning is untouched."""
    monkeypatch.delenv("TUNE_MOE_EP_SIZE", raising=False)
    monkeypatch.delenv("TUNE_MOE_EP_RESCORE_TOPN", raising=False)
    if ep_size is not None:
        monkeypatch.setenv("TUNE_MOE_EP_SIZE", ep_size)
    if topn is not None:
        monkeypatch.setenv("TUNE_MOE_EP_RESCORE_TOPN", topn)

    tuner = FmoeTuner.__new__(FmoeTuner)
    out = tuner._ep_rescore_topn(_profile_df(), _KEY, object())
    assert "_us_ep" not in out.columns


def test_ep_rescore_ranks_on_measured_time(monkeypatch):
    """The EP time decides the winner, and `us` keeps its dense meaning.

    kn2_b wins on dense us (120 < 150) but loses once measured, which is the
    inversion this phase exists to catch.
    """
    monkeypatch.setenv("TUNE_MOE_EP_SIZE", "8")
    monkeypatch.setenv("TUNE_MOE_EP_RESCORE_TOPN", "24")

    tuner = FmoeTuner.__new__(FmoeTuner)
    tuner.columns = ["kernelName1", "kernelName2", "block_m", "us"]
    tuner.untunedf = None
    times = {"kn2_a": 380.0, "kn2_b": 790.0}
    seen = []

    def fake_run_config(args):
        cfg = pd.read_csv(os.environ["AITER_CONFIG_FMOE"])
        kn2 = cfg["kernelName2"].iloc[0]
        seen.append(kn2)
        return [{"status": "ok", "e2e_us": times[kn2]}]

    tuner.run_config = fake_run_config
    tuner._clear_op_caches = lambda: None
    monkeypatch.setattr(
        tuner_mod.FmoeTuner,
        "get_arg_defaults",
        lambda self: {"config_env_name": "AITER_CONFIG_FMOE"},
    )

    out = tuner._ep_rescore_topn(_profile_df(), _KEY, object())

    assert set(seen) == {"kn2_a", "kn2_b"}, "not every candidate was dispatched"
    assert out["_us_ep"].tolist() == [380.0, 790.0]
    assert out["us"].tolist() == [150.0, 120.0], "dense us must not be overwritten"
    winner = out.loc[out["_us_ep"].idxmin()]
    assert winner["kernelName2"] == "kn2_a"
    # Ranking on dense us alone would have picked the loser.
    assert out.loc[out["us"].idxmin()]["kernelName2"] == "kn2_b"


def test_select_best_candidate_prefers_measured_ep_time():
    """The row that post_process writes out is the one that was measured.

    _ep_rescore_topn is only useful if its column actually decides; here the
    dense `us` ranking disagrees, so ranking on it would undo the whole phase.
    """
    df = _profile_df()
    df["_us_ep"] = [380.0, 790.0]

    best, rank_col = tuner_mod.select_best_candidate(df)

    assert rank_col == "_us_ep"
    assert best["kernelName2"] == "kn2_a"
    assert df.loc[df["us"].idxmin()]["kernelName2"] == "kn2_b"


def test_select_best_candidate_falls_back_to_us():
    """Without rescoring the tuner must behave exactly as it always has."""
    best, rank_col = tuner_mod.select_best_candidate(_profile_df())

    assert rank_col == "us"
    assert best["kernelName2"] == "kn2_b"


def test_ep_rescore_restores_config_env(monkeypatch):
    """A leaked AITER_CONFIG_FMOE would silently repoint the rest of the run."""
    monkeypatch.setenv("TUNE_MOE_EP_SIZE", "8")
    monkeypatch.setenv("TUNE_MOE_EP_RESCORE_TOPN", "24")
    monkeypatch.setenv("AITER_CONFIG_FMOE", "/tmp/sentinel.csv")

    tuner = FmoeTuner.__new__(FmoeTuner)
    tuner.columns = ["kernelName1", "kernelName2", "block_m", "us"]
    tuner.untunedf = None
    tuner.run_config = lambda args: [{"status": "ok", "e2e_us": 100.0}]
    tuner._clear_op_caches = lambda: None
    monkeypatch.setattr(
        tuner_mod.FmoeTuner,
        "get_arg_defaults",
        lambda self: {"config_env_name": "AITER_CONFIG_FMOE"},
    )

    tuner._ep_rescore_topn(_profile_df(), _KEY, object())
    assert os.environ["AITER_CONFIG_FMOE"] == "/tmp/sentinel.csv"


# -- cuda.Event timing --------------------------------------------------------


@pytest.mark.parametrize(
    ("env", "expected"), [(None, False), ("0", False), ("1", True)]
)
def test_use_cuda_event_env(monkeypatch, env, expected):
    from aiter.test_common import _default_use_cuda_event

    monkeypatch.delenv("AITER_USE_CUDA_EVENT", raising=False)
    if env is not None:
        monkeypatch.setenv("AITER_USE_CUDA_EVENT", env)
    assert _default_use_cuda_event() is expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_run_perftest_times_with_cuda_event(monkeypatch):
    """The regression this guards: candidates scoring 0 us.

    Inside forked mp_tuner workers roctracer is dead and the profiler returns
    no CUDA events, so run_perftest reported 0 for every candidate and the run
    finished with "no valida data after post process".
    """
    from aiter.test_common import run_perftest

    monkeypatch.setenv("AITER_USE_CUDA_EVENT", "1")
    x = torch.randn(1024, 1024, device="cuda")
    _, us = run_perftest(torch.matmul, x, x, num_iters=20, num_warmup=2)
    assert us > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_explicit_use_cuda_event_overrides_env(monkeypatch):
    """Callers that pass the flag keep deciding for themselves."""
    from aiter.test_common import run_perftest

    monkeypatch.setenv("AITER_USE_CUDA_EVENT", "0")
    x = torch.randn(512, 512, device="cuda")
    _, us = run_perftest(
        torch.matmul, x, x, num_iters=20, num_warmup=2, use_cuda_event=True
    )
    assert us > 0
