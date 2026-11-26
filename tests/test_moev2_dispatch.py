# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""aiter.moev2 dispatch pieces on the CPU.

Per-key equality with the legacy code over the whole snapshot runs in a subprocess
(tests/moe_dispatch/key_checks.py, differential.py); the rest are unit tests.

NOTE: part of the aiter.moev2 migration (under active development, not graduated).
Please coordinate changes with the moev2 owners (@qli88). See aiter/moev2/README.md.

Golden snapshots: after an intended change to legacy MoE dispatch, run
`python tests/moe_dispatch/snapshot.py --update` and paste the changes it prints into
the PR.
"""

import importlib
import inspect
import logging
import pathlib
import subprocess
import sys
import types

import pytest
import torch

import aiter.fused_moe as legacy
from aiter import ActivationType, QuantType, dtypes
from aiter.moev2 import api
from aiter.moev2.dispatch import (
    buf,
    kernel_names,
    ops,
    overrides,
    plan,
    resolver,
    tuned,
)
from aiter.moev2.problem import (
    FEATURES,
    ActSpec,
    FrozenMap,
    MoeProblem,
    ScaleSpec,
    Shape,
    WeightLayout,
    scale_spec,
)
from aiter.moev2.segments.base import Cap, Params, Segment
from aiter.presorted_moe import register_presorted_moe

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_HERE))


def _subprocess_check(script):
    proc = subprocess.run(
        [sys.executable, str(_HERE / "moe_dispatch" / script), "--check"],
        cwd=_REPO,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout[-6000:] + proc.stderr[-3000:]


@pytest.fixture(autouse=True)
def _gfx950(request):
    """Dispatch reads the arch, CU count and the arch's fp8; simulate an MI355."""
    from moe_dispatch import arch_sim

    if "live_chip_info" in request.fixturenames:
        yield  # that fixture's test is about the real chip-info helpers
        return
    with arch_sim.simulated_arch("gfx950"):
        yield


@pytest.fixture
def live_chip_info():
    """The real chip-info module, out of the arch simulation, with its caches reset.

    `simulated_arch` replaces `get_gfx` and `get_gfx_runtime` with plain functions, so a
    test about which of the two dispatch calls cannot run inside it.
    """
    from aiter.jit.utils import chip_info

    def clear():
        # A test may have replaced one of these; whatever stands in needs no clearing.
        for name in ("get_gfx", "get_gfx_custom_op_core"):
            cache_clear = getattr(getattr(chip_info, name), "cache_clear", None)
            if cache_clear is not None:
                cache_clear()

    clear()
    try:
        yield chip_info
    finally:
        clear()


def test_problems_and_tuned_lookup_match_legacy_for_every_key():
    _subprocess_check("key_checks.py")


def test_feature_variants_are_legacy_calls_with_one_change():
    from moe_dispatch import keys

    combo = (ActivationType.Silu, dtypes.bf16, dtypes.fp8, dtypes.fp8)
    combo += (QuantType.per_Tensor, True, False, (0, 0))
    base = keys._make_key("grid", "base", (64, 7168, 256, 256, 8), combo)
    variants = keys.feature_keys([base], dtypes.fp8)
    assert [k.variant for k in variants] == list(keys._feature_variants(dtypes.fp8))
    signature = inspect.signature(legacy.get_2stage_cfgs.__wrapped__)
    for key in variants:
        signature.bind(*key.args, **key.call_kwargs())
        assert key.args != base.args or key.kwargs != base.kwargs, key.variant
    assert len({k.label() for k in variants}) == len(variants)


def _key(token=64, quant=QuantType.per_Tensor, fp8=None, **kwargs):
    fp8 = dtypes.fp8 if fp8 is None else fp8
    args = (token, 7168, 256, 256, 8, dtypes.bf16, fp8, fp8, quant, True)
    args += (ActivationType.Silu, False, 0, 0)
    return args, kwargs


def _problem(arch="gfx950", cu_num=256, **kwargs):
    args, kw = _key(**kwargs)
    return MoeProblem.from_legacy_key(*args, **kw, arch=arch, cu_num=cu_num)


# 2.1 problem


def test_problem_round_trips_through_the_legacy_key():
    for kwargs in (
        {},
        {"is_ep": True, "has_activation_scales": True},
        {"has_activation_scales": True, "_has_input_activation_scales": True},
        {"_exclude_mxmoe": True},
        {"is_shuffled": False},
        {"is_shuffled": True, "opus_weights_shuffled": False},
        {"config_file": "/tmp/t.csv", "_disable_inline_sort": True},
        {"q_dtype_a2": dtypes.fp8, "input_dtype": dtypes.bf16, "swiglu_limit": 7.0},
        {"has_global_scale": True},
    ):
        p = _problem(**kwargs)
        args, kw = p.legacy_key()
        again = MoeProblem.from_legacy_key(*args, **kw, arch="gfx950", cu_num=256)
        assert again == p and hash(again) == hash(p), kwargs


def test_problem_from_legacy_key_takes_the_legacy_parameters():
    legacy_names = list(
        inspect.signature(legacy.get_2stage_cfgs.__wrapped__).parameters
    )
    ours = [
        n
        for n, prm in inspect.signature(MoeProblem.from_legacy_key).parameters.items()
        if prm.kind is not prm.KEYWORD_ONLY
    ]
    assert ours == legacy_names


def test_problem_rejects_unknown_features_and_inconsistent_shuffle():
    with pytest.raises(ValueError, match="unknown MoeProblem features"):
        _problem().with_features(add={"telepathy"})
    with pytest.raises(ValueError, match="is_shuffled"):
        _problem(is_shuffled=False, opus_weights_shuffled=True)
    assert "telepathy" not in FEATURES


def test_problem_derived_predicates():
    p = _problem(is_ep=True, is_shuffled=True, opus_weights_shuffled=False)
    assert p.is_ep and p.any_shuffled and not p.fully_shuffled
    assert p.quant_type == QuantType.per_Tensor and p.is_g1u1
    assert p.rows_per_expert == 64 * 8 // 256
    assert p.layout.shuffle == "mixed"
    assert _problem(token=1024).shape.m_bucket == 1024


def test_a16w4_situv2_predicates_match_the_legacy_expressions():
    for out in (dtypes.bf16, dtypes.fp16, dtypes.fp32):
        for shuffled in (True, False):
            for doweight in (True, False):
                args = (32, 3072, 3072, 128, 4, out, dtypes.bf16, dtypes.fp4x2)
                args += (
                    QuantType.per_1x32,
                    True,
                    ActivationType.Situv2,
                    doweight,
                    0,
                    0,
                )
                p = MoeProblem.from_legacy_key(
                    *args, is_shuffled=shuffled, arch="gfx950", cu_num=256
                )
                legacy_format = (
                    p.quant_type == QuantType.per_1x32
                    and p.w1.dtype == dtypes.fp4x2
                    and p.act_in.dtype == dtypes.bf16
                    and p.activation.activation == ActivationType.Situv2
                )
                assert p.is_a16w4_situv2 == legacy_format
                assert p.a16w4_situv2_kernel_ok == (
                    legacy_format
                    and out in (dtypes.bf16, dtypes.fp16)
                    and shuffled
                    and not doweight
                )


def test_scale_spec_keeps_the_group_of_unquantized_dtypes():
    assert scale_spec(QuantType.per_1x32, dtypes.bf16).group != ()
    assert scale_spec(QuantType.per_1x32, dtypes.fp4x2).scale_dtype == dtypes.fp8_e8m0
    assert scale_spec(QuantType.No, dtypes.bf16).scale_dtype is None
    assert (
        not ScaleSpec(dtypes.bf16).is_quantized and ScaleSpec(dtypes.fp8).is_quantized
    )


def test_problem_str_is_readable():
    text = str(_problem(is_ep=True))
    assert "M=64" in text and "silu" in text and "features=ep_mask" in text


def test_build_problem_from_a_custom_op_call():
    hidden = torch.empty(3, 64, dtype=dtypes.bf16)
    w1 = torch.empty(4, 32, 64, dtype=dtypes.fp8)
    w2 = torch.empty(4, 64, 16, dtype=dtypes.fp8)
    w1.is_shuffled = w2.is_shuffled = True
    call = {
        name: prm.default
        for name, prm in inspect.signature(legacy.fused_moe).parameters.items()
    }
    call.update(
        hidden_states=hidden,
        w1=w1,
        w2=w2,
        topk_weight=torch.empty(3, 2),
        topk_ids=torch.zeros(3, 2, dtype=torch.int32),
        quant_type=QuantType.per_Tensor,
    )
    p = api.build_problem(api.op_call(call), arch="gfx950", cu_num=256)
    # Legacy keys the dispatch on the padded token count.
    assert (p.shape.m, p.shape.model_dim, p.shape.inter_dim) == (
        legacy.get_padded_M(3),
        64,
        16,
    )
    assert p.shape.experts == 4 and p.shape.topk == 2 and p.fully_shuffled
    assert p.quant_type == QuantType.per_Tensor and p.out_dtype == dtypes.bf16


# 2.2 ops, buffers, plans, capabilities


def test_op_chain_and_buffers():
    assert ops.chain(_problem())[0] is ops.Op.SORT
    assert ops.Op.TAIL not in ops.chain(_problem())
    assert str(buf.Buf("inter", ScaleSpec(dtypes.fp8)))
    with pytest.raises(ValueError):
        buf.Buf("bogus", ScaleSpec(dtypes.fp8))


def test_params_and_frozen_map():
    p = Params(kernel="k", ksplit=0)
    assert p.kernel == "k" and p.replace(ksplit=2).ksplit == 2 and p.ksplit == 0
    assert hash(p) == hash(Params({"ksplit": 0, "kernel": "k"}))
    with pytest.raises(AttributeError):
        _ = p.missing
    with pytest.raises(TypeError):
        p["kernel"] = "x"
    assert FrozenMap({"a": 1}) == FrozenMap({"a": 1})


def test_cap_rejects_with_a_readable_reason():
    fp8 = frozenset({scale_spec(QuantType.per_Tensor, dtypes.fp8)})
    cap = Cap(
        arch=frozenset({"gfx950"}),
        act_in=fp8,
        w1=fp8,
        w2=fp8,
        activation=frozenset({"silu"}),
        features=frozenset({"ep_mask"}),
    )
    assert cap.reject_reason(_problem()) is None
    assert cap.reject_reason(_problem(is_ep=True)) is None
    assert "arch gfx942" in cap.reject_reason(_problem(arch="gfx942"))
    assert "feature bias1" in cap.reject_reason(_problem(has_stage1_bias=True))
    assert "padding" in cap.reject_reason(
        _problem().__class__.from_legacy_key(
            *(_key()[0][:12] + (128, 0)), arch="gfx950", cu_num=256
        )
    )
    assert "extras tuned_table" in cap.reject_reason(_problem(config_file="/t.csv"))
    assert "act_in float8_e4m3fn/per_token not in" in cap.reject_reason(
        _problem(quant=QuantType.per_Token)
    )


def test_segment_without_capabilities_accepts_nothing():
    class Bare(Segment):
        name = "bare"

    assert not Bare().accepts(_problem())
    assert Bare().lossy_values == FrozenMap()


def test_plan_describes_its_structure():
    from aiter.moev2.migration import a8w8_pt

    plans, rejected = resolver.feasible_plans(_problem(), a8w8_pt.SEGMENTS)
    names = sorted(" | ".join(s.name for s in structure) for structure in plans)
    assert names == [
        "impl.whole_graph",
        "sort | quant.a1 | asm.stage1 | quant.inter | ck.gemm2",
        "sort | quant.a1 | ck.gemm1 | quant.inter | ck.gemm2",
    ]
    assert not rejected
    res = resolver.resolve(_problem(), a8w8_pt.SEGMENTS)
    assert res.plan.source in plan.SOURCES
    assert res.plan.structure() and "ck.gemm1" in res.plan.describe()


# 2.3 kernel names


def _csv_kernel_names():
    import pandas as pd

    names = set()
    configs = _REPO / "aiter" / "configs"
    for csv in sorted(configs.rglob("*fmoe*.csv")):
        if "untuned" in csv.name:
            continue
        df = pd.read_csv(csv)
        for column in ("kernelName1", "kernelName2"):
            if column in df.columns:
                names |= {n for n in df[column] if not kernel_names.is_empty(n)}
    return sorted(names)


def test_every_shipped_kernel_name_parses_and_round_trips():
    names = _csv_kernel_names()
    assert len(names) > 100
    for name in names:
        parsed = kernel_names.parse(name)
        assert parsed is not None and parsed.family in kernel_names.FAMILIES, name
        assert parsed.format() == name


def test_kernel_name_edge_cases():
    assert kernel_names.parse("") is None and kernel_names.parse(float("nan")) is None
    with pytest.raises(kernel_names.KernelNameError):
        kernel_names.parse("totally_new_kernel_family_x")
    assert kernel_names.try_parse("totally_new_kernel_family_x") is None
    # One shipped ASM name declares a length one more than its symbol.
    odd = "_ZN5aiter47fmoe_stage1_bf16_pertokenFp8_g1u1_16x128_4tg_pf2E"
    assert kernel_names.parse(odd).format() == odd


# 2.4 / 2.5 resolver, overrides, explain


def test_overrides_read_the_environment_at_call_time(monkeypatch):
    monkeypatch.setenv("AITER_BYPASS_TUNE_CONFIG", "1")
    monkeypatch.setenv("AITER_KSPLIT", "-1")
    monkeypatch.setenv("AITER_USE_NT", "0")
    monkeypatch.setenv("AITER_ONLINE_TUNE", "1")
    assert overrides.bypass_tuned() and overrides.ksplit_disabled()
    assert overrides.nt_override() == 0 and overrides.online_tune()
    monkeypatch.delenv("AITER_BYPASS_TUNE_CONFIG")
    monkeypatch.delenv("AITER_KSPLIT")
    monkeypatch.delenv("AITER_USE_NT")
    monkeypatch.delenv("AITER_ONLINE_TUNE")
    assert not overrides.bypass_tuned() and not overrides.ksplit_disabled()
    assert overrides.nt_override() == -1 and not overrides.online_tune()


def _resolve_with_row(monkeypatch, row):
    from aiter.moev2.migration import a8w8_pt

    monkeypatch.setattr(tuned, "lookup", lambda problem: row)
    return resolver.resolve(_problem(), a8w8_pt.SEGMENTS)


@pytest.mark.parametrize(
    "row,reason",
    [
        ({"run_1stage": 1, "kernelName1": "", "kernelName2": ""}, "1-stage"),
        ({"flat": 3, "kernelName1": "", "kernelName2": ""}, "post-reduction"),
        (
            {
                "kernelName1": "flydsl_moe1_afp8_wfp8_bf16_t32x128x128",
                "kernelName2": "",
            },
            "no segments",
        ),
        ({"kernelName1": " moe_ck2stages_gemm1_x", "kernelName2": ""}, "blanks"),
        ({"kernelName1": "brand_new_kernel", "kernelName2": ""}, "unknown kernel name"),
        ({"kernelName1": 3.0, "kernelName2": ""}, "not a string"),
    ],
)
def test_rows_of_unported_kernels_are_declined_silently(
    monkeypatch, caplog, row, reason
):
    # caplog listens on the root logger; aiter's logger does not propagate.
    monkeypatch.setattr(logging.getLogger("aiter"), "propagate", True)
    res = _resolve_with_row(monkeypatch, row)
    assert res.plan is None and reason in res.declined
    with caplog.at_level(logging.INFO, logger="aiter"):
        res.emit()
    assert not res.logs and caplog.text == ""


def test_a_declined_call_runs_on_legacy(monkeypatch):
    from aiter.moev2 import paths
    from aiter.moev2.migration import a8w8_pt

    prepares = []

    def prologue(call, remap, *, prepare=True):
        prepares.append(prepare)
        return {"hidden_states": torch.empty(2, 64)}

    monkeypatch.setattr(a8w8_pt, "_dispatch", lambda *a, **k: None)
    monkeypatch.setattr(a8w8_pt.executor, "prologue", prologue)
    monkeypatch.setattr(api, "dispatch_key", lambda c: ((), {}, None))
    assert a8w8_pt.run({}) is paths.DECLINED
    assert prepares == [False]


def test_explain_lists_problem_structures_and_choice(monkeypatch, caplog):
    from aiter.moev2.migration import a8w8_pt

    # caplog listens on the root logger; aiter's logger does not propagate.
    monkeypatch.setattr(logging.getLogger("aiter"), "propagate", True)
    monkeypatch.setenv("AITER_MOE_EXPLAIN", "1")
    monkeypatch.setattr(tuned, "lookup", lambda problem: None)
    res = resolver.resolve(_problem(), a8w8_pt.SEGMENTS)
    text = "\n".join(res.explain)
    assert "problem: gfx950 M=64" in text and "feasible structures (3)" in text
    assert "tuned lookup: miss" in text and "(fp8_per_tensor[otherwise])" in text
    assert "selected" in text
    with caplog.at_level(logging.INFO, logger="aiter"):
        res.emit()
    assert resolver.EXPLAIN_PREFIX + "problem:" in caplog.text
    monkeypatch.delenv("AITER_MOE_EXPLAIN")
    assert resolver.resolve(_problem(), a8w8_pt.SEGMENTS).explain == ()


def test_default_resolution_logs_like_legacy(monkeypatch):
    from aiter.moev2.migration import a8w8_pt

    monkeypatch.setattr(tuned, "lookup", lambda problem: None)
    res = resolver.resolve(_problem(), a8w8_pt.SEGMENTS)
    (_, first), (_, second) = res.logs
    assert first.startswith("run_1stage = False, xbf16 = False, ksplit = 0 q_type = ")
    assert second.startswith("[fused_moe] using 2stage default for ('gfx950', 256, 64,")
    assert res.metadata.stage1.func.__name__ == "ck_moe_stage1"


# 2.6 a8w8_pt


def test_no_one_stage_heuristic_for_per_tensor():
    # a8w8_pt has no 1-stage segment: the legacy default never picks 1-stage for it.
    for table in legacy.fused_moe_1stage_dict.values():
        for key in table:
            assert key[1] != QuantType.per_Tensor, key


def test_op_call_matches_what_legacy_passes_to_the_custom_op(monkeypatch):
    seen = {}

    def fake_op(**kwargs):
        seen.update(kwargs)
        return "out"

    monkeypatch.setattr(legacy, "fused_moe_", fake_op)
    scatter = types.SimpleNamespace(
        source_token_map=torch.zeros(3, dtype=torch.int32),
        arena_handle=11,
        combine_input_offset=12,
        slot_stride_bytes=13,
        max_tokens_per_rank=14,
        world_size=15,
        combine_quant_bits=8,
    )
    for extra in (
        {},
        {
            "stage2_scatter": scatter,
            "block_size_M": 64,
            "quant_type_a": QuantType.per_Tensor,
        },
    ):
        seen.clear()
        call = {
            name: prm.default
            for name, prm in inspect.signature(legacy.fused_moe).parameters.items()
        }
        call.update(
            hidden_states="h", w1="w1", w2="w2", topk_weight="tw", topk_ids="ti",
            activation=ActivationType.Gelu, quant_type=QuantType.per_Tensor, **extra,
        )  # fmt: skip
        assert legacy.fused_moe(**call) == "out"
        assert api.op_call(call) == seen


def _per_tensor_op_call():
    call = {
        name: prm.default
        for name, prm in inspect.signature(legacy.fused_moe_fake).parameters.items()
    }
    call.update(
        hidden_states=torch.empty(2, 64, dtype=dtypes.bf16),
        w1=torch.empty(4, 32, 64, dtype=dtypes.fp8),
        w2=torch.empty(4, 64, 16, dtype=dtypes.fp8),
        topk_weight=torch.empty(2, 2),
        topk_ids=torch.zeros(2, 2, dtype=torch.int32),
        quant_type=QuantType.per_Tensor.value,
    )
    call["w1"].is_shuffled = call["w2"].is_shuffled = True
    return call


def test_claims_only_per_tensor_fp8(monkeypatch):
    from aiter.moev2.migration import a8w8_pt

    monkeypatch.setattr(a8w8_pt, "get_gfx_runtime", lambda: "gfx950")
    call = _per_tensor_op_call()
    assert a8w8_pt.claims(call) and a8w8_pt.why_not(call) is None
    for limit in (0, 0.0, float("inf")):
        assert a8w8_pt.claims({**call, "swiglu_limit": limit}), limit
    assert a8w8_pt.why_not({**call, "swiglu_limit": 7.0}) == "swiglu_limit=7.0 clamps"
    for change in (
        {"quant_type": QuantType.per_Token.value},
        {"activation": ActivationType.Swiglu.value},
        {"bias1": torch.empty(1)},
        {"hidden_pad": 64},
        {"quant_dtype_a2": dtypes.fp8},
        {"w1": torch.empty(4, 32, 64, dtype=torch.int8)},
        {"dtype": torch.float32},
        {"herd_topk": 1},
        {"deterministic": True},
        {"w1_global_scale": torch.empty(4, 2)},
        {"w2_global_scale": torch.empty(4)},
    ):
        assert not a8w8_pt.claims({**call, **change}), change
        assert a8w8_pt.why_not({**call, **change}), change
    register_presorted_moe({"M": 2})
    try:
        assert not a8w8_pt.claims(call)
    finally:
        register_presorted_moe(None)
    assert a8w8_pt.claims(call)
    monkeypatch.setenv("AITER_ONLINE_TUNE", "1")
    assert not a8w8_pt.claims(call)
    monkeypatch.setattr(a8w8_pt, "get_gfx_runtime", lambda: "gfx1201")
    monkeypatch.delenv("AITER_ONLINE_TUNE")
    assert not a8w8_pt.claims(call)


def test_claims_use_the_live_arch_on_multi_target_builds(monkeypatch, live_chip_info):
    """A multi-target build: get_gfx() names an arch the path declines, the GPU is one
    it claims. Setting GPU_ARCHS would not do: `get_gfx_custom_op` caches a layer below
    the two caches this can clear, so what it returns then depends on the environment.
    """
    from aiter.moev2.migration import a8w8_pt

    for module in (live_chip_info, a8w8_pt):
        if hasattr(module, "get_gfx"):  # also catch a path that imports it for itself
            monkeypatch.setattr(module, "get_gfx", lambda: "gfx1250")
    monkeypatch.setattr(a8w8_pt, "get_gfx_runtime", lambda: "gfx942")
    assert a8w8_pt.claims(_per_tensor_op_call())


def test_a_module_first_imported_in_a_simulation_follows_the_later_ones(
    monkeypatch, tmp_path, live_chip_info
):
    """Kernel modules load at launch, inside a simulation, and bind its fake chip-info."""
    from moe_dispatch import arch_sim

    name = "_late_chip_info_user"
    (tmp_path / f"{name}.py").write_text(
        "from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)

    with arch_sim.simulated_arch("gfx942"):
        late = importlib.import_module(name)
        assert (late.get_gfx_runtime(), late.get_cu_num()) == ("gfx942", 304)
    assert late.get_cu_num is live_chip_info.get_cu_num
    assert late.get_gfx_runtime is live_chip_info.get_gfx_runtime
    with arch_sim.simulated_arch("gfx942_cu80"):
        assert late.get_cu_num() == 80
    with arch_sim.simulated_arch("gfx1201"):
        assert late.get_gfx_runtime() == "gfx1201"


def test_tuned_cache_clear_also_resets_path_caches():
    from aiter.moev2.migration import a8w8_pt

    assert a8w8_pt._dispatch in tuned._CACHES and a8w8_pt._resolve in tuned._CACHES


def test_v2_reads_and_resets_with_the_legacy_table(monkeypatch):
    from aiter.moev2.migration import a8w8_pt

    p = _problem()
    keys, _ = tuned.lookup_keys(p)
    key = keys[:7] + (str(p.activation.activation),) + keys[8:]
    row = {"kernelName1": "", "kernelName2": "", "block_m": 64, "ksplit": 0}
    monkeypatch.setattr(legacy, "cfg_2stages", ({key: row}, {}))
    assert tuned.lookup(p) is row
    a8w8_pt._resolve(p, False)
    assert a8w8_pt._resolve.cache_info().currsize == 1
    legacy.get_2stage_cfgs.cache_clear()
    assert a8w8_pt._resolve.cache_info().currsize == 0


def test_the_log_only_relookup_runs_after_the_sort(monkeypatch, caplog):
    from aiter.moev2 import hooks
    from aiter.moev2.migration import a8w8_pt
    from aiter.moev2.segments.quant import QuantA1

    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise Stop

    def sort(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype, *args, **kw):
        seen.append(caplog.text.count(line))
        if len(seen) == 1:
            raise Stop
        empty = torch.empty(0, dtype=torch.int32)
        out = torch.empty(topk_ids.shape[0], model_dim, dtype=moebuf_dtype)
        return empty, empty, empty, empty, out

    monkeypatch.setattr(a8w8_pt, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(a8w8_pt, "get_cu_num", lambda: 256)
    monkeypatch.setattr(tuned, "lookup", lambda problem: None)
    monkeypatch.setattr(QuantA1, "run", stop)
    # caplog listens on the root logger; aiter's logger does not propagate.
    monkeypatch.setattr(logging.getLogger("aiter"), "propagate", True)
    line, seen = "[fused_moe] using 2stage default", []
    with caplog.at_level(logging.INFO, logger="aiter"), hooks.set_sort_impl(sort):
        # Legacy logs the lookup once before sorting and once more after it.
        for _ in range(2):
            with pytest.raises(Stop):
                a8w8_pt.run(_per_tensor_op_call())
    assert seen == [1, 1] and caplog.text.count(line) == 2


def test_the_relookup_buckets_the_hidden_states_rows(monkeypatch):
    from aiter.moev2 import hooks
    from aiter.moev2.migration import a8w8_pt
    from aiter.moev2.segments.quant import QuantA1

    class Stop(Exception):
        pass

    def stop(*args, **kwargs):
        raise Stop

    def sort(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype, *args, **kw):
        empty = torch.empty(0, dtype=torch.int32)
        out = torch.empty(topk_ids.shape[0], model_dim, dtype=moebuf_dtype)
        return empty, empty, empty, empty, out

    tokens = []

    def lookup(problem):
        tokens.append(problem.shape.m)

    monkeypatch.setattr(a8w8_pt, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(a8w8_pt, "get_cu_num", lambda: 256)
    monkeypatch.setattr(tuned, "lookup", lookup)
    monkeypatch.setattr(QuantA1, "run", stop)
    call = _per_tensor_op_call()
    call["hidden_states"] = torch.empty(3, 64, dtype=dtypes.bf16)
    # fused_moe_2stages buckets hidden_states.shape[0]; the first lookup, topk_ids rows.
    with hooks.set_sort_impl(sort), pytest.raises(Stop):
        a8w8_pt.run(call)
    assert tokens == [2, 4]


_CK_ROW = {
    "kernelName1": "moe_ck2stages_gemm1_256x32x64x128_1x4_MulABScale_v1_Nswizzle0_Quant1_MulRoutedWeight1_silu_F8_F8_B16",
    "kernelName2": "moe_ck2stages_gemm2_256x32x64x128_1x4_MulABScaleExpertWeight_v1_Nswizzle0_Quant1_MulRoutedWeight0_F8_F8_B16",
    "block_m": 32,
    "ksplit": 0,
}


def _sort_not_reached(*args, **kwargs):
    raise AssertionError("v2 ran a call legacy runs differently")


def _relookup_test_call(monkeypatch, row):
    """A call with `a1_scale` whose second lookup (no activation scales) finds `row`."""
    from aiter.moev2.migration import a8w8_pt

    def lookup(problem):
        return None if "act_scales" in problem.features else row

    monkeypatch.setattr(tuned, "lookup", lookup)
    monkeypatch.setattr(a8w8_pt, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(a8w8_pt, "get_cu_num", lambda: 256)
    call = _per_tensor_op_call()
    call["a1_scale"] = torch.ones(1, dtype=torch.float32)
    return call


def test_a_relookup_that_selects_another_plan_still_runs_the_first(monkeypatch):
    from aiter.moev2 import hooks
    from aiter.moev2.migration import a8w8_pt

    class Sorted(Exception):
        pass

    def sort(*args, **kwargs):
        raise Sorted

    # Legacy's _metadata_transform runs the first lookup's plan whatever the second finds.
    call = _relookup_test_call(monkeypatch, _CK_ROW)
    args, kwargs = _key(has_activation_scales=True)
    assert a8w8_pt.dispatch_record(args, kwargs).stage1.keywords["kernelName"] == ""
    with hooks.set_sort_impl(sort), pytest.raises(Sorted):
        a8w8_pt.run(call)


def test_a_call_whose_relookup_v2_declines_is_declined(monkeypatch):
    from moe_dispatch.differential import captured_logs

    from aiter.moev2 import hooks, paths
    from aiter.moev2.migration import a8w8_pt

    call = _relookup_test_call(monkeypatch, {**_CK_ROW, "run_1stage": True})
    args, kwargs = _key(has_activation_scales=True)
    with captured_logs() as logs, hooks.set_sort_impl(_sort_not_reached):
        assert a8w8_pt.dispatch_record(args, kwargs) is None
        assert a8w8_pt.run(call) is paths.DECLINED
    assert logs == []


def test_a_call_whose_relookup_raises_is_declined(monkeypatch):
    from moe_dispatch.differential import captured_logs

    from aiter.moev2 import hooks, paths
    from aiter.moev2.migration import a8w8_pt

    def lookup(problem):
        if problem.shape.m == 4:
            raise RuntimeError("the second lookup raises")

    monkeypatch.setattr(tuned, "lookup", lookup)
    monkeypatch.setattr(a8w8_pt, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(a8w8_pt, "get_cu_num", lambda: 256)
    call = _per_tensor_op_call()
    call["hidden_states"] = torch.empty(3, 64, dtype=dtypes.bf16)
    with captured_logs() as logs, hooks.set_sort_impl(_sort_not_reached):
        assert a8w8_pt.run(call) is paths.DECLINED
    assert logs == []


@pytest.mark.parametrize(
    "shuffle",
    [{"is_shuffled": False}, {"is_shuffled": True, "opus_weights_shuffled": False}],
)
def test_unshuffled_fp8_on_gfx950_raises_like_legacy(shuffle):
    from moe_dispatch.differential import captured_logs

    from aiter.moev2.migration import a8w8_pt

    args, kwargs = _key(**shuffle)
    # Load both tables first so their load messages are not in the compared logs.
    with captured_logs():
        legacy.get_2stage_cfgs.__wrapped__(*_key()[0])
        tuned._default_table()
    with captured_logs() as legacy_logs, pytest.raises(RuntimeError) as legacy_error:
        legacy.get_2stage_cfgs.__wrapped__(*args, **kwargs)
    with captured_logs() as v2_logs, pytest.raises(RuntimeError) as v2_error:
        a8w8_pt.dispatch_record(args, kwargs)
    assert "shuffle_weight() layout" in str(legacy_error.value)
    assert str(v2_error.value) == str(legacy_error.value)
    assert v2_logs == legacy_logs and len(v2_logs) == 2


def test_a_tuned_ck_row_on_unshuffled_gfx950_fp8_raises_before_logging(
    monkeypatch, caplog
):
    from aiter.moev2.migration import a8w8_pt

    row = {
        "kernelName1": "moe_ck2stages_gemm1_256x32x64x128_1x4_MulABScale_v1_Nswizzle0_Quant1_MulRoutedWeight1_silu_F8_F8_B16",
        "kernelName2": "moe_ck2stages_gemm2_256x32x64x128_1x4_MulABScaleExpertWeight_v1_Nswizzle0_Quant1_MulRoutedWeight0_F8_F8_B16",
        "block_m": 32,
        "ksplit": 0,
    }
    monkeypatch.setattr(tuned, "lookup", lambda problem: row)
    # caplog listens on the root logger; aiter's logger does not propagate.
    monkeypatch.setattr(logging.getLogger("aiter"), "propagate", True)
    with (
        caplog.at_level(logging.INFO, logger="aiter"),
        pytest.raises(RuntimeError, match="shuffle_weight"),
    ):
        resolver.resolve(_problem(is_shuffled=False), a8w8_pt.SEGMENTS)
    assert caplog.text == ""
    assert resolver.resolve(_problem(), a8w8_pt.SEGMENTS).plan.source == "tuned"


@pytest.mark.parametrize(
    "a2_scale,zeroed", [(None, True), (torch.ones(1, dtype=torch.float32), False)]
)
@pytest.mark.parametrize("row_major", [True, False])
def test_two_stage_inputs_contiguous_and_a2_zeroed_like_legacy(
    monkeypatch, a2_scale, zeroed, row_major
):
    from aiter.moev2 import hooks
    from aiter.moev2.execute import executor
    from aiter.moev2.migration import a8w8_pt
    from aiter.moev2.segments.quant import QuantA1

    class Stop(Exception):
        pass

    def sort(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype, *args, **kw):
        empty = torch.empty(0, dtype=torch.int32)
        out = torch.empty(topk_ids.shape[0], model_dim, dtype=moebuf_dtype)
        return empty, empty, empty, empty, out

    def quant_a1(self, quant_type, hidden_states, *args):
        seen.append(hidden_states.is_contiguous())
        return hidden_states, None

    def record_kernel(stage, call):
        raise Stop

    zeros = torch.zeros
    zeroed_shapes = []

    def spy_zeros(*shape, **kwargs):
        zeroed_shapes.append(tuple(shape[0] if len(shape) == 1 else shape))
        return zeros(*shape, **kwargs)

    monkeypatch.setattr(a8w8_pt, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(a8w8_pt, "get_cu_num", lambda: 256)
    monkeypatch.setattr(tuned, "lookup", lambda problem: None)
    monkeypatch.setattr(QuantA1, "run", quant_a1)
    monkeypatch.setattr(hooks, "record_kernel", record_kernel)
    monkeypatch.setattr(executor.torch, "zeros", spy_zeros)
    seen = []
    call = _per_tensor_op_call()
    wide = torch.empty(2, 128, dtype=dtypes.bf16)
    # [:, :64] keeps unit column stride (legacy copies it late); [:, ::2] does not.
    call["hidden_states"] = wide[:, :64] if row_major else wide[:, ::2]
    call["a2_scale"] = a2_scale
    with hooks.set_sort_impl(sort), pytest.raises(Stop):
        a8w8_pt.run(call)
    assert seen == [True]
    assert ((2, 2, 16) in zeroed_shapes) == zeroed


def test_a_strided_input_is_copied_before_the_output_overlap_check(monkeypatch):
    from aiter.moev2 import hooks
    from aiter.moev2.migration import a8w8_pt
    from aiter.moev2.segments.quant import QuantA1

    class Stop(Exception):
        pass

    def sort(topk_ids, topk_weights, num_experts, model_dim, moebuf_dtype, *args, **kw):
        empty = torch.empty(0, dtype=torch.int32)
        return empty, empty, empty, empty, kw["output"]

    def stop(*args, **kwargs):
        raise Stop

    monkeypatch.setattr(a8w8_pt, "get_gfx_runtime", lambda: "gfx950")
    monkeypatch.setattr(a8w8_pt, "get_cu_num", lambda: 256)
    monkeypatch.setattr(tuned, "lookup", lambda problem: None)
    monkeypatch.setattr(QuantA1, "run", stop)
    call = _per_tensor_op_call()
    buf = torch.empty(256, dtype=dtypes.bf16)
    # Legacy copies a non-row-major input first, so the output may share its storage.
    call["hidden_states"] = buf.view(2, 128)[:, ::2]
    call["output"] = buf[:128].view(2, 64)
    with hooks.set_sort_impl(sort), pytest.raises(Stop):
        a8w8_pt.run(call)


def test_shape_helpers_moved_verbatim():
    from aiter.moev2 import shapes
    from aiter.moev2.segments import ck

    assert legacy.get_padded_M is shapes.get_padded_M
    assert legacy.get_inter_dim is shapes.get_inter_dim
    assert legacy._is_row_major_2d is shapes._is_row_major_2d
    assert legacy._require_shuffled_gfx950_fp8 is ck._require_shuffled_gfx950_fp8
    assert shapes.get_inter_dim((4, 32, 64), (4, 64, 16)) == (4, 64, 16)


def test_shape_is_frozen():
    s = Shape(1, 2, 3, 4, 5)
    with pytest.raises(AttributeError):
        s.m = 3
    assert ActSpec("silu").activation == ActivationType.Silu
    assert WeightLayout("16x16", "separated").g1u1
