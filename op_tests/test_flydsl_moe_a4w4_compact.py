# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL A4W4 compact MoE (MXFP4 prefill) through ``aiter.fused_moe``.

Weights are prepared exactly as the per_1x32 fp4 MoE paths expect them
(``shuffle_weight`` (16, 16) + ``e8m0_shuffle``, gate/up SEPARATED) and a tuned
``flydsl_a4w4c_g1_*`` / ``flydsl_a4w4c_g2_*`` row routes the call to the compact
family. The output is compared with a torch reference that quantizes
activations with AITER's runtime MX rule.
"""

import os

import pandas as pd
import pytest
import torch

import aiter
import aiter.fused_moe as fused_moe_module
import aiter.fused_moe_registry as _registry
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, get_2stage_cfgs, get_padded_M
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import a4w4c_kname, moe_a4w4_compact
from aiter.ops.flydsl.moe_a4w4_compact import tune_space
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="FlyDSL A4W4 compact MoE is gfx950-only"
)

MODEL_DIM = 6144
EXPERTS = 129
TOPK = 5
SWIGLU_LIMIT = 7.0
TUNED_COLUMNS = [
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
    "ksplit",
    "us1",
    "kernelName1",
    "err1",
    "us2",
    "kernelName2",
    "err2",
    "us",
    "run_1stage",
    "xbf16",
    "flat",
    "tflops",
    "bw",
    "_tag",
]


def _skewed_ids(tokens, routed, gen):
    """Hot expert 0 on every token, expert 1 on 257 rows (BM 256 + 1), expert 2 on
    one row, experts 120.. empty; other slots spread over 3..119, distinct per token."""
    assert tokens > 258 and routed >= 2
    scores = torch.rand(tokens, 117, device="cuda", generator=gen)
    ids = scores.topk(routed - 1, dim=-1).indices.to(torch.int32) + 3
    ids[:257, 0] = 1
    ids[257, 0] = 2
    hot = torch.zeros(tokens, 1, device="cuda", dtype=torch.int32)
    return torch.cat([hot, ids], dim=1)


def _problem(tokens, inter_dim, shared_expert=True, seed=0, routing="uniform"):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, MODEL_DIM, device="cuda", generator=gen).to(dtypes.bf16)
    w1 = torch.randn(EXPERTS, 2 * inter_dim, MODEL_DIM, device="cuda", generator=gen)
    w2 = torch.randn(EXPERTS, MODEL_DIM, inter_dim, device="cuda", generator=gen)
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1_q, w1_s = quant((w1 / 10).to(dtypes.bf16), quant_dtype=dtypes.fp4x2)
    w2_q, w2_s = quant((w2 / 30).to(dtypes.bf16), quant_dtype=dtypes.fp4x2)
    w1_q = w1_q.view(EXPERTS, 2 * inter_dim, MODEL_DIM // 2)
    w2_q = w2_q.view(EXPERTS, MODEL_DIM, inter_dim // 2)
    routed = TOPK - 1 if shared_expert else TOPK
    if routing == "skewed":
        ids = _skewed_ids(tokens, routed, gen)
    else:
        scores = torch.rand(tokens, EXPERTS - 1, device="cuda", generator=gen)
        ids = scores.topk(routed, dim=-1).indices.to(torch.int32)
    weights = torch.rand(tokens, routed, device="cuda", generator=gen)
    if shared_expert:
        shared = torch.full((tokens, 1), EXPERTS - 1, device="cuda", dtype=torch.int32)
        ids = torch.cat([ids, shared], dim=1)
        weights = torch.cat([weights, torch.ones(tokens, 1, device="cuda")], dim=1)
    return x, (w1_q, w1_s, w2_q, w2_s), ids, weights.float()


def _aiter_weights(w1_q, w1_s, w2_q, w2_s):
    w1 = shuffle_weight(w1_q, layout=(16, 16))
    w2 = shuffle_weight(w2_q, layout=(16, 16))
    w1.is_shuffled = w2.is_shuffled = True
    return w1, w2, fp4_utils.e8m0_shuffle(w1_s), fp4_utils.e8m0_shuffle(w2_s)


def _dequant(q, s):
    values = fp4_utils.mxfp4_to_f32(q.view(torch.uint8))
    scales = fp4_utils.e8m0_to_f32(s.view(torch.uint8).view(*q.shape[:-1], -1))
    return (values.view(*scales.shape, 32) * scales.unsqueeze(-1)).view(values.shape)


def _reference(x, weights, ids, topk_weight):
    w1_q, w1_s, w2_q, w2_s = weights
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    a_q, a_s = quant(x, quant_dtype=dtypes.fp4x2)
    a = _dequant(a_q.view(x.shape[0], -1), a_s.view(x.shape[0], -1))
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    inter_dim = w2_q.shape[-1] * 2
    w1_s = w1_s.view(EXPERTS, 2 * inter_dim, -1)
    w2_s = w2_s.view(EXPERTS, MODEL_DIM, -1)
    for expert in ids.unique().tolist():
        token, slot = (ids == expert).nonzero(as_tuple=True)
        gate_up = a[token] @ _dequant(w1_q[expert], w1_s[expert]).T
        h = aiter.fused_moe.swiglu(
            gate_up[:, :inter_dim], gate_up[:, inter_dim:], limit=SWIGLU_LIMIT
        )
        h_q, h_s = quant(h, quant_dtype=dtypes.fp4x2)
        h = _dequant(h_q.view(h.shape[0], -1), h_s.view(h.shape[0], -1))
        y = h @ _dequant(w2_q[expert], w2_s[expert]).T
        out.index_add_(0, token, y * topk_weight[token, slot, None])
    return out


def _install_row(path, monkeypatch, tokens, inter_dim, names, block_m=None):
    """Point the tuned-config lookup at a single compact row for this shape."""
    kname1, kname2 = names
    if block_m is None:
        block_m = a4w4c_kname.parse_knames(kname1, kname2)["BM1"]
    row = dict.fromkeys(TUNED_COLUMNS, 0)
    row.update(
        gfx="gfx950",
        cu_num=torch.cuda.get_device_properties(0).multi_processor_count,
        token=get_padded_M(tokens),
        model_dim=MODEL_DIM,
        inter_dim=inter_dim,
        expert=EXPERTS,
        topk=TOPK,
        act_type=str(ActivationType.Swiglu),
        dtype=str(dtypes.bf16),
        q_dtype_a=str(dtypes.fp4x2),
        q_dtype_w=str(dtypes.fp4x2),
        q_type=str(QuantType.per_1x32),
        use_g1u1=1,
        block_m=block_m,
        kernelName1=kname1,
        kernelName2=kname2,
        err1="0%",
        err2="0%",
        _tag="",
    )
    pd.DataFrame([row], columns=TUNED_COLUMNS).to_csv(path, index=False)
    monkeypatch.setenv("AITER_CONFIG_FMOE", str(path))
    _reset_tuned_config_caches()


def _spy_impl(monkeypatch):
    """Record every call the registry makes to the compact impl."""
    calls = []
    impl = moe_a4w4_compact.run_moe_a4w4_compact

    def spy(request, config):
        calls.append(config)
        return impl(request, config)

    monkeypatch.setitem(_registry._IMPLEMENTATIONS, a4w4c_kname.IMPL_NAME, spy)
    return calls


@pytest.fixture
def tuned_row(tmp_path, monkeypatch):
    def install(tokens, inter_dim, names=None):
        names = names or tune_space(inter_dim)[-1]
        path = tmp_path / "moe_a4w4_compact_tuned_fmoe.csv"
        _install_row(path, monkeypatch, tokens, inter_dim, names)
        return names

    calls = _spy_impl(monkeypatch)
    install.calls = calls
    yield install
    monkeypatch.undo()
    _reset_tuned_config_caches()
    assert calls, "fused_moe did not dispatch to FlyDSL A4W4 compact MoE"


def _reset_tuned_config_caches():
    from aiter.jit.core import AITER_CONFIGS

    type(AITER_CONFIGS).get_config_file.cache_clear()
    fused_moe_module.cfg_2stages = None
    get_2stage_cfgs.cache_clear()


def _run(x, weights, ids, topk_weight, **kwargs):
    w1, w2, w1_s, w2_s = _aiter_weights(*weights)
    return fused_moe(
        x,
        w1,
        w2,
        topk_weight,
        ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_s,
        w2_scale=w2_s,
        dtype=dtypes.bf16,
        swiglu_limit=SWIGLU_LIMIT,
        **kwargs,
    )


def _rel_l2(out, ref):
    return ((out.float() - ref).norm() / ref.norm()).item()


def _max_token_rel_l2(out, ref):
    """Worst per-token relative L2: a dropped or corrupted row cannot hide in the mean."""
    return (
        ((out.float() - ref).norm(dim=1) / ref.norm(dim=1).clamp_min(1e-6)).max().item()
    )


def _check(out, ref, tol=2e-2, token_tol=1e-2):
    assert _rel_l2(out, ref) < tol
    assert _max_token_rel_l2(out, ref) < token_tol


@pytest.mark.parametrize("inter_dim", [384, 768, 1536])
@pytest.mark.parametrize("tokens", [512, 1000, 4096])
def test_moe_a4w4_compact_matches_reference(tuned_row, inter_dim, tokens):
    tuned_row(tokens, inter_dim)
    x, weights, ids, topk_weight = _problem(tokens, inter_dim)
    out = _run(x, weights, ids, topk_weight)
    _check(out, _reference(x, weights, ids, topk_weight))


@pytest.mark.parametrize("inter_dim", [384, 768, 1536])
def test_moe_a4w4_compact_every_shipped_config(tuned_row, inter_dim):
    x, weights, ids, topk_weight = _problem(777, inter_dim)
    ref = _reference(x, weights, ids, topk_weight)
    for names in tune_space(inter_dim):
        tuned_row(777, inter_dim, names)
        _assert_one_dispatch(tuned_row.calls, names, x, weights, ids, topk_weight, ref)


def _assert_one_dispatch(calls, names, x, weights, ids, topk_weight, ref, tol=2e-2):
    before = len(calls)
    out = _run(x, weights, ids, topk_weight)
    assert len(calls) == before + 1, ("no compact dispatch", names)
    assert calls[-1] == a4w4c_kname.impl_config(*names), (calls[-1], names)
    assert _rel_l2(out, ref) < tol, names
    assert _max_token_rel_l2(out, ref) < 1e-2, names


@pytest.mark.parametrize("persist", [True, False], ids=["il4_persist", "non_persist"])
@pytest.mark.parametrize("inter_dim", [384, 768, 1536])
def test_moe_a4w4_compact_skewed_routing(tuned_row, inter_dim, persist):
    names = next(
        n
        for n in tune_space(inter_dim)
        if (
            ("_il4_" in n[0] and "_persist" in n[0])
            if persist
            else "_persist" not in n[0] + n[1]
        )
    )
    x, weights, ids, topk_weight = _problem(4096, inter_dim, routing="skewed")
    counts = torch.bincount(ids.flatten().long(), minlength=EXPERTS)
    assert (counts[: EXPERTS - 1] == 0).sum() >= 4
    assert (counts == 1).any() and (counts == 257).any() and counts[0] == 4096
    assert all(len(set(r)) == TOPK for r in ids.tolist())
    assert (ids[:, -1] == EXPERTS - 1).all()
    tuned_row(4096, inter_dim, names)
    ref = _reference(x, weights, ids, topk_weight)
    _assert_one_dispatch(tuned_row.calls, names, x, weights, ids, topk_weight, ref)


def test_moe_a4w4_compact_arbitrary_routing(tuned_row):
    """Tokens without the shared expert must stay exact even for fused-combine rows."""
    names = next(n for n in tune_space(768) if "_fc_" in n[1])
    tuned_row(2048, 768, names)
    x, weights, ids, topk_weight = _problem(2048, 768, shared_expert=False)
    out = _run(x, weights, ids, topk_weight)
    _check(out, _reference(x, weights, ids, topk_weight))


def _checkpoint_layer(path, layer, inter_dim):
    """TP rank-0 shard of one MiniMax-M3-MXFP4 MoE layer, shared expert as E-1."""
    import json

    from safetensors import safe_open

    with open(os.path.join(path, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"language_model.model.layers.{layer}.block_sparse_moe."
    experts = [
        (f"experts.{e}.w1", f"experts.{e}.w3", f"experts.{e}.w2") for e in range(128)
    ]
    experts.append(
        (
            "shared_experts.gate_proj",
            "shared_experts.up_proj",
            "shared_experts.down_proj",
        )
    )
    files = {}

    def get(name, index=slice(None)):
        file = weight_map[prefix + name]
        if file not in files:
            files[file] = safe_open(os.path.join(path, file), "pt", device="cpu")
        return files[file].get_slice(prefix + name)[index].cuda()

    rows, cols = slice(0, inter_dim), (slice(None), slice(0, inter_dim // 32))
    w1, s1, w2, s2 = [], [], [], []
    for gate, up, down in experts:
        w1.append(torch.cat([get(f"{gate}.weight", rows), get(f"{up}.weight", rows)]))
        s1.append(
            torch.cat(
                [get(f"{gate}.weight_scale", rows), get(f"{up}.weight_scale", rows)]
            )
        )
        w2.append(get(f"{down}.weight", (slice(None), slice(0, inter_dim // 2))))
        s2.append(get(f"{down}.weight_scale", cols))
    w1, w2 = torch.stack(w1).view(dtypes.fp4x2), torch.stack(w2).view(dtypes.fp4x2)
    s1 = torch.stack(s1).view(EXPERTS * 2 * inter_dim, -1)
    s2 = torch.stack(s2).view(EXPERTS * MODEL_DIM, -1)
    return (w1, s1, w2, s2), get("gate.weight").float(), get("e_score_correction_bias")


@pytest.mark.skipif(
    not os.environ.get("AITER_MINIMAX_M3_MXFP4_PATH"),
    reason="set AITER_MINIMAX_M3_MXFP4_PATH to a MiniMax-M3-MXFP4 checkpoint",
)
@pytest.mark.parametrize("inter_dim", [384, 768, 1536])
def test_moe_a4w4_compact_checkpoint_weights(tuned_row, inter_dim):
    """Real MiniMax-M3-MXFP4 expert weights and router, one MoE layer, every config."""
    path = os.environ["AITER_MINIMAX_M3_MXFP4_PATH"]
    weights, gate, bias = _checkpoint_layer(path, 30, inter_dim)
    gen = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(2048, MODEL_DIM, device="cuda", generator=gen).to(dtypes.bf16)
    scores = torch.sigmoid(x.float() @ gate.T)
    ids = (scores + bias.float()).topk(TOPK - 1, dim=-1).indices
    topk_weight = scores.gather(1, ids)
    topk_weight = topk_weight / topk_weight.sum(-1, keepdim=True)
    shared = torch.full((x.shape[0], 1), EXPERTS - 1, device="cuda")
    ids = torch.cat([ids, shared], dim=1).to(torch.int32)
    topk_weight = torch.cat([topk_weight, torch.ones_like(shared)], dim=1).float()
    ref = _reference(x, weights, ids, topk_weight)
    for names in tune_space(inter_dim):
        tuned_row(x.shape[0], inter_dim, names)
        _assert_one_dispatch(
            tuned_row.calls, names, x, weights, ids, topk_weight, ref, tol=1e-2
        )


def test_moe_a4w4_compact_layout_roundtrip():
    _, (w1_q, w1_s, w2_q, w2_s), _, _ = _problem(1, 384)
    w1, w2, w1_sh, w2_sh = _aiter_weights(w1_q, w1_s, w2_q, w2_s)
    for shuffled, raw in ((w1, w1_q), (w2, w2_q)):
        # B atom: [E][N/16][K/128][lane][16 B], lane l = row l % 16, bytes 16 (l // 16)
        e, n, kh = raw.shape
        atoms = raw.view(torch.uint8).view(e, n // 16, 16, kh // 64, 4, 16)
        atoms = atoms.permute(0, 1, 3, 4, 2, 5).reshape(e, n, kh)
        assert torch.equal(shuffled.view(torch.uint8), atoms)
    weights = moe_a4w4_compact._weights(w1, w2, w1_sh, w2_sh)
    assert weights.b1.data_ptr() == w1.data_ptr()
    assert weights.b2.data_ptr() == w2.data_ptr()
    for packed, raw in ((w1_sh, w1_s), (w2_sh, w2_s)):
        e_n, groups = raw.shape
        unpacked = moe_a4w4_compact._unshuffle_scale(packed, 1, e_n, groups)
        assert torch.equal(unpacked.view(e_n, groups), raw.view(torch.uint8))


def test_moe_a4w4_compact_kname_roundtrip():
    """Every shipped table cell names both stages and parses back to its settings."""
    for inter_dim in (384, 768, 1536):
        for tokens, cell in moe_a4w4_compact.load_table(inter_dim).items():
            assert tokens >= moe_a4w4_compact.MIN_TOKENS
            kname1, kname2 = a4w4c_kname.kernel_names(cell)
            assert kname1.startswith(a4w4c_kname.G1_PREFIX)
            assert kname2.startswith(a4w4c_kname.G2_PREFIX)
            for name in (kname1, kname2):
                assert "," not in name and " " not in name and "__" not in name
            parsed = a4w4c_kname.parse_knames(kname1, kname2)
            assert parsed == moe_a4w4_compact.cfg_kwargs(cell), (kname1, kname2)


def _fc_free(kwargs):
    fc_keys = ("FC", "BMF", "NWF", "WMF", "DF", "pipeF", "diagF")
    return {k: v for k, v in kwargs.items() if k not in fc_keys}


def test_moe_a4w4_compact_kname_roundtrip_synthetic():
    """FC without diagF must not inherit the stage-2 diag (ag136 is not a fused token)."""
    cell = {
        "s1": {"BM": 128, "NW": 4, "pipe": "hybrid2", "D": 3},
        "s2": {"BM": 64, "NW": 4, "pipe": "hybrid2", "D": 2, "diag": "wpe3+ag136"},
        "global": {"FC": 1, "BMF": 128, "pipeF": "hybrid", "DF": 3},
    }
    kname1, kname2 = a4w4c_kname.kernel_names(cell)
    parsed = a4w4c_kname.parse_knames(kname1, kname2)
    expected = moe_a4w4_compact.cfg_kwargs(cell)
    assert _fc_free(parsed) == _fc_free(expected)
    assert "diagF" not in parsed
    assert parsed["FC"] == 1 and parsed["BMF"] == 128 and parsed["DF"] == 3
    assert parsed["NWF"] == 4 and parsed["WMF"] == 1 and parsed["pipeF"] == "hybrid"
    assert kname2.endswith("_fc_128x256_hybrid_d3_w4"), kname2


def test_moe_a4w4_compact_kname_rejects_malformed():
    kname1, kname2 = tune_space(768)[-1]
    assert "_fc_" in kname2
    for bad in (
        (kname1 + "_bogus", kname2),
        (kname1.replace("x256_", "x128_", 1), kname2),
        (kname1.replace("_ht", ""), kname2),
        (kname1, kname2.replace("_g2_", "_g1_", 1)),
        (kname1, kname2 + "_persist"),
        (kname1, kname2 + "_fexp"),
        (kname1, kname2 + "_vacc"),
        (kname1, kname2 + "_persist2"),
        (kname1.replace("_w4_wm2_", "_w4_wm3_", 1), kname2),
        (kname1.replace("_w4_wm2_", "_w4_wm0_", 1), kname2),
        (kname1.replace("_w4_wm2_", "_w3_wm2_", 1), kname2),
        (kname1, ""),
        (kname1, "flydsl_moe2_layout_garbage"),
    ):
        with pytest.raises(ValueError):
            a4w4c_kname.parse_knames(*bad)
        with pytest.raises(ValueError):
            a4w4c_kname.impl_config(*bad)


def test_moe_a4w4_compact_rejects_unsupported():
    from aiter.fused_moe_registry import FusedMoeRequest

    x, weights, ids, topk_weight = _problem(512, 768)
    w1, w2, w1_s, w2_s = _aiter_weights(*weights)
    base = {
        "hidden_states": x,
        "w1": w1,
        "w2": w2,
        "topk_weight": topk_weight,
        "topk_ids": ids,
        "activation": ActivationType.Swiglu,
        "quant_type": QuantType.per_1x32,
        "w1_scale": w1_s,
        "w2_scale": w2_s,
    }
    unsupported_reason = moe_a4w4_compact.unsupported_reason
    assert unsupported_reason(FusedMoeRequest(**base)) is None
    for change in (
        {"activation": ActivationType.Silu},
        {"intermediate_pad": 128},
        {"hidden_pad": 256},
        {"doweight_stage1": True},
        {"expert_mask": torch.ones(EXPERTS, device="cuda")},
        {"gate_mode": "interleave"},
    ):
        assert unsupported_reason(FusedMoeRequest(**{**base, **change})) is not None


def _request(tokens, inter_dim, device="cuda"):
    from aiter.fused_moe_registry import FusedMoeRequest

    def empty(*shape, dtype):
        return torch.empty(*shape, dtype=dtype, device=device)

    w1 = empty(EXPERTS, 2 * inter_dim, MODEL_DIM // 2, dtype=dtypes.fp4x2)
    w2 = empty(EXPERTS, MODEL_DIM, inter_dim // 2, dtype=dtypes.fp4x2)
    w1.is_shuffled = w2.is_shuffled = True
    return FusedMoeRequest(
        hidden_states=empty(tokens, MODEL_DIM, dtype=dtypes.bf16),
        w1=w1,
        w2=w2,
        topk_weight=empty(tokens, TOPK, dtype=torch.float32),
        topk_ids=empty(tokens, TOPK, dtype=torch.int32),
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
    )


def test_moe_a4w4_compact_rejects_shapes():
    reason = moe_a4w4_compact.unsupported_reason
    assert reason(_request(512, 768, "meta")) is None
    assert reason(_request(512, 192, "meta")) is not None
    assert reason(_request(180000, 768, "meta")) == (
        "T * model_dim too large for 32-bit offsets"
    )


def test_moe_a4w4_compact_inter_dim_192_falls_back(tmp_path, monkeypatch):
    """A compact row for inter_dim % 128 != 0 is ignored, not dispatched."""
    tokens, inter_dim = 512, 192
    calls = _spy_impl(monkeypatch)
    names = tune_space(384)[0]
    _install_row(tmp_path / "tuned.csv", monkeypatch, tokens, inter_dim, names)
    try:
        metadata = get_2stage_cfgs(
            *_cfg_args(tokens, inter_dim), opus_weights_shuffled=True
        )
        assert metadata.full_impl is None
        assert moe_a4w4_compact.unsupported_reason(_request(tokens, inter_dim))
        x, weights, ids, topk_weight = _problem(tokens, inter_dim)
        try:
            _run(x, weights, ids, topk_weight)
        except NotImplementedError as error:
            assert "compact" not in str(error), error
        assert not calls
    finally:
        monkeypatch.undo()
        _reset_tuned_config_caches()


@pytest.mark.parametrize("kname2", ["flydsl_moe2_layout_garbage", ""])
def test_moe_a4w4_compact_bad_g2_name_falls_back(tmp_path, monkeypatch, kname2):
    """A row with a valid g1 and an unparsable g2 name uses the default path."""
    tokens, inter_dim = 1024, 768
    calls = _spy_impl(monkeypatch)
    kname1 = tune_space(inter_dim)[0][0]
    _install_row(
        tmp_path / "tuned.csv",
        monkeypatch,
        tokens,
        inter_dim,
        (kname1, kname2),
        block_m=a4w4c_kname.parse_knames(*tune_space(inter_dim)[0])["BM1"],
    )
    try:
        metadata = get_2stage_cfgs(
            *_cfg_args(tokens, inter_dim), opus_weights_shuffled=True
        )
        assert metadata.full_impl is None
        x, weights, ids, topk_weight = _problem(tokens, inter_dim)
        out = _run(x, weights, ids, topk_weight)
        assert not calls
        # The reference is the untuned default path the bad row falls back to.
        empty = tmp_path / "empty.csv"
        pd.DataFrame(columns=TUNED_COLUMNS).to_csv(empty, index=False)
        monkeypatch.setenv("AITER_CONFIG_FMOE", str(empty))
        _reset_tuned_config_caches()
        ref = _run(x, weights, ids, topk_weight)
        assert _rel_l2(out, ref.float()) < 1e-2
        assert not calls
    finally:
        monkeypatch.undo()
        _reset_tuned_config_caches()


def test_moe_a4w4_compact_shipped_rows_resolve():
    """Every shipped compact row parses, names a table cell and resolves to the impl."""
    path = os.path.join(
        os.path.dirname(aiter.__file__),
        "configs/model_configs/minimax_m3_fp4_tuned_fmoe.csv",
    )
    rows = pd.read_csv(path)
    rows = rows[rows["kernelName1"].map(a4w4c_kname.is_a4w4c_kname)]
    assert len(rows) == 14
    for _, row in rows.iterrows():
        names = (row["kernelName1"], row["kernelName2"])
        assert names in tune_space(int(row["inter_dim"])), names
        assert int(row["block_m"]) == a4w4c_kname.parse_knames(*names)["BM1"]
        assert abs(row["us1"] + row["us2"] - row["us"]) < 1e-3
        impl_name = _registry.make_fused_moe_impl_kernel_name(
            a4w4c_kname.IMPL_NAME, a4w4c_kname.impl_config(*names)
        )
        assert _registry.resolve_fused_moe_impl(impl_name) is not None


def _cfg_args(tokens, inter_dim, intermediate_pad=0):
    return (
        get_padded_M(tokens),
        MODEL_DIM,
        inter_dim,
        EXPERTS,
        TOPK,
        dtypes.bf16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Swiglu,
        False,
        0,
        intermediate_pad,
    )


def test_moe_a4w4_compact_dispatch_metadata(tmp_path, monkeypatch):
    """A per-stage row resolves to the whole-graph impl; a padded call does not."""
    names = tune_space(768)[-1]
    _install_row(tmp_path / "tuned.csv", monkeypatch, 2048, 768, names)
    try:
        metadata = get_2stage_cfgs(*_cfg_args(2048, 768), opus_weights_shuffled=True)
        assert metadata.full_impl is not None
        assert metadata.block_m == a4w4c_kname.parse_knames(*names)["BM1"]
        padded = get_2stage_cfgs(
            *_cfg_args(2048, 768, intermediate_pad=128), opus_weights_shuffled=True
        )
        assert padded.full_impl is None
    finally:
        monkeypatch.undo()
        _reset_tuned_config_caches()


def _pad_weights(weights, inter_dim, padded):
    """Zero-pad an MXFP4 inter_dim shard as vLLM does at TP8 (scales 0x7F)."""
    w1_q, w1_s, w2_q, w2_s = weights
    pad = padded - inter_dim
    w1 = w1_q.view(torch.uint8).view(EXPERTS, 2, inter_dim, -1)
    w1 = torch.nn.functional.pad(w1, (0, 0, 0, pad))
    s1 = w1_s.view(torch.uint8).view(EXPERTS, 2, inter_dim, -1)
    s1 = torch.nn.functional.pad(s1, (0, 0, 0, pad), value=0x7F)
    w2 = torch.nn.functional.pad(w2_q.view(torch.uint8), (0, pad // 2))
    s2 = w2_s.view(torch.uint8).view(EXPERTS, MODEL_DIM, -1)
    s2 = torch.nn.functional.pad(s2, (0, pad // 32), value=0x7F)
    return (
        w1.view(EXPERTS, 2 * padded, -1).view(dtypes.fp4x2),
        s1.view(EXPERTS * 2 * padded, -1).view(dtypes.fp8_e8m0),
        w2.view(dtypes.fp4x2),
        s2.view(EXPERTS * MODEL_DIM, -1).view(dtypes.fp8_e8m0),
    )


def test_moe_a4w4_compact_skips_padded_calls(tmp_path, monkeypatch):
    """A padded call never reaches the compact family, even with a tuned row."""
    tokens, inter_dim, padded = 1024, 384, 512
    calls = _spy_impl(monkeypatch)
    path = tmp_path / "moe_a4w4_compact_tuned_fmoe.csv"
    _install_row(path, monkeypatch, tokens, padded, tune_space(inter_dim)[-1])
    try:
        x, weights, ids, topk_weight = _problem(tokens, inter_dim)
        weights = _pad_weights(weights, inter_dim, padded)
        try:
            _run(x, weights, ids, topk_weight, intermediate_pad=padded - inter_dim)
        except NotImplementedError as error:  # a non-compact path may reject padding
            assert "compact" not in str(error), error
        assert not calls, "padded call dispatched to the compact family"
    finally:
        monkeypatch.undo()
        _reset_tuned_config_caches()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", *os.sys.argv[1:]]))


FC_TOKENS = 2048


def _fc_taken():
    (run,) = moe_a4w4_compact._runs.values()
    assert run.FC
    return int(run.plan_scratch[1][run.W.E].item())


def _routing(case, tokens, inter_dim):
    x, weights, ids, topk_weight = _problem(tokens, inter_dim, shared_expert=True)
    if case == "no_shared":
        _, _, ids, _ = _problem(tokens, inter_dim, shared_expert=False, seed=1)
    elif case == "some_lack":
        _, _, plain, _ = _problem(tokens, inter_dim, shared_expert=False, seed=1)
        ids[::3] = plain[::3]
    elif case == "some_twice":
        ids[::5, 0] = EXPERTS - 1
    elif case == "balanced":
        # count[E-1] == T, yet token 2i has E-1 twice and token 2i+1 none.
        _, _, plain, _ = _problem(tokens, inter_dim, shared_expert=False, seed=1)
        ids[0::2, 0] = EXPERTS - 1
        ids[1::2] = plain[1::2]
    return x, weights, ids.contiguous(), topk_weight


@pytest.mark.parametrize("inter_dim", [768, 1536])
@pytest.mark.parametrize(
    "case", ["once", "no_shared", "some_lack", "some_twice", "balanced"]
)
def test_moe_a4w4_compact_fc_any_routing(tuned_row, inter_dim, case):
    names = next(n for n in tune_space(inter_dim) if "_fc_" in n[1])
    tuned_row(FC_TOKENS, inter_dim, names)
    x, weights, ids, topk_weight = _routing(case, FC_TOKENS, inter_dim)
    out = _run(x, weights, ids, topk_weight)
    torch.cuda.synchronize()
    assert _fc_taken() == (case == "once")
    _check(out, _reference(x, weights, ids, topk_weight))


def test_moe_a4w4_compact_fc_routing_changes(tuned_row):
    """One cached run: valid and invalid batches alternate without stale state."""
    names = next(n for n in tune_space(768) if "_fc_" in n[1])
    tuned_row(FC_TOKENS, 768, names)
    for case in ("once", "some_twice", "once", "some_lack", "once"):
        x, weights, ids, topk_weight = _routing(case, FC_TOKENS, 768)
        out = _run(x, weights, ids, topk_weight)
        torch.cuda.synchronize()
        assert _fc_taken() == (case == "once"), case
        ref = _reference(x, weights, ids, topk_weight)
        _check(out, ref)


@pytest.mark.parametrize(
    "edit",
    [
        lambda k1, k2: (k1.replace("_d3_", "_d3_d3_", 1), k2),
        lambda k1, k2: (k1, k2.replace("_nt", "_nt_nt", 1)),
        lambda k1, k2: (k1.replace("_persist_fexp", "_fexp_persist", 1), k2),
        lambda k1, k2: (k1.replace("_fexp", "", 1), k2),
        lambda k1, k2: (k1, k2.replace("_s2tl", "", 1)),
        lambda k1, k2: (
            k1.replace("_wm2_persist", "", 1).replace("256x256", "256x512", 1),
            k2,
        ),
    ],
    ids=[
        "dup_d",
        "dup_nt",
        "reordered",
        "s1tr_no_fexp",
        "il4_s2_no_s2tl",
        "il4_no_persist",
    ],
)
def test_moe_a4w4_compact_kname_strict(edit):
    names = next(n for n in tune_space(768) if "_s1tr" in n[0] and "_il4_" in n[1])
    a4w4c_kname.impl_config(*names)
    bad = edit(*names)
    assert bad != names
    with pytest.raises(ValueError):
        a4w4c_kname.impl_config(*bad)


def test_moe_a4w4_compact_scales_follow_inplace_updates():
    _, weights, _, _ = _problem(1, 384)
    w1, w2, w1_s, w2_s = _aiter_weights(*weights)
    first = moe_a4w4_compact._weights(w1, w2, w1_s, w2_s)
    w1_s.view(torch.uint8).add_(1)
    second = moe_a4w4_compact._weights(w1, w2, w1_s, w2_s)
    assert not torch.equal(first.bs1, second.bs1)


def test_moe_a4w4_compact_rejects_bf16_activations():
    from aiter.fused_moe_registry import FusedMoeRequest

    x, weights, ids, topk_weight = _problem(512, 384)
    w1, w2, w1_s, w2_s = _aiter_weights(*weights)
    request = FusedMoeRequest(
        hidden_states=x,
        w1=w1,
        w2=w2,
        topk_weight=topk_weight,
        topk_ids=ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=w1_s,
        w2_scale=w2_s,
        q_dtype_a=dtypes.bf16,
    )
    assert moe_a4w4_compact.unsupported_reason(request) is not None
