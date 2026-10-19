# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Sort-folded BM16 MXFP4 GEMM1 for the GLM-5.2 TP8 routed MoE (gfx950).

Inside its guard, fused_moe skips the BM16 aux sort launch and the GEMM1 head
builds the sort tables itself (aiter/ops/flydsl/kernels/mxfp4_gemm1_sort_fold.py).
These tests compare it with the unfolded path of the same build
(``fused_moe._mxfp4_sort_fold_disabled``) and with a torch reference, under
both stage-2 epilogues and CUDA graphs, and check that every call outside the
guard launches exactly the kernels it launched before.
"""

import contextlib
from pathlib import Path

import pytest
import torch

import aiter
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, torch_moe_stage1, torch_moe_stage2
from aiter.ops.flydsl import mxfp4_gemm1_kernels
from aiter.ops.shuffle import shuffle_weight
from aiter.utility import fp4_utils


def _on_gfx950():
    try:
        from aiter.jit.utils.chip_info import get_gfx_runtime

        return torch.cuda.is_available() and get_gfx_runtime() == "gfx950"
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _on_gfx950(), reason="MXFP4 MoE needs gfx950")

MODEL_DIM = 6144
TOPK = 9
SHARED_EXPERT = 256
GLM5_CSV = (
    Path(__file__).resolve().parents[1]
    / "aiter/configs/model_configs/glm5_fp4_tuned_fmoe.csv"
)
FOLD_M = [*range(5, 17), *range(33, 65)]
ALL_M = list(range(1, 65))
EDGE_M = [1, 4, 5, 7, 8, 12, 16, 17, 32, 33, 39, 48, 64]
# 1 - cosine similarity against the torch reference; the kernels land near
# 6e-6, a misrouted row moves it to 1e-2.
REF_COS_DIFF = 1e-4


class _Weights:
    """GLM-5.2-style MXFP4 expert weights: unshuffled (reference) and preshuffled."""

    def __init__(self, inter_dim, experts=SHARED_EXPERT + 1, seed=0):
        gen = torch.Generator(device="cuda").manual_seed(seed)
        torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
        self.inter_dim = inter_dim
        self.experts = experts
        w1 = torch.randn(
            (experts, 2 * inter_dim, MODEL_DIM),
            generator=gen,
            device="cuda",
            dtype=dtypes.bf16,
        )
        w2 = torch.randn(
            (experts, MODEL_DIM, inter_dim),
            generator=gen,
            device="cuda",
            dtype=dtypes.bf16,
        )
        w1_qt, self.w1_scale = torch_quant(w1, quant_dtype=dtypes.fp4x2)
        w2_qt, self.w2_scale = torch_quant(w2, quant_dtype=dtypes.fp4x2)
        del w1, w2
        self.w1_qt = w1_qt.view(experts, 2 * inter_dim, MODEL_DIM // 2)
        self.w2_qt = w2_qt.view(experts, MODEL_DIM, inter_dim // 2)
        self.w1 = shuffle_weight(self.w1_qt, layout=(16, 16))
        self.w2 = shuffle_weight(self.w2_qt, layout=(16, 16))
        self.w1_scale_shuf = fp4_utils.e8m0_shuffle(self.w1_scale)
        self.w2_scale_shuf = fp4_utils.e8m0_shuffle(self.w2_scale)


@pytest.fixture(scope="module")
def tp8():
    return _Weights(inter_dim=256)


@pytest.fixture(scope="module")
def tp4():
    return _Weights(inter_dim=512, seed=1)


def _inputs(M, routing, seed, experts=SHARED_EXPERT + 1):
    """8 routed experts plus the fused shared expert in the last slot."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    routed = experts - 1
    hidden = torch.randn((M, MODEL_DIM), generator=gen, device="cuda")
    hidden = hidden.to(dtypes.bf16)
    ids = torch.argsort(torch.rand((M, routed), generator=gen, device="cuda"), dim=-1)[
        :, : TOPK - 1
    ]
    if routing == "collide":
        # every token picks the same experts: one expert owns several m-blocks
        ids = ids[:1].expand(M, -1)
    elif routing == "dup":
        # repeated expert ids inside a row
        ids = ids.clone()
        ids[:, 1] = ids[:, 0]
        ids[:, 3] = ids[:, 2]
    elif routing == "pad":
        # CUDA-graph padding rows: zero hidden state, every slot on expert 0
        ids = ids.clone()
        pad = M // 2
        ids[M - pad :] = 0
        hidden[M - pad :] = 0
    weights = torch.softmax(
        torch.randn((M, TOPK - 1), generator=gen, device="cuda"), dim=-1
    )
    ids = torch.cat(
        [ids, torch.full((M, 1), experts - 1, device="cuda", dtype=ids.dtype)], -1
    )
    if routing == "pad":
        ids[M - M // 2 :, -1] = 0
    weights = torch.cat([weights, torch.ones((M, 1), device="cuda")], -1)
    return hidden, weights.float().contiguous(), ids.to(dtypes.i32).contiguous()


def _call(w, hidden, weights, ids):
    return fused_moe(
        hidden,
        w.w1,
        w.w2,
        weights,
        ids,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Silu,
        w1_scale=w.w1_scale_shuf,
        w2_scale=w.w2_scale_shuf,
        gate_mode="separated",
        moe_sorting_dispatch_policy=2,
    )


def _torch_ref(w, hidden, weights, ids):
    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    a1, a1_scale = torch_quant(hidden, quant_dtype=dtypes.fp4x2)
    out1 = torch_moe_stage1(
        a1,
        w.w1_qt,
        w.w2_qt,
        weights,
        ids,
        dtype=dtypes.fp32,
        activation=ActivationType.Silu,
        quant_type=QuantType.per_1x32,
        a1_scale=a1_scale,
        w1_scale=w.w1_scale,
    )
    a2, a2_scale = torch_quant(out1, quant_dtype=dtypes.fp4x2)
    return torch_moe_stage2(
        a2.view(hidden.shape[0], TOPK, -1),
        w.w1_qt,
        w.w2_qt,
        weights,
        ids,
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=w.w2_scale,
        a2_scale=a2_scale,
        doweight=True,
    )


def _rel(a, b):
    a, b = a.float(), b.float()
    return float((a - b).norm() / b.norm().clamp_min(1e-12))


def _cos_diff(a, b):
    a, b = a.double(), b.double()
    return float(1 - 2 * (a * b).sum() / (a * a + b * b).sum())


@contextlib.contextmanager
def _fold_disabled(disabled):
    old = fm._mxfp4_sort_fold_disabled
    fm._mxfp4_sort_fold_disabled = disabled
    try:
        yield
    finally:
        fm._mxfp4_sort_fold_disabled = old


def _use_config(monkeypatch, csv_path):
    monkeypatch.setenv("AITER_CONFIG_FMOE", str(csv_path))
    monkeypatch.setattr(fm, "cfg_2stages", None)
    fm.AITER_CONFIGS.get_config_file.cache_clear()
    fm.get_2stage_cfgs.cache_clear()


@pytest.fixture(params=["atomic", "reduce"])
def epilogue(request, monkeypatch, tmp_path):
    """Tuned GLM-5 rows as shipped (atomic stage 2), or with every TP8 BM16
    stage-2 row switched to its deterministic reduce sibling."""
    if request.param == "atomic":
        _use_config(monkeypatch, GLM5_CSV)
    else:
        lines = GLM5_CSV.read_text().splitlines()
        out = [lines[0]]
        for line in lines[1:]:
            f = line.split(",")
            if (
                f[3:7] == ["6144", "256", "257", "9"]
                and f[17].startswith("flydsl_mxmoe_g1")
                and "_atomic" in f[20]
            ):
                f[20] = f[20].replace("_atomic", "_reduce")
            out.append(",".join(f))
        csv = tmp_path / "glm5_fp4_reduce_fmoe.csv"
        csv.write_text("\n".join(out) + "\n")
        _use_config(monkeypatch, csv)
    yield request.param
    fm.AITER_CONFIGS.get_config_file.cache_clear()
    fm.get_2stage_cfgs.cache_clear()


@pytest.fixture
def fold_calls(monkeypatch):
    calls = []
    real = mxfp4_gemm1_kernels.flydsl_mxfp4_gemm1_sort_fold

    def spy(**kwargs):
        calls.append(kwargs["hidden_states"].shape[0])
        return real(**kwargs)

    monkeypatch.setattr(mxfp4_gemm1_kernels, "flydsl_mxfp4_gemm1_sort_fold", spy)
    return calls


def _tp8_metadata(M, inter_dim=256):
    return fm.get_2stage_cfgs(
        fm.get_padded_M(M),
        MODEL_DIM,
        inter_dim,
        SHARED_EXPERT + 1,
        TOPK,
        dtypes.bf16,
        dtypes.fp4x2,
        dtypes.fp4x2,
        QuantType.per_1x32,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        "separated",
        opus_weights_shuffled=True,
    )


def _stage2_is_reduce(M):
    stage2 = _tp8_metadata(M).stage2
    name = stage2.keywords.get("kernelName2") or stage2.keywords.get("kernelName")
    return "_reduce" in name


def _check_against_main(w, hidden, weights, ids, fold_calls, main_ok=True):
    M = hidden.shape[0]
    with _fold_disabled(True):
        mains = [_call(w, hidden, weights, ids) for _ in range(3)]
    assert not fold_calls
    folds = [_call(w, hidden, weights, ids) for _ in range(2)]
    torch.cuda.synchronize()
    assert len(fold_calls) == (2 if M in FOLD_M else 0)
    for out in folds:
        assert torch.isfinite(out).all()
    if _stage2_is_reduce(M):
        # Deterministic stage 2: the fold must reproduce it bit for bit.
        assert torch.equal(folds[0], folds[1])
        if main_ok:
            assert all(torch.equal(mains[0], o) for o in mains[1:] + folds)
    elif main_ok:
        # Atomic stage 2 is order-nondeterministic in bf16; stay within the
        # spread main shows against itself.
        noise = max(_rel(mains[0], mains[1]), _rel(mains[0], mains[2]))
        bound = max(2 * noise, 1e-2)
        for out in folds:
            assert _rel(out, mains[0]) <= bound, (M, _rel(out, mains[0]), noise)
    return folds[0]


@pytest.mark.parametrize("M", ALL_M)
def test_fold_matches_main_and_torch(tp8, epilogue, fold_calls, M):
    hidden, weights, ids = _inputs(M, "uniform", seed=M)
    out = _check_against_main(tp8, hidden, weights, ids, fold_calls)
    ref = _torch_ref(tp8, hidden, weights, ids)
    assert _cos_diff(out, ref) < REF_COS_DIFF


@pytest.mark.parametrize("routing", ["dup", "pad", "collide"])
@pytest.mark.parametrize("M", EDGE_M)
def test_fold_routing_edge_cases(tp8, epilogue, fold_calls, routing, M):
    hidden, weights, ids = _inputs(M, routing, seed=1000 + M)
    # Main's Opus aux sort (M >= 39 here) mishandles an expert repeated inside
    # a row; there the fold is checked against the torch reference only.
    main_ok = routing not in ("dup", "pad") or not fm._aux_uses_opus(
        fm.AUX_SORT_OPUS, 16, M * TOPK, SHARED_EXPERT + 1
    )
    out = _check_against_main(tp8, hidden, weights, ids, fold_calls, main_ok)
    ref = _torch_ref(tp8, hidden, weights, ids)
    if routing == "pad":
        assert torch.equal(out[M - M // 2 :], torch.zeros_like(out[M - M // 2 :]))
    # Outside the fold this is main's own handling of repeated ids, which the
    # comparison above already pins; the reference check is for the fold.
    if M in FOLD_M or routing != "dup":
        assert _cos_diff(out, ref) < REF_COS_DIFF


@pytest.mark.parametrize(
    ("M", "routing"),
    [(5, "dup"), (8, "dup"), (16, "dup"), (33, "dup"), (33, "collide")]
    + [(48, "uniform"), (64, "collide")],
)
def test_fold_sort_tables_match_aux_sort(tp8, epilogue, M, routing):
    """The published sort tables equal the aux sort's, block by block."""
    hidden, weights, ids = _inputs(M, routing, seed=7 * M)
    tables = {}
    for name, disabled in (("main", True), ("fold", False)):
        seen = []
        real = fm.fused_moe_2stages

        def spy(*args, **kwargs):
            seen.append(args[4:8])
            return real(*args, **kwargs)

        fm.fused_moe_2stages = spy
        try:
            with _fold_disabled(disabled):
                _call(tp8, hidden, weights, ids)
        finally:
            fm.fused_moe_2stages = real
        torch.cuda.synchronize()
        sorted_ids, sorted_weights, sorted_experts, num_valid = seen[0]
        rows = int(num_valid[0])
        assert int(num_valid[1]) == M
        blocks = {}
        for b in range(rows // 16):
            key = int(sorted_experts[b])
            span = slice(16 * b, 16 * b + 16)
            routes = sorted(
                zip(sorted_ids[span].tolist(), sorted_weights[span].tolist())
            )
            blocks.setdefault(key, []).extend(
                r for r in routes if (r[0] & 0xFFFFFF) < M
            )
        tables[name] = (rows, {k: sorted(v) for k, v in blocks.items()})
    assert tables["fold"] == tables["main"]


@pytest.mark.parametrize("M", [5, 6, 7, 12, 36, 64])
def test_fold_cuda_graph_replay(tp8, epilogue, fold_calls, M):
    hidden, weights, ids = _inputs(M, "uniform", seed=50 + M)
    static = [hidden.clone(), weights.clone(), ids.clone()]
    # Warm up on the capture stream, as serving frameworks do, so per-stream
    # scratch exists before capture.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call(tp8, *static)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = _call(tp8, *static)
    assert M not in FOLD_M or fold_calls
    for step in range(4):
        h, wts, i = _inputs(M, "collide" if step % 2 else "uniform", seed=500 + step)
        static[0].copy_(h)
        static[1].copy_(wts)
        static[2].copy_(i)
        graph.replay()
        torch.cuda.synchronize()
        with _fold_disabled(True):
            ref = _call(tp8, h, wts, i)
        if _stage2_is_reduce(M):
            assert torch.equal(out, ref)
        else:
            assert _rel(out, ref) < 1e-2


def _kernel_names(fn):
    from torch.profiler import ProfilerActivity, profile

    fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted(
        e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    )


@pytest.mark.parametrize("M", [1, 2, 3, 4, 17, 24, 32, 65, 128])
def test_outside_m_range_runs_main_kernels(tp8, epilogue, fold_calls, M):
    hidden, weights, ids = _inputs(M, "uniform", seed=M)
    folded = _kernel_names(lambda: _call(tp8, hidden, weights, ids))
    with _fold_disabled(True):
        main = _kernel_names(lambda: _call(tp8, hidden, weights, ids))
    assert not fold_calls
    assert folded == main
    assert not any("sort_fold" in name for name in folded)


@pytest.mark.parametrize("M", [6, 12, 48])
def test_inside_guard_replaces_sort_launch(tp8, epilogue, M):
    hidden, weights, ids = _inputs(M, "uniform", seed=M)
    folded = _kernel_names(lambda: _call(tp8, hidden, weights, ids))
    with _fold_disabled(True):
        main = _kernel_names(lambda: _call(tp8, hidden, weights, ids))
    assert any("sort_fold" in name for name in folded)
    assert not any("sort" in name and "sort_fold" not in name for name in folded)
    assert any("sort" in name for name in main)
    gemm2 = [n for n in main if "gemm2" in n or "moe_reduction" in n]
    assert gemm2 and all(n in folded for n in gemm2)


@pytest.mark.parametrize("M", [6, 12, 48])
def test_other_model_dims_run_main_kernels(tp4, epilogue, fold_calls, M):
    """GLM-5 TP4 (inter_dim 512) uses the same tuned GEMM1 name but stays out."""
    w = tp4
    hidden, weights, ids = _inputs(M, "uniform", seed=M)
    folded = _kernel_names(lambda: _call(w, hidden, weights, ids))
    with _fold_disabled(True):
        main = _kernel_names(lambda: _call(w, hidden, weights, ids))
    assert not fold_calls
    assert folded == main


def test_guard_needs_gfx950(tp8, monkeypatch):
    _use_config(monkeypatch, GLM5_CSV)
    hidden, weights, ids = _inputs(6, "uniform", seed=6)
    meta = _tp8_metadata(6)
    kwargs = dict(
        model_dim=MODEL_DIM,
        inter_dim=256,
        global_E=257,
        output=None,
        swiglu_limit=None,
    )
    args = (meta, hidden, tp8.w1, ids, weights)
    assert fm._use_mxfp4_sort_fold(*args, **kwargs)
    for arch in ("gfx942", "gfx1250"):
        monkeypatch.setattr(fm, "get_gfx_runtime", lambda arch=arch: arch)
        assert not fm._use_mxfp4_sort_fold(*args, **kwargs)
    monkeypatch.setattr(fm, "get_gfx_runtime", lambda: "gfx950")
    assert not fm._use_mxfp4_sort_fold(*args, **{**kwargs, "swiglu_limit": 7.0})
    assert not fm._use_mxfp4_sort_fold(
        *args, **{**kwargs, "output": torch.empty_like(hidden)}
    )
    fm.get_2stage_cfgs.cache_clear()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
