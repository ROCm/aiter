# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_moe's eager host path must not change what runs.

In eager mode fused_moe calls the fused_moe_ body without the torch.ops
dispatch, _fused_moe_impl reuses the plan resolved for an earlier call with the
same signature, and its own sort writes into reused buffers. Each test compares
against the path with all three off (_plain_eager patched to False), checks the
plan bookkeeping, or checks a case that must keep the original behavior
(torch.compile, dispatch modes, CUDA graph capture, a wrapped get_2stage_cfgs).

The MoE kernels add the top-k contributions with atomics, so two calls are not
bitwise equal in general. With one nonzero top-k weight per token (the others
exactly zero, and x + 0 == x) they are, and that is what the comparisons use.
"""

import dataclasses

import pytest
import torch

import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

E, D, INTER, TOPK = 64, 2048, 512, 8
DEV = "cuda"


def _quant_weight(w, qt):
    if qt == QuantType.No:
        return w, None
    fp8 = dtypes.fp8
    if qt == QuantType.per_Token:
        scale = w.abs().amax(dim=-1, keepdim=True).float() / torch.finfo(fp8).max
        return (w / scale.to(w.dtype)).to(fp8), scale
    scale = w.abs().amax().float() / torch.finfo(fp8).max  # per_Tensor
    return (w / scale.to(w.dtype)).to(fp8), scale.view(1).repeat(w.shape[0])


_WEIGHTS = {}


def _weights(qt):
    if qt not in _WEIGHTS:
        g = torch.Generator(device=DEV).manual_seed(1)
        w1 = torch.randn(E, 2 * INTER, D, device=DEV, generator=g) * 0.05
        w2 = torch.randn(E, D, INTER, device=DEV, generator=g) * 0.05
        w1, s1 = _quant_weight(w1.to(torch.bfloat16), qt)
        w2, s2 = _quant_weight(w2.to(torch.bfloat16), qt)
        w1 = shuffle_weight(w1, layout=(16, 16))
        w2 = shuffle_weight(w2, layout=(16, 16))
        w1.is_shuffled = True
        w2.is_shuffled = True
        _WEIGHTS[qt] = (w1, w2, s1, s2)
    return _WEIGHTS[qt]


def _routes(M, ep, seed):
    """Inputs with one nonzero weight per token (on a local expert under EP)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    n_global = 2 * E if ep else E
    x = torch.randn(M, D, device=DEV, generator=g).to(torch.bfloat16)
    logits = torch.randn(M, n_global, device=DEV, generator=g)
    weights, ids = torch.topk(logits.softmax(-1), TOPK, dim=-1)
    ids = ids.to(torch.int32)
    mask = None
    local = torch.ones_like(ids, dtype=torch.bool)
    if ep:
        mask = torch.zeros(n_global, dtype=torch.int32, device=DEV)
        mask[:E] = 1
        local = ids < E
    first = torch.argmax(local.int(), dim=1)
    rows = torch.arange(M, device=DEV)
    keep = torch.zeros_like(weights)
    keep[rows, first] = weights[rows, first] * local[rows, first]
    return x, keep, ids, mask


def _call(qt, x, w, ids, mask, ntok=None, **kw):
    w1, w2, s1, s2 = _weights(qt)
    if ntok is not None:
        ntok = torch.tensor([ntok], dtype=torch.int32, device=DEV)
    return fm.fused_moe(
        x,
        w1,
        w2,
        w,
        ids,
        expert_mask=mask,
        activation=ActivationType.Silu,
        quant_type=qt,
        w1_scale=s1,
        w2_scale=s2,
        num_local_tokens=ntok,
        **kw,
    )


@pytest.fixture(autouse=True)
def _fresh_plans():
    fm._MOE_PLANS.clear()
    yield
    fm._MOE_PLANS.clear()


def _count_resolutions(monkeypatch):
    """Count the calls that resolve a plan (a cache hit resolves nothing)."""
    calls = []
    original = fm.resolve_activation_dtype

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(fm, "resolve_activation_dtype", counting)
    return calls


def _stage_names(fn):
    fm.kernel_bench_callable = []
    try:
        fn()
        return [name for name, _ in fm.kernel_bench_callable]
    finally:
        fm.kernel_bench_callable = None


# (quant, tokens, expert parallel, num_local_tokens): the asm 1-stage and the
# CK 2-stage pipelines, EP masking, and rows past num_local_tokens (padding).
CASES = [
    (QuantType.per_Token, 300, False, None),
    (QuantType.per_Token, 300, True, None),
    (QuantType.per_Token, 300, True, 300),
    (QuantType.per_Token, 300, True, 297),
    (QuantType.per_Token, 7, False, None),
    (QuantType.per_Token, 7, True, 5),
    (QuantType.No, 64, False, None),
    (QuantType.per_Tensor, 64, True, None),
]


@pytest.mark.parametrize("qt,M,ep,ntok", CASES)
def test_bitwise_equal_to_original_path(monkeypatch, qt, M, ep, ntok):
    x, w, ids, mask = _routes(M, ep, seed=M)
    with monkeypatch.context() as m:
        m.setattr(fm, "_plain_eager", lambda t: False)
        ref = _call(qt, x, w, ids, mask, ntok).clone()
        assert not fm._MOE_PLANS
    rows = M if ntok is None else ntok  # rows past it are padding nobody writes
    miss = _call(qt, x, w, ids, mask, ntok).clone()
    assert len(fm._MOE_PLANS) == 1
    resolved = _count_resolutions(monkeypatch)
    # another shape in between: another plan, other views of the sort buffers
    xo, wo, ido, masko = _routes(37, ep, seed=1)
    _call(qt, xo, wo, ido, masko)
    hit = _call(qt, x, w, ids, mask, ntok).clone()
    assert len(resolved) == 1  # the 37-token call resolved; the repeat did not
    torch.cuda.synchronize()
    assert torch.equal(miss[:rows], ref[:rows])
    assert torch.equal(hit[:rows], ref[:rows])


def test_plan_key_separates_what_resolution_reads(monkeypatch):
    qt = QuantType.per_Token
    x, w, ids, mask = _routes(300, True, seed=0)
    _call(qt, x, w, ids, mask)
    _call(qt, x, w, ids, mask)
    assert len(fm._MOE_PLANS) == 1
    _call(qt, x, w, ids, mask, ntok=300)  # has_num_local_tokens
    _call(qt, x, w, ids, None)  # is_ep
    _call(qt, x[:299], w[:299], ids[:299], mask)  # tokens
    _call(qt, x, w, ids, mask, swiglu_limit=0.0)  # another limit argument
    assert len(fm._MOE_PLANS) == 5
    resolved = _count_resolutions(monkeypatch)
    _call(qt, x, w, ids, mask, ntok=300)
    _call(qt, x, w, ids, None)
    assert not resolved
    out = torch.empty_like(x)
    assert _call(qt, x, w, ids, mask, output=out) is out
    assert len(resolved) == 0  # a caller output is checked per call, not keyed


def test_plan_cache_is_bounded(monkeypatch):
    monkeypatch.setattr(fm, "_MOE_PLANS_MAX", 2)
    qt = QuantType.per_Token
    for M in (64, 65, 66):
        x, w, ids, mask = _routes(M, False, seed=M)
        _call(qt, x, w, ids, mask)
    assert len(fm._MOE_PLANS) == 1


def test_wrapped_get_2stage_cfgs_is_asked_every_call(monkeypatch):
    """A caller that wraps get_2stage_cfgs (ATOM's SGLang plugin does) sees
    every lookup, and a plan never outlives the config it was resolved from."""
    qt = QuantType.per_Token
    x, w, ids, mask = _routes(7, True, seed=7)
    original = fm.get_2stage_cfgs
    tokens_asked = []

    def asm_for_decode(token, *args, **kwargs):
        tokens_asked.append(token)
        # Like the plugin: re-ask above the decode cutoff to get the asm 1-stage.
        return original(max(token, 17), *args, **kwargs)

    assert _stage_names(lambda: _call(qt, x, w, ids, mask)) == ["stage1", "stage2"]
    monkeypatch.setattr(fm, "get_2stage_cfgs", asm_for_decode)
    for _ in range(3):
        assert _stage_names(lambda: _call(qt, x, w, ids, mask)) == ["stage1"]
    # The first call finds the cached plan stale (one lookup) and resolves
    # again (another); the next two reuse the new plan after one lookup each.
    assert len(tokens_asked) == 4
    monkeypatch.setattr(fm, "get_2stage_cfgs", original)
    assert _stage_names(lambda: _call(qt, x, w, ids, mask)) == ["stage1", "stage2"]

    # A wrapper that builds a new config object per call is never served a plan.
    monkeypatch.setattr(
        fm, "get_2stage_cfgs", lambda *a, **k: dataclasses.replace(original(*a, **k))
    )
    resolved = _count_resolutions(monkeypatch)
    for _ in range(3):
        _call(qt, x, w, ids, mask)
    assert len(resolved) == 3


def test_get_2stage_cfgs_cache_clear_is_honored(monkeypatch):
    qt = QuantType.per_Token
    x, w, ids, mask = _routes(300, False, seed=3)
    _call(qt, x, w, ids, mask)
    (plan,) = fm._MOE_PLANS.values()
    resolved = _count_resolutions(monkeypatch)
    fm.get_2stage_cfgs.cache_clear()
    _call(qt, x, w, ids, mask)
    assert len(resolved) == 1
    (replanned,) = fm._MOE_PLANS.values()
    assert replanned.metadata is not plan.metadata


def test_plain_eager_only_for_real_eager_tensors(monkeypatch):
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.utils._python_dispatch import TorchDispatchMode

    x = torch.zeros(2, device=DEV)
    assert fm._plain_eager(x)
    assert fm._op_or_eager_body(fm.fused_moe_, x) is fm.fused_moe_._eager_body

    class Passthrough(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    with Passthrough():
        assert not fm._plain_eager(x)
    with FakeTensorMode() as mode:
        fake = mode.from_tensor(x)
        assert not fm._plain_eager(fake)
    assert not fm._plain_eager(fake)  # a fake tensor outside its mode
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    assert not fm._plain_eager(x)
    assert fm._op_or_eager_body(fm.fused_moe_, x) is fm.fused_moe_


def test_torch_compile_keeps_the_custom_op():
    qt = QuantType.per_Token
    x, w, ids, mask = _routes(300, True, seed=5)
    ref = _call(qt, x, w, ids, mask).clone()
    targets = []

    def recording_backend(gm, example_inputs):
        targets.extend(str(n.target) for n in gm.graph.nodes if n.op == "call_function")
        return gm.forward

    compiled = torch.compile(
        lambda x, w, ids, mask: _call(qt, x, w, ids, mask), backend=recording_backend
    )
    try:
        out = compiled(x, w, ids, mask)
    finally:
        torch._dynamo.reset()
    torch.cuda.synchronize()
    assert any("fused_moe_" in t for t in targets), targets
    assert torch.equal(out, ref)


def test_cuda_graph_capture_and_replay(monkeypatch):
    qt = QuantType.per_Token
    x, w, ids, mask = _routes(300, True, seed=11)
    xs, ws, idss = x.clone(), w.clone(), ids.clone()
    ref = _call(qt, x, w, ids, mask).clone()  # eager first: plan and sort buffers exist
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call(qt, xs, ws, idss, mask)
    torch.cuda.current_stream().wait_stream(stream)
    scratch = [entry[0] for entry in fm._SORT_SCRATCH.values()]
    sorts = []
    original_sort = fm.moe_sorting

    def recording_sort(*args, **kwargs):
        ret = original_sort(*args, **kwargs)
        sorts.append((kwargs.get("_scratch"), ret[0]))
        return ret

    monkeypatch.setattr(fm, "moe_sorting", recording_sort)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = _call(qt, xs, ws, idss, mask)
    monkeypatch.setattr(fm, "moe_sorting", original_sort)
    # the sort asked for reused buffers, and capture allocated its own instead
    ((asked, sorted_ids),) = sorts
    assert asked
    for buf in scratch:
        lo = buf.data_ptr()
        assert not lo <= sorted_ids.data_ptr() < lo + buf.numel()
    # eager calls between replays (other routes, other sizes) write those buffers
    xo, wo, ido, _ = _routes(300, True, seed=12)
    for _ in range(3):
        graph.replay()
        _call(qt, xo, wo, ido, mask)
        _call(qt, xo[:100], wo[:100], ido[:100], mask)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, ref)


def test_two_streams_do_not_share_sort_buffers():
    qt = QuantType.per_Token
    inputs = [_routes(300, True, seed=s) for s in (21, 22)]
    refs = [_call(qt, *i).clone() for i in inputs]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for s in streams:
        s.wait_stream(torch.cuda.current_stream())
    outs = [[], []]
    for _ in range(4):
        for k, s in enumerate(streams):
            with torch.cuda.stream(s):
                outs[k].append(_call(qt, *inputs[k]))
    torch.cuda.synchronize()
    keys = {(dev, raw) for dev, raw in fm._SORT_SCRATCH}
    assert {(torch.cuda.current_device(), s.cuda_stream) for s in streams} <= keys
    for k in range(2):
        for o in outs[k]:
            assert torch.equal(o, refs[k])


def test_bench_hook_gets_its_own_sort_buffers():
    qt = QuantType.per_Token
    x, w, ids, mask = _routes(300, False, seed=31)
    _call(qt, x, w, ids, mask)
    scratch = [entry[0] for entry in fm._SORT_SCRATCH.values()]
    fm.kernel_bench_callable = []
    try:
        _call(qt, x, w, ids, mask)
        sorted_ids = fm.kernel_bench_callable[0][1].args[4]
    finally:
        fm.kernel_bench_callable = None
    for buf in scratch:
        lo = buf.data_ptr()
        assert not lo <= sorted_ids.data_ptr() < lo + buf.numel()


@pytest.mark.parametrize("M", [1, 7, 300, 4096])
@pytest.mark.parametrize("ep", [False, True])
def test_moe_sorting_scratch_matches(M, ep):
    _, w, ids, mask = _routes(M, ep, seed=M)
    n_global = 2 * E if ep else E
    args = (ids, w, n_global, D, torch.bfloat16, 32, mask)
    ref = fm.moe_sorting(*args)
    got = fm.moe_sorting(*args, _scratch=True)
    torch.cuda.synchronize()
    valid = int(ref[3][0])
    assert torch.equal(got[3], ref[3])
    assert torch.equal(got[0][:valid], ref[0][:valid])
    assert torch.equal(got[1][:valid], ref[1][:valid])
    assert torch.equal(got[2][: valid // 32], ref[2][: valid // 32])
    assert torch.equal(got[4], ref[4])  # moe_buf: zeroed output rows


@pytest.mark.parametrize("policy", [0, 1, 2])
def test_workspace_size_is_the_ops(policy):
    dev = torch.cuda.current_device()
    for tokens in (1, 7, 64, 2112, 16384):
        for experts, topk in ((64, 8), (257, 9), (514, 12)):
            assert fm._moe_sorting_opus_workspace_size(
                tokens, experts, topk, policy, dev
            ) == fm.aiter.moe_sorting_opus_get_workspace_size(
                tokens, experts, topk, policy, dev
            )
