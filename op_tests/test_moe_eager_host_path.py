# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_moe's eager host path must not change what runs.

In plain eager mode fused_moe calls the fused_moe_ body without the torch.ops
dispatch, _fused_moe_impl reuses the plan resolved for an earlier call with the
same signature, and its own sort writes into reused buffers. The sweep runs
each case both ways, "eager" and "op" (all three off: _plain_eager false, as
under a tracer), checks them bit for bit against each other and against the
torch reference, and times them. The checks before it cover the plan
bookkeeping and the cases that keep the original behavior: torch.compile,
dispatch modes, CUDA graph capture, a wrapped get_2stage_cfgs, two streams and
the kernel bench hook.

The MoE kernels add the top-k contributions with atomics, so two calls are not
bitwise equal in general. With one nonzero top-k weight per token (the others
exactly zero, and x + 0 == x) they are, and that is what the comparisons use.
"""

import argparse
import contextlib
import dataclasses
import itertools
import time
from unittest import mock

import pandas as pd
import torch

import aiter
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_weight
from aiter.test_common import benchmark, run_perftest

SUPPORTED_GFX = ["gfx942", "gfx950"]
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
    """`(w1, w2, w1_scale, w2_scale)` as the kernels take them (shuffled), and
    the dequantized, unshuffled pair the torch reference takes."""
    if qt not in _WEIGHTS:
        g = torch.Generator(device=DEV).manual_seed(1)
        w1 = torch.randn(E, 2 * INTER, D, device=DEV, generator=g) * 0.05
        w2 = torch.randn(E, D, INTER, device=DEV, generator=g) * 0.05
        kernel, scales, ref = [], [], []
        for w in (w1.to(dtypes.bf16), w2.to(dtypes.bf16)):
            q, scale = _quant_weight(w, qt)
            ref.append(q.float() if scale is None else q.float() * scale.view(E, -1, 1))
            q = shuffle_weight(q, layout=(16, 16))
            q.is_shuffled = True
            kernel.append(q)
            scales.append(scale)
        _WEIGHTS[qt] = (*kernel, *scales), tuple(ref)
    return _WEIGHTS[qt]


def _routes(M, ep, seed):
    """Inputs with one nonzero weight per token (on a local expert under EP)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    n_global = 2 * E if ep else E
    x = torch.randn(M, D, device=DEV, generator=g).to(dtypes.bf16)
    logits = torch.randn(M, n_global, device=DEV, generator=g)
    weights, ids = torch.topk(logits.softmax(-1), TOPK, dim=-1)
    ids = ids.to(dtypes.i32)
    mask = None
    local = torch.ones_like(ids, dtype=torch.bool)
    if ep:
        mask = torch.zeros(n_global, dtype=dtypes.i32, device=DEV)
        mask[:E] = 1
        local = ids < E
    first = torch.argmax(local.int(), dim=1)
    rows = torch.arange(M, device=DEV)
    keep = torch.zeros_like(weights)
    keep[rows, first] = weights[rows, first] * local[rows, first]
    return x, keep, ids, mask


def _call(qt, x, w, ids, mask, ntok=None, **kw):
    w1, w2, s1, s2 = _weights(qt)[0]
    if ntok is not None:
        ntok = torch.tensor([ntok], dtype=dtypes.i32, device=DEV)
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


def run_torch(qt, x, w, ids, mask):
    w1, w2 = _weights(qt)[1]
    return fm.torch_moe(x, w1, w2, w, ids, expert_mask=mask)


def _row_err(out, ref):
    """Largest per-row relative L2 error. A row the routing leaves empty must
    come out exactly zero (0/0 counts as 0, anything else over 0 as inf)."""
    out, ref = out.to(dtypes.fp32), ref.to(dtypes.fp32)
    rel = (out - ref).norm(dim=-1) / ref.norm(dim=-1)
    return rel.nan_to_num(nan=0.0).max().item()


def _fresh():
    fm._MOE_PLANS.clear()


def _original_path():
    """All three shortcuts off, as when fused_moe is not plain eager."""
    return mock.patch.object(fm, "_plain_eager", lambda t: False)


@contextlib.contextmanager
def _resolutions():
    """Yields a list that gets one entry per call that resolves a plan (a cache
    hit resolves nothing)."""
    calls = []
    original = fm.resolve_activation_dtype

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    with mock.patch.object(fm, "resolve_activation_dtype", counting):
        yield calls


def _stage_names(fn):
    fm.kernel_bench_callable = []
    try:
        fn()
        return [name for name, _ in fm.kernel_bench_callable]
    finally:
        fm.kernel_bench_callable = None


def _host_us(fn, iters=50):
    """Host time to issue one call; the GPU is not waited on in between."""
    fn()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    elapsed = time.perf_counter() - start
    torch.cuda.synchronize()
    return elapsed / iters * 1e6


@benchmark()
def test_fused_moe_host_path(quant_type, tokens, ep, pad):
    """`pad`: None calls without num_local_tokens; otherwise num_local_tokens
    is `tokens - pad`, and the `pad` rows past it are padding nobody writes."""
    x, w, ids, mask = _routes(tokens, ep, seed=tokens)
    ntok = None if pad is None else tokens - pad
    rows = tokens if ntok is None else ntok

    def run():
        return _call(quant_type, x, w, ids, mask, ntok)

    _fresh()
    with _original_path():
        ref = run().clone()
    assert not fm._MOE_PLANS
    miss = run().clone()
    assert len(fm._MOE_PLANS) == 1
    with _resolutions() as resolved:
        # another shape in between: another plan, other views of the sort buffers
        _call(quant_type, *_routes(37, ep, seed=1))
        hit = run().clone()
    assert len(resolved) == 1  # the 37-token call resolved; the repeat did not
    torch.cuda.synchronize()
    assert torch.equal(miss[:rows], ref[:rows])
    assert torch.equal(hit[:rows], ref[:rows])

    expected = run_torch(quant_type, x, w, ids, mask)[:rows]
    # The kernels round the activations to the weights' dtype before each GEMM
    # (fp8 e4m3 keeps 3 mantissa bits, bf16 7); the reference keeps them in
    # fp32. Single elements near zero move too much for a per-element bound,
    # so rows are compared: a zeroed or misrouted row is off by about 1.
    tol = 0.01 if quant_type == QuantType.No else 0.1
    # Routes the kernels compute (local, in the rows before num_local_tokens),
    # and the weights of the experts they touch.
    routes = ids[:rows] if mask is None else ids[:rows][ids[:rows] < E]
    w1 = _weights(quant_type)[0][0]
    flops = 2 * routes.numel() * 3 * INTER * D
    nbytes = routes.unique().numel() * 3 * INTER * D * w1.element_size()
    nbytes += 2 * rows * D * x.element_size()
    # Names instead of the enum's int and pandas' NaN in the table.
    ret = {"quant_type": str(quant_type), "pad": str(pad), "gfx": get_gfx_runtime()}
    for name, path in (("op", _original_path), ("eager", contextlib.nullcontext)):
        with path():
            out, us = run_perftest(run)
            host_us = _host_us(run)
        err = _row_err(out[:rows], expected)
        assert err < tol, f"{name}: row relative error {err:.4f} >= {tol}"
        ret[f"{name} us"] = us
        ret[f"{name} host us"] = host_us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def test_plan_key_separates_what_resolution_reads():
    _fresh()
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
    with _resolutions() as resolved:
        _call(qt, x, w, ids, mask, ntok=300)
        _call(qt, x, w, ids, None)
        assert not resolved
        out = torch.empty_like(x)
        assert _call(qt, x, w, ids, mask, output=out) is out
        assert not resolved  # a caller output is checked per call, not keyed


def test_plan_cache_is_bounded():
    _fresh()
    qt = QuantType.per_Token
    with mock.patch.object(fm, "_MOE_PLANS_MAX", 2):
        for M in (64, 65, 66):
            x, w, ids, mask = _routes(M, False, seed=M)
            _call(qt, x, w, ids, mask)
    assert len(fm._MOE_PLANS) == 1


def test_wrapped_get_2stage_cfgs_is_asked_every_call():
    """A framework that wraps get_2stage_cfgs sees every lookup, and a plan
    never outlives the config it was resolved from."""
    _fresh()
    qt = QuantType.per_Token
    x, w, ids, mask = _routes(7, True, seed=7)
    original = fm.get_2stage_cfgs
    tokens_asked = []

    def asm_for_decode(token, *args, **kwargs):
        tokens_asked.append(token)
        # Re-ask above the decode cutoff, so decode gets the asm 1-stage.
        return original(max(token, 17), *args, **kwargs)

    assert _stage_names(lambda: _call(qt, x, w, ids, mask)) == ["stage1", "stage2"]
    with mock.patch.object(fm, "get_2stage_cfgs", asm_for_decode):
        for _ in range(3):
            assert _stage_names(lambda: _call(qt, x, w, ids, mask)) == ["stage1"]
    # The first call finds the cached plan stale (one lookup) and resolves
    # again (another); the next two reuse the new plan after one lookup each.
    assert len(tokens_asked) == 4
    assert _stage_names(lambda: _call(qt, x, w, ids, mask)) == ["stage1", "stage2"]

    # A wrapper that builds a new config object per call is never served a plan.
    def new_object(*args, **kwargs):
        return dataclasses.replace(original(*args, **kwargs))

    with mock.patch.object(
        fm, "get_2stage_cfgs", new_object
    ), _resolutions() as resolved:
        for _ in range(3):
            _call(qt, x, w, ids, mask)
    assert len(resolved) == 3


def test_get_2stage_cfgs_cache_clear_is_honored():
    _fresh()
    qt = QuantType.per_Token
    x, w, ids, mask = _routes(300, False, seed=3)
    _call(qt, x, w, ids, mask)
    (plan,) = fm._MOE_PLANS.values()
    with _resolutions() as resolved:
        fm.get_2stage_cfgs.cache_clear()
        _call(qt, x, w, ids, mask)
    assert len(resolved) == 1
    (replanned,) = fm._MOE_PLANS.values()
    assert replanned.metadata is not plan.metadata


def test_plain_eager_only_for_real_eager_tensors():
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
    with mock.patch.object(torch.compiler, "is_compiling", lambda: True):
        assert not fm._plain_eager(x)
        assert fm._op_or_eager_body(fm.fused_moe_, x) is fm.fused_moe_


def test_torch_compile_keeps_the_custom_op():
    _fresh()
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


def test_cuda_graph_capture_and_replay():
    _fresh()
    qt = QuantType.per_Token
    x, w, ids, mask = _routes(300, True, seed=11)
    xs, ws, idss = x.clone(), w.clone(), ids.clone()
    ref = _call(qt, x, w, ids, mask).clone()  # eager first: plan and sort buffers exist
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        _call(qt, xs, ws, idss, mask)
    torch.cuda.current_stream().wait_stream(stream)
    sorts = []
    original_sort = fm.moe_sorting

    def recording_sort(*args, **kwargs):
        ret = original_sort(*args, **kwargs)
        sorts.append((kwargs.get("_scratch"), ret[0]))
        return ret

    graph = torch.cuda.CUDAGraph()
    with mock.patch.object(fm, "moe_sorting", recording_sort), torch.cuda.graph(graph):
        out = _call(qt, xs, ws, idss, mask)
    # the sort asked for reused buffers, and capture allocated its own instead
    # (on its own stream too: no scratch buffer holds what the graph recorded)
    ((asked, sorted_ids),) = sorts
    assert asked
    for buf, _ in fm._SORT_SCRATCH.values():
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
    _fresh()
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
    _fresh()
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


def test_moe_sorting_scratch_matches():
    for M, ep in itertools.product((1, 7, 300, 4096), (False, True)):
        _, w, ids, mask = _routes(M, ep, seed=M)
        n_global = 2 * E if ep else E
        args = (ids, w, n_global, D, dtypes.bf16, 32, mask)
        ref = fm.moe_sorting(*args)
        got = fm.moe_sorting(*args, _scratch=True)
        torch.cuda.synchronize()
        valid = int(ref[3][0])
        assert torch.equal(got[3], ref[3])
        assert torch.equal(got[0][:valid], ref[0][:valid])
        assert torch.equal(got[1][:valid], ref[1][:valid])
        assert torch.equal(got[2][: valid // 32], ref[2][: valid // 32])
        assert torch.equal(got[4], ref[4])  # moe_buf: zeroed output rows


def test_workspace_size_is_the_ops():
    dev = torch.cuda.current_device()
    for policy, tokens, (experts, topk) in itertools.product(
        (0, 1, 2), (1, 7, 64, 2112, 16384), ((64, 8), (257, 9), (514, 12))
    ):
        assert fm._moe_sorting_opus_workspace_size(
            tokens, experts, topk, policy, dev
        ) == aiter.moe_sorting_opus_get_workspace_size(
            tokens, experts, topk, policy, dev
        )


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "fused_moe eager host path: unsupported on %s; skipping",
            get_gfx_runtime(),
        )
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument(
        "-q",
        "--quant",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[QuantType.per_Token, QuantType.per_Tensor, QuantType.No],
        help="QuantType of the weights. e.g.: -q per_Token No",
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[7, 300],
        help="Tokens per call.",
    )
    parser.add_argument(
        "-e",
        "--ep",
        type=dtypes.str2bool,
        nargs="*",
        default=[False, True],
        help="Expert parallel: half of the global experts are local.",
    )
    parser.add_argument(
        "-p",
        "--pad",
        type=lambda s: None if s.lower() == "none" else int(s),
        nargs="*",
        default=[None, 0, 3],
        help="Rows past num_local_tokens (none: no num_local_tokens).",
    )
    args = parser.parse_args()

    checks = [
        test_plan_key_separates_what_resolution_reads,
        test_plan_cache_is_bounded,
        test_wrapped_get_2stage_cfgs_is_asked_every_call,
        test_get_2stage_cfgs_cache_clear_is_honored,
        test_plain_eager_only_for_real_eager_tensors,
        test_torch_compile_keeps_the_custom_op,
        test_cuda_graph_capture_and_replay,
        test_two_streams_do_not_share_sort_buffers,
        test_bench_hook_gets_its_own_sort_buffers,
        test_moe_sorting_scratch_matches,
        test_workspace_size_is_the_ops,
    ]
    for check in checks:
        check()
        aiter.logger.info("fused_moe eager host path: %s passed", check.__name__)

    df = []
    for case in itertools.product(args.quant, args.tokens, args.ep, args.pad):
        quant_type, _, _, pad = case
        if quant_type == QuantType.per_Tensor and pad is not None:
            # per_tensor_quant_hip takes no num_rows (num_local_tokens).
            continue
        df.append(test_fused_moe_host_path(*case))
    df = pd.DataFrame(df)
    aiter.logger.info(
        "fused_moe eager host path summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
