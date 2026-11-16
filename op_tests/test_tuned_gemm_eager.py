# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The eager fast path of `tgemm.mm` / `gemm_a16w16` / `gemm_a16w16_out`.

An unscaled GEMM whose tuned config is hipBLASLt skips the custom ops outside
torch.compile and calls the hipBLASLt entry directly. These tests hold it to
the custom-op path bit for bit, check that it declines everything that path
may treat differently, and that torch.compile and CUDA graphs see what they
saw before.

Run:
    python op_tests/test_tuned_gemm_eager.py
    pytest op_tests/test_tuned_gemm_eager.py -q
"""

import sys

import pandas as pd
import pytest
import torch
import torch.nn.functional as F

import aiter.tuned_gemm as tuned
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.ops.opus._arch import _device_arch_and_cu
from aiter.tuned_gemm import tgemm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU with hipBLASLt"
)

BF16, FP16, FP32 = dtypes.bf16, dtypes.fp16, dtypes.fp32
# The real custom ops, taken before any test swaps the module attributes.
OP = tuned._gemm_a16w16_op
OP_OUT = tuned._gemm_a16w16_out_op


def _gfx_cu():
    return _device_arch_and_cu(torch.device("cuda", torch.cuda.current_device()))


def _key(m, n, k, bias, dtype, otype, bpreshuffle=False):
    gfx, cu = _gfx_cu()
    return (gfx, cu, m, n, k, bias, str(dtype), str(otype), False, bpreshuffle)


def _config(m, n, k, bias, dtype, otype, bpreshuffle=False):
    gfx, cu = _gfx_cu()
    return tuned.get_GEMM_A16W16_config(
        M=m,
        N=n,
        K=k,
        bias=bias,
        dtype=str(dtype),
        otype=str(otype),
        scaleAB=False,
        bpreshuffle=bpreshuffle,
        gfx=gfx,
        cu_num=cu,
    )


def _tuned_hipblaslt_rows(limit=6):
    """Tuned hipBLASLt rows for this GPU from the BF16 CSV, unshuffled."""
    gfx, cu = _gfx_cu()
    df = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_GEMM_BF16_FILE)
    df = df[
        (df["gfx"] == gfx)
        & (df["cu_num"] == cu)
        & (df["libtype"] == "hipblaslt")
        & df["dtype"].isin([str(BF16), str(FP16)])
        & (~df["bpreshuffle"].astype(bool))
        & (~df["scaleAB"].astype(bool))
    ].drop_duplicates(["M", "N", "K", "bias", "dtype", "outdtype"])
    rows = []
    for r in df.head(limit).itertuples():
        rows.append(
            (
                int(r.M),
                int(r.N),
                int(r.K),
                bool(r.bias),
                getattr(torch, r.dtype.removeprefix("torch.")),
                getattr(torch, r.outdtype.removeprefix("torch.")),
            )
        )
    return rows


def _rand(*shape, dtype=BF16, scale=1.0):
    return torch.randn(*shape, device="cuda").mul_(scale).to(dtype)


def _equal_or_nondeterministic(fast, ref, ref_again):
    """Bitwise equality, except where the solution itself is not deterministic
    (split-K / Stream-K order): then the two op-path runs differ as well."""
    if torch.equal(ref, ref_again):
        assert torch.equal(fast, ref)
    else:
        torch.testing.assert_close(fast.float(), ref.float(), atol=1e-1, rtol=1e-2)


@pytest.fixture
def fallbacks(monkeypatch):
    """Calls that reached the custom ops, by op."""
    calls = {"gemm_a16w16": 0, "gemm_a16w16_out": 0}

    def op(*args, **kwargs):
        calls["gemm_a16w16"] += 1
        return OP(*args, **kwargs)

    def op_out(*args, **kwargs):
        calls["gemm_a16w16_out"] += 1
        return OP_OUT(*args, **kwargs)

    monkeypatch.setattr(tuned, "_gemm_a16w16_op", op)
    monkeypatch.setattr(tuned, "_gemm_a16w16_out_op", op_out)
    return calls


@pytest.fixture
def table(monkeypatch):
    """Install a tuned table of our own: ``table[key] = row``."""
    rows = {}
    monkeypatch.setattr(tuned, "get_GEMM_A16W16_config_", lambda: rows)
    tuned.get_GEMM_A16W16_config.cache_clear()
    yield rows
    tuned.get_GEMM_A16W16_config.cache_clear()


@pytest.fixture(scope="module", autouse=True)
def hipblaslt_ready():
    """The first hipBLASLt GEMM of a process creates the extension on the op
    path; run one so every test starts with the fast path available."""
    tuned.hipb_create_extension()
    tuned.extensions_created = True
    x, w = _rand(64, 256), _rand(128, 256)
    OP(x, w)
    torch.cuda.synchronize()


def _check(fallbacks, x, w, bias, otype, out_kind, expect_fast):
    """`tgemm.mm` against the custom op, for one call shape."""
    shape = (*x.shape[:-1], w.shape[0])
    ref = OP(x, w, bias, otype)
    ref_again = OP(x, w, bias, otype)
    before = dict(fallbacks)
    if out_kind is None:
        y = tgemm.mm(x, w, bias, otype=otype)
    else:
        out = torch.full(shape, float("nan"), dtype=otype or x.dtype, device="cuda")
        if out_kind == "mm":
            y = tgemm.mm(x, w, bias, otype=otype, out=out)
        else:
            assert tuned.gemm_a16w16_out(x, w, out, bias=bias, otype=otype) is None
            y = out
        assert y.data_ptr() == out.data_ptr()
    torch.cuda.synchronize()
    took_op = sum(fallbacks.values()) - sum(before.values())
    assert took_op == (0 if expect_fast else 1)
    assert y.shape == shape and y.dtype == ref.dtype
    _equal_or_nondeterministic(y, ref, ref_again)


MODEL_SHAPES = [  # Qwen3.8-Flash-Next prefill BF16 GEMMs (TP2, SP shard)
    (1056, 324, 10240),
    (1056, 10240, 320),
    (2112, 513, 2560),
    (2112, 48, 2560),
    (2112, 640, 2560),
]


@pytest.mark.parametrize("out_kind", [None, "mm", "out_op"])
@pytest.mark.parametrize("mnk", MODEL_SHAPES)
def test_model_shapes_match_custom_op(fallbacks, mnk, out_kind):
    m, n, k = mnk
    x, w = _rand(m, k), _rand(n, k, scale=k**-0.5)
    hip = _config(m, n, k, False, BF16, BF16)["libtype"] == "hipblaslt"
    _check(fallbacks, x, w, None, BF16, out_kind, expect_fast=hip)


@pytest.mark.parametrize(
    "row", _tuned_hipblaslt_rows() if torch.cuda.is_available() else [], ids=str
)
def test_tuned_rows_match_custom_op(fallbacks, row):
    m, n, k, has_bias, dtype, otype = row
    x, w = _rand(m, k, dtype=dtype), _rand(n, k, dtype=dtype, scale=k**-0.5)
    bias = _rand(n, dtype=dtype) if has_bias else None
    assert _config(m, n, k, has_bias, dtype, otype)["libtype"] == "hipblaslt"
    _check(fallbacks, x, w, bias, otype, None, expect_fast=True)
    _check(fallbacks, x, w, bias, otype, "mm", expect_fast=True)


# hipBLASLt's own heuristic (solidx -1) through a tuned table of our own, so the
# combinations below hit the fast path on any GPU.
HEURISTIC_CASES = [
    # m, n, k, bias, dtype, otype, input rank
    (256, 512, 1024, False, BF16, BF16, 2),
    (256, 512, 1024, True, BF16, BF16, 2),
    (96, 384, 768, False, FP16, FP16, 2),
    (96, 384, 768, True, FP16, FP16, 2),
    (128, 256, 512, False, BF16, FP32, 2),
    (2 * 64, 512, 1024, False, BF16, BF16, 3),
    (2 * 64, 512, 1024, True, BF16, BF16, 3),
]


@pytest.mark.parametrize("out_kind", [None, "mm", "out_op"])
@pytest.mark.parametrize("case", HEURISTIC_CASES, ids=str)
def test_bias_otype_rank_match_custom_op(fallbacks, table, case, out_kind):
    m, n, k, has_bias, dtype, otype, rank = case
    table[_key(m, n, k, has_bias, dtype, otype)] = {
        "libtype": "hipblaslt",
        "solidx": -1,
    }
    x = _rand(m, k, dtype=dtype)
    if rank == 3:
        x = x.view(2, m // 2, k)
    w = _rand(n, k, dtype=dtype, scale=k**-0.5)
    bias = _rand(n, dtype=dtype) if has_bias else None
    _check(fallbacks, x, w, bias, otype, out_kind, expect_fast=True)


def test_otype_defaults_to_input_dtype(fallbacks, table):
    table[_key(64, 128, 256, False, BF16, BF16)] = {
        "libtype": "hipblaslt",
        "solidx": -1,
    }
    x, w = _rand(64, 256), _rand(128, 256)
    y = tuned.gemm_a16w16(x, w)
    assert fallbacks["gemm_a16w16"] == 0 and y.dtype == BF16
    assert torch.equal(y, OP(x, w))


def test_preshuffled_weight_matches_custom_op(fallbacks):
    from aiter.ops.shuffle import shuffle_weight

    if _gfx_cu()[0] != "gfx942":
        pytest.skip("BF16 preshuffled weights default to hipBLASLt on gfx942 only")
    m, n, k = 128, 256, 512
    x, w = _rand(m, k), shuffle_weight(_rand(n, k, scale=k**-0.5), layout=(16, 16))
    assert (
        _config(m, n, k, False, BF16, BF16, bpreshuffle=True)["libtype"] == "hipblaslt"
    )
    _check(fallbacks, x, w, None, BF16, None, expect_fast=True)


def test_declines_configs_that_are_not_hipblaslt(fallbacks, table):
    m, n, k = 64, 128, 256
    table[_key(m, n, k, False, BF16, BF16)] = {"libtype": "torch", "solidx": 0}
    x, w = _rand(m, k), _rand(n, k)
    y = tgemm.mm(x, w)
    assert fallbacks["gemm_a16w16"] == 1
    torch.testing.assert_close(y, F.linear(x, w), atol=0, rtol=0)


def test_replaced_config_function_is_called_every_call(fallbacks, monkeypatch):
    """A wrapper installed over `get_GEMM_A16W16_config` (ATOM's SGLang plugin
    does this, keyed on a flag it flips at run time) is asked on every call and
    its answer is never cached."""
    m, n, k = 1056, 324, 10240
    if _config(m, n, k, False, BF16, BF16)["libtype"] != "hipblaslt":
        pytest.skip("needs this shape tuned to hipBLASLt on this GPU")
    original = tuned.get_GEMM_A16W16_config
    state = {"bypass": False, "calls": 0}

    def replacement(*args, **kwargs):
        state["calls"] += 1
        config = original(*args, **kwargs)
        if state["bypass"] and config.get("libtype") == "hipblaslt":
            return dict(config, libtype="torch", solidx=0)
        return config

    monkeypatch.setattr(tuned, "get_GEMM_A16W16_config", replacement)
    x, w = _rand(m, k), _rand(n, k, scale=k**-0.5)
    ref = OP(x, w)  # the op path, through the replacement
    expect_ops = 0
    for bypass in (False, True, False, True):
        state["bypass"] = bypass
        calls, ops = state["calls"], fallbacks["gemm_a16w16"]
        y = tgemm.mm(x, w)
        if bypass:
            # asked by the fast path, then again by the op path it hands over to
            assert state["calls"] - calls == 2 and fallbacks["gemm_a16w16"] == ops + 1
            torch.testing.assert_close(y, F.linear(x, w), atol=0, rtol=0)
            expect_ops += 1
        else:
            assert state["calls"] - calls == 1 and fallbacks["gemm_a16w16"] == ops
            assert torch.equal(y, ref)
    assert fallbacks["gemm_a16w16"] == expect_ops


def test_scaled_calls_take_the_custom_op(monkeypatch):
    seen = []
    monkeypatch.setattr(tuned, "_hipblaslt_eager", lambda *a: seen.append(a))
    monkeypatch.setattr(tuned, "_gemm_a16w16_op", lambda *a, **k: "op")
    monkeypatch.setattr(tuned, "_gemm_a16w16_out_op", lambda *a, **k: None)
    x, w, s = _rand(8, 16), _rand(8, 16), torch.ones(1, device="cuda")
    assert tgemm.mm(x, w, scale_a=s, scale_b=s) == "op"
    assert tgemm.mm(x, w, scale_c=s) == "op"
    out = torch.empty(8, 8, dtype=BF16, device="cuda")
    assert tgemm.mm(x, w, scale_a=s, out=out) is out
    assert seen == []


def test_declines_grad_tracked_inputs(fallbacks):
    m, n, k = 1056, 324, 10240
    if _config(m, n, k, False, BF16, BF16)["libtype"] != "hipblaslt":
        pytest.skip("needs this shape tuned to hipBLASLt on this GPU")
    x, w = _rand(m, k), _rand(n, k, scale=k**-0.5).requires_grad_(True)
    y = tgemm.mm(x, w)  # grad mode on: autograd must see the op
    assert fallbacks["gemm_a16w16"] == 1 and y.requires_grad
    with torch.no_grad():
        z = tgemm.mm(x, w)
    assert fallbacks["gemm_a16w16"] == 1 and not z.requires_grad
    assert torch.equal(y.detach(), z)


def test_declines_under_fake_tensor_mode(fallbacks):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(1056, 10240, dtype=BF16, device="cuda")
        w = torch.empty(324, 10240, dtype=BF16, device="cuda")
        y = tgemm.mm(x, w)
    assert isinstance(y, FakeTensor) and tuple(y.shape) == (1056, 324)
    assert fallbacks["gemm_a16w16"] == 1
    # A fake tensor outside its mode still has to reach the op's fake.
    y2 = tgemm.mm(x, w)
    assert isinstance(y2, FakeTensor) and fallbacks["gemm_a16w16"] == 2


def test_dispatch_and_function_modes_see_the_op(fallbacks):
    from torch.overrides import TorchFunctionMode
    from torch.utils._python_dispatch import TorchDispatchMode

    seen = []

    class Record(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            seen.append(str(func))
            return func(*args, **(kwargs or {}))

    class Passthrough(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    x, w = _rand(1056, 10240), _rand(324, 10240, scale=0.01)
    with Record():
        tgemm.mm(x, w)
    assert any("aiter.gemm_a16w16" in s for s in seen)
    with Passthrough():
        tgemm.mm(x, w)
    assert fallbacks["gemm_a16w16"] == 2


def test_tensor_subclasses():
    class Plain(torch.nn.Parameter):
        pass

    from torch._subclasses.fake_tensor import FakeTensor

    assert tuned._eager_tensor_type(torch.Tensor)
    assert tuned._eager_tensor_type(torch.nn.Parameter)
    assert tuned._eager_tensor_type(Plain)
    assert not tuned._eager_tensor_type(FakeTensor)
    assert not tuned._eager_tensor_type(int)


def test_parameter_weights_take_the_fast_path(fallbacks):
    m, n, k = 1056, 324, 10240
    if _config(m, n, k, False, BF16, BF16)["libtype"] != "hipblaslt":
        pytest.skip("needs this shape tuned to hipBLASLt on this GPU")
    x = _rand(m, k)
    w = torch.nn.Parameter(_rand(n, k, scale=k**-0.5), requires_grad=False)
    y = tgemm.mm(x, w)
    assert fallbacks["gemm_a16w16"] == 0 and torch.equal(y, OP(x, w))


def test_unflattenable_input_takes_the_custom_op(fallbacks):
    x = _rand(4, 64, 8192).transpose(0, 1)  # `view(-1, K)` refuses it
    w = _rand(32, 8192, scale=8192**-0.5)
    y = tgemm.mm(x, w)
    assert fallbacks["gemm_a16w16"] == 1
    torch.testing.assert_close(y, F.linear(x, w), atol=0, rtol=0)


def test_shape_recording_keeps_the_custom_op(fallbacks, monkeypatch):
    recorded = []
    monkeypatch.setattr(tuned, "_record_gemm_shapes", lambda: True)
    monkeypatch.setattr(tuned, "save_shapes", lambda *a: recorded.append(a))
    monkeypatch.setattr(tuned, "_eager_hipb", None)  # resolve again
    x, w = _rand(64, 256), _rand(128, 256)
    tgemm.mm(x, w)
    tgemm.mm(x, w)
    assert fallbacks["gemm_a16w16"] == 2 and len(recorded) == 2
    assert tuned._eager_hipb is False


def test_weight_handles_follow_the_descriptor(fallbacks, table):
    """The cached handle of a weight is keyed by its descriptor alone: a freed
    weight whose memory a new one reuses, a view sharing the pointer with a
    different shape, and a strided view all compute what the op path does."""
    import gc
    import weakref

    m, n, k = 128, 256, 512
    for nn, kk in ((n, k), (n // 2, k), (n, k // 2)):
        table[_key(m, nn, kk, False, BF16, BF16)] = {
            "libtype": "hipblaslt",
            "solidx": -1,
        }
    x = _rand(m, k)
    w1 = _rand(n, k, scale=k**-0.5)
    assert torch.equal(tgemm.mm(x, w1), OP(x, w1))
    ptr, alive = w1.data_ptr(), weakref.ref(w1)
    del w1
    gc.collect()
    assert alive() is None  # the cache keeps no tensor alive
    w2 = _rand(n, k, scale=k**-0.5)
    if w2.data_ptr() == ptr:  # same descriptor as the freed weight
        assert torch.equal(tgemm.mm(x, w2), OP(x, w2))
    assert torch.equal(tgemm.mm(x, w2[: n // 2]), OP(x, w2[: n // 2]))
    xs = x[:, : k // 2]
    assert torch.equal(tgemm.mm(xs, w2[:, : k // 2]), OP(xs, w2[:, : k // 2]))
    assert sum(fallbacks.values()) == 0


def test_weight_handles_are_bounded(monkeypatch, table):
    monkeypatch.setattr(tuned, "_EAGER_WEIGHT_HANDLES_MAX", 4)
    monkeypatch.setattr(tuned, "_eager_weight_handles", tuned.OrderedDict())
    m, k = 64, 256
    x = _rand(m, k)
    keep = []
    for n in range(16, 16 * 9, 16):
        table[_key(m, n, k, False, BF16, BF16)] = {"libtype": "hipblaslt", "solidx": -1}
        w = _rand(n, k)
        keep.append(w)
        assert torch.equal(tgemm.mm(x, w), OP(x, w))
        assert len(tuned._eager_weight_handles) <= 4
    assert torch.equal(tgemm.mm(x, keep[0]), OP(x, keep[0]))  # evicted, rebuilt


def test_first_call_runs_the_custom_op(fallbacks, monkeypatch):
    """Until the op path has created the hipBLASLt extension, it runs the call."""
    monkeypatch.setattr(tuned, "extensions_created", False)
    monkeypatch.setattr(tuned, "_eager_hipb", None)
    m, n, k = 1056, 324, 10240
    if _config(m, n, k, False, BF16, BF16)["libtype"] != "hipblaslt":
        pytest.skip("needs this shape tuned to hipBLASLt on this GPU")
    x, w = _rand(m, k), _rand(n, k, scale=k**-0.5)
    first = tgemm.mm(x, w)
    assert fallbacks["gemm_a16w16"] == 1 and tuned.extensions_created
    second = tgemm.mm(x, w)
    assert fallbacks["gemm_a16w16"] == 1 and torch.equal(first, second)


def test_compile_records_the_custom_ops():
    m, n, k = 256, 512, 1024
    x, w, b = _rand(m, k), _rand(n, k, scale=k**-0.5), _rand(n)
    out = torch.empty(m, n, dtype=BF16, device="cuda")
    targets = []

    def backend(gm, example_inputs):
        targets.extend(str(node.target) for node in gm.graph.nodes)
        return gm.forward

    def fn(x, w, b, out):
        y = tgemm.mm(x, w, b)
        tgemm.mm(x, w, out=out)
        return y

    torch._dynamo.reset()
    y = torch.compile(fn, backend=backend, fullgraph=True)(x, w, b, out)
    torch.cuda.synchronize()
    assert "aiter.gemm_a16w16" in targets
    assert "aiter.gemm_a16w16_out" in targets
    assert torch.equal(y, OP(x, w, b))
    assert torch.equal(out, OP(x, w))


def test_cuda_graph_capture_and_replay(fallbacks):
    m, n, k = 1056, 324, 10240
    if _config(m, n, k, False, BF16, BF16)["libtype"] != "hipblaslt":
        pytest.skip("needs this shape tuned to hipBLASLt on this GPU")
    x, w = _rand(m, k), _rand(n, k, scale=k**-0.5)
    out = torch.empty(m, n, dtype=BF16, device="cuda")
    tgemm.mm(x, w)  # warm up outside the capture
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        y = tgemm.mm(x, w)
        tgemm.mm(x, w, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    assert sum(fallbacks.values()) == 0
    for _ in range(2):
        x.copy_(_rand(m, k))
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        ref = OP(x, w)
        assert torch.equal(y, ref) and torch.equal(out, ref)
    graph.reset()


def main():
    status = pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    return 0 if status == pytest.ExitCode.OK else 1


if __name__ == "__main__":
    sys.exit(main())
