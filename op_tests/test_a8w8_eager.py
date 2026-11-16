# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The eager fast paths of `gemm_a8w8_bpreshuffle` and `per_token_quant_hip`.

A bpreshuffle GEMM that runs a CK kernel and a dynamic per-token quant skip
their custom ops outside torch.compile and call the pybind entries directly.
These tests hold them to the custom-op path bit for bit, check that they
decline everything that path may treat differently, and that torch.compile
and CUDA graphs see what they saw before.

Run:
    python op_tests/test_a8w8_eager.py
    pytest op_tests/test_a8w8_eager.py -q
"""

import sys

import pandas as pd
import pytest
import torch

import aiter
import aiter.jit.core as jit_core
import aiter.ops.gemm_op_a8w8 as gop
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num
from aiter.jit.utils.chip_info import get_gfx_runtime as get_gfx
from aiter.ops import quant
from aiter.ops.shuffle import shuffle_weight

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU with the CK a8w8 kernels"
)

BF16, FP16, FP8, I8 = dtypes.bf16, dtypes.fp16, dtypes.fp8, dtypes.i8
# The real custom ops, taken before any test swaps the module attributes.
GEMM_OP = gop._gemm_a8w8_bpreshuffle_op
QUANT_OP = quant._per_token_quant_hip_op
gemm = gop.gemm_a8w8_bpreshuffle
per_token_quant = quant.per_token_quant_hip

# FP8 linear layers of a PTPC-FP8 checkpoint at TP2 (GDN in/out projections,
# attention qkv/o), at a 2112-token prefill chunk and a few small batches, plus
# shapes with no tuned row (the default CK kernel), e.g. 2112x3584x2560 on gfx942.
MODEL_SHAPES = [
    (2112, 8192, 2560),
    (2112, 2560, 3072),
    (2112, 6656, 2560),
    (2112, 3584, 2560),
    (1, 8192, 2560),
    (33, 2560, 3072),
]


def _config(m, n, k):
    return gop.get_GEMM_config_with_quant_type(
        m, n, k, FP8, AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE
    )


def _runs_ck(m, n, k):
    config = _config(m, n, k)
    return config["libtype"] == "ck" if config is not None else True


def _tuned_rows(libtype, limit=4):
    df = pd.read_csv(AITER_CONFIGS.AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE_FILE)
    df = df[
        (df["gfx"] == get_gfx())
        & (df["cu_num"] == get_cu_num())
        & (df["libtype"] == libtype)
        & (df["q_dtype_w"] == str(FP8))
    ].drop_duplicates(["M", "N", "K"])
    return [(int(r.M), int(r.N), int(r.K)) for r in df.head(limit).itertuples()]


def _rand(*shape, dtype=BF16, scale=1.0):
    return torch.randn(*shape, device="cuda").mul_(scale).to(dtype)


def _weight(n, k):
    w, w_scale = aiter.pertoken_quant(_rand(n, k, scale=k**-0.5), quant_dtype=FP8)
    return shuffle_weight(w, layout=(16, 16)), w_scale


def _inputs(m, n, k):
    xq, x_scale = QUANT_OP(_rand(m, k), quant_dtype=FP8)
    wq, w_scale = _weight(n, k)
    return xq, wq, x_scale, w_scale


@pytest.fixture
def fallbacks(monkeypatch):
    """Calls that reached the custom ops, by op."""
    calls = {"gemm": 0, "quant": 0}

    def gemm_op(*args, **kwargs):
        calls["gemm"] += 1
        return GEMM_OP(*args, **kwargs)

    def quant_op(*args, **kwargs):
        calls["quant"] += 1
        return QUANT_OP(*args, **kwargs)

    monkeypatch.setattr(gop, "_gemm_a8w8_bpreshuffle_op", gemm_op)
    monkeypatch.setattr(quant, "_per_token_quant_hip_op", quant_op)
    return calls


@pytest.fixture(scope="module", autouse=True)
def fast_paths_ready():
    """The first call of each runs the op path, which loads the module; run
    one so every test starts with the fast paths available."""
    xq, wq, x_scale, w_scale = _inputs(64, 256, 512)
    gemm(xq, wq, x_scale, w_scale)
    per_token_quant(_rand(64, 512), quant_dtype=FP8)
    assert gop._eager_a8w8_ck and quant._eager_quant


def test_public_names_are_the_fast_paths():
    assert aiter.gemm_a8w8_bpreshuffle is gemm
    assert aiter.per_token_quant_hip is per_token_quant
    assert quant.get_hip_quant(aiter.QuantType.per_Token) is per_token_quant


@pytest.mark.parametrize("mnk", MODEL_SHAPES, ids=str)
@pytest.mark.parametrize("dtype", [BF16, FP16], ids=str)
def test_model_shapes_match_custom_ops(fallbacks, mnk, dtype):
    m, n, k = mnk
    if not _runs_ck(m, n, k):
        pytest.skip("this shape is tuned to a non-CK kernel on this GPU")
    x = _rand(m, k, dtype=dtype)
    xq, x_scale = per_token_quant(x, quant_dtype=FP8)
    xq_ref, x_scale_ref = QUANT_OP(x, quant_dtype=FP8)
    assert torch.equal(xq.view(torch.uint8), xq_ref.view(torch.uint8))
    assert torch.equal(x_scale, x_scale_ref)
    wq, w_scale = _weight(n, k)
    y = gemm(xq, wq, x_scale, w_scale, dtype=dtype)
    ref = GEMM_OP(xq, wq, x_scale, w_scale, None, dtype)
    assert fallbacks == {"gemm": 0, "quant": 0}
    assert y.dtype == dtype and torch.equal(y, ref)


@pytest.mark.parametrize("mnk", _tuned_rows("ck"), ids=str)
def test_tuned_ck_rows_match_custom_op(fallbacks, mnk):
    m, n, k = mnk
    xq, wq, x_scale, w_scale = _inputs(m, n, k)
    y = gemm(xq, wq, x_scale, w_scale)
    assert fallbacks["gemm"] == 0
    assert torch.equal(y, GEMM_OP(xq, wq, x_scale, w_scale))


@pytest.mark.parametrize(
    "mnk",
    _tuned_rows("flydsl", 2) + _tuned_rows("cktile", 2) + _tuned_rows("asm", 2),
    ids=str,
)
def test_declines_rows_that_are_not_ck(fallbacks, mnk):
    m, n, k = mnk
    xq, wq, x_scale, w_scale = _inputs(m, n, k)
    y = gemm(xq, wq, x_scale, w_scale)
    assert fallbacks["gemm"] == 1
    assert torch.equal(y, GEMM_OP(xq, wq, x_scale, w_scale))


def test_replaced_config_function_is_called_every_call(fallbacks, monkeypatch):
    """A replacement of the module attribute is honoured on every call."""
    seen = []

    def lookup(m, n, k, q_dtype_w, tuned_file=None):
        seen.append((m, n, k))
        return {"libtype": "ck", "splitK": 0}

    monkeypatch.setattr(gop, "get_GEMM_config_with_quant_type", lookup)
    xq, wq, x_scale, w_scale = _inputs(128, 256, 1024)
    y1 = gemm(xq, wq, x_scale, w_scale)
    y2 = gemm(xq, wq, x_scale, w_scale)
    assert seen == [(128, 256, 1024)] * 2 and fallbacks["gemm"] == 0
    ref = GEMM_OP(xq, wq, x_scale, w_scale)  # the op path asks it too
    assert len(seen) == 3
    torch.testing.assert_close(y1, ref, atol=0, rtol=0)
    torch.testing.assert_close(y2, ref, atol=0, rtol=0)


def test_declines_arguments_the_op_rejects_or_pads(fallbacks):
    m, n, k = 64, 256, 512
    xq, wq, x_scale, w_scale = _inputs(m, n, k)
    with pytest.raises(Exception, match="bias"):
        gemm(xq, wq, x_scale, w_scale, bias=torch.zeros(n, device="cuda"))
    with pytest.raises(Exception, match="not supported"):
        gemm(xq, wq, x_scale, w_scale, dtype=torch.float32)
    assert fallbacks["gemm"] == 2
    # K-padded weight: the op zero-pads XQ to the weight's K.
    wq_pad, w_scale_pad = _weight(n, k + 128)
    y = gemm(xq, wq_pad, x_scale, w_scale_pad)
    assert fallbacks["gemm"] == 3
    assert torch.equal(y, GEMM_OP(xq, wq_pad, x_scale, w_scale_pad))


def test_declines_grad_tracked_inputs(fallbacks):
    xq, wq, x_scale, w_scale = _inputs(64, 256, 512)
    w_scale.requires_grad_(True)
    y = gemm(xq, wq, x_scale, w_scale)  # grad mode on: autograd must see the op
    assert fallbacks["gemm"] == 1
    with torch.no_grad():
        z = gemm(xq, wq, x_scale, w_scale)
    assert fallbacks["gemm"] == 1 and torch.equal(y.detach(), z)
    x = _rand(64, 512).requires_grad_(True)
    q, s = per_token_quant(x, quant_dtype=FP8)
    assert fallbacks["quant"] == 1
    with torch.no_grad():
        q2, s2 = per_token_quant(x, quant_dtype=FP8)
    assert fallbacks["quant"] == 1
    assert torch.equal(q.view(torch.uint8), q2.view(torch.uint8))
    assert torch.equal(s.detach(), s2)


def test_declines_under_fake_tensor_mode(fallbacks):
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    with FakeTensorMode():
        x = torch.empty(2112, 2560, dtype=BF16, device="cuda")
        xq, x_scale = per_token_quant(x, quant_dtype=FP8)
        wq = torch.empty(8192, 2560, dtype=FP8, device="cuda")
        w_scale = torch.empty(8192, 1, dtype=torch.float32, device="cuda")
        y = gemm(xq, wq, x_scale, w_scale)
    assert isinstance(xq, FakeTensor) and isinstance(y, FakeTensor)
    assert tuple(y.shape) == (2112, 8192) and tuple(x_scale.shape) == (2112, 1)
    assert fallbacks == {"gemm": 1, "quant": 1}
    # Fake tensors outside their mode still have to reach the ops' fakes.
    y2 = gemm(xq, wq, x_scale, w_scale)
    assert isinstance(y2, FakeTensor) and fallbacks["gemm"] == 2


def test_dispatch_and_function_modes_see_the_ops(fallbacks):
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

    x = _rand(64, 512)
    wq, w_scale = _weight(256, 512)
    with Record():
        xq, x_scale = per_token_quant(x, quant_dtype=FP8)
        gemm(xq, wq, x_scale, w_scale)
    assert any("aiter.per_token_quant_hip" in s for s in seen)
    assert any("aiter.gemm_a8w8_bpreshuffle" in s for s in seen)
    with Passthrough():
        xq, x_scale = per_token_quant(x, quant_dtype=FP8)
        gemm(xq, wq, x_scale, w_scale)
    assert fallbacks == {"gemm": 2, "quant": 2}


def test_rebound_ops_take_the_custom_op(fallbacks, monkeypatch):
    """A replaced inner op is what the op path calls, so the fast path, which
    would stand in for it, declines."""
    calls = []
    ck, dq = gop.gemm_a8w8_bpreshuffle_ck, quant.dynamic_per_token_scaled_quant

    def ck_spy(*args, **kwargs):
        calls.append("ck")
        return ck(*args, **kwargs)

    def dq_spy(*args, **kwargs):
        calls.append("quant")
        return dq(*args, **kwargs)

    monkeypatch.setattr(gop, "gemm_a8w8_bpreshuffle_ck", ck_spy)
    monkeypatch.setattr(quant, "dynamic_per_token_scaled_quant", dq_spy)
    m, n, k = 2112, 2560, 3072
    if not _runs_ck(m, n, k):
        pytest.skip("this shape is tuned to a non-CK kernel on this GPU")
    xq, x_scale = per_token_quant(_rand(m, k), quant_dtype=FP8)
    wq, w_scale = _weight(n, k)
    gemm(xq, wq, x_scale, w_scale)
    assert fallbacks == {"gemm": 1, "quant": 1} and calls == ["quant", "ck"]


def test_first_call_runs_the_custom_op(fallbacks, monkeypatch):
    monkeypatch.setattr(gop, "_eager_a8w8_ck", None)
    monkeypatch.setattr(quant, "_eager_quant", None)
    x = _rand(64, 512)
    wq, w_scale = _weight(256, 512)
    for _ in range(2):
        xq, x_scale = per_token_quant(x, quant_dtype=FP8)
        gemm(xq, wq, x_scale, w_scale)
    assert fallbacks == {"gemm": 1, "quant": 1}
    assert gop._eager_a8w8_ck and quant._eager_quant


def test_argument_logging_keeps_the_custom_op(fallbacks, monkeypatch):
    monkeypatch.setattr(jit_core, "AITER_LOG_MORE", 2)
    monkeypatch.setattr(gop, "_eager_a8w8_ck", None)
    monkeypatch.setattr(quant, "_eager_quant", None)
    monkeypatch.setattr(
        "aiter.test_common.log_args", lambda *a, **k: None, raising=False
    )
    x = _rand(64, 512)
    wq, w_scale = _weight(256, 512)
    for _ in range(2):
        xq, x_scale = per_token_quant(x, quant_dtype=FP8)
        gemm(xq, wq, x_scale, w_scale)
    assert fallbacks == {"gemm": 2, "quant": 2}
    assert gop._eager_a8w8_ck is False and quant._eager_quant is False


@pytest.mark.parametrize("shape", [(2112, 2560), (1, 3072), (4, 33, 2560)], ids=str)
@pytest.mark.parametrize("in_dtype", [BF16, FP16], ids=str)
@pytest.mark.parametrize("quant_dtype", [FP8, I8], ids=str)
def test_quant_matches_custom_op(fallbacks, shape, in_dtype, quant_dtype):
    x = _rand(*shape, dtype=in_dtype, scale=3.0)
    q, s = per_token_quant(x, quant_dtype=quant_dtype)
    q_ref, s_ref = QUANT_OP(x, quant_dtype=quant_dtype)
    assert fallbacks["quant"] == 0
    assert q.dtype == quant_dtype and q.shape == x.shape
    assert s.dtype == torch.float32 and s.shape == (*shape[:-1], 1)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    assert torch.equal(s, s_ref)


def test_quant_num_rows(fallbacks):
    x = _rand(256, 2560)
    num_rows = torch.tensor([100], dtype=torch.int32, device="cuda")
    q, s = per_token_quant(x, quant_dtype=FP8, num_rows=num_rows)
    q_ref, s_ref = QUANT_OP(x, quant_dtype=FP8, num_rows=num_rows)
    assert torch.equal(q[:100].view(torch.uint8), q_ref[:100].view(torch.uint8))
    assert torch.equal(s[:100], s_ref[:100])
    assert fallbacks["quant"] == 0


def test_static_quant_scale_takes_the_custom_op(fallbacks):
    x = _rand(64, 512)
    with pytest.raises(Exception, match="static per token quant"):
        per_token_quant(x, scale=torch.ones(1, device="cuda"), quant_dtype=FP8)
    assert fallbacks["quant"] == 1


def test_compile_records_the_custom_ops():
    m, n, k = 256, 512, 1024
    x = _rand(m, k)
    wq, w_scale = _weight(n, k)
    targets = []

    def backend(gm, example_inputs):
        targets.extend(str(node.target) for node in gm.graph.nodes)
        return gm.forward

    def fn(x, wq, w_scale):
        xq, x_scale = per_token_quant(x, quant_dtype=FP8)
        return gemm(xq, wq, x_scale, w_scale)

    torch._dynamo.reset()
    y = torch.compile(fn, backend=backend, fullgraph=True)(x, wq, w_scale)
    torch.cuda.synchronize()
    assert "aiter.per_token_quant_hip" in targets
    assert "aiter.gemm_a8w8_bpreshuffle" in targets
    xq, x_scale = QUANT_OP(x, quant_dtype=FP8)
    assert torch.equal(y, GEMM_OP(xq, wq, x_scale, w_scale))


def test_cuda_graph_capture_and_replay(fallbacks):
    m, n, k = 2112, 8192, 2560
    if not _runs_ck(m, n, k):
        pytest.skip("this shape is tuned to a non-CK kernel on this GPU")
    x = _rand(m, k)
    wq, w_scale = _weight(n, k)
    xq, x_scale = per_token_quant(x, quant_dtype=FP8)  # warm up outside the capture
    gemm(xq, wq, x_scale, w_scale)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        xq, x_scale = per_token_quant(x, quant_dtype=FP8)
        y = gemm(xq, wq, x_scale, w_scale)
    torch.cuda.current_stream().wait_stream(stream)
    assert fallbacks == {"gemm": 0, "quant": 0}
    for _ in range(2):
        x.copy_(_rand(m, k))
        graph.replay()
        torch.cuda.synchronize()
        xq_ref, x_scale_ref = QUANT_OP(x, quant_dtype=FP8)
        assert torch.equal(xq.view(torch.uint8), xq_ref.view(torch.uint8))
        assert torch.equal(y, GEMM_OP(xq_ref, wq, x_scale_ref, w_scale))
    graph.reset()


def main():
    status = pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    return 0 if status == pytest.ExitCode.OK else 1


if __name__ == "__main__":
    sys.exit(main())
