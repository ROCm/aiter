# SPDX-License-Identifier: MIT
"""Exercise per-call dispatch through the wrapper in the normal GPU CI suite."""

from functools import partial

import pytest
import torch

from aiter.ops.triton.attention import mha


@pytest.fixture
def stride_widths(monkeypatch):
    monkeypatch.setattr(mha, "_USE_INT64_STRIDES", True)
    monkeypatch.setattr(mha, "_MHA_IMPL", "default")
    original = mha._attn_fwd
    widths = []

    class RecordWidth:
        def __getitem__(self, grid):
            launch = original[grid]

            def run(*args, **kwargs):
                widths.append(kwargs["USE_INT64_STRIDES"])
                return launch(*args, **kwargs)

            return run

    monkeypatch.setattr(mha, "_attn_fwd", RecordWidth())
    return widths


@pytest.mark.parametrize(
    "nq,nk,dim,causal,left",
    [
        (129, 129, 128, False, -1),
        (17, 33, 128, True, 31),
        (65, 129, 80, False, -1),
        (17, 17, 8, False, -1),
    ],
)
def test_varlen_stride_dispatch_parity(nq, nk, dim, causal, left, stride_widths):
    torch.manual_seed(47)
    # Offset, noncontiguous input views also exercise masked head padding.
    q, k, v = [
        torch.randn(n * 2 + 1, h, dim + 8, device="cuda", dtype=torch.bfloat16)[
            1::2, :, :dim
        ]
        for n, h in ((nq, 8), (nk, 2), (nk, 2))
    ]
    cuq = torch.tensor([0, nq], device=q.device, dtype=torch.int32)
    cuk = torch.tensor([0, nk], device=q.device, dtype=torch.int32)
    invoke = partial(
        mha.flash_attn_varlen_func,
        q,
        k,
        v,
        cuq,
        cuk,
        nq,
        nk,
        causal=causal,
        window_size=(left, -1),
    )
    with torch.inference_mode():
        control = invoke()
        candidate = invoke(prefer_int32_strides=True)
        assert stride_widths == [True, False]
        torch.testing.assert_close(candidate, control, rtol=0, atol=0)

        qq, kk, vv = [x.transpose(0, 1).float() for x in (q, k, v)]
        kk, vv = kk.repeat_interleave(4, 0), vv.repeat_interleave(4, 0)
        scores = (qq @ kk.transpose(-1, -2)) * dim**-0.5
        qi = torch.arange(nq, device=q.device)[:, None] + nk - nq
        ki = torch.arange(nk, device=q.device)[None, :]
        mask = torch.ones(nq, nk, device=q.device, dtype=torch.bool)
        if causal:
            mask &= ki <= qi
        if left >= 0:
            mask &= ki >= qi - left
        ref = (scores.masked_fill(~mask, -torch.inf).softmax(-1) @ vv).transpose(0, 1)
        torch.testing.assert_close(candidate.float(), ref, atol=0.02, rtol=0.03)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            invoke(prefer_int32_strides=True)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = invoke(prefer_int32_strides=True)
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, control, rtol=0, atol=0)
        assert mha._USE_INT64_STRIDES is True


def test_training_preserves_int64_and_gradients(stride_widths):
    torch.manual_seed(48)
    tensors = [
        torch.randn(65, 4, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        for _ in range(3)
    ]
    cu = torch.tensor([0, 65], device=tensors[0].device, dtype=torch.int32)
    results = []
    for prefer in (False, True):
        for x in tensors:
            x.grad = None
        out = mha.flash_attn_varlen_func(
            *tensors, cu, cu, 65, 65, prefer_int32_strides=prefer
        )
        out.float().square().sum().backward()
        results.append([out.detach().clone(), *[x.grad.clone() for x in tensors]])
    assert stride_widths == [True, True]
    for a, b in zip(*results):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("backend,impl", [("gluon", "default"), ("triton", "dao_ai")])
def test_unsupported_implementation_rejected(backend, impl, monkeypatch):
    monkeypatch.setattr(mha, "_MHA_IMPL", impl)
    with pytest.raises(ValueError, match="prefer_int32_strides"):
        mha.flash_attn_varlen_func(
            None,
            None,
            None,
            None,
            None,
            1,
            1,
            backend=backend,
            prefer_int32_strides=True,
        )
