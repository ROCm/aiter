# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""``flydsl_lm_head_gemm`` (bf16 [M, 5120] x [129280, 5120]^T) against an fp64
reference, both weight layouts, every kernel variant, graph replay.

Run: ``python3 -m pytest op_tests/flydsl_tests/test_flydsl_lm_head_gemm.py -q``
(gfx950 only; the weight is the real 1.32 GB size).
"""

import sys

import pytest
import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

if not torch.cuda.is_available() or get_gfx_runtime() != "gfx950":
    pytest.skip("flydsl_lm_head_gemm needs gfx950", allow_module_level=True)

from aiter.ops.flydsl import lm_head_gemm as L
from aiter.ops.flydsl.lm_head_gemm import (
    K,
    N,
    flydsl_lm_head_gemm,
    lm_head_gemm_preferred,
    lm_head_gemm_supported,
    shuffle_weight,
    unshuffle_weight,
    warmup_lm_head_gemm,
)

# every kernel variant boundary: small 1-64, large 65-256 (rotated K starts),
# large above 256 (lockstep), each M-tile choice, and a 2048-row chunked call
MS = [1, 2, 7, 15, 16, 17, 33, 64, 65, 70, 80, 96, 97, 112, 127, 128, 129, 160]
MS += [161, 192, 193, 224, 225, 256, 257, 300, 384, 513, 640, 768]


def reference(x, w):
    """The fp64 product of the bf16 operands, rounded once to bf16."""
    return (x.double() @ w.double().t()).bfloat16()


@pytest.fixture(scope="module")
def weights():
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(N, K, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    return {False: w, True: shuffle_weight(w)}


def check_numeric(y, ref):
    """Per row: cosine >= 0.99999 and relative L2 <= 1e-3 (fp32 sums of 5120
    products against fp64; the order of the sum is the kernel's)."""
    yf, rf = y.double(), ref.double()
    assert not torch.isnan(yf).any(), "unwritten (NaN sentinel) outputs"
    cos = torch.nn.functional.cosine_similarity(yf, rf, dim=1)
    l2 = (yf - rf).norm(dim=1) / rf.norm(dim=1)
    assert bool(torch.all(cos >= 0.99999)), cos.min().item()
    assert bool(torch.all(l2 <= 1e-3)), l2.max().item()
    # an element-wise bound as well: a few bf16 ulps of the row scale
    err = (yf - rf).abs().max(dim=1).values / rf.abs().max(dim=1).values
    assert bool(torch.all(err <= 2e-2)), err.max().item()


def test_shuffle_is_a_permutation(weights):
    w, ws = weights[False], weights[True]
    assert torch.equal(unshuffle_weight(ws).view(torch.int16), w.view(torch.int16))
    flat = ws.view(-1)
    for n, k in [(0, 0), (15, 63), (16, 64), (129279, 5119), (12345, 4321)]:
        idx = (((n // 16) * (K // 64) + k // 64) * 16 + n % 16) * 64 + k % 64
        assert flat[idx].view(torch.int16) == w[n, k].view(torch.int16)


@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("m", MS)
def test_reference(weights, shuffled, m):
    w = weights[shuffled]
    x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
    y = torch.full((m, N), float("nan"), device="cuda", dtype=torch.bfloat16)
    flydsl_lm_head_gemm(x, w, y, shuffled=shuffled)
    check_numeric(y, reference(x, weights[False]))
    old = y.clone()
    for _ in range(2):  # deterministic bits
        y.fill_(float("nan"))
        flydsl_lm_head_gemm(x, w, y, shuffled=shuffled)
        assert torch.equal(y.view(torch.int16), old.view(torch.int16))


def test_allocates_output_and_chunks_above_2048_rows(weights):
    """The op's own output buffer; 2100 rows run as a 2048-row and a 52-row
    launch with the bits of separate calls."""
    x = torch.randn(2100, K, device="cuda").to(torch.bfloat16)
    y = flydsl_lm_head_gemm(x, weights[False])
    assert y.shape == (2100, N) and y.dtype == torch.bfloat16
    check_numeric(y[::97], reference(x[::97], weights[False]))
    tail = flydsl_lm_head_gemm(x[2048:].contiguous(), weights[False])
    assert torch.equal(y[2048:].view(torch.int16), tail.view(torch.int16))


def test_layouts_agree(weights):
    x = torch.randn(80, K, device="cuda").to(torch.bfloat16)
    a = flydsl_lm_head_gemm(x, weights[False], shuffled=False)
    b = flydsl_lm_head_gemm(x, weights[True], shuffled=True)
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_zero_rows(weights):
    x = torch.empty(0, K, device="cuda", dtype=torch.bfloat16)
    assert flydsl_lm_head_gemm(x, weights[False]).shape == (0, N)


@pytest.mark.parametrize("shuffled", [False, True])
def test_nonfinite_propagation(weights, shuffled):
    w = weights[False].clone()
    w[0, 2] = float("nan")
    w[129279, 8] = float("inf")
    w[16].zero_()
    x = torch.randn(3, K, device="cuda").to(torch.bfloat16)
    x[0, 1] = float("nan")
    x[1, 3] = float("inf")
    x[2].zero_()
    y = flydsl_lm_head_gemm(x, shuffle_weight(w) if shuffled else w, shuffled=shuffled)
    ref = reference(x, w)
    assert torch.equal(torch.isnan(y), torch.isnan(ref))
    assert torch.equal(torch.isposinf(y), torch.isposinf(ref))
    assert torch.equal(torch.isneginf(y), torch.isneginf(ref))
    # a zero row stays 0 except at the NaN / inf weight columns (0 x inf = NaN)
    fin = torch.isfinite(ref[2])
    assert bool(torch.all(y[2][fin] == 0)) and int((~fin).sum()) == 2


@pytest.mark.parametrize("shuffled", [False, True])
def test_graph_replay_no_compile_after_warmup(weights, shuffled, monkeypatch):
    from flydsl.compiler.jit_function import MlirCompiler

    warmup_lm_head_gemm(torch.device("cuda"), shuffled)
    compiled = []
    original = MlirCompiler.compile

    def track(*args, **kwargs):
        compiled.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(MlirCompiler, "compile", track)
    w = weights[shuffled]
    for m in (1, 17, 80, 128, 129, 161, 192, 193, 224, 256, 300, 513):
        x = torch.randn(m, K, device="cuda").to(torch.bfloat16)
        y = torch.empty(m, N, device="cuda", dtype=torch.bfloat16)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flydsl_lm_head_gemm(x, w, y, shuffled=shuffled)
        for _ in range(2):
            x.normal_()
            y.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check_numeric(y, reference(x, weights[False]))
    assert compiled == []


def test_warmup_rows_cover_every_variant():
    variants = {L._variant(m, False) for m in range(1, L.MAX_M + 1)}
    assert {L._variant(m, False) for m in L.WARMUP_ROWS} == variants
    assert len(L.WARMUP_ROWS) == len(variants)


def test_supported(weights):
    w = weights[False]
    assert lm_head_gemm_supported(w) and lm_head_gemm_supported(weights[True])
    assert not lm_head_gemm_supported(w[:, : K - 64])
    assert not lm_head_gemm_supported(w.T)
    assert not lm_head_gemm_supported(torch.empty(N, K, dtype=torch.float16))
    assert not lm_head_gemm_supported(torch.empty(N - 16, K, dtype=torch.bfloat16))
    bias = torch.zeros(N, dtype=torch.bfloat16, device=w.device)
    assert not lm_head_gemm_supported(w, bias)
    with pytest.raises(ValueError):
        flydsl_lm_head_gemm(torch.zeros(1, K, dtype=torch.bfloat16), w.T)


def test_preferred_rows():
    assert [m for m in range(1, 1025) if lm_head_gemm_preferred(m)] == [
        m for lo, hi in L.PREFERRED_ROWS for m in range(lo, hi + 1)
    ]


@pytest.mark.parametrize("shuffled", [False, True])
def test_rows_do_not_depend_on_m(weights, shuffled):
    """A row's bits are the same for every M within one kernel and K start
    rule (four waves: up to 64 rows; eight waves: 65-256 rows with rotated K
    starts; above 256 rows: all start at block 0)."""
    w = weights[shuffled]
    x = torch.randn(768, K, device="cuda").to(torch.bfloat16)
    full = flydsl_lm_head_gemm(x, w, shuffled=shuffled)
    mid = flydsl_lm_head_gemm(x[:256].contiguous(), w, shuffled=shuffled)
    small = flydsl_lm_head_gemm(x[:64].contiguous(), w, shuffled=shuffled)
    for m in (1, 16, 33):
        y = flydsl_lm_head_gemm(x[:m].contiguous(), w, shuffled=shuffled)
        assert torch.equal(y.view(torch.int16), small[:m].view(torch.int16)), m
    for m in (65, 96, 128, 160, 192, 224):
        y = flydsl_lm_head_gemm(x[:m].contiguous(), w, shuffled=shuffled)
        assert torch.equal(y.view(torch.int16), mid[:m].view(torch.int16)), m
    for m in (257, 320, 384, 512, 640):
        y = flydsl_lm_head_gemm(x[:m].contiguous(), w, shuffled=shuffled)
        assert torch.equal(y.view(torch.int16), full[:m].view(torch.int16)), m


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
