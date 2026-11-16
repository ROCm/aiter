# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""`hipb_mm` keeps the hipBLASLt layouts, descriptor and algorithm of an
unscaled GEMM with a given solution index, and rebuilds nothing on later calls
but the bias pointer. A call served that way must give what a freshly built
descriptor gives, bit for bit. The kept descriptors are per thread.

Run:
    python op_tests/test_hipb_mm_reuse.py
    pytest op_tests/test_hipb_mm_reuse.py -q
"""

import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from aiter import dtypes, hipb_create_extension, hipb_destroy_extension, hipb_mm
from aiter.ops.gradlib import hipb_findallsols

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU with hipBLASLt"
)

BF16, FP16 = dtypes.bf16, dtypes.fp16


@pytest.fixture(scope="module", autouse=True)
def extension():
    hipb_create_extension()
    yield


def _fresh_extension():
    """Destroy and recreate the extension: every kept descriptor goes with it."""
    torch.cuda.synchronize()
    hipb_destroy_extension()
    hipb_create_extension()


def _rand(*shape, dtype=BF16, scale=1.0):
    return torch.randn(*shape, device="cuda").mul_(scale).to(dtype)


def _solution(x, wt, bias=None, otype=None):
    sols = hipb_findallsols(x, wt, bias=bias, out_dtype=otype)
    if not sols:
        pytest.skip("no hipBLASLt solution for this problem")
    return sols[0]


CASES = [
    # m, n, k, dtype, otype, bias
    (1056, 324, 10240, BF16, BF16, False),
    (2112, 513, 2560, BF16, BF16, False),
    (256, 512, 1024, BF16, BF16, True),
    (96, 384, 768, FP16, FP16, True),
    (128, 256, 512, BF16, dtypes.fp32, False),
]


@pytest.mark.parametrize("case", CASES, ids=str)
def test_kept_descriptors_match_fresh_ones(case):
    m, n, k, dtype, otype, has_bias = case
    x, w = _rand(m, k, dtype=dtype), _rand(n, k, dtype=dtype, scale=k**-0.5)
    b1 = _rand(n, dtype=dtype) if has_bias else None
    sol = _solution(x, w.t(), b1, otype)
    first = hipb_mm(x, w.t(), sol, bias=b1, out_dtype=otype)  # builds them
    again = hipb_mm(x, w.t(), sol, bias=b1, out_dtype=otype)  # reuses them
    torch.cuda.synchronize()
    if not torch.equal(first, again):
        pytest.skip("solution is not deterministic run to run")
    if has_bias:
        # Same configuration, another bias: only the pointer may change.
        b2 = _rand(n, dtype=dtype)
        kept = hipb_mm(x, w.t(), sol, bias=b2, out_dtype=otype)
        _fresh_extension()
        fresh = hipb_mm(x, w.t(), sol, bias=b2, out_dtype=otype)
        torch.cuda.synchronize()
        assert torch.equal(kept, fresh)
        assert not torch.equal(kept, first)
    else:
        _fresh_extension()
        fresh = hipb_mm(x, w.t(), sol, out_dtype=otype)
        torch.cuda.synchronize()
        assert torch.equal(first, fresh)


def test_configurations_do_not_mix():
    """Interleaved shapes, leading dimensions and bias on/off each keep their
    own descriptors."""
    k = 512
    big = _rand(256, 2 * k)
    x_strided = big[:, :k]  # lda = 2k
    x = x_strided.contiguous()
    w = _rand(384, k, scale=k**-0.5)
    b = _rand(384)
    sol = _solution(x, w.t())
    sol_b = _solution(x, w.t(), b)
    calls = [
        lambda: hipb_mm(x, w.t(), sol),
        lambda: hipb_mm(x_strided, w.t(), sol),
        lambda: hipb_mm(x[:128], w.t(), sol),
        lambda: hipb_mm(x, w.t(), sol_b, bias=b),
    ]
    kept = [[f() for f in calls] for _ in range(2)]
    _fresh_extension()
    fresh = [f() for f in calls]
    torch.cuda.synchronize()
    for i in range(len(calls)):
        assert torch.equal(kept[0][i], fresh[i]) and torch.equal(kept[1][i], fresh[i])
    assert torch.equal(fresh[0], fresh[1])  # same values, lda differs
    assert torch.equal(fresh[0][:128], fresh[2])


def test_many_configurations():
    """More configurations than are kept (one per leading dimension here): the
    set is dropped and rebuilt, and every call still computes what a fresh
    descriptor does."""
    m, n, k, step, count = 64, 64, 64, 64, 1100
    w = _rand(n, k, scale=k**-0.5)
    buf = _rand(m * (k + step * count))
    views = [buf.as_strided((m, k), (k + step * i, 1)) for i in range(count)]
    sol = _solution(views[0], w.t())
    kept = [hipb_mm(x, w.t(), sol) for x in views]
    _fresh_extension()
    for i in (0, 1, 511, 1023, 1024, 1099):
        assert torch.equal(kept[i], hipb_mm(views[i], w.t(), sol))
        assert torch.equal(kept[i], hipb_mm(views[i].contiguous(), w.t(), sol))


def test_graph_replay_after_reuse():
    m, n, k = 1056, 324, 10240
    x, w = _rand(m, k), _rand(n, k, scale=k**-0.5)
    sol = _solution(x, w.t())
    out = torch.empty(m, n, dtype=BF16, device="cuda")
    hipb_mm(x, w.t(), sol, out=out)  # build outside the capture
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        hipb_mm(x, w.t(), sol, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    x.copy_(_rand(m, k))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, hipb_mm(x, w.t(), sol))
    graph.reset()


def _thread_calls():
    """Interleaved configurations, including one configuration that every
    thread calls with a bias of its own."""
    k = 512
    x = _rand(256, k)
    w = _rand(384, k, scale=k**-0.5)
    sol = _solution(x, w.t())
    biases = [_rand(384) for _ in range(4)]
    sol_b = _solution(x, w.t(), biases[0])
    x2, w2 = _rand(96, 768), _rand(320, 768, scale=768**-0.5)
    sol2 = _solution(x2, w2.t())

    def calls(i):
        b = biases[i % len(biases)]
        return [
            lambda: hipb_mm(x, w.t(), sol_b, bias=b),
            lambda: hipb_mm(x, w.t(), sol),
            lambda: hipb_mm(x2, w2.t(), sol2),
            lambda: hipb_mm(x[:128], w.t(), sol_b, bias=b),
        ]

    return calls


def test_threads_keep_their_own_descriptors():
    """Each thread builds and reuses its own descriptors; results match a
    single thread bit for bit. The binding holds the GIL, so Python threads
    take turns rather than calling hipb_mm at the same instant."""
    calls = _thread_calls()
    threads = 4
    expected = [[f() for f in calls(i)] for i in range(threads)]
    torch.cuda.synchronize()
    barrier = threading.Barrier(threads)

    def work(i):
        barrier.wait()
        rounds = [[f() for f in calls(i)] for _ in range(20)]
        torch.cuda.synchronize()
        return rounds

    with ThreadPoolExecutor(threads) as pool:
        results = list(pool.map(work, range(threads)))
    for i, rounds in enumerate(results):
        for outs in rounds:
            for got, want in zip(outs, expected[i]):
                assert torch.equal(got, want)


def test_destroy_drops_other_threads_descriptors():
    """A thread that outlives an extension destroy/create rebuilds its
    descriptors for the new extension."""
    calls = _thread_calls()
    with ThreadPoolExecutor(1) as worker:

        def run():
            outs = [f() for f in calls(1)]
            torch.cuda.synchronize()
            return outs

        before = worker.submit(run).result()
        _fresh_extension()
        after = worker.submit(run).result()
    fresh = run()
    for b, a, f in zip(before, after, fresh):
        assert torch.equal(b, f) and torch.equal(a, f)


def test_short_lived_threads():
    """Descriptors of a thread are released when it exits."""
    calls = _thread_calls()
    expected = [f() for f in calls(2)]
    out = []
    for _ in range(16):
        t = threading.Thread(target=lambda: out.append([f() for f in calls(2)]))
        t.start()
        t.join()
    torch.cuda.synchronize()
    assert len(out) == 16
    for outs in out:
        for got, want in zip(outs, expected):
            assert torch.equal(got, want)


def main():
    status = pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    return 0 if status == pytest.ExitCode.OK else 1


if __name__ == "__main__":
    sys.exit(main())
