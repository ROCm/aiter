# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""`hipb_mm` keeps the hipBLASLt layouts, descriptor and algorithm of an
unscaled GEMM with a given solution index, and rebuilds nothing on later calls
but the bias pointer. A call served that way must give what a freshly built
descriptor gives, bit for bit wherever the kernel itself is reproducible; every
result is also held to a float32 reference. The kept descriptors are per
thread.

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
from aiter.ops.gradlib import getHipblasltKernelName, hipb_findallsols

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU with hipBLASLt"
)

BF16, FP16 = dtypes.bf16, dtypes.fp16
# Against the float32 reference: bf16 output keeps 8 significant bits.
ATOL = RTOL = 1e-2


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


def _reference(x, w, bias=None, otype=None):
    """``x @ w.t() + bias`` computed by torch in float32."""
    out = x.float() @ w.float().t()
    if bias is not None:
        out += bias.float()
    return out.to(otype or x.dtype)


def _solution(x, wt, bias=None, otype=None, also=()):
    """The first of a few solutions whose results, for ``x`` and every input in
    ``also``, agree between two freshly built descriptors, so that reuse can be
    checked bit for bit; else the first solution."""
    sols = hipb_findallsols(x, wt, bias=bias, out_dtype=otype)
    if not sols:
        pytest.skip("no hipBLASLt solution for this problem")
    inputs = (x, *also)
    for sol in sols[:8]:
        first = [hipb_mm(a, wt, sol, bias=bias, out_dtype=otype) for a in inputs]
        _fresh_extension()
        again = [hipb_mm(a, wt, sol, bias=bias, out_dtype=otype) for a in inputs]
        torch.cuda.synchronize()
        if all(torch.equal(f, a) for f, a in zip(first, again)):
            return sol
    return sols[0]


def _kernel(sol):
    try:
        return getHipblasltKernelName(sol)
    except RuntimeError:  # only names the kernel in a failure message
        return "?"


def _differences(a, b):
    differ = a != b
    delta = (a.float() - b.float()).abs().max().item()
    return (
        f"{int(differ.sum())} of {a.numel()} elements differ, "
        f"max |diff| {delta:.3g}, first at {differ.nonzero()[:4].tolist()}"
    )


def _check_reuse(kept, call, ref, sol):
    """``kept`` are results of ``call`` served by kept descriptors.

    ``call`` is repeated on freshly built extensions. If those results agree
    bit for bit, the kernel is reproducible for this input, and every kept
    result must give the same bits: a reuse bug (a stale bias pointer, another
    configuration's layouts) changes them. If they do not agree, the kernel
    itself varies run to run (for instance it splits K with atomics), and kept
    results are held to the float32 reference ``ref`` instead, which those bugs
    still fail. Fresh results are always held to ``ref``."""
    fresh = []

    def reproducible(samples):
        for _ in range(samples):
            _fresh_extension()
            fresh.append(call())
        torch.cuda.synchronize()
        return all(torch.equal(fresh[0], f) for f in fresh[1:])

    def where():
        return f"solution {sol} ({_kernel(sol)})"

    exact = reproducible(2)
    if exact and not all(torch.equal(k, fresh[0]) for k in kept):
        # Two agreeing samples do not make a kernel reproducible.
        exact = reproducible(4)
    for f in fresh:
        torch.testing.assert_close(
            f, ref, atol=ATOL, rtol=RTOL, msg=lambda m: f"fresh, {where()}: {m}"
        )
    for i, k in enumerate(kept):
        if exact:
            assert torch.equal(k, fresh[0]), (
                f"kept result {i} vs {len(fresh)} identical fresh ones, {where()}: "
                + _differences(k, fresh[0])
            )
        else:
            torch.testing.assert_close(
                k,
                ref,
                atol=ATOL,
                rtol=RTOL,
                msg=lambda m, i=i: f"kept result {i}, fresh ones vary, {where()}: {m}",
            )


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
    # The second call, at least, reuses the descriptors.
    kept = [hipb_mm(x, w.t(), sol, bias=b1, out_dtype=otype) for _ in range(2)]
    if has_bias:
        # Same configuration, another bias: only the pointer may change.
        b2 = _rand(n, dtype=dtype)
        other = hipb_mm(x, w.t(), sol, bias=b2, out_dtype=otype)
        _check_reuse(
            [other],
            lambda: hipb_mm(x, w.t(), sol, bias=b2, out_dtype=otype),
            _reference(x, w, b2, otype),
            sol,
        )
    _check_reuse(
        kept,
        lambda: hipb_mm(x, w.t(), sol, bias=b1, out_dtype=otype),
        _reference(x, w, b1, otype),
        sol,
    )


def test_configurations_do_not_mix():
    """Interleaved shapes, leading dimensions and bias on/off each keep their
    own descriptors."""
    k = 512
    big = _rand(256, 2 * k)
    x_strided = big[:, :k]  # lda = 2k
    x = x_strided.contiguous()
    w = _rand(384, k, scale=k**-0.5)
    b = _rand(384)
    sol = _solution(x, w.t(), also=[x_strided, x[:128]])
    sol_b = _solution(x, w.t(), b)
    calls = [
        (lambda: hipb_mm(x, w.t(), sol), _reference(x, w), sol),
        (lambda: hipb_mm(x_strided, w.t(), sol), _reference(x, w), sol),
        (lambda: hipb_mm(x[:128], w.t(), sol), _reference(x[:128], w), sol),
        (lambda: hipb_mm(x, w.t(), sol_b, bias=b), _reference(x, w, b), sol_b),
    ]
    kept = [[call() for call, _, _ in calls] for _ in range(2)]
    for i, (call, ref, s) in enumerate(calls):
        _check_reuse([kept[0][i], kept[1][i]], call, ref, s)


def test_many_configurations():
    """More configurations than are kept (one per leading dimension here): the
    set is dropped and rebuilt. Calls before and after the drop, on kept or
    rebuilt descriptors, compute what a fresh descriptor does."""
    # K spans several unroll steps: at K = 64, one step, the hand-written gfx950
    # kernel hipBLASLt lists first for this problem can read LDS before its loads
    # land.
    m, n, k, step, count = 64, 64, 512, 64, 1100
    w = _rand(n, k, scale=k**-0.5)
    buf = _rand(m * (k + step * count))
    views = [buf.as_strided((m, k), (k + step * i, 1)) for i in range(count)]
    checked = (0, 1, 511, 1023, 1024, 1099)
    sol = _solution(views[0], w.t(), also=[views[i] for i in checked[1:]])
    kept = [hipb_mm(x, w.t(), sol) for x in views]
    # Past the drop, the last views still have their descriptors; the first
    # ones are built again.
    again = [hipb_mm(views[i], w.t(), sol) for i in checked]
    for i, other in zip(checked, again):
        x = views[i]
        _check_reuse(
            [kept[i], other],
            lambda x=x: hipb_mm(x, w.t(), sol),
            _reference(x, w),
            sol,
        )


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
    graph.reset()
    _check_reuse([out], lambda: hipb_mm(x, w.t(), sol), _reference(x, w), sol)


def _thread_calls():
    """Interleaved configurations, including one configuration that every
    thread calls with a bias of its own. Returns a table from a key to (call,
    reference, solution), and the keys thread ``i`` calls."""
    k = 512
    x = _rand(256, k)
    w = _rand(384, k, scale=k**-0.5)
    sol = _solution(x, w.t())
    biases = [_rand(384) for _ in range(4)]
    sol_b = _solution(x, w.t(), biases[0], also=[x[:128]])
    x2, w2 = _rand(96, 768), _rand(320, 768, scale=768**-0.5)
    sol2 = _solution(x2, w2.t())
    table = {
        "x": (lambda: hipb_mm(x, w.t(), sol), _reference(x, w), sol),
        "x2": (lambda: hipb_mm(x2, w2.t(), sol2), _reference(x2, w2), sol2),
    }
    for j, b in enumerate(biases):
        table["x+b", j] = (
            lambda b=b: hipb_mm(x, w.t(), sol_b, bias=b),
            _reference(x, w, b),
            sol_b,
        )
        table["x[:128]+b", j] = (
            lambda b=b: hipb_mm(x[:128], w.t(), sol_b, bias=b),
            _reference(x[:128], w, b),
            sol_b,
        )

    def keys(i):
        j = i % len(biases)
        return [("x+b", j), "x", "x2", ("x[:128]+b", j)]

    return table, keys


def _run(table, keys):
    return [(key, table[key][0]()) for key in keys]


def _check_results(table, results):
    """``results``: (key, output) pairs, each served by kept descriptors."""
    kept = {}
    for key, out in results:
        kept.setdefault(key, []).append(out)
    for key, outs in kept.items():
        _check_reuse(outs, *table[key])


def test_threads_keep_their_own_descriptors():
    """Each thread builds and reuses its own descriptors; results match fresh
    ones. The binding holds the GIL, so Python threads take turns rather than
    calling hipb_mm at the same instant."""
    table, keys = _thread_calls()
    threads = 4
    barrier = threading.Barrier(threads)

    def work(i):
        barrier.wait()
        rounds = [r for _ in range(20) for r in _run(table, keys(i))]
        torch.cuda.synchronize()
        return rounds

    with ThreadPoolExecutor(threads) as pool:
        results = list(pool.map(work, range(threads)))
    _check_results(table, (r for rounds in results for r in rounds))


def test_destroy_drops_other_threads_descriptors():
    """A thread that outlives an extension destroy/create rebuilds its
    descriptors for the new extension."""
    table, keys = _thread_calls()
    with ThreadPoolExecutor(1) as worker:

        def run():
            outs = _run(table, keys(1))
            torch.cuda.synchronize()
            return outs

        before = worker.submit(run).result()
        _fresh_extension()
        after = worker.submit(run).result()
    _check_results(table, before + after)


def test_short_lived_threads():
    """Descriptors of a thread are released when it exits."""
    table, keys = _thread_calls()
    out = []
    for _ in range(16):
        t = threading.Thread(target=lambda: out.append(_run(table, keys(2))))
        t.start()
        t.join()
    torch.cuda.synchronize()
    assert len(out) == 16
    _check_results(table, (r for outs in out for r in outs))


def main():
    status = pytest.main([__file__, "-q", "-p", "no:cacheprovider"])
    return 0 if status == pytest.ExitCode.OK else 1


if __name__ == "__main__":
    sys.exit(main())
