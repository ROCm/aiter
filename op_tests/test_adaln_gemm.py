# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""AdaLN modulation GEMMs (aiter.ops.adaln_gemm): every manifest row against torch and an fp64 reference."""
import pytest
import torch

from aiter.ops.adaln_gemm import _manifest, adaln_dgrad, adaln_fwd, adaln_wgrad

ROWS = sorted(_manifest())
pytestmark = pytest.mark.skipif(not ROWS or not torch.cuda.is_available(), reason="no AdaLN GEMM kernels / GPU")


def _err(x, gold):
    return ((x.double() - gold).abs().max() / gold.abs().max()).item()


@pytest.mark.parametrize("row", ROWS, ids=[f"pass{p}_N{n}_K{k}" for p, n, k in ROWS])
def test_adaln_gemm(row):
    p, N, K = row
    g = torch.Generator(device="cuda").manual_seed(N + K + p)
    rnd = lambda *s, scale=1.0: torch.randn(*s, device="cuda", dtype=torch.bfloat16, generator=g) * scale  # noqa: E731
    if p == 2:
        dy, x = rnd(32, N), rnd(32, K)
        out = torch.empty(N, K, device="cuda", dtype=torch.bfloat16)
        adaln_wgrad(dy, x, out)
        assert torch.equal(out.view(torch.int16), torch.mm(dy.t(), x).view(torch.int16))
        return
    if p == 0:
        x, w, b = rnd(32, K), rnd(N, K, scale=0.02), rnd(N)
        gold = x.double() @ w.double().t() + b.double()
        ref = torch.addmm(b, x, w.t())
        run = lambda o: adaln_fwd(x, w, b, o)  # noqa: E731
        out = torch.empty(32, N, device="cuda", dtype=torch.bfloat16)
    else:
        dy, w = rnd(32, N), rnd(N, K, scale=0.02)
        gold = dy.double() @ w.double()
        ref = torch.mm(dy, w)
        run = lambda o: adaln_dgrad(dy, w, o)  # noqa: E731
        out = torch.empty(32, K, device="cuda", dtype=torch.bfloat16)
    run(out)
    first = out.clone()
    for _ in range(3):  # deterministic, and the dgrad counters reset themselves between calls
        out.fill_(float("nan"))
        run(out)
        assert torch.equal(out.view(torch.int16), first.view(torch.int16))
    assert _err(out, gold) <= 1.05 * _err(ref, gold) + 1e-6
