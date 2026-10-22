# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""RMSNorm contract and changing-input graph replay."""

import pytest
import test_fused_norm_rope_attn_rope_cast_fp8_qk as oracle
import torch
from test_fused_norm_rope_attn_rope_cast import simple_inputs
from test_fused_norm_rope_attn_rope_cast_fp8_qk import dense, fixture

from aiter import fused_norm_rope_attn_rope_cast
from aiter.ops.fused_norm_rope_attn_rope_cast import fused_norm_rope_attn_rope_cast_fwd

pytestmark = oracle.pytestmark


@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize(
    "bad", ["rotated", "prepared", "zero", "negative", "nan", "infinity", "underflow"]
)
def test_norm_contract(direct, bad):
    args = simple_inputs()
    kw = {"q_norm": True, "q_is_rotated": False}
    if bad == "rotated":
        kw["q_is_rotated"] = True
    elif bad == "prepared":
        args[0] = args[0].to(torch.float8_e4m3fn)
        kw.update(
            fp8_qk=True,
            q_is_rotated=True,
            q_scale=torch.full((1, 64, 16), 127, device="cuda", dtype=torch.uint8),
        )
    else:
        kw["norm_eps"] = {
            "zero": 0,
            "negative": -1,
            "nan": float("nan"),
            "infinity": float("inf"),
            "underflow": 1e-40,
        }[bad]
    with pytest.raises((ValueError, RuntimeError), match="q_norm|norm_eps"):
        if direct:
            q = args[0]
            outputs = [
                torch.empty((1, 8, 4096), device=q.device, dtype=torch.float8_e4m3fn),
                torch.empty((1, 8, 128), device=q.device, dtype=torch.float8_e8m0fnu),
                torch.empty((1, 64), device=q.device),
                torch.empty((1, 64), device=q.device),
            ]
            fused_norm_rope_attn_rope_cast_fwd(
                *args, *outputs, None, None, None, None, None, 512**-0.5, **kw
            )
        else:
            fused_norm_rope_attn_rope_cast(*args, **kw)


@pytest.mark.parametrize("tokens", [0, 3])
def test_norm_zero_q(tokens):
    args = simple_inputs(tokens)
    out = fused_norm_rope_attn_rope_cast(*args, q_norm=True, q_is_rotated=False)
    assert all(torch.isfinite(x.float()).all() for x in out)
    assert torch.count_nonzero(out[2]) == torch.count_nonzero(out[3]) == 0


@pytest.mark.parametrize("page", [0, 128])
def test_norm_recomputed_on_graph_replay(page):
    from test_fused_norm_rope_attn_rope_cast import csr

    args, kw, rows, kv, sink = fixture(128, page, True)
    q, pool, _, _, cos, sin, pos = args
    rows = [
        (torch.arange(1153, device=q.device) * 17 + i) % len(kv)
        for i in range(len(rows))
    ]
    ids, ptr = csr(rows, torch.int32, -1)
    if not page:
        ids = ids.long() * 1056 + 9
    args = (q, pool, ids, ptr, cos, sin, pos)
    kw.update(
        q_norm=True, norm_eps=0.125, q_is_rotated=False, fp8_qk=True, attn_sink=sink
    )
    out = fused_norm_rope_attn_rope_cast(*args, **kw)
    buffers = dict(zip(("out", "out_scale", "max_logits", "lse"), out))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_norm_rope_attn_rope_cast(*args, **kw, **buffers)
    # Make the norm scale change after capture, including zero-valued heads.
    q.mul_(0.25)
    q[:, ::11] = 0
    graph.replay()
    normalized = (
        q.float()
        * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + kw["norm_eps"])
    ).bfloat16()
    _, mx, lse = dense(normalized, kv, rows, cos, sin, pos, sink, True)
    torch.testing.assert_close(out[2], mx, rtol=1e-4, atol=3e-5)
    torch.testing.assert_close(out[3], lse, rtol=1e-4, atol=3e-5)
