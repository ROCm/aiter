# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Slice 1: validate fused_moe(out=, residual=) semantics.

The public ``fused_moe`` wrapper gained two optional args used by the folded
MoE-sum + all-reduce path:

  * ``residual`` -- a [tokens, model_dim] addend (the shared-expert output)
    added after the routed-expert topk combine (pre all-reduce).
  * ``out``      -- a caller-provided [tokens, model_dim] output buffer the
    combined result is written into (so it can be the CustomAllreduce
    registered IPC buffer).

Semantics (validated here at world_size=1, where the all-reduce is identity):

    fused_moe(..., out=buf, residual=r)  ==  fused_moe(...) + r     (and returns buf)

This is the quant-agnostic API/semantics slice. The true one-pass fold (down-GEMM
writing straight into ``out`` and the sort-prologue zero-init initialised with
``residual``) plus the cross-rank reduce are validated on the MXFP4/TP hardware.

Run (1 GPU):
    python op_tests/test_fused_moe_out_residual_single_gpu.py
"""

import torch

# `import aiter.fused_moe` registers torch.ops.aiter.fused_moe_ (exercised by
# the trace-safety check below).
from aiter.fused_moe import fused_moe, fused_topk
from aiter.test_common import checkAllclose


def _build_inputs(token, model_dim, inter_dim, E, topk, dtype):
    torch.manual_seed(0)
    x = torch.randn((token, model_dim), dtype=dtype, device="cuda")
    # g1u1 layout (gate+up) for Silu.
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype, device="cuda") / 10.0
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype, device="cuda") / 10.0
    score = torch.randn((token, E), dtype=dtype, device="cuda")
    topk_weights, topk_ids = fused_topk(x, score, topk, True)
    return x, w1, w2, topk_weights, topk_ids


def run_case(token=64, model_dim=512, inter_dim=256, E=8, topk=2, dtype=torch.bfloat16):
    x, w1, w2, tw, tid = _build_inputs(token, model_dim, inter_dim, E, topk, dtype)

    base = fused_moe(x, w1, w2, tw, tid)
    assert base.shape == (token, model_dim), base.shape

    # 1) out= : same values, and the returned tensor IS the caller buffer.
    buf = torch.empty_like(base)
    ret = fused_moe(x, w1, w2, tw, tid, out=buf)
    assert ret.data_ptr() == buf.data_ptr(), "out= must return the caller buffer"
    e_out = checkAllclose(
        base, ret, msg="out= equals plain fused_moe", atol=2e-2, rtol=2e-2
    )

    # 2) residual= : equals base + residual.
    r = torch.randn_like(base)
    ret2 = fused_moe(x, w1, w2, tw, tid, residual=r)
    e_res = checkAllclose(
        base.float() + r.float(),
        ret2.float(),
        msg="residual= equals fused_moe + residual",
        atol=2e-2,
        rtol=2e-2,
    )

    # 3) out= and residual= together, into the buffer.
    buf2 = torch.empty_like(base)
    ret3 = fused_moe(x, w1, w2, tw, tid, out=buf2, residual=r)
    assert ret3.data_ptr() == buf2.data_ptr()
    e_both = checkAllclose(
        base.float() + r.float(),
        ret3.float(),
        msg="out=+residual= equals fused_moe + residual (in buffer)",
        atol=2e-2,
        rtol=2e-2,
    )

    # 4) shape guards.
    for bad in (torch.empty((token + 1, model_dim), dtype=dtype, device="cuda"),):
        try:
            fused_moe(x, w1, w2, tw, tid, out=bad)
            raise AssertionError("expected assert on bad out shape")
        except AssertionError as e:
            if "out buffer mismatch" not in str(e):
                raise

    print(
        {
            "token": token,
            "model_dim": model_dim,
            "E": E,
            "topk": topk,
            "dtype": str(dtype),
            "err_out": e_out,
            "err_res": e_res,
            "err_both": e_both,
        }
    )
    return True


def run_trace_safety(
    token=64, model_dim=512, inter_dim=256, E=8, topk=2, dtype=torch.bfloat16
):
    """The ``out``/``residual`` args must be trace-safe.

    ``fused_moe_`` is registered as ``torch.ops.aiter.fused_moe_`` with a fake
    (``fused_moe_fake``). We assert two things:

      (a) the fake handles ``out``/``residual`` under ``FakeTensorMode`` (this is
          the path Dynamo takes when it meets the op) -- correct shape and, when
          ``out`` is given, the result aliases ``out``;
      (b) ``torch.compile`` executing a function that calls the op with
          ``out``/``residual`` produces no error at the op boundary and matches
          eager numerically. Since the op is an opaque custom op, it lowers to a
          single graph node rather than graph-breaking.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    x, w1, w2, tw, tid = _build_inputs(token, model_dim, inter_dim, E, topk, dtype)

    # (a) fake-impl under FakeTensorMode with out= and residual=.
    fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
    with fake_mode:
        fx = torch.empty((token, model_dim), dtype=dtype, device="cuda")
        fout = torch.empty((token, model_dim), dtype=dtype, device="cuda")
        fres = torch.empty((token, model_dim), dtype=dtype, device="cuda")
        fw1 = torch.empty_like(w1)
        fw2 = torch.empty_like(w2)
        ftw = torch.empty_like(tw)
        ftid = torch.empty_like(tid)
        fr = torch.ops.aiter.fused_moe_(
            fx, fw1, fw2, ftw, ftid, out=fout, residual=fres
        )
        assert tuple(fr.shape) == (token, model_dim), fr.shape
        assert fr.dtype == dtype
    print("  [trace] FakeTensorMode fused_moe_(out=,residual=) ok:", tuple(fr.shape))

    # (b) torch.compile a fn that calls the op with out=/residual=, match eager.
    r = torch.randn((token, model_dim), dtype=dtype, device="cuda")

    def fn(x, w1, w2, tw, tid, out, residual):
        return torch.ops.aiter.fused_moe_(
            x, w1, w2, tw, tid, out=out, residual=residual
        )

    eager_buf = torch.empty((token, model_dim), dtype=dtype, device="cuda")
    eager = fn(x, w1, w2, tw, tid, eager_buf, r).clone()

    torch._dynamo.reset()
    compiled = torch.compile(fn, backend="eager", fullgraph=True)
    comp_buf = torch.empty((token, model_dim), dtype=dtype, device="cuda")
    compiled_out = compiled(x, w1, w2, tw, tid, comp_buf, r)

    e_trace = checkAllclose(
        eager.float(),
        compiled_out.float(),
        msg="torch.compile(fullgraph=True) fused_moe_(out=,residual=) == eager",
        atol=2e-2,
        rtol=2e-2,
    )
    print("  [trace] compile(fullgraph=True) numeric ok, err:", e_trace)
    return True


if __name__ == "__main__":
    run_case()
    run_trace_safety()
    print("PASS")
