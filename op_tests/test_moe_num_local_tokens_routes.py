# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""fused_moe(num_local_tokens=...) across the 2-stage routes.

The sort pads sorted_token_ids with the token capacity M (rows of topk_ids),
so every consumer sees pad ids >= M. These tests run each route with
nlt < M, nlt = M - 1 and nlt = M and check the first nlt rows against an
unquantized (or dequantized-weight) reference. Rows past nlt are unspecified
and are not checked.

The timing test guards the CK stage-2 slowdown that pad ids below M caused:
every pad entry was computed and accumulated into one row.
"""

import csv
import functools
import os
import tempfile

import pytest
import torch
import torch.nn.functional as F

import aiter
import aiter.fused_moe as fm
from aiter import ActivationType, QuantType, dtypes
from aiter.fused_moe import fused_moe, fused_topk
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_cu_num
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.quant import per_1x32_f8_scale_f8_quant, pertoken_quant
from aiter.ops.shuffle import (
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)

_ROCM = torch.cuda.is_available() and bool(torch.version.hip)
_ARCH = torch.cuda.get_device_properties(0).gcnArchName if _ROCM else ""
pytestmark = pytest.mark.skipif(
    not (_ROCM and "gfx950" in _ARCH), reason="routes are tuned for gfx950"
)

_CSV_HEADER = (
    "cu_num,token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,"
    "q_dtype_w,q_type,use_g1u1,doweight_stage1,block_m,ksplit,us1,kernelName1,"
    "err1,us2,kernelName2,err2,us,run_1stage,tflops,bw,_tag"
)


def logits_diff(ref, out):
    r, o = ref.double(), out.double()
    return (1 - 2 * (r * o).sum() / (r * r + o * o).sum()).item()


def _moe_ref(x, w1, w2, ids, weights, *, limit=None):
    """Unquantized silu(gate) * up reference, accumulated in fp32."""
    expected = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    for e in range(w1.shape[0]):
        rows, slots = torch.where(ids == e)
        if rows.numel() == 0:
            continue
        gate, up = F.linear(x[rows].float(), w1[e].float()).chunk(2, dim=-1)
        if limit is not None:
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        y = F.linear(F.silu(gate) * up, w2[e].float())
        expected.index_add_(0, rows, y * weights[rows, slots, None].float())
    return expected


def _routing(x, E, topk):
    logits = torch.randn(x.shape[0], E, dtype=x.dtype, device=x.device)
    return fused_topk(x, logits, topk, True)


class Case:
    def __init__(self, M, D, call, ref, tol, tuned_row=None):
        self.M, self.D, self.call, self.ref, self.tol = M, D, call, ref, tol
        self.tuned_row = tuned_row


def _bf16_case(M, D, I, E, topk, *, limit=None):
    torch.manual_seed(0)
    x = torch.randn(M, D, dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn(E, 2 * I, D, dtype=dtypes.bf16, device="cuda") * 0.05
    w2 = torch.randn(E, D, I, dtype=dtypes.bf16, device="cuda") * 0.05
    tw, ti = _routing(x, E, topk)
    W1, W2 = shuffle_weight(w1, layout=(16, 16)), shuffle_weight(w2, layout=(16, 16))
    kw = {} if limit is None else {"swiglu_limit": limit}

    def call(nlt, output=None):
        return fused_moe(
            x,
            W1,
            W2,
            tw,
            ti,
            quant_type=QuantType.No,
            activation=ActivationType.Silu,
            num_local_tokens=nlt,
            output=output,
            **kw,
        )

    return Case(M, D, call, _moe_ref(x, w1, w2, ti, tw, limit=limit), 1e-3)


def _per_token_case(M, D, I, E, topk, qdtype):
    torch.manual_seed(0)
    x = torch.randn(M, D, dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn(E, 2 * I, D, dtype=dtypes.bf16, device="cuda") * 0.05
    w2 = torch.randn(E, D, I, dtype=dtypes.bf16, device="cuda") * 0.05
    tw, ti = _routing(x, E, topk)
    w1q, w1s = pertoken_quant(w1, quant_dtype=qdtype)
    w2q, w2s = pertoken_quant(w2, quant_dtype=qdtype)
    W1, W2 = shuffle_weight(w1q, layout=(16, 16)), shuffle_weight(w2q, layout=(16, 16))

    def call(nlt, output=None):
        return fused_moe(
            x,
            W1,
            W2,
            tw,
            ti,
            quant_type=QuantType.per_Token,
            activation=ActivationType.Silu,
            w1_scale=w1s,
            w2_scale=w2s,
            num_local_tokens=nlt,
            output=output,
        )

    return Case(M, D, call, _moe_ref(x, w1, w2, ti, tw), 3e-2)


def _a16w4_case(M, D, I, E, topk, act, *, gate_up, act_fp8, tuned=None):
    """MXFP4 weights; the reference uses the dequantized weights."""
    torch.manual_seed(0)
    x = torch.randn(M, D, dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn(E, 2 * I, D, dtype=dtypes.bf16, device="cuda")
    w2 = torch.randn(E, D, I, dtype=dtypes.bf16, device="cuda")
    tw, ti = _routing(x, E, topk)
    quant = aiter.get_torch_quant(QuantType.per_1x32)
    w1q, w1s = quant(w1, quant_dtype=dtypes.fp4x2)
    w2q, w2s = quant(w2, quant_dtype=dtypes.fp4x2)
    w1q, w2q = w1q.view(E, 2 * I, D // 2), w2q.view(E, D, I // 2)
    del w1, w2
    situ = {"situ_beta": 1.0, "situ_linear_beta": 1.0}
    kw = situ if act == ActivationType.Situv2 else {}
    if act_fp8:
        a1, a1s = per_1x32_f8_scale_f8_quant(
            x, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    else:
        a1, a1s = x, None
    o1 = fm.torch_moe_stage1(
        a1,
        w1q,
        w2q,
        tw,
        ti,
        dtype=dtypes.bf16,
        activation=act,
        quant_type=QuantType.per_1x32,
        a1_scale=a1s,
        w1_scale=w1s,
        w1_bias=None,
        **kw,
    )
    if act_fp8:
        a2, a2s = per_1x32_f8_scale_f8_quant(
            o1, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    else:
        a2, a2s = o1, None
    ref = fm.torch_moe_stage2(
        a2.view(M, topk, -1),
        w1q,
        w2q,
        tw,
        ti,
        dtype=dtypes.bf16,
        quant_type=QuantType.per_1x32,
        w2_scale=w2s,
        a2_scale=a2s,
        w2_bias=None,
        doweight=True,
    ).float()
    W1 = shuffle_weight_a16w4(w1q, 16, gate_up)
    W2 = shuffle_weight_a16w4(w2q, 16, False)
    S1 = shuffle_scale_a16w4(w1s, E, gate_up)
    S2 = shuffle_scale_a16w4(w2s, E, False)
    mode = GateMode.INTERLEAVE.value if gate_up else GateMode.SEPARATED.value

    def call(nlt, output=None):
        return fused_moe(
            x,
            W1,
            W2,
            tw,
            ti,
            w1_scale=S1,
            w2_scale=S2,
            quant_type=QuantType.per_1x32,
            activation=act,
            gate_mode=mode,
            num_local_tokens=nlt,
            output=output,
            **({"beta": 1.0, "linear_beta": 1.0} if kw else {}),
        )

    return Case(M, D, call, ref, 3e-2, tuned_row=tuned)


_OPUS_S1 = "cktile_a8w4_bm32"
_OPUS_S2 = "opus_moe2_afp8_wfp4_fp8_t32x256x256_sbm32_occ4_rbn2240"


def _opus_row(M, D, I, E, topk):
    return [
        get_cu_num(),
        M,
        D,
        I,
        E,
        topk,
        "ActivationType.Swiglu",
        "torch.bfloat16",
        "torch.float8_e4m3fn",
        "torch.float4_e2m1fn_x2",
        "QuantType.per_1x32",
        1,
        0,
        32,
        0,
        0,
        _OPUS_S1,
        "0.0%",
        0,
        _OPUS_S2,
        "0.0%",
        0,
        0,
        0,
        0,
        "",
    ]


ROUTES = {
    "ck_bf16": lambda: _bf16_case(64, 2048, 192, 128, 8),
    "ck_fp8_per_token": lambda: _per_token_case(8, 2560, 256, 8, 2, dtypes.fp8),
    "asm_stage1_int8": lambda: _per_token_case(64, 2560, 256, 8, 2, dtypes.i8),
    "cktile_a16w4": lambda: _a16w4_case(
        64, 1024, 256, 8, 2, ActivationType.Swiglu, gate_up=False, act_fp8=False
    ),
    "flydsl_a16w4": lambda: _a16w4_case(
        64, 1024, 256, 8, 2, ActivationType.Situv2, gate_up=False, act_fp8=False
    ),
    "flydsl_a16w16": lambda: _bf16_case(64, 1024, 256, 8, 2, limit=7.0),
    "opus_a8w4": lambda: _a16w4_case(
        128,
        3072,
        3072,
        128,
        4,
        ActivationType.Swiglu,
        gate_up=True,
        act_fp8=True,
        tuned=_opus_row(128, 3072, 3072, 128, 4),
    ),
}


@functools.cache
def _case(name):
    return ROUTES[name]()


def _use_tuned_row(monkeypatch, row):
    path = os.path.join(tempfile.mkdtemp(), "tuned.csv")
    with open(path, "w", newline="") as f:
        f.write(_CSV_HEADER + "\n")
        csv.writer(f).writerow(row)
    monkeypatch.setattr(
        type(AITER_CONFIGS),
        "AITER_CONFIG_FMOE_FILE",
        property(lambda self: path),
        raising=False,
    )
    monkeypatch.setattr(fm, "cfg_2stages", None)
    fm.get_2stage_cfgs.cache_clear()


def _nlt(tokens):
    return torch.tensor([tokens], dtype=torch.int32, device="cuda")


@pytest.mark.parametrize("route", list(ROUTES))
@pytest.mark.parametrize("which", ["three_quarters", "capacity_minus_one", "capacity"])
def test_valid_rows_match_the_reference(route, which, monkeypatch):
    case = _case(route)
    if case.tuned_row is not None:
        _use_tuned_row(monkeypatch, case.tuned_row)
    M = case.M
    tokens = {"three_quarters": M * 3 // 4, "capacity_minus_one": M - 1}.get(which, M)
    out = case.call(_nlt(tokens))
    torch.cuda.synchronize()
    assert out.shape == (M, case.D)
    valid = out[:tokens]
    assert torch.isfinite(valid.float()).all()
    diff = logits_diff(case.ref[:tokens], valid)
    assert diff < case.tol, f"{route} nlt={tokens}/{M}: logits_diff {diff:.3e}"


def _time(fn):
    fn()
    torch.cuda.synchronize()
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    best = float("inf")
    for _ in range(5):
        ev[0].record()
        for _ in range(20):
            fn()
        ev[1].record()
        torch.cuda.synchronize()
        best = min(best, ev[0].elapsed_time(ev[1]) * 50)
    return best


def _stage2_time_us(M, D, I, E, topk, tokens, monkeypatch):
    torch.manual_seed(0)
    x = torch.randn(M, D, dtype=dtypes.bf16, device="cuda")
    w1 = torch.randn(E, 2 * I, D, dtype=dtypes.bf16, device="cuda") * 0.05
    w2 = torch.randn(E, D, I, dtype=dtypes.bf16, device="cuda") * 0.05
    tw, ti = _routing(x, E, topk)
    W1, W2 = shuffle_weight(w1, layout=(16, 16)), shuffle_weight(w2, layout=(16, 16))
    captured = {}
    real = aiter.ck_moe_stage2_fwd

    def spy(*args, **kwargs):
        captured["call"] = (args, dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(aiter, "ck_moe_stage2_fwd", spy)
    fm.get_2stage_cfgs.cache_clear()
    nlt = None if tokens is None else _nlt(tokens)
    fused_moe(
        x,
        W1,
        W2,
        tw,
        ti,
        quant_type=QuantType.No,
        activation=ActivationType.Silu,
        num_local_tokens=nlt,
        output=output,
    )
    torch.cuda.synchronize()
    monkeypatch.undo()
    args, kwargs = captured["call"]
    out = args[6]

    def stage2():
        out.zero_()
        return real(*args, **kwargs)

    return _time(stage2)


@pytest.mark.parametrize(
    "M,tokens",
    [(64, 48), (64, 63), (1024, 128), (1024, 1000)],
)
def test_stage2_time_does_not_depend_on_the_pad_count(M, tokens, monkeypatch):
    # Before the sentinel fix this was 11x-28x slower than nlt = M.
    shape = (M, 2048, 192, 128, 8)
    full = _stage2_time_us(*shape, M, monkeypatch)
    partial = _stage2_time_us(*shape, tokens, monkeypatch)
    assert (
        partial < 2.0 * full
    ), f"stage 2: {partial:.1f} us (nlt={tokens}) vs {full:.1f} us (nlt={M})"
