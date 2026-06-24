# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""SCRATCH bench (do not ship): a16w4 CK-Tile (via fused_moe) head-to-head.

Replicates the a16w4 (QuantType.per_1x32, bf16 activation, fp4x2 weight) input
build from op_tests/test_moe_2stage.py::test_fmoe and calls the *public*
fused_moe directly on gfx942, bypassing the gfx950-only skip (that skip lives in
the harness, NOT in fused_moe). It instruments which backend get_2stage_cfgs
actually selects (CK-Tile / FlyDSL / asm / ck2stages) and the runtime q_dtype_a,
times fused_moe end-to-end with run_perftest, and verifies correctness against
the SAME torch reference (torch_moe_stage1/torch_moe_stage2) at the EXISTING
strict tolerance (err==0 and logits_diff<=0.01). No tolerances are widened and no
library code is modified.

Usage:
    HIP_VISIBLE_DEVICES=1 python op_tests/_bench_a16w4_cktile_vs_flydsl_gfx942.py
"""

import os
import functools
import logging

import torch

import aiter
from aiter import dtypes
from aiter.test_common import checkAllclose, run_perftest
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.ops.flydsl.moe_common import GateMode

import aiter.fused_moe as fm
from aiter.fused_moe import (
    fused_topk,
    fused_moe,
    get_2stage_cfgs,
    get_padded_M,
    torch_moe_stage1,
    torch_moe_stage2,
)
from aiter.jit.utils.chip_info import get_gfx

torch.set_default_device("cuda")


def _calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return float(1 - sim)


def _func_name(partial_or_fn):
    f = getattr(partial_or_fn, "func", partial_or_fn)
    return getattr(f, "__name__", repr(f))


def _backend_label(meta):
    s1 = _func_name(meta.stage1)
    s2 = _func_name(meta.stage2)

    def classify(name):
        if "cktile" in name:
            return "CK-Tile"
        if "flydsl" in name:
            return "FlyDSL"
        if name.startswith("asm_"):
            return "ASM"
        if "ck_moe" in name or "ck2stages" in name:
            return "CK2stages"
        return name

    return f"stage1={classify(s1)}({s1}), stage2={classify(s2)}({s2})"


# --- instrument get_2stage_cfgs so we capture the real dispatch ----------------
_captured = {}


def _wrap_get_2stage_cfgs(orig):
    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        meta = orig(*args, **kwargs)
        # positional signature: token, model_dim, inter_dim, expert, topk, dtype,
        #   q_dtype_a, q_dtype_w, q_type, use_g1u1, activation, ...
        q_dtype_a = args[6] if len(args) > 6 else kwargs.get("q_dtype_a")
        q_dtype_w = args[7] if len(args) > 7 else kwargs.get("q_dtype_w")
        _captured["q_dtype_a"] = q_dtype_a
        _captured["q_dtype_w"] = q_dtype_w
        _captured["backend"] = _backend_label(meta)
        _captured["fuse_quant"] = meta.fuse_quant
        _captured["ksplit"] = meta.ksplit
        return meta

    return wrapper


def build_a16w4(token, model_dim, inter_dim, E, topk, dtype=dtypes.bf16, seed=0):
    """Faithful a16w4 build (use_g1u1=True), mirroring test_fmoe lines 82-258."""
    torch.manual_seed(seed)
    qType = aiter.QuantType.per_1x32
    WQDType = dtypes.fp4x2
    torch_quant = aiter.get_torch_quant(qType)

    input = torch.randn((token, model_dim), dtype=dtype)
    w1 = torch.randn((E, inter_dim * 2, model_dim), dtype=dtype)
    exp_bias1 = torch.clamp(torch.randn((E, inter_dim * 2), dtype=dtype), -1.0, 1.0)
    w2 = torch.randn((E, model_dim, inter_dim), dtype=dtype)
    exp_bias2 = torch.clamp(torch.randn((E, model_dim), dtype=dtype), -1.0, 1.0)
    score = torch.randn((token, E), dtype=dtype)

    topk_weights, topk_ids = fused_topk(input, score, topk, True)

    w1_qt, w1_scale = torch_quant(w1, quant_dtype=WQDType)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=WQDType)
    # per_1x32, WQDType != i4x2 -> pack last dim by 2
    w1_qt = w1_qt.view(w1.shape[0], w1.shape[1], w1.shape[2] // 2)
    w2_qt = w2_qt.view(w2.shape[0], w2.shape[1], w2.shape[2] // 2)

    a1_qt = input.to(dtypes.bf16)
    a1_scale = None

    exp_bias1_aiter = exp_bias1.to(dtypes.fp32)
    exp_bias2_aiter = exp_bias2.to(dtypes.fp32)

    # preshuffle (a16w4 branch)
    w1_qt_aiter = shuffle_weight_a16w4(w1_qt, 16, True)
    w1_scale_aiter = shuffle_scale_a16w4(w1_scale, E, True)
    w2_qt_aiter = shuffle_weight_a16w4(w2_qt, 16, False)
    w2_scale_aiter = shuffle_scale_a16w4(w2_scale, E, False)

    return dict(
        input=input,
        dtype=dtype,
        token=token,
        model_dim=model_dim,
        inter_dim=inter_dim,
        E=E,
        topk=topk,
        qType=qType,
        WQDType=WQDType,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        # references use the unshuffled quantized weights + scales
        w1_qt=w1_qt,
        w2_qt=w2_qt,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_qt=a1_qt,
        a1_scale=a1_scale,
        exp_bias1=exp_bias1,
        exp_bias2=exp_bias2,
        # kernel-shuffled inputs for fused_moe
        w1_qt_aiter=w1_qt_aiter,
        w2_qt_aiter=w2_qt_aiter,
        w1_scale_aiter=w1_scale_aiter,
        w2_scale_aiter=w2_scale_aiter,
        exp_bias1_aiter=exp_bias1_aiter,
        exp_bias2_aiter=exp_bias2_aiter,
    )


def run_one(b, actType, gateMode):
    token = b["token"]
    model_dim = b["model_dim"]
    inter_dim = b["inter_dim"]
    E = b["E"]
    topk = b["topk"]
    dtype = b["dtype"]
    qType = b["qType"]

    _captured.clear()

    # Probe the real dispatch + decide stage1 ref dtype exactly like the harness.
    meta = get_2stage_cfgs(
        get_padded_M(token),
        model_dim,
        inter_dim,
        E,
        topk,
        dtype,
        dtypes.bf16,  # q_dtype_a on gfx942 a16w4 stays bf16 (probe; real value captured below)
        b["WQDType"],
        qType,
        True,  # use_g1u1
        actType,
        False,  # doweight_stage1
        0,
        0,
        True,  # is_shuffled
        gateMode.value,
    )
    stage1_ref_dtype = dtype
    if (
        actType == aiter.ActivationType.Swiglu
        and qType == aiter.QuantType.per_1x32
        and b["WQDType"] == dtypes.fp4x2
        and meta.fuse_quant == "fp4"
    ):
        stage1_ref_dtype = dtypes.fp32

    # ---- torch reference (same as harness) ----
    out1_ref = torch_moe_stage1(
        b["a1_qt"],
        b["w1_qt"],
        b["w2_qt"],
        b["topk_weights"],
        b["topk_ids"],
        dtype=stage1_ref_dtype,
        activation=actType,
        quant_type=qType,
        a1_scale=b["a1_scale"],
        w1_scale=b["w1_scale"],
        w1_bias=b["exp_bias1"],
        doweight=False,
    )
    a2_qt = out1_ref.view(token, topk, -1)
    out2_ref = torch_moe_stage2(
        a2_qt,
        b["w1_qt"],
        b["w2_qt"],
        b["topk_weights"],
        b["topk_ids"],
        dtype=dtype,
        quant_type=qType,
        w2_scale=b["w2_scale"],
        a2_scale=None,
        w2_bias=b["exp_bias2"],
        doweight=True,
    )

    # ---- timed public fused_moe (instrumented) ----
    err = None
    has_nan = None
    logits_diff = None
    us = None
    backend = None
    qa = None
    try:
        out2_ck, us = run_perftest(
            fused_moe,
            b["input"],
            b["w1_qt_aiter"],
            b["w2_qt_aiter"],
            b["topk_weights"],
            b["topk_ids"],
            w1_scale=b["w1_scale_aiter"],
            w2_scale=b["w2_scale_aiter"],
            quant_type=qType,
            activation=actType,
            doweight_stage1=False,
            bias1=b["exp_bias1_aiter"],
            bias2=b["exp_bias2_aiter"],
            gate_mode=gateMode.value,
            num_iters=50,
            num_warmup=10,
        )
        backend = _captured.get("backend")
        qa = _captured.get("q_dtype_a")
        has_nan = bool(out2_ck.isnan().any().item())
        err = checkAllclose(out2_ref, out2_ck, msg=f"a16w4 fused_moe {us:.2f}us")
        logits_diff = _calc_diff(out2_ref, out2_ck)
    except Exception as e:
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "backend": _captured.get("backend"),
            "q_dtype_a": _captured.get("q_dtype_a"),
        }

    strict_pass = (not has_nan) and not (err != 0 and logits_diff > 0.01)
    return {
        "ok": True,
        "us": us,
        "backend": backend,
        "q_dtype_a": qa,
        "fuse_quant": _captured.get("fuse_quant"),
        "ksplit": _captured.get("ksplit"),
        "err": err,
        "logits_diff": logits_diff,
        "has_nan": has_nan,
        "strict_pass": strict_pass,
    }


def main():
    gfx = get_gfx()
    print(f"gfx = {gfx}")
    if gfx != "gfx942":
        print("[STOP] not gfx942; the routing question is gfx942-specific.")
        return
    print(f"HIP_VISIBLE_DEVICES = {os.environ.get('HIP_VISIBLE_DEVICES')}")
    print(f"device = {torch.cuda.get_device_properties(0).gcnArchName}")

    # Instrument dispatch.
    fm.get_2stage_cfgs = _wrap_get_2stage_cfgs(fm.get_2stage_cfgs)

    model_dim, inter_dim, E, topk = 4096, 1536, 32, 1
    tokens = [1024, 4096, 8192]
    # a16w4 effective gate mode (test_moe_2stage._effective_gate_mode(bf16, fp4x2)).
    gate_interleave = GateMode.INTERLEAVE
    acts = [aiter.ActivationType.Swiglu, aiter.ActivationType.Silu]

    rows = []
    for tok in tokens:
        b = build_a16w4(tok, model_dim, inter_dim, E, topk)
        for act in acts:
            tag = f"tokens={tok} act={act.name} gate={gate_interleave.name}"
            print(f"\n===== {tag} =====")
            r = run_one(b, act, gate_interleave)
            r["tokens"] = tok
            r["act"] = act.name
            rows.append(r)
            if r["ok"]:
                print(
                    f"  backend: {r['backend']}\n"
                    f"  q_dtype_a={r['q_dtype_a']} fuse_quant={r['fuse_quant']} ksplit={r['ksplit']}\n"
                    f"  fused_moe e2e: {r['us']:.2f} us  | err={r['err']} "
                    f"logits_diff={r['logits_diff']:.2e} nan={r['has_nan']} "
                    f"-> {'PASS' if r['strict_pass'] else 'FAIL'}"
                )
            else:
                print(f"  ERROR: {r['error']}\n  backend(probe)={r['backend']} q_dtype_a={r['q_dtype_a']}")

    print("\n\n================ SUMMARY (CK-Tile via fused_moe, e2e) ================")
    print(f"{'tokens':>7} {'act':>7} {'backend(stage1/stage2)':>26} {'qa':>6} "
          f"{'us':>9} {'pass':>5}")
    for r in rows:
        if r["ok"]:
            be = r["backend"].split(",")[0].replace("stage1=", "")
            print(f"{r['tokens']:>7} {r['act']:>7} {be:>26} {str(r['q_dtype_a']):>6} "
                  f"{r['us']:>9.2f} {str(r['strict_pass']):>5}")
        else:
            print(f"{r['tokens']:>7} {r['act']:>7} {'ERROR: '+r['error'][:40]:>26}")


if __name__ == "__main__":
    main()
