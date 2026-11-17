# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 MoE stage 2 on the gfx950 kernels of hsa/gfx950/moe_stage2_mxfp4_t1
(AITER_MXFP4_MOE_STAGE2_T1=1).

The kernels serve one shape: 128 experts, top-4, model_dim 6144, inter_dim 1536
(MiniMax-M3.1-Flash at TP2). The GPU test calls _mxfp4_a4w4_stage2_fw on the same
sorted inputs with the switch off and on, at a token count in each served range,
and checks that the kernels served the call and that their output agrees with
an fp32 reference and with AITER's stage 2 for the same kernelName2. The CPU
tests check the token-count ranges and the kernel registry.

Run (the checks, then a latency table of both paths):
    python3 op_tests/test_moe_stage2_mxfp4_t1.py
    pytest op_tests/test_moe_stage2_mxfp4_t1.py -q
"""

import ctypes
import os

import pytest
import torch

import aiter.fused_moe as fm
import aiter.ops.asm.moe_stage2_mxfp4_t1 as t1
from aiter import dtypes
from aiter.fused_moe import moe_sorting
from aiter.jit.core import AITER_ASM_DIR
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.asm.asm_utils import load_asm_cfg_csv
from aiter.ops.quant import mxfp4_moe_sort_fwd, per_1x32_f4_quant
from aiter.ops.shuffle import shuffle_scale, shuffle_weight
from aiter.test_common import checkAllclose, run_perftest
from aiter.utility import fp4_utils

E, TOPK, MODEL_DIM, INTER_DIM = 128, 4, 6144, 1536
_G2 = "flydsl_moe2_layout_afp4_wfp4_bf16_"
# (tokens, block_m, kernelName2, kernel expected to serve). The kernel names are
# AITER's picks for this shape in the tuned config the measurements used.
CASES = [
    (1024, 64, _G2 + "t64x256x256_atomic_nt_sbm64", "t1g2_v2ao_bm64"),
    (2048, 128, _G2 + "t128x256x256_atomic_nt_sbm128", "t1g2_v2ao_bm128"),
    (2048, 128, _G2 + "t128x256x256_scatter_sbm128", "t1g2_v2n_bm128"),
    (4096, 128, _G2 + "t128x128x256_atomic_sbm128", "t1g2_v2n_bm128"),
    (16384, 128, _G2 + "t128x256x256_scatter_sbm128", "t1g2_v4g8_bm128"),
]

_ON_GFX950 = torch.cuda.is_available() and get_gfx() == "gfx950"
_SKIP = pytest.mark.skipif(
    not _ON_GFX950, reason="the stage-2 kernels are gfx950 code objects"
)


def _dequant(q, scale, rows, cols):
    """fp32 values of MXFP4 data q [rows, cols / 2] with e8m0 scales [rows, cols / 32]."""
    v = fp4_utils.mxfp4_to_f32(q.view(torch.uint8)).view(rows, cols // 32, 32)
    s = fp4_utils.e8m0_to_f32(scale.view(torch.uint8)).view(rows, cols // 32, 1)
    return (v * s).view(rows, cols)


def _stage2_inputs(token, block_m, seed=0):
    """Sorted stage-2 inputs as AITER's a4w4 stage 1 leaves them, and an fp32
    reference of the weighted stage-2 output."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    scores = torch.randn((token, E), dtype=torch.float32, device="cuda")
    topk_ids = scores.topk(TOPK, dim=1).indices.to(torch.int32)
    topk_weights = torch.softmax(
        torch.randn((token, TOPK), dtype=torch.float32, device="cuda"), dim=1
    )
    sti, sw, sei, nvi, _, _, rev = moe_sorting(
        topk_ids,
        topk_weights,
        E,
        MODEL_DIM,
        torch.bfloat16,
        block_size=block_m,
        accumulate=False,
        output_aux="opus",
    )
    a2 = torch.randn((token * TOPK, INTER_DIM), dtype=torch.bfloat16, device="cuda")
    a2_q, a2_s = per_1x32_f4_quant(a2, quant_dtype=dtypes.fp4x2)
    a2_q = a2_q.view(torch.uint8).view(token * TOPK, INTER_DIM // 2)
    max_sorted = sti.shape[0]
    pos = torch.arange(max_sorted, device="cuda")
    tok = (sti & 0xFFFFFF).long()
    row = (tok * TOPK + (sti >> 24).long()).clamp(max=token * TOPK - 1)
    valid = (pos < nvi[0]) & (tok < token)
    inter_q = torch.zeros(
        (max_sorted, INTER_DIM // 2), dtype=torch.uint8, device="cuda"
    )
    inter_q[valid] = a2_q[row[valid]]
    inter_s = mxfp4_moe_sort_fwd(
        a2_s.view(token * TOPK, INTER_DIM // 32),
        sorted_ids=sti,
        num_valid_ids=nvi,
        token_num=token,
        cols=INTER_DIM,
    )
    w2 = torch.randn((E, MODEL_DIM, INTER_DIM), dtype=torch.bfloat16, device="cuda")
    w2_q, w2_scale = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w2_q = w2_q.view(E, MODEL_DIM, INTER_DIM // 2)
    w2_s = w2_scale.view(E * MODEL_DIM, INTER_DIM // 32)
    a_f = _dequant(a2_q, a2_s, token * TOPK, INTER_DIM).view(token, TOPK, INTER_DIM)
    ref = torch.zeros((token, MODEL_DIM), dtype=torch.float32, device="cuda")
    for e in range(E):
        t, k = (topk_ids == e).nonzero(as_tuple=True)
        if t.numel():
            s = w2_s[e * MODEL_DIM : (e + 1) * MODEL_DIM]
            w_f = _dequant(w2_q[e], s, MODEL_DIM, INTER_DIM)
            ref.index_add_(0, t, (a_f[t, k] @ w_f.T) * topk_weights[t, k][:, None])
    return {
        "sti": sti,
        "sw": sw,
        "sei": sei,
        "nvi": nvi,
        "rev": rev,
        "inter_q": inter_q,
        "inter_s": inter_s,
        "w1": torch.empty((E, 2 * INTER_DIM, 1), dtype=torch.uint8, device="cuda"),
        "w2": shuffle_weight(w2_q, (16, 16), is_guinterleave=False, gate_up=False),
        "w2_scale": shuffle_scale(w2_scale, E, is_guinterleave=False, gate_up=False),
        "ref": ref,
    }


def _stage2(d, token, block_m, kname, out=None):
    """One _mxfp4_a4w4_stage2_fw call; moe_out zeroed, as the sort leaves it for
    an atomic kernelName2."""
    if out is None:
        out = torch.zeros((token, MODEL_DIM), dtype=torch.bfloat16, device="cuda")
    return fm._mxfp4_a4w4_stage2_fw(
        d["inter_q"],
        d["w1"],
        d["w2"],
        d["sti"],
        d["sei"],
        d["nvi"],
        out,
        TOPK,
        w2_scale=d["w2_scale"],
        a2_scale=d["inter_s"],
        block_m=block_m,
        sorted_weights=d["sw"],
        kernelName2=kname,
        reverse_sorted=d["rev"],
    )


def _served_kernels(monkeypatch_or_none, on):
    """Turn the switch on or off; returns the list the served kernel names go to."""
    served = []
    real = t1._kernel

    def spy(arch, block_m, atomic, dense):
        cfg, path = real(arch, block_m, atomic, dense)
        served.append(cfg["knl_name"])
        return cfg, path

    if monkeypatch_or_none is None:
        fm._MXFP4_MOE_STAGE2_T1 = on
        t1._kernel = spy
    else:
        monkeypatch_or_none.setattr(fm, "_MXFP4_MOE_STAGE2_T1", on)
        monkeypatch_or_none.setattr(t1, "_kernel", spy)
    return served


def _check(ref, out, label):
    assert not out.isnan().any().item(), f"{label}: output has NaN"
    err = checkAllclose(ref, out, rtol=0.05, atol=1.0, tol_err_ratio=0.05, msg=label)
    assert err <= 0.05, f"{label}: {err:.2%} of the elements outside tolerance"


@_SKIP
@pytest.mark.parametrize("token,block_m,kname,kernel", CASES)
def test_stage2_matches_aiter(token, block_m, kname, kernel, monkeypatch):
    d = _stage2_inputs(token, block_m)
    _served_kernels(monkeypatch, False)
    ref_aiter = _stage2(d, token, block_m, kname).float()
    served = _served_kernels(monkeypatch, True)
    out = _stage2(d, token, block_m, kname).float()
    torch.cuda.synchronize()
    assert served == [kernel], f"served {served}, expected {kernel}"
    _check(d["ref"], ref_aiter, f"aiter {kname} M{token}: ")
    _check(d["ref"], out, f"{kernel} M{token}: ")
    _check(ref_aiter, out, f"{kernel} vs aiter M{token}: ")


@_SKIP
def test_decline_below_the_served_range(monkeypatch):
    token, block_m, kname = 512, 64, CASES[0][2]
    d = _stage2_inputs(token, block_m)
    served = _served_kernels(monkeypatch, True)
    _stage2(d, token, block_m, kname)
    torch.cuda.synchronize()
    assert served == [], f"M{token} served by {served}"


@pytest.mark.parametrize(
    "M,block_m,atomic,want",
    [
        (1023, 64, True, None),
        (1024, 64, True, (1, 0)),
        (1024, 32, True, None),
        (1024, 128, False, None),
        (1535, 128, False, None),
        (1536, 128, False, (0, 0)),
        (2285, 128, True, (1, 0)),
        (2286, 128, True, (0, 0)),
        (12287, 128, False, (0, 0)),
        (12288, 128, False, (0, 1)),
        (12288, 64, True, (0, 1)),
        (12288, 32, False, (0, 0)),
    ],
)
def test_pick_kernel(M, block_m, atomic, want):
    assert t1.pick_kernel(M, block_m, atomic) == want


def test_registry_matches_launcher():
    d = os.path.join(AITER_ASM_DIR, "gfx950", t1._SUBDIR)
    rows = load_asm_cfg_csv(os.path.join(d, t1._CSV))
    keys = {(r["BM"], r["atomic"], r["dense"]) for r in rows}
    for bm in t1._BMS:
        for M in (1024, 1536, 2285, 12288):
            for atomic in (False, True):
                pick = t1.pick_kernel(M, bm, atomic)
                assert pick is None or (bm, *pick) in keys, (M, bm, atomic)
    for r in rows:
        args = t1._AtomicKernelArgs if r["atomic"] else t1._KernelArgs
        assert r["kernarg_size"] == ctypes.sizeof(args), r["knl_name"]
        assert (r["NE"], r["TOPK"], r["D_HIDDEN"], r["D_INTER"]) == (
            E,
            TOPK,
            MODEL_DIM,
            INTER_DIM,
        )
        assert os.path.isfile(os.path.join(d, r["co_name"])), r["co_name"]


def _bench():
    print("| tokens | block_m | kernelName2 | AITER us | kernel | us | speedup |")
    print("|---:|---:|---|---:|---|---:|---:|")
    for token, block_m, kname, kernel in CASES:
        d = _stage2_inputs(token, block_m)
        del d["ref"]
        out = torch.zeros((token, MODEL_DIM), dtype=torch.bfloat16, device="cuda")
        us = []
        for on in (False, True):
            _served_kernels(None, on)
            _, t = run_perftest(_stage2, d, token, block_m, kname, out)
            us.append(t)
        print(
            f"| {token} | {block_m} | {kname} | {us[0]:.1f} | {kernel} | {us[1]:.1f} "
            f"| {us[0] / us[1]:.2f}x |"
        )


if __name__ == "__main__":
    rc = pytest.main([__file__, "-q"])
    if rc == 0 and _ON_GFX950:
        _bench()
    raise SystemExit(rc)
