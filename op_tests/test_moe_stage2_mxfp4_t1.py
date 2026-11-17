# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP4 MoE stage 2 on the gfx950 kernels of hsa/gfx950/moe_stage2_mxfp4_t1
(AITER_MXFP4_MOE_STAGE2_T1=1).

The kernels serve one shape: 128 experts, top-4, model_dim 6144, inter_dim 1536
(MiniMax-M3.1-Flash at TP2), for a scatter kernelName2 at 4096 tokens or more. The GPU
test calls _mxfp4_a4w4_stage2_fw on the same sorted inputs with the switch off and on,
with uniform routing and with all tokens on the same 4 experts, and checks that the
kernels served the call, that the output matches AITER's and that its error against an
fp32 reference is no larger than AITER's. The CPU tests check the token-count range,
every reason a call is declined, and the kernel registry.

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
from aiter.test_common import run_perftest
from aiter.utility import fp4_utils

E, TOPK, MODEL_DIM, INTER_DIM, BLOCK_M = 128, 4, 6144, 1536, 128
_G2 = "flydsl_moe2_layout_afp4_wfp4_bf16_"
SCATTER = _G2 + "t128x256x256_scatter_sbm128"  # AITER's pick for 4097..32768 tokens
ATOMIC = _G2 + "t128x128x256_atomic_sbm128"  # AITER's pick for 2049..4096 tokens
# (tokens, kernel expected to serve, routing). "hot" sends every token to the same 4
# experts, so the dense kernel gets full groups of blocks of one expert.
CASES = [
    (4097, "t1g2_v2n_bm128", "uniform"),
    (4540, "t1g2_v2n_bm128", "hot"),
    (9567, "t1g2_v2n_bm128", "uniform"),
    (12288, "t1g2_v4g8_bm128", "uniform"),
    (16384, "t1g2_v4g8_bm128", "hot"),
]

_ON_GFX950 = torch.cuda.is_available() and get_gfx() == "gfx950"
_SKIP = pytest.mark.skipif(
    not _ON_GFX950, reason="the stage-2 kernels are gfx950 code objects"
)
_REAL_KERNEL = t1._kernel


def _dequant(q, scale, rows, cols):
    """fp32 values of MXFP4 data q [rows, cols / 2] with e8m0 scales [rows, cols / 32]."""
    v = fp4_utils.mxfp4_to_f32(q.view(torch.uint8)).view(rows, cols // 32, 32)
    s = fp4_utils.e8m0_to_f32(scale.view(torch.uint8)).view(rows, cols // 32, 1)
    return (v * s).view(rows, cols)


def _stage2_inputs(token, seed=0, routing="uniform"):
    """Sorted stage-2 inputs as AITER's a4w4 stage 1 leaves them, and an fp32
    reference of the weighted stage-2 output."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    scores = torch.randn((token, E), dtype=torch.float32, device="cuda")
    if routing == "hot":
        scores[:, :TOPK] += 100.0
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
        block_size=BLOCK_M,
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


def _stage2(d, token, kname, out=None):
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
        block_m=BLOCK_M,
        sorted_weights=d["sw"],
        kernelName2=kname,
        reverse_sorted=d["rev"],
    )


def _served_kernels(monkeypatch_or_none, on):
    """Turn the switch on or off; returns the list the served kernel names go to."""
    served = []

    def spy(arch, dense):
        cfg, path = _REAL_KERNEL(arch, dense)
        served.append(cfg["knl_name"])
        return cfg, path

    if monkeypatch_or_none is None:
        fm._MXFP4_MOE_STAGE2_T1 = on
        t1._kernel = spy
    else:
        monkeypatch_or_none.setattr(fm, "_MXFP4_MOE_STAGE2_T1", on)
        monkeypatch_or_none.setattr(t1, "_kernel", spy)
    return served


def _rel_l2(x, ref):
    x, ref = x.double(), ref.double()
    return ((x - ref).norm() / ref.norm()).item()


def _check(ref, aiter_out, out, label):
    """The error against the fp32 reference is at most 1.1x AITER's own, which
    bf16 rounding and accumulation order set. A wrong tile, even one CTA's,
    costs 10x or more."""
    assert not out.isnan().any().item(), f"{label}: output has NaN"
    err_aiter, err = _rel_l2(aiter_out, ref), _rel_l2(out, ref)
    same = (out == aiter_out).float().mean().item()
    print(
        f"{label}: rel_l2 vs fp32 reference: aiter {err_aiter:.2e}, kernel {err:.2e}; "
        f"{same:.4%} of the elements equal AITER's"
    )
    assert err <= 1.1 * err_aiter + 1e-4, f"{label}: {err:.2e} > aiter {err_aiter:.2e}"


@_SKIP
@pytest.mark.parametrize("token,kernel,routing", CASES)
def test_stage2_matches_aiter(token, kernel, routing, monkeypatch):
    d = _stage2_inputs(token, routing=routing)
    _served_kernels(monkeypatch, False)
    ref_aiter = _stage2(d, token, SCATTER).float()
    served = _served_kernels(monkeypatch, True)
    out = _stage2(d, token, SCATTER).float()
    torch.cuda.synchronize()
    assert served == [kernel], f"served {served}, expected {kernel}"
    label = f"{kernel} M{token} {routing}"
    _check(d["ref"], ref_aiter, out, label)
    assert _rel_l2(out, ref_aiter) < 1e-3, f"{label}: differs from AITER's output"


@_SKIP
@pytest.mark.parametrize(
    "token,kname,exact",
    [(2048, SCATTER, True), (4095, SCATTER, True), (4540, ATOMIC, False)],
    ids=["below the range", "just below the range", "atomic kernelName2"],
)
def test_declined_call_keeps_aiter_output(token, kname, exact, monkeypatch):
    d = _stage2_inputs(token)
    _served_kernels(monkeypatch, False)
    ref_aiter = _stage2(d, token, kname).float()
    served = _served_kernels(monkeypatch, True)
    out = _stage2(d, token, kname).float()
    torch.cuda.synchronize()
    assert served == [], f"M{token} served by {served}"
    if exact:  # AITER's scatter path is deterministic
        assert torch.equal(out, ref_aiter)
    else:  # its atomic kernels add in a different order on each run
        _check(d["ref"], ref_aiter, out, f"M{token} {kname[len(_G2) :]}")


@pytest.mark.parametrize(
    "M,block_m,atomic,want",
    [
        (4095, 128, False, None),
        (4096, 128, False, 0),
        (12287, 128, False, 0),
        (12288, 128, False, 1),
        (32768, 128, False, 1),
        (4096, 128, True, None),
        (16384, 128, True, None),
        (16384, 64, False, None),
        (16384, 256, False, None),
    ],
)
def test_pick_kernel(M, block_m, atomic, want):
    assert t1.pick_kernel(M, block_m, atomic) == want


def _m(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="meta")


def _call(M=4096, **override):
    """The arguments of a call the kernels serve, on meta tensors, with the
    overrides applied."""
    n = M * TOPK + E * (BLOCK_M - 1)
    kw = {
        "inter_states": _m((n, INTER_DIM // 2), torch.uint8),
        "w2": _m((E, MODEL_DIM, INTER_DIM // 2), torch.uint8),
        "sorted_token_ids": _m((n,), torch.int32),
        "sorted_expert_ids": _m((n // BLOCK_M,), torch.int32),
        "num_valid_ids": _m((2,), torch.int32),
        "moe_out": _m((M, MODEL_DIM), torch.bfloat16),
        "topk": TOPK,
        "w2_scale": _m((E * MODEL_DIM, INTER_DIM // 32), torch.uint8),
        "a2_scale": _m((n, INTER_DIM // 32), torch.uint8),
        "block_m": BLOCK_M,
        "sorted_weights": _m((n,), torch.float32),
        "topk_weights": None,
        "bias2": None,
        "reverse_sorted": _m((n,), torch.int32),
        "use_valid_token_count": False,
        "atomic": False,
    }
    kw.update(override)
    return kw


@pytest.mark.parametrize("kw,want", [({}, 0), ({"M": 16384}, 1)], ids=["flat", "dense"])
def test_eligible_calls(kw, want):
    assert t1._eligible(**_call(**kw)) == want


_DECLINED = {
    "M below the range": {"M": 4095},
    "atomic": {"atomic": True},
    "no block_m": {"block_m": None},
    "block_m 64": {"block_m": 64},
    "block_m 256": {"block_m": 256},
    "bias2": {"bias2": _m((E, MODEL_DIM), torch.float32)},
    "topk_weights": {"topk_weights": _m((4096, TOPK), torch.float32)},
    "no sorted_weights": {"sorted_weights": None},
    "no reverse_sorted": {"reverse_sorted": None},
    "no a2_scale": {"a2_scale": None},
    "no w2_scale": {"w2_scale": None},
    "valid token count": {"use_valid_token_count": True},
    "top-5": {"topk": 5},
    "inter_states fp8": {
        "inter_states": _m((9000, INTER_DIM // 2), torch.float8_e4m3fn)
    },
    "inter_states width 1536": {"inter_states": _m((9000, INTER_DIM), torch.uint8)},
    "inter_states 3-d": {"inter_states": _m((9000, 2, INTER_DIM // 4), torch.uint8)},
    "129 experts": {"w2": _m((E + 1, MODEL_DIM, INTER_DIM // 2), torch.uint8)},
    "w2 fp8": {"w2": _m((E, MODEL_DIM, INTER_DIM // 2), torch.float8_e4m3fn)},
    "a2_scale fp32": {"a2_scale": _m((9000, INTER_DIM // 32), torch.float32)},
    "w2_scale fp32": {"w2_scale": _m((E * MODEL_DIM, INTER_DIM // 32), torch.float32)},
    "moe_out fp16": {"moe_out": _m((4096, MODEL_DIM), torch.float16)},
    "moe_out width 3072": {"moe_out": _m((4096, MODEL_DIM // 2), torch.bfloat16)},
    "moe_out strided": {"moe_out": _m((4096, 2 * MODEL_DIM), torch.bfloat16)[:, ::2]},
    "sorted_weights fp16": {"sorted_weights": _m((9000,), torch.float16)},
    "sorted_token_ids int64": {"sorted_token_ids": _m((9000,), torch.int64)},
    "sorted_expert_ids int64": {"sorted_expert_ids": _m((70,), torch.int64)},
    "num_valid_ids int64": {"num_valid_ids": _m((2,), torch.int64)},
}


@pytest.mark.parametrize("override", _DECLINED.values(), ids=_DECLINED.keys())
def test_eligible_declines(override):
    assert t1._eligible(**_call(**override)) is None


def test_eligible_declines_a_shrunk_inter_dim():
    w2 = _m((E, MODEL_DIM, INTER_DIM // 2), torch.uint8)
    w2.inter_real = 1024
    assert t1._eligible(**_call(w2=w2)) is None


def test_launcher_returns_false_off_cuda():
    """An eligible call on non-CUDA tensors writes nothing and returns False."""
    assert t1.moe_stage2_mxfp4_t1(**_call()) is False


def test_check_installed(monkeypatch, tmp_path):
    t1.check_installed()
    monkeypatch.setattr(t1, "AITER_ASM_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="moe_stage2_mxfp4_t1"):
        t1.check_installed()


def test_registry_matches_launcher():
    d = os.path.join(AITER_ASM_DIR, "gfx950", t1._SUBDIR)
    rows = load_asm_cfg_csv(os.path.join(d, t1._CSV))
    assert {(r["BM"], r["atomic"], r["dense"]) for r in rows} == {
        (BLOCK_M, 0, 0),
        (BLOCK_M, 0, 1),
    }
    for M in (t1._MIN_M, t1._DENSE_MIN_M - 1, t1._DENSE_MIN_M, 32768):
        dense = t1.pick_kernel(M, BLOCK_M, False)
        assert any(r["dense"] == dense for r in rows), M
    for r in rows:
        assert r["kernarg_size"] == ctypes.sizeof(t1._KernelArgs), r["knl_name"]
        assert (r["NE"], r["TOPK"], r["D_HIDDEN"], r["D_INTER"]) == (
            E,
            TOPK,
            MODEL_DIM,
            INTER_DIM,
        )
        assert os.path.isfile(os.path.join(d, r["co_name"])), r["co_name"]
    files = {f for f in os.listdir(d) if f.endswith(".co")}
    assert files == {r["co_name"] for r in rows}, "a .co the registry does not list"


def _bench():
    print("| tokens | routing | AITER us | kernel | us | speedup |")
    print("|---:|---|---:|---|---:|---:|")
    for token, kernel, routing in CASES:
        d = _stage2_inputs(token, routing=routing)
        del d["ref"]
        out = torch.zeros((token, MODEL_DIM), dtype=torch.bfloat16, device="cuda")
        us = []
        for on in (False, True):
            _served_kernels(None, on)
            _, t = run_perftest(_stage2, d, token, SCATTER, out)
            us.append(t)
        print(
            f"| {token} | {routing} | {us[0]:.1f} | {kernel} | {us[1]:.1f} "
            f"| {us[0] / us[1]:.2f}x |"
        )


if __name__ == "__main__":
    rc = pytest.main([__file__, "-q"])
    if rc == 0 and _ON_GFX950:
        _bench()
    raise SystemExit(rc)
