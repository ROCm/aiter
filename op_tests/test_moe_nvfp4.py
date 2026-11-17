# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""NVFP4 MoE (aiter.ops.moe_nvfp4) vs a dequantized fp32 torch reference.

Covers moe_nvfp4 and the fused_moe path (w1/w2_global_scale) on the MiniMax-M3.1
shapes (TP4 / TP8, E 128, topk 4), the main path and the fallback (T 2048, E 640),
bf16 and f16 activations, the stage-2 top-k sum by atomics and by the reduce
kernel (_reduce), expert contributions that cancel beyond the f16 range, a dispatch
CSV (AITER_CONFIG_MOE_NVFP4) naming a kernel outside the prebuilt module
(per-kernel JIT), CSV rows whose kernels cannot run their shape, the fused_moe
errors, weights / inputs / raw GEMM entry arguments that are not one layer, and a
caller output buffer.

python3 op_tests/test_moe_nvfp4.py: these tests, then the moe_nvfp4 / fused_moe
benchmark sweep (main).
"""

import argparse
import itertools
import os
import subprocess
import sys

import pandas as pd
import pytest
import torch

import aiter
from aiter import ActivationType, NvFp4MoeWeights, dtypes, moe_nvfp4, uses_fallback
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
from aiter.ops.moe_nvfp4_common import reduce_twin
from aiter.ops.shuffle import shuffle_scale_nvfp4_moe, shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.utility.fp4_utils import mxfp4_to_f32


def _is_gfx950() -> bool:
    try:
        return torch.cuda.is_available() and get_gfx_runtime() == "gfx950"
    except (KeyError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _is_gfx950(), reason="NVFP4 MoE kernels need a gfx950 GPU"
)

H, E, TOPK = 6144, 128, 4
SHAPES = {"tp4": 768, "tp8": 384}
TOKENS = [1, 8, 64, 1024, 2048]
DTYPES = [torch.bfloat16, torch.float16]
# _check tolerance, 2x (bf16) / 5x (f16) the worst element of these shapes (0.019 /
# 0.0008 of the row RMS; bf16: output rounding and bf16 top-k atomics)
RTOL, ATOL = 1e-2, {torch.bfloat16: 4e-2, torch.float16: 4e-3}
# Shipped TP4 main-path stage-1 kernel without _xcdremap: not in the prebuilt module
JIT_S1 = "nvfp4_moe_s1_32x256_xdma_waves8_wbuf2x128_xbuf3_gs6"
JIT_S2 = (
    "nvfp4_moe_s2_32x256_xres_waves4_wbuf3x256_kdim768_gs6_upfront_occ3_staged_outbf16"
)


def _weights(I, seed=0, E=E, H=H):
    """Raw NVFP4 weights: packed E2M1 [E, N, K/2], E4M3 scale bits [E, N, K/16]
    (1.875 .. 448), fp32 1 / global scale [E, 2] and [E]."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def u8(lo, hi, *s):
        t = torch.randint(lo, hi, s, generator=g, device="cuda", dtype=torch.int32)
        return t.to(torch.uint8)

    w13, sc13 = u8(0, 256, E, 2 * I, H // 2), u8(0x3F, 0x7F, E, 2 * I, H // 16)
    w2, sc2 = u8(0, 256, E, H, I // 2), u8(0x3F, 0x7F, E, H, I // 16)
    g13 = (torch.rand(E, 2, generator=g, device="cuda") * 0.5 + 0.5) * 2e-4
    g2 = (torch.rand(E, generator=g, device="cuda") * 0.5 + 0.5) * 2e-4
    return w13, sc13, w2, sc2, g13, g2


def _fused_args(raw):
    """fused_moe weight arguments of the raw weights."""
    w13, sc13, w2, sc2, g13, g2 = raw
    f8 = torch.float8_e4m3fn
    return {
        "w1": shuffle_weight(w13.view(dtypes.fp4x2), (16, 32)),
        "w2": shuffle_weight(w2.view(dtypes.fp4x2), (16, 32)),
        "w1_scale": shuffle_scale_nvfp4_moe(sc13.view(f8)),
        "w2_scale": shuffle_scale_nvfp4_moe(sc2.view(f8)),
        "w1_global_scale": g13,
        "w2_global_scale": g2,
    }


@pytest.fixture(scope="module", params=list(SHAPES))
def shape(request):
    I = SHAPES[request.param]
    raw = _weights(I)
    return I, raw, NvFp4MoeWeights(*raw), _fused_args(raw)


def _dequant(w, sc, gm):
    """[N, K/2] packed, [N, K/16] scale bits, scalar multiplier -> fp32 [N, K]."""
    v = mxfp4_to_f32(w)
    N, K = v.shape
    s = sc.view(torch.float8_e4m3fn).float().view(N, K // 16, 1)
    return (v.view(N, K // 16, 16) * s).view(N, K) * gm


def _reference(x, raw, tid, tw):
    """fp32 MoE on the f16-rounded input and f16-rounded stage-1 output (the
    kernels' activation dtype), SwiGLU-OAI (alpha 1.702, limit 7)."""
    w13, sc13, w2, sc2, g13, g2 = raw
    I = w13.shape[1] // 2
    xf = x.half().float()
    out = torch.zeros(x.shape[0], w2.shape[1], device=x.device)
    ids, wts, topk = tid.view(-1), tw.view(-1).float(), tid.shape[1]
    for e in ids.unique().tolist():
        rows = (ids == e).nonzero().view(-1)
        w = torch.cat([_dequant(w13[e, :I], sc13[e, :I], g13[e, 0]),
                       _dequant(w13[e, I:], sc13[e, I:], g13[e, 1])])  # fmt: skip
        h = xf[rows // topk] @ w.t()
        gate, up = h[:, :I].clamp(max=7.0), h[:, I:].clamp(-7.0, 7.0)
        a2 = (gate * torch.sigmoid(1.702 * gate) * (up + 1)).half().float()
        y = a2 @ _dequant(w2[e], sc2[e], g2[e]).t()
        out.index_add_(0, rows // topk, y * wts[rows].view(-1, 1))
    return out


def _inputs(T, dtype, seed, E=E, topk=TOPK, H=H):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(T, H, generator=g, device="cuda").to(dtype)
    scores = torch.randn(T, E, generator=g, device="cuda").softmax(-1)
    tw, tid = torch.topk(scores, topk, dim=-1)
    return x, tid.to(torch.int32), tw.float()


def _check(out, ref, msg):
    """Asserts every element of out is within RTOL / ATOL of the reference, both
    divided by the reference row RMS: the rounding error of a row grows with its
    magnitude, and a single wrong element, row or expert fails (a global relative
    norm can hide them). Returns the mismatch share (0)."""
    rms = ref.pow(2).mean(-1, keepdim=True).sqrt()
    err = checkAllclose(
        ref / rms, out.float() / rms, rtol=RTOL, atol=ATOL[out.dtype], msg=msg
    )
    assert err == 0, f"{msg}: {err:.2%} of the elements off"
    return err


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("T", TOKENS)
def test_moe_nvfp4(shape, T, dtype):
    I, raw, W, fused = shape
    x, tid, tw = _inputs(T, dtype, seed=T)
    ref = _reference(x, raw, tid, tw)
    assert uses_fallback(T, TOPK, E) == (T == 2048)
    for name, out in (
        ("moe_nvfp4", moe_nvfp4(W, x, tid, tw)),
        ("fused_moe", fused_moe(x, topk_weight=tw, topk_ids=tid,
                                activation=ActivationType.Swiglu, **fused)),
    ):  # fmt: skip
        assert out.dtype == dtype and out.shape == (T, H), name
        assert torch.isfinite(out).all(), name
        _check(out, ref, f"{name} I={I} T={T} {dtype}")


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("T", [1, 16, 2048])
def test_out(shape, T, dtype):
    """Caller output (NaN-filled) on the main path (T 1, 16) and the fallback
    (T 2048): fully written (atomics targets zeroed); a bad out raises."""
    I, raw, W, _ = shape
    x, tid, tw = _inputs(T, dtype, seed=T + 1)
    buf = torch.full((T, H), float("nan"), dtype=dtype, device="cuda")
    assert moe_nvfp4(W, x, tid, tw, out=buf) is buf
    assert torch.isfinite(buf).all()
    _check(buf, _reference(x, raw, tid, tw), f"out I={I} T={T} {dtype}")
    strided = torch.empty(T, 2 * H, dtype=dtype, device="cuda")[:, ::2]
    for bad in (buf[:, :-1], strided, buf.float()):
        with pytest.raises(ValueError, match="out must be"):
            moe_nvfp4(W, x, tid, tw, out=bad)


@pytest.mark.parametrize("dtype", DTYPES)
def test_many_experts(dtype):
    """E 640 > MAX_SORT_EXPERTS: the fallback path at a small T."""
    E2, T = 640, 8
    raw = _weights(SHAPES["tp8"], E=E2)
    x, tid, tw = _inputs(T, dtype, seed=2, E=E2)
    assert uses_fallback(T, TOPK, E2)
    out = moe_nvfp4(NvFp4MoeWeights(*raw), x, tid, tw)
    assert out.dtype == dtype and torch.isfinite(out).all()
    _check(out, _reference(x, raw, tid, tw), f"E={E2} T={T} {dtype}")


@pytest.mark.parametrize("T", [1024, 2048])
def test_reduce_bf16(shape, T, monkeypatch):
    """bf16 output through the _reduce twin of the picked (atomics) stage-2 kernel:
    accuracy and a bitwise-repeatable result."""
    from aiter.ops import moe_nvfp4 as op

    I, raw, W, _ = shape
    pick = op._pick

    def reduce_pick(*args):
        kernel1, kernel2 = pick(*args)
        return kernel1, op._kernel(reduce_twin(kernel2["name"]))

    monkeypatch.setattr(op, "_pick", reduce_pick)
    kernel2 = op._pick(W, T, TOPK, "bf16", uses_fallback(T, TOPK, E))[1]
    assert kernel2["reduce"] and kernel2["md"] == op.MD_NAME, kernel2["name"]
    x, tid, tw = _inputs(T, torch.bfloat16, seed=T)
    out = moe_nvfp4(W, x, tid, tw)
    _check(out, _reference(x, raw, tid, tw), f"reduce I={I} T={T}")
    assert torch.equal(out, moe_nvfp4(W, x, tid, tw))


@pytest.mark.parametrize("T", [1, 16])
@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("dtype", DTYPES)
def test_cancellation(dtype, fallback, T):
    """Two experts with opposite down weights: each contribution (+-86016) is above
    the f16 range, their sum is 0 (f16 x: no inf - inf in the output atomics, T 1,
    or in the top-k reduce, T 16: default-rule _reduce)."""
    I, E2 = SHAPES["tp8"], 2

    def full(v, *s):
        return torch.full(s, v, dtype=torch.uint8, device="cuda")

    # fp4 0x2 = 1, 0xA = -1, E4M3 0x38 = 1: stage 1 gives the SwiGLU-OAI bound 56
    w2 = torch.cat([full(0x22, 1, H, I // 2), full(0xAA, 1, H, I // 2)])
    W = NvFp4MoeWeights(
        full(0x22, E2, 2 * I, H // 2), full(0x38, E2, 2 * I, H // 16),
        w2, full(0x38, E2, H, I // 16),
        torch.ones(E2, 2, device="cuda"), torch.full((E2,), 8.0, device="cuda"),
    )  # fmt: skip
    x = torch.ones(T, H, dtype=dtype, device="cuda")
    tid = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda").repeat(T, 1)
    tw = torch.full((T, 2), 0.5, device="cuda")
    out = moe_nvfp4(W, x, tid, tw, fallback=fallback)
    assert out.dtype == dtype and (out == 0).all(), out.float().abs().max()


def _check_jit_csv():
    """Run in a subprocess with AITER_CONFIG_MOE_NVFP4 set (read at import)."""
    from aiter.ops import moe_nvfp4 as op

    raw = _weights(SHAPES["tp4"])
    W = NvFp4MoeWeights(*raw)
    assert op._main_kernel(JIT_S1) is None
    kernel1, kernel2 = op._pick(W, 1000, TOPK, "bf16", False)
    assert kernel1["name"] == JIT_S1 and kernel1["md"] != op.MD_NAME, kernel1["name"]
    assert kernel2["name"] == JIT_S2, kernel2["name"]
    x, tid, tw = _inputs(1000, torch.bfloat16, seed=1)
    _check(moe_nvfp4(W, x, tid, tw), _reference(x, raw, tid, tw), "JIT CSV kernel")


def test_jit_csv(tmp_path):
    path = tmp_path / "tuned_moe_nvfp4.csv"
    path.write_text(
        "cu_num,token,model_dim,inter_dim,expert,topk,dtype,kernelName1,kernelName2\n"
        f"{get_cu_num()},1024,{H},768,{E},{TOPK},torch.bfloat16,{JIT_S1},{JIT_S2}\n"
    )
    env = dict(os.environ, AITER_CONFIG_MOE_NVFP4=str(path))
    r = subprocess.run([sys.executable, __file__, "--jit-csv"], env=env, check=False,
                       capture_output=True, text=True, timeout=1800)  # fmt: skip
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]


def test_bad_csv_row(monkeypatch):
    """Tuned CSV rows whose kernels cannot run their shape raise ValueError:
    another BM per stage, and a stage-2 kernel of the TP8 K."""
    from aiter.ops import moe_nvfp4 as op
    from aiter.ops.moe_nvfp4_dispatch import default_rows

    op.prepare()
    I = SHAPES["tp4"]
    rows = default_rows(H, I, "bf16", "bf16", False)
    tp8 = default_rows(H, SHAPES["tp8"], "bf16", "bf16", False)
    bad = [(rows[0][1][0], rows[1][2][0]), (rows[1][1][0], tp8[1][2][0])]
    for E2, (s1, s2) in enumerate(bad, start=1):
        key = (H, I, E2, TOPK, "bf16")  # E2: a key _tables has not cached
        monkeypatch.setitem(op._DISPATCH, key, ((8, s1, s2),))
        with pytest.raises(ValueError, match="token=8"):
            op._tables(*key)


# NvFp4MoeWeights inputs that are not one layer, from the raw (or shuffled) tensors
BAD_LAYERS = {
    "w13 2D": lambda r: {0: r[0][0]},
    "w13 odd N": lambda r: {0: r[0][:, 1:]},
    "w2 experts": lambda r: {2: r[2][1:]},
    "sc13 K": lambda r: {1: r[1][:, :, 16:]},
    "sc2 dtype": lambda r: {3: r[3].view(torch.int16)},
    "g13 one element": lambda r: {4: r[4][:1, :1]},
    "g13 [E]": lambda r: {4: r[4][:, 0]},
    "g2 one element": lambda r: {5: r[5][:1]},
    "g2 int": lambda r: {5: r[5].int()},
    "g13 cpu": lambda r: {4: r[4].cpu()},
    "g2 cpu": lambda r: {5: r[5].cpu()},
    "sc2 cpu": lambda r: {3: r[3].cpu()},
}


@pytest.mark.parametrize("shuffled", [False, True])
@pytest.mark.parametrize("case", list(BAD_LAYERS))
def test_bad_layer(case, shuffled):
    """NvFp4MoeWeights raises ValueError (no launch) unless its tensors are one layer
    on one device; a shuffled scale must have its shuffled size."""
    raw = list(_weights(SHAPES["tp8"], E=8))
    for i in (1, 3) if shuffled else ():
        raw[i] = shuffle_scale_nvfp4_moe(raw[i])
    NvFp4MoeWeights(*raw, shuffled=shuffled)
    for i, t in BAD_LAYERS[case](raw).items():
        raw[i] = t
    with pytest.raises(ValueError, match="must be"):
        NvFp4MoeWeights(*raw, shuffled=shuffled)


def test_bad_inputs():
    """moe_nvfp4 raises ValueError (no launch) for x, topk_ids or topk_w of another
    shape or off the weights device."""
    W = NvFp4MoeWeights(*_weights(SHAPES["tp8"], E=8))
    x, tid, tw = _inputs(8, torch.bfloat16, seed=0, E=8)
    for args in (
        (x.cpu(), tid, tw), (x, tid.cpu(), tw), (x, tid, tw.cpu()),
        (x[:4], tid, tw), (x, tid[:, :2], tw), (x, tid.view(-1), tw),
    ):  # fmt: skip
        with pytest.raises(ValueError, match="must be"):
            moe_nvfp4(W, *args)


def test_gemm_entry_errors():
    """The raw GEMM entry (_moe_nvfp4_gemm) rejects tensors that do not match its
    kernel and shape before the launch: wrong-stage weights or scales, inv_gs not
    [E, 2], x / out of other sizes, a CPU tensor."""
    from aiter.ops import moe_nvfp4 as op

    I, E2, T = SHAPES["tp8"], 8, 8
    W = NvFp4MoeWeights(*_weights(I, E=E2))
    kernel1 = op._pick(W, T, TOPK, "f16", False)[0]
    sort = tuple(torch.zeros(64, dtype=torch.int32, device="cuda") for _ in range(4))
    x = torch.zeros(T, H, dtype=torch.float16, device="cuda")
    out = torch.empty(T * TOPK, I, dtype=torch.float16, device="cuda")
    good = {"x": x, "w": W.w13, "w_sc": W.s13, "inv_gs": W.g13, "out": out}
    for match, bad in (
        ("w must be", {"w": W.w2}),
        ("w_sc must be", {"w_sc": W.s2}),
        ("inv_gs must be", {"inv_gs": W.g13[:1, :1]}),
        ("inv_gs must be", {"inv_gs": W.g2}),
        ("x or out size", {"x": x[:1]}),
        ("x or out size", {"out": out[:T]}),
        ("one GPU", {"inv_gs": W.g13.cpu()}),
    ):
        a = dict(good, **bad)
        with pytest.raises(RuntimeError, match=match):
            op._gemm(kernel1, sort, T, TOPK, E2, a["x"], a["w"], a["w_sc"],
                     a["inv_gs"], a["out"], K=H, ntiles=2 * I // 16, n_out=I)  # fmt: skip


# fused_moe arguments the NVFP4 path rejects (8 tokens), from its fused_moe inputs
ERRORS = {
    "expert_mask": lambda f: {"expert_mask": torch.ones(E, device="cuda").int()},
    "one global scale": lambda f: {"w2_global_scale": None},
    "global scale shape": lambda f: {"w1_global_scale": f["w1_global_scale"][:1]},
    "global scale cpu": lambda f: {"w2_global_scale": f["w2_global_scale"].cpu()},
    "fp8 w1": lambda f: {"w1": f["w1"].view(dtypes.fp8)},
    "unshuffled w1": lambda f: {"w1": f["w1"].view(torch.uint8).view(dtypes.fp4x2)},
    "scale layout": lambda f: {"w1_scale": f["w1_scale"].reshape(-1)[:1024]},
    "scale dtype": lambda f: {"w2_scale": f["w2_scale"].view(torch.int16)},
    "herd": lambda f: {"herd_topk": 3},
    "bias": lambda f: {"bias2": torch.zeros(E, H, device="cuda")},
    "doweight_stage1": lambda f: {"doweight_stage1": True},
    "a1_scale": lambda f: {"a1_scale": torch.ones(8, H // 32, device="cuda")},
    "num_local_tokens": lambda f: {
        "num_local_tokens": torch.full((1,), 8, dtype=torch.int32, device="cuda")
    },
    "padding": lambda f: {"hidden_pad": 128},
    "gate_mode": lambda f: {"gate_mode": "interleave"},
    "activation": lambda f: {"activation": ActivationType.Silu},
    "swiglu_limit": lambda f: {"swiglu_limit": 10.0},
    "dtype": lambda f: {"dtype": torch.float16},
}


@pytest.mark.parametrize("case", list(ERRORS))
def test_fused_moe_errors(shape, case):
    fused = shape[3]
    x, tid, tw = _inputs(8, torch.bfloat16, seed=0)
    kw = dict(fused, activation=ActivationType.Swiglu)
    kw.update(ERRORS[case](fused))
    w1, w2 = kw.pop("w1"), kw.pop("w2")
    with pytest.raises(ValueError):
        fused_moe(x, w1, w2, tw, tid, **kw)


@benchmark()
def test_moe_nvfp4_benchmark(dtype, token, model_dim, inter_dim, expert, topk):
    """moe_nvfp4 (the vLLM call) and fused_moe on random routing, each checked by
    _check. flops / bytes as test_moe_2stage: x, out and the NVFP4 weights and
    scales of at most token * topk experts."""
    raw = _weights(inter_dim, E=expert, H=model_dim)
    W, fused = NvFp4MoeWeights(*raw), _fused_args(raw)
    x, tid, tw = _inputs(token, dtype, seed=token, E=expert, topk=topk, H=model_dim)
    ref = _reference(x, raw, tid, tw)
    candidates = {
        "moe_nvfp4": lambda: moe_nvfp4(W, x, tid, tw),
        "fused_moe": lambda: fused_moe(x, topk_weight=tw, topk_ids=tid,
                                       activation=ActivationType.Swiglu, **fused),
    }  # fmt: skip
    flops = 2 * 3 * token * topk * model_dim * inter_dim
    w_vals = 3 * model_dim * inter_dim
    nbytes = 2 * x.nbytes + min(expert, token * topk) * (w_vals // 2 + w_vals // 16)
    ret = {"gfx": get_gfx_runtime()}
    for name, fn in candidates.items():
        out, us = run_perftest(fn)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = _check(out, ref, f"{name} T={token} I={inter_dim}")
    return ret


# Script-only benchmark (main); the tests above own the pytest coverage
test_moe_nvfp4_benchmark.__test__ = False


def main() -> int:
    if sys.argv[1:] == ["--jit-csv"]:
        _check_jit_csv()
        return 0
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="NVFP4 MoE tests, then the moe_nvfp4 / fused_moe benchmark",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=DTYPES,
        nargs="*",
        default=[torch.bfloat16],
        metavar="{bf16, fp16}",
        help="""Activation and output data type.
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-dim",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(H, I) for I in SHAPES.values()],
        help="""Model dimension and inter dimension (per TP rank).
    e.g.: -dim 6144,768""",
    )
    parser.add_argument(
        "-t",
        "--tokenNum",
        type=int,
        nargs="*",
        default=TOKENS,
        help="""Number of tokens.
    e.g.: -t 1024""",
    )
    parser.add_argument(
        "-e",
        "--expert",
        type=int,
        nargs="*",
        default=[E],
        help="""Number of experts.
    e.g.: -e 128""",
    )
    parser.add_argument(
        "-k",
        "--topk",
        type=int,
        nargs="*",
        default=[TOPK],
        help="""Number of top experts.
    e.g.: -k 4""",
    )
    args = parser.parse_args()

    result = pytest.main([__file__, "-q"])
    if result != pytest.ExitCode.OK:
        return int(result)
    if not _is_gfx950():
        aiter.logger.warning("moe_nvfp4 benchmark needs a gfx950 GPU; skipping sweep")
        return 0
    rows = [
        test_moe_nvfp4_benchmark(dtype, token, model_dim, inter_dim, expert, topk)
        for dtype, (model_dim, inter_dim), token, expert, topk in itertools.product(
            args.dtype, args.dim, args.tokenNum, args.expert, args.topk
        )
    ]
    aiter.logger.info(
        "moe_nvfp4 summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
