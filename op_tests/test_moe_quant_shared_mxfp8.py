# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MXFP8 copy of the MoE source rows, written by the routed A1 quant.

A shared expert run beside a routed MXFP4 MoE reads the same bf16 rows as
MXFP8. ``flydsl_moe_fused_quant_preshuffle(mxfp8_copy_out=...)`` writes that
copy from the routed quant's own pass. For every quant variant the M3.1 MoE
dispatches (route-indexed, token-multidest plain and K-split, the compact-scale
build) it checks

* the routed payload, scale and route outputs are bitwise those of the same
  launch without the copy, padding included;
* the copy is bitwise per_1x32_mx_quant_hip(fp8, e8m0), the separate launch
  aiter falls back to (``mxfp8_row_copy``), and a torch model of the exponent
  rule (ceil(amax / 448), amax floored at 1e-10);

and times the fused launch against the routed quant and the split quant.
"""

import argparse
import importlib.util
import itertools
from pathlib import Path

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, run_perftest

pytest.importorskip("flydsl")
from aiter.ops.flydsl import grouped_moe_gfx1250 as gm
from aiter.ops.flydsl import moe_kernels as mk
from aiter.ops.quant import per_1x32_mx_quant_hip

SUPPORTED_GFX = ["gfx1250"]
# MiniMax-M3.1: hidden 6144, 128 routed experts, top-4.
MODEL_DIM, EXPERTS, TOPK = 6144, 128, 4
INTER_DIM = 768  # TP4 slice of the routed experts' 3072
# What the M3.1 MoE hands each quant: decode (r1, tile 16) and prefill (r8).
DECODE_TOKENS = [8, 32, 192, 512]
PREFILL_TOKENS = [4096, 16384, 32768]
# How the MoE reaches each variant; ``env`` is AITER_FLYDSL_TOKEN_MULTIDEST_QUANT.
VARIANTS = {
    # Token-multidest on a ready route map (K-split at decode sizes).
    "multidest": {"env": "1", "min_tokens": 64},
    # Compact-scale quant (first launch of the pair).
    "compact": {"compact": True, "env": "1", "min_tokens": 64},
    # Route-indexed on a ready route map (K-split or full row by grid size).
    "routeks": {"env": "0"},
    # Route-indexed with an EP valid-route count (all routes valid here): the
    # copy is not folded and goes out in its own launch (mxfp8_row_copy).
    "routeks_sep": {"env": "0", "valid_routes": True, "max_tokens": 192},
}


def mxfp8_reference(x):
    """The shared expert's split quant, in torch: E4M3 values with an E8M0
    exponent per 32 -- ceil_pow2(max(amax, 1e-10) * f32(1/448)) -- as ATOM's
    ``quantize_fp8`` and aiter's per_1x32_mx_quant_hip compute it."""
    g = x.float().reshape(x.shape[0], -1, 32)
    amax = (
        g.abs()
        .amax(-1)
        .clamp_min(torch.tensor(1e-10, dtype=torch.float32, device=x.device))
    )
    bits = (
        amax * torch.tensor(1.0 / 448.0, dtype=torch.float32, device=x.device)
    ).view(torch.int32)
    code = (bits >> 23) + ((bits & 0x7FFFFF) != 0).int()
    recip = ((254 - code).clamp_min(0) << 23).view(torch.float32)
    q = (g * recip[..., None]).clamp(-448.0, 448.0).to(dtypes.fp8)
    return q.reshape(x.shape).view(torch.uint8), code.to(torch.uint8)


def _hidden(tokens, seed):
    """bf16 rows with the edges a block scale meets: zero, tiny and wide
    blocks, exact powers of two and 448 * 2^k maxima."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(tokens, MODEL_DIM, generator=g, device="cuda")
    blocks = x.view(tokens, -1, 32)
    blocks[:, 0] = 0.0
    blocks[:, 1] *= 1e-12
    blocks[:, 2] *= 1e4
    blocks[:, 3, 0] = 448.0
    blocks[:, 4, 0] = -896.0
    blocks[:, 5] = torch.ldexp(torch.ones_like(blocks[:, 5]), blocks[:, 5].int())
    return x.bfloat16()


class _Case:
    """One quant launch as the MoE makes it, with copy on or off."""

    def __init__(self, variant, tokens, seed=0):
        cfg = VARIANTS[variant]
        self.cfg = cfg
        self.tokens = tokens
        self.rep = 1 if tokens <= 512 else 8
        tile = 16 * self.rep
        numel = tokens * TOPK
        self.cm = gm._tdm_align_up(numel + EXPERTS * tile - TOPK, tile)
        self.hidden = _hidden(tokens, seed)
        ws = MODEL_DIM // 32
        # A ready route map: any one-to-one placement of the routes. Token 1's
        # first route and all of token 2's are dropped (row -1, as HERD and
        # padded rows leave them); the copy still covers both tokens.
        g = torch.Generator(device="cuda").manual_seed(seed + 2)
        perm = torch.randperm(self.cm, generator=g, device="cuda")[:numel]
        self.rows = perm.to(torch.int32).reshape(tokens, TOPK)
        self.rows[1, 0] = -1
        self.rows[2] = -1
        compact = cfg.get("compact", False)
        self.out = [
            (
                torch.full(
                    (self.cm, MODEL_DIM // 2), 173, dtype=torch.uint8, device="cuda"
                ),
                torch.full(
                    (
                        (tokens, ws)
                        if compact
                        else (1, self.cm // self.rep, ws * self.rep)
                    ),
                    197,
                    dtype=torch.uint8,
                    device="cuda",
                ),
                self.rows.clone(),
                torch.full((self.cm,), -1, dtype=torch.int32, device="cuda"),
            )
            for _ in range(2)
        ]
        self.copy = (
            torch.full((tokens, MODEL_DIM), 211, dtype=torch.uint8, device="cuda"),
            torch.full((tokens, ws), 223, dtype=torch.uint8, device="cuda"),
        )

    def launch(self, slot, copy):
        payload, scale, rows, r2t = self.out[slot]
        kw = {
            "wmma_rep": self.rep,
            "quant_mode": "fp4",
            "source_topk": TOPK,
            "out_payload": payload.view(1, self.cm, -1),
            "out_scale": scale,
        }
        if copy:
            kw["mxfp8_copy_out"] = self.copy
        kw["topids_to_rows"] = rows
        if self.cfg.get("compact"):
            kw["row_to_token"] = r2t
        if self.cfg.get("valid_routes"):
            kw["num_valid_routes"] = torch.tensor(
                [rows.numel()], dtype=torch.int32, device="cuda"
            )
        mk.flydsl_moe_fused_quant_preshuffle(
            self.hidden.view(1, self.tokens, MODEL_DIM), 1, self.cm, **kw
        )


def _kernel_names(fn):
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted(
        {
            e.name
            for e in prof.events()
            if e.device_type.name == "CUDA" and not e.name.startswith("Memcpy")
        }
    )


@benchmark()
def moe_quant_mxfp8_copy_case(variant, tokens):
    cfg = VARIANTS[variant]
    case = _Case(variant, tokens)
    case.launch(0, copy=False)
    case.launch(1, copy=True)
    torch.cuda.synchronize()
    for a, b in zip(*case.out):
        assert torch.equal(a, b), f"{variant} T={tokens}: routed output changed"
    ref_q, ref_s = per_1x32_mx_quant_hip(
        case.hidden, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    model_q, model_s = mxfp8_reference(case.hidden)
    got_q, got_s = case.copy
    assert torch.equal(got_q, ref_q.view(torch.uint8)), f"{variant}: payload"
    assert torch.equal(got_s, ref_s.view(torch.uint8)), f"{variant}: scale"
    assert torch.equal(got_q, model_q) and torch.equal(got_s, model_s), variant
    # The separate launch aiter falls back to where a variant takes no copy.
    sep = tuple(torch.empty_like(t) for t in case.copy)
    mk.mxfp8_row_copy(case.hidden, sep)
    assert torch.equal(sep[0], got_q) and torch.equal(sep[1], got_s), variant
    fused_names = _kernel_names(lambda: case.launch(1, copy=True))
    folded = [k for k in fused_names if k.endswith("_mx8cp")]
    if cfg.get("valid_routes"):  # the copy is its own launch
        assert len(fused_names) == 2 and not folded, fused_names
    else:  # one launch, the copy build of the expected family
        family = "fusepre" if cfg.get("compact") else variant.split("_")[0]
        assert len(fused_names) == 1 and family in folded[0], fused_names
    _, us_routed = run_perftest(lambda: case.launch(0, copy=False))
    _, us_fused = run_perftest(lambda: case.launch(1, copy=True))
    _, us_split_quant = run_perftest(
        lambda: per_1x32_mx_quant_hip(
            case.hidden, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
        )
    )
    # bf16 read once, FP4 payload + scale per route, FP8 copy + scale.
    nbytes = tokens * MODEL_DIM * (2 + TOPK * (0.5 + 1 / 32) + 1 + 1 / 32)
    return {
        "gfx": get_gfx_runtime(),
        "kernel": fused_names[0] if len(fused_names) == 1 else ",".join(fused_names),
        "launches": len(fused_names),
        "routed us": us_routed,
        "split quant us": us_split_quant,
        "fused us": us_fused,
        "fused TB/s": nbytes / us_fused / 1e6,
        "bitwise": True,
        "env": cfg["env"],
    }


def _cases():
    for variant, tokens in itertools.product(VARIANTS, DECODE_TOKENS + PREFILL_TOKENS):
        cfg = VARIANTS[variant]
        if not cfg.get("min_tokens", 0) <= tokens <= cfg.get("max_tokens", tokens):
            continue
        yield variant, tokens


def _run_case(variant, tokens):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("AITER_FLYDSL_TOKEN_MULTIDEST_QUANT", VARIANTS[variant]["env"])
        return moe_quant_mxfp8_copy_case(variant, tokens)


@pytest.mark.parametrize("variant,tokens", list(_cases()))
def test_mxfp8_copy_bitwise(variant, tokens):
    if not torch.cuda.is_available() or get_gfx_runtime() not in SUPPORTED_GFX:
        pytest.skip("gfx1250 pk8 quant")
    _run_case(variant, tokens)


def test_host_copy_out_is_checked():
    x = torch.zeros(4, 64, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        mk._check_mxfp8_copy_out(
            x, (torch.empty(4, 64, dtype=torch.uint8), torch.empty(4, 3))
        )
    mk._check_mxfp8_copy_out(
        x, (torch.empty(4, 64, dtype=torch.uint8), torch.empty(4, 2, dtype=torch.uint8))
    )


def _fused_moe_args(tokens):
    """The arguments flydsl_tests/test_flydsl_grouped_gemm.py hands fused_moe
    for an M3.1 TP4 a4w4 SwiGLU MoE (preshuffled MXFP4 weights), captured
    rather than rebuilt so the layout cannot drift from the grouped path's."""
    path = Path(__file__).parent / "flydsl_tests/test_flydsl_grouped_gemm.py"
    spec = importlib.util.spec_from_file_location("_grouped_gemm_ref", path)
    ref = importlib.util.module_from_spec(spec)
    captured = []
    # The module sets the default device to cuda when it loads.
    default = torch.get_default_device()
    try:
        spec.loader.exec_module(ref)
        ref.fused_moe = lambda *a, **k: captured.append((a, k)) or torch.zeros(
            a[0].shape, dtype=torch.bfloat16, device="cuda"
        )
        ref._torch_moe_ref = lambda hidden, *a, **k: torch.zeros(
            hidden.shape, dtype=torch.bfloat16, device="cuda"
        )
        torch.set_default_device("cuda")
        ref._run_grouped_via_fused_moe(
            experts=EXPERTS,
            tokens=tokens,
            topk=TOPK,
            model_dim=MODEL_DIM,
            inter_dim=INTER_DIM,
            data_format="a4w4",
            activation=aiter.ActivationType.Swiglu,
            use_bias=False,
            seed=1,
        )
    finally:
        torch.set_default_device(default)
    args, kwargs = captured[-1]
    return list(args), kwargs


@pytest.mark.parametrize("tokens", [8, 192, 4096])
def test_public_fused_moe_copy(tokens):
    """aiter.fused_moe(mxfp8_copy_out=...) on the M3.1 grouped path: the MoE
    output is unchanged, the copy is the split quant's, and no launch is added."""
    if not torch.cuda.is_available() or get_gfx_runtime() not in SUPPORTED_GFX:
        pytest.skip("gfx1250 grouped MoE")
    args, kwargs = _fused_moe_args(tokens)
    hidden = _hidden(tokens, 5)
    args[0] = hidden
    copy = (
        torch.empty(tokens, hidden.shape[1], dtype=torch.uint8, device="cuda"),
        torch.empty(tokens, hidden.shape[1] // 32, dtype=torch.uint8, device="cuda"),
    )
    plain = fused_moe(*args, **kwargs)
    fused = fused_moe(*args, **kwargs, mxfp8_copy_out=copy)
    torch.cuda.synchronize()
    assert torch.equal(plain, fused)
    want_q, want_s = per_1x32_mx_quant_hip(
        hidden, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    assert torch.equal(copy[0], want_q.view(torch.uint8))
    assert torch.equal(copy[1], want_s.view(torch.uint8))
    plain_k = _kernel_names(lambda: fused_moe(*args, **kwargs))
    fused_k = _kernel_names(lambda: fused_moe(*args, **kwargs, mxfp8_copy_out=copy))
    assert len(plain_k) == len(fused_k), (plain_k, fused_k)
    assert any(k.endswith("_mx8cp") for k in fused_k), fused_k


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "MoE quant MXFP8 copy is gfx1250 only; skipping %s", get_gfx_runtime()
        )
        return
    torch.set_default_device("cuda")
    parser = argparse.ArgumentParser(description="MoE A1 quant + MXFP8 copy")
    parser.add_argument(
        "-v", "--variant", nargs="*", default=list(VARIANTS), choices=list(VARIANTS)
    )
    parser.add_argument(
        "-t", "--tokens", type=int, nargs="*", default=DECODE_TOKENS + PREFILL_TOKENS
    )
    args = parser.parse_args()
    rows = []
    for variant, tokens in _cases():
        if variant not in args.variant or tokens not in args.tokens:
            continue
        rows.append(_run_case(variant, tokens))
    df = pd.DataFrame(rows)
    aiter.logger.info(
        "MoE A1 quant + MXFP8 copy summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
