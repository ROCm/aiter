# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 fused MiniMax-M3 decode QKV chain (mxfp8_qkv_norm_rope_nvfp4_insert).

Against the unfused chain the model runs: gemm_a8w8_blockscale(isBpreshuffled=True)
+ fused_qknorm_idxrqknorm (NVFP4 row-layout K/V and index rows).
  * the fused kernel's pre-norm rows (qkv_out) are within the GEMM's tolerance of an
    fp64 reference and equal the tuned GEMM's bf16 rows but for rare one-ulp flips;
  * fused_qknorm_idxrqknorm on those rows reproduces every byte the fused kernel
    wrote: q, index-q, K/V payload + scales, index payload + scales, and nothing
    else of the pools (pad slots -1 included);
  * a second launch without qkv_out writes the same bytes (deterministic).

Run with: python -m pytest -q op_tests/test_mxfp8_qkv_norm_rope_nvfp4_insert.py
"""

import pytest
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale, shuffle_weight

D, PAGE, K = 128, 128, 6144
NQ, NKV, ROT = 16, 1, 64
EPS = 1e-6


@pytest.fixture(scope="module", autouse=True)
def _gpu():
    if not torch.cuda.is_available() or get_gfx_runtime() != "gfx1250":
        pytest.skip("gfx1250 only")


def make_case(m, niq, seed, pages=512):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)
    n = (NQ + 2 * NKV + niq + 1) * D

    def e8m0(shape, lo, hi):
        return (
            torch.randint(lo, hi, shape, generator=g, device=dev, dtype=torch.int32)
            .to(torch.uint8)
            .view(torch.float8_e8m0fnu)
        )

    x = torch.randn(m, K, generator=g, device=dev).clamp(-8, 8).to(torch.float8_e4m3fn)
    w = (
        (torch.randn(n, K, generator=g, device=dev) * 0.5)
        .clamp(-8, 8)
        .to(torch.float8_e4m3fn)
    )
    c = dict(m=m, n=n, niq=niq, x=x, xs=e8m0((m, K // 32), 120, 127), w=w,
             ws=e8m0((n, K // 32), 118, 126), pages=pages)  # fmt: skip
    c["w_sh"] = shuffle_weight(w, layout=(16, 16))
    c["ws_sh"] = shuffle_mxfp8_bpreshuffle_scale(c["ws"])
    c["nw"] = [
        (torch.randn(D, generator=g, device=dev) * 0.3).to(torch.bfloat16)
        for _ in range(4)
    ]
    inv = 1.0 / (
        1e7 ** (torch.arange(0, ROT, 2, device=dev, dtype=torch.float64) / ROT)
    )
    fr = torch.outer(torch.arange(8192, device=dev, dtype=torch.float64), inv)
    c["cos_sin"] = torch.cat([fr.cos(), fr.sin()], -1).to(torch.bfloat16).contiguous()
    c["pos"] = torch.randint(0, 8192, (m,), generator=g, device=dev, dtype=torch.int64)
    slots = torch.randperm(pages * PAGE, generator=g, device=dev)[:m].to(torch.int64)
    slots[3::11] = -1
    c["slots"] = slots
    return c


def pools(c):
    dev, p, m = c["x"].device, c["pages"], c["m"]
    u8 = dict(dtype=torch.uint8, device=dev)
    return dict(
        kp=torch.full((p, NKV, PAGE, D // 2), 0xEE, **u8),
        vp=torch.full((p, NKV, PAGE, D // 2), 0xEE, **u8),
        ksc=torch.full((p, NKV, PAGE, D // 16), 0xEE, **u8),
        vsc=torch.full((p, NKV, PAGE, D // 16), 0xEE, **u8),
        ip=torch.full((p, PAGE, D // 2), 0xEE, **u8),
        isc=torch.full((p, PAGE, D // 16), 0xEE, **u8),
        q=torch.full((m, NQ * D), -7.0, dtype=torch.bfloat16, device=dev),
        iq=torch.full((m, c["niq"] * D), -7.0, dtype=torch.bfloat16, device=dev),
    )


def run_tail(c, qkv, o):
    nw = c["nw"]
    aiter.fused_qknorm_idxrqknorm(
        qkv, nw[0], nw[1], c["cos_sin"], c["pos"], NQ, NKV, ROT, EPS, nw[2], nw[3],
        c["niq"], c["slots"], o["kp"], o["vp"], o["ip"], PAGE, o["q"], o["iq"],
        c["slots"], kv_cache_dtype="nvfp4", index_cache_dtype="nvfp4",
        k_scale=o["ksc"], v_scale=o["vsc"], asm_layout=False, index_scale=o["isc"],
    )  # fmt: skip


def run_fused(c, o, qkv_out=None):
    from aiter.ops.flydsl.mxfp8_qkv_rope_nvfp4_gfx1250 import (
        mxfp8_qkv_norm_rope_nvfp4_insert,
    )

    nw = c["nw"]
    mxfp8_qkv_norm_rope_nvfp4_insert(
        c["x"], c["xs"], c["w_sh"], c["ws_sh"], nw[0], nw[1], nw[2], nw[3],
        c["cos_sin"], c["pos"], c["slots"], c["slots"], o["kp"], o["vp"], o["ksc"],
        o["vsc"], o["ip"], o["isc"], o["q"], o["iq"], EPS,
        num_index_heads=c["niq"], qkv_out=qkv_out,
    )  # fmt: skip


def deq(codes, scales):
    s = torch.exp2(scales.view(torch.uint8).double() - 127.0)
    return (codes.double().view(codes.shape[0], -1, 32) * s[..., None]).view(
        codes.shape[0], -1
    )


@pytest.mark.parametrize("m, niq", [(1, 1), (16, 1), (33, 1), (64, 1), (80, 1), (96, 1),
                                    (100, 1), (128, 1), (150, 1), (192, 1), (64, 4),
                                    (96, 4), (128, 4)])  # fmt: skip
def test_fused_chain(m, niq):
    c = make_case(m, niq, seed=m * 7 + niq)
    of = pools(c)
    qkv = torch.full((m, c["n"]), float("nan"), dtype=torch.bfloat16, device="cuda")
    run_fused(c, of, qkv_out=qkv)
    ref = deq(c["x"], c["xs"]) @ deq(c["w"], c["ws"]).T
    cur = aiter.gemm_a8w8_blockscale(
        c["x"],
        c["w_sh"],
        c["xs"],
        c["ws_sh"],
        dtype=torch.bfloat16,
        isBpreshuffled=True,
    )
    torch.cuda.synchronize()
    refb = ref.to(torch.bfloat16).double()
    bad = (
        ((qkv.double() - refb).abs() > 1e-2 + 1e-2 * refb.abs()).double().mean().item()
    )
    assert not torch.isnan(qkv).any() and bad < 1e-3, bad
    same = (qkv.view(torch.int16) == cur.view(torch.int16)).double().mean().item()
    assert same > 0.995, same
    ot = pools(c)
    run_tail(c, qkv.clone(), ot)
    o2 = pools(c)
    run_fused(c, o2)
    torch.cuda.synchronize()
    for key in of:
        a = of[key].view(torch.uint8)
        assert torch.equal(a, ot[key].view(torch.uint8)), key
        assert torch.equal(a, o2[key].view(torch.uint8)), key
