# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
from unittest import mock

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.shuffle import shuffle_mxfp8_bpreshuffle_scale, shuffle_weight
from aiter.test_common import benchmark, checkAllclose, run_perftest

D, PAGE, EPS = 128, 128, 1e-6
# m, num_heads, num_kv_heads, num_index_heads, rotary_dim, hidden_size
CASES = [
    (1, 16, 1, 1, 64, 6144),  # MiniMax-M3.1 TP4 (tuned rows)
    (8, 16, 1, 4, 64, 6144),
    (96, 16, 1, 1, 64, 6144),
    (150, 16, 1, 1, 64, 6144),
    (128, 16, 1, 4, 64, 6144),
    (5, 8, 2, 2, 128, 4096),  # GQA, full rotary (default config)
    (64, 8, 2, 2, 128, 4096),
    (33, 4, 4, 1, 96, 2560),  # 4 kv heads, rotary 96 (no xor partner lane)
    (40, 2, 3, 2, 120, 1152),  # K not a multiple of 512: one K split
]


def make_case(m, nq, nkv, niq, rot, k, pages=256):
    torch.manual_seed(m * 7 + niq)
    n = (nq + 2 * nkv + niq + 1) * D
    x = torch.randn(m, k).clamp(-8, 8).to(torch.float8_e4m3fn)
    w = (torch.randn(n, k) * 0.5).clamp(-8, 8).to(torch.float8_e4m3fn)
    e8m0 = torch.float8_e8m0fnu
    xs = torch.randint(120, 127, (m, k // 32), dtype=torch.uint8).view(e8m0)
    ws = torch.randint(118, 126, (n, k // 32), dtype=torch.uint8).view(e8m0)
    inv = 1e7 ** -(torch.arange(0, rot, 2, dtype=torch.float64) / rot)
    fr = torch.outer(torch.arange(8192, dtype=torch.float64), inv)
    slots = torch.randperm(pages * PAGE)[:m]
    slots[3::11] = -1  # skipped tokens
    c = {"x": x, "xs": xs, "w": w, "ws": ws, "n": n, "slots": slots}
    c.update(nq=nq, nkv=nkv, niq=niq, rot=rot)
    c["w_sh"] = shuffle_weight(w, layout=(16, 16))
    c["ws_sh"] = shuffle_mxfp8_bpreshuffle_scale(ws)
    c["nw"] = [(torch.randn(D) * 0.3).bfloat16() for _ in range(4)]
    c["cos_sin"] = torch.cat([fr.cos(), fr.sin()], -1).bfloat16()
    c["pos"] = torch.randint(0, 8192, (m,))

    def full(*shape):
        return torch.full(shape, 0xEE, dtype=torch.uint8)

    # pools poisoned: every byte is compared. K and V are the halves of one pool
    # [pages, 2, nkv, 128, row bytes], so the page stride is not nkv rows.
    c["pools"] = lambda: {
        "kv": full(pages, 2, nkv, PAGE, D // 2), "kvs": full(pages, 2, nkv, PAGE, D // 16),
        "ip": full(pages, PAGE, D // 2), "isc": full(pages, PAGE, D // 16),
        "q": torch.full((m, nq * D), -7.0).bfloat16(),
        "iq": torch.full((m, niq * D), -7.0).bfloat16(),
    }  # fmt: skip
    return c


def run_unfused(c, o):
    qkv = aiter.gemm_a8w8_blockscale(
        c["x"], c["w_sh"], c["xs"], c["ws_sh"], dtype=dtypes.bf16, isBpreshuffled=True
    )
    run_tail(c, qkv.clone(), o)
    return qkv


def run_tail(c, qkv, o):
    nw, s = c["nw"], c["slots"]
    aiter.fused_qknorm_idxrqknorm(
        qkv, nw[0], nw[1], c["cos_sin"], c["pos"], c["nq"], c["nkv"], c["rot"], EPS,
        nw[2], nw[3], c["niq"], s, o["kv"][:, 0], o["kv"][:, 1], o["ip"], PAGE, o["q"],
        o["iq"], s, kv_cache_dtype="nvfp4", index_cache_dtype="nvfp4",
        k_scale=o["kvs"][:, 0], v_scale=o["kvs"][:, 1], asm_layout=False,
        index_scale=o["isc"],
    )  # fmt: skip


def run_fused(c, o, qkv_out=None):
    from aiter.ops.flydsl.mxfp8_qkv_rope_nvfp4_gfx1250 import (
        mxfp8_qkv_norm_rope_nvfp4_insert,
    )

    nw, s = c["nw"], c["slots"]
    mxfp8_qkv_norm_rope_nvfp4_insert(
        c["x"], c["xs"], c["w_sh"], c["ws_sh"], *nw, c["cos_sin"], c["pos"], s, s,
        o["kv"][:, 0], o["kv"][:, 1], o["kvs"][:, 0], o["kvs"][:, 1], o["ip"],
        o["isc"], o["q"], o["iq"], c["nq"], c["nkv"], c["rot"], EPS,
        num_index_heads=c["niq"], qkv_out=qkv_out,
    )  # fmt: skip


def run_torch(c):  # fp64 on the CPU
    def deq(codes, scales):
        s = torch.exp2(scales.view(torch.uint8).cpu().double() - 127.0)
        return (codes.cpu().double().unflatten(1, (-1, 32)) * s[..., None]).flatten(1)

    return (deq(c["x"], c["xs"]) @ deq(c["w"], c["ws"]).T).bfloat16()


@benchmark()
def test_mxfp8_qkv_chain(m, nq, nkv, niq, rot, k):
    c = make_case(m, nq, nkv, niq, rot, k)
    of, ou, o2 = c["pools"](), c["pools"](), c["pools"]()
    qkv = torch.full((m, c["n"]), float("nan"), dtype=dtypes.bf16)
    _, us_fused = run_perftest(run_fused, c, of, qkv)
    cur, us_unfused = run_perftest(run_unfused, c, ou)
    err = checkAllclose(run_torch(c).float(), qkv.cpu().float(), rtol=1e-2, atol=1e-2)
    assert err < 1e-3, f"pre-norm rows vs fp64: {err:.2%}"
    same = (qkv.view(torch.int16) == cur.view(torch.int16)).double().mean().item()
    assert same > 0.995, f"pre-norm rows == gemm_a8w8_blockscale's for {same:.2%} only"
    ot = c["pools"]()
    run_tail(c, qkv.clone(), ot)  # the unfused tail on the fused GEMM's rows
    run_fused(c, o2)  # without qkv_out
    for key in of:
        assert torch.equal(of[key].view(torch.uint8), ot[key].view(torch.uint8)), key
        assert torch.equal(o2[key].view(torch.uint8), of[key].view(torch.uint8)), key
    return {"fused us": us_fused, "unfused us": us_unfused, "gemm bitwise": same}


def test_chain_gate():
    from aiter.ops.flydsl import mxfp8_qkv_rope_nvfp4_gfx1250 as chain
    from aiter.ops.flydsl.kernels.mxfp8_qkv_rope_nvfp4_gfx1250 import (
        chain_config_error,
    )

    for key, rows in chain._tuned_table().items():  # every tuned row is valid
        for m, cfg in rows:
            assert chain_config_error(key[-1], *cfg[:8]) is None, (key, m, cfg)

    ok = chain.mxfp8_qkv_chain_supported
    m3 = (16, 1, D, 64, 6144)  # (num_heads, num_kv_heads, head_dim, rotary_dim, K)
    gqa = (8, 2, D, 128, 4096)  # no tuned rows: the default config
    with (
        mock.patch("aiter.jit.utils.chip_info.get_gfx_runtime", return_value="gfx1250"),
        mock.patch("aiter.jit.utils.chip_info.get_cu_num", return_value=256),
    ):
        assert ok(1, *m3) and ok(192, *m3) and ok(128, *m3, num_index_heads=4)
        assert not ok(193, *m3) and not ok(129, *m3, num_index_heads=4)
        assert ok(1, *gqa) and ok(64, *gqa) and not ok(65, *gqa)
        assert ok(33, 4, 4, D, 96, 2560) and ok(40, 2, 3, D, 120, 1152)
        assert not ok(8, 16, 1, 64, 64, 6144)  # head_dim 64
        assert not ok(8, 16, 1, D, 60, 6144)  # rotary_dim not a multiple of 8
        assert not ok(8, 16, 1, D, 136, 6144)  # rotary_dim > head_dim
        assert not ok(8, 16, 1, D, 64, 6000)  # K not a multiple of 128
        assert not ok(8, *m3, num_index_heads=0) and not ok(0, *m3)
    with mock.patch("aiter.jit.utils.chip_info.get_gfx_runtime", return_value="gfx950"):
        assert not ok(64, *m3)


def main():
    parser = argparse.ArgumentParser(description="fused MXFP8 QKV chain (gfx1250)")
    parser.add_argument(
        "-s", "--shape", type=dtypes.str2tuple, nargs="*", default=CASES,
        help="m,num_heads,num_kv_heads,num_index_heads,rotary_dim,hidden_size",
    )  # fmt: skip
    args = parser.parse_args()
    test_chain_gate()
    if get_gfx_runtime() != "gfx1250":
        aiter.logger.warning("the fused QKV chain runs on gfx1250 only; skipping")
        return
    torch.set_default_device("cuda")
    df = [test_mxfp8_qkv_chain(*shape) for shape in args.shape]
    aiter.logger.info("mxfp8_qkv_chain:\n%s", pd.DataFrame(df).to_markdown(index=False))


if __name__ == "__main__":
    main()
