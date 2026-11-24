# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Run on gfx1250: python -m pytest op_tests/test_mla_nvfp4.py -q.

The NumPy oracle checks the specified quantized algorithm, independently of
native lane/LDS layout. It is an implementation check, not evidence of model
accuracy. The primary operator accuracy reference is full FP32 softmax attention
on original BF16 inputs in test_mla_decode_pagesize64.py::TestMlaNvfp4Page64.
Set AITER_NVFP4_TEST_LARGE=1 to exercise both KV streams above 4 GiB (~12 GiB).
"""

import math
import os
import struct

import numpy as np
import pytest
import torch

import aiter
from aiter.ops.mla_nvfp4 import (
    mla_decode_nvfp4_fwd,
    mla_nvfp4_asm_fwd,
    mla_nvfp4_quantize_q,
    mla_nvfp4_update_cache,
    prepare_mla_nvfp4_params,
)

F4 = np.array(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32
)


def e4(c):
    c = np.asarray(c, np.uint8)
    e, m = (c >> 3).astype(np.int32), (c & 7).astype(np.float32)
    return np.where(
        c == 127, np.nan, np.where(e == 0, m / 512, (1 + m / 8) * np.exp2(e - 7))
    ).astype(np.float32)


S8 = e4(np.arange(127))


def nearest(x, table):
    hi = np.searchsorted(table, x).clip(0, len(table) - 1)
    lo = (hi - 1).clip(0)
    a, b = x - table[lo], table[hi] - x
    return np.where((a < b) | ((a == b) & ((lo & 1) == 0)), lo, hi).astype(np.uint8)


def quant_blocks(x, gain, ideal=False):
    x = np.asarray(x, np.float32)
    am = np.max(np.abs(x), axis=-1)
    div = np.where(am == 0, np.float32(1), am / np.float32(6))
    sf = nearest(np.clip(np.float32(gain) * div, 1 / 512, 448), S8)
    if not ideal:
        div = S8[sf] / np.float32(gain)
    scaled = x / div[..., None]
    code = nearest(np.abs(scaled), F4[:8]) | (np.signbit(scaled).astype(np.uint8) * 8)
    return code, sf


def quant_rows(x, gain):
    shape = x.shape
    c, s = quant_blocks(x.reshape(*shape[:-1], -1, 16), gain)
    return c.reshape(shape), s


def quant_rope(x, gain):
    c, s = quant_rows(x, gain)
    residual = x - F4[c] * np.repeat(S8[s], 16, axis=-1) / np.float32(gain)
    ec, es = quant_rows(residual, gain)
    return c, s, ec, es


def pack4(c):
    return c[..., ::2] | (c[..., 1::2] << 4)


def pack(qc, qs, cc, cs):
    b, _, _ = qc.shape
    p = cc.shape[0]
    qn = pack4(qc[..., :512]).copy()
    qa = np.full((b, 128, 128), 127, np.uint8)
    qa[..., :64], qa[..., 64:104] = pack4(qc[..., 512:]), qs
    cn = (
        pack4(cc[..., :512])
        .reshape(p, 4, 16, 16, 16)
        .transpose(0, 1, 3, 2, 4)
        .copy()
        .reshape(p, 16384)
    )
    ca = np.full((p, 4, 8, 16, 16), 127, np.uint8)
    for g in range(4):
        for f in range(2):
            ca[:, g, 2 * f : 2 * f + 2] = (
                pack4(cc[:, g * 16 : (g + 1) * 16, 512 + f * 64 : 576 + f * 64])
                .reshape(p, 16, 2, 16)
                .transpose(0, 2, 1, 3)
            )
        ca[:, g, 4:6] = (
            cs[:, g * 16 : (g + 1) * 16, :32]
            .reshape(p, 16, 2, 16)
            .transpose(0, 2, 1, 3)
        )
        ca[:, g, 6, :, :8] = cs[:, g * 16 : (g + 1) * 16, 32:]
    return qn, qa, cn, ca.reshape(p, 8192)


def gpu(x):
    return torch.from_numpy(np.ascontiguousarray(x)).to("cuda")


def make_case(lengths, mode="random", threshold=1.0, scale=576**-0.5):
    lengths = np.array(lengths, np.int32)
    counts = (lengths + 63) // 64
    ptr = np.r_[0, counts.cumsum()].astype(np.int32)
    p, b = max(1, int(ptr[-1])), len(lengths)
    rng = np.random.default_rng(321)
    idx = rng.permutation(int(ptr[-1])).astype(np.int32)
    last = np.where(counts, (lengths - 1) % 64 + 1, 0).astype(np.int32)
    qc = rng.integers(0, 16, (b, 128, 640), dtype=np.uint8)
    qs = rng.choice(np.array([0x30, 0x38, 0x40, 0x44], np.uint8), (b, 128, 40))
    cc = rng.integers(0, 16, (p, 64, 640), dtype=np.uint8)
    cs = rng.choice(np.array([0x58, 0x60, 0x64, 0x68], np.uint8), (p, 64, 40))
    cs[..., :32] = np.repeat(cs[:, ::16, :32], 16, axis=1)
    # Only NoPE shares scales over tokens. RoPE/residual vary at every token.
    cs[..., 36:] = rng.choice(np.array([0x30, 0x38, 0x40], np.uint8), (p, 64, 4))
    qs[..., 36:] = 0x18
    if mode == "rescale":
        qc.fill(2)
        qc[:, 64:, :512] = 10
        qc[..., 512:] = 0
        qs.fill(0x38)
        cc.fill(2)
        cc[..., 512:] = 0
        cs.fill(0x60)
        for r in range(b):
            for j, page in enumerate(idx[ptr[r] : ptr[r + 1]]):
                cc[page, :, :512] = [10, 2, 11, 3, 12, 4, 13, 5][(j // 2) % 8]
    elif mode in ("nope", "rope", "residual_q", "residual_k"):
        qrange = {
            "nope": (0, 512),
            "rope": (512, 576),
            "residual_q": (576, 640),
            "residual_k": (512, 576),
        }[mode]
        qc[..., : qrange[0]] = 0
        qc[..., qrange[1] :] = 0
        if mode == "nope":
            cc[..., 512:] = 0
        elif mode in ("rope", "residual_q"):
            cc[..., 576:] = 0
        else:
            cc[..., 512:576] = 0
    if mode == "constant_v":
        cc[..., :512] = 2
        cs[..., :32] = 0x60
    q = F4[qc] * np.repeat(e4(qs), 16, axis=-1)
    c = F4[cc] * np.repeat(e4(cs), 16, axis=-1)
    for r in range(b):
        if counts[r]:
            cs[idx[ptr[r + 1] - 1], last[r] :] = 127
    qn, qa, cn, ca = map(gpu, pack(qc, qs, cc, cs))
    out_storage = torch.full(
        (b * 128 * 512 + 256,), float("nan"), dtype=torch.bfloat16, device="cuda"
    )
    lse_storage = torch.full((b * 128 + 256,), 12345.0, device="cuda")
    args = {
        "q_nope": qn,
        "q_aux": qa,
        "kv_nope": cn,
        "kv_aux": ca,
        "out": out_storage[: b * 128 * 512].view(b, 128, 512),
        "kv_indptr": gpu(ptr),
        "kv_indices": gpu(idx),
        "kv_last_page_lens": gpu(last),
        "lse": lse_storage[: b * 128].view(b, 128),
        "params": prepare_mla_nvfp4_params(
            device="cuda", physical_pages=p, sm_scale=scale, rebase_threshold=threshold
        ),
        "return_lse": False,
    }
    logical = {
        "q": q,
        "c": c,
        "lengths": lengths,
        "idx": idx,
        "ptr": ptr,
        "threshold": threshold,
        "beta": np.float32(scale * math.log2(math.e) / (6.72 * 89.6)),
    }
    return args, logical, (out_storage, lse_storage)


def quantized_algorithm_oracle(case):
    outputs = []
    for r, length in enumerate(case["lengths"]):
        if length == 0:
            outputs.append(np.zeros((128, 512), np.float32))
            continue
        c = case["c"][case["idx"][case["ptr"][r] : case["ptr"][r + 1]]].reshape(
            -1, 640
        )[:length]
        q = case["q"][r]
        anchor = None
        den, acc = np.zeros(128, np.float32), np.zeros((128, 512), np.float32)
        for start in range(0, int(length), 128):
            ct = c[start : start + 128]
            n = len(ct)
            score = np.zeros((128, n), np.float32)
            for k in range(4):
                score += (
                    q[:, k * 128 : (k + 1) * 128].astype(np.float64)
                    @ ct[:, k * 128 : (k + 1) * 128].astype(np.float64).T
                ).astype(np.float32)
            score += (
                np.concatenate((q[:, 512:576], q[:, 512:576]), axis=1).astype(
                    np.float64
                )
                @ ct[:, 512:].astype(np.float64).T
            ).astype(np.float32)
            score += (
                q[:, 576:].astype(np.float64) @ ct[:, 512:576].astype(np.float64).T
            ).astype(np.float32)
            raw = np.full((128, 128), -np.inf, np.float32)
            raw[:, :n] = score
            raw = raw.reshape(128, 8, 16)
            gm = raw.max(axis=2) * case["beta"]
            tilemax = gm.max(axis=1)
            if anchor is None:
                anchor = tilemax.copy()
            else:
                new = np.where(tilemax - anchor > case["threshold"], tilemax, anchor)
                alpha = np.exp2((anchor - new).astype(np.float64)).astype(np.float32)
                den *= alpha
                acc *= alpha[:, None]
                anchor = new
            # Local N16 exponentials under a common N128 online anchor.
            # Each group quantizes 6*e to FP4 and 5*w to FP8. Both numerator
            # and denominator use that same quantized mass. Masked groups are zero.
            # Evaluate exp2 in FP64 then round, to avoid NumPy FP32 SIMD error
            # moving a probability across an E2M1 midpoint.
            exponent = (
                raw.astype(np.float64) * float(case["beta"])
                - np.maximum(gm, np.finfo(np.float32).min)[..., None]
            ).astype(np.float32)
            local = np.exp2(exponent.astype(np.float64)).astype(np.float32)
            delta = gm - anchor[:, None]
            relative = np.exp2(delta.astype(np.float64)).astype(np.float32)
            p = (
                F4[nearest(local * np.float32(6), F4[:8])]
                * S8[nearest(relative * 5, S8)][..., None]
            )
            den += p.astype(np.float64).sum(axis=(1, 2)).astype(np.float32)
            acc += (
                p.reshape(128, 128)[:, :n].astype(np.float64)
                @ ct[:, :512].astype(np.float64)
            ).astype(np.float32)
        outputs.append(acc * ((np.float32(1) / den) * np.float32(1 / 89.6))[:, None])
    if not outputs:
        return np.empty((0, 128, 512), np.float32)
    o = torch.from_numpy(np.stack(outputs)).to(torch.bfloat16).float().numpy()
    return o


def check(args, logical, guards=None):
    out, lse = mla_decode_nvfp4_fwd(**args)
    assert lse is None
    expected = quantized_algorithm_oracle(logical)
    actual = out.float().cpu().numpy()
    if actual.size:
        assert np.isfinite(actual).all()
        rms = np.sqrt(np.mean(expected.astype(np.float64) ** 2))
        diff = actual - expected
        assert np.sqrt(np.mean(diff.astype(np.float64) ** 2)) <= 0.003 * max(rms, 1e-8)
        assert np.max(np.abs(diff)) <= 0.025
    if guards:
        assert torch.isnan(guards[0][-256:]).all()
        assert torch.all(guards[1] == 12345)
    return out.clone()


@pytest.fixture(scope="module", autouse=True)
def require_gpu():
    if not torch.cuda.is_available() or not torch.cuda.get_device_properties(
        0
    ).gcnArchName.startswith("gfx1250"):
        pytest.skip("requires gfx1250")
    if not aiter.is_gfx1250_asm_supported():
        pytest.skip("requires gfx1250 B0+")


@pytest.mark.parametrize(
    "mode",
    ["random", "rescale", "nope", "rope", "residual_q", "residual_k", "constant_v"],
)
def test_decode_quantized_algorithm(mode):
    args, logical, guards = make_case(
        [
            0,
            1,
            15,
            16,
            17,
            31,
            32,
            33,
            63,
            64,
            65,
            127,
            128,
            129,
            511,
            512,
            513,
            1023,
            1024,
            1025,
            1153,
            2049,
            4097,
        ],
        mode,
    )
    expected = check(args, logical, guards)
    for _ in range(2):
        mla_decode_nvfp4_fwd(**args)
        assert torch.equal(args["out"], expected)
        assert (args["lse"] == 12345).all()


def test_threshold_and_params():
    args, logical, guards = make_case([1, 129, 1153, 2049], "rescale")
    decoded = struct.unpack("<fffI", args["params"].cpu().numpy().tobytes())
    expected = struct.unpack(
        "<fffI",
        struct.pack("<fffI", logical["beta"], 1 / 89.6, 1.0, args["kv_nope"].shape[0]),
    )
    assert decoded == expected
    check(args, logical, guards)


def test_constant_v_normalization():
    args, logical, guards = make_case([1, 17, 65, 129, 513, 2049], "constant_v")
    out = check(args, logical, guards)
    # With matching quantized numerator/denominator mass, O must preserve V.
    expected = torch.full_like(out, 32 / 89.6)
    torch.testing.assert_close(out, expected, atol=0, rtol=0)


def test_empty_batch():
    args, logical, guards = make_case([])
    check(args, logical, guards)


def test_native_public_and_dispatch():
    args, logical, guards = make_case([0, 65, 129])
    expected = check(args, logical, guards)
    mla_nvfp4_asm_fwd(
        *(
            args[n]
            for n in (
                "q_nope",
                "q_aux",
                "kv_nope",
                "kv_aux",
                "kv_indptr",
                "kv_indices",
                "kv_last_page_lens",
                "out",
                "lse",
                "params",
            )
        )
    )
    assert torch.equal(args["out"], expected)
    kwargs = {
        "q": args["q_nope"],
        "kv_buffer": args["kv_nope"],
        "o": args["out"],
        "qo_indptr": None,
        "kv_indptr": args["kv_indptr"],
        "kv_indices": args["kv_indices"],
        "kv_last_page_lens": args["kv_last_page_lens"],
        "max_seqlen_q": 1,
        "page_size": 64,
        "kv_layout": "nvfp4_n128",
        "q_aux": args["q_aux"],
        "kv_aux": args["kv_aux"],
        "nvfp4_params": args["params"],
        "nvfp4_lse": args["lse"],
        "return_lse": False,
    }
    out, lse = aiter.mla.mla_decode_fwd(**kwargs)
    assert torch.equal(out, expected) and lse is None
    for change in (
        {"num_kv_splits": 2},
        {"max_seqlen_q": 2},
        {"cp_world_size": 2},
        {"sm_scale": 0.1},
        {"return_logits": True},
        {"kv_csr_layout": "token"},
        {"return_lse": True},
    ):
        with pytest.raises(ValueError):
            aiter.mla.mla_decode_fwd(**(kwargs | change))
    with pytest.raises(ValueError, match="does not produce LSE"):
        mla_decode_nvfp4_fwd(**(args | {"return_lse": True}))
    out, lse = mla_decode_nvfp4_fwd(**(args | {"lse": None}))
    assert torch.equal(out, expected) and lse is None
    out, lse = aiter.mla.mla_decode_fwd(**(kwargs | {"nvfp4_lse": None}))
    assert torch.equal(out, expected) and lse is None
    assert (args["lse"] == 12345).all()


def test_contracts():
    args, _, _ = make_case([64])
    for name, tensor in (
        ("q_aux", args["q_aux"].float()),
        ("kv_nope", args["kv_nope"][:, :-1]),
        ("kv_indptr", args["kv_indptr"].long()),
        ("lse", args["lse"].to(torch.bfloat16)),
        ("params", args["params"].float()),
        ("q_nope", args["q_nope"][:, ::2]),
    ):
        with pytest.raises((RuntimeError, ValueError)):
            mla_decode_nvfp4_fwd(**(args | {name: tensor}))
    bad = torch.empty(args["kv_aux"].numel() + 1, device="cuda", dtype=torch.uint8)[
        1:
    ].view_as(args["kv_aux"])
    with pytest.raises(RuntimeError):
        mla_decode_nvfp4_fwd(**(args | {"kv_aux": bad}))
    for kw in (
        {"physical_pages": 0},
        {"sm_scale": float("nan")},
        {"rebase_threshold": 0},
        {"rebase_threshold": 6},
        {"rebase_threshold": float("nan")},
        {"rebase_threshold": 8},
    ):
        with pytest.raises(ValueError):
            prepare_mla_nvfp4_params(
                **({"device": "cuda", "physical_pages": 1, "sm_scale": 0.1} | kw)
            )
    with pytest.raises(TypeError, match="aux_layout"):
        mla_nvfp4_update_cache(None, None, None, None, None, None, aux_layout="n32")


def test_stream_graph_compile():
    args, logical, _ = make_case([0, 65, 129])
    expected = check(args, logical)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        mla_decode_nvfp4_fwd(**args)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            mla_decode_nvfp4_fwd(**args)
        graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    assert torch.equal(args["out"], expected)
    args["kv_indptr"].zero_()
    args["kv_last_page_lens"].zero_()
    graph.replay()
    torch.cuda.synchronize()
    assert (args["out"] == 0).all() and (args["lse"] == 12345).all()
    args, logical, _ = make_case([65])
    args["lse"] = None
    compiled = torch.compile(mla_decode_nvfp4_fwd, fullgraph=True)
    out, lse = compiled(**args)
    out = out.clone()
    expected = check(args, logical)
    assert torch.equal(out, expected) and lse is None


def test_quantize_q():
    rng = np.random.default_rng(19)
    raw = torch.from_numpy(rng.normal(0, 0.2, (1, 128, 576)).astype(np.float32)).to(
        torch.bfloat16
    )
    x = raw.float().numpy()
    nc, ns = quant_rows(x[..., :512], 6.72)
    rc, rs, ec, es = quant_rope(x[..., 512:], 6.72)
    qc, qs = np.concatenate((nc, rc, ec), axis=-1), np.concatenate(
        (ns, rs, es), axis=-1
    )
    qn, qa = mla_nvfp4_quantize_q(raw.cuda())
    ref = pack(
        qc, qs, np.zeros((1, 64, 640), np.uint8), np.zeros((1, 64, 40), np.uint8)
    )
    np.testing.assert_array_equal(qn.cpu().numpy(), ref[0])
    np.testing.assert_array_equal(qa.cpu().numpy(), ref[1])


def cpu_cache(raw, valid):
    cc = np.zeros((2, 64, 640), np.uint8)
    cs = np.full((2, 64, 40), 127, np.uint8)
    x = np.where(np.arange(16)[:, None] < valid, raw, 0).astype(np.float32)
    for g in range(32):
        codes, sf = quant_blocks(x[:, g * 16 : (g + 1) * 16].reshape(256), 89.6, True)
        cc[1, 48:64, g * 16 : (g + 1) * 16] = codes.reshape(16, 16)
        cs[1, 48 : 48 + valid, g] = sf
    rc, rs, ec, es = quant_rope(x[:, 512:], 89.6)
    cc[1, 48:64, 512:] = np.concatenate((rc, ec), axis=-1)
    cs[1, 48 : 48 + valid, 32:] = np.concatenate((rs, es), axis=-1)[:valid]
    return pack(
        np.zeros((1, 128, 640), np.uint8),
        np.zeros((1, 128, 40), np.uint8),
        cc,
        cs,
    )[2:]


def test_cache_tail_requantization_and_graph():
    rng = np.random.default_rng(31)
    raw = torch.from_numpy(rng.normal(0, 0.15, (1, 16, 576)).astype(np.float32)).to(
        torch.bfloat16
    )
    raw[:, :5, :512] = 0.125
    raw[:, 5, :512] = 6
    raw[:, 6:] = float("nan")
    tiles = raw.cuda()
    pages = gpu(np.array([1], np.int32))
    groups = gpu(np.array([3], np.int32))
    valid = gpu(np.array([5], np.int32))
    cn = torch.zeros((2, 16384), device="cuda", dtype=torch.uint8)
    ca = torch.full((2, 8192), 127, device="cuda", dtype=torch.uint8)
    mla_nvfp4_update_cache(tiles, pages, groups, valid, cn, ca)
    previous = cn.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mla_nvfp4_update_cache(tiles, pages, groups, valid, cn, ca)
    valid.fill_(6)
    graph.replay()
    ref_cn, ref_ca = cpu_cache(raw[0].float().numpy(), 6)
    np.testing.assert_array_equal(cn[1, 12288:].cpu().numpy(), ref_cn[1, 12288:])
    assert not torch.equal(cn[1, 12288:12544], previous[1, 12288:12544])
    assert (cn[0] == 0).all() and (ca[0] == 127).all()
    np.testing.assert_array_equal(ca[1, 6144:].cpu().numpy(), ref_ca[1, 6144:])
    assert (ca[1, :6144] == 127).all()


def test_bf16_producers_to_decode():
    rng = np.random.default_rng(43)
    lengths = np.array([0, 17, 65], np.int32)
    ptr, idx = np.array([0, 0, 1, 3], np.int32), np.array([2, 0, 1], np.int32)
    page_lens = np.array([64, 1, 17], np.int32)
    rawq = torch.from_numpy(rng.normal(0, 0.2, (3, 128, 576)).astype(np.float32)).to(
        torch.bfloat16
    )
    rawkv = torch.from_numpy(rng.normal(0, 0.2, (3, 64, 576)).astype(np.float32)).to(
        torch.bfloat16
    )
    for p, n in enumerate(page_lens):
        rawkv[p, n:] = float("nan")
    q = rawq.float().numpy()
    nc, ns = quant_rows(q[..., :512], 6.72)
    rc, rs, ec, es = quant_rope(q[..., 512:], 6.72)
    qc, qs = np.concatenate((nc, rc, ec), axis=-1), np.concatenate(
        (ns, rs, es), axis=-1
    )
    cc, cs = np.zeros((3, 64, 640), np.uint8), np.ones((3, 64, 40), np.uint8)
    for p, n in enumerate(page_lens):
        for tile in range(4):
            start = tile * 16
            x = rawkv[p, start : start + 16].float().numpy()
            x = np.where(np.arange(16)[:, None] < max(0, int(n) - start), x, 0)
            for g in range(32):
                code, sf = quant_blocks(
                    x[:, g * 16 : (g + 1) * 16].reshape(256), 89.6, True
                )
                cc[p, start : start + 16, g * 16 : (g + 1) * 16] = code.reshape(16, 16)
                cs[p, start : start + 16, g] = sf
            rc, rs, ec, es = quant_rope(x[:, 512:], 89.6)
            cc[p, start : start + 16, 512:] = np.concatenate((rc, ec), axis=-1)
            cs[p, start : start + 16, 32:] = np.concatenate((rs, es), axis=-1)
    qn, qa = mla_nvfp4_quantize_q(rawq.cuda())
    cn = torch.empty((3, 16384), device="cuda", dtype=torch.uint8)
    ca = torch.empty((3, 8192), device="cuda", dtype=torch.uint8)
    mla_nvfp4_update_cache(
        rawkv.cuda().reshape(12, 16, 576),
        gpu(np.repeat(np.arange(3, dtype=np.int32), 4)),
        gpu(np.tile(np.arange(4, dtype=np.int32), 3)),
        gpu(
            np.maximum(
                0,
                np.minimum(16, page_lens[:, None] - np.arange(4, dtype=np.int32) * 16),
            ).reshape(-1)
        ),
        cn,
        ca,
    )
    args = {
        "q_nope": qn,
        "q_aux": qa,
        "kv_nope": cn,
        "kv_aux": ca,
        "out": torch.empty((3, 128, 512), device="cuda", dtype=torch.bfloat16),
        "kv_indptr": torch.empty(4, device="cuda", dtype=torch.int32),
        "kv_indices": torch.empty(6, device="cuda", dtype=torch.int32),
        "kv_last_page_lens": torch.empty(3, device="cuda", dtype=torch.int32),
        "params": prepare_mla_nvfp4_params(
            device="cuda", physical_pages=3, sm_scale=0.07
        ),
        "lse": torch.empty((3, 128), device="cuda"),
        "return_lse": False,
    }
    # Existing GPU dense-to-CSR adapter; unused table entries are poisoned.
    aiter.mla.mla_decode_prepare_page_csr(
        gpu(lengths),
        gpu(np.array([[-1, -1], [2, -1], [0, 1]], np.int32)),
        args["kv_indptr"],
        args["kv_indices"],
        args["kv_last_page_lens"],
    )
    logical = {
        "q": F4[qc] * np.repeat(S8[qs], 16, axis=-1),
        "c": F4[cc] * np.repeat(S8[cs], 16, axis=-1),
        "lengths": lengths,
        "idx": idx,
        "ptr": ptr,
        "threshold": 1.0,
        "beta": np.float32(0.07 * math.log2(math.e) / (6.72 * 89.6)),
    }
    check(args, logical)


@pytest.mark.skipif(
    os.getenv("AITER_NVFP4_TEST_LARGE") != "1", reason="opt-in 12 GiB allocation"
)
def test_pages_above_4gib():
    if torch.cuda.mem_get_info()[0] < 14 * 1024**3:
        pytest.skip("needs 14 GiB free")
    args, logical, _ = make_case([64])
    p = (1 << 19) + 1
    cn = torch.empty((p, 16384), device="cuda", dtype=torch.uint8)
    ca = torch.empty((p, 8192), device="cuda", dtype=torch.uint8)
    cn[-1].copy_(args["kv_nope"][0])
    ca[-1].copy_(args["kv_aux"][0])
    args.update(
        kv_nope=cn,
        kv_aux=ca,
        kv_indices=gpu(np.array([p - 1], np.int32)),
        params=prepare_mla_nvfp4_params(
            device="cuda", physical_pages=p, sm_scale=576**-0.5
        ),
    )
    check(args, logical)
