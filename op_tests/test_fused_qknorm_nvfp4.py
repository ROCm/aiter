# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""fused_qknorm_idxrqknorm with kv_cache_dtype / index_cache_dtype = "nvfp4".

The expected pages are built in torch from the bf16 K / V / index-K rows the same op writes
with kv_cache_dtype "auto" (norm + RoPE, rounded to bf16; checked against torch in
test_fused_qknorm_idxrqknorm.py): scale = e4m3_rne(min(amax/6/gs, 448)) per 16 elements,
e2m1 nearest grid value with ties to the even code, bytes placed by the page format below.
Whole pages are compared byte for byte (checkAllclose with zero tolerance), on pages
pre-filled with garbage (a V nibble must clear its old value), with scattered, skipped and
octet-sharing slots. The summary table times the nvfp4 write against the mxfp4 and fp8 write
modes of the same kernel at 1 / 64 / 8192 / 32768 tokens.

Page format (uint8; 128 tokens per page, head_dim 128, one e4m3 scale per 16 elements, gs the
fp32 global scale given as k_scale / v_scale, 1.0 for the index key):
  index page 9216 B : [128 tok x 64 B e2m1][128 tok x 8 B e4m3]
  KV page 2 * nkv * 9216 B : [K data | K scale | V data | V scale], each side nkv heads of
    8192 B data + 1024 B scale; K data token-major, K scale in MFMA lane order, V data in
    32-bit words (8 tokens x 4 bit per word), V scale [group][token].

Row layout (kv_cache_dtype "nvfp4" with byte k_scale / v_scale planes; gfx950 and gfx1250):
per (page, kv head) an e2m1 payload [128 tok x 64 B] and an E4M3 scale plane [128 tok x 8 B],
no global scale; with index_cache_dtype "nvfp4" and index_scale the index key too, per page.
test_nvfp4_rows checks it byte for byte against "auto" + ref_rows, on strided fields of one
pre-filled pool, also with odd payload bases and odd page and head strides.
"""

import argparse
import itertools
import math
import subprocess
import sys
from unittest import mock

import pandas as pd
import torch

import aiter
import aiter.ops.fused_qknorm_idxrqknorm as op
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

HEAD_DIM = 128
PAGE = 128
HEAD_DATA = PAGE * 64  # 8192 B
HEAD_SCALE = PAGE * 8  # 1024 B
HEAD_SIDE = HEAD_DATA + HEAD_SCALE  # 9216 B, also an index page
GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
ONE_SIXTH = torch.tensor(1 / 6, dtype=torch.float32)


def ref_quantize(x: torch.Tensor, gs: float):
    """x fp32 [N, 128] -> (codes int64 [N, 128], e4m3 scale bytes uint8 [N, 8])."""
    g = x.view(-1, 8, 16)
    gs_t = torch.tensor(gs, dtype=torch.float32)
    sf = (g.abs().amax(-1) * ONE_SIXTH.to(x.device) * (1.0 / gs_t)).clamp(max=448)
    sf8 = sf.cpu().to(torch.float8_e4m3fn).to(x.device)  # RNE, in range
    sv = sf8.float()
    quant = torch.where(sv != 0, 1.0 / (sv * gs_t), torch.zeros((), device=x.device))
    y = g * quant[..., None]
    dist = (y.abs()[..., None] - GRID.to(x.device)).abs()
    is_min = dist == dist.amin(-1, keepdim=True)
    ids = torch.arange(8, device=x.device)
    even = torch.where(is_min & (ids % 2 == 0), ids, 8).amin(-1)
    mag = torch.where(even < 8, even, dist.argmin(-1))
    codes = mag | (torch.signbit(y).long() << 3)
    return codes.reshape(-1, 128), sf8.view(torch.uint8)


def k_scale_off(pos: torch.Tensor, g: torch.Tensor):
    r = pos % 32
    n = 4 * (r // 8) + r % 4
    return ((((pos // 32) * 4 + g // 2) * 16 + n) * 2 + (r % 8) // 4) * 2 + g % 2


def ref_pages(pages: torch.Tensor, rows: torch.Tensor, slots, side, nkv, gs):
    """Write rows [T, nkv, 128] (fp32) of side 0 (K) / 1 (V) into uint8 pages [P, 2*nkv*9216]
    (in place) the way the page layout says; slots < 0 skip."""
    ok = (slots >= 0).nonzero().flatten()
    t = rows[ok]
    blk, pos = slots[ok] // PAGE, slots[ok] % PAGE
    flat, stride = pages.view(-1), pages.stride(0)
    codes, sb = ref_quantize(t.reshape(-1, HEAD_DIM), gs)
    codes = codes.view(-1, nkv, HEAD_DIM)
    sb = sb.view(-1, nkv, 8).long()
    base = blk * stride
    for h in range(nkv):
        data = side * nkv * HEAD_SIDE + h * HEAD_DATA
        scale = side * nkv * HEAD_SIDE + nkv * HEAD_DATA + h * HEAD_SCALE
        g = torch.arange(8, device=pages.device)
        if side == 0:
            c = codes[:, h].view(-1, 64, 2)
            byte = (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)
            offs = base[:, None] + data + pos[:, None] * 64 + torch.arange(64)
            flat[offs] = byte
            offs = base[:, None] + scale + k_scale_off(pos[:, None], g)
        else:
            d = torch.arange(HEAD_DIM, device=pages.device)
            for j in range(8):  # tokens sharing a byte: one phase of pos % 8 at a time
                m = (pos % 8) == j
                if not m.any():
                    continue
                off = (
                    base[m][:, None]
                    + data
                    + ((pos[m] // 8)[:, None] * 128 + d) * 4
                    + j // 2
                )
                shift = 4 * (j % 2)
                old = flat[off].long() & ~(0xF << shift)
                flat[off] = (old | (codes[m, h] << shift)).to(torch.uint8)
            offs = base[:, None] + scale + g * PAGE + pos[:, None]
        flat[offs] = sb[:, h].to(torch.uint8)


def ref_index_pages(pages: torch.Tensor, rows: torch.Tensor, slots):
    ok = (slots >= 0).nonzero().flatten()
    blk, pos = slots[ok] // PAGE, slots[ok] % PAGE
    codes, sb = ref_quantize(rows[ok], 1.0)
    c = codes.view(-1, 64, 2)
    byte = (c[..., 0] | (c[..., 1] << 4)).to(torch.uint8)
    flat, stride = pages.view(-1), pages.stride(0)
    flat[blk[:, None] * stride + pos[:, None] * 64 + torch.arange(64)] = byte
    flat[blk[:, None] * stride + HEAD_DATA + pos[:, None] * 8 + torch.arange(8)] = sb


def make_slots(mode: str, tokens: int, pages: int):
    if mode == "scatter":  # decode-like: one token per request, any page
        s = torch.randperm(pages * PAGE, dtype=torch.int64)[:tokens]
    elif mode == "seq":  # prefill chunk from the start of a page: whole octets
        s = torch.arange(tokens, dtype=torch.int64)
    elif mode == "seq_off3":  # chunk starting mid-octet: no whole octet, shared words
        s = torch.arange(tokens, dtype=torch.int64) + 3
    elif mode == "seq_gaps":  # octets with a skipped token and a hole, a run on page 1
        s = torch.arange(tokens, dtype=torch.int64) + PAGE
        s[9:] += 1  # a hole: slot 9 + PAGE stays empty, the octets after it are shifted
        s[5::37] = -1
    elif mode == "mtp":  # 4 consecutive tokens per request: two requests per octet
        r = torch.randperm(pages * PAGE // 4, dtype=torch.int64)[: -(-tokens // 4)]
        s = (r[:, None] * 4 + torch.arange(4)).flatten()[:tokens]
    else:
        raise ValueError(mode)
    return s.cuda()


def stress_qkv(qkv, nq, nkv, niq):
    """Rows the plain random case does not reach: V is raw, so set it directly."""
    v0 = (nq + nkv) * HEAD_DIM
    v = qkv[:, v0 : v0 + nkv * HEAD_DIM].view(-1, nkv, 8, 16)
    n = v.shape[0]
    if n < 8:
        return
    mid = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0] * 2)
    v[0] = 0  # zero rows
    v[1, :, 3] = 0  # a zero group in a live row
    v[2] = mid * 0.5  # e2m1 midpoints / ties, scale 0.5
    v[3] = -mid * 3.0  # negative, scale 3
    v[4, :, 1] = 1e4 * torch.randn(nkv, 16)  # scale clamps at 448, codes saturate
    v[4, :, 2] = 1e-30  # underflow to e4m3 0
    v[5, :, 4] = torch.tensor(
        [-0.0] * 8 + [0.0] * 8
    )  # signed zeros next to a live group
    v[5, :, 4, 0] = 1.0
    v[6, :, 0] = 1e-3 * mid  # e4m3 subnormal scales
    v[7] = (torch.arange(16) - 7.5) * 0.7


@benchmark()
def test_nvfp4_write(
    num_tokens, num_kv_heads, slot_mode, gs, perf=False, variant="base"
):
    """variant: base; padded (KV page stride 64 B above the page size, the pad bytes must stay);
    skip (skip_index_branch with index_cache_dtype "auto": index operands untouched);
    skip_bare (the same with no index operands passed)."""
    skip, bare = variant.startswith("skip"), variant == "skip_bare"
    nq, niq = 16, 1  # M3.1 TP4
    dtype = torch.bfloat16
    torch.manual_seed(7)
    eps, rotary_dim, max_pos = 1e-6, 64, 4096
    qn, kn, iqn, ikn = (torch.randn(HEAD_DIM, dtype=dtype) * 0.1 for _ in range(4))
    inv = 1.0 / (
        5e6 ** (torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim)
    )
    fr = torch.einsum("i,j->ij", torch.arange(max_pos, dtype=torch.float32), inv)
    cos_sin = torch.cat((fr.cos(), fr.sin()), -1).to(dtype)
    pos = torch.randint(0, max_pos, (num_tokens,), dtype=torch.int64)
    row = (nq + 2 * num_kv_heads + niq + 1) * HEAD_DIM
    qkv = torch.randn(num_tokens, row, dtype=dtype)
    stress_qkv(qkv, nq, num_kv_heads, niq)
    # a zero row and a half-zero row of K / index-K (the norm keeps them zero)
    for c0 in (nq * HEAD_DIM, row - HEAD_DIM):
        qkv[0, c0 : c0 + HEAD_DIM] = 0
        if num_tokens > 1:
            qkv[1, c0 : c0 + HEAD_DIM : 2] = 0

    pages = -(-num_tokens // PAGE) + 3
    slots = make_slots(slot_mode, num_tokens, pages)
    pages = max(pages, int(slots.max()) // PAGE + 1)
    ident = torch.arange(pages * PAGE, dtype=torch.int64)[:num_tokens]

    def call(qkv_, kv_pages, idx_pages, q_out, iq_out, ks, vs, **kw):
        iw, islots = ((None, None), None) if bare else ((iqn, ikn), slots)
        if bare:
            idx_pages = iq_out = None
        aiter.fused_qknorm_idxrqknorm(
            qkv_, qn, kn, cos_sin, pos, nq, num_kv_heads, rotary_dim, eps, *iw,
            niq, slots, kv_pages, kv_pages, idx_pages, PAGE, q_out, iq_out, islots,
            k_scale=ks, v_scale=vs, **kw,
        )  # fmt: skip

    # bf16 rows of what the op computes: "auto" write into a one-slot-per-token scratch
    sk = torch.empty(pages, PAGE, num_kv_heads, HEAD_DIM, dtype=dtype)
    sv, sik = torch.empty_like(sk), torch.empty(pages * PAGE, HEAD_DIM, dtype=dtype)
    q_ref = torch.empty(num_tokens, nq * HEAD_DIM, dtype=dtype)
    iq_ref = torch.empty(num_tokens, niq * HEAD_DIM, dtype=torch.float8_e4m3fn)
    aiter.fused_qknorm_idxrqknorm(
        qkv.clone(), qn, kn, cos_sin, pos, nq, num_kv_heads, rotary_dim, eps, iqn, ikn,
        niq, ident, sk, sv, sik, PAGE, q_ref, iq_ref, ident, index_cache_dtype="auto",
    )  # fmt: skip
    k_rows = sk.view(-1, num_kv_heads, HEAD_DIM)[:num_tokens].float().cpu()
    v_rows = sv.view(-1, num_kv_heads, HEAD_DIM)[:num_tokens].float().cpu()
    i_rows = sik[:num_tokens].float().cpu()

    g = torch.Generator().manual_seed(11)
    pad = 64 if variant == "padded" else 0
    kv_garbage = torch.randint(
        0,
        256,
        (pages, 2 * num_kv_heads * HEAD_SIDE + pad),
        dtype=torch.uint8,
        generator=g,
        device="cpu",
    )
    idx_garbage = torch.randint(
        0, 256, (pages, HEAD_SIDE), dtype=torch.uint8, generator=g, device="cpu"
    )
    ks, vs = torch.tensor([gs]), torch.tensor([gs * 0.5])
    kv_full, idx_pages = kv_garbage.clone().cuda(), idx_garbage.clone().cuda()
    kv_pages = kv_full[:, : 2 * num_kv_heads * HEAD_SIDE]  # a narrowed view when padded
    q_out = torch.empty_like(q_ref)
    iq_out = torch.empty_like(iq_ref)
    if skip:  # the index operands must come back unchanged
        iq_out.fill_(1.0)
        iq_ref = iq_out.clone()
        i_kw = {"skip_index_branch": True, "index_cache_dtype": "auto"}
    else:
        i_kw = {"index_cache_dtype": "nvfp4"}
    call(qkv.clone(), kv_pages, idx_pages, q_out, iq_out, ks, vs,
         kv_cache_dtype="nvfp4", **i_kw)  # fmt: skip

    want_kv, want_idx = kv_garbage.clone().cuda(), idx_garbage.clone().cuda()
    cs = slots.cpu()
    for side, (rows, sc) in enumerate(((k_rows, gs), (v_rows, gs * 0.5))):
        ref_pages(want_kv, rows.cuda(), cs.cuda(), side, num_kv_heads, sc)
    if not skip:
        ref_index_pages(want_idx, i_rows.cuda(), cs.cuda())

    ret = {
        "kv err": exact(kv_full, want_kv, "kv pages: "),
        "idx err": exact(idx_pages, want_idx, "index pages: "),
        "q err": max(
            exact(q_out, q_ref, "q_out: "), exact(iq_out, iq_ref, "index q_out: ")
        ),
    }
    if perf:
        _, us = run_perftest(
            call,
            qkv,
            kv_pages,
            idx_pages,
            q_out,
            iq_out,
            ks,
            vs,
            kv_cache_dtype="nvfp4",
            **i_kw,
            num_iters=50,
            num_warmup=20,
        )
        ret["us"] = us
        ret["TB/s"] = write_bytes(qkv, q_out, iq_out, "nvfp4") / us / 1e6
    return ret


# cache bytes written per token for K + V + index-K (nkv 1), by write mode
CACHE_BYTES_PER_TOKEN = {
    "nvfp4": 3 * HEAD_SIDE // PAGE,  # 216: 2 x 72 B (K, V) + 72 B (index)
    "mxfp4": 3 * (HEAD_DATA + PAGE * 4) // PAGE,  # 204: 3 x 68 B
    "fp8": 2 * HEAD_DIM
    + 2 * 4
    + (HEAD_DATA + PAGE * 4) // PAGE,  # K, V e4m3 + 2 scales
}


def write_bytes(qkv, q_out, iq_out, mode):
    """Nominal HBM traffic of one call: qkv read, q / index-q written, cache rows written."""
    return (
        qkv.nbytes
        + q_out.nbytes
        + iq_out.nbytes
        + (qkv.shape[0] * CACHE_BYTES_PER_TOKEN[mode])
    )


@benchmark()
def test_write_modes(num_tokens):
    """us and TB/s of the nvfp4, mxfp4 and fp8 write modes of the one kernel (M3.1 TP4 shapes,
    nkv 1: scattered slots up to 64 tokens, a sequential prefill chunk above). The nvfp4 row
    also checks the bytes."""
    slot_mode = "scatter" if num_tokens <= 64 else "seq"
    ret = test_nvfp4_write(num_tokens, 1, slot_mode, 1.0, perf=True)
    ret["nvfp4 us"], ret["nvfp4 TB/s"] = ret.pop("us"), ret.pop("TB/s")
    nq, nkv, niq, rotary_dim = 16, 1, 1, 64
    dtype = torch.bfloat16
    w = [torch.randn(HEAD_DIM, dtype=dtype) * 0.1 for _ in range(4)]
    cs = torch.randn(4096, rotary_dim, dtype=dtype)
    pos = torch.randint(0, 4096, (num_tokens,), dtype=torch.int64)
    qkv = torch.randn(num_tokens, (nq + 2 * nkv + niq + 1) * HEAD_DIM, dtype=dtype)
    pages = -(-num_tokens // PAGE) + 2
    slots = make_slots(slot_mode, num_tokens, pages)
    q_out = torch.empty(num_tokens, nq * HEAD_DIM, dtype=dtype)
    iq_out = torch.empty(num_tokens, niq * HEAD_DIM, dtype=torch.float8_e4m3fn)
    mx_page = HEAD_DATA + PAGE * 4  # 8704 B: [128 x 64 B e2m1][128 x 4 B e8m0]
    mx_idx = torch.zeros(pages, mx_page, dtype=torch.uint8)

    def call(kc, vc, idx, block, **kw):
        aiter.fused_qknorm_idxrqknorm(
            qkv, w[0], w[1], cs, pos, nq, nkv, rotary_dim, 1e-6, w[2], w[3], niq,
            slots, kc, vc, idx, block, q_out, iq_out, slots, **kw,
        )  # fmt: skip

    mx_kv = torch.zeros(2, pages, nkv, mx_page, dtype=torch.uint8)
    _, us = run_perftest(
        call, mx_kv[0], mx_kv[1], mx_idx, PAGE,
        kv_cache_dtype="mxfp4", index_cache_dtype="mxfp4",
        num_iters=50, num_warmup=20,
    )  # fmt: skip
    ret["mxfp4 us"], ret["mxfp4 TB/s"] = (
        us,
        write_bytes(qkv, q_out, iq_out, "mxfp4") / us / 1e6,
    )
    # fp8 per-token K/V in SHUFFLE pages of 16 tokens, mxfp4 index
    nb = pages * PAGE // 16
    kc = torch.zeros(nb, nkv, HEAD_DIM // 16, 16, 16, dtype=torch.float8_e4m3fn)
    vc = torch.zeros(nb, nkv, 1, HEAD_DIM, 16, dtype=torch.float8_e4m3fn)
    ksc, vsc = torch.zeros(nb, nkv, 16), torch.zeros(nb, nkv, 16)
    _, us = run_perftest(
        call, kc, vc, mx_idx, 16,
        kv_cache_dtype="fp8", index_cache_dtype="mxfp4", k_scale=ksc, v_scale=vsc,
        asm_layout=True, num_iters=50, num_warmup=20,
    )  # fmt: skip
    ret["fp8 us"], ret["fp8 TB/s"] = (
        us,
        write_bytes(qkv, q_out, iq_out, "fp8") / us / 1e6,
    )
    return ret


REJECT = {  # name -> (kwargs of the op call, message of the refusal); a failed check ends the process
    "good": ({}, None),
    "no index nvfp4": (
        {"index_cache_dtype": "auto"},
        "go together",
    ),
    "index nvfp4 only, kv auto": (
        {"kv_cache_dtype": "auto"},
        "go together",
    ),
    "index nvfp4 only, kv mxfp4": (
        {"kv_cache_dtype": "mxfp4"},
        "go together",
    ),
    "kv nvfp4, index mxfp4": (
        {"index_cache_dtype": "mxfp4"},
        "go together",
    ),
    "index not whole pages": (
        {"idx_bytes": HEAD_SIDE + 128},
        "whole 9216 B pages",
    ),
    "index fewer pages than kv": (
        {"idx_bytes": HEAD_SIDE},
        "index_cache must contain at least num_blocks * block_size rows",
    ),
    "no scales": (
        {"k_scale": None, "v_scale": None},
        "takes the fp32 global k_scale/v_scale",
    ),
    "scale [2]": ({"k_scale": torch.ones(2)}, "one element each"),
    "block 16": ({"block": 16}, "block_size must be 128"),
    "aliased pages": ({"alias": True}, "page stride must be at least the page size"),
}


def run_reject(name):
    """One op call of a valid nvfp4 configuration with the changes of REJECT[name]."""
    if name in ROWS_REJECT:
        run_rows_reject(name)
        return
    nkv, n = 1, 4
    qkv = torch.randn(n, (4 + 2 * nkv + 2) * HEAD_DIM, dtype=torch.bfloat16)
    w = torch.zeros(HEAD_DIM, dtype=torch.bfloat16)
    cs = torch.zeros(8, 64, dtype=torch.bfloat16)
    pos = torch.zeros(n, dtype=torch.int64)
    slots = torch.arange(n, dtype=torch.int64)
    kv = torch.zeros(2, 2 * nkv * HEAD_SIDE, dtype=torch.uint8)
    kw = {
        "kv_cache_dtype": "nvfp4",
        "index_cache_dtype": "nvfp4",
        "k_scale": torch.ones(1),
        "v_scale": torch.ones(1),
    }
    kw.update(REJECT[name][0])
    block = kw.pop("block", PAGE)
    idx = torch.zeros(kw.pop("idx_bytes", 2 * HEAD_SIDE), dtype=torch.uint8)
    if kw.pop("alias", False):
        kv = kv[:1].expand(2, -1)  # page stride 0: both pages are the same bytes
    aiter.fused_qknorm_idxrqknorm(
        qkv, w, w, cs, pos, 4, nkv, 64, 1e-6, w, w, 1, slots, kv, kv, idx, block,
        None, None, slots, **kw,
    )  # fmt: skip
    torch.cuda.synchronize()


def test_reject():
    """Each contract violation is refused with its own message; the failed check ends the
    process, so every case runs in a child."""
    for name, (_, msg) in REJECT.items():
        r = subprocess.run(
            [sys.executable, __file__, "--reject", name],
            capture_output=True,
            text=True,
            check=False,
        )
        if msg is None:
            assert r.returncode == 0, r.stderr[-500:]
        else:
            assert r.returncode != 0 and msg in r.stderr, f"{name}: {r.stderr[-500:]}"
    return "ok"


# ---- row layout ----------------------------------------------------------------------------
ROWS_GFX = ("gfx950", "gfx1250")
ROW_PAYLOAD = HEAD_DIM // 2  # 64 B of e2m1 per token row
ROW_SCALES = HEAD_DIM // 16  # 8 E4M3 scale bytes per token row


def ref_rows(x: torch.Tensor):
    """Model-dtype rows [..., 128] -> (payload uint8 [..., 64], scale bytes uint8 [..., 8]):
    scale = e4m3(amax / 6) (RNE, at most 448), codes = e2m1(x * (1 / scale)) half to even,
    its sign included (a negative element of a zero-scale block codes as +0, as -0.0 does).
    On the CPU, in IEEE arithmetic: x * (1 / scale) may be an fp32 denormal."""
    blocks = x.float().cpu().unflatten(-1, (-1, 16))
    scale = (blocks.abs().amax(-1) / 6.0).clamp(max=448.0).to(torch.float8_e4m3fn)
    s = scale.float()
    inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
    y = blocks * inv[..., None]
    a = y.abs()
    mag = sum((a > t).int() for t in (0.25, 1.25, 2.5, 5.0)) + sum(
        (a >= t).int() for t in (0.75, 1.75, 3.5)
    )
    codes = (mag | torch.where(y < 0, 8, 0)).flatten(-2).unflatten(-1, (-1, 2))
    payload = (codes[..., 0] | (codes[..., 1] << 4)).to(torch.uint8)
    return payload.cuda(), scale.view(torch.uint8).cuda()


def pool_fields(pages: int, *shapes, seed: int = 5, odd: bool = False):
    """Fields of one entry-major uint8 pool of random bytes (one entry a page, 64 B of pad at
    its end), the strided views a KV pool hands out: (pool, [field [pages, *shape], ...]).
    odd: one more byte before the fields and after each (page, head)'s rows, so that every
    field has an odd page stride, an odd head stride where it has more than one head, and
    every payload an odd base.
    """
    rows = [math.prod(s[-2:]) + odd for s in shapes]  # bytes a (page, head) takes
    sizes = [math.prod(s[:-2]) * r for s, r in zip(shapes, rows)]
    gen = torch.Generator().manual_seed(seed)
    pool = torch.randint(
        0,
        256,
        (pages, odd + sum(sizes) + 64),
        dtype=torch.uint8,
        generator=gen,
        device="cpu",
    ).cuda()
    offs = list(itertools.accumulate(sizes, initial=int(odd)))
    return pool, [
        pool[:, o : o + n]
        .unflatten(1, (*s[:-2], r))[..., : r - odd]
        .unflatten(-1, s[-2:])
        for o, n, s, r in zip(offs, sizes, shapes, rows)
    ]


def rows_layer(num_tokens: int, nkv: int, dtype):
    """Op operands of an M3-shaped sparse layer (16 q heads and one index head per kv head)
    whose rows spread over E4M3's range, with the V edge rows of stress_qkv (ties, clamps,
    subnormal scales, signed zeros), V blocks of either sign that quantize to a zero scale, V
    elements whose x / scale is an fp32 denormal, and every 5th index key near zero (zero-
    and subnormal-scale blocks)."""
    torch.manual_seed(num_tokens + nkv)
    nq, niq, rotary_dim = 16 * nkv, 1, 64
    heads = nq + 2 * nkv + niq + 1
    mags = torch.exp2(torch.randint(-8, 6, (num_tokens, heads, 1)).float())
    qkv = (torch.randn(num_tokens, heads, HEAD_DIM) * mags).to(dtype)
    qkv[2::5, -1] *= 1e-4  # the index key
    qkv = qkv.view(num_tokens, heads * HEAD_DIM)
    stress_qkv(qkv, nq, nkv, niq)
    v0 = (nq + nkv) * HEAD_DIM
    qkv[0, v0 : v0 + HEAD_DIM] = 0  # an all-zero V row
    v = qkv[:, v0 : v0 + HEAD_DIM].view(num_tokens, 8, 16)
    v[8::7, 5] = (torch.randn(16) * 1e-4).to(dtype)  # scale 0: every code is +0
    v[9::7, 6] = -torch.finfo(dtype).tiny  # x / scale underflows, the code is -0
    v[9::7, 6, 0] = 2000.0
    w = [(torch.randn(HEAD_DIM) * 0.5 + 1).to(dtype) for _ in range(4)]
    cos_sin = torch.randn(4096, rotary_dim).to(dtype)
    pos = torch.randint(0, 4096, (num_tokens,), dtype=torch.int64)
    return {"qkv": qkv, "w": w, "cos_sin": cos_sin, "pos": pos, "nq": nq, "nkv": nkv,
            "niq": niq, "rot": rotary_dim}  # fmt: skip


def rows_args(layer, slots, k, v, index, q_out, iq_out, qkv=None):
    """Positional operands of one op call; qkv (normed in place) defaults to a fresh copy."""
    w = layer["w"]
    qkv = layer["qkv"].clone() if qkv is None else qkv
    return [
        qkv, w[0], w[1], layer["cos_sin"], layer["pos"], layer["nq"], layer["nkv"],
        layer["rot"], 1e-6, w[2], w[3], layer["niq"], slots, k, v, index, PAGE, q_out,
        iq_out, slots,
    ]  # fmt: skip


def rows_call(*args, qkv=None, **kw):
    aiter.fused_qknorm_idxrqknorm(*rows_args(*args, qkv=qkv), asm_layout=False, **kw)


def exact(a, b, msg):
    return checkAllclose(
        a.view(torch.uint8).float(), b.view(torch.uint8).float(), 0, 0, msg=msg
    )


@benchmark()
def test_nvfp4_rows(
    num_tokens, num_kv_heads, dtype, index_dtype, scale_dtype, skip, odd=False
):
    """The row-layout K/V write against "auto" + ref_rows, byte for byte. index_dtype is the
    index cache: "fp8" / "auto" (unused with skip, skip_index_branch), or "nvfp4", the index
    rows (index_scale) against "auto" + ref_rows. scale_dtype is the view
    (uint8 / float8_e4m3fn) the scale planes are passed as. All row fields are strided views
    of one pool of random bytes, compared whole (bytes no live token owns must stay); odd
    gives the payloads odd bases and every field odd strides (pool_fields), which the
    16-bit payload stores must take too. Slots are scattered, every 17th token a pad (-1)
    that must write nothing."""
    nkv, idx_rows = num_kv_heads, index_dtype == "nvfp4"
    layer = rows_layer(num_tokens, nkv, dtype)
    nq, niq = layer["nq"], layer["niq"]
    pages = -(-num_tokens // PAGE) + 3
    slots = torch.randperm(pages * PAGE)[:num_tokens].to(torch.int64)
    slots[1::17] = -1
    idx_t = dtypes.fp8 if index_dtype == "fp8" else dtype
    skip_kw = {"skip_index_branch": skip}

    # The unfused reference: "auto" K/V (and index rows) into page-128 caches.
    ref_k = torch.zeros(pages, PAGE, nkv, HEAD_DIM, dtype=dtype)
    ref_v = torch.zeros_like(ref_k)
    ref_index = torch.zeros(pages * PAGE, HEAD_DIM, dtype=idx_t)
    ref_q = torch.empty(num_tokens, nq * HEAD_DIM, dtype=dtype)
    ref_iq = torch.full((num_tokens, niq * HEAD_DIM), 1.0, dtype=dtype)
    rows_call(layer, slots, ref_k, ref_v, ref_index, ref_q, ref_iq, kv_cache_dtype="auto",
              index_cache_dtype="auto" if idx_rows else index_dtype, **skip_kw)  # fmt: skip

    kv = [(nkv, PAGE, w) for w in (ROW_PAYLOAD, ROW_SCALES)] * 2
    ix = [(PAGE, ROW_PAYLOAD), (PAGE, ROW_SCALES)] if idx_rows else []
    pool, (kp, ks, vp, vs, *ifields) = pool_fields(pages, *kv, *ix, odd=odd)
    before = pool.clone()
    sdt = {"uint8": torch.uint8, "e4m3": torch.float8_e4m3fn}[scale_dtype]
    kw = {"kv_cache_dtype": "nvfp4", "k_scale": ks.view(sdt), "v_scale": vs.view(sdt),
          "index_cache_dtype": index_dtype, **skip_kw}  # fmt: skip
    if idx_rows:
        index, kw["index_scale"] = ifields[0], ifields[1].view(sdt)
    else:
        index = torch.zeros_like(ref_index)
    q_out = torch.empty_like(ref_q)
    iq_out = torch.full_like(ref_iq, 1.0)
    rows_call(layer, slots, kp, vp, index, q_out, iq_out, **kw)

    want = before.clone()

    def want_of(t):
        return want.as_strided(t.shape, t.stride(), t.storage_offset())

    live = slots >= 0
    page, pos = slots[live] // PAGE, slots[live] % PAGE
    for cache, payload, scales in ((ref_k, kp, ks), (ref_v, vp, vs)):
        want_p, want_s = ref_rows(cache[page, pos])  # [live tokens, nkv, ...]
        want_of(payload)[page, :, pos] = want_p
        want_of(scales)[page, :, pos] = want_s
    if idx_rows:
        want_p, want_s = ref_rows(ref_index[slots[live]])
        want_of(ifields[0])[page, pos] = want_p
        want_of(ifields[1])[page, pos] = want_s
        idx_err = max(exact(f, want_of(f), "index rows: ") for f in ifields)
    else:
        idx_err = exact(index, ref_index, "index cache: ")
    ret = {
        "gfx": get_gfx_runtime(),
        # the whole pool: payload and scale rows, and every byte no live token owns
        "kv err": exact(pool, want, "row-layout pool: "),
        "idx err": idx_err,
        "q err": max(exact(q_out, ref_q, "q_out: "), exact(iq_out, ref_iq, "iq_out: ")),
    }
    # timed on one qkv buffer (no copy in the loop); the checks above are done
    _, us = run_perftest(
        rows_call, layer, slots, kp, vp, index, q_out, iq_out, qkv=layer["qkv"], **kw,
        num_iters=50, num_warmup=20,
    )  # fmt: skip
    idx_row = ROW_PAYLOAD + ROW_SCALES if idx_rows else HEAD_DIM * index.element_size()
    nbytes = (
        layer["qkv"].nbytes
        + q_out.nbytes
        + (0 if skip else iq_out.nbytes + num_tokens * idx_row)
        + num_tokens * 2 * nkv * (ROW_PAYLOAD + ROW_SCALES)
    )
    ret["us"], ret["TB/s"] = us, nbytes / us / 1e6
    return ret


def rows_reject_operands(index_rows: bool = False):
    """args, kwargs of a valid row-layout call (fp8 index, two pages, one kv head), or with the
    NVFP4 index rows."""
    layer = rows_layer(4, 1, torch.bfloat16)
    slots = torch.arange(4, dtype=torch.int64)
    kv = [(1, PAGE, w) for w in (ROW_PAYLOAD, ROW_SCALES)] * 2
    ix = [(PAGE, ROW_PAYLOAD), (PAGE, ROW_SCALES)]
    _, (kp, ks, vp, vs, ip, isc) = pool_fields(2, *kv, *ix, seed=1)
    index = ip if index_rows else torch.zeros(2 * PAGE, HEAD_DIM, dtype=dtypes.fp8)
    q_out = torch.empty(4, layer["nq"] * HEAD_DIM, dtype=torch.bfloat16)
    iq_out = torch.empty(4, HEAD_DIM, dtype=torch.bfloat16)
    args = rows_args(layer, slots, kp, vp, index, q_out, iq_out)
    kw = {"kv_cache_dtype": "nvfp4", "index_cache_dtype": "fp8", "k_scale": ks,
          "v_scale": vs, "asm_layout": False}  # fmt: skip
    if index_rows:
        kw.update(index_cache_dtype="nvfp4", index_scale=isc)
    return args, kw


# name -> (edit of the valid call's (args, kwargs), message of the refusal); the "good" ones
# show the harness is sound; "index rows" cases start from the call with NVFP4 index rows.
# args[13] is kv_cache_k, args[15] index_cache, args[16] block_size. The four tokens write
# page 0 only, so the aliased views (pages expanded from one) would take in-bounds stores.
ROWS_REJECT = {
    "rows good": (lambda a, kw: None, None),
    "rows scale strides": (
        lambda a, kw: kw.update(v_scale=kw["v_scale"].contiguous()),
        "equal strides",
    ),
    "rows v_scale fp32": (
        lambda a, kw: kw.update(v_scale=torch.ones(1)),
        "must both be byte planes",
    ),
    "rows block 64": (lambda a, kw: a.__setitem__(16, 64), "must be 128"),
    "rows payload 72 B": (
        lambda a, kw: a.__setitem__(13, torch.zeros(2, 1, PAGE, 72, dtype=torch.uint8)),
        "[num_pages, num_kv_heads, 128, 64]",
    ),
    "rows payload aliased": (
        lambda a, kw: a.__setitem__(13, a[13][:1].expand(2, -1, -1, -1)),
        "non-overlapping",
    ),
    "rows scales aliased": (  # both planes: their strides stay equal
        lambda a, kw: kw.update(
            {n: kw[n][:1].expand(2, -1, -1, -1) for n in ("k_scale", "v_scale")}
        ),
        "non-overlapping",
    ),
    "rows fp4 index": (
        lambda a, kw: kw.update(index_cache_dtype="nvfp4"),
        "row-layout nvfp4 (index_scale) index cache",
    ),
    "rows mxfp4 index": (
        lambda a, kw: kw.update(index_cache_dtype="mxfp4"),
        "row-layout nvfp4 (index_scale) index cache",
    ),
    "rows index_scale fp8 index": (
        lambda a, kw: kw.update(index_scale=kw["k_scale"][:, 0]),
        "requires index_cache_dtype='nvfp4'",
    ),
    "index rows good": (lambda a, kw: None, None),
    "index rows scale 4 B": (
        lambda a, kw: kw.update(index_scale=kw["index_scale"][:, :, :4]),
        "byte [num_pages, 128, 8] planes",
    ),
    "index rows payload aliased": (
        lambda a, kw: a.__setitem__(15, a[15][:1].expand(2, -1, -1)),
        "non-overlapping",
    ),
    "index rows scales aliased": (
        lambda a, kw: kw.update(index_scale=kw["index_scale"][:1].expand(2, -1, -1)),
        "non-overlapping",
    ),
}


def run_rows_reject(name):
    args, kw = rows_reject_operands(index_rows=name.startswith("index rows"))
    ROWS_REJECT[name][0](args, kw)
    aiter.fused_qknorm_idxrqknorm(*args, **kw)
    torch.cuda.synchronize()


def test_rows_reject():
    """Contract violations of the row layout are refused (each in a child: a failed host
    check ends the process), and the wrapper's arch gate tells the layouts apart by the
    scale dtype: byte planes run on gfx950 and gfx1250, fp32 globals on gfx950 only."""
    for name, (_, msg) in ROWS_REJECT.items():
        r = subprocess.run(
            [sys.executable, __file__, "--reject", name],
            capture_output=True,
            text=True,
            check=False,
        )
        if msg is None:
            assert r.returncode == 0, r.stderr[-500:]
        else:
            assert r.returncode != 0 and msg in r.stderr, f"{name}: {r.stderr[-500:]}"
    args, kw = rows_reject_operands()
    page_kw = {**kw, "k_scale": torch.ones(1), "v_scale": torch.ones(1)}
    for gfx, call_kw, msg in (
        ("gfx942", kw, "requires gfx950 or gfx1250"),
        ("gfx1250", page_kw, "require gfx950"),
    ):
        with mock.patch.object(op, "get_gfx_runtime", return_value=gfx):
            try:
                aiter.fused_qknorm_idxrqknorm(*args, **call_kw)
            except RuntimeError as e:
                assert msg in str(e), e
            else:
                raise AssertionError(f"accepted on {gfx}")
    return "ok"


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="fused_qknorm_idxrqknorm nvfp4 pages vs the torch reference, byte for byte",
)
parser.add_argument("-t", "--tokens", type=int, nargs="*", default=[1, 7, 64, 300])
parser.add_argument("-n", "--nkv", type=int, nargs="*", default=[1, 2, 4])
parser.add_argument("--gs", type=float, nargs="*", default=[1.0, 0.37, 3.0])
parser.add_argument(
    "--slots",
    type=str,
    nargs="*",
    default=["scatter", "seq", "seq_off3", "seq_gaps", "mtp"],
)
parser.add_argument(
    "--perf_tokens",
    type=int,
    nargs="*",
    default=[1, 64, 8192, 32768],
    help="token counts of the nvfp4 vs mxfp4 / fp8 write timing table",
)
parser.add_argument(
    "--rows_tokens",
    type=int,
    nargs="*",
    default=[1, 37, 3000],
    help="token counts of the row-layout sweep (with --nkv, --dtype, --index)",
)
parser.add_argument(
    "-d", "--dtype", type=str, nargs="*", default=["bf16", "fp16"], help="row layout"
)
parser.add_argument(
    "--index",
    type=str,
    nargs="*",
    default=["fp8", "auto", "nvfp4"],
    help="index cache dtypes of the row-layout sweep (nvfp4: the index rows)",
)
parser.add_argument(
    "--reject", type=str, default=None, help="(child) run one refusal case"
)


def run_rows(args):
    """The row layout (byte scale planes), on gfx950 and gfx1250."""
    aiter.logger.info("row-layout reject: %s", test_rows_reject())
    scale = {"fp8": "e4m3"}  # the other index dtypes pass uint8 scale planes
    cases = [
        (t, n, d, i, scale.get(i, "uint8"), False)
        for t, n, d, i in itertools.product(
            args.rows_tokens, args.nkv, args.dtype, args.index
        )
    ]
    # the index-less (skip_index_branch) call
    cases += [
        (t, n, "bf16", "auto", "uint8", True)
        for t in args.rows_tokens
        for n in args.nkv
    ]
    # K/V and index rows in fields with odd bases and odd page and head strides
    cases += [
        (t, n, "bf16", "nvfp4", "uint8", False, True)
        for t in args.rows_tokens
        for n in args.nkv
    ]
    df = pd.DataFrame(
        [test_nvfp4_rows(t, n, dtypes.d_dtypes[d], *c) for t, n, d, *c in cases]
    )
    aiter.logger.info(
        "nvfp4 row-layout write vs torch reference (markdown):\n%s",
        df.round(4).to_markdown(index=False),
    )
    bad = df[["kv err", "idx err", "q err"]].to_numpy().sum()
    assert bad == 0, "nvfp4 row-layout K/V, index rows or q differ from the reference"


def main():
    args = parser.parse_args()
    if args.reject:
        run_reject(args.reject)
        return
    # the live GPU, not GPU_ARCHS
    gfx = get_gfx_runtime()
    if gfx in ROWS_GFX:
        run_rows(args)
    # the page layout, the mxfp4 / fp8 timing arms and the e4m3 index q need gfx950
    if gfx != "gfx950":
        aiter.logger.warning(
            "fused_qknorm_idxrqknorm nvfp4 pages are tested on gfx950 only, not %s: skipping",
            gfx,
        )
        return
    aiter.logger.info("reject: %s", test_reject())
    cases = [
        (t, n, s, gs)
        for t, n, s, gs in itertools.product(args.tokens, args.nkv, args.slots, args.gs)
        if not (s in ("seq_gaps", "mtp") and t < 64)
    ]
    # the layouts the sweep above does not run: padded page stride, index-less (skip) layers
    cases += [
        (t, n, s, 0.37, False, v)
        for v in ("padded", "skip", "skip_bare")
        for t, n, s in itertools.product(args.tokens, args.nkv, ("scatter", "seq_off3"))
    ]
    df = pd.DataFrame([test_nvfp4_write(*c) for c in cases])
    aiter.logger.info(
        "nvfp4 write vs torch reference (markdown):\n%s",
        df.round(4).to_markdown(index=False),
    )
    perf = pd.DataFrame([test_write_modes(t) for t in args.perf_tokens])
    perf_cols = ["num_tokens"] + [c for c in perf.columns if c.endswith(("us", "TB/s"))]
    aiter.logger.info(
        "write modes of fused_qknorm_idxrqknorm, us and TB/s (markdown):\n%s",
        perf[perf_cols].round(3).to_markdown(index=False),
    )
    errs = ["kv err", "idx err", "q err"]
    bad = df[errs].to_numpy().sum() + perf[errs].to_numpy().sum()
    assert bad == 0, "nvfp4 pages or q differ from the reference"


if __name__ == "__main__":
    main()
