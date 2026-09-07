# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 paged MQA logits (OPUS, 32x32x64 MFMA) -- correctness and perf on gfx950.

    python3 op_tests/test_pa_mqa_logits_opus.py              # corner cases + both sweeps
    python3 op_tests/test_pa_mqa_logits_opus.py --corner      # corner cases only
    python3 op_tests/test_pa_mqa_logits_opus.py --prefill --bs 1 4 20

WHY THIS TEST IS SHAPED THE WAY IT IS
-------------------------------------
The kernel's two E8M0 scale arrays carry a layout only it can read, and a wrong
permutation of them is SILENT: the byte counts match every other fp4 scale layout, so
the result is plausible-looking wrong logits rather than an error. Two consequences for
this file, both deliberate:

* **Every correctness case uses random data.** Uniform or all-ones inputs pass under any
  permutation of K, because a dot product does not care what order it sums in. A
  K-permutation bug is invisible to them and was historically found only on random data.
* **FlyDSL is scored as a second opinion, not a nice-to-have.** It reads the SAME ``q``
  and ``kv_cache`` bytes (our layout for those is byte-identical to its ABI) but builds
  its scales in its OWN 16x16 layout from the same natural E8M0. So agreement between
  the two is independent evidence that our scale permutation is right -- which the
  dequantized reference alone cannot give, since it shares this file's understanding of
  the layout. Both are checked.

The reference is a dequantize-then-matmul in fp32 over the exact same quantized values
the kernel sees, so it isolates layout and reduction from quantization error: `err` is
expected at ~1e-6 (fp32 accumulation order), not at fp4 resolution.
"""

import argparse
import itertools
import random
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn.functional as F

from aiter.ops.opus.pa_mqa_logits_opus import (
    compute_prefill_windows,
    pa_mqa_logits_mxfp4_decode,
    pa_mqa_logits_mxfp4_prefill,
)
from aiter.test_common import run_perftest

dev = "cuda"

HEADS = 64
HEAD_DIM = 128
KV_BLOCK_SIZE = 64  # page size
SCALE_BLOCK = 32  # E8M0 block
WEIGHT_SCALE = 1.5

# The 32x32x64 scale geometry: a lane's 4 E8M0 bytes live in one aligned dword, indexed
# [.., g(K_CHUNKS), m(MFMA_N), byte]. See the wrapper's module docstring.
K_TILES = HEAD_DIM // 64  # 2  (MFMA_K = 64)
K_CHUNKS = 64 // SCALE_BLOCK  # 2  (32-K chunks per k-tile)
MFMA_N = 32
SCALE_BYTES = 4  # K_TILES * M_TILES == K_TILES * N_TILES == 4
BLOCKS_ROW = HEAD_DIM // SCALE_BLOCK  # 4 natural E8M0 blocks per row

# FlyDSL's own (16x16x128) scale geometry, used only to build its inputs.
FLY_MFMA_M = 16
FLY_KVS_NTPW = 4

PREFILL_BLOCK_KS = (64, 256)
PREFILL_TOTAL_QLEN = 16384
PREFILL_QMIN = 800
DECODE_CTA_TARGET = 1024
N_COS_SAMPLE = 8

FP4_E2M1_MAX = 6.0
_FP4_GRID_VALUES = [
    -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0,
    0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
]  # fmt: skip
_E2M1_LUT = [0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7]
_E2M1_INV_LUT = [7, 8, 9, 10, 11, 12, 13, 14, 7, 6, 5, 4, 3, 2, 1, 0]


# ── MXFP4 quant / dequant ─────────────────────────────────────────────────────
def fp4_quant(x, block_size=SCALE_BLOCK):
    """[..., d] float -> (packed nibbles [..., d/2] uint8, e8m0 [..., d/block] uint8).

    Low nibble = even element, matching the kernel and the FlyDSL fp4 ABI.
    """
    *prefix, d = x.shape
    assert d % block_size == 0
    x_blk = x.float().reshape(*prefix, d // block_size, block_size)
    amax = x_blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    exp_biased = (
        (torch.ceil(torch.log2(amax / FP4_E2M1_MAX)) + 127.0)
        .clamp(0.0, 255.0)
        .to(torch.uint8)
    )
    e8m0 = exp_biased.squeeze(-1).contiguous()
    x_scaled = x_blk / torch.pow(2.0, exp_biased.float() - 127.0)
    grid = torch.tensor(_FP4_GRID_VALUES, dtype=torch.float32, device=x.device)
    idx = (x_scaled.unsqueeze(-1) - grid).abs().argmin(dim=-1)
    lut = torch.tensor(_E2M1_LUT, dtype=torch.uint8, device=x.device)
    nibbles = lut[idx].reshape(*prefix, d)
    packed = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8)
    return packed.contiguous(), e8m0


def fp4_dequant(packed, e8m0, block_size=SCALE_BLOCK):
    *prefix, d_half = packed.shape
    d = d_half * 2
    nibbles = torch.empty(*prefix, d, dtype=torch.uint8, device=packed.device)
    nibbles[..., 0::2] = packed & 0xF
    nibbles[..., 1::2] = (packed >> 4) & 0xF
    inv = torch.tensor(_E2M1_INV_LUT, dtype=torch.long, device=packed.device)
    grid = torch.tensor(_FP4_GRID_VALUES, dtype=torch.float32, device=packed.device)
    vals = grid[inv[nibbles.long()]]
    scale = torch.pow(2.0, e8m0.float() - 127.0)
    return (
        vals.reshape(*prefix, d // block_size, block_size) * scale.unsqueeze(-1)
    ).reshape(*prefix, d)


# ── the two scale layouts, from one natural E8M0 array ────────────────────────
# Natural E8M0 is [rows, BLOCKS_ROW] with block b covering K[32b : 32b+32]. Both
# shuffles below are pure permutations of it; neither drops or duplicates a byte.
def scale_to_opus(e8_nat, rows_per_group):
    """[rows, 4] -> [rows/rows_per_group, K_CHUNKS, MFMA_N, SCALE_BYTES].

    `rows_per_group` is MFMA_N * n_tiles: 64 heads per query row for q_scale, 64 page
    tokens per block for kv_scale. Within a group, row index i splits as
    (tile, m) = divmod(i, MFMA_N) and block b as (kt, g) = divmod(b, K_CHUNKS); the
    byte index is kt * n_tiles + tile.
    """
    n_tiles = rows_per_group // MFMA_N
    groups = e8_nat.shape[0] // rows_per_group
    return (
        e8_nat.reshape(groups, n_tiles, MFMA_N, K_TILES, K_CHUNKS)
        .permute(0, 4, 2, 3, 1)  # [group, g, m, kt, tile]
        .reshape(groups, K_CHUNKS, MFMA_N, SCALE_BYTES)
        .contiguous()
    )


def q_scale_flydsl(e8_nat, total_tokens):
    """[T*H, 4] -> FlyDSL's [T, K_TILES_16, 4, 16, QS_PAD]; K_TILES_16 = D/128 = 1."""
    m_tiles = HEADS // FLY_MFMA_M  # 4
    qs_pad = ((m_tiles + 3) // 4) * 4
    qe = (
        e8_nat.reshape(total_tokens, m_tiles, FLY_MFMA_M, HEAD_DIM // 128, 4)
        .permute(0, 3, 4, 2, 1)  # [T, K_TILES_16, 4, 16, M_TILES]
        .contiguous()
    )
    return F.pad(qe, (0, qs_pad - m_tiles)).contiguous()


def kv_scale_flydsl(e8_nat, num_blocks):
    """[num_blocks*PAGE, 4] -> FlyDSL's [num_blocks, 1, 4, PAGE], sflat=(o%16)*4+(o//16)."""
    o = torch.arange(KV_BLOCK_SIZE, device=e8_nat.device)
    sflat = (o % FLY_MFMA_M) * FLY_KVS_NTPW + (o // FLY_MFMA_M)
    out = torch.zeros(
        num_blocks, 1, 4, KV_BLOCK_SIZE, dtype=torch.uint8, device=e8_nat.device
    )
    # src[blk, b, o] = e8m0(token o of page blk, block b); the advanced index on the
    # last axis sends token o to slot sflat[o], leaving the other axes in place.
    src = e8_nat.reshape(num_blocks, KV_BLOCK_SIZE, 4).permute(0, 2, 1)  # [nb, 4, PAGE]
    out[:, 0, :, sflat] = src
    return out.contiguous()


# ── input builders ────────────────────────────────────────────────────────────
@dataclass
class Inputs:
    q_packed: torch.Tensor  # [T, H, D/2]  natural
    q_scale: torch.Tensor  # [T, 2, 32, 4] opus
    q_scale_fly: torch.Tensor  # FlyDSL layout
    q_dq: torch.Tensor  # [T, H, D]  dequantized, for the reference
    weights: torch.Tensor  # [T, H] bf16 natural
    kv_cache: torch.Tensor  # [num_blocks, 4, PAGE, 16]  FlyDSL/opus shared
    kv_scale: torch.Tensor  # [num_blocks, 2, 32, 4] opus
    kv_scale_fly: torch.Tensor
    kv_dq: torch.Tensor  # [bs, t_max, D] dequantized
    block_tables: torch.Tensor
    max_seq_len: int


def pages_for(max_end, block_k):
    """Pages per sequence, rounded so a CTA never indexes past the table."""
    chunks = max(1, (max_end + block_k - 1) // block_k)
    return max(KV_BLOCK_SIZE // KV_BLOCK_SIZE, chunks * (block_k // KV_BLOCK_SIZE))


def build_inputs(bs, max_end, total_tokens, block_k, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    mbps = pages_for(max_end, block_k)
    t_max = mbps * KV_BLOCK_SIZE
    num_blocks = bs * mbps

    # --- KV: quantize natural rows, then write the two layouts -------------
    kv = torch.randn(bs * t_max, HEAD_DIM, generator=g, device=dev, dtype=torch.float32)
    kv_packed, kv_e8 = fp4_quant(kv)
    kv_dq = fp4_dequant(kv_packed, kv_e8).reshape(bs, t_max, HEAD_DIM)
    # kv_cache[blk, b, o, :] = 16 packed bytes of K[token o][32b:32b+32].
    kv_cache = (
        kv_packed.reshape(num_blocks, KV_BLOCK_SIZE, BLOCKS_ROW, 16)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    kv_scale = scale_to_opus(kv_e8, KV_BLOCK_SIZE)
    kv_scale_fly = kv_scale_flydsl(kv_e8, num_blocks)
    block_tables = (
        torch.arange(num_blocks, dtype=torch.int32, device=dev).reshape(bs, mbps)
    )

    # --- Q + weights ------------------------------------------------------
    q = torch.randn(
        total_tokens * HEADS, HEAD_DIM, generator=g, device=dev, dtype=torch.float32
    )
    q_packed_flat, q_e8 = fp4_quant(q)
    q_dq = fp4_dequant(q_packed_flat, q_e8).reshape(total_tokens, HEADS, HEAD_DIM)
    q_packed = q_packed_flat.reshape(total_tokens, HEADS, HEAD_DIM // 2).contiguous()
    q_scale = scale_to_opus(q_e8, HEADS)
    weights = torch.randn(
        total_tokens, HEADS, generator=g, device=dev, dtype=torch.float32
    ).to(torch.bfloat16)

    return Inputs(
        q_packed=q_packed,
        q_scale=q_scale,
        q_scale_fly=q_scale_flydsl(q_e8, total_tokens),
        q_dq=q_dq,
        weights=weights,
        kv_cache=kv_cache,
        kv_scale=kv_scale,
        kv_scale_fly=kv_scale_fly,
        kv_dq=kv_dq,
        block_tables=block_tables,
        max_seq_len=t_max,
    )


# ── reference ─────────────────────────────────────────────────────────────────
def ref_rows(inp, rows, rb, ls, le):
    """Reference logits for a few sampled rows: {row: (start, end, [values])}."""
    w = inp.weights.float()
    ref = {}
    for r in rows:
        b, s, e = int(rb[r]), int(ls[r]), int(le[r])
        if e <= s:
            ref[r] = (s, e, None)
            continue
        k = inp.kv_dq[b, s:e]  # [n, D]
        scores = torch.relu(inp.q_dq[r] @ k.T)  # [H, n]
        ref[r] = (s, e, (scores * w[r, :, None]).sum(0) * WEIGHT_SCALE)
    return ref


def max_err(out, ref):
    err = 0.0
    for r, (s, e, vals) in ref.items():
        if vals is None:
            continue
        got = out[r, s:e].float()
        err = max(err, (got - vals).abs().max().item() / vals.abs().max().clamp(min=1e-6))
    return err


def oob_is_neginf(out, rb, ls, le):
    """Every cell outside [local_start, local_end) must be left at the -inf pre-fill."""
    col = torch.arange(out.shape[1], device=out.device).unsqueeze(0)
    inside = (col >= ls.unsqueeze(1)) & (col < le.unsqueeze(1))
    return bool(torch.isneginf(out[~inside]).all().item())


def sample_rows(total, le, n=N_COS_SAMPLE, seed=0):
    nonempty = torch.nonzero(le > 0).flatten().tolist()
    if not nonempty:
        return []
    rng = random.Random(seed)
    return sorted(rng.sample(nonempty, min(n, len(nonempty))))


# ── FlyDSL candidates (second opinion on the scale layout) ────────────────────
def flydsl_prefill(inp, rb, ls, le, total_q, block_k):
    try:
        from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4_prefill import (
            compute_prefill_schedule,
            flydsl_pa_mqa_logits_fp4_prefill,
        )
    except Exception:  # noqa: BLE001
        return None
    msl = inp.max_seq_len
    _, cta_info, n_ctas = compute_prefill_schedule(
        rb, ls, le, block_k, total_q, msl
    )
    out = torch.full((total_q, msl), float("-inf"), dtype=torch.float32, device=dev)

    def launch():
        flydsl_pa_mqa_logits_fp4_prefill(
            inp.q_packed, inp.q_scale_fly, inp.kv_cache.view(-1, 1, 4, KV_BLOCK_SIZE, 16),
            inp.kv_scale_fly, inp.block_tables, inp.weights, rb, ls, le, msl,
            weight_scale=WEIGHT_SCALE, block_k=block_k, kv_block_size=KV_BLOCK_SIZE,
            num_warps=4 if block_k == 256 else 1, parallel_unit_num=total_q,
            out=out, cta_info=cta_info, n_ctas=n_ctas,
        )  # fmt: skip
        return out

    return launch


def flydsl_decode(inp, ctx, batch, next_n, block_k):
    """FlyDSL fp4 decode -- a DIFFERENT op from its prefill one, with its own ABI.

    Its q / q_scale are [B, next_n, ...] where ours are packed [B*next_n, ...]; fixed
    MTP packs rows in (b, n) order, so a reshape is the whole conversion. `weights`
    stays packed. It runs a persistent grid from `compute_varctx_schedule` rather than
    our schedule-free 3D grid, and gets it precomputed here so the timed region is a
    pure launch on both sides. Fixed-MTP only -- there is no varqlen decode on that side.
    """
    try:
        from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4
        from aiter.ops.flydsl.kernels.mqa_logits.pa_mqa_logits_fp4 import (
            compute_varctx_schedule,
        )
    except Exception as e:  # noqa: BLE001
        print(f"    [flydsl decode unavailable] {type(e).__name__}: {e}")
        return None

    msl = inp.max_seq_len
    q_nn = inp.q_packed.reshape(batch, next_n, HEADS, HEAD_DIM // 2).contiguous()
    qs_nn = inp.q_scale_fly.reshape(
        batch, next_n, *inp.q_scale_fly.shape[1:]
    ).contiguous()
    _, cta_info, total_ctas = compute_varctx_schedule(
        ctx, block_k, None, msl, next_n=next_n
    )
    out = torch.full(
        (batch * next_n, msl), float("-inf"), dtype=torch.float32, device=dev
    )

    def launch():
        flydsl_pa_mqa_logits_fp4(
            q_nn, qs_nn, inp.kv_cache.view(-1, 1, 4, KV_BLOCK_SIZE, 16),
            inp.kv_scale_fly, inp.block_tables, inp.weights, ctx, msl,
            weight_scale=WEIGHT_SCALE, next_n=next_n, block_k=block_k,
            kv_block_size=KV_BLOCK_SIZE, num_warps=4 if block_k == 256 else 1,
            out=out, cta_info=cta_info, total_ctas=total_ctas,
        )  # fmt: skip
        return out

    try:
        launch()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"    [flydsl decode failed] {type(e).__name__}: {e}")
        return None
    return launch


# ── correctness ───────────────────────────────────────────────────────────────
def check_prefill(bs, windows_per_batch, seed, block_k, label):
    """One ragged-prefill case: explicit per-row (start, end) windows, random data."""
    qlens = [len(w) for w in windows_per_batch]
    total_q = sum(qlens)
    max_end = max(e for w in windows_per_batch for (_, e) in w)
    inp = build_inputs(bs, max_end, total_q, block_k, seed)

    rb, ls, le = [], [], []
    for b, w in enumerate(windows_per_batch):
        for s, e in w:
            rb.append(b)
            ls.append(s)
            le.append(e)
    rb = torch.tensor(rb, dtype=torch.int32, device=dev)
    ls = torch.tensor(ls, dtype=torch.int32, device=dev)
    le = torch.tensor(le, dtype=torch.int32, device=dev)

    out = pa_mqa_logits_mxfp4_prefill(
        inp.q_packed, inp.q_scale, inp.kv_cache, inp.kv_scale, inp.block_tables,
        inp.weights, rb, ls, le, inp.max_seq_len,
        weight_scale=WEIGHT_SCALE, block_k=block_k, kv_block_size=KV_BLOCK_SIZE,
    )  # fmt: skip
    torch.cuda.synchronize()

    rows = sample_rows(total_q, le, seed=seed)
    err = max_err(out, ref_rows(inp, rows, rb, ls, le))
    oob = oob_is_neginf(out, rb, ls, le)

    fly_err = float("nan")
    fly = flydsl_prefill(inp, rb, ls, le, total_q, block_k)
    if fly is not None:
        out_f = fly()
        torch.cuda.synchronize()
        col = torch.arange(inp.max_seq_len, device=dev).unsqueeze(0)
        m = (col >= ls.unsqueeze(1)) & (col < le.unsqueeze(1))
        scale = out[m].abs().max().clamp(min=1e-6)
        fly_err = ((out[m] - out_f[m]).abs().max() / scale).item()

    ok = err < 2e-5 and oob and (fly_err != fly_err or fly_err < 2e-5)
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<34} bk={block_k:3d} "
          f"err={err:.2e} vs_flydsl={fly_err:.2e} oob_neginf={oob}")  # fmt: skip
    return ok


def check_decode(bs, next_n, context_lens, seed, block_k, label):
    """One fixed-MTP decode case; the window is derived in-kernel, so this tests that too."""
    total_q = bs * next_n
    max_end = max(context_lens)
    inp = build_inputs(bs, max_end, total_q, block_k, seed)
    ctx = torch.tensor(context_lens, dtype=torch.int32, device=dev)

    out = pa_mqa_logits_mxfp4_decode(
        inp.q_packed, inp.q_scale, inp.kv_cache, inp.kv_scale, inp.block_tables,
        inp.weights, ctx, inp.max_seq_len, next_n,
        split_ctx_len=inp.max_seq_len, weight_scale=WEIGHT_SCALE,
        block_k=block_k, kv_block_size=KV_BLOCK_SIZE,
    )  # fmt: skip
    torch.cuda.synchronize()

    # The MTP tail-causal rule the kernel derives internally, restated here so a
    # disagreement shows up as a window mismatch rather than silently passing.
    rb, ls, le = [], [], []
    for b in range(bs):
        for n in range(next_n):
            rb.append(b)
            ls.append(0)
            le.append(max(context_lens[b] - (next_n - 1 - n), 0))
    rb = torch.tensor(rb, dtype=torch.int32, device=dev)
    ls = torch.tensor(ls, dtype=torch.int32, device=dev)
    le = torch.tensor(le, dtype=torch.int32, device=dev)

    rows = sample_rows(total_q, le, seed=seed)
    err = max_err(out, ref_rows(inp, rows, rb, ls, le))
    oob = oob_is_neginf(out, rb, ls, le)
    ok = err < 2e-5 and oob
    print(f"  [{'PASS' if ok else 'FAIL'}] {label:<34} bk={block_k:3d} "
          f"err={err:.2e} oob_neginf={oob}")  # fmt: skip
    return ok


def run_corner():
    """Cases chosen to hit the pipeline and window corners, all on random data."""
    print("=" * 78)
    print("[corner] MXFP4 32x32x64 prefill + decode, random data, both block_k")
    print("=" * 78)
    oks = []
    for block_k in PREFILL_BLOCK_KS:
        # ragged windows incl. non-zero starts and non-32-aligned bounds
        oks.append(check_prefill(2, [[(0, 50), (0, 120), (0, 200)], [(0, 40), (0, 100)]],
                                 0, block_k, "ragged, 2 batches"))  # fmt: skip
        oks.append(check_prefill(3, [[(0, 30)], [(0, 200)], [(0, 100), (0, 150)]],
                                 2, block_k, "ragged, 3 batches"))  # fmt: skip
        oks.append(check_prefill(2, [[(10, 50), (64, 200)], [(0, 100), (130, 256)]],
                                 4, block_k, "non-zero lower bounds"))  # fmt: skip
        # tile-boundary +-1 for both block_k, and the 1-/2-tile pipeline corners
        bounds = [(0, n) for n in (1, 63, 64, 65, 127, 128, 129, 191, 192, 193)]
        oks.append(check_prefill(1, [bounds], 5, block_k, "tile boundaries +-1"))
        # empty window (must early-out without storing) and qlen > ctx
        oks.append(check_prefill(2, [[(0, 0), (0, 33)], [(0, 96), (0, 0)]],
                                 6, block_k, "empty windows"))  # fmt: skip
        # MFMA_N=32 alignment: windows that end mid-tile in every residue class
        oks.append(check_prefill(1, [[(0, 32 * 3 + r) for r in range(1, 9)]],
                                 7, block_k, "mid-tile ends"))  # fmt: skip
        # decode: pure decode, MTP, and a context at a tile boundary
        oks.append(check_decode(2, 1, [128, 200], 8, block_k, "decode next_n=1"))
        oks.append(check_decode(3, 4, [256, 129, 64], 9, block_k, "decode MTP next_n=4"))
        oks.append(check_decode(1, 8, [block_k * 2 + 1], 10, block_k, "decode tile+1"))
    print(f"\n  {sum(oks)}/{len(oks)} cases pass")
    return all(oks)


# ── perf ──────────────────────────────────────────────────────────────────────
def gen_prefill_qlens(bs, total=PREFILL_TOTAL_QLEN, qmin=PREFILL_QMIN, seed=0):
    g = random.Random(seed)
    extra = total - bs * qmin
    w = [g.random() for _ in range(bs)]
    s = sum(w) or 1.0
    parts = [qmin + int(extra * wi / s) for wi in w]
    parts[0] += total - sum(parts)
    return parts


def bench_prefill(bs_list, block_ks, iters, warmup):
    rows = []
    for bs, block_k in itertools.product(bs_list, block_ks):
        qlens = gen_prefill_qlens(bs, seed=bs)
        total_q = sum(qlens)
        inp = build_inputs(bs, max(qlens), total_q, block_k, seed=bs)
        cu = torch.tensor(
            [0] + list(itertools.accumulate(qlens)), dtype=torch.int32, device=dev
        )
        ctx = torch.tensor(qlens, dtype=torch.int32, device=dev)
        rb, ls, le = compute_prefill_windows(cu, ctx, total_q)
        out = torch.full(
            (total_q, inp.max_seq_len), float("-inf"), dtype=torch.float32, device=dev
        )

        def ours():
            return pa_mqa_logits_mxfp4_prefill(
                inp.q_packed, inp.q_scale, inp.kv_cache, inp.kv_scale, inp.block_tables,
                inp.weights, rb, ls, le, inp.max_seq_len, weight_scale=WEIGHT_SCALE,
                block_k=block_k, kv_block_size=KV_BLOCK_SIZE, out=out,
            )  # fmt: skip

        _, us = run_perftest(ours, num_iters=iters, num_warmup=warmup)
        n_logits = int((le - ls).clamp(min=0).sum().item())
        row = {
            "bs": bs, "block_k": block_k, "total_q": total_q,
            "max_win": int(le.max()), "n_logits": n_logits,
            "ours us": round(us, 2),
            "TFLOPS": round(2 * HEADS * HEAD_DIM * n_logits / us / 1e6, 1),
        }  # fmt: skip

        fly = flydsl_prefill(inp, rb, ls, le, total_q, block_k)
        if fly is not None:
            _, us_f = run_perftest(fly, num_iters=iters, num_warmup=warmup)
            row["flydsl us"] = round(us_f, 2)
            row["ours/flydsl"] = f"{us / us_f - 1:+.1%}"
        rows.append(row)
        del inp, out
        torch.cuda.empty_cache()
    print("\nPrefill (causal, ctx == qlen; random data)")
    print(pd.DataFrame(rows).to_markdown(index=False))


def bench_decode(shapes, block_ks, iters, warmup):
    rows = []
    for (batch, max_ctx, next_n), block_k in itertools.product(shapes, block_ks):
        g = random.Random(batch + max_ctx + next_n)
        ctxs = [
            ((g.randint(int(0.9 * max_ctx), max_ctx) + KV_BLOCK_SIZE - 1)
             // KV_BLOCK_SIZE) * KV_BLOCK_SIZE
            for _ in range(batch)
        ]  # fmt: skip
        total_q = batch * next_n
        inp = build_inputs(batch, max(ctxs), total_q, block_k, seed=batch + next_n)
        ctx = torch.tensor(ctxs, dtype=torch.int32, device=dev)
        out = torch.full(
            (total_q, inp.max_seq_len), float("-inf"), dtype=torch.float32, device=dev
        )

        def ours():
            return pa_mqa_logits_mxfp4_decode(
                inp.q_packed, inp.q_scale, inp.kv_cache, inp.kv_scale, inp.block_tables,
                inp.weights, ctx, inp.max_seq_len, next_n,
                split_ctx_len=inp.max_seq_len, weight_scale=WEIGHT_SCALE,
                block_k=block_k, kv_block_size=KV_BLOCK_SIZE, out=out,
            )  # fmt: skip

        _, us = run_perftest(ours, num_iters=iters, num_warmup=warmup)
        n_logits = sum(max(c - (next_n - 1 - n), 0) for c in ctxs for n in range(next_n))
        rows.append({
            "batch": batch, "next_n": next_n, "max_ctx": max_ctx, "block_k": block_k,
            "total_q": total_q, "ours us": round(us, 2),
            "TFLOPS": round(2 * HEADS * HEAD_DIM * n_logits / us / 1e6, 1),
        })  # fmt: skip
        del inp, out
        torch.cuda.empty_cache()
    print("\nDecode (fixed MTP; random data)")
    print(pd.DataFrame(rows).to_markdown(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corner", action="store_true", help="corner cases only")
    ap.add_argument("--prefill", action="store_true")
    ap.add_argument("--decode", action="store_true")
    ap.add_argument("--bs", type=int, nargs="*", default=[1, 2, 4, 8, 12, 16, 20])
    ap.add_argument("--block-k", type=int, nargs="*", default=list(PREFILL_BLOCK_KS))
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    a = ap.parse_args()

    only_bench = a.prefill or a.decode
    ok = True
    if not only_bench:
        ok = run_corner()
    if a.corner:
        raise SystemExit(0 if ok else 1)

    if not only_bench or a.prefill:
        bench_prefill(a.bs, a.block_k, a.iters, a.warmup)
    if not only_bench or a.decode:
        bench_decode(
            [(8, 8192, 8), (32, 8192, 4), (128, 8192, 1), (128, 1024, 8)],
            a.block_k, a.iters, a.warmup,
        )  # fmt: skip
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
