#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter op-test for ``flydsl_pa_mqa_logits_fp4_prefill`` and
``flydsl_pa_mqa_logits_fp4_varqlen``.

Validates the ragged-prefill FP4 paged MQA logits kernel (gfx950) against a
pure-torch reference. Each query row owns a seq-local window
``[local_start, local_end)`` into its sequence's paged FP4 KV cache, read
straight from ``block_tables`` (no cp_gather staging).

The variable-qlen path (``flydsl_pa_mqa_logits_fp4_varqlen``) is the same kernel
driven by a ``cu_seq_q`` (per-batch qlen prefix-sum) adapter with MTP tail-causal
windows ``[0, ctx_b - qlen_b + n]``; those cases also report TFLOPS / GB/s.

Accuracy (torch ref):
  - vs exact FP4-dequant ref  -> kernel correctness (cos ~ 1.0)
  - vs full-precision bf16 ref -> FP4 quant accuracy

Performance (vs the ATOM FP8 path, if importable):
  ATOM prefill produces the indexer logits via
    cp_gather_indexer_k_quant_cache  (paged FP8 -> contiguous k_fp8/k_scale)
  + fp8_mqa_logits                   (contiguous K).
  Timed against the single fp4 paged kernel (which eliminates the gather).

Usage:
    python op_tests/test_flydsl_pa_mqa_logits_fp4_prefill.py
    python op_tests/test_flydsl_pa_mqa_logits_fp4_prefill.py --prefill-bs 4 --ctx 2048 --n-q 64
"""

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import is_flydsl_available
from aiter.test_common import benchmark, checkAllclose, run_perftest

dev = "cuda"
SCALE_BLOCK = 32
MFMA_M = 16

_FP4_GRID_VALUES = [
    -6.0,
    -4.0,
    -3.0,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
]
_E2M1_LUT = [0xF, 0xE, 0xD, 0xC, 0xB, 0xA, 0x9, 0x0, 0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7]
_E2M1_INV_LUT = [7, 8, 9, 10, 11, 12, 13, 14, 7, 6, 5, 4, 3, 2, 1, 0]
KVS_NTPW = 4
_ITERS = 50
_WARMUP = 10


# ── FP4 quant / dequant ──────────────────────────────────────────────


def fp4_quant_e2m1_with_e8m0(x, block_size=SCALE_BLOCK):
    *prefix, d = x.shape
    assert d % block_size == 0
    x_blk = x.float().reshape(*prefix, d // block_size, block_size)
    amax = x_blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    exp_unbiased = torch.ceil(torch.log2(amax / 6.0))
    exp_biased = (exp_unbiased + 127.0).clamp(0.0, 255.0).to(torch.uint8)
    e8m0 = exp_biased.squeeze(-1).contiguous()
    scale = torch.pow(2.0, exp_biased.float() - 127.0)
    x_scaled = x_blk / scale
    grid = torch.tensor(_FP4_GRID_VALUES, dtype=torch.float32, device=x.device)
    idx = (x_scaled.unsqueeze(-1) - grid).abs().argmin(dim=-1)
    lut = torch.tensor(_E2M1_LUT, dtype=torch.uint8, device=x.device)
    nibbles = lut[idx].reshape(*prefix, d)
    packed = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).to(torch.uint8)
    return packed.contiguous(), e8m0


def fp4_dequant_e2m1_with_e8m0(packed, e8m0, block_size=SCALE_BLOCK):
    *prefix, d_half = packed.shape
    d = d_half * 2
    low = packed & 0xF
    high = (packed >> 4) & 0xF
    nibbles = torch.empty(*prefix, d, dtype=torch.uint8, device=packed.device)
    nibbles[..., 0::2] = low
    nibbles[..., 1::2] = high
    inv = torch.tensor(_E2M1_INV_LUT, dtype=torch.long, device=packed.device)
    grid = torch.tensor(_FP4_GRID_VALUES, dtype=torch.float32, device=packed.device)
    vals = grid[inv[nibbles.long()]]
    scale = torch.pow(2.0, e8m0.float() - 127.0)
    return (
        vals.reshape(*prefix, d // block_size, block_size) * scale.unsqueeze(-1)
    ).reshape(*prefix, d)


# ── Host-side FP4 layout writers (kernel ABI) ────────────────────────


def quant_q_fp4_preshuffle(q):
    """[total_tokens, H, head_dim] -> q_fp4 [T,H,D/2], q_scale [T,K_TILES,4,16,QS_PAD]."""
    total_tokens, heads, head_dim = q.shape
    m_tiles = heads // MFMA_M
    k_tiles = head_dim // 128
    packed, e8m0 = fp4_quant_e2m1_with_e8m0(q.reshape(total_tokens * heads, head_dim))
    q_fp4 = packed.reshape(total_tokens, heads, head_dim // 2)
    q_e8m0 = e8m0.reshape(total_tokens, heads, head_dim // 32)
    qs_pad = ((m_tiles + 3) // 4) * 4
    qe = (
        q_e8m0.reshape(total_tokens, m_tiles, 16, k_tiles, 4)
        .permute(0, 3, 4, 2, 1)
        .contiguous()
    )
    return q_fp4, torch.nn.functional.pad(qe, (0, qs_pad - m_tiles)).contiguous()


def indexer_k_fp4_paged_preshuffle(k, slot_mapping, kv_cache, kv_scale, kv_block_size):
    """Production-shaped (slot_mapping) FP4 paged-preshuffle K writer.

    Per token at (physical_block p, block_offset o):
      kv_cache[p, kt, kc, o, :]  = 16 packed bytes for K[(kt*4+kc)*32 : +32]
      kv_scale[p, kt, kc, sflat] = e8m0 byte, sflat = (o%16)*4 + (o//16)
    """
    _num_tokens, head_dim = k.shape
    k_tiles = head_dim // 128
    packed, e8m0 = fp4_quant_e2m1_with_e8m0(k)
    valid = slot_mapping >= 0
    sm = slot_mapping[valid].long()
    if sm.numel() == 0:
        return kv_cache, kv_scale
    packed = packed[valid].view(-1, k_tiles, 4, 16)
    e8m0 = e8m0[valid].view(-1, k_tiles, 4)
    phys = sm // kv_block_size
    boff = sm % kv_block_size
    kv_cache[phys, :, :, boff, :] = packed
    sflat = (boff % 16) * KVS_NTPW + (boff // 16)
    kv_scale[phys, :, :, sflat] = e8m0
    return kv_cache, kv_scale


# ── Reference ────────────────────────────────────────────────────────


def ref_prefill_logits(
    q_in, kv_in, weights, row_to_batch, ls, le, max_seq_len, weight_scale=1.0
):
    total_tokens = q_in.shape[0]
    out = torch.full(
        (total_tokens, max_seq_len), float("-inf"), device=dev, dtype=torch.float32
    )
    for r in range(total_tokens):
        b, s, e = int(row_to_batch[r]), int(ls[r]), int(le[r])
        if e <= s:
            continue
        qk = q_in[r].float() @ kv_in[b, s:e].float().T
        qk = torch.relu(qk) * weights[r].float()[:, None]
        out[r, s:e] = qk.sum(dim=0) * weight_scale
    return out


def _cos(a, b):
    a, b = a.double(), b.double()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-12)


# ── Driver ───────────────────────────────────────────────────────────


def run_case(
    bs,
    windows_per_batch,
    heads=64,
    head_dim=128,
    kv_block_size=64,
    block_k=256,
    parallel_unit_num=512,
    seed=0,
    bench=False,
    iters=50,
    warmup=10,
):
    from aiter.ops.flydsl import (
        flydsl_pa_mqa_logits_fp4_prefill,
        flydsl_pa_mqa_logits_fp4_prefill_packed,
        pack_prefill_row_info,
        prepare_pa_mqa_logits_fp4_prefill_packed,
    )

    torch.manual_seed(seed)
    max_end = max(
        (w if isinstance(w, int) else w[1]) for ws in windows_per_batch for w in ws
    )
    max_blocks_per_seq = max(
        (max_end + block_k - 1) // block_k * (block_k // kv_block_size),
        block_k // kv_block_size,
    )
    t_max = max_blocks_per_seq * kv_block_size
    max_seq_len = t_max
    num_blocks = max_blocks_per_seq * bs

    kv_bf16 = torch.randn(bs, t_max, head_dim, dtype=torch.bfloat16, device=dev)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=dev).reshape(
        bs, max_blocks_per_seq
    )

    kv_fp4_d, kv_e8_d = fp4_quant_e2m1_with_e8m0(kv_bf16.reshape(-1, head_dim))
    kv_dq = fp4_dequant_e2m1_with_e8m0(
        kv_fp4_d.reshape(bs, t_max, head_dim // 2),
        kv_e8_d.reshape(bs, t_max, head_dim // 32),
    )

    k_flat = kv_bf16.reshape(bs * t_max, head_dim)
    tb = torch.arange(bs, device=dev).repeat_interleave(t_max)
    tt = torch.arange(t_max, device=dev).repeat(bs)
    phys = block_tables[tb, tt // kv_block_size].long()
    slot_mapping = (phys * kv_block_size + (tt % kv_block_size)).to(torch.int32)
    k_tiles = head_dim // 128
    kv_cache = torch.zeros(
        num_blocks, k_tiles, 4, kv_block_size, 16, dtype=torch.uint8, device=dev
    )
    kv_scale = torch.zeros(
        num_blocks, k_tiles, 4, kv_block_size, dtype=torch.uint8, device=dev
    )
    indexer_k_fp4_paged_preshuffle(
        k_flat, slot_mapping, kv_cache, kv_scale, kv_block_size
    )

    rb, ls, le = [], [], []
    for b in range(bs):
        for w in windows_per_batch[b]:
            s, e = (0, w) if isinstance(w, int) else (w[0], w[1])
            rb.append(b)
            ls.append(s)
            le.append(e)
    total_tokens = len(rb)
    # Bounded split targets at least one CTA per query row.
    parallel_unit_num = max(parallel_unit_num, total_tokens)
    row_to_batch = torch.tensor(rb, dtype=torch.int32, device=dev)
    local_starts = torch.tensor(ls, dtype=torch.int32, device=dev)
    local_ends = torch.tensor(le, dtype=torch.int32, device=dev)
    row_info = pack_prefill_row_info(row_to_batch, local_starts, local_ends)

    q_bf16 = torch.randn(
        total_tokens, heads, head_dim, dtype=torch.bfloat16, device=dev
    )
    weights = (
        torch.randn(total_tokens, heads, dtype=torch.float32, device=dev) * 0.1
    ).to(torch.bfloat16)
    weight_scale = 1.5
    q_fp4, q_scale = quant_q_fp4_preshuffle(q_bf16)
    q_e8 = fp4_quant_e2m1_with_e8m0(q_bf16.reshape(total_tokens * heads, head_dim))[
        1
    ].reshape(total_tokens, heads, head_dim // 32)
    q_dq = fp4_dequant_e2m1_with_e8m0(
        q_fp4.reshape(total_tokens, heads, head_dim // 2), q_e8
    )

    ref_fp4 = ref_prefill_logits(
        q_dq, kv_dq, weights, row_to_batch, ls, le, max_seq_len, weight_scale
    )
    ref_bf16 = ref_prefill_logits(
        q_bf16, kv_bf16, weights, row_to_batch, ls, le, max_seq_len, weight_scale
    )

    out = torch.full(
        (total_tokens, max_seq_len), float("-inf"), dtype=torch.float32, device=dev
    )
    out_compat = torch.full_like(out, float("-inf"))
    out_prepared = torch.full_like(out, float("-inf"))
    prepared = prepare_pa_mqa_logits_fp4_prefill_packed(
        total_tokens=total_tokens,
        heads=heads,
        head_dim=head_dim,
        max_blocks_per_seq=max_blocks_per_seq,
        max_seq_len=max_seq_len,
        block_k=block_k,
        kv_block_size=kv_block_size,
        parallel_unit_num=parallel_unit_num,
    )

    def run_fp4():
        flydsl_pa_mqa_logits_fp4_prefill_packed(
            q_fp4,
            q_scale,
            kv_cache,
            kv_scale,
            block_tables,
            weights,
            row_info,
            max_seq_len,
            weight_scale=weight_scale,
            block_k=block_k,
            kv_block_size=kv_block_size,
            parallel_unit_num=parallel_unit_num,
            out=out,
        )
        return out

    def run_compat():
        flydsl_pa_mqa_logits_fp4_prefill(
            q_fp4,
            q_scale,
            kv_cache,
            kv_scale,
            block_tables,
            weights,
            row_to_batch,
            local_starts,
            local_ends,
            max_seq_len,
            weight_scale=weight_scale,
            block_k=block_k,
            kv_block_size=kv_block_size,
            parallel_unit_num=parallel_unit_num,
            out=out_compat,
            row_info_out=row_info,
        )
        return out_compat

    def run_prepared():
        prepared(
            out_prepared,
            q_fp4,
            q_scale,
            kv_cache,
            kv_scale,
            block_tables,
            weights,
            row_info,
            weight_scale=weight_scale,
        )
        return out_prepared

    run_fp4()
    run_compat()
    run_prepared()
    torch.cuda.synchronize()
    assert torch.equal(out, out_compat), "three-tensor compatibility path differs"
    assert torch.equal(out, out_prepared), "prepared packed launcher differs"

    m = ~torch.isneginf(ref_fp4)
    cos_exact = _cos(out[m], ref_fp4[m]).item()
    cos_bf16 = _cos(out[m], ref_bf16[m]).item()
    oob_ok = bool(torch.isneginf(out[~m]).all().item()) if (~m).any() else True
    print(
        f"  bs={bs} heads={heads} total_tokens={total_tokens} "
        f"cos_exact={cos_exact:.6f} cos_bf16={cos_bf16:.6f} oob_neginf={oob_ok}"
    )
    assert cos_exact > 0.99, f"kernel vs FP4-dequant ref cos {cos_exact:.4f} < 0.99"
    assert cos_bf16 > 0.95, f"kernel vs bf16 ref cos {cos_bf16:.4f} < 0.95"
    assert oob_ok, "OOB cells were not left at -inf"

    if not bench:
        return

    pairs = sum(e - s for s, e in zip(ls, le))
    kv_tokens = sum(
        max((e for b_row, e in zip(rb, le) if b_row == b), default=0) for b in range(bs)
    )
    flops = pairs * heads * (2 * head_dim + 3)
    bytes_total = (
        total_tokens * heads * (head_dim // 2 + head_dim // 32)
        + kv_tokens * (head_dim // 2 + head_dim // 32)
        + total_tokens * heads * weights.element_size()
        + block_tables.numel() * block_tables.element_size()
        + pairs * out.element_size()
    )
    candidates = {"packed": run_fp4}
    ret = {"gfx": get_gfx(), "pairs": pairs}
    for name, fn in candidates.items():
        candidate_out, us = run_perftest(
            fn,
            num_iters=iters,
            num_warmup=warmup,
            use_cuda_event=True,
        )
        err = checkAllclose(
            ref_fp4[m].to(torch.float32),
            candidate_out[m].to(torch.float32),
            rtol=0.05,
            atol=0.05,
            msg=f"{name}: FP4 prefill",
            printLog=False,
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = bytes_total / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_prefill(bs, n_q, ctx, heads, head_dim):
    ctx = max(64, (ctx // 64) * 64)
    windows = [[ctx] * n_q for _ in range(bs)]
    return run_case(
        bs,
        windows,
        heads=heads,
        head_dim=head_dim,
        bench=True,
        iters=_ITERS,
        warmup=_WARMUP,
    )


def _bench_vs_atom(
    kv_bf16,
    slot_mapping,
    block_tables,
    q_bf16,
    weights,
    row_to_batch,
    local_ends,
    ls,
    le,
    ref_bf16,
    heads,
    head_dim,
    kv_block_size,
    t_max,
    bs,
    us_fp4,
    iters,
    warmup,
):
    atom_ok = False
    try:
        from aiter import (
            cp_gather_indexer_k_quant_cache,
            dtypes,
            indexer_k_quant_and_cache,
        )
        from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits

        total_tokens = q_bf16.shape[0]
        num_blocks = block_tables.numel()
        # committed length per sequence = max local_end among its rows
        ctx_b = [0] * bs
        for r in range(total_tokens):
            ctx_b[int(row_to_batch[r])] = max(ctx_b[int(row_to_batch[r])], int(le[r]))
        cu = [0]
        for c in ctx_b:
            cu.append(cu[-1] + c)
        total_committed = cu[-1]

        kv_cache_fp8 = torch.zeros(
            (num_blocks, kv_block_size, head_dim + 4), dtype=dtypes.fp8, device=dev
        )
        indexer_k_quant_and_cache(
            kv_bf16.reshape(bs * t_max, head_dim),
            kv_cache_fp8,
            slot_mapping.to(torch.int64),
            head_dim,
            "ue8m0",
            True,
        )
        cu_committed = torch.tensor(cu, dtype=torch.int32, device=dev)
        dst_k = torch.empty((total_committed, head_dim), dtype=dtypes.fp8, device=dev)
        dst_scale = torch.empty((total_committed, 1), dtype=torch.float32, device=dev)
        q_fp8 = q_bf16.to(dtypes.fp8)
        cu_starts = torch.tensor(
            [cu[int(row_to_batch[r])] + ls[r] for r in range(total_tokens)],
            dtype=torch.int32,
            device=dev,
        )
        cu_ends = torch.tensor(
            [cu[int(row_to_batch[r])] + le[r] for r in range(total_tokens)],
            dtype=torch.int32,
            device=dev,
        )

        def atom_logits():
            cp_gather_indexer_k_quant_cache(
                kv_cache_fp8,
                dst_k,
                dst_scale.view(dtypes.fp8),
                block_tables,
                cu_committed,
                True,
            )
            return fp8_mqa_logits(
                q_fp8,
                dst_k,
                dst_scale,
                weights.float(),
                cu_starts,
                cu_ends,
                clean_logits=False,
            )

        atom_out = atom_logits()
        torch.cuda.synchronize()
        vo, vr = [], []
        for r in range(total_tokens):
            b, e = int(row_to_batch[r]), int(le[r])
            vo.append(atom_out[r, cu[b] : cu[b] + e])
            vr.append(ref_bf16[r, :e])
        cos_atom = _cos(torch.cat(vo), torch.cat(vr)).item()

        _, us_gather = run_perftest(
            cp_gather_indexer_k_quant_cache,
            kv_cache_fp8,
            dst_k,
            dst_scale.view(dtypes.fp8),
            block_tables,
            cu_committed,
            True,
            num_iters=iters,
            num_warmup=warmup,
        )
        _, us_atom = run_perftest(atom_logits, num_iters=iters, num_warmup=warmup)
        atom_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"  [perf] ATOM path unavailable ({type(e).__name__}: {e})")

    print("\n  {:<28} | {:>10}".format("path", "us"))
    print("  " + "-" * 42)
    print("  {:<28} | {:>10.2f}".format("FP4 paged (single kernel)", us_fp4))
    if atom_ok:
        print("  {:<28} | {:>10.2f}".format("ATOM cp_gather", us_gather))
        print("  {:<28} | {:>10.2f}".format("ATOM cp_gather+fp8_logits", us_atom))
        print(f"  [accuracy] ATOM fp8 vs bf16 ref cos={cos_atom:.6f}")
        print(f"  speedup (ATOM total / FP4) = {us_atom / us_fp4:.2f}x")
    print()


# ── Variable-qlen (per-batch MTP) via cu_seq_q ───────────────────────


def _mtp_windows_ref(qlens, ctxs):
    """Independent (python) construction of the MTP tail-causal windows, used
    both for the torch reference and to unit-check the device-side builder."""
    rb, ls, le = [], [], []
    for b, (ql, ctx) in enumerate(zip(qlens, ctxs)):
        for n in range(ql):
            rb.append(b)
            ls.append(0)
            le.append(max(0, ctx - ql + n + 1))
    return rb, ls, le


def run_varqlen_case(
    qlens,
    ctxs,
    heads=64,
    head_dim=128,
    kv_block_size=64,
    block_k=256,
    seed=0,
    bench=True,
    iters=50,
    warmup=10,
    padded_total_q=None,
    check_graph=False,
):
    from aiter.ops.flydsl import (
        compute_varqlen_row_info,
        flydsl_pa_mqa_logits_fp4_prefill_packed,
        flydsl_pa_mqa_logits_fp4_varqlen,
    )
    from aiter.ops.flydsl.kernels.pa_mqa_logits_fp4_prefill import (
        compute_varqlen_windows,
    )

    torch.manual_seed(seed)
    bs = len(qlens)
    assert len(ctxs) == bs
    real_total_q = int(sum(qlens))
    total_q = real_total_q if padded_total_q is None else int(padded_total_q)
    assert total_q >= real_total_q

    max_end = max(ctxs)
    max_blocks_per_seq = max(
        (max_end + block_k - 1) // block_k * (block_k // kv_block_size),
        block_k // kv_block_size,
    )
    t_max = max_blocks_per_seq * kv_block_size
    max_seq_len = t_max
    num_blocks = max_blocks_per_seq * bs

    # ---- paged FP4 KV (dense per-batch cache, contiguous block_tables) ----
    kv_bf16 = torch.randn(bs, t_max, head_dim, dtype=torch.bfloat16, device=dev)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=dev).reshape(
        bs, max_blocks_per_seq
    )
    kv_fp4_d, kv_e8_d = fp4_quant_e2m1_with_e8m0(kv_bf16.reshape(-1, head_dim))
    kv_dq = fp4_dequant_e2m1_with_e8m0(
        kv_fp4_d.reshape(bs, t_max, head_dim // 2),
        kv_e8_d.reshape(bs, t_max, head_dim // 32),
    )
    k_flat = kv_bf16.reshape(bs * t_max, head_dim)
    tb = torch.arange(bs, device=dev).repeat_interleave(t_max)
    tt = torch.arange(t_max, device=dev).repeat(bs)
    phys = block_tables[tb, tt // kv_block_size].long()
    slot_mapping = (phys * kv_block_size + (tt % kv_block_size)).to(torch.int32)
    k_tiles = head_dim // 128
    kv_cache = torch.zeros(
        num_blocks, k_tiles, 4, kv_block_size, 16, dtype=torch.uint8, device=dev
    )
    kv_scale = torch.zeros(
        num_blocks, k_tiles, 4, kv_block_size, dtype=torch.uint8, device=dev
    )
    indexer_k_fp4_paged_preshuffle(
        k_flat, slot_mapping, kv_cache, kv_scale, kv_block_size
    )

    # ---- packed (ragged) Q / weights: [total_q, ...] ----
    q_bf16 = torch.randn(total_q, heads, head_dim, dtype=torch.bfloat16, device=dev)
    weights = (torch.randn(total_q, heads, dtype=torch.float32, device=dev) * 0.1).to(
        torch.bfloat16
    )
    weight_scale = 1.5
    q_fp4, q_scale = quant_q_fp4_preshuffle(q_bf16)
    q_e8 = fp4_quant_e2m1_with_e8m0(q_bf16.reshape(total_q * heads, head_dim))[
        1
    ].reshape(total_q, heads, head_dim // 32)
    q_dq = fp4_dequant_e2m1_with_e8m0(
        q_fp4.reshape(total_q, heads, head_dim // 2), q_e8
    )

    # ---- cu_seq_q (prefix-sum of per-batch qlen) + per-batch KV length ----
    cu = [0]
    for ql in qlens:
        cu.append(cu[-1] + ql)
    cu_seq_q = torch.tensor(cu, dtype=torch.int32, device=dev)
    context_lens = torch.tensor(ctxs, dtype=torch.int32, device=dev)

    # ---- unit-check the device builder against the independent python windows ----
    rb_ref, ls_ref, le_ref = _mtp_windows_ref(qlens, ctxs)
    rb_k, ls_k, le_k = compute_varqlen_windows(cu_seq_q, context_lens, total_q)
    row_info_k = compute_varqlen_row_info(cu_seq_q, context_lens, total_q)
    assert rb_k[:real_total_q].tolist() == rb_ref
    assert ls_k[:real_total_q].tolist() == ls_ref
    assert le_k[:real_total_q].tolist() == le_ref
    assert not torch.any(le_k[real_total_q:]), "padded rows must have empty windows"
    assert row_info_k[:real_total_q, 0].tolist() == rb_ref
    assert row_info_k[:real_total_q, 1].tolist() == ls_ref
    assert row_info_k[:real_total_q, 2].tolist() == le_ref
    assert not torch.any(row_info_k[real_total_q:, 2])
    assert not torch.any(row_info_k[:, 3]), "reserved row-info field must be zero"
    rb_ref += [0] * (total_q - real_total_q)
    ls_ref += [0] * (total_q - real_total_q)
    le_ref += [0] * (total_q - real_total_q)

    ref_fp4 = ref_prefill_logits(
        q_dq, kv_dq, weights, rb_ref, ls_ref, le_ref, max_seq_len, weight_scale
    )
    ref_bf16 = ref_prefill_logits(
        q_bf16, kv_bf16, weights, rb_ref, ls_ref, le_ref, max_seq_len, weight_scale
    )

    out = flydsl_pa_mqa_logits_fp4_varqlen(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        max_seq_len,
        cu_seq_q=cu_seq_q,
        context_lens=context_lens,
        next_n_max=max(max(qlens), 1),
        weight_scale=weight_scale,
        block_k=block_k,
        kv_block_size=kv_block_size,
    )
    torch.cuda.synchronize()
    out_packed = flydsl_pa_mqa_logits_fp4_prefill_packed(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        row_info_k,
        max_seq_len,
        weight_scale=weight_scale,
        block_k=block_k,
        kv_block_size=kv_block_size,
        parallel_unit_num=1024,
    )
    torch.cuda.synchronize()
    out_row_info_route = flydsl_pa_mqa_logits_fp4_varqlen(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        max_seq_len,
        row_info=row_info_k,
        weight_scale=weight_scale,
        block_k=block_k,
        kv_block_size=kv_block_size,
        parallel_unit_num=1024,
    )
    torch.cuda.synchronize()
    assert torch.equal(
        out_packed, out_row_info_route
    ), "row_info varqlen route differs from packed prefill"

    # Passing compatibility windows alongside cu_seq_q must still select the
    # unified decode path and match bit-for-bit.
    out_reuse = flydsl_pa_mqa_logits_fp4_varqlen(
        q_fp4,
        q_scale,
        kv_cache,
        kv_scale,
        block_tables,
        weights,
        max_seq_len,
        cu_seq_q=cu_seq_q,
        context_lens=context_lens,
        next_n_max=max(max(qlens), 1),
        windows=(rb_k, ls_k, le_k),
        weight_scale=weight_scale,
        block_k=block_k,
        kv_block_size=kv_block_size,
    )
    torch.cuda.synchronize()
    assert torch.equal(out, out_reuse), "windows= reuse path differs from cu_seq_q path"

    m = ~torch.isneginf(ref_fp4)
    cos_exact = _cos(out[m], ref_fp4[m]).item()
    cos_bf16 = _cos(out[m], ref_bf16[m]).item()
    cos_packed = _cos(out_packed[m], ref_fp4[m]).item()
    oob_ok = bool(torch.isneginf(out[~m]).all().item()) if (~m).any() else True
    packed_oob_ok = (
        bool(torch.isneginf(out_packed[~m]).all().item()) if (~m).any() else True
    )
    print(
        f"  qlens={qlens} ctxs={ctxs} heads={heads} "
        f"real_q={real_total_q} allocated_q={total_q} "
        f"cos_exact={cos_exact:.6f} cos_packed={cos_packed:.6f} "
        f"cos_bf16={cos_bf16:.6f} oob_neginf={oob_ok and packed_oob_ok}"
    )
    assert cos_exact > 0.99, f"kernel vs FP4-dequant ref cos {cos_exact:.4f} < 0.99"
    assert cos_packed > 0.99, f"packed prefill cos {cos_packed:.4f} < 0.99"
    assert cos_bf16 > 0.95, f"kernel vs bf16 ref cos {cos_bf16:.4f} < 0.95"
    assert oob_ok, "OOB cells were not left at -inf"
    assert packed_oob_ok, "packed prefill wrote OOB cells"

    if check_graph:
        graph_out = torch.full_like(out, float("-inf"))

        def graph_launch():
            graph_out.fill_(float("-inf"))
            flydsl_pa_mqa_logits_fp4_varqlen(
                q_fp4,
                q_scale,
                kv_cache,
                kv_scale,
                block_tables,
                weights,
                max_seq_len,
                cu_seq_q=cu_seq_q,
                context_lens=context_lens,
                next_n_max=max(max(qlens), 1),
                weight_scale=weight_scale,
                block_k=block_k,
                kv_block_size=kv_block_size,
                out=graph_out,
            )

        graph_launch()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_launch()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, graph_out), "CUDAGraph replay differs from eager decode"

        row_info_graph = torch.empty_like(row_info_k)
        compute_varqlen_row_info(
            cu_seq_q,
            context_lens,
            total_q,
            out=row_info_graph,
        )
        packed_graph_out = torch.full_like(out_packed, float("-inf"))

        def packed_graph_launch():
            packed_graph_out.fill_(float("-inf"))
            flydsl_pa_mqa_logits_fp4_prefill_packed(
                q_fp4,
                q_scale,
                kv_cache,
                kv_scale,
                block_tables,
                weights,
                row_info_graph,
                max_seq_len,
                weight_scale=weight_scale,
                block_k=block_k,
                kv_block_size=kv_block_size,
                parallel_unit_num=1024,
                out=packed_graph_out,
            )

        packed_graph_launch()
        torch.cuda.synchronize()
        packed_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(packed_graph):
            packed_graph_launch()
        compute_varqlen_row_info(
            cu_seq_q,
            context_lens,
            total_q,
            out=row_info_graph,
        )
        packed_graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(
            out_packed, packed_graph_out
        ), "packed row-info CUDAGraph replay differs"

    if not bench:
        return

    # Isolate the bounded-split kernel launch.
    target_ctas = 1024
    out_buf = torch.full(
        (total_q, max_seq_len), float("-inf"), dtype=torch.float32, device=dev
    )

    def run_fp4():
        flydsl_pa_mqa_logits_fp4_varqlen(
            q_fp4,
            q_scale,
            kv_cache,
            kv_scale,
            block_tables,
            weights,
            max_seq_len,
            cu_seq_q=cu_seq_q,
            context_lens=context_lens,
            next_n_max=max(max(qlens), 1),
            weight_scale=weight_scale,
            block_k=block_k,
            kv_block_size=kv_block_size,
            parallel_unit_num=target_ctas,
            out=out_buf,
        )
        return out_buf

    # Useful work: exact ragged (row, token) pairs; KV uses the per-batch
    # unique footprint, assuming cross-row L2 reuse.
    head_dim_packed, head_dim_scales = head_dim // 2, head_dim // 32
    win = [int(e - s) for s, e in zip(ls_ref, le_ref)]
    pairs = sum(win)
    kv_uni = {}
    for b, e in zip(rb_ref, le_ref):
        kv_uni[b] = max(kv_uni.get(b, 0), int(e))
    kv_tokens = sum(kv_uni.values())
    flops = pairs * heads * (2 * head_dim + 3)
    bytes_total = (
        total_q * heads * (head_dim_packed + head_dim_scales)  # Q fp4 + scale
        + kv_tokens * (head_dim_packed + head_dim_scales)  # KV fp4 + scale
        + total_q * heads * weights.element_size()
        + block_tables.numel() * block_tables.element_size()
        + pairs * out_buf.element_size()
    )
    candidates = {"bounded": run_fp4}
    ret = {"gfx": get_gfx(), "real_q": real_total_q, "pairs": pairs}
    for name, fn in candidates.items():
        candidate_out, us = run_perftest(
            fn,
            num_iters=iters,
            num_warmup=warmup,
            use_cuda_event=True,
        )
        err = checkAllclose(
            ref_fp4[m].to(torch.float32),
            candidate_out[m].to(torch.float32),
            rtol=0.05,
            atol=0.05,
            msg=f"{name}: ragged FP4 decode",
            printLog=False,
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = bytes_total / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_varqlen(qlens, ctxs, heads, padded_total_q):
    return run_varqlen_case(
        list(qlens),
        list(ctxs),
        heads=heads,
        bench=True,
        iters=_ITERS,
        warmup=_WARMUP,
        padded_total_q=padded_total_q or None,
        check_graph=bool(padded_total_q),
    )


def _parse_ragged_shape(value):
    parts = value.split(":")
    if len(parts) not in (2, 3):
        raise argparse.ArgumentTypeError("expected q0,q1,...:ctx0,ctx1,...[:padded_q]")
    qlens = tuple(int(x) for x in parts[0].split(","))
    ctxs = tuple(int(x) for x in parts[1].split(","))
    if len(qlens) != len(ctxs):
        raise argparse.ArgumentTypeError(
            "qlens and context_lens must have equal length"
        )
    padded = int(parts[2]) if len(parts) == 3 else 0
    return qlens, ctxs, padded


def main():
    if get_gfx() != "gfx950":
        aiter.logger.warning(
            "flydsl FP4 paged MQA logits unsupported on %s; skipping", get_gfx()
        )
        return
    if not is_flydsl_available():
        aiter.logger.warning("flydsl is unavailable; skipping FP4 paged MQA logits")
        return

    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    ap.add_argument("--prefill-bs", type=int, nargs="*", default=[2])
    ap.add_argument("--ctx", type=int, nargs="*", default=[512])
    ap.add_argument("--n-q", type=int, nargs="*", default=[8])
    ap.add_argument("--heads", type=int, nargs="*", default=[64])
    ap.add_argument("--head-dim", type=int, nargs="*", default=[128])
    ap.add_argument(
        "--ragged",
        type=_parse_ragged_shape,
        nargs="*",
        default=[
            ((4, 1, 3), (512, 0, 257), 12),
            ((2, 0, 3), (384, 256, 640), 0),
        ],
        help="qlens:context_lens[:padded_total_q]",
    )
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    global _ITERS, _WARMUP
    _ITERS, _WARMUP = args.iters, args.warmup

    # Non-zero lower bounds are prefill-specific corner coverage.
    run_case(2, [[(10, 50), (64, 200)], [(0, 100), (130, 256)]], seed=4)

    prefill_rows = [
        test_prefill(bs, n_q, ctx, heads, head_dim)
        for bs, n_q, ctx, heads, head_dim in itertools.product(
            args.prefill_bs,
            args.n_q,
            args.ctx,
            args.heads,
            args.head_dim,
        )
    ]
    aiter.logger.info(
        "flydsl FP4 prefill summary (markdown):\n%s",
        pd.DataFrame(prefill_rows).to_markdown(index=False),
    )

    varqlen_rows = [
        test_varqlen(qlens, ctxs, heads, padded)
        for (qlens, ctxs, padded), heads in itertools.product(args.ragged, args.heads)
    ]
    aiter.logger.info(
        "flydsl FP4 varqlen decode summary (markdown):\n%s",
        pd.DataFrame(varqlen_rows).to_markdown(index=False),
    )


if __name__ == "__main__":
    main()
