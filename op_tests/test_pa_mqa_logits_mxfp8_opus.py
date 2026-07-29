#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter op-test for the MXFP8 paged MQA logits OPUS kernel (gfx950).

Validates ``pa_mqa_logits_mxfp8_prefill`` (ragged windows) and
``pa_mqa_logits_mxfp8_varqlen`` (per-batch variable qlen / MTP tail-causal) against
a pure-torch reference:

  - vs exact FP8-dequant ref   -> kernel correctness (cos ~ 1.0)
  - vs full-precision bf16 ref -> FP8 quant accuracy

Q/KV are MXFP8-quantized + preshuffled on the host into the exact kernel ABI
(E4M3 data + E8M0 block scales, block=32); the persistent-grid schedule and the
varqlen windows are built device-side by the op.

Usage:
    python op_tests/test_pa_mqa_logits_mxfp8_opus.py
    python op_tests/test_pa_mqa_logits_mxfp8_opus.py --bench --bs 4 --ctx 2048 --n_q 64
"""

import argparse

import torch
import torch.nn.functional as F

from aiter.ops.pa_mqa_logits_mxfp8_opus import (
    compute_prefill_schedule,
    compute_varqlen_windows,
    pa_mqa_logits_mxfp8_prefill,
    pa_mqa_logits_mxfp8_varqlen,
)

dev = "cuda"
SCALE_BLOCK = 32
MFMA_M = 16
FP8_E4M3_MAX = 448.0
KVS_NTPW = 4


# ── MXFP8 (E4M3 + E8M0) quant / dequant ──────────────────────────────


def fp8_quant_e4m3_with_e8m0(x, block_size=SCALE_BLOCK):
    """[..., d] float -> (fp8 E4M3 bytes [..., d] uint8, e8m0 [..., d/block] uint8)."""
    *prefix, d = x.shape
    assert d % block_size == 0
    x_blk = x.float().reshape(*prefix, d // block_size, block_size)
    amax = x_blk.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    exp_unbiased = torch.ceil(torch.log2(amax / FP8_E4M3_MAX))
    exp_biased = (exp_unbiased + 127.0).clamp(0.0, 254.0).to(torch.uint8)
    e8m0 = exp_biased.squeeze(-1).contiguous()
    scale = torch.pow(2.0, exp_biased.float() - 127.0)
    x_scaled = (x_blk / scale).reshape(*prefix, d)
    fp8 = x_scaled.to(torch.float8_e4m3fn)
    fp8_bytes = fp8.view(torch.uint8).contiguous()
    return fp8_bytes, e8m0


def fp8_dequant_e4m3_with_e8m0(fp8_bytes, e8m0, block_size=SCALE_BLOCK):
    *prefix, d = fp8_bytes.shape
    vals = fp8_bytes.view(torch.float8_e4m3fn).float()
    scale = torch.pow(2.0, e8m0.float() - 127.0)
    return (
        vals.reshape(*prefix, d // block_size, block_size) * scale.unsqueeze(-1)
    ).reshape(*prefix, d)


# ── Host-side FP8 layout writers (kernel ABI, mirrors logits_host.cc) ──


def quant_q_fp8_preshuffle(q):
    """[T, H, head_dim] -> q_fp8 [T,H,D] uint8, q_scale [T,K_TILES,4,16,QS_PAD] uint8."""
    total_tokens, heads, head_dim = q.shape
    m_tiles = heads // MFMA_M
    k_tiles = head_dim // 128
    fp8, e8m0 = fp8_quant_e4m3_with_e8m0(q.reshape(total_tokens * heads, head_dim))
    q_fp8 = fp8.reshape(total_tokens, heads, head_dim)
    q_e8m0 = e8m0.reshape(total_tokens, heads, head_dim // 32)
    qs_pad = ((m_tiles + 3) // 4) * 4
    qe = (
        q_e8m0.reshape(total_tokens, m_tiles, 16, k_tiles, 4)
        .permute(0, 3, 4, 2, 1)
        .contiguous()
    )
    return q_fp8, F.pad(qe, (0, qs_pad - m_tiles)).contiguous()


def indexer_k_fp8_paged_preshuffle(k, slot_mapping, kv_cache, kv_scale, kv_block_size):
    """Paged-preshuffle FP8 K writer.

    Per token at (physical block p, block offset o):
      kv_cache[p, kt, c, o, :]  = 16 fp8 bytes for K[(kt*8+c)*16 : +16]  (c in 0..7)
      kv_scale[p, kt, kc, sflat] = e8m0 byte, sflat = (o%16)*4 + (o//16)  (kc in 0..3)
    """
    _num_tokens, head_dim = k.shape
    k_tiles = head_dim // 128
    fp8, e8m0 = fp8_quant_e4m3_with_e8m0(k)
    valid = slot_mapping >= 0
    sm = slot_mapping[valid].long()
    if sm.numel() == 0:
        return kv_cache, kv_scale
    fp8 = fp8[valid].view(-1, k_tiles, 8, 16)
    e8m0 = e8m0[valid].view(-1, k_tiles, 4)
    phys = sm // kv_block_size
    boff = sm % kv_block_size
    kv_cache[phys, :, :, boff, :] = fp8
    sflat = (boff % 16) * KVS_NTPW + (boff // 16)
    kv_scale[phys, :, :, sflat] = e8m0
    return kv_cache, kv_scale


# ── Reference ────────────────────────────────────────────────────────


def ref_prefill_logits(q_in, kv_in, weights, row_to_batch, ls, le, max_seq_len, ws=1.0):
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
        out[r, s:e] = qk.sum(dim=0) * ws
    return out


def _cos(a, b):
    a, b = a.double(), b.double()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-12)


# ── Shared setup: build preshuffled KV + Q for a batch of ragged windows ──


def _build_inputs(bs, windows_per_batch, heads, head_dim, kv_block_size, block_k, seed):
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

    kv_fp8_d, kv_e8_d = fp8_quant_e4m3_with_e8m0(kv_bf16.reshape(-1, head_dim))
    kv_dq = fp8_dequant_e4m3_with_e8m0(
        kv_fp8_d.reshape(bs, t_max, head_dim),
        kv_e8_d.reshape(bs, t_max, head_dim // 32),
    )

    k_flat = kv_bf16.reshape(bs * t_max, head_dim)
    tb = torch.arange(bs, device=dev).repeat_interleave(t_max)
    tt = torch.arange(t_max, device=dev).repeat(bs)
    phys = block_tables[tb, tt // kv_block_size].long()
    slot_mapping = (phys * kv_block_size + (tt % kv_block_size)).to(torch.int32)
    k_tiles = head_dim // 128
    kv_cache = torch.zeros(
        num_blocks, k_tiles, 8, kv_block_size, 16, dtype=torch.uint8, device=dev
    )
    kv_scale = torch.zeros(
        num_blocks, k_tiles, 4, kv_block_size, dtype=torch.uint8, device=dev
    )
    indexer_k_fp8_paged_preshuffle(k_flat, slot_mapping, kv_cache, kv_scale, kv_block_size)

    return (
        kv_bf16,
        kv_dq,
        kv_cache,
        kv_scale,
        block_tables,
        t_max,
        max_seq_len,
    )


def _build_q(total_tokens, heads, head_dim, weight_scale):
    q_bf16 = torch.randn(total_tokens, heads, head_dim, dtype=torch.bfloat16, device=dev)
    weights = (
        torch.randn(total_tokens, heads, dtype=torch.float32, device=dev) * 0.1
    ).to(torch.bfloat16)
    q_fp8, q_scale = quant_q_fp8_preshuffle(q_bf16)
    q_e8 = fp8_quant_e4m3_with_e8m0(q_bf16.reshape(total_tokens * heads, head_dim))[
        1
    ].reshape(total_tokens, heads, head_dim // 32)
    q_dq = fp8_dequant_e4m3_with_e8m0(q_fp8, q_e8)
    return q_bf16, q_fp8, q_scale, q_dq, weights


# ── Prefill driver ───────────────────────────────────────────────────


def run_prefill_case(
    bs,
    windows_per_batch,
    heads=64,
    head_dim=128,
    kv_block_size=64,
    block_k=256,
    parallel_unit_num=512,
    seed=0,
):
    (
        kv_bf16,
        kv_dq,
        kv_cache,
        kv_scale,
        block_tables,
        t_max,
        max_seq_len,
    ) = _build_inputs(bs, windows_per_batch, heads, head_dim, kv_block_size, block_k, seed)

    rb, ls, le = [], [], []
    for b in range(bs):
        for w in windows_per_batch[b]:
            s, e = (0, w) if isinstance(w, int) else (w[0], w[1])
            rb.append(b)
            ls.append(s)
            le.append(e)
    total_tokens = len(rb)
    parallel_unit_num = max(parallel_unit_num, total_tokens)
    row_to_batch = torch.tensor(rb, dtype=torch.int32, device=dev)
    local_starts = torch.tensor(ls, dtype=torch.int32, device=dev)
    local_ends = torch.tensor(le, dtype=torch.int32, device=dev)

    weight_scale = 1.5
    q_bf16, q_fp8, q_scale, q_dq, weights = _build_q(total_tokens, heads, head_dim, weight_scale)

    ref_fp8 = ref_prefill_logits(q_dq, kv_dq, weights, row_to_batch, ls, le, max_seq_len, weight_scale)
    ref_bf16 = ref_prefill_logits(q_bf16, kv_bf16, weights, row_to_batch, ls, le, max_seq_len, weight_scale)

    out = pa_mqa_logits_mxfp8_prefill(
        q_fp8.view(torch.float8_e4m3fn),
        q_scale,
        kv_cache.view(torch.float8_e4m3fn),
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
    )
    torch.cuda.synchronize()

    m = ~torch.isneginf(ref_fp8)
    cos_fp8 = _cos(out[m], ref_fp8[m]).item()
    cos_bf16 = _cos(out[m], ref_bf16[m]).item()
    oob_ok = bool(torch.isneginf(out[~m]).all().item()) if (~m).any() else True
    ok = cos_fp8 > 0.999 and cos_bf16 > 0.99 and oob_ok
    print(
        f"  [prefill] bs={bs} tt={total_tokens} cos_fp8={cos_fp8:.6f} "
        f"cos_bf16={cos_bf16:.6f} oob_neginf={oob_ok} {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


# ── Varqlen / MTP driver ─────────────────────────────────────────────


def run_varqlen_case(
    bs,
    qlens,
    context_lens,
    heads=64,
    head_dim=128,
    kv_block_size=64,
    block_k=256,
    seed=0,
):
    # per-batch single window [0, ctx] just to size the KV cache large enough.
    windows_per_batch = [[(0, int(context_lens[b]))] for b in range(bs)]
    (
        kv_bf16,
        kv_dq,
        kv_cache,
        kv_scale,
        block_tables,
        t_max,
        max_seq_len,
    ) = _build_inputs(bs, windows_per_batch, heads, head_dim, kv_block_size, block_k, seed)

    cu = [0]
    for q in qlens:
        cu.append(cu[-1] + int(q))
    total_q = cu[-1]
    cu_seq_q = torch.tensor(cu, dtype=torch.int32, device=dev)
    ctx = torch.tensor(context_lens, dtype=torch.int32, device=dev)

    # reference windows (MTP tail-causal): row r in batch b, n-th token ->
    # [0, ctx_b - (qlen_b - 1 - n)).
    rb, ls, le = [], [], []
    for b in range(bs):
        ql = int(qlens[b])
        for n in range(ql):
            rb.append(b)
            ls.append(0)
            le.append(max(int(context_lens[b]) - (ql - 1 - n), 0))
    row_to_batch = torch.tensor(rb, dtype=torch.int32, device=dev)

    weight_scale = 1.5
    q_bf16, q_fp8, q_scale, q_dq, weights = _build_q(total_q, heads, head_dim, weight_scale)

    ref_fp8 = ref_prefill_logits(q_dq, kv_dq, weights, row_to_batch, ls, le, max_seq_len, weight_scale)
    ref_bf16 = ref_prefill_logits(q_bf16, kv_bf16, weights, row_to_batch, ls, le, max_seq_len, weight_scale)

    out = pa_mqa_logits_mxfp8_varqlen(
        q_fp8.view(torch.float8_e4m3fn),
        q_scale,
        kv_cache.view(torch.float8_e4m3fn),
        kv_scale,
        block_tables,
        weights,
        max_seq_len,
        cu_seq_q=cu_seq_q,
        context_lens=ctx,
        weight_scale=weight_scale,
        block_k=block_k,
        kv_block_size=kv_block_size,
    )
    torch.cuda.synchronize()

    m = ~torch.isneginf(ref_fp8)
    cos_fp8 = _cos(out[m], ref_fp8[m]).item()
    cos_bf16 = _cos(out[m], ref_bf16[m]).item()
    ok = cos_fp8 > 0.999 and cos_bf16 > 0.99
    print(
        f"  [varqlen] bs={bs} total_q={total_q} cos_fp8={cos_fp8:.6f} "
        f"cos_bf16={cos_bf16:.6f} {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


# ── Perf (aligned with the FlyDSL op-test: run_perftest avg-us, schedule
#    precomputed once so only the kernel launch is timed; vs FlyDSL fp4 paged
#    and the ATOM cp_gather + Triton fp8_mqa_logits path). ──


def _time_graph(fn, iters=100, warmup=20, k_in_graph=25):
    """Serving-like back-to-back timing: capture K kernels into ONE graph so each
    replay runs K kernels gap-free on the GPU (host can't inject idle between
    them -> no dispatch-gap clock throttling), then time replays with cuda events.
    Returns avg us per single kernel."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(k_in_graph):
            fn()
    for _ in range(warmup):
        g.replay()
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    reps = max(1, iters // k_in_graph)
    st.record()
    for _ in range(reps):
        g.replay()
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) * 1000.0 / (reps * k_in_graph)  # ms -> us per kernel


def run_bench(bs, n_q, ctx, heads=64, head_dim=128, kv_block_size=64, block_k=256,
              parallel_unit_num=512, iters=100, warmup=20, seed=0, graph=False):
    from aiter.test_common import run_perftest

    def measure(fn):
        if graph:
            return _time_graph(fn, iters, warmup)
        return run_perftest(fn, num_iters=iters, num_warmup=warmup)[1]

    # uniform workload: bs batches, each n_q query rows over window [0, ctx).
    windows_per_batch = [[(0, ctx)] * n_q for _ in range(bs)]
    (kv_bf16, kv_dq, kv_cache, kv_scale, block_tables, t_max, max_seq_len) = _build_inputs(
        bs, windows_per_batch, heads, head_dim, kv_block_size, block_k, seed
    )
    total_tokens = bs * n_q
    rb = torch.arange(bs, device=dev).repeat_interleave(n_q).to(torch.int32)
    ls = torch.zeros(total_tokens, dtype=torch.int32, device=dev)
    le = torch.full((total_tokens,), ctx, dtype=torch.int32, device=dev)
    if parallel_unit_num is None:
        parallel_unit_num = 512
    parallel_unit_num = max(parallel_unit_num, total_tokens)

    weight_scale = 1.5
    q_bf16, q_fp8, q_scale, q_dq, weights = _build_q(total_tokens, heads, head_dim, weight_scale)

    # our schedule + out buffer (precomputed once; bench times the launch only)
    cta_info, n_ctas = compute_prefill_schedule(
        rb, ls, le, block_k, parallel_unit_num, max_seq_len
    )
    out = torch.full((total_tokens, max_seq_len), float("-inf"), dtype=torch.float32, device=dev)
    q_fp8_v = q_fp8.view(torch.float8_e4m3fn)
    kv_cache_v = kv_cache.view(torch.float8_e4m3fn)

    def run_ours():
        pa_mqa_logits_mxfp8_prefill(
            q_fp8_v, q_scale, kv_cache_v, kv_scale, block_tables, weights,
            rb, ls, le, max_seq_len, weight_scale=weight_scale, block_k=block_k,
            kv_block_size=kv_block_size, parallel_unit_num=parallel_unit_num,
            out=out, cta_info=cta_info, n_ctas=n_ctas,
        )

    run_ours()
    torch.cuda.synchronize()
    m = ~torch.isneginf(out)
    ref_bf16 = ref_prefill_logits(q_bf16, kv_bf16, weights, rb, ls, le, max_seq_len, weight_scale)
    cos_bf16 = _cos(out[m], ref_bf16[m]).item()
    us_ours = measure(run_ours)

    print(
        f"\n=== bench: bs={bs} n_q={n_q} ctx={ctx} total_tokens={total_tokens} "
        f"n_ctas={n_ctas} tiles/row={(ctx + block_k - 1)//block_k} "
        f"mode={'graph' if graph else 'eager'} (cos_bf16={cos_bf16:.5f}) ==="
    )
    rows = [("ours fp8 paged (single kernel)", us_ours)]

    # ---- FlyDSL fp4 paged, same bf16 data + same workload ----
    try:
        import test_flydsl_pa_mqa_logits_fp4_prefill as fp4t
        from aiter.ops.flydsl import flydsl_pa_mqa_logits_fp4_prefill
        from aiter.ops.flydsl.kernels.pa_mqa_logits_fp4_prefill import (
            compute_prefill_schedule as fp4_sched,
        )

        k_flat = kv_bf16.reshape(bs * t_max, head_dim)
        tb = torch.arange(bs, device=dev).repeat_interleave(t_max)
        tt = torch.arange(t_max, device=dev).repeat(bs)
        phys = block_tables[tb, tt // kv_block_size].long()
        slot_mapping = (phys * kv_block_size + (tt % kv_block_size)).to(torch.int32)
        num_blocks = block_tables.numel()
        k_tiles = head_dim // 128
        kv4 = torch.zeros(num_blocks, k_tiles, 4, kv_block_size, 16, dtype=torch.uint8, device=dev)
        kvs4 = torch.zeros(num_blocks, k_tiles, 4, kv_block_size, dtype=torch.uint8, device=dev)
        fp4t.indexer_k_fp4_paged_preshuffle(k_flat, slot_mapping, kv4, kvs4, kv_block_size)
        q_fp4, q_scale4 = fp4t.quant_q_fp4_preshuffle(q_bf16)
        _, cta4, n4 = fp4_sched(rb, ls, le, block_k, parallel_unit_num, max_seq_len)
        out4 = torch.full((total_tokens, max_seq_len), float("-inf"), dtype=torch.float32, device=dev)

        def run_fp4():
            flydsl_pa_mqa_logits_fp4_prefill(
                q_fp4, q_scale4, kv4, kvs4, block_tables, weights, rb, ls, le,
                max_seq_len, weight_scale=weight_scale, block_k=block_k,
                kv_block_size=kv_block_size, parallel_unit_num=parallel_unit_num,
                out=out4, cta_info=cta4, n_ctas=n4,
            )

        run_fp4()
        torch.cuda.synchronize()
        us_fp4 = measure(run_fp4)
        rows.append(("flydsl fp4 paged (single kernel)", us_fp4))
    except Exception as e:  # noqa: BLE001
        print(f"  [perf] FlyDSL fp4 path unavailable ({type(e).__name__}: {e})")
        us_fp4 = None

    # ---- ATOM: cp_gather + Triton fp8_mqa_logits (the ~74us baseline) ----
    try:
        from aiter import cp_gather_indexer_k_quant_cache, dtypes, indexer_k_quant_and_cache
        from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits

        num_blocks = block_tables.numel()
        k_flat = kv_bf16.reshape(bs * t_max, head_dim)
        tb = torch.arange(bs, device=dev).repeat_interleave(t_max)
        tt = torch.arange(t_max, device=dev).repeat(bs)
        phys = block_tables[tb, tt // kv_block_size].long()
        slot_mapping = (phys * kv_block_size + (tt % kv_block_size)).to(torch.int64)
        cu = torch.arange(0, (bs + 1) * ctx, ctx, dtype=torch.int32, device=dev)
        total_committed = bs * ctx
        kv_cache_fp8 = torch.zeros((num_blocks, kv_block_size, head_dim + 4), dtype=dtypes.fp8, device=dev)
        indexer_k_quant_and_cache(k_flat, kv_cache_fp8, slot_mapping, head_dim, "ue8m0", True)
        dst_k = torch.empty((total_committed, head_dim), dtype=dtypes.fp8, device=dev)
        dst_scale = torch.empty((total_committed, 1), dtype=torch.float32, device=dev)
        q_fp8_atom = q_bf16.to(dtypes.fp8)
        cu_starts = (rb.to(torch.int64) * ctx + ls.to(torch.int64)).to(torch.int32)
        cu_ends = (rb.to(torch.int64) * ctx + le.to(torch.int64)).to(torch.int32)

        def atom_logits():
            cp_gather_indexer_k_quant_cache(
                kv_cache_fp8, dst_k, dst_scale.view(dtypes.fp8), block_tables, cu, True
            )
            return fp8_mqa_logits(
                q_fp8_atom, dst_k, dst_scale, weights.float(), cu_starts, cu_ends, clean_logits=False
            )

        atom_logits()
        torch.cuda.synchronize()
        us_atom = measure(atom_logits)
        rows.append(("ATOM cp_gather+fp8_logits", us_atom))
    except Exception as e:  # noqa: BLE001
        print(f"  [perf] ATOM path unavailable ({type(e).__name__}: {e})")

    print("\n  {:<34} | {:>10}".format("path", "us"))
    print("  " + "-" * 48)
    for name, us in rows:
        print("  {:<34} | {:>10.2f}".format(name, us))
    if us_fp4:
        print(f"\n  ours/fp4 = {us_ours / us_fp4:.3f}x (｜1.0 = parity with fp4)")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=5120)
    ap.add_argument("--n_q", type=int, default=256)
    ap.add_argument("--pun", type=int, default=512)
    ap.add_argument("--graph", action="store_true", help="time with CUDA graph instead of eager")
    args = ap.parse_args()

    if args.bench:
        return run_bench(args.bs, args.n_q, args.ctx, parallel_unit_num=args.pun,
                         graph=args.graph)

    print("=== MXFP8 paged MQA logits (opus, gfx950) ===")
    rc = 0
    # ragged prefill windows (mix zero / non-zero lower bounds, short + long)
    rc |= run_prefill_case(2, [[(0, 50), (0, 120), (0, 200)], [(0, 40), (0, 100)]], seed=0)
    rc |= run_prefill_case(3, [[(0, 30)], [(0, 200)], [(0, 100), (0, 150)]], seed=2)
    rc |= run_prefill_case(2, [[(10, 50), (64, 200)], [(0, 100), (130, 256)]], seed=4)
    rc |= run_prefill_case(2, [[(0, 1500)], [(0, 1200), (300, 1400)]], parallel_unit_num=4, seed=6)
    rc |= run_prefill_case(2, [[(0, 2048)], [(0, 4096)]], seed=8)
    rc |= run_prefill_case(2, [[(0, 512), (0, 1024), (0, 1536)], [(0, 2000)]], seed=10)
    rc |= run_prefill_case(2, [[(100, 2048), (512, 4096)], [(0, 8192)]], seed=12)
    rc |= run_prefill_case(2, [[(0, 3000), (1000, 5000)], [(0, 6000), (2048, 6144)]], seed=14)
    rc |= run_prefill_case(2, [[(0, 4096)], [(0, 4096)]], parallel_unit_num=4, seed=16)

    # varqlen / MTP (per-batch query length via qlens, tail-causal; may be ragged)
    rc |= run_varqlen_case(3, [1, 1, 1], [200, 512, 1000], seed=20)
    rc |= run_varqlen_case(2, [2, 2], [300, 1500], seed=22)
    rc |= run_varqlen_case(4, [4, 4, 4, 4], [256, 800, 2048, 4096], seed=24)
    # fully ragged per-batch query length (no single next_n)
    rc |= run_varqlen_case(3, [1, 3, 2], [200, 1500, 800], seed=26)

    print("  ALL PASS" if rc == 0 else "  SOME FAILED")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
