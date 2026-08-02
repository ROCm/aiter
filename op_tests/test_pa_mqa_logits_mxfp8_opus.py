# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter op-test + perf sweep for the MXFP8 paged MQA logits OPUS kernel (gfx950).

Follows the aiter op_test standard (.claude/skills/aiter-op-test): two ``@benchmark``
sweep functions -- ``test_prefill`` (ragged causal windows) and ``test_decode`` (MTP
tail-causal) -- each is BOTH a correctness check and a perf sweep, ending in one
markdown summary table apiece. Candidates: ``ours`` (MXFP8 paged) vs ``triton`` (fp8).
Per candidate the table records ``us`` / ``TFLOPS`` / ``TB/s`` / ``err``.

Per query row ``r`` over window ``[s, e)``:
    out[r, s:e] = sum_H( relu(Q[r] . Kᵀ) * weight[r] ) * weight_scale.
Q/KV are MXFP8-quantized + preshuffled on the host into the kernel ABI (E4M3 data +
E8M0 block scales, block=32). setup (quant/preshuffle/gather/window build/out alloc)
is done OUTSIDE the timed region; only the kernel launch is timed.

Correctness note: a full dense torch reference at model scale (prefill sums to 16384
query rows over up to 16k ctx) is infeasible, so ``err`` is ``checkAllclose`` over a
handful of randomly sampled query rows vs an fp32 torch reference built from the same
FP8-dequantized inputs the kernel consumes (``ours`` -> ~0; ``triton`` reuses the ref
with a looser ratio, so its ``err`` also folds in triton's own fp8 quant distance).

A short ragged / non-4-aligned / varqlen correctness pass (small shapes, full-tensor
``checkAllclose``) runs first -- it guards the out-store alignment fix (see
gcnasm/opus_logits/KNOWN_ISSUE_out_store_alignment.md) which the perf sweep shapes
do not exercise.

Use an IDLE GPU (contention skews the small decode kernels).

    python op_tests/test_pa_mqa_logits_mxfp8_opus.py                    # both sweeps
    python op_tests/test_pa_mqa_logits_mxfp8_opus.py --scenario prefill
"""

import argparse
import itertools
import random

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.opus.pa_mqa_logits_mxfp8_opus import (
    compute_prefill_windows,
    pa_mqa_logits_mxfp8_fwd_decode,
    pa_mqa_logits_mxfp8_fwd_prefill,
)
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

torch.set_default_device("cuda")

dev = "cuda"
SUPPORTED_GFX = ["gfx950"]

# ── op geometry / kernel ABI constants ───────────────────────────────
HEADS = 64
HEAD_DIM = 128
KV_BLOCK_SIZE = 64
SCALE_BLOCK = 32
MFMA_M = 16
FP8_E4M3_MAX = 448.0
KVS_NTPW = 4
WEIGHT_SCALE = 1.5

# ── sweep defaults (model-derived) ───────────────────────────────────
PREFILL_BLOCK_K = 64  # prefill -> 1-warp variant
PREFILL_TOTAL_QLEN = 16384  # sum of per-batch qlen (== ctx, causal)
PREFILL_QMIN = 800
DECODE_BLOCK_K = 64  # decode is 1-warp only
DECODE_CTA_TARGET = 1024  # context-split fill target for the 1-warp decode
TRITON_DECODE_CHUNKK = 256  # triton deepgemm's own KV tiling
N_COS_SAMPLE = 8  # sampled query rows used for the correctness check

# run_perftest knobs (set in main(); kept off the test-fn signature so the table's
# left columns stay the shape params only).
_ITERS = 100
_WARMUP = 20
_SEED = 0
_CHECK = True


# ═════════════════════════════════════════════════════════════════════
# MXFP8 (E4M3 + E8M0) quant / dequant + host-side preshuffle writers
# ═════════════════════════════════════════════════════════════════════


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


# ═════════════════════════════════════════════════════════════════════
# Torch reference (NOT timed, NOT in the table)
# ═════════════════════════════════════════════════════════════════════


def ref_row_logits(q_dq, kv_dq, weights, b, s, e, ws):
    """fp32 logits for one query row over [s, e): sum_H(relu(q·kᵀ)·w) · ws -> [e-s]."""
    qk = q_dq.float() @ kv_dq[b, s:e].float().T
    qk = torch.relu(qk) * weights.float()[:, None]
    return qk.sum(dim=0) * ws


def ref_full_prefill(q_dq, kv_dq, weights, rb, ls, le, max_seq_len, ws):
    """Dense fp32 reference over ALL rows (only feasible for the small corner cases)."""
    total = q_dq.shape[0]
    out = torch.full(
        (total, max_seq_len), float("-inf"), device=dev, dtype=torch.float32
    )
    for r in range(total):
        b, s, e = int(rb[r]), int(ls[r]), int(le[r])
        if e > s:
            out[r, s:e] = ref_row_logits(q_dq[r], kv_dq, weights[r], b, s, e, ws)
    return out


def sample_rows(total, rb, ls, le, n=N_COS_SAMPLE, seed=0):
    """Pick up to ``n`` random rows with a non-empty window -> [(r, b, s, e), ...]."""
    g = random.Random(seed)
    picks = []
    for r in g.sample(range(total), min(n * 4, total)):
        s, e, b = int(ls[r]), int(le[r]), int(rb[r])
        if e > s:
            picks.append((r, b, s, e))
        if len(picks) >= n:
            break
    return picks


def ref_sampled(q_dq, kv_dq, weights, samples, ws):
    """Concatenated fp32 reference over the sampled rows' in-window cells."""
    if not samples:
        return torch.zeros(0, device=dev, dtype=torch.float32)
    return torch.cat(
        [
            ref_row_logits(q_dq[r], kv_dq, weights[r], b, s, e, ws)
            for r, b, s, e in samples
        ]
    )


def candidate_err(name, out, extract, samples, ref_fp8, ref_bf16, tag):
    """checkAllclose one candidate's sampled output. ``ours`` consumes the same
    FP8-dequant inputs as ``ref_fp8`` -> bit-exact (err ~ 0, the tight gate). A
    reference kernel (triton) uses its own fp8 quant, so it's scored vs the bf16
    ground truth ``ref_bf16`` with an fp8-appropriate tolerance -- an accuracy
    number, not a bit-match. Returns the mismatch fraction (0 = all close)."""
    if not samples:
        return None
    if name == "ours":
        return checkAllclose(
            ref_fp8,
            extract(out),
            rtol=2e-2,
            atol=2e-2,
            tol_err_ratio=0.05,
            msg=f"ours: {tag}",
        )
    return checkAllclose(
        ref_bf16,
        extract(out),
        rtol=6e-2,
        atol=6e-2,
        tol_err_ratio=0.9,
        msg=f"{name} (vs bf16): {tag}",
    )


# ═════════════════════════════════════════════════════════════════════
# Shared input builders (paged preshuffled KV + Q), mirror the model layout
# ═════════════════════════════════════════════════════════════════════


def build_paged_kv(bs, max_end, kv_block_size, block_k, seed):
    """bf16 KV -> (kv_bf16, kv_dq, kv_cache fp8, kv_scale, block_tables, t_max, max_seq_len)."""
    torch.manual_seed(seed)
    max_blocks_per_seq = max(
        (max_end + block_k - 1) // block_k * (block_k // kv_block_size),
        block_k // kv_block_size,
    )
    t_max = max_blocks_per_seq * kv_block_size
    max_seq_len = t_max
    num_blocks = max_blocks_per_seq * bs

    kv_bf16 = torch.randn(bs, t_max, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=dev).reshape(
        bs, max_blocks_per_seq
    )
    kv_fp8_d, kv_e8_d = fp8_quant_e4m3_with_e8m0(kv_bf16.reshape(-1, HEAD_DIM))
    kv_dq = fp8_dequant_e4m3_with_e8m0(
        kv_fp8_d.reshape(bs, t_max, HEAD_DIM),
        kv_e8_d.reshape(bs, t_max, HEAD_DIM // 32),
    )

    k_flat = kv_bf16.reshape(bs * t_max, HEAD_DIM)
    tb = torch.arange(bs, device=dev).repeat_interleave(t_max)
    tt = torch.arange(t_max, device=dev).repeat(bs)
    phys = block_tables[tb, tt // kv_block_size].long()
    slot_mapping = (phys * kv_block_size + (tt % kv_block_size)).to(torch.int32)
    k_tiles = HEAD_DIM // 128
    kv_cache = torch.zeros(
        num_blocks, k_tiles, 8, kv_block_size, 16, dtype=torch.uint8, device=dev
    )
    kv_scale = torch.zeros(
        num_blocks, k_tiles, 4, kv_block_size, dtype=torch.uint8, device=dev
    )
    indexer_k_fp8_paged_preshuffle(
        k_flat, slot_mapping, kv_cache, kv_scale, kv_block_size
    )
    return kv_bf16, kv_dq, kv_cache, kv_scale, block_tables, t_max, max_seq_len


def build_q(total_tokens):
    q_bf16 = torch.randn(
        total_tokens, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=dev
    )
    weights = (
        torch.randn(total_tokens, HEADS, dtype=torch.float32, device=dev) * 0.1
    ).to(torch.bfloat16)
    q_fp8, q_scale = quant_q_fp8_preshuffle(q_bf16)
    q_e8 = fp8_quant_e4m3_with_e8m0(q_bf16.reshape(total_tokens * HEADS, HEAD_DIM))[
        1
    ].reshape(total_tokens, HEADS, HEAD_DIM // 32)
    q_dq = fp8_dequant_e4m3_with_e8m0(q_fp8, q_e8)
    return q_bf16, q_fp8, q_scale, q_dq, weights


# ═════════════════════════════════════════════════════════════════════
# Triton reference candidates (setup outside timing; return timed closure + extractor)
# ═════════════════════════════════════════════════════════════════════


def make_triton_prefill(kv_bf16, block_tables, q_bf16, weights, ctxs, samples):
    """Triton prefill candidate: paged->contiguous gather (setup) + fp8_mqa_logits.
    Returns (launch_fn, extract_fn) or None if triton is unavailable."""
    try:
        from aiter import (
            cp_gather_indexer_k_quant_cache,
            indexer_k_quant_and_cache,
        )
        from aiter.ops.triton.attention.fp8_mqa_logits import fp8_mqa_logits
    except Exception as e:  # noqa: BLE001
        print(f"    [triton prefill unavailable] {type(e).__name__}: {e}")
        return None

    bs, t_max, head_dim = kv_bf16.shape
    num_blocks = block_tables.numel()
    k_flat = kv_bf16.reshape(bs * t_max, head_dim)
    tb = torch.arange(bs, device=dev).repeat_interleave(t_max)
    tt = torch.arange(t_max, device=dev).repeat(bs)
    phys = block_tables[tb, tt // KV_BLOCK_SIZE].long()
    slot_mapping = (phys * KV_BLOCK_SIZE + (tt % KV_BLOCK_SIZE)).to(torch.int64)

    kv_cache_fp8 = torch.zeros(
        (num_blocks, KV_BLOCK_SIZE, head_dim + 4), dtype=dtypes.fp8, device=dev
    )
    indexer_k_quant_and_cache(
        k_flat, kv_cache_fp8, slot_mapping, head_dim, "ue8m0", True
    )

    # committed context per batch == full seq (qlen == ctx); contiguous prefix.
    cu = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.tensor(ctxs, dtype=torch.int32, device=dev).cumsum(0)
    total_committed = int(cu[-1].item())
    dst_k = torch.empty((total_committed, head_dim), dtype=dtypes.fp8, device=dev)
    dst_scale = torch.empty((total_committed, 1), dtype=torch.float32, device=dev)

    q_fp8_atom = q_bf16.to(dtypes.fp8)
    # kernel needs cu_starts/cu_ends over ALL rows (causal [0, n+1)) in gathered K space.
    rb_all, le_all = [], []
    for b in range(bs):
        n_rows = int(ctxs[b])
        rb_all += [b] * n_rows
        le_all += list(range(1, n_rows + 1))
    rb_all_t = torch.tensor(rb_all, dtype=torch.int64, device=dev)
    le_all_t = torch.tensor(le_all, dtype=torch.int32, device=dev)
    cu_starts = cu[rb_all_t].to(torch.int32)
    cu_ends = (cu[rb_all_t] + le_all_t).to(torch.int32)
    # triton's kernel has no weight_scale scalar; fold ours' into the per-head weights.
    w_f32 = weights.float() * WEIGHT_SCALE

    # gather is one-time prep (not timed here); run it once for correctness + warm.
    cp_gather_indexer_k_quant_cache(
        kv_cache_fp8, dst_k, dst_scale.view(dtypes.fp8), block_tables, cu, True
    )
    torch.cuda.synchronize()

    def launch():
        return fp8_mqa_logits(
            q_fp8_atom, dst_k, dst_scale, w_f32, cu_starts, cu_ends, clean_logits=False
        )

    cu_cpu = cu.tolist()

    def extract(out):
        # out[r] is indexed in gathered contiguous K space: window [cu[b], cu[b]+e).
        return torch.cat(
            [out[r, cu_cpu[b] + s : cu_cpu[b] + e] for r, b, s, e in samples]
        )

    return launch, extract


def make_triton_decode(
    kv_bf16,
    block_tables,
    q_bf16_dec,
    weights,
    context_lens,
    t_max,
    num_blocks,
    next_n,
    samples,
):
    """Triton decode candidate: deepgemm reads paged KV directly (no host prep).
    Returns (launch_fn, extract_fn) or None if unavailable."""
    try:
        from aiter.ops.shuffle import shuffle_weight
        from aiter.ops.triton.attention.pa_mqa_logits import (
            deepgemm_fp8_paged_mqa_logits,
        )
        from aiter.ops.triton.utils.types import get_fp8_e4m3_dtype
    except Exception as e:  # noqa: BLE001
        print(f"    [triton decode unavailable] {type(e).__name__}: {e}")
        return None

    fp8_dtype = get_fp8_e4m3_dtype()
    batch_size = q_bf16_dec.shape[0]
    kbs = KV_BLOCK_SIZE

    kv_blocks = kv_bf16.reshape(num_blocks, kbs, 1, HEAD_DIM)
    sf = kv_blocks.abs().float().amax(dim=3, keepdim=True).clamp(1e-4) / 240.0
    x_scaled = (kv_blocks * (1.0 / sf)).to(fp8_dtype)

    index_dim = HEAD_DIM + 4  # deepgemm layout: [nb, block, 1, D + 4B fp32 scale]
    kv_cache_fp8 = torch.empty(
        (num_blocks, kbs * index_dim), dtype=torch.uint8, device=dev
    )
    kv_cache_fp8[:, : kbs * HEAD_DIM] = x_scaled.reshape(
        num_blocks, kbs * HEAD_DIM
    ).view(torch.uint8)
    kv_cache_fp8[:, kbs * HEAD_DIM :] = sf.reshape(num_blocks, kbs).view(torch.uint8)
    kv_cache_fp8 = kv_cache_fp8.view(num_blocks, kbs, 1, index_dim)

    split = kv_cache_fp8.view(num_blocks, kbs * index_dim)
    data = shuffle_weight(
        split[:, : kbs * HEAD_DIM].contiguous().view(num_blocks, kbs, HEAD_DIM)
    )
    split[:, : kbs * HEAD_DIM] = data.reshape(num_blocks, kbs * HEAD_DIM)

    q_fp8 = q_bf16_dec.to(fp8_dtype).contiguous()  # [B, next_n, H, D]
    # triton's kernel has no weight_scale scalar; fold ours' into the per-head weights.
    w_fp32 = (weights.float() * WEIGHT_SCALE).contiguous()
    out_fp8 = torch.full(
        (batch_size * next_n, t_max), float("-inf"), dtype=torch.float32, device=dev
    )

    def launch():
        deepgemm_fp8_paged_mqa_logits(
            q_fp8,
            kv_cache_fp8,
            w_fp32,
            out_fp8,
            context_lens,
            block_tables,
            t_max,
            ChunkK=TRITON_DECODE_CHUNKK,
            Preshuffle=True,
            KVBlockSize=kbs,
            WavePerEU=2,
        )
        return out_fp8

    try:
        launch()
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001
        print(f"    [triton decode failed] {type(e).__name__}: {e}")
        return None

    def extract(out):
        return torch.cat([out[r, s:e] for r, _, s, e in samples])

    return launch, extract


# ═════════════════════════════════════════════════════════════════════
# Roofline metrics
# ═════════════════════════════════════════════════════════════════════


def roofline(total_q, n_logits):
    """MQA logits does, per in-window logit, a Q[H,D]·k[D] mul-add over H heads.
    FLOPs = 2 * H * D * n_logits         (the QKᵀ mul-add dominates relu + head-sum)
    bytes = Q(fp8) read + K(fp8) read (once per row window) + out(fp32) write
    """
    flops = 2 * HEADS * HEAD_DIM * n_logits
    nbytes = total_q * HEADS * HEAD_DIM * 1 + n_logits * HEAD_DIM * 1 + n_logits * 4
    return flops, nbytes


def gen_prefill_qlens(bs, total=PREFILL_TOTAL_QLEN, qmin=PREFILL_QMIN, seed=0):
    """Split ``total`` into ``bs`` per-batch qlens, each >= qmin, sum == total."""
    assert bs * qmin <= total, f"bs={bs} * qmin={qmin} > total={total}"
    g = random.Random(seed)
    extra = total - bs * qmin
    w = [g.random() for _ in range(bs)]
    s = sum(w) or 1.0
    parts = [qmin + int(extra * wi / s) for wi in w]
    parts[0] += total - sum(parts)
    assert min(parts) >= qmin and sum(parts) == total
    return parts


def gen_decode_ctxs(batch, max_ctx, kv_block_size=KV_BLOCK_SIZE, seed=0):
    """Per-batch ctx in [0.9*max_ctx, max_ctx], rounded up to kv_block_size."""
    low = int(0.9 * max_ctx)
    g = torch.Generator(device=dev).manual_seed(seed)
    raw = torch.randint(low, max_ctx + 1, (batch,), generator=g, device=dev).tolist()
    cap = (max_ctx // kv_block_size) * kv_block_size
    return [
        min(((c + kv_block_size - 1) // kv_block_size) * kv_block_size, cap)
        for c in raw
    ]


# ═════════════════════════════════════════════════════════════════════
# @benchmark sweep functions -- one table each
# ═════════════════════════════════════════════════════════════════════


@benchmark()
def test_prefill(bs):
    seed = _SEED + bs
    qlens = gen_prefill_qlens(bs, seed=seed)  # per-batch qlen == ctx (causal)
    ctxs = list(qlens)
    total_q = sum(qlens)

    kv_bf16, kv_dq, kv_cache, kv_scale, block_tables, _t_max, max_seq_len = (
        build_paged_kv(bs, max(ctxs), KV_BLOCK_SIZE, PREFILL_BLOCK_K, seed)
    )
    cu = torch.tensor(
        [0] + list(itertools.accumulate(qlens)), dtype=torch.int32, device=dev
    )
    context_lens = torch.tensor(ctxs, dtype=torch.int32, device=dev)
    q_bf16, q_fp8, q_scale, q_dq, weights = build_q(total_q)
    q_fp8_v = q_fp8.view(torch.float8_e4m3fn)
    kv_cache_v = kv_cache.view(torch.float8_e4m3fn)

    # per-row windows (device build, cudagraph-safe) -- setup, not timed.
    rb, ls, le = compute_prefill_windows(cu, context_lens, total_q)
    n_logits = int((le - ls).clamp(min=0).sum().item())
    flops, nbytes = roofline(total_q, n_logits)

    samples = sample_rows(total_q, rb, ls, le, seed=seed) if _CHECK else []
    ref_fp8 = (
        ref_sampled(q_dq, kv_dq, weights, samples, WEIGHT_SCALE) if samples else None
    )
    ref_bf16 = (
        ref_sampled(q_bf16, kv_bf16, weights, samples, WEIGHT_SCALE)
        if samples
        else None
    )

    out = torch.full(
        (total_q, max_seq_len), float("-inf"), dtype=torch.float32, device=dev
    )

    def ours_launch():
        pa_mqa_logits_mxfp8_fwd_prefill(
            q_fp8_v,
            q_scale,
            kv_cache_v,
            kv_scale,
            block_tables,
            weights,
            rb,
            ls,
            le,
            out,
            total_q,
            WEIGHT_SCALE,
            PREFILL_BLOCK_K,
            KV_BLOCK_SIZE,
            max_seq_len,
        )
        return out

    def ours_extract(o):
        return torch.cat([o[r, s:e] for r, _, s, e in samples])

    candidates = {"ours": (ours_launch, ours_extract)}
    tri = make_triton_prefill(kv_bf16, block_tables, q_bf16, weights, ctxs, samples)
    if tri is not None:
        candidates["triton"] = tri

    ret = {
        "gfx": get_gfx(),
        "total_q": total_q,
        "max_seq_len": max_seq_len,
        "n_ctas": total_q,
    }
    for name, (launch, extract) in candidates.items():
        launch()
        torch.cuda.synchronize()
        o, us = run_perftest(launch, num_iters=_ITERS, num_warmup=_WARMUP)
        err = candidate_err(
            name, o, extract, samples, ref_fp8, ref_bf16, f"prefill logits (bs={bs})"
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err

    torch.cuda.empty_cache()
    return ret


@benchmark()
def test_decode(batch, max_ctx, next_n):
    seed = _SEED + batch + max_ctx + next_n
    ctxs = gen_decode_ctxs(batch, max_ctx, KV_BLOCK_SIZE, seed=seed)
    total_q = batch * next_n

    kv_bf16, kv_dq, kv_cache, kv_scale, block_tables, t_max, max_seq_len = (
        build_paged_kv(batch, max(ctxs), KV_BLOCK_SIZE, DECODE_BLOCK_K, seed)
    )
    num_blocks = block_tables.numel()
    cu = torch.arange(0, (batch + 1) * next_n, next_n, dtype=torch.int32, device=dev)
    context_lens = torch.tensor(ctxs, dtype=torch.int32, device=dev)
    q_bf16, q_fp8, q_scale, q_dq, weights = build_q(total_q)
    q_fp8_v = q_fp8.view(torch.float8_e4m3fn)
    kv_cache_v = kv_cache.view(torch.float8_e4m3fn)

    # context-split only when the query rows alone can't fill the GPU (schedule-free).
    max_chunks = max(1, (max_seq_len + DECODE_BLOCK_K - 1) // DECODE_BLOCK_K)
    split_kv = (
        1
        if total_q >= DECODE_CTA_TARGET
        else min(max_chunks, (DECODE_CTA_TARGET + total_q - 1) // total_q)
    )
    n_ctas = split_kv * next_n * batch

    # MTP tail-causal windows (device build) -- for roofline + correctness only.
    rb, ls, le = compute_prefill_windows(cu, context_lens, total_q)
    n_logits = int((le - ls).clamp(min=0).sum().item())
    flops, nbytes = roofline(total_q, n_logits)

    samples = sample_rows(total_q, rb, ls, le, seed=seed) if _CHECK else []
    ref_fp8 = (
        ref_sampled(q_dq, kv_dq, weights, samples, WEIGHT_SCALE) if samples else None
    )
    ref_bf16 = (
        ref_sampled(q_bf16, kv_bf16, weights, samples, WEIGHT_SCALE)
        if samples
        else None
    )

    out = torch.full(
        (total_q, max_seq_len), float("-inf"), dtype=torch.float32, device=dev
    )

    def ours_launch():
        pa_mqa_logits_mxfp8_fwd_decode(
            q_fp8_v,
            q_scale,
            kv_cache_v,
            kv_scale,
            block_tables,
            weights,
            cu,
            context_lens,
            out,
            batch,
            next_n,
            split_kv,
            WEIGHT_SCALE,
            DECODE_BLOCK_K,
            KV_BLOCK_SIZE,
            max_seq_len,
        )
        return out

    def ours_extract(o):
        return torch.cat([o[r, s:e] for r, _, s, e in samples])

    candidates = {"ours": (ours_launch, ours_extract)}
    q_bf16_dec = q_bf16.reshape(batch, next_n, HEADS, HEAD_DIM).contiguous()
    tri = make_triton_decode(
        kv_bf16,
        block_tables,
        q_bf16_dec,
        weights,
        context_lens,
        t_max,
        num_blocks,
        next_n,
        samples,
    )
    if tri is not None:
        candidates["triton"] = tri

    ret = {
        "gfx": get_gfx(),
        "total_q": total_q,
        "max_seq_len": max_seq_len,
        "n_ctas": n_ctas,
    }
    for name, (launch, extract) in candidates.items():
        launch()
        torch.cuda.synchronize()
        o, us = run_perftest(launch, num_iters=_ITERS, num_warmup=_WARMUP)
        err = candidate_err(
            name,
            o,
            extract,
            samples,
            ref_fp8,
            ref_bf16,
            f"decode logits (b={batch} mtp={next_n} ctx~{max_ctx})",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err

    torch.cuda.empty_cache()
    return ret


# ═════════════════════════════════════════════════════════════════════
# Corner-case correctness (small shapes, full-tensor checkAllclose)
# Guards ragged / non-4-aligned / varqlen paths the perf sweep never exercises.
# ═════════════════════════════════════════════════════════════════════


def _check_prefill_case(bs, windows_per_batch, seed):
    max_end = max(
        (w if isinstance(w, int) else w[1]) for ws in windows_per_batch for w in ws
    )
    _kv_bf16, kv_dq, kv_cache, kv_scale, block_tables, _t_max, max_seq_len = (
        build_paged_kv(bs, max_end, KV_BLOCK_SIZE, PREFILL_BLOCK_K, seed)
    )
    rb, ls, le = [], [], []
    for b in range(bs):
        for w in windows_per_batch[b]:
            s, e = (0, w) if isinstance(w, int) else (w[0], w[1])
            rb.append(b)
            ls.append(s)
            le.append(e)
    total = len(rb)
    _q_bf16, q_fp8, q_scale, q_dq, weights = build_q(total)

    ref = ref_full_prefill(q_dq, kv_dq, weights, rb, ls, le, max_seq_len, WEIGHT_SCALE)
    from aiter.ops.opus.pa_mqa_logits_mxfp8_opus import pa_mqa_logits_mxfp8_prefill

    out = pa_mqa_logits_mxfp8_prefill(
        q_fp8.view(torch.float8_e4m3fn),
        q_scale,
        kv_cache.view(torch.float8_e4m3fn),
        kv_scale,
        block_tables,
        weights,
        torch.tensor(rb, dtype=torch.int32, device=dev),
        torch.tensor(ls, dtype=torch.int32, device=dev),
        torch.tensor(le, dtype=torch.int32, device=dev),
        max_seq_len,
        weight_scale=WEIGHT_SCALE,
        block_k=PREFILL_BLOCK_K,
        kv_block_size=KV_BLOCK_SIZE,
    )
    torch.cuda.synchronize()
    m = ~torch.isneginf(ref)
    err = checkAllclose(
        ref[m],
        out[m],
        rtol=2e-2,
        atol=2e-2,
        tol_err_ratio=0.05,
        msg=f"prefill corner bs={bs} tt={total}",
    )
    oob_ok = bool(torch.isneginf(out[~m]).all().item()) if (~m).any() else True
    ok = err < 0.05 and oob_ok
    print(
        f"  [prefill] bs={bs} tt={total} err={err:.4f} oob_neginf={oob_ok} "
        f"{'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


def _check_varqlen_case(bs, qlens, context_lens, seed):
    from aiter.ops.opus.pa_mqa_logits_mxfp8_opus import pa_mqa_logits_mxfp8_decode

    _kv_bf16, kv_dq, kv_cache, kv_scale, block_tables, _t_max, max_seq_len = (
        build_paged_kv(bs, max(context_lens), KV_BLOCK_SIZE, DECODE_BLOCK_K, seed)
    )
    cu = [0]
    for q in qlens:
        cu.append(cu[-1] + int(q))
    total_q = cu[-1]
    cu_seq_q = torch.tensor(cu, dtype=torch.int32, device=dev)
    ctx = torch.tensor(context_lens, dtype=torch.int32, device=dev)

    rb, ls, le = [], [], []
    for b in range(bs):
        ql = int(qlens[b])
        for n in range(ql):
            rb.append(b)
            ls.append(0)
            le.append(max(int(context_lens[b]) - (ql - 1 - n), 0))
    _q_bf16, q_fp8, q_scale, q_dq, weights = build_q(total_q)
    ref = ref_full_prefill(q_dq, kv_dq, weights, rb, ls, le, max_seq_len, WEIGHT_SCALE)

    out = pa_mqa_logits_mxfp8_decode(
        q_fp8.view(torch.float8_e4m3fn),
        q_scale,
        kv_cache.view(torch.float8_e4m3fn),
        kv_scale,
        block_tables,
        weights,
        ctx,
        max_seq_len,
        max(int(q) for q in qlens),
        split_ctx_len=max_seq_len,
        cu_seq_q=cu_seq_q,
        weight_scale=WEIGHT_SCALE,
        kv_block_size=KV_BLOCK_SIZE,
    )
    torch.cuda.synchronize()
    m = ~torch.isneginf(ref)
    err = checkAllclose(
        ref[m],
        out[m],
        rtol=2e-2,
        atol=2e-2,
        tol_err_ratio=0.05,
        msg=f"varqlen corner bs={bs} total_q={total_q}",
    )
    ok = err < 0.05
    print(
        f"  [varqlen] bs={bs} total_q={total_q} err={err:.4f} {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


def _check_fixed_mtp_case(bs, next_n, context_lens, seed):
    """Fixed-MTP decode: every batch has exactly ``next_n`` tokens (uniform qlen),
    exercising the ``cu_seq_q=None`` path of the decode wrapper (uniform cu_seq_q
    built in-kernel). Windows are tail-causal: row (b, n) -> [0, ctx_b-(next_n-1-n))."""
    from aiter.ops.opus.pa_mqa_logits_mxfp8_opus import pa_mqa_logits_mxfp8_decode

    _kv_bf16, kv_dq, kv_cache, kv_scale, block_tables, _t_max, max_seq_len = (
        build_paged_kv(bs, max(context_lens), KV_BLOCK_SIZE, DECODE_BLOCK_K, seed)
    )
    total_q = bs * next_n
    ctx = torch.tensor(context_lens, dtype=torch.int32, device=dev)

    rb, ls, le = [], [], []
    for b in range(bs):
        for n in range(next_n):
            rb.append(b)
            ls.append(0)
            le.append(max(int(context_lens[b]) - (next_n - 1 - n), 0))
    _q_bf16, q_fp8, q_scale, q_dq, weights = build_q(total_q)
    ref = ref_full_prefill(q_dq, kv_dq, weights, rb, ls, le, max_seq_len, WEIGHT_SCALE)

    out = pa_mqa_logits_mxfp8_decode(  # cu_seq_q omitted -> fixed-MTP (uniform) path
        q_fp8.view(torch.float8_e4m3fn),
        q_scale,
        kv_cache.view(torch.float8_e4m3fn),
        kv_scale,
        block_tables,
        weights,
        ctx,
        max_seq_len,
        next_n,
        split_ctx_len=max_seq_len,
        weight_scale=WEIGHT_SCALE,
        kv_block_size=KV_BLOCK_SIZE,
    )
    torch.cuda.synchronize()
    m = ~torch.isneginf(ref)
    err = checkAllclose(
        ref[m],
        out[m],
        rtol=2e-2,
        atol=2e-2,
        tol_err_ratio=0.05,
        msg=f"fixed-mtp corner bs={bs} next_n={next_n}",
    )
    ok = err < 0.05
    print(
        f"  [fixed-mtp] bs={bs} next_n={next_n} total_q={total_q} ctx={list(context_lens)} "
        f"err={err:.4f} {'PASS' if ok else 'FAIL'}"
    )
    return 0 if ok else 1


def run_corner_correctness():
    print(
        "=== MXFP8 paged MQA logits correctness "
        "(ragged / unaligned / kv-tile boundary / varqlen / fixed-MTP) ==="
    )
    rc = 0
    # ragged windows (mix zero / non-zero lower bounds, short + long)
    rc |= _check_prefill_case(
        2, [[(0, 50), (0, 120), (0, 200)], [(0, 40), (0, 100)]], 0
    )
    rc |= _check_prefill_case(2, [[(10, 50), (64, 200)], [(0, 100), (130, 256)]], 4)
    rc |= _check_prefill_case(2, [[(100, 2048), (512, 4096)], [(0, 8192)]], 12)
    # non-4-aligned local_start (guards the out-store alignment fix): 1-token + spanning.
    rc |= _check_prefill_case(2, [[(0, 1), (17, 33)], [(63, 65), (255, 257)]], 34)
    # exhaustive start sweep: every local_start in [0, 130) with a fixed 40-wide window.
    rc |= _check_prefill_case(1, [[(s, s + 40) for s in range(130)]], 52)
    rc |= _check_prefill_case(
        2, [[(0, 0), (0, 200)], [(100, 100), (0, 128)]], 40
    )  # zero-len mix
    # kv-tile boundary (KV_BLOCK_SIZE == block_k == 64): window END at every multiple
    # of 64 +/- 2 -> guards win_tiles = ceil((le-chunk_start)/64) at exact / off-by-few.
    tile_ends = [
        (0, base + d)
        for base in (64, 128, 256, 320, 512, 576)
        for d in (-2, -1, 0, 1, 2)
    ]
    rc |= _check_prefill_case(1, [tile_ends], 60)
    # window START/END straddling a tile boundary (start in tile A, end in tile B).
    rc |= _check_prefill_case(
        2, [[(62, 130), (64, 192), (66, 200)], [(126, 260), (128, 256), (130, 320)]], 62
    )
    # varqlen / MTP tail-causal (fully ragged, empty batch, qlen>ctx clamp, next_n=8)
    rc |= _check_varqlen_case(3, [1, 3, 2], [200, 1500, 800], 26)
    rc |= _check_varqlen_case(3, [2, 0, 3], [384, 256, 640], 28)
    rc |= _check_varqlen_case(2, [8, 3], [4, 500], 44)
    rc |= _check_varqlen_case(4, [16, 8, 24, 4], [4096, 2048, 4096, 1024], 50)
    rc |= _check_varqlen_case(
        3, [2, 3, 1], [64, 127, 256], 72
    )  # boundary ctx, ragged qlen
    # fixed-MTP decode (uniform next_n per batch, cu_seq_q=None path); ctx on / off tile edges.
    rc |= _check_fixed_mtp_case(2, 1, [63, 129], 68)  # next_n=1 (pure decode), ctx +/-1
    rc |= _check_fixed_mtp_case(
        4, 2, [64, 65, 128, 127], 64
    )  # next_n=2, tail across the edge
    rc |= _check_fixed_mtp_case(3, 4, [256, 257, 320], 66)  # next_n=4
    rc |= _check_fixed_mtp_case(4, 8, [512, 511, 576, 320], 70)  # next_n=8 uniform
    print("  ALL PASS" if rc == 0 else "  SOME FAILED")
    return rc


# ═════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════


def export_df(df, out_prefix, scenario):
    """Optional side dump (--out): write the summary table to <prefix>_<scenario>.csv/.json."""
    if not out_prefix:
        return
    csv_path = f"{out_prefix}_{scenario}.csv"
    json_path = f"{out_prefix}_{scenario}.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    aiter.logger.info("wrote %s and %s", csv_path, json_path)


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "pa_mqa_logits_mxfp8 unsupported on %s (needs %s); skipping",
            get_gfx(),
            SUPPORTED_GFX,
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "--scenario",
        type=str,
        choices=["prefill", "decode"],
        nargs="*",
        default=["prefill", "decode"],
        help="which sweeps to run (each -> one table).",
    )
    parser.add_argument(
        "--prefill-bs",
        type=int,
        nargs="*",
        default=list(range(1, 21)),
        help="prefill batch sizes (per-batch qlen==ctx, sum==16384, causal).",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help="decode batch sizes.",
    )
    parser.add_argument(
        "--max-ctx",
        type=int,
        nargs="*",
        default=[1024, 8192],
        help="decode per-batch ctx upper bound (ctx in [0.9*max, max]).",
    )
    parser.add_argument(
        "--mtp",
        type=int,
        nargs="*",
        default=[1, 4, 8],
        help="decode next_n (MTP width).",
    )
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="skip the sampled correctness check.",
    )
    parser.add_argument(
        "--no-corner",
        action="store_true",
        help="skip the ragged/unaligned/varqlen correctness pass before the sweep.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="optional prefix; also dump each table to <prefix>_<scenario>.csv/.json.",
    )
    args = parser.parse_args()

    global _ITERS, _WARMUP, _SEED, _CHECK
    _ITERS, _WARMUP, _SEED, _CHECK = (
        args.iters,
        args.warmup,
        args.seed,
        not args.no_check,
    )

    if not args.no_corner:
        run_corner_correctness()

    if "prefill" in args.scenario:
        df = pd.DataFrame([test_prefill(bs) for bs in args.prefill_bs])
        aiter.logger.info(
            "pa_mqa_logits_mxfp8 prefill summary (markdown):\n%s",
            df.to_markdown(index=False),
        )
        export_df(df, args.out, "prefill")

    if "decode" in args.scenario:
        df = pd.DataFrame(
            [
                test_decode(batch, max_ctx, next_n)
                for max_ctx, next_n, batch in itertools.product(
                    args.max_ctx, args.mtp, args.batch
                )
            ]
        )
        aiter.logger.info(
            "pa_mqa_logits_mxfp8 decode summary (markdown):\n%s",
            df.to_markdown(index=False),
        )
        export_df(df, args.out, "decode")


if __name__ == "__main__":
    main()
