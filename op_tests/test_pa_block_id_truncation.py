# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Reproduce the aiter ASM paged-attention block_id truncation issue.

When the block_id loaded from the block_tables tensor crosses 65,535
(= 2^16), the aiter precompiled ASM `pa_*.co` family on gfx950/gfx942
reads from the wrong physical KV slot — consistent with a 16-bit
narrowing (`block_id & 0xFFFF`) of the loaded value before it is used
in slot-address arithmetic.

Strategy:
  * Allocate a KV pool with > 65,535 physical blocks (NUM_BLOCKS = 70,000).
  * Fill two specific blocks (one below 65,535, one above) with a
    distinctive constant, leave everything else at zero.
  * Run pa_fwd_asm on a single sequence whose block_tables points at
    each chosen block in turn, with context_lens = block_size.
  * Because the chosen block is filled with a constant V, the attention
    output equals that constant (softmax over a single block's slots
    sums to 1, weighted with constant V).

If the kernel narrows block_id to 16 bits, the high block_id (= 67,000)
wraps to 67,000 - 65,536 = 1,464, an unfilled block that contains zeros,
so the output collapses to ~0 instead of the expected fingerprint.

Empirical result on gfx950 (MI355X), aiter built 2026-04-20+:
  Both the qlen=1 kernel (`pa_bf16_noquant_gqa8_1tg_4w.co`) and the
  qlen=4 MTP kernel (`pa_bf16_noquant_gqa8_1tg_4w_mtp_msk1.co`) return
  0.0000 for block_id = 67,000 instead of the expected 0.7500. The wrap
  target (1,464) matches `block_id & 0xFFFF`.

  The reproduction requires NUM_KV_HEADS = 8 to match production
  per-block stride (32 KB). With NUM_KV_HEADS = 1 (4 KB stride) the bug
  does not surface — likely because some tile-level address calculation
  in the kernel only narrows block_id when iterating over enough KV
  heads. Either way, this file reproduces the production-relevant
  configuration.

Run:
    pytest /root/aiter/op_tests/test_pa_block_id_truncation.py -v -s

Or as a script:
    python /root/aiter/op_tests/test_pa_block_id_truncation.py
"""

import pytest
import torch

import aiter

# ---------- configuration matching the ATOM Eagle3 draft signature ----------
# Production layout per TP=8 rank: num_q_heads = num_kv_heads = 8 (full MHA).
# aiter's gqa-rounding selects the gqa8 kernel either way.
#
# Critical: per-block stride must match production for the i32-overflow
# hypothesis to be testable. With NUM_KV_HEADS=8, HEAD_DIM=128, BLOCK_SIZE=16,
# bf16 elem_size=2:
#     per_block_stride = 16 × 8 × 128 × 2 = 32,768 bytes
#     i32 overflow boundary = 2^31 / 32768 = 65,536
# Lowering NUM_KV_HEADS would shrink the stride and push the overflow
# boundary far above any practical block_id, masking the bug.
NUM_Q_HEADS = 8
NUM_KV_HEADS = 8
HEAD_DIM = 128
BLOCK_SIZE = 16

# Need num_blocks > 65535 to trigger the crossing.
NUM_BLOCKS = 70_000

# Block IDs to fingerprint and probe. Layout:
#   1,000   — safely below the boundary (sanity baseline)
#   65,535  — last value that fits in u16 (= 0xFFFF). Should still read
#             correctly even if the kernel does `block_id & 0xFFFF`,
#             because that operation is a no-op here.
#   65,536  — first value that overflows u16 (= 0x10000). If the kernel
#             narrows to 16 bits, this wraps to 0 and reads block 0.
#   67,000  — well above the boundary; wraps to 67000 - 65536 = 1,464.
SAFE_BLOCK_ID = 1_000
EDGE_LAST_SAFE = 65_535
EDGE_FIRST_BUGGY = 65_536
BUGGY_BLOCK_ID = 67_000

# Distinct fingerprint per block — kept small (< 1.0) to stay well within
# bf16 precision after softmax normalization.
SIG_SAFE = 0.50
SIG_EDGE_LAST = 0.30
SIG_EDGE_FIRST = 0.40
SIG_BUGGY = 0.75

_FINGERPRINTS = [
    (SAFE_BLOCK_ID, SIG_SAFE, "below_65535"),
    (EDGE_LAST_SAFE, SIG_EDGE_LAST, "edge_65535_last_u16"),
    (EDGE_FIRST_BUGGY, SIG_EDGE_FIRST, "edge_65536_first_overflow"),
    (BUGGY_BLOCK_ID, SIG_BUGGY, "above_65535"),
]


def _build_kv_cache():
    """Allocate a sparse bf16 KV pool with two fingerprinted blocks."""
    dtype = torch.bfloat16
    x = 16 // dtype.itemsize  # = 8 for bf16
    assert HEAD_DIM % x == 0

    # K layout: [num_blocks, num_kv_heads, head_dim/x, block_size, x]
    k_cache = torch.zeros(
        NUM_BLOCKS,
        NUM_KV_HEADS,
        HEAD_DIM // x,
        BLOCK_SIZE,
        x,
        dtype=dtype,
        device="cuda",
    )
    # V layout: [num_blocks, num_kv_heads, head_dim, block_size]
    v_cache = torch.zeros(
        NUM_BLOCKS,
        NUM_KV_HEADS,
        HEAD_DIM,
        BLOCK_SIZE,
        dtype=dtype,
        device="cuda",
    )

    for block_id, sig, _label in _FINGERPRINTS:
        k_cache[block_id].fill_(sig)
        v_cache[block_id].fill_(sig)

    return k_cache, v_cache


def _run_pa_fwd_asm(k_cache, v_cache, target_block_id, max_qlen=1):
    """Run pa_fwd_asm with a single sequence that contains exactly one block,
    that block being `target_block_id`. Returns the attention output value.

    `max_qlen` selects the kernel family:
      max_qlen=1 → mtp=0 → pa_bf16_noquant_gqa8_1tg_4w.co (non-MTP decode)
      max_qlen=4 → mtp=14→1 → pa_bf16_noquant_gqa8_1tg_4w_mtp_msk1.co (MTP)
    """
    NUM_PAGES = 16
    block_tables = torch.full(
        (1, NUM_PAGES), target_block_id, dtype=torch.int32, device="cuda"
    )
    context_lens = torch.full(
        (1,), BLOCK_SIZE * NUM_PAGES, dtype=torch.int32, device="cuda"
    )
    cu_seqlens_q = torch.tensor([0, max_qlen], dtype=torch.int32, device="cuda")

    # Query: arbitrary nonzero values — softmax will normalize, V is constant.
    query = torch.ones(
        max_qlen, NUM_Q_HEADS, HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )

    out = aiter.pa_fwd_asm(
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_tables.stride(0),
        max_qlen=max_qlen,
        K_QScale=None,
        V_QScale=None,
        out_=None,
        qo_indptr=cu_seqlens_q,
        high_precision=0,
    )
    # Output shape: [max_qlen, num_q_heads, head_dim] — all elements should
    # equal the fingerprint of target_block_id (because V is constant in
    # that block and softmax weights sum to 1).
    return out.float().mean().item()


@pytest.mark.parametrize(
    "block_id,expected_sig,label",
    _FINGERPRINTS,
)
@pytest.mark.parametrize(
    "max_qlen,kernel_label",
    [
        (1, "qlen1_non_MTP_kernel"),
        (4, "qlen4_MTP_kernel"),
    ],
)
def test_pa_fwd_asm_block_id_no_truncation(
    block_id, expected_sig, label, max_qlen, kernel_label
):
    """Output for a single-block sequence must match that block's fingerprint
    regardless of whether block_id is below or above 65,535. Run for both
    qlen=1 (non-MTP decode kernel) and qlen=4 (MTP kernel)."""
    k_cache, v_cache = _build_kv_cache()
    actual = _run_pa_fwd_asm(k_cache, v_cache, block_id, max_qlen=max_qlen)

    msg = (
        f"[{kernel_label}/{label}] block_id={block_id} max_qlen={max_qlen}: "
        f"expected output ≈ {expected_sig}, got {actual:.6f}. "
    )
    if block_id >= 65_536:
        wrap = block_id - 65_536
        msg += (
            f"If the kernel narrows block_id to 16 bits, the high block_id "
            f"would wrap to block {wrap} (unfilled, = 0), so output collapses "
            f"toward ~0. Observed value of ~0 here is the bug signature."
        )
    assert actual == pytest.approx(expected_sig, abs=1e-2), msg


if __name__ == "__main__":
    # Standalone runner for quick repro without pytest infrastructure.
    print(
        f"Allocating KV pool: {NUM_BLOCKS} blocks × bf16 "
        f"× {NUM_KV_HEADS} kv_head × {HEAD_DIM} head_dim × {BLOCK_SIZE} block_size"
    )
    k_cache, v_cache = _build_kv_cache()
    print(f"  K cache {tuple(k_cache.shape)} = {k_cache.numel() * 2 / 1e9:.2f} GB")
    print(f"  V cache {tuple(v_cache.shape)} = {v_cache.numel() * 2 / 1e9:.2f} GB")
    print()

    for max_qlen, kernel_label in [(1, "qlen1_non_MTP"), (4, "qlen4_MTP")]:
        print(f"=== {kernel_label} (max_qlen={max_qlen}) ===")
        for block_id, expected, label in _FINGERPRINTS:
            actual = _run_pa_fwd_asm(k_cache, v_cache, block_id, max_qlen=max_qlen)
            status = "OK" if abs(actual - expected) < 1e-2 else "BUG"
            print(
                f"[{status}] block_id={block_id:>7d}  expected={expected:.4f}  "
                f"actual={actual:.4f}  Δ={actual - expected:+.4f}  ({label})"
            )
            if status == "BUG" and block_id >= 65_536:
                wrap = block_id & 0xFFFF
                print(
                    f"  → if block_id is narrowed to 16 bits, "
                    f"reads block {wrap} instead (unfilled = 0)."
                )
        print()

    # ---- Performance comparison ----
    # Measure latency across different block_id ranges and batch sizes
    # to verify no performance regression from the rebase fix.

    print("=== Performance Comparison ===")
    print(
        f"{'scenario':<30s} {'batch':>5s} {'ctx_len':>7s} {'max_qlen':>8s} "
        f"{'avg_us':>8s} {'std_us':>8s}"
    )
    print("-" * 75)

    PERF_NUM_WARMUP = 5
    PERF_NUM_ITERS = 50

    perf_configs = [
        ("low_block_ids", 1000, 1),
        ("high_block_ids", 67000, 1),
        ("low_block_ids", 1000, 4),
        ("high_block_ids", 67000, 4),
    ]

    for num_seqs in [1, 8, 32]:
        for label, base_block_id, max_qlen in perf_configs:
            num_pages = 16
            block_tables = torch.full(
                (num_seqs, num_pages),
                base_block_id,
                dtype=torch.int32,
                device="cuda",
            )
            for i in range(num_seqs):
                block_tables[i] = base_block_id + i
                k_cache[base_block_id + i].fill_(0.25)
                v_cache[base_block_id + i].fill_(0.25)

            ctx_len = BLOCK_SIZE * num_pages
            context_lens = torch.full(
                (num_seqs,),
                ctx_len,
                dtype=torch.int32,
                device="cuda",
            )
            total_q = num_seqs * max_qlen
            cu_seqlens_q = torch.arange(
                0,
                total_q + 1,
                max_qlen,
                dtype=torch.int32,
                device="cuda",
            )
            query = torch.randn(
                total_q,
                NUM_Q_HEADS,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device="cuda",
            )

            def _run():
                return aiter.pa_fwd_asm(
                    query,
                    k_cache,
                    v_cache,
                    block_tables,
                    context_lens,
                    block_tables.stride(0),
                    max_qlen=max_qlen,
                    K_QScale=None,
                    V_QScale=None,
                    out_=None,
                    qo_indptr=cu_seqlens_q,
                    high_precision=0,
                )

            for _ in range(PERF_NUM_WARMUP):
                _run()
            torch.cuda.synchronize()

            start_events = [
                torch.cuda.Event(enable_timing=True) for _ in range(PERF_NUM_ITERS)
            ]
            end_events = [
                torch.cuda.Event(enable_timing=True) for _ in range(PERF_NUM_ITERS)
            ]
            for i in range(PERF_NUM_ITERS):
                start_events[i].record()
                _run()
                end_events[i].record()
            torch.cuda.synchronize()

            latencies = [
                s.elapsed_time(e) * 1000 for s, e in zip(start_events, end_events)
            ]
            avg_us = sum(latencies) / len(latencies)
            std_us = (sum((x - avg_us) ** 2 for x in latencies) / len(latencies)) ** 0.5

            tag = f"{label}_qlen{max_qlen}"
            print(
                f"  {tag:<28s} {num_seqs:>5d} {ctx_len:>7d} {max_qlen:>8d} "
                f"{avg_us:>8.2f} {std_us:>8.2f}"
            )
        print()

    print(
        "Note: low_block_ids (<65536) vs high_block_ids (>65536) should show\n"
        "      similar latency — any significant gap indicates a regression."
    )
