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

Fix:
  The SP3 paged-attention kernels were converted from 32-bit `buffer_load`
  (whose voffset overflows once block_id × per_block_stride crosses 2^31)
  to 64-bit `global_load` absolute addressing, which addresses the full KV
  pool beyond 4GB. After the rebuild, block_id = 67,000 reads the correct
  physical block across the rebuilt .co set:
    * query dtype:   bf16 (pa_bf16_*) and fp16 (pa_fp16_*)
    * KV quant:      noquant / per-token fp8 (hp=1 `_2tg_4w_hp` + hp=2
                     `_2tg_4w_uhp`) / per-token int8
    * gqa family:    gqa8 (qlen=1 decode + qlen=4 MTP) and gqa16 (qlen=1)
  See the parametrized tests below for the exact coverage matrix.

  NOT reachable through pa_fwd_asm (and therefore not asserted here):
    * `_mtp_msk0` kernels — pa_fwd_asm always picks msk=1 for MTP; the msk0
      variants are dispatched only by the persistent-split path
      (pa_persistent_fwd / pa_ps_fwd_asm, mask=0).
    * gqa16 + MTP (qlen=4) — needs qTile = gqa_ratio*qlen = 64 > 16, which the
      dispatcher rejects with a hard abort.
    * plain `_2tg_4w` (fp8/int8, high_precision=0) and the int8 decode path —
      return 0 for a single-block sequence in this build.

Empirical (pre-fix) result on gfx950 (MI355X), aiter built 2026-04-20:
  Both the qlen=1 kernel (`pa_bf16_noquant_gqa8_1tg_4w.co`) and the
  qlen=4 MTP kernel (`pa_bf16_noquant_gqa8_1tg_4w_mtp_msk1.co`) returned
  0.0000 for block_id = 67,000 instead of the expected 0.7500. The wrap
  target (1,464) matches `block_id & 0xFFFF`. This test now guards against
  a regression of that fix.

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
from aiter import pertoken_quant

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

# To exercise the gqa16 kernel family we keep NUM_KV_HEADS=8 (so the 32 KB
# per-block stride and the 65,536 overflow boundary are unchanged) and raise
# the query-head count so gqa_ratio = 128 / 8 = 16 rounds to the gqa16 kernel.
GQA16_Q_HEADS = 128

# Query/KV dtype families. The rebuilt .co set splits into a `pa_bf16_*` half
# (bf16 query) and a `pa_fp16_*` half (fp16 query); both must read >4GB blocks
# correctly, so we exercise both.
Q_DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}

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
SAFE_BLOCK_ID = 1
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


# ---------- KV-cache quantization variants ----------------------------------
# The >4GB / block_id-truncation fix lives in the SP3 paged-attention kernels.
# The non-quantized (bf16/fp16 KV) kernels use an unconditional 64-bit
# global_load; the quantized (pertoken fp8 / int8 KV, "W8") kernels received
# the same 64-bit global_load treatment in the *_MTP variants. We therefore
# exercise all three KV families so the fix is validated across the kernels
# that were actually rebuilt:
#   "noquant" → pa_bf16_noquant_gqa8_*           (bf16 KV)
#   "fp8"     → pa_bf16_pertokenFp8_gqa8_*        (per-token fp8 KV)
#   "int8"    → pa_bf16_pertokenInt8_gqa8_*       (per-token int8 KV)
KV_QUANTS = ["noquant", "fp8", "int8"]

# fp8 decode (max_qlen=1) only produces correct results with high_precision>=1;
# with high_precision=0 the fp8 decode path returns 0. int8/noquant ignore it.
_HIGH_PRECISION = {"noquant": 0, "fp8": 1, "int8": 0}

# Dequant noise budget on top of the bf16 baseline tolerance.
_ABS_TOL = {"noquant": 1e-2, "fp8": 2e-2, "int8": 3e-2}

# Cache the (possibly large, ~5GB) quantized KV pools so the parametrized
# pytest cases don't re-quantize 70k blocks for every block_id/qlen combo.
# Keyed by (q_dtype_str, kv_quant); the pool only depends on the KV dtype and
# NUM_KV_HEADS, so gqa8 and gqa16 (which differ only in query-head count) share
# the same cached pool.
_KV_CACHE_BY_QUANT = {}


def _pertoken_quant_kvcache_symm(k_cache, v_cache, quant_dtype):
    """Per-token symmetric quantization of the KV pool, producing the ASM
    kernel's expected layout + per-token scales. Mirrors the helper in
    op_tests/test_pa_mtp.py."""
    num_blocks = k_cache.shape[0]
    num_heads = k_cache.shape[1]
    head_dim = v_cache.shape[2]
    block_size = v_cache.shape[3]

    k_cache_permute = (
        k_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )
    v_cache_permute = (
        v_cache.permute(0, 1, 3, 2)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )

    k_quant, k_scale_asm = pertoken_quant(k_cache_permute, quant_dtype=quant_dtype)
    v_quant, v_scale_asm = pertoken_quant(v_cache_permute, quant_dtype=quant_dtype)

    quant_x = 16 // quant_dtype.itemsize
    k_quant = (
        k_quant.view(num_blocks, num_heads, block_size, head_dim // quant_x, quant_x)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    v_quant = (
        v_quant.view(num_blocks, num_heads, block_size, head_dim)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    # ASM kernel consumes the raw per-token scales ([num_blocks, num_heads,
    # block_size, 1]); the flattened [num_heads, total_tokens] form is only for
    # the torch reference.
    return k_quant, v_quant, k_scale_asm, v_scale_asm


def _asm_V_shuffle(VC):
    """Reshape V into the (block_size/x, head_size, x) layout the ASM kernel
    expects. Mirrors op_tests/test_pa_mtp.py:asm_V_shuffle."""
    x = 16 // VC.element_size()
    num_blocks, num_kv_heads, head_size, block_size = VC.shape
    VC = VC.view(num_blocks, num_kv_heads, head_size, block_size // x, x)
    return VC.permute(0, 1, 3, 2, 4).contiguous()


def _build_kv_cache(q_dtype_str="bf16", kv_quant="noquant"):
    """Allocate a sparse KV pool with fingerprinted blocks.

    Returns (k_cache, v_cache, k_scale, v_scale). For "noquant" the caches are
    in the query dtype (bf16/fp16) and the scales are None; for "fp8"/"int8"
    the caches are quantized and laid out (incl. V shuffle) the way the ASM
    kernel expects, with per-token scales.
    """
    key = (q_dtype_str, kv_quant)
    if key in _KV_CACHE_BY_QUANT:
        return _KV_CACHE_BY_QUANT[key]

    dtype = Q_DTYPES[q_dtype_str]
    x = 16 // dtype.itemsize  # = 8 for bf16/fp16
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

    if kv_quant == "noquant":
        result = (k_cache, v_cache, None, None)
    else:
        quant_dtype = aiter.dtypes.fp8 if kv_quant == "fp8" else aiter.dtypes.i8
        k_q, v_q, k_scale, v_scale = _pertoken_quant_kvcache_symm(
            k_cache, v_cache, quant_dtype
        )
        result = (k_q, _asm_V_shuffle(v_q), k_scale, v_scale)

    _KV_CACHE_BY_QUANT[key] = result
    return result


def _run_pa_fwd_asm(
    k_cache,
    v_cache,
    k_scale,
    v_scale,
    target_block_id,
    max_qlen=1,
    kv_quant="noquant",
    num_q_heads=NUM_Q_HEADS,
    q_dtype_str="bf16",
    high_precision=None,
):
    """Run pa_fwd_asm with a single sequence that contains exactly one block,
    that block being `target_block_id`. Returns the attention output value.

    `max_qlen` selects the kernel family:
      max_qlen=1 → mtp=0 → non-MTP decode kernel
      max_qlen=4 → mtp=14→1 → MTP kernel
    `kv_quant` selects the KV dtype family (noquant/fp8/int8).
    `num_q_heads` controls gqa_ratio (= num_q_heads / NUM_KV_HEADS) and hence
    whether the gqa8 (=8) or gqa16 (=128/8=16) kernel is dispatched.
    `q_dtype_str` selects the bf16 vs fp16 query/kernel family.
    `high_precision` overrides the per-quant default (used to select the fp8
    `_2tg_4w_uhp` kernel via high_precision=2).
    """
    hp = _HIGH_PRECISION[kv_quant] if high_precision is None else high_precision
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
        max_qlen, num_q_heads, HEAD_DIM, dtype=Q_DTYPES[q_dtype_str], device="cuda"
    )

    out = aiter.pa_fwd_asm(
        query,
        k_cache,
        v_cache,
        block_tables,
        context_lens,
        block_tables.stride(0),
        max_qlen=max_qlen,
        K_QScale=k_scale,
        V_QScale=v_scale,
        out_=None,
        qo_indptr=cu_seqlens_q,
        high_precision=hp,
    )
    # Output shape: [max_qlen, num_q_heads, head_dim] — all elements should
    # equal the fingerprint of target_block_id (because V is constant in
    # that block and softmax weights sum to 1).
    return out.float().mean().item()


def _assert_block_id_fingerprint(actual, expected_sig, block_id, kv_quant, tag):
    """Shared assertion: the single-block attention output must equal the
    block's fingerprint, with a clear >4GB-truncation diagnostic on failure."""
    msg = (
        f"[{tag}] block_id={block_id}: expected output ≈ {expected_sig}, "
        f"got {actual:.6f}. "
    )
    if block_id >= 65_536:
        wrap = block_id - 65_536
        msg += (
            f"If the kernel narrows block_id to 16 bits, the high block_id "
            f"would wrap to block {wrap} (unfilled, = 0), so output collapses "
            f"toward ~0. Observed value of ~0 here is the bug signature."
        )
    assert actual == pytest.approx(expected_sig, abs=_ABS_TOL[kv_quant]), msg


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
@pytest.mark.parametrize("kv_quant", KV_QUANTS)
@pytest.mark.parametrize("q_dtype", list(Q_DTYPES))
def test_pa_fwd_asm_block_id_no_truncation(
    q_dtype, kv_quant, block_id, expected_sig, label, max_qlen, kernel_label
):
    """gqa8 kernels: output for a single-block sequence must match that block's
    fingerprint regardless of whether block_id is below or above 65,535.

    Swept across the rebuilt gqa8 .co set: query dtype (bf16/fp16) × KV quant
    (noquant / per-token fp8 / per-token int8) × qlen=1 (non-MTP decode) and
    qlen=4 (MTP), so the 64-bit global_load >4GB fix is covered everywhere."""
    if kv_quant == "int8" and max_qlen == 1:
        pytest.skip(
            "int8 KV decode (max_qlen=1, mtp=0) is unsupported in this build "
            "(returns 0 for all block_ids regardless of the >4GB fix); int8 is "
            "covered through its MTP kernel at max_qlen=4."
        )
    k_cache, v_cache, k_scale, v_scale = _build_kv_cache(q_dtype, kv_quant)
    actual = _run_pa_fwd_asm(
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        block_id,
        max_qlen=max_qlen,
        kv_quant=kv_quant,
        num_q_heads=NUM_Q_HEADS,
        q_dtype_str=q_dtype,
    )
    _assert_block_id_fingerprint(
        actual,
        expected_sig,
        block_id,
        kv_quant,
        f"{q_dtype}/gqa8/{kv_quant}/{kernel_label}/{label}",
    )


@pytest.mark.parametrize(
    "block_id,expected_sig,label",
    _FINGERPRINTS,
)
@pytest.mark.parametrize("q_dtype", list(Q_DTYPES))
def test_pa_fwd_asm_block_id_no_truncation_gqa16(
    q_dtype, block_id, expected_sig, label
):
    """gqa16 kernel (pa_*_noquant_gqa16_1tg_4w.co): same >4GB block_id check
    with gqa_ratio=16 (NUM_Q_HEADS=128 / NUM_KV_HEADS=8, stride still 32 KB).

    Restricted to noquant + qlen=1: the gqa16 quant decode path has no non-MTP
    kernel (returns 0), and gqa16 × qlen=4 would need qTile=64 (>16) which the
    dispatcher rejects with a hard abort. Both are unrelated to the >4GB fix."""
    k_cache, v_cache, k_scale, v_scale = _build_kv_cache(q_dtype, "noquant")
    actual = _run_pa_fwd_asm(
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        block_id,
        max_qlen=1,
        kv_quant="noquant",
        num_q_heads=GQA16_Q_HEADS,
        q_dtype_str=q_dtype,
    )
    _assert_block_id_fingerprint(
        actual,
        expected_sig,
        block_id,
        "noquant",
        f"{q_dtype}/gqa16/noquant/qlen1/{label}",
    )


@pytest.mark.parametrize(
    "block_id,expected_sig,label",
    _FINGERPRINTS,
)
@pytest.mark.parametrize("q_dtype", list(Q_DTYPES))
def test_pa_fwd_asm_block_id_no_truncation_fp8_uhp(
    q_dtype, block_id, expected_sig, label
):
    """fp8 ultra-high-precision decode kernel (pa_*_pertokenFp8_gqa8_2tg_4w_uhp.co):
    selected via high_precision=2 at qlen=1. This is a distinct rebuilt .co from
    the high_precision=1 `_2tg_4w_hp` kernel covered by the main test, so it gets
    its own >4GB block_id check.

    (The plain `_2tg_4w` kernel, high_precision=0, returns 0 for a single-block
    sequence in this build and is therefore not reachable for this fingerprint
    test via pa_fwd_asm; same for the int8 `_2tg_4w` decode path.)"""
    k_cache, v_cache, k_scale, v_scale = _build_kv_cache(q_dtype, "fp8")
    actual = _run_pa_fwd_asm(
        k_cache,
        v_cache,
        k_scale,
        v_scale,
        block_id,
        max_qlen=1,
        kv_quant="fp8",
        num_q_heads=NUM_Q_HEADS,
        q_dtype_str=q_dtype,
        high_precision=2,
    )
    _assert_block_id_fingerprint(
        actual,
        expected_sig,
        block_id,
        "fp8",
        f"{q_dtype}/gqa8/fp8_uhp/qlen1/{label}",
    )


if __name__ == "__main__":
    # Standalone runner for quick repro without pytest infrastructure.
    print(
        f"Allocating KV pool: {NUM_BLOCKS} blocks × bf16 "
        f"× {NUM_KV_HEADS} kv_head × {HEAD_DIM} head_dim × {BLOCK_SIZE} block_size"
    )
    k_cache, v_cache, _, _ = _build_kv_cache("bf16", "noquant")
    print(f"  K cache {tuple(k_cache.shape)} = {k_cache.numel() * 2 / 1e9:.2f} GB")
    print(f"  V cache {tuple(v_cache.shape)} = {v_cache.numel() * 2 / 1e9:.2f} GB")
    print()

    def _report(
        tag,
        kq,
        vq,
        ks,
        vs,
        max_qlen,
        kv_quant,
        num_q_heads,
        q_dtype,
        high_precision=None,
    ):
        print(f"=== {tag} ===")
        for block_id, expected, label in _FINGERPRINTS:
            actual = _run_pa_fwd_asm(
                kq,
                vq,
                ks,
                vs,
                block_id,
                max_qlen=max_qlen,
                kv_quant=kv_quant,
                num_q_heads=num_q_heads,
                q_dtype_str=q_dtype,
                high_precision=high_precision,
            )
            status = "OK" if abs(actual - expected) < _ABS_TOL[kv_quant] else "BUG"
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

    for q_dtype in Q_DTYPES:
        # gqa8 family: all KV quants, qlen 1 + 4 (int8 decode unsupported).
        for kv_quant in KV_QUANTS:
            kq, vq, ks, vs = _build_kv_cache(q_dtype, kv_quant)
            for max_qlen, kernel_label in [(1, "qlen1_non_MTP"), (4, "qlen4_MTP")]:
                if kv_quant == "int8" and max_qlen == 1:
                    continue  # int8 non-MTP decode unsupported; covered via MTP.
                _report(
                    f"{q_dtype} / gqa8 / {kv_quant} / {kernel_label} (max_qlen={max_qlen})",
                    kq,
                    vq,
                    ks,
                    vs,
                    max_qlen,
                    kv_quant,
                    NUM_Q_HEADS,
                    q_dtype,
                )
        # fp8 ultra-high-precision decode kernel (_2tg_4w_uhp via high_precision=2).
        kq, vq, ks, vs = _build_kv_cache(q_dtype, "fp8")
        _report(
            f"{q_dtype} / gqa8 / fp8_uhp / qlen1_non_MTP (max_qlen=1, hp=2)",
            kq,
            vq,
            ks,
            vs,
            1,
            "fp8",
            NUM_Q_HEADS,
            q_dtype,
            high_precision=2,
        )
        # gqa16 family: noquant + qlen1 only (see test docstring for why).
        kq, vq, ks, vs = _build_kv_cache(q_dtype, "noquant")
        _report(
            f"{q_dtype} / gqa16 / noquant / qlen1_non_MTP (max_qlen=1)",
            kq,
            vq,
            ks,
            vs,
            1,
            "noquant",
            GQA16_Q_HEADS,
            q_dtype,
        )

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
        ("low_block_ids", 1, 1),
        ("high_block_ids", 67000, 1),
        ("low_block_ids", 100, 4),
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
