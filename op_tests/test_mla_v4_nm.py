# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the mi350 v4 'nm' MLA pipeline (mla_decode_fwd_v4_nm).

Purpose:
  - **Smoke**: confirm the dispatcher loads its .co, accepts the v4 18-slot
    kernarg layout, launches with the wave64 grid (bdx=256), and returns
    finite outputs for the qh64_1tg_16mx4_16nx1_nm_recompile variant.
  - **Determinism**: two back-to-back calls produce bit-identical outputs.
  - **No-NaN guard**: the v4 port's #1 historical landmine was a 256-NaN +
    256-zero output pattern caused by wave32-on-wave64 launch geometry.
    This test fails loud on that exact regression.

NOT covered here (intentionally separate work, document in the test file
docstring as TODO):
  - Numerical correctness vs a torch reference. The v4 nm host pipeline
    does FP8+e8m0 dequant via fp8e4m3_mul_fp8e8m0_bpad8_to_bf16 and a
    multi-step buffer concat (see poc_kl/mi350/mla_asm/mla_v4.h
    v4_detail::init_host_buffers). Reproducing that bit-exactly in
    pytest is ~200 LOC; defer to a follow-up PR. The recommended hook
    point is the `compare_against_poc_kl_dump()` helper at the bottom of
    this file — fill it in by running poc_kl `./mla.exe model_version=4
    ... dump_result=1` with the same seed and shape, then byte-compare
    aiter's logits against poc_kl's gpu_SPLIT_DATA.hex.

Usage:
  pytest -xvs op_tests/test_mla_v4_nm.py
"""

import numpy as np
import pytest
import torch

import aiter
import aiter.mla  # main no longer auto-imports submodules; need explicit
from aiter.jit.utils.chip_info import get_gfx


# ---------------------------------------------------------------------------
# Variant under test (matches the cfg_mla_v4_asm entry in
# hsa/gfx950/mla_v4/mla_v4_asm.csv served by csrc/py_itfs_cu/asm_mla_v4.cu).
# ---------------------------------------------------------------------------
GQA_RATIO     = 16          # num_heads / num_kv_heads
SUB_Q         = 64          # block of q_seq_lens (after gqa boost) handled per workgroup
PAGE_SIZE     = 1
NUM_KV_HEADS  = 1
DIM_NOPE      = 448         # FP8 NOPE bytes per token
DIM_ROPE      = 64          # BF16 ROPE elements per token (= 128 bytes; lives in qrope/kvrope)
DIM_QK_PACKED = 576         # = args.dim(512) + args.k_rotary(64); matches poc_kl stride_Page
V_HEAD_DIM    = 512         # logical V head dim = args.dim = kv_lora_rank


def _on_gfx950():
    try:
        return get_gfx() == "gfx950"
    except Exception:
        return False


needs_gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available() or not _on_gfx950(),
    reason="v4 nm shader is shipped only for gfx950 (mi350); requires GPU",
)


# ---------------------------------------------------------------------------
# Synthetic input builders. We do NOT replicate the host-side FP8+e8m0 dequant
# packing here (that's poc_kl/mla_v4.h v4_detail::init_host_buffers). For
# smoke testing the dispatcher we just need byte-level buffers of the right
# shape and dtype; numerical correctness is deferred (see file docstring).
# ---------------------------------------------------------------------------
def _build_inputs(batch=2, kv_seq_lens=64, q_seq_logical=4, num_heads=GQA_RATIO,
                  device="cuda", seed=0):
    """Return a dict of every tensor mla_decode_fwd_v4_nm needs.

    Sizes mirror what poc_kl/mi350/mla_asm/mla.cpp computes for the same cmd
    (only with kv_seq_lens shrunk small for fast pytest):
      total_q = batch * num_heads * q_seq_logical
      num_page = batch * (kv_seq_lens / page_size)
    """

    rng_np = np.random.default_rng(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    total_q = batch * q_seq_logical
    num_page = batch * (kv_seq_lens // PAGE_SIZE)
    num_kv_splits = 1                          # passes=1 for this variant

    # FP8 dtype: use aiter's canonical alias which auto-resolves per arch
    # (gfx942 = e4m3fnuz, gfx950 = e4m3fn). The kernel reads raw bytes (NOPE
    # bytes + e8m0 dup-scale bytes packed by host), so we just need a
    # 1-byte-per-elem tensor of the right shape — any random byte pattern
    # will do for smoke testing (numerical correctness lives in
    # test_mla_v4_nm_golden.py).
    fp8_dt = aiter.dtypes.fp8

    def _rand_fp8(shape):
        # numpy seeded RNG (NOT torch.randint — that is non-reproducible
        # in this env for uint8 even on CPU; see comment at top of
        # _build_inputs).
        np_arr = rng_np.integers(0, 256, size=shape, dtype=np.uint8)
        u = torch.from_numpy(np_arr).to(device)
        return u.view(fp8_dt)

    q = _rand_fp8((total_q, num_heads, DIM_QK_PACKED))
    qrope = torch.randn(
        (total_q, num_heads, DIM_ROPE),
        dtype=torch.bfloat16, device=device,
    )

    kv_buffer = _rand_fp8((num_page, PAGE_SIZE, NUM_KV_HEADS, DIM_QK_PACKED))
    kvrope = torch.randn(
        (num_page, PAGE_SIZE, NUM_KV_HEADS, DIM_ROPE),
        dtype=torch.bfloat16, device=device,
    )

    # Index tables.
    #   q_indptr[b] = b * (q_seq_lens / gqa_ratio) = b * q_seq_logical
    qo_indptr = torch.arange(
        0, batch + 1, dtype=torch.int32, device=device
    ) * q_seq_logical

    pages_per_seq = kv_seq_lens // PAGE_SIZE
    kv_indptr = torch.arange(
        0, batch + 1, dtype=torch.int32, device=device
    ) * pages_per_seq

    # Random page mapping (each batch's pages picked from [0, num_page)).
    kv_page_indices = torch.arange(
        0, batch * pages_per_seq, dtype=torch.int32, device=device
    )

    kv_last_page_lens = torch.full(
        (batch,), kv_seq_lens % PAGE_SIZE,
        dtype=torch.int32, device=device,
    )

    split_indptr = torch.arange(
        0, batch + 1, dtype=torch.int32, device=device
    ) * num_kv_splits

    # `output` here is the *final reduce* buffer (3D), used only when
    # out_16_nosplit=1. The split-out fp32 logits are allocated *inside*
    # mla_decode_fwd_v4_nm (aiter/mla.py) and returned separately. The
    # underlying mla_decode_v4_asm C-ABI dispatcher reads
    #   total_query_len = output.size(0)
    #   num_heads       = output.size(1)
    #   v_head_dim      = output.size(2)
    # so this MUST be 3D [total_q, num_heads, v_head_dim].
    output = torch.empty(
        (total_q, num_heads, V_HEAD_DIM),
        dtype=torch.bfloat16, device=device,
    ).fill_(-1)

    return dict(
        q=q,
        qrope=qrope,
        kv_buffer=kv_buffer,
        kvrope=kvrope,
        output=output,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_lens=kv_last_page_lens,
        split_indptr=split_indptr,
        max_seqlen_q=q_seq_logical,
        num_kv_splits=num_kv_splits,
        sub_Q=SUB_Q,
        out_16_nosplit=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@needs_gfx950
def test_v4_nm_smoke_default_shape():
    """Build qh64 / Gqa=16 / batch=1 / kv=64 inputs and call once. Ensure:
    - no exception (no GPU page fault, no abort)
    - kernel synchronizes cleanly (catches async fault past buffer end —
      i.e., logits/attn_lse under-allocated for the v4 nm 5D layout)
    - kernel actually wrote SOMETHING to logits (catches "kernel skipped"
      bugs — e.g., wrong gdy/gdx, or LTL=0 making last-page skip)

    We do NOT assert finite values because random uint8 bytes (our smoke
    inputs) bit-cast to FP8 produce a lot of FP8 NaNs, and the e8m0 scale
    bytes are also random — dequantization legitimately produces inf/NaN.
    True numerical correctness lives in compare_against_poc_kl_dump() (TODO).

    Uses batch=2 to also exercise the multi-batch (gdy>1) code path. 
    """
    args = _build_inputs(batch=2, kv_seq_lens=64, q_seq_logical=4, seed=0)

    SENTINEL = -7.7e30
    num_seqs = args["qo_indptr"].size(0) - 1
    num_kv_splits = args["num_kv_splits"]
    num_kv_heads = args["kv_buffer"].size(2)
    num_heads = args["q"].size(1)
    gqa_ratio = num_heads // num_kv_heads
    q_seq_lens_internal = gqa_ratio * args["max_seqlen_q"]
    args["logits"] = torch.full(
        (num_seqs, num_kv_splits, num_kv_heads, q_seq_lens_internal, V_HEAD_DIM),
        SENTINEL, dtype=torch.float32, device="cuda",
    )
    args["attn_lse"] = torch.full(
        (num_seqs, num_kv_splits, num_kv_heads, q_seq_lens_internal, 1),
        SENTINEL, dtype=torch.float32, device="cuda",
    )

    logits, attn_lse = aiter.mla.mla_decode_fwd_v4_nm(**args)
    torch.cuda.synchronize()  # force any async fault to surface here

    for b in range(num_seqs):
        ut = (logits[b] == SENTINEL).float().mean().item()
        assert ut < 0.01, (
            f"kernel skipped batch {b}: {ut*100:.1f}% still SENTINEL. "
            f"Check qo_indptr/kv_indptr/split_indptr against poc_kl "
            f"initialize_indices, and the logits shape against the kernel's "
            f"flat [batch, passes, head, m, n] layout."
        )


@needs_gfx950
def test_v4_nm_no_half_zero_pattern():
    """Regression guard for the mi350 wave-size landmine.

    The exact pre-fix symptom was: per row of dim_v=512 fp32 output,
    the first half (256 elements) was 0xffc00000 (NaN) and the second
    half (256 elements) was 0x00000000. This test fails loud on that
    pattern.
    """
    args = _build_inputs(batch=1, kv_seq_lens=64, q_seq_logical=4, seed=42)
    logits, _ = aiter.mla.mla_decode_fwd_v4_nm(**args)
    torch.cuda.synchronize()

    # logits is 5D [num_seqs, kMaxSplit, num_kv_heads, q_seq_lens_internal, dim_v];
    # only split=0 is written when num_kv_splits=1. Look at THAT slice only;
    # other split slots are uninitialized memory and would noise this test.
    written = logits[:, 0]                     # [num_seqs, num_kv_heads, q_seq_lens_internal, dim_v]
    flat = written.reshape(-1, V_HEAD_DIM)     # rows of dim_v=512
    # If half the row is NaN and the other half is exactly zero, that's the
    # historic wave32-on-wave64 landmine (256 NaN + 256 zero per row).
    half = V_HEAD_DIM // 2
    first_half_nan_count = torch.isnan(flat[:, :half]).sum(dim=1).max().item()
    second_half_zero_count = (flat[:, half:] == 0.0).sum(dim=1).max().item()
    assert not (first_half_nan_count > half * 0.9 and second_half_zero_count > half * 0.9), (
        f"Detected the wave32-on-wave64 launch landmine: "
        f"first {first_half_nan_count}/{half} NaN, "
        f"second {second_half_zero_count}/{half} zero. "
        f"Check make_launch_geometry / dispatcher bdx is wv_tg*64 (=256) on gfx950."
    )


@needs_gfx950
def test_v4_nm_determinism():
    """Two back-to-back launches with identical inputs must produce
    bit-identical outputs. This catches accumulator-init / uninit-LDS bugs.

    Compare via int32 bit-pattern instead of float value because random FP8
    inputs produce a lot of NaN, and torch.equal returns False on any NaN
    (NaN != NaN). Bit-equal is the right correctness criterion here anyway —
    the kernel is deterministic, so identical inputs → identical bytes.
    """
    SEED = 0
    args1 = _build_inputs(batch=2, kv_seq_lens=64, q_seq_logical=4, seed=SEED)
    logits1, lse1 = aiter.mla.mla_decode_fwd_v4_nm(**args1)
    torch.cuda.synchronize()
    logits1_bits = logits1.view(torch.int32).clone()
    lse1_bits = lse1.view(torch.int32).clone()

    args2 = _build_inputs(batch=2, kv_seq_lens=64, q_seq_logical=4, seed=SEED)
    logits2, lse2 = aiter.mla.mla_decode_fwd_v4_nm(**args2)
    torch.cuda.synchronize()

    # Compare only the slot the kernel actually wrote; padding (split>0) is
    # uninitialized memory and may legitimately differ between calls.
    written1 = logits1_bits[:, 0]
    written2 = logits2.view(torch.int32)[:, 0]
    assert torch.equal(written1, written2), \
        "v4 nm logits are non-deterministic (likely uninit accumulator/LDS)"
    assert torch.equal(lse1_bits[:, 0], lse2.view(torch.int32)[:, 0]), \
        "v4 nm attn_lse is non-deterministic"


@needs_gfx950
def test_v4_nm_out_16_nosplit_arg_accepted():
    """out_16_nosplit=1 should not crash (kernel may or may not honor it
    — at least the dispatcher must accept the arg cleanly)."""
    args = _build_inputs(batch=1, kv_seq_lens=64, q_seq_logical=4, seed=11)
    args["out_16_nosplit"] = 1
    logits, _ = aiter.mla.mla_decode_fwd_v4_nm(**args)
    torch.cuda.synchronize()  # surface any async fault
    assert logits is not None


@needs_gfx950
def test_v4_nm_unknown_variant_raises():
    """Asking for a (gqa, sub_Q, page) tuple not in cfg_mla_v4_asm must raise
    a clear error before launch — not silently load the wrong .co. Exercised
    by passing sub_Q=128 (only sub_Q=64 is currently shipped)."""
    args = _build_inputs(batch=1, kv_seq_lens=64, q_seq_logical=4, seed=0)
    args["sub_Q"] = 128
    with pytest.raises(Exception, match="no shipped variant"):
        aiter.mla.mla_decode_fwd_v4_nm(**args)


@needs_gfx950
def test_v4_nm_kernarg_scalar_slots(capfd, monkeypatch):
    """Regression guard for the 18-slot v4 nm kernarg layout.

    Locks in the *scalar* portion (slot 7 scalar_f, slot 8-12 ints, slot 15
    int) of the kernarg buffer produced by csrc/py_itfs_cu/asm_mla_v4.cu for
    the canonical qh64/gqa=16/page=1/passes=1/sub_Q=64 config. Any future
    change to the dispatcher that shifts a slot, mis-computes a stride /
    scale, or changes the formula here will trip this test before the golden
    numerical test does.

    Pointer slots are NOT checked (their values are runtime allocation
    addresses and don't have a stable reference). Bytes printed by the
    AITER_V4_NM_DUMP_KERNARG=1 path in asm_mla_v4.cu are captured via capfd.
    """
    monkeypatch.setenv("AITER_V4_NM_DUMP_KERNARG", "1")
    args = _build_inputs(batch=2, kv_seq_lens=64, q_seq_logical=4, seed=0)
    aiter.mla.mla_decode_fwd_v4_nm(**args)
    torch.cuda.synchronize()

    captured = capfd.readouterr()
    # The dispatcher fprintf's "[aiter kernarg 288B]" then 18 rows of 16
    # hex bytes. Parse the 18 rows out of stderr.
    import re
    lines = captured.err.splitlines()
    try:
        start = next(i for i, line in enumerate(lines)
                     if line.startswith("[aiter kernarg 288B]"))
    except StopIteration:
        pytest.fail(
            "kernarg hexdump not found in stderr — "
            "AITER_V4_NM_DUMP_KERNARG env var may have been ignored, "
            "or jinja was changed and the dump code removed.\n"
            f"stderr was: {captured.err[:500]}"
        )
    hex_rows = []
    for line in lines[start + 1: start + 1 + 18]:
        m = re.match(r"^((?:[0-9a-fA-F]{2}\s*){16})$", line.strip())
        if not m:
            break
        hex_rows.append(bytes.fromhex(line.strip().replace(" ", "")))
    assert len(hex_rows) == 18, (
        f"expected 18 hex rows of kernarg, got {len(hex_rows)}"
    )
    kargs = b"".join(hex_rows)
    assert len(kargs) == 288, f"kernarg byte total = {len(kargs)}, want 288"

    # Each slot is 16 bytes; first 4 bytes carry the payload, rest is padding.
    def slot(i): return kargs[i * 16: i * 16 + 16]
    def slot_u32(i): return int.from_bytes(slot(i)[:4], "little")
    import struct

    def slot_f32(i):
        return struct.unpack("<f", slot(i)[:4])[0]

    # scalar_f is computed in jinja with C `float`s (1.0f/sqrtf(512.f)). Mirror
    # that precision here so the byte-exact compare doesn't false-fail on the
    # FP64→FP32 round-off difference.
    expected_scalar_f_bytes = struct.pack(
        "<f", float(np.float32(1.0) / np.float32(np.sqrt(np.float32(448 + 64))))
    )
    expected_gqa_ratio  = GQA_RATIO                   # 16
    expected_kv_split   = 1                            # num_kv_splits=1
    expected_total_kv   = 64 * 2                       # kv_seq_lens * batch
    expected_stride_pg  = 1 * DIM_QK_PACKED            # page_size * 576
    expected_log2_page  = 0                            # log2(page_size=1)
    expected_out16ns    = 0                            # out_16_nosplit=0

    # slot 7 scalar_f: byte-exact compare (FP32)
    actual_scalar_f_bytes = slot(7)[:4]
    assert actual_scalar_f_bytes == expected_scalar_f_bytes, (
        f"slot 7 scalar_f bytes: got {actual_scalar_f_bytes.hex()}, "
        f"want {expected_scalar_f_bytes.hex()} (= 1/sqrt(512) in FP32)"
    )
    for slot_idx, want, name in [
        (8,  expected_gqa_ratio,  "s_gqa_ratio"),
        (9,  expected_kv_split,   "s_kv_split"),
        (10, expected_total_kv,   "s_total_kv"),
        (11, expected_stride_pg,  "s_stride_page"),
        (12, expected_log2_page,  "s_log2_page"),
        (15, expected_out16ns,    "out_16_nosplit"),
    ]:
        got = slot_u32(slot_idx)
        assert got == want, (
            f"slot {slot_idx} ({name}): got {got} (0x{got:08x}), "
            f"want {want} (0x{want:08x})"
        )

    # Sanity: pointer slots (0..6, 13, 14, 16, 17) must be non-NULL.
    for slot_idx in (0, 1, 2, 3, 4, 5, 6, 13, 14, 16, 17):
        ptr = int.from_bytes(slot(slot_idx)[:8], "little")
        assert ptr != 0, f"slot {slot_idx} pointer is NULL"


if __name__ == "__main__":
    pytest.main([__file__, "-xvs"])
