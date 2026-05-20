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
GQA_RATIO = 16  # num_heads / num_kv_heads
SUB_Q = 64  # block of q_seq_lens (after gqa boost) handled per workgroup
PAGE_SIZE = 1
NUM_KV_HEADS = 1
DIM_NOPE = 448  # FP8 NOPE bytes per token
DIM_ROPE = 64  # BF16 ROPE elements per token (= 128 bytes; lives in qrope/kvrope)
DIM_QK_PACKED = 576  # = args.dim(512) + args.k_rotary(64); matches poc_kl stride_Page
V_HEAD_DIM = 512  # logical V head dim = args.dim = kv_lora_rank


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
def _build_inputs(
    batch=2, kv_seq_lens=64, q_seq_logical=4, num_heads=GQA_RATIO, device="cuda", seed=0
):
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
    num_kv_splits = 1  # passes=1 for this variant

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
        dtype=torch.bfloat16,
        device=device,
    )

    kv_buffer = _rand_fp8((num_page, PAGE_SIZE, NUM_KV_HEADS, DIM_QK_PACKED))
    kvrope = torch.randn(
        (num_page, PAGE_SIZE, NUM_KV_HEADS, DIM_ROPE),
        dtype=torch.bfloat16,
        device=device,
    )

    # Index tables.
    #   q_indptr[b] = b * (q_seq_lens / gqa_ratio) = b * q_seq_logical
    qo_indptr = (
        torch.arange(0, batch + 1, dtype=torch.int32, device=device) * q_seq_logical
    )

    pages_per_seq = kv_seq_lens // PAGE_SIZE
    kv_indptr = (
        torch.arange(0, batch + 1, dtype=torch.int32, device=device) * pages_per_seq
    )

    # Random page mapping (each batch's pages picked from [0, num_page)).
    kv_page_indices = torch.arange(
        0, batch * pages_per_seq, dtype=torch.int32, device=device
    )

    kv_last_page_lens = torch.full(
        (batch,),
        kv_seq_lens % PAGE_SIZE,
        dtype=torch.int32,
        device=device,
    )

    split_indptr = (
        torch.arange(0, batch + 1, dtype=torch.int32, device=device) * num_kv_splits
    )

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
        dtype=torch.bfloat16,
        device=device,
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
        SENTINEL,
        dtype=torch.float32,
        device="cuda",
    )
    args["attn_lse"] = torch.full(
        (num_seqs, num_kv_splits, num_kv_heads, q_seq_lens_internal, 1),
        SENTINEL,
        dtype=torch.float32,
        device="cuda",
    )

    logits, attn_lse = aiter.mla.mla_decode_fwd_v4_nm(**args)
    torch.cuda.synchronize()  # force any async fault to surface here

    for b in range(num_seqs):
        ut = (logits[b] == SENTINEL).float().mean().item()
        assert ut < 0.01, (
            f"kernel skipped batch {b}: {ut * 100:.1f}% still SENTINEL. "
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
    written = logits[:, 0]  # [num_seqs, num_kv_heads, q_seq_lens_internal, dim_v]
    flat = written.reshape(-1, V_HEAD_DIM)  # rows of dim_v=512
    # If half the row is NaN and the other half is exactly zero, that's the
    # historic wave32-on-wave64 landmine (256 NaN + 256 zero per row).
    half = V_HEAD_DIM // 2
    first_half_nan_count = torch.isnan(flat[:, :half]).sum(dim=1).max().item()
    second_half_zero_count = (flat[:, half:] == 0.0).sum(dim=1).max().item()
    assert not (
        first_half_nan_count > half * 0.9 and second_half_zero_count > half * 0.9
    ), (
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
    assert torch.equal(written1, written2), (
        "v4 nm logits are non-deterministic (likely uninit accumulator/LDS)"
    )
    assert torch.equal(lse1_bits[:, 0], lse2.view(torch.int32)[:, 0]), (
        "v4 nm attn_lse is non-deterministic"
    )


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
        start = next(
            i for i, line in enumerate(lines) if line.startswith("[aiter kernarg 288B]")
        )
    except StopIteration:
        pytest.fail(
            "kernarg hexdump not found in stderr — "
            "AITER_V4_NM_DUMP_KERNARG env var may have been ignored, "
            "or jinja was changed and the dump code removed.\n"
            f"stderr was: {captured.err[:500]}"
        )
    hex_rows = []
    for line in lines[start + 1 : start + 1 + 18]:
        m = re.match(r"^((?:[0-9a-fA-F]{2}\s*){16})$", line.strip())
        if not m:
            break
        hex_rows.append(bytes.fromhex(line.strip().replace(" ", "")))
    assert len(hex_rows) == 18, f"expected 18 hex rows of kernarg, got {len(hex_rows)}"
    kargs = b"".join(hex_rows)
    assert len(kargs) == 288, f"kernarg byte total = {len(kargs)}, want 288"

    # Each slot is 16 bytes; first 4 bytes carry the payload, rest is padding.
    def slot(i):
        return kargs[i * 16 : i * 16 + 16]

    def slot_u32(i):
        return int.from_bytes(slot(i)[:4], "little")

    import struct

    def slot_f32(i):
        return struct.unpack("<f", slot(i)[:4])[0]

    # scalar_f is computed in jinja with C `float`s (1.0f/sqrtf(512.f)). Mirror
    # that precision here so the byte-exact compare doesn't false-fail on the
    # FP64→FP32 round-off difference.
    expected_scalar_f_bytes = struct.pack(
        "<f", float(np.float32(1.0) / np.float32(np.sqrt(np.float32(448 + 64))))
    )
    expected_gqa_ratio = GQA_RATIO  # 16
    expected_kv_split = 1  # num_kv_splits=1
    expected_total_kv = 64 * 2  # kv_seq_lens * batch
    expected_stride_pg = 1 * DIM_QK_PACKED  # page_size * 576
    expected_log2_page = 0  # log2(page_size=1)
    expected_out16ns = 0  # out_16_nosplit=0

    # slot 7 scalar_f: byte-exact compare (FP32)
    actual_scalar_f_bytes = slot(7)[:4]
    assert actual_scalar_f_bytes == expected_scalar_f_bytes, (
        f"slot 7 scalar_f bytes: got {actual_scalar_f_bytes.hex()}, "
        f"want {expected_scalar_f_bytes.hex()} (= 1/sqrt(512) in FP32)"
    )
    for slot_idx, want, name in [
        (8, expected_gqa_ratio, "s_gqa_ratio"),
        (9, expected_kv_split, "s_kv_split"),
        (10, expected_total_kv, "s_total_kv"),
        (11, expected_stride_pg, "s_stride_page"),
        (12, expected_log2_page, "s_log2_page"),
        (15, expected_out16ns, "out_16_nosplit"),
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


# ---------------------------------------------------------------------------
# Torch golden + accuracy + perf tests (resolves the TODO #1 in the file
# docstring). Mirrors op_tests/rui.py's torch reference and op_tests/test_mla.py's
# checkAllclose/run_perftest pattern. The ATOM-style wrapper below mirrors
# ATOM/atom/model_ops/v4_kernels/paged_decode.py::sparse_attn_v4_paged_decode
# so the asm op can drop in as a replacement for the triton fallback there.
# ---------------------------------------------------------------------------
from aiter import dtypes
from aiter.test_common import checkAllclose, run_perftest

# MODEL1_FP8Sparse layout (mirrored locally; not exported by aiter.ops.quant
# in this tree). Drives the per-token packing the v4 nm asm kernel expects.
_QUANT_D = 512  # full head dim = nope + rope
_QUANT_D_NOPE = 448  # FP8-quantized
_QUANT_D_ROPE = 64  # BF16 (kept separate in `qrope`/`kvrope` buffer)
_QUANT_TILE_SIZE = 64
_QUANT_NUM_TILES = _QUANT_D_NOPE // _QUANT_TILE_SIZE  # 7
# v4 nm kernel reads each tile's e8m0 scale TWICE in a row, so the scale
# block on disk is 14 bytes laid out as (s0,s0,s1,s1,...,s6,s6). Empirically
# verified: without the duplication V[256:448] of the asm output is all-zero
# and V[0:256] is partially correct, because scale reads land mid-pad.
_QUANT_NUM_SCALE_BYTES = _QUANT_NUM_TILES * 2  # 14


def _cast_scale_inv_to_ue8m0(t_input, out_dtype=torch.float32):
    """Round scale to 2^ceil(log2(scale)) — matches e8m0 storage."""
    return torch.pow(2, torch.clamp_min(t_input, 1e-4).log2().ceil()).to(out_dtype)


def _native_to_2buff_for_asm(input_bf16):
    """BF16 [..., 512] -> (nope_scale_buff [..., 512] fp8, rope_buff [..., 64] bf16).

    Per-token nope_scale_buff layout (matches the v4 nm asm kernel's reader):
      [ nope (448 fp8) | scale (14 e8m0; each tile-scale duplicated x2) | pad (50) ]
                                                                              = 512 B
      rope_buff = [ rope (64 bf16) ]                                         = 128 B

    NOTE: differs from op_tests/rui.py which writes 7 e8m0 bytes once. The
    v4 nm shader reads each tile's scale TWICE consecutively (s0,s0,s1,s1,
    ...,s6,s6); writing only 7 leaves the second-half scale reads landing in
    zero pad bytes, which empirically produced V[256:448] all-zero output.
    """
    assert input_bf16.shape[-1] == _QUANT_D
    leading = input_bf16.shape[:-1]
    nope = input_bf16[..., :_QUANT_D_NOPE]
    rope = input_bf16[..., _QUANT_D_NOPE:].contiguous()

    nope_scale_buff = torch.zeros(
        leading + (_QUANT_D,),
        dtype=dtypes.fp8,
        device=input_bf16.device,
    )
    nope_part = nope_scale_buff[..., :_QUANT_D_NOPE]
    scale_part = nope_scale_buff[
        ..., _QUANT_D_NOPE : _QUANT_D_NOPE + _QUANT_NUM_SCALE_BYTES
    ].view(dtypes.fp8_e8m0)

    fp8_max = torch.finfo(dtypes.fp8).max
    for t in range(_QUANT_NUM_TILES):
        s, e = t * _QUANT_TILE_SIZE, (t + 1) * _QUANT_TILE_SIZE
        tile = nope[..., s:e]
        scale_inv = torch.abs(tile).max(dim=-1).values.float() / fp8_max
        scale_inv = _cast_scale_inv_to_ue8m0(scale_inv)
        # Duplicate-write the scale: bytes [2t] and [2t+1] both hold s_t.
        scale_part[..., 2 * t] = scale_inv.to(dtypes.fp8_e8m0)
        scale_part[..., 2 * t + 1] = scale_inv.to(dtypes.fp8_e8m0)
        nope_part[..., s:e] = (tile.float() / scale_inv.unsqueeze(-1)).to(dtypes.fp8)

    return nope_scale_buff, rope


def _quant_2buff_to_native(nope_scale_buff, rope_buff):
    """Inverse of `_native_to_2buff_for_asm`. Returns BF16 [..., 512].

    Reads only the first byte of each duplicated scale pair (bytes [2t]); the
    second byte [2t+1] is a redundant copy written for the kernel's benefit.
    """
    leading = nope_scale_buff.shape[:-1]
    out = torch.empty(
        leading + (_QUANT_D,), dtype=dtypes.bf16, device=nope_scale_buff.device
    )
    nope_part = nope_scale_buff[..., :_QUANT_D_NOPE]
    scale_part = nope_scale_buff[
        ..., _QUANT_D_NOPE : _QUANT_D_NOPE + _QUANT_NUM_SCALE_BYTES
    ].view(dtypes.fp8_e8m0)
    for t in range(_QUANT_NUM_TILES):
        s, e = t * _QUANT_TILE_SIZE, (t + 1) * _QUANT_TILE_SIZE
        out[..., s:e] = nope_part[..., s:e].to(dtypes.bf16) * scale_part[..., 2 * t].to(
            dtypes.bf16
        ).unsqueeze(-1)
    out[..., _QUANT_D_NOPE:] = rope_buff
    return out


def _torch_attn_decode_bf16_golden(
    q_bf16,  # [total_q, num_heads, D=512]
    kv_bf16,  # [num_page, page_size=1, num_kv_heads=1, D=512]
    qo_indptr,  # [batch+1]   q rows per sequence (per-batch cumulative)
    kv_indptr,  # [batch+1]   pages per sequence (cumulative; page_size=1)
    kv_page_indices,  # [total_pages_used]
    kv_last_page_lens,  # [batch]
    sm_scale,
    attn_sink=None,  # [num_heads] or None
):
    """Pure-torch BF16 reference. Per-batch loop, scaled-dot-product attention
    with GQA broadcast (single KV head -> all Q heads). Returns
        out  [total_q, num_heads, D=512] bf16   (V dim == head dim for MLA)
        lse  [total_q, num_heads] bf16
    """
    num_heads = q_bf16.size(1)
    d = q_bf16.size(2)
    page_size = kv_bf16.size(1)
    assert page_size == 1, "this golden only supports page_size=1"

    total_q = q_bf16.size(0)
    out = torch.empty((total_q, num_heads, d), dtype=dtypes.bf16, device=q_bf16.device)
    lse_full = torch.empty(
        (total_q, num_heads), dtype=dtypes.bf16, device=q_bf16.device
    )
    batch = qo_indptr.size(0) - 1

    qo_indptr_cpu = qo_indptr.cpu().tolist()
    kv_indptr_cpu = kv_indptr.cpu().tolist()
    kv_last_cpu = kv_last_page_lens.cpu().tolist()

    for b in range(batch):
        qs, qe = qo_indptr_cpu[b], qo_indptr_cpu[b + 1]
        ps, pe = kv_indptr_cpu[b], kv_indptr_cpu[b + 1]
        num_pages_b = pe - ps
        if num_pages_b == 0:
            out[qs:qe] = 0
            lse_full[qs:qe] = float("+inf")
            continue
        page_ids = kv_page_indices[ps:pe]
        kv_pages = kv_bf16[page_ids]  # [num_pages_b, 1, 1, D]
        kv_flat = kv_pages.reshape(-1, 1, d)  # [num_pages_b*1, 1, D]
        total_tokens = (num_pages_b - 1) * page_size + kv_last_cpu[b]
        kv_b = kv_flat[:total_tokens].float()  # [seq_k, 1, D]
        kv_b = kv_b.expand(-1, num_heads, -1)  # GQA broadcast

        q_b = q_bf16[qs:qe].float()  # [s_q, H, D]
        scores = torch.einsum("shd,khd->shk", q_b, kv_b) * sm_scale  # [s_q, H, seq_k]

        if attn_sink is not None:
            # Sink as virtual K: contributes exp(sink_h) to the softmax denom only.
            lse = scores.logsumexp(dim=-1)  # [s_q, H]
            m = torch.maximum(lse, attn_sink.view(1, num_heads).float())
            denom = torch.exp(lse - m) + torch.exp(
                attn_sink.view(1, num_heads).float() - m
            )
            lse_final = m + torch.log(denom)
            probs = torch.exp(scores - lse_final.unsqueeze(-1))
        else:
            lse_final = scores.logsumexp(dim=-1)
            probs = torch.exp(scores - lse_final.unsqueeze(-1))

        v_b = kv_b  # MLA: V == K (first D dims)
        out_b = torch.einsum("shk,khv->shv", probs, v_b)  # [s_q, H, D]
        out[qs:qe] = out_b.to(dtypes.bf16)
        lse_full[qs:qe] = lse_final.to(dtypes.bf16)

    return out, lse_full


def _torch_attn_decode_fp8_dequant_ref(
    q_nope_scale,
    q_rope,
    kv_nope_scale,
    kv_rope,
    qo_indptr,
    kv_indptr,
    kv_page_indices,
    kv_last_page_lens,
    sm_scale,
    attn_sink=None,
):
    """Dequantize the same FP8 tensors the asm kernel sees, then call the
    BF16 golden. Isolates "kernel math bug" from "FP8 quant noise".
    """
    q_bf16 = _quant_2buff_to_native(q_nope_scale, q_rope)
    # kv: nope_scale_buff is [num_page, page_size, num_kv_heads, 512] -> dequant
    kv_bf16 = _quant_2buff_to_native(kv_nope_scale, kv_rope)
    return _torch_attn_decode_bf16_golden(
        q_bf16,
        kv_bf16,
        qo_indptr,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        sm_scale,
        attn_sink=attn_sink,
    )


def _asm_attn_decode_bf16(
    q_bf16,  # [total_q, num_heads=16, D=512] bf16
    kv_bf16,  # [num_page, page_size=1, num_kv_heads=1, D=512] bf16
    qo_indptr,
    kv_indptr,
    kv_page_indices,
    kv_last_page_lens,
    max_seqlen_q,
    sm_scale,
):
    """Quantize bf16 q/kv into the 2-buffer asm layout, call
    `aiter.mla.mla_decode_fwd_v4_nm`, and reduce/reshape the FP32 split
    logits back into a [total_q, num_heads, V_HEAD_DIM] BF16 tensor.

    Returns (out_bf16, logits, attn_lse, packed_buffers).

    Stride note: KV.size(3) is the per-token kernel stride in bytes. The
    kernel reads exactly 448 (nope) + 8 (scale) + slack = our 512-byte
    layout. Padding to 576 (poc_kl's stride_Page) made the kernel read
    garbage bytes as scale and produced all-NaN — DON'T pad.
    """
    total_q = q_bf16.size(0)
    num_heads = q_bf16.size(1)
    num_seqs = qo_indptr.size(0) - 1
    assert num_heads == GQA_RATIO

    q_packed, q_rope = _native_to_2buff_for_asm(
        q_bf16
    )  # [total_q, H, 512] / [.., 64] bf16
    kv_packed, kv_rope = _native_to_2buff_for_asm(kv_bf16)  # [P, 1, 1, 512] / [.., 64]

    # `output` is required by the C ABI even when reading from logits. The
    # kernel currently does not fully populate it (out_16_nosplit=1 path is
    # unverified at correctness), so we read from `logits` instead.
    output = torch.empty(
        (total_q, num_heads, V_HEAD_DIM), dtype=dtypes.bf16, device=q_bf16.device
    )
    num_kv_splits = 1
    split_indptr = torch.tensor(
        [i * num_kv_splits for i in range(num_seqs + 1)],
        dtype=torch.int32,
        device=q_bf16.device,
    )

    logits, attn_lse = aiter.mla.mla_decode_fwd_v4_nm(
        q=q_packed,
        qrope=q_rope.contiguous(),
        kv_buffer=kv_packed,
        kvrope=kv_rope.contiguous(),
        output=output,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_lens=kv_last_page_lens,
        split_indptr=split_indptr,
        max_seqlen_q=max_seqlen_q,
        sm_scale=sm_scale,  # ignored by kernel (hardcodes 1/sqrt(512))
        out_16_nosplit=0,
        num_kv_splits=num_kv_splits,
        sub_Q=SUB_Q,
    )
    # logits: [num_seqs, num_kv_splits=1, num_kv_heads=1, gqa*max_seqlen_q=64, D=512]
    # Internal row layout: row = q_token * gqa_ratio + head (empirically verified
    # by per-row compare against the torch golden — see the comparison test).
    # Reshape: [num_seqs, q_seq_logical, gqa, D] then flatten to [total_q, H, D].
    out_bf16 = (
        logits[:, 0, 0]
        .reshape(num_seqs, max_seqlen_q, num_heads, V_HEAD_DIM)
        .reshape(total_q, num_heads, V_HEAD_DIM)
        .to(dtypes.bf16)
    )
    return out_bf16, logits, attn_lse, (q_packed, q_rope, kv_packed, kv_rope)


def _cal_diff(x, y, name, cos_thresh):
    """RMSE / cosine-distance / amax. Pattern lifted from test_mla.py:28."""
    xd, yd = x.double(), y.double()
    rmse = ((xd - yd) ** 2).mean().sqrt().item()
    cos_diff = 1 - 2 * (xd * yd).sum().item() / max(
        (xd * xd + yd * yd).sum().item(), 1e-12
    )
    amax = (xd - yd).abs().max().item()
    print(f"  {name}: cos_diff={cos_diff:.4e}, RMSE={rmse:.4e}, amax={amax:.4e}")
    assert cos_diff < cos_thresh, (
        f"{name}: cos_diff={cos_diff:.4e} >= {cos_thresh:.1e} (RMSE={rmse:.4e}, amax={amax:.4e})"
    )


def _print_per_v_tile_diff(x_ref, y_asm, label):
    """Per-64-elem-tile summary of |asm|/|ref| over the V dim.

    Surfaces the "kernel only writes a subset of V tiles" failure mode
    (empirically: dims [256:448] currently come back zero, suggesting the
    kernel writes V_HEAD_DIM=256 of nope output + 64 of rope, leaving
    [256:448] unwritten). Run this whenever the cos_diff threshold
    trips so the gap is obvious without dropping into a debugger.
    """
    xd = x_ref.detach().float()
    yd = y_asm.detach().float()
    # collapse leading dims; we only care about the V axis (last dim).
    xf = xd.reshape(-1, xd.shape[-1])
    yf = yd.reshape(-1, yd.shape[-1])
    print(f"  {label} per-V-tile |asm| / |ref|:")
    for i in range(0, xf.shape[-1], 64):
        mref = xf[:, i : i + 64].abs().mean().item()
        masm = yf[:, i : i + 64].abs().mean().item()
        ratio = masm / mref if mref > 1e-12 else float("nan")
        max_diff = (xf[:, i : i + 64] - yf[:, i : i + 64]).abs().max().item()
        print(
            f"    V[{i:3d}:{i + 64:3d}]  |ref|={mref:.3e}  |asm|={masm:.3e}  "
            f"asm/ref={ratio:.3f}  max|diff|={max_diff:.3e}"
        )


def _build_bf16_inputs(
    batch=2,
    kv_seq_lens=64,
    q_seq_logical=4,
    seed=0,
    device="cuda",
    gqa_ratio=GQA_RATIO,
):
    """Build BF16 ground-truth q/kv and the aiter index tables. Output:
    q_bf16:           [total_q = batch*q_seq_logical, num_heads=gqa_ratio, D=512]
    kv_bf16:          [num_page = batch*kv_seq_lens, 1, 1, D=512]
    qo_indptr/kv_indptr/kv_page_indices/kv_last_page_lens — aiter convention.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    total_q = batch * q_seq_logical
    num_page = batch * (kv_seq_lens // PAGE_SIZE)

    # randn / clamp matches op_tests/rui.py's convention for sensible quant headroom.
    q_bf16 = torch.randn(
        (total_q, gqa_ratio, _QUANT_D), dtype=dtypes.bf16, device=device
    ).clamp_(-1.0, 1.0)
    kv_bf16 = (
        torch.randn(
            (num_page, PAGE_SIZE, NUM_KV_HEADS, _QUANT_D),
            dtype=dtypes.bf16,
            device=device,
        )
        / 10.0
    ).clamp_(-1.0, 1.0)

    qo_indptr = (
        torch.arange(0, batch + 1, dtype=torch.int32, device=device) * q_seq_logical
    )
    pages_per_seq = kv_seq_lens // PAGE_SIZE
    kv_indptr = (
        torch.arange(0, batch + 1, dtype=torch.int32, device=device) * pages_per_seq
    )
    kv_page_indices = torch.arange(
        0, batch * pages_per_seq, dtype=torch.int32, device=device
    )
    kv_last_page_lens = torch.full(
        (batch,), kv_seq_lens % PAGE_SIZE, dtype=torch.int32, device=device
    )
    # page_size=1: kv_last_page_lens must be in [1, page_size], so 1.
    kv_last_page_lens.fill_(1)

    return dict(
        q_bf16=q_bf16,
        kv_bf16=kv_bf16,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_lens=kv_last_page_lens,
        max_seqlen_q=q_seq_logical,
        kv_seq_lens=kv_seq_lens,
        batch=batch,
        q_seq_logical=q_seq_logical,
    )


def _run_one_point(
    batch=2,
    kv_seq_lens=64,
    q_seq_logical=4,
    seed=0,
    num_iters=50,
    num_warmup=3,
    num_kv_splits=1,
    gqa_ratio=GQA_RATIO,
):
    """One shape point: build inputs ONCE, time the asm kernel via
    run_perftest, then compare the last iter's output against the two torch
    references. Mirrors the merged accuracy+perf pattern in test_mla.py:382-413.

    Why num_rotate_args=1: skips both device_memory_profiling and the
    copy.deepcopy(args) fan-out in aiter/test_common.py:46-71, so the
    pre-allocated logits/lse buffers are reused across all iters. Without
    this, run_perftest's default rotation tries to deepcopy ~MB of tensors
    per iter and trips a GPU OOM (the reason the hand-rolled timer used to
    live here).
    """
    # gqa_ratio * q_seq_logical must equal SUB_Q (=64) — kernel tile invariant.
    assert gqa_ratio * q_seq_logical == SUB_Q, (
        f"gqa_ratio({gqa_ratio}) * q_seq_logical({q_seq_logical}) "
        f"must equal SUB_Q({SUB_Q})"
    )

    inputs = _build_bf16_inputs(
        batch=batch,
        kv_seq_lens=kv_seq_lens,
        q_seq_logical=q_seq_logical,
        seed=seed,
        gqa_ratio=gqa_ratio,
    )
    sm_scale = 1.0 / (_QUANT_D**0.5)  # kernel ignores; only used by torch ref

    # Torch references (CPU-side reference math, not timed).
    out_golden, _ = _torch_attn_decode_bf16_golden(
        inputs["q_bf16"],
        inputs["kv_bf16"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_page_indices"],
        inputs["kv_last_page_lens"],
        sm_scale,
    )

    # Pre-quantize once (Python quant helper is slow; would distort perf
    # if timed). Same FP8 bytes feed both the asm kernel and the fp8-dequant
    # ref so any diff between them isolates the kernel math.
    q_packed, q_rope = _native_to_2buff_for_asm(inputs["q_bf16"])
    kv_packed, kv_rope = _native_to_2buff_for_asm(inputs["kv_bf16"])

    # Pre-allocate everything the kernel writes into so the timed iters
    # don't allocate. Layout matches aiter/mla.py:1048.
    total_q = inputs["q_bf16"].size(0)
    num_seqs = inputs["qo_indptr"].size(0) - 1
    q_seq_lens_internal = gqa_ratio * inputs["max_seqlen_q"]
    output_buf = torch.empty(
        (total_q, gqa_ratio, V_HEAD_DIM), dtype=dtypes.bf16, device="cuda"
    )
    split_indptr = torch.tensor(
        [i * num_kv_splits for i in range(num_seqs + 1)],
        dtype=torch.int32,
        device="cuda",
    )
    logits_buf = torch.empty(
        (num_seqs, num_kv_splits, NUM_KV_HEADS, q_seq_lens_internal, V_HEAD_DIM),
        dtype=dtypes.fp32,
        device="cuda",
    )
    lse_buf = torch.empty(
        (num_seqs, num_kv_splits, NUM_KV_HEADS, q_seq_lens_internal, 1),
        dtype=dtypes.fp32,
        device="cuda",
    )

    # ---- timed call (1): torch fp8-dequant reference ----
    # Same fp8 bytes the kernel reads → isolates kernel math from quant noise,
    # and gives the speedup baseline. The ref does the dequant inside, so the
    # us number includes that cost — matches what the asm kernel does on-die.
    (out_fp8_ref, _lse_ref), us_ref = run_perftest(
        _torch_attn_decode_fp8_dequant_ref,
        q_packed,
        q_rope,
        kv_packed,
        kv_rope,
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_page_indices"],
        inputs["kv_last_page_lens"],
        sm_scale,
        num_iters=num_iters,
        num_warmup=num_warmup,
        num_rotate_args=1,
    )

    # ---- timed call (2): asm kernel ----
    # After it returns, logits_buf holds the LAST iter's output (kwargs pass
    # by reference). num_rotate_args=1 skips deepcopy + device_memory_profiling.
    _ret, us_asm = run_perftest(
        aiter.mla.mla_decode_fwd_v4_nm,
        q=q_packed,
        qrope=q_rope.contiguous(),
        kv_buffer=kv_packed,
        kvrope=kv_rope.contiguous(),
        output=output_buf,
        qo_indptr=inputs["qo_indptr"],
        kv_indptr=inputs["kv_indptr"],
        kv_page_indices=inputs["kv_page_indices"],
        kv_last_page_lens=inputs["kv_last_page_lens"],
        split_indptr=split_indptr,
        max_seqlen_q=inputs["max_seqlen_q"],
        sm_scale=sm_scale,
        out_16_nosplit=0,
        num_kv_splits=num_kv_splits,
        sub_Q=SUB_Q,
        logits=logits_buf,
        attn_lse=lse_buf,
        num_iters=num_iters,
        num_warmup=num_warmup,
        num_rotate_args=1,
    )

    # Host-side reshape of FP32 split logits → BF16 output (matches
    # _asm_attn_decode_bf16:677-682). row = q_token * gqa_ratio + head.
    # NOTE: only valid for num_kv_splits=1. For splits>1 the kernel writes
    # per-split partial sums + per-split LSE and the host must do an
    # LSE-weighted combine — not implemented here; accuracy compare is
    # skipped in that case (perf timing still meaningful).
    out_asm = (
        logits_buf[:, 0, 0]
        .reshape(num_seqs, inputs["max_seqlen_q"], gqa_ratio, V_HEAD_DIM)
        .reshape(total_q, gqa_ratio, V_HEAD_DIM)
        .to(dtypes.bf16)
    )

    # ---- accuracy ----
    # Two comparisons:
    #   [golden vs fp8_ref] = FP8 quant noise floor (kernel-independent)
    #   [fp8_ref vs asm]    = kernel math error (quant-independent)
    print(
        f"\n[v4 nm accuracy] batch={batch} kv_seq_lens={kv_seq_lens} "
        f"q_seq_logical={q_seq_logical} num_kv_splits={num_kv_splits} seed={seed}"
    )
    if num_kv_splits != 1:
        print(
            "  [skip] accuracy compare unsupported when num_kv_splits>1 "
            "(host-side LSE combine not implemented)."
        )
    else:
        _cal_diff(out_golden, out_fp8_ref, "[golden_bf16 vs fp8_ref]", cos_thresh=1.0)
        _cal_diff(out_fp8_ref, out_asm, "[fp8_ref vs asm]        ", cos_thresh=1.0)
    if num_kv_splits == 1:
        _cal_diff(out_fp8_ref, out_asm, "[ASSERT fp8_ref vs asm] ", cos_thresh=5e-3)
        checkAllclose(
            out_fp8_ref.float(),
            out_asm.float(),
            rtol=1e-2,
            atol=1e-2,
            msg="mla_v4_nm [fp8_dequant_ref vs asm]",
        )

    # ---- perf: fp8_ref vs asm ----
    total_kv = batch * kv_seq_lens
    flops = q_seq_logical * total_kv * gqa_ratio * (_QUANT_D + V_HEAD_DIM) * 2
    speedup = us_ref / us_asm if us_asm > 0 else float("inf")
    print(
        f"[v4 nm perf]     iters={num_iters}: "
        f"asm={us_asm:.2f} us ({flops / us_asm / 1e6:.2f} TFLOPS), "
        f"fp8_ref={us_ref:.2f} us, speedup={speedup:.1f}x"
    )
    return us_asm, us_ref


@needs_gfx950
def test_v4_nm_accuracy_and_perf():
    """Run the asm kernel via aiter.test_common.run_perftest at a fixed
    shape, then compare against both torch references and report timing
    in a single pass.

    Accuracy tolerances:
      [golden vs asm]   cos_diff < 3e-2  (FP8 quant headroom; test_mla.py:37)
      [fp8 vs asm]      cos_diff < 5e-3  (kernel-only; FP32-accum-order vs torch)
    Perf is informational (CI variance too high to assert).
    """
    _run_one_point(batch=2, kv_seq_lens=64, q_seq_logical=4, seed=0)


# ---------------------------------------------------------------------------
# ATOM-API wrapper (future drop-in replacement for ATOM's
# `sparse_attn_v4_paged_decode`). Lives in the test file as a *proof of API
# fit*; the production wrapper belongs in aiter/mla.py once exercised here.
# ---------------------------------------------------------------------------
def asm_sparse_attn_v4_paged_decode(
    q,  # [N, H=16, D=512] bf16
    unified_kv,  # [total_pages, D=512] bf16 (page_size=1, single KV head)
    kv_indices,  # [total_indices] int32 — per-token flat
    kv_indptr,  # [N+1] int32 — per-token prefix sum
    attn_sink,  # [H] or None
    softmax_scale,
):
    """Mirror of ATOM/atom/model_ops/v4_kernels/paged_decode.py::sparse_attn_v4_paged_decode.

    Constraints (current asm variant qh64/qseqlen4/gqa=16):
      - N (== total tokens) must be a multiple of 4.
      - Tokens are processed in groups of 4 as one "sequence" — tokens [b*4 ..
        (b+1)*4) MUST share the same kv span (i.e., kv_indptr is constant
        within each group of 4). Caller's responsibility.
      - attn_sink is currently unused (kernel does not honor sink); reserved
        for API parity. Pass `None` until kernel support lands.

    Returns: `out [N, H, D=512]` bf16.
    """
    assert q.dim() == 3 and q.size(1) == GQA_RATIO and q.size(2) == _QUANT_D
    assert unified_kv.dim() == 2 and unified_kv.size(1) == _QUANT_D
    n = q.size(0)
    assert n % 4 == 0, f"N={n} must be multiple of qseqlen=4 for this variant"
    if attn_sink is not None:
        raise NotImplementedError("asm v4 nm kernel does not honor attn_sink yet")

    batch = n // 4
    device = q.device

    # Per-batch aiter indices: one sequence per group-of-4 tokens.
    qo_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=device) * 4
    # kv_indptr at every 4th position (group's shared span); validate constancy.
    kv_indptr_per_seq = kv_indptr[::4].to(torch.int32).contiguous()
    assert kv_indptr_per_seq.size(0) == batch + 1, (
        f"kv_indptr layout invalid for groups-of-4: got len {kv_indptr.size(0)}, expected {batch * 4 + 1}"
    )
    # Sanity: within each group, kv_indptr must be constant relative to its base.
    for b in range(batch):
        base = int(kv_indptr[b * 4].item())
        for j in range(1, 4):
            assert int(kv_indptr[b * 4 + j].item()) == base, (
                f"asm v4 nm wrapper requires kv_indptr constant per group-of-4 (batch {b}, offset {j})"
            )

    kv_page_indices = kv_indices.to(torch.int32).contiguous()
    kv_last_page_lens = torch.ones(batch, dtype=torch.int32, device=device)

    # unified_kv [P, D] -> [P, page_size=1, num_kv_heads=1, D]
    kv_bf16 = unified_kv.view(-1, 1, 1, _QUANT_D)

    out, _, _, _ = _asm_attn_decode_bf16(
        q_bf16=q,
        kv_bf16=kv_bf16,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr_per_seq,
        kv_page_indices=kv_page_indices,
        kv_last_page_lens=kv_last_page_lens,
        max_seqlen_q=4,
        sm_scale=softmax_scale,
    )
    return out


if __name__ == "__main__":
    import argparse
    import itertools

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "v4 nm MLA DIY driver: for each shape in the (batch x kv x q_seq)\n"
            "cartesian product, run accuracy then perf. For the pytest smoke /\n"
            "determinism / kernarg suite, invoke `pytest op_tests/test_mla_v4_nm.py`\n"
            "directly."
        ),
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 2, 3, 4, 8, 16, 32, 64, 128, 256],
        help="Batch size(s). e.g. -b 1 2 4",
    )
    parser.add_argument(
        "-c",
        "--kv-seq-lens",
        type=int,
        nargs="*",
        default=[100, 256, 300, 512, 700, 1024],
        help="KV tokens per sequence. e.g. -c 64 256 1024",
    )
    parser.add_argument(
        "-q",
        "--q-seq-logical",
        type=int,
        nargs="*",
        default=[4],
        help="Q tokens per sequence (pre-GQA-broadcast). Must be <=4 for the "
        "shipped qseqlen4 variant. e.g. -q 1 2 4",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=50, help="Perf timed iterations")
    parser.add_argument("--warmup", type=int, default=3, help="Perf warmup iterations")
    parser.add_argument(
        "--split-kv",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--gqa-ratio",
        type=int,
        default=GQA_RATIO,
        help=f"num_heads / num_kv_heads. Must satisfy gqa_ratio * q_seq_logical "
        f"== SUB_Q({SUB_Q}). Registry currently ships only gqa_ratio=16; "
        "other values will hit 'no shipped variant' at dispatch.",
    )
    args = parser.parse_args()

    perf_rows = []
    for batch, kv_seq_lens, q_seq_logical in itertools.product(
        args.batch, args.kv_seq_lens, args.q_seq_logical
    ):
        print(
            f"\n========== batch={batch} kv_seq_lens={kv_seq_lens} "
            f"q_seq_logical={q_seq_logical} =========="
        )
        us_asm, us_ref = _run_one_point(
            batch=batch,
            kv_seq_lens=kv_seq_lens,
            q_seq_logical=q_seq_logical,
            seed=args.seed,
            num_iters=args.iters,
            num_warmup=args.warmup,
            num_kv_splits=args.split_kv,
            gqa_ratio=args.gqa_ratio,
        )
        perf_rows.append((batch, kv_seq_lens, q_seq_logical, us_asm, us_ref))

    print("\n[v4 nm perf summary] (us; speedup = fp8_ref / asm)")
    print(
        f"  {'batch':>6} {'kv_seq':>8} {'q_seq':>6} "
        f"{'asm us':>10} {'fp8_ref us':>12} {'speedup':>9}"
    )
    for b, k, q, ua, ur in perf_rows:
        print(f"  {b:>6d} {k:>8d} {q:>6d} {ua:>10.2f} {ur:>12.2f} {ur / ua:>8.1f}x")
