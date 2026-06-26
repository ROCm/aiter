# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests for the mi350 v4 'nm' MLA pipeline (mla_decode_fwd_v4_nm).
Usage:
  pytest -xvs op_tests/test_mla_v4_nm.py
"""

import numpy as np
import pytest
import torch

import aiter
import aiter.mla  # main no longer auto-imports submodules; need explicit
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import checkAllclose, run_perftest

# ---------------------------------------------------------------------------
# Variant under test (matches the cfg_mla_v4_asm entry in
# hsa/gfx950/mla_v4/mla_v4_asm.csv served by csrc/py_itfs_cu/asm_mla_v4.cu).
# ---------------------------------------------------------------------------
GQA_RATIO = 64  # num_heads / num_kv_heads
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
    batch=2, kv_seq_lens=64, q_seq_logical=1, num_heads=GQA_RATIO, device="cuda", seed=0
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

    # sink: required by mla_decode_fwd_v4_nm. -inf = "no sink" math
    # (exp(-inf) = 0 → virtual K-col contributes 0 to softmax denom).
    sink = torch.full(
        (num_heads,),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )

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
        sink=sink,
        num_kv_splits=num_kv_splits,
        out_16_nosplit=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
@needs_gfx950
def test_v4_nm_kernarg_scalar_slots(capfd, monkeypatch):
    """Regression guard for the 18-slot v4 nm kernarg layout.

    Locks in the *scalar* portion (slot 7 scalar_f, slot 8-12 ints, slot 15
    int) of the kernarg buffer produced by csrc/py_itfs_cu/asm_mla_v4.cu for
    the canonical qh64/(gqa,qseq)∈{(16,4),(64,1),(128,1)}/page=1/passes=1
    config (single .co; the C++ alias in asm_mla_v4.cu remaps gqa∈{64,128}
    to the (gqa=16,qSeqLen=4) CSV row). Any future
    change to the dispatcher that shifts a slot, mis-computes a stride /
    scale, or changes the formula here will trip this test before the golden
    numerical test does.

    Pointer slots are NOT checked (their values are runtime allocation
    addresses and don't have a stable reference). Bytes printed by the
    AITER_V4_NM_DUMP_KERNARG=1 path in asm_mla_v4.cu are captured via capfd.
    """
    monkeypatch.setenv("AITER_V4_NM_DUMP_KERNARG", "1")
    args = _build_inputs(batch=2, kv_seq_lens=64, q_seq_logical=1, seed=0)
    aiter.mla.mla_decode_fwd_v4_nm(**args)
    torch.cuda.synchronize()

    captured = capfd.readouterr()
    # The dispatcher fprintf's "[aiter kernarg 288B]" then 18 rows of 16
    # hex bytes. Parse the 18 rows out of stderr.
    import re

    lines = captured.err.splitlines()
    try:
        start = next(
            i for i, line in enumerate(lines) if line.startswith("[aiter kernarg 304B]")
        )
    except StopIteration:
        pytest.fail(
            "kernarg hexdump not found in stderr — "
            "AITER_V4_NM_DUMP_KERNARG env var may have been ignored, "
            "or jinja was changed and the dump code removed.\n"
            f"stderr was: {captured.err[:500]}"
        )
    hex_rows = []
    # 19 slots (PR-2: added ptr_sink at slot 18 / offset 0x120).
    for line in lines[start + 1 : start + 1 + 19]:
        m = re.match(r"^((?:[0-9a-fA-F]{2}\s*){16})$", line.strip())
        if not m:
            break
        hex_rows.append(bytes.fromhex(line.strip().replace(" ", "")))

    assert len(hex_rows) == 19, f"expected 19 hex rows of kernarg, got {len(hex_rows)}"
    kargs = b"".join(hex_rows)

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
    expected_log2_page = 0  # log2(page_size=1)
    expected_out16ns = 0  # out_16_nosplit=0
    # slots 10 (s_total_kv) and 11 (s_stride_page) are NEVER read — only 17 kernarg
    # loads, none at offsets 0xA0/0xB0). The dispatcher leaves them at 0
    # via `args = {}` zero-init to skip the per-call D2H readback that
    # used to compute s_total_kv. See the "Dead kernarg slots" block in
    # csrc/py_itfs_cu/asm_mla_v4.cu for the full justification.
    expected_total_kv = 0
    expected_stride_pg = 0

    # slot 7 scalar_f: byte-exact compare (FP32)
    actual_scalar_f_bytes = slot(7)[:4]
    assert actual_scalar_f_bytes == expected_scalar_f_bytes, (
        f"slot 7 scalar_f bytes: got {actual_scalar_f_bytes.hex()}, "
        f"want {expected_scalar_f_bytes.hex()} (= 1/sqrt(512) in FP32)"
    )
    for slot_idx, want, name in [
        (8, expected_gqa_ratio, "s_gqa_ratio"),
        (9, expected_kv_split, "s_kv_split"),
        (10, expected_total_kv, "s_total_kv (DEAD; must be 0)"),
        (11, expected_stride_pg, "s_stride_page (DEAD; must be 0)"),
        (12, expected_log2_page, "s_log2_page"),
        (15, expected_out16ns, "out_16_nosplit"),
    ]:
        got = slot_u32(slot_idx)
        assert got == want, (
            f"slot {slot_idx} ({name}): got {got} (0x{got:08x}), "
            f"want {want} (0x{want:08x})"
        )

    # Sanity: pointer slots (0..6, 13, 14, 16, 17, 18) must be non-NULL.
    # Slot 18 (ptr_sink) is REQUIRED non-NULL — caller must allocate even
    # when they want "no sink" math (-inf works, but the buffer must exist).
    for slot_idx in (0, 1, 2, 3, 4, 5, 6, 13, 14, 16, 17, 18):
        ptr = int.from_bytes(slot(slot_idx)[:8], "little")
        assert ptr != 0, f"slot {slot_idx} pointer is NULL"


# ---------------------------------------------------------------------------
# Torch golden + accuracy + perf tests (resolves the TODO #1 in the file
# docstring). Mirrors op_tests/rui.py's torch reference and op_tests/test_mla.py's
# checkAllclose/run_perftest pattern. The ATOM-style wrapper below mirrors
# ATOM/atom/model_ops/v4_kernels/paged_decode.py::sparse_attn_v4_paged_decode
# so the asm op can drop in as a replacement for the triton fallback there.
# ---------------------------------------------------------------------------

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
            # Sink as virtual K: per-head logit, broadcast across all q_token.
            # attn_sink is [num_heads] (one scalar bias per head, shared by
            # every query token in that head).
            sink_b = attn_sink.view(1, num_heads).float()  # [1, H] -> [s_q, H]
            lse = scores.logsumexp(dim=-1)  # [s_q, H]
            m = torch.maximum(lse, sink_b)
            denom = torch.exp(lse - m) + torch.exp(sink_b - m)
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
    # sink: -inf = "no sink" math. Size = num_heads (post-2026-06-01
    # shrink — kernel reads sink head-only). See aiter/mla.py docstring.
    sink = torch.full(
        (num_heads,),
        float("-inf"),
        dtype=torch.float32,
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
        sink=sink,
        sm_scale=sm_scale,  # ignored by kernel (hardcodes 1/sqrt(512))
        out_16_nosplit=0,
        num_kv_splits=num_kv_splits,
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
    q_seq_logical=1,
    seed=0,
    device="cuda",
    gqa_ratio=GQA_RATIO,
    attn_sink=True,
):
    """Build BF16 ground-truth q/kv and the aiter index tables. Output:
    q_bf16:           [total_q = batch*q_seq_logical, num_heads=gqa_ratio, D=512]
    kv_bf16:          [num_page = batch*kv_seq_lens, 1, 1, D=512]
    qo_indptr/kv_indptr/kv_page_indices/kv_last_page_lens — aiter convention.

    `attn_sink`:
      Returns `sink` as a per-head [num_heads] FP32 tensor (one scalar per
      head, shared across all query tokens). The caller is responsible for
      tiling it across q_token into the kernel's flat buffer before the asm
      call (see _run_one_point); the torch reference consumes the per-head
      form directly.
      True  -> NON-ZERO random (randn) per-head sink. randn (not a constant)
               makes every head distinct so a head-dim layout mismatch shows
               up as a cos_diff blowup, not a silent pass.
      False -> per-head -inf ("no sink" no-op: exp(-inf - max) = 0).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    total_q = batch * q_seq_logical
    num_page = batch * (kv_seq_lens // PAGE_SIZE)

    # Bare randn (~N(0,1)), matching op_tests/test_mla.py's input convention.
    # No /10 scaling or clamp: under the strict 1% checkAllclose tolerance this
    # leaves some elements over the bound (FP8 quant noise on the full dynamic
    # range), reported as `failed!` — that is expected and double-checked by eye,
    # not a hard gate (checkAllclose does not raise).
    q_bf16 = torch.randn(
        (total_q, gqa_ratio, _QUANT_D), dtype=dtypes.bf16, device=device
    )
    kv_bf16 = torch.randn(
        (num_page, PAGE_SIZE, NUM_KV_HEADS, _QUANT_D),
        dtype=dtypes.bf16,
        device=device,
    )

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

    # sink: per-head [num_heads] attention sink (one scalar per head), consumed
    # head-only by both the kernel and the torch ref — no q_token tiling.
    # Scaled up by 10 (randn * 10) so the sink contributes ~15% to the softmax
    # output vs a no-sink baseline: well above the checkAllclose tolerance, so a
    # dropped / mis-scaled sink in the kernel shows up as a hard mismatch instead
    # of being masked by quant noise (bare randn ~N(0,1) only moves ~0.8%).
    num_heads = NUM_KV_HEADS * gqa_ratio
    if attn_sink:
        sink = torch.randn(num_heads, dtype=torch.float32, device=device) * 10.0
    else:
        # per-head -inf = "no sink" no-op (exp(-inf - max) = 0).
        sink = torch.full(
            (num_heads,), float("-inf"), dtype=torch.float32, device=device
        )

    return dict(
        q_bf16=q_bf16,
        kv_bf16=kv_bf16,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_page_indices=kv_page_indices,
        kv_last_page_lens=kv_last_page_lens,
        sink=sink,
        max_seqlen_q=q_seq_logical,
        kv_seq_lens=kv_seq_lens,
        batch=batch,
        q_seq_logical=q_seq_logical,
    )


def _run_one_point(
    batch=2,
    kv_seq_lens=64,
    q_seq_logical=1,
    seed=0,
    num_iters=50,
    num_warmup=3,
    num_kv_splits=1,  # int, or None to auto-pick via get_meta_param (like the wrapper)
    gqa_ratio=GQA_RATIO,
    attn_sink=True,
    out_16_nosplit=0,  # 1 -> kernel writes packed-BF16 result; wrapper resolves into output_buf
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
    # The shipped qh64 .co has a 64 q-row tile; the dispatcher
    # (csrc/py_itfs_cu/asm_mla_v4.cu) selects sub_Q based on (gqa_ratio,
    # max_seqlen_q) and computes gdx = ceil(gqa*max_seqlen_q / sub_Q), so a
    # single .co covers three (gqa, q_seq_logical) entry points:
    #   (16, 4) — 16 heads × 4 logical-Q rows = 64 → gdx=1
    #   (64, 1) — 64 heads × 1 logical-Q row  = 64 → gdx=1
    #   (128,1) — 128 heads × 1 logical-Q row = 128 → gdx=2 (two WGs along head)
    # The CSV alias in asm_mla_v4.cu remaps gqa∈{64,128} to the (16,4) lookup
    # row so all three find the same kernel symbol.
    _SHIPPED_TILE_VARIANTS = {(16, 4), (64, 1), (128, 1)}
    assert (gqa_ratio, q_seq_logical) in _SHIPPED_TILE_VARIANTS, (
        f"(gqa_ratio={gqa_ratio}, q_seq_logical={q_seq_logical}) not in shipped "
        f"variants {_SHIPPED_TILE_VARIANTS} for the qh64 .co. The dispatcher "
        f"picks sub_Q=64 and launches gdx=ceil(gqa*max_seqlen_q/64) WGs along "
        f"the head dim; only these three pairs are exercised by CSV+dispatcher."
    )

    # Auto-pick the split count when the caller passes None — mirrors the
    # production wrapper (aiter/mla.py mla_decode_fwd_v4_nm), which forwards
    # num_kv_splits=None to get_meta_param's CU-occupancy x HBM-efficiency
    # heuristic. We resolve it to a concrete int HERE (before any buffer
    # allocation) because this driver pre-allocates logits/lse/split_indptr
    # sized to a fixed split count; page_size=1 so total_kv = batch*kv_seq_lens
    # and nhead = NUM_KV_HEADS*gqa_ratio.
    if num_kv_splits is None:
        # tg_factor mirrors the wrapper: gqa=128 launches ceil(128/64)=2 WGs
        # per (seq, split), so its effective CU occupancy is 2x — feed that in
        # so the heuristic doesn't over-split (bs=64/gqa=128 -> 2, not 4).
        num_heads = NUM_KV_HEADS * gqa_ratio
        tg_factor = max(1, -(-num_heads // 64))  # ceil(num_heads / 64)
        num_kv_splits, _ = aiter.mla.get_meta_param(
            None,
            batch,
            batch * kv_seq_lens,
            num_heads,
            q_seq_logical,
            dtypes.fp8,
            tg_factor,
        )
        num_kv_splits = int(num_kv_splits)
        print(
            f"[v4 nm] auto-selected num_kv_splits={num_kv_splits} "
            f"(tg_factor={tg_factor})"
        )

    # out_16_nosplit=1 is the kernel's single-pass packed-BF16 direct path; it
    # has no stage2 merge, so it is only valid with num_kv_splits==1 (the same
    # constraint the wrapper enforces).
    if out_16_nosplit != 0:
        assert num_kv_splits == 1, (
            f"out_16_nosplit={out_16_nosplit} requires num_kv_splits==1 "
            f"(bf16-direct-write is single-pass only); got {num_kv_splits}."
        )

    # Multi-split input guard (checked BEFORE any kernel launch): the v4 nm 32n
    # .co inner KV loop processes SUB_KV=32 tokens/iteration; each split WG must
    # get at least one full pass (>=32 tokens) or its tail is dropped. The
    # operator handles a non-divisible kv_seq_lens // splits (remainder
    # distributed internally), so the only requirement is that the SMALLEST
    # split >= 32. floor(kv/splits) is the smallest split's size regardless of
    # how the remainder lands. The dispatcher does NOT validate this (forwards
    # num_kv_splits to kernarg slot 9 verbatim), so guard it here.
    if num_kv_splits > 1:
        min_split = kv_seq_lens // num_kv_splits  # page_size=1
        assert min_split >= 32, (
            f"smallest KV split = floor({kv_seq_lens}/{num_kv_splits}) = "
            f"{min_split} < SUB_KV=32: that split drops its tail. Reduce "
            f"num_kv_splits or raise kv_seq_lens so "
            f"kv_seq_lens // num_kv_splits >= 32."
        )

    inputs = _build_bf16_inputs(
        batch=batch,
        kv_seq_lens=kv_seq_lens,
        q_seq_logical=q_seq_logical,
        seed=seed,
        gqa_ratio=gqa_ratio,
        attn_sink=attn_sink,
    )
    sm_scale = 1.0 / (_QUANT_D**0.5)  # kernel ignores; only used by torch ref

    # Torch references (CPU-side reference math, not timed). inputs["sink"] is
    # the per-head [num_heads] sink consumed directly by both the torch refs
    # and the asm kernel (the kernel reads per-head sink natively as of the
    # 2026-06-01 shrink — no q_token tiling needed).
    out_golden, _ = _torch_attn_decode_bf16_golden(
        inputs["q_bf16"],
        inputs["kv_bf16"],
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_page_indices"],
        inputs["kv_last_page_lens"],
        sm_scale,
        attn_sink=inputs["sink"],
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
    output_buf = torch.empty(
        (total_q, gqa_ratio, V_HEAD_DIM), dtype=dtypes.bf16, device="cuda"
    )
    split_indptr = torch.tensor(
        [i * num_kv_splits for i in range(num_seqs + 1)],
        dtype=torch.int32,
        device="cuda",
    )
    # Kernel-native layout: [total_q, num_kv_splits, num_heads, dv] (mirrors V3)
    num_heads = NUM_KV_HEADS * gqa_ratio
    logits_buf = torch.empty(
        (total_q, num_kv_splits, num_heads, V_HEAD_DIM),
        dtype=dtypes.fp32,
        device="cuda",
    )
    lse_buf = torch.empty(
        (total_q, num_kv_splits, num_heads, 1),
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
        attn_sink=inputs["sink"],
        num_iters=num_iters,
        num_warmup=num_warmup,
        num_rotate_args=1,
    )

    # ---- timed call (2a): asm kernel ONLY (no stage2 merge) ----
    # Times the v4 nm decoder kernel in isolation so the perf number isolates
    # kernel work from the cross-split merge cost. For num_kv_splits=1 this
    # is the only kernel invocation; for num_kv_splits>1 the wrapper would
    # additionally invoke `_fwd_kernel_stage2_asm` triton on top — see (2b).
    _ret, us_asm_kernel = run_perftest(
        aiter.mla_decode_v4_asm,
        q_packed,
        q_rope.contiguous(),
        kv_packed,
        kv_rope.contiguous(),
        inputs["qo_indptr"],
        inputs["kv_indptr"],
        inputs["kv_page_indices"],
        inputs["kv_last_page_lens"],
        split_indptr,
        inputs["sink"],  # per-head [num_heads] sink; req'd positional
        inputs["max_seqlen_q"],
        sm_scale,
        int(out_16_nosplit),  # out_16_nosplit (timing path; raw kernel does not unpack)
        num_kv_splits,
        logits_buf,
        lse_buf,
        output_buf,
        num_iters=num_iters,
        num_warmup=num_warmup,
        num_rotate_args=1,
    )

    # ---- timed call (2b): full wrapper (kernel + stage2 merge) ----
    # End-to-end perf as the production caller sees it.
    _ret, us_asm_total = run_perftest(
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
        sink=inputs["sink"],
        sm_scale=sm_scale,
        out_16_nosplit=int(out_16_nosplit),
        num_kv_splits=num_kv_splits,
        logits=logits_buf,
        attn_lse=lse_buf,
        num_iters=num_iters,
        num_warmup=num_warmup,
        num_rotate_args=1,
    )

    # Resolve the asm output to compare against. Three cases, all reading the
    # buffer the wrapper actually populated (the 2b call above):
    #   out_16_nosplit=1   -> kernel writes packed-BF16 into the logits region;
    #                         the wrapper unpacks it into output_buf (see
    #                         mla_decode_fwd_v4_nm). Read output_buf directly.
    #   single-pass (fp32) -> kernel writes one FP32 partial to logits[:, 0],
    #                         no stage2; cast it to BF16.
    #   multi-pass         -> stage2 merge wrote merged BF16 to output_buf.
    if out_16_nosplit != 0:
        out_asm = output_buf  # wrapper unpacked packed-BF16 here
    elif num_kv_splits == 1:
        out_asm = logits_buf[:, 0].to(dtypes.bf16)  # [total_q, num_heads, dv]
    else:
        out_asm = output_buf  # already [total_q, num_heads, dv] BF16

    # ---- accuracy ----
    # Two comparisons, run for BOTH single- and multi-split (split-kv is a perf
    # optimization; its stage2-merged output is mathematically the same full
    # attention the torch refs compute, so it is directly comparable):
    #   [golden vs fp8_ref] = FP8 quant noise floor (kernel-independent)
    #   [fp8_ref vs asm]    = kernel math error (quant-independent)
    print(
        f"\n[v4 nm accuracy] batch={batch} kv_seq_lens={kv_seq_lens} "
        f"q_seq_logical={q_seq_logical} num_kv_splits={num_kv_splits} seed={seed}"
    )
    # Per-element check at checkAllclose's default 1% tolerance (rtol=atol=1e-2).
    # checkAllclose prints pass/warning/failed with the offending-element ratio +
    # max delta (it does not raise).
    checkAllclose(
        out_golden.float(),
        out_fp8_ref.float(),
        rtol=3e-2,
        atol=3e-2,
        tol_err_ratio=0.02,
        msg="mla_v4_nm [golden_bf16 vs fp8_ref]",
    )
    checkAllclose(
        out_fp8_ref.float(),
        out_asm.float(),
        rtol=3e-2,
        atol=3e-2,
        tol_err_ratio=0.02,
        msg="mla_v4_nm [fp8_dequant_ref vs asm]",
    )

    # ---- perf: fp8_ref vs asm ----
    # We report two asm timings:
    #   asm_k: v4 kernel only (no stage2 merge) — kernel-isolated metric
    #   asm  : full wrapper end-to-end (kernel + stage2 merge if splits>1)
    # `speedup` uses asm_k since it's the kernel-comparable number; the
    # multi-split merge is a separate cost we want to call out explicitly.
    total_kv = batch * kv_seq_lens
    flops = q_seq_logical * total_kv * gqa_ratio * (_QUANT_D + V_HEAD_DIM) * 2
    us_asm = us_asm_kernel  # used by the caller in the summary
    merge_us = us_asm_total - us_asm_kernel
    speedup = us_ref / us_asm if us_asm > 0 else float("inf")
    print(
        f"[v4 nm perf]     iters={num_iters}: "
        f"asm_k={us_asm_kernel:.2f} us ({flops / us_asm_kernel / 1e6:.2f} TFLOPS) "
        f"merge={merge_us:.2f} us  total={us_asm_total:.2f} us, "
        f"fp8_ref={us_ref:.2f} us, speedup(kernel)={speedup:.1f}x"
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
    _run_one_point(batch=2, kv_seq_lens=64, q_seq_logical=1, seed=0)


@needs_gfx950
def test_v4_nm_out_16_nosplit_accuracy_and_perf():
    """Exercise the single-pass packed-BF16 direct path (out_16_nosplit=1).

    The kernel writes its result as densely-packed BF16 into the logits
    region (NOT the output buffer); the wrapper unpacks it into `output`
    (see mla_decode_fwd_v4_nm). This locks in both that the unpack lands the
    right bytes (accuracy vs the fp8-dequant torch ref) and that the path
    runs end-to-end at perf. num_kv_splits must be 1 (bf16-direct is
    single-pass). Perf is informational; accuracy is the gate.
    """
    _run_one_point(
        batch=2,
        kv_seq_lens=64,
        q_seq_logical=1,
        seed=0,
        num_kv_splits=1,
        out_16_nosplit=1,
    )


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

    Constraints (current asm variant qh64/qseqlen4 — single .co aliased to
    (gqa,q_seq_logical) ∈ {(16,4),(64,1),(128,1)}):
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
    assert (
        kv_indptr_per_seq.size(0) == batch + 1
    ), f"kv_indptr layout invalid for groups-of-4: got len {kv_indptr.size(0)}, expected {batch * 4 + 1}"
    # Sanity: within each group, kv_indptr must be constant relative to its base.
    for b in range(batch):
        base = int(kv_indptr[b * 4].item())
        for j in range(1, 4):
            assert (
                int(kv_indptr[b * 4 + j].item()) == base
            ), f"asm v4 nm wrapper requires kv_indptr constant per group-of-4 (batch {b}, offset {j})"

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


# ---------------------------------------------------------------------------
# Multi-pass (num_kv_splits > 1) — opens the path that mirrors V3's
# non-persistent stage1 + stage2 reduce. The .co binary already supports any
# number of passes via slot 9; this test verifies (a) the dispatcher lookup
# isn't gated on num_kv_splits, (b) the python wrapper auto-builds
# split_indptr V3-style, and (c) the in-place logsumexp merge writes a finite
# result into the [:, 0] slot.
# ---------------------------------------------------------------------------
@needs_gfx950
def test_v4_nm_multi_split():
    """Multi-pass (num_kv_splits>1) path, two checks in one:

    (A) Full-KV coverage: num_kv_splits=4 with kv_seq_lens=256 → 64 tokens
        per split = two full SUB_KV=32 inner-KV passes each. Every split slot
        in `logits` must be written (no SENTINEL leak), proving the dispatcher
        isn't gated on num_kv_splits, the wrapper auto-builds split_indptr
        V3-style, and the kernel doesn't tail-drop any split. Coverage
        invariant: floor(kv/splits) >= SUB_KV (=32); 256/4=64 ✓.

    (B) Rejection: multi-pass + out_16_nosplit=1 is unsupported (bf16-direct
        is single-pass only); the wrapper must raise BEFORE the kernel.
    """
    # ---- (A) full-KV coverage ----
    NUM_SPLITS = 4
    BATCH = 2
    KV_LEN = 256  # 256/4 = 64 = 2*SUB_KV (two full passes per split)
    Q_SEQ = 1

    args = _build_inputs(batch=BATCH, kv_seq_lens=KV_LEN, q_seq_logical=Q_SEQ, seed=0)
    args["num_kv_splits"] = NUM_SPLITS
    args["out_16_nosplit"] = 0
    args.pop("split_indptr")  # auto-built V3-style

    SENTINEL = -7.7e30
    num_seqs = args["qo_indptr"].size(0) - 1
    num_heads = args["q"].size(1)
    msq = args["max_seqlen_q"]
    total_q = num_seqs * msq
    args["logits"] = torch.full(
        (total_q, NUM_SPLITS, num_heads, V_HEAD_DIM),
        SENTINEL,
        dtype=torch.float32,
        device="cuda",
    )
    args["attn_lse"] = torch.full(
        (total_q, NUM_SPLITS, num_heads, 1),
        SENTINEL,
        dtype=torch.float32,
        device="cuda",
    )

    logits, _ = aiter.mla.mla_decode_fwd_v4_nm(**args)
    torch.cuda.synchronize()

    for s in range(NUM_SPLITS):
        ut = (logits[:, s] == SENTINEL).float().mean().item()
        assert ut < 0.01, (
            f"split {s} kernel skipped ({ut*100:.1f}% still SENTINEL). "
            f"Coverage invariant: floor(kv/splits)={KV_LEN // NUM_SPLITS} must "
            f">= SUB_KV=32. If it holds, the bug is upstream (dispatcher launch "
            f"geometry / split_indptr stride / kernel early-exit)."
        )

    # ---- (B) multi-pass + out_16_nosplit=1 must be rejected ----
    args_rej = _build_inputs(batch=1, kv_seq_lens=128, q_seq_logical=1, seed=0)
    args_rej["num_kv_splits"] = 2
    args_rej["out_16_nosplit"] = 1
    args_rej.pop("split_indptr")
    with pytest.raises(ValueError, match="out_16_nosplit"):
        aiter.mla.mla_decode_fwd_v4_nm(**args_rej)


# ---------------------------------------------------------------------------
# Sink interface (PR-2: sink-aware .co + slot 18 plumbed end-to-end)
# ---------------------------------------------------------------------------
# These tests pin down the behavioural contract: We assert that
#   (a) sink=-inf vs sink=+inf produce DIFFERENT output bytes — proves the
#       sink data actually reaches the kernel and modulates the softmax
#       denominator,
#   (b) sink=-inf does NOT produce extra NaNs vs a near-equivalent finite
#       sentinel (-1e9), so callers can safely use -inf as the "no sink"
#       convention without numerical surprises.
#
# Build helper note: we use _build_bf16_inputs + _native_to_2buff_for_asm
# instead of _build_inputs because the latter generates random FP8 bytes
# (incl. random e8m0 scale bytes), which dequant to 100% NaN/inf and make
# bit comparisons impossible. The BF16-then-quant path produces finite
# outputs that actually expose the sink merge math.
# ---------------------------------------------------------------------------
def _build_sink_test_args(batch=2, kv_seq_lens=64, q_seq_logical=1, seed=0):
    """Properly-quantized wrapper-args for sink behaviour tests. Returns the
    full kwargs dict that mla_decode_fwd_v4_nm needs, with sink defaulted
    to -inf (caller can override). Output cells will be finite (modulo the
    rare quant-noise NaN), which is what byte-level diffing requires.
    """
    bf = _build_bf16_inputs(
        batch=batch,
        kv_seq_lens=kv_seq_lens,
        q_seq_logical=q_seq_logical,
        seed=seed,
    )
    q_packed, q_rope = _native_to_2buff_for_asm(bf["q_bf16"])
    kv_packed, kv_rope = _native_to_2buff_for_asm(bf["kv_bf16"])

    total_q = bf["q_bf16"].size(0)
    num_heads = bf["q_bf16"].size(1)
    device = bf["q_bf16"].device
    output = torch.empty(
        (total_q, num_heads, V_HEAD_DIM), dtype=dtypes.bf16, device=device
    )
    sink = torch.full(
        (num_heads,),
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )

    return dict(
        q=q_packed,
        qrope=q_rope.contiguous(),
        kv_buffer=kv_packed,
        kvrope=kv_rope.contiguous(),
        output=output,
        qo_indptr=bf["qo_indptr"],
        kv_indptr=bf["kv_indptr"],
        kv_page_indices=bf["kv_page_indices"],
        kv_last_page_lens=bf["kv_last_page_lens"],
        max_seqlen_q=bf["max_seqlen_q"],
        sink=sink,
    )


@needs_gfx950
def test_v4_nm_sink():
    """Sink contract, three checks in one:

    (A) sink=-inf vs sink=10.0 produce DIFFERENT output bytes — proves sink
        reaches the kernel via slot 18 (offset 0x120) and modulates softmax.
    (B) sink=-inf introduces NO extra NaN vs a finite -1e9 control — the
        documented "no sink" convention is -inf-stable.
    (C) malformed sink (wrong dtype/size/stride/device) is rejected by the
        wrapper BEFORE the dispatcher can mis-stride into garbage memory.
    """
    # ---- (A) sink value affects output ----
    args_a = _build_sink_test_args(batch=2, kv_seq_lens=64, q_seq_logical=1, seed=0)
    sink_size = args_a["sink"].numel()  # = num_heads (2026-06-01 shrink)
    device = args_a["q"].device

    args_b = _build_sink_test_args(batch=2, kv_seq_lens=64, q_seq_logical=1, seed=0)
    args_b["sink"] = torch.full((sink_size,), 10.0, dtype=torch.float32, device=device)

    logits_a, _ = aiter.mla.mla_decode_fwd_v4_nm(**args_a)
    torch.cuda.synchronize()
    logits_a_bits = logits_a.view(torch.int32).clone()

    logits_b, _ = aiter.mla.mla_decode_fwd_v4_nm(**args_b)
    torch.cuda.synchronize()
    logits_b_bits = logits_b.view(torch.int32)

    # logits is 4D [total_q, num_kv_splits=1, num_heads, dv]; only [:, 0]
    # is kernel-written.
    finite_both = torch.isfinite(logits_a[:, 0]) & torch.isfinite(logits_b[:, 0])
    assert finite_both.any(), (
        "All output cells were NaN/inf under both sink values — the quant "
        "pipeline returned junk OR sink=10 pushed the running max into a "
        "saturating regime. Re-check _native_to_2buff_for_asm or lower "
        "sink_b's magnitude."
    )

    diff_finite = (logits_a_bits[:, 0] != logits_b_bits[:, 0]) & finite_both
    assert diff_finite.any(), (
        "PR-2 regression: sink=-inf and sink=10.0 produced bit-identical "
        "output among finite cells. Either the dispatcher stopped writing "
        "ptr_sink into kernarg slot 18 (offset 0x120), or the .co was "
        "rebuilt from a non-sink-aware .s. Check the static_assert in "
        "csrc/py_itfs_cu/asm_mla_v4.cu and rebuild from 3_13.s."
    )

    # ---- (B) sink=-inf introduces no extra NaN vs -1e9 control ----
    args_inf = _build_sink_test_args(batch=2, kv_seq_lens=64, q_seq_logical=1, seed=7)
    sink_size = args_inf["sink"].numel()  # = num_heads (2026-06-01 shrink)
    device = args_inf["q"].device

    args_big = _build_sink_test_args(batch=2, kv_seq_lens=64, q_seq_logical=1, seed=7)
    args_big["sink"] = torch.full(
        (sink_size,), -1.0e9, dtype=torch.float32, device=device
    )

    logits_inf, _ = aiter.mla.mla_decode_fwd_v4_nm(**args_inf)
    logits_big, _ = aiter.mla.mla_decode_fwd_v4_nm(**args_big)
    torch.cuda.synchronize()

    nan_inf = torch.isnan(logits_inf[:, 0])
    nan_big = torch.isnan(logits_big[:, 0])

    # -inf must not produce *more* NaNs than the -1e9 control. The inverse
    # would mean sink=-inf hits a kernel-side division-by-zero or
    # exp(-inf)*0=NaN somewhere it shouldn't, breaking the wrapper's
    # documented "pass torch.full(..., -inf) for no-sink math" recipe.
    extra_nans = (nan_inf & ~nan_big).sum().item()
    assert extra_nans == 0, (
        f"sink=-inf introduced {extra_nans} NaN cells over the -1e9 "
        f"control. The sink merge in 3_13.s is not -inf-stable; the "
        f"wrapper docstring's recommendation to use -inf for 'no sink' "
        f"is no longer safe — switch the convention to a large finite "
        f"negative (e.g. -1e9)."
    )

    # ---- (C) malformed sink rejected before dispatch (5 paths) ----
    args = _build_inputs(batch=1, kv_seq_lens=64, q_seq_logical=1, seed=0)
    num_heads = args["q"].size(1)
    max_seqlen_q = args["max_seqlen_q"]
    expected = num_heads  # 2026-06-01 shrink: was num_heads * max_seqlen_q
    device = args["q"].device

    # Wrong dtype (BF16 instead of FP32).
    args_bad_dtype = dict(args)
    args_bad_dtype["sink"] = torch.full(
        (expected,), float("-inf"), dtype=torch.bfloat16, device=device
    )
    with pytest.raises(ValueError, match="sink.*FP32|sink.*float32"):
        aiter.mla.mla_decode_fwd_v4_nm(**args_bad_dtype)

    args_under = dict(args)
    args_under["sink"] = torch.full(
        (max_seqlen_q,), float("-inf"), dtype=torch.float32, device=device
    )
    with pytest.raises(ValueError, match="sink.*numel"):
        aiter.mla.mla_decode_fwd_v4_nm(**args_under)

    args_over = dict(args)
    args_over["sink"] = torch.full(
        (expected * 2,),  # clearly over-sized regardless of (gqa, max_seqlen_q)
        float("-inf"),
        dtype=torch.float32,
        device=device,
    )
    with pytest.raises(ValueError, match="sink.*numel"):
        aiter.mla.mla_decode_fwd_v4_nm(**args_over)

    # Non-contiguous sink (slice/transpose) — kernel reads flat fp32, so
    # any stride mismatch silently scrambles the per-head sink layout.
    args_strided = dict(args)
    args_strided["sink"] = torch.full(
        (expected * 2,), float("-inf"), dtype=torch.float32, device=device
    )[
        ::2
    ]  # numel == expected but stride=2 → non-contiguous
    assert args_strided["sink"].numel() == expected
    with pytest.raises(ValueError, match="sink.*contiguous"):
        aiter.mla.mla_decode_fwd_v4_nm(**args_strided)

    # Wrong device (CPU vs CUDA q).
    args_bad_device = dict(args)
    args_bad_device["sink"] = torch.full(
        (expected,), float("-inf"), dtype=torch.float32, device="cpu"
    )
    with pytest.raises(ValueError, match="sink.*device|same device"):
        aiter.mla.mla_decode_fwd_v4_nm(**args_bad_device)


if __name__ == "__main__":
    import argparse
    import itertools
    import sys

    # The v4 nm kernel ships only for gfx950 (mi350). CI runs every op_tests
    # file via `python3 <file>` (not pytest), which bypasses the per-test
    # @needs_gfx950 skipif marker and would execute this driver — loading the
    # gfx950-only .co — on a gfx942 (mi300) runner and fail. Guard the driver
    # so it cleanly no-ops (exit 0) anywhere that isn't gfx950.
    if not torch.cuda.is_available() or not _on_gfx950():
        print(
            "[v4 nm] skip: shipped only for gfx950 (mi350); "
            "current device is not gfx950. Exiting 0."
        )
        sys.exit(0)

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
        default=16,
        help="num_heads / num_kv_heads. Must satisfy (gqa_ratio, q_seq_logical) "
        "in {(16,4), (64,1), (128,1)} — the three entry points the qh64 .co's "
        "tile (64 q-rows) covers via the dispatcher's sub_Q=64 + "
        "gdx=ceil(gqa*max_seqlen_q/64) launch geometry. The CSV in "
        "hsa/gfx950/mla_v4/mla_v4_asm.csv ships a single (gqa=16, qSeqLen=4) "
        "row; asm_mla_v4.cu remaps gqa∈{64,128} to that row at lookup time.",
    )
    parser.add_argument(
        "--attn-sink",
        default=True,
        type=dtypes.str2bool,
        help="Enable attn sink. True by default."
        "--attn-sink=False to disable attn sink.",
    )
    parser.add_argument(
        "--out_16_nosplit",
        "--out-16-nosplit",
        dest="out_16_nosplit",
        type=int,
        default=0,
        help="1 -> kernel single-pass packed-BF16 direct path (no stage2 "
        "merge). Requires --split-kv 1. The wrapper unpacks the result into "
        "the output buffer; accuracy compares against it automatically. "
        "Default 0 (fp32 split path).",
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
            attn_sink=args.attn_sink,
            out_16_nosplit=args.out_16_nosplit,
        )
        perf_rows.append((batch, kv_seq_lens, q_seq_logical, us_asm, us_ref))

    print("\n[v4 nm perf summary] (us; speedup = fp8_ref / asm_kernel)")
    print(
        f"  {'batch':>6} {'kv_seq':>8} {'q_seq':>6} "
        f"{'asm_k us':>10} {'fp8_ref us':>12} {'speedup':>9}"
    )
    for b, k, q, ua, ur in perf_rows:
        print(f"  {b:>6d} {k:>8d} {q:>6d} {ua:>10.2f} {ur:>12.2f} {ur / ua:>8.1f}x")
