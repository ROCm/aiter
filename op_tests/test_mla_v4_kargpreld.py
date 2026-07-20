# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""aiter op_test for the gfx1250 v4 'nm' MLA decode pipeline (mla_decode_fwd_v4_nm).

Structured per `.claude/skills/aiter-op-test/SKILL.md` (mirrors
`op_tests/test_quant.py` / `op_tests/test_batched_gemm_bf16.py`):

  - `test_mla_v4_nm` is the `@benchmark()` perf+accuracy fn. Its call args
    (batch, kv_seq_lens, q_seq_logical, num_kv_splits, gqa_ratio, attn_sink)
    become the summary-table's left columns. It times each kernel *candidate*
    with `run_perftest`, checks it against the torch reference with
    `checkAllclose`, and records `us` / `TFLOPS` / `TB/s` / `err` per candidate.
  - The torch reference (`_torch_attn_decode_*`) is used for correctness ONLY;
    it is never timed and never enters the table.
  - `main()` gates on `get_gfx()` (v4 nm is shipped only for gfx1250), sweeps
    the shape lists with `itertools.product`, and prints one markdown table.

NOT covered here (deferred, see TODO): bit-exact numerical parity vs a poc_kl
dump. The v4 nm host pipeline does FP8+e8m0 dequant via
fp8e4m3_mul_fp8e8m0_bpad8_to_bf16 + a multi-step buffer concat
(poc_kl/gfx1250/mla/mla_v4.h v4_detail::init_host_buffers). Reproducing that
bit-exactly is ~200 LOC; the torch fp8-dequant reference here bounds kernel
math error to FP8 quant noise, which is sufficient for CI.

Usage:
  # perf+accuracy sweep (prints the markdown table — the deliverable):
  ENABLE_CK=0 python op_tests/test_mla_v4_kargpreld.py
"""

import argparse
import itertools
from dataclasses import dataclass

import pandas as pd
import torch

import aiter
import aiter.mla  # main no longer auto-imports submodules; need explicit
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

# v4 nm shader is shipped only for gfx1250; keep a positive allow-list so an
# unknown new card doesn't silently run an unbuilt kernel and crash.
SUPPORTED_GFX = ["gfx1250"]

# ---------------------------------------------------------------------------
# Variant under test (matches the cfg_mla_v4_asm entry in
# hsa/gfx1250/mla_v4/mla_v4_asm.csv served by csrc/py_itfs_cu/asm_mla_v4.cu).
# ---------------------------------------------------------------------------
GQA_RATIO = 16  # num_heads / num_kv_heads
PAGE_SIZE = 1
NUM_KV_HEADS = 1
DIM_NOPE = 448  # FP8 NOPE bytes per token
DIM_ROPE = 64  # BF16 ROPE elements per token (= 128 bytes; lives in qrope/kvrope)
DIM_QK_PACKED = 576  # = args.dim(512) + args.k_rotary(64); matches poc_kl stride_Page
V_HEAD_DIM = 512  # logical V head dim = args.dim = kv_lora_rank

# Perf iteration counts (kept out of the @benchmark signature so they don't
# become table columns). main() overrides these from --iters / --warmup.
_PERF = {"num_iters": 2, "num_warmup": 1}
_SEED = 0


# ---------------------------------------------------------------------------
# gfx1250 v4 nm kernel variants. In this (num_kv_heads=1) test:
#   nhead       == gqa_ratio        (num query heads)
#   decode_qlen == q_seq_logical    (logical Q rows per sequence)
# The single shipped sparse .co (mla_a8w8_qh64_1tg_16mx4_64nx1_sparse) has a
# 64 q-row tile, so it serves exactly the three "16mx4-64nx1" entries that
# satisfy nhead*decode_qlen == 64: (16,4), (64,1), (128,1). The sweep
# auto-skips any combo the dispatcher can't resolve.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Mlagfx1250KernelVariant:
    name: str
    nhead: int
    decode_qlen: int


_gfx1250_KERNEL_VARIANTS = [
    Mlagfx1250KernelVariant(name="qh16-q4-16mx4-64nx1-np", nhead=16, decode_qlen=4),
    Mlagfx1250KernelVariant(name="qh64-q1-16mx4-64nx1-np", nhead=64, decode_qlen=1),
    Mlagfx1250KernelVariant(name="qh128-q1-16mx4-64nx1-np", nhead=128, decode_qlen=1),
]
_gfx1250_VARIANT_BY_KEY = {
    (v.nhead, v.decode_qlen): v for v in _gfx1250_KERNEL_VARIANTS
}
_gfx1250_VARIANT_BY_KEY_NAME = {v.name: v for v in _gfx1250_KERNEL_VARIANTS}

# The shipped qh64 .co has a 64 q-row tile; the dispatcher
# (csrc/py_itfs_cu/asm_mla_v4.cu) picks sub_Q=64 and launches
# gdx=ceil(gqa*max_seqlen_q/64) WGs, so a single .co covers three
# (gqa, q_seq_logical) entry points. Anything else is not shipped.
_SHIPPED_TILE_VARIANTS = {(16, 4), (64, 1), (128, 1)}

# Default sweep grids (mirrors test_mla_gfx1250_triton.py).
_gfx1250_CTX_LENS = [13, 61, 128 + 3, 256 + 67, 1024, 4096, 16384]
_gfx1250_BATCH_SIZES = [3, 17, 32, 64]
_gfx1250_SPLIT_PER_BATCH = [1, 2, 4, 8]


# ---------------------------------------------------------------------------
# MODEL1_FP8Sparse packing layout (mirrored locally; not exported by
# aiter.ops.quant in this tree). Drives the per-token packing the v4 nm asm
# kernel expects.
# ---------------------------------------------------------------------------
_QUANT_D = 512  # full head dim = nope + rope
_QUANT_D_NOPE = 448  # FP8-quantized
_QUANT_D_ROPE = 64  # BF16 (kept separate in `qrope`/`kvrope` buffer)
_QUANT_TILE_SIZE = 64
_QUANT_NUM_TILES = _QUANT_D_NOPE // _QUANT_TILE_SIZE  # 7
# v4 nm kernel reads each tile's e8m0 scale TWICE in a row, so the scale
# block on disk is 14 bytes laid out as (s0,s0,s1,s1,...,s6,s6). Without the
# duplication V[256:448] of the asm output is all-zero.
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

    The v4 nm shader reads each tile's scale TWICE consecutively (s0,s0,s1,s1,
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


# ---------------------------------------------------------------------------
# Torch reference (correctness only — NEVER timed, NEVER in the table).
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Input builders.
# ---------------------------------------------------------------------------
def _build_bf16_inputs(
    batch=2,
    kv_seq_lens=64,
    q_seq_logical=4,
    seed=0,
    device="cuda",
    gqa_ratio=GQA_RATIO,
    attn_sink=True,
):
    """Build BF16 ground-truth q/kv and the aiter index tables (dequantable,
    used by the accuracy/perf path):
      q_bf16:  [total_q = batch*q_seq_logical, num_heads=gqa_ratio, D=512]
      kv_bf16: [num_page = batch*kv_seq_lens, 1, 1, D=512]

    `attn_sink`:
      True  -> NON-ZERO random (randn*10) per-head sink so a head-dim layout
               mismatch shows up as an err blowup, not a silent pass.
      False -> per-head -inf ("no sink" no-op: exp(-inf - max) = 0).
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    total_q = batch * q_seq_logical
    num_page = batch * (kv_seq_lens // PAGE_SIZE)

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

    num_heads = NUM_KV_HEADS * gqa_ratio
    if attn_sink:
        # randn*10 so the sink contributes materially (~15%) to the softmax;
        # well above tolerance, so a dropped/mis-scaled sink is a hard mismatch.
        sink = torch.randn(num_heads, dtype=torch.float32, device=device) * 10.0
    else:
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


# ---------------------------------------------------------------------------
# @benchmark perf + accuracy fn — the summary-table producer.
# Its call args become the table's left-hand columns (SKILL rule 2).
# ---------------------------------------------------------------------------
@benchmark()
def test_mla_v4_nm(
    batch=2,
    kv_seq_lens=64,
    q_seq_logical=4,
    num_kv_splits=1,
    gqa_ratio=GQA_RATIO,
    attn_sink=True,
):
    """Time each v4 nm kernel candidate, check it against the torch fp8-dequant
    reference, and return per-candidate `us` / `TFLOPS` / `TB/s` / `err`.

    Candidates (SKILL "candidates live in a dict; build ret in a loop"):
      - `v4_nm`     : fp32 split path (reads logits[:, 0] or stage2-merged out).
      - `v4_nm_o16` : single-pass packed-BF16 direct path (out_16_nosplit=1).
                      Only valid when num_kv_splits==1 (bf16-direct is
                      single-pass), so it is conditionally added and left NaN
                      elsewhere (SKILL "skip a candidate in configs it can't do").

    Accuracy gate = fp8_dequant_ref vs asm (kernel math error, quant-independent).
    A second unrecorded check (golden_bf16 vs fp8_ref) reports the FP8 quant
    noise floor. Perf (us/TFLOPS/TB-s) is informational (CI variance too high to
    assert on).
    """
    assert (gqa_ratio, q_seq_logical) in _SHIPPED_TILE_VARIANTS, (
        f"(gqa_ratio={gqa_ratio}, q_seq_logical={q_seq_logical}) not in shipped "
        f"variants {_SHIPPED_TILE_VARIANTS}: the qh64 .co picks sub_Q=64 and "
        f"launches gdx=ceil(gqa*max_seqlen_q/64); only these three pairs resolve."
    )
    # Multi-split guard: the .co inner KV loop processes pass_size=16 tokens per
    # iteration; the SMALLEST split must be >= 16 or its tail is dropped.
    if num_kv_splits > 1:
        min_split = kv_seq_lens // num_kv_splits  # page_size=1
        assert min_split >= 16, (
            f"smallest KV split = floor({kv_seq_lens}/{num_kv_splits}) = "
            f"{min_split} < pass_size=16: that split drops its tail. Reduce "
            f"num_kv_splits or raise kv_seq_lens."
        )

    device = "cuda"
    inputs = _build_bf16_inputs(
        batch=batch,
        kv_seq_lens=kv_seq_lens,
        q_seq_logical=q_seq_logical,
        seed=_SEED,
        gqa_ratio=gqa_ratio,
        attn_sink=attn_sink,
    )
    sm_scale = 1.0 / (_QUANT_D**0.5)  # kernel ignores; only used by torch ref

    # Pre-quantize once (Python quant helper is slow; must not be timed). The
    # same FP8 bytes feed both the asm kernel and the fp8-dequant ref, so any
    # diff between them isolates kernel math.
    q_packed, q_rope = _native_to_2buff_for_asm(inputs["q_bf16"])
    kv_packed, kv_rope = _native_to_2buff_for_asm(inputs["kv_bf16"])

    # ---- torch reference (NOT timed, NOT in table) ----
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
    out_fp8_ref, _ = _torch_attn_decode_fp8_dequant_ref(
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
    )
    # Unrecorded: FP8 quant noise floor (kernel-independent).
    checkAllclose(
        out_golden.to(dtypes.fp32),
        out_fp8_ref.to(dtypes.fp32),
        rtol=3e-2,
        atol=3e-2,
        tol_err_ratio=0.02,
        msg="mla_v4_nm [golden_bf16 vs fp8_ref]",
    )

    # Pre-allocate everything the kernel writes into so the timed iters don't
    # allocate (layout matches aiter/mla.py). Reused across all iters via
    # num_rotate_args=1.
    total_q = inputs["q_bf16"].size(0)
    num_seqs = inputs["qo_indptr"].size(0) - 1
    num_heads = NUM_KV_HEADS * gqa_ratio
    output_buf = torch.empty(
        (total_q, gqa_ratio, V_HEAD_DIM), dtype=dtypes.bf16, device=device
    )
    split_indptr = torch.tensor(
        [i * num_kv_splits for i in range(num_seqs + 1)],
        dtype=torch.int32,
        device=device,
    )
    logits_buf = torch.empty(
        (total_q, num_kv_splits, num_heads, V_HEAD_DIM),
        dtype=dtypes.fp32,
        device=device,
    )
    lse_buf = torch.empty(
        (total_q, num_kv_splits, num_heads, 1), dtype=dtypes.fp32, device=device
    )

    common_kwargs = dict(
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
        num_kv_splits=num_kv_splits,
        logits=logits_buf,
        attn_lse=lse_buf,
    )

    # out_16_nosplit=1 is single-pass only (no stage2 merge) -> num_kv_splits==1.
    candidates = {"v4_nm": 0}
    if num_kv_splits == 1:
        candidates["v4_nm_o16"] = 1

    # Roofline: full attention does q_seq_logical rows attending kv_seq_lens
    # keys per seq (x batch), each a (QK^T + PV) dot over (_QUANT_D + V_HEAD_DIM).
    total_kv = batch * kv_seq_lens
    flops = 2 * q_seq_logical * total_kv * gqa_ratio * (_QUANT_D + V_HEAD_DIM)
    # Bytes: FP8 q/kv (1B) + BF16 rope (2B) read + BF16 output written.
    per_tok = _QUANT_D * 1 + DIM_ROPE * 2
    nbytes = (
        total_q * gqa_ratio * per_tok
        + total_kv * per_tok
        + total_q * gqa_ratio * V_HEAD_DIM * 2
    )

    ret = {"gfx": get_gfx()}
    for name, o16 in candidates.items():
        (logits, _lse), us = run_perftest(
            aiter.mla.mla_decode_fwd_v4_nm,
            out_16_nosplit=int(o16),
            **common_kwargs,
            num_iters=_PERF["num_iters"],
            num_warmup=_PERF["num_warmup"],
        )
        # Resolve the buffer the wrapper actually populated:
        #   o16=1        -> wrapper unpacked packed-BF16 into output_buf.
        #   single-pass  -> kernel wrote one FP32 partial to logits[:, 0].
        #   multi-pass   -> stage2 merge wrote merged BF16 to output_buf.
        if o16:
            out_asm = output_buf
        elif num_kv_splits == 1:
            out_asm = logits[:, 0].to(dtypes.bf16)
        else:
            out_asm = output_buf

        err = checkAllclose(
            out_fp8_ref.to(dtypes.fp32),
            out_asm.to(dtypes.fp32),
            rtol=3e-2,
            atol=3e-2,
            tol_err_ratio=0.02,
            msg=f"{name}: mla_v4_nm [fp8_dequant_ref vs asm]",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err

    return ret


# ---------------------------------------------------------------------------
# ATOM-API wrapper (future drop-in for ATOM's `sparse_attn_v4_paged_decode`).
# Lives here as a *proof of API fit*; the production wrapper belongs in
# aiter/mla.py once exercised.
# ---------------------------------------------------------------------------
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
    `aiter.mla.mla_decode_fwd_v4_nm`, and reduce/reshape the FP32 split logits
    back into a [total_q, num_heads, V_HEAD_DIM] BF16 tensor.

    Returns (out_bf16, logits, attn_lse, packed_buffers).
    """
    total_q = q_bf16.size(0)
    num_heads = q_bf16.size(1)
    num_seqs = qo_indptr.size(0) - 1
    assert num_heads == GQA_RATIO

    q_packed, q_rope = _native_to_2buff_for_asm(q_bf16)
    kv_packed, kv_rope = _native_to_2buff_for_asm(kv_bf16)

    output = torch.empty(
        (total_q, num_heads, V_HEAD_DIM), dtype=dtypes.bf16, device=q_bf16.device
    )
    num_kv_splits = 1
    split_indptr = torch.tensor(
        [i * num_kv_splits for i in range(num_seqs + 1)],
        dtype=torch.int32,
        device=q_bf16.device,
    )
    sink = torch.full(
        (num_heads,), float("-inf"), dtype=torch.float32, device=q_bf16.device
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
    # logits row layout: row = q_token * gqa_ratio + head. Reshape to
    # [num_seqs, q_seq_logical, gqa, D] then flatten to [total_q, H, D].
    out_bf16 = (
        logits[:, 0, 0]
        .reshape(num_seqs, max_seqlen_q, num_heads, V_HEAD_DIM)
        .reshape(total_q, num_heads, V_HEAD_DIM)
        .to(dtypes.bf16)
    )
    return out_bf16, logits, attn_lse, (q_packed, q_rope, kv_packed, kv_rope)


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
      - Tokens are processed in groups of 4 as one "sequence"; tokens
        [b*4 .. (b+1)*4) MUST share the same kv span. Caller's responsibility.
      - attn_sink is currently unused; reserved for API parity. Pass `None`.

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

    qo_indptr = torch.arange(0, batch + 1, dtype=torch.int32, device=device) * 4
    kv_indptr_per_seq = kv_indptr[::4].to(torch.int32).contiguous()
    assert (
        kv_indptr_per_seq.size(0) == batch + 1
    ), f"kv_indptr layout invalid for groups-of-4: got len {kv_indptr.size(0)}, expected {batch * 4 + 1}"
    for b in range(batch):
        base = int(kv_indptr[b * 4].item())
        for j in range(1, 4):
            assert (
                int(kv_indptr[b * 4 + j].item()) == base
            ), f"asm v4 nm wrapper requires kv_indptr constant per group-of-4 (batch {b}, offset {j})"

    kv_page_indices = kv_indices.to(torch.int32).contiguous()
    kv_last_page_lens = torch.ones(batch, dtype=torch.int32, device=device)
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
# __main__ sweep: arch gate -> itertools.product over the shape lists ->
# one markdown summary table (SKILL rules 6-9).
# ---------------------------------------------------------------------------
def main():
    # Whole-op arch gate goes HERE (not inside the @benchmark fn, which always
    # returns the call-args dict). Positive allow-list.
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("mla_v4_nm unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="v4 nm MLA decode: per-shape accuracy + perf, one markdown table.",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=_gfx1250_BATCH_SIZES,
        help="Batch size(s). e.g. -b 1 16 32",
    )
    parser.add_argument(
        "-c",
        "--kv-seq-lens",
        type=int,
        nargs="*",
        default=_gfx1250_CTX_LENS,
        help="KV tokens per sequence (context length). e.g. -c 64 256 1024",
    )
    parser.add_argument(
        "--variant",
        nargs="*",
        choices=[v.name for v in _gfx1250_KERNEL_VARIANTS],
        default=[v.name for v in _gfx1250_KERNEL_VARIANTS],
        help="Kernel variant name(s); nhead/decode_qlen taken from the table.",
    )
    parser.add_argument(
        "--split-kv",
        type=int,
        nargs="*",
        default=_gfx1250_SPLIT_PER_BATCH,
        help="num_kv_splits value(s) to sweep. e.g. --split-kv 1 2 4 8",
    )
    parser.add_argument(
        "--attn-sink",
        type=dtypes.str2bool,
        nargs="*",
        default=[True],
        help="attn sink value(s) to sweep. e.g. --attn-sink True False",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iters", type=int, default=50, help="Perf timed iterations")
    parser.add_argument("--warmup", type=int, default=2, help="Perf warmup iterations")
    args = parser.parse_args()

    global _SEED
    _SEED = args.seed
    _PERF["num_iters"] = args.iters
    _PERF["num_warmup"] = args.warmup

    nhead_combos = [
        (
            _gfx1250_VARIANT_BY_KEY_NAME[name].nhead,
            _gfx1250_VARIANT_BY_KEY_NAME[name].decode_qlen,
        )
        for name in args.variant
    ]

    df = []
    for (nhead, decode_qlen), batch, kv_seq_lens, split_kv, sink in itertools.product(
        nhead_combos, args.batch, args.kv_seq_lens, args.split_kv, args.attn_sink
    ):
        try:
            df.append(
                test_mla_v4_nm(
                    batch=batch,
                    kv_seq_lens=kv_seq_lens,
                    q_seq_logical=decode_qlen,
                    num_kv_splits=split_kv,
                    gqa_ratio=nhead,
                    attn_sink=sink,
                )
            )
        except (RuntimeError, AssertionError) as exc:
            # not-yet-shipped (nhead, decode_qlen) or an invalid split combo:
            # skip + sweep on (SKILL "skip a config it does not support").
            msg = str(exc).splitlines()[0] if str(exc) else type(exc).__name__
            aiter.logger.warning(
                "[skip] nhead=%d q=%d b=%d c=%d split=%d: %s",
                nhead,
                decode_qlen,
                batch,
                kv_seq_lens,
                split_kv,
                msg,
            )

    df = pd.DataFrame(df)
    aiter.logger.info("mla_v4_nm summary (markdown):\n%s", df.to_markdown(index=False))


if __name__ == "__main__":
    main()
