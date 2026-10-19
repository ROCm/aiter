# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

# gfx1250 / mi400 MLA fp8 decode test.
#
#   # Single public dispatch case:
#   python3 op_tests/test_mla_decode_pagesize64.py -n 8,1
#
#   # Sweep all supported public dispatch cases:
#   python3 op_tests/test_mla_decode_pagesize64.py
#
#   # QH128 automatically uses kv-shuffled PS64 (one query, no KV splitting):
#   python3 op_tests/test_mla_decode_pagesize64.py -n 128,1
#
#   # GPU contract tests, including graph replay and all page tails:
#   python3 -m pytest op_tests/test_mla_decode_pagesize64.py -q
#
#   # Peak-performance sweep from the gfx1250 MLA report:
#   python3 op_tests/test_mla_decode_pagesize64.py -n 8,1 8，2 16，1 32，1 -b 1024 -c 16384 --split_kv auto


import argparse
import itertools
import math
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

import aiter
import aiter.mla
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.mla import mla_decode_fwd_qh128_asm
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_printoptions(sci_mode=False)

SUPPORTED_GFX = ["gfx1250"]


def check_support(dtype, kv_dtype, nhead):
    return dtype == dtypes.fp8 and kv_dtype == dtypes.fp8


# Dispatch cases covered by this UT. QH128 uses the dedicated kv-shuffled
# PS64 kernel; other cases use the registered gfx1250 MLA dispatcher.
_MI400_DISPATCH_CASES = [
    (8, 1),
    (8, 2),
    (8, 3),
    (8, 4),
    (16, 1),
    (16, 2),
    (16, 4),
    (32, 1),
    (64, 1),
    (128, 1),
]


def _pack_rope_split3_q_pages(tensor, nope_dim, rope_dim, padded_stride_bytes=768):
    shape = tensor.shape
    assert shape[-1] == nope_dim + rope_dim
    elem_size = tensor.element_size()
    if padded_stride_bytes % elem_size != 0:
        raise ValueError("rope_split3 padded stride must be element aligned")
    padded_dim = padded_stride_bytes // elem_size
    if padded_dim < shape[-1]:
        raise ValueError(
            f"rope_split3 padded dim {padded_dim} is smaller than Q dim {shape[-1]}"
        )

    # Mirror poc_kl pack_q_page1_padded(): each logical Q row stores
    # [nope][rope] followed by zero padding up to a 768-byte row stride.
    rows = tensor.reshape(-1, shape[-1])
    padded = torch.zeros(
        (rows.shape[0], padded_dim),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    padded[:, : shape[-1]].copy_(rows)
    return torch.as_strided(
        padded,
        size=shape,
        stride=(
            shape[1] * shape[2] * padded_dim,
            shape[2] * padded_dim,
            padded_dim,
            1,
        ),
    )


def _pack_rope_split2_kv_pages(tensor, nope_dim, rope_dim):
    pages, page_size, nhead_kv, head_dim = tensor.shape
    assert nhead_kv == 1
    assert head_dim == nope_dim + rope_dim
    packed = torch.cat(
        (
            tensor[..., :nope_dim].reshape(pages, page_size * nope_dim),
            tensor[..., nope_dim:].reshape(pages, page_size * rope_dim),
        ),
        dim=-1,
    )
    return packed.reshape(pages, page_size, nhead_kv, head_dim).contiguous()


def _pack_kv_shuffled_pages(tensor, nope_dim=512, rope_dim=64):
    """Pack FP8 page64 KV for the shuffled QH64 CSR and QH128 dense kernels."""
    pages = tensor.shape[0]
    kv = tensor.reshape(pages, 64, nope_dim + rope_dim)
    return torch.cat(
        [
            part.reshape(pages, 4, 16, dim // 16, 16)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
            .reshape(pages, -1)
            for part, dim in [
                (kv[..., :nope_dim], nope_dim),
                (kv[..., nope_dim:], rope_dim),
            ]
        ],
        dim=1,
    ).view(pages, 1, 64, nope_dim + rope_dim)


def _make_page_permutation(num_pages, *, shuffle):
    if not shuffle:
        return list(range(num_pages))
    if num_pages <= 1:
        return list(range(num_pages))
    for step in (7, 5, 3):
        if num_pages % step != 0:
            return [(i * step + 1) % num_pages for i in range(num_pages)]
    return list(reversed(range(num_pages)))


def _make_scales(batch, device, *, enabled):
    if not enabled:
        return (
            torch.ones((1,), dtype=torch.float32, device=device),
            torch.ones((1,), dtype=torch.float32, device=device),
        )
    q_scale = torch.linspace(0.75, 1.25, 1, dtype=torch.float32, device=device)
    kv_scale = torch.linspace(1.20, 0.80, 1, dtype=torch.float32, device=device)
    return q_scale, kv_scale


def _make_mla_mi400_case(
    *,
    batch,
    ctx_lens,
    nhead,
    decode_qlen,
    num_kv_splits,
    page_indices_oob=0,
    use_non_unit_scales=True,
):
    repo_hsa_dir = Path(__file__).resolve().parents[1] / "hsa"
    os.environ["AITER_ASM_DIR"] = str(repo_hsa_dir)

    device = torch.device("cuda")
    page_size = 64
    num_pages_per_batch = (ctx_lens + page_size - 1) // page_size

    if num_kv_splits is None and num_pages_per_batch == 0:
        num_kv_splits = 1
    if num_kv_splits is None:
        # Mirror mla_decode_fwd(num_kv_splits=None): resolve the auto split count
        # and its indptr through the shared meta-param heuristic so the case
        # carries a concrete value for the shape checks and the kernel args.
        num_kv_splits, num_kv_splits_indptr = aiter.mla.get_meta_param(
            None,
            batch,
            batch * num_pages_per_batch,
            nhead,
            decode_qlen,
            dtypes.fp8,
        )
        num_kv_splits = int(num_kv_splits)
    else:
        assert num_kv_splits > 0
        num_kv_splits_indptr = (
            torch.arange(batch + 1, dtype=torch.int32, device=device) * num_kv_splits
        )
    torch.manual_seed(
        20260513
        + batch * 1009
        + ctx_lens
        + nhead * 7
        + decode_qlen
        + num_kv_splits * 101
    )

    last_page_len = (ctx_lens % page_size or page_size) if ctx_lens else 0
    kv_last_page_lens = torch.full(
        (batch,), last_page_len, dtype=torch.int32, device=device
    )
    # gfx1250/mi400 stage1 asm kernel consumes a PAGE-level kv_indptr directly
    # (it walks the page-level kv_indices block table). Build it here as the
    # per-batch prefix sum of page counts so mla.py no longer needs to convert a
    # token-level kv_indptr. With uniform ctx_lens this is [0, npb, 2*npb, ...].
    kv_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(
        torch.full((batch,), num_pages_per_batch, dtype=torch.int32, device=device),
        dim=0,
    )
    q_scale, kv_scale = _make_scales(batch, device, enabled=use_non_unit_scales)

    return {
        "page_size": page_size,
        "num_kv_splits": num_kv_splits,
        "num_pages_per_batch": num_pages_per_batch,
        "kv_last_page_lens": kv_last_page_lens,
        "kv_indptr": kv_indptr,
        "num_kv_splits_indptr": num_kv_splits_indptr,
        "q_scale": q_scale,
        "kv_scale": kv_scale,
    }


def _make_mla_mi400_kv_case(
    *,
    kv_buffer_bf16,
    batch,
    ctx_lens,
    qk_head_dim,
    v_head_dim,
    page_indices_oob,
    fallback_fill_value=None,
    shuffle_pages=True,
    kv_shuffled=False,
):
    """Build segmented/shuffled page64 KV for gfx1250.

    Returns (kv_buffer, kv_buffer_ref, kv_indices):
      kv_buffer     : fp8 (float8_e4m3fn), aiter PAGE-level seg-pack, shape
                      [num_pages, page_size, 1, 576] holding
                      [page_size*512 (nope) | page_size*64 (pe)] per page
                      (page_size=64). This is what mla.mla_decode_fwd consumes.
                      Built by _pack_rope_split2_kv_pages. With kv_shuffled,
                      use [num_pages, 1, page_size, 576] with each plane tiled
                      into 16-token x 16-dimension blocks instead.
      kv_buffer_ref : fp8 (float8_e4m3fn), TOKEN-major scattered cache
                      [num_pages, page_size, 1, 576] (pages placed at their
                      physical ids); consumed only by the PyTorch fp32 reference.
      kv_indices    : int32 PAGE-level block table [batch*(npb+oob)] of physical
                      page ids (compact, OOB padding appended after valid pages).
    """
    device = torch.device("cuda")
    page_size = 64
    nhead_kv = 1
    num_pages_per_batch = (ctx_lens + page_size - 1) // page_size
    total_page_indices = batch * (num_pages_per_batch + page_indices_oob)
    total_pages = batch * num_pages_per_batch
    if total_pages == 0:
        ref = torch.full((1, 64, 1, 576), float("nan"), dtype=dtypes.fp8, device=device)
        pack_kv = _pack_kv_shuffled_pages if kv_shuffled else _pack_rope_split2_kv_pages
        return (
            pack_kv(ref, v_head_dim, qk_head_dim - v_head_dim),
            ref,
            torch.full((total_page_indices,), -1, dtype=torch.int32, device=device),
        )

    kv_buffer_source_bf16 = kv_buffer_bf16.view(-1, page_size, nhead_kv, qk_head_dim)
    available_pages = kv_buffer_source_bf16.size(0)
    if available_pages >= total_pages:
        kv_buffer_logical_bf16 = kv_buffer_source_bf16[:total_pages].contiguous()
    else:
        kv_buffer_logical_bf16 = torch.empty(
            (total_pages, page_size, nhead_kv, qk_head_dim),
            dtype=kv_buffer_source_bf16.dtype,
            device=kv_buffer_source_bf16.device,
        )
        kv_buffer_logical_bf16[:available_pages] = kv_buffer_source_bf16
        fallback_shape = (
            total_pages - available_pages,
            page_size,
            nhead_kv,
            qk_head_dim,
        )
        if fallback_fill_value is None:
            kv_buffer_logical_bf16[available_pages:] = torch.randn(
                fallback_shape,
                dtype=kv_buffer_source_bf16.dtype,
                device=kv_buffer_source_bf16.device,
            )
        else:
            kv_buffer_logical_bf16[available_pages:] = torch.full(
                fallback_shape,
                fallback_fill_value,
                dtype=kv_buffer_source_bf16.dtype,
                device=kv_buffer_source_bf16.device,
            )
    # Poison the unused tail of every batch's last (partially filled) page with
    # NaN. When ctx_lens % page_size != 0 the final logical page of each batch
    # keeps only last_page_len valid tokens; slots [last_page_len:page_size] are
    # never valid KV. The kernel must honor kv_last_page_lens / kv_indptr and
    # never read past them, so a correct kernel still yields a finite, matching
    # output. The PyTorch reference excludes this tail via kv[:ctx_lens].
    last_page_len = ctx_lens % page_size or page_size
    if last_page_len != page_size:
        last_logical_pages = [(b + 1) * num_pages_per_batch - 1 for b in range(batch)]
        kv_buffer_logical_bf16[last_logical_pages, last_page_len:] = float("nan")

    # The kernel consumes a compact block table, with OOB padding only after all
    # valid pages. KV pages are scattered into their physical page ids.
    shuffled_page_indices = _make_page_permutation(total_pages, shuffle=shuffle_pages)
    kv_buffer_scattered_bf16 = torch.empty_like(kv_buffer_logical_bf16)
    kv_indices = torch.zeros(total_page_indices, dtype=torch.int32, device=device)
    physical_ids = torch.tensor(shuffled_page_indices, dtype=torch.int64, device=device)
    kv_buffer_scattered_bf16[physical_ids] = kv_buffer_logical_bf16
    kv_indices[:total_pages] = physical_ids.to(torch.int32)

    kv_buffer_ref = kv_buffer_scattered_bf16.to(dtypes.fp8)
    pack_kv = _pack_kv_shuffled_pages if kv_shuffled else _pack_rope_split2_kv_pages
    kv_buffer = pack_kv(
        kv_buffer_ref.view(total_pages, page_size, nhead_kv, qk_head_dim),
        v_head_dim,
        qk_head_dim - v_head_dim,
    )
    return kv_buffer, kv_buffer_ref, kv_indices


def _make_mla_mi400_q_case(
    *, q_fp8, batch, decode_qlen, nhead, qk_head_dim, v_head_dim
):
    """Build the Q input for the gfx1250 seg asm decode.

    Returns q: fp8 (float8_e4m3fn), shape [total_q, nhead, 576], NON-contiguous
    768-padded selected layout -- per-head row stride = 768 elems (=768 B in
    fp8), i.e. each head's 576 values ([nope 512][rope 64]) followed by 192 B of
    zero padding (_MLA_Q_OUT_PADDED_DIM). Built by _pack_rope_split3_q_pages +
    as_strided. (The PyTorch fp32 reference instead reads the unpadded q_fp8
    directly.)
    """
    q = q_fp8.view(batch, decode_qlen, nhead, qk_head_dim)
    q = _pack_rope_split3_q_pages(
        q,
        v_head_dim,
        qk_head_dim - v_head_dim,
    )
    return torch.as_strided(
        q,
        size=(batch * decode_qlen, nhead, qk_head_dim),
        stride=(nhead * q.stride(2), q.stride(2), q.stride(3)),
    )


def _apply_causal_mask_(logits):
    # Matches the causal/tail mask shape used by the reference attention.
    _, s_q, s_k = logits.shape
    mask = torch.ones(s_q, s_k, dtype=torch.bool, device=logits.device).tril(
        diagonal=s_k - s_q
    )
    logits.masked_fill_(mask.logical_not().unsqueeze(0), float("-inf"))


def _ref_mla_mi400(
    case,
    q_ref,
    kv_buffer_ref,
    kv_indices,
    batch_size,
    ctx_lens,
    decode_qlen,
    nhead_kv,
    qk_head_dim,
    v_head_dim,
    mask,
):
    """PyTorch fp32 analytic reference (qk_head_dim=576 = nope 512 + rope 64).

    Inputs it reads (both UNPACKED relative to the aiter kernel layouts; both
    fp8 then upcast to fp32 here so the r eference carries no extra quant error):
      q_ref         : fp8 (float8_e4m3fn), CONTIGUOUS [total_q, nhead, 576]
                      (the plain q_fp8, NOT the 768-padded selected layout the
                      asm kernel consumes). Upcast via .float() * q_scale.
      kv_buffer_ref : fp8 (float8_e4m3fn), TOKEN-major scattered cache
                      [num_pages, page_size, 1, 576] (pages at physical ids, NOT
                      the seg-packed layout). Gathered per batch by physical
                      page id (kv_indices), upcast via .float() * kv_scale, then
                      reshaped to [ctx_lens, 1, 576]; key=full 576, value=[:512].
    Output: bf16 [total_q, nhead, 512] (softmax(QK^T/sqrt(576))·V, causal mask).
    """
    outputs = []
    num_pages = case["num_pages_per_batch"]
    kv_source = kv_buffer_ref
    for b in range(batch_size):
        q_start = b * decode_qlen
        q_end = q_start + decode_qlen
        q_scale = case["q_scale"][0 if case["q_scale"].numel() == 1 else b]
        kv_scale = case["kv_scale"][0 if case["kv_scale"].numel() == 1 else b]
        q = q_ref[q_start:q_end].float() * q_scale
        page_indices = kv_indices[b * num_pages : (b + 1) * num_pages].long()
        kv = torch.index_select(kv_source.float(), 0, page_indices) * kv_scale
        kv = kv.reshape(-1, nhead_kv, qk_head_dim)
        kv = kv[:ctx_lens]
        key = kv
        value = kv[..., :v_head_dim]

        logits = torch.einsum("qhd,kmd->hqk", q, key) * (1.0 / (qk_head_dim**0.5))
        if mask:
            _apply_causal_mask_(logits)
        weights = torch.softmax(logits, dim=-1).nan_to_num(0.0)
        outputs.append(torch.einsum("hqk,kmd->qhd", weights, value).to(torch.bfloat16))
    return torch.cat(outputs, dim=0)


def _cosine_diff(actual, expected):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    numerator = 2 * (actual.double() * expected.double()).sum()
    denominator = (
        (actual.double().square() + expected.double().square()).sum().clamp_min(1e-12)
    )
    return (1 - (numerator / denominator)).item()


@benchmark()
def test_mla(
    batch,
    ctx_len,
    nhead,
    decode_qlen,
    split_kv,
    mask,
    dtype,
    kv_dtype,
    init,
    kv_layout="auto",
):
    page_size = 64
    if kv_layout == "auto":
        kv_layout = (
            "kv_shuffled"
            if nhead == 128 or (nhead == 64 and decode_qlen > 1)
            else "token_major"
        )
    kv_shuffled = kv_layout == "kv_shuffled"
    dense_shuffled = kv_shuffled and nhead == 128
    if kv_shuffled and (nhead not in (64, 128) or (nhead == 128 and decode_qlen != 1)):
        raise ValueError(
            "Shuffled page64 supports QH64 multiple queries or QH128 decode_qlen=1"
        )
    if not kv_shuffled and nhead == 64 and decode_qlen != 1:
        raise ValueError("QH64 token_major supports decode_qlen=1; use kv_shuffled")
    if dense_shuffled and split_kv != 1:
        raise ValueError(
            "Shuffled PS64 kernel requires nhead=128, decode_qlen=1, split_kv=1"
        )
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    nhead_kv = 1
    v_head_dim = kv_lora_rank
    page_indices_oob = 4

    case = None
    if nhead == 64:
        resolved_splits = split_kv
        if kv_shuffled and split_kv is None:
            index_count = batch * ((ctx_len + 63) // 64 + page_indices_oob)
            resolved_splits, _ = aiter.mla.get_meta_param(
                None,
                batch,
                index_count * 64,
                64,
                1,
                dtypes.fp8,
                tg_factor=decode_qlen,
            )
            resolved_splits = int(resolved_splits)
        case = _make_mla_mi400_case(
            batch=batch,
            ctx_lens=ctx_len,
            nhead=nhead,
            decode_qlen=decode_qlen,
            num_kv_splits=resolved_splits,
            page_indices_oob=page_indices_oob,
        )

    kv_max_sz = 65536 * 32  # Remaining framework KV capacity after weights.
    num_page = (
        batch * ((ctx_len + page_size - 1) // page_size)
        if dense_shuffled
        else (kv_max_sz + page_size - 1) // page_size
    )
    input_fill_value = 0.25 if init == "const0.25" else None
    if input_fill_value is None:
        kv_buffer = torch.randn(
            (num_page * page_size, 1, qk_head_dim),
            dtype=torch.bfloat16,
        )
    else:
        kv_buffer = torch.full(
            (num_page * page_size, 1, qk_head_dim),
            input_fill_value,
            dtype=torch.bfloat16,
        )

    qo_indptr = torch.zeros(batch + 1, dtype=torch.int)
    seq_lens_qo = torch.full((batch,), decode_qlen, dtype=torch.int)
    qo_indptr[1 : batch + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    if input_fill_value is None:
        q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16)
    else:
        q = torch.full(
            (total_q, nhead, qk_head_dim),
            input_fill_value,
            dtype=torch.bfloat16,
        )

    kv_buffer_mi400, kv_buffer_ref_mi400, kv_indices_mi400 = _make_mla_mi400_kv_case(
        kv_buffer_bf16=kv_buffer,
        batch=batch,
        ctx_lens=ctx_len,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        page_indices_oob=page_indices_oob,
        fallback_fill_value=input_fill_value,
        kv_shuffled=kv_shuffled,
    )
    q_fp8_mi400 = q.to(dtypes.fp8)
    q_mi400 = _make_mla_mi400_q_case(
        q_fp8=q_fp8_mi400,
        batch=batch,
        decode_qlen=decode_qlen,
        nhead=nhead,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
    )
    if case is None:
        case = _make_mla_mi400_case(
            batch=batch,
            ctx_lens=ctx_len,
            nhead=nhead,
            decode_qlen=decode_qlen,
            num_kv_splits=split_kv,
            page_indices_oob=page_indices_oob,
        )

    # Prepare the dense table and lengths outside the measured/captured launch.
    seq_lens = page_table = None
    if dense_shuffled:
        page_table = kv_indices_mi400[: batch * case["num_pages_per_batch"]].view(
            batch, case["num_pages_per_batch"]
        )
        seq_lens = torch.full((batch,), ctx_len, dtype=torch.int32)

    def run_mla_decode(out_tensor):
        return aiter.mla.mla_decode_fwd(
            q_mi400,
            kv_buffer_mi400,
            out_tensor,
            None if dense_shuffled else qo_indptr,
            None if dense_shuffled else case["kv_indptr"],
            None if dense_shuffled else kv_indices_mi400,
            None if dense_shuffled else case["kv_last_page_lens"],
            decode_qlen,
            case["page_size"],
            nhead_kv,
            1.0 / (qk_head_dim**0.5),
            num_kv_splits=case["num_kv_splits"],
            num_kv_splits_indptr=(
                None if dense_shuffled else case["num_kv_splits_indptr"]
            ),
            q_scale=case["q_scale"],
            kv_scale=case["kv_scale"],
            return_lse=True,
            kv_layout=kv_layout,
            seq_lens=seq_lens,
            page_table=page_table,
            causal=bool(mask),
        )

    out = torch.zeros((batch * decode_qlen, nhead, v_head_dim), dtype=torch.bfloat16)

    total_kv = batch * ctx_len
    flops = decode_qlen * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
    nbytes = (
        total_kv * nhead_kv * qk_head_dim * (torch.finfo(dtypes.fp8).bits // 8)
        + total_q * nhead * qk_head_dim * (torch.finfo(dtypes.fp8).bits // 8)
        + total_q * nhead * v_head_dim * (torch.finfo(torch.bfloat16).bits // 8)
    )

    attn, us = run_perftest(run_mla_decode, out)
    attn_logits, attn_lse = attn
    out_check = out.clone()

    logits_shape = (batch * decode_qlen, case["num_kv_splits"], nhead, v_head_dim)
    if case["num_kv_splits"] == 1:
        logits_shape = (batch * decode_qlen, nhead, v_head_dim)
    assert out_check.shape == (batch * decode_qlen, nhead, v_head_dim)
    assert attn_logits.shape == logits_shape
    assert attn_lse.shape == (batch * decode_qlen, nhead)

    final_out_finite = torch.isfinite(out_check.detach().float().cpu()).all().item()
    if final_out_finite:
        ref = _ref_mla_mi400(
            case,
            q_fp8_mi400,
            kv_buffer_ref_mi400,
            kv_indices_mi400,
            batch,
            ctx_len,
            decode_qlen,
            nhead_kv,
            qk_head_dim,
            v_head_dim,
            mask,
        )
        err = checkAllclose(
            ref.to(dtypes.fp32),
            out_check.to(dtypes.fp32),
            rtol=6e-2,
            atol=6e-2,
            tol_err_ratio=0.05,
            msg="mi400: mla_decode_mi400",
        )
        cos_diff = _cosine_diff(out_check, ref)
    else:
        err = float("inf")
        cos_diff = float("inf")

    assert final_out_finite, "MLA produced NaN or Inf"
    assert err <= 0.05, f"MLA mismatch ratio {err:.3%} exceeds 5%"

    ret = {
        "gfx": get_gfx(),
        "kv_layout": kv_layout,
        "kv_shuffled": kv_shuffled,
        "num_kv_splits": case["num_kv_splits"],
        "init": init,
        "mi400 us": us,
        "mi400 TFLOPS": flops / us / 1e6,
        "mi400 TB/s": nbytes / us / 1e6,
        "mi400 err": err,
        "mi400 cos_diff": cos_diff,
        "mi400 final_out_finite": final_out_finite,
    }
    return ret


test_mla.__test__ = False  # CLI benchmark; pytest collects the contract tests below.


def _str2split(value):
    if isinstance(value, str) and value.lower() == "auto":
        return None
    return int(value)


def _format_summary(rows):
    df = pd.DataFrame(rows)
    if "split_kv" in df:
        df = df.drop(columns=["split_kv"])

    init_order = {init: idx for idx, init in enumerate(["randn", "const0.25"])}
    df["_init_order"] = df["init"].map(init_order).fillna(len(init_order))
    sort_columns = [
        "_init_order",
        "batch",
        "ctx_len",
        "nhead",
        "decode_qlen",
        "mask",
        "num_kv_splits",
    ]
    df = df.sort_values(sort_columns).drop(columns=["_init_order"])

    columns = [
        "batch",
        "ctx_len",
        "nhead",
        "decode_qlen",
        "mask",
        "num_kv_splits",
        "dtype",
        "kv_dtype",
        "gfx",
        "kv_layout",
        "kv_shuffled",
        "init",
        "mi400 us",
        "mi400 TFLOPS",
        "mi400 TB/s",
        "mi400 err",
        "mi400 cos_diff",
        "mi400 final_out_finite",
    ]
    return df[[column for column in columns if column in df.columns]]


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("test_mla_mi400 unsupported on %s; skipping", get_gfx())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.fp8],
        nargs="*",
        default=[dtypes.fp8],
        metavar="{fp8}",
        help="""Q dtype. MI400 MLA currently supports fp8.
        e.g.: -d fp8""",
    )
    parser.add_argument(
        "--kv-dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.fp8],
        nargs="*",
        default=[dtypes.fp8],
        metavar="{fp8}",
        help="""KV dtype. MI400 MLA currently supports fp8.
        e.g.: --kv-dtype fp8""",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=[1, 2, 4],
        help="""Batch size.
        e.g.: -b 1 2 4""",
    )
    parser.add_argument(
        "-c",
        "--ctxLen",
        type=int,
        nargs="*",
        default=[17, 65, 128, 1024],
        help="""Context length.
        e.g.: -c 17""",
    )
    parser.add_argument(
        "-n",
        "--nhead",
        type=dtypes.str2tuple,
        nargs="*",
        default=None,
        help="""Public MI400 dispatch case as GQA,decode_qlen.
        QH128 defaults to kv-shuffled with one KV split.
        QH64 supports arbitrary positive query lengths with kv_shuffled.
        e.g.: -n 8,3 128,1; --kv-shuffled -n 64,2 64,4""",
    )
    parser.add_argument(
        "--split-kv",
        "--split_kv",
        type=_str2split,
        nargs="*",
        default=None,
        help="""KV split count per batch, or auto.
        Default: 1 for QH128; 1, 2, 3 for other head counts.
        e.g.: --split_kv 1 2 3 auto""",
    )
    parser.add_argument(
        "--mask",
        type=int,
        nargs="*",
        choices=[0, 1],
        default=[1],
        help="""Attention mask selector: 0 uses all valid KV, 1 enables causal masking.
        e.g.: --mask 0 1""",
    )
    parser.add_argument(
        "--init",
        choices=["randn", "const0.25"],
        nargs="*",
        default=["randn"],
        help="""Input initializer. const0.25 fills Q/KV/fallback pages with 0.25.
        e.g.: --init randn const0.25""",
    )
    parser.add_argument(
        "--kv-layout",
        choices=["auto", "token_major", "kv_shuffled"],
        default="auto",
        help="auto selects shuffled for QH128 or QH64 multiple queries; token_major uses legacy segmented PS64",
    )
    parser.add_argument(
        "--kv-shuffled",
        action="store_true",
        help="Select QH64/QH128 shuffled cases; QH128 also selects it in auto mode.",
    )
    args = parser.parse_args()
    if args.kv_shuffled:
        if args.kv_layout == "token_major":
            parser.error("--kv-shuffled conflicts with --kv-layout token_major")
        args.kv_layout = "kv_shuffled"
    if args.nhead is None:
        args.nhead = (
            [(128, 1), (64, 1)]
            if args.kv_layout == "kv_shuffled"
            else _MI400_DISPATCH_CASES
        )
    if any(
        len(case) != 2
        or (case not in _MI400_DISPATCH_CASES and not (case[0] == 64 and case[1] >= 1))
        for case in args.nhead
    ):
        parser.error(
            "unsupported head/query pair; QH64 accepts any positive query length"
        )
    if args.kv_layout == "token_major" and any(
        case not in _MI400_DISPATCH_CASES for case in args.nhead
    ):
        parser.error(
            "QH64 token_major requires -n 64,1; use kv_shuffled for multiple queries"
        )
    if args.kv_layout == "kv_shuffled" and any(
        not (case[0] == 64 and case[1] >= 1 or case == (128, 1)) for case in args.nhead
    ):
        parser.error("kv_shuffled requires QH64 positive query length or -n 128,1")
    if (
        (128, 1) in args.nhead
        and args.kv_layout != "token_major"
        and args.split_kv is not None
        and any(splits != 1 for splits in args.split_kv)
    ):
        parser.error("QH128 kv_shuffled requires --split-kv 1")
    dispatch_cases = [
        (nhead, decode_qlen, split_kv)
        for nhead, decode_qlen in args.nhead
        for split_kv in (
            args.split_kv
            if args.split_kv is not None
            else (
                [1] if nhead == 128 and args.kv_layout != "token_major" else [1, 2, 3]
            )
        )
    ]
    torch.set_default_device("cuda")

    rows = []
    for (
        (nhead, decode_qlen, split_kv),
        dtype,
        kv_dtype,
        batch,
        ctx_len,
        mask,
        init,
    ) in itertools.product(
        dispatch_cases,
        args.dtype,
        args.kv_dtype,
        args.batch,
        args.ctxLen,
        args.mask,
        args.init,
    ):
        if not check_support(dtype, kv_dtype, nhead):
            aiter.logger.warning(
                "skipping unsupported MLA config: dtype=%s kv_dtype=%s nhead=%d",
                dtype,
                kv_dtype,
                nhead,
            )
            continue
        rows.append(
            test_mla(
                batch,
                ctx_len,
                nhead,
                decode_qlen,
                split_kv,
                mask,
                dtype,
                kv_dtype,
                init,
                args.kv_layout,
            )
        )

    if not rows:
        aiter.logger.warning("mla_decode_pagesize64: no supported cases selected")
        return

    df = _format_summary(rows)
    aiter.logger.info(
        "mla_decode_pagesize64 summary (markdown):\n%s",
        df.to_markdown(index=False),
    )


def _make_qh128_case(lengths, stride=576, scale_values=(0.75, 1.25)):
    page_size = 64
    generator = torch.Generator().manual_seed(19)
    batch = len(lengths)
    counts = [(n + page_size - 1) // page_size for n in lengths]
    pages = max(1, sum(counts))
    width = max(1, max(counts, default=0))
    ids = torch.randperm(pages, generator=generator)
    table = torch.full((batch, width + 3), -1, dtype=torch.int32)
    q = torch.randn(batch, 128, stride, generator=generator).to(torch.float8_e4m3fn)
    # Poison Q padding: the kernel may not read beyond the 576 logical values.
    q[..., 576:] = float("nan")
    kv = torch.randn(pages, page_size, 576, generator=generator).to(torch.float8_e4m3fn)
    offset = 0
    for row, (length, count) in enumerate(zip(lengths, counts)):
        table[row, :count] = ids[offset : offset + count]
        offset += count
        if length % page_size:
            kv[table[row, count - 1], length % page_size :] = float("nan")
    raw = torch.full(
        (batch * 128 * 512 + 32,), 123, dtype=torch.bfloat16, device="cuda"
    )
    out = raw[16:-16].view(batch, 128, 512)
    raw_lse = torch.full(
        (batch * 128 + 32,), float("nan"), dtype=torch.float32, device="cuda"
    )
    lse = raw_lse[16:-16].view(batch, 128)
    # Allocate scales once during setup; every eager/graph launch reuses them.
    scales = [
        torch.tensor([v], dtype=torch.float32, device="cuda") for v in scale_values
    ]
    kwargs = {
        "q": q.to("cuda")[..., :576],
        "kv": _pack_kv_shuffled_pages(kv).to("cuda"),
        "out": out,
        "seq_lens": torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        "page_table": table.to("cuda")[:, :width],
        "q_scale": scales[0],
        "kv_scale": scales[1],
        "softmax_scale": 1 / 24,
    }
    return kwargs, lse, raw, q[..., :576].float(), kv.float(), table, lengths


def _ref_qh128_case(case):
    args, _, _, q, kv, table, lengths = case
    page_size = kv.shape[1]
    q_scale = args["q_scale"].cpu().item()
    kv_scale = args["kv_scale"].cpu().item()
    outputs = torch.zeros(len(lengths), 128, 512)
    lse = torch.full((len(lengths), 128), -float("inf"))
    for row, length in enumerate(lengths):
        if not length:
            continue
        count = (length + page_size - 1) // page_size
        keys = kv[table[row, :count].long()].reshape(-1, 576)[:length]
        scores = (q[row] @ keys.T) * (q_scale * kv_scale * args["softmax_scale"])
        outputs[row] = torch.softmax(scores, -1) @ (keys[:, :512] * kv_scale)
        lse[row] = torch.logsumexp(scores, -1)
    return outputs, lse


def _check_qh128_case(case):
    args, lse, raw, *_, lengths = case
    expected, expected_lse = _ref_qh128_case(case)
    got = args["out"].float().cpu()
    assert torch.isfinite(got).all()
    assert (raw[:16] == 123).all() and (raw[-16:] == 123).all()
    assert torch.isnan(lse._base[:16]).all() and torch.isnan(lse._base[-16:]).all()
    denom = expected.square().mean().sqrt().clamp_min(1e-8)
    assert (got - expected).square().mean().sqrt() / denom < 0.06
    torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.02, rtol=0.002)
    empty = torch.tensor(lengths) == 0
    assert (got[empty] == 0).all() and torch.isneginf(lse.cpu()[empty]).all()


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250",
    reason="requires gfx1250",
)
class TestMlaQh128:
    @pytest.mark.parametrize(
        "lengths,stride",
        [
            ([0, 1, 63, 64, 65], 576),
            (list(range(1, 65)), 768),
            ([127, 128, 129, 319, 320, 321], 576),
            ([1024, 4096, 4097, 5120], 576),
        ],
    )
    def test_reference_and_lse(self, lengths, stride):
        case = _make_qh128_case(lengths, stride)
        args, lse, *_ = case
        assert mla_decode_fwd_qh128_asm(**args, lse=lse) is args["out"]
        _check_qh128_case(case)
        previous = args["out"].clone()
        args["out"].zero_()
        assert mla_decode_fwd_qh128_asm(**args) is args["out"]
        torch.testing.assert_close(args["out"], previous, atol=0, rtol=0)

    @pytest.mark.parametrize("scale_values", [(1.0, 1.0), (0.75, 1.25)])
    def test_stream_graph_and_dynamic_metadata(self, scale_values):
        case = _make_qh128_case([65, 321], scale_values=scale_values)
        args, lse, *_ = case
        mla_decode_fwd_qh128_asm(**args, lse=lse)
        reference_output = args["out"].clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            mla_decode_fwd_qh128_asm(**args, lse=lse)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                mla_decode_fwd_qh128_asm(**args, lse=lse)
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(args["out"], reference_output, atol=0, rtol=0)
        args["seq_lens"].zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert (args["out"] == 0).all() and torch.isneginf(lse).all()

    def test_ps64_invalid_metadata_is_not_addressed(self):
        for bad_length, bad_page in [
            (None, -1),
            (None, 1000000),
            (-1, None),
            (1000000, None),
        ]:
            args, lse, *_ = _make_qh128_case([129])
            if bad_length is not None:
                args["seq_lens"].fill_(bad_length)
            if bad_page is not None:
                args["page_table"][0, 1] = bad_page
            mla_decode_fwd_qh128_asm(**args, lse=lse)
            assert (args["out"] == 0).all() and torch.isneginf(lse).all()

    def test_host_contract_rejection_and_empty_batch(self):
        args, lse, *_ = _make_qh128_case([64])
        broken = dict(args)
        broken["q"] = args["q"][:, :64]
        with pytest.raises(RuntimeError, match="Q must be"):
            mla_decode_fwd_qh128_asm(**broken, lse=lse)
        broken = dict(args)
        broken["seq_lens"] = args["seq_lens"].to(torch.int64)
        with pytest.raises(RuntimeError, match="dtype"):
            mla_decode_fwd_qh128_asm(**broken, lse=lse)
        broken = dict(args)
        broken["page_table"] = torch.zeros((1, 4), dtype=torch.int32, device="cuda")[
            :, ::2
        ]
        with pytest.raises(RuntimeError, match="page_table"):
            mla_decode_fwd_qh128_asm(**broken, lse=lse)
        broken = dict(args)
        broken["kv"] = torch.empty(
            (1, 1, 1, 576),
            dtype=torch.float8_e4m3fn,
            device="cuda",
        )
        with pytest.raises(RuntimeError, match="KV must be"):
            mla_decode_fwd_qh128_asm(**broken, lse=lse)
        args, lse, *_ = _make_qh128_case([])
        assert mla_decode_fwd_qh128_asm(**args, lse=lse).numel() == 0

    def test_rejects_missing_metadata(self):
        args, lse, *_ = _make_qh128_case([65])
        for name in ("seq_lens", "page_table"):
            broken = dict(args)
            broken[name] = None
            with pytest.raises(
                ValueError, match="seq_lens and page_table are required"
            ):
                mla_decode_fwd_qh128_asm(**broken, lse=lse)

    def test_shared_pages_and_unit_scales(self):
        case = _make_qh128_case([65, 65], scale_values=(1.0, 1.0))
        args, lse, _, _, _, table, _ = case
        table[1].copy_(table[0])
        args["page_table"][1].copy_(args["page_table"][0])
        mla_decode_fwd_qh128_asm(**args, lse=lse)
        _check_qh128_case(case)

    @pytest.mark.parametrize(
        "missing_scales", [("q_scale",), ("kv_scale",), ("q_scale", "kv_scale")]
    )
    def test_requires_explicit_scales(self, missing_scales):
        args, lse, *_ = _make_qh128_case([65])
        for name in missing_scales:
            args[name] = None
        with pytest.raises(ValueError, match="q_scale and kv_scale are required"):
            mla_decode_fwd_qh128_asm(**args, lse=lse)

    @staticmethod
    def _require_large_address_memory(nbytes):
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info()[0] < nbytes + (1 << 30):
            pytest.skip("insufficient free memory for a >4 GiB address regression")

    def test_q_and_out_beyond_4gib(self):
        # Q's padded stride crosses 4 GiB at B=43691; O crosses it at B=32768.
        batch = 48000
        self._require_large_address_memory(batch * 128 * (768 + 512 * 2 + 4))
        case = _make_qh128_case([65], stride=768)
        args, lse, *_ = case
        mla_decode_fwd_qh128_asm(**args, lse=lse)
        expected_out, expected_lse = args["out"].clone(), lse.clone()
        q = torch.zeros((batch, 128, 768), dtype=args["q"].dtype, device="cuda")[
            ..., :576
        ]
        q[-1:].copy_(args["q"])
        out = torch.full((batch, 128, 512), 123, dtype=torch.bfloat16, device="cuda")
        large_lse = torch.full((batch, 128), float("nan"), device="cuda")
        seq_lens = torch.zeros(batch, dtype=torch.int32, device="cuda")
        seq_lens[-1] = 65
        page_table = torch.full(
            (batch, args["page_table"].shape[1]),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        page_table[-1:].copy_(args["page_table"])
        args.update(seq_lens=seq_lens, page_table=page_table)
        args.update(q=q, out=out)
        mla_decode_fwd_qh128_asm(**args, lse=large_lse)
        torch.testing.assert_close(out[-1:], expected_out, atol=0, rtol=0)
        torch.testing.assert_close(large_lse[-1:], expected_lse, atol=0, rtol=0)
        for row in (0, 32768, 43691, batch - 2):
            assert (out[row] == 0).all() and torch.isneginf(large_lse[row]).all()

    def test_kv_beyond_4gib(self):
        self._require_large_address_memory((1 << 32) + (1 << 20))
        case = _make_qh128_case([65, 129])
        args, lse, *_ = case
        offset = (1 << 32) // (64 * 576) + 128
        kv = torch.empty(
            (offset + args["kv"].shape[0], 1, 64, 576),
            dtype=args["kv"].dtype,
            device="cuda",
        )
        # If a byte address wraps, it reads zeros instead of the selected KV.
        kv.view(-1)[: 1 << 20].zero_()
        kv[offset:].copy_(args["kv"])
        args["kv"] = kv
        args["page_table"] = args["page_table"] + offset
        mla_decode_fwd_qh128_asm(**args, lse=lse)
        _check_qh128_case(case)

    def test_page_table_beyond_4gib(self):
        self._require_large_address_memory((1 << 32) + 128)
        case = _make_qh128_case([65, 129])
        args, lse, *_ = case
        stride = (1 << 30) + 16  # int32 elements; row 1 starts beyond 4 GiB.
        table = torch.empty_strided(
            args["page_table"].shape, (stride, 1), dtype=torch.int32, device="cuda"
        )
        # Poison where row 1 would land if its byte offset wrapped at 32 bits.
        table.as_strided((args["page_table"].shape[1],), (1,), storage_offset=16).fill_(
            -1
        )
        table.copy_(args["page_table"])
        args["page_table"] = table
        mla_decode_fwd_qh128_asm(**args, lse=lse)
        _check_qh128_case(case)


def _run_atom_shuffled(args, **overrides):
    kwargs = {
        "q": args["q"],
        "kv_buffer": args["kv"],
        "o": args["out"],
        "qo_indptr": None,
        "kv_indptr": None,
        "kv_indices": None,
        "kv_last_page_lens": None,
        "max_seqlen_q": 1,
        "page_size": 64,
        "sm_scale": args["softmax_scale"],
        "q_scale": args["q_scale"],
        "kv_scale": args["kv_scale"],
        "kv_layout": "kv_shuffled",
        "seq_lens": args["seq_lens"],
        "page_table": args["page_table"],
    }
    kwargs.update(overrides)
    return aiter.mla.mla_decode_fwd(**kwargs)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250",
    reason="requires gfx1250",
)
class TestMlaV3Unified:
    @pytest.mark.parametrize("return_lse", [False, True])
    @pytest.mark.parametrize("causal", [False, True])
    def test_atom_cpu_scales_and_return_contract(self, return_lse, causal):
        case = _make_qh128_case([0, 65, 129])
        args, *_ = case
        expected, expected_lse = _ref_qh128_case(case)
        out, lse = _run_atom_shuffled(
            args,
            q_scale=torch.tensor(0.75, dtype=torch.float32, device="cpu"),
            kv_scale=torch.tensor(1.25, dtype=torch.float32, device="cpu"),
            return_lse=return_lse,
            causal=causal,
        )
        assert out is args["out"]
        torch.testing.assert_close(out.float().cpu(), expected, atol=0.06, rtol=0.06)
        if return_lse:
            assert lse.shape == (3, 128) and lse.dtype == torch.float32
            torch.testing.assert_close(lse.cpu(), expected_lse, atol=0.02, rtol=0.002)
        else:
            assert lse is None

    def test_scale_cache_reuses_values_and_observes_cpu_updates(self):
        case = _make_qh128_case([65], scale_values=(0.875, 1.125))
        args, *_ = case
        cpu_q = torch.tensor(0.875, dtype=torch.float32, device="cpu")
        cpu_kv = torch.tensor(1.125, dtype=torch.float32, device="cpu")
        scale = aiter.mla._mla_scale_on_device(cpu_q, args["q"], "q_scale")
        out, _ = _run_atom_shuffled(args, q_scale=cpu_q, kv_scale=cpu_kv)
        before = aiter.mla._mla_cpu_scale_on_device.cache_info()
        _run_atom_shuffled(args, q_scale=cpu_q.clone(), kv_scale=cpu_kv.clone())
        after = aiter.mla._mla_cpu_scale_on_device.cache_info()
        assert after.misses == before.misses and after.hits >= before.hits + 2
        assert aiter.mla._mla_scale_on_device(cpu_q, args["q"], "q_scale") is scale
        torch.testing.assert_close(
            out.float().cpu(), _ref_qh128_case(case)[0], atol=0.06, rtol=0.06
        )
        cpu_q.fill_(0.5)
        args["q_scale"].fill_(0.5)
        assert aiter.mla._mla_scale_on_device(cpu_q, args["q"], "q_scale") is not scale
        _run_atom_shuffled(args, q_scale=cpu_q, kv_scale=cpu_kv)
        torch.testing.assert_close(
            out.float().cpu(), _ref_qh128_case(case)[0], atol=0.06, rtol=0.06
        )

    @pytest.mark.parametrize("unit_scales", [False, True])
    def test_atom_graph_replay_with_cpu_scales(self, unit_scales):
        values = (1.0, 1.0) if unit_scales else (0.75, 1.25)
        args, *_ = _make_qh128_case([65, 129], scale_values=values)
        scales = {
            "q_scale": None if unit_scales else torch.tensor(values[0], device="cpu"),
            "kv_scale": None if unit_scales else torch.tensor(values[1], device="cpu"),
        }
        expected, expected_lse = _run_atom_shuffled(args, return_lse=True, **scales)
        expected = expected.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                out, lse = _run_atom_shuffled(args, return_lse=True, **scales)
            graph.replay()
        torch.cuda.current_stream().wait_stream(stream)
        torch.testing.assert_close(out, expected, atol=0, rtol=0)
        torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)
        args["seq_lens"].zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert (out == 0).all() and torch.isneginf(lse).all()

    def test_uncached_cpu_scale_rejected_during_capture(self, monkeypatch):
        args, *_ = _make_qh128_case([65])
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(RuntimeError, match="Warm up"):
            aiter.mla._mla_scale_on_device(
                torch.tensor(123.125, device="cpu"), args["q"], "q_scale"
            )

    @pytest.mark.parametrize(
        "overrides,message",
        [
            ({"max_seqlen_q": 2}, "max_seqlen_q=1"),
            ({"page_size": 1}, "page_size=64"),
            ({"nhead_kv": 2}, "nhead_kv=1"),
            ({"num_kv_splits": 2}, "KV splitting"),
            ({"return_logits": True}, "partial logits"),
            ({"cp_world_size": 2}, "context parallelism"),
            ({"logit_cap": 1.0}, "logit_cap"),
            ({"intra_batch_mode": True}, "split/persistent metadata"),
            ({"kv_layout": "typo"}, "unknown MLA kv_layout"),
            ({"seq_lens": None}, "seq_lens and page_table"),
        ],
    )
    def test_unsupported_options_fail_before_launch(
        self, overrides, message, monkeypatch
    ):
        args, *_ = _make_qh128_case([65])

        def unexpected_launch(*args, **kwargs):
            pytest.fail("unsupported inputs reached the native launcher")

        monkeypatch.setattr(aiter.mla, "mla_ps64_qh128_fp8_asm_fwd", unexpected_launch)
        with pytest.raises((ValueError, RuntimeError), match=message):
            _run_atom_shuffled(args, **overrides)

    @pytest.mark.parametrize(
        "field,message",
        [
            ("q", "Q must be"),
            ("kv_buffer", "dtype"),
            ("o", "dtype"),
            ("seq_lens", "seq_lens"),
            ("page_table", "page_table"),
            ("q_scale", "FP32"),
        ],
    )
    def test_atom_tensor_contract_checked_in_python(self, field, message, monkeypatch):
        args, *_ = _make_qh128_case([65])
        invalid = {
            "q": args["q"][:, :32],
            "kv_buffer": args["kv"].to(torch.bfloat16),
            "o": args["out"].float(),
            "seq_lens": torch.zeros(2, dtype=torch.int32, device="cuda"),
            "page_table": torch.zeros((2, 4), dtype=torch.int32, device="cuda"),
            "q_scale": torch.tensor(1.0, dtype=torch.float64, device="cpu"),
        }

        def unexpected_launch(*args, **kwargs):
            pytest.fail("invalid tensors reached the native launcher")

        monkeypatch.setattr(aiter.mla, "mla_ps64_qh128_fp8_asm_fwd", unexpected_launch)
        with pytest.raises((ValueError, RuntimeError), match=message):
            _run_atom_shuffled(args, **{field: invalid[field]})

    def test_wrong_arch_has_no_fallback(self, monkeypatch):
        args, *_ = _make_qh128_case([65])
        monkeypatch.setattr(aiter.mla, "get_gfx", lambda: "gfx950")
        with pytest.raises(RuntimeError, match="requires gfx1250"):
            _run_atom_shuffled(args)

    @pytest.mark.parametrize("nhead,decode_qlen", _MI400_DISPATCH_CASES)
    @pytest.mark.parametrize("splits", [1, 3])
    def test_segmented_variants_keep_reference_output(self, nhead, decode_qlen, splits):
        batch, ctx_len = 2, 321
        case = _make_mla_mi400_case(
            batch=batch,
            ctx_lens=ctx_len,
            nhead=nhead,
            decode_qlen=decode_qlen,
            num_kv_splits=splits,
        )
        q_ref = (torch.randn(batch * decode_qlen, nhead, 576, device="cuda") * 0.25).to(
            dtypes.fp8
        )
        q = _make_mla_mi400_q_case(
            q_fp8=q_ref,
            batch=batch,
            decode_qlen=decode_qlen,
            nhead=nhead,
            qk_head_dim=576,
            v_head_dim=512,
        )
        kv, kv_ref, indices = _make_mla_mi400_kv_case(
            kv_buffer_bf16=torch.randn(
                batch * 6 * 64, 1, 576, dtype=torch.bfloat16, device="cuda"
            )
            * 0.25,
            batch=batch,
            ctx_lens=ctx_len,
            qk_head_dim=576,
            v_head_dim=512,
            page_indices_oob=4,
        )
        out = torch.empty(
            (batch * decode_qlen, nhead, 512), dtype=torch.bfloat16, device="cuda"
        )
        aiter.mla.mla_decode_fwd(
            q,
            kv,
            out,
            torch.arange(batch + 1, dtype=torch.int32, device="cuda") * decode_qlen,
            case["kv_indptr"],
            indices,
            case["kv_last_page_lens"],
            decode_qlen,
            page_size=64,
            num_kv_splits=splits,
            num_kv_splits_indptr=case["num_kv_splits_indptr"],
            q_scale=case["q_scale"],
            kv_scale=case["kv_scale"],
        )
        ref = _ref_mla_mi400(
            case, q_ref, kv_ref, indices, batch, ctx_len, decode_qlen, 1, 576, 512, 1
        )
        assert torch.isfinite(out).all()
        torch.testing.assert_close(out, ref, atol=0.06, rtol=0.06)


def _make_qh64_case(
    batch,
    ctx_len,
    splits,
    init="randn",
    *,
    full_pool=False,
    lengths=None,
    q_len=1,
    q_lengths=None,
    causal=True,
):
    """Deterministic aiter QH64 fixture, with true mixed-CSR empty rows."""
    if lengths is None:
        lengths = [ctx_len] * batch
    lengths = list(lengths)
    if len(lengths) != batch or min(lengths) < 0 or splits is not None and splits < 1:
        raise ValueError("invalid batch lengths or split count")
    q_lengths = list(q_lengths) if q_lengths is not None else [q_len] * batch
    if len(q_lengths) != batch or min(q_lengths) < 0:
        raise ValueError("invalid query lengths")
    max_q, total_q = max(q_lengths), sum(q_lengths)
    counts = [(length + 63) // 64 for length in lengths]
    n_pages = sum(counts)
    if splits is None:
        splits, _ = aiter.mla.get_meta_param(
            None,
            batch,
            (n_pages + batch * 4) * 64,
            64,
            1,
            dtypes.fp8,
            tg_factor=max(1, max_q),
        )
        splits = int(splits)
    case = _make_mla_mi400_case(
        batch=batch,
        ctx_lens=ctx_len,
        nhead=64,
        decode_qlen=max(1, max_q),
        num_kv_splits=splits,
        page_indices_oob=4,
    )
    source_pages = 32768 if full_pool else max(1, n_pages)
    shape = (source_pages * 64, 1, 576)
    if init == "randn":
        source = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        q = torch.randn((total_q, 64, 576), dtype=torch.bfloat16, device="cuda")
    elif init == "const0.25":
        source = torch.full(shape, 0.25, dtype=torch.bfloat16, device="cuda")
        q = torch.full((total_q, 64, 576), 0.25, dtype=torch.bfloat16, device="cuda")
    else:
        raise ValueError(init)
    if len(set(lengths)) == 1:
        packed, kv_ref, indices = _make_mla_mi400_kv_case(
            kv_buffer_bf16=source,
            batch=batch,
            ctx_lens=lengths[0],
            qk_head_dim=576,
            v_head_dim=512,
            page_indices_oob=4,
            fallback_fill_value=0.25 if init == "const0.25" else None,
        )
    else:
        # The same random draws and page-scatter operations as the uniform
        # helper; CSR stores no page for an empty row, including all-zero batches.
        logical = source.view(-1, 64, 1, 576)[: max(1, n_pages)].clone()
        end = 0
        for length, count in zip(lengths, counts):
            end += count
            if count and length % 64:
                logical[end - 1, length % 64 :] = float("nan")
        if n_pages == 0:
            logical.fill_(float("nan"))
        physical_ids = torch.tensor(
            _make_page_permutation(n_pages, shuffle=True),
            dtype=torch.int64,
            device="cuda",
        )
        scattered = torch.empty_like(logical)
        if n_pages:
            scattered[physical_ids] = logical
        else:
            scattered.copy_(logical)
        indices = torch.zeros(n_pages + batch * 4, dtype=torch.int32, device="cuda")
        indices[:n_pages] = physical_ids.to(torch.int32)
        kv_ref = scattered.to(torch.float8_e4m3fn)
        packed = _pack_rope_split2_kv_pages(kv_ref, 512, 64)
    # Drop the large initialization pool before packing or calculating references.
    del source
    q_ref = q.to(torch.float8_e4m3fn)
    q_packed = _make_mla_mi400_q_case(
        q_fp8=q_ref,
        batch=total_q,
        decode_qlen=1,
        nhead=64,
        qk_head_dim=576,
        v_head_dim=512,
    )
    case.update(
        batch=batch,
        ctx_len=ctx_len,
        lengths=lengths,
        q_ref=q_ref,
        q=q_packed,
        q_lengths=q_lengths,
        max_q=max_q,
        total_q=total_q,
        causal=bool(causal),
        kv_ref=kv_ref,
        kv_indices=indices,
        token_major=packed,
        kv_shuffled=_pack_kv_shuffled_pages(kv_ref),
        init=init,
        qo_indptr=torch.tensor(
            [0] + list(itertools.accumulate(q_lengths)),
            dtype=torch.int32,
            device="cuda",
        ),
        kv_indptr=torch.tensor(
            [0] + list(itertools.accumulate(counts)), dtype=torch.int32, device="cuda"
        ),
        kv_last_page_lens=torch.tensor(
            [((length - 1) % 64 + 1) if length else 0 for length in lengths],
            dtype=torch.int32,
            device="cuda",
        ),
    )
    return case


def _effective_qh64_kv_lengths(case, causal=None):
    """Visible token counts in flattened query order; causal is bottom-right."""
    causal = case.get("causal", True) if causal is None else bool(causal)
    return [
        max(0, min(length, length - qlen + qi + 1)) if causal else length
        for length, qlen in zip(case["lengths"], case["q_lengths"])
        for qi in range(qlen)
    ]


def _ref_qh64_case(case, mask=None):
    """FP32 analytic attention with safe, right-aligned all-masked rows.

    Keep the original decode-one reference operations for qlen=1 so existing
    explicit-split inputs and reference values remain unchanged.
    """
    causal = case.get("causal", True) if mask is None else bool(mask)
    outputs = torch.zeros(
        (case["total_q"], 64, 512), dtype=torch.bfloat16, device="cuda"
    )
    lses = torch.full((case["total_q"], 64), -float("inf"), device="cuda")
    starts, qstarts = case["kv_indptr"].tolist(), case["qo_indptr"].tolist()
    for b, (length, qlen) in enumerate(zip(case["lengths"], case["q_lengths"])):
        if not length or not qlen:
            continue
        indices = case["kv_indices"][starts[b] : starts[b + 1]]
        q = case["q_ref"][qstarts[b] : qstarts[b + 1]]
        kv = case["kv_ref"].float().index_select(0, indices.long()).reshape(-1, 1, 576)
        kv = kv[:length] * case["kv_scale"][0]
        logits = torch.einsum(
            "qhd,kmd->hqk", q.float() * case["q_scale"][0], kv
        ) / math.sqrt(576)
        if causal:
            _apply_causal_mask_(logits)
        weights = torch.softmax(logits, dim=-1).nan_to_num(0.0)
        if qlen == 1:
            local = dict(case, num_pages_per_batch=starts[b + 1] - starts[b])
            outputs[qstarts[b] : qstarts[b + 1]] = _ref_mla_mi400(
                local, q, case["kv_ref"], indices, 1, length, 1, 1, 576, 512, causal
            )
        else:
            outputs[qstarts[b] : qstarts[b + 1]] = torch.einsum(
                "hqk,kmd->qhd", weights, kv[..., :512]
            ).to(torch.bfloat16)
        lses[qstarts[b] : qstarts[b + 1]] = torch.logsumexp(logits, dim=-1).T
    return outputs, lses


def _tail_guarded(t):
    """Port the existing qh32 OOB allocator: data ends at a 2 MiB boundary.

    Call only in a subprocess started with PYTORCH_NO_HIP_MEMORY_CACHING=1.
    Q is guarded through its physical 768-byte rows, not a contiguous 576 copy.
    The cache guard includes complete physical pages; NaN tests protect page tails.
    """
    t = t.contiguous()
    n, size, page = t.numel(), t.element_size(), 2 * 1024 * 1024
    if not n:
        return t
    count = ((n * size + page - 1) // page) * page // size
    full = torch.empty(count, dtype=t.dtype, device=t.device)
    flat = full[count - n :]
    flat.copy_(t.reshape(-1))
    assert (flat.data_ptr() + n * size) % page == 0, "guard is not page aligned"
    return flat.view(t.shape)


def _guard_qh64_case(case):
    case = dict(case)
    physical_q = torch.zeros(
        (case["q"].shape[0], 64, 768), dtype=case["q"].dtype, device="cuda"
    )
    physical_q[..., :576].copy_(case["q"])
    physical_q = _tail_guarded(physical_q)
    case["q"] = physical_q[..., :576]
    assert case["q"].stride() == (64 * 768, 768, 1)
    for name in (
        "token_major",
        "kv_shuffled",
        "qo_indptr",
        "kv_indptr",
        "kv_indices",
        "kv_last_page_lens",
        "num_kv_splits_indptr",
        "q_scale",
        "kv_scale",
    ):
        case[name] = _tail_guarded(case[name])
    return case


def _run_qh64_case(case, layout="kv_shuffled", **overrides):
    out = torch.full(
        (case["q"].shape[0], 64, 512), 123, dtype=torch.bfloat16, device="cuda"
    )
    kwargs = {
        "q": case["q"],
        "kv_buffer": case[layout],
        "o": out,
        "qo_indptr": case["qo_indptr"],
        "kv_indptr": case["kv_indptr"],
        "kv_indices": case["kv_indices"],
        "kv_last_page_lens": case["kv_last_page_lens"],
        "max_seqlen_q": case.get("max_q", 1),
        "page_size": 64,
        "nhead_kv": 1,
        "sm_scale": 1 / 24,
        "num_kv_splits": case["num_kv_splits"],
        "num_kv_splits_indptr": case["num_kv_splits_indptr"],
        "q_scale": case["q_scale"],
        "kv_scale": case["kv_scale"],
        "return_lse": True,
        "kv_layout": layout,
        "causal": case.get("causal", True),
    }
    kwargs.update(overrides)
    logits, lse = aiter.mla.mla_decode_fwd(**kwargs)
    return kwargs["o"], lse, logits


def _check_qh64_case(case, *, compare_layouts=None):
    expected, expected_lse = _ref_qh64_case(case)
    out, lse, _ = _run_qh64_case(case)
    assert torch.isfinite(out).all()
    err = checkAllclose(
        expected.float(),
        out.float(),
        rtol=0.06,
        atol=0.06,
        tol_err_ratio=0.05,
        msg="qh64 shuffled",
    )
    assert err <= 0.05, f"output mismatch ratio {err:.3%}"
    torch.testing.assert_close(lse, expected_lse, atol=0.02, rtol=0.002)
    empty = torch.tensor(_effective_qh64_kv_lengths(case), device="cuda") == 0
    assert (out[empty] == 0).all() and torch.isneginf(lse[empty]).all()
    if compare_layouts is None:
        compare_layouts = case.get("max_q", 1) == 1
    if compare_layouts:
        other, other_lse, _ = _run_qh64_case(case, "token_major")
        # Integer views compare the actual bits, including signed zero.
        assert torch.equal(out.view(torch.int16), other.view(torch.int16))
        assert torch.equal(lse.view(torch.int32), other_lse.view(torch.int32))
    return out, lse


def _check_qh64_bigraw(offset, splits, ctx_len, q_len=1, *, kv_csr_layout="page"):
    case = _make_qh64_case(2, ctx_len, splits, q_len=q_len)

    def run():
        if kv_csr_layout == "token":
            ptr, slots, last = _make_qh64_token_csr(case)
            return _run_qh64_case(
                case,
                kv_csr_layout="token",
                kv_indptr=ptr,
                kv_indices=slots,
                kv_last_page_lens=last,
            )
        return _run_qh64_case(case)

    expected, expected_lse = run()[:2]
    pages = case["kv_shuffled"].shape[0]
    nbytes = (offset + pages) * 64 * 576
    if torch.cuda.mem_get_info()[0] < nbytes + (1 << 30):
        pytest.skip("insufficient memory for large page address regression")
    storage = torch.empty((offset + pages, 1, 64, 576), dtype=dtypes.fp8, device="cuda")
    # Poison both potential 32-bit wrap destinations and the true page tail.
    storage.view(-1)[: 1 << 20].fill_(float("nan"))
    for page in range(offset, offset + pages):
        wrapped = (page * 64 * 576) % (1 << 32)
        storage.view(-1)[wrapped : wrapped + 64 * 576].fill_(float("nan"))
    storage[offset:].copy_(case["kv_shuffled"])
    case["kv_shuffled"] = storage
    case["kv_indices"] = case["kv_indices"] + offset
    out, lse, _ = run()
    assert torch.equal(out.view(torch.int16), expected.view(torch.int16))
    assert torch.equal(lse.view(torch.int32), expected_lse.view(torch.int32))


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250",
    reason="requires gfx1250",
)
class TestMlaQh64Shuffled:
    @pytest.mark.parametrize("batch", [1, 3])
    @pytest.mark.parametrize(
        "ctx_len", [1, 15, 16, 17, 31, 33, 48, 63, 64, 65, 128, 1024]
    )
    @pytest.mark.parametrize("splits", [1, 2, 3, None])
    @pytest.mark.parametrize("init", ["randn", "const0.25"])
    def test_reference_lse_and_bitwise_layouts(self, batch, ctx_len, splits, init):
        _check_qh64_case(_make_qh64_case(batch, ctx_len, splits, init))

    @pytest.mark.parametrize(
        "lengths", [[0], [0, 0, 0], [0, 1, 0, 65], [17, 0, 129, 0]]
    )
    @pytest.mark.parametrize("splits", [1, 2, 3, None])
    def test_zerolen_and_empty_splits(self, lengths, splits):
        _check_qh64_case(
            _make_qh64_case(len(lengths), max(lengths), splits, lengths=lengths)
        )

    @pytest.mark.parametrize(
        "ctx_len,splits",
        [(4096, 1), (16384, 1), (131072, 1), (16384, None), (131072, None)],
    )
    def test_long_context_and_auto_splits(self, ctx_len, splits):
        case = _make_qh64_case(1, ctx_len, splits)
        if splits is None:
            assert case["num_kv_splits"] > 1
        _check_qh64_case(case)

    @pytest.mark.parametrize("ctx_len", [0, 1, 17, 65, 1024])
    @pytest.mark.parametrize("splits", [1, 3])
    def test_raw_stage1_r_lse_valid_count_bitwise(self, ctx_len, splits):
        case = _make_qh64_case(3, ctx_len, splits)
        results = []
        for layout in ("token_major", "kv_shuffled"):
            r = torch.full((3, splits, 64, 512), 12345.0, device="cuda")
            lse = torch.full((3, splits, 64, 1), 12345.0, device="cuda")
            valid = torch.full((3,), splits, dtype=torch.int32, device="cuda")
            out = torch.empty((3, 64, 512), dtype=torch.bfloat16, device="cuda")
            final_lse = torch.empty((3, 64), device="cuda")
            aiter.mla_decode_stage1_asm_fwd(
                case["q"],
                case[layout],
                case["qo_indptr"],
                case["kv_indptr"],
                case["kv_indices"],
                case["kv_last_page_lens"],
                case["num_kv_splits_indptr"],
                None,
                None,
                None,
                1,
                64,
                1,
                1 / 24,
                r,
                lse,
                out,
                final_lse,
                case["q_scale"],
                case["kv_scale"],
                valid_split_count=valid,
                use_valid_split_count_reduce=int(splits > 1),
                kv_shuffled=int(layout == "kv_shuffled"),
            )
            results.append((r, lse, valid))
        token_r, token_lse, token_valid = results[0]
        shuffled_r, shuffled_lse, shuffled_valid = results[1]
        active = min(splits, (ctx_len + 63) // 64)
        assert (shuffled_valid == active).all()
        if active:
            assert torch.equal(token_valid, shuffled_valid)
        for a, b in (
            (token_r[:, :active], shuffled_r[:, :active]),
            (token_lse[:, :active], shuffled_lse[:, :active]),
        ):
            assert torch.equal(a.view(torch.int32), b.view(torch.int32))
        # The new shader materializes an identity for every inactive query/split;
        # the original token-major shader may leave these slots unwritten.
        assert (shuffled_r[:, active:] == 0).all()
        assert (shuffled_lse[:, active:] == -1e20).all()

    def test_zerolen_auto_without_index_padding(self):
        case = _make_qh64_case(3, 0, 1)
        case["kv_indices"] = case["kv_indices"][:0]
        out, lse, _ = _run_qh64_case(
            case, num_kv_splits=None, num_kv_splits_indptr=None
        )
        assert (out == 0).all() and torch.isneginf(lse).all()

    @pytest.mark.parametrize("splits", [1, 3])
    def test_zerolen_with_empty_physical_pool(self, splits):
        case = _make_qh64_case(3, 0, splits)
        out, lse, _ = _run_qh64_case(case, kv_buffer=case["kv_shuffled"][:0])
        assert (out == 0).all() and torch.isneginf(lse).all()

    @pytest.mark.parametrize("splits", [1, 3])
    def test_direct_bf16_without_lse(self, splits):
        case = _make_qh64_case(3, 65, splits)
        expected, _ = _ref_qh64_case(case)
        out, _, _ = _run_qh64_case(case, return_lse=False)
        assert torch.isfinite(out).all()
        err = checkAllclose(
            expected.float(), out.float(), rtol=0.06, atol=0.06, tol_err_ratio=0.05
        )
        assert err <= 0.05
        other, _, _ = _run_qh64_case(case, "token_major", return_lse=False)
        assert torch.equal(out.view(torch.int16), other.view(torch.int16))

    @pytest.mark.parametrize(
        "offset",
        sorted(
            set(
                [
                    (1 << bits) // (64 * 576) + side
                    for bits in (31, 32)
                    for side in (-1, 0, 1)
                ]
                + [(1 << bits) + side for bits in (18, 19) for side in (-1, 0, 1)]
            )
        ),
    )
    @pytest.mark.parametrize("splits", [1, 3])
    def test_bigraw_page_address(self, offset, splits):
        _check_qh64_bigraw(offset, splits, 129)

    @pytest.mark.parametrize("offset", [(1 << 32) // (64 * 576) + 1, (1 << 19) + 1])
    @pytest.mark.parametrize("splits", [1, 3])
    def test_bigraw_hot_loop(self, offset, splits):
        # 64 pages per row exercise the rolling LTD address updates beyond
        # bootstrap, including both scheduled update implementations.
        _check_qh64_bigraw(offset, splits, 4096)

    @pytest.mark.parametrize(
        "ctx_len,splits",
        [
            (1, 1),
            (15, 3),
            (16, 2),
            (17, 3),
            (31, 3),
            (33, 3),
            (48, 2),
            (63, 3),
            (64, 1),
            (65, 3),
            (0, 3),
            (129, 3),
        ],
    )
    def test_oob_guarded_allocations(self, ctx_len, splits):
        code = (
            "import sys; sys.path.insert(0, 'op_tests'); "
            "from test_mla_decode_pagesize64 import _run_qh64_oob_worker; "
            f"_run_qh64_oob_worker({ctx_len}, {splits})"
        )
        env = dict(
            os.environ, PYTORCH_NO_HIP_MEMORY_CACHING="1", HSA_ENABLE_COREDUMP="0"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize(
        "field,value,message",
        [
            ("max_seqlen_q", 0, "max_seqlen_q"),
            ("page_size", 1, "page_size=64"),
            ("nhead_kv", 2, "nhead_kv=1"),
            ("qo_indptr", None, "CSR metadata"),
            ("kv_indptr", None, "CSR metadata"),
            ("kv_indices", None, "CSR metadata"),
            ("kv_last_page_lens", None, "CSR metadata"),
            ("num_kv_splits", 0, "positive"),
            ("sm_scale", float("nan"), "softmax_scale must be finite"),
            ("sm_scale", float("inf"), "softmax_scale must be finite"),
            ("sm_scale", float("-inf"), "softmax_scale must be finite"),
            ("cp_world_size", 2, "context parallelism"),
            ("logit_cap", 1.0, "logit_cap"),
            ("intra_batch_mode", True, "persistent metadata"),
        ],
    )
    def test_invalid_options_rejected(self, field, value, message, monkeypatch):
        case = _make_qh64_case(1, 65, 1)

        def unexpected(*args, **kwargs):
            pytest.fail("invalid input reached stage1")

        monkeypatch.setattr(aiter, "mla_decode_stage1_asm_fwd", unexpected)
        with pytest.raises((ValueError, RuntimeError), match=message):
            _run_qh64_case(case, **{field: value})

    @pytest.mark.parametrize(
        "kind,message",
        [
            ("q_stride", "stride 768"),
            ("q_heads", "Q must be"),
            ("kv_dtype", "dtype"),
            ("kv_shape", "KV must be"),
            ("metadata_dtype", "dtype"),
            ("output_dtype", "dtype"),
            ("scale", "FP32"),
        ],
    )
    def test_invalid_tensors_rejected(self, kind, message, monkeypatch):
        case = _make_qh64_case(1, 65, 1)
        bad = {
            "q_stride": {"q": case["q"].contiguous()},
            "q_heads": {"q": case["q"][:, :32]},
            "kv_dtype": {"kv_buffer": case["kv_shuffled"].bfloat16()},
            "kv_shape": {"kv_buffer": case["kv_shuffled"][:, :, :32]},
            "metadata_dtype": {"kv_indptr": case["kv_indptr"].long()},
            "output_dtype": {"o": torch.empty((1, 64, 512), device="cuda")},
            "scale": {"q_scale": torch.ones(2, device="cuda")},
        }

        def unexpected(*args, **kwargs):
            pytest.fail("invalid tensor reached stage1")

        monkeypatch.setattr(aiter, "mla_decode_stage1_asm_fwd", unexpected)
        with pytest.raises((ValueError, RuntimeError), match=message):
            _run_qh64_case(case, **bad[kind])

    def test_alternate_kv_view_and_graph(self):
        case = _make_qh64_case(2, 65, 3)
        expected, expected_lse = _run_qh64_case(case)[:2]
        case["kv_shuffled"] = case["kv_shuffled"].view(-1, 64, 1, 576)
        _run_qh64_case(case)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, lse, _ = _run_qh64_case(case)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected) and torch.equal(lse, expected_lse)


def _run_qh64_gluon_case(case, *, out=None):
    """Public Gluon comparison using identical FP8 storage and scalar scales.

    Gluon decode derives one uniform query length from total_q / batch. Ragged
    query CSR and empty causal prefixes are tested against the analytic reference.
    """
    from aiter.ops.triton.attention.mla import mla_decode_fwd as gluon_decode

    q_lengths = case.get("q_lengths", [1] * case["batch"])
    assert len(set(q_lengths)) == 1 and q_lengths[0] > 0
    assert min(case["lengths"]) >= q_lengths[0]
    assert case.get("causal", True)
    starts = case["kv_indptr"].tolist()
    counts = [b - a for a, b in itertools.pairwise(starts)]
    table = torch.zeros(
        (case["batch"], max(counts)), dtype=torch.int32, device=case["q"].device
    )
    for b, count in enumerate(counts):
        table[b, :count] = case["kv_indices"][starts[b] : starts[b + 1]]
    if out is None:
        out = torch.full(
            (case["q"].shape[0], 64, 512),
            float("nan"),
            dtype=torch.bfloat16,
            device=case["q"].device,
        )
    gluon_decode(
        q=case["q"],
        kv_buffer=case["kv_shuffled"],
        out=out,
        cu_seqlens_q=case["qo_indptr"],
        seqused_k=torch.tensor(case["lengths"], dtype=torch.int32, device=out.device),
        max_seqlen_kv=max(case["lengths"]),
        block_tables=table,
        softmax_scale=1 / 24,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        causal=True,
        q_descale=case["q_scale"],
        kv_descale=case["kv_scale"],
        shuffled_kv_cache=True,
    )
    return out


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250",
    reason="requires gfx1250",
)
class TestMlaQh64ShuffledCausal:
    @pytest.mark.parametrize("lengths", [[0, 65, 1], [1024, 17, 0], [0, 0, 0]])
    def test_public_auto_with_compact_csr(self, lengths, monkeypatch):
        case = _make_qh64_case(3, max(lengths), 1, lengths=lengths, q_len=4)
        case["kv_indices"] = case["kv_indices"][: sum((x + 63) // 64 for x in lengths)]
        original = aiter.mla_decode_stage1_asm_fwd
        selected = []

        def observe(*args, **kwargs):
            selected.append(args[14].shape[1])
            return original(*args, **kwargs)

        monkeypatch.setattr(aiter, "mla_decode_stage1_asm_fwd", observe)
        expected, expected_lse = _ref_qh64_case(case)
        out, lse, _ = _run_qh64_case(
            case, num_kv_splits=None, num_kv_splits_indptr=None
        )
        assert torch.isfinite(out).all()
        err = checkAllclose(expected.float(), out.float(), rtol=0.06, atol=0.06)
        assert err <= 0.05
        torch.testing.assert_close(lse, expected_lse, atol=0.02, rtol=0.002)
        expected_splits, _ = (
            aiter.mla.get_meta_param(
                None,
                3,
                case["kv_indices"].numel() * 64,
                64,
                1,
                dtypes.fp8,
                tg_factor=4,
            )
            if case["kv_indices"].numel()
            else (1, None)
        )
        assert selected == [int(expected_splits)]

    @pytest.mark.parametrize(
        "q_len,ctx_len",
        [
            (2, 1),
            (2, 2),
            (2, 3),
            (3, 3),
            (4, 3),
            (4, 4),
            (4, 5),
            (3, 15),
            (3, 16),
            (3, 17),
            (4, 31),
            (4, 32),
            (4, 33),
            (4, 63),
            (4, 64),
            (4, 65),
            (8, 127),
            (8, 128),
            (8, 129),
            (8, 1024),
            (65, 64),
            (65, 129),
        ],
    )
    @pytest.mark.parametrize("splits", [1, 3, None])
    @pytest.mark.parametrize("init", ["randn", "const0.25"])
    def test_reference_and_lse(self, q_len, ctx_len, splits, init):
        _check_qh64_case(_make_qh64_case(3, ctx_len, splits, init, q_len=q_len))

    @pytest.mark.parametrize(
        "q_lengths,lengths",
        [
            ([0, 2, 1, 4], [0, 1, 65, 129]),
            ([3, 0, 2, 5], [0, 65, 1, 4]),
            ([1, 4, 2], [0, 0, 0]),
            ([2, 3, 1], [1, 17, 64]),
        ],
    )
    @pytest.mark.parametrize("splits", [1, 3, None])
    @pytest.mark.parametrize("causal", [False, True])
    def test_variable_query_and_zero_lengths(self, q_lengths, lengths, splits, causal):
        case = _make_qh64_case(
            len(lengths),
            max(lengths),
            splits,
            lengths=lengths,
            q_lengths=q_lengths,
            causal=causal,
        )
        _check_qh64_case(case, compare_layouts=False)

    @pytest.mark.parametrize("splits", [1, 3])
    @pytest.mark.parametrize("return_lse", [False, True])
    @pytest.mark.parametrize("causal", [False, True])
    def test_output_modes_and_unit_scales(self, splits, return_lse, causal):
        case = _make_qh64_case(2, 65, splits, q_len=4, causal=causal)
        case["q_scale"].fill_(1)
        case["kv_scale"].fill_(1)
        expected, expected_lse = _ref_qh64_case(case)
        out, lse, _ = _run_qh64_case(case, return_lse=return_lse)
        err = checkAllclose(expected.float(), out.float(), rtol=0.06, atol=0.06)
        assert err <= 0.05 and torch.isfinite(out).all()
        if return_lse:
            torch.testing.assert_close(lse, expected_lse, atol=0.02, rtol=0.002)

    @pytest.mark.parametrize("splits", [1, 3])
    @pytest.mark.parametrize("causal", [False, True])
    def test_raw_partials_and_batch_split_upper_bound(self, splits, causal):
        case = _make_qh64_case(
            2, 65, splits, lengths=[65, 1], q_lengths=[4, 2], causal=causal
        )
        r = torch.full((6, splits, 64, 512), 12345.0, device="cuda")
        lse = torch.full((6, splits, 64, 1), 12345.0, device="cuda")
        valid = torch.full((2,), -1, dtype=torch.int32, device="cuda")
        aiter.mla_decode_stage1_asm_fwd(
            case["q"],
            case["kv_shuffled"],
            case["qo_indptr"],
            case["kv_indptr"],
            case["kv_indices"],
            case["kv_last_page_lens"],
            case["num_kv_splits_indptr"],
            None,
            None,
            None,
            case["max_q"],
            64,
            1,
            1 / 24,
            r,
            lse,
            torch.empty((6, 64, 512), dtype=torch.bfloat16, device="cuda"),
            torch.empty((6, 64), device="cuda"),
            case["q_scale"],
            case["kv_scale"],
            valid_split_count=valid,
            use_valid_split_count_reduce=int(splits > 1),
            causal=causal,
            kv_shuffled=1,
        )
        assert valid.tolist() == [min(splits, 2), 1]
        effective = _effective_qh64_kv_lengths(case)
        qstarts = case["qo_indptr"].tolist()
        starts = case["kv_indptr"].tolist()
        for b in range(2):
            ids = case["kv_indices"][starts[b] : starts[b + 1]].long()
            kv = case["kv_ref"].float().index_select(0, ids).reshape(-1, 576)
            kv = kv * case["kv_scale"][0]
            for qi in range(qstarts[b], qstarts[b + 1]):
                for split in range(min(splits, (effective[qi] + 63) // 64)):
                    positions = [
                        token
                        for block in range(split * 64, effective[qi], splits * 64)
                        for token in range(block, min(block + 64, effective[qi]))
                    ]
                    values = kv[positions]
                    scores = (
                        case["q_ref"][qi].float() * case["q_scale"][0] @ values.T / 24
                    )
                    expected = torch.softmax(scores, -1) @ values[:, :512]
                    err = checkAllclose(expected, r[qi, split], rtol=0.06, atol=0.06)
                    assert err <= 0.05
                    torch.testing.assert_close(
                        lse[qi, split, :, 0],
                        torch.logsumexp(scores, -1),
                        atol=0.02,
                        rtol=0.002,
                    )

    @pytest.mark.parametrize("return_lse", [False, True])
    def test_empty_query_batch(self, return_lse, monkeypatch):
        case = _make_qh64_case(2, 0, 3, q_lengths=[0, 0])

        def unexpected(*args, **kwargs):
            pytest.fail("empty query batch reached stage1")

        monkeypatch.setattr(aiter, "mla_decode_stage1_asm_fwd", unexpected)
        out, lse, _ = _run_qh64_case(case, max_seqlen_q=1, return_lse=return_lse)
        assert out.shape == (0, 64, 512)
        if return_lse:
            assert lse.shape == (0, 64)

    @pytest.mark.parametrize("value", [0, -1, 1.5])
    def test_invalid_query_limit_rejected_before_stage1(self, value, monkeypatch):
        case = _make_qh64_case(2, 65, 1, q_len=4)

        def unexpected(*args, **kwargs):
            pytest.fail("invalid query limit reached stage1")

        monkeypatch.setattr(aiter, "mla_decode_stage1_asm_fwd", unexpected)
        with pytest.raises(ValueError, match="max_seqlen_q"):
            _run_qh64_case(case, max_seqlen_q=value)

    @pytest.mark.parametrize("splits", [1, 3])
    def test_future_token_does_not_change_earlier_queries(self, splits):
        case = _make_qh64_case(1, 8, splits, "const0.25", q_len=4)
        before, before_lse = _check_qh64_case(case)
        page = int(case["kv_indices"][0])
        case["kv_ref"][page, 7].fill_(4)
        case["kv_shuffled"] = _pack_kv_shuffled_pages(case["kv_ref"])
        after, after_lse = _check_qh64_case(case)
        assert torch.equal(before[:-1], after[:-1])
        assert torch.equal(before_lse[:-1], after_lse[:-1])
        assert not torch.equal(before[-1], after[-1])
        noncausal = dict(case, causal=False)
        full, _ = _check_qh64_case(noncausal)
        assert not torch.equal(full[:-1], after[:-1])
        assert torch.equal(full, full[-1:].expand_as(full))

    @pytest.mark.parametrize("q_len", [1, 2, 3, 4, 8])
    @pytest.mark.parametrize("ctx_len", [64, 128, 1024, 4096])
    def test_same_input_gluon(self, q_len, ctx_len):
        case = _make_qh64_case(3, ctx_len, None, q_len=q_len)
        expected, _ = _ref_qh64_case(case)
        asm, _ = _check_qh64_case(case, compare_layouts=False)
        gluon = _run_qh64_gluon_case(case)
        assert torch.isfinite(gluon).all()
        for actual, label in ((gluon, "Gluon reference"), (asm, "ASM vs Gluon")):
            target = expected if label == "Gluon reference" else gluon
            err = checkAllclose(
                target.float(),
                actual.float(),
                rtol=0.06,
                atol=0.06,
                tol_err_ratio=0.05,
                msg=label,
            )
            assert err <= 0.05

    def test_graph_replay_reads_variable_query_metadata(self):
        case = _make_qh64_case(2, 65, 3, q_lengths=[1, 3])
        _run_qh64_case(case)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, lse, _ = _run_qh64_case(case)
        for q_lengths, lengths in (([1, 3], [65, 65]), ([2, 2], [64, 1])):
            case["q_lengths"] = list(q_lengths)
            case["lengths"] = list(lengths)
            case["qo_indptr"].copy_(
                torch.tensor(
                    [0] + list(itertools.accumulate(q_lengths)),
                    dtype=torch.int32,
                    device="cuda",
                )
            )
            # Shorten both CSR page ranges; row 1 then uses the old first
            # sequence's second physical page, whose first token is valid.
            if lengths == [64, 1]:
                case["kv_indptr"].copy_(
                    torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda")
                )
            case["kv_last_page_lens"].copy_(
                torch.tensor(
                    [((length - 1) % 64 + 1) if length else 0 for length in lengths],
                    dtype=torch.int32,
                    device="cuda",
                )
            )
            expected, expected_lse = _ref_qh64_case(case)
            out.fill_(float("nan"))
            lse.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.isfinite(out).all()
            err = checkAllclose(expected.float(), out.float(), rtol=0.06, atol=0.06)
            assert err <= 0.05
            torch.testing.assert_close(lse, expected_lse, atol=0.02, rtol=0.002)

    @pytest.mark.parametrize(
        "q_len,ctx_len,splits", [(4, 0, 3), (4, 1, 3), (4, 65, 1), (8, 129, 3)]
    )
    def test_oob_guarded_allocations(self, q_len, ctx_len, splits):
        code = (
            "import sys; sys.path.insert(0, 'op_tests'); "
            "from test_mla_decode_pagesize64 import _run_qh64_oob_worker; "
            f"_run_qh64_oob_worker({ctx_len}, {splits}, {q_len})"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            env=dict(
                os.environ, PYTORCH_NO_HIP_MEMORY_CACHING="1", HSA_ENABLE_COREDUMP="0"
            ),
            text=True,
            capture_output=True,
            check=False,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr

    @pytest.mark.parametrize("splits", [1, 3])
    @pytest.mark.parametrize("ctx_len", [129, 4096])
    def test_bigraw_causal(self, splits, ctx_len):
        _check_qh64_bigraw((1 << 32) // (64 * 576) + 1, splits, ctx_len, q_len=4)


def _qh64_page_inputs(case):
    starts = case["kv_indptr"].tolist()
    width = max((b - a for a, b in itertools.pairwise(starts)), default=0)
    table = torch.full(
        (case["batch"], width + 2), -1, dtype=torch.int32, device="cuda"
    )[:, :width]
    for row in range(case["batch"]):
        table[row, : starts[row + 1] - starts[row]] = case["kv_indices"][
            starts[row] : starts[row + 1]
        ]
    lengths = torch.tensor(case["lengths"], dtype=torch.int32, device="cuda")
    indptr = torch.empty_like(case["kv_indptr"])
    indices = torch.full(
        (case["batch"] * width,), -123, dtype=torch.int32, device="cuda"
    )
    last = torch.empty_like(case["kv_last_page_lens"])
    return lengths, table, indptr, indices, last


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250",
    reason="requires gfx1250",
)
class TestMlaQh64PageMetadata:
    @pytest.mark.parametrize("batch", [1, 3])
    @pytest.mark.parametrize("q_len", [1, 4])
    @pytest.mark.parametrize("ctx_len", [64, 128])
    @pytest.mark.parametrize("splits", [1, 3, None])
    def test_page_csr_same_input_reference_and_gluon(
        self, batch, q_len, ctx_len, splits
    ):
        case = _make_qh64_case(batch, ctx_len, 1, q_len=q_len)
        args = _qh64_page_inputs(case)
        indptr, indices, last = aiter.mla.mla_decode_prepare_page_csr(*args)
        torch.testing.assert_close(indptr, case["kv_indptr"], atol=0, rtol=0)
        torch.testing.assert_close(last, case["kv_last_page_lens"], atol=0, rtol=0)
        count = int(indptr[-1])
        torch.testing.assert_close(
            indices[:count], case["kv_indices"][:count], atol=0, rtol=0
        )
        case.update(kv_indptr=indptr, kv_indices=indices, kv_last_page_lens=last)
        out, lse, _ = _run_qh64_case(
            case, num_kv_splits=splits, num_kv_splits_indptr=None
        )
        expected, expected_lse = _ref_qh64_case(case)
        gluon = _run_qh64_gluon_case(case)
        torch.testing.assert_close(out, expected, atol=0.06, rtol=0.06)
        torch.testing.assert_close(out, gluon, atol=0.06, rtol=0.06)
        torch.testing.assert_close(lse, expected_lse, atol=0.02, rtol=0.002)

    @pytest.mark.parametrize("lengths", [[0, 65, 129], [0, 0, 0], [1, 2, 0]])
    @pytest.mark.parametrize("causal", [False, True])
    @pytest.mark.parametrize("splits", [1, 3])
    def test_ragged_empty_and_poisoned_tail(self, lengths, causal, splits):
        case = _make_qh64_case(
            3, max(lengths), splits, lengths=lengths, q_lengths=[0, 4, 2], causal=causal
        )
        args = _qh64_page_inputs(case)
        indptr, indices, last = aiter.mla.mla_decode_prepare_page_csr(*args)
        count = int(indptr[-1])
        assert (indices[count:] == -123).all()
        case.update(kv_indptr=indptr, kv_indices=indices, kv_last_page_lens=last)
        _check_qh64_case(case, compare_layouts=False)

    def test_graph_replay_rebuilds_csr_without_reallocating(self):
        case = _make_qh64_case(2, 129, 3, q_len=4)
        args = _qh64_page_inputs(case)
        lengths, _table, indptr, indices, last = args
        case.update(kv_indptr=indptr, kv_indices=indices, kv_last_page_lens=last)
        out = torch.empty((8, 64, 512), dtype=torch.bfloat16, device="cuda")

        def run():
            aiter.mla.mla_decode_prepare_page_csr(*args)
            _run_qh64_case(case, o=out)

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        for new_lengths in ([129, 129], [65, 0], [0, 0]):
            case["lengths"] = list(new_lengths)
            lengths.copy_(torch.tensor(new_lengths, dtype=torch.int32, device="cuda"))
            out.fill_(float("nan"))
            graph.replay()
            expected, _ = _ref_qh64_case(case)
            torch.testing.assert_close(out, expected, atol=0.06, rtol=0.06)
            assert torch.equal(
                indptr,
                torch.tensor(
                    [
                        0,
                        (new_lengths[0] + 63) // 64,
                        sum((x + 63) // 64 for x in new_lengths),
                    ],
                    dtype=torch.int32,
                    device="cuda",
                ),
            )

    @pytest.mark.parametrize(
        "kind", ["seq_dtype", "table_stride", "capacity", "last_shape", "device"]
    )
    def test_invalid_metadata_rejected(self, kind):
        args = list(_qh64_page_inputs(_make_qh64_case(2, 128, 1)))
        if kind == "seq_dtype":
            args[0] = args[0].long()
        elif kind == "table_stride":
            args[1] = args[1][:, ::2]
        elif kind == "capacity":
            args[3] = args[3][:0]
        elif kind == "last_shape":
            args[4] = args[4][:1]
        else:
            args[1] = args[1].cpu()
        with pytest.raises(ValueError):
            aiter.mla.mla_decode_prepare_page_csr(*args)

    def test_empty_batch(self):
        empty = torch.empty(0, dtype=torch.int32, device="cuda")
        ptr = torch.full((1,), -1, dtype=torch.int32, device="cuda")
        aiter.mla.mla_decode_prepare_page_csr(
            empty, empty.view(0, 1), ptr, empty, empty
        )
        assert ptr.item() == 0


def _make_qh64_token_csr(case, *, capacity=None):
    """Expand the page fixture into ATOM slot IDs without changing its reference.

    Returns (token_indptr, slots, last). Rows are packed without page padding;
    unused capacity is poisoned. Last-page values deliberately contain ones,
    as in token/page_size=1 metadata, and must be ignored by token mode.
    """
    lengths = case["lengths"]
    starts = case["kv_indptr"].tolist()
    total = sum(lengths)
    capacity = total if capacity is None else capacity
    if capacity < total:
        raise ValueError("token CSR capacity is smaller than the token count")
    device = case["kv_indices"].device
    ptr = torch.tensor(
        [0] + list(itertools.accumulate(lengths)), dtype=torch.int32, device=device
    )
    slots = torch.full((capacity,), -1, dtype=torch.int32, device=device)
    offsets = torch.arange(64, dtype=torch.int32, device=device)
    end = 0
    for row, length in enumerate(lengths):
        pages = case["kv_indices"][starts[row] : starts[row + 1]]
        slots[end : end + length] = (pages[:, None] * 64 + offsets).flatten()[:length]
        end += length
    return ptr, slots, torch.ones_like(case["kv_last_page_lens"])


def _check_qh64_token_case(case, *, token_csr=None, return_lse=True):
    expected, expected_lse = _ref_qh64_case(case)
    page_out, page_lse, _ = _run_qh64_case(case, return_lse=return_lse)
    ptr, slots, last = _make_qh64_token_csr(case) if token_csr is None else token_csr
    out, lse, _ = _run_qh64_case(
        case,
        kv_indptr=ptr,
        kv_indices=slots,
        kv_last_page_lens=last,
        kv_csr_layout="token",
        return_lse=return_lse,
    )
    assert torch.equal(out.view(torch.int16), page_out.view(torch.int16))
    assert torch.isfinite(out).all()
    err = checkAllclose(expected.float(), out.float(), rtol=0.06, atol=0.06)
    assert err <= 0.05
    if return_lse:
        assert torch.equal(lse.view(torch.int32), page_lse.view(torch.int32))
        torch.testing.assert_close(lse, expected_lse, atol=0.02, rtol=0.002)
    empty = torch.tensor(_effective_qh64_kv_lengths(case), device=out.device) == 0
    assert (out[empty] == 0).all()
    if return_lse:
        assert torch.isneginf(lse[empty]).all()
    return out, lse


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx1250",
    reason="requires gfx1250",
)
class TestMlaQh64TokenCsr:
    def test_default_page_mode_is_unchanged(self):
        case = _make_qh64_case(2, 65, 3, q_lengths=[1, 3])
        default = _run_qh64_case(case)
        explicit = _run_qh64_case(case, kv_csr_layout="page")
        for before, after in zip(default, explicit):
            assert torch.equal(before, after)

    @pytest.mark.parametrize("ctx_len", [0, 1, 63, 64, 65, 127, 128, 129, 4096])
    @pytest.mark.parametrize("splits", [1, 3])
    @pytest.mark.parametrize("causal", [False, True])
    def test_page_parity_and_reference(self, ctx_len, splits, causal):
        case = _make_qh64_case(
            3,
            ctx_len,
            splits,
            lengths=[1, ctx_len, 0],
            q_lengths=[4, 2, 1],
            causal=causal,
        )
        _check_qh64_token_case(case)

    @pytest.mark.parametrize("splits", [1, 3])
    @pytest.mark.parametrize("return_lse", [False, True])
    def test_ragged_queries_last_values_and_capacity(self, splits, return_lse):
        case = _make_qh64_case(
            4, 129, splits, lengths=[1, 65, 0, 129], q_lengths=[0, 4, 2, 3]
        )
        ptr, slots, last = _make_qh64_token_csr(case, capacity=512)
        assert ptr.tolist() == [0, 1, 66, 66, 195]
        # Neither shader length calculation nor stage2 may use these values.
        last.copy_(torch.tensor([-123, 0, 999, 17], device=last.device))
        _check_qh64_token_case(
            case, token_csr=(ptr, slots, last), return_lse=return_lse
        )

    @pytest.mark.parametrize("splits", [1, 3])
    @pytest.mark.parametrize("return_lse", [False, True])
    def test_empty_kv_and_physical_pool(self, splits, return_lse):
        case = _make_qh64_case(3, 0, splits, q_lengths=[0, 4, 2])
        case["kv_shuffled"] = case["kv_shuffled"][:0]
        _check_qh64_token_case(case, return_lse=return_lse)

    @pytest.mark.parametrize("lengths", [[0, 65, 1], [1024, 17, 0], [0, 0, 0]])
    @pytest.mark.parametrize("padding", [0, 257])
    def test_auto_uses_token_capacity(self, lengths, padding, monkeypatch):
        case = _make_qh64_case(3, max(lengths), 1, lengths=lengths, q_len=4)
        ptr, slots, last = _make_qh64_token_csr(case, capacity=sum(lengths) + padding)
        original = aiter.mla.get_meta_param
        observed_capacity = []

        def observe(*args, **kwargs):
            observed_capacity.append(args[2])
            return original(*args, **kwargs)

        monkeypatch.setattr(aiter.mla, "get_meta_param", observe)
        expected, expected_lse = _ref_qh64_case(case)
        out, lse, _ = _run_qh64_case(
            case,
            kv_indptr=ptr,
            kv_indices=slots,
            kv_last_page_lens=last,
            kv_csr_layout="token",
            num_kv_splits=None,
            num_kv_splits_indptr=None,
        )
        assert observed_capacity == ([slots.numel()] if slots.numel() else [])
        err = checkAllclose(expected.float(), out.float(), rtol=0.06, atol=0.06)
        assert err <= 0.05 and torch.isfinite(out).all()
        torch.testing.assert_close(lse, expected_lse, atol=0.02, rtol=0.002)

    @pytest.mark.parametrize("splits", [1, 3])
    @pytest.mark.parametrize("causal", [False, True])
    def test_raw_partials_and_valid_count(self, splits, causal):
        case = _make_qh64_case(
            3, 129, splits, lengths=[1, 129, 0], q_lengths=[4, 2, 1], causal=causal
        )
        ptr, slots, last = _make_qh64_token_csr(case)
        results = []
        for is_token in (False, True):
            r = torch.full((7, splits, 64, 512), 12345.0, device="cuda")
            lse = torch.full((7, splits, 64, 1), 12345.0, device="cuda")
            valid = torch.full((3,), -1, dtype=torch.int32, device="cuda")
            aiter.mla_decode_stage1_asm_fwd(
                case["q"],
                case["kv_shuffled"],
                case["qo_indptr"],
                ptr if is_token else case["kv_indptr"],
                slots if is_token else case["kv_indices"],
                last if is_token else case["kv_last_page_lens"],
                case["num_kv_splits_indptr"],
                None,
                None,
                None,
                case["max_q"],
                64,
                1,
                1 / 24,
                r,
                lse,
                torch.empty((7, 64, 512), dtype=torch.bfloat16, device="cuda"),
                torch.empty((7, 64), device="cuda"),
                case["q_scale"],
                case["kv_scale"],
                valid_split_count=valid,
                use_valid_split_count_reduce=int(splits > 1),
                causal=causal,
                kv_shuffled=1,
                kv_indptr_is_token_level=is_token,
            )
            results.append((r, lse, valid))
        for page, token in zip(*results):
            assert torch.equal(page.view(torch.int32), token.view(torch.int32))
        assert results[1][2].tolist() == [1, min(splits, 3), 0]

    def test_graph_replay_reads_token_metadata(self):
        case = _make_qh64_case(2, 129, 3, q_lengths=[1, 3])
        ptr, slots, last = _make_qh64_token_csr(case, capacity=258)

        def run():
            return _run_qh64_case(
                case,
                kv_indptr=ptr,
                kv_indices=slots,
                kv_last_page_lens=last,
                kv_csr_layout="token",
            )

        run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out, lse, _ = run()
        for lengths, q_lengths in (
            ([129, 129], [1, 3]),
            ([1, 65], [2, 2]),
            ([0, 0], [3, 1]),
        ):
            case["lengths"] = lengths
            case["q_lengths"] = q_lengths
            case["qo_indptr"].copy_(
                torch.tensor([0] + list(itertools.accumulate(q_lengths)), device="cuda")
            )
            # Compact the page reference rows, then regenerate the matching
            # token CSR into the same graph-owned pointers.
            counts = [(n + 63) // 64 for n in lengths]
            case["kv_indptr"].copy_(
                torch.tensor([0] + list(itertools.accumulate(counts)), device="cuda")
            )
            case["kv_last_page_lens"].copy_(
                torch.tensor(
                    [((n - 1) % 64 + 1) if n else 0 for n in lengths], device="cuda"
                )
            )
            next_ptr, next_slots, _ = _make_qh64_token_csr(case, capacity=slots.numel())
            ptr.copy_(next_ptr)
            slots.copy_(next_slots)
            last.fill_(-123)
            expected, expected_lse = _ref_qh64_case(case)
            out.fill_(float("nan"))
            lse.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            err = checkAllclose(expected.float(), out.float(), rtol=0.06, atol=0.06)
            assert err <= 0.05 and torch.isfinite(out).all()
            torch.testing.assert_close(lse, expected_lse, atol=0.02, rtol=0.002)

    @pytest.mark.parametrize("mode", ["invalid", None, 1])
    def test_unknown_layout_rejected(self, mode, monkeypatch):
        case = _make_qh64_case(1, 1, 1)

        def unexpected(*args, **kwargs):
            pytest.fail("invalid layout reached stage1")

        monkeypatch.setattr(aiter, "mla_decode_stage1_asm_fwd", unexpected)
        with pytest.raises(ValueError, match="kv_csr_layout"):
            _run_qh64_case(case, kv_csr_layout=mode)

    @pytest.mark.parametrize(
        "variant", ["token_major", "qh128", "page_size", "persistent", "cp"]
    )
    def test_unsupported_scope_rejected(self, variant, monkeypatch):
        case = _make_qh64_case(1, 1, 1)
        overrides = {
            "token_major": {"kv_layout": "token_major"},
            "qh128": {"q": torch.empty((1, 128, 576), device="cuda")},
            "page_size": {"page_size": 1},
            "persistent": {"work_meta_data": torch.empty(1, device="cuda")},
            "cp": {"cp_world_size": 2},
        }

        def unexpected(*args, **kwargs):
            pytest.fail("unsupported token CSR reached stage1")

        monkeypatch.setattr(aiter, "mla_decode_stage1_asm_fwd", unexpected)
        with pytest.raises(
            ValueError, match="token CSR|page_size=64|persistent|context parallelism"
        ):
            _run_qh64_case(case, kv_csr_layout="token", **overrides[variant])

    @pytest.mark.parametrize(
        "variant", ["token_major", "page_size", "persistent", "cp"]
    )
    def test_native_token_scope_rejected(self, variant, monkeypatch):
        case = _make_qh64_case(1, 1, 1)
        original = aiter.mla_decode_stage1_asm_fwd

        def invalid_native(*args, **kwargs):
            args = list(args)
            kwargs["kv_indptr_is_token_level"] = True
            if variant == "token_major":
                kwargs["kv_shuffled"] = 0
            elif variant == "page_size":
                args[11] = 1
            elif variant == "persistent":
                args[7] = torch.empty(1, device="cuda")
            else:
                args[21] = 2
            return original(*args, **kwargs)

        monkeypatch.setattr(aiter, "mla_decode_stage1_asm_fwd", invalid_native)
        with pytest.raises(RuntimeError, match="token CSR|context parallelism"):
            _run_qh64_case(case, kv_csr_layout="token")

    @pytest.mark.parametrize("splits", [1, 3])
    @pytest.mark.parametrize(
        "offset",
        [
            (1 << 32) // (64 * 576) - 1,
            (1 << 32) // (64 * 576) + 1,
            (1 << 19) + 1,
        ],
    )
    def test_bigraw_token_hot_loop(self, offset, splits):
        # All 64 logical pages come from token slots; both bootstrap and rolling
        # page loads must retain the high half of physical byte offsets.
        _check_qh64_bigraw(offset, splits, 4096, q_len=4, kv_csr_layout="token")

    @pytest.mark.parametrize(
        "ctx_len,splits",
        [(0, 3), (1, 1), (63, 3), (64, 1), (65, 3), (129, 3), (4096, 3)],
    )
    def test_oob_guarded_token_slots(self, ctx_len, splits):
        code = (
            "import sys; sys.path.insert(0, 'op_tests'); "
            "from test_mla_decode_pagesize64 import _run_qh64_token_oob_worker; "
            f"_run_qh64_token_oob_worker({ctx_len}, {splits})"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            cwd=Path(__file__).resolve().parents[1],
            env=dict(
                os.environ, PYTORCH_NO_HIP_MEMORY_CACHING="1", HSA_ENABLE_COREDUMP="0"
            ),
            text=True,
            capture_output=True,
            check=False,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def _run_qh64_token_oob_worker(ctx_len, splits):
    lengths = [1, ctx_len, 0] if ctx_len else [0, 0, 0]
    case = _guard_qh64_case(
        _make_qh64_case(3, ctx_len, splits, lengths=lengths, q_lengths=[4, 2, 1])
    )
    token_csr = tuple(_tail_guarded(t) for t in _make_qh64_token_csr(case))
    original = aiter.mla_decode_stage1_asm_fwd

    def guarded(*args, **kwargs):
        args = list(args)
        replacements = {}
        for index in (14, 15, 16, 17, 23):
            if args[index] is not None:
                replacements[index] = args[index]
                args[index] = _tail_guarded(args[index])
        original(*args, **kwargs)
        torch.cuda.synchronize()
        for index, target in replacements.items():
            target.copy_(args[index])

    aiter.mla_decode_stage1_asm_fwd = guarded
    _check_qh64_token_case(case, token_csr=token_csr)
    torch.cuda.synchronize()


def _run_qh64_oob_worker(ctx_len, splits, q_len=1):
    case = _guard_qh64_case(_make_qh64_case(3, ctx_len, splits, q_len=q_len))
    # Guard every raw stage1 output as well as all its caller-owned input buffers.
    original = aiter.mla_decode_stage1_asm_fwd

    def guarded(*args, **kwargs):
        args = list(args)
        replacements = {}
        for index in (14, 15, 16, 17, 23):
            if args[index] is not None:
                replacements[index] = args[index]
                args[index] = _tail_guarded(args[index])
        original(*args, **kwargs)
        torch.cuda.synchronize()
        for index, target in replacements.items():
            target.copy_(args[index])

    aiter.mla_decode_stage1_asm_fwd = guarded
    _check_qh64_case(case)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
