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
import os
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
    """Pack FP8 page64 KV into the kv-shuffled layout consumed by the QH128 CO."""
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

    last_page_len = ctx_lens % page_size or page_size
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
        weights = torch.softmax(logits, dim=-1)
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
        kv_layout = "kv_shuffled" if nhead == 128 else "token_major"
    kv_shuffled = kv_layout == "kv_shuffled"
    if kv_shuffled and (decode_qlen != 1 or split_kv != 1):
        raise ValueError(
            "Shuffled PS64 kernel requires nhead=128, decode_qlen=1, split_kv=1"
        )
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    nhead_kv = 1
    v_head_dim = kv_lora_rank
    page_indices_oob = 4

    kv_max_sz = 65536 * 32  # Remaining framework KV capacity after weights.
    num_page = (
        batch * ((ctx_len + page_size - 1) // page_size)
        if kv_shuffled
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
    if kv_shuffled:
        page_table = kv_indices_mi400[: batch * case["num_pages_per_batch"]].view(
            batch, case["num_pages_per_batch"]
        )
        seq_lens = torch.full((batch,), ctx_len, dtype=torch.int32)

    def run_mla_decode(out_tensor):
        return aiter.mla.mla_decode_fwd(
            q_mi400,
            kv_buffer_mi400,
            out_tensor,
            None if kv_shuffled else qo_indptr,
            None if kv_shuffled else case["kv_indptr"],
            None if kv_shuffled else kv_indices_mi400,
            None if kv_shuffled else case["kv_last_page_lens"],
            decode_qlen,
            case["page_size"],
            nhead_kv,
            1.0 / (qk_head_dim**0.5),
            num_kv_splits=case["num_kv_splits"],
            num_kv_splits_indptr=None if kv_shuffled else case["num_kv_splits_indptr"],
            q_scale=case["q_scale"],
            kv_scale=case["kv_scale"],
            return_lse=True,
            kv_layout=kv_layout,
            seq_lens=seq_lens,
            page_table=page_table,
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
        choices=_MI400_DISPATCH_CASES,
        nargs="*",
        default=None,
        help="""Public MI400 dispatch case as GQA,decode_qlen.
        QH128 defaults to kv-shuffled with one KV split.
        e.g.: -n 8,3 128,1""",
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
        help="""Attention mask selector: 0 disables causal/tail mask, 1 enables it.
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
        help="auto selects shuffled for QH128; token_major uses legacy segmented PS64",
    )
    parser.add_argument(
        "--kv-shuffled",
        action="store_true",
        help="Select QH128 cases; kv-shuffled is automatic for -n 128,1.",
    )
    args = parser.parse_args()
    if args.kv_shuffled:
        if args.kv_layout == "token_major":
            parser.error("--kv-shuffled conflicts with --kv-layout token_major")
        args.kv_layout = "kv_shuffled"
    if args.nhead is None:
        args.nhead = (
            [(128, 1)] if args.kv_layout == "kv_shuffled" else _MI400_DISPATCH_CASES
        )
    if args.kv_layout == "kv_shuffled" and any(case != (128, 1) for case in args.nhead):
        parser.error("kv_shuffled requires -n 128,1")
    if (
        (128, 1) in args.nhead
        and args.kv_layout != "token_major"
        and args.split_kv is not None
        and any(splits != 1 for splits in args.split_kv)
    ):
        parser.error("kv_shuffled requires --split-kv 1")
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
            "q": args["q"][:, :64],
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


if __name__ == "__main__":
    main()
