# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import os
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

import aiter
import aiter.mla
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.attention import mla_decode_stage1_asm_fwd
from aiter.test_common import benchmark, checkAllclose, run_perftest

# In lean containers, aiter.__init__ can skip bulk op exports when optional
# dependencies are unavailable. Register the op the mi400 sweep needs explicitly.
aiter.mla_decode_stage1_asm_fwd = mla_decode_stage1_asm_fwd

torch.set_default_device("cuda")
torch.set_printoptions(sci_mode=False)

# current supported case in decode MLA: mtp == 0, 1, 2, 3 (decode_qlen = 1, 2, 3, 4)
# qdtype bf16, kdtype bf16: nhead16, nhead128
# qdtype fp8, kdtype fp8: nhead16, nhead128


def check_support(dtype, kv_dtype, nhead):
    if dtype == dtypes.fp8 and kv_dtype == dtypes.bf16:
        return False
    return True


def cal_diff(
    x: torch.Tensor, y: torch.Tensor, name: str, use_fp8: bool = False
) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    # print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    if use_fp8:
        assert cos_diff < 3e-2
    else:
        assert cos_diff < 1e-5


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    dtype,
    is_causal=True,
) -> torch.Tensor:
    attn_weights = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if is_causal:
        s_q = query.shape[0]
        s_k = key.shape[0]
        attn_bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        temp_mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)
        attn_weights += attn_bias
    lse = attn_weights.logsumexp(dim=-1)
    attn_weights = torch.softmax(attn_weights, dim=-1)

    out = torch.einsum("hqk,khd->qhd", attn_weights.float(), value.float())
    return out.to(dtype), lse


def torch_mha_extend(
    q,  # [total_q, nheads, headdim_q]
    k,  # [num_page * page_size, nhead_kv, qk_head_dim]
    v,  # [num_page * page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    dtype,
):
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    ks = torch.tensor_split(k, kv_indptr.tolist()[1:])
    vs = torch.tensor_split(v, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    for i in range(bs):
        q = qs[i]
        k = ks[i]
        v = vs[i]
        o, _ = ref_masked_attention(q, k, v, sm_scale, dtype)
        os.append(o)
    o = torch.concat(os)
    return o


def torch_mla_extend(
    q,  # [total_q, nheads, headdim_q]
    kvc_cache,  # [num_page * page_size, nhead_kv, qk_head_dim]
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    kv_lora_rank,
    qk_rope_head_dim,
    dtype,
    is_causal=True,
):
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    bs = qo_indptr.shape[0] - 1

    os = []
    lses = []
    for i in range(bs):
        kvc = kvs[i]
        q = qs[i]
        k = kvc
        v, _ = torch.split(kvc, [kv_lora_rank, qk_rope_head_dim], dim=-1)
        o, lse = ref_masked_attention(q, k, v, sm_scale, dtype, is_causal=is_causal)
        os.append(o)
        lses.append(lse)
    o = torch.concat(os)
    lse = torch.concat(lses, dim=1).transpose(0, 1)
    return o, lse


# ###########################################################################
# gfx1250 / mi400 MLA decode
#
# Merged into the standard test_mla driver: when --mi400 is active, the driver
# overrides its sweep dims to the mi400 combos and test_mla() routes each
# (nhead=Gqa, decode_qlen, batch, ctx_len) combo through the mi400 fp8 decode
# check below. Unsupported (Gqa, decode_qlen) combos are skipped.
# Exercises the shader variants registered in
# hsa/gfx1250/mla/mla_asm.csv. Active only when get_gfx() == "gfx1250".
# ###########################################################################


@dataclass(frozen=True)
class MlaMi400KernelVariant:
    name: str
    nhead: int
    decode_qlen: int


_MI400_KERNEL_VARIANTS = [
    MlaMi400KernelVariant(name="qh16-q1-16mx1-32nx4-np-3p", nhead=16, decode_qlen=1),
    MlaMi400KernelVariant(
        name="qh16-q2-16mx2-32nx4-np-3p",
        nhead=16,
        decode_qlen=2,
    ),
    MlaMi400KernelVariant(
        name="qh32-q1-32mx1-32nx4-np-3p",
        nhead=32,
        decode_qlen=1,
    ),
    MlaMi400KernelVariant(name="qh16-q4-16mx4-64nx1-np", nhead=16, decode_qlen=4),
    MlaMi400KernelVariant(
        name="qh64-q1-16mx4-64nx1-np",
        nhead=64,
        decode_qlen=1,
    ),
    MlaMi400KernelVariant(
        name="qh128-q1-16mx4-64nx1-np",
        nhead=128,
        decode_qlen=1,
    ),
]

# Dispatch key (nhead, decode_qlen) -> variant. Source of truth for which
# (nhead, decode_qlen) combos the mi400 decode check supports.
_MI400_VARIANT_BY_KEY = {(v.nhead, v.decode_qlen): v for v in _MI400_KERNEL_VARIANTS}

# mi400 driver sweep dims (applied as arg overrides when --mi400 is active).
_MI400_NHEAD = [(v.nhead, v.decode_qlen) for v in _MI400_KERNEL_VARIANTS]
_MI400_CTX_LENS = [1,2,3,4,5,6,7,11,13,17, 1024 + 1, 1536 + 3, 8192 + 17, 8704 + 19, 16384]
_MI400_BATCH_SIZES = [1, 64, 256]
_MI400_SPLIT_PER_BATCH = [1, 2]

# Exact poc_kl parity sweep: when --poc_kl on, the mi400 driver tests ONLY the
# cases defined in poc_kl/mi400/mla/run.sh (one tuple per run.sh test), instead
# of the cartesian sweep above. Each tuple is the aiter equivalent of a run.sh
# `test_kl_mla_*` case:
#   (nhead, decode_qlen, batch, ctx_lens, split_per_batch, mask)
# Mapping from run.sh args: gqa_ratio->nhead, kernel 16mx{N}->decode_qlen,
# batch->batch, kv_seq_lens->ctx_lens, passes->split_per_batch.
# block_size=64 / fp8 are fixed in the mi400 path. Q layout is controlled
# by _Q_PATTERN below.
_POC_KL_CASES = [
    (16, 1, 1, 1024, 1, 0),  # test_kl_mla_a8w8_qh16_1tg_16mx1_32nx4_np_3p_test
    (16, 2, 2, 1024, 2, 1),  # test_kl_mla_a8w8_qh16_1tg_16mx2_32nx4_np_3p_test
    (32, 1, 1, 2048, 2, 0),  # test_kl_mla_a8w8_qh32_1tg_32mx1_32nx4_np_3p_test
    (64, 1, 4, 962, 2, 0),  # test_kl_mla_a8w8_qh64_1tg_16mx4_64nx1_np_test
    (128, 1, 4, 962, 2, 0),  # gfx1250 gqa128 uses qh64 16mx4 kernel with doubled grid-z
    (16, 4, 4, 578, 2, 1),  # test_kl_mla_a8w8_qh16_1tg_16mx4_64nx1_np_test
]
# Distinct (nhead, decode_qlen) variant keys present in _POC_KL_CASES, order
# preserved. Used as args.nhead grouping when --poc_kl on.
_POC_KL_NHEAD = list(
    dict.fromkeys((n, dq) for (n, dq, _b, _c, _s, _m) in _POC_KL_CASES)
)

# Set True at runtime when --poc_kl on; selects the exact run.sh case sweep.
_RUN_POC_KL = False

# Q layout selector for mi400:
#   0: nhead*(nope+rope) original contiguous per-head layout
#   1: [..., nhead*nope] + [..., nhead*rope] (current rope_split2 layout)
#   3: nhead*(nope+rope) per-head layout with a 768-byte padded row stride
_Q_PATTERN = 3


def _pack_rope_split2_q_pages(tensor, nope_dim, rope_dim):
    shape = tensor.shape
    assert shape[-1] == nope_dim + rope_dim
    # [..., nhead, nope+rope] -> [..., nhead*nope] + [..., nhead*rope]
    # -> [..., nhead, nope+rope], with nope/rope groups contiguous across heads.
    packed = torch.cat(
        (
            tensor[..., :nope_dim].reshape(*shape[:-2], shape[-2] * nope_dim),
            tensor[..., nope_dim:].reshape(*shape[:-2], shape[-2] * rope_dim),
        ),
        dim=-1,
    )
    return packed.reshape(shape).contiguous()


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
    use_non_unit_scales=True,
):
    repo_hsa_dir = Path(__file__).resolve().parents[1] / "hsa"
    os.environ["AITER_ASM_DIR"] = str(repo_hsa_dir)

    device = torch.device("cuda")
    assert num_kv_splits > 0
    torch.manual_seed(
        20260513
        + batch * 1009
        + ctx_lens
        + nhead * 7
        + decode_qlen
        + num_kv_splits * 101
    )

    page_size = 64
    num_pages_per_batch = (ctx_lens + page_size - 1) // page_size

    last_page_len = ctx_lens % page_size or page_size
    kv_last_page_lens = torch.full(
        (batch,), last_page_len, dtype=torch.int32, device=device
    )
    num_kv_splits_indptr = (
        torch.arange(batch + 1, dtype=torch.int32, device=device) * num_kv_splits
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
    shuffle_pages=True,
):
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
        kv_buffer_logical_bf16[available_pages:] = torch.randn(
            (total_pages - available_pages, page_size, nhead_kv, qk_head_dim),
            dtype=kv_buffer_source_bf16.dtype,
            device=kv_buffer_source_bf16.device,
        )
    # The kernel consumes a compact block table, with OOB padding only after all
    # valid pages. KV pages are scattered into their physical page ids.
    shuffled_page_indices = _make_page_permutation(total_pages, shuffle=shuffle_pages)
    kv_buffer_scattered_bf16 = torch.empty_like(kv_buffer_logical_bf16)
    kv_indices = torch.zeros(total_page_indices, dtype=torch.int32, device=device)
    for logical_page, physical_page in enumerate(shuffled_page_indices):
        kv_buffer_scattered_bf16[physical_page] = kv_buffer_logical_bf16[logical_page]
        kv_indices[logical_page] = physical_page

    kv_buffer_ref = kv_buffer_scattered_bf16.to(dtypes.fp8)
    kv_buffer = _pack_rope_split2_kv_pages(
        kv_buffer_ref.view(total_pages, page_size, nhead_kv, qk_head_dim),
        v_head_dim,
        qk_head_dim - v_head_dim,
    )
    return kv_buffer, kv_buffer_ref, kv_indices


def _make_mla_mi400_q_case(
    *, q_fp8, batch, decode_qlen, nhead, qk_head_dim, v_head_dim
):
    q = q_fp8.view(batch, decode_qlen, nhead, qk_head_dim)
    if _Q_PATTERN == 1:
        q = _pack_rope_split2_q_pages(
            q,
            v_head_dim,
            qk_head_dim - v_head_dim,
        )
    elif _Q_PATTERN == 3:
        q = _pack_rope_split3_q_pages(
            q,
            v_head_dim,
            qk_head_dim - v_head_dim,
        )
    elif _Q_PATTERN != 0:
        raise ValueError(f"_Q_PATTERN must be 0, 1, or 3, got {_Q_PATTERN}")
    if _Q_PATTERN == 3:
        return torch.as_strided(
            q,
            size=(batch * decode_qlen, nhead, qk_head_dim),
            stride=(nhead * q.stride(2), q.stride(2), q.stride(3)),
        )
    return q.contiguous().view(batch * decode_qlen, nhead, qk_head_dim)


def _apply_causal_mask_(logits):
    # Matches the causal/tail mask shape used by ref_masked_attention().
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
    ctx_lens,
    batch_size,
    nhead,
    kv_lora_rank,
    qk_nope_head_dim,
    qk_rope_head_dim,
    v_head_dim,
    dtype,
    kvtype,
    page_size,
    varlen,
    decode_qlen,
    split_per_batch=None,
    return_lse=False,
    sequential_page_indices=False,
    mi400=False,
    mask=1,
):
    ret = {}

    kv_max_sz = (
        65536 * 32
    )  # calculated by rest of mem after weight loaded in frameworks
    num_page = (kv_max_sz + page_size - 1) // page_size

    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int)
    seq_lens_qo = torch.empty(batch_size, dtype=torch.int)
    seq_lens_kv = torch.empty(batch_size, dtype=torch.int)
    kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
    if varlen:
        for i in range(batch_size):
            seq_lens_kv[i] = max(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens)
            seq_lens_qo[i] = max(
                min(random.normalvariate(ctx_lens, ctx_lens / 2), ctx_lens), 1
            )
    else:
        seq_lens_kv.fill_(ctx_lens)
        seq_lens_qo.fill_(ctx_lens)
    kv_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_kv, dim=0)
    if sequential_page_indices:
        # page_id == logical token index; needs pool >= ctx and byte offset can exceed 2^32
        num_page = max(num_page, kv_indptr[-1].item() + 10000)
    n_kv_idx = kv_indptr[-1].item() + 10000
    if sequential_page_indices:
        kv_indices = torch.arange(n_kv_idx, dtype=torch.int)
    else:
        kv_indices = torch.randint(0, num_page, (n_kv_idx,), dtype=torch.int)
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    max_seqlen_qo = seq_lens_qo.max().item()
    max_seqlen_kv = seq_lens_kv.max().item()
    total_qo = qo_indptr[-1].item()
    total_kv = kv_indptr[-1].item()
    kv_buffer = torch.randn(
        (num_page * page_size, 1, kv_lora_rank + qk_rope_head_dim),
        dtype=torch.bfloat16,
    )

    # for none absorb (mha)
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    sm_scale = 1.0 / (qk_head_dim**0.5)

    # ############################## normal: prefill
    def test_normal_prefill():
        q = torch.randn((total_qo, nhead, qk_head_dim), dtype=torch.bfloat16)
        k = torch.randn((total_kv, nhead, qk_head_dim), dtype=torch.bfloat16)
        v = torch.randn((total_kv, nhead, v_head_dim), dtype=torch.bfloat16)

        out_ref = torch_mha_extend(
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            dtype=dtype,
        )

        out_aiter, us_aiter = run_perftest(
            aiter.flash_attn_varlen_func,
            q,
            k,
            v,
            qo_indptr,
            kv_indptr,
            max_seqlen_qo,
            max_seqlen_kv,
            softmax_scale=sm_scale,
            causal=True,
        )

        flop = (
            batch_size
            * nhead
            * 2
            * (ctx_lens * qk_head_dim * ctx_lens + ctx_lens * ctx_lens * v_head_dim)
        )
        checkAllclose(
            out_ref.to(torch.float),
            out_aiter.to(torch.float),
            msg=f"mla_prefill-normal    [torch vs  aiter_ck]: {us_aiter:>8.2f} us...... {flop/us_aiter/1000/1000:>8.2f} TFlops",
        )
        return us_aiter

    out_dtype = torch.bfloat16

    us_aiter = None
    # Prefill ref builds [nhead, (batch*ctx)^2] fp32 attn weights; bound both
    # the lazy "tile area" gate and the per-call ctx so decode-scale ctx_lens
    # (1M+) never trigger the O(N^2) ref.
    if (
        (dtype == torch.bfloat16 and kvtype == torch.bfloat16)
        and batch_size * ctx_lens * nhead < 256 * 8192 * 16
        and ctx_lens <= 16384
    ):
        us_aiter = test_normal_prefill()
        ret["prefill:ck_192"] = us_aiter

    torch.cuda.empty_cache()
    # absorb init
    qk_head_dim = kv_lora_rank + qk_rope_head_dim
    nhead_kv = 1
    v_head_dim = kv_lora_rank
    sm_scale = 1.0 / (qk_head_dim**0.5)

    # test prefill
    # ############################## absorb: prefill
    def test_absorb_prefill():
        q = torch.randn((total_qo, nhead, qk_head_dim), dtype=torch.bfloat16)

        out_ref, _ = torch_mla_extend(
            q,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            dtype=out_dtype,
        )

        # #triton version
        # prefix_indptr = kv_indptr - qo_indptr
        # tmp = kv_indptr[1:] - seq_lens_qo
        # tmp_inpptr, _ = torch.concat([kv_indptr[1:], tmp]).sort()
        # prefix_kv_indices = kv_indices.tensor_split(tmp_inpptr.tolist())
        # extend_kv_indices = torch.concat(
        #     [el for i, el in enumerate(prefix_kv_indices) if i % 2 == 1]
        # )
        # prefix_kv_indices = torch.concat(
        #     [el for i, el in enumerate(prefix_kv_indices) if i % 2 == 0]
        # )
        # extend_kvc = torch.index_select(kv_buffer, 0, extend_kv_indices)
        # out_triton = torch.empty((total_qo, nhead, v_head_dim), dtype=dtype).fill_(-1)
        # _, us_triton = run_perftest(
        #     mla_extend_ref.extend_attention_fwd,
        #     q,
        #     extend_kvc,
        #     extend_kvc[..., :kv_lora_rank],
        #     out_triton,
        #     kv_buffer,
        #     kv_buffer[..., :kv_lora_rank],
        #     qo_indptr,
        #     prefix_indptr,
        #     prefix_kv_indices,
        #     None,
        #     None,
        #     max_seqlen_qo,
        #     sm_scale,
        #     num_iters=5,
        # )
        # checkAllclose(
        #     out_ref,
        #     out_triton,
        #     msg=f"mla_prefill-absorb    [torch vs    triton]:{us_torch:>8.2f} us vs {us_triton:>8.2f} us......",
        # )

        out_asm = torch.empty((total_qo, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
        (attn_logits, attn_lse), us_asm = run_perftest(
            aiter.mla.mla_prefill_fwd,
            q,
            kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            sm_scale,
        )

        checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_prefill-absorb    [torch vs aiter_asm]: {us_asm:>8.2f} us......",
        )
        return us_asm

    us_asm = None
    # Absorb-prefill ref (mla_torch) builds [nhead, (batch*ctx_kv)^2] fp32 attn
    # weights -- O(N^2) memory. Tile-area gate alone is not enough: bh16 CI
    # sweeps run with decode-scale ctx_lens (-c 49152, -c 98304, -c 10000000)
    # and would OOM the host. Mirror the normal-prefill gate's explicit
    # ctx_lens <= 16384 cap to skip the ref for those configs.
    if (
        (dtype == torch.bfloat16 and kvtype == torch.bfloat16 and nhead in [16, 128])
        and batch_size * ctx_lens * nhead < 32 * 8192 * 16
        and ctx_lens <= 16384
    ):
        us_asm = test_absorb_prefill()
        ret["prefill:asm_576"] = us_asm

    torch.cuda.empty_cache()

    # ############################## absorb: decode
    # seq_lens_qo = torch.randint(1, 5, (batch_size,), dtype=torch.int)
    # if nhead == 16 and decode_qlen != 1:
    #     return
    seq_lens_qo.fill_(decode_qlen)

    max_seqlen_qo = seq_lens_qo.max().item()
    qo_indptr[1 : batch_size + 1] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = qo_indptr[-1].item()
    q = torch.randn((total_q, nhead, qk_head_dim), dtype=torch.bfloat16)

    # troch implementation. mi400 uses its own _ref_mla_mi400 golden (built on
    # fp8-dequantized, page-gathered inputs), so skip the standard bf16 ref.
    if not mi400:
        out_ref, lse_ref = torch_mla_extend(
            q,
            kv_buffer,
            qo_indptr,
            kv_indptr,
            kv_indices,
            sm_scale,
            kv_lora_rank,
            qk_rope_head_dim,
            is_causal=True,
            dtype=out_dtype,
        )

    # Triton implementation
    # if decode_qlen == 1:
    #     if qk_head_dim != v_head_dim:
    #         out_triton = q.new_empty((total_q, nhead, v_head_dim)).fill_(-1)
    #     else:
    #         out_triton = torch.empty_like(q)

    #     num_kv_splits = 16
    #     attn_logits = torch.empty(
    #         (total_q, nhead, num_kv_splits, v_head_dim + 1),
    #         dtype=dtypes.fp32,
    #     )
    #     _, us_ref = run_perftest(
    #         mla_decode_ref.decode_attention_fwd,
    #         q,
    #         kv_buffer,
    #         kv_buffer[..., :kv_lora_rank],
    #         out_triton,
    #         kv_indptr,
    #         kv_indices,
    #         attn_logits,
    #         num_kv_splits,
    #         sm_scale,
    #         num_iters=5,
    #     )
    #     # logits_ref, lse_ref = attn_logits.split([v_head_dim, 1], dim=-1)
    #     # logits_ref = rearrange(logits_ref, "bs h sp d -> bs sp h d")
    #     # lse_ref = rearrange(lse_ref, "bs h sp d -> bs sp h d")
    #     checkAllclose(
    #         out_ref,
    #         out_triton,
    #         msg=f"mla_decode-absorb    [golden vs    triton]:{us_torch_decode:>8.2f} us vs {us_ref:>8.2f} us......",
    #     )

    def test_absorb_decode_bf16():
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q,
            kv_buffer.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            page_size,
            nhead_kv,
            sm_scale,
            num_kv_splits=split_per_batch,
            return_lse=return_lse,
        )

        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )
        if return_lse and attn_lse is not None:
            checkAllclose(
                lse_ref,
                attn_lse.reshape(total_q, nhead),
                msg=f"mla_decode-absorb    [lse_ref vs attn_lse]: {us_asm_decode:>8.2f} us......",
            )
        return err, us_asm_decode

    def test_absorb_decode_fp8():
        if dtype != dtypes.fp8 and nhead == 128:
            aiter.logger.info("don't support this case:\n")
            return None, 1e12
        kv_last_page_lens = torch.ones(batch_size, dtype=torch.int)
        out_asm = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)

        q_fp8 = q.to(dtype)
        q_scale = None
        if dtype == dtypes.fp8:
            q_scale = torch.ones([1], dtype=torch.float, device="cuda")
        else:
            aiter.logger.info("don't support this case.")
            return None, 1e12

        kv_buffer_fp8 = kv_buffer.to(kvtype)
        kv_scale = torch.ones([1], dtype=torch.float, device="cuda")

        (attn_logits, attn_lse), us_asm_decode = run_perftest(
            aiter.mla.mla_decode_fwd,
            q_fp8 if dtype == dtypes.fp8 else q,
            kv_buffer_fp8.view(num_page, page_size, nhead_kv, qk_head_dim),
            out_asm,
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            max_seqlen_qo,
            page_size,
            nhead_kv,
            sm_scale,
            q_scale=q_scale,
            kv_scale=kv_scale,
            num_kv_splits=split_per_batch,
        )

        # print(f"{out_ref.view(total_q, -1)=}")
        # print(f"{out_asm.view(total_q, -1)=}")
        # checkAllclose(logits_ref, attn_logits,
        #               msg=f'attn_logits [golden vs aiter_asm]')
        # checkAllclose(lse_ref, attn_lse, msg="attn_lse    [golden vs aiter_asm]")
        err = checkAllclose(
            out_ref,
            out_asm,
            msg=f"mla_decode-absorb_fp8    [golden vs aiter_asm]: {us_asm_decode:>8.2f} us......",
        )

        cal_diff(out_ref, out_asm, "out", True)
        return err, us_asm_decode

    def test_absorb_decode_gluon():
        from aiter.ops.triton.gluon.mla_decode_gluon import mla_decode_gluon

        out_gluon = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)

        q_nope = q[:, :, :v_head_dim].view(batch_size, nhead, v_head_dim)
        q_pe = q[:, :, v_head_dim:].view(batch_size, nhead, qk_head_dim - v_head_dim)

        # KV: flat [N, 576] buffer; the kernel uses KV_PE_OFFSET (default 512)
        # to reach k_pe columns and picks buffer_load vs global_load internally.
        kv_c = kv_buffer.view(-1, qk_head_dim)

        # Varlen=False: reshape kv_indices as block_table [batch, ctx_lens]
        # Varlen=True : pass kv_indices + kv_indptr
        if not varlen:
            page_table = kv_indices[:total_kv].view(batch_size, ctx_lens)
            seq_info = seq_lens_kv
            use_2d_view = True
        else:
            page_table = kv_indices
            seq_info = kv_indptr
            use_2d_view = False

        (attn_logits, attn_lse), us_gluon_decode = run_perftest(
            mla_decode_gluon,
            q_nope,
            q_pe,
            kv_c,
            out_gluon.view(batch_size, nhead, v_head_dim),
            page_table,
            seq_info,
            sm_scale,
            use_2d_view=use_2d_view,
            min_kv_seq_len=ctx_lens,
            return_lse=return_lse,
        )

        err = checkAllclose(
            out_ref,
            out_gluon,
            msg=f"mla_decode-absorb    [golden vs gluon_mla]: {us_gluon_decode:>8.2f} us......",
        )
        if return_lse and attn_lse is not None:
            checkAllclose(
                lse_ref,
                attn_lse.reshape(total_q, nhead),
                msg=f"mla_decode-absorb    [lse_ref vs gluon_mla_lse]: {us_gluon_decode:>8.2f} us......",
            )
        return err, us_gluon_decode

    def test_absorb_decode_gluon_bh16(name):
        # Shared bh16bn{64,128} runner. The wrapper dispatches on
        # (nhead, kv dtype): name='bh16bn128' -> cast kv to fp8;
        # name='bh16bn64' -> keep bf16. -lse also validates the returned lse.
        from aiter.ops.triton.gluon.mla_decode_gluon import mla_decode_gluon

        out_gluon = torch.empty((total_q, nhead, v_head_dim), dtype=out_dtype).fill_(-1)
        q_nope = q[:, :, :v_head_dim].view(batch_size, nhead, v_head_dim)
        q_pe = q[:, :, v_head_dim:].view(batch_size, nhead, qk_head_dim - v_head_dim)

        kv_c = kv_buffer.view(-1, qk_head_dim)
        if name == "bh16bn128":
            kv_c = kv_c.to(dtypes.fp8)

        if not varlen:
            page_table = kv_indices[:total_kv].view(batch_size, ctx_lens)
            seq_info = seq_lens_kv
            use_2d_view = True
        else:
            page_table = kv_indices
            seq_info = kv_indptr
            use_2d_view = False

        (_, lse), us_decode = run_perftest(
            mla_decode_gluon,
            q_nope,
            q_pe,
            kv_c,
            out_gluon.view(batch_size, nhead, v_head_dim),
            page_table,
            seq_info,
            sm_scale,
            use_2d_view=use_2d_view,
            kv_scale=1.0,
            min_kv_seq_len=ctx_lens,
            return_lse=return_lse,
        )

        err = checkAllclose(
            out_ref,
            out_gluon,
            msg=f"mla_decode-absorb    [golden vs gluon_{name}]: {us_decode:>8.2f} us......",
        )
        cal_diff(out_ref, out_gluon, f"out_gluon_{name}", use_fp8=(name == "bh16bn128"))
        if return_lse and lse is not None:
            checkAllclose(
                lse_ref,
                lse.reshape(total_q, nhead),
                msg=f"mla_decode-absorb    [lse_ref vs gluon_{name}_lse]: {us_decode:>8.2f} us......",
            )
        return err, us_decode

    def test_absorb_decode_mi400():
        # mi400 (gfx1250) fp8 MLA decode, dispatched as a decode backend peer of
        # the bf16/fp8/gluon paths. It derives fp8 + selected Q/KV layout
        # from the standard bf16 inputs and checks against _ref_mla_mi400.
        # Dispatch key is (nhead, decode_qlen); unsupported combos are recorded
        # as skipped (not failures) so the driver does not abort.
        ret["mi400:nhead"] = nhead
        ret["mi400:decode_qlen"] = decode_qlen
        ret["mi400:batch"] = batch_size
        ret["mi400:ctx"] = ctx_lens
        ret["mi400:num_kv_splits"] = split_per_batch
        ret["mi400:mask"] = mask
        ret["mi400:skipped"] = True
        ret["mi400:passed"] = None
        ret["mi400:finite"] = None
        ret["mi400:cos_diff"] = None
        ret["mi400:us"] = None
        ret["mi400:TFLOPS"] = None
        ret["mi400:TB/s"] = None

        variant = _MI400_VARIANT_BY_KEY.get((nhead, decode_qlen))
        if variant is None:
            ret["mi400:reason"] = "unsupported (nhead,decode_qlen)"
            aiter.logger.info(
                "mla_decode-mi400 [nhead=%d decode_qlen=%d]: skipped (unsupported dispatch combo)",
                nhead,
                decode_qlen,
            )
            return

        ret["mi400:variant"] = variant.name
        ret["mi400:skipped"] = False
        # Looser than the generic fp8 3e-2 tolerance: with page shuffle + OOB +
        # non-unit scales all on, short-KV / multi-batch combos (e.g. q4,
        # batch=2, ctx=65) sit just above 3e-2 from fp8 quant noise.
        cos_threshold = 5e-2
        # mi400-specific coverage knobs are fixed fully-on (page shuffle + OOB
        # padding + non-unit scales) for every supported combo.
        page_indices_oob = 4
        kv_buffer_mi400, kv_buffer_ref_mi400, kv_indices_mi400 = (
            _make_mla_mi400_kv_case(
                kv_buffer_bf16=kv_buffer,
                batch=batch_size,
                ctx_lens=ctx_lens,
                qk_head_dim=qk_head_dim,
                v_head_dim=v_head_dim,
                page_indices_oob=page_indices_oob,
            )
        )
        q_fp8_mi400 = q.to(dtypes.fp8)
        q_mi400 = _make_mla_mi400_q_case(
            q_fp8=q_fp8_mi400,
            batch=batch_size,
            decode_qlen=decode_qlen,
            nhead=nhead,
            qk_head_dim=qk_head_dim,
            v_head_dim=v_head_dim,
        )
        case = _make_mla_mi400_case(
            batch=batch_size,
            ctx_lens=ctx_lens,
            nhead=nhead,
            decode_qlen=decode_qlen,
            num_kv_splits=split_per_batch,
        )

        # Single launch for functional/numerical validation, kept separate from
        # the perf loop below so the correctness check always inspects one clean
        # launch into the freshly zeroed out buffer.
        out_mi400 = torch.zeros(
            (
                batch_size * decode_qlen,
                nhead,
                v_head_dim,
            ),
            dtype=torch.bfloat16,
        )
        attn_logits, attn_lse = aiter.mla.mla_decode_fwd(
            q_mi400,
            kv_buffer_mi400,
            out_mi400,
            qo_indptr,
            case["kv_indptr"],
            kv_indices_mi400,
            case["kv_last_page_lens"],
            decode_qlen,
            case["page_size"],
            nhead_kv,
            1.0 / (qk_head_dim**0.5),
            num_kv_splits=case["num_kv_splits"],
            num_kv_splits_indptr=case["num_kv_splits_indptr"],
            q_scale=case["q_scale"],
            kv_scale=case["kv_scale"],
            return_lse=True,
        )
        out_check = out_mi400.clone()

        out_shape = (
            batch_size * decode_qlen,
            nhead,
            v_head_dim,
        )
        logits_shape = (
            batch_size * decode_qlen,
            case["num_kv_splits"],
            nhead,
            v_head_dim,
        )
        if case["num_kv_splits"] == 1:
            logits_shape = (
                batch_size * decode_qlen,
                nhead,
                v_head_dim,
            )
        # Structural shape checks are hard asserts: they must always hold.
        assert out_check.shape == out_shape
        assert attn_logits.shape == logits_shape
        assert attn_lse.shape == (batch_size * decode_qlen, nhead)

        finite = (
            torch.isfinite(out_check.detach().float().cpu()).all().item()
            and torch.isfinite(attn_logits.detach().float().cpu()).all().item()
            and torch.isfinite(attn_lse.detach().float().cpu()).all().item()
        )
        if finite:
            expected = _ref_mla_mi400(
                case,
                q_fp8_mi400,
                kv_buffer_ref_mi400,
                kv_indices_mi400,
                batch_size,
                ctx_lens,
                decode_qlen,
                nhead_kv,
                qk_head_dim,
                v_head_dim,
                mask,
            )
            cos_diff = _cosine_diff(out_check, expected)
        else:
            cos_diff = float("inf")

        passed = finite and cos_diff < cos_threshold
        ret["mi400:finite"] = finite
        ret["mi400:cos_diff"] = cos_diff
        ret["mi400:passed"] = passed
        aiter.logger.info(
            "mla_decode-mi400 [%s | batch=%d ctx=%d splits=%d]: finite=%s cos_diff=%.3e %s",
            variant.name,
            batch_size,
            ctx_lens,
            case["num_kv_splits"],
            finite,
            cos_diff,
            "passed" if passed else "FAILED",
        )

        if _RUN_POC_KL:
            # poc_kl parity mode: one correctness launch per kernel only, no
            # perf loop (keeps mi400:us/TFLOPS/TB/s as None from init above).
            return

        # Performance: zero-initialized split/out buffers make the repeated
        # launches safe, so time the kernel over the standard perftest loop.
        # Correctness was already validated above on the single launch.
        _, us_mi400 = run_perftest(
            aiter.mla.mla_decode_fwd,
            q_mi400,
            kv_buffer_mi400,
            out_mi400,
            qo_indptr,
            case["kv_indptr"],
            kv_indices_mi400,
            case["kv_last_page_lens"],
            decode_qlen,
            case["page_size"],
            nhead_kv,
            1.0 / (qk_head_dim**0.5),
            num_kv_splits=case["num_kv_splits"],
            num_kv_splits_indptr=case["num_kv_splits_indptr"],
            q_scale=case["q_scale"],
            kv_scale=case["kv_scale"],
            return_lse=True,
        )

        total_q = batch_size * decode_qlen
        total_kv = batch_size * ctx_lens
        mi_flops = decode_qlen * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
        mi_bytes = (
            total_kv * nhead_kv * qk_head_dim * (torch.finfo(dtypes.fp8).bits // 8)
            + total_q * nhead * qk_head_dim * (torch.finfo(dtypes.fp8).bits // 8)
            + total_q * nhead * v_head_dim * (torch.finfo(torch.bfloat16).bits // 8)
        )
        ret["mi400:us"] = us_mi400
        ret["mi400:TFLOPS"] = mi_flops / us_mi400 / 1e6
        ret["mi400:TB/s"] = mi_bytes / us_mi400 / 1e6
        aiter.logger.info(
            "mla_decode-mi400 [%s | batch=%d ctx=%d]: %8.2f us  %7.2f TFLOPS  %7.2f TB/s",
            variant.name,
            batch_size,
            ctx_lens,
            us_mi400,
            ret["mi400:TFLOPS"],
            ret["mi400:TB/s"],
        )

    err = None
    us_asm_decode = 1e12
    # The ASM decode baseline aborts for these MLA configs when lse is requested
    if return_lse:
        pass
    elif mi400:
        test_absorb_decode_mi400()
    elif (dtype == torch.bfloat16 and kvtype == torch.bfloat16) and nhead in [
        16,
        32,
        64,
        128,
    ]:
        err, us_asm_decode = test_absorb_decode_bf16()
    elif kvtype == dtypes.fp8 and nhead in [8, 16, 128]:
        err, us_asm_decode = test_absorb_decode_fp8()

    # Standard decode perf/throughput bookkeeping; mi400 records its own
    # mi400:* keys inside the sub-test and skips this block.
    if not mi400:
        ret["decode:err"] = err
        ret["decode:asm_576"] = us_asm_decode

        flops = decode_qlen * total_kv * nhead * (qk_head_dim + v_head_dim) * 2
        bytes = (
            total_kv * nhead_kv * qk_head_dim * (torch.finfo(kvtype).bits // 8)
            + total_q * nhead * qk_head_dim * (torch.finfo(dtype).bits // 8)
            + total_q * nhead * v_head_dim * (torch.finfo(out_dtype).bits // 8)
        )

        ret["decode:flops"] = flops
        ret["decode:bytes"] = bytes
        ret["decode:TFLOPS"] = flops / us_asm_decode / 1e6
        ret["decode:TB/s"] = bytes / us_asm_decode / 1e6

    # Gluon MLA decode test
    # Example: -c 16384 -b 64 128 -n 64,1 128,1 -d bf16 -kvd bf16
    NUM_XCDS_GFX950 = 8
    BLOCK_H_GLUON = 64
    if (
        get_gfx() == "gfx950"
        and dtype == torch.bfloat16
        and kvtype == torch.bfloat16
        and nhead in (64, 128)
        and decode_qlen == 1
        and v_head_dim == 512
        and (qk_head_dim - v_head_dim) == 64
        and batch_size in (64, 128, 256)
        and page_size == 1
    ):
        base_grid = (
            NUM_XCDS_GFX950
            * ((nhead + BLOCK_H_GLUON - 1) // BLOCK_H_GLUON)
            * (batch_size // NUM_XCDS_GFX950)
        )
        splits_needed = max(1, (256 + base_grid - 1) // base_grid)
        # Round up to a power of two: 1 << (n - 1).bit_length() for n >= 1.
        num_kv_splits = 1 << (splits_needed - 1).bit_length()
        # PIPELINE_STAGES=3, BLOCK_N=64 → 192; mirror wrapper's bound.
        min_ctx_required = num_kv_splits * (192 + num_kv_splits)
        if ctx_lens > min_ctx_required:
            err_gluon, us_gluon_decode = test_absorb_decode_gluon()
            ret["decode:gluon_err"] = err_gluon
            ret["decode:gluon_576"] = us_gluon_decode
            ret["decode:gluon_TFLOPS"] = flops / us_gluon_decode / 1e6
            ret["decode:gluon_TB/s"] = bytes / us_gluon_decode / 1e6

    # Gluon MLA bh16bn128 decode test
    # Example: -c 10000000 -b 1 -n 16,1 -d bf16 -kvd fp8
    if (
        get_gfx() == "gfx950"
        and dtype == torch.bfloat16
        and kvtype == dtypes.fp8
        and nhead <= 16
        and decode_qlen == 1
        and batch_size == 1
        and v_head_dim == 512
        and (qk_head_dim - v_head_dim) == 64
        and page_size == 1
        and ctx_lens >= 1
    ):
        err_gluon, us_gluon_decode = test_absorb_decode_gluon_bh16("bh16bn128")
        ret["decode:gluon_err"] = err_gluon
        ret["decode:gluon_576"] = us_gluon_decode
        ret["decode:gluon_TFLOPS"] = flops / us_gluon_decode / 1e6
        ret["decode:gluon_TB/s"] = bytes / us_gluon_decode / 1e6

    # Gluon MLA bh16bn64 decode test
    # Example: -c 10000 -b 1 3 4 -n 16,1 -d bf16 -kvd bf16 [-lse]
    if (
        get_gfx() == "gfx950"
        and dtype == torch.bfloat16
        and kvtype == torch.bfloat16
        and nhead <= 16
        and decode_qlen == 1
        and v_head_dim == 512
        and (qk_head_dim - v_head_dim) == 64
        and page_size == 1
        and 1 <= batch_size <= 256
        and ctx_lens >= 1
    ):
        err_gluon, us_gluon_decode = test_absorb_decode_gluon_bh16("bh16bn64")
        ret["decode:gluon_err"] = err_gluon
        ret["decode:gluon_576"] = us_gluon_decode
        ret["decode:gluon_TFLOPS"] = flops / us_gluon_decode / 1e6
        ret["decode:gluon_TB/s"] = bytes / us_gluon_decode / 1e6

    return ret


parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-k",
    "--kv_lora_rank",
    type=int,
    default=512,
    help="""kv lora rank.
    e.g.: -k 512""",
)
parser.add_argument(
    "-qn",
    "--qk_nope_head_dim",
    type=int,
    default=128,
    help="""qk nope head dim.
    e.g.: -qn 128""",
)
parser.add_argument(
    "-qr",
    "--qk_rope_head_dim",
    type=int,
    default=64,
    help="""qk rope head dim.
    e.g.: -qr 64""",
)
parser.add_argument(
    "-vh",
    "--v_head_dim",
    type=int,
    default=128,
    help="""v head dim.
    e.g.: -vh 128""",
)
parser.add_argument(
    "-blk",
    "--block_size",
    type=int,
    default=1,
    help="""Block size.
    e.g.: -blk 1""",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=dtypes.str2Dtype,
    nargs="*",
    default="bf16,",
    choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
    metavar="{bf16, fp8}",
    help="""Data type of Q.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-kvd",
    "--kv_dtype",
    nargs="*",
    type=dtypes.str2Dtype,
    default="bf16,",
    choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
    metavar="{bf16, fp8}",
    help="""Data type of KV.
    e.g.: -kvd bf16""",
)
parser.add_argument(
    "-c",
    "--ctxLen",
    type=int,
    nargs="*",
    default=[21, 64, 256, 512, 1200, 3200, 5200, 8192],
    help="""Context length.
    e.g.: -c 21""",
)
parser.add_argument(
    "-b",
    "--batchSize",
    type=int,
    nargs="*",
    default=[1, 3, 5, 16, 32, 64, 128, 256],
    help="""Batch size.
    e.g.: -b 16""",
)
parser.add_argument(
    "-n",
    "--nhead",
    type=dtypes.str2tuple,
    choices=[
        (4, 1),
        (8, 1),
        (12, 1),
        (16, 1),
        (16, 2),
        (16, 4),
        (32, 1),
        (32, 2),
        (32, 4),
        (64, 1),
        (128, 1),
        (128, 2),
        (128, 4),
    ],
    nargs="*",
    const=None,
    default=[(16, 1), (16, 2), (16, 4), (128, 1), (128, 2)],
    help="""Number of nhead and decode_qlen.
    e.g.: -n 16,1""",
)
parser.add_argument(
    "-splits",
    "--split_per_batch",
    type=int,
    nargs="*",
    default=[None],
    help="""kv seqlens split num for per batch.
    e.g.: -ms 32""",
)
parser.add_argument(
    "--mask",
    type=int,
    nargs="+",
    choices=[0, 1],
    default=[1],
    help="""mi400 attention mask selector: 0 disables causal/tail mask, 1 enables it.
    e.g.: --mask 0 1""",
)
parser.add_argument(
    "--varlen",
    action="store_true",
    help="""variable kv seqlens per batch. Default: False.
    --varlen # True""",
)
parser.add_argument(
    "-lse",
    "--return_lse",
    action="store_true",
    help="""return lse. Default: False.
    --lse # True""",
)
parser.add_argument(
    "--sequential-page-indices",
    action="store_true",
    help="""Use kv_indices[i]=i (sequential physical page id) instead of random pages.
    Expands KV pool to cover ctx length (tests 64-bit page_idx * stride).""",
)
parser.add_argument(
    "--mi400",
    choices=["auto", "on", "off"],
    default="off",
    help="""Run the gfx1250/mi400 MLA decode sweep instead of the default sweep.
    auto (default): run mi400 sweep iff get_gfx()=="gfx1250".
    on: force the mi400 sweep. off: never run the mi400 sweep.""",
)
parser.add_argument(
    "--poc_kl",
    choices=["on", "off"],
    default="off",
    help="""Test ONLY the exact cases from poc_kl/mi400/mla/run.sh (5 cases:
    qh16 16mx1/16mx2/16mx4 + qh32 32mx1 + qh64 16mx4). Implies the mi400 path. Overrides the
    mi400 cartesian sweep. Default: off.""",
)


args = parser.parse_args()


def _detect_gfx():
    try:
        return get_gfx()
    except Exception:
        return None


_run_poc_kl = args.poc_kl == "on"
# poc_kl parity implies the mi400 fp8 decode path.
_run_mi400 = (
    _run_poc_kl
    or args.mi400 == "on"
    or (args.mi400 == "auto" and _detect_gfx() == "gfx1250")
)
_RUN_POC_KL = _run_poc_kl

if _run_mi400:
    # mi400 reuses the standard driver + test_mla(mi400=True); override the
    # sweep dims to the mi400 fp8 decode combos. nhead carries (gqa, decode_qlen);
    # unsupported combos self-skip inside the mi400 check.
    args.dtype = [dtypes.fp8]
    args.kv_dtype = [dtypes.fp8]
    args.nhead = _POC_KL_NHEAD if _run_poc_kl else _MI400_NHEAD
    args.ctxLen = _MI400_CTX_LENS
    args.batchSize = _MI400_BATCH_SIZES
    args.split_per_batch = _MI400_SPLIT_PER_BATCH
    args.block_size = 64
    args.kv_lora_rank = 512
    args.qk_rope_head_dim = 64
    args.varlen = False

mi400_failures = []
for nhead, decode_qlen in args.nhead:
    df = []
    if _run_poc_kl:
        # Exact run.sh cases for this (nhead, decode_qlen) variant only.
        _combos = [
            (ctx, batch, split, mask)
            for (n, dq, batch, ctx, split, mask) in _POC_KL_CASES
            if (n, dq) == (nhead, decode_qlen)
        ]
        _param_iter = [
            (dtype, kvtype, ctx, batch, split, mask)
            for dtype in args.dtype
            for kvtype in args.kv_dtype
            for (ctx, batch, split, mask) in _combos
        ]
    else:
        mask_choices = args.mask if _run_mi400 else [1]
        _param_iter = itertools.product(
            args.dtype,
            args.kv_dtype,
            args.ctxLen,
            args.batchSize,
            args.split_per_batch,
            mask_choices,
        )
    for dtype, kvtype, ctx_len, batch_size, split_per_batch, mask in _param_iter:
        if check_support(dtype, kvtype, nhead):
            ret = test_mla(
                ctx_len,
                batch_size,
                nhead,
                args.kv_lora_rank,
                args.qk_nope_head_dim,
                args.qk_rope_head_dim,
                args.v_head_dim,
                dtype,
                kvtype,
                args.block_size,
                varlen=args.varlen,
                decode_qlen=decode_qlen,
                split_per_batch=split_per_batch,
                return_lse=args.return_lse,
                sequential_page_indices=args.sequential_page_indices,
                mi400=_run_mi400,
                mask=mask,
            )
            df.append(ret)
            if (
                _run_mi400
                and not ret.get("mi400:skipped", True)
                and not ret.get("mi400:passed", False)
            ):
                mi400_failures.append(
                    (
                        ret.get("mi400:variant"),
                        batch_size,
                        ctx_len,
                        ret.get("mi400:cos_diff"),
                    )
                )
    df = pd.DataFrame(df)
    # df.to_csv(f"mla_nhead{nhead}decode_qlen{decode_qlen}.csv")
    df_md = df.to_markdown(index=False)
    aiter.logger.info("mla summary (markdown):\n%s", df_md)

if _run_mi400 and mi400_failures:
    raise AssertionError(f"mi400 MLA numerics failed for: {mi400_failures}")
