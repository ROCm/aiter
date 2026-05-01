# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx1250 MLA smoke and numerical tests.

This exercises the mi400 MLA shader variants registered in
`hsa/gfx1250/mla/mla_asm.csv` through aiter's loader and launcher. The shaders
come from the poc_kl mi400 MLA port and cover the supported `(Gqa, qSeqLen)`
dispatch combinations. Smoke checks load/launch and output shape; numerics
compare the output against a PyTorch reference across supported combinations.
"""

from dataclasses import dataclass
import os
from pathlib import Path

import pytest
import torch

import aiter
import aiter.mla as mla
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.attention import mla_decode_stage1_asm_fwd

# In lean containers, aiter.__init__ can skip bulk op exports when optional
# dependencies are unavailable. Register the op this test needs explicitly.
aiter.mla_decode_stage1_asm_fwd = mla_decode_stage1_asm_fwd


@dataclass(frozen=True)
class MlaMi400CaseConfig:
    name: str
    batch: int = 2
    kv_seq_len: int = 578
    shuffle_pages: bool = True
    page_indices_oob: int = 4
    use_non_unit_scales: bool = False


@dataclass(frozen=True)
class MlaMi400KernelVariant:
    name: str
    gqa_ratio: int
    q_seq_len: int


_POC_DUMP_CASE = MlaMi400CaseConfig(name="poc-kl-dump-shape")

_POC_DUMP_VARIANT = MlaMi400KernelVariant(
    name="qh16-q1-16mx1-32nx4-np-3p",
    gqa_ratio=16,
    q_seq_len=1,
)

_MI400_KERNEL_VARIANTS = [
    _POC_DUMP_VARIANT,
    MlaMi400KernelVariant(
        name="qh16-q2-16mx2-32nx4-np-3p",
        gqa_ratio=16,
        q_seq_len=2,
    ),
    MlaMi400KernelVariant(
        name="qh16-q4-16mx4-64nx1-np",
        gqa_ratio=16,
        q_seq_len=4,
    ),
    MlaMi400KernelVariant(
        name="qh64-q1-16mx4-64nx1-np",
        gqa_ratio=64,
        q_seq_len=1,
    ),
]


_MI400_CASES = [
    _POC_DUMP_CASE,
    MlaMi400CaseConfig(name="single-batch-short-kv", batch=1, kv_seq_len=65),
    MlaMi400CaseConfig(name="two-batch-two-pages-contiguous", batch=2, kv_seq_len=128, shuffle_pages=False),
    MlaMi400CaseConfig(
        name="two-batch-long-kv-scaled",
        batch=2,
        kv_seq_len=578,
        use_non_unit_scales=True,
    ),
    MlaMi400CaseConfig(name="three-batch-partial-page", batch=3, kv_seq_len=257),
]


def _numel(shape):
    n = 1
    for dim in shape:
        n *= dim
    return n


def _load_raw_fp8(path, shape, device):
    raw = torch.frombuffer(bytearray(Path(path).read_bytes()), dtype=torch.uint8).clone()
    assert raw.numel() == _numel(shape)
    return raw.view(dtypes.fp8).reshape(shape).to(device)


def _load_raw_float32(path, shape, device):
    raw = torch.frombuffer(bytearray(Path(path).read_bytes()), dtype=torch.float32).clone()
    assert raw.numel() == _numel(shape)
    return raw.reshape(shape).to(device)


def _load_raw_float32_prefix(path, shape, device):
    raw = torch.frombuffer(bytearray(Path(path).read_bytes()), dtype=torch.float32).clone()
    numel = _numel(shape)
    assert raw.numel() >= numel
    return raw[:numel].reshape(shape).to(device)


def _pack_rope_split2_pages(tensor, nope_dim, rope_dim):
    shape = tensor.shape
    assert shape[-1] == nope_dim + rope_dim
    packed = torch.cat(
        (
            tensor[..., :nope_dim].reshape(*shape[:-2], shape[-2] * nope_dim),
            tensor[..., nope_dim:].reshape(*shape[:-2], shape[-2] * rope_dim),
        ),
        dim=-1,
    )
    return packed.reshape(shape).contiguous()


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


def _require_gfx1250():
    if not torch.cuda.is_available():
        pytest.skip("requires a CUDA/HIP GPU")
    try:
        gfx = get_gfx()
    except Exception as exc:
        pytest.skip(f"cannot detect gfx arch: {exc}")
    if gfx != "gfx1250":
        pytest.skip(f"requires gfx1250, got {gfx}")


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
            torch.ones((batch,), dtype=torch.float32, device=device),
            torch.ones((batch,), dtype=torch.float32, device=device),
        )
    q_scale = torch.linspace(0.75, 1.25, batch, dtype=torch.float32, device=device)
    kv_scale = torch.linspace(1.20, 0.80, batch, dtype=torch.float32, device=device)
    return q_scale, kv_scale


def _make_mla_mi400_case(monkeypatch, config, variant):
    repo_hsa_dir = Path(__file__).resolve().parents[1] / "hsa"
    monkeypatch.setenv("AITER_ASM_DIR", str(repo_hsa_dir))

    device = torch.device("cuda")
    torch.manual_seed(20260513 + len(config.name))

    batch = config.batch
    q_seq_len = variant.q_seq_len
    kv_seq_len = config.kv_seq_len
    page_size = 64
    num_kv_splits = 1
    nhead = variant.gqa_ratio
    nhead_kv = 1
    qk_head_dim = 576
    v_head_dim = 512
    num_pages_per_batch = (kv_seq_len + page_size - 1) // page_size
    page_indices_oob = config.page_indices_oob
    total_page_indices = batch * (num_pages_per_batch + page_indices_oob)
    total_pages = batch * num_pages_per_batch

    q_bf16 = torch.randn(
        (batch * q_seq_len, nhead, qk_head_dim), dtype=torch.bfloat16, device=device
    )
    kv_buffer_logical_bf16 = torch.randn(
        (total_pages, page_size, nhead_kv, qk_head_dim), dtype=torch.bfloat16, device=device
    )
    # Match poc_kl mla_shuffle() output for kl_mla_mtp0_np_3p_k128_test.
    # The kernel consumes a compact block table, with OOB padding only after all
    # valid pages. KV pages are scattered into their physical page ids.
    shuffled_page_indices = _make_page_permutation(total_pages, shuffle=config.shuffle_pages)
    kv_buffer_bf16 = torch.empty_like(kv_buffer_logical_bf16)
    kv_indices = torch.zeros(total_page_indices, dtype=torch.int32, device=device)
    for logical_page, physical_page in enumerate(shuffled_page_indices):
        kv_buffer_bf16[physical_page] = kv_buffer_logical_bf16[logical_page]
        kv_indices[logical_page] = physical_page
    q_ref = q_bf16.to(dtypes.fp8)
    kv_buffer_ref = kv_buffer_bf16.to(dtypes.fp8)
    q = _pack_rope_split2_pages(
        q_ref.view(batch, q_seq_len, nhead, qk_head_dim),
        v_head_dim,
        qk_head_dim - v_head_dim,
    ).view(batch * q_seq_len, nhead, qk_head_dim)
    kv_buffer = _pack_rope_split2_kv_pages(
        kv_buffer_ref.view(total_pages, page_size, nhead_kv, qk_head_dim),
        v_head_dim,
        qk_head_dim - v_head_dim,
    )
    out = torch.empty((batch * q_seq_len, nhead, v_head_dim), dtype=torch.bfloat16, device=device)

    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * q_seq_len
    kv_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * kv_seq_len
    last_page_len = kv_seq_len % page_size or page_size
    kv_last_page_lens = torch.full(
        (batch,), last_page_len, dtype=torch.int32, device=device
    )
    num_kv_splits_indptr = torch.arange(batch + 1, dtype=torch.int32, device=device) * num_kv_splits
    q_scale, kv_scale = _make_scales(batch, device, enabled=config.use_non_unit_scales)

    poc_dump_dir = os.getenv("AITER_MLA_LOAD_POC_DUMP_DIR")
    if poc_dump_dir:
        if config != _POC_DUMP_CASE or variant != _POC_DUMP_VARIANT:
            pytest.skip("poc_kl dump replay mode only supports the canonical dump shape")
        poc_dump_dir = Path(poc_dump_dir)
        q = _load_raw_fp8(poc_dump_dir / "q.bin", (batch * q_seq_len, nhead, qk_head_dim), device)
        kv_buffer = _load_raw_fp8(
            poc_dump_dir / "kv_buffer.bin",
            (total_pages, page_size, nhead_kv, qk_head_dim),
            device,
        )
        q_scale = _load_raw_float32(poc_dump_dir / "q_scale.bin", (batch,), device)
        kv_scale = _load_raw_float32(poc_dump_dir / "kv_scale.bin", (batch,), device)
        q_ref = None
        kv_buffer_ref = None

    return {
        "q": q,
        "kv_buffer": kv_buffer,
        "q_ref": q_ref,
        "kv_buffer_ref": kv_buffer_ref,
        "out": out,
        "qo_indptr": qo_indptr,
        "kv_indptr": kv_indptr,
        "kv_indices": kv_indices,
        "kv_last_page_lens": kv_last_page_lens,
        "q_seq_len": q_seq_len,
        "kv_seq_len": kv_seq_len,
        "page_size": page_size,
        "nhead_kv": nhead_kv,
        "nhead": nhead,
        "gqa_ratio": variant.gqa_ratio,
        "qk_head_dim": qk_head_dim,
        "v_head_dim": v_head_dim,
        "num_kv_splits": num_kv_splits,
        "num_kv_splits_indptr": num_kv_splits_indptr,
        "q_scale": q_scale,
        "kv_scale": kv_scale,
        "batch": batch,
        "num_pages_per_batch": num_pages_per_batch,
        "case_name": config.name,
        "variant_name": variant.name,
    }


def _run_mla_mi400(case, *, return_lse=False):
    attn_logits, attn_lse = mla.mla_decode_fwd(
        case["q"],
        case["kv_buffer"],
        case["out"],
        case["qo_indptr"],
        case["kv_indptr"],
        case["kv_indices"],
        case["kv_last_page_lens"],
        case["q_seq_len"],
        case["page_size"],
        case["nhead_kv"],
        1.0 / (case["qk_head_dim"] ** 0.5),
        num_kv_splits=case["num_kv_splits"],
        num_kv_splits_indptr=case["num_kv_splits_indptr"],
        q_scale=case["q_scale"],
        kv_scale=case["kv_scale"],
        return_lse=return_lse,
    )
    return attn_logits, attn_lse


def _ref_mla_mi400(case):
    outputs = []
    num_pages = case["num_pages_per_batch"]
    q_source = case["q_ref"] if case.get("q_ref") is not None else case["q"]
    kv_source = (
        case["kv_buffer_ref"]
        if case.get("kv_buffer_ref") is not None
        else case["kv_buffer"]
    )
    for b in range(case["batch"]):
        q_start = b * case["q_seq_len"]
        q_end = q_start + case["q_seq_len"]
        q = q_source[q_start:q_end].float() * case["q_scale"][b]
        page_indices = case["kv_indices"][b * num_pages : (b + 1) * num_pages].long()
        kv = torch.index_select(kv_source.float(), 0, page_indices) * case["kv_scale"][b]
        kv = kv.reshape(-1, case["nhead_kv"], case["qk_head_dim"])
        kv = kv[: case["kv_seq_len"]]
        key = kv
        value = kv[..., : case["v_head_dim"]]

        logits = torch.einsum("qhd,kmd->hqk", q, key) * (1.0 / (case["qk_head_dim"] ** 0.5))
        weights = torch.softmax(logits, dim=-1)
        outputs.append(torch.einsum("hqk,kmd->qhd", weights, value).to(torch.bfloat16))
    return torch.cat(outputs, dim=0)


def _assert_cosine_close(actual, expected, *, threshold=3e-2):
    actual = actual.detach().float().cpu()
    expected = expected.detach().float().cpu()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    numerator = 2 * (actual.double() * expected.double()).sum()
    denominator = (actual.double().square() + expected.double().square()).sum().clamp_min(1e-12)
    cosine_diff = 1 - (numerator / denominator)
    assert cosine_diff.item() < threshold


@pytest.mark.parametrize("variant", _MI400_KERNEL_VARIANTS, ids=lambda variant: variant.name)
@pytest.mark.parametrize("config", _MI400_CASES, ids=lambda config: config.name)
def test_mla_mi400_minimal_smoke(monkeypatch, config, variant):
    _require_gfx1250()
    case = _make_mla_mi400_case(monkeypatch, config, variant)

    attn_logits, attn_lse = _run_mla_mi400(case)

    assert case["out"].shape == (
        case["batch"] * case["q_seq_len"],
        case["nhead"],
        case["v_head_dim"],
    )
    assert attn_logits.shape == (
        case["batch"] * case["q_seq_len"],
        case["num_kv_splits"],
        case["nhead"],
        case["v_head_dim"],
    )
    assert attn_lse is None


@pytest.mark.parametrize("variant", _MI400_KERNEL_VARIANTS, ids=lambda variant: variant.name)
@pytest.mark.parametrize("config", _MI400_CASES, ids=lambda config: config.name)
def test_mla_mi400_numerics(monkeypatch, config, variant):
    _require_gfx1250()
    case = _make_mla_mi400_case(monkeypatch, config, variant)

    attn_logits, attn_lse = _run_mla_mi400(case, return_lse=True)

    assert torch.isfinite(attn_logits.detach().float().cpu()).all()
    assert torch.isfinite(attn_lse.detach().float().cpu()).all()
    assert torch.isfinite(case["out"].detach().float().cpu()).all()
    assert attn_logits.shape == (
        case["batch"] * case["q_seq_len"],
        case["num_kv_splits"],
        case["nhead"],
        case["v_head_dim"],
    )
    assert case["out"].shape == (
        case["batch"] * case["q_seq_len"],
        case["nhead"],
        case["v_head_dim"],
    )
    assert attn_lse.shape == (case["batch"] * case["q_seq_len"], case["nhead"])

    poc_dump_dir = os.getenv("AITER_MLA_LOAD_POC_DUMP_DIR")
    expected = (
        _load_raw_float32_prefix(Path(poc_dump_dir) / "splitData.bin", case["out"].shape, case["out"].device)
        if poc_dump_dir
        else _ref_mla_mi400(case)
    )
    _assert_cosine_close(case["out"], expected)
