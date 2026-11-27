# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942 asm MLA decode with KV views the 32-bit-offset kernels cannot address.

Most gfx942 decode kernels read KV through one buffer descriptor based at
``kv_buffer.data_ptr()``, at a 32-bit byte offset of
``u24(page) * u24(page_stride)``. Rows past 4 GiB, or pages 16 MiB or more
apart, are read from the wrong address without a fault. Each test decodes the
same KV rows from such a view and from one the kernels can address: the outputs
must match, or ``mla_decode_fwd`` must reject the view.
"""

import math
import re
from typing import NamedTuple

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.mla import mla_decode_fwd
from aiter.ops.attention import mla_decode_asm_query

torch.set_default_device("cuda")

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx942", reason="covers the gfx942 asm MLA decode kernels"
)

# num_records of the KV buffer descriptor in the 32-bit-offset kernels.
KV_DESC_BYTES = (1 << 32) - 16
BATCH, KV_LEN = 4, 1024
SM_SCALE = 576**-0.5
BUF_BYTES = (1 << 32) + (64 << 20)
BF16, FP8, BYTE = dtypes.bf16, dtypes.fp8, torch.uint8


class Case(NamedTuple):
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    nhead: int
    qlen: int
    persistent: bool
    lse: bool = False
    page_size: int = 1
    # Whether the selected kernel addresses KV with 64-bit math. If not, the
    # launcher must reject KV views it cannot address.
    kv_addr64: bool = False
    # False if the kernel's output for the same inputs varies from run to run.
    deterministic: bool = True

    @property
    def page_bytes(self) -> int:
        if self.kv_dtype == BYTE:
            return self.page_size * 656
        return self.page_size * 576 * self.kv_dtype.itemsize


def _type_name(dtype: torch.dtype) -> str:
    return {BF16: "bf16", FP8: "fp8", BYTE: "byte"}[dtype]


def kernel_name(case: Case) -> str:
    _, _, _, kernel = mla_decode_asm_query(
        get_gfx(),
        _type_name(case.q_dtype),
        _type_name(case.kv_dtype),
        case.nhead,
        1,
        case.qlen,
        case.persistent,
        True,
        case.lse,
        False,
        case.page_size,
    )
    m = re.search(r"_ZN5aiter\d+(\w+)E$", kernel)
    return m.group(1) if m else kernel


def _case_id(case: Case) -> str:
    return (
        f"{_type_name(case.q_dtype)}-{_type_name(case.kv_dtype)}-h{case.nhead}"
        f"-q{case.qlen}-{'ps' if case.persistent else 'nps'}"
        + ("-lse" if case.lse else "")
        + (f"-page{case.page_size}" if case.page_size != 1 else "")
    )


# The gfx942 decode entries of hsa/gfx942/mla/mla_asm.csv, with the kernel
# mla_decode_fwd selects in the comment. Not covered: the gqa64 kernels.
# mla_decode_fwd folds fp8 h64 onto the h16 kernels, and the bf16 m64x1 kernel
# does not apply the causal mask, so it has no reference to check against.
CASES = [
    Case(FP8, FP8, 16, 1, False),  # mla_a8w8_qh16_qseqlen1_gqaratio16
    Case(FP8, FP8, 16, 1, True),  # mla_a8w8_qh16_qseqlen1_gqaratio16_ps
    Case(FP8, FP8, 16, 2, False),  # mla_a8w8_qh16_qseqlen2_gqaratio16
    Case(FP8, FP8, 16, 2, True, deterministic=False),  # ..._qseqlen2_gqaratio16_ps
    Case(FP8, FP8, 16, 4, False),  # mla_a8w8_qh64_qseqlen4_gqaratio16
    Case(FP8, FP8, 16, 4, True),  # mla_a8w8_qh64_qseqlen4_gqaratio16_ps
    Case(FP8, FP8, 8, 1, False),  # mla_a8w8_qh8_qseqlen1_gqaratio8_v1
    Case(FP8, FP8, 128, 1, False),  # mla_a8w8_qh128_m32x4_n16x2_msk1
    Case(FP8, FP8, 128, 1, True),  # mla_a8w8_qh128_m32x4_n16x2_msk0_ps
    Case(BF16, BF16, 16, 1, False, kv_addr64=True),  # mla_dec_stage1_..._subQ16_mqa16
    Case(BF16, BF16, 16, 1, True, kv_addr64=True),  # mla_a16w16_qh16_m16x4_..._ps
    Case(BF16, BF16, 16, 2, True, kv_addr64=True),
    Case(BF16, BF16, 16, 4, True, kv_addr64=True),
    Case(BF16, BF16, 16, 4, True, lse=True),  # mla_a16w16_qh16_m16x4_..._lse_ps
    Case(BF16, BF16, 16, 2, False),  # mla_a16w16_qh16_m16x4_n16x1_coex0_mask1
    Case(BF16, BF16, 16, 4, False),
    Case(BF16, BF16, 16, 8, False),  # mla_a16w16_qh16_m32x4_n16x1_coex0_mask1
    Case(BF16, BF16, 8, 1, False),  # mla_a16w16_qh8_qseqlen1_gqaratio8_v3
    Case(BF16, BF16, 8, 2, True),  # mla_dec_stage1_..._subQ16_mqa16_ps
    Case(BF16, BF16, 32, 1, False),  # mla_a16w16_qh16_m32x1_n16x1_coex0_mask1
    Case(BF16, BF16, 128, 1, False, kv_addr64=True),  # ..._subQ128_mqa128
    Case(BF16, FP8, 16, 4, True),  # mla_a16w8_qh16_m16x4_n16x1_coex0_mask1_ps
    Case(BF16, BYTE, 16, 1, True, page_size=64),  # ..._ps_page64_ds32
]


@pytest.fixture(scope="module")
def kv_storage():
    free, _ = torch.cuda.mem_get_info()
    if free < BUF_BYTES + (4 << 30):
        pytest.skip(f"needs {BUF_BYTES / 2**30 + 4:.1f} GiB free, {free / 2**30:.1f}")
    buf = torch.zeros(BUF_BYTES, dtype=torch.uint8)
    yield buf
    del buf
    torch.cuda.empty_cache()


def kv_view(buf: torch.Tensor, case: Case, num_pages: int) -> torch.Tensor:
    raw = buf[: num_pages * case.page_bytes]
    if case.kv_dtype == BYTE:
        return raw.view(num_pages, case.page_bytes)
    return raw.view(case.kv_dtype).view(num_pages, case.page_size, 1, 576)


def _raw_pages(kv: torch.Tensor) -> torch.Tensor:
    """Pages of ``kv`` as untyped words, for gathers and scatters on any dtype."""
    word = torch.uint8 if kv.element_size() == 1 else torch.int16
    return kv.view(word).view(kv.shape[0], -1)


class Inputs(NamedTuple):
    q: torch.Tensor
    pages: torch.Tensor  # [num_pages, page_bytes] uint8, in request order
    tokens: torch.Tensor  # [num_pages * page_size, 576] fp32, the dequantized KV
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_last_page_lens: torch.Tensor
    metadata: dict
    kv_len: int


def make_inputs(case: Case, kv_len: int = KV_LEN) -> Inputs:
    torch.manual_seed(0)
    pages_per_req = kv_len // case.page_size
    num_pages = BATCH * pages_per_req
    if case.kv_dtype == BYTE:
        nope = torch.randn(num_pages, case.page_size * 512).to(FP8)
        scale = torch.ones(num_pages, case.page_size * 4, dtype=torch.float32)
        rope = torch.randn(num_pages, case.page_size * 64).to(BF16)
        pages = torch.cat(
            [t.view(torch.uint8) for t in (nope, scale, rope)], dim=1
        ).contiguous()
        tokens = torch.cat(
            [
                nope.float().view(-1, 512),
                rope.float().view(-1, 64),
            ],
            dim=1,
        )
    else:
        kv = torch.randn(num_pages, case.page_size, 1, 576).to(case.kv_dtype)
        pages = kv.view(num_pages, -1).view(torch.uint8)
        tokens = kv.float().view(-1, 576)
    q = torch.randn(BATCH * case.qlen, case.nhead, 576).to(case.q_dtype)
    qo_indptr = torch.arange(BATCH + 1, dtype=torch.int32) * case.qlen
    kv_indptr = torch.arange(BATCH + 1, dtype=torch.int32) * pages_per_req
    last = torch.full((BATCH,), case.page_size, dtype=torch.int32)

    metadata = {}
    if case.persistent:
        names = (
            "work_meta_data",
            "work_indptr",
            "work_info_set",
            "reduce_indptr",
            "reduce_final_map",
            "reduce_partial_map",
        )
        info = aiter.get_mla_metadata_info_v1(
            BATCH, case.qlen, case.nhead, case.q_dtype, case.kv_dtype, is_sparse=False
        )
        metadata = {n: torch.empty(s, dtype=t) for n, (s, t) in zip(names, info)}
        fp8_h64 = case.nhead == 64 and case.q_dtype == FP8
        aiter.get_mla_metadata_v1(
            qo_indptr,
            kv_indptr,
            last,
            case.nhead,
            1,
            True,
            metadata["work_meta_data"],
            metadata["work_info_set"],
            metadata["work_indptr"],
            metadata["reduce_indptr"],
            metadata["reduce_final_map"],
            metadata["reduce_partial_map"],
            page_size=case.page_size,
            kv_granularity=max(case.page_size, 32 if fp8_h64 else 16),
            max_seqlen_qo=case.qlen,
            uni_seqlen_qo=case.qlen,
            dtype_q=case.q_dtype,
            dtype_kv=FP8 if case.kv_dtype == BYTE else case.kv_dtype,
        )
    return Inputs(q, pages, tokens, qo_indptr, kv_indptr, last, metadata, kv_len)


def reference(case: Case, inp: Inputs) -> torch.Tensor:
    out = torch.empty(BATCH * case.qlen, case.nhead, 512)
    kv_len = inp.kv_len
    for r in range(BATCH):
        k = inp.tokens[r * kv_len : (r + 1) * kv_len]
        q = inp.q[r * case.qlen : (r + 1) * case.qlen].float()
        s = torch.einsum("thd,ld->thl", q, k) * SM_SCALE
        visible = kv_len - case.qlen + 1 + torch.arange(case.qlen)
        mask = torch.arange(kv_len)[None, :] >= visible[:, None]
        s.masked_fill_(mask[:, None, :], float("-inf"))
        out[r * case.qlen : (r + 1) * case.qlen] = torch.einsum(
            "thl,ld->thd", s.softmax(-1), k[:, :512]
        )
    return out


def place(kv: torch.Tensor, inp: Inputs, first_page: int) -> torch.Tensor:
    """Write the KV pages at ``first_page`` onward, shuffled; returns kv_indices."""
    num_pages = inp.pages.shape[0]
    perm = torch.randperm(
        num_pages, generator=torch.Generator().manual_seed(1), device="cpu"
    )
    kv_indices = (first_page + perm).to(device="cuda", dtype=torch.int32)
    word = torch.uint8 if kv.element_size() == 1 else torch.int16
    _raw_pages(kv)[kv_indices.long()] = inp.pages.view(word)
    return kv_indices


def decode(case: Case, inp: Inputs, kv: torch.Tensor, kv_indices: torch.Tensor):
    one = torch.ones(1, dtype=torch.float32)
    out = torch.full(
        (BATCH * case.qlen, case.nhead, 512), float("nan"), dtype=torch.bfloat16
    )
    _, lse = mla_decode_fwd(
        inp.q,
        kv,
        out,
        inp.qo_indptr,
        inp.kv_indptr,
        kv_indices,
        inp.kv_last_page_lens,
        case.qlen,
        page_size=case.page_size,
        sm_scale=SM_SCALE,
        q_scale=one if case.q_dtype == FP8 else None,
        kv_scale=one if case.kv_dtype != BF16 else None,
        return_lse=case.lse,
        **inp.metadata,
    )
    torch.cuda.synchronize()
    return out, lse


def check_matches(case, name, what, out, lse, out_ref, lse_ref):
    row_diff = (out.float() - out_ref.float()).abs().amax(-1)
    if case.lse:
        row_diff = torch.maximum(row_diff, (lse - lse_ref).abs())
    diff = row_diff.max().item()
    moved = (row_diff > 0).float().mean().item()
    print(f"{name}: {what}: max |out - out_ref| = {diff:.3g}, rows moved {moved:.1%}")
    # A nondeterministic kernel moves the odd (token, head) row from run to run;
    # reading the wrong KV moves all of them.
    allowed = 0.0 if case.deterministic else 1 / 16
    assert moved <= allowed, (
        f"{name} silently read the wrong KV rows ({what}): "
        f"max |out - out_ref| = {diff:.3g}, rows moved {moved:.1%}"
    )


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_kv_rows_past_4gib(kv_storage, case: Case):
    """Identical KV rows below and past 4 GiB must decode identically."""
    name = kernel_name(case)
    inp = make_inputs(case)
    num_pages = inp.pages.shape[0]
    # The wrapped address of a high page lands past the low pages, on zeros.
    high_page = math.ceil((1 << 32) / case.page_bytes) + num_pages + 64
    kv = kv_view(kv_storage, case, high_page + num_pages)
    try:
        low_indices = place(kv, inp, 0)
        out_low, lse_low = decode(case, inp, kv[:num_pages], low_indices)
        ref = reference(case, inp)
        err = (out_low.float() - ref).abs().max().item() / ref.abs().max().item()
        tol = 0.02 if case.kv_dtype == BF16 else 0.1
        assert err < tol, f"{name}: below 4 GiB, relative error {err:.3g}"

        high_indices = place(kv, inp, high_page)
        try:
            out_high, lse_high = decode(case, inp, kv, high_indices)
        except RuntimeError as e:
            print(f"{name}: rejected: {e}")
            assert not case.kv_addr64, f"{name} rejected a KV view it can address: {e}"
            assert "KV view spans" in str(e), e
            return
        check_matches(
            case, name, "rows past 4 GiB", out_high, lse_high, out_low, lse_low
        )
        assert case.kv_addr64, f"{name} is listed with 32-bit KV offsets"
    finally:
        kv_storage.zero_()


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_kv_rows_at_descriptor_limit(kv_storage, case: Case):
    """A KV view that ends at the descriptor's last byte decodes correctly."""
    name = kernel_name(case)
    inp = make_inputs(case)
    num_pages = inp.pages.shape[0]
    view_pages = KV_DESC_BYTES // case.page_bytes
    kv = kv_view(kv_storage, case, view_pages)
    try:
        out_low, lse_low = decode(case, inp, kv, place(kv, inp, 0))
        top_indices = place(kv, inp, view_pages - num_pages)
        out_top, lse_top = decode(case, inp, kv, top_indices)
        check_matches(
            case, name, "rows at the top of 4 GiB", out_top, lse_top, out_low, lse_low
        )
    finally:
        kv_storage.zero_()


# The 32-bit-offset kernels multiply by the low 24 bits of the page stride.
STRIDE_CASES = [c for c in CASES if c.page_size == 1 and c.kv_dtype != BYTE]


@pytest.mark.parametrize("case", STRIDE_CASES, ids=_case_id)
def test_page_stride_past_16mib(kv_storage, case: Case):
    """Pages 16 MiB + 1 row apart decode like the same pages packed densely."""
    name = kernel_name(case)
    # 192 pages 16 MiB apart stay inside the 4 GiB descriptor.
    inp = make_inputs(case, kv_len=48)
    num_pages = inp.pages.shape[0]
    stride = (1 << 24) // case.kv_dtype.itemsize + 576
    dense = torch.empty(num_pages, 1, 1, 576, dtype=case.kv_dtype)
    kv = kv_storage.view(case.kv_dtype).as_strided(
        (num_pages, 1, 1, 576), (stride, 576, 576, 1)
    )
    assert (num_pages - 1) * stride + 576 <= KV_DESC_BYTES // case.kv_dtype.itemsize
    try:
        out_dense, lse_dense = decode(case, inp, dense, place(dense, inp, 0))
        try:
            out, lse = decode(case, inp, kv, place(kv, inp, 0))
        except RuntimeError as e:
            print(f"{name}: rejected: {e}")
            assert not case.kv_addr64, f"{name} rejected a KV view it can address: {e}"
            assert "page stride" in str(e), e
            return
        check_matches(case, name, "16 MiB page stride", out, lse, out_dense, lse_dense)
        assert case.kv_addr64, f"{name} is listed with 32-bit KV offsets"
    finally:
        kv_storage.zero_()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
