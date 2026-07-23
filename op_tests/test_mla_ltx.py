# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA decode micro-tests: large page_id, KV byte offset, and >4GB pools.

Use -n nhead or -n nhead,decode_qlen (decode_qlen = MTP+1, same as test_mla.py).
page_size=1; compares PyTorch golden vs mla_decode_fwd (ASM .co).
.co basenames are resolved from hsa/<gfx>/mla/mla_asm.csv (same as C++ dispatch).

Examples:
  python op_tests/test_mla_ltx.py --suite boundary -d bf16 -kvd bf16 -n 64,1
  python op_tests/test_mla_ltx.py --suite page16m -d fp8 -kvd fp8 -n 16,1 --ps ps --lse off
  MLA_PAGE_OOB_NUM_PAGES=3800000 python op_tests/test_mla_ltx.py --suite over4g
"""

from __future__ import annotations

import argparse
import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import checkAllclose

torch.set_default_device("cuda")

# --- Fixed MLA layout (decode absorb, page_size=1) ---
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
PAGE_SIZE = 1
NHEAD_KV = 1
DECODE_QLEN = 1
BATCH_SIZE = 1
SUB_KV_TILE = 128
SAFE_PAGE_BASE = 1_000
PAGE_ID_16M = 16_000_000
SEED_CHUNK_PAGES = 262_144

SUB_KV_KERNEL = SUB_KV_TILE  # legacy alias for bench_mla_ckv.py


def _parse_nhead_decode_qlen(value: int | tuple) -> tuple[int, int]:
    """Like test_mla.py -n: int nhead -> decode_qlen=1; '16,4' -> (16, 4)."""
    if isinstance(value, int):
        return value, 1
    if isinstance(value, tuple):
        if len(value) == 1:
            return int(value[0]), 1
        if len(value) >= 2:
            return int(value[0]), int(value[1])
    raise ValueError(f"invalid -n / --nhead value: {value!r}")


def _csv_type_name(dtype) -> str:
    if dtype == dtypes.fp8:
        return "fp8"
    if dtype in (dtypes.bf16, torch.bfloat16):
        return "bf16"
    raise ValueError(f"unsupported dtype for mla_asm.csv: {dtype}")


def _dtype_element_size(dtype) -> int:
    return 1 if dtype == dtypes.fp8 else 2


@dataclass
class PointCase:
    page_base: int
    ctx_len: int
    label: str
    suite: str


@dataclass
class Harness:
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    nhead: int
    decode_qlen: int = DECODE_QLEN

    @property
    def use_fp8(self) -> bool:
        return self.q_dtype == dtypes.fp8 and self.kv_dtype == dtypes.fp8

    @property
    def bytes_per_page(self) -> int:
        return QK_HEAD_DIM * _dtype_element_size(self.kv_dtype)

    def csv_dispatch_keys(self) -> dict[str, object]:
        return {
            "qType": _csv_type_name(self.q_dtype),
            "kvType": _csv_type_name(self.kv_dtype),
            "Gqa": self.nhead,
            "qSeqLen": self.decode_qlen,
            "prefill": 0,
            "causal": 0,
        }

    def summary(self) -> str:
        q = _csv_type_name(self.q_dtype)
        kv = _csv_type_name(self.kv_dtype)
        return f"q={q} kv={kv} nhead={self.nhead} decode_qlen={self.decode_qlen}"

    def page_id_last_safe_2g(self) -> int:
        return ((1 << 31) - 1) // self.bytes_per_page

    def page_id_first_over_2g(self) -> int:
        return self.page_id_last_safe_2g() + 1

    def page_id_last_safe_4g(self) -> int:
        return ((1 << 32) - 1) // self.bytes_per_page

    def page_id_first_over_4g(self) -> int:
        return self.page_id_last_safe_4g() + 1

    def mega_ctx_len(self) -> int:
        return self.page_id_first_over_4g() + 16


HARNESS = Harness(dtypes.bf16, dtypes.bf16, 64, 1)

# Module-level names for bench_mla_ckv.py and similar imports.
NHEAD = HARNESS.nhead
USE_FP8 = HARNESS.use_fp8
BYTES_PER_PAGE = HARNESS.bytes_per_page
PAGE_ID_FIRST_OVER_4G = HARNESS.page_id_first_over_4g()
DECODE_QLEN = HARNESS.decode_qlen


def _sync_module_aliases() -> None:
    global NHEAD, USE_FP8, BYTES_PER_PAGE, PAGE_ID_FIRST_OVER_4G, DECODE_QLEN
    NHEAD = HARNESS.nhead
    USE_FP8 = HARNESS.use_fp8
    BYTES_PER_PAGE = HARNESS.bytes_per_page
    PAGE_ID_FIRST_OVER_4G = HARNESS.page_id_first_over_4g()
    DECODE_QLEN = HARNESS.decode_qlen


def apply_config(
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    nhead: int,
    decode_qlen: int = DECODE_QLEN,
) -> None:
    if q_dtype == dtypes.fp8 and kv_dtype == dtypes.bf16:
        raise ValueError(
            "fp8 Q with bf16 KV is not supported (see test_mla.py check_support)"
        )
    global HARNESS
    HARNESS = Harness(q_dtype, kv_dtype, nhead, decode_qlen)
    _sync_module_aliases()


def apply_preset(name: str) -> None:
    presets = {
        "qh64_bf16": (dtypes.bf16, dtypes.bf16, 64),
        "qh16_fp8": (dtypes.fp8, dtypes.fp8, 16),
    }
    if name not in presets:
        raise ValueError(f"unknown preset {name}")
    q, kv, n = presets[name]
    apply_config(q, kv, n)


_apply_preset = apply_preset  # legacy alias


def _point_cases_for(h: Harness) -> list[PointCase]:
    p2g = h.page_id_last_safe_2g()
    p2g1 = h.page_id_first_over_2g()
    p4g = h.page_id_last_safe_4g()
    p4g1 = h.page_id_first_over_4g()
    return [
        PointCase(SAFE_PAGE_BASE, 1, "below_2g_offset", "boundary"),
        PointCase(p2g, 1, "edge_2g_last_safe", "boundary"),
        PointCase(p2g1, 1, "edge_2g_first_overflow", "boundary"),
        PointCase(p4g, 1, "edge_4g_last_safe", "boundary"),
        PointCase(p4g1, 1, "edge_4g_first_overflow", "boundary"),
        PointCase(p4g1, 16, "seq16_from_4g_overflow", "boundary"),
        PointCase(p4g1, SUB_KV_TILE, "ctx128_at_4g_boundary", "over4g"),
        PointCase(65_409, SUB_KV_TILE, "cross_window_65536_subkv", "pa_window"),
        PointCase(PAGE_ID_16M, 1, "page_id_16m", "page16m"),
        PointCase(PAGE_ID_16M, SUB_KV_TILE, "page_id_16m_ctx128", "page16m"),
        PointCase(1 << 24, 1, "page_id_2p24", "page16m"),
    ]


_POINT_CASES = _point_cases_for(HARNESS)
_SEQUENTIAL_CASES = [(HARNESS.mega_ctx_len(), "sequential_mega_over4g", "mega")]


# --- mla_asm.csv -> .co basename (logging / pytest tags; matches dispatch) ---
_AML_ASM_CACHE: dict[str, list[dict[str, object]]] = {}


def _aiter_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _co_dir(aiter_root: Path | None = None) -> Path:
    root = aiter_root or _aiter_root()
    return root / "hsa" / get_gfx() / "mla"


def _load_asm_csv(aiter_root: Path | None = None) -> list[dict[str, object]]:
    path = _co_dir(aiter_root) / "mla_asm.csv"
    key = str(path.resolve())
    if key not in _AML_ASM_CACHE:
        ints = {"Gqa", "ps", "qSeqLen", "prefill", "causal", "lse", "cprr"}
        with path.open(newline="") as f:
            _AML_ASM_CACHE[key] = [
                {k: (int(v) if k in ints else v) for k, v in row.items()}
                for row in csv.DictReader(f)
            ]
    return _AML_ASM_CACHE[key]


def co_name(persistent: bool, lse: bool, *, aiter_root: Path | None = None) -> str:
    base = HARNESS.csv_dispatch_keys()
    lse_flag = 1 if (lse and persistent) else 0
    ps = 1 if persistent else 0
    for row in _load_asm_csv(aiter_root):
        if any(row.get(k) != v for k, v in base.items()):
            continue
        if row["ps"] == ps and row["lse"] == lse_flag and row["cprr"] == 0:
            return str(row["co_name"])
    raise KeyError(f"no csv row {HARNESS.summary()} ps={ps} lse={lse_flag} cprr=0")


_co_name = co_name


# --- golden ---
def ref_masked_attention(query, key, value, scale, dtype, is_causal=True):
    w = torch.einsum("qhd,khd->hqk", query.float(), key.float()) * scale
    if is_causal:
        s_q, s_k = query.shape[0], key.shape[0]
        bias = torch.zeros(s_q, s_k, dtype=query.dtype)
        mask = torch.ones(s_q, s_k, dtype=torch.bool).tril(diagonal=s_k - s_q)
        bias.masked_fill_(mask.logical_not(), float("-inf"))
        w += bias
    w = torch.softmax(w, dim=-1)
    return torch.einsum("hqk,khd->qhd", w.float(), value.float()).to(dtype)


def torch_mla_extend(
    q, kvc_cache, qo_indptr, kv_indptr, kv_indices, sm_scale, dtype, is_causal=True
):
    qs = torch.tensor_split(q, qo_indptr.tolist()[1:])
    kvc = torch.index_select(kvc_cache, 0, kv_indices)
    kvs = torch.tensor_split(kvc, kv_indptr.tolist()[1:])
    outs = []
    for i in range(qo_indptr.shape[0] - 1):
        kvc_i, q_i = kvs[i], qs[i]
        v, _ = torch.split(kvc_i, [KV_LORA_RANK, QK_ROPE_HEAD_DIM], dim=-1)
        outs.append(ref_masked_attention(q_i, kvc_i, v, sm_scale, dtype, is_causal))
    return torch.concat(outs)


# --- KV pool ---
def _seed_pages(kv_buffer: torch.Tensor, ranges: Iterable[tuple[int, int]]) -> None:
    for base, length in ranges:
        if length <= 4096:
            for pid in range(base, base + length):
                gen = torch.Generator(device="cuda")
                gen.manual_seed(pid & 0xFFFFFFFF)
                kv_buffer[pid].copy_(
                    torch.randn(1, QK_HEAD_DIM, device="cuda", generator=gen) * 0.02
                )
            continue
        gen = torch.Generator(device="cuda")
        gen.manual_seed((base & 0xFFFFFFFF) ^ (length & 0xFFFFFFFF))
        for start in range(base, base + length, SEED_CHUNK_PAGES):
            end = min(start + SEED_CHUNK_PAGES, base + length)
            n = end - start
            kv_buffer[start:end].copy_(
                torch.randn(n, 1, QK_HEAD_DIM, device="cuda", generator=gen) * 0.02
            )


def _build_kv_pool(num_pages: int, ranges: list[tuple[int, int]]) -> torch.Tensor:
    elem = HARNESS.kv_dtype
    kv = torch.zeros((num_pages, NHEAD_KV, QK_HEAD_DIM), dtype=elem, device="cuda")
    _seed_pages(kv, ranges)
    return kv


def _build_persistent_metadata(
    qo_indptr,
    kv_indptr,
    kv_last_page_lens,
    max_split_per_batch,
    *_legacy,
):
    """Persistent MLA metadata (_ps kernels). *_legacy absorbs old extra args."""
    bs = qo_indptr.shape[0] - 1
    dtype = dtypes.bf16
    sizes = aiter.get_mla_metadata_info_v1(
        bs,
        HARNESS.decode_qlen,
        HARNESS.nhead,
        dtype,
        dtype,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=max_split_per_batch,
        intra_batch_mode=False,
    )

    def buf(i):
        n, t = sizes[i]
        return torch.empty(n, dtype=t, device="cuda")

    wmd, wi, wis, ri, rfm, rpm = (buf(i) for i in range(6))
    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_last_page_lens,
        HARNESS.nhead // NHEAD_KV,
        NHEAD_KV,
        False,
        wmd,
        wis,
        wi,
        ri,
        rfm,
        rpm,
        page_size=PAGE_SIZE,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=HARNESS.decode_qlen,
        uni_seqlen_qo=HARNESS.decode_qlen,
        fast_mode=True,
        max_split_per_batch=max_split_per_batch,
        intra_batch_mode=False,
        dtype_q=dtype,
        dtype_kv=dtype,
    )
    return dict(
        work_meta_data=wmd,
        work_indptr=wi,
        work_info_set=wis,
        reduce_indptr=ri,
        reduce_final_map=rfm,
        reduce_partial_map=rpm,
    )


# --- decode: golden vs ASM ---
def _make_indptr(ctx_len: int, page_base: int | None):
    qlen = HARNESS.decode_qlen
    kv_indptr = torch.tensor([0, ctx_len], dtype=torch.int, device="cuda")
    if page_base is None:
        kv_indices = torch.arange(ctx_len, dtype=torch.int, device="cuda")
    else:
        kv_indices = torch.arange(
            page_base, page_base + ctx_len, dtype=torch.int, device="cuda"
        )
    qo_indptr = torch.tensor([0, qlen], dtype=torch.int, device="cuda")
    return qo_indptr, kv_indptr, kv_indices


def mla_decode_asm(
    kv_buffer,
    num_pages,
    qo_indptr,
    kv_indptr,
    kv_indices,
    *,
    persistent: bool,
    return_lse: bool,
    max_split: int,
):
    sm = 1.0 / (QK_HEAD_DIM**0.5)
    qlen = HARNESS.decode_qlen
    kv_lens = torch.ones(BATCH_SIZE, dtype=torch.int, device="cuda")
    q = torch.randn((qlen, HARNESS.nhead, QK_HEAD_DIM), dtype=torch.bfloat16)
    kv_bf16 = (
        kv_buffer.to(torch.bfloat16) if HARNESS.kv_dtype == dtypes.fp8 else kv_buffer
    )
    ref = torch_mla_extend(
        q, kv_bf16, qo_indptr, kv_indptr, kv_indices, sm, torch.bfloat16, True
    )

    out = torch.empty((qlen, HARNESS.nhead, V_HEAD_DIM), dtype=torch.bfloat16).fill_(-1)
    kv_view = kv_buffer.view(num_pages, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM)
    q_asm = q.to(dtypes.fp8) if HARNESS.q_dtype == dtypes.fp8 else q
    kw = dict(
        page_size=PAGE_SIZE,
        nhead_kv=NHEAD_KV,
        sm_scale=sm,
        return_lse=return_lse,
    )
    if HARNESS.use_fp8:
        kw["q_scale"] = torch.ones(1, dtype=torch.float, device="cuda")
        kw["kv_scale"] = torch.ones(1, dtype=torch.float, device="cuda")
    if persistent:
        kw["num_kv_splits"] = max_split
        kw.update(_build_persistent_metadata(qo_indptr, kv_indptr, kv_lens, max_split))
    aiter.mla.mla_decode_fwd(
        q_asm,
        kv_view,
        out,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        qlen,
        **kw,
    )
    return ref, out


_decode = mla_decode_asm


def run_point(
    kv_buffer, num_pages, page_base, ctx_len, *, persistent, return_lse, max_split
):
    qo, kv_i, kv_x = _make_indptr(ctx_len, page_base)
    ref, asm = mla_decode_asm(
        kv_buffer,
        num_pages,
        qo,
        kv_i,
        kv_x,
        persistent=persistent,
        return_lse=return_lse,
        max_split=max_split,
    )
    return ref, asm, kv_x


_run_mla_decode_point = run_point


def run_sequential(kv_buffer, num_pages, ctx_len, *, persistent, return_lse, max_split):
    qo, kv_i, kv_x = _make_indptr(ctx_len, None)
    ref, asm = mla_decode_asm(
        kv_buffer,
        num_pages,
        qo,
        kv_i,
        kv_x,
        persistent=persistent,
        return_lse=return_lse,
        max_split=max_split,
    )
    return ref, asm, kv_x


_run_mla_decode_sequential = run_sequential


def _page_offset(page_id: int) -> int:
    return page_id * HARNESS.bytes_per_page


def _pool_bytes(num_pages: int) -> int:
    return num_pages * HARNESS.bytes_per_page


def _need_num_pages(
    points: list[PointCase], seq: list[tuple[int, str]], ctx_override: int
) -> int:
    need = 10_000
    for c in points:
        ctx = ctx_override if ctx_override > 0 else c.ctx_len
        need = max(need, c.page_base + ctx)
    for ctx, _ in seq:
        need = max(need, ctx + 10_000)
    return need + 1


def _seed_ranges(
    points: list[PointCase], seq: list[tuple[int, str]], ctx_override: int
):
    ranges = []
    for c in points:
        ctx = ctx_override if ctx_override > 0 else c.ctx_len
        ranges.append((c.page_base, ctx))
    for ctx, _ in seq:
        ranges.append((0, ctx))
    return ranges


def _amax_ok(ref, asm, tol=0.05) -> tuple[bool, float]:
    amax = (ref.float() - asm.float()).abs().max().item()
    return amax < tol, amax


def _filter_points(suite: str) -> list[PointCase]:
    if suite == "all":
        return list(_POINT_CASES)
    return [c for c in _POINT_CASES if c.suite == suite]


# --- pytest ---
_PYTEST_BOUNDARY = [
    (c.page_base, c.ctx_len, c.label) for c in _POINT_CASES if c.suite == "boundary"
]


@pytest.fixture(scope="module")
def kv_pool_boundary():
    n = int(
        os.environ.get("MLA_PAGE_OOB_NUM_PAGES", str(PAGE_ID_FIRST_OVER_4G + 2_000))
    )
    ranges = _seed_ranges(
        [PointCase(b, c, l, "boundary") for b, c, l in _PYTEST_BOUNDARY], [], 0
    )
    try:
        pool = _build_kv_pool(n, ranges)
    except torch.cuda.OutOfMemoryError as e:
        pytest.skip(str(e))
    yield pool, n
    del pool
    torch.cuda.empty_cache()


@pytest.mark.skipif(get_gfx() != "gfx950", reason="gfx950 MLA asm")
@pytest.mark.parametrize("page_base,ctx_len,label", _PYTEST_BOUNDARY)
@pytest.mark.parametrize("persistent", [True, False], ids=["ps", "nps"])
@pytest.mark.parametrize("return_lse", [False, True], ids=["nolse", "lse"])
def test_mla_qh64_co_boundary(
    kv_pool_boundary, page_base, ctx_len, label, persistent, return_lse
):
    pool, n = kv_pool_boundary
    ref, asm, _ = run_point(
        pool,
        n,
        page_base,
        ctx_len,
        persistent=persistent,
        return_lse=return_lse,
        max_split=1,
    )
    tag = f"{label}_{co_name(persistent, return_lse)}"
    assert checkAllclose(ref, asm, msg=tag, rtol=3e-2, atol=3e-2) == 0, tag


def _select_cases(args) -> tuple[list[PointCase], list[tuple[int, str]]]:
    points = _filter_points(args.suite)
    if args.skip_mega:
        seq: list[tuple[int, str]] = []
    elif args.mega_ctx > 0:
        seq = [(args.mega_ctx, f"sequential_ctx{args.mega_ctx}")]
        if args.suite not in ("mega", "all"):
            points = []
    elif args.suite in ("all", "mega"):
        seq = [(c[0], c[1]) for c in _SEQUENTIAL_CASES]
    else:
        seq = []
    if args.page_base > 0:
        ctx = args.ctx if args.ctx > 0 else 1
        points = [
            PointCase(args.page_base, ctx, f"page_id_{args.page_base}", "page16m")
        ]
        seq = []
    if args.cases:
        want = {x.strip() for x in args.cases.split(",")}
        points = [c for c in points if c.label in want]
        seq = [(ctx, lab) for ctx, lab in seq if lab in want]
    return points, seq


def _main():
    p = argparse.ArgumentParser(
        description="MLA decode: large page_id / >4GB KV pool tests"
    )
    p.add_argument(
        "--suite",
        choices=["boundary", "over4g", "pa_window", "mega", "page16m", "all"],
        default="all",
    )
    p.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        default=dtypes.bf16,
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
        metavar="{bf16, fp8}",
        help="Q dtype (same as test_mla.py -d)",
    )
    p.add_argument(
        "-kvd",
        "--kv_dtype",
        type=dtypes.str2Dtype,
        default=dtypes.bf16,
        choices=[dtypes.d_dtypes["bf16"], dtypes.d_dtypes["fp8"]],
        metavar="{bf16, fp8}",
        help="KV dtype (same as test_mla.py -kvd)",
    )
    p.add_argument(
        "-n",
        "--nhead",
        type=dtypes.str2tuple,
        default="64,1",
        help="nhead and decode_qlen (MTP+1), same as test_mla.py -n. e.g. -n 16,1 or -n 16,4",
    )
    p.add_argument("--ctx", type=int, default=0)
    p.add_argument("--page-base", type=int, default=0)
    p.add_argument("--num-kv-splits", type=int, default=1)
    p.add_argument("--cases", type=str, default="")
    p.add_argument("--ps", choices=["ps", "nps", "both"], default="both")
    p.add_argument("--lse", choices=["on", "off", "both"], default="both")
    p.add_argument(
        "--aiter-root",
        type=str,
        default="",
        help="Aiter repo root (default: parent of op_tests; used to read mla_asm.csv)",
    )
    p.add_argument("--skip-mega", action="store_true")
    p.add_argument("--mega-ctx", type=int, default=0)
    args = p.parse_args()

    nhead, decode_qlen = _parse_nhead_decode_qlen(args.nhead)
    apply_config(args.dtype, args.kv_dtype, nhead, decode_qlen)
    global _POINT_CASES, _SEQUENTIAL_CASES
    _POINT_CASES = _point_cases_for(HARNESS)
    _SEQUENTIAL_CASES = [(HARNESS.mega_ctx_len(), "sequential_mega_over4g", "mega")]

    if get_gfx() != "gfx950":
        print(f"skip: gfx={get_gfx()} (need gfx950)")
        return 1

    aiter_root = Path(args.aiter_root or _aiter_root())

    ps_list = {"ps": [True], "nps": [False], "both": [True, False]}[args.ps]
    lse_list = {"on": [True], "off": [False], "both": [True, False]}[args.lse]
    points, seq = _select_cases(args)

    num_pages = int(
        os.environ.get(
            "MLA_PAGE_OOB_NUM_PAGES", str(_need_num_pages(points, seq, args.ctx))
        )
    )
    print(
        f"config={HARNESS.summary()} suite={args.suite} pages={num_pages} "
        f"pool_GiB={_pool_bytes(num_pages) / 2**30:.2f} "
        f"4G_page_id>={HARNESS.page_id_first_over_4g()} stride={HARNESS.bytes_per_page}"
    )

    try:
        kv = _build_kv_pool(num_pages, _seed_ranges(points, seq, args.ctx))
    except torch.cuda.OutOfMemoryError as e:
        print(f"OOM: {e}")
        return 2

    failed = 0

    def report_point(label, page_base, ctx, ref, asm, kv_idx):
        nonlocal failed
        max_pid = page_base + ctx - 1
        off = _page_offset(max_pid)
        ok, amax = _amax_ok(ref, asm)
        failed += int(not ok)
        cross = (page_base & 0xFFFF0000) != (max_pid & 0xFFFF0000)
        print(
            f"[{'PASS' if ok else 'FAIL'}] {label:42s} base={page_base:>10d} ctx={ctx:>4d} "
            f"pid..{max_pid} off={off} off>4G={off >= 1 << 32} cross_64k={cross} amax={amax:.6f}"
        )
        if not ok:
            print(f"  kv head={kv_idx[:4].tolist()} tail={kv_idx[-4:].tolist()}")

    def report_seq(label, ctx, ref, asm, kv_idx):
        nonlocal failed
        off = _page_offset(ctx - 1)
        ok, amax = _amax_ok(ref, asm)
        failed += int(not ok)
        print(
            f"[{'PASS' if ok else 'FAIL'}] {label:42s} ctx={ctx:>10d} "
            f"pool_GiB={_pool_bytes(num_pages) / 2**30:.2f} max_off={off} amax={amax:.6f}"
        )
        if not ok:
            print(f"  kv tail={kv_idx[-4:].tolist()}")

    for persistent in ps_list:
        for lse in lse_list:
            co = co_name(persistent, lse, aiter_root=aiter_root)
            print(f"\n===== {co} =====")
            for c in points:
                ctx = args.ctx or c.ctx_len
                ref, asm, kv_idx = run_point(
                    kv,
                    num_pages,
                    c.page_base,
                    ctx,
                    persistent=persistent,
                    return_lse=lse,
                    max_split=args.num_kv_splits,
                )
                report_point(c.label, c.page_base, ctx, ref, asm, kv_idx)
            for ctx, label in seq:
                ref, asm, kv_idx = run_sequential(
                    kv,
                    num_pages,
                    ctx,
                    persistent=persistent,
                    return_lse=lse,
                    max_split=args.num_kv_splits,
                )
                report_seq(label, ctx, ref, asm, kv_idx)

    print(f"\ndone: {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())

# test command:python op_tests/test_mla_ltx.py   --page-base 16999216 --ctx 1   -d fp8 -kvd fp8 -n 16,1   --ps ps --lse off
