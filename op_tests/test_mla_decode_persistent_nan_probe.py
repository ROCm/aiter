# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Op-level probe for non-finite values on the persistent MLA decode path.

Unlike ``test_mla_reduce_nan_guard.py`` -- which drives ``mla_reduce_v1`` in
isolation with injected partials -- this file runs the real decode pipeline:

    get_mla_metadata_v1()            # planner
      -> mla_decode_stage1_asm_fwd() # split-KV attention, writes the partials
        -> mla_reduce_v1()           # cross-split reduction

and answers, with hardware data, where a non-finite value can come from:

  H1  "a slot the planner reserved but stage1 never wrote"
      (empty split / degenerate split / unused tile). The partial buffers are
      allocated with ``torch.empty`` in ``aiter/mla.py``, so an unwritten slot
      that the reduce still reads contains whatever was in that memory.
      Note the torch reference (``torch_mla_extend_split_kv`` in
      test_mla_persistent.py) sizes its partial buffers with ``zeros`` /
      ``-inf`` fill, i.e. it defines unwritten slots as neutral -- the asm path
      has no such guarantee.

  H2  "stage1 itself wrote a non-finite LSE / output"

The two are told apart by pre-filling the partial buffers with a distinctive
finite sentinel before stage1 runs and classifying every slot afterwards:

    still == SENTINEL  -> never written by stage1            (H1)
    non-finite         -> written non-finite by stage1       (H2)
    finite, != SENTINEL-> written normally

Run:
    pytest -q op_tests/test_mla_decode_persistent_nan_probe.py

Env knobs:
    MLA_NAN_DUMP_DIR    where to write the input dump on failure (default ".")
    MLA_NAN_DUMP_FULL   set to 1 to include q / kv_buffer in the dump
"""

import os

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx

# A finite value that never occurs in a real partial output or LSE. Slots that
# still hold it after stage1 were not written by stage1.
SENTINEL = -1.2345678e30

DUMP_DIR = os.environ.get("MLA_NAN_DUMP_DIR", ".")
DUMP_FULL = os.environ.get("MLA_NAN_DUMP_FULL", "0") not in ("", "0")

# DeepSeek-style MLA decode geometry: kv_lora_rank 512 + rope 64, one KV head,
# page_size 1, fp8 KV cache. nhead 16 == 128 heads over TP8.
KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
NHEAD_KV = 1
PAGE_SIZE = 1


class Case:
    """One decode configuration.

    ``min_kv`` is the shortest KV length generated. Causal decode requires
    kv_len >= q_len (otherwise the leading query rows see no key at all and the
    softmax is over an empty set); keep it at decode_qlen for in-contract runs.
    """

    def __init__(
        self, batch, nhead, decode_qlen, ctx_len, max_split_per_batch, min_kv=None
    ):
        self.batch = batch
        self.nhead = nhead
        self.decode_qlen = decode_qlen
        self.ctx_len = ctx_len
        self.max_split_per_batch = max_split_per_batch
        self.min_kv = decode_qlen if min_kv is None else min_kv

    @property
    def id(self):
        return (
            f"b{self.batch}_nh{self.nhead}_q{self.decode_qlen}"
            f"_ctx{self.ctx_len}_split{self.max_split_per_batch}"
        )


CASES = [
    # decode, no speculative tokens -- the persistent planner pins 64 splits
    Case(batch=64, nhead=16, decode_qlen=1, ctx_len=4096, max_split_per_batch=64),
    # MTP / speculative decode: 4 draft tokens, 32 splits
    Case(batch=64, nhead=16, decode_qlen=4, ctx_len=4096, max_split_per_batch=32),
    # short and highly variable contexts -- maximises empty / degenerate splits
    Case(batch=128, nhead=16, decode_qlen=1, ctx_len=256, max_split_per_batch=64),
    Case(batch=128, nhead=16, decode_qlen=4, ctx_len=256, max_split_per_batch=32),
    # long contexts, every sequence split many ways
    Case(batch=32, nhead=16, decode_qlen=1, ctx_len=16384, max_split_per_batch=64),
    Case(batch=32, nhead=16, decode_qlen=4, ctx_len=16384, max_split_per_batch=32),
]

CASE_IDS = [c.id for c in CASES]

# Seeds swept by test_sweep_seeds_final_lse_is_finite. Each seed reshuffles the
# KV lengths, so the planner produces a different split / tile layout.
SWEEP_SEEDS = list(range(16))


def _skip_if_unsupported():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    if get_gfx() not in ("gfx942", "gfx950"):
        pytest.skip(f"persistent MLA decode not supported on {get_gfx()}")


class Fixture:
    """Everything ``mla_decode_fwd`` needs, plus the planner metadata."""

    pass


def _build(case, seed=0, kv_lens=None, fixed_kv_total=False):
    """Build inputs + planner metadata for one case (varlen contexts).

    ``kv_lens`` forces the per-request KV lengths; it must be applied here so
    that the planner sees the same lengths the kernels will.
    ``fixed_kv_total`` keeps the total page count constant across seeds, which
    cuda-graph replay needs (the captured graph owns fixed-size buffers).
    """
    torch.manual_seed(seed)
    dev = "cuda"
    bs = case.batch

    # Variable KV lengths so the planner produces uneven splits; the shortest
    # sequences are the ones that yield degenerate / empty splits.
    lo = max(1, case.min_kv)
    if kv_lens is not None:
        seq_lens_kv = torch.as_tensor(kv_lens, dtype=torch.int32).reshape(bs).clone()
    elif fixed_kv_total:
        # A server's paged KV pool has a fixed size, so cuda-graph replay needs
        # the page count to stay constant while the per-request split changes.
        # Pair the requests up as (L + d, L - d): the sum is always bs * L.
        assert bs % 2 == 0, "fixed_kv_total needs an even batch"
        base = max(lo, case.ctx_len // 2)
        d = torch.randint(0, base - lo + 1, (bs // 2,), dtype=torch.int32)
        seq_lens_kv = torch.empty(bs, dtype=torch.int32)
        seq_lens_kv[0::2] = base + d
        seq_lens_kv[1::2] = base - d
    else:
        seq_lens_kv = torch.randint(
            lo, max(lo + 1, case.ctx_len + 1), (bs,), dtype=torch.int32
        )
    kv_block_nums = (seq_lens_kv + PAGE_SIZE - 1) // PAGE_SIZE
    kv_last_page_lens = torch.where(
        seq_lens_kv % PAGE_SIZE == 0,
        torch.full_like(seq_lens_kv, PAGE_SIZE),
        seq_lens_kv % PAGE_SIZE,
    )

    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32)
    kv_indptr[1:] = torch.cumsum(kv_block_nums, dim=0)
    num_page = int(kv_indptr[-1].item())
    kv_indices = torch.randperm(num_page, dtype=torch.int32)

    seq_lens_qo = torch.full((bs,), case.decode_qlen, dtype=torch.int32)
    qo_indptr = torch.zeros(bs + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.cumsum(seq_lens_qo, dim=0)
    total_q = int(qo_indptr[-1].item())
    max_seqlen_qo = case.decode_qlen

    f = Fixture()
    f.case = case
    f.seed = seed
    f.bs = bs
    f.num_page = num_page
    f.total_q = total_q
    f.max_seqlen_qo = max_seqlen_qo
    f.sm_scale = 1.0 / (QK_HEAD_DIM**0.5)

    f.qo_indptr = qo_indptr.to(dev)
    f.kv_indptr = kv_indptr.to(dev)
    f.kv_indices = kv_indices.to(dev)
    f.kv_last_page_lens = kv_last_page_lens.to(dev)
    f.seq_lens_kv = seq_lens_kv.to(dev)

    q_bf16 = torch.randn(
        (total_q, case.nhead, QK_HEAD_DIM), dtype=torch.bfloat16, device=dev
    )
    kv_bf16 = torch.randn(
        (num_page, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM),
        dtype=torch.bfloat16,
        device=dev,
    )
    f.q = q_bf16.to(dtypes.fp8)
    f.kv_buffer = kv_bf16.to(dtypes.fp8)
    f.q_scale = torch.ones([1], dtype=torch.float32, device=dev)
    f.kv_scale = torch.ones([1], dtype=torch.float32, device=dev)

    (
        (wmd_size, wmd_type),
        (wi_size, wi_type),
        (wis_size, wis_type),
        (ri_size, ri_type),
        (rfm_size, rfm_type),
        (rpm_size, rpm_type),
    ) = aiter.get_mla_metadata_info_v1(
        bs,
        max_seqlen_qo,
        case.nhead,
        dtypes.fp8,
        dtypes.fp8,
        is_sparse=False,
        fast_mode=True,
        num_kv_splits=case.max_split_per_batch,
        intra_batch_mode=False,
    )

    f.work_meta_data = torch.empty(wmd_size, dtype=wmd_type, device=dev)
    f.work_indptr = torch.empty(wi_size, dtype=wi_type, device=dev)
    f.work_info_set = torch.empty(wis_size, dtype=wis_type, device=dev)
    f.reduce_indptr = torch.empty(ri_size, dtype=ri_type, device=dev)
    f.reduce_final_map = torch.empty(rfm_size, dtype=rfm_type, device=dev)
    f.reduce_partial_map = torch.empty(rpm_size, dtype=rpm_type, device=dev)

    common = dict(
        page_size=PAGE_SIZE,
        kv_granularity=max(PAGE_SIZE, 16),
        max_seqlen_qo=int(max_seqlen_qo),
        uni_seqlen_qo=case.decode_qlen,
        fast_mode=True,
        max_split_per_batch=case.max_split_per_batch,
        intra_batch_mode=False,
    )
    positional = (
        f.qo_indptr,
        f.kv_indptr,
        f.kv_last_page_lens,
        case.nhead // NHEAD_KV,
        NHEAD_KV,
        False,
        f.work_meta_data,
        f.work_info_set,
        f.work_indptr,
        f.reduce_indptr,
        f.reduce_final_map,
        f.reduce_partial_map,
    )
    # The dtype kwargs were renamed dtype_q/dtype_kv -> dtype_q_nope/dtype_kv_nope;
    # accept either so the probe runs against released images too.
    for dtype_kw in (
        {"dtype_q_nope": dtypes.fp8, "dtype_kv_nope": dtypes.fp8},
        {"dtype_q": dtypes.fp8, "dtype_kv": dtypes.fp8},
    ):
        try:
            aiter.get_mla_metadata_v1(*positional, **common, **dtype_kw)
            break
        except (RuntimeError, TypeError) as exc:
            if "keyword argument" not in str(exc):
                raise
    else:
        raise RuntimeError("get_mla_metadata_v1: no known dtype kwarg spelling")

    # Partial-buffer geometry, mirroring aiter/mla.py's persistent branch.
    f.partial_rows = int(f.reduce_partial_map.size(0)) * max_seqlen_qo
    return f


def _stage1_write_rows(f):
    """Rows of the partial buffer that stage1 is scheduled to write.

    Derived from work_info_set exactly as the torch reference
    (``torch_mla_extend_split_kv``) does: a work with partial_qo_loc != -1
    writes rows [loc, loc + (qo_end - qo_start)).
    """
    num_works = int(f.work_indptr[-1].item())
    mask = torch.zeros(f.partial_rows, dtype=torch.bool)
    if num_works == 0:
        return mask, 0
    wis = f.work_info_set[:num_works].detach().cpu().to(torch.int64)
    loc = wis[:, 1]
    qlen = wis[:, 3] - wis[:, 2]
    for i in range(num_works):
        li = int(loc[i].item())
        if li == -1:
            continue
        n = int(qlen[i].item())
        if li + n <= f.partial_rows:
            mask[li : li + n] = True
        elif li < f.partial_rows:
            mask[li : f.partial_rows] = True
    return mask, num_works


def _reduce_read_rows(f):
    """Rows of the partial buffer that mla_reduce_v1 reads.

    Mirrors the indexing in ``torch_mla_reduce_v1``: for every non-empty reduce
    tile and every split in it, rows [partial_qo_loc + i] for i in
    [0, q_end - q_start).
    """
    ri = f.reduce_indptr.detach().cpu().to(torch.int64)
    rpm = f.reduce_partial_map.detach().cpu().to(torch.int64)
    rfm = f.reduce_final_map.detach().cpu().to(torch.int64)

    mask = torch.zeros(f.partial_rows, dtype=torch.bool)
    num_tiles = ri.numel() - 1
    empty_tiles = 0
    split_hist = {}
    for t in range(num_tiles):
        s, e = int(ri[t].item()), int(ri[t + 1].item())
        if s == e:
            empty_tiles += 1
            continue
        nsplit = e - s
        split_hist[nsplit] = split_hist.get(nsplit, 0) + 1
        q_start = int(rfm[t, 0].item())
        q_end = int(rfm[t, 1].item())
        qlen = max(0, q_end - q_start)
        for k in range(s, e):
            loc = int(rpm[k].item())
            if loc < 0:
                continue
            hi = min(loc + qlen, f.partial_rows)
            if loc < hi:
                mask[loc:hi] = True
    return mask, num_tiles, empty_tiles, split_hist


def _dump(tag, f, extra):
    """Persist the mla_decode_fwd inputs that produced a non-finite value."""
    os.makedirs(DUMP_DIR, exist_ok=True)
    path = os.path.join(DUMP_DIR, f"mla_decode_fwd_nan_{tag}.pt")
    payload = {
        "case": {
            "batch": f.case.batch,
            "nhead": f.case.nhead,
            "decode_qlen": f.case.decode_qlen,
            "ctx_len": f.case.ctx_len,
            "max_split_per_batch": f.case.max_split_per_batch,
            "seed": f.seed,
            "gfx": get_gfx(),
        },
        "scalars": {
            "max_seqlen_q": f.max_seqlen_qo,
            "page_size": PAGE_SIZE,
            "nhead_kv": NHEAD_KV,
            "sm_scale": f.sm_scale,
            "num_kv_splits": f.case.max_split_per_batch,
            "q_dtype": str(f.q.dtype),
            "kv_dtype": str(f.kv_buffer.dtype),
            "q_shape": tuple(f.q.shape),
            "kv_shape": tuple(f.kv_buffer.shape),
        },
        # planner inputs
        "qo_indptr": f.qo_indptr.cpu(),
        "kv_indptr": f.kv_indptr.cpu(),
        "kv_indices": f.kv_indices.cpu(),
        "kv_last_page_lens": f.kv_last_page_lens.cpu(),
        "seq_lens_kv": f.seq_lens_kv.cpu(),
        # planner outputs
        "work_meta_data": f.work_meta_data.cpu(),
        "work_indptr": f.work_indptr.cpu(),
        "work_info_set": f.work_info_set.cpu(),
        "reduce_indptr": f.reduce_indptr.cpu(),
        "reduce_final_map": f.reduce_final_map.cpu(),
        "reduce_partial_map": f.reduce_partial_map.cpu(),
        "q_scale": f.q_scale.cpu(),
        "kv_scale": f.kv_scale.cpu(),
    }
    payload.update(extra)
    if DUMP_FULL:
        payload["q"] = f.q.cpu()
        payload["kv_buffer"] = f.kv_buffer.cpu()
    else:
        payload["note"] = (
            "q / kv_buffer omitted; regenerate with torch.manual_seed(seed) and "
            "the 'case' entry, or re-run with MLA_NAN_DUMP_FULL=1"
        )
    torch.save(payload, path)
    print(f"\n[dump] mla_decode_fwd inputs written to {path}")
    return path


def _nonfinite_report(name, t):
    nan = int(torch.isnan(t).sum().item())
    inf = int(torch.isinf(t).sum().item())
    n = t.numel()
    idx = torch.nonzero(~torch.isfinite(t.reshape(-1)))[:8].reshape(-1).tolist()
    return (
        f"{name}: {nan} NaN + {inf} Inf out of {n} "
        f"({100.0 * (nan + inf) / max(n, 1):.4f}%), first flat indices {idx}"
    )


# ---------------------------------------------------------------------------
# 1. Planner-only check: is every slot the reduce reads also scheduled to be
#    written by stage1?  This is hypothesis H1, decided from metadata alone.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_reduce_read_set_is_written_by_stage1(case):
    _skip_if_unsupported()
    f = _build(case)

    write_mask, num_works = _stage1_write_rows(f)
    read_mask, num_tiles, empty_tiles, split_hist = _reduce_read_rows(f)

    read_only = read_mask & ~write_mask
    n_read = int(read_mask.sum().item())
    n_write = int(write_mask.sum().item())
    n_gap = int(read_only.sum().item())

    print(
        f"\n[{case.id}] partial_rows={f.partial_rows} works={num_works} "
        f"reduce_tiles={num_tiles} (empty {empty_tiles}) "
        f"splits_per_tile={dict(sorted(split_hist.items()))}\n"
        f"          rows written by stage1={n_write} read by reduce={n_read} "
        f"read-but-never-written={n_gap}"
    )

    if n_gap:
        gaps = torch.nonzero(read_only).reshape(-1)[:16].tolist()
        _dump(f"planner_{case.id}", f, {"read_but_unwritten_rows": gaps})
    assert n_gap == 0, (
        f"{n_gap} partial-buffer rows are read by mla_reduce_v1 but no stage1 "
        f"work writes them (first: {torch.nonzero(read_only).reshape(-1)[:16].tolist()}). "
        f"Those rows come straight from torch.empty, so the reduce consumes "
        f"uninitialised memory."
    )


# ---------------------------------------------------------------------------
# 2. Hardware check: run stage1 over sentinel-filled partial buffers and
#    classify every slot the reduce will read. Decides H1 vs H2.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_stage1_partials_are_written_and_finite(case):
    _skip_if_unsupported()
    f = _build(case)
    dev = "cuda"
    nhead = case.nhead

    # Same shapes aiter/mla.py allocates, but sentinel-filled instead of empty.
    logits = torch.full(
        (f.partial_rows, 1, nhead, V_HEAD_DIM),
        SENTINEL,
        dtype=torch.float32,
        device=dev,
    )
    attn_lse = torch.full(
        (f.partial_rows, 1, nhead, 1), SENTINEL, dtype=torch.float32, device=dev
    )
    o = torch.zeros((f.total_q, nhead, V_HEAD_DIM), dtype=torch.bfloat16, device=dev)
    final_lse = torch.zeros((f.total_q, nhead), dtype=torch.float32, device=dev)

    aiter.mla_decode_stage1_asm_fwd(
        f.q,
        f.kv_buffer,
        f.qo_indptr,
        f.kv_indptr,
        f.kv_indices,
        f.kv_last_page_lens,
        None,  # num_kv_splits_indptr -- unused in persistent mode
        f.work_meta_data,
        f.work_indptr,
        f.work_info_set,
        f.max_seqlen_qo,
        PAGE_SIZE,
        NHEAD_KV,
        f.sm_scale,
        logits,
        attn_lse,
        o,
        final_lse,
        f.q_scale,
        f.kv_scale,
        None,  # g_kv_indptr
        1,  # cp_world_size
        0,  # cp_rank
    )
    torch.cuda.synchronize()

    read_mask, num_tiles, empty_tiles, split_hist = _reduce_read_rows(f)
    read_rows = read_mask.to(dev)

    lse2d = attn_lse.reshape(f.partial_rows, nhead)
    out3d = logits.reshape(f.partial_rows, nhead, V_HEAD_DIM)

    untouched_lse = lse2d == SENTINEL
    untouched_out = (out3d == SENTINEL).all(dim=-1)
    nonfinite_lse = ~torch.isfinite(lse2d)
    nonfinite_out = ~torch.isfinite(out3d).all(dim=-1)

    rows = read_rows.unsqueeze(1).expand_as(untouched_lse)
    n_slots = int(rows.sum().item())
    stats = {
        "read_slots": n_slots,
        "untouched_lse": int((untouched_lse & rows).sum().item()),
        "untouched_out": int((untouched_out & rows).sum().item()),
        "nonfinite_lse": int((nonfinite_lse & rows).sum().item()),
        "nonfinite_out": int((nonfinite_out & rows).sum().item()),
    }
    finite_lse = lse2d[rows & ~untouched_lse & ~nonfinite_lse]
    stats["written_lse_max"] = (
        float(finite_lse.max().item()) if finite_lse.numel() else float("nan")
    )
    stats["written_lse_min"] = (
        float(finite_lse.min().item()) if finite_lse.numel() else float("nan")
    )

    print(
        f"\n[{case.id}] stage1 partial-slot attribution over the reduce read set\n"
        f"          reduce tiles={num_tiles} (empty {empty_tiles}) "
        f"splits_per_tile={dict(sorted(split_hist.items()))}\n"
        f"          read (row,head) slots      : {stats['read_slots']}\n"
        f"          H1 never written (LSE)     : {stats['untouched_lse']}\n"
        f"          H1 never written (output)  : {stats['untouched_out']}\n"
        f"          H2 written non-finite LSE  : {stats['nonfinite_lse']}\n"
        f"          H2 written non-finite out  : {stats['nonfinite_out']}\n"
        f"          written LSE range          : "
        f"[{stats['written_lse_min']:.6g}, {stats['written_lse_max']:.6g}]"
    )

    bad = (
        stats["untouched_lse"]
        or stats["untouched_out"]
        or stats["nonfinite_lse"]
        or stats["nonfinite_out"]
    )
    if bad:
        rows_lse = torch.nonzero(untouched_lse & rows)[:8].tolist()
        rows_nf = torch.nonzero(nonfinite_lse & rows)[:8].tolist()
        _dump(
            f"stage1_{case.id}",
            f,
            {
                "attribution": stats,
                "untouched_lse_slots": rows_lse,
                "nonfinite_lse_slots": rows_nf,
                "partial_lse": attn_lse.detach().cpu(),
            },
        )

    assert stats["untouched_lse"] == 0 and stats["untouched_out"] == 0, (
        f"H1 confirmed: stage1 left {stats['untouched_lse']} LSE and "
        f"{stats['untouched_out']} output slots at the sentinel, yet "
        f"mla_reduce_v1 reads them. In production those slots hold torch.empty "
        f"garbage. stats={stats}"
    )
    assert stats["nonfinite_lse"] == 0 and stats["nonfinite_out"] == 0, (
        f"H2 confirmed: mla_decode_stage1_asm_fwd wrote "
        f"{stats['nonfinite_lse']} non-finite LSE and "
        f"{stats['nonfinite_out']} non-finite output slots that the reduce "
        f"reads. stats={stats}"
    )


# ---------------------------------------------------------------------------
# 3. Full op: mla_decode_fwd end to end, assert the final LSE / output are
#    finite, and dump the whole call on failure.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_mla_decode_fwd_final_lse_is_finite(case):
    _skip_if_unsupported()
    f = _build(case)
    nhead = case.nhead

    o = torch.empty(
        (f.total_q, nhead, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    ).fill_(-1)

    logits, final_lse = aiter.mla.mla_decode_fwd(
        f.q,
        f.kv_buffer,
        o,
        f.qo_indptr,
        f.kv_indptr,
        f.kv_indices,
        f.kv_last_page_lens,
        f.max_seqlen_qo,
        PAGE_SIZE,
        NHEAD_KV,
        f.sm_scale,
        num_kv_splits=case.max_split_per_batch,
        work_meta_data=f.work_meta_data,
        work_indptr=f.work_indptr,
        work_info_set=f.work_info_set,
        reduce_indptr=f.reduce_indptr,
        reduce_final_map=f.reduce_final_map,
        reduce_partial_map=f.reduce_partial_map,
        q_scale=f.q_scale,
        kv_scale=f.kv_scale,
        intra_batch_mode=False,
        return_lse=True,
    )
    torch.cuda.synchronize()

    assert final_lse is not None, "return_lse=True must produce a final LSE"

    lse_bad = int((~torch.isfinite(final_lse)).sum().item())
    out_bad = int((~torch.isfinite(o.float())).sum().item())
    partial_bad = int((~torch.isfinite(logits)).sum().item())

    print(
        f"\n[{case.id}] {_nonfinite_report('final_lse', final_lse)}\n"
        f"          {_nonfinite_report('out', o.float())}\n"
        f"          {_nonfinite_report('partial_output', logits)}"
    )

    if lse_bad or out_bad:
        _dump(
            f"decode_{case.id}",
            f,
            {
                "final_lse": final_lse.detach().cpu(),
                "out": o.detach().cpu(),
                "partial_output_nonfinite_count": partial_bad,
                "counts": {
                    "final_lse_nonfinite": lse_bad,
                    "out_nonfinite": out_bad,
                },
            },
        )

    assert lse_bad == 0, (
        f"mla_decode_fwd returned {lse_bad} non-finite final LSE values "
        f"({partial_bad} non-finite partial-output elements upstream); "
        f"inputs dumped to {DUMP_DIR}"
    )
    assert out_bad == 0, (
        f"mla_decode_fwd returned {out_bad} non-finite output elements "
        f"({partial_bad} non-finite partial-output elements upstream); "
        f"inputs dumped to {DUMP_DIR}"
    )


# ---------------------------------------------------------------------------
# 4. Empty reduce tiles. The planner leaves reduce_final_map / reduce_partial_map
#    untouched for tiles with zero splits, and those buffers come from
#    torch.empty -- so an empty tile carries a garbage [q_start, q_end) range.
#    If mla_reduce_v1 did not skip empty tiles it would scribble over final
#    output rows that stage1 already wrote directly (partial_qo_loc == -1).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_empty_reduce_tiles_do_not_write_final_output(case):
    _skip_if_unsupported()
    f = _build(case)
    dev = "cuda"
    nhead = case.nhead

    ri = f.reduce_indptr.detach().cpu().to(torch.int64)
    rfm = f.reduce_final_map.detach().cpu().to(torch.int64)
    num_tiles = ri.numel() - 1
    empty = [t for t in range(num_tiles) if int(ri[t]) == int(ri[t + 1])]

    # Rows the non-empty tiles legitimately write.
    legit = torch.zeros(f.total_q, dtype=torch.bool)
    for t in range(num_tiles):
        if int(ri[t]) == int(ri[t + 1]):
            continue
        s, e = int(rfm[t, 0].item()), int(rfm[t, 1].item())
        s, e = max(0, s), min(f.total_q, e)
        if s < e:
            legit[s:e] = True

    # As planned, the empty tiles carry whatever torch.empty left behind -- in
    # practice values far outside [0, total_q), which cannot hit anything. That
    # makes the check vacuous, so rewrite them to the most destructive in-range
    # value instead: every empty tile now claims the whole output. If the kernel
    # honours reduce_final_map for a zero-split tile, every row gets clobbered.
    oob = 0
    for t in empty:
        s, e = int(rfm[t, 0].item()), int(rfm[t, 1].item())
        if s < 0 or e > f.total_q or s >= e:
            oob += 1
    if empty:
        idx = torch.tensor(empty, dtype=torch.int64, device=f.reduce_final_map.device)
        f.reduce_final_map[idx, 0] = 0
        f.reduce_final_map[idx, 1] = f.total_q
    at_risk = ~legit

    # Finite partials so nothing but the tile bookkeeping can produce garbage.
    partial_output = torch.zeros(
        (f.partial_rows, 1, nhead, V_HEAD_DIM), dtype=torch.float32, device=dev
    )
    partial_lse = torch.zeros(
        (f.partial_rows, 1, nhead, 1), dtype=torch.float32, device=dev
    )
    # kn_mla_reduce_v1 only accepts a bf16/fp16 final output, same as production.
    out = torch.full(
        (f.total_q, nhead, V_HEAD_DIM), SENTINEL, dtype=torch.bfloat16, device=dev
    )
    lse = torch.full((f.total_q, nhead), SENTINEL, dtype=torch.float32, device=dev)
    sent_bf16 = out[0, 0, 0].clone()  # SENTINEL after bf16 rounding

    aiter.mla_reduce_v1(
        partial_output,
        partial_lse,
        f.reduce_indptr,
        f.reduce_final_map,
        f.reduce_partial_map,
        f.max_seqlen_qo,
        case.max_split_per_batch,
        out,
        lse,
    )
    torch.cuda.synchronize()

    touched = ((out != sent_bf16).any(dim=-1) | (lse != SENTINEL)).any(dim=-1).cpu()
    clobbered = torch.nonzero(touched & at_risk).reshape(-1).tolist()

    print(
        f"\n[{case.id}] empty reduce tiles={len(empty)}/{num_tiles} "
        f"({oob} of them had an out-of-range final_map as planned; all were "
        f"rewritten to [0, {f.total_q}) to make the check meaningful)\n"
        f"          rows legitimately written : {int(legit.sum().item())}\n"
        f"          rows at risk             : {int(at_risk.sum().item())}\n"
        f"          rows actually clobbered   : {len(clobbered)}"
    )

    if empty:
        assert int(at_risk.sum().item()) > 0, "check is vacuous: nothing at risk"
    assert not clobbered, (
        f"{len(clobbered)} final-output rows outside any non-empty reduce tile "
        f"were written by mla_reduce_v1 (first: {clobbered[:16]}). A zero-split "
        f"tile's reduce_final_map is never initialised by the planner, so the "
        f"kernel must skip it rather than honour the range it finds."
    )


# ---------------------------------------------------------------------------
# 5. Seed sweep over the full op -- the planner layout changes with every seed,
#    which is the cheapest way to hit a rare tile/split configuration.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", SWEEP_SEEDS)
@pytest.mark.parametrize(
    "case", CASES[:4], ids=[c.id for c in CASES[:4]]
)  # skip the two 16K-context cases: too slow to sweep
def test_sweep_seeds_final_lse_is_finite(case, seed):
    _skip_if_unsupported()
    f = _build(case, seed=seed)
    nhead = case.nhead

    o = torch.empty(
        (f.total_q, nhead, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    ).fill_(-1)
    logits, final_lse = aiter.mla.mla_decode_fwd(
        f.q,
        f.kv_buffer,
        o,
        f.qo_indptr,
        f.kv_indptr,
        f.kv_indices,
        f.kv_last_page_lens,
        f.max_seqlen_qo,
        PAGE_SIZE,
        NHEAD_KV,
        f.sm_scale,
        num_kv_splits=case.max_split_per_batch,
        work_meta_data=f.work_meta_data,
        work_indptr=f.work_indptr,
        work_info_set=f.work_info_set,
        reduce_indptr=f.reduce_indptr,
        reduce_final_map=f.reduce_final_map,
        reduce_partial_map=f.reduce_partial_map,
        q_scale=f.q_scale,
        kv_scale=f.kv_scale,
        intra_batch_mode=False,
        return_lse=True,
    )
    torch.cuda.synchronize()

    lse_bad = int((~torch.isfinite(final_lse)).sum().item())
    out_bad = int((~torch.isfinite(o.float())).sum().item())
    partial_bad = int((~torch.isfinite(logits)).sum().item())
    if lse_bad or out_bad or partial_bad:
        print(
            f"\n[{case.id} seed={seed}] "
            f"final_lse_nonfinite={lse_bad} out_nonfinite={out_bad} "
            f"partial_nonfinite={partial_bad}"
        )
        _dump(
            f"sweep_{case.id}_seed{seed}",
            f,
            {
                "final_lse": final_lse.detach().cpu(),
                "out": o.detach().cpu(),
                "counts": {
                    "final_lse_nonfinite": lse_bad,
                    "out_nonfinite": out_bad,
                    "partial_output_nonfinite": partial_bad,
                },
            },
        )
    assert lse_bad == 0 and out_bad == 0 and partial_bad == 0, (
        f"seed {seed}: final_lse={lse_bad} out={out_bad} partial={partial_bad} "
        f"non-finite; inputs dumped to {DUMP_DIR}"
    )


# ---------------------------------------------------------------------------
# 6. Documented out-of-contract input: kv_len < q_len.
#
#    Under a causal mask the leading query rows of such a request see no key at
#    all, so their softmax is over an empty set and stage1 emits NaN. This is
#    NOT a kernel defect on in-contract input -- decode always has kv_len >=
#    q_len -- but it is a concrete way stage1 can put NaN into the LSE, so it is
#    pinned down here rather than left as speculation.
#
#    Note the affected work has partial_qo_loc == -1 (stage1 writes the final
#    output directly), so the NaN never travels through mla_reduce_v1.
# ---------------------------------------------------------------------------
def test_short_kv_makes_stage1_emit_nan_directly():
    _skip_if_unsupported()
    case = Case(
        batch=8,
        nhead=16,
        decode_qlen=4,
        ctx_len=8,
        max_split_per_batch=32,
        min_kv=1,  # deliberately allows kv_len < q_len
    )
    kv_len = 2
    # Forced through _build so the planner sees the same lengths as the kernels.
    f = _build(case, seed=0, kv_lens=[kv_len] * case.batch)
    nhead = case.nhead
    o = torch.empty(
        (f.total_q, nhead, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    ).fill_(-1)
    _, final_lse = aiter.mla.mla_decode_fwd(
        f.q,
        f.kv_buffer,
        o,
        f.qo_indptr,
        f.kv_indptr,
        f.kv_indices,
        f.kv_last_page_lens,
        f.max_seqlen_qo,
        PAGE_SIZE,
        NHEAD_KV,
        f.sm_scale,
        num_kv_splits=case.max_split_per_batch,
        work_meta_data=f.work_meta_data,
        work_indptr=f.work_indptr,
        work_info_set=f.work_info_set,
        reduce_indptr=f.reduce_indptr,
        reduce_final_map=f.reduce_final_map,
        reduce_partial_map=f.reduce_partial_map,
        q_scale=f.q_scale,
        kv_scale=f.kv_scale,
        intra_batch_mode=False,
        return_lse=True,
    )
    torch.cuda.synchronize()

    bad_rows = torch.nonzero((~torch.isfinite(final_lse)).any(dim=1)).reshape(-1)
    bad = set(bad_rows.cpu().tolist())

    # Under the causal mask the first (q_len - kv_len) rows of each request see
    # no key at all.
    masked = set()
    for b in range(case.batch):
        start = b * case.decode_qlen
        for i in range(max(0, case.decode_qlen - kv_len)):
            masked.add(start + i)

    print(
        f"\n[short-kv] kv_len={kv_len} q_len={case.decode_qlen}: "
        f"{len(bad)} of {f.total_q} final rows non-finite\n"
        f"          rows with no visible key : {len(masked)}\n"
        f"          non-finite rows          : {sorted(bad)[:16]}\n"
        f"          unexpected (finite-able) : {sorted(bad - masked)[:16]}"
    )

    assert bad, (
        "expected the masked-out query rows of an out-of-contract kv_len < "
        "q_len request to come back non-finite, got none"
    )
    assert not (bad - masked), (
        f"non-finite rows outside the fully-masked set: {sorted(bad - masked)[:16]} "
        f"-- those rows do have visible keys and must be finite"
    )


def _decode(f, case):
    """One mla_decode_fwd call; returns (out, final_lse, partial_output)."""
    o = torch.empty(
        (f.total_q, case.nhead, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    ).fill_(-1)
    logits, final_lse = aiter.mla.mla_decode_fwd(
        f.q,
        f.kv_buffer,
        o,
        f.qo_indptr,
        f.kv_indptr,
        f.kv_indices,
        f.kv_last_page_lens,
        f.max_seqlen_qo,
        PAGE_SIZE,
        NHEAD_KV,
        f.sm_scale,
        num_kv_splits=case.max_split_per_batch,
        work_meta_data=f.work_meta_data,
        work_indptr=f.work_indptr,
        work_info_set=f.work_info_set,
        reduce_indptr=f.reduce_indptr,
        reduce_final_map=f.reduce_final_map,
        reduce_partial_map=f.reduce_partial_map,
        q_scale=f.q_scale,
        kv_scale=f.kv_scale,
        intra_batch_mode=False,
        return_lse=True,
    )
    return o, final_lse, logits


# ---------------------------------------------------------------------------
# 7. The torch.empty hypothesis, tested directly.
#
#    aiter/mla.py allocates the partial output / LSE buffers with torch.empty,
#    so in a long-running server they come back holding whatever the caching
#    allocator last had there. Here we deliberately fill blocks of exactly those
#    shapes with NaN and free them immediately before the decode, so the
#    allocator hands the poisoned memory straight back to mla_decode_fwd.
#
#    The test is only meaningful if the poison actually lands, so it verifies
#    that the returned partial buffer still contains NaN in the slots stage1
#    does not write -- and then requires the final output to be finite anyway.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_poisoned_allocator_blocks_do_not_reach_the_output(case):
    _skip_if_unsupported()
    f = _build(case)
    nhead = case.nhead

    # Same shapes aiter/mla.py is about to ask torch.empty for.
    blocks = [
        torch.empty(
            (f.partial_rows, 1, nhead, V_HEAD_DIM), dtype=torch.float32, device="cuda"
        ),
        torch.empty(
            (f.partial_rows, 1, nhead, 1), dtype=torch.float32, device="cuda"
        ),
        torch.empty((f.total_q, nhead), dtype=torch.float32, device="cuda"),
    ]
    for b in blocks:
        b.fill_(float("nan"))
    torch.cuda.synchronize()
    del blocks  # back to the caching allocator, still full of NaN

    o, final_lse, partial = _decode(f, case)
    torch.cuda.synchronize()

    partial_nan = int((~torch.isfinite(partial)).sum().item())
    read_mask, _, _, _ = _reduce_read_rows(f)
    read = read_mask.to(partial.device)
    p2 = partial.reshape(f.partial_rows, nhead, V_HEAD_DIM)
    read_nan = int((~torch.isfinite(p2[read])).sum().item())
    unread_nan = int((~torch.isfinite(p2[~read])).sum().item())

    lse_bad = int((~torch.isfinite(final_lse)).sum().item())
    out_bad = int((~torch.isfinite(o.float())).sum().item())

    print(
        f"\n[{case.id}] poisoned-allocator decode\n"
        f"          partial rows total/read   : {f.partial_rows}/"
        f"{int(read_mask.sum().item())}\n"
        f"          NaN left in unread slots  : {unread_nan}  "
        f"(poison landed: {unread_nan > 0})\n"
        f"          NaN in slots reduce reads : {read_nan}\n"
        f"          final_lse non-finite      : {lse_bad}\n"
        f"          out non-finite            : {out_bad}"
    )

    if lse_bad or out_bad:
        _dump(
            f"poison_{case.id}",
            f,
            {
                "final_lse": final_lse.detach().cpu(),
                "out": o.detach().cpu(),
                "counts": {
                    "partial_nan_total": partial_nan,
                    "partial_nan_read": read_nan,
                    "partial_nan_unread": unread_nan,
                    "final_lse_nonfinite": lse_bad,
                    "out_nonfinite": out_bad,
                },
            },
        )

    assert read_nan == 0, (
        f"{read_nan} NaN elements survive in partial slots that mla_reduce_v1 "
        f"reads -- stage1 did not overwrite the poisoned allocator memory there"
    )
    assert lse_bad == 0 and out_bad == 0, (
        f"poisoned allocator memory reached the result: final_lse={lse_bad} "
        f"out={out_bad} non-finite; inputs dumped to {DUMP_DIR}"
    )


# ---------------------------------------------------------------------------
# 8. CUDA-graph capture + replay, the way a server actually drives this op:
#    buffers are allocated once, the planner rewrites the metadata in place
#    between steps, and the captured graph replays over the same memory.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("case", CASES[:4], ids=[c.id for c in CASES[:4]])
def test_cuda_graph_replay_final_lse_is_finite(case):
    _skip_if_unsupported()
    f = _build(case, seed=0, fixed_kv_total=True)
    nhead = case.nhead

    o = torch.empty(
        (f.total_q, nhead, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
    ).fill_(-1)

    def run():
        return aiter.mla.mla_decode_fwd(
            f.q,
            f.kv_buffer,
            o,
            f.qo_indptr,
            f.kv_indptr,
            f.kv_indices,
            f.kv_last_page_lens,
            f.max_seqlen_qo,
            PAGE_SIZE,
            NHEAD_KV,
            f.sm_scale,
            num_kv_splits=case.max_split_per_batch,
            work_meta_data=f.work_meta_data,
            work_indptr=f.work_indptr,
            work_info_set=f.work_info_set,
            reduce_indptr=f.reduce_indptr,
            reduce_final_map=f.reduce_final_map,
            reduce_partial_map=f.reduce_partial_map,
            q_scale=f.q_scale,
            kv_scale=f.kv_scale,
            intra_batch_mode=False,
            return_lse=True,
        )

    # Warm up on a side stream, then capture.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            run()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            logits, final_lse = run()
    except RuntimeError as exc:
        pytest.skip(f"mla_decode_fwd is not cuda-graph capturable here: {exc}")

    # Replay with fresh metadata each step, exactly like a server does: the
    # planner rewrites the same tensors in place, then the graph replays.
    worst = {"lse": 0, "out": 0, "partial": 0, "step": -1}
    steps = 12
    for step in range(1, steps + 1):
        g = _build(case, seed=100 + step, fixed_kv_total=True)
        f.work_meta_data.copy_(g.work_meta_data)
        f.work_indptr.copy_(g.work_indptr)
        f.work_info_set.copy_(g.work_info_set)
        f.reduce_indptr.copy_(g.reduce_indptr)
        f.reduce_final_map.copy_(g.reduce_final_map)
        f.reduce_partial_map.copy_(g.reduce_partial_map)
        f.kv_indptr.copy_(g.kv_indptr)
        f.kv_indices.copy_(g.kv_indices)
        f.kv_last_page_lens.copy_(g.kv_last_page_lens)
        f.q.copy_(g.q)
        f.kv_buffer.copy_(g.kv_buffer)
        del g

        graph.replay()
        torch.cuda.synchronize()

        lse_bad = int((~torch.isfinite(final_lse)).sum().item())
        out_bad = int((~torch.isfinite(o.float())).sum().item())
        p_bad = int((~torch.isfinite(logits)).sum().item())
        if lse_bad + out_bad + p_bad > worst["lse"] + worst["out"] + worst["partial"]:
            worst = {
                "lse": lse_bad,
                "out": out_bad,
                "partial": p_bad,
                "step": step,
            }
        if lse_bad or out_bad:
            _dump(
                f"graph_{case.id}_step{step}",
                f,
                {
                    "final_lse": final_lse.detach().cpu(),
                    "out": o.detach().cpu(),
                    "counts": {
                        "step": step,
                        "final_lse_nonfinite": lse_bad,
                        "out_nonfinite": out_bad,
                        "partial_output_nonfinite": p_bad,
                    },
                },
            )
            break

    print(
        f"\n[{case.id}] cuda-graph replay x{steps}: worst step={worst['step']} "
        f"final_lse={worst['lse']} out={worst['out']} partial={worst['partial']} "
        f"non-finite"
    )

    assert worst["lse"] == 0 and worst["out"] == 0, (
        f"cuda-graph replay produced non-finite results at step {worst['step']}: "
        f"final_lse={worst['lse']} out={worst['out']} partial={worst['partial']}; "
        f"inputs dumped to {DUMP_DIR}"
    )


# ---------------------------------------------------------------------------
# 9. The speculative-decode (MTP) metadata path. Instead of re-planning every
#    step, a server calls decode_update_mla_metadata_v1, which rewrites the
#    existing work / reduce metadata in place after the accepted-token count is
#    known, and folds qlen > 1 down to qlen == 1. Stale entries left behind by
#    that in-place rewrite are the most plausible way for the reduce to end up
#    reading a slot stage1 did not write, so check the same invariants there.
# ---------------------------------------------------------------------------
MTP_CASES = [c for c in CASES if c.decode_qlen > 1]


@pytest.mark.parametrize("case", MTP_CASES, ids=[c.id for c in MTP_CASES])
@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_mtp_metadata_update_keeps_the_read_set_written(case, seed):
    _skip_if_unsupported()
    from aiter.ops.attention import decode_update_mla_metadata_v1

    f = _build(case, seed=seed)  # planned for the draft step (qlen == 4)
    dev = "cuda"
    bs = case.batch
    nhead = case.nhead

    # Outcome of the speculative step, exactly as aiter's own example drives it.
    torch.manual_seed(1000 + seed)
    num_reject = torch.randint(0, case.decode_qlen, (bs,), dtype=torch.int32).to(dev)
    delta_csum = torch.cumsum(num_reject - 1, dim=0).to(torch.int32)
    kv_indptr = f.kv_indptr.clone()
    kv_indptr[1:] = kv_indptr[1:] - delta_csum
    kv_indptr = torch.clamp(kv_indptr, min=0)
    kv_indptr, _ = torch.cummax(kv_indptr, dim=0)  # keep it monotone
    num_page = int(kv_indptr[-1].item())
    if num_page <= bs:
        pytest.skip("degenerate KV pool after the reject bookkeeping")
    kv_indices = torch.randperm(num_page, dtype=torch.int32, device=dev)
    kv_buffer = (
        torch.randn(
            (num_page, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM),
            dtype=torch.bfloat16,
            device=dev,
        )
    ).to(dtypes.fp8)

    # The update folds qlen > 1 down to a single accepted token per request.
    qo_indptr = torch.arange(bs + 1, dtype=torch.int32, device=dev)

    for dtype_kw in (
        {"dtype_q": dtypes.fp8, "dtype_kv": dtypes.fp8},
        {"dtype_q_nope": dtypes.fp8, "dtype_kv_nope": dtypes.fp8},
    ):
        try:
            decode_update_mla_metadata_v1(
                qo_indptr,
                kv_indptr,
                f.kv_last_page_lens,
                nhead // NHEAD_KV,
                NHEAD_KV,
                False,
                f.work_meta_data,
                f.work_info_set,
                f.work_indptr,
                f.reduce_indptr,
                f.reduce_final_map,
                f.reduce_partial_map,
                page_size=PAGE_SIZE,
                kv_granularity=max(PAGE_SIZE, 16),
                max_seqlen_qo=1,
                num_reject_tokens=num_reject,
                **dtype_kw,
            )
            break
        except TypeError as exc:
            if "unexpected keyword" not in str(exc):
                raise
    else:
        pytest.skip("decode_update_mla_metadata_v1: no known dtype kwarg spelling")
    torch.cuda.synchronize()

    # Re-derive the invariants against the UPDATED metadata (qlen is now 1).
    f.qo_indptr = qo_indptr
    f.kv_indptr = kv_indptr
    f.kv_indices = kv_indices
    f.kv_buffer = kv_buffer
    f.total_q = bs
    f.max_seqlen_qo = 1
    f.partial_rows = int(f.reduce_partial_map.size(0))
    f.q = torch.randn(
        (bs, nhead, QK_HEAD_DIM), dtype=torch.bfloat16, device=dev
    ).to(dtypes.fp8)

    write_mask, num_works = _stage1_write_rows(f)
    read_mask, num_tiles, empty_tiles, split_hist = _reduce_read_rows(f)
    gap = int((read_mask & ~write_mask).sum().item())

    o = torch.empty(
        (bs, nhead, V_HEAD_DIM), dtype=torch.bfloat16, device=dev
    ).fill_(-1)
    logits, final_lse = aiter.mla.mla_decode_fwd(
        f.q,
        f.kv_buffer,
        o,
        f.qo_indptr,
        f.kv_indptr,
        f.kv_indices,
        f.kv_last_page_lens,
        1,
        PAGE_SIZE,
        NHEAD_KV,
        f.sm_scale,
        num_kv_splits=case.max_split_per_batch,
        work_meta_data=f.work_meta_data,
        work_indptr=f.work_indptr,
        work_info_set=f.work_info_set,
        reduce_indptr=f.reduce_indptr,
        reduce_final_map=f.reduce_final_map,
        reduce_partial_map=f.reduce_partial_map,
        q_scale=f.q_scale,
        kv_scale=f.kv_scale,
        intra_batch_mode=False,
        return_lse=True,
    )
    torch.cuda.synchronize()

    lse_bad = int((~torch.isfinite(final_lse)).sum().item())
    out_bad = int((~torch.isfinite(o.float())).sum().item())
    p_bad = int((~torch.isfinite(logits)).sum().item())

    print(
        f"\n[{case.id} seed={seed}] MTP metadata update "
        f"(rejects {num_reject.min().item()}..{num_reject.max().item()})\n"
        f"          works={num_works} tiles={num_tiles} (empty {empty_tiles}) "
        f"splits_per_tile={dict(sorted(split_hist.items()))}\n"
        f"          read-but-never-written rows : {gap}\n"
        f"          final_lse / out / partial non-finite : "
        f"{lse_bad} / {out_bad} / {p_bad}"
    )

    if gap or lse_bad or out_bad:
        _dump(
            f"mtp_{case.id}_seed{seed}",
            f,
            {
                "num_reject_tokens": num_reject.cpu(),
                "final_lse": final_lse.detach().cpu(),
                "out": o.detach().cpu(),
                "counts": {
                    "read_but_unwritten_rows": gap,
                    "final_lse_nonfinite": lse_bad,
                    "out_nonfinite": out_bad,
                    "partial_output_nonfinite": p_bad,
                },
            },
        )

    assert gap == 0, (
        f"after decode_update_mla_metadata_v1 the reduce reads {gap} partial "
        f"rows that no stage1 work writes -- stale metadata from the in-place "
        f"update; inputs dumped to {DUMP_DIR}"
    )
    assert lse_bad == 0 and out_bad == 0, (
        f"MTP metadata update produced non-finite results: final_lse={lse_bad} "
        f"out={out_bad} partial={p_bad}; inputs dumped to {DUMP_DIR}"
    )
