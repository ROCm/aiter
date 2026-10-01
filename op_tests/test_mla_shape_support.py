# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import itertools
import sys

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.ops.attention import (
    _MLA_ASM_STATUSES,
    MlaAsmStatus,
    MlaBackend,
    decode_update_mla_metadata_v1,
    get_mla_decode_shape_support,
    mla_decode_asm_query,
)

KV_LORA_RANK = 512
QK_ROPE = 64
KV_LENS = [100, 37]
MAX_SPLIT_PER_BATCH = 16

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() == 0,
    reason="needs a GPU",
)


def _dtype(name):
    return dtypes.fp8 if name == "fp8" else torch.bfloat16


def _fold(arch, d, n, msq=1):
    t = torch.float8_e4m3fn if d == "fp8" else torch.bfloat16
    s = get_mla_decode_shape_support(
        n, msq, t, t, arch=arch, enable_experimental=False, flydsl_ps1=False
    )
    return s.fold_factor if s.planner_accepts else None


def test_planner_verdict():
    assert [_fold("gfx950", "fp8", n) for n in (48, 64, 96, 128)] == [3, 1, 1, 1]
    assert _fold("gfx950", "fp8", 96, 6) == 1 and _fold("gfx950", "fp8", 96, 7) == 6
    assert _fold("gfx950", "fp8", 24) is None
    assert [_fold("gfx950", "bf16", n) for n in (40, 48, 144)] == [1, 1, 1]
    assert _fold("gfx942", "fp8", 64) == 1 and _fold("gfx942", "fp8", 64, 2) == 4
    assert _fold("gfx942", "bf16", 128) == 8
    assert _fold("gfx1250", "fp8", 64) == 4


def _asm_status(arch, heads, qlen, lse):
    code = mla_decode_asm_query(
        arch, "fp8", "fp8", heads, 1, qlen, True, True, lse, False
    )
    return _MLA_ASM_STATUSES.get(code[0], MlaAsmStatus.UNKNOWN)


def test_asm_lse_availability():
    built = [
        a
        for a in ("gfx942", "gfx950")
        if _asm_status(a, 16, 1, False) is not MlaAsmStatus.ARCH_NOT_BUILT
    ]
    if not built:
        pytest.skip("no gfx942/gfx950 asm table built")
    if "gfx942" in built:
        assert _asm_status("gfx942", 128, 1, True) is not MlaAsmStatus.OK
        assert _asm_status("gfx942", 64, 1, True) is MlaAsmStatus.OK
    if "gfx950" in built:
        assert _asm_status("gfx950", 16, 2, True) is not MlaAsmStatus.OK
        for heads in (32, 64, 128):
            assert _asm_status("gfx950", heads, 2, True) is MlaAsmStatus.OK, heads


def test_gfx1250_flydsl_ps1_backend(monkeypatch):
    monkeypatch.delenv("AITER_MLA_DECODE_PS1_ASM", raising=False)
    fp8 = torch.float8_e4m3fn

    def support(n, msq, **kw):
        return get_mla_decode_shape_support(
            n, msq, fp8, fp8, arch="gfx1250", enable_experimental=False, **kw
        )

    assert [support(96, q, flydsl_ps1=True).fold_factor for q in (1, 7)] == [1, 1]
    assert support(96, 1, flydsl_ps1=False).fold_factor == 6
    s = support(96, 4, flydsl_ps1=True, return_lse=True)
    assert s.backend is MlaBackend.PS1_FP8_ASM
    assert s.asm_status in (MlaAsmStatus.OK, MlaAsmStatus.ARCH_NOT_BUILT)
    assert s.supported is (s.asm_status is MlaAsmStatus.OK), s.reasons
    s = support(64, 1, flydsl_ps1=True, cp_round_robin=True)
    assert s.backend is MlaBackend.FLYDSL_PS1
    s = support(96, 1, flydsl_ps1=True, fast_mode=False)
    assert (s.backend, s.fold_factor) == (MlaBackend.PS1_FP8_ASM, 6)
    assert s.supported is False, s.reasons
    monkeypatch.setenv("AITER_MLA_DECODE_PS1_ASM", "0")
    assert support(128, 1, flydsl_ps1=True).backend is MlaBackend.FLYDSL_PS1
    s = support(64, 1, flydsl_ps1=True, fast_mode=False)
    assert (s.backend, s.fold_factor, s.supported) == (MlaBackend.FLYDSL_PS1, 4, False)


def _indptrs(bs, msq):
    kv = KV_LENS[:bs]
    qo = torch.arange(bs + 1, dtype=torch.int32, device="cuda") * msq
    kv_indptr = torch.tensor(
        [0] + list(itertools.accumulate(kv)), dtype=torch.int32, device="cuda"
    )
    last = torch.ones(bs, dtype=torch.int32, device="cuda")
    return qo, kv_indptr, last


def _run_planner(bs, msq, n, q, kv):
    kw = {"fast_mode": True, "intra_batch_mode": False}
    kw["max_split_per_batch"] = MAX_SPLIT_PER_BATCH
    info = aiter.get_mla_metadata_info_v1(
        bs, msq, n, q, kv, is_sparse=False, num_kv_splits=MAX_SPLIT_PER_BATCH, **kw
    )
    bufs = [
        (
            torch.zeros(s, dtype=t, device="cuda")
            if t == torch.uint64
            else torch.full(
                s if isinstance(s, tuple) else (s,), -7, dtype=t, device="cuda"
            )
        )
        for s, t in info
    ]
    wmd, wind, winfo, rind, rfm, rpm = bufs
    meta = (wmd, winfo, wind, rind, rfm, rpm)
    qo, kv_indptr, last = _indptrs(bs, msq)
    aiter.get_mla_metadata_v1(
        *(qo, kv_indptr, last, n, 1, True, *meta),
        page_size=1,
        kv_granularity=16,
        max_seqlen_qo=msq,
        uni_seqlen_qo=msq,
        dtype_q_nope=q,
        dtype_kv_nope=kv,
        **kw,
    )
    torch.cuda.synchronize()
    written = wind[wind != -7]
    n_work = int(written.max().item()) if written.numel() else 0
    n_bid = int(winfo[:n_work, 0].max().item()) + 1 if n_work else 0
    return n_bid, n_work, meta


@requires_gpu
def test_planner_fold_agreement():
    for n, msq, d in itertools.product(
        (16, 48, 64, 96, 128), (1, 2, 7), ("fp8", "bf16")
    ):
        s = get_mla_decode_shape_support(n, msq, _dtype(d), _dtype(d))
        if not s.planner_accepts:
            continue
        n_bid, _, _ = _run_planner(1, msq, n, _dtype(d), _dtype(d))
        assert n_bid == s.fold_factor * s.seqlen_fold, (n, msq, d, n_bid)


def _run_decode(n, msq, d, lse, bs=2):
    qd = _dtype(d)
    _, _, meta = _run_planner(bs, msq, n, qd, qd)
    qo, kv_indptr, last = _indptrs(bs, msq)
    total_q, total_kv = bs * msq, int(kv_indptr[-1].item())
    dim = KV_LORA_RANK + QK_ROPE
    g = torch.Generator().manual_seed(1234)
    qt = torch.randn(total_q, n, dim, generator=g).to("cuda").to(qd)
    kvb = torch.randn(total_kv, 1, 1, dim, generator=g).to("cuda").to(qd)
    kv_indices = torch.randperm(total_kv, generator=g).to(torch.int32).to("cuda")
    kw = {}
    if d == "fp8":
        kw["q_scale"] = torch.ones(1, dtype=torch.float, device="cuda")
        kw["kv_scale"] = torch.ones(1, dtype=torch.float, device="cuda")
    o = torch.empty(total_q, n, KV_LORA_RANK, dtype=torch.bfloat16, device="cuda")
    names = ("work_meta_data", "work_info_set", "work_indptr")
    names += ("reduce_indptr", "reduce_final_map", "reduce_partial_map")
    kw.update(zip(names, meta))
    aiter.mla.mla_decode_fwd(
        *(qt, kvb, o, qo, kv_indptr, kv_indices, last, msq),
        page_size=1,
        nhead_kv=1,
        sm_scale=1.0 / dim**0.5,
        return_lse=lse,
        causal=True,
        **kw,
    )
    torch.cuda.synchronize()


@requires_gpu
def test_decode_agreement():
    bad = []
    for n, msq, d, lse in itertools.product(
        (16, 48, 64, 128), (1, 2), ("fp8", "bf16"), (False, True)
    ):
        s = get_mla_decode_shape_support(n, msq, _dtype(d), _dtype(d), return_lse=lse)
        if not s.planner_accepts or not s.reduce_supported:
            assert s.supported is False
            continue
        try:
            _run_decode(n, msq, d, lse)
            ok = True
        except (RuntimeError, AttributeError):
            ok = False
        if ok != s.supported:
            bad.append((n, msq, d, lse, s.backend.value, s.supported))
    assert not bad, bad


@requires_gpu
def test_decode_update_keeps_planner_work_info():
    bs, n, qd = len(KV_LENS), 64, dtypes.fp8
    for msq in (1, 2, 4):
        _, n_work, meta = _run_planner(bs, msq, n, qd, qd)
        winfo = meta[1]
        planned = winfo[:n_work].clone()
        qo, kv_indptr, last = _indptrs(bs, msq)
        decode_update_mla_metadata_v1(
            *(qo, kv_indptr, last, n, 1, True, *meta),
            max_seqlen_qo=msq,
            dtype_q=qd,
            dtype_kv=qd,
            num_reject_tokens=torch.ones(bs, dtype=torch.int32, device="cuda"),
        )
        torch.cuda.synchronize()
        cols = [0, 4, 5, 6]
        assert torch.equal(winfo[:n_work, cols], planned[:, cols]), msq


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", *sys.argv[1:]]))
