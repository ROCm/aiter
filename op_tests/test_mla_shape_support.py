# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import inspect
import itertools
import sys

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.ops.attention import (
    _MLA_ASM_STATUSES,
    _MLA_BACKENDS,
    MlaAsmStatus,
    MlaBackend,
    _mla_decode_head_plan,
    _mla_decode_launch_plan,
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


@pytest.fixture(autouse=True)
def _no_opus_env(monkeypatch):
    # The tests pin the default backends; AITER_MLA_USE_OPUS is read on every
    # query and launch, so an inherited value would select OPUS instead. Tests
    # that cover OPUS set it themselves.
    monkeypatch.delenv("AITER_MLA_USE_OPUS", raising=False)


def _dtype(name):
    return dtypes.fp8 if name == "fp8" else torch.bfloat16


def _gfx950_dtype(name):
    # For queries pinned to arch="gfx950": the host's dtypes.fp8 is e4m3fnuz on
    # gfx942 (and uint8 without a GPU), which would pick a different backend.
    return torch.float8_e4m3fn if name == "fp8" else torch.bfloat16


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


def _non_persistent_query(arch, heads, qlen):
    code, _, _, kernel = mla_decode_asm_query(
        arch, "fp8", "fp8", heads, 1, qlen, False, True, False, False
    )
    return _MLA_ASM_STATUSES.get(code, MlaAsmStatus.UNKNOWN), kernel


def test_non_persistent_asm():
    built = [
        a
        for a in ("gfx942", "gfx950")
        if _asm_status(a, 16, 1, False) is not MlaAsmStatus.ARCH_NOT_BUILT
    ]
    if not built:
        pytest.skip("no gfx942/gfx950 asm table built")
    if "gfx942" in built:
        status, _ = _non_persistent_query("gfx942", 64, 1)
        assert status is MlaAsmStatus.REMAP_REJECT
    if "gfx950" in built:
        status, kernel = _non_persistent_query("gfx950", 64, 1)
        assert status is MlaAsmStatus.OK, status
        assert kernel.endswith("qseqlen1_gqaratio64E"), kernel
        status, _ = _non_persistent_query("gfx950", 64, 2)
        assert status is MlaAsmStatus.REMAP_REJECT
        fp8 = torch.float8_e4m3fn
        s = get_mla_decode_shape_support(
            8, 1, fp8, fp8, persistent=False, arch="gfx950"
        )
        assert not s.planner_accepts and s.supported is True, s.reasons
        assert s.backend is MlaBackend.ASM


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
    # #6563: 32/64 heads also at qlen <= 4 (heads x qlen <= 128), 16 heads at
    # qlen <= 4 only, and bf16/bf16 runs the FlyDSL JIT kernel (no asm table).
    assert support(32, 4, flydsl_ps1=True).backend is MlaBackend.FLYDSL_PS1
    assert support(32, 4, flydsl_ps1=True).fold_factor == 1
    assert support(16, 5, flydsl_ps1=True).backend is MlaBackend.ASM
    bf16 = torch.bfloat16
    for n in (64, 128):
        s = get_mla_decode_shape_support(
            *(n, 1, bf16, bf16),
            arch="gfx1250",
            enable_experimental=False,
            flydsl_ps1=True,
        )
        assert (s.backend, s.fold_factor) == (MlaBackend.FLYDSL_PS1, 1), n
    monkeypatch.setenv("AITER_MLA_DECODE_PS1_ASM", "0")
    assert support(128, 1, flydsl_ps1=True).backend is MlaBackend.FLYDSL_PS1
    s = support(64, 1, flydsl_ps1=True, fast_mode=False)
    assert (s.backend, s.fold_factor, s.supported) == (MlaBackend.FLYDSL_PS1, 4, False)


def test_ps1_asm_cprr_kernel(monkeypatch):
    monkeypatch.delenv("AITER_MLA_DECODE_PS1_ASM", raising=False)
    fp8 = torch.float8_e4m3fn

    def support(world, rr):
        return get_mla_decode_shape_support(
            *(128, 1, fp8, fp8),
            arch="gfx1250",
            enable_experimental=False,
            flydsl_ps1=True,
            cp_world_size=world,
            cp_round_robin=rr,
        )

    s = support(4, False)
    assert s.backend is MlaBackend.ASM and s.supported is False, s.reasons
    s = support(1, True)
    assert s.backend is MlaBackend.PS1_FP8_ASM
    if s.asm_status is MlaAsmStatus.ARCH_NOT_BUILT:
        pytest.skip("no gfx1250 PS1 asm table built")
    assert s.supported is True and "cprr" not in s.asm_kernel, s.asm_kernel
    s = support(4, True)
    assert s.supported is True and "cprr" in s.asm_kernel, s.asm_kernel


def test_hk_backend():
    fp8 = torch.float8_e4m3fn

    def support(n, msq, **kw):
        return get_mla_decode_shape_support(
            n, msq, fp8, fp8, arch="gfx950", enable_experimental=True, **kw
        )

    s = support(128, 1)
    assert (s.backend, s.supported) == (MlaBackend.HK, None), s.reasons
    for n, msq, kw in (
        (128, 1, {"return_lse": True}),
        (64, 2, {"causal": False}),
        (128, 1, {"cp_world_size": 2, "cp_round_robin": True}),
    ):
        s = support(n, msq, **kw)
        assert (s.backend, s.supported) == (MlaBackend.HK, False), (n, msq, kw)


def test_gfx1250_asm_page_size():
    fp8 = torch.float8_e4m3fn

    def support(page_size):
        return get_mla_decode_shape_support(
            *(16, 1, fp8, fp8),
            persistent=False,
            page_size=page_size,
            arch="gfx1250",
        )

    s = support(1)
    assert s.asm_status is MlaAsmStatus.NO_KERNEL and s.supported is False
    assert support(64).asm_status in (MlaAsmStatus.OK, MlaAsmStatus.ARCH_NOT_BUILT)


class _PlanCaptured(Exception):
    pass


def _launch_plan_call(monkeypatch, n, msq, **call_kw):
    """The head-plan inputs mla_decode_fwd uses, captured before any launch."""
    seen = {}

    def capture(*args, **kw):
        bound = inspect.signature(_mla_decode_launch_plan).bind(*args, **kw)
        seen.update(bound.arguments)
        raise _PlanCaptured

    monkeypatch.setattr(aiter.mla, "_mla_decode_launch_plan", capture)
    fp8, dim = torch.float8_e4m3fn, KV_LORA_RANK + QK_ROPE
    i32 = torch.int32
    with pytest.raises(_PlanCaptured):
        aiter.mla.mla_decode_fwd(
            torch.zeros(msq, n, dim).to(fp8),
            torch.zeros(4, 1, 1, dim).to(fp8),
            torch.empty(msq, n, KV_LORA_RANK, dtype=torch.bfloat16),
            torch.tensor([0, msq], dtype=i32),
            torch.tensor([0, 4], dtype=i32),
            torch.arange(4, dtype=i32),
            torch.ones(1, dtype=i32),
            msq,
            num_kv_splits=1,
            work_meta_data=torch.zeros(1, dtype=i32),
            q_scale=torch.ones(1),
            kv_scale=torch.ones(1),
            **call_kw,
        )
    return seen


def test_cp_world_size_matches_launch(monkeypatch):
    fp8 = torch.float8_e4m3fn
    flags = {"arch": "gfx1250", "enable_experimental": False, "flydsl_ps1": True}
    g_kv_indptr = torch.tensor([0, 4], dtype=torch.int32)
    backends = {}
    for world, rr in ((1, False), (4, False), (4, True)):
        kw = _launch_plan_call(
            monkeypatch,
            64,
            1,
            cp_world_size=world,
            cp_rank=0,
            g_kv_indptr=g_kv_indptr if rr else None,
        )
        assert (kw["cp_world_size"], kw["cp_round_robin"]) == (world, rr)
        shape = ("num_heads", "max_seqlen_qo", "q_dtype", "kv_dtype")
        args = [kw.pop(k) for k in shape]
        launch = _mla_decode_head_plan(*args, **kw, **flags)
        query = get_mla_decode_shape_support(
            64, 1, fp8, fp8, cp_world_size=world, cp_round_robin=rr, **flags
        )
        assert query.backend is _MLA_BACKENDS[launch.backend], (world, rr)
        backends[world, rr] = query.backend
    assert backends == {
        (1, False): MlaBackend.FLYDSL_PS1,
        (4, False): MlaBackend.ASM,
        (4, True): MlaBackend.FLYDSL_PS1,
    }
    # Without round-robin, CP takes persistent gfx1250 stage 1, which has no kernel.
    s = get_mla_decode_shape_support(64, 1, fp8, fp8, cp_world_size=4, **flags)
    assert s.supported is False, s.reasons


def test_opus_cp(monkeypatch):
    monkeypatch.setenv("AITER_MLA_USE_OPUS", "1")
    get_mla_decode_shape_support.cache_clear()
    fp8, bf16 = torch.float8_e4m3fn, torch.bfloat16

    def support(dtype, world, rr):
        return get_mla_decode_shape_support(
            *(16, 1, dtype, dtype),
            cp_world_size=world,
            cp_round_robin=rr,
            arch="gfx950",
            enable_experimental=False,
        )

    try:
        for dtype, world, rr, expect in (
            (fp8, 1, False, None),
            (fp8, 4, True, None),
            (fp8, 4, False, False),
            (bf16, 4, True, False),
        ):
            s = support(dtype, world, rr)
            assert s.backend is MlaBackend.OPUS, s.backend
            assert s.supported is expect, (dtype, world, rr, s.reasons)
    finally:
        monkeypatch.delenv("AITER_MLA_USE_OPUS")
        get_mla_decode_shape_support.cache_clear()


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


def _run_decode(n, msq, d, lse, bs=2, persistent=True, v_head_dim=KV_LORA_RANK):
    qd = _dtype(d)
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
    o = torch.empty(total_q, n, v_head_dim, dtype=torch.bfloat16, device="cuda")
    if persistent:
        _, _, meta = _run_planner(bs, msq, n, qd, qd)
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


def _dim_rejected(s):
    return any("v_head_dim 512" in r for r in s.reasons)


@requires_gpu
def test_decode_asm_route_needs_v_head_dim_512():
    # mla_reduce_v1 has a (12, 128) case for prefill, but the stage-1 asm decode
    # kernels only take v_head_dim 512; the reducer row must not make 128 pass.
    def support(v_head_dim):
        return get_mla_decode_shape_support(
            *(12, 1, _gfx950_dtype("fp8"), _gfx950_dtype("fp8")),
            v_head_dim=v_head_dim,
            arch="gfx950",
            enable_experimental=False,
            flydsl_ps1=False,
        )

    s = support(128)
    assert s.backend is MlaBackend.ASM, s.backend
    assert s.supported is False and _dim_rejected(s), s.reasons
    s512 = support(512)
    assert s512.backend is MlaBackend.ASM, s512.backend
    assert not _dim_rejected(s512), s512.reasons
    # A build without the gfx950 asm table (e.g. gfx942 only) rejects every shape.
    if s512.asm_status is not MlaAsmStatus.ARCH_NOT_BUILT:
        assert s512.supported is not False, s512.reasons


def test_decode_opus_needs_v_head_dim_512(monkeypatch):
    # opus_mla_decode_fwd checks that the output's last dim is 512 as well.
    monkeypatch.setenv("AITER_MLA_USE_OPUS", "1")
    for v_dim in (128, KV_LORA_RANK):
        s = get_mla_decode_shape_support(
            *(12, 1, _gfx950_dtype("fp8"), _gfx950_dtype("fp8")),
            v_head_dim=v_dim,
            arch="gfx950",
            enable_experimental=False,
            flydsl_ps1=False,
        )
        assert s.backend is MlaBackend.OPUS, s.backend
        if v_dim == KV_LORA_RANK:
            assert s.supported is not False and not _dim_rejected(s), s.reasons
        else:
            assert s.supported is False, s.reasons
            assert any(
                r.startswith("opus_mla_decode_fwd") and "v_head_dim 512" in r
                for r in s.reasons
            ), s.reasons


@requires_gpu
def test_decode_non_persistent_asm_needs_v_head_dim_512():
    # Non-persistent decode runs the same mla_decode_stage1_asm_fwd, which rejects
    # an output head dim other than 512, so v_head_dim 128 fails there too.
    run = aiter.get_gfx() == "gfx950"
    for d in ("bf16", "fp8"):
        for v_dim, ok in ((128, False), (KV_LORA_RANK, True)):
            s = get_mla_decode_shape_support(
                *(16, 1, _gfx950_dtype(d), _gfx950_dtype(d)),
                v_head_dim=v_dim,
                persistent=False,
                arch="gfx950",
            )
            assert s.backend is MlaBackend.ASM, s.backend
            assert _dim_rejected(s) is not ok, s.reasons
            # A build without the gfx950 asm table (e.g. gfx942 only) has no
            # kernel to launch and reports every shape as unsupported.
            built = s.asm_status is not MlaAsmStatus.ARCH_NOT_BUILT
            if not ok:
                assert s.supported is False, s.reasons
            elif built:
                assert s.supported is not False, s.reasons
            if not (run and built):
                continue
            if ok:
                _run_decode(16, 1, d, False, persistent=False, v_head_dim=v_dim)
            else:
                with pytest.raises(RuntimeError, match="v_head_dim 512"):
                    _run_decode(16, 1, d, False, persistent=False, v_head_dim=v_dim)


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
