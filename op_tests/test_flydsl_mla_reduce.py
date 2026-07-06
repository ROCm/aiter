# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pytest correctness coverage for the FlyDSL MLA reduce kernel.

Irregular-first: most cases use production-shaped metadata (variable per-tile
``n_splits``, gapped ``reduce_partial_map``, MLDS tier boundary, empty tiles).
Uniform/dense layouts are kept only as a small smoke layer. This mirrors real
split-KV decode, where every tile can need a different split count and the
partial buffer is a sparsely-indexed pool.
"""

import pytest
import torch

from op_tests.flydsl_mla_reduce_common import (
    build_degenerate_inputs,
    build_inputs,
    build_irregular_inputs,
    build_serving_mapped_slack_inputs,
    build_serving_sparse_grid_inputs,
    build_serving_stale_indptr_inputs,
    build_serving_true_oob_inputs,
    hip_ref,
    hip_ref_like_fout,
    make_runner,
    run_cudagraph_replay,
    torch_ref,
    torch_ref_gather,
)
from aiter.ops.flydsl.kernels.mla_reduce import select_tier

# DeepSeek shape: HIP MLA_REDUCE_ROUTER has a Dv=512 template, so these compare
# against the HIP kernel directly.
_HIP_SHAPE = (128, 512)
# GLM-5.2 production shape (tp=8). HIP has no Dv=256 template, so these compare
# against the torch online-softmax reference.
_GLM_SHAPE = (8, 256)

# Irregular scenarios: (id, splits_per_tile, gap_stride, M).
_IRREGULAR_SCENARIOS = [
    ("tier_mismatch", [8, 304], 1, 1),       # tile 0 small, tile 1 forces MLDS tier
    ("variable_splits", [4, 32, 8, 64], 1, 1),  # mixed per-tile counts
    ("gapped_pmap", [8, 8, 8, 8], 4, 1),     # non-dense gather rows
    ("empty_middle", [8, 0, 16, 8], 1, 1),   # empty tile + garbage final map
    ("mlds_boundary", [300], 1, 1),          # MLDS tier just under cap
    ("mlds_max", [304], 1, 1),               # LDS_MAX_SPLITS
    ("mtp_irregular", [8, 32, 16], 2, 4),    # MTP (M>1) + gaps
    ("pool_oversize", [8, 304], 8, 1),       # large slack in partial pool
]

# fp16 (in addition to the bf16 default) only on the most layout-sensitive cases,
# to keep the matrix small.
_HIP_FP16_IDS = {"tier_mismatch", "gapped_pmap"}
_TORCH_FP16_IDS = {"tier_mismatch", "mlds_max"}


def _expand(fp16_ids):
    cases = []
    for name, spt, gap, M in _IRREGULAR_SCENARIOS:
        cases.append((name, spt, gap, M, "bf16"))
        if name in fp16_ids:
            cases.append((name, spt, gap, M, "fp16"))
    return cases


_HIP_CASES = _expand(_HIP_FP16_IDS)
_TORCH_CASES = _expand(_TORCH_FP16_IDS)
_HIP_IDS = [f"{n}_{dt}" for n, _, _, _, dt in _HIP_CASES]
_TORCH_IDS = [f"{n}_{dt}" for n, _, _, _, dt in _TORCH_CASES]

# Uniform/dense smoke: one tile count, M=1, bf16 only. Just enough to cover each
# compile tier on both reference paths.
_SMOKE_TILES = 4
_SMOKE_CASES = [
    (_HIP_SHAPE, "hip", 2),     # simple
    (_HIP_SHAPE, "hip", 8),     # m64
    (_HIP_SHAPE, "hip", 64),    # m256
    (_HIP_SHAPE, "hip", 256),   # m256 upper
    (_GLM_SHAPE, "torch", 2),   # simple (GLM)
    (_GLM_SHAPE, "torch", 8),   # massive (GLM)
    (_GLM_SHAPE, "torch", 32),  # production split cap
    (_GLM_SHAPE, "torch", 256),  # stress
]
_SMOKE_IDS = [f"H{H}_Dv{Dv}_{ref}_s{S}" for (H, Dv), ref, S in _SMOKE_CASES]

# CUDA-graph replay: highest-risk irregular fixtures, to surface replay-only
# faults. (id, shape, ref, splits_per_tile, gap_stride, M).
_GRAPH_CASES = [
    ("tier_mismatch", _GLM_SHAPE, "torch", [8, 304], 1, 1),
    ("gapped_pmap", _HIP_SHAPE, "hip", [8, 8, 8, 8], 4, 1),
    ("empty_middle", _GLM_SHAPE, "torch", [8, 0, 16, 8], 1, 1),
    ("mlds_max", _GLM_SHAPE, "torch", [304], 1, 1),
    ("mtp_irregular", _GLM_SHAPE, "torch", [8, 32, 16], 2, 4),
]
_GRAPH_IDS = [c[0] for c in _GRAPH_CASES]

_DEGEN_TILES = [2, 4]


def _require_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")


def _out_dtype(dt: str) -> torch.dtype:
    return torch.bfloat16 if dt == "bf16" else torch.float16


def _out_atol(dt: str) -> float:
    return 6.3e-2 if dt == "bf16" else 2e-3


def _assert_close(fout, flse, ref_out, ref_lse, dt):
    atol = _out_atol(dt)
    out_err = (fout.float() - ref_out.float()).abs().max().item()
    lse_err = (flse - ref_lse).abs().max().item()
    assert out_err <= atol, f"out max_abs_err={out_err:.3e} > {atol}"
    assert lse_err <= 1e-3, f"lse max_abs_err={lse_err:.3e}"


def _masking_ref(po, pl, indptr, fmap, pmap, H, Dv, out_dtype, meta, M=1):
    return torch_ref_gather(
        po,
        pl,
        indptr,
        fmap,
        pmap,
        H,
        Dv,
        out_dtype,
        M,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )


def _run_guarded(
    po,
    pl,
    indptr,
    pmap,
    fmap,
    fout,
    flse,
    H,
    Dv,
    dt,
    meta,
    *,
    disable_guards=False,
    M=1,
):
    fout.zero_()
    flse.zero_()
    if meta.get("fout_slack_seed") is not None:
        sq = meta["store_slack_q"]
        fout[sq:].fill_(meta["fout_slack_seed"])
    run = make_runner(
        po,
        pl,
        indptr,
        pmap,
        fmap,
        fout,
        flse,
        H,
        Dv,
        dt,
        True,
        M,
        disable_guards=disable_guards,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    run()
    torch.cuda.synchronize()
    return fout.clone(), flse.clone()


def _assert_gather_differential(po, pl, indptr, fmap, pmap, fout, flse, H, Dv, dt, meta):
    ref_out, ref_lse = _masking_ref(po, pl, indptr, fmap, pmap, H, Dv, _out_dtype(dt), meta)
    on_out, on_lse = _run_guarded(
        po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, meta, disable_guards=False
    )
    off_out, off_lse = _run_guarded(
        po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, meta, disable_guards=True
    )
    _assert_close(on_out[: meta["logical_final_rows"]], on_lse[: meta["logical_final_rows"]], ref_out[: meta["logical_final_rows"]], ref_lse[: meta["logical_final_rows"]], dt)
    atol = _out_atol(dt)
    q_row = meta["gather_q_row"]
    gather_err = (off_out[q_row].float() - ref_out[q_row].float()).abs().max().item()
    assert gather_err > 5 * atol, (
        f"gather guard differential failed: guards-OFF row {q_row} "
        f"max_abs_err={gather_err:.3e} <= {5 * atol}"
    )


def _assert_store_differential(po, pl, indptr, fmap, pmap, fout, flse, H, Dv, dt, meta):
    ref_out, ref_lse = _masking_ref(po, pl, indptr, fmap, pmap, H, Dv, _out_dtype(dt), meta)
    on_out, on_lse = _run_guarded(
        po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, meta, disable_guards=False
    )
    off_out, _ = _run_guarded(
        po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, meta, disable_guards=True
    )
    _assert_close(on_out[: meta["logical_final_rows"]], on_lse[: meta["logical_final_rows"]], ref_out[: meta["logical_final_rows"]], ref_lse[: meta["logical_final_rows"]], dt)
    sq = meta["store_slack_q"]
    seed = meta["fout_slack_seed"]
    on_slack_err = (on_out[sq:].float() - seed).abs().max().item()
    assert on_slack_err <= _out_atol(dt), f"guards-ON mutated slack: err={on_slack_err:.3e}"
    atol = _out_atol(dt)
    off_slack_err = (off_out[sq:].float() - seed).abs().max().item()
    assert off_slack_err > 5 * atol, (
        f"store guard differential failed: guards-OFF slack "
        f"max_abs_err={off_slack_err:.3e} <= {5 * atol}"
    )


def _run_irregular(spt, gap, M, H, Dv, dt):
    """Build irregular inputs, run the kernel, return (po, pl, indptr, fmap, pmap,
    fout, flse)."""
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = build_irregular_inputs(
        spt, H, Dv, out_dtype, M=M, gap_stride=gap
    )
    fout.zero_()
    flse.zero_()
    run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, True, M)
    run()
    torch.cuda.synchronize()
    return po, pl, indptr, fmap, pmap, fout, flse


@pytest.mark.parametrize("case", _HIP_CASES, ids=_HIP_IDS)
def test_flydsl_mla_reduce_irregular_vs_hip(case):
    """Irregular metadata matches HIP kn_mla_reduce_v1 (DeepSeek shape, Dv=512)."""
    _require_cuda()
    name, spt, gap, M, dt = case
    H, Dv = _HIP_SHAPE
    po, pl, indptr, fmap, pmap, fout, flse = _run_irregular(spt, gap, M, H, Dv, dt)
    ref_out, ref_lse = hip_ref(
        po, pl, indptr, fmap, pmap, len(spt), H, Dv, _out_dtype(dt), M
    )
    _assert_close(fout, flse, ref_out, ref_lse, dt)


@pytest.mark.parametrize("case", _TORCH_CASES, ids=_TORCH_IDS)
def test_flydsl_mla_reduce_irregular_vs_torch_ref(case):
    """Irregular metadata matches the gather-based torch ref (GLM-5.2, Dv=256)."""
    _require_cuda()
    name, spt, gap, M, dt = case
    H, Dv = _GLM_SHAPE
    po, pl, indptr, fmap, pmap, fout, flse = _run_irregular(spt, gap, M, H, Dv, dt)
    ref_out, ref_lse = torch_ref_gather(
        po, pl, indptr, fmap, pmap, H, Dv, _out_dtype(dt), M
    )
    _assert_close(fout, flse, ref_out, ref_lse, dt)


@pytest.mark.parametrize("case", _SMOKE_CASES, ids=_SMOKE_IDS)
def test_flydsl_mla_reduce_uniform_smoke(case):
    """Dense/uniform smoke: each compile tier on both reference paths."""
    _require_cuda()
    (H, Dv), ref, S = case
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = build_inputs(
        _SMOKE_TILES, S, H, Dv, out_dtype
    )
    fout.zero_()
    flse.zero_()
    run = make_runner(
        po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, True,
        tier=select_tier(S),
    )
    run()
    torch.cuda.synchronize()
    if ref == "hip":
        ref_out, ref_lse = hip_ref(
            po, pl, indptr, fmap, pmap, _SMOKE_TILES, H, Dv, out_dtype
        )
    else:
        ref_out, ref_lse = torch_ref(po, pl, _SMOKE_TILES, S, H, Dv, out_dtype)
    _assert_close(fout, flse, ref_out, ref_lse, dt)


@pytest.mark.parametrize("case", _GRAPH_CASES, ids=_GRAPH_IDS)
def test_flydsl_mla_reduce_cudagraph_replay(case):
    """Irregular metadata stays correct under CUDA-graph capture + replay (the
    serving failure mode); no GPU fault and output matches the reference."""
    _require_cuda()
    name, (H, Dv), ref, spt, gap, M = case
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = build_irregular_inputs(
        spt, H, Dv, out_dtype, M=M, gap_stride=gap
    )
    fout.zero_()
    flse.zero_()
    run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, True, M)
    run_cudagraph_replay(run)
    if ref == "hip":
        ref_out, ref_lse = hip_ref(
            po, pl, indptr, fmap, pmap, len(spt), H, Dv, out_dtype, M
        )
    else:
        ref_out, ref_lse = torch_ref_gather(
            po, pl, indptr, fmap, pmap, H, Dv, out_dtype, M
        )
    _assert_close(fout, flse, ref_out, ref_lse, dt)


def test_flydsl_mla_reduce_small_split_cudagraph_replay():
    """Uniform 32-split tiles under cudagraph replay (runtime NLSE=1 path)."""
    _require_cuda()
    H, Dv = _GLM_SHAPE
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    spt = [32] * 4
    po, pl, indptr, fmap, pmap, fout, flse = build_irregular_inputs(
        spt, H, Dv, out_dtype, M=1, gap_stride=1
    )
    fout.zero_()
    flse.zero_()
    run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, True, M=1)
    run_cudagraph_replay(run)
    ref_out, ref_lse = torch_ref_gather(
        po, pl, indptr, fmap, pmap, H, Dv, out_dtype, M=1
    )
    _assert_close(fout, flse, ref_out, ref_lse, dt)


_SERVING_SHAPE = (16, 512)


@pytest.mark.slow
def test_serving_sparse_grid_vs_hip():
    """Jin Tao batch=8 layout: 16384-tile grid, 8 active tiles, garbage tail."""
    _require_cuda()
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = build_serving_sparse_grid_inputs(
        *_SERVING_SHAPE, out_dtype=out_dtype
    )
    fout.zero_()
    flse.zero_()
    run = make_runner(
        po, pl, indptr, pmap, fmap, fout, flse, *_SERVING_SHAPE, dt, True
    )
    run()
    torch.cuda.synchronize()
    ref_out, ref_lse = hip_ref_like_fout(po, pl, indptr, fmap, pmap, fout, flse)
    _assert_close(fout, flse, ref_out, ref_lse, dt)


@pytest.mark.slow
def test_serving_sparse_grid_cudagraph_replay():
    """16384-tile serving grid under CUDA-graph replay (prod failure mode)."""
    _require_cuda()
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = build_serving_sparse_grid_inputs(
        *_SERVING_SHAPE, out_dtype=out_dtype
    )
    fout.zero_()
    flse.zero_()
    run = make_runner(
        po, pl, indptr, pmap, fmap, fout, flse, *_SERVING_SHAPE, dt, True
    )
    run_cudagraph_replay(run)
    ref_out, ref_lse = hip_ref_like_fout(po, pl, indptr, fmap, pmap, fout, flse)
    _assert_close(fout, flse, ref_out, ref_lse, dt)


# ---------------------------------------------------------------------------
# Split-K cooperative reduction (opt-in): low-tile / high-split decode case.
# b1_s128 = 1 active tile x H=16 heads = 16 active blocks / 304 CUs, each
# serially reducing 128 splits. Split-K fans each head's 128 splits across K
# blocks (partial online-softmax) + a cheap combine. These cases prove the
# path stays numerically correct (vs torch ref AND HIP) and capture-safe.
# ---------------------------------------------------------------------------
_SPLITK_H, _SPLITK_DV = 16, 512
_SPLITK_GRID = 16384


def _build_splitk_b1_s128(out_dtype):
    """1 active tile x 128 splits in a 16384-tile serving grid (b1_s128)."""
    spt = [128] + [0] * (_SPLITK_GRID - 1)
    return build_irregular_inputs(
        spt, _SPLITK_H, _SPLITK_DV, out_dtype, M=1, gap_stride=1, pool_slack=0
    )


def _assert_splitk_engages(indptr, monkeypatch):
    from aiter.ops.flydsl.kernels.mla_reduce import plan_splitk

    diffs = indptr[1:] - indptr[:-1]
    engage, K, num_slots = plan_splitk(
        active_tiles=int((diffs > 1).sum().item()),
        H=_SPLITK_H,
        max_seqlen_q=1,
        max_splits=int(diffs.max().item()),
        num_cu=304,
    )
    assert engage, "split-K did not engage for b1_s128 (test is meaningless)"
    return K, num_slots


@pytest.mark.slow
@pytest.mark.parametrize("K", [4, 8, 16])
def test_splitk_b1_s128_vs_torch_ref(monkeypatch, K):
    """Split-K partial+combine matches the gather-based torch reference for the
    low-tile/high-split decode case, across split factors K."""
    _require_cuda()
    monkeypatch.setenv("AITER_MLA_REDUCE_SPLITK", "1")
    monkeypatch.setenv("MLA_SPLITK_FACTOR", str(K))
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = _build_splitk_b1_s128(out_dtype)
    _assert_splitk_engages(indptr, monkeypatch)
    fout.zero_()
    flse.zero_()
    run = make_runner(
        po, pl, indptr, pmap, fmap, fout, flse, _SPLITK_H, _SPLITK_DV, dt, True,
        tier=None,
    )
    run()
    torch.cuda.synchronize()
    ref_out, ref_lse = torch_ref_gather(
        po, pl, indptr, fmap, pmap, _SPLITK_H, _SPLITK_DV, out_dtype, 1
    )
    _assert_close(fout, flse, ref_out, ref_lse, dt)


@pytest.mark.slow
def test_splitk_b1_s128_vs_hip(monkeypatch):
    """Split-K matches the production HIP kn_mla_reduce_v1 (Dv=512 template)."""
    _require_cuda()
    monkeypatch.setenv("AITER_MLA_REDUCE_SPLITK", "1")
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = _build_splitk_b1_s128(out_dtype)
    _assert_splitk_engages(indptr, monkeypatch)
    fout.zero_()
    flse.zero_()
    run = make_runner(
        po, pl, indptr, pmap, fmap, fout, flse, _SPLITK_H, _SPLITK_DV, dt, True,
        tier=None,
    )
    run()
    torch.cuda.synchronize()
    ref_out, ref_lse = hip_ref_like_fout(po, pl, indptr, fmap, pmap, fout, flse)
    _assert_close(fout, flse, ref_out, ref_lse, dt)


@pytest.mark.slow
def test_splitk_b1_s128_cudagraph_replay(monkeypatch):
    """Split-K (2-kernel) stays correct under CUDA-graph capture + replay, with
    a pre-allocated scratch buffer — the capture-safety proof."""
    _require_cuda()
    monkeypatch.setenv("AITER_MLA_REDUCE_SPLITK", "1")
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = _build_splitk_b1_s128(out_dtype)
    _assert_splitk_engages(indptr, monkeypatch)
    fout.zero_()
    flse.zero_()
    run = make_runner(
        po, pl, indptr, pmap, fmap, fout, flse, _SPLITK_H, _SPLITK_DV, dt, True,
        tier=None,
    )
    run_cudagraph_replay(run)
    ref_out, ref_lse = torch_ref_gather(
        po, pl, indptr, fmap, pmap, _SPLITK_H, _SPLITK_DV, out_dtype, 1
    )
    _assert_close(fout, flse, ref_out, ref_lse, dt)


@pytest.mark.slow
def test_splitk_disabled_by_default(monkeypatch):
    """Without the env flag, plan_splitk never engages (default path untouched)."""
    from aiter.ops.flydsl.kernels.mla_reduce import plan_splitk, splitk_enabled

    monkeypatch.delenv("AITER_MLA_REDUCE_SPLITK", raising=False)
    assert not splitk_enabled()
    engage, _, _ = plan_splitk(
        active_tiles=1, H=16, max_seqlen_q=1, max_splits=128, num_cu=304
    )
    assert not engage


# ---------------------------------------------------------------------------
# Device-adaptive, capture-safe, default-able split-K (through the production
# wrapper flydsl_mla_reduce_v1). Unlike the opt-in plan_splitk above, the plan is
# taken from HOST-only values (final_output.size(0), num_kv_splits, num_cu) with
# no device read, so it can engage under CUDA-graph capture and is on by default.
# ---------------------------------------------------------------------------


def _build_da_single_tile(out_dtype, splits, pool=304):
    """One active tile (bs=1: final_output has a single row) with `splits`
    splits, partial pool sized to `pool` so the split count can be MUTATED up to
    `pool` across CUDA-graph replays (the device-adaptive stress)."""
    po, pl, indptr, fmap, pmap, fout, flse = build_irregular_inputs(
        [splits], _SPLITK_H, _SPLITK_DV, out_dtype, M=1, gap_stride=1,
        pool_slack=pool - splits,
    )
    pmap = torch.arange(pool, dtype=torch.int32, device=pmap.device)
    return po, pl, indptr, fmap, pmap, fout, flse


def test_da_splitk_no_engage_large_batch():
    """Default-safe: a saturated grid (base_slots >= num_cu) never engages, so
    large-batch decode keeps the unchanged single-kernel path."""
    from aiter.ops.flydsl.kernels.mla_reduce import plan_splitk_capture_safe

    engage, _, _ = plan_splitk_capture_safe(
        num_final_rows=32, H=16, max_seqlen_q=1, num_kv_splits=128, num_cu=304
    )
    assert not engage


def test_da_splitk_no_engage_low_splits():
    """Default-safe: too few splits (num_kv_splits < min) never engages, so the
    combine kernel is not paid on short-context / low-split decode."""
    from aiter.ops.flydsl.kernels.mla_reduce import plan_splitk_capture_safe

    engage, _, _ = plan_splitk_capture_safe(
        num_final_rows=1, H=16, max_seqlen_q=1, num_kv_splits=32, num_cu=304
    )
    assert not engage


@pytest.mark.slow
def test_da_splitk_wrapper_vs_hip(monkeypatch):
    """The default-able wrapper path (DA split-K on) matches HIP for b1_s128."""
    _require_cuda()
    from aiter.ops.flydsl import flydsl_mla_reduce_v1

    monkeypatch.setenv("AITER_MLA_REDUCE_DA_SPLITK", "1")
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = _build_da_single_tile(
        out_dtype, 128, pool=128
    )
    fout.zero_()
    flse.zero_()

    def run():
        flydsl_mla_reduce_v1(
            po, pl, indptr, fmap, pmap, 1, fout, flse, num_kv_splits=128
        )

    run_cudagraph_replay(run)
    ref_out, ref_lse = hip_ref_like_fout(po, pl, indptr, fmap, pmap, fout, flse)
    _assert_close(fout, flse, ref_out, ref_lse, dt)


@pytest.mark.slow
def test_da_splitk_capture_safe_varying_splits(monkeypatch):
    """THE capture-safety proof: ONE CUDA-graph capture (bs=1, grid/K/scratch
    baked from host num_kv_splits) stays correct across replays whose per-tile
    split count changes ON-DEVICE. The opt-in plan_splitk cannot do this (its
    grid/K come from a .item() read of the CSR at capture time)."""
    _require_cuda()
    from aiter.ops.flydsl import flydsl_mla_reduce_v1
    from aiter.ops.flydsl.kernels.mla_reduce import plan_splitk_capture_safe

    monkeypatch.setenv("AITER_MLA_REDUCE_DA_SPLITK", "1")
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    pool = 304
    nkv = 128
    po, pl, indptr, fmap, pmap, fout, flse = _build_da_single_tile(
        out_dtype, pool, pool=pool
    )
    engage, K, slots = plan_splitk_capture_safe(
        num_final_rows=1, H=_SPLITK_H, max_seqlen_q=1, num_kv_splits=nkv,
        num_cu=304,
    )
    assert engage and slots == _SPLITK_H, "DA split-K must engage for bs=1"

    def run():
        flydsl_mla_reduce_v1(
            po, pl, indptr, fmap, pmap, 1, fout, flse, num_kv_splits=nkv
        )

    # Warm up, then capture ONCE (grid/K/scratch fixed for this bs=1 bucket).
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.stream(side):
        run()
        side.synchronize()
        with torch.cuda.graph(graph, stream=side):
            run()
    torch.cuda.current_stream().wait_stream(side)

    # Replay the SAME graph with DIFFERENT per-tile split counts (bs fixed): the
    # device reads the mutated CSR each replay and adapts its K allocation.
    for s_k in [128, 304, 64, 200, 8, 96]:
        indptr[1] = s_k  # mutate the captured CSR in place (host->device copy)
        fout.zero_()
        flse.zero_()
        graph.replay()
        torch.cuda.synchronize()
        ref_out, ref_lse = torch_ref_gather(
            po, pl, indptr, fmap, pmap, _SPLITK_H, _SPLITK_DV, out_dtype, 1
        )
        _assert_close(fout, flse, ref_out, ref_lse, dt)


@pytest.mark.slow
@pytest.mark.parametrize("num_kv_splits", [32, 128])
def test_wrapper_host_tier_dispatch_cudagraph_replay(monkeypatch, num_kv_splits):
    """Host per-tier dispatch (AITER_MLA_REDUCE_HOST_TIER=1) stays correct under
    CUDA-graph capture/replay via the production wrapper.

    Proves alt#1: the wrapper picks the tier on the HOST from num_kv_splits (no
    device read), captures the per-tier kernel into the graph, and replays
    correctly against the HIP reference. num_kv_splits=32 -> M64 (the mid-tail
    win); num_kv_splits=128 -> M256 (over-provisioned tier still correct).
    """
    _require_cuda()
    from aiter.ops.flydsl import flydsl_mla_reduce_v1

    monkeypatch.setenv("AITER_MLA_REDUCE_HOST_TIER", "1")
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = build_serving_sparse_grid_inputs(
        *_SERVING_SHAPE, out_dtype=out_dtype
    )
    fout.zero_()
    flse.zero_()

    def run():
        flydsl_mla_reduce_v1(
            po, pl, indptr, fmap, pmap, 1, fout, flse,
            num_kv_splits=num_kv_splits,
        )

    run_cudagraph_replay(run)
    ref_out, ref_lse = hip_ref_like_fout(po, pl, indptr, fmap, pmap, fout, flse)
    _assert_close(fout, flse, ref_out, ref_lse, dt)


@pytest.mark.slow
def test_wrapper_host_tier_dispatch_heterogeneous_upper_bound(monkeypatch):
    """Critical safety case: heterogeneous per-tile splits [8, 304] with the
    host baking the num_kv_splits=304 (MLDS) tier. The MLDS body must correctly
    reduce the 8-split tile too (select_tier(upper_bound) caps LSE width only;
    each tile still reduces its actual n_splits)."""
    _require_cuda()
    from aiter.ops.flydsl import flydsl_mla_reduce_v1

    monkeypatch.setenv("AITER_MLA_REDUCE_HOST_TIER", "1")
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    H, Dv = _HIP_SHAPE
    po, pl, indptr, fmap, pmap, fout, flse = build_irregular_inputs(
        [8, 304], H, Dv, out_dtype, M=1, gap_stride=1
    )
    fout.zero_()
    flse.zero_()

    def run():
        flydsl_mla_reduce_v1(
            po, pl, indptr, fmap, pmap, 1, fout, flse, num_kv_splits=304
        )

    run_cudagraph_replay(run)
    ref_out, ref_lse = hip_ref_like_fout(po, pl, indptr, fmap, pmap, fout, flse)
    _assert_close(fout, flse, ref_out, ref_lse, dt)


@pytest.mark.slow
def test_serving_stale_indptr_cudagraph_replay():
    """Cudagraph replay after batch-8→batch-1 layout (guards-ON/OFF differential)."""
    _require_cuda()
    dt = "bf16"
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse, meta = build_serving_stale_indptr_inputs(
        *_SERVING_SHAPE, out_dtype=out_dtype
    )
    ref_out, ref_lse = _masking_ref(po, pl, indptr, fmap, pmap, *_SERVING_SHAPE, out_dtype, meta)
    on_out = fout.clone()
    on_lse = flse.clone()
    run_on = make_runner(
        po, pl, indptr, pmap, fmap, on_out, on_lse, *_SERVING_SHAPE, dt, True,
        disable_guards=False,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    on_out.zero_()
    on_lse.zero_()
    on_out[meta["logical_final_rows"]:].fill_(meta["fout_slack_seed"])
    run_cudagraph_replay(run_on)
    off_out = fout.clone()
    off_lse = flse.clone()
    run_off = make_runner(
        po, pl, indptr, pmap, fmap, off_out, off_lse, *_SERVING_SHAPE, dt, True,
        disable_guards=True,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    off_out.zero_()
    off_lse.zero_()
    off_out[meta["logical_final_rows"]:].fill_(meta["fout_slack_seed"])
    run_cudagraph_replay(run_off)
    _assert_close(
        on_out[: meta["logical_final_rows"]],
        on_lse[: meta["logical_final_rows"]],
        ref_out[: meta["logical_final_rows"]],
        ref_lse[: meta["logical_final_rows"]],
        dt,
    )
    atol = _out_atol(dt)
    q_row = meta["gather_q_row"]
    gather_err = (off_out[q_row].float() - ref_out[q_row].float()).abs().max().item()
    assert gather_err > 5 * atol
    sq = meta["store_slack_q"]
    off_slack_err = (off_out[sq:].float() - meta["fout_slack_seed"]).abs().max().item()
    assert off_slack_err > 5 * atol


def test_serving_gather_guard_differential():
    """Mapped-slack gather: guards-OFF miscompares, guards-ON matches masking ref."""
    _require_cuda()
    dt = "bf16"
    H, Dv = _SERVING_SHAPE
    po, pl, indptr, fmap, pmap, fout, flse, meta = build_serving_mapped_slack_inputs(
        H, Dv, _out_dtype(dt)
    )
    _assert_gather_differential(po, pl, indptr, fmap, pmap, fout, flse, H, Dv, dt, meta)


def test_serving_store_guard_differential():
    """Mapped-slack store: guards-OFF writes slack, guards-ON preserves seed."""
    _require_cuda()
    dt = "bf16"
    H, Dv = _SERVING_SHAPE
    po, pl, indptr, fmap, pmap, fout, flse, meta = build_serving_mapped_slack_inputs(
        H, Dv, _out_dtype(dt)
    )
    _assert_store_differential(po, pl, indptr, fmap, pmap, fout, flse, H, Dv, dt, meta)


def test_serving_gather_guard_differential_cudagraph_replay():
    _require_cuda()
    dt = "bf16"
    H, Dv = _SERVING_SHAPE
    po, pl, indptr, fmap, pmap, fout, flse, meta = build_serving_mapped_slack_inputs(
        H, Dv, _out_dtype(dt)
    )
    ref_out, ref_lse = _masking_ref(po, pl, indptr, fmap, pmap, H, Dv, _out_dtype(dt), meta)
    on_out = fout.clone()
    on_lse = flse.clone()
    run_on = make_runner(
        po, pl, indptr, pmap, fmap, on_out, on_lse, H, Dv, dt, True,
        disable_guards=False,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    on_out.zero_()
    on_lse.zero_()
    on_out[meta["store_slack_q"]:].fill_(meta["fout_slack_seed"])
    run_cudagraph_replay(run_on)
    off_out = fout.clone()
    off_lse = flse.clone()
    run_off = make_runner(
        po, pl, indptr, pmap, fmap, off_out, off_lse, H, Dv, dt, True,
        disable_guards=True,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    off_out.zero_()
    off_lse.zero_()
    off_out[meta["store_slack_q"]:].fill_(meta["fout_slack_seed"])
    run_cudagraph_replay(run_off)
    _assert_close(
        on_out[: meta["logical_final_rows"]],
        on_lse[: meta["logical_final_rows"]],
        ref_out[: meta["logical_final_rows"]],
        ref_lse[: meta["logical_final_rows"]],
        dt,
    )
    atol = _out_atol(dt)
    q_row = meta["gather_q_row"]
    gather_err = (off_out[q_row].float() - ref_out[q_row].float()).abs().max().item()
    assert gather_err > 5 * atol


def test_serving_store_guard_differential_cudagraph_replay():
    _require_cuda()
    dt = "bf16"
    H, Dv = _SERVING_SHAPE
    po, pl, indptr, fmap, pmap, fout, flse, meta = build_serving_mapped_slack_inputs(
        H, Dv, _out_dtype(dt)
    )
    ref_out, ref_lse = _masking_ref(po, pl, indptr, fmap, pmap, H, Dv, _out_dtype(dt), meta)
    on_out = fout.clone()
    on_lse = flse.clone()
    run_on = make_runner(
        po, pl, indptr, pmap, fmap, on_out, on_lse, H, Dv, dt, True,
        disable_guards=False,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    on_out.zero_()
    on_lse.zero_()
    on_out[meta["store_slack_q"]:].fill_(meta["fout_slack_seed"])
    run_cudagraph_replay(run_on)
    off_out = fout.clone()
    run_off = make_runner(
        po, pl, indptr, pmap, fmap, off_out, flse.clone(), H, Dv, dt, True,
        disable_guards=True,
        num_partial_rows=meta["logical_partial_rows"],
        num_final_rows=meta["logical_final_rows"],
    )
    off_out.zero_()
    off_out[meta["store_slack_q"]:].fill_(meta["fout_slack_seed"])
    run_cudagraph_replay(run_off)
    _assert_close(
        on_out[: meta["logical_final_rows"]],
        on_lse[: meta["logical_final_rows"]],
        ref_out[: meta["logical_final_rows"]],
        ref_lse[: meta["logical_final_rows"]],
        dt,
    )
    atol = _out_atol(dt)
    sq = meta["store_slack_q"]
    off_slack_err = (off_out[sq:].float() - meta["fout_slack_seed"]).abs().max().item()
    assert off_slack_err > 5 * atol


def test_serving_true_oob_no_fault():
    """Genuine OOB indices: guards-ON only (guards-OFF would abort the process)."""
    _require_cuda()
    dt = "bf16"
    H, Dv = _SERVING_SHAPE
    out_dtype = _out_dtype(dt)
    po, pl, indptr, fmap, pmap, fout, flse = build_serving_true_oob_inputs(
        H, Dv, out_dtype
    )
    fout.zero_()
    flse.zero_()
    run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, dt, True)
    run()
    torch.cuda.synchronize()
    ref_out, ref_lse = torch_ref_gather(
        po, pl, indptr, fmap, pmap, H, Dv, out_dtype,
        num_partial_rows=po.size(0),
        num_final_rows=fout.size(0),
    )
    _assert_close(
        fout, flse, ref_out[: fout.size(0)], ref_lse[: fout.size(0)], dt
    )


@pytest.mark.parametrize("num_tiles", _DEGEN_TILES, ids=[f"tiles{t}" for t in _DEGEN_TILES])
def test_flydsl_mla_reduce_degenerate_empty_tile(num_tiles):
    """Empty-tile guard: all-empty (n_splits=0) metadata never stores through the
    garbage q-ranges, leaving the output untouched."""
    _require_cuda()
    H, Dv = _GLM_SHAPE
    out_dtype = _out_dtype("bf16")
    po, pl, indptr, fmap, pmap, fout, flse = build_degenerate_inputs(
        num_tiles, H, Dv, out_dtype
    )
    fout.fill_(12345.0)
    flse.fill_(12345.0)
    expected_out = fout.clone()
    expected_lse = flse.clone()
    run = make_runner(po, pl, indptr, pmap, fmap, fout, flse, H, Dv, "bf16", True)
    run()
    torch.cuda.synchronize()
    assert torch.equal(fout, expected_out)
    assert torch.equal(flse, expected_lse)
