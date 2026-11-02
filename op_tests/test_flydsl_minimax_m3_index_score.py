# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Oracle and shape matrix for the MiniMax-M3 decode index-score kernel.

The oracle is deliberately written straight from the contract rather than
derived from any kernel source -- a reference that shares the implementation's
assumptions cannot catch the implementation's bugs. That is now the only
judge here: the Triton decode scorer this file once cross-checked against has
been deleted from ATOM, the FlyDSL kernel having replaced it outright, so
every candidate below is a FlyDSL config and the oracle is what they answer to.

Run:
    python op_tests/test_flydsl_minimax_m3_index_score.py
"""

import argparse
import itertools
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels import minimax_m3_index_score as kernel
from aiter.ops.flydsl.kernels.minimax_m3_index_score_prefill import (
    alloc_score as prefill_alloc_score,
)
from aiter.ops.flydsl.kernels.minimax_m3_index_score_prefill import (
    geometry as prefill_geometry,
)
from aiter.ops.flydsl.kernels.minimax_m3_index_score_prefill import (
    score_prefill_flydsl,
)
from aiter.ops.msa_block_select import pa_sparse_block_score_prefill
from aiter.test_common import benchmark, checkAllclose, run_perftest

LOG2E = 1.4426950409
P = 128  # Public operator contract, independent of the kernel implementation.
D = 128
# Score tensor tolerance, matching test_m3_indexer_context_parallel.py:80.
ATOL = RTOL = 3e-5

# Sentinel for "the kernel is contractually allowed to leave this untouched"
# NaN marks undefined reference slots; writes there are deliberately ignored.
UNWRITTEN = float("nan")
# A value no score can take, so "did the kernel write here" is unambiguous.
SENTINEL = -12345.678

DECODE_SUPPORTED_GFX = ["gfx950"]

# One mark per test rather than an autouse fixture consulting a list of test
# objects: that list had to be kept in step by hand and had already missed
# several CUDA-allocating tests. A decorator cannot be forgotten the same way.
requires_gfx950 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_gfx() not in DECODE_SUPPORTED_GFX,
    reason="GPU execution requires gfx950",
)


# ---------------------------------------------------------------------------
# Oracle -- transcribed from the contract, rule by rule.
# ---------------------------------------------------------------------------
def score_oracle(
    idx_q, cache, block_table, seq_lens, S, H, sm_scale, max_block, world=1, rank=0
):
    """Reference block scores, computed in fp32 on whatever device holds inputs.

    Returns [H, B*S, max_block] with NaN wherever the contract says the kernel
    need not write (blk >= ceil(seq_len/128)).
    """
    B = seq_lens.shape[0]
    dev = idx_q.device
    out = torch.full((H, B * S, max_block), UNWRITTEN, dtype=torch.float32, device=dev)
    scale = sm_scale * LOG2E

    # Column n = tok*H + head, so a reshape(S, H) recovers the two axes.
    tok_of_col = torch.arange(S, device=dev).repeat_interleave(H)
    pos_in_page = torch.arange(P, device=dev)

    for b in range(B):
        L = int(seq_lens[b])
        nblk = (L + P - 1) // P
        # Rule 1: accumulate in fp32. Rule 2: K is lifted to Q's dtype,
        # not the other way round -- do the cast before going to fp32 so the
        # fp8 -> bf16 rounding is reproduced, not skipped.
        q = idx_q[b * S : (b + 1) * S].reshape(S * H, D)
        # Rule 2: every column carries its own causal cutoff.
        cuts = L - S + tok_of_col + 1  # [S*H]

        for p in range(rank, nblk, world):
            page = int(block_table[b, p])
            k = cache[page].to(idx_q.dtype).float()  # [P, D]
            z = (k @ q.float().T) * scale  # [P, S*H]
            mask = (p * P + pos_in_page)[:, None] >= cuts[None, :]
            z = z.masked_fill(mask, float("-inf"))
            # Rule 3: a fully masked page still gets written (as -inf).
            out[:, b * S : (b + 1) * S, p // world] = z.amax(0).reshape(S, H).T

    return out


from aiter.ops.flydsl.kernels.minimax_m3_index_score import (
    geometry,
    score_flydsl,
)

# One (S, H, batch, max_block, cp_world, cp_rank) per geometry `geometry` can
# reach, found by enumerating its output over the whole envelope. Driving the
# arms through the real selector rather than a knob means the suite covers
# exactly what production compiles -- and if the selector starts choosing a
# geometry nothing covered, `test_geometry_points_are_exhaustive` fails rather
# than the arm silently going untested.
GEOM_POINTS = [
    (4, 4, 128, 1, 1, 0),  # tw1 ppw1 nt0
    (4, 4, 1, 1, 1, 0),  # tw1 ppw1 nt0 spread
    (16, 4, 1, 1, 1, 0),  # tw1 ppw1 nt2
    (4, 4, 1, 1, 4, 2),  # tw1 ppw1 nt2, CP
    (1, 1, 256, 8, 1, 0),  # tw1 ppw2 nt0
    (16, 4, 256, 32, 1, 0),  # tw1 ppw2 nt2
    (1, 1, 256, 32, 4, 2),  # tw1 ppw2 nt2, CP
    (8, 4, 16, 256, 1, 0),  # tw1 ppw3 nt0
    (1, 1, 256, 16, 1, 0),  # tw1 ppw4 nt0
    (16, 4, 256, 64, 1, 0),  # tw1 ppw4 nt2
    (1, 1, 256, 64, 4, 2),  # tw1 ppw4 nt2, CP
    (1, 1, 256, 2, 1, 0),  # tw2 ppw1 nt2
    (1, 1, 256, 2, 4, 2),  # tw2 ppw1 nt2, CP
    (1, 1, 1, 1, 1, 0),  # tw4 ppw1 nt2
    (1, 1, 1, 1, 4, 2),  # tw4 ppw1 nt2, CP
]


def geom_of(point, device=None):
    S, H, batch, max_block, world, rank = point
    return geometry(batch, max_block, S, H, world, rank, device)


def geom_id(point):
    S, H, b, mb, w, r = point
    tag = f"S{S}H{H}_b{b}_mb{mb}"
    return tag + (f"_cp{w}r{r}" if w > 1 else "")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def make_case(B, S, H, lens, seed=0, q_zero=False, q_slice=False):
    """Build one test case. `lens` is a per-request python list."""
    torch.manual_seed(seed)
    dev = "cuda"
    seq_lens = torch.tensor(lens, dtype=torch.int32, device=dev)
    max_block = max(1, max((L + P - 1) // P for L in lens))

    if q_slice:
        # A non-contiguous view whose head stride is still
        # that of the full tensor. Any kernel that recomputes strides as H*128
        # instead of reading them from the host breaks exactly here.
        full = torch.randn(B * S, 4, D, dtype=torch.bfloat16, device=dev)
        idx_q = full[:, 1 : 1 + H]
        assert idx_q.stride(0) == 4 * D
    elif q_zero:
        # Forced ties: every score identical, exercising top-k order stability.
        idx_q = torch.zeros(B * S, H, D, dtype=torch.bfloat16, device=dev)
    else:
        idx_q = torch.randn(B * S, H, D, dtype=torch.bfloat16, device=dev)

    # One distinct physical page per (request, logical block), shuffled, so a
    # kernel that confuses logical blk with physical page cannot pass.
    num_pages = B * max_block
    cache = torch.randn(num_pages, P, D, dtype=torch.bfloat16, device=dev).to(
        dtypes.fp8
    )
    block_table = (
        torch.randperm(num_pages, device=dev, dtype=torch.int32)
        .view(B, max_block)
        .contiguous()
    )
    return idx_q, cache, block_table, seq_lens, max_block


def build_matrix():
    """The shape matrix. (name, B, S, H, lens, kwargs)."""
    cases = []
    if True:
        tag = "fp8"
        # F = S*H sweep: below one tile, exactly one tile, multiple tiles.
        for S, H in ((1, 1), (4, 1), (4, 4), (8, 4), (16, 4)):
            cases.append((f"F{S*H}_s4096_{tag}", 2, S, H, [4096, 4096], {}))
        # Page boundaries and partial tails.
        for L in (4, 127, 128, 129, 130, 513, 4096, 8192):
            cases.append((f"L{L}_{tag}", 2, 4, 4, [L, L], {}))
        # The case the contract calls out by name: four queries whose causal
        # cutoffs straddle a page boundary (127/128/129/130).
        cases.append((f"causal_straddle_{tag}", 1, 4, 4, [130], {}))
        # Ragged batch: per-request loop bounds must be independent.
        cases.append((f"ragged_{tag}", 4, 4, 4, [130, 4096, 127, 513], {}))
        # The block -> (request, chunk) map is a prefix sum over per-request
        # chunk counts, so these two are its edge cases: a request that
        # contributes zero chunks (its cum entry repeats, and the map must skip
        # it rather than hand it work), and a skew wide enough that the holes
        # outnumber the work by 7:1.
        cases.append((f"ragged_zero_{tag}", 4, 4, 4, [0, 4096, 1, 513], {}))
        cases.append((f"ragged_skew_{tag}", 8, 4, 4, [8192] + [128] * 7, {}))
        cases.append((f"qzero_tie_{tag}", 2, 4, 4, [513, 513], {"q_zero": True}))
        cases.append((f"qslice_{tag}", 2, 4, 2, [513, 513], {"q_slice": True}))
    return cases


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare(got, ref):
    """Compare honouring the three-way contract: finite / -inf / unwritten."""
    ref_unwritten = torch.isnan(ref)
    ref_ninf = torch.isneginf(ref)
    ref_finite = ~ref_unwritten & ~ref_ninf

    # Rule 3: -inf must be reproduced as -inf, not as a large negative float.
    if not torch.equal(torch.isneginf(got) & ~ref_unwritten, ref_ninf):
        n = int((torch.isneginf(got) & ~ref_unwritten).ne(ref_ninf).sum())
        return False, f"-inf mismatch at {n} slots"

    g, r = got[ref_finite], ref[ref_finite]
    if g.numel() == 0:
        return True, "no finite slots"
    if not torch.isfinite(g).all():
        return False, f"{int((~torch.isfinite(g)).sum())} non-finite in finite region"
    err = (g - r).abs()
    tol = ATOL + RTOL * r.abs()
    if (err > tol).any():
        i = int((err - tol).argmax())
        return False, f"max_err={err.max():.3e} tol={tol.flatten()[i]:.3e}"
    return True, f"max_err={err.max():.3e}"


def test_metadata_rejects_cpu():
    q = torch.empty(1, 1, D, dtype=torch.bfloat16)
    k = torch.empty(1, P, D, dtype=torch.bfloat16)
    bt = torch.zeros(1, 1, dtype=torch.int32)
    lens = torch.zeros(1, dtype=torch.int32)
    # Supplied so the refusal is about the device, not about the two tensors
    # the predicate now requires in its own right.
    assert not kernel.index_score_supported(q, k, 1, 1, 1, bt, seq_lens=lens)
    # Prefill shares the gate, so it owes the same answer: it used to reach the
    # driver query first and raise `RuntimeError: No HIP GPUs are available` on
    # a runner with no card.
    cu = torch.tensor([0, 1], dtype=torch.int32)
    with pytest.raises(ValueError, match="requires CUDA tensors"):
        score_prefill_flydsl(q, k.to(dtypes.fp8), bt, cu, lens, lens, 1, P, 1.0)


def test_capacity_threshold(monkeypatch):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 16)
    # S16 H4 has no decode depth table, so this is the capacity estimate.
    exact = kernel.work_map_size(64, 16, 16, 4)
    smaller = kernel.work_map_size(63, 16, 16, 4)
    assert smaller > exact  # reproduces the non-monotonic sizing cliff
    assert kernel.work_map_capacity(64, 16, 16, 4) >= smaller
    # The depth ladder is non-monotonic too: past max_block 256 batch 17 takes
    # the deepest rung where batch 16 is still on the tabled one, so the larger
    # batch asks for fewer rows.
    assert kernel.work_map_size(16, 512, 4, 1) > kernel.work_map_size(17, 512, 4, 1)
    assert kernel.work_map_capacity(17, 512, 4, 1) >= kernel.work_map_size(
        16, 512, 4, 1
    )


@requires_gfx950
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.uint8])
def test_reject_cache_dtype(dtype):
    q, k, bt, lens, mb = make_case(1, 1, 1, [128])
    with pytest.raises(ValueError, match="cache"):
        score_flydsl(q, k.to(dtype), bt, lens, 1, 1, D**-0.5, mb)


@requires_gfx950
def test_reject_misaligned_cache():
    q, _, bt, _lens, mb = make_case(1, 1, 1, [128])
    k = torch.empty_strided(
        (1, P, D), (16512, 129, 1), dtype=torch.bfloat16, device="cuda"
    )
    assert not kernel.index_score_supported(q, k, 1, 1, mb, bt, seq_lens=_lens)


@requires_gfx950
def test_narrow_block_table_rejected():
    """Fewer columns than max_block is an out-of-bounds table load.

    Only the row count and the strides used to be checked, so a table narrower
    than the capacity it is paired with was accepted and read past its end.
    """
    q, k, bt, lens, mb = make_case(2, 4, 4, [4096, 4096])
    assert bt.shape[1] == mb > 1
    narrow = bt[:, : mb - 1].contiguous()
    assert not kernel.index_score_supported(q, k, 4, 4, mb, narrow, seq_lens=lens)
    with pytest.raises(ValueError, match="block_table"):
        score_flydsl(q, k, narrow, lens, 4, 4, D**-0.5, mb)


@requires_gfx950
def test_large_cache():
    dtype = dtypes.fp8
    elem = torch.empty((), dtype=dtype).element_size()
    boundary = (1 << 32) // (P * D * elem)
    required = (boundary + 2) * P * D * elem
    if torch.cuda.mem_get_info()[0] < required + (1 << 30):
        pytest.skip("need >5 GiB free for real >4 GiB page-address regression")
    q = torch.ones((1, 1, D), dtype=torch.bfloat16, device="cuda")
    cache = torch.empty((boundary + 2, P, D), dtype=dtype, device="cuda")
    ids = [boundary - 1, boundary, boundary + 1]
    for page, value in zip(ids, [1, 2, 3]):
        cache[page].fill_(value)
    bt = torch.tensor([ids], dtype=torch.int32, device="cuda")
    lens = torch.tensor([3 * P], dtype=torch.int32, device="cuda")
    got = score_flydsl(q, cache, bt, lens, 1, 1, D**-0.5, 3)
    ref = torch.tensor([1, 2, 3], device="cuda") * D**0.5 * LOG2E
    torch.testing.assert_close(got[0, 0], ref, atol=ATOL, rtol=RTOL)


@requires_gfx950
@pytest.mark.parametrize("request_id", [32767, 32768, 65534])
def test_packed_request(request_id):
    batch = request_id + 1
    q = torch.ones((batch, 1, D), dtype=torch.bfloat16, device="cuda")
    q[request_id].fill_(2)
    k = torch.ones((1, P, D), dtype=torch.bfloat16, device="cuda").to(dtypes.fp8)
    bt = torch.zeros((batch, 1), dtype=torch.int32, device="cuda")
    lens = torch.zeros(batch, dtype=torch.int32, device="cuda")
    lens[request_id] = P
    got = score_flydsl(q, k, bt, lens, 1, 1, D**-0.5, 1)
    torch.testing.assert_close(
        got[0, request_id, 0],
        torch.tensor(2 * D**0.5 * LOG2E, device="cuda"),
        atol=ATOL,
        rtol=RTOL,
    )


@requires_gfx950
@pytest.mark.parametrize("world,rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
@pytest.mark.parametrize("S,H", [(1, 1), (1, 4), (4, 2), (8, 4)])
@pytest.mark.parametrize("layout", ["contiguous", "feature"])
@pytest.mark.parametrize("map_in_graph", [False, True])
def test_graph_buffers(world, rank, S, H, layout, map_in_graph):
    B, mb = 3, 9  # Stable envelope larger than actual lengths, including CP.
    if (S, H) == (1, 4):
        cu = torch.cuda.get_device_properties("cuda").multi_processor_count
        mb = 2 * (cu // B)  # Exercise the two-token-wave automatic branch.
        assert geometry(B, mb, S, H)["waves_per_tok"] == 2
    q, cache, _, seq, _ = make_case(B, S, H, [0, 513, 1])
    bt = torch.zeros((B, mb * world), dtype=torch.int32, device="cuda")
    bt[:, :5] = torch.arange(5, device="cuda", dtype=torch.int32)
    cp = {"cp_world": world, "cp_rank": rank}
    buf = torch.empty(
        (kernel.work_map_capacity(B, mb, S, H, **cp), 2),
        device="cuda",
        dtype=torch.int32,
    )
    wm = kernel.build_work_map(seq, mb, S, H, out=buf, **cp)
    out = (
        torch.empty((H, B * S, mb), device="cuda", dtype=torch.float32)
        if layout == "contiguous"
        else torch.empty((mb, B * S, H), device="cuda", dtype=torch.float32).permute(
            2, 1, 0
        )
    )
    ptrs = out.data_ptr(), buf.data_ptr(), wm.data_ptr()

    def run():
        current = (
            kernel.build_work_map(seq, mb, S, H, out=buf, **cp) if map_in_graph else wm
        )
        return score_flydsl(
            q, cache, bt, seq, S, H, D**-0.5, mb, out=out, work_map=current, **cp
        )

    run()  # Finish JIT before capture.
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for lengths, shift in [([0, 513, 1], 0), ([130, 0, 257], 3), ([1, 129, 0], 1)]:
        seq.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        bt.copy_(
            (
                torch.arange(mb * world, device="cuda", dtype=torch.int32)[None, :]
                + shift
            )
            .remainder(cache.shape[0])
            .expand(B, -1)
        )
        if not map_in_graph:
            kernel.build_work_map(seq, mb, S, H, out=buf, **cp)
        graph.replay()
        ref = score_oracle(q, cache, bt, seq, S, H, D**-0.5, mb, world, rank)
        check_scores(out, ref)
        assert ptrs == (out.data_ptr(), buf.data_ptr(), wm.data_ptr())
        assert wm.shape[0] == kernel.work_map_size(B, mb, S, H, **cp)


@requires_gfx950
@pytest.mark.parametrize(
    "field",
    [
        "q_dtype",
        "q_rank",
        "q_stride",
        "q_base",
        "k_rank",
        "bt_dtype",
        "bt_stride",
        "lens_dtype",
        "lens_stride",
        "out_dtype",
        "out_shape",
        "out_stride",
        "map_dtype",
        "map_stride",
        "map_capacity",
        "map_exact",
        "device",
    ],
)
def test_invalid_metadata(field):
    q, k, bt, lens, mb = make_case(2, 4, 2, [130, 129])
    out = torch.empty((2, 8, mb), device="cuda", dtype=torch.float32)
    wm = kernel.build_work_map(lens, mb, 4, 2)
    if field == "q_dtype":
        q = q.float()
    elif field == "q_rank":
        q = q.flatten(0, 1)
    elif field == "q_stride":
        q = torch.empty_strided(q.shape, (258, 129, 1), device="cuda", dtype=q.dtype)
    elif field == "q_base":
        q = torch.empty(q.numel() + 1, device="cuda", dtype=q.dtype)[1:].view(q.shape)
    elif field == "k_rank":
        k = k.flatten(0, 1)
    elif field == "bt_dtype":
        bt = bt.long()
    elif field == "bt_stride":
        bt = torch.empty((2, mb * 2), device="cuda", dtype=torch.int32)[:, ::2]
    elif field == "lens_dtype":
        lens = lens.long()
    elif field == "lens_stride":
        lens = torch.zeros(4, device="cuda", dtype=torch.int32)[::2]
    elif field == "out_dtype":
        out = out.to(torch.bfloat16)
    elif field == "out_shape":
        out = out[:1]
    elif field == "out_stride":
        out = torch.empty(1, device="cuda").expand(out.shape)
    elif field == "map_dtype":
        wm = wm.long()
    elif field == "map_stride":
        wm = torch.empty((wm.shape[0], 4), device="cuda", dtype=torch.int32)[:, ::2]
    elif field == "map_capacity":
        wm = wm[:1]
    elif field == "map_exact":
        wm = torch.empty((wm.shape[0] + 1, 2), device="cuda", dtype=torch.int32)
    elif field == "device":
        k = k.cpu()
    assert not kernel.index_score_supported(
        q, k, 4, 2, mb, bt, seq_lens=lens, out=out, work_map=wm
    )
    with pytest.raises(ValueError):
        score_flydsl(q, k, bt, lens, 4, 2, D**-0.5, mb, out=out, work_map=wm)


@requires_gfx950
def test_aligned_padding():
    q, k, bt, lens, mb = make_case(2, 4, 2, [130, 513], q_slice=True)
    padded = torch.empty_strided(
        k.shape, (P * 144, 144, 1), device=k.device, dtype=k.dtype
    )
    padded.copy_(k)
    got = score_flydsl(q, padded, bt, lens, 4, 2, D**-0.5, mb)
    check_scores(got, score_oracle(q, k, bt, lens, 4, 2, D**-0.5, mb))


@pytest.mark.parametrize(
    "batch,mb", [(65536, 1), (1, 262145), (65535, 262144), (1, 1 << 24)]
)
def test_packing_bounds(batch, mb):
    with pytest.raises(ValueError):
        kernel.work_map_capacity(batch, mb, 1, 1)


@pytest.mark.parametrize("cu", [16, 80, 256, 304])
def test_capacity_envelope(monkeypatch, cu):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: cu)
    # One envelope now covers every geometry `geometry` can pick, so a
    # persistent buffer sized here stays valid whatever it resolves to.
    cap = kernel.work_map_capacity(64, 33, 8, 4)
    for batch, mb in itertools.product(range(1, 65), [1, 4, 8, 16, 17, 32, 33]):
        assert kernel.work_map_size(batch, mb, 8, 4) <= cap


# Every geometry field that changes the emitted code has to change the kernel
# name too: the name is the JIT cache key, so a collision serves the binary
# built for the other geometry, silently and with no error. Only the default
# keeps the bare name, which is what lets an untouched launch reuse a cache
# built before the field existed.
def test_kernel_name_separates_every_code_changing_field():
    base = _geom(ppw=1, nt_k=2)
    seen = {kernel.kernel_name(4, 1, base): "default"}
    probe = {
        "pages_per_wave": 2,
        "spread": 1,
        "waves_per_tok": 2,
        "nt_k": 0,
        "cp_world": 4,
    }
    assert probe.keys() <= base.keys(), "probe names a field geometry dropped"
    for field, value in probe.items():
        name = kernel.kernel_name(4, 1, {**base, field: value})
        assert name not in seen, f"{field}={value} collides with {seen[name]}"
        seen[name] = f"{field}={value}"
    # Ranks sharing a node share the JIT cache directory and compile DIFFERENT
    # kernels, since the block remap folds the rank in as a constant.
    cp = {**base, "cp_world": 4}
    r0 = kernel.kernel_name(4, 1, {**cp, "cp_rank": 0})
    r2 = kernel.kernel_name(4, 1, {**cp, "cp_rank": 2})
    assert r0 != r2 and r2 not in seen


# MiniMax-M3 decode launches, no CP, 1M-token capacity (8192 blocks):
# TP4 MTP is S=4, H=1; the other served launch is S=8, H=4.
@pytest.mark.parametrize(
    "S,H,batch,ppw",
    [
        (4, 1, 1, 1),
        (4, 1, 2, 1),
        (4, 1, 4, 1),
        (4, 1, 5, 2),
        (4, 1, 8, 2),
        (4, 1, 16, 2),
        (4, 1, 17, 4),
        (4, 1, 32, 4),
        (4, 1, 128, 4),
        (8, 4, 1, 1),
        (8, 4, 2, 1),
        (8, 4, 3, 2),
        (8, 4, 4, 2),
        (8, 4, 5, 3),
        (8, 4, 8, 3),
        (8, 4, 16, 3),
        (8, 4, 17, 4),
        (8, 4, 32, 4),
        (8, 4, 128, 4),
        # Every other one-feature-tile shape shares the S4H1 table.
        (1, 1, 1, 1),
        (1, 4, 4, 1),
        (4, 4, 8, 2),
        (8, 2, 16, 2),
        (16, 1, 17, 4),
    ],
)
@requires_gfx950
def test_served_decode_dispatch(S, H, batch, ppw):
    resolved = geometry(batch, 8192, S, H)
    assert (
        resolved["pages_per_wave"],
        resolved["nt_k"],
        resolved["waves_per_tok"],
    ) == (
        ppw,
        0,
        1,
    )


# Two things bound the depth, and both have to be in the answer: the tabled
# ladder, and the occupancy cap -- depth only pays when there are enough pages
# to fill the machine at that depth, and a full batch of *short* requests has a
# large batch and very little work.
@pytest.mark.parametrize(
    "S,H,batch,max_block,ppw,spread",
    [
        # 8K context (64 blocks): the default is 2, occupancy caps it below that.
        (4, 1, 8, 64, 1, 1),  # 512 pages: cap 1
        (4, 1, 16, 64, 1, 1),  # 1024 pages: cap 1
        (4, 1, 32, 64, 2, 0),  # 2048 pages: cap 2 == default
        (4, 1, 64, 64, 2, 0),  # 4096 pages: cap 4, default 2 binds
        (4, 1, 128, 64, 2, 0),  # 8192 pages: cap 8, default 2 binds
        (8, 4, 16, 64, 1, 1),  # table 3, cap 1
        (8, 4, 32, 64, 2, 0),  # table 4, cap 2
        (4, 1, 32, 256, 2, 0),  # still the shallow default at max_block 256
        # Past max_block 256 the ladder's deepest rung comes back, uncapped.
        (4, 1, 32, 4096, 4, 0),
        (8, 4, 32, 4096, 4, 0),
        (4, 1, 1, 4096, 1, 1),
    ],
)
@requires_gfx950
def test_served_depth_capped_by_occupancy(
    monkeypatch, S, H, batch, max_block, ppw, spread
):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 256)
    resolved = geometry(batch, max_block, S, H)
    assert (resolved["pages_per_wave"], resolved["spread"]) == (ppw, spread)
    # Whatever it picked, every CU gets a CTA unless there are fewer pages
    # than CUs to begin with.
    ctas = batch * -(-max_block // kernel.work_chunk(resolved))
    assert ctas >= 256 or batch * max_block < 256 * 4


@pytest.mark.parametrize("batch,max_block,tw", [(4, 64, 4), (4, 128, 2), (4, 129, 1)])
@requires_gfx950
def test_q1_token_split_regime_kept(monkeypatch, batch, max_block, tw):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 256)
    resolved = geometry(batch, max_block, 1, 4)
    # Small enough for the token split: it keeps its own measured regime.
    # One block past it, the S1 launch joins the one-tile depth table.
    assert resolved["waves_per_tok"] == tw
    assert resolved["spread"] == (1 if tw == 1 else 0)


# The spread map goes with one page per wave on the served path: H1Q4 up to
# batch 4, Q8H4 up to batch 2 -- and only while the row packing fits.
@pytest.mark.parametrize(
    "S,H,batch,max_block,spread",
    [
        (4, 1, 1, 8192, 1),
        (4, 1, 4, 8192, 1),
        (4, 1, 5, 8192, 0),
        (8, 4, 2, 8192, 1),
        (8, 4, 3, 8192, 0),
        (4, 1, 1, 0xFFFF, 1),
        (4, 1, 1, 0x10000, 0),
        (4, 1, 1, 1, 1),
    ],
)
@requires_gfx950
def test_spread_dispatch(S, H, batch, max_block, spread):
    resolved = geometry(batch, max_block, S, H)
    assert resolved["spread"] == spread
    rows = kernel.work_map_size(batch, max_block, S, H)
    if spread:  # one page per wave, plus the spare row per request
        assert rows == batch * ((max_block + 3) // 4 + 1)
    assert kernel.work_map_capacity(batch, max_block, S, H) >= rows


@requires_gfx950
def test_spread_declines_unpackable_bounds():
    """A spread row packs the first page in 16 bits and the page count in the
    top byte of the length word. Where that does not fit the selector takes the
    fixed-chunk map instead -- silently, because it is a scheduling choice and
    not an error -- so what has to hold is that it never picks spread there."""
    assert _geom_spread(kernel.SPREAD_MAX_BATCH, 64, 4, 1) in (0, 1)
    assert _geom_spread(kernel.SPREAD_MAX_BATCH + 1, 64, 4, 1) == 0
    assert _geom_spread(1, 0x10000, 4, 1) == 0  # first page past 16 bits
    assert _geom_spread(1, 0x8000, 4, 1, cp_world=4) == 0  # token span past 2**24
    # And the packing bound it is derived from agrees.
    assert not kernel._spread_fits(kernel.SPREAD_MAX_BATCH + 1, 64)
    assert not kernel._spread_fits(1, 0x8000, 4)
    assert kernel._spread_fits(1, 0x7FFF, 4)  # 4 * 0x7FFF * 128 < 2**24


def _geom_spread(batch, max_block, S, H, **cp):
    return geometry(batch, max_block, S, H, **cp)["spread"]


def decode_spread(wm):
    """(request, first page, pages, seq_len) per live row, holes checked."""
    wm = wm.cpu().long() & 0xFFFFFFFF
    live = wm[:, 1] != 0
    n = int(live.sum())
    assert live[:n].all() and not live[n:].any(), "holes must trail the work"
    assert not wm[n:].any(), "a hole is (0, 0)"
    w = wm[:n]
    return w[:, 0] >> 16, w[:, 0] & 0xFFFF, w[:, 1] >> 24, w[:, 1] & 0xFFFFFF


def _geom(spread=0, ppw=1, tw=1, world=1, rank=0, nt_k=0):
    """A geometry dict straight up, so a map test can pin the shape it means
    instead of hunting for bounds that make `geometry` pick it."""
    return {
        "waves_per_tok": tw,
        "pages_per_wave": ppw,
        "nt_k": nt_k,
        "spread": spread,
        "cp_world": world,
        "cp_rank": rank,
    }


def _local_pages(lens, world, rank):
    """Pages this rank owns per request: block p is global p*world + rank."""
    return [max(0, (L + P - 1) // P - rank + world - 1) // world for L in lens]


def _build_map(lens, max_block, geom, device="cuda"):
    seq = torch.tensor(lens, dtype=torch.int32, device=device)
    rows = len(lens) * kernel._grid_chunks(max_block, geom)
    return kernel._make_map(seq, max_block, geom, rows), seq


@requires_gfx950
@pytest.mark.parametrize("cu", [80, 256, 304])
def test_spread_map_covers_every_page_exactly_once(monkeypatch, cu):
    """The spread map's contract, stated as an invariant rather than as
    equality with a second implementation.

    A bit-for-bit comparison against a torch twin only says the two agree; it
    cannot survive either of them being rewritten. These four properties are
    what the kernel actually depends on, so they hold for any correct builder.

    `cu` is the row budget the plan apportions against, so it is forced rather
    than left as whatever this part reports -- otherwise every run tests one CU
    count and the clamp's behaviour on a wider or narrower part is unexamined.
    The env var alone would not do it: `get_cu_num` caches its first read, so
    the second parametrisation would silently re-run the first one's count.
    """
    import random

    from aiter.jit.utils import chip_info

    monkeypatch.setenv("CU_NUM", str(cu))
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: cu)
    gen = random.Random(cu)
    # First the case reported as dropping pages on a 304-CU part: 1028 pages
    # against a 276-row budget, which the per-row spans cover exactly.
    cases = [(1100, 1, 0, [1028 * P])]
    for _ in range(120):
        B = gen.choice([1, 2, 3, 4, 8])
        mb = gen.choice([1, 5, 64, 1100, 2049, 8192])
        world, rank = gen.choice([(1, 0), (1, 0), (4, 1), (2, 1)])
        cases.append(
            (mb, world, rank,
             [gen.choice([0, gen.randint(1, mb * world * P)]) for _ in range(B)])
        )  # fmt: skip
    for mb, world, rank, lens in cases:
        B = len(lens)
        g = _geom(spread=1, world=world, rank=rank)
        wm, seq = _build_map(lens, mb, g)
        b, first, span_, length = decode_spread(wm)
        assert ((1 <= span_) & (span_ <= 4)).all()
        assert torch.equal(length, seq.cpu().long()[b])
        nblk = _local_pages(lens, world, rank)
        seen = {}
        for r, f, s in zip(b.tolist(), first.tolist(), span_.tolist()):
            for page in range(f, f + s):
                seen[(r, page)] = seen.get((r, page), 0) + 1
        want = {(r, page) for r in range(B) for page in range(nblk[r])}
        assert set(seen) == want and set(seen.values()) <= {1}, lens


@requires_gfx950
@pytest.mark.parametrize("ppw", [1, 2, 3, 4])
def test_chunk_map_covers_every_chunk_exactly_once(ppw):
    """Same contract for the fixed-chunk map: request b's chunk c appears
    once, the holes trail, and a hole reads as (0, 0)."""
    import random

    gen = random.Random(ppw)
    for _ in range(60):
        B = gen.choice([1, 2, 4, 7, 64])
        mb = gen.choice([1, 5, 257, 1100, 8192])
        world, rank = gen.choice([(1, 0), (1, 0), (4, 3)])
        cap = mb * world * P
        lens = [min(gen.choice([0, 1, P, cap, gen.randint(1, cap)]), cap)
                for _ in range(B)]  # fmt: skip
        g = _geom(ppw=ppw, world=world, rank=rank)
        chunk = kernel.work_chunk(g)
        wm, seq = _build_map(lens, mb, g)
        w = wm.cpu().long() & 0xFFFFFFFF
        live = w[:, 1] != 0
        n = int(live.sum())
        assert live[:n].all() and not live[n:].any(), "holes must trail the work"
        assert not w[n:].any(), "a hole is (0, 0)"
        got = [(int(r) >> 16, int(r) & 0xFFFF) for r in w[:n, 0]]
        nblk = _local_pages(lens, world, rank)
        want = [(b, c) for b in range(B) for c in range(-(-nblk[b] // chunk))]
        assert sorted(got) == sorted(want), (B, mb, ppw, world, rank, lens)
        assert torch.equal(w[:n, 1], seq.cpu().long()[[b for b, _ in got]])


# Page counts either side of a CU round, where the fixed-chunk map spills.
SPREAD_CASES = [
    ("b1_spill", 4, 1, [1028 * P - 5], 1100),
    ("b2_tail", 4, 1, [1024 * P, 1], 1100),
    ("b4_even", 4, 1, [257 * P] * 4, 300),
    ("b4_ragged", 4, 1, [0, 3 * P + 1, 900 * P, 17], 1000),
    ("q8h4_b2", 8, 4, [700 * P + 3, 400 * P], 800),
]


@requires_gfx950
@pytest.mark.parametrize(
    "S,H,lens,mb", [c[1:] for c in SPREAD_CASES], ids=[c[0] for c in SPREAD_CASES]
)
def test_spread_scores(S, H, lens, mb):
    B = len(lens)
    q, cache, table, seq, _ = make_case(B, S, H, lens)
    bt = torch.zeros((B, mb), dtype=torch.int32, device="cuda")
    bt[:, : table.shape[1]] = table
    assert geometry(B, mb, S, H)["spread"] == 1
    got = score_flydsl(q, cache, bt, seq, S, H, D**-0.5, mb)
    ref = score_oracle(q, cache, bt, seq, S, H, D**-0.5, mb)
    check_scores(got, ref)
    # That the two maps describe the same pages is checked at the map level,
    # in `test_spread_map_covers_every_page_exactly_once` and its fixed-chunk
    # twin: the entry point resolves the geometry itself, so a map built for
    # the other one is rejected here rather than scored.


@requires_gfx950
@pytest.mark.parametrize("S,H,B", [(4, 1, 2), (4, 1, 4), (8, 4, 2)])
def test_spread_graph_replay(S, H, B):
    mb = 1100
    q, cache, _, seq, _ = make_case(B, S, H, [1] * B)
    cache = torch.randn(B * mb, P, D, device="cuda").to(torch.float8_e4m3fn)
    bt = torch.zeros((B, mb), dtype=torch.int32, device="cuda")
    assert geometry(B, mb, S, H)["spread"] == 1
    buf = torch.empty(
        (kernel.work_map_capacity(B, mb, S, H), 2), device="cuda", dtype=torch.int32
    )
    out = kernel.alloc_score(B, S, H, mb, q.device)

    def run():
        wm = kernel.build_work_map(seq, mb, S, H, out=buf)
        return score_flydsl(q, cache, bt, seq, S, H, D**-0.5, mb, out=out, work_map=wm)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    # Under one round, just over it, over two, a lone tail page, all empty.
    for total in ([5], [1028 * P - 7], [1024 * P, 1], [1100 * P - 1], [0]):
        lengths = (total + [0] * B)[:B]
        lengths = [min(x, mb * P) for x in lengths]
        if B > 1 and lengths[0] > P:
            lengths[1] = lengths[0] // 3  # ragged, still under capacity
        seq.copy_(torch.tensor(lengths, dtype=torch.int32, device="cuda"))
        bt.copy_(torch.randperm(B * mb, device="cuda", dtype=torch.int32).view(B, mb))
        graph.replay()
        check_scores(out, score_oracle(q, cache, bt, seq, S, H, D**-0.5, mb))


def test_cli_default_matrix():
    args = parse_args([])
    cases = select_cases(args)
    assert cases == build_matrix()
    assert len(cases) == 19
    assert args.layout == ["contiguous", "feature"]
    # One implementation per row now: the geometry is picked by `geometry`
    # from the shape, so there is nothing left to select among.
    assert len(cases) * len(args.layout) == 38


def test_cli_custom_cross_product():
    args = parse_args(["-b", "1", "3", "-s", "1,1,128", "4,2,513"])
    cases = select_cases(args)
    assert len(cases) == 4
    assert {(c[1], c[2], c[3], c[4][0]) for c in cases} == {
        (B, S, H, L)
        for B, (S, H, L) in itertools.product([1, 3], [(1, 1, 128), (4, 2, 513)])
    }
    assert all(c[4] == [c[4][0]] * c[1] and c[5] == {} for c in cases)
    assert len({c[0] for c in cases}) == 4
    assert select_cases(parse_args(["-s", "1,1,128"]))[0][1] == 2
    name = cases[0][0]
    assert select_cases(parse_args(["-b", "1", "-s", "1,1,128", "--case", name])) == [
        cases[0]
    ]


def test_cli_regression_filters():
    cases = select_cases(parse_args(["-b", "4"]))
    assert [c[0] for c in cases] == ["ragged_fp8", "ragged_zero_fp8"]
    assert select_cases(parse_args(["--case", "qslice_fp8"])) == [
        c for c in build_matrix() if c[0] == "qslice_fp8"
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["-b"],
        ["-s"],
        ["--case"],
        ["--case", "unknown"],
        ["-b", "3"],
        ["-s", "1,1,128", "-b"],
    ],
)
def test_cli_empty_cases(argv):
    assert select_cases(parse_args(argv)) == []


@pytest.mark.parametrize("axis", ["--layout"])
def test_cli_empty_axes(axis):
    args = parse_args([axis])
    assert list(itertools.product(select_cases(args), args.layout)) == []


@pytest.mark.parametrize(
    "argv",
    [
        ["-b", "0"],
        ["-b", "-1"],
        ["-b", "65536"],
        ["-s", "4"],
        ["-s", "4,2"],
        ["-s", "4,2,128,1"],
        ["-s", "0,2,128"],
        ["-s", "4,-1,128"],
        ["-s", "4,2,0"],
        ["-s", "4,2,3"],
        ["-s", "4,2,2147483647"],
        ["-s", "x,2,128"],
        ["--layout", "unknown"],
    ],
)
def test_cli_invalid_args(argv):
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


def test_k_axis_map_is_a_bijection():
    """Every head-dim position reaches exactly one (k-step, lane group, slot).

    A lane's 16 B K load covers k-elements 64i+16g..+16, a *permutation* of the
    MFMA's k axis. That is free because the dot sums over k and both operands
    go through the same map -- but only if it really is a permutation. Get it
    wrong and nothing faults, the scores are just paired from the wrong
    elements, which is why this is checked here and not left to the kernel.
    """
    lane_groups = kernel.WAVE // 16
    covered = sorted(
        (ks // kernel.K_PER_LOAD) * kernel.BLOCK_K
        + g * kernel.LANE_BLOCK
        + (ks % kernel.K_PER_LOAD) * kernel.LANE_K
        + v
        for ks in range(kernel.KSTEPS)
        for g in range(lane_groups)
        for v in range(kernel.LANE_K)
    )
    assert covered == list(range(D))
    assert kernel.KSTEPS * kernel.LANE_K * lane_groups == D
    assert kernel.LANE_BLOCK * lane_groups == kernel.BLOCK_K  # groups tile a block
    assert kernel.LANE_K * kernel.K_PER_LOAD == kernel.ACCESS_BYTES  # fp8, 1 B each


@requires_gfx950
def test_fp8_dtype_comes_from_aiter():
    """The accepted fp8 flavour is the arch's, not a literal in the kernel.

    e4m3fn and e4m3fnuz differ in exponent bias, so a kernel that names one
    and runs on the other is silently off by a factor of two. The kernel takes
    it from `aiter.dtypes`, which means the chip decides.
    """
    assert kernel.dtypes.fp8 is dtypes.fp8
    # gfx950 is the only arch this kernel accepts, and its fp8 is OCP e4m3fn.
    assert dtypes.fp8 is torch.float8_e4m3fn


@requires_gfx950
def test_empty_selection(caplog):
    caplog.handler.setLevel(0)
    aiter.logger.addHandler(caplog.handler)
    # `--ctx-len` with no values empties the prefill sweep too, so the run has
    # nothing at all to do and both halves say so.
    main(["--case", "not-a-case", "--ctx-len"])
    aiter.logger.removeHandler(caplog.handler)
    assert "no decode score tests executed" in caplog.text
    with pytest.raises(SystemExit) as exc:
        main(["-b", "0"])
    assert exc.value.code == 2


@requires_gfx950
def test_empty_cp_shard_narrow_table():
    """A rank whose shard is empty has no local page count, so it passes the
    GLOBAL one; the kernel clamps its speculative block-table load to column 0.
    The worst-case width bound `(max_block-1)*cp_world + cp_rank` rejects this
    legal call, which is why there is no such check."""
    q, k, bt, lens, mb = make_case(2, 4, 4, [0, 128])
    out = score_flydsl(q, k, bt, lens, 4, 4, D**-0.5, mb, cp_world=4, cp_rank=3)
    check_scores(out, score_oracle(q, k, bt, lens, 4, 4, D**-0.5, mb, 4, 3))


@requires_gfx950
@pytest.mark.parametrize("point", GEOM_POINTS, ids=geom_id)
def test_every_geometry_is_the_same_arithmetic(point):
    """One geometry per arm, chosen by the real selector, all against the
    oracle. These are the same sum reassociated over different splits of the
    page, wave and token axes, so a disagreement is a bug in the split and not
    a precision difference.

    Negative scores expose accidental zero-initialized tok_tile maxima; the
    zero-length and short requests cover empty and fully masked pages, and the
    longest one carries across pages into a partial final tok_tile.
    """
    S, H, batch, max_block, world, rank = point
    # `max_block` is this rank's LOCAL width, so the longest request must span
    # that many of its blocks: global block p*world + rank.
    glob = max_block * world
    lens = ([0, 1, 130] + [P] * batch)[: batch - 1] + [glob * P]
    lens = [min(x, glob * P) for x in lens]  # the longest request sets max_block
    q, cache, table, seq, mb = make_case(batch, S, H, lens)
    assert mb == glob, (mb, glob)
    local = -(-(mb - rank) // world)
    assert local == max_block, (local, max_block)
    # The table is indexed by the GLOBAL block even though the score tensor is
    # the local shard, so it keeps the global width.
    bt = table
    q.copy_(q.abs() + 0.25)
    # In page-sized chunks: the widest arms hold 64K pages, and a whole-tensor
    # fp32 round trip for the sign flip is 4 GiB of temporary.
    for i in range(0, cache.shape[0], 2048):
        blk = cache[i : i + 2048]
        blk.copy_(-(blk.float().abs() + 0.25))
    got_geom = geometry(batch, local, S, H, world, rank, q.device)
    assert tuple(sorted(got_geom.items())) == tuple(
        sorted(geom_of(point, q.device).items())
    )
    ref = score_oracle(q, cache, bt, seq, S, H, D**-0.5, local, world, rank)
    out = kernel.alloc_score(batch, S, H, local, q.device)
    out.fill_(float("nan"))
    got = score_flydsl(
        q, cache, bt, seq, S, H, D**-0.5, local, out=out, cp_world=world, cp_rank=rank
    )
    assert got.data_ptr() == out.data_ptr()
    ok, reason = compare(got, ref)
    assert ok, reason
    assert bool((got[torch.isfinite(ref)] < 0).all())


@requires_gfx950
def test_arch_rejected(monkeypatch):
    """An arch with neither MFMA generation is refused before anything else."""

    q, k, bt, lens, mb = make_case(1, 1, 1, [128])
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx90a"),
    )
    assert not kernel.index_score_supported(q, k, 1, 1, mb, bt, seq_lens=lens)
    with pytest.raises(ValueError, match="requires gfx950"):
        score_flydsl(q, k, bt, lens, 1, 1, D**-0.5, mb)


@pytest.mark.parametrize("operand", ["q", "out", "bt", "map"])
@requires_gfx950
def test_address_span_rejected_without_allocation(monkeypatch, operand):

    class MetadataTensor(torch.Tensor):
        @property
        def device(self):
            return torch.device("cuda", 0)

        def data_ptr(self):
            return 16

    def meta(shape, stride, dtype):
        return torch.Tensor._make_subclass(
            MetadataTensor,
            torch.empty_strided(shape, stride, dtype=dtype, device="meta"),
        )

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx950", multi_processor_count=16),
    )
    q = meta((2, 1, D), (D, D, 1), torch.bfloat16)
    k = meta((1, P, D), (P * D, D, 1), dtypes.fp8)
    bt = meta((2, 1), (1, 1), torch.int32)
    out = meta((1, 2, 1), (2, 1, 1), torch.float32)
    wm = meta((2, 2), (2, 1), torch.int32)
    if operand == "q":
        q = meta((2, 1, D), (0x7FFFFF80, D, 1), torch.bfloat16)
    if operand == "out":
        out = meta((1, 2, 1), (1, 1 << 30, 1), torch.float32)
    if operand == "bt":
        bt = meta((2, 1), (1 << 30, 1), torch.int32)
    if operand == "map":
        wm = meta((2, 2), (1 << 30, 1), torch.int32)
    lens = meta((1,), (1,), torch.int32)
    assert not kernel.index_score_supported(
        q, k, 1, 1, 1, bt, seq_lens=lens, out=out, work_map=wm
    )
    with pytest.raises(ValueError, match="span"):
        kernel._validate_metadata(q, k, 1, 1, 1, bt, out=out, work_map=wm)


def test_strided_span_metadata():
    # No GPU storage needed: count true strided addresses, not logical numel.
    x = torch.empty_strided((2, 128), (1 << 30, 1), device="meta")
    assert x.numel() == 256
    assert kernel._span(x) == (1 << 30) + 128


@requires_gfx950
def test_noncurrent_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible gfx950 GPUs")
    with torch.cuda.device(1):
        q, k, bt, lens, mb = make_case(1, 1, 1, [128])
        # Device 1 must not use device 0's compiled module handle.
    with torch.cuda.device(0):
        q0, k0, bt0, lens0, mb0 = make_case(1, 1, 1, [128])
        score_flydsl(q0, k0, bt0, lens0, 1, 1, D**-0.5, mb0)
        out = score_flydsl(q, k, bt, lens, 1, 1, D**-0.5, mb)
        assert torch.cuda.current_device() == 0
    check_scores(out, score_oracle(q, k, bt, lens, 1, 1, D**-0.5, mb))


@requires_gfx950
@pytest.mark.parametrize(
    "kw",
    [
        {"max_block": 0},
        {"max_block": -1},
        {"cp": {"cp_world": 0}},
        {"cp": {"cp_world": 2, "cp_rank": 2}},
        {"cp": {"cp_world": 1, "cp_rank": -1}},
    ],
)
def test_build_map_invalid_bounds(kw):
    """`geometry` is the only thing that picks a shape now, so the bounds it
    reads are the only thing left to reject."""
    lens = torch.ones(1, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError):
        kernel.build_work_map(lens, kw.get("max_block", 1), 1, 1, **kw.get("cp", {}))


@requires_gfx950
@pytest.mark.parametrize("scale", [1e-50, 1e39])
def test_scale_fp32_range(scale):
    q, k, bt, lens, mb = make_case(1, 2, 1, [129])
    with pytest.raises(ValueError, match="sm_scale"):
        score_flydsl(q, k, bt, lens, 2, 1, scale, mb)


def test_exact_size_uses_explicit_device(monkeypatch):

    from aiter.jit.utils import chip_info

    # An explicit device wins over the probed count -- but `CU_NUM` wins over
    # both, so drop it or this asserts whatever the shell happened to export.
    monkeypatch.delenv("CU_NUM", raising=False)
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 304)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=256),
    )
    # S16 H4 takes the capacity estimate, which is what reads the CU count.
    assert kernel.work_map_size(16, 1024, 16, 4, device=torch.device("cuda", 0)) == 1024


@requires_gfx950
def test_cu_override_reaches_both_sizing_paths(monkeypatch):
    """`work_map_size` has no tensor and resolves with `device=None`, while
    `build_work_map` resolves from `seq_lens.device`. Read different counts and
    the geometry differs: with `CU_NUM=64` on a 256-CU part one said 64 rows
    and the other 128, so the buffer the first sized was refused by the second.
    """
    monkeypatch.setenv("CU_NUM", "64")
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 64)
    dev = torch.device("cuda", 0)
    assert kernel.work_map_size(1, 128, 1, 1) == kernel.work_map_size(
        1, 128, 1, 1, device=dev
    )
    lens = torch.full((1,), 128 * P, dtype=torch.int32, device="cuda")
    rows = kernel.work_map_size(1, 128, 1, 1)
    buf = torch.empty((rows, 2), dtype=torch.int32, device="cuda")
    assert kernel.build_work_map(lens, 128, 1, 1, out=buf).shape[0] == rows


def test_prefill_reads_the_same_cu_override(monkeypatch):
    """Both scorers size their grids from the CU count, so an override that
    only one of them sees makes them disagree about the same machine. Prefill
    used to memoise its own copy, which froze whatever `CU_NUM` was at the
    first call -- and that copy was redundant: the `_auto_*` helpers that ask
    for it are themselves memoised, so the warm path never calls it at all.
    """
    from aiter.ops.flydsl.kernels import minimax_m3_index_score_prefill as pf

    assert pf._cu_count is kernel._cu_count
    from aiter.jit.utils import chip_info

    monkeypatch.setenv("CU_NUM", "64")
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 64)
    assert pf._cu_count() == 64
    # A later override has to be seen too, which a cached read would not be.
    monkeypatch.setenv("CU_NUM", "256")
    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 256)
    assert pf._cu_count() == 256


def test_supported_answers_for_the_whole_entry_point():
    """A dispatcher that trusts the predicate and routes here must not then
    take a raise. `_validate_metadata` skips `block_table`/`seq_lens` when they
    are None and never sees the scale, so the predicate checks both itself --
    it used to return True and let `score_flydsl` raise."""
    q = torch.empty(1, 1, D, dtype=torch.bfloat16)
    k = torch.empty(1, P, D, dtype=torch.bfloat16)
    bt = torch.zeros(1, 1, dtype=torch.int32)
    lens = torch.zeros(1, dtype=torch.int32)
    # Omitting them is a TypeError now, which is the right place to find out.
    with pytest.raises(TypeError):
        kernel.index_score_supported(q, k, 1, 1, 1)
    # Passing them as None explicitly still has to be a refusal, not a crash:
    # that is what a dispatcher holding an optional does.
    assert not kernel.index_score_supported(q, k, 1, 1, 1, None, seq_lens=lens)
    assert not kernel.index_score_supported(q, k, 1, 1, 1, bt, seq_lens=None)
    # And the scale, which the entry point rejects outside FP32's normal range.
    assert not kernel.index_score_supported(
        q, k, 1, 1, 1, bt, seq_lens=lens, sm_scale=1e39
    )


def test_auto_bounds_resolved_before_packing(monkeypatch):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 304)
    assert kernel.work_map_size(1, 262145, 1, 1) == 16385


def check_scores(got, ref):
    """Undefined slots may change; required -inf and finite slots may not."""
    defined = ~torch.isnan(ref)
    assert torch.equal(torch.isneginf(got) & defined, torch.isneginf(ref))
    finite = torch.isfinite(ref)
    err = checkAllclose(
        ref[finite].float(), got[finite].float(), atol=ATOL, rtol=RTOL, tol_err_ratio=0
    )
    assert err == 0, f"score mismatch fraction {err}"
    return err


@pytest.mark.skip(
    reason="perf sweep: run via python op_tests/test_flydsl_minimax_m3_index_score.py"
)
@benchmark()
def test_score(case, B, S, H, lens, layout, kwargs):
    q, cache, bt, seq, mb = make_case(B, S, H, lens, **kwargs)
    ref = score_oracle(q, cache, bt, seq, S, H, D**-0.5, mb)
    out = (
        torch.empty((H, B * S, mb), device=q.device, dtype=torch.float32)
        if layout == "contiguous"
        else torch.empty((mb, B * S, H), device=q.device, dtype=torch.float32).permute(
            2, 1, 0
        )
    )
    ret = {"gfx": get_gfx(), "timing": "repeated-buffer (warm effective TB/s)"}
    g = geometry(B, mb, S, H, device=q.device)
    ret["geometry"] = f"tw{g['waves_per_tok']}_L{g['pages_per_wave']}" + (
        "_sp" if g["spread"] else ""
    )
    capacity = kernel.work_map_capacity(B, mb, S, H)
    buf = torch.empty((capacity, 2), device=q.device, dtype=torch.int32)
    wm = kernel.build_work_map(seq, mb, S, H, out=buf)
    for name in ("flydsl",):

        def run(wm=wm):
            return score_flydsl(
                q, cache, bt, seq, S, H, D**-0.5, mb, out=out, work_map=wm
            )

        got, us = run_perftest(run, num_iters=10, num_rotate_args=1)
        assert got.data_ptr() == out.data_ptr()
        err = check_scores(got, ref)
        pages = sum((L + P - 1) // P for L in lens)
        flops = 2 * pages * P * D * S * H
        nbytes = (
            pages * P * D * cache.element_size() + q.numel() * 2 + pages * S * H * 4
        )
        ret.update(
            {
                f"{name} us": us,
                f"{name} TFLOPS": flops / us / 1e6,
                f"{name} TB/s": nbytes / us / 1e6,
                f"{name} err": err,
            }
        )
    return ret


# ==========================================================================
# PREFILL
#
# The judge is aiter's own `pa_sparse_block_score_prefill`. Comparing against
# it rather than a torch oracle is deliberate: the two implementations share no
# code and the operator's numerics are a contract, so the comparison is exact
# rather than tolerance-based.
#
# Two properties of that contract shape every check below:
#
#   * `out` may be left untouched past a query's causal window, so the buffer is
#     sentinel-filled and compared only where aiter marks it finite.
#     `alloc_score` hands back `torch.empty`, so a raw diff would be comparing
#     uninitialised memory.
#   * every config arm must agree bit for bit. These are the same arithmetic in
#     different geometries, not different approximations of it.


def make_prefill_case(batch, q_len, ctx_len, heads=1, seed=0):
    """A uniform ragged batch: every request has q_len new tokens on ctx_len.

    `heads` has to build the Q it promises: sizing the score buffer and the
    geometry by it while Q keeps one head makes every head sweep either an
    aiter shape rejection or an `out` rejection, so the axis reads as covered
    and runs nothing.
    """
    torch.manual_seed(seed)
    dev = "cuda"
    seq_lens = torch.full((batch,), ctx_len, dtype=torch.int32, device=dev)
    prefix = torch.full((batch,), ctx_len - q_len, dtype=torch.int32, device=dev)
    cu = torch.arange(0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=dev)
    max_block = -(-ctx_len // P)
    npages = batch * max_block
    q8 = (torch.randn(batch * q_len, heads, D, device=dev) / 4).to(dtypes.fp8)
    cache = (torch.randn(npages, P, D, device=dev) / 4).to(dtypes.fp8)
    # The block table is a permutation on purpose: a kernel that assumed pages
    # were visited in storage order would still pass on an identity table.
    bt = (
        torch.randperm(npages, device=dev, dtype=torch.int32)
        .view(batch, max_block)
        .contiguous()
    )
    return {
        "q8": q8,
        # aiter takes Q in fp8; the bf16 arm gets its exact upcast, so neither
        # side is answering a different question.
        "q_bf16": q8.float().bfloat16(),
        "cache": cache,
        "bt": bt,
        "cu": cu,
        "seq": seq_lens,
        "prefix": prefix,
        "max_q": q_len,
        "max_seq": ctx_len,
        "batch": batch,
        "mb": max_block,
    }


def aiter_scores(c):
    """aiter's scores, and the mask of slots it considers live."""
    heads = c["q8"].shape[1]
    ref = torch.full((heads, c["q8"].shape[0], c["mb"]), -float("inf"), device="cuda")
    pa_sparse_block_score_prefill(
        c["q8"],
        c["cache"],
        ref,
        c["bt"],
        c["cu"],
        c["seq"],
        max_query_len=c["max_q"],
        max_seq_len=c["max_seq"],
    )
    return ref, torch.isfinite(ref)


def run_prefill(c, cache=None, q=None, out=None):
    heads = c["q8"].shape[1]
    if out is None:
        out = prefill_alloc_score(c["q8"].shape[0], heads, c["mb"], "cuda")
    out.fill_(SENTINEL)
    score_prefill_flydsl(
        c["q_bf16"] if q is None else q,
        c["cache"] if cache is None else cache,
        c["bt"],
        c["cu"],
        c["seq"],
        c["prefix"],
        c["max_q"],
        c["max_seq"],
        1.0 / LOG2E,
        out=out,
    )
    return out.reshape(heads, c["q8"].shape[0], c["mb"])


def assert_matches_aiter(got, ref, live, what):
    unwritten = int((live & (got == SENTINEL)).sum())
    assert unwritten == 0, f"{what}: {unwritten} live slots left unwritten"
    diff = (got[live] - ref[live]).abs().max().item()
    assert diff == 0.0, f"{what}: max abs diff {diff} against aiter"


# (batch, q_len, ctx_len). Covers the short/grid-starved end, the ragged and
# non-power-of-two lengths, and contexts past the 32 MB point where the score
# tensor stops fitting in L2 (which is what sets the store merge width).
PREFILL_SHAPES = [
    (1, 256, 256),
    (1, 130, 1000),
    (3, 777, 5000),
    (1, 2048, 2048),
    (4, 512, 8192),
    (8, 512, 16384),
    (2, 2048, 32768),
    (1, 1024, 65536),
    (1, 4096, 131072),
    (1, 4096, 524288),
]

# One (batch, q_len, ctx) per prefill_geometry `prefill_geometry` can reach, found by
# enumerating its output over the whole envelope. The arms are driven through
# the real selector rather than a knob, so the suite covers exactly what
# production compiles -- and a prefill_geometry that stops being reachable shows up as
# a failure in `test_geometry_arms_are_exhaustive`, not as a silent gap.
GEOM_SHAPES = [
    (1, 128, 256),  # tile_q 64, 4 waves, L1, 16-wide (odd FEAT_TILES_PER_WAVE)
    (8, 512, 1000),  # tile_q 128, 4 waves, L1
    (8, 512, 2048),  # tile_q 128, 4 waves, L2
    (2, 128, 16384),  # tile_q 256, 8 waves, L1
    (4, 128, 16384),  # tile_q 256, 8 waves, L2
    (8, 2048, 16384),  # tile_q 256, 8 waves, L4 -- the production point
]


@requires_gfx950
@pytest.mark.parametrize(
    "shape", PREFILL_SHAPES, ids=lambda s: f"b{s[0]}_q{s[1]}_ctx{s[2]}"
)
def test_matches_aiter(shape):
    c = make_prefill_case(*shape)
    ref, live = aiter_scores(c)
    assert_matches_aiter(run_prefill(c), ref, live, f"{shape}")


@requires_gfx950
@pytest.mark.parametrize(
    "shape", GEOM_SHAPES, ids=lambda s: f"b{s[0]}_q{s[1]}_ctx{s[2]}"
)
def test_every_geometry_matches_aiter(shape):
    """Each prefill_geometry is the same arithmetic over a different split of the query
    tile, CTA width and page depth, judged bit for bit against aiter."""
    c = make_prefill_case(*shape)
    ref, live = aiter_scores(c)
    assert_matches_aiter(run_prefill(c), ref, live, f"{shape}")


@requires_gfx950
def test_geometry_arms_are_exhaustive():
    """Every prefill_geometry the selector can reach has an arm above.

    Without this, narrowing `prefill_geometry` silently drops coverage: the arms keep
    passing because they all collapse onto the surviving shape.
    """
    dev = torch.device("cuda")
    reachable = set()
    for ctx in (256, 1000, 2048, 5000, 8192, 16384, 32768, 131072, 524288):
        for q_len in (128, 130, 256, 512, 777, 1024, 2048, 4096):
            for batch in (1, 2, 3, 4, 8):
                if q_len > ctx:
                    continue
                g = prefill_geometry(q_len, batch, 1, -(-ctx // P), dev)
                reachable.add((g["tile_q"], g["waves"], g["pages_per_wave"], g["m32"]))
    covered = set()
    for batch, q_len, ctx in GEOM_SHAPES:
        g = prefill_geometry(q_len, batch, 1, -(-ctx // P), dev)
        covered.add((g["tile_q"], g["waves"], g["pages_per_wave"], g["m32"]))
    assert covered == reachable, f"uncovered: {sorted(reachable - covered)}"


@requires_gfx950
def test_mixed_query_lengths_match_aiter():
    """A short request next to a long one, which is where a dead tile costs.

    The grid is sized by `max_query_len`, so request B (one query row) gets as
    many tiles as request A (8192). Those tiles store nothing either way; what
    changed is that they now retire at the page-range guard instead of walking
    B's whole causal window first. Correctness is the part that has to be
    pinned here -- the saving is in the sweep.
    """
    dev = "cuda"
    torch.manual_seed(0)
    q_a, ctx_a, ctx_b = 1024, 1024, 16384
    cu = torch.tensor([0, q_a, q_a + 1], dtype=torch.int32, device=dev)
    seq = torch.tensor([ctx_a, ctx_b], dtype=torch.int32, device=dev)
    prefix = torch.tensor([ctx_a - q_a, ctx_b - 1], dtype=torch.int32, device=dev)
    mb = -(-ctx_b // P)
    npages = 2 * mb
    q8 = (torch.randn(q_a + 1, 1, D, device=dev) / 4).to(dtypes.fp8)
    cache = (torch.randn(npages, P, D, device=dev) / 4).to(dtypes.fp8)
    bt = torch.randperm(npages, device=dev, dtype=torch.int32).view(2, mb).contiguous()
    ref = torch.full((1, q_a + 1, mb), -float("inf"), device=dev)
    pa_sparse_block_score_prefill(
        q8, cache, ref, bt, cu, seq, max_query_len=q_a, max_seq_len=ctx_b
    )
    live = torch.isfinite(ref)
    out = prefill_alloc_score(q_a + 1, 1, mb, dev)
    out.fill_(SENTINEL)
    score_prefill_flydsl(
        q8.float().bfloat16(),
        cache,
        bt,
        cu,
        seq,
        prefix,
        q_a,
        ctx_b,
        1.0 / LOG2E,
        out=out,
    )
    got = out.reshape(1, q_a + 1, mb)
    assert_matches_aiter(got, ref, live, "mixed query lengths")


@requires_gfx950
def test_trailing_empty_request_stays_in_bounds():
    """`cu_seqlens_q[-2] == cu_seqlens_q[-1]`: the last request has no rows.

    Its tiles still stage Q, and the `q < q_len` clamp sent them to row
    `seq_start + 0`, which for a TRAILING empty request is `total_q` -- one row
    past `idx_q`. The buffer view's num_records is the view maximum, not this
    tensor, so nothing caught it.

    What this pins is the contract -- a trailing empty request changes no live
    score. The read being in bounds is NOT observable from here, since the
    staged value is discarded either way, so this test would also have passed
    before the clamp.
    """
    dev = "cuda"
    torch.manual_seed(0)
    q_len, ctx = 256, 2048
    mb = -(-ctx // P)
    q8 = (torch.randn(q_len, 1, D, device=dev) / 4).to(dtypes.fp8)
    cache = (torch.randn(2 * mb, P, D, device=dev) / 4).to(dtypes.fp8)
    bt = torch.randperm(2 * mb, device=dev, dtype=torch.int32).view(2, mb).contiguous()
    seq = torch.tensor([ctx, ctx], dtype=torch.int32, device=dev)
    prefix = torch.tensor([ctx - q_len, ctx], dtype=torch.int32, device=dev)
    q = q8.float().bfloat16()

    def run(cu):
        out = prefill_alloc_score(q_len, 1, mb, dev)
        out.fill_(SENTINEL)
        score_prefill_flydsl(
            q, cache, bt, cu, seq, prefix, q_len, ctx, 1.0 / LOG2E, out=out
        )
        return out.reshape(1, q_len, mb)

    alone = run(torch.tensor([0, q_len], dtype=torch.int32, device=dev))
    trailing = run(torch.tensor([0, q_len, q_len], dtype=torch.int32, device=dev))
    assert torch.equal(alone, trailing)


@requires_gfx950
def test_prefill_refuses_overlapping_pages():
    """`page_buffer` advances by `cache.stride(0)`, so a page stride under one
    whole page makes every page overlap the one before it. The rows inside a
    page are still contiguous, so the per-page checks pass and the kernel
    quietly scores the wrong pages."""
    c = make_prefill_case(1, 128, 2048)
    flat = c["cache"].reshape(-1)
    overlap = torch.as_strided(flat, (c["mb"], P, D), (1, D, 1))
    with pytest.raises(ValueError, match="non-overlapping pages"):
        run_prefill(c, cache=overlap)


@requires_gfx950
def test_prefill_refuses_misaligned_pages():
    """Clearing a whole page is not the same as starting on a 16-byte boundary.
    `ptr_buf_tensor` declares 16-byte alignment and the copy atom moves 16 B at
    a time, so an outer stride of 16385 fp8 elements leaves every page after
    the first unaligned -- in bounds, so only the alignment rule catches it.
    Decode already refused this; prefill accepted it."""
    c = make_prefill_case(1, 128, 512)
    flat = torch.empty(c["mb"] * 16385 + 16385, dtype=dtypes.fp8, device="cuda")
    skew = torch.as_strided(flat, (c["mb"], P, D), (16385, D, 1))
    with pytest.raises(ValueError, match="16-byte aligned"):
        run_prefill(c, cache=skew)


@requires_gfx950
@pytest.mark.parametrize("kernel_kind", ["decode", "prefill"])
def test_default_score_buffer_reads_as_minus_inf(kernel_kind):
    """Both kernels store only the pages inside a row's window, so the rest of
    the buffer is whatever was there. Allocated with `empty` that is arbitrary
    memory, and the top-k selector downstream can pick a block that does not
    exist. The slots the kernel skips have to read as the sentinel."""
    neg_inf = -float("inf")
    if kernel_kind == "prefill":
        # q256/ctx256 leaves a quarter of the buffer unstored. Called directly
        # rather than through `run_prefill`, which fills its own sentinel and
        # would hide exactly what this is checking.
        c = make_prefill_case(1, 256, 256)
        out = score_prefill_flydsl(
            c["q_bf16"],
            c["cache"],
            c["bt"],
            c["cu"],
            c["seq"],
            c["prefix"],
            c["max_q"],
            c["max_seq"],
            1.0 / LOG2E,
        )
    else:
        # A ragged batch: the short requests never store to their tail pages.
        lens = [512, 256, 128, 512]
        q, cache, bt, seq, mb = make_case(len(lens), 1, 1, lens)
        out = score_flydsl(q, cache, bt, seq, 1, 1, D**-0.5, mb)
    assert (out == neg_inf).any(), "nothing was skipped; the case proves nothing"
    assert torch.isfinite(out[out != neg_inf]).all()


@requires_gfx950
def test_prefill_refuses_overlapping_q_rows():
    """A head stride large enough is not enough: the ROW stride has to clear
    every head. `[rows, 2, 128]` strided (128, 128, 1) passes the head check
    while row 1 aliases head 1 of row 0."""
    c = make_prefill_case(1, 128, 2048, 2)
    rows = c["q8"].shape[0]
    flat = torch.empty(rows * 2 * D, dtype=torch.bfloat16, device="cuda")
    alias = torch.as_strided(flat, (rows, 2, D), (D, D, 1))
    out = prefill_alloc_score(rows, 2, c["mb"], "cuda")
    with pytest.raises(ValueError, match="may overlap"):
        score_prefill_flydsl(
            alias,
            c["cache"],
            c["bt"],
            c["cu"],
            c["seq"],
            c["prefix"],
            c["max_q"],
            c["max_seq"],
            1.0 / LOG2E,
            out=out,
        )


@requires_gfx950
def test_prefill_launcher_cache_is_per_device():
    """The launcher memoises its compiled handle, so one shared between two
    devices would dispatch the first device's handle on the second."""
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible gfx950 GPUs")
    from aiter.ops.flydsl.kernels import minimax_m3_index_score_prefill as pf

    g = prefill_geometry(128, 1, 1, 16, torch.device("cuda", 0))
    a = pf._get(g, torch.device("cuda", 0))
    b = pf._get(g, torch.device("cuda", 1))
    assert a is not b


@requires_gfx950
@pytest.mark.parametrize("stride1", [129, 130])
def test_prefill_refuses_misaligned_q_rows(stride1):
    """Non-overlapping is not enough: `q_fp8_at` turns the bf16 element offset
    into a dword offset and vector-loads it, so an odd or 2-mod-16 head stride
    reads off its boundary. 129 is the reviewer's case -- it passes the
    overlap check and still has head 1 starting at element 128."""
    c = make_prefill_case(1, 128, 2048)
    q = torch.empty_strided(
        (c["q8"].shape[0], 2, D),
        (2 * stride1, stride1, 1),
        dtype=torch.bfloat16,
        device="cuda",
    )
    out = prefill_alloc_score(c["q8"].shape[0], 2, c["mb"], "cuda")
    with pytest.raises(ValueError, match="16-byte aligned"):
        score_prefill_flydsl(
            q,
            c["cache"],
            c["bt"],
            c["cu"],
            c["seq"],
            c["prefix"],
            c["max_q"],
            c["max_seq"],
            1.0 / LOG2E,
            out=out,
        )


@requires_gfx950
@pytest.mark.parametrize("scale", [-1.0, 0.0, float("nan"), float("inf"), 1e-50, 1e39])
def test_prefill_scale_must_be_order_preserving(scale):
    """The epilogue takes each page's max and only then multiplies by the
    scale, so a negative one returns `max(dot) * scale`, not `max(dot * scale)`
    -- the wrong block, silently. Decode already refused these."""
    c = make_prefill_case(1, 128, 2048)
    out = prefill_alloc_score(c["q8"].shape[0], 1, c["mb"], "cuda")
    with pytest.raises(ValueError, match="sm_scale"):
        score_prefill_flydsl(
            c["q_bf16"],
            c["cache"],
            c["bt"],
            c["cu"],
            c["seq"],
            c["prefix"],
            c["max_q"],
            c["max_seq"],
            scale,
            out=out,
        )


@requires_gfx950
@pytest.mark.parametrize(
    "max_q,max_seq",
    [
        (0, 2048),
        (-1, 2048),
        (128, 0),
        (128, -1),
        (128.0, 2048),
        (128, 2048.0),
        (0x7FFFFFFF, 2048),
        (128, 0x7FFFFFFF),
    ],
)
def test_rejects_out_of_range_launch_bounds(max_q, max_seq):
    """The grid is sized from these two and nothing downstream re-checks them.

    `max_query_len=0` used to launch an empty query grid and return silently --
    the worst outcome, since the caller gets a buffer of sentinels and no
    signal. `max_seq_len=0` reached `hipModuleLaunchKernel` as a zero dimension
    and surfaced as `hipErrorInvalidValue`, which says nothing about which
    argument was wrong. At the other end both feed i32 token arithmetic, and
    `max_seq_len` rounded up to a page must still fit -- so the bound has to be
    refused before `alloc_score` sizes a buffer from it.
    """
    c = make_prefill_case(1, 128, 2048)
    with pytest.raises(ValueError, match="must be in"):
        score_prefill_flydsl(
            c["q_bf16"],
            c["cache"],
            c["bt"],
            c["cu"],
            c["seq"],
            c["prefix"],
            max_q,
            max_seq,
            1.0 / LOG2E,
        )


@requires_gfx950
def test_strided_cache_is_refused():
    """The LDS staging path reads a page as a flat run of 16 B chunks, which
    only describes the right bytes when the page is contiguous -- a strided
    page reads in bounds and scores the wrong features (0.209 max abs error on
    a [pages,128,256][..., :128] view).

    There is no second code path to fall back to any more, so the contract is
    refusal. Scoring it anyway is silently wrong, which is the one outcome a
    caller cannot detect.
    """
    c = make_prefill_case(1, 1024, 16384)
    wide = torch.zeros(
        c["cache"].shape[0], P, 2 * D, dtype=c["cache"].dtype, device="cuda"
    )
    wide[..., :D] = c["cache"]
    strided = wide[..., :D]
    assert strided.stride(1) == 2 * D and not strided.is_contiguous()
    with pytest.raises(ValueError, match="contiguous non-overlapping pages"):
        run_prefill(c, strided)
    # ...and the same cache, packed, still scores.
    ref, live = aiter_scores(c)
    assert_matches_aiter(run_prefill(c), ref, live, "contiguous")


@requires_gfx950
def test_merged_store_follows_the_out_layout():
    """`contig_k` merges a run of consecutive blocks into one vector store, and
    it is a property of the OUTPUT the caller passed, not a tuning knob. A
    padded row stride breaks the alignment the merge needs, so the kernel must
    fall back to per-block stores and still be bit-exact."""
    batch, q_len, ctx = 8, 2048, 16384  # the L4 arm, so a run is 4 blocks wide
    c = make_prefill_case(batch, q_len, ctx)
    ref, live = aiter_scores(c)
    rows, mb = c["q8"].shape[0], c["mb"]
    g = prefill_geometry(c["max_q"], batch, 1, mb, torch.device("cuda"))
    ppw = g["pages_per_wave"]
    assert ppw > 1, "pick a shape whose prefill_geometry actually merges stores"
    # A row stride that is NOT a multiple of the run width, so the merge is off.
    pad = next(mb + d for d in (1, 2, 3) if (mb + d) % ppw)
    padded = torch.empty((1, rows, pad), dtype=torch.float32, device="cuda")
    view = padded[:, :, :mb]
    got = run_prefill(c, out=view)
    assert_matches_aiter(got, ref, live, "padded out stride")
    assert torch.equal(got[live], run_prefill(c)[live])


@requires_gfx950
def test_rejects_malformed_tensors():
    """Everything below reaches the kernel as a bare pointer, so a wrong dtype
    is a silent reinterpretation rather than an error."""
    c = make_prefill_case(1, 256, 2048)
    base = (
        c["q_bf16"],
        c["cache"],
        c["bt"],
        c["cu"],
        c["seq"],
        c["prefix"],
        c["max_q"],
        c["max_seq"],
        1.0 / LOG2E,
    )
    score_prefill_flydsl(*base)  # the well-formed call must still work

    def swap(i, v):
        a = list(base)
        a[i] = v
        return a

    # Only stride(0)/stride(1) reach the kernel; every other index is built as
    # a packed element offset. A strided view of the right dtype and shape is
    # therefore read as if it were contiguous -- in bounds, wrong values -- so
    # each of these must be refused rather than scored.
    wide_q = torch.zeros(
        c["q_bf16"].shape[0], 1, 2 * D, dtype=c["q_bf16"].dtype, device="cuda"
    )
    wide_q[..., :D] = c["q_bf16"]
    wide_bt = torch.zeros(
        c["bt"].shape[0], 2 * c["bt"].shape[1], dtype=torch.int32, device="cuda"
    )
    wide_bt[:, ::2] = c["bt"]
    for idx, bad in (
        (4, c["seq"].long()),  # int64 seq_lens
        (5, c["prefix"].long()),  # int64 prefix_lens
        (3, c["cu"].long()),  # int64 cu_seqlens_q
        (2, c["bt"].long()),  # int64 block table
        (2, c["bt"][0]),  # 1-D block table
        (0, wide_q[..., ::2]),  # non-unit Q feature stride
        (2, wide_bt[:, ::2]),  # non-unit block-table column stride
        (3, c["cu"].repeat_interleave(2)[::2]),  # strided cu_seqlens_q
        (4, c["seq"].repeat_interleave(2)[::2]),  # strided seq_lens
        (5, c["prefix"].repeat_interleave(2)[::2]),  # strided prefix_lens
    ):
        with pytest.raises(ValueError):
            score_prefill_flydsl(*swap(idx, bad))

    with pytest.raises(ValueError):  # out sized for a smaller max_block
        score_prefill_flydsl(
            *base, out=prefill_alloc_score(c["q8"].shape[0], 1, c["mb"] // 2, "cuda")
        )


@requires_gfx950
def test_rejects_overlapping_out():
    """A page-contiguous block axis does not make the rows disjoint. At row
    stride 1 every query shares storage with its neighbour, so the kernel
    overwrites scores it already wrote (3.26 max abs error) while the call
    reports success. Each axis has to clear the extent below it."""
    c = make_prefill_case(1, 256, 2048)
    tq, mb = c["q8"].shape[0], c["mb"]
    buf = torch.full((tq * mb,), SENTINEL, dtype=torch.float32, device="cuda")

    def run_with(shape, stride):
        out = torch.as_strided(buf, shape, stride)
        score_prefill_flydsl(
            c["q_bf16"],
            c["cache"],
            c["bt"],
            c["cu"],
            c["seq"],
            c["prefix"],
            c["max_q"],
            c["max_seq"],
            1.0 / LOG2E,
            out=out,
        )

    for stride in (
        (tq * mb, 1, 1),  # rows alias
        (tq * mb, mb - 1, 1),  # rows overlap by one block
    ):
        with pytest.raises(ValueError, match="non-overlapping"):
            run_with((1, tq, mb), stride)
    # A zero head stride is rejected too, but by `_validate_tensor` before this
    # check sees it -- assert the rejection, not which guard caught it.
    with pytest.raises(ValueError):
        run_with((1, tq, mb), (0, mb, 1))

    # Exactly-packed and padded-but-disjoint are both still accepted.
    ref, live = aiter_scores(c)
    assert_matches_aiter(run_prefill(c, c["cache"]), ref, live, "packed out")
    wide = torch.full((1, tq, 2 * mb), SENTINEL, dtype=torch.float32, device="cuda")
    padded = wide[:, :, :mb]
    assert padded.stride() == (tq * 2 * mb, 2 * mb, 1)
    score_prefill_flydsl(
        c["q_bf16"],
        c["cache"],
        c["bt"],
        c["cu"],
        c["seq"],
        c["prefix"],
        c["max_q"],
        c["max_seq"],
        1.0 / LOG2E,
        out=padded,
    )
    assert_matches_aiter(padded.reshape(ref.shape), ref, live, "padded out")


# --------------------------------------------------------------------------- #
# Performance sweep
# --------------------------------------------------------------------------- #


def _prefill_work(batch, q_len, ctx_len):
    """(live (query, block) pairs, unique pages) for a uniform ragged batch.

    A query row i of a request sees `prefix + i + 1` context tokens, so it
    touches `ceil(that / P)` blocks. Summing the rows is what the kernel
    actually MFMAs; the unique page count is the K traffic a perfect cache
    would move, which is the honest roofline denominator -- the kernel re-reads
    a page once per query tile and counting those re-reads would flatter TB/s.
    """
    prefix = ctx_len - q_len
    pairs = sum(-(-(prefix + i + 1) // P) for i in range(q_len)) * batch
    return pairs, batch * (-(-ctx_len // P))


@pytest.mark.skip(
    reason="perf sweep: run via python op_tests/test_flydsl_minimax_m3_index_score.py"
)
@benchmark()
def test_score_prefill_perf(batch, q_len, ctx_len, heads):
    c = make_prefill_case(batch, q_len, ctx_len, heads)
    total_q = c["q8"].shape[0]
    cache = c["cache"]
    # aiter takes the fp8 Q straight; this kernel takes the same values as
    # bf16 and rounds them back down at staging, which is bit-identical.
    q = c["q_bf16"]
    fp8 = True  # the cache dtype, which is what aiter's build is keyed on

    ret = {"gfx": get_gfx()}
    g = prefill_geometry(c["max_q"], batch, heads, c["mb"], torch.device("cuda"))
    ret["prefill_geometry"] = f"q{g['tile_q']}_w{g['waves']}_L{g['pages_per_wave']}"
    ref = live = None
    # aiter is both the judge and the baseline, but it is built for
    # num_idx_heads 1 and 2 only ("no build for num_idx_heads=4") and takes an
    # fp8 cache, so outside that it is neither timed nor used as the aiter_scores.
    if fp8 and heads <= 2:
        score = torch.full((heads, total_q, c["mb"]), -float("inf"), device="cuda")

        def run_aiter(s=score):
            pa_sparse_block_score_prefill(
                c["q8"],
                c["cache"],
                s,
                c["bt"],
                c["cu"],
                c["seq"],
                max_query_len=c["max_q"],
                max_seq_len=c["max_seq"],
            )
            return s

        ref, us = run_perftest(run_aiter, num_iters=10, num_rotate_args=1)
        live = torch.isfinite(ref)
        ret["aiter us"] = us
        ret["aiter err"] = 0.0

    pairs, pages = _prefill_work(batch, q_len, ctx_len)
    flops = 2 * pairs * P * D * heads
    nbytes = (
        pages * P * D * cache.element_size()  # K, read once at best
        + total_q * heads * D * q.element_size()  # Q
        + pairs * heads * 4  # the fp32 scores actually written
    )

    for name in ("flydsl",):
        out = prefill_alloc_score(total_q, heads, c["mb"], "cuda")
        out.fill_(SENTINEL)

        def run_flydsl(out=out):
            return score_prefill_flydsl(
                q,
                cache,
                c["bt"],
                c["cu"],
                c["seq"],
                c["prefix"],
                c["max_q"],
                c["max_seq"],
                1.0 / LOG2E,
                out=out,
            )

        got, us = run_perftest(run_flydsl, num_iters=10, num_rotate_args=1)
        assert got.data_ptr() == out.data_ptr()
        if ref is None:
            # No aiter build for this head count; the parametrized tests
            # above carry correctness, so report the timing without a verdict
            # rather than inventing one.
            err = float("nan")
        else:
            err = checkAllclose(
                ref[live].float(),
                got.reshape(ref.shape)[live].float(),
                rtol=0,
                atol=0,
                msg=f"{name}: prefill index score",
            )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@requires_gfx950
def test_every_lazy_export_resolves():
    """`aiter.ops.flydsl` exports these lazily, so a name that stopped
    existing is an AttributeError at the caller's import rather than here.

    `MiniMaxM3IndexScoreConfig` pointed at a class this file deleted and
    nothing noticed, because the module imports fine until someone asks for
    that attribute -- and the consumer is in another repository.

    Needs a GPU despite touching no kernel: resolving every name imports
    modules outside M3 whose own import queries the arch.
    """
    import aiter.ops.flydsl as F
    from aiter.ops.flydsl import _LAZY_IMPORTS

    for name in _LAZY_IMPORTS:
        getattr(F, name)  # raises if the target moved or was removed
    assert set(_LAZY_IMPORTS) <= set(F.__all__)
    for name in (
        "minimax_m3_index_score_flydsl",
        "minimax_m3_index_score_prefill_flydsl",
        "minimax_m3_index_score_alloc",
        "minimax_m3_index_score_prefill_alloc",
    ):
        assert name in _LAZY_IMPORTS, f"{name} is not reachable from the package"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="MiniMax-M3 index-score correctness and performance sweep",
        epilog=(
            "Without --mnk, --dtype/--batch/--case filter the unchanged regression "
            "matrix (including ragged and sliced-query cases). With --mnk, sweep "
            "dtype x batch x (S,H,L) using uniform lengths [L] * B; batch defaults "
            "to 2. --case filters names in either mode. Empty lists select no work."
        ),
    )
    ap.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=None,
        help="batch sizes: regression filter, or custom sweep (default: 2)",
    )
    ap.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=None,
        metavar="S,H,L",
        help="custom shapes: query tokens/request, index heads, sequence length; "
        "not GEMM M,N,K; page size and head dimension remain 128",
    )
    ap.add_argument(
        "--case",
        nargs="*",
        default=None,
        help="case names from build_matrix, or custom_B{B}_S{S}_H{H}_L{L}_{bf16|fp8}",
    )
    ap.add_argument(
        "--q-len", type=int, nargs="*", default=[4096], help="prefill new tokens"
    )
    ap.add_argument(
        "--ctx-len",
        type=int,
        nargs="*",
        default=[8192, 32768, 131072, 524288],
        help="prefill context length sweep",
    )
    ap.add_argument(
        "--heads", type=int, nargs="*", default=[1], help="prefill index head sweep"
    )
    ap.add_argument(
        "--layout",
        nargs="*",
        choices=["contiguous", "feature"],
        default=["contiguous", "feature"],
    )
    args = ap.parse_args(argv)
    if args.batch is not None and any(not 1 <= b <= 65535 for b in args.batch):
        ap.error("batch sizes must be in [1, 65535]")
    if args.mnk is not None:
        for shape in args.mnk:
            if (
                not isinstance(shape, tuple)
                or len(shape) != 3
                or any(dim <= 0 for dim in shape)
                or shape[2] < shape[0]
                or shape[2] > 0x7FFFFFFF - 127
            ):
                ap.error("--mnk expects positive S,H,L with S <= L <= INT32_MAX-127")
    return args


def select_cases(args):
    """CPU-only selection; custom shapes never replace the regression defaults."""
    if args.mnk is None:
        cases = [c for c in build_matrix() if args.batch is None or c[1] in args.batch]
    else:
        cases = [
            (f"custom_B{B}_S{S}_H{H}_L{L}_fp8", B, S, H, [L] * B, {})
            for B, (S, H, L) in itertools.product(
                args.batch if args.batch is not None else [2], args.mnk
            )
        ]
    if args.case is not None:
        cases = [c for c in cases if c[0] in args.case]
    return cases


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available() or get_gfx() not in DECODE_SUPPORTED_GFX:
        aiter.logger.warning("index score requires gfx950; skipping sweep")
        return 0
    rows = []
    for (name, B, S, H, lens, kw), layout in itertools.product(
        select_cases(args), args.layout
    ):
        rows.append(test_score(name, B, S, H, lens, layout, kw))
    if rows:
        aiter.logger.info(
            "decode index score summary (markdown):\n%s",
            pd.DataFrame(rows).to_markdown(index=False),
        )
    else:
        aiter.logger.warning("Empty selection; no decode score tests executed")

    # A second table, not a merged one: the two @benchmark signatures have
    # different call args, so one frame would scatter NaN columns.
    pre = []
    for batch, q_len, ctx_len, heads in itertools.product(
        args.batch or [1], args.q_len, args.ctx_len, args.heads
    ):
        if q_len > ctx_len:
            continue  # a request cannot have more new tokens than context
        pre.append(test_score_prefill_perf(batch, q_len, ctx_len, heads))
    if pre:
        aiter.logger.info(
            "prefill index score summary (markdown):\n%s",
            pd.DataFrame(pre).to_markdown(index=False),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
