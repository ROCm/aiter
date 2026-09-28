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
from aiter.test_common import benchmark, checkAllclose, run_perftest

LOG2E = 1.4426950409
P = 128  # Public operator contract, independent of the kernel implementation.
D = 128
# Score tensor tolerance, matching test_m3_indexer_context_parallel.py:80.
ATOL = RTOL = 3e-5

# Sentinel for "the kernel is contractually allowed to leave this untouched"
# NaN marks undefined reference slots; writes there are deliberately ignored.
UNWRITTEN = float("nan")


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
    IndexScoreConfig,
    score_flydsl,
    selection_filter,
    shuffle_cache,
)

CANDIDATES = {}

# Default first, then one knob at a time, then the interactions. Named so a
# failure line says which knob broke.
for _tag, _cfg in [
    ("flydsl", {}),
    ("fly_shuf", {"shuffled": True}),
    ("fly_l2", {"pages_per_wave": 2}),
    ("fly_qlds", {"q_to_lds": True}),
    ("fly_fw2", {"feat_waves": 2}),
    ("fly_fw4", {"feat_waves": 4}),
    ("fly_fw2q", {"feat_waves": 2, "q_to_lds": True}),
    ("fly_tw2", {"token_waves": 2}),
    ("fly_tw4", {"token_waves": 4}),
    ("fly_tw2s", {"token_waves": 2, "shuffled": True}),
    ("fly_tw2l2", {"token_waves": 2, "pages_per_wave": 2}),
    ("fly_fw2tw2", {"feat_waves": 2, "token_waves": 2}),
    (
        "fly_allon",
        {"feat_waves": 2, "q_to_lds": True, "shuffled": True, "pages_per_wave": 2},
    ),
    (
        "fly_allon_tw",
        {
            "feat_waves": 2,
            "token_waves": 2,
            "q_to_lds": True,
            "shuffled": True,
            "pages_per_wave": 2,
        },
    ),
]:
    CANDIDATES[_tag] = IndexScoreConfig(**_cfg)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def make_case(B, S, H, lens, cache_dtype, seed=0, q_zero=False, q_slice=False):
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
    cache = torch.randn(num_pages, P, D, dtype=torch.bfloat16, device=dev)
    if cache_dtype != torch.bfloat16:
        cache = cache.to(cache_dtype)
    block_table = (
        torch.randperm(num_pages, device=dev, dtype=torch.int32)
        .view(B, max_block)
        .contiguous()
    )
    return idx_q, cache, block_table, seq_lens, max_block


def build_matrix(fp8_dtype):
    """The shape matrix. (name, B, S, H, lens, dtype, kwargs)."""
    cases = []
    for dt, tag in ((torch.bfloat16, "bf16"), (fp8_dtype, "fp8")):
        # F = S*H sweep: below one tile, exactly one tile, multiple tiles.
        for S, H in ((1, 1), (4, 1), (4, 4), (8, 4), (16, 4)):
            cases.append((f"F{S*H}_s4096_{tag}", 2, S, H, [4096, 4096], dt, {}))
        # Page boundaries and partial tails.
        for L in (4, 127, 128, 129, 130, 513, 4096, 8192):
            cases.append((f"L{L}_{tag}", 2, 4, 4, [L, L], dt, {}))
        # The case the contract calls out by name: four queries whose causal
        # cutoffs straddle a page boundary (127/128/129/130).
        cases.append((f"causal_straddle_{tag}", 1, 4, 4, [130], dt, {}))
        # Ragged batch: per-request loop bounds must be independent.
        cases.append((f"ragged_{tag}", 4, 4, 4, [130, 4096, 127, 513], dt, {}))
        # The block -> (request, chunk) map is a prefix sum over per-request
        # chunk counts, so these two are its edge cases: a request that
        # contributes zero chunks (its cum entry repeats, and the map must skip
        # it rather than hand it work), and a skew wide enough that the holes
        # outnumber the work by 7:1.
        cases.append((f"ragged_zero_{tag}", 4, 4, 4, [0, 4096, 1, 513], dt, {}))
        cases.append((f"ragged_skew_{tag}", 8, 4, 4, [8192] + [128] * 7, dt, {}))
        cases.append((f"qzero_tie_{tag}", 2, 4, 4, [513, 513], dt, {"q_zero": True}))
        cases.append((f"qslice_{tag}", 2, 4, 2, [513, 513], dt, {"q_slice": True}))
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
    assert not kernel.index_score_supported(q, k, 1, 1, 1)


def test_capacity_threshold(monkeypatch):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 16)
    cfg = IndexScoreConfig()
    # S16 H4 has no decode depth table, so this is the capacity estimate.
    exact = kernel.work_map_size(64, 16, 16, 4, cfg)
    smaller = kernel.work_map_size(63, 16, 16, 4, cfg)
    assert smaller > exact  # reproduces the non-monotonic sizing cliff
    assert kernel.work_map_capacity(64, 16, 16, 4, cfg) >= smaller
    # The depth tables are non-monotonic too (deeper at larger batches).
    assert kernel.work_map_size(16, 16, 4, 1) > kernel.work_map_size(17, 16, 4, 1)
    assert kernel.work_map_capacity(17, 16, 4, 1) >= kernel.work_map_size(16, 16, 4, 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.uint8])
def test_reject_cache_dtype(dtype):
    q, k, bt, lens, mb = make_case(1, 1, 1, [128], torch.bfloat16)
    with pytest.raises(ValueError, match="cache"):
        score_flydsl(q, k.to(dtype), bt, lens, 1, 1, D**-0.5, mb)


def test_reject_misaligned_cache():
    q, _, bt, _lens, mb = make_case(1, 1, 1, [128], torch.bfloat16)
    k = torch.empty_strided(
        (1, P, D), (16512, 129, 1), dtype=torch.bfloat16, device="cuda"
    )
    assert not kernel.index_score_supported(q, k, 1, 1, mb, bt)


def test_narrow_block_table_rejected():
    """Fewer columns than max_block is an out-of-bounds table load.

    Only the row count and the strides used to be checked, so a table narrower
    than the capacity it is paired with was accepted and read past its end.
    """
    q, k, bt, lens, mb = make_case(2, 4, 4, [4096, 4096], torch.bfloat16)
    assert bt.shape[1] == mb > 1
    narrow = bt[:, : mb - 1].contiguous()
    assert not kernel.index_score_supported(q, k, 4, 4, mb, narrow)
    with pytest.raises(ValueError, match="block_table"):
        score_flydsl(q, k, narrow, lens, 4, 4, D**-0.5, mb)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("shuffled", [False, True])
def test_large_cache(dtype, shuffled):
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
    got = score_flydsl(q, cache, bt, lens, 1, 1, D**-0.5, 3, shuffled=shuffled)
    ref = torch.tensor([1, 2, 3], device="cuda") * D**0.5 * LOG2E
    torch.testing.assert_close(got[0, 0], ref, atol=ATOL, rtol=RTOL)


@pytest.mark.parametrize("request_id", [32767, 32768, 65534])
def test_packed_request(request_id):
    batch = request_id + 1
    q = torch.ones((batch, 1, D), dtype=torch.bfloat16, device="cuda")
    q[request_id].fill_(2)
    k = torch.ones((1, P, D), dtype=torch.bfloat16, device="cuda")
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


@pytest.mark.parametrize("world,rank", [(1, 0), (2, 0), (2, 1), (4, 3)])
@pytest.mark.parametrize("S,H", [(1, 1), (1, 4), (4, 2), (8, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("layout", ["contiguous", "feature"])
@pytest.mark.parametrize("map_in_graph", [False, True])
def test_graph_buffers(world, rank, S, H, dtype, layout, map_in_graph):
    B, mb = 3, 9  # Stable envelope larger than actual lengths, including CP.
    if (S, H) == (1, 4):
        cu = torch.cuda.get_device_properties("cuda").multi_processor_count
        mb = 2 * (cu // B)  # Exercise the two-token-wave automatic branch.
        assert kernel.resolve_config(B, mb, IndexScoreConfig(), S, H).token_waves == 2
    q, cache, _, seq, _ = make_case(B, S, H, [0, 513, 1], dtype)
    bt = torch.zeros((B, mb * world), dtype=torch.int32, device="cuda")
    bt[:, :5] = torch.arange(5, device="cuda", dtype=torch.int32)
    cfg = IndexScoreConfig(cp_world=world, cp_rank=rank)
    buf = torch.empty(
        (kernel.work_map_capacity(B, mb, S, H, cfg), 2),
        device="cuda",
        dtype=torch.int32,
    )
    wm = kernel.build_work_map(seq, mb, S, H, cfg, out=buf)
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
            kernel.build_work_map(seq, mb, S, H, cfg, out=buf) if map_in_graph else wm
        )
        return score_flydsl(
            q, cache, bt, seq, S, H, D**-0.5, mb, out=out, cfg=cfg, work_map=current
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
            kernel.build_work_map(seq, mb, S, H, cfg, out=buf)
        graph.replay()
        ref = score_oracle(q, cache, bt, seq, S, H, D**-0.5, mb, world, rank)
        check_scores(out, ref)
        assert ptrs == (out.data_ptr(), buf.data_ptr(), wm.data_ptr())
        assert wm.shape[0] == kernel.work_map_size(B, mb, S, H, cfg)


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
    q, k, bt, lens, mb = make_case(2, 4, 2, [130, 129], torch.bfloat16)
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_aligned_padding(dtype):
    q, k, bt, lens, mb = make_case(2, 4, 2, [130, 513], dtype, q_slice=True)
    padded = torch.empty_strided(
        k.shape, (P * 144, 144, 1), device=k.device, dtype=dtype
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
    for fw, tw, ppw in itertools.product([1, 2], [1, 2], [0, 1, 2, 4]):
        cfg = IndexScoreConfig(feat_waves=fw, token_waves=tw, pages_per_wave=ppw)
        cap = kernel.work_map_capacity(64, 33, 8, 4, cfg)
        for batch, mb in itertools.product(range(1, 65), [1, 4, 8, 16, 17, 32, 33]):
            assert kernel.work_map_size(batch, mb, 8, 4, cfg) <= cap


@pytest.mark.parametrize("cu", [16, 80, 256, 304])
def test_auto_token_threshold(monkeypatch, cu):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: cu)
    cfg = IndexScoreConfig(cp_world=4, cp_rank=3)
    # Tail pages count as whole CTAs, not half pages in the split estimate.
    assert kernel.resolve_config(cu + 1, 1, cfg, 1, 4).token_waves == 1
    for batch, mb, expected in [
        (1, cu - 1, 4),
        (2, cu // 2, 4),
        (1, cu + 1, 2),
        (2, cu, 2),
        (1, 2 * cu + 1, 1),
    ]:
        resolved = kernel.resolve_config(batch, mb, cfg, 1, 4)
        assert resolved.token_waves == expected
        assert resolved.pages_per_wave == 1
        assert resolved.nt_k == 2
        assert kernel.resolve_config(batch, mb, resolved, 1, 4) == resolved
        assert kernel.work_map_size(batch, mb, 1, 4, cfg) == batch * (
            (mb + kernel.work_chunk(resolved) - 1) // kernel.work_chunk(resolved)
        )
    for S, H in [(2, 4), (4, 4), (8, 4), (1, 17), (0, 0)]:
        assert kernel.resolve_config(1, 1, cfg, S, H).token_waves == 1


@pytest.mark.parametrize(
    "overrides",
    [
        {"token_waves": 1},
        {"token_waves": 2},
        {"token_waves": 4},
        {"pages_per_wave": 1},
        {"pages_per_wave": 2},
        {"q_to_lds": True},
        {"waves_per_eu": 1},
        {"nt_k": 0},
        {"sched": 0},
    ],
)
@pytest.mark.parametrize("mb", [16, 384])
def test_auto_token_preserves_overrides(monkeypatch, overrides, mb):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 256)
    cfg = IndexScoreConfig(**overrides)
    resolved = kernel.resolve_config(1, mb, cfg, 1, 4)
    assert resolved.token_waves == overrides.get("token_waves", 1)
    for key, value in overrides.items():
        assert getattr(resolved, key) == value


# Every knob that changes the emitted code has to change the kernel name too:
# the name is the JIT cache key, so a collision serves the binary built for the
# other config, silently and with no error. Only the default keeps the bare
# name, which is what lets an untouched launch reuse a cache built before the
# knob existed.
@pytest.mark.parametrize(
    "cfg,mark",
    [
        (IndexScoreConfig(pages_per_wave=1), None),
        (IndexScoreConfig(pages_per_wave=1, spread=1), "_sp"),
        (IndexScoreConfig(pages_per_wave=1, shuffled=True), "_shuf"),
        (IndexScoreConfig(pages_per_wave=1, token_waves=2), "_tw2"),
        (IndexScoreConfig(pages_per_wave=1, feat_waves=2), "_fw2"),
        (IndexScoreConfig(pages_per_wave=1, nt_k=0), "_kc0"),
        (IndexScoreConfig(pages_per_wave=1, sched=1), "_sc1"),
        (IndexScoreConfig(pages_per_wave=1, cp_world=4, cp_rank=2), "_cp4r2"),
    ],
)
def test_kernel_name_separates_every_code_changing_knob(cfg, mark):
    base = kernel.kernel_name(4, 1, True, IndexScoreConfig(pages_per_wave=1))
    name = kernel.kernel_name(4, 1, True, cfg)
    if mark is None:
        assert name == base
        return
    assert name != base, f"{cfg} shares the JIT key with the default config"
    assert mark in name


def test_kernel_names_are_pairwise_distinct():
    from dataclasses import fields, replace

    base = IndexScoreConfig(pages_per_wave=1)
    seen = {kernel.kernel_name(4, 1, True, base): "default"}
    # One non-default value per field, so any field the name forgets collides.
    probe = {
        "pages_per_wave": 2,
        "spread": 1,
        "shuffled": True,
        "token_waves": 2,
        "feat_waves": 2,
        "q_to_lds": True,
        "waves_per_eu": 1,
        "nt_k": 0,
        "sched": 1,
        "cp_world": 4,
    }
    assert probe.keys() <= {f.name for f in fields(IndexScoreConfig)}
    for field, value in probe.items():
        cfg = replace(base, **{field: value})
        name = kernel.kernel_name(4, 1, True, cfg)
        assert name not in seen, f"{field}={value} collides with {seen[name]}"
        seen[name] = f"{field}={value}"


# `shuffled` reorders the K bytes; it does not change how much work there is.
# So it shares every geometry decision with the native layout and differs only
# in the cache policy, which the layout inverts: natively each 128 B line is
# requested twice and L1 turns the second into a hit, shuffled each line is
# requested once and the fill buys nothing.
@pytest.mark.parametrize(
    "S,H,batch,max_block",
    [
        (4, 1, 1, 64),
        (4, 1, 2, 64),
        (4, 1, 8, 64),
        (4, 1, 32, 64),
        (4, 1, 1, 1024),
        (4, 1, 8, 1024),
        (4, 1, 32, 4096),
        (4, 1, 128, 8192),
        (8, 4, 2, 64),
        (8, 4, 8, 1024),
        (8, 4, 32, 4096),
        (1, 4, 1, 16),
        (1, 4, 1, 384),
    ],
)
def test_shuffled_shares_dispatch_but_not_cache_policy(
    monkeypatch, S, H, batch, max_block
):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 256)
    native = kernel.resolve_config(batch, max_block, IndexScoreConfig(), S, H)
    shuf = kernel.resolve_config(
        batch, max_block, IndexScoreConfig(shuffled=True), S, H
    )
    assert (shuf.pages_per_wave, shuf.token_waves, shuf.feat_waves, shuf.spread) == (
        native.pages_per_wave,
        native.token_waves,
        native.feat_waves,
        native.spread,
    )
    assert shuf.nt_k == 2
    assert kernel.resolve_config(batch, max_block, shuf, S, H) == shuf


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
def test_served_decode_dispatch(S, H, batch, ppw):
    resolved = kernel.resolve_config(batch, 8192, IndexScoreConfig(), S, H)
    assert (resolved.pages_per_wave, resolved.nt_k, resolved.token_waves) == (
        ppw,
        0,
        1,
    )
    assert kernel.resolve_config(batch, 8192, resolved, S, H) == resolved


# The depth tables are keyed on batch, but depth only pays when there are
# enough pages to fill the machine at that depth. A full batch of *short*
# requests has a large batch and very little work, and the tabled depth then
# strands CUs: bs32 at an 8K context is 2048 pages, 128 CTAs on 256 CUs.
@pytest.mark.parametrize(
    "S,H,batch,max_block,ppw,spread",
    [
        # 8K context (64 blocks): the table wants 2 or more, occupancy caps it.
        (4, 1, 8, 64, 1, 1),  # 512 pages
        (4, 1, 16, 64, 1, 1),  # 1024 pages
        (4, 1, 32, 64, 2, 0),  # 2048 pages: cap 2, table 4
        (4, 1, 64, 64, 4, 0),  # 4096 pages: cap 4 == table
        (4, 1, 128, 64, 4, 0),  # 8192 pages: cap 8, table 4 wins
        (8, 4, 16, 64, 1, 1),  # table 3, cap 1
        (8, 4, 32, 64, 2, 0),  # table 4, cap 2
        # Long context keeps the tabled depth: the cap is far above it.
        (4, 1, 32, 256, 4, 0),
        (4, 1, 32, 4096, 4, 0),
        (8, 4, 32, 4096, 4, 0),
        (4, 1, 1, 4096, 1, 1),
    ],
)
def test_served_depth_capped_by_occupancy(
    monkeypatch, S, H, batch, max_block, ppw, spread
):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 256)
    resolved = kernel.resolve_config(batch, max_block, IndexScoreConfig(), S, H)
    assert (resolved.pages_per_wave, resolved.spread) == (ppw, spread)
    # Whatever it picked, every CU gets a CTA unless there are fewer pages
    # than CUs to begin with.
    ctas = batch * -(-max_block // kernel.work_chunk(resolved))
    assert ctas >= 256 or batch * max_block < 256 * 4
    assert kernel.resolve_config(batch, max_block, resolved, S, H) == resolved


@pytest.mark.parametrize("batch,max_block,tw", [(4, 64, 4), (4, 128, 2), (4, 129, 1)])
def test_q1_token_split_regime_kept(monkeypatch, batch, max_block, tw):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 256)
    resolved = kernel.resolve_config(batch, max_block, IndexScoreConfig(), 1, 4)
    # Small enough for the token split: it keeps its own measured regime.
    # One block past it, the S1 launch joins the one-tile depth table.
    assert resolved.token_waves == tw
    assert resolved.spread == (1 if tw == 1 else 0)


@pytest.mark.parametrize(
    "S,H,cfg",
    [
        (16, 4, IndexScoreConfig()),
        (8, 8, IndexScoreConfig()),
        (4, 1, IndexScoreConfig(cp_world=4, cp_rank=1)),
        (8, 4, IndexScoreConfig(cp_world=4, cp_rank=1)),
        (8, 4, IndexScoreConfig(nt_k=2)),
        (8, 4, IndexScoreConfig(sched=1)),
        (4, 1, IndexScoreConfig(nt_k=2)),
        (4, 1, IndexScoreConfig(nt_k=0)),
        (4, 1, IndexScoreConfig(pages_per_wave=4)),
        (4, 1, IndexScoreConfig(spread=0)),
    ],
)
def test_served_decode_dispatch_scope(monkeypatch, S, H, cfg):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 256)
    resolved = kernel.resolve_config(1, 8192, cfg, S, H)
    # Anything but the untouched served launch keeps the capacity estimate
    # (2 pages per wave at batch 1), the fixed-chunk map and, unless set, the
    # NT policy.
    assert resolved.nt_k == (2 if cfg.nt_k < 0 else cfg.nt_k)
    assert resolved.pages_per_wave == (cfg.pages_per_wave or 2)
    assert resolved.spread == 0


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
def test_spread_dispatch(S, H, batch, max_block, spread):
    resolved = kernel.resolve_config(batch, max_block, IndexScoreConfig(), S, H)
    assert resolved.spread == spread
    assert kernel.resolve_config(batch, max_block, resolved, S, H) == resolved
    rows = kernel.work_map_size(batch, max_block, S, H)
    if spread:  # one page per wave, plus the spare row per request
        assert rows == batch * ((max_block + 3) // 4 + 1)
    assert kernel.work_map_capacity(batch, max_block, S, H) >= rows


@pytest.mark.parametrize(
    "cfg",
    [
        IndexScoreConfig(spread=1, pages_per_wave=2),
        IndexScoreConfig(spread=1, token_waves=2),
        IndexScoreConfig(spread=1, feat_waves=2),
        IndexScoreConfig(spread=2),
    ],
)
def test_spread_rejects_split_geometry(cfg):
    assert not selection_filter(8, 4, cfg, arch="gfx950")


def test_spread_rejects_unpackable_bounds():
    cfg = IndexScoreConfig(spread=1, pages_per_wave=1, cp_world=4, cp_rank=0)
    kernel.work_map_size(1, 0x7FFF, 4, 1, cfg)  # 4 * 0x7FFF * 128 < 2**24
    with pytest.raises(ValueError, match="spread"):
        kernel.work_map_size(1, 0x8000, 4, 1, cfg)
    wide = IndexScoreConfig(spread=1, pages_per_wave=1)
    kernel.work_map_size(kernel.SPREAD_MAX_BATCH, 64, 4, 1, wide)
    with pytest.raises(ValueError, match="spread"):
        kernel.work_map_size(kernel.SPREAD_MAX_BATCH + 1, 64, 4, 1, wide)


def decode_spread(wm):
    """(request, first page, pages, seq_len) per live row, holes checked."""
    wm = wm.cpu().long() & 0xFFFFFFFF
    live = wm[:, 1] != 0
    n = int(live.sum())
    assert live[:n].all() and not live[n:].any(), "holes must trail the work"
    assert not wm[n:].any(), "a hole is (0, 0)"
    w = wm[:n]
    return w[:, 0] >> 16, w[:, 0] & 0xFFFF, w[:, 1] >> 24, w[:, 1] & 0xFFFFFF


@pytest.mark.parametrize("cu", [80, 256, 304])
def test_spread_map_coverage(cu):
    import random

    gen = random.Random(cu)
    for _ in range(300):
        B = gen.choice([1, 2, 3, 4, 8])
        mb = gen.choice([1, 5, 64, 1100, 2049, 8192])
        world, rank = gen.choice([(1, 0), (1, 0), (4, 1), (2, 1)])
        lens = [gen.choice([0, gen.randint(1, mb * world * P)]) for _ in range(B)]
        cfg = IndexScoreConfig(
            spread=1, pages_per_wave=1, token_waves=1, cp_world=world, cp_rank=rank
        )
        rows = B * kernel._grid_chunks(mb, cfg)
        seq = torch.tensor(lens, dtype=torch.int32)
        wm = kernel.make_spread_work_map(seq, rows, cu, world=world, rank=rank)
        b, first, span, length = decode_spread(wm)
        assert ((1 <= span) & (span <= 4)).all()
        assert torch.equal(length, seq.long()[b])
        nblk = [max(0, (L + P - 1) // P - rank + world - 1) // world for L in lens]
        seen = {}
        for r, f, s in zip(b.tolist(), first.tolist(), span.tolist()):
            for page in range(f, f + s):
                seen[(r, page)] = seen.get((r, page), 0) + 1
        want = {(r, page) for r in range(B) for page in range(nblk[r])}
        assert set(seen) == want and set(seen.values()) <= {1}, lens
        # Never more CU rounds than the pages (plus one cut per request) need.
        pages, reqs = sum(nblk), sum(n > 0 for n in nblk)
        rounds = max(1, -(-(pages + 4 * reqs) // (4 * cu)))
        assert b.numel() <= rounds * cu


@pytest.mark.parametrize("ppw", [1, 2, 3, 4])
def test_chunk_map_kernel_matches_reference(ppw):
    """The fused fixed-chunk builder must be the torch one, bit for bit."""
    import random

    gen = random.Random(ppw)
    for _ in range(40):
        B = gen.choice([1, 2, 4, 7, kernel.SPREAD_MAX_BATCH])
        mb = gen.choice([1, 5, 257, 1100, 8192])
        world, rank = gen.choice([(1, 0), (1, 0), (4, 3)])
        cap = mb * world * P
        lens = [min(gen.choice([0, 1, P, cap, gen.randint(1, cap)]), cap)
                for _ in range(B)]  # fmt: skip
        cfg = IndexScoreConfig(
            spread=0, pages_per_wave=ppw, token_waves=1,
            cp_world=world, cp_rank=rank,
        )  # fmt: skip
        chunk = kernel.work_chunk(cfg)
        rows = B * kernel._grid_chunks(mb, cfg)
        seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
        ref = kernel.make_work_map(seq, mb, chunk, world=world, rank=rank)
        got = torch.full((rows, 2), -7, dtype=torch.int32, device="cuda")
        kernel._run_chunk_map(seq, rows, chunk, got, world, rank)
        assert torch.equal(got, ref), (B, mb, ppw, world, rank, lens)


@pytest.mark.parametrize("cu", [80, 256])
def test_spread_map_kernel_matches_reference(cu):
    import random

    gen = random.Random(cu + 1)
    for _ in range(60):
        B = gen.choice([1, 2, 3, 4, 7, kernel.SPREAD_MAX_BATCH])
        mb = gen.choice([1, 4, 5, 1100, 8192])
        world, rank = gen.choice([(1, 0), (1, 0), (4, 3)])
        cap = mb * world * P
        lens = [gen.choice([0, 1, 4 * P, cap, gen.randint(1, cap)]) for _ in range(B)]
        lens = [min(x, cap) for x in lens]
        cfg = IndexScoreConfig(
            spread=1, pages_per_wave=1, token_waves=1, cp_world=world, cp_rank=rank
        )
        rows = B * kernel._grid_chunks(mb, cfg)
        seq = torch.tensor(lens, dtype=torch.int32, device="cuda")
        ref = kernel.make_spread_work_map(seq, rows, cu, world=world, rank=rank)
        got = torch.full((rows, 2), -7, dtype=torch.int32, device="cuda")
        kernel._run_spread_map(seq, rows, cu, got, world, rank)
        assert torch.equal(got, ref), (B, mb, world, rank, lens)


# Page counts either side of a CU round, where the fixed-chunk map spills.
SPREAD_CASES = [
    ("b1_spill", 4, 1, [1028 * P - 5], 1100),
    ("b2_tail", 4, 1, [1024 * P, 1], 1100),
    ("b4_even", 4, 1, [257 * P] * 4, 300),
    ("b4_ragged", 4, 1, [0, 3 * P + 1, 900 * P, 17], 1000),
    ("q8h4_b2", 8, 4, [700 * P + 3, 400 * P], 800),
]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "S,H,lens,mb", [c[1:] for c in SPREAD_CASES], ids=[c[0] for c in SPREAD_CASES]
)
def test_spread_scores(dtype, S, H, lens, mb):
    B = len(lens)
    q, cache, table, seq, _ = make_case(B, S, H, lens, dtype)
    bt = torch.zeros((B, mb), dtype=torch.int32, device="cuda")
    bt[:, : table.shape[1]] = table
    assert kernel.resolve_config(B, mb, IndexScoreConfig(), S, H).spread == 1
    got = score_flydsl(q, cache, bt, seq, S, H, D**-0.5, mb)
    ref = score_oracle(q, cache, bt, seq, S, H, D**-0.5, mb)
    check_scores(got, ref)
    # Same pages, same arithmetic: the fixed-chunk map must agree bit for bit.
    fixed = score_flydsl(
        q, cache, bt, seq, S, H, D**-0.5, mb, pages_per_wave=1, nt_k=0, spread=0
    )
    defined = ~torch.isnan(ref)
    assert torch.equal(got[defined], fixed[defined])


@pytest.mark.parametrize("S,H,B", [(4, 1, 2), (4, 1, 4), (8, 4, 2)])
def test_spread_graph_replay(S, H, B):
    mb = 1100
    q, cache, _, seq, _ = make_case(B, S, H, [1] * B, torch.float8_e4m3fn)
    cache = torch.randn(B * mb, P, D, device="cuda").to(torch.float8_e4m3fn)
    bt = torch.zeros((B, mb), dtype=torch.int32, device="cuda")
    assert kernel.resolve_config(B, mb, IndexScoreConfig(), S, H).spread == 1
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


@pytest.mark.parametrize("cu", [16, 80, 256, 304])
def test_auto_token_capacity_envelope(monkeypatch, cu):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: cu)
    # A maximum-query allocation must also cover Q1 replays, not just Q8.
    cap = kernel.work_map_capacity(64, 33, 8, 4)
    assert cap == 64 * 33
    for batch, mb, S in itertools.product(range(1, 65), [1, 4, 16, 17, 32, 33], [1, 8]):
        assert kernel.work_map_size(batch, mb, S, 4) <= cap
        assert kernel.work_map_capacity(batch, mb, S, 4) <= cap
    explicit = IndexScoreConfig(token_waves=1)
    assert kernel.work_map_capacity(64, 33, 1, 4, explicit) == 64 * 9
    # ...but not past the width where one page per chunk stops packing: the
    # token split cannot fire there either, and widening for it rejected bounds
    # that pack perfectly well at the depth a launch would actually take.
    wide = kernel.work_map_capacity(128, 262144, 4, 1)
    assert wide == 128 * (262144 // 4)
    assert kernel.work_map_size(128, 262144, 4, 1) <= wide


def test_auto_token_explicit_device(monkeypatch):
    from types import SimpleNamespace

    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 304)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=256),
    )
    assert kernel.resolve_config(1, 257, IndexScoreConfig(), 1, 4).token_waves == 4
    assert (
        kernel.resolve_config(
            1, 257, IndexScoreConfig(), 1, 4, device=torch.device("cuda", 0)
        ).token_waves
        == 2
    )


@pytest.mark.parametrize("world,rank", [(1, 0), (4, 0), (4, 3)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("split", [2, 4])
def test_auto_token_legacy_map(world, rank, dtype, split):
    B, S, H, mb = 3, 1, 4, 16
    if split == 2:
        cu = torch.cuda.get_device_properties("cuda").multi_processor_count
        mb = 2 * (cu // B)
    q, k, bt, seq, _ = make_case(B, S, H, [0, 129, mb * world * P], dtype)
    cfg = IndexScoreConfig(cp_world=world, cp_rank=rank)
    assert kernel.resolve_config(B, mb, cfg, S, H).token_waves == split
    # The pre-existing public map helpers allow omitted S/H. Their map remains
    # usable without guessing its contents or copying/synchronizing it on host.
    wm = kernel.build_work_map(seq, mb, cfg=cfg)
    assert wm.shape[0] == kernel.work_map_size(B, mb, cfg=cfg)
    got = score_flydsl(q, k, bt, seq, S, H, D**-0.5, mb, cfg=cfg, work_map=wm)
    check_scores(got, score_oracle(q, k, bt, seq, S, H, D**-0.5, mb, world, rank))
    with pytest.raises(ValueError, match="work_map"):
        score_flydsl(
            q,
            k,
            bt,
            seq,
            S,
            H,
            D**-0.5,
            mb,
            work_map=wm,
            token_waves=4,
            cp_world=world,
            cp_rank=rank,
        )
    assert kernel.work_chunk(IndexScoreConfig(pages_per_wave=2)) == 8
    assert kernel.reduce_lds_bytes(S, H, IndexScoreConfig(pages_per_wave=2)) == 0


def test_cli_default_matrix():
    args = parse_args([])
    cases = select_cases(args)
    assert cases == build_matrix(torch.float8_e4m3fn)
    assert len(cases) == 38
    assert args.impl == list(CANDIDATES)
    assert args.layout == ["contiguous", "feature"]
    assert (
        sum(
            selection_filter(c[2], c[3], CANDIDATES[name], arch="gfx950")
            for c, name, _ in itertools.product(cases, args.impl, args.layout)
        )
        == 652
    )


def test_cli_custom_cross_product(monkeypatch):
    # The shared fp8 alias is arch-dependent; model gfx950 without using a GPU.
    monkeypatch.setitem(dtypes.d_dtypes, "fp8", torch.float8_e4m3fn)
    args = parse_args(["-d", "bf16", "fp8", "-b", "1", "3", "-s", "1,1,128", "4,2,513"])
    cases = select_cases(args)
    assert len(cases) == 8
    assert {(c[1], c[2], c[3], c[4][0], c[5]) for c in cases} == {
        (B, S, H, L, dt)
        for dt, B, (S, H, L) in itertools.product(
            [torch.bfloat16, torch.float8_e4m3fn], [1, 3], [(1, 1, 128), (4, 2, 513)]
        )
    }
    assert all(c[4] == [c[4][0]] * c[1] and c[6] == {} for c in cases)
    assert len({c[0] for c in cases}) == 8
    assert select_cases(parse_args(["-s", "1,1,128"]))[0][1] == 2
    name = cases[0][0]
    assert select_cases(parse_args(["-b", "1", "-s", "1,1,128", "--case", name])) == [
        cases[0]
    ]


def test_cli_regression_filters():
    cases = select_cases(parse_args(["-d", "bf16", "-b", "4"]))
    assert [c[0] for c in cases] == ["ragged_bf16", "ragged_zero_bf16"]
    assert select_cases(parse_args(["--case", "qslice_fp8"])) == [
        c for c in build_matrix(torch.float8_e4m3fn) if c[0] == "qslice_fp8"
    ]


@pytest.mark.parametrize(
    "argv",
    [
        ["-d"],
        ["-b"],
        ["-s"],
        ["--case"],
        ["--case", "unknown"],
        ["-d", "bf16", "--case", "qslice_fp8"],
        ["-b", "3"],
        ["-s", "1,1,128", "-d"],
        ["-s", "1,1,128", "-b"],
    ],
)
def test_cli_empty_cases(argv):
    assert select_cases(parse_args(argv)) == []


@pytest.mark.parametrize("axis", ["--impl", "--layout"])
def test_cli_empty_axes(axis):
    args = parse_args([axis])
    assert list(itertools.product(select_cases(args), args.impl, args.layout)) == []


@pytest.mark.parametrize(
    "argv",
    [
        ["-d", "fp16"],
        ["-d", "fp32"],
        ["-d", "unknown"],
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
        ["--impl", "unknown"],
        ["--layout", "unknown"],
    ],
)
def test_cli_invalid_args(argv):
    with pytest.raises(SystemExit) as exc:
        parse_args(argv)
    assert exc.value.code == 2


def force_arch(monkeypatch, arch):
    """Report `arch` to the kernel while still running on this device.

    CDNA4 kept both the 16x16x16 MFMA and `v_cvt_pk_f32_fp8`, so the entire
    gfx942 code path -- offsets, loop counts, fragment widths, instruction
    selection -- builds and executes correctly on a gfx950 box. What this
    cannot answer is whether a gfx942 binary loads and runs on gfx942 silicon;
    that needs the part. Everything above the ISA is checked here.

    Only gcnArchName is substituted: the CU count has to stay real or
    `_occupancy_depth_cap` would resolve a different config and the comparison
    would no longer be about the arch.
    """
    real = torch.cuda.get_device_properties

    def fake(device):
        p = real(device)
        return SimpleNamespace(
            gcnArchName=arch, multi_processor_count=p.multi_processor_count
        )

    monkeypatch.setattr(torch.cuda, "get_device_properties", fake)


@pytest.mark.parametrize("arch", kernel.SUPPORTED_ARCHS)
@pytest.mark.parametrize("fp8", [False, True])
def test_arch_traits_are_consistent(arch, fp8):
    """The k-axis map is a bijection and the derived counts are self-consistent."""
    tr = kernel.arch_traits(arch, fp8)
    covered = sorted(
        tr.k_offset(ks, g) + v
        for ks in range(tr.ksteps)
        for g in range(kernel.LANE_GROUPS)
        for v in range(tr.lane_k)
    )
    # Every head-dim position reaches exactly one (k-step, lane group, slot).
    # A permutation of the k axis is free because the dot sums over it, but
    # only if it really is a permutation.
    assert covered == list(range(D))
    assert tr.ksteps * tr.lane_k * kernel.LANE_GROUPS == D
    assert tr.lane_k * (1 if fp8 else 2) * tr.k_per_load == kernel.ACCESS_BYTES


@pytest.mark.parametrize("fp8", [False, True])
def test_gfx942_changes_only_the_mfma_shape(fp8):
    """What must move between the two generations, and what must not.

    The invariant half is the point: K's addressing and the shuffled layout
    are built from `k_loads`, `chunk_elems`, `block_k` and `lane_block`, and
    halving the MFMA's k doubles the k-steps one 16 B access covers, so all
    four cancel out. That is what keeps gfx942 a small change and keeps one
    shuffled cache readable by both.
    """
    a = kernel.arch_traits("gfx950", fp8)
    b = kernel.arch_traits("gfx942", fp8)
    for field in ("k_loads", "chunk_elems", "q_loads", "block_k", "lane_block"):
        assert getattr(a, field) == getattr(b, field), field
    assert (b.mfma_k, b.lane_k, b.ksteps) == (
        a.mfma_k // 2,
        a.lane_k // 2,
        a.ksteps * 2,
    )
    assert b.k_per_load == a.k_per_load * 2


@pytest.mark.parametrize("fp8", [False, True])
def test_shuffle_layout_is_arch_independent(fp8):
    """ATOM writes this layout, so it must not depend on who reads it.

    Recomputed from each arch's traits rather than compared against a constant,
    so a future arch that *would* move the layout fails here instead of
    silently breaking a producer that cannot see this file.
    """

    def destination(tr):
        row = torch.arange(P)
        kk = torch.arange(D)
        panel, u = row // 16, row % 16
        i = kk // tr.block_k
        g = (kk % tr.block_k) // tr.lane_block
        v = kk % tr.lane_block
        slot = panel[:, None] * tr.k_loads + i[None, :]
        return (slot * 64 + (16 * g[None, :] + u[:, None])) * tr.chunk_elems + v[
            None, :
        ]

    assert torch.equal(
        destination(kernel.arch_traits("gfx950", fp8)),
        destination(kernel.arch_traits("gfx942", fp8)),
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "cfg",
    [
        IndexScoreConfig(pages_per_wave=1),
        IndexScoreConfig(pages_per_wave=2, feat_waves=2),
        IndexScoreConfig(pages_per_wave=2, token_waves=2),
        IndexScoreConfig(pages_per_wave=2, q_to_lds=True),
        IndexScoreConfig(pages_per_wave=2, shuffled=True),
    ],
)
def test_gfx942_path_matches_oracle(monkeypatch, dtype, cfg):
    """The CDNA3 shapes produce the same scores, on every staging path.

    This is also what proves the 16x16x16 MFMA really ran rather than the
    16x16x32 one under a different name: the gfx942 fragment is four bf16 per
    lane, and four elements fed through a k=32 MFMA cannot come out right. So
    a pass here means the shape, the k-axis map and the widening all agree.

    q_to_lds and the shuffled layout are in the list because they are where
    the two generations could diverge silently: the LDS fill indexes by Q
    access rather than k-step, and the shuffled reader bypasses k_offset
    entirely.
    """
    # S=8 H=4 is two feature tiles, so feat_waves=2 is actually exercised
    # rather than skipped.
    S, H = 8, 4
    assert not selection_filter(S, H, cfg) or kernel.feature_tiles(S, H) > 1
    q, cache, bt, lens, mb = make_case(3, S, H, [0, 129, 2305], dtype)
    ref = score_oracle(q, cache, bt, lens, S, H, D**-0.5, mb)
    k = shuffle_cache(cache) if cfg.shuffled else cache
    force_arch(monkeypatch, "gfx942")
    got = score_flydsl(q, k, bt, lens, S, H, D**-0.5, mb, cfg=cfg)
    ok, reason = compare(got, ref)
    assert ok, reason


def test_arch_separates_the_kernel_name():
    """Two generations of the same config are two binaries, not one."""
    cfg = IndexScoreConfig(pages_per_wave=1)
    names = {kernel.kernel_name(4, 4, False, cfg, a) for a in kernel.SUPPORTED_ARCHS}
    assert len(names) == len(kernel.SUPPORTED_ARCHS)
    # The default arch keeps the name it had before the port, so gfx950's
    # existing JIT cache entries are not orphaned.
    assert kernel.kernel_name(4, 4, False, cfg) == kernel.kernel_name(
        4, 4, False, cfg, "gfx950"
    )
    assert "gfx950" not in kernel.kernel_name(4, 4, False, cfg)


def test_fp8_dtype_comes_from_aiter():
    """The accepted fp8 flavour is the arch's, not a literal in the kernel.

    e4m3fn and e4m3fnuz differ in exponent bias, so a kernel that names one
    and runs on the other is silently off by a factor of two. Pinning this to
    `aiter.dtypes` means the chip decides.
    """
    assert kernel._fp8_dtype() is dtypes.fp8
    # gfx950 is the only arch this kernel accepts, and its fp8 is OCP e4m3fn.
    assert dtypes.fp8 is torch.float8_e4m3fn


def test_reject_fp8_fnuz_cache():
    """The other architecture's fp8 is rejected, not reinterpreted."""
    q, k, bt, lens, mb = make_case(1, 1, 1, [128], torch.bfloat16)
    fnuz = k.to(torch.float8_e4m3fnuz)
    assert not kernel.index_score_supported(q, fnuz, 1, 1, mb, bt)
    with pytest.raises(ValueError, match="cache"):
        score_flydsl(q, fnuz, bt, lens, 1, 1, D**-0.5, mb)
    with pytest.raises(ValueError, match="cache"):
        kernel.shuffle_cache(fnuz)


def test_cli_rejects_fp8_fnuz(monkeypatch):
    monkeypatch.setitem(dtypes.d_dtypes, "fp8", torch.float8_e4m3fnuz)
    with pytest.raises(SystemExit) as exc:
        parse_args(["-d", "fp8"])
    assert exc.value.code == 2


def test_empty_selection(caplog):
    caplog.handler.setLevel(0)
    aiter.logger.addHandler(caplog.handler)
    main(["--case", "not-a-case"])
    main(["--case", "F1_s4096_bf16", "--impl", "fly_fw4"])
    aiter.logger.removeHandler(caplog.handler)
    assert "no score tests executed" in caplog.text
    with pytest.raises(SystemExit) as exc:
        main(["--impl", "unknown"])
    assert exc.value.code == 2


@pytest.mark.parametrize("token_waves", [1, 2, 4])
def test_empty_cp_shard_narrow_table(token_waves):
    q, k, bt, lens, mb = make_case(2, 4, 4, [0, 128], torch.bfloat16)
    cfg = IndexScoreConfig(
        cp_world=4, cp_rank=3, token_waves=token_waves, pages_per_wave=2
    )
    out = score_flydsl(q, k, bt, lens, 4, 4, D**-0.5, mb, cfg=cfg)
    check_scores(out, score_oracle(q, k, bt, lens, 4, 4, D**-0.5, mb, 4, 3))


@pytest.mark.parametrize("S", [4, 8])
@pytest.mark.parametrize(
    "cfg",
    [
        IndexScoreConfig(pages_per_wave=1),
        IndexScoreConfig(pages_per_wave=2),
        IndexScoreConfig(pages_per_wave=4),
        IndexScoreConfig(pages_per_wave=2, token_waves=2),
        IndexScoreConfig(pages_per_wave=2, token_waves=4),
        IndexScoreConfig(pages_per_wave=2, feat_waves=2),
        IndexScoreConfig(pages_per_wave=2, shuffled=True),
        # The token split is newly reachable with the shuffled layout: it used
        # to be excluded from the auto tuner, so no launch resolved to both.
        # Shuffled addressing shifts by tw_slot rather than by the row stride,
        # and that shift is the part a token split moves.
        IndexScoreConfig(pages_per_wave=2, token_waves=2, shuffled=True),
        IndexScoreConfig(pages_per_wave=2, token_waves=4, shuffled=True),
        IndexScoreConfig(pages_per_wave=2, feat_waves=2, shuffled=True),
        IndexScoreConfig(pages_per_wave=2, sched=1),
        IndexScoreConfig(pages_per_wave=2, sched=2),
        IndexScoreConfig(pages_per_wave=2, sched=3),
        IndexScoreConfig(pages_per_wave=2, sched=4),
    ],
)
def test_fp8_panel_pipeline(S, cfg):
    if not selection_filter(S, 4, cfg):
        pytest.skip("feature split needs multiple feature tiles")
    # Negative scores expose accidental zero-initialized panel maxima. The
    # short requests cover empty/fully masked pages while the last request
    # exercises multiple page carries and a partial final panel.
    q, cache, bt, lens, mb = make_case(4, S, 4, [0, 1, 130, 2305], torch.float8_e4m3fn)
    q.copy_(q.abs() + 0.25)
    cache.copy_(-(cache.float().abs() + 0.25))
    ref = score_oracle(q, cache, bt, lens, S, 4, D**-0.5, mb)
    k = shuffle_cache(cache) if cfg.shuffled else cache
    out = kernel.alloc_score(4, S, 4, mb, q.device)
    out.fill_(float("nan"))
    wm = kernel.build_work_map(lens, mb, S, 4, cfg)
    got = score_flydsl(q, k, bt, lens, S, 4, D**-0.5, mb, out=out, cfg=cfg, work_map=wm)
    assert got.data_ptr() == out.data_ptr()
    ok, reason = compare(got, ref)
    assert ok, reason
    assert bool((got[torch.isfinite(ref)] < 0).all())


def test_arch_rejected(monkeypatch):
    """An arch with neither MFMA generation is refused before anything else."""
    from types import SimpleNamespace

    q, k, bt, lens, mb = make_case(1, 1, 1, [128], torch.bfloat16)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx90a"),
    )
    assert not kernel.index_score_supported(q, k, 1, 1, mb, bt)
    with pytest.raises(ValueError, match="requires one of"):
        score_flydsl(q, k, bt, lens, 1, 1, D**-0.5, mb)


@pytest.mark.parametrize("operand", ["q", "out", "bt", "map"])
def test_address_span_rejected_without_allocation(monkeypatch, operand):
    from types import SimpleNamespace

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
    k = meta((1, P, D), (P * D, D, 1), torch.bfloat16)
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
    assert not kernel.index_score_supported(q, k, 1, 1, 1, bt, out=out, work_map=wm)
    with pytest.raises(ValueError, match="span"):
        kernel._validate_metadata(q, k, 1, 1, 1, bt, out=out, work_map=wm)


def test_strided_span_metadata():
    # No GPU storage needed: count true strided addresses, not logical numel.
    x = torch.empty_strided((2, 128), (1 << 30, 1), device="meta")
    assert x.numel() == 256
    assert kernel._span(x) == (1 << 30) + 128


def test_noncurrent_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible gfx950 GPUs")
    with torch.cuda.device(1):
        q, k, bt, lens, mb = make_case(1, 1, 1, [128], torch.bfloat16)
        # Device 1 must not use device 0's compiled module handle.
    with torch.cuda.device(0):
        q0, k0, bt0, lens0, mb0 = make_case(1, 1, 1, [128], torch.bfloat16)
        score_flydsl(q0, k0, bt0, lens0, 1, 1, D**-0.5, mb0)
        out = score_flydsl(q, k, bt, lens, 1, 1, D**-0.5, mb)
        assert torch.cuda.current_device() == 0
    check_scores(out, score_oracle(q, k, bt, lens, 1, 1, D**-0.5, mb))


@pytest.mark.parametrize(
    "cfg",
    [
        IndexScoreConfig(feat_waves=0),
        IndexScoreConfig(token_waves=3),
        IndexScoreConfig(pages_per_wave=-1),
        IndexScoreConfig(cp_world=0),
        IndexScoreConfig(nt_k=9),
    ],
)
def test_build_map_invalid_config(cfg):
    lens = torch.ones(1, device="cuda", dtype=torch.int32)
    with pytest.raises(ValueError):
        kernel.build_work_map(lens, 1, 1, 1, cfg)


def test_shuffled_padding_rejected():
    q, k, bt, lens, mb = make_case(1, 1, 1, [128], torch.bfloat16)
    padded = torch.empty_strided(
        k.shape, (32768, 256, 1), device=k.device, dtype=k.dtype
    )
    padded.copy_(shuffle_cache(k))
    with pytest.raises(ValueError, match="shuffled"):
        score_flydsl(q, padded, bt, lens, 1, 1, D**-0.5, mb, shuffled=True)


@pytest.mark.parametrize("scale", [1e-50, 1e39])
def test_scale_fp32_range(scale):
    q, k, bt, lens, mb = make_case(1, 2, 1, [129], torch.bfloat16)
    with pytest.raises(ValueError, match="sm_scale"):
        score_flydsl(q, k, bt, lens, 2, 1, scale, mb)


def test_exact_size_uses_explicit_device(monkeypatch):
    from types import SimpleNamespace

    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 304)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=256),
    )
    # S16 H4 takes the capacity estimate, which is what reads the CU count.
    assert kernel.work_map_size(16, 1024, 16, 4, device=torch.device("cuda", 0)) == 1024


def test_auto_bounds_resolved_before_packing(monkeypatch):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 304)
    assert kernel.work_map_size(1, 262145, 1, 1) == 16385


def test_lds_uses_explicit_arch(monkeypatch):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_gfx", lambda: "gfx942")
    cfg = IndexScoreConfig(q_to_lds=True, token_waves=2, pages_per_wave=1, sched=0)
    assert kernel.selection_filter(64, 4, cfg, arch="gfx950")


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


@benchmark()
def test_score(case, B, S, H, lens, dtype, layout, impls, kwargs):
    q, cache, bt, seq, mb = make_case(B, S, H, lens, dtype, **kwargs)
    ref = score_oracle(q, cache, bt, seq, S, H, D**-0.5, mb)
    out = (
        torch.empty((H, B * S, mb), device=q.device, dtype=torch.float32)
        if layout == "contiguous"
        else torch.empty((mb, B * S, H), device=q.device, dtype=torch.float32).permute(
            2, 1, 0
        )
    )
    ret = {"gfx": get_gfx(), "timing": "repeated-buffer (warm effective TB/s)"}
    candidates = {
        name: CANDIDATES[name]
        for name in impls
        if selection_filter(S, H, CANDIDATES[name])
    }
    for name, cfg in candidates.items():
        k = shuffle_cache(cache) if cfg.shuffled else cache
        capacity = kernel.work_map_capacity(B, mb, S, H, cfg)
        buf = torch.empty((capacity, 2), device=q.device, dtype=torch.int32)
        wm = kernel.build_work_map(seq, mb, S, H, cfg, out=buf)

        def run(k=k, cfg=cfg, wm=wm):
            return score_flydsl(
                q, k, bt, seq, S, H, D**-0.5, mb, out=out, cfg=cfg, work_map=wm
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


# The standardized benchmark takes sweep arguments, not pytest fixtures.
test_score.__test__ = False


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
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        choices=[torch.bfloat16, torch.float8_e4m3fn],
        default=[torch.bfloat16, torch.float8_e4m3fn],
        help="cache dtype list: bf16 fp8 (FP8 E4M3FN only); queries remain BF16",
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
        "--impl", nargs="*", choices=list(CANDIDATES), default=list(CANDIDATES)
    )
    ap.add_argument(
        "--case",
        nargs="*",
        default=None,
        help="case names from build_matrix, or custom_B{B}_S{S}_H{H}_L{L}_{bf16|fp8}",
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
        cases = [
            c
            for c in build_matrix(torch.float8_e4m3fn)
            if c[5] in args.dtype and (args.batch is None or c[1] in args.batch)
        ]
    else:
        cases = [
            (
                f"custom_B{B}_S{S}_H{H}_L{L}_{'bf16' if dt == torch.bfloat16 else 'fp8'}",
                B,
                S,
                H,
                [L] * B,
                dt,
                {},
            )
            for dt, B, (S, H, L) in itertools.product(
                args.dtype, args.batch if args.batch is not None else [2], args.mnk
            )
        ]
    if args.case is not None:
        cases = [c for c in cases if c[0] in args.case]
    return cases


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available() or get_gfx() not in ["gfx950"]:
        aiter.logger.warning("index score requires gfx950; skipping sweep")
        return 0
    rows, skipped = [], 0
    for (name, B, S, H, lens, dt, kw), layout in itertools.product(
        select_cases(args), args.layout
    ):
        legal = [n for n in args.impl if selection_filter(S, H, CANDIDATES[n])]
        skipped += len(args.impl) - len(legal)
        if not legal:
            continue
        rows.append(test_score(name, B, S, H, lens, dt, layout, legal, kw))
    if not rows:
        aiter.logger.warning(
            "Empty selection or all candidates skipped; no score tests executed"
        )
        return 0
    aiter.logger.info(
        "index score summary (markdown):\n%s",
        pd.DataFrame(rows).to_markdown(index=False),
    )
    aiter.logger.info(
        "%d shape/layout rows, %d candidate checks, %d unsupported candidates skipped",
        len(rows),
        sum(sum(k.endswith(" err") for k in r) for r in rows),
        skipped,
    )
    return 0


_GPU_TESTS = (
    test_reject_cache_dtype,
    test_reject_misaligned_cache,
    test_large_cache,
    test_packed_request,
    test_graph_buffers,
    test_invalid_metadata,
    test_aligned_padding,
    test_empty_cp_shard_narrow_table,
    test_fp8_panel_pipeline,
    test_arch_rejected,
    test_shuffled_padding_rejected,
    test_scale_fp32_range,
    test_noncurrent_device,
    test_build_map_invalid_config,
    test_narrow_block_table_rejected,
    test_reject_fp8_fnuz_cache,
    # Asserts which fp8 flavour this chip speaks, so it is arch-dependent even
    # though it allocates nothing.
    test_fp8_dtype_comes_from_aiter,
    # Builds and runs the CDNA3 shapes on whatever part is present; see
    # force_arch for what that does and does not establish.
    test_gfx942_path_matches_oracle,
    # These build their inputs on the device even though what they assert is a
    # host-side property (map equality, spread coverage), so they need the gate
    # just as much as the tests that launch the scorer.
    test_chunk_map_kernel_matches_reference,
    test_spread_map_kernel_matches_reference,
    test_spread_scores,
    test_spread_graph_replay,
    test_auto_token_legacy_map,
)


@pytest.fixture(autouse=True)
def _gpu_test_gate(request):
    gpu_names = {test.__name__ for test in _GPU_TESTS} | {"test_empty_selection"}
    if request.node.originalname in gpu_names and (
        not torch.cuda.is_available() or get_gfx() != "gfx950"
    ):
        pytest.skip("GPU execution requires gfx950")


if __name__ == "__main__":
    sys.exit(main())
