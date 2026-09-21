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
    exact = kernel.work_map_size(64, 16, 1, 1, cfg)
    smaller = kernel.work_map_size(63, 16, 1, 1, cfg)
    assert smaller > exact  # reproduces the non-monotonic sizing cliff
    assert kernel.work_map_capacity(64, 16, 1, 1, cfg) >= smaller


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
@pytest.mark.parametrize("S,H", [(1, 1), (4, 2), (8, 4)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("layout", ["contiguous", "feature"])
@pytest.mark.parametrize("map_in_graph", [False, True])
def test_graph_buffers(world, rank, S, H, dtype, layout, map_in_graph):
    B, mb = 3, 9  # Stable envelope larger than actual lengths, including CP.
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


def test_arch_rejected(monkeypatch):
    from types import SimpleNamespace

    q, k, bt, lens, mb = make_case(1, 1, 1, [128], torch.bfloat16)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(gcnArchName="gfx942"),
    )
    assert not kernel.index_score_supported(q, k, 1, 1, mb, bt)
    with pytest.raises(ValueError, match="gfx950"):
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
    assert kernel.work_map_size(16, 1024, 1, 4, device=torch.device("cuda", 0)) == 1024


def test_auto_bounds_resolved_before_packing(monkeypatch):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_cu_num", lambda: 304)
    assert kernel.work_map_size(1, 262145, 1, 1) == 16385


def test_lds_uses_explicit_arch(monkeypatch):
    from aiter.jit.utils import chip_info

    monkeypatch.setattr(chip_info, "get_gfx", lambda: "gfx942")
    cfg = IndexScoreConfig(q_to_lds=True, token_waves=2, pages_per_wave=1, sched=0)
    assert kernel.selection_filter(64, 4, cfg, arch="gfx950")


def test_floor_slabs():
    from op_tests.bench_m3_index_score_flydsl import check_floor

    check_floor()


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
    test_arch_rejected,
    test_shuffled_padding_rejected,
    test_scale_fp32_range,
    test_floor_slabs,
    test_noncurrent_device,
    test_build_map_invalid_config,
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
