# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness matrix for the MiniMax-M3 prefill index-score kernel.

The judge is aiter's own `pa_sparse_block_score_prefill`, which computes the
same scores without the FlyDSL kernel's length limit. Comparing against it
rather than against a torch oracle is deliberate here: the two implementations
share no code, and the operator's numerics are a contract (Q is rounded down to
the cache's fp8, matching `tl.dot(q.to(k.dtype), k)`), so the comparison is
exact rather than tolerance-based.

Two properties of the contract shape every check below:

  * `out` may be left untouched past a query's causal window. The kernel is
    only required to write the slots aiter marks finite, so the output buffer
    is sentinel-filled and compared only there. `alloc_score` hands back
    `torch.empty`, so a raw buffer diff would be comparing uninitialised
    memory -- which has produced false mismatches before.
  * every config arm must agree bit for bit. These are the same arithmetic in
    different geometries, not different approximations of it.

Run:
    python op_tests/test_flydsl_minimax_m3_index_score_prefill.py
"""

import sys

import pytest
import torch

from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.kernels.minimax_m3_index_score import shuffle_cache
from aiter.ops.flydsl.kernels.minimax_m3_index_score_prefill import (
    PrefillScoreConfig,
    alloc_score,
    resolve_config,
    score_prefill_flydsl,
)
from aiter.ops.msa_block_select import pa_sparse_block_score_prefill

LOG2E = 1.4426950409
PAGE = 128  # operator contract, not a kernel constant
HEAD_DIM = 128
# A value no score can take, so "did the kernel write here" is unambiguous.
SENTINEL = -12345.678

pytestmark = pytest.mark.skipif(
    get_gfx() != "gfx950", reason="prefill index scorer is built for gfx950"
)


def make_case(batch, q_len, ctx_len, seed=0):
    """A uniform ragged batch: every request has q_len new tokens on ctx_len."""
    torch.manual_seed(seed)
    dev = "cuda"
    seq_lens = torch.full((batch,), ctx_len, dtype=torch.int32, device=dev)
    prefix = torch.full((batch,), ctx_len - q_len, dtype=torch.int32, device=dev)
    cu = torch.arange(0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=dev)
    max_block = -(-ctx_len // PAGE)
    npages = batch * max_block
    q8 = (torch.randn(batch * q_len, 1, HEAD_DIM, device=dev) / 4).to(dtypes.fp8)
    cache = (torch.randn(npages, PAGE, HEAD_DIM, device=dev) / 4).to(dtypes.fp8)
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


def reference(c):
    """aiter's scores, and the mask of slots it considers live."""
    ref = torch.full((1, c["q8"].shape[0], c["mb"]), -float("inf"), device="cuda")
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


def run(c, cache, q=None, **cfg_kwargs):
    out = alloc_score(c["q8"].shape[0], 1, c["mb"], "cuda")
    out.fill_(SENTINEL)
    score_prefill_flydsl(
        c["q_bf16"] if q is None else q,
        cache,
        c["bt"],
        c["cu"],
        c["seq"],
        c["prefix"],
        c["max_q"],
        c["max_seq"],
        1.0 / LOG2E,
        out=out,
        cfg=PrefillScoreConfig(**cfg_kwargs),
    )
    return out.reshape(1, c["q8"].shape[0], c["mb"])


def assert_matches(got, ref, live, what):
    unwritten = int((live & (got == SENTINEL)).sum())
    assert unwritten == 0, f"{what}: {unwritten} live slots left unwritten"
    diff = (got[live] - ref[live]).abs().max().item()
    assert diff == 0.0, f"{what}: max abs diff {diff} against aiter"


# (batch, q_len, ctx_len). Covers the short/grid-starved end, the ragged and
# non-power-of-two lengths, and contexts past the 32 MB point where the score
# tensor stops fitting in L2 (which is what sets the store merge width).
SHAPES = [
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

# Every arm is the same arithmetic in a different geometry.
CONFIGS = [
    ("auto_shuffled", {"shuffled": True}),
    ("auto_rowmajor", {}),
    ("rowmajor_klds", {"k_lds": 1}),
    ("rowmajor_dma", {"k_lds": 1, "dma": 1}),
    ("register_k", {"shuffled": True, "k_lds": 0}),
    ("tile_q128", {"shuffled": True, "tile_q": 128, "k_lds": 1}),
    ("ppw1", {"shuffled": True, "pages_per_wave": 1}),
    ("ppw4", {"shuffled": True, "pages_per_wave": 4}),
    ("no_m32", {"shuffled": True, "m32": 0}),
]


@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"b{s[0]}_q{s[1]}_ctx{s[2]}")
@pytest.mark.parametrize("name,kw", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_matches_aiter(shape, name, kw):
    c = make_case(*shape)
    ref, live = reference(c)
    cache = shuffle_cache(c["cache"]) if kw.get("shuffled") else c["cache"]
    assert_matches(run(c, cache, **kw), ref, live, f"{name} {shape}")


def test_shuffled_and_rowmajor_agree():
    """The two cache layouts are a permutation of each other, nothing more."""
    c = make_case(1, 1024, 65536)
    ref, live = reference(c)
    a = run(c, shuffle_cache(c["cache"]), shuffled=True)
    b = run(c, c["cache"])
    assert_matches(a, ref, live, "shuffled")
    assert_matches(b, ref, live, "row major")
    assert torch.equal(a[live], b[live])


@pytest.mark.parametrize("shuffled", [True, False])
def test_fp8_q_is_exact(shuffled):
    """An fp8 Q must give what a bf16 Q holding the same values gives.

    The kernel rounds Q down to the cache's fp8 anyway, so handing it the
    already-rounded values has to be a no-op -- it skips a conversion, it does
    not change one.
    """
    c = make_case(2, 1024, 32768)
    ref, live = reference(c)
    cache = shuffle_cache(c["cache"]) if shuffled else c["cache"]
    bf16 = run(c, cache, shuffled=shuffled)
    fp8 = run(c, cache, q=c["q8"], shuffled=shuffled)
    assert_matches(fp8, ref, live, "fp8 Q")
    assert torch.equal(bf16[live], fp8[live])


def test_strided_cache_is_handled_on_every_entry_form():
    """A non-contiguous page must be correct however the config was supplied.

    The LDS staging path reads a page as a flat run of 16 B chunks, which only
    describes the right bytes when the page is contiguous; the guard that
    demotes it once carried a `cfg is not None` term and so skipped the default
    call -- scoring silently wrong (0.209 max abs error) on exactly the entry
    everyone uses.
    """
    c = make_case(1, 1024, 16384)
    ref, live = reference(c)
    wide = torch.zeros(
        c["cache"].shape[0], PAGE, 2 * HEAD_DIM, dtype=c["cache"].dtype, device="cuda"
    )
    wide[..., :HEAD_DIM] = c["cache"]
    strided = wide[..., :HEAD_DIM]
    assert strided.stride(1) == 2 * HEAD_DIM and not strided.is_contiguous()

    out = alloc_score(c["q8"].shape[0], 1, c["mb"], "cuda")
    out.fill_(SENTINEL)
    score_prefill_flydsl(
        c["q_bf16"],
        strided,
        c["bt"],
        c["cu"],
        c["seq"],
        c["prefix"],
        c["max_q"],
        c["max_seq"],
        1.0 / LOG2E,
        out=out,
    )  # cfg omitted entirely: the form that used to bypass the guard
    assert_matches(out.reshape(ref.shape), ref, live, "strided, cfg=None")

    # Auto demotes; explicit off is already off. Both must be correct.
    for kw in ({}, {"k_lds": 0}):
        assert_matches(run(c, strided, **kw), ref, live, f"strided {kw}")

    # Explicit k_lds=1 asks for something this cache cannot satisfy. Refusing
    # is the contract: quietly building a different kernel would contradict
    # "explicit values are kept" and show up only as a performance mystery.
    with pytest.raises(ValueError, match="contiguous page"):
        run(c, strided, k_lds=1)
    # ...and the same request on a contiguous page still works.
    assert_matches(run(c, c["cache"], k_lds=1), ref, live, "contiguous k_lds=1")


@pytest.mark.parametrize("shape", [(1, 128, 384), (1, 256, 256), (2, 128, 512)])
def test_stages_on_padded_chunk_axis(shape):
    """`chunks` is rounded up for the XCD swizzle, so a CTA can draw an empty
    chunk whose first block is already past max_block. The stages>0 prologue
    prefetches before any emptiness guard, and an unclamped block id becomes
    the base of a buffer descriptor."""
    c = make_case(*shape)
    ref, live = reference(c)
    cs = shuffle_cache(c["cache"])
    for stages in (0, 2):
        assert_matches(
            run(c, cs, shuffled=True, stages=stages), ref, live, f"stages={stages}"
        )


def test_cdna4_only_knobs_are_arch_gated():
    """k128 and m32 lower to `rocdl.cdna4.MFMA_Scale`, which gfx942 lacks, and
    this entry accepts gfx942. Auto must not select them there, and an explicit
    request must be refused rather than built."""
    from aiter.ops.flydsl.kernels.minimax_m3_index_score_prefill import (
        build_prefill_score,
    )

    dev = torch.device("cuda")
    on = resolve_config(
        4096,
        1,
        1,
        1024,
        cfg=PrefillScoreConfig(shuffled=True),
        device=dev,
        arch="gfx950",
    )
    assert (on.k128, on.m32) == (1, 1) and on.k_lds == 1

    off = resolve_config(
        4096,
        1,
        1,
        1024,
        cfg=PrefillScoreConfig(shuffled=True),
        device=dev,
        arch="gfx942",
    )
    assert (off.k128, off.m32, off.k_lds) == (0, 0, 0)
    # The wide CTA rides on k_lds; taking it without one gives a wave every
    # feature tile and spills.
    assert off.waves == 4 and off.tile_q == 128

    with pytest.raises(ValueError, match="MFMA_Scale"):
        build_prefill_score(
            True,
            PrefillScoreConfig(shuffled=True, k128=1, k_lds=1, fp8_mfma=1),
            "gfx942",
        )


def test_rejects_malformed_tensors():
    """Everything below reaches the kernel as a bare pointer, so a wrong dtype
    is a silent reinterpretation rather than an error."""
    c = make_case(1, 256, 2048)
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

    for idx, bad in (
        (4, c["seq"].long()),  # int64 seq_lens
        (5, c["prefix"].long()),  # int64 prefix_lens
        (3, c["cu"].long()),  # int64 cu_seqlens_q
        (2, c["bt"].long()),  # int64 block table
        (2, c["bt"][0]),  # 1-D block table
    ):
        with pytest.raises(ValueError):
            score_prefill_flydsl(*swap(idx, bad))

    with pytest.raises(ValueError):  # out sized for a smaller max_block
        score_prefill_flydsl(
            *base, out=alloc_score(c["q8"].shape[0], 1, c["mb"] // 2, "cuda")
        )


def test_shuffled_cache_must_be_packed():
    """A shuffled page is read as a flat run of 16 B units, so a strided view
    describes the wrong bytes. The strided guard used to exclude `shuffled`,
    which left this accepted and scoring 1.83 off instead of raising."""
    c = make_case(1, 256, 2048)
    shuf = shuffle_cache(c["cache"])
    wide = torch.zeros(
        shuf.shape[0], PAGE, 2 * HEAD_DIM, dtype=shuf.dtype, device="cuda"
    )
    wide[..., :HEAD_DIM] = shuf
    strided = wide[..., :HEAD_DIM]
    assert strided.stride(1) == 2 * HEAD_DIM

    with pytest.raises(ValueError, match="packed within each page"):
        run(c, strided, shuffled=True)
    # The packed cache is still accepted and still right.
    ref, live = reference(c)
    assert_matches(run(c, shuf, shuffled=True), ref, live, "packed shuffled")


def test_bf16_cache_path():
    """A bf16 cache takes a different arithmetic path, and nothing else here
    builds one -- `make_case` is fp8 only, so `score_page`'s non-k128 tail and
    `k_operand`'s `convert_k` arm had no coverage at all. Judged against torch
    rather than aiter, whose prefill scorer takes an fp8 cache."""
    torch.manual_seed(0)
    dev = "cuda"
    batch, q_len, ctx = 1, 256, 2048
    mb = ctx // PAGE
    seq = torch.full((batch,), ctx, dtype=torch.int32, device=dev)
    prefix = torch.full((batch,), ctx - q_len, dtype=torch.int32, device=dev)
    cu = torch.arange(0, (batch + 1) * q_len, q_len, dtype=torch.int32, device=dev)
    q = (torch.randn(batch * q_len, 1, HEAD_DIM, device=dev) / 4).bfloat16()
    cache = (torch.randn(batch * mb, PAGE, HEAD_DIM, device=dev) / 4).bfloat16()
    bt = torch.arange(batch * mb, device=dev, dtype=torch.int32).view(batch, mb)
    bt = bt.contiguous()

    out = alloc_score(batch * q_len, 1, mb, dev)
    out.fill_(SENTINEL)
    score_prefill_flydsl(
        q, cache, bt, cu, seq, prefix, q_len, ctx, 1.0 / LOG2E, out=out
    )
    got = out.reshape(1, batch * q_len, mb).float()

    pos = torch.arange(PAGE, device=dev)
    unwritten = torch.tensor(SENTINEL, dtype=torch.float32).item()
    for row in (0, 1, q_len // 2, q_len - 1):
        cut = ctx - q_len + row + 1
        for blk in (0, 1, mb // 2, mb - 1):
            k = cache[int(bt[0, blk])].float()
            z = k @ q[row, 0].float()
            z = z.masked_fill(blk * PAGE + pos >= cut, -float("inf"))
            ref, mine = z.max().item(), got[0, row, blk].item()
            if ref == -float("inf"):
                # SENTINEL is a python double; the buffer holds its fp32
                # rounding, so compare against what was actually stored.
                # Past the causal window the kernel may leave the slot alone;
                # see this module's docstring. Either answer is in contract.
                assert mine in (
                    unwritten,
                    -float("inf"),
                ), f"row {row} blk {blk} is fully masked but holds {mine}"
                continue
            assert mine != unwritten, f"row {row} blk {blk} left unwritten"
            assert abs(ref - mine) <= 3e-2 * max(
                1.0, abs(ref)
            ), f"row {row} blk {blk}: got {mine}, torch {ref}"


def test_resolve_config_is_idempotent():
    dev = torch.device("cuda")
    for kw in ({}, {"shuffled": True}, {"shuffled": True, "tile_q": 128}):
        once = resolve_config(
            4096, 1, 1, 1024, cfg=PrefillScoreConfig(**kw), device=dev
        )
        assert resolve_config(4096, 1, 1, 1024, cfg=once, device=dev) == once


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
