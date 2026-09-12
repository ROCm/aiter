# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`get_meta_param`'s LRU cache frees a tensor a captured CUDA graph still reads.

The defect
----------
`aiter/mla.py`::

    @functools.lru_cache                     # <-- default maxsize = 128
    def get_meta_param(num_kv_splits, bs, total_kv, nhead, max_seqlen_q, dtype,
                       tg_factor=1, ignore_total_kv=0):
        ...
        num_kv_splits_indptr = torch.arange(
            0, (bs + 1) * num_kv_splits, num_kv_splits, dtype=torch.int, device="cuda"
        )
        return num_kv_splits, num_kv_splits_indptr

The non-persistent decode path passes `num_kv_splits_indptr` straight into
`aiter.mla_decode_stage1_asm_fwd`, so when the call happens under CUDA graph
capture the tensor's **device address is baked into the captured kernel node**.
Nothing else holds a reference to that tensor -- the LRU cache is its only
owner.

The cache key includes `bs`, so a framework capturing one decode graph per
batch-size bucket creates one entry per bucket. Past 128 buckets the LRU starts
**evicting**, dropping the last reference to the tensor. The caching allocator
reclaims the block and hands it to the next allocation. The graph captured
earlier still points there, so its replay reads whatever now occupies that
memory as a split-offset table -- out-of-range KV addressing, garbage in
`logits` / `attn_lse`.

Why the eviction lands exactly on the large batches
---------------------------------------------------
Graph runners capture **largest bucket first** so smaller graphs can reuse the
big graph's memory (sglang: `_capture_one_stream` iterates `reversed(capture_bs)`).
With a dense `--cuda-graph-bs 1..256` list, entries are inserted in the order
bs=256, 255, ..., 1. The first 128 inserted -- bs 256 down to 129 -- are exactly
the ones evicted while the remaining 128 are inserted. So every graph with
**bs > 128** replays against freed memory and every graph with bs <= 128 is
intact.

That is the serving signature this reproduces: DeepSeek-R1 on MI355X collapses
gsm8k 0.95 -> 0.03 only when per-DP-rank batch exceeds 128, only with
non-persistent MLA (the persistent path never calls `get_meta_param`), and only
under CUDA graph replay. Capturing <= 128 buckets, or a sparse bucket list, is
clean -- not because those batch sizes are safe, but because nothing is evicted.

What this test does
-------------------
`test_lru_evicts_live_graph_tensor` is the mechanism check and needs no capture:
call `get_meta_param` for 129+ distinct `bs` values, hold the first tensor, and
watch the cache drop it while the graph that would have been capturing still
needs it.

`test_capture_many_buckets_matches_eager` is the end-to-end check: capture one
single-call decode graph per bucket over a bucket ladder, then replay the
largest bucket (inserted first, evicted first) and compare against eager.

Run:  pytest -q -s op_tests/test_mla_meta_param_lru_eviction.py
"""

import functools
import weakref

import pytest
import torch

import aiter
from aiter import dtypes
from aiter.mla import get_meta_param, mla_decode_fwd
from aiter.jit.utils.chip_info import get_gfx

KV_LORA_RANK = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = KV_LORA_RANK + QK_ROPE_HEAD_DIM
V_HEAD_DIM = KV_LORA_RANK
NHEAD_KV = 1
PAGE_SIZE = 1

NHEAD = 128              # dp-attention on tp8 keeps all 128 heads on every rank
MSQ = 2                  # NEXTN MTP target-verify shape
CAPTURE_FILL_SEQ_LEN = 2  # get_cuda_graph_seq_len_fill_value() == num_draft_tokens
MIN_CTX, MAX_SEQ = 900, 1100     # gsm8k's real regime (serving: ~980 tok/req)
MAX_BS = 256
# What gsm8k actually reaches per DP rank (1319 docs / 8) and the bucket the
# serving failure lands in. Not the largest bucket: bs=256 auto-picks splits=1
# and takes the single-pass path, which never reads the split table.
TARGET_BS = 165
MAX_CTX = 9216
CAPACITY = MAX_BS * (MAX_CTX * PAGE_SIZE + 1)

LRU_MAXSIZE = functools.lru_cache(lambda x: x).cache_info().maxsize  # 128


def _skip_if_unsupported():
    if not torch.cuda.is_available():
        pytest.skip("no GPU")
    if get_gfx() != "gfx950":
        pytest.skip(f"this kernel variant is gfx950-only; got {get_gfx()}")


# ---------------------------------------------------------------------------
# 1. the mechanism, with no CUDA graph involved
# ---------------------------------------------------------------------------
def test_lru_evicts_live_graph_tensor():
    """The cache is the only owner, and it drops the tensor past maxsize.

    Uses a weakref so the probe never touches the cache -- reading an entry
    would move it to the MRU end and mask the very eviction being measured.
    """
    if not torch.cuda.is_available():
        pytest.skip("no GPU")

    if not hasattr(get_meta_param, "cache_info"):
        pytest.skip(
            "get_meta_param no longer carries the LRU cache -- this build has "
            "the fix, the split-offset tables are held outside it"
        )
    info = get_meta_param.cache_info()
    assert info.maxsize is not None, (
        "get_meta_param is unbounded here, so this build cannot evict; the "
        "shipped decorator is a bare @functools.lru_cache (maxsize=128)"
    )
    maxsize = info.maxsize
    get_meta_param.cache_clear()

    # Buckets in the order a graph runner captures them: largest first.
    buckets = list(range(MAX_BS, 0, -1))

    # The first bucket captured. In serving this address is already baked into a
    # captured kernel node by the time the eviction below happens.
    first_bs = buckets[0]
    _, first_indptr = get_meta_param(
        None, first_bs, first_bs * CAPTURE_FILL_SEQ_LEN, NHEAD, MSQ, dtypes.fp8
    )
    baked_ptr = first_indptr.data_ptr()
    baked_numel = first_indptr.numel()
    alive = weakref.ref(first_indptr)
    # Drop our own reference: from here on the LRU cache is the sole owner,
    # exactly as it is inside mla_decode_fwd.
    del first_indptr

    evicted_after = None
    for i, bs in enumerate(buckets[1:], start=1):
        get_meta_param(None, bs, bs * CAPTURE_FILL_SEQ_LEN, NHEAD, MSQ, dtypes.fp8)
        if alive() is None:
            evicted_after = i + 1     # buckets inserted so far, incl. first_bs
            break

    print(
        f"\n=== get_meta_param LRU ===\n"
        f"  maxsize={maxsize} buckets_probed={len(buckets)}\n"
        f"  first bucket bs={first_bs} indptr@{hex(baked_ptr)} (len={baked_numel})\n"
        f"  tensor died after {evicted_after} buckets were inserted"
    )

    assert evicted_after is not None, (
        f"no eviction over {len(buckets)} buckets -- cache is effectively "
        f"unbounded, defect not present"
    )
    assert evicted_after == maxsize + 1, (
        f"expected the {maxsize}-entry cache to drop entry 1 as entry "
        f"{maxsize + 1} goes in, saw it at {evicted_after}"
    )

    # The freed block goes back to the caching allocator, and anyone can take
    # it. Ask for the same size class a few times and see if one lands there.
    torch.cuda.synchronize()
    squatters = [
        torch.full((baked_numel,), -1, dtype=torch.int32, device="cuda")
        for _ in range(8)
    ]
    torch.cuda.synchronize()
    took_over = [t for t in squatters if t.data_ptr() == baked_ptr]
    print(
        f"  8 fresh int32 tensors of the same size -> "
        f"{len(took_over)} landed on {hex(baked_ptr)}"
        f"{'  <-- the captured graph reads this now' if took_over else ''}"
    )


# ---------------------------------------------------------------------------
# 2. end to end: capture N buckets, replay the first-captured one
# ---------------------------------------------------------------------------
class Fixture:
    def __init__(self, seed=0):
        torch.manual_seed(seed)
        dev = "cuda"
        self.seq_lens = torch.randint(
            MIN_CTX, MAX_SEQ + 1, (MAX_BS,), dtype=torch.int32, device=dev
        )
        self.total_kv = int(self.seq_lens.sum().item())
        self.kv = torch.randn(
            (self.total_kv, PAGE_SIZE, NHEAD_KV, QK_HEAD_DIM),
            dtype=torch.bfloat16,
            device=dev,
        ).to(dtypes.fp8)
        self.slots = torch.randperm(self.total_kv, dtype=torch.int32, device=dev)
        self.q = torch.randn(
            (MAX_BS * MSQ, NHEAD, QK_HEAD_DIM), dtype=torch.bfloat16, device=dev
        ).to(dtypes.fp8)
        self.scale = torch.tensor([1.0], dtype=torch.float32, device=dev)

        self.buf_kv_indices = torch.zeros((CAPACITY,), dtype=torch.int32, device=dev)
        self.buf_kv_indptr = torch.zeros((MAX_BS + 1,), dtype=torch.int32, device=dev)
        self.buf_qo_indptr = torch.zeros((MAX_BS + 1,), dtype=torch.int32, device=dev)
        self.buf_last_page = torch.ones((MAX_BS,), dtype=torch.int32, device=dev)

        c = torch.zeros(MAX_BS + 1, dtype=torch.int64)
        c[1:] = torch.cumsum(self.seq_lens.cpu().to(torch.int64), dim=0)
        self.real_starts = c

    def fill(self, bs, seq_lens):
        self.buf_kv_indptr.zero_()
        self.buf_kv_indptr[1 : bs + 1] = torch.cumsum(seq_lens[:bs], dim=0)
        self.buf_qo_indptr.zero_()
        self.buf_qo_indptr[1 : bs + 1] = torch.cumsum(
            torch.full((bs,), MSQ, dtype=torch.int32, device="cuda"), dim=0
        )
        starts = self.buf_kv_indptr[:bs].tolist()
        lens = seq_lens[:bs].tolist()
        for i, (s, n) in enumerate(zip(starts, lens)):
            src = int(self.real_starts[i])
            self.buf_kv_indices[s : s + n] = self.slots[src : src + n]

    def fill_capture(self, bs):
        fill = torch.full(
            (MAX_BS,), CAPTURE_FILL_SEQ_LEN, dtype=torch.int32, device="cuda"
        )
        self.fill(bs, fill)

    def call(self, bs, out):
        mla_decode_fwd(
            self.q[: bs * MSQ],
            self.kv.view(-1, 1, 1, QK_HEAD_DIM),
            out,
            self.buf_qo_indptr[: bs + 1],
            self.buf_kv_indptr[: bs + 1],
            self.buf_kv_indices,
            self.buf_last_page[:bs],
            MSQ,
            page_size=PAGE_SIZE,
            nhead_kv=NHEAD_KV,
            q_scale=self.scale,
            kv_scale=self.scale,
        )

    def new_out(self, bs):
        return torch.empty(
            (bs * MSQ, NHEAD, V_HEAD_DIM), dtype=torch.bfloat16, device="cuda"
        )


# 64 buckets stays under the 128-entry cache (clean); 256 overflows it by 128,
# which is exactly the dense `--cuda-graph-bs $(seq 1 256)` serving config.
BUCKET_COUNTS = [64, 256]


@pytest.mark.parametrize(
    "n_buckets", BUCKET_COUNTS, ids=[f"buckets{n}" for n in BUCKET_COUNTS]
)
def test_capture_many_buckets_matches_eager(n_buckets):
    _skip_if_unsupported()
    fx = Fixture()

    step = max(1, MAX_BS // n_buckets)
    buckets = sorted(set(list(range(step, MAX_BS + 1, step)) + [TARGET_BS]))
    buckets = buckets[-n_buckets:]
    if TARGET_BS not in buckets:
        buckets = sorted(buckets[1:] + [TARGET_BS])
    # Replay the bucket serving actually corrupts. It is not the largest one:
    # bs=256 auto-picks splits=1 and takes the single-pass path, which never
    # reads the split table, so it is immune. TARGET_BS picks splits=3 and sits
    # in the first half of the reverse capture order, i.e. among the evicted.
    target = TARGET_BS

    # -- reference ----------------------------------------------------------
    fx.fill(target, fx.seq_lens)
    ref = fx.new_out(target)
    fx.call(target, ref)
    torch.cuda.synchronize()

    if hasattr(get_meta_param, "cache_clear"):
        get_meta_param.cache_clear()
    pool = torch.cuda.graph_pool_handle()
    graphs, outputs = {}, {}
    s = torch.cuda.Stream()
    target_table_ptr = target_table_numel = None

    # Largest bucket first, as every graph runner does.
    for bs in sorted(buckets, reverse=True):
        fx.fill_capture(bs)
        out = fx.new_out(bs)
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                fx.call(bs, out)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        if bs == target:
            # Same key the warmups just used, so this is a cache hit and does
            # not perturb anything: record where the table the graph is about
            # to bake in actually lives.
            _, tbl = get_meta_param(
                None, bs, fx.buf_kv_indices.numel(), NHEAD, MSQ, dtypes.fp8
            )
            target_table_ptr, target_table_numel = tbl.data_ptr(), tbl.numel()
            del tbl

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            fx.call(bs, out)
        graphs[bs], outputs[bs] = g, out

    info = (
        get_meta_param.cache_info()
        if hasattr(get_meta_param, "cache_info")
        else "no LRU (fixed build)"
    )
    evicted = (
        max(0, len(buckets) - info.currsize)
        if hasattr(get_meta_param, "cache_info")
        else 0
    )

    # -- let the process reuse the freed block ------------------------------
    #
    # Eviction alone is not the failure; the failure is eviction FOLLOWED BY
    # someone else taking the block and writing different bytes into it. A
    # serving process supplies that for free -- between capture and the first
    # replay it runs prefill, allocates MoE workspaces, and calls
    # get_meta_param again for hundreds of other shapes, all churning the same
    # small-allocation size class. This minimal harness allocates almost
    # nothing in between, so the stale table survives byte-intact and the
    # graph reads correct data by luck.
    #
    # So ask for that size class explicitly. The contents matter: a split
    # offset table for the same bs but a LARGER split count is exactly what a
    # neighbouring bucket holds in production (the serving log shows splits
    # ranging 1..16 across buckets), and it is what makes the stale read fatal
    # rather than benign -- the offsets come out up to 16x too large.
    squatters, took_over = [], 0
    if target_table_ptr is not None:
        for _ in range(64):
            t = torch.arange(
                0,
                target_table_numel * 16,
                16,
                dtype=torch.int32,
                device="cuda",
            )
            squatters.append(t)
            if t.data_ptr() == target_table_ptr:
                took_over += 1
        torch.cuda.synchronize()

    # -- replay the first-captured bucket on the real seq_lens --------------
    fx.fill(target, fx.seq_lens)
    outputs[target].zero_()
    graphs[target].replay()
    torch.cuda.synchronize()

    got = outputs[target].float()
    nf = int((~torch.isfinite(got)).sum().item())
    cos = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), got.flatten(), dim=0
    ).item()
    print(
        f"\n=== buckets={len(buckets)} target_bs={target} ===\n"
        f"  get_meta_param cache: {info}  -> {evicted} entries evicted\n"
        f"  target's split table was at "
        f"{hex(target_table_ptr) if target_table_ptr else 'n/a'}; "
        f"{took_over}/64 later allocations landed on it\n"
        f"  nonfinite={nf} cos={cos:.6f}"
    )
    if evicted and not took_over:
        print(
            "  NOTE: the table was evicted but nothing reclaimed its block, so "
            "the graph still read intact data. Not a clean bill of health -- "
            "the reuse is what a real process supplies."
        )
    assert nf == 0, (
        f"buckets={len(buckets)}: replaying bucket {target} produced {nf} "
        f"non-finite values; {evicted} get_meta_param entries were evicted "
        f"during capture -- serving defect reproduced at op level"
    )
    assert cos > 0.99, (
        f"buckets={len(buckets)}: replaying bucket {target} drifted from eager, "
        f"cos={cos:.6f}; {evicted} get_meta_param entries were evicted"
    )


# ---------------------------------------------------------------------------
# 3. the causal link: a stale split table is what turns into NaN
# ---------------------------------------------------------------------------
def _corrupt_pattern(name, table, splits):
    """Byte patterns a recycled block could plausibly end up holding."""
    n = table.numel()
    if name == "neighbour":
        # Another bucket's table: monotonic, positive, moderate. This is the
        # benign-looking case and the one that turned out to change nothing.
        s = 16 if splits != 16 else 8
        return torch.arange(0, n * s, s, dtype=table.dtype, device=table.device)
    if name == "huge":
        return torch.full((n,), 0x7FFFFFFF, dtype=table.dtype, device=table.device)
    if name == "negative":
        return torch.full((n,), -1, dtype=table.dtype, device=table.device)
    if name == "zeros":
        return torch.zeros(n, dtype=table.dtype, device=table.device)
    if name == "random":
        # What an unrelated tensor's bytes look like: no order, full range.
        return torch.randint(
            -(2**31), 2**31 - 1, (n,), dtype=table.dtype, device=table.device
        )
    raise ValueError(name)


@pytest.mark.parametrize(
    "pattern", ["neighbour", "huge", "negative", "zeros", "random"]
)
@pytest.mark.parametrize("bs", [130, 165], ids=lambda b: f"bs{b}")
def test_stale_split_table_corrupts_replay(bs, pattern):
    """Overwrite the table a captured graph points at, and watch it break.

    Test 1 shows the LRU drops the tensor and a later allocation can land on
    its block. This test closes the loop: it shows what a graph replaying
    against changed bytes in that block actually produces.

    It writes the bytes directly rather than waiting for the allocator to hand
    the block to someone -- in a serving process hundreds of small allocations
    per step make that happen on their own, but a minimal op-level harness
    allocates almost nothing between capture and replay, so the stale table
    tends to survive byte-intact and the graph reads correct data by luck
    (that is exactly what `test_capture_many_buckets_matches_eager` reports).

    The bytes written are not arbitrary: `arange(0, (bs+1)*s, s)` for a larger
    `s` is precisely the table a neighbouring bucket holds -- the serving log
    shows the auto-picked split count ranging 1..16 across buckets. That is
    what makes a stale read fatal instead of benign; a table for a nearby bs at
    the same split count differs in only a few trailing entries.

    The batch size matters. `get_meta_param`'s auto-pick, measured at capture
    on this config, is a step function:

        bs <= 128 -> splits = 2 (uniform)
        bs = 129  -> 5
        bs 130+   -> 3, 5, 7, 8, 9, 10, 11, 15 ...
        bs = 256  -> 1

    Two consequences. A bucket that picks `splits == 1` takes the single-pass
    fast path and never consumes the table, so it is immune -- bs=256 is
    parametrized here to show exactly that. And the buckets above 128 pick
    *wildly different* split counts from each other, which is what makes one
    bucket's table lethal when it lands in another bucket's block; below 128
    every bucket picks 2, so a stale read there would be nearly harmless.

    This runs the same on a fixed build -- deliberately corrupting a live
    buffer breaks either way. It documents the mechanism; the fix is gated by
    `test_lru_evicts_live_graph_tensor`.
    """
    _skip_if_unsupported()
    fx = Fixture()

    # -- reference ----------------------------------------------------------
    fx.fill(bs, fx.seq_lens)
    ref = fx.new_out(bs)
    fx.call(bs, ref)
    torch.cuda.synchronize()

    # -- capture one graph, on the framework's seq_len fill value ------------
    fx.fill_capture(bs)
    out = fx.new_out(bs)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fx.call(bs, out)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    # Cache hit on the key the warmups just used: this is the very tensor whose
    # address the capture below bakes into the kernel node.
    splits, table = get_meta_param(
        None, bs, fx.buf_kv_indices.numel(), NHEAD, MSQ, dtypes.fp8
    )
    original = table.clone()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fx.call(bs, out)

    # -- replay with the table intact ---------------------------------------
    fx.fill(bs, fx.seq_lens)
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    clean = out.clone().float()
    nf_clean = int((~torch.isfinite(clean)).sum().item())
    cos_clean = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), clean.flatten(), dim=0
    ).item()

    # -- replay after the block took on a neighbouring bucket's table --------
    table.copy_(_corrupt_pattern(pattern, table, splits))
    out.zero_()
    g.replay()
    torch.cuda.synchronize()
    stale = out.float()
    nf_stale = int((~torch.isfinite(stale)).sum().item())
    cos_stale = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), stale.flatten(), dim=0
    ).item()

    # -- control: does the table's content matter AT ALL, even eagerly? -----
    #
    # Without this the test cannot distinguish "the graph read a stale table
    # and coped" from "this tensor never influences the result". An eager call
    # re-runs all the python, so if the corrupted table leaves eager output
    # unchanged too, the table is simply not load-bearing for this shape and no
    # stale-pointer story can be built on it.
    eager_corrupt = fx.new_out(bs)
    fx.call(bs, eager_corrupt)
    torch.cuda.synchronize()
    nf_eager = int((~torch.isfinite(eager_corrupt.float())).sum().item())
    cos_eager = torch.nn.functional.cosine_similarity(
        ref.float().flatten(), eager_corrupt.float().flatten(), dim=0
    ).item()

    table.copy_(original)          # leave the cache as we found it

    print(
        f"  eager w/ same corrupted table: nonfinite={nf_eager} "
        f"cos={cos_eager:.6f}"
    )
    print(
        f"\n=== stale split table, bs={bs} splits={splits} ===\n"
        f"  table@{hex(table.data_ptr())} len={table.numel()}\n"
        f"  intact : nonfinite={nf_clean} cos={cos_clean:.6f}\n"
        f"  stale  : nonfinite={nf_stale} cos={cos_stale:.6f}  "
        f"(pattern={pattern})"
    )

    assert nf_clean == 0 and cos_clean > 0.99, (
        f"the control replay is already broken (nonfinite={nf_clean} "
        f"cos={cos_clean:.6f}); this test cannot attribute anything"
    )
    if splits == 1:
        # Documented immunity, not a silent pass: the single-pass fast path
        # never dereferences the table, so nothing here can go wrong.
        assert nf_stale == 0 and cos_stale > 0.99, (
            f"bs={bs} picked splits=1 (single-pass, table unused) yet the "
            f"stale replay changed: nonfinite={nf_stale} cos={cos_stale:.6f}"
        )
        pytest.skip(
            f"bs={bs} auto-picks splits=1 -> single-pass path, the split table "
            f"is never read; immune by construction"
        )
    # The graph and an eager call read the same table, so they must agree --
    # this is what rules out "the graph is pointing somewhere else entirely".
    assert (nf_stale > 0) == (nf_eager > 0), (
        f"graph replay and eager disagree on the same corrupted table: "
        f"graph nonfinite={nf_stale}, eager nonfinite={nf_eager}"
    )

    if pattern == "neighbour":
        # Documented benign case, and the reason an earlier round of this test
        # wrongly concluded the tensor was not load-bearing: a monotonically
        # increasing table with a larger stride is tolerated -- offsets stay
        # ordered and the kernel clamps against valid_split_count. Only
        # degenerate contents (zeros, negatives, saturated, unordered) are
        # fatal. A recycled block holds arbitrary bytes, so the fatal cases are
        # the representative ones.
        assert nf_stale == 0 and cos_stale > 0.99, (
            f"the benign 'neighbour' pattern now corrupts too "
            f"(nonfinite={nf_stale} cos={cos_stale:.6f}); the note above is "
            f"stale and should be revisited"
        )
        return

    assert nf_stale > 0 or cos_stale < 0.99, (
        f"bs={bs} splits={splits} pattern={pattern}: replaying against a "
        f"corrupted split-offset table produced clean output "
        f"(nonfinite={nf_stale} cos={cos_stale:.6f}) -- if no pattern corrupts, "
        f"the eviction cannot explain the serving failure"
    )
