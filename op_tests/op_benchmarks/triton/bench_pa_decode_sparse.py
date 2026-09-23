# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark for ``pa_decode_sparse`` — DeepSeek-V4 sparse MLA paged decode.

Providers
---------
``bf16``            a16w16: bf16 Q ``[T, H, 512]`` against a bf16 pool
                    ``[P, 512]``. The reference for what quantization buys.
``v4_a8w8_2buff``   a8w8 over ATOM's two-buffer pool: 448 fp8 NoPE | 14 dup E8M0
                    group scales | 50 pad, plus a bf16 ``[P, 64]`` RoPE plane,
                    with the matching packed fp8 Q. Byte-for-byte the layout the
                    MLA-v4 asm decode kernel reads.
``asm``             the reference point for ``v4_a8w8_2buff``:
                    ``aiter.mla.mla_decode_fwd_v4_nm`` ->
                    ``_ZN5aiter35mla_a8w8_qh64_1tg_16mx4_64nx1_sparseE``. Reads
                    the exact same tensors (gfx1250 only, gqa in {16, 64, 128}).
``v4_a8w8``         a8w8 over **vLLM's own paged layout**: ``[nb, page, 584]``
                    uint8 fp8_ds_mla — 448 B fp8 NoPE | 128 B bf16 RoPE per
                    token, then a per-block trailer of 8 UE8M0 scale bytes each.
                    This is what a stock vLLM deployment allocates, handed over
                    with no repack, so it is the layout that matters for
                    serving.

Same quantized bytes throughout: ``v4_a8w8`` and ``v4_a8w8_2buff`` are two
layouts of one packing, so a difference between them is addressing, not
numerics.

Index streams
-------------
vLLM's decode attends over TWO paged caches per token, in one pass:

  ``main_cache``   ``swa_k_cache``: a 128-position sliding window, page 64. Its
                   slots are a contiguous range of POSITIONS resolved through
                   the block table, so they run contiguously inside a page and
                   the page order comes from the allocator. That is the only
                   shape production makes for this cache, so it is not an
                   option here -- it is simply what the bench builds.
  ``extra_cache``  ``kv_cache``: the compressed keys the sparse top-k selects.
                   **256** on CSA layers (compress_ratio 4, page 64) or **8** on
                   HCA layers (compress_ratio 128, page **2**). Sorted global
                   slot ids, scattered rather than windowed.

So the production shapes are **128 + 256** and **128 + 8**, not a single 384- or
136-long stream. ``--kv_len`` is the main stream; ``--extra-len`` /
``--extra-page`` add the second one. Only ``v4_a8w8`` can attend over both in
one pass, so the other providers sit the two-stream rows out rather than report
a different amount of work in the same column.

Usage
-----
Every example passes each argument that shapes the measurement, defaults
included, so the command can be read and reproduced on its own. ``--shape`` is
``T H D KV_LEN``; the pairs below differ only in **T** (16 and 64), which are
two decode concurrencies, not a repeat.

``--timer profiler`` throughout because wall clock cannot see these kernels: the
Python driver costs ~65us per call and dominates them, so a trivial shape
wall-clocks the same as a large one.

  # DSv4-Pro decode, CSA layers (compress_ratio 4): 128 SWA + 256 top-k,
  # both caches paged 64. H=32 is TP4 (per rank) at concurrency 16 and 64;
  # H=128 is DP-attention, which keeps every head on every rank.
  python op_tests/op_benchmarks/triton/bench_pa_decode_sparse.py \
      --shape 16 32 512 128 \
      --shape 64 32 512 128 \
      --shape 512 128 512 128 \
      --providers v4_a8w8 \
      --page 64 \
      --extra-len 256 \
      --extra-page 64 \
      --metric time \
      --timer profiler

  # DSv4-Pro decode, HCA layers (compress_ratio 128): 128 SWA + 8 top-k, and
  # the top-k cache is paged 2, not 64. Same three parallelism points.
  python op_tests/op_benchmarks/triton/bench_pa_decode_sparse.py \
      --shape 16 32 512 128 \
      --shape 64 32 512 128 \
      --shape 512 128 512 128 \
      --providers v4_a8w8 \
      --page 64 \
      --extra-len 8 \
      --extra-page 2 \
      --metric time \
      --timer profiler

  # ATOM: the two-buffer pool against the asm kernel it feeds.
  #
  # ATOM is SINGLE-STREAM over one unified pool, so kv_len here is the COMBINED
  # length -- 136 on HCA layers, 384 on CSA -- where the vLLM examples above
  # split the same work into 128 + 8 and 128 + 256 across two caches.
  #
  # Rows 1-4 are TP4 (H=32) at concurrency 16 and 64; rows 5-6 are
  # DP-attention (H=128) at concurrency 512.
  #
  # asm is blank on the H=32 rows: its .co only resolves for gqa in
  # {16, 64, 128}, so those four show the 2buff path alone.
  #
  # No --page: that pool is flat [P, 512], so the page size reaches these two
  # only through the index stream, and they measure flat across 2..128 -- the
  # rows are 512 B, so adjacent slots are not adjacent cache lines.
  python op_tests/op_benchmarks/triton/bench_pa_decode_sparse.py \
      --shape 16 32 512 136 \
      --shape 16 32 512 384 \
      --shape 64 32 512 136 \
      --shape 64 32 512 384 \
      --shape 512 128 512 136 \
      --shape 512 128 512 384 \
      --providers v4_a8w8_2buff asm \
      --extra-len 0 \
      --metric time \
      --timer profiler

  # The same CSA shapes in bandwidth rather than latency.
  python op_tests/op_benchmarks/triton/bench_pa_decode_sparse.py \
      --shape 64 32 512 128 \
      --shape 512 128 512 128 \
      --providers v4_a8w8 \
      --page 64 \
      --extra-len 256 \
      --extra-page 64 \
      --metric bandwidth \
      --timer profiler
"""

import argparse
import os
import sys

# Run as a script, sys.path[0] is THIS directory, so `import aiter` finds
# whatever is installed rather than the checkout this file belongs to -- and
# those are not always the same tree. Prefer the checkout; the resolved path is
# printed at startup so a surprising number can be traced back to a build.
if not os.environ.get("AITER_BENCH_USE_INSTALLED"):
    _REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    if os.path.isdir(os.path.join(_REPO, "aiter")):
        sys.path.insert(0, _REPO)

import torch
import triton

from aiter.ops.triton.attention.pa_decode_sparse import (
    _pa_decode_sparse_v4,
    pa_decode_sparse,
)
from aiter.ops.triton.utils._triton import arch_info

# DSv4 MLA head geometry: one 512-wide latent per token, of which the last 64
# dims are the RoPE half. V is the whole row, so the output is 512 wide too.
NOPE_DIM = 448
ROPE_DIM = 64
HEAD_DIM = NOPE_DIM + ROPE_DIM  # 512
FP8_DTYPE = torch.float8_e4m3fn  # OCP e4m3 — what the asm .co consumes
NUM_TILES = NOPE_DIM // 64  # 7 E8M0 quant groups

# The asm decode kernel is dispatched per (gqa, qSeqLen); at qSeqLen=1 only
# these head counts resolve to a shipped .co (hsa/gfx1250/mla_v4/mla_v4_asm.csv
# + the gqa remap in csrc/py_itfs_cu/asm_mla_v4.cu).
_ASM_SHIPPED_GQA = (16, 64, 128)

ALL_PROVIDERS = ("bf16", "v4_a8w8_2buff", "asm", "v4_a8w8")
METRICS = ("time", "bandwidth", "throughput")
# vLLM's fp8_ds_mla record: 448 B fp8 NoPE | 128 B bf16 RoPE | 8 B UE8M0
# (7 real scales + 1 pad), and the scales live in a per-block trailer.
V4_ROW_BYTES = NOPE_DIM + 2 * ROPE_DIM  # 576
V4_SC_RAW = NUM_TILES + 1  # 8
V4_REC_BYTES = V4_ROW_BYTES + V4_SC_RAW  # 584
DEFAULT_PAGE = 64  # what the DSv4 SWA cache uses

# DSv4-Pro head counts: 128 Q heads total. Under DP-attention every rank keeps
# all 128; under pure TP they shard, so TP4 gives 32 per rank -- which is what
# a `serve_dsv4.sh pro --tp 4` run actually executes, and what the decode grid
# in a trace shows (heads_blocks = ceil(32/16) = 2).
#
# kv_len is 384 on the CSA layers (compress_ratio 4) and 136 on the HCA layers
# (compress_ratio 128). T is the decode token count, i.e. the concurrency.
TP4_HEADS = 128 // 4
DEFAULT_SHAPES = [
    (T, TP4_HEADS, HEAD_DIM, kv_len) for kv_len in (136, 384) for T in (16, 64)
]
# Wider sweep, for scaling questions rather than the production point.
SWEEP_SHAPES = [
    (T, H, HEAD_DIM, kv_len)
    for kv_len in (136, 384)
    for H in (32, 128)
    for T in (1, 16, 64, 128, 512)
]


# ---------------------------------------------------------------------------
# Packing (mirrors op_tests/triton_tests/attention/test_pa_decode_sparse.py and
# ATOM atom/model_ops/v4_kernels/v4_quant.py)
# ---------------------------------------------------------------------------
def v4_pack_2buff(x_bf16):
    """``[..., 512]`` bf16 NoPE||RoPE -> ``(packed [..., 512] fp8, rope [..., 64] bf16)``."""
    lead = x_bf16.shape[:-1]
    nope = x_bf16[..., :NOPE_DIM].float()
    rope = x_bf16[..., NOPE_DIM:].contiguous()

    tiled = nope.reshape(*lead, NUM_TILES, 64)
    fp8_max = float(torch.finfo(FP8_DTYPE).max)
    scale = torch.pow(
        2.0, torch.clamp_min(tiled.abs().amax(dim=-1) / fp8_max, 1e-4).log2().ceil()
    )
    nope_fp8 = (tiled / scale.unsqueeze(-1)).to(FP8_DTYPE).reshape(*lead, NOPE_DIM)
    e8m0 = (scale.log2().round().to(torch.int32) + 127).clamp(0, 254).to(torch.uint8)

    packed = torch.zeros((*lead, HEAD_DIM), dtype=torch.uint8, device=x_bf16.device)
    packed[..., :NOPE_DIM] = nope_fp8.view(torch.uint8)
    packed[..., NOPE_DIM : NOPE_DIM + 2 * NUM_TILES] = e8m0.repeat_interleave(2, dim=-1)
    return packed.view(FP8_DTYPE), rope


def v4_pack_unified(packed_2buff, rope, page):
    """2buff row + RoPE plane -> ``[nb, page, 584]`` uint8, vLLM's fp8_ds_mla.

    Re-lays out the SAME bytes ``v4_pack_2buff`` produced, so a kernel reading
    either format sees identical quantized values. Mirrors
    op_tests/triton_tests/attention/test_pa_decode_sparse.py.
    """
    u8 = packed_2buff.view(torch.uint8)
    p = u8.shape[0]
    assert p % page == 0, f"{p} rows is not a whole number of {page}-row pages"
    nb = p // page
    data = torch.cat(
        [
            u8[:, :NOPE_DIM],
            rope.reshape(p, ROPE_DIM).view(torch.uint8).reshape(p, 2 * ROPE_DIM),
        ],
        dim=-1,
    )  # [P, 576]
    # 2buff stores each group's scale byte twice; the unified trailer once.
    scales = torch.zeros(p, V4_SC_RAW, dtype=torch.uint8, device=u8.device)
    scales[:, :NUM_TILES] = u8[:, NOPE_DIM : NOPE_DIM + 2 * NUM_TILES : 2]
    cache = torch.empty(nb, page * V4_REC_BYTES, dtype=torch.uint8, device=u8.device)
    cache[:, : page * V4_ROW_BYTES] = data.reshape(nb, page * V4_ROW_BYTES)
    cache[:, page * V4_ROW_BYTES :] = scales.reshape(nb, page * V4_SC_RAW)
    return cache.reshape(nb, page, V4_REC_BYTES)


def paged_cache(packed_2buff, rope, page):
    """``v4_pack_unified`` with a block stride the descriptors can address.

    The data descriptor counts 64-byte rows, so the cache's stride(0) must be a
    multiple of 64. A contiguous page-64 cache already is (64*584 = 37376). A
    contiguous page-2 one is not (2*584 = 1168), and DSv4-Pro's HCA layers page
    the compressed cache 2 tokens to a block -- so pad it into a wider stride,
    which is also how vLLM's caches look: views into one shared allocation.
    """
    cache = v4_pack_unified(packed_2buff, rope, page)
    nb, _, rec = cache.shape
    stride0 = page * rec
    if stride0 % 64 == 0:
        return cache
    stride0 = ((stride0 + 63) // 64) * 64
    big = torch.zeros(nb * stride0, dtype=torch.uint8, device=cache.device)
    view = big.as_strided((nb, page, rec), (stride0, rec, 1))
    view.copy_(cache)
    return view


def build_topk_indices(T, n, npages, page, device, gen):
    """Per-token slot ids for the top-k stream: sorted, scattered.

    `compute_global_topk_ragged_indices_and_indptr` hands the kernel global slot
    ids in ascending order, selected from the compressed sequence -- so they are
    sorted but not contiguous, unlike the SWA window.
    """
    pool = npages * page
    rows = [
        torch.sort(torch.randperm(pool, device=device, generator=gen)[:n]).values
        for _ in range(T)
    ]
    return torch.cat(rows).to(torch.int32)


def build_slot_indices(T, lens, npages, page, device, gen):
    """Per-token slot ids, shaped as vLLM's SWA stream.

    Mirrors what `_compute_swa_indices_and_lens_kernel` produces:

        slot = block_table[pos // page] * page + pos % page

    over ascending positions -- so the slots are contiguous inside a page, and
    the page order is whatever the allocator gave. This is the only shape
    production produces for the main cache.

    Pages are dealt out disjointly across tokens, so the working set stays the
    whole pool. Letting tokens share rows left much of it L2-resident and made
    the kernel look ~28 % faster than a real trace.
    """
    # One block table per token, over disjoint pages.
    need = [int((int(n) + page - 1) // page) + 1 for n in lens]
    if sum(need) > npages:
        raise ValueError(
            f"the window needs {sum(need)} pages but the pool has {npages}; "
            "raise kv_len or lower T"
        )
    # Shuffled, because that is what a fragmented allocator gives. A
    # sequential table is the lucky case and nothing guarantees it.
    order = torch.randperm(npages, device=device, generator=gen)
    rows = []
    cursor = 0
    for t, n in enumerate(lens):
        n = int(n)
        table = order[cursor : cursor + need[t]]
        cursor += need[t]
        # start part-way into the first page, which is the normal case and the
        # one that makes the first tile of an aligned tiling partial
        off0 = int(torch.randint(0, page, (1,), device=device, generator=gen))
        pos = off0 + torch.arange(n, device=device)
        rows.append(table[pos // page] * page + (pos % page))
    return torch.cat(rows).to(torch.int32)


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
_INPUT_CACHE = {}


def build_inputs(T, H, D, kv_len, var_len=False, seed=0, device="cuda",
                 page=DEFAULT_PAGE, extra_len=0, extra_page=DEFAULT_PAGE):
    """Inputs for one shape, memoized.

    perf_report calls the bench fn once per (shape, provider), so without the
    cache a 5-provider sweep rebuilds and re-quantizes the whole KV pool 5
    times -- at T=512, kv_len=384 that is ~100M elements quantized three
    different ways per provider, which dominated the sweep's wall clock.
    """
    key = (T, H, D, kv_len, bool(var_len), seed, str(device), page,
           extra_len, extra_page)
    if key in _INPUT_CACHE:
        return _INPUT_CACHE[key]
    torch.manual_seed(seed)
    # A window needs whole pages, and one more per token than its length
    # implies, because it can start part-way into the first one.
    npages = -(-(T * kv_len) // page) + 2 * T
    pages = npages * page

    q = torch.randn(T, H, D, dtype=torch.bfloat16, device=device) * 0.5
    kv = torch.randn(pages, D, dtype=torch.bfloat16, device=device) * 0.5
    sink = torch.randn(H, dtype=torch.float32, device=device) * 0.1

    if var_len:
        lens = torch.randint(1, kv_len + 1, (T,), device=device, dtype=torch.int64)
    else:
        lens = torch.full((T,), kv_len, device=device, dtype=torch.int64)
    indptr = torch.zeros(T + 1, device=device, dtype=torch.int64)
    indptr[1:] = lens.cumsum(0)
    total_indices = int(indptr[-1].item())
    # Every gathered row is DISTINCT across the whole batch: one permutation of
    # the pool, sliced per token. randint-with-replacement let rows repeat both
    # inside a token's context and across tokens, so much of the pool stayed
    # L2-resident and the kernel looked faster than it is in production -- the
    # isolated numbers came out ~28% under the same asm kernel's time in a real
    # DSv4 trace. With a permutation the working set is the entire pool
    # (T * kv_len rows), which is what a real KV cache looks like.
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    indices = build_slot_indices(T, lens.tolist(), npages, page, device, gen)
    # var_len with a window shortens each token's run, so the indptr above
    # already matches what build_slot_indices produced.
    total_indices = int(indices.numel())

    kv_packed, kv_rope = v4_pack_2buff(kv)
    q_packed, q_rope = v4_pack_2buff(q)
    unified = paged_cache(kv_packed, kv_rope, page)

    # The top-k stream: its own cache, its own scattered index set.
    extra = None
    if extra_len:
        x_npages = -(-(T * extra_len * 4) // extra_page)
        x_pool = x_npages * extra_page
        x_kv = torch.randn(x_pool, D, dtype=torch.bfloat16, device=device) * 0.5
        x_packed, x_rope = v4_pack_2buff(x_kv)
        extra = {
            "cache": paged_cache(x_packed, x_rope, extra_page),
            "indices": build_topk_indices(
                T, extra_len, x_npages, extra_page, device, gen
            ),
            "indptr": (
                torch.arange(T + 1, device=device) * extra_len
            ).to(torch.int32),
            "n_idx": T * extra_len,
        }

    _INPUT_CACHE[key] = {
        "extra": extra,
        "unified": unified,
        "page": page,
        "q": q,
        "kv": kv,
        "q_packed": q_packed,
        "q_rope": q_rope,
        "kv_packed": kv_packed,
        "kv_rope": kv_rope,
        "indices": indices,
        "indptr": indptr.to(torch.int32),
        "sink": sink,
        "total_indices": total_indices,
        "softmax_scale": float(D) ** -0.5,
    }
    return _INPUT_CACHE[key]


def _make_fn(provider, inp, T, H, D):
    """Return ``(callable, bytes_moved)`` for one provider, or None if unsupported."""
    if inp["extra"] is not None and provider != "v4_a8w8":
        # Only the vLLM paged path attends over two caches in one pass. Timing
        # the others on the main stream alone would be a different amount of
        # work reported in the same column.
        return None
    ind, iptr, sink = inp["indices"], inp["indptr"], inp["sink"]
    scale = inp["softmax_scale"]
    n_idx = inp["total_indices"]
    out_bytes = T * H * D * 2

    if provider == "bf16":
        q, kv = inp["q"], inp["kv"]
        fn = lambda: pa_decode_sparse(q, kv, ind, iptr, sink, scale, has_invalid=False)
        # gathered KV + Q read + output written
        return fn, n_idx * D * 2 + T * H * D * 2 + out_bytes

    if provider == "v4_a8w8_2buff":
        kvp, kvr = inp["kv_packed"], inp["kv_rope"]
        fn = lambda: pa_decode_sparse(
            inp["q_packed"],
            kvp,
            ind,
            iptr,
            sink,
            scale,
            has_invalid=False,
            unified_kv_rope=kvr,
            q_rope=inp["q_rope"],
        )
        kv_row = HEAD_DIM * 1 + ROPE_DIM * 2  # 512 B fp8 + 128 B bf16 RoPE
        return fn, n_idx * kv_row + T * H * kv_row + out_bytes

    if provider == "asm":
        if arch_info.get_arch() != "gfx1250" or H not in _ASM_SHIPPED_GQA:
            return None
        try:
            import aiter.mla
        except ImportError:
            return None
        # Dispatched from a prebuilt .co + a csv row; a tree without those
        # assets has no asm decode to time.
        mla_decode_fwd_v4_nm = getattr(aiter.mla, "mla_decode_fwd_v4_nm", None)
        if mla_decode_fwd_v4_nm is None:
            return None
        qp, qr = inp["q_packed"], inp["q_rope"]
        kvp = inp["kv_packed"].view(-1, 1, 1, HEAD_DIM)
        kvr = inp["kv_rope"].view(-1, 1, 1, ROPE_DIM)
        qo_indptr = torch.arange(0, T + 1, dtype=torch.int32, device=qp.device)
        out = torch.empty((T, H, HEAD_DIM), dtype=torch.bfloat16, device=qp.device)
        fn = lambda: mla_decode_fwd_v4_nm(
            qp,
            qr,
            kvp,
            kvr,
            out,
            qo_indptr,
            iptr,
            ind,
            1,  # max_seqlen_q — page_size=1, one query row per sequence
            sink=sink,
            sm_scale=scale,
        )
        kv_row = HEAD_DIM * 1 + ROPE_DIM * 2
        return fn, n_idx * kv_row + T * H * kv_row + out_bytes

    if provider == "v4_a8w8":
        # vLLM's own paged layout: [nb, page, 584] uint8, no companion RoPE
        # plane. This is what a stock deployment allocates, and the only
        # provider here that can attend over both streams in one pass.
        unified, qp, qr = inp["unified"], inp["q_packed"], inp["q_rope"]
        x = inp["extra"]
        kw = {}
        if x is not None:
            kw = {
                "extra_cache": x["cache"],
                "extra_indices": x["indices"],
                "extra_indptr": x["indptr"],
            }
        fn = lambda: _pa_decode_sparse_v4(
            qp, unified, ind, iptr, sink, scale,
            q_rope=qr, has_invalid=False, block_k=inp.get("block_k"),
            main_is_window=inp.get("main_is_window", False), **kw,
        )
        kv_row = HEAD_DIM * 1 + ROPE_DIM * 2
        keys = n_idx + (x["n_idx"] if x is not None else 0)
        return fn, keys * kv_row + T * H * kv_row + out_bytes

    raise ValueError(f"unknown provider {provider}")


# ---------------------------------------------------------------------------
# Bench
# ---------------------------------------------------------------------------
_KERNEL_TAGS = ("pa_decode_sparse", "mla_", "_v4_", "sparse_attn_decode")


def _device_ms(fn, provider, iters=50):
    """Milliseconds of DEVICE time per call, summed over this op's kernels.

    Wall clock cannot see these kernels: the Python driver costs ~65 us per
    call, so a trivial shape measures the same as a large one. The profiler
    reports what the GPU actually spent, which is the only basis on which two
    implementations here can be compared.
    """
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total = 0.0
    for e in prof.key_averages():
        dev = getattr(e, "self_device_time_total", None)
        if dev is None:
            dev = getattr(e, "self_cuda_time_total", 0.0)
        if dev and any(tag in e.key for tag in _KERNEL_TAGS):
            total += dev
    return total / iters * 1e-3  # us -> ms, to match do_bench's unit


def bench_fn(T, H, D, kv_len, provider, metric, var_len, cudagraph, rep,
             profile_dir=None, page=DEFAULT_PAGE, block_k=None, timer="wall",
             extra_len=0, extra_page=DEFAULT_PAGE, main_is_window=False):
    inp = build_inputs(T, H, D, kv_len, var_len=var_len, page=page,
                       extra_len=extra_len, extra_page=extra_page)
    inp["block_k"] = block_k
    inp["main_is_window"] = main_is_window
    made = _make_fn(provider, inp, T, H, D)
    if made is None:
        return float("nan")
    fn, nbytes = made

    try:
        fn()  # compile / dispatch outside the timed region
        torch.cuda.synchronize()
    except Exception as e:  # noqa: BLE001 — one bad shape must not kill the sweep
        print(f"  [{provider}] T={T} H={H} kv_len={kv_len}: {e}", file=sys.stderr)
        return float("nan")

    if profile_dir:
        # Steady-state capture: the kernel is already compiled and warm above,
        # so the trace holds only the launches we care about.
        import os

        os.makedirs(profile_dir, exist_ok=True)
        for _ in range(10):
            fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=False,
            with_stack=False,
        ) as prof:
            for _ in range(50):
                fn()
            torch.cuda.synchronize()
        tag = f"{provider}_T{T}_H{H}_kv{kv_len}"
        prof.export_chrome_trace(os.path.join(profile_dir, f"{tag}.json"))
        top = prof.key_averages().table(
            sort_by="self_device_time_total", row_limit=8
        )
        print(f"\n--- {tag} ---\n{top}", file=sys.stderr)

    if timer == "profiler":
        ms = _device_ms(fn, provider)
    elif cudagraph:
        if provider.startswith("v4_a8w8") and rep > 20:
            print(
                f"  [{provider}] --cudagraph with --rep {rep}: this driver "
                "allocates split-K partials per call, so the capture holds one "
                "set per iteration. Use --rep 5, or --timer profiler.",
                file=sys.stderr,
            )
        ms = triton.testing.do_bench_cudagraph(fn, rep=rep)
    else:
        ms = triton.testing.do_bench(fn, warmup=max(1, rep // 4), rep=rep)

    if metric == "time":
        return ms * 1e3  # us
    if metric == "bandwidth":
        return nbytes / (ms * 1e-3) * 1e-12  # TB/s
    if metric == "throughput":
        # per token: H heads x kv_len keys x (QK over D + PV over D), 2 flop each
        flops = 2.0 * H * inp["total_indices"] * 2 * D
        return flops / (ms * 1e-3) * 1e-12  # TFLOP/s
    raise ValueError(f"unknown metric {metric}")


def run_benchmark(args):
    providers = list(args.providers)
    unit = {"time": "us", "bandwidth": "TB/s", "throughput": "TFLOP/s"}[args.metric]

    if args.shape:
        shapes = [tuple(sh) for sh in args.shape]
    elif args.sweep:
        shapes = SWEEP_SHAPES
    else:
        shapes = DEFAULT_SHAPES

    # extra_len rides along as a column so a row says which layer type it is:
    # kv_len=128 alone does not distinguish CSA (128 + 256) from HCA (128 + 8).
    shapes = [tuple(sh) + (args.extra_len,) for sh in shapes]

    benchmark = triton.testing.Benchmark(
        x_names=["T", "H", "D", "kv_len", "extra_len"],
        x_vals=shapes,
        line_arg="provider",
        line_vals=providers,
        line_names=list(providers),  # perf_report appends the ylabel
        # one style per provider -- cycled, so adding a provider cannot
        # IndexError in triton's plotting path
        styles=[
            [
                ("green", "-"),
                ("blue", "-"),
                ("cyan", "--"),
                ("magenta", "--"),
                ("red", "-"),
            ][i % 5]
            for i in range(len(providers))
        ],
        ylabel=unit,
        plot_name=(
            f"pa-decode-sparse-{args.metric}"
            f"-{args.timer}"
            f"{f'-x{args.extra_len}' if args.extra_len else ''}"
            f"{f'-k{args.block_k}' if args.block_k else ''}"
            f"{'-window' if args.main_is_window else ''}"
            f"{'-cudagraph' if args.cudagraph else ''}"
            f"{'-varlen' if args.var_len else ''}"
        ),
        args={},
    )

    @triton.testing.perf_report([benchmark])
    def _bench(T, H, D, kv_len, extra_len, provider):
        return bench_fn(
            T,
            H,
            D,
            kv_len,
            provider,
            args.metric,
            args.var_len,
            args.cudagraph,
            args.rep,
            args.profile,
            args.page,
            args.block_k,
            args.timer,
            extra_len,
            args.extra_page,
            args.main_is_window,
        )

    _bench.run(save_path="." if args.o else None, print_data=True)

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Benchmark pa_decode_sparse (DSv4 sparse MLA decode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep T x H x kv_len instead of the TP4 production points.",
    )
    p.add_argument(
        "--profile",
        metavar="DIR",
        default=None,
        help="Capture a torch profiler trace of the timed region into DIR "
        "(one file per shape/provider). Use with a single --shape and a single "
        "--providers entry, otherwise the trace mixes launches.",
    )
    p.add_argument(
        "--shape",
        action="append",
        nargs=4,
        type=int,
        metavar=("T", "H", "D", "KV_LEN"),
        default=None,
        help="Shape to benchmark; repeat the flag for several. Default is the "
        "TP4 production set (T 16/64, H 32, kv_len 136/384).",
    )
    p.add_argument(
        "--providers",
        nargs="+",
        choices=ALL_PROVIDERS,
        default=list(ALL_PROVIDERS),
        help="Which implementations to time.",
    )
    p.add_argument("--metric", choices=METRICS, default="time")
    p.add_argument(
        "--extra-len",
        type=int,
        default=0,
        help="Keys in the SECOND (top-k) stream, attended in the same pass. "
        "vLLM's decode always has one: 256 on CSA layers (compress_ratio 4) or "
        "8 on HCA layers (compress_ratio 128), against a 128-key SWA main "
        "stream. 0 leaves the bench single-stream. Only v4_a8w8 supports it.",
    )
    p.add_argument(
        "--extra-page",
        type=int,
        default=DEFAULT_PAGE,
        help="Page size of the top-k cache. 64 on CSA layers; DSv4-Pro's HCA "
        "layers page it **2**, which is why the 128+8 example passes it.",
    )
    p.add_argument(
        "--page",
        type=int,
        default=DEFAULT_PAGE,
        help="Paged-cache page size. 64 is what vLLM's DSv4 SWA cache uses; "
        "its HCA compressed cache is paged 2. Sets the run length within a "
        "page, so it shapes the index stream as well as the v4_a8w8 layout -- "
        "but ATOM's pool is flat and its providers measure flat across 2..128, "
        "so it is only meaningful for v4_a8w8.",
    )
    p.add_argument(
        "--block-k",
        type=int,
        default=None,
        help="Override the KV tile size for v4_a8w8. The driver ties "
        "it to block_h (16, or 64 at block_h=128); under the window path a "
        "32-row tile is often better, because async_load costs one TDM "
        "instruction whatever the tile while an int32 gather costs "
        "ceil(BLOCK_K/8).",
    )
    p.add_argument(
        "--main-is-window",
        action="store_true",
        help="Attend the MAIN stream as a sliding window: read each page's "
        "first slot and async_load the tile, instead of async_gather over a "
        "slot vector. Only v4_a8w8 supports it, and it requires --block-k to "
        "divide --page. Off by default, which is the driver's default.",
    )
    p.add_argument(
        "--var_len",
        action="store_true",
        help="Random per-token kv_len in [1, KV_LEN] instead of a fixed length.",
    )
    p.add_argument(
        "--timer",
        choices=("wall", "profiler"),
        default="wall",
        help="How to measure. 'wall' is do_bench (or do_bench_cudagraph with "
        "--cudagraph). 'profiler' sums DEVICE time over this op's kernels, "
        "which is the only way to see them: the Python driver costs ~65us per "
        "call, so a wall-clock run reports the driver and a trivial shape "
        "measures the same as a large one.",
    )
    p.add_argument(
        "--cudagraph",
        action="store_true",
        help="Use do_bench_cudagraph instead of do_bench — removes the launch "
        "overhead that dominates these small, bandwidth-bound decode kernels.",
    )
    p.add_argument(
        "--rep",
        type=int,
        default=100,
        help="Milliseconds of timed repetitions per measurement. Lower it to "
        "keep a sweep short; the DSv4 decode kernels are tens of us, so even "
        "a few ms of reps is many iterations.",
    )
    p.add_argument("-o", action="store_true", help="Write the results to ./")
    return p.parse_args(argv)


def main(argv=None):
    if not torch.cuda.is_available():
        raise SystemExit("pa_decode_sparse benchmark requires a CUDA/HIP device")
    import aiter

    print(f"aiter: {os.path.dirname(aiter.__file__)}", file=sys.stderr)
    run_benchmark(parse_args(argv))


if __name__ == "__main__":
    main()
