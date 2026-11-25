# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""RDNA decode and prefill attention (aiter.ops.rdna_attn) on gfx1150..gfx1153
and RDNA4 (gfx1200, gfx1201).

Every build of the current GPU's tables is timed and checked against the fp32
reference, one table per kind of launch: decode, prefill, prefill batches,
non-causal builds and encoder attention. Then the behaviour the tables do not
show: CUDA-graph replay, multimodal ranges, sliding windows over freed pages
and the calls the kernels refuse.

The first launch of a head dim and dtype builds its module, so a full run
builds one per (head dim, dtype) of the tables.

    python3 op_tests/test_rdna_attn.py
    python3 op_tests/test_rdna_attn.py -d fp16 -s 32,8,128,0 -m 1 4
"""

import argparse
import functools
import itertools

import numpy as np
import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops import rdna_attn as hip
from aiter.ops.torch_ref.rdna_attn import full_attention, paged_attention
from aiter.test_common import benchmark, checkAllclose, run_perftest

SUPPORTED_GFX = ["gfx1150", "gfx1151", "gfx1152", "gfx1153", "gfx1200", "gfx1201"]

# Relative error bound per dtype, for decode.
TOL = {torch.float16: 1e-3, torch.bfloat16: 8e-3}
# Absolute error bound per dtype, for prefill.
PREFILL_TOL = {torch.float16: 4e-3, torch.bfloat16: 2e-2}
DTYPES = list(TOL)
# Launches timed per row: a full run times every build of the tables.
ITERS = 20

# (M, S) per sequence of a prefill batch.
BATCHES = {
    "decode+chunk+prompt": [(1, 6000), (300, 2348), (40, 40)],
    "8 prompts": [(256, 256)] * 8,
    "unequal decodes": [(1, 37), (3, 1000)],
    "padded": [(64, 1064), (0, 0), (17, 17)],
}
# Segment lengths of a non-causal batch.
NON_CAUSAL_BATCHES = {
    "5 x 577": [577] * 5,
    "mixed": [1024, 257, 9, 100, 64],
    "windows": [64] * 20 + [16, 36, 48],
}
# Segment lengths of an encoder call.
ENCODER_SEGMENTS = {
    "short": [4, 8, 1, 3],
    "mixed": [577, 5, 1024, 8, 9, 64, 8],
    "first step": [9] * 7,
    "one": [4096],
    "one short": [5],
}


# ---------------------------------------------------------------- helpers


def _dtype_name(dtype) -> str:
    return "bf16" if dtype == torch.bfloat16 else "fp16"


def _rounds_p(v) -> bool:
    """On RDNA4, fp16 WMMA decode builds with more than 8 rows per workgroup
    take P rounded to fp16 into P@V."""
    if not isinstance(v, hip.KernelVariant):
        return False
    gqa = v.num_q_heads // v.num_kv_heads
    rows = gqa * v.max_query_len // v.row_groups
    return (
        hip.device_family() == "GFX1201"
        and v.dtype == torch.float16
        and not v.dot_product
        and rows > 8
    )


def _max_rel(got, ref, v=None) -> float:
    """Largest relative error, with a floor. With P rounded to fp16 every term
    errs by up to 2^-11, so outputs near zero are judged against the largest
    one."""
    floor = 1e-3
    if v is not None and _rounds_p(v):
        floor = max(floor, ref.abs().max().item())
    return ((got - ref).abs() / ref.abs().clamp_min(floor)).max().item()


def _assert_decode_close(got, ref, v, what) -> float:
    """Every element within the decode bound of its reference; the mismatch
    fraction (0)."""
    tol = TOL[v.dtype]
    floor = ref.abs().max().item() if _rounds_p(v) else 1e-3
    err = checkAllclose(
        ref,
        got.float().nan_to_num(1e9),
        rtol=tol,
        atol=tol * floor,
        msg=f"{what}: ",
        printLog=False,
    )
    assert err == 0, f"{what}: {err:.2%} of elements outside {tol}"
    return err


def _assert_prefill_close(got, ref, dtype, what) -> float:
    """Every element within the prefill bound of its reference; the mismatch
    fraction (0)."""
    tol = PREFILL_TOL[dtype]
    err = checkAllclose(
        ref,
        got.float().nan_to_num(1e9),
        rtol=0,
        atol=tol,
        msg=f"{what}: ",
        printLog=False,
    )
    assert err == 0, f"{what}: {err:.2%} of elements outside {tol}"
    return err


def _visible_keys(m: int, s: int, window: int) -> int:
    """Keys seen by the last m tokens of a sequence of s, summed."""
    prefix = s - m
    if not window:
        return m * prefix + m * (m + 1) // 2
    return sum(min(prefix + t + 1, window) for t in range(m))


def _roofline_metrics(us, flops, nbytes) -> dict:
    return {"us": us, "TFLOPS": flops / us / 1e6, "TB/s": nbytes / us / 1e6}


def _causal_work(hq, hkv, d, m, s, window, itemsize=2):
    """(FLOPs, bytes) of m query tokens at the end of a sequence of s: Q@K and
    P@V over the keys each token sees; Q, the visible KV and the output."""
    flops = 4.0 * hq * d * _visible_keys(m, s, window)
    keys = min(s, window + m - 1) if window else s
    nbytes = (2 * m * hq * d + keys * hkv * 2 * d) * itemsize
    return flops, nbytes


def _paged_inputs(seq_len, v, seed=0):
    """Q and an HND KV cache whose last page is partly used."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    num_blocks = -(-seq_len // v.page_size)
    kv = torch.randn(
        num_blocks,
        v.num_kv_heads,
        v.page_size,
        2 * v.head_size,
        device=dev,
        dtype=v.dtype,
    )
    q = torch.randn(
        v.max_query_len, v.num_q_heads, v.head_size, device=dev, dtype=v.dtype
    )
    return q * 0.5, kv * 0.5, torch.arange(num_blocks, device=dev, dtype=torch.int32)


def _batch_inputs(v, lens):
    """Each sequence's (q, kv, block table), and the batch's q, kv and block
    table over one cache."""
    width = max(lens) // v.page_size + 1
    parts = [_paged_inputs(max(s, 16), v, seed=i) for i, s in enumerate(lens)]
    kv = torch.cat([p[1] for p in parts])
    first = torch.tensor([0] + [p[1].shape[0] for p in parts]).cumsum(0)
    block_table = torch.zeros(len(lens), width, device=kv.device, dtype=torch.int32)
    for i, p in enumerate(parts):
        block_table[i, : p[1].shape[0]] = p[2] + int(first[i])
    return parts, torch.cat([p[0] for p in parts]), kv, block_table


def _prefill_inputs(v, m, seq_len, seed=0):
    """Q for m tokens and a shuffled HND cache of seq_len keys."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    num_blocks = -(-seq_len // v.page_size)
    kv = torch.randn(
        num_blocks + 2, v.num_kv_heads, v.page_size, 2 * v.head_size, device=dev
    )
    q = torch.randn(m, v.num_q_heads, v.head_size, device=dev)
    block_table = torch.randperm(num_blocks + 2, device=dev)[:num_blocks]
    return (
        (q * 0.5).to(v.dtype),
        (kv * 0.5).to(v.dtype),
        block_table.to(torch.int32).contiguous(),
    )


def _prefill_batch(v, lens, seed=0):
    """One cache, each sequence its own shuffled pages; q packed."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    pages = [-(-s // v.page_size) for _, s in lens]
    total = sum(pages) + 2
    kv = torch.randn(total, v.num_kv_heads, v.page_size, 2 * v.head_size, device=dev)
    order = torch.randperm(total, device=dev).to(torch.int32)
    block_table = torch.zeros(len(lens), max(pages), dtype=torch.int32, device=dev)
    for i, (first, n) in enumerate(zip(np.cumsum([0] + pages), pages)):
        block_table[i, :n] = order[first : first + n]
    starts = torch.tensor(np.cumsum([0] + [m for m, _ in lens]), dtype=torch.int32)
    q = torch.randn(int(starts[-1]), v.num_q_heads, v.head_size, device=dev)
    seq_lens = torch.tensor([s for _, s in lens], dtype=torch.int32)
    return (q * 0.5).to(v.dtype), (kv * 0.5).to(v.dtype), block_table, starts, seq_lens


def _non_causal_inputs(v, lens, seed=0):
    """Q, K and V of sequences of `lens` tokens, one after the other, as views
    of one (tokens, 3, H, D) projection (a token stride of 3 H D), as an
    encoder's QKV projection gives them."""
    torch.manual_seed(seed)
    n = sum(lens)
    qkv = torch.randn(n, 3, v.num_q_heads, v.head_size, device="cuda") * 0.5
    qkv = qkv.to(v.dtype)
    q, k, val = qkv[:, 0], qkv[:, 1, : v.num_kv_heads], qkv[:, 2, : v.num_kv_heads]
    return q, k, val, np.cumsum([0] + list(lens))


def _mm_ranges_tensor(mm_ranges, device):
    """One sequence's ranges as (1, ranges, 2), padded with an empty range."""
    pairs = [*mm_ranges, (0, 0)]
    return torch.tensor([pairs], device=device, dtype=torch.int32)


def _decode_mm_buf(ranges, device, capacity=4):
    """The decode kernel's [count, start0, end0, ...], empty slots zeroed."""
    buf = torch.zeros(1 + 2 * capacity, dtype=torch.int32)
    flat = [len(ranges), *(x for r in ranges for x in r)]
    buf[: len(flat)] = torch.tensor(flat, dtype=torch.int32)
    return buf.to(device)


def _decode_mm(module, v, q, kv, block_table, seq_len, buf):
    out = torch.full_like(q, float("nan"))
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    module.decode_attn(
        q,
        kv,
        block_table,
        out,
        *hip.make_scratch(v, q.device),
        seq_lens,
        q.shape[2] ** -0.5,
        buf,
    )
    torch.cuda.synchronize()
    return out.float().nan_to_num(1e9)


def _decode_launch(module, q, kv, block_table, out, scratch, seq_lens, scale):
    """A launch of a decode build into `out`, which it returns."""

    def launch():
        module.decode_attn(q, kv, block_table, out, *scratch, seq_lens, scale)
        return out

    return launch


def _built(dtype=torch.float16, batch=0, **want):
    """The first build of the tables with these properties, loaded.

    Args:
        dtype: Element type.
        batch: 1 for a batched build.
        **want: Predicates on KernelVariant fields, by name.

    Returns:
        (variant, module).

    """
    for v in hip.variants():
        if (
            v.dtype == dtype
            and v.batched == batch
            and all(pred(getattr(v, k)) for k, pred in want.items())
        ):
            module = hip.load(v)
            assert module is not None, f"{v.name} is not in the tables"
            return v, module
    raise AssertionError(f"no build with {sorted(want)}")


def _prefill_rows():
    return [v for rows in hip._prefill_table().values() for v in rows]


def _refused(match: str, launch) -> None:
    """`launch` must raise a RuntimeError whose message contains `match`."""
    try:
        launch()
    except RuntimeError as e:
        assert match in str(e), f"refused for another reason: {e}"
        return
    torch.cuda.synchronize()
    raise AssertionError(f"not refused (expected {match!r})")


# ------------------------------------------------- every build, in tables


@benchmark()
def test_decode(hq, hkv, d, window, page, m, dtype, batched):
    """The decode build of one M: one sequence at three lengths, or a batch of
    a long, a short and a padded sequence."""
    v = hip.variant_for(hq, hkv, d, m, page, 1, window, dtype, bool(batched))
    assert v is not None, (hq, hkv, d, window, page, m, dtype, batched)
    module = hip.load(v)
    assert module is not None, v.name
    assert not (batched and v.dot_product), v.name
    dev = torch.device("cuda")
    s = max(1024, window + 64, v.page_size + 1000)
    scale = d**-0.5
    if batched:
        lens = [s, 48, 0]
        parts, q, kv, block_table = _batch_inputs(v, lens)
        scratch = hip.make_scratch(v, dev, max_seqs=len(lens))
        seq_lens = torch.tensor(lens, device=dev, dtype=torch.int32)
        # A padded sequence (S=0) must stay unwritten.
        out = torch.full_like(q, 7.0)

        launch = _decode_launch(
            module, q, kv, block_table, out, scratch, seq_lens, scale
        )
        got, us = run_perftest(launch, num_iters=ITERS)
        for i, n in enumerate(lens):
            rows = got[i * m : (i + 1) * m]
            if n == 0:
                assert (rows == 7.0).all(), f"{v.name}: a padded sequence was written"
                continue
            ref = paged_attention(parts[i][0], parts[i][1], n, window)
            err = _assert_decode_close(rows, ref, v, f"{v.name}: sequence {i}, S={n}")
        work = [_causal_work(hq, hkv, d, m, n, window) for n in lens if n]
    else:
        # Launches sharing one scratch, each on new inputs.
        scratch = hip.make_scratch(v, dev)
        for seed, n in enumerate((48, s + 1000, s)):
            q, kv, block_table = _paged_inputs(n, v, seed)
            out = torch.full_like(q, float("nan"))
            seq_lens = torch.tensor([n], device=dev, dtype=torch.int32)
            launch = _decode_launch(
                module, q, kv, block_table, out, scratch, seq_lens, scale
            )
            got, us = run_perftest(launch, num_iters=ITERS)
            ref = paged_attention(q, kv, n, window)
            err = _assert_decode_close(got, ref, v, f"{v.name}, S={n}")
        # An image ending among the query tokens: rows on each side differ.
        ranges = [(s - m - 40, s - m + m // 2)]
        buf = _decode_mm_buf(ranges, dev)
        got = _decode_mm(module, v, q, kv, block_table, s, buf)
        with_image = paged_attention(q, kv, s, window, ranges)
        _assert_decode_close(got, with_image, v, f"{v.name}, image")
        # At M=1 the image ends at the only query token, so nothing changes.
        if m > 1:
            assert _max_rel(ref, with_image) > 10 * TOL[dtype], f"{v.name}, image"
        work = [_causal_work(hq, hkv, d, m, s, window)]
    flops, nbytes = map(sum, zip(*work))
    return {
        "gfx": get_gfx_runtime(),
        **_roofline_metrics(us, flops, nbytes),
        "err": err,
    }


@benchmark()
def test_prefill(hq, hkv, d, window, page, m, prefix, dtype):
    """The prefill build serving m query tokens after `prefix` cached ones."""
    v = hip.prefill_variant_for(hq, hkv, d, page, 1, window, dtype, m)
    assert v is not None, (hq, hkv, d, window, page, m, dtype)
    module = hip.load_prefill(v)
    assert module is not None, v
    dev = torch.device("cuda")
    scratch = hip.make_prefill_scratch(
        hip.prefill_variants(hq, hkv, d, page, window, dtype), dev
    )
    seq_len = m + prefix
    q, kv, block_table = _prefill_inputs(v, m, seq_len, seed=m + prefix)
    out = torch.full_like(q, float("nan"))
    seq_lens = torch.tensor([seq_len], device=dev, dtype=torch.int32)

    def launch():
        module.prefill_attn(
            q, kv, block_table, out, *scratch, seq_lens, seq_len, d**-0.5
        )
        return out

    got, us = run_perftest(launch, num_iters=ITERS)
    ref = paged_attention(q, kv, seq_len, window, block_table=block_table)
    err = _assert_prefill_close(got, ref, dtype, f"{v}, M={m}, prefix={prefix}")
    flops, nbytes = _causal_work(hq, hkv, d, m, seq_len, window)
    return {
        "gfx": get_gfx_runtime(),
        "min_query_len": v.min_query_len,
        **_roofline_metrics(us, flops, nbytes),
        "err": err,
    }


def _launch_prefill_batch(v, q, kv, block_table, starts, seq_lens, out, mm_ranges=None):
    """The launch of a batch in one plan, on the build and head group of its
    cell. mm_ranges: each sequence's multimodal ranges."""
    dev = q.device
    shape = (v.num_q_heads, v.num_kv_heads, v.head_size, v.page_size)
    build, head_group = hip.prefill_batch_variant_for(
        *shape, 1, v.window, v.dtype, starts.numpy(), seq_lens.numpy()
    )
    assert build is not None, v
    plan, head_group = hip.plan_prefill(
        starts.numpy(),
        seq_lens.numpy(),
        build,
        v.window,
        hip.prefill_target_workgroups(dev),
        head_group,
    )
    scratch = hip.make_prefill_scratch(
        [*hip.prefill_variants(*shape, v.window, v.dtype), build], dev
    )
    ranges = None
    if mm_ranges is not None:
        width = 1 + max(map(len, mm_ranges))
        ranges = torch.tensor(
            [[*r] + [(0, 0)] * (width - len(r)) for r in mm_ranges],
            device=dev,
            dtype=torch.int32,
        )
    module = hip.load_prefill(build)
    plan, starts_dev, lens_dev = (
        torch.from_numpy(plan).to(dev),
        starts.to(dev),
        seq_lens.to(dev),
    )

    def launch():
        module.prefill_attn(
            q,
            kv,
            block_table,
            out,
            *scratch,
            lens_dev,
            int(seq_lens.max()),
            v.head_size**-0.5,
            ranges,
            plan=plan,
            query_start_loc=starts_dev,
            head_group=head_group,
        )
        return out

    return launch


@benchmark()
def test_prefill_batch(hq, hkv, d, window, dtype, batch):
    """A batch in one planned launch, each sequence against its reference."""
    v = hip.prefill_variants(hq, hkv, d, 16, window, dtype)[0]
    # A prefix has to outgrow the window to be cut by it.
    lens = [(m, s + window if m != s else s) for m, s in BATCHES[batch]]
    q, kv, block_table, starts, seq_lens = _prefill_batch(v, lens)
    out = torch.full_like(q, float("nan"))
    launch = _launch_prefill_batch(v, q, kv, block_table, starts, seq_lens, out)
    got, us = run_perftest(launch, num_iters=ITERS)
    err = 0
    for i, (m, s) in enumerate(lens):
        if not m:
            continue
        rows = slice(int(starts[i]), int(starts[i + 1]))
        pages = block_table[i, : -(-s // v.page_size)]
        ref = paged_attention(q[rows], kv, s, window, block_table=pages)
        err = _assert_prefill_close(
            got[rows], ref, dtype, f"{batch}: sequence {i} (M={m}, S={s})"
        )
    flops, nbytes = map(
        sum, zip(*(_causal_work(hq, hkv, d, m, s, window) for m, s in lens if m))
    )
    return {
        "gfx": get_gfx_runtime(),
        **_roofline_metrics(us, flops, nbytes),
        "err": err,
    }


def _non_causal_work(hq, hkv, d, lens, itemsize=2):
    """(FLOPs, bytes) of sequences whose tokens see every key of their own."""
    flops = sum(4.0 * hq * d * n * n for n in lens)
    nbytes = sum((2 * hq + 2 * hkv) * n * d * itemsize for n in lens)
    return flops, nbytes


@benchmark()
def test_non_causal(hq, hkv, d, m, dtype):
    """The non-causal build serving one sequence of m tokens: each token sees
    every key, read in place."""
    v = hip.prefill_variant_for(hq, hkv, d, 16, 1, 0, dtype, m, causal=False)
    assert v is not None, (hq, hkv, d, m, dtype)
    module = hip.load_prefill(v)
    assert module is not None, v
    dev = torch.device("cuda")
    scratch = hip.make_prefill_scratch(
        hip.prefill_variants(hq, hkv, d, 16, 0, dtype, causal=False), dev
    )
    q, k, val, _ = _non_causal_inputs(v, [m], seed=m)
    out = torch.full(q.shape, float("nan"), dtype=q.dtype, device=dev)
    seq_lens = torch.tensor([m], device=dev, dtype=torch.int32)

    def launch():
        module.encoder_attn(q, k, val, out, *scratch, seq_lens, m, d**-0.5)
        return out

    got, us = run_perftest(launch, num_iters=ITERS)
    err = _assert_prefill_close(got, full_attention(q, k, val), dtype, f"{v}, L={m}")
    flops, nbytes = _non_causal_work(hq, hkv, d, [m])
    return {
        "gfx": get_gfx_runtime(),
        "min_query_len": v.min_query_len,
        **_roofline_metrics(us, flops, nbytes),
        "err": err,
    }


@benchmark()
def test_non_causal_batch(hq, hkv, d, dtype, batch):
    """An encoder's segments in one planned launch, each against its
    reference."""
    lens = NON_CAUSAL_BATCHES[batch]
    dev = torch.device("cuda")
    seq_lens = np.array(lens)
    rows = hip.prefill_variants(hq, hkv, d, 16, 0, dtype, causal=False)
    q, k, val, starts = _non_causal_inputs(rows[0], lens)
    v, head_group = hip.prefill_batch_variant_for(
        hq, hkv, d, 16, 1, 0, dtype, starts, seq_lens, causal=False
    )
    plan, head_group = hip.plan_prefill(
        starts, seq_lens, v, 0, hip.prefill_target_workgroups(dev), head_group
    )
    scratch = hip.make_prefill_scratch(rows, dev)
    out = torch.full(q.shape, float("nan"), dtype=dtype, device=dev)
    module = hip.load_prefill(v)
    lens_dev = torch.tensor(lens, device=dev, dtype=torch.int32)
    plan = torch.from_numpy(plan).to(dev)
    starts_dev = torch.from_numpy(starts.astype(np.int32)).to(dev)

    def launch():
        module.encoder_attn(
            q,
            k,
            val,
            out,
            *scratch,
            lens_dev,
            max(lens),
            d**-0.5,
            plan=plan,
            query_start_loc=starts_dev,
            head_group=head_group,
        )
        return out

    got, us = run_perftest(launch, num_iters=ITERS)
    for i, n in enumerate(lens):
        tokens = slice(int(starts[i]), int(starts[i + 1]))
        err = _assert_prefill_close(
            got[tokens],
            full_attention(q[tokens], k[tokens], val[tokens]),
            dtype,
            f"{batch}: sequence {i} (L={n})",
        )
    flops, nbytes = _non_causal_work(hq, hkv, d, lens)
    return {
        "gfx": get_gfx_runtime(),
        **_roofline_metrics(us, flops, nbytes),
        "err": err,
    }


@benchmark()
def test_encoder(hq, hkv, d, dtype, segments):
    """encoder_forward: every segment, however short, in one non-causal
    launch, into the output it allocates."""
    lens = ENCODER_SEGMENTS[segments]
    n = sum(lens)
    torch.manual_seed(0)
    q, k, v = (
        (torch.randn(n, h, d, device="cuda") * 0.5).to(dtype) for h in (hq, hkv, hkv)
    )
    assert hip.encoder_unsupported(q, hkv) is None
    out, us = run_perftest(
        hip.encoder_forward, q, k, v, lens, d**-0.5, num_iters=ITERS, num_rotate_args=1
    )
    assert out.shape == q.shape and out.dtype == dtype
    err, start = 0, 0
    for i, s in enumerate(lens):
        rows = slice(start, start + s)
        ref = full_attention(q[rows], k[rows], v[rows])
        err = _assert_prefill_close(out[rows], ref, dtype, f"segment {i} (L={s})")
        start += s
    flops, nbytes = _non_causal_work(hq, hkv, d, lens)
    return {
        "gfx": get_gfx_runtime(),
        **_roofline_metrics(us, flops, nbytes),
        "err": err,
    }


# ----------------------------------------------- behaviour outside tables


def check_lookups():
    """A shape's steps of M, and the build each M runs on."""
    for (
        hq,
        hkv,
        d,
        page,
        window,
        dtype,
        causal,
    ), steps in hip._prefill_table().items():
        if not causal or page != 16:
            continue
        for v in steps:
            for m in (v.min_query_len, v.min_query_len + 1):
                assert hip.prefill_variant_for(hq, hkv, d, 16, 1, window, dtype, m) == v
        args = (hq, hkv, d, 16, 1, window, dtype)
        assert hip.prefill_variant_for(*args, hip.PREFILL_MIN_M - 1) is None
        assert hip.prefill_variant_for(*args, hip.PREFILL_MIN_M) == steps[0]
        assert hip.prefill_variant_for(*args, 1 << 20) == steps[-1]
    # A layout or shape without a build has none.
    v = hip.variants()[0]
    shape = (v.num_q_heads, v.num_kv_heads, v.head_size, v.max_query_len, v.page_size)
    assert hip.variant_for(*shape, 0, v.window, v.dtype) is None
    assert hip.variant_for(3, 3, 40, 1, 16, 1, 0, torch.float16) is None


def check_prefill_batch_lookup():
    """A batch takes the cell of the largest steps at or below its features;
    padding is left out of them and keys are capped at the window."""
    table = {
        (32, 8, 128, 16, 0, 1): [(1, 0, 16, 8), (1, 2048, 128, 1), (256, 512, 512, 4)]
    }
    real = hip._prefill_batch_table
    hip._prefill_batch_table = lambda: table
    try:
        cell = functools.partial(hip.prefill_batch_cell, 32, 8, 128, 16, 0)
        assert cell(4, 100) == (16, 8)
        assert cell(4, 5000) == (128, 1)
        assert cell(300, 5000) == (512, 4)
        assert cell(300, 100) == (512, 4)  # no keys step at or below: the smallest
        assert hip.prefill_batch_cell(16, 2, 64, 16, 0, 4, 100) is None
    finally:
        hip._prefill_batch_table = real
    starts = np.array([0, 4, 4, 12])
    seq_lens = np.array([100, 0, 8])
    mean_m, keys = hip.prefill_batch_features(starts, seq_lens, 0)
    assert mean_m == 6
    # Sequence 0: context 96, keys 96 + 2.5; sequence 1: context 0, keys 4.5.
    assert abs(keys - (4 * 98.5 + 8 * 4.5) / 12) < 1e-9
    windowed = hip.prefill_batch_features(starts, seq_lens, 10)[1]
    assert abs(windowed - (4 * 10 + 8 * 4.5) / 12) < 1e-9


def _steps(v, m, s, window):
    """Key steps of each tile of a sequence, as the kernel computes them."""
    g = v.num_q_heads // v.num_kv_heads
    context, out = s - m, []
    for t in range(-(-m * g // v.block_rows)):
        first, last = (
            t * v.block_rows // g,
            min(m - 1, ((t + 1) * v.block_rows - 1) // g),
        )
        begin = 0
        if window:
            begin = max(0, context + first - window + 1) // v.key_tile * v.key_tile
        out.append(-(-(context + last + 1 - begin) // v.key_tile))
    return out


def check_prefill_plan(batch, window):
    """The work list covers every tile once, heaviest first, within the
    scratch's bounds."""
    target = 160
    v = next(v for v in _prefill_rows() if v.causal and v.dtype == torch.float16)
    # The most counters and partials the bounds allow.
    lens = [(1, 32768)] * 19 + [(16, 16)] if batch == "at the bound" else BATCHES[batch]
    starts = np.cumsum([0] + [m for m, _ in lens])
    plan, head_group = hip.plan_prefill(
        starts, np.array([s for _, s in lens]), v, window, target
    )
    seq, tile = plan[:, 0], plan[:, 1] & 0xFFFF
    seg, nseg, slot = plan[:, 2] & 0xFFFF, plan[:, 2] >> 16, plan[:, 3]
    steps = {
        (i, t): n
        for i, (m, s) in enumerate(lens)
        for t, n in enumerate(_steps(v, m, s, window))
    }
    entries = sorted(zip(seq.tolist(), tile.tolist(), seg.tolist(), nseg.tolist()))
    want = sorted(
        (i, t, k, n)
        for (i, t) in steps
        for n in {int(nseg[(seq == i) & (tile == t)][0])}
        for k in range(n)
    )
    assert entries == want
    assert set(steps) == set(zip(seq.tolist(), tile.tolist()))
    split = nseg > 1
    assert all(
        steps[(i, t)] // n >= 16
        for i, t, n in zip(seq[split], tile[split], nseg[split])
    )
    weight = np.array([steps[(i, t)] / n for i, t, n in zip(seq, tile, nseg)])
    assert (np.diff(weight) <= 0).all()
    split_tiles = {(i, t) for i, t in zip(seq[split], tile[split])}
    assert len(split_tiles) * v.num_kv_heads < target
    assert int(nseg[split & (seg == 0)].sum()) * v.num_kv_heads < 2 * target
    assert ((slot >= 0) == split).all()
    assert head_group == (v.num_kv_heads if split_tiles else 0)


def check_decode_window_never_reads_freed_pages(want):
    """Freed pages before the window do not reach the output."""
    v, module = _built(**want)
    seq_len = 2016
    q, kv, block_table = _paged_inputs(seq_len, v)
    ref = paged_attention(q, kv, seq_len, v.window)
    first = max(0, seq_len - v.max_query_len - (v.window - 1)) // v.page_size
    kv = torch.cat([kv, torch.full_like(kv[:1], float("nan"))])
    block_table = block_table.clone()
    block_table[:first] = kv.shape[0] - 1
    out = torch.full_like(q, float("nan"))
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    module.decode_attn(
        q,
        kv,
        block_table,
        out,
        *hip.make_scratch(v, q.device),
        seq_lens,
        v.head_size**-0.5,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    _assert_decode_close(out, ref, v, f"{v.name}, freed pages")


def check_prefill_window_never_reads_freed_pages():
    """Freed pages before the window do not reach the output."""
    v = next(v for v in _prefill_rows() if v.window and v.dtype == torch.float16)
    module = hip.load_prefill(v)
    m, seq_len = v.min_query_len, v.min_query_len + 3 * v.window
    q, kv, block_table = _prefill_inputs(v, m, seq_len)
    ref = paged_attention(q, kv, seq_len, v.window, block_table=block_table)
    first = (seq_len - m - (v.window - 1)) // v.page_size
    kv = torch.cat([kv, torch.full_like(kv[:1], float("nan"))])
    block_table = block_table.clone()
    block_table[:first] = kv.shape[0] - 1
    out = torch.full_like(q, float("nan"))
    args = (v.num_q_heads, v.num_kv_heads, v.head_size, v.page_size, v.window)
    scratch = hip.make_prefill_scratch(hip.prefill_variants(*args, v.dtype), q.device)
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    module.prefill_attn(
        q, kv, block_table, out, *scratch, seq_lens, seq_len, v.head_size**-0.5
    )
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    _assert_prefill_close(out, ref, v.dtype, "freed pages")


def check_prefill_past_the_last_step():
    """M past the last step runs the last row."""
    v = next(
        rows[-1]
        for rows in hip._prefill_table().values()
        if rows[-1].dtype == torch.float16
        and rows[-1].head_size == 128
        and rows[-1].causal
    )
    module = hip.load_prefill(v)
    args = (v.num_q_heads, v.num_kv_heads, v.head_size, v.page_size, v.window)
    scratch = hip.make_prefill_scratch(
        hip.prefill_variants(*args, v.dtype), torch.device("cuda")
    )
    for seed, (m, p) in enumerate([(8193, 0), (12000, 1000)]):
        seq_len = m + p
        q, kv, block_table = _prefill_inputs(v, m, seq_len, seed)
        out = torch.full_like(q, float("nan"))
        seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
        module.prefill_attn(
            q, kv, block_table, out, *scratch, seq_lens, seq_len, v.head_size**-0.5
        )
        torch.cuda.synchronize()
        ref = paged_attention(q, kv, seq_len, v.window, block_table=block_table)
        _assert_prefill_close(out, ref, v.dtype, f"{v}, M={m}, prefix={p}")


def check_graph_replay_follows_seq_lens():
    """A captured graph follows the sequence length on the device."""
    v, module = _built(
        window=lambda x: not x,
        dot_product=lambda x: not x,
        max_segments=lambda x: x > 1,
    )
    long_s = 1024
    q, kv, block_table = _paged_inputs(long_s, v)
    scratch = hip.make_scratch(v, q.device)
    out = torch.empty_like(q)
    seq_lens = torch.tensor([48], device=q.device, dtype=torch.int32)

    def launch():
        module.decode_attn(
            q, kv, block_table, out, *scratch, seq_lens, q.shape[2] ** -0.5
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for s in (48, 208, long_s):
        seq_lens.fill_(s)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _assert_decode_close(out, paged_attention(q, kv, s), v, f"replay, S={s}")


def check_decode_mm_prefix(want, min_m, image, dtype):
    """Decode-sized calls inside an image see the image's later keys, and the
    count comes from device memory."""
    v, module = _built(dtype, max_query_len=lambda m: m >= min_m, **want)
    m, seq_len = v.max_query_len, max(2000, v.window + 1000) + 7
    q, kv, block_table = _paged_inputs(seq_len, v)
    # Keys past the sequence must not be read, even inside an image.
    kv[-1, :, seq_len % v.page_size :] = float("nan")
    ranges = image(seq_len, m, v.window)
    got = _decode_mm(
        module, v, q, kv, block_table, seq_len, _decode_mm_buf(ranges, q.device)
    )
    ref = paged_attention(q, kv, seq_len, v.window, ranges)
    _assert_decode_close(got, ref, v, v.name)
    causal = paged_attention(q, kv, seq_len, v.window)
    assert _max_rel(causal, ref) > 10 * TOL[v.dtype], "the image changed nothing"
    # With several images over the query tokens, each one shows.
    cut = [r for r in ranges if r[0] < r[1] and r[1] >= seq_len - m]
    for r in cut if len(cut) > 1 else []:
        rest = [x for x in ranges if x != r]
        without = paged_attention(q, kv, seq_len, v.window, rest)
        assert _max_rel(without, ref) > 10 * TOL[v.dtype], f"{r} changed nothing"


# Dot-product builds go up to M=3; an image starting among M=2's query tokens
# would only hold the last one.
_MM_BUILDS = {
    "wmma_window segments": (
        {
            "window": bool,
            "dot_product": lambda x: not x,
            "max_segments": lambda x: x > 1,
        },
        4,
    ),
    "wmma segments": (
        {
            "window": lambda x: not x,
            "dot_product": lambda x: not x,
            "max_segments": lambda x: x > 1,
        },
        4,
    ),
    "wmma": (
        {
            "window": lambda x: not x,
            "dot_product": lambda x: not x,
            "max_segments": lambda x: x == 1,
        },
        4,
    ),
    "dot_product": ({"dot_product": bool}, 3),
}
_MM_IMAGES = {
    # Over every query token, beside a past image.
    "covers_all": lambda s, m, w: [(100, 355), (s - m - 40, s - 1)],
    # Ending, or starting, among the query tokens: the rows after it, or
    # before it, stay causal.
    "ends_mid": lambda s, m, w: [(s - m - 40, s - m + m // 2)],
    "starts_mid": lambda s, m, w: [(s - m + m // 2, s - 1)],
    # Past the sequence, as in a chunk ending inside the image.
    "past_seq_len": lambda s, m, w: [(s - m - 40, s + 200)],
    # Longer than the window, which still bounds earlier keys.
    "longer_than_window": lambda s, m, w: [(s - (w or 1000) - 500, s - 1)],
    # Two images over the query tokens: rows end at either one.
    "two_in_span": lambda s, m, w: [(s - m - 40, s - m + 1), (s - m + 2, s - 1)],
}


def check_decode_mm_ranges_stop_at_count_and_capacity(case):
    """The kernel reads min(count, capacity) ranges: those it reads apply, and
    a range past them, over every query token, changes nothing."""
    v, module = _built(
        window=lambda x: not x,
        dot_product=lambda x: not x,
        max_query_len=lambda m: m >= 4,
    )
    m, seq_len = v.max_query_len, 2000
    q, kv, block_table = _paged_inputs(seq_len, v)
    trap = (seq_len - m - 40, seq_len - 1)
    if case == "zero":
        # The trap in the buffer, past count.
        ranges, count, size = [], 0, 3
    else:
        # count 5 in a 2-range buffer, the trap just past its end. The second
        # range ends among the query tokens, so the trap would extend it.
        ranges = [(100, 355), (seq_len - m - 40, seq_len - m + m // 2)]
        count, size = 5, 5
    flat = [count, *(x for r in ranges for x in r), *trap]
    buf = torch.tensor(flat, dtype=torch.int32, device=q.device)[:size]
    got = _decode_mm(module, v, q, kv, block_table, seq_len, buf)
    ref = paged_attention(q, kv, seq_len, 0, ranges)
    _assert_decode_close(got, ref, v, v.name)
    # Reading the trap, or ignoring the ranges that fit, must show.
    for wrong in (ranges + [trap], []):
        if wrong != ranges:
            wrong_ref = paged_attention(q, kv, seq_len, 0, wrong)
            assert _max_rel(wrong_ref, ref) > 10 * TOL[v.dtype], wrong


def check_graph_replay_follows_mm_ranges():
    """A captured graph reads the ranges rewritten in place each step, and
    none are left over when they go back to empty."""
    v, module = _built(
        window=lambda x: not x,
        dot_product=lambda x: not x,
        max_query_len=lambda m: m >= 4,
    )
    seq_len = 1024
    q, kv, block_table = _paged_inputs(seq_len, v)
    scratch = hip.make_scratch(v, q.device)
    out = torch.empty_like(q)
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    buf = _decode_mm_buf([], q.device)

    def launch():
        module.decode_attn(
            q, kv, block_table, out, *scratch, seq_lens, q.shape[2] ** -0.5, buf
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    m = v.max_query_len
    for ranges in ([], [(seq_len - m - 40, seq_len - 1)], []):
        # A smaller count leaves earlier ranges past it.
        flat = _decode_mm_buf(ranges, q.device)[: 1 + 2 * len(ranges)]
        buf[: flat.numel()].copy_(flat)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        ref = paged_attention(q, kv, seq_len, 0, ranges)
        _assert_decode_close(out, ref, v, f"replay, ranges={ranges}")


def check_batched_decode_follows_mm_ranges_per_sequence():
    """A batched build reads each sequence's row of ranges, eagerly and when a
    captured graph replays after the rows and lengths are rewritten. A padding
    sequence (S=0) stays unwritten whatever its row holds, also when the batch
    shrinks between replays."""
    v, module = _built(batch=1, max_query_len=lambda m: m >= 4)
    m, full = v.max_query_len, [1024, 2000, 1500, 1200]
    parts, q, kv, block_table = _batch_inputs(v, full)
    scratch = hip.make_scratch(v, q.device, max_seqs=len(full))
    seq_lens = torch.tensor(full, device=q.device, dtype=torch.int32)
    out = torch.empty_like(q)
    # A spare row, as a server's buffer has one per sequence it may batch.
    buf = torch.zeros(len(full) + 1, 9, dtype=torch.int32, device=q.device)

    def launch():
        module.decode_attn(
            q, kv, block_table, out, *scratch, seq_lens, q.shape[2] ** -0.5, buf
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()

    def image(s):
        # Ending among the query tokens: the rows before its end see later keys.
        return [(s - m - 40, s - m + m // 2)]

    # (lengths, each sequence's ranges); S=0 is CUDA-graph padding, whose row
    # keeps an earlier step's ranges.
    stale = image(full[3])
    steps = [
        ([1024, 2000, 1500, 0], [image(1024), [], image(1500), stale]),
        ([1024, 2000, 0, 0], [[], image(2000), stale, stale]),
        (full, [[], [], [], image(1200)]),
    ]
    for lens, step in steps:
        seq_lens.copy_(torch.tensor(lens, dtype=torch.int32))
        for i, ranges in enumerate(step):
            buf[i] = _decode_mm_buf(ranges, q.device)
        for run in (launch, graph.replay):
            out.fill_(7.0)
            run()
            torch.cuda.synchronize()
            for i, (s, ranges) in enumerate(zip(lens, step)):
                got = out[i * m : (i + 1) * m].float()
                if s == 0:
                    assert (got == 7.0).all(), f"{lens}: padding {i} was written"
                    continue
                ref = paged_attention(parts[i][0], parts[i][1], s, v.window, ranges)
                _assert_decode_close(got, ref, v, f"{step}, sequence {i}")
                if ranges:
                    causal = paged_attention(parts[i][0], parts[i][1], s, v.window)
                    assert _max_rel(causal, ref) > 10 * TOL[v.dtype], "no change"


def check_prefill_mm_prefix(shape, m, prefix, mm_ranges):
    """Tokens of a multimodal range see the whole range."""
    hq, hkv, d, window, dtype = shape
    v = hip.prefill_variant_for(hq, hkv, d, 16, 1, window, dtype, m)
    if v is None:
        aiter.logger.warning("no prefill build for %s; mm prefix skipped", shape)
        return
    module = hip.load_prefill(v)
    scratch = hip.make_prefill_scratch(
        hip.prefill_variants(hq, hkv, d, 16, window, dtype), torch.device("cuda")
    )
    seq_len = m + prefix
    q, kv, block_table = _prefill_inputs(v, m, seq_len)
    out = torch.full_like(q, float("nan"))
    seq_lens = torch.tensor([seq_len], device=q.device, dtype=torch.int32)
    module.prefill_attn(
        q,
        kv,
        block_table,
        out,
        *scratch,
        seq_lens,
        seq_len,
        d**-0.5,
        _mm_ranges_tensor(mm_ranges, q.device)[0],
    )
    torch.cuda.synchronize()
    ref = paged_attention(q, kv, seq_len, window, mm_ranges, block_table)
    _assert_prefill_close(out, ref, dtype, f"{v}, M={m}, ranges={mm_ranges}")


def check_prefill_batch_mm_prefix(shape):
    """A batch with each sequence's own multimodal ranges."""
    hq, hkv, d, window, dtype = shape
    rows = hip.prefill_variants(hq, hkv, d, 16, window, dtype)
    if not rows:
        aiter.logger.warning("no prefill build for %s; batch mm prefix skipped", shape)
        return
    v = rows[0]
    lens = [(600, 600), (1, 2000), (700, 3700), (300, 800)]
    ranges = [[(40, 295), (400, 469)], [], [(3100, 3355)], [(450, 560)]]
    q, kv, block_table, starts, seq_lens = _prefill_batch(v, lens)
    out = torch.full_like(q, float("nan"))
    _launch_prefill_batch(v, q, kv, block_table, starts, seq_lens, out, ranges)()
    torch.cuda.synchronize()
    for i, (m, s) in enumerate(lens):
        rows = slice(int(starts[i]), int(starts[i + 1]))
        pages = block_table[i, : -(-s // v.page_size)]
        ref = paged_attention(q[rows], kv, s, window, ranges[i], pages)
        _assert_prefill_close(out[rows], ref, dtype, f"sequence {i} (M={m}, S={s})")


def check_encoder_segments():
    """No tokens: an empty output and no launch. Segments that do not add up
    to the tokens are refused: they would address past Q, K and V or leave
    rows unwritten. An empty segment among the others changes nothing."""
    rows = [v for v in _prefill_rows() if not v.causal]
    if not rows:
        return
    v = rows[0]
    scale = v.head_size**-0.5
    q, k, val, _ = _non_causal_inputs(v, [])
    out = hip.encoder_forward(q, k, val, [], scale)
    assert out.shape == q.shape and out.dtype == q.dtype
    q, k, val, _ = _non_causal_inputs(v, [128])
    for bad in ([64, 128], [64], [], [192, -64]):
        try:
            hip.encoder_forward(q, k, val, bad, scale)
        except ValueError as e:
            assert "must partition" in str(e), e
        else:
            raise AssertionError(f"segments {bad} of 128 tokens were not refused")
    out = hip.encoder_forward(q, k, val, [64, 0, 64], scale)
    torch.cuda.synchronize()
    for rows_ in (slice(0, 64), slice(64, 128)):
        ref = full_attention(q[rows_], k[rows_], val[rows_])
        _assert_prefill_close(out[rows_], ref, v.dtype, "empty segment between two")


def check_refusals():
    """Calls the kernels cannot serve raise instead of miscomputing."""
    dev = torch.device("cuda")
    v, module = _built()
    q, kv, block_table = _paged_inputs(64, v)
    seq_lens = torch.tensor([64], device=dev, dtype=torch.int32)
    scale = v.head_size**-0.5

    # The other 16-bit dtype: its module holds no kernel of this build.
    for dtype in DTYPES:
        other = next(x for x in DTYPES if x != dtype)
        ov, omodule = _built(dtype=other)
        oq, okv, otable = _paged_inputs(1024, ov)
        oq, okv = oq.to(dtype), okv.to(dtype)
        launch = _decode_launch(
            omodule,
            oq,
            okv,
            otable,
            torch.empty_like(oq),
            hip.make_scratch(ov, dev),
            torch.tensor([1024], device=dev, dtype=torch.int32),
            scale,
        )
        _refused("not built for its head dim, dtype", launch)

    # A build of another head dim, launched in this head dim's module.
    other = next(
        o for o in hip.variants() if o.head_size != v.head_size and not o.batched
    )
    _refused(
        "not built for its head dim",
        lambda: hip.rdna_decode_attn(
            q,
            kv,
            block_table,
            torch.empty_like(q),
            *hip.make_scratch(v, dev),
            seq_lens,
            hip._builds(0).index(other),
            scale,
        ),
    )

    # Only an odd-length int32 [count, start0, end0, ...], or a row of it per
    # sequence, on the device reaches the decode kernel.
    good = _decode_mm_buf([(10, 40)], dev)
    for bad in (good[:4], good.long(), good.cpu(), good.view(1, -1)[:0]):
        _refused(
            "mm_ranges must be",
            lambda bad=bad: module.decode_attn(
                q,
                kv,
                block_table,
                torch.empty_like(q),
                *hip.make_scratch(v, dev),
                seq_lens,
                scale,
                bad,
            ),
        )

    # A block table without pages: the kernel would index before its start.
    _refused(
        "block table must be int32 rows of at least one page",
        lambda: module.decode_attn(
            q,
            kv,
            block_table.view(1, -1)[:, :0],
            torch.empty_like(q),
            *hip.make_scratch(v, dev),
            seq_lens,
            scale,
        ),
    )

    # The kernels read lengths and scratch as packed buffers: a strided view
    # would be read as its neighbours.
    acc, m_, l_, cnt = hip.make_scratch(v, dev)
    out = torch.empty_like(q)
    strided_m = torch.empty(2 * m_.numel(), device=dev)[::2]
    _refused(
        "scratch must be contiguous",
        lambda: module.decode_attn(
            q, kv, block_table, out, acc, strided_m, l_, cnt, seq_lens, scale
        ),
    )
    bv, bmodule = _built(batch=1)
    _, bq, bkv, btable = _batch_inputs(bv, [64, 48])
    bacc, bm, bl, bcnt = hip.make_scratch(bv, dev, max_seqs=2)
    # Two lengths, every other int: read in a row they would be 64 and 0.
    strided_lens = torch.tensor([64, 0, 48, 0], device=dev, dtype=torch.int32)[::2]
    _refused(
        "seq_lens must be a contiguous int32",
        lambda: bmodule.decode_attn(
            bq,
            bkv,
            btable,
            torch.empty_like(bq),
            bacc,
            bm,
            bl,
            bcnt,
            strided_lens,
            bv.head_size**-0.5,
        ),
    )
    # A batch advances each buffer by its own rows per sequence: twice the
    # sequences of half the rows hold the same elements, and would overlap.
    halves = [
        torch.empty(2 * x.shape[0], x.shape[1] // 2, device=dev) for x in (bm, bl)
    ]
    _refused(
        "scratch too small",
        lambda: bmodule.decode_attn(
            bq,
            bkv,
            btable,
            torch.empty_like(bq),
            bacc,
            *halves,
            bcnt,
            torch.tensor([64, 48], device=dev, dtype=torch.int32),
            bv.head_size**-0.5,
        ),
    )
    wide_l = torch.empty(2, 2 * bl.shape[1], device=dev)
    _refused(
        "m and l must hold the same rows",
        lambda: bmodule.decode_attn(
            bq,
            bkv,
            btable,
            torch.empty_like(bq),
            bacc,
            bm,
            wide_l,
            bcnt,
            torch.tensor([64, 48], device=dev, dtype=torch.int32),
            bv.head_size**-0.5,
        ),
    )

    # Only one sequence's (ranges, 2) on the device reaches the prefill
    # kernel: the batch's (sequences, ranges, 2) would unmask other sequences'
    # keys.
    pv = next(x for x in _prefill_rows() if x.causal and x.dtype == torch.float16)
    pmodule = hip.load_prefill(pv)
    scratch = hip.make_prefill_scratch([pv], dev)
    pq, pkv, ptable = _prefill_inputs(pv, 64, 64)
    good = _mm_ranges_tensor([(10, 40)], dev)
    for bad in (good, torch.zeros(2, 3, dtype=torch.int32, device=dev), good[0].cpu()):
        _refused(
            "mm_ranges must be",
            lambda bad=bad: pmodule.prefill_attn(
                pq,
                pkv,
                ptable,
                torch.empty_like(pq),
                *scratch,
                seq_lens,
                64,
                pv.head_size**-0.5,
                bad,
            ),
        )

    # A plan's query starts are read as consecutive ints too.
    lens = [(64, 64), (32, 32)]
    bq2, bkv2, btable2, starts, lens_cpu = _prefill_batch(pv, lens)
    launch = _launch_prefill_batch(
        pv, bq2, bkv2, btable2, starts, lens_cpu, torch.empty_like(bq2)
    )
    launch()
    torch.cuda.synchronize()
    build, head_group = hip.prefill_batch_variant_for(
        pv.num_q_heads,
        pv.num_kv_heads,
        pv.head_size,
        pv.page_size,
        1,
        pv.window,
        pv.dtype,
        starts.numpy(),
        lens_cpu.numpy(),
    )
    plan, head_group = hip.plan_prefill(
        starts.numpy(),
        lens_cpu.numpy(),
        build,
        pv.window,
        hip.prefill_target_workgroups(dev),
        head_group,
    )
    strided_starts = torch.stack([starts, starts], 1).to(dev)[:, 0]
    _refused(
        "query_start_loc must be a contiguous int32",
        lambda: hip.load_prefill(build).prefill_attn(
            bq2,
            bkv2,
            btable2,
            torch.empty_like(bq2),
            *hip.make_prefill_scratch([build], dev),
            lens_cpu.to(dev),
            64,
            pv.head_size**-0.5,
            plan=torch.from_numpy(plan).to(dev),
            query_start_loc=strided_starts,
            head_group=head_group,
        ),
    )

    # A plan may split the tiles of a build that never splits one sequence's
    # KV, so its scratch is checked too.
    unsplit = next(
        (x for x in _prefill_rows() if x.causal and x.max_segments == 1), None
    )
    if unsplit is not None:
        uq, ukv, utable, ustarts, ulens = _prefill_batch(unsplit, [(64, 64), (32, 32)])
        uplan, ugroup = hip.plan_prefill(
            ustarts.numpy(),
            ulens.numpy(),
            unsplit,
            unsplit.window,
            hip.prefill_target_workgroups(dev),
        )
        empty = (
            torch.empty(1, device=dev),
            torch.empty(1, device=dev),
            torch.zeros(1, dtype=torch.int32, device=dev),
        )
        _refused(
            "scratch too small",
            lambda: hip.load_prefill(unsplit).prefill_attn(
                uq,
                ukv,
                utable,
                torch.empty_like(uq),
                *empty,
                ulens.to(dev),
                64,
                unsplit.head_size**-0.5,
                plan=torch.from_numpy(uplan).to(dev),
                query_start_loc=ustarts.to(dev),
                head_group=ugroup,
            ),
        )

    # Non-causal builds run only through encoder_attn, causal ones only
    # through prefill_attn.
    non_causal = [x for x in _prefill_rows() if not x.causal]
    causal_sizes = {x.head_size for x in _prefill_rows() if x.causal}
    nv = next((x for x in non_causal if x.head_size in causal_sizes), None)
    if nv is not None:
        assert all(x.window == 0 for x in non_causal)
        nscratch = hip.make_prefill_scratch([nv], dev)
        nq, nk, nval, _ = _non_causal_inputs(nv, [64])
        zeros = torch.zeros(
            4, nv.num_kv_heads, 16, 2 * nv.head_size, dtype=nv.dtype, device=dev
        )
        table = torch.arange(4, dtype=torch.int32, device=dev)
        out = torch.empty(nq.shape, dtype=nq.dtype, device=dev)
        _refused(
            "rdna_encoder_attn",
            lambda: hip.load_prefill(nv).prefill_attn(
                nq.contiguous(), zeros, table, out, *nscratch, seq_lens, 64, 0.1
            ),
        )
        cv = next(
            x
            for x in _prefill_rows()
            if x.causal and x.head_size == nv.head_size and x.dtype == nv.dtype
        )
        _refused(
            "rdna_prefill_attn",
            lambda: hip.load_prefill(cv).encoder_attn(
                nq, nk, nval, out, *nscratch, seq_lens, 64, 0.1
            ),
        )


def run_checks():
    """The behaviour the tables do not show."""
    check_lookups()
    check_prefill_batch_lookup()
    for batch, window in itertools.product([*BATCHES, "at the bound"], (0, 512)):
        check_prefill_plan(batch, window)
    for want in (
        {
            "window": bool,
            "dot_product": lambda x: not x,
            "max_segments": lambda x: x > 1,
        },
        {
            "window": bool,
            "dot_product": lambda x: not x,
            "max_segments": lambda x: x == 1,
        },
        {"window": bool, "dot_product": bool},
    ):
        check_decode_window_never_reads_freed_pages(want)
    check_prefill_window_never_reads_freed_pages()
    check_prefill_past_the_last_step()
    check_graph_replay_follows_seq_lens()
    for (want, min_m), image, dtype in itertools.product(
        _MM_BUILDS.values(), _MM_IMAGES.values(), DTYPES
    ):
        check_decode_mm_prefix(want, min_m, image, dtype)
    for case in ("zero", "over_capacity"):
        check_decode_mm_ranges_stop_at_count_and_capacity(case)
    check_graph_replay_follows_mm_ranges()
    check_batched_decode_follows_mm_ranges_per_sequence()
    for shape, (m, prefix, ranges) in itertools.product(
        [
            (32, 16, 256, 1024, torch.float16),
            (32, 4, 512, 0, torch.float16),
            (32, 16, 256, 1024, torch.bfloat16),
        ],
        [
            (600, 0, [(40, 295), (400, 469)]),
            (700, 3000, [(3100, 3355)]),
            (300, 500, [(450, 560)]),
            # Longer than the 1024-key window: the window bounds earlier keys.
            (2048, 512, [(600, 2099)]),
        ],
    ):
        check_prefill_mm_prefix(shape, m, prefix, ranges)
    for shape in (
        (32, 16, 256, 1024, torch.float16),
        (32, 4, 512, 0, torch.bfloat16),
    ):
        check_prefill_batch_mm_prefix(shape)
    check_encoder_segments()
    check_refusals()
    aiter.logger.info("rdna_attn behaviour checks passed")


# ------------------------------------------------------------------- main


def summarize(name, rows):
    if rows:
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "%s summary (markdown):\n%s", name, df.to_markdown(index=False)
        )


def main():
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning("rdna_attn unsupported on %s; skipping", get_gfx_runtime())
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.d_dtypes["fp16"], dtypes.d_dtypes["bf16"]],
        help="""Data type.
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "-s",
        "--shape",
        type=dtypes.str2tuple,
        nargs="*",
        default=None,
        help="""Hq,Hkv,D,window (default: every shape of the GPU's tables).
    e.g.: -s 32,8,128,0""",
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        nargs="*",
        default=list(range(1, hip.MAX_M + 1)),
        help="""Query tokens per sequence of a decode.
    e.g.: -m 1 4""",
    )
    args = parser.parse_args()

    def shapes_of(variants):
        """(Hq, Hkv, D, window, page size) of some builds, those of --shape."""
        shapes = sorted(
            {
                (v.num_q_heads, v.num_kv_heads, v.head_size, v.window, v.page_size)
                for v in variants
            }
        )
        if args.shape is None:
            return shapes
        return [s for s in shapes if s[:4] in set(args.shape)]

    decode_shapes = shapes_of(hip.variants())
    causal_shapes = shapes_of(v for v in _prefill_rows() if v.causal)
    non_causal_shapes = shapes_of(v for v in _prefill_rows() if not v.causal)
    # Batches run on the page size a server's cache has by default.
    batch_shapes = [s[:4] for s in causal_shapes if s[4] == 16]

    for dtype in args.dtype:
        name = _dtype_name(dtype)
        summarize(
            f"rdna decode {name}",
            [
                test_decode(hq, hkv, d, window, page, m, dtype, batched)
                for (hq, hkv, d, window, page), m, batched in itertools.product(
                    decode_shapes, args.m, (0, 1)
                )
            ],
        )
        # Each step of M of a shape, at its first M and past it after a prefix;
        # the first step also at the fewest tokens the kernel serves.
        rows = []
        for hq, hkv, d, window, page in causal_shapes:
            steps = hip.prefill_variants(hq, hkv, d, page, window, dtype)
            prefix = max(4096, 2 * window)
            cells = [(hip.PREFILL_MIN_M, prefix), (hip.PREFILL_MIN_M + 50, 0)]
            for v in steps:
                cells += [(v.min_query_len, 0), (v.min_query_len + 37, prefix)]
            rows += [
                test_prefill(hq, hkv, d, window, page, m, p, dtype) for m, p in cells
            ]
        summarize(f"rdna prefill {name}", rows)
        summarize(
            f"rdna prefill batch {name}",
            [
                test_prefill_batch(hq, hkv, d, window, dtype, batch)
                for (hq, hkv, d, window), batch in itertools.product(
                    batch_shapes, BATCHES
                )
            ],
        )
        # Each step, also past its last whole key tile, including a last tile
        # whose later 16-key sub-tiles hold no key.
        rows = []
        for hq, hkv, d, _, _ in non_causal_shapes:
            steps = hip.prefill_variants(hq, hkv, d, 16, 0, dtype, causal=False)
            lens = [hip.PREFILL_MIN_M, 100]
            for v in steps:
                whole = -(-v.min_query_len // v.key_tile) * v.key_tile
                lens += [v.min_query_len, v.min_query_len + 37, whole + 8, whole + 16]
            rows += [test_non_causal(hq, hkv, d, m, dtype) for m in dict.fromkeys(lens)]
        summarize(f"rdna non-causal {name}", rows)
        summarize(
            f"rdna non-causal batch {name}",
            [
                test_non_causal_batch(hq, hkv, d, dtype, batch)
                for (hq, hkv, d, _, _), batch in itertools.product(
                    non_causal_shapes, NON_CAUSAL_BATCHES
                )
            ],
        )
        summarize(
            f"rdna encoder {name}",
            [
                test_encoder(hq, hkv, d, dtype, segments)
                for (hq, hkv, d, _, _), segments in itertools.product(
                    non_causal_shapes, ENCODER_SEGMENTS
                )
            ],
        )

    run_checks()


if __name__ == "__main__":
    main()
